# 行列演算ユニット活用の証跡（REQ-11・TASK-11.3・#70）

## 目的・出典

REQ-11（行列演算ユニットの活用・Should。2026 v2 書き直し）は、v1 の
「CubeCL autotune ＋ `CUBECL_DEBUG_LOG` によるカーネルソースダンプ」前提を
廃し、**リポジトリ内の自作カーネルソースにおける該当命令の実在**と
**ベンチマーク実測ログ**を証跡とする方式に転換した
（`docs/spec/04-requirements.md` REQ-11、`docs/spec/05-tasks.md` TASK-11.3）。

TASK-11.1（#59 系。CUDA WMMA/mma カーネル実装）・TASK-11.2（#66 系。
ディスパッチ規則実装）は完了済みであり、本ドキュメントは REQ-11 系列の
最終タスクとして、分散している証跡材料を集約する。個々の実測値・命令
リストの実体はここでは複製せず、各ソース・各 `docs/perf/*` ドキュメントへの
参照リンクで結合する（`.claude/rules/code-comment-style.md`「陳腐化しやすい
実装詳細の重複を書かない」）。

## 証跡取得方針

REQ-11 の充足は次の 2 本立てで証跡化する。

1. **命令の実在**（静的証跡）: 行列演算ユニット専用命令（CUDA `wmma::*`／
   PTX `mma.sync`・`ldmatrix`・`cp.async`、Metal `simdgroup_float8x8` 系）が
   リポジトリ内の手書きカーネルソース文字列中に実在することを、
   `#[cfg(test)]` の contains 検査で機械検証する。GPU 実機を必要とせず
   Linux CI（self-hosted）上で完結する。
2. **性能発現**（実測証跡）: `docs/perf/` 配下の各ドキュメントに、実機
   （DGX Spark GB10・Apple Silicon 等）実行後の TFLOPS・数値一致判定結果を
   記録する。実測値の記録責務は各 perf ドキュメントが担い、本ドキュメントは
   参照とディスパッチ証跡を担う（二重管理禁止）。

### 外部ダンプ手段は不要

v1（Burn/CubeCL 統合）はカーネルを実行時に動的生成するフレームワークで
あったため、生成コードの実在を確認するには `CUBECL_DEBUG_LOG` 等の外部
ダンプ手段が必須だった。v2 は REQ-1 に基づく完全自作コアであり、行列演算
ユニット命令を含むカーネルソースは**手書きの文字列リテラルとして
リポジトリに実在**する（`crates/backend-cuda/src/kernels*.rs` の Rust
文字列定数、`crates/backend-metal/src/shaders/gemm.metal` のファイル本体）。
したがって命令の実在確認に外部ダンプ手段・実行時ロギングは不要であり、
`grep`／`include_str!` ベースの静的検査のみで完結する。これが REQ-11 が
v1 前提を廃した理由そのものである。

## 命令実在一覧表

行番号はリファクタリングで陳腐化しやすいため記載せず、テスト名・カーネル
名をアンカーとする（再現手順は次節）。

| バックエンド | ソースファイル | カーネル | 行列演算ユニット命令 | 実在検査テスト |
|---|---|---|---|---|
| CUDA | [`kernels.rs`](../crates/backend-cuda/src/kernels.rs) | `gemm_wmma_tf32` | `wmma::fragment`・`wmma::load_matrix_sync`・`wmma::mma_sync`・`wmma::store_matrix_sync`・`wmma::__float_to_tf32` | `wmma_tf32_constants_match_kernel_source_defines`（定数整合。命令実在は同ファイル内カーネルソース文字列 `WMMA_TF32_F32` に静的に含まれる） |
| CUDA | [`kernels_wmma.rs`](../crates/backend-cuda/src/kernels_wmma.rs) | `gemm_wmma_f16` | `wmma::fragment`・`wmma::load_matrix_sync`・`wmma::mma_sync`・`wmma::store_matrix_sync`・`wmma::fill_fragment` | `wmma_f16_source_uses_wmma_instructions` |
| CUDA | [`kernels_wmma_opt.rs`](../crates/backend-cuda/src/kernels_wmma_opt.rs) | TF32/f16 最適化版（ダブルバッファリング） | 上記 wmma 系一式（TF32 版は `wmma::__float_to_tf32` を含む） | `wmma_tf32_opt_source_uses_wmma_instructions`・`wmma_f16_opt_source_uses_wmma_instructions` |
| CUDA | [`kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) | 低レベル PTX 経路（`gemm_mma_f16`） | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`・`ldmatrix.sync.aligned.m8n8.x4.shared.b16`・`ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`・`cp.async.cg.shared.global`・`cp.async.commit_group`・`cp.async.wait_group` | `mma_f16_source_uses_mma_sync_ldmatrix_cp_async_instructions` |
| Metal | [`gemm.metal`](../crates/backend-metal/src/shaders/gemm.metal) | `gemm_simdgroup`・`gemm_simdgroup_tiled` | `simdgroup_float8x8`・`simdgroup_load`・`simdgroup_multiply_accumulate`・`simdgroup_store` | [`shader_source_evidence.rs::gemm_metal_source_uses_simdgroup_matrix_instructions`](../crates/backend-metal/tests/shader_source_evidence.rs)（本 #70 で追加） |

各ファイルは REQ-8（境界検査の維持方針）に基づく手動境界チェックの実在
検査（`wmma_f16_source_retains_req8_boundary_guards` 等）も併設している。
境界検査自体の詳細は各ファイル冒頭コメントを参照。

## 実在検査の再現手順

```sh
# CUDA 側 4 ファイルの命令実在検査（Linux・GPU なしで実行可。文字列検査のみ）
cargo test -p backend-cuda

# Metal 側の命令実在検査（Linux・GPU なしで実行可。文字列検査のみ）
cargo test -p backend-metal --test shader_source_evidence

# 命令実在を直接 grep で確認する場合の例（テストと同じ検査対象文字列）
grep -n "wmma::mma_sync" crates/backend-cuda/src/kernels_wmma.rs
grep -n "simdgroup_multiply_accumulate" crates/backend-metal/src/shaders/gemm.metal
```

## ディスパッチ規則との対応

`select_gemm_kernel`（[`crates/tensor-core/src/dispatch.rs`](../crates/tensor-core/src/dispatch.rs)。
TASK-11.2・#68）が `DeviceCaps`／`GemmShape`／`DType` から `KernelKind`
（`Naive`／`Tiled`／`Wmma` 等）を決定し、各バックエンドの `gemm.rs`／
`gemm_wmma.rs`／`gemm_mma.rs` が対応するカーネル文字列
（本ドキュメントの命令実在一覧表の対象）を選択・起動する。設計の詳細
（HW 判定・形状判定・dtype ゲート・決定表）は
[`docs/dispatch-rules-design.md`](./dispatch-rules-design.md) を参照し、
本ドキュメントでは重複記述しない。

**フォールバック 3 条件**（[`cuda-tensor-core-design.md`](./cuda-tensor-core-design.md) 7 節・
[`cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md) 5 節から要約）:

1. toolkit 非搭載・NVRTC が `<mma.h>` を解決できない環境
2. compute capability が WMMA（cc 7.0+）／TF32（cc 8.0+）／mma パイプライン
   （`MIN_COMPUTE_CAPABILITY_MAJOR = 8`）の要件を満たさない環境
3. M/N/K がタイル最小単位に満たない極小形状

いずれの条件でも tiled 経路（`KernelKind::Tiled`）へフォールバックする。

## ベンチ実測ログ

実測値の記録は下記各ドキュメントが担う（本ドキュメントでの複製はしない）。
いずれも「計測手順＋テンプレート＋実機実行後転記」の運用であり、本
実装時点（Linux・libnvrtc 非搭載・Metal 実機なし環境）では新規実測は
できない。

| ドキュメント | 対応 | 実測状態 |
|---|---|---|
| [`docs/perf/cuda-tensor-core-measurement.md`](./perf/cuda-tensor-core-measurement.md) | #64（TASK-11.1e）。WMMA TF32／f16 の TFLOPS・複合判定記録 | 実測未実施（記入待ちテンプレート） |
| [`docs/perf/cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md) | #187（TASK-11.1h）。mma パイプライン実測 | 実測未実施（記入待ちテンプレート） |
| [`docs/perf/cuda-tensor-core-tolerance-evaluation.md`](./perf/cuda-tensor-core-tolerance-evaluation.md) | #186（TASK-11.1g）。数値一致閾値の再評価 | RTX 3060（sm_86）実測済み・**閾値超過が判明**（次節参照） |
| [`docs/perf/metal-gemm-dynamic-tile.md`](./perf/metal-gemm-dynamic-tile.md) | #188。Metal simdgroup／動的タイル実測 | 実測未実施（記入待ちテンプレート） |
| [`docs/perf/dispatch-boundary-measurement.md`](./perf/dispatch-boundary-measurement.md) | #69。境界形状 256〜1024 の経路比較 | 実測未実施（記入待ちテンプレート） |

実機実行後の転記フロー: 実機（DGX Spark GB10・Apple Silicon 等）確保後に
`make test-ignored-cuda`／`make test-ignored-metal`（`#[ignore]` 分離済みの
実機依存テスト・ベンチを実行）で計測し、5 回計測の中央値を各 `docs/perf/*`
ドキュメントのテンプレート該当欄に転記する（`.claude/rules/coding-rust.md`
「テスト・ベンチ」節）。

tiled f32 経路のみ PoC-v2-3 で実機実測済み（1.832 TFLOPS、M=N=K=4096。
[`cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md) 4 節）。

## 前提条件: #186 数値一致閾値の未解決事項

[`cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md) 5 節の
指示に従い、本ドキュメントも次を前提条件として明記する。

- #186（TASK-11.1g）で RTX 3060（compute capability 8.6）実機の誤差分布を
  実測した結果、**TF32 経路は全形状で現行閾値（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）を著しく超過し、f16 経路も大きな K で閾値を超過する
  ことが判明した**
  （[`cuda-tensor-core-tolerance-evaluation.md`](./perf/cuda-tensor-core-tolerance-evaluation.md) 4 節「結論」）。
- 閾値定数自体は**一切変更していない**（ユーザー承認なしの緩和は行わない
  方針、`.claude/rules/security.md` A08）。改定候補は同ドキュメントに記載
  されているが、いずれも REQ-2 改定（正本 spec リポジトリ側での対応）が
  必要でありスコープ外。
- 本事実は sm_121（GB10）実機ではなく sm_86（RTX 3060）実測に基づくもので
  あり、Tensor Core の世代差による差異が出る可能性があるため、sm_121 に
  そのまま適用しない。

本ドキュメント・関連テストはこの未解決事項を変更しない（ガードレール
閾値・テスト許容誤差の変更はユーザー承認必須。`.claude/rules/delegation-impl.md`）。
