# CUDA GEMM WMMA(TF32) 経路への Phase B 技法横展開 記録（#500）

イシュー #500「perf(backend-cuda): kernels_wmma_opt.rs（TF32 経路）へ Phase B 技法を横展開」の
選別根拠・SMEM 予算試算・A/B 計測手順・記録テンプレート。GEMM 性能改善ツリー #479 → Phase B
（#492〜#499・`crates/backend-cuda/src/kernels_mma.rs` の f16 `mma.sync` 経路）で確立した技法のうち
WMMA C++ API（`nvcuda::wmma`）の TF32 経路（`crates/backend-cuda/src/kernels_wmma_opt.rs`）へ適用可能な
ものを横展開した記録。

## 1. 背景

REQ-8 の CUDA f32 行（対 PyTorch 比 25.64%・`wmma_tf32` opt 経路。`docs/perf/cuda-floor-remeasurement.md`）
を改善するため、Phase B 確立技法のうち適用可能なものを `WMMA_TF32_F32_STAGED_BODY`（新規追加。エント
リポイント `gemm_wmma_tf32_staged`）へ実装した。既存 `WMMA_TF32_F32_OPT_BODY`（`__syncthreads()` ベースの
2 面ダブルバッファ・cp.async 不使用）は削除・改変せず、cp.async 16 バイト整列非対応形状
（`n % 4 != 0 || k % 4 != 0`）のフォールバック先として温存する。

## 2. 技法の選別

| 技法 | 判断 | 理由 |
|---|---|---|
| B-1/B-5: cp.async 多段化（Stages=3）＋ issue 分散 | **適用** | グローバル→共有メモリのコピーは WMMA/mma に依存しない。f32 は 16B = 4 要素粒度（f16 は 8 要素）。`wait_group (STAGES-2)` の正しさ論証（`kernels_mma.rs::MMA_STAGES` 直下コメント）をそのまま踏襲 |
| B-4: warp 内先読み（タイル内限定） | **適用** | `load_matrix_sync` を K サブステップ（K_TILE=16 / FRAG_K=8 → 2 サブステップ）で fragment 2 面バッファ化し、`mma_sync` 発行前に次サブステップを先読み。#495 と同じ「タイル内先読み限定」（クロスタイル不採用理由も踏襲） |
| B-2: レジスタブロッキング | 適用済み | 現行カーネルが既に warp あたり 2×2 fragment |
| B-7: バンクコンフリクト対策パディング | 適用済み | A_PAD/B_PAD の +4 パディングが既に存在。cp.async 化に伴い「PAD が 4 要素（16B）の倍数」の const アサーションを追加 |
| B-3: タイル拡大（BM/BN/BK） | **除外** | WMMA の `store_matrix_sync` は fragment を無条件書き込みするためエピローグ用共有メモリ `c_tile`（BM×BN×4B）が必須。BN=128 では 3 ステージ SMEM＋c_tile が静的 48KiB 上限を超過する（3 節試算）。64×64 のまま Stages=3 なら収まる。SMEM エピローグ再利用（union）による拡大は実測環境が無い現状ではリスク過大として見送り、除外理由を記録する |
| B-6: 蛇行（serpentine） | **除外** | 純粋な発行順並べ替えで実測でのみ価値が確認できる技法。#497 は PR #657 で「未計測のまま本番導入しない」契約により撤回済み。同じ契約に従い実機セッションでの A/B 後に再提案（`docs/perf/cuda-gemm-serpentine-ab.md` と同型の扱い） |
| B-8: L2 タイル→SM スウィズル | **除外** | mma 側でも既定経路への導入は実測昇格待ちの opt-in 変種（#499・`docs/perf/cuda-gemm-swizzle-ab.md`）。既定 TF32 経路への未実測導入は同プロセスと矛盾するため、mma 側の昇格判断後に横展開する |

## 3. SMEM 予算試算

既定構成（`WmmaTf32StagedKernelConfig::default_tf32_staged()`。block_m=block_n=64、k_tile=16、stages=3）:

- `as_tile`: `stages * block_m * a_pad * 4B` = `3 * 64 * 20 * 4` = 15,360B（`a_pad = k_tile + 4 = 20`）
- `bs_tile`: `stages * k_tile * b_pad * 4B` = `3 * 16 * 68 * 4` = 13,056B（`b_pad = block_n + 4 = 68`）
- `c_tile`（エピローグ。既存 opt と同一の無パディング 64×64）: `block_m * block_n * 4B` = `64 * 64 * 4` =
  16,384B
- 合計: `15,360 + 13,056 + 16,384` = **44,800B** ≤ `MMA_STATIC_SMEM_LIMIT_BYTES`（49,152B。48KiB）

`crates/backend-cuda/src/kernels_wmma_opt.rs::validate_wmma_tf32_staged_config` がこの式を `checked_*`
演算で実行時にも fail-closed 検証する（`kernels_wmma_opt::tests::
validate_wmma_tf32_staged_config_accepts_default_and_rejects_smem_overflow` が既定構成の受理と
`stages=8`（130,048B。上限超過）の拒否を単体テストで固定する）。

**`stages` の上限は SMEM 予算だけでは決まらない**（codex-review 指摘）。ループ内 `cp.async.wait_group
(STAGES-2)` の即値は PTX 上 0〜7 の範囲に制限されるため、`stages <= 9` が別枠の必須条件になる。
block_m/block_n/k_tile を warp タイル辺（32）・FRAG_K（8）まで小さくした構成では、SMEM 予算内のまま
`stages` を 10 以上へ増やせてしまい、SMEM 検査のみでは `render_wmma_tf32_staged` が成功したうえで
NVRTC コンパイルの段階まで無効な PTX であることが判明しない。`validate_wmma_tf32_staged_config` は
`stages - 2 > 7` を独立に fail-closed 拒否し、`kernels_wmma_opt::tests::
validate_wmma_tf32_staged_config_rejects_stages_exceeding_wait_group_bound`／
`..._accepts_stages_at_wait_group_bound` が最小タイル構成（block_m=block_n=32, k_tile=8）で
`stages=10` の拒否・`stages=9` の受理（境界値）を単体テストで固定する。

BN=128（B-3 適用時の試算）: `as_tile` 30,720B（`block_m=128, a_pad=20`）+ `bs_tile` 13,056B（k_tile/b_pad
は N のみ拡大なら b_pad=132 で `3*16*132*4`=25,344B）+ `c_tile` 65,536B（128×128×4B）で合計 121,600B 超
（実際には BM/BN 双方拡大でさらに増える）。48KiB を大きく超過するため B-3 は除外（2 節）。

## 4. 実装

- `crates/backend-cuda/src/kernels_wmma_opt.rs`: `WmmaTf32StagedKernelConfig`（`WmmaOptKernelConfig` とは
  独立した struct。既存 TF32 opt・f16 opt の共有 config へ `stages` を追加すると全既存リテラル構築が
  影響を受けるため独立させた）・`render_wmma_tf32_staged`・`WMMA_TF32_F32_STAGED_BODY`（エントリポイント
  `gemm_wmma_tf32_staged`）
- `crates/backend-cuda/src/gemm.rs`: `CudaGemm::wmma_tf32_staged` フィールド（`Option<CudaFunction>`。
  コンパイル失敗は opt/basic の可用性を道連れにしない）・`run_wmma_tf32` の 3 段選択（staged → opt →
  basic。整列条件 `wmma_tf32_staged_alignment_ok(n, k)` を満たさない形状は opt へフォールバック）・
  `launch_wmma_tf32`（デバイス常駐版）も同じ 3 段選択に更新
- テスト: `kernels_wmma_opt.rs` 内 `#[cfg(test)]`（ソース証跡・定数突合・fail-closed 検証。GPU 不要）・
  `tests/gemm_wmma_tf32_staged.rs`（`#[ignore]`。整列形状の形状網羅・K=4096 ストレス・k=0/m=0 no-op・
  性能比較）

## 5. 非後退契約

- **ルーティング変更（PR #678 codex-review P1 指摘対応）**: `tests/common/parity_baseline.rs::
  ParityPath::WmmaTf32Opt` の記録行（64×64×64・512×512×512・512×512×4096。いずれも 4 の倍数）は、
  staged カーネル実装済み環境の実機では `run_wmma_tf32`（公開 API）が staged 経路を自動選択するため、
  公開 API 経由の非後退検査は opt カーネル自体の回帰を検出できなくなる欠陥があった。修正として、この
  経路の非後退検査は `backend_cuda::gemm::tests::wmma_tf32_opt_kernel_parity_does_not_regress`
  （`src/gemm.rs`。private field 経由で 3 段選択を経由せず opt カーネルを強制実行）へ移設した。新設の
  `ParityPath::WmmaTf32Staged`（`tests/parity_nonregression.rs::check_wmma_tf32_staged_baseline`）が
  公開 API 経由で staged 経路を検査する（staged はこの経由で正しく強制実行できる）。staged 行は実機
  未到達のため記録値未確定（`baseline_provenance_unconfirmed: true`）で fail-closed に倒れる。実機
  再計測時（6 節）に両経路の fail_count・mean_abs_diff を記録し確定させる。
- `RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`（ガードレール閾値）は無変更（
  `tolerance_constants_are_pinned` が bit 等値で機械検査する）。

## 6. 実機計測手順（DGX Spark GB10・sm_121）

base（`kernels_wmma_opt::wmma_tf32_f32_opt_source()`。既存 opt 経路）と head（`wmma_tf32_f32_staged_source()`。
本イシューの staged 経路）を比較する。接続・転送手順は `docs/real-hardware-verification-env.md` に従う
（実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin
gh pr checkout <PR番号>   # 本イシューの実装 PR

# ソース証跡・定数突合・fail-closed 検証（GPU 不要。通常 CI 相当）。
cargo test -p backend-cuda --lib kernels_wmma_opt

# 実機必須テスト（数値一致確認を TFLOPS 比較より前に必須で行う）。
cargo test -p backend-cuda --test gemm_wmma_tf32_staged -- --ignored --test-threads=1
cargo test -p backend-cuda --test gemm_wmma_tf32_opt -- --ignored --test-threads=1
cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1
# opt カーネル単独の非後退・staged 対 opt の TFLOPS 比較（private field 経由。gemm.rs 内）。
cargo test -p backend-cuda --lib -- --ignored --test-threads=1 wmma_tf32_opt_kernel_parity_does_not_regress wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096

# cuda_floor_bench（対 PyTorch 比。既存 f32 行の再測定）。
cargo run -p backend-cuda --example cuda_floor_bench --release -- --shapes 4096x4096x4096
```

## 7. 実測結果（実機セッションで記入するプレースホルダ。推定値の記載禁止）

| 項目 | base（opt。既存） | head（staged。#500） | 備考 |
|---|---|---|---|
| M=N=K=4096 TFLOPS（5 回計測中央値。staged 対 opt） | 未計測 | 未計測 | `backend_cuda::gemm::tests::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`（`src/gemm.rs`） |
| M=N=K=4096 TFLOPS（5 回計測中央値。staged 対 tiled f32） | 未計測 | 未計測 | `tests/gemm_wmma_tf32_staged.rs::wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096` |
| 対 PyTorch 比（REQ-8 f32 行） | 25.64%（`cuda-floor-remeasurement.md` 実測値） | 未計測 | 目標 35% |
| parity fail_count/total（wmma_tf32_opt 行） | 既存 baseline（`parity_baseline.rs`） | 未計測 | 非後退確認 |
| parity mean_abs_diff | 既存 baseline | 未計測 | 非後退確認 |

「4096 の対 PyTorch 比 25.64% 超（目標 35%）」の確定は実機計測でのみ判定可能であり、本 PR の作成セッション
では実機（`spark-dbd9`）へ到達できなかったため未計測のまま達成を主張しない。実機再計測はイシュー #502
（Phase B 完了時点の f32/f16 再計測）へ引き継ぐ。

## 8. スコープ外（追跡）

- 蛇行・L2 スウィズル・タイル拡大の TF32 経路適用: 2 節で選別除外。実機 A/B（蛇行は #497 の再提案手順、
  スウィズルは #499 の昇格判断）後の後続対応
- 実機実測（本記録の 7 節）: イシュー #502 へ引き継ぐ
- f16 opt 経路（`render_wmma_f16_opt`）への同技法適用: 本イシューは TF32 経路限定。f16 は mma 経路
  （Phase B 適用済み）が主力のため対象外
