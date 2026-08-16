# CUDA GEMM `mma.sync`/`ldmatrix`/`cp.async` パイプライン 計測記録（#187・TASK-11.1h）

イシュー #187「perf(backend-cuda): TASK-11.1h mma.sync/ldmatrix・cp.async パイプラインの実装」の実測記録テンプレート。
受け入れ条件「tiled 実装比の性能向上と対 PyTorch 比の実測記録（5 回中央値）」「数値一致複合判定の通過」に対応する。

## 状態: 実測未実施・NVRTC 未検証（実装環境に CUDA driver はあるが NVRTC がない）

本実装セッションの実行環境には CUDA **driver**（`libcuda`）が実在し、`nvidia-smi` で以下の実機が確認できた:

```
NVIDIA GeForce RTX 3060（compute capability 8.6・Driver Version 595.71.05・CUDA Version 13.2 表記）
```

これは `mma.sync`/`ldmatrix`/`cp.async` 経路が要求する compute capability 8.0 以上（`gemm_mma.rs::MIN_COMPUTE_CAPABILITY_MAJOR`）
を満たすため、`CudaDevice::new(0)` は成功し `CudaMmaGemm::new` の compute capability ゲートは通過する。しかし **NVRTC
（`libnvrtc`）はこの環境に存在せず**、`compile_ptx` は `CudaError::NvrtcUnavailable` を返す（`nvcc`／CUDA toolkit 自体が
未導入。`which nvcc` は not found、`ldconfig -p` に `libnvrtc` の記載なしを確認済み）。

したがって、本ファイルの CUDA C++／インライン PTX ソース（`crates/backend-cuda/src/kernels_mma.rs::MMA_F16`）は
**この実装セッション中に一度も NVRTC の構文検証を通過していない**。`docs/perf/metal-gemm-dynamic-tile.md` の先例
（「実機での最初の実行が構文検証を兼ねる」）と同じ位置づけであり、**RTX 3060（sm_86）を含むいかなる実機でも未検証**
である点に注意（本 GPU の存在は「driver の到達性」を確認できただけで、「カーネルソースの構文的妥当性」の証拠にはならない）。
sm_121（DGX Spark GB10。設計上のターゲット）での挙動はさらに別途の検証が必要（実装計画 8 節「リスク」）。

本実装セッションで検証済みの事項:

- `cargo build -p backend-cuda`／`cargo test -p backend-cuda`／`cargo clippy --workspace --all-targets --all-features -- -D warnings`
  が全て green（Rust 側の型・API 契約・カーネル定数の内部整合性はコンパイル時 `const _: () = assert!(...)`（`kernels_mma.rs`）で検査済み）
- `cargo run -p backend-cuda --example gemm_mma_bench` を実際に実行し、CUDA driver 到達→NVRTC 不在検出→graceful skip の
  分岐が意図どおり動作することを確認済み（出力はいずれの経路も `NvrtcUnavailable` で skip）
- `kernels_mma.rs` 内の `#[cfg(test)]`（tensor core 命令実在検査・タイル定数整合検査・REQ-8 境界チェック実在検査）が
  全て green

未検証の事項（#64／#65 へ引き継ぐ。集約先: [`docs/cuda-tensor-core-knowledge.md`](../cuda-tensor-core-knowledge.md) 2.4 節・4 節）:

- NVRTC がインライン PTX（`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`・`ldmatrix.sync.aligned.m8n8.x4/x2.trans.shared.b16`・
  `cp.async.cg.shared.global`／`.commit_group`／`.wait_group`）を受理するかどうか
- `ldmatrix.x4`/`x2.trans` のレーン→共有メモリアドレス対応（`kernels_mma.rs` 内 `a_quad_row`/`a_quad_col`/`b_quad` 計算）が
  `mma.sync` フラグメントのレーン→レジスタ対応と正しく整合しているか（実行時の数値照合でのみ確認可能）
- `cp.async` の 16 バイト整列制約下（`n`/`k` が 8 の倍数）での実際のスループット・正しさ
- sm_121 固有の挙動

## 計測手順（compute capability 8.0 以上・NVRTC 搭載の実機）

```sh
git fetch origin
git checkout perf/187-mma-sync-cp-async-pipeline   # 本イシューの実装ブランチ
cargo run -p backend-cuda --example gemm_mma_bench --release
```

出力形式（`examples/gemm_mma_bench.rs` 参照）:

- `size=<N>` 行: 正方形状（512/1024/2048/4096）で tiled f32／WMMA f16／`mma.sync` f16 の TFLOPS と
  `mma_over_tiled`・`mma_over_wmma` 比（初期化に失敗した経路は `n/a` として該当列のみ欠落する）
- `size=4096 mma_over_pytorch_f16=...` 行: PoC-v2-3 実測の PyTorch f16 実効値（97.6 TFLOPS）に対する比

数値一致確認（受け入れ条件に必須の前提。性能値採用より先に実施すること）:

```sh
cargo test -p backend-cuda -- --ignored --nocapture
```

`tests/cpu_cuda_mma_parity.rs` の全ケース（タイル倍数形状・8 の倍数の非タイル倍数エッジ形状・K=4096 ストレス・
WMMA 経路との相互比較）と `tests/gemm_mma.rs` の `#[ignore]` ケース（ゼロ次元形状・整列制約拒否）が PASS することを
先に確認する。

## 実測結果（記入待ち）

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU | （記入: 例 NVIDIA H100・DGX Spark GB10 等。compute capability を明記） |
| driver / NVRTC バージョン | （記入: `nvidia-smi`・`nvcc --version` 相当） |
| rustc | （記入: `rustc --version`） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`crates/backend-cuda/examples/gemm_mma_bench.rs::SEED`） |

### 正方形状（tiled f32／WMMA f16／mma.sync f16）

| size | tiled f32 TFLOPS | WMMA f16 TFLOPS | mma.sync f16 TFLOPS | mma/tiled | mma/wmma |
|------|------|------|------|------|------|
| 512  | | | | | |
| 1024 | | | | | |
| 2048 | | | | | |
| 4096 | | | | | |

### 対 PyTorch 比（size=4096・PoC-v2-3 実測 97.6 TFLOPS 基準）

| mma.sync f16 TFLOPS (size=4096) | 対 PyTorch 比 |
|------|------|
| | |

### 数値一致複合判定

| テスト | 結果 |
|--------|------|
| `mma_f16_parity_smoke_env_adaptive`（通常 CI・16x8x16） | （記入） |
| `mma_f16_matches_reference_across_shapes`（`--ignored`） | （記入） |
| `mma_f16_k4096_stress`（`--ignored`） | （記入） |
| `mma_f16_cross_check_against_wmma_f16`（`--ignored`） | （記入） |

複合判定を外れた場合は閾値を緩和せず、誤差分布データを #186（Tensor Core 経路の数値一致閾値の実測再評価）へ引き渡す
（`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。

## 段階別の改善内訳（実測後に記入）

実装計画は 3 段階（段階 1: `mma.sync`+`ldmatrix`+同期ロード、段階 2: `cp.async` パイプライン化、段階 3: swizzle）を
想定していたが、本実装は段階 3（XOR swizzle）を不採用としている（下記「スコープ外」参照）。段階 1→2 の改善幅を
分離計測したい場合は、`kernels_mma.rs::MMA_F16` の `cp.async`/`ldmatrix` 呼び出しを同期ロードへ一時的に置き換えた
比較ビルドで計測し、本節に追記する。

## Phase B 完了時点の再計測（#502）

イシュー #502「Phase B 完了時点の f32/f16 スループットと対 PyTorch 比を再計測・記録」の記録節。
GEMM 性能改善ツリー Phase B（親 #490）の B-0〜B-10（#491〜#501）は全て CLOSED 済みだが、B-1〜B-9 の
カーネル改修は NVRTC 非搭載環境で実装されたため、実機（DGX Spark GB10・sm_121）での構文検証・性能実測が
本イシュー時点でも一度も行われていない。

### 実機到達性の確認結果（2026-08-16・本実装セッション）

- `docs/real-hardware-verification-env.local.md`（実ホスト名・接続情報を記す Git 管理外ファイル）は本
  worktree に**存在しない**（`docs/real-hardware-verification-env.local.md.example` のみ存在）。
- 実機ホスト名（`spark-dbd9`。#500・#656 で使用実績のある個体名）は本ホストから名前解決できず、SSH
  到達性を確認する前提（`CUDA_NODE` の取得）が満たせない。
- したがって実装計画の実機到達性ゲート（手順 3）は**不達**と判定し、実機実測は行わず安全側（推定値を
  記載しない）に倒した。§7「実測結果」以下の表は未計測のまま確定させる（#656・#500 §7 の先例と同じ
  判断）。

### 実装した変更（実機セッションが即実行できるようにする準備。実測は含まない）

`crates/backend-cuda/examples/cuda_floor_bench.rs` の判定対象外形状に 1024 を追加した
（`REFERENCE_ONLY_SIZES: [usize; 2] = [512, 1024]`。従来は `REFERENCE_ONLY_SIZE = 512` の単一形状）。
`JUDGED_SIZES`（2048・4096）・候補下限の丸め規則・経路選択ロジックは無変更。PoC-v2-3 固定値
（`pytorch_f32_fixed`/`pytorch_f16_fixed`）に 1024 の実測が存在しないため、同一実機再計測値
（`CUDA_FLOOR_BENCH_PYTORCH_{F32,F16}_1024`）の env 注入がある場合のみ 1024 の対 PyTorch 比を表示し、
未注入時は既存の `is_finite()` フィルタで自然に `n/a` 表示になる（新規の特殊分岐は追加していない。
`pytorch_f32_fixed`/`pytorch_f16_fixed` ドキュメンテーションコメント参照）。

検証済み（本ホスト・GPU 不要のロジック検査のみ。実機到達性が無いため以下に限定）:

- `cargo fmt --all` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`: green
- `cargo build -p backend-cuda --all-targets`: green（`cudarc` 動的ロードのため CUDA toolkit 非搭載環境でも成立）
- `cargo test -p backend-cuda`（`cuda_floor_bench` の `#[cfg(test)]` 単体検査 13 件を含む）: green
- `git diff origin/main -- crates/backend-cuda/tests/common crates/backend-cuda/src crates/bench-harness`:
  無差分（tolerance 定数・parity fixture・`FloorSpec`・カーネルソースは本イシューで一切変更していない）

### 目標達成状況・未達要因

**未計測のため判定不能。** 目標（対 PyTorch 比 f32 35〜40% / f16 25〜30%。#490 本文の期待効果、ベース
ラインは f32 25.64% / f16 12.97%〈`docs/perf/performance-floor-decision.md` §9〉）に対する達成状況・
未達要因の分析は、実機到達後の再実行セッションへ引き継ぐ。REQ-8 下限（f32 25% / f16 10%。
`bench-harness` の確定値）は本イシューでは変更しない（#577〈Phase F・人間承認タスク〉へ申し送り）。

### 実機セッションでの再実行手順（引き継ぎ）

1. `docs/real-hardware-verification-env.local.md`（`.example` を基に用意）から `CUDA_NODE` を取得し
   `ssh -o BatchMode=yes -o ConnectTimeout=10 "$CUDA_NODE" 'hostname && nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader'`
   で到達性・GPU 排他性を確認する
2. `docs/real-hardware-verification-env.md` §3 の rsync 手順でコードを転送し、`.rev-stamp` でリビジョン
   一致を確認する
3. `cargo test -p backend-cuda --test parity_nonregression -- --ignored --test-threads=1` ほか
   `--ignored` テスト群で数値一致・parity 非後退を性能値採用より先に確認する（後退時は性能値を採用せず
   打ち切る）
4. `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py <size> 20 20` を
   size ∈ {512, 1024, 2048, 4096} × {f32, f16} で同一実機実行し、PyTorch 参照値を再計測する
5. `CUDA_FLOOR_BENCH_PYTORCH_{F32,F16}_{512,1024,2048,4096}` と `CUDA_FLOOR_BENCH_PYTORCH_SOURCE` を
   設定し `cargo run -p backend-cuda --example cuda_floor_bench --release --locked` を 3 回反復実行、
   run 間中央値を代表値として本節・`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` §7 へ機械転記する

## スコープ外（out-of-scope-tracking.md に従い記録）

- **XOR swizzle によるバンクコンフリクト低減**（実装計画「段階 3」）: 索引演算が最も複雑でありながら、本実装
  セッションでは NVRTC によるコンパイル検証ができず誤りを検出できないため不採用とした。`kernels_wmma.rs`
  （#61）が確立した「実機未接続・コンパイル未検証時はリスク最小化のため縮小する」判断を踏襲する
  （`kernels_mma.rs` 冒頭コメント参照）。性能実測が可能な環境で導入を検討する。
- **TF32 の `mma.sync` 化**: #62 の帰結を見て判断（実装計画 8 節）。
- **ディスパッチ規則への組み込み**（どの経路をいつ選ぶか）: TASK-11.2（#66）のスコープ。`CudaMmaGemm` は独立構造体
  として提供する。
- **ブロックタイル拡大・warp あたり複数 mma タイル化・レジスタブロッキング**: 実装計画候補値（128x128・BK=32）
  からの縮小（`BM=32`・`BN=64`・`BK=32`・1 warp = 1 mma タイル）を、コンパイル未検証環境でのリスク最小化として
  採用した。共有メモリ使用量（18432B）は per-block 48KiB 上限に対し余裕があるため、実機実測が可能になった段階で
  拡大を検討できる（`kernels_mma.rs` 冒頭コメント「タイル構成」参照）。
- **sm_121（DGX Spark GB10）固有の実機実測・数値一致検証**: #64 のスコープ。
