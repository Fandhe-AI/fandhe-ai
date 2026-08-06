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

未検証の事項（#64／#65 へ引き継ぐ）:

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
