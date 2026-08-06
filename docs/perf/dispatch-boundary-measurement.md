# GEMM 経路選択 境界形状 実測記録（#69・TASK-11.2c）

イシュー #69「test(backend): TASK-11.2c 境界形状の実測再検証」の実測記録テンプレート。
受け入れ条件「境界値の実測記録と採用した閾値の根拠が残されている」に対応する。

対象は `crates/tensor-core/src/dispatch.rs::select_gemm_kernel`（TASK-11.2b・#68 で実装済みの
自作ディスパッチ規則）が参照する暫定閾値 3 つ（`METAL_SIMDGROUP_MIN_DIM`・`CUDA_WMMA_MIN_CC`・
CUDA の「形状下限なし」設計）。設計文書は `docs/dispatch-rules-design.md`（§2〜§4・#67）。

## 状態: 実測未実施（実機なし。実装環境は Linux・NVRTC 非搭載）

本実装セッションは Linux worktree で行っており、Metal 実機（Apple Silicon）・CUDA NVRTC 搭載実機
（DGX Spark GB10 等）のいずれも同一セッションで使用できない（`nvidia-smi` 相当のドライバ照会は
先行イシュー #64・#187 で RTX 3060・compute capability 8.6 の存在を確認済みだが、`libnvrtc` が
この環境に存在せず `CudaError::NvrtcUnavailable` に必ず倒れる。`docs/perf/cuda-gemm-mma-pipeline.md`
「状態」節と同一の制約）。本ファイルは計測手順・記録テンプレート・**採用中の暫定閾値の根拠**を
整備し、実機実行後に下記テンプレートへ結果を転記する運用とする（`docs/perf/cuda-tensor-core-measurement.md`
〈#64〉・`docs/perf/metal-gemm-dynamic-tile.md`〈#188〉と同じ先例に従う）。

代わりに以下は本実装セッションで検証済み:

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載でもビルド成立）
  を崩していない
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda --release` / `cargo test -p backend-metal --release`（Linux で実行可能な
  範囲。新規 `#[ignore]` テストは通常実行から除外される）
- `cargo test -p backend-cuda --release -- --list` / `-p backend-metal -- --list` — 新規 `#[ignore]`
  テスト（`dispatch_boundary.rs` の各 `#[test]`）がテスト一覧に登録されている
- Metal 側の型検査は Makefile のクロスチェック方式（`--target aarch64-apple-darwin`）で
  `crates/backend-metal/tests/dispatch_boundary.rs` の結線を確認済み

## 計測手順

### Metal（Apple Silicon 実機）

```sh
git fetch origin
git checkout test/69-dispatch-boundary-measurement   # 本イシューの実装ブランチ
cargo test -p backend-metal --release -- --ignored --nocapture dispatch_boundary
```

出力形式（`crates/backend-metal/tests/dispatch_boundary.rs` 参照）:

- `dispatch_boundary_record dim=<N> path=tiled tflops=...` / `path=simdgroup_auto tflops=...
  simdgroup_auto_over_tiled=<比率>` 行: `min(M,N,K)` = 256/384/448/512/576/640/768/1024（正方）での
  2 経路比較
- `dispatch_boundary_route_record dim=<N> min_dim_threshold=512 expected_kernel=<KernelKind>
  result=parity_pass` 行: 各境界形状で `dispatch_backend_auto` が決定表どおりの経路を選び、
  CPU 参照実装との複合判定に通過したことの記録

### CUDA（DGX Spark GB10・compute capability 8.0 以上・NVRTC 搭載）

```sh
git fetch origin
git checkout test/69-dispatch-boundary-measurement
cargo test -p backend-cuda --release -- --ignored --nocapture dispatch_boundary
```

出力形式（`crates/backend-cuda/tests/dispatch_boundary.rs` 参照）:

- `dispatch_boundary_record dim=<N> path=tiled_f32|wmma_tf32_opt|tiled_f16|wmma_f16_opt
  tflops=... matrix_unit_over_tiled=<比率>` 行: 小形状 128/256/384/512 での MatrixUnit(WMMA) 対
  Tiled 比較（「形状下限なし」規則の実測根拠）
- `dispatch_boundary_record dim=<N> path=wmma_f16_opt|mma_sync_f16 tflops=... mma_over_wmma=<比率>`
  行: 大形状 2048/4096 での `mma.sync` パイプライン対基本 WMMA 比較（TMA 選好整理の実測根拠）

数値一致確認（受け入れ条件に必須の前提。性能値採用より先に実施すること）:

```sh
cargo test -p backend-metal --release -- --ignored --nocapture   # gemm_auto_parity.rs・dispatch_boundary.rs 双方
cargo test -p backend-cuda --release -- --ignored --nocapture    # gemm_auto.rs・cpu_cuda_*_parity.rs 双方
```

`tests/gemm_auto_parity.rs`（Metal）・`tests/gemm_auto.rs`／`tests/cpu_cuda_wmma_parity.rs`（CUDA）の
既存 `#[ignore]` ケースが全て PASS することを先に確認する（`dispatch_boundary.rs` 自体も
`assert_parity` で複合判定を検証するが、経路選択の網羅的な数値一致検証は既存ファイルが担当する）。

## 実測結果（記入待ち）

### 計測環境

| 項目 | 値 |
|------|-----|
| Metal GPU | （記入: 例 Apple M4 Max） |
| Metal OS | （記入: macOS バージョン） |
| CUDA GPU | （記入: 例 DGX Spark GB10。compute capability を明記） |
| CUDA driver / NVRTC バージョン | （記入: `nvidia-smi`・`nvcc --version` 相当） |
| rustc | （記入: `rustc --version`） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |

### Metal: 境界形状（tiled 対 simdgroup_matrix 動的タイル選択）

| min(M,N,K) | tiled TFLOPS | simdgroup_auto TFLOPS | auto/tiled | `select_gemm_kernel` 期待経路 | 選択経路一致 | parity |
|------|------|------|------|------|------|------|
| 256  | | | | Tiled | | |
| 384  | | | | Tiled | | |
| 448  | | | | Tiled | | |
| 512  | | | | MatrixUnit | | |
| 576  | | | | MatrixUnit | | |
| 640  | | | | MatrixUnit | | |
| 768  | | | | MatrixUnit | | |
| 1024 | | | | MatrixUnit | | |

### CUDA: 小形状（tiled 対 WMMA・形状下限なし規則の検証）

| dim | tiled f32 TFLOPS | WMMA TF32(opt) TFLOPS | tf32 matrix_unit/tiled | tiled f16 TFLOPS | WMMA f16(opt) TFLOPS | f16 matrix_unit/tiled |
|------|------|------|------|------|------|------|
| 128  | | | | | | |
| 256  | | | | | | |
| 384  | | | | | | |
| 512  | | | | | | |

### CUDA: 大形状（WMMA 対 mma.sync パイプライン・TMA 選好整理の検証）

| dim | WMMA f16(opt) TFLOPS | mma.sync f16 TFLOPS | mma/wmma |
|------|------|------|------|
| 2048 | | | |
| 4096 | | | |

## 採用閾値の根拠表

| 閾値・設計 | 現行値 | 根拠（v1 参考値の出典＋v2 保守設計の理由） | 実測後の判定基準 |
|---|---|---|---|
| `METAL_SIMDGROUP_MIN_DIM`（`crates/tensor-core/src/dispatch.rs`） | `512` | v1 PoC-8 実測は 256/512 の 2 点のみ（`docs/dispatch-rules-design.md` §3.1「同上 `:75`」）。境界形状（384・640 等）は未計測のため 512 を暫定的に踏襲し、閾値未満は保守的に tiled へ倒す設計（§3.2） | 上表「Metal: 境界形状」でクロスオーバー（`auto/tiled` が 1.0 を跨ぐ形状）を特定する。クロスオーバー形状が 512 から大きく外れる場合（例: 384 や 768 で既に逆転）は閾値をクロスオーバー形状へ更新する後続 PR を起票する。511/512 付近で単調にクロスするなら 512 を確定値として採用しコメントの「暫定」表記のみ更新する |
| `CUDA_WMMA_MIN_CC`（`(7, 0)`） | `(7, 0)` | WMMA は Volta（cc 7.0）以降という一般的な NVIDIA アーキテクチャ世代対応（`docs/dispatch-rules-design.md` §2 表）。cc 世代境界自体の実機再確認は本イシューのスコープだが、cc 7.x の実機（Volta/Turing）が本セッション・後続実機いずれにも存在しないため世代境界の実測は据え置く | 実機実測の対象外（compute capability 境界の実測には該当世代の実機が必要）。RTX 3060（cc 8.6）・GB10（cc 12.1）いずれも `cc >= 7.0` を満たすため、本イシューの実測は「ゲートを満たした場合に MatrixUnit が有利であること」の検証に限定し、`(7, 0)` 自体は変更しない |
| CUDA の「形状下限なし」設計（§3.2） | 形状閾値なし（HW ゲートのみ） | GB10 実測で最小形状 256 でも accelerated が unit の約 1.4〜1.6 倍優位（`docs/dispatch-rules-design.md` §3.1 `:126`・`:140`）という v1 CubeCL 前提の参考値 | 上表「CUDA: 小形状」の `matrix_unit/tiled` が 128/256/384/512 のいずれでも 1.0 を上回れば設計を維持する。1.0 を下回る形状が観測された場合（小形状で tiled が有利に逆転）は Metal と同様の形状閾値導入を検討し、別レビュー・別 PR で提案する（本イシューでは導入しない） |
| CUDA「TMA 選好はディスパッチ条件でなくカーネル内部チューニング」の整理（§3.2「TMA の扱い」） | `select_gemm_kernel` は `mma.sync` パイプラインと基本 WMMA を区別しない（`CudaGemmAuto::run_f16` は `CudaWmmaGemm` のみを呼ぶ） | v1 実測（`poc-8-matrix-unit/README.md:125`）は M=N=K=2048/4096 で TMA 系候補が最速だが、これは「同じ Tensor Core 経路内でのカーネル変種選択」であり「Tensor Core 経路を使うか否か」の分岐条件ではないという v2 設計判断 | 上表「CUDA: 大形状」で `mma_over_wmma` が 1.0 を上回れば整理を維持する（パイプライン差は経路選択の分岐にしない設計のまま、カーネル内部の既定実装をどちらにするかの判断材料として別途記録する）。1.0 を大きく下回る場合（mma パイプラインが未成熟で基本 WMMA より遅い）は `CudaGemmAuto::run_f16` の既定カーネル選択を見直す別 Issue を起票する |

閾値変更が必要と判断した場合は、本表の「実測後の判定基準」に従い `crates/tensor-core/src/dispatch.rs`
の定数・ドキュメンテーションコメントを更新する別レビュー・別 PR で対応する（ガードレール閾値・テスト
許容誤差の変更ではないため本イシュー実装フローの`.claude/rules/security.md` 対象外だが、
`.claude/rules/delegation-impl.md`「実装 Agent にガードレール閾値・テスト許容誤差を緩和させない」との
混同を避けるため、本イシュー内では変更しない）。

## 未実施・後続作業

- 本ファイルの「実測結果」節は Apple Silicon 実機（Metal）・DGX Spark GB10 等（CUDA、NVRTC 搭載）
  での `cargo test --release -- --ignored --nocapture` 実行後に埋める
- 実測に基づく閾値変更（`METAL_SIMDGROUP_MIN_DIM` の更新・CUDA 形状閾値の導入検討）は上記「採用閾値の
  根拠表」の判定基準に従い、別レビュー・別 PR で行う（本イシューのスコープ外。実装計画 §7）
- 証跡整備（カーネルソース内命令＋ベンチログの体系化）は #70（TASK-11.3）が担当する
