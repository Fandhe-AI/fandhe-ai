# CUDA Tensor Core（WMMA TF32／f16）実機実測 記録（#64・TASK-11.1e）

イシュー #64「test(backend-cuda): TASK-11.1e 実機実測・数値一致検証（`#[ignore]` 分離）」の実測記録テンプレート。
受け入れ条件「実機実測記録（TFLOPS・複合判定通過）が残されている」に対応する。

## 状態: 実測未実施（実装環境は Linux＋RTX 3060・libnvrtc 非搭載のため NVRTC コンパイル不可）

本実装セッションの環境は Linux（RTX 3060、compute capability 8.6）だが CUDA toolkit（`libnvrtc`）が
未搭載のため、WMMA カーネルの NVRTC コンパイル・実行検証はできない。toolkit の導入はグローバル状態の
変更にあたるため本セッションでは行わない（`.claude/rules/ci.md`「グローバル状態を汚す処理を書かない」の
趣旨を作業環境にも適用した安全側判断）。Metal 側の確立済み先例（[`metal-gemm-dynamic-tile.md`](./metal-gemm-dynamic-tile.md)・
[`../backend-metal-real-device-testing.md`](../backend-metal-real-device-testing.md))と同じく、実測テスト・
構造化出力・記録テンプレートを本セッションの成果物として固定し、実機（DGX Spark GB10 等）実行後に
下記テンプレートへ転記する運用とする。

代わりに以下は本実装セッションで検証済み:

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載環境でもビルド成立する。
  `.claude/rules/coding-rust.md`）を崩していないことを確認済み
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda --release` — 既存スモークテスト（環境適応。CUDA 非搭載環境では早期 return で
  green）に回帰がないこと、新規 `#[ignore]` テスト 2 件（[`tensor_core_tflops_record`]・
  [`tensor_core_parity_record`]）が通常実行から除外されていること
- `cargo test -p backend-cuda --release -- --list` — 上記 2 件がテスト一覧に登録されていること

## 計測手順（DGX Spark GB10 等 CUDA 実機）

```sh
git fetch origin
git checkout test/64-cuda-real-device-measurement   # 本イシューの実装ブランチ
make test-ignored-cuda                              # backend-cuda に限定した #[ignore] テスト実行（release）
# 相当コマンド（本ファイルの 2 テストのみに絞る場合）:
cargo test -p backend-cuda --release -- --ignored --nocapture tensor_core_
```

出力形式（`crates/backend-cuda/tests/tensor_core_real_device.rs` 参照）:

- `environment: ...` 行: `CudaDevice::name()`／`compute_capability()`／`arch()`（下表「計測環境」への転記元）
- `tflops_record path=<経路> tflops=<値> report=<BenchReport JSON>` 行: tiled f32／WMMA TF32 opt／WMMA f16
  opt 各経路の TFLOPS 換算値と `bench_harness::report::BenchReport::to_json`（warmup 20・計測 20・中央値・
  Q1/Q3 を含む構造化出力。TASK-8.1 準拠）
- `parity_record path=<経路> shape=512x512x512 result=pass ...` 行: `backend_cpu::assert_parity`
  （REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）通過の記録

数値一致確認（受け入れ条件に必須の前提。上記 `tensor_core_parity_record` に加え、経路別 parity テストも
先に確認してから性能値を採用する）:

```sh
cargo test -p backend-cuda --release -- --ignored --nocapture wmma_
```

## 実測結果（記入待ち）

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU（`CudaDevice::name()`） | （記入: 例 NVIDIA GB10） |
| compute capability（`CudaDevice::compute_capability()`） | （記入: 例 (12, 1)） |
| arch（`CudaDevice::arch()`） | （記入） |
| driver バージョン | （記入: `nvidia-smi` 出力等） |
| rustc | （記入: `rustc --version`） |
| commit SHA | （記入） |
| 実施日 | （記入） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | tiled/TF32: `0xACE1`、WMMA f16: `0xBEEF01`（`tests/tensor_core_real_device.rs::tensor_core_tflops_record`） |

### TFLOPS 実測（M=N=K=4096）

| 経路 | opt 可用性 | median TFLOPS | Q1 TFLOPS | Q3 TFLOPS | 対 tiled 比 | 対 PoC-v2-3（1.832 TFLOPS）比 |
|------|------|------|------|------|------|------|
| tiled f32（基準） | - | | | | 1.00 | |
| WMMA TF32（opt） | （記入: true/false） | | | | | |
| WMMA f16（opt） | （記入: true/false） | | | | | |

### 複合判定通過（M=N=K=512）

| 経路 | 判定 | 備考 |
|------|------|------|
| WMMA TF32（opt） | （記入: pass/fail） | `backend_cpu::assert_parity`（相対 1e-3 未満 または 絶対 1e-5 未満） |
| WMMA f16（opt） | （記入: pass/fail） | 同上（f16→f32 参照計算→f16 丸め→f32 化の量子化手順。`tests/cpu_cuda_wmma_parity.rs` と同一） |

複合判定が実機で外れた場合は許容誤差を緩和せず、本節に実測値・エラー内容を記録したうえで #186 の
閾値実測再評価へ引き渡す（`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず
人間の承認を経る」・`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で
緩和しない」）。

## 関連イシューとの役割分担（二重管理を避ける）

- **#186**（Tensor Core 経路の数値一致閾値の実測再評価）: TF32/f16 経路の誤差分布実測・閾値そのものの
  再評価。本ファイルは「現行閾値〈1e-3/1e-5〉で通過するか」の記録に留め、閾値の妥当性検討自体は #186 へ委ねる
- **#187**（mma.sync/cp.async パイプラインの実測）: 対 PyTorch 比を含む、本ファイルより高度な最適化経路の
  性能検証。本ファイルは既存 WMMA 基本・opt カーネルの記録に限定する
- **TASK-11.3**（証跡整備・`docs/matrix-unit-dispatch.md`）: ディスパッチ規則（どの形状・cc でどのカーネルを
  選ぶか）の証跡。本ファイルは実測値の記録のみを担い、ディスパッチ規則の記述は行わない

## 未実施・後続作業

- 本ファイルの「実測結果」節は DGX Spark GB10 等 CUDA 実機での `make test-ignored-cuda` 実行後に埋める
  （実機アクセス確保後の作業。新規 Issue 起票はユーザー承認が必要なため本セッションでは行わない）
- 実測値が閾値境界付近・PoC-v2-3 比で予期しない結果の場合、本節に根拠を追記したうえで #186／#187 との
  役割分担に従って後続対応を切り出す
