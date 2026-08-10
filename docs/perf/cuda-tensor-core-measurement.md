# CUDA Tensor Core（WMMA TF32／f16）実機実測 記録（#64・TASK-11.1e）

イシュー #64「test(backend-cuda): TASK-11.1e 実機実測・数値一致検証（`#[ignore]` 分離）」の実測記録テンプレート。
受け入れ条件「実機実測記録（TFLOPS・複合判定通過）が残されている」に対応する。

## 状態: 実測完了（DGX Spark GB10。#389）

下記「実測結果」節は #389（CUDA 実機 `#[ignore]` テスト 51 件の実行・結果記録）で DGX Spark GB10 実機
実行時に埋めた。詳細な失敗内訳・エスカレーション先は
[`../backend-cuda-real-device-testing.md`](../backend-cuda-real-device-testing.md) を正とし、本ファイルは
本セッション由来のテンプレート節への転記のみに留める（二重管理を避ける）。

## 参考: 本ファイル作成時点（実装環境は Linux＋RTX 3060・libnvrtc 非搭載）の制約記録

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

## 実測結果（#389・2026-08-10 実測）

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU（`CudaDevice::name()`） | NVIDIA GB10 |
| compute capability（`CudaDevice::compute_capability()`） | (12, 1) |
| arch（`CudaDevice::arch()`） | `compute_121` |
| driver バージョン | 580.159.03 |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| commit SHA | `720bf633e12471526a31dbe632a86bbe2150a8f4` |
| 実施日 | 2026-08-10 |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | tiled/TF32: `0xACE1`、WMMA f16: `0xBEEF01`（`tests/tensor_core_real_device.rs::tensor_core_tflops_record`） |

### TFLOPS 実測（M=N=K=4096。単発〈並列〉実行時点の値。並列競合の影響は
`../backend-cuda-real-device-testing.md` 5.1 節参照）

| 経路 | opt 可用性 | median TFLOPS | 対 tiled 比 | 対 PoC-v2-3（1.832 TFLOPS）比 |
|------|------|------|------|------|
| tiled f32（基準） | - | 0.189〜0.233 | 1.00 | 約 10〜13% |
| WMMA TF32（opt） | true | 0.201〜0.251 | 約 1.06〜1.08 倍 | 約 11〜14% |
| WMMA f16（opt） | true | 0.380〜0.466 | 約 2.0 倍 | 約 21〜25% |

tiled f32 の TFLOPS が PoC-v2-3 実測値（1.832 TFLOPS）を大きく下回っている点・WMMA opt 経路が tiled を
明確に上回らない点は、5.1 節が示す並列実行時の計測歪みの影響を受けた可能性が高く、本ファイル単独では
「実機の実性能」と断定しない（#390/#391 に引き継ぐ）。

**tiled f32 基準値の突合（重要）**: 上表の tiled f32（0.189〜0.233 TFLOPS。`tensor_core_real_device.rs::
tensor_core_tflops_record` 実測）は、同じ M=N=K=4096 形状を別バイナリ（`gemm_wmma_tf32_opt.rs::
wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`）で計測した tiled f32 基準値 1.187〜1.237 TFLOPS と
約 5 倍乖離している（詳細・原因分析は
[`../backend-cuda-real-device-testing.md`](../backend-cuda-real-device-testing.md) 5.1 節「tiled f32
基準値の突合」を参照）。`tensor_core_tflops_record` は同一バイナリ内に GPU を使う
`tensor_core_parity_record` を併載しており、直後の並列競合フレーキー性の記述（5.1 節）と整合する形で
低い方の値も歪んでいる疑いが強い。したがって上表から導いた「対 PoC-v2-3 約 10〜25%」という評価は
過小評価の可能性があり、**どちらの値が実機の実性能に近いかは本ファイル単独では確定できない**
（直列実行下での再測定を #390／#391 に引き継ぐ）。

### 複合判定通過（M=N=K=512。TF32／f16）

| 経路 | 判定 | 備考 |
|------|------|------|
| WMMA TF32（`tensor_core_parity_record`） | **fail**（fail_count=42493/262144, mean_abs_diff=1.574e-3） | `backend_cpu::assert_parity`（相対 1e-3 未満 または 絶対 1e-5 未満）。恒常的 fail |
| WMMA f16（`tensor_core_parity_record`） | **未計測（到達せず）** | `tensor_core_parity_record` は TF32 判定（`backend_cpu::assert_parity`。`#[track_caller]` 付き `assert!` で FAIL 時に panic する）を先に実行するため、TF32 側が panic した時点で同一テスト関数内の f16 判定コードには到達しない（`crates/backend-cuda/tests/tensor_core_real_device.rs::tensor_core_parity_record` 参照）。512×512×512 形状での f16 実測値はこのテストからは得られない |

複合判定が実機で外れたため、許容誤差は緩和せず、実測値を上記の通り記録したうえで #186 の閾値実測
再評価へ引き渡した（`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず人間の
承認を経る」・`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。
f16 経路（他形状・K=4096 ストレスケースを含む）・その他形状での parity 実測を含む完全な内訳は
[`../backend-cuda-real-device-testing.md`](../backend-cuda-real-device-testing.md) 5.3 節を参照。

## 関連イシューとの役割分担（二重管理を避ける）

- **#186**（Tensor Core 経路の数値一致閾値の実測再評価）: TF32/f16 経路の誤差分布実測・閾値そのものの
  再評価。本ファイルは「現行閾値〈1e-3/1e-5〉で通過するか」の記録に留め、閾値の妥当性検討自体は #186 へ委ねる
- **#187**（mma.sync/cp.async パイプラインの実測）: 対 PyTorch 比を含む、本ファイルより高度な最適化経路の
  性能検証。本ファイルは既存 WMMA 基本・opt カーネルの記録に限定する
- **TASK-11.3**（証跡整備・`docs/matrix-unit-dispatch.md`）: ディスパッチ規則（どの形状・cc でどのカーネルを
  選ぶか）の証跡。本ファイルは実測値の記録のみを担い、ディスパッチ規則の記述は行わない

## 未実施・後続作業

- 実測完了（#389）。TFLOPS の並列競合影響切り分け・PyTorch 比の精緻化は #390／#391 に、複合判定
  fail の閾値再評価は #186 に引き渡し済み（本ファイル・`../backend-cuda-real-device-testing.md` 参照）
