# Metal `min`・CUDA/Metal `log_softmax` forward 実機ランブック（イシュー #2155）

Metal `min`（`crate::reduce::MetalReduce::run_min_all_f32`／
`run_min_axis_f32`。`crate::ops::MetalBackendOps::min` 経由）・
CUDA/Metal `log_softmax` forward（`CudaSoftmax::run_log_softmax_f32`／
`MetalSoftmax::run_log_softmax_f32`。`CudaBackendOps::log_softmax`／
`MetalBackendOps::log_softmax` 経由）の実機（Apple Silicon・DGX Spark
GB10 等）実測ランブック。本実装エージェントの実行環境（Linux）には
いずれの実機への到達手段もないため、実測は未実施のまま Mac／DGX
Spark セッションへ申し送る（`metal-argext-1951/README.md`・
`metal-reduce-1895/README.md` と同型）。

設計判断・契約の正は `docs/backend-metal-reduce-sum-design.md` §13
（`min`）・`docs/gpu-log-softmax-forward-decision.md`（`log_softmax`
forward）。

## 実行コマンド

```sh
# Linux（本実装セッション）で完了済みの型検査・非実機テスト:

# min（Metal）
cargo test -p fandhe-ai-backend-metal --lib reduce_model
cargo test -p fandhe-ai-backend-metal --test reduce_source_evidence
cargo test -p fandhe-ai-backend-metal --test backend_ops_real_device \
  min_shape_errors_without_device_init

# log_softmax forward（CUDA）
cargo test -p fandhe-ai-backend-cuda --lib kernels_softmax
cargo test -p fandhe-ai-backend-cuda --test backend_ops_real_device \
  log_softmax_non_last_axis_is_unsupported_without_device
cargo test -p fandhe-ai-backend-cuda --test backend_ops_real_device \
  log_softmax_parity_smoke_env_adaptive
cargo test -p fandhe-ai-backend-cuda --test log_softmax_parity \
  log_softmax_parity_smoke_env_adaptive

# log_softmax forward（Metal）
cargo test -p fandhe-ai-backend-metal --test rmsnorm_softmax_source_evidence
cargo test -p fandhe-ai-backend-metal --test backend_ops_real_device \
  log_softmax_non_last_axis_unsupported_without_device_init
cargo test -p fandhe-ai-backend-metal --test softmax_parity \
  backend_ops_log_softmax_non_final_axis_is_unsupported

# 共通
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
cargo check -p fandhe-ai --tests --target aarch64-apple-darwin
cargo test -p fandhe-ai --test api_surface
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked

# Apple Silicon 実機（Mac セッション。--release 推奨）:
cargo test -p fandhe-ai-backend-metal --release --test reduce_parity -- \
  --ignored --nocapture metal_min
cargo test -p fandhe-ai-backend-metal --release --test softmax_parity -- \
  --ignored --nocapture log_softmax
cargo test -p fandhe-ai-backend-metal --release --test backend_ops_real_device -- \
  --ignored --nocapture backend_ops_min_matches_cpu
cargo test -p fandhe-ai --release --test reduce_backend_parity -- \
  --ignored --nocapture metal_min_all_and_axis_forward_and_backward_match_cpu

# A/B 5 run 中央値（Mac: FANDHE_BENCH_DEVICE=metal・cpu の両方を計測して
# ratio を算出。crates/facade/tests/gpu_min_log_softmax_bench.rs）:
FANDHE_BENCH_DEVICE=cpu cargo test -p fandhe-ai --release \
  --test gpu_min_log_softmax_bench -- --ignored --nocapture
FANDHE_BENCH_DEVICE=metal cargo test -p fandhe-ai --release \
  --test gpu_min_log_softmax_bench -- --ignored --nocapture

# DGX Spark GB10 等 CUDA 実機:
cargo test -p fandhe-ai-backend-cuda --release --test log_softmax_parity -- \
  --ignored --nocapture
FANDHE_BENCH_DEVICE=cuda cargo test -p fandhe-ai --release \
  --test gpu_min_log_softmax_bench -- --ignored --nocapture log_softmax_forward_fixed_shapes

# 既存 #[ignore] 群の非後退確認（新規カーネル追加による回帰がないこと。
# 両実機とも実施）。
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
cargo test -p fandhe-ai --release -- --ignored --nocapture
```

## 保存すべきログ

- 上記各コマンドの標準出力（pass/fail・所要時間）
- `docs/real-hardware-verification-env.local.md.example` に準じた
  `env_info.txt`（macOS／CUDA ドライバのバージョン・チップ／GPU 世代。
  内部ホスト名は含めない）
- A/B 計測（下記「5 run 中央値・非後退判定」参照）の生ログ

## 事前登録判定規則（事後の緩和は禁止。FAIL は是正せず記録のみ）

1. **parity**（新規実機テストがすべて pass すること）
   - `min` は CPU 参照実装（`fandhe_ai_backend_cpu::reduction::min`）
     と**値一致**（`0.0 == -0.0` として比較）。全要素 NaN の縮約は
     `+inf`。±0 を含まない入力は bit 一致を追加確認してよい。
   - `log_softmax` forward は REQ-2 統一複合判定（相対誤差 1e-3 未満
     または絶対誤差 1e-5 未満）。非有限の行（全 `-inf`・`+inf` 混在・
     NaN 混在）は NaN クラス一致。
2. **checksum 完全一致**
   - GPU 経路の 5 run 間で出力 checksum（bit 列の FNV-1a 等）が完全
     一致すること（決定性）。
   - `min` は base（ホストフォールバック。イシュー #2155 以前の
     `Var::min` 経路）と after（GPU）の間でも checksum が完全一致
     すること（±0 を含まない入力で）。
   - `log_softmax` は base/after 間の checksum 一致を要求しない
     （`exp2` と `exp` の丸め差があるため）。REQ-2 判定で代替する。
3. **5 run 中央値・非後退 ratio<=1.00**
   - 固定形状: `min` は `[4096,4096]` の `dim=None`／`dim=0`／`dim=1`、
     `log_softmax` は `[4096,4096]` の `dim=1` と `[256,32768]` の
     `dim=1`（CUDA 2 パス経路・Metal 2 パス経路）。
   - after の中央値 / base の中央値 <= 1.00 なら合格。FAIL は記録の
     みとする（H2D/D2H 往復込みでホストに劣る形状もありうることを
     事前に明記する）。
4. **非後退**: 既存の `#[ignore]` 群（`reduce_parity`・
   `softmax_parity`・`log_softmax_backward_parity`・facade の
   `softmax_backend_parity`／`reduce_backend_parity` 等）に新規 FAIL
   がないこと。

## 実測結果（未実施）

| テスト | 結果 | ログファイル |
|---|---|---|
| `reduce_parity.rs::metal_min_all_matches_cpu` | 未実測 | - |
| `reduce_parity.rs::metal_min_axis_matches_cpu` | 未実測 | - |
| `reduce_parity.rs::metal_min_nan_zero_inf_edge_cases_match_cpu` | 未実測 | - |
| `backend_ops_real_device.rs::backend_ops_min_matches_cpu`（Metal） | 未実測 | - |
| `reduce_backend_parity.rs::metal_min_all_and_axis_forward_and_backward_match_cpu`（facade） | 未実測 | - |
| `softmax_parity.rs::metal_log_softmax_matches_cpu_across_shapes` | 未実測 | - |
| `softmax_parity.rs::metal_log_softmax_extreme_and_nonfinite_rows` | 未実測 | - |
| `softmax_parity.rs::metal_log_softmax_run_to_run_is_bit_identical` | 未実測 | - |
| `softmax_parity.rs::backend_ops_log_softmax_matches_cpu_reference_across_shapes`（Metal） | 未実測 | - |
| `log_softmax_parity.rs::log_softmax_matches_cpu_across_shapes`（CUDA） | 未実測 | - |
| `log_softmax_parity.rs::log_softmax_extreme_and_nonfinite_rows`（CUDA） | 未実測 | - |
| `log_softmax_parity.rs::log_softmax_run_to_run_is_bit_identical`（CUDA） | 未実測 | - |
| A/B 5 run 中央値（`min`／`log_softmax`） | 未実測 | - |
| 既存 `#[ignore]` 群の非後退（Metal／CUDA／facade） | 未実測 | - |

実測が完了したら本表・上記判定規則の合否・実測環境（チップ／GPU
世代・OS／ドライババージョン）を追記し、`docs/README.md`・
`docs/backend-metal-reduce-sum-design.md` §13・
`docs/gpu-log-softmax-forward-decision.md` §5 の該当記述を更新する。
