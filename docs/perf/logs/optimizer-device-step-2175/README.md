# RmsProp・Adagrad・LAMB 常駐 step（`BackendOps::{rmsprop,adagrad,lamb}_step_device`）
GPU 実装の実機ランブック（イシュー #2175）

イシュー #2175 は `BackendOps::rmsprop_step_device`／`adagrad_step_device`／
`lamb_step_device`（`_tracked` 版込み。`RmsPropStepConfig`／
`AdagradStepConfig`／`LambStepConfig`・`docs/device-resident-update-design.md`
「RmsProp・Adagrad・LAMB の常駐 step 結線（イシュー #2175）」節）を新設し、
`Adam`／`AdamW`（#1959）と同じく **CPU 実装のみ**を提供した。CUDA／Metal
は既定実装（常に `BackendError::Unsupported` を返す fail-closed）のまま
であり、GPU カーネル自体が未実装のため、本ディレクトリに記録すべき実機
実測は本イシュー時点では存在しない。

以下は CUDA／Metal のカーネルを実装する**後続イシュー**（ユーザー承認を
得て別途起票する。`out-of-scope-tracking.md` 対象）向けに、その時点で
満たすべき事前登録判定規則・実行コマンド形・保存すべきログ一覧を
あらかじめ固定しておくものである（`adam-device-step-1959/README.md` と
同型の「未実測のまま申し送る」形式を踏襲）。

## 現状（本イシュー #2175 時点）

- CPU 実装（`crates/backend-cpu/src/ops.rs::CpuBackendOps::
  {rmsprop,adagrad,lamb}_step_device`）はホスト参照実装（`crate::
  nn::optim::{rmsprop::RmsProp, adagrad::Adagrad, lamb::Lamb}`）と
  **bit 完全一致**（`crates/backend-cpu/tests/{rmsprop,adagrad,
  lamb}_device_parity.rs`・`crates/facade/tests/device_param_store_
  {rmsprop,adagrad,lamb}_train.rs`。いずれも Linux で完走・pass 済み）。
- CUDA／Metal は各 `*_step_device` の既定実装（`Unsupported`）のまま。
  `DeviceParamStore::step_rmsprop`／`step_adagrad`／`step_lamb` を
  CUDA／Metal バックエンドで呼ぶと、実 GPU の有無に関わらず常に
  `BackendError::Unsupported` を返す（fail-closed。実機なしでも
  Linux で確認可能）。

## 後続イシューが実装すべきカーネル（未着手）

- `crates/backend-cuda/src/{rmsprop,adagrad,lamb}.rs`（新規）・
  `ops.rs::CudaBackendOps::{rmsprop,adagrad,lamb}_step_device(_tracked)`
  の override（`adam.rs`／`kernels_adam.rs` を鏡写しにする設計）。
- `crates/backend-metal/src/{rmsprop,adagrad,lamb}.rs`（新規）・
  `shaders/{rmsprop,adagrad,lamb}.metal`・`ops.rs::MetalBackendOps::
  {rmsprop,adagrad,lamb}_step_device_impl(token)` + override（`adam.rs`／
  `shaders/adam.metal` を鏡写しにする設計）。
- LAMB は `segment_numels` ごとの L2 norm reduction（`norm_x`／
  `norm_t`）を GPU 上で行うカーネルが追加で必要（CPU 実装は
  `crates/backend-cpu/src/ops.rs::lamb_step_device` の segment ループを
  参照。パラメータテンソルごとの独立縮約という「layer-wise」契約を GPU
  側でも維持すること）。

## 実行コマンド（後続イシューでの実施を想定）

```sh
# Linux（実装セッション）で完了させておく型検査・非実機テスト:
cargo test -p fandhe-ai-backend-cuda --lib rmsprop
cargo test -p fandhe-ai-backend-cuda --lib adagrad
cargo test -p fandhe-ai-backend-cuda --lib lamb
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
cargo check -p fandhe-ai --tests --target aarch64-apple-darwin

# DGX Spark GB10（CUDA 実機）:
cargo test -p fandhe-ai-backend-cuda --release --test rmsprop_device_real_device -- \
  --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test adagrad_device_real_device -- \
  --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test lamb_device_real_device -- \
  --ignored --nocapture

# Apple Silicon（Metal 実機。--release 推奨）:
cargo test -p fandhe-ai-backend-metal --release --test rmsprop_device_parity -- \
  --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release --test adagrad_device_parity -- \
  --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release --test lamb_device_parity -- \
  --ignored --nocapture

# 既存 #[ignore] 群の非後退確認（新規カーネル追加による回帰がないこと。
# SGD／Adam 常駐経路を含む）:
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture

# CPU 版 record_only ベンチ（正しさは hard assert・性能比は目視確認）:
cargo test -p fandhe-ai --release --test device_param_store_optim_bench -- \
  --ignored --nocapture
```

## 保存すべきログ

- 上記各コマンドの標準出力（pass/fail・所要時間）
- `docs/real-hardware-verification-env.local.md.example` に準じた
  `env_info.txt`（内部ホスト名・チップ／GPU 世代を含めない範囲での
  実行環境情報）

## 事前登録判定規則

CUDA／Metal のカーネル実装（後続イシュー）に適用する受け入れ基準。

1. **正しさ（REQ-2）**: GPU 実装によるデバイス常駐 1 step（複数 step
   累積を含む）の結果が、ホスト参照実装（`RmsProp::step`／
   `Adagrad::step`／`Lamb::step`）と統一複合判定（`.claude/rules/
   coding-rust.md`「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）で
   一致すること（`crates/backend-cpu/tests/{rmsprop,adagrad,
   lamb}_device_parity.rs` と同型の突合を GPU 側でも実施）。要素ごとの
   更新式自体はレーン間で縮約を伴わない（LAMB の trust ratio 縮約
   〈`norm_x`／`norm_t`〉を除く）ため、CPU 実装が達成した bit 完全
   一致を GPU にまで拡大要求しない。LAMB の trust ratio 縮約について
   結合順序が単一の連続ループと異なる場合は `.claude/rules/
   coding-rust.md`「結合順序が単一の連続 K ループと異なるカーネルの
   parity テスト判定方式」を適用する。
2. **run-to-run bit 同一**: 同一入力・同一設定での GPU 実行を複数回
   繰り返し、出力が bit 単位で決定的であること。
3. **性能**: 5 run 中央値（`bench_harness::median_q1_q3`）で計測し、
   最終パラメータの checksum（`to_bits` の fold）が host 経路と完全
   一致すること（正しさなので hard assert）。resident／host の所要
   時間比（ratio）は record_only（CI／開発機は専有ゲートでないため
   hard assert しない）。
4. **SGD／Adam 常駐経路の非後退**: 既存の `sgd_step_device(_tracked)`／
   `adam_step_device(_tracked)`・`DeviceParamStore::step()`／
   `step_adam`／`step_adamw` の既存 `#[ignore]` テストが引き続き
   pass すること。
5. **FAIL の扱い**: 実測が上記のいずれかで FAIL した場合、是正は
   行わず「FAIL のまま記録する」（`.claude/rules/coding-rust.md`
   「バックエンド間数値一致テストの許容誤差を単独で緩和しない」方針に
   従い、閾値・baseline を事後に緩めない）。
