# Adam／AdamW 常駐 step（`BackendOps::adam_step_device`）GPU 実装の
実機ランブック（イシュー #1959）

イシュー #1959 は `BackendOps::adam_step_device`／`adam_step_device_tracked`
（`AdamStepConfig`・`docs/device-resident-update-design.md`「Adam／AdamW
の常駐 step 結線」節）を新設し、**CPU 実装のみ**を提供した。CUDA／Metal
は既定実装（常に `BackendError::Unsupported` を返す fail-closed）のまま
であり、GPU カーネル自体が未実装のため、本ディレクトリに記録すべき実機
実測は本イシュー時点では存在しない。

以下は CUDA／Metal のカーネルを実装する**後続イシュー**（ユーザー承認を
得て別途起票する。`out-of-scope-tracking.md` 対象）向けに、その時点で
満たすべき事前登録判定規則・実行コマンド形・保存すべきログ一覧を
あらかじめ固定しておくものである（`metal-argext-1951/README.md`・
`metal-reduce-1895/README.md` と同型の「未実測のまま申し送る」形式を
踏襲）。

## 現状（本イシュー #1959 時点）

- CPU 実装（`crates/backend-cpu/src/ops.rs::CpuBackendOps::
  adam_step_device`）はホスト参照実装（`crate::nn::optim::{adam::Adam,
  adamw::AdamW}`）と **bit 完全一致**（`crates/backend-cpu/tests/
  adam_device_parity.rs`・`crates/facade/tests/
  device_param_store_adam_train.rs`。いずれも Linux で完走・pass 済み）。
- CUDA／Metal は `BackendOps::adam_step_device` の既定実装（
  `Unsupported`）のまま。`DeviceParamStore::step_adam`／`step_adamw` を
  CUDA／Metal バックエンドで呼ぶと、実 GPU の有無に関わらず常に
  `BackendError::Unsupported` を返す（fail-closed。実機なしでも
  Linux で確認可能——`docs/device-resident-update-design.md` 参照）。

## 確認（2026-09-18・Mac セッション）

本イシュー #1959 のマージ時点（base `a1c50f61`）において、Metal カーネルの
実装は存在せず（既定 `Unsupported` のままで、関連テストも新規追加されていない）。
よって 2026-09-18 の Apple M4 Max 実機での実測対象外。

## 後続イシューが実装すべきカーネル（未着手）

- `crates/backend-cuda/src/{adam.rs, kernels_adam.rs}`（新規）・
  `context_cache::cached_adam`・`ops.rs::CudaBackendOps::adam_step_device
  (_tracked)` の override。
- `crates/backend-metal/src/{adam.rs, shaders/adam.metal}`（新規）・
  `context_cache::cached_adam`・`ops.rs::MetalBackendOps::
  adam_step_device_impl(token)` + 2 件の override（`sgd.rs`／
  `MetalSgd` を鏡写しにする設計。実装計画参照）。

## 実行コマンド（後続イシューでの実施を想定）

```sh
# Linux（実装セッション）で完了させておく型検査・非実機テスト:
cargo test -p fandhe-ai-backend-cuda --lib adam
cargo test -p fandhe-ai-backend-metal --lib adam_model  # ホスト逐語モデルのテストを想定
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
cargo check -p fandhe-ai --tests --target aarch64-apple-darwin

# DGX Spark GB10（CUDA 実機）:
cargo test -p fandhe-ai-backend-cuda --release --test adam_device_real_device -- \
  --ignored --nocapture

# Apple Silicon（Metal 実機。--release 推奨）:
cargo test -p fandhe-ai-backend-metal --release --test adam_device_parity -- \
  --ignored --nocapture

# 既存 #[ignore] 群の非後退確認（新規カーネル追加による回帰がないこと。
# SGD 常駐経路〈sgd_device_parity.rs・device_param_store_backend_parity.rs〉
# を含む）:
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

## 保存すべきログ

- 上記各コマンドの標準出力（pass/fail・所要時間）
- `docs/real-hardware-verification-env.local.md.example` に準じた
  `env_info.txt`（内部ホスト名・チップ／GPU 世代を含めない範囲での
  実行環境情報）

## 事前登録判定規則

CUDA／Metal のカーネル実装（後続イシュー）に適用する受け入れ基準。
`.claude/rules/coding-rust.md`「バックエンド間数値一致は統一複合判定
『相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満』」に従う（CPU 実装が
達成した bit 完全一致を GPU にまで拡大要求しない——正規化統計・勾配の
長軸縮約と異なり、Adam の要素ごとの更新式自体はレーン間で縮約を伴わ
ないため REQ-2 統一複合判定がそのまま適用対象になる）。

1. **正しさ（REQ-2）**: GPU 実装によるデバイス常駐 1 step（複数 step
   累積を含む）の結果が、ホスト参照実装（`Adam::step`／`AdamW::step`）
   と統一複合判定で一致すること（`crates/backend-cpu/tests/
   adam_device_parity.rs` と同型の突合を GPU 側でも実施）。
2. **run-to-run bit 同一**: 同一入力・同一設定での GPU 実行を複数回
   繰り返し、出力が bit 単位で決定的であること。
3. **SGD 常駐経路の非後退**: 既存の `sgd_step_device(_tracked)`・
   `DeviceParamStore::step()`・CUDA Graph capture（`captured_
   segment_key`／`run_captured_sgd_step_segment`）関連の `#[ignore]`
   テスト群が新規カーネル追加により後退しないこと（`step_adam`／
   `step_adamw` は `step()` を一切変更しない設計〈`docs/
   device-resident-update-design.md`〉のため、構造的に非後退となる
   はずだが実機でも確認する）。
4. FAIL は是正せず記録のみとする（事後緩和なし）。
