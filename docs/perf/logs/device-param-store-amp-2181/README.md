# AMP（GradScaler）の DeviceParamStore 常駐 step 結線 実機ランブック（イシュー #2181）

イシュー #2181 は `DeviceParamStore::step_amp`／`step_adam_amp`／
`step_adamw_amp`（`crates/autodiff/src/optim/device_store/amp.rs`）と
facade `Tape::step_device_param_store_amp`／`_adam_amp`／`_adamw_amp`
（イシュー #1959「Adam／AdamW の常駐 step 結線」・#2175「RmsProp／
Adagrad／LAMB の常駐 step 結線」と同型の薄い委譲）を新設した。AMP 自体
は新規 `BackendOps`／`MemoryOps` trait メソッドを追加しないホスト計算
フォールバック（`DeviceParamStore::param_grads_to_host` で一度ホストへ
実体化してから unscale・非有限検出する）のため、**AMP 結線それ自体に
GPU カーネルはない**。したがって本ディレクトリの実機実測は「AMP 結線が
CUDA／Metal 上でも既存 optimizer カーネルと同様に動作すること」の確認
であり、`adam-device-step-1959/README.md`・`optimizer-device-step-2175/
README.md` と同型の「未実測のまま申し送る」形式を踏襲する。

## 現状（本イシュー #2181 時点）

- CPU（`fandhe_ai::tape()` 既定バックエンド）では、`crates/facade/tests/
  device_param_store_amp_train.rs` の 4 テストが Linux で完走・pass 済み
  （`GradScalerConfig { init_scale: 1.0, growth_interval: u64::MAX, .. }`
  固定時の非 AMP 経路との per-step bit 完全一致〈SGD／Adam／AdamW〉・
  overflow による skip／backoff・pending 消費）。
- `step_device_param_store_amp`（SGD）は `sgd_step_device`（CUDA／Metal
  とも実装済み）を経由するため、AMP 結線自体は CUDA／Metal でも動作
  する見込みだが実機未確認。
- `step_device_param_store_adam_amp`／`_adamw_amp` は既存
  `step_adam`／`step_adamw` を経由するが、`adam_step_device` 自体が
  CUDA／Metal では既定実装（`Unsupported`）のまま（#1959 コメント・
  `adam-device-step-1959/README.md` 参照）という**AMP とは独立の既存
  制約**があり、CUDA／Metal 上で `step_device_param_store_adam_amp`
  等を呼ぶと（AMP の unscale・非有限検出を通過した後で）
  `BackendError::Unsupported` になる。この制約は AMP 結線が解消する
  スコープではない。

## 後続イシューが確認すべきこと

1. **SGD AMP（CUDA／Metal）**: `Sequential::forward_resident` →
   `GradScaler::scale_loss` → `tape.backward_device_param_store` →
   `tape.step_device_param_store_amp` のループを CUDA（DGX Spark
   GB10）・Metal（Apple Silicon 実機）で数 step 回し、CPU 参照実装
   （`crates/facade/tests/device_param_store_amp_train.rs` と同型の
   `GradScalerConfig` scale-one 固定・非 AMP 経路突合）と REQ-2 統一
   複合判定（相対誤差 1e-3 未満または絶対誤差 1e-5 未満）で比較する。
2. **Adam／AdamW AMP（CUDA／Metal）**: 上記「後続イシューで CUDA／
   Metal 専用 `adam_step_device` カーネルが実装された場合」に限り、
   同様の scale-one 固定突合を追加する（現状は `Unsupported` のため
   計測不能）。
3. **overflow skip の実機確認**: CUDA／Metal の `MemoryOps::download`
   往復（AMP のホスト計算フォールバック経路）で非有限値が正しく
   ホストへ転送され `unscale_grads` が検出できることを確認する
   （CPU では `crates/facade/tests/device_param_store_amp_train.rs::
   step_amp_skip_on_overflow_leaves_params_unchanged_and_backoffs`
   で確認済み）。

## 判定規則（後続イシューが実施する際の事前登録）

- 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `.claude/rules/coding-rust.md`）を CPU 参照実装との突合に適用する。
- tolerance・baseline 自体の変更はユーザー承認必須（変更しない）。
- 5 回計測の中央値を性能計測に用いる場合は
  `.claude/rules/coding-rust.md`「テスト・ベンチ」節に従う。

## 実行コマンド（申し送り時点の想定）

```bash
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test device_param_store_amp_train -- --include-ignored

# Metal（Apple Silicon）
cargo test -p fandhe-ai --test device_param_store_amp_train -- --include-ignored
```

（現状 `device_param_store_amp_train.rs` に `#[ignore]` テストはなく、
CUDA／Metal 専用テストの追加は後続イシューの作業に含める。）
