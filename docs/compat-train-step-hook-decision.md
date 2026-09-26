# 学習 step カスタムフック（`train_step_fn`）設計判断記録

イシュー #2184（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」）。`docs/compat-grad-accumulation-decision.md`（#2180）・
`docs/autodiff-ema-decision.md`（#2179）と同型の記録。

## §0 結論

`fit()` 系（`Sequential::fit`／`fit_with_callbacks`／`fit_with_metrics`。
`crates/facade/src/compat/training.rs::run_fit`）が呼ぶ既定のバッチ
処理（`bind → forward → T::loss_for → backward → trainable_grads →
optimizer.step → apply_parameters`）を、ユーザー定義のフックへ丸ごと
差し替えられる内部機構（`CustomStepHook`。Keras `Model.train_step()`
相当）を実装した。フックは `(&Sequential, x_batch, y_batch, &mut
OptimizerState)` を受け取り `(loss: f32, updated: Option<Vec<Tensor
<f32>>>)` を返す。`fit_with_callbacks_named`／`run_fit` へ非公開引数
`custom_step: Option<&mut CustomStepHook<'_, T>>` を追加し、既存 3
入口（`fit`／`fit_with_callbacks`／`fit_with_metrics`）はいずれも
`None` で委譲するため既存挙動は変わらない（R3）。AMP
（`compile_with_amp`）・勾配累積（`accumulate_steps > 1`）との併用は
引数検査で `InvalidArgument`（fail-closed）に拒否する。

**facade 公開面は追加していない（承認待ち）**: イシュー #2184・親
#2131 のいずれにも所有者の承認コメントはない（着手前に `gh issue view
--json comments` で確認済み）。親 #2131 の「facade 公開面の拡張は
設計判断記録 → 承認 → 実装の 2 段」規則（先例 #2177・#2179・#2180・
#2171・#2173・#2176・#2178・#2198）に従い、`TrainStepFn`／
`TrainStepOptimizer`／`TrainStepOutput`・`Sequential::
fit_with_train_step` は追加せず保留する。現状は `#[cfg(test)] fn
Sequential::fit_custom_step_for_test` 経由でのみ到達でき、通常経路
（公開 API のみ）ではカスタム学習 step は使えない。

## §1 背景・要件（原文はイシュー本文。ここでは構造化要約のみを記す）

- R1: fit の 1 バッチ分の処理（forward・損失計算・backward・optimizer
  更新）を、ユーザー定義のコールバックへ差し替えられる
- R2: コールバックが受け取るのは「モデル・入力バッチ・正解バッチ・
  optimizer」の 4 つ。戻り値としてそのバッチの損失（スカラー）を返す
- R3: コールバックを指定しない場合、既存の fit ループの挙動は変わら
  ない（bit 完全一致）
- R4: 損失関数と optimizer step を自作した学習ループが、期待どおり
  収束することをテストで確認する

**契約（不変条件）**: tolerance・baseline・`Cargo.toml`／`Cargo.lock`・
ガードレール閾値・`docs/spec/` は変更していない。新規 `unsafe` は
入れていない。新規 `Op`／`BackendOps`／カーネルも追加していない
（ホスト側の制御フローだけの変更）。crates.io 出荷済み `fandhe-ai
=0.9.0` の公開 API は破壊していない。

**スコープ外**（§6 参照）: `test_step_fn`／`predict_step_fn`（推論専用
フック）・AMP との併用・勾配累積との併用・公開 API の新設。

## §2 設計

### 2.1 イシューの字面とのずれ（意図的）

| 項目 | イシューの字面 | 採用 | 理由 |
|---|---|---|---|
| 格納場所 | `FitConfig` のフィールド | `fit_with_callbacks_named`／`run_fit` への別引数（内部） | `FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`（`crates/facade/tests/api_surface.rs::fit_config_keeps_copy_eq_for_0_9_0_compat` が固定）を持つ。`Option<Box<dyn Fn..>>` を追加すると `Copy`／`Eq` が外れ 0.9.0 非破壊契約に反する。また正解バッチは `&Tensor<T>`（`T: FitTarget`）で generic だが `FitConfig` は非 generic のため、どのみち格納できない |
| モデルの受け取り方 | `&Sequential` | `&Sequential`（字面どおり） | フック内から `compile`／`fit`／`set_training` を再入呼び出しする事故を型で防ぐ |
| 戻り値 | `f32` | `(f32, Option<Vec<Tensor<f32>>>)` | `&Sequential` では `apply_parameters`（`&mut self` 必須）を呼べないため、更新後パラメータを返してもらい `run_fit` 側が適用する。`None` は AMP の非有限勾配スキップと同じ意味論で更新をスキップする |
| optimizer | `&mut Optimizer` | `&mut OptimizerState` | `Optimizer` は状態を持たない `Copy` の構成 enum。状態を持つのは非公開の `OptimizerState` |
| クロージャの trait | `Fn` | `FnMut` | 状態を持つ自作 optimizer・呼び出し回数の計測などを許すため |

### 2.2 内部配線

`CustomStepHook<'h, T>`（`crates/facade/src/compat/training.rs`。非公開
`type` エイリアス）:

```rust
type CustomStepHook<'h, T> = dyn FnMut(
        &Sequential,
        &Tensor<f32>,
        &Tensor<T>,
        &mut OptimizerState,
    ) -> Result<(f32, Option<Vec<Tensor<f32>>>), AutodiffError>
    + 'h;
```

`fit_with_callbacks_named`／`run_fit` へ `custom_step: Option<&mut
CustomStepHook<'_, T>>` を追加（既存 3 入口は `None` で委譲）。
`run_fit` のバッチループでは、`custom_step` が `Some(hook)` のとき
既存の `updated` 算出ブロック（AMP 分岐と非 AMP・勾配累積の分岐）を
丸ごと迂回し、`hook(&*self, &x_batch, &y_batch, &mut
compiled.optimizer)` を呼ぶ。返った `loss` は既存経路と同一のサンプル
数重み付き平均集計式（`weighted_sum += loss as f64 * n_batch as
f64`）へそのまま合流し、返った `Some(updated)` は既存の
`apply_parameters` 経路（パラメータ個数・shape 検査を含む）へ合流
する。`None` のときは既存コードを 1 行も変えずに通す（R3）。

テスト専用入口 `#[cfg(test)] fn Sequential::
fit_custom_step_for_test`（同ファイル）は `fit_with_callbacks_named`
へ `Some(hook)` を渡すだけの薄い委譲。`CustomStepHook` を型に含む
シグネチャのため `pub(crate)` にはできない（非公開型 `OptimizerState`
が公開経路のシグネチャに出ると `private_interfaces` lint に掛かる）。
同じ `training.rs` モジュール内の `#[cfg(test)] mod` はこの非公開
メソッドを問題なく呼べる。

### 2.3 fail-closed の組み合わせ検査

`fit_with_callbacks_named` の既存 `accumulate_steps` 検査の直後に置く:

- `custom_step.is_some() && compiled.amp.is_some()` → `InvalidArgument`
  （`GradScaler` の `scale_loss`／`unscale`／`update` は既定 step の
  一部で、フックが迂回すると scaler 状態が崩れるため）
- `custom_step.is_some() && config.accumulate_steps > 1` →
  `InvalidArgument`（更新の粒度をフック側が持つため、`accumulate_steps`
  のウィンドウ処理と両立しない）

いずれも `self.compiled = Some(compiled)` を書き戻してから返す（既存
の他引数検査と同じ fail-closed パターン）。

### 2.4 意味論

- 1 epoch の処理順は変えない: epoch 開始時の LR 同期 → バッチループ
  （フック）→ validation（`compiled.loss` を使う既存の
  `run_evaluate_with_metrics`）→ epoch 末の callbacks
- `compile()` は引き続き必須（optimizer 状態の供給源が compile だけ
  のため）
- フック使用時は学習側で `compiled.loss` を使わない。そのため
  `Loss` と `T` の組み合わせ不整合は validation を渡したときの評価
  経路でだけ検出される
- train モードは既存どおり fit の開始時に `set_training(true)`
  し、終了時に必ず元へ戻す（フックは `&Sequential` なのでモードを
  変えられない）
- フックが panic した場合: `catch_unwind` はしないので `compiled` が
  取り外されたままになり、モデルは未 compile 状態に落ちる（隠さず
  doc に明記済み）。`Err` を返した場合は既存の書き戻し経路で compiled
  状態を維持する

## §3 Keras／PyTorch との差分

Keras `Model.train_step(data)` は metrics の dict を返し、
`self.optimizer.apply_gradients` で変数を直接書き換える。本実装は
`&Sequential`（`&mut` ではない）を渡すため、フックはパラメータを直接
書き換えられず、更新後パラメータを戻り値として返し `run_fit` 側が
`apply_parameters` で適用する（§2.1 参照）。PyTorch は学習ループその
ものをユーザーが書くため専用 API を持たないが、本実装は Keras 風
`fit()` の内部でバッチ処理だけを差し替え可能にする。

## §4 数値一致と R4 の実測

- **T1（R3・配線の証明）**: 既定 step（`bind → forward →
  mse_loss_with(Reduction::Mean) → backward → trainable_grads →
  opt.step`）を手で再実装したフックで fit すると、`History.loss`・
  学習後パラメータが既定の `fit` と **bit 完全一致**する（SGD・AdamW
  の 2 構成・epochs=3・端数バッチを含む構成 N=7・batch=2。
  `crates/facade/src/compat/training.rs::train_step_tests::
  custom_step_reimplementing_default_matches_fit_bit_exact_{sgd,
  adamw}`）
- **T2（R4）**: 損失を自作（`Reduction::Sum`。既存の `Loss` enum に
  存在しない組み合わせ）し、optimizer を使わずホスト側で `p - lr * g`
  を計算する手書き SGD（`lr=0.02`）で更新するフックを、固定シードの
  合成回帰データ（N=16・batch=4・30 epoch）で実行した
  （`custom_loss_and_host_sgd_converges`）:
  - (a) 独立に組んだ手動ループ（facade 公開 `DataLoader` を直接使い、
    同じフック関数を同じ初期重み・同じバッチ順序〈shuffle なし〉で
    呼ぶ）と `History.loss`・学習後パラメータが **bit 完全一致**する
  - (b) 決定的シードのため、epoch 0 の loss より最終 epoch の loss が
    必ず小さいことを固定した（実測値は `first`／`last` をアサーション
    メッセージに含める形で記録。フレーキーな比率閾値は使わない）
- `(loss, None)` を返すフックはパラメータが bit 完全一致で不変
  （`custom_step_none_update_leaves_params_unchanged`）
- フックが `Err` を返すと fit が `Err` になり、`is_compiled()` は
  true・training モードは呼び出し前の値に復元される
  （`custom_step_error_propagates_and_keeps_compiled`）
- AMP・`accumulate_steps > 1` との併用は `InvalidArgument`
  （fail-closed。パラメータは不変・`is_compiled()` は維持。
  `custom_step_rejected_with_amp`・
  `custom_step_rejected_with_accumulate_steps_gt_one`）
- validation・callbacks（`EarlyStopping`）併用時も `val_loss` が
  epoch 数分埋まる（`custom_step_with_validation_and_callbacks`）
- フックが shape の違うパラメータを返すと `apply_parameters` 経由で
  `Err` になる（`custom_step_shape_mismatch_update_is_rejected`）

CUDA／Metal 固有の処理は追加していない（ホスト側の制御フローだけの
変更）ため、実機 parity の `#[ignore]` テストも `docs/perf/logs/` への
申し送りも発生しない。

## §5 承認事項（未承認のため保留）

facade（`compat`）公開面の新設は次が未承認。
`crates/facade/src/lib.rs::TrainStepHoldDoctestGuard`（正のプローブ
doctest）・`crates/facade/tests/api_surface.rs` の 4 テスト
（`train_step_hold_doctest_globs_all_pub_modules`・
`train_step_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_reexport_or_declare_train_step_items`・その自己
テスト）が機械的に固定する:

1. `pub type TrainStepFn<'h, T> = dyn FnMut(&Sequential, &Tensor<f32>,
   &Tensor<T>, &mut TrainStepOptimizer<'_>) -> Result<TrainStepOutput,
   AutodiffError> + 'h;`
2. `pub struct TrainStepOptimizer<'a>`: `step(&mut self, params: &[&
   Tensor<f32>], grads: &[&Tensor<f32>]) -> Result<Vec<Tensor<f32>>,
   AutodiffError>` と `lr(&self) -> f32`。`set_lr` は出さない（`history.lr`
   は epoch 開始時に記録され `LrSchedule` も epoch 開始時に LR を同期
   し直すため、フック内で LR を書き換えると記録と食い違う）
3. `#[non_exhaustive] pub struct TrainStepOutput`: loss と `Option<Vec
   <Tensor<f32>>>`、ビルダー `TrainStepOutput::new(loss)`／
   `with_updated(params)`
4. `Sequential::fit_with_train_step<T: FitTarget>(&mut self, x, y,
   config, validation, callbacks, metrics, train_step: &mut
   TrainStepFn<'_, T>) -> Result<History, AutodiffError>`
5. `FitConfig::train_step_fn` は `Copy`／`Eq` と generic `T` の問題
   （§2.1）のため採らない

**承認後の作業手順**: 公開型・入口の追加、テストを `#[cfg(test)] mod
train_step_tests`（`crates/facade/src/compat/training.rs`）から
`crates/facade/tests/compat_sequential_train_step.rs`（外部統合テスト
クレート）へ移設、`fit_custom_step_for_test` の削除、
`TrainStepHoldDoctestGuard` と対応する 4 テストの削除（正のガード・
facade テストへ置き換え）。

承認を得た日が来たら、上記を実施する。

## §6 スコープ外（out-of-scope-tracking）

- facade 公開面の新設（§5 参照）
- `test_step_fn`／`predict_step_fn`（推論専用のフック）
- AMP（`compile_with_amp`）とカスタム学習 step の併用（§2.3）
- 勾配累積（`accumulate_steps > 1`）とカスタム学習 step の併用（§2.3）
- `TrainStepOptimizer::set_lr` の公開（§5 の 2 参照）
- `DeviceParamStore` 常駐経路（`step_device_param_store` 系）でのカス
  タム学習 step

新規 Issue の起票はユーザー承認後に行う。

## §7 検証コマンドと結果

```
cargo fmt --all -- --check
cargo clippy -p fandhe-ai --all-targets --no-deps -- -D warnings
cargo test -p fandhe-ai --lib compat::training::train_step_tests
cargo test -p fandhe-ai --lib compat::training::accumulate_tests
cargo test -p fandhe-ai --test compat_sequential_fit \
  --test compat_sequential_callbacks --test compat_sequential_metrics \
  --test compat_sequential_fit_amp --test compat_sequential_fit_optimizers
cargo test -p fandhe-ai --test api_surface train_step
cargo test -p fandhe-ai --test api_surface \
  hold_doctest_probe_blocks_reference_every_glob_imported_item
cargo test -p fandhe-ai --test api_surface fit_config_keeps_copy_eq_for_0_9_0_compat
cargo test -p fandhe-ai --doc
cargo test -p fandhe-ai
cargo build --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

全て green（2026-09-26 実測。ローカル worktree）。`cargo clippy -p
fandhe-ai --all-targets`（`--no-deps` なし・`--all-features` 付き）で
このローカル環境固有に発生する `fandhe-ai-backend-cuda` の
`dead_code` lint は `docs/compat-grad-accumulation-decision.md` §7 の
注記と同じ既知事象（origin/main でも再現・本 PR の変更とは無関係）で
あり、`--no-deps` を付けると green になることを確認済み。
