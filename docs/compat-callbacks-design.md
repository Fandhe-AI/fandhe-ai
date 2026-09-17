# callbacks（EarlyStopping／ModelCheckpoint／LR scheduler 連携）の設計判断記録

イシュー #1763（親 #1618）。`docs/compat-api-scope.md` §1.2「Keras 風
`Sequential` の層追加と `compile()`／`fit()`／`evaluate()`／callbacks の
最小版」行のうち、`docs/compat-fit-evaluate-design.md`（#1761）が対象外
としていた callbacks（`EarlyStopping`／`ModelCheckpoint`）・
`validation_data`・LR スケジューラ連携を実装した。metrics（accuracy 等）
は引き続き対象外（§8）。

## 1. 目的・スコープ

`Sequential::fit`（既存契約不変）の拡張版として
`Sequential::fit_with_callbacks(x, y, config, validation, callbacks)` を
追加する。`validation=None`・`callbacks=&mut []` のとき `fit` と完全に
同一の演算列・戻り値になる（`fit_with_callbacks_empty_matches_fit_bit_exact`
で検証）。

**新規 `Op`／`BackendOps`／`Var` メソッド／VJP／カーネルは一切追加
しない**——callbacks はホスト側の状態機械のみで構成される（REQ-9
「互換 API 層は自作コアの上の薄いラッパーに徹する」方針の継承）。

対象内:

- `EarlyStopping`（監視指標が `patience` epoch 連続で改善しなければ
  学習を打ち切る）
- `ModelCheckpoint`（改善時、または毎 epoch、モデルの `state_dict()`
  スナップショットを in-memory で保持する）
- `LrSchedule`（既存の `LrScheduler`／`ReduceLrOnPlateau` を optimizer
  へ結線する。epoch 単位のみ）
- `validation`（検証データ。callback の `Monitor::ValLoss` の入力）

対象外は §8 を参照。

## 2. 学習率更新 API（`Sgd`／`AdamW`／`Adam::set_lr`）

`docs/compat-fit-evaluate-design.md`（#1761）は「`fandhe_ai::optim` の
optimizer は学習率更新 API を持たないため LR スケジューラ結線が
できない」ことを対象外の理由として明記していた。本イシューでまず
この欠落を埋める: `Sgd::set_lr`（`crates/autodiff/src/optim/sgd.rs`）・
`AdamW::set_lr`（`crates/autodiff/src/nn/optim/adamw.rs`）・
`Adam::set_lr`（`crates/autodiff/src/nn/optim/adam.rs`）を追加した。

- **意味論**: PyTorch の `param_group["lr"] = new_lr` と同じ——学習率
  のみを書き換え、momentum バッファ・moment 推定値（`m`／`v`）・
  `step_count`／`beta*_pow_t` 等の内部状態には一切触れない（次の
  `step()` から新しい `lr` が適用される）
- **検証**: 各 optimizer の `new`（構築時）と同一基準（有限かつ非負）
  で `new_lr` を検査し、失敗時は既存の `lr` を変更しない
  （`AutodiffError::InvalidArgument`。fail-closed）
- `Sgd` は `config()` アクセサも新設した（`AdamW`／`Adam` には既に
  存在した非対称を解消）
- 結線自体（epoch ごとに `lr_at` の返り値で `set_lr` を呼ぶループ）は
  本イシューの facade 側（`compat::Sequential::fit_with_callbacks`）の
  責務であり、autodiff 側はあくまで「optimizer に学習率を可変化する
  入口を用意する」ところまでを担う

## 3. 公開 API（`fandhe_ai::compat`。すべて `crates/facade/src/compat/callbacks.rs`）

```rust
pub enum Monitor { Loss, ValLoss }               // #[non_exhaustive]。既定は文脈依存
pub enum MonitorMode { Min, Max }                // #[non_exhaustive]。既定 Min

pub struct EarlyStopping { .. }
impl EarlyStopping {
    pub fn new(patience: usize) -> Self;                          // monitor=ValLoss, mode=Min, min_delta=0.0, restore_best_weights=false
    pub fn monitor(self, m: Monitor) -> Self;
    pub fn mode(self, m: MonitorMode) -> Self;
    pub fn min_delta(self, d: f32) -> Result<Self, AutodiffError>; // 有限かつ非負以外は InvalidArgument
    pub fn restore_best_weights(self, on: bool) -> Self;
    pub fn stopped_epoch(&self) -> Option<usize>;
    pub fn best_value(&self) -> Option<f32>;
    pub fn best_epoch(&self) -> Option<usize>;
}

pub struct ModelCheckpoint { .. }
impl ModelCheckpoint {
    pub fn new() -> Self;                                         // monitor=ValLoss, mode=Min, save_best_only=true
    pub fn monitor(self, m: Monitor) -> Self;
    pub fn mode(self, m: MonitorMode) -> Self;
    pub fn save_best_only(self, on: bool) -> Self;
    pub fn best_state_dict(&self) -> Option<&HashMap<String, Tensor<f32>>>;
    pub fn take_best_state_dict(&mut self) -> Option<HashMap<String, Tensor<f32>>>;
    pub fn best_value(&self) -> Option<f32>;
    pub fn best_epoch(&self) -> Option<usize>;
}
impl Default for ModelCheckpoint

pub struct LrSchedule { .. }
impl LrSchedule {
    pub fn per_epoch(s: impl LrScheduler + 'static) -> Self;
    pub fn plateau(s: ReduceLrOnPlateau) -> Self;                 // monitor 既定 ValLoss
    pub fn plateau_with_monitor(s: ReduceLrOnPlateau, m: Monitor) -> Self;
    pub fn epoch(&self) -> usize;
}

#[non_exhaustive]
pub enum Callback { EarlyStopping(EarlyStopping), ModelCheckpoint(ModelCheckpoint), LrSchedule(LrSchedule) }
```

```rust
// crates/facade/src/compat/training.rs（変更）
pub struct History { pub loss: Vec<f32>, pub val_loss: Vec<f32>, pub lr: Vec<f32> } // #[non_exhaustive]
impl Sequential {
    pub fn fit<T: FitTarget>(&mut self, x, y, config) -> Result<History, _>;   // = fit_with_callbacks(x, y, config, None, &mut [])。既存契約不変
    pub fn fit_with_callbacks<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>, y: &Tensor<T>,
        config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback],
    ) -> Result<History, AutodiffError>;
}
```

## 4. 設計判断の要点

### 4.1 `Callback` は閉じた `#[non_exhaustive] enum`（trait object によるユーザー拡張は対象外）

理由:

1. `&mut Sequential` を callback へ渡すと、`fit_with_callbacks` 内部の
   借用構造（`compiled` の一時取り外し・`SequentialVars::bind` の
   `&self` 借用）と衝突する
2. REQ-9「薄いラッパーに徹する」・REQ-12（`BackendOps` 注入 API を
   設けない）の精神——公開面を最小化する

ユーザー定義 callback（trait object による拡張点）は切り出し候補
（§8）。

### 4.2 `callbacks` は `FitConfig` に含めない

`FitConfig` は `Copy + Eq`（`crates/facade/tests/api_surface.rs::
fit_types_are_reachable_via_facade_only` が `assert_eq!` で固定）で
あり、状態保持型の callback（`EarlyStopping`・`ModelCheckpoint`・
`ReduceLrOnPlateau`）はこの制約に馴染まない。`&mut [Callback]` として
呼び出し元が所有し、`fit_with_callbacks` 後に `stopped_epoch()`／
`best_state_dict()` 等を読める（Keras の callback オブジェクト参照と
同じ設計）。

### 4.3 `ModelCheckpoint` はファイルへ書かない（in-memory スナップショットのみ）

safetensors save／load 自体は **#2019 で facade 公開済み**
（`fandhe_ai::interop::safetensors`。`docs/facade-safetensors-exposure-
decision.md` §11）だが、`ModelCheckpoint` からそれを呼ぶ薄いラッパー
結線自体は本 issue（#1763）のスコープ外のまま維持する（in-memory
スナップショット限定の設計は変更しない。ファイル保存版は引き続き
切り出し候補〈§8〉）。永続化はユーザーが `best_state_dict()`／
`take_best_state_dict()` で取り出した `HashMap<String, Tensor<f32>>`
を自前で扱い、復元は既存の `Sequential::load_state_dict`（アトミック。
イシュー #1752）で行う。ファイル保存版は切り出し候補（§8）。

### 4.4 改善判定と NaN 契約（`best: Option<f32>`）

`MonitorMode::Min`／`Max` の改善判定は
`fandhe_ai_autodiff::nn::optim::reduce_lr_on_plateau::ReduceLrOnPlateau::
is_better` と同型の比較式（`value < best - min_delta`／
`value > best + min_delta`）を使う。`f32` の順序比較演算子は NaN に
対して必ず `false` を返す言語仕様のため、**NaN は決して改善として
扱われない**契約が明示的な `is_nan()` 分岐なしに自然に成立する。

`best` は内部で `Option<f32>`（未観測は `None`）として保持し、比較の
たびに `best.unwrap_or(mode.initial_best())`（`Min` なら `+INF`・`Max`
なら `-INF`）を実効の比較対象にする——固定値を `new()` 時点で
`self.best` へ先に書き込む実装だと、`mode()` ビルダーを `new()` の
後・観測前に呼んだ場合に誤った初期値が残ってしまう（`Min` 用の
`+INF` のまま `Max` へ切り替えると最初の観測が絶対に改善にならない
バグを生む）。`Option` 方式はビルダーの呼び出し順序に依存しない。

初回観測が NaN の場合でも `NaN < +INF` は `false` となり NaN が
`best` になることはない（NaN が続く限り非改善カウンタが増加し続け、
いずれ `patience` で停止する。`EarlyStopping::patience` の停止条件は
`wait >= patience`——patience=0 は改善 epoch の直後には停止せず、
最初の非改善 epoch で即座に停止する契約。統合テスト
`early_stopping_patience_zero_stops_at_first_non_improving_epoch` で
固定）。

### 4.5 LR 同期のタイミング

optimizer への学習率書き込みは **epoch 開始時のみ**行う（`PerEpoch`
なら `scheduler.lr_at(next_epoch)`、`Plateau` なら
`sched.current_lr()`）。epoch 末の `LrSchedule::advance` は（`Plateau`
の場合）`ReduceLrOnPlateau::step` を呼び内部状態を進めるだけで、
optimizer への即時書き戻しは行わない——次回の epoch 開始同期が必ず
直後に続くため冗長（2 回の `fit_with_callbacks` 呼び出しをまたぐ
場合でも、`sched` 自身の内部状態が呼び出しをまたいで継続するため
次の fit 呼び出しの epoch 開始同期で正しい値が反映される。
統合テスト `lr_schedule_epoch_counter_persists_across_fit_calls`）。

`history.lr[e]` は「epoch `e` の全バッチが実際に使った学習率」で
あり、`LrSchedule` が 1 つもない場合も常に記録する。

### 4.6 epoch 番号の数え方（callback ごとに異なる）

- `EarlyStopping` は `fit_with_callbacks` 呼び出しごとに内部状態
  （`best`／`wait`／`stopped_epoch` 等）を**リセット**する（Keras
  `on_train_begin` と同じ）。ここでの「epoch」は**その fit 呼び出し内
  のローカル epoch 番号**（`History::loss`／`val_loss` の添字と同じ）
- `ModelCheckpoint`・`LrSchedule` は複数回の `fit_with_callbacks`
  呼び出しをまたいで状態を継続する（前者は「これまでの最良」を、
  後者は内部の `next_epoch` カウンタを維持する。optimizer 状態が
  `fit` 呼び出しをまたいで継続する既存契約
  `fit_twice_equals_fit_once_with_double_epochs` と整合させるため）。
  ここでの「epoch」は**その callback インスタンスが観測した epoch 末
  呼び出しの通算回数**

両者の定義は独立であり、`History` の添字（常に fit 呼び出しローカル）
とは別物であることに注意する（`crates/facade/src/compat/callbacks.rs`
モジュール冒頭 doc に明記）。

### 4.7 `restore_best_weights` の意味論

`callbacks` 中に `restore_best_weights(true)` の `EarlyStopping` が
あり、かつ best スナップショットが記録されていれば、
`fit_with_callbacks` が返る直前（学習打ち切り・完走いずれの終了経路
でも）に `Sequential::load_state_dict` でそこへ復元する（Keras 3 の
`restore_best_weights=True` と同じ意味論）。スナップショット自体は
改善 epoch でのみ遅延評価クロージャ経由で取得する（`state_dict()` の
clone コストを改善 epoch のみへ限定する）。

### 4.8 callback 処理順序

epoch 末の callback 処理は `callbacks` の**スライス順**で行う。
いずれかが学習打ち切りを要求しても、当該 epoch の他の callback は
必ず処理してから fit ループを抜ける（同一 epoch 内の
`ModelCheckpoint`／`LrSchedule` の処理を取りこぼさない）。

### 4.9 引数検査（`Monitor::ValLoss` × `validation=None`）

`callbacks` のいずれかが `Monitor::ValLoss`（`EarlyStopping`／
`ModelCheckpoint`／`LrSchedule::plateau` の既定値）を監視するのに
`validation.is_none()` の場合、train／eval モード変更前・`compiled`
取り外し状態のまま `AutodiffError::InvalidArgument` を返す
（fail-closed。既存 `epochs == 0` の早期検査と同型）。

## 5. `Sequential::fit_with_callbacks` の内部フロー（概略）

```text
fit_with_callbacks:
  compiled = self.compiled.take()?
  epochs==0 → 書き戻し → Err
  validation.is_none() && いずれかの callback が ValLoss を監視 → 書き戻し → Err
  EarlyStopping を reset_for_fit()
  prev = training(); set_training(true)
  result = run_fit(&mut compiled, x, y, config, validation, callbacks)
  set_training(prev); self.compiled = Some(compiled); result

run_fit:
  for epoch_local in 0..epochs:
    for cb in LrSchedule: compiled.optimizer.set_lr(cb.lr_for_epoch_begin())?
    history.lr.push(compiled.optimizer.lr())
    （既存バッチループ）
    history.loss.push(...)
    if validation: history.val_loss.push(run_evaluate(...)?)
    stop = false
    for cb in callbacks（スライス順）:
      ModelCheckpoint => observe(value, self)
      LrSchedule      => advance(monitor_value)?
      EarlyStopping   => if observe(value, epoch_local, || self.state_dict()) { stop = true }
    if stop: break
  for cb in EarlyStopping: if let Some(state) = take_restore_state() { self.load_state_dict(state)? }
  Ok(history)
```

## 6. 正しさの検証

`crates/facade/tests/compat_sequential_callbacks.rs`（`fandhe_ai` のみ
import・固定シード・実機非依存）:

- `fit_with_callbacks_empty_matches_fit_bit_exact`: `fit` と
  `fit_with_callbacks(.., None, &mut [])` の演算列・戻り値の完全一致
- `history_lr_records_optimizer_lr_without_callbacks`
- `early_stopping_stops_after_patience_with_constant_loss`／
  `early_stopping_patience_zero_stops_at_first_non_improving_epoch`／
  `early_stopping_state_resets_at_fit_start`／
  `early_stopping_min_delta_rejects_nan_and_negative`／
  `early_stopping_restore_best_weights_bit_exact`（大きな lr で発散
  させ epoch 0 が best のままであることを前提検査で確認したうえで
  双子モデルの epoch 1 時点パラメータと bit 完全一致を検証）
- `model_checkpoint_keeps_best_state_dict`（best 値の callback インス
  タンスをまたいだ継続も検証）
- `lr_schedule_per_epoch_matches_manual_set_lr_loop`（`history.lr` が
  `StepLr::lr_at` と bit 一致し、最終パラメータが手動 `set_lr` ループ
  と bit 一致）
- `lr_schedule_epoch_counter_persists_across_fit_calls`
- `lr_schedule_plateau_reduces_lr_and_history_reflects_it`（`patience=0`・
  `threshold` を巨大値にして初回観測のみ改善という決定的な条件を
  作り、手動ループとの bit 完全一致を検証）
- `validation_val_loss_matches_evaluate_bit_exact`
- `monitor_val_loss_without_validation_is_rejected`
- `callbacks_error_keeps_model_compiled_and_restores_mode`（ユーザー
  定義 `LrScheduler` が epoch 1 で NaN を返すケースで `set_lr` が
  `InvalidArgument` を返し、`is_compiled()`／train-eval モードが
  復元されることを固定）
- `fit_with_callbacks_alongside_shuffle_is_reproducible_under_manual_seed`

autodiff 側単体テスト（`crates/autodiff/src/optim/sgd.rs`・
`crates/autodiff/src/nn/optim/{adamw,adam}.rs`）: `set_lr` の拒否
（NaN／±inf／負値）・momentum バッファ／moment 推定値／`step_count`
が `set_lr` 前後で不変であることを検証。

`crates/facade/tests/api_surface.rs::fit_types_are_reachable_via_facade_only`
を拡張し、新規型（`Callback`／`EarlyStopping`／`ModelCheckpoint`／
`LrSchedule`／`Monitor`／`MonitorMode`）の構築・`Sgd::set_lr` の
facade 経由到達性を固定した。`fandhe_ai::optim`（`optim.rs`）は純
再エクスポート契約（`optim_module_reexports_exactly_expected_surface`／
`optim_module_is_pure_reexport`）を変更していない——`set_lr` は既存
再エクスポート型（`Sgd`／`AdamW`／`Adam`）へのメソッド追加のため、
`optim.rs` の `pub use` 行自体に変更はない。

## 7. セキュリティ考慮（OWASP Top 10）

- **A03 インジェクション／入力検証**: `patience`（`usize`）・
  `min_delta`（有限・非負）・`set_lr`（有限・非負）を fail-closed で
  検査。NaN は決して「改善」と判定しない。`Monitor::ValLoss` ×
  `validation` なしは早期 `InvalidArgument`。`History` の `Vec` は
  `try_reserve_exact` で capacity overflow panic を回避
  （`val_loss`／`lr` も `loss` と同じ方式）。本番経路に `unwrap`／
  `expect` を置いていない
- **A08 ソフトウェア・データ整合性**: パラメータ復元は既存のアトミック
  `load_state_dict`（two-pass・ロールバック）のみを経由し、部分適用
  状態を残さない。`fit_with_callbacks` 失敗時も `compiled` を必ず
  書き戻しモデルを「未 compile」状態へ落とさない（既存契約踏襲）
- **A06 脆弱コンポーネント**: 依存クレートの追加・更新なし
- **A04／設計**: ファイル I/O・シェル呼び出し・ネットワークを一切
  導入しない（`ModelCheckpoint` は in-memory）。`unsafe` なし。
  REQ-12 に抵触する引数なし・`api_surface.rs` の既存 guard（compat
  pub fn が生 `fandhe_ai_autodiff::Tape` を取らない・`onnx-interop`
  非依存）を維持

## 8. 対象外・切り出し候補

- ユーザー定義 callback（trait object による拡張点）
- `ModelCheckpoint` のファイル保存（safetensors。facade 公開自体は
  #2019 で完了済み〈`fandhe_ai::interop::safetensors`〉だが、
  `ModelCheckpoint` からの結線は未実装のまま）
- metrics（accuracy 等）・`Monitor` の metrics 拡張
- `DataLoader` を直接受ける `fit` 入口・追加 `Loss` variant・
  デバイス常駐学習（`DeviceParamStore`）／GPU `Tape`／AMP／gradient
  clipping との結線
- `TerminateOnNaN` 相当（現状は `ReduceLrOnPlateau::step` の非有限
  拒否で fail-closed に停止する）
- OneCycle 等の**バッチ単位**スケジューリング（本イシューは epoch
  単位のみ）

これらは自動運転モードでの実装のため Issue 未起票のまま記録する
（`.claude/rules/out-of-scope-tracking.md`）。
