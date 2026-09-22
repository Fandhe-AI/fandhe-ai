# Keras 風 `compile()`／`fit()`／`evaluate()` 最小版の設計判断記録

イシュー #1761（親 #1618）。`docs/compat-api-scope.md` §1.2「Keras 風
`Sequential` の層追加と `compile()`／`fit()`／`evaluate()`／callbacks の
最小版」行のうち、`compile()`／`fit()`／`evaluate()` 最小版を実装した。
`add_*` 拡張は兄弟イシュー #1760。callbacks（`EarlyStopping`／
`ModelCheckpoint`／LR スケジューラ連携）・`validation_data` は
兄弟イシュー #1763 で実装済み（設計判断は
`docs/compat-callbacks-design.md` を参照）。

## 1. 目的・スコープ

`bind → forward → loss → backward → trainable_grads → optimizer.step →
apply_parameters` という定型（`crates/facade/src/compat/sequential.rs`
モジュール doc の doctest・`crates/facade/tests/compat_sequential_train.rs`）
を `Sequential::{compile, fit, evaluate}` の 3 メソッドへ畳み込む。
**新規 `Op`／`BackendOps` メソッド／VJP／カーネルは一切追加しない**——
既存公開 API（`Sequential::bind` 等・`fandhe_ai::optim` の
optimizer・`fandhe_ai::data::DataLoader`・`Var::mse_loss`／
`cross_entropy_loss`）の合成のみで実装した（REQ-9「薄いラッパーに
徹する」）。

対象外: `DataLoader` を直接受ける `fit` 入口・BCE／Huber 等の追加
`Loss` variant・デバイス常駐学習（`DeviceParamStore`）・GPU `Tape`
指定・AMP・gradient clipping。callbacks・`validation_data`・LR
スケジューラ連携は #1763 で実装済み（`docs/compat-callbacks-design.md`）。
metrics（accuracy・precision・recall・F1・confusion matrix）は
`Sequential::fit_with_metrics`（イシュー #2072・親 #2059）で実装済み
（`docs/compat-metrics-design.md`）。上記の残りはいずれも #1763 以降へ
引き継ぐ。

## 2. 公開 API

```rust
pub enum Loss { Mse, CrossEntropy }               // #[non_exhaustive]
pub enum Optimizer { Sgd(SgdConfig), AdamW(AdamWConfig), Adam(AdamConfig) } // #[non_exhaustive]
pub struct FitConfig { .. }                        // epochs/batch_size/shuffle/drop_last
pub struct History { pub loss: Vec<f32> }           // #[non_exhaustive]
pub trait FitTarget: Element + private::Sealed { .. } // f32/i32 のみ実装

impl Sequential {
    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss) -> Result<(), AutodiffError>;
    pub fn is_compiled(&self) -> bool;
    pub fn fit<T: FitTarget>(&mut self, x: &Tensor<f32>, y: &Tensor<T>, config: FitConfig) -> Result<History, AutodiffError>;
    pub fn evaluate<T: FitTarget>(&mut self, x: &Tensor<f32>, y: &Tensor<T>, batch_size: usize) -> Result<f32, AutodiffError>;
}
```

すべて `crates/facade/src/compat/training.rs`（新規ファイル）に実装し、
`compat/mod.rs` から `Loss`／`Optimizer`／`FitConfig`／`FitTarget`／
`History` を再エクスポートする。`Sequential` 自体は `compat/
sequential.rs` に既存のまま置き、`compiled: Option<training::Compiled>`
フィールド（`pub(super)`）のみを追加した。

## 3. 設計判断

### 3.1 loss × target dtype の分岐（sealed `FitTarget`）

`Loss::Mse` は `Tensor<f32>` target・`Loss::CrossEntropy` は
`Tensor<i32>` target（クラス添字）を要求するが、`compile()` の時点では
`fit`／`evaluate` に渡される target の型（ジェネリクス `T`）がまだ
分からない。逆に `fit`／`evaluate` はジェネリクス `T: FitTarget` として
呼ばれるため、`loss` と `T` の対応は実行時にしか判定できない。

`fandhe_ai_tensor_core::CastElement` と同型の sealed trait パターン
（`private::Sealed` を `f32`／`i32` にのみ実装）を採用し、`FitTarget::
loss_for(loss, tape, pred, target_batch) -> Result<Var, AutodiffError>`
という `#[doc(hidden)]` メソッドへ dispatch する。組み合わせが不整合
（`Loss::Mse` × `Tensor<i32>` 等）なら `AutodiffError::InvalidArgument`
を返す（fail-closed）。

target の `Var` 化は [`Tape::var_no_grad`](compat-api-scope.md)
（イシュー #1748）を使う——target へ勾配は不要であり、`bind` 済みの
学習可能パラメータの葉ノードと混同しないためである。

### 3.2 借用衝突の解決（`compiled` の take/restore）

`fit` 内で `Sequential::bind(&self)` の借用が生きている間は
`self.compiled` を `&mut` で触れない（`SequentialVars` が `&'m
Sequential` を保持するため）。`self.compiled.take()` で一旦取り外し、
別変数 `compiled: Compiled` として `run_fit(&mut self, compiled: &mut
Compiled, ..)` へ渡すことで、`self.bind(&tape)`（`&self`）と
`compiled.optimizer.step`（`&mut compiled`、`self` とは独立の借用）を
同時に生かす。`run_fit` の結果を問わず必ず `self.compiled =
Some(compiled)` へ書き戻す（fail-closed: 失敗した `fit` の後もモデルを
「未 compile」状態へ落とさない。optimizer 状態は成功した step の分だけ
進んだまま保持される——Keras の途中失敗後の再 `fit` 継続と同じ挙動）。

### 3.3 train／eval モードの復元

`fit` は呼び出し前のモード（`Sequential::training()`）を記憶し学習中は
train モードへ切り替える（`add_dropout` 入りモデルのマスク適用のため）。
`evaluate` は逆に eval モードへ切り替える（決定的な評価のため）。いずれも
成功・失敗いずれの経路でも呼び出し前のモードへ復元する。

処理順序は「未 compile 検査 → 引数検査（`epochs == 0` 等） →
モード切替 → 本体 → モード復元」。早期 `Err`（未 compile・`epochs ==
0`）はモード変更前に返るため復元不要——モード変更が必要なのは本体の
実行中に発生した `Err` のみ。

### 3.4 サンプル数重み付き平均（Keras `History.history["loss"]` 相当）

各バッチの loss を `f64` で `Σ(batch_loss * batch_n) / Σ batch_n` として
集計し、epoch の最後に 1 回だけ `f32` へ downcast する（Keras の
`fit`／`evaluate` の集計方式と同じ）。`batch_size == N`（全件 1
バッチ）のとき、`n * l / n` は `f64` で exact なため単一バッチの直接
計算と bit 完全一致する（`evaluate_full_batch_matches_direct_loss_
bit_exact` で検証）。

### 3.5 `FitConfig::shuffle` の既定値（`false`）

Keras `fit` の既定 `shuffle=True` とは異なり、`DataLoaderConfig::new`
と同じ「既定でグローバル RNG を消費しない」側へ揃えた——安全側・
既存慣習との整合を優先する（自動運転モードでの判断）。`shuffle` は
バッチの行順を変えるため縮約順序が変わり手動ループと bit 一致しない
——bit 一致テスト（`fit_matches_manual_loop_bit_exact` 等）はすべて
`shuffle(false)` で行う。Keras 相当にしたい呼び出し元は
`.shuffle(true)` を明示する。

### 3.6 `DataError` → `AutodiffError` の変換

`fandhe_ai_tensor_core::data::DataError` から `AutodiffError` への
`From` 実装は存在しないため（既存ギャップ。`data_loader.rs` 冒頭
コメント参照）、`TensorDataset::new`／`DataLoader::new`／バッチ取得
それぞれで明示的に `.map_err(|e| AutodiffError::InvalidArgument(format!("Sequential::fit: {e}")))`
する。

### 3.7 対象外事項

3.1 冒頭の「目的・スコープ」節を参照。callbacks・`validation_data`・
LR スケジューラ連携は #1763 で実装済み（`docs/compat-callbacks-design.md`）。
metrics は #2072 で実装済み（`docs/compat-metrics-design.md`）。
`DataLoader` 直接入力は引き続き対象外のまま。

## 4. 正しさの検証

- `crates/facade/tests/compat_sequential_fit.rs`（統合テスト）:
  - `fit_sgd_mse_converges`／`fit_cross_entropy_with_i32_targets_converges`:
    収束判定（新設の収束判定であり既存 tolerance の緩和ではない）
  - `fit_matches_manual_loop_bit_exact`: `shuffle(false)`・
    `batch_size = N` での `fit` と手動ループ（`bind → forward →
    mse_loss → backward → trainable_grads → Sgd::step →
    apply_parameters`）の loss 系列・最終パラメータの bit 完全一致
  - `fit_twice_equals_fit_once_with_double_epochs`: optimizer 状態が
    `fit` 呼び出しをまたいで継続する契約
  - `fit_minibatch_shuffle_is_reproducible_under_manual_seed`:
    `manual_seed` 固定下での再現性
  - `recompile_replaces_optimizer_state`: 再 `compile()` が momentum
    バッファ等の optimizer 状態をリセットする契約（momentum 有り
    `SgdConfig` で、再 compile 後の `fit(1)` が velocity=None の素の
    `Sgd::step` と bit 一致することで検証）
  - `fit_and_evaluate_reject_uncompiled_model`／`fit_rejects_zero_
    epochs`／`fit_rejects_sample_count_mismatch`／`fit_rejects_zero_
    batch_size`（`evaluate` も同型）: fail-closed 検査・Err 後も
    `is_compiled()` が維持されること
  - `evaluate_full_batch_matches_direct_loss_bit_exact`／
    `evaluate_minibatch_matches_weighted_mean_of_manual_batches`:
    §3.4 の集計契約
  - `fit_restores_training_mode_and_evaluate_is_deterministic_with_
    dropout`: §3.3 のモード復元契約・Dropout 入りモデルでの決定的評価
- `crates/facade/tests/api_surface.rs::fit_types_are_reachable_via_
  facade_only`: 新規公開型が `fandhe_ai` のみの import で構築・型境界
  として使えることの固定

## 4. AMP（`compile_with_amp`）統合（イシュー #1961・親 #1958）

`Sequential::compile_with_amp`（低精度 forward・#1960 + `GradScaler`・
#1721）は `compile`／`fit` と独立の追加メソッドとして実装した（既存
`compile`／`fit`／`evaluate` のシグネチャ・戻り値・演算列は無変更）。
設計判断の詳細は `docs/autodiff-low-precision-linear-design.md` §7 を
正とし、本節では `compile`／`fit` 側の設計との関係のみを記す。

- **AMP 状態の置き場所**: `GradScaler` は optimizer と同じく `fit`
  呼び出しをまたいで継続すべき状態（`fit(1)+fit(1)==fit(2)` 契約と
  整合）。`FitConfig` は `Copy`＋`Eq` 導出済みで f32 フィールドを足すと
  破壊的変更になるため、`Compiled`（`compile()` 済み状態を保持する
  非公開 struct）へ `amp: Option<AmpState>` として追加した——`FitConfig`
  自体は不変のまま。
- **skip 時の loss 記録**: `History::loss` は「scale 前の素の loss」を
  常に記録する（skip した step でも記録する。overflow をそのまま
  可視化するため）。`unscale` 後の skip 判定は `optimizer.step`／
  `apply_parameters` のみに作用し、loss 集計（サンプル数重み付き
  平均）には影響しない。
- **validation は f32 のまま**: `run_evaluate`（`fit_with_callbacks` の
  validation フェーズ・`evaluate` 自身）は `compiled.amp` を参照しない
  ため、AMP 使用時も常に f32 で評価する（実装計画 §8 のスコープ外
  整理。決定的な検証値を保つ狙い）。
- **callbacks（#1763）との関係**: `Callback::LrSchedule`／
  `EarlyStopping`／`ModelCheckpoint` はいずれも `history`（f32 の
  `loss`／`val_loss`）のみを見るため、AMP の有無に関わらず既存の
  callbacks 契約（`docs/compat-callbacks-design.md`）はそのまま成立
  する（`fit_with_callbacks` と `compile_with_amp` は独立に組み合わせ
  可能。組み合わせ専用テストは追加していない——両者とも `run_fit`
  経由で `compiled.amp` を見るだけの直交した分岐のため）。
