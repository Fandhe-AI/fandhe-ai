//! Keras 風 `compile()`／`fit()`／`evaluate()` 最小版（イシュー #1761・
//! 親 #1618・`docs/compat-api-scope.md` §1.2「Keras 風 `Sequential` の層
//! 追加と `compile()`／`fit()`／`evaluate()`／callbacks の最小版」行）。
//!
//! `Sequential::bind → forward → loss → backward → trainable_grads →
//! optimizer.step → apply_parameters` という定型（`sequential.rs`
//! モジュール doc の doctest・`tests/compat_sequential_train.rs`）を
//! `compile`／`fit`／`evaluate` の 3 メソッドへ畳み込む。**新規
//! `Op`／`BackendOps` メソッド／VJP／カーネルは一切追加しない**——
//! 既存公開 API（`Sequential::bind` 等・[`crate::optim`] の
//! optimizer・[`crate::data::DataLoader`]・`Var::mse_loss`／
//! `cross_entropy_loss`）の合成のみで実装する（REQ-9「薄いラッパーに
//! 徹する」）。
//!
//! callbacks（`EarlyStopping`／`ModelCheckpoint`）・`validation_data`・
//! metrics・LR スケジューラ連携（`fandhe_ai::optim` の optimizer は
//! 学習率更新 API を持たないため本 issue で結線不可）・`DataLoader` を
//! 直接受ける `fit` 入口は対象外（兄弟イシュー #1763 へ引き継ぐ）。

use crate::optim::{Adam, AdamConfig, AdamW, AdamWConfig, Sgd, SgdConfig};
use crate::{AutodiffError, Tensor};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_tensor_core::Element;
use fandhe_ai_tensor_core::data::{DataLoader, DataLoaderConfig, TensorDataset};

use super::sequential::Sequential;

/// `compile()` の `loss` 引数（`Reduction::Mean` 固定。#[non_exhaustive]
/// のため後続の損失追加〈#1763 以降〉が既存呼び出し元の非網羅的
/// `match` を破壊しない）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// [`crate::Var::mse_loss`]（`Reduction::Mean`）。`target` は
    /// `Tensor<f32>`（`pred` と同 shape）。
    Mse,
    /// [`crate::Var::cross_entropy_loss`]（`class_dim = 1`・
    /// `Reduction::Mean`）。`logits` は `[N, C]`・`target` は
    /// `Tensor<i32>` `[N]`（クラス添字）。
    CrossEntropy,
}

/// `compile()` の `optimizer` 引数（既存 [`crate::optim`] の
/// `*Config` 型を保持する variant のみ。ハイパーパラメータ検証は
/// `compile()` 内で各 `*::new` へ委譲する）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    Sgd(SgdConfig),
    AdamW(AdamWConfig),
    Adam(AdamConfig),
}

/// `fit()` の構成（Keras `fit(epochs=, batch_size=, shuffle=)` の
/// サブセット）。
///
/// **`shuffle` の既定値（`false`）について**: Keras `fit` の既定
/// `shuffle=True` とは異なり、[`fandhe_ai_tensor_core::data::
/// DataLoaderConfig::new`] と同じ「既定でグローバル RNG を消費しない」
/// 側へ揃えた（`docs/compat-fit-evaluate-design.md` 参照）。Keras
/// 相当にしたい呼び出し元は [`Self::shuffle`]`(true)` を明示する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitConfig {
    epochs: usize,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl FitConfig {
    pub fn new(epochs: usize, batch_size: usize) -> Self {
        FitConfig {
            epochs,
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    /// `true` で各 epoch 開始前にサンプル順をシャッフルする
    /// （[`fandhe_ai_tensor_core::data::DataLoaderConfig::shuffle`] への
    /// 委譲。グローバル RNG〈[`crate::manual_seed`]〉を消費する）。
    pub fn shuffle(mut self, on: bool) -> Self {
        self.shuffle = on;
        self
    }

    /// `true` で `batch_size` に満たない末尾バッチを切り捨てる
    /// （[`fandhe_ai_tensor_core::data::DataLoaderConfig::drop_last`]
    /// への委譲）。
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }
}

/// `fit()` の戻り値（Keras `History.history["loss"]` 相当）。
///
/// `loss[i]` は epoch `i`（0-indexed）の学習損失を、各バッチの
/// サンプル数で重み付けした平均（`Σ(batch_loss * batch_n) / Σ batch_n`。
/// `f64` で集計し最後に 1 回 `f32` へ downcast）として記録する。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct History {
    pub loss: Vec<f32>,
}

/// `fit()`／`evaluate()` の target 要素型（sealed。[`crate::CastElement`]
/// と同型の非公開 `private::Sealed` 経由）。`f32`（[`Loss::Mse`]）・
/// `i32`（[`Loss::CrossEntropy`]）のみ実装する——他の型を受け付ける
/// 誤用をコンパイル時に排除する（`.claude/rules/security.md` A03 の
/// 精神を型で担保する）。
pub trait FitTarget: Element + private::Sealed {
    /// `pred`（forward 出力）と `target_batch`（このバッチの正解値）から
    /// `loss` に応じた損失 `Var` を構築する。`loss` と `Self` の組が
    /// 対応しない場合（例: `Loss::Mse` × `i32` target）は
    /// `AutodiffError::InvalidArgument` を返す（fail-closed。loss と
    /// target の dtype 不整合は compile 時点では target 型が未知のため
    /// fit／evaluate 呼び出し時にのみ検出できる）。
    #[doc(hidden)]
    fn loss_for<'t>(
        loss: Loss,
        tape: &'t crate::Tape,
        pred: &crate::Var<'t>,
        target_batch: &Tensor<Self>,
    ) -> Result<crate::Var<'t>, AutodiffError>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
}

impl FitTarget for f32 {
    fn loss_for<'t>(
        loss: Loss,
        tape: &'t crate::Tape,
        pred: &crate::Var<'t>,
        target_batch: &Tensor<f32>,
    ) -> Result<crate::Var<'t>, AutodiffError> {
        match loss {
            Loss::Mse => {
                let target_var = tape.var_no_grad(target_batch);
                pred.mse_loss_with(&target_var, Reduction::Mean)
            }
            Loss::CrossEntropy => Err(AutodiffError::InvalidArgument(
                "Sequential::fit/evaluate: Loss::CrossEntropy には Tensor<i32> の \
                 target（クラス添字）が必要（Tensor<f32> が渡された）"
                    .to_string(),
            )),
        }
    }
}

impl FitTarget for i32 {
    fn loss_for<'t>(
        loss: Loss,
        _tape: &'t crate::Tape,
        pred: &crate::Var<'t>,
        target_batch: &Tensor<i32>,
    ) -> Result<crate::Var<'t>, AutodiffError> {
        match loss {
            Loss::CrossEntropy => pred.cross_entropy_loss(target_batch, 1, Reduction::Mean),
            Loss::Mse => Err(AutodiffError::InvalidArgument(
                "Sequential::fit/evaluate: Loss::Mse には Tensor<f32> の target が必要\
                 （Tensor<i32> が渡された）"
                    .to_string(),
            )),
        }
    }
}

/// `compile()` で構築した optimizer 本体（[`crate::optim::Sgd`]／
/// [`crate::optim::AdamW`]／[`crate::optim::Adam`] のいずれか）。
/// 3 者は `step` のシグネチャが異なる（`Sgd::step` は位置対応スライス
/// 2 本・`AdamW`／`Adam::step` はタプルスライス 1 本）ため、ここで
/// [`Sequential::trainable_parameters`]／`grad_refs` から共通の
/// `Result<Vec<Tensor<f32>>, AutodiffError>` へ橋渡しする。
enum OptimizerState {
    Sgd(Sgd),
    AdamW(AdamW),
    Adam(Adam),
}

impl std::fmt::Debug for OptimizerState {
    // `AdamW`／`Adam` は `Debug` を実装していない（内部の `m`／`v`
    // モーメントバッファを丸ごと出力する `Debug` 導出をあえて設けて
    // いない設計。`nn::optim::adamw`／`adam` 参照）ため、variant 名の
    // みを出す非網羅的な `Debug` を手書きする（`derive` 不可）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerState::Sgd(sgd) => f.debug_tuple("Sgd").field(sgd).finish(),
            OptimizerState::AdamW(_) => f.debug_tuple("AdamW").finish(),
            OptimizerState::Adam(_) => f.debug_tuple("Adam").finish(),
        }
    }
}

impl OptimizerState {
    fn new(optimizer: Optimizer) -> Result<Self, AutodiffError> {
        match optimizer {
            Optimizer::Sgd(config) => Ok(OptimizerState::Sgd(Sgd::new(config)?)),
            Optimizer::AdamW(config) => Ok(OptimizerState::AdamW(AdamW::new(config)?)),
            Optimizer::Adam(config) => Ok(OptimizerState::Adam(Adam::new(config)?)),
        }
    }

    /// `params.len() != grads.len()` を各 `step` 実装（`Sgd::step` は
    /// 検査済み）へ委譲する前に、`AdamW`／`Adam::step` が要求する
    /// `&[(&Tensor, &Tensor)]` への zip 変換自体が短い側で黙って
    /// 切り詰められてしまう（fail-closed 違反）のを防ぐため、ここで
    /// 明示的に事前検査する。
    fn step(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        if params.len() != grads.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sequential::fit: trainable_parameters().len() ({}) != \
                 trainable_grads().len() ({})",
                params.len(),
                grads.len()
            )));
        }
        match self {
            OptimizerState::Sgd(sgd) => sgd.step(params, grads),
            OptimizerState::AdamW(adamw) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                adamw.step(&pairs)
            }
            OptimizerState::Adam(adam) => {
                let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().copied().zip(grads.iter().copied()).collect();
                adam.step(&pairs)
            }
        }
    }
}

/// [`Sequential::compiled`] が保持する optimizer／loss の組
/// （非公開・`pub(super)` で `sequential.rs` からのみ到達可能）。
#[derive(Debug)]
pub(super) struct Compiled {
    optimizer: OptimizerState,
    loss: Loss,
}

/// 未 compile のモデルへ `fit`／`evaluate` を呼んだ場合の共通エラー。
fn not_compiled(method: &str) -> AutodiffError {
    AutodiffError::InvalidArgument(format!(
        "Sequential::{method}: compile() が呼ばれていない（optimizer／loss が未設定）"
    ))
}

impl Sequential {
    /// optimizer／loss を設定する（Keras `model.compile(optimizer, loss)`
    /// 相当）。ハイパーパラメータ検証（`lr < 0.0` 等）は各
    /// `Sgd`／`AdamW`／`Adam::new` へ委譲する。
    ///
    /// 既に compile 済みのモデルへ再度呼ぶと、それまでの optimizer
    /// 状態（momentum バッファ・m/v 等）を破棄して新しい `optimizer`／
    /// `loss` へ置き換える（Keras の再 `compile()` と同じ挙動）。
    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss) -> Result<(), AutodiffError> {
        self.compiled = Some(Compiled {
            optimizer: OptimizerState::new(optimizer)?,
            loss,
        });
        Ok(())
    }

    /// [`Self::compile`] 済みかどうか。
    pub fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }

    /// `x`（`[N, ...]`）・`y`（`[N, ...]`）を `config.epochs` 回学習する
    /// （Keras `model.fit(x, y, epochs=, batch_size=)` 相当）。
    ///
    /// 1 バッチあたりの演算列は [`crate::compat::SequentialVars::forward`] →
    /// `T::loss_for` → [`crate::Tape::backward`] →
    /// [`crate::compat::SequentialVars::trainable_grads`] → optimizer `step` →
    /// [`Self::apply_parameters`] という、`sequential.rs` モジュール
    /// doc の doctest と同一の並びである（`fandhe_ai::tape()`。既定
    /// CPU・`CpuBackendOps`・融合有効）。
    ///
    /// # train／eval モード
    ///
    /// 呼び出し前のモード（[`Self::training`]）を記憶し、学習中は
    /// train モードへ切り替える（[`Self::add_dropout`] 入りモデルの
    /// マスク適用のため）。成功・失敗（バッチ処理中のエラー）いずれの
    /// 場合も元のモードへ復元する。
    ///
    /// # エラー
    /// - 未 compile → `InvalidArgument`（[`Self::is_compiled`] が
    ///   `false` のまま維持される）
    /// - `config.epochs == 0` → `InvalidArgument`
    /// - `x`／`y` のサンプル数不一致・`batch_size == 0` →
    ///   [`fandhe_ai_tensor_core::data::DataError`] 由来の
    ///   `InvalidArgument`
    /// - `loss` と `T`（target dtype）の組み合わせが不整合
    ///   （[`FitTarget::loss_for`] 参照）→ `InvalidArgument`
    /// - バッチ処理中に失敗した場合も、compile 済み状態
    ///   （[`Self::is_compiled`]`() == true`）は維持したまま（optimizer
    ///   状態は成功した step の分だけ進んだまま）エラーを返す
    ///   （fail-closed。モデルを「未 compile」状態へ落とさない）
    pub fn fit<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
    ) -> Result<History, AutodiffError> {
        // (1) 未 compile 検査・compiled の一時取り出し（借用衝突回避。
        // `run_fit` 内で `bind`〈&self 借用〉と `self.compiled`〈&mut
        // self 借用〉を同時に持てないため、compiled を一旦取り外して
        // 別変数として扱う。結果を問わず必ず書き戻す）。
        let mut compiled = self.compiled.take().ok_or_else(|| not_compiled("fit"))?;

        // (2) 引数検査（モード変更・DataLoader 構築より前。早期 Err は
        // モード変更前のため復元不要）。
        if config.epochs == 0 {
            self.compiled = Some(compiled);
            return Err(AutodiffError::InvalidArgument(
                "Sequential::fit: epochs == 0".to_string(),
            ));
        }

        // (3) train モードへ切り替え（Dropout 入りモデルのマスク適用の
        // ため）。復元は成功・失敗いずれの経路でも必ず行う。
        let prev_training = self.training();
        self.set_training(true);

        let result = self.run_fit(&mut compiled, x, y, config);

        // (4) モード復元・compiled の書き戻し（結果を問わず必ず行う。
        // fail-closed: 失敗した fit の後もモデルを「未 compile」状態へ
        // 落とさない）。
        self.set_training(prev_training);
        self.compiled = Some(compiled);
        result
    }

    /// [`Self::fit`] の本体（compiled を取り出し済みの状態で呼ばれる。
    /// `&mut self` と `compiled: &mut Compiled` を独立した借用として
    /// 受け取ることで、`self.bind(&tape)`〈`&self`〉と
    /// `compiled.optimizer.step`〈`&mut compiled`〉を同時に生かせる）。
    fn run_fit<T: FitTarget>(
        &mut self,
        compiled: &mut Compiled,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        config: FitConfig,
    ) -> Result<History, AutodiffError> {
        let to_invalid_arg = |e: fandhe_ai_tensor_core::data::DataError| {
            AutodiffError::InvalidArgument(format!("Sequential::fit: {e}"))
        };
        let x_dataset = TensorDataset::new(x.clone()).map_err(to_invalid_arg)?;
        let y_dataset = TensorDataset::new(y.clone()).map_err(to_invalid_arg)?;
        let loader = DataLoader::new(
            (x_dataset, y_dataset),
            DataLoaderConfig::new(config.batch_size)
                .shuffle(config.shuffle)
                .drop_last(config.drop_last),
        )
        .map_err(to_invalid_arg)?;

        let mut history = History {
            loss: Vec::with_capacity(config.epochs),
        };

        for _ in 0..config.epochs {
            let mut weighted_sum = 0.0f64;
            let mut count = 0usize;

            for batch in &loader {
                let (x_batch, y_batch) = batch
                    .map_err(|e| AutodiffError::InvalidArgument(format!("Sequential::fit: {e}")))?;
                let n_batch = x_batch.shape().first().copied().unwrap_or(0);

                let updated = {
                    let tape = crate::tape();
                    let bound = self.bind(&tape);
                    let x_var = tape.var(&x_batch);

                    let pred = bound.forward(&tape, &x_var)?;
                    let loss_var = T::loss_for(compiled.loss, &tape, &pred, &y_batch)?;
                    let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                        AutodiffError::InvalidArgument(
                            "Sequential::fit: loss の shape が [] ではない\
                             （loss 演算の契約違反）"
                                .to_string(),
                        )
                    })?;
                    weighted_sum += loss_scalar as f64 * n_batch as f64;
                    count += n_batch;

                    let grads = tape.backward(&loss_var)?;
                    let grad_refs = bound.trainable_grads(&grads)?;
                    let param_refs = self.trainable_parameters();
                    compiled.optimizer.step(&param_refs, &grad_refs)?
                };
                self.apply_parameters(updated)?;
            }

            if count == 0 {
                // `drop_last = true` で全バッチが落ちた場合（0 除算を
                // 黙って NaN にしない。fail-closed）。
                return Err(AutodiffError::InvalidArgument(
                    "Sequential::fit: この epoch で処理されたサンプルが 0 件\
                     （drop_last により全バッチが切り捨てられた可能性がある）"
                        .to_string(),
                ));
            }
            history.loss.push((weighted_sum / count as f64) as f32);
        }

        Ok(history)
    }

    /// `x`／`y` に対する平均損失を評価する（Keras
    /// `model.evaluate(x, y, batch_size=)` 相当。学習は行わない）。
    ///
    /// `batch_size == x.shape()[0]`（全件 1 バッチ）のとき、
    /// [`Sequential::forward`] → `T::loss_for` の直接計算と bit 完全
    /// 一致する。`batch_size` を分割した場合はサンプル数重み付き平均
    /// （[`Self::fit`] の `History::loss` と同じ集計方式）。
    ///
    /// # train／eval モード
    ///
    /// [`Self::fit`] と同様、呼び出し前のモードを記憶し評価中は eval
    /// モード（[`Self::eval`]）へ切り替える（Dropout 入りモデルでも
    /// 決定的な評価を得るため）。成功・失敗いずれの場合も元のモードへ
    /// 復元する。
    ///
    /// # エラー
    /// [`Self::fit`] と同じ検査（未 compile・サンプル数不一致・
    /// `batch_size == 0`・`loss` と `T` の不整合）に従う。
    pub fn evaluate<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        batch_size: usize,
    ) -> Result<f32, AutodiffError> {
        let loss = self
            .compiled
            .as_ref()
            .ok_or_else(|| not_compiled("evaluate"))?
            .loss;

        let prev_training = self.training();
        self.set_training(false);
        let result = self.run_evaluate::<T>(x, y, batch_size, loss);
        self.set_training(prev_training);
        result
    }

    fn run_evaluate<T: FitTarget>(
        &self,
        x: &Tensor<f32>,
        y: &Tensor<T>,
        batch_size: usize,
        loss: Loss,
    ) -> Result<f32, AutodiffError> {
        let to_invalid_arg = |e: fandhe_ai_tensor_core::data::DataError| {
            AutodiffError::InvalidArgument(format!("Sequential::evaluate: {e}"))
        };
        let x_dataset = TensorDataset::new(x.clone()).map_err(to_invalid_arg)?;
        let y_dataset = TensorDataset::new(y.clone()).map_err(to_invalid_arg)?;
        let loader = DataLoader::new((x_dataset, y_dataset), DataLoaderConfig::new(batch_size))
            .map_err(to_invalid_arg)?;

        let mut weighted_sum = 0.0f64;
        let mut count = 0usize;

        for batch in &loader {
            let (x_batch, y_batch) = batch.map_err(|e| {
                AutodiffError::InvalidArgument(format!("Sequential::evaluate: {e}"))
            })?;
            let n_batch = x_batch.shape().first().copied().unwrap_or(0);

            let tape = crate::tape();
            let x_var = tape.var(&x_batch);
            let pred = self.forward(&tape, &x_var)?;
            let loss_var = T::loss_for(loss, &tape, &pred, &y_batch)?;
            let loss_scalar = loss_var.to_tensor().get(&[]).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "Sequential::evaluate: loss の shape が [] ではない\
                     （loss 演算の契約違反）"
                        .to_string(),
                )
            })?;
            weighted_sum += loss_scalar as f64 * n_batch as f64;
            count += n_batch;
        }

        if count == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Sequential::evaluate: 処理されたサンプルが 0 件".to_string(),
            ));
        }
        Ok((weighted_sum / count as f64) as f32)
    }
}
