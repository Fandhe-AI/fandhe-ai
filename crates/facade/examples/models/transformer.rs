//! 参照モデル `TransformerClassifier` の examples 側アダプタ（イシュー #2202・親 #2190。
//! 本体は #2975 で `fandhe_ai::models::TransformerClassifier` として公開済み。旧名
//! `Transformer`／`TransformerConfig`）。
//!
//! `resnet.rs` と同じく、examples 限定 trait `ReferenceModule`／`Trainable` を公開型へ実装する
//! 薄い委譲だけを置く（パス呼び出しは inherent が優先されるため再帰しない）。ファイル名は
//! `#[path]` の変更を最小にするため据え置いている。

use fandhe_ai::optim::Adam;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

pub use fandhe_ai::models::{TransformerClassifier, TransformerClassifierConfig};

use super::reference_module::{ReferenceModule, Trainable};

impl ReferenceModule for TransformerClassifier {
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        TransformerClassifier::forward(self, tape, x)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        TransformerClassifier::named_parameters(self)
    }

    fn set_training(&mut self, training: bool) {
        TransformerClassifier::set_training(self, training);
    }

    fn is_training(&self) -> bool {
        self.training()
    }
}

impl Trainable for TransformerClassifier {
    fn train_step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<i32>,
        opt: &mut Adam,
    ) -> Result<f32, AutodiffError> {
        TransformerClassifier::train_step(self, x, y, opt)
    }
}
