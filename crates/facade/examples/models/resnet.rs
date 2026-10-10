//! 参照モデル `ResNet` の examples 側アダプタ（イシュー #2202・親 #2190。本体は #2975 で
//! `fandhe_ai::models::ResNet` として公開済み）。
//!
//! モデル定義は公開面へ移したため、ここには examples 限定 trait `ReferenceModule`／
//! `Trainable`（`reference_module.rs`）を公開型へ実装する薄い委譲だけを置く。
//! `ResNet::forward(self, ..)` のようなパス呼び出しでは inherent メソッドが trait メソッドより
//! 優先されるため、委譲は再帰しない。trait は examples ローカルなので orphan rule を満たす。
//! `ResNetBlock` は公開面にコンストラクタ・`forward` を持たないため trait は実装しない。
//! `reference_module.rs` と同じく取り込み側が `#[path]` で個別に取り込む
//! （`docs/reference-models-decision.md` #2202 節「取り込み方」・§11.8）。

use fandhe_ai::optim::Adam;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

pub use fandhe_ai::models::ResNet;

use super::reference_module::{ReferenceModule, Trainable};

impl ReferenceModule for ResNet {
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ResNet::forward(self, tape, x)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        ResNet::named_parameters(self)
    }

    fn set_training(&mut self, training: bool) {
        ResNet::set_training(self, training);
    }

    fn is_training(&self) -> bool {
        self.training()
    }
}

impl Trainable for ResNet {
    fn train_step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<i32>,
        opt: &mut Adam,
    ) -> Result<f32, AutodiffError> {
        ResNet::train_step(self, x, y, opt)
    }
}
