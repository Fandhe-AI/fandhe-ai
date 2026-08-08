//! 損失関数群（親イシュー #189「損失関数（MSE・CrossEntropy）の実装」）。
//!
//! `nn::activation`（`nn/activation.rs`）と同じ設計方針を踏襲する:
//! 各構造体は `Var`（`crate::var`）の対応メソッドを呼ぶだけの薄い
//! ラッパーに徹し（REQ-9「互換 API 層は自作コアの上の薄いラッパーに
//! 徹する」の精神を `nn` モジュールにも適用。`nn/mod.rs` の境界説明
//! 参照）、共通 `Module` trait は未定義のため個別に `forward` を公開する
//! （trait 統一は #94/#95 側で設計する）。
//!
//! - #190（TASK-9.1c 相当）で `MseLoss`（`Var::mse_loss_with` の
//!   ラッパー）を追加した。
//! - #191 で `CrossEntropyLoss`（`Var::cross_entropy_loss` の
//!   ラッパー）を追加した。log-softmax → NLL を個別オペ合成せず、
//!   `Var::cross_entropy_loss`（`crate::var`）側で 1 個の融合オペ
//!   （`tape::Op::CrossEntropyLoss`）として実装する（実装計画 §3.1）。
//!   **Softmax 単体の公開活性化は本イシューのスコープ外**のまま維持する
//!   （`nn/activation.rs` の「Softmax は CE と密結合のため対象外」判断を
//!   踏襲）。
//!
//! `Reduction`（mean/sum 縮約）は MSE・CrossEntropy の両損失で共有する
//! ため `crate::var::Reduction`（#190 が定義）をそのまま再利用し、
//! `nn::loss` 側には重複定義を置かない。

use tensor_core::Tensor;

use crate::error::AutodiffError;
pub use crate::var::Reduction;
use crate::var::Var;

/// 平均二乗誤差損失。`Var::mse_loss_with` の薄いラッパー
/// （PyTorch `nn.MSELoss` 相当）。`Default` は `Reduction::Mean`
/// （PyTorch `nn.MSELoss` の既定 `reduction='mean'` と一致）。
#[derive(Debug, Clone, Copy)]
pub struct MseLoss {
    reduction: Reduction,
}

impl Default for MseLoss {
    fn default() -> Self {
        MseLoss {
            reduction: Reduction::Mean,
        }
    }
}

impl MseLoss {
    /// 縮約種別を指定して構築する。
    pub fn new(reduction: Reduction) -> Self {
        MseLoss { reduction }
    }

    /// `pred`（予測値）・`target`（正解値）から損失を計算する。
    /// shape 不一致・クロステープは `Var::mse_loss_with` の検査
    /// （`AutodiffError`）をそのまま返す。
    pub fn forward<'t>(&self, pred: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        pred.mse_loss_with(target, self.reduction)
    }
}

/// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191）。
/// `Var::cross_entropy_loss`（`crate::var`）の薄いラッパー。
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    /// クラス次元（PyTorch の `[N, C, d1..]` 形状は `class_dim = 1` に
    /// 相当する）。
    pub class_dim: usize,
    pub reduction: Reduction,
}

impl CrossEntropyLoss {
    /// `logits`（予測値・追跡対象）と `targets`（正解クラス添字・
    /// 非追跡）から損失を計算する。検査・数値安定化の実体は
    /// `Var::cross_entropy_loss` 側にあり、ここでは呼び出すだけ
    /// （「薄いラッパー性」は `tests/nn_cross_entropy.rs` で検証する）。
    pub fn forward<'t>(
        &self,
        logits: &Var<'t>,
        targets: &Tensor<i32>,
    ) -> Result<Var<'t>, AutodiffError> {
        logits.cross_entropy_loss(targets, self.class_dim, self.reduction)
    }
}

#[cfg(test)]
mod tests {
    //! `nn::loss::MseLoss::forward` が、対応する `Var` メソッド直接呼び
    //! 出しと同一の値・テープ記録を返すことを検証する（「薄いラッパー
    //! 性」の担保。`nn::activation` のテスト方針と同型。#190）。
    //! `CrossEntropyLoss::forward` の同種検証は `tests/nn_cross_entropy.rs`
    //! に含む（#191）。

    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn default_reduction_is_mean() {
        assert_eq!(MseLoss::default().reduction, Reduction::Mean);
    }

    #[test]
    fn forward_mean_matches_var_mse_loss() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape.var(&tensor_core::Tensor::new(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap());
        let target =
            tape.var(&tensor_core::Tensor::new(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = MseLoss::default().forward(&pred, &target).unwrap();
        let via_var = pred.mse_loss(&target).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn forward_sum_matches_var_mse_loss_with() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape.var(&tensor_core::Tensor::new(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap());
        let target =
            tape.var(&tensor_core::Tensor::new(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]).unwrap());

        let via_module = MseLoss::new(Reduction::Sum)
            .forward(&pred, &target)
            .unwrap();
        let via_var = pred.mse_loss_with(&target, Reduction::Sum).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn forward_propagates_shape_mismatch_error() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let pred = tape.var(&tensor_core::Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let target = tape.var(&tensor_core::Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        let err = MseLoss::default().forward(&pred, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }
}
