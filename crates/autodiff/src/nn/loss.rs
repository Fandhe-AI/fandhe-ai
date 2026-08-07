//! 損失関数群（#190。親イシュー #189「損失関数（MSE・CrossEntropy）の
//! 実装」）。
//!
//! `nn::activation`（`nn/activation.rs`）と同じ設計方針を踏襲する:
//! 各構造体は `Var::mse_loss_with`（`crate::var`）を呼ぶだけの薄い
//! ラッパーに徹し（REQ-9「互換 API 層は自作コアの上の薄いラッパーに
//! 徹する」）、共通 `Module` trait は未定義のため個別に
//! `forward(&self, pred: &Var<'t>, target: &Var<'t>) -> Result<Var<'t>,
//! AutodiffError> ` を公開する（trait 統一は #94/#95 側で設計する）。
//!
//! CrossEntropy（log-softmax 安定化を要する）は #191 のスコープであり
//! ここには含めない。

use crate::error::AutodiffError;
use crate::var::{Reduction, Var};

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

#[cfg(test)]
mod tests {
    //! `nn::loss::MseLoss::forward` が、対応する `Var` メソッド直接呼び
    //! 出しと同一の値・テープ記録を返すことを検証する（「薄いラッパー
    //! 性」の担保。`nn::activation` のテスト方針と同型。#190）。

    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn default_reduction_is_mean() {
        assert_eq!(MseLoss::default().reduction, Reduction::Mean);
    }

    #[test]
    fn forward_mean_matches_var_mse_loss() {
        let tape = Tape::new();
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
        let tape = Tape::new();
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
        let tape = Tape::new();
        let pred = tape.var(&tensor_core::Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
        let target = tape.var(&tensor_core::Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());

        let err = MseLoss::default().forward(&pred, &target).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }
}
