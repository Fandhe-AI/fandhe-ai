//! 学習率スケジューラの最小セット（constant / step。親イシュー #192・
//! 本イシュー #195）。
//!
//! [`LrScheduler::lr_at`] は `step`（epoch またはイテレーション番号）を
//! 受け取り学習率を返す **stateless な純関数**として設計する
//! （呼び出し側で内部状態を持たず毎回同じ step を渡せば同じ値を返す。
//! 決定性・テスト容易性を優先し、optimizer 側の可変状態とは分離する）。
//!
//! SGD/AdamW（#193/#194）との結線（`lr_at(step)` の返り値を optimizer
//! の更新式へ渡す配線）は本イシューのスコープ外（PR 本文の
//! 「対象外（out-of-scope）」参照）。呼び出し側が
//! `let lr = scheduler.lr_at(step);` のように取り出して使う想定。

use crate::error::AutodiffError;

/// step 番号 → 学習率を返す純関数の共通 trait。
pub trait LrScheduler {
    /// `step`（0 始まりの epoch またはイテレーション番号）に対応する
    /// 学習率を返す。
    fn lr_at(&self, step: usize) -> f32;
}

/// 常に `base_lr` を返すスケジューラ（スケジューリングなしの既定値）。
pub struct ConstantLr {
    base_lr: f32,
}

impl ConstantLr {
    /// `base_lr` は有限かつ正の値でなければならない。
    ///
    /// # Errors
    ///
    /// `base_lr` が非有限または 0 以下の場合は
    /// `AutodiffError::InvalidArgument`（fail-closed。0・負の学習率は
    /// 「学習が進まない／発散する」設定であり、呼び出し側の誤り混入を
    /// 早期に検出する）。
    pub fn new(base_lr: f32) -> Result<Self, AutodiffError> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "base_lr は有限かつ正の値でなければならない: {base_lr}"
            )));
        }
        Ok(Self { base_lr })
    }
}

impl LrScheduler for ConstantLr {
    fn lr_at(&self, _step: usize) -> f32 {
        self.base_lr
    }
}

/// PyTorch `torch.optim.lr_scheduler.StepLR` と同一の階段減衰:
/// `lr(step) = base_lr * gamma^(step / step_size)`（`step / step_size`
/// は整数除算・切り捨て）。
pub struct StepLr {
    base_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLr {
    /// `base_lr` は有限かつ正、`step_size` は 1 以上、`gamma` は有限
    /// かつ正でなければならない。
    ///
    /// # Errors
    ///
    /// いずれかの条件を満たさない場合は `AutodiffError::InvalidArgument`
    /// （fail-closed。`step_size == 0` は次段の整数除算がゼロ除算になる
    /// ため事前に弾く。`gamma <= 0` は学習率が非正・振動する設定であり
    /// 意図しない値混入を早期検出する）。
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Result<Self, AutodiffError> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "base_lr は有限かつ正の値でなければならない: {base_lr}"
            )));
        }
        if step_size == 0 {
            return Err(AutodiffError::InvalidArgument(
                "step_size は 1 以上でなければならない".to_string(),
            ));
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "gamma は有限かつ正の値でなければならない: {gamma}"
            )));
        }
        Ok(Self {
            base_lr,
            step_size,
            gamma,
        })
    }
}

impl LrScheduler for StepLr {
    fn lr_at(&self, step: usize) -> f32 {
        let exponent = (step / self.step_size) as i32;
        self.base_lr * self.gamma.powi(exponent)
    }
}
