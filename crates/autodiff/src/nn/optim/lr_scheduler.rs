//! 学習率スケジューラの最小セット（constant / step / cosine annealing /
//! exponential / linear warmup。親イシュー #192・#195・#1745〈親
//! #1611〉）。
//!
//! [`LrScheduler::lr_at`] は `step`（epoch またはイテレーション番号）を
//! 受け取り学習率を返す **stateless な純関数**として設計する
//! （呼び出し側で内部状態を持たず毎回同じ step を渡せば同じ値を返す。
//! 決定性・テスト容易性を優先し、optimizer 側の可変状態とは分離する）。
//!
//! SGD/AdamW（#193/#194）との結線（`lr_at(step)` の返り値を optimizer
//! の更新式へ渡す配線）は本モジュールのスコープ外（PR 本文の
//! 「対象外（out-of-scope）」参照）。呼び出し側が
//! `let lr = scheduler.lr_at(step);` のように取り出して使う想定。
//!
//! **#1745 で追加した 3 種**（[`CosineAnnealingLr`]・
//! [`ExponentialLr`]・[`LinearWarmupLr`]）は式ベース（stateless 純
//! 関数）で表現できる PyTorch 準拠のスケジューラであり、状態保持型の
//! `ReduceLROnPlateau`（loss 履歴に依存）・`OneCycleLR`（フェーズ管理を
//! 要する）は対象外（親 #1611 の兄弟イシュー #1746／#1747 が担当）。
//! いずれも `f64` で中間計算し最後に 1 回だけ `f32` へ downcast する
//! （`cos`／`powf` の libm 差による ULP 揺れを `f32` 直計算より抑える
//! 精度方針。bit 同一契約は主張しない。`.claude/rules/coding-rust.md`
//! の matmul 系 FMA 契約とは独立の軸）。`Op`／`BackendOps`／`Var`
//! への拡張は行わない（テンソル演算ではなくホスト側の `f32` 純関数の
//! ため。REQ-9 Tier 1 の「scheduler」行・`docs/compat-api-scope.md`
//! §1.2 参照）。

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
/// PyTorch `torch.optim.lr_scheduler.CosineAnnealingLR` の閉形式
/// （`_get_closed_form_lr`）と同一のコサインアニーリング:
/// `lr(step) = eta_min + (base_lr - eta_min) * (1 + cos(π * step / t_max)) / 2`。
///
/// **`step > t_max` の挙動**: PyTorch の閉形式は周期的であり
/// `warm restart` を行わない（`step == t_max` で `eta_min` に達し、
/// `step == 2 * t_max` で `base_lr` へ戻る）。TensorFlow
/// `tf.keras.optimizers.schedules.CosineDecay` のように
/// `min(step, decay_steps)` で clamp **しない**（PyTorch 準拠を採用。
/// clamp が必要な呼び出し側は `lr_at(step.min(t_max))` のように自分で
/// 適用する）。
pub struct CosineAnnealingLr {
    base_lr: f32,
    t_max: usize,
    eta_min: f32,
}

impl CosineAnnealingLr {
    /// `base_lr` は有限かつ正、`t_max` は 1 以上、`eta_min` は有限かつ
    /// `0 <= eta_min <= base_lr` でなければならない。
    ///
    /// # Errors
    ///
    /// いずれかの条件を満たさない場合は `AutodiffError::InvalidArgument`
    /// （fail-closed。`t_max == 0` は `step / t_max` の除算がゼロ除算に
    /// なるため事前に弾く。`eta_min` の範囲検査は「下限が上限
    /// `base_lr` を超える」逆転設定を早期に拒否する。`eta_min == 0.0`
    /// と `eta_min == base_lr` はいずれも受理する境界値）。
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> Result<Self, AutodiffError> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "base_lr は有限かつ正の値でなければならない: {base_lr}"
            )));
        }
        if t_max == 0 {
            return Err(AutodiffError::InvalidArgument(
                "t_max は 1 以上でなければならない".to_string(),
            ));
        }
        if !eta_min.is_finite() || eta_min < 0.0 || eta_min > base_lr {
            return Err(AutodiffError::InvalidArgument(format!(
                "eta_min は有限かつ 0 以上 base_lr 以下でなければならない: \
                 eta_min={eta_min} base_lr={base_lr}"
            )));
        }
        Ok(Self {
            base_lr,
            t_max,
            eta_min,
        })
    }
}

impl LrScheduler for CosineAnnealingLr {
    fn lr_at(&self, step: usize) -> f32 {
        // `cos` の libm 差による ULP 揺れを抑えるため `f64` で中間計算し、
        // 最後に 1 回だけ `f32` へ downcast する（bit 同一契約は主張
        // しない。`.claude/rules/coding-rust.md` の FMA 契約とは独立の
        // 精度方針）。
        let base_lr = self.base_lr as f64;
        let eta_min = self.eta_min as f64;
        let phase = std::f64::consts::PI * (step as f64) / (self.t_max as f64);
        (eta_min + (base_lr - eta_min) * (1.0 + phase.cos()) / 2.0) as f32
    }
}

/// PyTorch `torch.optim.lr_scheduler.ExponentialLR` と同一の指数減衰:
/// `lr(step) = base_lr * gamma^step`。
pub struct ExponentialLr {
    base_lr: f32,
    gamma: f32,
}

impl ExponentialLr {
    /// `base_lr` は有限かつ正、`gamma` は有限かつ正でなければならない
    /// （`gamma > 1.0` は拒否しない。`StepLr` と同一の検査規則）。
    ///
    /// # Errors
    ///
    /// いずれかの条件を満たさない場合は `AutodiffError::InvalidArgument`
    /// （fail-closed）。
    pub fn new(base_lr: f32, gamma: f32) -> Result<Self, AutodiffError> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "base_lr は有限かつ正の値でなければならない: {base_lr}"
            )));
        }
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "gamma は有限かつ正の値でなければならない: {gamma}"
            )));
        }
        Ok(Self { base_lr, gamma })
    }
}

impl LrScheduler for ExponentialLr {
    fn lr_at(&self, step: usize) -> f32 {
        // `StepLr::lr_at` の `as i32` キャスト（巨大 step でラップし
        // うる）は踏襲せず、`step as f64` で計算する。`f64` の指数計算
        // ・1 回の downcast は `CosineAnnealingLr` と同じ精度方針。
        let base_lr = self.base_lr as f64;
        let gamma = self.gamma as f64;
        (base_lr * gamma.powf(step as f64)) as f32
    }
}

/// PyTorch に同名クラスはないが、`torch.optim.lr_scheduler.LinearLR` の
/// `end_factor = 1.0` 固定形として定義する線形ウォームアップ:
/// `lr(step) = base_lr * (start_factor + (1 - start_factor)
/// * min(step, warmup_steps) / warmup_steps)`（`step >= warmup_steps`
/// 以降は `base_lr` を保持する）。
pub struct LinearWarmupLr {
    base_lr: f32,
    warmup_steps: usize,
    start_factor: f32,
}

impl LinearWarmupLr {
    /// `base_lr` は有限かつ正、`warmup_steps` は 1 以上、`start_factor`
    /// は有限かつ `0 < start_factor <= 1` でなければならない
    /// （PyTorch `LinearLR` 自身の検査と同じ）。
    ///
    /// # Errors
    ///
    /// いずれかの条件を満たさない場合は `AutodiffError::InvalidArgument`
    /// （fail-closed。`warmup_steps == 0` は `step / warmup_steps` の
    /// 除算がゼロ除算になるため事前に弾く）。
    pub fn new(
        base_lr: f32,
        warmup_steps: usize,
        start_factor: f32,
    ) -> Result<Self, AutodiffError> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "base_lr は有限かつ正の値でなければならない: {base_lr}"
            )));
        }
        if warmup_steps == 0 {
            return Err(AutodiffError::InvalidArgument(
                "warmup_steps は 1 以上でなければならない".to_string(),
            ));
        }
        if !start_factor.is_finite() || start_factor <= 0.0 || start_factor > 1.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "start_factor は有限かつ 0 より大きく 1 以下でなければならない: \
                 {start_factor}"
            )));
        }
        Ok(Self {
            base_lr,
            warmup_steps,
            start_factor,
        })
    }
}

impl LrScheduler for LinearWarmupLr {
    fn lr_at(&self, step: usize) -> f32 {
        let base_lr = self.base_lr as f64;
        let start_factor = self.start_factor as f64;
        let progress = (step.min(self.warmup_steps) as f64) / (self.warmup_steps as f64);
        (base_lr * (start_factor + (1.0 - start_factor) * progress)) as f32
    }
}
