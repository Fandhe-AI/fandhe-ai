//! 検証指標の停滞を検知して学習率を下げる状態保持型スケジューラ
//! （PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau` 相当。
//! イシュー #1746・親 #1611。`docs/compat-api-scope.md` §1.2 Tier 1
//! 「scheduler（Cosine／Exponential／Plateau／OneCycle）」の Plateau 分）。
//!
//! # `lr_scheduler` モジュールの stateless 契約に対する唯一の例外
//!
//! [`super::lr_scheduler`] モジュール冒頭 doc は `LrScheduler::lr_at` を
//! 「呼び出し側で内部状態を持たず毎回同じ step を渡せば同じ値を返す
//! stateless な純関数」として設計すると明記している。[`ReduceLrOnPlateau`]
//! は検証指標（`step` 番号ではなく、呼び出しごとに変わる外部観測値）に
//! 応じて `patience` カウンタ・`best` 値・`cooldown` カウンタという
//! 内部可変状態を進める必要があり、この契約に当てはまらない。
//!
//! そのため本型は [`LrScheduler`] を実装しつつ、`lr_at(_step)` は
//! **引数を無視して `current_lr()` を返すだけ**（[`super::lr_scheduler::ConstantLr`]
//! の `_step` 無視と同型）とし、状態を進める唯一の入口は
//! [`ReduceLrOnPlateau::step`]（検証指標を受け取る）に限定する。
//! `&dyn LrScheduler` 経由で `lr_at` だけを呼んでも状態は変化しない
//! （呼び出し側は `step(metric)` を明示的に呼ぶ必要がある）。
//!
//! # PyTorch との対応・意味論
//!
//! `torch/optim/lr_scheduler.py::ReduceLROnPlateau` に準拠する:
//!
//! - `is_better(a, best)`:
//!   - [`PlateauMode::Min`] + [`ThresholdMode::Rel`]: `a < best * (1 - threshold)`
//!   - [`PlateauMode::Min`] + [`ThresholdMode::Abs`]: `a < best - threshold`
//!   - [`PlateauMode::Max`] + [`ThresholdMode::Rel`]: `a > best * (1 + threshold)`
//!   - [`PlateauMode::Max`] + [`ThresholdMode::Abs`]: `a > best + threshold`
//! - [`ReduceLrOnPlateau::step`] の順序:
//!   1. `is_better(metric, best)` なら `best = metric; num_bad_epochs = 0`、
//!      そうでなければ `num_bad_epochs += 1`
//!   2. `cooldown_counter > 0` なら `cooldown_counter -= 1; num_bad_epochs = 0`
//!      （cooldown 中は bad epoch をカウントしない）
//!   3. `num_bad_epochs > patience`（**厳密に大なり**。`patience + 1` 回目の
//!      連続悪化で発火する）なら `new_lr = max(lr * factor, min_lr)`。
//!      `lr - new_lr > eps` のときのみ実際に `lr = new_lr` を適用する
//!      （微小な変化は無視する）。**eps ガードで据え置いた場合でも**
//!      `cooldown_counter = cooldown; num_bad_epochs = 0` にリセットする
//!      （PyTorch 実装どおり。据え置きと未発火を区別しない）。
//!
//! # fail-closed 逸脱（PyTorch との相違点）
//!
//! PyTorch は `metric` が NaN でも黙って「悪化」として扱い処理を継続
//! するが、本実装は `.claude/rules/coding-rust.md`（本番経路で `unwrap`/
//! `expect` を使わない）・`clip::clip_grad_norm` 等の既存 fail-closed
//! 契約に合わせ、`metric` が非有限（NaN／±inf）のとき
//! `AutodiffError::InvalidArgument` を返し状態を変更しない。

use crate::error::AutodiffError;

use super::lr_scheduler::LrScheduler;

/// 監視指標の改善方向（PyTorch `mode` 引数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// 指標が小さいほど良い（例: validation loss）。既定値。
    Min,
    /// 指標が大きいほど良い（例: validation accuracy）。
    Max,
}

/// 改善判定のしきい値の解釈方法（PyTorch `threshold_mode` 引数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMode {
    /// `best` に対する相対値としてしきい値を解釈する。既定値。
    Rel,
    /// `best` に対する絶対値としてしきい値を解釈する。
    Abs,
}

/// [`ReduceLrOnPlateau::new`] のハイパーパラメータ。
///
/// [`Default`] は PyTorch `ReduceLROnPlateau` の既定値と一致する
/// （`mode=min, factor=0.1, patience=10, threshold=1e-4,
/// threshold_mode=rel, cooldown=0, min_lr=0, eps=1e-8`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReduceLrOnPlateauConfig {
    /// 監視指標の改善方向。
    pub mode: PlateauMode,
    /// 発火時に学習率へ乗じる係数（`0 < factor < 1`）。
    pub factor: f32,
    /// 改善なしを許容する連続観測回数。`0` は 1 回の悪化で即発火する。
    pub patience: usize,
    /// 改善判定のしきい値（非負）。
    pub threshold: f32,
    /// `threshold` の解釈方法。
    pub threshold_mode: ThresholdMode,
    /// 発火直後に `num_bad_epochs` のカウントを再開しない観測回数
    /// （非負）。
    pub cooldown: usize,
    /// 学習率の下限（非負）。`base_lr` 未満でなければならない。
    pub min_lr: f32,
    /// 学習率の実際の変化がこの値以下なら適用しない微小変化ガード
    /// （非負）。
    pub eps: f32,
}

impl Default for ReduceLrOnPlateauConfig {
    fn default() -> Self {
        Self {
            mode: PlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: ThresholdMode::Rel,
            cooldown: 0,
            min_lr: 0.0,
            eps: 1e-8,
        }
    }
}

/// 検証指標の停滞を検知して学習率を下げる状態保持型スケジューラ
/// （モジュール冒頭 doc 参照）。
pub struct ReduceLrOnPlateau {
    config: ReduceLrOnPlateauConfig,
    lr: f32,
    best: f32,
    num_bad_epochs: usize,
    cooldown_counter: usize,
}

impl ReduceLrOnPlateau {
    /// `base_lr` と設定から構築する。
    ///
    /// # Errors
    ///
    /// 以下のいずれかで `AutodiffError::InvalidArgument`（fail-closed）:
    /// - `base_lr` が非有限または `<= 0`
    /// - `factor` が非有限、または `(0, 1)` の開区間外（PyTorch も
    ///   `factor >= 1.0` を「学習率が増加してしまう」設定として拒否する）
    /// - `threshold`／`min_lr`／`eps` が非有限または負
    /// - `min_lr > base_lr`（最初の発火で学習率が上昇してしまう矛盾設定）
    pub fn new(base_lr: f32, config: ReduceLrOnPlateauConfig) -> Result<Self, AutodiffError> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "base_lr は有限かつ正の値でなければならない: {base_lr}"
            )));
        }
        if !config.factor.is_finite() || config.factor <= 0.0 || config.factor >= 1.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "factor は有限かつ (0, 1) の範囲でなければならない: {}",
                config.factor
            )));
        }
        if !config.threshold.is_finite() || config.threshold < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "threshold は有限かつ非負でなければならない: {}",
                config.threshold
            )));
        }
        if !config.min_lr.is_finite() || config.min_lr < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "min_lr は有限かつ非負でなければならない: {}",
                config.min_lr
            )));
        }
        if !config.eps.is_finite() || config.eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "eps は有限かつ非負でなければならない: {}",
                config.eps
            )));
        }
        if config.min_lr > base_lr {
            return Err(AutodiffError::InvalidArgument(format!(
                "min_lr（{}）は base_lr（{base_lr}）以下でなければならない",
                config.min_lr
            )));
        }

        let best = match config.mode {
            PlateauMode::Min => f32::INFINITY,
            PlateauMode::Max => f32::NEG_INFINITY,
        };

        Ok(Self {
            config,
            lr: base_lr,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        })
    }

    /// `metric`（検証指標。例: 検証 loss）を 1 回分観測し内部状態を
    /// 進める。更新後の学習率を返す（PyTorch `scheduler.step(val_loss)`
    /// 相当。詳細な発火手順はモジュール冒頭 doc「PyTorch との対応・
    /// 意味論」節を参照）。
    ///
    /// # Errors
    ///
    /// `metric` が非有限（NaN／±inf）の場合は
    /// `AutodiffError::InvalidArgument`（fail-closed。「fail-closed 逸脱」
    /// 節参照）。この場合内部状態は変更しない。
    pub fn step(&mut self, metric: f32) -> Result<f32, AutodiffError> {
        if !metric.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "metric は有限でなければならない: {metric}"
            )));
        }

        if self.is_better(metric) {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.config.patience {
            let new_lr = (self.lr * self.config.factor).max(self.config.min_lr);
            if self.lr - new_lr > self.config.eps {
                self.lr = new_lr;
            }
            // eps ガードで据え置いた場合でも cooldown・カウンタは
            // リセットする（PyTorch 実装どおり。モジュール冒頭 doc
            // 「PyTorch との対応・意味論」節の手順 3 を参照）。
            self.cooldown_counter = self.config.cooldown;
            self.num_bad_epochs = 0;
        }

        Ok(self.lr)
    }

    /// 現在の学習率を返す（状態は進めない）。
    pub fn current_lr(&self) -> f32 {
        self.lr
    }

    /// 診断用: 現時点の best 指標値。
    pub fn best(&self) -> f32 {
        self.best
    }

    /// 診断用: 連続悪化観測回数。
    pub fn num_bad_epochs(&self) -> usize {
        self.num_bad_epochs
    }

    /// 診断用: 残り cooldown 観測回数。
    pub fn cooldown_counter(&self) -> usize {
        self.cooldown_counter
    }

    /// `metric` が現在の `best` より改善しているかを判定する
    /// （モジュール冒頭 doc の `is_better` 定義。`config.mode`／
    /// `config.threshold_mode` に応じて 4 通りに分岐する）。
    fn is_better(&self, metric: f32) -> bool {
        match (self.config.mode, self.config.threshold_mode) {
            (PlateauMode::Min, ThresholdMode::Rel) => {
                metric < self.best * (1.0 - self.config.threshold)
            }
            (PlateauMode::Min, ThresholdMode::Abs) => metric < self.best - self.config.threshold,
            (PlateauMode::Max, ThresholdMode::Rel) => {
                metric > self.best * (1.0 + self.config.threshold)
            }
            (PlateauMode::Max, ThresholdMode::Abs) => metric > self.best + self.config.threshold,
        }
    }
}

impl LrScheduler for ReduceLrOnPlateau {
    /// `step` 引数は無視して [`ReduceLrOnPlateau::current_lr`] を返す
    /// （[`super::lr_scheduler::ConstantLr::lr_at`] と同型。状態を進める
    /// には [`ReduceLrOnPlateau::step`] を明示的に呼ぶ必要がある。
    /// モジュール冒頭 doc「stateless 契約に対する唯一の例外」節参照）。
    fn lr_at(&self, _step: usize) -> f32 {
        self.current_lr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_base_lr() {
        let config = ReduceLrOnPlateauConfig::default();
        assert!(ReduceLrOnPlateau::new(0.0, config).is_err());
        assert!(ReduceLrOnPlateau::new(-0.1, config).is_err());
        assert!(ReduceLrOnPlateau::new(f32::NAN, config).is_err());
        assert!(ReduceLrOnPlateau::new(f32::INFINITY, config).is_err());
    }

    #[test]
    fn rejects_invalid_factor() {
        let base = ReduceLrOnPlateauConfig::default();
        for factor in [0.0, 1.0, 1.5, f32::NAN, -0.5] {
            let config = ReduceLrOnPlateauConfig { factor, ..base };
            assert!(
                ReduceLrOnPlateau::new(0.1, config).is_err(),
                "factor={factor} は拒否されるはず"
            );
        }
    }

    #[test]
    fn rejects_min_lr_greater_than_base_lr() {
        let config = ReduceLrOnPlateauConfig {
            min_lr: 0.5,
            ..ReduceLrOnPlateauConfig::default()
        };
        assert!(ReduceLrOnPlateau::new(0.1, config).is_err());
    }

    #[test]
    fn lr_at_ignores_step_and_returns_current_lr() {
        let sched = ReduceLrOnPlateau::new(0.1, ReduceLrOnPlateauConfig::default()).unwrap();
        assert_eq!(sched.lr_at(0), 0.1);
        assert_eq!(sched.lr_at(999), 0.1);
    }
}
