//! 学習率スケジューラの最小セット（constant / step / cosine annealing /
//! exponential / linear warmup。親イシュー #192・#195・#1745〈親
//! #1611〉）。
//!
//! [`LrScheduler::lr_at`] は `step`（epoch またはイテレーション番号）を
//! 受け取り学習率を返す **stateless な純関数**として設計する
//! （呼び出し側で内部状態を持たず毎回同じ step を渡せば同じ値を返す。
//! 決定性・テスト容易性を優先し、optimizer 側の可変状態とは分離する）。
//!
//! **唯一の例外**: [`super::reduce_lr_on_plateau::ReduceLrOnPlateau`]
//! （イシュー #1746）は検証指標の停滞検知のため `patience`／`best`／
//! `cooldown` の内部可変状態を持つ状態保持型スケジューラであり、この
//! stateless 契約には当てはまらない。同型は `lr_at(_step)` を「引数を
//! 無視して現在の学習率を返すだけ」の形で [`LrScheduler`] を実装し、
//! 状態を進める入口は専用の `step(metric)` に分離する（詳細は同モジュール
//! 冒頭 doc を参照）。
//!
//! SGD/AdamW（#193/#194）との結線（`lr_at(step)` の返り値を optimizer
//! の更新式へ渡す配線）は本モジュールのスコープ外（PR 本文の
//! 「対象外（out-of-scope）」参照）。呼び出し側が
//! `let lr = scheduler.lr_at(step);` のように取り出して使う想定。
//!
//! **#1745 で追加した 3 種**（[`CosineAnnealingLr`]・
//! [`ExponentialLr`]・[`LinearWarmupLr`]）は式ベース（stateless 純
//! 関数）で表現できる PyTorch 準拠のスケジューラである。状態保持型の
//! `ReduceLROnPlateau`（loss 履歴に依存）は対象外のまま（親 #1611 の
//! 兄弟イシュー #1746 が担当）。
//!
//! **#1747 で追加した [`OneCycleLr`]**（PyTorch
//! `torch.optim.lr_scheduler.OneCycleLR` 相当）は「フェーズ管理を要する」
//! ため当初は状態保持型として見送っていたが、`new` 構築時にフェーズ
//! 境界（`end_step`・`start_lr`・`end_lr` の表）を事前計算して保持する
//! ことで `lr_at` 自体は参照のみの **stateless 純関数**として表現できる
//! （内部可変状態を持たない。`OneCycleLr` インスタンス自体は不変な
//! フェーズ表を保持するのみで、他のスケジューラと同じ `LrScheduler`
//! trait を実装できる）。momentum cycling（`cycle_momentum` 等）は
//! 対象外（`OneCycleLr` モジュール doc 参照）。`cycle_momentum` 抜きの
//! lr 系列のみを提供する。
//!
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
/// PyTorch `torch.optim.lr_scheduler.OneCycleLR` の `anneal_strategy`
/// （`'cos'` あるいは `'linear'`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneCycleAnneal {
    /// コサインアニーリング（PyTorch 既定）。
    Cos,
    /// 線形アニーリング。
    Linear,
}

/// [`OneCycleLr`] のハイパーパラメータ。`AdamWConfig` 等と同じ Config
/// 構造体方式を採る（引数 7 個の positional `new` を避ける）。
///
/// `max_lr`／`total_steps` に妥当な既定値はないため `Default` は実装
/// しない。[`OneCycleLrConfig::new`] が PyTorch の残り 5 フィールドの
/// 既定値を埋めて返す。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneCycleLrConfig {
    /// サイクルのピーク学習率（PyTorch `max_lr`）。
    pub max_lr: f32,
    /// サイクル全体のステップ数（PyTorch `total_steps`）。呼び出し側が
    /// `epochs * steps_per_epoch` 等から求めて渡す想定
    /// （`epochs`／`steps_per_epoch` からの自動導出は対象外）。
    pub total_steps: usize,
    /// 上昇フェーズに充てる割合（PyTorch `pct_start`。既定 `0.3`）。
    pub pct_start: f32,
    /// アニーリング方式（PyTorch `anneal_strategy`。既定 `Cos`）。
    pub anneal_strategy: OneCycleAnneal,
    /// 初期学習率 `initial_lr = max_lr / div_factor`
    /// （PyTorch `div_factor`。既定 `25.0`）。
    pub div_factor: f32,
    /// 最終学習率 `min_lr = initial_lr / final_div_factor`
    /// （PyTorch `final_div_factor`。既定 `1e4`）。
    pub final_div_factor: f32,
    /// 3 フェーズ形式（上昇 → `max_lr` から `initial_lr` への下降 →
    /// `initial_lr` から `min_lr` への下降）を使うか（PyTorch
    /// `three_phase`。既定 `false`＝2 フェーズ形式）。
    pub three_phase: bool,
}

impl OneCycleLrConfig {
    /// `max_lr`／`total_steps` 以外を PyTorch `OneCycleLR` の既定値
    /// （`pct_start=0.3`・`anneal_strategy='cos'`・`div_factor=25.0`・
    /// `final_div_factor=1e4`・`three_phase=false`）で埋めて構築する。
    pub fn new(max_lr: f32, total_steps: usize) -> Self {
        Self {
            max_lr,
            total_steps,
            pct_start: 0.3,
            anneal_strategy: OneCycleAnneal::Cos,
            div_factor: 25.0,
            final_div_factor: 1e4,
            three_phase: false,
        }
    }
}

/// 1 フェーズ分の区間情報（`new` 構築時に事前計算して保持する）。
/// `end_step` はそのフェーズが終わる step 番号（`f64`。`total_steps`
/// が `usize` から変換された値のため小数にはならないが、`lr_at` 側の
/// 補間計算と型を揃えるため `f64` で保持する）。
#[derive(Debug, Clone, Copy)]
struct OneCyclePhase {
    end_step: f64,
    start_lr: f64,
    end_lr: f64,
}

/// PyTorch `torch.optim.lr_scheduler.OneCycleLR` 相当の 1 サイクル
/// 学習率スケジューラ（Smith, 2018 "Super-Convergence"）。
///
/// `new` 構築時にフェーズ境界（`OneCyclePhase` の列）を事前計算して
/// 保持するため、[`LrScheduler::lr_at`] 自体は参照のみで完結する
/// stateless 純関数として実装できる（モジュール冒頭 doc 参照）。
///
/// # 数値仕様（PyTorch `OneCycleLR.__init__`／`get_lr` の再現）
///
/// `initial_lr = max_lr / div_factor`・`min_lr = initial_lr /
/// final_div_factor` から、2 フェーズ形式（`three_phase=false`）では
/// `[0, pct_start*total_steps-1]`（`initial_lr → max_lr`）・
/// `[pct_start*total_steps-1, total_steps-1]`（`max_lr → min_lr`）の
/// 2 区間、3 フェーズ形式では `max_lr → initial_lr → min_lr` の
/// 3 区間を作る。各区間内は `anneal_strategy` に従い
/// コサイン（`start*cos²(πp/2) + end*sin²(πp/2)`。`start`／`end` の
/// 大きさが極端に異なる設定でも符号付き差分を経由しない桁落ち耐性の
/// ある重み付き和で補間する。`lr_at` 実装コメント参照）または線形
/// （`(end-start)*p+start`）で補間する（`p` は区間内の進捗 `[0,1]`）。
///
/// # `step >= total_steps` の扱い（PyTorch との意図的な相違）
///
/// PyTorch は `step > total_steps` で `ValueError` を送出する
/// （`get_lr` が呼ばれるたびに例外を投げうる設計）。[`LrScheduler::
/// lr_at`] は `Result` を返せない契約（trait 定義）のため、代わりに
/// `step` を `total_steps - 1` へ clamp し最終フェーズの `end_lr`
/// （`min_lr`）を返し続ける（panic しない。呼び出し側が学習ループを
/// 継続しても発散しない安全側の挙動）。
pub struct OneCycleLr {
    phases: Vec<OneCyclePhase>,
    anneal_strategy: OneCycleAnneal,
    /// `phases` 末尾の `end_step`（`f64`）を `usize` へ戻した値。`lr_at`
    /// の `step` clamp に使う（`total_steps - 1` と同値）。
    last_step: usize,
}

impl OneCycleLr {
    /// [`OneCycleLrConfig`] を検証して構築する。
    ///
    /// # Errors
    ///
    /// 以下のいずれかを満たさない場合は `AutodiffError::InvalidArgument`
    /// （fail-closed）:
    ///
    /// - `max_lr` は有限かつ正
    /// - `total_steps` は 1 以上
    /// - `pct_start` は有限かつ `0 < pct_start < 1`（PyTorch は
    ///   `[0, 1]` を許すが、`0`／`1` は一方のフェーズの長さが 0 になる
    ///   退化設定のためここでは拒否する）
    /// - `div_factor`／`final_div_factor` は有限かつ正（`< 1.0` は
    ///   PyTorch 自身が拒否しないため、ここでも拒否しない。`StepLr`
    ///   の `gamma > 1.0` 許容と同じ「過剰に拒否しない」規則）
    /// - 計算されるフェーズ境界（`end_step` の列）が `0` から狭義単調
    ///   増加であること（`pct_start`・`total_steps` の組合せによっては
    ///   最初の `end_step` が `0` 以下、または 3 フェーズ形式で
    ///   フェーズ 2 の `end_step` がフェーズ 3 の `end_step`
    ///   （`total_steps - 1`）を超える退化設定になりうるため、この
    ///   段階で明示的に拒否する）
    pub fn new(config: OneCycleLrConfig) -> Result<Self, AutodiffError> {
        let OneCycleLrConfig {
            max_lr,
            total_steps,
            pct_start,
            anneal_strategy,
            div_factor,
            final_div_factor,
            three_phase,
        } = config;

        if !max_lr.is_finite() || max_lr <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "max_lr は有限かつ正の値でなければならない: {max_lr}"
            )));
        }
        if total_steps == 0 {
            return Err(AutodiffError::InvalidArgument(
                "total_steps は 1 以上でなければならない".to_string(),
            ));
        }
        if !pct_start.is_finite() || pct_start <= 0.0 || pct_start >= 1.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "pct_start は有限かつ 0 より大きく 1 未満でなければならない: \
                 {pct_start}"
            )));
        }
        if !div_factor.is_finite() || div_factor <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "div_factor は有限かつ正の値でなければならない: {div_factor}"
            )));
        }
        if !final_div_factor.is_finite() || final_div_factor <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "final_div_factor は有限かつ正の値でなければならない: \
                 {final_div_factor}"
            )));
        }

        // `f64` で中間計算する（モジュール冒頭 doc の精度方針）。
        let max_lr_f64 = max_lr as f64;
        let total = total_steps as f64;
        let pct = pct_start as f64;
        let div_factor_f64 = div_factor as f64;
        let final_div_factor_f64 = final_div_factor as f64;

        let initial_lr = max_lr_f64 / div_factor_f64;
        let min_lr = initial_lr / final_div_factor_f64;

        // 導出値（`initial_lr`／`min_lr`）は `f64` では有限でも、
        // `lr_at` が最終的に返す `f32` へ変換した時点で overflow
        // （`max_lr` が極端に大きく `div_factor` が極端に小さい等）
        // して `infinity` になったり、underflow して `0.0` に丸め
        // られたりしうる（codex-review 指摘）。ここで `f32` 表現
        // 可能性を検証し、表現不能な設定は構築時点で fail-closed に
        // 拒否する（`step` ごとに `infinity`／`0.0` を返す壊れた
        // `lr_at` を後から観測させない）。
        let initial_lr_f32 = initial_lr as f32;
        if !initial_lr_f32.is_finite() || initial_lr_f32 <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "initial_lr(=max_lr/div_factor={initial_lr}) が f32 として表現不能\
                 （変換結果 {initial_lr_f32}）: max_lr={max_lr} div_factor={div_factor}"
            )));
        }
        let min_lr_f32 = min_lr as f32;
        if !min_lr_f32.is_finite() || min_lr_f32 <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "min_lr(=initial_lr/final_div_factor={min_lr}) が f32 として表現不能\
                 （変換結果 {min_lr_f32}）: max_lr={max_lr} div_factor={div_factor} \
                 final_div_factor={final_div_factor}"
            )));
        }

        let phases = if three_phase {
            vec![
                OneCyclePhase {
                    end_step: pct * total - 1.0,
                    start_lr: initial_lr,
                    end_lr: max_lr_f64,
                },
                OneCyclePhase {
                    end_step: 2.0 * pct * total - 2.0,
                    start_lr: max_lr_f64,
                    end_lr: initial_lr,
                },
                OneCyclePhase {
                    end_step: total - 1.0,
                    start_lr: initial_lr,
                    end_lr: min_lr,
                },
            ]
        } else {
            vec![
                OneCyclePhase {
                    end_step: pct * total - 1.0,
                    start_lr: initial_lr,
                    end_lr: max_lr_f64,
                },
                OneCyclePhase {
                    end_step: total - 1.0,
                    start_lr: max_lr_f64,
                    end_lr: min_lr,
                },
            ]
        };

        // フェーズ境界は 0 から狭義単調増加でなければならない
        // （退化設定の事前拒否。`new` の doc 参照）。
        let mut prev_end = 0.0_f64;
        for (index, phase) in phases.iter().enumerate() {
            if phase.end_step <= prev_end {
                return Err(AutodiffError::InvalidArgument(format!(
                    "pct_start/total_steps/three_phase の組合せによりフェーズ境界が \
                     単調増加でない（phase[{index}].end_step={} <= 直前の境界 {prev_end}）: \
                     pct_start={pct_start} total_steps={total_steps} three_phase={three_phase}",
                    phase.end_step
                )));
            }
            prev_end = phase.end_step;
        }

        // `total_steps >= 1` を上で検査済みのため `total_steps - 1` は
        // 常に非負。
        let last_step = total_steps - 1;

        Ok(Self {
            phases,
            anneal_strategy,
            last_step,
        })
    }
}

impl LrScheduler for OneCycleLr {
    fn lr_at(&self, step: usize) -> f32 {
        // PyTorch の `step > total_steps` での `ValueError` の代わりに
        // `total_steps - 1` へ clamp する（`new` doc の「`step >=
        // total_steps` の扱い」節参照。`lr_at` は `Result` を返せない
        // ため fail-closed に panic するのではなく、安全側の値
        // （`min_lr` を保持し続ける）を返す）。
        let step = step.min(self.last_step) as f64;

        let mut start_step = 0.0_f64;
        let last_index = self.phases.len().saturating_sub(1);
        for (index, phase) in self.phases.iter().enumerate() {
            let is_last = index == last_index;
            if step <= phase.end_step || is_last {
                let span = phase.end_step - start_step;
                // `new` でフェーズ境界の狭義単調増加を検証済みのため
                // `span > 0.0`（ゼロ除算にならない）。
                let p = (step - start_step) / span;
                // 区間の両端（`p<=0.0`／`p>=1.0`）では補間式を経由せず
                // `start_lr`／`end_lr` を直接返す（codex-review 指摘）。
                // `(end_lr-start_lr)*p+start_lr` 等の補間式は、
                // `start_lr`／`end_lr` の大きさが極端に異なる設定
                // （`final_div_factor` が非常に大きい等）では `end_lr`
                // が `start_lr` の桁に埋もれて減算時に丸め落ちし、
                // `p==1.0` でも厳密に `end_lr` を再現しない場合がある
                // （例: `end_lr - start_lr` が `f64` の丸めで
                // `-start_lr` に一致してしまい、結果が `0.0` になる）。
                // 直接返すことで「最終ステップ以降は `min_lr`（最終
                // フェーズの `end_lr`）を返し続ける」契約（`new` doc
                // 「`step >= total_steps` の扱い」節）を厳密に満たす。
                if p <= 0.0 {
                    return phase.start_lr as f32;
                }
                if p >= 1.0 {
                    return phase.end_lr as f32;
                }
                let value = match self.anneal_strategy {
                    OneCycleAnneal::Cos => {
                        // codex-review 指摘: `end_lr + (start_lr-end_lr)/2
                        // *(cos_p+1)` は `start_lr`／`end_lr` の大きさが
                        // 極端に異なる設定（例: `div_factor=1e20`）では
                        // `start_lr - end_lr` の減算段階で小さい方が
                        // 完全に丸め落ち、区間内部（`p` が 0 側でも）で
                        // 本来 `start_lr` に近いはずの値が `end_lr` の
                        // 桁に埋もれて誤った値（極端な場合 `0.0`）を返す
                        // （端点は既に上の `p<=0.0`/`p>=1.0` 早期 return
                        // で回避済みだが、区間内部はこの式のままでは
                        // 保護されない）。`(1+cos θ)/2 = cos²(θ/2)`・
                        // `(1-cos θ)/2 = sin²(θ/2)` の倍角恒等式を使い、
                        // 符号付き差分を経由しない非負の重み付き和
                        // （`start_lr*w_start + end_lr*w_end`）へ書き
                        // 換えることで、どちらの重みが優勢でも他方の
                        // 値を完全には失わない補間にする。
                        let half_theta = std::f64::consts::PI * p / 2.0;
                        let w_end = half_theta.sin().powi(2);
                        let w_start = half_theta.cos().powi(2);
                        phase.start_lr * w_start + phase.end_lr * w_end
                    }
                    OneCycleAnneal::Linear => (phase.end_lr - phase.start_lr) * p + phase.start_lr,
                };
                return value as f32;
            }
            start_step = phase.end_step;
        }

        // `phases` は `new` で必ず 2 個以上（2 フェーズ形式）または
        // 3 個以上（3 フェーズ形式）構築するため、ループは必ず上の
        // `return` で終わる（最終フェーズが `is_last` で確実に一致
        // する）。到達しないが、`f32` を返す契約を満たすため保険的に
        // 最終フェーズの `end_lr` を返す。
        self.phases
            .last()
            .map(|phase| phase.end_lr as f32)
            .unwrap_or(0.0)
    }
}
