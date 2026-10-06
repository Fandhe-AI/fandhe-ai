//! Adafactor（Shazeer & Stern, 2018。`torch.optim.Adafactor` 相当）。
//!
//! `torch.optim.Adafactor` の単一テンソル実装（`_single_tensor_adafactor`）
//! と同一系列を再現する（イシュー #2656・親 #2654。参照値は
//! `tests/fixtures/adafactor-pytorch-reference/adafactor_reference.json`。
//! 実 PyTorch 2.14.0+cpu 実行値。README 参照）。演算順（`t` は更新後の
//! step 数）:
//!
//! ```text
//! w     = t^beta2_decay                    # 「1 - beta2_t」。lerp の重み
//! rho   = min(lr, 1/sqrt(t))
//! alpha = max(eps2, RMS(param)) * rho      # weight decay より前の param
//! weight_decay != 0 なら param *= (1 - lr*weight_decay)   # decoupled
//!
//! rank >= 2（末尾 2 次元で因子分解。先頭次元はバッチ面として独立）:
//!   row_var = lerp(row_var, sum_j(g^2)/m, w)
//!   col_var = lerp(col_var, sum_i(g^2)/n, w)
//!   v       = row_var * col_var / clamp_min(mean_i(row_var), eps1)
//! rank <= 1:
//!   variance = lerp(variance, g*g, w);  v = variance
//!
//! u      = rsqrt(clamp_min(v, eps1^2)) * g
//! denom  = max(1, ||u|| / (sqrt(numel) * d))
//! param += u * (-alpha / denom)
//! ```
//!
//! **数値方針**: スカラー係数は `f64` で計算し使用直前に 1 回だけ `f32`
//! へ落とす。長軸縮約（二乗和・ノルム・行平均）は要素を先に `f64` へ昇格
//! してから index 順に蓄積し、1 回だけ `f32` へ downcast する
//! （`.claude/rules/coding-rust.md` の正規化統計の契約。先例 `lamb.rs`）。
//! Python の `max`／`min` は NaN を第 1 引数側へ倒し、torch の `clamp_min`
//! は NaN を伝播する。両者は別ヘルパ（`py_max`／`clamp_min_nan`）で書き
//! 分け、`f32::max` は使わない。
//!
//! **非有限入力の契約**（PyTorch 準拠で伝播させ、黙って握りつぶさない）:
//! NaN／inf 勾配は `row_var`／`col_var`／`variance` を汚染し、以後そのス
//! ロット（因子分解ではその行・列を共有する要素）は回復しない。fixture の
//! `edge` ブロックで実測固定している。学習ループ側の非有限検出（`amp`）・
//! `clip` は `nn/optim/mod.rs` の適用順序契約どおり optimizer の前段にある。
//!
//! 要素数 0 のスロットは PyTorch が `ZeroDivisionError` になる入力であり、
//! `step()` が状態変更前に `InvalidArgument` で拒否する。rank 0 は
//! `variance` 経路で受理する（PyTorch と同じ）。
//!
//! `nn/optim/mod.rs` の doc が示す通り `step()` は `(param, grad)` の参照列を
//! 受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／`BackendOps` に
//! 一切依存しない値型・純関数であり、新規 `Op`／`BackendOps` メソッド／VJP
//! は追加していない（カーネルなし）。`rprop.rs` を鏡写しにした別実装で、
//! 内部ループは他 optimizer と共通化しない。`ParamGroupStep`／
//! `OptimizerStateDict` は未実装（決定記録 §10）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.Adafactor` と同一の既定値（`eps = (None, 1e-3)` の `None`
/// は `finfo(float32).eps` で、`f32::EPSILON` と同値）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdafactorConfig {
    /// 学習率（`rho` の上限・weight decay 係数に効く）。有限かつ `>= 0.0`。
    pub lr: f32,
    /// 第 2 モーメント減衰指数（`beta2_decay`）。有限かつ `<= 0.0`。
    pub beta2_decay: f32,
    /// 分母の下限（`eps[0]`）。有限かつ `eps1^2 >= f32::MIN_POSITIVE`
    /// （約 `1.1e-19` 以上。0・極小値は `rsqrt(0)` が inf を生みゼロ勾配で
    /// NaN になるため拒否。PyTorch は `>= 0` を許す）。
    pub eps1: f32,
    /// パラメータ RMS の下限（`eps[1]`）。有限かつ `>= 0.0`。
    pub eps2: f32,
    /// 更新クリップ閾値（`d`）。有限かつ `>= 1.0`。
    pub d: f32,
    /// decoupled weight decay 係数。有限かつ `>= 0.0`。
    pub weight_decay: f32,
}

impl Default for AdafactorConfig {
    fn default() -> AdafactorConfig {
        AdafactorConfig {
            lr: 1e-2,
            beta2_decay: -0.8,
            eps1: f32::EPSILON,
            eps2: 1e-3,
            d: 1.0,
            weight_decay: 0.0,
        }
    }
}

/// 第 2 モーメントの保持形。初回 `step()` の `param` の rank で確定する。
enum SlotMoments {
    /// rank >= 2: 末尾 2 次元 `[n, m]`・先頭次元の積 `B` に対し
    /// `row_var` は `B*n` 要素、`col_var` は `B*m` 要素。
    Factored {
        row_var: Vec<f32>,
        col_var: Vec<f32>,
    },
    /// rank 0・1: 要素ごとの `variance`。
    Full { variance: Vec<f32> },
}

struct SlotState {
    shape: Vec<usize>,
    moments: SlotMoments,
}

/// Adafactor optimizer 本体。ハイパーパラメータ・step 数・スロットごとの
/// 状態を保持する。`step_count` は全スロットで同値のため optimizer 全体で
/// 1 個とする（`NAdam` の `mu_product` と同じ理由）。
pub struct Adafactor {
    config: AdafactorConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

/// Python 組み込み `max(a, b)`: `b` が `a` より真に大きいときだけ `b`。
/// `b` が NaN なら `a` が返る。
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// torch の `clamp_min`: NaN は伝播する（`f32::max` は NaN を捨てるため
/// 使えない）。
fn clamp_min_nan(x: f32, lo: f32) -> f32 {
    if x.is_nan() {
        x
    } else if x < lo {
        lo
    } else {
        x
    }
}

/// `lerp_(start, end, weight)`。`adamax.rs`／`rmsprop.rs` と同じ 2 分岐形。
fn lerp(start: f32, end: f32, weight: f32) -> f32 {
    if weight.abs() < 0.5 {
        start + weight * (end - start)
    } else {
        end - (end - start) * (1.0 - weight)
    }
}

/// 要素を先に `f64` へ昇格してから二乗し index 順に蓄積、`f64` で `sqrt`
/// した L2 ノルム（`f64` のまま返す）。RMS 等の除算まで `f64` で行う呼び出し
/// 側向け（`[f32::MAX, f32::MAX]` でも `f32` へ落とす前なので inf にならない）。
fn norm2_f64(values: impl Iterator<Item = f32>) -> f64 {
    let mut acc = 0.0f64;
    for v in values {
        let v = v as f64;
        acc = v.mul_add(v, acc);
    }
    acc.sqrt()
}

/// [`norm2_f64`] を 1 回だけ `f32` へ downcast した L2 ノルム。
fn norm2(values: impl Iterator<Item = f32>) -> f32 {
    norm2_f64(values) as f32
}

fn validate_hyperparameters(who: &str, c: &AdafactorConfig) -> Result<(), AutodiffError> {
    let bad = |name: &str, rule: &str, value: f32| {
        Err(AutodiffError::InvalidArgument(format!(
            "{who}: {name} must be {rule}, got {value}"
        )))
    };
    if !(c.lr.is_finite() && c.lr >= 0.0) {
        return bad("lr", "finite and >= 0.0", c.lr);
    }
    if !(c.beta2_decay.is_finite() && c.beta2_decay <= 0.0) {
        return bad("beta2_decay", "finite and <= 0.0", c.beta2_decay);
    }
    // `eps1^2` を `f32` へ変換した値が 0 や非正規数へ潰れると、ゼロ勾配で
    // `rsqrt(0) * 0 = NaN` になる。`eps1^2 >= f32::MIN_POSITIVE` を要求する。
    if !(c.eps1.is_finite()
        && c.eps1 > 0.0
        && ((c.eps1 as f64 * c.eps1 as f64) as f32) >= f32::MIN_POSITIVE)
    {
        return bad(
            "eps1",
            "finite and > 0.0 with eps1^2 >= f32::MIN_POSITIVE (about 1.1e-19)",
            c.eps1,
        );
    }
    if !(c.eps2.is_finite() && c.eps2 >= 0.0) {
        return bad("eps2", "finite and >= 0.0", c.eps2);
    }
    if !(c.d.is_finite() && c.d >= 1.0) {
        return bad("d", "finite and >= 1.0", c.d);
    }
    if !(c.weight_decay.is_finite() && c.weight_decay >= 0.0) {
        return bad("weight_decay", "finite and >= 0.0", c.weight_decay);
    }
    Ok(())
}

impl Adafactor {
    /// ハイパーパラメータを検証して構築する。
    ///
    /// # Errors
    ///
    /// 非有限値・範囲外（`lr < 0`・`beta2_decay > 0`・`eps1 <= 0`・
    /// `eps2 < 0`・`d < 1`・`weight_decay < 0`）は
    /// `AutodiffError::InvalidArgument`。
    pub fn new(config: AdafactorConfig) -> Result<Adafactor, AutodiffError> {
        validate_hyperparameters("Adafactor::new", &config)?;
        Ok(Adafactor {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &AdafactorConfig {
        &self.config
    }

    /// `config.lr` のみを書き換える。次の `step()` から即時に `rho` と
    /// weight decay へ効く（PyTorch で `param_group["lr"]` を変更した場合と
    /// 同じ）。状態は不変。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値なら `AutodiffError::InvalidArgument`
    /// （`self.config` は変更しない）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adafactor::set_lr: lr must be finite and >= 0.0, got {new_lr}"
            )));
        }
        self.config.lr = new_lr;
        Ok(())
    }

    /// 実行済み `step()` 回数。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// `rprop.rs` と同じ 2 段構成（副作用なしの検証 → 状態変更）で、失敗時は
    /// 状態（`step_count`・スロット）を一切変更しない。
    pub fn step(
        &mut self,
        params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        for (param, grad) in params_and_grads {
            if grad.shape() != param.shape() {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: grad.shape().to_vec(),
                    rhs: param.shape().to_vec(),
                }));
            }
            if param.numel() == 0 {
                return Err(AutodiffError::InvalidArgument(
                    "Adafactor::step: parameters with zero elements are not supported \
                     (RMS and update norm divide by sqrt(numel))"
                        .to_string(),
                ));
            }
        }
        if !self.states.is_empty() {
            if params_and_grads.len() != self.states.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Adafactor::step: slot count changed across calls (expected {}, got {}); \
                     Adafactor state (row_var/col_var/variance) is keyed by call-order slot \
                     index and cannot be resized after the first step()",
                    self.states.len(),
                    params_and_grads.len()
                )));
            }
            for (slot, (param, _)) in self.states.iter().zip(params_and_grads.iter()) {
                if param.shape() != slot.shape.as_slice() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: param.shape().to_vec(),
                        rhs: slot.shape.clone(),
                    }));
                }
            }
        }

        // 遅延初期化より前に確定させ、Err 時に状態が変化しないアトミック性を
        // 保つ（`rprop.rs` の同箇所と同じ理由）。
        let next_step_count = self.step_count.checked_add(1).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Adafactor::step: step_count overflow: too many step() calls for this optimizer \
                 to advance further"
                    .to_string(),
            )
        })?;

        if self.states.is_empty() && !params_and_grads.is_empty() {
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| {
                    let shape = param.shape().to_vec();
                    let moments = if shape.len() >= 2 {
                        let n = shape[shape.len() - 2];
                        let m = shape[shape.len() - 1];
                        let batch = param.numel() / (n * m);
                        SlotMoments::Factored {
                            row_var: vec![0.0f32; batch * n],
                            col_var: vec![0.0f32; batch * m],
                        }
                    } else {
                        SlotMoments::Full {
                            variance: vec![0.0f32; param.numel()],
                        }
                    };
                    SlotState { shape, moments }
                })
                .collect();
        }
        self.step_count = next_step_count;

        let cfg = self.config;
        let t = next_step_count as f64;
        let lr = cfg.lr as f64;
        let w = t.powf(cfg.beta2_decay as f64) as f32;
        let rho = {
            let inv = 1.0 / t.sqrt();
            if inv < lr { inv } else { lr }
        };
        let eps1 = cfg.eps1;
        let eps1_sq = (cfg.eps1 as f64 * cfg.eps1 as f64) as f32;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let numel = param_data.len();
            let sqrt_numel = (numel as f64).sqrt();

            // `alpha` は weight decay より前の param の RMS から求める。
            let rms_p = norm2_f64(param_data.iter().copied()) / sqrt_numel;
            let alpha = py_max(cfg.eps2 as f64, rms_p) * rho;

            let mut new_param: Vec<f32> = param_data.to_vec();
            if cfg.weight_decay != 0.0 {
                let decay = (1.0 - lr * cfg.weight_decay as f64) as f32;
                for p in new_param.iter_mut() {
                    *p *= decay;
                }
            }

            // 分散推定値 `v` の更新と `u = rsqrt(clamp_min(v, eps1^2)) * g`。
            let mut update = vec![0.0f32; numel];
            match &mut slot.moments {
                SlotMoments::Factored { row_var, col_var } => {
                    let n = slot.shape[slot.shape.len() - 2];
                    let m = slot.shape[slot.shape.len() - 1];
                    let batch = numel / (n * m);
                    for b in 0..batch {
                        let face = &grad_data[b * n * m..(b + 1) * n * m];
                        let rv = &mut row_var[b * n..(b + 1) * n];
                        let cv = &mut col_var[b * m..(b + 1) * m];
                        for (i, r) in rv.iter_mut().enumerate() {
                            let nr = norm2(face[i * m..(i + 1) * m].iter().copied());
                            let row_mean = nr * nr / m as f32;
                            *r = lerp(*r, row_mean, w);
                        }
                        for (j, c) in cv.iter_mut().enumerate() {
                            let nc = norm2((0..n).map(|i| face[i * m + j]));
                            let col_mean = nc * nc / n as f32;
                            *c = lerp(*c, col_mean, w);
                        }
                        let mean_r = {
                            let mut acc = 0.0f64;
                            for r in rv.iter() {
                                acc += *r as f64;
                            }
                            (acc / n as f64) as f32
                        };
                        let denom_r = clamp_min_nan(mean_r, eps1);
                        for (i, r) in rv.iter().enumerate() {
                            for (j, c) in cv.iter().enumerate() {
                                let idx = b * n * m + i * m + j;
                                // 外積（K=1）→ 除算の順を保つ。
                                let v = *r * *c / denom_r;
                                let v = clamp_min_nan(v, eps1_sq);
                                update[idx] = 1.0 / v.sqrt() * grad_data[idx];
                            }
                        }
                    }
                }
                SlotMoments::Full { variance } => {
                    for i in 0..numel {
                        let g = grad_data[i];
                        variance[i] = lerp(variance[i], g * g, w);
                        let v = clamp_min_nan(variance[i], eps1_sq);
                        update[i] = 1.0 / v.sqrt() * g;
                    }
                }
            }

            let norm_u = norm2_f64(update.iter().copied());
            let denom = py_max(1.0, norm_u / (sqrt_numel * cfg.d as f64));
            let coef = (-alpha / denom) as f32;
            for i in 0..numel {
                new_param[i] = f32::mul_add(update[i], coef, new_param[i]);
            }

            out.push(Tensor::new(new_param, &slot.shape)?);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn vals(x: &Tensor<f32>) -> Vec<f32> {
        crate::eval::dense_vec(x)
    }

    fn close(a: f32, b: f64) -> bool {
        let d = (a as f64 - b).abs();
        d < 1e-5 || d / b.abs().max(1e-30) < 1e-3
    }

    #[test]
    fn py_max_and_clamp_min_nan_semantics() {
        assert_eq!(py_max(1.0, 2.0), 2.0);
        assert_eq!(py_max(1.0, 0.5), 1.0);
        // Python の max は NaN を第 1 引数側へ倒す。
        assert_eq!(py_max(1.0, f64::NAN), 1.0);
        assert!(clamp_min_nan(f32::NAN, 1.0).is_nan());
        assert_eq!(clamp_min_nan(0.5, 1.0), 1.0);
        assert_eq!(clamp_min_nan(2.0, 1.0), 2.0);
    }

    #[test]
    fn rejects_invalid_hyperparameters() {
        let d = AdafactorConfig::default();
        let bad = [
            AdafactorConfig { lr: -1.0, ..d },
            AdafactorConfig { lr: f32::NAN, ..d },
            AdafactorConfig {
                lr: f32::INFINITY,
                ..d
            },
            AdafactorConfig {
                beta2_decay: 0.1,
                ..d
            },
            AdafactorConfig {
                beta2_decay: f32::NAN,
                ..d
            },
            AdafactorConfig { eps1: 0.0, ..d },
            AdafactorConfig { eps1: -1.0, ..d },
            AdafactorConfig { eps1: 1e-30, ..d },
            AdafactorConfig {
                eps1: f32::INFINITY,
                ..d
            },
            AdafactorConfig { eps2: -1.0, ..d },
            AdafactorConfig {
                eps2: f32::NAN,
                ..d
            },
            AdafactorConfig { d: 0.5, ..d },
            AdafactorConfig { d: f32::NAN, ..d },
            AdafactorConfig {
                weight_decay: -0.1,
                ..d
            },
            AdafactorConfig {
                weight_decay: f32::INFINITY,
                ..d
            },
        ];
        for cfg in bad {
            assert!(
                matches!(Adafactor::new(cfg), Err(AutodiffError::InvalidArgument(_))),
                "{cfg:?}"
            );
        }
        assert!(Adafactor::new(d).is_ok());
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(matches!(
            opt.step(&[(&param, &grad)]),
            Err(AutodiffError::Shape(_))
        ));
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn rejects_zero_element_param_without_state_change() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![], &[0]);
        let g = t(vec![], &[0]);
        assert!(matches!(
            opt.step(&[(&p, &g)]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert_eq!(opt.step_count(), 0);
        assert!(opt.states.is_empty());
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        assert!(matches!(
            opt.step(&[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0], &[2]);
        let g1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        let p2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let g2 = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(matches!(
            opt.step(&[(&p2, &g2)]),
            Err(AutodiffError::Shape(_))
        ));
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let g1 = t(vec![0.1, -0.1, 0.3, 0.2], &[2, 2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        let bad_grad = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(opt.step(&[(&p1, &bad_grad)]).is_err());
        assert_eq!(opt.step_count(), 1);

        let mut opt_ref = Adafactor::new(AdafactorConfig::default()).unwrap();
        opt_ref.step(&[(&p1, &g1)]).unwrap();
        let a = opt.step(&[(&p1, &g1)]).unwrap();
        let b = opt_ref.step(&[(&p1, &g1)]).unwrap();
        assert_eq!(vals(&a[0]), vals(&b[0]));
    }

    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let bad_p = t(vec![1.0, 2.0], &[2]);
        let bad_g = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(opt.step(&[(&bad_p, &bad_g)]).is_err());
        let pa = t(vec![1.0], &[1]);
        let ga = t(vec![0.1], &[1]);
        let pb = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let gb = t(vec![0.1, 0.1, 0.1, 0.1], &[2, 2]);
        assert!(opt.step(&[(&pa, &ga), (&pb, &gb)]).is_ok());
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn step_count_overflow_is_rejected_without_state_change() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        opt.step_count = u64::MAX;
        let p = t(vec![1.0], &[1]);
        let g = t(vec![0.1], &[1]);
        assert!(matches!(
            opt.step(&[(&p, &g)]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert_eq!(opt.step_count(), u64::MAX);
        assert!(
            opt.states.is_empty(),
            "overflow 時に遅延初期化してはならない"
        );
    }

    /// t=1・rank 1 の閉形式: `w=1` より `variance=g²`・`u=sign(g)`・`denom=1`
    /// （`d=1`・`|u|` の RMS が 1 のため）。`param -= max(eps2, RMS(param)) *
    /// min(lr, 1) * sign(g)`。
    #[test]
    fn first_step_rank1_matches_closed_form() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![3.0, 4.0], &[2]);
        let g = t(vec![0.5, -2.0], &[2]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        let rms = (12.5f64).sqrt();
        let alpha = rms * 0.01;
        let v = vals(&out[0]);
        assert!(close(v[0], 3.0 - alpha), "{v:?}");
        assert!(close(v[1], 4.0 + alpha), "{v:?}");
    }

    /// t=1・rank 2 の閉形式: `v_ij = R_i * C_j / mean_i(R_i)`（`w=1`）。
    #[test]
    fn first_step_rank2_matches_closed_form() {
        let cfg = AdafactorConfig {
            d: 1e6,
            ..AdafactorConfig::default()
        };
        let mut opt = Adafactor::new(cfg).unwrap();
        let p = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let g = [0.5f64, -1.0, 2.0, 0.25, 1.5, -0.75];
        let gt = t(g.iter().map(|x| *x as f32).collect(), &[2, 3]);
        let out = opt.step(&[(&p, &gt)]).unwrap();

        let r: Vec<f64> = (0..2)
            .map(|i| (0..3).map(|j| g[i * 3 + j].powi(2)).sum::<f64>() / 3.0)
            .collect();
        let c: Vec<f64> = (0..3)
            .map(|j| (0..2).map(|i| g[i * 3 + j].powi(2)).sum::<f64>() / 2.0)
            .collect();
        let mean_r = (r[0] + r[1]) / 2.0;
        let rms_p = (91.0f64 / 6.0).sqrt();
        let alpha = rms_p * 0.01;
        let got = vals(&out[0]);
        for i in 0..2 {
            for j in 0..3 {
                let v = r[i] * c[j] / mean_r;
                let u = g[i * 3 + j] / v.sqrt();
                let expected = p_at(i * 3 + j) - alpha * u;
                assert!(close(got[i * 3 + j], expected), "({i},{j}) {got:?}");
            }
        }
        fn p_at(k: usize) -> f64 {
            (k + 1) as f64
        }
    }

    #[test]
    fn zero_grad_keeps_param() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let g = t(vec![0.0; 4], &[2, 2]);
        for _ in 0..3 {
            let out = opt.step(&[(&p, &g)]).unwrap();
            assert_eq!(vals(&out[0]), vals(&p));
        }
    }

    #[test]
    fn rank0_uses_full_variance_path() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![2.0], &[]);
        let g = t(vec![0.3], &[]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        assert!(out[0].shape().is_empty());
        // rank 0: RMS = 2、u = sign(g)、alpha = 2*0.01。
        assert!(close(vals(&out[0])[0], 2.0 - 0.02));
    }

    /// rank 3 のバッチ面は互いに独立に因子分解される。
    #[test]
    fn rank3_batches_are_independent() {
        let face_a = vec![0.5f32, -1.0, 2.0, 0.25, 1.5, -0.75];
        let face_b = vec![3.0f32, 0.1, -0.2, 0.7, -0.4, 0.9];
        let pa = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pb = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut both_g = face_a.clone();
        both_g.extend(&face_b);
        let mut both_p = pa.clone();
        both_p.extend(&pb);
        let mut o3 = Adafactor::new(AdafactorConfig::default()).unwrap();
        let mut oa = Adafactor::new(AdafactorConfig::default()).unwrap();
        let mut ob = Adafactor::new(AdafactorConfig::default()).unwrap();
        let mut p3 = t(both_p, &[2, 2, 3]);
        let mut pa_t = t(pa, &[2, 3]);
        let mut pb_t = t(pb, &[2, 3]);
        for _ in 0..4 {
            p3 = o3
                .step(&[(&p3, &t(both_g.clone(), &[2, 2, 3]))])
                .unwrap()
                .remove(0);
            pa_t = oa
                .step(&[(&pa_t, &t(face_a.clone(), &[2, 3]))])
                .unwrap()
                .remove(0);
            pb_t = ob
                .step(&[(&pb_t, &t(face_b.clone(), &[2, 3]))])
                .unwrap()
                .remove(0);
        }
        // RMS・update norm は face をまたぐため完全一致は要求しない。因子分解
        // 状態が独立であること（row_var／col_var の面別長さ）を直接確認する。
        match &o3.states[0].moments {
            SlotMoments::Factored { row_var, col_var } => {
                assert_eq!(row_var.len(), 4);
                assert_eq!(col_var.len(), 6);
                let SlotMoments::Factored {
                    row_var: ra,
                    col_var: ca,
                } = &oa.states[0].moments
                else {
                    panic!("rank 2 は Factored のはず");
                };
                let SlotMoments::Factored {
                    row_var: rb,
                    col_var: cb,
                } = &ob.states[0].moments
                else {
                    panic!("rank 2 は Factored のはず");
                };
                assert_eq!(&row_var[..2], ra.as_slice());
                assert_eq!(&row_var[2..], rb.as_slice());
                assert_eq!(&col_var[..3], ca.as_slice());
                assert_eq!(&col_var[3..], cb.as_slice());
            }
            SlotMoments::Full { .. } => panic!("rank 3 は Factored のはず"),
        }
        let _ = (p3, pa_t, pb_t);
    }

    /// `[f32::MAX, f32::MAX]` の param でも RMS が inf にならず（`f64` のまま
    /// 除算してから使う）、結果が有限になる。
    #[test]
    fn huge_param_rms_stays_finite() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![f32::MAX, f32::MAX], &[2]);
        let g = t(vec![1.0, 1.0], &[2]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        assert!(vals(&out[0]).iter().all(|x| x.is_finite()));
    }

    /// 受理される最小級の `eps1` でもゼロ勾配で NaN にならない。
    #[test]
    fn tiny_accepted_eps1_zero_grad_is_not_nan() {
        let cfg = AdafactorConfig {
            eps1: 1.2e-19,
            ..AdafactorConfig::default()
        };
        let mut opt = Adafactor::new(cfg).unwrap();
        let p = t(vec![1.0, 2.0], &[2]);
        let g = t(vec![0.0, 0.0], &[2]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        assert!(vals(&out[0]).iter().all(|x| x.is_finite()));
        let mut opt2 = Adafactor::new(cfg).unwrap();
        let p2 = t(vec![1.0; 6], &[2, 3]);
        let g2 = t(vec![0.0; 6], &[2, 3]);
        let out2 = opt2.step(&[(&p2, &g2)]).unwrap();
        assert!(vals(&out2[0]).iter().all(|x| x.is_finite()));
    }

    /// `eps2` が RMS より大きいとき、ゼロ初期化 param も動く。
    #[test]
    fn eps2_floor_moves_zero_initialized_param() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![0.0, 0.0], &[2]);
        let g = t(vec![1.0, -1.0], &[2]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        // alpha = eps2 * min(lr, 1) = 1e-3 * 1e-2。
        assert!(close(vals(&out[0])[0], -1e-5));
        assert!(close(vals(&out[0])[1], 1e-5));
    }

    /// `rho = min(lr, 1/sqrt(t))`: lr が大きいと t>=2 で `1/sqrt(t)` が勝つ。
    #[test]
    fn rho_uses_inverse_sqrt_when_lr_is_large() {
        let cfg = AdafactorConfig {
            lr: 1.0,
            ..AdafactorConfig::default()
        };
        let mut opt = Adafactor::new(cfg).unwrap();
        let g = t(vec![1.0], &[1]);
        let p0 = t(vec![10.0], &[1]);
        let p1 = opt.step(&[(&p0, &g)]).unwrap().remove(0);
        // t=1: rho=1, alpha=10, u=1 → 0。
        assert!(close(vals(&p1)[0], 0.0));
        let p2 = opt.step(&[(&p1, &g)]).unwrap().remove(0);
        // t=2: rho=1/sqrt2, alpha = max(1e-3, 0) * rho = 1e-3/sqrt2。
        assert!(close(vals(&p2)[0], -1e-3 / 2.0f64.sqrt()));
    }

    /// `d` 未満の RMS では clip されず、`d` が 1 で RMS(u) > 1 のとき clip。
    #[test]
    fn update_clip_divides_by_denom_when_rms_exceeds_d() {
        // 要素 [1, 1] の grad で v=eps1^2 へ clamp されるほど小さい分散になる
        // ことはないため、u の RMS は 1 のまま。clip が効く入力として
        // rank 2 の不均一 grad を使い、denom >= 1 の分岐を通す。
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![1.0; 4], &[2, 2]);
        let g = t(vec![10.0, 0.001, 0.001, 0.001], &[2, 2]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        assert!(vals(&out[0]).iter().all(|x| x.is_finite()));
    }

    #[test]
    fn weight_decay_applies_after_alpha_uses_pre_decay_param() {
        let cfg = AdafactorConfig {
            lr: 0.5,
            weight_decay: 0.2,
            ..AdafactorConfig::default()
        };
        let mut opt = Adafactor::new(cfg).unwrap();
        let p = t(vec![2.0], &[1]);
        let g = t(vec![1.0], &[1]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        // alpha = 2 * min(0.5, 1) = 1（減衰前の RMS=2）。decay = 1-0.5*0.2 =
        // 0.9 → 1.8 - 1.0 * sign(g)。
        assert!(close(vals(&out[0])[0], 2.0 * 0.9 - 1.0));
    }

    #[test]
    fn nan_grad_poisons_slot_state() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        let p = t(vec![1.0, 1.0], &[2]);
        let g = t(vec![f32::NAN, 1.0], &[2]);
        let out = opt.step(&[(&p, &g)]).unwrap().remove(0);
        assert!(vals(&out)[0].is_nan());
        let g2 = t(vec![1.0, 1.0], &[2]);
        let out2 = opt.step(&[(&out, &g2)]).unwrap().remove(0);
        assert!(vals(&out2)[0].is_nan());
    }

    #[test]
    fn set_lr_validates_and_takes_effect_next_step() {
        let mut opt = Adafactor::new(AdafactorConfig::default()).unwrap();
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                opt.set_lr(bad),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert_eq!(opt.config().lr, 1e-2);
        }
        let p = t(vec![1.0], &[1]);
        let g = t(vec![1.0], &[1]);
        let mut a = Adafactor::new(AdafactorConfig::default()).unwrap();
        let mut b = Adafactor::new(AdafactorConfig::default()).unwrap();
        b.set_lr(0.001).unwrap();
        assert_eq!(b.config().lr, 0.001);
        let ra = a.step(&[(&p, &g)]).unwrap();
        let rb = b.step(&[(&p, &g)]).unwrap();
        assert_ne!(vals(&ra[0]), vals(&rb[0]));
    }
}
