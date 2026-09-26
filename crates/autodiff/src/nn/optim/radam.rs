//! RAdam（Liu et al., 2019。`torch.optim.RAdam` 相当）。
//!
//! `torch.optim.RAdam` の単一テンソル実装（`_single_tensor_radam`。
//! weight decay（`decoupled_weight_decay` なら `param *= 1 - lr*wd`、
//! それ以外は `grad += wd*param`）→ `exp_avg.lerp_(grad, 1-beta1)` →
//! `exp_avg_sq = beta2*v + (1-beta2)*grad^2` →
//! `bc1 = 1-beta1^step`・`bc2 = 1-beta2^step` →
//! `bias_corrected_exp_avg = exp_avg / bc1` →
//! `rho_inf = 2/(1-beta2) - 1` →
//! `rho_t = rho_inf - 2*step*beta2^step/bc2` →
//! `rho_t > 5.0` なら
//! `rect = sqrt((rho_t-4)(rho_t-2)*rho_inf / ((rho_inf-4)(rho_inf-2)*rho_t))`・
//! `adaptive_lr = sqrt(bc2) / (sqrt(exp_avg_sq)+eps)` を用いて
//! `param -= bias_corrected_exp_avg * lr * adaptive_lr * rect`（左結合の
//! 評価順）、それ以外は `param -= bias_corrected_exp_avg * lr` の演算順）
//! と同一系列を再現する（イシュー #2171・親 #2131。参照値は
//! `tests/fixtures/radam-pytorch-reference/radam_reference.json`。実
//! PyTorch 2.14.0+cpu 実行値。README 参照。同 README に `rho_t` 分岐
//! 境界の実測記録あり）。
//!
//! `adamw.rs`（[`super::AdamW`]）を鏡写しにした別実装であり、内部
//! ループの共通化は行わない。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／
//! `BackendOps` に一切依存しない値型・純関数であり、新規 `Op`／
//! `BackendOps` メソッド／VJP は追加していない（カーネルなし）。

use std::collections::HashMap;

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.RAdam` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RAdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// `true` のとき decoupled weight decay（`param *= 1 - lr*wd`）を
    /// 使う。`false`（既定）は coupled（`grad += wd*param`）。
    pub decoupled_weight_decay: bool,
}

impl Default for RAdamConfig {
    fn default() -> RAdamConfig {
        RAdamConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
        }
    }
}

struct SlotState {
    shape: Vec<usize>,
    exp_avg: Vec<f32>,
    exp_avg_sq: Vec<f32>,
}

/// RAdam optimizer 本体。ハイパーパラメータ（[`RAdamConfig`]）と、
/// step 数・bias correction 用の `beta1^t`／`beta2^t` 逐次積（f64）・
/// スロットごとの状態（`SlotState`）を保持する。
pub struct RAdam {
    config: RAdamConfig,
    step_count: u64,
    beta1_pow_t: f64,
    beta2_pow_t: f64,
    states: Vec<SlotState>,
}

impl RAdam {
    /// ハイパーパラメータを検証して構築する。
    pub fn new(config: RAdamConfig) -> Result<RAdam, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RAdam::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RAdam::new: beta1 must be in [0.0, 1.0), got {}",
                config.beta1
            )));
        }
        if !(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RAdam::new: beta2 must be in [0.0, 1.0), got {}",
                config.beta2
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RAdam::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RAdam::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        Ok(RAdam {
            config,
            step_count: 0,
            beta1_pow_t: 1.0,
            beta2_pow_t: 1.0,
            states: Vec::new(),
        })
    }

    pub fn config(&self) -> &RAdamConfig {
        &self.config
    }

    /// 学習率のみを書き換える（`AdamW::set_lr` と同じ意味論）。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値の場合は
    /// `AutodiffError::InvalidArgument`（fail-closed）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RAdam::set_lr: lr must be finite and >= 0.0, got {new_lr}"
            )));
        }
        self.config.lr = new_lr;
        Ok(())
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    /// `rmsprop.rs::RmsProp::step` と同じ 2 段構成（検証専用フェーズ→
    /// 状態変更フェーズ）を採る。
    pub fn step(
        &mut self,
        params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        if self.states.is_empty() {
            for (param, grad) in params_and_grads {
                if grad.shape() != param.shape() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: grad.shape().to_vec(),
                        rhs: param.shape().to_vec(),
                    }));
                }
            }
        } else {
            if params_and_grads.len() != self.states.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "RAdam::step: slot count changed across calls (expected {}, got {}); \
                     RAdam state (exp_avg/exp_avg_sq) is keyed by call-order slot index and \
                     cannot be resized after the first step()",
                    self.states.len(),
                    params_and_grads.len()
                )));
            }

            for (slot, (param, grad)) in self.states.iter().zip(params_and_grads.iter()) {
                if param.shape() != slot.shape.as_slice() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: param.shape().to_vec(),
                        rhs: slot.shape.clone(),
                    }));
                }
                if grad.shape() != param.shape() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: grad.shape().to_vec(),
                        rhs: param.shape().to_vec(),
                    }));
                }
            }
        }

        if self.states.is_empty() && !params_and_grads.is_empty() {
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| SlotState {
                    shape: param.shape().to_vec(),
                    exp_avg: vec![0.0f32; param.numel()],
                    exp_avg_sq: vec![0.0f32; param.numel()],
                })
                .collect();
        }

        self.step_count += 1;
        let step = self.step_count as f64;
        self.beta1_pow_t *= self.config.beta1 as f64;
        self.beta2_pow_t *= self.config.beta2 as f64;
        let bc1 = (1.0 - self.beta1_pow_t) as f32;
        let bc2 = 1.0 - self.beta2_pow_t;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;
        let decoupled = self.config.decoupled_weight_decay;

        // `rho_inf = 2/(1-beta2) - 1`・
        // `rho_t = rho_inf - 2*step*beta2^step/bc2`（f64。PyTorch の
        // Python float 演算を再現する。`beta2 as f64` の丸めが
        // `rho_t` の 5.0 近傍の判定を反転させないことは fixture 生成
        // スクリプトが `assert` 済み——`radam-pytorch-reference/
        // README.md` 参照）。
        let rho_inf = 2.0 / (1.0 - beta2 as f64) - 1.0;
        let rho_t = rho_inf - 2.0 * step * (beta2 as f64).powf(step) / bc2;

        // `rho_t > 5.0` の分岐でのみ rect・adaptive_lr の係数を用意する
        // （`rect` の分母は `rho_t > 5` の場合しか評価しないためゼロ
        // 除算は起きない）。
        let rect = if rho_t > 5.0 {
            Some(
                (((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                    .sqrt() as f32,
            )
        } else {
            None
        };
        let bc2_sqrt = (bc2.sqrt()) as f32;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let mut g = grad_data[i];
                let mut p = param_data[i];
                if weight_decay != 0.0 {
                    if decoupled {
                        p *= 1.0 - lr * weight_decay;
                    } else {
                        g = f32::mul_add(weight_decay, param_data[i], g);
                    }
                }

                // `exp_avg.lerp_(grad, 1-beta1)`（`rmsprop.rs` centered
                // 分岐と同じ 2 分岐形）。
                let start = slot.exp_avg[i];
                let end = g;
                let weight = one_minus_beta1;
                slot.exp_avg[i] = if weight.abs() < 0.5 {
                    start + weight * (end - start)
                } else {
                    end - (end - start) * beta1
                };

                // `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)`。
                slot.exp_avg_sq[i] = f32::mul_add(beta2, slot.exp_avg_sq[i], (1.0 - beta2) * g * g);

                let bias_corrected_exp_avg = slot.exp_avg[i] / bc1;

                p -= if let Some(rect) = rect {
                    // `adaptive_lr = sqrt(bc2) / (sqrt(exp_avg_sq)+eps)`。
                    let adaptive_lr = bc2_sqrt / (slot.exp_avg_sq[i].sqrt() + eps);
                    // `bias_corrected_exp_avg * lr * adaptive_lr * rect`
                    // （左結合の評価順）。
                    bias_corrected_exp_avg * lr * adaptive_lr * rect
                } else {
                    bias_corrected_exp_avg * lr
                };

                new_param.push(p);
            }

            out.push(Tensor::new(new_param, &slot.shape)?);
        }

        Ok(out)
    }
}

/// [`super::OptimizerStateDict`]（イシュー #2174。`state_dict` モジュール
/// 冒頭 doc「キー配置」節）。`kind = "radam"`・バッファ名
/// `exp_avg`／`exp_avg_sq`・`beta1_pow_t`／`beta2_pow_t` あり・
/// `mu_product` なし。検証本体は `super::state_dict::decode_state_dict`
/// へ委譲する薄い shim。
impl super::OptimizerStateDict for RAdam {
    fn state_dict(&self) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let mut out = HashMap::with_capacity(4 + self.states.len() * 2);
        out.insert(
            super::state_dict::marker_key("radam"),
            Tensor::new(vec![super::state_dict::FORMAT_VERSION], &[1])?,
        );
        out.insert(
            super::state_dict::STEP_COUNT_KEY.to_string(),
            super::state_dict::encode_u16x4_tensor(self.step_count)?,
        );
        out.insert(
            super::state_dict::BETA1_POW_T_KEY.to_string(),
            super::state_dict::encode_f64_tensor(self.beta1_pow_t)?,
        );
        out.insert(
            super::state_dict::BETA2_POW_T_KEY.to_string(),
            super::state_dict::encode_f64_tensor(self.beta2_pow_t)?,
        );
        for (i, slot) in self.states.iter().enumerate() {
            out.insert(
                super::state_dict::slot_key(i, "exp_avg"),
                Tensor::new(slot.exp_avg.clone(), &slot.shape)?,
            );
            out.insert(
                super::state_dict::slot_key(i, "exp_avg_sq"),
                Tensor::new(slot.exp_avg_sq.clone(), &slot.shape)?,
            );
        }
        Ok(out)
    }

    fn load_state_dict(
        &mut self,
        state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        let decoded = super::state_dict::decode_state_dict(
            "radam",
            &state,
            &["exp_avg", "exp_avg_sq"],
            true,
            true,
            false,
        )?;
        let states = decoded
            .slots
            .into_iter()
            .map(|(shape, mut buffers)| {
                let exp_avg = buffers.remove("exp_avg").unwrap_or_default();
                let exp_avg_sq = buffers.remove("exp_avg_sq").unwrap_or_default();
                SlotState {
                    shape,
                    exp_avg,
                    exp_avg_sq,
                }
            })
            .collect();
        self.step_count = decoded.step_count;
        self.beta1_pow_t = decoded.beta1_pow_t.unwrap_or(1.0);
        self.beta2_pow_t = decoded.beta2_pow_t.unwrap_or(1.0);
        self.states = states;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    #[test]
    fn rejects_negative_lr() {
        let cfg = RAdamConfig {
            lr: -1.0,
            ..RAdamConfig::default()
        };
        assert!(matches!(
            RAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_beta_out_of_range() {
        for cfg in [
            RAdamConfig {
                beta1: 1.0,
                ..RAdamConfig::default()
            },
            RAdamConfig {
                beta2: -0.1,
                ..RAdamConfig::default()
            },
        ] {
            assert!(matches!(
                RAdam::new(cfg),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = RAdamConfig {
            eps: 0.0,
            ..RAdamConfig::default()
        };
        assert!(matches!(
            RAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = RAdamConfig {
            weight_decay: f32::NAN,
            ..RAdamConfig::default()
        };
        assert!(matches!(
            RAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();
        let step_count_before = opt.step_count();

        let param2 = t(vec![1.0, 2.0], &[2]);
        let bad_grad = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &bad_grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
        assert_eq!(
            opt.step_count(),
            step_count_before,
            "shape エラー発生時に step_count が進んではならない"
        );

        let mut opt_ref = RAdam::new(RAdamConfig::default()).unwrap();
        opt_ref.step(&[(&param1, &grad1)]).unwrap();
        let param3 = t(vec![1.0, 2.0], &[2]);
        let grad3 = t(vec![0.1, 0.1], &[2]);
        let out_after_failed = opt.step(&[(&param3, &grad3)]).unwrap();
        let out_ref = opt_ref.step(&[(&param3, &grad3)]).unwrap();
        assert_eq!(
            crate::eval::dense_vec(&out_after_failed[0]),
            crate::eval::dense_vec(&out_ref[0])
        );
    }

    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();

        let bad_param = t(vec![1.0, 2.0], &[2]);
        let bad_grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let first = opt.step(&[(&bad_param, &bad_grad)]);
        assert!(matches!(first, Err(AutodiffError::Shape(_))));

        let param_a = t(vec![1.0], &[1]);
        let grad_a = t(vec![0.1], &[1]);
        let param_b = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad_b = t(vec![0.1, 0.1, 0.1], &[3]);
        let second = opt.step(&[(&param_a, &grad_a), (&param_b, &grad_b)]);
        assert!(
            second.is_ok(),
            "1 回目の失敗が state を汚染し 2 回目が誤って拒否された: {second:?}"
        );
        assert_eq!(
            opt.step_count(),
            1,
            "成功した step のみ step_count が進むべき"
        );
    }

    /// `rho_t <= 5` の step では `p - bias_corrected_exp_avg*lr` に
    /// 一致すること（実装計画 §3.5 固有テスト）。`beta2=0.999` は
    /// t=1 で `rho_t=1.0`（`<= 5`）。
    #[test]
    fn non_rectified_branch_matches_plain_bias_corrected_update() {
        let cfg = RAdamConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
        };
        let mut opt = RAdam::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let exp_avg = (1.0 - cfg.beta1) * g0;
        let bc1 = 1.0 - cfg.beta1;
        let bias_corrected_exp_avg = exp_avg / bc1;
        let expected = p0 - bias_corrected_exp_avg * cfg.lr;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "non-rectified 分岐との不一致: actual={actual} expected={expected}"
        );
    }

    /// `decoupled_weight_decay=true` かつ勾配が 0 のとき
    /// `p *= 1 - lr*wd` になること。
    #[test]
    fn decoupled_weight_decay_with_zero_grad_scales_param() {
        let cfg = RAdamConfig {
            lr: 0.1,
            weight_decay: 0.5,
            decoupled_weight_decay: true,
            ..RAdamConfig::default()
        };
        let mut opt = RAdam::new(cfg).unwrap();
        let param = t(vec![2.0], &[1]);
        let grad = t(vec![0.0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();
        // 勾配 0 なら bias_corrected_exp_avg も 0 のままなので更新項は
        // 消え、decoupled decay のみが残るはず（rho_t <= 5 の t=1 分岐）。
        let expected = 2.0 * (1.0 - cfg.lr * cfg.weight_decay);
        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "decoupled weight decay の適用結果が不一致: actual={actual} expected={expected}"
        );
    }

    // =========================================================================
    // set_lr
    // =========================================================================

    #[test]
    fn set_lr_updates_config_lr_and_keeps_other_fields() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        opt.set_lr(0.01).unwrap();
        let cfg = opt.config();
        assert_eq!(cfg.lr, 0.01);
        assert_eq!(cfg.beta1, RAdamConfig::default().beta1);
        assert_eq!(cfg.beta2, RAdamConfig::default().beta2);
    }

    #[test]
    fn set_lr_rejects_negative_and_non_finite() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        for bad in [-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = opt.set_lr(bad).unwrap_err();
            assert!(matches!(err, AutodiffError::InvalidArgument(_)));
            assert_eq!(opt.config().lr, RAdamConfig::default().lr);
        }
    }

    #[test]
    fn set_lr_does_not_reset_step_count_or_moments() {
        let mut opt = RAdam::new(RAdamConfig::default()).unwrap();
        let param = t(vec![0.5], &[1]);
        let grad = t(vec![0.3], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        assert_eq!(opt.step_count(), 1);

        opt.set_lr(0.0001).unwrap();
        assert_eq!(opt.step_count(), 1);
    }
}
