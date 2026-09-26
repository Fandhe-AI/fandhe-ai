//! NAdam（Dozat, 2016。`torch.optim.NAdam` 相当）。
//!
//! `torch.optim.NAdam` の単一テンソル実装（`_single_tensor_nadam`。
//! `bc2 = 1 - beta2^step` → weight decay（`decoupled_weight_decay` なら
//! `param *= 1 - lr*wd`、それ以外は `grad += wd*param`）→
//! `mu = beta1*(1 - 0.5*0.96^(step*momentum_decay))`・
//! `mu_next = beta1*(1 - 0.5*0.96^((step+1)*momentum_decay))` →
//! `mu_product *= mu` → `exp_avg.lerp_(grad, 1-beta1)` →
//! `exp_avg_sq = beta2*v + (1-beta2)*grad^2` →
//! `denom = sqrt(exp_avg_sq / bc2) + eps` →
//! `param += (-lr*(1-mu)/(1-mu_product)) * grad / denom` →
//! `param += (-lr*mu_next/(1-mu_product*mu_next)) * exp_avg / denom`
//! の演算順）と同一系列を再現する（イシュー #2171・親 #2131。参照値は
//! `tests/fixtures/nadam-pytorch-reference/nadam_reference.json`。実
//! PyTorch 2.14.0+cpu 実行値。README 参照）。
//!
//! `adamw.rs`（[`super::AdamW`]）を鏡写しにした別実装であり、内部
//! ループの共通化は行わない。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／
//! `BackendOps` に一切依存しない値型・純関数であり、新規 `Op`／
//! `BackendOps` メソッド／VJP は追加していない（カーネルなし）。
//!
//! # `mu_product` の扱い
//!
//! PyTorch は `mu_product` を `_get_scalar_dtype()`（既定 `float32`）の
//! **パラメータごとの** state として持つが、`mu`（`beta1`／
//! `momentum_decay`／`step` のみに依存し `param`／`grad` を参照しない）
//! の逐次積であるため、同一 optimizer インスタンス内の全スロットで値が
//! 一致する。本実装はこれを踏まえ `mu_product` を optimizer 全体で 1 個
//! （`f32`）持つ（`beta1_pow_t`／`beta2_pow_t` を `f64` で持つ
//! `adamw.rs` とは異なり、PyTorch の state dtype に合わせて `f32` で
//! 持つ）。

use std::collections::HashMap;

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.NAdam` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NAdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// momentum cache `mu` の減衰係数（`0.96^(step*momentum_decay)`）。
    /// 有限かつ `>= 0.0` を [`NAdam::new`] が検証する。
    pub momentum_decay: f32,
    /// `true` のとき decoupled weight decay（`param *= 1 - lr*wd`）を
    /// 使う。`false`（既定）は coupled（`grad += wd*param`）。
    pub decoupled_weight_decay: bool,
}

impl Default for NAdamConfig {
    fn default() -> NAdamConfig {
        NAdamConfig {
            lr: 2e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum_decay: 4e-3,
            decoupled_weight_decay: false,
        }
    }
}

/// パラメータスロットごとの状態。`mu_product` はモジュール冒頭 doc の
/// 通り optimizer 全体で共有するため本構造体には含めない。
struct SlotState {
    shape: Vec<usize>,
    exp_avg: Vec<f32>,
    exp_avg_sq: Vec<f32>,
}

/// NAdam optimizer 本体。
pub struct NAdam {
    config: NAdamConfig,
    step_count: u64,
    beta2_pow_t: f64,
    /// モジュール冒頭 doc「`mu_product` の扱い」参照。PyTorch の
    /// `_get_scalar_dtype()`（既定 `float32`）に合わせ `f32` で持つ。
    mu_product: f32,
    states: Vec<SlotState>,
}

impl NAdam {
    /// ハイパーパラメータを検証して構築する。
    pub fn new(config: NAdamConfig) -> Result<NAdam, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::new: beta1 must be in [0.0, 1.0), got {}",
                config.beta1
            )));
        }
        if !(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::new: beta2 must be in [0.0, 1.0), got {}",
                config.beta2
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        if !(config.momentum_decay.is_finite() && config.momentum_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::new: momentum_decay must be finite and >= 0.0, got {}",
                config.momentum_decay
            )));
        }
        Ok(NAdam {
            config,
            step_count: 0,
            beta2_pow_t: 1.0,
            mu_product: 1.0,
            states: Vec::new(),
        })
    }

    pub fn config(&self) -> &NAdamConfig {
        &self.config
    }

    /// 学習率のみを書き換える（`AdamW::set_lr` と同じ意味論。他の
    /// ハイパーパラメータ・`state`〈`step_count`／`beta2_pow_t`／
    /// `mu_product`／`exp_avg`／`exp_avg_sq`〉は不変のまま保つ）。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値の場合は
    /// `AutodiffError::InvalidArgument`（fail-closed）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "NAdam::set_lr: lr must be finite and >= 0.0, got {new_lr}"
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
                    "NAdam::step: slot count changed across calls (expected {}, got {}); \
                     NAdam state (exp_avg/exp_avg_sq) is keyed by call-order slot index and \
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
        let step = self.step_count;
        self.beta2_pow_t *= self.config.beta2 as f64;
        let bc2 = (1.0 - self.beta2_pow_t) as f32;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let eps = self.config.eps;
        let momentum_decay = self.config.momentum_decay as f64;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;
        let decoupled = self.config.decoupled_weight_decay;

        // `mu = beta1*(1 - 0.5*0.96^(step*momentum_decay))`・
        // `mu_next = beta1*(1 - 0.5*0.96^((step+1)*momentum_decay))`。
        // PyTorch は Python float（f64）でこの式を評価するため、
        // `powf` を使い f64 で計算してから f32 へ落とす（`adamw.rs` の
        // 逐次積〈`beta.powi`〉とは異なり、`momentum_decay` が非整数の
        // 実数指数を要するため）。
        let mu = beta1 as f64 * (1.0 - 0.5 * 0.96f64.powf(step as f64 * momentum_decay));
        let mu_next = beta1 as f64 * (1.0 - 0.5 * 0.96f64.powf((step + 1) as f64 * momentum_decay));
        self.mu_product *= mu as f32;
        let mu_product = self.mu_product;
        let mu_product_next = mu_product * mu_next as f32;

        let coef_grad = -lr * (1.0 - mu as f32) / (1.0 - mu_product);
        let coef_exp_avg = -lr * mu_next as f32 / (1.0 - mu_product_next);

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
                        // `param.mul_(1 - lr*wd)`（勾配は変更しない）。
                        p *= 1.0 - lr * weight_decay;
                    } else {
                        // `grad = grad.add(param, alpha=weight_decay)`。
                        g = f32::mul_add(weight_decay, param_data[i], g);
                    }
                }

                // `exp_avg.lerp_(grad, 1-beta1)`（`rmsprop.rs` centered
                // 分岐と同じ 2 分岐形。桁落ち対策の理由も同じ）。
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

                // `denom = exp_avg_sq.div(bc2).sqrt()` → `.add_(eps)`
                // （割ってから sqrt、eps は sqrt の後）。
                let denom = (slot.exp_avg_sq[i] / bc2).sqrt() + eps;

                // 2 つの addcdiv を別々に適用する
                // （`param.addcdiv_(grad, denom, value=coef_grad)` →
                // `param.addcdiv_(exp_avg, denom, value=coef_exp_avg)`）。
                p += coef_grad * g / denom;
                p += coef_exp_avg * slot.exp_avg[i] / denom;

                new_param.push(p);
            }

            out.push(Tensor::new(new_param, &slot.shape)?);
        }

        Ok(out)
    }
}

/// [`super::OptimizerStateDict`]（イシュー #2174。`state_dict` モジュール
/// 冒頭 doc「キー配置」節）。`kind = "nadam"`・バッファ名
/// `exp_avg`／`exp_avg_sq`・`beta2_pow_t`／`mu_product` あり
/// （`beta1_pow_t` なし。NAdam は `beta1` の bias correction を
/// `mu_product` で表現する）。検証本体は
/// `super::state_dict::decode_state_dict` へ委譲する薄い shim。
impl super::OptimizerStateDict for NAdam {
    fn state_dict(&self) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let mut out = HashMap::with_capacity(4 + self.states.len() * 2);
        out.insert(
            super::state_dict::marker_key("nadam"),
            Tensor::new(vec![super::state_dict::FORMAT_VERSION], &[1])?,
        );
        out.insert(
            super::state_dict::STEP_COUNT_KEY.to_string(),
            super::state_dict::encode_u16x4_tensor(self.step_count)?,
        );
        out.insert(
            super::state_dict::BETA2_POW_T_KEY.to_string(),
            super::state_dict::encode_f64_tensor(self.beta2_pow_t)?,
        );
        out.insert(
            super::state_dict::MU_PRODUCT_KEY.to_string(),
            Tensor::new(vec![self.mu_product], &[1])?,
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
            "nadam",
            &state,
            &["exp_avg", "exp_avg_sq"],
            false,
            true,
            true,
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
        self.beta2_pow_t = decoded.beta2_pow_t.unwrap_or(1.0);
        self.mu_product = decoded.mu_product.unwrap_or(1.0);
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
        let cfg = NAdamConfig {
            lr: -1.0,
            ..NAdamConfig::default()
        };
        assert!(matches!(
            NAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_beta_out_of_range() {
        for cfg in [
            NAdamConfig {
                beta1: 1.0,
                ..NAdamConfig::default()
            },
            NAdamConfig {
                beta2: -0.1,
                ..NAdamConfig::default()
            },
        ] {
            assert!(matches!(
                NAdam::new(cfg),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = NAdamConfig {
            eps: 0.0,
            ..NAdamConfig::default()
        };
        assert!(matches!(
            NAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_momentum_decay() {
        let cfg = NAdamConfig {
            momentum_decay: -0.1,
            ..NAdamConfig::default()
        };
        assert!(matches!(
            NAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = NAdamConfig {
            weight_decay: f32::NAN,
            ..NAdamConfig::default()
        };
        assert!(matches!(
            NAdam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
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
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
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

        let mut opt_ref = NAdam::new(NAdamConfig::default()).unwrap();
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
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();

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

    /// `decoupled_weight_decay=true` かつ勾配が 0 のとき、
    /// `p *= 1 - lr*wd` になること（実装計画 §3.5 固有テスト）。
    #[test]
    fn decoupled_weight_decay_with_zero_grad_scales_param() {
        let cfg = NAdamConfig {
            lr: 0.1,
            weight_decay: 0.5,
            decoupled_weight_decay: true,
            ..NAdamConfig::default()
        };
        let mut opt = NAdam::new(cfg).unwrap();
        let param = t(vec![2.0], &[1]);
        let grad = t(vec![0.0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();
        // 勾配 0 なら exp_avg・exp_avg_sq も 0 のままなので addcdiv 項は
        // 消え、decoupled decay のみが残るはず。
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
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
        opt.set_lr(0.01).unwrap();
        let cfg = opt.config();
        assert_eq!(cfg.lr, 0.01);
        assert_eq!(cfg.beta1, NAdamConfig::default().beta1);
        assert_eq!(cfg.beta2, NAdamConfig::default().beta2);
        assert_eq!(cfg.momentum_decay, NAdamConfig::default().momentum_decay);
    }

    #[test]
    fn set_lr_rejects_negative_and_non_finite() {
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
        for bad in [-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = opt.set_lr(bad).unwrap_err();
            assert!(matches!(err, AutodiffError::InvalidArgument(_)));
            assert_eq!(opt.config().lr, NAdamConfig::default().lr);
        }
    }

    #[test]
    fn set_lr_does_not_reset_step_count_or_moments() {
        let mut opt = NAdam::new(NAdamConfig::default()).unwrap();
        let param = t(vec![0.5], &[1]);
        let grad = t(vec![0.3], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        assert_eq!(opt.step_count(), 1);

        opt.set_lr(0.001).unwrap();
        assert_eq!(opt.step_count(), 1);
    }
}
