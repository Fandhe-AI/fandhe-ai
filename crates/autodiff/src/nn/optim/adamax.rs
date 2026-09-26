//! Adamax（Kingma & Ba, 2015 §7.1。`torch.optim.Adamax` 相当）。
//!
//! `torch.optim.Adamax` の単一テンソル実装（`_single_tensor_adamax`。
//! `weight_decay != 0` なら `grad = grad + weight_decay * param` →
//! `exp_avg.lerp_(grad, 1-beta1)` →
//! `exp_inf = max(exp_inf*beta2, |grad|+eps)`（`torch.maximum` は NaN
//! 伝播版。`f32::max` はどちらかが NaN のとき NaN を無視するため使え
//! ない）→ `clr = lr / (1 - beta1^step)` →
//! `param -= clr * exp_avg / exp_inf` の演算順）と同一系列を再現する
//! （イシュー #2171・親 #2131。参照値は
//! `tests/fixtures/adamax-pytorch-reference/adamax_reference.json`。
//! 実 PyTorch 2.14.0+cpu 実行値。README 参照）。
//!
//! `adamw.rs`（[`super::AdamW`]）を鏡写しにした別実装であり、内部
//! ループの共通化は行わない（既存 fixture テストの統一複合判定では
//! 共通化による bit ドリフトを検出できないため）。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／
//! `BackendOps` に一切依存しない値型・純関数であり、新規 `Op`／
//! `BackendOps` メソッド／VJP は追加していない（カーネルなし）。

use std::collections::HashMap;

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.Adamax` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamaxConfig {
    /// 学習率（`lr`）。有限かつ `>= 0.0` を [`Adamax::new`] が検証する。
    pub lr: f32,
    /// 1 次モーメントの指数移動平均係数（`beta1`）。有限かつ
    /// `[0.0, 1.0)` を [`Adamax::new`] が検証する。
    pub beta1: f32,
    /// 無限norm（`exp_inf`）の指数移動平均係数（`beta2`）。有限かつ
    /// `[0.0, 1.0)` を [`Adamax::new`] が検証する。
    pub beta2: f32,
    /// ゼロ除算防止項（`eps`）。`|grad| + eps` として `exp_inf` 更新に
    /// 織り込む（PyTorch と同順）。有限かつ `> 0.0` を [`Adamax::new`]
    /// が検証する。
    pub eps: f32,
    /// L2 正則化係数。PyTorch Adamax と同じ coupled 方式。有限かつ
    /// `>= 0.0` を [`Adamax::new`] が検証する。
    pub weight_decay: f32,
}

impl Default for AdamaxConfig {
    fn default() -> AdamaxConfig {
        AdamaxConfig {
            lr: 2e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの状態。初回
/// `step()` 呼び出しで渡された `param` の shape から遅延初期化する
/// （`adamw.rs::SlotState` と同じ理由）。
struct SlotState {
    shape: Vec<usize>,
    exp_avg: Vec<f32>,
    exp_inf: Vec<f32>,
}

/// Adamax optimizer 本体。ハイパーパラメータ（[`AdamaxConfig`]）と、
/// step 数・bias correction 用の `beta1^t` 逐次積（f64。PyTorch の
/// Python float は f64）・スロットごとの状態（`SlotState`）を保持する。
pub struct Adamax {
    config: AdamaxConfig,
    step_count: u64,
    beta1_pow_t: f64,
    states: Vec<SlotState>,
}

impl Adamax {
    /// ハイパーパラメータを検証して構築する。`lr`/`weight_decay` は
    /// 有限かつ非負、`beta1`/`beta2` は `[0, 1)`、`eps` は有限かつ正
    /// （0 を許すと `exp_inf` が 0 のスロット・step でゼロ除算になる
    /// ため構築不可能な引数として弾く）。
    pub fn new(config: AdamaxConfig) -> Result<Adamax, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adamax::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adamax::new: beta1 must be in [0.0, 1.0), got {}",
                config.beta1
            )));
        }
        if !(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adamax::new: beta2 must be in [0.0, 1.0), got {}",
                config.beta2
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adamax::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adamax::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        Ok(Adamax {
            config,
            step_count: 0,
            beta1_pow_t: 1.0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &AdamaxConfig {
        &self.config
    }

    /// 学習率のみを書き換える（LR scheduler 連携用。`AdamW::set_lr`と
    /// 同じ意味論。`beta1`／`beta2`／`eps`／`weight_decay`・`state`
    /// （`step_count`／`beta1_pow_t`／`exp_avg`／`exp_inf`）は不変の
    /// まま保つ）。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値の場合は
    /// `AutodiffError::InvalidArgument`（fail-closed。検証失敗時は
    /// `self.config` を変更しない）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adamax::set_lr: lr must be finite and >= 0.0, got {new_lr}"
            )));
        }
        self.config.lr = new_lr;
        Ok(())
    }

    /// 実行済み `step()` 回数（bias correction の `t`）。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// `rmsprop.rs::RmsProp::step` と同じ 2 段構成を採る: 副作用（状態
    /// バッファ・`step_count`／`beta1_pow_t` の更新）を一切加えない
    /// 検証専用フェーズで全スロットの shape を確認しきってから、状態
    /// 変更フェーズへ進む。
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
                    "Adamax::step: slot count changed across calls (expected {}, got {}); \
                     Adamax state (exp_avg/exp_inf) is keyed by call-order slot index and \
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
                    exp_inf: vec![0.0f32; param.numel()],
                })
                .collect();
        }

        self.step_count += 1;
        self.beta1_pow_t *= self.config.beta1 as f64;
        // `clr = lr / (1 - beta1^step)`。PyTorch は `bias_correction =
        // 1 - beta1 ** _get_value(step_t)`（Python float・f64）を経由
        // して `clr = lr / bias_correction` を計算するため、係数は f64
        // で計算し最後に f32 へ落とす（`adamw.rs::AdamW::step` の
        // `step_size` と同じ方針）。
        let bias_correction = 1.0 - self.beta1_pow_t;
        let clr = (self.config.lr as f64 / bias_correction) as f32;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let eps = self.config.eps;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let mut g = grad_data[i];
                // `grad = grad.add(param, alpha=weight_decay)`
                // （weight_decay == 0 のときは演算自体を skip する）。
                if self.config.weight_decay != 0.0 {
                    g = f32::mul_add(self.config.weight_decay, param_data[i], g);
                }

                // `exp_avg.lerp_(grad, 1-beta1)`。`rmsprop.rs` の
                // centered 分岐と同じ 2 分岐形（`|weight| < 0.5` なら
                // 始点基準、それ以外は終点基準。桁落ち対策の理由も同じ）。
                let start = slot.exp_avg[i];
                let end = g;
                let weight = one_minus_beta1;
                slot.exp_avg[i] = if weight.abs() < 0.5 {
                    start + weight * (end - start)
                } else {
                    end - (end - start) * beta1
                };

                // `exp_inf = max(exp_inf*beta2, |grad|+eps)`。
                // `torch.maximum` は NaN 伝播版（どちらかが NaN なら
                // NaN を返す）だが `f32::max` は NaN を無視するため、
                // NaN 伝播の意味論を明示的に自前実装する。
                let scaled_inf = beta2 * slot.exp_inf[i];
                let abs_grad_eps = g.abs() + eps;
                slot.exp_inf[i] = if scaled_inf.is_nan() || abs_grad_eps.is_nan() {
                    f32::NAN
                } else if scaled_inf >= abs_grad_eps {
                    scaled_inf
                } else {
                    abs_grad_eps
                };

                // `param.addcdiv_(exp_avg, exp_inf, value=-clr)`。
                new_param.push(param_data[i] - clr * slot.exp_avg[i] / slot.exp_inf[i]);
            }

            out.push(Tensor::new(new_param, &slot.shape)?);
        }

        Ok(out)
    }
}

/// [`super::OptimizerStateDict`]（イシュー #2174。`state_dict` モジュール
/// 冒頭 doc「キー配置」節）。`kind = "adamax"`・バッファ名
/// `exp_avg`／`exp_inf`・`beta1_pow_t` あり（`beta2_pow_t`／
/// `mu_product` なし。Adamax は `beta2`〈`exp_inf` の無限norm 更新〉に
/// bias correction を適用しない）。検証本体は
/// `super::state_dict::decode_state_dict` へ委譲する薄い shim。
impl super::OptimizerStateDict for Adamax {
    fn state_dict(&self) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let mut out = HashMap::with_capacity(3 + self.states.len() * 2);
        out.insert(
            super::state_dict::marker_key("adamax"),
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
        for (i, slot) in self.states.iter().enumerate() {
            out.insert(
                super::state_dict::slot_key(i, "exp_avg"),
                Tensor::new(slot.exp_avg.clone(), &slot.shape)?,
            );
            out.insert(
                super::state_dict::slot_key(i, "exp_inf"),
                Tensor::new(slot.exp_inf.clone(), &slot.shape)?,
            );
        }
        Ok(out)
    }

    fn load_state_dict(
        &mut self,
        state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        let decoded = super::state_dict::decode_state_dict(
            "adamax",
            &state,
            &["exp_avg", "exp_inf"],
            true,
            false,
            false,
        )?;
        let states = decoded
            .slots
            .into_iter()
            .map(|(shape, mut buffers)| {
                let exp_avg = buffers.remove("exp_avg").unwrap_or_default();
                let exp_inf = buffers.remove("exp_inf").unwrap_or_default();
                SlotState {
                    shape,
                    exp_avg,
                    exp_inf,
                }
            })
            .collect();
        self.step_count = decoded.step_count;
        self.beta1_pow_t = decoded.beta1_pow_t.unwrap_or(1.0);
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
        let cfg = AdamaxConfig {
            lr: -1.0,
            ..AdamaxConfig::default()
        };
        assert!(matches!(
            Adamax::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_beta_out_of_range() {
        for cfg in [
            AdamaxConfig {
                beta1: 1.0,
                ..AdamaxConfig::default()
            },
            AdamaxConfig {
                beta2: -0.1,
                ..AdamaxConfig::default()
            },
        ] {
            assert!(matches!(
                Adamax::new(cfg),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = AdamaxConfig {
            eps: 0.0,
            ..AdamaxConfig::default()
        };
        assert!(matches!(
            Adamax::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_weight_decay() {
        let cfg = AdamaxConfig {
            weight_decay: -0.1,
            ..AdamaxConfig::default()
        };
        assert!(matches!(
            Adamax::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = AdamaxConfig {
            weight_decay: f32::NAN,
            ..AdamaxConfig::default()
        };
        assert!(matches!(
            Adamax::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
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
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
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

        let mut opt_ref = Adamax::new(AdamaxConfig::default()).unwrap();
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
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();

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

    /// `exp_inf` の更新は NaN 伝播版 max であること（`torch.maximum`
    /// 相当。実装計画 §3.5「Adamax: NaN 伝播 max」）。`f32::max` の
    /// NaN 無視挙動へ回帰した場合に検出するテスト。
    #[test]
    fn exp_inf_update_propagates_nan() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let nan_grad = t(vec![f32::NAN], &[1]);
        let out = opt.step(&[(&param, &nan_grad)]).unwrap();
        assert!(
            out[0].get(&[0]).unwrap().is_nan(),
            "NaN 勾配は exp_inf 経由でパラメータへ NaN として伝播するはず"
        );
    }

    /// t=1 の閉形式との一致を固定する。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = AdamaxConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
        };
        let mut opt = Adamax::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let exp_avg = (1.0 - cfg.beta1) * g0;
        let exp_inf = g0.abs() + cfg.eps;
        let clr = cfg.lr / (1.0 - cfg.beta1);
        let expected = p0 - clr * exp_avg / exp_inf;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "閉形式との不一致: actual={actual} expected={expected}"
        );
    }

    // =========================================================================
    // set_lr
    // =========================================================================

    #[test]
    fn set_lr_updates_config_lr_and_keeps_other_fields() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        opt.set_lr(0.01).unwrap();
        let cfg = opt.config();
        assert_eq!(cfg.lr, 0.01);
        assert_eq!(cfg.beta1, AdamaxConfig::default().beta1);
        assert_eq!(cfg.beta2, AdamaxConfig::default().beta2);
        assert_eq!(cfg.weight_decay, AdamaxConfig::default().weight_decay);
    }

    #[test]
    fn set_lr_rejects_negative_and_non_finite() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        for bad in [-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = opt.set_lr(bad).unwrap_err();
            assert!(matches!(err, AutodiffError::InvalidArgument(_)));
            assert_eq!(opt.config().lr, AdamaxConfig::default().lr);
        }
    }

    #[test]
    fn set_lr_does_not_reset_step_count_or_moments() {
        let mut opt = Adamax::new(AdamaxConfig::default()).unwrap();
        let param = t(vec![0.5], &[1]);
        let grad = t(vec![0.3], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        assert_eq!(opt.step_count(), 1);

        opt.set_lr(0.001).unwrap();
        assert_eq!(opt.step_count(), 1);
    }
}
