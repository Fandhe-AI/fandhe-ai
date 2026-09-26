//! Adadelta（Zeiler, 2012。`torch.optim.Adadelta` 相当）。
//!
//! `torch.optim.Adadelta` の単一テンソル実装（`_single_tensor_adadelta`。
//! `weight_decay != 0` なら `grad = grad + weight_decay * param` →
//! `square_avg = rho * square_avg + (1-rho) * grad^2` →
//! `std = sqrt(square_avg + eps)`（**eps は sqrt の内側**。
//! `RmsProp`〈`rmsprop.rs`〉は sqrt の後に eps を加算する点で異なる）→
//! `delta = sqrt(acc_delta + eps) / std * grad` →
//! `acc_delta = rho * acc_delta + (1-rho) * delta^2` →
//! `param -= lr * delta` の演算順）と同一系列を再現する（イシュー
//! #2171・親 #2131「PyTorch／TF 置き換えの API 網羅」。参照値は
//! `tests/fixtures/adadelta-pytorch-reference/adadelta_reference.json`。
//! 実 PyTorch 2.14.0+cpu 実行値。README 参照）。
//!
//! `rmsprop.rs`（[`super::RmsProp`]）を鏡写しにした別実装であり、内部
//! ループの共通化は行わない（既存 fixture テストの統一複合判定では
//! 共通化による bit ドリフトを検出できないため。`rmsprop.rs` 冒頭 doc
//! と同じ理由）。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／
//! `BackendOps` に一切依存しない値型・純関数であり、新規 `Op`／
//! `BackendOps` メソッド／VJP は追加していない（カーネルなし）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.Adadelta` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdadeltaConfig {
    /// 学習率（`lr`）。有限かつ `>= 0.0` を [`Adadelta::new`] が検証する。
    pub lr: f32,
    /// 二乗移動平均の減衰率（`rho`）。有限かつ `[0.0, 1.0)` を
    /// [`Adadelta::new`] が検証する（`rho = 1.0` は `square_avg`／
    /// `acc_delta` が一切更新されない退化ケースのため意図的に拒否する。
    /// `RmsProp::new` の `alpha` 検査と同じ理由）。
    pub rho: f32,
    /// ゼロ除算防止項（`eps`）。`sqrt` の**内側**に加算する（PyTorch
    /// と同順。`RmsProp` の「sqrt の後」とは逆）。有限かつ `> 0.0` を
    /// [`Adadelta::new`] が検証する。
    pub eps: f32,
    /// L2 正則化係数。PyTorch Adadelta と同じ coupled 方式（勾配へ
    /// `weight_decay * param` を加算してから以降の更新式へ渡す）。
    /// 有限かつ `>= 0.0` を [`Adadelta::new`] が検証する。
    pub weight_decay: f32,
}

impl Default for AdadeltaConfig {
    fn default() -> AdadeltaConfig {
        AdadeltaConfig {
            lr: 1.0,
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの状態。初回
/// `step()` 呼び出しで渡された `param` の shape から遅延初期化する
/// （`Adadelta::new` の時点ではパラメータ数・shape を知らないため。
/// `rmsprop.rs::SlotState` と同じ理由）。
struct SlotState {
    shape: Vec<usize>,
    square_avg: Vec<f32>,
    acc_delta: Vec<f32>,
}

/// Adadelta optimizer 本体。ハイパーパラメータ（[`AdadeltaConfig`]）と、
/// step 数・スロットごとの状態（`SlotState`）を保持する。Adam 系と異な
/// り step 数に依存するバイアス補正はない（`step_count` は診断用に保持
/// するのみ）。
pub struct Adadelta {
    config: AdadeltaConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

impl Adadelta {
    /// ハイパーパラメータを検証して構築する。`lr`/`weight_decay` は
    /// 有限かつ非負、`rho` は有限かつ `[0, 1)`（`rho = 1` は
    /// `square_avg`/`acc_delta` が一切更新されない退化ケースのため
    /// 意図的に拒否する）、`eps` は有限かつ正（0 を許すと初回 step で
    /// `std`/`delta` の分母がゼロになりうるため構築不可能な引数として
    /// 弾く）。
    pub fn new(config: AdadeltaConfig) -> Result<Adadelta, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adadelta::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.rho.is_finite() && (0.0..1.0).contains(&config.rho)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adadelta::new: rho must be in [0.0, 1.0), got {}",
                config.rho
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adadelta::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adadelta::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        Ok(Adadelta {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &AdadeltaConfig {
        &self.config
    }

    /// 学習率のみを書き換える（LR scheduler 連携用。`AdamW::set_lr`・
    /// `RmsProp` と同じ意味論。`rho`／`eps`／`weight_decay` は不変の
    /// まま保つ）。`state`（`square_avg`／`acc_delta`）・`step_count`
    /// はリセットしない。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値の場合は
    /// `AutodiffError::InvalidArgument`（[`Adadelta::new`] の `lr` 検査
    /// と同一基準。fail-closed）。検証失敗時は `self.config` を変更
    /// しない。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adadelta::set_lr: lr must be finite and >= 0.0, got {new_lr}"
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
    /// `rmsprop.rs::RmsProp::step` と同じ 2 段構成を採る: 副作用
    /// （状態バッファの更新）を一切加えない検証専用フェーズで全スロット
    /// の shape を確認しきってから、状態変更フェーズへ進む（形状エラー
    /// 発生時に状態バッファが部分更新されたまま残ると、後続の成功する
    /// step が破損した状態から学習してしまうため）。
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
                    "Adadelta::step: slot count changed across calls (expected {}, got {}); \
                     Adadelta state (square_avg/acc_delta) is keyed by call-order slot index \
                     and cannot be resized after the first step()",
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
                    square_avg: vec![0.0f32; param.numel()],
                    acc_delta: vec![0.0f32; param.numel()],
                })
                .collect();
        }

        self.step_count += 1;

        let rho = self.config.rho;
        let one_minus_rho = 1.0 - rho;
        let lr = self.config.lr;
        let eps = self.config.eps;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let mut g = grad_data[i];
                // `grad = grad.add(param, alpha=weight_decay)`
                // （weight_decay == 0 のときは演算自体を skip する。
                // RMSprop と同じ coupled L2 方式）。
                if self.config.weight_decay != 0.0 {
                    g = f32::mul_add(self.config.weight_decay, param_data[i], g);
                }

                // `square_avg.mul_(rho).addcmul_(grad, grad, value=1-rho)`。
                slot.square_avg[i] = f32::mul_add(rho, slot.square_avg[i], one_minus_rho * g * g);

                // `std = square_avg.add(eps).sqrt_()`（eps は sqrt の
                // **内側**）。
                let std = (slot.square_avg[i] + eps).sqrt();
                // `delta = acc_delta.add(eps).sqrt_()` →
                // `delta.div_(std).mul_(grad)`。
                let delta = (slot.acc_delta[i] + eps).sqrt() / std * g;

                // `acc_delta.mul_(rho).addcmul_(delta, delta, value=1-rho)`。
                slot.acc_delta[i] =
                    f32::mul_add(rho, slot.acc_delta[i], one_minus_rho * delta * delta);

                // `param.add_(delta, alpha=-lr)`。
                new_param.push(param_data[i] - lr * delta);
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

    #[test]
    fn rejects_negative_lr() {
        let cfg = AdadeltaConfig {
            lr: -1.0,
            ..AdadeltaConfig::default()
        };
        assert!(matches!(
            Adadelta::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_rho_out_of_range() {
        let cfg = AdadeltaConfig {
            rho: 1.0,
            ..AdadeltaConfig::default()
        };
        assert!(matches!(
            Adadelta::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let cfg = AdadeltaConfig {
            rho: -0.1,
            ..AdadeltaConfig::default()
        };
        assert!(matches!(
            Adadelta::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = AdadeltaConfig {
            eps: 0.0,
            ..AdadeltaConfig::default()
        };
        assert!(matches!(
            Adadelta::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_weight_decay() {
        let cfg = AdadeltaConfig {
            weight_decay: -0.1,
            ..AdadeltaConfig::default()
        };
        assert!(matches!(
            Adadelta::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = AdadeltaConfig {
            weight_decay: f32::NAN,
            ..AdadeltaConfig::default()
        };
        assert!(matches!(
            Adadelta::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    /// `rmsprop.rs::state_not_mutated_after_failed_step` と同型の回帰
    /// テスト: 形状エラーで `step` が失敗した場合、`step_count`／
    /// `square_avg`／`acc_delta` のいずれも部分更新されず呼び出し前の
    /// 状態のまま残ることを確認する。
    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
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

        let mut opt_ref = Adadelta::new(AdadeltaConfig::default()).unwrap();
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

    /// codex-review 指摘の回帰テスト（`rmsprop.rs` 先例と同型）: 初回
    /// `step()` が shape 不一致で失敗しても `self.states` が確定して
    /// はならない。
    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();

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

    /// `lr = 0` のとき更新がないこと（実装計画 §3.5「Adadelta 固有
    /// テスト」）。
    #[test]
    fn zero_lr_does_not_update_param() {
        let cfg = AdadeltaConfig {
            lr: 0.0,
            ..AdadeltaConfig::default()
        };
        let mut opt = Adadelta::new(cfg).unwrap();
        let param = t(vec![1.0, -2.0], &[2]);
        let grad = t(vec![0.3, -0.4], &[2]);
        let out = opt.step(&[(&param, &grad)]).unwrap();
        assert_eq!(out[0].get(&[0]).unwrap(), 1.0);
        assert_eq!(out[0].get(&[1]).unwrap(), -2.0);
    }

    /// t=1 の閉形式との一致を固定する（`acc_delta` は初回 step 前は
    /// 0 のため `delta = sqrt(eps) / sqrt((1-rho)*g^2 + eps) * g`）。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = AdadeltaConfig {
            lr: 0.5,
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
        };
        let mut opt = Adadelta::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let square_avg = (1.0 - cfg.rho) * g0 * g0;
        let std = (square_avg + cfg.eps).sqrt();
        let delta = (0.0f32 + cfg.eps).sqrt() / std * g0;
        let expected = p0 - cfg.lr * delta;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "閉形式との不一致: actual={actual} expected={expected}"
        );
    }

    // =========================================================================
    // set_lr（AdamW/RmsProp と同型の学習率更新 API）
    // =========================================================================

    #[test]
    fn set_lr_updates_config_lr_and_keeps_other_fields() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        opt.set_lr(0.2).unwrap();
        let cfg = opt.config();
        assert_eq!(cfg.lr, 0.2);
        assert_eq!(cfg.rho, AdadeltaConfig::default().rho);
        assert_eq!(cfg.eps, AdadeltaConfig::default().eps);
        assert_eq!(cfg.weight_decay, AdadeltaConfig::default().weight_decay);
    }

    #[test]
    fn set_lr_rejects_negative_and_non_finite() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        for bad in [-0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = opt.set_lr(bad).unwrap_err();
            assert!(matches!(err, AutodiffError::InvalidArgument(_)));
            assert_eq!(opt.config().lr, AdadeltaConfig::default().lr);
        }
    }

    #[test]
    fn set_lr_does_not_reset_step_count_or_state() {
        let mut opt = Adadelta::new(AdadeltaConfig::default()).unwrap();
        let param = t(vec![0.5], &[1]);
        let grad = t(vec![0.3], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        assert_eq!(opt.step_count(), 1);

        opt.set_lr(0.1).unwrap();
        assert_eq!(opt.step_count(), 1);
    }
}
