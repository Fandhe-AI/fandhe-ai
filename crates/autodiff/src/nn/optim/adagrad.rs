//! Adagrad（Duchi et al., 2011。`torch.optim.Adagrad` 相当）。
//!
//! `torch.optim.Adagrad` の単一テンソル実装（`_single_tensor_adagrad`。
//! `step += 1` → `weight_decay != 0` のとき
//! `grad = grad + weight_decay * param`（coupled L2） →
//! `clr = lr / (1 + (step-1) * lr_decay)` →
//! `state_sum += grad^2` → `std = sqrt(state_sum) + eps` →
//! `param -= clr * grad / std` の演算順）と同一系列を再現する
//! （イシュー #1743・親 #1610「optimizer（Adam／RMSprop／Adagrad／
//! LAMB）」。受け入れ条件の読み替えは `nn/optim/mod.rs` 冒頭 doc・
//! `rmsprop.rs` 末尾の注記を参照）。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。呼び出し元
//! （学習ループ）は `Linear::from_parameters` 等で層を再構築する
//! 既存の不変更新パターン（`tests/nn_train_convergence.rs`）にそのまま
//! 差し込める。
//!
//! `adamw.rs`（[`super::AdamW`]）・`rmsprop.rs`（[`super::RmsProp`]）を
//! 鏡写しにした別実装であり、内部ループの共通化は行わない（既存
//! fixture テストの統一複合判定では共通化による bit ドリフトを検出
//! できないため。イシュー #1743 実装計画 §3.2）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.Adagrad` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdagradConfig {
    pub lr: f32,
    pub lr_decay: f32,
    pub weight_decay: f32,
    pub initial_accumulator_value: f32,
    pub eps: f32,
}

impl Default for AdagradConfig {
    fn default() -> AdagradConfig {
        AdagradConfig {
            lr: 1e-2,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの累積二乗和
/// 状態。初回 `step()` 呼び出しで渡された `param` の shape から
/// 遅延初期化する（`AdamW::SlotState`／`RmsProp::SlotState` と同じ
/// 理由）。PyTorch は `state["sum"] = torch.full_like(param,
/// initial_accumulator_value)` で初期化する（0 スタートではない）。
struct SlotState {
    shape: Vec<usize>,
    state_sum: Vec<f32>,
}

/// Adagrad optimizer 本体。ハイパーパラメータ（[`AdagradConfig`]）と、
/// step 数・スロットごとの状態（`SlotState`）を保持する。
pub struct Adagrad {
    config: AdagradConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

impl Adagrad {
    /// ハイパーパラメータを検証して構築する。`lr`/`weight_decay`/
    /// `lr_decay`/`initial_accumulator_value` は有限かつ非負、`eps`
    /// は有限かつ正（PyTorch は `eps >= 0` を許すが、0 は初回 step で
    /// ゼロ除算になるため `AdamW::new`／`RmsProp::new` と同様に
    /// 構築不可能な引数として弾く。意図的な差異）。
    pub fn new(config: AdagradConfig) -> Result<Adagrad, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adagrad::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.lr_decay.is_finite() && config.lr_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adagrad::new: lr_decay must be finite and >= 0.0, got {}",
                config.lr_decay
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adagrad::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        if !(config.initial_accumulator_value.is_finite()
            && config.initial_accumulator_value >= 0.0)
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adagrad::new: initial_accumulator_value must be finite and >= 0.0, got {}",
                config.initial_accumulator_value
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adagrad::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        Ok(Adagrad {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    pub fn config(&self) -> &AdagradConfig {
        &self.config
    }

    /// 実行済み `step()` 回数。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// `adamw.rs::AdamW::step`／`rmsprop.rs::RmsProp::step` と同じ
    /// 2 段構成を採る: 副作用（状態バッファの更新）を一切加えない
    /// 検証専用フェーズで全スロットの shape を確認しきってから、
    /// 状態変更フェーズへ進む（形状エラー発生時に `step_count`／
    /// `state_sum` が部分更新されたまま残ると、後続の成功する step が
    /// 破損した状態から学習してしまうため）。
    pub fn step(
        &mut self,
        params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        if self.states.is_empty() && !params_and_grads.is_empty() {
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| SlotState {
                    shape: param.shape().to_vec(),
                    state_sum: vec![self.config.initial_accumulator_value; param.numel()],
                })
                .collect();
        }

        if params_and_grads.len() != self.states.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adagrad::step: slot count changed across calls (expected {}, got {}); \
                 Adagrad state (state_sum) is keyed by call-order slot index and cannot \
                 be resized after the first step()",
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

        self.step_count += 1;
        let step = self.step_count;

        // `clr = lr / (1 + (step - 1) * lr_decay)`。PyTorch の
        // `bias_correction` 等と同様 f64 で計算してから `f32` へ
        // 1 回 downcast する（`AdamW::step` の bias correction と同じ
        // 理由: `step` 由来の丸めを PyTorch の Python float 演算に
        // 寄せる）。
        let clr = (self.config.lr as f64 / (1.0 + (step - 1) as f64 * self.config.lr_decay as f64))
            as f32;
        let eps = self.config.eps;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            // `param`/`grad` は読み取り専用の走査のみ（`state_sum`・
            // new_param は別バッファへ積む）なので、contiguous 入力に
            // 対する不要コピーを避ける `dense_vec_ref`（`Cow<[f32]>`）
            // を使う（`adamw.rs::AdamW::step` と同じ変更。イシュー
            // #1026）。
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let mut g = grad_data[i];
                // PyTorch: `grad = grad.add(param, alpha=weight_decay)`
                // （weight_decay == 0 のときは演算自体を skip する。
                // coupled L2 方式は `RmsProp::step` と同じ）。
                if self.config.weight_decay != 0.0 {
                    g = f32::mul_add(self.config.weight_decay, param_data[i], g);
                }

                // `state_sum.addcmul_(g, g, value=1)`。
                slot.state_sum[i] = f32::mul_add(g, g, slot.state_sum[i]);
                let std = slot.state_sum[i].sqrt() + eps;

                // `param.addcdiv_(g, std, value=-clr)` は
                // `param + (-clr) * g / std` を ATen が左から
                // `value*t1/t2` の順に評価する（`AdamW`／`RmsProp` の
                // 括りと同じ）。
                new_param.push(param_data[i] - (clr * g) / std);
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
        let cfg = AdagradConfig {
            lr: -1.0,
            ..AdagradConfig::default()
        };
        assert!(matches!(
            Adagrad::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_lr_decay() {
        let cfg = AdagradConfig {
            lr_decay: -0.1,
            ..AdagradConfig::default()
        };
        assert!(matches!(
            Adagrad::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_initial_accumulator_value() {
        let cfg = AdagradConfig {
            initial_accumulator_value: -0.1,
            ..AdagradConfig::default()
        };
        assert!(matches!(
            Adagrad::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = AdagradConfig {
            eps: 0.0,
            ..AdagradConfig::default()
        };
        assert!(matches!(
            Adagrad::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = AdagradConfig {
            weight_decay: f32::NAN,
            ..AdagradConfig::default()
        };
        assert!(matches!(
            Adagrad::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Adagrad::new(AdagradConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Adagrad::new(AdagradConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Adagrad::new(AdagradConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    /// Bugbot 指摘（`AdamW`）と同型の回帰テスト: 形状エラーで `step`
    /// が失敗した場合、`step_count`／`state_sum` のいずれも部分更新
    /// されず呼び出し前の状態のまま残ることを確認する。
    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = Adagrad::new(AdagradConfig::default()).unwrap();
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

        let mut opt_ref = Adagrad::new(AdagradConfig::default()).unwrap();
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

    /// `lr_decay = 0` のとき `clr` が `lr` と厳密一致することを確認
    /// する（`clr = lr / (1 + (step-1)*lr_decay)` の分母が常に 1）。
    #[test]
    fn lr_decay_zero_keeps_clr_equal_to_lr() {
        let cfg = AdagradConfig {
            lr: 0.05,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        };
        let mut opt = Adagrad::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let state_sum = g0 * g0;
        let std = state_sum.sqrt() + cfg.eps;
        let expected = p0 - (cfg.lr * g0) / std;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "lr_decay=0 での clr==lr 前提が崩れている: actual={actual} expected={expected}"
        );
    }

    /// `initial_accumulator_value` が初期 `state_sum` に反映される
    /// ことを直接確認する。
    #[test]
    fn initial_accumulator_value_seeds_state_sum() {
        let cfg = AdagradConfig {
            lr: 0.05,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.5,
            eps: 1e-10,
        };
        let mut opt = Adagrad::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        // state_sum は initial_accumulator_value から始まる（0 では
        // ない）ので、std は sqrt(0.5 + g0^2) + eps になる。
        let state_sum = cfg.initial_accumulator_value + g0 * g0;
        let std = state_sum.sqrt() + cfg.eps;
        let expected = p0 - (cfg.lr * g0) / std;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "initial_accumulator_value が state_sum に反映されていない: \
             actual={actual} expected={expected}"
        );
    }

    /// t=1 の閉形式との一致を固定する。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = AdagradConfig {
            lr: 0.05,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        };
        let mut opt = Adagrad::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let state_sum = g0 * g0;
        let std = state_sum.sqrt() + cfg.eps;
        let expected = p0 - (cfg.lr * g0) / std;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "閉形式との不一致: actual={actual} expected={expected}"
        );
    }
}
