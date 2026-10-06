//! Lion（Chen et al., 2023「Symbolic Discovery of Optimization Algorithms」）。
//!
//! 参照実装は google/automl `lion/lion_pytorch.py`（Apache-2.0）の更新則。
//! **`torch.optim` 2.14.0 に Lion は存在しない**（イシュー #2656・親
//! #2654 で確認）ため、参照値は同更新則を実 PyTorch 2.14.0+cpu のテンソル
//! 演算で実行した値
//! （`tests/fixtures/lion-pytorch-reference/lion_reference.json`。README
//! 参照）で、`torch.optim` の実装との突合ではない。独立性の補強として
//! `tests/nn_optim_lion.rs` に論文の式どおりの `f64` 参照実装を置く。
//!
//! 演算順（スロットごとの状態は `exp_avg`〈0 初期化〉）:
//!
//! ```text
//! param  *= (1 - lr*weight_decay)            # decoupled。wd=0 でも常に適用
//! update  = exp_avg*beta1 + grad*(1-beta1)   # 補間係数は beta1（更新前の exp_avg）
//! param  += sign(update) * (-lr)
//! exp_avg = exp_avg*beta2 + grad*(1-beta2)   # 保存する移動平均の係数は beta2
//! ```
//!
//! **`f32::signum` を使わない理由**: torch の `sign` は `±0.0` と NaN を 0
//! にする（`torch_sign`）。`f32::signum` は `+0.0` に 1.0・NaN に NaN を
//! 返す。**`update` の計算に `f32::mul_add` も `lerp` も使わない理由**:
//! torch は `exp_avg*beta1` と `grad*(1-beta1)` を別々のテンソル演算で丸め
//! てから加算する。`sign` は不連続で、相殺付近の 1 ulp 差がパラメータ差
//! `2*lr` の符号反転へ増幅されるため、ここだけは torch と同じ「積 2 回 →
//! 加算 1 回」の `f32` 演算列にする。
//!
//! **非有限入力の契約**（公式実装準拠で伝播させる）: NaN 勾配の要素は
//! `update = NaN` → `sign = 0` で当 step は動かず、`exp_avg` に NaN が残り
//! 以後その要素は weight decay 以外で動かなくなる。fixture の `edge`
//! ブロックで実測固定している。
//!
//! `step_count` は更新式には使わない（API 対称性と overflow 検査のため
//! 保持）。`nn/optim/mod.rs` の doc が示す通り `step()` は `(param, grad)`
//! の参照列を受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／
//! `BackendOps` に一切依存しない値型・純関数であり、新規 `Op`／`BackendOps`
//! メソッド／VJP は追加していない（カーネルなし）。`rprop.rs` を鏡写しに
//! した別実装で、内部ループは他 optimizer と共通化しない（`torch_sign` も
//! 意図的な重複）。`ParamGroupStep`／`OptimizerStateDict` は未実装
//! （決定記録 §10）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// 公式参照実装の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LionConfig {
    /// 学習率。有限かつ `>= 0.0`。
    pub lr: f32,
    /// 更新方向の補間係数。`[0, 1)`。
    pub beta1: f32,
    /// 移動平均の減衰係数。`[0, 1)`。
    pub beta2: f32,
    /// decoupled weight decay 係数。有限かつ `>= 0.0`（公式実装は無検査）。
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> LionConfig {
        LionConfig {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

/// torch の `sign`（`(0 < x) - (x < 0)`）。`±0.0` と NaN は `0.0`。
fn torch_sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

struct SlotState {
    shape: Vec<usize>,
    exp_avg: Vec<f32>,
}

/// Lion optimizer 本体。
pub struct Lion {
    config: LionConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

fn validate_hyperparameters(who: &str, c: &LionConfig) -> Result<(), AutodiffError> {
    if !(c.lr.is_finite() && c.lr >= 0.0) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: lr must be finite and >= 0.0, got {}",
            c.lr
        )));
    }
    for (name, v) in [("beta1", c.beta1), ("beta2", c.beta2)] {
        if !(v.is_finite() && (0.0..1.0).contains(&v)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}: {name} must be in [0, 1), got {v}"
            )));
        }
    }
    if !(c.weight_decay.is_finite() && c.weight_decay >= 0.0) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}: weight_decay must be finite and >= 0.0, got {}",
            c.weight_decay
        )));
    }
    Ok(())
}

impl Lion {
    /// ハイパーパラメータを検証して構築する。
    ///
    /// # Errors
    ///
    /// 非有限値・範囲外（`lr < 0`・`beta` が `[0, 1)` の外・
    /// `weight_decay < 0`）は `AutodiffError::InvalidArgument`。
    pub fn new(config: LionConfig) -> Result<Lion, AutodiffError> {
        validate_hyperparameters("Lion::new", &config)?;
        Ok(Lion {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &LionConfig {
        &self.config
    }

    /// `config.lr` のみを書き換える。次の `step()` から即時に効く。状態は
    /// 不変。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値なら `AutodiffError::InvalidArgument`
    /// （`self.config` は変更しない）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lion::set_lr: lr must be finite and >= 0.0, got {new_lr}"
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
        }
        if !self.states.is_empty() {
            if params_and_grads.len() != self.states.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Lion::step: slot count changed across calls (expected {}, got {}); \
                     Lion state (exp_avg) is keyed by call-order slot index and cannot be \
                     resized after the first step()",
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

        let next_step_count = self.step_count.checked_add(1).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Lion::step: step_count overflow: too many step() calls for this optimizer \
                 to advance further"
                    .to_string(),
            )
        })?;

        if self.states.is_empty() && !params_and_grads.is_empty() {
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| SlotState {
                    shape: param.shape().to_vec(),
                    exp_avg: vec![0.0f32; param.numel()],
                })
                .collect();
        }
        self.step_count = next_step_count;

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let one_minus_beta1 = complement_like_python(beta1);
        let one_minus_beta2 = complement_like_python(beta2);
        // `param.mul_(1 - lr*weight_decay)`: Python float（f64）で計算し f32 へ。
        let decay = (1.0 - lr as f64 * self.config.weight_decay as f64) as f32;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let g = grad_data[i];
                let decayed = param_data[i] * decay;
                // 積 2 回 → 加算 1 回（`mul_add` を使わない。モジュール doc 参照）。
                let update = slot.exp_avg[i] * beta1 + g * one_minus_beta1;
                new_param.push(decayed + torch_sign(update) * -lr);
                // `exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)`。
                slot.exp_avg[i] = f32::mul_add(g, one_minus_beta2, slot.exp_avg[i] * beta2);
            }

            out.push(Tensor::new(new_param, &slot.shape)?);
        }

        Ok(out)
    }
}

/// 補間係数 `1 - beta` を torch 参照実装（Python float〈f64〉の `1 - beta`）と同じ精度で求める。
///
/// `LionConfig` は `f32` で `beta` を保持するため、`1.0f32 - beta` や
/// `1.0 - beta as f64` は f32 化済みの `beta`（例 0.9f32 = 0.89999998...）を
/// 引いてしまい、`0.1` ではなく `0.100000024` になる。`sign` が不連続な Lion では
/// 相殺付近でこの差が更新方向の反転になる。利用者が指定した十進値を復元するため、
/// `f32` の最短往復表現を f64 として読み直してから `1 - beta` を f64 で計算し、
/// 最後に 1 回 f32 へ丸める（torch の `alpha` が kernel でスカラー型へ変換される挙動に相当）。
fn complement_like_python(beta: f32) -> f32 {
    let b64 = beta
        .to_string()
        .parse::<f64>()
        .unwrap_or_else(|_| f64::from(beta));
    (1.0 - b64) as f32
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

    #[test]
    fn torch_sign_matches_torch_semantics() {
        assert_eq!(torch_sign(2.0), 1.0);
        assert_eq!(torch_sign(-2.0), -1.0);
        assert_eq!(torch_sign(0.0), 0.0);
        assert_eq!(torch_sign(-0.0), 0.0);
        assert_eq!(torch_sign(f32::NAN), 0.0);
        assert_eq!(torch_sign(f32::INFINITY), 1.0);
        assert_eq!(torch_sign(f32::NEG_INFINITY), -1.0);
    }

    #[test]
    fn rejects_invalid_hyperparameters() {
        let d = LionConfig::default();
        let bad = [
            LionConfig { lr: -1.0, ..d },
            LionConfig { lr: f32::NAN, ..d },
            LionConfig {
                lr: f32::INFINITY,
                ..d
            },
            LionConfig { beta1: 1.0, ..d },
            LionConfig { beta1: -0.1, ..d },
            LionConfig {
                beta1: f32::NAN,
                ..d
            },
            LionConfig { beta2: 1.0, ..d },
            LionConfig { beta2: -0.1, ..d },
            LionConfig {
                weight_decay: -0.1,
                ..d
            },
            LionConfig {
                weight_decay: f32::INFINITY,
                ..d
            },
        ];
        for cfg in bad {
            assert!(
                matches!(Lion::new(cfg), Err(AutodiffError::InvalidArgument(_))),
                "{cfg:?}"
            );
        }
        assert!(Lion::new(d).is_ok());
        assert!(
            Lion::new(LionConfig {
                beta1: 0.0,
                beta2: 0.0,
                ..d
            })
            .is_ok()
        );
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(matches!(
            opt.step(&[(&param, &grad)]),
            Err(AutodiffError::Shape(_))
        ));
    }

    #[test]
    fn rejects_slot_count_and_shape_change_after_first_step() {
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0], &[2]);
        let g1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        assert!(matches!(
            opt.step(&[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
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
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0], &[2]);
        let g1 = t(vec![0.1, -0.1], &[2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        let bad_grad = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(opt.step(&[(&p1, &bad_grad)]).is_err());
        assert_eq!(opt.step_count(), 1);

        let mut opt_ref = Lion::new(LionConfig::default()).unwrap();
        opt_ref.step(&[(&p1, &g1)]).unwrap();
        let a = opt.step(&[(&p1, &g1)]).unwrap();
        let b = opt_ref.step(&[(&p1, &g1)]).unwrap();
        assert_eq!(vals(&a[0]), vals(&b[0]));
    }

    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        let bad_p = t(vec![1.0, 2.0], &[2]);
        let bad_g = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(opt.step(&[(&bad_p, &bad_g)]).is_err());
        let pa = t(vec![1.0], &[1]);
        let ga = t(vec![0.1], &[1]);
        let pb = t(vec![1.0, 2.0, 3.0], &[3]);
        let gb = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(opt.step(&[(&pa, &ga), (&pb, &gb)]).is_ok());
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn step_count_overflow_is_rejected_without_state_change() {
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        opt.step_count = u64::MAX;
        let p = t(vec![1.0], &[1]);
        let g = t(vec![0.1], &[1]);
        assert!(matches!(
            opt.step(&[(&p, &g)]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert_eq!(opt.step_count(), u64::MAX);
        assert!(opt.states.is_empty());
    }

    /// t=1 の閉形式: `param*(1-lr*wd) - lr*sign(g)`・`exp_avg=(1-beta2)*g`。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = LionConfig {
            lr: 0.05,
            weight_decay: 0.1,
            ..LionConfig::default()
        };
        let mut opt = Lion::new(cfg).unwrap();
        let p = t(vec![0.5, -0.5, 0.25], &[3]);
        let g = t(vec![0.3, -7.0, 0.0], &[3]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        let decay = (1.0f64 - 0.05f64 * 0.1f64) as f32;
        assert_eq!(
            vals(&out[0]),
            vec![0.5 * decay - 0.05, -0.5 * decay + 0.05, 0.25 * decay]
        );
        let ea = &opt.states[0].exp_avg;
        assert!((ea[0] - 0.01 * 0.3).abs() < 1e-7);
        assert!((ea[1] - 0.01 * -7.0).abs() < 1e-6);
        assert_eq!(ea[2], 0.0);
    }

    /// `beta1` は補間（update）・`beta2` は保存側に使われる。取り違えると
    /// t=2 の符号が変わる入力。
    #[test]
    fn beta_roles_are_not_swapped() {
        // beta1=0.5・beta2=0.9。step1 g=+1 → exp_avg=0.1。
        // step2 g=-0.2: update = 0.1*0.5 + (-0.2)*0.5 = -0.05 → sign=-1
        // （取り違え: 0.1*0.9 + (-0.2)*0.1 = +0.07 → sign=+1）。
        let cfg = LionConfig {
            lr: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            weight_decay: 0.0,
        };
        let mut opt = Lion::new(cfg).unwrap();
        let p0 = t(vec![0.0], &[1]);
        let p1 = opt.step(&[(&p0, &t(vec![1.0], &[1]))]).unwrap().remove(0);
        assert_eq!(vals(&p1), vec![-0.1]);
        let p2 = opt.step(&[(&p1, &t(vec![-0.2], &[1]))]).unwrap().remove(0);
        assert_eq!(vals(&p2), vec![-0.1 + 0.1]);
    }

    /// 補間係数は f64 で `1 - beta` を求める（f32 引き算だと 0.100000024 になる）。
    /// 相殺付近（`exp_avg*beta1 + g*(1-beta1)` が 0 の近傍）で更新方向が
    /// 参照実装（係数 0.1）と一致することを確かめる。
    #[test]
    fn interpolation_coefficient_matches_f64_reference_near_cancellation() {
        assert_eq!(complement_like_python(0.9), 0.1f32);
        assert_eq!(complement_like_python(0.99), (1.0f64 - 0.99f64) as f32);
        let cfg = LionConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        };
        let mut opt = Lion::new(cfg).unwrap();
        opt.states = vec![SlotState {
            shape: vec![1],
            exp_avg: vec![1.0],
        }];
        // 参照（係数 0.1）: update = 0.9f32 + (-8.999999)*0.1 = +5.96e-8 → sign=+1。
        // f32 の 1-beta1（0.100000024）だと -1.19e-7 → sign=-1 になり param が逆方向へ動く。
        let p = t(vec![0.0], &[1]);
        let out = opt.step(&[(&p, &t(vec![-8.999999], &[1]))]).unwrap();
        assert_eq!(vals(&out[0]), vec![-0.1]);
    }

    #[test]
    fn zero_grad_element_only_decays() {
        let cfg = LionConfig {
            lr: 0.1,
            weight_decay: 0.5,
            ..LionConfig::default()
        };
        let mut opt = Lion::new(cfg).unwrap();
        let p = t(vec![2.0], &[1]);
        let g = t(vec![0.0], &[1]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        assert_eq!(vals(&out[0]), vec![2.0 * (1.0 - 0.1f64 * 0.5) as f32]);
    }

    /// NaN 勾配の要素は当 step 動かず、`exp_avg` の NaN により以後凍結する
    /// （weight decay のみ効く）。
    #[test]
    fn nan_grad_freezes_element() {
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        let p = t(vec![1.0, 1.0], &[2]);
        let out1 = opt
            .step(&[(&p, &t(vec![f32::NAN, 1.0], &[2]))])
            .unwrap()
            .remove(0);
        assert_eq!(vals(&out1)[0], 1.0);
        assert!(opt.states[0].exp_avg[0].is_nan());
        let out2 = opt
            .step(&[(&out1, &t(vec![1.0, 1.0], &[2]))])
            .unwrap()
            .remove(0);
        assert_eq!(vals(&out2)[0], 1.0);
        assert!(vals(&out2)[1] < vals(&out1)[1]);
    }

    #[test]
    fn set_lr_validates_and_takes_effect_next_step() {
        let mut opt = Lion::new(LionConfig::default()).unwrap();
        opt.set_lr(0.5).unwrap();
        assert_eq!(opt.config().lr, 0.5);
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                opt.set_lr(bad),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert_eq!(opt.config().lr, 0.5);
        }
        let p = t(vec![0.0], &[1]);
        let out = opt.step(&[(&p, &t(vec![1.0], &[1]))]).unwrap();
        assert_eq!(vals(&out[0]), vec![-0.5]);
    }
}
