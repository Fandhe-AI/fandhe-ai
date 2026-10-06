//! Rprop（Riedmiller & Braun, 1993。`torch.optim.Rprop` 相当）。
//!
//! `torch.optim.Rprop` の単一テンソル実装（`_single_tensor_rprop`）と同一
//! 系列を再現する（イシュー #2655・親 #2654。参照値は
//! `tests/fixtures/rprop-pytorch-reference/rprop_reference.json`。実
//! PyTorch 2.14.0+cpu 実行値。README 参照）。演算順:
//!
//! ```text
//! prod      = grad * prev                  # f32 の積（アンダーフローで 0 になりうる）
//! s         = sign(prod)                   # torch の sign（±0・NaN は 0）
//! factor    = s>0 ? eta_plus : s<0 ? eta_minus : 1
//! step_size = clamp(step_size * factor, step_size_min, step_size_max)
//! grad      = (s<0 の要素は 0)             # 符号反転した要素は今回動かさない
//! param    -= sign(grad) * step_size
//! prev      = grad                         # ゼロ化後の grad
//! ```
//!
//! **`f32::signum` を使わない理由**: torch の `sign` は `(0 < x) - (x < 0)`
//! で `±0.0` と NaN を 0 にする（PyTorch 2.14.0 実測。fixture の `edge`
//! ブロック）。`f32::signum` は `+0.0` に 1.0・NaN に NaN を返すため、ゼロ
//! 勾配で更新が走る・NaN が伝播する誤実装になる。したがって NaN 勾配の要素
//! は当 step では更新されず `prev` に NaN が残り、次 step の積が NaN →
//! `sign` が 0 → `factor = 1` として再開する（PyTorch 準拠。黙って握り
//! つぶすのではなく本 doc で契約化する）。
//!
//! `step_size` はスロットごとに初回 `step()` 時点の `lr` で全要素初期化
//! される。以後 `lr` は更新値に影響しない（[`Rprop::set_lr`] の doc 参照）。
//!
//! `nn/optim/mod.rs` の doc が示す通り `step()` は `(param, grad)` の参照列を
//! 受け取り更新後 `Tensor<f32>` の列を返す。`Tape`／`Var`／`BackendOps` に
//! 一切依存しない値型・純関数であり、新規 `Op`／`BackendOps` メソッド／VJP
//! は追加していない（カーネルなし）。`adamax.rs` を鏡写しにした別実装で、
//! 内部ループは他 optimizer と共通化しない。`ParamGroupStep`／
//! `OptimizerStateDict` は未実装（決定記録 §10）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.Rprop` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RpropConfig {
    /// 初期 step size（`lr`）。有限かつ `>= 0.0` を [`Rprop::new`] が検証する。
    pub lr: f32,
    /// 勾配符号反転時の step size 縮小係数（`etas[0]`）。
    pub eta_minus: f32,
    /// 勾配同符号継続時の step size 拡大係数（`etas[1]`）。
    /// 有限かつ `0 < eta_minus < 1 < eta_plus` を [`Rprop::new`] が検証する。
    pub eta_plus: f32,
    /// step size の下限（`step_sizes[0]`）。
    pub step_size_min: f32,
    /// step size の上限（`step_sizes[1]`）。有限かつ
    /// `0 <= step_size_min <= step_size_max` を [`Rprop::new`] が検証する
    /// （PyTorch は無検査。`f32::clamp` の境界 panic を到達不能にするため）。
    pub step_size_max: f32,
}

impl Default for RpropConfig {
    fn default() -> RpropConfig {
        RpropConfig {
            lr: 1e-2,
            eta_minus: 0.5,
            eta_plus: 1.2,
            step_size_min: 1e-6,
            step_size_max: 50.0,
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

/// パラメータスロットごとの状態。初回 `step()` で渡された `param` の shape
/// から遅延初期化する（`adamax.rs::SlotState` と同じ理由）。
struct SlotState {
    shape: Vec<usize>,
    prev: Vec<f32>,
    step_size: Vec<f32>,
}

/// Rprop optimizer 本体。ハイパーパラメータ・step 数・スロットごとの状態を
/// 保持する。
pub struct Rprop {
    config: RpropConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

impl Rprop {
    /// ハイパーパラメータを検証して構築する。
    ///
    /// # Errors
    ///
    /// 非有限値・範囲外（`lr < 0`、`0 < eta_minus < 1 < eta_plus` 違反、
    /// `step_size_min < 0`、`step_size_min > step_size_max`）は
    /// `AutodiffError::InvalidArgument`。
    pub fn new(config: RpropConfig) -> Result<Rprop, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Rprop::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.eta_minus.is_finite()
            && config.eta_plus.is_finite()
            && 0.0 < config.eta_minus
            && config.eta_minus < 1.0
            && 1.0 < config.eta_plus)
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Rprop::new: etas must satisfy 0 < eta_minus < 1 < eta_plus, got ({}, {})",
                config.eta_minus, config.eta_plus
            )));
        }
        if !(config.step_size_min.is_finite()
            && config.step_size_max.is_finite()
            && config.step_size_min >= 0.0
            && config.step_size_min <= config.step_size_max)
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Rprop::new: step sizes must satisfy 0 <= step_size_min <= step_size_max, got ({}, {})",
                config.step_size_min, config.step_size_max
            )));
        }
        Ok(Rprop {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &RpropConfig {
        &self.config
    }

    /// `config.lr` のみを書き換える。Rprop の `lr` は初回 `step()` の
    /// `step_size` 初期化にしか使われないため、初回 step 後の呼び出しは
    /// 更新値に影響しない（`step_size` の再スケールはしない。PyTorch で
    /// `param_group["lr"]` を変更した場合と同じ）。状態は不変。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値なら `AutodiffError::InvalidArgument`
    /// （`self.config` は変更しない）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Rprop::set_lr: lr must be finite and >= 0.0, got {new_lr}"
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
    /// `adamax.rs` と同じ 2 段構成（副作用なしの検証 → 状態変更）で、
    /// 失敗時は状態（`step_count`・スロット）を一切変更しない。
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
                    "Rprop::step: slot count changed across calls (expected {}, got {}); \
                     Rprop state (prev/step_size) is keyed by call-order slot index and \
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

        // 遅延初期化より前に確定させ、Err 時に状態が変化しないアトミック性を
        // 保つ（`adamax.rs` の同箇所と同じ理由）。
        let next_step_count = self.step_count.checked_add(1).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Rprop::step: step_count overflow: too many step() calls for this optimizer \
                 to advance further"
                    .to_string(),
            )
        })?;

        if self.states.is_empty() && !params_and_grads.is_empty() {
            let lr = self.config.lr;
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| SlotState {
                    shape: param.shape().to_vec(),
                    prev: vec![0.0f32; param.numel()],
                    step_size: vec![lr; param.numel()],
                })
                .collect();
        }
        self.step_count = next_step_count;

        let eta_minus = self.config.eta_minus;
        let eta_plus = self.config.eta_plus;
        let lo = self.config.step_size_min;
        let hi = self.config.step_size_max;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let g = grad_data[i];
                let s = torch_sign(g * slot.prev[i]);
                let factor = if s > 0.0 {
                    eta_plus
                } else if s < 0.0 {
                    eta_minus
                } else {
                    1.0
                };
                // `step_size.mul_(sign).clamp_(min, max)`。境界は `new` で
                // `min <= max`・有限を検証済みのため `clamp` は panic しない。
                let ss = (slot.step_size[i] * factor).clamp(lo, hi);
                slot.step_size[i] = ss;

                // 符号反転した要素は勾配を 0 にして今回動かさない。
                let g_eff = if s < 0.0 { 0.0 } else { g };
                new_param.push(param_data[i] - torch_sign(g_eff) * ss);
                slot.prev[i] = g_eff;
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
        let d = RpropConfig::default();
        let bad = [
            RpropConfig { lr: -1.0, ..d },
            RpropConfig { lr: f32::NAN, ..d },
            RpropConfig {
                lr: f32::INFINITY,
                ..d
            },
            RpropConfig {
                eta_minus: 0.0,
                ..d
            },
            RpropConfig {
                eta_minus: 1.0,
                ..d
            },
            RpropConfig { eta_plus: 1.0, ..d },
            RpropConfig {
                eta_plus: f32::INFINITY,
                ..d
            },
            RpropConfig {
                eta_minus: f32::NAN,
                ..d
            },
            RpropConfig {
                step_size_min: -1.0,
                ..d
            },
            RpropConfig {
                step_size_min: 1.0,
                step_size_max: 0.5,
                ..d
            },
            RpropConfig {
                step_size_max: f32::NAN,
                ..d
            },
            RpropConfig {
                step_size_max: f32::INFINITY,
                ..d
            },
        ];
        for cfg in bad {
            assert!(
                matches!(Rprop::new(cfg), Err(AutodiffError::InvalidArgument(_))),
                "{cfg:?}"
            );
        }
        assert!(Rprop::new(d).is_ok());
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(matches!(
            opt.step(&[(&param, &grad)]),
            Err(AutodiffError::Shape(_))
        ));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        assert!(matches!(
            opt.step(&[]),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0], &[2]);
        let g1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        let p2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let g2 = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(matches!(
            opt.step(&[(&p2, &g2)]),
            Err(AutodiffError::Shape(_))
        ));
    }

    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0], &[2]);
        let g1 = t(vec![0.1, -0.1], &[2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        let bad_grad = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(matches!(
            opt.step(&[(&p1, &bad_grad)]),
            Err(AutodiffError::Shape(_))
        ));
        assert_eq!(opt.step_count(), 1);

        let mut opt_ref = Rprop::new(RpropConfig::default()).unwrap();
        opt_ref.step(&[(&p1, &g1)]).unwrap();
        let a = opt.step(&[(&p1, &g1)]).unwrap();
        let b = opt_ref.step(&[(&p1, &g1)]).unwrap();
        assert_eq!(vals(&a[0]), vals(&b[0]));
    }

    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
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
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
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

    /// t=1 の閉形式: `param - lr * sign(grad)`（`lr` が境界内のとき）。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = RpropConfig {
            lr: 0.05,
            ..RpropConfig::default()
        };
        let mut opt = Rprop::new(cfg).unwrap();
        let p = t(vec![0.5, -0.5, 0.25], &[3]);
        let g = t(vec![0.3, -7.0, 0.0], &[3]);
        let out = opt.step(&[(&p, &g)]).unwrap();
        assert_eq!(vals(&out[0]), vec![0.5 - 0.05, -0.5 + 0.05, 0.25]);
    }

    #[test]
    fn zero_grad_keeps_param() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let p = t(vec![1.0, -1.0], &[2]);
        let g = t(vec![0.0, -0.0], &[2]);
        for _ in 0..3 {
            let out = opt.step(&[(&p, &g)]).unwrap();
            assert_eq!(vals(&out[0]), vec![1.0, -1.0]);
        }
    }

    /// 符号反転 step では param 不変・`step_size` が `eta_minus` 倍・`prev` が 0。
    #[test]
    fn sign_flip_freezes_element_and_shrinks_step_size() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let p0 = t(vec![0.0], &[1]);
        let out1 = opt.step(&[(&p0, &t(vec![1.0], &[1]))]).unwrap();
        let out2 = opt.step(&[(&out1[0], &t(vec![-1.0], &[1]))]).unwrap();
        assert_eq!(vals(&out2[0]), vals(&out1[0]));
        assert_eq!(opt.states[0].step_size[0], 0.01f32 * 0.5);
        assert_eq!(opt.states[0].prev[0], 0.0);
    }

    #[test]
    fn same_sign_grows_step_size_and_clamps_at_bounds() {
        let cfg = RpropConfig {
            step_size_min: 5e-3,
            step_size_max: 2e-2,
            ..RpropConfig::default()
        };
        let mut opt = Rprop::new(cfg).unwrap();
        let mut p = t(vec![0.0], &[1]);
        for _ in 0..8 {
            p = opt.step(&[(&p, &t(vec![1.0], &[1]))]).unwrap().remove(0);
        }
        assert_eq!(opt.states[0].step_size[0], 2e-2);

        // 下限: 反転で eta_minus 倍 → 下限へクランプ。
        let mut opt = Rprop::new(RpropConfig {
            step_size_min: 8e-3,
            step_size_max: 2e-2,
            ..RpropConfig::default()
        })
        .unwrap();
        let p = t(vec![0.0], &[1]);
        let out = opt.step(&[(&p, &t(vec![1.0], &[1]))]).unwrap();
        opt.step(&[(&out[0], &t(vec![-1.0], &[1]))]).unwrap();
        assert_eq!(opt.states[0].step_size[0], 8e-3);
    }

    /// NaN 勾配の要素は当 step 更新されず `prev` に NaN が残り、次 step は
    /// `sign(NaN) = 0` として再開する（PyTorch 2.14.0 実測と同じ）。
    #[test]
    fn nan_grad_is_not_applied_and_resumes_next_step() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        let p = t(vec![0.0], &[1]);
        let out1 = opt.step(&[(&p, &t(vec![f32::NAN], &[1]))]).unwrap();
        assert_eq!(vals(&out1[0]), vec![0.0]);
        assert!(opt.states[0].prev[0].is_nan());
        let out2 = opt.step(&[(&out1[0], &t(vec![1.0], &[1]))]).unwrap();
        assert_eq!(vals(&out2[0]), vec![-0.01]);
    }

    #[test]
    fn set_lr_validates_and_keeps_other_fields_and_state() {
        let mut opt = Rprop::new(RpropConfig::default()).unwrap();
        opt.set_lr(0.5).unwrap();
        assert_eq!(opt.config().lr, 0.5);
        assert_eq!(opt.config().eta_plus, RpropConfig::default().eta_plus);
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                opt.set_lr(bad),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert_eq!(opt.config().lr, 0.5);
        }
    }

    /// 初回 step 後の `set_lr` は更新値に影響しない（`step_size` 初期化にしか
    /// 使われないため）。
    #[test]
    fn set_lr_after_first_step_does_not_change_updates() {
        let p = t(vec![0.0, 0.0], &[2]);
        let g = t(vec![1.0, -1.0], &[2]);
        let mut a = Rprop::new(RpropConfig::default()).unwrap();
        let mut b = Rprop::new(RpropConfig::default()).unwrap();
        let a1 = a.step(&[(&p, &g)]).unwrap();
        let b1 = b.step(&[(&p, &g)]).unwrap();
        b.set_lr(0.9).unwrap();
        let a2 = a.step(&[(&a1[0], &g)]).unwrap();
        let b2 = b.step(&[(&b1[0], &g)]).unwrap();
        assert_eq!(vals(&a2[0]), vals(&b2[0]));
        assert_eq!(a.step_count(), 2);
        assert_eq!(b.step_count(), 2);
    }
}
