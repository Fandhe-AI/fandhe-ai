//! ASGD（Averaged SGD, Polyak & Juditsky, 1992。`torch.optim.ASGD` 相当）。
//!
//! `torch.optim.ASGD` の単一テンソル実装（`_single_tensor_asgd`）と同一系列
//! を再現する（イシュー #2655・親 #2654。参照値は
//! `tests/fixtures/asgd-pytorch-reference/asgd_reference.json`。実 PyTorch
//! 2.14.0+cpu 実行値。README 参照）。演算順（スロットごとの状態は `ax`〈0
//! 初期化〉・`eta`〈初期値 `lr`〉・`mu`〈初期値 1〉）:
//!
//! ```text
//! step += 1
//! weight_decay != 0 なら grad += weight_decay * param     # 減衰前の param
//! param *= (1 - lambd * eta)
//! param -= eta * grad
//! mu != 1 なら ax += (param - ax) * mu、mu == 1 なら ax = param
//! eta = lr / (1 + lambd * lr * step)^alpha                # 次の step で使う
//! mu  = 1 / max(1, step - t0)                             # 同上
//! ```
//!
//! 更新には「前 step の終わりに計算した `eta`／`mu`」を使い、当 step の終わ
//! りに次回分を計算する。PyTorch は `eta`／`mu` を `float32` スカラーテンソ
//! ルで保持するため、係数は `f64` で計算して `f32` へ落としてから保持・使用
//! する（`mu != 1` の判定も保持した `f32` 値で行う）。
//!
//! ASGD の成果物は平均化パラメータ `ax` であり、PyTorch では
//! `optimizer.state[p]["ax"]` からしか取れない。値型では
//! [`Asgd::averaged_params`] で取り出す。
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

/// `torch.optim.ASGD` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsgdConfig {
    /// 学習率（`lr`）。有限かつ `>= 0.0` を [`Asgd::new`] が検証する。
    pub lr: f32,
    /// 減衰項係数（`lambd`）。有限かつ `>= 0.0`（PyTorch は無検査。負値は
    /// べき乗の底が負になりうるため弾く）。
    pub lambd: f32,
    /// `eta` 更新のべき指数（`alpha`）。有限かつ `>= 0.0`（PyTorch は無検査。
    /// 負値は `eta` が発散しうるため弾く）。
    pub alpha: f32,
    /// 平均化を開始する step（`t0`）。有限かつ `>= 0.0`（PyTorch は無検査）。
    pub t0: f32,
    /// L2 正則化係数。有限かつ `>= 0.0`（coupled 方式）。
    pub weight_decay: f32,
}

impl Default for AsgdConfig {
    fn default() -> AsgdConfig {
        AsgdConfig {
            lr: 1e-2,
            lambd: 1e-4,
            alpha: 0.75,
            t0: 1e6,
            weight_decay: 0.0,
        }
    }
}

/// パラメータスロットごとの状態。初回 `step()` で渡された `param` の shape
/// から遅延初期化する。`eta`／`mu` は PyTorch の state 構造と同じくスロット
/// ごとに持つ。
struct SlotState {
    shape: Vec<usize>,
    ax: Vec<f32>,
    eta: f32,
    mu: f32,
}

/// ASGD optimizer 本体。
pub struct Asgd {
    config: AsgdConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

impl Asgd {
    /// ハイパーパラメータを検証して構築する。
    ///
    /// # Errors
    ///
    /// いずれかが非有限または負値なら `AutodiffError::InvalidArgument`。
    pub fn new(config: AsgdConfig) -> Result<Asgd, AutodiffError> {
        for (name, v) in [
            ("lr", config.lr),
            ("lambd", config.lambd),
            ("alpha", config.alpha),
            ("t0", config.t0),
            ("weight_decay", config.weight_decay),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Asgd::new: {name} must be finite and >= 0.0, got {v}"
                )));
            }
        }
        Ok(Asgd {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &AsgdConfig {
        &self.config
    }

    /// `config.lr` のみを書き換える。保持済みの `eta` は次の `step()` でその
    /// まま使われ、新しい `lr` はその step の終わりに計算する `eta` から効く
    /// （PyTorch で `param_group["lr"]` を変更した場合と同じ 1 step 遅れ）。
    /// 状態は不変。
    ///
    /// # Errors
    ///
    /// `new_lr` が非有限または負値なら `AutodiffError::InvalidArgument`
    /// （`self.config` は変更しない）。
    pub fn set_lr(&mut self, new_lr: f32) -> Result<(), AutodiffError> {
        if !(new_lr.is_finite() && new_lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Asgd::set_lr: lr must be finite and >= 0.0, got {new_lr}"
            )));
        }
        self.config.lr = new_lr;
        Ok(())
    }

    /// 実行済み `step()` 回数。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// 平均化パラメータ `ax`（スロット順。PyTorch の `state[p]["ax"]` 相当）。
    /// 初回 `step()` 前は空列。
    pub fn averaged_params(&self) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        self.states
            .iter()
            .map(|slot| Tensor::new(slot.ax.clone(), &slot.shape).map_err(AutodiffError::from))
            .collect()
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
                    "Asgd::step: slot count changed across calls (expected {}, got {}); \
                     ASGD state (ax/eta/mu) is keyed by call-order slot index and \
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
                "Asgd::step: step_count overflow: too many step() calls for this optimizer \
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
                    ax: vec![0.0f32; param.numel()],
                    eta: lr,
                    mu: 1.0,
                })
                .collect();
        }
        self.step_count = next_step_count;

        let lambd = self.config.lambd as f64;
        let lr = self.config.lr as f64;
        let alpha = self.config.alpha as f64;
        let t0 = self.config.t0 as f64;
        let weight_decay = self.config.weight_decay;
        // PyTorch の `step` は float32 テンソルだが 2^24 step 超の飽和は再現
        // せず `u64` を f64 へ変換して使う（決定記録 §5）。
        let step = next_step_count as f64;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            let eta = slot.eta;
            // `param.mul_(1 - lambd * eta_value)`: Python float（f64）で計算し
            // f32 へ落とす。
            let decay = (1.0 - lambd * eta as f64) as f32;
            let mu = slot.mu;

            for i in 0..param_data.len() {
                let mut g = grad_data[i];
                if weight_decay != 0.0 {
                    g = f32::mul_add(weight_decay, param_data[i], g);
                }
                let decayed = param_data[i] * decay;
                let updated = f32::mul_add(-eta, g, decayed);
                new_param.push(updated);

                slot.ax[i] = if mu != 1.0 {
                    slot.ax[i] + (updated - slot.ax[i]) * mu
                } else {
                    updated
                };
            }

            slot.eta = (lr / (1.0 + lambd * lr * step).powf(alpha)) as f32;
            slot.mu = (1.0 / (step - t0).max(1.0)) as f32;

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
    fn rejects_invalid_hyperparameters() {
        let d = AsgdConfig::default();
        let bad = [
            AsgdConfig { lr: -1.0, ..d },
            AsgdConfig { lr: f32::NAN, ..d },
            AsgdConfig { lambd: -1e-4, ..d },
            AsgdConfig {
                lambd: f32::INFINITY,
                ..d
            },
            AsgdConfig { alpha: -0.5, ..d },
            AsgdConfig {
                alpha: f32::NAN,
                ..d
            },
            AsgdConfig { t0: -1.0, ..d },
            AsgdConfig {
                t0: f32::INFINITY,
                ..d
            },
            AsgdConfig {
                weight_decay: -0.1,
                ..d
            },
            AsgdConfig {
                weight_decay: f32::NAN,
                ..d
            },
        ];
        for cfg in bad {
            assert!(
                matches!(Asgd::new(cfg), Err(AutodiffError::InvalidArgument(_))),
                "{cfg:?}"
            );
        }
        assert!(Asgd::new(d).is_ok());
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(matches!(
            opt.step(&[(&param, &grad)]),
            Err(AutodiffError::Shape(_))
        ));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
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
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
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
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
        let p1 = t(vec![1.0, 2.0], &[2]);
        let g1 = t(vec![0.1, -0.1], &[2]);
        opt.step(&[(&p1, &g1)]).unwrap();
        let bad_grad = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(matches!(
            opt.step(&[(&p1, &bad_grad)]),
            Err(AutodiffError::Shape(_))
        ));
        assert_eq!(opt.step_count(), 1);

        let mut opt_ref = Asgd::new(AsgdConfig::default()).unwrap();
        opt_ref.step(&[(&p1, &g1)]).unwrap();
        let a = opt.step(&[(&p1, &g1)]).unwrap();
        let b = opt_ref.step(&[(&p1, &g1)]).unwrap();
        assert_eq!(vals(&a[0]), vals(&b[0]));
    }

    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
        let bad_p = t(vec![1.0, 2.0], &[2]);
        let bad_g = t(vec![1.0, 2.0, 3.0], &[3]);
        assert!(opt.step(&[(&bad_p, &bad_g)]).is_err());
        assert!(opt.averaged_params().unwrap().is_empty());
        let pa = t(vec![1.0], &[1]);
        let ga = t(vec![0.1], &[1]);
        let pb = t(vec![1.0, 2.0, 3.0], &[3]);
        let gb = t(vec![0.1, 0.1, 0.1], &[3]);
        assert!(opt.step(&[(&pa, &ga), (&pb, &gb)]).is_ok());
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn step_count_overflow_is_rejected_without_state_change() {
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
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

    #[test]
    fn averaged_params_is_empty_before_first_step() {
        let opt = Asgd::new(AsgdConfig::default()).unwrap();
        assert!(opt.averaged_params().unwrap().is_empty());
    }

    /// t=1 の閉形式: `param*(1 - lambd*lr) - lr*(grad + wd*param)`、`ax == 更新後 param`。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = AsgdConfig {
            lr: 0.1,
            lambd: 0.01,
            alpha: 0.75,
            t0: 1e6,
            weight_decay: 0.2,
        };
        let mut opt = Asgd::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let out = opt
            .step(&[(&t(vec![p0], &[1]), &t(vec![g0], &[1]))])
            .unwrap();
        let expected = p0 * (1.0 - cfg.lambd * cfg.lr) - cfg.lr * (g0 + cfg.weight_decay * p0);
        let actual = vals(&out[0])[0];
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        assert_eq!(vals(&opt.averaged_params().unwrap()[0]), vec![actual]);
    }

    /// 2 step 目以降の `eta`／`mu` が式どおりであること。
    #[test]
    fn eta_and_mu_follow_schedule() {
        let cfg = AsgdConfig {
            lr: 0.1,
            lambd: 0.01,
            alpha: 0.75,
            t0: 2.0,
            weight_decay: 0.0,
        };
        let mut opt = Asgd::new(cfg).unwrap();
        let mut p = t(vec![0.0], &[1]);
        let g = t(vec![1.0], &[1]);
        for step in 1..=5u32 {
            p = opt.step(&[(&p, &g)]).unwrap().remove(0);
            let s = step as f64;
            let eta =
                (0.1f64 / (1.0 + 0.01f64 as f32 as f64 * 0.1f32 as f64 * s).powf(0.75)) as f32;
            let mu = (1.0 / (s - 2.0).max(1.0)) as f32;
            assert!((opt.states[0].eta - eta).abs() < 1e-6, "step {step}");
            assert_eq!(opt.states[0].mu, mu, "step {step}");
        }
    }

    /// `mu != 1` 分岐で `ax += (param - ax) * mu` の平均化になること。
    #[test]
    fn averaging_branch_uses_mu() {
        let cfg = AsgdConfig {
            lr: 0.1,
            lambd: 0.0,
            alpha: 0.0,
            t0: 1.0,
            weight_decay: 0.0,
        };
        let mut opt = Asgd::new(cfg).unwrap();
        let g = t(vec![1.0], &[1]);
        let mut p = t(vec![0.0], &[1]);
        let mut history = Vec::new();
        for _ in 0..4 {
            p = opt.step(&[(&p, &g)]).unwrap().remove(0);
            history.push((vals(&p)[0], vals(&opt.averaged_params().unwrap()[0])[0]));
        }
        // step1/2 は mu == 1（max(1, step - 1) = 1）で ax = param。
        assert_eq!(history[0].0, history[0].1);
        assert_eq!(history[1].0, history[1].1);
        // step3 は直前 step2 の終わりに mu = 1/max(1, 2-1) = 1。step4 の更新で
        // 使う mu は step3 の終わりの 1/max(1, 3-1) = 0.5。
        assert_eq!(history[2].0, history[2].1);
        let expected = history[2].1 + (history[3].0 - history[2].1) * 0.5;
        assert!((history[3].1 - expected).abs() < 1e-7);
        assert_ne!(history[3].0, history[3].1);
    }

    #[test]
    fn set_lr_validates_and_applies_one_step_late() {
        let mut opt = Asgd::new(AsgdConfig::default()).unwrap();
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                opt.set_lr(bad),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert_eq!(opt.config().lr, AsgdConfig::default().lr);
        }
        let p = t(vec![0.0], &[1]);
        let g = t(vec![1.0], &[1]);
        let mut a = Asgd::new(AsgdConfig::default()).unwrap();
        let mut b = Asgd::new(AsgdConfig::default()).unwrap();
        let a1 = a.step(&[(&p, &g)]).unwrap();
        let b1 = b.step(&[(&p, &g)]).unwrap();
        b.set_lr(0.5).unwrap();
        assert_eq!(b.config().lr, 0.5);
        assert_eq!(b.config().alpha, AsgdConfig::default().alpha);
        assert_eq!(b.step_count(), 1);
        // 2 step 目の更新は保持済み eta（旧 lr 由来）を使うため一致する。
        let a2 = a.step(&[(&a1[0], &g)]).unwrap();
        let b2 = b.step(&[(&b1[0], &g)]).unwrap();
        assert_eq!(vals(&a2[0]), vals(&b2[0]));
        // 3 step 目は新 lr 由来の eta が効き、差が出る。
        let a3 = a.step(&[(&a2[0], &g)]).unwrap();
        let b3 = b.step(&[(&b2[0], &g)]).unwrap();
        assert_ne!(vals(&a3[0]), vals(&b3[0]));
    }
}
