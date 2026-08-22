//! AdamW（decoupled weight decay。Loshchilov & Hutter, 2019）。
//!
//! `torch.optim.AdamW` の単一テンソル実装（`step_size = lr /
//! bias_correction1`・`denom = sqrt(v) / sqrt(bias_correction2) + eps`
//! の演算順）と同一系列を再現する（イシュー #194・受け入れ条件
//! 「PyTorch AdamW と同一系列の更新値一致テストが green」。
//! `tests/nn_optim_adamw.rs::adamw_matches_pytorch_reference` が
//! `tests/fixtures/adamw-pytorch-reference/` の実測値と突合する）。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。呼び出し元
//! （学習ループ）は `Linear::from_parameters` 等で層を再構築する
//! 既存の不変更新パターン（`tests/nn_train_convergence.rs`）にそのまま
//! 差し込める。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec;

/// `torch.optim.AdamW` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> AdamWConfig {
        AdamWConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの 1 次
/// （`m`）・2 次（`v`）モーメント推定値。初回 `step()` 呼び出しで
/// 渡された `param` の shape から遅延初期化する（`AdamW::new` の
/// 時点ではパラメータ数・shape を知らないため）。
struct SlotState {
    shape: Vec<usize>,
    m: Vec<f32>,
    v: Vec<f32>,
}

/// AdamW optimizer 本体。ハイパーパラメータ（[`AdamWConfig`]）と、
/// step 数・bias correction 用の `beta^t` 逐次積・スロットごとの
/// モーメント推定値（`SlotState`）を保持する。
pub struct AdamW {
    config: AdamWConfig,
    step_count: u64,
    // β^t の逐次積を f64 で保持する（PyTorch の Python float は f64）。
    // `beta.powi(t as i32)` の毎 step 再計算だと t が大きい場合に
    // `i32` へのキャストが問題になりうる上、逐次積のほうが PyTorch の
    // `bias_correction1 = 1 - beta1 ** step`（Python の整数べき乗を
    // 都度評価する実装）と丸め挙動が近い（f64 の乗算を step 回
    // 繰り返す点で一致する）。要素演算自体は f32・`f32::mul_add`
    // （FMA 契約。`.claude/rules/coding-rust.md`）で行うが、bias
    // correction 係数のみ f64 で保持し PyTorch の丸め挙動へ寄せる。
    beta1_pow_t: f64,
    beta2_pow_t: f64,
    states: Vec<SlotState>,
}

impl AdamW {
    /// ハイパーパラメータを検証して構築する。`lr`/`weight_decay` は
    /// 有限かつ非負、`beta1`/`beta2` は `[0, 1)`、`eps` は有限かつ正
    /// （0 を許すと `v` が 0 のスロット・step でゼロ除算になるため
    /// `Linear::new` の `in_features == 0` 検証と同様、テンソル生成に
    /// 進む前に構築不可能な引数として弾く。`error.rs` の
    /// `InvalidArgument` doc 参照）。
    pub fn new(config: AdamWConfig) -> Result<AdamW, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "AdamW::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "AdamW::new: beta1 must be in [0.0, 1.0), got {}",
                config.beta1
            )));
        }
        if !(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "AdamW::new: beta2 must be in [0.0, 1.0), got {}",
                config.beta2
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "AdamW::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "AdamW::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        Ok(AdamW {
            config,
            step_count: 0,
            beta1_pow_t: 1.0,
            beta2_pow_t: 1.0,
            states: Vec::new(),
        })
    }

    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }

    /// 実行済み `step()` 回数（bias correction の `t`）。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// 初回呼び出しでスロット数・各スロットの shape を確定し
    /// （`param.shape()` から `m`/`v` を 0 初期化する）、2 回目以降は
    /// スロット数・shape の一致を検査する（呼び出し元がパラメータ集合を
    /// step 間で変えるのは大抵バグであり、`Linear::from_parameters` が
    /// 外部由来パラメータの shape を検証するのと同じ理由で、ここでも
    /// 黙って状態を破棄・再初期化せずエラーにする。`.claude/rules/
    /// security.md` A03）。grad が存在しないパラメータ（PyTorch の
    /// `grad=None` 相当）は呼び出し元がそもそも本メソッドへ渡さない
    /// 契約とする（本メソッドは常に全スロットを更新する）。
    pub fn step(
        &mut self,
        params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        if self.states.is_empty() && !params_and_grads.is_empty() {
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| SlotState {
                    shape: param.shape().to_vec(),
                    m: vec![0.0f32; param.numel()],
                    v: vec![0.0f32; param.numel()],
                })
                .collect();
        }

        if params_and_grads.len() != self.states.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "AdamW::step: slot count changed across calls (expected {}, got {}); \
                 AdamW state (m/v) is keyed by call-order slot index and cannot be \
                 resized after the first step()",
                self.states.len(),
                params_and_grads.len()
            )));
        }

        // 副作用（step_count・beta*_pow_t・m/v の更新）を一切加えない
        // 検証専用フェーズ。全スロットの shape を先に確認しきってから
        // 状態変更フェーズへ進む（Bugbot 指摘: 形状エラー発生時に
        // step_count・bias-correction・m/v が部分更新のまま残ると、
        // 後続の成功する step が誤った t を適用し破損した状態から
        // 学習してしまうため。検証と状態変更を分離して防ぐ）。
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
        self.beta1_pow_t *= self.config.beta1 as f64;
        self.beta2_pow_t *= self.config.beta2 as f64;
        let bias_correction1 = 1.0 - self.beta1_pow_t;
        let bias_correction2 = 1.0 - self.beta2_pow_t;
        let bias_correction2_sqrt = (bias_correction2.sqrt()) as f32;
        let step_size = (self.config.lr as f64 / bias_correction1) as f32;
        let decay_factor = 1.0 - self.config.lr * self.config.weight_decay;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            let param_data = dense_vec(param);
            let grad_data = dense_vec(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                let g = grad_data[i];
                // decoupled weight decay（Loshchilov & Hutter, 2019）:
                // 勾配へ加算せず（L2 正則化ではない）、パラメータへ直接
                // 乗算で適用する。PyTorch AdamW と同じ演算順（更新式
                // 全体より前）で行う。
                let p_decayed = param_data[i] * decay_factor;

                let m = f32::mul_add(self.config.beta1, slot.m[i], (1.0 - self.config.beta1) * g);
                let v = f32::mul_add(
                    self.config.beta2,
                    slot.v[i],
                    (1.0 - self.config.beta2) * g * g,
                );
                slot.m[i] = m;
                slot.v[i] = v;

                let denom = v.sqrt() / bias_correction2_sqrt + self.config.eps;
                new_param.push(p_decayed - step_size * m / denom);
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
        let cfg = AdamWConfig {
            lr: -1.0,
            ..AdamWConfig::default()
        };
        assert!(matches!(
            AdamW::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_beta_out_of_range() {
        let cfg = AdamWConfig {
            beta1: 1.0,
            ..AdamWConfig::default()
        };
        assert!(matches!(
            AdamW::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let cfg = AdamWConfig {
            beta2: -0.1,
            ..AdamWConfig::default()
        };
        assert!(matches!(
            AdamW::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = AdamWConfig {
            eps: 0.0,
            ..AdamWConfig::default()
        };
        assert!(matches!(
            AdamW::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = AdamWConfig {
            weight_decay: f32::NAN,
            ..AdamWConfig::default()
        };
        assert!(matches!(
            AdamW::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        // 2 回目に空スロット列を渡すとスロット数不一致でエラーになる。
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    /// Bugbot 指摘の回帰テスト: 形状エラーで `step` が失敗した場合、
    /// `step_count`／`beta*_pow_t`／`m`／`v` のいずれも部分更新されず
    /// 呼び出し前の状態のまま残ることを確認する（状態変更前に全形状
    /// 検証を完了させる。crates/autodiff/src/nn/optim/adamw.rs 内の
    /// 検証専用フェーズのコメント参照）。
    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();
        let step_count_before = opt.step_count();

        // 2 回目呼び出しで shape 不一致を発生させる（grad の shape が
        // param と一致しない）。
        let param2 = t(vec![1.0, 2.0], &[2]);
        let bad_grad = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &bad_grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
        assert_eq!(
            opt.step_count(),
            step_count_before,
            "shape エラー発生時に step_count が進んではならない"
        );

        // 状態が破損していないことを、同じ入力で 3 回目 step() を
        // 呼んだ結果が「shape エラーが起きなかった場合の 2 回目の
        // step()」と一致することで間接的に確認する。
        let mut opt_ref = AdamW::new(AdamWConfig::default()).unwrap();
        opt_ref.step(&[(&param1, &grad1)]).unwrap();
        let param3 = t(vec![1.0, 2.0], &[2]);
        let grad3 = t(vec![0.1, 0.1], &[2]);
        let out_after_failed = opt.step(&[(&param3, &grad3)]).unwrap();
        let out_ref = opt_ref.step(&[(&param3, &grad3)]).unwrap();
        assert_eq!(dense_vec(&out_after_failed[0]), dense_vec(&out_ref[0]));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    /// decoupled 性の直接確認: 勾配が常に 0 のとき `m`/`v` は 0 のまま
    /// 保たれ、パラメータは `p *= (1 - lr * weight_decay)` のみで
    /// 更新される（Adam 由来の適応的スケーリング項が寄与しないため、
    /// 減衰が「勾配に混ぜた L2 正則化」ではないことを固定する）。
    #[test]
    fn decoupled_weight_decay_without_grad() {
        let cfg = AdamWConfig {
            lr: 0.1,
            weight_decay: 0.2,
            ..AdamWConfig::default()
        };
        let mut opt = AdamW::new(cfg).unwrap();
        let mut param = t(vec![1.0, -2.0], &[2]);
        let grad = t(vec![0.0, 0.0], &[2]);

        let decay_factor = 1.0 - cfg.lr * cfg.weight_decay;
        for _ in 0..5 {
            let out = opt.step(&[(&param, &grad)]).unwrap();
            let expected: Vec<f32> = (0..param.numel())
                .map(|i| param.get(&[i]).unwrap() * decay_factor)
                .collect();
            let actual: Vec<f32> = (0..out[0].numel())
                .map(|i| out[0].get(&[i]).unwrap())
                .collect();
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!(
                    (a - e).abs() < 1e-6,
                    "decoupled weight decay の期待値と不一致: {a} vs {e}"
                );
            }
            param = out.into_iter().next().unwrap();
        }
    }

    /// `weight_decay = 0` のとき decay 項が完全に消えること（乗算係数が
    /// 厳密に 1.0 になる）を確認する。
    #[test]
    fn weight_decay_zero_reduces_to_adam() {
        let cfg = AdamWConfig {
            weight_decay: 0.0,
            ..AdamWConfig::default()
        };
        let mut opt = AdamW::new(cfg).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();
        // grad=0 なら m=v=0 のままなので update 項も 0 になり、
        // decay=0 のパラメータは不変のはず。
        assert_eq!(out[0].get(&[0]).unwrap(), 1.0);
    }

    /// t=1 の bias correction 込み閉形式との一致を固定する
    /// （`bias_correction1 = 1 - beta1`・`bias_correction2 = 1 - beta2`
    /// のため、`m = (1-beta1)*g`・`v = (1-beta2)*g^2` から
    /// `step_size = lr/(1-beta1)`・`denom = sqrt(v)/sqrt(1-beta2) + eps`
    /// が閉形式で計算できる）。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = AdamWConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let mut opt = AdamW::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let m = (1.0 - cfg.beta1) * g0;
        let v = (1.0 - cfg.beta2) * g0 * g0;
        let bc1 = 1.0 - cfg.beta1 as f64;
        let bc2 = 1.0 - cfg.beta2 as f64;
        let step_size = (cfg.lr as f64 / bc1) as f32;
        let denom = v.sqrt() / (bc2.sqrt() as f32) + cfg.eps;
        let expected = p0 - step_size * m / denom;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "閉形式との不一致: actual={actual} expected={expected}"
        );
    }
}
