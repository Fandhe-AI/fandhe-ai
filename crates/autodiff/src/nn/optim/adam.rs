//! Adam（coupled L2 weight decay。Kingma & Ba, 2015）。
//!
//! `torch.optim.Adam(weight_decay>0)` は decay を**勾配へ加算**
//! （`grad = grad.add(param, alpha=weight_decay)`）してから moment 更新へ
//! 進む「L2 正則化」方式（coupled）であり、[`super::AdamW`]（decay を
//! パラメータへ直接乗算する decoupled 方式。Loshchilov & Hutter, 2019）
//! とは `weight_decay > 0` のとき異なる更新値を生む
//! （`docs/compat-feature-gap.md` §2.9「Adam（coupled L2 weight decay）」・
//! イシュー #1742・親 #1610）。両者は `weight_decay == 0` で完全に一致する
//! （decay 項自体が寄与しないため。`adam_wd_zero_bit_matches_adamw_wd_zero`
//! で bit 一致を固定する）。
//!
//! 本ファイルは `adamw.rs` を意図的に鏡写しにした**別実装**である
//! （共通コードへの抽出はしない。`AdamW` は crates.io 出荷済み公開 API
//! であり、内部ループの共通化 refactor は既存 fixture テストの
//! 統一複合判定〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉では
//! bit ドリフトを検出できないリスクがあるため。`docs/compat-feature-gap.md`
//! §2.9 が示す「`AdamW` の decay 適用箇所を分岐する薄い派生」という
//! 性質は、コード共有ではなく本ファイルの構造的な相似性と、
//! `AdamW` との恒等式テスト（`crates/autodiff/tests/nn_optim_adam.rs`）
//! で担保する）。差分は decay の適用箇所のみ:
//! `AdamW` は `p_decayed = param * (1 - lr*weight_decay)` を先に計算し
//! 生の `grad` で moment を更新するのに対し、`Adam` は
//! `g_eff = grad + weight_decay*param`（`weight_decay != 0` のときのみ。
//! PyTorch `_single_tensor_adam` と同じ分岐で、`weight_decay == 0` では
//! 演算自体を skip し `grad` をそのまま使う。これにより `Adam(wd=0)` は
//! decay 項の乗算が単に `1.0` になるだけの `AdamW(wd=0)` と bit 単位で
//! 同一の演算列になる）で moment を更新し、`param` 自体への decay 乗算は
//! 行わない。
//!
//! `super::mod` doc が示す通り、`step()` は `(param, grad)` の参照列を
//! 受け取り更新後 `Tensor<f32>` の列を返す（`AdamW::step` と同じ
//! シグネチャ・呼び出しパターン）。
//!
//! **`DeviceParamStore` 非対応**（イシュー #1742 のスコープ外）:
//! `crate::optim::device_store::DeviceParamStore::step` は
//! `BackendOps::sgd_step_device` 専用のデバイス常駐更新経路であり、
//! `Adam` を含む Adam 系 optimizer は結線されていない。`Adam::step` は
//! 本ファイルの `AdamW::step` と同様、ホスト `Tensor<f32>` を介した
//! optimizer step のみを提供する
//! （`crates/facade/src/optim.rs`「デバイス常駐更新との違い」節）。
//!
//! 新規 `Op`／`BackendOps` メソッド／`Var` メソッド／VJP は追加しない
//! （`AdamW` と同様、`Tape`/`Var`/`BackendOps` に依存しない値型・純関数。
//! `crates/facade/src/optim.rs`「REQ-12 との整合」節）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.Adam` と同一の既定値。`AdamWConfig::default()` の
/// `weight_decay = 0.01` とは異なり、`torch.optim.Adam` の既定
/// `weight_decay=0` をそのまま採用する（`default_weight_decay_is_zero`
/// でドリフトを固定する）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> AdamConfig {
        AdamConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの 1 次
/// （`m`）・2 次（`v`）モーメント推定値。初回 `step()` 呼び出しで
/// 渡された `param` の shape から遅延初期化する（`AdamW` の
/// `SlotState` と同一構造）。
struct SlotState {
    shape: Vec<usize>,
    m: Vec<f32>,
    v: Vec<f32>,
}

/// Adam（coupled L2 weight decay）optimizer 本体。ハイパーパラメータ
/// （[`AdamConfig`]）と、step 数・bias correction 用の `beta^t` 逐次積・
/// スロットごとのモーメント推定値（`SlotState`）を保持する
/// （`AdamW` と同一構造。フィールド単位の相違はない）。
pub struct Adam {
    config: AdamConfig,
    step_count: u64,
    // β^t の逐次積を f64 で保持する理由は `AdamW::beta1_pow_t` と同一
    // （PyTorch の Python float が f64 であることへ丸め挙動を寄せる）。
    beta1_pow_t: f64,
    beta2_pow_t: f64,
    states: Vec<SlotState>,
}

impl Adam {
    /// ハイパーパラメータを検証して構築する（`AdamW::new` と同一基準:
    /// `lr`/`weight_decay` は有限かつ非負、`beta1`/`beta2` は `[0, 1)`、
    /// `eps` は有限かつ正）。
    pub fn new(config: AdamConfig) -> Result<Adam, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adam::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adam::new: beta1 must be in [0.0, 1.0), got {}",
                config.beta1
            )));
        }
        if !(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adam::new: beta2 must be in [0.0, 1.0), got {}",
                config.beta2
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adam::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Adam::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        Ok(Adam {
            config,
            step_count: 0,
            beta1_pow_t: 1.0,
            beta2_pow_t: 1.0,
            states: Vec::new(),
        })
    }

    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    /// 実行済み `step()` 回数（bias correction の `t`）。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// `AdamW::step` と同一の遅延初期化・検証専用フェーズ／状態変更
    /// フェーズの分離契約（形状エラー発生時に `step_count`／
    /// `beta*_pow_t`／`m`／`v` を部分更新しない fail-closed 契約）を
    /// そのまま踏襲する。
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
                "Adam::step: slot count changed across calls (expected {}, got {}); \
                 Adam state (m/v) is keyed by call-order slot index and cannot be \
                 resized after the first step()",
                self.states.len(),
                params_and_grads.len()
            )));
        }

        // 検証専用フェーズ（状態変更前に全スロットの shape を確認しきる。
        // `AdamW::step` と同じ Bugbot 是正契約）。
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
        let weight_decay = self.config.weight_decay;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            // 読み取り専用走査には `dense_vec_ref`（`Cow<[f32]>`。
            // contiguous 入力でコピー不要）を使う（`AdamW::step` と同じ
            // 変更パターン。イシュー #1026）。
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let mut new_param = Vec::with_capacity(param_data.len());

            for i in 0..param_data.len() {
                // coupled L2（PyTorch `_single_tensor_adam` の
                // `if weight_decay != 0: grad = grad.add(param,
                // alpha=weight_decay)` と同じ分岐）。`weight_decay == 0.0`
                // では演算自体を skip し生の `g` をそのまま使うことで、
                // `AdamW(wd=0)` の decay 乗算（`* 1.0`）と同様に
                // 「decay 項が寄与しない」だけでなく「decay 項の演算列
                // 自体が現れない」形にし、`-0.0`／`±inf` 等の縁で
                // `f32::mul_add(0.0, p, g)` を経由する場合との潜在的な
                // 符号・NaN 差異を避ける（`adam_wd_zero_bit_matches_adamw_wd_zero`
                // が要求する bit 一致の構造的根拠）。
                let g = grad_data[i];
                let g_eff = if weight_decay != 0.0 {
                    f32::mul_add(weight_decay, param_data[i], g)
                } else {
                    g
                };

                let m = f32::mul_add(
                    self.config.beta1,
                    slot.m[i],
                    (1.0 - self.config.beta1) * g_eff,
                );
                let v = f32::mul_add(
                    self.config.beta2,
                    slot.v[i],
                    (1.0 - self.config.beta2) * g_eff * g_eff,
                );
                slot.m[i] = m;
                slot.v[i] = v;

                let denom = v.sqrt() / bias_correction2_sqrt + self.config.eps;
                new_param.push(param_data[i] - step_size * m / denom);
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
        let cfg = AdamConfig {
            lr: -1.0,
            ..AdamConfig::default()
        };
        assert!(matches!(
            Adam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_beta_out_of_range() {
        let cfg = AdamConfig {
            beta1: 1.0,
            ..AdamConfig::default()
        };
        assert!(matches!(
            Adam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let cfg = AdamConfig {
            beta2: -0.1,
            ..AdamConfig::default()
        };
        assert!(matches!(
            Adam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = AdamConfig {
            eps: 0.0,
            ..AdamConfig::default()
        };
        assert!(matches!(
            Adam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = AdamConfig {
            weight_decay: f32::NAN,
            ..AdamConfig::default()
        };
        assert!(matches!(
            Adam::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Adam::new(AdamConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Adam::new(AdamConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Adam::new(AdamConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    /// 形状エラー発生時に `step_count`／`beta*_pow_t`／`m`／`v` が
    /// 部分更新されず呼び出し前の状態のまま残ることを確認する
    /// （`AdamW::state_not_mutated_after_failed_step` と同型の回帰）。
    #[test]
    fn state_not_mutated_after_failed_step() {
        let mut opt = Adam::new(AdamConfig::default()).unwrap();
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

        let mut opt_ref = Adam::new(AdamConfig::default()).unwrap();
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

    /// `AdamConfig::default().weight_decay == 0.0`（PyTorch
    /// `torch.optim.Adam` の既定値）を固定する。`AdamWConfig::default()`
    /// の `0.01` とのドリフトを防ぐ。
    #[test]
    fn default_weight_decay_is_zero() {
        assert_eq!(AdamConfig::default().weight_decay, 0.0);
    }

    /// t=1 の bias correction 込み閉形式との一致を固定する（`wd>0`
    /// を含む。`g_eff = mul_add(wd, p0, g0)` から
    /// `m=(1-beta1)*g_eff`・`v=(1-beta2)*g_eff^2`・
    /// `step_size=lr/(1-beta1)`・`denom=sqrt(v)/sqrt(1-beta2)+eps` が
    /// 閉形式で計算できる。`AdamW::first_step_matches_closed_form` と
    /// 異なり `param_data[i]` 自体には decay を乗算しない点が焦点）。
    #[test]
    fn first_step_matches_closed_form_with_weight_decay() {
        let cfg = AdamConfig {
            lr: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
        };
        let mut opt = Adam::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let g_eff = f32::mul_add(cfg.weight_decay, p0, g0);
        let m = (1.0 - cfg.beta1) * g_eff;
        let v = (1.0 - cfg.beta2) * g_eff * g_eff;
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

    /// coupled 性の直接確認: `grad=0` でも `weight_decay>0` かつ
    /// `param != 0` なら `g_eff != 0` になり m／v が非ゼロへ動く（decay
    /// が「勾配へ混ぜた L2 正則化」であることの固定）。さらに更新が
    /// 純粋な乗算減衰（`AdamW` の `p *= (1 - lr*wd)`）**ではない**ことも
    /// 確認する（coupled と decoupled の差を直接固定。`AdamW::
    /// decoupled_weight_decay_without_grad` との対比）。
    #[test]
    fn coupled_decay_with_zero_grad_moves_moments() {
        let cfg = AdamConfig {
            lr: 0.1,
            weight_decay: 0.2,
            ..AdamConfig::default()
        };
        let mut opt = Adam::new(cfg).unwrap();
        let param = t(vec![1.0, -2.0], &[2]);
        let grad = t(vec![0.0, 0.0], &[2]);

        let out = opt.step(&[(&param, &grad)]).unwrap();
        // grad=0 でも decay により g_eff = wd*param != 0 のはずなので
        // 更新は「param * (1 - lr*wd)」という単純乗算にはならない
        // （AdamW の decoupled 経路と異なり、adaptive スケーリング項
        //〈bias correction・sqrt(v)〉が更新式に混ざるため）。
        let decoupled_style_factor = 1.0 - cfg.lr * cfg.weight_decay;
        let naive_decoupled_0 = param.get(&[0]).unwrap() * decoupled_style_factor;
        let actual_0 = out[0].get(&[0]).unwrap();
        assert!(
            (actual_0 - naive_decoupled_0).abs() > 1e-6,
            "coupled L2 の更新が decoupled 方式の単純乗算と一致してしまっている: \
             actual={actual_0} naive_decoupled={naive_decoupled_0}"
        );
    }
}
