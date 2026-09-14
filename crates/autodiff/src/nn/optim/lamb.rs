//! LAMB（Layer-wise Adaptive Moments optimizer for Batch training。
//! You et al., 2019, "Large Batch Optimization for Deep Learning:
//! Training BERT in 76 minutes", arXiv:1904.00962 Algorithm 2）。
//!
//! `torch.optim` 本体に LAMB 相当の実装は**ない**（TF Addons／apex／
//! `torch_optimizer` 由来。`docs/compat-feature-gap.md` §2.9）。本実装は
//! Algorithm 2（bias correction 込み）をそのまま再現し、φ（trust ratio
//! のスケーリング関数）は恒等写像固定とする——`torch_optimizer.Lamb`
//! の `‖x‖` clamp（上限 10）や apex の `max_grad_norm`／NVLAMB 除外は
//! **採用しない**（イシュー #1744・親 #1610「対象外」節）。
//!
//! **`AdamW`（decoupled）・`Adam`（coupled L2。`super::adam`）とは
//! weight decay の適用が異なる**: LAMB は paper 定義どおり decay を
//! 更新方向 `u` へ coupled で織り込む（`u = r + weight_decay * x`）。
//! `AdamW` の乗算 decoupled 減衰とは構造が異なる点に注意。
//!
//! # 実装形（Algorithm 2 と数学的に同値）
//!
//! パラメータテンソル `x`（1 スロット = 1 layer）ごとに、moment 更新
//! （`m`／`v`・bias correction）は [`super::AdamW`] と同一の演算列
//! （`step_size = lr / bias_correction1`・`denom = sqrt(v) /
//! sqrt(bias_correction2) + eps`）を使う。trust ratio 適用は `lr` を
//! 先に折り込んだ形で計算する:
//!
//! ```text
//! s_i = step_size * m_i / denom_i                    // = lr * r_i（AdamW と同一の s）
//! t_i = fma(lr * weight_decay, x_i, s_i)              // = lr * u_i（weight_decay=0 では t_i = s_i）
//! f   = lr * ‖x‖₂ / ‖t‖₂                              // = trust ratio（‖x‖₂==0 または ‖t‖₂==0 のとき f=1.0）
//! x_new_i = fma(-f, t_i, x_i)
//! ```
//!
//! `f * t_i = lr * trust * u_i` であり paper の更新 `x - lr*trust*u` と
//! 同値（`f = lr*‖x‖/‖t‖ = lr*‖x‖/(lr*‖u‖) = ‖x‖/‖u‖ = trust`）。
//! `‖x‖₂ == 0`（fallback `f = 1.0`）かつ `weight_decay == 0` のとき
//! `x_new_i = x_i - s_i` となり **[`super::AdamW`]（`weight_decay=0`）の
//! 1 step と bit 完全一致**する（`crates/autodiff/tests/nn_optim_lamb.rs`
//! の恒等式テストが検証する）。`lr == 0` は `t = 0` → `‖t‖=0` →
//! fallback `f=1.0` → 更新量 `f*t = 0` となり特別扱い不要。
//!
//! trust ratio は 1 パラメータテンソルごとに独立に計算する（複数の
//! `(param, grad)` ペアを `step()` へ渡しても、norm は各ペアの要素
//! だけで縮約し、ペア間で合算しない。paper の「layer-wise」の意味）。
//!
//! # norm の数値契約
//!
//! `‖x‖₂`・`‖t‖₂` は f64 アキュムレータの逐次和（index 順に
//! `acc += (v as f64) * (v as f64)`）→ f64 で `sqrt` → f64 のまま
//! `f = lr * norm_x / norm_t` を計算し、**1 回だけ** `f32` へ downcast
//! する（`.claude/rules/coding-rust.md`「勾配の長軸縮約は f64
//! アキュムレータで統一する」に従う。LAMB はホスト側 optimizer のため
//! `Var::norm_l2`〈`Op::VectorNorm`。tape が必要〉は呼ばない——コード
//! 依存ではなく同じ f64 縮約契約に従うという意味での整合）。
//!
//! # 非有限 norm の扱い（fail-closed）
//!
//! `norm_x`／`norm_t`（および導出される trust ratio `f`）のいずれかが
//! 非有限（NaN／Inf）になった場合、`step()` は `Err(InvalidArgument)`
//! を返す。[`super::AdamW`]／[`super::Adam`] が非有限勾配を黙って
//! パラメータへ伝播させるのとは意図的に異なる——trust ratio は 1
//! テンソル全体で共有するスカラー係数であり、1 要素の NaN／Inf が
//! `norm_t`（ひいては `f`）を汚染するとテンソル全体の更新値が破壊
//! されるため、検証段階で明示的に拒否する（`.claude/rules/security.md`
//! A08）。
//!
//! **`norm_x`／`norm_t`／`f` の検査だけでは不十分な経路がある**
//! （codex-review 指摘・PR #1856）: 勾配 `g` 自体は有限でも
//! `g*g`（`v` 更新の一部）が f32 の表現域を超えて `v` が非有限（Inf）
//! になりうる。この非有限な `v` は `denom = sqrt(v)/... + eps` を
//! 経て `s`／`t` をゼロへ押しつぶすことがあり、その場合 `norm_t` は
//! 有限（ゼロ）に収まって上記の norm 検査を通過してしまう
//! （trust ratio はゼロ除算を通らない `norm_t==0` フォールバック
//! 経路に入る）。これを放置すると非有限な `v` がそのままコミット
//! フェーズで `self.states` へ保存され、以後 `grad` が有限値へ戻っても
//! `v = beta2*Inf + ...` は Inf のまま回復せず当該要素の更新が恒久的に
//! 停止する。このため `m`／`v` そのものの有限性も、状態へコミットする
//! 前の計算フェーズで直接検証する（`step()` 内 `if !m.is_finite() ||
//! !v.is_finite()` 分岐）。
//!
//! `step()` は検証（状態変更なし）→ 計算（状態変更なし。ここで
//! `m`／`v`・非有限 norm を検出する）→ コミット（ここで初めて状態を
//! 変更する）の 3 フェーズで構成する。どのフェーズで失敗しても
//! `step_count`／`beta*_pow_t`／`m`／`v`（初回呼び出しの場合は
//! `self.states` 自体）は一切変更されない
//! （`crates/autodiff/tests/nn_optim_lamb.rs`・本ファイル末尾の
//! ユニットテストで固定する）。
//!
//! # `DeviceParamStore` 非対応
//!
//! `crate::optim::device_store::DeviceParamStore::step` は
//! `BackendOps::sgd_step_device` 専用のデバイス常駐更新経路であり、
//! `Lamb` は結線されていない（`AdamW`／`Adam` と同様）。LAMB の
//! デバイス常駐化にはパラメータテンソルごとの L2 norm reduction
//! カーネルと trust ratio 適用カーネル（3 バックエンド）が必要で、
//! 本イシュー（#1744）の対象外。`Lamb::step` はホスト `Tensor<f32>`
//! を介した optimizer step のみを提供する。
//!
//! `super::mod` doc が示す通り、`step()` は `(param, grad)` の参照列を
//! 受け取り更新後 `Tensor<f32>` の列を返す（`AdamW::step`／`Adam::step`
//! と同一シグネチャ）。
//!
//! 新規 `Op`／`BackendOps` メソッド／`Var` メソッド／VJP は追加しない
//! （`AdamW`／`Adam` と同様、`Tape`/`Var`/`BackendOps` に依存しない
//! 値型・純関数。`crates/facade/src/optim.rs`「REQ-12 との整合」節）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// LAMB のハイパーパラメータ。フィールド構成は [`super::AdamWConfig`]
/// と同一。既定値は paper／apex／`torch_optimizer` 共通の LAMB 既定
/// （`eps=1e-6` は `AdamW`／`Adam` の `1e-8` と異なる点に注意）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for LambConfig {
    fn default() -> LambConfig {
        LambConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの 1 次
/// （`m`）・2 次（`v`）モーメント推定値。初回 `step()` 呼び出しの
/// コミットフェーズで確定する（`AdamW::SlotState` と同一構造）。
struct SlotState {
    shape: Vec<usize>,
    m: Vec<f32>,
    v: Vec<f32>,
}

/// LAMB optimizer 本体。ハイパーパラメータ（[`LambConfig`]）と、step
/// 数・bias correction 用の `beta^t` 逐次積・スロットごとのモーメント
/// 推定値（`SlotState`）を保持する。
pub struct Lamb {
    config: LambConfig,
    step_count: u64,
    // β^t の逐次積を f64 で保持する理由は `AdamW::beta1_pow_t` と同一
    // （PyTorch の Python float が f64 であることへ丸め挙動を寄せる）。
    beta1_pow_t: f64,
    beta2_pow_t: f64,
    states: Vec<SlotState>,
}

impl Lamb {
    /// ハイパーパラメータを検証して構築する（`AdamW::new`／`Adam::new`
    /// と同一基準: `lr`/`weight_decay` は有限かつ非負、`beta1`/`beta2`
    /// は `[0, 1)`、`eps` は有限かつ正）。
    pub fn new(config: LambConfig) -> Result<Lamb, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lamb::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.beta1.is_finite() && (0.0..1.0).contains(&config.beta1)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lamb::new: beta1 must be in [0.0, 1.0), got {}",
                config.beta1
            )));
        }
        if !(config.beta2.is_finite() && (0.0..1.0).contains(&config.beta2)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lamb::new: beta2 must be in [0.0, 1.0), got {}",
                config.beta2
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lamb::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Lamb::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        Ok(Lamb {
            config,
            step_count: 0,
            beta1_pow_t: 1.0,
            beta2_pow_t: 1.0,
            states: Vec::new(),
        })
    }

    /// 構築時に確定したハイパーパラメータへの参照。
    pub fn config(&self) -> &LambConfig {
        &self.config
    }

    /// 実行済み `step()` 回数（bias correction の `t`）。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// モジュール doc「非有限 norm の扱い（fail-closed）」節が示す
    /// 3 フェーズ（検証 → 計算 → コミット）で構成する。検証・計算の
    /// いずれのフェーズも `self.states`／`step_count`／`beta*_pow_t`
    /// を変更しない（初回呼び出しが失敗した場合、`self.states` は
    /// 空のまま残る）。
    pub fn step(
        &mut self,
        params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        // 検証フェーズ（状態変更なし。`self.states` の遅延初期化も
        // 行わない）: `grad.shape() == param.shape()` はスロット数・
        // 既存状態を一切参照せず判定できるため先に全ペアを検証する
        // （`Adam::step` と同じ理由。codex-review 指摘対応の並び順）。
        for (param, grad) in params_and_grads.iter() {
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
                    "Lamb::step: slot count changed across calls (expected {}, got {}); \
                     Lamb state (m/v) is keyed by call-order slot index and cannot be \
                     resized after the first step()",
                    self.states.len(),
                    params_and_grads.len()
                )));
            }
            for (slot, (param, _grad)) in self.states.iter().zip(params_and_grads.iter()) {
                if param.shape() != slot.shape.as_slice() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: param.shape().to_vec(),
                        rhs: slot.shape.clone(),
                    }));
                }
            }
        }

        // 次の step のハイパーパラメータ由来の係数をローカル値として
        // 計算する（`self` への書き込みはコミットフェーズまで行わない）。
        let beta1_pow_t = self.beta1_pow_t * self.config.beta1 as f64;
        let beta2_pow_t = self.beta2_pow_t * self.config.beta2 as f64;
        let bias_correction1 = 1.0 - beta1_pow_t;
        let bias_correction2 = 1.0 - beta2_pow_t;
        let bias_correction2_sqrt = bias_correction2.sqrt() as f32;
        let step_size = (self.config.lr as f64 / bias_correction1) as f32;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;

        // 計算フェーズ（状態変更なし）: 全スロットの新 m/v/t/trust
        // ratio をスクラッチへ計算する。いずれかのスロットで非有限
        // norm を検出したら、他スロットの計算結果も含めて一切コミット
        // せず即 `Err` を返す（`self.states` は未変更のまま）。
        struct Scratch {
            m: Vec<f32>,
            v: Vec<f32>,
            t: Vec<f32>,
            f: f32,
        }

        let mut scratch: Vec<Scratch> = Vec::with_capacity(params_and_grads.len());
        for (idx, (param, grad)) in params_and_grads.iter().enumerate() {
            let param_data = dense_vec_ref(param);
            let grad_data = dense_vec_ref(grad);
            let existing = self.states.get(idx);
            let mut new_m = Vec::with_capacity(param_data.len());
            let mut new_v = Vec::with_capacity(param_data.len());
            let mut t = Vec::with_capacity(param_data.len());
            // trust ratio の分子・分母（このスロットの要素のみで縮約
            // する。「layer-wise」契約: ペア間で norm を合算しない）。
            let mut norm_x_sq: f64 = 0.0;
            let mut norm_t_sq: f64 = 0.0;

            for i in 0..param_data.len() {
                let g = grad_data[i];
                let prev_m = existing.map(|s| s.m[i]).unwrap_or(0.0);
                let prev_v = existing.map(|s| s.v[i]).unwrap_or(0.0);
                let m = f32::mul_add(self.config.beta1, prev_m, (1.0 - self.config.beta1) * g);
                let v = f32::mul_add(self.config.beta2, prev_v, (1.0 - self.config.beta2) * g * g);
                // codex-review 指摘（PR #1856）: `g` 自体は有限でも
                // `g*g` が f32 の表現域を超えて `v` が非有限（Inf）に
                // なりうる（例: `g=1e30` → `g*g=1e60` は overflow）。
                // 後続の `denom`／`s`／`t` の計算で偶然 `t=0` になり
                // `norm_t` が有限（ゼロ）へ収まってしまうと、モジュール
                // doc「非有限 norm の扱い」節の `norm_x`／`norm_t`／`f`
                // の有限性検査だけではこの非有限な `v`（`m` も同様）を
                // 検出できず、後段のコミットフェーズで `slot.v` へ Inf
                // がそのまま保存されてしまう。以後 `grad` が有限値へ
                // 戻っても `v = beta2*Inf + ...` は Inf のまま回復せず
                // 当該要素の更新が恒久的に停止する（trust ratio は
                // ゼロ除算を通らないため既存の `norm_t==0` フォール
                // バック経路では検出できない）。状態（`m`／`v`）そのもの
                // の有限性を計算フェーズで直接検証し、非有限なら状態を
                // 一切変更せず `Err` を返す（モジュール doc「非有限
                // norm の扱い（fail-closed）」節と同じ 3 フェーズ契約
                // に従う）。
                if !m.is_finite() || !v.is_finite() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Lamb::step: slot {idx}: non-finite second moment estimate \
                         (m={m}, v={v}) at element {i}; refusing to commit state that \
                         would become unrecoverable (see module doc \"非有限 norm の扱い\")"
                    )));
                }
                new_m.push(m);
                new_v.push(v);

                let denom = v.sqrt() / bias_correction2_sqrt + self.config.eps;
                let s = step_size * m / denom;
                // `weight_decay == 0.0` では decay 項の演算自体を skip
                // する（`Adam` の coupled decay 分岐と同じ理由で、
                // `AdamW(wd=0)` との恒等式テストが要求する bit 一致を
                // 成り立たせるため。`t_i = s_i` のまま `fma` を通さない）。
                let ti = if weight_decay == 0.0 {
                    s
                } else {
                    f32::mul_add(lr * weight_decay, param_data[i], s)
                };
                t.push(ti);

                let xi = param_data[i] as f64;
                norm_x_sq += xi * xi;
                let ti64 = ti as f64;
                norm_t_sq += ti64 * ti64;
            }

            let norm_x = norm_x_sq.sqrt();
            let norm_t = norm_t_sq.sqrt();
            if !norm_x.is_finite() || !norm_t.is_finite() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Lamb::step: slot {idx}: non-finite parameter or update norm \
                     (norm_x={norm_x}, norm_t={norm_t}); trust ratio cannot be computed"
                )));
            }
            let f: f32 = if norm_x == 0.0 || norm_t == 0.0 {
                // どちらかの norm がゼロのときは trust ratio を 1.0
                // へフォールバックする（モジュール doc「実装形」節。
                // `weight_decay == 0` かつ `norm_x == 0` では
                // `x_new = x - t = x - s` となり `AdamW(wd=0)` の
                // 1 step と bit 一致する）。
                1.0
            } else {
                (lr as f64 * norm_x / norm_t) as f32
            };
            if !f.is_finite() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Lamb::step: slot {idx}: trust ratio is non-finite \
                     (norm_x={norm_x}, norm_t={norm_t})"
                )));
            }

            scratch.push(Scratch {
                m: new_m,
                v: new_v,
                t,
                f,
            });
        }

        // コミットフェーズ: ここで初めて状態を変更する。計算フェーズが
        // 全スロットで成功した場合のみ到達する。
        if self.states.is_empty() && !params_and_grads.is_empty() {
            self.states = params_and_grads
                .iter()
                .map(|(param, _)| SlotState {
                    shape: param.shape().to_vec(),
                    m: Vec::new(),
                    v: Vec::new(),
                })
                .collect();
        }
        self.step_count += 1;
        self.beta1_pow_t = beta1_pow_t;
        self.beta2_pow_t = beta2_pow_t;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for ((slot, sc), (param, _grad)) in self
            .states
            .iter_mut()
            .zip(scratch)
            .zip(params_and_grads.iter())
        {
            slot.m = sc.m;
            slot.v = sc.v;

            let param_data = dense_vec_ref(param);
            let mut new_param = Vec::with_capacity(param_data.len());
            for i in 0..param_data.len() {
                new_param.push(f32::mul_add(-sc.f, sc.t[i], param_data[i]));
            }
            out.push(Tensor::new(new_param, &slot.shape)?);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::adamw::{AdamW, AdamWConfig};
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    #[test]
    fn rejects_negative_lr() {
        let cfg = LambConfig {
            lr: -1.0,
            ..LambConfig::default()
        };
        assert!(matches!(
            Lamb::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_beta_out_of_range() {
        let cfg = LambConfig {
            beta1: 1.0,
            ..LambConfig::default()
        };
        assert!(matches!(
            Lamb::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let cfg = LambConfig {
            beta2: -0.1,
            ..LambConfig::default()
        };
        assert!(matches!(
            Lamb::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = LambConfig {
            eps: 0.0,
            ..LambConfig::default()
        };
        assert!(matches!(
            Lamb::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = LambConfig {
            weight_decay: f32::NAN,
            ..LambConfig::default()
        };
        assert!(matches!(
            Lamb::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = Lamb::new(LambConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = Lamb::new(LambConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = Lamb::new(LambConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    /// 非有限勾配は非有限 norm を経て `Err(InvalidArgument)` になる
    /// （モジュール doc「非有限 norm の扱い（fail-closed）」節。
    /// `AdamW`／`Adam` が非有限勾配を黙って伝播させるのとは異なる）。
    #[test]
    fn non_finite_grad_is_rejected() {
        let mut opt = Lamb::new(LambConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![f32::NAN, 0.1], &[2]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
        assert_eq!(
            opt.step_count(),
            0,
            "非有限 norm エラー時に step_count が更新されてはならない"
        );

        let grad_inf = t(vec![f32::INFINITY, 0.1], &[2]);
        let result_inf = opt.step(&[(&param, &grad_inf)]);
        assert!(matches!(result_inf, Err(AutodiffError::InvalidArgument(_))));
        assert_eq!(opt.step_count(), 0);
    }

    /// codex-review 指摘（PR #1856）の回帰テスト: 勾配自体は有限
    /// （`g=1e30`）でも `v = beta2*prev_v + (1-beta2)*g*g` の `g*g`
    /// が f32 の表現域を超えて非有限（Inf）になりうる。この非有限な
    /// `v` は `denom = sqrt(v)/... + eps` を経て `s = step_size*m/denom`
    /// をゼロへ押しつぶすため `t=0` となり、`norm_t` は有限（ゼロ）に
    /// 収まってしまう——モジュール doc「非有限 norm の扱い」節の
    /// `norm_x`／`norm_t`／`f` の有限性検査だけでは検出できない
    /// （trust ratio はゼロ除算を通らない `norm_t==0` フォールバック
    /// 経路に入るため）。本テストは `m`／`v` 自体の有限性検査
    /// （上記コメント参照）がこの経路を正しく検出し、非有限な `v` が
    /// `self.states` へコミットされないこと、および直後に有限な勾配で
    /// 再試行した際に更新が恒久的に停止しない（フレッシュな状態から
    /// 正常に 1 step 進む）ことを確認する。
    #[test]
    fn non_finite_second_moment_from_extreme_finite_grad_is_rejected_without_state_corruption() {
        let mut opt = Lamb::new(LambConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad_extreme = t(vec![1e30_f32], &[1]);

        let result = opt.step(&[(&param, &grad_extreme)]);
        assert!(
            matches!(result, Err(AutodiffError::InvalidArgument(_))),
            "非有限な二次モーメント（v）は状態未変更のまま拒否されるはず: {result:?}"
        );
        assert_eq!(
            opt.step_count(),
            0,
            "非有限な二次モーメント検出時に step_count が更新されてはならない"
        );

        // 直後に有限な勾配で再試行すると、フレッシュな初回呼び出しと
        // して正常に成功する（＝拒否された呼び出しの Inf が
        // `self.states` へ残って以後の更新を停止させていないことの
        // 直接証拠）。
        let grad_normal = t(vec![0.1_f32], &[1]);
        let updated = opt
            .step(&[(&param, &grad_normal)])
            .unwrap_or_else(|e| panic!("非有限 v 拒否直後の有限勾配での再試行が失敗: {e}"));
        assert_eq!(updated.len(), 1);
        assert_eq!(opt.step_count(), 1);
        assert!(
            updated[0].get(&[0]).unwrap().is_finite(),
            "状態破損がなければ更新後の値は有限のはず"
        );
    }

    /// codex-review（P2）・Cursor Bugbot 指摘（PR #1852）と同型の回帰
    /// テスト: 形状エラー・非有限 norm エラーいずれで `step()` が
    /// 失敗しても `self.states` が未変更（初回呼び出しなら空のまま）
    /// であることを、失敗した呼び出しとは異なる shape の param で
    /// 再試行して間接的に確認する（`nn_optim_adam.rs::
    /// adam_step_rejects_bad_grad_shape_without_mutating_state_on_first_call`
    /// と同じ検証手法）。
    #[test]
    fn state_not_mutated_after_failed_step() {
        // 1) shape エラー（初回呼び出し）。
        let mut opt = Lamb::new(LambConfig::default()).unwrap();
        let bad_param = t(vec![1.0, 2.0, 3.0], &[3]);
        let bad_grad = t(vec![0.1, 0.2], &[2]);
        let err = opt.step(&[(&bad_param, &bad_grad)]);
        assert!(matches!(err, Err(AutodiffError::Shape(_))));
        assert_eq!(opt.step_count(), 0);

        // 失敗した呼び出しとは異なる shape（`[2]`）で再試行し、真の
        // 初回呼び出しとして成功することを確認する。
        let other_param = t(vec![10.0, 20.0], &[2]);
        let other_grad = t(vec![0.1, 0.2], &[2]);
        let updated = opt
            .step(&[(&other_param, &other_grad)])
            .unwrap_or_else(|e| panic!("shape エラー直後の異なる shape での再試行が失敗: {e}"));
        assert_eq!(updated.len(), 1);
        assert_eq!(opt.step_count(), 1);

        // 2) 非有限 norm エラー（初回呼び出し）。
        let mut opt2 = Lamb::new(LambConfig::default()).unwrap();
        let nan_param = t(vec![1.0, 2.0, 3.0], &[3]);
        let nan_grad = t(vec![f32::NAN, 0.1, 0.2], &[3]);
        let err2 = opt2.step(&[(&nan_param, &nan_grad)]);
        assert!(matches!(err2, Err(AutodiffError::InvalidArgument(_))));
        assert_eq!(opt2.step_count(), 0);

        let other_param2 = t(vec![5.0, 6.0], &[2]);
        let other_grad2 = t(vec![0.1, 0.2], &[2]);
        let updated2 = opt2
            .step(&[(&other_param2, &other_grad2)])
            .unwrap_or_else(|e| {
                panic!("非有限 norm エラー直後の異なる shape での再試行が失敗: {e}")
            });
        assert_eq!(updated2.len(), 1);
        assert_eq!(opt2.step_count(), 1);
    }

    /// モジュール doc「実装形」節の恒等式: `x = 0` テンソル・
    /// `weight_decay = 0` では trust ratio が 1.0 へフォールバックし、
    /// 更新値が [`super::AdamW`]`(weight_decay=0)` の 1 step と bit
    /// 完全一致する。
    #[test]
    fn zero_norm_param_falls_back_to_trust_ratio_one() {
        let cfg = LambConfig {
            weight_decay: 0.0,
            ..LambConfig::default()
        };
        let mut lamb = Lamb::new(cfg).unwrap();
        let mut adamw = AdamW::new(AdamWConfig {
            lr: cfg.lr,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            eps: cfg.eps,
            weight_decay: 0.0,
        })
        .unwrap();

        let param = t(vec![0.0, 0.0, 0.0], &[3]);
        let grad = t(vec![0.3, -0.2, 0.1], &[3]);

        let lamb_out = lamb.step(&[(&param, &grad)]).unwrap();
        let adamw_out = adamw.step(&[(&param, &grad)]).unwrap();

        for i in 0..3 {
            let a = lamb_out[0].get(&[i]).unwrap();
            let b = adamw_out[0].get(&[i]).unwrap();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "index={i}: x=0・wd=0 では Lamb と AdamW(wd=0) は bit 一致するはず (a={a} b={b})"
            );
        }
    }

    /// `lr = 0` では `t = 0` となり trust ratio フォールバックにより
    /// 更新量がゼロになる（`x_new == x`。`Err` にはならない）。
    #[test]
    fn lr_zero_yields_no_update() {
        let cfg = LambConfig {
            lr: 0.0,
            ..LambConfig::default()
        };
        let mut opt = Lamb::new(cfg).unwrap();
        let param = t(vec![1.0, -2.0, 3.0], &[3]);
        let grad = t(vec![0.3, -0.2, 0.1], &[3]);

        let out = opt.step(&[(&param, &grad)]).unwrap();
        for i in 0..3 {
            assert_eq!(
                out[0].get(&[i]).unwrap().to_bits(),
                param.get(&[i]).unwrap().to_bits(),
                "index={i}: lr=0 では更新後の値が元の param と bit 一致するはず"
            );
        }
    }
}
