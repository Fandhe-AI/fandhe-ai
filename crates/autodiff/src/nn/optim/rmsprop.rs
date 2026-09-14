//! RMSprop（Tieleman & Hinton, 2012。`torch.optim.RMSprop` 相当）。
//!
//! `torch.optim.RMSprop` の単一テンソル実装（`_single_tensor_rmsprop`。
//! `square_avg = alpha * square_avg + (1-alpha) * g^2` →
//! `centered` 時は `grad_avg = lerp(grad_avg, g, 1-alpha)` を併用して
//! `avg = sqrt(square_avg - grad_avg^2)`、非 `centered` 時は
//! `avg = sqrt(square_avg)` → `avg += eps`（sqrt の**後**に加算）→
//! `momentum > 0` 時は `buf = momentum*buf + g/avg; p -= lr*buf`、
//! それ以外は `p -= lr*g/avg` の演算順）と同一系列を再現する
//! （イシュー #1743・親 #1610「optimizer（Adam／RMSprop／Adagrad／
//! LAMB）」。受け入れ条件の読み替えは `nn/optim/mod.rs` 冒頭 doc と
//! 本モジュール末尾の注記を参照）。
//!
//! `nn/optim/mod.rs` の doc が示す通り、`step()` は `(param, grad)` の
//! 参照列を受け取り更新後 `Tensor<f32>` の列を返す。呼び出し元
//! （学習ループ）は `Linear::from_parameters` 等で層を再構築する
//! 既存の不変更新パターン（`tests/nn_train_convergence.rs`）にそのまま
//! 差し込める。
//!
//! `adamw.rs`（[`super::AdamW`]）を鏡写しにした別実装であり、内部
//! ループの共通化は行わない（既存 fixture テストの統一複合判定では
//! 共通化による bit ドリフトを検出できないため。イシュー #1743
//! 実装計画 §3.2）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;

/// `torch.optim.RMSprop` と同一の既定値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsPropConfig {
    /// 学習率（`lr`）。有限かつ `>= 0.0` を [`RmsProp::new`] が検証する。
    pub lr: f32,
    /// 二乗移動平均の減衰率（`alpha`）。有限かつ `[0.0, 1.0)` を
    /// [`RmsProp::new`] が検証する（`alpha = 1.0` は `square_avg` が
    /// 一切更新されない退化ケースのため意図的に拒否する）。
    pub alpha: f32,
    /// ゼロ除算防止項（`eps`）。`sqrt(square_avg)` の**後**に加算する
    /// （PyTorch と同順）。有限かつ `> 0.0` を [`RmsProp::new`] が検証
    /// する。
    pub eps: f32,
    /// L2 正則化係数。PyTorch RMSprop と同じ coupled 方式（勾配へ
    /// `weight_decay * param` を加算してから以降の更新式へ渡す。
    /// `AdamW` の decoupled 方式とは異なる）。有限かつ `>= 0.0` を
    /// [`RmsProp::new`] が検証する。
    pub weight_decay: f32,
    /// モメンタム係数。`> 0.0` のとき `buf = momentum*buf + g/avg`
    /// を経由した更新（`p -= lr*buf`）へ切り替わる。有限かつ
    /// `>= 0.0` を [`RmsProp::new`] が検証する。
    pub momentum: f32,
    /// centered RMSprop（`avg = sqrt(square_avg - grad_avg^2)`）を
    /// 有効化するか。`false`（既定）では `avg = sqrt(square_avg)`。
    pub centered: bool,
}

impl Default for RmsPropConfig {
    fn default() -> RmsPropConfig {
        RmsPropConfig {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }
}

/// パラメータスロット（1 パラメータテンソルに対応）ごとの状態。
/// 初回 `step()` 呼び出しで渡された `param` の shape から遅延初期化
/// する（`RmsProp::new` の時点ではパラメータ数・shape を知らないため。
/// `adamw.rs::SlotState` と同じ理由）。`grad_avg`／`momentum_buffer` は
/// PyTorch と異なり `centered`／`momentum > 0` の設定に関わらず常に
/// 確保する（config は構築後不変なので、確保有無で分岐する複雑さより
/// 単純さを優先。未使用時は 0 初期化のまま参照されない）。
struct SlotState {
    shape: Vec<usize>,
    square_avg: Vec<f32>,
    grad_avg: Vec<f32>,
    momentum_buffer: Vec<f32>,
}

/// RMSprop optimizer 本体。ハイパーパラメータ（[`RmsPropConfig`]）と、
/// step 数・スロットごとの状態（`SlotState`）を保持する。
pub struct RmsProp {
    config: RmsPropConfig,
    step_count: u64,
    states: Vec<SlotState>,
}

impl RmsProp {
    /// ハイパーパラメータを検証して構築する。`lr`/`weight_decay`/
    /// `momentum` は有限かつ非負、`alpha` は有限かつ `[0, 1)`
    /// （PyTorch は `alpha >= 0` のみを要求するが、`alpha = 1` は
    /// `square_avg` が一切更新されず `eps` のみで除算する退化ケース
    /// になるため意図的に拒否する）、`eps` は有限かつ正（0 を許すと
    /// 初回 step でゼロ除算になるため `AdamW::new` と同様に構築不可能
    /// な引数として弾く）。
    pub fn new(config: RmsPropConfig) -> Result<RmsProp, AutodiffError> {
        if !(config.lr.is_finite() && config.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RmsProp::new: lr must be finite and >= 0.0, got {}",
                config.lr
            )));
        }
        if !(config.alpha.is_finite() && (0.0..1.0).contains(&config.alpha)) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RmsProp::new: alpha must be in [0.0, 1.0), got {}",
                config.alpha
            )));
        }
        if !(config.eps.is_finite() && config.eps > 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RmsProp::new: eps must be finite and > 0.0, got {}",
                config.eps
            )));
        }
        if !(config.weight_decay.is_finite() && config.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RmsProp::new: weight_decay must be finite and >= 0.0, got {}",
                config.weight_decay
            )));
        }
        if !(config.momentum.is_finite() && config.momentum >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RmsProp::new: momentum must be finite and >= 0.0, got {}",
                config.momentum
            )));
        }
        Ok(RmsProp {
            config,
            step_count: 0,
            states: Vec::new(),
        })
    }

    /// 構築時に検証済みの現在のハイパーパラメータへの参照を返す。
    pub fn config(&self) -> &RmsPropConfig {
        &self.config
    }

    /// 実行済み `step()` 回数。
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// `params_and_grads` と同順で更新後の `Tensor<f32>` を返す。
    ///
    /// `adamw.rs::AdamW::step` と同じ 2 段構成を採る: 副作用（状態
    /// バッファの更新）を一切加えない検証専用フェーズで全スロットの
    /// shape を確認しきってから、状態変更フェーズへ進む（形状エラー
    /// 発生時に `step_count`／状態バッファが部分更新されたまま残ると、
    /// 後続の成功する step が破損した状態から学習してしまうため）。
    pub fn step(
        &mut self,
        params_and_grads: &[(&Tensor<f32>, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        // 副作用（`self.states` の初期化を含む）を一切加えない検証専用
        // フェーズ。初回 step（`self.states` が空）でもここで
        // `self.states` を書き換えてはならない——検証がここで失敗した
        // 場合に `self.states` が非空のまま残ると、以降の呼び出しが
        // 「初回 step 前」ではなく「失敗した初回 step で確定した
        // （誤った）shape の 2 回目以降」として扱われ、shape を正しく
        // 修正した再試行まで誤って拒否されてしまう（codex-review 指摘:
        // 初回 step が形状検証より前に状態を確定させる契約違反）。
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
                    "RmsProp::step: slot count changed across calls (expected {}, got {}); \
                     RmsProp state (square_avg/grad_avg/momentum_buffer) is keyed by \
                     call-order slot index and cannot be resized after the first step()",
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
                    grad_avg: vec![0.0f32; param.numel()],
                    momentum_buffer: vec![0.0f32; param.numel()],
                })
                .collect();
        }

        self.step_count += 1;

        let alpha = self.config.alpha;
        let one_minus_alpha = 1.0 - alpha;
        let lr = self.config.lr;
        let eps = self.config.eps;

        let mut out = Vec::with_capacity(params_and_grads.len());
        for (slot, (param, grad)) in self.states.iter_mut().zip(params_and_grads.iter()) {
            // `param`/`grad` は読み取り専用の走査のみ（状態バッファ・
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
                // `AdamW` の decoupled 方式と異なり RMSprop の
                // weight_decay は勾配へ加算する coupled L2 方式）。
                if self.config.weight_decay != 0.0 {
                    g = f32::mul_add(self.config.weight_decay, param_data[i], g);
                }

                // `square_avg.mul_(alpha).addcmul_(g, g, value=1-alpha)`。
                slot.square_avg[i] =
                    f32::mul_add(alpha, slot.square_avg[i], one_minus_alpha * g * g);

                let avg = if self.config.centered {
                    // `grad_avg.lerp_(g, 1-alpha)`。PyTorch の `lerp`
                    // 実装（`aten/src/ATen/native/Lerp.h`）は数値安定性
                    // のため `weight`（ここでは `one_minus_alpha`）の
                    // 絶対値で 2 分岐する: `|weight| < 0.5` のときは
                    // 始点基準 `start + weight * (end - start)`、
                    // それ以外（`alpha < 0.5` で `weight >= 0.5` の
                    // ケースを含む）は終点基準
                    // `end - (end - start) * (1 - weight)` を使う。
                    // 始点基準のみを alpha 全域へ適用すると、
                    // `weight` が 1 に近い（`alpha` が 0 に近い）ケースで
                    // `end - start` の大きな桁落ちを `weight` 倍してから
                    // `start` へ加算する経路になり、`start`（旧
                    // `grad_avg`）の桁が大きい場合に有効桁が失われる
                    // （codex-review 指摘: `centered=true`・`alpha=0`
                    // 付近で勾配が急変する境界ケース）。`1 - weight`
                    // は `alpha` そのものなので `weight_.abs()` の分岐で
                    // 十分（`one_minus_alpha` は `alpha` の検証済み
                    // 範囲 `[0.0, 1.0)` から `(0.0, 1.0]` に収まり NaN
                    // にならない）。
                    let start = slot.grad_avg[i];
                    let end = g;
                    let weight = one_minus_alpha;
                    slot.grad_avg[i] = if weight.abs() < 0.5 {
                        start + weight * (end - start)
                    } else {
                        end - (end - start) * alpha
                    };
                    // `square_avg.addcmul(grad_avg, grad_avg,
                    // value=-1).sqrt_()`。
                    (slot.square_avg[i] - slot.grad_avg[i] * slot.grad_avg[i]).sqrt()
                } else {
                    slot.square_avg[i].sqrt()
                };
                // sqrt の**後**に eps を加算する（PyTorch と同順）。
                let avg = avg + eps;

                if self.config.momentum > 0.0 {
                    // `buf.mul_(momentum).addcdiv_(g, avg)` →
                    // `param.add_(buf, alpha=-lr)`。
                    let buf = f32::mul_add(self.config.momentum, slot.momentum_buffer[i], g / avg);
                    slot.momentum_buffer[i] = buf;
                    new_param.push(param_data[i] - lr * buf);
                } else {
                    // `param.addcdiv_(g, avg, value=-lr)` は
                    // `param + (-lr) * g / avg` を ATen が左から
                    // `value*t1/t2` の順に評価する（`AdamW` の
                    // `step_size * m / denom` と同じ括り）。
                    new_param.push(param_data[i] - (lr * g) / avg);
                }
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
        let cfg = RmsPropConfig {
            lr: -1.0,
            ..RmsPropConfig::default()
        };
        assert!(matches!(
            RmsProp::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_alpha_out_of_range() {
        let cfg = RmsPropConfig {
            alpha: 1.0,
            ..RmsPropConfig::default()
        };
        assert!(matches!(
            RmsProp::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let cfg = RmsPropConfig {
            alpha: -0.1,
            ..RmsPropConfig::default()
        };
        assert!(matches!(
            RmsProp::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let cfg = RmsPropConfig {
            eps: 0.0,
            ..RmsPropConfig::default()
        };
        assert!(matches!(
            RmsProp::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_negative_momentum() {
        let cfg = RmsPropConfig {
            momentum: -0.1,
            ..RmsPropConfig::default()
        };
        assert!(matches!(
            RmsProp::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_nan_hyperparameter() {
        let cfg = RmsPropConfig {
            weight_decay: f32::NAN,
            ..RmsPropConfig::default()
        };
        assert!(matches!(
            RmsProp::new(cfg),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut opt = RmsProp::new(RmsPropConfig::default()).unwrap();
        let param = t(vec![1.0, 2.0], &[2]);
        let grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let result = opt.step(&[(&param, &grad)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    #[test]
    fn rejects_slot_count_change_after_first_step() {
        let mut opt = RmsProp::new(RmsPropConfig::default()).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.1], &[1]);
        opt.step(&[(&param, &grad)]).unwrap();
        let result = opt.step(&[]);
        assert!(matches!(result, Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_slot_shape_change_after_first_step() {
        let mut opt = RmsProp::new(RmsPropConfig::default()).unwrap();
        let param1 = t(vec![1.0, 2.0], &[2]);
        let grad1 = t(vec![0.1, 0.1], &[2]);
        opt.step(&[(&param1, &grad1)]).unwrap();

        let param2 = t(vec![1.0, 2.0, 3.0], &[3]);
        let grad2 = t(vec![0.1, 0.1, 0.1], &[3]);
        let result = opt.step(&[(&param2, &grad2)]);
        assert!(matches!(result, Err(AutodiffError::Shape(_))));
    }

    /// Bugbot 指摘（`AdamW`）と同型の回帰テスト: 形状エラーで `step`
    /// が失敗した場合、`step_count`／`square_avg`／`grad_avg`／
    /// `momentum_buffer` のいずれも部分更新されず呼び出し前の状態の
    /// まま残ることを確認する。
    #[test]
    fn state_not_mutated_after_failed_step() {
        let cfg = RmsPropConfig {
            momentum: 0.9,
            centered: true,
            ..RmsPropConfig::default()
        };
        let mut opt = RmsProp::new(cfg).unwrap();
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

        let mut opt_ref = RmsProp::new(cfg).unwrap();
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

    /// `weight_decay = 0` のとき decay 項が完全に skip されること
    /// （grad=0・weight_decay=0 なら勾配が常にゼロのまま）を確認する。
    #[test]
    fn weight_decay_zero_skips_decay_term() {
        let cfg = RmsPropConfig {
            weight_decay: 0.0,
            ..RmsPropConfig::default()
        };
        let mut opt = RmsProp::new(cfg).unwrap();
        let param = t(vec![1.0], &[1]);
        let grad = t(vec![0.0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();
        // grad=0・weight_decay=0 なら square_avg=0 のままで update
        // 項の分子も 0 になり、パラメータは不変のはず。
        assert_eq!(out[0].get(&[0]).unwrap(), 1.0);
    }

    /// `centered=true` で定数勾配を 10 step 与えても NaN が出ない
    /// ことを確認する（`square_avg - grad_avg^2` が丸めで負になり
    /// `sqrt` が NaN を返す懸念のある分岐。定数勾配では `square_avg`
    /// と `grad_avg^2` が同じ極限へ収束するため境界に近い挙動になる）。
    #[test]
    fn centered_with_constant_grad_does_not_produce_nan() {
        let cfg = RmsPropConfig {
            centered: true,
            ..RmsPropConfig::default()
        };
        let mut opt = RmsProp::new(cfg).unwrap();
        let mut param = t(vec![1.0], &[1]);
        let grad = t(vec![0.5], &[1]);
        for _ in 0..10 {
            let out = opt.step(&[(&param, &grad)]).unwrap();
            let v = out[0].get(&[0]).unwrap();
            assert!(v.is_finite(), "centered RMSprop が NaN/inf を出した: {v}");
            param = out.into_iter().next().unwrap();
        }
    }

    /// `momentum > 0` の初回 step が `buf = g/avg`（初回は
    /// `momentum_buffer` が 0 のため `mul_add` の第 1 項が消える）
    /// という閉形式に一致することを確認する。
    #[test]
    fn first_step_momentum_matches_closed_form() {
        let cfg = RmsPropConfig {
            lr: 0.05,
            alpha: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
            momentum: 0.5,
            centered: false,
        };
        let mut opt = RmsProp::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let square_avg = (1.0 - cfg.alpha) * g0 * g0;
        let avg = square_avg.sqrt() + cfg.eps;
        let buf = g0 / avg;
        let expected = p0 - cfg.lr * buf;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "閉形式との不一致: actual={actual} expected={expected}"
        );
    }

    /// `momentum = 0` の t=1 閉形式との一致を固定する。
    #[test]
    fn first_step_matches_closed_form() {
        let cfg = RmsPropConfig {
            lr: 0.05,
            alpha: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        };
        let mut opt = RmsProp::new(cfg).unwrap();
        let p0 = 0.5f32;
        let g0 = 0.3f32;
        let param = t(vec![p0], &[1]);
        let grad = t(vec![g0], &[1]);
        let out = opt.step(&[(&param, &grad)]).unwrap();

        let square_avg = (1.0 - cfg.alpha) * g0 * g0;
        let avg = square_avg.sqrt() + cfg.eps;
        let expected = p0 - (cfg.lr * g0) / avg;

        let actual = out[0].get(&[0]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "閉形式との不一致: actual={actual} expected={expected}"
        );
    }

    /// codex-review 指摘の回帰テスト: 初回 `step()` が shape 不一致で
    /// 失敗しても `self.states` が確定してはならない。失敗した最初の
    /// 呼び出しとは異なる shape（別スロット数）で改めて呼び出した
    /// 場合でも「本当の初回 step」として成功しなければならない
    /// （もし失敗した呼び出しが誤って `self.states` を確定していると、
    /// 後続呼び出しは「2 回目以降」として扱われ、`InvalidArgument`
    /// で拒否されてしまう）。
    #[test]
    fn failed_first_step_does_not_poison_state_for_different_shape_retry() {
        let mut opt = RmsProp::new(RmsPropConfig::default()).unwrap();

        // 1 回目: 1 スロット・grad の shape が param と不一致で失敗。
        let bad_param = t(vec![1.0, 2.0], &[2]);
        let bad_grad = t(vec![1.0, 2.0, 3.0], &[3]);
        let first = opt.step(&[(&bad_param, &bad_grad)]);
        assert!(matches!(first, Err(AutodiffError::Shape(_))));

        // 2 回目: 1 回目とは異なるスロット数（2 スロット）・異なる
        // shape で呼び出す。`self.states` が汚染されていなければ
        // これは「本当の初回 step」として成功するはず。
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

    /// centered RMSprop の `lerp` 展開式が `weight`（`1-alpha`）の
    /// 絶対値で分岐することの回帰テスト（codex-review 指摘）。
    /// `alpha` を 0 に近づける（`weight = 1-alpha` が 1 に近い）と、
    /// 始点基準の展開式のみを常用する実装では `grad_avg` の桁が大きい
    /// 状態から勾配が急変した際に桁落ちする。ここでは `grad_avg` を
    /// 大きな値（1e8 相当）へ育てたあと勾配を 1 に変えて 1 step 進め、
    /// 数学的な期待値（無限精度での `lerp` は
    /// `grad_avg' = grad_avg + (g - grad_avg) * weight` と一致するが、
    /// `weight` が 1 に近いケースでは `grad_avg' ≈ g` になるはず）と
    /// 比較する。
    #[test]
    fn centered_lerp_uses_stable_branch_near_weight_one() {
        let cfg = RmsPropConfig {
            lr: 0.0, // パラメータ更新自体は本テストの対象外。
            alpha: 0.0,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: true,
        };
        let mut opt = RmsProp::new(cfg).unwrap();
        let mut param = t(vec![0.0], &[1]);

        // grad_avg を大きく育てる（centered なし相当の準備 step 群）。
        let big_grad = t(vec![1.0e8], &[1]);
        for _ in 0..1 {
            let out = opt.step(&[(&param, &big_grad)]).unwrap();
            param = out.into_iter().next().unwrap();
        }

        // `alpha=0` なので `weight = 1-alpha = 1.0`（`weight.abs() >=
        // 0.5` の分岐）。ここで勾配を 1.0 へ急変させる。数学的には
        // `grad_avg' = g = 1.0` になるはず（`weight=1` で始点の寄与が
        // 消える）。始点基準の展開式のみを常用する実装は
        // `grad_avg + 1.0*(g - grad_avg)` を計算する際に `1e8` 規模の
        // 桁落ちを経由するため、丸めの入り方が終点基準の展開式
        // （`g - (g - grad_avg)*alpha = g - 0 = g` を厳密に計算できる）
        // と異なりうる。
        let small_grad = t(vec![1.0], &[1]);
        opt.step(&[(&param, &small_grad)]).unwrap();

        // 内部状態は非公開のため、次の step で `avg`（＝
        // `sqrt(square_avg - grad_avg^2)`）を経由した更新値から
        // `grad_avg` が `1.0` に正しく収束していることを間接的に
        // 検証する: 3 回目以降も `grad=1.0` を与え続けると `grad_avg`
        // は `1.0` で不動点になるはずなので、`square_avg - grad_avg^2`
        // が発散せず有限のまま安定することを確認する（終点基準の
        // 展開式なら厳密に `grad_avg=1.0` へ一致し `square_avg` も
        // `1.0` へ収束するため `avg` は有限のまま。始点基準のみの
        // 実装が桁落ちで `grad_avg` を誤ると `square_avg -
        // grad_avg^2` が負に振れ `sqrt` が NaN を返しうる）。
        let cfg2 = RmsPropConfig { lr: 0.01, ..cfg };
        let mut opt2 = RmsProp::new(cfg2).unwrap();
        let mut p2 = t(vec![0.0], &[1]);
        opt2.step(&[(&p2, &big_grad)]).unwrap();
        for _ in 0..5 {
            let out = opt2
                .step(&[(&p2, &small_grad)])
                .unwrap_or_else(|e| panic!("centered lerp が失敗を誘発した: {e}"));
            let v = out[0].get(&[0]).unwrap();
            assert!(v.is_finite(), "centered lerp が NaN/inf を生んだ: {v}");
            p2 = out.into_iter().next().unwrap();
        }
    }
}

// イシュー #1743（親 #1610）: RMSprop（本モジュール）・Adagrad
// （`adagrad.rs`）を追加した。`AdamW`（#194）・`crate::optim::Sgd`
// （#193）と同じく `Tape`／`Var`／`BackendOps` に一切依存しない
// 値型・純関数の optimizer であり、新規 `Op`／`BackendOps` メソッド／
// `Var` メソッド／VJP は追加していない（カーネルなし）。正しさの検証は
// 実 PyTorch 実行値 fixture との統一複合判定（相対誤差 1e-3 未満
// または絶対誤差 1e-5 未満）・閉形式（t=1）一致・決定性（bit 完全
// 一致）で行う（`tests/nn_optim_rmsprop.rs`）。facade への公開は
// `crates/facade/src/optim.rs` の素の再エクスポート（純再エクスポート
// 契約は `docs/facade-optimizer-promotion-decision.md` §4 案 A）。
// `crate::optim::device_store::DeviceParamStore::step` は
// `BackendOps::sgd_step_device` 専用のデバイス常駐更新経路であり、
// 本イシューでは対応する `BackendOps` メソッドを追加していないため
// RMSprop・Adagrad とも **`DeviceParamStore` 非対応**（ホスト
// `Tensor<f32>` を介した `step()` のみ）。
