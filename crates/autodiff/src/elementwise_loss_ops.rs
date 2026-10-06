//! pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL の
//! 自由関数（イシュー #2652・親 #2651「PyTorch／TF 置き換えの API 網羅」）。
//!
//! | 関数 | PyTorch 相当 |
//! |---|---|
//! | [`bce_with_logits_loss_with`] | `F.binary_cross_entropy_with_logits(pos_weight=)` |
//! | [`hinge_embedding_loss`] | `F.hinge_embedding_loss` |
//! | [`soft_margin_loss`] | `F.soft_margin_loss` |
//! | [`gaussian_nll_loss`] | `F.gaussian_nll_loss` |
//!
//! **facade 非公開（意図的）**: `crate::loss_ops` と同じ判断枠組み。`Var` は facade から
//! 再エクスポートされるため `Var` への inherent メソッド追加は即座に公開面へ出る。公開形の
//! 推奨案（`Var` の委譲メソッド 4 本）は未承認（承認依頼 #2677・公開 #2678）のため、
//! 自由関数として `Var` の外に置き到達不能にする。facade 側の保留ガード
//! （`crates/facade/src/lib.rs::ElementwiseLossOpsHoldDoctestGuard`）が再エクスポート・
//! 同名メソッドを拒否する。
//!
//! **方式**: 新規 `Op` 4 種（`Op::BceWithLogitsPosWeightLoss` 等）とホスト参照実装
//! （`crate::eval::elementwise_loss`）。`BackendOps` は拡張せず常に `push_eager` する
//! （`Op::PoissonNllLoss` と同型。CUDA／Metal の tape からも同じホスト経路で到達する）。
//! [`bce_with_logits_loss_with`] は `pos_weight` 未指定のとき既存
//! `Var::bce_with_logits_loss` へ丸ごと委譲し、出荷済み経路と bit 一致する
//! （`Op::BceLoss`・`backend-cpu/src/bce.rs` は変更しない）。
//!
//! **数値契約**: 要素を先に `f64` へ昇格し index 順に蓄積して最後に 1 回だけ `f32` へ
//! downcast する。`Mean` は `numel` で除算、`numel == 0` は損失 `0.0`・勾配は空
//! （既存 `mse_loss` 規約。PyTorch の `Mean` は `NaN`）。
//!
//! **PyTorch との差分**（正は `docs/autodiff-elementwise-loss-ops-decision.md` §5）:
//! - `y`（Hinge／SoftMargin）は厳密に ±1 のみ許容（PyTorch は任意値）。
//! - `pos_weight` は有限かつ非負のみ（PyTorch は負値も受ける）。`input` shape へ右寄せ
//!   broadcast でき結果が `input` shape と一致する形に限る（入力を拡大しない）。
//! - GaussianNLL の `var` は `input` と同 shape 限定（PyTorch の `[..., 1]` 形・スカラーは
//!   非対応。呼び出し側が `Var::reshape`／`Var::broadcast_to` で揃える）。`var` の負値と
//!   `NaN` は `InvalidArgument`（PyTorch は負値のみ拒否）。
//! - SoftMargin は安定形 softplus で評価するため、PyTorch が `inf`（勾配 `NaN`）になる
//!   大振幅でも有限値を返す。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 検査順序は
//! ①`check_same_tape` → ②shape（`require_same_shape`・pos_weight の broadcast 可否）→
//! ③確保前バイト数上限（`checked_bytes_for::<f32>`）→ ④スカラー引数・非追跡テンソルの値検査
//! → ⑤実体化 → ⑥実体化後の値検査（`var >= 0`）→ ⑦forward → ⑧`push_eager`。エラー時に
//! tape へ孤児ノードを残さない。

use fandhe_ai_tensor_core::{Tensor, require_same_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::loss_ops::{check_pm_one_labels, materialize_pair, materialize_triple};
use crate::tape::Op;
use crate::var::{Reduction, Var};

/// [`bce_with_logits_loss_with`] のオプション（現状は `pos_weight` のみ）。
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct BceWithLogitsOptions {
    pos_weight: Option<Tensor<f32>>,
}

impl BceWithLogitsOptions {
    /// 正例の重み（PyTorch `pos_weight`）。`input` shape へ右寄せ broadcast できる形
    /// （`[C]`・`[1]`・`[]`・`[1, C]`・完全一致）で、全要素が有限かつ非負でなければならない。
    pub fn pos_weight(mut self, pos_weight: Tensor<f32>) -> Self {
        self.pos_weight = Some(pos_weight);
        self
    }

    pub(crate) fn pos_weight_value(&self) -> Option<&Tensor<f32>> {
        self.pos_weight.as_ref()
    }
}

/// [`gaussian_nll_loss`] のオプション。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GaussianNllOptions {
    full: bool,
    eps: f32,
}

impl Default for GaussianNllOptions {
    /// PyTorch 既定: `full = false`・`eps = 1e-6`。
    fn default() -> Self {
        Self {
            full: false,
            eps: 1e-6,
        }
    }
}

impl GaussianNllOptions {
    /// `true` のとき定数項 `0.5·ln(2π)` を要素ごとに加える。
    pub fn full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// `var` のクランプ下限（有限かつ非負）。
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub(crate) fn full_value(&self) -> bool {
        self.full
    }

    pub(crate) fn eps_value(&self) -> f32 {
        self.eps
    }
}

/// pos_weight 付き BCEWithLogits（`input` は logits、`target` とともに追跡対象。
/// イシュー #2652）。
///
/// `l = (1 − y)·x + (1 + (p − 1)·y)·softplus(−x)`（`p` は broadcast 済み `pos_weight`）。
/// 勾配は `dx = s·[(1 − y) − lw·σ(−x)]`・`dy = s·[−x + (p − 1)·softplus(−x)]`
/// （`pos_weight` は非追跡）。`options` が既定（`pos_weight` なし）のときは
/// `Var::bce_with_logits_loss` へ丸ごと委譲する。
pub fn bce_with_logits_loss_with<'t>(
    input: &Var<'t>,
    target: &Var<'t>,
    reduction: Reduction,
    options: &BceWithLogitsOptions,
) -> Result<Var<'t>, AutodiffError> {
    let Some(pos_weight) = options.pos_weight_value() else {
        return input.bce_with_logits_loss(target, reduction);
    };
    input.check_same_tape(target)?;
    let input_shape = input.shape();
    require_same_shape(&input_shape, &target.shape())?;
    checked_bytes_for::<f32>(&input_shape)?;
    // `broadcast_to` は右寄せで各軸が一致か 1 の形のみ許し、結果は常に `input_shape`
    // （入力を拡大しない）。確保量は `input` の numel 以下。
    let expanded = pos_weight.broadcast_to(&input_shape)?;
    for w in eval::dense_vec(&expanded) {
        if !w.is_finite() || w < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "bce_with_logits_loss_with: pos_weight は有限かつ非負でなければならない（got {w}）"
            )));
        }
    }
    let pos_weight = eval::build_tensor(eval::dense_vec(&expanded), &input_shape);

    let (input_val, target_val) = materialize_pair(input, target)?;
    let value = eval::elementwise_loss::bce_with_logits_pos_weight_loss_forward(
        &input_val,
        &target_val,
        &pos_weight,
        reduction,
    );
    let id = input.tape().push_eager(
        Op::BceWithLogitsPosWeightLoss {
            input: input.node_id(),
            target: target.node_id(),
            pos_weight,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// HingeEmbedding 損失（`input` は追跡対象、`y` は非追跡で同 shape・厳密に ±1。
/// イシュー #2652）。
///
/// `y == 1` は `l = x`、`y == −1` は `l = max(0, margin − x)`（`NaN` は伝播）。
/// 勾配は `y == 1` で `dx = s`、`y == −1` は `margin − x > 0`（境界ちょうどは 0。
/// PyTorch 2.14.0 実測）のときのみ `dx = −s`。
pub fn hinge_embedding_loss<'t>(
    input: &Var<'t>,
    y: &Tensor<f32>,
    margin: f32,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    let input_shape = input.shape();
    require_same_shape(y.shape(), &input_shape)?;
    checked_bytes_for::<f32>(&input_shape)?;
    if !margin.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "hinge_embedding_loss: margin は有限でなければならない（got {margin}）"
        )));
    }
    check_pm_one_labels(y, "hinge_embedding_loss")?;

    let input_val = materialize_single(input)?;
    let value =
        eval::elementwise_loss::hinge_embedding_loss_forward(&input_val, y, margin, reduction);
    let id = input.tape().push_eager(
        Op::HingeEmbeddingLoss {
            input: input.node_id(),
            y: y.clone(),
            margin,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// SoftMargin 損失（`input` は追跡対象、`y` は非追跡で同 shape・厳密に ±1。
/// イシュー #2652）。
///
/// `l = ln(1 + exp(−y·x))`（安定形で評価）。勾配は `dx = s·(−y)·σ(−y·x)`。
pub fn soft_margin_loss<'t>(
    input: &Var<'t>,
    y: &Tensor<f32>,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    let input_shape = input.shape();
    require_same_shape(y.shape(), &input_shape)?;
    checked_bytes_for::<f32>(&input_shape)?;
    check_pm_one_labels(y, "soft_margin_loss")?;

    let input_val = materialize_single(input)?;
    let value = eval::elementwise_loss::soft_margin_loss_forward(&input_val, y, reduction);
    let id = input.tape().push_eager(
        Op::SoftMarginLoss {
            input: input.node_id(),
            y: y.clone(),
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// GaussianNLL 損失（`input`・`target`・`var` の 3 入力すべてが追跡対象・同 shape。
/// イシュー #2652）。
///
/// `c = max(var, eps)`・`l = 0.5·(ln c + (x − t)²/c)`（`full` のとき `0.5·ln(2π)` 加算）。
/// 勾配は `dx = s·d/c`・`dt = −dx`・`dvar = s·0.5·(1/c − d²/c²)`。`var < eps` の要素でも
/// `dvar` を 0 にしない（PyTorch は `no_grad` 下の clamp で勾配を遮らない）。
pub fn gaussian_nll_loss<'t>(
    input: &Var<'t>,
    target: &Var<'t>,
    var: &Var<'t>,
    options: &GaussianNllOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    input.check_same_tape(target)?;
    input.check_same_tape(var)?;
    let input_shape = input.shape();
    require_same_shape(&input_shape, &target.shape())?;
    require_same_shape(&input_shape, &var.shape())?;
    checked_bytes_for::<f32>(&input_shape)?;

    let eps = options.eps_value();
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "gaussian_nll_loss: eps は有限かつ非負でなければならない（got {eps}）"
        )));
    }

    let (input_val, target_val, var_val) = materialize_triple(input, target, var)?;
    // 負値と NaN は `!(v >= 0.0)` で拒否する（PyTorch は負値のみ拒否し NaN は通す差分）。
    for v in eval::dense_vec(&var_val) {
        if v.is_nan() || v < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "gaussian_nll_loss: var は非負でなければならない（NaN も不可。got {v}）"
            )));
        }
    }
    let value = eval::elementwise_loss::gaussian_nll_loss_forward(
        &input_val,
        &target_val,
        &var_val,
        eps,
        options.full_value(),
        reduction,
    );
    let id = input.tape().push_eager(
        Op::GaussianNllLoss {
            input: input.node_id(),
            target: target.node_id(),
            var: var.node_id(),
            options: options.clone(),
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// 単一入力を層 1 で実体化する（[`materialize_pair`] の 1 入力版）。
fn materialize_single<'t>(a: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = a.tape().nodes.borrow();
    let ops = a.tape().ops();
    Ok(crate::tape::materialize_fallible(&nodes, ops, a.node_id())?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致")
    }

    fn tape() -> Tape {
        Tape::new_with_ops(crate::test_support::test_ops())
    }

    fn scalar(v: &Var<'_>) -> f32 {
        v.to_tensor().host_slice()[0]
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn hinge_matches_hand_values_and_boundary_grad() {
        let tp = tape();
        let x = tp.var(&t(vec![0.5, -1.0, 2.0, 0.2, 1.0], &[5]));
        let y = t(vec![1.0, -1.0, -1.0, 1.0, -1.0], &[5]);
        // l = [0.5, 2.0, 0.0, 0.2, 0.0]（最後は境界 x == margin）
        let sum = hinge_embedding_loss(&x, &y, 1.0, Reduction::Sum).unwrap();
        assert!(close(scalar(&sum), 2.7, 1e-6));
        let g = tp.backward(&sum).unwrap();
        let dx = g.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![1.0, -1.0, 0.0, 1.0, 0.0]);
        let mean = hinge_embedding_loss(&x, &y, 1.0, Reduction::Mean).unwrap();
        assert!(close(scalar(&mean), 0.54, 1e-6));
    }

    #[test]
    fn soft_margin_gaussian_and_bce_match_hand_values() {
        let tp = tape();
        let x = tp.var(&t(vec![0.0, 0.0], &[2]));
        let y = t(vec![1.0, -1.0], &[2]);
        let sm = soft_margin_loss(&x, &y, Reduction::Sum).unwrap();
        assert!(close(scalar(&sm), 2.0 * std::f32::consts::LN_2, 1e-6));

        let xi = tp.var(&t(vec![1.0], &[1]));
        let ti = tp.var(&t(vec![0.0], &[1]));
        let vi = tp.var(&t(vec![0.5], &[1]));
        let d = GaussianNllOptions::default();
        let g = gaussian_nll_loss(&xi, &ti, &vi, &d, Reduction::Sum).unwrap();
        assert!(close(scalar(&g), 0.5 * (0.5f32.ln() + 2.0), 1e-6));
        let full = d.clone().full(true);
        let gf = gaussian_nll_loss(&xi, &ti, &vi, &full, Reduction::Sum).unwrap();
        assert!(close(scalar(&gf) - scalar(&g), 0.918_938_5, 1e-5));

        let bx = tp.var(&t(vec![0.0], &[1]));
        let by = tp.var(&t(vec![1.0], &[1]));
        let opts = BceWithLogitsOptions::default().pos_weight(t(vec![2.0], &[1]));
        let b = bce_with_logits_loss_with(&bx, &by, Reduction::Sum, &opts).unwrap();
        assert!(close(scalar(&b), 2.0 * std::f32::consts::LN_2, 1e-6));
    }

    /// 中心差分（f32 forward・h = 1e-2）と VJP の一致（全追跡入力）。
    #[test]
    fn vjp_matches_central_difference() {
        let xs = vec![0.3f32, -0.8, 1.4, -0.1];
        let ts = vec![0.1f32, 0.5, 0.9, 0.0];
        let vs = vec![0.6f32, 1.2, 0.8, 0.4];
        let pw = t(vec![0.5, 2.0], &[2]);
        let eval_losses = |x: &[f32], tt: &[f32], v: &[f32]| -> (f32, f32) {
            let tp = tape();
            let xv = tp.var(&t(x.to_vec(), &[2, 2]));
            let tv = tp.var(&t(tt.to_vec(), &[2, 2]));
            let vv = tp.var(&t(v.to_vec(), &[2, 2]));
            let o = BceWithLogitsOptions::default().pos_weight(pw.clone());
            let b = bce_with_logits_loss_with(&xv, &tv, Reduction::Sum, &o).unwrap();
            let d = GaussianNllOptions::default();
            let g = gaussian_nll_loss(&xv, &tv, &vv, &d, Reduction::Sum).unwrap();
            (scalar(&b), scalar(&g))
        };
        let tp = tape();
        let xv = tp.var(&t(xs.clone(), &[2, 2]));
        let tv = tp.var(&t(ts.clone(), &[2, 2]));
        let vv = tp.var(&t(vs.clone(), &[2, 2]));
        let o = BceWithLogitsOptions::default().pos_weight(pw.clone());
        let b = bce_with_logits_loss_with(&xv, &tv, Reduction::Sum, &o).unwrap();
        let d = GaussianNllOptions::default();
        let g = gaussian_nll_loss(&xv, &tv, &vv, &d, Reduction::Sum).unwrap();
        let gb = tp.backward(&b).unwrap();
        let gg = tp.backward(&g).unwrap();
        let h = 1e-2f32;
        for i in 0..4 {
            let bump = |base: &[f32], dlt: f32| {
                let mut v = base.to_vec();
                v[i] += dlt;
                v
            };
            let diff =
                |p: (f32, f32), m: (f32, f32)| ((p.0 - m.0) / (2.0 * h), (p.1 - m.1) / (2.0 * h));
            let fd_x = diff(
                eval_losses(&bump(&xs, h), &ts, &vs),
                eval_losses(&bump(&xs, -h), &ts, &vs),
            );
            let fd_t = diff(
                eval_losses(&xs, &bump(&ts, h), &vs),
                eval_losses(&xs, &bump(&ts, -h), &vs),
            );
            let fd_v = diff(
                eval_losses(&xs, &ts, &bump(&vs, h)),
                eval_losses(&xs, &ts, &bump(&vs, -h)),
            );
            let at =
                |gr: &crate::Gradients, v: &Var<'_>| gr.get(v).unwrap().unwrap().host_slice()[i];
            assert!(close(at(&gb, &xv), fd_x.0, 2e-2), "bce dx[{i}]");
            assert!(close(at(&gg, &xv), fd_x.1, 2e-2), "gnll dx[{i}]");
            assert!(close(at(&gb, &tv), fd_t.0, 2e-2), "bce dy[{i}]");
            assert!(close(at(&gg, &tv), fd_t.1, 2e-2), "gnll dt[{i}]");
            assert!(close(at(&gg, &vv), fd_v.1, 2e-2), "gnll dvar[{i}]");
        }
    }

    #[test]
    fn empty_tensors_give_zero_loss_and_empty_grad() {
        let tp = tape();
        let x = tp.var(&t(vec![], &[0]));
        let y = t(vec![], &[0]);
        for r in [Reduction::Mean, Reduction::Sum] {
            let h = hinge_embedding_loss(&x, &y, 1.0, r).unwrap();
            assert_eq!(scalar(&h), 0.0);
            let g = tp.backward(&h).unwrap();
            assert_eq!(g.get(&x).unwrap().unwrap().numel(), 0);
            let s = soft_margin_loss(&x, &y, r).unwrap();
            assert_eq!(scalar(&s), 0.0);
            let tv = tp.var(&t(vec![], &[0]));
            let vv = tp.var(&t(vec![], &[0]));
            let n = gaussian_nll_loss(&x, &tv, &vv, &GaussianNllOptions::default(), r).unwrap();
            assert_eq!(scalar(&n), 0.0);
            let gn = tp.backward(&n).unwrap();
            assert_eq!(gn.get(&vv).unwrap().unwrap().numel(), 0);
            let o = BceWithLogitsOptions::default().pos_weight(t(vec![1.0], &[1]));
            let b = bce_with_logits_loss_with(&x, &tv, r, &o).unwrap();
            assert_eq!(scalar(&b), 0.0);
            let gb = tp.backward(&b).unwrap();
            assert_eq!(gb.get(&tv).unwrap().unwrap().numel(), 0);
        }
    }

    #[test]
    fn errors_do_not_leave_orphan_nodes() {
        let tp = tape();
        let x = tp.var(&t(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]));
        let tgt = tp.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));
        let other = tape();
        let foreign = other.var(&t(vec![0.0; 4], &[2, 2]));
        let before = tp.len();
        let r = Reduction::Mean;
        let pw = |data: Vec<f32>, shape: &[usize]| {
            BceWithLogitsOptions::default().pos_weight(t(data, shape))
        };
        // BCE: broadcast 不可・負値・非有限・別 tape
        assert!(bce_with_logits_loss_with(&x, &tgt, r, &pw(vec![1.0; 3], &[3])).is_err());
        assert!(bce_with_logits_loss_with(&x, &tgt, r, &pw(vec![1.0; 8], &[2, 2, 2])).is_err());
        assert!(bce_with_logits_loss_with(&x, &tgt, r, &pw(vec![-1.0], &[1])).is_err());
        assert!(bce_with_logits_loss_with(&x, &tgt, r, &pw(vec![f32::NAN], &[1])).is_err());
        assert!(bce_with_logits_loss_with(&x, &tgt, r, &pw(vec![f32::INFINITY], &[1])).is_err());
        assert!(bce_with_logits_loss_with(&x, &foreign, r, &pw(vec![1.0], &[1])).is_err());
        // hinge／soft margin: y の値・shape、margin 非有限
        let bad_y = t(vec![1.0, 0.0, -1.0, 1.0], &[2, 2]);
        let ok_y = t(vec![1.0, -1.0, -1.0, 1.0], &[2, 2]);
        assert!(hinge_embedding_loss(&x, &bad_y, 1.0, r).is_err());
        assert!(hinge_embedding_loss(&x, &ok_y, f32::NAN, r).is_err());
        assert!(hinge_embedding_loss(&x, &t(vec![1.0; 2], &[2]), 1.0, r).is_err());
        assert!(soft_margin_loss(&x, &bad_y, r).is_err());
        assert!(soft_margin_loss(&x, &t(vec![1.0; 2], &[2]), r).is_err());
        // gaussian: eps 非有限・負、var 負・NaN、shape 不一致、別 tape
        let var_ok = tp.var(&t(vec![1.0; 4], &[2, 2]));
        let var_neg = tp.var(&t(vec![1.0, -0.1, 1.0, 1.0], &[2, 2]));
        let var_nan = tp.var(&t(vec![1.0, f32::NAN, 1.0, 1.0], &[2, 2]));
        let var_bad_shape = tp.var(&t(vec![1.0; 2], &[2]));
        let before_vars = tp.len();
        let d = GaussianNllOptions::default();
        assert!(gaussian_nll_loss(&x, &tgt, &var_neg, &d, r).is_err());
        assert!(gaussian_nll_loss(&x, &tgt, &var_nan, &d, r).is_err());
        assert!(gaussian_nll_loss(&x, &tgt, &var_bad_shape, &d, r).is_err());
        assert!(gaussian_nll_loss(&x, &foreign, &var_ok, &d, r).is_err());
        assert!(gaussian_nll_loss(&x, &tgt, &var_ok, &d.clone().eps(f32::NAN), r).is_err());
        assert!(gaussian_nll_loss(&x, &tgt, &var_ok, &d.clone().eps(-1.0), r).is_err());
        assert_eq!(
            tp.len(),
            before_vars,
            "エラー経路で tape にノードを残さない"
        );
        assert!(before_vars > before);
    }

    #[test]
    fn default_options_delegate_to_existing_bce() {
        let tp = tape();
        let x = tp.var(&t(vec![0.3, -1.2, 2.5, 0.0], &[2, 2]));
        let y = tp.var(&t(vec![1.0, 0.0, 0.7, 1.0], &[2, 2]));
        let a = x.bce_with_logits_loss(&y, Reduction::Mean).unwrap();
        let n1 = tp.len();
        let b =
            bce_with_logits_loss_with(&x, &y, Reduction::Mean, &BceWithLogitsOptions::default())
                .unwrap();
        assert_eq!(tp.len(), n1 + 1, "委譲経路は既存と同じく 1 ノード");
        assert_eq!(scalar(&a).to_bits(), scalar(&b).to_bits());
    }

    #[test]
    fn upstream_gradient_scale_is_applied() {
        let tp = tape();
        let x = tp.var(&t(vec![0.5, -1.0], &[2]));
        let y = t(vec![1.0, -1.0], &[2]);
        let l = soft_margin_loss(&x, &y, Reduction::Sum).unwrap();
        let three = tp.var_no_grad(&t(vec![3.0], &[]));
        let scaled = l.mul(&three).unwrap();
        let g1 = tp.backward(&l).unwrap();
        let g3 = tp.backward(&scaled).unwrap();
        let a = g1.get(&x).unwrap().unwrap().host_slice().into_owned();
        let b = g3.get(&x).unwrap().unwrap().host_slice().into_owned();
        for (u, v) in a.iter().zip(&b) {
            assert!(close(*v, 3.0 * u, 1e-6));
        }
    }

    #[test]
    fn gaussian_var_below_eps_still_passes_dvar() {
        let tp = tape();
        let x = tp.var(&t(vec![1.0], &[1]));
        let tv = tp.var(&t(vec![0.0], &[1]));
        let vv = tp.var(&t(vec![0.0], &[1]));
        let d = GaussianNllOptions::default();
        let l = gaussian_nll_loss(&x, &tv, &vv, &d, Reduction::Sum).unwrap();
        let g = tp.backward(&l).unwrap();
        let dv = g.get(&vv).unwrap().unwrap().host_slice()[0];
        assert!(dv != 0.0, "var < eps でも dvar を遮らない（got {dv}）");
    }
}
