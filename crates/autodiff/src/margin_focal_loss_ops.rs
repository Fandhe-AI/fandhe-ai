//! MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss の
//! 自由関数（イシュー #2653・親 #2651「PyTorch／TF 置き換えの API 網羅」）。
//!
//! | 関数 | PyTorch 相当 |
//! |---|---|
//! | [`multi_margin_loss`] | `F.multi_margin_loss` |
//! | [`multilabel_margin_loss`] | `F.multilabel_margin_loss` |
//! | [`multilabel_soft_margin_loss`] | `F.multilabel_soft_margin_loss` |
//! | [`sigmoid_focal_loss`] | `torchvision.ops.sigmoid_focal_loss` の式（`torch.nn.functional` に focal loss は無い） |
//!
//! **facade 非公開（意図的）**: `crate::loss_ops`・`crate::elementwise_loss_ops` と同じ判断
//! 枠組み。`Var` は facade から再エクスポートされるため `Var` への inherent メソッド追加は
//! 即座に公開面へ出る。公開形の推奨案（`Var` の委譲メソッド 4 本）は未承認（承認依頼 #2677・
//! 公開 #2678）のため、自由関数として `Var` の外に置き到達不能にする。facade 側の保留ガード
//! （`crates/facade/src/lib.rs::MarginFocalLossOpsHoldDoctestGuard`）が再エクスポート・
//! 同名メソッドを拒否する。
//!
//! **方式**: 新規 `Op` 4 種（`Op::MultiMarginLoss` 等）とホスト参照実装
//! （`crate::eval::margin_focal_loss`）。`BackendOps` は拡張せず常に `push_eager` する
//! （`Op::PoissonNllLoss` と同型。CUDA／Metal の tape からも同じホスト経路で到達する）。
//! 追跡対象は 4 損失とも `input` のみ。`target`（添字・ラベル）と `weight` は非追跡データとして
//! `Op` へ埋め込む（既存 `bce_with_logits_loss` は target も `Var` である点が異なる）。
//!
//! **数値契約**: 要素を先に `f64` へ昇格し index 順に蓄積して最後に 1 回だけ `f32` へ
//! downcast する。`Mean` の分母は multi 系 3 種が行数 `N`（rank 1 入力は 1）、focal が
//! `numel`。multi 系は `Sum` でも行内で `/C` する（PyTorch と同じ）。`N == 0`
//! （focal は `numel == 0`）は損失 `0.0`・勾配は空（既存 `mse_loss` 規約。PyTorch の `Mean` は
//! `NaN`）。multi 系で `C == 0` かつ `N > 0` は `/C` が未定義のため `InvalidArgument`。
//!
//! **PyTorch との差分**（正は `docs/autodiff-margin-focal-loss-ops-decision.md` §5）:
//! - `reduction='none'` は非対応（`Reduction` は `Mean`／`Sum` のみ）。
//! - `multi_margin_loss` の `p` は `{1, 2}` のみ。`weight` は有限かつ非負のみ。
//! - `multilabel_soft_margin_loss` は rank 1・2 のみ。`target` は `[0, 1]` の有限値のみ。
//! - `sigmoid_focal_loss` は sigmoid 版のみ（softmax 版・torchvision の「`alpha < 0` で無効」は
//!   `alpha = None` で表す）。飽和域（`γ < 1`）で PyTorch が f32 評価で NaN 勾配を出す点は
//!   本実装が桁落ちしない形で有限値を返す。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 検査順序は
//! ①rank・shape → ②確保前バイト数上限（`checked_bytes_for::<f32>`）→ ③スカラー引数
//! （`p`・`margin`・`alpha`・`gamma`）→ ④非追跡テンソルの値（weight・添字範囲・ラベル範囲）→
//! ⑤実体化 → ⑥forward → ⑦`push_eager`。エラー時に tape へ孤児ノードを残さない。
//! `multilabel_margin_loss` は行あたり `O(C²)` 時間（target 数 × クラス数）だが確保量は
//! 入力 numel と長さ `C` のマスクのみ。
//!
//! **公開状況（イシュー #2678・#2677）**: `multilabel_margin_loss` の公開は #2678 では保留（`Reduction` の公開経路〈#2602〉が未整備のため）だったが、
//! #2602 のマージ後に #2677 で `Var` の 1 行委譲メソッドとして公開した。他 3 本とオプション型は保留のまま。
//! 上の「未承認」「保留」「承認依頼は #2677」の記述は #2677 時点のもので、承認形の公開は #2678 で行った
//! （ルート #2499 の承認コメント issuecomment-6033824965・`docs/compat-api-scope.md` §5.1）。

use fandhe_ai_tensor_core::{Tensor, require_same_shape};

use crate::bool_ops::checked_bytes_for;
use crate::elementwise_loss_ops::materialize_single;
use crate::error::AutodiffError;
use crate::eval;
use crate::tape::Op;
use crate::var::{Reduction, Var};

/// [`multi_margin_loss`] のオプション。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultiMarginOptions {
    p: u8,
    margin: f32,
    weight: Option<Tensor<f32>>,
}

impl Default for MultiMarginOptions {
    /// PyTorch 既定: `p = 1`・`margin = 1.0`・`weight = None`。
    fn default() -> Self {
        Self {
            p: 1,
            margin: 1.0,
            weight: None,
        }
    }
}

impl MultiMarginOptions {
    /// ヒンジの冪（`1` または `2`。それ以外は実行時に `InvalidArgument`）。
    pub fn p(mut self, p: u8) -> Self {
        self.p = p;
        self
    }

    /// マージン（有限でなければならない）。
    pub fn margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    /// クラス重み `[C]`（有限かつ非負。target クラスの重みが行の損失へ掛かる）。
    pub fn weight(mut self, weight: Tensor<f32>) -> Self {
        self.weight = Some(weight);
        self
    }

    pub(crate) fn p_value(&self) -> u8 {
        self.p
    }

    pub(crate) fn margin_value(&self) -> f32 {
        self.margin
    }

    pub(crate) fn weight_value(&self) -> Option<&Tensor<f32>> {
        self.weight.as_ref()
    }
}

/// [`multilabel_soft_margin_loss`] のオプション。
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct MultiLabelSoftMarginOptions {
    weight: Option<Tensor<f32>>,
}

impl MultiLabelSoftMarginOptions {
    /// クラス重み `[C]`（有限かつ非負）。
    pub fn weight(mut self, weight: Tensor<f32>) -> Self {
        self.weight = Some(weight);
        self
    }

    pub(crate) fn weight_value(&self) -> Option<&Tensor<f32>> {
        self.weight.as_ref()
    }
}

/// [`sigmoid_focal_loss`] のオプション。
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SigmoidFocalLossOptions {
    alpha: Option<f32>,
    gamma: f32,
}

impl Default for SigmoidFocalLossOptions {
    /// torchvision 既定: `alpha = Some(0.25)`・`gamma = 2.0`。
    fn default() -> Self {
        Self {
            alpha: Some(0.25),
            gamma: 2.0,
        }
    }
}

impl SigmoidFocalLossOptions {
    /// クラス間重み `α`（有限かつ `[0, 1]`）。`None` で重み付けなし
    /// （torchvision の「負値で無効」に相当）。
    pub fn alpha(mut self, alpha: Option<f32>) -> Self {
        self.alpha = alpha;
        self
    }

    /// 集中パラメータ `γ`（有限かつ非負）。
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    pub(crate) fn alpha_value(&self) -> Option<f32> {
        self.alpha
    }

    pub(crate) fn gamma_value(&self) -> f32 {
        self.gamma
    }
}

/// rank 1／2 の入力から `(N, C)` を得る。`C == 0` かつ `N > 0` は `InvalidArgument`。
fn rows_cols_checked(shape: &[usize], fn_name: &str) -> Result<(usize, usize), AutodiffError> {
    if shape.is_empty() || shape.len() > 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{fn_name}: input は rank 1 または 2 でなければならない（got {shape:?}）"
        )));
    }
    let (n, c) = eval::margin_focal_loss::rows_cols(shape);
    if c == 0 && n > 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{fn_name}: クラス数 C == 0 は拒否する（`/C` が未定義。got {shape:?}）"
        )));
    }
    Ok((n, c))
}

/// クラス重み `[C]` を検査して `Vec<f32>` へ取り出す（shape 厳密一致・有限・非負）。
fn checked_class_weight(
    weight: &Tensor<f32>,
    c: usize,
    fn_name: &str,
) -> Result<Vec<f32>, AutodiffError> {
    if weight.shape() != [c] {
        return Err(AutodiffError::InvalidArgument(format!(
            "{fn_name}: weight の shape は [{c}] でなければならない（got {:?}）",
            weight.shape()
        )));
    }
    let values = eval::dense_vec(weight);
    for &w in &values {
        if !w.is_finite() || w < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{fn_name}: weight は有限かつ非負でなければならない（got {w}）"
            )));
        }
    }
    Ok(values)
}

/// `[0, 1]` の有限ラベルであることを検査する。
fn check_unit_labels(target: &Tensor<f32>, fn_name: &str) -> Result<(), AutodiffError> {
    for v in eval::dense_vec(target) {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(AutodiffError::InvalidArgument(format!(
                "{fn_name}: target は有限かつ [0, 1] でなければならない（got {v}）"
            )));
        }
    }
    Ok(())
}

/// MultiMargin 損失（`input` は `[C]` または `[N, C]` で追跡対象、`target` は非追跡の
/// クラス添字。イシュー #2653）。
///
/// `z_j = margin − x_y + x_j`（`j ≠ y`）のうち `z_j > 0` のものを `p` 乗して和を取り、
/// `w[y]·Σ/C` を行の損失とする（`Sum` でも `/C` する）。`Mean` の分母は行数 `N`。
/// `z == 0`・`NaN` は損失にも勾配にも寄与しない（PyTorch 2.14.0 実測）。`target` は
/// rank ≤ 1 かつ要素数 `N`、各要素は `0 <= t < C`。
pub fn multi_margin_loss<'t>(
    input: &Var<'t>,
    target: &Tensor<i32>,
    options: &MultiMarginOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    const NAME: &str = "multi_margin_loss";
    let input_shape = input.shape();
    let (n, c) = rows_cols_checked(&input_shape, NAME)?;
    if target.shape().len() > 1 || target.numel() != n {
        return Err(AutodiffError::InvalidArgument(format!(
            "{NAME}: target は rank 1 以下で要素数 N={n} でなければならない（got {:?}）",
            target.shape()
        )));
    }
    checked_bytes_for::<f32>(&input_shape)?;
    let p = options.p_value();
    if p != 1 && p != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{NAME}: p は 1 または 2 でなければならない（got {p}）"
        )));
    }
    let margin = options.margin_value();
    if !margin.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "{NAME}: margin は有限でなければならない（got {margin}）"
        )));
    }
    let weight = match options.weight_value() {
        Some(w) => Some(checked_class_weight(w, c, NAME)?),
        None => None,
    };
    let targets = eval::dense_vec_i32(target);
    for &t in &targets {
        if t < 0 || (t as usize) >= c {
            return Err(AutodiffError::InvalidArgument(format!(
                "{NAME}: target 添字は 0 <= t < C={c} でなければならない（got {t}）"
            )));
        }
    }

    let input_val = materialize_single(input)?;
    let value = eval::margin_focal_loss::multi_margin_loss_forward(
        &input_val,
        &targets,
        weight.as_deref(),
        p,
        margin,
        reduction,
    );
    let id = input.tape().push_eager(
        Op::MultiMarginLoss {
            input: input.node_id(),
            targets,
            weight,
            p,
            margin,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// MultiLabelMargin 損失（`input` は `[C]` または `[N, C]` で追跡対象、`target` は非追跡で
/// `input` と同 shape のクラス添字。イシュー #2653）。
///
/// 行ごとに先頭から最初の負値の手前までを target 列とし、
/// `L = Σ_{t∈列} Σ_{d∉target 集合} max(0, 1 − x_t + x_d) / C`。重複した target 添字は
/// 重複分だけ加算される。`Mean` の分母は行数 `N`。`target` の全要素（終端以降を含む）は
/// `-1 <= t < C` でなければならない（PyTorch も終端以降を範囲検査する）。
pub fn multilabel_margin_loss<'t>(
    input: &Var<'t>,
    target: &Tensor<i32>,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    const NAME: &str = "multilabel_margin_loss";
    let input_shape = input.shape();
    let (_, c) = rows_cols_checked(&input_shape, NAME)?;
    require_same_shape(target.shape(), &input_shape)?;
    checked_bytes_for::<f32>(&input_shape)?;
    let targets = eval::dense_vec_i32(target);
    for &t in &targets {
        if t < -1 || (t >= 0 && (t as usize) >= c) {
            return Err(AutodiffError::InvalidArgument(format!(
                "{NAME}: target 添字は -1 <= t < C={c} でなければならない（got {t}）"
            )));
        }
    }

    let input_val = materialize_single(input)?;
    let value =
        eval::margin_focal_loss::multilabel_margin_loss_forward(&input_val, &targets, reduction);
    let id = input.tape().push_eager(
        Op::MultiLabelMarginLoss {
            input: input.node_id(),
            targets,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// MultiLabelSoftMargin 損失（`input` は logits で追跡対象、`target` は非追跡で同 shape の
/// `[0, 1]` ラベル。`[C]` または `[N, C]`。イシュー #2653）。
///
/// `l = t·softplus(−x) + (1−t)·softplus(x)`、行の損失は `Σ_c w_c·l / C`。
/// `Mean` の分母は行数 `N`。勾配は `dx = s·w_c·(σ(x) − t)/C`。
pub fn multilabel_soft_margin_loss<'t>(
    input: &Var<'t>,
    target: &Tensor<f32>,
    options: &MultiLabelSoftMarginOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    const NAME: &str = "multilabel_soft_margin_loss";
    let input_shape = input.shape();
    let (_, c) = rows_cols_checked(&input_shape, NAME)?;
    require_same_shape(target.shape(), &input_shape)?;
    checked_bytes_for::<f32>(&input_shape)?;
    let weight = match options.weight_value() {
        Some(w) => Some(checked_class_weight(w, c, NAME)?),
        None => None,
    };
    check_unit_labels(target, NAME)?;

    let input_val = materialize_single(input)?;
    let value = eval::margin_focal_loss::multilabel_soft_margin_loss_forward(
        &input_val,
        target,
        weight.as_deref(),
        reduction,
    );
    let id = input.tape().push_eager(
        Op::MultiLabelSoftMarginLoss {
            input: input.node_id(),
            target: target.clone(),
            weight,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// sigmoid focal loss（`input` は logits で追跡対象、`target` は非追跡で同 shape の
/// `[0, 1]` ラベル。rank 任意。イシュー #2653）。
///
/// `ce = (1−t)·x + softplus(−x)`・`q = 1 − p_t = t·σ(−x) + (1−t)·σ(x)`・
/// `l = α_t·ce·q^γ`（`α_t = α·t + (1−α)(1−t)`。`alpha = None` は 1）。
/// `Mean` の分母は `numel`。勾配は
/// `dx = s·α_t·[(p − t)·q^γ + γ·ce·q^(γ−1)·(1 − 2t)·p·(1 − p)]`（`γ == 0` は第 2 項 0）。
pub fn sigmoid_focal_loss<'t>(
    input: &Var<'t>,
    target: &Tensor<f32>,
    options: &SigmoidFocalLossOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError> {
    const NAME: &str = "sigmoid_focal_loss";
    let input_shape = input.shape();
    require_same_shape(target.shape(), &input_shape)?;
    checked_bytes_for::<f32>(&input_shape)?;
    let alpha = options.alpha_value();
    if let Some(a) = alpha
        && (!a.is_finite() || !(0.0..=1.0).contains(&a))
    {
        return Err(AutodiffError::InvalidArgument(format!(
            "{NAME}: alpha は有限かつ [0, 1] でなければならない（got {a}）"
        )));
    }
    let gamma = options.gamma_value();
    if !gamma.is_finite() || gamma < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{NAME}: gamma は有限かつ非負でなければならない（got {gamma}）"
        )));
    }
    check_unit_labels(target, NAME)?;

    let input_val = materialize_single(input)?;
    let value = eval::margin_focal_loss::sigmoid_focal_loss_forward(
        &input_val, target, alpha, gamma, reduction,
    );
    let id = input.tape().push_eager(
        Op::SigmoidFocalLoss {
            input: input.node_id(),
            target: target.clone(),
            alpha,
            gamma,
            reduction,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致")
    }

    fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
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
    fn multi_margin_matches_hand_values() {
        let tp = tape();
        // 行 0: y=0, x=[2.0, 1.5, 0.0] → z = [_, 0.5, -1.0] → h=0.5 → L=0.5/3
        // 行 1: y=2, x=[0.0, 0.5, 1.0] → z = [0, 0.5] の j=0: 1-1+0=0 → 0, j=1: 0.5 → L=0.5/3
        let x = tp.var(&t(vec![2.0, 1.5, 0.0, 0.0, 0.5, 1.0], &[2, 3]));
        let y = ti(vec![0, 2], &[2]);
        let d = MultiMarginOptions::default();
        let sum = multi_margin_loss(&x, &y, &d, Reduction::Sum).unwrap();
        assert!(close(scalar(&sum), 1.0 / 3.0, 1e-6));
        let mean = multi_margin_loss(&x, &y, &d, Reduction::Mean).unwrap();
        assert!(close(scalar(&mean), 0.5 / 3.0, 1e-6));
        // p=2: z² → 0.25/3 ずつ
        let p2 = multi_margin_loss(&x, &y, &d.clone().p(2), Reduction::Sum).unwrap();
        assert!(close(scalar(&p2), 0.5 / 3.0, 1e-6));
        // rank 1（N = 1）
        let x1 = tp.var(&t(vec![2.0, 1.5, 0.0], &[3]));
        let r1 = multi_margin_loss(&x1, &ti(vec![0], &[]), &d, Reduction::Mean).unwrap();
        assert!(close(scalar(&r1), 0.5 / 3.0, 1e-6));
    }

    #[test]
    fn multilabel_margin_matches_hand_values_and_duplicates() {
        let tp = tape();
        // target 列 [3, 0]（-1 で終端）。x=[0.1,0.2,0.4,0.8]。非 target = {1, 2}
        // t=3: (1-0.8+0.2)=0.4, (1-0.8+0.4)=0.6 / t=0: (1-0.1+0.2)=1.1, (1-0.1+0.4)=1.3 → 3.4/4
        let x = tp.var(&t(vec![0.1, 0.2, 0.4, 0.8], &[4]));
        let y = ti(vec![3, 0, -1, 1], &[4]);
        let l = multilabel_margin_loss(&x, &y, Reduction::Sum).unwrap();
        assert!(close(scalar(&l), 3.4 / 4.0, 1e-6));
        let g = tp.backward(&l).unwrap();
        let dx = g.get(&x).unwrap().unwrap().host_slice().into_owned();
        // t=3 が 2 組、t=0 が 2 組 → dx3 = -2/4・dx0 = -2/4・dx1 = dx2 = +2/4
        for (a, b) in dx.iter().zip([-0.5f32, 0.5, 0.5, -0.5]) {
            assert!(close(*a, b, 1e-6), "{dx:?}");
        }
    }

    #[test]
    fn multilabel_soft_margin_and_focal_match_hand_values() {
        let tp = tape();
        let x = tp.var(&t(vec![0.0, 0.0], &[2]));
        let tg = t(vec![1.0, 0.0], &[2]);
        let l = multilabel_soft_margin_loss(
            &x,
            &tg,
            &MultiLabelSoftMarginOptions::default(),
            Reduction::Sum,
        )
        .unwrap();
        // 各要素 ln2、C=2 で割って和（N=1）→ ln2
        assert!(close(scalar(&l), std::f32::consts::LN_2, 1e-6));
        // focal: alpha なし・gamma=0 は BCE と一致
        let o = SigmoidFocalLossOptions::default().alpha(None).gamma(0.0);
        let f = sigmoid_focal_loss(&x, &tg, &o, Reduction::Sum).unwrap();
        assert!(close(scalar(&f), 2.0 * std::f32::consts::LN_2, 1e-6));
        // alpha=0.25, gamma=2, x=0: q=0.5 → l = α_t·ln2·0.25
        let d = SigmoidFocalLossOptions::default();
        let f2 = sigmoid_focal_loss(&x, &tg, &d, Reduction::Sum).unwrap();
        let ln2 = std::f32::consts::LN_2;
        assert!(close(scalar(&f2), (0.25 + 0.75) * ln2 * 0.25, 1e-6));
    }

    /// 中心差分（f32 forward・h = 1e-2）と VJP の一致。
    #[test]
    fn vjp_matches_central_difference() {
        let xs = vec![0.3f32, -0.8, 1.4, -0.1, 0.9, 0.2];
        let tgt_f = t(vec![0.1, 0.9, 1.0, 0.0, 0.5, 0.3], &[2, 3]);
        let tgt_c = ti(vec![2, 0], &[2]);
        let tgt_m = ti(vec![1, 2, -1, 0, -1, 0], &[2, 3]);
        let w = t(vec![0.5, 2.0, 1.2], &[3]);
        let eval_all = |x: &[f32]| -> [f32; 5] {
            let tp = tape();
            let xv = tp.var(&t(x.to_vec(), &[2, 3]));
            let o = MultiMarginOptions::default().weight(w.clone());
            [
                scalar(&multi_margin_loss(&xv, &tgt_c, &o, Reduction::Sum).unwrap()),
                scalar(&multi_margin_loss(&xv, &tgt_c, &o.clone().p(2), Reduction::Mean).unwrap()),
                scalar(&multilabel_margin_loss(&xv, &tgt_m, Reduction::Sum).unwrap()),
                scalar(
                    &multilabel_soft_margin_loss(
                        &xv,
                        &tgt_f,
                        &MultiLabelSoftMarginOptions::default().weight(w.clone()),
                        Reduction::Sum,
                    )
                    .unwrap(),
                ),
                scalar(
                    &sigmoid_focal_loss(
                        &xv,
                        &tgt_f,
                        &SigmoidFocalLossOptions::default().gamma(1.5),
                        Reduction::Mean,
                    )
                    .unwrap(),
                ),
            ]
        };
        let tp = tape();
        let xv = tp.var(&t(xs.clone(), &[2, 3]));
        let o = MultiMarginOptions::default().weight(w.clone());
        let losses = [
            multi_margin_loss(&xv, &tgt_c, &o, Reduction::Sum).unwrap(),
            multi_margin_loss(&xv, &tgt_c, &o.clone().p(2), Reduction::Mean).unwrap(),
            multilabel_margin_loss(&xv, &tgt_m, Reduction::Sum).unwrap(),
            multilabel_soft_margin_loss(
                &xv,
                &tgt_f,
                &MultiLabelSoftMarginOptions::default().weight(w.clone()),
                Reduction::Sum,
            )
            .unwrap(),
            sigmoid_focal_loss(
                &xv,
                &tgt_f,
                &SigmoidFocalLossOptions::default().gamma(1.5),
                Reduction::Mean,
            )
            .unwrap(),
        ];
        let grads: Vec<_> = losses.iter().map(|l| tp.backward(l).unwrap()).collect();
        let h = 1e-2f32;
        for i in 0..6 {
            let mut plus = xs.clone();
            plus[i] += h;
            let mut minus = xs.clone();
            minus[i] -= h;
            let (fp, fm) = (eval_all(&plus), eval_all(&minus));
            for k in 0..5 {
                let fd = (fp[k] - fm[k]) / (2.0 * h);
                let an = grads[k].get(&xv).unwrap().unwrap().host_slice()[i];
                assert!(close(an, fd, 3e-2), "loss {k} dx[{i}]: an={an} fd={fd}");
            }
        }
    }

    #[test]
    fn empty_tensors_give_zero_loss_and_empty_grad() {
        let tp = tape();
        let x = tp.var(&t(vec![], &[0, 3]));
        for r in [Reduction::Mean, Reduction::Sum] {
            let a = multi_margin_loss(&x, &ti(vec![], &[0]), &MultiMarginOptions::default(), r)
                .unwrap();
            assert_eq!(scalar(&a), 0.0);
            assert_eq!(
                tp.backward(&a).unwrap().get(&x).unwrap().unwrap().numel(),
                0
            );
            let b = multilabel_margin_loss(&x, &ti(vec![], &[0, 3]), r).unwrap();
            assert_eq!(scalar(&b), 0.0);
            let c = multilabel_soft_margin_loss(
                &x,
                &t(vec![], &[0, 3]),
                &MultiLabelSoftMarginOptions::default(),
                r,
            )
            .unwrap();
            assert_eq!(scalar(&c), 0.0);
            let d = sigmoid_focal_loss(
                &x,
                &t(vec![], &[0, 3]),
                &SigmoidFocalLossOptions::default(),
                r,
            )
            .unwrap();
            assert_eq!(scalar(&d), 0.0);
            assert_eq!(
                tp.backward(&d).unwrap().get(&x).unwrap().unwrap().numel(),
                0
            );
        }
    }

    /// 空バッチ `[0, C]` は `C` が巨大でも長さ `C` の補助バッファを確保せず 0 損失になる
    /// （`checked_bytes_for` は numel = 0 で素通りするため。Bugbot 指摘）。
    #[test]
    fn empty_batch_with_huge_c_does_not_allocate_row_buffers() {
        let tp = tape();
        let c = 1usize << 40;
        let x = tp.var(&t(vec![], &[0, c]));
        for r in [Reduction::Mean, Reduction::Sum] {
            let b = multilabel_margin_loss(&x, &ti(vec![], &[0, c]), r).unwrap();
            assert_eq!(scalar(&b), 0.0);
            assert_eq!(
                tp.backward(&b).unwrap().get(&x).unwrap().unwrap().numel(),
                0
            );
        }
    }

    #[test]
    fn errors_do_not_leave_orphan_nodes() {
        let tp = tape();
        let x = tp.var(&t(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]));
        let before = tp.len();
        let r = Reduction::Mean;
        let mm = MultiMarginOptions::default();
        // multi_margin: target 範囲外・負・shape・p・margin・weight
        assert!(multi_margin_loss(&x, &ti(vec![0, 3], &[2]), &mm, r).is_err());
        assert!(multi_margin_loss(&x, &ti(vec![0, -1], &[2]), &mm, r).is_err());
        assert!(multi_margin_loss(&x, &ti(vec![0], &[1]), &mm, r).is_err());
        assert!(multi_margin_loss(&x, &ti(vec![0, 1, 0, 1], &[2, 2]), &mm, r).is_err());
        assert!(multi_margin_loss(&x, &ti(vec![0, 1], &[2]), &mm.clone().p(3), r).is_err());
        assert!(
            multi_margin_loss(&x, &ti(vec![0, 1], &[2]), &mm.clone().margin(f32::NAN), r).is_err()
        );
        let bad_w = |data: Vec<f32>, shape: &[usize]| mm.clone().weight(t(data, shape));
        let y = ti(vec![0, 1], &[2]);
        assert!(multi_margin_loss(&x, &y, &bad_w(vec![1.0; 2], &[2]), r).is_err());
        assert!(multi_margin_loss(&x, &y, &bad_w(vec![1.0, -1.0, 1.0], &[3]), r).is_err());
        assert!(multi_margin_loss(&x, &y, &bad_w(vec![1.0, f32::NAN, 1.0], &[3]), r).is_err());
        // C == 0 かつ N > 0 の拒否
        let zero_c = tp.var(&t(vec![], &[2, 0]));
        let r3 = tp.var(&t(vec![0.0; 8], &[2, 2, 2]));
        let before_zero = tp.len();
        assert!(multi_margin_loss(&zero_c, &ti(vec![0, 0], &[2]), &mm, r).is_err());
        assert!(multilabel_margin_loss(&zero_c, &ti(vec![], &[2, 0]), r).is_err());
        // rank 3 の拒否
        assert!(
            multilabel_soft_margin_loss(
                &r3,
                &t(vec![0.0; 8], &[2, 2, 2]),
                &MultiLabelSoftMarginOptions::default(),
                r
            )
            .is_err()
        );
        // multilabel_margin: 範囲外（終端以降を含む）・shape 不一致
        assert!(multilabel_margin_loss(&x, &ti(vec![0, -1, 3, 0, 1, 2], &[2, 3]), r).is_err());
        assert!(multilabel_margin_loss(&x, &ti(vec![0, -2, 0, 0, 1, 2], &[2, 3]), r).is_err());
        assert!(multilabel_margin_loss(&x, &ti(vec![0, 1], &[2]), r).is_err());
        // multilabel_soft_margin: ラベル範囲・NaN・shape・weight
        let sm = MultiLabelSoftMarginOptions::default();
        assert!(multilabel_soft_margin_loss(&x, &t(vec![0.5; 6], &[3, 2]), &sm, r).is_err());
        assert!(multilabel_soft_margin_loss(&x, &t(vec![1.5; 6], &[2, 3]), &sm, r).is_err());
        assert!(multilabel_soft_margin_loss(&x, &t(vec![f32::NAN; 6], &[2, 3]), &sm, r).is_err());
        assert!(
            multilabel_soft_margin_loss(
                &x,
                &t(vec![0.5; 6], &[2, 3]),
                &sm.clone().weight(t(vec![1.0; 2], &[2])),
                r
            )
            .is_err()
        );
        // focal: alpha・gamma・ラベル・shape
        let fo = SigmoidFocalLossOptions::default();
        let lab = t(vec![0.5; 6], &[2, 3]);
        assert!(sigmoid_focal_loss(&x, &lab, &fo.clone().alpha(Some(1.5)), r).is_err());
        assert!(sigmoid_focal_loss(&x, &lab, &fo.clone().alpha(Some(f32::NAN)), r).is_err());
        assert!(sigmoid_focal_loss(&x, &lab, &fo.clone().gamma(-1.0), r).is_err());
        assert!(sigmoid_focal_loss(&x, &lab, &fo.clone().gamma(f32::INFINITY), r).is_err());
        assert!(sigmoid_focal_loss(&x, &t(vec![2.0; 6], &[2, 3]), &fo, r).is_err());
        assert!(sigmoid_focal_loss(&x, &t(vec![0.5; 3], &[3]), &fo, r).is_err());
        assert_eq!(
            tp.len(),
            before_zero,
            "エラー経路で tape にノードを残さない"
        );
        assert!(before_zero > before);
    }

    #[test]
    fn unit_weights_match_no_weight_and_upstream_scale_is_applied() {
        let tp = tape();
        let x = tp.var(&t(vec![0.5, -1.0, 0.3, 1.2, 0.1, -0.4], &[2, 3]));
        let y = ti(vec![1, 2], &[2]);
        let a = multi_margin_loss(&x, &y, &MultiMarginOptions::default(), Reduction::Sum).unwrap();
        let b = multi_margin_loss(
            &x,
            &y,
            &MultiMarginOptions::default().weight(t(vec![1.0; 3], &[3])),
            Reduction::Sum,
        )
        .unwrap();
        assert_eq!(scalar(&a).to_bits(), scalar(&b).to_bits());
        let three = tp.var_no_grad(&t(vec![3.0], &[]));
        let scaled = a.mul(&three).unwrap();
        let g1 = tp.backward(&a).unwrap();
        let g3 = tp.backward(&scaled).unwrap();
        let u = g1.get(&x).unwrap().unwrap().host_slice().into_owned();
        let v = g3.get(&x).unwrap().unwrap().host_slice().into_owned();
        for (p, q) in u.iter().zip(&v) {
            assert!(close(*q, 3.0 * p, 1e-6));
        }
    }
}
