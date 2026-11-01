//! f64 テープの `create_graph`（二階微分の土台）第 1 段（イシュー #3785・
//! 親 #3719。設計は `docs/autodiff-functional-transforms-design.md` §35、
//! 実装記録は同 §37）。
//!
//! # 役割と呼び出し文脈
//!
//! f32 の [`crate::tape::Tape::backward_create_graph`]（`create_graph.rs`）と
//! 同じ「子テープ方式」を f64 の独立グラフ（[`super::TapeF64`]／
//! [`super::VarF64`]）へ移したもの。親テープ上の loss の祖先を子テープへ
//! 写し（full mirror）、その写しの上に**逆伝播グラフを `VarF64` 演算として**
//! 積む。結果の勾配 [`CreateGraphResultF64::grad`] は子テープ上の追跡対象
//! 値であり、子テープの `backward` でさらに微分できる（double-VJP 法。
//! 追跡表の項目 2〔`TapeF64`／`VarF64` の `jvp` 系〕の前提）。
//!
//! - 呼び出し元: 第 1 段では autodiff 内部のテストのみ。facade への公開
//!   （newtype `CreateGraphResultF64`）は #3786、`jvp` 系は #3787・#3788。
//! - 呼び出し先: 親側は無変更の [`super::TapeF64::backward`]（`first_order`）、
//!   子側は `VarF64` の `add`／`mul`／`div`／`pow` と crate 内部の
//!   `cg_*` forward。
//!
//! # 第 1 段の範囲（§35.11）
//!
//! 対応 Op は `add`／`mul`／`div`／`pow` と、勾配グラフの閉包性のための内部
//! variant（`BroadcastTo`／`ReduceTo`／`Log`／`Where`）。`matmul`／`sum`／
//! `mean` は第 2 段（#3786）、`max` は fail-closed（`Backward` を返す）。
//! 追跡対象（`requires_grad`）の祖先に対象外 Op があれば、子テープへ何も
//! 書く前に拒否する。`requires_grad == false` の部分木は下へたどらない
//! ため、定数側に対象外 Op があっても通す（f32 と同じ規則）。
//!
//! # 数値契約
//!
//! - `first_order()` は素の `backward` の結果そのもの（構造上 bit 同一）。
//! - `grad()`（子テープ上の勾配）は 1 階勾配と数学的に同じだが、式の組み
//!   立てが `binary_partials_f64` と異なる（例: `Div` は `1/b` を先に作る）
//!   ため bit 一致は契約にしない。判定は REQ-2 の統一複合判定
//!   （相対 1e-3 未満 または 絶対 1e-5 未満）をそのまま使う。
//! - `Pow` は選ばれない枝の `0 * inf = NaN` を避けるため `Where` で安全な
//!   値へ置き換える。da 側は「`b == 0` かつ底が有限の正の値でない」位置
//!   だけ、db 側は `a == 0` の位置だけを置き換える。底が正の位置は式の
//!   依存関係を残し、混合二階微分（`1/a`）を定数マスクで消さない。値は
//!   1 階の `binary_partials_f64` と REQ-2 の範囲で一致する（違いうるのは
//!   零の符号だけ）。
//!
//! # メモリ
//!
//! full mirror は loss の祖先の値を子テープへ複製する（`Tensor` の内部は
//! `Arc` 共有だが、子テープのノード数は祖先数に比例して増える）。

use fandhe_ai_tensor_core::Tensor;

use super::{
    BinOpF64, GradientsF64, NodeF64, NodeIdF64, OpF64, TapeF64, TapeIdF64, VarF64,
    broadcast_value_f64,
};
use crate::error::AutodiffError;

/// 子テープ上の `Option<VarF64>` 列（写し・勾配の保持用）。
type VarSlots<'c, 't> = Vec<Option<VarF64<'c, 't>>>;

/// [`TapeF64::backward_create_graph`] の結果（f32 の
/// [`crate::create_graph::CreateGraphResult`] の f64 版）。
///
/// - `first_order`: 親テープ上の素の逆伝播結果（無変更の `backward`）。
/// - `grad`: 子テープ上の勾配（追跡対象。さらに微分できる）。
/// - `child_var`: 親の `VarF64` に対応する子テープ上の写し。
///
/// 別テープの `VarF64` は `TapeMismatch`、`requires_grad == false` は
/// `GradientTrackingDisabled`、loss から届かないノードは `Ok(None)`。
/// `TapeF64` のノード列は追記専用で世代（epoch）を持たないため、
/// f32 版のような世代検査は要らない。
pub struct CreateGraphResultF64<'c, 't> {
    first_order: GradientsF64,
    parent_id: TapeIdF64,
    mirror: VarSlots<'c, 't>,
    cgrads: VarSlots<'c, 't>,
}

impl<'c, 't> CreateGraphResultF64<'c, 't> {
    /// 親テープ上の素の 1 階勾配（`TapeF64::backward` の結果と bit 同一）。
    pub fn first_order(&self) -> &GradientsF64 {
        &self.first_order
    }

    /// `var` の勾配を子テープ上の追跡対象値として返す。
    pub fn grad(&self, var: &VarF64<'_, 't>) -> Result<Option<VarF64<'c, 't>>, AutodiffError> {
        self.check(var)?;
        Ok(self.cgrads.get(var.id.0).copied().flatten())
    }

    /// `var` に対応する子テープ上の写しを返す。
    pub fn child_var(&self, var: &VarF64<'_, 't>) -> Result<Option<VarF64<'c, 't>>, AutodiffError> {
        self.check(var)?;
        Ok(self.mirror.get(var.id.0).copied().flatten())
    }

    fn check(&self, var: &VarF64<'_, 't>) -> Result<(), AutodiffError> {
        if var.graph.id != self.parent_id {
            return Err(AutodiffError::TapeMismatch);
        }
        if !var.graph.nodes.borrow()[var.id.0].requires_grad {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        Ok(())
    }
}

impl<'t> TapeF64<'t> {
    /// `loss` の逆伝播グラフを、空の子テープ `child` 上へ `VarF64` 演算として
    /// 記録する（二階以上の微分の土台。f32 の `Tape::backward_create_graph`
    /// の f64 版）。非スカラーの `loss` は全要素 1 のシードで暗黙に総和
    /// 射影する（`backward` と同じ）。
    ///
    /// **入口検査（順序固定）**:
    /// 1. `loss` が `self` に属さない → `TapeMismatch`
    /// 2. `child` が `self` と同一 → `Backward`（`RefCell` の二重借用も防ぐ）
    /// 3. `child` のデバイスが異なる → `DeviceMismatch`
    /// 4. `child` が空でない → `Backward`
    /// 5. 追跡対象の祖先に第 1 段の対象外 Op がある → `Backward`
    ///    （子テープへ書く前に拒否する）
    /// 6. 素の `backward` で `first_order` を得る（追跡なしの loss はここで
    ///    既存どおり拒否される）
    ///
    /// 構築の途中で失敗した場合、`child` は空へ戻る（`rollback_to`）。
    /// 数値契約とメモリは本モジュールの doc を参照。
    ///
    /// ```
    /// use fandhe_ai_autodiff::Tape;
    /// use fandhe_ai_autodiff::f64_autograd::TapeF64;
    /// use fandhe_ai_tensor_core::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let base = Tape::new();
    /// let parent = TapeF64::new(&base);
    /// let child = TapeF64::new(&base);
    /// let x = parent.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[3])?);
    /// // loss = x^3（非スカラーのまま扱う。シードは全要素 1）
    /// let loss = x.mul(&x)?.mul(&x)?;
    /// let cg = parent.backward_create_graph(&loss, &child)?;
    /// // 一階: 3x^2、二階: 6x
    /// let g = cg.grad(&x)?.ok_or("x へ勾配が届く")?;
    /// let cx = cg.child_var(&x)?.ok_or("x の写しがある")?;
    /// let h = child.backward(&g)?;
    /// let h = h.get(&cx)?.ok_or("二階勾配が届く")?;
    /// let got = h.host_slice();
    /// for (v, x) in got.iter().zip([1.0f64, 2.0, 3.0]) {
    ///     assert!((v - 6.0 * x).abs() < 1e-9);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn backward_create_graph<'c>(
        &self,
        loss: &VarF64<'_, 't>,
        child: &'c TapeF64<'t>,
    ) -> Result<CreateGraphResultF64<'c, 't>, AutodiffError> {
        if loss.graph.id != self.id {
            return Err(AutodiffError::TapeMismatch);
        }
        if child.id == self.id {
            return Err(AutodiffError::Backward(
                "f64 create_graph: 子テープに親と同一の TapeF64 は指定できない".into(),
            ));
        }
        if child.tape.device() != self.tape.device() {
            return Err(AutodiffError::DeviceMismatch {
                requested: child.tape.device(),
                actual: self.tape.device(),
            });
        }
        if !child.is_empty() {
            return Err(AutodiffError::Backward(
                "f64 create_graph: 子テープは空である必要がある（既存ノードを持つ \
                 TapeF64 は渡せない）"
                    .into(),
            ));
        }
        let ancestors = self.collect_ancestors_f64(loss.id);
        self.validate_ancestors_f64(&ancestors)?;
        let first_order = self.backward(loss)?;
        let (mirror, cgrads) = self.build_with_rollback(loss.id, &ancestors, child)?;
        Ok(CreateGraphResultF64 {
            first_order,
            parent_id: self.id,
            mirror,
            cgrads,
        })
    }

    /// `root` から `requires_grad` の経路で到達できる祖先の集合（添字で
    /// 引く真偽列）。`requires_grad == false` のノードは含めるが下へは
    /// たどらない。
    fn collect_ancestors_f64(&self, root: NodeIdF64) -> Vec<bool> {
        let nodes = self.nodes.borrow();
        let mut seen = vec![false; nodes.len()];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if seen[id.0] {
                continue;
            }
            seen[id.0] = true;
            if nodes[id.0].requires_grad {
                stack.extend(op_inputs_f64(&nodes[id.0].op));
            }
        }
        seen
    }

    /// 追跡対象の祖先に第 1 段の対象外 Op があれば `Backward`
    /// （fail-closed。定数側の部分木は検査しない）。
    fn validate_ancestors_f64(&self, ancestors: &[bool]) -> Result<(), AutodiffError> {
        let nodes = self.nodes.borrow();
        for (id, node) in nodes.iter().enumerate() {
            if ancestors[id] && node.requires_grad && !supports_create_graph_f64(&node.op) {
                return Err(AutodiffError::Backward(format!(
                    "f64 create_graph: NodeId {id} の演算は第 1 段の対象外（matmul・sum・\
                     mean は第 2 段、max は fail-closed。設計記録 §35.11）"
                )));
            }
        }
        Ok(())
    }

    /// 写しと勾配グラフを組む。失敗時は `child` を組む前の状態へ戻す。
    fn build_with_rollback<'c>(
        &self,
        loss_id: NodeIdF64,
        ancestors: &[bool],
        child: &'c TapeF64<'t>,
    ) -> Result<(VarSlots<'c, 't>, VarSlots<'c, 't>), AutodiffError> {
        let point = child.rollback_point();
        let result = self.build_mirror_f64(ancestors, child).and_then(|mirror| {
            self.build_cgrads_f64(loss_id, ancestors, &mirror, child)
                .map(|cgrads| (mirror, cgrads))
        });
        if result.is_err() {
            child.rollback_to(point);
        }
        result
    }

    /// 祖先を昇順に子テープへ写す。`requires_grad == false` は定数、
    /// 追跡対象の葉は追跡対象の葉、それ以外は演算を再生する。
    fn build_mirror_f64<'c>(
        &self,
        ancestors: &[bool],
        child: &'c TapeF64<'t>,
    ) -> Result<VarSlots<'c, 't>, AutodiffError> {
        let nodes = self.nodes.borrow();
        let mut mirror: VarSlots<'c, 't> = vec![None; nodes.len()];
        for (id, node) in nodes.iter().enumerate() {
            if !ancestors[id] {
                continue;
            }
            let var = if !node.requires_grad {
                child.var_no_grad(&node.value)
            } else if matches!(node.op, OpF64::Leaf) {
                child.var(&node.value)
            } else {
                replay_op_f64(node, &mirror)?
            };
            mirror[id] = Some(var);
        }
        // `child` の借用先は `self` ではないので、ここで親の借用を閉じる。
        drop(nodes);
        Ok(mirror)
    }

    /// 逆順に走査し、各 Op の VJP を子テープ上の `VarF64` 演算で積む。
    fn build_cgrads_f64<'c>(
        &self,
        loss_id: NodeIdF64,
        ancestors: &[bool],
        mirror: &[Option<VarF64<'c, 't>>],
        child: &'c TapeF64<'t>,
    ) -> Result<VarSlots<'c, 't>, AutodiffError> {
        let nodes = self.nodes.borrow();
        let n = nodes.len();
        let mut cgrads: VarSlots<'c, 't> = vec![None; n];
        let loss_shape = nodes[loss_id.0].shape.clone();
        let seed = Tensor::full(&loss_shape, 1.0f64).map_err(AutodiffError::Shape)?;
        cgrads[loss_id.0] = Some(child.var_no_grad(&seed));

        for id in (0..n).rev() {
            if !ancestors[id] || !nodes[id].requires_grad {
                continue;
            }
            let Some(g) = cgrads[id] else {
                continue;
            };
            let contributions = vjp_contributions_f64(&nodes, mirror, child, id, g)?;
            for (target, contribution) in contributions {
                if !nodes[target.0].requires_grad {
                    continue;
                }
                cgrads[target.0] = Some(match cgrads[target.0] {
                    None => contribution,
                    Some(existing) => existing.add(&contribution)?,
                });
            }
        }
        Ok(cgrads)
    }
}

/// `op` の入力ノード（`requires_grad` の伝播・祖先収集用）。
fn op_inputs_f64(op: &OpF64) -> Vec<NodeIdF64> {
    match op {
        OpF64::Leaf => vec![],
        OpF64::Add(a, b)
        | OpF64::Mul(a, b)
        | OpF64::Div(a, b)
        | OpF64::Pow(a, b)
        | OpF64::MatMul(a, b) => vec![*a, *b],
        OpF64::Sum { input, .. }
        | OpF64::Mean { input, .. }
        | OpF64::Max { input, .. }
        | OpF64::BroadcastTo { input }
        | OpF64::ReduceTo { input } => vec![*input],
        OpF64::Log(input) => vec![*input],
        OpF64::Where {
            cond,
            on_true,
            on_false,
        } => vec![*cond, *on_true, *on_false],
    }
}

/// 第 1 段で `create_graph` が扱える Op か。ワイルドカードを使わず、
/// Op の追加時に必ず判断を強制する（迂回経路を作らない。
/// `.claude/rules/security.md` A08）。
fn supports_create_graph_f64(op: &OpF64) -> bool {
    match op {
        OpF64::Leaf
        | OpF64::Add(..)
        | OpF64::Mul(..)
        | OpF64::Div(..)
        | OpF64::Pow(..)
        | OpF64::BroadcastTo { .. }
        | OpF64::ReduceTo { .. }
        | OpF64::Log(..)
        | OpF64::Where { .. } => true,
        OpF64::MatMul(..) | OpF64::Sum { .. } | OpF64::Mean { .. } | OpF64::Max { .. } => false,
    }
}

/// 写し済みの `id` を引く。未登録は祖先集合と写しの不整合（契約違反）。
fn mirrored<'c, 't>(
    mirror: &[Option<VarF64<'c, 't>>],
    id: NodeIdF64,
) -> Result<VarF64<'c, 't>, AutodiffError> {
    mirror.get(id.0).copied().flatten().ok_or_else(|| {
        AutodiffError::Backward(format!(
            "f64 create_graph: NodeId {} の写しが無い（祖先集合の不整合）",
            id.0
        ))
    })
}

/// 追跡対象の非葉ノード `node` の演算を子テープ上で再生する。
fn replay_op_f64<'c, 't>(
    node: &NodeF64,
    mirror: &[Option<VarF64<'c, 't>>],
) -> Result<VarF64<'c, 't>, AutodiffError> {
    match node.op {
        OpF64::Leaf => Err(AutodiffError::Backward(
            "f64 create_graph: 葉ノードの写しは build_mirror_f64 が直接作る（契約違反）".into(),
        )),
        OpF64::Add(a, b) => mirrored(mirror, a)?.add(&mirrored(mirror, b)?),
        OpF64::Mul(a, b) => mirrored(mirror, a)?.mul(&mirrored(mirror, b)?),
        OpF64::Div(a, b) => mirrored(mirror, a)?.div(&mirrored(mirror, b)?),
        OpF64::Pow(a, b) => mirrored(mirror, a)?.pow(&mirrored(mirror, b)?),
        OpF64::BroadcastTo { input } => mirrored(mirror, input)?.cg_broadcast_to(&node.shape),
        OpF64::ReduceTo { input } => mirrored(mirror, input)?.cg_reduce_to(&node.shape),
        OpF64::Log(input) => mirrored(mirror, input)?.cg_log(),
        OpF64::Where {
            cond,
            on_true,
            on_false,
        } => mirrored(mirror, cond)?
            .cg_where(&mirrored(mirror, on_true)?, &mirrored(mirror, on_false)?),
        OpF64::MatMul(..) | OpF64::Sum { .. } | OpF64::Mean { .. } | OpF64::Max { .. } => {
            Err(AutodiffError::Backward(
                "f64 create_graph: 第 1 段の対象外の演算（検査をすり抜けた契約違反）".into(),
            ))
        }
    }
}

/// 0-rank の定数（`var_no_grad`）。
fn scalar_const<'c, 't>(child: &'c TapeF64<'t>, v: f64) -> Result<VarF64<'c, 't>, AutodiffError> {
    let t = Tensor::new(vec![v], &[]).map_err(AutodiffError::Shape)?;
    Ok(child.var_no_grad(&t))
}

/// `shape` を全要素 `v` で埋めた定数。
fn full_const<'c, 't>(
    child: &'c TapeF64<'t>,
    shape: &[usize],
    v: f64,
) -> Result<VarF64<'c, 't>, AutodiffError> {
    super::checked_bytes_for_f64(shape).map_err(AutodiffError::Shape)?;
    let t = Tensor::full(shape, v).map_err(AutodiffError::Shape)?;
    Ok(child.var_no_grad(&t))
}

/// `value` を `shape` へブロードキャストし、`pred` を満たす位置を 1.0、
/// それ以外を 0.0 とした定数マスク。
fn mask_const<'c, 't>(
    child: &'c TapeF64<'t>,
    value: &Tensor<f64>,
    shape: &[usize],
    pred: impl Fn(f64) -> bool,
) -> Result<VarF64<'c, 't>, AutodiffError> {
    let bc = broadcast_value_f64(value, shape)?;
    let data: Vec<f64> = bc
        .host_slice()
        .iter()
        .map(|&v| if pred(v) { 1.0 } else { 0.0 })
        .collect();
    let t = Tensor::new(data, shape).map_err(AutodiffError::Shape)?;
    Ok(child.var_no_grad(&t))
}

/// `a`・`b` を `shape` へブロードキャストし、`pred(a, b)` を満たす位置を
/// 1.0、それ以外を 0.0 とした定数マスク（`Pow` の da 側の安全置換用）。
fn pair_mask_const<'c, 't>(
    child: &'c TapeF64<'t>,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    shape: &[usize],
    pred: impl Fn(f64, f64) -> bool,
) -> Result<VarF64<'c, 't>, AutodiffError> {
    super::checked_bytes_for_f64(shape).map_err(AutodiffError::Shape)?;
    let a_bc = broadcast_value_f64(a, shape)?;
    let b_bc = broadcast_value_f64(b, shape)?;
    let data: Vec<f64> = a_bc
        .host_slice()
        .iter()
        .zip(b_bc.host_slice().iter())
        .map(|(&x, &y)| if pred(x, y) { 1.0 } else { 0.0 })
        .collect();
    let t = Tensor::new(data, shape).map_err(AutodiffError::Shape)?;
    Ok(child.var_no_grad(&t))
}

/// ノード `id` の VJP を子テープ上の `VarF64` 演算で組み、
/// `(寄与先, 寄与勾配)` を返す。寄与勾配は寄与先の shape を持つ。
fn vjp_contributions_f64<'c, 't>(
    nodes: &[NodeF64],
    mirror: &[Option<VarF64<'c, 't>>],
    child: &'c TapeF64<'t>,
    id: usize,
    g: VarF64<'c, 't>,
) -> Result<Vec<(NodeIdF64, VarF64<'c, 't>)>, AutodiffError> {
    let node = &nodes[id];
    let ctx = VjpCtx {
        nodes,
        mirror,
        child,
    };
    match node.op {
        OpF64::Leaf => Ok(vec![]),
        OpF64::Add(a, b) => binary_contributions_f64(&ctx, BinOpF64::Add, id, (a, b), g),
        OpF64::Mul(a, b) => binary_contributions_f64(&ctx, BinOpF64::Mul, id, (a, b), g),
        OpF64::Div(a, b) => binary_contributions_f64(&ctx, BinOpF64::Div, id, (a, b), g),
        OpF64::Pow(a, b) => binary_contributions_f64(&ctx, BinOpF64::Pow, id, (a, b), g),
        OpF64::BroadcastTo { input } => {
            let da = g.cg_reduce_to(&nodes[input.0].shape)?;
            Ok(vec![(input, da)])
        }
        OpF64::ReduceTo { input } => {
            let da = g.cg_broadcast_to(&nodes[input.0].shape)?;
            Ok(vec![(input, da)])
        }
        OpF64::Log(input) => {
            let da = g.div(&mirrored(mirror, input)?)?;
            Ok(vec![(input, da)])
        }
        OpF64::Where {
            cond,
            on_true,
            on_false,
        } => {
            let cond_m = mirrored(mirror, cond)?;
            let zeros = full_const(child, &node.shape, 0.0)?;
            let mut out = Vec::with_capacity(2);
            if nodes[on_true.0].requires_grad {
                out.push((on_true, cond_m.cg_where(&g, &zeros)?));
            }
            if nodes[on_false.0].requires_grad {
                out.push((on_false, cond_m.cg_where(&zeros, &g)?));
            }
            Ok(out)
        }
        OpF64::MatMul(..) | OpF64::Sum { .. } | OpF64::Mean { .. } | OpF64::Max { .. } => {
            Err(AutodiffError::Backward(
                "f64 create_graph: 第 1 段の対象外の演算（検査をすり抜けた契約違反）".into(),
            ))
        }
    }
}

/// VJP 構築の共有文脈（親ノード列・写し・子テープ）。
#[derive(Clone, Copy)]
struct VjpCtx<'a, 'c, 't> {
    nodes: &'a [NodeF64],
    mirror: &'a [Option<VarF64<'c, 't>>],
    child: &'c TapeF64<'t>,
}

/// 二項演算 4 種の勾配グラフ。`g` は出力 shape、寄与は各入力の元 shape
/// へ `cg_reduce_to` で縮約する。追跡不要な側は組まない。
fn binary_contributions_f64<'c, 't>(
    ctx: &VjpCtx<'_, 'c, 't>,
    op: BinOpF64,
    id: usize,
    (a, b): (NodeIdF64, NodeIdF64),
    g: VarF64<'c, 't>,
) -> Result<Vec<(NodeIdF64, VarF64<'c, 't>)>, AutodiffError> {
    let VjpCtx {
        nodes,
        mirror,
        child,
    } = *ctx;
    let (a_shape, b_shape) = (&nodes[a.0].shape, &nodes[b.0].shape);
    let (need_a, need_b) = (nodes[a.0].requires_grad, nodes[b.0].requires_grad);
    let (a_m, b_m) = (mirrored(mirror, a)?, mirrored(mirror, b)?);
    let mut out = Vec::with_capacity(2);
    match op {
        BinOpF64::Add => {
            if need_a {
                out.push((a, g.cg_reduce_to(a_shape)?));
            }
            if need_b {
                out.push((b, g.cg_reduce_to(b_shape)?));
            }
        }
        BinOpF64::Mul => {
            if need_a {
                out.push((a, g.mul(&b_m)?.cg_reduce_to(a_shape)?));
            }
            if need_b {
                out.push((b, g.mul(&a_m)?.cg_reduce_to(b_shape)?));
            }
        }
        BinOpF64::Div => {
            if need_a {
                let factor = scalar_const(child, 1.0)?.div(&b_m)?;
                out.push((a, g.mul(&factor)?.cg_reduce_to(a_shape)?));
            }
            if need_b {
                // -a / b^2 を -(a/b)/b で組む（1 階と同じ overflow 耐性）。
                let factor = scalar_const(child, -1.0)?.mul(&a_m.div(&b_m)?.div(&b_m)?)?;
                out.push((b, g.mul(&factor)?.cg_reduce_to(b_shape)?));
            }
        }
        BinOpF64::Pow => {
            let out_shape = &node_shape(nodes, id);
            let a_bc = a_m.cg_broadcast_to(out_shape)?;
            let ones = full_const(child, out_shape, 1.0)?;
            let zeros = full_const(child, out_shape, 0.0)?;
            if need_a {
                let b_bc = b_m.cg_broadcast_to(out_shape)?;
                let neg_one = scalar_const(child, -1.0)?;
                // 指数 0 の位置の扱い（`factor = b * a^(b-1)` の逆伝播係数が
                // `(b-1) b a^(b-2)` になり、微小な正の底では `a^(b-2)` が overflow して
                // `0 * inf = NaN` になるのを避ける）。
                // - 指数が非追跡、または底が有限の正でない: 混合項は存在しない（または
                //   定義できない）ので係数を定数 0 にする。
                // - 指数が追跡され底が有限の正: 混合二階微分 `d/db [b a^(b-1)] = 1/a` を
                //   残すため `factor = (b * a^b) / a` の形で組む。`a^b = 1`（b = 0）の位置
                //   でのみ使うので overflow せず、`a` 側の微分も `0/a` で NaN にならない。
                // 逆数 `1/a` が overflow する非正規化数域の底（指数 0・指数追跡）は、
                // 混合二階微分 `1/a` が f64 で表せず、`inf * 0 = NaN` が `d2/da2`（本来 0）
                // を汚染するため、構築時に型付きエラーで拒否する。
                if need_b {
                    // 形状が異なる場合は broadcast 後の対応が必要なため、全要素の組合せで保守的に判定する。
                    let any_zero_b = b_m.value().host_slice().contains(&0.0);
                    let any_tiny_a = a_m
                        .value()
                        .host_slice()
                        .iter()
                        .any(|&x| x.is_finite() && x > 0.0 && !(1.0 / x).is_finite());
                    if any_zero_b && any_tiny_a {
                        return Err(AutodiffError::Backward(
                            "Pow: 指数 0 かつ底の逆数が overflow する非正規化数域では create_graph の混合二階微分を表せない".to_string(),
                        ));
                    }
                }
                let pairs = |a: f64, b: f64| -> (bool, bool) {
                    let b0 = b == 0.0;
                    let pos = a.is_finite() && a > 0.0;
                    (b0 && !(need_b && pos), b0 && need_b && pos)
                };
                let mask_zero =
                    pair_mask_const(child, &a_m.value(), &b_m.value(), out_shape, |a, b| {
                        pairs(a, b).0
                    })?;
                let mask_alt =
                    pair_mask_const(child, &a_m.value(), &b_m.value(), out_shape, |a, b| {
                        pairs(a, b).1
                    })?;
                let safe_a = mask_zero.cg_where(&ones, &a_bc)?;
                // 通常式は alt 位置も 1 に差し替える（そこで `a^(b-2)` を作らない）。
                let normal_a = mask_alt.cg_where(&ones, &safe_a)?;
                let a_pow = normal_a.pow(&b_bc.add(&neg_one)?)?;
                let normal = b_bc.mul(&a_pow)?;
                // alt 式は alt 位置の底だけを使い、他位置は 1 に差し替える。
                let alt_a = mask_alt.cg_where(&a_bc, &ones)?;
                let alt = b_bc.mul(&alt_a.pow(&b_bc)?)?.div(&alt_a)?;
                let factor = mask_alt.cg_where(&alt, &normal)?;
                let factor = mask_zero.cg_where(&zeros, &factor)?;
                out.push((a, g.mul(&factor)?.cg_reduce_to(a_shape)?));
            }
            if need_b {
                let y_m = mirrored(mirror, NodeIdF64(id))?;
                let mask_a0 = mask_const(child, &a_m.value(), out_shape, |v| v == 0.0)?;
                let safe_a = mask_a0.cg_where(&ones, &a_bc)?;
                let factor = mask_a0.cg_where(&zeros, &y_m.mul(&safe_a.cg_log()?)?)?;
                out.push((b, g.mul(&factor)?.cg_reduce_to(b_shape)?));
            }
        }
    }
    Ok(out)
}

fn node_shape(nodes: &[NodeF64], id: usize) -> Vec<usize> {
    nodes[id].shape.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;
    use crate::tape::Tape;

    fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
    }

    /// 対象外 Op・別デバイス等の入口検査で失敗したあとも `child` が空の
    /// まま残る（同じ child で正しい loss を渡せる）ことの確認。
    #[test]
    fn child_stays_empty_after_unsupported_op_rejection() {
        let base = Tape::new_with_ops(naive_ops());
        let parent = TapeF64::new(&base);
        let child = TapeF64::new(&base);
        let x = parent.var(&t(vec![1.0, 2.0], &[2]));
        let bad = x.max(None).expect("max");
        assert!(matches!(
            parent.backward_create_graph(&bad, &child),
            Err(AutodiffError::Backward(_))
        ));
        assert!(child.is_empty());
        let good = x.mul(&x).expect("mul");
        assert!(parent.backward_create_graph(&good, &child).is_ok());
    }

    /// 構築の途中で失敗したら `rollback_to` で子テープが空へ戻る。祖先集合に
    /// 不整合（`Mul` の入力 `x` を祖先から外す）を注入して `mirrored` を
    /// 失敗させる。
    #[test]
    fn build_failure_rolls_child_back_to_empty() {
        let base = Tape::new_with_ops(naive_ops());
        let parent = TapeF64::new(&base);
        let child = TapeF64::new(&base);
        let k = parent.var_no_grad(&t(vec![1.0], &[1]));
        let x = parent.var(&t(vec![2.0], &[1]));
        let y = x.mul(&k).expect("mul");
        // 祖先: k（定数）と y のみ。x を外す。
        let mut anc = vec![false; 3];
        anc[0] = true;
        anc[2] = true;
        let err = parent.build_with_rollback(y.id, &anc, &child);
        assert!(matches!(err, Err(AutodiffError::Backward(_))));
        assert!(child.is_empty());
    }

    /// 新しい内部 variant の 1 階 VJP（Where・Log・BroadcastTo・ReduceTo）が
    /// 閉形式と一致する。
    #[test]
    fn internal_variants_first_order_vjp_match_closed_forms() {
        let base = Tape::new_with_ops(naive_ops());
        let tape = TapeF64::new(&base);
        let x = tape.var(&t(vec![1.0, 2.0, 4.0], &[3]));
        let y = tape.var(&t(vec![5.0, 6.0, 7.0], &[3]));
        let cond = tape.var_no_grad(&t(vec![1.0, 0.0, 1.0], &[3]));
        // loss = where(cond, ln x, y) を broadcast して reduce した形
        let lx = x.cg_log().expect("log");
        let w = cond.cg_where(&lx, &y).expect("where");
        let b = w.cg_broadcast_to(&[2, 3]).expect("bcast");
        let r = b.cg_reduce_to(&[3]).expect("reduce");
        let grads = tape.backward(&r).expect("backward");
        // d/dx = cond * 2 / x、d/dy = (1-cond) * 2（broadcast で 2 行分）
        let gx = grads.get(&x).expect("get").expect("some");
        let gy = grads.get(&y).expect("get").expect("some");
        let want_x = [2.0 / 1.0, 0.0, 2.0 / 4.0];
        let want_y = [0.0, 2.0, 0.0];
        for (got, want) in gx.host_slice().iter().zip(want_x) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        for (got, want) in gy.host_slice().iter().zip(want_y) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }
}
