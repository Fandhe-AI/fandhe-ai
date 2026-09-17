//! 子テープ方式の高階微分（`create_graph`。イシュー #1942・親 #1940・
//! 設計 `docs/autodiff-higher-order-grad-decision.md` §7〜§9）。
//!
//! [`Tape::backward_create_graph`] は既存の 1 階 [`Tape::backward`]
//! （`backward.rs::backward_impl`。無変更）を呼んで 1 階勾配
//! （[`CreateGraphResult::first_order`]）を得たうえで、**呼び出し側が
//! あらかじめ構築した別インスタンスの空 `Tape`**（`child`）へ、
//! `loss` から到達する祖先ノードの forward 値を「写し」（mirror。
//! [`build_mirror`]）として登録し、その写しを使って VJP の各ステップを
//! `Var` 演算として記録する（[`build_cgrads`]）。子テープ上に記録された
//! 1 階勾配（[`CreateGraphResult::grad`] が返す `Var<'c>`）は通常の
//! `Var` と同じくさらに `child.backward(..)` で微分できるため、これが
//! 二階微分（grad of grad）の実体となる。
//!
//! **既存契約への影響（設計 doc §3「契約整理」）**: 1 階
//! `backward_impl`・`grad.rs`・REQ-2 統一複合判定・tolerance／baseline
//! は一切変更しない——本モジュールは `Tape::backward`（公開 API）を
//! 素のまま呼ぶのみで、`backward.rs`／`grad.rs` へのコード変更を伴わ
//! ない。二階側の数値方式（本モジュールが子テープへ記録する `Var`
//! 演算列）は既存 VJP ヘルパー（`grad.rs`）の数値方式（`f64` 縮約
//! 契約等）とは独立の実装であり、bit 同一は主張しない——正しさは
//! 有限差分突合（`tests/create_graph.rs`）で検証する。
//!
//! REQ-12「利用者向け融合制御 API を提供しない」は、`create_graph` が
//! 二階の勾配グラフを構築するかどうかの選択であり `docs/
//! fusion-graph-design.md` §3.3 の融合境界制御（forward の elementwise
//! 遅延グラフ）とは別軸のため抵触しない（設計 doc §9「REQ-12 整理」。
//! 子テープに記録された `Var` 演算自体は通常どおり融合対象になりうる）。
//!
//! **facade 非公開**（設計 doc §9「facade」・§10 承認事項 5 は未承認）:
//! `crates/facade` の `Tape` newtype は本モジュールの型・メソッドを
//! 一切再エクスポートしない。`docs/compat-api-scope.md` §5 の範囲拡張
//! 手続きを経ていないため、内部クレート（`fandhe_ai_autodiff`）限定の
//! 機能として留める。
//!
//! **初期スコープ（設計 doc §8。[`Op::supports_create_graph`] が判定する
//! 対象）**: `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・
//! `Sum`・`Mean`・`Reshape`・`BroadcastTo` の 11 variant のみ。それ以外の
//! 追跡対象 Op（`MatMul` を含む）へ到達した場合は
//! `Err(AutodiffError::Backward)`（fail-closed。#1943 等の後続イシューへ
//! 引き継ぐ）。`resident`／`fused` 経路（`ResidentLeaf`／
//! `LinearResident`／`LinearAct`）・checkpoint 済み親テープも同様に
//! fail-closed で拒否する。

use fandhe_ai_tensor_core::Tensor;

use crate::backward::Gradients;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Op, Tape, TapeId, TapeNode, materialize_fallible};
use crate::var::Var;

/// [`Tape::backward_create_graph`] の戻り値。1 階勾配
/// （[`Self::first_order`]。既存 `backward_impl` の無変更な結果）に
/// 加え、子テープ `'c` 上に構築した「元テープの祖先ノードの写し」
/// （[`Self::child_var`]）と「その 1 階勾配（子テープ上の `Var`）」
/// （[`Self::grad`]）を保持する。
#[derive(Debug)]
pub struct CreateGraphResult<'c> {
    first_order: Gradients,
    parent_id: TapeId,
    parent_epoch: u64,
    /// 元テープの `NodeId.0` を添字とする、子テープ上の写し
    /// （祖先ノードのみ `Some`）。
    mirror: Vec<Option<Var<'c>>>,
    /// 同じ添字で、子テープ上に構築した 1 階勾配 `Var`（loss から
    /// 到達した対象ノードのみ `Some`）。
    cgrads: Vec<Option<Var<'c>>>,
}

impl<'c> CreateGraphResult<'c> {
    /// 既存 `Tape::backward` と同一の 1 階勾配（`Tensor<f32>` の
    /// 入れ物）。子テープ構築とは独立のパス（`Tape::backward` をその
    /// まま呼ぶのみ）で計算するため、単体で `backward` を呼んだ場合と
    /// bit 同一（設計 doc §9「1 階勾配の bit 同一性」選択肢 (i)。
    /// 推奨として本イシューで採用）。
    pub fn first_order(&self) -> &Gradients {
        &self.first_order
    }

    /// `parent_var`（元テープ上の `Var`）の 1 階勾配を、子テープ上の
    /// `Var<'c>` として返す（これをさらに `child.backward(..)` で
    /// 微分すれば二階勾配が得られる）。
    ///
    /// - `parent_var` が本 `CreateGraphResult` を生んだ親テープ・世代と
    ///   一致しない場合: `Err(TapeMismatch)`。
    /// - `parent_var` が `requires_grad == false`（`var_no_grad`／
    ///   `detach` の葉、またはそれのみを祖先に持つノード）の場合:
    ///   `Err(GradientTrackingDisabled)`（`Gradients::get` と同じ区別。
    ///   `docs/autodiff-nograd-leaf-dinput-skip-decision.md` §5）。
    /// - `loss` から未到達、または対象 Op が
    ///   [`Op::supports_create_graph`] を満たさない部分木の外側にある
    ///   場合: `Ok(None)`。
    pub fn grad(&self, parent_var: &Var<'_>) -> Result<Option<Var<'c>>, AutodiffError> {
        self.check(parent_var)?;
        Ok(self.cgrads.get(parent_var.node_id().0).copied().flatten())
    }

    /// `parent_var` の子テープ上の写し（forward 値の再生・または
    /// 定数葉化）を返す。`cg.grad(&cg.child_var(&x)?.unwrap())` の
    /// ように、子テープ上でさらに `backward` した [`Gradients`] から
    /// 二階勾配を読み出す際の鍵として使う。
    ///
    /// 検査は [`Self::grad`] と同一。
    pub fn child_var(&self, parent_var: &Var<'_>) -> Result<Option<Var<'c>>, AutodiffError> {
        self.check(parent_var)?;
        Ok(self.mirror.get(parent_var.node_id().0).copied().flatten())
    }

    fn check(&self, parent_var: &Var<'_>) -> Result<(), AutodiffError> {
        if parent_var.tape_id() != self.parent_id || parent_var.tape_epoch() != self.parent_epoch {
            return Err(AutodiffError::TapeMismatch);
        }
        if !parent_var.requires_grad() {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        Ok(())
    }
}

impl Tape {
    /// 子テープ方式の `create_graph`（PyTorch
    /// `torch.autograd.grad(..., create_graph=True)` 相当。イシュー
    /// #1942）。`loss` を起点に逆伝播しつつ、
    /// [`Op::supports_create_graph`] を満たす祖先ノードの VJP を
    /// `child`（呼び出し側があらかじめ [`Tape::new_with_ops`] 等で
    /// 構築した**空**の `Tape`）上へ `Var` 演算として記録する。
    ///
    /// **入口検査（順序固定）**:
    /// 1. `loss` が `self` に属さない → `Err(TapeMismatch)`。
    /// 2. `child` が `self` と同一テープ → `Err(Backward)`
    ///    （親と子を同一インスタンスにはできない）。
    /// 3. `child.device() != self.device()` → `Err(DeviceMismatch)`。
    /// 4. `child` が空でない → `Err(Backward)`（子テープの葉プレ
    ///    フィックス契約〈`Tape::reset` doc〉を素直に保つため、既存
    ///    ノードを持つテープの再利用は許さない）。
    /// 5. `self` に checkpoint 区間が登録済み → `Err(Backward)`
    ///    （[`Tape::has_registered_checkpoints`] doc 参照）。
    /// 6. 上記を満たせば既存 [`Tape::backward`] をそのまま呼んで 1 階
    ///    勾配を得る（`loss.requires_grad() == false` の拒否もここで
    ///    既存どおり発生する）。
    ///
    /// 4〜6 のいずれかで失敗した場合、`child` へは一切書き込まない
    /// （4 の事前検査により `child` の空性が確認済みのため、途中失敗で
    /// `child` へノードが残ることはない）。
    pub fn backward_create_graph<'c>(
        &self,
        loss: &Var<'_>,
        child: &'c Tape,
    ) -> Result<CreateGraphResult<'c>, AutodiffError> {
        if loss.tape_id() != self.id {
            return Err(AutodiffError::TapeMismatch);
        }
        if child.id == self.id {
            return Err(AutodiffError::Backward(
                "create_graph: 子テープに親と同一の Tape は指定できない".into(),
            ));
        }
        if child.device() != self.device() {
            return Err(AutodiffError::DeviceMismatch {
                requested: child.device(),
                actual: self.device(),
            });
        }
        if !child.is_empty() {
            return Err(AutodiffError::Backward(
                "create_graph: 子テープは空である必要がある（既存ノードを持つ Tape は渡せない）"
                    .into(),
            ));
        }
        if self.has_registered_checkpoints() {
            return Err(AutodiffError::Backward(
                "create_graph: checkpoint 区間が登録済みの親テープでは create_graph は未対応（\
                 docs/autodiff-higher-order-grad-decision.md §8 参照）"
                    .into(),
            ));
        }

        let first_order = self.backward(loss)?;

        let ancestors = collect_ancestors(self, loss.node_id());
        let n = self.nodes.borrow().len();
        let mut mirror: Vec<Option<Var<'c>>> = vec![None; n];
        build_mirror(self, child, &ancestors, &mut mirror)?;
        let cgrads = build_cgrads(self, child, &ancestors, &mirror, loss.node_id())?;

        Ok(CreateGraphResult {
            first_order,
            parent_id: self.id,
            parent_epoch: self.epoch(),
            mirror,
            cgrads,
        })
    }
}

/// `root` から [`Op::for_each_input`] を辿って到達する祖先ノードの
/// `NodeId` を昇順（元テープの発生順。トポロジカル順と一致する——
/// 入力は常に自身より小さい `NodeId` を持つため。`Tape::push_*` の
/// 追記専用構造から導かれる不変条件）で返す。
///
/// **`requires_grad == false` のノードでは descend しない**（設計 doc
/// §8「`var_no_grad`／`detach` 葉との相互作用」）: 当該ノードは
/// [`build_mirror`] が forward 値をそのまま定数葉化するため、その入力
/// 側を祖先集合へ含める必要がない——むしろ含めると、対象外の Op
/// （resident 経路等）がたまたま `requires_grad == false` の部分木に
/// あるだけで `Err` になってしまい、PyTorch の「定数として扱われる
/// 演算は任意の Op でよい」という直感的な挙動を壊す。
fn collect_ancestors(tape: &Tape, root: NodeId) -> Vec<NodeId> {
    let nodes = tape.nodes.borrow();
    let n = nodes.len();
    let mut visited = vec![false; n];
    let mut stack = vec![root];
    let mut ids: Vec<NodeId> = Vec::new();
    while let Some(id) = stack.pop() {
        if visited[id.0] {
            continue;
        }
        visited[id.0] = true;
        ids.push(id);
        let node = &nodes[id.0];
        if node.requires_grad {
            node.op.for_each_input(|input_id| stack.push(input_id));
        }
    }
    ids.sort_unstable_by_key(|id| id.0);
    ids
}

/// 祖先ノードの「写し」を子テープ上へ構築する（[`Tape::backward_
/// create_graph`] の本体その 1）。**2 段構成**（`Tape::reset` の葉
/// プレフィックス契約を子テープ側でも保つため。設計 doc §9「世代
/// 契約」）:
///
/// 1. 祖先すべてを昇順で走査し、葉相当（`Op::Leaf`、または
///    `requires_grad == false` の任意ノード）を先に子テープの葉として
///    登録する——`requires_grad == false` の任意ノードは、その Op が
///    `resident`／`fused`／未対応のいずれであっても、値を実体化して
///    子テープの `var_no_grad` 葉へ変換するだけで済む（replay 不要。
///    ただし resident／fused 経路〈`ResidentLeaf`／`LinearResident`／
///    `LinearAct`〉は値の実体化自体が成立しないため、`requires_grad`
///    の値に関わらず本段の**手前**で無条件に拒否する）。
/// 2. 残り（`requires_grad == true` かつ非葉）を昇順で走査し、
///    [`Op::supports_create_graph`] を満たす場合のみ `Var` 演算として
///    再生する。満たさない場合は `Err`（fail-closed。#1943 等へ引き継ぐ
///    未対応 Op）。
fn build_mirror<'c>(
    parent: &Tape,
    child: &'c Tape,
    ancestors: &[NodeId],
    mirror: &mut [Option<Var<'c>>],
) -> Result<(), AutodiffError> {
    let parent_nodes = parent.nodes.borrow();
    let parent_ops = parent.ops();

    // 段 1: resident／fused 経路を拒否しつつ、葉相当を先にすべて登録する。
    for &id in ancestors {
        let node = &parent_nodes[id.0];
        if matches!(
            node.op,
            Op::ResidentLeaf { .. } | Op::LinearResident { .. } | Op::LinearAct { .. }
        ) {
            return Err(AutodiffError::Backward(format!(
                "create_graph: resident／fused 経路の Op（NodeId({}))は非対応（\
                 docs/autodiff-higher-order-grad-decision.md §8 参照）",
                id.0
            )));
        }
        if !node.requires_grad {
            let value = materialize_fallible(&parent_nodes, parent_ops, id)?.clone();
            mirror[id.0] = Some(child.var_no_grad(&value));
            continue;
        }
        if matches!(node.op, Op::Leaf) {
            let value = materialize_fallible(&parent_nodes, parent_ops, id)?.clone();
            mirror[id.0] = Some(child.var(&value));
        }
    }

    // 段 2: 残り（requires_grad == true かつ非葉）を再生する。
    for &id in ancestors {
        if mirror[id.0].is_some() {
            continue;
        }
        let node = &parent_nodes[id.0];
        if !node.op.supports_create_graph() {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 未対応の Op（NodeId({}))へ到達した（\
                 docs/autodiff-higher-order-grad-decision.md §8 の対象 Op のみ再生可能）",
                id.0
            )));
        }
        let replayed = replay_op(node.op.clone(), &node.shape, mirror)?;
        mirror[id.0] = Some(replayed);
    }

    Ok(())
}

/// 子テープ上の写しを取得する（未構築なら内部不変条件違反として
/// `Err`。本番経路 panic 禁止方針〈`.claude/rules/coding-rust.md`〉に
/// 従い `unwrap`/`expect` は使わない）。[`collect_ancestors`]・
/// [`build_mirror`] の不変条件（祖先はトポロジカル順に登録されるため、
/// 非葉ノードの再生・逆走査の時点で入力側の写しは必ず存在する）が
/// 保たれている限り、この `Err` 分岐は実行時には到達しない。
fn get_mirror<'c>(mirror: &[Option<Var<'c>>], id: NodeId) -> Result<Var<'c>, AutodiffError> {
    mirror[id.0].ok_or_else(|| {
        AutodiffError::Backward(format!(
            "create_graph: 祖先ノード（NodeId({}))の写しが未構築（内部不変条件違反）",
            id.0
        ))
    })
}

/// 祖先ノード 1 個分の forward を子テープ上で再生する（[`build_mirror`]
/// 段 2 の本体）。呼び出し元は `op.supports_create_graph()` を確認済み
/// のため、ここでの `_ =>` 到達は契約違反（`AutodiffError::Backward`
/// で fail-closed に拒否する。本番経路 panic 禁止方針に従い `panic!`
/// ではなく型付きエラーとする）。
///
/// `shape` は再生対象ノード自身の `TapeNode::shape`（`Op::Reshape`／
/// `Op::BroadcastTo` の目的 shape はこの `shape` から読む——`Op` 自身は
/// 入力 `NodeId` のみを保持し目的 shape を持たないため）。
fn replay_op<'c>(
    op: Op,
    shape: &[usize],
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match op {
        Op::Add(a, b) => get_mirror(mirror, a)?.add(&get_mirror(mirror, b)?),
        Op::Mul(a, b) => get_mirror(mirror, a)?.mul(&get_mirror(mirror, b)?),
        Op::Relu(a) => Ok(get_mirror(mirror, a)?.relu()),
        Op::Exp(a) => Ok(get_mirror(mirror, a)?.exp()),
        Op::Tanh(a) => Ok(get_mirror(mirror, a)?.tanh()),
        Op::Sigmoid(a) => Ok(get_mirror(mirror, a)?.sigmoid()),
        Op::Sum { input, dim } => get_mirror(mirror, input)?.sum(dim),
        Op::Mean { input, dim } => get_mirror(mirror, input)?.mean(dim),
        Op::Reshape { input } => get_mirror(mirror, input)?.contiguous()?.reshape(shape),
        Op::BroadcastTo { input } => get_mirror(mirror, input)?.broadcast_to(shape),
        _ => Err(AutodiffError::Backward(
            "create_graph: replay_op: supports_create_graph() が true の未対応 Op（内部契約違反）"
                .into(),
        )),
    }
}

/// [`Tape::backward_create_graph`] の逆走査（本体その 2。VJP を `Var`
/// 演算として子テープへ記録する）。`backward_impl`（`backward.rs`）と
/// 同型の「発生順とは逆順に走査し寄与を蓄積する」構造だが、蓄積先の
/// 値そのものが `Tensor<f32>` ではなく子テープ上の `Var<'c>`（さらに
/// 微分可能）である点が異なる。
fn build_cgrads<'c>(
    parent: &Tape,
    child: &'c Tape,
    ancestors: &[NodeId],
    mirror: &[Option<Var<'c>>],
    loss_id: NodeId,
) -> Result<Vec<Option<Var<'c>>>, AutodiffError> {
    let parent_nodes = parent.nodes.borrow();
    let parent_ops = parent.ops();
    let n = parent_nodes.len();
    let mut cgrads: Vec<Option<Var<'c>>> = vec![None; n];

    // シード: `Tape::backward`（`backward.rs::backward_impl`）と同じ
    // 「非スカラー loss は全要素 1 の暗黙の総和射影」意味論。
    let loss_shape = parent_nodes[loss_id.0].shape.clone();
    let seed_tensor = Tensor::full(&loss_shape, 1.0f32)?;
    cgrads[loss_id.0] = Some(child.var_no_grad(&seed_tensor));

    for &id in ancestors.iter().rev() {
        let Some(g) = cgrads[id.0] else {
            continue;
        };
        let node = &parent_nodes[id.0];
        match node.op.clone() {
            Op::Leaf => {}
            Op::Add(a, b) => {
                let a_shape = parent_nodes[a.0].shape.clone();
                let b_shape = parent_nodes[b.0].shape.clone();
                let da = reduce_to(&g, &a_shape)?;
                let db = reduce_to(&g, &b_shape)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::Mul(a, b) => {
                let a_m = get_mirror(mirror, a)?;
                let b_m = get_mirror(mirror, b)?;
                let ga = g.mul(&b_m)?;
                let gb = g.mul(&a_m)?;
                let da = reduce_to(&ga, &a_m.shape())?;
                let db = reduce_to(&gb, &b_m.shape())?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::Relu(a) => {
                // 劣勾配は x = 0 で 0（`grad.rs::Op::Relu` の VJP と同じ
                // 規約）。マスクは**入力側**の実測値（親テープ上で既に
                // 実体化済み）から作る——出力値ではなく入力値を見るのは
                // `grad.rs` の既存規約（`v > 0.0`）と一致させるため。
                let a_val = materialize_fallible(&parent_nodes, parent_ops, a)?;
                let mask = positive_mask(a_val)?;
                let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
                let da = Var::where_cond(&mask, &g, &zeros)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Exp(a) => {
                // d/dx exp(x) = exp(x)。子テープ上の写し（= exp(x) の
                // 子テープ上の再生値）を再利用し、`exp` を再計算しない
                // （`grad.rs::Op::Exp` の `out_value` 再利用と同じ方針）。
                let out_c = get_mirror(mirror, id)?;
                let da = g.mul(&out_c)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Tanh(a) => {
                // d/dx tanh(x) = 1 - tanh(x)^2。
                let out_c = get_mirror(mirror, id)?;
                let one = child.var_no_grad(&Tensor::full(&out_c.shape(), 1.0f32)?);
                let sq = out_c.mul(&out_c)?;
                let factor = one.sub(&sq)?;
                let da = g.mul(&factor)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Sigmoid(a) => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))。
                let out_c = get_mirror(mirror, id)?;
                let one = child.var_no_grad(&Tensor::full(&out_c.shape(), 1.0f32)?);
                let comp = one.sub(&out_c)?;
                let factor = out_c.mul(&comp)?;
                let da = g.mul(&factor)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Sum { input, dim } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let da = match dim {
                    None => g.broadcast_to(&input_shape)?,
                    Some(axis) => g
                        .contiguous()?
                        .unsqueeze(axis)?
                        .broadcast_to(&input_shape)?,
                };
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Mean { input, dim } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let count: usize = match dim {
                    None => input_shape.iter().product(),
                    Some(axis) => input_shape[axis],
                };
                let scale = child.var_no_grad(&Tensor::scalar(1.0f32 / count as f32));
                let g_scaled = g.mul(&scale)?;
                let da = match dim {
                    None => g_scaled.broadcast_to(&input_shape)?,
                    Some(axis) => g_scaled
                        .contiguous()?
                        .unsqueeze(axis)?
                        .broadcast_to(&input_shape)?,
                };
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Reshape { input } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let da = g.contiguous()?.reshape(&input_shape)?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::BroadcastTo { input } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let da = reduce_to(&g, &input_shape)?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            _ => {
                return Err(AutodiffError::Backward(
                    "create_graph: build_cgrads: supports_create_graph() が true の未対応 Op\
                     （内部契約違反）"
                        .into(),
                ));
            }
        }
    }

    Ok(cgrads)
}

/// 同一入力ノードへ複数経路から流入した勾配（子テープ上の `Var`）を
/// `Var::add` で合算する（`backward.rs::accumulate` の子テープ版）。
/// `target` が `requires_grad == false`（`docs/
/// autodiff-nograd-leaf-dinput-skip-decision.md` の契約と同型）なら
/// 寄与を捨てる——`backward_impl` が `requires_grad == false` の
/// ノードへの寄与を `accumulate` へ渡さず捨てるのと同じ意味論
/// （`target` の `mirror` は `var_no_grad` の定数葉であり、その勾配を
/// 蓄積する意味がないため）。
fn accumulate<'c>(
    parent_nodes: &[TapeNode],
    cgrads: &mut [Option<Var<'c>>],
    target: NodeId,
    contribution: Var<'c>,
) -> Result<(), AutodiffError> {
    if !parent_nodes[target.0].requires_grad {
        return Ok(());
    }
    match cgrads[target.0] {
        Some(existing) => {
            cgrads[target.0] = Some(existing.add(&contribution)?);
        }
        None => {
            cgrads[target.0] = Some(contribution);
        }
    }
    Ok(())
}

/// broadcast された `v` を `target_shape` へ縮約する（`Var::add`／
/// `mul` の VJP が使う `reduce_to_shape`〈`grad.rs`〉の `Var` 演算版）。
/// 子テープ上へ `Var::sum_dims`（`keepdim=true`）＋必要なら
/// `Var::reshape` として記録する——`grad.rs::reduce_to_shape` の数値
/// 方式（生テンソルの直接縮約）とは独立の実装であり、bit 同一は主張
/// しない（正しさは有限差分突合〈`tests/create_graph.rs`〉のみを根拠
/// とする。モジュール doc 参照）。
fn reduce_to<'c>(v: &Var<'c>, target_shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    let v_shape = v.shape();
    if v_shape == target_shape {
        return Ok(*v);
    }
    let rank_diff = v_shape.len().saturating_sub(target_shape.len());
    let mut dims: Vec<usize> = (0..rank_diff).collect();
    for (i, (&vs, &ts)) in v_shape[rank_diff..]
        .iter()
        .zip(target_shape.iter())
        .enumerate()
    {
        if ts == 1 && vs != 1 {
            dims.push(rank_diff + i);
        }
    }
    if dims.is_empty() {
        // `v_shape != target_shape` かつ縮約対象軸が 1 つもない場合は
        // 通常到達しない（有効な broadcast 由来の shape であれば必ず
        // 上のいずれかの条件に当たる）が、安全側に `reshape` へ委ね、
        // 要素数不一致なら型付きエラーとして fail-closed に拒否する。
        return v.contiguous()?.reshape(target_shape);
    }
    let reduced = v.sum_dims(&dims, true)?;
    let reduced_shape = reduced.shape();
    if reduced_shape == target_shape {
        Ok(reduced)
    } else {
        reduced.contiguous()?.reshape(target_shape)
    }
}

/// `t`（親テープ側の実測値）の各要素が正かどうかを表す `Tensor<bool>`
/// を構築する（`Op::Relu` の VJP マスク。`v > 0.0`。NaN はマスク
/// 不成立——`grad.rs::elementwise_mul_mask` が `Op::Relu` に適用する
/// 規約と同じ）。
fn positive_mask(t: &Tensor<f32>) -> Result<Tensor<bool>, AutodiffError> {
    let contiguous = t.contiguous();
    let data: Vec<bool> = contiguous
        .as_slice()
        .map(|s| s.iter().map(|&v| v > 0.0).collect())
        .unwrap_or_default();
    Tensor::new(data, contiguous.shape()).map_err(AutodiffError::from)
}
