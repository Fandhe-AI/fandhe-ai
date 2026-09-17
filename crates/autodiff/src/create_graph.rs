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
//! 演算列）は既存 VJP ヘルパー（`grad.rs`）とは独立の実装であり、
//! 一般には bit 同一を主張しない——正しさは有限差分突合
//! （`tests/create_graph.rs`）で検証する。ただし broadcast 縮約
//! （`reduce_to`）は例外で、`grad.rs::reduce_to_shape` の f32 逐次和
//! アルゴリズムを `Var::narrow`／`Var::add` の連鎖として逐語再現して
//! おり、1 階 VJP と数値方式（縮約順序・アキュムレータ精度）が一致
//! する（`reduce_to` のドキュメント参照。codex-review 指摘・PR
//! #1998 で是正——当初 `Var::sum_dims`〈CPU `sum` の `f64` アキュム
//! レータ縮約〉を使っていたため、桁落ちを伴う broadcast 入力で 1 階
//! 勾配と乖離し REQ-2 統一複合判定を満たさない具体例があった）。
//! `Op::Add` の bias パターン（`upstream: [m, n]` → `[n]`／`[1, n]`
//! の行方向縮約）に限っては [`reduce_bias_grad_var`] が
//! `grad.rs::reduce_bias_grad`（f64 相当のアキュムレータ。2026-09-12
//! ユーザー承認・`.claude/rules/coding-rust.md` の勾配長軸縮約契約）
//! と数値方式を揃える別経路を使う——`eval::reduce_bias_grad_rows`
//! （1 階 VJP と同一のホスト関数）でホスト側の値を直接計算し
//! `Tape::push_eager` で子テープへ登録する（`Var::sum` は
//! `child.ops()` の実装〈テスト用 `naive_ops()` は f32 逐次和〉に
//! 依存するため使わない。codex-review 指摘・PR #1998 是正）。
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
//! **対象スコープ（設計 doc §8。`Op::supports_create_graph` が判定する
//! 対象）**: `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・
//! `Sum`・`Mean`・`Reshape`・`BroadcastTo`・`MatMul`（rank 2 × rank 2
//! 限定。#1943）の 12 variant のみ。それ以外の追跡対象 Op・rank≥3 の
//! `MatMul` へ到達した場合は `Err(AutodiffError::Backward)`
//! （fail-closed。後続イシューへ引き継ぐ）。`resident`／`fused` 経路
//! （`ResidentLeaf`／`LinearResident`／`LinearAct`）・checkpoint 済み
//! 親テープも同様に fail-closed で拒否する（[`validate_ancestors`]。
//! `child` へ一切書き込む前の入口で判定するため、途中失敗時も
//! `child` は無変更のまま保たれる）。
//!
//! **`MatMul` の数値契約（イシュー #1943・PR #2003 codex-review 指摘
//! で是正）**: 子テープの matmul VJP（[`build_cgrads`] の
//! `Op::MatMul` 腕）は [`Var::matmul_fp32_strict`]（`ops().
//! gemm_fp32_strict` 経由。`crate::var` 限定公開）を使い、1 階
//! `matmul_vjp`（`grad.rs`。同じく `ops.gemm_fp32_strict` 経由）と
//! 入口を揃えている——当初案の `Var::matmul`（`ops().gemm`）は CUDA
//! TF32 opt-in（`docs/cuda-tf32-optin-api-decision.md`）が有効な間、
//! 1 階 `matmul_vjp` が守る「バックプロパゲーションは常に FP32 厳密」
//! という契約を二階微分の記録経路でだけ破ってしまうため、`Var::
//! matmul_fp32_strict` へ切り替えた。CPU バックエンドは両者が同一
//! カーネルへ帰着するため bit 同一のまま不変。さらに、
//! `matmul_fp32_strict` が記録する `Op::MatMul` ノードは
//! `TapeNode::fp32_strict` フラグにより activation checkpointing
//! （`Tape::checkpoint`／`Var::checkpoint_from`）の解放対象からも
//! 除外される（`release_checkpoint_region` doc 参照）——`Op::MatMul`
//! variant 自体は forward 精度の情報を持たないため、解放後の再計算
//! （`recompute_value`）が非厳密な `matmul_forward`（`ops.gemm`）を
//! 使ってしまう事故を防ぐ。本モジュールは元々「子テープの数値方式は
//! 一般に bit 同一を主張せず、正しさは有限差分突合で検証する」立場
//! （本ファイル冒頭の既存契約節）を取っており、この整理はその範囲内
//! に収まる（tolerance／baseline は無変更）。

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
    ///   `Op::supports_create_graph` を満たさない部分木の外側にある
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
    /// `Op::supports_create_graph` を満たす祖先ノードの VJP を
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
    ///    （`Tape::has_registered_checkpoints` doc 参照）。
    /// 6. `loss` から到達する祖先ノードを `collect_ancestors` で
    ///    走査し、`validate_ancestors` で resident／fused 経路・
    ///    未対応 Op・rank≥3 の `MatMul` を事前拒否する
    ///    （`Err(AutodiffError::Backward)`）。**素の [`Tape::backward`]
    ///    より前に行う**——resident グラフ（`Op::ResidentLeaf`／
    ///    `Op::LinearResident`）に対しては素の `backward` 自体が
    ///    `AutodiffError::InvalidArgument`（`DeviceParamStore::
    ///    backward` を使えという誤誘導的なメッセージ）を返してしまう
    ///    ため、先に構造的検査で正確な型付きエラーを返す
    ///    （イシュー #1943）。
    /// 7. 上記を満たせば既存 [`Tape::backward`] をそのまま呼んで 1 階
    ///    勾配を得る（`loss.requires_grad() == false` の拒否もここで
    ///    既存どおり発生する）。
    ///
    /// 4〜7 のいずれかで失敗した場合、`child` へは一切書き込まない
    /// （4 の事前検査により `child` の空性が確認済みであること、6 が
    /// `build_mirror`／`build_cgrads` より前に走ることの両方に
    /// より、途中失敗で `child` へノードが残ることはない）。
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

        // `ancestors` の収集・検査（[`validate_ancestors`]）は `self.
        // backward(loss)` より前に行う——resident グラフ（`Op::
        // ResidentLeaf`／`Op::LinearResident`）は素の `Tape::backward`
        // 自体が `AutodiffError::InvalidArgument`（`DeviceParamStore::
        // backward` を使えという誤誘導的なメッセージ）を返してしまう
        // ため、先に構造的な事前検査で「create_graph は resident 経路
        // 非対応」という型付き `Err(Backward)` を返す方が呼び出し側に
        // とって正確（イシュー #1943）。`collect_ancestors`／
        // `validate_ancestors` はいずれも `self`（構造の読み取りのみ）
        // に依存し `backward` の結果を必要としないため、順序を入れ替え
        // ても既存の 1 階勾配計算（[`Self::backward`]。無変更）には
        // 影響しない。
        let ancestors = collect_ancestors(self, loss.node_id());
        validate_ancestors(self, &ancestors)?;

        let first_order = self.backward(loss)?;

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

/// [`Tape::backward_create_graph`] の入口検査 7（[`build_mirror`]／
/// [`build_cgrads`] が `child` へ一切書き込む前に呼ぶ純関数。イシュー
/// #1943）。`ancestors`（[`collect_ancestors`] の結果）を走査し、
/// 以下のいずれかに該当するノードがあれば即座に `Err` を返す:
///
/// (a) `Op::ResidentLeaf`／`Op::LinearResident`／`Op::LinearAct`
///     （`requires_grad` の値に関わらず。[`build_mirror`] 段 1 の
///     拒否と同一条件——値の実体化自体が成立しないため）。
/// (b) `requires_grad == true` かつ非葉（`Op::Leaf` でない）で
///     `Op::supports_create_graph() == false`。
/// (c) `requires_grad == true` の `Op::MatMul` で、いずれかの入力
///     shape の rank が 2 でない（子テープの matmul VJP は
///     rank 2 × rank 2 限定。[`Op::supports_create_graph`] の doc
///     参照）。
///
/// **多層防御**: [`build_mirror`]／[`build_cgrads`] 自身も同型の
/// 拒否分岐（(a) は段 1 の手前・(b) は段 2 の `supports_create_graph`
/// 検査）を保持したまま残す——本関数はそれらより前に実行され、失敗
/// 時に `child` へ一部ノードが残る事態（4 の事前検査が保証するのは
/// 「呼び出し開始時点で `child` が空」であって「呼び出し失敗時に
/// `child` が空のまま」ではない）を避けるための追加ゲートである。
fn validate_ancestors(parent: &Tape, ancestors: &[NodeId]) -> Result<(), AutodiffError> {
    let parent_nodes = parent.nodes.borrow();
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
        if !node.requires_grad || matches!(node.op, Op::Leaf) {
            continue;
        }
        if !node.op.supports_create_graph() {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 未対応の Op（NodeId({}))へ到達した（\
                 docs/autodiff-higher-order-grad-decision.md §8 の対象 Op のみ再生可能）",
                id.0
            )));
        }
        if let Op::MatMul(a, b) = node.op {
            let a_rank = parent_nodes[a.0].shape.len();
            let b_rank = parent_nodes[b.0].shape.len();
            if a_rank != 2 || b_rank != 2 {
                return Err(AutodiffError::Backward(format!(
                    "create_graph: rank≥3 の MatMul（NodeId({}))は非対応（\
                     子テープの matmul VJP は rank 2 × rank 2 限定。\
                     docs/autodiff-higher-order-grad-decision.md §14 参照）",
                    id.0
                )));
            }
        }
    }
    Ok(())
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
///    `Op::supports_create_graph` を満たす場合のみ `Var` 演算として
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
        let replayed = replay_op(node.op.clone(), &node.shape, node.fp32_strict, mirror)?;
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
///
/// `fp32_strict` は再生対象ノード自身の `TapeNode::fp32_strict`
/// （PR #2003 codex-review 指摘で追加）。`Op::MatMul` variant 自体は
/// forward 精度の情報を持たないため、親テープ側で
/// `Var::matmul_fp32_strict` により記録されたノード（1 階
/// `matmul_vjp` の `d_input`／`d_weight` 等）を再生する際は、子テープ
/// 側でも `Var::matmul_fp32_strict` を使い、通常の `Var::matmul`
/// （`ops.gemm`。CUDA TF32 opt-in が有効な場合は非厳密）へ落ちて
/// 「バックプロパゲーションは常に FP32 厳密」という契約
/// （`.claude/rules/coding-rust.md`・`Var::matmul_fp32_strict` doc）を
/// 二階微分の記録経路でだけ破ることを防ぐ。
fn replay_op<'c>(
    op: Op,
    shape: &[usize],
    fp32_strict: bool,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match op {
        Op::Add(a, b) => get_mirror(mirror, a)?.add(&get_mirror(mirror, b)?),
        Op::Mul(a, b) => get_mirror(mirror, a)?.mul(&get_mirror(mirror, b)?),
        // rank 2 × rank 2 限定（`validate_ancestors` (c) が入口で
        // 事前検査済み）。親ノードが `fp32_strict` フラグを立てて
        // いれば（`Var::matmul_fp32_strict` 由来）子テープ側も
        // `matmul_fp32_strict` で再生し、`ops.gemm_fp32_strict`
        // 経由の厳密精度契約を引き継ぐ。フラグが立っていなければ
        // 通常の `Var::matmul`（`ops.gemm`）で forward と同一カーネル
        // を経由する。
        Op::MatMul(a, b) => {
            let lhs = get_mirror(mirror, a)?;
            let rhs = get_mirror(mirror, b)?;
            if fp32_strict {
                lhs.matmul_fp32_strict(&rhs)
            } else {
                lhs.matmul(&rhs)
            }
        }
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
                // codex-review 指摘（PR #1998）: `grad.rs::Op::Add` の
                // VJP は bias パターン（`upstream: [m, n]` →
                // `[n]`／`[1, n]` の行方向縮約）に限り `reduce_bias_
                // grad`（f64 相当アキュムレータ）へ委譲する
                // （`.claude/rules/coding-rust.md` の勾配長軸縮約
                // 契約・2026-09-12 ユーザー承認）。子テープ側も同じ
                // 分岐を [`reduce_bias_grad_var`] で再現し、bias
                // パターン以外は従来どおり [`reduce_to`]（`reduce_to_
                // shape` と同じ f32 逐次和）のまま維持する。
                let da = reduce_bias_grad_var(child, &g, &a_shape)?;
                let db = reduce_bias_grad_var(child, &g, &b_shape)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::Mul(a, b) => {
                let a_m = get_mirror(mirror, a)?;
                let b_m = get_mirror(mirror, b)?;
                let ga = g.mul(&b_m)?;
                let gb = g.mul(&a_m)?;
                let da = reduce_to(child, &ga, &a_m.shape())?;
                let db = reduce_to(child, &gb, &b_m.shape())?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::MatMul(a, b) => {
                // 1 階 `grad.rs::matmul_vjp`（rank 2 経路）と同一の
                // オペランド順序（`da = gemm(g, bᵀ)`・
                // `db = gemm(aᵀ, g)`）を `transpose` と
                // `Var::matmul_fp32_strict`（codex-review 指摘。
                // PR #2003）の合成として子テープ上へ記録する。
                // `Var::matmul`（`ops.gemm`）ではなく
                // `matmul_fp32_strict`（`ops.gemm_fp32_strict`）を
                // 使うのは、CUDA TF32 opt-in（`set_cuda_gemm_precision`）
                // が有効な間も 1 階 `matmul_vjp` と同じく
                // バックプロパゲーションを FP32 厳密のまま保つため
                // （`Var::matmul_fp32_strict` doc 参照）。
                // `requires_grad == false` 側（`accumulate` が捨てる）
                // でも VJP 自体は記録して構わない——`transpose`／
                // `matmul_fp32_strict` は追加の副作用を持たないため
                // 無駄なノードが増えるだけで正しさに影響しない
                // （既存 `Op::Add`／`Mul` 腕と同じ方針）。
                let a_m = get_mirror(mirror, a)?;
                let b_m = get_mirror(mirror, b)?;
                let da = g.matmul_fp32_strict(&b_m.transpose(0, 1)?)?;
                let db = a_m.transpose(0, 1)?.matmul_fp32_strict(&g)?;
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
                let da = reduce_to(child, &g, &input_shape)?;
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
///
/// **数値方式は `grad.rs::reduce_to_shape` と同一のアルゴリズムを
/// `Var` 演算列として逐語再現する**（`Var::narrow` で縮約対象軸を
/// 長さ 1 のスライスへ分解し、`Var::add` で添字昇順に逐次加算する。
/// `reduce_to_shape` の二重ループ `for axis { for a in 0..axis_len {
/// reduced[...] += data[src] } }` と同じ縮約順序・同じ f32 逐次和で
/// あり、CPU 上の `Var::add` は要素ごとの単純加算〈f64 アキュムレータ
/// を挟まない〉のため 1 階 VJP の `reduce_to_shape` と bit 同一になる
/// ——当初実装〈`Var::sum_dims`。CPU `sum` の f64 アキュムレータ縮約〉
/// では、桁落ちを伴う broadcast 入力〈大きさの異なる値が完全に
/// 相殺するケース〉で 1 階勾配と乖離し REQ-2 統一複合判定を満たさない
/// 具体例が確認されたため是正した。codex-review 指摘（PR #1998）。
fn reduce_to<'c>(
    child: &'c Tape,
    v: &Var<'c>,
    target_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let v_shape = v.shape();
    if v_shape == target_shape {
        return Ok(*v);
    }
    debug_assert!(
        v_shape.len() >= target_shape.len(),
        "reduce_to: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = v_shape.len().saturating_sub(target_shape.len());
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let mut cur = *v;
    let mut cur_shape = v_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let axis_len = cur_shape[axis];
            if axis_len == 0 {
                // 合法な空テンソル（縮約対象の軸長が 0。例:
                // `x: [3]` を `broadcast_to(&[0, 3])` した結果を
                // 逆縮約する場合）。`reduce_to_shape`（1 階 VJP）の
                // 対応する二重ループは `for a in 0..axis_len {...}`
                // が 0 回実行されるため、ゼロ初期化された `reduced`
                // バッファがそのまま結果になる（codex-review P2
                // 是正・PR #1998）。`Var::narrow(axis, 0, 1)` は
                // `axis_len == 0` では範囲外（`NarrowOutOfBounds`）
                // になるため呼ばず、縮約後 shape を持つ明示的な
                // ゼロ定数へ直接差し替える。
                let mut reduced_shape = cur_shape.clone();
                reduced_shape[axis] = 1;
                cur = child.var_no_grad(&Tensor::zeros(&reduced_shape)?);
                cur_shape[axis] = 1;
                continue;
            }
            // `reduce_to_shape` の `for a in 0..axis_len { reduced[..] +=
            // data[src] }` を、添字昇順の `Var::narrow`＋`Var::add` の
            // 逐次連鎖として同じ順序で再現する（f32 逐次和・f64
            // アキュムレータなし）。
            let mut acc = cur.narrow(axis, 0, 1)?;
            for a in 1..axis_len {
                let slice = cur.narrow(axis, a, 1)?;
                acc = acc.add(&slice)?;
            }
            cur = acc;
            cur_shape[axis] = 1;
        }
    }
    if cur_shape == target_shape {
        Ok(cur)
    } else {
        // `cur_shape` は `padded_target`（先頭 `rank_diff` 個の 1 軸を
        // 含む）と一致しているはずであり、要素数は `target_shape` と
        // 同一のため `reshape` で先頭の 1 軸を落とせる。
        cur.contiguous()?.reshape(target_shape)
    }
}

/// `Op::Add` の bias パターンに限り `grad.rs::reduce_bias_grad`（f64
/// 相当のアキュムレータで行方向を縮約する）と数値方式を揃える
/// `reduce_to` の薄いラッパー（codex-review 指摘・PR #1998 是正）。
///
/// **背景**: `.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64`
/// アキュムレータで統一する」契約は `Op::LinearResident`／
/// `Op::LinearAct` の bias フォールバックに加え、2026-09-12 ユーザー
/// 承認により `Op::Add`（`nn::Linear` 既定 forward 経路 `matmul → add`
/// が経由する）の bias パターンへも横展開済み（`grad.rs::
/// reduce_bias_grad`）。本モジュール（子テープ上の `Var` 演算列）の
/// `reduce_to` は 1 階 `reduce_to_shape`（`f32` 逐次和・bias 以外の
/// 一般的な broadcast 縮約）を逐語再現する設計のため、`Op::Add` を
/// 無条件に `reduce_to` へ委譲すると bias パターンでのみ 1 階勾配
/// （`reduce_bias_grad` 経由）と符号レベルで乖離する（例:
/// `x: [3, 1]`・`b: [1]`・`c = [1e8, 1, -1e8]: [3, 1]` に対する
/// `loss = ((x + b) * c).sum()` で `b` の 1 階勾配は `1` だが `f32`
/// 逐次和では丸め誤差により `0` になる）。
///
/// **判定条件は `reduce_bias_grad` と同一**（`g_shape.len() == 2` かつ
/// `target_shape` が「軸 0 方向の縮約」を表す形状——末尾次元が `g` の
/// 列数と一致し、それより前の全次元が `1`）。条件を満たさない場合は
/// 従来どおり [`reduce_to`] へそのまま委譲し挙動を変えない（bias
/// パターン限定の横展開であり、汎用 `Op::Add`・`reduce_to` 本体は
/// 不変）。
///
/// **数値方式の再現方法**: `Var::sum(Some(0))` は `child.ops().sum()`
/// （`BackendOps` 実装依存。テスト用 `naive_ops()` は `f32` 逐次和・
/// 本番 `backend-cpu` は `f64` アキュムレータの `axis_reduce_sum` など
/// バックエンドごとに異なる）を経由するため、`Var::sum` へは委譲
/// **しない**（当初案は `naive_ops()` を使う統合テストで 1 階 VJP と
/// 再度乖離する回帰があった。実装時の実測で判明）。代わりに
/// `eval::reduce_bias_grad_rows`（`grad.rs::reduce_bias_grad` が呼ぶ
/// のと**同一のホスト関数**。行 `0..m` を列ごとに `f64` で逐次加算し
/// 最後に 1 回 `f32` へ downcast。`m == 1` の短絡〈符号付きゼロ保持〉
/// も同関数内で処理される）を直接呼んでホスト側で縮約後の値を計算し、
/// [`Tape::push_eager`]（`pub(crate)`）で `Op::Sum { input: g.id, dim:
/// Some(0) }` として子テープへ登録する。`child.ops()` の実装（テスト
/// 用 `naive_ops()` を含む）に依存せず 1 階 VJP と常に一致する。
///
/// `Op` タグを `Op::Sum { dim: Some(0) }` のまま維持することは、
/// `child.backward(..)`（さらなる微分。設計上 3 階微分は対象外だが、
/// 2 階勾配 `cg.grad(&b)` 自身を通常の `Var` として扱うために必要）に
/// 対する整合性を壊さない——`grad.rs::Op::Sum` の VJP（上流勾配を
/// `input` の shape へ broadcast するのみ）は前方値の精度に一切
/// 依存しないため、値を host 側で計算し直しても VJP の正しさは
/// 影響を受けない。
fn reduce_bias_grad_var<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    target_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let g_shape = g.shape();
    let is_row_axis_reduction = g_shape.len() == 2
        && target_shape.last() == Some(&g_shape[1])
        && target_shape[..target_shape.len().saturating_sub(1)]
            .iter()
            .all(|&d| d == 1);
    if !is_row_axis_reduction {
        return reduce_to(child, g, target_shape);
    }
    let g_val = {
        let nodes = child.nodes.borrow();
        materialize_fallible(&nodes, child.ops(), g.node_id())?.clone()
    };
    let reduced = crate::eval::reduce_bias_grad_rows(&g_val);
    let value = Tensor::new(reduced, &[g_shape[1]])?;
    let id = child.push_eager(
        Op::Sum {
            input: g.node_id(),
            dim: Some(0),
        },
        value,
    );
    Var::from_raw(child, id).reshape(target_shape)
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
