//! 動的テープ（Wengert list）本体。
//!
//! PoC-v2-2 が確定した「動的テープ式」記録方式
//! （`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`）を
//! `docs/public-api-design.md` §3.1 の productize 済み API 形状で実装する。
//! `Var`（`var.rs`）の演算メソッドが `Tape::push`／`push_lazy`（下記）を
//! 呼んで発生順に `TapeNode` を追記し、`Tape::backward`（`backward.rs`・
//! TASK-1.5c・#18）はこの記録を逆走査して勾配を計算する（`Op` が入力
//! `NodeId` を保持するため、逆走査の各ノードから入力ノードを直接辿れる）。
//!
//! **TASK-12.1d（#164）による拡張**: `Tape` は `ops: Box<dyn BackendOps +
//! Send>`（バックエンド実行の必須所有値）を保持し、`add`／`mul`／
//! `relu`／`exp`／`tanh`（elementwise 5 演算）は forward 値計算を
//! 即座に行わず `TapeNode::value`（`OnceCell`）を空のまま記録する
//! （遅延グラフの延長。`docs/fusion-graph-design.md` §3.5.1）。この
//! 遅延グラフを `matmul`／`sum`／`max`・`Tape::backward`・`Var::value`／
//! `to_tensor` が読み出す際に本ファイルの [`materialize_fallible`]／
//! [`materialize_non_fallible`] が実体化する（同 §3.5.2・§3.5.3）。
//! `FusionPlan::from_ops` + `BackendOps::run_fused`（`tensor-core`）を
//! 試み、`Unsupported` の場合は `ops` 自身の per-op メソッドへ逐次
//! フォールバックする。

use std::cell::{OnceCell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

use tensor_core::{BackendError, BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

use crate::error::AutodiffError;

/// テープの識別子。プロセス全体で単調増加するカウンタから発行する。
///
/// ポインタ等値（`ptr::eq`）ではなく専用 ID を用いる理由: スコープ末で
/// 破棄された `Tape` のメモリ領域は後続の `Tape::new(ops)` に再利用され
/// うるため、ポインタ比較は別テープを同一と誤判定する（false positive）
/// 余地が残る（`docs/public-api-design.md` §3.1）。単調増加 ID は
/// プロセス生存中に衝突しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeId(u64);

/// プロセス全体で共有する `TapeId` 発行カウンタ。`Tape::new(ops)` からのみ
/// インクリメントされる（`fetch_add` は複数スレッドから並行に `Tape` を
/// 生成しても一意性を保つ）。
static NEXT_TAPE_ID: AtomicU64 = AtomicU64::new(0);

/// テープ内ノードの識別子。`nodes: Vec<TapeNode>` への添字を直接表す
/// newtype（`docs/public-api-design.md` §3.1 は `pub type` 相当だが、
/// 生の `usize` と取り違えないよう newtype 化する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub(crate) usize);

/// テープへ記録される演算の種別。入力を `NodeId` で保持することで、
/// `Tape::backward`（`backward.rs`・#18・TASK-1.5c）が発生順とは逆順に
/// ノード列を走査し、各ノードの出力側勾配を入力ノードへ伝播できる。
///
/// クレート非公開: `Var`（`var.rs`）の演算メソッドのみが構築し、
/// 呼び出し側（autodiff クレートの利用者）には値と shape のみが
/// `Var::value`/`to_tensor` 経由で見える設計とする
/// （`docs/public-api-design.md` §3.2 の演算セットに 1:1 対応）。
///
/// 各 variant の入力 `NodeId`/`dim` フィールドは `grad.rs`（#17・
/// TASK-1.5b）の `vjp()` ディスパッチが読み出す（`op.clone()` して各
/// variant を分解し、入力側 `NodeId` へ勾配を割り当てる）。
/// `Tape::backward`（`backward.rs`・#18・TASK-1.5c）はノード列を逆走査
/// しながら `vjp()` を呼び、返った寄与を入力ノードへ蓄積する側。
///
/// **`Copy` を持たない理由**: `CrossEntropyLoss`（#191）の `targets` は
/// クラス添字（`Tensor<i32>`）を直接保持する非追跡ペイロードであり、
/// `Tensor<T>` は `Clone` のみ（`Copy` 不可）のため `Op` 全体も
/// `Clone` に留める（`grad.rs::vjp` は `op: &Op` を受け取り
/// `op.clone()` してから `match` する）。
#[derive(Debug, Clone)]
pub(crate) enum Op {
    /// `Tape::var()` が登録する非追跡入力（葉ノード）。逆伝播の起点
    /// にはなるが、それ自体の入力ノードは持たない。常に実体化済み
    /// （`push_eager`。TASK-12.1d・#164）。
    Leaf,
    /// 非 elementwise。常に実体化済み（`push_eager`）。
    MatMul(NodeId, NodeId),
    /// elementwise binary。遅延評価対象（`push_lazy`。TASK-12.1d・#164）。
    Add(NodeId, NodeId),
    /// elementwise binary。遅延評価対象。
    Mul(NodeId, NodeId),
    /// elementwise unary。遅延評価対象。
    Relu(NodeId),
    /// elementwise unary。遅延評価対象。
    Exp(NodeId),
    /// elementwise unary。遅延評価対象。
    Tanh(NodeId),
    /// シグモイド（`1 / (1 + exp(-x))`）。TASK-9.1b（#92）で
    /// `nn::activation::Sigmoid`（`nn/activation.rs`）から使う活性化
    /// プリミティブとして追加。`BackendOps` に対応メソッドがないため
    /// 融合対象外とし常に実体化済み（`push_eager`。TASK-12.1d・#164）。
    /// 数値安定形の forward は `eval.rs`（`eval::sigmoid`）、VJP は
    /// `grad.rs`（`out_value` 再利用方式）を参照。
    Sigmoid(NodeId),
    /// 非 elementwise。常に実体化済み。
    Sum { input: NodeId, dim: Option<usize> },
    /// 非 elementwise。常に実体化済み。
    Max { input: NodeId, dim: Option<usize> },
    /// 平均二乗誤差。`BackendOps` に対応メソッドがないため融合対象外
    /// とし常に実体化済み（`push_eager`）。`reduction` は #190
    /// （TASK-9.1c 相当・`nn::loss`）で mean/sum の両縮約に対応するため
    /// struct variant 化した（旧 `MseLoss(NodeId, NodeId)` は mean
    /// 固定だった）。`crate::var::Reduction` を再利用し、`grad.rs` の
    /// `vjp()` が `pred`/`target` への勾配スケールを縮約種別ごとに
    /// 分岐する。
    MseLoss {
        pred: NodeId,
        target: NodeId,
        reduction: crate::var::Reduction,
    },
    /// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191・
    /// 親イシュー #189）。`BackendOps` に対応メソッドがないため融合対象外
    /// とし常に実体化済み（`push_eager`）。log-softmax → NLL を個別オペ
    /// 合成せず、`MseLoss` と同じく forward/backward を解析形で閉じられる
    /// 1 個の融合オペとして実装する（実装計画 §3.1。PyTorch
    /// `F.cross_entropy` も内部で同等の融合実装）。
    ///
    /// `targets`（クラス添字）は非追跡データのため `Var`/`NodeId` を
    /// 持たず、`Op` payload に直接 `Tensor<i32>` を埋め込む（勾配は
    /// `logits` の 1 系統のみ定義され、`targets` 側には流れない。
    /// `grad.rs::vjp` の `CrossEntropyLoss` 分岐参照）。`reduction` は
    /// `MseLoss` と同じ `crate::var::Reduction`（#190 定義）を再利用する
    /// （`nn::loss` 側に重複定義は置かない）。
    CrossEntropyLoss {
        logits: NodeId,
        targets: Tensor<i32>,
        class_dim: usize,
        reduction: crate::var::Reduction,
    },
}

impl Op {
    /// elementwise 5 演算（融合の直接対象。`add`/`mul`/`relu`/`exp`/
    /// `tanh`）かどうか（TASK-12.1d・#164。`docs/fusion-graph-design.md`
    /// §3.5.1「遅延を許容する演算とその場で実体化する演算の切り分け」）。
    fn is_lazy_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add(..) | Op::Mul(..) | Op::Relu(..) | Op::Exp(..) | Op::Tanh(..)
        )
    }
}

/// テープ上の 1 ノード。演算種別（`Op`）・構造的に確定する出力 shape・
/// 実体化済みの値（空は「未実体化」を表す。TASK-12.1d・#164）を保持する
/// （PoC-v2-2 の `TapeNode { op, value }` 構造を踏襲。f64→f32 化は
/// `docs/public-api-design.md` §3.1 の確定事項）。`op` は `grad.rs` の
/// `vjp()` ディスパッチが `Tape::backward`（`backward.rs`）の逆走査から
/// 読み出す。
///
/// `shape` は実体化なしに算出できる（`add`/`mul` は broadcast、
/// `matmul` は行列積、`sum`/`max` は縮約、`relu`/`exp`/`tanh` は恒等の
/// shape 計算式であり、いずれも入力の `shape` フィールドだけを読めば
/// 求まる。`var.rs` の既存 shape 検証ロジックは今日すでに
/// `.value().shape()` ではなく形状情報のみを消費している）。
///
/// `value: OnceCell<Tensor<f32>>` は `add`/`mul`/`relu`/`exp`/`tanh`
/// （elementwise 5 演算）に限り空のまま記録される（遅延グラフの延長。
/// `docs/fusion-graph-design.md` §3.5.1）。`matmul`/`sum`/`max`・
/// `Op::Leaf`・`Sigmoid`/`MseLoss`/`CrossEntropyLoss`（`BackendOps` に
/// 対応メソッドがない演算）は常に返る前に `OnceCell::from(...)` で
/// 即座に埋める。`OnceCell::get_or_init`／`set` はいずれも `&self`
/// （共有参照）で呼べるため、`RefCell<Vec<TapeNode>>` の `borrow()`
/// （共有借用）だけで埋められる。
#[derive(Debug)]
pub(crate) struct TapeNode {
    pub(crate) op: Op,
    pub(crate) shape: Vec<usize>,
    pub(crate) value: OnceCell<Tensor<f32>>,
}

/// 演算を記録する Wengert list。`Var`（`var.rs`）上の演算のみがここに
/// 記録される。`tensor_core::Tensor<f32>` に対する演算はテープを構築
/// しないため、追跡の有無は型（`Tensor<f32>` か `Var` か）で表現される
/// （`docs/public-api-design.md` §3.1「型分離方式」）。
///
/// ノード列は `RefCell` で包み、`Var` からの内部可変性による追記を
/// 可能にする。PoC-v2-2 確定 API（`Tape::matmul(&mut self, ...)`）は
/// `&mut Tape` を要求するが、本実装では `Var` 単体を式中で連鎖させたい
/// （`a.matmul(&b)?.relu()` 等）ため、`RefCell` + 共有参照 `&'t Tape`
/// 方式（`docs/public-api-design.md` §3.1 が productize 時に許容する
/// 選択肢）を採用する。
///
/// **`ops` フィールド（TASK-12.1d・#164。必須所有値）**: `Tape::new(ops)`
/// で構築したどの `Tape` でも常に埋まっている（`Option` を経由しない。
/// `docs/fusion-graph-design.md` §1「`None` に相当する値がそもそも
/// 存在しない」）。`Box<dyn BackendOps + Send>`（`Send` 境界あり）と
/// することで `Tape` の全フィールド（`id`・`nodes`・`ops`）が `Send` を
/// 満たし、`Tape: Send` の自動導出は後退しない（同文書 §3.4）。
/// `Sync` はいずれのフィールドにも要求しない（`RefCell`／`OnceCell` に
/// より元々 `!Sync`）。`BackendOps` は `Debug` をスーパートレイトに
/// 持たないため `#[derive(Debug)]` は撤去し、`ops` の中身を表示しない
/// 手書き `impl fmt::Debug` へ置き換える（下記）。
///
/// **学習ループでの運用**: ノード列をクリアする `reset()`/`clear()`
/// 相当の API は本イシューでは提供しない。学習ループはステップごとに
/// 新しい `Tape::new(ops)` を生成・破棄する運用を前提とする
/// （`docs/public-api-design.md` §3.1.1「未決事項として記録」）。
pub struct Tape {
    pub(crate) id: TapeId,
    pub(crate) nodes: RefCell<Vec<TapeNode>>,
    pub(crate) ops: Box<dyn BackendOps + Send>,
}

impl std::fmt::Debug for Tape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ops`（`Box<dyn BackendOps + Send>`）の中身は `Debug` を実装
        // しない（`BackendOps` は `Debug` をスーパートレイトに持たない。
        // 外部実装を破壊しないための既存方針）ため表示しない。`Tape:
        // Debug` という公開契約自体は変更しない（実装手段が `derive`
        // から手書きへ変わるのみ。`docs/fusion-graph-design.md` §3.4）。
        f.debug_struct("Tape").finish_non_exhaustive()
    }
}

impl Tape {
    /// 新しいテープを生成する（TASK-12.1d・#164 の破壊的変更: 無引数
    /// `Tape::new() -> Tape`・`impl Default for Tape` は削除され、本
    /// シグネチャへ置き換わる）。`ops` はこのテープ上のすべての
    /// バックエンド実行（融合実行・per-op フォールバック・`matmul`/
    /// `sum`/`max` の実行）に使われる必須所有値であり、`facade`
    /// （TASK-9.3。未実装）の composition root が解決した具体
    /// `BackendOps` 実装、明示指定した `Device` の結線結果、または
    /// テスト用フィクスチャのいずれかを渡す（`docs/
    /// fusion-graph-design.md` §1・§3.4）。`TapeId` は `NEXT_TAPE_ID`
    /// から新規発行されるため、同時に存在する複数の `Tape` 間で衝突
    /// しない。
    pub fn new(ops: Box<dyn BackendOps + Send>) -> Tape {
        Tape {
            id: TapeId(NEXT_TAPE_ID.fetch_add(1, Ordering::Relaxed)),
            nodes: RefCell::new(Vec::new()),
            ops,
        }
    }

    /// 非追跡の `Tensor<f32>` を、テープ上の葉ノード（`Op::Leaf`）として
    /// 登録する。以後この `Var` を起点とする演算はすべて `self` へ記録
    /// される。葉ノードは常に実体化済み（`push_eager`）。
    pub fn var(&self, tensor: &Tensor<f32>) -> crate::var::Var<'_> {
        let id = self.push_eager(Op::Leaf, tensor.clone());
        crate::var::Var::from_raw(self, id)
    }

    /// 現在記録済みのノード数を返す。受け入れ条件（forward 実行時に
    /// テープへ演算が記録される）の検証・デバッグ用途の最小限のアクセサ
    /// （`tests/tape_recording.rs`）。
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// テープにノードが 1 つも記録されていないか判定する。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// この `Tape` が保持する `ops`（バックエンド実行の必須所有値）への
    /// 借用。`backward.rs`／`grad.rs` が `materialize_fallible` を呼ぶ際
    /// に使う。
    pub(crate) fn ops(&self) -> &dyn BackendOps {
        self.ops.as_ref()
    }

    /// **非 elementwise・常に実体化済み**のノードを追記する（`matmul`/
    /// `sum`/`max`・`Op::Leaf`・`Sigmoid`/`MseLoss`/`CrossEntropyLoss`
    /// から呼ばれる。TASK-12.1d・#164 で `push` から改称）。`op` の入力側
    /// `Ref` を保持したまま呼ばれると `RefCell` の二重可変借用 panic に
    /// なるため、呼び出し元は値計算を終えて借用（`Ref`）を閉じてから
    /// 本関数を呼ぶ契約とする（`Var::value`/`to_tensor` のドキュメント
    /// 参照）。
    pub(crate) fn push_eager(&self, op: Op, value: Tensor<f32>) -> NodeId {
        let shape = value.shape().to_vec();
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeId(nodes.len());
        nodes.push(TapeNode {
            op,
            shape,
            value: OnceCell::from(value),
        });
        id
    }

    /// **elementwise 5 演算（`add`/`mul`/`relu`/`exp`/`tanh`）専用**の
    /// 遅延追記経路（TASK-12.1d・#164）。`value` を空の `OnceCell` の
    /// まま記録し、自身の出力を実体化せずに返す（4〜6 段連鎖を実現する
    /// 主要因。`docs/fusion-graph-design.md` §3.5.1）。`shape` は構造的に
    /// 確定済みの出力 shape（呼び出し元が `broadcast_shape` 等で計算
    /// 済み）を渡す。
    pub(crate) fn push_lazy(&self, op: Op, shape: Vec<usize>) -> NodeId {
        debug_assert!(op.is_lazy_elementwise());
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeId(nodes.len());
        nodes.push(TapeNode {
            op,
            shape,
            value: OnceCell::new(),
        });
        id
    }
}

// =====================================================================
// materialize ヘルパー（TASK-12.1d・#164。`docs/fusion-graph-design.md`
// §3.5.2・§3.5.3）。
// =====================================================================

/// `id` を起点とする遅延部分グラフ（`add`/`mul`/`relu`/`exp`/`tanh` の
/// 未実体化ノードの連結成分）を後方走査し、`FusionPlan::from_ops` へ
/// 渡す `FusedOpKind` 列・葉ノード実体・出力 shape を組み立てる
/// （`docs/fusion-graph-design.md` §3.5.2 手順 1・2 の共通部分。層 1・
/// 層 2 のいずれからも呼ばれる）。
///
/// `id` 自身が既に実体化済み（`OnceCell::get().is_some()`）の場合は
/// 空プラン（`ops` 空・`leaf_count == 0`）を返す—呼び出し元
/// （[`materialize_fallible`]／[`materialize_non_fallible`]）はこの場合
/// `run_fused`/フォールバックを試みる前に早期リターンする。
///
/// **NodeId の単調増加が非巡回性を保証する**（`docs/
/// fusion-graph-design.md` §3.5.1「走査順が既に発生順トポロジカル順で
/// あること」）: `id.0` から `0` へ向けた降順の単純な線形スキャンだけで
/// 「参照済みノードをそれより手前で必ず処理し終えている」ことが保証
/// される。**連鎖長上限（設計書 §3.2 (d)・§3.5.4）は本実装では適用しない
/// （scope reduction。#164 実装ノート）**: `push_lazy` 時点でのその場
/// 実体化による上限適用は追加の状態追跡を要し、本イシュー時点では
/// `run_fused` が常に `Unsupported`（#163 未マージ）のため機能上の差異が
/// ない。融合カーネル生成（#163）が実際に結線される際に上限適用を
/// 追加することを想定し、`docs/spec/05-tasks.md` の枠外
/// （out-of-scope-tracking.md）として PR 本文に記録する。
fn build_lazy_plan(
    nodes: &[TapeNode],
    id: NodeId,
) -> Result<(FusionPlan, Vec<Tensor<f32>>, NodeId), BackendError> {
    use std::collections::HashMap;

    let mut reachable: HashMap<usize, ()> = HashMap::new();
    reachable.insert(id.0, ());
    let mut interior: Vec<usize> = Vec::new();
    let mut leaf_order: Vec<usize> = Vec::new();
    let mut leaf_index_of: HashMap<usize, usize> = HashMap::new();

    for cur in (0..=id.0).rev() {
        if !reachable.contains_key(&cur) {
            continue;
        }
        let node = &nodes[cur];
        if node.value.get().is_some() {
            // 実体化済み: このノードで走査を打ち切り、葉として扱う
            // （`Op::Leaf`・`matmul`/`sum`/`max`・`Sigmoid` 等の非
            // elementwise、または以前に別経路から実体化済みの
            // elementwise ノードのいずれも該当しうる）。
            leaf_index_of.entry(cur).or_insert_with(|| {
                let idx = leaf_order.len();
                leaf_order.push(cur);
                idx
            });
            continue;
        }
        match &node.op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                interior.push(cur);
                reachable.insert(a.0, ());
                reachable.insert(b.0, ());
            }
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => {
                interior.push(cur);
                reachable.insert(a.0, ());
            }
            // 非 elementwise は `push_eager` により常に実体化済みの
            // はずであり、この分岐には構造上到達しない。それでも
            // 本番経路 panic 禁止のため、到達した場合は安全側で葉
            // として扱う（`OnceCell` 不変条件違反の防御的フォール）。
            _ => {
                leaf_index_of.entry(cur).or_insert_with(|| {
                    let idx = leaf_order.len();
                    leaf_order.push(cur);
                    idx
                });
            }
        }
    }
    interior.reverse(); // 発生順（トポロジカル順）へ揃える

    let leaf_count = leaf_order.len();
    let node_index = |n: usize, interior: &[usize]| -> usize {
        if let Some(&li) = leaf_index_of.get(&n) {
            li
        } else {
            // interior 内での位置（昇順走査中は必ず存在する契約）。
            let pos = interior.iter().position(|&x| x == n).unwrap_or(0);
            leaf_count + pos
        }
    };

    // `FusionPlan::from_ops`（`tensor-core::fusion::plan`）は `ops[0..
    // leaf_count)` に `FusedOpKind::Input { leaf_index }` が発生順（`0..
    // leaf_count` を一度ずつ）で並んでいることを構築時に検証する契約
    // （`plan.rs` の `from_ops` ドキュメント「検証」節）。`node_index` は
    // 葉を `0..leaf_count` の位置として扱うため、`ops` 自身にもその位置に
    // 対応する `Input` エントリを明示的に積む必要がある——省略すると
    // `ops[0]`（先頭の interior エントリ）が「まだ存在しない位置」を
    // 参照したとみなされ `FusionPlanError::IndexOutOfRange` で拒否される。
    let mut ops: Vec<FusedOpKind> = Vec::with_capacity(leaf_count + interior.len());
    for leaf_index in 0..leaf_count {
        ops.push(FusedOpKind::Input { leaf_index });
    }
    for &cur in &interior {
        let kind = match &nodes[cur].op {
            Op::Add(a, b) => FusedOpKind::Add {
                lhs: node_index(a.0, &interior),
                rhs: node_index(b.0, &interior),
            },
            Op::Mul(a, b) => FusedOpKind::Mul {
                lhs: node_index(a.0, &interior),
                rhs: node_index(b.0, &interior),
            },
            Op::Relu(a) => FusedOpKind::Relu {
                input: node_index(a.0, &interior),
            },
            Op::Exp(a) => FusedOpKind::Exp {
                input: node_index(a.0, &interior),
            },
            Op::Tanh(a) => FusedOpKind::Tanh {
                input: node_index(a.0, &interior),
            },
            _ => unreachable_op_kind(),
        };
        ops.push(kind);
    }

    let output_shape = nodes[id.0].shape.clone();
    let leaves: Vec<Tensor<f32>> = leaf_order
        .iter()
        .map(|&n| lazy_leaf_value(&nodes[n]))
        .collect();

    let plan = FusionPlan::from_ops(ops, output_shape, DType::F32, leaf_count)?;
    Ok((plan, leaves, id))
}

/// `node.shape` から全要素 `0.0` のテンソルを構築する安全側フォール
/// バック（PR #403 codex-review P1 是正）。`Tensor::from_shape_fill` は
/// `tensor-core::checked_numel` による要素数積オーバーフロー検査を
/// 経る `Result` 返り値になったため（`tensor.rs` の該当コメント参照）、
/// 本ファイルの契約違反時フォールバック 3 箇所（[`lazy_leaf_value`]・
/// [`fallback_per_op`]・`eval_fallback`）で共有する。渡す `shape` は
/// いずれも既存の `TapeNode` から読んだ値（過去に妥当な shape として
/// 構築済み）であり実運用でオーバーフローは起こらないが、`Err` 分岐は
/// `debug_assert!` で検知しつつ [`tensor_core::Tensor::scalar`]（真に
/// infallible）へ吸収し、本番経路 panic 禁止方針
/// （`.claude/rules/coding-rust.md`）を保つ。
fn safe_zeros(shape: &[usize]) -> Tensor<f32> {
    Tensor::from_shape_fill(shape, |_| 0.0).unwrap_or_else(|_| {
        debug_assert!(
            false,
            "safe_zeros: shape の要素数積がオーバーフローした（契約違反）"
        );
        Tensor::scalar(0.0)
    })
}

/// `interior` 収集ロジックが `_` 分岐（非 elementwise が未実体化のまま
/// 到達する契約違反）へ到達した場合の安全側フォールバック値。呼び出し元
/// （`build_lazy_plan`）はこの分岐へ実際には到達しない設計だが、
/// `match` の網羅性を保つために用意する（`.claude/rules/coding-rust.md`
/// 本番経路 panic 禁止方針。`unreachable!()` を直接書かない）。
fn unreachable_op_kind() -> FusedOpKind {
    debug_assert!(
        false,
        "build_lazy_plan: 非 elementwise ノードが未実体化のまま interior へ混入した（契約違反）"
    );
    FusedOpKind::Add { lhs: 0, rhs: 0 }
}

/// 実体化済みノードの値を読む（`build_lazy_plan` の葉収集専用）。
/// `OnceCell::get().is_some()` を確認済みの呼び出し元からのみ呼ばれる
/// ため理論上必ず `Some` を返すが、`unwrap()`/`expect()` は使わず、
/// `None` の場合は `shape` から構築した安全側フォールバック（全要素
/// `0.0`）を返す（本番経路 panic 禁止方針）。
fn lazy_leaf_value(node: &TapeNode) -> Tensor<f32> {
    match node.value.get() {
        Some(t) => t.clone(),
        None => {
            debug_assert!(
                false,
                "lazy_leaf_value: 実体化済みのはずのノードが未実体化だった（契約違反）"
            );
            safe_zeros(&node.shape)
        }
    }
}

/// per-op フォールバック（`run_fused` が `Unsupported` を返した場合に
/// 遅延部分グラフを `ops` の既存 per-op メソッドで逐次再計算する。
/// `docs/fusion-graph-design.md` §3.4「fail-safe 方針」・§3.5.2 手順 3）。
/// `build_lazy_plan` と同じ後方走査を再実行し、今度は実際に値を計算する
/// （プランと違い、この経路は shape 検証済みの `ops` 呼び出しの結果を
/// そのまま使うため `FusionPlan` を経由しない）。
fn fallback_per_op(
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
) -> Result<Tensor<f32>, BackendError> {
    use std::collections::HashMap;

    // `build_lazy_plan` と同じ走査で interior（未実体化 elementwise の
    // 連結成分。発生順）を求める。
    let mut reachable: HashMap<usize, ()> = HashMap::new();
    reachable.insert(id.0, ());
    let mut interior: Vec<usize> = Vec::new();
    for cur in (0..=id.0).rev() {
        if !reachable.contains_key(&cur) {
            continue;
        }
        let node = &nodes[cur];
        if node.value.get().is_some() {
            continue;
        }
        match &node.op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                interior.push(cur);
                reachable.insert(a.0, ());
                reachable.insert(b.0, ());
            }
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => {
                interior.push(cur);
                reachable.insert(a.0, ());
            }
            _ => {}
        }
    }
    interior.reverse();

    let mut computed: HashMap<usize, Tensor<f32>> = HashMap::new();
    let value_of = |n: usize, computed: &HashMap<usize, Tensor<f32>>| -> Tensor<f32> {
        if let Some(t) = nodes[n].value.get() {
            t.clone()
        } else {
            computed
                .get(&n)
                .cloned()
                .unwrap_or_else(|| lazy_leaf_value(&nodes[n]))
        }
    };
    for &cur in &interior {
        let result = match &nodes[cur].op {
            Op::Add(a, b) => ops.add(&value_of(a.0, &computed), &value_of(b.0, &computed))?,
            Op::Mul(a, b) => ops.mul(&value_of(a.0, &computed), &value_of(b.0, &computed))?,
            Op::Relu(a) => ops.relu(&value_of(a.0, &computed))?,
            Op::Exp(a) => ops.exp(&value_of(a.0, &computed))?,
            Op::Tanh(a) => ops.tanh(&value_of(a.0, &computed))?,
            _ => continue,
        };
        computed.insert(cur, result);
    }

    computed.get(&id.0).cloned().map(Ok).unwrap_or_else(|| {
        // interior が空（`id` が既に実体化済みだった呼び出し）の場合は
        // 呼び出し元が別途処理する契約のため、通常この分岐には来ない。
        Ok(nodes[id.0]
            .value
            .get()
            .cloned()
            .unwrap_or_else(|| safe_zeros(&nodes[id.0].shape)))
    })
}

/// 層 1（fallible 境界）: 後続の fallible `Var` 演算（`matmul`/`sum`/
/// `max`）・`Tape::backward` が forward 記録済みの未実体化ノードを
/// 読み出す際に**必ず**使用する（`docs/fusion-graph-design.md` §3.5.2）。
/// `Var::value`（層 2）は本関数を呼ばない。
///
/// `run_fused` の失敗のうち `BackendError::Unsupported`（能力不足）
/// **以外**は eager フォールバックで吸収せず型付きエラーのまま呼び出し
/// 元へ返す。`Unsupported` の場合のみ `ops` の per-op メソッドへ逐次
/// フォールバックし、それも失敗すれば `?` で伝播する（§3.5.2 手順 3・4。
/// `eval.rs` への最終手段フォールバックは層 2 限定であり層 1 では
/// 使わない）。
///
/// 実体化が完了したら実体化を要求した対象ノード自身の `OnceCell` にのみ
/// `set()` する（連鎖の途中ノードの `OnceCell` は空のまま残す。融合の
/// 利得〈中間 `Tensor` 実体化の除去〉を保つため）。`OnceCell::set` の
/// `Err`（fan-out に伴う二重到達）は通常分岐として扱い、`get()` で
/// 読み直した既存値を正として使う（`panic!`／`unwrap()`／`expect()` は
/// いずれも使わない）。
pub(crate) fn materialize_fallible<'a>(
    nodes: &'a [TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
) -> Result<&'a Tensor<f32>, AutodiffError> {
    if let Some(v) = nodes[id.0].value.get() {
        return Ok(v);
    }

    let (plan, leaves, _root) = build_lazy_plan(nodes, id).map_err(AutodiffError::Backend)?;
    let leaf_refs: Vec<&Tensor<f32>> = leaves.iter().collect();
    let computed = match ops.run_fused(&plan, &leaf_refs) {
        Ok(t) => t,
        Err(BackendError::Unsupported(_)) => {
            fallback_per_op(nodes, ops, id).map_err(AutodiffError::Backend)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };

    match nodes[id.0].value.set(computed) {
        Ok(()) => {}
        Err(_rejected) => { /* fan-out 二重到達: 既存値を正として使う */ }
    }
    nodes[id.0].value.get().ok_or_else(|| {
        AutodiffError::Backward(
            "materialize_fallible: OnceCell 不変条件違反（set 直後に get が None を返した）".into(),
        )
    })
}

/// 層 2（非 fallible 境界）: `Var::value`／`Var::to_tensor`・
/// `Gradients::get` が使う（`docs/fusion-graph-design.md` §3.5.3）。
/// `matmul`/`sum`/`max`・`Tape::backward` は本関数を呼ばない
/// （[`materialize_fallible`] を使う）。
///
/// `ops.run_fused` が `Ok` 以外を返した場合はエラー種別を区別せず、
/// まず `ops` 自身の per-op メソッドへフォールバックし、それも失敗した
/// 場合に限り `autodiff` 自身の `eval.rs`（#164 で非 panic 構造へ改修
/// 済み）を最終手段として用いて再計算する。必ず `Tensor<f32>` を返し、
/// `panic!` も `Err` も返さない。
pub(crate) fn materialize_non_fallible<'a>(
    nodes: &'a [TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
) -> &'a Tensor<f32> {
    // `OnceCell::get_or_init`（設計書 §3.5.3 のスケッチどおり）は非
    // fallible なクロージャを取り `&self` のみで完結するため、層 1
    // （`materialize_fallible`）のような手動 set/get の 2 段階処理・
    // fan-out 二重到達の分岐が不要になる（`get_or_init` 自体が
    // 「既に初期化済みならそれを返す」を保証する）。クロージャ内は
    // 融合加速 → per-op フォールバック → `eval.rs` 最終手段の順に必ず
    // 値を返すため、この経路は構造的に失敗しない
    // （`docs/fusion-graph-design.md` §3.5.3 (i)〜(iii)）。
    nodes[id.0].value.get_or_init(|| {
        build_lazy_plan(nodes, id)
            .ok()
            .and_then(|(plan, leaves, _root)| {
                let leaf_refs: Vec<&Tensor<f32>> = leaves.iter().collect();
                ops.run_fused(&plan, &leaf_refs).ok()
            })
            .or_else(|| fallback_per_op(nodes, ops, id).ok())
            .unwrap_or_else(|| eval_fallback(nodes, id))
    })
}

/// `materialize_non_fallible` の最終手段: `ops` の per-op メソッドも
/// 失敗した場合、`autodiff` 自身の `eval.rs`（#164 で非 panic 構造へ
/// 改修済み）を用いてトポロジカル順に逐次再計算する（`docs/
/// fusion-graph-design.md` §3.5.3。層 2 限定の例外）。
fn eval_fallback(nodes: &[TapeNode], id: NodeId) -> Tensor<f32> {
    use std::collections::HashMap;

    let mut reachable: HashMap<usize, ()> = HashMap::new();
    reachable.insert(id.0, ());
    let mut interior: Vec<usize> = Vec::new();
    for cur in (0..=id.0).rev() {
        if !reachable.contains_key(&cur) {
            continue;
        }
        let node = &nodes[cur];
        if node.value.get().is_some() {
            continue;
        }
        match &node.op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                interior.push(cur);
                reachable.insert(a.0, ());
                reachable.insert(b.0, ());
            }
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => {
                interior.push(cur);
                reachable.insert(a.0, ());
            }
            _ => {}
        }
    }
    interior.reverse();

    let mut computed: HashMap<usize, Tensor<f32>> = HashMap::new();
    let value_of = |n: usize, computed: &HashMap<usize, Tensor<f32>>| -> Tensor<f32> {
        if let Some(t) = nodes[n].value.get() {
            t.clone()
        } else {
            computed
                .get(&n)
                .cloned()
                .unwrap_or_else(|| lazy_leaf_value(&nodes[n]))
        }
    };
    for &cur in &interior {
        let result = match &nodes[cur].op {
            Op::Add(a, b) => crate::eval::add(&value_of(a.0, &computed), &value_of(b.0, &computed)),
            Op::Mul(a, b) => crate::eval::mul(&value_of(a.0, &computed), &value_of(b.0, &computed)),
            Op::Relu(a) => crate::eval::relu(&value_of(a.0, &computed)),
            Op::Exp(a) => crate::eval::exp(&value_of(a.0, &computed)),
            Op::Tanh(a) => crate::eval::tanh(&value_of(a.0, &computed)),
            _ => continue,
        };
        computed.insert(cur, result);
    }

    computed
        .get(&id.0)
        .cloned()
        .unwrap_or_else(|| safe_zeros(&nodes[id.0].shape))
}
