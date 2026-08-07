//! 動的テープ（Wengert list）本体。
//!
//! PoC-v2-2 が確定した「動的テープ式」記録方式
//! （`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`）を
//! `docs/public-api-design.md` §3.1 の productize 済み API 形状で実装する。
//! `Var`（`var.rs`）の演算メソッドが `Tape::push` を呼んで発生順に
//! `TapeNode` を追記し、`Tape::backward`（`backward.rs`・TASK-1.5c・
//! #18）はこの記録を逆走査して勾配を計算する（`Op` が入力 `NodeId` を
//! 保持するため、逆走査の各ノードから入力ノードを直接辿れる）。

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use tensor_core::Tensor;

/// テープの識別子。プロセス全体で単調増加するカウンタから発行する。
///
/// ポインタ等値（`ptr::eq`）ではなく専用 ID を用いる理由: スコープ末で
/// 破棄された `Tape` のメモリ領域は後続の `Tape::new()` に再利用され
/// うるため、ポインタ比較は別テープを同一と誤判定する（false positive）
/// 余地が残る（`docs/public-api-design.md` §3.1）。単調増加 ID は
/// プロセス生存中に衝突しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeId(u64);

/// プロセス全体で共有する `TapeId` 発行カウンタ。`Tape::new()` からのみ
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
    /// にはなるが、それ自体の入力ノードは持たない。
    Leaf,
    MatMul(NodeId, NodeId),
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Relu(NodeId),
    Exp(NodeId),
    Tanh(NodeId),
    /// シグモイド（`1 / (1 + exp(-x))`）。TASK-9.1b（#92）で
    /// `nn::activation::Sigmoid`（`nn/activation.rs`）から使う活性化
    /// プリミティブとして追加。数値安定形の forward は `eval.rs`
    /// （`eval::sigmoid`）、VJP は `grad.rs`（`out_value` 再利用方式）
    /// を参照。
    Sigmoid(NodeId),
    Sum {
        input: NodeId,
        dim: Option<usize>,
    },
    Max {
        input: NodeId,
        dim: Option<usize>,
    },
    /// 平均二乗誤差。`reduction` は #190（TASK-9.1c 相当・`nn::loss`）で
    /// mean/sum の両縮約に対応するため struct variant 化した
    /// （旧 `MseLoss(NodeId, NodeId)` は mean 固定だった）。
    /// `crate::var::Reduction` を再利用し、`grad.rs` の `vjp()` が
    /// `pred`/`target` への勾配スケールを縮約種別ごとに分岐する。
    MseLoss {
        pred: NodeId,
        target: NodeId,
        reduction: crate::var::Reduction,
    },
    /// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191・
    /// 親イシュー #189）。log-softmax → NLL を個別オペ合成せず、
    /// `MseLoss` と同じく forward/backward を解析形で閉じられる
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

/// テープ上の 1 ノード。演算種別（`Op`）と、その演算を順伝播で評価
/// した結果値を保持する（PoC-v2-2 の `TapeNode { op, value }` 構造を
/// 踏襲。f64→f32 化は `docs/public-api-design.md` §3.1 の確定事項）。
/// `op` は `grad.rs` の `vjp()` ディスパッチが `Tape::backward`
/// （`backward.rs`）の逆走査から読み出す。
#[derive(Debug)]
pub(crate) struct TapeNode {
    pub(crate) op: Op,
    pub(crate) value: Tensor<f32>,
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
/// **学習ループでの運用**: ノード列をクリアする `reset()`/`clear()`
/// 相当の API は本イシューでは提供しない。学習ループはステップごとに
/// 新しい `Tape::new()` を生成・破棄する運用を前提とする
/// （`docs/public-api-design.md` §3.1.1「未決事項として記録」）。
#[derive(Debug)]
pub struct Tape {
    pub(crate) id: TapeId,
    pub(crate) nodes: RefCell<Vec<TapeNode>>,
}

impl Default for Tape {
    fn default() -> Self {
        Tape::new()
    }
}

impl Tape {
    /// 新しいテープを生成する。`TapeId` は `NEXT_TAPE_ID` から新規発行
    /// されるため、同時に存在する複数の `Tape` 間で衝突しない。
    pub fn new() -> Tape {
        Tape {
            id: TapeId(NEXT_TAPE_ID.fetch_add(1, Ordering::Relaxed)),
            nodes: RefCell::new(Vec::new()),
        }
    }

    /// 非追跡の `Tensor<f32>` を、テープ上の葉ノード（`Op::Leaf`）として
    /// 登録する。以後この `Var` を起点とする演算はすべて `self` へ記録
    /// される。
    pub fn var(&self, tensor: &Tensor<f32>) -> crate::var::Var<'_> {
        let id = self.push(Op::Leaf, tensor.clone());
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

    /// `Var` の演算メソッド（`var.rs`）から呼ばれる、テープへの唯一の
    /// 追記経路。`op` の入力側 `Ref` を保持したまま呼ばれると
    /// `RefCell` の二重可変借用 panic になるため、呼び出し元は値計算
    /// を終えて借用（`Ref`）を閉じてから本関数を呼ぶ契約とする
    /// （`Var::value`/`to_tensor` のドキュメント参照）。
    pub(crate) fn push(&self, op: Op, value: Tensor<f32>) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeId(nodes.len());
        nodes.push(TapeNode { op, value });
        id
    }
}
