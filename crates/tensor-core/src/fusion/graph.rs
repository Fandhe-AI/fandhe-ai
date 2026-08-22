//! 融合グラフの中間表現（IR）本体（TASK-12.1a・#161 設計の実装。#162）。
//!
//! `docs/fusion-graph-design.md` §2 の型スケッチをそのまま実装する。
//! `fandhe_ai_autodiff::tape`（`crates/autodiff/src/tape.rs:35` 付近の
//! `NodeId`／`Op`／`Tape`）と同型の「発生順 `Vec` 追記の DAG、入力は
//! 常に自ノードより小さい ID」という構成を踏襲するが、本モジュールは
//! `autodiff` に依存せず `tensor-core` 内に閉じる（§2.5「配置」の決定）。
//!
//! [`FusionGraph::push`] は構築時に検証を先行させる
//! （OWASP A03 観点。`.claude/rules/security.md`・設計書 §5）:
//! 入力 `FusionNodeId` が既存ノードの範囲内にあること、elementwise
//! binary（`Add`／`Mul`）のオペランド shape が一致すること、さらに
//! elementwise（unary／binary いずれも）の出力 shape（呼び出し側が渡す
//! `meta.shape`）が入力 shape と恒等であること（設計書 §2.3「elementwise
//! の出力 shape は入力の shape フィールドだけを読めば求まる」恒等計算の
//! 前提）を、ノードを追記する前に検査し、違反時は [`FusionGraphError`]
//! で拒否する。呼び出し側（#163／#164）が誤った出力 shape を渡すバグを
//! 構築時に検出する検証境界としての役割を担う。

use crate::dispatch::DType;
use crate::error::ShapeError;

/// 融合グラフのノード種別（設計書 §2.1・イシュー #586 で境界を再定義・
/// #588 で `Sub`／`Div`／`Broadcast` を追加）。
///
/// `BackendOps`（`crate::backend_ops::BackendOps`）の各メソッドと 1:1
/// 対応させる: elementwise binary（`add`／`mul`／`sub`／`div`）・unary
/// （`relu`／`exp`／`tanh`／`rsqrt`）に加え、reduction（`sum`／`max`）も
/// **セグメント軸（`dim`）が一致する限り融合セグメント内に組み込める**
/// （#586。RMSNorm／softmax のような `reduce → elementwise` 連鎖を単一
/// セグメントで扱う土台。`detect.rs` の走査がセグメント軸の確定・一致
/// 判定を担う）。`Broadcast` は縮約済みテンソルを元の行 shape へ論理拡張
/// するノードで、reduction の出力を再びセグメント内の elementwise へ
/// 合流させる（#588。「行方向 reduction → 派生スカラー → 同一行へ
/// broadcast 適用」という RMSNorm／softmax 型の 2 パス構造を表現する
/// ために必須）。`gemm` のみが常に融合境界ノード（融合セグメントに
/// 組み込まず、そこで走査を打ち切る印として扱う。`detect.rs` 参照）で
/// あり続ける。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FusionOp {
    /// リーフノード（グラフへの外部入力テンソル）。
    Input,
    // elementwise binary（backend_ops.rs の `add`/`mul` に対応）
    Add(FusionNodeId, FusionNodeId),
    Mul(FusionNodeId, FusionNodeId),
    /// 減算（`lhs - rhs`）。softmax の `x - max(x)` を表現するために
    /// #588 で追加。`Add`／`Mul` と同じオペランド・出力 shape 恒等契約
    /// （`push` の検証枝を共有する）。
    Sub(FusionNodeId, FusionNodeId),
    /// 除算（`lhs / rhs`）。softmax の `exp(..) / sum(..)` を表現するために
    /// #588 で追加。`Add`／`Mul` と同じ検証契約に相乗りする。
    Div(FusionNodeId, FusionNodeId),
    // elementwise unary（`relu`/`exp`/`tanh`/`rsqrt` に対応）
    Relu(FusionNodeId),
    Exp(FusionNodeId),
    Tanh(FusionNodeId),
    /// `1/sqrt(x)`。RMSNorm 融合（#586）の構成要素として追加。elementwise
    /// unary と同じ恒等 shape 契約（`push` の出力 shape 検証・
    /// `is_elementwise()`）を持つ。
    Rsqrt(FusionNodeId),
    // 常に融合境界ノード（融合しない。到達時に実体化する。設計書 §3 参照）
    Gemm(FusionNodeId, FusionNodeId),
    /// reduction（縮約）。`dim: None` は全軸縮約、`dim: Some(axis)` は
    /// 指定軸のみの縮約（`backend-cpu::reduction` の契約と同型）。
    /// #586 以降、セグメント軸が一致する限り融合セグメントへ組み込める
    /// （`detect.rs` 参照。組み込まれない場合は従来どおり境界ノード）。
    Sum {
        input: FusionNodeId,
        dim: Option<usize>,
    },
    /// [`FusionOp::Sum`] と同じ軸一致融合ルールに従う reduction。
    Max {
        input: FusionNodeId,
        dim: Option<usize>,
    },
    /// 縮約済みテンソル `input` をセグメント軸に沿って元の行 shape へ
    /// 論理拡張する（#588。値の複製をレジスタ内で行う契約であり、
    /// strided view ではない）。`dim: None` は `input` が rank 0（`[]`）
    /// で任意 shape への全 broadcast、`dim: Some(axis)` は `input` の
    /// shape に軸 `axis` を再挿入した shape が出力になる契約
    /// （`FusionOp::Sum`／`Max` の逆演算に相当。`push` の構築時検証は
    /// `crate::ops_shape::reduce_out_shape` を再利用し二重定義しない）。
    Broadcast {
        input: FusionNodeId,
        dim: Option<usize>,
    },
}

impl FusionOp {
    /// このノードが直接参照する入力ノード ID 一覧（発生順）。
    ///
    /// `detect.rs` の後方走査が、ノード種別に応じた分岐を書かずに
    /// 「このノードの入力を辿る」処理を共通化するために使う。
    /// `Input` は入力を持たない葉ノードのため空を返す。
    pub(crate) fn inputs(&self) -> Vec<FusionNodeId> {
        match self {
            FusionOp::Input => Vec::new(),
            FusionOp::Add(a, b)
            | FusionOp::Mul(a, b)
            | FusionOp::Sub(a, b)
            | FusionOp::Div(a, b)
            | FusionOp::Gemm(a, b) => vec![*a, *b],
            FusionOp::Relu(a) | FusionOp::Exp(a) | FusionOp::Tanh(a) | FusionOp::Rsqrt(a) => {
                vec![*a]
            }
            FusionOp::Sum { input, .. }
            | FusionOp::Max { input, .. }
            | FusionOp::Broadcast { input, .. } => vec![*input],
        }
    }

    /// elementwise 8 演算（融合対象・恒等 shape 契約を持つ）かどうか
    /// （設計書 §2.1「融合の直接対象」・#586 で `Rsqrt` を追加・#588 で
    /// `Sub`／`Div` を追加）。`detect.rs` の連鎖検出が「常に境界」の
    /// `Gemm`・reduction（[`is_reduction`](Self::is_reduction)）・
    /// broadcast（[`is_broadcast`](Self::is_broadcast)）・外部入力
    /// （`Input`）と区別するために使う。**`Broadcast` は恒等 shape 契約を
    /// 持たない**（出力 shape が入力より rank が高い）ため、ここには
    /// 含めない（#588 実装計画 §3.1）。
    pub(crate) fn is_elementwise(&self) -> bool {
        matches!(
            self,
            FusionOp::Add(..)
                | FusionOp::Mul(..)
                | FusionOp::Sub(..)
                | FusionOp::Div(..)
                | FusionOp::Relu(..)
                | FusionOp::Exp(..)
                | FusionOp::Tanh(..)
                | FusionOp::Rsqrt(..)
        )
    }

    /// reduction（`Sum`／`Max`）かどうか（#586）。`detect.rs` がセグメント
    /// 軸一致判定の対象ノードを識別するために使う。
    pub(crate) fn is_reduction(&self) -> bool {
        matches!(self, FusionOp::Sum { .. } | FusionOp::Max { .. })
    }

    /// `Broadcast`（縮約済みテンソルの行方向拡張）かどうか（#588）。
    /// `detect.rs` が reduction と同じセグメント軸一致判定の対象として
    /// 扱うために使う（`is_reduction` と対）。
    pub(crate) fn is_broadcast(&self) -> bool {
        matches!(self, FusionOp::Broadcast { .. })
    }

    /// 融合セグメントへ組み込みうる演算（elementwise ∪ reduction ∪
    /// broadcast）かどうか（#586・#588 で拡張。`Gemm`／`Input` のみが
    /// 常に組み込み対象外）。現状 `detect.rs` は reduction／broadcast の
    /// 組み込み可否をセグメント軸一致で個別に判定するため本メソッドを
    /// 直接は使わないが、`is_elementwise`／`is_reduction`／`is_broadcast`
    /// の対概念として設計書の境界再定義を型で表現する（将来の呼び出し元
    /// 向けの安定 API）。
    pub(crate) fn is_fusable(&self) -> bool {
        self.is_elementwise() || self.is_reduction() || self.is_broadcast()
    }

    /// reduction の縮約軸を返す（`Sum`／`Max` 以外は `None`）。`detect.rs`
    /// のセグメント軸一致判定・`plan.rs` の `dim`→`axis` 写像が使う。
    pub(crate) fn reduction_dim(&self) -> Option<Option<usize>> {
        match self {
            FusionOp::Sum { dim, .. } | FusionOp::Max { dim, .. } => Some(*dim),
            _ => None,
        }
    }

    /// `Broadcast` の拡張軸を返す（`Broadcast` 以外は `None`）。
    /// [`reduction_dim`](Self::reduction_dim) と対の役割を持つ（#588。
    /// `detect.rs` のセグメント軸一致判定・`plan.rs` の `dim`→`axis`
    /// 写像が使う）。
    pub(crate) fn broadcast_dim(&self) -> Option<Option<usize>> {
        match self {
            FusionOp::Broadcast { dim, .. } => Some(*dim),
            _ => None,
        }
    }
}

/// 融合グラフ内ノードの識別子。`nodes: Vec<FusionNode>` への添字を直接
/// 表す newtype（`fandhe_ai_autodiff::tape::NodeId`〈`tape.rs:35`〉と同型パターン。
/// 生の `usize` と取り違えないための型区別）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct FusionNodeId(pub(crate) usize);

/// 融合可否判定に使う静的メタデータ（shape・contiguous・dtype。
/// 設計書 §2.3）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeMeta {
    pub(crate) shape: Vec<usize>,
    /// contiguous かどうか。`false` の場合は transpose／broadcast view を
    /// 示唆し、`detect.rs` の非融合フォールバック判定に使う
    /// （設計書 §2.3「1 フィールドの真偽値判定」）。
    pub(crate) contiguous: bool,
    pub(crate) dtype: DType,
}

impl NodeMeta {
    pub(crate) fn new(shape: Vec<usize>, contiguous: bool, dtype: DType) -> Self {
        Self {
            shape,
            contiguous,
            dtype,
        }
    }
}

/// 融合グラフの 1 ノード（設計書 §2.2）。
#[derive(Debug, Clone)]
pub(crate) struct FusionNode {
    pub(crate) op: FusionOp,
    pub(crate) meta: NodeMeta,
    /// このノードの出力を入力として参照するノード数（fan-out。
    /// 設計書 §2.4）。`FusionGraph::push` が参照側ノード追加時に加算
    /// 維持する（fan-out はそれ自体を融合不能条件にしない。PoC-9
    /// `ew_fanout` 実測根拠、設計書 §2.4）。
    pub(crate) use_count: usize,
}

/// 融合グラフ構築時の検証エラー（OWASP A03 観点。設計書 §5「グラフ構築
/// API は既存の `ShapeError`（`error.rs:19` 付近）経路をそのまま再利用し、
/// 融合グラフ構築時に独自の検証経路を新設しない」の実体）。
///
/// elementwise binary のオペランド shape 不一致は `tensor-core` 全体で
/// 共通の [`ShapeError::ShapeMismatch`] をそのまま包んで返す
/// （`ops_shape.rs`・`typed.rs` と同一のエラー型・パターンマッチ資産を
/// #163／#165 の融合グラフエラー処理からも使えるようにするため。
/// レビュー指摘により `FusionGraphError` 独自の同型 variant を廃止）。
/// 範囲外ノード ID 検査のみは融合グラフ固有の不変条件
/// 〈入力 ID は常に自ノードより小さい〉であり `ShapeError` に対応する
/// 概念が存在しないため、`NodeIdOutOfRange` として本型に残す。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FusionGraphError {
    /// 参照した入力 `FusionNodeId` が現在のグラフサイズの範囲外
    /// （まだ push されていないノードを指す）。
    NodeIdOutOfRange { id: usize, len: usize },
    /// elementwise binary（`Add`／`Mul`）のオペランド shape が一致しない
    /// （broadcast view はここでは扱わない。`contiguous == false` として
    /// `detect.rs` の非融合フォールバック判定に委ねる設計書 §2.3 の方針）。
    /// 既存 `ShapeError::ShapeMismatch` をそのまま包む。
    Shape(ShapeError),
    /// [`FusionPlan::from_segment`]（`plan.rs`）が走査するノードの `op`
    /// が elementwise 5 演算（`Add`／`Mul`／`Relu`／`Exp`／`Tanh`）以外
    /// （`Input`／`Gemm`／`Sum`／`Max`）だった。#162 の連鎖検出
    /// （`detect.rs::detect_fusion`）は `is_elementwise()` を満たす
    /// ノードのみを `segment.nodes` へ挿入するため通常経路では到達
    /// しないが、`segment` と `graph` の不整合（呼び出し元のバグ）を
    /// 検出する防御的検証として区別する（`NodeIdOutOfRange` は「ID が
    /// 範囲外」という別の不変条件の違反を表すため意味論的に転用しない。
    /// レビュー指摘 #163）。
    UnexpectedOpKind { id: usize },
    /// [`FusionPlan::from_segment`]（`plan.rs`）が Add／Mul／Relu／Exp／
    /// Tanh のオペランドとして参照する元 `FusionNodeId` が、`segment`
    /// の再番号付け表（`leaves` ＋ `nodes` から構築した `index_of`）に
    /// 存在しない（`segment.nodes` の走査順序に対し、参照先の
    /// `FusionNodeId` が `segment.leaves` にも `segment.nodes` にも
    /// 含まれない）。`segment` と `graph` の不整合（呼び出し元のバグ）を
    /// 検出する防御的検証として `UnexpectedOpKind` と同様に区別する
    /// （レビュー指摘 #400: `index_of[&id]` の直接添字アクセスは
    /// 不整合な `segment` に対して panic するため、`HashMap::get` と
    /// 本 variant による fail-closed な処理へ置き換える）。
    DanglingOperandReference { id: usize },
    /// [`FusionPlan::from_segment`]（`plan.rs`）内の `compute_row_fusion`
    /// （#588・§3.5）が返す検証エラーを包む。`detect_fusion`・`push` の
    /// 検証を通過した通常経路（`detect.rs` が確定したセグメント軸・
    /// `push` が確定した shape 契約）では到達しない防御的検証だが、
    /// `from_segment` の戻り値型を `FusionGraphError` に統一するために
    /// ここへ合流させる（#588 実装計画 §3.6）。
    Plan(Box<super::plan::FusionPlanError>),
}

impl std::fmt::Display for FusionGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionGraphError::NodeIdOutOfRange { id, len } => {
                write!(
                    f,
                    "fusion node id {id} out of range (graph has {len} nodes)"
                )
            }
            FusionGraphError::Shape(err) => write!(f, "fusion operand shape error: {err}"),
            FusionGraphError::UnexpectedOpKind { id } => {
                write!(
                    f,
                    "fusion segment references node {id} whose op is not one of the 5 elementwise kinds"
                )
            }
            FusionGraphError::DanglingOperandReference { id } => {
                write!(
                    f,
                    "fusion segment operand references node {id} which is absent from the segment's leaves/nodes"
                )
            }
            FusionGraphError::Plan(err) => {
                write!(f, "fusion segment row-fusion metadata error: {err}")
            }
        }
    }
}

impl std::error::Error for FusionGraphError {}

impl From<ShapeError> for FusionGraphError {
    fn from(err: ShapeError) -> Self {
        FusionGraphError::Shape(err)
    }
}

impl From<super::plan::FusionPlanError> for FusionGraphError {
    fn from(err: super::plan::FusionPlanError) -> Self {
        FusionGraphError::Plan(Box::new(err))
    }
}

/// 融合グラフ本体（設計書 §2.2）。ノード ID＋隣接（入力エッジ）リストに
/// よる DAG。`fandhe_ai_autodiff::Tape` と同様、ノードは発生順に `Vec` へ追記され、
/// 入力は常に自ノードより小さい `FusionNodeId` を指す不変条件を持つ
/// （`push` が検証する）。
#[derive(Debug, Clone, Default)]
pub(crate) struct FusionGraph {
    nodes: Vec<FusionNode>,
}

impl FusionGraph {
    pub(crate) fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// 現在のノード数。
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 全ノードへの読み取り専用スライス（発生順）。`plan.rs` の
    /// `FusionPlan::from_graph`／`FusionPlan::ops`（設計書 §3.4 の DTO
    /// アクセサ）が発生順トポロジカル走査に使う。
    pub(crate) fn nodes_ref(&self) -> &[FusionNode] {
        &self.nodes
    }

    /// `id` のノードを参照する（`detect.rs` の後方走査から使う）。
    ///
    /// `id` は `push` の不変条件（自ノードより小さい入力 ID のみを許可）
    /// により通常は既存ノードを指すが、本メソッドは `pub(crate)` として
    /// クレート内の広い範囲から呼ばれうる。#163／#164 の結線コードが
    /// 未検証の `FusionNodeId` をそのまま渡す将来の呼び出しでも範囲外
    /// アクセスが panic に発展しないよう、範囲検証を本メソッドに一元化し
    /// `Result` で返す（#399 codex-review 指摘: `pub(crate)` であることは
    /// `.claude/rules/coding-rust.md`「本番経路で unwrap/expect を使わない」
    /// の型付きエラー契約を免除する根拠にはならない）。
    pub(crate) fn node(&self, id: FusionNodeId) -> Result<&FusionNode, FusionGraphError> {
        self.nodes
            .get(id.0)
            .ok_or(FusionGraphError::NodeIdOutOfRange {
                id: id.0,
                len: self.nodes.len(),
            })
    }

    /// 新規ノードを追記する。検証を先行させる（OWASP A03。モジュール
    /// 冒頭コメント参照）:
    ///
    /// 1. `op` が参照する入力 `FusionNodeId` がすべて現在のグラフサイズ
    ///    範囲内にあること（範囲外は [`FusionGraphError::NodeIdOutOfRange`]）
    /// 2. `Add`／`Mul`（elementwise binary）はオペランド shape が一致する
    ///    こと（不一致は既存 [`ShapeError::ShapeMismatch`] を包んだ
    ///    [`FusionGraphError::Shape`]）
    ///
    /// 検証を通過した場合のみノードを追記し、参照された入力ノードの
    /// `use_count` を加算する（設計書 §2.4）。
    pub(crate) fn push(
        &mut self,
        op: FusionOp,
        meta: NodeMeta,
    ) -> Result<FusionNodeId, FusionGraphError> {
        let inputs = op.inputs();
        for input in &inputs {
            if input.0 >= self.nodes.len() {
                return Err(FusionGraphError::NodeIdOutOfRange {
                    id: input.0,
                    len: self.nodes.len(),
                });
            }
        }

        // elementwise binary のみオペランド shape 一致を要求する（`Gemm`
        // は `[m, k] @ [k, n]` のように内部次元のみ一致すればよく、この
        // 検証は elementwise 連鎖検出の前提を保つためのもの。設計書
        // §2.3「broadcast view は contiguous == false として非融合判定
        // 側で扱う」により、shape が完全一致しないブロードキャストは
        // ここでは拒否せず、呼び出し側が事前に broadcast 済み shape ＋
        // `contiguous: false` で表現する契約とする）。
        if let FusionOp::Add(a, b)
        | FusionOp::Mul(a, b)
        | FusionOp::Sub(a, b)
        | FusionOp::Div(a, b) = &op
        {
            let lhs_shape = &self.nodes[a.0].meta.shape;
            let rhs_shape = &self.nodes[b.0].meta.shape;
            if lhs_shape != rhs_shape {
                return Err(ShapeError::ShapeMismatch {
                    lhs: lhs_shape.clone(),
                    rhs: rhs_shape.clone(),
                }
                .into());
            }
        }

        // elementwise（unary／binary）は出力 shape が入力 shape と恒等
        // でなければならない（設計書 §2.3「elementwise の出力 shape は
        // 入力の shape フィールドだけを読めば求まる」恒等計算という前提）。
        // 呼び出し側が渡す `meta.shape`（このノード自身の出力 shape）が
        // この恒等性から外れていないかを構築時に検査する。これを怠ると
        // `detect.rs`（#163／#164）の連鎖検出が誤った出力 shape を信じて
        // 走査することになり、呼び出し側のバグを検出できない
        // （push 前レビュー指摘。モジュール冒頭コメント「構築時に検証を
        // 先行させる」の対象に含める）。
        match &op {
            FusionOp::Add(a, _)
            | FusionOp::Mul(a, _)
            | FusionOp::Sub(a, _)
            | FusionOp::Div(a, _) => {
                // binary は rhs との一致を上の検査で確認済みのため、lhs
                // との一致のみ確認すれば三者の恒等性が揃う。
                let operand_shape = &self.nodes[a.0].meta.shape;
                if operand_shape != &meta.shape {
                    return Err(ShapeError::ShapeMismatch {
                        lhs: meta.shape.clone(),
                        rhs: operand_shape.clone(),
                    }
                    .into());
                }
            }
            FusionOp::Relu(a) | FusionOp::Exp(a) | FusionOp::Tanh(a) | FusionOp::Rsqrt(a) => {
                let input_shape = &self.nodes[a.0].meta.shape;
                if input_shape != &meta.shape {
                    return Err(ShapeError::ShapeMismatch {
                        lhs: meta.shape.clone(),
                        rhs: input_shape.clone(),
                    }
                    .into());
                }
            }
            // reduction（`Sum`／`Max`）は #586 でセグメント内包対象になった
            // ため、出力 shape も構築時に fail-closed で検証する
            // （`backend-cpu::reduction` の shape 契約の正: `dim=None` は
            // rank 0〈`[]`〉、`dim=Some(axis)` は縮約軸を除去した shape。
            // 検証ロジック自体は `ops_shape::reduce_out_shape` を再利用し
            // 二重定義しない）。軸範囲外は `ShapeError::AxisOutOfRange`
            // （`reduce_out_shape` から `?` で伝播）、縮約後 shape の不一致は
            // `ShapeError::ShapeMismatch` として拒否する。
            FusionOp::Sum { input, dim } | FusionOp::Max { input, dim } => {
                let input_shape = &self.nodes[input.0].meta.shape;
                let expected_shape = crate::ops_shape::reduce_out_shape(input_shape, *dim)?;
                if expected_shape != meta.shape {
                    return Err(ShapeError::ShapeMismatch {
                        lhs: meta.shape.clone(),
                        rhs: expected_shape,
                    }
                    .into());
                }
            }
            // `Broadcast`（#588）は reduction の逆演算: 出力 shape
            // （`meta.shape`。このノード自身の出力）を `dim` で縮約すると
            // 入力 shape に戻ることを要求する（`reduce_out_shape` を
            // 逆方向に適用する形で再利用し、専用の逆算ロジックを二重に
            // 書かない。`dim: None` の場合 `reduce_out_shape` は常に
            // `Vec::new()` を返すため、この検査は「入力が rank 0（`[]`）
            // であること」に自然に帰着する——#588 実装計画 §3.1 が定める
            // 「`dim: None` は入力が rank 0 で任意 shape への全 broadcast」
            // の契約そのもの）。軸範囲外は `ShapeError::AxisOutOfRange`
            // （`reduce_out_shape` から `?` で伝播）。
            FusionOp::Broadcast { input, dim } => {
                let input_shape = &self.nodes[input.0].meta.shape;
                let reduced_output = crate::ops_shape::reduce_out_shape(&meta.shape, *dim)?;
                if &reduced_output != input_shape {
                    return Err(ShapeError::ShapeMismatch {
                        lhs: input_shape.clone(),
                        rhs: reduced_output,
                    }
                    .into());
                }
            }
            // `Input`・`Gemm` は出力 shape が入力の恒等関数ではない
            // （`Input` は外部から与えられ、`Gemm` は常に融合境界として
            // ここでは検証しない。設計書 §2.1）。
            FusionOp::Input | FusionOp::Gemm(..) => {}
        }

        let new_id = FusionNodeId(self.nodes.len());
        for input in &inputs {
            self.nodes[input.0].use_count += 1;
        }
        self.nodes.push(FusionNode {
            op,
            meta,
            use_count: 0,
        });
        Ok(new_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_meta(shape: &[usize]) -> NodeMeta {
        NodeMeta::new(shape.to_vec(), true, DType::F32)
    }

    #[test]
    fn push_input_then_elementwise_chain_succeeds() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let sum = g.push(FusionOp::Add(x, y), f32_meta(&[4])).unwrap();
        let relu = g.push(FusionOp::Relu(sum), f32_meta(&[4])).unwrap();

        assert_eq!(g.len(), 4);
        assert_eq!(g.node(relu).unwrap().op, FusionOp::Relu(sum));
        // x・y はそれぞれ Add の入力として 1 回ずつ参照される。
        assert_eq!(g.node(x).unwrap().use_count, 1);
        assert_eq!(g.node(y).unwrap().use_count, 1);
        // sum は Relu の入力として 1 回参照される。
        assert_eq!(g.node(sum).unwrap().use_count, 1);
    }

    #[test]
    fn push_rejects_out_of_range_node_id() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        // まだ存在しない FusionNodeId(5) を参照する Add は拒否される。
        let bogus = FusionNodeId(5);
        let err = g.push(FusionOp::Add(x, bogus), f32_meta(&[4])).unwrap_err();
        assert_eq!(err, FusionGraphError::NodeIdOutOfRange { id: 5, len: 1 });
        // 検証失敗時はノードが追記されない（グラフサイズ不変）。
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn node_rejects_out_of_range_id() {
        // #399 codex-review 指摘: `node` 自体が範囲外 ID を型付きエラーで
        // 拒否することを直接検証する（push の入力検証を経由しない、
        // `node` 単独呼び出しの防御境界）。
        let mut g = FusionGraph::new();
        let _x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let err = g.node(FusionNodeId(1)).unwrap_err();
        assert_eq!(err, FusionGraphError::NodeIdOutOfRange { id: 1, len: 1 });
    }

    #[test]
    fn push_rejects_binary_shape_mismatch() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[2, 4])).unwrap();
        let err = g.push(FusionOp::Mul(x, y), f32_meta(&[4])).unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![4],
                rhs: vec![2, 4],
            })
        );
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn push_rejects_unary_output_shape_mismatch() {
        // Relu は恒等 shape のはず。呼び出し側が誤った meta.shape（[999]）
        // を渡した場合、入力（[4]）との不一致として拒否されなければ
        // ならない（push 前レビュー指摘: unary の出力 shape は無検証で
        // 通過していた）。
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let err = g.push(FusionOp::Relu(x), f32_meta(&[999])).unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![999],
                rhs: vec![4],
            })
        );
        // 検証失敗時はノードが追記されない（グラフサイズ不変）。
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn push_rejects_binary_output_shape_mismatch() {
        // オペランド同士（x・y）は shape 一致するが、meta.shape（このノード
        // 自身の出力 shape）が [999] と偽った場合は別途拒否されなければ
        // ならない（push 前レビュー指摘: オペランド一致検査だけでは
        // meta.shape 自体の恒等性は検証できていなかった）。
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let err = g.push(FusionOp::Add(x, y), f32_meta(&[999])).unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![999],
                rhs: vec![4],
            })
        );
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn push_gemm_boundary_does_not_require_equal_shape() {
        // Gemm は [m, k] @ [k, n] のように内部次元のみ一致すればよく、
        // shape 完全一致は要求しない（elementwise binary とは異なる）。
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[2, 3])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[3, 5])).unwrap();
        let gemm = g.push(FusionOp::Gemm(a, b), f32_meta(&[2, 5])).unwrap();
        assert_eq!(g.node(gemm).unwrap().op, FusionOp::Gemm(a, b));
    }

    /// #586: `Rsqrt` は他の elementwise unary と同じ恒等 shape 契約を持つ。
    #[test]
    fn push_rsqrt_accepts_identity_shape() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let r = g.push(FusionOp::Rsqrt(x), f32_meta(&[4])).unwrap();
        assert_eq!(g.node(r).unwrap().op, FusionOp::Rsqrt(x));
        assert!(g.node(r).unwrap().op.is_elementwise());
    }

    /// #586: `Rsqrt` も unary elementwise と同様、偽った出力 shape を
    /// 拒否する（push 前レビュー指摘 push_rejects_unary_output_shape_mismatch
    /// と同型の反例）。
    #[test]
    fn push_rejects_rsqrt_output_shape_mismatch() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let err = g.push(FusionOp::Rsqrt(x), f32_meta(&[999])).unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![999],
                rhs: vec![4],
            })
        );
        assert_eq!(g.len(), 1);
    }

    /// #586: `Sum { dim: None }`（全軸縮約）は `backend-cpu::reduction::sum`
    /// の shape 契約どおり rank 0（`[]`）のみを受理する。
    #[test]
    fn push_accepts_sum_full_reduction_scalar_shape() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let sum = g
            .push(
                FusionOp::Sum {
                    input: x,
                    dim: None,
                },
                f32_meta(&[]),
            )
            .unwrap();
        assert_eq!(
            g.node(sum).unwrap().op,
            FusionOp::Sum {
                input: x,
                dim: None
            }
        );
    }

    /// #586: `Sum { dim: Some(axis) }` は縮約軸を除去した shape のみを
    /// 受理する（`ops_shape::reduce_out_shape` と同一の契約）。
    #[test]
    fn push_rejects_sum_output_shape_mismatch() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[2, 4])).unwrap();
        // 正しい出力 shape は axis=0 を除去した [4] のはずが [1] を渡す。
        let err = g
            .push(
                FusionOp::Sum {
                    input: x,
                    dim: Some(0),
                },
                f32_meta(&[1]),
            )
            .unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![1],
                rhs: vec![4],
            })
        );
        assert_eq!(g.len(), 1);
    }

    /// #586: 軸範囲外の `Sum`／`Max` は `ShapeError::AxisOutOfRange`
    /// （`ops_shape::reduce_out_shape` から `?` で伝播）で拒否される。
    #[test]
    fn push_rejects_max_axis_out_of_range() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[2, 4])).unwrap();
        let err = g
            .push(
                FusionOp::Max {
                    input: x,
                    dim: Some(5),
                },
                f32_meta(&[2]),
            )
            .unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::AxisOutOfRange { axis: 5, rank: 2 })
        );
    }

    /// #586: `is_reduction`／`is_fusable` の境界再定義そのものを固定する。
    #[test]
    fn is_reduction_and_is_fusable_classify_ops_correctly() {
        let add = FusionOp::Add(FusionNodeId(0), FusionNodeId(0));
        let sum = FusionOp::Sum {
            input: FusionNodeId(0),
            dim: None,
        };
        let gemm = FusionOp::Gemm(FusionNodeId(0), FusionNodeId(0));
        let input = FusionOp::Input;

        assert!(!add.is_reduction() && add.is_fusable());
        assert!(sum.is_reduction() && sum.is_fusable());
        assert!(!gemm.is_reduction() && !gemm.is_fusable());
        assert!(!input.is_reduction() && !input.is_fusable());
    }

    /// #588: `Sub`／`Div` は `Add`／`Mul` と同じオペランド shape 一致・
    /// 出力 shape 恒等契約を持つ elementwise binary である。
    #[test]
    fn push_accepts_sub_and_div_with_identity_shape() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let sub = g.push(FusionOp::Sub(x, y), f32_meta(&[4])).unwrap();
        let div = g.push(FusionOp::Div(x, y), f32_meta(&[4])).unwrap();
        assert!(g.node(sub).unwrap().op.is_elementwise());
        assert!(g.node(div).unwrap().op.is_elementwise());
    }

    /// #588: `Sub`／`Div` もオペランド shape 不一致を `Add`／`Mul` と同型に
    /// 拒否する（反例は `push_rejects_binary_shape_mismatch` と同構成）。
    #[test]
    fn push_rejects_sub_operand_shape_mismatch() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[2, 4])).unwrap();
        let err = g.push(FusionOp::Sub(x, y), f32_meta(&[4])).unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![4],
                rhs: vec![2, 4],
            })
        );
    }

    /// #588: `Div` の出力 shape 恒等違反（`push_rejects_binary_output_shape_mismatch`
    /// と同型の反例）。
    #[test]
    fn push_rejects_div_output_shape_mismatch() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let err = g.push(FusionOp::Div(x, y), f32_meta(&[999])).unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![999],
                rhs: vec![4],
            })
        );
    }

    /// #588 §3.1: `Broadcast { dim: Some(axis) }` は `input` の shape に
    /// 軸を再挿入した shape のみを受理する（`[2] --dim Some(1)--> [2, 8]`）。
    #[test]
    fn push_accepts_broadcast_with_axis_reinsertion_shape() {
        let mut g = FusionGraph::new();
        let scalar_per_row = g.push(FusionOp::Input, f32_meta(&[2])).unwrap();
        let bc = g
            .push(
                FusionOp::Broadcast {
                    input: scalar_per_row,
                    dim: Some(1),
                },
                f32_meta(&[2, 8]),
            )
            .unwrap();
        assert_eq!(
            g.node(bc).unwrap().op,
            FusionOp::Broadcast {
                input: scalar_per_row,
                dim: Some(1)
            }
        );
        assert!(!g.node(bc).unwrap().op.is_elementwise());
        assert!(g.node(bc).unwrap().op.is_broadcast());
        assert!(g.node(bc).unwrap().op.is_fusable());
    }

    /// #588 §3.1: `Broadcast { dim: None }` は入力が rank 0（`[]`）である
    /// 場合のみ任意 shape への全 broadcast を受理する。
    #[test]
    fn push_accepts_broadcast_full_from_scalar_input() {
        let mut g = FusionGraph::new();
        let scalar = g.push(FusionOp::Input, f32_meta(&[])).unwrap();
        let bc = g
            .push(
                FusionOp::Broadcast {
                    input: scalar,
                    dim: None,
                },
                f32_meta(&[4, 8]),
            )
            .unwrap();
        assert_eq!(
            g.node(bc).unwrap().op,
            FusionOp::Broadcast {
                input: scalar,
                dim: None
            }
        );
    }

    /// #588 §3.1: `dim: None` なのに入力が rank 0 でない場合は
    /// `reduce_out_shape(meta.shape, None)`（常に `[]`）との不一致として
    /// 拒否される。
    #[test]
    fn push_rejects_broadcast_full_from_non_scalar_input() {
        let mut g = FusionGraph::new();
        let non_scalar = g.push(FusionOp::Input, f32_meta(&[2])).unwrap();
        let err = g
            .push(
                FusionOp::Broadcast {
                    input: non_scalar,
                    dim: None,
                },
                f32_meta(&[2, 8]),
            )
            .unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![2],
                rhs: Vec::new(),
            })
        );
    }

    /// #588 §3.1: `Broadcast` の逆縮約 shape が入力 shape と一致しない
    /// 場合は `ShapeMismatch` で拒否される（`push_rejects_sum_output_shape_mismatch`
    /// と対称の反例）。
    #[test]
    fn push_rejects_broadcast_output_shape_mismatch() {
        let mut g = FusionGraph::new();
        let per_row = g.push(FusionOp::Input, f32_meta(&[2])).unwrap();
        // 正しい出力 shape は axis=1 に per_row の shape [2] を再挿入した
        // [2, 8] のはずが [3, 8] を渡す（reduce_out_shape([3, 8], Some(1))
        // == [3] が per_row の shape [2] と不一致）。
        let err = g
            .push(
                FusionOp::Broadcast {
                    input: per_row,
                    dim: Some(1),
                },
                f32_meta(&[3, 8]),
            )
            .unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![2],
                rhs: vec![3],
            })
        );
    }

    /// #588 §3.1: 軸範囲外の `Broadcast` は `ShapeError::AxisOutOfRange`
    /// （`reduce_out_shape` から `?` で伝播）で拒否される。
    #[test]
    fn push_rejects_broadcast_axis_out_of_range() {
        let mut g = FusionGraph::new();
        let scalar = g.push(FusionOp::Input, f32_meta(&[])).unwrap();
        let err = g
            .push(
                FusionOp::Broadcast {
                    input: scalar,
                    dim: Some(5),
                },
                f32_meta(&[4]),
            )
            .unwrap_err();
        assert_eq!(
            err,
            FusionGraphError::Shape(ShapeError::AxisOutOfRange { axis: 5, rank: 1 })
        );
    }
}
