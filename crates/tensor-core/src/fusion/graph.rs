//! 融合グラフの中間表現（IR）本体（TASK-12.1a・#161 設計の実装。#162）。
//!
//! `docs/fusion-graph-design.md` §2 の型スケッチをそのまま実装する。
//! `autodiff::tape`（`crates/autodiff/src/tape.rs:35` 付近の
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

/// 融合グラフのノード種別（設計書 §2.1）。
///
/// `BackendOps`（`crate::backend_ops::BackendOps`）の各メソッドと 1:1
/// 対応させる: elementwise binary（`add`／`mul`）・unary（`relu`／
/// `exp`／`tanh`）が融合対象、`gemm`／`sum`／`max` は融合境界ノード
/// （融合セグメントに組み込まず、そこで走査を打ち切る印として扱う。
/// `detect.rs` 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FusionOp {
    /// リーフノード（グラフへの外部入力テンソル）。
    Input,
    // elementwise binary（backend_ops.rs の `add`/`mul` に対応）
    Add(FusionNodeId, FusionNodeId),
    Mul(FusionNodeId, FusionNodeId),
    // elementwise unary（`relu`/`exp`/`tanh` に対応）
    Relu(FusionNodeId),
    Exp(FusionNodeId),
    Tanh(FusionNodeId),
    // 融合境界ノード（融合しない。到達時に実体化する。設計書 §3 参照）
    Gemm(FusionNodeId, FusionNodeId),
    Sum {
        input: FusionNodeId,
        dim: Option<usize>,
    },
    Max {
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
            FusionOp::Add(a, b) | FusionOp::Mul(a, b) | FusionOp::Gemm(a, b) => vec![*a, *b],
            FusionOp::Relu(a) | FusionOp::Exp(a) | FusionOp::Tanh(a) => vec![*a],
            FusionOp::Sum { input, .. } | FusionOp::Max { input, .. } => vec![*input],
        }
    }

    /// elementwise 5 演算（融合対象）かどうか（設計書 §2.1「融合の直接
    /// 対象」）。`detect.rs` の連鎖検出が融合境界（`Gemm`／`Sum`／`Max`）
    /// と外部入力（`Input`）を区別するために使う。
    pub(crate) fn is_elementwise(&self) -> bool {
        matches!(
            self,
            FusionOp::Add(..)
                | FusionOp::Mul(..)
                | FusionOp::Relu(..)
                | FusionOp::Exp(..)
                | FusionOp::Tanh(..)
        )
    }
}

/// 融合グラフ内ノードの識別子。`nodes: Vec<FusionNode>` への添字を直接
/// 表す newtype（`autodiff::tape::NodeId`〈`tape.rs:35`〉と同型パターン。
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
        }
    }
}

impl std::error::Error for FusionGraphError {}

impl From<ShapeError> for FusionGraphError {
    fn from(err: ShapeError) -> Self {
        FusionGraphError::Shape(err)
    }
}

/// 融合グラフ本体（設計書 §2.2）。ノード ID＋隣接（入力エッジ）リストに
/// よる DAG。`autodiff::Tape` と同様、ノードは発生順に `Vec` へ追記され、
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

    /// `id` のノードを参照する（`detect.rs` の後方走査から使う）。
    ///
    /// `id` は `push` の不変条件（自ノードより小さい入力 ID のみを許可）
    /// により常に既存ノードを指すはずだが、本メソッドはその不変条件に
    /// 依存する呼び出し側からのみ使われる `pub(crate)` API のため、
    /// 範囲外アクセスは呼び出し側のバグとして扱い `panic` させる
    /// （`push` が唯一のノード追加経路であり、外部入力の直接検証は
    /// `push` 側で完結している。本番経路〈ユーザー向け公開 API〉には
    /// 露出しない `pub(crate)` 内部 API のため `.claude/rules/
    /// coding-rust.md`「本番経路で unwrap/expect を使わない」の対象外）。
    pub(crate) fn node(&self, id: FusionNodeId) -> &FusionNode {
        &self.nodes[id.0]
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
        if let FusionOp::Add(a, b) | FusionOp::Mul(a, b) = &op {
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
            FusionOp::Add(a, _) | FusionOp::Mul(a, _) => {
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
            FusionOp::Relu(a) | FusionOp::Exp(a) | FusionOp::Tanh(a) => {
                let input_shape = &self.nodes[a.0].meta.shape;
                if input_shape != &meta.shape {
                    return Err(ShapeError::ShapeMismatch {
                        lhs: meta.shape.clone(),
                        rhs: input_shape.clone(),
                    }
                    .into());
                }
            }
            // `Input`・`Gemm`・`Sum`・`Max` は出力 shape が入力の恒等関数
            // ではない（`Input` は外部から与えられ、`Gemm`／`Sum`／`Max`
            // は融合境界としてここでは検証しない。設計書 §2.1）。
            FusionOp::Input | FusionOp::Gemm(..) | FusionOp::Sum { .. } | FusionOp::Max { .. } => {}
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
        assert_eq!(g.node(relu).op, FusionOp::Relu(sum));
        // x・y はそれぞれ Add の入力として 1 回ずつ参照される。
        assert_eq!(g.node(x).use_count, 1);
        assert_eq!(g.node(y).use_count, 1);
        // sum は Relu の入力として 1 回参照される。
        assert_eq!(g.node(sum).use_count, 1);
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
        assert_eq!(g.node(gemm).op, FusionOp::Gemm(a, b));
    }
}
