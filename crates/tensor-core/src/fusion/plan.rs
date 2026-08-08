//! 融合カーネル生成向け公開 DTO（`FusionPlan`。TASK-12.1c・#163）。
//!
//! `docs/fusion-graph-design.md` §3.4「遅延グラフと `BackendOps`・
//! `Tensor` 契約の接続」が確定した契約をそのまま実装する。`FusionOp`／
//! `FusionNode`／`FusionGraph`（`graph.rs`。§2）は `pub(crate)` のまま
//! 変更しない設計判断（§2.5）のため、`backend-cpu`／`backend-cuda`／
//! `backend-metal`（`pub trait BackendOps` を実装する外部クレート）が
//! 融合グラフの内容を読み取る唯一の経路が本モジュールの公開 DTO
//! （[`FusionPlan`]・[`FusedOpKind`]・[`FusedNodeIndex`]）である。
//!
//! # 構築経路（設計書 §3.4 の 2 系統）
//!
//! - [`FusionPlan::from_segment`]（`pub(crate)`）: `tensor-core` 内で
//!   `FusionGraph` と `detect::FusionSegment`（#162 の連鎖検出結果）から
//!   構築する経路。設計書は `from_graph(graph, root) -> FusionPlan`
//!   というスケッチを示すが、本実装は #162 で確定した `FusionSegment`
//!   （検出済みセグメント。境界ノード・葉が既に分離済み）を受け取る
//!   形へ変更する（検出の二重実行を避け、`Result` で型付きエラーを
//!   返せるようにするため。`docs/fusion-graph-design.md` §3.4 へ本 PR で
//!   追記する）。
//! - [`FusionPlan::from_ops`]（`pub` + `#[doc(hidden)]`）: `autodiff`
//!   クレート専用のクレート間構築経路（設計書 §3.4「`autodiff` クレート
//!   専用の構築経路」）。`tensor-core` 内部の `pub(crate)` 型を一切
//!   経由せず、既に `pub` な DTO のみから構築する。**実装計画 #163 の
//!   前倒し判断**: 設計書は「実装は #164 のスコープ」とするが、CPU
//!   融合カーネル（`backend-cpu::fused_elementwise`）の受け入れ検証
//!   （融合 vs 非融合の数値一致）に `backend-cpu` の統合テストから
//!   `FusionPlan` を構築するクレート間経路が必須であるため、#163 で
//!   前倒し実装する（シグネチャ・検証仕様は設計書 §3.4 で確定済みであり
//!   前倒しに設計変更を伴わない）。
//!
//! # 出力ノードの契約（本モジュールが確定する暗黙の設計判断）
//!
//! 設計書は「どのノードが出力か」を明示するアクセサを定義していない
//! （[`FusionPlan::ops`] のみで演算列を公開する）。本実装は
//! **「発生順（トポロジカル順）で最後の [`FusedOpKind`] エントリが
//! このプランの出力ノードである」という契約を確定する**——融合対象
//! セグメントは常に単一の出力（`root`。設計書 §3.2「`root` を出力と
//! する部分グラフ」）を持ち、トポロジカル順で `root` より後ろに現れる
//! セグメント内ノードは存在しえない（`detect_fusion` は `root` 以下の
//! ID のみを走査する）ため、この契約は `from_segment` の構築から自然に
//! 導かれる。`from_ops`（`autodiff` 側の構築経路）もこの契約に従う
//! ことを呼び出し規約として要求し、`from_ops` の防御的検証で
//! 「末尾エントリが `Input` ではないこと」を検査することで運用上の
//! 逸脱を検出する（`backend-cpu::fused_elementwise::run_fused_elementwise`
//! がこの契約を読んで出力レジスタを決定する）。

use super::detect::FusionSegment;
use super::graph::{FusionGraph, FusionGraphError, FusionOp};
use crate::device::BackendError;
use crate::dispatch::DType;
use crate::error::ShapeError;
use std::collections::HashMap;
use std::fmt;

/// [`FusionPlan`] 内のノード位置を指す公開インデックス。内部の
/// `FusionNodeId`（`pub(crate)`、`graph.rs`）はそのまま公開できないため、
/// `FusionPlan` 内でのみ意味を持つ 0 起点の連番（発生順）として別の型を
/// 用意する（設計書 §3.4）。
pub type FusedNodeIndex = usize;

/// [`FusionPlan::ops`] が列挙する 1 ノード分の演算内容。内部
/// `pub(crate)` の [`FusionOp`]（`graph.rs`・§2.1）と 1:1 対応するが、
/// 融合境界ノード（`Gemm`／`Sum`／`Max`）は設計書 §3.2 (a)(b) のとおり
/// `FusionPlan` 内に現れない（実体化境界のため、融合対象区間そのものに
/// は含まれない）ので列挙しない。フィールドは [`FusedNodeIndex`]
/// （plain `usize`）のみで構成し、`pub(crate)` 型を一切参照しない
/// （設計書 §3.4「privacy 制約」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOpKind {
    /// 葉ノード（このプランへの外部入力）。`leaf_index` は `run_fused`
    /// の `leaves: &[&Tensor<f32>]` の添字と対応する
    /// （`backend-cpu::fused_elementwise::run_fused_elementwise` の
    /// 呼び出し契約）。
    Input {
        leaf_index: FusedNodeIndex,
    },
    Add {
        lhs: FusedNodeIndex,
        rhs: FusedNodeIndex,
    },
    Mul {
        lhs: FusedNodeIndex,
        rhs: FusedNodeIndex,
    },
    Relu {
        input: FusedNodeIndex,
    },
    Exp {
        input: FusedNodeIndex,
    },
    Tanh {
        input: FusedNodeIndex,
    },
}

/// [`FusionPlan::from_ops`] の防御的検証エラー（設計書 §5「グラフ構築
/// API はテンソル shape／stride の検証を先行させる」と同型の契約。
/// OWASP A03 観点。`.claude/rules/security.md`）。
///
/// `from_ops` は `autodiff` 専用の内部契約であり、呼び出し元
/// （`autodiff` 側の materialize ヘルパー、実装は #164）は検証済みの
/// `ops` しか渡さない想定のため、実運用では到達しない防御的検証と
/// 位置付ける（設計書 §3.4「実運用では到達しない防御的検証と位置付ける」）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionPlanError {
    /// `Add`／`Mul`／`Relu`／`Exp`／`Tanh` が参照する [`FusedNodeIndex`]
    /// が、自ノードの発生順位置より手前（トポロジカル順）を指していない
    /// （`graph.rs::FusionGraph` の「入力は常に自ノードより小さい ID」
    /// 不変条件〈§2.2〉と同じ契約を `ops` の Vec 上でも要求する）。
    IndexOutOfRange { index: usize, at: usize },
    /// `Input { leaf_index }` の `leaf_index` が `leaf_count` の範囲外。
    LeafIndexOutOfRange {
        leaf_index: usize,
        leaf_count: usize,
    },
    /// `ops` 内の `Input` エントリ数が宣言された `leaf_count` と一致しない。
    LeafCountMismatch { declared: usize, actual: usize },
    /// `ops` 内に同じ `leaf_index` を持つ `Input` エントリが 2 個以上
    /// 存在する（レビュー指摘 #400: `leaf_count` 個数一致と各
    /// `leaf_index < leaf_count` の検証だけでは、例えば
    /// `leaf_count == 2` に対し `leaf_index: 0` を 2 個・`leaf_index: 1`
    /// を 0 個並べる入力を検出できない。この場合 `leaf_index: 1` が
    /// 一度も使われないまま融合結果が非融合結果と静かに乖離しうるため、
    /// 各 `leaf_index` が `0..leaf_count` に一度ずつ出現することを
    /// 型付きエラーで拒否する）。
    DuplicateLeafIndex { leaf_index: usize },
    /// `ops` が空、または `Input` エントリのみで構成される（elementwise
    /// ノードが 1 個も無い＝融合する意味がない）。
    NoElementwiseNode,
    /// `ops` の末尾エントリが `Input` である（本モジュール冒頭「出力
    /// ノードの契約」参照。末尾が出力ノードでなければ
    /// `run_fused_elementwise` がどのレジスタを書き出すべきか判定
    /// できない）。
    LastOpIsInput,
    /// `output_shape` の要素数積が `usize` でオーバーフローする
    /// （`tensor-core::tensor::checked_numel` と同じ検査を `from_ops`
    /// でも行う。`from_segment` 経路は `output_shape` を既存の検証済み
    /// `Tensor` の shape からのみ導出するため対象外だが、`from_ops` は
    /// `autodiff` から任意の `Vec<usize>` を直接受け取る公開経路であり、
    /// `FusionPlan` 単体の型不変条件として shape 妥当性を保証する必要が
    /// ある。レビュー指摘 #163: 検証を欠くと不正 shape でも構築に成功し、
    /// 後段〈`backend-cpu::run_fused_elementwise`〉の shape 一致検証に
    /// 検証責務が漏れ出す）。
    OutputShapeOverflow,
}

impl fmt::Display for FusionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionPlanError::IndexOutOfRange { index, at } => write!(
                f,
                "fusion plan node {at} references non-preceding index {index} (topological order violation)"
            ),
            FusionPlanError::LeafIndexOutOfRange {
                leaf_index,
                leaf_count,
            } => write!(
                f,
                "fusion plan leaf_index {leaf_index} out of range (leaf_count={leaf_count})"
            ),
            FusionPlanError::LeafCountMismatch { declared, actual } => write!(
                f,
                "fusion plan leaf_count mismatch: declared={declared}, actual Input entries={actual}"
            ),
            FusionPlanError::DuplicateLeafIndex { leaf_index } => {
                write!(f, "fusion plan has duplicate Input leaf_index {leaf_index}")
            }
            FusionPlanError::NoElementwiseNode => {
                write!(f, "fusion plan has no elementwise node (Input-only ops)")
            }
            FusionPlanError::LastOpIsInput => {
                write!(
                    f,
                    "fusion plan's last op must not be Input (output node contract)"
                )
            }
            FusionPlanError::OutputShapeOverflow => {
                write!(f, "fusion plan output_shape element count overflows usize")
            }
        }
    }
}

impl std::error::Error for FusionPlanError {}

/// [`FusionPlanError`] を [`BackendError`] へ変換する（TASK-12.1d・
/// #164）。`autodiff::tape::build_lazy_plan`
/// （`crates/autodiff/src/tape.rs`）は自身の遅延ノード連鎖から組み立てた
/// `ops` を [`FusionPlan::from_ops`] へ渡し、戻り値を `Result<_,
/// BackendError>`（層 1／層 2 の実体化ヘルパーが共通で扱うエラー型。
/// `AutodiffError::Backend` 経由で呼び出し元へ伝播する）へ `?` でそのまま
/// 合流させる契約のため、この変換が必要になる。`from_ops` の防御的検証
/// エラーは実運用では到達しない想定（本モジュール冒頭「構築経路」参照）
/// のため、各 variant を対応する [`ShapeError::ElementCountMismatch`]／
/// [`ShapeError::ElementCountOverflow`] へロスありで丸める（`autodiff`
/// 側は `Unsupported` と区別さえできればよく、fail-safe な per-op
/// フォールバック〈`materialize_fallible`〉へ委ねる設計とは独立に、この
/// 変換自体は「型不変条件違反」を表す `ShapeMismatch` 系へ寄せる）。
impl From<FusionPlanError> for BackendError {
    fn from(err: FusionPlanError) -> Self {
        match err {
            FusionPlanError::IndexOutOfRange { index, at } => {
                BackendError::ShapeMismatch(ShapeError::ElementCountMismatch {
                    expected: at,
                    actual: index,
                })
            }
            FusionPlanError::LeafIndexOutOfRange {
                leaf_index,
                leaf_count,
            } => BackendError::ShapeMismatch(ShapeError::ElementCountMismatch {
                expected: leaf_count,
                actual: leaf_index,
            }),
            FusionPlanError::LeafCountMismatch { declared, actual } => {
                BackendError::ShapeMismatch(ShapeError::ElementCountMismatch {
                    expected: declared,
                    actual,
                })
            }
            FusionPlanError::DuplicateLeafIndex { leaf_index } => {
                BackendError::ShapeMismatch(ShapeError::ElementCountMismatch {
                    expected: leaf_index,
                    actual: leaf_index,
                })
            }
            FusionPlanError::NoElementwiseNode | FusionPlanError::LastOpIsInput => {
                BackendError::Unsupported(err.to_string())
            }
            FusionPlanError::OutputShapeOverflow => {
                BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
            }
        }
    }
}

/// `run_fused`（`BackendOps` の非破壊拡張。実装は #164）へ渡す公開の
/// 不透明ハンドル。`BackendOps` は `pub trait`（`backend-cpu`／
/// `backend-cuda`／`backend-metal` が実装）のため、その既定メソッドの
/// 引数型は `pub` でなければならない（privacy 制約。設計書 §3.4）。
/// 内部の融合 IR（`FusionGraph`／`FusionNode`／`FusionOp`。§2、
/// `pub(crate)` のまま変更しない）はフィールドとして直接保持せず、
/// 構築時に本モジュール限定の内部表現（`ops`／`output_shape`／
/// `dtype`／`leaf_count`／`use_counts`）へ変換し尽くす（`tensor-core`
/// 外からは構築・分解できない。読み取りは下記 `impl FusionPlan` の
/// `pub` メソッドを通じてのみ行う）。
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// 発生順（トポロジカル順）の演算列。`from_segment` は
    /// `[leaf 0, leaf 1, ..., leaf N-1, elementwise node 0, ...]` の順で
    /// 構築する（`leaf_index` とプラン内 [`FusedNodeIndex`] 0..N-1 が
    /// 一致する契約。モジュール冒頭「出力ノードの契約」も参照）。
    ops: Vec<FusedOpKind>,
    output_shape: Vec<usize>,
    dtype: DType,
    leaf_count: usize,
    /// `ops[i]` がプラン内（このセグメント内）の他ノードから参照される
    /// 回数。境界ノード（Gemm／Sum／Max。プラン外）からの参照は含まない
    /// （設計書 §3.4 `use_count` アクセサの契約）。
    use_counts: Vec<usize>,
}

impl FusionPlan {
    /// `graph` のうち `segment` が指す融合対象区間（境界ノードを含まない、
    /// #162 の連鎖検出が既に確定した elementwise 連鎖）から
    /// [`FusionPlan`] を構築する（`pub(crate)`。`tensor-core` 内で
    /// `FusionGraph` が既に存在する場合の構築経路。設計書 §3.4）。
    ///
    /// `segment.leaves`（昇順・重複なし）を `leaf_index` 0..N-1 へ、
    /// `segment.nodes`（発生順＝トポロジカル順に昇順ソート済み。
    /// `detect.rs::FusionSegment` の契約）をその直後の連番へ再番号付け
    /// する。`segment.root` は常に `segment.nodes` の末尾（最大
    /// `FusionNodeId`）であるため、構築される `ops` の末尾エントリが
    /// 自然にプランの出力ノードになる（モジュール冒頭「出力ノードの
    /// 契約」）。
    ///
    /// # エラー
    ///
    /// `graph.node` が返す [`FusionGraphError::NodeIdOutOfRange`]、
    /// `segment.nodes` が elementwise 5 演算以外のノードを含む場合の
    /// [`FusionGraphError::UnexpectedOpKind`]、または Add／Mul／Relu／
    /// Exp／Tanh のオペランドが再番号付け表に存在しない場合の
    /// [`FusionGraphError::DanglingOperandReference`] を返す（いずれも
    /// `segment` が `graph` と整合しない場合の防御。通常は #162 の検出
    /// 結果をそのまま渡す限り到達しない。レビュー指摘 #400: 従来は
    /// `index_of[&id]` の直接添字アクセスで panic していた）。
    pub(crate) fn from_segment(
        graph: &FusionGraph,
        segment: &FusionSegment,
    ) -> Result<FusionPlan, FusionGraphError> {
        let leaf_count = segment.leaves.len();

        // 再番号付け表: 元の FusionNodeId(usize) -> プラン内 FusedNodeIndex。
        let mut index_of: HashMap<usize, FusedNodeIndex> =
            HashMap::with_capacity(leaf_count + segment.nodes.len());

        let mut ops: Vec<FusedOpKind> = Vec::with_capacity(leaf_count + segment.nodes.len());
        for (leaf_index, leaf_id) in segment.leaves.iter().enumerate() {
            index_of.insert(leaf_id.0, leaf_index);
            ops.push(FusedOpKind::Input { leaf_index });
        }
        // 先にすべてのセグメントノードへ番号を割り当てる（fan-in で
        // 後続ノードが手前のノードを参照するため、`FusedOpKind` を
        // 組み立てる前に index_of を完成させておく必要がある）。
        for (offset, node_id) in segment.nodes.iter().enumerate() {
            index_of.insert(node_id.0, leaf_count + offset);
        }

        for node_id in &segment.nodes {
            let node = graph.node(*node_id)?;
            // `segment.nodes` は #162 の連鎖検出（`is_elementwise()` を
            // 満たすノードのみを `included` へ挿入する。`detect.rs`
            // 参照）が確定した集合であるため、ここに現れる `op` は常に
            // elementwise 5 演算のいずれかである。`Input`／`Gemm`／
            // `Sum`／`Max` が紛れ込む経路は存在しないが、`graph`
            // （#399 の教訓どおり `pub(crate)` であっても未検証入力を
            // panic で処理しない）と `segment` の整合をここでも
            // 型付きエラーで扱う（呼び出し元のバグにより不整合な
            // `segment` が渡された場合の防御的検証）。
            let kind = match &node.op {
                FusionOp::Add(a, b) => FusedOpKind::Add {
                    lhs: lookup_index(&index_of, a.0)?,
                    rhs: lookup_index(&index_of, b.0)?,
                },
                FusionOp::Mul(a, b) => FusedOpKind::Mul {
                    lhs: lookup_index(&index_of, a.0)?,
                    rhs: lookup_index(&index_of, b.0)?,
                },
                FusionOp::Relu(a) => FusedOpKind::Relu {
                    input: lookup_index(&index_of, a.0)?,
                },
                FusionOp::Exp(a) => FusedOpKind::Exp {
                    input: lookup_index(&index_of, a.0)?,
                },
                FusionOp::Tanh(a) => FusedOpKind::Tanh {
                    input: lookup_index(&index_of, a.0)?,
                },
                FusionOp::Input
                | FusionOp::Gemm(..)
                | FusionOp::Sum { .. }
                | FusionOp::Max { .. } => {
                    // 到達しない防御的分岐（上記コメント参照）。専用の
                    // `FusionGraphError::UnexpectedOpKind` を返す
                    // （`NodeIdOutOfRange` は「ID が範囲外」という別の
                    // 不変条件の違反を表すため意味論的に転用しない。
                    // レビュー指摘 #163）。
                    return Err(FusionGraphError::UnexpectedOpKind { id: node_id.0 });
                }
            };
            ops.push(kind);
        }

        let root_node = graph.node(segment.root)?;
        let output_shape = root_node.meta.shape.clone();
        let dtype = root_node.meta.dtype;

        let use_counts = compute_use_counts(&ops);

        Ok(FusionPlan {
            ops,
            output_shape,
            dtype,
            leaf_count,
            use_counts,
        })
    }

    /// `autodiff` クレート専用の構築経路（設計書 §3.4「`autodiff`
    /// クレート専用の構築経路」）。`tensor-core` 内部の `pub(crate)` 型
    /// （`FusionGraph`／`FusionNode`／`FusionOp`）を一切経由せず、既に
    /// `pub` な DTO（[`FusedOpKind`]／[`DType`]／[`FusedNodeIndex`]）
    /// だけから直接構築する。`#[doc(hidden)]` を付す理由: この経路は
    /// `autodiff` という単一の内部利用者のためのクレート間契約であり、
    /// 利用者向けの融合制御 API ではない（REQ-12「利用者が明示的に融合を
    /// 制御する API は提供しないこと」への抵触を避けるため）。
    ///
    /// # 検証（設計書 §5・OWASP A03。実運用では到達しない防御的検証）
    ///
    /// - `ops[i]` が参照する [`FusedNodeIndex`] はすべて `i` より小さい
    ///   こと（トポロジカル順の不変条件。`graph.rs::FusionGraph::push`
    ///   と同じ「入力は常に自ノードより小さい」契約）
    /// - `Input { leaf_index }` の `leaf_index` は `leaf_count` 未満であること
    /// - `ops` 内の `Input` エントリ数が `leaf_count` と一致すること
    /// - 各 `leaf_index` が `0..leaf_count` に重複なく一度ずつ出現すること
    ///   （レビュー指摘 #400: 個数一致・範囲チェックのみでは
    ///   `leaf_index` の重複〈他の leaf が未使用のまま融合結果が
    ///   非融合結果と静かに乖離しうる〉を見逃す）
    /// - `ops` が空でなく、`Input` 以外のノードを最低 1 個含むこと
    /// - `ops` の末尾エントリが `Input` でないこと（モジュール冒頭
    ///   「出力ノードの契約」）
    #[doc(hidden)]
    pub fn from_ops(
        ops: Vec<FusedOpKind>,
        output_shape: Vec<usize>,
        dtype: DType,
        leaf_count: usize,
    ) -> Result<FusionPlan, FusionPlanError> {
        if ops.is_empty() {
            return Err(FusionPlanError::NoElementwiseNode);
        }

        let mut input_count = 0usize;
        let mut has_elementwise = false;
        // `leaf_index` ごとの出現済みフラグ（レビュー指摘 #400:
        // `leaf_count` 個数一致・範囲チェックだけでは重複した
        // `leaf_index` を見逃す。`0..leaf_count` へ一度ずつ出現する
        // ことをここで確定する）。
        let mut leaf_seen = vec![false; leaf_count];
        for (at, op) in ops.iter().enumerate() {
            match *op {
                FusedOpKind::Input { leaf_index } => {
                    input_count += 1;
                    if leaf_index >= leaf_count {
                        return Err(FusionPlanError::LeafIndexOutOfRange {
                            leaf_index,
                            leaf_count,
                        });
                    }
                    if std::mem::replace(&mut leaf_seen[leaf_index], true) {
                        return Err(FusionPlanError::DuplicateLeafIndex { leaf_index });
                    }
                }
                FusedOpKind::Add { lhs, rhs } | FusedOpKind::Mul { lhs, rhs } => {
                    has_elementwise = true;
                    check_preceding(lhs, at)?;
                    check_preceding(rhs, at)?;
                }
                FusedOpKind::Relu { input }
                | FusedOpKind::Exp { input }
                | FusedOpKind::Tanh { input } => {
                    has_elementwise = true;
                    check_preceding(input, at)?;
                }
            }
        }

        if input_count != leaf_count {
            return Err(FusionPlanError::LeafCountMismatch {
                declared: leaf_count,
                actual: input_count,
            });
        }
        if !has_elementwise {
            return Err(FusionPlanError::NoElementwiseNode);
        }
        // `ops.is_empty()` を上で弾いているため添字アクセスは安全。
        if matches!(ops[ops.len() - 1], FusedOpKind::Input { .. }) {
            return Err(FusionPlanError::LastOpIsInput);
        }
        // `output_shape` の要素数積オーバーフロー検査（`tensor::
        // checked_numel` と同方針。`FusionPlanError::OutputShapeOverflow`
        // ドキュメント参照）。`from_ops` は `autodiff` から任意の
        // `Vec<usize>` を直接受け取るため、`FusionPlan` 構築時点で
        // shape の妥当性を確定させる（レビュー指摘 #163）。
        crate::tensor::checked_numel(&output_shape)
            .map_err(|_| FusionPlanError::OutputShapeOverflow)?;

        let use_counts = compute_use_counts(&ops);

        Ok(FusionPlan {
            ops,
            output_shape,
            dtype,
            leaf_count,
            use_counts,
        })
    }

    /// 発生順（トポロジカル順。`graph.rs` 「ノードは発生順に `Vec` へ
    /// 追記」）で [`FusedOpKind`] を列挙する。`backend-cpu::
    /// fused_elementwise::run_fused_elementwise` はこの順で辿ることで、
    /// 各ノードの入力（`lhs`／`rhs`／`input` が指す [`FusedNodeIndex`]）
    /// が走査済みであることを保証できる（トポロジカル順の定義そのもの。
    /// モジュール冒頭「出力ノードの契約」: 最後のエントリが出力）。
    pub fn ops(&self) -> impl Iterator<Item = FusedOpKind> + '_ {
        self.ops.iter().copied()
    }

    /// このプランが表す出力テンソルの shape（`NodeMeta.shape`。§2.3）。
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// このプランの dtype（§2.1 のとおり現状は常に `DType::F32`）。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// このプランが要求する葉ノード（外部入力）の個数。`run_fused` の
    /// `leaves: &[&Tensor<f32>]` の長さはこの値と一致する契約とし、
    /// 不一致はカーネル側（`backend-cpu::fused_elementwise`）が shape
    /// 検証と同様の扱いで拒否する。
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// 指定ノードの被参照数（`NodeMeta.use_count`〈§2.4〉のプラン内
    /// 版を公開する。fan-out のレジスタ内解決が読む）。**この値はプラン
    /// 内（融合セグメント内）からの被参照数のみを数える**。境界ノード
    /// （Gemm／Sum／Max。プラン外）から参照される場合、その参照はここに
    /// 含まれない（設計書 §3.4）。範囲外の `node` には `0` を返す
    /// （`Option` を避け設計書のシグネチャ `-> usize` を維持する。
    /// 範囲外は「プラン内から一度も参照されない」の極限として自然に
    /// 扱える）。
    pub fn use_count(&self, node: FusedNodeIndex) -> usize {
        self.use_counts.get(node).copied().unwrap_or(0)
    }
}

/// `index_of`（元の `FusionNodeId(usize)` -> プラン内 [`FusedNodeIndex`]
/// の再番号付け表）から `id` を引く（[`FusionPlan::from_segment`] 専用
/// ヘルパー）。`segment` と `graph` が整合していれば `segment.nodes` が
/// 参照するオペランドは必ず `segment.leaves` または手前の
/// `segment.nodes` に含まれ本関数は成功するが、呼び出し元のバグにより
/// `segment` が `graph` と不整合な場合（レビュー指摘 #400: 従来の
/// `index_of[&id]` 直接添字アクセスは未検出のまま panic していた）に
/// [`FusionGraphError::DanglingOperandReference`] を返す fail-closed な
/// 経路とする。
fn lookup_index(
    index_of: &HashMap<usize, FusedNodeIndex>,
    id: usize,
) -> Result<FusedNodeIndex, FusionGraphError> {
    index_of
        .get(&id)
        .copied()
        .ok_or(FusionGraphError::DanglingOperandReference { id })
}

/// `at` より手前（トポロジカル順で先行するノード）を指しているかを
/// 検証する（[`FusionPlan::from_ops`] 専用ヘルパー）。
fn check_preceding(referenced: FusedNodeIndex, at: usize) -> Result<(), FusionPlanError> {
    if referenced >= at {
        return Err(FusionPlanError::IndexOutOfRange {
            index: referenced,
            at,
        });
    }
    Ok(())
}

/// `ops` の各エントリがプラン内の他ノードから参照される回数を集計する
/// （[`FusionPlan::use_count`] の実体。`from_segment`／`from_ops` の
/// いずれからも共通利用する）。
fn compute_use_counts(ops: &[FusedOpKind]) -> Vec<usize> {
    let mut counts = vec![0usize; ops.len()];
    for op in ops {
        match *op {
            FusedOpKind::Add { lhs, rhs } | FusedOpKind::Mul { lhs, rhs } => {
                counts[lhs] += 1;
                counts[rhs] += 1;
            }
            FusedOpKind::Relu { input }
            | FusedOpKind::Exp { input }
            | FusedOpKind::Tanh { input } => {
                counts[input] += 1;
            }
            FusedOpKind::Input { .. } => {}
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::detect::{FusionDecision, detect_fusion};
    use crate::fusion::graph::{FusionNodeId, NodeMeta};

    fn f32_meta(shape: &[usize]) -> NodeMeta {
        NodeMeta::new(shape.to_vec(), true, DType::F32)
    }

    /// #162 レビュー申し送りの反例そのもの: 境界ノード（`Sum`）から
    /// セグメント内部ノードが外部参照される場合でも、プラン内
    /// `use_count` にはその外部参照が含まれないことを固定する。
    #[test]
    fn from_segment_use_count_excludes_external_boundary_reference() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let a = g.push(FusionOp::Add(x, y), f32_meta(&[4])).unwrap();
        let b = g.push(FusionOp::Mul(a, a), f32_meta(&[4])).unwrap();
        let c = g.push(FusionOp::Relu(b), f32_meta(&[4])).unwrap();
        let d = g.push(FusionOp::Exp(c), f32_meta(&[4])).unwrap();
        // 境界ノード（Sum）が a を外部から参照する（segment 外）。
        let _sum = g
            .push(
                FusionOp::Sum {
                    input: a,
                    dim: None,
                },
                f32_meta(&[1]),
            )
            .unwrap();

        let decision = detect_fusion(&g, d).unwrap();
        let segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };
        // 4 段連鎖（a, b, c, d）+ 境界外参照ノード a。
        assert_eq!(segment.nodes.len(), 4);

        let plan = FusionPlan::from_segment(&g, &segment).unwrap();
        // a はプラン内で b の両オペランドとして 2 回参照される
        // （Sum からの外部参照はここに含まれない）。
        let a_index = plan.leaf_count(); // leaves の直後が a（最初のセグメントノード）。
        assert_eq!(plan.use_count(a_index), 2);
    }

    #[test]
    fn from_segment_fan_out_reuses_register_index() {
        // a = x + y; b = a * a; c = b + x; d = c.relu() （4 段・fan-out あり）。
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let a = g.push(FusionOp::Add(x, y), f32_meta(&[4])).unwrap();
        let b = g.push(FusionOp::Mul(a, a), f32_meta(&[4])).unwrap();
        let c = g.push(FusionOp::Add(b, x), f32_meta(&[4])).unwrap();
        let d = g.push(FusionOp::Relu(c), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, d).unwrap();
        let segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };
        assert_eq!(segment.leaves, vec![FusionNodeId(0), FusionNodeId(1)]); // x, y 昇順

        let plan = FusionPlan::from_segment(&g, &segment).unwrap();
        assert_eq!(plan.leaf_count(), 2);
        let ops: Vec<FusedOpKind> = plan.ops().collect();
        // leaf 0=x, leaf 1=y, 2=a(Add(0,1)), 3=b(Mul(2,2)), 4=c(Add(3,0)), 5=d(Relu(4))
        assert_eq!(ops.len(), 6);
        assert!(matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }));
        assert!(matches!(ops[1], FusedOpKind::Input { leaf_index: 1 }));
        assert!(matches!(ops[2], FusedOpKind::Add { lhs: 0, rhs: 1 }));
        assert!(matches!(ops[3], FusedOpKind::Mul { lhs: 2, rhs: 2 }));
        assert!(matches!(ops[4], FusedOpKind::Add { lhs: 3, rhs: 0 }));
        assert!(matches!(ops[5], FusedOpKind::Relu { input: 4 }));
        // 最後のエントリが出力ノード（モジュール冒頭「出力ノードの契約」）。
        assert_eq!(ops.len() - 1, 5);
        // x（leaf 0）は Add(a) と Add(c) の 2 か所から参照される（fan-out）。
        assert_eq!(plan.use_count(0), 2);
    }

    #[test]
    fn from_segment_fan_in_confluence() {
        // (a+b)*(c+d) 形の fan-in（4 入力・3 elementwise ノード + 1 段
        // 足して 4 段にする）。
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let c = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let d = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let ab = g.push(FusionOp::Add(a, b), f32_meta(&[4])).unwrap();
        let cd = g.push(FusionOp::Add(c, d), f32_meta(&[4])).unwrap();
        let prod = g.push(FusionOp::Mul(ab, cd), f32_meta(&[4])).unwrap();
        let out = g.push(FusionOp::Relu(prod), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, out).unwrap();
        let segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };
        let plan = FusionPlan::from_segment(&g, &segment).unwrap();
        assert_eq!(plan.leaf_count(), 4);
        let ops: Vec<FusedOpKind> = plan.ops().collect();
        assert_eq!(ops.len(), 8); // 4 leaves + ab + cd + prod + out
        assert!(matches!(ops[7], FusedOpKind::Relu { .. }));
    }

    /// レビュー指摘 #400（P1）: `segment` が `graph` と不整合な場合
    /// （呼び出し元のバグにより、あるノードが参照するオペランドが
    /// `segment.leaves`／`segment.nodes` のいずれにも含まれない）でも
    /// `from_segment` が panic せず [`FusionGraphError::
    /// DanglingOperandReference`] を返すことを固定する。従来は
    /// `index_of[&id]` の直接添字アクセスで panic していた。
    #[test]
    fn from_segment_rejects_dangling_operand_reference() {
        // (a+b)*(c+d) 形の fan-in（`from_segment_fan_in_confluence` と
        // 同型。連鎖長 4 で最小連鎖長を満たし Fuse 判定される）。
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let c = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let d = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let ab = g.push(FusionOp::Add(a, b), f32_meta(&[4])).unwrap();
        let cd = g.push(FusionOp::Add(c, d), f32_meta(&[4])).unwrap();
        let prod = g.push(FusionOp::Mul(ab, cd), f32_meta(&[4])).unwrap();
        let out = g.push(FusionOp::Relu(prod), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, out).unwrap();
        let mut segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };
        // `segment` を人為的に `graph` と不整合にする: `ab` が参照する
        // `a` を leaves から取り除く（呼び出し元のバグを模擬）。
        segment.leaves.retain(|id| *id != a);

        let err = FusionPlan::from_segment(&g, &segment).unwrap_err();
        assert_eq!(err, FusionGraphError::DanglingOperandReference { id: a.0 });
    }

    #[test]
    fn from_ops_rejects_forward_reference() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 5 }, // まだ存在しない位置を参照
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::IndexOutOfRange { index: 5, at: 1 });
    }

    #[test]
    fn from_ops_rejects_leaf_index_out_of_range() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 3 }, // leaf_count=1 の範囲外
                FusedOpKind::Relu { input: 0 },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FusionPlanError::LeafIndexOutOfRange {
                leaf_index: 3,
                leaf_count: 1
            }
        );
    }

    /// レビュー指摘 #400（P2）: `leaf_count` 個数一致・範囲チェックのみ
    /// では `leaf_index` の重複（`leaf_count=2` に対し `leaf_index: 0`
    /// を 2 個並べ `leaf_index: 1` を欠落させる入力）を検出できない。
    #[test]
    fn from_ops_rejects_duplicate_leaf_index() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Input { leaf_index: 0 }, // leaf_index: 1 が欠落・0 が重複
                FusedOpKind::Relu { input: 0 },
            ],
            vec![4],
            DType::F32,
            2,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::DuplicateLeafIndex { leaf_index: 0 });
    }

    #[test]
    fn from_ops_rejects_input_count_mismatch() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
            ],
            vec![4],
            DType::F32,
            2, // leaf_count=2 だが Input エントリは 1 個のみ
        )
        .unwrap_err();
        assert_eq!(
            err,
            FusionPlanError::LeafCountMismatch {
                declared: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn from_ops_rejects_input_only_plan() {
        let err = FusionPlan::from_ops(
            vec![FusedOpKind::Input { leaf_index: 0 }],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::NoElementwiseNode);
    }

    #[test]
    fn from_ops_rejects_trailing_input_after_elementwise_node() {
        // has_elementwise は満たすが、末尾エントリが Input のまま
        // （出力ノードの契約〈モジュール冒頭〉違反）。
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
                FusedOpKind::Input { leaf_index: 1 },
            ],
            vec![4],
            DType::F32,
            2,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::LastOpIsInput);
    }

    #[test]
    fn from_ops_rejects_output_shape_overflow() {
        // レビュー指摘 #163 の反例: 要素数積が usize::MAX を超える
        // `output_shape` を渡しても、他の検査（トポロジカル順・
        // leaf_count 一致・出力ノード契約）はすべて素通りしてしまう
        // ため、`from_ops` 単体でオーバーフロー検査を行わない限り
        // `FusionPlan` が不正な型不変条件を持ったまま構築できてしまう。
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
            ],
            vec![usize::MAX, 2],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::OutputShapeOverflow);
    }

    #[test]
    fn from_ops_accepts_valid_chain() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Input { leaf_index: 1 },
                FusedOpKind::Add { lhs: 0, rhs: 1 },
                FusedOpKind::Relu { input: 2 },
            ],
            vec![4],
            DType::F32,
            2,
        )
        .unwrap();
        assert_eq!(plan.leaf_count(), 2);
        assert_eq!(plan.output_shape(), &[4]);
        assert_eq!(plan.dtype(), DType::F32);
        assert_eq!(plan.use_count(2), 1);
        assert_eq!(plan.use_count(0), 1);
        // 範囲外は 0 を返す。
        assert_eq!(plan.use_count(999), 0);
    }
}
