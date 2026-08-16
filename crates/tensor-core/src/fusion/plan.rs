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
/// `pub(crate)` の [`FusionOp`]（`graph.rs`・§2.1）と 1:1 対応する。
/// `Gemm` のみが常に融合境界ノードのため設計書 §3.2 (b) のとおり
/// `FusionPlan` 内に現れない（実体化境界のため、融合対象区間そのものに
/// は含まれない）。`Sum`／`Max`（reduction）はイシュー #586 の境界
/// 再定義により、セグメント軸が一致する限り `FusionPlan` 内に現れうる
/// （`graph.rs`・`detect.rs` の doc 参照。CPU カーネル実装自体は本イシュー
/// のスコープ外＝ G-3 以降であり、`backend-cpu::fused_elementwise` は
/// これらを含む plan を pre-scan で `BackendError::Unsupported` として
/// fail-closed に拒否する）。フィールドは [`FusedNodeIndex`]（plain
/// `usize`）のみで構成し、`pub(crate)` 型を一切参照しない（設計書 §3.4
/// 「privacy 制約」）。
///
/// # 後方互換性（codex-review PR #648 P1 是正）
///
/// `tensor-core` は `publish = false`（workspace `Cargo.toml`）かつ
/// `docs/compat-api-scope.md` §0 が定める**内部クレート**であり、`facade`
/// のみが「サポートされる公開 API 面」である（`facade::compat` に
/// `FusedOpKind` は再エクスポートされない）。よって本 enum が Rust の
/// 可視性として `pub`（`backend-cpu`／`autodiff` からのクレート間参照に
/// 必要なため）であることと、外部利用者向けにサポートされる公開面で
/// あることは区別する（`docs/compat-api-scope.md` §0 で `autodiff::
/// Tape::new_with_ops` 等に対し既に確立済みの区別と同じ整理）。
/// それでも本 workspace 内の `backend-cpu`／`autodiff` の各クレートは
/// `FusedOpKind` を跨クレートで exhaustive match するため、`#[non_exhaustive]`
/// を付けて「variant 追加は非破壊」という前方互換性を型で保証する
/// （バリアント単位の `#[non_exhaustive]` ではなく enum 全体へ付与:
/// バリアント単位だと `FusedOpKind::Add { .. }` 等のクレート外構築
/// 〈`autodiff/src/tape.rs`〉自体が壊れるため）。呼び出し側の match は
/// 必ず `_` 分岐を持つ（`backend-cpu::fused_elementwise::eval_one` 等）。
#[non_exhaustive]
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
    /// 減算（`lhs - rhs`）。softmax の `x - max(x)` を表現するために
    /// #588 で追加。`graph.rs::FusionOp::Sub` と 1:1 対応。
    Sub {
        lhs: FusedNodeIndex,
        rhs: FusedNodeIndex,
    },
    /// 除算（`lhs / rhs`）。softmax の `exp(..) / sum(..)` を表現するため
    /// #588 で追加。`graph.rs::FusionOp::Div` と 1:1 対応。
    Div {
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
    /// `1/sqrt(x)`。RMSNorm 融合（#586）の構成要素。`graph.rs::
    /// FusionOp::Rsqrt` と 1:1 対応。
    Rsqrt {
        input: FusedNodeIndex,
    },
    /// reduction（縮約）。`axis: None` は全軸縮約、`axis: Some(a)` は
    /// 指定軸のみの縮約（`graph.rs::FusionOp::Sum` の `dim` フィールドを
    /// この公開 DTO の命名規約〈`axis`〉へ写像する。#586 イシュー指定の
    /// フィールド名）。
    Sum {
        input: FusedNodeIndex,
        axis: Option<usize>,
    },
    /// [`FusedOpKind::Sum`] と同じ写像規約に従う reduction。
    Max {
        input: FusedNodeIndex,
        axis: Option<usize>,
    },
    /// 縮約済みテンソル `input` をセグメント軸に沿って元の行 shape へ
    /// 論理拡張する（#588）。`graph.rs::FusionOp::Broadcast` と 1:1
    /// 対応し、`dim`→`axis` の写像規約は `Sum`／`Max` と同一。
    Broadcast {
        input: FusedNodeIndex,
        axis: Option<usize>,
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
    /// `ops` が空、または `Input` エントリのみで構成される（`Input` 以外
    /// のノードが 1 個も無い＝融合する意味がない。variant 名は公開面の
    /// ため維持するが、#586 で reduction／`Rsqrt` も対象へ加わったため
    /// 「非 `Input` ノードが 1 個も無い」の意味へ拡張する。命名は互換性
    /// のため据え置く）。
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
    /// `ops` 内の `Sum`／`Max` エントリの `axis` が、セグメント内で最初に
    /// 現れた reduction が確定した軸と一致しない（#586・`detect.rs` の
    /// セグメント軸一致判定〈グラフ構築経路〉と同じ不変条件を、`ops`
    /// を直接受け取る `from_ops` 経路でも防御的に検証する。`from_ops`
    /// は `autodiff` から任意の `Vec<FusedOpKind>` を受け取る公開経路
    /// であり、`detect_fusion` の判定を経由しないため、この検証を欠くと
    /// 軸不一致の reduction が混在した不正な `FusionPlan` が構築できて
    /// しまう）。`#[non_exhaustive]` のため後方互換に新規追加できる。
    MismatchedReductionAxis {
        at: usize,
        expected: Option<usize>,
        actual: Option<usize>,
    },
    /// `ops` 内の `Broadcast` エントリの `axis` が、セグメント内で最初に
    /// 現れた reduction／broadcast が確定した軸と一致しない（#588。
    /// [`MismatchedReductionAxis`](Self::MismatchedReductionAxis) と対称の
    /// 検証だが、`detect.rs::detect_fusion` が reduction／broadcast を
    /// 同じセグメント軸判定の対象として扱う一方（`graph.rs::FusionOp::
    /// reduction_dim`／`broadcast_dim` の対）、`Sum`／`Max` の軸不一致と
    /// `Broadcast` の軸不一致を呼び出し元が区別して診断できるよう別
    /// variant にする。`#[non_exhaustive]` のため後方互換に新規追加
    /// できる）。
    MismatchedBroadcastAxis {
        at: usize,
        expected: Option<usize>,
        actual: Option<usize>,
    },
    /// [`FusionPlan::row_fusion`]（#588・§3.5）が導出する行融合メタデータ
    /// の軸が `output_shape` の rank 範囲外（`axis: Some(a)` で
    /// `a >= output_shape.len()`）。`compute_row_fusion`（本モジュール
    /// 下部の共通ヘルパー）が返す fail-closed 検証エラー。
    RowAxisOutOfRange { axis: usize, rank: usize },
    /// `ops` が `Broadcast` を 1 個以上含むにもかかわらず、末尾（出力）
    /// ノードが reduction（`Sum`／`Max`）である——すなわち出力 shape が
    /// 行 shape へ復元されないまま終端する（#588 実装計画 §3.5）。
    /// RMSNorm／softmax のように「broadcast で行 shape へ戻してから
    /// 終端する」パターンのみを本イシューでは表現可能とする安全側の
    /// 制約であり、broadcast 後に再縮約して終端するプランは表現不能
    /// として fail-closed に拒否する（緩和は将来イシュー。#588 実装計画
    /// §9「スコープ外・申し送り」）。`at` は末尾ノードのインデックス。
    ReducedOutputWithBroadcast { at: usize },
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
            FusionPlanError::MismatchedReductionAxis {
                at,
                expected,
                actual,
            } => write!(
                f,
                "fusion plan node {at} reduction axis {actual:?} does not match segment axis {expected:?}"
            ),
            FusionPlanError::MismatchedBroadcastAxis {
                at,
                expected,
                actual,
            } => write!(
                f,
                "fusion plan node {at} broadcast axis {actual:?} does not match segment axis {expected:?}"
            ),
            FusionPlanError::RowAxisOutOfRange { axis, rank } => write!(
                f,
                "fusion plan row-fusion axis {axis} out of range (output rank={rank})"
            ),
            FusionPlanError::ReducedOutputWithBroadcast { at } => write!(
                f,
                "fusion plan node {at} (last op) is a reduction while the plan contains Broadcast; \
                 broadcast-containing plans must terminate with a row-shaped output"
            ),
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
            // #586: 軸不一致は「型不変条件違反」というより「この
            // FusionPlan は融合として成立しない」という意味論のため、
            // 実測値の丸め対象である ShapeMismatch 系ではなく
            // `NoElementwiseNode`／`LastOpIsInput` と同じ `Unsupported`
            // へ寄せる（`err.to_string()` で `at`/`expected`/`actual` を
            // 保持したメッセージのまま伝播する）。
            // #588: `MismatchedBroadcastAxis`／`ReducedOutputWithBroadcast`
            // は `MismatchedReductionAxis` と同じ「融合として成立しない」
            // 意味論のため `Unsupported` へ寄せる。`RowAxisOutOfRange` は
            // 行融合メタデータ導出時点の shape 不整合であり
            // `OutputShapeOverflow` と同種の型不変条件違反のため
            // `ShapeMismatch` 系へ寄せる。
            FusionPlanError::MismatchedReductionAxis { .. }
            | FusionPlanError::MismatchedBroadcastAxis { .. }
            | FusionPlanError::ReducedOutputWithBroadcast { .. } => {
                BackendError::Unsupported(err.to_string())
            }
            FusionPlanError::RowAxisOutOfRange { axis, rank } => {
                BackendError::ShapeMismatch(ShapeError::AxisOutOfRange { axis, rank })
            }
        }
    }
}

/// 「1 セグメントが 1 行を保持できるか」の境界値（#588）。TileKernels
/// `swiglu_forward_and_per_token_cast_kernel.py:30-42` の実測（非信頼
/// データ・イシュー本文の参照実装の要約。`TILE_X == 1 and hidden <= 8192`
/// のとき 1 CTA が 1 トークンの全 hidden をレジスタ常駐できる）を、融合
/// softmax／RMSNorm のタイル形状決定則としてそのまま転用した値。**あくまで
/// ヒント**であり、バックエンド（CUDA／Metal／CPU）は `row_len` から自前の
/// smem／レジスタ予算で再判定してよい（#588 実装計画 §3.5）。ガードレール
/// 閾値・テスト許容誤差ではない実装定数のためユーザー承認は要さないが、
/// [`crate::fusion::detect::MAX_FUSED_CHAIN_LEN`] と同様 G-8 計測（#602 等）
/// で見直し可能な形で定数化しておく。
pub const MAX_SINGLE_PASS_ROW_LEN: usize = 8192;

/// 行方向 reduction＋broadcast 融合プランの行メタデータ（#588・受け入れ
/// 基準 2）。「行方向 reduction → 派生スカラー（rstd／max／sum）→ 同一行へ
/// broadcast 適用」という 2 パス構造を持つプラン（[`FusionPlan::row_fusion`]
/// が `Some` を返すプラン）について、1 パス実装（1 実行単位が行全体を
/// レジスタ常駐できる）と 2 パス実装（行を分割して複数回走査する）の
/// どちらをバックエンドが選ぶべきかの判断材料を提供する。フィールドは
/// private とし、下記アクセサ経由でのみ読み取れる（`FusionPlan` 本体と
/// 同じ「`tensor-core` 外からは構築・分解できない」設計方針）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowFusionMeta {
    axis: Option<usize>,
    row_len: usize,
    single_pass_hint: bool,
}

impl RowFusionMeta {
    /// セグメント軸（`FusedOpKind::Sum`／`Max`／`Broadcast` の `axis` と
    /// 同じ値。`None` は全軸縮約を表す）。
    pub fn axis(&self) -> Option<usize> {
        self.axis
    }

    /// 行長（`axis: Some(a)` なら `output_shape[a]`、`axis: None`
    /// （全縮約）なら `output_shape` の要素数積）。
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// `row_len <= MAX_SINGLE_PASS_ROW_LEN` のヒント値（§本モジュール
    /// `MAX_SINGLE_PASS_ROW_LEN` の doc 参照。あくまでヒントであり
    /// バックエンドが自前の予算で再判定してよい）。
    pub fn single_pass_hint(&self) -> bool {
        self.single_pass_hint
    }
}

/// [`RowFusionMeta`] の導出（`from_segment`／`from_ops` の共通ヘルパー。
/// #588 実装計画 §3.5「二重実装しない」）。`ops` が `Broadcast` を 1 個も
/// 含まなければ `Ok(None)`（#586 までの reduction-only プランは後方互換で
/// `None` のまま。§3.5）。含む場合は fail-closed 検証（§3.6）を行った上で
/// `Ok(Some(_))` を返す。
///
/// # エラー
///
/// - [`FusionPlanError::RowAxisOutOfRange`]: セグメント軸 `Some(a)` が
///   `output_shape` の rank 範囲外。
/// - [`FusionPlanError::ReducedOutputWithBroadcast`]: `ops` の末尾
///   （出力ノード）が reduction（`Sum`／`Max`）——broadcast で行 shape へ
///   復元し終端する RMSNorm／softmax 型のみを表現可能とする安全側の制約
///   （§3.5・§9「スコープ外・申し送り」）。
fn compute_row_fusion(
    ops: &[FusedOpKind],
    output_shape: &[usize],
) -> Result<Option<RowFusionMeta>, FusionPlanError> {
    // セグメント軸: `ops` 内で最初に現れた Sum／Max／Broadcast の
    // `axis`（`from_segment`〈`detect.rs` のセグメント軸一致判定〉・
    // `from_ops`〈本モジュール `check_preceding` 呼び出し元〉のいずれの
    // 経路でも、この時点までに全 reduction／broadcast エントリが同一軸を
    // 持つことが既に検証済みのため、最初の 1 個を読めば十分。
    let axis = ops.iter().find_map(|op| match *op {
        FusedOpKind::Sum { axis, .. }
        | FusedOpKind::Max { axis, .. }
        | FusedOpKind::Broadcast { axis, .. } => Some(axis),
        _ => None,
    });

    let has_broadcast = ops
        .iter()
        .any(|op| matches!(op, FusedOpKind::Broadcast { .. }));
    if !has_broadcast {
        return Ok(None);
    }

    // `ops` は空ではない契約（`from_segment`／`from_ops` いずれも
    // `Input` 以外を最低 1 個含むことを別途検証済み）。
    let last_at = ops.len() - 1;
    if matches!(
        ops[last_at],
        FusedOpKind::Sum { .. } | FusedOpKind::Max { .. }
    ) {
        return Err(FusionPlanError::ReducedOutputWithBroadcast { at: last_at });
    }

    // `axis` は `has_broadcast` が真である以上、上の `find_map` で必ず
    // `Some` になる（`Broadcast` エントリ自身がこの `find_map` に
    // 一致するため）。
    let axis = axis.unwrap_or(None);

    let row_len = match axis {
        Some(a) => {
            let rank = output_shape.len();
            *output_shape
                .get(a)
                .ok_or(FusionPlanError::RowAxisOutOfRange { axis: a, rank })?
        }
        // 全縮約（`axis: None`）: 行長は出力テンソルの要素数積
        // （`RowFusionMeta::row_len` doc 参照）。`output_shape` は
        // 呼び出し元（`from_segment`: 実テンソルの shape・`from_ops`:
        // `OutputShapeOverflow` 検査済み）でオーバーフローしないことが
        // 既に確定しているため、素朴な乗算で求める。
        None => output_shape.iter().product(),
    };

    Ok(Some(RowFusionMeta {
        axis,
        row_len,
        single_pass_hint: row_len <= MAX_SINGLE_PASS_ROW_LEN,
    }))
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
    /// 行方向 reduction＋broadcast 融合の行メタデータ（#588・§3.5）。
    /// `ops` が `Broadcast` を 1 個以上含むプランのみ `Some`（#586 までの
    /// reduction-only プランは `None` のまま後方互換を保つ）。
    row_fusion: Option<RowFusionMeta>,
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
    /// `segment.nodes` が `Input`／`Gemm`（#586 以降も常に境界）を含む
    /// 場合の [`FusionGraphError::UnexpectedOpKind`]、または Add／Mul／
    /// Relu／Exp／Tanh／Rsqrt／Sum／Max のオペランドが再番号付け表に
    /// 存在しない場合の [`FusionGraphError::DanglingOperandReference`]
    /// を返す（いずれも
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
            // 満たすノード、および #586 でセグメント軸が一致する
            // reduction〈`Sum`／`Max`〉のみを `included` へ挿入する。
            // `detect.rs` 参照）が確定した集合であるため、ここに現れる
            // `op` は常に「常に境界」の `Input`／`Gemm` 以外である。
            // それでも `graph`（#399 の教訓どおり `pub(crate)` であっても
            // 未検証入力を panic で処理しない）と `segment` の整合を
            // ここでも型付きエラーで扱う（呼び出し元のバグにより不整合な
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
                // #588: `Sub`／`Div` は `Add`／`Mul` と同じオペランド解決に
                // 相乗りする。
                FusionOp::Sub(a, b) => FusedOpKind::Sub {
                    lhs: lookup_index(&index_of, a.0)?,
                    rhs: lookup_index(&index_of, b.0)?,
                },
                FusionOp::Div(a, b) => FusedOpKind::Div {
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
                FusionOp::Rsqrt(a) => FusedOpKind::Rsqrt {
                    input: lookup_index(&index_of, a.0)?,
                },
                // #586: reduction（Sum／Max）はセグメント軸が一致する限り
                // `detect_fusion` がセグメントへ組み込むため（`detect.rs`
                // 参照）、ここに現れうる。`dim`→`axis` の命名写像のみ行う
                // （検証は `detect_fusion` が構築済みで、ここでは再検証
                // しない。`segment` と `graph` が不整合な場合の入力参照
                // 解決は他の枝と同じ `lookup_index` に委ねる）。
                FusionOp::Sum { input: a, dim } => FusedOpKind::Sum {
                    input: lookup_index(&index_of, a.0)?,
                    axis: *dim,
                },
                FusionOp::Max { input: a, dim } => FusedOpKind::Max {
                    input: lookup_index(&index_of, a.0)?,
                    axis: *dim,
                },
                // #588: `Broadcast` も reduction と同じくセグメント軸が
                // 一致する限り `detect_fusion` がセグメントへ組み込む
                // ため、ここに現れうる。`dim`→`axis` の写像規約は
                // `Sum`／`Max` と同一。
                FusionOp::Broadcast { input: a, dim } => FusedOpKind::Broadcast {
                    input: lookup_index(&index_of, a.0)?,
                    axis: *dim,
                },
                FusionOp::Input | FusionOp::Gemm(..) => {
                    // 到達しない防御的分岐（上記コメント参照）。専用の
                    // `FusionGraphError::UnexpectedOpKind` を返す
                    // （`NodeIdOutOfRange` は「ID が範囲外」という別の
                    // 不変条件の違反を表すため意味論的に転用しない。
                    // レビュー指摘 #163）。`Gemm`／`Input` のみが常に
                    // 境界であり `segment.nodes` に現れえない（#586）。
                    return Err(FusionGraphError::UnexpectedOpKind { id: node_id.0 });
                }
            };
            ops.push(kind);
        }

        let root_node = graph.node(segment.root)?;
        let output_shape = root_node.meta.shape.clone();
        let dtype = root_node.meta.dtype;

        let use_counts = compute_use_counts(&ops);
        // #588 §3.5: `compute_row_fusion` のエラーは detect_fusion／push
        // の検証を通過した通常経路では到達しない防御的検証だが、戻り値型
        // を `FusionGraphError` に統一するため `FusionGraphError::Plan`
        // （`graph.rs` の `From<FusionPlanError>` 実装）へ `?` で合流させる。
        let row_fusion = compute_row_fusion(&ops, &output_shape)?;

        Ok(FusionPlan {
            ops,
            output_shape,
            dtype,
            leaf_count,
            use_counts,
            row_fusion,
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
    /// - `Sum`／`Max` エントリの `axis` が、`ops` 内で最初に現れた
    ///   reduction／broadcast の `axis` と一致すること（#586。
    ///   `detect_fusion` のセグメント軸一致判定〈`detect.rs`〉と同じ
    ///   不変条件を `ops` を直接受け取るこの経路でも検証する。不一致は
    ///   [`FusionPlanError::MismatchedReductionAxis`]）
    /// - `Broadcast` エントリの `axis` が同じセグメント軸と一致すること
    ///   （#588。不一致は [`FusionPlanError::MismatchedBroadcastAxis`]）
    /// - [`compute_row_fusion`] の検証（#588・§3.5）: `Broadcast` を含む
    ///   場合、セグメント軸が `output_shape` の rank 範囲内であること
    ///   （[`FusionPlanError::RowAxisOutOfRange`]）、末尾（出力）ノードが
    ///   reduction でないこと（[`FusionPlanError::ReducedOutputWithBroadcast`]）
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
        // 変数名は据え置くが #586 以降「`Input` 以外のノードを 1 個以上
        // 含むか」（elementwise ∪ reduction ∪ Rsqrt）を表す（上記
        // `NoElementwiseNode` doc 参照）。
        let mut has_elementwise = false;
        // `leaf_index` ごとの出現済みフラグ（レビュー指摘 #400:
        // `leaf_count` 個数一致・範囲チェックだけでは重複した
        // `leaf_index` を見逃す。`0..leaf_count` へ一度ずつ出現する
        // ことをここで確定する）。
        let mut leaf_seen = vec![false; leaf_count];
        // セグメント軸（#586・#588 で broadcast にも適用範囲を拡張。
        // `detect.rs::detect_fusion` の `segment_axis` と同じ役割を
        // `ops` の直接検証側でも担う）。
        let mut segment_axis: Option<Option<usize>> = None;
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
                FusedOpKind::Add { lhs, rhs }
                | FusedOpKind::Mul { lhs, rhs }
                | FusedOpKind::Sub { lhs, rhs }
                | FusedOpKind::Div { lhs, rhs } => {
                    has_elementwise = true;
                    check_preceding(lhs, at)?;
                    check_preceding(rhs, at)?;
                }
                FusedOpKind::Relu { input }
                | FusedOpKind::Exp { input }
                | FusedOpKind::Tanh { input }
                | FusedOpKind::Rsqrt { input } => {
                    has_elementwise = true;
                    check_preceding(input, at)?;
                }
                FusedOpKind::Sum { input, axis } | FusedOpKind::Max { input, axis } => {
                    has_elementwise = true;
                    check_preceding(input, at)?;
                    let expected = *segment_axis.get_or_insert(axis);
                    if expected != axis {
                        return Err(FusionPlanError::MismatchedReductionAxis {
                            at,
                            expected,
                            actual: axis,
                        });
                    }
                }
                // #588: `Broadcast` も reduction と同じセグメント軸一致
                // 判定に参加する（「Input 以外を最低 1 個含む」判定への
                // 寄与も reduction と同様）。軸不一致は reduction とは
                // 別 variant（`MismatchedBroadcastAxis`）で診断性を保つ。
                FusedOpKind::Broadcast { input, axis } => {
                    has_elementwise = true;
                    check_preceding(input, at)?;
                    let expected = *segment_axis.get_or_insert(axis);
                    if expected != axis {
                        return Err(FusionPlanError::MismatchedBroadcastAxis {
                            at,
                            expected,
                            actual: axis,
                        });
                    }
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
        let row_fusion = compute_row_fusion(&ops, &output_shape)?;

        Ok(FusionPlan {
            ops,
            output_shape,
            dtype,
            leaf_count,
            use_counts,
            row_fusion,
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

    /// 行方向 reduction＋broadcast 融合の行メタデータ（#588・受け入れ
    /// 基準 2）。`ops` が `Broadcast` を 1 個以上含むプランのみ `Some`
    /// を返す（#586 までの reduction-only プランは `None`。後方互換）。
    /// バックエンド（`run_fused` 実装）はこれを読んで 1 パス／2 パスの
    /// カーネル実装を選択できる（`RowFusionMeta::single_pass_hint`
    /// doc 参照）。
    pub fn row_fusion(&self) -> Option<&RowFusionMeta> {
        self.row_fusion.as_ref()
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
            FusedOpKind::Add { lhs, rhs }
            | FusedOpKind::Mul { lhs, rhs }
            | FusedOpKind::Sub { lhs, rhs }
            | FusedOpKind::Div { lhs, rhs } => {
                counts[lhs] += 1;
                counts[rhs] += 1;
            }
            FusedOpKind::Relu { input }
            | FusedOpKind::Exp { input }
            | FusedOpKind::Tanh { input }
            | FusedOpKind::Rsqrt { input }
            | FusedOpKind::Sum { input, .. }
            | FusedOpKind::Max { input, .. }
            | FusedOpKind::Broadcast { input, .. } => {
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
        // `dim: None`（全軸縮約）の出力 shape 契約は rank 0（`[]`。#586
        // で構築時検証を追加。`graph.rs::FusionGraph::push` 参照）。
        let _sum = g
            .push(
                FusionOp::Sum {
                    input: a,
                    dim: None,
                },
                f32_meta(&[]),
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

    /// #586: `detect_fusion` がセグメント軸一致で reduction を組み込んだ
    /// 場合、`from_segment` が `dim`→`axis` の命名写像を保ったまま
    /// `FusedOpKind::Sum`／`Rsqrt` へ変換することを固定する（RMSNorm 様
    /// 連鎖: `Mul(x,x) → Sum{None} → Rsqrt → Mul`）。
    #[test]
    fn from_segment_converts_reduction_and_rsqrt_with_axis_mapping() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let sq = g.push(FusionOp::Mul(x, x), f32_meta(&[4])).unwrap();
        let sum = g
            .push(
                FusionOp::Sum {
                    input: sq,
                    dim: None,
                },
                f32_meta(&[]),
            )
            .unwrap();
        let rsqrt = g.push(FusionOp::Rsqrt(sum), f32_meta(&[])).unwrap();
        let out = g.push(FusionOp::Mul(rsqrt, rsqrt), f32_meta(&[])).unwrap();

        let decision = detect_fusion(&g, out).unwrap();
        let segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };

        let plan = FusionPlan::from_segment(&g, &segment).unwrap();
        let ops: Vec<FusedOpKind> = plan.ops().collect();
        // leaf 0=x, 1=sq(Mul(0,0)), 2=sum(Sum{axis:None}(1)), 3=rsqrt(2), 4=out(Mul(3,3))
        assert_eq!(ops.len(), 5);
        assert!(matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }));
        assert!(matches!(ops[1], FusedOpKind::Mul { lhs: 0, rhs: 0 }));
        assert!(matches!(
            ops[2],
            FusedOpKind::Sum {
                input: 1,
                axis: None
            }
        ));
        assert!(matches!(ops[3], FusedOpKind::Rsqrt { input: 2 }));
        assert!(matches!(ops[4], FusedOpKind::Mul { lhs: 3, rhs: 3 }));
        // sum（プラン内 index 2）は Rsqrt から 1 回参照される。
        assert_eq!(plan.use_count(2), 1);
    }

    /// #586: `from_ops` は `ops` 内で最初に現れた reduction の `axis` を
    /// セグメント軸として確定し、以降一致する `Sum`／`Max` を受理する。
    #[test]
    fn from_ops_accepts_matching_reduction_axis() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
                FusedOpKind::Sum {
                    input: 1,
                    axis: Some(0),
                },
                FusedOpKind::Exp { input: 2 },
                FusedOpKind::Max {
                    input: 3,
                    axis: Some(0),
                },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap();
        assert_eq!(plan.leaf_count(), 1);
        assert_eq!(plan.use_count(1), 1);
    }

    /// 受け入れ基準（#586）の `from_ops` 側固定: `axis` が一致しない
    /// `Sum`／`Max` を混在させると [`FusionPlanError::
    /// MismatchedReductionAxis`] で拒否される。
    #[test]
    fn from_ops_rejects_mismatched_reduction_axis() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
                FusedOpKind::Sum {
                    input: 1,
                    axis: Some(0),
                },
                FusedOpKind::Exp { input: 2 },
                FusedOpKind::Max {
                    input: 3,
                    axis: Some(1),
                },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FusionPlanError::MismatchedReductionAxis {
                at: 4,
                expected: Some(0),
                actual: Some(1),
            }
        );
    }

    /// #586: `Rsqrt` 単独でも `from_ops` の「非 `Input` ノードを最低 1 個
    /// 含む」判定（`NoElementwiseNode` の拡張された意味論）を満たす。
    #[test]
    fn from_ops_accepts_rsqrt_only_chain() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Rsqrt { input: 0 },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap();
        assert_eq!(plan.use_count(0), 1);
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 3」）: `from_segment`
    /// で RMSNorm プランを構築すると `row_fusion()` が
    /// `Some`（axis: None・row_len: 8・single_pass_hint: true）になる
    /// （`detect.rs::rmsnorm_pattern_with_explicit_broadcast_is_a_single_segment`
    /// と同型のグラフ）。
    #[test]
    fn from_segment_builds_rmsnorm_plan_with_row_fusion_metadata() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let sq = g.push(FusionOp::Mul(x, x), f32_meta(&[8])).unwrap();
        let sum = g
            .push(
                FusionOp::Sum {
                    input: sq,
                    dim: None,
                },
                f32_meta(&[]),
            )
            .unwrap();
        let rsqrt = g.push(FusionOp::Rsqrt(sum), f32_meta(&[])).unwrap();
        let bc = g
            .push(
                FusionOp::Broadcast {
                    input: rsqrt,
                    dim: None,
                },
                f32_meta(&[8]),
            )
            .unwrap();
        let out = g.push(FusionOp::Mul(bc, x), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, out).unwrap();
        let segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };
        let plan = FusionPlan::from_segment(&g, &segment).unwrap();
        let ops: Vec<FusedOpKind> = plan.ops().collect();
        // leaf 0=x, 1=sq(Mul(0,0)), 2=sum(Sum{None}(1)), 3=rsqrt(2),
        // 4=bc(Broadcast{None}(3)), 5=out(Mul(4,0))
        assert_eq!(ops.len(), 6);
        assert!(matches!(
            ops[4],
            FusedOpKind::Broadcast {
                input: 3,
                axis: None
            }
        ));
        let row_fusion = plan.row_fusion().expect("row_fusion must be Some");
        assert_eq!(row_fusion.axis(), None);
        assert_eq!(row_fusion.row_len(), 8);
        assert!(row_fusion.single_pass_hint());
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 3」）: `from_segment`
    /// で softmax プラン（7 ops + 葉 1）を構築すると `row_fusion()` が
    /// `Some`（axis: Some(1)・row_len: 8・single_pass_hint: true）になる
    /// （`detect.rs::softmax_pattern_with_seven_nodes_is_a_single_segment`
    /// と同型のグラフ）。
    #[test]
    fn from_segment_builds_softmax_plan_with_row_fusion_metadata() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[2, 8])).unwrap();
        let mx = g
            .push(
                FusionOp::Max {
                    input: x,
                    dim: Some(1),
                },
                f32_meta(&[2]),
            )
            .unwrap();
        let bc1 = g
            .push(
                FusionOp::Broadcast {
                    input: mx,
                    dim: Some(1),
                },
                f32_meta(&[2, 8]),
            )
            .unwrap();
        let sub = g.push(FusionOp::Sub(x, bc1), f32_meta(&[2, 8])).unwrap();
        let exp = g.push(FusionOp::Exp(sub), f32_meta(&[2, 8])).unwrap();
        let sm = g
            .push(
                FusionOp::Sum {
                    input: exp,
                    dim: Some(1),
                },
                f32_meta(&[2]),
            )
            .unwrap();
        let bc2 = g
            .push(
                FusionOp::Broadcast {
                    input: sm,
                    dim: Some(1),
                },
                f32_meta(&[2, 8]),
            )
            .unwrap();
        let div = g.push(FusionOp::Div(exp, bc2), f32_meta(&[2, 8])).unwrap();

        let decision = detect_fusion(&g, div).unwrap();
        let segment = match decision {
            FusionDecision::Fuse(seg) => seg,
            other => panic!("expected Fuse, got {other:?}"),
        };
        let plan = FusionPlan::from_segment(&g, &segment).unwrap();
        assert_eq!(plan.leaf_count(), 1);
        let ops: Vec<FusedOpKind> = plan.ops().collect();
        assert_eq!(ops.len(), 8); // 1 leaf + mx + bc1 + sub + exp + sm + bc2 + div
        assert!(matches!(ops[7], FusedOpKind::Div { .. }));
        let row_fusion = plan.row_fusion().expect("row_fusion must be Some");
        assert_eq!(row_fusion.axis(), Some(1));
        assert_eq!(row_fusion.row_len(), 8);
        assert!(row_fusion.single_pass_hint());
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 3」）: `from_ops`
    /// でも RMSNorm パターンを構築でき、`row_fusion()` が
    /// `from_segment` と同じ結果になる（autodiff 経路の対称性）。
    #[test]
    fn from_ops_builds_rmsnorm_plan_with_row_fusion_metadata() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Mul { lhs: 0, rhs: 0 },
                FusedOpKind::Sum {
                    input: 1,
                    axis: None,
                },
                FusedOpKind::Rsqrt { input: 2 },
                FusedOpKind::Broadcast {
                    input: 3,
                    axis: None,
                },
                FusedOpKind::Mul { lhs: 4, rhs: 0 },
            ],
            vec![8],
            DType::F32,
            1,
        )
        .unwrap();
        let row_fusion = plan.row_fusion().expect("row_fusion must be Some");
        assert_eq!(row_fusion.axis(), None);
        assert_eq!(row_fusion.row_len(), 8);
        assert!(row_fusion.single_pass_hint());
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 3」）: `from_ops`
    /// でも softmax パターンを構築できる（autodiff 経路の対称性）。
    #[test]
    fn from_ops_builds_softmax_plan_with_row_fusion_metadata() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Max {
                    input: 0,
                    axis: Some(1),
                },
                FusedOpKind::Broadcast {
                    input: 1,
                    axis: Some(1),
                },
                FusedOpKind::Sub { lhs: 0, rhs: 2 },
                FusedOpKind::Exp { input: 3 },
                FusedOpKind::Sum {
                    input: 4,
                    axis: Some(1),
                },
                FusedOpKind::Broadcast {
                    input: 5,
                    axis: Some(1),
                },
                FusedOpKind::Div { lhs: 4, rhs: 6 },
            ],
            vec![2, 8],
            DType::F32,
            1,
        )
        .unwrap();
        let row_fusion = plan.row_fusion().expect("row_fusion must be Some");
        assert_eq!(row_fusion.axis(), Some(1));
        assert_eq!(row_fusion.row_len(), 8);
        assert!(row_fusion.single_pass_hint());
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 2」）: `row_len ==
    /// MAX_SINGLE_PASS_ROW_LEN`（8192）ちょうどは `single_pass_hint` が
    /// `true`（境界を含む側）。
    #[test]
    fn row_fusion_single_pass_hint_true_at_exact_boundary() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Broadcast {
                    input: 0,
                    axis: None,
                },
            ],
            vec![MAX_SINGLE_PASS_ROW_LEN],
            DType::F32,
            1,
        )
        .unwrap();
        let row_fusion = plan.row_fusion().expect("row_fusion must be Some");
        assert_eq!(row_fusion.row_len(), MAX_SINGLE_PASS_ROW_LEN);
        assert!(row_fusion.single_pass_hint());
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 2」）: `row_len ==
    /// MAX_SINGLE_PASS_ROW_LEN + 1`（8193）は `single_pass_hint` が
    /// `false`（境界の外側）。
    #[test]
    fn row_fusion_single_pass_hint_false_just_above_boundary() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Broadcast {
                    input: 0,
                    axis: None,
                },
            ],
            vec![MAX_SINGLE_PASS_ROW_LEN + 1],
            DType::F32,
            1,
        )
        .unwrap();
        let row_fusion = plan.row_fusion().expect("row_fusion must be Some");
        assert_eq!(row_fusion.row_len(), MAX_SINGLE_PASS_ROW_LEN + 1);
        assert!(!row_fusion.single_pass_hint());
    }

    /// 受け入れ基準（#586 との後方互換。#588 実装計画 §3.5）: `Broadcast`
    /// を含まない reduction-only プランは `row_fusion() == None` のまま。
    #[test]
    fn row_fusion_is_none_for_reduction_only_plan() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
                FusedOpKind::Sum {
                    input: 1,
                    axis: Some(0),
                },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap();
        assert!(plan.row_fusion().is_none());
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 4」）: `Broadcast`
    /// の軸がセグメント軸と一致しない場合、`from_ops` は
    /// [`FusionPlanError::MismatchedBroadcastAxis`] で fail-closed に
    /// 拒否する（`from_ops_rejects_mismatched_reduction_axis` と対称）。
    #[test]
    fn from_ops_rejects_mismatched_broadcast_axis() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Sum {
                    input: 0,
                    axis: Some(0),
                },
                FusedOpKind::Broadcast {
                    input: 1,
                    axis: Some(1),
                },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(
            err,
            FusionPlanError::MismatchedBroadcastAxis {
                at: 2,
                expected: Some(0),
                actual: Some(1),
            }
        );
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 4」）: `Broadcast`
    /// のセグメント軸が `output_shape` の rank 範囲外の場合、
    /// [`FusionPlanError::RowAxisOutOfRange`] で拒否される。
    #[test]
    fn from_ops_rejects_row_axis_out_of_range() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Broadcast {
                    input: 0,
                    axis: Some(5),
                },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::RowAxisOutOfRange { axis: 5, rank: 1 });
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 4」）: `Broadcast`
    /// を含むにもかかわらず末尾（出力）ノードが reduction（`Sum`／`Max`）
    /// の場合、[`FusionPlanError::ReducedOutputWithBroadcast`] で
    /// fail-closed に拒否される（broadcast 後に再縮約して終端するプラン
    /// は #588 のスコープ外。実装計画 §9）。
    #[test]
    fn from_ops_rejects_reduced_output_with_broadcast() {
        let err = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Broadcast {
                    input: 0,
                    axis: None,
                },
                FusedOpKind::Sum {
                    input: 1,
                    axis: None,
                },
            ],
            vec![4],
            DType::F32,
            1,
        )
        .unwrap_err();
        assert_eq!(err, FusionPlanError::ReducedOutputWithBroadcast { at: 2 });
    }

    /// #588: `use_count` 集計に `Sub`／`Div`／`Broadcast` が正しく寄与する
    /// （softmax パターンで `exp`〈プラン内 index 4〉が `Sum` と `Div`
    /// の両方から参照されることを固定する）。
    #[test]
    fn from_ops_use_count_includes_sub_div_broadcast() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Max {
                    input: 0,
                    axis: Some(1),
                },
                FusedOpKind::Broadcast {
                    input: 1,
                    axis: Some(1),
                },
                FusedOpKind::Sub { lhs: 0, rhs: 2 },
                FusedOpKind::Exp { input: 3 },
                FusedOpKind::Sum {
                    input: 4,
                    axis: Some(1),
                },
                FusedOpKind::Broadcast {
                    input: 5,
                    axis: Some(1),
                },
                FusedOpKind::Div { lhs: 4, rhs: 6 },
            ],
            vec![2, 8],
            DType::F32,
            1,
        )
        .unwrap();
        // exp（index 4）は Sum（index 5）と Div（index 7）の 2 か所から
        // 参照される。
        assert_eq!(plan.use_count(4), 2);
    }
}
