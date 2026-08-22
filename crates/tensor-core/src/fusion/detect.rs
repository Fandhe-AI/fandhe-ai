//! elementwise 連鎖検出（融合判定）アルゴリズム（TASK-12.1b・#162）。
//!
//! `docs/fusion-graph-design.md` §6.1「#162（連鎖検出）」が確定する
//! スコープ:「§2（グラフ表現・ノード種別・メタデータ・fan-out）を用いた
//! 融合可能連鎖（elementwise のみで閉じた 4〜6 段の連結成分）の検出
//! アルゴリズム」を実装する。`graph.rs` の [`FusionGraph`] を読み取り
//! 専用で走査するのみで、グラフを変更しない・グローバル状態を持たない
//! **副作用なしの純関数**とする（`dispatch::select_gemm_kernel` と同じ
//! 決定性の方針。設計書 §3.4）。
//!
//! [`FusionPlan`]（融合カーネル生成向け公開 DTO）・実際のカーネル実行
//! （`BackendOps::run_fused`）は本イシューのスコープ外であり、#163／
//! #164 が担当する（設計書 §6.1）。本モジュールは「融合すべきか・
//! どこまでか」の判定結果（[`FusionDecision`]）のみを返す。

use super::graph::{FusionGraph, FusionGraphError, FusionNodeId, FusionOp};
use crate::dispatch::DType;

/// elementwise 連鎖長の上限（TASK-12.1 の内容規定「4〜6 段程度」の
/// 上限値。PoC-9 `ew6` 実測と整合。設計書 §3.2 (d)）。
///
/// 連鎖がこの段数に達すると、それより深いノードへの走査を打ち切り、
/// 打ち切り位置のノードの入力を外部入力（葉）として扱う
/// （`detect_fusion` のドキュメント参照）。ガードレール閾値・テスト
/// 許容誤差ではなく実装判断の定数のため変更にユーザー承認は要さないが、
/// TASK-12.2（#166・実測）で見直し可能な形で定数化しておく
/// （実装計画イシュー #162 §8「リスク・判断の記録」）。
///
/// **単一真実源（#404）**: 本定数は `detect_fusion`（本モジュール）に
/// 加え、`fandhe_ai_autodiff::tape::Tape::push_lazy` の push 時上限適用（設計書
/// §3.5.4）からも参照される。`crate::fusion::mod.rs` の `pub use` を
/// 経由してクレートルート（`fandhe_ai_tensor_core::MAX_FUSED_CHAIN_LEN`）から
/// 公開する（`autodiff` → `tensor-core` の依存方向のみで完結し、逆依存
/// を作らない）。値の重複定義を避けるため、遅延評価経路側で同名の
/// 定数を再定義しないこと。
///
/// **#588 での意味論の精密化**: 本定数は **elementwise ノード数のみ**
/// に適用する（`FusionOp::is_elementwise()` を満たすノード。reduction・
/// broadcast は含まない）。softmax パターン（`Max → Broadcast → Sub →
/// Exp → Sum → Broadcast → Div` の 7 ノード）は elementwise が 3 個
/// （`Sub`／`Exp`／`Div`）のみのため、この上限の下で全 7 ノードが単一
/// セグメントに収まる。reduction／broadcast は行スカラー 1 個分の
/// レジスタしか消費せず、elementwise 中間列とはレジスタ圧のコスト構造が
/// 異なるため elementwise 数のみを数える対象とする判断（#588 実装計画
/// §3.3）。総ノード数の暴走防止は [`MAX_FUSED_SEGMENT_NODES`] が別途担う。
pub const MAX_FUSED_CHAIN_LEN: usize = 6;

/// 融合セグメントの総ノード数上限（#588）。
///
/// [`MAX_FUSED_CHAIN_LEN`] が elementwise ノード数のみに適用される
/// 意味論へ精密化されたことに伴い、reduction／broadcast を含めた総数の
/// 暴走を防ぐために新設する決定的打ち切り上限（挿入前判定。既存の
/// `MAX_FUSED_CHAIN_LEN` 挿入前判定パターンをそのまま踏襲する）。
/// `2 × MAX_FUSED_CHAIN_LEN` を初期値とする（RMSNorm・softmax のような
/// 「reduction/broadcast が elementwise と概ね同数以下で交互に現れる」
/// 典型パターンを 1 セグメントに収めつつ、無制限な連鎖成長を防ぐための
/// 実装判断の定数。ガードレール閾値・テスト許容誤差ではないためユーザー
/// 承認は要さないが、[`MAX_FUSED_CHAIN_LEN`] と同様 TASK-12.2 系の実測
/// （#602 等）で見直し可能な形で定数化しておく。#588 実装計画 §3.3）。
pub const MAX_FUSED_SEGMENT_NODES: usize = 2 * MAX_FUSED_CHAIN_LEN;

/// 融合セグメント成立に要する最小 elementwise ノード数。
///
/// `docs/fusion-graph-design.md` §1・本モジュールの初期スコープ規定
/// 「elementwise 演算連鎖（4〜6 段程度）」（TASK-12.1 の内容規定。
/// `docs/spec/05-tasks.md:370`）の下限側に合わせ、[`MAX_FUSED_CHAIN_LEN`]
/// と対になる下限値として 4 を要求する（#399 codex-review 指摘: 2 段
/// からの融合を許すと初期スコープの「4〜6 段程度」と矛盾する）。
/// 2〜3 段の短い連鎖はカーネル呼び出し削減効果が小さく PoC-9 実測の
/// 対象外であるため非融合フォールバックとする。
pub(crate) const MIN_FUSED_CHAIN_LEN: usize = 4;

/// [`detect_fusion`] が融合しないと判定した理由。呼び出し側（#163 の
/// `FusionPlan::from_graph` 相当・#165 のテスト）がフォールバック経路を
/// 判別できるよう、`Fallback` の中身を理由付き enum として持たせる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FallbackReason {
    /// 走査の起点（root）自体が elementwise 5 演算ではない
    /// （`Input`／`Gemm`／`Sum`／`Max`）。`detect_fusion` は融合境界
    /// ノードそのものを起点に呼ばれるべきではなく、呼び出し側の
    /// 呼び出し方の誤りを表す（設計書 §3.2 (a)(b) の境界ノードは
    /// セグメントに含めない方針の裏返し）。
    RootNotElementwise,
    /// セグメント内に `contiguous == false`（transpose／broadcast view）
    /// のノードを検出した（設計書 §2.3・§3.2 (e)）。
    NonContiguous,
    /// セグメント内に `dtype != F32` のノードを検出した（`BackendOps`
    /// が f32 固定スコープであることに対応する防御的判定。設計書 §2.1）。
    UnsupportedDtype,
    /// 検出された elementwise 連結成分のノード数が [`MIN_FUSED_CHAIN_LEN`]
    /// 未満（融合効果のない単一ノード等）。
    ChainTooShort,
}

/// 検出された融合可能セグメント（設計書 §6.1 「#163 が `FusionPlan::
/// from_graph` 相当で読む」ことを想定した中間結果）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FusionSegment {
    /// セグメントに含まれる elementwise ノード ID（発生順＝トポロジカル
    /// 順に昇順ソート済み）。
    pub(crate) nodes: Vec<FusionNodeId>,
    /// セグメント外部から読み込む葉ノード ID（外部入力。`Input`・融合
    /// 境界ノード〈`Gemm`／`Sum`／`Max`〉・連鎖長上限による打ち切り位置
    /// の入力のいずれか。昇順ソート済み・重複なし）。
    pub(crate) leaves: Vec<FusionNodeId>,
    /// 走査の起点（セグメントの出力ノード）。
    pub(crate) root: FusionNodeId,
}

/// [`detect_fusion`] の判定結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FusionDecision {
    /// 融合可能: 検出されたセグメントを返す。
    Fuse(FusionSegment),
    /// 融合不能: 理由を返す（セグメント全体が非融合フォールバックとなり、
    /// `ops` の per-op メソッドを記録順に逐次呼び出す経路〈設計書 §2.5〉
    /// へ委ねられる想定。フォールバック経路の実装自体は #164 のスコープ）。
    Fallback(FallbackReason),
}

/// `root` を出力とする融合可能な elementwise／reduction 連鎖を検出する
/// （設計書 §3.2 の実体化条件・§6.1 #162 のスコープに対応する本体。
/// イシュー #586 で reduction（`Sum`／`Max`）をセグメントへ組み込める
/// よう境界判定を再定義した）。
///
/// # アルゴリズム
///
/// `root` から入力方向へ**後方走査**する。ノード ID は発生順（追記順）
/// で、`FusionGraph::push` の不変条件により入力 ID は常に自ノードより
/// 小さいため、`root.0` から `0` へ向けた降順の単純な線形スキャンだけで
/// 「参照済みノードをそれより手前で必ず処理し終えている」ことが保証
/// される（キュー・スタックによる探索順の非決定性を排除できる。本関数
/// が純関数・決定的であるための構造上の根拠）。
///
/// 1. 到達可能集合 `reachable` を `{root}` で初期化する。セグメント軸
///    `segment_axis`（`Option<Option<usize>>`。外側 `None` は「まだ
///    reduction／broadcast を含んでいない」、内側は reduction／broadcast
///    の `dim`）を `None` で初期化する。elementwise 数の別カウンタ
///    `elementwise_count` を 0 で初期化する（#588。「#588 での意味論の
///    精密化」参照）。
/// 2. `id` を `root.0` から `0` まで降順に走査し、`reachable` に含まれる
///    ノードのみを処理する:
///    - elementwise（`Add`／`Mul`／`Sub`／`Div`／`Relu`／`Exp`／`Tanh`／
///      `Rsqrt`）なら:
///      - `dtype != F32` または `contiguous == false` を検出した時点で
///        走査全体を打ち切り [`FusionDecision::Fallback`] を返す
///        （設計書 §3.2 (e)・セグメント全体を非融合にする方針）。
///      - `elementwise_count` が [`MAX_FUSED_CHAIN_LEN`] に達しているか
///        `included.len()`（総数）が [`MAX_FUSED_SEGMENT_NODES`] に
///        **達していれば**、このノードはセグメントへ追加せず・入力も
///        `reachable` へ展開しない（＝それ自身が打ち切り境界となり、
///        後段の葉抽出で外部入力として扱われる。設計書 §3.2 (d)）。
///        この判定は「`included` へ挿入する前」に行う。fan-in で同一
///        ノードが複数経路から `reachable` に入りうるため、挿入後に
///        上限を検査すると別経路から先に `reachable` 入りしていた
///        ノードが上限到達後も処理され各カウンタが上限を超過しうる
///        （#162 レビュー指摘）。挿入前判定であれば各カウンタは走査順・
///        到達経路によらず常に上限以下に収まる。
///      - 上限未到達ならセグメントへ追加し（`elementwise_count` も
///        加算）、このノードの入力を `reachable` へ追加して走査を継続
///        する。
///    - reduction（`Sum`／`Max`）または broadcast（`Broadcast`。#588）
///      なら（#586 の境界再定義・#588 で broadcast へ拡張）:
///      - `dtype != F32` または `contiguous == false` は elementwise と
///        同様に走査全体を打ち切る。
///      - `segment_axis` が未確定（`None`）なら、このノードの `dim` を
///        セグメント軸として確定する（`Option<usize>::get_or_insert`。
///        「セグメント内で最初に含まれる reduction／broadcast の `dim`
///        がセグメント軸を確定する」の実体。降順走査のため root に近い
///        側＝走査で最初に到達するノードが優先される——softmax パターン
///        では `Broadcast` が `Sum`／`Max` より root 側に現れるため、
///        `Broadcast` が先にセグメント軸を確定し、後続（深部）の
///        `Sum`／`Max` がそれと一致する必要がある）。
///      - `dim` が `segment_axis` と一致すれば、`included.len()`（総数）
///        が [`MAX_FUSED_SEGMENT_NODES`] 未満である限りセグメントへ
///        組み込む（`elementwise_count` は加算しない。#588 の意味論
///        精密化により reduction／broadcast は elementwise 数上限の
///        対象外）。
///      - `dim` が `segment_axis` と**一致しなければ**、このノードは
///        セグメントへ組み込まず・入力も展開しない（＝境界ノードとして
///        葉になる。受け入れ基準「reduction 軸が一致しない連鎖は分断
///        される」の実体。broadcast も同じ扱い）。
///    - 融合境界（`Gemm`）または `Input` なら、それ自体を入力方向へは
///      展開せず、そのノード ID を後段で葉として扱う（設計書 §3.2
///      (a)(b)）。ただしこの時点でも `dtype != F32` または
///      `contiguous == false` を検出した時点で走査全体を打ち切り
///      [`FusionDecision::Fallback`] を返す（`graph.rs` の
///      `FusionGraph::push` の binary shape 検証コメントが明言する契約: broadcast
///      view は `push` 時点では拒否せず `contiguous: false` として本
///      関数の非融合フォールバック判定に委ねられる。境界ノードだから
///      といって検証を素通りさせない）。軸不一致で境界化した
///      reduction／broadcast ノードもこの分岐と同じ検証（`dtype`／
///      `contiguous`）を受ける。
/// 3. 走査完了後、セグメントに含まれるノードが 1 個でも `dtype`／
///    `contiguous` 違反で打ち切られていなければ、セグメントの各ノードが
///    参照する入力のうちセグメントに含まれないものを葉として集約する
///    （重複除去・昇順ソート）。
/// 4. セグメントのノード数が [`MIN_FUSED_CHAIN_LEN`] 未満なら
///    [`FallbackReason::ChainTooShort`] で非融合とする（この判定は
///    従来どおり `included.len()`〈総数〉で行う。#588 実装計画 §3.3）。
///
/// **root は従来どおり elementwise のみ許可**（reduction root は
/// [`FallbackReason::RootNotElementwise`] のまま。RMSNorm／softmax の
/// 実用セグメントは elementwise で終端するため必要十分であり、既存挙動
/// の回帰面を最小化する安全側の判断。将来必要になれば別イシューで
/// 拡張する。#586 実装計画 §3.3）。
///
/// fan-out（同一ノードが複数ノードから参照される）・fan-in（複数の
/// elementwise 連鎖が 1 ノードへ合流する `(a+b)*(c+d)` 形）はいずれも
/// `reachable`（集合）と降順走査により自然に扱える。fan-out はそれ自体
/// 融合不能条件にしない（設計書 §2.4・PoC-9 `ew_fanout` 実測根拠）。
///
/// # エラー
///
/// `root` が `graph` の範囲外（既存ノード数以上）の場合は
/// [`FusionGraphError::NodeIdOutOfRange`] を返す（#399 codex-review 指摘:
/// 本関数は `pub(crate)` だが、後続の本番結線コード（#163／#164）から
/// 任意の `FusionNodeId` を渡され得る。範囲検証は `FusionGraph::node`
/// （`graph.rs`）に一元化されており、本関数はそれをそのまま `?` で
/// 伝播するのみ。呼び出し側のバグ検出という当初の意図は型付きエラー
/// として表現し直し、release ビルドでも同じ検証を維持する）。
pub(crate) fn detect_fusion(
    graph: &FusionGraph,
    root: FusionNodeId,
) -> Result<FusionDecision, FusionGraphError> {
    if !graph.node(root)?.op.is_elementwise() {
        return Ok(FusionDecision::Fallback(FallbackReason::RootNotElementwise));
    }

    // 到達可能（＝セグメント候補として展開されうる）ノード ID の集合。
    // `BTreeSet` を使うのは走査ロジック自体には無関係（`contains` の
    // みに使う）だが、内部状態も含め非決定的なイテレーション順に依存
    // しない実装であることを明示するため統一する。
    use std::collections::BTreeSet;
    let mut reachable: BTreeSet<usize> = BTreeSet::new();
    reachable.insert(root.0);

    // セグメントに含まれる elementwise／reduction／broadcast ノード
    // （発見順。降順走査のため root に近い側から入る。最終的に昇順へ
    // ソートし直す）。
    let mut included: BTreeSet<usize> = BTreeSet::new();

    // セグメントに含まれる elementwise ノード数（#588: `included.len()`
    // 〈総数〉とは別に数える。`MAX_FUSED_CHAIN_LEN` は elementwise 数のみ
    // に適用する意味論へ精密化されたため。関数doc「#588 での意味論の
    // 精密化」参照）。
    let mut elementwise_count: usize = 0;

    // セグメント軸（#586・#588 で broadcast にも適用範囲を拡張）。`None`
    // は「まだ reduction／broadcast を含んでいない」、`Some(dim)` は
    // 確定済みのセグメント軸（`dim` は reduction／broadcast 自体の
    // `Option<usize>`）。最初に組み込まれた reduction／broadcast がこれを
    // 確定する（関数doc「アルゴリズム」節参照）。
    let mut segment_axis: Option<Option<usize>> = None;

    for id in (0..=root.0).rev() {
        if !reachable.contains(&id) {
            continue;
        }
        let node = graph.node(FusionNodeId(id))?;

        if node.op.is_elementwise() {
            if node.meta.dtype != DType::F32 {
                return Ok(FusionDecision::Fallback(FallbackReason::UnsupportedDtype));
            }
            if !node.meta.contiguous {
                return Ok(FusionDecision::Fallback(FallbackReason::NonContiguous));
            }

            // 上限判定は `included` へ挿入する**前**に行う。fan-in を含む
            // DAG では同一ノードが複数経路から先に `reachable` へ入りうる
            // ため、挿入後に上限を検査すると別経路由来のノードが上限到達後
            // も処理されてしまい `included.len()` が上限を超過する（#162
            // レビュー指摘の反例: バランス木 fan-in）。挿入前判定なら
            // 各カウンタは到達経路・走査順によらず常に上限以下に収まり、
            // 上限到達後のノードはそのまま外部入力（葉）として後段の葉
            // 抽出に委ねられる（設計書 §3.2 (d)）。elementwise は
            // [`MAX_FUSED_CHAIN_LEN`]（elementwise 数のみに適用。#588）と
            // [`MAX_FUSED_SEGMENT_NODES`]（総数。#588）の両方を満たす
            // 必要がある。
            if elementwise_count >= MAX_FUSED_CHAIN_LEN || included.len() >= MAX_FUSED_SEGMENT_NODES
            {
                continue;
            }
            included.insert(id);
            elementwise_count += 1;
            for input in node.op.inputs() {
                reachable.insert(input.0);
            }
            continue;
        }

        // reduction（Sum／Max）と broadcast（#588）は同じセグメント軸
        // 一致判定を共有する（reduction が軸を確定する、broadcast が
        // その軸で組み込まれる、あるいはその逆——降順走査のため softmax
        // パターンでは root 側の `Broadcast` が先に軸を確定し、後続
        // （深部）の `Sum`／`Max` がそれと一致する必要がある。#588
        // 実装計画 §3.4）。総数上限のみを適用し（elementwise 数上限は
        // 適用しない。#588 の意味論精密化）、`MAX_FUSED_CHAIN_LEN` は
        // 検査しない。
        let dim = node.op.reduction_dim().or_else(|| node.op.broadcast_dim());
        if let Some(dim) = dim {
            if node.meta.dtype != DType::F32 {
                return Ok(FusionDecision::Fallback(FallbackReason::UnsupportedDtype));
            }
            if !node.meta.contiguous {
                return Ok(FusionDecision::Fallback(FallbackReason::NonContiguous));
            }

            // `get_or_insert` は未確定なら `dim` で確定して `&mut` を返し、
            // 確定済みならその既存値への `&mut` を返す。いずれの場合も
            // 比較先は「確定済みのセグメント軸」になるため、初回は必ず
            // 一致（このノードが軸を確定する）、2 回目以降は既存軸との
            // 一致判定になる。
            let axis_matches = *segment_axis.get_or_insert(dim) == dim;
            if !axis_matches {
                // 軸不一致: セグメントへ組み込まず・入力も展開しない
                // （境界ノードとして葉になる。受け入れ基準「reduction
                // 軸が一致しない連鎖は分断される」の実体。broadcast も
                // 同じ扱い）。
                continue;
            }

            if included.len() >= MAX_FUSED_SEGMENT_NODES {
                continue;
            }
            included.insert(id);
            for input in node.op.inputs() {
                reachable.insert(input.0);
            }
            continue;
        }

        // 融合境界（Gemm）または Input。ここでは展開せず、葉抽出（後段）
        // に委ねる。ただし葉として使われる以上、`graph.rs` が明言する
        // 「broadcast view は contiguous: false として非融合判定側
        // 〈本関数〉に委ねる」契約の受け手としてここで検証する（#162
        // レビュー指摘: 境界ノードの contiguous/dtype が未検証だった）。
        if node.meta.dtype != DType::F32 {
            return Ok(FusionDecision::Fallback(FallbackReason::UnsupportedDtype));
        }
        if !node.meta.contiguous {
            return Ok(FusionDecision::Fallback(FallbackReason::NonContiguous));
        }
    }

    if included.len() < MIN_FUSED_CHAIN_LEN {
        return Ok(FusionDecision::Fallback(FallbackReason::ChainTooShort));
    }

    // 葉抽出: セグメントに含まれる各ノードの入力のうち、セグメントに
    // 含まれないものを外部入力として集約する（重複除去は BTreeSet が
    // 兼ねる。昇順は BTreeSet のイテレーション順そのもの）。
    let mut leaves: BTreeSet<usize> = BTreeSet::new();
    for &id in &included {
        let node = graph.node(FusionNodeId(id))?;
        for input in node.op.inputs() {
            if !included.contains(&input.0) {
                leaves.insert(input.0);
            }
        }
    }

    Ok(FusionDecision::Fuse(FusionSegment {
        nodes: included.into_iter().map(FusionNodeId).collect(),
        leaves: leaves.into_iter().map(FusionNodeId).collect(),
        root,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::graph::NodeMeta;

    fn f32_meta(shape: &[usize]) -> NodeMeta {
        NodeMeta::new(shape.to_vec(), true, DType::F32)
    }

    fn non_contiguous_meta(shape: &[usize]) -> NodeMeta {
        NodeMeta::new(shape.to_vec(), false, DType::F32)
    }

    /// PoC-9 `ew4` 相当: 4 段連鎖（Add→Relu→Exp→Tanh）を構成して
    /// 「4〜6 段程度」の下限（[`MIN_FUSED_CHAIN_LEN`]）ちょうどが
    /// 融合可能と判定されることを検証する。
    #[test]
    fn detects_four_stage_elementwise_chain_as_single_segment() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let w = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let n1 = g.push(FusionOp::Add(x, w), f32_meta(&[8])).unwrap();
        let n2 = g.push(FusionOp::Relu(n1), f32_meta(&[8])).unwrap();
        let n3 = g.push(FusionOp::Exp(n2), f32_meta(&[8])).unwrap();
        let n4 = g.push(FusionOp::Tanh(n3), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, n4).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.root, n4);
        assert_eq!(seg.nodes, vec![n1, n2, n3, n4]);
        assert_eq!(seg.leaves, vec![x, w]);
    }

    /// PoC-9 `ew6` 相当: ちょうど上限（6 段）は打ち切りなしで検出される。
    #[test]
    fn detects_six_stage_chain_at_exactly_the_cap() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let mut prev = x;
        let mut chain = Vec::new();
        for _ in 0..6 {
            let n = g.push(FusionOp::Relu(prev), f32_meta(&[8])).unwrap();
            chain.push(n);
            prev = n;
        }
        let root = *chain.last().unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes.len(), MAX_FUSED_CHAIN_LEN);
        assert_eq!(seg.nodes, chain);
        assert_eq!(seg.leaves, vec![x]);
    }

    /// 7 段以上は上限で決定的に打ち切られ、打ち切り境界の入力が葉として
    /// 扱われる（設計書 §3.2 (d)）。
    #[test]
    fn seven_stage_chain_is_cut_off_deterministically_at_cap() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let mut prev = x;
        let mut chain = Vec::new();
        for _ in 0..7 {
            let n = g.push(FusionOp::Relu(prev), f32_meta(&[8])).unwrap();
            chain.push(n);
            prev = n;
        }
        let root = *chain.last().unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes.len(), MAX_FUSED_CHAIN_LEN);
        // root から数えて 6 段（chain[1..=6]、末尾 6 要素）のみが
        // セグメントに含まれる。打ち切り境界である chain[0] は
        // chain[1]（セグメントに含まれる）の入力として参照されるため
        // 未使用ノードにはならず、セグメント外部の入力＝葉として
        // 扱われる（これが「6 段ちょうど」を root 側から数える仕様の
        // 帰結）。
        assert_eq!(&seg.nodes, &chain[1..]);
        assert_eq!(seg.leaves, vec![chain[0]]);
    }

    /// PoC-9 `ew_fanout` 相当: `a = x + y; b = a * a; c = b + x; d = relu(c)`。
    /// fan-out（a が 2 回参照される）は融合不能条件にしない
    /// （設計書 §2.4）。`d = relu(c)` はテストのための水増しではなく、
    /// `docs/spec/03-poc/poc-9-kernel-fusion/README.md:46` が定義する
    /// `ew_fanout` パターン本体（`a=x+y; b=a*a; c=b+x; sigmoid(c)`）の
    /// 4 段目 `sigmoid(c)` に対応する（`FusionOp` は `Sigmoid` を
    /// 持たないため、同じく非 fallible な単項 elementwise 演算である
    /// `Relu` で代替する）。設計書 §2.4 の引用（`a = x + y; b = a * a;
    /// c = b + x` まで）は「`a` が 2 回消費される」点を説明するための
    /// 省略引用であり、実際の PoC-9 パターン自体が既に 4 段構成である
    /// ため、本テストは [`MIN_FUSED_CHAIN_LEN`] の初期スコープ内に
    /// 収まる（#399 codex-review 指摘への回答、Bugbot 指摘への回答:
    /// 3 段版 `Add→Mul→Add` を融合対象にする設計要求は存在しない。
    /// その反例テストは
    /// [`three_stage_fanout_chain_falls_back_below_minimum_length`] を
    /// 参照）。
    #[test]
    fn fanout_chain_is_detected_as_single_segment_with_correct_use_count() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let a = g.push(FusionOp::Add(x, y), f32_meta(&[8])).unwrap();
        let b = g.push(FusionOp::Mul(a, a), f32_meta(&[8])).unwrap();
        let c = g.push(FusionOp::Add(b, x), f32_meta(&[8])).unwrap();
        let d = g.push(FusionOp::Relu(c), f32_meta(&[8])).unwrap();

        assert_eq!(
            g.node(a).unwrap().use_count,
            2,
            "a は b と c から参照される"
        );

        let decision = detect_fusion(&g, d).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![a, b, c, d]);
        assert_eq!(seg.leaves, vec![x, y]);
    }

    /// Bugbot 指摘（PR #399 review thread）への直接の回答: 設計書 §2.4 の
    /// `a = x + y; b = a * a; c = b + x`（`sigmoid(c)` を含まない省略
    /// 引用）をそのまま 3 段の `Add→Mul→Add` として構成した場合は
    /// [`MIN_FUSED_CHAIN_LEN`]（4）未満であり非融合フォールバックとなる
    /// ことを明示的に固定する。これは回帰ではなく、TASK-12.1 の内容規定
    /// 「elementwise 演算連鎖（4〜6 段程度）を初期スコープとする」
    /// （`docs/spec/05-tasks.md:370`・`docs/fusion-graph-design.md:15`）
    /// の下限側そのものである。実際の PoC-9 `ew_fanout` パターン本体は
    /// `docs/spec/03-poc/poc-9-kernel-fusion/README.md:46` のとおり
    /// `sigmoid(c)` を含む 4 段構成であり、
    /// [`fanout_chain_is_detected_as_single_segment_with_correct_use_count`]
    /// がその 4 段版（`Relu` で `Sigmoid` 代替）を融合可能として検証済み
    /// のため、3 段版が非融合になっても設計上要求されるパターンの取りこぼしにはならない。
    #[test]
    fn three_stage_fanout_chain_falls_back_below_minimum_length() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let a = g.push(FusionOp::Add(x, y), f32_meta(&[8])).unwrap();
        let b = g.push(FusionOp::Mul(a, a), f32_meta(&[8])).unwrap();
        let c = g.push(FusionOp::Add(b, x), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, c).unwrap();
        assert_eq!(
            decision,
            FusionDecision::Fallback(FallbackReason::ChainTooShort)
        );
    }

    /// fan-in: `(a + b) * (c + d)` 形の合流が単一セグメントとして検出
    /// される（`FusionGraph` は DAG 一般をサポートするため fan-in も
    /// 通常ケース。設計書 §6.2「同一テープ内での遅延グラフの合流は
    /// 正規サポート対象」）。elementwise 連鎖の初期スコープ（4〜6 段。
    /// `docs/fusion-graph-design.md:15`・`docs/spec/05-tasks.md:370`
    /// TASK-12.1）に収まる構成にするため `relu(root)` を末尾に追加して
    /// いる（[`MIN_FUSED_CHAIN_LEN`] を満たすための水増しではなく、
    /// スコープ内の代表例として構成している）。
    #[test]
    fn fanin_confluence_is_detected_as_single_segment() {
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let c = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let d = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let left = g.push(FusionOp::Add(a, b), f32_meta(&[8])).unwrap();
        let right = g.push(FusionOp::Add(c, d), f32_meta(&[8])).unwrap();
        let mul = g.push(FusionOp::Mul(left, right), f32_meta(&[8])).unwrap();
        let root = g.push(FusionOp::Relu(mul), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![left, right, mul, root]);
        assert_eq!(seg.leaves, vec![a, b, c, d]);
    }

    /// #162 レビュー指摘の反例: バランス木 fan-in で `MAX_FUSED_CHAIN_LEN`
    /// が破られないことを検証する（`x=Input; n1..n4=Relu(x)×4;
    /// n5=Add(n1,n2); n6=Add(n3,n4); n7=Add(n5,n6)=root`）。
    ///
    /// 降順走査では n7→n6→n5→n4→n3→n2 の時点で `included.len()==6`
    /// （上限到達）になるが、n1 は n5 処理時点（`included.len()==3`）で
    /// 既に `reachable` へ追加済みのため、挿入後に上限を検査する実装
    /// だと n1 も追加されてしまい `included.len()==7` に達し上限超過と
    /// なる。挿入前判定であれば n1 は上限到達後の処理対象になっても
    /// `included` へ追加されず、外部入力（葉）として扱われる。
    #[test]
    fn fanin_balanced_tree_does_not_exceed_chain_len_cap() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let n1 = g.push(FusionOp::Relu(x), f32_meta(&[8])).unwrap();
        let n2 = g.push(FusionOp::Relu(x), f32_meta(&[8])).unwrap();
        let n3 = g.push(FusionOp::Relu(x), f32_meta(&[8])).unwrap();
        let n4 = g.push(FusionOp::Relu(x), f32_meta(&[8])).unwrap();
        let n5 = g.push(FusionOp::Add(n1, n2), f32_meta(&[8])).unwrap();
        let n6 = g.push(FusionOp::Add(n3, n4), f32_meta(&[8])).unwrap();
        let n7 = g.push(FusionOp::Add(n5, n6), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, n7).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert!(
            seg.nodes.len() <= MAX_FUSED_CHAIN_LEN,
            "MAX_FUSED_CHAIN_LEN を超過した: {} 個",
            seg.nodes.len()
        );
        assert_eq!(seg.nodes, vec![n2, n3, n4, n5, n6, n7]);
        assert_eq!(seg.leaves, vec![x, n1]);
    }

    /// #162 レビュー指摘: 葉ノード（融合境界／`Input`）の `contiguous`
    /// が検証されず、transpose／broadcast view がそのまま `Fuse` として
    /// 通過してしまう不具合の再発防止。`graph.rs` の
    /// `FusionGraph::push` の binary shape 検証コメントが明言する「broadcast view は
    /// `contiguous: false` として本関数の非融合フォールバックに委ねる」
    /// 契約を、セグメント内部ノードだけでなく葉ノードでも満たすこと
    /// を検証する。
    #[test]
    fn non_contiguous_leaf_input_falls_back_the_whole_segment() {
        let mut g = FusionGraph::new();
        // transpose/broadcast view 相当（contiguous = false）の葉。
        let x = g.push(FusionOp::Input, non_contiguous_meta(&[4])).unwrap();
        let n1 = g.push(FusionOp::Relu(x), f32_meta(&[4])).unwrap();
        let root = g.push(FusionOp::Exp(n1), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        assert_eq!(
            decision,
            FusionDecision::Fallback(FallbackReason::NonContiguous)
        );
    }

    /// PoC-9 `ew_matmul_ew` 相当: Gemm 境界でセグメントが分断される
    /// （設計書 §3.2 (b)）。elementwise 連鎖の初期スコープ（4〜6 段。
    /// `docs/fusion-graph-design.md:15`・`docs/spec/05-tasks.md:370`
    /// TASK-12.1）に収まる構成にするため `tanh`／2 個目の `relu` を
    /// 末尾に追加している。
    #[test]
    fn gemm_boundary_splits_the_segment() {
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let gemm = g.push(FusionOp::Gemm(a, b), f32_meta(&[4, 4])).unwrap();
        let relu = g.push(FusionOp::Relu(gemm), f32_meta(&[4, 4])).unwrap();
        let exp = g.push(FusionOp::Exp(relu), f32_meta(&[4, 4])).unwrap();
        let tanh = g.push(FusionOp::Tanh(exp), f32_meta(&[4, 4])).unwrap();
        let relu2 = g.push(FusionOp::Relu(tanh), f32_meta(&[4, 4])).unwrap();

        let decision = detect_fusion(&g, relu2).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        // Gemm 自体はセグメントに含まれず、その出力が葉として扱われる。
        assert_eq!(seg.nodes, vec![relu, exp, tanh, relu2]);
        assert_eq!(seg.leaves, vec![gemm]);
    }

    /// #586 で境界を再定義する前は Sum／Max がセグメントを常に分断して
    /// いたが、再定義後は「セグメント軸が一致する reduction」はセグメント
    /// へ組み込まれる（`detect_fusion` doc 「アルゴリズム」節）。本テストは
    /// 旧仕様（`Sum` が常に境界）の固定テストを新仕様（単一 `Sum` はセグ
    /// メント軸を確定しつつそのまま組み込まれる）へ書き換えたもの——
    /// `relu`→`sum{dim=Some(0)}`→`exp`→`tanh`→`relu2`→`tanh2` の 6 段連鎖
    /// が丸ごと 1 セグメントとして検出され、`MAX_FUSED_CHAIN_LEN`
    /// ちょうどで打ち切られることを固定する（打ち切り境界は `relu` の
    /// 入力 `x`）。「reduction 軸が一致しない連鎖は分断される」反例は
    /// [`mismatched_reduction_axis_splits_the_segment`] を参照。
    #[test]
    fn sum_with_matching_segment_axis_is_fused_not_a_boundary() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let relu = g.push(FusionOp::Relu(x), f32_meta(&[4, 4])).unwrap();
        let sum = g
            .push(
                FusionOp::Sum {
                    input: relu,
                    dim: Some(0),
                },
                f32_meta(&[4]),
            )
            .unwrap();
        let exp = g.push(FusionOp::Exp(sum), f32_meta(&[4])).unwrap();
        let tanh = g.push(FusionOp::Tanh(exp), f32_meta(&[4])).unwrap();
        let relu2 = g.push(FusionOp::Relu(tanh), f32_meta(&[4])).unwrap();
        let tanh2 = g.push(FusionOp::Tanh(relu2), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, tanh2).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![relu, sum, exp, tanh, relu2, tanh2]);
        assert_eq!(seg.nodes.len(), MAX_FUSED_CHAIN_LEN);
        assert_eq!(seg.leaves, vec![x]);
    }

    /// 受け入れ基準（#586）: RMSNorm 様連鎖（`Mul(x,x) → Sum{dim} →
    /// Rsqrt → Mul`）が単一セグメントとして検出される。
    #[test]
    fn rmsnorm_like_chain_with_full_reduction_is_a_single_segment() {
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
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![sq, sum, rsqrt, out]);
        assert_eq!(seg.leaves, vec![x]);
    }

    /// 受け入れ基準（#586）: 同一軸の reduction を 2 個含む連鎖も融合可能
    /// （`Sum{dim=Some(0)}` を 2 回、間に elementwise を挟んで構成）。
    #[test]
    fn two_same_axis_reductions_are_fused_into_one_segment() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[2, 3, 4])).unwrap();
        let n1 = g.push(FusionOp::Relu(x), f32_meta(&[2, 3, 4])).unwrap();
        let r1 = g
            .push(
                FusionOp::Sum {
                    input: n1,
                    dim: Some(0),
                },
                f32_meta(&[3, 4]),
            )
            .unwrap();
        let n2 = g.push(FusionOp::Exp(r1), f32_meta(&[3, 4])).unwrap();
        let r2 = g
            .push(
                FusionOp::Sum {
                    input: n2,
                    dim: Some(0),
                },
                f32_meta(&[4]),
            )
            .unwrap();
        let n3 = g.push(FusionOp::Tanh(r2), f32_meta(&[4])).unwrap();
        let root = g.push(FusionOp::Relu(n3), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![n1, r1, n2, r2, n3, root]);
        assert_eq!(seg.nodes.len(), MAX_FUSED_CHAIN_LEN);
        assert_eq!(seg.leaves, vec![x]);
    }

    /// 受け入れ基準（#586・実装計画 §5「reduction 軸が一致しない連鎖は
    /// 分断される」）の直接固定: セグメント軸を最初に確定する
    /// `r2`（`dim=Some(0)`）と異なる軸を持つ `r1`（`dim=Some(2)`）は
    /// セグメントへ組み込まれず境界（葉）になる。`r1` の手前
    /// （`n1`／`x`）はそもそも `reachable` に入らないため走査対象にも
    /// ならない。
    #[test]
    fn mismatched_reduction_axis_splits_the_segment() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[2, 3, 4])).unwrap();
        let n1 = g.push(FusionOp::Relu(x), f32_meta(&[2, 3, 4])).unwrap();
        let r1 = g
            .push(
                FusionOp::Sum {
                    input: n1,
                    dim: Some(2),
                },
                f32_meta(&[2, 3]),
            )
            .unwrap();
        let n2 = g.push(FusionOp::Exp(r1), f32_meta(&[2, 3])).unwrap();
        let r2 = g
            .push(
                FusionOp::Sum {
                    input: n2,
                    dim: Some(0),
                },
                f32_meta(&[3]),
            )
            .unwrap();
        let n3 = g.push(FusionOp::Tanh(r2), f32_meta(&[3])).unwrap();
        let root = g.push(FusionOp::Relu(n3), f32_meta(&[3])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        // r1（軸不一致）は組み込まれず、その手前（n1・x）も未到達のまま
        // セグメント外＝r1 自身が葉になる。
        assert_eq!(seg.nodes, vec![n2, r2, n3, root]);
        assert_eq!(seg.leaves, vec![r1]);
    }

    /// reduction root は #586 以降も従来どおり `RootNotElementwise`
    /// フォールバック（実装計画 §3.3「root は従来どおり elementwise の
    /// み許可」）。`gemm_root_is_rejected_as_not_elementwise` と同型。
    #[test]
    fn reduction_root_is_rejected_as_not_elementwise() {
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

        let decision = detect_fusion(&g, sum).unwrap();
        assert_eq!(
            decision,
            FusionDecision::Fallback(FallbackReason::RootNotElementwise)
        );
    }

    /// `contiguous == false` が連鎖に混在するとセグメント全体が非融合
    /// フォールバックになる（設計書 §2.3・§3.2 (e)）。
    #[test]
    fn non_contiguous_node_falls_back_the_whole_segment() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        // transpose/broadcast view 相当（contiguous = false）のノード。
        let view = g
            .push(FusionOp::Relu(x), non_contiguous_meta(&[4]))
            .unwrap();
        let root = g.push(FusionOp::Exp(view), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        assert_eq!(
            decision,
            FusionDecision::Fallback(FallbackReason::NonContiguous)
        );
    }

    /// 単一 elementwise ノードは最小長未満のため非融合。
    #[test]
    fn single_elementwise_node_is_below_minimum_length() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();
        let root = g.push(FusionOp::Relu(x), f32_meta(&[4])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        assert_eq!(
            decision,
            FusionDecision::Fallback(FallbackReason::ChainTooShort)
        );
    }

    /// root 自体が融合境界（Gemm）の場合は呼び出し方の誤りとして
    /// `RootNotElementwise` を返す（呼び出し側は elementwise ノードを
    /// root に選ぶ契約）。
    #[test]
    fn gemm_root_is_rejected_as_not_elementwise() {
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let gemm = g.push(FusionOp::Gemm(a, b), f32_meta(&[4, 4])).unwrap();

        let decision = detect_fusion(&g, gemm).unwrap();
        assert_eq!(
            decision,
            FusionDecision::Fallback(FallbackReason::RootNotElementwise)
        );
    }

    /// 決定性: 同一グラフに対する検出結果は常に同一（純関数性。設計書
    /// §3.4）。複数回呼び出して結果が変わらないことを確認する。
    #[test]
    fn detect_fusion_is_deterministic_across_repeated_calls() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let a = g.push(FusionOp::Add(x, y), f32_meta(&[8])).unwrap();
        let b = g.push(FusionOp::Mul(a, a), f32_meta(&[8])).unwrap();
        let c = g.push(FusionOp::Add(b, x), f32_meta(&[8])).unwrap();

        let first = detect_fusion(&g, c).unwrap();
        for _ in 0..10 {
            assert_eq!(detect_fusion(&g, c).unwrap(), first);
        }
    }

    /// #399 codex-review 指摘の再発防止: 範囲外の `root`（まだ push
    /// されていないノード ID）を渡した場合、release ビルドでも
    /// `graph.node(root)` の添字アクセスで panic せず、型付きエラー
    /// [`FusionGraphError::NodeIdOutOfRange`] を返すことを検証する
    /// （`debug_assert!` のみに依存していた従来実装は release ビルド
    /// では検証が消え本番経路の panic になっていた）。
    #[test]
    fn out_of_range_root_returns_typed_error_instead_of_panicking() {
        let mut g = FusionGraph::new();
        let _x = g.push(FusionOp::Input, f32_meta(&[4])).unwrap();

        let result = detect_fusion(&g, FusionNodeId(5));
        assert_eq!(
            result,
            Err(FusionGraphError::NodeIdOutOfRange { id: 5, len: 1 })
        );
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 1」）: RMSNorm 行方向
    /// パターン（`Mul(x,x) → Sum{None} → Rsqrt → Broadcast{None} →
    /// Mul(bc, x)`）が単一セグメントとして検出される。#586 テスト
    /// [`rmsnorm_like_chain_with_full_reduction_is_a_single_segment`]
    /// との違いは、正規化係数を `Broadcast` で明示的に元の行 shape へ
    /// 拡張してから `x` と乗じる点（RMSNorm の実際の計算構造そのもの）。
    #[test]
    fn rmsnorm_pattern_with_explicit_broadcast_is_a_single_segment() {
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
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![sq, sum, rsqrt, bc, out]);
        assert_eq!(seg.leaves, vec![x]);
    }

    /// 受け入れ基準（#588・実装計画 §6「受け入れ基準 1・3」）: softmax
    /// パターン（`Max → Broadcast → Sub → Exp → Sum → Broadcast → Div` の
    /// 7 ノード）が単一セグメントとして検出される。elementwise は
    /// `Sub`／`Exp`／`Div` の 3 個のみで [`MAX_FUSED_CHAIN_LEN`]（6）以下
    /// のため、#588 の意味論精密化（elementwise 数のみに上限を適用）に
    /// より全 7 ノードが 1 セグメントに収まることを直接検証する。
    #[test]
    fn softmax_pattern_with_seven_nodes_is_a_single_segment() {
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
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![mx, bc1, sub, exp, sm, bc2, div]);
        assert_eq!(seg.nodes.len(), 7);
        assert_eq!(seg.leaves, vec![x]);
    }

    /// 受け入れ基準（#588・実装計画 §6）: `Broadcast` の軸がセグメント軸
    /// （root 側の `Sum{dim:Some(0)}` が確定）と一致しない場合、
    /// `mismatched_reduction_axis_splits_the_segment`（`Sum`／`Sum` 間の
    /// 軸不一致）と同様に境界（葉）化される。
    #[test]
    fn mismatched_broadcast_axis_splits_the_segment() {
        let mut g = FusionGraph::new();
        let per_row = g.push(FusionOp::Input, f32_meta(&[2])).unwrap();
        // 軸不一致になる Broadcast（root 側で確定するセグメント軸
        // Some(0) とは異なる Some(1) を使う）。
        let bc_wrong = g
            .push(
                FusionOp::Broadcast {
                    input: per_row,
                    dim: Some(1),
                },
                f32_meta(&[2, 3]),
            )
            .unwrap();
        let n2 = g.push(FusionOp::Exp(bc_wrong), f32_meta(&[2, 3])).unwrap();
        let seg_axis_setter = g
            .push(
                FusionOp::Sum {
                    input: n2,
                    dim: Some(0),
                },
                f32_meta(&[3]),
            )
            .unwrap();
        let n3 = g
            .push(FusionOp::Tanh(seg_axis_setter), f32_meta(&[3]))
            .unwrap();
        let root = g.push(FusionOp::Relu(n3), f32_meta(&[3])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        // bc_wrong（軸不一致）は組み込まれず、その手前（per_row）も
        // 未到達のままセグメント外＝bc_wrong 自身が葉になる。
        assert_eq!(seg.nodes, vec![n2, seg_axis_setter, n3, root]);
        assert_eq!(seg.leaves, vec![bc_wrong]);
    }

    /// #588 実装計画 §3.3: `MAX_FUSED_SEGMENT_NODES`（総数上限。12）は
    /// [`MAX_FUSED_CHAIN_LEN`]（elementwise 数上限。6）とは独立に決定的
    /// 打ち切りを行う。本テストは elementwise が root の 1 個のみで、
    /// 残り全てが `Broadcast{dim:None}`（scalar 上の恒等写像として連鎖
    /// させる）という「reduction／broadcast が elementwise 数上限を
    /// 大きく下回るまま総数上限に達する」構成にすることで、打ち切りが
    /// 総数上限単独の効果であることを示す。
    #[test]
    fn total_node_cap_truncates_independently_of_elementwise_cap() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[])).unwrap();
        let mut prev = x;
        let mut broadcasts = Vec::new();
        for _ in 0..12 {
            let b = g
                .push(
                    FusionOp::Broadcast {
                        input: prev,
                        dim: None,
                    },
                    f32_meta(&[]),
                )
                .unwrap();
            broadcasts.push(b);
            prev = b;
        }
        let root = g.push(FusionOp::Relu(prev), f32_meta(&[])).unwrap();

        let decision = detect_fusion(&g, root).unwrap();
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes.len(), MAX_FUSED_SEGMENT_NODES);
        // root（elementwise 1 個）+ broadcasts[1..12]（11 個）で 12 個
        // ちょうど。broadcasts[0] が打ち切り境界となり葉になる
        // （`seven_stage_chain_is_cut_off_deterministically_at_cap` と
        // 同型の「root から数えて上限個」パターン）。
        assert_eq!(&seg.nodes[..11], &broadcasts[1..]);
        assert_eq!(seg.nodes[11], root);
        assert_eq!(seg.leaves, vec![broadcasts[0]]);
    }
}
