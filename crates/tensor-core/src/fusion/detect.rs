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

use super::graph::{FusionGraph, FusionNodeId, FusionOp};
use crate::dispatch::DType;

/// elementwise 連鎖長の上限（TASK-12.1 の内容規定「4〜6 段程度」の
/// 上限値。PoC-9 `ew6` 実測と整合。設計書 §3.2 (d)）。
///
/// 連鎖がこの段数に達すると、それより深いノードへの走査を打ち切り、
/// 打ち切り位置のノードの入力を外部入力（葉）として扱う
/// （[`detect_fusion`] のドキュメント参照）。ガードレール閾値・テスト
/// 許容誤差ではなく実装判断の定数のため変更にユーザー承認は要さないが、
/// TASK-12.2（#166・実測）で見直し可能な形で定数化しておく
/// （実装計画イシュー #162 §8「リスク・判断の記録」）。
pub(crate) const MAX_FUSED_CHAIN_LEN: usize = 6;

/// 融合セグメント成立に要する最小 elementwise ノード数。
///
/// 設計書に明示規定はないが、単一ノードには融合効果がない
/// （カーネル呼び出し 1 回を融合なしの通常呼び出し 1 回に置き換える
/// だけで、複数カーネル呼び出しの削減という融合の目的を果たさない）
/// ため、安全側の実装判断として 2 以上を要求する（#162 §8 に記録済み。
/// TASK-12.2〈#166〉の実測で見直し可能）。
pub(crate) const MIN_FUSED_CHAIN_LEN: usize = 2;

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

/// `root` を出力とする融合可能な elementwise 連鎖を検出する（設計書
/// §3.2 の実体化条件・§6.1 #162 のスコープに対応する本体）。
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
/// 1. 到達可能集合 `reachable` を `{root}` で初期化する。
/// 2. `id` を `root.0` から `0` まで降順に走査し、`reachable` に含まれる
///    ノードのみを処理する:
///    - elementwise（`Add`／`Mul`／`Relu`／`Exp`／`Tanh`）なら:
///      - `dtype != F32` または `contiguous == false` を検出した時点で
///        走査全体を打ち切り [`FusionDecision::Fallback`] を返す
///        （設計書 §3.2 (e)・セグメント全体を非融合にする方針）。
///      - セグメントへ追加する。現在のセグメントサイズが
///        [`MAX_FUSED_CHAIN_LEN`] **未満**であれば、このノードの入力を
///        `reachable` へ追加して走査を継続する（**上限に達した場合は
///        追加しない**＝それ以上深いノードへは到達不能になり、後段の
///        葉抽出で自動的に外部入力として扱われる。設計書 §3.2 (d)）。
///    - 融合境界（`Gemm`／`Sum`／`Max`）または `Input` なら、それ自体を
///      入力方向へは展開せず、そのノード ID を後段で葉として扱う
///      （設計書 §3.2 (a)(b)）。
/// 3. 走査完了後、セグメントに含まれるノードが 1 個でも `dtype`／
///    `contiguous` 違反で打ち切られていなければ、セグメントの各ノードが
///    参照する入力のうちセグメントに含まれないものを葉として集約する
///    （重複除去・昇順ソート）。
/// 4. セグメントのノード数が [`MIN_FUSED_CHAIN_LEN`] 未満なら
///    [`FallbackReason::ChainTooShort`] で非融合とする。
///
/// fan-out（同一ノードが複数ノードから参照される）・fan-in（複数の
/// elementwise 連鎖が 1 ノードへ合流する `(a+b)*(c+d)` 形）はいずれも
/// `reachable`（集合）と降順走査により自然に扱える。fan-out はそれ自体
/// 融合不能条件にしない（設計書 §2.4・PoC-9 `ew_fanout` 実測根拠）。
pub(crate) fn detect_fusion(graph: &FusionGraph, root: FusionNodeId) -> FusionDecision {
    debug_assert!(
        root.0 < graph.len(),
        "detect_fusion は既存ノードの root でのみ呼ばれる契約（呼び出し側のバグ検出用）"
    );

    if !graph.node(root).op.is_elementwise() {
        return FusionDecision::Fallback(FallbackReason::RootNotElementwise);
    }

    // 到達可能（＝セグメント候補として展開されうる）ノード ID の集合。
    // `BTreeSet` を使うのは走査ロジック自体には無関係（`contains` の
    // みに使う）だが、内部状態も含め非決定的なイテレーション順に依存
    // しない実装であることを明示するため統一する。
    use std::collections::BTreeSet;
    let mut reachable: BTreeSet<usize> = BTreeSet::new();
    reachable.insert(root.0);

    // セグメントに含まれる elementwise ノード（発見順。降順走査のため
    // root に近い側から入る。最終的に昇順へソートし直す）。
    let mut included: BTreeSet<usize> = BTreeSet::new();

    for id in (0..=root.0).rev() {
        if !reachable.contains(&id) {
            continue;
        }
        let node = graph.node(FusionNodeId(id));
        if !node.op.is_elementwise() {
            // 融合境界（Gemm/Sum/Max）または Input。ここでは展開せず、
            // 葉抽出（後段）に委ねる。
            continue;
        }
        if node.meta.dtype != DType::F32 {
            return FusionDecision::Fallback(FallbackReason::UnsupportedDtype);
        }
        if !node.meta.contiguous {
            return FusionDecision::Fallback(FallbackReason::NonContiguous);
        }

        included.insert(id);
        if included.len() < MAX_FUSED_CHAIN_LEN {
            for input in node.op.inputs() {
                reachable.insert(input.0);
            }
        }
        // 上限に達した場合はここで入力を reachable へ追加しない
        // （= それらは後段の葉抽出でセグメント外部の入力として現れる。
        // 設計書 §3.2 (d)「打ち切り位置より深いノードの出力を葉として
        // 扱う」の実体）。
    }

    if included.len() < MIN_FUSED_CHAIN_LEN {
        return FusionDecision::Fallback(FallbackReason::ChainTooShort);
    }

    // 葉抽出: セグメントに含まれる各ノードの入力のうち、セグメントに
    // 含まれないものを外部入力として集約する（重複除去は BTreeSet が
    // 兼ねる。昇順は BTreeSet のイテレーション順そのもの）。
    let mut leaves: BTreeSet<usize> = BTreeSet::new();
    for &id in &included {
        let node = graph.node(FusionNodeId(id));
        for input in node.op.inputs() {
            if !included.contains(&input.0) {
                leaves.insert(input.0);
            }
        }
    }

    FusionDecision::Fuse(FusionSegment {
        nodes: included.into_iter().map(FusionNodeId).collect(),
        leaves: leaves.into_iter().map(FusionNodeId).collect(),
        root,
    })
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

    /// PoC-9 `ew4` 相当: `y = relu(exp(x + w))` 的な 3 段（Add→Exp→Relu）
    /// では最小長は満たすが、4 段連鎖（Add→Relu→Exp→Tanh）を構成して
    /// 「4〜6 段程度」の下限側を検証する。
    #[test]
    fn detects_four_stage_elementwise_chain_as_single_segment() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let w = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let n1 = g.push(FusionOp::Add(x, w), f32_meta(&[8])).unwrap();
        let n2 = g.push(FusionOp::Relu(n1), f32_meta(&[8])).unwrap();
        let n3 = g.push(FusionOp::Exp(n2), f32_meta(&[8])).unwrap();
        let n4 = g.push(FusionOp::Tanh(n3), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, n4);
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

        let decision = detect_fusion(&g, root);
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

        let decision = detect_fusion(&g, root);
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes.len(), MAX_FUSED_CHAIN_LEN);
        // root から数えて 6 段（chain[1..=6]、末尾 6 要素）のみが
        // セグメントに含まれ、打ち切り境界（chain[0]）の入力である
        // 葉ノード x のみが外部入力として残る（chain[0] 自身はセグメント
        // にもリーフにも含まれない = 実体化不要な中間ノードとして扱わ
        // れず、単に走査対象外になる。これは「6 段ちょうど」を root 側
        // から数える仕様の帰結であり、chain[0] の出力は誰からも参照され
        // ないため実際には未使用ノードになる）。
        assert_eq!(&seg.nodes, &chain[1..]);
        assert_eq!(seg.leaves, vec![chain[0]]);
    }

    /// PoC-9 `ew_fanout` 相当: `a = x + y; b = a * a; c = b + x`。
    /// fan-out（a が 2 回参照される）は融合不能条件にしない
    /// （設計書 §2.4）。
    #[test]
    fn fanout_chain_is_detected_as_single_segment_with_correct_use_count() {
        let mut g = FusionGraph::new();
        let x = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let y = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let a = g.push(FusionOp::Add(x, y), f32_meta(&[8])).unwrap();
        let b = g.push(FusionOp::Mul(a, a), f32_meta(&[8])).unwrap();
        let c = g.push(FusionOp::Add(b, x), f32_meta(&[8])).unwrap();

        assert_eq!(g.node(a).use_count, 2, "a は b と c から参照される");

        let decision = detect_fusion(&g, c);
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![a, b, c]);
        assert_eq!(seg.leaves, vec![x, y]);
    }

    /// fan-in: `(a + b) * (c + d)` 形の合流が単一セグメントとして検出
    /// される（`FusionGraph` は DAG 一般をサポートするため fan-in も
    /// 通常ケース。設計書 §6.2「同一テープ内での遅延グラフの合流は
    /// 正規サポート対象」）。
    #[test]
    fn fanin_confluence_is_detected_as_single_segment() {
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let c = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let d = g.push(FusionOp::Input, f32_meta(&[8])).unwrap();
        let left = g.push(FusionOp::Add(a, b), f32_meta(&[8])).unwrap();
        let right = g.push(FusionOp::Add(c, d), f32_meta(&[8])).unwrap();
        let root = g.push(FusionOp::Mul(left, right), f32_meta(&[8])).unwrap();

        let decision = detect_fusion(&g, root);
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![left, right, root]);
        assert_eq!(seg.leaves, vec![a, b, c, d]);
    }

    /// PoC-9 `ew_matmul_ew` 相当: Gemm 境界でセグメントが分断される
    /// （設計書 §3.2 (b)）。
    #[test]
    fn gemm_boundary_splits_the_segment() {
        let mut g = FusionGraph::new();
        let a = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let b = g.push(FusionOp::Input, f32_meta(&[4, 4])).unwrap();
        let gemm = g.push(FusionOp::Gemm(a, b), f32_meta(&[4, 4])).unwrap();
        let relu = g.push(FusionOp::Relu(gemm), f32_meta(&[4, 4])).unwrap();
        let exp = g.push(FusionOp::Exp(relu), f32_meta(&[4, 4])).unwrap();

        let decision = detect_fusion(&g, exp);
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        // Gemm 自体はセグメントに含まれず、その出力が葉として扱われる。
        assert_eq!(seg.nodes, vec![relu, exp]);
        assert_eq!(seg.leaves, vec![gemm]);
    }

    /// Sum／Max 境界も同様にセグメントを分断する（設計書 §3.2 (a)）。
    #[test]
    fn sum_and_max_boundary_split_the_segment() {
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

        let decision = detect_fusion(&g, tanh);
        let FusionDecision::Fuse(seg) = decision else {
            panic!("expected Fuse, got {decision:?}");
        };
        assert_eq!(seg.nodes, vec![exp, tanh]);
        assert_eq!(seg.leaves, vec![sum]);

        // relu 自身を root にすれば、その手前で完結する別セグメントとして
        // 独立に検出できる（sum 側からは辿れない独立した連結成分）。
        let relu_decision = detect_fusion(&g, relu);
        let FusionDecision::Fallback(reason) = relu_decision else {
            panic!("expected Fallback (single node < MIN_FUSED_CHAIN_LEN)");
        };
        assert_eq!(reason, FallbackReason::ChainTooShort);
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

        let decision = detect_fusion(&g, root);
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

        let decision = detect_fusion(&g, root);
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

        let decision = detect_fusion(&g, gemm);
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

        let first = detect_fusion(&g, c);
        for _ in 0..10 {
            assert_eq!(detect_fusion(&g, c), first);
        }
    }
}
