//! split-K 2 パス GEMM（イシュー #1474）の parity 非後退契約のベースライン
//! fixture と検査ユーティリティ。
//!
//! # 位置づけ
//!
//! split-K 経路（`MetalGemm::dispatch_split_k_strided_prepared*`）は、パス 1
//! で K 方向を `partitions` 個の区間へ分割してそれぞれ独立に部分和（f32・
//! `simdgroup_multiply_accumulate` による FMA 連鎖）を求め、パス 2 で
//! パーティション昇順の固定順序 Neumaier 補償和により結合する。この結合は
//! 単一の連続 K ループで求めた古典（classic）経路の FMA 累積とは加算の
//! 結合順序（associativity）が異なるため、丸め誤差の生じ方も異なる。
//!
//! **実機実測で確認した事実（イシュー #1474。`docs/perf/metal-gemm-
//! splitk-two-pass.md` §5 に記録）**: 古典経路は `(32,32,8192)` を含む
//! 全対象形状で CPU 参照実装（`matmul_reference_fma`）と bit 完全一致する
//! （`crates/backend-metal/tests/zz_diag_classic_k8192.rs` 相当の診断で
//! `fail_count=0`・`max_abs_diff=0.0` を確認済み）。一方 split-K 経路は、
//! 縮約側を単純な逐次加算から Neumaier 補償和へ改善した後でも、対象 11
//! 形状（`docs/backend-metal-splitk-decision.md` §3 の 9 形状 + K 端数
//! 境界 2 形状）× NN/NT/TN/TT のうち大半で REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）
//! の要素単位 fail が 1〜8 件／1024〜16384 要素発生する。これはパーティション
//! 分割そのものに起因する構造的特性（結合順序の違いによる丸め誤差）であり、
//! `partitions=2`（最小分割）の時点で既に一部形状で許容誤差を超える誤差が
//! 観測されるため、縮約アルゴリズムの改善だけでは解消できない
//! （`docs/perf/metal-gemm-splitk-two-pass.md` §5.2 の `partitions` 別
//! 感度診断を参照）。
//!
//! この特性は `docs/spec/04-requirements.md` REQ-2 2026-09-02 追記
//! （TF32/f16 Tensor Core 経路の受け入れ判定方式）が CUDA 側で確立した
//! 「厳密ゼロ fail 判定は実機実測で成立が確認された形状に限り、成立しない
//! 形状は実測 baseline 非後退方式を正式な受け入れ判定とする」という方針
//! と同種であり（split-K も「単一カーネル内 FMA 連鎖とは異なる結合順序で
//! 数値を得る」という点で TF32/f16 経路と同じ性質を持つ）、本モジュールは
//! `crates/backend-cuda/tests/common/parity_baseline.rs` と同型の非後退
//! 契約を Metal split-K 向けに提供する。判定式・閾値定数
//! （`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）自体は一切変更せず、
//! `fandhe_ai_backend_cpu::parity::compare` をそのまま呼んで得た集計結果
//! （`fail_count`・`mean_abs_diff`・`max_abs_diff`・`max_rel_err`）が
//! 実測ベースラインを上回っていないかのみを検査する。
//!
//! **適用拡張の承認状況（イシュー #1511・2026-09-10 ユーザー承認。
//! `docs/backend-metal-splitk-parity-judgment-decision.md` §7）**:
//! 上記の TF32/f16 Tensor Core 経路限定だった spec REQ-2 追記の判定
//! パターンを Metal f32 split-K 経路へ適用拡張してよいこと・対象 11
//! 形状すべてへ一律に baseline 方式を適用すること（CUDA 先例の形状二分
//! 方式は不採用）・本モジュール `BASELINES` の 11 行を承認済み ceiling
//! 値として用いることが承認された。これより緩める「上方更新」は改めて
//! ユーザー承認が必要（下記 `assert_no_split_k_parity_regression` の doc
//! コメント参照）。
//!
//! 本経路は opt-in・`dispatch_auto` へ未結線のプロトタイプ（イシュー
//! #1474 のスコープ。性能 A/B は #1475、本番結線可否は #1476 で
//! **結線しないと確定**〈`docs/backend-metal-splitk-decision.md` §4〉）
//! であり、本非後退契約は「正しさが実測どおりであること」を機械的に
//! 固定する目的に限る。判定方式は PR #1496 の codex-review 指摘を受け
//! いったん `assert_parity`（厳密ゼロ fail 判定）へ差し戻されたが、上記
//! 承認を受けてイシュー #1512 で `tests/gemm_splitk_parity.rs` から
//! 本モジュール経由（`assert_no_split_k_parity_regression`）の判定へ
//! 再切替済み（`docs/perf/metal-gemm-splitk-two-pass.md` §5.5・§5.8）。
//! `SPLIT_K_NUMERIC_CONTRACT_APPROVED`（自動判定入口 `dispatch_split_k_
//! strided_prepared` のゲート）の解除は別イシュー（#1513）のスコープで
//! 本モジュールの承認範囲外。

#![allow(dead_code)] // テストファイルごとに使う関数が異なるため。

use fandhe_ai_backend_cpu::parity::CompareReport;

/// 経路・形状ごとの記録済みベースライン 1 行。
///
/// `total`/`baseline_fail_count`/各 ceiling は、本イシュー（#1474）の
/// 実機実測（Apple M4 Max。`docs/perf/logs/metal-gemm-splitk-two-pass-1474/`）
/// からの転記であり推定値は含まない。4 種の転置パターン（NN/NT/TN/TT）は
/// 論理的に同一の行列積を異なる物理レイアウトで計算するだけであり、実測でも
/// 常に同一の集計値になることを確認済みのため、`(m, n, k)` 単位で 1 行のみ
/// 持つ（転置パターンでは分岐しない）。
#[derive(Debug, Clone, Copy)]
pub struct SplitKParityBaseline {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub total: usize,
    pub baseline_fail_count: usize,
    /// 表記丸め対応の天井値（実測値の表示桁の最終桁を切り上げた値。
    /// tolerance 定数の緩和ではなく表記丸め誤差の吸収のみ）。
    pub baseline_mean_abs_diff_ceiling: f64,
    pub baseline_max_abs_diff_ceiling: f64,
    pub baseline_max_rel_err_ceiling: f64,
}

/// 記録済みベースライン一覧（`docs/perf/metal-gemm-splitk-two-pass.md` §5
/// の表・11 行。`docs/backend-metal-splitk-decision.md` §3 の対象 9 形状 +
/// K 端数境界 2 形状）。
///
/// 出典: イシュー #1474 実機実測（Apple M4 Max。2026-09-09）。承認記録:
/// `docs/backend-metal-splitk-parity-judgment-decision.md` §7 の表
/// （2026-09-10 ユーザー承認。実測値・ceiling とも本 11 行と一致）・
/// `docs/perf/metal-gemm-splitk-two-pass.md` §5.3 の全数実測結果表。
/// 未計測形状の行追加は実機実測とセットでのみ行う（推定値の捏造をしない）。
pub static BASELINES: &[SplitKParityBaseline] = &[
    SplitKParityBaseline {
        m: 32,
        n: 32,
        k: 2048,
        total: 1024,
        baseline_fail_count: 0,
        baseline_mean_abs_diff_ceiling: 1.0e-5,
        baseline_max_abs_diff_ceiling: 9.0e-5,
        baseline_max_rel_err_ceiling: 3.0e-4,
    },
    SplitKParityBaseline {
        m: 32,
        n: 32,
        k: 4096,
        total: 1024,
        baseline_fail_count: 0,
        baseline_mean_abs_diff_ceiling: 2.0e-5,
        baseline_max_abs_diff_ceiling: 3.0e-4,
        baseline_max_rel_err_ceiling: 2.0e-4,
    },
    SplitKParityBaseline {
        m: 32,
        n: 32,
        k: 8192,
        total: 1024,
        baseline_fail_count: 1,
        baseline_mean_abs_diff_ceiling: 4.0e-5,
        baseline_max_abs_diff_ceiling: 4.0e-4,
        baseline_max_rel_err_ceiling: 1.2e-3,
    },
    SplitKParityBaseline {
        m: 64,
        n: 64,
        k: 2048,
        total: 4096,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 1.0e-5,
        baseline_max_abs_diff_ceiling: 1.0e-4,
        baseline_max_rel_err_ceiling: 4.0e-3,
    },
    SplitKParityBaseline {
        m: 64,
        n: 64,
        k: 4096,
        total: 4096,
        baseline_fail_count: 0,
        baseline_mean_abs_diff_ceiling: 2.0e-5,
        baseline_max_abs_diff_ceiling: 3.0e-4,
        baseline_max_rel_err_ceiling: 8.0e-4,
    },
    SplitKParityBaseline {
        m: 64,
        n: 64,
        k: 8192,
        total: 4096,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 4.0e-5,
        baseline_max_abs_diff_ceiling: 4.0e-4,
        baseline_max_rel_err_ceiling: 2.3e-3,
    },
    SplitKParityBaseline {
        m: 128,
        n: 128,
        k: 2048,
        total: 16384,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 1.0e-5,
        baseline_max_abs_diff_ceiling: 1.2e-4,
        baseline_max_rel_err_ceiling: 3.4e-3,
    },
    SplitKParityBaseline {
        m: 128,
        n: 128,
        k: 4096,
        total: 16384,
        baseline_fail_count: 8,
        baseline_mean_abs_diff_ceiling: 2.0e-5,
        baseline_max_abs_diff_ceiling: 1.7e-4,
        baseline_max_rel_err_ceiling: 8.6e-3,
    },
    SplitKParityBaseline {
        m: 128,
        n: 128,
        k: 8192,
        total: 16384,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 4.0e-5,
        baseline_max_abs_diff_ceiling: 4.0e-4,
        baseline_max_rel_err_ceiling: 1.7e-3,
    },
    SplitKParityBaseline {
        m: 64,
        n: 64,
        k: 2056,
        total: 4096,
        baseline_fail_count: 1,
        baseline_mean_abs_diff_ceiling: 1.0e-5,
        baseline_max_abs_diff_ceiling: 1.3e-4,
        baseline_max_rel_err_ceiling: 1.3e-3,
    },
    SplitKParityBaseline {
        m: 128,
        n: 128,
        k: 2064,
        total: 16384,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 1.0e-5,
        baseline_max_abs_diff_ceiling: 1.3e-4,
        // この形状は真値が 0 近傍の要素で桁落ちが起きやすく相対誤差の
        // 外れ値が大きい（実測 max_rel_err=0.1309）。abs diff 自体は他行
        // と同水準に小さく、桁落ちに起因する相対誤差の性質として想定
        // 範囲内（`docs/perf/metal-gemm-splitk-two-pass.md` §5.3）。
        baseline_max_rel_err_ceiling: 0.14,
    },
];

/// `(m, n, k)` に対応するベースライン行を探す。見つからなければ `None`
/// （未登録形状での呼び出しは呼び出し側で明示的に扱う。fail-open で
/// 素通りさせない）。
pub fn find_baseline(m: usize, n: usize, k: usize) -> Option<&'static SplitKParityBaseline> {
    BASELINES.iter().find(|b| b.m == m && b.n == n && b.k == k)
}

/// `report` が `baseline` を後退していないか検査する（`total` 一致・
/// `fail_count`／`mean_abs_diff`／`max_abs_diff`／`max_rel_err` の 4 点が
/// いずれもベースライン以下）。tolerance 定数（`RELATIVE_TOLERANCE`/
/// `ABSOLUTE_RESCUE_THRESHOLD`）自体には一切触れない
/// （`fandhe_ai_backend_cpu::parity::compare` の出力をそのまま比較する
/// だけ）。
pub fn assert_no_split_k_parity_regression(
    context: &str,
    report: &CompareReport,
    baseline: &SplitKParityBaseline,
) {
    assert_eq!(
        report.total, baseline.total,
        "{context}: total 要素数が記録済みベースライン（{}）と一致しません（実測={}）。\
         形状・シードが変わっていないか確認してください。",
        baseline.total, report.total
    );
    assert!(
        report.fail_count <= baseline.baseline_fail_count,
        "{context}: fail_count が記録済みベースライン（{}）を後退しています（実測={}）。",
        baseline.baseline_fail_count,
        report.fail_count
    );
    assert!(
        report.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling,
        "{context}: mean_abs_diff が記録済みベースライン天井（{:.6e}）を後退しています\
         （実測={:.6e}）。",
        baseline.baseline_mean_abs_diff_ceiling,
        report.mean_abs_diff
    );
    assert!(
        report.max_abs_diff <= baseline.baseline_max_abs_diff_ceiling,
        "{context}: max_abs_diff が記録済みベースライン天井（{:.6e}）を後退しています\
         （実測={:.6e}）。",
        baseline.baseline_max_abs_diff_ceiling,
        report.max_abs_diff
    );
    assert!(
        report.max_rel_err <= baseline.baseline_max_rel_err_ceiling,
        "{context}: max_rel_err が記録済みベースライン天井（{:.6e}）を後退しています\
         （実測={:.6e}）。",
        baseline.baseline_max_rel_err_ceiling,
        report.max_rel_err
    );
}
