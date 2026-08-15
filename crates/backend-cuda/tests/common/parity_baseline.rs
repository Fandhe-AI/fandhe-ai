//! parity 非後退契約（イシュー #491・GEMM 性能改善ツリー B-0・親 #490）の
//! ベースライン fixture と検査ユーティリティ。
//!
//! # 位置づけ
//!
//! `wmma_tf32`・`wmma_tf32_opt`・`mma_f16` 系経路は REQ-2 統一複合判定
//! （`backend_cpu::RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）で恒常
//! fail の既知状態にある（#186 由来。`docs/backend-cuda-real-device-testing.md`
//! §5.3 に実測記録済み）。以降の Phase B/C カーネル改修は「parity green」を
//! 受け入れ条件にできないため、本モジュールは「fail 比率・平均絶対誤差が
//! 記録済みベースラインを上回らない」ことを機械検査する**非後退契約**を提供
//! する。判定式・閾値定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）
//! 自体は一切変更しない（`backend_cpu::parity` を単に呼ぶだけで複製しない）。
//!
//! ## 承認記録（PR #640 codex-review 指摘対応）
//!
//! `baseline_fail_count`/`baseline_mean_abs_diff_ceiling` は
//! `RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`（tolerance 定数）とは
//! **別の概念**であり、両者を混同しない: tolerance 定数は「1 要素あたりの
//! 合否判定式」そのもの、baseline のこの 2 値は「その判定式を通した後の
//! 集計結果（fail 件数・平均絶対誤差）が既知の恒常 fail 状態から悪化して
//! いないか」を見る非後退の上限である（`assert_no_parity_regression` は
//! `report.total`・`fail_count`・`mean_abs_diff` の 3 点比較のみを行い、
//! tolerance 定数自体には一切触れない。`tolerance_constants_are_pinned` が
//! 定数の無断変更を別途 bit 等値で検知する）。
//!
//! この非後退基準（「恒常 fail 経路は green を要求せず、記録済み
//! ベースラインを上回らないことを機械検査する」設計）自体は、本モジュールが
//! 実装するイシュー #491 の受け入れ基準に明記された**ユーザー承認済みの
//! 仕様**であり、本 PR が新設した緩和ではない（イシュー #491 本文
//! 「§1.2 parity 非後退契約」2 番目の受け入れ基準を単一の正として参照する。
//! `docs/perf/cuda-parity-baseline.md` §6「ベースライン更新規約」に
//! 「上方更新（緩和）はユーザー承認必須」と明記し、本表の初期値（下方
//! 更新にすら当たらない初回記録）とは区別している）。
//!
//! `crates/backend-cuda/tests/*.rs` は独立クレート扱いのため、各テスト
//! ファイルから `mod common;` で本モジュールを共有する
//! （`crates/autodiff/tests/common/mod.rs` と同型のパターン）。
//!
//! 正本ドキュメントは `docs/perf/cuda-parity-baseline.md`
//! （ベースライン更新規約・出典・実測環境の記録）。本ファイルは機械検査の
//! 正であり、ドキュメント側と二重管理しない。

#![allow(dead_code)] // テストファイルごとに使う関数・定数が異なるため。

/// 非後退契約の対象経路（Issue #491 が対象とする 3 経路）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityPath {
    /// `CudaGemm::run_wmma_tf32`（基本 WMMA(TF32) カーネル）。
    WmmaTf32,
    /// `CudaGemm::run_wmma_tf32`（opt カーネル利用可能環境。
    /// `wmma_tf32_opt_available()` で分岐する経路）。
    WmmaTf32Opt,
    /// `CudaMmaGemm::run_f16`（`mma.sync`/`ldmatrix`/`cp.async` 経路）。
    MmaF16,
}

impl std::fmt::Display for ParityPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ParityPath::WmmaTf32 => "wmma_tf32",
            ParityPath::WmmaTf32Opt => "wmma_tf32_opt",
            ParityPath::MmaF16 => "mma_f16",
        };
        f.write_str(s)
    }
}

/// 経路・形状ごとの記録済みベースライン 1 行。
///
/// `total`/`baseline_fail_count`/`baseline_mean_abs_diff_ceiling` は
/// `docs/backend-cuda-real-device-testing.md` §5.3（DGX Spark GB10・sm_121
/// 実機実測）からの転記であり、推定値は含まない。各行の出典・実測日は
/// `docs/perf/cuda-parity-baseline.md` の表に対応する。
#[derive(Debug, Clone, Copy)]
pub struct ParityBaseline {
    pub path: ParityPath,
    pub context: &'static str,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub seed: u64,
    pub total: usize,
    pub baseline_fail_count: usize,
    /// 表記丸め対応の天井値（`docs/perf/cuda-parity-baseline.md` §「表記丸め
    /// 対応」）。§5.3 記録値の最終桁を切り上げた値を格納する（許容誤差
    /// 〈tolerance 定数〉の緩和ではなく、表記丸め誤差の吸収のみ。判定式・
    /// 閾値定数は一切変更しない）。
    pub baseline_mean_abs_diff_ceiling: f64,
    /// この行の記録値が「意図したカーネル版で実測されたか」の provenance
    /// 保証が取れていない場合に `true`（PR #640 codex-review 指摘対応・
    /// イシュー #491）。
    ///
    /// **fail-closed 契約（PR #640 codex-review P1 再指摘対応。旧
    /// `pending_basic_remeasurement` からの改名）**: `true` の行は
    /// `assert_no_parity_regression` が判定を試みず**必ず panic する**
    /// （黙って skip しない。旧実装は provenance 不確実な行をスキップし
    /// 「実機テストは正常終了と shape 一致だけで通過する」状態を許してい
    /// たが、これは非後退ゲートが機能していないのに green に見える回帰
    /// だった）。記録元テストが opt カーネル可用性を確認せず `run_wmma_tf32`
    /// を呼んだため、記録値が opt カーネルの実測結果である可能性が高く、
    /// 基本版カーネル専用の実測経路
    /// （`backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`。
    /// `src/gemm.rs` 内のライブラリ単体テスト）と比較すると、カーネルが
    /// 異なり fail_count・mean_abs_diff の分布も異なるため、後退していない
    /// 変更を拒否する false-fail と、基本版の後退を見逃す false-pass の
    /// 両方が生じ非後退ゲートとして成立しない。実機再測定でこの provenance
    /// 不確実性を解消したら `false` へ更新し、基本版カーネル専用の実測値へ
    /// 差し替える（推定値での上書きはしない。
    /// `docs/perf/cuda-parity-baseline.md` §「既知の限界」）。それまでの間、
    /// 該当行を含む実機必須テストは実行のたびに fail し続ける契約であり、
    /// これは意図した挙動である（本リポで既知の受け入れ済み状態。
    /// `docs/backend-cuda-real-device-testing.md` §5.3・§7 参照）。
    pub basic_kernel_baseline_unconfirmed: bool,
}

/// 記録済みベースライン一覧（`docs/perf/cuda-parity-baseline.md` の表・6 行）。
///
/// 出典: `docs/backend-cuda-real-device-testing.md` §5.3（DGX Spark GB10・
/// sm_121・2026 年 8 月時点実測）。§5.3 の記録は `assert_parity` が最初の
/// fail で panic する契約のため「各テストで最初に fail した (形状, シード)
/// の値」のみが実測されている。未計測形状の行追加は実機実測とセットでのみ
/// 行う（推定値の捏造をしない。`docs/perf/cuda-parity-baseline.md`
/// 「ベースライン更新規約」）。
pub static BASELINES: &[ParityBaseline] = &[
    // wmma_tf32: 形状網羅の最小ケース（32x32x32、seed=2000）。
    // `tests/gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes` の
    // 先頭ケース（`assert_wmma_tf32_parity(&gemm, ctx, 2000 + 0, 32, 32, 32)`）。
    //
    // 既知の限界（PR #640 Cursor Bugbot 指摘・未解決）: 出典テストは
    // `run_wmma_tf32`（opt 可用なら opt 優先）を opt 可用性確認なしに
    // 呼ぶため、この記録値が実際には opt カーネル計測である可能性がある
    // （DGX Spark GB10 実機では opt が概ね利用可能。
    // `docs/perf/cuda-parity-baseline.md` §3「既知の限界」参照）。基本版
    // カーネル専用の単体テスト
    // （`backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`。
    // `src/gemm.rs`）による非後退検査との比較でこの provenance 不確実性を
    // 認識したうえで扱うこと。実機再測定でのみ解消可能（推定値での上書きは
    // しない）。
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 32x32x32 seed=2000",
        m: 32,
        n: 32,
        k: 32,
        seed: 2000,
        total: 32 * 32,
        baseline_fail_count: 154,
        baseline_mean_abs_diff_ceiling: 3.699e-4,
        // PR #640 codex-review P1 指摘対応: 記録元は opt 可用性を確認せず
        // `run_wmma_tf32` を呼ぶため opt 実測の可能性が高く、基本版専用
        // エントリとの比較に使えない（上記ドキュメンテーションコメント参照）。
        basic_kernel_baseline_unconfirmed: true,
    },
    // wmma_tf32: K=4096 ストレスケース先頭（256x256x4096、seed=8888）。
    // `tests/gemm_wmma_tf32.rs::wmma_tf32_k4096_stress_poc_v2_5` の先頭呼出し。
    //
    // 既知の限界: 上の 32x32x32 行と同じ provenance 不確実性が該当する
    // （出典テストが opt 可用性を確認せず `run_wmma_tf32` を呼ぶため）。
    ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress)",
        m: 256,
        n: 256,
        k: 4096,
        seed: 8888,
        total: 256 * 256,
        baseline_fail_count: 10647,
        baseline_mean_abs_diff_ceiling: 4.477e-3,
        // PR #640 codex-review P1 指摘対応: 32x32x32 行と同じ provenance
        // 不確実性が該当する（上記ドキュメンテーションコメント参照）。
        basic_kernel_baseline_unconfirmed: true,
    },
    // wmma_tf32_opt: 512x512x512。エントリポイントは `run_wmma_tf32` と共通
    // だが、記録元 `tensor_core_real_device.rs::tensor_core_parity_record`
    // TF32 部分は事前に `wmma_tf32_opt_available()` を assert してから計測
    // しているため（`gemm.run_wmma_tf32(...)`、seed=0x7A0）、opt 経路の
    // ベースラインとして扱う（`WmmaTf32` のままだと `check_wmma_tf32_baseline`
    // が opt 可用性を検査せず、opt 計測値を basic カーネルの計測結果と
    // 比較してしまう。Cursor Bugbot 指摘対応）。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record)",
        m: 512,
        n: 512,
        k: 512,
        seed: 0x7A0,
        total: 512 * 512,
        baseline_fail_count: 42493,
        baseline_mean_abs_diff_ceiling: 1.575e-3,
        // opt 可用性を計測前に assert 済み（記録元コメント参照）。
        // provenance 不確実性なし。
        basic_kernel_baseline_unconfirmed: false,
    },
    // wmma_tf32_opt: 形状網羅の先頭ケース（64x64x64、seed=3000）。
    // `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes`
    // の先頭ケース（`assert_wmma_tf32_opt_parity(&gemm, ctx, 3000 + 0, 64, 64, 64)`）。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 64x64x64 seed=3000",
        m: 64,
        n: 64,
        k: 64,
        seed: 3000,
        total: 64 * 64,
        baseline_fail_count: 699,
        baseline_mean_abs_diff_ceiling: 5.677e-4,
        // 記録元テストは opt 経路専用（`gemm_wmma_tf32_opt.rs`）のため
        // provenance 不確実性なし。
        basic_kernel_baseline_unconfirmed: false,
    },
    // wmma_tf32_opt: K=4096 ストレスケース先頭（512x512x4096、seed=0xC0FFEE）。
    // `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_k4096_stress` の先頭呼出し。
    ParityBaseline {
        path: ParityPath::WmmaTf32Opt,
        context: "wmma_tf32_opt 512x512x4096 seed=0xC0FFEE",
        m: 512,
        n: 512,
        k: 4096,
        seed: 0xC0FFEE,
        total: 512 * 512,
        baseline_fail_count: 43019,
        baseline_mean_abs_diff_ceiling: 4.464e-3,
        // 記録元テストは opt 経路専用（`gemm_wmma_tf32_opt.rs`）のため
        // provenance 不確実性なし。
        basic_kernel_baseline_unconfirmed: false,
    },
    // mma_f16: K=4096 ストレスケース（256x256x4096、seed=9999）。
    // `tests/cpu_cuda_mma_parity.rs::mma_f16_k4096_stress` の先頭呼出し
    // （`assert_mma_f16_parity(&gemm, ctx, 9999, 256, 256, 4096)`）。
    ParityBaseline {
        path: ParityPath::MmaF16,
        context: "mma_f16 256x256x4096 seed=9999",
        m: 256,
        n: 256,
        k: 4096,
        seed: 9999,
        total: 256 * 256,
        baseline_fail_count: 101,
        baseline_mean_abs_diff_ceiling: 7.647e-5,
        // `run_f16` は基本/opt の分岐を持たないため provenance 不確実性なし。
        basic_kernel_baseline_unconfirmed: false,
    },
];

/// 非後退判定: `report` がベースラインを上回っていないことを assert する。
///
/// `backend_cpu::compare` が返す `CompareReport` をそのまま使い、判定式
/// （複合判定の合否）を複製・改変しない（`.claude/rules/security.md` A08
/// 「判定の迂回経路を作らない」）。
///
/// 3 点を fail-closed で検査する:
/// 1. `report.total == baseline.total`（形状・比較対象のずれの検出。
///    fixture の `total` 定義とテスト側の実測形状がずれていた場合に
///    「たまたま非後退に見える」誤判定を防ぐ）
/// 2. `report.fail_count <= baseline.baseline_fail_count`
/// 3. `report.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling`
///
/// # Panics
///
/// いずれかの検査に失敗した場合、`backend_cpu::assert_parity` と同水準の
/// 診断情報（fail_count・mean_abs_diff 等の分布統計）を付けて panic する。
/// `baseline.basic_kernel_baseline_unconfirmed == true` の場合は上記 3 点の
/// 比較を行わず、実機再測定が必要である旨のメッセージで即座に panic する
/// （fail-closed。`ParityBaseline::basic_kernel_baseline_unconfirmed` の
/// ドキュメンテーションコメント参照。PR #640 codex-review P1 再指摘対応:
/// 呼び出し側で判定をスキップさせず、この関数自身が迂回不能な唯一の
/// 判定経路として fail-closed を保証する）。
#[track_caller]
pub fn assert_no_parity_regression(
    context: &str,
    report: &backend_cpu::CompareReport,
    baseline: &ParityBaseline,
) {
    assert!(
        !baseline.basic_kernel_baseline_unconfirmed,
        "{context}: parity 非後退契約 FAIL — この行は基本版カーネル専用の \
         確定ベースラインが未整備です（basic_kernel_baseline_unconfirmed \
         == true）。provenance 不確実性により非後退判定に使えないため、\
         黙って skip せず fail-closed で失敗させています。CUDA 実機で \
         基本版カーネル単独を再測定し、確定値を記録したうえで \
         basic_kernel_baseline_unconfirmed: false へ更新してください \
         （推定値の記入は禁止。docs/perf/cuda-parity-baseline.md \
         §「既知の限界」・§「ベースライン更新規約」参照。実測 fail_count=\
         {}/{}, mean_abs_diff={:.6e} は参考値であり合否判定には使っていません）",
        report.fail_count, report.total, report.mean_abs_diff,
    );
    assert_eq!(
        report.total, baseline.total,
        "{context}: 比較対象の要素数が baseline({}) と一致しません（形状・\
         比較対象がずれている可能性があります。baseline={:?}）",
        baseline.total, baseline
    );
    assert!(
        report.fail_count <= baseline.baseline_fail_count,
        "{context}: parity 非後退契約 FAIL — fail_count が後退しました \
         (actual={}/{}, baseline={}/{}, mean_abs_diff={:.6e}, \
         baseline_mean_abs_diff_ceiling={:.6e})",
        report.fail_count,
        report.total,
        baseline.baseline_fail_count,
        baseline.total,
        report.mean_abs_diff,
        baseline.baseline_mean_abs_diff_ceiling,
    );
    assert!(
        report.mean_abs_diff <= baseline.baseline_mean_abs_diff_ceiling,
        "{context}: parity 非後退契約 FAIL — mean_abs_diff が後退しました \
         (actual={:.6e}, baseline_ceiling={:.6e}, fail_count={}/{}, \
         baseline_fail_count={})",
        report.mean_abs_diff,
        baseline.baseline_mean_abs_diff_ceiling,
        report.fail_count,
        report.total,
        baseline.baseline_fail_count,
    );
}

/// tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）の
/// 無断変更を機械検知する（受け入れ基準 3）。
///
/// bit 等値で assert する: 浮動小数点定数の意図しない再定義（丸め違い等）
/// も検出対象に含めるため、`==` による厳密比較を用いる。
#[track_caller]
pub fn assert_tolerance_constants_pinned() {
    assert_eq!(
        backend_cpu::RELATIVE_TOLERANCE,
        1e-3,
        "RELATIVE_TOLERANCE が無断変更されています（ガードレール閾値の変更は \
         ユーザー承認必須。.claude/rules/security.md A08）"
    );
    assert_eq!(
        backend_cpu::ABSOLUTE_RESCUE_THRESHOLD,
        1e-5,
        "ABSOLUTE_RESCUE_THRESHOLD が無断変更されています（ガードレール閾値の \
         変更はユーザー承認必須。.claude/rules/security.md A08）"
    );
}
