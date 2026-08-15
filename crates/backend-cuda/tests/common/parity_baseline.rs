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
    },
    // wmma_tf32: K=4096 ストレスケース先頭（256x256x4096、seed=8888）。
    // `tests/gemm_wmma_tf32.rs::wmma_tf32_k4096_stress_poc_v2_5` の先頭呼出し。
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
#[track_caller]
pub fn assert_no_parity_regression(
    context: &str,
    report: &backend_cpu::CompareReport,
    baseline: &ParityBaseline,
) {
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
