//! parity 非後退契約のテスト（イシュー #491・GEMM 性能改善ツリー B-0）。
//!
//! `wmma_tf32`・`wmma_tf32_opt`・`mma_f16` 系経路は REQ-2 統一複合判定で
//! 恒常 fail の既知状態にある（#186・`docs/backend-cuda-real-device-testing.md`
//! §5.3）。以降の Phase B/C カーネル改修は「parity green」ではなく
//! 「非後退」（fail 比率・平均絶対誤差が記録済みベースラインを上回らない）
//! で受け入れ判定する必要があり、本ファイルはその機械検査を提供する。
//!
//! # テスト構成
//!
//! - **通常 CI（`#[ignore]` なし・GPU 不要）**: [`common::parity_baseline`]
//!   の tolerance 定数 pin・fixture 自己整合性・検査関数自体の
//!   falsification（常に pass する壊れ方の防止）を検査する。GPU・CUDA
//!   非搭載環境（GitHub ホステッド runner 等）でも実行できる純粋なロジック
//!   検査のみで構成する。
//! - **実機必須（`#[ignore = "CUDA 実機必須"]`）**: fixture の各行について
//!   記録時と同一の (エントリポイント・形状・シード・データ生成・参照計算
//!   経路) で GPU 実行し、[`common::parity_baseline::assert_no_parity_regression`]
//!   で非後退を検査する。既存の `assert_parity` ベーステスト（最初の fail
//!   で panic）と異なり、fixture の全行を検査する（1 行の fail で残りの
//!   行の検査を打ち切らない。どの行が後退したかを特定可能にするため）。

mod common;

use backend_cuda::{CudaDevice, CudaGemm, CudaMmaGemm};
use common::parity_baseline::{
    BASELINES, ParityBaseline, ParityPath, assert_no_parity_regression,
    assert_tolerance_constants_pinned,
};
use half::f16;

// --- 通常 CI テスト（GPU 不要） ---

/// 受け入れ基準 3: tolerance 定数（`RELATIVE_TOLERANCE`/
/// `ABSOLUTE_RESCUE_THRESHOLD`）の無断変更を機械検知する。
#[test]
fn tolerance_constants_are_pinned() {
    assert_tolerance_constants_pinned();
}

/// fixture 自体の妥当性検査: 各行の `baseline_fail_count <= total`・
/// `total == m*n`・3 経路すべてに 1 行以上存在することを確認する。
/// fixture 値の入力ミス（転記ミス等）を CI で機械的に検出する。
#[test]
fn baseline_fixture_is_self_consistent() {
    assert!(!BASELINES.is_empty(), "BASELINES must not be empty");

    for b in BASELINES {
        let expected_total = (b.m as usize) * (b.n as usize);
        assert_eq!(
            b.total, expected_total,
            "{}: total は m*n と一致する必要があります（total={}, m*n={}）",
            b.context, b.total, expected_total
        );
        assert!(
            b.baseline_fail_count <= b.total,
            "{}: baseline_fail_count({}) が total({}) を超えています",
            b.context,
            b.baseline_fail_count,
            b.total
        );
        assert!(
            b.baseline_mean_abs_diff_ceiling >= 0.0 && b.baseline_mean_abs_diff_ceiling.is_finite(),
            "{}: baseline_mean_abs_diff_ceiling は有限の非負値である必要があります（値={}）",
            b.context,
            b.baseline_mean_abs_diff_ceiling
        );
    }

    for path in [
        ParityPath::WmmaTf32,
        ParityPath::WmmaTf32Opt,
        ParityPath::MmaF16,
    ] {
        assert!(
            BASELINES.iter().any(|b| b.path == path),
            "経路 {path} に対応する baseline 行が 1 件も存在しません"
        );
    }
}

/// 検査関数自体の falsification テスト（`.claude/rules/coding-rust.md`
/// 「本番経路で unwrap/expect を使わない」とは別軸の品質観点: 検査
/// ユーティリティが「常に pass する」壊れ方をしていないことを固定する。
/// `backend-cpu/src/parity.rs` の既存テストと同方針）。
///
/// ベースラインを人為的に上回る合成 `CompareReport` を与えたとき、
/// `assert_no_parity_regression` が panic することを確認する。
#[test]
#[should_panic(expected = "非後退契約 FAIL")]
fn assert_no_parity_regression_panics_on_fail_count_regression() {
    let baseline = ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "synthetic",
        m: 4,
        n: 4,
        k: 4,
        seed: 1,
        total: 16,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 1e-4,
    };
    // fail_count がベースライン(2)を上回る合成レポート。
    let a = vec![0.0f32; baseline.total];
    let mut b = vec![0.0f32; baseline.total];
    // 3 セルを大きく乖離させ、複合判定で fail させる
    // （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満のいずれも満たさない値）。
    b[0] = 1.0;
    b[1] = 1.0;
    b[2] = 1.0;
    let report = backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic", &report, &baseline);
}

/// mean_abs_diff 側の非後退違反を単独で検知することを確認する
/// （fail_count は据え置き、mean_abs_diff のみベースライン超過とする
/// 合成ケース）。
#[test]
#[should_panic(expected = "非後退契約 FAIL")]
fn assert_no_parity_regression_panics_on_mean_abs_diff_regression() {
    let baseline = ParityBaseline {
        path: ParityPath::MmaF16,
        context: "synthetic mean_abs_diff",
        m: 2,
        n: 2,
        k: 2,
        seed: 1,
        total: 4,
        // fail_count は緩く設定し、mean_abs_diff 側のみで fail させる。
        baseline_fail_count: 4,
        baseline_mean_abs_diff_ceiling: 1e-6,
    };
    let a = vec![0.0f32; baseline.total];
    // 絶対誤差救済閾値(1e-5)未満のため複合判定は pass するが、
    // mean_abs_diff(約 5e-6) は baseline_mean_abs_diff_ceiling(1e-6) を上回る。
    let b = vec![5e-6f32; baseline.total];
    let report = backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic mean_abs_diff", &report, &baseline);
}

/// total 不一致（形状・比較対象のずれ）を fail-closed で検知することを
/// 確認する。
#[test]
#[should_panic(expected = "baseline")]
fn assert_no_parity_regression_panics_on_total_mismatch() {
    let baseline = ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "synthetic total mismatch",
        m: 4,
        n: 4,
        k: 4,
        seed: 1,
        total: 16,
        baseline_fail_count: 100,
        baseline_mean_abs_diff_ceiling: 1.0,
    };
    // total が baseline(16) と異なる合成レポート。
    let a = vec![0.0f32; 9];
    let b = vec![0.0f32; 9];
    let report = backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic total mismatch", &report, &baseline);
}

// --- 実機必須テスト（`#[ignore]`。CUDA 実機・compute capability 8.0 以降必須） ---

/// fixture の各行を記録時と同一の入力（seed・形状・生成手順）で再現し、
/// GPU 実行結果を非後退判定する。1 行の fail で残りの行の検査を打ち切ら
/// ない（複数行が同時に後退した場合でもすべて検出できるよう、最後に
/// まとめて assert する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn parity_baselines_do_not_regress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    let mma_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let mut failures: Vec<String> = Vec::new();

    for baseline in BASELINES {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match baseline.path {
                ParityPath::WmmaTf32 | ParityPath::WmmaTf32Opt => {
                    check_wmma_tf32_baseline(&gemm, baseline);
                }
                ParityPath::MmaF16 => {
                    check_mma_f16_baseline(&mma_gemm, baseline);
                }
            }));
        if let Err(err) = result {
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic (詳細不明)".to_string());
            failures.push(format!("{}: {msg}", baseline.context));
        }
    }

    assert!(
        failures.is_empty(),
        "parity 非後退契約 FAIL（{} 件）:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// TF32 経路（`wmma_tf32`/`wmma_tf32_opt`）1 行の非後退検査。
///
/// 参照値は `backend_cpu::matmul_reference_fma`（非量子化。
/// `docs/backend-cuda-real-device-testing.md` §5.3 が確認済みの意図的設計を
/// 踏襲する）。
///
/// **PR #640 codex-review 指摘対応**: 以前は `WmmaTf32`（基本版）・
/// `WmmaTf32Opt` の両行を `run_wmma_tf32` に送っていたが、この API は opt
/// カーネルが利用可能なら常に opt を優先する（`gemm.rs::run_wmma_tf32`
/// 参照）ため、opt が使える実機では `WmmaTf32` 行も実質 opt カーネルを
/// 検査してしまい、基本版カーネル単独の後退を検出できない盲点があった。
/// 現在は経路ごとにエントリポイントを分ける: `WmmaTf32` 行は
/// [`CudaGemm::run_wmma_tf32_basic_for_test`]（`internal-testing` feature
/// でのみコンパイルされるテスト専用口。基本版カーネルを opt の可用性に
/// 関わらず直接指定する。`crates/backend-cuda/Cargo.toml` 参照）を、
/// `WmmaTf32Opt` 行は引き続き `run_wmma_tf32` を使う。どちらも「実際に
/// 意図した版のカーネルを踏んだ」ことを事前の可用性 assert で保証してから
/// 実行する（`wmma_tf32_available`/`wmma_tf32_opt_available` の対称な
/// 事前検査。既存テスト `gemm_wmma_tf32_opt.rs` と同じ考え方）。
fn check_wmma_tf32_baseline(gemm: &CudaGemm, baseline: &ParityBaseline) {
    match baseline.path {
        ParityPath::WmmaTf32 => assert!(
            gemm.wmma_tf32_available(),
            "{}: basic WMMA(TF32) kernel must be available on this ignored test runner (reason: {:?})",
            baseline.context,
            gemm.wmma_tf32_unavailable_reason()
        ),
        ParityPath::WmmaTf32Opt => assert!(
            gemm.wmma_tf32_opt_available(),
            "{}: opt kernel must be available on this ignored test runner (reason: {:?})",
            baseline.context,
            gemm.wmma_tf32_opt_unavailable_reason()
        ),
        ParityPath::MmaF16 => unreachable!("check_wmma_tf32_baseline は TF32 系経路専用"),
    }

    let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
    let a = rng.fill_vec((baseline.m as usize) * (baseline.k as usize));
    let b = rng.fill_vec((baseline.k as usize) * (baseline.n as usize));

    let mut c_ref = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
    backend_cpu::matmul_reference_fma(
        &a,
        &b,
        &mut c_ref,
        baseline.m as usize,
        baseline.n as usize,
        baseline.k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed baseline input");

    let c_gpu = match baseline.path {
        ParityPath::WmmaTf32 => gemm
            .run_wmma_tf32_basic_for_test(&a, &b, baseline.m, baseline.n, baseline.k)
            .expect(
                "CudaGemm::run_wmma_tf32_basic_for_test must succeed on a compute capability >= 8.0 test runner",
            ),
        ParityPath::WmmaTf32Opt => gemm
            .run_wmma_tf32(&a, &b, baseline.m, baseline.n, baseline.k)
            .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner"),
        ParityPath::MmaF16 => unreachable!("check_wmma_tf32_baseline は TF32 系経路専用"),
    };

    let report = backend_cpu::compare(&c_gpu, &c_ref).expect("shape must match baseline fixture");
    assert_no_parity_regression(baseline.context, &report, baseline);
}

/// f16 経路（`mma_f16`）1 行の非後退検査。
///
/// 参照値は「f16→f32→`matmul_reference_fma`→f16 丸め→f32」の量子化込み
/// 経路（`tests/cpu_cuda_mma_parity.rs::assert_mma_f16_parity` と同一手順。
/// GPU 側エピローグ store の丸めを参照側にも反映させる）。
fn check_mma_f16_baseline(gemm: &CudaMmaGemm, baseline: &ParityBaseline) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((baseline.m as usize) * (baseline.k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((baseline.k as usize) * (baseline.n as usize));

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
    backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        baseline.m as usize,
        baseline.n as usize,
        baseline.k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed baseline input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .run_f16(&a_f16, &b_f16, baseline.m, baseline.n, baseline.k)
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    let report = backend_cpu::compare(&c_gpu_f32, &c_ref_rounded)
        .expect("shape must match baseline fixture");
    assert_no_parity_regression(baseline.context, &report, baseline);
}
