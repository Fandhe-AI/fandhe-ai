//! f16 WMMA opt（共有メモリ・タイル最適化版）GEMM の CPU-CUDA 数値一致
//! 回帰テスト（TASK-11.1d・#63）。
//!
//! `tests/cpu_cuda_wmma_parity.rs`（#61）と同じ方針・同じ複合判定例外
//! （WMMA f16 経路は #61 の受け入れ条件により複合判定〈1e-3/1e-5〉の対象。
//! 本ファイル冒頭コメント参照元の説明をそのまま踏襲する）。判定は
//! `fandhe_ai_backend_cpu::assert_parity` に一本化し閾値をローカル複製しない。
//!
//! **公開 API との関係**: `CudaWmmaGemm::run_f16` は opt カーネルが
//! `CudaWmmaGemm::new` 時点でコンパイル・ロードに成功していれば自動的に
//! opt 経路を選ぶ（`gemm_wmma.rs` フォールバック方針。専用の切替 API は
//! 存在しない）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaWmmaGemm};
use half::f16;

mod common;

/// 決定的シードで A・B（f16）を生成し、f16→f32→参照 matmul→f16 丸め→f32 の
/// 経路で得た参照値と `run_f16`（opt カーネルが利用可能ならそちら）の出力を
/// `assert_parity` で照合する（`tests/cpu_cuda_wmma_parity.rs::
/// assert_wmma_f16_parity` と同一手順）。
fn assert_wmma_f16_opt_parity(
    gemm: &CudaWmmaGemm,
    context: &str,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("CudaWmmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu_f32, &c_ref_rounded);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tests/cpu_cuda_wmma_parity.rs::wmma_f16_parity_smoke_env_adaptive` と
/// 同じ分岐パターン。opt カーネルのブロックタイル 1 個ぶん（64×64×64）で
/// 複合判定を実施する。
#[test]
fn wmma_f16_opt_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaWmmaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaWmmaGemm::new: {other}"),
    };

    assert_wmma_f16_opt_parity(&gemm, "smoke 64x64x64", 1, 64, 64, 64);
}

/// 実機（DGX Spark GB10 等）必須の形状網羅テスト。受け入れ条件の本体。
///
/// opt カーネル固有のタイル境界（ブロックタイル 64、fragment 16）を踏む
/// 形状を含む: ブロックタイル倍数（64×64×64・128×128×128）・非倍数境界
/// （63×65×33・65×63×17）・非正方（64×96×256）・極小（1×1×1）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_opt_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (63, 65, 33),
        (65, 63, 17),
        (64, 96, 256),
        (1, 1, 1),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_wmma_f16_opt_parity(&gemm, &context, 4000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（`tests/cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress`
/// と同じ形状で opt 経路の桁落ち耐性を確認する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_opt_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    assert_wmma_f16_opt_parity(&gemm, "K4096 stress 256x256x4096", 8889, 256, 256, 4096);
}

/// `wmma_f16_opt_k4096_stress`（256×256×4096・seed=8889）に対する**非後退
/// 監視の併設テスト**（イシュー #1106・GB10 全件洗い出し。
/// `tests/cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress_non_regression`
/// と同型のパターン）。
///
/// `run_f16`（opt カーネルが利用可能な環境では自動的にそちらへルーティング
/// される公開 API）は K=4096 蓄積により既知の tail 超過を持つため、
/// `wmma_f16_opt_k4096_stress` 本体は `assert_parity`（REQ-2 受け入れ条件）
/// を維持したまま、本テストは parity 非後退契約で「既知の不合格分布から
/// 悪化していないか」を別観点として監視する。**`assert_parity` を置き換
/// えるものではない**（`wmma_f16_k4096_stress_non_regression` ドキュメン
/// テーションコメントと同じ設計。`docs/perf/cuda-parity-baseline.md`
/// §10.4 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_opt_k4096_stress_non_regression() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let (m, n, k, seed) = (256u32, 256u32, 4096u32, 8889u64);
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();
    let c_gpu_f16 = gemm
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("CudaWmmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    let baseline = common::parity_baseline::BASELINES
        .iter()
        .find(|b| {
            b.path == common::parity_baseline::ParityPath::WmmaF16Opt
                && b.m == m
                && b.n == n
                && b.k == k
                && b.seed == seed
        })
        .expect("wmma_f16_opt 256x256x4096 seed=8889 baseline row must exist in fixture");
    let report = fandhe_ai_backend_cpu::compare(&c_gpu_f32, &c_ref_rounded)
        .expect("shape must match baseline fixture");
    common::parity_baseline::assert_no_parity_regression(
        "wmma_f16_opt_k4096_stress_non_regression 256x256x4096",
        &report,
        baseline,
    );
}

/// k==0（`tests/gemm_wmma.rs::wmma_f16_zero_k_returns_all_zero` の opt
/// 経路版。opt/基本どちらが選ばれても早期 return の契約は共通）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_opt_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[], 2, 3, 0)
        .expect("k==0 must be treated as a no-op returning all-zero C");
    assert_eq!(c, vec![f16::ZERO; 6]);
}

/// m==0／n==0（`tests/gemm_wmma.rs::wmma_f16_zero_dim_shape_returns_empty_without_launch`
/// の opt 経路版）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_opt_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[f16::ONE; 4], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_f16(&[f16::ONE; 2], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}
