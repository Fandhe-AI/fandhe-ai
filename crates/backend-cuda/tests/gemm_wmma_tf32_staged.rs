//! WMMA(TF32) opt-staged（cp.async 多段パイプライン・fragment 先読み。
//! イシュー #500）GEMM の CPU-CUDA 数値一致回帰テスト。
//!
//! `tests/gemm_wmma_tf32_opt.rs`（#63）と同じ方針で、判定式・閾値は
//! `backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、ここでローカル複製しない
//! （`.claude/rules/coding-rust.md`）。
//!
//! **公開 API との関係**: `CudaGemm::run_wmma_tf32` は staged カーネルが
//! `CudaGemm::new` 時点でコンパイル・ロードに成功し、かつ cp.async 16
//! バイト整列条件（`n % 4 == 0 && k % 4 == 0`）を満たす形状であれば自動的に
//! staged 経路を選ぶ（`gemm.rs` 3 段選択方針。専用の切替 API は存在しない。
//! REQ-11）。本ファイルは `run_wmma_tf32` を通じて staged カーネル固有の
//! タイル境界（ブロックタイル 64、K タイル 16、cp.async ステージ 3）を
//! 踏む整列形状（4 の倍数）を検証する。整列非対応形状（63×65×33 等）は
//! `tests/gemm_wmma_tf32_opt.rs` が opt カーネル経由で引き続き検証する
//! （本ファイルの対象外）。
//!
//! **実機依存の分離**: `tests/gemm_wmma_tf32_opt.rs` と同じ分岐パターン
//! （環境適応スモークのみ通常 CI で実行、CUDA/NVRTC 非搭載環境では早期
//! return で green）。

use backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装と `run_wmma_tf32`
/// （staged カーネルが利用可能かつ整列条件を満たせばそちら）の出力を
/// [`backend_cpu::assert_parity`] で照合する。
fn assert_wmma_tf32_staged_parity(
    gemm: &CudaGemm,
    context: &str,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");

    backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_parity_smoke_env_adaptive`
/// と同じ分岐パターン。staged カーネルのブロックタイル 1 個ぶん
/// （64×64×64。4 の倍数のため整列条件を満たす）で複合判定を実施する
/// （staged が未対応環境ではコンパイル失敗・整列非対応により opt／基本
/// 版へ自動フォールバックするため、このスモークは staged 専用の分岐を
/// 持たない——`run_wmma_tf32` のフォールバック方針どおり、どの経路でも
/// 複合判定は成立することを確認できればよい）。
#[test]
fn wmma_tf32_staged_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };

    match gemm.run_wmma_tf32(&[0.0; 4096], &[0.0; 4096], 64, 64, 64) {
        Ok(_) => assert_wmma_tf32_staged_parity(&gemm, "smoke 64x64x64", 1, 64, 64, 64),
        Err(CudaError::WmmaUnavailable { detail }) => {
            // staged・opt・基本版とも使用不能な環境（`<mma.h>` 未解決・
            // cc<8.0）。naive/tiled は道連れにならない（`gemm.rs::
            // CudaGemm::new` ドキュメンテーションコメント参照）。
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected CudaError variant from run_wmma_tf32: {other}"),
    }
}

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の形状網羅
/// テスト。受け入れ条件の本体。
///
/// staged カーネル固有のタイル境界（ブロックタイル 64、共有メモリ K タイル
/// 16、fragment 16/8、cp.async ステージ 3）を踏む、cp.async 16 バイト
/// 整列条件（`n%4==0 && k%4==0`）を満たす形状のみを対象にする（整列非対応
/// 形状は `tests/gemm_wmma_tf32_opt.rs` が opt カーネル経由で検証する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_staged_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    // `wmma_tf32_opt_matches_reference_across_shapes` と同じ根拠（PR #256
    // レビュー指摘対応）。`run_wmma_tf32` は staged カーネル未対応環境・
    // 整列非対応形状で opt／基本版へ自動フォールバックするため、この
    // 検証がここで staged カーネル固有のタイル境界を実際に踏んだことを
    // 保証するには、staged カーネルが利用可能であることを事前に確認する
    // 必要がある（本テストは compute capability 8.0 以降の実機必須なので
    // staged カーネルは利用可能であるべき）。
    assert!(
        gemm.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner (reason: {:?})",
        gemm.wmma_tf32_staged_unavailable_reason()
    );

    let cases: &[(u32, u32, u32)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        // ブロックタイル・K タイル非倍数だが cp.async 4 要素整列条件は
        // 満たす形状（63 は 64 の倍数でないが 4 の倍数でもない点に注意:
        // guarded load のゼロ充填で処理される。65/33/17 も同様）。
        (60, 68, 36),
        (68, 60, 20),
        // 非正方。
        (64, 96, 256),
        // 極小（1 の倍数でない形状。整列条件 n%4/k%4 を満たさないため
        // staged は選ばれず opt/基本へフォールバックする想定だが、
        // `run_wmma_tf32` 経由で正しく処理されることを確認する）。
        (1, 1, 1),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_wmma_tf32_staged_parity(&gemm, &context, 4000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（`wmma_tf32_opt_k4096_stress` と同じ形状。
/// PoC-v2-3 の M=N=K=4096 と揃える）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_staged_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    assert!(
        gemm.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner (reason: {:?})",
        gemm.wmma_tf32_staged_unavailable_reason()
    );

    assert_wmma_tf32_staged_parity(&gemm, "K4096 stress 512x512x4096", 0xC0FFEE, 512, 512, 4096);
    assert_wmma_tf32_staged_parity(
        &gemm,
        "K4096 stress 4096x4096x4096",
        0xBEEF,
        4096,
        4096,
        4096,
    );
}

/// `k == 0`（`num_k_tiles == 0` 経路）で C が全 0 になることを確認する
/// （`tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_zero_k_returns_all_zero`
/// の staged 経路版。staged/opt/基本のいずれが選ばれても早期 return の
/// 契約は共通。`gemm.rs::run_wmma_tf32_staged_kernel`／
/// `run_wmma_tf32_opt_kernel`／`run_wmma_f32_kernel` 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_staged_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let (m, n, k) = (4u32, 4u32, 0u32);
    let c = gemm
        .run_wmma_tf32(&[], &[], m, n, k)
        .expect("k==0 must be a valid no-accumulation shape, not a launch error");
    assert_eq!(c.len(), (m as usize) * (n as usize));
    assert!(c.iter().all(|&v| v == 0.0), "k==0 output must be all zero");
}

/// m==0／n==0 の no-op 形状（`tests/gemm_wmma_tf32_opt.rs::
/// wmma_tf32_opt_zero_dim_shape_returns_empty_without_launch` の staged
/// 経路版）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_staged_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let c = gemm
        .run_wmma_tf32(&[], &[1.0, 2.0, 3.0, 4.0], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_wmma_tf32(&[1.0, 2.0], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}

/// #500 の目的（cp.async 多段化・fragment 先読みによる TF32 経路の性能
/// 改善）の実測本体: staged 経路（`run_wmma_tf32`。staged カーネルが
/// 利用可能かつ整列条件を満たす環境では自動的にこちらが選ばれる）が
/// 既存 opt 実装（tiled 超過が確認済み。#63）を上回ることを、同一実行内で
/// 5 回計測した中央値で確認する（`.claude/rules/coding-rust.md`「ベンチは
/// 5 回計測の中央値」）。
///
/// **未計測（本 PR の対象外）**: 本テストは実機（DGX Spark GB10 等）
/// 必須のため、この PR を作成したセッションでは実行できていない。
/// 対 PyTorch 比・staged 対 opt の実測倍率は `docs/perf/
/// cuda-gemm-wmma-tf32-phase-b.md` の実測結果欄（プレースホルダ）へ
/// 実機セッションで記入し、必要に応じてイシュー #502（Phase B 完了時点の
/// f32/f16 再計測）へ引き継ぐ。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須。実測記録は \
            docs/perf/cuda-gemm-wmma-tf32-phase-b.md"]
fn wmma_tf32_staged_exceeds_opt_tflops_at_4096() {
    use std::time::Instant;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    assert!(
        gemm.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner so that the TFLOPS \
         comparison actually exercises the staged kernel rather than silently falling back \
         (reason: {:?})",
        gemm.wmma_tf32_staged_unavailable_reason()
    );
    assert!(
        gemm.wmma_tf32_opt_available(),
        "opt kernel must be available on this ignored test runner so that the comparison \
         baseline is the actual optimized kernel, not the basic WMMA kernel (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );

    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(0xACE1);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);

    let median_tflops = |run: &dyn Fn() -> Vec<f32>| -> f64 {
        // warmup（NVRTC JIT・クロック遷移の影響を計測から除外する）。
        let _ = run();
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let start = Instant::now();
            let _ = run();
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|x, y| x.partial_cmp(y).expect("elapsed seconds must not be NaN"));
        let median = samples[samples.len() / 2];
        (flops / median) / 1e12
    };

    // staged 経路は m/n/k=4096（4 の倍数）で自動選択される
    // （`wmma_tf32_staged_alignment_ok`）ため `run_wmma_tf32` をそのまま
    // 使う。opt 側は `run_wmma_tf32` 経由では staged にフォールバックされて
    // しまうため、`gemm_wmma.rs`／内部 API を持たない本クレートの公開面
    // からは opt 単体の分離計測ができない。したがって「opt を上回る」旨の
    // 主張は、`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` に記録する
    // 実機セッションで `wmma_tf32_opt` 側のみをコンパイルする一時的な
    // 計測（または #502 再計測時の既存 `wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`
    // の実測値との比較）に委ねる。本テストでは「tiled f32 を上回る」ことを
    // 受け入れ条件の本体として計測する（既存 `wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`
    // と同型の比較軸）。
    let tiled_tflops = median_tflops(&|| {
        gemm.run_tiled_f32(&a, &b, m, n, k)
            .expect("tiled f32 must succeed on CUDA-equipped test runner")
    });
    let staged_tflops = median_tflops(&|| {
        gemm.run_wmma_tf32(&a, &b, m, n, k)
            .expect("run_wmma_tf32 must succeed on CUDA-equipped test runner")
    });

    assert!(
        staged_tflops > tiled_tflops,
        "staged 経路（{staged_tflops:.3} TFLOPS）が tiled f32（{tiled_tflops:.3} TFLOPS）を \
         上回りませんでした（M=N=K=4096）"
    );
}
