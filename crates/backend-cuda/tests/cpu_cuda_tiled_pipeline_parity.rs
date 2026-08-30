//! tiled pipeline（cp.async 多段パイプラインを FP32 SIMT 経路へ導入した
//! 変種カーネル。イシュー #1033）の CPU-CUDA 数値一致回帰テスト。
//!
//! `tests/gemm_wmma_tf32_staged.rs` と同じ方針で、判定式・閾値は
//! `fandhe_ai_backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満」の唯一の実体）に一本化し、ここでローカル
//! 複製しない（`.claude/rules/coding-rust.md`）。
//!
//! **本番経路との関係**: `CudaGemm::run_tiled_pipeline_f32` は本番既定
//! 経路（`run_tiled_f32`）を置き換えない選択可能な変種であり、本ファイルの
//! 全テストは明示的にこの API を呼ぶ（`kernels_tiled_pipeline.rs` 冒頭
//! コメント「位置づけ・非結線」参照）。
//!
//! **既存 `run_tiled_f32` との相互比較**: `tiled_pipeline_matches_tiled_f32`
//! が、CPU 参照値だけでなく既存 `TILED_F32` カーネルの出力とも複合判定で
//! 一致することを確認する（実装計画 §4「既存 tiled f32 との相互比較」）。
//!
//! **実機依存の分離**: 環境適応スモークのみ通常 CI で実行、CUDA/NVRTC
//! 非搭載環境・cp.async 非対応（sm_80 未満）環境では早期 return で green
//! （`tests/gemm_wmma_tf32_staged.rs` と同じ分岐パターン）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装と `run_tiled_pipeline_f32`
/// の出力を [`fandhe_ai_backend_cpu::assert_parity`] で照合する。
fn assert_tiled_pipeline_parity(gemm: &CudaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_tiled_pipeline_f32 must succeed on a cp.async-capable test runner");

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tests/gemm_wmma_tf32_staged.rs::wmma_tf32_staged_parity_smoke_env_adaptive`
/// と同じ分岐パターン。ブロックタイル 1 個ぶん（64×64×64。4 の倍数のため
/// cp.async 整列条件を満たす）で複合判定を実施する。
#[test]
fn tiled_pipeline_parity_smoke_env_adaptive() {
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

    if !gemm.tiled_pipeline_available() {
        // cp.async は sm_80 (Ampere) 以降限定。本番 5 カーネル
        // （naive/tiled）は道連れにならない（`gemm.rs::CudaGemm::new`
        // ドキュメンテーションコメント参照）。
        let reason = gemm.tiled_pipeline_unavailable_reason();
        assert!(reason.is_some_and(|r| !r.is_empty()));
        return;
    }

    assert_tiled_pipeline_parity(&gemm, "smoke 64x64x64", 1, 64, 64, 64);
}

/// 実機（compute capability 8.0 以降、cp.async 対応）必須の形状網羅
/// テスト。受け入れ条件の本体。タイル倍数形状・4 の倍数の非タイル倍数
/// エッジ形状（cp.async 16 バイト整列条件は満たすがブロックタイル・K
/// タイル非倍数）を含む（実装計画 §4「タイル倍数形状・4 の倍数の
/// 非タイル倍数エッジ形状」）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    let cases: &[(u32, u32, u32)] = &[
        // ブロックタイル倍数（64）。
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        // 4 の倍数だがブロックタイル（64）・K タイル（16）の非倍数
        // （cp.async 4 要素整列は満たすが末尾タイルの guarded load・
        // guarded store 分岐を実際に踏む）。
        (60, 68, 36),
        (68, 60, 20),
        // 非正方。
        (64, 96, 256),
        // 極小（4 の倍数の最小形状）。
        (4, 4, 4),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_tiled_pipeline_parity(&gemm, &context, 5000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（`wmma_tf32_staged_k4096_stress` と同じ形状。
/// PoC-v2-3 の M=N=K=4096 と揃える）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    assert_tiled_pipeline_parity(&gemm, "K4096 stress 512x512x4096", 0xC0FFEE, 512, 512, 4096);
    assert_tiled_pipeline_parity(
        &gemm,
        "K4096 stress 4096x4096x4096",
        0xBEEF,
        4096,
        4096,
        4096,
    );
}

/// 実装計画 §4「既存 tiled f32 との相互比較」: 同一入力に対する
/// `run_tiled_f32`（本番既定経路）と `run_tiled_pipeline_f32`（本イシュー
/// の変種）の出力が統一複合判定で一致することを確認する。両カーネルは
/// 演算順序が異なる（`TILED_F32` は 32×32 タイル・1 スレッド 1 要素、
/// tiled pipeline は 64×64 タイル・4×4 レジスタブロッキング）ため bit
/// 完全一致は主張しない。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_matches_tiled_f32() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    let cases: &[(u32, u32, u32)] = &[(64, 64, 64), (256, 256, 256), (60, 68, 36)];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let mut rng = bench_harness::rng::Xorshift64Star::new(6000 + idx as u64);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let c_tiled = gemm
            .run_tiled_f32(&a, &b, m, n, k)
            .expect("run_tiled_f32 must succeed on CUDA-equipped test runner");
        let c_pipeline = gemm
            .run_tiled_pipeline_f32(&a, &b, m, n, k)
            .expect("run_tiled_pipeline_f32 must succeed on cp.async-capable test runner");

        let context = format!("tiled_f32 vs tiled_pipeline shape m={m} n={n} k={k}");
        fandhe_ai_backend_cpu::assert_parity(&context, &c_pipeline, &c_tiled);
    }
}

/// `k == 0`（`num_k_tiles == 0` 経路）で C が全 0 になることを確認する
/// （`tests/gemm_wmma_tf32_staged.rs::wmma_tf32_staged_zero_k_returns_all_zero`
/// の tiled pipeline 版）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    let (m, n, k) = (4u32, 4u32, 0u32);
    let c = gemm
        .run_tiled_pipeline_f32(&[], &[], m, n, k)
        .expect("k==0 must be a valid no-accumulation shape, not a launch error");
    assert_eq!(c.len(), (m as usize) * (n as usize));
    assert!(c.iter().all(|&v| v == 0.0), "k==0 output must be all zero");
}

/// m==0／n==0 の no-op 形状（`tests/gemm_wmma_tf32_staged.rs::
/// wmma_tf32_staged_zero_dim_shape_returns_empty_without_launch` の
/// tiled pipeline 版）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");

    // m==0: a は空（m*k==0）、b は k*n==16 要素。
    let c = gemm
        .run_tiled_pipeline_f32(&[], &[0.0; 16], 0, 4, 4)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    // n==0: a は m*k==16 要素、b は空（k*n==0）。
    let c = gemm
        .run_tiled_pipeline_f32(&[0.0; 16], &[], 4, 0, 4)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}

/// cp.async 16 バイト整列条件（`n % 4 == 0 && k % 4 == 0`）を満たさない
/// 形状は `CudaError::InvalidShape` を返すことを確認する（フォールバック
/// 経路を持たない単独の選択可能変種のため fail-closed に拒否する契約。
/// `gemm.rs::tiled_pipeline_alignment_ok` ドキュメンテーションコメント
/// 参照）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_rejects_misaligned_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    // n=5 は 4 の倍数でない（a: m*k=16 要素、b: k*n=20 要素）。
    let err = gemm
        .run_tiled_pipeline_f32(&[0.0; 16], &[0.0; 20], 4, 5, 4)
        .expect_err("misaligned n must be rejected");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}

/// codex-review P1 指摘（PR #1071）の回帰テスト: `compile_tiled_pipeline_variant`
/// で**別の** `CudaDevice`（＝別 `CudaContext`。`CudaDevice::new` は同一
/// ordinal でも呼ぶたびに新しい `CudaContext` を生成する。
/// `context_cache.rs::ContextKey` ドキュメントコメント参照）から得た
/// ハンドルを、それとは異なる `CudaGemm` インスタンスへ渡すと、
/// `launch_tiled_pipeline_f32` が `unsafe` launch へ到達する前に
/// `CudaError::TiledPipelineContextMismatch` を返し fail-closed に拒否する
/// ことを確認する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_rejects_mismatched_context_handle() {
    // `gemm` を構築する context と、`other_func` を構築する context を
    // 意図的に分ける（同一 ordinal でも `CudaDevice::new` の呼び出しごとに
    // 別 `CudaContext` が生成される）。
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    let other_device =
        CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let other_func = CudaGemm::compile_tiled_pipeline_variant(&other_device, 3)
        .expect("tiled pipeline variant compilation must succeed on the other context");

    let (a_dev, b_dev) = gemm
        .upload_f32(&[0.0f32; 16], &[0.0f32; 16])
        .expect("upload_f32 must succeed for a well-formed 4x4x4 shape");
    let mut c_dev = gemm
        .alloc_output_f32(4, 4)
        .expect("alloc_output_f32 must succeed for a well-formed 4x4 output shape");

    let err = gemm
        .launch_tiled_pipeline_f32(&other_func, &a_dev, &b_dev, &mut c_dev, 4, 4, 4)
        .expect_err(
            "launching a handle compiled against a different CudaContext must be rejected \
             before reaching the unsafe launch",
        );
    assert!(matches!(
        err,
        CudaError::TiledPipelineContextMismatch { .. }
    ));
}
