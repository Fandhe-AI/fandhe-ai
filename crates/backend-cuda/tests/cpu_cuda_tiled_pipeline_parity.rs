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

/// イシュー #1137 本番結線ゲート A（最優先）: `run_tiled_f32_classic`
/// （classic 版・`kernels::TILED_F32` 直接起動）と `run_tiled_pipeline_f32`
/// （pipeline 版）の出力が**ビット完全一致**することを確認する。
///
/// 両カーネルはタイル（64×64×16）・4×4 レジスタブロッキング・スレッド→
/// 要素マッピング・K 昇順の逐次積和が同一で、差は `acc += a*b`（NVRTC
/// 既定の fmad 縮約）と `fmaf()`（明示）のみのため（`kernels_tiled_pipeline.rs`
/// 冒頭コメント「`kernels::TILED_F32` との違い」参照）、実測で bit 一致が
/// 成立する限り本テストは `assert_eq!` を使う。**FAIL した場合、
/// `select_tiled_f32_kernel`（`gemm.rs`）による本番結線は
/// `docs/kernel-fusion.md` §2.2 の融合 epilogue bit 完全一致契約
/// （`gemm_bias_act` vs 非融合合成）と両立しない可能性があるため、
/// 実装計画 §4 Step 5「不採用分岐」に従い結線を revert する判断材料と
/// する**（本テスト自体は診断用に残す）。
///
/// 旧テスト名 `tiled_pipeline_matches_tiled_f32`（複合判定版）から
/// bit 一致検査へ強化した（#1137）。旧版は `run_tiled_f32`（無印）を
/// base に使っていたが、無印は #1137 以降パイプラインへ分岐しうるため
/// base 側は常に `run_tiled_f32_classic` を使う必要がある。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_matches_tiled_f32_classic_bit_exact() {
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

        let c_classic = gemm
            .run_tiled_f32_classic(&a, &b, m, n, k)
            .expect("run_tiled_f32_classic must succeed on CUDA-equipped test runner");
        let c_pipeline = gemm
            .run_tiled_pipeline_f32(&a, &b, m, n, k)
            .expect("run_tiled_pipeline_f32 must succeed on cp.async-capable test runner");

        assert_eq!(
            c_pipeline, c_classic,
            "tiled_f32_classic vs tiled_pipeline shape m={m} n={n} k={k}: bit 完全一致しません \
             （#1137 本番結線の前提が崩れている）"
        );
    }
}

/// イシュー #1137 本番結線ゲート B の一部: `run_tiled_f32`（無印・本番
/// 既定入口）が `tiled_f32_kernel_kind`（`gemm.rs` 内部の選択判定）の
/// 契約どおり、整列形状では実際に pipeline 版と bit 一致する出力を
/// 返すことを end-to-end で確認する（`select_tiled_f32_kernel` の結線が
/// 機能していることの回帰テスト。`tiled_f32_kernel_for` で分岐先も確認）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn run_tiled_f32_dispatches_to_pipeline_for_aligned_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    let (m, n, k) = (256u32, 256u32, 256u32);
    assert_eq!(
        gemm.tiled_f32_kernel_for(n, k),
        fandhe_ai_backend_cuda::TiledF32Kernel::Pipeline,
        "aligned shape (n={n}, k={k}) must select Pipeline when pipeline is available"
    );

    let mut rng = bench_harness::rng::Xorshift64Star::new(7001);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_dispatch = gemm
        .run_tiled_f32(&a, &b, m, n, k)
        .expect("run_tiled_f32 must succeed on CUDA-equipped test runner");
    let c_pipeline = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("run_tiled_pipeline_f32 must succeed on cp.async-capable test runner");

    assert_eq!(
        c_dispatch, c_pipeline,
        "run_tiled_f32 (dispatch) must match run_tiled_pipeline_f32 bit-for-bit for an \
         aligned shape once #1137 wiring routes it to the pipeline kernel"
    );
}

/// イシュー #1137 本番結線ゲート B の一部: 非整列形状（cp.async 16 バイト
/// 整列制約 `n % 4 == 0 && k % 4 == 0` を満たさない）では `run_tiled_f32`
/// が常に classic 版へフォールバックし、`run_tiled_pipeline_f32`
/// （フォールバックを持たない単独変種）が `InvalidShape` で拒否する形状
/// でも `run_tiled_f32` 自体は成功することを確認する（fail-closed
/// フォールバックの回帰テスト）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn run_tiled_f32_falls_back_to_classic_for_unaligned_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");

    // n=66（4 の倍数でない）・k=34（4 の倍数でない）: 非正方・非整列形状。
    let (m, n, k) = (60u32, 66u32, 34u32);
    assert_eq!(
        gemm.tiled_f32_kernel_for(n, k),
        fandhe_ai_backend_cuda::TiledF32Kernel::Classic,
        "unaligned shape (n={n}, k={k}) must always select Classic"
    );

    let mut rng = bench_harness::rng::Xorshift64Star::new(7002);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_dispatch = gemm
        .run_tiled_f32(&a, &b, m, n, k)
        .expect("run_tiled_f32 must succeed for an unaligned shape via classic fallback");
    let c_classic = gemm
        .run_tiled_f32_classic(&a, &b, m, n, k)
        .expect("run_tiled_f32_classic must succeed on CUDA-equipped test runner");
    assert_eq!(
        c_dispatch, c_classic,
        "unaligned shape must route run_tiled_f32 to the classic kernel bit-for-bit"
    );

    // `run_tiled_pipeline_f32`（フォールバックを持たない単独変種）は
    // 同じ非整列形状を fail-closed に拒否する（既存契約。回帰確認）。
    let err = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect_err("run_tiled_pipeline_f32 must reject unaligned shapes");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
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

/// codex-review P0 指摘（PR #1071）の回帰テスト: `func` は `gemm` 自身の
/// context から生成された正しいハンドルであっても、`a_dev`/`b_dev`/`c_dev`
/// が**別の** `CudaDevice`（＝別 `CudaContext`。上記テストと同じ
/// `CudaDevice::new` の性質）から確保した `CudaSlice` であれば
/// `launch_tiled_pipeline_f32` が `unsafe` launch へ到達する前に
/// `CudaError::TiledPipelineContextMismatch` を返し fail-closed に拒否する
/// ことを確認する。`func` の context 一致検証（上記テスト）だけでは
/// バッファの生成元 context 不一致を検出できないという穴を塞いだことの
/// 回帰テスト（`validate_gemm_dims` は長さのみ検証するため、同じ長さの
/// 別 context バッファは以前この検証を素通りしていた）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_rejects_mismatched_context_buffers() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );
    let func = CudaGemm::compile_tiled_pipeline_variant(&device, 3)
        .expect("tiled pipeline variant compilation must succeed against gemm's own context");

    // `other_gemm` は別 `CudaDevice`（＝別 `CudaContext`）で構築し、その
    // `upload_f32`/`alloc_output_f32` で確保したバッファを `gemm` 側の
    // 正しい `func` と組み合わせて渡す（func は一致・バッファのみ不一致）。
    let other_device =
        CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let other_gemm = CudaGemm::new(&other_device)
        .expect("tiled pipeline kernel compilation must succeed on the other context");

    let (a_dev, b_dev) = other_gemm
        .upload_f32(&[0.0f32; 16], &[0.0f32; 16])
        .expect("upload_f32 must succeed for a well-formed 4x4x4 shape on the other context");
    let mut c_dev = other_gemm.alloc_output_f32(4, 4).expect(
        "alloc_output_f32 must succeed for a well-formed 4x4 output shape on the other context",
    );

    let err = gemm
        .launch_tiled_pipeline_f32(&func, &a_dev, &b_dev, &mut c_dev, 4, 4, 4)
        .expect_err(
            "launching with buffers allocated on a different CudaContext must be rejected \
             before reaching the unsafe launch, even when func's context matches",
        );
    assert!(matches!(
        err,
        CudaError::TiledPipelineContextMismatch { .. }
    ));
}

/// codex-review P1 指摘（PR #1071）の回帰テスト:
/// `launch_tiled_pipeline_f32`（常駐 API）は `run_tiled_pipeline_f32` と
/// 同じく m==0／n==0 を no-op として受理し、`unsafe` launch へ到達せず
/// `Ok(())` を返すことを確認する（`tiled_pipeline_zero_dim_shape_returns_
/// empty_without_launch` の常駐 API 版）。修正前は検証後に無条件で
/// grid_dim の一方が 0 の `LaunchConfig` を構築して driver launch へ
/// 進んでいたため、ゼロ次元形状で `CUDA_ERROR_INVALID_VALUE` 等になり
/// 得た（`launch_tiled_pipeline_f32` ドキュメンテーションコメント参照）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn launch_tiled_pipeline_zero_dim_shape_is_noop_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "tiled pipeline kernel must be available on this ignored test runner (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );
    let func = CudaGemm::compile_tiled_pipeline_variant(&device, 3)
        .expect("tiled pipeline variant compilation must succeed");

    // m==0: a_dev は空（m*k==0）、b_dev は k*n==16 要素、c_dev は空
    // （m*n==0）。
    let (a_dev, b_dev) = gemm
        .upload_f32(&[], &[0.0f32; 16])
        .expect("upload_f32 must succeed for a well-formed m==0 shape");
    let mut c_dev = gemm
        .alloc_output_f32(0, 4)
        .expect("alloc_output_f32 must succeed for a well-formed m==0 output shape");
    gemm.launch_tiled_pipeline_f32(&func, &a_dev, &b_dev, &mut c_dev, 0, 4, 4)
        .expect("m==0 must be treated as a no-op, not a driver launch error");

    // n==0: a_dev は m*k==16 要素、b_dev は空（k*n==0）、c_dev は空
    // （m*n==0）。
    let (a_dev, b_dev) = gemm
        .upload_f32(&[0.0f32; 16], &[])
        .expect("upload_f32 must succeed for a well-formed n==0 shape");
    let mut c_dev = gemm
        .alloc_output_f32(4, 0)
        .expect("alloc_output_f32 must succeed for a well-formed n==0 output shape");
    gemm.launch_tiled_pipeline_f32(&func, &a_dev, &b_dev, &mut c_dev, 4, 0, 4)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
}
