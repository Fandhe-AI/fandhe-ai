//! persistent タイルキュー版 pipeline カーネル（grid=SM 数・atomic タイル
//! 取得。イシュー #1346）の CPU-CUDA 数値一致・非 persistent 版との出力
//! bit 同一回帰テスト。
//!
//! `tests/cpu_cuda_tiled_pipeline_parity.rs` と同じ方針で、判定式・閾値は
//! `fandhe_ai_backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3
//! 未満または絶対誤差 1e-5 未満」の唯一の実体）に一本化し、ここでローカル
//! 複製しない（`.claude/rules/coding-rust.md`）。
//!
//! **本番経路との関係**: `CudaGemm::run_tiled_pipeline_persistent_f32` は
//! 本番既定経路（`run_tiled_f32`）を置き換えない選択可能な opt-in 変種
//! であり、本ファイルの全テストは明示的にこの API を呼ぶ
//! （`kernels_tiled_pipeline.rs::TP_KERNEL_PERSISTENT_PREFIX` 冒頭
//! コメント「位置づけ・非結線」参照）。
//!
//! **AC-2（受け入れ基準本体）**: `tiled_pipeline_persistent_matches_non_
//! persistent_bit_exact` が、非 persistent 版
//! （`CudaGemm::run_tiled_pipeline_f32`）と persistent 版
//! （`CudaGemm::run_tiled_pipeline_persistent_f32`）の出力が**全形状で
//! bit 同一**であることを `assert_eq!` で検証する（`kernels_tiled_
//! pipeline.rs::TP_KERNEL_PERSISTENT_PREFIX` ドキュメンテーションコメント
//! 「bit 同一の根拠」参照）。
//!
//! **実機依存の分離**: 環境適応スモークのみ通常 CI で実行、CUDA/NVRTC
//! 非搭載環境・cp.async 非対応（sm_80 未満）環境では早期 return で green
//! （`tests/cpu_cuda_tiled_pipeline_parity.rs` と同じ分岐パターン）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// タイル倍数形状・4 の倍数の非タイル倍数エッジ形状（cp.async 16 バイト
/// 整列条件は満たすがブロックタイル・K タイル非倍数）・タイル数が grid
/// （SM 数）を下回る／上回る双方の形状を網羅する（イシュー #1346 実装
/// 計画 §C「形状群」）。
fn persistent_bit_exact_shapes() -> Vec<(u32, u32, u32)> {
    vec![
        // タイル数 < grid（SM 数）想定の小形状。
        (64, 64, 64),
        (128, 64, 64),
        // タイル数 ≫ grid 想定の大形状（K 支配的形状を含む）。
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (256, 256, 4096),
        // 端数タイル（ブロックタイル・K タイル非倍数。n/k は 4 の倍数を
        // 維持しつつ m/n/k いずれも TP_BM(64)/TP_BN(64)/TP_BK(16) の倍数
        // から外す）。
        (60, 68, 36),
        (544, 256, 2048),
        (4100, 1028, 64),
        // m または n がちょうどタイル境界。
        (128, 200, 64),
        (200, 128, 64),
    ]
}

/// [`persistent_bit_exact_shapes`] の全形状で `run_tiled_pipeline_f32`
/// （非 persistent 版）と `run_tiled_pipeline_persistent_f32`（persistent
/// 版）の出力が bit 同一であること、および CPU 参照実装との複合判定を
/// 検証する（AC-2 本体）。`blocks_per_sm` は `Some(1)`（grid=SM 数。
/// イシュー #1346 の受け入れ条件が挙げる構成）と `None`（占有率実測既定）
/// の双方で検査する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_persistent_matches_non_persistent_bit_exact() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "non-persistent tiled pipeline kernel must be available on this ignored test runner \
         (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    for blocks_per_sm in [Some(1u32), None] {
        let mut persistent_func = CudaGemm::compile_tiled_pipeline_persistent_variant(
            &device,
            3,
            blocks_per_sm,
        )
        .unwrap_or_else(|e| {
            panic!(
                "compile_tiled_pipeline_persistent_variant(blocks_per_sm={blocks_per_sm:?}) \
                         must succeed on this ignored test runner: {e}"
            )
        });

        for (seed, (m, n, k)) in persistent_bit_exact_shapes().into_iter().enumerate() {
            let mut rng = bench_harness::rng::Xorshift64Star::new(seed as u64 + 1);
            let a = rng.fill_vec((m as usize) * (k as usize));
            let b = rng.fill_vec((k as usize) * (n as usize));

            let c_non_persistent =
                gemm.run_tiled_pipeline_f32(&a, &b, m, n, k)
                    .unwrap_or_else(|e| {
                        panic!("run_tiled_pipeline_f32 must succeed for m={m},n={n},k={k}: {e}")
                    });
            let c_persistent = gemm
                .run_tiled_pipeline_persistent_f32(&mut persistent_func, &a, &b, m, n, k)
                .unwrap_or_else(|e| {
                    panic!(
                        "run_tiled_pipeline_persistent_f32(blocks_per_sm={blocks_per_sm:?}) must \
                         succeed for m={m},n={n},k={k}: {e}"
                    )
                });

            assert_eq!(
                c_persistent, c_non_persistent,
                "persistent 版と非 persistent 版の出力が bit 同一ではありません \
                 (blocks_per_sm={blocks_per_sm:?}, m={m}, n={n}, k={k})"
            );

            let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
            fandhe_ai_backend_cpu::matmul_reference_fma(
                &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
            )
            .expect("matmul_reference_fma shape validation must pass for well-formed test input");
            fandhe_ai_backend_cpu::assert_parity(
                &format!("persistent tiled pipeline m={m} n={n} k={k}"),
                &c_persistent,
                &c_ref,
            );
        }
    }
}

/// 同一入力に対する persistent 版の連続 2 回起動が同一出力を返す（atomic
/// タイル取得順が実行のたびに変わりうる可能性を許容しても、タイル内計算
/// が [`TP_TILE_CORE`] 共有のため出力自体は決定的に一致する契約）ことを
/// 検証する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_persistent_repeated_launch_is_deterministic() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_persistent_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_persistent_variant must succeed");

    let mut rng = bench_harness::rng::Xorshift64Star::new(7);
    let (m, n, k) = (512u32, 512u32, 512u32);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let first = gemm
        .run_tiled_pipeline_persistent_f32(&mut func, &a, &b, m, n, k)
        .expect("first run_tiled_pipeline_persistent_f32 call must succeed");
    let second = gemm
        .run_tiled_pipeline_persistent_f32(&mut func, &a, &b, m, n, k)
        .expect("second run_tiled_pipeline_persistent_f32 call must succeed");

    assert_eq!(
        first, second,
        "persistent 版の連続 2 回起動（同一ハンドル・同一入力）は同一出力になるはず"
    );
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tiled_pipeline_parity_smoke_env_adaptive` と同じ分岐パターン。
#[test]
fn tiled_pipeline_persistent_parity_smoke_env_adaptive() {
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
        // cp.async は sm_80 (Ampere) 以降限定。非 persistent 版と同じ
        // 早期 return（`gemm.rs::CudaGemm::new` ドキュメンテーションコメント
        // 参照）。
        return;
    }

    let mut func = match CudaGemm::compile_tiled_pipeline_persistent_variant(&device, 3, Some(1)) {
        Ok(func) => func,
        Err(err) => {
            // SM 数取得不能・占有率実測不能等の環境要因は早期 return
            // （fail-soft。非 persistent 版が使える環境でも persistent 版
            // 固有の要求〈SM 数照会〉が満たせない場合がありうる）。
            match err {
                CudaError::TiledPipelineUnavailable { detail } => {
                    assert!(!detail.is_empty());
                    return;
                }
                other => panic!(
                    "unexpected CudaError variant from compile_tiled_pipeline_persistent_variant: {other}"
                ),
            }
        }
    };

    let mut rng = bench_harness::rng::Xorshift64Star::new(1);
    let (m, n, k) = (64u32, 64u32, 64u32);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_non_persistent = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_tiled_pipeline_f32 must succeed on a cp.async-capable test runner");
    let c_persistent = gemm
        .run_tiled_pipeline_persistent_f32(&mut func, &a, &b, m, n, k)
        .expect(
            "CudaGemm::run_tiled_pipeline_persistent_f32 must succeed on a cp.async-capable \
             test runner",
        );
    assert_eq!(
        c_persistent, c_non_persistent,
        "smoke 64x64x64: persistent 版と非 persistent 版の出力が bit 同一ではありません"
    );
}

/// [`CudaGemm::compile_tiled_pipeline_persistent_variant`] の
/// `blocks_per_sm = Some(0)` 拒否（`CudaError::InvalidKernelConfig`）を
/// 検証する（GPU が使える環境であれば `CudaDevice::new`/`CudaGemm::new`
/// 自体は成功する前提のためスモークとして非 `#[ignore]` にはしない。
/// SM 数照会・カーネルコンパイルを伴うため実機 `#[ignore]` に分類する）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn compile_tiled_pipeline_persistent_variant_rejects_zero_blocks_per_sm() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    // `PersistentTiledPipelineFunction` は `Debug` を実装しないため
    // （`CudaFunction`／`CudaSlice` が実装しない。他の `TiledPipelineFunction`
    // 系テストと同じ制約）、`expect_err` ではなく `match` で検査する。
    let err = match CudaGemm::compile_tiled_pipeline_persistent_variant(&device, 3, Some(0)) {
        Err(e) => e,
        Ok(_) => panic!("blocks_per_sm = Some(0) must be rejected before any kernel launch"),
    };
    assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
}

/// [`CudaGemm::launch_tiled_pipeline_persistent_f32`] が `func` の context
/// 不一致を fail-closed に拒否することを検証する
/// （`tiled_pipeline_rejects_mismatched_context_handle` の persistent 版）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_persistent_rejects_mismatched_context_handle() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");

    let other_device =
        CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let mut other_func =
        CudaGemm::compile_tiled_pipeline_persistent_variant(&other_device, 3, Some(1))
            .expect("compile_tiled_pipeline_persistent_variant must succeed on the other context");

    let (a_dev, b_dev) = gemm
        .upload_f32(&[0.0f32; 16], &[0.0f32; 16])
        .expect("upload_f32 must succeed for a well-formed 4x4x4 shape");
    let mut c_dev = gemm
        .alloc_output_f32(4, 4)
        .expect("alloc_output_f32 must succeed for a well-formed 4x4 output shape");

    let err = gemm
        .launch_tiled_pipeline_persistent_f32(&mut other_func, &a_dev, &b_dev, &mut c_dev, 4, 4, 4)
        .expect_err(
            "launching a persistent handle compiled against a different CudaContext must be \
             rejected before reaching the unsafe launch",
        );
    assert!(matches!(
        err,
        CudaError::TiledPipelineContextMismatch { .. }
    ));
}

/// [`CudaGemm::launch_tiled_pipeline_persistent_f32`] が `m == 0 || n == 0`
/// を no-op として受理し `unsafe` launch へ到達せず `Ok(())` を返す
/// ことを検証する（`launch_tiled_pipeline_zero_dim_shape_is_noop_without_
/// launch` の persistent 版）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn launch_tiled_pipeline_persistent_zero_dim_shape_is_noop_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_persistent_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_persistent_variant must succeed");

    let (a_dev, b_dev) = gemm
        .upload_f32(&[], &[])
        .expect("upload_f32 must succeed for an empty shape");
    let mut c_dev = gemm
        .alloc_output_f32(0, 0)
        .expect("alloc_output_f32 must succeed for a 0x0 output shape");

    gemm.launch_tiled_pipeline_persistent_f32(&mut func, &a_dev, &b_dev, &mut c_dev, 0, 4, 4)
        .expect("m == 0 must be treated as a no-op and return Ok(())");
    gemm.launch_tiled_pipeline_persistent_f32(&mut func, &a_dev, &b_dev, &mut c_dev, 4, 0, 4)
        .expect("n == 0 must be treated as a no-op and return Ok(())");
}

/// 非整列形状（`n % 4 != 0` または `k % 4 != 0`）が
/// `run_tiled_pipeline_persistent_f32` から `CudaError::InvalidShape` で
/// fail-closed に拒否されることを検証する（非 persistent 版
/// `tiled_pipeline_rejects_misaligned_shape` と同じ契約）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_persistent_rejects_misaligned_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_persistent_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_persistent_variant must succeed");

    let a = vec![0.0f32; 64 * 65];
    let b = vec![0.0f32; 65 * 66];
    let err = gemm
        .run_tiled_pipeline_persistent_f32(&mut func, &a, &b, 64, 66, 65)
        .expect_err("n%4==0 かつ k%4==0 を満たさない形状は InvalidShape で拒否されるはず");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}

/// `k == 0` の場合、`run_tiled_pipeline_persistent_f32` がカーネル起動を
/// 回避し `m*n` 要素の全 0 ベクタを返すことを検証する（非 persistent 版
/// `tiled_pipeline_zero_k_returns_all_zero` と同じ契約。
/// `run_f32_kernel`／`launch_tiled_pipeline_persistent_f32` は `k == 0` を
/// 実際に起動する契約〈両版とも全 0 を書く〉が、`run_*` ホスト API は
/// `run_tiled_pipeline_f32` と同じ早期 return 契約を持つ）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn tiled_pipeline_persistent_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_persistent_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_persistent_variant must succeed");

    let c = gemm
        .run_tiled_pipeline_persistent_f32(&mut func, &[], &[], 4, 4, 0)
        .expect("k == 0 must succeed and return an all-zero vector without launching the kernel");
    assert_eq!(c, vec![0.0f32; 16]);
}
