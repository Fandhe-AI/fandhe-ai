//! 最終 wave 限定 Stream-K（固定順序 fixup。イシュー #1358）版 pipeline
//! カーネルの決定性・full タイル領域の bit 同一・CPU 参照実装との複合
//! 判定統計を検証する回帰テスト。
//!
//! `tests/cpu_cuda_tiled_pipeline_persistent_parity.rs` と同じ方針で、
//! 判定式・閾値は `fandhe_ai_backend_cpu::assert_parity`（統一複合判定
//! 「相対誤差 1e-3 未満または絶対誤差 1e-5 未満」の唯一の実体）を使う。
//!
//! **AC 本体（イシュー #1358 の受け入れ条件）**:
//! [`streamk_repeated_launch_is_deterministic`] が、同一入力・同一ハンドル
//! および別ハンドルへの複数回起動で出力が bit 同一であることを
//! `assert_eq!` で検証する（`kernels_tiled_pipeline.rs::TP_SK_FIXUP_KERNEL`
//! ドキュメンテーションコメント「決定性の根拠」参照）。
//!
//! **残タイルの数値一致は非 Stream-K 版と bit 同一ではない**（K 連鎖の
//! 分割による丸め差。設計上の既知差分。合否判定・baseline 追加の要否は
//! 兄弟イシュー #1359 が担う。本ファイルは統計出力のみ行い
//! `assert_parity`（厳密ゼロ fail 判定）は残タイルへは適用しない）。
//!
//! **実機依存の分離**: 環境適応スモークのみ通常 CI で実行、CUDA/NVRTC
//! 非搭載環境・cp.async 非対応（sm_80 未満）環境では早期 return で green
//! （`tests/cpu_cuda_tiled_pipeline_parity.rs` と同じ分岐パターン）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// Stream-K の分割が実際に発生する（`remainder_tiles > 0`）ことを狙う
/// 形状群（`persistent_bit_exact_shapes`〈`cpu_cuda_tiled_pipeline_
/// persistent_parity.rs`〉と同じ動機の形状網羅に加え、タイル数が SM 数
/// の倍数から外れる値を優先する）。整列制約（`n % 4 == 0 && k % 4 ==
/// 0`）を満たす。
fn streamk_shapes() -> Vec<(u32, u32, u32)> {
    vec![
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (256, 256, 4096),
        (448, 448, 1024),
        (60, 68, 36),
        (544, 256, 2048),
        (4100, 1028, 64),
    ]
}

/// [`streamk_shapes`] の全形状で、full タイル領域（Stream-K 版が計画
/// する `full_tiles` に属するタイル）が非 Stream-K 版
/// （`run_tiled_pipeline_f32`）と bit 同一であることを検証する
/// （`kernels_tiled_pipeline.rs::TP_SK_TILE_CORE` ドキュメンテーション
/// コメントの決定性論証点 1 の実機裏付け）。full タイルはタイル走査順
/// （行主導）の先頭 `full_tiles` 個であるため、出力配列の先頭から
/// `full_tiles` タイル分の行を比較する（`n` がタイル幅の倍数でない
/// 端数形状は最終行タイルが列方向に跨るため、要素単位ではなく行単位
/// （`TP_BM` 行区切り）で比較する）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn streamk_full_tiles_match_non_persistent_bit_exact() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "non-persistent tiled pipeline kernel must be available on this ignored test runner \
         (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    for blocks_per_sm in [Some(1u32), None] {
        let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, blocks_per_sm)
            .unwrap_or_else(|e| {
                panic!(
                    "compile_tiled_pipeline_streamk_variant(blocks_per_sm={blocks_per_sm:?}) \
                     must succeed on this ignored test runner: {e}"
                )
            });

        for (seed, (m, n, k)) in streamk_shapes().into_iter().enumerate() {
            let mut rng = bench_harness::rng::Xorshift64Star::new(seed as u64 + 1);
            let a = rng.fill_vec((m as usize) * (k as usize));
            let b = rng.fill_vec((k as usize) * (n as usize));

            let c_non_streamk = gemm
                .run_tiled_pipeline_f32(&a, &b, m, n, k)
                .unwrap_or_else(|e| {
                    panic!("run_tiled_pipeline_f32 must succeed for m={m},n={n},k={k}: {e}")
                });
            let (c_streamk, plan) = gemm
                .run_tiled_pipeline_streamk_f32(&mut func, &a, &b, m, n, k)
                .unwrap_or_else(|e| {
                    panic!(
                        "run_tiled_pipeline_streamk_f32(blocks_per_sm={blocks_per_sm:?}) must \
                         succeed for m={m},n={n},k={k}: {e}"
                    )
                });

            assert_eq!(
                c_streamk.len(),
                c_non_streamk.len(),
                "出力長が一致しません (m={m}, n={n}, k={k})"
            );

            // full タイル領域（先頭 full_tiles*TP_BM 行。TP_BM=64 は
            // internal-diagnostics 限定の公開定数がないため、64x64 版
            // 固定という契約〈実装計画 §2 対象外「128x64 は対象外」〉
            // から直接埋め込む。行末が m 未満に切り詰められる可能性が
            // あるため min で clamp する）。
            const TP_BM: usize = 64;
            let full_rows = ((plan.full_tiles as usize) * TP_BM).min(m as usize);
            let full_elems = full_rows * (n as usize);
            assert_eq!(
                c_streamk[..full_elems],
                c_non_streamk[..full_elems],
                "full タイル領域が非 Stream-K 版と bit 同一ではありません \
                 (blocks_per_sm={blocks_per_sm:?}, m={m}, n={n}, k={k}, \
                 full_tiles={}, remainder_tiles={})",
                plan.full_tiles,
                plan.remainder_tiles,
            );

            // 残タイル領域は CPU 参照実装との複合判定統計のみ出力する
            // （合否判定は行わない。ファイル冒頭コメント参照）。
            if plan.is_active() {
                let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
                fandhe_ai_backend_cpu::matmul_reference_fma(
                    &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
                )
                .expect(
                    "matmul_reference_fma shape validation must pass for well-formed test input",
                );
                let report =
                    fandhe_ai_backend_cpu::compare(&c_streamk[full_elems..], &c_ref[full_elems..])
                        .expect("compare must succeed for equal-length well-formed slices");
                eprintln!(
                    "streamk remainder-tile composite check (informational, not a pass/fail \
                     gate; see #1359): m={m} n={n} k={k} blocks_per_sm={blocks_per_sm:?} \
                     remainder_tiles={} q={} max_contributors={} fail_count={} total={} \
                     max_abs_diff={} max_rel_err={}",
                    plan.remainder_tiles,
                    plan.q,
                    plan.max_contributors,
                    report.fail_count,
                    report.total,
                    report.max_abs_diff,
                    report.max_rel_err,
                );
            }
        }
    }
}

/// 同一ハンドル・同一入力での複数回起動、および別ハンドルでの起動が
/// いずれも同一出力を返す（決定性契約。イシュー #1358 の受け入れ条件
/// 本体）ことを検証する。分割が実際に発生する形状（`remainder_tiles >
/// 0`）を precondition として少なくとも 1 形状で成立することを assert
/// する（実装計画 §3.4 点 1）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn streamk_repeated_launch_is_deterministic() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");

    let mut any_active = false;
    for blocks_per_sm in [Some(1u32), None] {
        let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, blocks_per_sm)
            .unwrap_or_else(|e| {
                panic!(
                    "compile_tiled_pipeline_streamk_variant(blocks_per_sm={blocks_per_sm:?}) \
                     must succeed on this ignored test runner: {e}"
                )
            });

        for (seed, (m, n, k)) in streamk_shapes().into_iter().enumerate() {
            let mut rng = bench_harness::rng::Xorshift64Star::new(seed as u64 + 1001);
            let a = rng.fill_vec((m as usize) * (k as usize));
            let b = rng.fill_vec((k as usize) * (n as usize));

            let (first, plan) = gemm
                .run_tiled_pipeline_streamk_f32(&mut func, &a, &b, m, n, k)
                .unwrap_or_else(|e| {
                    panic!("first run_tiled_pipeline_streamk_f32 failed for m={m},n={n},k={k}: {e}")
                });
            if plan.is_active() {
                any_active = true;
            }
            let (second, _) = gemm
                .run_tiled_pipeline_streamk_f32(&mut func, &a, &b, m, n, k)
                .unwrap_or_else(|e| {
                    panic!(
                        "second run_tiled_pipeline_streamk_f32 failed for m={m},n={n},k={k}: {e}"
                    )
                });
            assert_eq!(
                first, second,
                "同一ハンドル・同一入力の連続 2 回起動は同一出力になるはず \
                 (blocks_per_sm={blocks_per_sm:?}, m={m}, n={n}, k={k})"
            );

            // 別ハンドルでの起動も同一出力になることを確認する。
            let mut other_func =
                CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, blocks_per_sm)
                    .unwrap_or_else(|e| {
                        panic!(
                            "compile_tiled_pipeline_streamk_variant (second handle) must \
                             succeed: {e}"
                        )
                    });
            let (third, _) = gemm
                .run_tiled_pipeline_streamk_f32(&mut other_func, &a, &b, m, n, k)
                .unwrap_or_else(|e| {
                    panic!(
                        "run_tiled_pipeline_streamk_f32 (other handle) failed for \
                         m={m},n={n},k={k}: {e}"
                    )
                });
            assert_eq!(
                first, third,
                "別ハンドルでの起動は同一出力になるはず (blocks_per_sm={blocks_per_sm:?}, \
                 m={m}, n={n}, k={k})"
            );
        }
    }

    assert!(
        any_active,
        "streamk_shapes() のいずれの形状・blocks_per_sm 構成でも Stream-K 分割 \
         （remainder_tiles > 0）が発生しませんでした。決定性契約の検証対象が \
         K 分割を経ないままでは受け入れ条件〈AC 本体〉を検証したことになりません"
    );
}

/// `R == 0` または `Q >= nk` になる形状（`streamk_plan` が非活性と判定
/// する構成）で、Stream-K 版が persistent 版・非 persistent 版と全体
/// bit 同一であることを検証する（実装計画 §3.4 点 3）。`num_sms *
/// blocks_per_sm` が出力タイル総数を割り切る形状を `blocks_per_sm =
/// Some(1)` で構成する（`persistent_grid_blocks`/`streamk_plan` の
/// `T mod G == 0` 非活性条件に依らず、`num_sms` 自体が偶数であれば
/// `T` をタイル数 `num_sms` に一致させることで確実に非活性化できる）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn streamk_inactive_matches_non_streamk_bit_exact() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_streamk_variant must succeed");
    let grid_capacity = func.grid_capacity();
    assert!(grid_capacity > 0);

    // T == grid_capacity（blocks_per_sm=1 なので = num_sms）になるよう
    // 正方形状 m=n=grid_capacity*TP_BM（TP_BM=64 固定。実装計画 §2）を
    // 選ぶ。T = ceil(m/64)*ceil(n/64) = grid_capacity になり
    // T mod G == 0（実際は T == G）で非活性となる。
    const TP_BM: u32 = 64;
    let side = grid_capacity * TP_BM;
    let (m, n, k) = (side, side, 256u32);

    let mut rng = bench_harness::rng::Xorshift64Star::new(31);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_non_streamk = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("run_tiled_pipeline_f32 must succeed on a cp.async-capable test runner");
    let (c_streamk, plan) = gemm
        .run_tiled_pipeline_streamk_f32(&mut func, &a, &b, m, n, k)
        .expect("run_tiled_pipeline_streamk_f32 must succeed on a cp.async-capable test runner");

    assert!(
        !plan.is_active(),
        "この形状は非活性（T mod G == 0）になる想定でしたが activated: {plan:?}"
    );
    assert_eq!(
        c_streamk, c_non_streamk,
        "非活性形状で Stream-K 版と非 Stream-K 版が bit 同一ではありません \
         (m={m}, n={n}, k={k})"
    );
}

/// [`CudaGemm::compile_tiled_pipeline_streamk_variant`] の `blocks_per_sm
/// = Some(0)` 拒否（`CudaError::InvalidKernelConfig`）を検証する
/// （`compile_tiled_pipeline_persistent_variant_rejects_zero_blocks_per_sm`
/// と同型）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn compile_tiled_pipeline_streamk_variant_rejects_zero_blocks_per_sm() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let err = match CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, Some(0)) {
        Err(e) => e,
        Ok(_) => panic!("blocks_per_sm = Some(0) must be rejected before any kernel launch"),
    };
    assert!(matches!(err, CudaError::InvalidKernelConfig { .. }));
}

/// [`CudaGemm::launch_tiled_pipeline_streamk_f32`] が `func` の context
/// 不一致を fail-closed に拒否することを検証する
/// （`tiled_pipeline_persistent_rejects_mismatched_context_handle` の
/// Stream-K 版）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn streamk_rejects_mismatched_context_handle() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");

    let other_device =
        CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let mut other_func =
        CudaGemm::compile_tiled_pipeline_streamk_variant(&other_device, 3, Some(1))
            .expect("compile_tiled_pipeline_streamk_variant must succeed on the other context");

    let (a_dev, b_dev) = gemm
        .upload_f32(&[0.0f32; 16], &[0.0f32; 16])
        .expect("upload_f32 must succeed for a well-formed 4x4x4 shape");
    let mut c_dev = gemm
        .alloc_output_f32(4, 4)
        .expect("alloc_output_f32 must succeed for a well-formed 4x4 output shape");

    let err = gemm
        .launch_tiled_pipeline_streamk_f32(&mut other_func, &a_dev, &b_dev, &mut c_dev, 4, 4, 4)
        .expect_err(
            "launching a streamk handle compiled against a different CudaContext must be \
             rejected before reaching the unsafe launch",
        );
    assert!(matches!(
        err,
        CudaError::TiledPipelineContextMismatch { .. }
    ));
}

/// `m == 0 || n == 0` が no-op（`unsafe` launch へ到達せず非活性 plan を
/// 返す）ことを検証する（`launch_tiled_pipeline_persistent_zero_dim_
/// shape_is_noop_without_launch` の Stream-K 版）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn launch_tiled_pipeline_streamk_zero_dim_shape_is_noop_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_streamk_variant must succeed");

    let (a_dev, b_dev) = gemm
        .upload_f32(&[], &[0.0f32; 16])
        .expect("upload_f32 must succeed for a well-formed m==0 shape");
    let mut c_dev = gemm
        .alloc_output_f32(0, 4)
        .expect("alloc_output_f32 must succeed for a well-formed m==0 output shape");
    let plan = gemm
        .launch_tiled_pipeline_streamk_f32(&mut func, &a_dev, &b_dev, &mut c_dev, 0, 4, 4)
        .expect("m == 0 must be treated as a no-op and return Ok(plan)");
    assert!(!plan.is_active());

    let (a_dev, b_dev) = gemm
        .upload_f32(&[0.0f32; 16], &[])
        .expect("upload_f32 must succeed for a well-formed n==0 shape");
    let mut c_dev = gemm
        .alloc_output_f32(4, 0)
        .expect("alloc_output_f32 must succeed for a well-formed n==0 output shape");
    let plan = gemm
        .launch_tiled_pipeline_streamk_f32(&mut func, &a_dev, &b_dev, &mut c_dev, 4, 0, 4)
        .expect("n == 0 must be treated as a no-op and return Ok(plan)");
    assert!(!plan.is_active());
}

/// 非整列形状（`n % 4 != 0` または `k % 4 != 0`）が
/// `run_tiled_pipeline_streamk_f32` から `CudaError::InvalidShape` で
/// fail-closed に拒否されることを検証する（`tiled_pipeline_persistent_
/// rejects_misaligned_shape` と同じ契約）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn streamk_rejects_misaligned_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_streamk_variant must succeed");

    let a = vec![0.0f32; 64 * 65];
    let b = vec![0.0f32; 65 * 66];
    let err = gemm
        .run_tiled_pipeline_streamk_f32(&mut func, &a, &b, 64, 66, 65)
        .expect_err("n%4==0 かつ k%4==0 を満たさない形状は InvalidShape で拒否されるはず");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}

/// `k == 0` の場合、`run_tiled_pipeline_streamk_f32` がカーネル起動を
/// 回避し `m*n` 要素の全 0 ベクタを返すことを検証する
/// （`tiled_pipeline_persistent_zero_k_returns_all_zero` と同じ契約）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以降、cp.async 対応）必須"]
fn streamk_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let mut func = CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, Some(1))
        .expect("compile_tiled_pipeline_streamk_variant must succeed");

    let (c, plan) = gemm
        .run_tiled_pipeline_streamk_f32(&mut func, &[], &[], 4, 4, 0)
        .expect("k == 0 must succeed and return an all-zero vector without launching the kernel");
    assert_eq!(c, vec![0.0f32; 16]);
    assert!(!plan.is_active());
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tiled_pipeline_persistent_parity_smoke_env_adaptive` と同じ分岐
/// パターン。
#[test]
fn tiled_pipeline_streamk_parity_smoke_env_adaptive() {
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
        // 早期 return。
        return;
    }

    let mut func = match CudaGemm::compile_tiled_pipeline_streamk_variant(&device, 3, Some(1)) {
        Ok(func) => func,
        Err(err) => match err {
            CudaError::TiledPipelineUnavailable { detail } => {
                assert!(!detail.is_empty());
                return;
            }
            other => panic!(
                "unexpected CudaError variant from compile_tiled_pipeline_streamk_variant: \
                 {other}"
            ),
        },
    };

    let mut rng = bench_harness::rng::Xorshift64Star::new(1);
    let (m, n, k) = (64u32, 64u32, 64u32);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_non_streamk = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_tiled_pipeline_f32 must succeed on a cp.async-capable test runner");
    let (c_streamk, plan) = gemm
        .run_tiled_pipeline_streamk_f32(&mut func, &a, &b, m, n, k)
        .expect(
            "CudaGemm::run_tiled_pipeline_streamk_f32 must succeed on a cp.async-capable test \
             runner",
        );
    // 64x64x64 は単一タイル（T=1）のため常に非活性（full タイルのみ）。
    assert!(!plan.is_active());
    assert_eq!(
        c_streamk, c_non_streamk,
        "smoke 64x64x64: Stream-K 版と非 Stream-K 版の出力が bit 同一ではありません"
    );
}
