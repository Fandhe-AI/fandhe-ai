//! f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM（`CudaMmaGemm`）の API 健全性
//! テスト（TASK-11.1h・#187）。
//!
//! `tests/gemm_wmma.rs`（#61）と同じ設計方針: CUDA 搭載・非搭載どちらの
//! 環境でも green になる（初期化・エラー型の契約確認）。数値一致回帰
//! テストは `tests/cpu_cuda_mma_parity.rs` に分離する。カーネルソース内の
//! tensor core 命令実在検査・タイル定数整合検査は `kernels_mma.rs` 内部の
//! `#[cfg(test)]`（クレート内部限定の `pub(crate)`/private 定数を参照する
//! 必要があるため。統合テストからは到達できない）で行う。
//!
//! 本実装セッションの実行環境は CUDA driver（`libcuda`。compute
//! capability 8.6・RTX 3060 実機）はあるが NVRTC（`libnvrtc`）が無い
//! （`kernels_mma.rs` 冒頭コメント「検証状態」参照）。よって
//! `CudaDevice::new` は成功し `CudaMmaGemm::new` は
//! `CudaError::NvrtcUnavailable` を返す分岐を必ず通る。この分岐を
//! `tests/gemm_wmma.rs` と同じ形で green として扱うことが本ファイルの
//! 前提である。

use backend_cuda::{CudaDevice, CudaError, CudaMmaGemm};

/// `CudaMmaGemm::new` は CUDA 非搭載環境で panic せず型付きエラーを返す。
/// CUDA 搭載・cc>=8.0 環境では `mma.sync`/`ldmatrix`/`cp.async` カーネルの
/// コンパイルが成功することを検証する。
#[test]
fn new_does_not_panic_and_returns_typed_result() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            // CUDA 非搭載環境（self-hosted CI 想定）: panic せず型付き
            // エラーが返ることそのものが受け入れ条件。
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            // libcuda は存在するが cuInit 等が失敗したケース。プローブは
            // 通過しているため panic しない前提は保たれる。
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaMmaGemm::new(&device) {
        Ok(_gemm) => {
            // CUDA + cc>=8.0 + NVRTC あり環境: mma カーネルのコンパイルが
            // 成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            // libcuda はあるが libnvrtc が dlopen できない環境（本実装
            // セッションの実行環境がまさにこの分岐。cc=8.6 の実機で
            // 確認済み。`kernels_mma.rs` 冒頭「検証状態」参照）。panic
            // しないことが本テストの主張であり、このケースを許容する。
            assert!(!detail.is_empty());
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            // cc < 8.0 の実機（cp.async/ldmatrix 非対応）。panic せず
            // 型付きエラーで拒否されることが受け入れ条件。
            assert!(!detail.is_empty());
            assert!(detail.contains("compute capability"));
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaGemm::new: {other}"),
    }
}

/// `MIN_COMPUTE_CAPABILITY_MAJOR`（8.0）が WMMA 経路（cc>=7.0）より厳しい
/// ことを、実機に依存せず `TensorCoreUnsupported` の `Display` 文言経由で
/// 確認する（`kernels_mma.rs` 冒頭コメント「命令選定・sm_80+ ゲート」参照）。
#[test]
fn tensor_core_unsupported_display_mentions_compute_capability_8() {
    use std::error::Error;

    let err = CudaError::TensorCoreUnsupported {
        detail: "mma.sync/ldmatrix/cp.async path requires compute capability >= 8.0, \
                  but device reports 7.5"
            .to_string(),
    };
    let msg = err.to_string();
    assert!(!msg.is_empty());
    assert!(msg.contains("compute capability"));
    assert!(err.source().is_none());
}

/// m==0／n==0（`CudaWmmaGemm::run_f16` と同じ no-op 形状）で `run_f16` を
/// 呼んでも CUDA 起動そのものが発生せず、空の結果を返すことを実機で
/// 確認する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[half::f16::ONE; 8], 0, 8, 8)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_f16(&[half::f16::ONE; 8], &[], 8, 0, 8)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}

/// k==0 は A/B が空スライスになるため、カーネル起動を回避し C = 全 0 を
/// 返す契約（`gemm_wmma.rs::run_f16` の k==0 早期 return と同じ根拠）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[], 8, 8, 0)
        .expect("k==0 must be treated as a no-op returning all-zero C");
    assert_eq!(c, vec![half::f16::ZERO; 64]);
}

/// 本経路固有の整列制約（`n`/`k` が 8 の倍数。`kernels_mma.rs` 冒頭
/// コメント「整列制約」）は実機非依存で検証できる（GPU 起動前の
/// ホスト側検証のため）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_rejects_non_multiple_of_eight_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let a = vec![half::f16::ONE; 8 * 9];
    let b = vec![half::f16::ONE; 9 * 8];
    let err = gemm
        .run_f16(&a, &b, 8, 8, 9)
        .expect_err("k=9 is not a multiple of 8 and must be rejected before GPU launch");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}

/// PR #255 レビュー指摘の回帰防止: `(m,n,k)=(8,7,0)` は `n=7` が
/// `validate_mma_alignment` の整列制約（8 の倍数）を満たさないが、
/// `k==0` の no-op 形状であるため実際にはカーネルを起動せず、整列検証
/// より前に早期 return で `Ok` を返すべき契約（`gemm_mma.rs::run_f16`
/// ドキュメンテーションコメント「ホスト側形状検証を 3 段で行う」参照）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_accepts_noop_shape_with_misaligned_n_when_k_is_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[], 8, 7, 0)
        .expect("k==0 no-op shape with misaligned n must not be rejected by alignment checks");
    assert_eq!(c, vec![half::f16::ZERO; 8 * 7]);
}

/// PR #255 レビュー指摘の回帰防止: `m` が大きすぎて
/// `mma_launch_config` の grid_dim.y（`m.div_ceil(MMA_BM)`）が CUDA の
/// 65,535 上限を超える形状は、形状・整列検証は満たしていても
/// `CudaError::InvalidShape` で明示的に拒否されるべき契約
/// （`gemm_mma.rs::validate_mma_grid_bounds` 参照。この検証がないと
/// ドライバのカーネル起動時エラーとして現れ、原因の切り分けが難しい）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_rejects_m_exceeding_grid_dim_y_limit() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    // MMA_BM=32: 65_535 * 32 + 32 = 2_097_152 で div_ceil(m, 32) = 65_536
    // > 65_535。a/b は形状検証にのみ使われ実際には確保しない大きさなので
    // 空スライスにせず最小限の妥当な長さで構築する必要はない
    // （validate_gemm_dims が先に走るため長さは合わせる）。
    let m: u32 = 65_535 * 32 + 32;
    let n: u32 = 8;
    let k: u32 = 8;
    let a = vec![half::f16::ONE; (m as usize) * (k as usize)];
    let b = vec![half::f16::ONE; (k as usize) * (n as usize)];
    let err = gemm
        .run_f16(&a, &b, m, n, k)
        .expect_err("m exceeding grid_dim.y's 65,535 limit must be rejected before GPU launch");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}
