//! f16 WMMA GEMM（`CudaWmmaGemm`）の API 健全性テスト（TASK-11.1b・#61）。
//!
//! `tests/gemm_naive.rs`（#33）と同じ設計方針: CUDA 搭載・非搭載どちらの
//! 環境でも green になる（初期化・エラー型の契約確認）。数値一致回帰
//! テストは `tests/cpu_cuda_wmma_parity.rs` に分離する。カーネルソース
//! 内の tensor core 命令実在検査・タイル定数整合検査は `kernels_wmma.rs`
//! 内部の `#[cfg(test)]`（クレート内部限定の `pub(crate)`/private 定数を
//! 参照する必要があるため。統合テストからは到達できない）で行う。

use backend_cuda::{CudaDevice, CudaError, CudaWmmaGemm};

/// `CudaWmmaGemm::new` は CUDA 非搭載環境で panic せず型付きエラーを返す。
/// CUDA 搭載・cc>=7.0 環境では WMMA f16 カーネルのコンパイルが成功する
/// ことを検証する。
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

    match CudaWmmaGemm::new(&device) {
        Ok(_gemm) => {
            // CUDA + cc>=7.0 環境: WMMA f16 カーネルのコンパイルが成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            // libcuda はあるが libnvrtc が dlopen できない環境（CUDA driver
            // のみインストール済みで toolkit 非搭載のケース）。panic しない
            // ことが本テストの主張であり、このケースも許容する。
            assert!(!detail.is_empty());
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            // cc < 7.0 の実機（WMMA 非対応）。panic せず型付きエラーで
            // 拒否されることが受け入れ条件（TensorCoreUnsupported の主役）。
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected CudaError variant from CudaWmmaGemm::new: {other}"),
    }
}

/// `CudaError::TensorCoreUnsupported` の `Display` が非空でメッセージを
/// 含むこと（実機非依存）。`InvalidShape` 同様、公開 API から到達できる
/// 範囲でエラー型の契約を確認する。
#[test]
fn tensor_core_unsupported_display_is_non_empty_and_contains_detail() {
    use std::error::Error;

    let err = CudaError::TensorCoreUnsupported {
        detail: "WMMA requires compute capability >= 7.0, but device reports 6.1".to_string(),
    };
    let msg = err.to_string();
    assert!(!msg.is_empty());
    assert!(msg.contains("compute capability"));
    assert!(err.source().is_none());
}

/// m==0／n==0（`CudaGemm::run_naive_f32` と同じ no-op 形状）で `run_f16`
/// を呼んでも CUDA 起動そのものが発生せず、空の結果を返すことを実機で
/// 確認する（`gemm_naive.rs::naive_f32_zero_dim_shape_returns_empty_without_launch`
/// と同じ根拠。0 次元 grid の起動は CUDA ドライバに拒否される）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[half::f16::ONE; 4], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_f16(&[half::f16::ONE; 2], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}

/// k==0 は A/B が空スライスになるため、カーネル起動を回避し C = 全 0 を
/// 返す契約（`gemm.rs::run_f32_kernel` の k==0 早期 return と同じ根拠）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let c = gemm
        .run_f16(&[], &[], 2, 3, 0)
        .expect("k==0 must be treated as a no-op returning all-zero C");
    assert_eq!(c, vec![half::f16::ZERO; 6]);
}
