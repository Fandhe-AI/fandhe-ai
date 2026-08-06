//! `CudaDevice`／`CudaError` の環境適応型テスト。
//!
//! CUDA 搭載・非搭載どちらの環境でも green になる設計とする。CI
//! self-hosted runner は CUDA toolkit 非搭載のため、本テストの
//! `Err(DriverUnavailable)` 分岐が実際に検証される（#35 の前提）。
//! 実機（DGX Spark GB10）でのみ意味を持つ肯定的検証（デバイス名の
//! 実値確認等）は #36 のスコープであり、本テストには含めない
//! （重複防止。`.claude/rules/coding-rust.md` の実機依存分離方針）。

use backend_cuda::{CudaDevice, CudaError};

/// 受け入れ条件そのもの: `CudaDevice::new` は CUDA 非搭載環境で panic
/// せず型付きエラーを返す。CUDA 搭載環境ではメタデータが取得できる
/// ことを検証する。どちらの分岐も panic しないことが本テストの主張。
#[test]
fn new_does_not_panic_and_returns_typed_result() {
    match CudaDevice::new(0) {
        Ok(dev) => {
            // CUDA 搭載環境: メタデータが妥当な形式であることを確認する。
            assert!(!dev.name().is_empty(), "device name must not be empty");
            let (major, minor) = dev.compute_capability();
            assert!(major > 0, "compute capability major must be positive");
            assert!(minor >= 0, "compute capability minor must be non-negative");
            assert_eq!(
                dev.arch(),
                format!("compute_{major}{minor}"),
                "arch must be NVRTC --gpu-architecture compatible compute_XY form"
            );
            assert_eq!(dev.ordinal(), 0);
        }
        Err(e) => {
            // CUDA 非搭載環境（self-hosted CI 想定）: panic せず
            // 型付きエラーが返ることそのものが受け入れ条件。
            match e {
                CudaError::DriverUnavailable { detail } => {
                    assert!(!detail.is_empty(), "detail message must not be empty");
                }
                CudaError::Driver(_) => {
                    // libcuda は存在するが cuInit 等が失敗したケース
                    // （ドライババージョン不一致等）。プローブは
                    // 通過しているため panic しない前提は保たれる。
                }
                other => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
            }
        }
    }
}

/// プローブ（`is_available`）と `new` の判定が整合していることを確認する。
/// `is_available() == false` の場合、`new(0)` は必ず `DriverUnavailable`
/// を返す（panic 回避ゲートが正しく先行していることの検証）。
#[test]
fn is_available_false_implies_new_returns_driver_unavailable() {
    if CudaDevice::is_available() {
        // CUDA 搭載環境ではこの分岐の対象外（別の正系テストで検証済み）。
        return;
    }

    match CudaDevice::new(0) {
        Err(CudaError::DriverUnavailable { .. }) => {}
        Err(other) => {
            panic!("expected DriverUnavailable when is_available() is false, got: {other}")
        }
        Ok(_) => panic!("new(0) succeeded despite is_available() returning false"),
    }
}

/// `device_count` も `new` と同じプローブゲートを経由することを確認する。
#[test]
fn device_count_does_not_panic() {
    match CudaDevice::device_count() {
        Ok(count) => {
            // CUDA 搭載環境: 台数は非負（usize なので型で保証済み）。
            let _ = count;
        }
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(CudaError::Driver(_)) => {
            // libcuda はロードできるが cuInit 等が失敗したケース（ドライバ
            // バージョン不一致・GPU 非搭載コンテナ等）。`new` 側の同種分岐
            // （L36-40）と同じ理由で正当なエラーとして許容する。
        }
        Err(other) => panic!("unexpected CudaError variant from device_count: {other}"),
    }
}

/// エラー型の単体テスト（環境非依存）: `Display` 表示・`From` 変換・
/// `source()` を検証する。
mod error_type {
    use backend_cuda::CudaError;
    use std::error::Error;

    #[test]
    fn driver_unavailable_display_is_non_empty_and_contains_detail() {
        let err = CudaError::DriverUnavailable {
            detail: "libcuda not found".to_string(),
        };
        let msg = err.to_string();
        assert!(!msg.is_empty());
        assert!(msg.contains("libcuda not found"));
        // 存在プローブの失敗は driver API 呼び出し由来のエラーではない
        // ため、source() は None を返す契約とする（error.rs 参照）。
        assert!(err.source().is_none());
    }

    #[test]
    fn nvrtc_unavailable_display_is_non_empty_and_contains_detail() {
        let err = CudaError::NvrtcUnavailable {
            detail: "libnvrtc not found".to_string(),
        };
        let msg = err.to_string();
        assert!(!msg.is_empty());
        assert!(msg.contains("libnvrtc not found"));
        assert!(err.source().is_none());
    }
}
