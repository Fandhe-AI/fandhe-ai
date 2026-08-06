//! `CudaDeviceProvider`（TASK-1.9a・#44）の非実機テスト。
//!
//! CI（self-hosted、CUDA toolkit 非搭載環境を含む）でも `panic!` せず
//! `Ok` を返すことを検証する（REQ-1「toolkit 非搭載環境でもビルド・実行
//! 成立」の実行時側の受け皿。`.claude/rules/ci.md`）。実機（DGX Spark
//! GB10）前提の検証（device 0 選択・デバイス名非空）は `#[ignore]` で
//! 分離する（`.claude/rules/coding-rust.md`）。

use backend_cuda::CudaDeviceProvider;
use tensor_core::device::{Device, DeviceProvider};

#[test]
fn enumerate_never_panics_regardless_of_driver_presence() {
    let provider = CudaDeviceProvider::new();

    // ドライバ不在なら Ok(vec![])、搭載環境なら Ok(非空) のいずれかであり、
    // どちらの場合も Err にならず panic もしない。
    let devices = provider.enumerate().expect("enumerate must not error");

    if provider.is_available() {
        assert!(!devices.is_empty());
    } else {
        assert!(devices.is_empty());
    }
}

#[test]
fn select_on_absent_ordinal_returns_typed_error_not_panic() {
    let provider = CudaDeviceProvider::new();

    // 存在しない可能性が高い大きな ordinal を選択させ、ドライバの有無に
    // かかわらず型付きエラーで返る（`unwrap`/`expect` させない）ことを
    // 確認する。
    let result = provider.select(Device::Cuda(9999));

    if provider.is_available() {
        // 実機環境でも ordinal 9999 が存在する可能性は極めて低いため
        // エラー経路を期待する。
        assert!(result.is_err());
    } else {
        assert!(result.is_err());
    }
}

#[test]
fn backend_name_is_cuda() {
    let provider = CudaDeviceProvider::new();
    assert_eq!(provider.backend_name(), "cuda");
}

/// 実機（DGX Spark GB10）依存の検証。CUDA ドライバが実際に device 0 を
/// 検出し、デバイス名が非空であることを確認する（`.claude/rules/coding-rust.md`
/// の実機依存テスト分離方針）。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）が必要。make test-ignored 相当で実行する"]
fn select_device_zero_on_real_hardware() {
    let provider = CudaDeviceProvider::new();

    let info = provider
        .select(Device::Cuda(0))
        .expect("device 0 must be selectable on CUDA-equipped hardware");

    assert_eq!(info.device, Device::Cuda(0));
    assert!(!info.name.is_empty());
}
