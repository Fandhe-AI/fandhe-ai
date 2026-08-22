//! `MetalDeviceProvider`（TASK-1.9a・#44）のテスト。`cfg(target_os = "macos")`
//! 限定（`Device::Metal`・`objc2` 系依存自体が macOS 限定のため。
//! `.claude/rules/deps-policy.md`）。実機（Metal 対応 Mac）依存の検証は
//! `#[ignore]` で分離する（`.claude/rules/coding-rust.md`）。

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalDeviceProvider;
use fandhe_ai_tensor_core::device::DeviceProvider;

#[test]
fn backend_name_is_metal() {
    let provider = MetalDeviceProvider::new();
    assert_eq!(provider.backend_name(), "metal");
}

/// macOS ランナー上でも Metal 非対応構成（ヘッドレス CI 等）はありうる
/// ため、`enumerate` が `panic!` せず `Ok` を返すことのみを通常 CI で検証
/// する（Metal デバイス 0 件を許容する）。
#[test]
fn enumerate_never_panics() {
    let provider = MetalDeviceProvider::new();

    let devices = provider.enumerate().expect("enumerate must not error");

    if provider.is_available() {
        assert!(!devices.is_empty());
    } else {
        assert!(devices.is_empty());
    }
}

/// 実機（Metal 対応 Mac）依存の検証。デバイスが実際に 1 件以上検出され、
/// 名前が非空であることを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn select_metal_device_on_real_hardware() {
    use fandhe_ai_tensor_core::device::Device;

    let provider = MetalDeviceProvider::new();

    let info = provider
        .select(Device::Metal)
        .expect("Metal device must be selectable on Metal-equipped hardware");

    assert_eq!(info.device, Device::Metal);
    assert!(!info.name.is_empty());
}
