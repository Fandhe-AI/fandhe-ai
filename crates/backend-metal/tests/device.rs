//! `MetalDeviceProvider`（TASK-1.9a・#44）のテスト。`cfg(target_os = "macos")`
//! 限定（`Device::Metal`・`objc2` 系依存自体が macOS 限定のため。
//! `.claude/rules/deps-policy.md`）。実機（Metal 対応 Mac）依存の検証は
//! `#[ignore]` で分離する（`.claude/rules/coding-rust.md`）。

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalDeviceProvider;
use fandhe_ai_tensor_core::device::DeviceProvider;
use objc2_metal::{MTLCopyAllDevices, MTLDevice, MTLGPUFamily};

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
    // `probe_all` は Apple ファミリ（`supportsFamily(Apple1)`）と確認できた
    // GPU にのみ `Some(32)` を報告し、Intel／AMD GPU 搭載 Mac では `None` を
    // 返す契約のため、Apple GPU 確認時のみ `Some(32)` を要求する（#2125）。
    // `select(Device::Metal)` は `MTLCopyAllDevices()` の先頭を返すため、判定も
    // 同じ列挙の先頭デバイスで行う（マルチ GPU Mac では
    // `MTLCreateSystemDefaultDevice()` と別デバイスになりうる）。
    let is_apple_gpu = MTLCopyAllDevices()
        .to_vec()
        .first()
        .is_some_and(|device| device.supportsFamily(MTLGPUFamily::Apple1));
    if is_apple_gpu {
        assert_eq!(info.warp_width, Some(32));
    } else {
        assert!(matches!(info.warp_width, Some(32) | None));
    }
}

/// Apple GPU の simdgroup 幅は 32（#2125）。`probe_all` は Apple ファミリ
/// （`supportsFamily(Apple1)`）と確認できない GPU（Intel/AMD 搭載 Mac 等）では
/// `None`（不明）を報告する契約のため、全列挙デバイスに `Some(32)` は要求せず
/// `Some(32)` または `None` のみを許容する。Apple Silicon 実機での `Some(32)`
/// は `select_metal_device_on_real_hardware`（`#[ignore]`）で検証する。
#[test]
fn enumerated_devices_report_warp_width_32_or_unknown() {
    for info in MetalDeviceProvider::new().enumerate().expect("enumerate") {
        assert!(
            matches!(info.warp_width, Some(32) | None),
            "warp_width must be Some(32) or None, got {:?}",
            info.warp_width
        );
    }
}
