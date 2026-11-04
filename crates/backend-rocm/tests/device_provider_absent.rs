//! `RocmDeviceProvider`（イシュー #3853）の CI 向け契約テスト（`#[ignore]` なし）。
//!
//! ROCm 非搭載の ubuntu-latest では「不在側の分岐」を通り、AMD ホストでは何もしない（または
//! 一貫性だけを見る）ので、どのホストでも壊れない。実際に不在側を通る保証は
//! `build-no-cuda-toolkit` ジョブ（`scripts/check-rocm-absent.sh assert` の後に test を回す）が担う。

use fandhe_ai_backend_rocm::{RocmDeviceProvider, is_available};
use fandhe_ai_tensor_core::{BackendError, Device, DeviceProvider};

#[test]
fn rocm_device_provider_enumerate_is_empty_without_hip() {
    if is_available() {
        return; // AMD ホストでは対象外（実機は facade 側の ignored テスト）
    }
    let p = RocmDeviceProvider::new();
    assert_eq!(p.enumerate().expect("enumerate は Err にしない").len(), 0);
}

#[test]
fn rocm_device_provider_select_is_device_unavailable_without_hip() {
    if is_available() {
        return;
    }
    let p = RocmDeviceProvider::new();
    assert!(matches!(
        p.select(Device::Rocm(0)),
        Err(BackendError::DeviceUnavailable(_))
    ));
}

#[test]
fn rocm_device_provider_rejects_other_ordinals_and_devices_on_any_host() {
    let p = RocmDeviceProvider::new();
    for d in [
        Device::Rocm(1),
        Device::Rocm(usize::MAX),
        Device::Cpu,
        Device::Cuda(0),
    ] {
        assert!(
            matches!(p.select(d), Err(BackendError::DeviceUnavailable(_))),
            "{d:?}"
        );
    }
}

#[test]
fn rocm_device_provider_is_consistent_on_any_host() {
    let p = RocmDeviceProvider::new();
    assert_eq!(p.backend_name(), "rocm");
    let listed = p.enumerate().expect("enumerate は Err にしない");
    assert_eq!(p.is_available(), !listed.is_empty());
    assert!(listed.len() <= 1);
    if let Some(info) = listed.first() {
        assert_eq!(info.device, Device::Rocm(0));
        assert!(p.select(Device::Rocm(0)).is_ok());
    }
}
