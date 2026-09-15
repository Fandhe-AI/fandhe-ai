//! `fandhe_ai::available_devices`（イシュー #1614）の契約テスト
//! （`docs/facade-device-transfer-enumeration-design.md` §5「テスト」
//! (B) 節）。
//!
//! 実行環境（CUDA driver・Metal デバイスの有無）に応じて期待集合が
//! 変わるため、`tape_construction.rs` と同じ実行環境適応型パターン
//! （`CudaDeviceProvider::is_available()`／`MetalDeviceProvider::
//! is_available()` を先に問い合わせ、期待される `Device` variant の
//! 有無をそれと突き合わせる）を使う。

use fandhe_ai::Device;

/// `Device::Cpu` は実行環境によらず常に列挙される（`CpuDeviceProvider`
/// は常に 1 件返す）。
#[test]
fn available_devices_always_includes_cpu() {
    let devices = fandhe_ai::available_devices();
    assert!(
        devices.contains(&Device::Cpu),
        "available_devices() は常に Device::Cpu を含むはず: {devices:?}"
    );
}

/// 返された全 `Device` に対して `tape_for` が `Ok` になる（`available_
/// devices` の「一貫性契約」の直接検証）。
#[test]
fn all_available_devices_are_constructible_via_tape_for() {
    let devices = fandhe_ai::available_devices();
    for device in devices {
        fandhe_ai::tape_for(device).unwrap_or_else(|e| {
            panic!("available_devices() が返した {device:?} の tape_for が失敗した: {e}")
        });
    }
}

/// 2 回連続で呼んでも同一の `Vec<Device>` を返す（決定的順序契約）。
#[test]
fn available_devices_is_deterministic_across_calls() {
    let first = fandhe_ai::available_devices();
    let second = fandhe_ai::available_devices();
    assert_eq!(
        first, second,
        "available_devices() は呼び出しごとに同じ列挙を返すはず"
    );
}

/// CUDA driver の有無と `Device::Cuda(_)` の有無が整合する（`CudaDevice
/// Provider::is_available()` を実測してから突き合わせる。実機なしの
/// 通常 CI 環境でも実行可能）。ordinal は 0 から昇順で連続すること
/// （`enumerate_all` の順序契約）も確認する。
#[test]
fn cuda_devices_presence_matches_provider_availability() {
    let provider = fandhe_ai_backend_cuda::CudaDeviceProvider::new();
    let devices = fandhe_ai::available_devices();
    let cuda_devices: Vec<usize> = devices
        .iter()
        .filter_map(|d| match d {
            Device::Cuda(ordinal) => Some(*ordinal),
            _ => None,
        })
        .collect();

    if !fandhe_ai_tensor_core::DeviceProvider::is_available(&provider) {
        assert!(
            cuda_devices.is_empty(),
            "CUDA driver 不在にもかかわらず Device::Cuda(_) が列挙された: {cuda_devices:?}"
        );
        return;
    }
    assert!(
        !cuda_devices.is_empty(),
        "CUDA driver が利用可能なのに Device::Cuda(_) が 1 件も列挙されなかった"
    );
    let expected: Vec<usize> = (0..cuda_devices.len()).collect();
    assert_eq!(
        cuda_devices, expected,
        "Device::Cuda(ordinal) は 0 から昇順で連続することを期待する"
    );
}

/// Metal デバイスの有無と `Device::Metal` の有無が整合する（macOS
/// 限定。`Device::Metal` variant 自体が `cfg(target_os = "macos")`
/// 限定のためテスト関数のコンパイル自体を macOS 限定にする——
/// `unique_backend_parity.rs`／`cast_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
fn metal_device_presence_matches_provider_availability() {
    let provider = fandhe_ai_backend_metal::MetalDeviceProvider::new();
    let devices = fandhe_ai::available_devices();
    let has_metal = devices.contains(&Device::Metal);

    assert_eq!(
        has_metal,
        fandhe_ai_tensor_core::DeviceProvider::is_available(&provider),
        "Device::Metal の有無が MetalDeviceProvider::is_available() と一致しない"
    );
}

/// macOS 以外のビルドでは `Device::Metal` variant 自体が存在しない
/// ため、`available_devices()` の結果にも当然含まれない（コンパイル
/// 時に自明だが、意図を明示するための対照テスト）。
#[cfg(not(target_os = "macos"))]
#[test]
fn non_macos_build_never_yields_metal_variant() {
    let devices = fandhe_ai::available_devices();
    // `Device::Metal` は non-macOS ビルドではそもそも型として
    // 存在しないため、Debug 表記に "Metal" が現れないことで代替確認する。
    for device in &devices {
        assert!(
            !format!("{device:?}").contains("Metal"),
            "non-macOS ビルドで Metal 由来の Device が列挙された: {device:?}"
        );
    }
}
