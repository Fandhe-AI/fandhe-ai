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
/// 通常 CI 環境でも実行可能）。`CudaDeviceProvider::enumerate` は
/// `probe` に失敗した ordinal を `filter_map` で除外するため（`crates/
/// backend-cuda/src/device.rs::enumerate`）、GPU 0 の初期化にのみ失敗
/// し GPU 1 が利用可能な環境では `[1]` のように非連続・非 0 始まりの
/// 列挙が正当な結果になりうる。よって「0 から昇順で連続する」という
/// 連続性の期待は誤りであり（codex-review 指摘。PR #1864）、ここでは
/// `enumerate_all`／`enumerate` の実装契約が保証する「ordinal の
/// 昇順・重複なし」のみを検証する。
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
    assert!(
        cuda_devices.windows(2).all(|pair| pair[0] < pair[1]),
        "Device::Cuda(ordinal) は昇順・重複なしであることを期待する: {cuda_devices:?}"
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
// ---------------------------------------------------------------------------
// イシュー #3853: ROCm（`Device::Rocm(0)`）の結線。ROCm 非搭載の CI でも通る
// 環境適応型（provider の判定と突き合わせる）。
// ---------------------------------------------------------------------------

fn rocm_provider_available() -> bool {
    fandhe_ai_tensor_core::DeviceProvider::is_available(
        &fandhe_ai_backend_rocm::RocmDeviceProvider::new(),
    )
}

/// `Device::Rocm(_)` の有無が provider の判定と一致し、あるときは `Rocm(0)` がちょうど 1 件。
#[test]
fn available_devices_excludes_rocm_without_hip() {
    let rocm: Vec<Device> = fandhe_ai::available_devices()
        .into_iter()
        .filter(|d| matches!(d, Device::Rocm(_)))
        .collect();
    if rocm_provider_available() {
        assert_eq!(rocm, vec![Device::Rocm(0)]);
    } else {
        assert!(rocm.is_empty(), "ROCm 不在なのに列挙された: {rocm:?}");
    }
}

/// provider が使えないとき `tape_for(Rocm(0))`・メモリ API は `DeviceUnavailable`（panic しない）。
#[test]
fn tape_for_rocm_is_device_unavailable_without_hip() {
    if rocm_provider_available() {
        return;
    }
    assert!(matches!(
        fandhe_ai::tape_for(Device::Rocm(0)),
        Err(fandhe_ai::BackendError::DeviceUnavailable(_))
    ));
    assert!(fandhe_ai::release_cached_memory(Device::Rocm(0)).is_err());
    assert!(fandhe_ai::memory_pool_stats(Device::Rocm(0)).is_err());
}

/// 範囲外 ordinal は無条件に `DeviceUnavailable`（承認 #3771 の点 4）。
#[test]
fn tape_for_rocm_out_of_range_ordinal_is_err() {
    for d in [Device::Rocm(1), Device::Rocm(usize::MAX)] {
        assert!(
            matches!(
                fandhe_ai::tape_for(d),
                Err(fandhe_ai::BackendError::DeviceUnavailable(_))
            ),
            "{d:?}"
        );
    }
}

/// 列挙順は Cpu → Cuda → Metal → Rocm（rank が単調非減少）。
#[test]
fn available_devices_order_is_cpu_cuda_metal_rocm() {
    fn rank(d: &Device) -> u8 {
        match d {
            Device::Cpu => 0,
            Device::Cuda(_) => 1,
            #[cfg(target_os = "macos")]
            Device::Metal => 2,
            Device::Rocm(_) => 3,
            _ => 4,
        }
    }
    let ranks: Vec<u8> = fandhe_ai::available_devices().iter().map(rank).collect();
    assert!(ranks.windows(2).all(|w| w[0] <= w[1]), "{ranks:?}");
}

/// 性能比較の事前登録（設計記録 §13）用の計測。assert なし・結果は標準出力のみ。
#[test]
#[ignore = "計測専用（設計記録 §13 の事前登録規則。5 回別プロセスで中央値）"]
fn available_devices_call_time_report() {
    const N: u32 = 200;
    let _ = fandhe_ai::available_devices(); // ウォームアップ
    let start = std::time::Instant::now();
    for _ in 0..N {
        std::hint::black_box(fandhe_ai::available_devices());
    }
    println!(
        "available_devices_ns_per_call={}",
        start.elapsed().as_nanos() / u128::from(N)
    );
}
