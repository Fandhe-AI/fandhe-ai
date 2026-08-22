//! TASK-1.9a（#44）の受け入れ条件「3 バックエンド（CPU／CUDA／Metal）が
//! 同一 trait でデバイス列挙・選択できる」を直接検証する統合テスト。
//!
//! `backend-cpu::CpuDeviceProvider`・`fandhe_ai_backend_cuda::CudaDeviceProvider`・
//! （macOS のみ）`fandhe_ai_backend_metal::MetalDeviceProvider` を
//! `Vec<Box<dyn DeviceProvider>>` へ格納し、`fandhe_ai_tensor_core::device` の
//! `enumerate_all`／`select_from` が単一の trait オブジェクト経由で
//! 3 バックエンドすべてを横断できることを確認する。3 バックエンド網羅の
//! 本格的な統合テスト（実カーネル呼び出しを含む）は TASK-1.9d（#47）が
//! `backend_ops_integration.rs`（同ディレクトリ）以下で担い、本テストは
//! デバイス抽象層の受け入れ条件（列挙・選択）に限定する。

use fandhe_ai_backend_cpu::CpuDeviceProvider;
use fandhe_ai_backend_cuda::CudaDeviceProvider;
use fandhe_ai_tensor_core::device::{Device, DeviceProvider, enumerate_all, select_from};

#[cfg(target_os = "macos")]
use fandhe_ai_backend_metal::MetalDeviceProvider;

/// CPU・CUDA・（macOS のみ）Metal の provider を `Box<dyn DeviceProvider>`
/// として束ねる。CUDA ドライバ・Metal デバイスが実行環境に無くても
/// provider 自体は構築でき、`enumerate`／`select` が fail-safe に応答する
/// （モジュール冒頭コメント参照）。
fn all_providers() -> Vec<Box<dyn DeviceProvider>> {
    // 非 macOS では Metal 分の push が存在せず不変のままになるため
    // `unused_mut` を許容する（macOS では実際に mut のため無害）。
    // `-D warnings`（`.claude/rules/coding-rust.md`）環境で非 macOS
    // ビルドが失敗しないようにするための cfg 非依存の対処。
    #[allow(unused_mut)]
    let mut providers: Vec<Box<dyn DeviceProvider>> = vec![
        Box::new(CpuDeviceProvider::new()),
        Box::new(CudaDeviceProvider::new()),
    ];
    #[cfg(target_os = "macos")]
    providers.push(Box::new(MetalDeviceProvider::new()));
    providers
}

#[test]
fn same_trait_enumerates_all_configured_backends() {
    let providers = all_providers();
    let refs: Vec<&dyn DeviceProvider> = providers.iter().map(|p| p.as_ref()).collect();

    // CPU は常時利用可能なため、CUDA／Metal の実行環境有無に関わらず
    // 少なくとも 1 件（CPU）は必ず列挙される。
    let all = enumerate_all(&refs);

    assert!(
        all.iter().any(|info| info.device == Device::Cpu),
        "CPU デバイスは常に列挙されるはず"
    );
}

#[test]
fn same_trait_selects_cpu_through_shared_dispatch_helper() {
    let providers = all_providers();
    let refs: Vec<&dyn DeviceProvider> = providers.iter().map(|p| p.as_ref()).collect();

    // 3 バックエンドを横断する `select_from` が、`Device::Cpu` を渡した
    // 際に正しく `backend_name() == "cpu"` の provider へディスパッチ
    // できることを確認する（受け入れ条件「同一 trait で選択できる」）。
    let info = select_from(&refs, Device::Cpu).expect("CPU selection must succeed");

    assert_eq!(info.device, Device::Cpu);
    assert_eq!(info.name, "cpu");
}

#[test]
fn each_provider_reports_distinct_backend_name() {
    let providers = all_providers();
    let names: Vec<&'static str> = providers.iter().map(|p| p.backend_name()).collect();

    assert!(names.contains(&"cpu"));
    assert!(names.contains(&"cuda"));
    #[cfg(target_os = "macos")]
    assert!(names.contains(&"metal"));
}

#[test]
fn cuda_absence_does_not_break_other_backends_enumeration() {
    // CUDA ドライバ非搭載の CI（self-hosted Linux 等）でも、CPU（と macOS
    // なら Metal）の列挙が止まらないことを確認する（fail-safe。
    // `.claude/rules/ci.md` の実機依存分離方針の受け皿）。
    let providers = all_providers();
    let refs: Vec<&dyn DeviceProvider> = providers.iter().map(|p| p.as_ref()).collect();

    let cuda_provider = refs
        .iter()
        .find(|p| p.backend_name() == "cuda")
        .expect("cuda provider must be registered");
    let cuda_enumerate_result = cuda_provider.enumerate();

    assert!(
        cuda_enumerate_result.is_ok(),
        "CudaDeviceProvider::enumerate はドライバ不在でも Err にならない"
    );

    let all = enumerate_all(&refs);
    assert!(all.iter().any(|info| info.device == Device::Cpu));
}
