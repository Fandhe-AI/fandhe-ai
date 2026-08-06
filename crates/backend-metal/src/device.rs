//! Metal バックエンドのデバイス列挙・選択（TASK-1.9a・#44）。
//!
//! `tensor-core::device::DeviceProvider` の Metal 実装。`objc2`／
//! `objc2-foundation`／`objc2-metal` は `cfg(target_os = "macos")` 限定の
//! 許容依存であり（`.claude/rules/deps-policy.md`）、本モジュールも同じ
//! cfg でクレート全体（`lib.rs`）から分離する。非 macOS 環境ではこの
//! ファイル自体がコンパイル対象に入らない（`Device::Metal` variant の
//! cfg 境界と整合。TASK-1.9a 実装計画 §3.3）。
//!
//! `MTLCopyAllDevices()`（システム内の全 Metal デバイス列挙）・
//! `MTLDevice::name`／`recommendedMaxWorkingSetSize` はいずれも objc2-metal
//! が safe 関数として提供する（`crates/backend-metal` 内部に `unsafe` を
//! 追加しない。`.claude/rules/security.md`）。

use objc2_metal::{MTLCopyAllDevices, MTLDevice};
use tensor_core::device::{BackendError, Device, DeviceInfo, DeviceProvider};

/// Metal バックエンドの `DeviceProvider` 実装。`Device::Metal` は ordinal
/// を持たない単一 variant のため（`docs/public-api-design.md` §4.1）、
/// `select` は「1 台以上の Metal デバイスが検出できるか」のみを判定する。
/// 複数 GPU を個別に選択する API（ordinal 拡張）は本イシューのスコープ外
/// （§4.1 は `Metal` に ordinal を持たせておらず、拡張する場合は設計書側
/// の変更が必要）。
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalDeviceProvider;

impl MetalDeviceProvider {
    /// 新規 provider を構築する。macOS 上の Metal デバイス検出自体は
    /// `is_available`／`enumerate`／`select` 呼び出し時に行う。
    pub fn new() -> Self {
        Self
    }

    /// `MTLCopyAllDevices()` で検出したデバイスを `DeviceInfo` へ写像する。
    /// システムに Metal デバイスが 1 つも無い場合は空 `Vec` を返す
    /// （fail-safe。`.claude/rules/coding-rust.md`）。
    fn probe_all() -> Vec<DeviceInfo> {
        MTLCopyAllDevices()
            .to_vec()
            .into_iter()
            .map(|device| {
                DeviceInfo::new(
                    Device::Metal,
                    device.name().to_string(),
                    Some(device.recommendedMaxWorkingSetSize()),
                    None,
                )
            })
            .collect()
    }
}

impl DeviceProvider for MetalDeviceProvider {
    fn backend_name(&self) -> &'static str {
        "metal"
    }

    fn is_available(&self) -> bool {
        !Self::probe_all().is_empty()
    }

    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        Ok(Self::probe_all())
    }

    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
        match device {
            Device::Metal => Self::probe_all().into_iter().next().ok_or_else(|| {
                BackendError::DeviceUnavailable("no Metal device detected".to_string())
            }),
            other => Err(BackendError::DeviceUnavailable(format!(
                "MetalDeviceProvider cannot select {other:?}"
            ))),
        }
    }
}
