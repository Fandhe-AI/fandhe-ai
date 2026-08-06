//! CPU バックエンドのデバイス列挙・選択（TASK-1.9a・#44）。
//!
//! `tensor-core::device::DeviceProvider` の CPU 実装。CPU バックエンドは
//! `backend-cuda`／`backend-metal` と異なり実行時プローブを要さず常に
//! 利用可能であるため、参照実装として `enumerate_all`／`select_from`
//! （`tensor-core::device`）の非エラー経路を確認する基準点になる
//! （`.claude/rules/coding-rust.md` の「CPU 参照実装」方針と同種の位置付け）。

use tensor_core::device::{BackendError, Device, DeviceInfo, DeviceProvider};

/// CPU バックエンドの `DeviceProvider` 実装。デバイスは常に `Device::Cpu`
/// 1 件のみで、`compute_units` は論理コア数
/// （`std::thread::available_parallelism`。std のみで新規依存を要しない）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuDeviceProvider;

impl CpuDeviceProvider {
    /// 新規 provider を構築する。CPU は常時利用可能なため、構築自体が
    /// 失敗することはない。
    pub fn new() -> Self {
        Self
    }

    /// 論理コア数を取得する。取得に失敗した場合（`available_parallelism`
    /// がプラットフォーム制約で `Err` を返す場合）は `None` とし、
    /// 呼び出し元を `panic!`／`unwrap()` させない
    /// （`.claude/rules/coding-rust.md`）。
    fn compute_units() -> Option<u32> {
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get() as u32)
    }
}

impl DeviceProvider for CpuDeviceProvider {
    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        Ok(vec![DeviceInfo::new(
            Device::Cpu,
            "cpu",
            None,
            Self::compute_units(),
        )])
    }

    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
        match device {
            Device::Cpu => Ok(DeviceInfo::new(
                Device::Cpu,
                "cpu",
                None,
                Self::compute_units(),
            )),
            // CUDA／Metal のデバイスを CPU provider に選択させるのは呼び
            // 出し側（`select_from`）の誤配線であり、backend_name の不一致
            // として扱う（型付きエラーで通知し panic しない）。
            other => Err(BackendError::DeviceUnavailable(format!(
                "CpuDeviceProvider cannot select {other:?}"
            ))),
        }
    }
}
