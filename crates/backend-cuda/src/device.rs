//! CUDA バックエンドのデバイス列挙・選択（TASK-1.9a・#44）。
//!
//! `tensor-core::device::DeviceProvider` の CUDA 実装。`cudarc` は無条件
//! 依存＋動的ロード方式であるため（`.claude/rules/deps-policy.md`）、CUDA
//! toolkit・ドライバが非搭載の環境でも本クレートはビルドが成立する。この
//! 契約の実行時側の受け皿として、本 provider はドライバ不在時に
//! `panic!`／`unwrap()` せず `is_available() == false`・
//! `enumerate() == Ok(vec![])` を返す（REQ-1・`docs/public-api-design.md`
//! §4.4 `BackendError::CudaUnavailable` のコメント参照）。
//!
//! `cudarc` の driver API（`CudaContext::new`／`device_count`／`name`／
//! `total_mem`／`attribute`）はいずれも `Result` を返す safe API であり
//! （PoC-v2-5 実測: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/cuda/mod.rs:92,120-121`）、
//! 本ファイルは `unsafe` を使用しない。

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, DriverError};
use tensor_core::device::{BackendError, Device, DeviceInfo, DeviceProvider};

/// CUDA バックエンドの `DeviceProvider` 実装。`enumerate`／`select` の
/// 呼び出しごとに `CudaContext::device_count`／`CudaContext::new` で
/// プローブする（本イシューのスコープはデバイス検出・プロパティ取得のみで
/// あり、コンテキストの常駐・再利用は `BackendOps` 結線を担う TASK-1.9c
/// （#46）に引き継ぐ）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaDeviceProvider;

impl CudaDeviceProvider {
    /// 新規 provider を構築する。CUDA ドライバの検出自体は
    /// `is_available`／`enumerate`／`select` 呼び出し時に遅延して行う
    /// （構築時点ではプローブしない）。
    pub fn new() -> Self {
        Self
    }

    /// 指定 ordinal のデバイス情報を取得する。`CudaContext::new` が
    /// 失敗した場合（ドライバ不在・範囲外 ordinal 等）は `DriverError`
    /// をそのまま呼び出し元へ伝播し、`CudaUnavailable`／
    /// `DeviceUnavailable` への変換は呼び出し元（`enumerate`／`select`）
    /// が文脈に応じて行う。
    fn probe(ordinal: usize) -> Result<DeviceInfo, DriverError> {
        let ctx = CudaContext::new(ordinal)?;
        let name = ctx.name()?;
        // total_mem／attribute の取得失敗はプロパティ欠損として
        // Option::None に落とし、デバイス自体の検出成功を優先する
        // （name／select の成否がドライバ疎通の主判定材料であるため）。
        let total_memory_bytes = ctx.total_mem().ok().map(|bytes| bytes as u64);
        let compute_units = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .ok()
            .and_then(|count| u32::try_from(count).ok());
        Ok(DeviceInfo::new(
            Device::Cuda(ordinal),
            name,
            total_memory_bytes,
            compute_units,
        ))
    }
}

impl DeviceProvider for CudaDeviceProvider {
    fn backend_name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        matches!(CudaContext::device_count(), Ok(count) if count > 0)
    }

    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        // ドライバ不在（toolkit 非搭載環境等）は `Err` ではなく空列挙を
        // 返す。呼び出し元（`enumerate_all`）が「1 バックエンドの不在で
        // 全体の列挙が止まらない」ことを前提にできるようにするため
        // （モジュール冒頭コメント参照）。
        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(_) => return Ok(vec![]),
        };
        let devices = (0..count.max(0) as usize)
            .filter_map(|ordinal| Self::probe(ordinal).ok())
            .collect();
        Ok(devices)
    }

    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
        let ordinal = match device {
            Device::Cuda(ordinal) => ordinal,
            other => {
                return Err(BackendError::DeviceUnavailable(format!(
                    "CudaDeviceProvider cannot select {other:?}"
                )));
            }
        };
        Self::probe(ordinal).map_err(|err| BackendError::CudaUnavailable(format!("{err:?}")))
    }
}
