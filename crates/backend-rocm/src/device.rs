//! ROCm の `DeviceProvider`（イシュー #3853・親 #3711。承認: #3771 の所有者回答）。
//!
//! 役割: `facade` の `available_devices`／`resolve_ops` が ROCm デバイスの存在を検証・列挙するための
//! 入口。`backend-cuda::CudaDeviceProvider` と同型で、`facade` だけが使い、`facade` は本型を
//! 再エクスポートしない（`api_surface.rs` の否定ガード）。
//!
//! 承認済みの設計（`docs/backend-rocm-device-exposure-decision.md` §10・§13）:
//! - 対象は `Device::Rocm(0)` の 1 台だけ（HIP デバイス ID 0 固定。`RocmGemmF32::new` と同じ）。
//!   ordinal 1 以上は HIP に触れる前に `DeviceUnavailable` にする
//! - gfx 判定は案 (c): 既存の FFI だけで gfx942 の code object をデバイス 0 へロードできるかを見る
//!   （`RocmGemmF32::new` = HIP ロード → HIPRTC ロード → `device_count` → コンパイル → `Module::load`）。
//!   新しい HIP シンボルも `unsafe` も足さない
//! - 失敗は固定文言の `DeviceUnavailable`。成功だけをプロセス全体でキャッシュし、失敗は固定しない
//!   （後から ROCm が使えるようになる環境で再試行できる）

use std::sync::OnceLock;

use fandhe_ai_tensor_core::{BackendError, Device, DeviceInfo, DeviceProvider};

use crate::gemm::RocmGemmF32;
use crate::hip;
use crate::target::GfxTarget;

/// 判定済みの gfx target（成功だけを保持する）。
static PROBED: OnceLock<GfxTarget> = OnceLock::new();

/// 承認済みの唯一の対象 target（allowlist）。
const TARGET: GfxTarget = GfxTarget::Gfx942;

/// ROCm 用 `DeviceProvider`。状態を持たず、判定結果はプロセス全体の `OnceLock` に載る。
#[derive(Debug, Clone, Copy, Default)]
pub struct RocmDeviceProvider;

impl RocmDeviceProvider {
    /// 状態を持たない provider を作る。HIP には触れない。
    pub fn new() -> Self {
        Self
    }

    /// `select` と同じ検証をして gfx target を返す。`facade::resolve_ops` が
    /// `RocmBackendOps::new(target)` を作るために呼ぶ（成功済みならキャッシュヒットで軽い）。
    pub fn resolve_target(&self, device: Device) -> Result<GfxTarget, BackendError> {
        let ordinal = validate_ordinal(device)?;
        probe_target(ordinal)
    }
}

/// `device` が `Rocm(0)` であることを HIP に触れる前に検証し、HIP デバイス ID を返す。
///
/// `Rocm(n >= 1)`（`usize::MAX` を含む）と他 variant は `DeviceUnavailable`。
fn validate_ordinal(device: Device) -> Result<i32, BackendError> {
    match device {
        Device::Rocm(n) => i32::try_from(n).ok().filter(|&id| id == 0).ok_or_else(|| {
            unavailable("only Device::Rocm(0) is supported by this fandhe-ai build")
        }),
        _ => Err(unavailable(
            "RocmDeviceProvider cannot select a non-ROCm device",
        )),
    }
}

fn unavailable(msg: &str) -> BackendError {
    BackendError::DeviceUnavailable(msg.to_string())
}

/// HIP デバイス `ordinal` に gfx942 の code object をロードできるかを fail-closed で判定する。
///
/// エラーメッセージは固定文言のみ（`RocmError` の Display は HIPRTC ログやライブラリ候補名を含みうる）。
fn probe_target(ordinal: i32) -> Result<GfxTarget, BackendError> {
    if let Some(t) = PROBED.get() {
        return Ok(*t);
    }
    let rt = hip::load().map_err(|_| unavailable("ROCm runtime (HIP) is not available"))?;
    let count = rt
        .device_count()
        .map_err(|_| unavailable("ROCm runtime (HIP) is not available"))?;
    if count < 1 || ordinal < 0 || ordinal >= count {
        return Err(unavailable("no HIP device is available"));
    }
    // 案 (c): GEMM カーネルのコンパイルとロードが通れば gfx942 の code object がデバイス 0 で動く。
    // 同じスレッドのカーネルキャッシュも温まる。
    crate::ops::cached(&crate::ops::GEMM_CACHE, TARGET, RocmGemmF32::new)
        .map_err(|_| unavailable("gfx942 code object could not be loaded on HIP device 0"))?;
    Ok(*PROBED.get_or_init(|| TARGET))
}

fn info(target: GfxTarget) -> DeviceInfo {
    DeviceInfo::new(Device::Rocm(0), "rocm:gfx942", None, None)
        .with_warp_width(Some(target.wave_size()))
}

impl DeviceProvider for RocmDeviceProvider {
    fn backend_name(&self) -> &'static str {
        "rocm"
    }

    /// `enumerate` が空でないこと（HIP はあるが HIPRTC か gfx942 のロードが無いホストで、
    /// 一貫性契約〈列挙された Device は `tape_for` が Ok〉が崩れないようにする。設計記録 §13 D1）。
    fn is_available(&self) -> bool {
        self.enumerate().map(|v| !v.is_empty()).unwrap_or(false)
    }

    /// 使えるとき `Rocm(0)` を 1 件だけ返す。使えないときは `Ok(vec![])`（fail-safe）。
    fn enumerate(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        Ok(match probe_target(0) {
            Ok(t) => vec![info(t)],
            Err(_) => Vec::new(),
        })
    }

    fn select(&self, device: Device) -> Result<DeviceInfo, BackendError> {
        self.resolve_target(device).map(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_ordinal_table() {
        assert_eq!(validate_ordinal(Device::Rocm(0)).ok(), Some(0));
        for d in [
            Device::Cpu,
            Device::Cuda(0),
            Device::Rocm(1),
            Device::Rocm(usize::MAX),
        ] {
            assert!(
                matches!(validate_ordinal(d), Err(BackendError::DeviceUnavailable(_))),
                "{d:?}"
            );
        }
    }

    #[test]
    fn backend_name_is_rocm() {
        assert_eq!(RocmDeviceProvider::new().backend_name(), "rocm");
    }
}
