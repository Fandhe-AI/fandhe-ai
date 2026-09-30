//! warp 幅に依存するカーネル定数の導出模型（イシュー #2125・AMD ROCm
//! readiness。`docs/backend-abstraction-amd-readiness-decision.md` §2・§3）。
//!
//! 役割: NVRTC ソース（rmsnorm／softmax／log_softmax／mse）へ `WARP_SIZE`／
//! `WARP_HALF`／`WARP_FULL_MASK`／`WARPS_PER_BLOCK` を数値 `#define` として
//! レンダリング時注入する際の数値の単一の導出元（イシュー #2126。呼び出し元:
//! `kernels_rmsnorm.rs`／`kernels_softmax.rs`／`kernels_mse.rs` の `render_*`、
//! それを呼ぶ `rmsnorm.rs`／`softmax.rs`／`mse.rs` の `new`）。実行時の
//! デバイス属性（`CudaDevice::warp_size()`＝`DeviceInfo::warp_width` と同じ
//! 取得元）から [`WarpGeometry::for_cuda`] で構築する。
//!
//! CUDA の `__shfl_xor_sync`／`__syncwarp` のマスク引数は 32 bit のため、
//! 現時点の CUDA 注入は width 32 のみ受理し、それ以外は NVRTC へ渡す前に
//! `InvalidKernelConfig` で fail-closed 拒否する（wave64／HIP 対応は後続
//! #2127 以降。`full_lane_mask` の 64 bit 値をそのまま注入すると誤動作する）。
//!
//! 数値のみを出力し文字列入力を受けない設計とする（NVRTC ソースへ外部入力を
//! 混入させないため。`kernels_tiled_pipeline*.rs::render_source` と同方針）。

use crate::device::CudaDevice;
use crate::error::CudaError;
#[cfg(test)]
use fandhe_ai_tensor_core::device::DeviceInfo;

/// CUDA ソースへ注入できる唯一の warp 幅（マスク引数が 32 bit のため）。
pub(crate) const CUDA_SUPPORTED_WARP_WIDTH: u32 = 32;

/// warp／wave 幅から導出されるカーネル定数の集合。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WarpGeometry {
    width: u32,
}

impl WarpGeometry {
    /// 2 のべき乗かつ `1..=64` のみ受理する（0・非 2 べき・64 超は `None`。
    /// fail-closed）。
    pub(crate) fn new(width: u32) -> Option<Self> {
        (width.is_power_of_two() && width <= 64).then_some(Self { width })
    }

    /// `device.warp_size()`（ドライバ属性）から構築する。取得失敗（`None`）・
    /// 非 2 べき・CUDA 注入が対応しない幅は 32 と推定せず型付きエラーで
    /// 拒否する（fail-closed）。
    pub(crate) fn for_cuda(device: &CudaDevice) -> Result<Self, CudaError> {
        Self::from_cuda_warp_size(device.warp_size())
    }

    /// [`Self::for_cuda`] の純粋部（実機なしでテスト可能にするため分離）。
    pub(crate) fn from_cuda_warp_size(size: Option<u32>) -> Result<Self, CudaError> {
        let width = size.ok_or_else(|| CudaError::InvalidKernelConfig {
            detail: "warp 幅を取得できないためカーネルへ WARP_SIZE を注入できない".to_string(),
        })?;
        let geom = Self::new(width).ok_or_else(|| CudaError::InvalidKernelConfig {
            detail: format!("warp 幅 {width} は 2 のべき乗かつ 64 以下でない"),
        })?;
        geom.ensure_cuda_injectable()?;
        Ok(geom)
    }

    /// 幅（lane 数。不変条件の固定テストが参照する）。
    #[cfg(test)]
    pub(crate) fn width(self) -> u32 {
        self.width
    }

    /// CUDA ソースへ注入可能な幅（32）か検査する。`render_*` が文字列を
    /// 組み立てる前に必ず呼ぶ（64 幅を 32 bit マスクへ流すと黙って誤った
    /// 結果になるため fail-closed）。
    pub(crate) fn ensure_cuda_injectable(self) -> Result<(), CudaError> {
        if self.width == CUDA_SUPPORTED_WARP_WIDTH {
            Ok(())
        } else {
            Err(CudaError::InvalidKernelConfig {
                detail: format!(
                    "warp 幅 {} は CUDA カーネルへ注入できない（__shfl_xor_sync のマスクは 32 bit。\
                     対応幅は {CUDA_SUPPORTED_WARP_WIDTH} のみ）",
                    self.width
                ),
            })
        }
    }

    /// NVRTC ソース先頭へ連結する数値 `#define` のプレフィクスを返す
    /// （`kernels_tiled_pipeline.rs::render_defines` と同方針: 数値のみを
    /// `format!` で焼き込み、文字列の外部入力は受けない）。`block_dim` を
    /// 渡すと `WARPS_PER_BLOCK` も定義する（割り切れなければ fail-closed）。
    pub(crate) fn render_defines(self, block_dim: Option<u32>) -> Result<String, CudaError> {
        self.ensure_cuda_injectable()?;
        let mut out = format!(
            "#define WARP_SIZE {}\n#define WARP_HALF {}\n#define WARP_FULL_MASK 0x{:08x}u\n\
             #define WARP_SHFL_XOR(v, off) __shfl_xor_sync(WARP_FULL_MASK, (v), (off))\n\
             #define WARP_SYNC() __syncwarp(WARP_FULL_MASK)\n",
            self.width,
            self.half_width(),
            self.full_lane_mask() as u32,
        );
        if let Some(dim) = block_dim {
            let n = self
                .warps_per_block(dim)
                .ok_or_else(|| CudaError::InvalidKernelConfig {
                    detail: format!("block_dim {dim} は warp 幅 {} の倍数でない", self.width),
                })?;
            out.push_str(&format!("#define WARPS_PER_BLOCK {n}\n"));
        }
        Ok(out)
    }

    /// `DeviceInfo::warp_width` から構築する（`None` を伝播する）。
    #[cfg(test)]
    pub(crate) fn from_device_info(info: &DeviceInfo) -> Option<Self> {
        info.warp_width.and_then(Self::new)
    }

    /// butterfly reduction の初期 offset（32 → 16。設計記録 §2 (B)）。
    pub(crate) fn half_width(self) -> u32 {
        self.width / 2
    }

    /// 全レーンのマスク（32 → `0xffff_ffff`、64 → `u64::MAX`。設計記録 §3
    /// 「wave64 ではマスクが 64 bit 化しうる」）。
    pub(crate) fn full_lane_mask(self) -> u64 {
        if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }

    /// ブロック内の warp 数（`warp_sums[N]`／`lane < N` の導出元。設計記録
    /// §2 (B')）。`block_dim` が 0 または幅の倍数でなければ `None`。
    pub(crate) fn warps_per_block(self, block_dim: u32) -> Option<u32> {
        (block_dim != 0 && block_dim.is_multiple_of(self.width)).then(|| block_dim / self.width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels_mse::MSE_BLOCK_DIM;
    use crate::kernels_rmsnorm::RMSNORM_BLOCK_DIM;
    use crate::kernels_softmax::SOFTMAX_BLOCK_DIM;
    use fandhe_ai_tensor_core::device::Device;

    #[test]
    fn cuda_injection_is_fail_closed() {
        assert!(WarpGeometry::from_cuda_warp_size(None).is_err());
        assert!(WarpGeometry::from_cuda_warp_size(Some(0)).is_err());
        assert!(WarpGeometry::from_cuda_warp_size(Some(48)).is_err());
        assert!(WarpGeometry::from_cuda_warp_size(Some(64)).is_err());
        let g = WarpGeometry::from_cuda_warp_size(Some(32)).expect("32");
        assert_eq!(g.width(), 32);
        // width 64 はレンダ前に拒否される（32 bit マスクへ流さない）。
        let g64 = WarpGeometry::new(64).expect("64");
        assert!(g64.render_defines(Some(256)).is_err());
    }

    #[test]
    fn render_defines_width32() {
        let g = WarpGeometry::new(32).expect("32");
        let d = g.render_defines(Some(256)).expect("render");
        for needle in [
            "#define WARP_SIZE 32\n",
            "#define WARP_HALF 16\n",
            "#define WARP_FULL_MASK 0xffffffffu\n",
            "#define WARPS_PER_BLOCK 8\n",
        ] {
            assert!(d.contains(needle), "{needle}");
        }
        assert!(
            !g.render_defines(None)
                .expect("render")
                .contains("WARPS_PER_BLOCK")
        );
        assert!(g.render_defines(Some(48)).is_err());
    }

    #[test]
    fn width32_matches_current_kernel_literals() {
        let g = WarpGeometry::new(32).expect("32");
        // kernels_mse.rs の `offset = 16`・`0xffffffff`・`warp_sums[8]`。
        assert_eq!(g.half_width(), 16);
        assert_eq!(g.full_lane_mask(), 0xffff_ffff);
        assert_eq!(g.warps_per_block(MSE_BLOCK_DIM), Some(8));
        assert_eq!(g.warps_per_block(RMSNORM_BLOCK_DIM), Some(1));
        assert_eq!(g.warps_per_block(SOFTMAX_BLOCK_DIM), Some(1));
    }

    #[test]
    fn width64_derivations() {
        let g = WarpGeometry::new(64).expect("64");
        assert_eq!(g.half_width(), 32);
        assert_eq!(g.full_lane_mask(), u64::MAX);
        assert_eq!(g.warps_per_block(256), Some(4));
    }

    #[test]
    fn invalid_values_are_rejected() {
        for w in [0, 48, 128] {
            assert_eq!(WarpGeometry::new(w), None);
        }
        let g = WarpGeometry::new(32).expect("32");
        assert_eq!(g.warps_per_block(0), None);
        assert_eq!(g.warps_per_block(48), None);
    }

    #[test]
    fn from_device_info_propagates_none() {
        let cpu = DeviceInfo::new(Device::Cpu, "cpu", None, None);
        assert_eq!(WarpGeometry::from_device_info(&cpu), None);
        let gpu = cpu.with_warp_width(Some(32));
        assert_eq!(WarpGeometry::from_device_info(&gpu), WarpGeometry::new(32));
    }
}
