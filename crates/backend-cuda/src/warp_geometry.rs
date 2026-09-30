//! warp 幅に依存するカーネル定数の導出模型（イシュー #2125・AMD ROCm
//! readiness。`docs/backend-abstraction-amd-readiness-decision.md` §2・§3）。
//!
//! 役割: #2126 が NVRTC／HIPRTC ソースへ `WARP_SIZE`／`WARP_HALF` 等を
//! レンダリング時注入する際の数値の単一の導出元になる予定の模型。現時点では
//! カーネルソース・起動経路へ結線しておらず（`cfg(test)` 限定）、width 32 の
//! 導出値が現行カーネルのリテラルと一致することを固定テストで示す。
//! 結線時（#2126）に `cfg(test)` を外す。
//!
//! 数値のみを出力し文字列入力を受けない設計とする（NVRTC ソースへ外部入力を
//! 混入させないため。`kernels_tiled_pipeline*.rs::render_source` と同方針）。

use fandhe_ai_tensor_core::device::DeviceInfo;

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

    /// `DeviceInfo::warp_width` から構築する（`None` を伝播する）。
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
