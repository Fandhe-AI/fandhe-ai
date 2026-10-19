//! ROCm 用の wave 幅注入部品 `WarpGeometry`（イシュー #3432・分解案 R1）。
//!
//! 役割: wave 幅（CDNA = 64・RDNA = 32）から決まる縮約系カーネルの定数
//! （`WARP_SIZE`・`WARP_HALF`・`WARPS_PER_BLOCK`）と HIP 用シャッフルマクロを、検証済みの数値と
//! 固定テンプレートだけで HIPRTC ソースへ注入する。R3 縮約・R4 softmax・R5 loss・R6 正規化・
//! R12 paged attention の render 関数から呼ばれる予定（本 Issue 時点で本番経路の呼び出し元はない）。
//!
//! CUDA 版（`backend-cuda/src/warp_geometry.rs`）との差分（単純移植にしない）:
//! - 幅は 32 と 64 を受け付ける（CUDA 版は 32 のみ）。
//! - `__shfl_xor(var, laneMask, width)` はマスク引数を取らない。`_sync` 版は 64 bit マスクを取るが、
//!   マスク値をソースへ入れず、固定オプション契約（HIPRTC オプションは `GfxTarget` 由来のみ）を
//!   保つため非 sync 版を採用する。
//! - lane mask は 64 bit（[`WarpGeometry::full_lane_mask`]）。32 bit へ切り詰めない。
//! - `WARPS_PER_BLOCK = block_dim / wave 幅` からバッファ長・読み出し境界を導く
//!   （CUDA の 8 warp 固定を置換すると wave64 では未初期化読みになる。
//!   `docs/backend-rocm-ops-coverage-plan.md` §6）。
//!
//! 出典（2026-10-18 確認。HIP C++ 言語拡張 Warp cross-lane functions）:
//! `T __shfl_xor(T var, int laneMask, int width=warpSize)`、
//! `T __shfl_xor_sync(unsigned long long mask, ...)`（ROCm 7.0 で既定有効・
//! `HIP_DISABLE_WARP_SYNC_BUILTINS` で無効化）。`__syncwarp` は同ページに記載がなく確認できないため
//! `WARP_SYNC` は出さない（wave 内の受け渡しは全スレッドが到達する `__syncthreads()` で組む）。
//!
//! 入力は `u32` 2 つと allowlist の enum だけで、`format!` に埋め込むのは数値のみ（A03）。

use crate::error::RocmError;
use crate::target::GfxTarget;

/// ブロックあたりスレッド数の上限（`kernels_gemm.rs` の `MAX_THREADS` と同じ）。
const MAX_BLOCK_DIM: u32 = 1024;

/// 検証済みの wave 幅（32 または 64）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpGeometry {
    width: u32,
}

/// [`WarpGeometry::render_hip_defines`] だけが作れる `#define` 群。任意の文字列は包めない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpDefines(String);

impl WarpDefines {
    /// HIPRTC ソースの先頭に連結できる `#define` 行群。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn bad(detail: String) -> RocmError {
    RocmError::InvalidKernelConfig { detail }
}

impl WarpGeometry {
    /// 32 / 64 のみ受け付ける（fail-closed）。
    pub fn new(width: u32) -> Result<Self, RocmError> {
        match width {
            32 | 64 => Ok(Self { width }),
            _ => Err(bad(format!("wave 幅は 32 か 64 のみ: {width}"))),
        }
    }

    /// デバイス照会値などを受ける純関数。`None`（取得できない）は推定せず拒否する。
    pub fn from_wave_size(size: Option<u32>) -> Result<Self, RocmError> {
        match size {
            Some(w) => Self::new(w),
            None => Err(bad(
                "wave 幅を取得できない（32/64 と推定しない）".to_string()
            )),
        }
    }

    /// allowlist の gfx target の固定値から作る。
    pub fn for_target(target: GfxTarget) -> Self {
        Self {
            width: target.wave_size(),
        }
    }

    /// wave 幅。
    pub fn width(self) -> u32 {
        self.width
    }

    /// butterfly 縮約の初期オフセット（幅の半分）。
    pub fn half_width(self) -> u32 {
        self.width / 2
    }

    /// 全 lane 有効の 64 bit マスク（HIP の lane mask は 64 bit。`__ballot` の戻り値と照合する用途）。
    pub fn full_lane_mask(self) -> u64 {
        if self.width == 64 {
            u64::MAX
        } else {
            0xffff_ffff
        }
    }

    /// `block_dim / 幅`。0・幅の倍数でない・1024 超、および 2 段目縮約を先頭 1 wave で賄えない
    /// （warp 数 > 幅）場合は拒否する。
    pub fn warps_per_block(self, block_dim: u32) -> Result<u32, RocmError> {
        if block_dim == 0 || block_dim > MAX_BLOCK_DIM || !block_dim.is_multiple_of(self.width) {
            return Err(bad(format!(
                "block_dim は 1..={MAX_BLOCK_DIM} の wave 幅({})の倍数のみ: {block_dim}",
                self.width
            )));
        }
        let warps = block_dim / self.width;
        if warps > self.width {
            return Err(bad(format!("warp 数 {warps} が wave 幅を超える")));
        }
        Ok(warps)
    }

    /// 固定テンプレートと検証済み数値だけで `#define` を作る。`block_dim` が `Some` のときだけ
    /// `WARPS_PER_BLOCK` を出す。
    pub fn render_hip_defines(self, block_dim: Option<u32>) -> Result<WarpDefines, RocmError> {
        let warps = block_dim.map(|d| self.warps_per_block(d)).transpose()?;
        let mut s = format!(
            "#define WARP_SIZE {}\n#define WARP_HALF {}\n#define WARP_FULL_MASK 0x{:016x}ull\n",
            self.width,
            self.half_width(),
            self.full_lane_mask()
        );
        s.push_str("#define WARP_LANE(tid) ((tid) % WARP_SIZE)\n");
        s.push_str("#define WARP_ID(tid) ((tid) / WARP_SIZE)\n");
        s.push_str("#define WARP_SHFL_XOR(v, off) __shfl_xor((v), (off), WARP_SIZE)\n");
        if let Some(w) = warps {
            s.push_str(&format!("#define WARPS_PER_BLOCK {w}\n"));
        }
        Ok(WarpDefines(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_only_32_and_64() {
        assert!(WarpGeometry::new(32).is_ok());
        assert!(WarpGeometry::new(64).is_ok());
        for w in [0, 1, 16, 48, 128, u32::MAX] {
            assert!(WarpGeometry::new(w).is_err(), "{w}");
        }
        assert!(WarpGeometry::from_wave_size(None).is_err());
        assert_eq!(WarpGeometry::from_wave_size(Some(64)).unwrap().width(), 64);
        assert_eq!(WarpGeometry::for_target(GfxTarget::Gfx942).width(), 64);
    }

    #[test]
    fn derived_values() {
        let g32 = WarpGeometry::new(32).unwrap();
        let g64 = WarpGeometry::new(64).unwrap();
        assert_eq!((g32.half_width(), g32.full_lane_mask()), (16, 0xffff_ffff));
        assert_eq!((g64.half_width(), g64.full_lane_mask()), (32, u64::MAX));
        assert_eq!(g32.warps_per_block(256).unwrap(), 8);
        assert_eq!(g64.warps_per_block(256).unwrap(), 4);
        for d in [0, 48, 1025, 2048] {
            assert!(g32.warps_per_block(d).is_err(), "{d}");
        }
        assert!(g64.warps_per_block(32).is_err());
        assert!(g64.warps_per_block(100).is_err());
    }

    #[test]
    fn render_exact_lines() {
        let d = WarpGeometry::new(64)
            .unwrap()
            .render_hip_defines(Some(256))
            .unwrap();
        let lines: Vec<&str> = d.as_str().lines().collect();
        assert_eq!(
            lines,
            [
                "#define WARP_SIZE 64",
                "#define WARP_HALF 32",
                "#define WARP_FULL_MASK 0xffffffffffffffffull",
                "#define WARP_LANE(tid) ((tid) % WARP_SIZE)",
                "#define WARP_ID(tid) ((tid) / WARP_SIZE)",
                "#define WARP_SHFL_XOR(v, off) __shfl_xor((v), (off), WARP_SIZE)",
                "#define WARPS_PER_BLOCK 4",
            ]
        );
        let g32 = WarpGeometry::new(32).unwrap();
        let d32 = g32.render_hip_defines(None).unwrap();
        assert!(
            d32.as_str()
                .contains("WARP_FULL_MASK 0x00000000ffffffffull")
        );
        assert!(!d32.as_str().contains("WARPS_PER_BLOCK"));
        assert_eq!(d32.as_str().lines().count(), 6);
        assert_eq!(d32, g32.render_hip_defines(None).unwrap());
    }

    #[test]
    fn render_rejects_bad_block_dim_and_forbidden_text() {
        let g = WarpGeometry::new(64).unwrap();
        assert!(g.render_hip_defines(Some(100)).is_err());
        let s = g.render_hip_defines(Some(256)).unwrap();
        for f in [
            "__shfl_xor_sync",
            "0xffffffffu",
            "warp_sums[8]",
            "lane < 8",
            "__syncwarp",
        ] {
            assert!(!s.as_str().contains(f), "{f}");
        }
    }
}
