//! f32 GEMM の ROCm 実行入口（コンパイル → ロード → 起動）。
//!
//! 役割: `backend-cuda::gemm` の ROCm 側対応物の最小版。公開型 [`RocmGemmF32`] が HIP/HIPRTC を
//! ロードし、検証済みソースをコンパイルして module を保持し、`run` で `C = A·B`（行優先・f32）を
//! 計算する。ホスト側で形状・グリッドを fail-closed に検証してから起動する（REQ-8 のカーネル側
//! 境界チェックと二重化）。facade への統合は #3157、実機での実行確認は #3159。

use crate::error::RocmError;
use crate::hip::{self, HipRuntime};
use crate::hiprtc::{HiprtcRuntime, compile_code_object};
use crate::kernels_gemm::{
    GEMM_F32_ARG_COUNT, GEMM_F32_KERNEL_NAME, GemmF32Tile, render_gemm_f32_source,
};
use crate::module::{DeviceBuffer, Module};
use crate::target::GfxTarget;
use std::ffi::c_void;
use std::sync::Arc;

/// grid.y の保守的な上限（実際の上限照会は #3159）。
const MAX_GRID_Y: usize = 65535;

fn bad(detail: String) -> RocmError {
    RocmError::InvalidGemmDims { detail }
}

/// ホスト側の形状・スライス長検証（CUDA 版 `validate_gemm_dims`＋`validate_tiled_k_bound` 相当）。
///
/// カーネル引数は C の `int`（32bit 符号付き）かつ添字が `int` 演算のため、m・n・k と
/// `m*k`・`k*n`・`m*n` が `i32::MAX` 以下であることを要求する。
pub(crate) fn validate_gemm_dims(
    tile: &GemmF32Tile,
    a_len: usize,
    b_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), RocmError> {
    let max = i32::MAX as usize;
    if m > max || n > max || k > max {
        return Err(bad(format!("m/n/k は i32::MAX 以下: m={m}, n={n}, k={k}")));
    }
    let prod = |x: usize, y: usize, what: &str| {
        x.checked_mul(y)
            .filter(|&p| p <= max)
            .ok_or_else(|| bad(format!("{what} が i32::MAX を超えます")))
    };
    let mk = prod(m, k, "m*k")?;
    let kn = prod(k, n, "k*n")?;
    prod(m, n, "m*n")?;
    if a_len != mk {
        return Err(bad(format!("a の長さ不一致: 期待 {mk}, 実際 {a_len}")));
    }
    if b_len != kn {
        return Err(bad(format!("b の長さ不一致: 期待 {kn}, 実際 {b_len}")));
    }
    let k_limit = max - (tile.bk() as usize - 1);
    if k > k_limit {
        return Err(bad(format!(
            "k はタイル添字計算のため {k_limit} 以下: k={k}"
        )));
    }
    Ok(())
}

/// grid／block の 3 次元寸法。
type Dim3 = (u32, u32, u32);

/// 起動ジオメトリ `(grid, block)`。`m`・`n` は検証済み・非 0 であること。
pub(crate) fn launch_geometry(
    tile: &GemmF32Tile,
    m: usize,
    n: usize,
) -> Result<(Dim3, Dim3), RocmError> {
    let gx = n.div_ceil(tile.bn() as usize);
    let gy = m.div_ceil(tile.bm() as usize);
    if gy > MAX_GRID_Y {
        return Err(bad(format!("grid.y が上限 {MAX_GRID_Y} を超えます: {gy}")));
    }
    // grid*block が u32 に収まること（HIP の制約）。
    let total_x = (gx as u64) * u64::from(tile.threads_x());
    if total_x > u64::from(u32::MAX) {
        return Err(bad("grid.x*block.x が u32 を超えます".into()));
    }
    let gx = u32::try_from(gx).map_err(|_| bad("grid.x が u32 を超えます".into()))?;
    let gy = gy as u32; // MAX_GRID_Y 以下
    Ok(((gx, gy, 1), (tile.threads_x(), tile.threads_y(), 1)))
}

/// コンパイル済みの f32 GEMM カーネルを保持し、実行する。
pub struct RocmGemmF32 {
    tile: GemmF32Tile,
    // Drop 順: module（アンロード）→ rt。module が `Arc` を持つため順序に依らず安全。
    module: Module,
    rt: Arc<HipRuntime>,
    /// module・バッファ・同期の対象デバイス ID（`new` が選択したもの）。
    device: i32,
}

impl RocmGemmF32 {
    /// HIP/HIPRTC をロードし、既定タイルのカーネルを `target` 向けにコンパイル・ロードする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice` を返す（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let rt = Arc::new(hip::load()?);
        let hiprtc = HiprtcRuntime::load()?;
        if rt.device_count()? < 1 {
            return Err(RocmError::NoDevice);
        }
        let device = 0;
        let (tile, module) = {
            // `new` 中も呼び出し元の選択を壊さないようガードで復元する。
            let _guard = rt.select_device(device)?;
            let tile = GemmF32Tile::DEFAULT;
            let src = render_gemm_f32_source(&tile);
            let code = compile_code_object(&hiprtc, &src, target)?;
            let module = Module::load(Arc::clone(&rt), code, device)?;
            (tile, module)
        };
        Ok(Self {
            tile,
            module,
            rt,
            device,
        })
    }

    /// `C[m×n] = A[m×k]·B[k×n]`（行優先）を計算して返す。
    pub fn run(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, RocmError> {
        validate_gemm_dims(&self.tile, a.len(), b.len(), m, n, k)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let (grid, block) = launch_geometry(&self.tile, m, n)?;
        // module の所有デバイスを選び、バッファ確保・起動・同期を同一デバイスで行う。
        // 先に宣言するためバッファの Drop（hipFree）後に呼び出し元の選択へ復元される。
        let _guard = self.rt.select_device(self.device)?;
        let func = self.module.function(GEMM_F32_KERNEL_NAME)?;
        let da = DeviceBuffer::new(Arc::clone(&self.rt), a.len())?;
        let db = DeviceBuffer::new(Arc::clone(&self.rt), b.len())?;
        let dc = DeviceBuffer::new(Arc::clone(&self.rt), m * n)?;
        da.upload(a)?;
        db.upload(b)?;

        // 検証済みのため i32 に収まる。
        let mut pa = da.as_device_ptr();
        let mut pb = db.as_device_ptr();
        let mut pc = dc.as_device_ptr();
        let mut im = m as i32;
        let mut in_ = n as i32;
        let mut ik = k as i32;
        let mut params: [*mut c_void; GEMM_F32_ARG_COUNT] = [
            (&mut pa as *mut *mut c_void).cast(),
            (&mut pb as *mut *mut c_void).cast(),
            (&mut pc as *mut *mut c_void).cast(),
            (&mut im as *mut i32).cast(),
            (&mut in_ as *mut i32).cast(),
            (&mut ik as *mut i32).cast(),
        ];
        // SAFETY: (U5) `params` の個数・型・順序はカーネル宣言 `(const float* a, const float* b,
        // float* c, int m, int n, int k)` と一致し（`GEMM_F32_ARG_COUNT` とソース署名をテストで
        // 固定）、指す変数は下の `synchronize` 完了まで本スコープで生存する。グリッドは
        // `div_ceil` で出力全体を覆い、範囲外アクセスはカーネル内の手動境界チェック（REQ-8）が防ぐ。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()?;

        let mut out = vec![0.0f32; m * n];
        dc.download(&mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: GemmF32Tile = GemmF32Tile::DEFAULT;
    const MAX: usize = i32::MAX as usize;

    #[test]
    fn accepts_ordinary_and_zero_dims() {
        assert!(validate_gemm_dims(&T, 6, 12, 2, 4, 3).is_ok());
        assert!(validate_gemm_dims(&T, 0, 0, 2, 4, 0).is_ok());
        assert!(validate_gemm_dims(&T, 0, 0, 0, 0, 0).is_ok());
    }

    #[test]
    fn rejects_length_mismatch() {
        assert!(validate_gemm_dims(&T, 5, 12, 2, 4, 3).is_err());
        assert!(validate_gemm_dims(&T, 6, 11, 2, 4, 3).is_err());
    }

    #[test]
    fn rejects_i32_overflow() {
        assert!(validate_gemm_dims(&T, 0, 0, MAX + 1, 1, 0).is_err());
        assert!(validate_gemm_dims(&T, 0, 0, 1, MAX + 1, 0).is_err());
        assert!(validate_gemm_dims(&T, 0, 0, 1, 1, MAX + 1).is_err());
        // 積が i32::MAX を超える。
        assert!(validate_gemm_dims(&T, 0, 0, 1 << 16, 1, (1 << 15) + 1).is_err());
        assert!(validate_gemm_dims(&T, 0, 0, usize::MAX, 2, 2).is_err());
    }

    #[test]
    fn rejects_k_beyond_tile_bound() {
        let limit = MAX - (T.bk() as usize - 1);
        assert!(validate_gemm_dims(&T, limit, limit, 1, 1, limit).is_ok());
        assert!(validate_gemm_dims(&T, limit + 1, limit + 1, 1, 1, limit + 1).is_err());
    }

    #[test]
    fn geometry_covers_output() {
        let (g, b) = launch_geometry(&T, 65, 1).unwrap();
        assert_eq!(g, (1, 2, 1));
        assert_eq!(b, (16, 16, 1));
        let (g, _) = launch_geometry(&T, 64, 129).unwrap();
        assert_eq!(g, (3, 1, 1));
    }

    #[test]
    fn geometry_rejects_huge_grid_y() {
        assert!(launch_geometry(&T, 64 * MAX_GRID_Y + 1, 1).is_err());
        assert!(launch_geometry(&T, 64 * MAX_GRID_Y, 1).is_ok());
    }
}
