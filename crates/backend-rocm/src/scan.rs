//! 累積和・累積積の ROCm 実行入口（`backend-cuda::scan` の移植。イシュー #3439・R8）。
//!
//! 役割: [`RocmScan`] が HIP/HIPRTC をロードして `kernels_scan` のソースをコンパイルし、
//! `run_cumsum_f32`／`run_cumprod_f32` でホストのスライス `x`（`[outer, axis_len, inner]` の
//! 行優先）から同形の結果を返す。`BackendOps::cumsum`／`cumprod` への結線（形状再検査・
//! `Tensor` 化）は `RocmBackendOps`（#3431・R0）側の責務で、本型はスライス入出力のみ。
//! ホスト側で `i32` 範囲・要素数の checked 演算を検証してから起動する（カーネル側 REQ-8 と二重化）。

use crate::error::RocmError;
use crate::kernels_scan::{CUMPROD_SPEC, CUMSUM_SPEC, render_scan_source};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "scan";

/// 検証済みの scan 寸法（カーネル引数は `i32`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScanDims {
    pub(crate) lanes: usize,
    pub(crate) numel: usize,
    pub(crate) lanes_i: i32,
    pub(crate) axis_len_i: i32,
    pub(crate) inner_i: i32,
}

fn fit_i32(value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op: OP,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

/// `x.len() == outer*axis_len*inner` と各引数の `i32` 収容を検証する。
pub(crate) fn validate_scan_dims(
    x_len: usize,
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Result<ScanDims, RocmError> {
    let bad = |detail: &str| RocmError::InvalidOpArgs {
        op: OP,
        detail: detail.into(),
    };
    let lanes = outer
        .checked_mul(inner)
        .ok_or_else(|| bad("outer*inner が usize を超えます"))?;
    let numel = lanes
        .checked_mul(axis_len)
        .ok_or_else(|| bad("lanes*axis_len が usize を超えます"))?;
    if x_len != numel {
        return Err(RocmError::InvalidOpArgs {
            op: OP,
            detail: format!("x の長さ不一致: 期待 {numel}, 実際 {x_len}"),
        });
    }
    // カーネルは `long long` で添字を計算するが、引数は `int`。numel も i32 内に限る
    // （CUDA 版と同じ上限。バッファ・Metal との一貫性）。
    fit_i32(numel, "numel")?;
    Ok(ScanDims {
        lanes,
        numel,
        lanes_i: fit_i32(lanes, "lanes")?,
        axis_len_i: fit_i32(axis_len, "axis_len")?,
        inner_i: fit_i32(inner, "inner")?,
    })
}

/// コンパイル済みの scan カーネルを保持し、実行する。
pub struct RocmScan {
    host: KernelHost,
}

impl RocmScan {
    /// HIP/HIPRTC をロードし scan カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_scan_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// 累積和（`f64` アキュムレータ・`dim` 昇順逐次）。
    pub fn run_cumsum_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run(&CUMSUM_SPEC, x, outer, axis_len, inner)
    }

    /// 累積積（`f64` アキュムレータ・`dim` 昇順逐次）。
    pub fn run_cumprod_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run(&CUMPROD_SPEC, x, outer, axis_len, inner)
    }

    fn run(
        &self,
        spec: &KernelSpec,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let d = validate_scan_dims(x.len(), outer, axis_len, inner)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let dx = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), x)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), d.numel)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.lanes_i),
            Arg::I32(d.axis_len_i),
            Arg::I32(d.inner_i),
        ];
        self.host.run(OP, spec, d.lanes, &args)?;
        dout.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ordinary_and_empty() {
        let d = validate_scan_dims(24, 2, 4, 3).unwrap();
        assert_eq!((d.lanes, d.numel), (6, 24));
        assert_eq!(validate_scan_dims(0, 2, 0, 3).unwrap().numel, 0);
        assert_eq!(validate_scan_dims(0, 0, 4, 3).unwrap().lanes, 0);
    }

    #[test]
    fn rejects_length_mismatch_and_overflow() {
        assert!(validate_scan_dims(23, 2, 4, 3).is_err());
        assert!(validate_scan_dims(0, usize::MAX, 1, 2).is_err());
        assert!(validate_scan_dims(0, 1 << 20, 1 << 20, 1).is_err());
        let max = i32::MAX as usize;
        assert!(matches!(
            validate_scan_dims(0, 1, max + 1, 0),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}
