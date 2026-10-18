//! softmax 系の ROCm 実行入口（`backend-cuda::softmax`・`log_softmax_backward` の移植。
//! イシュー #3435・分解案 R4）。
//!
//! 役割: [`RocmSoftmax`]（`softmax`・`log_softmax`）と [`RocmLogSoftmaxBackward`] が HIP/HIPRTC を
//! ロードして `kernels_softmax`／`kernels_log_softmax_backward` のソースを `target` 向けに
//! コンパイルし、ホストのスライス（行優先 `[rows, cols]`）から同形の結果を返す。
//! `BackendOps::softmax`／`log_softmax`／`log_softmax_backward` への結線（最終軸判定・形状再検査・
//! `Tensor` 化）は `ops.rs`（`RocmBackendOps`）の責務で、本ファイルはスライス入出力のみ。
//!
//! 起動形は「1 block = 1 wave（幅は `WarpGeometry::for_target`）= 1 行」。forward は
//! `grid = min(rows, SOFTMAX_MAX_GRID)` でカーネル内 grid-stride が全行を覆い、backward は
//! `grid = rows`。ホスト側で `checked_mul`・長さ一致・`i32` 収容を起動前に検証する（カーネル側の
//! REQ-8 と二重化）。最終軸以外の `dim` は呼び出し側（`ops.rs`）が `Unsupported` を返す。

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::RocmError;
use crate::kernels_log_softmax_backward::{
    LOG_SOFTMAX_BACKWARD_SPEC, render_log_softmax_backward_source,
};
use crate::kernels_softmax::{
    LOG_SOFTMAX_SPEC, SOFTMAX_MAX_GRID, SOFTMAX_SPEC, render_softmax_source,
};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;
use crate::warp_geometry::WarpGeometry;

const OP_SOFTMAX: &str = "softmax";
const OP_LOG_SOFTMAX: &str = "log_softmax";
const OP_BACKWARD: &str = "log_softmax_backward";

/// 検証済みの行カーネル寸法（カーネル引数は `i32`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SoftmaxDims {
    pub(crate) rows: usize,
    pub(crate) numel: usize,
    pub(crate) rows_i: i32,
    pub(crate) cols_i: i32,
}

fn fit_i32(op: &'static str, value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

/// `len == rows*cols`（checked）と `rows`・`cols`・`numel` の `i32` 収容を検証する。
pub(crate) fn validate_softmax_dims(
    op: &'static str,
    len: usize,
    rows: usize,
    cols: usize,
) -> Result<SoftmaxDims, RocmError> {
    let numel = rows
        .checked_mul(cols)
        .ok_or_else(|| RocmError::InvalidOpArgs {
            op,
            detail: "rows*cols が usize を超えます".into(),
        })?;
    if len != numel {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("入力の長さ不一致: 期待 {numel}, 実際 {len}"),
        });
    }
    // カーネルは `long long` で添字を計算するが、引数は `int`。numel も i32 内に限る
    // （CUDA 版・scan.rs と同じ上限）。
    fit_i32(op, numel, "numel")?;
    Ok(SoftmaxDims {
        rows,
        numel,
        rows_i: fit_i32(op, rows, "rows")?,
        cols_i: fit_i32(op, cols, "cols")?,
    })
}

/// forward の grid.x: `min(rows, SOFTMAX_MAX_GRID)`（`rows >= 1`）。カーネル内 grid-stride が覆う。
pub(crate) fn softmax_grid(rows: usize) -> u32 {
    u32::try_from(rows.min(SOFTMAX_MAX_GRID as usize)).unwrap_or(SOFTMAX_MAX_GRID)
}

/// コンパイル済みの softmax／log_softmax カーネルを保持し、実行する。
pub struct RocmSoftmax {
    host: KernelHost,
    width: u32,
}

impl RocmSoftmax {
    /// HIP/HIPRTC をロードし softmax 系カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let geom = WarpGeometry::for_target(target);
        let src = render_softmax_source(geom)?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
            width: geom.width(),
        })
    }

    /// 行 softmax（最終軸。online `(m, l)`・`exp2f`）。
    pub fn run_softmax_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run(OP_SOFTMAX, &SOFTMAX_SPEC, x, rows, cols)
    }

    /// 行 log_softmax（`(x - m) - ln(l)`）。
    pub fn run_log_softmax_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run(OP_LOG_SOFTMAX, &LOG_SOFTMAX_SPEC, x, rows, cols)
    }

    fn run(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let d = validate_softmax_dims(op, x.len(), rows, cols)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let dx = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), x)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), d.numel)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.rows_i),
            Arg::I32(d.cols_i),
            Arg::F32(std::f32::consts::LOG2_E),
        ];
        self.host.run_geometry(
            op,
            spec,
            (softmax_grid(d.rows), 1, 1),
            (self.width, 1, 1),
            &args,
        )?;
        dout.to_vec()
    }
}

/// コンパイル済みの `log_softmax_backward` カーネルを保持し、実行する。
pub struct RocmLogSoftmaxBackward {
    host: KernelHost,
    width: u32,
}

impl RocmLogSoftmaxBackward {
    /// HIP/HIPRTC をロードし backward カーネルを `target` 向けにコンパイルする。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let geom = WarpGeometry::for_target(target);
        let src = render_log_softmax_backward_source(geom)?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
            width: geom.width(),
        })
    }

    /// `dx = g - exp(y) * Σ_dim(g)`（`Σ` は行ごと `f64` 逐次和）。`y`・`g` は `[rows, cols]`。
    pub fn run_log_softmax_backward_f32(
        &self,
        y: &[f32],
        g: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let d = validate_softmax_dims(OP_BACKWARD, y.len(), rows, cols)?;
        validate_softmax_dims(OP_BACKWARD, g.len(), rows, cols)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let dy = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), y)?;
        let dg = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), g)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), d.numel)?;
        let args = [
            Arg::Ptr(dy.as_device_ptr()),
            Arg::Ptr(dg.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.rows_i),
            Arg::I32(d.cols_i),
        ];
        // grid.x = rows（i32 収容検証済みで HIP の grid.x 上限内。CTA ガード `row >= rows` あり）。
        self.host.run_geometry(
            OP_BACKWARD,
            &LOG_SOFTMAX_BACKWARD_SPEC,
            (u32::try_from(d.rows).unwrap_or(u32::MAX), 1, 1),
            (self.width, 1, 1),
            &args,
        )?;
        dout.to_vec()
    }
}

type SoftmaxCache = RefCell<Vec<(GfxTarget, Rc<RocmSoftmax>)>>;
type BackwardCache = RefCell<Vec<(GfxTarget, Rc<RocmLogSoftmaxBackward>)>>;

thread_local! {
    /// target ごとにコンパイル済みの実行型を遅延保持する（`ops.rs::cached_gemm` と同型。失敗は
    /// キャッシュしない。生ポインタを持つ実行型は `!Send` のため thread_local に置く）。
    static SOFTMAX_CACHE: SoftmaxCache = const { RefCell::new(Vec::new()) };
    static BACKWARD_CACHE: BackwardCache = const { RefCell::new(Vec::new()) };
}

fn cache_destroyed() -> RocmError {
    RocmError::InvalidKernelConfig {
        detail: "softmax キャッシュ（thread_local）が破棄済みです".into(),
    }
}

/// `target` 用の [`RocmSoftmax`] を取得する（無ければ構築して入れる）。`ops.rs` から呼ばれる。
pub(crate) fn cached_softmax(target: GfxTarget) -> Result<Rc<RocmSoftmax>, RocmError> {
    let hit = SOFTMAX_CACHE
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, v)| Rc::clone(v))
        })
        .map_err(|_| cache_destroyed())?;
    if let Some(v) = hit {
        return Ok(v);
    }
    // 借用を解いてから構築する（再入しても二重借用にならない）。
    let v = Rc::new(RocmSoftmax::new(target)?);
    let _ = SOFTMAX_CACHE.try_with(|c| c.borrow_mut().push((target, Rc::clone(&v))));
    Ok(v)
}

/// `target` 用の [`RocmLogSoftmaxBackward`] を取得する（無ければ構築して入れる）。
pub(crate) fn cached_log_softmax_backward(
    target: GfxTarget,
) -> Result<Rc<RocmLogSoftmaxBackward>, RocmError> {
    let hit = BACKWARD_CACHE
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, v)| Rc::clone(v))
        })
        .map_err(|_| cache_destroyed())?;
    if let Some(v) = hit {
        return Ok(v);
    }
    let v = Rc::new(RocmLogSoftmaxBackward::new(target)?);
    let _ = BACKWARD_CACHE.try_with(|c| c.borrow_mut().push((target, Rc::clone(&v))));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ordinary_and_empty() {
        let d = validate_softmax_dims("t", 12, 3, 4).unwrap();
        assert_eq!((d.rows, d.numel, d.rows_i, d.cols_i), (3, 12, 3, 4));
        assert_eq!(validate_softmax_dims("t", 0, 0, 4).unwrap().numel, 0);
        assert_eq!(validate_softmax_dims("t", 0, 3, 0).unwrap().numel, 0);
    }

    #[test]
    fn rejects_length_mismatch_and_overflow() {
        assert!(validate_softmax_dims("t", 11, 3, 4).is_err());
        assert!(validate_softmax_dims("t", 0, usize::MAX, 2).is_err());
        let big = i32::MAX as usize + 1;
        assert!(matches!(
            validate_softmax_dims("t", big, big, 1),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            validate_softmax_dims("t", big, 1, big),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }

    #[test]
    fn grid_is_clamped_and_nonzero() {
        assert_eq!(softmax_grid(1), 1);
        assert_eq!(softmax_grid(7), 7);
        assert_eq!(softmax_grid(65535), 65535);
        assert_eq!(softmax_grid(65536), 65535);
        assert_eq!(softmax_grid(i32::MAX as usize), 65535);
    }
}
