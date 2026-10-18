//! gather／scatter／one_hot の ROCm 実行入口（`backend-cuda::gather_scatter` の移植。
//! イシュー #3439・R8）。
//!
//! 役割: [`RocmGatherScatter`] が HIP/HIPRTC をロードして `kernels_gather_scatter` のソースを
//! コンパイルし、ホストのスライス（行優先・contiguous）から結果の `Vec` を返す。
//! `BackendOps::gather`／`scatter`／`one_hot` への結線（`gather_out_shape` 等による形状再検査・
//! `Tensor` 化・`ScatterReduce` との対応付け）は `RocmBackendOps`（#3431・R0）側の責務。
//!
//! 本層でもホスト側で fail-closed に検証してから起動する（A04／A08）。形状の整合（rank・dim・
//! 非 `dim` 軸の大小）・スライス長・`i32` 収容に加え、index の値域を**全要素走査**して範囲外を
//! 拒否する（カーネルの範囲外フォールバックが `0.0` を書いて `Ok` を返す経路を塞ぐ）。

use crate::error::RocmError;
use crate::kernels_gather_scatter::{
    GATHER_SPEC, ONE_HOT_SPEC, SCATTER_ADD_SPEC, SCATTER_OVERWRITE_SPEC,
    render_gather_scatter_source,
};
use crate::launch::{Arg, KernelHost};
use crate::shape_util::{checked_numel, fit_numel_i32, row_major_strides_i32, shape_to_i32};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

/// scatter の集約方式（`ScatterReduce` と同じ 2 値。対応付けは R0 の結線層で行う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterMode {
    /// 上書き（同一位置への複数書き込みは index の走査順で最後の値が残る）。
    Overwrite,
    /// 加算（`f64` アキュムレータ・逐次加算・最後に 1 回 `f32` へ downcast）。
    Add,
}

fn bad(op: &'static str, detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op, detail }
}

/// rank・dim・非 `dim` 軸の大小（`idx_s <= in_s`）を検証する。
fn validate_axes(
    op: &'static str,
    in_shape: &[usize],
    idx_shape: &[usize],
    dim: usize,
) -> Result<(), RocmError> {
    let rank = in_shape.len();
    if rank == 0 {
        return Err(bad(op, "rank 0 は未対応です".into()));
    }
    if idx_shape.len() != rank {
        return Err(bad(
            op,
            format!("rank 不一致: input {rank}, index {}", idx_shape.len()),
        ));
    }
    if dim >= rank {
        return Err(bad(op, format!("dim={dim} が rank={rank} 以上です")));
    }
    for a in (0..rank).filter(|&a| a != dim) {
        if idx_shape[a] > in_shape[a] {
            return Err(bad(
                op,
                format!(
                    "軸 {a} で index サイズ {} が input サイズ {} を超えます",
                    idx_shape[a], in_shape[a]
                ),
            ));
        }
    }
    Ok(())
}

/// index の全要素が `[0, limit)` にあることを検証する（A08）。
fn validate_index_range(op: &'static str, index: &[i32], limit: usize) -> Result<(), RocmError> {
    for (pos, &v) in index.iter().enumerate() {
        let in_range = usize::try_from(v).is_ok_and(|u| u < limit);
        if !in_range {
            return Err(bad(
                op,
                format!("index[{pos}]={v} が範囲 [0, {limit}) の外です"),
            ));
        }
    }
    Ok(())
}

/// gather の検証結果。
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct GatherPlan {
    pub(crate) numel_out: usize,
    pub(crate) numel_i: i32,
    pub(crate) in_dim_size_i: i32,
}

/// gather の引数検証（`index_shape` は出力 shape と同一）。
pub(crate) fn validate_gather(
    input_len: usize,
    index: &[i32],
    in_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> Result<GatherPlan, RocmError> {
    const OP: &str = "gather";
    validate_axes(OP, in_shape, index_shape, dim)?;
    // 出力（= index_shape）が空なら入力 shape の積を取る前に短絡する（CUDA 原型と同順。
    // 末尾 0 軸 + 巨大な先頭軸で積が overflow しても空 Vec を返す）。
    if index_shape.contains(&0) {
        return Ok(GatherPlan {
            numel_out: 0,
            numel_i: 0,
            in_dim_size_i: 0,
        });
    }
    let numel_in = checked_numel(OP, in_shape)?;
    let numel_out = checked_numel(OP, index_shape)?;
    if input_len != numel_in {
        return Err(bad(
            OP,
            format!("input の長さ不一致: 期待 {numel_in}, 実際 {input_len}"),
        ));
    }
    if index.len() != numel_out {
        return Err(bad(
            OP,
            format!("index の長さ不一致: 期待 {numel_out}, 実際 {}", index.len()),
        ));
    }
    let numel_i = fit_numel_i32(OP, numel_out, "numel_out")?;
    let in_dim_size_i = fit_numel_i32(OP, in_shape[dim], "in_dim_size")?;
    validate_index_range(OP, index, in_shape[dim])?;
    Ok(GatherPlan {
        numel_out,
        numel_i,
        in_dim_size_i,
    })
}

/// scatter の検証結果。
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ScatterPlan {
    pub(crate) numel_out: usize,
    pub(crate) numel_index: usize,
    pub(crate) numel_i: i32,
}

/// scatter の引数検証（出力 shape は `in_shape` と同一）。
pub(crate) fn validate_scatter(
    input_len: usize,
    index: &[i32],
    src_len: usize,
    in_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) -> Result<ScatterPlan, RocmError> {
    const OP: &str = "scatter";
    validate_axes(OP, in_shape, index_shape, dim)?;
    // 出力（= in_shape）が空なら積を取る前に短絡する（CUDA 原型と同順。overflow 回避）。
    if in_shape.contains(&0) {
        return Ok(ScatterPlan {
            numel_out: 0,
            numel_index: 0,
            numel_i: 0,
        });
    }
    let numel_out = checked_numel(OP, in_shape)?;
    let numel_index = checked_numel(OP, index_shape)?;
    if input_len != numel_out {
        return Err(bad(
            OP,
            format!("input の長さ不一致: 期待 {numel_out}, 実際 {input_len}"),
        ));
    }
    if index.len() != numel_index || src_len != numel_index {
        return Err(bad(
            OP,
            format!(
                "index／src の長さ不一致: 期待 {numel_index}, index {}, src {src_len}",
                index.len()
            ),
        ));
    }
    let numel_i = fit_numel_i32(OP, numel_out, "numel_out")?;
    validate_index_range(OP, index, in_shape[dim])?;
    Ok(ScatterPlan {
        numel_out,
        numel_index,
        numel_i,
    })
}

/// one_hot の検証結果。
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct OneHotPlan {
    pub(crate) numel_out: usize,
    pub(crate) numel_i: i32,
    pub(crate) num_classes_i: i32,
}

/// one_hot の引数検証（出力は `index.len() * num_classes` 要素）。
pub(crate) fn validate_one_hot(index: &[i32], num_classes: usize) -> Result<OneHotPlan, RocmError> {
    const OP: &str = "one_hot";
    let numel_out = index
        .len()
        .checked_mul(num_classes)
        .ok_or(RocmError::SizeLimitExceeded {
            op: OP,
            n: index.len(),
            limit: usize::MAX,
        })?;
    if numel_out == 0 {
        return Ok(OneHotPlan {
            numel_out: 0,
            numel_i: 0,
            num_classes_i: 0,
        });
    }
    let numel_i = fit_numel_i32(OP, numel_out, "numel_out")?;
    let num_classes_i = fit_numel_i32(OP, num_classes, "num_classes")?;
    validate_index_range(OP, index, num_classes)?;
    Ok(OneHotPlan {
        numel_out,
        numel_i,
        num_classes_i,
    })
}

/// コンパイル済みの gather／scatter／one_hot カーネルを保持し、実行する。
pub struct RocmGatherScatter {
    host: KernelHost,
}

impl RocmGatherScatter {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_gather_scatter_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// `out[i] = input[... dim 軸の座標を index[i] に置換 ...]`。出力 shape は `index_shape`。
    pub fn run_gather_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        index: &[i32],
        index_shape: &[usize],
        dim: usize,
    ) -> Result<Vec<f32>, RocmError> {
        const OP: &str = "gather";
        let plan = validate_gather(input.len(), index, in_shape, index_shape, dim)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        let out_shape = shape_to_i32(OP, index_shape)?;
        let in_strides = row_major_strides_i32(OP, in_shape)?;
        let rank_i = fit_numel_i32(OP, in_shape.len(), "rank")?;
        let dim_i = fit_numel_i32(OP, dim, "dim")?;

        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let didx = TypedDeviceBuffer::<i32>::from_slice(rt(), index)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let dshape = TypedDeviceBuffer::<i32>::from_slice(rt(), &out_shape)?;
        let dstrides = TypedDeviceBuffer::<i32>::from_slice(rt(), &in_strides)?;
        let args = [
            Arg::Ptr(din.as_device_ptr()),
            Arg::Ptr(didx.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::Ptr(dshape.as_device_ptr()),
            Arg::Ptr(dstrides.as_device_ptr()),
            Arg::I32(rank_i),
            Arg::I32(dim_i),
            Arg::I32(plan.numel_i),
            Arg::I32(plan.in_dim_size_i),
        ];
        self.host.run(OP, &GATHER_SPEC, plan.numel_out, &args)?;
        dout.to_vec()
    }

    /// `out = input` のうち、`index` が指す位置へ `src` を上書き／加算する。出力 shape は `in_shape`。
    #[allow(clippy::too_many_arguments)] // scatter の意味論（input／index／src／shape 2 種／dim／mode）を 1 関数に集約する（backend-cuda::run_scatter_f32 と同形）
    pub fn run_scatter_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        index: &[i32],
        index_shape: &[usize],
        src: &[f32],
        dim: usize,
        mode: ScatterMode,
    ) -> Result<Vec<f32>, RocmError> {
        const OP: &str = "scatter";
        let plan = validate_scatter(input.len(), index, src.len(), in_shape, index_shape, dim)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        if plan.numel_index == 0 {
            return Ok(input.to_vec());
        }
        let out_shape = shape_to_i32(OP, in_shape)?;
        let index_shape_i = shape_to_i32(OP, index_shape)?;
        let index_strides = row_major_strides_i32(OP, index_shape)?;
        let rank_i = fit_numel_i32(OP, in_shape.len(), "rank")?;
        let dim_i = fit_numel_i32(OP, dim, "dim")?;
        let spec = match mode {
            ScatterMode::Overwrite => &SCATTER_OVERWRITE_SPEC,
            ScatterMode::Add => &SCATTER_ADD_SPEC,
        };

        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let dinput = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dindex = TypedDeviceBuffer::<i32>::from_slice(rt(), index)?;
        let dsrc = TypedDeviceBuffer::<f32>::from_slice(rt(), src)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let dout_shape = TypedDeviceBuffer::<i32>::from_slice(rt(), &out_shape)?;
        let dindex_shape = TypedDeviceBuffer::<i32>::from_slice(rt(), &index_shape_i)?;
        let dindex_strides = TypedDeviceBuffer::<i32>::from_slice(rt(), &index_strides)?;
        let args = [
            Arg::Ptr(dinput.as_device_ptr()),
            Arg::Ptr(dindex.as_device_ptr()),
            Arg::Ptr(dsrc.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::Ptr(dout_shape.as_device_ptr()),
            Arg::Ptr(dindex_shape.as_device_ptr()),
            Arg::Ptr(dindex_strides.as_device_ptr()),
            Arg::I32(rank_i),
            Arg::I32(dim_i),
            Arg::I32(plan.numel_i),
        ];
        self.host.run(OP, spec, plan.numel_out, &args)?;
        dout.to_vec()
    }

    /// `out[r, c] = (index[r] == c) ? 1.0 : 0.0`（`index.len() * num_classes` 要素）。
    pub fn run_one_hot_f32(
        &self,
        index: &[i32],
        num_classes: usize,
    ) -> Result<Vec<f32>, RocmError> {
        const OP: &str = "one_hot";
        let plan = validate_one_hot(index, num_classes)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let dindex = TypedDeviceBuffer::<i32>::from_slice(rt(), index)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let args = [
            Arg::Ptr(dindex.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(plan.num_classes_i),
            Arg::I32(plan.numel_i),
        ];
        self.host.run(OP, &ONE_HOT_SPEC, plan.numel_out, &args)?;
        dout.to_vec()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_output_short_circuits_before_overflowing_input_numel() {
        let big = usize::MAX / 2 + 1;
        let g = validate_gather(0, &[], &[big, 4, 0], &[big, 4, 0], 2).unwrap();
        assert_eq!(g.numel_out, 0);
        let s = validate_scatter(0, &[], 0, &[big, 4, 0], &[big, 4, 0], 2).unwrap();
        assert_eq!(s.numel_out, 0);
    }

    use super::*;

    #[test]
    fn gather_accepts_valid_and_rejects_bad_axes() {
        let p = validate_gather(6, &[0, 1, 1, 0], &[2, 3], &[2, 2], 1).unwrap();
        assert_eq!((p.numel_out, p.in_dim_size_i), (4, 3));
        // 非 dim 軸が input より大きい。
        assert!(validate_gather(6, &[0; 12], &[2, 3], &[4, 3], 1).is_err());
        // rank 不一致・dim 範囲外・rank 0。
        assert!(validate_gather(6, &[0; 2], &[2, 3], &[2], 0).is_err());
        assert!(validate_gather(6, &[0; 4], &[2, 3], &[2, 2], 2).is_err());
        assert!(validate_gather(1, &[], &[], &[], 0).is_err());
    }

    #[test]
    fn gather_rejects_length_mismatch_and_out_of_range_index() {
        assert!(validate_gather(5, &[0, 0, 0, 0], &[2, 3], &[2, 2], 1).is_err());
        assert!(validate_gather(6, &[0, 0, 0], &[2, 3], &[2, 2], 1).is_err());
        assert!(validate_gather(6, &[0, 3, 0, 0], &[2, 3], &[2, 2], 1).is_err());
        assert!(validate_gather(6, &[0, -1, 0, 0], &[2, 3], &[2, 2], 1).is_err());
        // 空の出力は起動不要として受理される。
        assert_eq!(
            validate_gather(6, &[], &[2, 3], &[2, 0], 1)
                .unwrap()
                .numel_out,
            0
        );
    }

    #[test]
    fn scatter_validation() {
        let p = validate_scatter(6, &[0, 1], 2, &[2, 3], &[1, 2], 1).unwrap();
        assert_eq!((p.numel_out, p.numel_index), (6, 2));
        assert!(validate_scatter(6, &[0, 3], 2, &[2, 3], &[1, 2], 1).is_err());
        assert!(validate_scatter(6, &[0, 1], 1, &[2, 3], &[1, 2], 1).is_err());
        assert!(validate_scatter(6, &[0, 1], 2, &[2, 3], &[3, 2], 1).is_err());
        assert!(validate_scatter(5, &[0, 1], 2, &[2, 3], &[1, 2], 1).is_err());
        // dim 軸は index 側が input より長くてもよい（重複 index）。
        assert!(validate_scatter(6, &[0, 1, 2, 0, 1, 2, 0, 1], 8, &[2, 3], &[2, 4], 1).is_ok());
    }

    #[test]
    fn one_hot_validation() {
        let p = validate_one_hot(&[0, 2, 1], 3).unwrap();
        assert_eq!((p.numel_out, p.num_classes_i), (9, 3));
        assert!(validate_one_hot(&[0, 3], 3).is_err());
        assert!(validate_one_hot(&[-1], 3).is_err());
        assert_eq!(validate_one_hot(&[], 5).unwrap().numel_out, 0);
        assert_eq!(validate_one_hot(&[1], 0).unwrap().numel_out, 0);
        assert!(validate_one_hot(&[0; 4], usize::MAX).is_err());
        assert!(validate_one_hot(&[0; 2], i32::MAX as usize).is_err());
    }
}
