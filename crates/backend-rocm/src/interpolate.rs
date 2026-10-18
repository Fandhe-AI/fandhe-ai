//! interpolate（nearest・bilinear）の ROCm 実行入口（`backend-cuda::interpolate` の移植。
//! イシュー #3439・R8）。
//!
//! 役割: [`RocmInterpolate`] が HIP/HIPRTC をロードして `kernels_interpolate` のソースを
//! コンパイルし、行優先 contiguous のホストスライスから結果の `Vec` を返す。`BackendOps::
//! interpolate`（`InterpolateMode` の振り分け・`interpolate_out_shape_for_mode` による形状
//! 再検査・`Nearest`／`Bilinear` 以外の `Unsupported`・`Tensor` 化）は `RocmBackendOps`
//! （#3431・R0）側の責務で、本層は 2 モードの実行だけを持つ。
//!
//! ホスト側で fail-closed に検証する（A04）: rank 一致・非空間軸の一致・空間軸の入力サイズ非 0
//! （0 だとカーネルのクランプが `-1` 添字になり範囲外読み出しになる）・`i32` 収容。bilinear の
//! `scale_h`／`scale_w` は呼び出し元から受けず本層で [`bilinear_scale`] により求める。

use crate::error::RocmError;
use crate::kernels_interpolate::{BILINEAR_SPEC, NEAREST_SPEC, render_interpolate_source};
use crate::launch::{Arg, KernelHost};
use crate::shape_util::{checked_numel, fit_numel_i32, row_major_strides_i32, shape_to_i32};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "interpolate";

fn bad(detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op: OP, detail }
}

/// bilinear の 1 軸分のスケール（`tensor-core::interpolate::bilinear_scale` の同一式の複製）。
///
/// align_corners なら `(in-1)/(out-1)`（out ≤ 1 は 0）、そうでなければ `in/out`。整数側で `-1`
/// してから `f32` へ変換する（`in` が 2^24 を超えても四隅一致が保たれる）。tensor-core は本クレートの
/// 通常依存ではない（R0 が導入）ため複製し、一致はテストで dev-dependency の原本と照合する。
pub(crate) fn bilinear_scale(in_size: usize, out_size: usize, align_corners: bool) -> f32 {
    if out_size == 0 {
        return 0.0;
    }
    if align_corners {
        if out_size > 1 {
            (in_size.saturating_sub(1) as f32) / ((out_size - 1) as f32)
        } else {
            0.0
        }
    } else {
        in_size as f32 / out_size as f32
    }
}

/// 検証結果（起動引数）。
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct InterpPlan {
    pub(crate) numel_out: usize,
    pub(crate) numel_i: i32,
    pub(crate) rank_i: i32,
}

/// 共通検証。`spatial_start` 以降の軸が空間軸（それより前は素通しで in/out 一致を要求する）。
fn validate_common(
    input_len: usize,
    in_shape: &[usize],
    out_shape: &[usize],
    spatial_start: usize,
) -> Result<InterpPlan, RocmError> {
    let rank = in_shape.len();
    if rank == 0 || out_shape.len() != rank {
        return Err(bad(format!(
            "rank 不正: input {rank}, output {}",
            out_shape.len()
        )));
    }
    if spatial_start >= rank {
        return Err(bad(format!(
            "spatial_start={spatial_start} は rank={rank} 未満が必要です"
        )));
    }
    if in_shape[..spatial_start] != out_shape[..spatial_start] {
        return Err(bad("非空間軸の in/out サイズが一致しません".into()));
    }
    let numel_in = checked_numel(OP, in_shape)?;
    let numel_out = checked_numel(OP, out_shape)?;
    if input_len != numel_in {
        return Err(bad(format!(
            "input の長さ不一致: 期待 {numel_in}, 実際 {input_len}"
        )));
    }
    if numel_out > 0 && in_shape[spatial_start..].contains(&0) {
        return Err(bad("空の空間軸から非空の出力は作れません".into()));
    }
    let numel_i = if numel_out == 0 {
        0
    } else {
        fit_numel_i32(OP, numel_out, "numel_out")?
    };
    Ok(InterpPlan {
        numel_out,
        numel_i,
        rank_i: fit_numel_i32(OP, rank, "rank")?,
    })
}

/// nearest の検証。
pub(crate) fn validate_nearest(
    input_len: usize,
    in_shape: &[usize],
    out_shape: &[usize],
    spatial_start: usize,
) -> Result<InterpPlan, RocmError> {
    validate_common(input_len, in_shape, out_shape, spatial_start)
}

/// bilinear の検証（空間軸は末尾 2 軸ちょうど）。
pub(crate) fn validate_bilinear(
    input_len: usize,
    in_shape: &[usize],
    out_shape: &[usize],
) -> Result<InterpPlan, RocmError> {
    let rank = in_shape.len();
    if rank < 2 {
        return Err(bad(format!("bilinear は rank 2 以上が必要です: {rank}")));
    }
    validate_common(input_len, in_shape, out_shape, rank - 2)
}

/// コンパイル済みの interpolate カーネルを保持し、実行する。
pub struct RocmInterpolate {
    host: KernelHost,
}

impl RocmInterpolate {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_interpolate_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// nearest（`src = min((c*in)/out, in-1)`）。`spatial_start` 以降の軸が空間軸。
    pub fn run_nearest_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        spatial_start: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = validate_nearest(input.len(), in_shape, out_shape, spatial_start)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        let spatial_start_i = fit_numel_i32(OP, spatial_start, "spatial_start")?;
        self.launch(
            &NEAREST_SPEC,
            input,
            in_shape,
            out_shape,
            &plan,
            &[Arg::I32(plan.rank_i), Arg::I32(spatial_start_i)],
            plan.numel_i,
        )
    }

    /// bilinear（末尾 2 軸。`f32::mul_add` と同じ明示 FMA。CPU 参照実装と bit 一致する設計）。
    pub fn run_bilinear_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        align_corners: bool,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = validate_bilinear(input.len(), in_shape, out_shape)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        let rank = in_shape.len();
        let scale_h = bilinear_scale(in_shape[rank - 2], out_shape[rank - 2], align_corners);
        let scale_w = bilinear_scale(in_shape[rank - 1], out_shape[rank - 1], align_corners);
        self.launch(
            &BILINEAR_SPEC,
            input,
            in_shape,
            out_shape,
            &plan,
            &[
                Arg::I32(plan.rank_i),
                Arg::F32(scale_h),
                Arg::F32(scale_w),
                Arg::I32(i32::from(align_corners)),
            ],
            plan.numel_i,
        )
    }

    /// 共通の確保・起動・読み戻し。`mid` は `rank` 以降〜`numel` 直前の引数。
    #[allow(clippy::too_many_arguments)] // 2 カーネルの共通部（バッファ 5 本 + 引数）を 1 か所へ集約する
    fn launch(
        &self,
        spec: &crate::launch::KernelSpec,
        input: &[f32],
        in_shape: &[usize],
        out_shape: &[usize],
        plan: &InterpPlan,
        mid: &[Arg],
        numel_i: i32,
    ) -> Result<Vec<f32>, RocmError> {
        let out_shape_i = shape_to_i32(OP, out_shape)?;
        let in_shape_i = shape_to_i32(OP, in_shape)?;
        let in_strides = row_major_strides_i32(OP, in_shape)?;

        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let dout_shape = TypedDeviceBuffer::<i32>::from_slice(rt(), &out_shape_i)?;
        let din_shape = TypedDeviceBuffer::<i32>::from_slice(rt(), &in_shape_i)?;
        let din_strides = TypedDeviceBuffer::<i32>::from_slice(rt(), &in_strides)?;
        let mut args = vec![
            Arg::Ptr(din.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::Ptr(dout_shape.as_device_ptr()),
            Arg::Ptr(din_shape.as_device_ptr()),
            Arg::Ptr(din_strides.as_device_ptr()),
        ];
        args.extend_from_slice(mid);
        args.push(Arg::I32(numel_i));
        self.host.run(OP, spec, plan.numel_out, &args)?;
        dout.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_validation() {
        let p = validate_nearest(2 * 3 * 4, &[2, 3, 4], &[2, 5, 8], 1).unwrap();
        assert_eq!((p.numel_out, p.rank_i), (80, 3));
        // 非空間軸の不一致・rank 不一致・spatial_start 範囲外・長さ不一致。
        assert!(validate_nearest(24, &[2, 3, 4], &[3, 5, 8], 1).is_err());
        assert!(validate_nearest(24, &[2, 3, 4], &[2, 5], 1).is_err());
        assert!(validate_nearest(24, &[2, 3, 4], &[2, 3, 4], 3).is_err());
        assert!(validate_nearest(23, &[2, 3, 4], &[2, 5, 8], 1).is_err());
        assert!(validate_nearest(0, &[], &[], 0).is_err());
        // 入力の空間軸が 0 で出力が非空は拒否、出力が空なら受理。
        assert!(validate_nearest(0, &[2, 0], &[2, 3], 1).is_err());
        assert_eq!(
            validate_nearest(0, &[2, 0], &[2, 0], 1).unwrap().numel_out,
            0
        );
    }

    #[test]
    fn bilinear_validation() {
        assert!(validate_bilinear(12, &[3, 4], &[6, 8]).is_ok());
        assert!(validate_bilinear(4, &[4], &[8]).is_err());
        // 先頭軸は素通し（一致が必要）。
        assert!(validate_bilinear(24, &[2, 3, 4], &[3, 6, 8]).is_err());
        assert!(validate_bilinear(0, &[1, 0, 4], &[1, 2, 4]).is_err());
        assert!(validate_bilinear(24, &[2, 3, 4], &[2, 6, 8]).is_ok());
    }

    #[test]
    fn i32_overflow_is_rejected() {
        let big = i32::MAX as usize;
        assert!(validate_nearest(2, &[1, 2], &[1, big + 1], 1).is_err());
        assert!(validate_nearest(2, &[1, 2], &[1, big], 1).is_ok());
    }

    #[test]
    fn bilinear_scale_matches_tensor_core_original() {
        for in_size in [0usize, 1, 2, 3, 7, 16, 255, 1000, 16_777_217] {
            for out_size in [0usize, 1, 2, 5, 8, 33, 1024] {
                for ac in [false, true] {
                    assert_eq!(
                        bilinear_scale(in_size, out_size, ac).to_bits(),
                        fandhe_ai_tensor_core::bilinear_scale(in_size, out_size, ac).to_bits(),
                        "in={in_size} out={out_size} ac={ac}"
                    );
                }
            }
        }
    }
}
