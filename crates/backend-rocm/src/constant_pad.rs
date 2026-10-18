//! 定数 pad の ROCm 実行入口（`backend-cuda::constant_pad` の移植。イシュー #3438・R7）。
//!
//! 役割: [`RocmConstantPad`] が HIP/HIPRTC をロードして `kernels_constant_pad` のソースをコンパイルし、
//! 行優先 contiguous のホストスライスから結果の `Vec` を返す。`BackendOps::pad` の shape 再検査
//! （`pad_out_shape`）・`Unsupported` の事前確定・`Tensor` 化は `RocmBackendOps`（`ops/conv_pool.rs`）側の
//! 責務で、本層は出力 shape の導出・検証と実行だけを持つ。
//!
//! 検証は HIP に触れない純関数 [`validate_pad`] に集約する（A04）: `pads.len() == rank`・
//! `out = in + before + after` の checked 計算・スライス長・`numel_out` の `i32` 収容（超過は
//! `SizeLimitExceeded`。`ops` が `Unsupported` へ写す）。出力が空なら入力 shape の積を取る前に短絡し、
//! 入力が空で出力が非空なら `value` で埋めて返す（デバイス不要）。`numel_out` が `i32` に収まれば
//! 各軸サイズ・`before`・入力ストライドもすべて `i32` に収まる（各軸 ≤ 出力軸、積 ≤ numel_out）。

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::RocmError;
use crate::im2col::fit_i32_limit;
use crate::kernels_constant_pad::{PAD_SPEC, render_constant_pad_source};
use crate::launch::{Arg, KernelHost};
use crate::shape_util::{checked_numel, row_major_strides_i32, shape_to_i32};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "pad";

fn bad(detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op: OP, detail }
}

/// 検証結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PadPlan {
    /// 出力 shape（`in + before + after`）。
    pub(crate) out_shape: Vec<usize>,
    /// 出力要素数（0 なら起動しない）。
    pub(crate) numel_out: usize,
    /// 入力のいずれかの軸が 0（出力が非空なら全要素が `value`）。
    pub(crate) input_is_empty: bool,
}

/// pad の shape 検証（純関数。入力スライスの長さは見ない）。出力 shape・要素数・`i32` 収容を確定する。
pub(crate) fn plan_pad(in_shape: &[usize], pads: &[(usize, usize)]) -> Result<PadPlan, RocmError> {
    let rank = in_shape.len();
    if pads.len() != rank {
        return Err(bad(format!(
            "pads の長さ不一致: rank {rank}, pads {}",
            pads.len()
        )));
    }
    let mut out_shape = Vec::with_capacity(rank);
    for (a, (&dim, &(before, after))) in in_shape.iter().zip(pads).enumerate() {
        let out = dim
            .checked_add(before)
            .and_then(|v| v.checked_add(after))
            .ok_or_else(|| bad(format!("軸 {a} の出力サイズが usize を超えます")))?;
        out_shape.push(out);
    }
    let input_is_empty = in_shape.contains(&0);
    // 出力が空なら積を取る前に短絡する（0 軸より前の巨大軸で積が overflow しても拒否しない）。
    if out_shape.contains(&0) {
        return Ok(PadPlan {
            out_shape,
            numel_out: 0,
            input_is_empty,
        });
    }
    let numel_out = checked_numel(OP, &out_shape)?;
    fit_i32_limit(OP, numel_out)?;
    Ok(PadPlan {
        out_shape,
        numel_out,
        input_is_empty,
    })
}

/// pad の検証（純関数）。[`plan_pad`] に加え、入力が非空なら `input_len` が入力要素数と一致することを見る。
pub(crate) fn validate_pad(
    input_len: usize,
    in_shape: &[usize],
    pads: &[(usize, usize)],
) -> Result<PadPlan, RocmError> {
    let plan = plan_pad(in_shape, pads)?;
    if plan.numel_out > 0 && !plan.input_is_empty {
        let numel_in = checked_numel(OP, in_shape)?;
        if input_len != numel_in {
            return Err(bad(format!(
                "input の長さ不一致: 期待 {numel_in}, 実際 {input_len}"
            )));
        }
    }
    Ok(plan)
}

/// コンパイル済みの pad カーネルを保持し、実行する。
pub struct RocmConstantPad {
    host: KernelHost,
}

impl RocmConstantPad {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_constant_pad_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// 定数 pad。`pads[a] = (before, after)`。出力 shape は `in_shape[a] + before + after`。
    pub fn run_pad_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        pads: &[(usize, usize)],
        value: f32,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = validate_pad(input.len(), in_shape, pads)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        if in_shape.is_empty() {
            // rank 0: 出力は 1 要素で、入力もちょうど 1 要素（CUDA 版と同じ）。
            return match input {
                [v] => Ok(vec![*v]),
                _ => Err(bad(format!(
                    "rank 0 の入力は 1 要素が必要です: {}",
                    input.len()
                ))),
            };
        }
        if plan.input_is_empty {
            return Ok(vec![value; plan.numel_out]);
        }
        let out_shape_i = shape_to_i32(OP, &plan.out_shape)?;
        let in_shape_i = shape_to_i32(OP, in_shape)?;
        let in_strides = row_major_strides_i32(OP, in_shape)?;
        let before_i = pads
            .iter()
            .map(|&(b, _)| fit_i32_limit(OP, b))
            .collect::<Result<Vec<_>, _>>()?;
        let rank_i = fit_i32_limit(OP, in_shape.len())?;
        let numel_i = fit_i32_limit(OP, plan.numel_out)?;

        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let dout_shape = TypedDeviceBuffer::<i32>::from_slice(rt(), &out_shape_i)?;
        let din_shape = TypedDeviceBuffer::<i32>::from_slice(rt(), &in_shape_i)?;
        let dbefore = TypedDeviceBuffer::<i32>::from_slice(rt(), &before_i)?;
        let dstrides = TypedDeviceBuffer::<i32>::from_slice(rt(), &in_strides)?;
        let args = [
            Arg::Ptr(din.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::Ptr(dout_shape.as_device_ptr()),
            Arg::Ptr(din_shape.as_device_ptr()),
            Arg::Ptr(dbefore.as_device_ptr()),
            Arg::Ptr(dstrides.as_device_ptr()),
            Arg::I32(rank_i),
            Arg::I32(numel_i),
            Arg::F32(value),
        ];
        self.host.run(OP, &PAD_SPEC, plan.numel_out, &args)?;
        dout.to_vec()
    }
}

type PadCache = RefCell<Vec<(GfxTarget, Rc<RocmConstantPad>)>>;

thread_local! {
    /// target ごとにコンパイル済みの実行型を遅延保持する（`softmax.rs` と同型。失敗はキャッシュしない）。
    static PAD_CACHE: PadCache = const { RefCell::new(Vec::new()) };
}

/// `target` 用の [`RocmConstantPad`] を取得する（無ければ構築して入れる）。`ops/conv_pool.rs` から呼ばれる。
pub(crate) fn cached_constant_pad(target: GfxTarget) -> Result<Rc<RocmConstantPad>, RocmError> {
    let hit = PAD_CACHE
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, v)| Rc::clone(v))
        })
        .map_err(|_| RocmError::InvalidKernelConfig {
            detail: "pad キャッシュ（thread_local）が破棄済みです".into(),
        })?;
    if let Some(v) = hit {
        return Ok(v);
    }
    // 借用を解いてから構築する（再入しても二重借用にならない）。
    let v = Rc::new(RocmConstantPad::new(target)?);
    let _ = PAD_CACHE.try_with(|c| c.borrow_mut().push((target, Rc::clone(&v))));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_shapes_and_lengths() {
        let p = validate_pad(6, &[2, 3], &[(1, 0), (0, 2)]).unwrap();
        assert_eq!(p.out_shape, [3, 5]);
        assert_eq!(p.numel_out, 15);
        assert!(!p.input_is_empty);
        // rank・長さの不一致。
        assert!(validate_pad(6, &[2, 3], &[(1, 0)]).is_err());
        assert!(validate_pad(5, &[2, 3], &[(0, 0), (0, 0)]).is_err());
        // rank 0。
        let p = validate_pad(1, &[], &[]).unwrap();
        assert_eq!((p.numel_out, p.out_shape.len()), (1, 0));
    }

    #[test]
    fn empty_input_with_nonempty_output_is_flagged() {
        let p = validate_pad(0, &[0, 3], &[(2, 1), (0, 0)]).unwrap();
        assert!(p.input_is_empty);
        assert_eq!(p.numel_out, 9);
    }

    #[test]
    fn empty_output_short_circuits_before_overflowing_input_numel() {
        let big = usize::MAX / 2 + 1;
        let p = validate_pad(0, &[big, 4, 0], &[(0, 0), (0, 0), (0, 0)]).unwrap();
        assert_eq!(p.numel_out, 0);
    }

    #[test]
    fn overflow_and_i32_limit() {
        assert!(matches!(
            validate_pad(0, &[usize::MAX], &[(1, 0)]),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        let big = i32::MAX as usize;
        assert!(matches!(
            validate_pad(0, &[0], &[(big, 1)]),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        assert!(validate_pad(0, &[0], &[(big, 0)]).is_ok());
    }
}
