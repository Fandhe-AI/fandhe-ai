//! sort／topk の ROCm 実行入口（`backend-cuda::sort`・`sort_model::plan_sort` の移植。
//! イシュー #3439・R8）。
//!
//! 役割: [`RocmSort`] が HIP/HIPRTC をロードして `kernels_sort` のソースをコンパイルし、
//! 行優先 contiguous のホストスライスから `(values, index)` を返す。`topk` は `out_len = k`・
//! `descending = largest` として同じ経路を使う。`BackendOps::sort`／`topk` への結線
//! （`sort_out_shape`／`topk_out_shape` による形状再検査・`Tensor` 化）は `RocmBackendOps`
//! （#3431・R0）側の責務。
//!
//! 順序契約（CPU 参照実装と同一）: 安定（同値は元添字順）・NaN は最大（昇順では末尾、降順では先頭）・
//! ±0 は同値。ビットニックソートの `j`・`k` ループはホスト側で、各ステップを同期起動する。
//! ホスト側で `dim_size`・`lines * padded` の `i32` 収容と `out_len <= dim_size` を fail-closed に
//! 検証してから起動する。

use crate::error::RocmError;
use crate::kernels_sort::{BITONIC_STEP_SPEC, BUILD_KEYS_SPEC, FINALIZE_SPEC, render_sort_source};
use crate::launch::{Arg, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "sort";

/// ライン分解と 2 のべき乗パディングの計画（`backend-cuda::sort_model::SortPlan` 相当）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SortPlan {
    pub(crate) dim_size: usize,
    pub(crate) inner: usize,
    pub(crate) lines: usize,
    pub(crate) padded: usize,
}

fn bad(detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op: OP, detail }
}

/// `shape`／`dim` から [`SortPlan`] を作り、`dim_size` と `lines * padded` が `i32::MAX` 以下で
/// あることを検証する（CUDA 版 `plan_sort` と同じ上限）。
pub(crate) fn plan_sort(shape: &[usize], dim: usize) -> Result<SortPlan, RocmError> {
    let limit = i32::MAX as usize;
    let dim_size = *shape
        .get(dim)
        .ok_or_else(|| bad(format!("dim={dim} が rank={} の範囲外です", shape.len())))?;
    let overflow = |n: usize| RocmError::SizeLimitExceeded { op: OP, n, limit };
    let outer = shape[..dim]
        .iter()
        .try_fold(1usize, |acc, &s| acc.checked_mul(s))
        .ok_or_else(|| overflow(usize::MAX))?;
    let inner = shape[dim + 1..]
        .iter()
        .try_fold(1usize, |acc, &s| acc.checked_mul(s))
        .ok_or_else(|| overflow(usize::MAX))?;
    if dim_size > limit {
        return Err(overflow(dim_size));
    }
    let padded = dim_size
        .checked_next_power_of_two()
        .ok_or_else(|| overflow(dim_size))?;
    let lines = outer
        .checked_mul(inner)
        .ok_or_else(|| overflow(usize::MAX))?;
    let total_keys = lines
        .checked_mul(padded)
        .ok_or_else(|| overflow(usize::MAX))?;
    if total_keys > limit {
        return Err(overflow(total_keys));
    }
    Ok(SortPlan {
        dim_size,
        inner,
        lines,
        padded,
    })
}

/// 検証済みの sort 実行計画。
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SortRun {
    pub(crate) plan: SortPlan,
    pub(crate) out_len: usize,
    pub(crate) total_out: usize,
}

/// sort/topk の引数検証（`input.len() == lines * dim_size`・`out_len <= dim_size`）。
pub(crate) fn validate_sort(
    input_len: usize,
    shape: &[usize],
    dim: usize,
    out_len: usize,
) -> Result<SortRun, RocmError> {
    let plan = plan_sort(shape, dim)?;
    if out_len > plan.dim_size {
        return Err(bad(format!(
            "out_len {out_len} が dim_size {} を超えます",
            plan.dim_size
        )));
    }
    // lines * dim_size は lines * padded 以下で上限検査済みのため overflow しない。
    let expected = plan.lines * plan.dim_size;
    if input_len != expected {
        return Err(bad(format!(
            "input の長さ不一致: 期待 {expected}, 実際 {input_len}"
        )));
    }
    Ok(SortRun {
        plan,
        out_len,
        total_out: plan.lines * out_len,
    })
}

/// コンパイル済みの sort カーネルを保持し、実行する。
pub struct RocmSort {
    host: KernelHost,
}

impl RocmSort {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_sort_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// `dim` 軸に沿って各ラインを整列し、先頭 `out_len` 個の `(values, index)` を返す。
    ///
    /// 出力は `[.., out_len, ..]`（`dim` 軸だけ `out_len`）の行優先。`descending` で降順。
    pub fn run_sort_f32(
        &self,
        input: &[f32],
        shape: &[usize],
        dim: usize,
        descending: bool,
        out_len: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), RocmError> {
        let run = validate_sort(input.len(), shape, dim, out_len)?;
        if run.total_out == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let SortPlan {
            dim_size,
            inner,
            lines,
            padded,
        } = run.plan;
        let to_i64 = |v: usize| {
            i64::try_from(v).map_err(|_| RocmError::SizeLimitExceeded {
                op: OP,
                n: v,
                limit: i64::MAX as usize,
            })
        };
        let (lines_i, dim_size_i, inner_i, padded_i, out_len_i) = (
            to_i64(lines)?,
            to_i64(dim_size)?,
            to_i64(inner)?,
            to_i64(padded)?,
            to_i64(out_len)?,
        );
        let numel_in_i = to_i64(input.len())?;
        let numel_out_i = to_i64(run.total_out)?;
        let total_keys = lines * padded;

        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let dinput = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dkeys = TypedDeviceBuffer::<u64>::new_zeroed(rt(), total_keys)?;
        let dvalues = TypedDeviceBuffer::<f32>::new_zeroed(rt(), run.total_out)?;
        let dindex = TypedDeviceBuffer::<i32>::new_zeroed(rt(), run.total_out)?;

        let build = [
            Arg::Ptr(dinput.as_device_ptr()),
            Arg::Ptr(dkeys.as_device_ptr()),
            Arg::I64(lines_i),
            Arg::I64(dim_size_i),
            Arg::I64(inner_i),
            Arg::I64(padded_i),
            Arg::I32(i32::from(descending)),
            Arg::I64(numel_in_i),
        ];
        self.host.run(OP, &BUILD_KEYS_SPEC, total_keys, &build)?;

        if padded > 1 {
            let mut k = 2usize;
            while k <= padded {
                let mut j = k / 2;
                while j >= 1 {
                    let step = [
                        Arg::Ptr(dkeys.as_device_ptr()),
                        Arg::I64(to_i64(j)?),
                        Arg::I64(to_i64(k)?),
                        Arg::I64(padded_i),
                        Arg::I64(lines_i),
                    ];
                    self.host.run(OP, &BITONIC_STEP_SPEC, total_keys, &step)?;
                    j /= 2;
                }
                k *= 2;
            }
        }

        let finalize = [
            Arg::Ptr(dinput.as_device_ptr()),
            Arg::Ptr(dkeys.as_device_ptr()),
            Arg::Ptr(dvalues.as_device_ptr()),
            Arg::Ptr(dindex.as_device_ptr()),
            Arg::I64(lines_i),
            Arg::I64(dim_size_i),
            Arg::I64(inner_i),
            Arg::I64(padded_i),
            Arg::I64(out_len_i),
            Arg::I64(numel_in_i),
            Arg::I64(numel_out_i),
        ];
        self.host
            .run(OP, &FINALIZE_SPEC, run.total_out, &finalize)?;
        Ok((dvalues.to_vec()?, dindex.to_vec()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_basic_shapes() {
        let p = plan_sort(&[3, 4], 1).unwrap();
        assert_eq!((p.dim_size, p.inner, p.lines, p.padded), (4, 1, 3, 4));
        let p = plan_sort(&[3, 4], 0).unwrap();
        assert_eq!((p.dim_size, p.inner, p.lines, p.padded), (3, 4, 4, 4));
        let p = plan_sort(&[2, 5, 3], 1).unwrap();
        assert_eq!((p.dim_size, p.inner, p.lines, p.padded), (5, 3, 6, 8));
        // dim_size 0 → padded 1。
        let p = plan_sort(&[2, 0], 1).unwrap();
        assert_eq!((p.dim_size, p.padded), (0, 1));
    }

    #[test]
    fn plan_rejects_bad_dim_and_overflow() {
        assert!(matches!(
            plan_sort(&[1], 1),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(plan_sort(&[], 0).is_err());
        assert!(matches!(
            plan_sort(&[i32::MAX as usize + 1], 0),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        // lines * padded が i32::MAX を超える。
        assert!(matches!(
            plan_sort(&[1 << 20, 1 << 12], 1),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        assert!(plan_sort(&[usize::MAX, 2, 2], 1).is_err());
    }

    #[test]
    fn validate_checks_lengths_and_out_len() {
        let r = validate_sort(12, &[3, 4], 1, 2).unwrap();
        assert_eq!(r.total_out, 6);
        assert!(validate_sort(11, &[3, 4], 1, 2).is_err());
        assert!(validate_sort(12, &[3, 4], 1, 5).is_err());
        assert_eq!(validate_sort(12, &[3, 4], 1, 0).unwrap().total_out, 0);
        assert_eq!(validate_sort(0, &[2, 0], 1, 0).unwrap().total_out, 0);
    }
}
