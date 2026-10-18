//! 汎用縮約 `sum`／`max`／`min` の ROCm 実行入口（`backend-cuda::reduce` の移植。
//! イシュー #3434・分解案 R3）。
//!
//! 役割: [`RocmReduce`] が HIP/HIPRTC をロードして `kernels_reduce` のソース（wave 幅を
//! `WarpGeometry` で注入済み）をコンパイルし、ホストのスライスに対して全軸縮約（2 段）と
//! 単一軸縮約を実行する。`BackendOps::sum`／`max`／`min` への結線（形状再検査・空縮約の
//! CPU 委譲・`Tensor` 化）は `RocmBackendOps`（`ops.rs`）の責務で、本型はスライス入出力のみ。
//! ホスト側で `i32` 範囲・要素数の checked 演算を検証してから起動する（カーネル側 REQ-8 と二重化）。
//!
//! 全軸縮約は `reduce_num_blocks(numel) = min(ceil(numel / 256), 1024)` ブロックの 1 段目
//! （`run_blocks`。grid-stride ループ）→ 1 ブロックの 2 段目の順に起動し、partial は
//! sum が `f64`・max／min が `f32` のデバイスバッファに置く。軸縮約は 1 スレッド = 1 出力。
//! 起動の `unsafe` は `KernelHost::run` の 1 か所のままで、本ファイルに `unsafe` はない。

use crate::error::RocmError;
use crate::kernels_reduce::{
    MAX_ALL_FINALIZE_SPEC, MAX_ALL_PARTIAL_SPEC, MAX_AXIS_SPEC, MIN_ALL_FINALIZE_SPEC,
    MIN_ALL_PARTIAL_SPEC, MIN_AXIS_SPEC, REDUCE_BLOCK_DIM, REDUCE_MAX_BLOCKS,
    SUM_ALL_FINALIZE_SPEC, SUM_ALL_PARTIAL_SPEC, SUM_AXIS_SPEC, render_reduce_source,
};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "reduce";

/// 縮約の種類。`ops.rs` の dispatch と `reduce_model.rs` がカーネル選択に使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReduceKind {
    /// 総和（`f64` アキュムレータ）。
    Sum,
    /// 最大値（`fmaxf`。NaN 非伝播）。
    Max,
    /// 最小値（`fminf`。NaN 非伝播）。
    Min,
}

impl ReduceKind {
    fn specs(
        self,
    ) -> (
        &'static KernelSpec,
        &'static KernelSpec,
        &'static KernelSpec,
    ) {
        match self {
            Self::Sum => (
                &SUM_ALL_PARTIAL_SPEC,
                &SUM_ALL_FINALIZE_SPEC,
                &SUM_AXIS_SPEC,
            ),
            Self::Max => (
                &MAX_ALL_PARTIAL_SPEC,
                &MAX_ALL_FINALIZE_SPEC,
                &MAX_AXIS_SPEC,
            ),
            Self::Min => (
                &MIN_ALL_PARTIAL_SPEC,
                &MIN_ALL_FINALIZE_SPEC,
                &MIN_AXIS_SPEC,
            ),
        }
    }
}

fn bad(detail: impl Into<String>) -> RocmError {
    RocmError::InvalidOpArgs {
        op: OP,
        detail: detail.into(),
    }
}

fn fit_i32(value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op: OP,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

/// shape と縮約軸から `(outer, axis_len, inner)` を求める（`backend-cpu::reduction::axis_reduce`
/// と同じ分解）。`axis < shape.len()` は呼び出し側（`reduce_out_shape`）が検査済みの契約だが、
/// 範囲外でも panic せず `InvalidOpArgs` を返す。積は `checked_mul` で検査する（A03）。
pub(crate) fn reduce_axis_layout(
    shape: &[usize],
    axis: usize,
) -> Result<(usize, usize, usize), RocmError> {
    if axis >= shape.len() {
        return Err(bad(format!(
            "axis={axis} が rank={} の範囲外です",
            shape.len()
        )));
    }
    let prod = |dims: &[usize], name: &str| {
        dims.iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad(format!("{name} の次元積が usize を超えます")))
    };
    let outer = prod(&shape[..axis], "outer")?;
    let inner = prod(&shape[axis + 1..], "inner")?;
    outer
        .checked_mul(inner)
        .ok_or_else(|| bad("outer*inner が usize を超えます"))?;
    Ok((outer, shape[axis], inner))
}

/// 全軸縮約の入力長を `i32` 収容で検証する（カーネル引数 `numel` は `int`）。
pub(crate) fn validate_reduce_all(numel: usize) -> Result<i32, RocmError> {
    fit_i32(numel, "numel")
}

/// 軸縮約の寸法を検証し、出力要素数 `outer * inner` を返す。
///
/// `x_len == outer*axis_len*inner`・各値と `numel`・出力要素数の `i32` 収容を要求する。
pub(crate) fn validate_reduce_axis(
    x_len: usize,
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Result<usize, RocmError> {
    let total_out = outer
        .checked_mul(inner)
        .ok_or_else(|| bad("outer*inner が usize を超えます"))?;
    let numel = total_out
        .checked_mul(axis_len)
        .ok_or_else(|| bad("outer*inner*axis_len が usize を超えます"))?;
    if x_len != numel {
        return Err(bad(format!("x の長さ不一致: 期待 {numel}, 実際 {x_len}")));
    }
    fit_i32(numel, "numel")?;
    fit_i32(outer, "outer")?;
    fit_i32(axis_len, "axis_len")?;
    fit_i32(inner, "inner")?;
    fit_i32(total_out, "outer*inner")?;
    Ok(total_out)
}

/// 全軸縮約 1 段目の起動ブロック数 `min(ceil(numel / 256), 1024)`。`numel > 0` が前提。
pub(crate) fn reduce_num_blocks(numel: usize) -> u32 {
    let blocks = numel.div_ceil(REDUCE_BLOCK_DIM as usize);
    blocks.min(REDUCE_MAX_BLOCKS as usize) as u32
}

/// コンパイル済みの縮約カーネルを保持し、実行する。
pub struct RocmReduce {
    host: KernelHost,
}

impl RocmReduce {
    /// HIP/HIPRTC をロードし縮約カーネルを `target`（wave 幅を含む）向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_reduce_source(target)?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// 全要素の総和（`f64` アキュムレータ・最後に 1 回 `f32`）。空入力は 0.0。
    pub fn run_sum_all_f32(&self, x: &[f32]) -> Result<f32, RocmError> {
        self.run_all(ReduceKind::Sum, x)
    }

    /// 全要素の最大値。空入力は `InvalidOpArgs`（単位元を持たない）。
    pub fn run_max_all_f32(&self, x: &[f32]) -> Result<f32, RocmError> {
        self.run_all(ReduceKind::Max, x)
    }

    /// 全要素の最小値。空入力は `InvalidOpArgs`。
    pub fn run_min_all_f32(&self, x: &[f32]) -> Result<f32, RocmError> {
        self.run_all(ReduceKind::Min, x)
    }

    /// `[outer, axis_len, inner]`（行優先）の `axis_len` 軸の総和。
    pub fn run_sum_axis_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run_axis(ReduceKind::Sum, x, outer, axis_len, inner)
    }

    /// `[outer, axis_len, inner]` の `axis_len` 軸の最大値。`axis_len == 0` は `InvalidOpArgs`。
    pub fn run_max_axis_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run_axis(ReduceKind::Max, x, outer, axis_len, inner)
    }

    /// `[outer, axis_len, inner]` の `axis_len` 軸の最小値。`axis_len == 0` は `InvalidOpArgs`。
    pub fn run_min_axis_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        self.run_axis(ReduceKind::Min, x, outer, axis_len, inner)
    }

    /// `ops.rs` の dispatch 入口（全軸）。
    pub(crate) fn run_all(&self, kind: ReduceKind, x: &[f32]) -> Result<f32, RocmError> {
        let numel_i = validate_reduce_all(x.len())?;
        if x.is_empty() {
            return match kind {
                ReduceKind::Sum => Ok(0.0),
                _ => Err(bad("空の縮約（max/min は単位元を持たない）")),
            };
        }
        let (partial_spec, finalize_spec, _) = kind.specs();
        let num_blocks = reduce_num_blocks(x.len());
        let num_blocks_i = fit_i32(num_blocks as usize, "num_blocks")?;
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), 1)?;
        // sum の partial は f64（最後の 1 回だけ f32 へ downcast する契約）。max/min は f32。
        let is_sum = kind == ReduceKind::Sum;
        let partial_f64 = is_sum
            .then(|| TypedDeviceBuffer::<f64>::new_zeroed(rt.clone(), num_blocks as usize))
            .transpose()?;
        let partial_f32 = (!is_sum)
            .then(|| TypedDeviceBuffer::<f32>::new_zeroed(rt, num_blocks as usize))
            .transpose()?;
        let partial_ptr = partial_f64
            .as_ref()
            .map(TypedDeviceBuffer::as_device_ptr)
            .or_else(|| partial_f32.as_ref().map(TypedDeviceBuffer::as_device_ptr))
            .ok_or_else(|| bad("partial バッファを確保できません"))?;
        self.host.run_blocks(
            OP,
            partial_spec,
            num_blocks,
            &[
                Arg::Ptr(din.as_device_ptr()),
                Arg::Ptr(partial_ptr),
                Arg::I32(numel_i),
            ],
        )?;
        // 2 段目は 1 ブロック（256 スレッド）。
        self.host.run(
            OP,
            finalize_spec,
            REDUCE_BLOCK_DIM as usize,
            &[
                Arg::Ptr(partial_ptr),
                Arg::Ptr(dout.as_device_ptr()),
                Arg::I32(num_blocks_i),
            ],
        )?;
        let out = dout.to_vec()?;
        out.first()
            .copied()
            .ok_or_else(|| bad("出力が 1 要素ではありません"))
    }

    /// `ops.rs` の dispatch 入口（軸指定）。
    pub(crate) fn run_axis(
        &self,
        kind: ReduceKind,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let total_out = validate_reduce_axis(x.len(), outer, axis_len, inner)?;
        if total_out == 0 {
            return Ok(Vec::new());
        }
        if axis_len == 0 {
            return match kind {
                ReduceKind::Sum => Ok(vec![0.0; total_out]),
                _ => Err(bad("空の縮約（max/min は単位元を持たない）")),
            };
        }
        let (_, _, axis_spec) = kind.specs();
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, total_out)?;
        self.host.run(
            OP,
            axis_spec,
            total_out,
            &[
                Arg::Ptr(din.as_device_ptr()),
                Arg::Ptr(dout.as_device_ptr()),
                Arg::I32(fit_i32(outer, "outer")?),
                Arg::I32(fit_i32(axis_len, "axis_len")?),
                Arg::I32(fit_i32(inner, "inner")?),
            ],
        )?;
        dout.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_layout_first_middle_last() {
        assert_eq!(reduce_axis_layout(&[3, 4096], 0).unwrap(), (1, 3, 4096));
        assert_eq!(reduce_axis_layout(&[2, 3, 4], 1).unwrap(), (2, 3, 4));
        assert_eq!(reduce_axis_layout(&[64, 784], 1).unwrap(), (64, 784, 1));
        assert_eq!(reduce_axis_layout(&[5], 0).unwrap(), (1, 5, 1));
    }

    #[test]
    fn axis_layout_rejects_bad_axis_and_overflow() {
        assert!(reduce_axis_layout(&[2, 3], 2).is_err());
        assert!(reduce_axis_layout(&[], 0).is_err());
        assert!(reduce_axis_layout(&[usize::MAX, 2, 3], 2).is_err());
        assert!(reduce_axis_layout(&[3, 2, usize::MAX], 0).is_err());
    }

    #[test]
    fn validate_all_and_axis() {
        assert_eq!(validate_reduce_all(7).unwrap(), 7);
        assert!(validate_reduce_all(i32::MAX as usize + 1).is_err());
        assert_eq!(validate_reduce_axis(24, 2, 3, 4).unwrap(), 8);
        assert_eq!(validate_reduce_axis(0, 2, 0, 3).unwrap(), 6);
        assert_eq!(validate_reduce_axis(0, 0, 4, 3).unwrap(), 0);
        // 長さ不一致・積の overflow・i32 超過。
        assert!(validate_reduce_axis(23, 2, 3, 4).is_err());
        assert!(validate_reduce_axis(0, usize::MAX, 1, 2).is_err());
        assert!(validate_reduce_axis(0, 1 << 20, 1 << 20, 1).is_err());
        let big = i32::MAX as usize + 1;
        assert!(matches!(
            validate_reduce_axis(0, 1, big, 0),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // outer*inner が i32 を超える（個別は収まる）。
        assert!(validate_reduce_axis(0, 65536, 0, 32768).is_err());
    }

    #[test]
    fn num_blocks_is_capped() {
        assert_eq!(reduce_num_blocks(1), 1);
        assert_eq!(reduce_num_blocks(256), 1);
        assert_eq!(reduce_num_blocks(257), 2);
        assert_eq!(reduce_num_blocks(256 * 1024), 1024);
        assert_eq!(reduce_num_blocks(256 * 1024 + 1), 1024);
        assert_eq!(reduce_num_blocks(i32::MAX as usize), 1024);
    }
}
