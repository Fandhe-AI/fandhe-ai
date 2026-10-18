//! `argmax`／`argmin` の ROCm 実行入口（`backend-cuda::arg_reduce` の移植。
//! イシュー #3434・分解案 R3）。
//!
//! 役割: [`RocmArgReduce`] が HIP/HIPRTC をロードして `kernels_arg_reduce` のソースを
//! コンパイルし、ホストのスライスに対する全軸 2 段（チャンク分割）／単一軸の添字縮約を実行する。
//! `BackendOps::argmax`／`argmin` への結線（形状再検査・空縮約の CPU 委譲・`Tensor<i32>` 化）は
//! `RocmBackendOps`（`ops.rs`）の責務で、本型はスライス入出力のみ。
//! ホスト側で `i32` 範囲・要素数の checked 演算を検証してから起動する（カーネル側 REQ-8 と二重化）。
//! 起動の `unsafe` は `KernelHost::run` の 1 か所のままで、本ファイルに `unsafe` はない。

use crate::error::RocmError;
use crate::kernels_arg_reduce::{
    ARGMAX_ALL_FINALIZE_SPEC, ARGMAX_ALL_PARTIAL_SPEC, ARGMAX_AXIS_SPEC, ARGMIN_ALL_FINALIZE_SPEC,
    ARGMIN_ALL_PARTIAL_SPEC, ARGMIN_AXIS_SPEC, render_arg_reduce_source,
};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::reduce::validate_reduce_axis;
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "arg_reduce";

/// 全軸縮約 1 段目の 1 チャンクあたり最小要素数（小さな入力で過剰分割しない）。
pub(crate) const ARG_MIN_CHUNK: usize = 4096;

/// 全軸縮約 1 段目のチャンク数上限（`256 * 16`。2 段目が単一スレッドで逐次走査するため、
/// 大きくしすぎると 2 段目が支配的になる）。
pub(crate) const ARG_MAX_CHUNKS: usize = 256 * 16;

/// 添字縮約の種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArgReduceKind {
    /// 最大値の添字（`v > best` のときのみ更新）。
    Max,
    /// 最小値の添字（`v < best` のときのみ更新）。
    Min,
}

impl ArgReduceKind {
    fn specs(
        self,
    ) -> (
        &'static KernelSpec,
        &'static KernelSpec,
        &'static KernelSpec,
    ) {
        match self {
            Self::Max => (
                &ARGMAX_ALL_PARTIAL_SPEC,
                &ARGMAX_ALL_FINALIZE_SPEC,
                &ARGMAX_AXIS_SPEC,
            ),
            Self::Min => (
                &ARGMIN_ALL_PARTIAL_SPEC,
                &ARGMIN_ALL_FINALIZE_SPEC,
                &ARGMIN_AXIS_SPEC,
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

/// 全軸縮約のチャンク計画 `(chunk_len, num_chunks)`。連続・昇順で `0..numel` をちょうど覆い、
/// `num_chunks <= ARG_MAX_CHUNKS`。`numel == 0` は `(0, 0)`。
pub(crate) fn arg_all_plan(numel: usize) -> (usize, usize) {
    if numel == 0 {
        return (0, 0);
    }
    let num_chunks = numel.div_ceil(ARG_MIN_CHUNK).clamp(1, ARG_MAX_CHUNKS);
    let chunk_len = numel.div_ceil(num_chunks).max(1);
    // `chunk_len` は `num_chunks` からの逆算のため、余りで実際に必要なチャンク数が下回りうる
    // （空チャンクを作らない再計算。上限には抵触しない）。
    let num_chunks = numel.div_ceil(chunk_len);
    (chunk_len, num_chunks)
}

/// コンパイル済みの argmax／argmin カーネルを保持し、実行する。
pub struct RocmArgReduce {
    host: KernelHost,
}

impl RocmArgReduce {
    /// HIP/HIPRTC をロードし argmax／argmin カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_arg_reduce_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// 全要素の最大値の添字（タイは最初・NaN は無視・全 NaN は 0）。空入力は `InvalidOpArgs`。
    pub fn run_argmax_all_f32(&self, x: &[f32]) -> Result<i32, RocmError> {
        self.run_all(ArgReduceKind::Max, x)
    }

    /// 全要素の最小値の添字。空入力は `InvalidOpArgs`。
    pub fn run_argmin_all_f32(&self, x: &[f32]) -> Result<i32, RocmError> {
        self.run_all(ArgReduceKind::Min, x)
    }

    /// `[outer, axis_len, inner]` の `axis_len` 軸の最大値の添字（出力は `outer * inner` 要素）。
    /// `axis_len == 0` かつ出力が非空なら `InvalidOpArgs`。
    pub fn run_argmax_axis_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<i32>, RocmError> {
        self.run_axis(ArgReduceKind::Max, x, outer, axis_len, inner)
    }

    /// `[outer, axis_len, inner]` の `axis_len` 軸の最小値の添字。
    pub fn run_argmin_axis_f32(
        &self,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<i32>, RocmError> {
        self.run_axis(ArgReduceKind::Min, x, outer, axis_len, inner)
    }

    /// `ops.rs` の dispatch 入口（全軸）。
    pub(crate) fn run_all(&self, kind: ArgReduceKind, x: &[f32]) -> Result<i32, RocmError> {
        if x.is_empty() {
            return Err(bad("空の縮約（argmax/argmin は単位元を持たない）"));
        }
        let numel_i = fit_i32(x.len(), "numel")?;
        let (chunk_len, num_chunks) = arg_all_plan(x.len());
        let chunk_len_i = fit_i32(chunk_len, "chunk_len")?;
        let num_chunks_i = fit_i32(num_chunks, "num_chunks")?;
        let (partial_spec, finalize_spec, _) = kind.specs();
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let pval = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), num_chunks)?;
        let pidx = TypedDeviceBuffer::<i32>::new_zeroed(rt.clone(), num_chunks)?;
        let dout = TypedDeviceBuffer::<i32>::new_zeroed(rt, 1)?;
        self.host.run(
            OP,
            partial_spec,
            num_chunks,
            &[
                Arg::Ptr(din.as_device_ptr()),
                Arg::Ptr(pval.as_device_ptr()),
                Arg::Ptr(pidx.as_device_ptr()),
                Arg::I32(numel_i),
                Arg::I32(chunk_len_i),
                Arg::I32(num_chunks_i),
            ],
        )?;
        // 2 段目は単一スレッド（カーネル内ガード `blockIdx.x == 0 && threadIdx.x == 0`）。
        self.host.run(
            OP,
            finalize_spec,
            1,
            &[
                Arg::Ptr(pval.as_device_ptr()),
                Arg::Ptr(pidx.as_device_ptr()),
                Arg::Ptr(dout.as_device_ptr()),
                Arg::I32(num_chunks_i),
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
        kind: ArgReduceKind,
        x: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<i32>, RocmError> {
        let total_out = validate_reduce_axis(x.len(), outer, axis_len, inner)?;
        if total_out == 0 {
            return Ok(Vec::new());
        }
        if axis_len == 0 {
            return Err(bad("空の縮約（argmax/argmin は単位元を持たない）"));
        }
        let (_, _, axis_spec) = kind.specs();
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dout = TypedDeviceBuffer::<i32>::new_zeroed(rt, total_out)?;
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

    /// 計画がチャンクを連続・昇順で `0..numel` ちょうどに覆い、上限を超えないこと。
    fn assert_plan_covers_range(numel: usize) {
        let (chunk_len, num_chunks) = arg_all_plan(numel);
        assert!(num_chunks <= ARG_MAX_CHUNKS, "numel={numel}");
        if numel == 0 {
            assert_eq!((chunk_len, num_chunks), (0, 0));
            return;
        }
        assert!(chunk_len >= 1 && num_chunks >= 1, "numel={numel}");
        let mut covered = 0usize;
        for t in 0..num_chunks {
            let start = t * chunk_len;
            let end = ((t + 1) * chunk_len).min(numel);
            assert_eq!(
                start, covered,
                "チャンク {t} が連続していない: numel={numel}"
            );
            assert!(end > start, "空チャンク t={t}: numel={numel}");
            covered = end;
        }
        assert_eq!(covered, numel);
    }

    #[test]
    fn plan_covers_range_for_representative_sizes() {
        for numel in [
            0,
            1,
            2,
            4095,
            4096,
            4097,
            8191,
            8192,
            8193,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS - 1,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS + 1,
            10_000_000,
            i32::MAX as usize,
        ] {
            assert_plan_covers_range(numel);
        }
    }

    #[test]
    fn plan_small_input_is_single_chunk() {
        assert_eq!(arg_all_plan(1), (1, 1));
        assert_eq!(arg_all_plan(4096), (4096, 1));
    }

    #[test]
    fn fit_i32_rejects_overflow() {
        assert_eq!(fit_i32(5, "n").unwrap(), 5);
        assert!(matches!(
            fit_i32(i32::MAX as usize + 1, "n"),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}
