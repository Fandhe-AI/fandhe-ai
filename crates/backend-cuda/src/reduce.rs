//! 汎用 reduction（`sum`／`max`。全軸・単一軸）の起動 API（NVRTC
//! コンパイル・保持・実行。イシュー #1584・親イシュー #1571）。
//!
//! `mse.rs::CudaMse`・`elementwise.rs::CudaElementwise` と同じ構成方針を
//! 踏襲する: [`CudaReduce::new`] が `CudaDevice` から 8 カーネル
//! （`kernels_reduce.rs`）を NVRTC コンパイルして保持し、以降は
//! `run_sum_all_f32`／`run_max_all_f32`／`run_sum_axis_f32`／
//! `run_max_axis_f32` へホスト側スライスを渡すだけで GPU 実行できる。
//! `ops.rs::CudaBackendOps::sum`／`max` から `BackendOps` の実装として
//! 呼ばれる（軸の妥当性検査自体は `ops.rs` が `reduce_out_shape` で行い、
//! 本モジュールは `(outer, axis_len, inner)` へ分解された後の実行を担う）。
//!
//! 空縮約の意味論（`backend-cpu::reduction` と同一の述語）:
//! - `sum` 全軸 `numel == 0` → `0.0`。軸指定で `axis_len == 0` →
//!   出力を全 `0.0` で埋める（カーネル起動を行わない）。
//! - `max` 全軸 `numel == 0`、または `axis_len == 0 && outer*inner > 0`
//!   → [`CudaError::EmptyReduction`]（`backend-cpu::reduction::
//!   ReduceError::EmptyReduction` と同一の意味論・`Display` 文言。
//!   `ops.rs` が同一の文言 `"empty reduction for op \"max\""` を持つ
//!   `BackendError::KernelLaunchFailed` へ写像する）。`outer*inner == 0`
//!   （出力自体が空）は vacuous に成功（空 `Vec`）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_reduce::{self, REDUCE_BLOCK_DIM, REDUCE_LASTAXIS_BLOCK_DIM, REDUCE_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `outer`/`axis_len`/`inner`（および全軸縮約の `numel`）が
/// `i32::MAX` に収まることを検証する（カーネル引数 `int` は C の 32bit
/// 符号付き整数のため。`elementwise.rs::validate_elementwise_len` と
/// 同じ理由）。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::InvalidReduceShape {
        detail: format!("reduce dimension must fit in i32 (kernel argument type): {name}={value}"),
    })
}

/// `shape`・縮約軸 `axis` から `(outer, axis_len, inner)` を導出する純
/// 関数（`backend-cpu::reduction::axis_reduce` の `outer_dims`／
/// `axis_len`／`inner_dims` 分解と同一）。`axis` は呼び出し元
/// （`ops.rs`）が `reduce_out_shape` で範囲検査済みであることを前提と
/// する（本関数自体は範囲外 `axis` を検査しない）。`outer`/`inner` は
/// `checked_mul` でオーバーフローを検出する（`ShapeError::
/// ElementCountOverflow` 相当。OWASP A03）。
pub(crate) fn reduce_axis_layout(
    shape: &[usize],
    axis: usize,
) -> Result<(usize, usize, usize), CudaError> {
    let outer = shape[..axis].iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| CudaError::InvalidReduceShape {
                detail: "reduce_axis_layout: outer dimension product overflow".to_string(),
            })
    })?;
    let axis_len = shape[axis];
    let inner = shape[axis + 1..].iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| CudaError::InvalidReduceShape {
                detail: "reduce_axis_layout: inner dimension product overflow".to_string(),
            })
    })?;
    outer
        .checked_mul(inner)
        .ok_or_else(|| CudaError::InvalidReduceShape {
            detail: "reduce_axis_layout: outer * inner overflow".to_string(),
        })?;
    Ok((outer, axis_len, inner))
}

/// 全軸縮約 1 段目の起動ブロック数を決定する。
/// `min(ceil_div(numel, REDUCE_BLOCK_DIM), REDUCE_MAX_BLOCKS)`
/// （`mse.rs::mse_num_blocks` と同一契約）。`numel > 0` を前提とする。
fn reduce_num_blocks(numel: u32) -> u32 {
    numel.div_ceil(REDUCE_BLOCK_DIM).min(REDUCE_MAX_BLOCKS)
}

/// reduction 8 カーネル（全軸 sum/max 各 2 段・単一軸 sum/max 各 2 種。
/// いずれも f32 入出力）のコンパイル済みハンドルを保持する。
pub struct CudaReduce {
    stream: Arc<CudaStream>,
    /// `elementwise.rs::CudaElementwise::ordinal` と同じ役割
    /// （`Self::with_driver_call` が `context_cache::with_driver_call`
    /// を呼ぶ際のキー。イシュー #1349・PR #1390 是正の踏襲）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    sum_all_partial_f32: CudaFunction,
    sum_all_finalize_f32: CudaFunction,
    sum_axis_f32: CudaFunction,
    sum_lastaxis_f32: CudaFunction,
    max_all_partial_f32: CudaFunction,
    max_all_finalize_f32: CudaFunction,
    max_axis_f32: CudaFunction,
    max_lastaxis_f32: CudaFunction,
}

impl CudaReduce {
    /// `device` 上で reduction 8 カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する（`mse.rs::CudaMse::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let sum_all_partial_f32 = compile_and_load!(
            kernels_reduce::REDUCE_SUM_ALL_PARTIAL_F32,
            "reduce_sum_all_partial_f32"
        );
        let sum_all_finalize_f32 = compile_and_load!(
            kernels_reduce::REDUCE_SUM_ALL_FINALIZE_F32,
            "reduce_sum_all_finalize_f32"
        );
        let sum_axis_f32 =
            compile_and_load!(kernels_reduce::REDUCE_SUM_AXIS_F32, "reduce_sum_axis_f32");
        let sum_lastaxis_f32 = compile_and_load!(
            kernels_reduce::REDUCE_SUM_LASTAXIS_F32,
            "reduce_sum_lastaxis_f32"
        );
        let max_all_partial_f32 = compile_and_load!(
            kernels_reduce::REDUCE_MAX_ALL_PARTIAL_F32,
            "reduce_max_all_partial_f32"
        );
        let max_all_finalize_f32 = compile_and_load!(
            kernels_reduce::REDUCE_MAX_ALL_FINALIZE_F32,
            "reduce_max_all_finalize_f32"
        );
        let max_axis_f32 =
            compile_and_load!(kernels_reduce::REDUCE_MAX_AXIS_F32, "reduce_max_axis_f32");
        let max_lastaxis_f32 = compile_and_load!(
            kernels_reduce::REDUCE_MAX_LASTAXIS_F32,
            "reduce_max_lastaxis_f32"
        );

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            sum_all_partial_f32,
            sum_all_finalize_f32,
            sum_axis_f32,
            sum_lastaxis_f32,
            max_all_partial_f32,
            max_all_finalize_f32,
            max_axis_f32,
            max_lastaxis_f32,
        })
    }

    /// `CudaReduce` の driver 呼び出しを CUDA Graph capture 排他へ参加
    /// させる共通ヘルパー（`mse.rs::CudaMse::with_driver_call` と同じ
    /// 設計。PR #1390 是正の踏襲）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// 全軸 `sum`（`Σ a[i]`。`double` アキュムレータ。本ファイル冒頭
    /// コメント「空縮約の意味論」参照）。`numel == 0` はカーネル起動を
    /// 回避し `0.0` を返す。
    pub fn run_sum_all_f32(&self, a: &[f32]) -> Result<f32, CudaError> {
        let numel = a.len();
        validate_i32_bound(numel, "numel")?;
        if numel == 0 {
            return Ok(0.0);
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;

            let num_blocks = reduce_num_blocks(numel as u32);
            // `reduce_sum_all_partial_f32` の finalize 用 `double` partial
            // バッファは `pool.rs::PoolDtype` が f32/f16 のみ対応のため
            // （`docs/backend-cuda-pool-allocator-decision.md`）、高々
            // `REDUCE_MAX_BLOCKS`（1024）要素の小規模・per-call 確保として
            // プールを経由せず `stream.alloc_zeros` で直接確保する（`mse.rs`
            // の `alloc_uninit_f32` 経由プール確保とは異なる経路だが、
            // `reduce_sum_all_partial_f32` は起動する `num_blocks` 個の
            // ブロックそれぞれが `partial[blockIdx.x]` を必ず 1 回書くため
            // 内容は上書きされる。ゼロ初期化はプール非経由確保の安全側
            // 既定として採用）。
            let mut partial_dev = self.stream.alloc_zeros::<f64>(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `a_dev` は `numel` 要素の H2D 済みデバイスバッファ、
            // `partial_dev` は `num_blocks` 要素確保済みでカーネルが
            // `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ書く
            // （`kernels_reduce.rs::REDUCE_SUM_ALL_PARTIAL_F32` 参照）。
            // カーネル内 grid-stride ループは `idx < numel` を維持する
            // （REQ-8）ため OOB 読み出しは起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.sum_all_partial_f32)
                    .arg(&a_dev)
                    .arg(&mut partial_dev)
                    .arg(&numel_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべて書き
            // 込み済み、`out_dev` は `reduce_sum_all_finalize_f32` が
            // 単一ブロックの lane 0 で `out[0]` を必ず 1 回書くため
            // `alloc_uninit_f32(1)` を使える。`num_partials_i` は本関数
            // 内の単一の `num_blocks` 由来の値（呼び出し元がずらせない）。
            unsafe {
                self.stream
                    .launch_builder(&self.sum_all_finalize_f32)
                    .arg(&partial_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&num_partials_i)
                    .launch(finalize_cfg)?;
            }

            let host: Vec<f32> = readback(&self.stream, &out_dev.as_view())?;
            Ok(host.first().copied().unwrap_or(0.0))
        })
    }

    /// 全軸 `max`（`fmaxf` 厳密選択。本ファイル冒頭コメント「空縮約の
    /// 意味論」参照）。`numel == 0` は [`CudaError::EmptyReduction`]
    /// を返す（`backend-cpu::reduction::max` と同一の意味論）。
    pub fn run_max_all_f32(&self, a: &[f32]) -> Result<f32, CudaError> {
        let numel = a.len();
        validate_i32_bound(numel, "numel")?;
        if numel == 0 {
            return Err(CudaError::EmptyReduction { op: "max" });
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;

            let num_blocks = reduce_num_blocks(numel as u32);
            let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `run_sum_all_f32` と同一の根拠（`partial` は f32・
            // `-INFINITY` 単位元で全ブロックが 1 回だけ書く）。
            unsafe {
                self.stream
                    .launch_builder(&self.max_all_partial_f32)
                    .arg(&a_dev)
                    .arg(&mut partial_dev.as_view_mut())
                    .arg(&numel_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `run_sum_all_f32` の finalize と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.max_all_finalize_f32)
                    .arg(&partial_dev.as_view())
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&num_partials_i)
                    .launch(finalize_cfg)?;
            }

            let host: Vec<f32> = readback(&self.stream, &out_dev.as_view())?;
            Ok(host.first().copied().unwrap_or(f32::NEG_INFINITY))
        })
    }

    /// 単一軸 `sum`。`inner == 1`（最終軸縮約）は coalesced な
    /// `reduce_sum_lastaxis_f32` へ、それ以外は汎用 `reduce_sum_axis_f32`
    /// へルーティングする（`kernels_reduce.rs` 冒頭コメント参照）。
    /// `axis_len == 0` は出力（`outer*inner` 要素）を全 `0.0` で埋め、
    /// カーネル起動を回避する（`sum` の空縮約は単位元 `0.0` を持つ
    /// ため）。
    pub fn run_sum_axis_f32(
        &self,
        a: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let total_out = validate_axis_layout(a.len(), outer, axis_len, inner)?;
        if axis_len == 0 {
            return Ok(vec![0.0; total_out]);
        }
        if total_out == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(total_out)?;

            let (outer_i, axis_len_i, inner_i) = (outer as i32, axis_len as i32, inner as i32);

            if inner == 1 {
                let num_blocks = (outer as u32).clamp(1, REDUCE_MAX_BLOCKS);
                let cfg = LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (REDUCE_LASTAXIS_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (rows_i, cols_i) = (outer_i, axis_len_i);
                // SAFETY: `a_dev` は `outer*axis_len` 要素（`inner==1`）の
                // H2D 済みバッファ、`out_dev` は `outer` 要素確保済みで
                // カーネルは persistent row loop で `row < rows` を維持
                // したまま各行 1 回だけ `out[row]` を書く
                // （`kernels_reduce.rs::REDUCE_SUM_LASTAXIS_F32` 参照）。
                unsafe {
                    self.stream
                        .launch_builder(&self.sum_lastaxis_f32)
                        .arg(&a_dev)
                        .arg(&mut out_dev.as_view_mut())
                        .arg(&rows_i)
                        .arg(&cols_i)
                        .launch(cfg)?;
                }
            } else {
                let cfg = LaunchConfig {
                    grid_dim: ((total_out as u32).div_ceil(REDUCE_BLOCK_DIM), 1, 1),
                    block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `a_dev` は `outer*axis_len*inner` 要素の H2D 済み
                // バッファ、`out_dev` は `total_out = outer*inner` 要素
                // 確保済みでカーネルは `idx < total`（REQ-8）を維持した
                // まま各出力要素を 1 回だけ書く
                // （`kernels_reduce.rs::REDUCE_SUM_AXIS_F32` 参照）。
                unsafe {
                    self.stream
                        .launch_builder(&self.sum_axis_f32)
                        .arg(&a_dev)
                        .arg(&mut out_dev.as_view_mut())
                        .arg(&outer_i)
                        .arg(&axis_len_i)
                        .arg(&inner_i)
                        .launch(cfg)?;
                }
            }

            readback(&self.stream, &out_dev.as_view())
        })
    }

    /// 単一軸 `max`。[`Self::run_sum_axis_f32`] と同一のルーティング
    /// 方針。`axis_len == 0 && total_out > 0` は
    /// [`CudaError::EmptyReduction`] を返す（`backend-cpu::reduction::
    /// max` と同一の意味論）。`total_out == 0`（出力自体が空）は
    /// vacuous に成功する。
    pub fn run_max_axis_f32(
        &self,
        a: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let total_out = validate_axis_layout(a.len(), outer, axis_len, inner)?;
        if axis_len == 0 {
            if total_out > 0 {
                return Err(CudaError::EmptyReduction { op: "max" });
            }
            return Ok(Vec::new());
        }
        if total_out == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(total_out)?;

            let (outer_i, axis_len_i, inner_i) = (outer as i32, axis_len as i32, inner as i32);

            if inner == 1 {
                let num_blocks = (outer as u32).clamp(1, REDUCE_MAX_BLOCKS);
                let cfg = LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (REDUCE_LASTAXIS_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (rows_i, cols_i) = (outer_i, axis_len_i);
                // SAFETY: `run_sum_axis_f32` の lastaxis 分岐と同一の根拠。
                unsafe {
                    self.stream
                        .launch_builder(&self.max_lastaxis_f32)
                        .arg(&a_dev)
                        .arg(&mut out_dev.as_view_mut())
                        .arg(&rows_i)
                        .arg(&cols_i)
                        .launch(cfg)?;
                }
            } else {
                let cfg = LaunchConfig {
                    grid_dim: ((total_out as u32).div_ceil(REDUCE_BLOCK_DIM), 1, 1),
                    block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `run_sum_axis_f32` の汎用分岐と同一の根拠。
                unsafe {
                    self.stream
                        .launch_builder(&self.max_axis_f32)
                        .arg(&a_dev)
                        .arg(&mut out_dev.as_view_mut())
                        .arg(&outer_i)
                        .arg(&axis_len_i)
                        .arg(&inner_i)
                        .launch(cfg)?;
                }
            }

            readback(&self.stream, &out_dev.as_view())
        })
    }
}

/// `run_*_axis_f32` 共通の起動前検証: `outer`/`axis_len`/`inner` が
/// `i32::MAX` に収まり、`outer*axis_len*inner == a.len()`（呼び出し元の
/// `Tensor` 実体化と `(outer, axis_len, inner)` 分解が矛盾しないこと）を
/// 確認し、`total_out = outer*inner` を返す。
fn validate_axis_layout(
    a_len: usize,
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Result<usize, CudaError> {
    validate_i32_bound(outer, "outer")?;
    validate_i32_bound(axis_len, "axis_len")?;
    validate_i32_bound(inner, "inner")?;

    let expected_len = outer
        .checked_mul(axis_len)
        .and_then(|v| v.checked_mul(inner))
        .ok_or_else(|| CudaError::InvalidReduceShape {
            detail: "validate_axis_layout: outer * axis_len * inner overflow".to_string(),
        })?;
    if expected_len != a_len {
        return Err(CudaError::InvalidReduceShape {
            detail: format!(
                "validate_axis_layout: outer*axis_len*inner ({expected_len}) != a.len() \
                 ({a_len})"
            ),
        });
    }

    let total_out = outer
        .checked_mul(inner)
        .ok_or_else(|| CudaError::InvalidReduceShape {
            detail: "validate_axis_layout: outer * inner overflow".to_string(),
        })?;
    validate_i32_bound(total_out, "total_out")?;
    Ok(total_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_axis_layout_decomposes_middle_axis() {
        // shape [2, 3, 4], axis=1 → outer=2, axis_len=3, inner=4
        let (outer, axis_len, inner) = reduce_axis_layout(&[2, 3, 4], 1).unwrap();
        assert_eq!((outer, axis_len, inner), (2, 3, 4));
    }

    #[test]
    fn reduce_axis_layout_decomposes_last_axis() {
        // shape [64, 784], axis=1 → outer=64, axis_len=784, inner=1
        let (outer, axis_len, inner) = reduce_axis_layout(&[64, 784], 1).unwrap();
        assert_eq!((outer, axis_len, inner), (64, 784, 1));
    }

    #[test]
    fn reduce_axis_layout_decomposes_first_axis() {
        // shape [3, 4096], axis=0 → outer=1, axis_len=3, inner=4096
        let (outer, axis_len, inner) = reduce_axis_layout(&[3, 4096], 0).unwrap();
        assert_eq!((outer, axis_len, inner), (1, 3, 4096));
    }

    #[test]
    fn validate_axis_layout_accepts_matching_len() {
        let total_out = validate_axis_layout(24, 2, 3, 4).unwrap();
        assert_eq!(total_out, 8);
    }

    #[test]
    fn validate_axis_layout_rejects_len_mismatch() {
        let err = validate_axis_layout(23, 2, 3, 4).unwrap_err();
        assert!(matches!(err, CudaError::InvalidReduceShape { .. }));
    }

    #[test]
    fn validate_i32_bound_rejects_exceeding_i32_max() {
        let err = validate_i32_bound(i32::MAX as usize + 1, "numel").unwrap_err();
        assert!(matches!(err, CudaError::InvalidReduceShape { .. }));
    }

    #[test]
    fn validate_i32_bound_accepts_i32_max() {
        assert!(validate_i32_bound(i32::MAX as usize, "numel").is_ok());
    }

    #[test]
    fn reduce_num_blocks_caps_at_max_blocks() {
        assert_eq!(reduce_num_blocks(1), 1);
        assert_eq!(
            reduce_num_blocks(REDUCE_MAX_BLOCKS * REDUCE_BLOCK_DIM * 4),
            REDUCE_MAX_BLOCKS
        );
    }
}
