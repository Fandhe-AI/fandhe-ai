//! `sort`／`topk`（`torch.sort`／`torch.topk` 相当。イシュー #1741）の
//! 起動 API（NVRTC コンパイル・保持・実行）。
//!
//! `unique.rs::CudaUnique` と同じ構成方針を踏襲する: [`CudaSort::new`]
//! が `CudaDevice` から 3 カーネル（`kernels_sort.rs`）を NVRTC
//! コンパイルして保持し、以降は [`CudaSort::run_sort_f32`] へホスト側
//! スライスを渡すだけで GPU 実行できる。`ops.rs::CudaBackendOps::
//! sort`／`topk` から `BackendOps` の実装として呼ばれる。
//!
//! # アルゴリズム
//!
//! [`crate::sort_model`] モジュール doc（鍵設計・ライン分解・
//! ビットニックソートのパディング）が正。本ファイルはその GPU 側実行
//! 手順（build → sort（`k`/`j` の入れ子ループ）→ finalize）を
//! 実装する:
//!
//! 1. `crate::sort_model::plan_sort` で `shape`／`dim` から起動計画
//!    （`outer`／`dim_size`／`inner`／`lines`／`padded`）を導出する
//!    （バックエンド固有上限超過は [`CudaError::SortSizeLimitExceeded`]
//!    として GPU 起動前に返す）。
//! 2. `keys`（`lines * padded` 長。`u64`）・`values`（`lines * out_len`
//!    長。`f32`）・`index`（`lines * out_len` 長。`i32`）を
//!    `alloc_zeros` で確保する（3 バッファともカーネルが全要素を
//!    書き切るためゼロ初期化は無害。プール結線は本イシューのスコープ
//!    外）。
//! 3. `sort_build_keys_u64` → `bitonic_step_u64`（`k`/`j` ループ）→
//!    `sort_finalize_f32` を同一ストリームへ順次投入する（同一
//!    ストリームの FIFO 順序保証によりステップ間の依存が正しく直列化
//!    される。`docs/backend-cuda-async-execution-design.md` §3 と同じ
//!    前提）。
//! 4. `readback` で `values`／`index` を 1 回ずつ D2H する
//!    （`memory.rs::readback` の唯一の同期点契約に従う）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_sort::{self, SORT_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::sort_model::{SortPlan, plan_sort};

/// `value` を添字系カーネル引数（`long long`）向けに `i64` へ変換する
/// （`descending` フラグのみ `int` を直接使うため本関数の対象外）。
/// `usize` が 64bit の実行環境では通常失敗しないが、`gather_scatter.rs::
/// validate_i32_bound` と同じ「型変換失敗を fail-closed で型付き
/// エラーへ変換する」方針を踏襲する。
fn checked_i64(value: usize, name: &str) -> Result<i64, CudaError> {
    i64::try_from(value).map_err(|_| CudaError::InvalidSortShape {
        detail: format!("sort/topk argument does not fit in i64: {name}={value}"),
    })
}

/// 3 カーネル（`sort_build_keys_u64`／`bitonic_step_u64`／
/// `sort_finalize_f32`）のコンパイル済みハンドルを保持する。
pub struct CudaSort {
    stream: Arc<CudaStream>,
    /// `unique.rs::CudaUnique::ordinal` と同じ役割（`Self::
    /// with_driver_call` が `context_cache::with_driver_call` を呼ぶ際の
    /// キー）。
    ordinal: usize,
    build_keys: CudaFunction,
    bitonic_step: CudaFunction,
    finalize: CudaFunction,
}

impl CudaSort {
    /// `device` 上で 3 カーネルを NVRTC コンパイルし保持するハンドルを
    /// 構築する（`unique.rs::CudaUnique::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        // `gather_scatter.rs::CudaGatherScatter::new` と同じ理由で
        // カーネルごとに独立した NVRTC コンパイル＋モジュールロードを
        // 行う（1 ソースへ連結すると `extern "C"` シンボルの衝突・
        // ビルド設定の混在を避けにくいため。`compile_and_load!` は
        // 同ファイルのローカルマクロと同型の複製）。
        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let build_keys =
            compile_and_load!(kernels_sort::SORT_BUILD_KEYS_U64, "sort_build_keys_u64");
        let bitonic_step = compile_and_load!(kernels_sort::BITONIC_STEP_U64, "bitonic_step_u64");
        let finalize = compile_and_load!(kernels_sort::SORT_FINALIZE_F32, "sort_finalize_f32");

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            build_keys,
            bitonic_step,
            finalize,
        })
    }

    /// `CudaSort` の driver 呼び出しを CUDA Graph capture 排他へ参加
    /// させる共通ヘルパー（`unique.rs::CudaUnique::with_driver_call` と
    /// 同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `torch.sort`／`torch.topk`（`sorted=True`）共通の実行本体
    /// （本ファイル冒頭コメント「アルゴリズム」節参照）。`out_len` は
    /// `sort` なら `dim_size`、`topk` なら `k`。呼び出し元（`ops.rs`）
    /// が `shape`（`input.shape()`）・`dim`・`out_len` を事前に検査
    /// （[`fandhe_ai_tensor_core::sort_out_shape`]／
    /// [`fandhe_ai_tensor_core::topk_out_shape`]）済みの契約。
    pub fn run_sort_f32(
        &self,
        input: &[f32],
        shape: &[usize],
        dim: usize,
        descending: bool,
        out_len: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), CudaError> {
        let plan = plan_sort(shape, dim).map_err(|e| match e {
            crate::sort_model::SortPrepareError::DimSizeTooLarge { dim_size } => {
                CudaError::InvalidSortShape {
                    detail: format!("dim_size too large for kernel argument: {dim_size}"),
                }
            }
            crate::sort_model::SortPrepareError::SizeLimitExceeded { total, limit } => {
                CudaError::SortSizeLimitExceeded { total, limit }
            }
        })?;
        let SortPlan {
            dim_size,
            inner,
            lines,
            padded,
            ..
        } = plan;

        // `out_len` はキー配列（`lines * padded` 長）から `values`／
        // `index` を復元するための読み出し本数を決める。本メソッドの
        // doc は「呼び出し元 `ops.rs` が事前検査済みの契約」と記すが、
        // 本メソッド自体は `pub` で crate 内から直接到達しうるため
        // ここでも独立に検証する（Metal 側 `sort.rs::MetalSort::
        // run_sort_f32` に対する PR #1844 codex-review P0 是正の同型
        // 適用）。`out_len > dim_size` を確保・乗算前に拒否しないと、
        // `kernels_sort.rs::SORT_FINALIZE_F32` が `keys[line * padded +
        // o]`（`o < out_len`）で `padded` 境界を超えて読み出しうる
        // （`dim_size <= padded` は常に成り立つため `out_len <=
        // dim_size` の検証で `o < padded` も保証される）。
        if out_len > dim_size {
            return Err(CudaError::InvalidSortShape {
                detail: format!("out_len {out_len} exceeds dim_size {dim_size}"),
            });
        }

        let numel_in = checked_i64(input.len(), "numel_in")?;
        let numel_out = checked_i64(lines * out_len, "numel_out")?;
        let lines_i = checked_i64(lines, "lines")?;
        let dim_size_i = checked_i64(dim_size, "dim_size")?;
        let inner_i = checked_i64(inner, "inner")?;
        let padded_i = checked_i64(padded, "padded")?;
        let out_len_i = checked_i64(out_len, "out_len")?;
        let descending_i: i32 = if descending { 1 } else { 0 };

        if input.len() != lines * dim_size {
            return Err(CudaError::InvalidSortShape {
                detail: format!(
                    "input length {} does not match lines*dim_size {}",
                    input.len(),
                    lines * dim_size
                ),
            });
        }
        let total_keys = lines * padded;
        let total_out = lines * out_len;
        // 空出力（`out_len == 0`。`topk(k=0)` 等）は GPU 起動なしで早期
        // 処理する——`MetalBuffer`／`MetalIndexBuffer` の 0 バイト確保
        // 拒否と同じ理由で、`alloc_zeros::<T>(0)` を避ける安全側の判断
        // （呼び出し元 `ops.rs` は shape 検査の時点で既に空出力を弾く
        // 契約だが、本メソッドは `pub` で `ops.rs` を経由しない直接
        // 呼び出しも構文上可能なため独立に検査する）。
        if total_out == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let mut keys_dev = self.stream.alloc_zeros::<u64>(total_keys)?;
            let mut values_dev = self.stream.alloc_zeros::<f32>(total_out)?;
            let mut index_dev = self.stream.alloc_zeros::<i32>(total_out)?;

            let build_cfg = LaunchConfig {
                grid_dim: ((total_keys as u32).div_ceil(SORT_BLOCK_DIM).max(1), 1, 1),
                block_dim: (SORT_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev` は `input.len()`（`numel_in`）要素・
            // `keys_dev` は `total_keys` 要素確保済み。カーネルは
            // `gid < total`（＝`total_keys`）・`in_pos < numel_in` を
            // 維持したまま読み書きする（`kernels_sort.rs::
            // SORT_BUILD_KEYS_U64` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.build_keys)
                    .arg(&input_dev)
                    .arg(&mut keys_dev)
                    .arg(&lines_i)
                    .arg(&dim_size_i)
                    .arg(&inner_i)
                    .arg(&padded_i)
                    .arg(&descending_i)
                    .arg(&numel_in)
                    .launch(build_cfg)?;
            }

            if padded > 1 {
                let mut k = 2usize;
                while k <= padded {
                    let mut j = k / 2;
                    while j >= 1 {
                        let j_i = checked_i64(j, "j")?;
                        let k_i = checked_i64(k, "k")?;
                        let step_cfg = LaunchConfig {
                            grid_dim: ((total_keys as u32).div_ceil(SORT_BLOCK_DIM).max(1), 1, 1),
                            block_dim: (SORT_BLOCK_DIM, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        // SAFETY: `keys_dev` は `total_keys` 要素確保済み。
                        // カーネルは `gid < total`（＝`total_keys`）・
                        // `i`／`ixj` とも `< padded` を維持したまま
                        // `keys_dev` 内の要素同士を read-modify-write
                        // するのみ（`kernels_sort.rs::BITONIC_STEP_U64`
                        // 参照）。
                        unsafe {
                            self.stream
                                .launch_builder(&self.bitonic_step)
                                .arg(&mut keys_dev)
                                .arg(&j_i)
                                .arg(&k_i)
                                .arg(&padded_i)
                                .arg(&lines_i)
                                .launch(step_cfg)?;
                        }
                        j /= 2;
                    }
                    k *= 2;
                }
            }

            let finalize_cfg = LaunchConfig {
                grid_dim: ((total_out as u32).div_ceil(SORT_BLOCK_DIM).max(1), 1, 1),
                block_dim: (SORT_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev`／`keys_dev`（読み出し専用）と
            // `values_dev`／`index_dev`（`total_out` 要素確保済み）
            // への書き込み。カーネルは `gid < total`（＝
            // `total_out`）・`in_pos < numel_in`・`out_pos <
            // numel_out` を維持する（`kernels_sort.rs::
            // SORT_FINALIZE_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.finalize)
                    .arg(&input_dev)
                    .arg(&keys_dev)
                    .arg(&mut values_dev)
                    .arg(&mut index_dev)
                    .arg(&lines_i)
                    .arg(&dim_size_i)
                    .arg(&inner_i)
                    .arg(&padded_i)
                    .arg(&out_len_i)
                    .arg(&numel_in)
                    .arg(&numel_out)
                    .launch(finalize_cfg)?;
            }

            let values = readback::<f32, _>(&self.stream, &values_dev)?;
            let index = readback::<i32, _>(&self.stream, &index_dev)?;
            Ok((values, index))
        })
    }
}
