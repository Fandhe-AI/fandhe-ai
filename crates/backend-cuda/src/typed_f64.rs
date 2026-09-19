//! `TypedOps<f64>` の CUDA 実装（イシュー #2060・親 #1650・
//! `docs/backend-dtype-dispatch-design.md` §4.2「最小集合 8 演算」）。
//!
//! # 方針: ネイティブ CUDA カーネル実装（性能目的なし・数値契約優先）
//!
//! 従来（イシュー #1703）は 8 演算すべてを driver 非接触の
//! `BackendError::Unsupported` で即座に拒否していた（設計 §8「f64 GPU
//! カーネル（CUDA SIMT）: 性能目的がないため優先度は低いが、
//! `Unsupported` から始める既定実装は含めてよい」という引き継ぎ事項）。
//! 本イシューはその引き継ぎを実施し、性能最適化（BLIS packing・split-K
//! 等）を伴わない素朴な CUDA C カーネル（`kernels_typed_f64.rs`）へ
//! 差し替える。数値契約（bit 完全一致／REQ-2 統一複合判定の使い分け）は
//! `kernels_typed_f64.rs` モジュール doc を正とする。
//!
//! # メモリ確保: プール非経由
//!
//! `crate::pool::PoolDtype`（`crate::pool::CudaAllocator`）は現状
//! f32／f16 のみ対応しており f64 への拡張は行わない（f64 に性能目標が
//! ないため）。出力バッファは `self.stream.alloc_zeros::<f64>(numel)`
//! （安全な zero 初期化。`reduce.rs::run_sum_all_f32` の `partial`
//! バッファ確保と同じ判断）で直接確保する。
//!
//! # 構成方針（`elementwise.rs`／`reduce.rs`／`cast.rs` と同型）
//!
//! [`CudaTypedF64::new`] が `CudaDevice` から 12 カーネルを一括 NVRTC
//! コンパイルして保持し、以降は `run_*` へホスト側スライスを渡すだけで
//! GPU 実行できる（H2D 転送 → 起動 → 同期 → D2H 転送を内部で完結させる）。
//! `context_cache::cached_typed_f64` を経由してプロセス内キャッシュから
//! 取得する（`cached_elementwise`／`cached_reduce` と同型）。
//!
//! # スコープ外
//!
//! CUDA `double` 専用の性能最適化（BLIS packing・warp 最適化・split-K
//! 等）・bf16（#1704）・Metal（#1705）・`Var`／`Tape`／VJP・facade 公開面
//! への昇格は対象外（`docs/backend-dtype-dispatch-design.md` §8・親
//! #1650 の後続イシュー分担）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    Tensor, TypedOps, elementwise_out_shape, gemm_out_shape, reduce_out_shape,
};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::elementwise::{validate_elementwise_binary_dims, validate_elementwise_len};
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::kernels_typed_f64::{
    self, TYPED_F64_BLOCK_DIM, TYPED_F64_GEMM_BLOCK_DIM, TYPED_F64_REDUCE_MAX_BLOCKS,
};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::ops::CudaBackendOps;
use crate::reduce::{reduce_axis_layout, validate_axis_layout, validate_i32_bound};

/// `numel` に対し [`TYPED_F64_BLOCK_DIM`] を `div_ceil` で包含する
/// グリッド次元を構築する（`elementwise.rs::elementwise_launch_config`
/// と同じ「末尾ブロックの余剰スレッドはカーネル内境界チェックに委ねる」
/// 契約。REQ-8）。
fn ew_launch_config(numel: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (numel.div_ceil(TYPED_F64_BLOCK_DIM), 1, 1),
        block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `m`／`n` を [`TYPED_F64_GEMM_BLOCK_DIM`] で切り上げ包含するグリッド
/// 次元を構築する（`gemm.rs::launch_config` と同じ考え方）。
fn gemm_launch_config(m: u32, n: u32) -> LaunchConfig {
    let block_dim = TYPED_F64_GEMM_BLOCK_DIM;
    LaunchConfig {
        grid_dim: (n.div_ceil(block_dim.0), m.div_ceil(block_dim.1), 1),
        block_dim,
        shared_mem_bytes: 0,
    }
}

/// 全軸縮約 1 段目の起動ブロック数を決定する
/// （`reduce.rs::reduce_num_blocks` と同一契約）。
fn reduce_num_blocks(numel: u32) -> u32 {
    numel
        .div_ceil(TYPED_F64_BLOCK_DIM)
        .min(TYPED_F64_REDUCE_MAX_BLOCKS)
}

/// `TypedOps<f64>` 8 演算・12 カーネル（`kernels_typed_f64.rs`）の
/// コンパイル済みハンドルを保持する（`elementwise::CudaElementwise`／
/// `reduce::CudaReduce` と同型の構成）。
pub struct CudaTypedF64 {
    stream: Arc<CudaStream>,
    /// `Self::with_driver_call` が `context_cache::with_driver_call`
    /// を呼ぶ際のキー（`CudaElementwise::ordinal` と同じ役割）。
    ordinal: usize,
    gemm_f64: CudaFunction,
    add_f64: CudaFunction,
    mul_f64: CudaFunction,
    relu_f64: CudaFunction,
    exp_f64: CudaFunction,
    tanh_f64: CudaFunction,
    sum_all_partial_f64: CudaFunction,
    sum_all_finalize_f64: CudaFunction,
    sum_axis_f64: CudaFunction,
    max_all_partial_f64: CudaFunction,
    max_all_finalize_f64: CudaFunction,
    max_axis_f64: CudaFunction,
}

impl CudaTypedF64 {
    /// `device` 上で 12 カーネルを NVRTC コンパイルし保持するハンドルを
    /// 構築する（`reduce::CudaReduce::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let gemm_f64 = compile_and_load!(kernels_typed_f64::GEMM_NAIVE_F64, "gemm_naive_f64");
        let add_f64 = compile_and_load!(kernels_typed_f64::EW_ADD_F64, "ew_add_f64");
        let mul_f64 = compile_and_load!(kernels_typed_f64::EW_MUL_F64, "ew_mul_f64");
        let relu_f64 = compile_and_load!(kernels_typed_f64::EW_RELU_F64, "ew_relu_f64");
        let exp_f64 = compile_and_load!(kernels_typed_f64::EW_EXP_F64, "ew_exp_f64");
        let tanh_f64 = compile_and_load!(kernels_typed_f64::EW_TANH_F64, "ew_tanh_f64");
        let sum_all_partial_f64 = compile_and_load!(
            kernels_typed_f64::REDUCE_SUM_ALL_PARTIAL_F64,
            "reduce_sum_all_partial_f64"
        );
        let sum_all_finalize_f64 = compile_and_load!(
            kernels_typed_f64::REDUCE_SUM_ALL_FINALIZE_F64,
            "reduce_sum_all_finalize_f64"
        );
        let sum_axis_f64 = compile_and_load!(
            kernels_typed_f64::REDUCE_SUM_AXIS_F64,
            "reduce_sum_axis_f64"
        );
        let max_all_partial_f64 = compile_and_load!(
            kernels_typed_f64::REDUCE_MAX_ALL_PARTIAL_F64,
            "reduce_max_all_partial_f64"
        );
        let max_all_finalize_f64 = compile_and_load!(
            kernels_typed_f64::REDUCE_MAX_ALL_FINALIZE_F64,
            "reduce_max_all_finalize_f64"
        );
        let max_axis_f64 = compile_and_load!(
            kernels_typed_f64::REDUCE_MAX_AXIS_F64,
            "reduce_max_axis_f64"
        );

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            gemm_f64,
            add_f64,
            mul_f64,
            relu_f64,
            exp_f64,
            tanh_f64,
            sum_all_partial_f64,
            sum_all_finalize_f64,
            sum_axis_f64,
            max_all_partial_f64,
            max_all_finalize_f64,
            max_axis_f64,
        })
    }

    /// `CudaTypedF64` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`CudaElementwise::with_driver_call` と
    /// 同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// naive GEMM（f64）。`gemm.rs::validate_gemm_dims`（dtype 非依存）を
    /// 再利用して起動前に形状を検証する（`gemm.rs::run_naive_f32` と
    /// 同一手順）。`m == 0 || n == 0` は空、`k == 0` は全 0 出力を
    /// カーネル起動なしで返す（`gemm.rs::run_f32_kernel` と同じ
    /// 0 バイト確保回避の理由）。
    pub fn run_gemm(
        &self,
        a: &[f64],
        b: &[f64],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f64>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f64; (m as usize) * (n as usize)]);
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let b_dev = self.stream.clone_htod(b)?;
            let mut c_dev = self
                .stream
                .alloc_zeros::<f64>((m as usize) * (n as usize))?;

            let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
            let cfg = gemm_launch_config(m, n);

            // SAFETY: `a_dev`（`a.len()` 要素）／`b_dev`（`b.len()` 要素）は
            // 直上の `clone_htod` で H2D 済み、`c_dev`（`m*n` 要素）は
            // 直上の `alloc_zeros` で確保済み。`validate_gemm_dims` により
            // `a.len() == m*k`／`b.len() == k*n` を検証済みのため、
            // カーネル内 `if (row < m && col < n)`（REQ-8）の境界チェック
            // と合わせて OOB 読み書きは起きない。
            unsafe {
                self.stream
                    .launch_builder(&self.gemm_f64)
                    .arg(&a_dev)
                    .arg(&b_dev)
                    .arg(&mut c_dev)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&k_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &c_dev)
        })
    }

    /// 二項演算共通の起動手続き（`add`／`mul`）。
    fn run_binary(&self, func: &CudaFunction, a: &[f64], b: &[f64]) -> Result<Vec<f64>, CudaError> {
        validate_elementwise_binary_dims(a.len(), b.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let b_dev = self.stream.clone_htod(b)?;
            let mut out_dev = self.stream.alloc_zeros::<f64>(numel)?;

            let cfg = ew_launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: `a_dev`／`b_dev`（`numel` 要素）は直上で H2D 済み、
            // `out_dev`（`numel` 要素）は直上で確保済み。カーネル内
            // `if (idx < numel)`（REQ-8）と合わせて OOB は起きない。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&a_dev)
                    .arg(&b_dev)
                    .arg(&mut out_dev)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev)
        })
    }

    /// 単項演算共通の起動手続き（`relu`／`exp`／`tanh`）。
    fn run_unary(&self, func: &CudaFunction, a: &[f64]) -> Result<Vec<f64>, CudaError> {
        validate_elementwise_len(a.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let mut out_dev = self.stream.alloc_zeros::<f64>(numel)?;

            let cfg = ew_launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: `run_binary` と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&a_dev)
                    .arg(&mut out_dev)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev)
        })
    }

    /// `add`。[`Self::run_binary`] への薄い委譲。
    pub fn run_add(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, CudaError> {
        self.run_binary(&self.add_f64, a, b)
    }

    /// `mul`。同上。
    pub fn run_mul(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>, CudaError> {
        self.run_binary(&self.mul_f64, a, b)
    }

    /// `relu`。[`Self::run_unary`] への薄い委譲。
    pub fn run_relu(&self, a: &[f64]) -> Result<Vec<f64>, CudaError> {
        self.run_unary(&self.relu_f64, a)
    }

    /// `exp`。同上。
    pub fn run_exp(&self, a: &[f64]) -> Result<Vec<f64>, CudaError> {
        self.run_unary(&self.exp_f64, a)
    }

    /// `tanh`。同上。
    pub fn run_tanh(&self, a: &[f64]) -> Result<Vec<f64>, CudaError> {
        self.run_unary(&self.tanh_f64, a)
    }

    /// 全軸 `sum`（`kernels_typed_f64.rs` 冒頭コメント「REQ-2 複合判定を
    /// 適用する」節）。`numel == 0` はカーネル起動を回避し `0.0` を返す
    /// （`backend-cpu::typed_f64::sum_f64` の空縮約契約と同一）。
    pub fn run_sum_all(&self, a: &[f64]) -> Result<f64, CudaError> {
        let numel = a.len();
        validate_i32_bound(numel, "numel")?;
        if numel == 0 {
            return Ok(0.0);
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;

            let num_blocks = reduce_num_blocks(numel as u32);
            let mut partial_dev = self.stream.alloc_zeros::<f64>(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `a_dev` は `numel` 要素の H2D 済みバッファ、
            // `partial_dev` は `num_blocks` 要素確保済みでカーネルが
            // `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ書く
            // （`kernels_typed_f64.rs::REDUCE_SUM_ALL_PARTIAL_F64` 参照）。
            // grid-stride ループは `idx < numel`（REQ-8）を維持する。
            unsafe {
                self.stream
                    .launch_builder(&self.sum_all_partial_f64)
                    .arg(&a_dev)
                    .arg(&mut partial_dev)
                    .arg(&numel_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.stream.alloc_zeros::<f64>(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべて
            // 書き込み済み、`out_dev` は単一ブロックの lane 0 が
            // `out[0]` を必ず 1 回書くため確保のみで足りる。
            unsafe {
                self.stream
                    .launch_builder(&self.sum_all_finalize_f64)
                    .arg(&partial_dev)
                    .arg(&mut out_dev)
                    .arg(&num_partials_i)
                    .launch(finalize_cfg)?;
            }

            let host: Vec<f64> = readback(&self.stream, &out_dev)?;
            Ok(host.first().copied().unwrap_or(0.0))
        })
    }

    /// 全軸 `max`（REQ-2 複合判定対象）。`numel == 0` は
    /// [`CudaError::EmptyReduction`] を返す（`backend-cpu::typed_f64::
    /// max_f64` と同一の意味論）。
    pub fn run_max_all(&self, a: &[f64]) -> Result<f64, CudaError> {
        let numel = a.len();
        validate_i32_bound(numel, "numel")?;
        if numel == 0 {
            return Err(CudaError::EmptyReduction { op: "max" });
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;

            let num_blocks = reduce_num_blocks(numel as u32);
            let mut partial_dev = self.stream.alloc_zeros::<f64>(num_blocks as usize)?;

            let numel_i = numel as i32;
            let partial_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `run_sum_all` と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.max_all_partial_f64)
                    .arg(&a_dev)
                    .arg(&mut partial_dev)
                    .arg(&numel_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.stream.alloc_zeros::<f64>(1)?;
            let num_partials_i = num_blocks as i32;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `run_sum_all` の finalize と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.max_all_finalize_f64)
                    .arg(&partial_dev)
                    .arg(&mut out_dev)
                    .arg(&num_partials_i)
                    .launch(finalize_cfg)?;
            }

            let host: Vec<f64> = readback(&self.stream, &out_dev)?;
            Ok(host.first().copied().unwrap_or(f64::NEG_INFINITY))
        })
    }

    /// 単一軸 `sum`（bit 完全一致を狙う）。`axis_len == 0` は出力
    /// （`outer*inner` 要素）を全 `0.0` で埋め、カーネル起動を回避する
    /// （`sum` の空縮約は単位元 `0.0` を持つため。`reduce.rs::
    /// run_sum_axis_f32` と同一契約）。
    pub fn run_sum_axis(
        &self,
        a: &[f64],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f64>, CudaError> {
        let total_out = validate_axis_layout(a.len(), outer, axis_len, inner)?;
        if axis_len == 0 {
            return Ok(vec![0.0; total_out]);
        }
        if total_out == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let mut out_dev = self.stream.alloc_zeros::<f64>(total_out)?;

            let (outer_i, axis_len_i, inner_i) = (outer as i32, axis_len as i32, inner as i32);
            let cfg = LaunchConfig {
                grid_dim: ((total_out as u32).div_ceil(TYPED_F64_BLOCK_DIM), 1, 1),
                block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `a_dev` は `outer*axis_len*inner` 要素の H2D 済み
            // バッファ、`out_dev` は `total_out = outer*inner` 要素確保
            // 済みでカーネルは `idx < total`（REQ-8）を維持したまま各
            // 出力要素を 1 回だけ書く。
            unsafe {
                self.stream
                    .launch_builder(&self.sum_axis_f64)
                    .arg(&a_dev)
                    .arg(&mut out_dev)
                    .arg(&outer_i)
                    .arg(&axis_len_i)
                    .arg(&inner_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev)
        })
    }

    /// 単一軸 `max`（bit 完全一致を狙う）。[`Self::run_sum_axis`] と
    /// 同一のルーティング方針・空縮約契約（`axis_len` がゼロかつ
    /// `total_out` が正の場合は [`CudaError::EmptyReduction`]。
    /// `total_out` がゼロの場合は vacuous に成功）。
    pub fn run_max_axis(
        &self,
        a: &[f64],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<f64>, CudaError> {
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
            let mut out_dev = self.stream.alloc_zeros::<f64>(total_out)?;

            let (outer_i, axis_len_i, inner_i) = (outer as i32, axis_len as i32, inner as i32);
            let cfg = LaunchConfig {
                grid_dim: ((total_out as u32).div_ceil(TYPED_F64_BLOCK_DIM), 1, 1),
                block_dim: (TYPED_F64_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `run_sum_axis` と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.max_axis_f64)
                    .arg(&a_dev)
                    .arg(&mut out_dev)
                    .arg(&outer_i)
                    .arg(&axis_len_i)
                    .arg(&inner_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev)
        })
    }
}

/// [`CudaBackendOps::sum`／`max`（f64 版）] が `sum`／`max` のどちらを
/// 実行するかを選ぶ内部専用の選択子（`ops.rs::ReduceKind` の f64 版。
/// `tensor-core` 公開 API の一部ではない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceKindF64 {
    Sum,
    Max,
}

impl TypedOps<f64> for CudaBackendOps {
    /// `CudaTypedF64::run_gemm` への薄いパススルー。driver に触れる前に
    /// `gemm_out_shape`（`tensor-core`）で shape 検証・`u32` 変換を行う
    /// （`typed_f16.rs::TypedOps::<f16>::gemm` と同型の「driver 呼び出し
    /// 前に事前検証」契約）。
    fn gemm(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        let out_shape =
            gemm_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;

        let m = u32::try_from(a.shape()[0]).map_err(|_| {
            BackendError::KernelLaunchFailed("gemm: m exceeds u32 range".to_string())
        })?;
        let k = u32::try_from(a.shape()[1]).map_err(|_| {
            BackendError::KernelLaunchFailed("gemm: k exceeds u32 range".to_string())
        })?;
        let n = u32::try_from(b.shape()[1]).map_err(|_| {
            BackendError::KernelLaunchFailed("gemm: n exceeds u32 range".to_string())
        })?;

        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        let typed = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_typed_f64(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || typed.run_gemm(a_slice, b_slice, m, n, k),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 二項 elementwise 共通のディスパッチ（`add`）。`ops.rs::
    /// CudaBackendOps::elementwise_binary`（f32 版）と同型: ブロード
    /// キャスト → `contiguous()` 実体化 → `CudaTypedF64` 取得 → 実行。
    fn add(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        self.elementwise_binary_f64(a, b, |t, a_s, b_s| t.run_add(a_s, b_s))
    }

    /// `mul`。同上。
    fn mul(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        self.elementwise_binary_f64(a, b, |t, a_s, b_s| t.run_mul(a_s, b_s))
    }

    /// `relu`。内部ヘルパー `unary_f64` への委譲。
    fn relu(&self, a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        self.unary_f64(a, |t, a_s| t.run_relu(a_s))
    }

    /// `exp`。同上。
    fn exp(&self, a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        self.unary_f64(a, |t, a_s| t.run_exp(a_s))
    }

    /// `tanh`。同上。
    fn tanh(&self, a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        self.unary_f64(a, |t, a_s| t.run_tanh(a_s))
    }

    /// 全軸・単一軸 `sum`。内部ヘルパー `reduce_dispatch_f64` への委譲。
    fn sum(&self, a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        self.reduce_dispatch_f64(a, dim, ReduceKindF64::Sum)
    }

    /// 全軸・単一軸 `max`。同上。
    fn max(&self, a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        self.reduce_dispatch_f64(a, dim, ReduceKindF64::Max)
    }
}

impl CudaBackendOps {
    /// `CudaTypedF64` スイートをプロセス内キャッシュ経由で取得する
    /// （`ops.rs::CudaBackendOps::elementwise_binary` の `cached_elementwise`
    /// 取得と同型）。
    fn cached_typed_f64_suite(&self) -> Result<Arc<CudaTypedF64>, BackendError> {
        self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_typed_f64(&device)
            },
        )
    }

    /// 二項 elementwise 共通のディスパッチ本体（`add`／`mul` 共有）。
    /// `ops.rs::CudaBackendOps::elementwise_binary`（f32 版）の f64 版。
    fn elementwise_binary_f64(
        &self,
        a: &Tensor<f64>,
        b: &Tensor<f64>,
        run: impl FnOnce(&CudaTypedF64, &[f64], &[f64]) -> Result<Vec<f64>, CudaError>,
    ) -> Result<Tensor<f64>, BackendError> {
        // `elementwise_out_shape` は `Tensor::broadcast_with` と同じ規則
        // で出力 shape のみを算出する（実際のブロードキャストは
        // `broadcast_with` に委ねる。f32 版が `broadcast_with` の戻り値
        // から shape を得ているのに対し、ここでは事前に検証してから
        // 進める構成にして呼び出し順を明確にする）。
        elementwise_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        let (a_bc, b_bc) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let out_shape = a_bc.shape().to_vec();

        let a_owned = a_bc.contiguous();
        let b_owned = b_bc.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: rhs not contiguous".into())
        })?;

        let typed = self.cached_typed_f64_suite()?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&typed, a_slice, b_slice),
        )?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 単項 elementwise 共通のディスパッチ本体（`relu`／`exp`／`tanh`
    /// 共有）。
    fn unary_f64(
        &self,
        a: &Tensor<f64>,
        run: impl FnOnce(&CudaTypedF64, &[f64]) -> Result<Vec<f64>, CudaError>,
    ) -> Result<Tensor<f64>, BackendError> {
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: input not contiguous".into())
        })?;

        let typed = self.cached_typed_f64_suite()?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || run(&typed, a_slice),
        )?;
        Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// `sum`／`max`（f64 版）共通のディスパッチ本体。`ops.rs::
    /// CudaBackendOps::reduce_dispatch`（f32 版）と同型: `reduce_out_shape`
    /// で `dim` の範囲検査・出力 shape を導出した後、`a.contiguous()` で
    /// 密なバッファへ実体化してから渡す。エラー写像は `ops.rs::
    /// map_reduce_error`（f32 版と共用。`CudaError::EmptyReduction`／
    /// `InvalidReduceShape` は dtype 非依存の意味論のため）。
    fn reduce_dispatch_f64(
        &self,
        a: &Tensor<f64>,
        dim: Option<usize>,
        kind: ReduceKindF64,
    ) -> Result<Tensor<f64>, BackendError> {
        let out_shape = reduce_out_shape(a.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("reduce: input not contiguous".into())
        })?;

        let typed = self.cached_typed_f64_suite()?;

        let data = match dim {
            None => {
                let value =
                    self.with_driver_call(&[], crate::ops::map_reduce_error, || match kind {
                        ReduceKindF64::Sum => typed.run_sum_all(a_slice),
                        ReduceKindF64::Max => typed.run_max_all(a_slice),
                    })?;
                vec![value]
            }
            Some(axis) => {
                let (outer, axis_len, inner) = reduce_axis_layout(a_owned.shape(), axis)
                    .map_err(crate::ops::map_reduce_error)?;
                self.with_driver_call(&[], crate::ops::map_reduce_error, || match kind {
                    ReduceKindF64::Sum => typed.run_sum_axis(a_slice, outer, axis_len, inner),
                    ReduceKindF64::Max => typed.run_max_axis(a_slice, outer, axis_len, inner),
                })?
            }
        };
        Tensor::new(data, &out_shape).map_err(BackendError::ShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::BackendOps;

    fn ops() -> CudaBackendOps {
        // ordinal は driver に触れない検証のみ実行する限り任意の値でよい。
        CudaBackendOps::new(0)
    }

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    /// `typed_ops_f64()` accessor が `Some` を返す（結線済み）ことを
    /// 確認する。
    #[test]
    fn typed_ops_f64_accessor_is_some() {
        let c = ops();
        assert!(BackendOps::typed_ops_f64(&c).is_some());
    }

    /// shape 不一致は driver に一切触れる前に `ShapeMismatch` を返す
    /// （`CudaUnavailable` にはならない。ordinal 0 が driver 不在環境
    /// でも本テストは成立する）。
    #[test]
    fn gemm_rejects_shape_mismatch_before_touching_driver() {
        let c = ops();
        let a = t(&[1.0, 2.0, 3.0], &[1, 3]);
        let b = t(&[1.0, 2.0], &[2, 1]);
        let err = TypedOps::<f64>::gemm(&c, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    /// `add`／`mul` は shape 不一致（ブロードキャスト不能）を driver に
    /// 触れる前に `ShapeMismatch` として拒否する。
    #[test]
    fn add_mul_reject_shape_mismatch_before_touching_driver() {
        let c = ops();
        let a = t(&[1.0, -2.0, 3.0], &[3]);
        let b = t(&[3.0, 4.0], &[2]);
        let add_err = TypedOps::<f64>::add(&c, &a, &b).unwrap_err();
        assert!(matches!(add_err, BackendError::ShapeMismatch(_)));
        let mul_err = TypedOps::<f64>::mul(&c, &a, &b).unwrap_err();
        assert!(matches!(mul_err, BackendError::ShapeMismatch(_)));
    }

    /// `sum`／`max` は縮約軸 `dim` の範囲検査が driver に触れる前に行われ、
    /// rank を超える `dim` は `ShapeMismatch` を返す。
    #[test]
    fn sum_max_reject_out_of_range_dim_before_touching_driver() {
        let c = ops();
        let a = t(&[1.0, -2.0], &[2]);
        let sum_err = TypedOps::<f64>::sum(&c, &a, Some(5)).unwrap_err();
        assert!(matches!(sum_err, BackendError::ShapeMismatch(_)));
        let max_err = TypedOps::<f64>::max(&c, &a, Some(5)).unwrap_err();
        assert!(matches!(max_err, BackendError::ShapeMismatch(_)));
    }

    /// `relu`／`exp`／`tanh` は単項演算で shape 検証点を持たないため、
    /// 有効な入力に対する呼び出しが driver 不在環境で
    /// `BackendError::CudaUnavailable` 系のエラーのみを返す（それ以外の
    /// 種別を返さない）ことを確認する（`typed_f16.rs` の同型テストと
    /// 同じ意図。実際の数値は GB10 実機 `#[ignore]` テストへ引き継ぐ）。
    #[test]
    fn relu_exp_tanh_accept_valid_input_without_unexpected_error_kind() {
        let ops = ops();
        let a = t(&[1.0, -2.0], &[2]);
        for result in [
            TypedOps::<f64>::relu(&ops, &a),
            TypedOps::<f64>::exp(&ops, &a),
            TypedOps::<f64>::tanh(&ops, &a),
        ] {
            if let Err(e) = result {
                assert!(
                    matches!(
                        e,
                        BackendError::CudaUnavailable(_) | BackendError::KernelLaunchFailed(_)
                    ),
                    "unexpected error variant for valid-shape input: {e:?}"
                );
            }
        }
    }
}
