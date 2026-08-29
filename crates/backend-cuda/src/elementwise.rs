//! elementwise（`add`／`mul`／`relu`／`exp`／`tanh`）の起動 API（NVRTC
//! コンパイル・保持・実行。イシュー #599）。
//!
//! `gemm.rs::CudaGemm` と同じ構成方針を踏襲する: `CudaElementwise::new` が
//! `CudaDevice` から 5 カーネルを一括 NVRTC コンパイルして保持し、以降は
//! `run_*_f32` へホスト側スライスを渡すだけで GPU 実行できる（H2D 転送 →
//! 起動 → 同期 → D2H 転送を内部で完結させる）。カーネルソース自体は
//! `kernels_elementwise.rs`（NVRTC 文字列埋め込み）に閉じ込め、本モジュール
//! はコンパイル結果（`CudaFunction`）の保持とメモリ転送・起動手続きのみを
//! 扱う（`gemm.rs` 冒頭コメントと同じ責務分離）。
//!
//! `ops.rs::CudaBackendOps` から `BackendOps::add`／`mul`／`relu`／`exp`／
//! `tanh` の実装として呼ばれる。ブロードキャスト対応（NumPy 互換）は
//! `ops.rs` 側が `Tensor::broadcast_with` → `contiguous()` で同一 shape の
//! 密なバッファへ実体化してから本モジュールへ渡す契約（本モジュール自体は
//! 同一長バッファの 1:1 演算のみを扱う。`kernels_elementwise.rs` 冒頭
//! コメント「ブロードキャスト」参照）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_elementwise::{self, EW_BLOCK_DIM};
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// elementwise 演算 1 回あたりのブロック次元（1 次元、`EW_BLOCK_DIM` 幅）。
const EW_BLOCK: (u32, u32, u32) = (EW_BLOCK_DIM, 1, 1);

/// `a_len`／`b_len` が一致し、かつ `i32::MAX` に収まることを検証する
/// （二項演算向け）。
///
/// カーネル引数 `int numel` は C の 32bit 符号付き整数のため、GEMM の
/// `gemm.rs::validate_gemm_dims` と同じ理由で起動前に上限を検査する
/// （OWASP A03。`.claude/rules/security.md`）。`pub(crate)`:
/// 実機非依存の単体テスト（本ファイル末尾 `#[cfg(test)]`）から直接呼べる
/// よう公開範囲をクレート内に限定する。
pub(crate) fn validate_elementwise_binary_dims(
    a_len: usize,
    b_len: usize,
) -> Result<(), CudaError> {
    if a_len != b_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("elementwise length mismatch: a_len={a_len}, b_len={b_len}"),
        });
    }
    validate_elementwise_len(a_len)
}

/// 単項演算向け: 長さが `i32::MAX` に収まることのみを検証する。
pub(crate) fn validate_elementwise_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "elementwise numel must fit in i32 (kernel argument type): numel={len}"
            ),
        });
    }
    Ok(())
}

/// `numel` に対し `EW_BLOCK` を `div_ceil` で包含するグリッド次元を構築する
/// （`gemm.rs::launch_config` と同じ「末尾ブロックの余剰スレッドはカーネル
/// 内境界チェックに委ねる」契約。REQ-8）。
fn elementwise_launch_config(numel: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (numel.div_ceil(EW_BLOCK.0), 1, 1),
        block_dim: EW_BLOCK,
        shared_mem_bytes: 0,
    }
}

/// elementwise 5 カーネル（`add`／`mul`／`relu`／`exp`／`tanh`。いずれも
/// f32）のコンパイル済みハンドルを保持する。
///
/// `stream` は [`CudaDevice`] から `Arc` クローンで受け取る（`gemm.rs` の
/// 共有契約どおり）。`new` 時に 5 カーネルを一括コンパイルするのは
/// `nvrtc::compile_ptx` の呼び出し契約（`gemm.rs::CudaGemm` のドキュメント
/// コメント参照）を守るためであり、`run_*` 呼び出しのたびに再コンパイル
/// しない。
pub struct CudaElementwise {
    stream: Arc<CudaStream>,
    /// 出力バッファのサイズクラス別プール（イシュー #1020・REQ-14）。
    /// `gemm.rs::CudaGemm::allocator` と同一の設計（`crate::pool` 冒頭
    /// コメント参照）。`context_cache::cached_allocator` 経由で
    /// `CudaGemm` と同じ `(ordinal, 既定 stream)` 単位プールを共有する。
    allocator: Arc<CudaAllocator>,
    add_f32: CudaFunction,
    mul_f32: CudaFunction,
    relu_f32: CudaFunction,
    exp_f32: CudaFunction,
    tanh_f32: CudaFunction,
}

impl CudaElementwise {
    /// `device` 上で elementwise 5 カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する。
    ///
    /// 手順: `kernels_elementwise::{EW_ADD_F32,EW_MUL_F32,EW_RELU_F32,
    /// EW_EXP_F32,EW_TANH_F32}` を `device.arch()` 向けに
    /// `nvrtc::compile_ptx` でコンパイル → `device.context().load_module()`
    /// → `load_function(...)`（`gemm.rs::CudaGemm::new` の naive/tiled
    /// 4 カーネルと同一手順）。コンパイル失敗（NVRTC 不在・構文エラー等）は
    /// `CudaError` として早期 return する（naive/tiled の 4 カーネルと同じ
    /// く `#include` を使わず全 compute capability で成立するため、
    /// WMMA(TF32) 系カーネルのような `Option` フィールド化・失敗の退避は
    /// 不要と判断した）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let add_ptx = compile_ptx(kernels_elementwise::EW_ADD_F32, arch)?;
        let mul_ptx = compile_ptx(kernels_elementwise::EW_MUL_F32, arch)?;
        let relu_ptx = compile_ptx(kernels_elementwise::EW_RELU_F32, arch)?;
        let exp_ptx = compile_ptx(kernels_elementwise::EW_EXP_F32, arch)?;
        let tanh_ptx = compile_ptx(kernels_elementwise::EW_TANH_F32, arch)?;

        let add_f32 = device
            .context()
            .load_module(add_ptx)?
            .load_function("ew_add_f32")?;
        let mul_f32 = device
            .context()
            .load_module(mul_ptx)?
            .load_function("ew_mul_f32")?;
        let relu_f32 = device
            .context()
            .load_module(relu_ptx)?
            .load_function("ew_relu_f32")?;
        let exp_f32 = device
            .context()
            .load_module(exp_ptx)?
            .load_function("ew_exp_f32")?;
        let tanh_f32 = device
            .context()
            .load_module(tanh_ptx)?
            .load_function("ew_tanh_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            allocator,
            add_f32,
            mul_f32,
            relu_f32,
            exp_f32,
            tanh_f32,
        })
    }

    /// 二項演算共通の起動手続き（H2D → 起動 → 同期 → D2H）。
    ///
    /// `a.len() == 0`（呼び出し元の shape が空要素）の場合は `gemm.rs`
    /// の `m == 0 || n == 0` 早期 return と同じ理由（0 バイトデバイス確保を
    /// 一部 CUDA driver が拒否しうる）でカーネル起動自体を回避し、空の
    /// 結果を返す。
    fn run_binary(&self, func: &CudaFunction, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        validate_elementwise_binary_dims(a.len(), b.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        // イシュー #1020: 全カーネル（`ew_add_f32` 等）が
        // `if (idx < numel)` ガード内で `out[idx]` を必ず埋める
        // （`kernels_elementwise.rs` 参照）ため `alloc_uninit_f32` を使う。
        let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

        let cfg = elementwise_launch_config(numel as u32);
        let numel_i = numel as i32;

        // SAFETY: カーネル引数（a_dev/b_dev/out_dev・numel_i）は上記で
        // 検証済みの numel と 1:1 対応するデバイスバッファ長・値であり、
        // カーネル内の手動境界チェック（`if (idx < numel)`。
        // `kernels_elementwise.rs` 参照、REQ-8）と合わせて OOB 読み書きが
        // 起きない根拠とする。グリッド次元は `div_ceil` で numel を包含
        // するよう構築しており（`elementwise_launch_config`）、末尾ブロック
        // の余剰スレッドはカーネル内境界チェックで弾かれる。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut out_dev.as_view_mut())
                .arg(&numel_i)
                .launch(cfg)?;
        }
        // 同期点は readback ヘルパーへ集約（#1013）。プール割当ハンドル
        // （`PooledCudaHandle`。イシュー #1020）は `DevicePtr` を直接実装しない
        // ため、論理長ビュー（`as_view()`）を渡す。
        crate::memory::readback(&self.stream, &out_dev.as_view())
    }

    /// 単項演算共通の起動手続き。[`Self::run_binary`] と同一構造。
    fn run_unary(&self, func: &CudaFunction, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        validate_elementwise_len(a.len())?;
        let numel = a.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let a_dev = self.stream.clone_htod(a)?;
        let mut out_dev = self.allocator.alloc_uninit_f32(numel)?;

        let cfg = elementwise_launch_config(numel as u32);
        let numel_i = numel as i32;

        // SAFETY: run_binary と同一の根拠（上記コメント参照）。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&mut out_dev.as_view_mut())
                .arg(&numel_i)
                .launch(cfg)?;
        }
        // 同期点は readback ヘルパーへ集約（#1013）。プール割当ハンドル
        // （`PooledCudaHandle`。イシュー #1020）は `DevicePtr` を直接実装しない
        // ため、論理長ビュー（`as_view()`）を渡す。
        crate::memory::readback(&self.stream, &out_dev.as_view())
    }

    /// `out[i] = a[i] + b[i]`（f32・同一長）。
    pub fn run_add_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_binary(&self.add_f32, a, b)
    }

    /// `out[i] = a[i] * b[i]`（f32・同一長）。
    pub fn run_mul_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_binary(&self.mul_f32, a, b)
    }

    /// `out[i] = max(a[i], 0)`（f32）。
    pub fn run_relu_f32(&self, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_unary(&self.relu_f32, a)
    }

    /// `out[i] = exp(a[i])`（f32、単精度 `expf`）。
    pub fn run_exp_f32(&self, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_unary(&self.exp_f32, a)
    }

    /// `out[i] = tanh(a[i])`（f32、単精度 `tanhf`）。
    pub fn run_tanh_f32(&self, a: &[f32]) -> Result<Vec<f32>, CudaError> {
        self.run_unary(&self.tanh_f32, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_elementwise_binary_dims_accepts_matching_lengths() {
        assert!(validate_elementwise_binary_dims(4, 4).is_ok());
    }

    #[test]
    fn validate_elementwise_binary_dims_rejects_length_mismatch() {
        let err = validate_elementwise_binary_dims(4, 5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_len_rejects_exceeding_i32_max() {
        let err = validate_elementwise_len(i32::MAX as usize + 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_elementwise_len_accepts_i32_max() {
        assert!(validate_elementwise_len(i32::MAX as usize).is_ok());
    }
}
