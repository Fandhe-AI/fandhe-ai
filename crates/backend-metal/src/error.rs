//! `backend-metal` 公開入口の型付きエラー。
//!
//! デバイス取得・コマンドキュー生成・バッファ確保の失敗を `Option`
//! （PoC-v2-4 の `MetalGemm::new` 等が返していた形）ではなく
//! `Result<_, MetalError>` として呼び出し元（TASK-1.8b/c の GEMM・
//! simdgroup カーネル実装。#39・#40）へ伝える。本番経路で `unwrap()` /
//! `expect()` を使わない方針（`.claude/rules/coding-rust.md`）に従い、
//! [`crate::context`]・[`crate::buffer`] はここで定義するバリアントのみ
//! を返す。

use std::fmt;

/// `backend-metal` の基盤層（デバイス・キュー・バッファ）で発生しうる
/// エラー。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、TASK-1.8b 以降でパイプライン
/// 構築・ディスパッチ関連のバリアントが増えても呼び出し側の網羅的
/// match を破壊しないため（`backend-cpu::GemmError` と同方針。
/// `crates/backend-cpu/src/gemm.rs` 参照）。
#[non_exhaustive]
#[derive(Debug)]
pub enum MetalError {
    /// `MTLCreateSystemDefaultDevice` が `None` を返した
    /// （Metal 非対応環境・GPU 無効化等）。
    DeviceUnavailable,
    /// `MTLDevice::newCommandQueue` が `None` を返した。
    CommandQueueCreation,
    /// `MTLDevice::newBufferWithBytes_length_options` /
    /// `newBufferWithLength_options` が `None` を返した
    /// （メモリ不足等。要求バイト数を診断用に保持する）。
    BufferAllocation { bytes: usize },
    /// バッファ長（要素数 × `size_of::<f32>()`）の算出が `usize` の
    /// 範囲でオーバーフローする（`checked_mul` によりアクセス前に検出
    /// する。OWASP A03 観点。`.claude/rules/security.md`。将来
    /// safetensors/ONNX 由来の形状がここへ流入しうるための前段検証）。
    AllocationSizeOverflow { len: usize },
    /// バッファ確保時に長さ 0 が渡された（0 バイトバッファは Metal 側の
    /// 挙動が不定であり、呼び出し側の形状検証漏れを早期に拒否する）。
    ZeroLengthAllocation,
    /// `MTLCommandQueue::commandBuffer` が `None` を返した。
    CommandBufferCreation,
    /// `MTLCommandBuffer::computeCommandEncoder` が `None` を返した。
    ComputeEncoderCreation,
    /// `waitUntilCompleted()` 完了後、コマンドバッファの `status` が
    /// `MTLCommandBufferStatus::Error` だった（GPU 側の fault・OOM・
    /// discarded work 等）。`commit()` 自体は成功として返るため、
    /// [`crate::context::MetalContext::dispatch_sync`] は完了後にこの
    /// 状態を確認しない限り GPU 側の失敗を `Ok(())` として握り潰して
    /// しまう（今後の GEMM 実装で出力バッファの古い／不完全な内容を
    /// 読む無言の数値誤りにつながるため、型付きエラーとして呼び出し元
    /// へ伝える）。`message` は `MTLCommandBuffer::error()` の
    /// `NSError` から得た診断用の文字列表現。
    CommandBufferExecutionFailed { message: String },
    /// `MTLDevice::newLibraryWithSource_options_error`（`shaders/gemm.metal`
    /// の実行時コンパイル。TASK-1.8b・#39）が失敗した。`message` は
    /// `NSError` の `localizedDescription`（構文エラー等の診断文字列）。
    LibraryCompilation { message: String },
    /// `MTLLibrary::newFunctionWithName` が `name` に対して `None` を
    /// 返した（`shaders/gemm.metal` 内の関数名との不一致。呼び出し元の
    /// 実装誤りであり通常到達しないが、`unwrap`/`expect` を避けるため
    /// 型付きエラーとして表現する）。
    FunctionNotFound { name: &'static str },
    /// `MTLDevice::newComputePipelineStateWithFunction_error` が失敗した。
    /// `message` は `NSError` の `localizedDescription`。
    PipelineCreation { message: String },
    /// GEMM 公開入口（[`crate::gemm`]）の形状検証で `m`・`n`・`k` の
    /// いずれかが 0 と判定された（`fandhe_ai_backend_cpu::gemm::GemmError` の
    /// `ZeroBlockSize` 相当。0 次元は Metal ディスパッチ・境界チェックの
    /// 前提を崩すため FFI 呼び出し前に拒否する）。
    ZeroDimension { m: usize, n: usize, k: usize },
    /// `a`（長さ `m*k` 期待）の要素数が一致しない。
    ALenMismatch { expected: usize, actual: usize },
    /// `b`（長さ `k*n` 期待）の要素数が一致しない。
    BLenMismatch { expected: usize, actual: usize },
    /// `c`（長さ `m*n` 期待）の要素数が一致しない。`crate::gemm::
    /// MetalGemm::dispatch_f16_prepared_unverified` が呼び出し元から渡された
    /// `c_buf`（`MetalHalfBuffer`）の実長を検証する際に使う（PR #346
    /// codex-review P1-1 指摘。公開コンストラクタで任意長のバッファを
    /// 渡せるため、エンコード前に厳密な長さ検証を行う必要がある）。
    CLenMismatch { expected: usize, actual: usize },
    /// `crate::gemm::MetalGemm::dispatch_f16_prepared_unverified` の実効
    /// 次元（`m_eff`/`n_eff`/`k_eff`）のいずれかが 8 の倍数でない
    /// （PR #346 codex-review P1-1 指摘。`shaders/gemm.metal` の
    /// `gemm_simdgroup_f16` は 1 threadgroup = C の 8×8 タイル 1 つを
    /// 前提とし、grid 計算（`crate::gemm::encode_dispatch_f16` の
    /// `dims.n / 8`・`dims.m / 8`）が非 8 倍数では末尾タイルを黙って
    /// 計算しない。`dispatch_f16_unverified` 経由（`pad8` 済み）では常に
    /// 満たされるが、`dispatch_f16_prepared_unverified` を直接呼ぶ経路
    /// 向けに明示検証する）。
    NotEightAligned {
        m_eff: usize,
        n_eff: usize,
        k_eff: usize,
    },
    /// `m*k`・`k*n`・`m*n` のいずれかが `usize` の範囲でオーバーフローする
    /// （`checked_mul` によりアクセス前に検出する。OWASP A03 観点）。
    DimProductOverflow,
    /// `m`・`n`・`k` のいずれかが `u32::MAX` を超え、`shaders/gemm.metal`
    /// の `Dims`（`uint` 3 個）へキャストできない（cast 前検証）。
    DimensionExceedsU32 { m: usize, n: usize, k: usize },
    /// `memory.rs::MetalMemory::download_inner` で `Tensor::new` に渡した
    /// `buffer.shape()` と実際に読み出したデータ長が不整合だった（通常
    /// 到達しない防御的経路。`backend-cuda::CudaError::InvalidShape` と
    /// 同種）。`detail` は元の `ShapeError` の `Display` 文字列表現を
    /// 保持し、`MetalError::BufferAllocation { bytes: 0 }` に化けて
    /// 実態と異なるエラー種別を報告しないようにする（レビュー指摘対応。
    /// upload_inner 側の同種到達不能パスは `BufferAllocation` を流用
    /// しているが、こちらは shape 不整合の詳細を保持する必要があるため
    /// 専用 variant とする）。
    ShapeMismatch { detail: String },
    /// `crate::row_kernel::validate_row_kernel_launch`（RMSNorm・softmax
    /// 共通のホスト側 fail-closed 検証。イシュー #604）が拒否した形状・
    /// `eps` 値。`detail` は元の
    /// `row_kernel::RowKernelValidationError` の `Display` 文字列表現。
    InvalidRowKernelShape { detail: String },
    /// `crate::rmsnorm::MetalRmsNorm::new`／`crate::softmax::MetalSoftmax::new`
    /// が構築直後に検証する `MTLComputePipelineState::threadExecutionWidth`
    /// が期待値（32。1 threadgroup = 1 simdgroup 固定の前提）と一致しない
    /// （イシュー #604 実装計画 §4.1「ホスト側で
    /// `threadExecutionWidth == 32` を起動前に検証」。デバイス・ドライバの
    /// 想定外挙動を fail-closed で検出する）。
    UnexpectedThreadExecutionWidth { expected: usize, actual: usize },
    /// elementwise（`crate::elementwise`）・`gemm_bias_act` 融合カーネル
    /// （`crate::gemm::MetalGemm::run_tiled_bias_act_f32`）の起動前 shape
    /// 検証が拒否した（イシュー #605。CUDA 側
    /// `CudaError::InvalidElementwiseShape` と同じ役割）。`detail` に
    /// 具体的な不整合内容（長さ不一致等）を保持する。
    InvalidElementwiseShape { detail: String },
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::DeviceUnavailable => {
                write!(
                    f,
                    "MTLCreateSystemDefaultDevice returned None (no Metal device available)"
                )
            }
            MetalError::CommandQueueCreation => {
                write!(f, "MTLDevice::newCommandQueue returned None")
            }
            MetalError::BufferAllocation { bytes } => {
                write!(f, "Metal buffer allocation failed for {bytes} bytes")
            }
            MetalError::AllocationSizeOverflow { len } => {
                write!(
                    f,
                    "buffer byte length overflows usize for len={len} elements"
                )
            }
            MetalError::ZeroLengthAllocation => {
                write!(f, "buffer allocation requested with zero length")
            }
            MetalError::CommandBufferCreation => {
                write!(f, "MTLCommandQueue::commandBuffer returned None")
            }
            MetalError::ComputeEncoderCreation => {
                write!(f, "MTLCommandBuffer::computeCommandEncoder returned None")
            }
            MetalError::CommandBufferExecutionFailed { message } => {
                write!(
                    f,
                    "Metal command buffer completed with MTLCommandBufferStatus::Error: {message}"
                )
            }
            MetalError::LibraryCompilation { message } => {
                write!(f, "MSL library compilation failed: {message}")
            }
            MetalError::FunctionNotFound { name } => {
                write!(
                    f,
                    "MTLLibrary::newFunctionWithName returned None for \"{name}\""
                )
            }
            MetalError::PipelineCreation { message } => {
                write!(f, "MTLComputePipelineState creation failed: {message}")
            }
            MetalError::ZeroDimension { m, n, k } => {
                write!(f, "gemm dimensions must be non-zero: m={m}, n={n}, k={k}")
            }
            MetalError::ALenMismatch { expected, actual } => {
                write!(f, "a length mismatch: expected {expected}, actual {actual}")
            }
            MetalError::BLenMismatch { expected, actual } => {
                write!(f, "b length mismatch: expected {expected}, actual {actual}")
            }
            MetalError::CLenMismatch { expected, actual } => {
                write!(f, "c length mismatch: expected {expected}, actual {actual}")
            }
            MetalError::NotEightAligned {
                m_eff,
                n_eff,
                k_eff,
            } => {
                write!(
                    f,
                    "effective dims must be multiples of 8: m_eff={m_eff}, n_eff={n_eff}, k_eff={k_eff}"
                )
            }
            MetalError::DimProductOverflow => {
                write!(f, "m*k, k*n or m*n overflows usize")
            }
            MetalError::DimensionExceedsU32 { m, n, k } => {
                write!(f, "gemm dimensions exceed u32::MAX: m={m}, n={n}, k={k}")
            }
            MetalError::ShapeMismatch { detail } => {
                write!(f, "shape mismatch: {detail}")
            }
            MetalError::InvalidRowKernelShape { detail } => {
                write!(f, "row kernel launch validation failed: {detail}")
            }
            MetalError::UnexpectedThreadExecutionWidth { expected, actual } => {
                write!(
                    f,
                    "unexpected threadExecutionWidth: expected {expected}, actual {actual}"
                )
            }
            MetalError::InvalidElementwiseShape { detail } => {
                write!(f, "invalid elementwise/gemm_bias_act shape: {detail}")
            }
        }
    }
}

impl std::error::Error for MetalError {}
