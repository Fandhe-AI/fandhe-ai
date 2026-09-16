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
    /// `crate::elementwise::pipeline_for_binary`／`pipeline_for_unary` が
    /// `BinaryElementwiseOp`／`UnaryElementwiseOp`（いずれも
    /// `#[non_exhaustive]`）の未知 variant を受け取った場合に返す
    /// （イシュー #1584。advisor 指摘: 未知 variant を `_ =>` で
    /// `add_f32`／`relu_f32` へフォールバックする実装は、将来 variant が
    /// 追加された際に「別の演算を代わりに計算して黙って成功する」
    /// fail-open になる。CUDA 側 `CudaError::UnsupportedElementwiseOp` と
    /// 同じ役割）。
    UnsupportedElementwiseOp { detail: String },
    /// `crate::context_cache`（プロセス内コンテキスト／カーネルスイート
    /// キャッシュ。イシュー #930。非公開モジュールのためリンクではなく
    /// コードスパン表記とする）の `Mutex` が poison していた。CUDA 側
    /// `CudaError::ContextCacheUnavailable`（feat/929-cuda-ctx-cache）と
    /// 同名・同義の専用 variant とする（`.claude/rules/coding-rust.md`
    /// 「本番経路で unwrap/expect を使わない」に従い panic させず、既存
    /// catch-all 分類〈`memory.rs::map_metal_error` の `other` アーム〉に
    /// 紛れ込ませず意図を明示する）。
    ContextCacheUnavailable { detail: String },
    /// `context.rs::MetalContext::batch`（コマンドバッファ共有バッチの
    /// `Mutex<BatchSlots>`。イシュー #1017）が poison していた。
    /// [`MetalError::ContextCacheUnavailable`] と同じ判断（panic させず
    /// 型付きエラーとして呼び出し元へ伝える。`.claude/rules/
    /// coding-rust.md`「本番経路で unwrap/expect を使わない」）だが、
    /// キャッシュとバッチ状態は別の `Mutex` インスタンスであるため
    /// variant を分ける。`detail` は poison を検出した操作
    /// （`encode`／`flush`／`synchronize`）の内訳を保持する。
    BatchStateUnavailable { detail: String },
    /// イシュー #1138: `crate::gemm::MetalGemm::dispatch_strided_tiled_prepared`
    /// の適格性ゲート（`crate::gemm::strided_tiled_eligibility`）が拒否した
    /// 入力（bias/act 付き・m/n/k 非 8 整除・leading dimension/offset 非
    /// 4 整除等。`gemm_simdgroup_tiled` の float4 ベクトルロード・8x8
    /// direct-load の前提を満たさない）。呼び出し元
    /// `dispatch_strided_bias_act_prepared` はこの `Err` を classic
    /// strided 経路（`gemm_tiled_bias_act`）へのフォールバック判断材料と
    /// して使う（fail-closed。`.claude/rules/security.md`「A03」参照）。
    StridedTiledIneligible { detail: String },
    /// `crate::gather_scatter::MetalGatherScatter::run_gather_f32`／
    /// `run_scatter_f32`（イシュー #1778）が起動前に独自検証する shape・
    /// `dim`・`index` 値域が不正だった。`pub` な起動 API を `ops.rs` を
    /// 経由せず直接呼び出す経路でも、カーネル（`shaders/
    /// gather_scatter.metal`）が `shapes` 定数バッファを
    /// `rank`／`in_shape`／`index_shape` 前提で読む都合上の GPU 側
    /// バッファ範囲外アクセスや、`usize` 積のオーバーフローを防ぐための
    /// 型付きエラー（`.claude/rules/security.md` A08。codex-review
    /// 指摘）。`detail` は元の
    /// [`fandhe_ai_tensor_core::ShapeError`] の `Display` 文字列表現。
    InvalidGatherScatterShape { detail: String },
    /// `crate::interpolate::MetalInterpolate::run_nearest_f32`（イシュー
    /// #1757）が起動前に独自検証する shape が不正だった。
    /// `InvalidGatherScatterShape` と同じ理由で独立 variant に分離する
    /// （`.claude/rules/security.md` A08）。`detail` は元の
    /// [`fandhe_ai_tensor_core::ShapeError`] の `Display` 文字列表現。
    InvalidInterpolateShape { detail: String },
    /// `crate::constant_pad::MetalConstantPad::run_pad_f32`（イシュー
    /// #1756）が起動前に独自検証する shape が不正だった。
    /// `InvalidGatherScatterShape` と同じ理由で独立 variant に分離する
    /// （`.claude/rules/security.md` A08）。`detail` は元の
    /// [`fandhe_ai_tensor_core::ShapeError`] の `Display` 文字列表現。
    InvalidConstantPadShape { detail: String },
    /// `crate::im2col::MetalIm2col::run_im2col_f32`／`run_col2im_f32`
    /// （イシュー #1768）が起動前に独自検証する形状（`Im2colDims` の
    /// 導出・ホストスライス実長の整合）が不正だった。
    /// `InvalidGatherScatterShape` と同じ理由で独立 variant に分離する
    /// （`.claude/rules/security.md` A08）。`detail` は元の
    /// [`crate::im2col_model::Im2colPrepareError`] の `Display` 文字列
    /// 表現、またはスライス長不整合の直接メッセージ。
    InvalidIm2colShape { detail: String },
    /// `crate::im2col_model::derive_im2col_dims`（`crate::im2col::
    /// MetalIm2col::run_im2col_f32`／`run_col2im_f32` が起動前に呼ぶ。
    /// イシュー #1768）が検知した「形状パラメータがカーネル `uint`
    /// 引数の範囲（`u32::MAX`）を超過」。`InvalidIm2colShape`
    /// （内部契約違反）とは区別する: col は入力の `kH·kW` 倍で現実的
    /// 形状でも上限へ到達しうるため、`ops.rs::map_im2col_error` は
    /// 本 variant のみ `BackendError::Unsupported`（ホスト
    /// フォールバック）へ写像する（`backend-cuda::error::
    /// CudaError::Im2colSizeLimitExceeded` と同型の設計判断）。
    Im2colSizeLimitExceeded { detail: String },
    /// `crate::batch_norm_model::validate_batch_norm_launch`
    /// （`crate::batch_norm::MetalBatchNorm::run_batch_norm_train_f32`／
    /// `run_batch_norm_infer_f32` が起動前に呼ぶ。イシュー #1736）が
    /// 検知した契約違反（`eps` 非有限・負／`n*c*spatial != x.len()`／
    /// `w`／`b`／`mean`／`var` の長さ不一致／`c*4 > isize::MAX`）。
    InvalidBatchNormShape { detail: String },
    /// [`Self::InvalidBatchNormShape`] と同じ検証が検知した
    /// 「`n`／`c`／`spatial`／`m`／`numel` のいずれかがカーネル引数の
    /// `uint`（`u32::MAX`）上限を超過」。`InvalidBatchNormShape`
    /// （内部契約違反）とは区別する: `ops.rs::map_batch_norm_error`
    /// は本 variant のみ `BackendError::Unsupported`（ホスト
    /// フォールバック）へ写像する（`Im2colSizeLimitExceeded` と同型の
    /// 設計判断。`backend-cuda::error::CudaError::
    /// BatchNormSizeLimitExceeded` と対になる）。
    BatchNormSizeLimitExceeded { detail: String },
    /// `crate::pooling_model::{derive_pool_dims, derive_adaptive_dims}`
    /// （`crate::pooling::MetalPooling::run_max_pool2d_f32`／
    /// `run_avg_pool2d_f32`／`run_adaptive_avg_pool2d_f32` が起動前に
    /// 呼ぶ。イシュー #1730）が検知した契約違反（rank 不一致・空間軸
    /// ゼロ長・カーネル／ストライド／dilation が 0・`ceil_mode ==
    /// true`・padding 超過・空窓構成・出力長の分子が負・ホスト
    /// スライス実長不一致等）。
    InvalidPoolingShape { detail: String },
    /// [`Self::InvalidPoolingShape`] と同じ検証が検知した「形状パラ
    /// メータの導出値がカーネル引数の `uint`（`u32::MAX`）上限、また
    /// は MaxPool 索引契約の `plane_in <= i32::MAX` を超過」
    /// （`InvalidPoolingShape`〈内部契約違反〉とは区別する:
    /// `ops.rs::map_pooling_error`〈存在する場合〉は本 variant のみ
    /// `BackendError::Unsupported` へ写像しホストフォールバックへ
    /// 委ねる。`Im2colSizeLimitExceeded` と同型の設計判断）。
    PoolingSizeLimitExceeded { detail: String },
    /// `crate::reduce_model::{plan_reduce_all, plan_reduce_axis}`
    /// （`crate::reduce::MetalReduce::run_sum_all_f32`／
    /// `run_sum_axis_f32` が起動前に呼ぶ。イシュー #1895）が検知した
    /// 契約違反（ホストスライス実長不一致・`outer*inner`／
    /// `lanes*axis_len` の `usize` オーバーフロー・カーネル `uint`
    /// 引数〈`numel`／`num_chunks`／`lanes`／`axis_len`／`inner`〉の
    /// `u32::MAX` 上限超過）。`InvalidGatherScatterShape` と同じ理由で
    /// 独立 variant に分離する（`.claude/rules/security.md` A08）。
    /// `detail` は元の [`crate::reduce_model::ReducePrepareError`] の
    /// `Display` 文字列表現、またはスライス長不整合の直接メッセージ。
    /// `ops.rs::MetalBackendOps::sum`（イシュー #1896）は起動前に
    /// `reduce_model::plan_reduce_all`／`plan_reduce_axis` を先出しして
    /// サイズ上限超過を `BackendError::Unsupported` へ写像するため、
    /// 本 variant が `ops.rs` 経由で観測されるのは呼び出し元の検査を
    /// すり抜けた内部契約違反の場合のみであり、`map_metal_error` の
    /// wildcard arm（`other => KernelLaunchFailed`）へ落ちる
    /// （`Im2colSizeLimitExceeded`／`PoolingSizeLimitExceeded` と異なり
    /// 専用の `Unsupported` 写像分岐は設けない）。
    InvalidReduceShape { detail: String },
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
            MetalError::UnsupportedElementwiseOp { detail } => {
                write!(f, "unsupported elementwise op: {detail}")
            }
            MetalError::ContextCacheUnavailable { detail } => {
                write!(f, "Metal context/kernel-suite cache unavailable: {detail}")
            }
            MetalError::BatchStateUnavailable { detail } => {
                write!(f, "Metal command batch state unavailable: {detail}")
            }
            MetalError::StridedTiledIneligible { detail } => {
                write!(f, "strided tiled GEMM route ineligible: {detail}")
            }
            MetalError::InvalidGatherScatterShape { detail } => {
                write!(f, "invalid gather/scatter shape, dim or index: {detail}")
            }
            MetalError::InvalidInterpolateShape { detail } => {
                write!(f, "invalid interpolate shape: {detail}")
            }
            MetalError::InvalidConstantPadShape { detail } => {
                write!(f, "invalid pad shape: {detail}")
            }
            MetalError::InvalidIm2colShape { detail } => {
                write!(f, "invalid im2col/col2im shape: {detail}")
            }
            MetalError::Im2colSizeLimitExceeded { detail } => {
                write!(f, "im2col/col2im size limit exceeded: {detail}")
            }
            MetalError::InvalidBatchNormShape { detail } => {
                write!(f, "invalid batch_norm shape: {detail}")
            }
            MetalError::BatchNormSizeLimitExceeded { detail } => {
                write!(f, "batch_norm size limit exceeded: {detail}")
            }
            MetalError::InvalidPoolingShape { detail } => {
                write!(f, "invalid pooling shape: {detail}")
            }
            MetalError::PoolingSizeLimitExceeded { detail } => {
                write!(f, "pooling size limit exceeded: {detail}")
            }
            MetalError::InvalidReduceShape { detail } => {
                write!(f, "invalid reduce shape: {detail}")
            }
        }
    }
}

impl std::error::Error for MetalError {}
