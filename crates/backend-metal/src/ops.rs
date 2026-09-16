//! Metal バックエンドの `BackendOps` 実装（TASK-1.9c・#46。イシュー #605 で
//! elementwise 5 演算・`gemm_bias_act` 実融合化を追加）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の Metal 実装。GEMM は
//! 既定で `gemm::MetalGemm::dispatch_auto`（動的タイル選択済み。
//! TASK-1.8c・#40）へ委譲する（既存カーネル・許容誤差・境界検査には
//! 触れない）。`dispatch_auto` は内部に split-K 2 パス経路への本番結線
//! ゲート（`self.split_k_auto_enabled`〈既定 `true` 固定〉・
//! `SPLIT_K_NUMERIC_CONTRACT_APPROVED`・実行時トグル `crate::
//! split_k_runtime::split_k_enabled()`〈既定 `true`〉。イシュー
//! #1516・#1545・#1547）を持つが、本モジュール（`gemm` メソッド）は
//! `dispatch_auto` を呼ぶのみでゲートの状態を意識しない——実行時トグルを
//! `false` にした場合は結線追加前と bit 同一の classic 経路のまま
//! （`docs/backend-metal-splitk-decision.md` §5）。片側のみが転置 view
//! （NT/TN。`autodiff::grad` の VJP が
//! `transpose2d` した勾配を渡す形状）の場合は `dispatch_auto` の代わりに
//! `gemm::MetalGemm::dispatch_strided_bias_act_prepared`（classic strided
//! カーネル）へ結線し、ホスト側の転置再パックコピーを省く（イシュー
//! #1215。`gemm_resident_lhs`〈#1040〉が確立した経路を `gemm` 本体へ
//! 拡張したもの。NN・TT・分類不能形状は従来どおり `contiguous()` +
//! `dispatch_auto` の bit 同一経路のまま——数値契約は
//! `docs/matmul-vjp-zero-copy-decision.md` §4.4 参照）。elementwise
//! （`add`／`mul`／`relu`／`exp`／`tanh`）は `elementwise::MetalElementwise`
//! へ委譲する（イシュー #605。CUDA 側 #599 の Metal 対応版）。汎用
//! reduction（`sum`／`max`）は未実装のまま
//! [`fandhe_ai_tensor_core::device::BackendError::Unsupported`] を返す（スコープ外。
//! out-of-scope-tracking.md 対象）。
//!
//! `cfg(target_os = "macos")` 限定（`objc2`／`objc2-foundation`／
//! `objc2-metal` と同じ cfg 境界。`.claude/rules/deps-policy.md`）。
//! 非 macOS 環境ではこのファイル自体がコンパイル対象に入らない
//! （`lib.rs` の cfg 境界と整合。`device.rs` と同方針）。
//!
//! `f16` の [`fandhe_ai_tensor_core::TypedOps<half::f16>`] 実装は
//! `crate::typed_f16` に置く（イシュー #1705。`typed_ops_f16` accessor
//! はこのファイルで結線する）。

use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    Activation, BackendOps, BatchNormTrainOutput, BceKind, BinaryElementwiseOp, Conv2dParams,
    DispatchFailureCell, FusionPlan, GruBackwardOutput, GruPointwiseOutput, HuberKind,
    InterpolateMode, KlDivTarget, LstmPointwiseOutput, MatrixNormOrd, MseReduction, Pool2dParams,
    QrFactors, ScalarBinaryOp, ScalarUnaryOp, ScatterReduce, ShapeError, SvdFactors, Tensor,
    UnaryElementwiseOp, adaptive_pool2d_out_shape, batch_norm_layout, gather_out_shape,
    im2col_out_shape, interpolate_out_shape_for_mode, one_hot_out_shape, pad_out_shape,
    pool2d_out_shape, reduce_out_shape, require_same_shape, row_norm_layout, row_softmax_layout,
    scatter_out_shape, sort_out_shape, topk_out_shape,
};

use crate::context::MetalContext;
use crate::context_cache;
use crate::elementwise::MetalElementwise;
use crate::error::MetalError;
use crate::gather_scatter_model::{GS_MAX_RANK, validate_index_range, validate_shapes_fit_u32};
use crate::layout::{self, MatrixLayout};
use crate::memory::{MetalBufferHandle, MetalMemory, map_metal_error};
use crate::row_kernel::{self, plan_dtype_is_f32};

/// `sort_model::SortPrepareError` を `BackendError` へ写像する
/// （`ops.rs::sort`／`topk` から呼ばれる。CUDA 側 `ops.rs::
/// map_sort_error` と同型の判断: `SizeLimitExceeded`〈合成キー配列長が
/// バックエンド固有上限を超過〉のみ `Unsupported` としてホスト
/// フォールバックへ委ねる。`DimSizeTooLarge`〈`dim_size` が
/// カーネル引数型の範囲〈`i32::MAX`〉を超える〉は `ShapeMismatch
/// (ShapeError::IndexRangeOverflow)` へ写像する——PR #1844 codex-review
/// 指摘の是正: 当初は `ElementCountOverflow` へ写像していたが、CPU
/// 参照実装（`backend-cpu::sort_topk`）が同じ状況（軸内添字の
/// `i32::try_from` 失敗）で返す variant と食い違っていた。CUDA 側
/// `ops.rs::map_sort_error`／`CudaError::SortDimSizeTooLarge` も同型に
/// 是正済み）。
/// `scan_model::plan_scan`（Metal `cumsum`／`cumprod` の起動前ホスト側
/// 検証。イシュー #1740・PR #1849 Cursor Bugbot 指摘の是正）のエラーを
/// `BackendOps::cumsum`／`cumprod` の戻り値へ変換する。唯一の variant
/// `SizeLimitExceeded`（`lanes`／`axis_len`／`inner` がカーネル `uint`
/// 引数の範囲〈`u32::MAX`〉を超過）を `BackendError::Unsupported` へ
/// 写像し、`Var::cumsum`／`cumprod` のホストフォールバック（`eval::
/// cumsum_along`／`cumprod_along`）へ委ねる（CUDA `ops.rs::
/// map_scan_error` の `ScanSizeLimitExceeded` → `Unsupported`・Metal
/// `map_sort_prepare_error` と同型）。要素数積の `usize` オーバー
/// フローは本関数の対象外で、`run_scan` が `checked_numel` で
/// `ShapeMismatch` として先に拒否する。
fn map_scan_prepare_error(err: crate::scan_model::ScanPrepareError) -> BackendError {
    match err {
        crate::scan_model::ScanPrepareError::SizeLimitExceeded { .. } => {
            BackendError::Unsupported(err.to_string())
        }
    }
}

fn map_sort_prepare_error(err: crate::sort_model::SortPrepareError) -> BackendError {
    match err {
        crate::sort_model::SortPrepareError::SizeLimitExceeded { .. } => {
            BackendError::Unsupported(err.to_string())
        }
        crate::sort_model::SortPrepareError::DimSizeTooLarge { dim_size } => {
            BackendError::ShapeMismatch(ShapeError::IndexRangeOverflow { index: dim_size })
        }
    }
}

std::thread_local! {
    /// [`MetalBackendOps::gemm_resident_lhs`]／[`MetalBackendOps::
    /// gemm_resident_rhs`] が「転置 view の zero-repack 経路」に乗れず
    /// `Tensor::contiguous()`（ホスト側転置コピー）へフォールバックした
    /// 回数（イシュー #1040。`gemm::BIAS_ACT_FUSED_LAUNCH_COUNT` と同型の
    /// 可観測点）。`crate::layout::classify_2d` が `None` を返す入力
    /// （stride 0 のブロードキャスト等の非対応形状）のみがこのフォール
    /// バックへ到達する。`pub(crate)`（`gemm::BIAS_ACT_FUSED_LAUNCH_COUNT`
    /// と同じ可視性方針。クレート境界外の `tests/gemm_resident_parity.rs`
    /// からは参照できないため、「フォールバック非経由」の確認は本ファイル
    /// 内の `#[cfg(test)]` クレート内テスト（macOS 実機・`#[ignore]`）に
    /// 委ね、外部テストファイルは数値一致のみを検証する契約とする）。
    pub(crate) static RESIDENT_HOST_REPACK_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };

    /// [`MetalBackendOps::gemm`] が NT/TN の strided 入口
    /// （`gemm_strided_nt_tn`）に乗れず `Tensor::contiguous()`
    /// （ホスト側転置コピー）へフォールバックした回数（イシュー #1215。
    /// `RESIDENT_HOST_REPACK_COUNT` と同型の可観測点だが対象メソッドが
    /// 異なるため独立カウンタとする。TT（両方転置）・`classify_2d` が
    /// `None` を返す入力〈stride 0 のブロードキャスト等〉が実際に
    /// `contiguous()` を要した場合（＝元から contiguous でなかった
    /// オペランド）にのみ本カウンタを増やす。両オペランドとも
    /// contiguous な NN（`VJP` が渡す通常の非転置入力）は増えないが、
    /// 両方とも `transposed: false` に分類されるが非 contiguous な
    /// 行優先 view（`ld > cols` の narrow 等）は本経路（従来
    /// `contiguous()` + `dispatch_auto`）に落ちるため増加しうる
    /// （`gemm` 内の `!a.is_contiguous()`／`!b.is_contiguous()`
    /// ガード参照）。`backend-cuda::ops::
    /// GEMM_HOST_REPACK_COUNT`〈イシュー #1214〉と同名・同意図の
    /// クロスバックエンド可観測点。`pub(crate)`（`RESIDENT_HOST_REPACK_COUNT`
    /// と同じ可視性方針。クレート境界外テストは数値一致のみ検証する）。
    pub(crate) static GEMM_HOST_REPACK_COUNT: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// GEMM オペランド 1 個をアップロードする（イシュー #1040）。
/// `layout::classify_2d` が分類できる view（行優先 contiguous・転置
/// view のいずれか）は [`MetalMemory::upload_view`] 経由で
/// `Tensor::as_view_slice`（借用）をそのままアップロードし、ホスト側の
/// 転置コピーを発生させない。分類できない形状（stride 0 の
/// ブロードキャスト等）のみ、従来どおり `MemoryOps::upload`
/// （`Tensor::contiguous()` 経由）へフォールバックし
/// [`RESIDENT_HOST_REPACK_COUNT`] を増やす。
///
/// 戻り値の [`MatrixLayout`] は `dispatch_strided_bias_act_prepared` へ
/// そのまま渡す（フォールバック時は `contiguous()` 後の実際の行優先
/// 形状に対応する NN レイアウトを返す）。
fn upload_operand_for_resident_gemm(
    mem: &MetalMemory,
    tensor: &Tensor<f32>,
) -> Result<
    (
        fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        MatrixLayout,
    ),
    BackendError,
> {
    if let Some(layout) = layout::classify_2d(tensor.shape(), tensor.strides())
        && let Some(slice) = tensor.as_view_slice()
    {
        let dev_buf = mem
            .upload_view(slice, tensor.shape())
            .map_err(map_metal_error)?;
        return Ok((dev_buf, layout));
    }
    RESIDENT_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
    let dev_buf = mem.upload(tensor)?;
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    let layout = MatrixLayout {
        rows,
        cols,
        ld: cols,
        transposed: false,
    };
    Ok((dev_buf, layout))
}

/// RNN／LSTM／GRU セル演算（イシュー #1647）の入口検査: `shape` が
/// rank-2 であることを検証する（`backend-cpu::ops::require_rank2`・
/// `backend-cuda::ops::require_rank2_cell` と同型。平坦化後の要素数
/// 一致だけでは異形状の取り違えを検出できないため、
/// `lstm_pointwise`／`lstm_hidden_backward`／`lstm_cell_backward`／
/// `gru_pointwise`／`gru_backward` の各エントリで使う。codex-review
/// P2 指摘）。
fn require_rank2_cell(shape: &[usize]) -> Result<(), BackendError> {
    if shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    Ok(())
}

/// `.contiguous()`（内部で無検査の `Vec::with_capacity(numel)` を
/// 呼ぶ）を呼び出す前に、要素型 `T` 換算のバイトサイズが `Vec` の
/// allocation 上限（`isize::MAX` バイト）に収まるか検査する
/// （`fandhe_ai_tensor_core::tensor::checked_numel_for::<T>` と
/// 同型の独立複製。同関数は `pub(crate)` でクレートを跨いで共有
/// できないため。`interpolate` は巨大な `broadcast_to` view を
/// 小さい `size` へ縮小できてしまう（forward 契約は出力 shape のみを
/// 検査する）ため、要素数積が `usize` の範囲に収まっても
/// `numel * size_of::<T>() > isize::MAX` となるケースがあり、
/// `.contiguous()` がそこで capacity overflow panic する。本番経路
/// panic 禁止規約 `.claude/rules/coding-rust.md` に反するため、
/// `.contiguous()` 呼び出し直前に本関数で確保前検査する
/// （当初 `interpolate` 限定〈イシュー #1834 Cursor Bugbot 指摘〉の
/// `checked_f32_bytes` を、`cast`〈イシュー #1751〉の複数要素型
/// 〈`f32`／`i32`／`i64`／`bool`〉に対応するため型パラメータ `T` へ
/// 一般化した）。
pub(crate) fn checked_bytes_for<T>(shape: &[usize]) -> Result<(), ShapeError> {
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let elem_size = std::mem::size_of::<T>();
    if elem_size > 0 {
        let bytes = numel
            .checked_mul(elem_size)
            .ok_or(ShapeError::ElementCountOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }
    Ok(())
}

/// `interpolate.rs::MetalInterpolate::run_nearest_f32` のエラーを
/// `BackendOps::interpolate` の戻り値へ変換する（イシュー #1757・
/// PR #1834 codex-review／Cursor Bugbot 指摘の是正）。
///
/// `run_nearest_f32` は自身の独立検査
/// （`interpolate_model::validate_interpolate_launch`。モジュール doc
/// 参照）に失敗すると `MetalError::InvalidInterpolateShape` を返すが、
/// 是正前はこの分岐も他の起動失敗と区別せず一律
/// `BackendError::KernelLaunchFailed` へ変換していた。これでは
/// `gather`／`scatter`／`pad`（`ops.rs` 内の同種カーネル）や
/// `backend-cuda::ops::map_interpolate_error`（同一イシューの CUDA 側
/// 実装）と異なり、呼び出し元が「形状不正」を `BackendError::
/// ShapeMismatch` として識別できず、`Var` 側のホストフォールバック
/// 判定（`Unsupported`／`ShapeMismatch` 分岐）を素通りしてしまう
/// （`u32::MAX` 超過等の入力は `size=[u32::MAX as usize + 1]` のように
/// `interpolate_out_shape` の共通形状検証は通過しうる。指摘の再現形状）。
/// `InvalidInterpolateShape` のみ `ShapeError::ElementCountOverflow`
/// 経由の `ShapeMismatch` へ、それ以外（デバイス・パイプライン起動
/// 失敗等）は従来どおり `KernelLaunchFailed` へ変換する
/// （`backend-cuda::ops::map_gather_scatter_error`／
/// `map_interpolate_error` と同型の変換方針）。
fn map_interpolate_error(err: MetalError) -> BackendError {
    match err {
        MetalError::InvalidInterpolateShape { .. } => {
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        }
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// `im2col.rs::MetalIm2col::run_im2col_f32`／`run_col2im_f32` のエラー
/// を `BackendOps::im2col`／`col2im` の戻り値へ変換する（イシュー
/// #1768）。[`MetalError::Im2colSizeLimitExceeded`]（形状パラメータが
/// カーネル引数 `uint` 上限を超過。`im2col.rs::map_prepare_error` が
/// [`crate::im2col_model::Im2colPrepareError::SizeLimitExceeded`] を
/// 写像したもの）**のみ** [`BackendError::Unsupported`] へ写像し、
/// `fandhe_ai_autodiff::grad::im2col_with_fallback`／
/// `col2im_with_fallback` のホストフォールバック（`eval::im2col`／
/// `col2im`）へ委ねる（col は入力の `kH·kW` 倍で現実的形状でも上限へ
/// 到達しうるため hard fail ではなくフォールバックが妥当。
/// `map_scan_prepare_error`・`backend-cuda::ops::map_im2col_error` と
/// 同じ設計判断）。[`MetalError::InvalidIm2colShape`]（内部契約違反。
/// 呼び出し元 `ops.rs` の事前検証を通過した入力からは実質到達しない
/// 防御的経路）は `ShapeError::ElementCountOverflow` へ、それ以外
/// （デバイス・パイプライン起動失敗等）は `KernelLaunchFailed` へ
/// 変換する（判定迂回経路を作らない。`.claude/rules/security.md`
/// A08）。
fn map_im2col_error(err: MetalError) -> BackendError {
    match err {
        MetalError::Im2colSizeLimitExceeded { .. } => BackendError::Unsupported(err.to_string()),
        MetalError::InvalidIm2colShape { .. } => {
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        }
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// `batch_norm.rs::MetalBatchNorm::run_batch_norm_train_f32`／
/// `run_batch_norm_infer_f32` のエラーを `BackendOps::
/// batch_norm_train`／`batch_norm_infer` の戻り値へ変換する（イシュー
/// #1736）。[`MetalError::BatchNormSizeLimitExceeded`]（`n`／`c`／
/// `spatial`／`m`／`numel` のいずれかがカーネル引数の `uint`
/// （`u32::MAX`）上限を超過）**のみ** [`BackendError::Unsupported`]
/// へ写像し、`fandhe_ai_autodiff::grad::batch_norm_train_with_fallback`／
/// `batch_norm_infer_with_fallback` のホストフォールバック
/// （`eval::batch_norm_train_channels`／`batch_norm_infer_channels`）
/// へ委ねる（`map_im2col_error` と同じ設計判断・`backend-cuda::ops::
/// map_batch_norm_error` と対になる）。[`MetalError::
/// InvalidBatchNormShape`]（`weight`／`bias`／`mean`／`var` の長さ
/// 不一致・`eps` 検査等）は CPU 側 `CpuBackendOps::batch_norm_train`
/// （`BackendError::KernelLaunchFailed` を返す）と同じ variant へ
/// 揃える（既存 CPU テストの `matches!(…, KernelLaunchFailed)` との
/// 整合）。それ以外（デバイス・パイプライン起動失敗等）も
/// `KernelLaunchFailed` へ変換する（判定迂回経路を作らない。
/// `.claude/rules/security.md` A08）。
fn map_batch_norm_error(err: MetalError) -> BackendError {
    match err {
        MetalError::BatchNormSizeLimitExceeded { .. } => BackendError::Unsupported(err.to_string()),
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// `pooling.rs::MetalPooling::run_max_pool2d_f32`／
/// `run_avg_pool2d_f32`／`run_adaptive_avg_pool2d_f32` のエラーを
/// `BackendOps::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d` の
/// 戻り値へ変換する（イシュー #1730・追従イシュー。#1607 ツリー）。
/// [`MetalError::PoolingSizeLimitExceeded`]（形状パラメータがカーネル
/// 引数 `uint` 上限を超過）**のみ** [`BackendError::Unsupported`] へ
/// 写像し、`fandhe_ai_autodiff` 側のホストフォールバック（`eval::
/// max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d`）へ委ねる
/// （`map_im2col_error`／`map_batch_norm_error` と同じ設計判断）。
/// [`MetalError::InvalidPoolingShape`]（内部契約違反。呼び出し元
/// `ops.rs` の事前検証を通過した入力からは実質到達しない防御的経路）
/// は `ShapeError::ElementCountOverflow` へ、それ以外（デバイス・
/// パイプライン起動失敗等）は `KernelLaunchFailed` へ変換する（判定
/// 迂回経路を作らない。`.claude/rules/security.md` A08）。
fn map_pooling_error(err: MetalError) -> BackendError {
    match err {
        MetalError::PoolingSizeLimitExceeded { .. } => BackendError::Unsupported(err.to_string()),
        MetalError::InvalidPoolingShape { .. } => {
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        }
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// RNN／LSTM／GRU 系エントリが形状比較の前に必要とする `gates * hidden`
/// （ゲート幅）を `checked_mul` で検証する（`backend-cpu::ops::
/// checked_gate_width`／`backend-cuda::ops::checked_gate_width` と
/// 同型）。
///
/// 本番経路 panic 禁止（AGENTS.md）: `4 * hidden`／`3 * hidden` を
/// 未検証のまま `require_same_shape` の期待値へ埋め込むと、`hidden`
/// が `usize::MAX` 近傍（要素数 0 の空テンソルは `shape[1]` を自由に
/// 取れる）のとき乗算が overflow して期待幅が小さい値へ周回し、
/// 本来 shape mismatch で拒否すべき不正な入力を誤って受理してしまう
/// （受理後は下層カーネルが `hidden` を使った添字アクセスで範囲外
/// 参照する）。イシュー #1647 codex-review P1 指摘。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, BackendError> {
    gates.checked_mul(hidden).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountMismatch {
            expected: usize::MAX,
            actual: 0,
        },
    ))
}

/// `shape` の要素積を `checked_mul` の連鎖で求め、`usize` オーバー
/// フロー時は `ShapeError::ElementCountOverflow` を返す
/// （`checked_bytes_for` 内の複製ロジックを独立関数化したもの。
/// `nll_loss`／`nll_loss_backward` の `NllLayout` 構築が
/// `input_shape`〈利用者から渡される任意の shape〉から直接
/// `.iter().product()` していた箇所を置き換える。PR #1850
/// codex-review P1 是正）。
/// イシュー #1751 で `pub(crate)` へ緩和した（`crate::cast` が同じ
/// 要素数積オーバーフロー検査を再利用するため。可視性の意味論・
/// 検査内容自体は変更しない）。
pub(crate) fn checked_shape_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// Metal バックエンドの `BackendOps` 実装。`Device::Metal` は ordinal を
/// 持たない単一 variant のため（`docs/public-api-design.md` §4.1・
/// `device.rs::MetalDeviceProvider` と同じ位置付け）、本実装は複数 GPU の
/// 個別選択をサポートしない（システムデフォルトの Metal デバイスに
/// 対応する）。
///
/// `MetalContext`／`MetalGemm`／`MetalElementwise`／`MetalRmsNorm`／
/// `MetalSoftmax` はいずれも `crate::context_cache` 経由でプロセス内
/// キャッシュから取得する（イシュー #930 で常駐化完了。診断 #927 が特定
/// した「演算メソッド呼び出しごとの都度構築」固定オーバーヘッド〈約 5 ms・
/// N 非依存〉を解消する。CUDA 側 `backend-cuda::ops::CudaBackendOps`
/// も同時期に同型キャッシュ〈#929〉へ移行済み）。
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackendOps;

impl MetalBackendOps {
    /// 新規 `MetalBackendOps` を構築する。構築自体はデバイス初期化を
    /// 行わないため常に成功する（実際の初期化は各メソッドが
    /// `MetalContext::new` を経由した時点）。
    pub fn new() -> Self {
        Self
    }

    /// 二項 elementwise 共通のディスパッチ（`add`／`mul`。イシュー #605）。
    ///
    /// `Tensor::broadcast_with`（NumPy 互換ブロードキャスト）で共通 shape
    /// の view を得たのち `contiguous()` で密なバッファへ実体化してから
    /// `MetalElementwise`（同一長バッファのみを扱う。`elementwise.rs`
    /// 冒頭コメント「ブロードキャスト」参照）へ渡す。`run` は
    /// `MetalElementwise::run_add_f32`／`run_mul_f32` のいずれかを呼ぶ
    /// クロージャとして呼び出し側から注入される（`backend-cuda::ops::
    /// CudaBackendOps::elementwise_binary` と同型の構成）。
    fn elementwise_binary(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        run: impl FnOnce(
            &MetalElementwise,
            &MetalContext,
            &[f32],
            &[f32],
        ) -> Result<Vec<f32>, MetalError>,
    ) -> Result<Tensor<f32>, BackendError> {
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

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = run(&ew, &ctx, a_slice, b_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 単項 elementwise 共通のディスパッチ（`relu`／`exp`／`tanh`。
    /// イシュー #605）。ブロードキャストが発生しない点を除き
    /// [`Self::elementwise_binary`] と同一構造。
    fn elementwise_unary(
        &self,
        a: &Tensor<f32>,
        run: impl FnOnce(&MetalElementwise, &MetalContext, &[f32]) -> Result<Vec<f32>, MetalError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = a.shape().to_vec();
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = run(&ew, &ctx, a_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 3 項 elementwise 共通のディスパッチ（`where_cond`。イシュー
    /// #1637）。[`Self::elementwise_binary`] と異なり broadcast は
    /// 行わない（`BackendOps::where_cond` doc の契約どおり、`cond`／
    /// `a`／`b` は呼び出し元〈`Var::where_cond`〉が同一 `out_shape` へ
    /// 実体化済みで渡す前提。ここでは形状一致を再検査するのみ・
    /// `.claude/rules/security.md` A08）。
    fn elementwise_ternary(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        run: impl FnOnce(
            &MetalElementwise,
            &MetalContext,
            &[f32],
            &[f32],
            &[f32],
        ) -> Result<Vec<f32>, MetalError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = a.shape().to_vec();
        if cond.shape() != out_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: cond.shape().to_vec(),
                rhs: out_shape,
            }));
        }
        if b.shape() != out_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: b.shape().to_vec(),
                rhs: out_shape,
            }));
        }

        let cond_owned = cond.contiguous();
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let cond_slice = cond_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: cond not contiguous".into())
        })?;
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: rhs not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = run(&ew, &ctx, cond_slice, a_slice, b_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// 二項＋スカラー elementwise 共通のディスパッチ（`masked_fill`。
    /// イシュー #1637）。broadcast は行わない（`BackendOps::
    /// masked_fill` doc の契約どおり `mask` は `x` と同一 shape）。
    fn elementwise_binary_scalar(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
        run: impl FnOnce(
            &MetalElementwise,
            &MetalContext,
            &[f32],
            &[f32],
            f32,
        ) -> Result<Vec<f32>, MetalError>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = x.shape().to_vec();
        if mask.shape() != out_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: mask.shape().to_vec(),
                rhs: out_shape,
            }));
        }

        let x_owned = x.contiguous();
        let mask_owned = mask.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: lhs not contiguous".into())
        })?;
        let mask_slice = mask_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("elementwise: mask not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = run(&ew, &ctx, x_slice, mask_slice, value)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::scalar_unary`] の Metal ディスパッチ（イシュー
    /// #1707・#1708・#1709〈`Clamp` 追加〉。CUDA 側
    /// `CudaBackendOps::scalar_unary_dispatch`〈#1700〉の Metal
    /// 対応版）。`crate::scalar_op_source` が対応する kind のみ
    /// パイプラインを生成・キャッシュして起動し、未対応 kind は
    /// `BackendError::Unsupported` を返す（呼び出し元
    /// `fandhe_ai_autodiff::grad::scalar_unary_with_fallback` がホスト
    /// 参照実装へフォールバックする既存契約。`BackendOps::scalar_unary`
    /// の既定トレイト実装と同じエラー種別を返すことで判定迂回経路を
    /// 作らない。`.claude/rules/security.md` A08）。
    ///
    /// 対応 kind 判定を `context_cache::cached_context()`（Metal デバイス
    /// 取得）より前に行う（CUDA 側と同じ理由。codex-review 指摘・
    /// PR #1781: 先に判定しないと、Metal 利用不可環境で未対応 kind に
    /// 対し本来返すべき `Unsupported` ではなく `DeviceUnavailable` を
    /// 返してしまい、フォールバック契約が後退する）。
    fn scalar_unary_dispatch(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if crate::scalar_op_source::unary_kernel_source(op).is_none() {
            return Err(BackendError::Unsupported(format!(
                "scalar_unary: Metal template kernel not implemented for {op:?} \
                 (いずれの sub issue にも含まれない kind。ホスト参照実装へフォールバック)"
            )));
        }

        let out_shape = a.shape().to_vec();
        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("scalar_unary: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let pipeline = context_cache::cached_scalar_unary_pipeline(&ctx, op)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let Some(pipeline) = pipeline else {
            return Err(BackendError::Unsupported(format!(
                "scalar_unary: Metal template kernel not implemented for {op:?} \
                 (いずれの sub issue にも含まれない kind。ホスト参照実装へフォールバック)"
            )));
        };
        // `Clamp` 等（イシュー #1709）のペイロード（`min`／`max`）を
        // 起動引数として渡す（`crate::scalar_op_source` モジュール doc
        // 「ペイロード seam」参照。ペイロードなし kind は空スライス）。
        let payload = crate::scalar_op_source::unary_payload(op);
        let out = ew
            .run_scalar_unary_f32(&ctx, &pipeline, a_slice, payload.as_slice())
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::scalar_unary_dispatch`] の 2 項版（イシュー #1707・
    /// #1708・#1709〈比較演算 6 種追加〉）。ブロードキャストは
    /// `elementwise_binary`（`add`／`mul`）と同じ `Tensor::broadcast_with`。
    fn scalar_binary_dispatch(
        &self,
        op: ScalarBinaryOp,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if crate::scalar_op_source::binary_kernel_source(op).is_none() {
            return Err(BackendError::Unsupported(format!(
                "scalar_binary: Metal template kernel not implemented for {op:?} \
                 (いずれの sub issue にも含まれない kind。ホスト参照実装へフォールバック)"
            )));
        }

        let (a_bc, b_bc) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let out_shape = a_bc.shape().to_vec();
        let a_owned = a_bc.contiguous();
        let b_owned = b_bc.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("scalar_binary: lhs not contiguous".into())
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("scalar_binary: rhs not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let pipeline = context_cache::cached_scalar_binary_pipeline(&ctx, op)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let Some(pipeline) = pipeline else {
            return Err(BackendError::Unsupported(format!(
                "scalar_binary: Metal template kernel not implemented for {op:?} \
                 (いずれの sub issue にも含まれない kind。ホスト参照実装へフォールバック)"
            )));
        };
        let out = ew
            .run_scalar_binary_f32(&ctx, &pipeline, a_slice, b_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::gemm`] の NT/TN 専用経路（イシュー #1215）。呼び出し元が
    /// `a`／`b` の `layout::classify_2d` 結果（`la`／`lb`）を既に取得済み
    /// （`transposed` フラグが互いに異なることを確認済み）である前提で、
    /// 両オペランドを `MetalMemory::upload_view`（借用スライスの
    /// zero-copy アップロード。`upload_operand_for_resident_gemm` と同じ
    /// 経路）でアップロードし、`gemm::MetalGemm::
    /// dispatch_strided_bias_act_prepared`（`gemm_resident_lhs`〈#1040〉
    /// が確立した classic strided カーネル入口。bias／activation は
    /// 使わない）で GEMM 本体を計算する。`m == 0 || n == 0 || k == 0` は
    /// `dispatch_auto` 経路の `validate_dims`（`ZeroDimension` を拒否）と
    /// 挙動を揃えるため、本経路は `classify_2d` が rows/cols == 0 を
    /// 一律 `None` とする契約（`layout.rs` doc 参照）によりそもそも
    /// 到達しない（呼び出し元 `gemm` のガードで従来経路へ落ちる）。
    ///
    /// `#[allow(clippy::too_many_arguments)]`（9/7）: `&self`・`a`・`la`・
    /// `b`・`lb`・`m`・`n`・`k`・`out_shape` はいずれも呼び出し元 `gemm`
    /// が既に導出済みの値で、構造体へまとめると呼び出し側の一時変数が
    /// 増えるだけで可読性が上がらない（`gemm.rs::dispatch_strided_
    /// bias_act_prepared`〈13 引数〉と同じ判断。同ファイル冒頭の
    /// 複数箇所で同様の `#[allow]` を使用済み）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_strided_nt_tn(
        &self,
        a: &Tensor<f32>,
        la: MatrixLayout,
        b: &Tensor<f32>,
        lb: MatrixLayout,
        m: usize,
        n: usize,
        k: usize,
        out_shape: &[usize],
    ) -> Result<Tensor<f32>, BackendError> {
        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());

        let a_slice = a
            .as_view_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let a_dev_buf = mem
            .upload_view(a_slice, a.shape())
            .map_err(map_metal_error)?;
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm: lhs buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let b_slice = b
            .as_view_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;
        let b_dev_buf = mem
            .upload_view(b_slice, b.shape())
            .map_err(map_metal_error)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm: rhs buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx, a_buf, 0, la, b_buf, 0, lb, None, false, c_buf, m, n, k,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        let out = mem.download(&c_dev_buf)?;
        debug_assert_eq!(out.shape(), out_shape);
        Ok(out)
    }

    /// [`BackendOps::gemm_fp32_strict_into`]／[`BackendOps::
    /// gemm_fp32_strict_into_tracked`] 共通の検証・ディスパッチ本体
    /// （イシュー #1555 追補・codex-review 指摘・PR #1556 でトークン
    /// 引数を追加する際に二重化を避けるため切り出した。`token` を
    /// 追加した以外の検証ロジック自体は無変更）。
    ///
    /// NN/TT・分類不能形状のホスト経路フォールバック（`self.
    /// gemm_fp32_strict(a, b)` → `MemoryOps::upload_into`）は既に
    /// `gemm_fp32_strict` 自身の内部で `ctx.synchronize()` まで完結する
    /// 同期経路であり、その復帰直後に `upload_into`（これも内部で
    /// download/upload の都度同期を伴う）が続くため、`token` を渡す
    /// 対象は NT/TN の encode-only 経路に限る（他クラスの `Unsupported`
    /// 迂回はここでは発生しない。トレイト側 doc 参照）。
    fn gemm_fp32_strict_into_impl(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        out_offset: usize,
        token: Option<&DispatchFailureCell>,
    ) -> Result<(), BackendError> {
        if out.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let out_shape = fandhe_ai_tensor_core::gemm_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let _ = out_shape; // shape 検証のみに使用（`gemm_fp32_strict_into` の CPU 実装と同型）

        // REQ-8「カーネル側の手動境界チェックを省略しない」・OWASP A03:
        // `out_offset + m*n` を `checked_mul`/`checked_add` で検査し、
        // `out.numel()` を超える書き込みを事前に拒否する（カーネル起動
        // 前・NT/TN 判定より前。範囲外オフセットは形状に関わらず常に
        // `InvalidArgument`）。
        let mn = m.checked_mul(n).ok_or_else(|| {
            BackendError::InvalidArgument("gemm_fp32_strict_into: m * n overflowed usize".into())
        })?;
        let end = out_offset.checked_add(mn).ok_or_else(|| {
            BackendError::InvalidArgument(
                "gemm_fp32_strict_into: out_offset + m * n overflowed usize".into(),
            )
        })?;
        if end > out.numel() {
            return Err(BackendError::InvalidArgument(format!(
                "gemm_fp32_strict_into: write range [{out_offset}, {end}) exceeds out buffer \
                 length {}",
                out.numel()
            )));
        }

        let layout_pair = match (
            layout::classify_2d(a.shape(), a.strides()),
            layout::classify_2d(b.shape(), b.strides()),
        ) {
            (Some(la), Some(lb))
                if la.transposed != lb.transposed
                    && a.as_view_slice().is_some()
                    && b.as_view_slice().is_some() =>
            {
                Some((la, lb))
            }
            _ => None,
        };

        // NT/TN 以外（NN・TT・分類不能形状。上記 doc「NN/TT・分類不能形状
        // を encode-only にしない理由」）は `Unsupported` を返さず、
        // ホスト経路 `gemm_fp32_strict`（trait 契約上 `Self::gemm` と
        // bit 同一）の戻り値を `MemoryOps::upload_into` で `out` へ
        // 書き込むフォールバックにする（codex-review 指摘・イシュー
        // #1555: `DeviceParamStore::fill_resident_weight_grad` は最初の
        // `Unsupported` でストア全体の `resident_grad_capability` を
        // `Some(false)` にキャッシュするため、形状単位で `Unsupported`
        // を返すと同一 backward 内で先に成功済みの resident slot が
        // `param_grads_to_host` から読めなくなる〈`MissingGradient`〉。
        // このフォールバックにより本メソッドは NT/TN 以外を含め常に成功し
        // `resident_grad_capability` が `Some(false)` へ倒れることは
        // なくなる）。
        let Some((la, lb)) = layout_pair else {
            let result = self.gemm_fp32_strict(a, b)?;
            let ctx = context_cache::cached_context().map_err(map_metal_error)?;
            let mem = MetalMemory::from_shared(ctx);
            return mem.upload_into(&result, out, out_offset);
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());

        let a_slice = a.as_view_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gemm_fp32_strict_into: lhs not contiguous".into())
        })?;
        let a_dev_buf = mem
            .upload_view(a_slice, a.shape())
            .map_err(map_metal_error)?;
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_fp32_strict_into: lhs buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let b_slice = b.as_view_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gemm_fp32_strict_into: rhs not contiguous".into())
        })?;
        let b_dev_buf = mem
            .upload_view(b_slice, b.shape())
            .map_err(map_metal_error)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_fp32_strict_into: rhs buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let out_handle = out
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(out_buf) = out_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_fp32_strict_into: out buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.encode_strided_bias_act_prepared_with_c_offset(
            &ctx, a_buf, 0, la, b_buf, 0, lb, None, false, out_buf, out_offset, m, n, k, token,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        Ok(())
    }
}

/// [`MetalBackendOps::gemm_bias_act`] が融合カーネル
/// （`gemm::MetalGemm::run_tiled_bias_act_f32`）と
/// `fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト実装
/// （非融合 `gemm`→`add`→`relu` 3 段合成）のどちらを経由するかを表す
/// （イシュー #605。CUDA 側 `fandhe_ai_backend_cuda::ops::GemmBiasActRoute`〈#599〉と
/// 同一の意味論）。
///
/// `backend-cpu::ops::CpuBackendOps::gemm_bias_act`・`backend-cuda::ops::
/// CudaBackendOps::gemm_bias_act` の分岐条件（`bias` が `None`、または
/// `bias.shape()` が厳密に `[n]` の場合にのみ融合カーネルへ進む）と同一の
/// 意味論を Metal 側にも適用する（バックエンド間で `gemm_bias_act` の
/// 経路依存の挙動差を作らない。イシュー #203 Review 指摘と同じ理由）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmBiasActRoute {
    /// 融合カーネル（epilogue 内で bias 加算・activation を適用）へ進む。
    Fused,
    /// デフォルト実装（`gemm`→`add`→act の非融合合成）へフォールバックする。
    ComposedFallback,
}

/// [`GemmBiasActRoute`] の選択ロジック（純関数。実機なしで単体テスト可能。
/// 本ファイル末尾 `#[cfg(test)]` 参照）。CUDA 側
/// `fandhe_ai_backend_cuda::ops::gemm_bias_act_route` と同一実装（`pub(crate)`:
/// `MetalBackendOps::gemm_bias_act` から呼ばれる）。
pub(crate) fn gemm_bias_act_route(bias_shape: Option<&[usize]>, n: usize) -> GemmBiasActRoute {
    match bias_shape {
        None => GemmBiasActRoute::Fused,
        Some(shape) if shape == [n] => GemmBiasActRoute::Fused,
        Some(_) => GemmBiasActRoute::ComposedFallback,
    }
}

/// `Device::Metal`（単一 variant。ordinal なし）に対応する
/// `&'static MetalMemory` をプロセス内シングルトンとして取得する
/// （イシュー #935）。
///
/// `BackendOps::memory_ops(&self) -> Option<&dyn MemoryOps>` は戻り値を
/// `&self`（`MetalBackendOps`。unit struct・`Copy`）の寿命へ束縛できる
/// 型で返す必要がある一方、`AllocationTracker` の計測系列（`docs/
/// device-resident-update-design.md` §3.3d）を維持するには `MetalMemory`
/// をプロセス全体で 1 個だけ共有しなければならない。`backend-cuda::ops::
/// static_cuda_memory` と同型の意図的な `Box::leak`（`context_cache.rs`
/// モジュール冒頭コメント「所有モデル・生存期間」と同じ「エントリは
/// プロセスの生存期間中 evict されない」設計に倣う。`Device::Metal` は
/// 単一デバイスのためキーは不要）。
///
/// `MetalMemory::from_shared`（イシュー #935 レビュー対応・`memory.rs`
/// 参照）が `context_cache::cached_context()` の返す `Arc<MetalContext>`
/// をそのまま受け取れるため、本関数はバッファ確保用の `MetalContext` と
/// カーネルディスパッチ（`sgd.rs::MetalSgd::run` 等）が使う `MetalContext`
/// を同一インスタンスに揃える（`docs/device-resident-update-design.md`
/// §3.3d「`XMemory` が持つ stream/context は必ず既存 `context_cache`
/// 経由で取得（独自初期化禁止）」）。`MetalMemory::new`（所有権を要求する
/// 既存の公開シグネチャ）は crates.io 公開済み API のため変更しない。
fn static_metal_memory() -> Result<&'static MetalMemory, BackendError> {
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<Option<&'static MetalMemory>>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| {
        BackendError::DeviceUnavailable("static_metal_memory: cache mutex poisoned".to_string())
    })?;
    if let Some(mem) = *guard {
        return Ok(mem);
    }
    let ctx = context_cache::cached_context().map_err(map_metal_error)?;
    let mem: &'static MetalMemory = Box::leak(Box::new(MetalMemory::from_shared(ctx)));
    *guard = Some(mem);
    Ok(mem)
}

/// [`MetalBackendOps::run_scan`] が `cumsum`／`cumprod` のいずれを
/// 起動するかを選択するための内部列挙（イシュー #1740。`backend-cuda::
/// ops::ScanOpKind` と同一設計だがクレートをまたいだ共有はしない
/// ——各バックエンドの `run_scan` はそれぞれの `*Scan` 型に閉じた
/// private ディスパッチのため）。
enum ScanOpKind {
    Sum,
    Prod,
}

impl MetalBackendOps {
    /// `Self::cumsum`／`cumprod`（イシュー #1740）の共通骨格。`dim` を
    /// [`reduce_out_shape`] で再検査し、`x.numel()`（内部で無検査の
    /// 乗算を呼ぶ）より前に要素数積のオーバーフローを検査
    /// （`gather_scatter_model::checked_numel` 適用方針。`unique` と
    /// 同型）してから `outer`／`axis_len`／`inner`（`backend-cpu::scan`
    /// と同一の lane 分解）を `scan_model::plan_scan` で導出し
    /// （カーネル `uint` 引数の上限超過はここで `Unsupported` →
    /// ホストフォールバックへ写像。`map_scan_prepare_error` 参照）
    /// `scan::MetalScan` へ委譲する。
    /// `shape` の要素数が 0（いずれかの軸が 0）の場合は GPU 起動なしで
    /// 空テンソルを返す（`checked_numel` は 0 を含む shape でも中間積
    /// オーバーフローの恐れがあるため、この早期リターンを先に行う。
    /// `backend-cpu::scan::cumsum`／`cumprod` と同じ順序）。
    fn run_scan(
        &self,
        x: &Tensor<f32>,
        dim: usize,
        kind: ScanOpKind,
    ) -> Result<Tensor<f32>, BackendError> {
        reduce_out_shape(x.shape(), Some(dim)).map_err(BackendError::ShapeMismatch)?;
        let shape = x.shape().to_vec();
        if shape.contains(&0) {
            return Tensor::new(Vec::new(), &shape).map_err(BackendError::ShapeMismatch);
        }
        crate::gather_scatter_model::checked_numel(&shape).map_err(BackendError::ShapeMismatch)?;
        // カーネル `uint` 引数（`lanes`／`axis_len`／`inner`）の上限超過を
        // `dispatch_sync` に入る前に先出しし、`Unsupported`（ホスト
        // フォールバック）へ写像する（`map_scan_prepare_error` 参照。
        // `Self::sort` の `plan_sort` 先出し・CUDA `ScanSizeLimitExceeded`
        // と同型。PR #1849 Cursor Bugbot 指摘の是正）。`scan.rs::
        // MetalScan::run_scan` 内の同じ `u32` 検査は「呼び出し元を信頼
        // しない」二重検査として残し、そこで失敗した場合は内部契約違反
        // として `KernelLaunchFailed` のまま伝播する。
        let plan = crate::scan_model::plan_scan(&shape, dim).map_err(map_scan_prepare_error)?;
        let (outer, axis_len, inner) = (plan.outer, plan.axis_len, plan.inner);

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("cumsum/cumprod: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let s = context_cache::cached_scan(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = match kind {
            ScanOpKind::Sum => s.run_cumsum_f32(&ctx, x_slice, outer, axis_len, inner),
            ScanOpKind::Prod => s.run_cumprod_f32(&ctx, x_slice, outer, axis_len, inner),
        }
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::sgd_step_device`]／
    /// [`BackendOps::sgd_step_device_tracked`] 共通の検証・ディスパッチ
    /// 本体（イシュー #1017 でトークン引数を追加する際に二重化を避ける
    /// ため切り出した。それ以前の検証ロジック自体は無変更）。
    fn sgd_step_device_impl(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
        token: Option<&DispatchFailureCell>,
    ) -> Result<(), BackendError> {
        if param.device() != Device::Metal || grad.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        if param.shape() != grad.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: param.shape().to_vec(),
                rhs: grad.shape().to_vec(),
            }));
        }
        let use_momentum = config.momentum != 0.0;
        if let Some(v) = &velocity {
            // デバイス不一致とテンソル shape 不一致を同一の
            // `ShapeMismatch` に丸めていた（Review 指摘。`backend-cuda`
            // 側 `ops.rs::sgd_step_device` と同型の問題）。
            // `BackendOps::sgd_step_device` の契約（`param`/`grad` と同じ
            // く、デバイス不一致は `DeviceMismatch` を返す）に velocity
            // も揃えるため、判定を分離する。
            if v.device() != Device::Metal {
                return Err(BackendError::DeviceMismatch);
            }
            if v.shape() != param.shape() {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: param.shape().to_vec(),
                    rhs: v.shape().to_vec(),
                }));
            }
        }
        if use_momentum && velocity.is_none() {
            return Err(BackendError::Unsupported(
                "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
            ));
        }

        let numel = param.numel();
        if numel == 0 {
            return Ok(());
        }

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let sgd = context_cache::cached_sgd(&ctx).map_err(map_metal_error)?;

        let grad_handle = grad
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(grad_metal_buf) = grad_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: grad buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let velocity_metal_buf = match &velocity {
            Some(v) => {
                let handle = v
                    .downcast_handle::<MetalBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                Some(handle.buffer.as_ref().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "sgd_step_device: velocity buffer has numel > 0 but no device allocation"
                            .into(),
                    )
                })?)
            }
            None => None,
        };

        let param_handle = param
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(param_metal_buf) = param_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "sgd_step_device: param buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let kernel_params = crate::sgd::SgdKernelParams {
            lr: config.lr,
            momentum: config.momentum,
            dampening: config.dampening,
            weight_decay: config.weight_decay,
            nesterov: config.nesterov,
            is_first_step: config.is_first_step,
        };
        sgd.run(
            &ctx,
            param_metal_buf,
            grad_metal_buf,
            velocity_metal_buf,
            numel,
            &kernel_params,
            token,
        )
        .map_err(map_metal_error)
    }

    /// [`Self::linear_forward_device`]／[`Self::
    /// linear_forward_device_tracked`] の共有本体（`gemm_fp32_strict_
    /// into_impl` と同じ二重化回避パターン）。`token` は
    /// [`gemm::MetalGemm::encode_strided_bias_act_prepared_with_c_
    /// offset`] へそのまま渡す。
    fn linear_forward_device_impl(
        &self,
        a: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
        token: Option<&DispatchFailureCell>,
    ) -> Result<fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Metal || w.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];
        if let Some(b) = bias {
            if b.device() != Device::Metal {
                return Err(BackendError::DeviceMismatch);
            }
            if b.shape() != [n] {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![n],
                }));
            }
        }
        let act_relu = match act {
            Activation::None => false,
            Activation::Relu => true,
            // `Activation` は `#[non_exhaustive]`。CPU／CUDA 実装と同じ
            // 方針で未知 variant を黙って恒等関数として扱わず明示的に
            // 拒否する。
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "linear_forward_device: unsupported activation {act:?}"
                )));
            }
        };
        if k == 0 {
            // `gemm_resident_rhs` と同じ理由（`Linear::new` が
            // `in_features == 0` を構築時に拒否するため到達不能）で
            // フォールバックを設けず型付きエラーで拒否する。
            return Err(BackendError::InvalidArgument(
                "linear_forward_device: k == 0 is unreachable via Linear::new (in_features == 0 \
                 is rejected at construction); a host epilogue fallback would require \
                 downloading the resident inputs, defeating the zero-D2H contract this method \
                 exists for"
                    .to_string(),
            ));
        }
        if m == 0 || n == 0 {
            return static_metal_memory()?.alloc_zeroed(&[m, n]);
        }

        let a_handle = a
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "linear_forward_device: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let a_layout = MatrixLayout {
            rows: m,
            cols: k,
            ld: k,
            transposed: false,
        };

        let w_handle = w
            .buffer()
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_buf) = w_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "linear_forward_device: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_layout = MatrixLayout {
            rows: k,
            cols: n,
            ld: n,
            transposed: false,
        };

        let bias_handle = bias
            .map(|b| {
                b.buffer()
                    .downcast_handle::<MetalBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)
                    .map(|h| (h, b.offset()))
            })
            .transpose()?;
        let bias_arg = match &bias_handle {
            Some((h, offset)) => {
                let buf = h.buffer.as_ref().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "linear_forward_device: bias buffer has numel > 0 but no device \
                         allocation"
                            .into(),
                    )
                })?;
                Some((buf, *offset))
            }
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        // 出力は呼び出し元へ escape するため `static_metal_memory()`（`memory_ops()`
        // と同一インスタンス）で確保する（本メソッド doc「出力バッファの
        // 確保元」参照。`gemm_resident_rhs` の一時 `MetalMemory::
        // from_shared` とは異なる）。
        let mem = static_metal_memory()?;
        let c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "linear_forward_device: output buffer has numel > 0 but no device allocation"
                    .into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        // `encode_strided_bias_act_prepared`（encode-only・待たない）を
        // 使う（本メソッド doc「同期契約」参照。codex-review・Cursor
        // Bugbot 指摘対応: `dispatch_strided_bias_act_prepared`
        //〈`ctx.dispatch_sync` 経由〉は呼ぶたびに `waitUntilCompleted`
        // するため、多層チェーンで層ごとの同期点が生じ「同期点を最終
        // `download` の 1 回へ集約する」契約を満たさなかった）。
        gemm.encode_strided_bias_act_prepared_with_c_offset(
            &ctx,
            a_buf,
            0,
            a_layout,
            w_buf,
            w.offset(),
            w_layout,
            bias_arg,
            act_relu,
            c_buf,
            0,
            m,
            n,
            k,
            token,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        // `download`／`synchronize` しない: 同期点は呼び出し元の
        // `download`（`synchronize`）へ集約される（本メソッド doc
        // 「同期契約」参照）。
        Ok(c_dev_buf)
    }
}

impl BackendOps for MetalBackendOps {
    fn device(&self) -> Device {
        Device::Metal
    }

    /// `static_metal_memory()`（プロセス内シングルトン）を返す
    /// （イシュー #935）。デバイス非対応等で初期化に失敗した場合は
    /// `None`（`memory_ops` のデフォルト契約と同じ fail-safe）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        static_metal_memory().ok().map(|m| m as &dyn MemoryOps)
    }

    /// `crate::typed_f16::MetalBackendOps`（`impl TypedOps<f16>`）への
    /// capability accessor（イシュー #1705）。`typed_ops_f64` は
    /// オーバーライドせず既定 `None`（恒久 `Unsupported`。MSL に
    /// `double` 型が存在せず構造的に不可。`docs/backend-dtype-
    /// dispatch-design.md` §5・§14）のまま残す。
    fn typed_ops_f16(&self) -> Option<&dyn fandhe_ai_tensor_core::TypedOps<half::f16>> {
        Some(self)
    }

    /// `crate::typed_bf16::impl TypedOps<half::bf16> for MetalBackendOps`
    /// （イシュー #1706）への accessor。`memory_ops`（上記）と異なり
    /// `TypedOps<bf16>` の実体は `self`（ZST）自身でありデバイス初期化に
    /// 一切触れないため、常に `Some(self)` を返す（`crate::typed_bf16`
    /// モジュール doc「accessor `typed_ops_bf16` は無条件に `Some(self)`
    /// を返す」参照。実行時の Metal 実機不在は各演算メソッド内部が型付き
    /// エラーで返す）。
    ///
    /// **`TypedOps` を top-level `use` しない**: `self.add`／`self.relu`
    /// 等の f32 専用内部呼び出しが `impl TypedOps<bf16>` の同名メソッド
    /// と衝突し「multiple applicable items in scope」で解決不能になる
    /// ため、戻り値型でのみ `fandhe_ai_tensor_core::TypedOps<half::bf16>`
    /// を完全修飾参照する（CPU／CUDA 側 `ops.rs` と同じ回避策）。
    fn typed_ops_bf16(&self) -> Option<&dyn fandhe_ai_tensor_core::TypedOps<half::bf16>> {
        Some(self)
    }

    /// [`fandhe_ai_tensor_core::CastOps`]（dtype 変換。イシュー #1751・
    /// 親 #1613・依存 #1750）の capability accessor。`typed_ops_bf16` と
    /// 同じ根拠（`impl CastOps for MetalBackendOps` の実体は `self`
    /// 自身であり、accessor 自体は Metal デバイスに一切触れない）で
    /// 無条件に `Some(self)` を返す。f64 2 方向は MSL `double` 非対応
    /// のため未オーバーライドのまま既定 `Unsupported` を返す
    /// （`crate::cast` モジュール doc 参照）。
    fn cast_ops(&self) -> Option<&dyn fandhe_ai_tensor_core::CastOps> {
        Some(self)
    }

    /// SGD の 1 パラメータ分の更新を in-place で実行する（イシュー #935・
    /// `docs/device-resident-update-design.md` §3.2・§5.2）。
    /// `context_cache::cached_sgd`（プロセス内 MSL コンパイル済みパイプ
    /// ラインキャッシュ）を経由するため、学習ループの 2 回目以降の
    /// ステップは再コンパイルを支払わない。
    ///
    /// 実体は `sgd_step_device_impl`（`token: None`）。
    /// [`BackendOps::sgd_step_device_tracked`] のオーバーライド
    /// （下記）とロジックを共有する（イシュー #1017。二重化しない）が、
    /// `token: None` を受けた `sgd.rs::MetalSgd::run` が `encode` 直後に
    /// `ctx.synchronize()` まで行うため、本メソッドは従来どおり
    /// **同期契約**（復帰時点で GPU 実行の完了・成否を返す）を保つ
    /// （PR #1057 レビュー指摘。バッチ化された非同期契約は
    /// `sgd_step_device_tracked` 限定）。
    fn sgd_step_device(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
    ) -> Result<(), BackendError> {
        self.sgd_step_device_impl(param, grad, velocity, config, None)
    }

    /// [`BackendOps::sgd_step_device_tracked`] の Metal オーバーライド
    /// （イシュー #1017・`docs/backend-metal-command-batching-design.md`
    /// §3.7）。`token` を `sgd_step_device_impl` → `sgd.rs::
    /// MetalSgd::run` → `context.rs::MetalContext::encode` へそのまま
    /// 渡し、encode と同一ロック区間でバッチへ登録させる。`token` が
    /// `Some` のため `MetalSgd::run` は `encode` 後に待たず、遅延実行
    /// （バッチ化）される非同期契約となる（`sgd_step_device` との違いは
    /// 同メソッド doc 参照）。
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::step`
    /// が呼び出し元となる（`device_store.rs` モジュール冒頭コメント
    /// 「遅延失敗トークン経由の poison」参照）。
    fn sgd_step_device_tracked(
        &self,
        param: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        grad: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        velocity: Option<&mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>>,
        config: &fandhe_ai_tensor_core::SgdStepConfig,
        token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.sgd_step_device_impl(param, grad, velocity, config, Some(token))
    }

    /// GEMM 本体（f32）。片側のみが転置 view（NT: `b` が転置・TN: `a` が
    /// 転置）の場合は `gemm_strided_nt_tn`（非公開ヘルパー。`dispatch_
    /// strided_bias_act_prepared` 経由の classic strided カーネル）へ
    /// 分岐し、
    /// ホスト側の転置再パックコピーを省く（イシュー #1215。呼び出し元は
    /// `autodiff::grad::matmul_vjp`／`Op::LinearResident` の VJP が
    /// `BackendOps::gemm_fp32_strict`〈既定実装から本メソッドへ委譲〉
    /// 経由で渡す転置 view）。NN（両方行優先）・TT（両方転置）・
    /// `layout::classify_2d` が分類できない形状（stride 0 の
    /// ブロードキャスト等）は従来どおり `contiguous()` +
    /// `dispatch_auto` へフォールバックする。
    ///
    /// **数値契約**: NN 経路は本イシュー導入前と bit 完全一致（カーネル
    /// 不変）。NT/TN 経路は `dispatch_auto`（`gemm_simdgroup_tiled`）とは
    /// 異なるカーネル（`gemm_tiled_bias_act`）を通るため累積順序が変わり
    /// うる——受け入れ判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または
    /// 絶対誤差 1e-5 未満。`gemm_resident_lhs`〈#1040〉と同じ契約）とし、
    /// `assert_eq!` によるビット一致は要求しない
    /// （`docs/matmul-vjp-zero-copy-decision.md` §4.4 参照）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::gemm_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        if let (Some(la), Some(lb)) = (
            layout::classify_2d(a.shape(), a.strides()),
            layout::classify_2d(b.shape(), b.strides()),
        ) && la.transposed != lb.transposed
            && a.as_view_slice().is_some()
            && b.as_view_slice().is_some()
        {
            return self.gemm_strided_nt_tn(a, la, b, lb, m, n, k, &out_shape);
        }

        // 従来経路（NN・TT・分類不能形状）。`contiguous()` が実際に
        // ホスト側コピーを発生させた場合のみ [`GEMM_HOST_REPACK_COUNT`]
        // を増やす（NN は元から contiguous のため増えない。TT・非対応
        // 形状のみ計上する）。
        if !a.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }
        if !b.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }

        // `dispatch_auto` は contiguous な `&[f32]` を要求する（CPU／CUDA
        // 実装と同じ契約）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: lhs not contiguous".into()))?;
        let b_slice = b_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("gemm: rhs not contiguous".into()))?;

        // コンテキスト取得（`context_cache::cached_context`）の失敗
        // （デバイス不在等）は `MetalDeviceProvider::select`（`device.rs`）
        // と同一分類の `BackendError::DeviceUnavailable` に統一する
        // （`map_metal_error` 経由。誤って `DeviceAllocationFailed`〈VRAM／
        // アロケータ起因〉に分類すると、呼び出し側が Metal デバイス不在を
        // 検知する経路が一方に偏る。Bugbot 指摘対応。PR #262 レビュー
        // スレッド。イシュー #930 でプロセス内キャッシュ経由へ変更した
        // 後もこの分類契約は不変）。
        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gemm
            .dispatch_auto(&ctx, a_slice, b_slice, m, n, k)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::gemm_batched`] の Metal オーバーライド（イシュー
    /// #1717・親 #1600・spec REQ-9 2026-09-12 追記 Tier 1「バッチ行列積」・
    /// `docs/compat-api-scope.md` §1.2）。
    ///
    /// 既定合成実装（`fandhe_ai_tensor_core::backend_ops::
    /// default_gemm_batched`。バッチをほどいて `batch_len` 回
    /// `self.gemm` を呼ぶ——各呼び出しが独自に upload・
    /// `dispatch_auto`〈内部 `ctx.synchronize()`〉・download する）が
    /// `batch_len` 回の GPU 同期・H2D／D2H を発生させるのに対し、本
    /// オーバーライドは **(a)** 正規化済みオペランド `a_norm`／`b_norm`
    /// （`[B, m, k]`／`[B, k, n]` の contiguous 3 次元。
    /// [`fandhe_ai_tensor_core::normalize_batched_operand`] が
    /// broadcast・`contiguous()` 化まで済ませる）を **1 回ずつ** upload
    /// し、**(b)** バッチごとの GEMM を
    /// `gemm::MetalGemm::encode_strided_bias_act_prepared_with_c_offset`
    /// （`gemm_fp32_strict_into`〈#1555〉・`linear_forward_device`
    /// 〈#1216〉・`gemm_resident_lhs`〈#1040〉が確立した encode-only
    /// パターン。`gemm.rs`／`shaders/gemm.metal` 自体は無変更）で
    /// **1 つのコマンドバッチへ encode するだけ**（待たない）で積み、
    /// **(c)** 最後の `download` 1 回だけが GPU 完了を待つ「バッチ
    /// ループ方式」にする。
    ///
    /// # 経路（`gemm.rs`・shader 無変更）
    ///
    /// 各バッチ `i` は NN レイアウト
    /// （`MatrixLayout { rows: m, cols: k, ld: k, transposed: false }`／
    /// `{ rows: k, cols: n, ld: n, transposed: false }`）で、オフセットは
    /// `a: i*m*k`・`b: i*k*n`・`c: i*m*n`（要素単位）。
    /// `a_norm.numel() == batch_len*m*k` 等は `normalize_batched_operand`
    /// が返す形状で構造的に保証されるため（`[B, rows, cols]` へ
    /// `reshape` 済み）、`i < batch_len` のオフセット積は
    /// `usize` オーバーフローしない。カーネル起動前の範囲検査自体は
    /// `gemm::validate_strided_dims_impl` が `a_buf.len()`／
    /// `b_buf.len()` に対するオフセット＋スパン、`c_offset + m*n <=
    /// c_len` を **バッチごとに** 行う（REQ-8「シェーダ・カーネル側の
    /// 手動境界チェックを省略しない」・OWASP A03）。
    ///
    /// # 数値契約
    ///
    /// rank 2 同士（`plan.batch_shape().is_empty()`）は [`Self::gemm`]
    /// へ直接委譲し bit 同一（既定合成実装・CPU オーバーライドと同じ
    /// 契約）。rank≥3 は classic strided カーネル
    /// （`gemm_tiled_bias_act`）を通るため、per-batch
    /// `MetalBackendOps::gemm`（`dispatch_auto` =
    /// `gemm_simdgroup_tiled`／split-K）とは **bit 同一を主張しない**
    /// （`gemm_strided_nt_tn`〈#1215〉・`gemm_collapsed_lhs`〈#1040〉と
    /// 同じ契約）。受け入れ判定は REQ-2 統一複合判定（相対誤差 1e-3
    /// 未満 または 絶対誤差 1e-5 未満）。
    ///
    /// 本経路は `dispatch_auto`／`crate::tile::select_route_for_device`
    /// を一切経由しないため、split-K 実行時トグル
    /// （`crate::split_k_runtime::split_k_enabled()`。既定 `true`。
    /// イシュー #1544・#1547）の状態に依存せず常に同一結果を返す
    /// （`tests/gemm_batched_parity.rs` の split-K トグル非干渉テストが
    /// 固定する）。
    ///
    /// # fail-closed（`DispatchFailureCell`）
    ///
    /// `sgd_step_device_tracked`／`gemm_fp32_strict_into_tracked` と
    /// 同様、GPU dispatch の失敗は encode 時点では判明せず
    /// `download` の同期点まで遅延する。本メソッドはローカルに
    /// `DispatchFailureCell` を作って各バッチの encode へ渡し、
    /// `download` 復帰後に `token.is_set()` を確認してから成功を
    /// 返す——共有 `MetalContext` を使う別スレッドが本メソッドより
    /// 先に `synchronize()` して失敗バッチを回収してしまっても、
    /// 本メソッドが誤って成功として返らないようにするための
    /// fail-closed 契約（`linear_forward_device` doc「同期契約」と
    /// 同型の懸念への対処）。
    ///
    /// # 退化形状（`m`／`n`／`k` が 0）
    ///
    /// `dispatch_auto` 経路の `validate_dims` は `m`／`n`／`k` が 0 の
    /// 形状を `MetalError::ZeroDimension` で拒否する契約だが（`Self::
    /// gemm_strided_nt_tn` doc 参照）、`encode_strided_bias_act_
    /// prepared_with_c_offset` が経由する `validate_strided_dims_impl`
    /// 自体はゼロ次元を拒否しない。本メソッドはゼロサイズの GPU
    /// dispatch を実際に発行することを避け、他バックエンド（CPU
    /// `gemm_batched`〈#1715〉・`gemm_collapsed_lhs`〈#1040〉）と挙動を
    /// 揃えるため、正規化（`normalize_batched_operand`。broadcast の
    /// 実体化を伴いうる）より前にホスト側で明示的に処理する
    /// （PR #1810 codex-review P2「空結果に対して broadcast を実体化
    /// しない」と同じ判断）: 出力要素数が 0（`batch_len == 0`／
    /// `m == 0`／`n == 0`）は空テンソルを、`k == 0`（`m, n > 0`）は
    /// GPU 起動なしの全 0 テンソルを返す。
    fn gemm_batched(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let plan = fandhe_ai_tensor_core::batched_matmul_plan(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;

        if plan.batch_shape().is_empty() {
            return self.gemm(a, b);
        }

        let (m, k, n) = (plan.m(), plan.k(), plan.n());
        let batch_len: usize = plan.batch_shape().iter().product();

        let mn = m
            .checked_mul(n)
            .ok_or(ShapeError::ElementCountOverflow)
            .map_err(BackendError::ShapeMismatch)?;
        let total = fandhe_ai_tensor_core::checked_gemm_batched_output_len(batch_len, m, n)
            .map_err(BackendError::ShapeMismatch)?;

        // 出力要素数 0（`batch_len == 0`／`m == 0`／`n == 0`）は
        // 正規化（broadcast 実体化）へ入らず即座に空テンソルを返す
        // （CPU `gemm_batched`〈#1715〉と同じ判断。上記 doc「退化形状」
        // 参照）。
        if total == 0 {
            return Tensor::new(Vec::new(), &plan.out_shape()).map_err(BackendError::ShapeMismatch);
        }

        if k == 0 {
            // `m, n > 0` かつ `k == 0`: 内積の項が 0 個のため結果は
            // 恒等的に 0。GPU dispatch を発行せずホスト側で直接構築する
            // （上記 doc「退化形状」参照。他バックエンドとのゼロ次元
            // 挙動整合が目的で、`validate_strided_dims_impl` 自体は
            // ゼロ次元を拒否しない）。
            return Tensor::from_shape_fill(&plan.out_shape(), |_| 0.0f32)
                .map_err(BackendError::ShapeMismatch);
        }

        // ホスト側再パックが発生するか（`Self::gemm` の NN/TT・分類
        // 不能形状と同じ計上規則。`normalize_batched_operand` は
        // `operand` が元から `[out_batch_shape, rows, cols]` と同一
        // shape で contiguous なら `Arc` 共有のみで実コピーを伴わない）。
        if !a.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }
        if !b.is_contiguous() {
            GEMM_HOST_REPACK_COUNT.with(|c| c.set(c.get() + 1));
        }

        let a_norm = fandhe_ai_tensor_core::normalize_batched_operand(a, plan.batch_shape(), m, k)?;
        let b_norm = fandhe_ai_tensor_core::normalize_batched_operand(b, plan.batch_shape(), k, n)?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());

        let a_dev_buf = mem.upload(&a_norm)?;
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_batched: lhs buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let b_dev_buf = mem.upload(&b_norm)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_batched: rhs buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let c_dev_buf = mem.alloc_zeroed(&[batch_len, m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_batched: out buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        let a_layout = MatrixLayout {
            rows: m,
            cols: k,
            ld: k,
            transposed: false,
        };
        let b_layout = MatrixLayout {
            rows: k,
            cols: n,
            ld: n,
            transposed: false,
        };

        // fail-closed トークン（上記 doc 参照）。本メソッド専用の
        // ローカルインスタンスで、`download` 復帰後に検査してから
        // 成功を返す。
        let token = DispatchFailureCell::new();
        for i in 0..batch_len {
            gemm.encode_strided_bias_act_prepared_with_c_offset(
                &ctx,
                a_buf,
                i * m * k,
                a_layout,
                b_buf,
                i * k * n,
                b_layout,
                None,
                false,
                c_buf,
                i * mn,
                m,
                n,
                k,
                Some(&token),
            )
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        }

        let out = mem.download(&c_dev_buf)?;
        if token.is_set() {
            return Err(BackendError::KernelLaunchFailed(
                "gemm_batched: GPU dispatch failed for one or more batches".into(),
            ));
        }
        out.reshape(&plan.out_shape())
            .map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::gemm_batched`] と同じバッチ行列積だが、各バッチの計算を
    /// [`Self::gemm_batched`] へそのまま委譲する（イシュー #1717）。
    ///
    /// trait doc（`fandhe_ai_tensor_core::backend_ops::BackendOps::
    /// gemm_batched_fp32_strict`）のとおり Metal は TF32 の概念を持た
    /// ず、rank 2 の `gemm_fp32_strict` は既定で `gemm` へ委譲される
    /// （`Self` は `gemm_fp32_strict` をオーバーライドしていない）。
    /// 本オーバーライドを置かない場合、`gemm_batched_fp32_strict` は
    /// 既定合成実装（バッチをほどいて `batch_len` 回 `gemm_fp32_strict`
    /// を呼ぶ）へ落ち、`autodiff::grad::matmul_vjp` の rank≥3 経路
    /// （`ops.gemm_batched_fp32_strict(...)` を呼ぶ）が forward
    /// （`Self::gemm_batched`）と異なるカーネル・per-batch 同期の経路を
    /// 通ってしまう（`docs/perf/train-backward-gemm-wiring.md` と同種の
    /// 「VJP がホスト scalar 経由に落ちる」問題を防ぐための明示的な
    /// 委譲）。
    fn gemm_batched_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.gemm_batched(a, b)
    }

    /// [`BackendOps::gemm_fp32_strict_into`] の Metal 実装（イシュー
    /// #1555・`docs/device-resident-update-design.md` 追補）。`Self::gemm`
    /// （イシュー #1215）の NT/TN 判定条件（`layout::classify_2d` が両方
    /// `Some` かつ `transposed` が互いに異なり、かつ両方
    /// `as_view_slice().is_some()`）を満たす形状は encode-only の直接
    /// 書き込み経路（下記）を使い、**それ以外（NN・TT・分類不能形状）は
    /// `Unsupported` を返さずホスト経路へフォールバックする**
    /// （codex-review 指摘・PR #1556。理由は下記「`Unsupported` を
    /// 返さない理由」参照）。
    ///
    /// - **NT/TN 経路のみ encode-only にできる理由**: `gemm`（NN 経路）は
    ///   `dispatch_auto`（split-K 実行時トグル込み。`Self::gemm` doc
    ///   参照）を経由しないと `gemm_fp32_strict`（`gemm` の既定委譲先）と
    ///   bit 同一にならない——本メソッドが使う
    ///   `dispatch_strided_bias_act_prepared` 系の classic カーネルは
    ///   split-K を持たないため NN の encode-only 化はしない。TT・
    ///   分類不能形状も `contiguous()`（ホスト側転置コピー）を経由する
    ///   必要があり、encode-only の zero-copy 化ができない。
    /// - **`Unsupported` を返さない理由**: `resident_grad_capability`
    ///   （呼び出し元 `DeviceParamStore::fill_resident_weight_grad`。
    ///   `crates/autodiff/src/optim/device_store.rs`）はストア全体で
    ///   1 度だけ確定するフラグであり、形状単位の非対応を
    ///   `Unsupported` として返すと、同じ backward の中で先に別の層が
    ///   NT/TN 経路で resident slot への書き込みに成功していても
    ///   `resident_grad_capability` が `Some(false)` に倒れ、以後
    ///   `param_grads_to_host`／`resident_grads_to_host` がストア全体の
    ///   resident slot を無視するため、既に充填済みの勾配が
    ///   `Gradients` 側に無く `MissingGradient` で失敗する（例:
    ///   `Linear(1, 8) → Linear(8, 4)` の 2 層 MLP。前段の d_weight は
    ///   `x_t` の strides が `[1, 1]` になり NN 扱いで分類不能、後段は
    ///   NT/TN で対応）。本メソッドは NN・TT・分類不能形状を「バックエンド
    ///   非対応」ではなく「encode-only 化できないだけの形状」として扱い、
    ///   `self.gemm_fp32_strict(a, b)`（trait 契約上 `Self::gemm` と bit
    ///   同一）の結果を [`MemoryOps::upload_into`] で `out` へ書き込む
    ///   ことで常に成功させ、`resident_grad_capability` が
    ///   `Some(false)` へ倒れることを防ぐ（デバイス不一致・範囲外
    ///   オフセットは従来どおり `DeviceMismatch`／`InvalidArgument`）。
    ///
    /// **同期の回収（NT/TN encode-only 経路）**: `Self::gemm_strided_nt_tn`
    /// （`gemm` の NT/TN 経路）
    /// は `dispatch_strided_bias_act_prepared`（内部で `ctx.synchronize()`
    /// する dispatch 版）→ `download` という 2 段の同期を経る。本メソッドは
    /// `gemm::MetalGemm::encode_strided_bias_act_prepared_with_c_offset`
    /// （encode-only 版。`Self::linear_forward_device`〈イシュー #1216〉と
    /// 同型のバッチ結合パターン）で GPU dispatch をバッチへ積むだけで
    /// 復帰し、`out` への書き込み完了は呼び出し元
    /// （`DeviceParamStore::step`／`resident_grads_to_host` 等）が後段の
    /// `MemoryOps::download`／`upload_into` で行う `synchronize()` まで
    /// 遅延する。GPU 側の dispatch 失敗（`CommandBufferExecutionFailed`
    /// 等）もその同期点まで遅延して表面化する（`linear_forward_device`
    /// doc「同期契約」と同一の契約）。
    ///
    /// **境界検査の順序**: `out.device()` → shape（`gemm_out_shape`）→
    /// `out_offset + m*n <= out.numel()`（REQ-8・OWASP A03）→ NT/TN 判定、
    /// の順に検査する。範囲外オフセットは NN 等の非対応形状であっても
    /// 常に `InvalidArgument`（`Unsupported` へ静かに丸めない。
    /// fail-closed）。
    ///
    /// **`upload_view` 一時バッファの生存**: `a_dev_buf`/`b_dev_buf`
    /// （`MetalMemory::upload_view`。`Backing::Owned`）は本メソッド復帰時
    /// に Rust 側スコープを抜けるが、
    /// `encode_strided_bias_act_prepared_with_c_offset` が `ctx.encode` の
    /// `resources` へ `raw()` を渡しており、その裏の `MTLBuffer` は
    /// `Batch::in_flight` の retain（`context.rs` の同種コメント参照）に
    /// よって GPU 完了まで生存するため、Rust 側の drop（Objective-C
    /// 参照カウントの減算）と衝突しない。
    ///
    /// **`resident_grad_capability`（呼び出し元 `DeviceParamStore::
    /// fill_resident_weight_grad`）との関係**: 本メソッドは NN・TT・
    /// 分類不能形状を含め `DeviceMismatch`／`InvalidArgument` 以外では
    /// 失敗しないため、`resident_grad_capability` は Metal では常に
    /// `Some(true)`（または致命的なデバイスエラー）へ確定する。CPU も
    /// 常に成功するため実質的な挙動は揃う（`docs/device-resident-update-
    /// design.md` §4「ホスト読み出し公開 API」・`crates/autodiff/src/
    /// optim/device_store.rs` 該当コメント参照）。
    ///
    /// 実体は `gemm_fp32_strict_into_impl`（`token: None`）。
    /// [`BackendOps::gemm_fp32_strict_into_tracked`] のオーバーライド
    /// （下記）とロジックを共有する（`sgd_step_device`／
    /// `sgd_step_device_tracked` と同じ二重化回避パターン。イシュー
    /// #1017・#1555）。
    fn gemm_fp32_strict_into(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        out_offset: usize,
    ) -> Result<(), BackendError> {
        self.gemm_fp32_strict_into_impl(a, b, out, out_offset, None)
    }

    /// [`BackendOps::gemm_fp32_strict_into_tracked`] の Metal オーバー
    /// ライド（イシュー #1555・codex-review 指摘・PR #1556）。`token` を
    /// `gemm_fp32_strict_into_impl` → `gemm::MetalGemm::
    /// encode_strided_bias_act_prepared_with_c_offset` → `context.rs::
    /// MetalContext::encode` へそのまま渡し、encode と同一ロック区間で
    /// バッチへ登録させる（`sgd_step_device_tracked` と同一の設計。
    /// トレイト側 doc「共有 `MetalContext` を使う別スレッドが先に
    /// `synchronize()` して...」参照）。
    fn gemm_fp32_strict_into_tracked(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        out_offset: usize,
        token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.gemm_fp32_strict_into_impl(a, b, out, out_offset, Some(token))
    }

    /// [`BackendOps::gemm_fp32_strict_into_with_bias_reduce_tracked`] の
    /// Metal 実装（イシュー #1566・`docs/backend-metal-command-batching-
    /// design.md` §10「案 A′」）。`gemm_fp32_strict_into_impl` と同じ
    /// NT/TN 判定条件（`layout::classify_2d` が両方 `Some` かつ
    /// `transposed` が互いに異なり、かつ両方 `as_view_slice().is_some()`）
    /// を使うが、bias 縮約も同時に試みる:
    ///
    /// - **NT/TN**: `gemm::MetalGemm::encode_weight_and_bias_grad_with_
    ///   offsets`（**同一 `ctx.encode` 呼び出し**で weight・bias 両方を
    ///   dispatch する。encode-only・待たない）へ委譲する。
    /// - **NN/TT・分類不能形状**: `gemm_fp32_strict_into_impl` と同じ
    ///   理由（`Unsupported` を返さない理由。同メソッド doc 参照）で
    ///   ホスト経路にフォールバックする。weight は
    ///   `self.gemm_fp32_strict(a, b)` → `upload_into`（既存と同型）。
    ///   bias は `b`（`g`）を `contiguous()` してから
    ///   `layout::reduce_bias_grad_rows_host`（行優先 `[m, n]` として
    ///   読む。`grad::reduce_to_shape` の rank-2→rank-1 特殊ケースと
    ///   同一アルゴリズムを独立実装。同関数 doc 参照）で計算し、
    ///   `upload_into` で `out` の `bias_offset` へ書き込む。この
    ///   2 回目の `upload_into`（weight に続く）は `docs/backend-metal-
    ///   command-batching-design.md` §10.2-3「`committed` が空なら
    ///   `synchronize` は早期リターン」により、NN/TT 経路が既に weight
    ///   の `upload_into` で同期済みのため事実上 no-op wait であり、
    ///   新規の恒久的な同期増加にはならない。
    ///
    /// **数値方式（2026-09-12 ユーザー承認 A・PR #1659 codex-review P1
    /// 是正）**: `.claude/rules/coding-rust.md` の勾配長軸縮約 `f64`
    /// アキュムレータ方針に従い、ホスト経路 `reduce_bias_grad_rows_host`
    /// は `f64` アキュムレータへ統一済み。GPU カーネル
    /// `gemm_bias_grad_reduce_f32`（`double` 非対応の Metal）は IEEE 754
    /// binary64 の逐次加算を 64bit 整数演算でソフトウェアエミュレート
    /// し（ホスト側逐語モデル `crate::soft_f64`）、同じ演算列を辿る。
    /// このため両経路は bit 完全一致する契約（`docs/backend-metal-
    /// command-batching-design.md` §10.14・`shaders/gemm.metal::
    /// gemm_bias_grad_reduce_f32` 冒頭コメント）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp32_strict_into_with_bias_reduce_tracked(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        out_offset: usize,
        bias: Option<(usize, usize)>,
        token: &DispatchFailureCell,
    ) -> Result<bool, BackendError> {
        if out.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let out_shape = fandhe_ai_tensor_core::gemm_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let _ = out_shape;

        // REQ-8・OWASP A03: weight・bias 双方の書き込み範囲（および両者の
        // 重複禁止。PR #1659 codex-review P1 是正: 従来は重複を検証して
        // おらず、weight・bias が同一領域を指すと bias の書き込みが
        // weight の一部を無言で上書きしていた）をカーネル起動・NT/TN
        // 判定より前に検証する（`gemm_fp32_strict_into_impl` と同じ順序
        // 規約）。検証本体（`m * n`／オフセット加算のオーバーフロー・
        // バッファ範囲超過・weight/bias 範囲重複）は Linux 実行可能な
        // 単体テストを持つ純関数 `layout::validate_gemm_bias_write_ranges`
        // に委譲する。
        layout::validate_gemm_bias_write_ranges(m, n, out_offset, out.numel(), bias)?;

        let layout_pair = match (
            layout::classify_2d(a.shape(), a.strides()),
            layout::classify_2d(b.shape(), b.strides()),
        ) {
            (Some(la), Some(lb))
                if la.transposed != lb.transposed
                    && a.as_view_slice().is_some()
                    && b.as_view_slice().is_some() =>
            {
                Some((la, lb))
            }
            _ => None,
        };

        let Some((la, lb)) = layout_pair else {
            // NN・TT・分類不能形状: `gemm_fp32_strict_into_impl` の
            // フォールバックと同じ理由で `Unsupported` を返さず常に
            // 成功させる（同メソッド doc「`Unsupported` を返さない
            // 理由」参照）。
            let result = self.gemm_fp32_strict(a, b)?;
            let ctx = context_cache::cached_context().map_err(map_metal_error)?;
            let mem = MetalMemory::from_shared(ctx);
            mem.upload_into(&result, out, out_offset)?;

            let Some((bias_offset, bn)) = bias else {
                return Ok(false);
            };
            let b_owned = b.contiguous();
            let b_slice = b_owned.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed(
                    "gemm_fp32_strict_into_with_bias_reduce_tracked: bias reduce source not \
                     contiguous after contiguous()"
                        .into(),
                )
            })?;
            // `contiguous()` 後は行優先密配置（`ld == n`・非転置）のため
            // `MatrixLayout` を直接構成できる（`layout::classify_2d` を
            // 経由する必要はない。値は `classify_2d(b.shape(),
            // b.contiguous().strides())` が返すものと同一）。
            let contiguous_layout = MatrixLayout {
                rows: k,
                cols: n,
                ld: n,
                transposed: false,
            };
            let contribution = layout::reduce_bias_grad_rows_host(b_slice, &contiguous_layout)?;
            debug_assert_eq!(contribution.len(), bn);
            let contribution_tensor =
                Tensor::new(contribution, &[bn]).map_err(BackendError::ShapeMismatch)?;
            let ctx2 = context_cache::cached_context().map_err(map_metal_error)?;
            let mem2 = MetalMemory::from_shared(ctx2);
            mem2.upload_into(&contribution_tensor, out, bias_offset)?;
            return Ok(true);
        };

        // NT/TN: encode-only（`gemm_fp32_strict_into_impl` と同じ
        // アップロード・ハンドル取得手順。bias 縮約も同一 `ctx.encode`
        // 呼び出しへ含める）。
        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());

        let a_slice = a.as_view_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed(
                "gemm_fp32_strict_into_with_bias_reduce_tracked: lhs not contiguous".into(),
            )
        })?;
        let a_dev_buf = mem
            .upload_view(a_slice, a.shape())
            .map_err(map_metal_error)?;
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_fp32_strict_into_with_bias_reduce_tracked: lhs buffer has numel > 0 but \
                 no device allocation"
                    .into(),
            ));
        };

        let b_slice = b.as_view_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed(
                "gemm_fp32_strict_into_with_bias_reduce_tracked: rhs not contiguous".into(),
            )
        })?;
        let b_dev_buf = mem
            .upload_view(b_slice, b.shape())
            .map_err(map_metal_error)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_fp32_strict_into_with_bias_reduce_tracked: rhs buffer has numel > 0 but \
                 no device allocation"
                    .into(),
            ));
        };

        let out_handle = out
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(out_buf) = out_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_fp32_strict_into_with_bias_reduce_tracked: out buffer has numel > 0 but \
                 no device allocation"
                    .into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let bias_written = gemm
            .encode_weight_and_bias_grad_with_offsets(
                &ctx,
                a_buf,
                0,
                la,
                b_buf,
                0,
                lb,
                out_buf,
                out_offset,
                bias,
                m,
                n,
                k,
                Some(token),
            )
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        Ok(bias_written)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、GEMM epilogue に bias 加算・
    /// activation を融合したカーネル
    /// （[`crate::gemm::MetalGemm::run_tiled_bias_act_f32`]）へ差し替える
    /// （イシュー #605）。CPU／CUDA 実装と同型の分岐（`gemm_bias_act_route`
    /// 参照）を採り、`bias` が `None` またはブロードキャストの厳密一致
    /// 形状 `[n]` の場合にのみ融合カーネルを使う。それ以外（`[1]`・
    /// `[1, n]` 等）はデフォルト実装と同じ 3 段合成（`self.gemm` →
    /// `self.add` → `self.relu`）へフォールバックする（本イシューで
    /// `add`／`relu` を実装済みのため CPU／CUDA と異なり `Unsupported` を
    /// 透過しない。モジュール冒頭コメント参照）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::gemm_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        let bias_shape = bias.map(|t| t.shape());
        match gemm_bias_act_route(bias_shape, n) {
            GemmBiasActRoute::ComposedFallback => {
                if let Some(bias) = bias {
                    // GEMM 本体を実行する前にブロードキャスト可否を検証
                    // する（CPU／CUDA 実装と同じ「カーネル本体アクセス前に
                    // 検証」の順序契約）。
                    fandhe_ai_tensor_core::broadcast_shape(&out_shape, bias.shape())
                        .map_err(BackendError::ShapeMismatch)?;
                }
                // `self.gemm`（`MetalGemm::dispatch_auto` 経由）は
                // `m`/`n`/`k == 0` を `ZeroDimension` として拒否する
                // （`gemm.rs::validate_dims`）が、CPU／CUDA の `gemm`
                // （`gemm_blis_parallel`・CUDA 側実装）はゼロ次元を合法な
                // 形状として受理しゼロ初期化バッファをそのまま返す。この
                // 非対称のため、ブロードキャスト bias（`[1]`・`[1, n]` 等。
                // `Fused` 経路に乗らない形状）かつゼロ次元の場合に
                // `self.gemm` を呼ぶと Metal のみ `ZeroDimension` で失敗し
                // `gemm_bias_act` の CPU／CUDA と共有される契約が破れる
                // （Cursor Bugbot 指摘。PR #717 レビュースレッド）。
                // `Fused` 経路（本関数末尾）は既にゼロ次元をホスト側
                // epilogue で受理しているため、ここでも `self.gemm` を
                // 経由せず CPU／CUDA と同じゼロ初期化 `m * n` 結果を直接
                // 構築することで両経路の zero-dim 挙動を揃える。
                let mut out = if m == 0 || n == 0 || k == 0 {
                    Tensor::new(vec![0.0f32; m * n], &out_shape)
                        .map_err(BackendError::ShapeMismatch)?
                } else {
                    self.gemm(a, b)?
                };
                if let Some(bias) = bias {
                    out = self.add(&out, bias)?;
                }
                out = match act {
                    Activation::None => out,
                    Activation::Relu => self.relu(&out)?,
                    // `Activation` は `#[non_exhaustive]`。CPU／CUDA 実装と
                    // 同じ方針で未知 variant を黙って恒等関数として扱わず
                    // 明示的に拒否する。
                    _ => {
                        return Err(BackendError::Unsupported(format!(
                            "gemm_bias_act: unsupported activation {act:?} in non-fused fallback path"
                        )));
                    }
                };
                Ok(out)
            }
            GemmBiasActRoute::Fused => {
                let act_relu = match act {
                    Activation::None => false,
                    Activation::Relu => true,
                    _ => {
                        return Err(BackendError::Unsupported(format!(
                            "gemm_bias_act: unsupported activation {act:?} in fused epilogue path"
                        )));
                    }
                };

                let a_owned = a.contiguous();
                let b_owned = b.contiguous();
                let a_slice = a_owned.as_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed("gemm_bias_act: lhs not contiguous".into())
                })?;
                let b_slice = b_owned.as_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed("gemm_bias_act: rhs not contiguous".into())
                })?;

                let bias_owned;
                let bias_slice = match bias {
                    Some(bias) => {
                        bias_owned = bias.contiguous();
                        Some(bias_owned.as_slice().ok_or_else(|| {
                            BackendError::KernelLaunchFailed(
                                "gemm_bias_act: bias not contiguous".into(),
                            )
                        })?)
                    }
                    None => None,
                };

                let ctx = context_cache::cached_context().map_err(map_metal_error)?;
                let gemm = context_cache::cached_gemm(&ctx)
                    .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
                let out = gemm
                    .run_tiled_bias_act_f32(&ctx, a_slice, b_slice, bias_slice, act_relu, m, n, k)
                    .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
                Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
            }
        }
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」）。`a`（活性化値）のみをホストから
    /// アップロードし、`w`／`bias` は [`crate::gemm::MetalGemm::
    /// dispatch_bias_act_prepared`]（イシュー #1022 で追加した
    /// prepared 版入口。#1023 でオフセット引数を追加し `DeviceBufferView`
    /// の部分範囲をそのまま `setBuffer:offset:` へ渡せるようにした）へ
    /// そのまま渡すことでこれらの download を発生させない（Apple Silicon
    /// の UMA・`StorageModeShared` のため CUDA のような明示同期は不要。
    /// `memory.rs` モジュールコメント参照）。
    fn gemm_resident_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];
        if let Some(b) = bias {
            if b.device() != Device::Metal {
                return Err(BackendError::DeviceMismatch);
            }
            if b.shape() != [n] {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![n],
                }));
            }
        }
        if k == 0 {
            // `fandhe_ai_autodiff::nn::linear::Linear::new` が
            // `in_features == 0` を構築時に拒否するため到達不能
            // （CPU／CUDA 実装の同分岐と同じ判断。`tensor-core::
            // backend_ops::BackendOps::gemm_resident_rhs` doc 参照）。
            return Err(BackendError::InvalidArgument(
                "gemm_resident_rhs: k == 0 is unreachable via Linear::new (in_features == 0 is \
                 rejected at construction); a host epilogue fallback would require downloading \
                 the resident bias, defeating the zero-D2H contract this method exists for"
                    .to_string(),
            ));
        }
        if m == 0 || n == 0 {
            return Tensor::new(Vec::new(), &[m, n]).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_buf) = w_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let bias_handle = bias
            .map(|b| {
                b.buffer()
                    .downcast_handle::<MetalBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)
                    .map(|h| (h, b.offset()))
            })
            .transpose()?;
        let bias_arg = match &bias_handle {
            Some((h, offset)) => {
                let buf = h.buffer.as_ref().ok_or_else(|| {
                    BackendError::DeviceAllocationFailed(
                        "gemm_resident_rhs: bias buffer has numel > 0 but no device allocation"
                            .into(),
                    )
                })?;
                Some((buf, *offset))
            }
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());
        // イシュー #1040: `a` が転置 view（`classify_2d` で分類可能）の
        // 場合は `Tensor::contiguous()`（ホスト側転置コピー）を経由せず
        // アップロードする（`upload_operand_for_resident_gemm` 参照）。
        let (a_dev_buf, a_layout) = upload_operand_for_resident_gemm(&mem, a)?;
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_layout = MatrixLayout {
            rows: k,
            cols: n,
            ld: n,
            transposed: false,
        };

        let c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_rhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx,
            a_buf,
            0,
            a_layout,
            w_buf,
            w.offset(),
            w_layout,
            bias_arg,
            false,
            c_buf,
            m,
            n,
            k,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        mem.download(&c_dev_buf)
    }

    /// `a`（デバイス常駐）・`w`（デバイス常駐）・`bias`（デバイス常駐・
    /// 任意）から `y = act(a @ w + bias)` を、入力・出力いずれも
    /// ホストへ実体化せずに計算する（イシュー #1216・`docs/inference-
    /// forward-fixed-cost-design.md` §3.2「段階 B」）。[`Self::
    /// gemm_resident_rhs`] と同じ融合カーネルの encode-only 版
    /// （[`crate::gemm::MetalGemm::encode_strided_bias_act_prepared`]。
    /// codex-review・Cursor Bugbot 指摘対応で同期版 [`crate::gemm::
    /// MetalGemm::dispatch_strided_bias_act_prepared`] から切替済み）を
    /// 使うが、`a` も呼び出し元が既にデバイスへ置いた [`fandhe_ai_tensor_core::buffer::DeviceBuffer`] として受け取り、結果も
    /// `DeviceBuffer` のまま返す点が異なる（`gemm_resident_rhs*` 系は
    /// 「`w`／`bias` のみ常駐」・本メソッドは「`a`／`w`／`bias`／戻り値の
    /// 全てが常駐」）。多層 MLP 推論チェーン（`fandhe_ai_autodiff::
    /// optim::device_store` の呼び出し元）が本メソッドを連鎖させる
    /// ことで、層間の D2H→H2D を発生させず最終出力の 1 回の `download`
    /// へ同期点を集約できる（trait 定義側 doc comment 参照）。
    ///
    /// **同期契約（イシュー #1017 コマンドバッファ共有との整合。#1216
    /// codex-review 指摘対応で encode-only 化）**:
    /// [`crate::gemm::MetalGemm::encode_strided_bias_act_prepared`] は
    /// `ctx.encode` でバッチへ積むのみで待たない（`ctx.dispatch_sync`
    /// を呼ぶ同期版と異なり、本メソッドの呼び出しごとに
    /// `waitUntilCompleted` が発生しない）。次層の dispatch は同一
    /// コマンドバッファ内
    /// （または `should_auto_flush` で分割された後続コマンドバッファ。
    /// 同一キューの serial 実行順）に積まれるため、前層出力 `c` を次層の
    /// `a` として読む順序は GPU 側で保証される。CPU 側からの読み取りは
    /// `download`（`synchronize`）のみで、Apple Silicon の UMA・
    /// `StorageModeShared` のため CUDA のような明示的なストリーム同期は
    /// 不要（`gemm_resident_rhs` doc 参照）。
    ///
    /// **出力バッファの確保元**: 呼び出し元へ escape する戻り値のため
    /// `static_metal_memory()`（`memory_ops()` と同一インスタンス・
    /// `context_cache::cached_context()` 共有）で確保する（REQ-14 の
    /// 単一計測系列。`docs/device-resident-update-design.md` §3.3d）。
    fn linear_forward_device(
        &self,
        a: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>, BackendError> {
        self.linear_forward_device_impl(a, w, bias, act, None)
    }

    /// [`BackendOps::linear_forward_device_tracked`] の Metal オーバー
    /// ライド（イシュー #1688 codex-review 指摘対応）。`token` を
    /// `linear_forward_device_impl` → `gemm::MetalGemm::
    /// encode_strided_bias_act_prepared_with_c_offset` → `context.rs::
    /// MetalContext::encode` へそのまま渡し、`encode` と同一ロック区間で
    /// バッチへ登録させる（`gemm_fp32_strict_into_tracked`／
    /// `sgd_step_device_tracked` と同一の設計。トレイト側 doc「共有
    /// `MetalContext` を使う別スレッドが先に `synchronize()` して...」
    /// 参照）。既定実装（`token` を無視して `linear_forward_device` へ
    /// 委譲するのみ）のままでは、`encode_strided_bias_act_prepared`
    /// （`token: None`）経由のため本メソッドが表す dispatch の失敗が
    /// どの `DispatchFailureCell` にも登録されず、共有 `MetalContext`
    /// を使う別スレッドが先に `synchronize()` して失敗バッチを回収して
    /// しまうと、本経路の `download`（次層以降の `synchronize`）は
    /// エラーを検出できず不正な結果を成功として返しかねない
    /// （fail-closed 維持規約違反）ため、このオーバーライドが必須。
    fn linear_forward_device_tracked(
        &self,
        a: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
        token: &DispatchFailureCell,
    ) -> Result<fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>, BackendError> {
        self.linear_forward_device_impl(a, w, bias, act, Some(token))
    }

    /// `a op b`（`op` は [`BinaryElementwiseOp`]）を `MetalBuffer`
    /// 常駐のまま計算する（イシュー #1584）。`elementwise::
    /// MetalElementwise::dispatch_binary_resident` へ委譲する（同メソッド
    /// doc「同期契約」参照: `MetalContext::dispatch_sync` を使うため
    /// 呼び出しごとに 1 回同期する。CUDA 版の「同期点を呼び出し元の
    /// `download` へ集約する」契約とは異なる）。`a`／`b` は shape 完全
    /// 一致限定（ブロードキャスト非対応）で、不一致は起動前に
    /// `ShapeMismatch` で拒否する。
    fn binary_elementwise_device(
        &self,
        op: BinaryElementwiseOp,
        a: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
        b: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
    ) -> Result<fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Metal || b.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        if a.shape() != b.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a.shape().to_vec(),
                rhs: b.shape().to_vec(),
            }));
        }
        let shape = a.shape().to_vec();
        let numel = a.numel();

        if numel == 0 {
            return static_metal_memory()?.alloc_zeroed(&shape);
        }

        let a_handle = a
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "binary_elementwise_device: a buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let b_handle = b
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "binary_elementwise_device: b buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        // 出力は呼び出し元へ escape するため `static_metal_memory()`
        // （`linear_forward_device` の「出力バッファの確保元」と同じ
        // 判断。REQ-14 の単一計測系列）。
        let mem = static_metal_memory()?;
        let out_dev_buf = mem.alloc_zeroed(&shape)?;
        let out_handle = out_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(out_buf) = out_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "binary_elementwise_device: output buffer has numel > 0 but no device \
                 allocation"
                    .into(),
            ));
        };

        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        ew.dispatch_binary_resident(&ctx, op, a_buf, b_buf, out_buf, numel)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        Ok(out_dev_buf)
    }

    /// [`Self::binary_elementwise_device`] の単項版（イシュー #1584）。
    fn unary_elementwise_device(
        &self,
        op: UnaryElementwiseOp,
        a: &fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
    ) -> Result<fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let shape = a.shape().to_vec();
        let numel = a.numel();

        if numel == 0 {
            return static_metal_memory()?.alloc_zeroed(&shape);
        }

        let a_handle = a
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "unary_elementwise_device: a buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = static_metal_memory()?;
        let out_dev_buf = mem.alloc_zeroed(&shape)?;
        let out_handle = out_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(out_buf) = out_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "unary_elementwise_device: output buffer has numel > 0 but no device allocation"
                    .into(),
            ));
        };

        let ew = context_cache::cached_elementwise(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        ew.dispatch_unary_resident(&ctx, op, a_buf, out_buf, numel)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        Ok(out_dev_buf)
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。`Op::LinearResident` の VJP が `d_input^T = w @ g^T`
    /// を計算するために使う。`bias` なし（`None`）で [`Self::
    /// gemm_resident_rhs`] と同じ [`crate::gemm::MetalGemm::
    /// dispatch_bias_act_prepared`] を呼ぶ（`has_bias=0`・`act=0` で
    /// 純粋な `w @ b` になる）。
    ///
    /// **同期の合流点（イシュー #1563）**: 本メソッド末尾の `mem.download`
    /// は内部で `MetalContext::synchronize` を呼ぶ（`download` doc
    /// 参照）。呼び出し元 `crates/autodiff/src/grad.rs` の
    /// `Op::LinearResident` VJP は、この呼び出しより**前**に同じ層の
    /// d_weight（`DeviceParamStore::fill_resident_weight_grad` 経由の
    /// encode-only dispatch。個別に同期しない）を積むよう順序付けられて
    /// いる（両者は独立計算のため出力は bit 同一）。このため本メソッドの
    /// `synchronize` は d_input 自身だけでなく、直前に積まれた同じ層の
    /// d_weight の GPU コマンドも一緒に完了させる合流点として働く
    /// （`docs/backend-metal-command-batching-design.md` §7.4）。
    fn gemm_resident_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Metal {
            return Err(BackendError::DeviceMismatch);
        }
        let w_shape = w.shape();
        if w_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: w_shape.len(),
            }));
        }
        let (p, q) = (w_shape[0], w_shape[1]);
        let b_shape = b.shape();
        if b_shape.len() != 2 || b_shape[0] != q {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: w_shape.to_vec(),
                rhs: b_shape.to_vec(),
            }));
        }
        let r = b_shape[1];
        if p == 0 || r == 0 {
            return Tensor::new(Vec::new(), &[p, r]).map_err(BackendError::ShapeMismatch);
        }
        if q == 0 {
            // `w` の縮約次元（`out_features`）が 0 の場合、GEMM の数学的
            // 定義どおり結果は全 0（CPU／CUDA 実装の同分岐と同じ判断）。
            // GPU 起動を回避してホスト側で直接構築する。
            return Tensor::from_shape_fill(&[p, r], |_| 0.0).map_err(BackendError::ShapeMismatch);
        }

        let w_handle = w
            .buffer()
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(w_buf) = w_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: w buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());
        // イシュー #1040: `Op::LinearResident` の VJP は `transpose2d`
        // した upstream 勾配（転置 view）をここへ渡す。`classify_2d` で
        // 分類できる限りホスト側転置コピーなしでアップロードする
        // （`upload_operand_for_resident_gemm` 参照）。
        let (b_dev_buf, b_layout) = upload_operand_for_resident_gemm(&mem, b)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: b buffer has numel > 0 but no device allocation".into(),
            ));
        };
        let w_layout = MatrixLayout {
            rows: p,
            cols: q,
            ld: q,
            transposed: false,
        };
        let c_dev_buf = mem.alloc_zeroed(&[p, r])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_resident_lhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx,
            w_buf,
            w.offset(),
            w_layout,
            b_buf,
            0,
            b_layout,
            None,
            false,
            c_buf,
            p,
            r,
            q,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        mem.download(&c_dev_buf)
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, ctx, a_s, b_s| ew.run_add_f32(ctx, a_s, b_s))
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary(a, b, |ew, ctx, a_s, b_s| ew.run_mul_f32(ctx, a_s, b_s))
    }

    /// `BackendOps::where_cond` の Metal 実装（イシュー #1637）。
    /// `elementwise::MetalElementwise::run_where_f32` へ委譲する。
    fn where_cond(
        &self,
        cond: &Tensor<f32>,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_ternary(cond, a, b, |ew, ctx, cond_s, a_s, b_s| {
            ew.run_where_f32(ctx, cond_s, a_s, b_s)
        })
    }

    /// `BackendOps::masked_fill` の Metal 実装（イシュー #1637）。
    fn masked_fill(
        &self,
        x: &Tensor<f32>,
        mask: &Tensor<f32>,
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_binary_scalar(x, mask, value, |ew, ctx, x_s, mask_s, v| {
            ew.run_masked_fill_f32(ctx, x_s, mask_s, v)
        })
    }

    /// `BackendOps::scalar_unary` の Metal 実装（イシュー #1707・#1708）。
    /// `Sqrt`（#1707）＋超越関数系 8 kind（`Neg`／`Abs`／`Log`／`Log2`／
    /// `Log10`／`Sin`／`Cos`／`Tan`。#1708）実装済み（`crate::
    /// scalar_op_source` モジュール doc「スコープ」参照。他 kind は既定
    /// `Unsupported` のまま）。
    fn scalar_unary(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.scalar_unary_dispatch(op, a)
    }

    /// `BackendOps::scalar_binary` の Metal 実装（イシュー #1707）。
    /// `Sub`／`Div`／`Pow` のみ実装済み。
    fn scalar_binary(
        &self,
        op: ScalarBinaryOp,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.scalar_binary_dispatch(op, a, b)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, ctx, a_s| ew.run_relu_f32(ctx, a_s))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, ctx, a_s| ew.run_exp_f32(ctx, a_s))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.elementwise_unary(a, |ew, ctx, a_s| ew.run_tanh_f32(ctx, a_s))
    }

    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::sum: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::max: reduction カーネル未実装（TASK-1.9c スコープ外）".into(),
        ))
    }

    /// `min`（イシュー #1720）は `Self::sum`／`Self::max` と同じく
    /// reduction カーネル未実装。`min` は `BackendOps` のデフォルト
    /// メソッド（既定 `Unsupported`）のため本オーバーライドは機能上
    /// 必須ではないが、`sum`／`max` と横並びで明示し「Metal は sum／
    /// max／min いずれも未実装」という事実を観測しやすくする
    /// （`Var::min` はホスト参照実装〈`eval::min`〉へフォールバック
    /// するため、この非対称〈`Var::max` は既存の必須メソッド契約上
    /// フォールバックを持たずエラーとなる〉は既知の事実として記録
    /// する。実装計画 §7「スコープ外」参照）。
    fn min(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::min: reduction カーネル未実装（イシュー #1720 スコープ外）".into(),
        ))
    }

    /// `argmax`（イシュー #1720）は GPU カーネル未実装。`Var::argmax`
    /// はホスト参照実装（`eval::argmax`）へフォールバックする。
    fn argmax(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<i32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::argmax: argmax カーネル未実装（イシュー #1720 スコープ外）".into(),
        ))
    }

    /// [`Self::argmax`] と同じ理由・同じ方針（イシュー #1720 スコープ
    /// 外）。
    fn argmin(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<i32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::argmin: argmin カーネル未実装（イシュー #1720 スコープ外）".into(),
        ))
    }

    /// 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`）は
    /// GPU カーネル未実装（設計文書「スコープ外」節）。既定
    /// `Unsupported` を明示オーバーライドし、`device_handle()` を経由
    /// しない（`Self::sum`／`Self::max` と同じ方針。`backend-cuda::ops::
    /// CudaBackendOps` と同様の対称な整備）。
    fn linalg_inv(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_inv: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_solve(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_solve: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_det(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_det: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_cholesky(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_cholesky: 線形代数カーネル未実装（イシュー #1621 \
             スコープ外）"
                .into(),
        ))
    }

    fn linalg_qr(&self, _a: &Tensor<f32>) -> Result<QrFactors, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_qr: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_svd(&self, _a: &Tensor<f32>) -> Result<SvdFactors, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_svd: 線形代数カーネル未実装（イシュー #1621 スコープ外）"
                .into(),
        ))
    }

    fn linalg_matrix_norm(
        &self,
        _a: &Tensor<f32>,
        _ord: MatrixNormOrd,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "MetalBackendOps::linalg_matrix_norm: 線形代数カーネル未実装（イシュー #1621 \
             スコープ外）"
                .into(),
        ))
    }

    /// `BackendOps::gather` の Metal 実装（イシュー #1778）。
    /// [`gather_out_shape`] で `input`／`index` の shape を再検査し、
    /// [`validate_shapes_fit_u32`]（rank・shape 要素の `u32` 収容）・
    /// [`validate_index_range`]（`index` の値域）でカーネル起動前の
    /// fail-closed 検査を行ってから `crate::gather_scatter::
    /// MetalGatherScatter::run_gather_f32` へ委譲する（実装側の独立
    /// 検査。`Var::gather` も同じ検査を済ませているが判定迂回経路を
    /// 作らないため実装側でも独立に検査する。`.claude/rules/
    /// security.md` A08。`crates/backend-cpu/src/ops.rs::gather` と
    /// 同じ二重検査方針）。
    fn gather(
        &self,
        input: &Tensor<f32>,
        dim: usize,
        index: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = gather_out_shape(input.shape(), index.shape(), dim)
            .map_err(BackendError::ShapeMismatch)?;
        let in_shape = input.shape().to_vec();
        let index_shape = index.shape().to_vec();
        // `gather_out_shape` はテンソルとして有効な shape（rank 一致・
        // 非 dim 軸整合）であることまでしか検査しないため、rank が
        // カーネルのスタック配列上限（`GS_MAX_RANK`）を超える有効な
        // テンソルもここまで到達しうる。以前は `validate_shapes_fit_u32`
        // の `RankMismatch` をそのまま `ShapeMismatch` へ変換していた
        // ため、shape=[1; 9] のような有効な高階テンソルの gather も
        // 「shape が不正」として autodiff まで伝播していた（変更前は
        // ホスト実装〈`Var` 側フォールバック〉へ委譲できていたのに
        // 委譲できなくなる後退。codex-review 指摘。イシュー #1799）。
        // カーネル固有の対応上限は入力形状の不正と区別し、`Unsupported`
        // で返す（呼び出し元がホストフォールバックへ切り替えられる
        // ようにする）。
        if in_shape.len() > GS_MAX_RANK || index_shape.len() > GS_MAX_RANK {
            return Err(BackendError::Unsupported(format!(
                "gather: rank exceeds Metal kernel limit (GS_MAX_RANK={GS_MAX_RANK}): \
                 in_shape rank={}, index_shape rank={}",
                in_shape.len(),
                index_shape.len()
            )));
        }
        validate_shapes_fit_u32(&[&in_shape, &index_shape]).map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let index_owned = index.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gather: input not contiguous".into())
        })?;
        let index_slice = index_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gather: index not contiguous".into())
        })?;

        let dim_size = in_shape[dim];
        validate_index_range(index_slice, dim, dim_size).map_err(BackendError::ShapeMismatch)?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let gs = context_cache::cached_gather_scatter(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gs
            .run_gather_f32(&ctx, input_slice, &in_shape, index_slice, &index_shape, dim)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::pad` の Metal 実装（イシュー #1756）。[`pad_out_shape`]
    /// で `input.shape()`／`pads` を再検査し、shape 要素の `u32` 収容
    /// （カーネル引数 `constant uint*` は 32bit のため）をカーネル起動前
    /// に fail-closed 検査してから `crate::constant_pad::
    /// MetalConstantPad::run_pad_f32` へ委譲する（`gather` と同じ二重
    /// 検査方針。`.claude/rules/security.md` A08）。pad は `dim` 軸限定の
    /// gather／scatter と異なりカーネル側が固定長スタック配列
    /// （`GS_MAX_RANK`）を使わない設計（`shaders/constant_pad.metal`
    /// モジュール doc 参照）のため rank 上限は存在しない。**この u32
    /// 収容検査には gather／scatter 専用の [`validate_shapes_fit_u32`]
    /// を使わない**（同関数は `GS_MAX_RANK`〈8〉の rank 上限も併せて
    /// 課すため、rank 9 以上の pad 入力を `BackendError::ShapeMismatch`
    /// として fail し、`Unsupported` でないためホストフォールバックへ
    /// 切り替わらず CPU／CUDA〈rank 上限なし〉と非対称になっていた
    /// ——Cursor Bugbot 指摘）。
    fn pad(
        &self,
        input: &Tensor<f32>,
        pads: &[(usize, usize)],
        value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = pad_out_shape(input.shape(), pads).map_err(BackendError::ShapeMismatch)?;
        let in_shape = input.shape().to_vec();
        for &d in in_shape.iter().chain(out_shape.iter()) {
            if d > u32::MAX as usize {
                return Err(BackendError::ShapeMismatch(
                    ShapeError::ElementCountOverflow,
                ));
            }
        }

        // `input` が空でも出力は非空になりうる（全要素 `value`）ため、
        // その場合は `input` を実体化・GPU 転送せずに埋める
        // （`constant_pad.rs::MetalConstantPad::run_pad_f32` 内の同型
        // 早期リターンと対称。ここで先に判定することで `.contiguous()`
        // 呼び出し自体を回避する）。
        if in_shape.contains(&0) {
            return Tensor::new(vec![value; out_shape.iter().product()], &out_shape)
                .map_err(BackendError::ShapeMismatch);
        }

        let input_owned = input.contiguous();
        let input_slice = input_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("pad: input not contiguous".into()))?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let cp = context_cache::cached_constant_pad(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = cp
            .run_pad_f32(&ctx, input_slice, &in_shape, pads, &out_shape, value)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::im2col` の Metal 実装（イシュー #1768）。
    /// [`im2col_out_shape`] で `input.shape()`／`params` を再検査して
    /// から `im2col::MetalIm2col::run_im2col_f32` へ委譲する（`pad` と
    /// 同じ二重検査方針。`.claude/rules/security.md` A08）。`conv2d`
    /// 自身は override しない（`fandhe_ai_tensor_core::backend_ops::
    /// BackendOps::conv2d` doc・設計 `docs/conv-ops-design.md` §9）。
    fn im2col(
        &self,
        input: &Tensor<f32>,
        params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape =
            im2col_out_shape(input.shape(), params).map_err(BackendError::ShapeMismatch)?;

        // 出力が空なら（`N`／`Cin` 系の軸が 0。空間軸は
        // `im2col_out_shape` が事前に拒否する契約）`input` を読む必要
        // が一切ない（`pad` の空出力早期リターンと同じ理由）。
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch);
        }
        checked_bytes_for::<f32>(input.shape()).map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("im2col: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ic = context_cache::cached_im2col(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = ic
            .run_im2col_f32(&ctx, input_slice, input.shape(), &out_shape, params)
            .map_err(map_im2col_error)?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::col2im` の Metal 実装（イシュー #1768）。
    /// [`im2col_out_shape`] で `input_shape`／`params` から期待する
    /// `d_col` 形状（`[N, G, Cin_g·kH·kW, Hout·Wout]`）を導出し、
    /// `d_col.shape()` との完全一致を検査したうえで `input_shape`
    /// （＝戻り値として確保する出力 shape）自体のバイトサイズも
    /// `checked_bytes_for::<f32>` で検査してから `im2col::MetalIm2col::
    /// run_col2im_f32` へ委譲する（`im2col` と同じ二重検査方針。
    /// `backend-cuda::ops::CudaBackendOps::col2im` の PR #1862
    /// codex-review 是正と同じ理由で、`d_col` のサイズに依らず
    /// `input_shape` 自体を確保前検査する）。
    fn col2im(
        &self,
        d_col: &Tensor<f32>,
        input_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Tensor<f32>, BackendError> {
        let expected_col_shape =
            im2col_out_shape(input_shape, params).map_err(BackendError::ShapeMismatch)?;
        if d_col.shape() != expected_col_shape.as_slice() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: d_col.shape().to_vec(),
                rhs: expected_col_shape,
            }));
        }
        checked_bytes_for::<f32>(input_shape).map_err(BackendError::ShapeMismatch)?;

        if input_shape.contains(&0) {
            return Tensor::new(Vec::new(), input_shape).map_err(BackendError::ShapeMismatch);
        }
        checked_bytes_for::<f32>(d_col.shape()).map_err(BackendError::ShapeMismatch)?;

        let d_col_owned = d_col.contiguous();
        let d_col_slice = d_col_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("col2im: d_col not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ic = context_cache::cached_im2col(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = ic
            .run_col2im_f32(&ctx, d_col_slice, d_col.shape(), input_shape, params)
            .map_err(map_im2col_error)?;
        Tensor::new(out, input_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::scatter` の Metal 実装（イシュー #1778）。
    /// [`scatter_out_shape`] で `input`／`index`／`src` の shape を
    /// 再検査し、[`Self::gather`] と同じ二重検査方針
    /// （[`validate_shapes_fit_u32`]／[`validate_index_range`]）を
    /// 適用してから `crate::gather_scatter::MetalGatherScatter::
    /// run_scatter_f32` へ委譲する。未知 `ScatterReduce` variant への
    /// フォールバックは `run_scatter_f32` 側（`Overwrite` へ委譲）が
    /// 担う（CPU 参照実装と同じ安全側の割り切り方針）。
    fn scatter(
        &self,
        input: &Tensor<f32>,
        dim: usize,
        index: &Tensor<i32>,
        src: &Tensor<f32>,
        reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = scatter_out_shape(input.shape(), index.shape(), src.shape(), dim)
            .map_err(BackendError::ShapeMismatch)?;
        let index_shape = index.shape().to_vec();
        // `Self::gather` と同じ理由（カーネル固有の rank 上限は入力形状
        // の不正と区別し `Unsupported` を返す。codex-review 指摘。
        // イシュー #1799）。
        if out_shape.len() > GS_MAX_RANK || index_shape.len() > GS_MAX_RANK {
            return Err(BackendError::Unsupported(format!(
                "scatter: rank exceeds Metal kernel limit (GS_MAX_RANK={GS_MAX_RANK}): \
                 out_shape rank={}, index_shape rank={}",
                out_shape.len(),
                index_shape.len()
            )));
        }
        validate_shapes_fit_u32(&[&out_shape, &index_shape])
            .map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let index_owned = index.contiguous();
        let src_owned = src.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("scatter: input not contiguous".into())
        })?;
        let index_slice = index_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("scatter: index not contiguous".into())
        })?;
        let src_slice = src_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("scatter: src not contiguous".into())
        })?;

        let dim_size = out_shape[dim];
        validate_index_range(index_slice, dim, dim_size).map_err(BackendError::ShapeMismatch)?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let gs = context_cache::cached_gather_scatter(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gs
            .run_scatter_f32(
                &ctx,
                input_slice,
                &out_shape,
                index_slice,
                &index_shape,
                src_slice,
                dim,
                reduce,
            )
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::interpolate` の Metal 実装（イシュー #1757・#1762）。
    /// [`fandhe_ai_tensor_core::interpolate_out_shape_for_mode`] で
    /// `input`／`size`／`mode` の shape を再検査してから
    /// `interpolate.rs::MetalInterpolate` の各 `mode` 別カーネルへ
    /// 委譲する（`gather`／`scatter` と同じ二重検査方針）。
    /// interpolate カーネルは gather／scatter と異なり座標配列を
    /// 保持しないため `GS_MAX_RANK` の rank 上限は課さない
    /// （`shaders/interpolate.metal` 冒頭コメント参照）。`mode` の
    /// 未知 variant（`InterpolateMode` は `#[non_exhaustive]`）は
    /// `BackendError::Unsupported` を返す fail-safe（`ops.rs` 内他
    /// メソッドの未知 variant 分岐と同型）。
    fn interpolate(
        &self,
        input: &Tensor<f32>,
        size: &[usize],
        mode: InterpolateMode,
    ) -> Result<Tensor<f32>, BackendError> {
        match mode {
            InterpolateMode::Nearest | InterpolateMode::Bilinear { .. } => {}
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "MetalBackendOps::interpolate: 未対応の InterpolateMode variant {mode:?}"
                )));
            }
        }
        let out_shape = interpolate_out_shape_for_mode(input.shape(), size, mode)
            .map_err(BackendError::ShapeMismatch)?;
        let in_shape = input.shape().to_vec();

        // 出力が空なら `input` の shape に依らず結果は必ず空
        // （`gather`／`scatter` の同型早期リターンと同じ理由。空間軸は
        // `interpolate_out_shape_for_mode` により非 0 が保証されるため、
        // ここで 0 を含みうるのは先頭の残り軸のみ）。
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch);
        }

        // `.contiguous()` より前に、`f32` 換算バイト数が `Vec` の
        // 確保上限に収まるか検査する（巨大な `broadcast_to` view を
        // 小さい `size` へ縮小するケースの capacity overflow panic
        // 防止。`checked_bytes_for` doc 参照。イシュー #1834 Cursor
        // Bugbot 指摘）。
        checked_bytes_for::<f32>(input.shape()).map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("interpolate: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let ip = context_cache::cached_interpolate(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = match mode {
            InterpolateMode::Nearest => ip
                .run_nearest_f32(&ctx, input_slice, &in_shape, size, &out_shape)
                .map_err(map_interpolate_error)?,
            InterpolateMode::Bilinear { align_corners } => ip
                .run_bilinear_f32(
                    &ctx,
                    input_slice,
                    &in_shape,
                    size,
                    &out_shape,
                    align_corners,
                )
                .map_err(map_interpolate_error)?,
            _ => unreachable!("checked above"),
        };
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::one_hot` の Metal 実装（**非微分演算**。イシュー
    /// #1755）。[`one_hot_out_shape`] で `index.shape()`／`num_classes`
    /// を再検査し、[`validate_index_range`]（`index` の値域）でカーネル
    /// 起動前の fail-closed 検査を行ってから `crate::gather_scatter::
    /// MetalGatherScatter::run_one_hot_f32` へ委譲する（[`Self::
    /// gather`] と同じ二重検査方針）。`gather`／`scatter` と異なり
    /// カーネルが座標展開・ストライド（`shapes` バッファ）を一切
    /// 使わないため、[`Self::gather`]／[`Self::scatter`] が行う
    /// `GS_MAX_RANK`／[`validate_shapes_fit_u32`] 検査は不要
    /// （`shaders/gather_scatter.metal::one_hot_f32` 参照）。
    fn one_hot(
        &self,
        index: &Tensor<i32>,
        num_classes: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape =
            one_hot_out_shape(index.shape(), num_classes).map_err(BackendError::ShapeMismatch)?;
        let index_shape = index.shape().to_vec();

        let index_owned = index.contiguous();
        let index_slice = index_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("one_hot: index not contiguous".into())
        })?;

        validate_index_range(index_slice, index_shape.len(), num_classes)
            .map_err(BackendError::ShapeMismatch)?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let gs = context_cache::cached_gather_scatter(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gs
            .run_one_hot_f32(&ctx, index_slice, &index_shape, num_classes)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::unique` の Metal 実装（イシュー #1734）。
    /// `x.contiguous()` で稠密化してから `unique.rs::MetalUnique::
    /// run_unique_f32`（ビットニックソート方式）へ委譲する。契約
    /// （totalOrder ソート・`==` による重複判定）は
    /// `fandhe_ai_tensor_core::BackendOps::unique` doc を正とする。
    ///
    /// `padded`（次の 2 のべき乗）がカーネル引数の範囲を超える場合は
    /// `Self::gather`／`scatter` の `GS_MAX_RANK` 超過と同じ設計判断で
    /// `BackendError::Unsupported` を返す（呼び出し元 `Var::unique` の
    /// ホストフォールバックへ委ねる。入力形状の不正〈`ShapeMismatch`〉
    /// とバックエンド固有上限〈`Unsupported`〉を区別する。イシュー
    /// #1799 レビュー指摘と同型の判断）。
    fn unique(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        // `x.numel()`（内部で無検査の `.iter().product()` を使い
        // `overflow-checks` 有効ビルドで panic しうる）を呼ぶ前に
        // 要素数積のオーバーフローを検査する（PR #1828 codex-review
        // P1 是正。`gather`／`scatter` が使う
        // `gather_scatter_model::checked_numel` と同一の検査を再利用）。
        crate::gather_scatter_model::checked_numel(x.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let n = x.numel();
        if n >= 2
            && let Err(e) = crate::unique_model::checked_padded_len(n)
        {
            return Err(BackendError::Unsupported(e.to_string()));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("unique: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let u = context_cache::cached_unique(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = u
            .run_unique_f32(&ctx, x_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let m = out.len();
        Tensor::new(out, &[m]).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::cumsum` の Metal 実装（イシュー #1740）。数値契約
    /// （lane ごとの binary64 ソフトウェアエミュレーションアキュムレータ
    /// 逐次計算・CPU 参照実装と bit 完全一致）は `shaders/scan.metal`
    /// 冒頭コメント・`fandhe_ai_tensor_core::BackendOps::cumsum` doc が
    /// 正。`dim` を [`reduce_out_shape`] で再検査してから
    /// `x.numel()`（内部で無検査の `.iter().product()` を使い
    /// `overflow-checks` 有効ビルドで panic しうる）を呼ぶ前に要素数積
    /// のオーバーフローを検査する（`unique` と同じ `gather_scatter_
    /// model::checked_numel` 適用方針。PR #1828 codex-review P1 の
    /// 教訓と同型）。
    fn cumsum(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        self.run_scan(x, dim, ScanOpKind::Sum)
    }

    /// `BackendOps::cumprod` の Metal 実装（イシュー #1740）。[`Self::
    /// cumsum`] と同じ検査・委譲構造だが `MetalScan::run_cumprod_f32`
    /// を起動する。
    fn cumprod(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        self.run_scan(x, dim, ScanOpKind::Prod)
    }

    /// `BackendOps::sort` の Metal 実装（イシュー #1741）。
    /// `x.contiguous()` で稠密化してから `sort.rs::MetalSort::
    /// run_sort_f32`（64bit 合成キー・ビットニックソート方式。
    /// `crate::sort_model` モジュール doc 参照）へ委譲する。契約
    /// （安定ソート相当の順序契約・NaN／±0 正規化・決定性）は
    /// `fandhe_ai_tensor_core::BackendOps::sort` doc を正とする。
    ///
    /// 合成キー配列長がカーネル引数の範囲（`i32::MAX` 相当）を超える
    /// 場合は `Self::unique` の `padded` 超過と同じ設計判断で
    /// `BackendError::Unsupported` を返し、呼び出し元 `Var::sort` の
    /// ホストフォールバックへ委ねる。
    fn sort(
        &self,
        input: &Tensor<f32>,
        dim: usize,
        descending: bool,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        let out_shape = sort_out_shape(input.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        if out_shape.contains(&0) {
            return Ok((
                Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch)?,
                Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch)?,
            ));
        }
        // `x.numel()`（無検査の `.iter().product()`）を呼ぶ前に要素数積
        // のオーバーフローを検査する（`Self::unique` と同一の PR #1828
        // codex-review 是正パターン）。
        crate::gather_scatter_model::checked_numel(input.shape())
            .map_err(BackendError::ShapeMismatch)?;
        // `sort.rs::MetalSort::run_sort_f32` 内部の `plan_sort` 検証を
        // `dispatch_sync` に入る前に先出しし、失敗理由（形状不正／
        // バックエンド固有上限超過）を区別してから写像する（`map_sort_error`
        // 参照。CUDA 側 `ops.rs::map_sort_error` と同型の判断）。
        crate::sort_model::plan_sort(input.shape(), dim).map_err(map_sort_prepare_error)?;
        let dim_size = input.shape()[dim];

        let input_owned = input.contiguous();
        let input_slice = input_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("sort: input not contiguous".into()))?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let s = context_cache::cached_sort(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (values, index) = s
            .run_sort_f32(&ctx, input_slice, input.shape(), dim, descending, dim_size)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok((
            Tensor::new(values, &out_shape).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(index, &out_shape).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// `BackendOps::topk` の Metal 実装（イシュー #1741）。`Self::sort`
    /// と同一の `sort.rs::MetalSort::run_sort_f32` へ `out_len = k`・
    /// `descending = largest` として委譲する（`largest=false` の
    /// upside-down 契約は `fandhe_ai_tensor_core::BackendOps::topk` doc
    /// を正とする）。
    fn topk(
        &self,
        input: &Tensor<f32>,
        dim: usize,
        k: usize,
        largest: bool,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        let out_shape =
            topk_out_shape(input.shape(), dim, k).map_err(BackendError::ShapeMismatch)?;
        if out_shape.contains(&0) {
            return Ok((
                Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch)?,
                Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch)?,
            ));
        }
        crate::gather_scatter_model::checked_numel(input.shape())
            .map_err(BackendError::ShapeMismatch)?;
        crate::sort_model::plan_sort(input.shape(), dim).map_err(map_sort_prepare_error)?;

        let input_owned = input.contiguous();
        let input_slice = input_owned
            .as_slice()
            .ok_or_else(|| BackendError::KernelLaunchFailed("topk: input not contiguous".into()))?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let s = context_cache::cached_sort(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (values, index) = s
            .run_sort_f32(&ctx, input_slice, input.shape(), dim, largest, k)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok((
            Tensor::new(values, &out_shape).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(index, &out_shape).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss`] の Metal 実装
    /// （イシュー #1045）。`Self::sum`／`Self::max`（汎用 reduction）とは
    /// 独立した専用融合カーネル（`crate::mse::MetalMse`）へのディスパッチ
    /// （`backend_ops.rs::BackendOps::mse_loss` doc の設計判断参照）。
    /// `reduction` に応じた `factor`（`Mean` は `1.0/n`、`Sum` は `1.0`）は
    /// ここで計算してカーネルへ渡す。未知 `MseReduction` variant は
    /// `backend-cpu`／`backend-cuda` と同じく `Unsupported` として拒否
    /// する。
    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss: target not contiguous".into())
        })?;
        let numel = pred_slice.len();
        let factor = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    1.0
                } else {
                    1.0 / numel as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "mse_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mse = context_cache::cached_mse(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let value = mse
            .run_mse_loss_f32(&ctx, pred_slice, target_slice, factor)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss_backward`] の Metal
    /// 実装（イシュー #1045）。`dTarget = −dPred` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側で符号反転して得る
    /// 契約のため、本メソッドは `dPred` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::mse_loss_backward` doc 参照）。
    fn mse_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss_backward: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("mse_loss_backward: target not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mse = context_cache::cached_mse(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = mse
            .run_mse_backward_f32(&ctx, pred_slice, target_slice, scale)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::huber_loss`] の Metal 実装
    /// （イシュー #1739）。[`Self::mse_loss`] と同じ構成（専用融合カーネル
    /// `crate::huber::MetalHuber` へのディスパッチ・`factor` 計算・未知
    /// `MseReduction` variant の `Unsupported` 拒否）。
    fn huber_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: HuberKind,
        delta: f32,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("huber_loss: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("huber_loss: target not contiguous".into())
        })?;
        let numel = pred_slice.len();
        let factor = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    1.0
                } else {
                    1.0 / numel as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "huber_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let huber = context_cache::cached_huber(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let value = huber
            .run_huber_loss_f32(&ctx, pred_slice, target_slice, kind, delta, factor)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::huber_loss_backward`] の Metal
    /// 実装（イシュー #1739）。`dTarget = −dPred` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側で符号反転して得る
    /// 契約のため、本メソッドは `dPred` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::huber_loss_backward` doc 参照。
    /// [`Self::mse_loss_backward`] と同型）。
    fn huber_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: HuberKind,
        delta: f32,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_owned = pred.contiguous();
        let target_owned = target.contiguous();
        let pred_slice = pred_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("huber_loss_backward: pred not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("huber_loss_backward: target not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let huber = context_cache::cached_huber(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = huber
            .run_huber_backward_f32(&ctx, pred_slice, target_slice, kind, delta, scale)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::bce_loss`] の Metal 実装
    /// （イシュー #1737。`mse_loss` と同型の委譲構成）。
    fn bce_loss(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: BceKind,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let input_owned = input.contiguous();
        let target_owned = target.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("bce_loss: input not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("bce_loss: target not contiguous".into())
        })?;
        let numel = input_slice.len();
        let factor = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    1.0
                } else {
                    1.0 / numel as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "bce_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let bce = context_cache::cached_bce(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let value = bce
            .run_bce_loss_f32(&ctx, input_slice, target_slice, kind, factor)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::bce_loss_backward`] の Metal
    /// 実装（イシュー #1737）。`dTarget` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側の逐次 map で計算
    /// する契約のため、本メソッドは `dInput` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::bce_loss_backward` doc 参照）。
    fn bce_loss_backward(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: BceKind,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let input_owned = input.contiguous();
        let target_owned = target.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("bce_loss_backward: input not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("bce_loss_backward: target not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let bce = context_cache::cached_bce(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = bce
            .run_bce_backward_f32(&ctx, input_slice, target_slice, kind, scale)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, input.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::nll_loss`] の Metal 実装
    /// （イシュー #1738。`mse_loss` と同型の委譲構成）。
    fn nll_loss(
        &self,
        input: &Tensor<f32>,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        let shape = input.shape();
        if class_dim >= shape.len() {
            return Err(BackendError::ShapeMismatch(ShapeError::AxisOutOfRange {
                axis: class_dim,
                rank: shape.len(),
            }));
        }
        let layout = crate::nll::NllLayout {
            outer: checked_shape_numel(&shape[..class_dim]).map_err(BackendError::ShapeMismatch)?,
            num_classes: shape[class_dim],
            inner: checked_shape_numel(&shape[class_dim + 1..])
                .map_err(BackendError::ShapeMismatch)?,
        };
        let input_owned = input.contiguous();
        let targets_owned = targets.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("nll_loss: input not contiguous".into())
        })?;
        let targets_slice = targets_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("nll_loss: targets not contiguous".into())
        })?;
        let n = layout
            .outer
            .checked_mul(layout.inner)
            .ok_or(BackendError::ShapeMismatch(
                ShapeError::ElementCountOverflow,
            ))?;
        let factor = match reduction {
            MseReduction::Mean => {
                if n == 0 {
                    1.0
                } else {
                    1.0 / n as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "nll_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let nll = context_cache::cached_nll(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let value = nll
            .run_nll_loss_f32(&ctx, input_slice, targets_slice, layout, factor)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::nll_loss_backward`] の Metal
    /// 実装（イシュー #1738）。`targets` は非追跡のため `dInput` のみを
    /// 返す契約（`backend_ops.rs::BackendOps::nll_loss_backward` doc
    /// 参照）。
    fn nll_loss_backward(
        &self,
        input_shape: &[usize],
        targets: &Tensor<i32>,
        class_dim: usize,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if class_dim >= input_shape.len() {
            return Err(BackendError::ShapeMismatch(ShapeError::AxisOutOfRange {
                axis: class_dim,
                rank: input_shape.len(),
            }));
        }
        let layout = crate::nll::NllLayout {
            outer: checked_shape_numel(&input_shape[..class_dim])
                .map_err(BackendError::ShapeMismatch)?,
            num_classes: input_shape[class_dim],
            inner: checked_shape_numel(&input_shape[class_dim + 1..])
                .map_err(BackendError::ShapeMismatch)?,
        };
        let targets_owned = targets.contiguous();
        let targets_slice = targets_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("nll_loss_backward: targets not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let nll = context_cache::cached_nll(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = nll
            .run_nll_backward_f32(&ctx, targets_slice, layout, scale)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, input_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::kl_div_loss`] の Metal 実装
    /// （イシュー #1738。`mse_loss` と同型の委譲構成）。
    fn kl_div_loss(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: KlDivTarget,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let input_owned = input.contiguous();
        let target_owned = target.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("kl_div_loss: input not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("kl_div_loss: target not contiguous".into())
        })?;
        let numel = input_slice.len();
        let factor = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    1.0
                } else {
                    1.0 / numel as f32
                }
            }
            MseReduction::Sum => 1.0,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "kl_div_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let kl_div = context_cache::cached_kl_div(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let value = kl_div
            .run_kl_div_loss_f32(&ctx, input_slice, target_slice, kind, factor)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::kl_div_loss_backward`] の
    /// Metal 実装（イシュー #1738）。`dTarget` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側の逐次 map で計算
    /// する契約のため、本メソッドは `dInput` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::kl_div_loss_backward` doc 参照）。
    fn kl_div_loss_backward(
        &self,
        input: &Tensor<f32>,
        target: &Tensor<f32>,
        kind: KlDivTarget,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(input.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let input_owned = input.contiguous();
        let target_owned = target.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("kl_div_loss_backward: input not contiguous".into())
        })?;
        let target_slice = target_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("kl_div_loss_backward: target not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let kl_div = context_cache::cached_kl_div(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = kl_div
            .run_kl_div_backward_f32(&ctx, input_slice, target_slice, kind, scale)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, input.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::rmsnorm`] の Metal 実装
    /// （イシュー #1596）。既存の [`Self::run_fused`] 経由（`match_
    /// rmsnorm_plan` の canonical プラン一致限定・`mean` 化なし・`eps`
    /// なし・`weight` なし）とは別の独立エントリで、[`row_norm_layout`]
    /// で `(rows, hidden)` を導出してから [`crate::rmsnorm::MetalRmsNorm::
    /// run_rmsnorm_f32`]（`mean` 化・`eps`・任意 `weight` を含む標準
    /// RMSNorm）を直接呼ぶ（`run_fused_rmsnorm` と同じ
    /// `context_cache::cached_rmsnorm` キャッシュを再利用する）。
    fn rmsnorm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("rmsnorm: input not contiguous".into())
        })?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(w.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("rmsnorm: weight not contiguous".into())
            })?),
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rmsnorm = context_cache::cached_rmsnorm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = rmsnorm
            .run_rmsnorm_f32(&ctx, x_slice, w_slice, eps, rows, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::layer_norm`] の Metal 実装
    /// （イシュー #1596）。[`Self::rmsnorm`] と同じ `row_norm_layout`
    /// 導出だが、`run_fused`（canonical 融合プラン一致経路）への
    /// LayerNorm 一致経路は追加しない——LayerNorm は本エントリ経由でのみ
    /// 到達する（`docs/norm-ops-design.md`）。新設カーネル
    /// [`crate::layer_norm::MetalLayerNorm::run_layer_norm_f32`]・
    /// `context_cache::cached_layer_norm` を使う。
    fn layer_norm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (rows, hidden) = row_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("layer_norm: input not contiguous".into())
        })?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(w.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("layer_norm: weight not contiguous".into())
            })?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_slice = match &b_owned {
            Some(b) => Some(b.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("layer_norm: bias not contiguous".into())
            })?),
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let layer_norm = context_cache::cached_layer_norm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = layer_norm
            .run_layer_norm_f32(&ctx, x_slice, w_slice, b_slice, eps, rows, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::batch_norm_train`] の Metal
    /// 実装（イシュー #1736・親 #1608）。チャネル軸・レイアウト契約の
    /// 導出は `batch_norm_layout`（`row_norm_layout` の BatchNorm 版。
    /// チャネル軸は常に dim 1）。新設カーネル `crate::batch_norm::
    /// MetalBatchNorm::run_batch_norm_train_f32`・`context_cache::
    /// cached_batch_norm` を使う（`docs/batch-norm-ops-design.md`
    /// §3.1「Metal（1 simdgroup = 1 channel）が再現すべき契約」）。
    fn batch_norm_train(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<BatchNormTrainOutput, BackendError> {
        let (n, c, spatial) = batch_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("batch_norm_train: input not contiguous".into())
        })?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(w.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("batch_norm_train: weight not contiguous".into())
            })?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_slice = match &b_owned {
            Some(b) => Some(b.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("batch_norm_train: bias not contiguous".into())
            })?),
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let batch_norm = context_cache::cached_batch_norm(&ctx).map_err(map_batch_norm_error)?;
        let raw = batch_norm
            .run_batch_norm_train_f32(&ctx, x_slice, w_slice, b_slice, eps, n, c, spatial)
            .map_err(map_batch_norm_error)?;
        Ok(BatchNormTrainOutput {
            output: Tensor::new(raw.out, x.shape()).map_err(BackendError::ShapeMismatch)?,
            batch_mean: Tensor::new(raw.mean, &[c]).map_err(BackendError::ShapeMismatch)?,
            batch_var: Tensor::new(raw.var, &[c]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::batch_norm_infer`] の Metal
    /// 実装（イシュー #1736・親 #1608）。[`Self::batch_norm_train`] と
    /// 同じ shape 契約だが、`mean`／`var`（呼び出し元が保持する
    /// running stats）をバッチから計算し直さずそのまま使う。
    fn batch_norm_infer(
        &self,
        x: &Tensor<f32>,
        mean: &Tensor<f32>,
        var: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        let (n, c, spatial) = batch_norm_layout(x.shape()).map_err(BackendError::ShapeMismatch)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("batch_norm_infer: input not contiguous".into())
        })?;
        let mean_owned = mean.contiguous();
        let mean_slice = mean_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("batch_norm_infer: mean not contiguous".into())
        })?;
        let var_owned = var.contiguous();
        let var_slice = var_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("batch_norm_infer: var not contiguous".into())
        })?;
        let w_owned = weight.map(|w| w.contiguous());
        let w_slice = match &w_owned {
            Some(w) => Some(w.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("batch_norm_infer: weight not contiguous".into())
            })?),
            None => None,
        };
        let b_owned = bias.map(|b| b.contiguous());
        let b_slice = match &b_owned {
            Some(b) => Some(b.as_slice().ok_or_else(|| {
                BackendError::KernelLaunchFailed("batch_norm_infer: bias not contiguous".into())
            })?),
            None => None,
        };

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let batch_norm = context_cache::cached_batch_norm(&ctx).map_err(map_batch_norm_error)?;
        let out = batch_norm
            .run_batch_norm_infer_f32(
                &ctx, x_slice, mean_slice, var_slice, w_slice, b_slice, eps, n, c, spatial,
            )
            .map_err(map_batch_norm_error)?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::max_pool2d` の Metal 実装（イシュー #1730・
    /// 追従イシュー。#1607 ツリー）。[`pool2d_out_shape`] で
    /// `input.shape()`／`params` を再検査してから `pooling::
    /// MetalPooling::run_max_pool2d_f32` へ委譲する（`im2col` と同じ
    /// 二重検査方針。`.contiguous()` 前の `checked_bytes_for` も
    /// `im2col`／`interpolate` と同型で適用し、非空入力の要素数積が
    /// `isize::MAX` バイト相当を超える場合の `.contiguous()` 側
    /// capacity overflow panic を未然に防ぐ。Cursor Bugbot 指摘の
    /// 是正・PR #1888）。`索引` テンソルの dtype は `i32` 固定
    /// （`crate::pooling::MetalPooling::run_max_pool2d_f32` doc 参照）。
    fn max_pool2d(
        &self,
        input: &Tensor<f32>,
        params: &Pool2dParams,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        let out_shape =
            pool2d_out_shape(input.shape(), params).map_err(BackendError::ShapeMismatch)?;
        if out_shape.contains(&0) {
            return Ok((
                Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch)?,
                Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch)?,
            ));
        }
        checked_bytes_for::<f32>(input.shape()).map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("max_pool2d: input not contiguous".into())
        })?;

        let [kh, kw] = params.kernel_size();
        let [sh, sw] = params.stride();
        let [ph, pw] = params.padding();
        let [dh, dw] = params.dilation();

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let pooling = context_cache::cached_pooling(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (values, indices) = pooling
            .run_max_pool2d_f32(
                &ctx,
                input_slice,
                input.shape(),
                (kh, kw),
                (sh, sw),
                (ph, pw),
                (dh, dw),
            )
            .map_err(map_pooling_error)?;
        Ok((
            Tensor::new(values, &out_shape).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(indices, &out_shape).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// `BackendOps::avg_pool2d` の Metal 実装（イシュー #1730・
    /// 追従イシュー）。[`pool2d_out_shape`] で `input.shape()`／
    /// `params` を再検査してから `pooling::MetalPooling::
    /// run_avg_pool2d_f32` へ委譲する（`.contiguous()` 前の
    /// `checked_bytes_for` は [`Self::max_pool2d`] と同じ理由）。
    fn avg_pool2d(
        &self,
        input: &Tensor<f32>,
        params: &Pool2dParams,
        count_include_pad: bool,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape =
            pool2d_out_shape(input.shape(), params).map_err(BackendError::ShapeMismatch)?;
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch);
        }
        checked_bytes_for::<f32>(input.shape()).map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("avg_pool2d: input not contiguous".into())
        })?;

        let [kh, kw] = params.kernel_size();
        let [sh, sw] = params.stride();
        let [ph, pw] = params.padding();
        let [dh, dw] = params.dilation();

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let pooling = context_cache::cached_pooling(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let values = pooling
            .run_avg_pool2d_f32(
                &ctx,
                input_slice,
                input.shape(),
                (kh, kw),
                (sh, sw),
                (ph, pw),
                (dh, dw),
                count_include_pad,
            )
            .map_err(map_pooling_error)?;
        Tensor::new(values, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// `BackendOps::adaptive_avg_pool2d` の Metal 実装（イシュー
    /// #1730・追従イシュー）。[`adaptive_pool2d_out_shape`] で
    /// `input.shape()`／`output_size` を再検査してから `pooling::
    /// MetalPooling::run_adaptive_avg_pool2d_f32` へ委譲する
    /// （`.contiguous()` 前の `checked_bytes_for` は [`Self::
    /// max_pool2d`] と同じ理由）。
    fn adaptive_avg_pool2d(
        &self,
        input: &Tensor<f32>,
        output_size: [usize; 2],
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = adaptive_pool2d_out_shape(input.shape(), output_size)
            .map_err(BackendError::ShapeMismatch)?;
        if out_shape.contains(&0) {
            return Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch);
        }
        checked_bytes_for::<f32>(input.shape()).map_err(BackendError::ShapeMismatch)?;

        let input_owned = input.contiguous();
        let input_slice = input_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("adaptive_avg_pool2d: input not contiguous".into())
        })?;
        let [oh, ow] = output_size;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let pooling = context_cache::cached_pooling(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let values = pooling
            .run_adaptive_avg_pool2d_f32(&ctx, input_slice, input.shape(), (oh, ow))
            .map_err(map_pooling_error)?;
        Tensor::new(values, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_pointwise`] の Metal
    /// 実装（イシュー #1647）。`hidden` は `c_prev` の列数から導出する。
    fn lstm_pointwise(
        &self,
        pre: &Tensor<f32>,
        c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        require_rank2_cell(c_prev.shape())?;
        let hidden = c_prev.shape()[1];
        let b_dim = c_prev.shape()[0];
        let gate_width = checked_gate_width(4, hidden)?;
        require_same_shape(pre.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        let pre_owned = pre.contiguous();
        let c_prev_owned = c_prev.contiguous();
        let pre_slice = pre_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_pointwise: pre not contiguous".into())
        })?;
        let c_prev_slice = c_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_pointwise: c_prev not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rnn = context_cache::cached_rnn_cell(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (gates, c, h) = rnn
            .run_lstm_pointwise_f32(&ctx, pre_slice, c_prev_slice, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok(LstmPointwiseOutput {
            gates: Tensor::new(gates, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            c: Tensor::new(c, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
            h: Tensor::new(h, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_hidden_backward`] の
    /// Metal 実装（イシュー #1647）。
    fn lstm_hidden_backward(
        &self,
        c: &Tensor<f32>,
        gate_o: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        require_rank2_cell(c.shape())?;
        require_same_shape(gate_o.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        let shape = c.shape().to_vec();
        let c_owned = c.contiguous();
        let gate_o_owned = gate_o.contiguous();
        let dh_owned = dh.contiguous();
        let c_slice = c_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_hidden_backward: c not contiguous".into())
        })?;
        let gate_o_slice = gate_o_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_hidden_backward: gate_o not contiguous".into())
        })?;
        let dh_slice = dh_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_hidden_backward: dh not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rnn = context_cache::cached_rnn_cell(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (d_pre_o, dc) = rnn
            .run_lstm_hidden_backward_f32(&ctx, c_slice, gate_o_slice, dh_slice)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok((
            Tensor::new(d_pre_o, &shape).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dc, &shape).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::lstm_cell_backward`] の
    /// Metal 実装（イシュー #1647）。`hidden` は `c_prev` の列数から
    /// 導出する。
    fn lstm_cell_backward(
        &self,
        gates_ifg: &Tensor<f32>,
        c_prev: &Tensor<f32>,
        dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        require_rank2_cell(c_prev.shape())?;
        let hidden = c_prev.shape()[1];
        let b_dim = c_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_ifg.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dc.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        let gates_owned = gates_ifg.contiguous();
        let c_prev_owned = c_prev.contiguous();
        let dc_owned = dc.contiguous();
        let gates_slice = gates_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_cell_backward: gates_ifg not contiguous".into())
        })?;
        let c_prev_slice = c_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_cell_backward: c_prev not contiguous".into())
        })?;
        let dc_slice = dc_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("lstm_cell_backward: dc not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rnn = context_cache::cached_rnn_cell(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (d_pre_ifg, dc_prev) = rnn
            .run_lstm_cell_backward_f32(&ctx, gates_slice, c_prev_slice, dc_slice, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok((
            Tensor::new(d_pre_ifg, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dc_prev, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gru_pointwise`] の Metal
    /// 実装（イシュー #1647）。`hidden` は `h_prev` の列数から導出する。
    fn gru_pointwise(
        &self,
        pre_i: &Tensor<f32>,
        pre_h: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        require_rank2_cell(h_prev.shape())?;
        let hidden = h_prev.shape()[1];
        let b_dim = h_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(pre_i.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(pre_h.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        let pre_i_owned = pre_i.contiguous();
        let pre_h_owned = pre_h.contiguous();
        let h_prev_owned = h_prev.contiguous();
        let pre_i_slice = pre_i_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_pointwise: pre_i not contiguous".into())
        })?;
        let pre_h_slice = pre_h_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_pointwise: pre_h not contiguous".into())
        })?;
        let h_prev_slice = h_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_pointwise: h_prev not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rnn = context_cache::cached_rnn_cell(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (gates, q, h) = rnn
            .run_gru_pointwise_f32(&ctx, pre_i_slice, pre_h_slice, h_prev_slice, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok(GruPointwiseOutput {
            gates: Tensor::new(gates, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            q: Tensor::new(q, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
            h: Tensor::new(h, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        })
    }

    /// [`fandhe_ai_tensor_core::BackendOps::gru_backward`] の Metal
    /// 実装（イシュー #1647）。`hidden` は `h_prev` の列数から導出する。
    fn gru_backward(
        &self,
        gates_rzn: &Tensor<f32>,
        q: &Tensor<f32>,
        h_prev: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        require_rank2_cell(h_prev.shape())?;
        let hidden = h_prev.shape()[1];
        let b_dim = h_prev.shape()[0];
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_rzn.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(q.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        let gates_owned = gates_rzn.contiguous();
        let q_owned = q.contiguous();
        let h_prev_owned = h_prev.contiguous();
        let dh_owned = dh.contiguous();
        let gates_slice = gates_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: gates_rzn not contiguous".into())
        })?;
        let q_slice = q_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: q not contiguous".into())
        })?;
        let h_prev_slice = h_prev_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: h_prev not contiguous".into())
        })?;
        let dh_slice = dh_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("gru_backward: dh not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rnn = context_cache::cached_rnn_cell(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let (d_pre_i, d_pre_h, dh_prev_direct) = rnn
            .run_gru_backward_f32(&ctx, gates_slice, q_slice, h_prev_slice, dh_slice, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Ok((
            Tensor::new(d_pre_i, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(d_pre_h, &[b_dim, gate_width]).map_err(BackendError::ShapeMismatch)?,
            Tensor::new(dh_prev_direct, &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?,
        ))
    }

    /// [`fandhe_ai_tensor_core::BackendOps::softmax`] の Metal 実装
    /// （イシュー #1594）。[`row_softmax_layout`] が非最終軸を `Ok(None)`
    /// として区別する契約に従い、その場合はデフォルトの
    /// `Unsupported`（`Var::softmax` がホスト参照実装へフォールバック
    /// する合図）と同じ挙動を返す。最終軸の場合は `run_fused_softmax`
    /// （下記。`run_fused` の softmax 一致経路）と同じ
    /// `context_cache::cached_softmax` キャッシュ・`MetalSoftmax::
    /// run_softmax_f32` を直接呼ぶ（融合プランを経由しない独立入口）。
    fn softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        let Some((rows, cols)) =
            row_softmax_layout(x.shape(), dim).map_err(BackendError::ShapeMismatch)?
        else {
            return Err(BackendError::Unsupported(
                "softmax: Metal 行カーネルは最終軸限定（非最終軸はホスト参照実装へ委ねる）".into(),
            ));
        };

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("softmax: input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let softmax = context_cache::cached_softmax(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = softmax
            .run_softmax_f32(&ctx, x_slice, rows, cols)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, x.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装
    /// （`Unsupported` fail-safe）を、canonical RMSNorm／softmax 融合プラン
    /// 検出時のみ融合カーネル（[`crate::rmsnorm::MetalRmsNorm`]／
    /// [`crate::softmax::MetalSoftmax`]）へルーティングする（イシュー #604）。
    /// CUDA 側 `CudaBackendOps::run_fused`
    /// （#592）の fail-closed 検証列をそのまま踏襲する:
    ///
    /// 1. `row_kernel::match_rmsnorm_plan`（6 op 列・leaf 1 個・
    ///    `axis: None` のみ受理）→ 一致時 [`crate::rmsnorm::MetalRmsNorm`] へ
    /// 2. `row_kernel::match_softmax_plan`（8 op 列・leaf 1 個・`axis` が
    ///    最終次元または `None` のみ受理）→ 一致時 [`crate::softmax::MetalSoftmax`] へ
    /// 3. どちらにも一致しないプランは `Unsupported` を返し per-op
    ///    フォールバックへ委ねる（allowlist 拒否・迂回経路を作らない。
    ///    `.claude/rules/security.md` A08）
    /// 4. 一致後も `plan.dtype() == DType::F32`・leaf 個数 = 1・
    ///    `leaf.shape() == plan.output_shape()` を明示検証する
    ///    （CUDA 側 codex-review 是正 PR #706 と同等の起動前検証）
    ///
    /// softmax の run_fused 配線は CUDA（G-7・#594）に先行するが、プラン
    /// 形状は `tensor-core` の融合 IR（#588）のテストで固定済みのため
    /// 乖離リスクは低い（`row_kernel.rs` モジュール冒頭コメント・`softmax.rs`
    /// ドキュメンテーションコメント「CUDA との parity 状況」参照）。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        if let Some(hidden) = row_kernel::match_rmsnorm_plan(plan) {
            return self.run_fused_rmsnorm(plan, leaves, hidden);
        }
        if let Some(hidden) = row_kernel::match_softmax_plan(plan) {
            return self.run_fused_softmax(plan, leaves, hidden);
        }
        Err(BackendError::Unsupported(
            "MetalBackendOps::run_fused: プランが canonical RMSNorm（x * rsqrt(sum(x^2))）／\
             softmax（exp(x-max(x))/sum）のいずれの形状にも一致しないため融合カーネルへ\
             ルーティングできない（#604 スコープ。呼び出し元の per-op フォールバックに委ねる）"
                .into(),
        ))
    }

    /// REQ-14 の明示解放 API（イシュー #1021）。`crate::pool::
    /// MetalAllocator::release_cached`（内部メソッド。設計文書 §3.1
    /// 「2 段構成の命名規約」・§3.6 (2) のバックエンド別フェーズ表
    /// 「Metal」列）へ委譲する。バイト数は
    /// `device_memory_pool_stats()`（`PoolStats::cached_bytes`）から
    /// 確認できるため戻り値からは捨てる（設計文書 §3.1 同節）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        context_cache::cached_allocator(&ctx)
            .map_err(map_metal_error)?
            .release_cached()
            .map(|_bytes| ())
            .map_err(map_metal_error)
    }

    /// デバイスメモリプールの統計スナップショット（イシュー #1021）。
    /// コンテキスト・アロケータ取得自体が失敗した場合（デバイス不在等）
    /// は `None`（プールを持たない扱い。`BackendOps::
    /// device_memory_pool_stats` の既定契約と同じ fail-safe）とし、
    /// panic・`Err` の握り潰しはしない（呼び出し元が診断目的でしか
    /// 使わない値のため、デバイス不在自体は他の演算メソッド呼び出しで
    /// 既に検出できる。ここで `Result` を返す必要はないという既存
    /// trait シグネチャ〈`Option<PoolStats>`〉の制約に従う）。
    fn device_memory_pool_stats(&self) -> Option<fandhe_ai_tensor_core::PoolStats> {
        let ctx = context_cache::cached_context().ok()?;
        let allocator = context_cache::cached_allocator(&ctx).ok()?;
        Some(allocator.stats())
    }
}

impl MetalBackendOps {
    /// `a`（rank >= 2 の `[B0, …, M, K]`）の先頭次元を行次元へ畳み、
    /// `b`（`[K, N]`）との GEMM を `[B0, …, M, N]` として計算する
    /// （イシュー #1040。バッチ matmul の公開 API 化は別イシュー——
    /// `BackendOps` trait は変更せず、本メソッドは `MetalBackendOps` の
    /// inherent メソッドとして追加する）。
    ///
    /// `crate::layout::collapse_leading_dims` が `Some` を返す場合
    /// （先頭次元が連続 view として畳める場合）は `a` を
    /// `contiguous()` せずそのまま `MatrixLayout` へ変換し、
    /// `gemm::MetalGemm::dispatch_strided_bias_act_prepared` へ渡す。
    /// `None`（collapse 不能な非連続 view）の場合は `a.contiguous()`
    /// 後に `[B0*…*M, K]` へ reshape してから同じ入口へ渡す
    /// （collapse 可否に関わらず数値結果は同一——`gemm.metal` の
    /// 添字計算は「連続な行優先バッファ」という前提のみに依存する）。
    ///
    /// `bias`・activation は扱わない（`gemm_bias_act` の trait 実装が
    /// 別途担う）。
    pub fn gemm_collapsed_lhs(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if a.rank() < 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a.rank(),
            }));
        }
        let b_shape = b.shape();
        if b_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: b_shape.len(),
            }));
        }
        let a_shape = a.shape().to_vec();
        let k = a_shape[a_shape.len() - 1];
        if b_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.clone(),
                rhs: b_shape.to_vec(),
            }));
        }
        let n = b_shape[1];
        let batch_dims = &a_shape[..a_shape.len() - 1];
        let out_shape: Vec<usize> = batch_dims.iter().copied().chain([n]).collect();
        let m: usize = batch_dims.iter().product();

        // イシュー #1040 是正（codex-review・Cursor Bugbot 指摘）: `m == 0`
        // （バッチ次元のいずれかが 0）または `n == 0` の場合、出力の要素数
        // は 0 であり GPU 起動は不要（`gemm_resident_lhs`／
        // `gemm_resident_rhs` と同じ「numel == 0 は空 Tensor」の判断。
        // Metal context 取得前に判定することで `MetalBufferHandle::
        // buffer == None`（空バッファ）に起因する `DeviceAllocationFailed`
        // を回避する）。`k == 0`（`m`・`n` は非 0）は GEMM の数学的定義
        // どおり結果が全 0（`gemm_resident_lhs` の同分岐と同じ判断）で、
        // こちらも GPU 起動を避けホスト側で直接構築する。
        if m == 0 || n == 0 {
            return Tensor::new(Vec::new(), &out_shape).map_err(BackendError::ShapeMismatch);
        }
        if k == 0 {
            return Tensor::from_shape_fill(&out_shape, |_| 0.0)
                .map_err(BackendError::ShapeMismatch);
        }

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let mem = MetalMemory::from_shared(ctx.clone());

        // `a` を collapse 可能なら zero-copy（`as_view_slice`）、
        // 不能なら `contiguous()` 後の実際の行優先形状で NN レイアウトを
        // 構築する（`upload_operand_for_resident_gemm` と同じ
        // 「分類できなければ contiguous へフォールバック」方針だが、
        // 本メソッドは rank >= 2 の先頭次元 collapse を扱うため
        // `collapse_leading_dims` を使う専用ロジックとする）。
        let (a_dev_buf, a_layout, m) = match layout::collapse_leading_dims(a.shape(), a.strides()) {
            Some(collapsed) => {
                let slice = a.as_view_slice().ok_or_else(|| {
                    BackendError::KernelLaunchFailed(
                        "gemm_collapsed_lhs: collapse_leading_dims succeeded but \
                         as_view_slice returned None (non-negative-stride invariant violated)"
                            .into(),
                    )
                })?;
                let dev_buf = mem.upload_view(slice, a.shape()).map_err(map_metal_error)?;
                let m = collapsed.rows;
                (dev_buf, collapsed, m)
            }
            None => {
                let a_owned = a.contiguous();
                let m: usize = batch_dims.iter().product();
                let a_reshaped = a_owned
                    .reshape(&[m, k])
                    .map_err(BackendError::ShapeMismatch)?;
                let dev_buf = mem.upload(&a_reshaped)?;
                let layout = MatrixLayout {
                    rows: m,
                    cols: k,
                    ld: k,
                    transposed: false,
                };
                (dev_buf, layout, m)
            }
        };
        let a_handle = a_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(a_buf) = a_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_collapsed_lhs: a buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let (b_dev_buf, b_layout) = upload_operand_for_resident_gemm(&mem, b)?;
        let b_handle = b_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(b_buf) = b_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_collapsed_lhs: b buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let c_dev_buf = mem.alloc_zeroed(&[m, n])?;
        let c_handle = c_dev_buf
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let Some(c_buf) = c_handle.buffer.as_ref() else {
            return Err(BackendError::DeviceAllocationFailed(
                "gemm_collapsed_lhs: output buffer has numel > 0 but no device allocation".into(),
            ));
        };

        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        gemm.dispatch_strided_bias_act_prepared(
            &ctx, a_buf, 0, a_layout, b_buf, 0, b_layout, None, false, c_buf, m, n, k,
        )
        .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;

        let c_tensor = mem.download(&c_dev_buf)?;
        c_tensor
            .reshape(&out_shape)
            .map_err(BackendError::ShapeMismatch)
    }

    /// 一致した leaf・shape・dtype を検証してから
    /// [`crate::rmsnorm::MetalRmsNorm::run_rmsnorm_f32_raw`] を `inv_n = 1.0`・`eps = 0.0`・
    /// `w = None`（`has_weight = 0`）で直接呼ぶ（プランの意味論 `x *
    /// rsqrt(sum(x^2))` に厳密一致させる。`mean` 化・`eps` 加算・`weight`
    /// 乗算を勝手に補わない。`backend-cuda::ops::CudaBackendOps::run_fused`
    /// と同じ検証順序: dtype → leaf 数 → leaf shape の順にカーネル起動前
    /// （デバイスアクセス前）で fail-closed に検証する）。
    fn run_fused_rmsnorm(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        let x = validate_fused_leaf(plan, leaves)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: rmsnorm input not contiguous".into())
        })?;

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let rmsnorm = context_cache::cached_rmsnorm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = rmsnorm
            .run_rmsnorm_f32_raw(&ctx, x_slice, None, 0.0, 1.0, 1, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`Self::run_fused_rmsnorm`] と同じ検証順序で
    /// [`crate::softmax::MetalSoftmax::run_softmax_f32`] を呼ぶ。
    ///
    /// `row_kernel::match_softmax_plan` は `axis: None`（全軸縮約。
    /// `rows = 1` 相当）だけでなく `axis` が最終次元（行方向縮約。
    /// `rows > 1` になりうる）も受理するため、[`Self::run_fused_rmsnorm`]
    /// と異なり `rows` を固定値にできない。`x_slice.len() / hidden`
    /// （`validate_fused_leaf` が `x.shape() == plan.output_shape()` を
    /// 検証済みのため `x_slice.len()` は `plan.output_shape()` の要素数積
    /// と一致する）から導出する。`hidden == 0` は
    /// `crate::softmax::MetalSoftmax::run_softmax_f32` 側の 0 要素早期 return 契約に委ね、
    /// ここでは `checked_div`（`hidden == 0` なら `rows = 0`）でゼロ除算
    /// のみを避ける。
    fn run_fused_softmax(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        let x = validate_fused_leaf(plan, leaves)?;

        let x_owned = x.contiguous();
        let x_slice = x_owned.as_slice().ok_or_else(|| {
            BackendError::KernelLaunchFailed("run_fused: softmax input not contiguous".into())
        })?;
        // `checked_div` を使い `hidden == 0` を除算前に排除する（clippy
        // `manual_checked_ops`。挙動は従来の if 分岐と同一: `hidden == 0`
        // の場合は `rows = 0`、それ以外は通常の整数除算）。
        let rows = x_slice.len().checked_div(hidden).unwrap_or(0);

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let softmax = context_cache::cached_softmax(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = softmax
            .run_softmax_f32(&ctx, x_slice, rows, hidden)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// [`MetalBackendOps::run_fused_rmsnorm`]／[`run_fused_softmax`] 共通の
/// カーネル起動前検証（デバイスアクセス前・fail-closed）。
///
/// `match_rmsnorm_plan`／`match_softmax_plan` は op 列・leaf 数・
/// `row_fusion()` の形状のみを照合し、`FusionPlan::from_ops` が受理しうる
/// 任意の `dtype` を検査しない。カーネル起動前に `plan.dtype() ==
/// DType::F32` を明示検証しないと、例えば `DType::F64` のプランでも f32
/// Metal カーネルとして実行されてしまう（CUDA 側 codex-review 指摘・
/// PR #706 レビューと同種の懸念）。leaf の shape が `plan.output_shape()`
/// と一致することも検証する: canonical プランは leaf 1 個・恒等 shape
/// （`axis: None`〈全軸縮約〉または行方向縮約後に broadcast で復元）の
/// 契約のため、要素数が一致しつつ shape（次元分割）が異なる leaf を
/// 拒否する（`backend-cpu::fused_elementwise` の leaf shape 検証と同じ
/// 契約）。
///
/// 検証済みの唯一の leaf（`&Tensor<f32>`）を戻り値として返す（呼び出し元
/// が `leaves` を再度パターンマッチする必要をなくし、`unreachable!` を
/// 使わずに済ませる。coding-rust.md「本番経路で unwrap/expect を使わない」
/// と同じ理由で、到達しないはずの分岐を panic で表現しない）。
fn validate_fused_leaf<'a>(
    plan: &FusionPlan,
    leaves: &[&'a Tensor<f32>],
) -> Result<&'a Tensor<f32>, BackendError> {
    if !plan_dtype_is_f32(plan) {
        return Err(BackendError::Unsupported(format!(
            "MetalBackendOps::run_fused: unsupported dtype {:?} (canonical fusion kernels \
             support F32 only)",
            plan.dtype()
        )));
    }
    let [x] = leaves else {
        return Err(BackendError::Unsupported(format!(
            "MetalBackendOps::run_fused: canonical プランは leaf 1 個を要求するが {} 個が渡された",
            leaves.len()
        )));
    };
    if x.shape() != plan.output_shape() {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: plan.output_shape().to_vec(),
            rhs: x.shape().to_vec(),
        }));
    }
    Ok(*x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- gemm_bias_act_route（pure・実機不要。イシュー #605） ---
    // CUDA 側 `fandhe_ai_backend_cuda::ops` の同名テスト群と同一の検証項目。

    #[test]
    fn gemm_bias_act_route_selects_fused_when_bias_is_none() {
        assert_eq!(gemm_bias_act_route(None, 8), GemmBiasActRoute::Fused);
    }

    #[test]
    fn gemm_bias_act_route_selects_fused_when_bias_shape_matches_n_exactly() {
        assert_eq!(gemm_bias_act_route(Some(&[8]), 8), GemmBiasActRoute::Fused);
    }

    #[test]
    fn gemm_bias_act_route_falls_back_when_bias_shape_is_broadcastable_but_not_n() {
        assert_eq!(
            gemm_bias_act_route(Some(&[1]), 8),
            GemmBiasActRoute::ComposedFallback
        );
        assert_eq!(
            gemm_bias_act_route(Some(&[1, 8]), 8),
            GemmBiasActRoute::ComposedFallback
        );
    }

    #[test]
    fn gemm_bias_act_route_falls_back_when_bias_len_mismatches_n() {
        assert_eq!(
            gemm_bias_act_route(Some(&[4]), 8),
            GemmBiasActRoute::ComposedFallback
        );
    }

    // --- gather／scatter の rank 上限フォールバック（イシュー #1778・
    // レビュー指摘の是正。イシュー #1799）。ops.rs 内の rank 検査は
    // Metal コンテキスト取得（`context_cache::cached_context()`）より
    // 前で早期 return するため、実機（Apple Silicon）非依存で検証
    // できる（`#[ignore]` 不要）。 ---

    #[test]
    fn gather_returns_unsupported_for_rank_exceeding_kernel_limit() {
        // `GS_MAX_RANK + 1` 階のテンソル（例: shape=[1; 9]）は
        // `gather_out_shape` 自体は受理する有効な shape だが、カーネルの
        // `thread ulong coords[GS_MAX_RANK]` 上限を超える。`ShapeMismatch`
        // ではなく `Unsupported` を返し、呼び出し元（`autodiff::grad::
        // gather_with_fallback`）がホスト参照実装へフォールバックできる
        // ことを確認する（codex-review 指摘）。
        let rank = GS_MAX_RANK + 1;
        let shape = vec![1usize; rank];
        let input = Tensor::new(vec![1.0f32], &shape).unwrap();
        let index = Tensor::<i32>::new(vec![0i32], &shape).unwrap();

        let ops = MetalBackendOps::new();
        let err = ops.gather(&input, 0, &index).unwrap_err();
        assert!(
            matches!(err, BackendError::Unsupported(_)),
            "rank={rank}（GS_MAX_RANK={GS_MAX_RANK} 超過）は Unsupported を返すべき: {err:?}"
        );
    }

    #[test]
    fn scatter_returns_unsupported_for_rank_exceeding_kernel_limit() {
        let rank = GS_MAX_RANK + 1;
        let shape = vec![1usize; rank];
        let input = Tensor::new(vec![1.0f32], &shape).unwrap();
        let index = Tensor::<i32>::new(vec![0i32], &shape).unwrap();
        let src = Tensor::new(vec![1.0f32], &shape).unwrap();

        let ops = MetalBackendOps::new();
        let err = ops
            .scatter(&input, 0, &index, &src, ScatterReduce::Overwrite)
            .unwrap_err();
        assert!(
            matches!(err, BackendError::Unsupported(_)),
            "rank={rank}（GS_MAX_RANK={GS_MAX_RANK} 超過）は Unsupported を返すべき: {err:?}"
        );
    }

    // --- im2col／col2im の N=0 早期リターン（イシュー #1769）。
    // `Self::im2col`／`Self::col2im` は出力（`im2col`）／`input_shape`
    // （`col2im`）が空（`0` を含む）のとき `context_cache::
    // cached_context()` より前で早期 return するため、`gather`／
    // `scatter` の rank 上限テストと同じ理由で実機非依存に検証できる
    // （`#[ignore]` 不要）。`backend-cuda::ops::tests::
    // im2col_returns_empty_for_zero_batch_1d_without_touching_device`
    // と同型。1d 形状（`H=1`・`kh=1`）で確認する。 ---

    #[test]
    fn im2col_returns_empty_for_zero_batch_1d_without_touching_device() {
        let params = Conv2dParams::new([1, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let input = Tensor::new(Vec::<f32>::new(), &[0usize, 1, 1, 4]).unwrap();

        let ops = MetalBackendOps::new();
        let out = ops.im2col(&input, &params).unwrap();
        assert_eq!(out.shape(), &[0, 1, 2, 3]);
        assert_eq!(out.as_slice().unwrap().len(), 0);
    }

    #[test]
    fn col2im_returns_empty_for_zero_batch_1d_without_touching_device() {
        let params = Conv2dParams::new([1, 2], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let input_shape = [0usize, 1, 1, 4];
        let col_shape = im2col_out_shape(&input_shape, &params).unwrap();
        let d_col = Tensor::new(Vec::<f32>::new(), &col_shape).unwrap();

        let ops = MetalBackendOps::new();
        let out = ops.col2im(&d_col, &input_shape, &params).unwrap();
        assert_eq!(out.shape(), &input_shape);
        assert_eq!(out.as_slice().unwrap().len(), 0);
    }

    // --- pooling（max_pool2d／avg_pool2d／adaptive_avg_pool2d）の
    // `BackendOps` 配線確認（イシュー #1730・追従イシュー。#1607
    // ツリー）。親 #1730 が本体カーネル（`crate::pooling::
    // MetalPooling`）を実装済みだったが `ops.rs` への override 配線が
    // 漏れており本番経路が既定の `Unsupported`（ホストフォールバック）
    // に落ちてカーネルがデッドコードになっていた事実を回帰させる
    // （`backend-cuda::ops::tests` の同型テストと対称）。`input.shape()`
    // の batch 軸が 0 の早期リターンは `col2im` と同じ理由で実機非依存
    // に検証できる（`#[ignore]` 不要。ただしこのファイル自体が
    // `cfg(target_os = "macos")` 限定のため macOS 上でのみ実行される）。
    // 実機〈Apple Silicon〉でのカーネル実行自体（数値一致）は未実測の
    // まま申し送る。 ---

    #[test]
    fn max_pool2d_returns_empty_for_zero_batch_without_touching_device() {
        let x = Tensor::<f32>::new(Vec::new(), &[0, 1, 4, 4]).unwrap();
        let params = Pool2dParams::new([2, 2], None, [0, 0], [1, 1]).unwrap();
        let ops = MetalBackendOps::new();
        let (values, indices) = ops
            .max_pool2d(&x, &params)
            .expect("empty max_pool2d must succeed");
        assert_eq!(values.shape(), &[0, 1, 2, 2]);
        assert_eq!(indices.shape(), &[0, 1, 2, 2]);
        assert_eq!(values.as_slice().unwrap().len(), 0);
    }

    #[test]
    fn avg_pool2d_returns_empty_for_zero_batch_without_touching_device() {
        let x = Tensor::<f32>::new(Vec::new(), &[0, 1, 4, 4]).unwrap();
        let params = Pool2dParams::new([2, 2], None, [0, 0], [1, 1]).unwrap();
        let ops = MetalBackendOps::new();
        let out = ops
            .avg_pool2d(&x, &params, true)
            .expect("empty avg_pool2d must succeed");
        assert_eq!(out.shape(), &[0, 1, 2, 2]);
        assert_eq!(out.as_slice().unwrap().len(), 0);
    }

    #[test]
    fn adaptive_avg_pool2d_returns_empty_for_zero_batch_without_touching_device() {
        let x = Tensor::<f32>::new(Vec::new(), &[0, 1, 4, 4]).unwrap();
        let ops = MetalBackendOps::new();
        let out = ops
            .adaptive_avg_pool2d(&x, [2, 2])
            .expect("empty adaptive_avg_pool2d must succeed");
        assert_eq!(out.shape(), &[0, 1, 2, 2]);
        assert_eq!(out.as_slice().unwrap().len(), 0);
    }

    // --- pooling（max_pool2d／avg_pool2d／adaptive_avg_pool2d）の
    // `.contiguous()` 呼び出し前の確保前検査（Cursor Bugbot 指摘・
    // PR #1888）。`interpolate_rejects_huge_broadcast_view_input_
    // without_panicking` と同型だが、pooling は出力 N／C が入力 N／C
    // の素通しであるため、N 軸を巨大化すると出力側の
    // `checked_numel_for`（`pool2d_out_shape`／
    // `adaptive_pool2d_out_shape` 内）が先に overflow を検出して
    // しまい `.contiguous()` 直前の新規ガード自体は不到達になる。
    // そこで **H 軸**（N／C=1 のまま）を `broadcast_to` で巨大化し、
    // max／avg は `kernel=stride=H_in` で `hout=1` へ縮約する（adaptive
    // は `output_size` が入力 H に依存しないためそのまま `[2, 2]`
    // で足りる）ことで出力 shape を小さく保ったまま入力側だけを
    // 巨大にし、`checked_bytes_for::<f32>(input.shape())` が
    // `.contiguous()`（内部で無検査の `Vec::with_capacity(numel)` を
    // 呼ぶ）より前に拒否することを固定する。Metal コンテキスト取得前
    // に完了するため実機非依存の通常テストとして Linux CI でも実行
    // できる（このファイル自体が `cfg(target_os = "macos")` 限定の
    // ため実際に実行されるのは macOS 上のみ）。 ---

    // fixture は `1usize << 59` を使う（イシュー #1897 是正）。
    // 当初 `1usize << 62` だった shape `[1, 1, H, 4]` は
    // `numel = H * 4 = 2usize.pow(64)` となり `Tensor::broadcast_to`
    // 自身の `checked_numel`（`usize` 積のオーバーフロー検査のみ・
    // バイトサイズは見ない）が `ElementCountOverflow` を返すため、
    // fixture 構築の `.unwrap()` 自体が panic していた（本検査対象の
    // `checked_bytes_for::<f32>` には到達しない）。`H = 1usize << 59`
    // は `numel = 2usize.pow(61)` で `usize` 積として overflow しない
    // ため `broadcast_to` は成功し、`checked_bytes_for::<f32>` の
    // `bytes = numel * size_of::<f32>() = 2usize.pow(63)` が
    // `isize::MAX`（`2usize.pow(63) - 1`）をちょうど 1 超えることで
    // 検査対象の分岐（バイトサイズ超過）を確実に通す。
    #[test]
    fn max_pool2d_rejects_huge_broadcast_view_input_without_panicking() {
        let base = Tensor::<f32>::new(vec![0.0f32; 4], &[1usize, 1, 1, 4]).unwrap();
        let huge = base.broadcast_to(&[1, 1, 1usize << 59, 4]).unwrap();
        let params = Pool2dParams::new([1usize << 59, 1], None, [0, 0], [1, 1]).unwrap();
        let ops = MetalBackendOps::new();
        let err = ops
            .max_pool2d(&huge, &params)
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn avg_pool2d_rejects_huge_broadcast_view_input_without_panicking() {
        let base = Tensor::<f32>::new(vec![0.0f32; 4], &[1usize, 1, 1, 4]).unwrap();
        let huge = base.broadcast_to(&[1, 1, 1usize << 59, 4]).unwrap();
        let params = Pool2dParams::new([1usize << 59, 1], None, [0, 0], [1, 1]).unwrap();
        let ops = MetalBackendOps::new();
        let err = ops
            .avg_pool2d(&huge, &params, true)
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn adaptive_avg_pool2d_rejects_huge_broadcast_view_input_without_panicking() {
        let base = Tensor::<f32>::new(vec![0.0f32; 4], &[1usize, 1, 1, 4]).unwrap();
        let huge = base.broadcast_to(&[1, 1, 1usize << 59, 4]).unwrap();
        let ops = MetalBackendOps::new();
        let err = ops
            .adaptive_avg_pool2d(&huge, [2, 2])
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    /// `pad` は `gather`／`scatter` と異なりカーネル側に rank 上限が
    /// 存在しない（`shaders/constant_pad.metal` モジュール doc 参照）
    /// ため、`GS_MAX_RANK`（8）を超える rank でも
    /// `BackendError::ShapeMismatch` を誤って返さないことを確認する
    /// （Cursor Bugbot 指摘の是正確認）。この shape 要素数積は
    /// `u32::MAX` 未満に収まるため、`Metal コンテキスト取得
    /// （`context_cache::cached_context()`）へ到達する前で誤検出される
    /// ことはないが、後続の GPU 起動自体は実機必須のためここでは検査
    /// しない。ここでは代わりに `u32::MAX` を超える shape 要素で早期
    /// return する回帰（`context_cache` へ到達する前に fail-closed
    /// で拒否される）を実機不要で確認する。
    #[test]
    fn pad_rejects_dim_exceeding_u32_before_touching_metal_context() {
        // rank 9 以上（`GS_MAX_RANK` 超過）でも受理されること自体は GPU
        // 起動が必要なため実機不要のここでは検証できない。代わりに
        // shape 要素が `u32::MAX` を超える場合に `ElementCountOverflow`
        // へ fail-closed で拒否され、`context_cache::cached_context()`
        // （実機必須）へ到達しないことを確認する（`validate_shapes_
        // fit_u32` 除去後も u32 収容検査自体は維持されていることの
        // 回帰）。他方の軸を 0 にして要素数積を 0 に保つことで、巨大な
        // `Vec<f32>` を確保せずに `Tensor::new` を成立させる（`unique_
        // rejects_transposed_shape_with_overflowing_intermediate_
        // product` と同じ発想）。
        let rank = GS_MAX_RANK + 1;
        let mut shape = vec![0usize; rank];
        shape[0] = (u32::MAX as usize) + 1;
        let input = Tensor::new(Vec::<f32>::new(), &shape).unwrap();
        let pads = vec![(0usize, 0usize); rank];

        let ops = MetalBackendOps::new();
        let err = ops.pad(&input, &pads, 0.0).unwrap_err();
        assert!(
            matches!(
                err,
                BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
            ),
            "rank={rank}・shape[0] が u32::MAX 超過は ElementCountOverflow を返すべき: {err:?}"
        );
    }

    /// [`MetalBackendOps::unique`] の回帰テスト（PR #1828 codex-review
    /// P1 是正確認・イシュー #1734）。`gather_returns_unsupported_for_
    /// rank_exceeding_kernel_limit` 系と同じ「実機に触れる前に早期
    /// return する」性質を利用し、`[0, 2, usize::MAX]` を
    /// `transpose(0, 2)` して得た `[usize::MAX, 2, 0]`（中間積
    /// `usize::MAX * 2` が `usize` の範囲を超えるが最終的な要素数は 0）
    /// を渡すと、`x.numel()`（内部で無検査の `.iter().product()` を
    /// 使う）を呼ぶ前に `checked_numel` で拒否し `overflow-checks`
    /// 有効ビルドでも panic せず型付きエラーを返すことを確認する
    /// （実機不要・CI Linux でも実行できる）。
    #[test]
    fn unique_rejects_transposed_shape_with_overflowing_intermediate_product() {
        let input_base = Tensor::<f32>::new(Vec::new(), &[0usize, 2usize, usize::MAX]).unwrap();
        let input = input_base.transpose(0, 2).unwrap();
        assert_eq!(input.shape(), &[usize::MAX, 2, 0]);

        let ops = MetalBackendOps::new();
        let err = ops.unique(&input).expect_err("overflow must be rejected");
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    /// Cursor Bugbot 指摘（イシュー #1834・PR レビュー）の回帰テスト。
    /// `[1]` を `broadcast_to([1usize << 63])` した巨大な非 contiguous
    /// view を小さい `size=[3]` へ縮小する `interpolate` は、
    /// `interpolate_out_shape` が検査する**出力**側のバイトサイズは
    /// 小さいため通過するが、`.contiguous()`（内部で無検査の
    /// `Vec::with_capacity(numel)` を呼ぶ）が**入力**側の巨大な要素数
    /// （`f32` 換算で `isize::MAX` バイトを超える）で capacity
    /// overflow panic しうる（是正前）。`checked_bytes_for::<f32>` による
    /// `.contiguous()` 呼び出し前の確保前検査は Metal コンテキスト
    /// 取得前に完了するため、実機非依存の通常テストとして Linux CI
    /// でも実行できる。
    #[test]
    fn interpolate_rejects_huge_broadcast_view_input_without_panicking() {
        let base = Tensor::<f32>::new(vec![0.0f32], &[1usize]).unwrap();
        let huge = base.broadcast_to(&[1usize << 63]).unwrap();
        assert_eq!(huge.shape(), &[1usize << 63]);

        let ops = MetalBackendOps::new();
        let err = ops
            .interpolate(&huge, &[3], InterpolateMode::Nearest)
            .expect_err("huge broadcast view の実体化は確保前に拒否されるはず");
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    /// [`map_interpolate_error`] の回帰テスト（PR #1834 codex-review・
    /// Cursor Bugbot 指摘の是正確認）。`MetalError::
    /// InvalidInterpolateShape`（`interpolate.rs::MetalInterpolate::
    /// run_nearest_f32` が内部の独立検査
    /// `interpolate_model::validate_interpolate_launch` の失敗時に
    /// 返す）が `BackendError::ShapeMismatch(ElementCountOverflow)` へ
    /// 変換されることを固定する。是正前はこの分岐が無く一律
    /// `BackendError::KernelLaunchFailed`（`interpolate.rs::
    /// MetalInterpolate::run_nearest_f32` 呼び出し箇所の汎用
    /// `.map_err(|e: MetalError| BackendError::KernelLaunchFailed(...))`
    /// 経由）になり、`in_shape=[1]`／`size=[u32::MAX as usize + 1]`
    /// のように共通形状検証（`interpolate_out_shape`）は通過するが
    /// `u32` 収容検査で拒否される入力（指摘の再現形状）が
    /// `ShapeMismatch` 判定を素通りしていた（`CudaBackendOps::
    /// interpolate` の `map_interpolate_error` とは非対称だった）。
    /// GPU 非依存: `MetalError` を直接構築して検証する（実機不要）。
    #[test]
    fn map_interpolate_error_treats_invalid_interpolate_shape_as_shape_mismatch() {
        let err = map_interpolate_error(MetalError::InvalidInterpolateShape {
            detail: "validate_interpolate_launch: dim exceeds u32::MAX".into(),
        });
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    /// [`map_interpolate_error`]: `InvalidInterpolateShape` 以外の
    /// `MetalError`（デバイス・パイプライン起動失敗等）は従来どおり
    /// `BackendError::KernelLaunchFailed` へ変換されることを固定する
    /// （上記テストと対で「形状エラーのみを区別する」契約を検証する）。
    #[test]
    fn map_interpolate_error_falls_back_to_kernel_launch_failed_for_other_errors() {
        let err = map_interpolate_error(MetalError::LibraryCompilation {
            message: "boom".into(),
        });
        assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
    }

    // --- gemm_resident_lhs／gemm_resident_rhs zero-repack 経路（イシュー
    // #1040。Metal 実機依存。`tests/gemm_bias_act_parity.rs` と同じ
    // 「pub(crate) カウンタはクレート内テストで検証」方針） ---

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn gemm_resident_lhs_transposed_b_does_not_increment_repack_counter() {
        let ops = MetalBackendOps::new();
        let mem = ops
            .memory_ops()
            .expect("MetalBackendOps must implement MemoryOps");
        let (p, q, r) = (4usize, 8usize, 5usize);
        let w = Tensor::new((0..p * q).map(|i| i as f32 * 0.1).collect(), &[p, q]).unwrap();
        let w_dev = mem.upload(&w).expect("w upload must succeed");
        let w_shape = [p, q];
        let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();

        // 転置 view（`Op::LinearResident` の VJP が渡す実際の形と同じ:
        // 元 [r, q] 行優先データを `transpose(0, 1)` して [q, r] として
        // 読む）。
        let b_rq = Tensor::new((0..r * q).map(|i| i as f32 * 0.01).collect(), &[r, q]).unwrap();
        let b_t = b_rq.transpose(0, 1).unwrap();
        assert!(
            b_t.as_slice().is_none(),
            "precondition: b_t must be non-contiguous"
        );

        let before = RESIDENT_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = ops
            .gemm_resident_lhs(w_view, &b_t)
            .expect("gemm_resident_lhs must succeed on Metal-equipped test runner");
        let after = RESIDENT_HOST_REPACK_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "transposed view input must not fall back to contiguous() host repack"
        );
    }

    // --- gemm の NT/TN strided 入口（イシュー #1215。Metal 実機依存。
    // `GEMM_HOST_REPACK_COUNT` は `pub(crate)` のため
    // クレート内テストでのみ検証する。上記 `gemm_resident_lhs` 系
    // テストと同じ方針） ---

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn gemm_nt_transposed_b_does_not_increment_repack_counter() {
        let ops = MetalBackendOps::new();
        let (m, k, n) = (4usize, 8usize, 5usize);
        let a = Tensor::new((0..m * k).map(|i| i as f32 * 0.1).collect(), &[m, k]).unwrap();
        // NT: `b` が転置 view（元 [n, k] 行優先データを転置して [k, n] として
        // 読む。`Op::LinearResident` の VJP が `matmul_vjp` の d_input =
        // `g @ Wᵀ` として渡す形と同型）。
        let b_nk = Tensor::new((0..n * k).map(|i| i as f32 * 0.01).collect(), &[n, k]).unwrap();
        let b_t = b_nk.transpose(0, 1).unwrap();
        assert!(
            b_t.as_slice().is_none(),
            "precondition: b_t must be non-contiguous"
        );

        let before = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = ops
            .gemm(&a, &b_t)
            .expect("gemm(NT) must succeed on Metal-equipped test runner");
        let after = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "NT (b transposed) must not fall back to contiguous() host repack"
        );
    }

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn gemm_tn_transposed_a_does_not_increment_repack_counter() {
        let ops = MetalBackendOps::new();
        let (m, k, n) = (4usize, 8usize, 5usize);
        // TN: `a` が転置 view（元 [k, m] 行優先データを転置して [m, k] として
        // 読む。`Op::LinearResident.d_weight` = `xᵀ @ g` が渡す形と同型）。
        let a_km = Tensor::new((0..k * m).map(|i| i as f32 * 0.1).collect(), &[k, m]).unwrap();
        let a_t = a_km.transpose(0, 1).unwrap();
        assert!(
            a_t.as_slice().is_none(),
            "precondition: a_t must be non-contiguous"
        );
        let b = Tensor::new((0..k * n).map(|i| i as f32 * 0.01).collect(), &[k, n]).unwrap();

        let before = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = ops
            .gemm(&a_t, &b)
            .expect("gemm(TN) must succeed on Metal-equipped test runner");
        let after = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        assert_eq!(
            before, after,
            "TN (a transposed) must not fall back to contiguous() host repack"
        );
    }

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn gemm_tt_both_transposed_increments_repack_counter_twice() {
        let ops = MetalBackendOps::new();
        let (m, k, n) = (4usize, 8usize, 5usize);
        // TT（両方転置）は NT/TN 分岐の対象外（`la.transposed != lb.transposed`
        // が成り立たない）ため、従来経路（`contiguous()` 2 回）へ落ちる
        // ことをカウンタで可観測にする。
        let a_km = Tensor::new((0..k * m).map(|i| i as f32 * 0.1).collect(), &[k, m]).unwrap();
        let a_t = a_km.transpose(0, 1).unwrap();
        let b_nk = Tensor::new((0..n * k).map(|i| i as f32 * 0.01).collect(), &[n, k]).unwrap();
        let b_t = b_nk.transpose(0, 1).unwrap();
        assert!(a_t.as_slice().is_none() && b_t.as_slice().is_none());

        let before = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = ops
            .gemm(&a_t, &b_t)
            .expect("gemm(TT) must succeed on Metal-equipped test runner");
        let after = GEMM_HOST_REPACK_COUNT.with(|c| c.get());
        assert_eq!(
            after - before,
            2,
            "TT (both transposed) must fall back to contiguous() host repack for both operands"
        );
    }
}
