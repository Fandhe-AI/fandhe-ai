//! RNN／LSTM／GRU セル演算（イシュー #1647）の起動 API（CUDA 側
//! `backend-cuda::rnn_cell`〈同イシュー〉の Metal 対応版）。
//!
//! [`MetalRnnCell::new`] が `shaders/rnn_cell.metal` を実行時コンパイル
//! して 5 パイプラインを保持し、`run_*_f32` へホスト側スライスを渡す
//! だけでバッファ確保・ディスパッチ・readback を内部で完結できる
//! （`crate::elementwise::MetalElementwise`・`crate::mse::MetalMse` と
//! 同じ構成方針）。`ops.rs::MetalBackendOps::{lstm_pointwise,
//! lstm_hidden_backward, lstm_cell_backward, gru_pointwise, gru_backward}`
//! から `BackendOps` の実装として呼ばれる。
//!
//! RNN（tanh 版）は専用カーネルを持たない（`fandhe_ai_autodiff::var::
//! rnn_cell_forward_value` が既存の `gemm_bias_act`／`add`／`tanh` の
//! 合成で閉じるため。設計 `docs/autodiff-rnn-cell-tape-design.md`
//! 決定 1）。

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLSize};

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{self, MtlPipeline};

/// 3 つの `Vec<f32>` を返す関数の戻り値型（`clippy::type_complexity`
/// 回避）。`run_lstm_pointwise_f32`／`run_gru_pointwise_f32`／
/// `run_gru_backward_f32` が共有する。
type TripleVecOutput = (Vec<f32>, Vec<f32>, Vec<f32>);

/// `shaders/rnn_cell.metal` のソース（5 カーネルを含む）。
const RNN_CELL_MSL_SRC: &str = include_str!("shaders/rnn_cell.metal");

/// 1 threadgroup あたりのスレッド数（1 次元）。
/// `elementwise.rs::EW_THREADGROUP_WIDTH`・`mse.rs::MSE_THREADGROUP_WIDTH`
/// と同じ値・同じ理由。
const RNN_THREADGROUP_WIDTH: usize = 256;

/// `hidden > 0` と `numel = B * hidden`（`numel % hidden == 0`）を検証
/// し `u32::MAX` 上限も検査する（`elementwise.rs::
/// validate_elementwise_len` と同じ理由）。
fn validate_rnn_dims(numel: usize, hidden: usize) -> Result<(), MetalError> {
    if hidden == 0 {
        return Err(MetalError::InvalidElementwiseShape {
            detail: "rnn cell hidden must be > 0".to_string(),
        });
    }
    if !numel.is_multiple_of(hidden) {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "rnn cell numel must be a multiple of hidden: numel={numel}, hidden={hidden}"
            ),
        });
    }
    if numel > u32::MAX as usize || hidden > u32::MAX as usize {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "rnn cell numel/hidden must fit in u32 (kernel argument type): numel={numel}, \
                 hidden={hidden}"
            ),
        });
    }
    Ok(())
}

/// 2 スライスの長さが一致することを検証する。
fn require_len(actual: usize, expected: usize, what: &str) -> Result<(), MetalError> {
    if actual != expected {
        return Err(MetalError::InvalidElementwiseShape {
            detail: format!(
                "rnn cell {what} length mismatch: expected={expected}, actual={actual}"
            ),
        });
    }
    Ok(())
}

fn dispatch_sizes(numel: usize) -> (MTLSize, MTLSize) {
    let threads_per_tg = MTLSize {
        width: RNN_THREADGROUP_WIDTH,
        height: 1,
        depth: 1,
    };
    let groups = numel.div_ceil(RNN_THREADGROUP_WIDTH);
    let threadgroups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    (threadgroups, threads_per_tg)
}

/// `u32` 値 1 個を `index` バッファへ即時複製する共通ヘルパー
/// （`mse.rs::encode_partial_dispatch` と同じ `setBytes_length_atIndex`
/// パターン）。
///
/// SAFETY: `setBytes_length_atIndex` は指定ポインタから指定バイト数を
/// 即座に複製する。`value` はローカル変数でありポインタは本呼び出し中
/// 生存し、長さは MSL 側 `constant uint&` 宣言の型と揃えている
/// （`shaders/rnn_cell.metal` 参照）。
fn set_u32_arg(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: u32, index: usize) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from(&value).cast(),
            std::mem::size_of::<u32>(),
            index,
        );
    }
}

/// `MetalBuffer` を `index` バッファスロットへ結線する共通ヘルパー。
///
/// SAFETY: `setBuffer_offset_atIndex` は生存中の `MTLBuffer` への参照を
/// 保持するのみで即座に読み書きしない（`mse.rs::encode_partial_dispatch`
/// と同種のコメント参照）。各バッファは呼び出し元 `ctx.dispatch_sync` が
/// 完了するまで生存する。
fn set_buffer_arg(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &MetalBuffer,
    index: usize,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf.raw()), 0, index);
    }
}

/// RNN／LSTM／GRU セル演算 5 カーネル（いずれも f32）のコンパイル済み
/// パイプラインを保持する。
pub struct MetalRnnCell {
    lstm_pointwise_f32: objc2::rc::Retained<MtlPipeline>,
    lstm_hidden_backward_f32: objc2::rc::Retained<MtlPipeline>,
    lstm_cell_backward_f32: objc2::rc::Retained<MtlPipeline>,
    gru_pointwise_f32: objc2::rc::Retained<MtlPipeline>,
    gru_backward_f32: objc2::rc::Retained<MtlPipeline>,
}

impl MetalRnnCell {
    /// `ctx` のデバイス上で RNN セル 5 カーネルを実行時コンパイルし
    /// パイプラインを構築する（`elementwise.rs::MetalElementwise::new`
    /// と同一手順）。
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        let src = objc2_foundation::NSString::from_str(RNN_CELL_MSL_SRC);
        let options = pipeline::compile_options();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&src, Some(&options))
            .map_err(|err| MetalError::LibraryCompilation {
                message: err.localizedDescription().to_string(),
            })?;

        let lstm_pointwise_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "lstm_pointwise_f32")?;
        let lstm_hidden_backward_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "lstm_hidden_backward_f32")?;
        let lstm_cell_backward_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "lstm_cell_backward_f32")?;
        let gru_pointwise_f32 =
            pipeline::make_pipeline(ctx.device(), &library, "gru_pointwise_f32")?;
        let gru_backward_f32 = pipeline::make_pipeline(ctx.device(), &library, "gru_backward_f32")?;

        Ok(Self {
            lstm_pointwise_f32,
            lstm_hidden_backward_f32,
            lstm_cell_backward_f32,
            gru_pointwise_f32,
            gru_backward_f32,
        })
    }

    /// LSTM セルの pointwise 段（決定 1・1b）。`pre: [B*4H]`（行優先）・
    /// `c_prev: [B*H]` から `(gates: [B*4H], c: [B*H], h: [B*H])` を
    /// 計算する。
    pub fn run_lstm_pointwise_f32(
        &self,
        ctx: &MetalContext,
        pre: &[f32],
        c_prev: &[f32],
        hidden: usize,
    ) -> Result<TripleVecOutput, MetalError> {
        let numel = c_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(pre.len(), numel * 4, "pre")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let pre_buf = MetalBuffer::new_with_data(ctx, pre)?;
        let c_prev_buf = MetalBuffer::new_with_data(ctx, c_prev)?;
        // イシュー #1021: 全カーネルが `idx < numel` ガード内で出力
        // バッファの全要素を必ず書くため `alloc_uninit_pooled` を使う
        // （`elementwise.rs::run_binary` と同じ適用条件）。
        let gates_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel * 4)?;
        let c_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;
        let h_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.lstm_pointwise_f32);
            set_buffer_arg(encoder, &pre_buf, 0);
            set_buffer_arg(encoder, &c_prev_buf, 1);
            set_buffer_arg(encoder, &gates_buf, 2);
            set_buffer_arg(encoder, &c_buf, 3);
            set_buffer_arg(encoder, &h_buf, 4);
            set_u32_arg(encoder, hidden as u32, 5);
            set_u32_arg(encoder, numel as u32, 6);
            let (threadgroups, threads_per_tg) = dispatch_sizes(numel);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        Ok((
            gates_buf.read_to_vec(),
            c_buf.read_to_vec(),
            h_buf.read_to_vec(),
        ))
    }

    /// `Op::LstmHidden` の VJP 補助。`c`・`gate_o`・`dh` はいずれも
    /// `[B*H]`。
    pub fn run_lstm_hidden_backward_f32(
        &self,
        ctx: &MetalContext,
        c: &[f32],
        gate_o: &[f32],
        dh: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
        let numel = c.len();
        require_len(gate_o.len(), numel, "gate_o")?;
        require_len(dh.len(), numel, "dh")?;
        if numel > u32::MAX as usize {
            return Err(MetalError::InvalidElementwiseShape {
                detail: format!("rnn cell numel must fit in u32: numel={numel}"),
            });
        }
        if numel == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let c_buf = MetalBuffer::new_with_data(ctx, c)?;
        let gate_o_buf = MetalBuffer::new_with_data(ctx, gate_o)?;
        let dh_buf = MetalBuffer::new_with_data(ctx, dh)?;
        let d_pre_o_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;
        let dc_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.lstm_hidden_backward_f32);
            set_buffer_arg(encoder, &c_buf, 0);
            set_buffer_arg(encoder, &gate_o_buf, 1);
            set_buffer_arg(encoder, &dh_buf, 2);
            set_buffer_arg(encoder, &d_pre_o_buf, 3);
            set_buffer_arg(encoder, &dc_buf, 4);
            set_u32_arg(encoder, numel as u32, 5);
            let (threadgroups, threads_per_tg) = dispatch_sizes(numel);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        Ok((d_pre_o_buf.read_to_vec(), dc_buf.read_to_vec()))
    }

    /// `Op::LstmCell` の VJP 補助。`gates_ifg: [B*3H]`・`c_prev`／
    /// `dc: [B*H]` から `(d_pre_ifg: [B*3H], dc_prev: [B*H])` を計算
    /// する。
    pub fn run_lstm_cell_backward_f32(
        &self,
        ctx: &MetalContext,
        gates_ifg: &[f32],
        c_prev: &[f32],
        dc: &[f32],
        hidden: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
        let numel = c_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(gates_ifg.len(), numel * 3, "gates_ifg")?;
        require_len(dc.len(), numel, "dc")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let gates_buf = MetalBuffer::new_with_data(ctx, gates_ifg)?;
        let c_prev_buf = MetalBuffer::new_with_data(ctx, c_prev)?;
        let dc_buf = MetalBuffer::new_with_data(ctx, dc)?;
        let d_pre_ifg_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel * 3)?;
        let dc_prev_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.lstm_cell_backward_f32);
            set_buffer_arg(encoder, &gates_buf, 0);
            set_buffer_arg(encoder, &c_prev_buf, 1);
            set_buffer_arg(encoder, &dc_buf, 2);
            set_buffer_arg(encoder, &d_pre_ifg_buf, 3);
            set_buffer_arg(encoder, &dc_prev_buf, 4);
            set_u32_arg(encoder, hidden as u32, 5);
            set_u32_arg(encoder, numel as u32, 6);
            let (threadgroups, threads_per_tg) = dispatch_sizes(numel);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        Ok((d_pre_ifg_buf.read_to_vec(), dc_prev_buf.read_to_vec()))
    }

    /// GRU セルの pointwise 段（決定 1c・5）。`pre_i`／`pre_h: [B*3H]`・
    /// `h_prev: [B*H]` から `(gates: [B*3H], q: [B*H], h: [B*H])` を
    /// 計算する。
    pub fn run_gru_pointwise_f32(
        &self,
        ctx: &MetalContext,
        pre_i: &[f32],
        pre_h: &[f32],
        h_prev: &[f32],
        hidden: usize,
    ) -> Result<TripleVecOutput, MetalError> {
        let numel = h_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(pre_i.len(), numel * 3, "pre_i")?;
        require_len(pre_h.len(), numel * 3, "pre_h")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let pre_i_buf = MetalBuffer::new_with_data(ctx, pre_i)?;
        let pre_h_buf = MetalBuffer::new_with_data(ctx, pre_h)?;
        let h_prev_buf = MetalBuffer::new_with_data(ctx, h_prev)?;
        let gates_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel * 3)?;
        let q_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;
        let h_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.gru_pointwise_f32);
            set_buffer_arg(encoder, &pre_i_buf, 0);
            set_buffer_arg(encoder, &pre_h_buf, 1);
            set_buffer_arg(encoder, &h_prev_buf, 2);
            set_buffer_arg(encoder, &gates_buf, 3);
            set_buffer_arg(encoder, &q_buf, 4);
            set_buffer_arg(encoder, &h_buf, 5);
            set_u32_arg(encoder, hidden as u32, 6);
            set_u32_arg(encoder, numel as u32, 7);
            let (threadgroups, threads_per_tg) = dispatch_sizes(numel);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        Ok((
            gates_buf.read_to_vec(),
            q_buf.read_to_vec(),
            h_buf.read_to_vec(),
        ))
    }

    /// `Op::GruCell` の VJP 補助。`gates_rzn: [B*3H]`・`q`／`h_prev`／
    /// `dh: [B*H]` から `(d_pre_i: [B*3H], d_pre_h: [B*3H],
    /// dh_prev_direct: [B*H])` を計算する。
    #[allow(clippy::too_many_arguments)]
    pub fn run_gru_backward_f32(
        &self,
        ctx: &MetalContext,
        gates_rzn: &[f32],
        q: &[f32],
        h_prev: &[f32],
        dh: &[f32],
        hidden: usize,
    ) -> Result<TripleVecOutput, MetalError> {
        let numel = h_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(gates_rzn.len(), numel * 3, "gates_rzn")?;
        require_len(q.len(), numel, "q")?;
        require_len(dh.len(), numel, "dh")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let gates_buf = MetalBuffer::new_with_data(ctx, gates_rzn)?;
        let q_buf = MetalBuffer::new_with_data(ctx, q)?;
        let h_prev_buf = MetalBuffer::new_with_data(ctx, h_prev)?;
        let dh_buf = MetalBuffer::new_with_data(ctx, dh)?;
        let d_pre_i_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel * 3)?;
        let d_pre_h_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel * 3)?;
        let dh_prev_direct_buf = MetalBuffer::alloc_uninit_pooled(ctx, numel)?;

        ctx.dispatch_sync(|encoder| {
            encoder.setComputePipelineState(&self.gru_backward_f32);
            set_buffer_arg(encoder, &gates_buf, 0);
            set_buffer_arg(encoder, &q_buf, 1);
            set_buffer_arg(encoder, &h_prev_buf, 2);
            set_buffer_arg(encoder, &dh_buf, 3);
            set_buffer_arg(encoder, &d_pre_i_buf, 4);
            set_buffer_arg(encoder, &d_pre_h_buf, 5);
            set_buffer_arg(encoder, &dh_prev_direct_buf, 6);
            set_u32_arg(encoder, hidden as u32, 7);
            set_u32_arg(encoder, numel as u32, 8);
            let (threadgroups, threads_per_tg) = dispatch_sizes(numel);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        })?;

        Ok((
            d_pre_i_buf.read_to_vec(),
            d_pre_h_buf.read_to_vec(),
            dh_prev_direct_buf.read_to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rnn_dims_rejects_zero_hidden() {
        let err = validate_rnn_dims(4, 0).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_rnn_dims_rejects_non_multiple() {
        let err = validate_rnn_dims(5, 2).unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_rnn_dims_accepts_multiple() {
        assert!(validate_rnn_dims(6, 2).is_ok());
    }

    #[test]
    fn require_len_rejects_mismatch() {
        let err = require_len(4, 5, "test").unwrap_err();
        assert!(matches!(err, MetalError::InvalidElementwiseShape { .. }));
    }
}
