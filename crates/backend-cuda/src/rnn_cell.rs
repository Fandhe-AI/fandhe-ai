//! RNN／LSTM／GRU セル演算（イシュー #1647）の起動 API（NVRTC コンパイル
//! ・保持・実行）。
//!
//! `elementwise.rs::CudaElementwise`・`mse.rs::CudaMse` と同じ構成方針を
//! 踏襲する: [`CudaRnnCell::new`] が `CudaDevice` から 5 カーネル
//! （`kernels_rnn_cell.rs`）を NVRTC コンパイルして保持し、以降は
//! `run_*_f32` へホスト側スライスを渡すだけで H2D → 起動 → 同期 → D2H を
//! 内部で完結できる。`ops.rs::CudaBackendOps::{lstm_pointwise,
//! lstm_hidden_backward, lstm_cell_backward, gru_pointwise, gru_backward}`
//! から `BackendOps` の実装として呼ばれる。
//!
//! RNN（tanh 版）は専用カーネルを持たない（`fandhe_ai_autodiff::var::
//! rnn_cell_forward_value` が既存の `gemm_bias_act`／`add`／`tanh` の
//! 合成で閉じるため。設計 `docs/autodiff-rnn-cell-tape-design.md`
//! 決定 1）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_rnn_cell::{self, RNN_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// 3 つの `Vec<f32>` を返す関数の戻り値型（`clippy::type_complexity`
/// 回避）。`run_lstm_pointwise_f32`／`run_gru_pointwise_f32`／
/// `run_gru_backward_f32` が共有する。
type TripleVecOutput = (Vec<f32>, Vec<f32>, Vec<f32>);

/// 1 起動あたりのブロック次元（1 次元、`RNN_BLOCK_DIM` 幅）。
const RNN_BLOCK: (u32, u32, u32) = (RNN_BLOCK_DIM, 1, 1);

/// `hidden > 0` と `numel = B * hidden`（`numel % hidden == 0`）を検証
/// し `i32::MAX` 上限も検査する（`elementwise.rs::
/// validate_elementwise_len` と同じ理由。カーネル引数 `int numel`／
/// `int hidden` は C の 32bit 符号付き整数のため）。
fn validate_rnn_dims(numel: usize, hidden: usize) -> Result<(), CudaError> {
    if hidden == 0 {
        return Err(CudaError::InvalidElementwiseShape {
            detail: "rnn cell hidden must be > 0".to_string(),
        });
    }
    if !numel.is_multiple_of(hidden) {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "rnn cell numel must be a multiple of hidden: numel={numel}, hidden={hidden}"
            ),
        });
    }
    if numel > i32::MAX as usize || hidden > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "rnn cell numel/hidden must fit in i32 (kernel argument type): numel={numel}, \
                 hidden={hidden}"
            ),
        });
    }
    Ok(())
}

/// 2 スライスの長さが一致することを検証する（`ops.rs` の事前検証を
/// 経ない直接呼び出しに対しても型付きエラーで拒否する契約。
/// `elementwise.rs::validate_elementwise_binary_dims` と同じ構成）。
fn require_len(actual: usize, expected: usize, what: &str) -> Result<(), CudaError> {
    if actual != expected {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!(
                "rnn cell {what} length mismatch: expected={expected}, actual={actual}"
            ),
        });
    }
    Ok(())
}

fn launch_config(numel: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (numel.div_ceil(RNN_BLOCK.0), 1, 1),
        block_dim: RNN_BLOCK,
        shared_mem_bytes: 0,
    }
}

/// RNN／LSTM／GRU セル演算 5 カーネル（いずれも f32）のコンパイル済み
/// ハンドルを保持する。
pub struct CudaRnnCell {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（`gemm.rs::CudaGemm::ordinal` と
    /// 同じ役割。`Self::with_driver_call` が capture 排他のキーとして
    /// 使う）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    lstm_pointwise_f32: CudaFunction,
    lstm_hidden_backward_f32: CudaFunction,
    lstm_cell_backward_f32: CudaFunction,
    gru_pointwise_f32: CudaFunction,
    gru_backward_f32: CudaFunction,
}

impl CudaRnnCell {
    /// `device` 上で RNN セル 5 カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する（`elementwise.rs::CudaElementwise::new` と
    /// 同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let lstm_pointwise_ptx = compile_ptx(kernels_rnn_cell::LSTM_POINTWISE_F32, arch)?;
        let lstm_hidden_backward_ptx =
            compile_ptx(kernels_rnn_cell::LSTM_HIDDEN_BACKWARD_F32, arch)?;
        let lstm_cell_backward_ptx = compile_ptx(kernels_rnn_cell::LSTM_CELL_BACKWARD_F32, arch)?;
        let gru_pointwise_ptx = compile_ptx(kernels_rnn_cell::GRU_POINTWISE_F32, arch)?;
        let gru_backward_ptx = compile_ptx(kernels_rnn_cell::GRU_BACKWARD_F32, arch)?;

        let lstm_pointwise_f32 = device
            .context()
            .load_module(lstm_pointwise_ptx)?
            .load_function("lstm_pointwise_f32")?;
        let lstm_hidden_backward_f32 = device
            .context()
            .load_module(lstm_hidden_backward_ptx)?
            .load_function("lstm_hidden_backward_f32")?;
        let lstm_cell_backward_f32 = device
            .context()
            .load_module(lstm_cell_backward_ptx)?
            .load_function("lstm_cell_backward_f32")?;
        let gru_pointwise_f32 = device
            .context()
            .load_module(gru_pointwise_ptx)?
            .load_function("gru_pointwise_f32")?;
        let gru_backward_f32 = device
            .context()
            .load_module(gru_backward_ptx)?
            .load_function("gru_backward_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            lstm_pointwise_f32,
            lstm_hidden_backward_f32,
            lstm_cell_backward_f32,
            gru_pointwise_f32,
            gru_backward_f32,
        })
    }

    /// `CudaRnnCell` の driver 呼び出し（H2D 転送・カーネル起動・D2H
    /// readback）を CUDA Graph capture 排他へ参加させる共通ヘルパー
    /// （`elementwise.rs::CudaElementwise::with_driver_call` と同じ
    /// 設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// LSTM セルの pointwise 段（決定 1・1b）。`pre: [B*4H]`（行優先）・
    /// `c_prev: [B*H]` から `(gates: [B*4H], c: [B*H], h: [B*H])` を
    /// 計算する。
    pub fn run_lstm_pointwise_f32(
        &self,
        pre: &[f32],
        c_prev: &[f32],
        hidden: usize,
    ) -> Result<TripleVecOutput, CudaError> {
        let numel = c_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(pre.len(), numel * 4, "pre")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        self.with_driver_call(|| {
            let pre_dev = self.stream.clone_htod(pre)?;
            let c_prev_dev = self.stream.clone_htod(c_prev)?;
            let mut gates_dev = self.allocator.alloc_uninit_f32(numel * 4)?;
            let mut c_dev = self.allocator.alloc_uninit_f32(numel)?;
            let mut h_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = launch_config(numel as u32);
            let hidden_i = hidden as i32;
            let numel_i = numel as i32;

            // SAFETY: 各バッファ長は上記で検証済みの `numel`／`hidden`
            // に対応し、カーネル内の手動境界チェック（`if (idx <
            // numel)`。`kernels_rnn_cell.rs` 参照、REQ-8）と合わせて
            // OOB 読み書きが起きない根拠とする。グリッド次元は
            // `div_ceil` で numel を包含するよう構築しており、末尾
            // ブロックの余剰スレッドはカーネル内境界チェックで弾かれる。
            unsafe {
                self.stream
                    .launch_builder(&self.lstm_pointwise_f32)
                    .arg(&pre_dev)
                    .arg(&c_prev_dev)
                    .arg(&mut gates_dev.as_view_mut())
                    .arg(&mut c_dev.as_view_mut())
                    .arg(&mut h_dev.as_view_mut())
                    .arg(&hidden_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }

            let gates = readback(&self.stream, &gates_dev.as_view())?;
            let c = readback(&self.stream, &c_dev.as_view())?;
            let h = readback(&self.stream, &h_dev.as_view())?;
            Ok((gates, c, h))
        })
    }

    /// `Op::LstmHidden` の VJP 補助。`c`・`gate_o`・`dh` はいずれも
    /// `[B*H]`。
    pub fn run_lstm_hidden_backward_f32(
        &self,
        c: &[f32],
        gate_o: &[f32],
        dh: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let numel = c.len();
        require_len(gate_o.len(), numel, "gate_o")?;
        require_len(dh.len(), numel, "dh")?;
        if numel > i32::MAX as usize {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!("rnn cell numel must fit in i32: numel={numel}"),
            });
        }
        if numel == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        self.with_driver_call(|| {
            let c_dev = self.stream.clone_htod(c)?;
            let gate_o_dev = self.stream.clone_htod(gate_o)?;
            let dh_dev = self.stream.clone_htod(dh)?;
            let mut d_pre_o_dev = self.allocator.alloc_uninit_f32(numel)?;
            let mut dc_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = launch_config(numel as u32);
            let numel_i = numel as i32;

            // SAFETY: run_lstm_pointwise_f32 と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.lstm_hidden_backward_f32)
                    .arg(&c_dev)
                    .arg(&gate_o_dev)
                    .arg(&dh_dev)
                    .arg(&mut d_pre_o_dev.as_view_mut())
                    .arg(&mut dc_dev.as_view_mut())
                    .arg(&numel_i)
                    .launch(cfg)?;
            }

            let d_pre_o = readback(&self.stream, &d_pre_o_dev.as_view())?;
            let dc = readback(&self.stream, &dc_dev.as_view())?;
            Ok((d_pre_o, dc))
        })
    }

    /// `Op::LstmCell` の VJP 補助。`gates_ifg: [B*3H]`・`c_prev`／
    /// `dc: [B*H]` から `(d_pre_ifg: [B*3H], dc_prev: [B*H])` を計算
    /// する。
    pub fn run_lstm_cell_backward_f32(
        &self,
        gates_ifg: &[f32],
        c_prev: &[f32],
        dc: &[f32],
        hidden: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let numel = c_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(gates_ifg.len(), numel * 3, "gates_ifg")?;
        require_len(dc.len(), numel, "dc")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        self.with_driver_call(|| {
            let gates_dev = self.stream.clone_htod(gates_ifg)?;
            let c_prev_dev = self.stream.clone_htod(c_prev)?;
            let dc_dev = self.stream.clone_htod(dc)?;
            let mut d_pre_ifg_dev = self.allocator.alloc_uninit_f32(numel * 3)?;
            let mut dc_prev_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = launch_config(numel as u32);
            let hidden_i = hidden as i32;
            let numel_i = numel as i32;

            // SAFETY: run_lstm_pointwise_f32 と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.lstm_cell_backward_f32)
                    .arg(&gates_dev)
                    .arg(&c_prev_dev)
                    .arg(&dc_dev)
                    .arg(&mut d_pre_ifg_dev.as_view_mut())
                    .arg(&mut dc_prev_dev.as_view_mut())
                    .arg(&hidden_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }

            let d_pre_ifg = readback(&self.stream, &d_pre_ifg_dev.as_view())?;
            let dc_prev = readback(&self.stream, &dc_prev_dev.as_view())?;
            Ok((d_pre_ifg, dc_prev))
        })
    }

    /// GRU セルの pointwise 段（決定 1c・5）。`pre_i`／`pre_h: [B*3H]`・
    /// `h_prev: [B*H]` から `(gates: [B*3H], q: [B*H], h: [B*H])` を
    /// 計算する。
    pub fn run_gru_pointwise_f32(
        &self,
        pre_i: &[f32],
        pre_h: &[f32],
        h_prev: &[f32],
        hidden: usize,
    ) -> Result<TripleVecOutput, CudaError> {
        let numel = h_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(pre_i.len(), numel * 3, "pre_i")?;
        require_len(pre_h.len(), numel * 3, "pre_h")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        self.with_driver_call(|| {
            let pre_i_dev = self.stream.clone_htod(pre_i)?;
            let pre_h_dev = self.stream.clone_htod(pre_h)?;
            let h_prev_dev = self.stream.clone_htod(h_prev)?;
            let mut gates_dev = self.allocator.alloc_uninit_f32(numel * 3)?;
            let mut q_dev = self.allocator.alloc_uninit_f32(numel)?;
            let mut h_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = launch_config(numel as u32);
            let hidden_i = hidden as i32;
            let numel_i = numel as i32;

            // SAFETY: run_lstm_pointwise_f32 と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.gru_pointwise_f32)
                    .arg(&pre_i_dev)
                    .arg(&pre_h_dev)
                    .arg(&h_prev_dev)
                    .arg(&mut gates_dev.as_view_mut())
                    .arg(&mut q_dev.as_view_mut())
                    .arg(&mut h_dev.as_view_mut())
                    .arg(&hidden_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }

            let gates = readback(&self.stream, &gates_dev.as_view())?;
            let q = readback(&self.stream, &q_dev.as_view())?;
            let h = readback(&self.stream, &h_dev.as_view())?;
            Ok((gates, q, h))
        })
    }

    /// `Op::GruCell` の VJP 補助。`gates_rzn: [B*3H]`・`q`／`h_prev`／
    /// `dh: [B*H]` から `(d_pre_i: [B*3H], d_pre_h: [B*3H],
    /// dh_prev_direct: [B*H])` を計算する。
    pub fn run_gru_backward_f32(
        &self,
        gates_rzn: &[f32],
        q: &[f32],
        h_prev: &[f32],
        dh: &[f32],
        hidden: usize,
    ) -> Result<TripleVecOutput, CudaError> {
        let numel = h_prev.len();
        validate_rnn_dims(numel, hidden)?;
        require_len(gates_rzn.len(), numel * 3, "gates_rzn")?;
        require_len(q.len(), numel, "q")?;
        require_len(dh.len(), numel, "dh")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        self.with_driver_call(|| {
            let gates_dev = self.stream.clone_htod(gates_rzn)?;
            let q_dev = self.stream.clone_htod(q)?;
            let h_prev_dev = self.stream.clone_htod(h_prev)?;
            let dh_dev = self.stream.clone_htod(dh)?;
            let mut d_pre_i_dev = self.allocator.alloc_uninit_f32(numel * 3)?;
            let mut d_pre_h_dev = self.allocator.alloc_uninit_f32(numel * 3)?;
            let mut dh_prev_direct_dev = self.allocator.alloc_uninit_f32(numel)?;

            let cfg = launch_config(numel as u32);
            let hidden_i = hidden as i32;
            let numel_i = numel as i32;

            // SAFETY: run_lstm_pointwise_f32 と同一の根拠。
            unsafe {
                self.stream
                    .launch_builder(&self.gru_backward_f32)
                    .arg(&gates_dev)
                    .arg(&q_dev)
                    .arg(&h_prev_dev)
                    .arg(&dh_dev)
                    .arg(&mut d_pre_i_dev.as_view_mut())
                    .arg(&mut d_pre_h_dev.as_view_mut())
                    .arg(&mut dh_prev_direct_dev.as_view_mut())
                    .arg(&hidden_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }

            let d_pre_i = readback(&self.stream, &d_pre_i_dev.as_view())?;
            let d_pre_h = readback(&self.stream, &d_pre_h_dev.as_view())?;
            let dh_prev_direct = readback(&self.stream, &dh_prev_direct_dev.as_view())?;
            Ok((d_pre_i, d_pre_h, dh_prev_direct))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rnn_dims_rejects_zero_hidden() {
        let err = validate_rnn_dims(4, 0).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_rnn_dims_rejects_non_multiple() {
        let err = validate_rnn_dims(5, 2).unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }

    #[test]
    fn validate_rnn_dims_accepts_multiple() {
        assert!(validate_rnn_dims(6, 2).is_ok());
    }

    #[test]
    fn require_len_rejects_mismatch() {
        let err = require_len(4, 5, "test").unwrap_err();
        assert!(matches!(err, CudaError::InvalidElementwiseShape { .. }));
    }
}
