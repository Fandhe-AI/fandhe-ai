//! LSTM／GRU セルの pointwise 段と backward の ROCm 実行入口（`backend-cuda::rnn_cell` の移植。
//! イシュー #3441・分解案 R10）。
//!
//! 役割: HIP/HIPRTC をロードして静的ソース（`kernels_rnn_cell`）をコンパイルし、ホストのスライスを
//! 受けて `Vec` を返す実行型 [`RocmRnnCell`] を提供する。`BackendOps` への結線（rank・形状の検査、
//! 空入力の CPU 委譲、`Tensor` 化、`RocmError` の写像）は `ops/rnn_cell.rs` の責務で、本ファイルは
//! スライス入出力と起動前検証のみ。
//!
//! 起動は `KernelHost::run` に集約され、呼び出しごとに `synchronize` する（Metal と同じ「呼び出しごとに
//! 同期」）。新しい `unsafe` はなく、監査済みの `KernelHost::run`・`TypedDeviceBuffer` を再利用する。
//! ホスト側で `i32` 範囲・`numel * gates` の checked 演算・スライス長を検証してから起動する
//! （カーネル側 REQ-8 の `gid >= numel` ガードと二重化。`base = row * gates * hidden` は
//! カーネル側で `long long` だが、ゲート配列全体を `i32` に収める CUDA 版の契約を踏襲する）。

use crate::elementwise::fit_i32;
use crate::error::RocmError;
use crate::kernels_rnn_cell::{
    GRU_BACKWARD_SPEC, GRU_POINTWISE_SPEC, LSTM_CELL_BACKWARD_SPEC, LSTM_HIDDEN_BACKWARD_SPEC,
    LSTM_POINTWISE_SPEC, render_rnn_cell_source,
};
use crate::launch::{Arg, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

/// 3 つの `Vec<f32>` を返す関数の戻り値型（`clippy::type_complexity` 回避）。
type Triple = (Vec<f32>, Vec<f32>, Vec<f32>);

const OP_LSTM_POINTWISE: &str = "lstm_pointwise";
const OP_LSTM_HIDDEN_BACKWARD: &str = "lstm_hidden_backward";
const OP_LSTM_CELL_BACKWARD: &str = "lstm_cell_backward";
const OP_GRU_POINTWISE: &str = "gru_pointwise";
const OP_GRU_BACKWARD: &str = "gru_backward";

fn invalid(op: &'static str, detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op, detail }
}

/// `hidden > 0`・`numel % hidden == 0`・`numel`／`hidden` の `i32` 収容・`numel * gates` の `i32` 収容を
/// 検査する（CUDA 版 `validate_rnn_dims` と同じ契約）。`gates` はゲート配列の 1 行あたりのブロック数
/// （LSTM pointwise は 4、GRU・`lstm_cell_backward` は 3、ゲート配列を持たない呼び出しは 1）。
/// 起動前に呼ばれる純関数で、HIP に触れない。
pub(crate) fn validate_rnn_dims(
    op: &'static str,
    numel: usize,
    hidden: usize,
    gates: usize,
) -> Result<(), RocmError> {
    if hidden == 0 {
        return Err(invalid(op, "hidden は 1 以上が必要です".into()));
    }
    if !numel.is_multiple_of(hidden) {
        return Err(invalid(
            op,
            format!("numel は hidden の倍数が必要です: numel={numel}, hidden={hidden}"),
        ));
    }
    fit_i32(op, numel, "numel")?;
    fit_i32(op, hidden, "hidden")?;
    match numel.checked_mul(gates) {
        Some(gated) if gated <= i32::MAX as usize => Ok(()),
        _ => Err(invalid(
            op,
            format!(
                "numel * gates は i32 に収まる必要があります（ゲート配列の添字基点）: \
                 numel={numel}, gates={gates}"
            ),
        )),
    }
}

/// スライス長の一致を検査する。
fn check_len(
    op: &'static str,
    name: &str,
    expected: usize,
    actual: usize,
) -> Result<(), RocmError> {
    if expected != actual {
        return Err(invalid(
            op,
            format!("{name} の長さ不一致: 期待 {expected}, 実際 {actual}"),
        ));
    }
    Ok(())
}

/// コンパイル済みの RNN セル 5 カーネルを保持し、実行する。
pub struct RocmRnnCell {
    host: KernelHost,
}

impl RocmRnnCell {
    /// HIP/HIPRTC をロードし 5 カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_rnn_cell_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// LSTM の pointwise 段。`pre: [B, 4H]`（列ブロック順 `i,f,g,o`）・`c_prev: [B, H]` から
    /// `(gates [B, 4H], c [B, H], h [B, H])` を返す。
    pub fn run_lstm_pointwise_f32(
        &self,
        pre: &[f32],
        c_prev: &[f32],
        hidden: usize,
    ) -> Result<Triple, RocmError> {
        let op = OP_LSTM_POINTWISE;
        let numel = c_prev.len();
        validate_rnn_dims(op, numel, hidden, 4)?;
        check_len(op, "pre", numel * 4, pre.len())?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_pre = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), pre)?;
        let d_c_prev = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), c_prev)?;
        let d_gates = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel * 4)?;
        let d_c = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel)?;
        let d_h = TypedDeviceBuffer::<f32>::new_zeroed(rt, numel)?;
        let args = [
            Arg::Ptr(d_pre.as_device_ptr()),
            Arg::Ptr(d_c_prev.as_device_ptr()),
            Arg::Ptr(d_gates.as_device_ptr()),
            Arg::Ptr(d_c.as_device_ptr()),
            Arg::Ptr(d_h.as_device_ptr()),
            Arg::I32(fit_i32(op, hidden, "hidden")?),
            Arg::I32(fit_i32(op, numel, "numel")?),
        ];
        self.host.run(op, &LSTM_POINTWISE_SPEC, numel, &args)?;
        Ok((d_gates.to_vec()?, d_c.to_vec()?, d_h.to_vec()?))
    }

    /// `Op::LstmHidden` の VJP 補助。`c`・`gate_o`・`dh` はいずれも `[B, H]`（同長）。
    /// `(d_pre_o, dc)` を返す。
    pub fn run_lstm_hidden_backward_f32(
        &self,
        c: &[f32],
        gate_o: &[f32],
        dh: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), RocmError> {
        let op = OP_LSTM_HIDDEN_BACKWARD;
        let numel = c.len();
        check_len(op, "gate_o", numel, gate_o.len())?;
        check_len(op, "dh", numel, dh.len())?;
        let numel_i = fit_i32(op, numel, "numel")?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_c = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), c)?;
        let d_o = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), gate_o)?;
        let d_dh = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), dh)?;
        let d_pre_o = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel)?;
        let d_dc = TypedDeviceBuffer::<f32>::new_zeroed(rt, numel)?;
        let args = [
            Arg::Ptr(d_c.as_device_ptr()),
            Arg::Ptr(d_o.as_device_ptr()),
            Arg::Ptr(d_dh.as_device_ptr()),
            Arg::Ptr(d_pre_o.as_device_ptr()),
            Arg::Ptr(d_dc.as_device_ptr()),
            Arg::I32(numel_i),
        ];
        self.host
            .run(op, &LSTM_HIDDEN_BACKWARD_SPEC, numel, &args)?;
        Ok((d_pre_o.to_vec()?, d_dc.to_vec()?))
    }

    /// `Op::LstmCell` の VJP 補助。`gates_ifg: [B, 3H]`・`c_prev`／`dc: [B, H]` から
    /// `(d_pre_ifg [B, 3H], dc_prev [B, H])` を返す。
    pub fn run_lstm_cell_backward_f32(
        &self,
        gates_ifg: &[f32],
        c_prev: &[f32],
        dc: &[f32],
        hidden: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), RocmError> {
        let op = OP_LSTM_CELL_BACKWARD;
        let numel = c_prev.len();
        validate_rnn_dims(op, numel, hidden, 3)?;
        check_len(op, "gates_ifg", numel * 3, gates_ifg.len())?;
        check_len(op, "dc", numel, dc.len())?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_gates = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), gates_ifg)?;
        let d_c_prev = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), c_prev)?;
        let d_dc = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), dc)?;
        let d_pre = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel * 3)?;
        let d_dc_prev = TypedDeviceBuffer::<f32>::new_zeroed(rt, numel)?;
        let args = [
            Arg::Ptr(d_gates.as_device_ptr()),
            Arg::Ptr(d_c_prev.as_device_ptr()),
            Arg::Ptr(d_dc.as_device_ptr()),
            Arg::Ptr(d_pre.as_device_ptr()),
            Arg::Ptr(d_dc_prev.as_device_ptr()),
            Arg::I32(fit_i32(op, hidden, "hidden")?),
            Arg::I32(fit_i32(op, numel, "numel")?),
        ];
        self.host.run(op, &LSTM_CELL_BACKWARD_SPEC, numel, &args)?;
        Ok((d_pre.to_vec()?, d_dc_prev.to_vec()?))
    }

    /// GRU の pointwise 段（`reset_after=True`）。`pre_i`／`pre_h: [B, 3H]`（列ブロック順 `r,z,n`）・
    /// `h_prev: [B, H]` から `(gates [B, 3H], q [B, H], h [B, H])` を返す。
    pub fn run_gru_pointwise_f32(
        &self,
        pre_i: &[f32],
        pre_h: &[f32],
        h_prev: &[f32],
        hidden: usize,
    ) -> Result<Triple, RocmError> {
        let op = OP_GRU_POINTWISE;
        let numel = h_prev.len();
        validate_rnn_dims(op, numel, hidden, 3)?;
        check_len(op, "pre_i", numel * 3, pre_i.len())?;
        check_len(op, "pre_h", numel * 3, pre_h.len())?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_pre_i = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), pre_i)?;
        let d_pre_h = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), pre_h)?;
        let d_h_prev = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), h_prev)?;
        let d_gates = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel * 3)?;
        let d_q = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel)?;
        let d_h = TypedDeviceBuffer::<f32>::new_zeroed(rt, numel)?;
        let args = [
            Arg::Ptr(d_pre_i.as_device_ptr()),
            Arg::Ptr(d_pre_h.as_device_ptr()),
            Arg::Ptr(d_h_prev.as_device_ptr()),
            Arg::Ptr(d_gates.as_device_ptr()),
            Arg::Ptr(d_q.as_device_ptr()),
            Arg::Ptr(d_h.as_device_ptr()),
            Arg::I32(fit_i32(op, hidden, "hidden")?),
            Arg::I32(fit_i32(op, numel, "numel")?),
        ];
        self.host.run(op, &GRU_POINTWISE_SPEC, numel, &args)?;
        Ok((d_gates.to_vec()?, d_q.to_vec()?, d_h.to_vec()?))
    }

    /// `Op::GruCell` の VJP 補助。`gates_rzn: [B, 3H]`・`q`／`h_prev`／`dh: [B, H]` から
    /// `(d_pre_i [B, 3H], d_pre_h [B, 3H], dh_prev_direct [B, H])` を返す。
    pub fn run_gru_backward_f32(
        &self,
        gates_rzn: &[f32],
        q: &[f32],
        h_prev: &[f32],
        dh: &[f32],
        hidden: usize,
    ) -> Result<Triple, RocmError> {
        let op = OP_GRU_BACKWARD;
        let numel = h_prev.len();
        validate_rnn_dims(op, numel, hidden, 3)?;
        check_len(op, "gates_rzn", numel * 3, gates_rzn.len())?;
        check_len(op, "q", numel, q.len())?;
        check_len(op, "dh", numel, dh.len())?;
        if numel == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let d_gates = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), gates_rzn)?;
        let d_q = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), q)?;
        let d_h_prev = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), h_prev)?;
        let d_dh = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), dh)?;
        let d_pre_i = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel * 3)?;
        let d_pre_h = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), numel * 3)?;
        let d_direct = TypedDeviceBuffer::<f32>::new_zeroed(rt, numel)?;
        let args = [
            Arg::Ptr(d_gates.as_device_ptr()),
            Arg::Ptr(d_q.as_device_ptr()),
            Arg::Ptr(d_h_prev.as_device_ptr()),
            Arg::Ptr(d_dh.as_device_ptr()),
            Arg::Ptr(d_pre_i.as_device_ptr()),
            Arg::Ptr(d_pre_h.as_device_ptr()),
            Arg::Ptr(d_direct.as_device_ptr()),
            Arg::I32(fit_i32(op, hidden, "hidden")?),
            Arg::I32(fit_i32(op, numel, "numel")?),
        ];
        self.host.run(op, &GRU_BACKWARD_SPEC, numel, &args)?;
        Ok((d_pre_i.to_vec()?, d_pre_h.to_vec()?, d_direct.to_vec()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_normal_and_empty() {
        validate_rnn_dims("t", 12, 4, 4).unwrap();
        validate_rnn_dims("t", 0, 4, 4).unwrap();
        validate_rnn_dims("t", i32::MAX as usize, 1, 1).unwrap();
    }

    #[test]
    fn validate_rejects_zero_hidden_and_ragged_numel() {
        assert!(matches!(
            validate_rnn_dims("t", 4, 0, 4),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            validate_rnn_dims("t", 7, 4, 4),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }

    #[test]
    fn validate_rejects_i32_overflow_of_gate_array() {
        // numel 単体は i32 に収まるが numel * 4 が超える。
        let numel = (i32::MAX as usize) / 2;
        assert!(matches!(
            validate_rnn_dims("t", numel, 1, 4),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            validate_rnn_dims("t", i32::MAX as usize + 1, 1, 1),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // checked_mul の overflow も型付きエラー（panic しない）。
        assert!(matches!(
            validate_rnn_dims("t", usize::MAX - usize::MAX % 2, 2, usize::MAX),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }

    #[test]
    fn check_len_reports_mismatch() {
        check_len("t", "x", 3, 3).unwrap();
        assert!(matches!(
            check_len("t", "x", 3, 2),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }
}
