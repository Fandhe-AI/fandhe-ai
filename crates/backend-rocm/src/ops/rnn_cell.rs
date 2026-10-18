//! `RocmBackendOps` の LSTM／GRU セル 5 メソッドの結線（イシュー #3441・分解案 R10）。
//!
//! 役割: `ops.rs` の `BackendOps` impl から 1 行で呼ばれ、(1) rank・形状の検査（CUDA 版
//! `backend-cuda::ops` と同じ検査列・同じエラー種別）、(2) 空入力（`B == 0` または `hidden == 0`）の
//! CPU 委譲、(3) contiguous 化、(4) 起動前検証（`validate_rnn_dims`。どちらの engine でも通す）、
//! (5) engine（Hip／テストの HostModel）の分岐、(6) `Tensor` の組み立て、を担う。カーネル起動そのものは
//! `rnn_cell.rs` の `RocmRnnCell`。
//!
//! ROCm 非搭載・コンパイル失敗は `DeviceUnavailable`／`KernelLaunchFailed` で返し、`Unsupported` へは
//! 落とさない（autodiff のホスト参照実装へ黙ってフォールバックさせない。security.md A08）。
//! 形状・引数の不正は ROCm の有無によらず、デバイス取得より前に型付きエラーで拒否する。
//!
//! 結線を子モジュールに置くのは、兄弟 issue（#3454 conv/pool・#3455 optim）の `ops.rs` 変更との
//! 衝突面を小さくするため。thread_local のキャッシュもここで宣言し、`ops.rs` の宣言群には触れない。

use std::cell::RefCell;

use fandhe_ai_tensor_core::{
    BackendError, BackendOps, GruBackwardOutput, GruPointwiseOutput, LstmPointwiseOutput,
    ShapeError, Tensor, require_same_shape,
};

use super::{Cache, Engine, RocmBackendOps, cached, host_slice};
use crate::error::{RocmError, to_backend_error};
use crate::rnn_cell::{RocmRnnCell, validate_rnn_dims};

thread_local! {
    /// RNN セル 5 カーネル（R10・#3441）。
    static RNN_CELL_CACHE: Cache<RocmRnnCell> = const { RefCell::new(Vec::new()) };
}

type Triple = (Vec<f32>, Vec<f32>, Vec<f32>);
type Pair = (Vec<f32>, Vec<f32>);

/// RNN セル系の隠れ状態は rank-2（`[B, H]`）でなければならない。
fn require_rank2_cell(shape: &[usize]) -> Result<(), BackendError> {
    if shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    Ok(())
}

/// `gates * hidden`（ゲート幅）を `checked_mul` で求める。overflow を放置すると期待幅が周回し、
/// 不正な形状を誤って受理して添字が範囲外になる（CUDA 版・CPU 版と同じ理由）。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, BackendError> {
    gates.checked_mul(hidden).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountMismatch {
            expected: usize::MAX,
            actual: 0,
        },
    ))
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<f32>, BackendError> {
    Tensor::new(data, shape).map_err(BackendError::ShapeMismatch)
}

/// 起動前検証を `RocmError` から `BackendError` へ写して実施する。
fn validate(
    op: &'static str,
    numel: usize,
    hidden: usize,
    gates: usize,
) -> Result<(), BackendError> {
    validate_rnn_dims(op, numel, hidden, gates).map_err(to_backend_error)
}

impl RocmBackendOps {
    fn host_lstm_pointwise(
        &self,
        pre: &[f32],
        c_prev: &[f32],
        hidden: usize,
    ) -> Result<Triple, RocmError> {
        match self.engine {
            Engine::Hip => cached(&RNN_CELL_CACHE, self.target, RocmRnnCell::new)?
                .run_lstm_pointwise_f32(pre, c_prev, hidden),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::rnn_cell_model::lstm_pointwise(pre, c_prev, hidden)),
        }
    }

    fn host_lstm_hidden_backward(
        &self,
        c: &[f32],
        gate_o: &[f32],
        dh: &[f32],
    ) -> Result<Pair, RocmError> {
        match self.engine {
            Engine::Hip => cached(&RNN_CELL_CACHE, self.target, RocmRnnCell::new)?
                .run_lstm_hidden_backward_f32(c, gate_o, dh),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::rnn_cell_model::lstm_hidden_backward(c, gate_o, dh)),
        }
    }

    fn host_lstm_cell_backward(
        &self,
        gates: &[f32],
        c_prev: &[f32],
        dc: &[f32],
        hidden: usize,
    ) -> Result<Pair, RocmError> {
        match self.engine {
            Engine::Hip => cached(&RNN_CELL_CACHE, self.target, RocmRnnCell::new)?
                .run_lstm_cell_backward_f32(gates, c_prev, dc, hidden),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::rnn_cell_model::lstm_cell_backward(
                gates, c_prev, dc, hidden,
            )),
        }
    }

    fn host_gru_pointwise(
        &self,
        pre_i: &[f32],
        pre_h: &[f32],
        h_prev: &[f32],
        hidden: usize,
    ) -> Result<Triple, RocmError> {
        match self.engine {
            Engine::Hip => cached(&RNN_CELL_CACHE, self.target, RocmRnnCell::new)?
                .run_gru_pointwise_f32(pre_i, pre_h, h_prev, hidden),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::rnn_cell_model::gru_pointwise(
                pre_i, pre_h, h_prev, hidden,
            )),
        }
    }

    fn host_gru_backward(
        &self,
        gates: &[f32],
        q: &[f32],
        h_prev: &[f32],
        dh: &[f32],
        hidden: usize,
    ) -> Result<Triple, RocmError> {
        match self.engine {
            Engine::Hip => cached(&RNN_CELL_CACHE, self.target, RocmRnnCell::new)?
                .run_gru_backward_f32(gates, q, h_prev, dh, hidden),
            #[cfg(test)]
            Engine::HostModel => Ok(crate::rnn_cell_model::gru_backward(
                gates, q, h_prev, dh, hidden,
            )),
        }
    }

    pub(super) fn rnn_lstm_pointwise(
        &self,
        pre: &Tensor<f32>,
        c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        const OP: &str = "lstm_pointwise";
        require_rank2_cell(c_prev.shape())?;
        let (b_dim, hidden) = (c_prev.shape()[0], c_prev.shape()[1]);
        let gate_width = checked_gate_width(4, hidden)?;
        require_same_shape(pre.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        if b_dim == 0 || hidden == 0 {
            // 空は HIP に触れず CPU と同じ結果・エラーにする。
            return self.cpu.lstm_pointwise(pre, c_prev);
        }
        validate(OP, b_dim * hidden, hidden, 4)?;
        let (pre_c, c_prev_c) = (pre.contiguous(), c_prev.contiguous());
        let (gates, c, h) = self
            .host_lstm_pointwise(host_slice(&pre_c, OP)?, host_slice(&c_prev_c, OP)?, hidden)
            .map_err(to_backend_error)?;
        Ok(LstmPointwiseOutput {
            gates: tensor(gates, &[b_dim, gate_width])?,
            c: tensor(c, &[b_dim, hidden])?,
            h: tensor(h, &[b_dim, hidden])?,
        })
    }

    pub(super) fn rnn_lstm_hidden_backward(
        &self,
        c: &Tensor<f32>,
        gate_o: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        const OP: &str = "lstm_hidden_backward";
        require_rank2_cell(c.shape())?;
        require_same_shape(gate_o.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), c.shape()).map_err(BackendError::ShapeMismatch)?;
        if c.numel() == 0 {
            return self.cpu.lstm_hidden_backward(c, gate_o, dh);
        }
        // hidden の概念を持たないため hidden = 1・gates = 1 で numel の i32 収容だけを検査する。
        validate(OP, c.numel(), 1, 1)?;
        let (c_c, o_c, dh_c) = (c.contiguous(), gate_o.contiguous(), dh.contiguous());
        let (d_pre_o, dc) = self
            .host_lstm_hidden_backward(
                host_slice(&c_c, OP)?,
                host_slice(&o_c, OP)?,
                host_slice(&dh_c, OP)?,
            )
            .map_err(to_backend_error)?;
        Ok((tensor(d_pre_o, c.shape())?, tensor(dc, c.shape())?))
    }

    pub(super) fn rnn_lstm_cell_backward(
        &self,
        gates_ifg: &Tensor<f32>,
        c_prev: &Tensor<f32>,
        dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        const OP: &str = "lstm_cell_backward";
        require_rank2_cell(c_prev.shape())?;
        let (b_dim, hidden) = (c_prev.shape()[0], c_prev.shape()[1]);
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_ifg.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dc.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        if b_dim == 0 || hidden == 0 {
            return self.cpu.lstm_cell_backward(gates_ifg, c_prev, dc);
        }
        validate(OP, b_dim * hidden, hidden, 3)?;
        let (g_c, c_prev_c, dc_c) = (gates_ifg.contiguous(), c_prev.contiguous(), dc.contiguous());
        let (d_pre, dc_prev) = self
            .host_lstm_cell_backward(
                host_slice(&g_c, OP)?,
                host_slice(&c_prev_c, OP)?,
                host_slice(&dc_c, OP)?,
                hidden,
            )
            .map_err(to_backend_error)?;
        Ok((
            tensor(d_pre, &[b_dim, gate_width])?,
            tensor(dc_prev, &[b_dim, hidden])?,
        ))
    }

    pub(super) fn rnn_gru_pointwise(
        &self,
        pre_i: &Tensor<f32>,
        pre_h: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        const OP: &str = "gru_pointwise";
        require_rank2_cell(h_prev.shape())?;
        let (b_dim, hidden) = (h_prev.shape()[0], h_prev.shape()[1]);
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(pre_i.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(pre_h.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        if b_dim == 0 || hidden == 0 {
            return self.cpu.gru_pointwise(pre_i, pre_h, h_prev);
        }
        validate(OP, b_dim * hidden, hidden, 3)?;
        let (i_c, h_c, hp_c) = (pre_i.contiguous(), pre_h.contiguous(), h_prev.contiguous());
        let (gates, q, h) = self
            .host_gru_pointwise(
                host_slice(&i_c, OP)?,
                host_slice(&h_c, OP)?,
                host_slice(&hp_c, OP)?,
                hidden,
            )
            .map_err(to_backend_error)?;
        Ok(GruPointwiseOutput {
            gates: tensor(gates, &[b_dim, gate_width])?,
            q: tensor(q, &[b_dim, hidden])?,
            h: tensor(h, &[b_dim, hidden])?,
        })
    }

    pub(super) fn rnn_gru_backward(
        &self,
        gates_rzn: &Tensor<f32>,
        q: &Tensor<f32>,
        h_prev: &Tensor<f32>,
        dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        const OP: &str = "gru_backward";
        require_rank2_cell(h_prev.shape())?;
        let (b_dim, hidden) = (h_prev.shape()[0], h_prev.shape()[1]);
        let gate_width = checked_gate_width(3, hidden)?;
        require_same_shape(gates_rzn.shape(), &[b_dim, gate_width])
            .map_err(BackendError::ShapeMismatch)?;
        require_same_shape(q.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        require_same_shape(dh.shape(), &[b_dim, hidden]).map_err(BackendError::ShapeMismatch)?;
        if b_dim == 0 || hidden == 0 {
            return self.cpu.gru_backward(gates_rzn, q, h_prev, dh);
        }
        validate(OP, b_dim * hidden, hidden, 3)?;
        let (g_c, q_c, hp_c, dh_c) = (
            gates_rzn.contiguous(),
            q.contiguous(),
            h_prev.contiguous(),
            dh.contiguous(),
        );
        let (d_pre_i, d_pre_h, direct) = self
            .host_gru_backward(
                host_slice(&g_c, OP)?,
                host_slice(&q_c, OP)?,
                host_slice(&hp_c, OP)?,
                host_slice(&dh_c, OP)?,
                hidden,
            )
            .map_err(to_backend_error)?;
        Ok((
            tensor(d_pre_i, &[b_dim, gate_width])?,
            tensor(d_pre_h, &[b_dim, gate_width])?,
            tensor(direct, &[b_dim, hidden])?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::GfxTarget;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};

    fn ops() -> RocmBackendOps {
        RocmBackendOps::with_host_model(GfxTarget::Gfx942)
    }

    /// 決定的データ。7 要素に 1 つ大きな負値（-50）を混ぜ、シグモイドの負側分岐を通す。
    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                if i % 7 == 3 {
                    return -50.0;
                }
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 2001) as f32
                    / 250.0
                    - 4.0
            })
            .collect()
    }

    fn t(shape: &[usize], seed: u32) -> Tensor<f32> {
        Tensor::new(data(shape.iter().product(), seed), shape).unwrap()
    }

    /// REQ-2 複合判定（許容誤差は不変）に加え、bit 一致も確認する。
    fn check(name: &str, got: &Tensor<f32>, want: &Tensor<f32>) {
        assert_eq!(got.shape(), want.shape(), "{name}");
        let g = got.contiguous();
        let w = want.contiguous();
        let (g, w) = (g.as_slice().unwrap(), w.as_slice().unwrap());
        assert_parity(name, g, w);
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(g), bits(w), "{name}: bit 一致");
    }

    const SHAPES: [(usize, usize); 3] = [(1, 1), (3, 5), (2, 300)];

    #[test]
    fn lstm_pointwise_matches_cpu() {
        for (b, h) in SHAPES {
            let (pre, c_prev) = (t(&[b, 4 * h], 1), t(&[b, h], 2));
            let got = ops().lstm_pointwise(&pre, &c_prev).unwrap();
            let want = CpuBackendOps.lstm_pointwise(&pre, &c_prev).unwrap();
            check("lstm gates", &got.gates, &want.gates);
            check("lstm c", &got.c, &want.c);
            check("lstm h", &got.h, &want.h);
        }
    }

    #[test]
    fn lstm_backward_pair_matches_cpu() {
        for (b, h) in SHAPES {
            let (c, o, dh) = (t(&[b, h], 3), t(&[b, h], 4), t(&[b, h], 5));
            let got = ops().lstm_hidden_backward(&c, &o, &dh).unwrap();
            let want = CpuBackendOps.lstm_hidden_backward(&c, &o, &dh).unwrap();
            check("d_pre_o", &got.0, &want.0);
            check("dc", &got.1, &want.1);

            let (g, cp, dc) = (t(&[b, 3 * h], 6), t(&[b, h], 7), t(&[b, h], 8));
            let got = ops().lstm_cell_backward(&g, &cp, &dc).unwrap();
            let want = CpuBackendOps.lstm_cell_backward(&g, &cp, &dc).unwrap();
            check("d_pre_ifg", &got.0, &want.0);
            check("dc_prev", &got.1, &want.1);
        }
    }

    #[test]
    fn gru_pointwise_and_backward_match_cpu() {
        for (b, h) in SHAPES {
            let (pi, ph, hp) = (t(&[b, 3 * h], 9), t(&[b, 3 * h], 10), t(&[b, h], 11));
            let got = ops().gru_pointwise(&pi, &ph, &hp).unwrap();
            let want = CpuBackendOps.gru_pointwise(&pi, &ph, &hp).unwrap();
            check("gru gates", &got.gates, &want.gates);
            check("gru q", &got.q, &want.q);
            check("gru h", &got.h, &want.h);

            let (g, q, dh) = (t(&[b, 3 * h], 12), t(&[b, h], 13), t(&[b, h], 14));
            let got = ops().gru_backward(&g, &q, &hp, &dh).unwrap();
            let want = CpuBackendOps.gru_backward(&g, &q, &hp, &dh).unwrap();
            check("d_pre_i", &got.0, &want.0);
            check("d_pre_h", &got.1, &want.1);
            check("dh_prev_direct", &got.2, &want.2);
        }
    }

    #[test]
    fn non_contiguous_inputs_are_accepted() {
        // [4h, b] を転置した [b, 4h]（非 contiguous）。
        let (b, h) = (3, 5);
        let pre = t(&[4 * h, b], 21).transpose_2d().unwrap();
        let c_prev = t(&[b, h], 22);
        let got = ops().lstm_pointwise(&pre, &c_prev).unwrap();
        let want = CpuBackendOps.lstm_pointwise(&pre, &c_prev).unwrap();
        check("lstm gates (strided)", &got.gates, &want.gates);
        check("lstm h (strided)", &got.h, &want.h);
    }

    #[test]
    fn shape_errors_are_shape_mismatch() {
        let o = ops();
        let c_prev = t(&[2, 3], 1);
        // rank != 2。
        assert!(matches!(
            o.lstm_pointwise(&t(&[8, 3, 1], 1), &t(&[24], 1)),
            Err(BackendError::ShapeMismatch(_))
        ));
        // pre の列幅が 4H でない。
        assert!(matches!(
            o.lstm_pointwise(&t(&[2, 11], 1), &c_prev),
            Err(BackendError::ShapeMismatch(_))
        ));
        // q／dh の形状不一致。
        let (g, hp) = (t(&[2, 9], 1), t(&[2, 3], 2));
        assert!(matches!(
            o.gru_backward(&g, &t(&[2, 4], 1), &hp, &hp),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.gru_backward(&g, &hp, &hp, &t(&[3, 3], 1)),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.lstm_hidden_backward(&hp, &hp, &t(&[2, 4], 1)),
            Err(BackendError::ShapeMismatch(_))
        ));
        assert!(matches!(
            o.lstm_cell_backward(&t(&[2, 8], 1), &hp, &hp),
            Err(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn empty_batch_and_zero_hidden_follow_cpu() {
        let o = ops();
        // B = 0。
        let (pre, c_prev) = (t(&[0, 12], 1), t(&[0, 3], 1));
        let got = o.lstm_pointwise(&pre, &c_prev).unwrap();
        let want = CpuBackendOps.lstm_pointwise(&pre, &c_prev).unwrap();
        assert_eq!(got.gates.shape(), want.gates.shape());
        assert_eq!(got.h.shape(), want.h.shape());
        // hidden = 0 は CPU と同じ結果種別（Ok／Err のどちらでも一致する）。
        let (pre0, c0) = (t(&[2, 0], 1), t(&[2, 0], 1));
        assert_eq!(
            o.lstm_pointwise(&pre0, &c0).is_ok(),
            CpuBackendOps.lstm_pointwise(&pre0, &c0).is_ok()
        );
        let (g0, hp0) = (t(&[2, 0], 1), t(&[2, 0], 1));
        assert_eq!(
            o.gru_backward(&g0, &hp0, &hp0, &hp0).is_ok(),
            CpuBackendOps.gru_backward(&g0, &hp0, &hp0, &hp0).is_ok()
        );
    }

    /// ROCm 非搭載の HIP engine は形状検査の後で `DeviceUnavailable`（`Unsupported` ではない）。
    /// 形状不正は `DeviceUnavailable` ではなく `ShapeMismatch`。
    #[test]
    fn rnn_cell_without_rocm_is_unavailable_after_validation() {
        if crate::is_available() {
            return;
        }
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let (pre, c_prev) = (t(&[2, 12], 1), t(&[2, 3], 2));
        assert!(matches!(
            o.lstm_pointwise(&pre, &c_prev),
            Err(BackendError::DeviceUnavailable(_))
        ));
        let (g, hp) = (t(&[2, 9], 1), t(&[2, 3], 2));
        assert!(matches!(
            o.lstm_hidden_backward(&hp, &hp, &hp),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.lstm_cell_backward(&g, &hp, &hp),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.gru_pointwise(&g, &g, &hp),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.gru_backward(&g, &hp, &hp, &hp),
            Err(BackendError::DeviceUnavailable(_))
        ));
        assert!(matches!(
            o.lstm_pointwise(&t(&[2, 11], 1), &c_prev),
            Err(BackendError::ShapeMismatch(_))
        ));
    }
}
