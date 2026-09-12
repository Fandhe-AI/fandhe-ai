//! RNN／LSTM／GRU セル演算の CPU カーネル（イシュー #1647・親 #1619 →
//! Phase 3 #1573 → ルート #1570）。
//!
//! `fandhe_ai_tensor_core::BackendOps::{lstm_pointwise,
//! lstm_hidden_backward, lstm_cell_backward, gru_pointwise,
//! gru_backward}`（既定 `Unsupported`）の CPU 実装本体。RNN（tanh 版）は
//! 専用カーネルを持たない（`fandhe_ai_autodiff::var::rnn_cell_forward_value`
//! が既存の `gemm_bias_act`／`add`／`tanh` の合成で閉じるため。設計
//! `docs/autodiff-rnn-cell-tape-design.md` 決定 1）。
//!
//! `ops.rs::CpuBackendOps` から `contiguous()`→`as_slice()`→本モジュール
//! の関数呼び出し→`Tensor::new` という薄い委譲層を経由して呼ばれる
//! （`mse.rs` と同型の構成方針。モジュール冒頭コメント参照）。
//!
//! # ゲート配置（決定 5。PyTorch 準拠）
//!
//! LSTM: `pre`／`gates_ifg`／`d_pre_*` はいずれも列ブロック順
//! `i,f,g,o`（各ブロック幅 `hidden`）。GRU: `r,z,n`。
//!
//! # 並列化・決定性
//!
//! 各演算はバッチ行ごとに独立（要素間の縮約を持たない pointwise 演算）
//! のため、rayon `par_chunks_mut`（行単位）で並列化する。順序に依存
//! しない map 演算であり、`elementwise.rs` の `par_iter_mut` と同じ
//! 決定性契約（スレッド数に依らず bit 決定的）を満たす。
//!
//! # FMA 契約
//!
//! 積が和へ流れる箇所（`c = f*c_prev + i*g`・`n = tanh(r*q + pre_i_n)`・
//! `h_gru = z*h_prev + (1-z)*n`）は `f32::mul_add` を用いる（CPU 参照
//! 実装の FMA 契約統一。`.claude/rules/coding-rust.md`）。数式の正は
//! `fandhe_ai_autodiff::eval::{lstm_pointwise, lstm_hidden_backward,
//! lstm_cell_backward, gru_pointwise, gru_backward}`（ホスト参照実装。
//! 同じ数式・同じ FMA 契約）。

use fandhe_ai_tensor_core::{BackendError, ShapeError};
use rayon::prelude::*;

/// 3 テンソル分の `Vec<f32>` を返す関数の戻り値型（`clippy::
/// type_complexity` 回避）。`lstm_pointwise`／`gru_pointwise`／
/// `gru_backward` が共有する。
type TripleVecOutput = (Vec<f32>, Vec<f32>, Vec<f32>);

/// 数値安定形のシグモイド（`fandhe_ai_autodiff::eval::sigmoid_scalar`
/// と同じ 2 分岐形。大きな負値入力での `exp` オーバーフローを回避）。
fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `numel` が `hidden * gates` の倍数であり、かつ `hidden > 0` である
/// ことを検証し、バッチサイズ `B = numel / (hidden * gates)` を返す。
fn derive_batch(numel: usize, hidden: usize, gates: usize) -> Result<usize, BackendError> {
    if hidden == 0 {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: 1,
                actual: 0,
            },
        ));
    }
    // 本番経路 panic 禁止（AGENTS.md）: `hidden * gates` は
    // `checked_mul` で検証する。overflow を放置すると（例:
    // `hidden = 1usize << 62`・`gates = 4` で `row` が 0 へ wrap
    // する）`row == 0` のまま以下の `numel % row`／`numel / row`
    // へ進みゼロ除算 panic を起こす（イシュー #1647 codex-review P1
    // 指摘）。`row == 0` は `hidden > 0` を検査済みのため overflow
    // 以外では起こり得ず、overflow 検査で自動的に排除される。
    let row = hidden
        .checked_mul(gates)
        .ok_or(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: usize::MAX,
                actual: 0,
            },
        ))?;
    if row == 0 || !numel.is_multiple_of(row) {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: row,
                actual: numel.checked_rem(row).unwrap_or(numel),
            },
        ));
    }
    Ok(numel / row)
}

/// 2 スライスの長さが `expected` に一致することを検証する
/// （`mse.rs::validate_mse_len` と同型）。
fn require_len(actual: usize, expected: usize) -> Result<(), BackendError> {
    if actual != expected {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch { expected, actual },
        ));
    }
    Ok(())
}

/// LSTM セルの pointwise 段（決定 1・1b）。`pre: [B, 4H]`（列ブロック順
/// `i,f,g,o`）・`c_prev: [B, H]` から `gates`（活性化後 `i,f,g,o`。
/// `[B, 4H]`）・`c`（`[B, H]`）・`h`（`[B, H]`）を計算する。
pub(crate) fn lstm_pointwise(
    pre: &[f32],
    c_prev: &[f32],
    hidden: usize,
) -> Result<TripleVecOutput, BackendError> {
    let b_dim = derive_batch(pre.len(), hidden, 4)?;
    require_len(c_prev.len(), b_dim * hidden)?;

    let mut gates = vec![0f32; b_dim * 4 * hidden];
    let mut c_out = vec![0f32; b_dim * hidden];
    let mut h_out = vec![0f32; b_dim * hidden];

    gates
        .par_chunks_mut(4 * hidden)
        .zip(c_out.par_chunks_mut(hidden))
        .zip(h_out.par_chunks_mut(hidden))
        .zip(pre.par_chunks(4 * hidden))
        .zip(c_prev.par_chunks(hidden))
        .for_each(|((((gate_row, c_row), h_row), pre_row), c_prev_row)| {
            for j in 0..hidden {
                let i_val = sigmoid_scalar(pre_row[j]);
                let f_val = sigmoid_scalar(pre_row[hidden + j]);
                let g_val = pre_row[2 * hidden + j].tanh();
                let o_val = sigmoid_scalar(pre_row[3 * hidden + j]);

                gate_row[j] = i_val;
                gate_row[hidden + j] = f_val;
                gate_row[2 * hidden + j] = g_val;
                gate_row[3 * hidden + j] = o_val;

                let c_val = f_val.mul_add(c_prev_row[j], i_val * g_val);
                c_row[j] = c_val;
                h_row[j] = o_val * c_val.tanh();
            }
        });

    Ok((gates, c_out, h_out))
}

/// [`crate::ops::CpuBackendOps::lstm_hidden_backward`] 本体（決定 1b・
/// 1b 追記）。`c`・`gate_o`・`dh` はいずれも `[B, H]`。
/// `d_pre_o = dh*tanh(c)*o*(1-o)`、`dc = dh*o*(1-tanh(c)^2)`。
pub(crate) fn lstm_hidden_backward(
    c: &[f32],
    gate_o: &[f32],
    dh: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
    require_len(gate_o.len(), c.len())?;
    require_len(dh.len(), c.len())?;

    let mut d_pre_o = vec![0f32; c.len()];
    let mut dc = vec![0f32; c.len()];
    d_pre_o
        .par_iter_mut()
        .zip(dc.par_iter_mut())
        .zip(c.par_iter())
        .zip(gate_o.par_iter())
        .zip(dh.par_iter())
        .for_each(|((((d_pre_o, dc), &c_val), &o_val), &dh_val)| {
            let tanh_c = c_val.tanh();
            *d_pre_o = dh_val * tanh_c * o_val * (1.0 - o_val);
            *dc = dh_val * o_val * (1.0 - tanh_c * tanh_c);
        });
    Ok((d_pre_o, dc))
}

/// [`crate::ops::CpuBackendOps::lstm_cell_backward`] 本体（決定 1b）。
/// `gates_ifg: [B, 3H]`（活性化後 `i,f,g`）・`c_prev: [B, H]`・
/// `dc: [B, H]` から `d_pre_ifg: [B, 3H]`・`dc_prev: [B, H]` を計算する。
pub(crate) fn lstm_cell_backward(
    gates_ifg: &[f32],
    c_prev: &[f32],
    dc: &[f32],
    hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
    let b_dim = derive_batch(gates_ifg.len(), hidden, 3)?;
    require_len(c_prev.len(), b_dim * hidden)?;
    require_len(dc.len(), b_dim * hidden)?;

    let mut d_pre_ifg = vec![0f32; b_dim * 3 * hidden];
    let mut dc_prev = vec![0f32; b_dim * hidden];

    d_pre_ifg
        .par_chunks_mut(3 * hidden)
        .zip(dc_prev.par_chunks_mut(hidden))
        .zip(gates_ifg.par_chunks(3 * hidden))
        .zip(c_prev.par_chunks(hidden))
        .zip(dc.par_chunks(hidden))
        .for_each(
            |((((d_pre_row, dc_prev_row), gate_row), c_prev_row), dc_row)| {
                for j in 0..hidden {
                    let i_val = gate_row[j];
                    let f_val = gate_row[hidden + j];
                    let g_val = gate_row[2 * hidden + j];
                    let dc_val = dc_row[j];

                    d_pre_row[j] = dc_val * g_val * i_val * (1.0 - i_val);
                    d_pre_row[hidden + j] = dc_val * c_prev_row[j] * f_val * (1.0 - f_val);
                    d_pre_row[2 * hidden + j] = dc_val * i_val * (1.0 - g_val * g_val);
                    dc_prev_row[j] = dc_val * f_val;
                }
            },
        );

    Ok((d_pre_ifg, dc_prev))
}

/// GRU セルの pointwise 段（決定 1c・5。`reset_after=True` 規約）。
/// `pre_i`／`pre_h: [B, 3H]`（列ブロック順 `r,z,n`）・`h_prev: [B, H]`
/// から `gates`（活性化後 `r,z,n`。`[B, 3H]`）・`q`（`pre_h` の n 列
/// ブロック。`[B, H]`）・`h`（`[B, H]`）を計算する。
pub(crate) fn gru_pointwise(
    pre_i: &[f32],
    pre_h: &[f32],
    h_prev: &[f32],
    hidden: usize,
) -> Result<TripleVecOutput, BackendError> {
    let b_dim = derive_batch(pre_i.len(), hidden, 3)?;
    require_len(pre_h.len(), b_dim * 3 * hidden)?;
    require_len(h_prev.len(), b_dim * hidden)?;

    let mut gates = vec![0f32; b_dim * 3 * hidden];
    let mut q_out = vec![0f32; b_dim * hidden];
    let mut h_out = vec![0f32; b_dim * hidden];

    gates
        .par_chunks_mut(3 * hidden)
        .zip(q_out.par_chunks_mut(hidden))
        .zip(h_out.par_chunks_mut(hidden))
        .zip(pre_i.par_chunks(3 * hidden))
        .zip(pre_h.par_chunks(3 * hidden))
        .zip(h_prev.par_chunks(hidden))
        .for_each(
            |(((((gate_row, q_row), h_row), pre_i_row), pre_h_row), h_prev_row)| {
                for j in 0..hidden {
                    let r_pre = pre_i_row[j] + pre_h_row[j];
                    let z_pre = pre_i_row[hidden + j] + pre_h_row[hidden + j];
                    let q_val = pre_h_row[2 * hidden + j];
                    let pre_i_n = pre_i_row[2 * hidden + j];

                    let r_val = sigmoid_scalar(r_pre);
                    let z_val = sigmoid_scalar(z_pre);
                    let n_val = r_val.mul_add(q_val, pre_i_n).tanh();

                    gate_row[j] = r_val;
                    gate_row[hidden + j] = z_val;
                    gate_row[2 * hidden + j] = n_val;
                    q_row[j] = q_val;

                    h_row[j] = z_val.mul_add(h_prev_row[j], (1.0 - z_val) * n_val);
                }
            },
        );

    Ok((gates, q_out, h_out))
}

/// [`crate::ops::CpuBackendOps::gru_backward`] 本体。`gates_rzn:
/// [B, 3H]`（活性化後 `r,z,n`）・`q: [B, H]`（決定 1c）・`h_prev:
/// [B, H]`・`dh: [B, H]` から `d_pre_i: [B, 3H]`・`d_pre_h: [B, 3H]`・
/// `dh_prev_direct: [B, H]` を計算する。
pub(crate) fn gru_backward(
    gates_rzn: &[f32],
    q: &[f32],
    h_prev: &[f32],
    dh: &[f32],
    hidden: usize,
) -> Result<TripleVecOutput, BackendError> {
    let b_dim = derive_batch(gates_rzn.len(), hidden, 3)?;
    require_len(q.len(), b_dim * hidden)?;
    require_len(h_prev.len(), b_dim * hidden)?;
    require_len(dh.len(), b_dim * hidden)?;

    let mut d_pre_i = vec![0f32; b_dim * 3 * hidden];
    let mut d_pre_h = vec![0f32; b_dim * 3 * hidden];
    let mut dh_prev_direct = vec![0f32; b_dim * hidden];

    // 7 系統の同時 `zip`（`d_pre_i`／`d_pre_h`／`dh_prev_direct`／
    // `gates_rzn`／`q`／`h_prev`／`dh`）は入れ子タプルパターンが深く
    // なり可読性を損なうため、出力 2 系統（`d_pre_i`／`d_pre_h`。同じ
    // `3*hidden` 幅で行対応）だけを `zip` して並列化し、残り 5 系統
    // （`dh_prev_direct`・4 入力）は行番号 `row` から明示的にスライス
    // する（`par_chunks_mut` が返す行の順序は発生順と一致するため、
    // `enumerate()` の `row` はそのままバッチ添字として使える）。
    d_pre_i
        .par_chunks_mut(3 * hidden)
        .zip(d_pre_h.par_chunks_mut(3 * hidden))
        .enumerate()
        .for_each(|(row, (d_pre_i_row, d_pre_h_row))| {
            let gate_row = &gates_rzn[row * 3 * hidden..(row + 1) * 3 * hidden];
            let q_row = &q[row * hidden..(row + 1) * hidden];
            let h_prev_row = &h_prev[row * hidden..(row + 1) * hidden];
            let dh_row = &dh[row * hidden..(row + 1) * hidden];

            for j in 0..hidden {
                let r_val = gate_row[j];
                let z_val = gate_row[hidden + j];
                let n_val = gate_row[2 * hidden + j];
                let q_val = q_row[j];
                let h_prev_val = h_prev_row[j];
                let dh_val = dh_row[j];

                let dn = dh_val * (1.0 - z_val);
                let dz = dh_val * (h_prev_val - n_val);
                let d_pre_n = dn * (1.0 - n_val * n_val);
                let dr = d_pre_n * q_val;
                let d_pre_r = dr * r_val * (1.0 - r_val);
                let d_pre_z = dz * z_val * (1.0 - z_val);

                d_pre_i_row[j] = d_pre_r;
                d_pre_i_row[hidden + j] = d_pre_z;
                d_pre_i_row[2 * hidden + j] = d_pre_n;

                d_pre_h_row[j] = d_pre_r;
                d_pre_h_row[hidden + j] = d_pre_z;
                d_pre_h_row[2 * hidden + j] = d_pre_n * r_val;
            }
        });

    dh_prev_direct
        .par_chunks_mut(hidden)
        .zip(gates_rzn.par_chunks(3 * hidden))
        .zip(dh.par_chunks(hidden))
        .for_each(|((out_row, gate_row), dh_row)| {
            for j in 0..hidden {
                out_row[j] = dh_row[j] * gate_row[hidden + j];
            }
        });

    Ok((d_pre_i, d_pre_h, dh_prev_direct))
}
