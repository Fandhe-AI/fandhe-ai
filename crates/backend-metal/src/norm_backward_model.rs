//! `shaders/norm_backward.metal`（RMSNorm／LayerNorm backward）の
//! **ホスト側逐語モデル**（イシュー #1953・親 #1947）。`cfg` を付けない
//! （`crate::soft_f64`・`crate::row_kernel` と同じ理由で Linux 実行可能
//! な単体テスト対象とする）。
//!
//! # 数値契約（REQ-2。bit 完全一致ではない）
//!
//! `docs/norm-ops-design.md`・`shaders/layer_norm.metal` 冒頭コメントが
//! 確立した契約（forward の `xhat` は CPU 参照実装と bit 一致しない・
//! REQ-2 統一複合判定〈相対誤差 1e-3 未満 または絶対誤差 1e-5 未満〉の
//! 範囲で一致させる）と同じ方針を backward にも適用する。イシュー
//! #1950（CUDA 版・兄弟 PR）の `kernels_norm_backward.rs`／
//! `tests/norm_backward_parity.rs` が確立した先例（dx／dw／db いずれも
//! `assert_parity`〈REQ-2〉で検証し bit 完全一致は主張しない）を踏襲する
//! （実装計画が当初想定した「dw／db は bit 完全一致」contract は、この
//! 先例確認の結果採用しない。理由: 正しく丸めた `f64` FMA
//! 〈`a*b+c` 全体を単一丸めで求める一般演算〉が必要になり実装コストが
//! 大きい一方、既存の縮約契約〈`.claude/rules/coding-rust.md`「要素積を
//! `f32` で確定してから `f64` へ昇格して蓄積する」〉と REQ-2 判定は
//! CUDA 側と一貫しており、bit 完全一致を主張する追加の価値がない）。
//!
//! # 演算列
//!
//! [`crate::soft_f64`] の `add_f64_bits`／`sub_f64_bits`／`mul_f64_bits`／
//! `div_f64_bits`／`widen_f32_bits`／`narrow_f64_bits`／
//! `rsqrt_newton_f64_bits` のみで構成し、`shaders/norm_backward.metal` の
//! `nb_f64_*` 系関数と 1 対 1 対応する（本ファイルはその Rust 側複製）。
//! 行内の縮約順序（レーンストライド 32 分担 → offset 16→1 butterfly）は
//! `fandhe_ai_autodiff::eval::warp_reduce_f64`・CUDA
//! `kernels_norm_backward.rs` と同一に揃える（イシュー #1950 の設計を
//! Metal simdgroup（32 レーン）版として再現）。

use crate::soft_f64::{
    add_f64_bits, div_f64_bits, mul_f64_bits, narrow_f64_bits, rsqrt_newton_f64_bits, sub_f64_bits,
    widen_f32_bits,
};

const LANES: usize = 32;

/// `simd_shuffle_xor` によるレーンストライド + butterfly 縮約
/// （`fandhe_ai_autodiff::eval::warp_reduce_f64` の Rust 側複製・
/// `nb_f64_warp_reduce` の MSL 側対応物と 1 対 1）。
fn warp_reduce_f64(hidden: usize, mut contribute: impl FnMut(usize, u64) -> u64) -> u64 {
    let mut lanes = [0u64; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut idx = lane;
        while idx < hidden {
            *slot = contribute(idx, *slot);
            idx += LANES;
        }
    }
    let mut offset = 16usize;
    while offset > 0 {
        let snapshot = lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = add_f64_bits(snapshot[lane], snapshot[lane ^ offset]);
        }
        offset >>= 1;
    }
    lanes[0]
}

/// [`rmsnorm_backward_rows`]／[`layer_norm_backward_rows`] の入力検査
/// エラー（`crate::batch_norm_model::BatchNormPrepareError`・
/// `crate::im2col_model::Im2colPrepareError` と同型の設計判断）。
///
/// 本モジュールは `pub mod`（クレート外から到達可能）である一方、
/// `rows`／`hidden` と `x`／`dy`／`w` の長さの整合はスライス添字アクセス
/// より前に検証しなければ境界外 panic が呼び出し側へ漏れる
/// （`.claude/rules/coding-rust.md`「本番経路で `unwrap()` / `expect()`
/// を使わない」・本番経路 panic 禁止方針。`crate::batch_norm_model::
/// validate_batch_norm_launch`〈PR #1881 codex-review 指摘〉と同じ
/// 対策）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormBackwardPrepareError {
    /// `rows*hidden` が overflow する・`x.len()`／`dy.len()` と
    /// 一致しない・`w` が `Some` の場合に `w.len() != hidden`。
    InvalidShape { detail: String },
}

impl std::fmt::Display for NormBackwardPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NormBackwardPrepareError::InvalidShape { detail } => {
                write!(f, "norm_backward invalid shape: {detail}")
            }
        }
    }
}

impl std::error::Error for NormBackwardPrepareError {}

/// [`layer_norm_backward_rows`] の戻り値型エイリアス。`(dx, dw, db)`
/// （`fandhe_ai_tensor_core::LayerNormBackwardOutput` と同じ
/// `clippy::type_complexity` 回避のための命名）。
pub type LayerNormBackwardRowsOutput = (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>);

/// 起動前 fail-closed 検証: `rows.checked_mul(hidden)` が
/// `x.len()`／`dy.len()` と一致し、`w` が `Some` のときは
/// `w.len() == hidden` であることを確認する（`crate::
/// batch_norm_model::validate_batch_norm_launch` と同型）。
fn validate_norm_backward_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    dy_len: usize,
    w_len: Option<usize>,
) -> Result<(), NormBackwardPrepareError> {
    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| NormBackwardPrepareError::InvalidShape {
            detail: format!("rows*hidden overflowed usize: rows={rows}, hidden={hidden}"),
        })?;
    if numel != x_len {
        return Err(NormBackwardPrepareError::InvalidShape {
            detail: format!("x length mismatch: rows*hidden={numel}, x.len()={x_len}"),
        });
    }
    if numel != dy_len {
        return Err(NormBackwardPrepareError::InvalidShape {
            detail: format!("dy length mismatch: rows*hidden={numel}, dy.len()={dy_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(NormBackwardPrepareError::InvalidShape {
            detail: format!("weight length mismatch: hidden={hidden}, w.len()={wl}"),
        });
    }
    Ok(())
}

/// RMSNorm 行内統計（`rstd`。`rmsnorm_bwd_dx_f32` パス 1 の逐語モデル）。
/// CPU 参照実装 `row_rms_stats` と異なりレーンストライド butterfly で
/// 二乗和を求める（forward `rmsnorm.metal` と同じ GPU 側都合。REQ-2
/// 範囲での一致に留まる）。
pub fn rms_row_rstd(row: &[f32], eps: f32) -> f32 {
    let hidden = row.len();
    let hidden_f64 = widen_f32_bits((hidden as f32).to_bits());
    let sq = warp_reduce_f64(hidden, |idx, acc| {
        let xv = widen_f32_bits(row[idx].to_bits());
        add_f64_bits(acc, mul_f64_bits(xv, xv))
    });
    let mean_sq = div_f64_bits(sq, hidden_f64);
    let ve = add_f64_bits(mean_sq, widen_f32_bits(eps.to_bits()));
    f32::from_bits(narrow_f64_bits(rsqrt_newton_f64_bits(ve)))
}

/// RMSNorm backward（`dx`・`dw`）の完全な行走査版ホストモデル
/// （`MetalNormBackward::run_rmsnorm_backward_f32` の逐語モデル。`rows`
/// 行すべてを走査する。`w` は `None` なら `dxhat = dy`）。
///
/// 入口で [`validate_norm_backward_launch`] を呼び `rows`／`hidden`・
/// `x.len()`／`dy.len()`／`w.len()` の不整合を型付き `Result` で拒否
/// してから本体処理へ入る（本番経路で panic させない方針。
/// `.claude/rules/coding-rust.md`）。
pub fn rmsnorm_backward_rows(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> Result<(Vec<f32>, Option<Vec<f32>>), NormBackwardPrepareError> {
    validate_norm_backward_launch(rows, hidden, x.len(), dy.len(), w.map(<[f32]>::len))?;
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Vec<u64> = vec![0u64; hidden]; // +0.0（f64 bits）。
    if rows == 0 || hidden == 0 {
        let dw = w.map(|_| dw_acc.into_iter().map(narrow_f64_bits_f32).collect());
        return Ok((dx, dw));
    }
    let hidden_f64 = widen_f32_bits((hidden as f32).to_bits());
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let rstd = rms_row_rstd(row, eps);
        let dxhat_at = |i: usize| -> f32 {
            let wv = w.map_or(1.0f32, |w| w[i]);
            dy_row[i] * wv
        };
        let dot = warp_reduce_f64(hidden, |i, acc| {
            let xhat = row[i] * rstd;
            let term = dxhat_at(i) * xhat;
            add_f64_bits(acc, widen_f32_bits(term.to_bits()))
        });
        let mean_dot = div_f64_bits(dot, hidden_f64);
        let rstd64 = widen_f32_bits(rstd.to_bits());
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, (&xv, dxv)) in row.iter().zip(dx_row.iter_mut()).enumerate() {
            let xhat = xv * rstd;
            let dxhat = dxhat_at(i);
            let term2 = mul_f64_bits(widen_f32_bits(xhat.to_bits()), mean_dot);
            let inner = sub_f64_bits(widen_f32_bits(dxhat.to_bits()), term2);
            let d64 = mul_f64_bits(rstd64, inner);
            *dxv = f32::from_bits(narrow_f64_bits(d64));
        }
        for (i, (&xv, &dyv)) in row.iter().zip(dy_row.iter()).enumerate() {
            let xhat = xv * rstd;
            let term = dyv * xhat;
            dw_acc[i] = add_f64_bits(dw_acc[i], widen_f32_bits(term.to_bits()));
        }
    }
    let dw = w.map(|_| dw_acc.into_iter().map(narrow_f64_bits_f32).collect());
    Ok((dx, dw))
}

fn narrow_f64_bits_f32(bits: u64) -> f32 {
    f32::from_bits(narrow_f64_bits(bits))
}

/// LayerNorm 行内統計（`mean`・`rstd`。`f64` bit パターンのまま返す。
/// `layer_norm_bwd_dx_f32` パス 1／2 の逐語モデル）。
pub fn ln_row_mean_rstd(row: &[f32], eps: f32) -> (u64, u64) {
    let hidden = row.len();
    let hidden_f64 = widen_f32_bits((hidden as f32).to_bits());
    let sum = warp_reduce_f64(hidden, |idx, acc| {
        add_f64_bits(acc, widen_f32_bits(row[idx].to_bits()))
    });
    let mean = div_f64_bits(sum, hidden_f64);
    let sq = warp_reduce_f64(hidden, |idx, acc| {
        let dev = sub_f64_bits(widen_f32_bits(row[idx].to_bits()), mean);
        add_f64_bits(acc, mul_f64_bits(dev, dev))
    });
    let var = div_f64_bits(sq, hidden_f64);
    let ve = add_f64_bits(var, widen_f32_bits(eps.to_bits()));
    let rstd = rsqrt_newton_f64_bits(ve);
    (mean, rstd)
}

/// LayerNorm backward（`dx`・`dw`・`db`）の完全な行走査版ホストモデル
/// （`MetalNormBackward::run_layer_norm_backward_f32` の逐語モデル）。
///
/// 入口で [`validate_norm_backward_launch`] を呼び `rows`／`hidden`・
/// `x.len()`／`dy.len()`／`w.len()` の不整合を型付き `Result` で拒否
/// してから本体処理へ入る（[`rmsnorm_backward_rows`] と同じ理由）。
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward_rows(
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> Result<LayerNormBackwardRowsOutput, NormBackwardPrepareError> {
    validate_norm_backward_launch(rows, hidden, x.len(), dy.len(), w.map(<[f32]>::len))?;
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Vec<u64> = vec![0u64; hidden];
    let mut db_acc: Vec<u64> = vec![0u64; hidden];
    if rows == 0 || hidden == 0 {
        let dw = w.map(|_| dw_acc.into_iter().map(narrow_f64_bits_f32).collect());
        let db = has_bias.then(|| db_acc.into_iter().map(narrow_f64_bits_f32).collect());
        return Ok((dx, dw, db));
    }
    let hidden_f64 = widen_f32_bits((hidden as f32).to_bits());
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let (mean, rstd) = ln_row_mean_rstd(row, eps);
        let xhat_at = |i: usize| -> f32 {
            let dev = sub_f64_bits(widen_f32_bits(row[i].to_bits()), mean);
            f32::from_bits(narrow_f64_bits(mul_f64_bits(dev, rstd)))
        };
        let dxhat_at = |i: usize| -> f32 {
            let wv = w.map_or(1.0f32, |w| w[i]);
            dy_row[i] * wv
        };
        let sum_dxhat = warp_reduce_f64(hidden, |i, acc| {
            add_f64_bits(acc, widen_f32_bits(dxhat_at(i).to_bits()))
        });
        let dot = warp_reduce_f64(hidden, |i, acc| {
            let term = dxhat_at(i) * xhat_at(i);
            add_f64_bits(acc, widen_f32_bits(term.to_bits()))
        });
        let mean_dxhat = div_f64_bits(sum_dxhat, hidden_f64);
        let mean_dot = div_f64_bits(dot, hidden_f64);
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, dxv) in dx_row.iter_mut().enumerate() {
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            let term2 = mul_f64_bits(widen_f32_bits(xhat.to_bits()), mean_dot);
            let inner = sub_f64_bits(
                sub_f64_bits(widen_f32_bits(dxhat.to_bits()), mean_dxhat),
                term2,
            );
            let d64 = mul_f64_bits(rstd, inner);
            *dxv = f32::from_bits(narrow_f64_bits(d64));
        }
        for (i, &dyv) in dy_row.iter().enumerate() {
            let xhat = xhat_at(i);
            let term = dyv * xhat;
            dw_acc[i] = add_f64_bits(dw_acc[i], widen_f32_bits(term.to_bits()));
            db_acc[i] = add_f64_bits(db_acc[i], widen_f32_bits(dyv.to_bits()));
        }
    }
    let dw = w.map(|_| dw_acc.into_iter().map(narrow_f64_bits_f32).collect());
    let db = has_bias.then(|| db_acc.into_iter().map(narrow_f64_bits_f32).collect());
    Ok((dx, dw, db))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_cpu::parity::assert_parity;

    /// `fandhe_ai_autodiff::eval::warp_reduce_f64`（private）のテスト
    /// 専用複製（プレーン `f64` 版。モジュール本体の [`warp_reduce_f64`]
    /// は `crate::soft_f64` bit 表現版のため名前を分ける）。GPU の 32
    /// レーン + butterfly 縮約順序を再現する（`crates/backend-metal/
    /// tests/norm_backward_parity.rs`・`fandhe_ai_autodiff::eval::
    /// row_ln_stats`／`rmsnorm_vjp_rows`／`layer_norm_vjp_rows` が
    /// 使う縮約順序と同一。単純な先頭からの逐次和は相殺入力で GPU 側と
    /// 乖離しうるため使わない）。
    fn cpu_warp_reduce_f64(hidden: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
        const LANES: usize = 32;
        let mut lanes = [0.0f64; LANES];
        for (lane, slot) in lanes.iter_mut().enumerate() {
            let mut idx = lane;
            while idx < hidden {
                *slot = contribute(idx, *slot);
                idx += LANES;
            }
        }
        let mut offset = 16usize;
        while offset > 0 {
            let snapshot = lanes;
            for (lane, slot) in lanes.iter_mut().enumerate() {
                *slot = snapshot[lane] + snapshot[lane ^ offset];
            }
            offset >>= 1;
        }
        lanes[0]
    }

    /// CPU ホスト参照実装（`fandhe_ai_autodiff::eval::row_rms_stats`。
    /// private のため同一アルゴリズムをテスト内に複製する。二乗和は
    /// 先頭からの逐次 `mul_add` 蓄積のまま（`row_rms_stats` 自身が
    /// butterfly ではなくこの順序を採用しているため一致させる）。
    fn cpu_row_rms_stats(row: &[f32], eps: f32) -> f32 {
        let mut acc = 0.0f64;
        for &v in row {
            let v = v as f64;
            acc = v.mul_add(v, acc);
        }
        (1.0f64 / acc.mul_add(1.0 / row.len() as f64, eps as f64).sqrt()) as f32
    }

    /// `fandhe_ai_autodiff::grad::rmsnorm_vjp_rows`（private）と同一
    /// アルゴリズムのテスト専用複製（`crates/backend-metal/tests/
    /// norm_backward_parity.rs::cpu_rmsnorm_backward_reference` と同型。
    /// `dot_acc` は [`cpu_warp_reduce_f64`] による butterfly 縮約で
    /// 求める——`rmsnorm_vjp_rows` が `eval::warp_reduce_f64` へ是正
    /// 済み〈PR #1995 codex-review P1〉であるのに合わせる）。
    fn cpu_rmsnorm_backward(
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
        dy: &[f32],
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let mut dx = vec![0.0f32; x.len()];
        let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
        let inv_n = 1.0f64 / hidden as f64;
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let dy_row = &dy[r * hidden..(r + 1) * hidden];
            let rstd = cpu_row_rms_stats(row, eps);
            let dxhat_at = |i: usize| -> f32 {
                let wv = w.map_or(1.0f32, |w| w[i]);
                dy_row[i] * wv
            };
            let dot_acc = cpu_warp_reduce_f64(hidden, |i, acc| {
                let xhat = row[i] * rstd;
                let term = dxhat_at(i) * xhat;
                acc + term as f64
            });
            let mean_dot = dot_acc * inv_n;
            let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
            for (i, (&xv, dxv)) in row.iter().zip(dx_row.iter_mut()).enumerate() {
                let xhat = xv * rstd;
                let dxhat = dxhat_at(i);
                let d = (rstd as f64) * (dxhat as f64 - (xhat as f64) * mean_dot);
                *dxv = d as f32;
            }
            if let Some(dw_acc) = dw_acc.as_mut() {
                for (i, (&xv, &dyv)) in row.iter().zip(dy_row.iter()).enumerate() {
                    let xhat = xv * rstd;
                    let term = dyv * xhat;
                    dw_acc[i] += term as f64;
                }
            }
        }
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        (dx, dw)
    }

    /// `fandhe_ai_autodiff::grad::layer_norm_vjp_rows`（private）と
    /// 同一アルゴリズムのテスト専用複製（`crates/backend-metal/tests/
    /// norm_backward_parity.rs::cpu_layer_norm_backward_reference` と
    /// 同型）。行内統計（`mean`／`var`）・`sum_dxhat`・`dot_acc` はいずれも
    /// [`cpu_warp_reduce_f64`] による butterfly 縮約で求める——
    /// `eval::row_ln_stats`／`layer_norm_vjp_rows` 自身が `eval::
    /// warp_reduce_f64` を使うのに合わせる（先頭からの逐次和は相殺
    /// 入力で乖離しうる）。
    fn cpu_layer_norm_backward(
        x: &[f32],
        w: Option<&[f32]>,
        has_bias: bool,
        eps: f32,
        rows: usize,
        hidden: usize,
        dy: &[f32],
    ) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
        let mut dx = vec![0.0f32; x.len()];
        let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
        let mut db_acc: Option<Vec<f64>> = has_bias.then(|| vec![0.0f64; hidden]);
        let n = hidden as f64;
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let dy_row = &dy[r * hidden..(r + 1) * hidden];
            let sum = cpu_warp_reduce_f64(hidden, |i, acc| acc + row[i] as f64);
            let mean = sum / n;
            let sq = cpu_warp_reduce_f64(hidden, |i, acc| {
                let d = row[i] as f64 - mean;
                d.mul_add(d, acc)
            });
            let var = sq / n;
            let rstd = 1.0f64 / (var + eps as f64).sqrt();
            let xhat_at = |i: usize| -> f32 { ((row[i] as f64 - mean) * rstd) as f32 };
            let dxhat_at = |i: usize| -> f32 {
                let wv = w.map_or(1.0f32, |w| w[i]);
                dy_row[i] * wv
            };
            let sum_dxhat = cpu_warp_reduce_f64(hidden, |i, acc| acc + dxhat_at(i) as f64);
            let dot_acc = cpu_warp_reduce_f64(hidden, |i, acc| {
                let term = dxhat_at(i) * xhat_at(i);
                acc + term as f64
            });
            let mean_dxhat = sum_dxhat / n;
            let mean_dot = dot_acc / n;
            for i in 0..hidden {
                let xhat = xhat_at(i);
                let dxhat = dxhat_at(i);
                let d = rstd * (dxhat as f64 - mean_dxhat - (xhat as f64) * mean_dot);
                dx[r * hidden + i] = d as f32;
            }
            if let Some(dw_acc) = dw_acc.as_mut() {
                for i in 0..hidden {
                    dw_acc[i] += (dy_row[i] * xhat_at(i)) as f64;
                }
            }
            if let Some(db_acc) = db_acc.as_mut() {
                for i in 0..hidden {
                    db_acc[i] += dy_row[i] as f64;
                }
            }
        }
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        (dx, dw, db)
    }

    /// 決定的シード PRNG（`bench_harness::rng::Xorshift64Star`。
    /// coding-rust.md「学習系回帰テストには決定的シード設定ユーティリ
    /// ティを使う」）で `[-amplitude, amplitude)` の値を生成する
    /// （`Xorshift64Star::next_f32` は `[-1.0, 1.0)` を返すため単純な
    /// 倍率適用でよい。旧実装のビットマスク `0x807F_FFFF` は指数部を
    /// 全消去し生成値を固定値へ丸めてしまっていた——codex-review 指摘）。
    fn gen_data(rng: &mut Xorshift64Star, n: usize, amplitude: f32) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32() * amplitude).collect()
    }

    #[test]
    fn rmsnorm_backward_matches_cpu_reference_req2() {
        for &(rows, hidden) in &[(1usize, 1usize), (2, 31), (3, 32), (5, 64), (2, 100)] {
            for has_weight in [false, true] {
                let mut rng = Xorshift64Star::new(
                    0x1234_5678_9ABC_DEF0 ^ (rows as u64) ^ ((hidden as u64) << 8),
                );
                let x = gen_data(&mut rng, rows * hidden, 2.0);
                let dy = gen_data(&mut rng, rows * hidden, 1.0);
                let w: Option<Vec<f32>> = has_weight.then(|| gen_data(&mut rng, hidden, 1.0));
                let w_slice = w.as_deref();
                let (dx_gpu, dw_gpu) = rmsnorm_backward_rows(&x, w_slice, 1e-5, rows, hidden, &dy)
                    .expect("valid shape");
                let (dx_cpu, dw_cpu) = cpu_rmsnorm_backward(&x, w_slice, 1e-5, rows, hidden, &dy);
                assert_parity("rmsnorm dx", &dx_gpu, &dx_cpu);
                match (dw_gpu, dw_cpu) {
                    (Some(a), Some(e)) => assert_parity("rmsnorm dw", &a, &e),
                    (None, None) => {}
                    _ => panic!("rmsnorm dw Some/None mismatch"),
                }
            }
        }
    }

    #[test]
    fn layer_norm_backward_matches_cpu_reference_req2() {
        for &(rows, hidden) in &[(1usize, 1usize), (2, 31), (3, 32), (5, 64), (2, 100)] {
            for has_weight in [false, true] {
                for has_bias in [false, true] {
                    let mut rng = Xorshift64Star::new(
                        0xABCD_1234_0000_0001 ^ (rows as u64) ^ ((hidden as u64) << 8),
                    );
                    let x = gen_data(&mut rng, rows * hidden, 2.0);
                    let dy = gen_data(&mut rng, rows * hidden, 1.0);
                    let w: Option<Vec<f32>> = has_weight.then(|| gen_data(&mut rng, hidden, 1.0));
                    let w_slice = w.as_deref();
                    let (dx_gpu, dw_gpu, db_gpu) =
                        layer_norm_backward_rows(&x, w_slice, has_bias, 1e-5, rows, hidden, &dy)
                            .expect("valid shape");
                    let (dx_cpu, dw_cpu, db_cpu) =
                        cpu_layer_norm_backward(&x, w_slice, has_bias, 1e-5, rows, hidden, &dy);
                    assert_parity("layer_norm dx", &dx_gpu, &dx_cpu);
                    match (dw_gpu, dw_cpu) {
                        (Some(a), Some(e)) => assert_parity("layer_norm dw", &a, &e),
                        (None, None) => {}
                        _ => panic!("layer_norm dw Some/None mismatch"),
                    }
                    match (db_gpu, db_cpu) {
                        (Some(a), Some(e)) => assert_parity("layer_norm db", &a, &e),
                        (None, None) => {}
                        _ => panic!("layer_norm db Some/None mismatch"),
                    }
                }
            }
        }
    }

    /// [`validate_norm_backward_launch`] が `rows*hidden` と
    /// `x`／`dy`／`w` の長さ不整合を境界外 panic ではなく型付き
    /// `Result::Err` で拒否することを確認する（P1 codex-review 指摘。
    /// `crate::batch_norm_model` 同型テストの踏襲）。
    #[test]
    fn rmsnorm_backward_rows_rejects_length_mismatch() {
        assert!(matches!(
            rmsnorm_backward_rows(&[1.0, 2.0], None, 1e-5, 2, 2, &[1.0, 2.0]),
            Err(NormBackwardPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            rmsnorm_backward_rows(&[1.0, 2.0], None, 1e-5, 1, 2, &[1.0]),
            Err(NormBackwardPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            rmsnorm_backward_rows(&[1.0, 2.0], Some(&[1.0]), 1e-5, 1, 2, &[1.0, 2.0]),
            Err(NormBackwardPrepareError::InvalidShape { .. })
        ));
    }

    #[test]
    fn layer_norm_backward_rows_rejects_length_mismatch() {
        assert!(matches!(
            layer_norm_backward_rows(&[1.0, 2.0], None, false, 1e-5, 2, 2, &[1.0, 2.0]),
            Err(NormBackwardPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            layer_norm_backward_rows(&[1.0, 2.0], None, false, 1e-5, 1, 2, &[1.0]),
            Err(NormBackwardPrepareError::InvalidShape { .. })
        ));
        assert!(matches!(
            layer_norm_backward_rows(&[1.0, 2.0], Some(&[1.0]), false, 1e-5, 1, 2, &[1.0, 2.0]),
            Err(NormBackwardPrepareError::InvalidShape { .. })
        ));
    }
}
