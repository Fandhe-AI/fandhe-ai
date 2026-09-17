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
pub fn rmsnorm_backward_rows(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Vec<u64> = vec![0u64; hidden]; // +0.0（f64 bits）。
    if rows == 0 || hidden == 0 {
        let dw = w.map(|_| dw_acc.into_iter().map(narrow_f64_bits_f32).collect());
        return (dx, dw);
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
    (dx, dw)
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
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward_rows(
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Vec<u64> = vec![0u64; hidden];
    let mut db_acc: Vec<u64> = vec![0u64; hidden];
    if rows == 0 || hidden == 0 {
        let dw = w.map(|_| dw_acc.into_iter().map(narrow_f64_bits_f32).collect());
        let db = has_bias.then(|| db_acc.into_iter().map(narrow_f64_bits_f32).collect());
        return (dx, dw, db);
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
    (dx, dw, db)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU ホスト参照実装（`fandhe_ai_autodiff::grad::rmsnorm_vjp_rows`。
    /// private のため同一アルゴリズムをテスト内に複製する。逐次和版
    /// （butterfly ではない）。`.claude/rules/coding-rust.md` の REQ-2
    /// 統一複合判定で本モジュールの出力と突き合わせる。
    fn cpu_row_rms_stats(row: &[f32], eps: f32) -> f32 {
        let mut acc = 0.0f64;
        for &v in row {
            let v = v as f64;
            acc = v.mul_add(v, acc);
        }
        (1.0f64 / acc.mul_add(1.0 / row.len() as f64, eps as f64).sqrt()) as f32
    }

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
            let mut dot_acc = 0.0f64;
            for (i, &xv) in row.iter().enumerate() {
                let xhat = xv * rstd;
                let term = dxhat_at(i) * xhat;
                dot_acc += term as f64;
            }
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
            let mut sum = 0.0f64;
            for &v in row {
                sum += v as f64;
            }
            let mean = sum / n;
            let mut sq = 0.0f64;
            for &v in row {
                let d = v as f64 - mean;
                sq = d.mul_add(d, sq);
            }
            let var = sq / n;
            let rstd = 1.0f64 / (var + eps as f64).sqrt();
            let xhat_at = |i: usize| -> f32 { ((row[i] as f64 - mean) * rstd) as f32 };
            let dxhat_at = |i: usize| -> f32 {
                let wv = w.map_or(1.0f32, |w| w[i]);
                dy_row[i] * wv
            };
            let mut sum_dxhat = 0.0f64;
            let mut dot_acc = 0.0f64;
            for i in 0..hidden {
                sum_dxhat += dxhat_at(i) as f64;
                let term = dxhat_at(i) * xhat_at(i);
                dot_acc += term as f64;
            }
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

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn f32(&mut self) -> f32 {
            let bits = (self.next() as u32) & 0x807F_FFFF;
            let v = f32::from_bits(bits);
            if v.is_nan() { 0.0 } else { v }
        }
    }

    fn assert_req2(actual: f32, expected: f32, ctx: &str) {
        if expected == 0.0 && actual == 0.0 {
            return;
        }
        let abs_diff = (actual - expected).abs();
        let rel_diff = abs_diff / expected.abs().max(actual.abs()).max(f32::EPSILON);
        assert!(
            abs_diff < 1e-5 || rel_diff < 1e-3,
            "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
        );
    }

    fn assert_req2_slice(actual: &[f32], expected: &[f32], ctx: &str) {
        assert_eq!(actual.len(), expected.len(), "{ctx}: 長さ不一致");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_req2(a, e, &format!("{ctx}[{i}]"));
        }
    }

    #[test]
    fn rmsnorm_backward_matches_cpu_reference_req2() {
        for &(rows, hidden) in &[(1usize, 1usize), (2, 31), (3, 32), (5, 64), (2, 100)] {
            for has_weight in [false, true] {
                let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (rows as u64) ^ ((hidden as u64) << 8));
                let x: Vec<f32> = (0..rows * hidden).map(|_| rng.f32() * 4.0 - 2.0).collect();
                let dy: Vec<f32> = (0..rows * hidden).map(|_| rng.f32() * 2.0 - 1.0).collect();
                let w: Option<Vec<f32>> =
                    has_weight.then(|| (0..hidden).map(|_| rng.f32() * 2.0 - 1.0).collect());
                let w_slice = w.as_deref();
                let (dx_gpu, dw_gpu) = rmsnorm_backward_rows(&x, w_slice, 1e-5, rows, hidden, &dy);
                let (dx_cpu, dw_cpu) = cpu_rmsnorm_backward(&x, w_slice, 1e-5, rows, hidden, &dy);
                assert_req2_slice(&dx_gpu, &dx_cpu, "rmsnorm dx");
                match (dw_gpu, dw_cpu) {
                    (Some(a), Some(e)) => assert_req2_slice(&a, &e, "rmsnorm dw"),
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
                    let mut rng =
                        Rng(0xABCD_1234_0000_0001 ^ (rows as u64) ^ ((hidden as u64) << 8));
                    let x: Vec<f32> = (0..rows * hidden).map(|_| rng.f32() * 4.0 - 2.0).collect();
                    let dy: Vec<f32> = (0..rows * hidden).map(|_| rng.f32() * 2.0 - 1.0).collect();
                    let w: Option<Vec<f32>> =
                        has_weight.then(|| (0..hidden).map(|_| rng.f32() * 2.0 - 1.0).collect());
                    let w_slice = w.as_deref();
                    let (dx_gpu, dw_gpu, db_gpu) =
                        layer_norm_backward_rows(&x, w_slice, has_bias, 1e-5, rows, hidden, &dy);
                    let (dx_cpu, dw_cpu, db_cpu) =
                        cpu_layer_norm_backward(&x, w_slice, has_bias, 1e-5, rows, hidden, &dy);
                    assert_req2_slice(&dx_gpu, &dx_cpu, "layer_norm dx");
                    match (dw_gpu, dw_cpu) {
                        (Some(a), Some(e)) => assert_req2_slice(&a, &e, "layer_norm dw"),
                        (None, None) => {}
                        _ => panic!("layer_norm dw Some/None mismatch"),
                    }
                    match (db_gpu, db_cpu) {
                        (Some(a), Some(e)) => assert_req2_slice(&a, &e, "layer_norm db"),
                        (None, None) => {}
                        _ => panic!("layer_norm db Some/None mismatch"),
                    }
                }
            }
        }
    }
}
