//! `kernels_fused_elementwise.rs` の静的インタプリタを CPU 上で (block, thread) ごとに再現する
//! テスト専用モデル（イシュー #3433・R2）。
//!
//! 役割: `[kind, a, b]` の `i32` プログラムを、カーネルと**同じガード**（`n_ops` の範囲・オペランド
//! 添字 `< i`・leaf 添字・未知 kind。違反時は quiet NaN を書く）で解釈する。ROCm 非搭載環境で
//! 符号化（`elementwise::encode_program`）とインタプリタの契約を結合して検証し、`RocmBackendOps` の
//! HostModel 経路（`ops/ew.rs`）が呼ぶ。式は Rust の `f32` 演算（`exp`・`tanh` はホスト libm）で
//! GPU の libm を模したものではない。実機の parity は #3159。

use crate::elementwise_model::relu_kernel;
use crate::kernels_fused_elementwise::{
    MAX_FUSED_OPS, OP_ADD, OP_EXP, OP_INPUT, OP_MUL, OP_RELU, OP_TANH,
};
use crate::launch::for_each_thread_gid;

/// プログラムを全スレッド分解釈する。`leaves` は `[leaf_count * numel]`。
pub(crate) fn simulate(
    words: &[i32],
    n_ops: i32,
    leaf_count: i32,
    leaves: &[f32],
    numel: usize,
) -> Vec<f32> {
    let mut out = vec![f32::from_bits(0x7fc0_1234); numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |gid| {
        if gid >= numel as i64 {
            return;
        }
        let g = gid as usize;
        out[g] = eval_one(words, n_ops, leaf_count, leaves, numel, gid);
        writes[g] += 1;
    });
    assert!(writes.iter().all(|&w| w == 1));
    out
}

fn eval_one(
    words: &[i32],
    n_ops: i32,
    leaf_count: i32,
    leaves: &[f32],
    numel: usize,
    gid: i64,
) -> f32 {
    let nan = f32::NAN;
    if n_ops <= 0 || n_ops as usize > MAX_FUSED_OPS {
        return nan;
    }
    let mut r = [0.0f32; MAX_FUSED_OPS];
    for i in 0..n_ops {
        let base = 3 * i as usize;
        let (kind, a, b) = (words[base], words[base + 1], words[base + 2]);
        let v = match kind {
            OP_INPUT => {
                if a < 0 || a >= leaf_count {
                    return nan;
                }
                leaves[(a as i64 * numel as i64 + gid) as usize]
            }
            OP_ADD | OP_MUL => {
                if a < 0 || a >= i || b < 0 || b >= i {
                    return nan;
                }
                if kind == OP_ADD {
                    r[a as usize] + r[b as usize]
                } else {
                    r[a as usize] * r[b as usize]
                }
            }
            OP_RELU | OP_EXP | OP_TANH => {
                if a < 0 || a >= i {
                    return nan;
                }
                let x = r[a as usize];
                match kind {
                    OP_RELU => relu_kernel(x),
                    OP_EXP => x.exp(),
                    _ => x.tanh(),
                }
            }
            _ => return nan,
        };
        r[i as usize] = v;
    }
    r[n_ops as usize - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(words: &[i32], n_ops: i32, leaf_count: i32, leaves: &[f32], numel: usize) -> Vec<f32> {
        simulate(words, n_ops, leaf_count, leaves, numel)
    }

    #[test]
    fn evaluates_simple_program() {
        // r0 = leaf0, r1 = leaf1, r2 = r0 + r1, r3 = relu(r2)
        let words = [OP_INPUT, 0, 0, OP_INPUT, 1, 0, OP_ADD, 0, 1, OP_RELU, 2, 0];
        let leaves = [1.0, -5.0, 3.0, 2.0, 1.0, 0.5];
        let out = run(&words, 4, 2, &leaves, 3);
        assert_eq!(out, vec![3.0, 0.0, 3.5]);
    }

    #[test]
    fn invalid_programs_write_nan() {
        let leaves = [1.0, 2.0];
        // オペランドが自ノード以降を指す。
        assert!(
            run(&[OP_INPUT, 0, 0, OP_ADD, 0, 1], 2, 1, &leaves, 2)
                .iter()
                .all(|v| v.is_nan())
        );
        // leaf 添字が範囲外。
        assert!(
            run(&[OP_INPUT, 1, 0], 1, 1, &leaves, 2)
                .iter()
                .all(|v| v.is_nan())
        );
        // 未知 kind。
        assert!(
            run(&[99, 0, 0], 1, 1, &leaves, 2)
                .iter()
                .all(|v| v.is_nan())
        );
        // n_ops が範囲外。
        assert!(
            run(&[OP_INPUT, 0, 0], 0, 1, &leaves, 2)
                .iter()
                .all(|v| v.is_nan())
        );
        assert!(
            run(&[OP_INPUT, 0, 0], 26, 1, &leaves, 2)
                .iter()
                .all(|v| v.is_nan())
        );
        // 負のオペランド。
        assert!(
            run(&[OP_INPUT, 0, 0, OP_RELU, -1, 0], 2, 1, &leaves, 2)
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn ragged_numel_is_covered() {
        for n in [1usize, 255, 256, 257] {
            let leaves: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let out = run(&[OP_INPUT, 0, 0, OP_MUL, 0, 0], 2, 1, &leaves, n);
            assert!(out.iter().enumerate().all(|(i, v)| *v == (i * i) as f32));
        }
    }
}
