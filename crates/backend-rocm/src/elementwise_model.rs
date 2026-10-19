//! `kernels_elementwise.rs`・`kernels_scalar_op.rs` のカーネルを CPU 上で (block, thread) ごとに
//! 再現するテスト専用モデル（イシュー #3433・R2）。
//!
//! 役割: AMD 実機がない環境で、1 次元グリッド分解・`gid >= numel` ガード・各出力がちょうど 1 回
//! 書かれること・カーネル式（三項 relu・scalar 式の Rust 訳）を構造的に検証する。
//! `RocmBackendOps::with_host_model` の経路（`ops/ew.rs`）が呼ぶ。
//!
//! 限界: 式は Rust の `f32` 演算（`exp`・`tanh`・`sin` 等はホスト libm）で、**GPU の libm を模した
//! ものではない**。したがって超越関数の一致は構造の検証にとどまり、実機での CPU との parity は
//! #3159 で測る。`erff` だけは Rust の標準に無いため、`ScalarUnaryOp::Erf` の `f64` 近似で代用する。
//! HIPRTC のコンパイル結果そのものの検証ではない。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp};

use crate::elementwise::{BinaryKind, UnaryKind};
use crate::launch::for_each_thread_gid;

/// 全スレッドを回し、`f(gid)` が返す値を `out[gid]` へ 1 回だけ書く（範囲外 gid は書かない）。
fn run_grid(numel: usize, mut f: impl FnMut(usize) -> f32) -> Vec<f32> {
    let mut out = vec![f32::NAN; numel];
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |gid| {
        if gid >= numel as i64 {
            return;
        }
        let g = gid as usize;
        out[g] = f(g);
        writes[g] += 1;
    });
    assert!(
        writes.iter().all(|&w| w == 1),
        "各出力はちょうど 1 回書かれる"
    );
    out
}

/// `ew_add_f32`／`ew_mul_f32`。
pub(crate) fn binary(kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    run_grid(a.len(), |g| match kind {
        BinaryKind::Add => a[g] + b[g],
        BinaryKind::Mul => a[g] * b[g],
    })
}

/// カーネルの relu（`x > 0.0f ? x : 0.0f`。NaN は 0）。
pub(crate) fn relu_kernel(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// `ew_relu_f32`／`ew_exp_f32`／`ew_tanh_f32`。
pub(crate) fn unary(kind: UnaryKind, a: &[f32]) -> Vec<f32> {
    run_grid(a.len(), |g| match kind {
        UnaryKind::Relu => relu_kernel(a[g]),
        UnaryKind::Exp => a[g].exp(),
        UnaryKind::Tanh => a[g].tanh(),
    })
}

/// `ew_where_f32`。
pub(crate) fn where_cond(cond: &[f32], a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(cond.len() == a.len() && cond.len() == b.len());
    run_grid(cond.len(), |g| if cond[g] != 0.0 { a[g] } else { b[g] })
}

/// `ew_masked_fill_f32`。
pub(crate) fn masked_fill(x: &[f32], mask: &[f32], value: f32) -> Vec<f32> {
    assert_eq!(x.len(), mask.len());
    run_grid(x.len(), |g| if mask[g] != 0.0 { value } else { x[g] })
}

/// `kernels_scalar_op` の unary 式の Rust 訳（`p0`・`p1` は `unary_payload` 順）。
/// 対象外 kind は `None`（`unary_spec` と一致することをテストで固定する）。
fn unary_expr(op: ScalarUnaryOp, x: f32, p0: f32, p1: f32) -> Option<f32> {
    Some(match op {
        ScalarUnaryOp::Sqrt => x.sqrt(),
        ScalarUnaryOp::Neg => -x,
        ScalarUnaryOp::Abs => x.abs(),
        ScalarUnaryOp::Log => x.ln(),
        ScalarUnaryOp::Log2 => x.log2(),
        ScalarUnaryOp::Log10 => x.log10(),
        ScalarUnaryOp::Sin => x.sin(),
        ScalarUnaryOp::Cos => x.cos(),
        ScalarUnaryOp::Tan => x.tan(),
        ScalarUnaryOp::Clamp { .. } => {
            if x.is_nan() {
                x
            } else if p0 > p1 {
                p1
            } else if x < p0 {
                p0
            } else if x > p1 {
                p1
            } else {
                x
            }
        }
        ScalarUnaryOp::LeakyRelu { .. } => {
            if x >= 0.0 {
                x
            } else {
                p0 * x
            }
        }
        ScalarUnaryOp::Hardswish => {
            let t = x + 3.0;
            x * (if t < 0.0 {
                0.0
            } else {
                (if t > 6.0 { 6.0 } else { t }) / 6.0
            })
        }
        ScalarUnaryOp::Silu => {
            if x >= 0.0 {
                x * (1.0 / (1.0 + (-x).exp()))
            } else {
                x * (x.exp() / (1.0 + x.exp()))
            }
        }
        ScalarUnaryOp::Elu { .. } => {
            if x > 0.0 {
                x
            } else {
                p0 * x.exp_m1()
            }
        }
        ScalarUnaryOp::Gelu => {
            0.5 * x * (1.0 + ScalarUnaryOp::Erf.apply(x * std::f32::consts::FRAC_1_SQRT_2))
        }
        ScalarUnaryOp::GeluTanh => {
            0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
        }
        ScalarUnaryOp::Softplus { .. } => {
            if x * p0 > p1 {
                x
            } else {
                (p0 * x).exp().ln_1p() / p0
            }
        }
        _ => return None,
    })
}

/// scalar unary のカーネル再現。対象外 kind は `None`。
pub(crate) fn scalar_unary(op: ScalarUnaryOp, a: &[f32]) -> Option<Vec<f32>> {
    use crate::kernels_scalar_op::{unary_payload, unary_spec};
    unary_spec(op)?;
    let (_, p) = unary_payload(op);
    Some(run_grid(a.len(), |g| {
        unary_expr(op, a[g], p[0], p[1]).expect("unary_spec と一致")
    }))
}

fn bool_f32(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// scalar binary のカーネル再現。対象外 kind は `None`。
pub(crate) fn scalar_binary(op: ScalarBinaryOp, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    crate::kernels_scalar_op::binary_spec(op)?;
    assert_eq!(a.len(), b.len());
    Some(run_grid(a.len(), |g| {
        let (x, y) = (a[g], b[g]);
        match op {
            ScalarBinaryOp::Sub => x - y,
            ScalarBinaryOp::Div => x / y,
            ScalarBinaryOp::Pow => x.powf(y),
            ScalarBinaryOp::Gt => bool_f32(x > y),
            ScalarBinaryOp::Ge => bool_f32(x >= y),
            ScalarBinaryOp::Lt => bool_f32(x < y),
            ScalarBinaryOp::Le => bool_f32(x <= y),
            ScalarBinaryOp::Eq => bool_f32(x == y),
            ScalarBinaryOp::Ne => bool_f32(x != y),
            _ => unreachable!("binary_spec が None を返す kind は上で除外済み"),
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels_scalar_op::tests::{supported_binary, supported_unary};

    #[test]
    fn grid_covers_ragged_blocks_exactly_once() {
        for n in [1usize, 255, 256, 257, 1000] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let out = binary(BinaryKind::Add, &a, &a);
            assert_eq!(out.len(), n);
            assert!(out.iter().enumerate().all(|(i, v)| *v == 2.0 * i as f32));
        }
        assert!(binary(BinaryKind::Add, &[], &[]).is_empty());
    }

    #[test]
    fn relu_kernel_is_ternary_with_nan_to_zero() {
        assert_eq!(relu_kernel(-1.0), 0.0);
        assert_eq!(relu_kernel(2.5), 2.5);
        assert_eq!(relu_kernel(f32::NAN), 0.0);
        assert_eq!(relu_kernel(-0.0).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn model_support_matches_spec_table() {
        let x = [0.5f32];
        for op in supported_unary() {
            assert!(scalar_unary(op, &x).is_some(), "{op:?}");
        }
        for op in supported_binary() {
            assert!(scalar_binary(op, &x, &x).is_some(), "{op:?}");
        }
        assert!(scalar_unary(ScalarUnaryOp::Floor, &x).is_none());
        assert!(scalar_binary(ScalarBinaryOp::Atan2, &x, &x).is_none());
    }
}
