//! `fandhe_ai_backend_cpu::{softmax, rmsnorm}` と `onnx-interop` 素朴実装の数値一致
//! 検証（イシュー #607 受入基準 3）。
//!
//! **素朴実装側のコードは変更しない**（`onnx_interop::ops::softmax`／
//! `onnx_interop::ops::layer_normalization` はそのまま）。判定式・許容
//! 誤差は `fandhe_ai_backend_cpu::parity`（REQ-2 統一複合判定）を唯一の参照とし
//! 再定義しない（`.claude/rules/coding-rust.md`）。
//!
//! ## softmax
//! `fandhe_ai_backend_cpu::softmax::run_softmax_f32`（行方向 2 次元入口）と
//! `onnx_interop::ops::softmax`（`axis` 指定の N 次元入口。`axis = -1`）を
//! 突き合わせる。
//!
//! ## RMSNorm と LayerNorm の等価条件
//! `onnx_interop::ops::layer_normalization` は平均減算（`(X - mean) /
//! sqrt(var + eps) * scale + bias`）を含むため、一般入力では RMSNorm
//! （`X * rsqrt(mean(X^2) + eps) * w`）と数学的に一致しない。
//! **行平均 0 に正規化した入力**を使うと、その行の分散 `var =
//! mean((X-mean)^2) = mean(X^2)`（`mean == 0` のため）となり、RMSNorm の
//! `mean(X^2)` と一致する。この条件下でのみ `bias = None`・`scale = w`・
//! 同一 `eps` として両実装が一致することをテストする（一般入力での一致は
//! 数学的に成立しないため要求しない）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32;
use fandhe_ai_backend_cpu::softmax::run_softmax_f32;
use fandhe_ai_tensor_core::Tensor;
use onnx_interop::ops::{LayerNormAttrs, layer_normalization, softmax};

#[test]
fn cpu_softmax_matches_onnx_naive_softmax_2d() {
    let rows = 5usize;
    let cols = 37usize; // NEON 4 要素幅の非倍数を含める。
    let x_data = Xorshift64Star::new(1234).fill_vec(rows * cols);

    let actual = run_softmax_f32(&x_data, rows, cols).unwrap();

    let x_tensor = Tensor::new(x_data, &[rows, cols]).unwrap();
    let expected_tensor = softmax(&x_tensor, -1).unwrap();
    let expected = expected_tensor.contiguous();
    let expected_slice = expected.as_slice().unwrap();

    assert_parity(
        "fandhe_ai_backend_cpu::softmax vs onnx_interop::ops::softmax (2D, axis=-1)",
        &actual,
        expected_slice,
    );
}

#[test]
fn cpu_softmax_matches_onnx_naive_softmax_3d() {
    let dims = [2usize, 3, 17];
    let numel: usize = dims.iter().product();
    let x_data = Xorshift64Star::new(4321).fill_vec(numel);

    let rows = dims[0] * dims[1];
    let cols = dims[2];
    let actual = run_softmax_f32(&x_data, rows, cols).unwrap();

    let x_tensor = Tensor::new(x_data, &dims).unwrap();
    let expected_tensor = softmax(&x_tensor, -1).unwrap();
    let expected = expected_tensor.contiguous();
    let expected_slice = expected.as_slice().unwrap();

    assert_parity(
        "fandhe_ai_backend_cpu::softmax vs onnx_interop::ops::softmax (3D, axis=-1)",
        &actual,
        expected_slice,
    );
}

/// 行平均 0 へ正規化した入力を生成する（RMSNorm-LayerNorm 等価条件。
/// モジュール冒頭コメント参照）。
fn zero_mean_rows(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let raw = Xorshift64Star::new(seed).fill_vec(rows * cols);
    let mut out = vec![0.0f32; raw.len()];
    for (out_row, in_row) in out.chunks_mut(cols).zip(raw.chunks(cols)) {
        let mean: f32 = in_row.iter().sum::<f32>() / cols as f32;
        for (o, &v) in out_row.iter_mut().zip(in_row.iter()) {
            *o = v - mean;
        }
    }
    out
}

#[test]
fn cpu_rmsnorm_matches_onnx_layer_norm_under_zero_mean_equivalence() {
    let rows = 4usize;
    let cols = 33usize; // NEON 端要素を含む。
    let eps = 1e-5f32;

    let x_data = zero_mean_rows(9999, rows, cols);
    let w_data = Xorshift64Star::new(5555).fill_vec(cols);

    let actual = run_rmsnorm_f32(&x_data, Some(&w_data), eps, rows, cols).unwrap();

    let x_tensor = Tensor::new(x_data, &[rows, cols]).unwrap();
    let scale_tensor = Tensor::new(w_data, &[cols]).unwrap();
    let attrs = LayerNormAttrs {
        axis: -1,
        epsilon: eps,
    };
    let expected_tensor = layer_normalization(&x_tensor, &scale_tensor, None, &attrs).unwrap();
    let expected = expected_tensor.contiguous();
    let expected_slice = expected.as_slice().unwrap();

    assert_parity(
        "fandhe_ai_backend_cpu::rmsnorm vs onnx_interop::ops::layer_normalization (zero-mean equivalence)",
        &actual,
        expected_slice,
    );
}
