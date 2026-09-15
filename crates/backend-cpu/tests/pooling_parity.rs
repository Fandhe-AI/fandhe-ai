//! `CpuBackendOps::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d`
//! （イシュー #1728）の受け入れ条件テスト。
//!
//! `fandhe_ai_autodiff::eval::{max_pool2d, avg_pool2d,
//! adaptive_avg_pool2d}`（ホスト参照実装。`autodiff` クレート非公開
//! のため `backend-cpu` から直接は呼べない）と CPU ネイティブ実装
//! （本クレート `pooling` モジュール）が同一アルゴリズムであることを、
//! `Var::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d` の 2 系統
//! 実行経路を間接的に突き合わせて確認する（`interpolate_parity.rs`／
//! `gather_scatter_parity.rs` と同型の `ForceEvalFallback` 方針）。
//! 両経路の forward（値・索引）／backward（VJP）が bit 完全一致する
//! ことを確認する（MaxPool は選択演算・AvgPool は `f64` 縮約契約が
//! CPU ネイティブ・ホスト参照実装いずれも同一アルゴリズムのため）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Pool2dParams, Tensor};

/// `CpuBackendOps` の必須メソッドへ委譲しつつ、対象の pooling
/// メソッドだけは意図的に override せずデフォルト（`Unsupported`）の
/// まま残すラッパー（`interpolate_parity.rs::ForceEvalFallback` と
/// 同型）。`Var::*_pool2d` を `autodiff::eval::*`（ホストフォール
/// バック経路）へ強制的に迂回させるための唯一の差分点。
struct ForceEvalFallback {
    inner: CpuBackendOps,
}

impl BackendOps for ForceEvalFallback {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    // `max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d` はデフォルト
    // 実装（`Unsupported`）のまま override しない。これが `eval::`
    // フォールバック経路の唯一の実現手段。
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn assert_bit_exact(label: &str, native: &Tensor<f32>, fallback: &Tensor<f32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: shape が一致しない"
    );
    let a = dense(native);
    let b = dense(fallback);
    assert_eq!(a.len(), b.len(), "{label}: 要素数が一致しない");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: 要素 {i} が bit 一致しない（native={x}, fallback={y}）"
        );
    }
}

fn assert_i32_exact(label: &str, native: &Tensor<i32>, fallback: &Tensor<i32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: shape が一致しない"
    );
    for i in 0..native.numel() {
        let coords = unravel(i, native.shape());
        assert_eq!(
            native.get(&coords),
            fallback.get(&coords),
            "{label}: 索引 {i} が一致しない"
        );
    }
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis].max(1);
        coords[axis] = flat % dim;
        flat /= dim;
    }
    coords
}

/// `max_pool2d`（重なり窓・NaN 混入）の CPU ネイティブ実装と
/// `eval::max_pool2d` フォールバックが forward（値・索引）で
/// bit 完全一致することを確認する。
#[test]
fn max_pool2d_forward_native_matches_eval_fallback_bit_exact() {
    let x = t(
        vec![1.0, f32::NAN, 2.0, 5.0, 4.0, 0.0, -1.0, 3.0, 2.5],
        &[1, 1, 3, 3],
    );

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let (native_out, native_idx) = native_x
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let (fallback_out, fallback_idx) = fallback_x
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();

    assert_bit_exact(
        "max_pool2d(forward)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    assert_i32_exact("max_pool2d(index)", &native_idx, &fallback_idx);
}

/// `max_pool2d` の backward（scatter_add ベース VJP）が CPU
/// ネイティブ実装経路と `eval::` フォールバック経路で bit 完全一致
/// することを確認する。
#[test]
fn max_pool2d_backward_native_matches_eval_fallback_bit_exact() {
    let x = t(
        vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0],
        &[1, 1, 3, 3],
    );

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let (native_out, _idx) = native_x
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    let native_loss = native_out.sum(None).unwrap();
    let native_grads = native_tape.backward(&native_loss).unwrap();
    let native_dx = native_grads
        .get(&native_x)
        .unwrap()
        .expect("x は loss に到達する");

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let (fallback_out, _idx2) = fallback_x
        .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
        .unwrap();
    let fallback_loss = fallback_out.sum(None).unwrap();
    let fallback_grads = fallback_tape.backward(&fallback_loss).unwrap();
    let fallback_dx = fallback_grads
        .get(&fallback_x)
        .unwrap()
        .expect("x は loss に到達する");

    assert_bit_exact("max_pool2d(backward dX)", native_dx, fallback_dx);
}

/// `avg_pool2d`（重なり窓・padding・count_include_pad 両値）の CPU
/// ネイティブ実装と `eval::avg_pool2d` フォールバックが forward で
/// bit 完全一致することを確認する（`f64` 縮約契約の一致）。
#[test]
fn avg_pool2d_forward_native_matches_eval_fallback_bit_exact() {
    let x = t(
        (0..24).map(|v| v as f32 * 0.37 - 3.0).collect(),
        &[1, 2, 3, 4],
    );

    for count_include_pad in [true, false] {
        let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let native_x = native_tape.var(&x);
        let native_out = native_x
            .avg_pool2d([2, 2], Some([1, 1]), [1, 1], false, count_include_pad)
            .unwrap();

        let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
            inner: CpuBackendOps::new(),
        }));
        let fallback_x = fallback_tape.var(&x);
        let fallback_out = fallback_x
            .avg_pool2d([2, 2], Some([1, 1]), [1, 1], false, count_include_pad)
            .unwrap();

        assert_bit_exact(
            "avg_pool2d(forward)",
            &native_out.to_tensor(),
            &fallback_out.to_tensor(),
        );
    }
}

/// `avg_pool2d` の backward が CPU ネイティブ実装経路と `eval::`
/// フォールバック経路で bit 完全一致することを確認する。
#[test]
fn avg_pool2d_backward_native_matches_eval_fallback_bit_exact() {
    let x = t(
        (0..16).map(|v| v as f32 * 0.5 - 1.0).collect(),
        &[1, 1, 4, 4],
    );

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x
        .avg_pool2d([2, 2], Some([2, 2]), [0, 0], false, true)
        .unwrap();
    let native_loss = native_out.sum(None).unwrap();
    let native_grads = native_tape.backward(&native_loss).unwrap();
    let native_dx = native_grads
        .get(&native_x)
        .unwrap()
        .expect("x は loss に到達する");

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x
        .avg_pool2d([2, 2], Some([2, 2]), [0, 0], false, true)
        .unwrap();
    let fallback_loss = fallback_out.sum(None).unwrap();
    let fallback_grads = fallback_tape.backward(&fallback_loss).unwrap();
    let fallback_dx = fallback_grads
        .get(&fallback_x)
        .unwrap()
        .expect("x は loss に到達する");

    assert_bit_exact("avg_pool2d(backward dX)", native_dx, fallback_dx);
}

/// `adaptive_avg_pool2d`（非割り切れ窓）の CPU ネイティブ実装と
/// `eval::adaptive_avg_pool2d` フォールバックが forward／backward で
/// bit 完全一致することを確認する。
#[test]
fn adaptive_avg_pool2d_forward_and_backward_native_matches_eval_fallback_bit_exact() {
    let x = t((0..21).map(|v| v as f32).collect(), &[1, 1, 3, 7]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.adaptive_avg_pool2d([2, 3]).unwrap();
    let native_loss = native_out.sum(None).unwrap();
    let native_grads = native_tape.backward(&native_loss).unwrap();
    let native_dx = native_grads
        .get(&native_x)
        .unwrap()
        .expect("x は loss に到達する");

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x.adaptive_avg_pool2d([2, 3]).unwrap();
    let fallback_loss = fallback_out.sum(None).unwrap();
    let fallback_grads = fallback_tape.backward(&fallback_loss).unwrap();
    let fallback_dx = fallback_grads
        .get(&fallback_x)
        .unwrap()
        .expect("x は loss に到達する");

    assert_bit_exact(
        "adaptive_avg_pool2d(forward)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    assert_bit_exact("adaptive_avg_pool2d(backward dX)", native_dx, fallback_dx);
}

/// `CpuBackendOps::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d`
/// が shape 再検査を実装側でも行い、不一致を `BackendError::
/// ShapeMismatch` として fail-closed に拒否することを確認する
/// （`.claude/rules/security.md` A08。トレイトメソッドを直接呼び
/// `Var` 側の検査を経由しない経路を対象とする）。
#[test]
fn backend_ops_pooling_rejects_zero_spatial_axis() {
    let ops = CpuBackendOps::new();
    let input = t(Vec::new(), &[1, 1, 0, 4]);
    let params = Pool2dParams::new([2, 2], None, [0, 0], [1, 1]).unwrap();

    let err = ops.max_pool2d(&input, &params).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));

    let err2 = ops.avg_pool2d(&input, &params, true).unwrap_err();
    assert!(matches!(err2, BackendError::ShapeMismatch(_)));

    let err3 = ops.adaptive_avg_pool2d(&input, [2, 2]).unwrap_err();
    assert!(matches!(err3, BackendError::ShapeMismatch(_)));
}
