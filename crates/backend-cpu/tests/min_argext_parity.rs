//! `CpuBackendOps::min`／`argmax`／`argmin`（イシュー #1720）の受け入れ
//! 条件テスト。
//!
//! `fandhe_ai_autodiff::eval::min`／`argmax`／`argmin`（ホスト参照実装。
//! `autodiff` クレート非公開のため `backend-cpu` から直接は呼べない）
//! と CPU ネイティブ実装（本クレート `reduction` モジュール）が同一
//! 意味論（`BackendOps::min`／`argmax`／`argmin` doc の NaN 非伝播・
//! タイ先勝ち契約）であることを、`Var::min`／`argmax`／`argmin` の
//! 2 系統実行経路を間接的に突き合わせて確認する（`sort_topk_parity.rs::
//! ForceEvalFallback` と同型の「1 系統だけ意図的に override しない」
//! ラッパー方針。`min`／`argmax`／`argmin` は `BackendOps` のデフォルト
//! メソッドのため、本ファイルの `ForceEvalFallback` は明示的に
//! override しないだけで自動的に `Unsupported` のまま残り、
//! `Var::min` 等をホストフォールバック（`eval::min` 等）へ強制的に
//! 迂回させる）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `CpuBackendOps` の必須メソッド（デフォルト実装を持たない 9 個）へ
/// 委譲しつつ、`min`／`argmax`／`argmin` は意図的に override せず
/// デフォルト（`Unsupported`）のまま残す（`sort_topk_parity.rs::
/// ForceEvalFallback` と同型）。
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
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.contiguous().as_slice().unwrap().to_vec()
}

fn dense_vec_i32(tensor: &Tensor<i32>) -> Vec<i32> {
    tensor.contiguous().as_slice().unwrap().to_vec()
}

fn assert_bit_exact(label: &str, native: &Tensor<f32>, fallback: &Tensor<f32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: shape が一致しない"
    );
    let a = dense_vec(native);
    let b = dense_vec(fallback);
    assert_eq!(a.len(), b.len(), "{label}: 要素数が一致しない");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: 要素 {i} が bit 一致しない（native={x}, fallback={y}）"
        );
    }
}

fn assert_index_exact(label: &str, native: &Tensor<i32>, fallback: &Tensor<i32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: index の shape が一致しない"
    );
    assert_eq!(
        dense_vec_i32(native),
        dense_vec_i32(fallback),
        "{label}: index が一致しない"
    );
}

/// `min`（NaN・タイ・-0.0/0.0 を含む）の CPU ネイティブ実装と
/// `eval::min` フォールバックが bit 完全一致することを確認する。
#[test]
fn min_native_matches_eval_fallback_bit_exact() {
    let x = t(
        vec![2.0, -1.0, f32::NAN, -0.0, 0.0, -1.0, f32::NAN],
        &[1, 7],
    );

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_var = native_tape.var(&x);
    let native_out = native_var.min(None).unwrap().to_tensor();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_var = fallback_tape.var(&x);
    let fallback_out = fallback_var.min(None).unwrap().to_tensor();

    assert_bit_exact("min(None)", &native_out, &fallback_out);
}

/// `min(Some(axis))`（tie を含む）の両経路 bit 一致。
#[test]
fn min_axis_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![-5.0, 1.0, -5.0, 1.0, -2.0, 4.0], &[2, 3]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_var = native_tape.var(&x);
    let native_out = native_var.min(Some(1)).unwrap().to_tensor();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_var = fallback_tape.var(&x);
    let fallback_out = fallback_var.min(Some(1)).unwrap().to_tensor();

    assert_bit_exact("min(Some(1))", &native_out, &fallback_out);
}

/// `argmax`／`argmin`（NaN・タイを含む）の両経路 index 完全一致。
#[test]
fn argmax_and_argmin_native_match_eval_fallback() {
    let x = t(vec![1.0, 5.0, f32::NAN, 5.0, -2.0, -2.0], &[2, 3]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_var = native_tape.var(&x);
    let native_argmax = native_var.argmax(None).unwrap();
    let native_argmin = native_var.argmin(None).unwrap();
    let native_argmax_axis = native_var.argmax(Some(1)).unwrap();
    let native_argmin_axis = native_var.argmin(Some(1)).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_var = fallback_tape.var(&x);
    let fallback_argmax = fallback_var.argmax(None).unwrap();
    let fallback_argmin = fallback_var.argmin(None).unwrap();
    let fallback_argmax_axis = fallback_var.argmax(Some(1)).unwrap();
    let fallback_argmin_axis = fallback_var.argmin(Some(1)).unwrap();

    assert_index_exact("argmax(None)", &native_argmax, &fallback_argmax);
    assert_index_exact("argmin(None)", &native_argmin, &fallback_argmin);
    assert_index_exact(
        "argmax(Some(1))",
        &native_argmax_axis,
        &fallback_argmax_axis,
    );
    assert_index_exact(
        "argmin(Some(1))",
        &native_argmin_axis,
        &fallback_argmin_axis,
    );
}

/// `min`／`argmax`／`argmin` が run-to-run で bit／index 完全一致する
/// （決定性契約。`sort_topk_parity.rs::sort_and_topk_are_run_to_run_
/// bit_identical` と同型）。
#[test]
fn min_and_argext_are_run_to_run_bit_identical() {
    let x = t(vec![3.0, -1.0, -1.0, 4.0, -5.0, -5.0, 2.0, 0.0], &[8]);

    let run_min = || {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let v = tape.var(&x);
        v.min(None).unwrap().to_tensor()
    };
    let run_argmin = || {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let v = tape.var(&x);
        v.argmin(None).unwrap()
    };

    let a = run_min();
    let b = run_min();
    assert_bit_exact("min run-to-run", &a, &b);

    let ia = run_argmin();
    let ib = run_argmin();
    assert_index_exact("argmin run-to-run", &ia, &ib);
}

/// 空縮約は `min`／`argmax`／`argmin` いずれも `AutodiffError` を返す
/// （単位元を持たないため。`BackendOps::min`／`argmax`／`argmin` doc
/// の空縮約契約）。
#[test]
fn min_and_argext_empty_reduction_are_errors() {
    let x = t(Vec::new(), &[0]);
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let v = tape.var(&x);

    assert!(v.min(None).is_err());
    assert!(v.argmax(None).is_err());
    assert!(v.argmin(None).is_err());
}
