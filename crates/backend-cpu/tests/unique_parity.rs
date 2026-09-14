//! `CpuBackendOps::unique`（イシュー #1734）の受け入れ条件テスト。
//!
//! `fandhe_ai_autodiff::eval::unique`（ホスト参照実装。`autodiff`
//! クレート非公開のため `backend-cpu` から直接は呼べない）と CPU
//! ネイティブ実装（本クレート `unique` モジュール）が同一アルゴリズム
//! （totalOrder ソート・`==` による重複判定）であることを、
//! `Var::unique` の 2 系統実行経路を間接的に突き合わせて確認する
//! （`gather_scatter_parity.rs` と同型の `ForceEvalFallback` 方針）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::Device;
use fandhe_ai_tensor_core::{BackendError, BackendOps, Tensor};

/// `CpuBackendOps` の必須メソッドへ委譲しつつ、`unique` だけは
/// 意図的に override せずデフォルト（`Unsupported`）のまま残す
/// ラッパー（`gather_scatter_parity.rs::ForceEvalFallback` と同型）。
/// `Var::unique` を `autodiff::eval::unique`（ホストフォールバック
/// 経路）へ強制的に迂回させるための唯一の差分点。
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
    // `unique` はデフォルト実装（`Unsupported`）のまま override しない。
    // これが `eval::unique` フォールバック経路の唯一の実現手段。
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn assert_bit_exact(label: &str, native: &Tensor<f32>, fallback: &Tensor<f32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: shape が一致しない"
    );
    let a: Vec<f32> = native.host_slice().into_owned();
    let b: Vec<f32> = fallback.host_slice().into_owned();
    assert_eq!(a.len(), b.len(), "{label}: 要素数が一致しない");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: 要素 {i} が bit 一致しない（native={x}, fallback={y}）"
        );
    }
}

/// `unique`（重複・NaN・±0 混在）の CPU ネイティブ実装と
/// `eval::unique` フォールバックが bit 完全一致することを確認する。
#[test]
fn unique_native_matches_eval_fallback_bit_exact() {
    let nan1 = f32::NAN;
    let nan2 = f32::from_bits(f32::NAN.to_bits() | 1);
    let x = t(
        vec![
            3.0,
            1.0,
            2.0,
            1.0,
            -0.0,
            0.0,
            nan1,
            nan2,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        &[10],
    );

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.unique().unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x.unique().unwrap();

    assert_bit_exact("unique", &native_out, &fallback_out);
}

/// 空入力（`numel == 0`）の CPU ネイティブ実装と `eval::unique`
/// フォールバックが shape `[0]` で一致することを確認する。
#[test]
fn unique_native_matches_eval_fallback_empty_input() {
    let x = t(Vec::new(), &[3, 0]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.unique().unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x.unique().unwrap();

    assert_bit_exact("unique(empty)", &native_out, &fallback_out);
}

/// 非 contiguous な入力（transpose 済み view）でも CPU ネイティブ
/// 実装と `eval::unique` フォールバックが bit 完全一致することを
/// 確認する。
#[test]
fn unique_native_matches_eval_fallback_non_contiguous() {
    let base = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&base);
    let native_t = native_x.permute(&[1, 0]).unwrap();
    let native_out = native_t.unique().unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&base);
    let fallback_t = fallback_x.permute(&[1, 0]).unwrap();
    let fallback_out = fallback_t.unique().unwrap();

    assert_bit_exact("unique(non_contiguous)", &native_out, &fallback_out);
}
