//! `CpuBackendOps::gather`／`scatter`（イシュー #1776）の受け入れ条件
//! テスト。
//!
//! `fandhe_ai_autodiff::eval::gather`／`scatter`（ホスト参照実装。
//! `autodiff` クレート非公開のため `backend-cpu` から直接は呼べない）
//! と CPU ネイティブ実装（本クレート `gather_scatter` モジュール）が
//! 同一アルゴリズム（`ScatterReduce` doc の決定的集約契約）であること
//! を、`Var::gather`／`scatter`／`scatter_add` の 2 系統実行経路を
//! 間接的に突き合わせて確認する: `CpuBackendOps` をそのまま渡した
//! `Tape` は本クレートのネイティブ実装を経由し、`gather`／`scatter`
//! だけを強制的に `Unsupported` にする [`ForceEvalFallback`] を渡した
//! `Tape` は `autodiff` のホストフォールバック（`eval::gather`／
//! `scatter`）を経由する（`fusion_effect_perf.rs::NonFusedCpuOps` と
//! 同型の「1 メソッドだけ意図的に override しない」ラッパー方針）。
//! 両経路の出力が `f32::to_bits()` で bit 完全一致することを固定し、
//! 将来どちらかの実装が並列化された際の決定性回帰を検知する。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, Tensor};

/// `CpuBackendOps` の必須メソッド（デフォルト実装を持たない 9 個）へ
/// 委譲しつつ、`gather`／`scatter` だけは意図的に override せずデフォルト
/// （`Unsupported`）のまま残すラッパー（`fusion_effect_perf.rs::
/// NonFusedCpuOps` と同型）。`Var::gather`／`scatter`／`scatter_add` を
/// `autodiff::eval::gather`／`scatter`（ホストフォールバック経路）へ
/// 強制的に迂回させるための唯一の差分点。
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
    // `gather`／`scatter` はデフォルト実装（`Unsupported`）のまま
    // override しない。これが `eval::` フォールバック経路の唯一の
    // 実現手段。
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn i32t(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
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

/// `gather`（重複読み出しを含む）の CPU ネイティブ実装と `eval::gather`
/// フォールバックが bit 完全一致することを確認する。
#[test]
fn gather_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = i32t(vec![0, 2, 2, 1], &[2, 2]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.gather(1, &index).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x.gather(1, &index).unwrap();

    assert_bit_exact("gather", &native_out.to_tensor(), &fallback_out.to_tensor());
}

/// `scatter`（`Overwrite`）の CPU ネイティブ実装と `eval::scatter`
/// フォールバックが bit 完全一致することを確認する。
#[test]
fn scatter_overwrite_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let index = i32t(vec![0, 2, 2, 1], &[2, 2]);
    let src = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_src = native_tape.var(&src);
    let native_out = native_x.scatter(1, &index, &native_src).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_src = fallback_tape.var(&src);
    let fallback_out = fallback_x.scatter(1, &index, &fallback_src).unwrap();

    assert_bit_exact(
        "scatter(overwrite)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
}

/// `scatter_add`（`Add`。重複添字を含み、`f64` アキュムレータでの
/// 決定的集約順序が両実装で一致することが焦点）の CPU ネイティブ
/// 実装と `eval::scatter` フォールバックが bit 完全一致することを
/// 確認する（将来どちらかが並列化された際の回帰検知を兼ねる）。
#[test]
fn scatter_add_native_matches_eval_fallback_bit_exact() {
    // 全 index が同一位置（row=0）を指し、5 要素を row-major 順に
    // 加算する（決定的集約順序の bit 一致確認が主眼）。
    let x = t(vec![0.1, 0.2, 0.3], &[1, 3]);
    let index = i32t(vec![0, 0, 0, 0, 0], &[1, 5]);
    let src = t(vec![1.0e20, 2.0e-20, 3.5, -1.0e20, 0.25], &[1, 5]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_src = native_tape.var(&src);
    let native_out = native_x.scatter_add(1, &index, &native_src).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_src = fallback_tape.var(&src);
    let fallback_out = fallback_x.scatter_add(1, &index, &fallback_src).unwrap();

    assert_bit_exact(
        "scatter_add",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
}

/// `CpuBackendOps::gather`／`scatter` が `ops_shape::gather_out_shape`／
/// `scatter_out_shape` による shape 再検査を実装側でも行い、
/// 不一致を `BackendError::ShapeMismatch` として fail-closed に拒否
/// することを確認する（`.claude/rules/security.md` A08。トレイト
/// メソッドを直接呼び `Var` 側の検査を経由しない経路を対象とする）。
#[test]
fn backend_ops_gather_rejects_axis_out_of_range() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
    let index = Tensor::<i32>::new(vec![0], &[1]).unwrap();
    let err = ops.gather(&input, 5, &index).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn backend_ops_scatter_rejects_index_src_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let index = Tensor::<i32>::new(vec![0, 1], &[1, 2]).unwrap();
    let src = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let err = ops
        .scatter(&input, 1, &index, &src, ScatterReduce::Overwrite)
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
