//! `CpuBackendOps::pad`（イシュー #1756）の受け入れ条件テスト。
//!
//! `fandhe_ai_autodiff::eval::pad`（ホスト参照実装。`autodiff` クレート
//! 非公開のため `backend-cpu` から直接は呼べない）と CPU ネイティブ
//! 実装（本クレート `constant_pad` モジュール）が同一アルゴリズムで
//! あることを、`Var::pad` の 2 系統実行経路を間接的に突き合わせて
//! 確認する: `CpuBackendOps` をそのまま渡した `Tape` は本クレートの
//! ネイティブ実装を経由し、`pad` だけを強制的に `Unsupported` にする
//! [`ForceEvalFallback`] を渡した `Tape` は `autodiff` のホスト
//! フォールバック（`eval::pad`）を経由する（`gather_scatter_parity.rs`
//! と同型の「1 メソッドだけ意図的に override しない」ラッパー方針）。
//! 両経路の出力が `f32::to_bits()` で bit 完全一致することを固定する
//! （pad は算術を含まない純粋なコピー演算のため bit 完全一致が数値
//! 契約——`.claude/rules/coding-rust.md`）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `CpuBackendOps` の必須メソッド（デフォルト実装を持たない 9 個）へ
/// 委譲しつつ、`pad` だけは意図的に override せずデフォルト
/// （`Unsupported`）のまま残すラッパー（`gather_scatter_parity.rs::
/// ForceEvalFallback` と同型）。`Var::pad` を `autodiff::eval::pad`
/// （ホストフォールバック経路）へ強制的に迂回させるための唯一の
/// 差分点。
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
    // `pad` はデフォルト実装（`Unsupported`）のまま override しない。
    // これが `eval::pad` フォールバック経路の唯一の実現手段。
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
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

fn run_pad_both(
    x: &Tensor<f32>,
    pads: &[(usize, usize)],
    value: f32,
) -> (
    fandhe_ai_tensor_core::Tensor<f32>,
    fandhe_ai_tensor_core::Tensor<f32>,
) {
    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(x);
    let native_out = native_x.pad(pads, value).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(x);
    let fallback_out = fallback_x.pad(pads, value).unwrap();

    (native_out.to_tensor(), fallback_out.to_tensor())
}

/// 1-D: 両側パディングが bit 完全一致することを確認する。
#[test]
fn pad_1d_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![1.0, 2.0, 3.0], &[3]);
    let (native, fallback) = run_pad_both(&x, &[(1, 2)], 0.0);
    assert_bit_exact("pad_1d", &native, &fallback);
}

/// 2-D: 両軸パディング・片側のみが bit 完全一致することを確認する。
#[test]
fn pad_2d_one_sided_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let (native, fallback) = run_pad_both(&x, &[(1, 0), (0, 2)], -1.5);
    assert_bit_exact("pad_2d_one_sided", &native, &fallback);
}

/// 3-D: 全軸パディングが bit 完全一致することを確認する。
#[test]
fn pad_3d_native_matches_eval_fallback_bit_exact() {
    let x = t((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]);
    let (native, fallback) = run_pad_both(&x, &[(1, 1), (0, 1), (2, 0)], 9.0);
    assert_bit_exact("pad_3d", &native, &fallback);
}

/// 空入力 → 非空出力が bit 完全一致することを確認する。
#[test]
fn pad_empty_input_to_nonempty_output_native_matches_eval_fallback_bit_exact() {
    let x = t(Vec::new(), &[0, 3]);
    let (native, fallback) = run_pad_both(&x, &[(2, 0), (0, 0)], 5.0);
    assert_bit_exact("pad_empty_input", &native, &fallback);
}

/// `pads` すべて 0（恒等コピー）が bit 完全一致することを確認する。
#[test]
fn pad_all_zero_pads_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let (native, fallback) = run_pad_both(&x, &[(0, 0), (0, 0)], 0.0);
    assert_bit_exact("pad_all_zero", &native, &fallback);
}

/// `CpuBackendOps::pad` が `ops_shape::pad_out_shape` による shape
/// 再検査を実装側でも行い、不一致を `BackendError::ShapeMismatch` と
/// して fail-closed に拒否することを確認する（`.claude/rules/
/// security.md` A08。トレイトメソッドを直接呼び `Var` 側の検査を
/// 経由しない経路を対象とする）。
#[test]
fn backend_ops_pad_rejects_rank_mismatch() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let err = ops.pad(&input, &[(1, 0)], 0.0).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
