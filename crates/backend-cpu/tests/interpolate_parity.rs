//! `CpuBackendOps::interpolate`（イシュー #1757）の受け入れ条件
//! テスト。
//!
//! `fandhe_ai_autodiff::eval::interpolate_nearest`（ホスト参照実装。
//! `autodiff` クレート非公開のため `backend-cpu` から直接は呼べない）
//! と CPU ネイティブ実装（本クレート `interpolate` モジュール）が
//! 同一の添字式（`src = (dst * in) / out`）であることを、
//! `Var::interpolate` の 2 系統実行経路を間接的に突き合わせて確認する
//! （`gather_scatter_parity.rs` と同型の `ForceEvalFallback` 方針）。
//! 両経路の出力が `f32::to_bits()` で bit 完全一致することを固定し、
//! 添字式が forward／backward 双方で乖離しないことを検知する
//! （`interpolate` は算術を含まない純粋なコピー演算のため、正しい
//! 実装同士は常に bit 完全一致するはず）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, InterpolateMode, Tensor};

/// `CpuBackendOps` の必須メソッドへ委譲しつつ、`interpolate` だけは
/// 意図的に override せずデフォルト（`Unsupported`）のまま残す
/// ラッパー（`gather_scatter_parity.rs::ForceEvalFallback` と同型）。
/// `Var::interpolate` を `autodiff::eval::interpolate_nearest`
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
    // `interpolate` はデフォルト実装（`Unsupported`）のまま override
    // しない。これが `eval::` フォールバック経路の唯一の実現手段。
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

/// `interpolate`（1-D アップサンプル）の CPU ネイティブ実装と
/// `eval::interpolate_nearest` フォールバックが bit 完全一致する
/// ことを確認する。
#[test]
fn interpolate_nearest_upsample_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![1.0, 2.0, 3.0], &[3]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x
        .interpolate(&[7], InterpolateMode::Nearest)
        .unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x
        .interpolate(&[7], InterpolateMode::Nearest)
        .unwrap();

    assert_bit_exact(
        "interpolate(upsample)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
}

/// `interpolate`（非整数比ダウンサンプル・2-D 先頭軸付き）の CPU
/// ネイティブ実装と `eval::interpolate_nearest` フォールバックが
/// bit 完全一致することを確認する。
#[test]
fn interpolate_nearest_downsample_2d_native_matches_eval_fallback_bit_exact() {
    let x = t((1..=16).map(|v| v as f32).collect(), &[2, 8]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x
        .interpolate(&[3], InterpolateMode::Nearest)
        .unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x
        .interpolate(&[3], InterpolateMode::Nearest)
        .unwrap();

    assert_bit_exact(
        "interpolate(downsample 2d)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
}

/// `interpolate` の backward（scatter_add ベース VJP）が CPU
/// ネイティブ実装経路と `eval::` フォールバック経路で bit 完全一致
/// することを確認する（forward だけでなく VJP の add アキュムレータ
/// も両経路一致することが焦点）。
#[test]
fn interpolate_nearest_backward_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![1.0, 2.0, 3.0], &[3]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x
        .interpolate(&[7], InterpolateMode::Nearest)
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
        .interpolate(&[7], InterpolateMode::Nearest)
        .unwrap();
    let fallback_loss = fallback_out.sum(None).unwrap();
    let fallback_grads = fallback_tape.backward(&fallback_loss).unwrap();
    let fallback_dx = fallback_grads
        .get(&fallback_x)
        .unwrap()
        .expect("x は loss に到達する");

    assert_bit_exact("interpolate backward dX", native_dx, fallback_dx);
}

/// `CpuBackendOps::interpolate` が `interpolate_out_shape` による
/// shape 再検査を実装側でも行い、不一致を `BackendError::
/// ShapeMismatch` として fail-closed に拒否することを確認する
/// （`.claude/rules/security.md` A08。トレイトメソッドを直接呼び
/// `Var` 側の検査を経由しない経路を対象とする）。
#[test]
fn backend_ops_interpolate_rejects_zero_spatial_axis() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let err = ops
        .interpolate(&input, &[0], InterpolateMode::Nearest)
        .unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

// --- bilinear（イシュー #1762） ---
//
// 受入契約は REQ-2 統一複合判定（`assert_parity`）——`Bilinear` は
// 算術を含むため `Nearest` のような bit 完全一致は前提としない
// （`InterpolateMode::Bilinear` doc 参照）。ただし CPU ネイティブ実装
// とホスト参照実装（`eval::interpolate_bilinear`）は**同じ Rust 関数
// 呼び出し**（`fandhe_ai_tensor_core::interpolate` の単一情報源）を
// 経由するため実際には bit 完全一致する——本テストはその実測を
// `assert_parity` で緩やかに、`assert_eq!` で厳密に二重に検証する。

/// `interpolate`（bilinear・`align_corners=false`）の CPU ネイティブ
/// 実装と `eval::interpolate_bilinear` フォールバックの parity。
#[test]
fn interpolate_bilinear_native_matches_eval_fallback_align_corners_false() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mode = InterpolateMode::Bilinear {
        align_corners: false,
    };

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.interpolate(&[5, 5], mode).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x.interpolate(&[5, 5], mode).unwrap();

    assert_bit_exact(
        "interpolate bilinear(align_corners=false)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    let a = dense_vec(&native_out.to_tensor());
    let b = dense_vec(&fallback_out.to_tensor());
    fandhe_ai_backend_cpu::parity::assert_parity(
        "interpolate bilinear(align_corners=false) native vs eval fallback",
        &a,
        &b,
    );
}

/// `interpolate`（bilinear・`align_corners=true`）の CPU ネイティブ
/// 実装と `eval::interpolate_bilinear` フォールバックの parity。
#[test]
fn interpolate_bilinear_native_matches_eval_fallback_align_corners_true() {
    let x = t((1..=25).map(|v| v as f32 * 0.37).collect(), &[5, 5]);
    let mode = InterpolateMode::Bilinear {
        align_corners: true,
    };

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.interpolate(&[2, 9], mode).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_out = fallback_x.interpolate(&[2, 9], mode).unwrap();

    assert_bit_exact(
        "interpolate bilinear(align_corners=true)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
}

/// `interpolate` backward（bilinear。4 近傍重み付き scatter_add ベース
/// VJP）の CPU ネイティブ実装経路と `eval::` フォールバック経路の
/// parity。
#[test]
fn interpolate_bilinear_backward_native_matches_eval_fallback() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mode = InterpolateMode::Bilinear {
        align_corners: false,
    };

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_out = native_x.interpolate(&[5, 5], mode).unwrap();
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
    let fallback_out = fallback_x.interpolate(&[5, 5], mode).unwrap();
    let fallback_loss = fallback_out.sum(None).unwrap();
    let fallback_grads = fallback_tape.backward(&fallback_loss).unwrap();
    let fallback_dx = fallback_grads
        .get(&fallback_x)
        .unwrap()
        .expect("x は loss に到達する");

    assert_bit_exact("interpolate bilinear backward dX", native_dx, fallback_dx);
}

/// `CpuBackendOps::interpolate`（bilinear）が `size.len() != 2` を
/// `BackendError::ShapeMismatch` として fail-closed に拒否することを
/// 確認する（`interpolate_out_shape_for_mode` の追加検査。トレイト
/// メソッドを直接呼び `Var` 側の検査を経由しない経路を対象とする）。
#[test]
fn backend_ops_interpolate_bilinear_rejects_size_len_other_than_two() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let mode = InterpolateMode::Bilinear {
        align_corners: false,
    };
    let err = ops.interpolate(&input, &[4], mode).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
    let err2 = ops.interpolate(&input, &[4, 4, 4], mode).unwrap_err();
    assert!(matches!(err2, BackendError::ShapeMismatch(_)));
}
