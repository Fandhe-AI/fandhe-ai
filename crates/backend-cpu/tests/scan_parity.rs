//! `CpuBackendOps::cumsum`／`cumprod`（イシュー #1731）の受け入れ条件
//! テスト。
//!
//! `gather_scatter_parity.rs` と同じ [`ForceEvalFallback`] 方式で、
//! CPU ネイティブ実装（本クレート `scan` モジュール）と
//! `fandhe_ai_autodiff::eval::cumsum_along`／`cumprod_along`（ホスト
//! フォールバック。`autodiff` クレート非公開のため `backend-cpu` から
//! 直接は呼べない）の 2 経路が bit 完全一致することを、`Var::cumsum`／
//! `cumprod` を経由して間接的に確認する。あわせて run-to-run 決定性
//! （同一入力で 2 回独立に呼び出しても bit 一致）・軸範囲外拒否・
//! 空 shape 契約も検証する。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

/// `CpuBackendOps` の必須メソッド（デフォルト実装を持たない 9 個）へ
/// 委譲しつつ、`cumsum`／`cumprod` だけは意図的に override せずデフォルト
/// （`Unsupported`）のまま残すラッパー（`gather_scatter_parity.rs::
/// ForceEvalFallback` と同型）。`Var::cumsum`／`cumprod` を
/// `autodiff::eval::cumsum_along`／`cumprod_along`（ホストフォール
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
    // `cumsum`／`cumprod` はデフォルト実装（`Unsupported`）のまま
    // override しない。これが `eval::` フォールバック経路の唯一の
    // 実現手段。
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

/// 決定的な固定入力（乱数クレート非依存。`.claude/rules/deps-policy.md`
/// に乱数クレートの許容区分がないため固定値配列で代替する）。
fn fixture_4x5() -> Tensor<f32> {
    t(
        vec![
            0.7, -1.3, 2.2, -0.4, 3.1, -2.5, 0.9, -0.1, 1.8, -3.3, 0.05, -0.6, 2.9, -1.1, 0.4, 1.6,
            -2.2, 0.3, -0.8, 2.4,
        ],
        &[4, 5],
    )
}

/// `cumsum` の CPU ネイティブ実装と `eval::cumsum_along` フォール
/// バックが両軸で bit 完全一致することを確認する。
#[test]
fn cumsum_native_matches_eval_fallback_bit_exact() {
    let x = fixture_4x5();
    for dim in 0..2 {
        let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let native_x = native_tape.var(&x);
        let native_out = native_x.cumsum(dim).unwrap();

        let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
            inner: CpuBackendOps::new(),
        }));
        let fallback_x = fallback_tape.var(&x);
        let fallback_out = fallback_x.cumsum(dim).unwrap();

        assert_bit_exact(
            &format!("cumsum dim={dim}"),
            &native_out.to_tensor(),
            &fallback_out.to_tensor(),
        );
    }
}

/// `cumprod` の CPU ネイティブ実装と `eval::cumprod_along` フォール
/// バックが両軸で bit 完全一致することを確認する。
#[test]
fn cumprod_native_matches_eval_fallback_bit_exact() {
    let x = fixture_4x5();
    for dim in 0..2 {
        let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let native_x = native_tape.var(&x);
        let native_out = native_x.cumprod(dim).unwrap();

        let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
            inner: CpuBackendOps::new(),
        }));
        let fallback_x = fallback_tape.var(&x);
        let fallback_out = fallback_x.cumprod(dim).unwrap();

        assert_bit_exact(
            &format!("cumprod dim={dim}"),
            &native_out.to_tensor(),
            &fallback_out.to_tensor(),
        );
    }
}

/// transpose view（strided 入力）でも `cumsum` ネイティブ実装が
/// フォールバックと bit 一致することを確認する。
#[test]
fn cumsum_transposed_view_native_matches_eval_fallback_bit_exact() {
    let x = fixture_4x5();

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_tr = native_x.transpose(0, 1).unwrap();
    let native_out = native_tr.cumsum(0).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_tr = fallback_x.transpose(0, 1).unwrap();
    let fallback_out = fallback_tr.cumsum(0).unwrap();

    assert_bit_exact(
        "cumsum transposed",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
}

/// `cumsum`／`cumprod` が run-to-run（同一プロセス内で 2 回独立に
/// 呼び出す）で bit 同一の決定的結果を返すことを確認する
/// （データ依存分岐のない逐次実装であることの直接検証）。
#[test]
fn cumsum_and_cumprod_are_run_to_run_bit_identical() {
    let x = fixture_4x5();
    let ops = CpuBackendOps::new();

    let a1 = BackendOps::cumsum(&ops, &x, 1).unwrap();
    let a2 = BackendOps::cumsum(&ops, &x, 1).unwrap();
    assert_bit_exact("cumsum run-to-run", &a1, &a2);

    let b1 = BackendOps::cumprod(&ops, &x, 1).unwrap();
    let b2 = BackendOps::cumprod(&ops, &x, 1).unwrap();
    assert_bit_exact("cumprod run-to-run", &b1, &b2);
}

/// `dim` が範囲外なら `CpuBackendOps::cumsum`／`cumprod` は
/// `BackendError::ShapeMismatch(AxisOutOfRange)` を返す（`Var` を
/// 経由しない直接呼び出しからの誤動作を防ぐ独立検査。
/// `.claude/rules/security.md` A08）。
#[test]
fn backend_ops_cumsum_and_cumprod_reject_axis_out_of_range() {
    let ops = CpuBackendOps::new();
    let x = t(vec![1.0, 2.0, 3.0], &[3]);

    let err = BackendOps::cumsum(&ops, &x, 1).unwrap_err();
    assert!(matches!(
        err,
        BackendError::ShapeMismatch(ShapeError::AxisOutOfRange { .. })
    ));

    let err = BackendOps::cumprod(&ops, &x, 1).unwrap_err();
    assert!(matches!(
        err,
        BackendError::ShapeMismatch(ShapeError::AxisOutOfRange { .. })
    ));
}

/// 3 次元・中間軸縮約（`outer > 1` かつ `inner > 1` が同時に成立する
/// shape `[2, 3, 4]`・`dim=1`）でも `cumsum`／`cumprod` の CPU ネイティブ
/// 実装が `eval::` フォールバックと bit 完全一致することを確認する
/// （レビュー指摘: 既存の他テストは rank ≤ 2 の形状に限られ、outer/
/// inner が両方 1 より大きいケースを直接検証していなかった。イシュー
/// #1731）。
#[test]
fn cumsum_and_cumprod_native_matches_eval_fallback_bit_exact_3d_middle_axis() {
    let x = t(
        vec![
            0.7, -1.3, 2.2, -0.4, 3.1, -2.5, 0.9, -0.1, 1.8, -3.3, 0.05, -0.6, 2.9, -1.1, 0.4, 1.6,
            -2.2, 0.3, -0.8, 2.4, 1.1, -0.2, 0.6, -1.9,
        ],
        &[2, 3, 4],
    );
    let dim = 1;

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_cumsum = native_x.cumsum(dim).unwrap();
    let native_cumprod = native_x.cumprod(dim).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_cumsum = fallback_x.cumsum(dim).unwrap();
    let fallback_cumprod = fallback_x.cumprod(dim).unwrap();

    assert_bit_exact(
        "cumsum 3d middle axis",
        &native_cumsum.to_tensor(),
        &fallback_cumsum.to_tensor(),
    );
    assert_bit_exact(
        "cumprod 3d middle axis",
        &native_cumprod.to_tensor(),
        &fallback_cumprod.to_tensor(),
    );
}

/// 空 shape（`[0, 3]`・`[2, 0]`）に対して `cumsum` が空出力を返す
/// ことを確認する（部分積オーバーフロー回避の早期 return）。
#[test]
fn cumsum_on_empty_shapes_returns_empty() {
    let ops = CpuBackendOps::new();

    let x1 = Tensor::<f32>::new(Vec::new(), &[0, 3]).unwrap();
    let out1 = BackendOps::cumsum(&ops, &x1, 0).unwrap();
    assert_eq!(out1.shape(), &[0, 3]);
    assert_eq!(dense_vec(&out1), Vec::<f32>::new());

    let x2 = Tensor::<f32>::new(Vec::new(), &[2, 0]).unwrap();
    let out2 = BackendOps::cumsum(&ops, &x2, 1).unwrap();
    assert_eq!(out2.shape(), &[2, 0]);
    assert_eq!(dense_vec(&out2), Vec::<f32>::new());
}
