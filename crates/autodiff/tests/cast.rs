//! `Var::cast`／`Var::to_f32`／`Tape::var_from`（イシュー #1750）の
//! 勾配契約・フォールバック規則・エラー伝播を検証する統合テスト。
//!
//! `unique_backend_parity.rs`／`mse_loss_fusion.rs` と同型の構成:
//! - 属性なし・`common::NaiveOps`（`cast_ops` accessor 既定 `None`）:
//!   ホスト参照実装へのフォールバック経路が実際に機能することを確認する。
//! - ローカルフィクスチャ（`CastOps` を `Some` で返すが個別方向が
//!   `Unsupported`／別エラー／不正 shape を返す）: フォールバック規則・
//!   エラー伝播・shape 事後検査を検証する（判定迂回経路を作らない。
//!   `.claude/rules/security.md` A08）。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, CastOps, Device, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- (b) 値・勾配契約 ---

/// `x.cast::<f64>()` の呼び出しが `y = x·x` の backward（`x` の勾配）に
/// 影響しないこと（非微分・detached な副作用なしの読み出し）を確認
/// する。
#[test]
fn cast_call_does_not_affect_unrelated_backward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 3.0, -1.0], &[3]));
    let y = x.mul(&x).unwrap();
    let loss = y.sum(None).unwrap();
    let grads_without_cast = tape.backward(&loss).unwrap();
    let dx_without_cast = grads_without_cast
        .get(&x)
        .expect("test fixture: get 自体は Err にならないはず")
        .expect("test fixture: x の勾配は存在するはず")
        .host_slice()
        .into_owned();

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&t(vec![2.0, 3.0, -1.0], &[3]));
    let _unused: Tensor<f64> = x2.cast().unwrap();
    let y2 = x2.mul(&x2).unwrap();
    let loss2 = y2.sum(None).unwrap();
    let grads_with_cast = tape2.backward(&loss2).unwrap();
    let dx_with_cast = grads_with_cast
        .get(&x2)
        .expect("test fixture: get 自体は Err にならないはず")
        .expect("test fixture: x2 の勾配は存在するはず")
        .host_slice()
        .into_owned();

    assert_eq!(
        dx_without_cast, dx_with_cast,
        "cast 呼び出しの有無で他の勾配計算が変化してはならない"
    );
}

/// `Var::to_f32()` が同一ノードを指す恒等射であり、backward で通常
/// どおり勾配が届くことを確認する。
#[test]
fn to_f32_receives_gradient_like_original_var() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = x.to_f32();
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads
        .get(&x)
        .expect("test fixture: get 自体は Err にならないはず")
        .expect("test fixture: x の勾配は存在するはず")
        .host_slice()
        .into_owned();
    assert_eq!(dx, vec![1.0, 1.0, 1.0], "sum の勾配は全要素 1 のはず");
}

/// `Tape::var_from(&Tensor<i32>)` で登録した葉に対する `sum().backward()`
/// が解析的に全要素 1 の勾配を返すことを確認する（`var_from` が
/// `Tape::var` と同じ葉ノード契約を持つことの直接検証）。
#[test]
fn var_from_leaf_receives_analytic_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape
        .var_from(&Tensor::new(vec![1i32, 2, 3, 4], &[4]).unwrap())
        .unwrap();
    let loss = x.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads
        .get(&x)
        .expect("test fixture: get 自体は Err にならないはず")
        .expect("test fixture: x の勾配は存在するはず")
        .host_slice()
        .into_owned();
    assert_eq!(dx, vec![1.0, 1.0, 1.0, 1.0]);
}

/// `Tape::var_from(&Tensor<bool>)`／`<i64>`／`<f64>` の値が
/// `tensor_core::cast::cast_to_f32` のホスト参照実装と bit 一致する
/// ことを確認する（`unique_backend_parity.rs` と同じ bit 比較方式）。
#[test]
fn var_from_values_match_host_reference_for_bool_i64_f64() {
    let tape = Tape::new_with_ops(common::naive_ops());

    let bool_in = Tensor::new(vec![true, false, true], &[3]).unwrap();
    let bool_var = tape.var_from(&bool_in).unwrap();
    let bool_ref = fandhe_ai_tensor_core::cast_to_f32(&bool_in).unwrap();
    assert_eq!(
        bool_var.to_tensor().host_slice().into_owned(),
        bool_ref.host_slice().into_owned()
    );

    let i64_in = Tensor::new(vec![i64::MIN, 0, i64::MAX], &[3]).unwrap();
    let i64_var = tape.var_from(&i64_in).unwrap();
    let i64_ref = fandhe_ai_tensor_core::cast_to_f32(&i64_in).unwrap();
    assert_eq!(
        i64_var.to_tensor().host_slice().into_owned(),
        i64_ref.host_slice().into_owned()
    );

    let f64_in = Tensor::new(vec![1.5f64, -2.5, f64::MAX], &[3]).unwrap();
    let f64_var = tape.var_from(&f64_in).unwrap();
    let f64_ref = fandhe_ai_tensor_core::cast_to_f32(&f64_in).unwrap();
    assert_eq!(
        f64_var.to_tensor().host_slice().into_owned(),
        f64_ref.host_slice().into_owned()
    );
}

// --- (c) フォールバック規則・エラー伝播 ---

/// `common::NaiveOps`（`cast_ops` accessor 既定 `None`）でも
/// `Var::cast`／`Tape::var_from` がホスト参照実装へフォールバックして
/// 正しく動作することを確認する。
#[test]
fn naive_ops_without_cast_ops_falls_back_to_host_reference() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.9, -1.9, f32::NAN], &[3]));
    let out: Tensor<i32> = x.cast().unwrap();
    assert_eq!(out.host_slice().into_owned(), vec![1, -1, 0]);
}

/// [`CastOps`] を `Some` で返すが全方向が既定実装（`Unsupported`）の
/// ままの `BackendOps` フィクスチャ。`Var::cast` がホスト参照実装へ
/// フォールバックすることを確認する（`MockOps`／`NaiveMockOps` と同型の
/// 最小実装ダミー。`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／
/// `max` はいずれも到達しないため `Unsupported` 固定でよい）。
struct AlwaysUnsupportedCastOps;
impl CastOps for AlwaysUnsupportedCastOps {}

struct OpsWithUnsupportedCast;

impl BackendOps for OpsWithUnsupportedCast {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn cast_ops(&self) -> Option<&dyn CastOps> {
        Some(&AlwaysUnsupportedCastOps)
    }
    fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: gemm".into()))
    }
    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: add".into()))
    }
    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: mul".into()))
    }
    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: relu".into()))
    }
    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: exp".into()))
    }
    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: tanh".into()))
    }
    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: sum".into()))
    }
    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: max".into()))
    }
}

#[test]
fn cast_ops_some_but_unsupported_falls_back_to_host_reference() {
    let tape = Tape::new_with_ops(Box::new(OpsWithUnsupportedCast));
    let x = tape.var(&t(vec![1.9, -1.9, f32::NAN], &[3]));
    let out: Tensor<i32> = x.cast().unwrap();
    assert_eq!(out.host_slice().into_owned(), vec![1, -1, 0]);
}

/// 個別方向が `Unsupported` 以外のエラーを返す `CastOps` フィクスチャ。
/// `Var::cast` がフォールバックせずそのままエラーを伝播することを
/// 確認する（`mse_loss_fusion.rs::*_error_other_than_unsupported_
/// propagates` と同型）。
struct OpsWithFailingCast;

impl BackendOps for OpsWithFailingCast {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn cast_ops(&self) -> Option<&dyn CastOps> {
        Some(&FailingCastOpsImpl)
    }
    fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: gemm".into()))
    }
    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: add".into()))
    }
    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: mul".into()))
    }
    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: relu".into()))
    }
    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: exp".into()))
    }
    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: tanh".into()))
    }
    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: sum".into()))
    }
    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: max".into()))
    }
}

/// `cast_f32_to_i32` が明示的に非 `Unsupported` エラーを返す `CastOps`
/// フィクスチャ（`AlwaysFailingCastOps` は説明用の未使用スタブとして
/// 残し、実際に使うのは本 struct）。
struct FailingCastOpsImpl;
impl CastOps for FailingCastOpsImpl {
    fn cast_f32_to_i32(&self, x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> {
        // shape 不一致エラーで「Unsupported 以外」を模す。
        let _ = x;
        Err(BackendError::ShapeMismatch(
            fandhe_ai_tensor_core::ShapeError::ElementCountOverflow,
        ))
    }
}

#[test]
fn cast_ops_error_other_than_unsupported_propagates() {
    let tape = Tape::new_with_ops(Box::new(OpsWithFailingCast));
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err: AutodiffError = x.cast::<i32>().unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::ShapeMismatch(_))
    ));
}

/// `CastOps` が不正 shape（入力と異なる shape）を返した場合、
/// `Var::cast` が `ShapeMismatch` で fail-closed に拒否することを
/// 確認する（`argext_with_fallback`／`one_hot_with_fallback` と同型の
/// 事後 shape 検査）。
struct WrongShapeCastOps;
impl CastOps for WrongShapeCastOps {
    fn cast_f32_to_i32(&self, _x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> {
        // 常に shape [1] を返す（入力 shape を無視した不正な実装）。
        Ok(Tensor::new(vec![0i32], &[1]).unwrap())
    }
}

struct OpsWithWrongShapeCast;

impl BackendOps for OpsWithWrongShapeCast {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn cast_ops(&self) -> Option<&dyn CastOps> {
        Some(&WrongShapeCastOps)
    }
    fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: gemm".into()))
    }
    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: add".into()))
    }
    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: mul".into()))
    }
    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: relu".into()))
    }
    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: exp".into()))
    }
    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: tanh".into()))
    }
    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: sum".into()))
    }
    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("mock: max".into()))
    }
}

#[test]
fn cast_ops_wrong_output_shape_is_rejected() {
    let tape = Tape::new_with_ops(Box::new(OpsWithWrongShapeCast));
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.cast::<i32>().unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::ShapeMismatch(_))
    ));
}
