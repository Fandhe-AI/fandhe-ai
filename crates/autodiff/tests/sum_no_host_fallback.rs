//! `Var::sum` がホスト参照実装へフォールバックしない契約を固定する
//! 統合テスト（`docs/backend-metal-reduce-sum-design.md` §10 案 C。
//! 2026-09-17 ユーザー承認済み・イシュー #1932）。
//!
//! `BackendOps::sum` が `BackendError::Unsupported` を返す最小実装ダミー
//! （`cast.rs::OpsWithUnsupportedCast` と同型）上で `Var::sum(None)`／
//! `Var::sum(Some(dim))` を呼び、`cumsum`（`Unsupported` 時のみ
//! `eval::cumsum_along` へフォールバック）とは対照的に
//! `AutodiffError::Backend(BackendError::Unsupported(_))` がそのまま
//! 伝播することを確認する。ホスト側 `eval::sum`（素の f32 逐次和）は
//! CPU 参照実装（f64 チャンク 2 段）と bit 一致しないため、silent
//! fallback を許すと `mean`／`var`／`std` 等 `sum` に依存する演算へ数値
//! 方式の非一貫性が波及する（`.claude/rules/security.md` A08「判定の
//! 迂回経路を作らない」）。本テストは判定迂回経路が将来混入した場合の
//! 検知が目的で、CPU／CUDA／Metal の本番 `sum` 実装には触れない。

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `sum` を含む全演算が `Unsupported` を返す最小実装ダミー。`Var::sum`
/// は `gemm`／`add` 等へ到達しないため固定値でよい。
struct OpsWithUnsupportedSum;

impl BackendOps for OpsWithUnsupportedSum {
    fn device(&self) -> Device {
        Device::Cpu
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

fn assert_unsupported_propagated(result: Result<impl Sized, AutodiffError>, label: &str) {
    match result {
        Err(AutodiffError::Backend(BackendError::Unsupported(msg))) => {
            assert_eq!(
                msg, "mock: sum",
                "{label}: 伝播したメッセージが BackendOps::sum 由来ではない"
            );
        }
        Err(other) => panic!("{label}: Unsupported 以外のエラーへ変換された: {other:?}"),
        Ok(_) => panic!("{label}: ホストへフォールバックして成功してしまった（案 C 違反）"),
    }
}

#[test]
fn sum_all_propagates_backend_unsupported_without_host_fallback() {
    let tape = Tape::new_with_ops(Box::new(OpsWithUnsupportedSum));
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert_unsupported_propagated(x.sum(None), "sum(None)");
}

#[test]
fn sum_axis_propagates_backend_unsupported_without_host_fallback() {
    let tape = Tape::new_with_ops(Box::new(OpsWithUnsupportedSum));
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    assert_unsupported_propagated(x.sum(Some(0)), "sum(Some(0))");
    assert_unsupported_propagated(x.sum(Some(1)), "sum(Some(1))");
}

/// 範囲外 `dim` はデバイス（`BackendOps::sum`）へ到達する前に
/// `Var::sum` 側の shape 検査で拒否される（`Unsupported` ではない）
/// ことを合わせて固定する。フォールバック規律とは独立の既存契約。
#[test]
fn sum_out_of_range_dim_is_rejected_before_backend_dispatch() {
    let tape = Tape::new_with_ops(Box::new(OpsWithUnsupportedSum));
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    match x.sum(Some(1)) {
        Err(AutodiffError::Backend(BackendError::Unsupported(_))) => {
            panic!("範囲外 dim が BackendOps::sum まで到達した")
        }
        Err(_) => {}
        Ok(_) => panic!("範囲外 dim が受理された"),
    }
}
