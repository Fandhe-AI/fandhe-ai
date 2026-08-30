//! `Var::mse_loss_with`／`grad::vjp` の `Op::MseLoss` 分岐が、
//! `BackendOps::mse_loss`／`mse_loss_backward`（イシュー #1045・親イシュー
//! #1043）の融合カーネルへ透過的に切り替わることの直接検証
//! （`fusion_backend_integration.rs` の `CountingFusedOps` と同型の
//! カウンタ付きフィクスチャ方式）。
//!
//! `autodiff` は具体バックエンドクレート（`backend-cpu` 等）へ依存しない
//! 設計上の不変条件（`docs/fusion-graph-design.md` §3.4）があるため、
//! `backend-cpu` を dev-dependency に追加せず、`common::NaiveOps` を
//! 委譲先として `mse_loss`／`mse_loss_backward` のみをオーバーライドする
//! フィクスチャで代替する（実装計画 §3 対象ファイル一覧の代替方針）。
//!
//! 4 点を固定する:
//! 1. 融合カーネル実装ありの `BackendOps` では forward/backward とも
//!    実際にそのメソッドが呼ばれ（`Arc<AtomicUsize>` カウンタ）、
//!    フォールバック経路（`eval::mse_loss`／`mse_loss_vjp`）と数値一致
//!    複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致する。
//! 2. 常時 `Unsupported` を返すフィクスチャ（融合カーネル未実装
//!    バックエンド相当）ではフォールバックが働き、解析値と一致する。
//! 3. `Unsupported` 以外のエラー（融合カーネルの実行時失敗を模す）は
//!    フォールバックせずそのまま伝播する（判定迂回経路を作らない。
//!    `.claude/rules/security.md` A08）。forward・backward 双方で確認する。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, MseReduction, Tensor};

/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）。`fandhe_ai_backend_cpu::parity::
/// assert_parity` を dev-dependency 追加なしに再利用できないため、
/// 同一判定式をテストローカルに再実装する（`sgd_device_parity.rs` の
/// `assert_close` と同方針）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

/// `common::NaiveOps` に委譲しつつ `mse_loss`／`mse_loss_backward` のみを
/// 素朴な参照実装でオーバーライドし、呼び出し回数を `Arc<AtomicUsize>`
/// （`Tape::new_with_ops` が `Box` の所有権を奪うため、呼び出し後も外部
/// から読める共有カウンタが必要。`CountingFusedOps` と同じ制約）で記録
/// するフィクスチャ。数式は `fandhe_ai_backend_cpu::mse` の CPU 融合
/// カーネルと同一だが独立実装であり、「融合経路が正しく呼ばれ、正しい
/// 値を返すこと」を実装の重複なしに検証する。
struct CountingMseOps {
    inner: Box<dyn BackendOps + Send>,
    forward_calls: Arc<AtomicUsize>,
    backward_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingMseOps {
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

    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        self.forward_calls.fetch_add(1, Ordering::SeqCst);
        let p = pred.contiguous();
        let tt = target.contiguous();
        let pd = p.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let numel = pd.len();
        if numel == 0 {
            return Tensor::new(vec![0.0], &[]).map_err(BackendError::ShapeMismatch);
        }
        let sum_sq: f32 = pd
            .iter()
            .zip(td.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum();
        let value = match reduction {
            MseReduction::Mean => sum_sq / numel as f32,
            MseReduction::Sum => sum_sq,
            _ => sum_sq,
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    fn mse_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.backward_calls.fetch_add(1, Ordering::SeqCst);
        let p = pred.contiguous();
        let tt = target.contiguous();
        let pd = p.as_slice().unwrap_or(&[]);
        let td = tt.as_slice().unwrap_or(&[]);
        let dpred: Vec<f32> = pd
            .iter()
            .zip(td.iter())
            .map(|(&x, &y)| scale * (x - y))
            .collect();
        Tensor::new(dpred, pred.shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// 常時 `Unsupported` を返すフィクスチャ（融合カーネル未実装バックエンド
/// 相当。`mse_loss`／`mse_loss_backward` を何もオーバーライドせず既定
/// 実装〈fail-safe `Unsupported`〉のまま使う）。
struct AlwaysUnsupportedMseOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysUnsupportedMseOps {
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

/// `Op::MseLoss` の forward・backward の一方で `Unsupported` 以外の
/// エラーを常に返すフィクスチャ（融合カーネルの実行時失敗を模す。
/// フォールバックせず伝播することを確認する対象）。`fail_forward` で
/// どちらを失敗させるかを切り替える（もう一方は `Unsupported` を返し
/// 通常どおりフォールバックさせる）。
struct AlwaysFailingMseOps {
    inner: Box<dyn BackendOps + Send>,
    fail_forward: bool,
}

impl BackendOps for AlwaysFailingMseOps {
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
    fn mse_loss(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingMseOps: simulated fused MSE forward failure (not Unsupported)".into(),
            ))
        } else {
            Err(BackendError::Unsupported("forward not under test".into()))
        }
    }
    fn mse_loss_backward(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        if self.fail_forward {
            Err(BackendError::Unsupported("backward not under test".into()))
        } else {
            Err(BackendError::KernelLaunchFailed(
                "AlwaysFailingMseOps: simulated fused MSE backward failure (not Unsupported)"
                    .into(),
            ))
        }
    }
}

#[test]
fn fused_mse_loss_is_invoked_and_matches_fallback() {
    let pred_data = vec![1.0, -2.0, 3.0, 0.5];
    let target_data = vec![0.5, -1.0, 2.5, 1.0];

    for reduction in [Reduction::Mean, Reduction::Sum] {
        let forward_calls = Arc::new(AtomicUsize::new(0));
        let backward_calls = Arc::new(AtomicUsize::new(0));
        let fused_ops = CountingMseOps {
            inner: common::naive_ops(),
            forward_calls: forward_calls.clone(),
            backward_calls: backward_calls.clone(),
        };
        let tape_fused = Tape::new_with_ops(Box::new(fused_ops));
        let pred_fused = tape_fused.var(&t(pred_data.clone(), &[2, 2]));
        let target_fused = tape_fused.var(&t(target_data.clone(), &[2, 2]));
        let loss_fused = pred_fused.mse_loss_with(&target_fused, reduction).unwrap();
        assert_eq!(
            forward_calls.load(Ordering::SeqCst),
            1,
            "{reduction:?}: mse_loss は forward で 1 回呼ばれる契約"
        );
        let value_fused = scalar(&loss_fused.to_tensor());
        let grads_fused = tape_fused.backward(&loss_fused).unwrap();
        assert_eq!(
            backward_calls.load(Ordering::SeqCst),
            1,
            "{reduction:?}: mse_loss_backward は backward で 1 回呼ばれる契約"
        );
        let dpred_fused = grads_fused
            .get(&pred_fused)
            .unwrap()
            .expect("pred は loss に到達する")
            .clone();

        // フォールバック経路（`NaiveOps`。既定 `Unsupported`）との突合。
        let tape_fallback = Tape::new_with_ops(common::naive_ops());
        let pred_fallback = tape_fallback.var(&t(pred_data.clone(), &[2, 2]));
        let target_fallback = tape_fallback.var(&t(target_data.clone(), &[2, 2]));
        let loss_fallback = pred_fallback
            .mse_loss_with(&target_fallback, reduction)
            .unwrap();
        let value_fallback = scalar(&loss_fallback.to_tensor());
        let grads_fallback = tape_fallback.backward(&loss_fallback).unwrap();
        let dpred_fallback = grads_fallback
            .get(&pred_fallback)
            .unwrap()
            .expect("pred は loss に到達する");

        assert_close(
            value_fused,
            value_fallback,
            &format!("mse_loss forward ({reduction:?})"),
        );
        for i in 0..2 {
            for j in 0..2 {
                let a = dpred_fused.get(&[i, j]).unwrap();
                let e = dpred_fallback.get(&[i, j]).unwrap();
                assert_close(a, e, &format!("mse_loss backward ({reduction:?})[{i},{j}]"));
            }
        }
    }
}

#[test]
fn fused_mse_loss_forward_falls_back_when_unsupported() {
    let ops = AlwaysUnsupportedMseOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let pred = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&t(vec![0.5, -1.0, 2.5, 1.0], &[2, 2]));
    let loss = pred.mse_loss_with(&target, Reduction::Mean).unwrap();
    // フォールバック（`eval::mse_loss`）の解析値: diff=[0.5,-1.0,0.5,-0.5]
    // → sq=[0.25,1.0,0.25,0.25] → mean=1.75/4=0.4375。
    assert_close(scalar(&loss.to_tensor()), 0.4375, "fallback forward mean");

    let grads = tape.backward(&loss).unwrap();
    let dpred = grads.get(&pred).unwrap().expect("到達する");
    // scale = 1*2/4 = 0.5 → dpred = 0.5*diff = [0.25,-0.5,0.25,-0.25]
    let expected = [0.25f32, -0.5, 0.25, -0.25];
    for (idx, &e) in expected.iter().enumerate() {
        let i = idx / 2;
        let j = idx % 2;
        assert_close(
            dpred.get(&[i, j]).unwrap(),
            e,
            &format!("fallback backward [{i},{j}]"),
        );
    }
}

#[test]
fn fused_mse_loss_forward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingMseOps {
        inner: common::naive_ops(),
        fail_forward: true,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));

    let result = pred.mse_loss_with(&target, Reduction::Mean);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

#[test]
fn fused_mse_loss_backward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingMseOps {
        inner: common::naive_ops(),
        fail_forward: false,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let pred = tape.var(&t(vec![1.0, 2.0], &[2]));
    let target = tape.var(&t(vec![0.0, 0.0], &[2]));
    // forward は `Unsupported` を返す設定（`fail_forward: false`）なので
    // 従来の `eval::mse_loss` へフォールバックして成功する。
    let loss = pred.mse_loss_with(&target, Reduction::Mean).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}
