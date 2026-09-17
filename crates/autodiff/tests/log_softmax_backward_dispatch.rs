//! `Op::LogSoftmax` の VJP（`grad::vjp`）が `BackendOps::
//! log_softmax_backward`（イシュー #1949・親 #1947「GPU ホスト
//! フォールバック残存演算の専用カーネル化」）へ透過的に切り替わる
//! ことの直接検証（`nll_kl_div_loss_fusion.rs` と同型のカウンタ付き
//! フィクスチャ方式）。
//!
//! `autodiff` は具体バックエンドクレート（`backend-cpu` 等）へ依存
//! しない設計上の不変条件（`docs/fusion-graph-design.md` §3.4）が
//! あるため、`backend-cpu` を dev-dependency に追加せず、
//! `common::NaiveOps` を委譲先として `log_softmax_backward` のみを
//! オーバーライドするフィクスチャで代替する。
//!
//! 4 点を固定する:
//! 1. `Unsupported` を返さないフィクスチャでは backward で実際に
//!    そのメソッドが呼ばれ（`Arc<AtomicUsize>` カウンタ）、返した値が
//!    そのまま入力勾配になる（forward 経路は無変更のため forward
//!    カウンタは持たない）。
//! 2. 常時 `Unsupported` を返すフィクスチャ（専用カーネル未実装
//!    バックエンド相当）では既存ホスト VJP（`grad::log_softmax_vjp_along`
//!    相当。テスト内に f64 逐次和参照式を複製した解析値）へ
//!    フォールバックし bit 完全一致する（変更前の挙動と同一。
//!    ホスト VJP 自体は本イシューで変更していないため）。
//! 3. `Unsupported` 以外のエラーはフォールバックせずそのまま伝播する
//!    （判定迂回経路を作らない。`.claude/rules/security.md` A08）。
//! 4. 戻り値 shape が `out`（`upstream`）と不一致な場合は
//!    `AutodiffError::Backend(BackendError::ShapeMismatch)` で拒否
//!    する（fail-closed。`Op::NllLoss` 分岐と同型の契約）。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `common::NaiveOps` に委譲しつつ `log_softmax_backward` のみを
/// 素朴な参照実装（`dx = g − exp(y)·Σ_dim(g)`。`Σ_dim` は `f64`
/// 逐次和）でオーバーライドし、呼び出し回数を記録するフィクスチャ
/// （`nll_kl_div_loss_fusion.rs::CountingNllOps` と同型）。
struct CountingLogSoftmaxBackwardOps {
    inner: Box<dyn BackendOps + Send>,
    backward_calls: Arc<AtomicUsize>,
}

impl BackendOps for CountingLogSoftmaxBackwardOps {
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

    fn log_softmax_backward(
        &self,
        out: &Tensor<f32>,
        upstream: &Tensor<f32>,
        dim: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        self.backward_calls.fetch_add(1, Ordering::SeqCst);
        Ok(reference_log_softmax_backward(out, upstream, dim))
    }
}

/// 常時 `Unsupported` を返すフィクスチャ（専用カーネル未実装
/// バックエンド相当。何もオーバーライドせず既定実装のまま使う）。
struct AlwaysUnsupportedOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysUnsupportedOps {
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

/// `Unsupported` 以外のエラーを常に返すフィクスチャ（専用カーネルの
/// 実行時失敗を模す。`nll_kl_div_loss_fusion.rs::AlwaysFailingNllOps`
/// と同型）。
struct AlwaysFailingOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for AlwaysFailingOps {
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
    fn log_softmax_backward(
        &self,
        _out: &Tensor<f32>,
        _upstream: &Tensor<f32>,
        _dim: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::KernelLaunchFailed(
            "AlwaysFailingOps: simulated fused log_softmax backward failure (not Unsupported)"
                .into(),
        ))
    }
}

/// 戻り値 shape をわざと壊すフィクスチャ（fail-closed 拒否の検証用）。
struct WrongShapeOps {
    inner: Box<dyn BackendOps + Send>,
}

impl BackendOps for WrongShapeOps {
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
    fn log_softmax_backward(
        &self,
        _out: &Tensor<f32>,
        _upstream: &Tensor<f32>,
        _dim: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        // 1 要素少ないダミー shape を返す（実際の `out`／`upstream`
        // shape とは常に不一致になる）。
        Tensor::new(vec![0.0f32], &[1]).map_err(BackendError::ShapeMismatch)
    }
}

/// `dx = g − exp(y)·Σ_dim(g)` の独立参照実装（`Σ_dim` は `dim` 添字
/// 昇順の `f64` 逐次和。`.claude/rules/coding-rust.md` の勾配長軸縮約
/// 契約と同じ結合順序）。テスト内に複製することで、変更前のホスト VJP
/// （`grad::log_softmax_vjp_along`。非公開のため直接は呼べない）との
/// bit 完全一致を検証する。
fn reference_log_softmax_backward(
    out: &Tensor<f32>,
    upstream: &Tensor<f32>,
    dim: usize,
) -> Tensor<f32> {
    let shape = out.shape().to_vec();
    let axis_len = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let y = out.contiguous();
    let y = y.as_slice().unwrap_or(&[]);
    let g = upstream.contiguous();
    let g = g.as_slice().unwrap_or(&[]);
    let mut dx = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0f64;
            for c in 0..axis_len {
                sum += g[(o * axis_len + c) * inner + i] as f64;
            }
            for c in 0..axis_len {
                let idx = (o * axis_len + c) * inner + i;
                dx[idx] = (g[idx] as f64 - y[idx].exp() as f64 * sum) as f32;
            }
        }
    }
    t(dx, &shape)
}

#[test]
fn fused_log_softmax_backward_is_invoked_and_used_directly() {
    let backward_calls = Arc::new(AtomicUsize::new(0));
    let ops = CountingLogSoftmaxBackwardOps {
        inner: common::naive_ops(),
        backward_calls: backward_calls.clone(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.1, 2.0, -1.5, 0.3], &[2, 2]));
    let out = input.log_softmax(1).unwrap();
    let weight = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let loss = out.mul(&weight).unwrap().sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    assert_eq!(
        backward_calls.load(Ordering::SeqCst),
        1,
        "log_softmax_backward は backward で 1 回呼ばれる契約"
    );
    let dinput = grads.get(&input).unwrap().expect("到達する");

    // upstream（weight）と out（forward 記録値）から独立参照実装で
    // 期待値を計算し、フィクスチャが返した値がそのまま入力勾配に
    // なっている（bit 完全一致）ことを確認する。
    let out_value = out.to_tensor();
    let expected = reference_log_softmax_backward(&out_value, &weight.to_tensor(), 1);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                dinput.get(&[i, j]).unwrap().to_bits(),
                expected.get(&[i, j]).unwrap().to_bits(),
                "[{i},{j}]"
            );
        }
    }
}

#[test]
fn log_softmax_backward_falls_back_to_host_when_unsupported() {
    let ops = AlwaysUnsupportedOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.1, 2.0, -1.5, 0.3, 1.0, -2.0], &[2, 3]));
    let out = input.log_softmax(1).unwrap();
    let weight = tape.var(&t(vec![1.0, -2.0, 0.5, 0.25, -0.75, 3.0], &[2, 3]));
    let loss = out.mul(&weight).unwrap().sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dinput = grads.get(&input).unwrap().expect("到達する");

    // フォールバック先（`grad::log_softmax_vjp_along`）は本テストの
    // 独立参照実装と同じ式・同じ結合順序（f64 逐次和）のため bit
    // 完全一致するはず（ホスト VJP 自体は本イシューで変更していない）。
    let out_value = out.to_tensor();
    let expected = reference_log_softmax_backward(&out_value, &weight.to_tensor(), 1);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(
                dinput.get(&[i, j]).unwrap().to_bits(),
                expected.get(&[i, j]).unwrap().to_bits(),
                "[{i},{j}]"
            );
        }
    }
}

#[test]
fn log_softmax_backward_error_other_than_unsupported_propagates() {
    let ops = AlwaysFailingOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.1, 2.0, -1.5, 0.3], &[2, 2]));
    let out = input.log_softmax(1).unwrap();
    let loss = out.sum(None).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "expected KernelLaunchFailed to propagate without fallback, got {result:?}"
    );
}

#[test]
fn log_softmax_backward_rejects_wrong_output_shape() {
    let ops = WrongShapeOps {
        inner: common::naive_ops(),
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let input = tape.var(&t(vec![0.1, 2.0, -1.5, 0.3], &[2, 2]));
    let out = input.log_softmax(1).unwrap();
    let loss = out.sum(None).unwrap();

    let result = tape.backward(&loss);
    assert!(
        matches!(
            result,
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ),
        "expected ShapeMismatch to be rejected fail-closed, got {result:?}"
    );
}
