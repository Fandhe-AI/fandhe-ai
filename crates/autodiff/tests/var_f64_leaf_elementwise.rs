//! `fandhe_ai_autodiff::f64_autograd`（イシュー #2195・親 #2142）の
//! 統合テスト。
//!
//! `crates/autodiff/tests/*.rs` は `autodiff` の公開 API のみを経由する
//! 別クレート扱いのため（`tests/common/mod.rs` モジュール doc と同じ
//! 理由）、`mod common;`（`NaiveOps`。`typed_ops_f64` は既定 `None`）の
//! 上で `TapeF64` を組み、ホスト参照実装の経路を検証する。ネイティブ
//! （`typed_ops_f64` が `Some`）経路の bit 一致は `crates/facade/tests/
//! var_f64_autograd_backend_parity.rs` が CPU 実装（`fandhe-ai-backend-cpu`）
//! を使って別途検証する（`autodiff` は具体バックエンドクレートへ依存
//! しないという設計上の不変条件〈`docs/fusion-graph-design.md` §3.4〉を
//! 崩さないため、本ファイルでは `backend-cpu` を dev-dependency に
//! 追加しない）。

mod common;

use fandhe_ai_autodiff::f64_autograd::TapeF64;
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, BackendOps, Device, Tensor, TypedOps};

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- 葉の生成 ---

#[test]
fn leaf_generation_preserves_value_and_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let x = graph.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert_eq!(x.shape(), vec![3]);
    assert_eq!(x.value().host_slice().into_owned(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn var_no_grad_leaf_rejects_gradient_tracking() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let x = graph.var_no_grad(&t(vec![1.0], &[]));
    let y = graph.var(&t(vec![2.0], &[]));
    let z = x.add(&y).expect("add は成功するはず");
    let grads = graph.backward(&z).expect("backward は成功するはず");
    assert!(matches!(
        grads.get(&x),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(
        grads
            .get(&y)
            .expect("y は requires_grad=true のはず")
            .is_some()
    );
}

// --- 1 step backward（4 演算） ---

#[test]
fn backward_combined_add_mul_div_matches_closed_form() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let x = graph.var(&t(vec![2.0], &[]));
    let w = graph.var(&t(vec![3.0], &[]));

    // y = x*w + x/w
    let mul = x.mul(&w).expect("mul");
    let div = x.div(&w).expect("div");
    let y = mul.add(&div).expect("add");
    assert_eq!(
        y.value().host_slice().into_owned(),
        vec![2.0 * 3.0 + 2.0 / 3.0]
    );

    let grads = graph.backward(&y).expect("backward");
    // dy/dx = w + 1/w, dy/dw = x - x/w^2
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    let dw = grads.get(&w).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![3.0 + 1.0 / 3.0]);
    assert_eq!(dw, vec![2.0 - 2.0 / (3.0 * 3.0)]);
}

#[test]
fn backward_pow_matches_closed_form() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![2.0], &[]));
    let b = graph.var(&t(vec![3.0], &[]));
    let y = a.pow(&b).expect("pow");
    assert_eq!(y.value().host_slice().into_owned(), vec![8.0]);

    let grads = graph.backward(&y).expect("backward");
    // dy/da = b*a^(b-1) = 3*4 = 12, dy/db = y*ln(a) = 8*ln(2)
    let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();
    let db = grads.get(&b).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(da, vec![12.0]);
    assert_eq!(db, vec![8.0 * 2.0f64.ln()]);
}

// --- ブロードキャスト ---

#[test]
fn broadcast_add_reduces_gradient_by_sequential_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = graph.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let y = a.add(&b).expect("add");
    assert_eq!(y.shape(), vec![2, 3]);

    let grads = graph.backward(&y).expect("backward");
    let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();
    let db = grads.get(&b).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(da, vec![1.0; 6]);
    // `[2,3]` の全 1 勾配を軸 0（長さ 2）で縮約すると各要素 2.0。
    assert_eq!(db, vec![2.0, 2.0, 2.0]);
}

#[test]
fn broadcast_mul_reduces_gradient_by_sequential_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = graph.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let y = a.mul(&b).expect("mul");
    let grads = graph.backward(&y).expect("backward");
    let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();
    let db = grads.get(&b).unwrap().unwrap().host_slice().into_owned();
    // da = broadcast(b) = [10,20,30,10,20,30]
    assert_eq!(da, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    // db = sum over axis 0 of a = [1+4, 2+5, 3+6]
    assert_eq!(db, vec![5.0, 7.0, 9.0]);
}

// --- fan-out（同じ葉を複数回使う） ---

#[test]
fn fan_out_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let x = graph.var(&t(vec![2.0], &[]));
    // y = x + x + x → dy/dx = 3
    let y = x.add(&x).expect("add").add(&x).expect("add");
    let grads = graph.backward(&y).expect("backward");
    let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(dx, vec![3.0]);
}

// --- pow の境界 ---

#[test]
fn pow_zero_base_and_zero_exponent_masks_gradient_and_forward_is_one() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![0.0], &[]));
    let b = graph.var(&t(vec![0.0], &[]));
    let y = a.pow(&b).expect("pow");
    assert_eq!(y.value().host_slice().into_owned(), vec![1.0]);
    let grads = graph.backward(&y).expect("backward");
    let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();
    let db = grads.get(&b).unwrap().unwrap().host_slice().into_owned();
    assert_eq!(da, vec![0.0]);
    assert_eq!(db, vec![0.0]);
}

#[test]
fn pow_negative_base_with_non_integer_exponent_forward_is_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![-2.0], &[]));
    let b = graph.var(&t(vec![0.5], &[]));
    let y = a.pow(&b).expect("pow は panic せず成功するはず");
    assert!(y.value().host_slice()[0].is_nan());
}

// --- div の境界 ---

#[test]
fn div_by_zero_forward_is_inf_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![1.0], &[]));
    let b = graph.var(&t(vec![0.0], &[]));
    let y = a.div(&b).expect("div は panic せず成功するはず");
    assert!(y.value().host_slice()[0].is_infinite());
}

// --- エラー ---

#[test]
fn cross_graph_operands_are_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph_a = TapeF64::new(&tape);
    let graph_b = TapeF64::new(&tape);
    let x = graph_a.var(&t(vec![1.0], &[]));
    let y = graph_b.var(&t(vec![1.0], &[]));
    assert!(matches!(x.add(&y), Err(AutodiffError::TapeMismatch)));
}

#[test]
fn backward_of_no_grad_only_loss_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var_no_grad(&t(vec![1.0], &[]));
    let b = graph.var_no_grad(&t(vec![2.0], &[]));
    let loss = a.add(&b).expect("add");
    assert!(matches!(
        graph.backward(&loss),
        Err(AutodiffError::Backward(_))
    ));
}

#[test]
fn incompatible_broadcast_shape_is_rejected_with_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![1.0, 2.0], &[2]));
    let b = graph.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert!(matches!(a.add(&b), Err(AutodiffError::Shape(_))));
}

// --- cast の契約 ---

#[test]
fn cast_to_f64_then_f64_backward_does_not_affect_f32_tape_backward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&Tensor::new(vec![2.0f32, 3.0, -1.0], &[3]).unwrap());
    let y = x.mul(&x).unwrap();
    let loss = y.sum(None).unwrap();

    // f64 グラフ側で cast した値から派生した計算・backward を挟む。
    let x_f64 = x.cast::<f64>().expect("cast は成功するはず");
    let graph = TapeF64::new(&tape);
    let leaf = graph.var(&x_f64);
    let doubled = leaf.mul(&leaf).expect("mul");
    let _ = graph
        .backward(&doubled)
        .expect("f64 グラフの backward は独立に成功するはず");

    // f32 テープの backward・勾配は f64 側の操作に影響されない。
    let grads = tape.backward(&loss).unwrap();
    let dx = grads
        .get(&x)
        .expect("test fixture: get 自体は Err にならないはず")
        .expect("test fixture: x の勾配は存在するはず")
        .host_slice()
        .into_owned();
    assert_eq!(dx, vec![4.0, 6.0, -2.0]);
}

// --- Some 経路（typed_ops_f64 をオーバーライドした BackendOps） ---

/// `crates/autodiff/src/tape.rs` の `typed_ops_accessor_tests` と同型の
/// 最小フィクスチャ。`add`／`mul` のみをネイティブ実装で計算し、それ
/// 以外の `TypedOps<f64>` 演算は `Unsupported` を返す（本テストの対象
/// 外）。
struct DummyF64Ops;

impl TypedOps<f64> for DummyF64Ops {
    fn gemm(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported("test fixture: gemm".into()))
    }
    fn add(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        let (ba, bb) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let shape = ba.shape().to_vec();
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..numel {
            let x = ba.get(&idx).unwrap_or(0.0);
            let y = bb.get(&idx).unwrap_or(0.0);
            data.push(x + y);
            for axis in (0..shape.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < shape[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
        Tensor::new(data, &shape).map_err(BackendError::ShapeMismatch)
    }
    fn mul(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        let (ba, bb) = a.broadcast_with(b).map_err(BackendError::ShapeMismatch)?;
        let shape = ba.shape().to_vec();
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..numel {
            let x = ba.get(&idx).unwrap_or(0.0);
            let y = bb.get(&idx).unwrap_or(0.0);
            data.push(x * y);
            for axis in (0..shape.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < shape[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
        Tensor::new(data, &shape).map_err(BackendError::ShapeMismatch)
    }
    fn relu(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported("test fixture: relu".into()))
    }
    fn exp(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported("test fixture: exp".into()))
    }
    fn tanh(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported("test fixture: tanh".into()))
    }
    fn sum(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported("test fixture: sum".into()))
    }
    fn max(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        Err(BackendError::Unsupported("test fixture: max".into()))
    }
}

struct OpsWithTypedF64 {
    inner: Box<dyn BackendOps + Send>,
    f64_ops: DummyF64Ops,
}

impl BackendOps for OpsWithTypedF64 {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> {
        Some(&self.f64_ops)
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

#[test]
fn add_and_mul_dispatch_to_native_typed_ops_f64_when_available() {
    let ops = OpsWithTypedF64 {
        inner: common::naive_ops(),
        f64_ops: DummyF64Ops,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![2.0, 3.0], &[2]));
    let b = graph.var(&t(vec![10.0, 20.0], &[2]));
    let sum = a.add(&b).expect("add はネイティブ経路で成功するはず");
    let prod = a.mul(&b).expect("mul はネイティブ経路で成功するはず");
    assert_eq!(sum.value().host_slice().into_owned(), vec![12.0, 23.0]);
    assert_eq!(prod.value().host_slice().into_owned(), vec![20.0, 60.0]);

    // `div`／`pow` は `TypedOps<f64>` に演算が存在しないため、ネイティブ
    // 実装（`Unsupported` を返す `gemm`／`relu` 等とは異なり、そもそも
    // `TypedOps<f64>` に variant がない）ではなく常にホスト参照実装で
    // 計算される。
    let quot = a.div(&b).expect("div はホスト経路で成功するはず");
    assert_eq!(quot.value().host_slice().into_owned(), vec![0.2, 0.15]);
}

#[test]
fn add_falls_back_to_host_when_native_typed_ops_f64_returns_unsupported() {
    struct UnsupportedF64Ops;
    impl TypedOps<f64> for UnsupportedF64Ops {
        fn gemm(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture".into()))
        }
        fn add(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: add".into()))
        }
        fn mul(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: mul".into()))
        }
        fn relu(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: relu".into()))
        }
        fn exp(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: exp".into()))
        }
        fn tanh(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: tanh".into()))
        }
        fn sum(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: sum".into()))
        }
        fn max(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("test fixture: max".into()))
        }
    }

    struct OpsWithUnsupportedF64 {
        inner: Box<dyn BackendOps + Send>,
        f64_ops: UnsupportedF64Ops,
    }
    impl BackendOps for OpsWithUnsupportedF64 {
        fn device(&self) -> Device {
            self.inner.device()
        }
        fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> {
            Some(&self.f64_ops)
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

    let ops = OpsWithUnsupportedF64 {
        inner: common::naive_ops(),
        f64_ops: UnsupportedF64Ops,
    };
    let tape = Tape::new_with_ops(Box::new(ops));
    let graph = TapeF64::new(&tape);
    let a = graph.var(&t(vec![2.0, 3.0], &[2]));
    let b = graph.var(&t(vec![10.0, 20.0], &[2]));
    let sum = a
        .add(&b)
        .expect("Unsupported はホスト実装へフォールバックするはず");
    assert_eq!(sum.value().host_slice().into_owned(), vec![12.0, 23.0]);
}
