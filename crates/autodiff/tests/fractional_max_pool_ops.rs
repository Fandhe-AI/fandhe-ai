//! `Var::fractional_max_pool2d`／`fractional_max_pool3d`（イシュー #3826。設計 `docs/autodiff-pool3d-ops-decision.md`
//! §20・§24）の乱数契約と `Tape`／`Var` 経由の end-to-end 統合テスト。
//!
//! - グローバル RNG（`rng::manual_seed`／`rng::rand`）に触るため、テストバイナリ内の `LOCK` で直列化する。
//!   他のテストバイナリとはプロセスが分かれるので RNG は共有されない。
//! - `common::naive_ops()` は `fractional_max_pool*_windows` を override しないため、共有ホストカーネル
//!   へのフォールバック経路を通る。CPU `BackendOps` との一致は
//!   `crates/facade/tests/fractional_max_pool_ops_backend_parity.rs` が担当する。
//! - PyTorch fixture との突合は `crates/autodiff/src/fractional_max_pool_ops.rs` の単体テスト
//!   （標本を明示する `pub(crate)` 入口が必要なため）。

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::fractional_max_pool::FractionalMaxPoolWindows;
use fandhe_ai_tensor_core::rng::{manual_seed, rand};
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn ramp(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    // 全要素が異なる値（タイなし）。
    t(
        (0..n).map(|i| ((i * 37) % 251) as f32 * 0.25).collect(),
        shape,
    )
}

fn next_unit() -> f32 {
    rand(&[1]).expect("rand").host_slice()[0]
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
}

#[test]
fn manual_seed_makes_fractional_max_pool_deterministic() {
    let _g = lock();
    for dim in [2usize, 3] {
        let shape: Vec<usize> = if dim == 2 {
            vec![2, 3, 12, 12]
        } else {
            vec![1, 2, 9, 9, 9]
        };
        let run = |seed: u64| {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&ramp(&shape));
            manual_seed(seed);
            let (y, idx) = if dim == 2 {
                x.fractional_max_pool2d([2, 2], [5, 5]).unwrap()
            } else {
                x.fractional_max_pool3d([2, 2, 2], [4, 4, 4]).unwrap()
            };
            (bits(&y.to_tensor().host_slice()), idx.host_slice().to_vec())
        };
        let a = run(7);
        assert_eq!(a, run(7), "dim={dim}: 同じシードは値・索引が bit 一致");
        assert!(
            (1..16).any(|s| run(s) != a),
            "dim={dim}: シードを変えれば窓が変わる形状のはず"
        );
    }
}

#[test]
fn consumes_exactly_n_c_d_samples_and_only_after_validation() {
    let _g = lock();
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&ramp(&[2, 3, 8, 8]));
    manual_seed(11);
    x.fractional_max_pool2d([2, 2], [4, 4]).unwrap();
    let after_op = next_unit();
    manual_seed(11);
    rand(&[2 * 3 * 2]).unwrap();
    assert_eq!(after_op, next_unit(), "2d の消費数はちょうど N·C·2");

    let x3 = tape.var(&ramp(&[2, 1, 6, 6, 6]));
    manual_seed(11);
    x3.fractional_max_pool3d([2, 2, 2], [3, 3, 3]).unwrap();
    let after_op = next_unit();
    manual_seed(11);
    rand(&[2 * 3]).unwrap();
    assert_eq!(after_op, next_unit(), "3d の消費数はちょうど N·C·3");

    // 拒否した入力は RNG を消費しない。
    manual_seed(11);
    let expected = next_unit();
    manual_seed(11);
    assert!(x.fractional_max_pool2d([0, 2], [2, 2]).is_err());
    assert!(x.fractional_max_pool2d([2, 2], [8, 8]).is_err());
    assert!(x.fractional_max_pool3d([2, 2, 2], [2, 2, 2]).is_err());
    assert_eq!(next_unit(), expected);
}

#[test]
fn gradient_matches_central_difference() {
    let _g = lock();
    let shape = [1usize, 2, 9, 9];
    let base = ramp(&shape).host_slice().to_vec();
    // 窓を固定するため、各評価の直前に同じシードをセットする。
    let eval = |xs: &[f32]| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.to_vec(), &shape));
        manual_seed(5);
        let (y, _) = x.fractional_max_pool2d([3, 3], [5, 5]).unwrap();
        y.to_tensor()
            .host_slice()
            .iter()
            .enumerate()
            .map(|(i, &v)| f64::from(v) * f64::from((i % 7) as f32 + 1.0))
            .sum()
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(base.clone(), &shape));
    manual_seed(5);
    let (y, _) = x.fractional_max_pool2d([3, 3], [5, 5]).unwrap();
    let w: Vec<f32> = (0..y.to_tensor().host_slice().len())
        .map(|i| (i % 7) as f32 + 1.0)
        .collect();
    let wv = tape.var_no_grad(&t(w, y.to_tensor().shape()));
    let loss = y.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().unwrap().host_slice().to_vec();
    let h = 0.05f32;
    for j in 0..base.len() {
        let (mut hi, mut lo) = (base.clone(), base.clone());
        hi[j] += h;
        lo[j] -= h;
        let fd = (eval(&hi) - eval(&lo)) / (2.0 * f64::from(h));
        assert!(
            common::req2_close(f64::from(dx[j]), fd),
            "x[{j}]: {} vs {fd}",
            dx[j]
        );
    }
}

#[test]
fn feasibility_boundary_differs_between_2d_and_3d() {
    let _g = lock();
    let tape = Tape::new_with_ops(common::naive_ops());
    manual_seed(1);
    let x2 = tape.var(&ramp(&[1, 1, 4, 4]));
    assert!(
        x2.fractional_max_pool2d([2, 2], [3, 3]).is_ok(),
        "2d: 等号は受理"
    );
    assert!(x2.fractional_max_pool2d([2, 2], [4, 3]).is_err());
    let x3 = tape.var(&ramp(&[1, 1, 4, 4, 4]));
    assert!(
        x3.fractional_max_pool3d([2, 2, 2], [3, 3, 3]).is_err(),
        "3d: 等号は拒否"
    );
    assert!(x3.fractional_max_pool3d([2, 2, 2], [2, 2, 2]).is_ok());
}

#[test]
fn full_overlap_windows_accumulate_gradients() {
    let _g = lock();
    // out = in - k + 1 なら窓は標本によらず決まる。入力は 1 要素だけ突出させる。
    let tape = Tape::new_with_ops(common::naive_ops());
    let mut data = vec![0.0f32; 9];
    data[4] = 10.0;
    let x = tape.var(&t(data, &[1, 1, 3, 3]));
    let (y, idx) = x.fractional_max_pool2d([2, 2], [2, 2]).unwrap();
    assert_eq!(idx.host_slice().to_vec(), vec![4, 4, 4, 4]);
    let loss = y.sum(None).unwrap();
    let g = tape.backward(&loss).unwrap();
    assert_eq!(
        g.get(&x).unwrap().unwrap().host_slice().to_vec(),
        vec![0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn empty_batch_and_channel_yield_empty_outputs_without_consuming() {
    let _g = lock();
    let tape = Tape::new_with_ops(common::naive_ops());
    manual_seed(3);
    let expected = next_unit();
    manual_seed(3);
    for shape in [[0usize, 2, 4, 4], [2, 0, 4, 4]] {
        let x = tape.var(&t(vec![], &shape));
        let (y, idx) = x.fractional_max_pool2d([2, 2], [2, 2]).unwrap();
        assert_eq!(y.to_tensor().shape(), &[shape[0], shape[1], 2, 2]);
        assert_eq!(idx.shape(), &[shape[0], shape[1], 2, 2]);
    }
    assert_eq!(next_unit(), expected, "N·C=0 は標本を引かない");
}

#[test]
fn empty_batch_still_rejects_plane_beyond_i32() {
    let _g = lock();
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 1, 2048, 1024, 1024]));
    assert!(matches!(
        x.fractional_max_pool3d([1, 1, 1], [1, 1, 1]),
        Err(AutodiffError::Shape(ShapeError::IndexRangeOverflow { .. }))
    ));
}

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let _g = lock();
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&ramp(&[1, 1, 4, 4]));
    let r3 = tape.var(&ramp(&[1, 4, 4]));
    let z = tape.var(&t(vec![], &[1, 1, 0, 4]));
    let before = tape.len();
    assert!(matches!(
        x.fractional_max_pool2d([0, 1], [1, 1]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        x.fractional_max_pool2d([1, 1], [1, 0]),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        r3.fractional_max_pool2d([1, 1], [1, 1]),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    assert!(z.fractional_max_pool2d([1, 1], [1, 1]).is_err());
    assert!(x.fractional_max_pool2d([usize::MAX, 2], [2, 2]).is_err());
    assert!(
        x.fractional_max_pool2d([1, 1], [usize::MAX, usize::MAX])
            .is_err()
    );
    assert_eq!(tape.len(), before, "検査は tape 操作より前なので増えない");
}

// --- フォールバックとエラー伝播 ---

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongValueShape,
    WrongIndexShape,
}

struct PoolMock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

impl BackendOps for PoolMock {
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
    fn fractional_max_pool2d_windows(
        &self,
        _input: &Tensor<f32>,
        _windows: &FractionalMaxPoolWindows<2>,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            // 正しい出力 shape は [1,1,2,2]。
            Mode::WrongValueShape => Ok((
                t(vec![0.0; 2], &[2]),
                Tensor::new(vec![0_i32; 4], &[1, 1, 2, 2]).unwrap(),
            )),
            Mode::WrongIndexShape => Ok((
                t(vec![0.0; 4], &[1, 1, 2, 2]),
                Tensor::new(vec![0_i32; 2], &[2]).unwrap(),
            )),
        }
    }
}

fn mock_tape(mode: Mode) -> (Tape, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(PoolMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        mode,
    }));
    (tape, calls)
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let _g = lock();
    let (tape, calls) = mock_tape(Mode::Unsupported);
    let x = tape.var(&ramp(&[1, 1, 3, 3]));
    let (y, idx) = x.fractional_max_pool2d([2, 2], [2, 2]).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 1, 2, 2]);
    assert_eq!(idx.shape(), &[1, 1, 2, 2]);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "backend を先に呼んでからフォールバックする"
    );
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let _g = lock();
    let (tape, _) = mock_tape(Mode::LaunchFailed);
    let x = tape.var(&ramp(&[1, 1, 3, 3]));
    assert!(matches!(
        x.fractional_max_pool2d([2, 2], [2, 2]),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error_without_orphan_nodes() {
    let _g = lock();
    for mode in [Mode::WrongValueShape, Mode::WrongIndexShape] {
        let (tape, _) = mock_tape(mode);
        let x = tape.var(&ramp(&[1, 1, 3, 3]));
        let before = tape.len();
        assert!(matches!(
            x.fractional_max_pool2d([2, 2], [2, 2]),
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
        assert_eq!(tape.len(), before);
    }
}
