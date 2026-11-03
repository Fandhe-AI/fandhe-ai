//! `compat::Sequential::add_fractional_max_pool2d`／`add_fractional_max_pool3d`（イシュー #3827・
//! 設計の正は `docs/autodiff-pool3d-ops-decision.md` §20・§25）の facade 公開面を検証する統合テスト。
//! `compat_sequential_adaptive_pool3d.rs` と同型に、乱数契約（§20.12）の検査を上乗せする。
//!
//! - 0 軸は構築時 `Err`、実行不能な形状は forward 時 `Err` で RNG を消費しない。
//! - 各経路の直前に `manual_seed` を置いた `predict`／`forward`／backward が `Var::fractional_max_pool*`
//!   の直接呼び出しと bit 完全一致する。同じシードの 2 回は一致する（受入の決定性）。
//! - 1 回の forward の RNG 消費数はちょうど `N·C·R`（`eval()` でも引く。Q4(a)）。
//! - 学習 1 step・`compile`／`fit`。常駐経路は `Unsupported`（fail-closed）。保存・復元は bit 一致。
//!   ONNX export は `UnsupportedLayer`。
//!
//! グローバル RNG を共有するため、全テストを `LOCK` で直列化する。ホストフォールバック経路のため
//! 実機非依存。実機 parity は未実測（Phase 10。`compat_sequential_fractional_max_pool_backend_parity.rs`）。

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, Tensor, manual_seed, rand};

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const SEED: u64 = 0x3827_0001;
const SEED_C: u64 = 0x3827_0002;
const SEED_L: u64 = 0x3827_0003;
/// 2d: 入力 `[2,3,6,6]` に対し out=4（alpha = 4/3 で整数にならない）。
const K2: [usize; 2] = [2, 2];
const O2: [usize; 2] = [4, 4];
/// 3d: 入力 `[1,2,5,5,5]` に対し out=3。
const K3: [usize; 3] = [2, 2, 2];
const O3: [usize; 3] = [3, 3, 3];

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn zigzag(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37) % 23) as f32) * 0.11 - 1.0)
        .collect()
}

fn input2() -> Tensor<f32> {
    tensor(zigzag(2 * 3 * 6 * 6), &[2, 3, 6, 6])
}

fn input3() -> Tensor<f32> {
    tensor(zigzag(2 * 5 * 5 * 5), &[1, 2, 5, 5, 5])
}

fn m2() -> Sequential {
    Sequential::new().add_fractional_max_pool2d(K2, O2).unwrap()
}

fn m3() -> Sequential {
    Sequential::new().add_fractional_max_pool3d(K3, O3).unwrap()
}

fn next_rand() -> f32 {
    dense_vec(&rand(&[1]).unwrap())[0]
}

fn unique_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fandhe_ai_3827_{tag}_{}", std::process::id()))
}

/// 破棄時に保存先を削除する RAII ガード。
struct DirGuard(PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn add_fractional_max_pool_rejects_zero_axes() {
    let _g = lock();
    for bad in [[0, 1], [1, 0]] {
        assert!(
            Sequential::new()
                .add_fractional_max_pool2d(bad, [1, 1])
                .is_err()
        );
        assert!(
            Sequential::new()
                .add_fractional_max_pool2d([1, 1], bad)
                .is_err()
        );
    }
    for bad in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        assert!(
            Sequential::new()
                .add_fractional_max_pool3d(bad, [1; 3])
                .is_err()
        );
        assert!(
            Sequential::new()
                .add_fractional_max_pool3d([1; 3], bad)
                .is_err()
        );
    }
}

#[test]
fn predict_and_forward_match_direct_call_bit_exact_under_same_seed() {
    let _g = lock();
    let tape = fandhe_ai::tape();

    let x = input2();
    let xv = tape.var(&x);
    manual_seed(SEED);
    let (d2, _) = xv.fractional_max_pool2d(K2, O2).unwrap();
    let d2 = d2.to_tensor();
    assert_eq!(d2.shape(), &[2, 3, 4, 4]);
    manual_seed(SEED);
    let p2 = m2().predict(&x).unwrap();
    assert_eq!(dense_vec(&p2), dense_vec(&d2));
    manual_seed(SEED);
    let f2 = m2().forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(dense_vec(&f2), dense_vec(&d2));

    let x = input3();
    let xv = tape.var(&x);
    manual_seed(SEED);
    let (d3, _) = xv.fractional_max_pool3d(K3, O3).unwrap();
    let d3 = d3.to_tensor();
    assert_eq!(d3.shape(), &[1, 2, 3, 3, 3]);
    manual_seed(SEED);
    assert_eq!(dense_vec(&m3().predict(&x).unwrap()), dense_vec(&d3));
}

#[test]
fn same_seed_twice_is_deterministic() {
    let _g = lock();
    let (x2, x3) = (input2(), input3());
    manual_seed(SEED);
    let a = (
        dense_vec(&m2().predict(&x2).unwrap()),
        dense_vec(&m3().predict(&x3).unwrap()),
    );
    manual_seed(SEED);
    let b = (
        dense_vec(&m2().predict(&x2).unwrap()),
        dense_vec(&m3().predict(&x3).unwrap()),
    );
    assert_eq!(a, b);
}

fn input_grad(is_2d: bool, direct: bool) -> Vec<f32> {
    let x_data = if is_2d { input2() } else { input3() };
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    manual_seed(SEED);
    let y = if direct {
        if is_2d {
            x.fractional_max_pool2d(K2, O2).unwrap().0
        } else {
            x.fractional_max_pool3d(K3, O3).unwrap().0
        }
    } else {
        let m = if is_2d { m2() } else { m3() };
        m.forward(&tape, &x).unwrap()
    };
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    dense_vec(grads.get(&x).unwrap().expect("入力へ勾配が到達するはず"))
}

#[test]
fn backward_matches_direct_graph_bit_exact() {
    let _g = lock();
    for is_2d in [true, false] {
        assert_eq!(
            input_grad(is_2d, false),
            input_grad(is_2d, true),
            "2d={is_2d}"
        );
    }
}

#[test]
fn forward_consumes_exactly_n_c_r_samples_even_in_eval() {
    let _g = lock();
    let (x2, x3) = (input2(), input3());
    for eval in [false, true] {
        // 2d: N·C·R = 2·3·2 = 12。
        let mut m = m2();
        if eval {
            m.eval();
        }
        manual_seed(SEED);
        m.predict(&x2).unwrap();
        let after_predict = next_rand();
        manual_seed(SEED);
        rand(&[12]).unwrap();
        assert_eq!(after_predict, next_rand(), "2d eval={eval}");

        // 3d: N·C·R = 1·2·3 = 6。
        let mut m = m3();
        if eval {
            m.eval();
        }
        manual_seed(SEED);
        m.predict(&x3).unwrap();
        let after_predict = next_rand();
        manual_seed(SEED);
        rand(&[6]).unwrap();
        assert_eq!(after_predict, next_rand(), "3d eval={eval}");
    }
}

#[test]
fn infeasible_shape_is_rejected_without_consuming_rng() {
    let _g = lock();
    // 2d: out + k - 1 <= in を超える。
    let too_big = Sequential::new()
        .add_fractional_max_pool2d([2, 2], [6, 6])
        .unwrap();
    manual_seed(SEED);
    assert!(too_big.predict(&input2()).is_err());
    let after_err = next_rand();
    manual_seed(SEED);
    assert_eq!(after_err, next_rand());
    // rank 違反。
    manual_seed(SEED);
    let x3 = input3();
    assert!(m2().predict(&x3).is_err());
    let after_err = next_rand();
    manual_seed(SEED);
    assert_eq!(after_err, next_rand());
}

// ---------------------------------------------------------------- 学習

/// `Conv2d(1→2, k=2) → FractionalMaxPool2d(k=2, out=2) → Flatten → Linear(8→3)`。入力 `[N,1,5,5]`。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 2, [2; 2], [1; 2], [0; 2], [1; 2], 1, SEED_C)
        .unwrap()
        .add_fractional_max_pool2d([2, 2], [2, 2])
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(8, 3, SEED_L)
        .unwrap()
}

fn train_data() -> (Tensor<f32>, Tensor<f32>) {
    (
        tensor(zigzag(4 * 25), &[4, 1, 5, 5]),
        tensor((0..12).map(|i| i as f32 * 0.1).collect(), &[4, 3]),
    )
}

#[test]
fn fractional_max_pool_does_not_change_trainable_parameters() {
    let _g = lock();
    let params = mixed_model().trainable_parameters().len();
    assert_eq!(params, 4, "conv(w,b) + linear(w,b)");
}

#[test]
fn compile_and_fit_run_with_fractional_max_pool() {
    let _g = lock();
    let (x, target) = train_data();
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let w0 = dense_vec(model.trainable_parameters()[0]);
    manual_seed(SEED);
    model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
    assert_ne!(dense_vec(model.trainable_parameters()[0]), w0);
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_fractional_max_pool_fail_closed() {
    let _g = lock();
    let linear_only = Sequential::new().add_linear(6, 6, SEED_L).unwrap();
    let init_tape = fandhe_ai::tape();
    let mut store = linear_only.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    for model in [m2(), m3()] {
        let tape = fandhe_ai::tape();
        assert!(matches!(
            model.init_device_param_store(&tape).unwrap_err(),
            BackendError::Unsupported(_)
        ));
        let x = tape.var(&input2());
        assert!(matches!(
            model.forward_resident(&tape, &x, &mut store).unwrap_err(),
            AutodiffError::Backend(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            model.predict_resident(&store, &input2()).unwrap_err(),
            AutodiffError::Backend(BackendError::Unsupported(_))
        ));
    }
}

// ---------------------------------------------------------------- 保存・ONNX

#[test]
fn save_model_round_trips_fractional_max_pool_bit_identically() {
    let _g = lock();
    for (model, x) in [(m2(), input2()), (m3(), input3())] {
        let dir = unique_dir("roundtrip");
        let _d = DirGuard(dir.clone());
        save_model(&model, &dir).expect("fractional max pool は保存できる（#3827）");
        let loaded = load_model(&dir).expect("復元できる");
        // RNG 状態は保存しないため、同じシードの下で比較する。
        manual_seed(SEED);
        let a = model.predict(&x).unwrap();
        manual_seed(SEED);
        let b = loaded.predict(&x).unwrap();
        assert_eq!(a.shape(), b.shape());
        assert!(
            dense_vec(&a)
                .iter()
                .zip(dense_vec(&b))
                .all(|(p, q)| p.to_bits() == q.to_bits())
        );
        drop(_d);
    }
}

#[test]
fn onnx_export_rejects_fractional_max_pool_as_unsupported() {
    let _g = lock();
    for m in [m2(), m3()] {
        let err = OnnxModel::from_sequential(&m).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
