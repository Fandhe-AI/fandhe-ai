//! `compat::Sequential::add_max_pool3d`／`add_avg_pool3d`（イシュー #3028・親 #3018。
//! 公開形の正は `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3）の facade 公開面を
//! 検証する統合テスト。`compat_sequential_pooling.rs`（2D）・`compat_sequential_conv3d.rs` と同型。
//!
//! - 無効引数（`kernel_size=0`・`stride=Some(0)`・`dilation=0`・`2*padding > kernel`）が `Err`（panic しない）。
//! - `predict`／`forward` が `Var::max_pool3d`／`avg_pool3d` の直接呼び出しと bit 完全一致。
//! - backward の入力勾配が直接グラフと bit 完全一致（Max・Avg 各 1）。
//! - 学習 1 step（`Conv3d → Pool3d → Flatten → Linear`）が参照グラフの 1 step と bit 一致し、
//!   `compile`／`fit` が動く。pool 層は `trainable_parameters` の件数・順序を変えない。
//! - 常駐経路は `Unsupported`（fail-closed）。保存・復元は `max_pool3d`／`avg_pool3d` kind（イシュー #3029）で bit 一致。
//!   ONNX export は `UnsupportedLayer`。
//!
//! ホストフォールバック経路のため実機（CUDA／Metal）非依存。実機 parity は未実測のまま。

use std::path::PathBuf;

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

const SEED_C: u64 = 0x3028_0001;
const SEED_L: u64 = 0x3028_0002;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale + offset).collect()
}

fn input() -> Tensor<f32> {
    tensor(ramp(2 * 3 * 5 * 6 * 6, 0.007, -0.4), &[2, 3, 5, 6, 6])
}

fn unique_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fandhe_ai_3028_{tag}_{}", std::process::id()))
}

/// 破棄時に保存先を削除する RAII ガード。
struct DirGuard(PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const K: [usize; 3] = [2, 2, 2];
const MAX_STRIDE: Option<[usize; 3]> = Some([1, 2, 1]);
const PAD: [usize; 3] = [1, 0, 1];
const DIL: [usize; 3] = [1, 2, 1];

// ---------------------------------------------------------------- 無効引数

#[test]
fn add_pool3d_rejects_invalid_arguments() {
    let one = [1usize; 3];
    let z = [0usize; 3];
    assert!(
        Sequential::new()
            .add_max_pool3d([0, 2, 2], None, z, one)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_max_pool3d(K, Some([1, 0, 1]), z, one)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_max_pool3d(K, None, z, [1, 0, 1])
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_max_pool3d(K, None, [2, 0, 0], one)
            .is_err()
    );
    for cip in [true, false] {
        assert!(
            Sequential::new()
                .add_avg_pool3d([2, 0, 2], None, z, cip)
                .is_err()
        );
        assert!(
            Sequential::new()
                .add_avg_pool3d(K, Some([0, 1, 1]), z, cip)
                .is_err()
        );
        assert!(
            Sequential::new()
                .add_avg_pool3d(K, None, [0, 2, 0], cip)
                .is_err()
        );
    }
}

// ---------------------------------------------------------------- forward / predict

#[test]
fn max_pool3d_predict_and_forward_match_direct_call_bit_exact() {
    let model = Sequential::new()
        .add_max_pool3d(K, MAX_STRIDE, PAD, DIL)
        .unwrap();
    let x = input();
    let predicted = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
    let (direct, _idx) = xv.max_pool3d(K, MAX_STRIDE, PAD, DIL, false).unwrap();

    assert_eq!(dense_vec(&predicted), dense_vec(&direct.to_tensor()));
    assert_eq!(dense_vec(&forwarded), dense_vec(&direct.to_tensor()));
}

#[test]
fn max_pool3d_default_stride_matches_direct_call() {
    let model = Sequential::new()
        .add_max_pool3d(K, None, [0; 3], [1; 3])
        .unwrap();
    let x = input();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let (direct, _) = xv.max_pool3d(K, None, [0; 3], [1; 3], false).unwrap();
    assert_eq!(
        dense_vec(&model.predict(&x).unwrap()),
        dense_vec(&direct.to_tensor())
    );
}

#[test]
fn avg_pool3d_predict_and_forward_match_direct_call_bit_exact() {
    for cip in [true, false] {
        let model = Sequential::new()
            .add_avg_pool3d(K, Some([1, 2, 1]), PAD, cip)
            .unwrap();
        let x = input();
        let predicted = model.predict(&x).unwrap();

        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
        let direct = xv
            .avg_pool3d(K, Some([1, 2, 1]), PAD, false, cip)
            .unwrap()
            .to_tensor();

        assert_eq!(dense_vec(&predicted), dense_vec(&direct), "cip={cip}");
        assert_eq!(dense_vec(&forwarded), dense_vec(&direct), "cip={cip}");
    }
}

#[test]
fn pool3d_rejects_wrong_rank_input() {
    let x4 = tensor(ramp(2 * 3 * 6 * 6, 0.01, 0.0), &[2, 3, 6, 6]);
    let m = Sequential::new()
        .add_max_pool3d(K, None, [0; 3], [1; 3])
        .unwrap();
    assert!(m.predict(&x4).is_err());
    let a = Sequential::new()
        .add_avg_pool3d(K, None, [0; 3], true)
        .unwrap();
    assert!(a.predict(&x4).is_err());
}

// ---------------------------------------------------------------- backward

#[test]
fn max_pool3d_backward_matches_direct_graph_bit_exact() {
    let x_data = input();
    let model = Sequential::new()
        .add_max_pool3d(K, MAX_STRIDE, PAD, DIL)
        .unwrap();
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let loss1 = model.forward(&tape1, &x1).unwrap().sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1.get(&x1).unwrap().expect("入力へ勾配が到達するはず");

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let (y2, _) = x2.max_pool3d(K, MAX_STRIDE, PAD, DIL, false).unwrap();
    let loss2 = y2.sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2.get(&x2).unwrap().expect("入力へ勾配が到達するはず");

    assert_eq!(dense_vec(dx1), dense_vec(dx2));
}

#[test]
fn avg_pool3d_backward_matches_direct_graph_bit_exact() {
    let x_data = input();
    let model = Sequential::new()
        .add_avg_pool3d(K, MAX_STRIDE, PAD, false)
        .unwrap();
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let loss1 = model.forward(&tape1, &x1).unwrap().sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1.get(&x1).unwrap().expect("入力へ勾配が到達するはず");

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = x2.avg_pool3d(K, MAX_STRIDE, PAD, false, false).unwrap();
    let loss2 = y2.sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2.get(&x2).unwrap().expect("入力へ勾配が到達するはず");

    assert_eq!(dense_vec(dx1), dense_vec(dx2));
}

// ---------------------------------------------------------------- 学習 1 step

/// `Conv3d(1→2, k=2) → pool(k=2, stride=1) → Flatten → Linear(16→3)`。入力 `[N,1,4,4,4]`。
fn mixed_model(max: bool) -> Sequential {
    let m = Sequential::new()
        .add_conv3d(1, 2, [2; 3], [1; 3], [0; 3], [1; 3], 1, SEED_C)
        .unwrap();
    let m = if max {
        m.add_max_pool3d([2; 3], Some([1; 3]), [0; 3], [1; 3])
            .unwrap()
    } else {
        m.add_avg_pool3d([2; 3], Some([1; 3]), [0; 3], true)
            .unwrap()
    };
    m.add_flatten(1, 4).add_linear(16, 3, SEED_L).unwrap()
}

fn train_data() -> (Tensor<f32>, Tensor<f32>) {
    (
        tensor(ramp(4 * 64, 0.01, -0.5), &[4, 1, 4, 4, 4]),
        tensor(ramp(4 * 3, 0.1, 0.0), &[4, 3]),
    )
}

#[test]
fn pool3d_does_not_change_trainable_parameters() {
    for max in [true, false] {
        let model = mixed_model(max);
        let params = model.trainable_parameters();
        assert_eq!(params.len(), 4, "conv(w,b) + linear(w,b)");
        assert_eq!(params[0].shape(), &[2, 1, 2, 2, 2]);
        assert_eq!(params[2].shape(), &[16, 3]);
    }
}

#[test]
fn one_training_step_matches_reference_graph_bit_exact() {
    let (x, target) = train_data();
    for max in [true, false] {
        // Sequential 経由の 1 step。
        let mut model = mixed_model(max);
        let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();

        // 参照: 同一 seed の pool なし 2 分割モデルへ `Var` メソッドで pool を直接挟む。
        let a = Sequential::new()
            .add_conv3d(1, 2, [2; 3], [1; 3], [0; 3], [1; 3], 1, SEED_C)
            .unwrap();
        let b = Sequential::new()
            .add_flatten(1, 4)
            .add_linear(16, 3, SEED_L)
            .unwrap();
        let mut sgd_ref = Sgd::new(SgdConfig::new(0.05)).unwrap();
        let updated_ref = {
            let tape = fandhe_ai::tape();
            let ba = a.bind(&tape);
            let bb = b.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let h = ba.forward(&tape, &xv).unwrap();
            let p = if max {
                h.max_pool3d([2; 3], Some([1; 3]), [0; 3], [1; 3], false)
                    .unwrap()
                    .0
            } else {
                h.avg_pool3d([2; 3], Some([1; 3]), [0; 3], false, true)
                    .unwrap()
            };
            let loss = bb.forward(&tape, &p).unwrap().mse_loss(&tv).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let mut grad_refs = ba.trainable_grads(&grads).unwrap();
            grad_refs.extend(bb.trainable_grads(&grads).unwrap());
            let mut param_refs = a.trainable_parameters();
            param_refs.extend(b.trainable_parameters());
            sgd_ref.step(&param_refs, &grad_refs).unwrap()
        };

        let got: Vec<Vec<f32>> = model
            .trainable_parameters()
            .iter()
            .map(|t| dense_vec(t))
            .collect();
        let want: Vec<Vec<f32>> = updated_ref.iter().map(dense_vec).collect();
        assert_eq!(got, want, "max={max}");
    }
}

#[test]
fn compile_and_fit_run_with_pool3d() {
    let (x, target) = train_data();
    for max in [true, false] {
        let mut model = mixed_model(max);
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
            .unwrap();
        let w0 = dense_vec(model.trainable_parameters()[0]);
        model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
        assert_ne!(dense_vec(model.trainable_parameters()[0]), w0, "max={max}");
    }
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_pool3d_fail_closed() {
    let max = Sequential::new()
        .add_max_pool3d(K, None, [0; 3], [1; 3])
        .unwrap();
    let avg = Sequential::new()
        .add_avg_pool3d(K, None, [0; 3], true)
        .unwrap();
    let linear_only = Sequential::new().add_linear(6, 6, SEED_L).unwrap();
    let init_tape = fandhe_ai::tape();
    let mut store = linear_only.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    for model in [&max, &avg] {
        let tape = fandhe_ai::tape();
        assert!(matches!(
            model.init_device_param_store(&tape).unwrap_err(),
            BackendError::Unsupported(_)
        ));
        let x = tape.var(&input());
        assert!(matches!(
            model.forward_resident(&tape, &x, &mut store).unwrap_err(),
            AutodiffError::Backend(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            model.predict_resident(&store, &input()).unwrap_err(),
            AutodiffError::Backend(BackendError::Unsupported(_))
        ));
    }
}

// ---------------------------------------------------------------- 保存・ONNX（fail-closed）

#[test]
fn save_model_round_trips_pool3d_bit_identically() {
    let model = Sequential::new()
        .add_max_pool3d(K, Some([1, 1, 1]), [1, 0, 0], [1; 3])
        .and_then(|m| m.add_avg_pool3d(K, None, [0; 3], false))
        .unwrap();
    let dir = unique_dir("roundtrip");
    let _g = DirGuard(dir.clone());
    save_model(&model, &dir).expect("pool3d は保存できる（#3029）");
    let loaded = load_model(&dir).expect("復元できる");
    let x = tensor(ramp(2 * 5 * 5 * 5, 0.07, -3.0), &[1, 2, 5, 5, 5]);
    let (a, b) = (model.predict(&x).unwrap(), loaded.predict(&x).unwrap());
    assert_eq!(a.shape(), b.shape());
    assert!(
        dense_vec(&a)
            .iter()
            .zip(dense_vec(&b))
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

#[test]
fn onnx_export_rejects_pool3d_as_unsupported() {
    let max = Sequential::new()
        .add_max_pool3d(K, None, [0; 3], [1; 3])
        .unwrap();
    let avg = Sequential::new()
        .add_avg_pool3d(K, None, [0; 3], true)
        .unwrap();
    for m in [&max, &avg] {
        let err = OnnxModel::from_sequential(m).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}

// ---------------------------------------------------------------- train／eval 非依存

#[test]
fn pool3d_is_mode_independent() {
    let mut model = Sequential::new()
        .add_max_pool3d(K, None, [0; 3], [1; 3])
        .unwrap();
    let x = input();
    model.train();
    let a = dense_vec(&model.predict(&x).unwrap());
    model.eval();
    let b = dense_vec(&model.predict(&x).unwrap());
    assert_eq!(a, b);
}
