//! `compat::Sequential::add_adaptive_avg_pool3d`／`add_adaptive_max_pool3d`（イシュー #3824・
//! 設計の正は `docs/autodiff-pool3d-ops-decision.md` §18）の facade 公開面を検証する統合テスト。
//! `compat_sequential_pool3d.rs` と同型。
//!
//! - `output_size` に 0 を含む場合が `Err`（panic しない）。
//! - `predict`／`forward` が `Var::adaptive_*_pool3d` の直接呼び出しと bit 完全一致（割り切れない重なり窓を含む）。
//! - backward の入力勾配が直接グラフと bit 完全一致。
//! - 学習 1 step・`compile`／`fit`・`trainable_parameters` の件数と順序。
//! - 常駐経路は `Unsupported`（fail-closed）。保存・復元は bit 一致。ONNX export は `UnsupportedLayer`。
//!
//! ホストフォールバック経路のため実機非依存。実機 parity は未実測（Phase 10 の #3860・#3861）。

use std::path::PathBuf;

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

const SEED_C: u64 = 0x3824_0001;
const SEED_L: u64 = 0x3824_0002;
/// 入力 `[2,3,5,6,6]` に対し D=5→2（重なり）・H=6→3（割り切れる）・W=6→4（重なり）。
const OUT: [usize; 3] = [2, 3, 4];

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

/// 単調ではない値（最大値の位置が窓ごとに変わる）を作る。
fn zigzag(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37) % 23) as f32) * 0.11 - 1.0)
        .collect()
}

fn input() -> Tensor<f32> {
    tensor(zigzag(2 * 3 * 5 * 6 * 6), &[2, 3, 5, 6, 6])
}

fn unique_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fandhe_ai_3824_{tag}_{}", std::process::id()))
}

/// 破棄時に保存先を削除する RAII ガード。
struct DirGuard(PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn avg() -> Sequential {
    Sequential::new().add_adaptive_avg_pool3d(OUT).unwrap()
}

fn max() -> Sequential {
    Sequential::new().add_adaptive_max_pool3d(OUT).unwrap()
}

#[test]
fn add_adaptive_pool3d_rejects_zero_axes() {
    for bad in [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]] {
        assert!(Sequential::new().add_adaptive_avg_pool3d(bad).is_err());
        assert!(Sequential::new().add_adaptive_max_pool3d(bad).is_err());
    }
}

#[test]
fn adaptive_pool3d_predict_and_forward_match_direct_call_bit_exact() {
    let x = input();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);

    let a = avg();
    let direct_a = xv.adaptive_avg_pool3d(OUT).unwrap().to_tensor();
    assert_eq!(dense_vec(&a.predict(&x).unwrap()), dense_vec(&direct_a));
    assert_eq!(
        dense_vec(&a.forward(&tape, &xv).unwrap().to_tensor()),
        dense_vec(&direct_a)
    );
    assert_eq!(direct_a.shape(), &[2, 3, 2, 3, 4]);

    let m = max();
    let (direct_m, _idx) = xv.adaptive_max_pool3d(OUT).unwrap();
    let direct_m = direct_m.to_tensor();
    assert_eq!(dense_vec(&m.predict(&x).unwrap()), dense_vec(&direct_m));
    assert_eq!(
        dense_vec(&m.forward(&tape, &xv).unwrap().to_tensor()),
        dense_vec(&direct_m)
    );
}

#[test]
fn adaptive_pool3d_rejects_wrong_rank_input() {
    let x4 = tensor(ramp(2 * 3 * 6 * 6, 0.01, 0.0), &[2, 3, 6, 6]);
    assert!(avg().predict(&x4).is_err());
    assert!(max().predict(&x4).is_err());
}

fn input_grad(model: &Sequential, direct: bool, is_max: bool) -> Vec<f32> {
    let x_data = input();
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = if direct {
        if is_max {
            x.adaptive_max_pool3d(OUT).unwrap().0
        } else {
            x.adaptive_avg_pool3d(OUT).unwrap()
        }
    } else {
        model.forward(&tape, &x).unwrap()
    };
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    dense_vec(grads.get(&x).unwrap().expect("入力へ勾配が到達するはず"))
}

#[test]
fn adaptive_pool3d_backward_matches_direct_graph_bit_exact() {
    assert_eq!(
        input_grad(&avg(), false, false),
        input_grad(&avg(), true, false)
    );
    assert_eq!(
        input_grad(&max(), false, true),
        input_grad(&max(), true, true)
    );
}

// ---------------------------------------------------------------- 学習 1 step

/// `Conv3d(1→2, k=2) → adaptive pool([2,2,2]) → Flatten → Linear(16→3)`。入力 `[N,1,4,4,4]`。
fn mixed_model(is_max: bool) -> Sequential {
    let m = Sequential::new()
        .add_conv3d(1, 2, [2; 3], [1; 3], [0; 3], [1; 3], 1, SEED_C)
        .unwrap();
    let m = if is_max {
        m.add_adaptive_max_pool3d([2; 3]).unwrap()
    } else {
        m.add_adaptive_avg_pool3d([2; 3]).unwrap()
    };
    m.add_flatten(1, 4).add_linear(16, 3, SEED_L).unwrap()
}

fn train_data() -> (Tensor<f32>, Tensor<f32>) {
    (
        tensor(zigzag(4 * 64), &[4, 1, 4, 4, 4]),
        tensor(ramp(4 * 3, 0.1, 0.0), &[4, 3]),
    )
}

#[test]
fn adaptive_pool3d_does_not_change_trainable_parameters() {
    for is_max in [true, false] {
        let model = mixed_model(is_max);
        let params = model.trainable_parameters();
        assert_eq!(params.len(), 4, "conv(w,b) + linear(w,b)");
        assert_eq!(params[0].shape(), &[2, 1, 2, 2, 2]);
        assert_eq!(params[2].shape(), &[16, 3]);
    }
}

#[test]
fn one_training_step_matches_reference_graph_bit_exact() {
    let (x, target) = train_data();
    for is_max in [true, false] {
        let mut model = mixed_model(is_max);
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
            let p = if is_max {
                h.adaptive_max_pool3d([2; 3]).unwrap().0
            } else {
                h.adaptive_avg_pool3d([2; 3]).unwrap()
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
        assert_eq!(got, want, "max={is_max}");
    }
}

#[test]
fn compile_and_fit_run_with_adaptive_pool3d() {
    let (x, target) = train_data();
    for is_max in [true, false] {
        let mut model = mixed_model(is_max);
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
            .unwrap();
        let w0 = dense_vec(model.trainable_parameters()[0]);
        model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
        assert_ne!(
            dense_vec(model.trainable_parameters()[0]),
            w0,
            "max={is_max}"
        );
    }
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_adaptive_pool3d_fail_closed() {
    let max = max();
    let avg = avg();
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
fn save_model_round_trips_adaptive_pool3d_bit_identically() {
    let model = Sequential::new()
        .add_adaptive_avg_pool3d([2, 3, 4])
        .and_then(|m| m.add_adaptive_max_pool3d([1, 2, 2]))
        .unwrap();
    let dir = unique_dir("roundtrip");
    let _g = DirGuard(dir.clone());
    save_model(&model, &dir).expect("adaptive pool3d は保存できる（#3824）");
    let loaded = load_model(&dir).expect("復元できる");
    let x = input();
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
fn onnx_export_rejects_adaptive_pool3d_as_unsupported() {
    for m in [&avg(), &max()] {
        let err = OnnxModel::from_sequential(m).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}

#[test]
fn adaptive_pool3d_is_mode_independent() {
    let mut model = max();
    let x = input();
    model.train();
    let a = dense_vec(&model.predict(&x).unwrap());
    model.eval();
    let b = dense_vec(&model.predict(&x).unwrap());
    assert_eq!(a, b);
}
