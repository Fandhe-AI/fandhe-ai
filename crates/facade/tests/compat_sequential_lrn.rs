//! `compat::Sequential::add_local_response_norm`（イシュー #3044・親 #3018。公開形の正は
//! `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3）の facade 公開面を検証する統合テスト。
//! `compat_sequential_fold_unfold.rs`（#3040）と同型。
//!
//! - 無効引数（`size == 0`・非有限な `alpha`／`beta`／`k`）が `Err`（panic しない）。
//! - `predict`／`forward` が `Var::local_response_norm` の直接呼び出しと bit 完全一致
//!   （rank 3／4／5・`size` 奇数／偶数／1／`> C`）。入力 rank 2 以下は `Err`。
//! - 入力勾配が直接グラフと bit 完全一致。学習 1 step が参照グラフと bit 一致し `compile`／`fit` が動く。
//! - 常駐経路は `Unsupported`（fail-closed。決定記録 §5）。保存は kind `local_response_norm` として
//!   保存・復元できる（イシュー #3045）。ONNX export は `UnsupportedLayer`。
//!
//! ホストフォールバック経路のため実機（CUDA／Metal）非依存。実機 parity は未実測のまま（#3171・#3172）。
#![cfg(unix)]

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

const SEED_C: u64 = 0x3044_0001;
const SEED_L: u64 = 0x3044_0002;

const ALPHA: f32 = 1e-2;
const BETA: f32 = 0.75;
const KK: f32 = 1.5;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    dense_vec(t).iter().map(|v| v.to_bits()).collect()
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale + offset).collect()
}

fn input(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(ramp(n, 0.37, -2.1), shape)
}

fn lrn(size: usize) -> Sequential {
    Sequential::new()
        .add_local_response_norm(size, ALPHA, BETA, KK)
        .unwrap()
}

// ---------------------------------------------------------------- 無効引数

#[test]
fn add_local_response_norm_rejects_invalid_arguments() {
    assert!(
        Sequential::new()
            .add_local_response_norm(0, ALPHA, BETA, KK)
            .is_err()
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            Sequential::new()
                .add_local_response_norm(3, bad, BETA, KK)
                .is_err(),
            "alpha={bad}"
        );
        assert!(
            Sequential::new()
                .add_local_response_norm(3, ALPHA, bad, KK)
                .is_err(),
            "beta={bad}"
        );
        assert!(
            Sequential::new()
                .add_local_response_norm(3, ALPHA, BETA, bad)
                .is_err(),
            "k={bad}"
        );
    }
}

// ---------------------------------------------------------------- forward / predict

#[test]
fn predict_and_forward_match_direct_call_bit_exact() {
    // size: 奇数 3／偶数 2／1／チャネル数 C=4 を超える 7。rank: 3／4／5。
    for shape in [vec![2usize, 4, 5], vec![2, 4, 3, 3], vec![2, 4, 2, 2, 3]] {
        for size in [1usize, 2, 3, 7] {
            let model = lrn(size);
            let x = input(&shape);
            let predicted = model.predict(&x).unwrap();
            let tape = fandhe_ai::tape();
            let xv = tape.var(&x);
            let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
            let direct = xv
                .local_response_norm(size, ALPHA, BETA, KK)
                .unwrap()
                .to_tensor();
            assert_eq!(predicted.shape(), x.shape(), "shape={shape:?} size={size}");
            assert_eq!(
                bits(&predicted),
                bits(&direct),
                "shape={shape:?} size={size}"
            );
            assert_eq!(
                bits(&forwarded),
                bits(&direct),
                "shape={shape:?} size={size}"
            );
        }
    }
}

#[test]
fn reject_wrong_rank_input() {
    let model = lrn(3);
    assert!(model.predict(&input(&[4, 6])).is_err());
    assert!(model.predict(&input(&[6])).is_err());
}

// ---------------------------------------------------------------- backward

#[test]
fn backward_matches_direct_graph_bit_exact() {
    let x_data = input(&[2, 4, 3, 3]);
    let model = lrn(3);
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let loss1 = model.forward(&tape1, &x1).unwrap().sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1.get(&x1).unwrap().expect("入力へ勾配が到達するはず");

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let loss2 = x2
        .local_response_norm(3, ALPHA, BETA, KK)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2.get(&x2).unwrap().expect("入力へ勾配が到達するはず");
    assert_eq!(bits(dx1), bits(dx2));
}

// ---------------------------------------------------------------- 学習 1 step

/// `Conv2d(1→2, k=2) → LocalResponseNorm(3) → Flatten(1,3) → Linear(18→3)`（入力 `[N,1,4,4]`。
/// conv 出力 `[N,2,3,3]`。`size=3 > C=2` も受理される）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 2, [2; 2], [1; 2], [0; 2], [1; 2], 1, SEED_C)
        .unwrap()
        .add_local_response_norm(3, ALPHA, BETA, KK)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(18, 3, SEED_L)
        .unwrap()
}

fn train_data() -> (Tensor<f32>, Tensor<f32>) {
    (
        tensor(ramp(4 * 16, 0.02, -0.5), &[4, 1, 4, 4]),
        tensor(ramp(4 * 3, 0.1, 0.0), &[4, 3]),
    )
}

#[test]
fn lrn_does_not_change_trainable_parameters() {
    let model = mixed_model();
    let params = model.trainable_parameters();
    assert_eq!(params.len(), 4, "conv(w,b) + linear(w,b)");
    assert_eq!(params[0].shape(), &[2, 1, 2, 2]);
    assert_eq!(params[2].shape(), &[18, 3]);
}

#[test]
fn one_training_step_matches_reference_graph_bit_exact() {
    let (x, target) = train_data();
    let mut model = mixed_model();
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

    // 参照: 同一 seed の LRN なし 2 分割モデルへ `Var` メソッドで直接挟む。
    let a = Sequential::new()
        .add_conv2d(1, 2, [2; 2], [1; 2], [0; 2], [1; 2], 1, SEED_C)
        .unwrap();
    let b = Sequential::new()
        .add_flatten(1, 3)
        .add_linear(18, 3, SEED_L)
        .unwrap();
    let mut sgd_ref = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let updated_ref = {
        let tape = fandhe_ai::tape();
        let ba = a.bind(&tape);
        let bb = b.bind(&tape);
        let xv = tape.var(&x);
        let tv = tape.var(&target);
        let h = ba.forward(&tape, &xv).unwrap();
        let n = h.local_response_norm(3, ALPHA, BETA, KK).unwrap();
        let loss = bb.forward(&tape, &n).unwrap().mse_loss(&tv).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let mut grad_refs = ba.trainable_grads(&grads).unwrap();
        grad_refs.extend(bb.trainable_grads(&grads).unwrap());
        let mut param_refs = a.trainable_parameters();
        param_refs.extend(b.trainable_parameters());
        sgd_ref.step(&param_refs, &grad_refs).unwrap()
    };

    let got: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|t| bits(t))
        .collect();
    let want: Vec<Vec<u32>> = updated_ref.iter().map(bits).collect();
    assert_eq!(got, want);
}

#[test]
fn compile_and_fit_run_with_lrn() {
    let (x, target) = train_data();
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let w0 = dense_vec(model.trainable_parameters()[0]);
    model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
    assert_ne!(dense_vec(model.trainable_parameters()[0]), w0);
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_lrn_fail_closed() {
    let model = lrn(3);
    let linear_only = Sequential::new().add_linear(6, 6, SEED_L).unwrap();
    let init_tape = fandhe_ai::tape();
    let mut store = linear_only.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let x = input(&[2, 4, 3, 3]);
    let tape = fandhe_ai::tape();
    assert!(matches!(
        model.init_device_param_store(&tape).unwrap_err(),
        BackendError::Unsupported(_)
    ));
    let xv = tape.var(&x);
    assert!(matches!(
        model.forward_resident(&tape, &xv, &mut store).unwrap_err(),
        AutodiffError::Backend(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        model.predict_resident(&store, &x).unwrap_err(),
        AutodiffError::Backend(BackendError::Unsupported(_))
    ));
}

// ---------------------------------------------------------------- 保存・ONNX（fail-closed）

/// イシュー #3045: 保存・復元（`local_response_norm` kind）。predict が bit 一致で復元される。
#[test]
fn save_model_round_trips_local_response_norm() {
    let mut model = lrn(3);
    model.eval();
    let guard = TempDirGuard::new("lrn_save_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    assert!(dir.exists());
    let loaded = load_model(&dir).expect("復元できるはず");
    let x = input(&[2, 4, 3, 3]);
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );
}

#[test]
fn onnx_export_rejects_local_response_norm_as_unsupported() {
    let model = lrn(3);
    let err = OnnxModel::from_sequential(&model).unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
}

// ---------------------------------------------------------------- train／eval 非依存

#[test]
fn lrn_is_mode_independent() {
    let mut model = lrn(3);
    let x = input(&[2, 4, 3, 3]);
    model.train();
    let a = bits(&model.predict(&x).unwrap());
    model.eval();
    let b = bits(&model.predict(&x).unwrap());
    assert_eq!(a, b);
}
