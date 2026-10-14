//! `compat::Sequential::add_conv_transpose3d`（イシュー #3032・親 #3018。公開形の正は
//! `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3）の facade 公開面を検証する統合テスト。
//! `compat_sequential_conv3d.rs`・`compat_sequential_pool3d.rs` と同型。
//!
//! - 無効引数（0 次元・stride 0・`groups` 違反・`output_padding >= stride`・`out_channels < groups`）が
//!   `Err(InvalidArgument)`（panic しない）。
//! - 出力 shape が PyTorch 式と一致し、`predict`／`forward`／`Var::conv_transpose3d` が bit 完全一致。
//! - backward（入力・weight・bias 勾配）と学習 1 step が `Var` 直書きの参照グラフと bit 一致し、
//!   `compile`／`fit` が `first_untracked_parametric_layer` に弾かれない。
//! - 常駐経路は `Unsupported`、保存は後続 issue（kind `conv_transpose3d`）まで `UnsupportedModel`
//!   （fail-closed）、ONNX export は `UnsupportedLayer`。
//!
//! ホストフォールバック経路のため実機（CUDA／Metal）非依存。実機 parity は未実測のまま。

use std::path::PathBuf;

use fandhe_ai::compat::{FitConfig, Loss, ModelIoError, Optimizer, Sequential, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

const SEED1: u64 = 0x3032_0001;
const SEED2: u64 = 0x3032_0002;

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

// ---------------------------------------------------------------- 構築・shape

#[test]
fn invalid_arguments_return_err_without_panic() {
    let (z, one, two) = ([0usize; 3], [1usize; 3], [2usize; 3]);
    let k = [3usize; 3];
    // in_channels = 0
    assert!(
        Sequential::new()
            .add_conv_transpose3d(0, 3, k, one, z, z, one, 1, SEED1)
            .is_err()
    );
    // stride = 0
    assert!(
        Sequential::new()
            .add_conv_transpose3d(2, 3, k, [1, 0, 1], z, z, one, 1, SEED1)
            .is_err()
    );
    // kernel = 0
    assert!(
        Sequential::new()
            .add_conv_transpose3d(2, 3, [3, 0, 3], one, z, z, one, 1, SEED1)
            .is_err()
    );
    // groups 割り切れ違反・groups = 0
    assert!(
        Sequential::new()
            .add_conv_transpose3d(3, 4, k, one, z, z, one, 2, SEED1)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_conv_transpose3d(2, 4, k, one, z, z, one, 0, SEED1)
            .is_err()
    );
    // output_padding >= stride（各軸。意図的な PyTorch 非互換）
    for axis in 0..3 {
        let mut op = z;
        op[axis] = 2;
        assert!(
            Sequential::new()
                .add_conv_transpose3d(2, 3, k, two, z, op, one, 1, SEED1)
                .is_err(),
            "axis {axis}: output_padding == stride"
        );
    }
    // out_channels < groups
    assert!(
        Sequential::new()
            .add_conv_transpose3d(4, 1, k, one, z, z, one, 2, SEED1)
            .is_err()
    );
}

#[test]
fn construction_error_is_autodiff_invalid_argument() {
    let err = Sequential::new()
        .add_conv_transpose3d(0, 3, [3; 3], [1; 3], [0; 3], [0; 3], [1; 3], 1, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn output_shape_matches_pytorch_formula() {
    // (D-1)*s - 2p + d(k-1) + op + 1（軸ごと。groups = 2・dilation・非等方カーネルを含む）。
    let (d, h, w) = (3usize, 4usize, 3usize);
    let (k, s, p, op, dil) = (
        [2usize, 3, 2],
        [2usize, 2, 1],
        [0usize, 1, 1],
        [1usize, 0, 0],
        [1usize, 1, 2],
    );
    let o = |i: usize, a: usize| (i - 1) * s[a] + dil[a] * (k[a] - 1) + op[a] + 1 - 2 * p[a];
    let (od, oh, ow) = (o(d, 0), o(h, 1), o(w, 2));
    let model = Sequential::new()
        .add_conv_transpose3d(4, 6, k, s, p, op, dil, 2, SEED1)
        .unwrap();
    let x = tensor(ramp(2 * 4 * d * h * w, 0.001, -0.2), &[2, 4, d, h, w]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[2, 6, od, oh, ow]);
}

// ---------------------------------------------------------------- forward 一致

const K: [usize; 3] = [3, 2, 3];
const S: [usize; 3] = [2, 1, 2];
const P: [usize; 3] = [1, 0, 1];
const OP: [usize; 3] = [1, 0, 0];
const DIL: [usize; 3] = [1, 1, 1];

fn single_layer() -> Sequential {
    Sequential::new()
        .add_conv_transpose3d(2, 3, K, S, P, OP, DIL, 1, SEED2)
        .unwrap()
}

#[test]
fn predict_matches_var_and_forward_bit_exact() {
    let model = single_layer();
    let w = model.state_dict()["0.weight"].clone();
    let b = model.state_dict()["0.bias"].clone();
    assert_eq!(w.shape(), &[2, 3, 3, 2, 3], "[in, out/groups, kD, kH, kW]");
    let x = tensor(ramp(2 * 2 * 3 * 3 * 3, 0.003, -0.3), &[2, 2, 3, 3, 3]);

    let predicted = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let via_var = xv
        .conv_transpose3d(&wv, Some(&bv), S, P, OP, DIL, 1)
        .unwrap()
        .to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
    assert_eq!(dense_vec(&predicted), dense_vec(&via_var));
}

#[test]
fn backward_grads_match_direct_var_graph_bit_exact() {
    let model = single_layer();
    let w = model.state_dict()["0.weight"].clone();
    let b = model.state_dict()["0.bias"].clone();
    let x = tensor(ramp(2 * 3 * 3 * 3, 0.01, -0.3), &[1, 2, 3, 3, 3]);

    // 層モデル経由。
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let y = bound.forward(&tape, &xv).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let gx = dense_vec(grads.get(&xv).unwrap().unwrap());
    let layer_grads = bound.trainable_grads(&grads).unwrap();

    // 参照: `Var` 直書き。
    let tape2 = fandhe_ai::tape();
    let xv2 = tape2.var(&x);
    let wv2 = tape2.var(&w);
    let bv2 = tape2.var(&b);
    let y2 = xv2
        .conv_transpose3d(&wv2, Some(&bv2), S, P, OP, DIL, 1)
        .unwrap();
    let loss2 = y2.mul(&y2).unwrap().sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();

    assert_eq!(gx, dense_vec(grads2.get(&xv2).unwrap().unwrap()));
    assert_eq!(layer_grads.len(), 2);
    assert_eq!(
        dense_vec(layer_grads[0]),
        dense_vec(grads2.get(&wv2).unwrap().unwrap())
    );
    assert_eq!(
        dense_vec(layer_grads[1]),
        dense_vec(grads2.get(&bv2).unwrap().unwrap())
    );
}

// ---------------------------------------------------------------- 学習経路

fn mixed_model() -> Sequential {
    // ConvTranspose3d → ReLU → Flatten → Linear の混在列（層順カーソルの対応を検査する）。
    // [N,1,2,2,2] → ConvT(k2,s2) → [N,2,4,4,4] → ReLU → Flatten → Linear
    Sequential::new()
        .add_conv_transpose3d(1, 2, [2; 3], [2; 3], [0; 3], [0; 3], [1; 3], 1, SEED1)
        .unwrap()
        .add_relu()
        .add_flatten(1, 4)
        .add_linear(2 * 4 * 4 * 4, 2, SEED2)
        .unwrap()
}

#[test]
fn trainable_parameters_order_is_consistent_across_apis() {
    let model = mixed_model();
    let trainable = model.trainable_parameters();
    // ConvTranspose3d(w,b) + Linear(w,b) の 4 件。
    assert_eq!(trainable.len(), 4);
    assert_eq!(trainable[0].shape(), &[1, 2, 2, 2, 2]);
    assert_eq!(trainable[1].shape(), &[2]);
    assert_eq!(model.state_dict()["0.weight"].shape(), &[1, 2, 2, 2, 2]);

    let x = tensor(ramp(2 * 8, 0.01, -0.4), &[2, 1, 2, 2, 2]);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let pred = bound.forward(&tape, &xv).unwrap();
    let target = tape.var(&tensor(vec![0.0; 2 * 2], &[2, 2]));
    let loss = pred.mse_loss(&target).unwrap();
    assert_eq!(bound.trainable_vars().len(), 4);
    let grads = tape.backward(&loss).unwrap();
    let g = bound.trainable_grads(&grads).unwrap();
    assert_eq!(g.len(), 4);
    assert_eq!(g[0].shape(), trainable[0].shape());
    assert_eq!(g[1].shape(), trainable[1].shape());
}

#[test]
fn apply_parameters_round_trips_and_rejects_shape_change() {
    let mut model = mixed_model();
    let same: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|t| (*t).clone())
        .collect();
    let x = tensor(ramp(8, 0.01, -0.4), &[1, 1, 2, 2, 2]);
    let before = dense_vec(&model.predict(&x).unwrap());
    model.apply_parameters(same.clone()).unwrap();
    assert_eq!(dense_vec(&model.predict(&x).unwrap()), before);

    let mut bad = same;
    bad[0] = tensor(vec![0.0; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);
    assert!(model.apply_parameters(bad).is_err());
    assert_eq!(dense_vec(&model.predict(&x).unwrap()), before);
}

#[test]
fn one_sgd_step_matches_direct_var_reference_bit_exact() {
    let mut model = single_layer();
    let w0 = model.state_dict()["0.weight"].clone();
    let b0 = model.state_dict()["0.bias"].clone();
    let x = tensor(ramp(2 * 3 * 3 * 3, 0.01, -0.3), &[1, 2, 3, 3, 3]);
    let out_shape = model.predict(&x).unwrap().shape().to_vec();
    let target = tensor(ramp(out_shape.iter().product(), 0.001, 0.1), &out_shape);

    // 層モデル: backward → Sgd::step → apply_parameters。
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

    // 参照: `Var` 直書き + 同一の Sgd。
    let mut sgd_ref = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let ref_updated = {
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let wv = tape.var(&w0);
        let bv = tape.var(&b0);
        let tv = tape.var(&target);
        let loss = xv
            .conv_transpose3d(&wv, Some(&bv), S, P, OP, DIL, 1)
            .unwrap()
            .mse_loss(&tv)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let gw = grads.get(&wv).unwrap().unwrap();
        let gb = grads.get(&bv).unwrap().unwrap();
        sgd_ref.step(&[&w0, &b0], &[gw, gb]).unwrap()
    };

    let params = model.trainable_parameters();
    assert_eq!(dense_vec(params[0]), dense_vec(&ref_updated[0]));
    assert_eq!(dense_vec(params[1]), dense_vec(&ref_updated[1]));
    assert_ne!(dense_vec(params[0]), dense_vec(&w0), "weight が更新される");
}

#[test]
fn compile_and_fit_update_conv_transpose3d() {
    let x = tensor(ramp(4 * 8, 0.01, -0.5), &[4, 1, 2, 2, 2]);
    let target = tensor(ramp(4 * 2, 0.1, 0.0), &[4, 2]);
    let mut fit_model = mixed_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let w0 = dense_vec(fit_model.trainable_parameters()[0]);
    fit_model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
    assert_ne!(dense_vec(fit_model.trainable_parameters()[0]), w0);
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_conv_transpose3d_fail_closed() {
    let model = single_layer();
    let tape = fandhe_ai::tape();
    assert!(matches!(
        model.init_device_param_store(&tape).unwrap_err(),
        BackendError::Unsupported(_)
    ));
}

// ---------------------------------------------------------------- 保存・ONNX（fail-closed）

#[test]
fn save_model_is_rejected_until_follow_up_issue() {
    // 保存（kind `conv_transpose3d`・8 キー）は後続 issue。それまで `UnsupportedModel` で拒否し、
    // 保存先に何も作らない。後続で実装する際は本テストを round trip へ置換する。
    let model = single_layer();
    let dir: PathBuf =
        std::env::temp_dir().join(format!("fandhe_ai_3032_save_{}", std::process::id()));
    let err = save_model(&model, &dir).unwrap_err();
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err:?}"
    );
    assert!(!dir.exists(), "拒否時は dir に何も作らない");
}

#[test]
fn onnx_export_rejects_conv_transpose3d_as_unsupported() {
    let model = Sequential::new()
        .add_conv_transpose3d(1, 1, [1; 3], [1; 3], [0; 3], [0; 3], [1; 3], 1, SEED1)
        .unwrap();
    let err = OnnxModel::from_sequential(&model).unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
}
