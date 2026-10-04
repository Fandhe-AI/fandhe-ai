//! `compat::Sequential::add_conv_transpose2d`（イシュー #2523・親 #2520・ルート #2499 の
//! 一括承認）の facade 公開面を検証する統合テスト。設計正本は `docs/conv-ops-design.md` §15。
//!
//! - 無効引数（`output_padding >= stride`・0 の stride／in_channels・groups 割り切れ違反）は `Err`。
//! - `predict` の出力 shape が PyTorch の式と一致し、`Var::conv_transpose2d` と bit 一致する。
//! - 学習経路（bind／forward／trainable_*／apply_parameters／fit）の層順対応。
//! - 常駐経路は `BackendError::Unsupported`（fail-closed）。
//! - `save_model`／`load_model` の bit 一致往復と改竄 manifest の拒否。
//! - ONNX export は `OnnxError::UnsupportedLayer`。

use std::path::PathBuf;

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

const SEED1: u64 = 0x2523_0001;
const SEED2: u64 = 0x2523_0002;

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

fn unique_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fandhe_ai_2523_{tag}_{}", std::process::id()))
}

/// 破棄時に保存先を削除する RAII ガード。
struct DirGuard(PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------- 構築拒否・shape

#[test]
fn add_conv_transpose2d_rejects_invalid_arguments() {
    let one = [1usize, 1];
    // output_padding >= stride
    assert!(
        Sequential::new()
            .add_conv_transpose2d(2, 3, [3, 3], [2, 2], [0, 0], [2, 0], one, 1, SEED1)
            .is_err()
    );
    // in_channels = 0
    assert!(
        Sequential::new()
            .add_conv_transpose2d(0, 3, [3, 3], one, [0, 0], [0, 0], one, 1, SEED1)
            .is_err()
    );
    // stride = 0
    assert!(
        Sequential::new()
            .add_conv_transpose2d(2, 3, [3, 3], [0, 1], [0, 0], [0, 0], one, 1, SEED1)
            .is_err()
    );
    // groups 割り切れ違反・groups = 0
    assert!(
        Sequential::new()
            .add_conv_transpose2d(3, 4, [3, 3], one, [0, 0], [0, 0], one, 2, SEED1)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_conv_transpose2d(2, 4, [3, 3], one, [0, 0], [0, 0], one, 0, SEED1)
            .is_err()
    );
}

#[test]
fn construction_error_is_autodiff_invalid_argument() {
    let err = Sequential::new()
        .add_conv_transpose2d(2, 3, [3, 3], [2, 2], [0, 0], [2, 2], [1, 1], 1, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn conv_transpose2d_output_shape_matches_pytorch_formula() {
    // (H-1)*s - 2p + d(k-1) + op + 1（軸ごと。groups = 2・dilation = 2 を含む）。
    let (h, w) = (4usize, 5usize);
    let (k, s, p, op, d) = (
        [3usize, 2],
        [2usize, 3],
        [1usize, 0],
        [1usize, 2],
        [2usize, 1],
    );
    let oh = (h - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + op[0] + 1;
    let ow = (w - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + op[1] + 1;
    let model = Sequential::new()
        .add_conv_transpose2d(4, 6, k, s, p, op, d, 2, SEED1)
        .unwrap();
    let x = tensor(ramp(2 * 4 * h * w, 0.01, -0.2), &[2, 4, h, w]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[2, 6, oh, ow]);
}

#[test]
fn conv_transpose2d_predict_matches_var_and_forward_bit_exact() {
    let model = Sequential::new()
        .add_conv_transpose2d(2, 3, [3, 3], [2, 2], [1, 1], [1, 1], [1, 1], 1, SEED2)
        .unwrap();
    let w = model.state_dict()["0.weight"].clone();
    let b = model.state_dict()["0.bias"].clone();
    assert_eq!(w.shape(), &[2, 3, 3, 3], "[in, out/groups, kH, kW]");
    let x = tensor(ramp(2 * 2 * 4 * 4, 0.02, -0.3), &[2, 2, 4, 4]);

    let predicted = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let via_var = xv
        .conv_transpose2d(&wv, Some(&bv), [2, 2], [1, 1], [1, 1], [1, 1], 1)
        .unwrap()
        .to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
    assert_eq!(dense_vec(&predicted), dense_vec(&via_var));
}

// ---------------------------------------------------------------- 学習経路

fn mixed_model() -> Sequential {
    // Conv2d → ConvTranspose2d → Flatten → Linear の混在列（層順カーソルの対応を検査する）。
    // [N,1,4,4] → [N,2,4,4] → ConvT(k3,s2,p1,op1) → [N,3,8,8] → Flatten → Linear
    Sequential::new()
        .add_conv2d(1, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, SEED1)
        .unwrap()
        .add_conv_transpose2d(2, 3, [3, 3], [2, 2], [1, 1], [1, 1], [1, 1], 1, SEED2)
        .unwrap()
        .add_relu()
        .add_flatten(1, 3)
        .add_linear(3 * 8 * 8, 2, SEED1)
        .unwrap()
}

#[test]
fn trainable_parameters_order_is_consistent_across_apis() {
    let model = mixed_model();
    let trainable = model.trainable_parameters();
    // Conv2d(w,b) + ConvTranspose2d(w,b) + Linear(w,b) の 6 件。
    assert_eq!(trainable.len(), 6);
    assert_eq!(trainable[2].shape(), &[2, 3, 3, 3]);
    assert_eq!(trainable[3].shape(), &[3]);
    assert_eq!(model.state_dict()["1.weight"].shape(), &[2, 3, 3, 3]);

    let x = tensor(ramp(2 * 16, 0.05, -0.4), &[2, 1, 4, 4]);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let pred = bound.forward(&tape, &xv).unwrap();
    let target = tape.var(&tensor(vec![0.0; 2 * 2], &[2, 2]));
    let loss = pred.mse_loss(&target).unwrap();
    assert_eq!(bound.trainable_vars().len(), 6);
    let grads = tape.backward(&loss).unwrap();
    let g = bound.trainable_grads(&grads).unwrap();
    assert_eq!(g.len(), 6);
    assert_eq!(g[2].shape(), trainable[2].shape());
    assert_eq!(g[3].shape(), trainable[3].shape());
}

#[test]
fn apply_parameters_round_trips_and_rejects_shape_change() {
    let mut model = mixed_model();
    let same: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|t| (*t).clone())
        .collect();
    let x = tensor(ramp(16, 0.05, -0.4), &[1, 1, 4, 4]);
    let before = dense_vec(&model.predict(&x).unwrap());
    model.apply_parameters(same.clone()).unwrap();
    assert_eq!(dense_vec(&model.predict(&x).unwrap()), before);

    let mut bad = same;
    bad[2] = tensor(vec![0.0; 2 * 3 * 3 * 2], &[2, 3, 3, 2]);
    assert!(model.apply_parameters(bad).is_err());
    assert_eq!(dense_vec(&model.predict(&x).unwrap()), before);
}

#[test]
fn sgd_step_updates_conv_transpose2d_and_fit_runs() {
    let mut model = mixed_model();
    let x = tensor(ramp(4 * 16, 0.03, -0.5), &[4, 1, 4, 4]);
    let target = tensor(ramp(4 * 2, 0.1, 0.0), &[4, 2]);
    let before = dense_vec(model.trainable_parameters()[2]);

    let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let updated = {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let xv = tape.var(&x);
        let tv = tape.var(&target);
        let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
        assert!(loss.to_tensor().get(&[]).unwrap().is_finite());
        let grads = tape.backward(&loss).unwrap();
        let grad_refs = bound.trainable_grads(&grads).unwrap();
        let param_refs = model.trainable_parameters();
        sgd.step(&param_refs, &grad_refs).unwrap()
    };
    model.apply_parameters(updated).unwrap();
    assert_ne!(dense_vec(model.trainable_parameters()[2]), before);

    // compile + fit が first_untracked_parametric_layer に弾かれず学習される。
    let mut fit_model = mixed_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let w0 = dense_vec(fit_model.trainable_parameters()[2]);
    fit_model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
    assert_ne!(dense_vec(fit_model.trainable_parameters()[2]), w0);
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_conv_transpose2d_fail_closed() {
    let model = Sequential::new()
        .add_conv_transpose2d(2, 3, [3, 3], [1, 1], [0, 0], [0, 0], [1, 1], 1, SEED1)
        .unwrap();
    let tape = fandhe_ai::tape();
    assert!(matches!(
        model.init_device_param_store(&tape).unwrap_err(),
        BackendError::Unsupported(_)
    ));
}

// ---------------------------------------------------------------- 保存・復元

fn model_for_save() -> Sequential {
    let mut m = Sequential::new()
        .add_conv_transpose2d(4, 6, [3, 2], [2, 2], [1, 0], [1, 0], [1, 1], 2, SEED1)
        .unwrap()
        .add_relu();
    m.eval();
    m
}

fn read_all_text(dir: &PathBuf) -> Vec<(PathBuf, String)> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter_map(|p| std::fs::read_to_string(&p).ok().map(|s| (p, s)))
        .collect()
}

#[test]
fn save_and_load_round_trip_bit_identically() {
    let model = model_for_save();
    let dir = unique_dir("rt");
    let _g = DirGuard(dir.clone());
    save_model(&model, &dir).unwrap();
    assert!(
        read_all_text(&dir)
            .iter()
            .any(|(_, s)| s.contains("conv_transpose2d")),
        "manifest の kind が conv_transpose2d"
    );

    let loaded = load_model(&dir).unwrap();
    let x = tensor(ramp(4 * 3 * 3, 0.07, -0.4), &[1, 4, 3, 3]);
    assert_eq!(
        dense_vec(&model.predict(&x).unwrap()),
        dense_vec(&loaded.predict(&x).unwrap())
    );
    let loaded_state = loaded.state_dict();
    for (k, v) in model.state_dict() {
        assert_eq!(dense_vec(&v), dense_vec(&loaded_state[&k]), "{k}");
    }
}

#[test]
fn tampered_manifest_is_rejected() {
    let model = model_for_save();
    let dir = unique_dir("tamper");
    let _g = DirGuard(dir.clone());
    save_model(&model, &dir).unwrap();
    let (manifest_path, original) = read_all_text(&dir)
        .into_iter()
        .find(|(_, s)| s.contains("conv_transpose2d"))
        .expect("manifest ファイル");
    for (from, to) in [
        // 未知キー追加
        ("\"output_padding_h\"", "\"bogus\":1,\"output_padding_h\""),
        // 欠落（キー名を差し替えて存在しない扱いにする）
        ("\"output_padding_h\"", "\"output_padding_x\""),
        // output_padding >= stride（stride_h=2 に対し 2）
        ("\"output_padding_h\":1", "\"output_padding_h\":2"),
        // groups 割り切れ違反
        ("\"groups\":2", "\"groups\":4"),
    ] {
        assert!(original.contains(from), "fixture: {from}");
        std::fs::write(&manifest_path, original.replacen(from, to, 1)).unwrap();
        assert!(
            load_model(&dir).is_err(),
            "改竄 {from} -> {to} を拒否するはず"
        );
    }
}

// ---------------------------------------------------------------- ONNX

#[test]
fn onnx_export_rejects_conv_transpose2d_as_unsupported() {
    let model = Sequential::new()
        .add_conv_transpose2d(1, 1, [1, 1], [1, 1], [0, 0], [0, 0], [1, 1], 1, SEED1)
        .unwrap();
    let err = OnnxModel::from_sequential(&model).unwrap_err();
    assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
}
