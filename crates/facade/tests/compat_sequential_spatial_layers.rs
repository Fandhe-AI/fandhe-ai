//! `compat::Sequential::add_conv_transpose1d`／`add_unflatten`（イシュー #2521・
//! 親 #2520・ルート #2499 の一括承認）の facade 公開面を検証する統合テスト。
//! 設計正本は `docs/autodiff-spatial-layers-decision.md` §6。
//!
//! - 無効引数（`output_padding >= stride`・`groups = 0`・割り切れ違反・空 `sizes`）は `Err`。
//! - `predict`（tape 不要経路）と `forward`（`fandhe_ai::tape()` 上）が bit 完全一致し、
//!   `ConvTranspose1d` の出力長は PyTorch の式と一致する。
//! - `Flatten` → `Unflatten` の往復で shape と値が元に戻る。
//! - 学習経路: `trainable_parameters`／`bind().trainable_vars()`／`trainable_grads()`／
//!   `state_dict` の順序・件数の一致、`apply_parameters` の shape 保存更新、SGD で loss 減少。
//! - 常駐経路: `ConvTranspose1d` は `BackendError::Unsupported`、`Unflatten` のみは通過。
//! - 保存・ONNX export: manifest／opset が未対応のため型付きエラーで fail-closed。

use std::path::PathBuf;

use fandhe_ai::compat::{FitConfig, Loss, ModelIoError, Optimizer, Sequential, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

const SEED1: u64 = 0x2521_0001;
const SEED2: u64 = 0x2521_0002;

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

/// 保存先として存在しないパスを返す（`save_model` が拒否時に何も作らないことの検査用）。
fn unique_missing_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fandhe_ai_2521_{tag}_{}_{}",
        std::process::id(),
        SEED1
    ))
}

// ---------------------------------------------------------------- add_conv_transpose1d

#[test]
fn add_conv_transpose1d_rejects_output_padding_ge_stride() {
    let err = Sequential::new()
        .add_conv_transpose1d(2, 3, 3, 2, 0, 2, 1, 1, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn add_conv_transpose1d_rejects_zero_groups_and_indivisible_groups() {
    assert!(
        Sequential::new()
            .add_conv_transpose1d(2, 4, 3, 1, 0, 0, 1, 0, SEED1)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_conv_transpose1d(3, 4, 3, 1, 0, 0, 1, 2, SEED1)
            .is_err()
    );
}

#[test]
fn conv_transpose1d_output_length_matches_pytorch_formula() {
    // (L-1)*stride - 2*padding + dilation*(k-1) + output_padding + 1
    let (l, stride, padding, k, dilation, output_padding) =
        (5usize, 3usize, 1usize, 3usize, 2usize, 2usize);
    let expected = (l - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;
    let model = Sequential::new()
        .add_conv_transpose1d(2, 4, k, stride, padding, output_padding, dilation, 1, SEED1)
        .unwrap();
    let x = tensor(ramp(2 * 2 * l, 0.03, -0.2), &[2, 2, l]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[2, 4, expected]);
}

#[test]
fn conv_transpose1d_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_conv_transpose1d(2, 3, 3, 2, 1, 1, 1, 1, SEED2)
        .unwrap()
        .add_relu();
    let x = tensor(ramp(2 * 2 * 6, 0.02, -0.3), &[2, 2, 6]);

    let predicted = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(predicted.shape(), forwarded.shape());
    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

// ---------------------------------------------------------------- add_unflatten

#[test]
fn add_unflatten_rejects_empty_sizes() {
    let err = Sequential::new()
        .add_unflatten(1, vec![])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn flatten_then_unflatten_round_trips() {
    let model = Sequential::new()
        .add_flatten(1, 2)
        .add_unflatten(1, vec![3, 4])
        .unwrap();
    let x = tensor(ramp(2 * 3 * 4, 0.5, 0.0), &[2, 3, 4]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[2, 3, 4]);
    assert_eq!(dense_vec(&y), dense_vec(&x));
}

#[test]
fn unflatten_size_product_mismatch_is_shape_error_at_forward() {
    let model = Sequential::new().add_unflatten(1, vec![3, 3]).unwrap();
    let x = tensor(ramp(2 * 8, 1.0, 0.0), &[2, 8]);
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)), "{err:?}");
}

// ---------------------------------------------------------------- 学習経路

fn mixed_model() -> Sequential {
    // [N, 6] → Linear(6, 8) → Unflatten(1, [2, 4]) → ConvTranspose1d(2, 3, k=3, s=2) → ReLU
    Sequential::new()
        .add_linear(6, 8, SEED1)
        .unwrap()
        .add_unflatten(1, vec![2, 4])
        .unwrap()
        .add_conv_transpose1d(2, 3, 3, 2, 0, 0, 1, 1, SEED2)
        .unwrap()
        .add_relu()
}

#[test]
fn trainable_parameters_order_is_consistent_across_apis() {
    let model = mixed_model();
    let trainable = model.trainable_parameters();
    // Linear(weight, bias) + ConvTranspose1d(weight, bias) の 4 件。
    assert_eq!(trainable.len(), 4);
    assert_eq!(trainable[0].shape(), &[6, 8]);
    assert_eq!(trainable[1].shape(), &[8]);
    assert_eq!(trainable[2].shape(), &[2, 3, 3], "[in, out/groups, k]");
    assert_eq!(trainable[3].shape(), &[3]);

    let state = model.state_dict();
    assert!(state.contains_key("0.weight") && state.contains_key("0.bias"));
    assert!(state.contains_key("2.weight") && state.contains_key("2.bias"));
    assert_eq!(state["2.weight"].shape(), &[2, 3, 3]);

    let x = tensor(ramp(3 * 6, 0.05, -0.4), &[3, 6]);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let pred = bound.forward(&tape, &xv).unwrap();
    let target = tape.var(&tensor(vec![0.0; 3 * 3 * 9], &[3, 3, 9]));
    let loss = pred.mse_loss(&target).unwrap();
    assert_eq!(bound.trainable_vars().len(), 4);
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(bound.trainable_grads(&grads).unwrap().len(), 4);
}

#[test]
fn apply_parameters_updates_conv_transpose1d_and_rejects_bad_updates() {
    let mut model = Sequential::new()
        .add_conv_transpose1d(1, 1, 1, 1, 0, 0, 1, 1, SEED1)
        .unwrap();
    model
        .apply_parameters(vec![tensor(vec![2.0], &[1, 1, 1]), tensor(vec![1.0], &[1])])
        .unwrap();
    let x = tensor(vec![3.0], &[1, 1, 1]);
    assert_eq!(dense_vec(&model.predict(&x).unwrap()), vec![7.0]);

    // shape 変更・要素数不足はモデル不変のまま拒否する。
    let before = dense_vec(&model.predict(&x).unwrap());
    assert!(
        model
            .apply_parameters(vec![
                tensor(vec![1.0; 2], &[1, 1, 2]),
                tensor(vec![0.0], &[1])
            ])
            .is_err()
    );
    assert!(
        model
            .apply_parameters(vec![tensor(vec![1.0], &[1, 1, 1])])
            .is_err()
    );
    assert_eq!(dense_vec(&model.predict(&x).unwrap()), before);
}

#[test]
fn train_loop_with_sgd_reduces_loss_and_updates_conv_transpose1d() {
    let mut model = Sequential::new()
        .add_linear(6, 8, SEED1)
        .unwrap()
        .add_unflatten(1, vec![2, 4])
        .unwrap()
        .add_conv_transpose1d(2, 3, 3, 2, 0, 0, 1, 1, SEED2)
        .unwrap();
    let x = tensor(
        (0..4 * 6).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect(),
        &[4, 6],
    );
    let target = tensor(
        (0..4 * 3 * 9)
            .map(|i| ((i % 5) as f32) * 0.1 - 0.2)
            .collect(),
        &[4, 3, 9],
    );
    let conv_weight_before = dense_vec(model.trainable_parameters()[2]);

    let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let mut losses = Vec::new();
    for _ in 0..40 {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let pred = bound.forward(&tape, &xv).unwrap();
            let loss = pred.mse_loss(&tv).unwrap();
            losses.push(loss.to_tensor().get(&[]).unwrap());
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }
    let (first, last) = (losses[0], *losses.last().unwrap());
    assert!(
        last < first * 0.9,
        "loss should decrease: {first} -> {last}"
    );
    assert_ne!(
        dense_vec(model.trainable_parameters()[2]),
        conv_weight_before,
        "ConvTranspose1d の weight が更新されていない"
    );
}

#[test]
fn compile_and_fit_accept_conv_transpose1d_model() {
    // `first_untracked_parametric_layer` が ConvTranspose1d を「追跡されない独自層」と
    // 誤判定しないこと（fit の入口検査を通り、weight が学習されること）を固定する。
    let mut model = mixed_model();
    let before = dense_vec(model.trainable_parameters()[2]);
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let x = tensor(ramp(4 * 6, 0.05, -0.5), &[4, 6]);
    let y = tensor(ramp(4 * 3 * 9, 0.01, 0.1), &[4, 3, 9]);
    model.fit(&x, &y, FitConfig::new(3, 4)).unwrap();
    assert_ne!(dense_vec(model.trainable_parameters()[2]), before);
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_conv_transpose1d_but_not_unflatten() {
    let model = Sequential::new()
        .add_conv_transpose1d(2, 3, 3, 1, 0, 0, 1, 1, SEED1)
        .unwrap();
    let tape = fandhe_ai::tape();
    assert!(matches!(
        model.init_device_param_store(&tape).unwrap_err(),
        BackendError::Unsupported(_)
    ));

    // Linear + Unflatten のみのモデルは常駐経路を拒否されずに通過する。
    let ok_model = Sequential::new()
        .add_linear(4, 6, SEED1)
        .unwrap()
        .add_unflatten(1, vec![2, 3])
        .unwrap();
    let init_tape = fandhe_ai::tape();
    let store = ok_model
        .init_device_param_store(&init_tape)
        .expect("Unflatten のみは常駐経路を拒否しない");
    drop(init_tape);
    let x = tensor(ramp(2 * 4, 0.1, -0.3), &[2, 4]);
    let resident = ok_model.predict_resident(&store, &x).unwrap();
    let host = ok_model.predict(&x).unwrap();
    assert_eq!(resident.shape(), host.shape());
    for (a, b) in dense_vec(&resident).iter().zip(dense_vec(&host)) {
        let diff = (a - b).abs();
        assert!(diff < 1e-5 || diff / b.abs().max(f32::MIN_POSITIVE) < 1e-3);
    }
}

// ---------------------------------------------------------------- 保存・ONNX（fail-closed）

#[test]
fn save_model_rejects_both_layers_without_touching_dir() {
    for (tag, model) in [
        (
            "ct1d",
            Sequential::new()
                .add_conv_transpose1d(1, 1, 1, 1, 0, 0, 1, 1, SEED1)
                .unwrap(),
        ),
        (
            "unflatten",
            Sequential::new().add_unflatten(1, vec![1, 2]).unwrap(),
        ),
    ] {
        let dir = unique_missing_dir(tag);
        let err = save_model(&model, &dir).unwrap_err();
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{tag}: {err:?}"
        );
        assert!(!dir.exists(), "{tag}: 拒否時に保存先を作ってはならない");
    }
}

#[test]
fn onnx_export_rejects_both_layers_as_unsupported() {
    for model in [
        Sequential::new()
            .add_conv_transpose1d(1, 1, 1, 1, 0, 0, 1, 1, SEED1)
            .unwrap(),
        Sequential::new().add_unflatten(1, vec![1, 2]).unwrap(),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
// ---------------------------------------------------------------- Var 委譲メソッド

#[test]
fn var_methods_match_layer_forward_and_backpropagate() {
    let model = Sequential::new()
        .add_conv_transpose1d(2, 3, 3, 2, 0, 1, 1, 1, SEED1)
        .unwrap();
    let w = model.state_dict()["0.weight"].clone();
    let b = model.state_dict()["0.bias"].clone();
    let x = tensor(ramp(2 * 2 * 4, 0.04, -0.2), &[2, 2, 4]);

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let wv = tape.var(&w);
    let bv = tape.var(&b);
    let via_var = xv.conv_transpose1d(&wv, Some(&bv), 2, 0, 1, 1, 1).unwrap();
    let via_layer = model.forward(&tape, &xv).unwrap();
    assert_eq!(
        dense_vec(&via_var.to_tensor()),
        dense_vec(&via_layer.to_tensor())
    );

    // backward: unflatten は勾配を恒等に流し、重みへ勾配が届く。
    let flat = tape.var(&tensor(ramp(2 * 6, 0.1, 0.0), &[2, 6]));
    let un = flat.unflatten(1, &[2, 3]).unwrap();
    assert_eq!(un.to_tensor().shape(), &[2, 2, 3]);
    let loss = via_var
        .sum(None)
        .unwrap()
        .add(&un.sum(None).unwrap())
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&wv).unwrap().is_some(), "weight 勾配が得られない");
    let g_flat = grads.get(&flat).unwrap().expect("unflatten 入力への勾配");
    assert!(g_flat.host_slice().iter().all(|&g| g == 1.0));
}
