//! `compat::Sequential::add_transformer`（イシュー #2533・親 #2531・ルート #2499 の一括承認）の
//! facade 公開面を検証する統合テスト。設計正本は `docs/autodiff-transformer-decoder-decision.md`。
//!
//! - 無効引数（0 層・割り切れない head 数・0 次元・非 relu 活性化）は `Err`（fail-closed）。
//! - `predict` と `bind().forward` が bit 一致し、`nn::Transformer` の手動合成とも bit 一致する。
//! - パラメータ数は `16 * N_enc + 2 + 26 * N_dec + 2`。`trainable_vars`／`trainable_grads` の
//!   順序が `named_parameters` と一致する。
//! - 学習経路（bind／forward／trainable_*／apply_parameters／fit）。
//! - 常駐経路は `BackendError::Unsupported`、ONNX export は `OnnxError::UnsupportedLayer`。
//! - `save_model`／`load_model` の bit 一致往復と改竄 manifest の拒否（巨大な層数を含む）。
//!
//! 実機（CUDA／Metal）は不要（CPU のみ。実機 parity は `compat_sequential_layers_backend_parity.rs`
//! の `#[ignore]` テスト）。

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::nn::TransformerConfig;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

#[cfg(unix)]
mod common;

const SEED1: u64 = 0x2533_0001;
const SEED2: u64 = 0x2533_0002;

/// 縮小構成（既定の 6／6／2048 は大きいため必ず縮める）。
fn small(enc: usize, dec: usize) -> TransformerConfig {
    TransformerConfig::new(4, 2)
        .with_num_encoder_layers(enc)
        .with_num_decoder_layers(dec)
        .with_dim_feedforward(8)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn input() -> Tensor<f32> {
    tensor(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    )
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    dense_vec(t).iter().map(|v| v.to_bits()).collect()
}

fn is_invalid_argument<T>(r: Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

// ---------------------------------------------------------------- 構築拒否

#[test]
fn add_transformer_rejects_invalid_configs() {
    // 0 層（encoder／decoder）
    assert!(is_invalid_argument(
        Sequential::new().add_transformer(small(0, 1), SEED1)
    ));
    assert!(is_invalid_argument(
        Sequential::new().add_transformer(small(1, 0), SEED1)
    ));
    // head 数が割り切れない
    assert!(is_invalid_argument(
        Sequential::new().add_transformer(TransformerConfig::new(6, 4), SEED1)
    ));
    // dim_feedforward = 0・d_model = 0
    assert!(is_invalid_argument(
        Sequential::new().add_transformer(small(1, 1).with_dim_feedforward(0), SEED1)
    ));
    assert!(is_invalid_argument(
        Sequential::new().add_transformer(TransformerConfig::new(0, 1), SEED1)
    ));
}

#[test]
fn add_transformer_rejects_non_relu_activation() {
    // `FeedForwardActivation` は facade から到達できないため内部クレート経由で作る。
    // 保存 manifest が活性化を持たないので Gelu は構築時に fail-closed で拒否する。
    let cfg = small(1, 1).with_activation(fandhe_ai_autodiff::nn::FeedForwardActivation::Gelu);
    assert!(is_invalid_argument(
        Sequential::new().add_transformer(cfg, SEED1)
    ));
}

// ---------------------------------------------------------------- 順伝播・bit 一致

#[test]
fn predict_matches_forward_and_manual_composition_bit_exact() {
    let cfg = small(1, 1);
    let model = Sequential::new().add_transformer(cfg, SEED1).unwrap();
    let x = input();

    let predicted = model.predict(&x).unwrap();
    assert_eq!(predicted.shape(), &[2, 3, 4]);

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(bits(&predicted), bits(&forwarded));

    // `nn::Transformer` の手動合成（src = tgt = x・mask なし・非 causal）と bit 一致する。
    // facade の既定 tape と同じ CPU バックエンド演算で合成する（`Tape::new()` は参照実装で
    // ulp 差が出るため bit 一致の比較対象にならない）。
    let raw = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
        fandhe_ai_backend_cpu::CpuBackendOps::new(),
    ));
    let manual = fandhe_ai::nn::Transformer::new(&cfg, SEED1).unwrap();
    let xr = raw.var(&x);
    let out = manual
        .bind(&raw)
        .forward(&xr, &xr, None, None, None, false)
        .unwrap()
        .to_tensor();
    assert_eq!(bits(&predicted), bits(&out));
}

// ---------------------------------------------------------------- パラメータ・順序

#[test]
fn trainable_parameter_count_follows_layer_counts() {
    let m11 = Sequential::new()
        .add_transformer(small(1, 1), SEED1)
        .unwrap();
    assert_eq!(m11.trainable_parameters().len(), 16 + 2 + 26 + 2);
    let m22 = Sequential::new()
        .add_transformer(small(2, 2), SEED1)
        .unwrap();
    assert_eq!(m22.trainable_parameters().len(), 2 * 16 + 2 + 2 * 26 + 2);
    let m13 = Sequential::new()
        .add_transformer(small(1, 3), SEED1)
        .unwrap();
    assert_eq!(m13.trainable_parameters().len(), 16 + 2 + 3 * 26 + 2);
}

#[test]
fn trainable_vars_and_grads_follow_named_parameters_order() {
    let model = Sequential::new()
        .add_transformer(small(2, 2), SEED1)
        .unwrap();
    let x = input();
    let target = tensor(vec![0.0f32; 2 * 3 * 4], &[2, 3, 4]);

    let named_shapes: Vec<Vec<usize>> = model
        .named_parameters()
        .iter()
        .map(|(_, t)| t.contiguous().shape().to_vec())
        .collect();
    let trainable_shapes: Vec<Vec<usize>> = model
        .trainable_parameters()
        .iter()
        .map(|t| t.contiguous().shape().to_vec())
        .collect();
    assert_eq!(named_shapes, trainable_shapes);

    // 並びは encoder 層 → encoder.norm → decoder 層 → decoder.norm。
    let names: Vec<String> = model
        .named_parameters()
        .iter()
        .map(|(n, _)| n.clone())
        .collect();
    assert!(names[0].ends_with("encoder.layers.0.self_attn.q_proj.weight"));
    assert!(names.iter().any(|n| n.ends_with("encoder.norm.weight")));
    assert!(names.last().unwrap().ends_with("decoder.norm.bias"));

    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
    let var_values: Vec<Tensor<f32>> = bound
        .trainable_vars()
        .iter()
        .map(|v| v.to_tensor())
        .collect();
    assert_eq!(var_values.len(), trainable_shapes.len());
    // shape 列に加え値も一致する（同 shape 同士の取り違えを検出する）。
    for (v, p) in var_values.iter().zip(model.trainable_parameters()) {
        assert_eq!(v.shape(), p.shape());
        assert_eq!(bits(v), bits(p));
    }
    let grads = tape.backward(&loss).unwrap();
    let grad_shapes: Vec<Vec<usize>> = bound
        .trainable_grads(&grads)
        .unwrap()
        .iter()
        .map(|g| g.contiguous().shape().to_vec())
        .collect();
    assert_eq!(grad_shapes, trainable_shapes);
}

#[test]
fn mixed_stack_consumes_vars_in_order() {
    // encoder／decoder 層／Transformer／linear の Vars 消費順が崩れていないことの回帰確認。
    let mut model = Sequential::new()
        .add_transformer_encoder(4, 2, 8, SEED1)
        .unwrap()
        .add_transformer_decoder_layer(4, 2, 8, SEED2)
        .unwrap()
        .add_transformer(small(1, 1), SEED1)
        .unwrap()
        .add_flatten(1, 2)
        .add_linear(12, 2, SEED1)
        .unwrap();
    let x = input();
    let target = tensor((0..4).map(|i| ((i % 3) as f32) * 0.1).collect(), &[2, 2]);
    let n_params = 16 + 26 + (16 + 2 + 26 + 2) + 2;
    assert_eq!(model.trainable_parameters().len(), n_params);

    let predicted = model.predict(&x).unwrap();
    {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let xv = tape.var(&x);
        let forwarded = bound.forward(&tape, &xv).unwrap().to_tensor();
        assert_eq!(bits(&predicted), bits(&forwarded));
    }

    let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let mut losses = Vec::new();
    for _ in 0..30 {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
            losses.push(loss.to_tensor().get(&[]).unwrap());
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            assert_eq!(grad_refs.len(), n_params);
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }
    assert!(*losses.last().unwrap() < losses[0] * 0.9, "{losses:?}");
}

// ---------------------------------------------------------------- 学習

#[test]
fn sgd_training_loop_reduces_loss_and_apply_parameters_reaches_predict() {
    let mut model = Sequential::new()
        .add_transformer(small(1, 1), SEED1)
        .unwrap();
    let x = input();
    let target = tensor(
        (0..2 * 3 * 4)
            .map(|i| ((i % 5) as f32) * 0.1 - 0.2)
            .collect(),
        &[2, 3, 4],
    );
    let before = model.predict(&x).unwrap();

    let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let mut losses = Vec::new();
    for _ in 0..30 {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
            losses.push(loss.to_tensor().get(&[]).unwrap());
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }
    assert!(
        *losses.last().unwrap() < losses[0] * 0.9,
        "loss should decrease: {losses:?}"
    );
    // 更新が predict へ反映される。
    assert_ne!(bits(&before), bits(&model.predict(&x).unwrap()));
}

#[test]
fn fit_runs_with_transformer_layer() {
    let mut model = Sequential::new()
        .add_transformer(small(1, 1), SEED1)
        .unwrap()
        .add_flatten(1, 2)
        .add_linear(12, 2, SEED2)
        .unwrap();
    let x = tensor(
        (0..4 * 3 * 4).map(|i| ((i as f32) * 0.07).sin()).collect(),
        &[4, 3, 4],
    );
    let y = tensor((0..8).map(|i| (i % 3) as f32 * 0.1).collect(), &[4, 2]);
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let w0 = dense_vec(model.trainable_parameters()[2]);
    let history = model
        .fit(&x, &y, FitConfig::new(2, 2))
        .expect("Transformer 層を含むモデルを fit できる");
    assert_eq!(history.loss.len(), 2);
    assert_ne!(dense_vec(model.trainable_parameters()[2]), w0);
}

// ---------------------------------------------------------------- 常駐・ONNX の fail-closed

#[test]
fn resident_paths_reject_unsupported() {
    let model = Sequential::new()
        .add_transformer(small(1, 1), SEED1)
        .unwrap();
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)));

    let linear = Sequential::new().add_linear(4, 4, SEED2).unwrap();
    let store = linear.init_device_param_store(&tape).unwrap();
    let err = model.predict_resident(&store, &input()).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::Unsupported(_))
    ));
}

#[test]
fn onnx_export_rejects_transformer_layer() {
    let model = Sequential::new()
        .add_transformer(small(1, 1), SEED1)
        .unwrap();
    let err = OnnxModel::from_sequential(&model).map(|_| ());
    assert!(
        matches!(err, Err(OnnxError::UnsupportedLayer { .. })),
        "{err:?}"
    );
}

// ---------------------------------------------------------------- 保存・復元

#[cfg(unix)]
mod model_io {
    use super::*;

    use std::path::Path;

    use fandhe_ai::compat::{ModelIoError, load_model, save_model};

    use crate::common::temp_dir::TempDirGuard;

    fn saved(label: &str) -> (TempDirGuard, std::path::PathBuf, String) {
        let mut model = Sequential::new()
            .add_transformer(small(1, 2).with_eps(1e-3), SEED1)
            .unwrap();
        model.eval();
        let guard = TempDirGuard::new(label);
        let dir = guard.path().join("m");
        save_model(&model, &dir).expect("保存できるはず");
        let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
        (guard, dir, text)
    }

    fn tampered(label: &str, from: &str, to: &str) -> ModelIoError {
        let (_g, dir, text) = saved(label);
        assert!(text.contains(from), "{label}: 置換対象 {from} がある");
        std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).unwrap();
        match load_model(&dir) {
            Ok(_) => panic!("{label}: 改竄された manifest は拒否されるはず"),
            Err(e) => e,
        }
    }

    #[test]
    fn round_trips_bit_identically_with_custom_eps() {
        let (_g, dir, text) = saved("rt");
        assert!(text.contains("\"kind\":\"transformer\""), "{text}");
        let mut model = Sequential::new()
            .add_transformer(small(1, 2).with_eps(1e-3), SEED1)
            .unwrap();
        model.eval();
        let loaded = load_model(&dir).expect("復元できるはず");
        let (a, b) = (model.state_dict(), loaded.state_dict());
        assert_eq!(a.len(), b.len());
        for (k, v) in &a {
            assert_eq!(bits(v), bits(&b[k]), "{k}");
        }
        let x = input();
        assert_eq!(
            bits(&model.predict(&x).unwrap()),
            bits(&loaded.predict(&x).unwrap())
        );
    }

    #[test]
    fn tampered_params_are_rejected() {
        let eps = "\"eps\":0.001";
        for (label, from, to) in [
            ("missing", ",\"eps\":0.001", ""),
            ("extra", eps, "\"eps\":0.001,\"x\":1"),
            (
                "type",
                "\"num_encoder_layers\":1",
                "\"num_encoder_layers\":\"1\"",
            ),
            (
                "zero-enc",
                "\"num_encoder_layers\":1",
                "\"num_encoder_layers\":0",
            ),
            (
                "zero-dec",
                "\"num_decoder_layers\":2",
                "\"num_decoder_layers\":0",
            ),
            ("eps-int", eps, "\"eps\":1"),
            ("eps-noncanonical", eps, "\"eps\":0.0010"),
        ] {
            let err = tampered(label, from, to);
            assert!(
                matches!(
                    err,
                    ModelIoError::Manifest { .. } | ModelIoError::Mismatch { .. }
                ),
                "{label}: {err}"
            );
        }
    }

    #[test]
    fn huge_layer_count_is_rejected_before_allocating_expected_keys() {
        // 層数だけを巨大にしても期待キー列を作る前に件数不一致で拒否される（巨大 Vec を作らない）。
        for (label, from, to) in [
            (
                "huge-enc",
                "\"num_encoder_layers\":1",
                "\"num_encoder_layers\":1000000000",
            ),
            (
                "huge-dec",
                "\"num_decoder_layers\":2",
                "\"num_decoder_layers\":18446744073709551615",
            ),
        ] {
            let err = tampered(label, from, to);
            assert!(
                matches!(
                    err,
                    ModelIoError::Mismatch { .. } | ModelIoError::Manifest { .. }
                ),
                "{label}: {err}"
            );
        }
    }

    #[test]
    fn non_finite_eps_cannot_be_saved() {
        // 非有限の eps は構築時に拒否されるか、保存時に `UnsupportedModel` で拒否される。
        match Sequential::new().add_transformer(small(1, 1).with_eps(f32::NAN), SEED1) {
            Err(_) => {}
            Ok(model) => {
                let guard = TempDirGuard::new("nan-eps");
                let dir = guard.path().join("m");
                let err = save_model(&model, &dir).unwrap_err();
                assert!(
                    matches!(err, ModelIoError::UnsupportedModel { .. }),
                    "{err}"
                );
                assert!(!Path::new(&dir).exists());
            }
        }
    }
}
