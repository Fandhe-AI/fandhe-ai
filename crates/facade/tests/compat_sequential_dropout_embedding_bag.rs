//! `compat::Sequential::add_dropout2d`／`add_alpha_dropout`／`add_embedding_bag` と
//! `Var::dropout2d`／`alpha_dropout`／`embedding_bag`（イシュー #2528・親 #2520・ルート #2499 の
//! 一括承認。`docs/autodiff-dropout-embedding-bag-decision.md` §6・§8）の facade 公開面を
//! CPU で検証する統合テスト。
//!
//! 3 層は内部実装（`nn::Dropout2d`／`AlphaDropout`／`EmbeddingBag`。#2161）の薄い委譲である。
//! facade 公開 API だけで次を固定する。
//!
//! - 構築検査（`p` の範囲・NaN、`num_embeddings == 0`、`padding_idx` の範囲外）と forward 時の
//!   検査（Dropout2d の rank・非整数 id）
//! - 数値: `Var` 直接呼び出しと層経路の bit 一致（train はグローバル RNG を同じ値へ再シード）、
//!   eval での恒等
//! - 学習経路（`bind`／`trainable_parameters`／`trainable_vars`／`trainable_grads`／
//!   `apply_parameters`／`fit`）
//! - 常駐経路: dropout 2 種は通過・EmbeddingBag は `Unsupported`（fail-closed）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄の型付き拒否・training 食い違いの拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_dropout_embedding_bag_backend_parity.rs`（`#[ignore]`）。

#![cfg(unix)]

use std::sync::{Mutex, MutexGuard};

use fandhe_ai::compat::{
    FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, EmbeddingBagMode, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

/// グローバル RNG を触るテストの直列化ロック。
///
/// `manual_seed` を呼ぶテストだけでなく、グローバル RNG を暗黙に引くテスト
/// （学習モードの dropout 系 `predict`／`forward`、`shuffle(true)` の `fit`、`rand`／`randn` 系）も
/// 取ること。取らないと並走テストが `manual_seed` と乱数消費の間で系列を進め、bit 一致が揺れる（#2968）。
fn rng_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn ramp(shape: &[usize], scale: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(
        (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * scale + 1.0)
            .collect(),
        shape,
    )
}

/// 各要素を id（f32）として詰めた `[b, l]` 入力（`i % vocab`）。
fn id_input(b: usize, l: usize, vocab: usize) -> Tensor<f32> {
    tensor((0..b * l).map(|i| (i % vocab) as f32).collect(), &[b, l])
}

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn invalid_p_is_rejected_at_construction() {
    for p in [-0.1_f32, 1.5, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            Sequential::new().add_dropout2d(p),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            Sequential::new().add_alpha_dropout(p),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    // 境界値は受理される。
    assert!(Sequential::new().add_dropout2d(0.0).is_ok());
    assert!(Sequential::new().add_dropout2d(1.0).is_ok());
    assert!(Sequential::new().add_alpha_dropout(0.0).is_ok());
    assert!(Sequential::new().add_alpha_dropout(1.0).is_ok());
}

#[test]
fn invalid_embedding_bag_arguments_are_rejected_at_construction() {
    assert!(matches!(
        Sequential::new().add_embedding_bag(0, 4, EmbeddingBagMode::Sum, None, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        Sequential::new().add_embedding_bag(3, 4, EmbeddingBagMode::Sum, Some(3), 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn forward_time_checks_reject_bad_rank_and_ids() {
    // Dropout2d は rank 4 限定（eval でも拒否される）。
    let mut d2 = Sequential::new().add_dropout2d(0.5).unwrap();
    assert!(d2.predict(&ramp(&[2, 3, 4], 1.0)).is_err());
    assert!(d2.predict(&ramp(&[2, 3, 4, 5, 6], 1.0)).is_err());
    d2.eval();
    assert!(d2.predict(&ramp(&[2, 3, 4], 1.0)).is_err());

    // EmbeddingBag: 非整数・負・範囲外・非有限の id は拒否。
    let bag = Sequential::new()
        .add_embedding_bag(4, 2, EmbeddingBagMode::Mean, None, 3)
        .unwrap();
    for bad in [1.5_f32, -1.0, 4.0, f32::NAN, f32::INFINITY] {
        let x = tensor(vec![0.0, bad, 1.0, 2.0], &[2, 2]);
        assert!(bag.predict(&x).is_err(), "id={bad}");
    }
    // ids の rank が 2 でない入力は拒否。
    assert!(bag.predict(&tensor(vec![0.0, 1.0], &[2])).is_err());
}

// ---------------------------------------------------------------------
// 数値
// ---------------------------------------------------------------------

#[test]
fn eval_mode_makes_dropout_layers_identity() {
    // 末尾の学習モード `predict` が `feature_dropout_mask` 経由でグローバル RNG を N×C 回引くため直列化する。
    let _guard = rng_lock();
    let x = ramp(&[2, 3, 4, 5], 1.0);
    let mut model = Sequential::new()
        .add_dropout2d(0.5)
        .unwrap()
        .add_alpha_dropout(0.5)
        .unwrap();
    model.eval();
    assert_eq!(bits(&model.predict(&x).unwrap()), bits(&x));
    // train へ戻すと p=1.0 の層は全ゼロ化できる（set_training が伝わっていること）。
    let mut all_drop = Sequential::new().add_dropout2d(1.0).unwrap();
    all_drop.eval();
    assert_eq!(bits(&all_drop.predict(&x).unwrap()), bits(&x));
    all_drop.train();
    let y = all_drop.predict(&x).unwrap();
    assert!(y.contiguous().as_slice().unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn train_mode_layer_path_matches_direct_var_call_bit_exact() {
    let _guard = rng_lock();
    let x = ramp(&[2, 3, 4, 5], 1.0);
    for p in [0.25_f32, 0.5, 1.0] {
        // Dropout2d
        let model = Sequential::new().add_dropout2d(p).unwrap();
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        fandhe_ai::manual_seed(123);
        let direct = xv.dropout2d(p, true).unwrap().to_tensor();
        fandhe_ai::manual_seed(123);
        let via_predict = model.predict(&x).unwrap();
        fandhe_ai::manual_seed(123);
        let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
        assert_eq!(bits(&via_predict), bits(&direct), "dropout2d p={p}");
        assert_eq!(bits(&via_forward), bits(&direct), "dropout2d p={p}");

        // AlphaDropout（rank 任意）
        let model = Sequential::new().add_alpha_dropout(p).unwrap();
        fandhe_ai::manual_seed(321);
        let direct = xv.alpha_dropout(p, true).unwrap().to_tensor();
        fandhe_ai::manual_seed(321);
        let via_predict = model.predict(&x).unwrap();
        fandhe_ai::manual_seed(321);
        let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
        assert_eq!(bits(&via_predict), bits(&direct), "alpha p={p}");
        assert_eq!(bits(&via_forward), bits(&direct), "alpha p={p}");
    }
}

#[test]
fn embedding_bag_layer_path_matches_direct_var_call_bit_exact() {
    for (mode, padding_idx) in [
        (EmbeddingBagMode::Sum, None),
        (EmbeddingBagMode::Mean, None),
        (EmbeddingBagMode::Max, None),
        (EmbeddingBagMode::Sum, Some(0)),
        (EmbeddingBagMode::Mean, Some(2)),
    ] {
        let model = Sequential::new()
            .add_embedding_bag(6, 4, mode, padding_idx, 17)
            .unwrap();
        let x = id_input(3, 4, 6);
        let weight = model.trainable_parameters()[0].clone();
        let ids = Tensor::<i32>::new(
            x.contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| *v as i32)
                .collect(),
            &[3, 4],
        )
        .unwrap();
        let tape = fandhe_ai::tape();
        let wv = tape.var(&weight);
        let direct = wv.embedding_bag(&ids, mode, padding_idx).unwrap();
        assert_eq!(direct.to_tensor().shape(), &[3, 4]);
        let via_predict = model.predict(&x).unwrap();
        let xv = tape.var(&x);
        let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
        assert_eq!(bits(&via_predict), bits(&direct.to_tensor()), "{mode:?}");
        assert_eq!(bits(&via_forward), bits(&direct.to_tensor()), "{mode:?}");
    }
}

// ---------------------------------------------------------------------
// 学習経路
// ---------------------------------------------------------------------

/// `EmbeddingBag → Linear` の最小学習モデル（入力 `[B, L]` の id・出力 `[B, 2]`）。
fn bag_model() -> Sequential {
    Sequential::new()
        .add_embedding_bag(8, 4, EmbeddingBagMode::Mean, None, 7)
        .unwrap()
        .add_linear(4, 2, 9)
        .unwrap()
}

#[test]
fn embedding_bag_is_tracked_as_one_trainable_parameter_in_layer_order() {
    let model = Sequential::new()
        .add_linear(3, 5, 1)
        .unwrap()
        .add_embedding_bag(8, 4, EmbeddingBagMode::Sum, None, 2)
        .unwrap()
        .add_dropout2d(0.1)
        .unwrap()
        .add_linear(4, 2, 3)
        .unwrap();
    // Linear(weight, bias) + EmbeddingBag(weight) + Linear(weight, bias)。
    let params = model.trainable_parameters();
    assert_eq!(params.len(), 5);
    assert_eq!(params[2].shape(), &[8, 4]);

    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let vars = bound.trainable_vars();
    assert_eq!(vars.len(), 5);
    assert_eq!(vars[2].to_tensor().shape(), &[8, 4]);
}

#[test]
fn bag_model_grads_flow_and_apply_parameters_updates_weight() {
    let mut model = bag_model();
    let x = id_input(3, 4, 8);
    let target = ramp(&[3, 2], 0.5);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let g = bound.trainable_grads(&grads).unwrap();
    assert_eq!(g.len(), 3);
    assert_eq!(g[0].shape(), &[8, 4]);
    assert!(
        g[0].contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .any(|v| *v != 0.0),
        "EmbeddingBag weight に勾配が届くはず"
    );
    drop(bound);

    let before = bits(model.trainable_parameters()[0]);
    let updated: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| {
            tensor(
                p.contiguous()
                    .as_slice()
                    .unwrap()
                    .iter()
                    .map(|v| v + 0.25)
                    .collect(),
                p.shape(),
            )
        })
        .collect();
    model.apply_parameters(updated).unwrap();
    assert_ne!(before, bits(model.trainable_parameters()[0]));
}

#[test]
fn fit_updates_embedding_bag_weight_with_finite_loss() {
    let mut model = bag_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    let x = id_input(6, 3, 8);
    let y = ramp(&[6, 2], 0.5);
    let before = bits(model.trainable_parameters()[0]);
    let history = model.fit(&x, &y, FitConfig::new(6, 2)).unwrap();
    assert_eq!(history.loss.len(), 6);
    assert!(
        history.loss.iter().all(|l| l.is_finite()),
        "{:?}",
        history.loss
    );
    assert_ne!(before, bits(model.trainable_parameters()[0]));
}

// ---------------------------------------------------------------------
// 常駐経路
// ---------------------------------------------------------------------

#[test]
fn resident_path_accepts_dropout_layers_and_matches_predict_in_eval() {
    let mut model = Sequential::new()
        .add_dropout2d(0.5)
        .unwrap()
        .add_alpha_dropout(0.5)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(12, 2, 5)
        .unwrap();
    model.eval();
    let x = ramp(&[2, 3, 2, 2], 1.0);
    let tape = fandhe_ai::tape();
    let store = model
        .init_device_param_store(&tape)
        .expect("無状態の dropout 層は常駐経路を通過するはず");
    let resident = model.predict_resident(&store, &x).unwrap();
    let host = model.predict(&x).unwrap();
    assert_eq!(bits(&resident), bits(&host));
}

#[test]
fn resident_path_rejects_embedding_bag_fail_closed() {
    let model = bag_model();
    let x = id_input(2, 3, 8);

    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");

    // predict_resident／forward_resident も、別モデル由来の store で到達しても拒否される。
    let store_tape = fandhe_ai::tape();
    let mut store = Sequential::new()
        .add_linear(2, 2, 1)
        .unwrap()
        .init_device_param_store(&store_tape)
        .unwrap();
    drop(store_tape);
    let err = model.predict_resident(&store, &x).unwrap_err();
    assert!(
        matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
        "{err:?}"
    );
    let tape2 = fandhe_ai::tape();
    let xv = tape2.var(&x);
    let err = model
        .forward_resident(&tape2, &xv, &mut store)
        .expect_err("forward_resident も拒否されるはず");
    assert!(
        matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------
// save_model／load_model
// ---------------------------------------------------------------------

#[test]
fn save_and_load_round_trip_is_bit_identical() {
    let mut model = Sequential::new()
        .add_dropout2d(0.25)
        .unwrap()
        .add_alpha_dropout(0.75)
        .unwrap();
    model.eval();
    let x = ramp(&[2, 3, 4, 4], 1.0);
    let guard = TempDirGuard::new("dropout_variants_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("dropout2d"), "{manifest}");
    assert!(manifest.contains("alpha_dropout"), "{manifest}");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );

    for (i, (mode, padding_idx)) in [
        (EmbeddingBagMode::Sum, None),
        (EmbeddingBagMode::Mean, Some(1)),
        (EmbeddingBagMode::Max, None),
    ]
    .into_iter()
    .enumerate()
    {
        let m = Sequential::new()
            .add_embedding_bag(6, 3, mode, padding_idx, 40 + i as u64)
            .unwrap()
            .add_linear(3, 2, 5)
            .unwrap();
        let ids = id_input(3, 4, 6);
        let guard = TempDirGuard::new(&format!("embedding_bag_round_trip_{i}"));
        let dir = guard.path().join("m");
        save_model(&m, &dir).expect("保存できるはず");
        let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
        assert!(manifest.contains("embedding_bag"), "{manifest}");
        let loaded = load_model(&dir).expect("復元できるはず");
        assert_eq!(
            bits(&m.predict(&ids).unwrap()),
            bits(&loaded.predict(&ids).unwrap()),
            "{mode:?}"
        );
    }
}

#[test]
fn save_rejects_layer_training_mismatch() {
    for build in [
        |m: Sequential| m.add_dropout2d(0.5).unwrap(),
        |m: Sequential| m.add_alpha_dropout(0.5).unwrap(),
    ] {
        let mut m = Sequential::new();
        m.eval();
        // eval の後に追加した層は training=true のためモデルのモードと食い違う。
        let m = build(m);
        let guard = TempDirGuard::new("dropout_variants_training_mismatch");
        let dir = guard.path().join("m");
        let err = save_model(&m, &dir).unwrap_err();
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{err:?}"
        );
        assert!(!dir.exists(), "拒否時は dir に何も作らない");
    }
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = Sequential::new()
        .add_embedding_bag(6, 3, EmbeddingBagMode::Max, Some(1), 40)
        .unwrap();
    let guard = TempDirGuard::new("embedding_bag_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    for (from, to) in [
        ("\"max\"", "\"min\""),
        ("\"max\"", "1"),
        ("\"mode\"", "\"modex\""),
        ("\"padding_idx\"", "\"padding_idxx\""),
        ("\"num_embeddings\":6", "\"num_embeddings\":\"6\""),
        ("\"num_embeddings\":6", "\"num_embeddings\":6,\"extra\":1"),
    ] {
        assert!(original.contains(from), "{from} not in {original}");
        std::fs::write(&path, original.replacen(from, to, 1)).unwrap();
        assert!(load_model(&dir).is_err(), "{from} → {to} は拒否されるはず");
    }

    let model = Sequential::new().add_dropout2d(0.5).unwrap();
    let guard = TempDirGuard::new("dropout2d_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    for (from, to) in [
        ("\"p\":0.5", "\"p\":\"0.5\""),
        ("\"p\":0.5", "\"p\":0.5,\"extra\":1"),
        ("\"p\":0.5", "\"q\":0.5"),
        ("\"p\":0.5", "\"p\":2.0"),
    ] {
        assert!(original.contains(from), "{from} not in {original}");
        std::fs::write(&path, original.replacen(from, to, 1)).unwrap();
        assert!(load_model(&dir).is_err(), "{from} → {to} は拒否されるはず");
    }
}

// ---------------------------------------------------------------------
// ONNX export（対象外。現行挙動の固定）
// ---------------------------------------------------------------------

#[test]
fn onnx_export_rejects_dropout_variants_and_embedding_bag() {
    for model in [
        Sequential::new().add_dropout2d(0.5).unwrap(),
        Sequential::new().add_alpha_dropout(0.5).unwrap(),
        Sequential::new()
            .add_embedding_bag(4, 2, EmbeddingBagMode::Sum, None, 1)
            .unwrap(),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
