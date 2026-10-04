//! `compat::Sequential::add_pixel_shuffle`／`add_pixel_unshuffle` と `Var::pixel_shuffle`／
//! `pixel_unshuffle`（イシュー #2526・親 #2520・ルート #2499 の一括承認。
//! `docs/autodiff-pixel-shuffle-decision.md` §6）の facade 公開面を CPU で検証する統合テスト。
//!
//! 2 層は学習可能パラメータを持たない無状態層で、内部実装（`nn::PixelShuffle`／
//! `nn::PixelUnshuffle`。#2162）の薄い委譲である。facade 公開 API だけで次を固定する。
//!
//! - 構築検査（倍率 `0` の即時拒否）と forward 時の遅延検査（rank・整除性）
//! - 数値: 手計算例・往復 bit 一致・`Var` 経路と層経路の bit 一致
//! - 学習経路（`bind`／`trainable_parameters`／`trainable_grads`／`apply_parameters`／`fit`）
//! - 常駐経路（対応扱い。既存の常駐非対応層との混在は従来どおり `Unsupported`）
//! - `save_model`／`load_model` の往復 bit 一致と manifest 改竄の型付き拒否
//! - ONNX export は `OnnxError::UnsupportedLayer`（現行挙動の固定）
//!
//! 実機 parity は `compat_sequential_pixel_shuffle_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

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
        (0..n).map(|i| ((i as f32) * 0.37).sin() * scale).collect(),
        shape,
    )
}

/// `Conv2d → PixelShuffle(2) → PixelUnshuffle(2) → Flatten → Linear` の混在モデル
/// （入力 `[2, 1, 2, 2]`・出力 `[2, 3]`。Conv2d は 4 チャネルへ広げる）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 11)
        .unwrap()
        .add_pixel_shuffle(2)
        .unwrap()
        .add_pixel_unshuffle(2)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(16, 3, 12)
        .unwrap()
}

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn zero_factor_is_rejected_at_construction() {
    assert!(matches!(
        Sequential::new().add_pixel_shuffle(0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        Sequential::new().add_pixel_unshuffle(0),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn rank_and_divisibility_are_checked_at_forward() {
    let shuffle = Sequential::new().add_pixel_shuffle(2).unwrap();
    // rank 2。
    assert!(shuffle.predict(&ramp(&[4, 4], 1.0)).is_err());
    // チャネル 3 は 2*2 で割り切れない。
    assert!(shuffle.predict(&ramp(&[1, 3, 2, 2], 1.0)).is_err());
    let unshuffle = Sequential::new().add_pixel_unshuffle(2).unwrap();
    // 空間軸 3 は 2 で割り切れない。
    assert!(unshuffle.predict(&ramp(&[1, 1, 3, 4], 1.0)).is_err());
    assert!(unshuffle.predict(&ramp(&[4, 4], 1.0)).is_err());
}

// ---------------------------------------------------------------------
// 数値
// ---------------------------------------------------------------------

#[test]
fn pixel_shuffle_r2_hand_computed_example() {
    let model = Sequential::new().add_pixel_shuffle(2).unwrap();
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], &[1, 4, 1, 1]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    assert_eq!(
        bits(&y),
        bits(&tensor(vec![0.0, 1.0, 2.0, 3.0], &[1, 1, 2, 2]))
    );

    // 入力 [8, 1, 2]（rank 3・C=8）→ 出力 [2, 2, 4]。
    // out[c, h*2+i, w*2+j] = in[c*4+i*2+j, h, w]。in の値は index そのもの（c_in*2 + w）。
    let x = tensor((0..16).map(|v| v as f32).collect(), &[8, 1, 2]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[2, 2, 4]);
    let y = y.contiguous();
    let v = y.as_slice().unwrap();
    // c=0, h=0, i=1, w=1, j=0 → c_in = 2, w = 1 → 値 2*2+1 = 5、出力位置 [0, 1, 2] = 1*4+2。
    assert_eq!(v[4 + 2], 5.0);
    // c=1, i=1, j=1, w=0, h=0 → c_in = 4+3 = 7 → 値 14、出力位置 [1, 1, 1] = 8 + 4 + 1。
    assert_eq!(v[8 + 4 + 1], 14.0);
}

#[test]
fn shuffle_then_unshuffle_round_trips_bit_exact() {
    let model = Sequential::new()
        .add_pixel_shuffle(2)
        .unwrap()
        .add_pixel_unshuffle(2)
        .unwrap();
    let x = ramp(&[2, 8, 3, 3], 1.0);
    assert_eq!(bits(&model.predict(&x).unwrap()), bits(&x));
}

#[test]
fn var_methods_match_layer_path_bit_exact() {
    let x = ramp(&[2, 8, 3, 3], 1.0);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let shuffled = xv.pixel_shuffle(2).unwrap();
    assert_eq!(shuffled.to_tensor().shape(), &[2, 2, 6, 6]);
    let model = Sequential::new().add_pixel_shuffle(2).unwrap();
    assert_eq!(
        bits(&shuffled.to_tensor()),
        bits(&model.predict(&x).unwrap())
    );
    let restored = shuffled.pixel_unshuffle(2).unwrap();
    assert_eq!(bits(&restored.to_tensor()), bits(&x));
}

#[test]
fn var_methods_reject_zero_and_bad_shapes() {
    let tape = fandhe_ai::tape();
    let xv = tape.var(&ramp(&[1, 4, 2, 2], 1.0));
    assert!(matches!(
        xv.pixel_shuffle(0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        xv.pixel_unshuffle(0),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(xv.pixel_shuffle(3).is_err());
    assert!(xv.pixel_unshuffle(3).is_err());
}

#[test]
fn mixed_model_predict_matches_tape_forward() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 2, 2], 1.0);
    let via_predict = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 3]);
    assert_eq!(bits(&via_predict), bits(&via_forward));
}

// ---------------------------------------------------------------------
// 学習経路
// ---------------------------------------------------------------------

#[test]
fn stateless_layers_do_not_add_trainable_parameters_and_grads_match() {
    let model = mixed_model();
    // Conv2d(weight, bias) + Linear(weight, bias) の 4 件だけ。
    let param_count = model.trainable_parameters().len();
    assert_eq!(param_count, 4);

    let x = ramp(&[2, 1, 2, 2], 1.0);
    let target = ramp(&[2, 3], 0.5);
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    assert_eq!(bound.trainable_vars().len(), param_count);
    let grads = tape.backward(&loss).unwrap();
    let grad_refs = bound.trainable_grads(&grads).unwrap();
    assert_eq!(grad_refs.len(), param_count);
}

#[test]
fn apply_parameters_succeeds_with_stateless_layers() {
    let mut model = mixed_model();
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
}

#[test]
fn fit_updates_parameters_through_stateless_layers() {
    let mut model = mixed_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let x = ramp(&[4, 1, 2, 2], 1.0);
    let y = ramp(&[4, 3], 0.5);
    let before: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    let history = model.fit(&x, &y, FitConfig::new(3, 2)).unwrap();
    assert_eq!(history.loss.len(), 3);
    assert!(
        history.loss.iter().all(|l| l.is_finite()),
        "{:?}",
        history.loss
    );
    let after: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| bits(p))
        .collect();
    assert_ne!(before, after, "fit でパラメータが更新されるはず");
}

// ---------------------------------------------------------------------
// 常駐経路
// ---------------------------------------------------------------------

/// 無状態 2 層 + Linear だけの常駐対応モデル（入力 `[2, 4, 2, 2]`）。
fn resident_ok_model() -> Sequential {
    Sequential::new()
        .add_pixel_shuffle(2)
        .unwrap()
        .add_pixel_unshuffle(2)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(16, 4, 21)
        .unwrap()
        .add_relu()
        .add_linear(4, 2, 22)
        .unwrap()
}

#[test]
fn stateless_layers_are_supported_on_resident_path() {
    let model = resident_ok_model();
    let x = ramp(&[2, 4, 2, 2], 1.0);
    let init_tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    let via_resident = model.predict_resident(&store, &x).unwrap();
    let via_predict = model.predict(&x).unwrap();
    assert_eq!(bits(&via_resident), bits(&via_predict));
}

#[test]
fn resident_path_still_rejects_mixed_model_with_conv() {
    // 常駐非対応の Conv2d を混ぜたモデルは従来どおり拒否される（新層で判定が緩んでいない）。
    let model = mixed_model();
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");

    let store_tape = fandhe_ai::tape();
    let store = resident_ok_model()
        .init_device_param_store(&store_tape)
        .unwrap();
    drop(store_tape);
    let x = ramp(&[2, 1, 2, 2], 1.0);
    let err = model.predict_resident(&store, &x).unwrap_err();
    assert!(
        matches!(err, AutodiffError::Backend(BackendError::Unsupported(_))),
        "{err:?}"
    );

    let tape2 = fandhe_ai::tape();
    let xv = tape2.var(&x);
    let mut store2 = store;
    let err = model
        .forward_resident(&tape2, &xv, &mut store2)
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
    let model = mixed_model();
    let x = ramp(&[2, 1, 2, 2], 1.0);
    let guard = TempDirGuard::new("pixel_shuffle_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("pixel_shuffle"), "{manifest}");
    assert!(manifest.contains("pixel_unshuffle"), "{manifest}");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        bits(&model.predict(&x).unwrap()),
        bits(&loaded.predict(&x).unwrap())
    );
}

#[test]
fn tampered_manifest_is_rejected_without_panic() {
    let model = Sequential::new().add_pixel_shuffle(2).unwrap();
    let guard = TempDirGuard::new("pixel_shuffle_tamper");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();
    // 必須キーの欠落（未知キー＋欠落）。
    let (from, to) = ("\"upscale_factor\"", "\"upscale_factorx\"");
    assert!(original.contains(from), "{original}");
    std::fs::write(&path, original.replace(from, to)).unwrap();
    assert!(load_model(&dir).is_err(), "{from} → {to} は拒否されるはず");
}

// ---------------------------------------------------------------------
// ONNX export（対象外。現行挙動の固定）
// ---------------------------------------------------------------------

#[test]
fn onnx_export_rejects_pixel_shuffle_layers() {
    for model in [
        Sequential::new().add_pixel_shuffle(2).unwrap(),
        Sequential::new().add_pixel_unshuffle(2).unwrap(),
    ] {
        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}
