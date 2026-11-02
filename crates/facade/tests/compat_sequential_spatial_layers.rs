//! `compat::Sequential::add_upsample`／`add_zero_pad2d`／`add_identity`（イシュー #2522・
//! ルート #2499 の 2026-10-04 一括承認。`docs/autodiff-spatial-layers-decision.md` §6）の
//! facade 公開面を CPU で検証する統合テスト。
//!
//! 3 層は学習可能パラメータを持たない無状態層で、内部実装（`nn::Upsample`／`nn::ZeroPad2d`／
//! `nn::Identity`。#2159）の薄い委譲である。本ファイルは次を facade 公開 API だけで固定する。
//!
//! - 構築検査（`add_upsample` の空 `size` 拒否）と forward 時の遅延検査（`Shape`）
//! - 数値: `ZeroPad2d` の 0 埋め・`Identity` の恒等・`Upsample(Nearest)` の添字式
//! - `predict`（host 経路）と `forward`（外部 `Tape` 経路）の bit 完全一致（混在モデル）
//! - 学習経路（`bind`／`trainable_parameters`／`trainable_grads`／`apply_parameters`／`fit`）
//! - 常駐経路（対応扱い。既存の常駐非対応層との混在は従来どおり `Unsupported`）
//! - `save_model`／`load_model` の往復 bit 一致
//!
//! 実機 parity は `compat_sequential_spatial_layers_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::compat::FitConfig;
use fandhe_ai::compat::{Loss, Optimizer, Sequential, load_model, save_model};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, InterpolateMode, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<u32> {
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

/// `Conv2d → ZeroPad2d → Upsample(Bilinear) → Identity → Flatten → Linear` の混在モデル
/// （入力 `[2, 1, 4, 4]`・出力 `[2, 3]`）。
fn mixed_model() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, 11)
        .unwrap()
        .add_zero_pad2d([1, 1, 1, 1])
        .add_upsample(
            vec![4, 4],
            InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .unwrap()
        .add_identity()
        .add_flatten(1, 3)
        .add_linear(32, 3, 12)
        .unwrap()
}

// ---------------------------------------------------------------------
// 構築検査・遅延検査
// ---------------------------------------------------------------------

#[test]
fn add_upsample_rejects_empty_size() {
    match Sequential::new().add_upsample(vec![], InterpolateMode::Nearest) {
        Err(AutodiffError::InvalidArgument(_)) => {}
        Err(other) => panic!("InvalidArgument のはず: {other:?}"),
        Ok(_) => panic!("空 size は構築時に拒否されるはず"),
    }
}

#[test]
fn upsample_mode_rank_mismatch_is_deferred_to_forward() {
    // Bilinear は空間 2 軸限定。size 1 軸は構築できるが forward で Shape になる（遅延検査）。
    let model = Sequential::new()
        .add_upsample(
            vec![4],
            InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .expect("構築時は mode と軸数の整合を見ない");
    let err = model.predict(&ramp(&[1, 1, 2, 2], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)), "{err:?}");
}

#[test]
fn zero_pad2d_rejects_rank1_input_at_forward() {
    let model = Sequential::new().add_zero_pad2d([1, 1, 1, 1]);
    let err = model.predict(&ramp(&[4], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)), "{err:?}");
}

// ---------------------------------------------------------------------
// 数値
// ---------------------------------------------------------------------

#[test]
fn zero_pad2d_pads_with_zeros_left_right_top_bottom() {
    // [left, right, top, bottom] = [1, 2, 0, 1]。入力 [1,1,1,2] → 出力 [1,1,2,5]。
    let model = Sequential::new().add_zero_pad2d([1, 2, 0, 1]);
    let x = tensor(vec![1.0, 2.0], &[1, 1, 1, 2]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 5]);
    let expected = [0.0_f32, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(dense(&y), expected_bits);
}

#[test]
fn identity_returns_input_bit_exact() {
    let model = Sequential::new().add_identity();
    let x = ramp(&[2, 3, 4], 2.5);
    assert_eq!(dense(&model.predict(&x).unwrap()), dense(&x));
}

#[test]
fn upsample_nearest_doubles_each_element() {
    let model = Sequential::new()
        .add_upsample(vec![4, 4], InterpolateMode::Nearest)
        .unwrap();
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 4]);
    let expected = [
        1.0_f32, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(dense(&y), expected_bits);
}

// ---------------------------------------------------------------------
// predict（host）と forward（Tape）の一致・学習経路
// ---------------------------------------------------------------------

#[test]
fn mixed_model_predict_matches_tape_forward() {
    let model = mixed_model();
    let x = ramp(&[2, 1, 4, 4], 1.0);
    let via_predict = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = model.forward(&tape, &xv).unwrap().to_tensor();
    assert_eq!(via_predict.shape(), &[2, 3]);
    assert_eq!(dense(&via_predict), dense(&via_forward));
}

#[test]
fn stateless_layers_do_not_add_trainable_parameters_and_grads_match() {
    let model = mixed_model();
    // Conv2d(weight, bias) + Linear(weight, bias) の 4 件だけ。
    let param_count = model.trainable_parameters().len();
    assert_eq!(param_count, 4);

    let x = ramp(&[2, 1, 4, 4], 1.0);
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
    let x = ramp(&[4, 1, 4, 4], 1.0);
    let y = ramp(&[4, 3], 0.5);
    let before: Vec<Vec<u32>> = model
        .trainable_parameters()
        .iter()
        .map(|p| dense(p))
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
        .map(|p| dense(p))
        .collect();
    assert_ne!(before, after, "fit でパラメータが更新されるはず");
}

// ---------------------------------------------------------------------
// 常駐経路
// ---------------------------------------------------------------------

/// 無状態 3 層 + Linear だけの常駐対応モデル（入力 `[2, 1, 3, 3]`）。
fn resident_ok_model() -> Sequential {
    Sequential::new()
        .add_zero_pad2d([1, 1, 1, 1])
        .add_upsample(vec![4, 4], InterpolateMode::Nearest)
        .unwrap()
        .add_identity()
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
    let x = ramp(&[2, 1, 3, 3], 1.0);
    let init_tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    let via_resident = model.predict_resident(&store, &x).unwrap();
    let via_predict = model.predict(&x).unwrap();
    assert_eq!(dense(&via_resident), dense(&via_predict));
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
    let x = ramp(&[2, 1, 4, 4], 1.0);
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
    let x = ramp(&[2, 1, 4, 4], 1.0);
    let guard = TempDirGuard::new("spatial_layers_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        dense(&model.predict(&x).unwrap()),
        dense(&loaded.predict(&x).unwrap())
    );
}

#[test]
fn save_and_load_round_trip_with_nearest_upsample_and_identity() {
    let model = resident_ok_model();
    let x = ramp(&[2, 1, 3, 3], 1.0);
    let guard = TempDirGuard::new("spatial_layers_round_trip_nearest");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        dense(&model.predict(&x).unwrap()),
        dense(&loaded.predict(&x).unwrap())
    );
}

// ---------------------------------------------------------------------
// イシュー #3814: add_upsample_scale_factor（案 A の別名メソッド）
// ---------------------------------------------------------------------

#[test]
fn add_upsample_scale_factor_rejects_invalid_scale() {
    for bad in [
        vec![],
        vec![0.0],
        vec![2.0, -1.0],
        vec![-0.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![f64::NEG_INFINITY, 2.0],
    ] {
        match Sequential::new().add_upsample_scale_factor(bad.clone(), InterpolateMode::Nearest) {
            Err(AutodiffError::InvalidArgument(_)) => {}
            Err(other) => panic!("{bad:?}: InvalidArgument のはず: {other:?}"),
            Ok(_) => panic!("{bad:?}: 構築時に拒否されるはず"),
        }
    }
}

#[test]
fn add_upsample_scale_factor_floors_output_size() {
    let up = Sequential::new()
        .add_upsample_scale_factor(vec![1.5, 1.5], InterpolateMode::Nearest)
        .unwrap();
    assert_eq!(
        up.predict(&ramp(&[1, 1, 3, 3], 1.0)).unwrap().shape(),
        &[1, 1, 4, 4]
    );
    let down = Sequential::new()
        .add_upsample_scale_factor(vec![0.5, 0.5], InterpolateMode::Nearest)
        .unwrap();
    assert_eq!(
        down.predict(&ramp(&[1, 1, 4, 4], 1.0)).unwrap().shape(),
        &[1, 1, 2, 2]
    );
}

#[test]
fn add_upsample_scale_factor_rejects_zero_output_and_rank_mismatch() {
    // 出力サイズが 0 になる倍率は forward 時に InvalidArgument。
    let tiny = Sequential::new()
        .add_upsample_scale_factor(vec![0.1, 0.1], InterpolateMode::Nearest)
        .unwrap();
    let err = tiny.predict(&ramp(&[1, 1, 3, 3], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
    // rank 不足の入力は Shape。
    let ok = Sequential::new()
        .add_upsample_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest)
        .unwrap();
    let err = ok.predict(&ramp(&[3], 1.0)).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)), "{err:?}");
    // 倍率の個数と mode の空間軸数の不整合は forward 時のエラー（遅延検査）。
    let mismatch = Sequential::new()
        .add_upsample_scale_factor(
            vec![2.0],
            InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .expect("構築時は mode と軸数の整合を見ない");
    assert!(mismatch.predict(&ramp(&[1, 1, 2, 2], 1.0)).is_err());
}

#[test]
fn add_upsample_scale_factor_matches_size_path_bit_exact() {
    let x = ramp(&[2, 1, 3, 5], 1.0);
    for (scale, size, mode) in [
        (vec![2.0, 2.0], vec![6, 10], InterpolateMode::Nearest),
        (vec![1.5, 1.7], vec![4, 8], InterpolateMode::Nearest),
        (
            vec![2.0, 1.5],
            vec![6, 7],
            InterpolateMode::Bilinear {
                align_corners: false,
            },
        ),
    ] {
        let by_scale = Sequential::new()
            .add_upsample_scale_factor(scale.clone(), mode)
            .unwrap();
        let by_size = Sequential::new().add_upsample(size, mode).unwrap();
        assert_eq!(
            dense(&by_scale.predict(&x).unwrap()),
            dense(&by_size.predict(&x).unwrap()),
            "{scale:?} {mode:?}"
        );
    }
}

#[test]
fn add_upsample_scale_factor_passes_train_and_resident_paths() {
    // 学習経路: fit でパラメータが更新される（前後に Linear を挟む）。
    let mut m = Sequential::new()
        .add_conv2d(1, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, 31)
        .unwrap()
        .add_upsample_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(128, 3, 32)
        .unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let xs = ramp(&[4, 1, 4, 4], 1.0);
    let ys = ramp(&[4, 3], 0.5);
    let b: Vec<Vec<u32>> = m.trainable_parameters().iter().map(|p| dense(p)).collect();
    let h = m.fit(&xs, &ys, FitConfig::new(3, 2)).unwrap();
    assert!(h.loss.iter().all(|l| l.is_finite()), "{:?}", h.loss);
    let a: Vec<Vec<u32>> = m.trainable_parameters().iter().map(|p| dense(p)).collect();
    assert_ne!(b, a, "fit でパラメータが更新されるはず");

    // 常駐経路: 無状態層として通過し predict と bit 一致。
    let resident = Sequential::new()
        .add_upsample_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(16, 4, 21)
        .unwrap()
        .add_relu()
        .add_linear(4, 2, 22)
        .unwrap();
    let rx = ramp(&[2, 1, 2, 2], 1.0);
    let init_tape = fandhe_ai::tape();
    let store = resident.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    assert_eq!(
        dense(&resident.predict_resident(&store, &rx).unwrap()),
        dense(&resident.predict(&rx).unwrap())
    );

    // Conv2d と混ぜた常駐は従来どおり拒否される。
    let mixed = Sequential::new()
        .add_conv2d(1, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, 11)
        .unwrap()
        .add_upsample_scale_factor(vec![2.0, 2.0], InterpolateMode::Nearest)
        .unwrap();
    let tape = fandhe_ai::tape();
    assert!(mixed.init_device_param_store(&tape).is_err());
}

#[test]
fn add_upsample_scale_factor_fn_pointer_signature_is_pinned() {
    let _: fn(Sequential, Vec<f64>, InterpolateMode) -> Result<Sequential, AutodiffError> =
        Sequential::add_upsample_scale_factor;
}

#[cfg(unix)]
#[test]
fn add_upsample_scale_factor_save_load_round_trip_bit_identical() {
    let model = Sequential::new()
        .add_upsample_scale_factor(vec![1.5, 2.0], InterpolateMode::Nearest)
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(48, 3, 5)
        .unwrap();
    let x = ramp(&[2, 1, 4, 4], 1.0);
    let guard = TempDirGuard::new("spatial_layers_scale_factor_round_trip");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
    assert!(
        text.contains("\"kind\":\"upsample_scale_factor\""),
        "{text}"
    );
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(
        dense(&model.predict(&x).unwrap()),
        dense(&loaded.predict(&x).unwrap())
    );
}
