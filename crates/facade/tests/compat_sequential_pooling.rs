//! `compat::Sequential::add_max_pool2d`／`add_max_pool1d`／
//! `add_avg_pool2d`／`add_avg_pool1d`／`add_adaptive_avg_pool2d`／
//! `add_adaptive_avg_pool1d`（イシュー #1957・親 #1618。2026-09-17
//! ユーザー承認〈選択肢 A・6 メソッド一括追加〉）の facade 公開面を
//! 検証する統合テスト。
//!
//! - 受入条件（issue 本文）: `Sequential` 経由（`predict`）の出力が
//!   `Var::max_pool2d` 等の直接呼び出しと **bit 完全一致**（6 型とも）。
//! - `predict`（tape 不要経路）と `forward`（`fandhe_ai::tape()` 上）が
//!   bit 完全一致（6 型とも）。
//! - backward の入力勾配が直接呼び出しで組んだ同一グラフと bit 完全
//!   一致（Max・Avg 代表各 1）。
//! - 無効引数（`kernel_size=0`・`stride=Some(0)`・`dilation=0`・
//!   `2*padding > kernel`・`output_size=0`）が `Err`（panic しない）。
//! - rank 不一致（2d 層へ rank 3・1d 層へ rank 4）が `Err`。
//! - パラメータ非寄与（`trainable_parameters().len()` が conv のみの
//!   件数と一致・`state_dict` round trip が pooling を挟んでも不変）。
//! - 学習ループ（conv → relu → max_pool2d／avg_pool2d）で loss が減少。
//! - 常駐経路 fail-closed（`init_device_param_store`／
//!   `forward_resident`／`predict_resident` が `Unsupported`）。
//! - train／eval 非依存（無状態層であることの確認）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fandhe_ai::tape()` 既定 CPU 経由）。CUDA／Metal 実機での facade
//! parity は未実測のまま Mac／GB10 セッションへ申し送り（既存 Pooling
//! カーネル自体の parity は #1902／#1903 で実測済み。本 issue は新規
//! カーネルを追加しない）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn input_2d() -> Tensor<f32> {
    tensor(
        (0..2 * 3 * 6 * 6)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect(),
        &[2, 3, 6, 6],
    )
}

fn input_1d() -> Tensor<f32> {
    tensor(
        (0..2 * 3 * 8).map(|i| (i as f32) * 0.01 - 0.3).collect(),
        &[2, 3, 8],
    )
}

// ---------------------------------------------------------------------
// 受入条件: Sequential 経由の出力が直接呼び出しと bit 完全一致。
// ---------------------------------------------------------------------

#[test]
fn max_pool2d_predict_matches_direct_call_bit_exact() {
    let model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let x = input_2d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let (direct, _index) = xv.max_pool2d([2, 2], None, [0, 0], [1, 1], false).unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

#[test]
fn max_pool1d_predict_matches_direct_call_bit_exact() {
    let model = Sequential::new().add_max_pool1d(2, Some(2), 0, 1).unwrap();
    let x = input_1d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let (direct, _index) = xv.max_pool1d(2, Some(2), 0, 1, false).unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

#[test]
fn avg_pool2d_predict_matches_direct_call_bit_exact_count_include_pad_true() {
    let model = Sequential::new()
        .add_avg_pool2d([3, 3], Some([1, 1]), [1, 1], true)
        .unwrap();
    let x = input_2d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let direct = xv
        .avg_pool2d([3, 3], Some([1, 1]), [1, 1], false, true)
        .unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

#[test]
fn avg_pool2d_predict_matches_direct_call_bit_exact_count_include_pad_false() {
    let model = Sequential::new()
        .add_avg_pool2d([3, 3], Some([1, 1]), [1, 1], false)
        .unwrap();
    let x = input_2d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let direct = xv
        .avg_pool2d([3, 3], Some([1, 1]), [1, 1], false, false)
        .unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

#[test]
fn avg_pool1d_predict_matches_direct_call_bit_exact() {
    let model = Sequential::new().add_avg_pool1d(2, None, 0, true).unwrap();
    let x = input_1d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let direct = xv.avg_pool1d(2, None, 0, false, true).unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

#[test]
fn adaptive_avg_pool2d_predict_matches_direct_call_bit_exact() {
    let model = Sequential::new().add_adaptive_avg_pool2d([2, 2]).unwrap();
    let x = input_2d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let direct = xv.adaptive_avg_pool2d([2, 2]).unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

#[test]
fn adaptive_avg_pool1d_predict_matches_direct_call_bit_exact() {
    let model = Sequential::new().add_adaptive_avg_pool1d(3).unwrap();
    let x = input_1d();

    let via_sequential = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let direct = xv.adaptive_avg_pool1d(3).unwrap();

    assert_eq!(dense_vec(&via_sequential), dense_vec(&direct.to_tensor()));
}

// ---------------------------------------------------------------------
// predict（tape 不要経路）と forward（tape 経路）の bit 完全一致。
// ---------------------------------------------------------------------

#[test]
fn max_pool2d_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let x = input_2d();

    let predicted = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn avg_pool2d_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_avg_pool2d([2, 2], None, [0, 0], true)
        .unwrap();
    let x = input_2d();

    let predicted = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

// ---------------------------------------------------------------------
// backward の bit 完全一致（Max・Avg 代表各 1）。
// ---------------------------------------------------------------------

#[test]
fn max_pool2d_backward_matches_direct_graph_bit_exact() {
    let x_data = input_2d();

    // Sequential 経由。
    let model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let y1 = model.forward(&tape1, &x1).unwrap();
    let loss1 = y1.sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1
        .get(&x1)
        .unwrap()
        .expect("入力 x1 には loss1 からの勾配が到達するはず");

    // 直接呼び出し。
    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let (y2, _index) = x2.max_pool2d([2, 2], None, [0, 0], [1, 1], false).unwrap();
    let loss2 = y2.sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2
        .get(&x2)
        .unwrap()
        .expect("入力 x2 には loss2 からの勾配が到達するはず");

    assert_eq!(dense_vec(dx1), dense_vec(dx2));
}

#[test]
fn avg_pool2d_backward_matches_direct_graph_bit_exact() {
    let x_data = input_2d();

    let model = Sequential::new()
        .add_avg_pool2d([2, 2], None, [0, 0], true)
        .unwrap();
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let y1 = model.forward(&tape1, &x1).unwrap();
    let loss1 = y1.sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1
        .get(&x1)
        .unwrap()
        .expect("入力 x1 には loss1 からの勾配が到達するはず");

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let y2 = x2.avg_pool2d([2, 2], None, [0, 0], false, true).unwrap();
    let loss2 = y2.sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2
        .get(&x2)
        .unwrap()
        .expect("入力 x2 には loss2 からの勾配が到達するはず");

    assert_eq!(dense_vec(dx1), dense_vec(dx2));
}

// ---------------------------------------------------------------------
// 無効引数の fail-fast（構築時点で `Err`。panic しない）。
// ---------------------------------------------------------------------

#[test]
fn add_max_pool2d_rejects_zero_kernel_size() {
    let err = Sequential::new()
        .add_max_pool2d([0, 2], None, [0, 0], [1, 1])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn add_max_pool2d_rejects_zero_stride() {
    let err = Sequential::new()
        .add_max_pool2d([2, 2], Some([0, 2]), [0, 0], [1, 1])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn add_max_pool2d_rejects_zero_dilation() {
    let err = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [0, 1])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn add_avg_pool2d_rejects_padding_exceeding_half_kernel() {
    let err = Sequential::new()
        .add_avg_pool2d([2, 2], None, [2, 0], true)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::InvalidArgument(_))
    ));
}

#[test]
fn add_adaptive_avg_pool2d_rejects_zero_output_size() {
    let err = Sequential::new()
        .add_adaptive_avg_pool2d([0, 2])
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_adaptive_avg_pool1d_rejects_zero_output_size() {
    let err = Sequential::new()
        .add_adaptive_avg_pool1d(0)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// ---------------------------------------------------------------------
// rank 不一致（2d 層へ rank 3・1d 層へ rank 4 入力）が `Err`。
// ---------------------------------------------------------------------

#[test]
fn max_pool2d_rejects_rank3_input() {
    let model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let x = input_1d(); // rank 3
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn avg_pool1d_rejects_rank4_input() {
    let model = Sequential::new().add_avg_pool1d(2, None, 0, true).unwrap();
    let x = input_2d(); // rank 4
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

// ---------------------------------------------------------------------
// パラメータ非寄与: conv → relu → pooling で trainable_parameters が
// conv のみの件数と一致・state_dict round trip が不変。
// ---------------------------------------------------------------------

#[test]
fn max_pool2d_does_not_contribute_trainable_parameters() {
    let model = Sequential::new()
        .add_conv2d(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 0x1234_5678)
        .unwrap()
        .add_relu()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();

    // Conv2d（weight・bias の 2 件）のみが学習可能パラメータ。
    // ReLU／MaxPool2d は無状態のため寄与しない。
    assert_eq!(model.trainable_parameters().len(), 2);

    let state = model.state_dict();
    // MaxPool2d 自身のキーは含まれない（Conv2d の weight／bias のみ）。
    assert_eq!(state.len(), 2);

    let x = input_2d();
    let out1 = model.predict(&x).unwrap();

    let mut model2 = model;
    model2.load_state_dict(state).unwrap();
    let out2 = model2.predict(&x).unwrap();

    assert_eq!(dense_vec(&out1), dense_vec(&out2));
}

// ---------------------------------------------------------------------
// 学習ループ: conv → relu → pooling を Sgd で数 step 回し loss が減少。
// ---------------------------------------------------------------------

#[test]
fn conv_relu_max_pool2d_training_loop_reduces_loss() {
    let mut model = Sequential::new()
        .add_conv2d(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 0x1234_5678)
        .unwrap()
        .add_relu()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();

    let x = input_2d();
    let target = tensor(vec![0.1f32; 2 * 4 * 3 * 3], &[2, 4, 3, 3]);

    let mut sgd = Sgd::new(SgdConfig::new(0.01)).unwrap();
    let mut losses = Vec::new();
    for _ in 0..5 {
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

    assert!(
        losses.last().unwrap() < &losses[0],
        "loss は減少するはず: {losses:?}"
    );
}

#[test]
fn conv_relu_avg_pool2d_training_loop_reduces_loss() {
    let mut model = Sequential::new()
        .add_conv2d(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 0x1234_5678)
        .unwrap()
        .add_relu()
        .add_avg_pool2d([2, 2], None, [0, 0], true)
        .unwrap();

    let x = input_2d();
    let target = tensor(vec![0.1f32; 2 * 4 * 3 * 3], &[2, 4, 3, 3]);

    let mut sgd = Sgd::new(SgdConfig::new(0.01)).unwrap();
    let mut losses = Vec::new();
    for _ in 0..5 {
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

    assert!(
        losses.last().unwrap() < &losses[0],
        "loss は減少するはず: {losses:?}"
    );
}

// ---------------------------------------------------------------------
// 常駐経路 fail-closed（Pooling を含む Sequential の 3 API すべて）。
// ---------------------------------------------------------------------

#[test]
fn init_device_param_store_rejects_pooling_layer() {
    let model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)));
}

#[test]
fn forward_resident_rejects_pooling_layer() {
    // 入口ガードは forward の実行より前に効くため、`store` は
    // 別モデル（Linear のみ）から得たものでよい（二重防御の検証。
    // `contains_resident_unsupported_layer` doc 参照）。
    let linear_only = Sequential::new().add_linear(6, 6, 0xAAAA_BBBB).unwrap();
    let init_tape = fandhe_ai::tape();
    let mut store = linear_only.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let pooling_model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let tape = fandhe_ai::tape();
    let x = tape.var(&input_2d());
    let err = pooling_model
        .forward_resident(&tape, &x, &mut store)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::Unsupported(_))
    ));
}

#[test]
fn predict_resident_rejects_pooling_layer() {
    let linear_only = Sequential::new().add_linear(6, 6, 0xAAAA_BBBB).unwrap();
    let init_tape = fandhe_ai::tape();
    let store = linear_only.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let pooling_model = Sequential::new()
        .add_avg_pool2d([2, 2], None, [0, 0], true)
        .unwrap();
    let x = input_2d();
    let err = pooling_model.predict_resident(&store, &x).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Backend(BackendError::Unsupported(_))
    ));
}

// ---------------------------------------------------------------------
// train／eval 非依存（無状態層であることの確認）。
// ---------------------------------------------------------------------

#[test]
fn max_pool2d_predict_is_mode_independent() {
    let mut model = Sequential::new()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let x = input_2d();

    let before = model.predict(&x).unwrap();
    model.eval();
    let after_eval = model.predict(&x).unwrap();
    model.train();
    let after_train = model.predict(&x).unwrap();

    assert_eq!(dense_vec(&before), dense_vec(&after_eval));
    assert_eq!(dense_vec(&before), dense_vec(&after_train));
}
