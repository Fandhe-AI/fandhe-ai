//! `nn::Dropout2d`／`nn::AlphaDropout`（イシュー #2161・親 #2131）の
//! 統合テスト。`nn_dropout.rs`（[`Dropout`]）と同じ方針: グローバル
//! RNG を `manual_seed` で固定し、`fandhe_ai_autodiff::rand`（内部の
//! `crate::grad::dropout_mask`／`alpha_dropout_mask_and_bias` が呼ぶのと
//! 同じ一様乱数列）から期待値を機械的に再構成する。
//!
//! グローバル RNG 状態を書き換えるテストはファイル局所 `Mutex` で
//! 直列化する（`nn_dropout.rs`・`crates/facade/tests/
//! rng_tensor_generation.rs` と同型）。

mod common;

use std::sync::Mutex;

use fandhe_ai_autodiff::nn::{AlphaDropout, Dropout2d, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape, manual_seed, rand};
use fandhe_ai_tensor_core::Tensor;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// --- Dropout2d -----------------------------------------------------------

#[test]
fn dropout2d_new_rejects_out_of_range_p() {
    assert!(Dropout2d::new(-0.1).is_err());
    assert!(Dropout2d::new(1.1).is_err());
    assert!(Dropout2d::new(f32::NAN).is_err());
}

#[test]
fn dropout2d_default_is_p_half_training_true() {
    let d = Dropout2d::default();
    assert_eq!(d.p(), 0.5);
    assert!(d.training());
}

#[test]
fn dropout2d_forward_rejects_rank_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let d = Dropout2d::new(0.5).expect("有効な p");
    let Err(err) = d.forward(&x) else {
        panic!("rank != 4 は Err を返すはず")
    };
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn dropout2d_eval_mode_is_identity_without_consuming_rng() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(42);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((0..24).map(|v| v as f32).collect(), &[2, 3, 2, 2]));
    let mut d = Dropout2d::new(0.5).expect("有効な p");
    d.set_training(false);

    manual_seed(42);
    let before = rand(&[1]).expect("shape [1] は失敗しない");

    manual_seed(42);
    let out = d.forward(&x).expect("eval モードは恒等写像");
    let after = rand(&[1]).expect("shape [1] は失敗しない");

    assert_eq!(out.to_tensor().host_slice(), x.to_tensor().host_slice());
    // 早期リターンのため RNG は消費されていない: 同じ seed 直後に
    // 「forward を挟まず 1 回 `rand([1])`」と「forward を挟んでから
    // 1 回 `rand([1])`」を比較し、両者が一致すれば forward が RNG を
    // 消費していないことを意味する。
    assert_eq!(before.host_slice(), after.host_slice());
}

#[test]
fn dropout2d_channel_is_fully_zero_or_fully_scaled() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(7);
    let tape = Tape::new_with_ops(common::naive_ops());
    let shape = [2usize, 3, 2, 2];
    let numel: usize = shape.iter().product();
    let x = tape.var(&t((0..numel).map(|v| v as f32 + 1.0).collect(), &shape));
    let p = 0.4_f32;
    let d = Dropout2d::new(p).expect("有効な p");

    let out = d.forward(&x).expect("rank 4・training 既定 true");
    let out_data = out.to_tensor().host_slice().into_owned();
    let scale = 1.0f32 / (1.0f32 - p);

    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    for ni in 0..n {
        for ci in 0..c {
            let base = (ni * c + ci) * h * w;
            let channel = &out_data[base..base + h * w];
            let all_zero = channel.iter().all(|&v| v == 0.0);
            let all_scaled_or_zero = channel
                .iter()
                .zip((0..numel).skip(base).take(h * w))
                .all(|(&v, idx)| v == 0.0 || v == (idx as f32 + 1.0) * scale);
            assert!(
                all_zero || all_scaled_or_zero,
                "channel (n={ni}, c={ci}) is neither fully zero nor fully scaled: {channel:?}"
            );
            // チャネル内は全要素が同じ「生存／死亡」でなければならない
            // （feature dropout の定義）。
            let alive_count = channel.iter().filter(|&&v| v != 0.0).count();
            assert!(
                alive_count == 0 || alive_count == h * w,
                "channel (n={ni}, c={ci}) is partially dropped: {channel:?}"
            );
        }
    }
}

#[test]
fn dropout2d_p_one_is_all_zero() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(1);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((0..16).map(|v| v as f32 + 1.0).collect(), &[1, 2, 2, 4]));
    let d = Dropout2d::new(1.0).expect("有効な p");
    let out = d.forward(&x).expect("rank 4・p=1.0");
    assert!(out.to_tensor().host_slice().iter().all(|&v| v == 0.0));
}

#[test]
fn dropout2d_forward_host_matches_module_forward_bit_exact() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [2usize, 2, 2, 2];
    let x_data: Vec<f32> = (0..16).map(|v| v as f32 + 1.0).collect();
    let ops = common::naive_ops();
    let d = Dropout2d::new(0.5).expect("有効な p");

    manual_seed(123);
    let via_forward_host = d
        .forward_host(&*ops, &t(x_data.clone(), &shape))
        .expect("forward_host は成功する");

    manual_seed(123);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data, &shape));
    let via_forward = d.forward(&x).expect("forward は成功する");

    assert_eq!(
        via_forward_host.host_slice(),
        via_forward.to_tensor().host_slice()
    );
}

// --- AlphaDropout ----------------------------------------------------------

const SELU_ALPHA: f64 = 1.7580993408473766;

/// `(noise, bias)` の要素ごとペアを返す（`crate::grad::
/// alpha_dropout_mask_and_bias` の doc「丸め順序」節と同じ手順を
/// テスト側で独立に再現する）。`noise + bias` へ合算しない理由:
/// `f32` の加算は丸めを伴うため、後から `bias` だけを取り出そうと
/// 減算で復元すると丸め誤差が混入しうる（自己レビューで判明）。
fn alpha_dropout_expected(uniforms: &[f32], p: f32) -> Vec<(f32, f32)> {
    let p64 = f64::from(p);
    let a = 1.0 / ((SELU_ALPHA * SELU_ALPHA * p64 + 1.0) * (1.0 - p64)).sqrt();
    let alpha_a = SELU_ALPHA * a;
    let alpha_a_p = alpha_a * p64;
    let a_f32 = a as f32;
    let alpha_a_f32 = alpha_a as f32;
    let alpha_a_p_f32 = alpha_a_p as f32;
    uniforms
        .iter()
        .map(|&u| {
            if u >= p {
                (a_f32, alpha_a_p_f32)
            } else {
                (0.0, -alpha_a_f32 + alpha_a_p_f32)
            }
        })
        .collect::<Vec<(f32, f32)>>()
}

#[test]
fn alpha_dropout_new_rejects_out_of_range_p() {
    assert!(AlphaDropout::new(-0.1).is_err());
    assert!(AlphaDropout::new(1.1).is_err());
}

#[test]
fn alpha_dropout_default_is_p_half() {
    let d = AlphaDropout::default();
    assert_eq!(d.p(), 0.5);
    assert!(d.training());
}

#[test]
fn alpha_dropout_p_one_is_all_zero() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, -4.0], &[4]));
    let d = AlphaDropout::new(1.0).expect("有効な p");
    let out = d.forward(&x).expect("p=1.0 は特例で全ゼロ");
    assert!(out.to_tensor().host_slice().iter().all(|&v| v == 0.0));
}

#[test]
fn alpha_dropout_eval_mode_is_identity() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, -4.0], &[4]));
    let mut d = AlphaDropout::new(0.5).expect("有効な p");
    d.set_training(false);
    let out = d.forward(&x).expect("eval モードは恒等写像");
    assert_eq!(out.to_tensor().host_slice(), x.to_tensor().host_slice());
}

#[test]
fn alpha_dropout_forward_matches_analytic_formula() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [8usize];
    let p = 0.3_f32;
    let x_value = 0.5_f32;

    manual_seed(2026);
    let uniforms = rand(&shape).expect("shape [8]");
    let uniforms_data = uniforms.host_slice().into_owned();

    manual_seed(2026);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![x_value; 8], &shape));
    let d = AlphaDropout::new(p).expect("有効な p");
    let out = d.forward(&x).expect("training 既定 true");
    let out_data = out.to_tensor().host_slice().into_owned();

    // `x` が全要素同一値のため `out = x_value * noise + bias`。
    let expected_pairs = alpha_dropout_expected(&uniforms_data, p);
    for (i, &(noise, bias)) in expected_pairs.iter().enumerate() {
        let expected = x_value * noise + bias;
        assert_eq!(out_data[i], expected, "index {i}");
    }
}

#[test]
fn alpha_dropout_forward_host_matches_module_forward_bit_exact() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [6usize];
    let x_data = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
    let ops = common::naive_ops();
    let d = AlphaDropout::new(0.4).expect("有効な p");

    manual_seed(99);
    let via_forward_host = d
        .forward_host(&*ops, &t(x_data.clone(), &shape))
        .expect("forward_host は成功する");

    manual_seed(99);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data, &shape));
    let via_forward = d.forward(&x).expect("forward は成功する");

    assert_eq!(
        via_forward_host.host_slice(),
        via_forward.to_tensor().host_slice()
    );
}

#[test]
fn alpha_dropout_backward_gradient_equals_noise() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(555);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let d = AlphaDropout::new(0.5).expect("有効な p");
    let out = d.forward(&x).expect("training 既定 true");
    let loss = out.sum(None).expect("全軸縮約は失敗しない");
    let grads = tape.backward(&loss).expect("x は requires_grad の葉");
    let dx = grads
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");

    // 手計算: `b` は `x` に依存しない定数のため `d(out)/dx = noise`。
    manual_seed(555);
    let uniforms = rand(&[4]).expect("shape [4]");
    let uniforms_data = uniforms.host_slice().into_owned();
    let p = 0.5_f32;
    let p64 = f64::from(p);
    let a = (1.0 / ((SELU_ALPHA * SELU_ALPHA * p64 + 1.0) * (1.0 - p64)).sqrt()) as f32;
    let expected: Vec<f32> = uniforms_data
        .iter()
        .map(|&u| if u >= p { a } else { 0.0 })
        .collect();

    assert_eq!(dx.host_slice().into_owned(), expected);
}
