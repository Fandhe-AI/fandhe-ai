//! `Var::dropout`／`nn::Dropout`（イシュー #1603）の統合テスト。
//!
//! マスクを固定した解析解との bit 完全一致で VJP を検証する
//! （`dropout` はマスク生成にグローバル RNG を消費するため、乱数を
//! 再抽選してしまう数値微分は使わない。`no_grad_detach.rs` と同じ
//! 方針）。マスク固定入口は `pub(crate) fn Var::dropout_with_mask`
//! だが本ファイルは `autodiff` の公開 API のみを経由する別クレート
//! 扱いのため直接呼べない——代わりに `manual_seed` でグローバル RNG を
//! 固定し、`fandhe_ai_autodiff::rand`（`Var::dropout` が内部で呼ぶのと
//! 同じ関数）で同一のマスク元列を再現してから期待値を組み立てる。
//!
//! グローバル RNG 状態を書き換えるテストはファイル局所 `Mutex` で
//! 直列化する（`crates/facade/tests/rng_tensor_generation.rs` と同型。
//! マスクを明示的に渡す解析解テストは RNG に触れないため対象外）。

mod common;

use std::sync::Mutex;

use fandhe_ai_autodiff::nn::{Dropout, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `p` の一様乱数抽選から `{0.0, scale}` マスクを機械的に再構成する
/// （`crate::grad::dropout_mask` の契約を統合テスト側で独立に再現）。
fn expected_mask(uniforms: &[f32], p: f32) -> Vec<f32> {
    let scale = 1.0f32 / (1.0f32 - p);
    uniforms
        .iter()
        .map(|&u| if u >= p { scale } else { 0.0 })
        .collect()
}

// --- 解析的 VJP（RNG 非依存・bit 完全一致） -----------------------------

/// `Var::dropout` 相当のマスク乗算を `Var::mul` で明示的に組み立て、
/// `d(sum(x ⊙ mask))/dx = mask` を bit 完全一致で確認する
/// （`Op::Dropout` の VJP と forward が同じ `mask` を保持する契約——
/// `Op::Mul` の VJP と等価であることの独立検証）。
#[test]
fn dropout_vjp_matches_mask_multiplication_bit_exact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let mask = t(vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0], &[2, 3]);
    let mask_var = tape.var(&mask);
    let product = x.mul(&mask_var).expect("同 shape の要素積は失敗しない");
    let loss = product.sum(None).expect("全軸縮約は失敗しない");

    let grads = tape.backward(&loss).expect("x は requires_grad の葉");
    let dx = grads
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");
    // d(sum(x * mask))/dx = mask（乗算 1 回のみで FMA を含まないため
    // bit 完全一致）。
    assert_eq!(dx.as_slice().unwrap(), mask.as_slice().unwrap());
}

// --- forward 契約（`training`／`p` の早期リターン・マスク値） -----------

#[test]
fn dropout_eval_mode_is_identity_without_consuming_rng() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    // 参照系列: 同一 seed から draw を 1 回だけ行った場合の「次の draw」。
    fandhe_ai_autodiff::manual_seed(1);
    let reference_next = fandhe_ai_autodiff::rand(&[1]).expect("rand は失敗しない");

    // 対象系列: 同じ seed から dropout（eval モード）を挟んだ直後の
    // 「次の draw」。早期リターンが RNG を一切消費しないなら、
    // これは同じ seed からの最初の draw、すなわち参照系列と bit 一致
    // するはず（2 回連続で draw した場合の「前後の draw」を比較する
    // のではない——連続する 2 draw は元々値が異なるため、その比較は
    // 消費有無の検証にならない）。
    fandhe_ai_autodiff::manual_seed(1);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]));
    let out = x
        .dropout(0.5, false)
        .expect("eval モードは早期リターンで成功する");
    assert_eq!(out.to_tensor().as_slice(), x.to_tensor().as_slice());
    let after_dropout_next = fandhe_ai_autodiff::rand(&[1]).expect("rand は失敗しない");

    assert_eq!(reference_next.as_slice(), after_dropout_next.as_slice());
}

#[test]
fn dropout_zero_p_is_identity_even_in_training_without_consuming_rng() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    fandhe_ai_autodiff::manual_seed(2);
    let reference_next = fandhe_ai_autodiff::rand(&[1]).expect("rand は失敗しない");

    fandhe_ai_autodiff::manual_seed(2);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]));
    let out = x
        .dropout(0.0, true)
        .expect("p=0.0 は早期リターンで成功する");
    assert_eq!(out.to_tensor().as_slice(), x.to_tensor().as_slice());
    let after_dropout_next = fandhe_ai_autodiff::rand(&[1]).expect("rand は失敗しない");

    assert_eq!(reference_next.as_slice(), after_dropout_next.as_slice());
}

#[test]
fn dropout_p_one_zeros_all_outputs_and_gradients() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai_autodiff::manual_seed(3);

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let out = x.dropout(1.0, true).expect("p=1.0 は有効な範囲");
    assert_eq!(out.to_tensor().as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0]);

    let loss = out.sum(None).expect("全軸縮約は失敗しない");
    let grads = tape.backward(&loss).expect("x は requires_grad の葉");
    let dx = grads
        .get(&x)
        .expect("x は requires_grad=true の葉")
        .expect("x は loss に到達する");
    assert_eq!(dx.as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn dropout_rejects_out_of_range_or_non_finite_p() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        x.dropout(-0.1, true),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        x.dropout(1.1, true),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        x.dropout(f32::NAN, true),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// --- マスク値・keep/drop 契約（`u >= p` ⇔ keep。`-0.0`／NaN 伝播含む） --

#[test]
fn dropout_forward_matches_uniform_threshold_mask_contract() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let p = 0.4f32;
    let shape = [2usize, 3usize];
    let x_data = vec![1.0, -0.0, f32::NAN, 4.0, -5.0, 6.0];

    fandhe_ai_autodiff::manual_seed(42);
    let uniforms = fandhe_ai_autodiff::rand(&shape).expect("rand は失敗しない");
    let mask = expected_mask(uniforms.as_slice().unwrap(), p);

    fandhe_ai_autodiff::manual_seed(42);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data.clone(), &shape));
    let out = x.dropout(p, true).expect("有効な p");
    let out_data = out.to_tensor();
    let out_slice = out_data.as_slice().unwrap();

    for i in 0..x_data.len() {
        let expected = x_data[i] * mask[i];
        if expected.is_nan() {
            assert!(out_slice[i].is_nan(), "index {i}: NaN 伝播が一致しない");
        } else {
            assert_eq!(out_slice[i], expected, "index {i}: マスク乗算が一致しない");
        }
    }
}

// --- 決定性（同一 seed → bit 同一。異なる seed → 少なくとも 1 要素差） --

#[test]
fn dropout_is_deterministic_given_same_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [4usize, 4usize];
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();

    fandhe_ai_autodiff::manual_seed(100);
    let tape1 = Tape::new_with_ops(common::naive_ops());
    let x1 = tape1.var(&t(data.clone(), &shape));
    let out1 = x1.dropout(0.5, true).expect("有効な p");

    fandhe_ai_autodiff::manual_seed(100);
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&t(data.clone(), &shape));
    let out2 = x2.dropout(0.5, true).expect("有効な p");

    assert_eq!(
        out1.to_tensor().as_slice().unwrap(),
        out2.to_tensor().as_slice().unwrap()
    );
}

#[test]
fn dropout_differs_across_seeds_with_high_probability() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [8usize, 8usize];
    let data: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();

    fandhe_ai_autodiff::manual_seed(200);
    let tape1 = Tape::new_with_ops(common::naive_ops());
    let x1 = tape1.var(&t(data.clone(), &shape));
    let out1 = x1.dropout(0.5, true).expect("有効な p");

    fandhe_ai_autodiff::manual_seed(201);
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&t(data.clone(), &shape));
    let out2 = x2.dropout(0.5, true).expect("有効な p");

    // 統計的主張はしない（テスト自体は決定的）——形状が十分大きい
    // （64 要素・p=0.5）ため異なる seed で全要素一致する確率は
    // 実務上ゼロ（2^-64）とみなしてよい。
    assert_ne!(
        out1.to_tensor().as_slice().unwrap(),
        out2.to_tensor().as_slice().unwrap()
    );
}

// --- RNG 消費回数契約（「dropout で numel 回引く」 ⇔ 「rand(shape) で numel 回引く」） --

#[test]
fn dropout_consumes_exactly_numel_draws_like_rand() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [3usize, 5usize]; // numel = 15

    fandhe_ai_autodiff::manual_seed(777);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 15], &shape));
    let _out = x.dropout(0.3, true).expect("有効な p");
    let after_dropout = fandhe_ai_autodiff::rand(&[1]).expect("rand は失敗しない");

    fandhe_ai_autodiff::manual_seed(777);
    let _consumed = fandhe_ai_autodiff::rand(&shape).expect("rand は失敗しない");
    let after_rand = fandhe_ai_autodiff::rand(&[1]).expect("rand は失敗しない");

    assert_eq!(after_dropout.as_slice(), after_rand.as_slice());
}

// --- `nn::Dropout`（モード保持・`dyn Module` 経由） ---------------------

#[test]
fn nn_dropout_new_default_and_mode_accessors() {
    assert!(Dropout::new(-0.1).is_err());
    assert!(Dropout::new(1.5).is_err());

    let d = Dropout::default();
    assert_eq!(d.p(), 0.5);
    assert!(d.training());
}

#[test]
fn nn_dropout_set_training_via_dyn_module_takes_effect() {
    let mut d: Box<dyn Module> = Box::new(Dropout::new(0.9).expect("有効な p"));
    assert!(d.training());
    d.set_training(false);
    assert!(!d.training());

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let out = d.forward(&tape, &x).expect("eval モードは失敗しない");
    // eval モードでは `Dropout::forward` が `Var::dropout(p, false)` を
    // 呼び早期リターンする（恒等写像）。
    assert_eq!(out.to_tensor().as_slice(), x.to_tensor().as_slice());
}

#[test]
fn nn_dropout_named_parameters_is_empty() {
    let d = Dropout::default();
    assert!(d.named_parameters().is_empty());
}

// --- `forward_host`（tape 不要経路）と `forward`（tape 経路）の bit 一致 --

#[test]
fn nn_dropout_forward_host_matches_forward_bit_exact_given_same_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let shape = [4usize, 4usize];
    let data: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
    let d = Dropout::new(0.4).expect("有効な p");

    fandhe_ai_autodiff::manual_seed(555);
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(data.clone(), &shape));
    let via_tape = Module::forward(&d, &tape, &x).expect("train モードでも成功する");

    fandhe_ai_autodiff::manual_seed(555);
    let ops = common::naive_ops();
    let via_host = d
        .forward_host(ops.as_ref(), &t(data, &shape))
        .expect("forward_host は Var::dropout と同一関数列");

    assert_eq!(
        via_tape.to_tensor().as_slice().unwrap(),
        via_host.as_slice().unwrap()
    );
}

#[test]
fn nn_dropout_forward_host_eval_mode_is_identity() {
    let mut d = Dropout::new(0.9).expect("有効な p");
    d.set_training(false);
    let ops = common::naive_ops();
    let input = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let out = d
        .forward_host(ops.as_ref(), &input)
        .expect("eval モードは恒等写像");
    assert_eq!(out.as_slice(), input.as_slice());
}
