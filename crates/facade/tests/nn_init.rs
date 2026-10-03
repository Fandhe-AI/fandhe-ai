//! `fandhe_ai::nn::init`（イシュー #2504）の facade 統合テスト。
//!
//! 役割: autodiff 側（`fandhe_ai_autodiff::nn::init`。#2140）の純再エクス
//! ポートであることを、決定性・内部クレートとの bit 一致・エラー伝播・
//! `compat::Sequential::load_state_dict` 経由の利用で確認する。ホスト側の
//! 生成のみでバックエンドは関与しないため、CUDA／Metal 実機は不要。
//! 乱数はプロセスグローバル RNG（`fandhe_ai::manual_seed`）を共有するため、
//! 本ファイル内のテストは `Mutex` で直列化する（`rng_tensor_generation.rs`
//! と同形）。

use std::sync::Mutex;

use fandhe_ai::compat::Sequential;
use fandhe_ai::nn::init::{
    FanMode, Nonlinearity, calculate_fan_in_and_fan_out, calculate_gain, constant, kaiming_normal,
    kaiming_uniform, normal, orthogonal, trunc_normal, uniform, xavier_normal, xavier_uniform,
};
use fandhe_ai::{AutodiffError, Tensor};

static RNG_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("test fixture: contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 9 関数を同じ順序で呼び、各出力の bit 列を返す（facade 経由）。
fn run_all_facade(seed: u64) -> Vec<Vec<u32>> {
    fandhe_ai::manual_seed(seed);
    vec![
        bits(&uniform(&[3, 4], -0.5, 0.5).expect("test fixture: uniform")),
        bits(&normal(&[3, 4], 0.0, 1.0).expect("test fixture: normal")),
        bits(&constant(&[3, 4], 0.25).expect("test fixture: constant")),
        bits(&xavier_uniform(&[3, 4], 1.0).expect("test fixture: xavier_uniform")),
        bits(&xavier_normal(&[3, 4], 1.0).expect("test fixture: xavier_normal")),
        bits(
            &kaiming_uniform(&[3, 4], 0.0, FanMode::FanIn, Nonlinearity::Relu)
                .expect("test fixture: kaiming_uniform"),
        ),
        bits(
            &kaiming_normal(&[3, 4], 0.0, FanMode::FanOut, Nonlinearity::Relu)
                .expect("test fixture: kaiming_normal"),
        ),
        bits(&orthogonal(&[4, 4], 1.0).expect("test fixture: orthogonal")),
        bits(&trunc_normal(&[3, 4], 0.0, 1.0, -2.0, 2.0).expect("test fixture: trunc_normal")),
    ]
}

/// 同じ呼び出し列を内部クレート `fandhe_ai_autodiff::nn::init` で直接実行する。
fn run_all_direct(seed: u64) -> Vec<Vec<u32>> {
    use fandhe_ai_autodiff::nn::init as d;
    fandhe_ai::manual_seed(seed);
    vec![
        bits(&d::uniform(&[3, 4], -0.5, 0.5).expect("test fixture: uniform")),
        bits(&d::normal(&[3, 4], 0.0, 1.0).expect("test fixture: normal")),
        bits(&d::constant(&[3, 4], 0.25).expect("test fixture: constant")),
        bits(&d::xavier_uniform(&[3, 4], 1.0).expect("test fixture: xavier_uniform")),
        bits(&d::xavier_normal(&[3, 4], 1.0).expect("test fixture: xavier_normal")),
        bits(
            &d::kaiming_uniform(&[3, 4], 0.0, d::FanMode::FanIn, d::Nonlinearity::Relu)
                .expect("test fixture: kaiming_uniform"),
        ),
        bits(
            &d::kaiming_normal(&[3, 4], 0.0, d::FanMode::FanOut, d::Nonlinearity::Relu)
                .expect("test fixture: kaiming_normal"),
        ),
        bits(&d::orthogonal(&[4, 4], 1.0).expect("test fixture: orthogonal")),
        bits(&d::trunc_normal(&[3, 4], 0.0, 1.0, -2.0, 2.0).expect("test fixture: trunc_normal")),
    ]
}

#[test]
fn same_seed_gives_bit_identical_outputs() {
    let _g = lock();
    assert_eq!(run_all_facade(1234), run_all_facade(1234));
    assert_ne!(run_all_facade(1234), run_all_facade(4321));
}

#[test]
fn facade_matches_internal_crate_bit_for_bit() {
    let _g = lock();
    assert_eq!(run_all_facade(99), run_all_direct(99));
}

#[test]
fn helper_functions_return_expected_values() {
    assert_eq!(calculate_gain(Nonlinearity::Relu), std::f32::consts::SQRT_2);
    assert_eq!(
        calculate_fan_in_and_fan_out(&[8, 3, 5, 5]).expect("test fixture: fan"),
        (3 * 25, 8 * 25)
    );
    assert!(calculate_fan_in_and_fan_out(&[7]).is_err());
}

#[test]
fn invalid_arguments_propagate_typed_errors() {
    let _g = lock();
    assert!(matches!(
        uniform(&[2, 2], 1.0, -1.0),
        Err(AutodiffError::InvalidArgument { .. })
    ));
    assert!(matches!(
        normal(&[2, 2], 0.0, -1.0),
        Err(AutodiffError::InvalidArgument { .. })
    ));
}

/// `compat::Sequential` の層へ初期化結果を `load_state_dict` で流し込める。
/// 注意（決定記録 §2.2）: `add_linear` の weight は `[in, out]`（PyTorch と
/// 転置）。`kaiming_uniform(&[in, out], ..)` は fan_in/fan_out の意味が
/// 入れ替わるため、ここでは形状だけが合えばよい `uniform`／`constant` を使う。
#[test]
fn init_values_load_into_sequential_via_state_dict() {
    let _g = lock();
    fandhe_ai::manual_seed(5);
    let w = uniform(&[2, 3], -0.1, 0.1).expect("test fixture: uniform");
    let b = constant(&[3], 0.0).expect("test fixture: constant");
    let mut seq = Sequential::new()
        .add_linear(2, 3, 0)
        .expect("test fixture: add_linear");
    let mut sd = seq.state_dict();
    sd.insert("0.weight".to_string(), w.clone());
    sd.insert("0.bias".to_string(), b.clone());
    seq.load_state_dict(sd)
        .expect("test fixture: load_state_dict");
    let got = seq.state_dict();
    assert_eq!(bits(&got["0.weight"]), bits(&w));
    assert_eq!(bits(&got["0.bias"]), bits(&b));
}

/// `manual_seed` は個別シード API（`add_linear(.., seed)`）に影響しない。
#[test]
fn manual_seed_does_not_affect_per_layer_seed() {
    let _g = lock();
    fandhe_ai::manual_seed(1);
    let a = Sequential::new()
        .add_linear(2, 3, 77)
        .expect("test fixture: add_linear");
    fandhe_ai::manual_seed(2);
    let b = Sequential::new()
        .add_linear(2, 3, 77)
        .expect("test fixture: add_linear");
    assert_eq!(
        bits(&a.state_dict()["0.weight"]),
        bits(&b.state_dict()["0.weight"])
    );
}
