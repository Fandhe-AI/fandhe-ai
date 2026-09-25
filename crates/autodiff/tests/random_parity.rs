//! `bernoulli`／`multinomial`／`normal`・`Generator`（イシュー #2156・親
//! #2131）の統合テスト。実体は `tensor-core::rng` にあり、本クレートは
//! `fandhe_ai_autodiff::{bernoulli, multinomial, normal, Generator,
//! manual_seed, RngError}` として素通しするだけ（`crates/autodiff/
//! src/lib.rs` 参照）。本ファイルは「グローバル自由関数（`manual_seed`
//! 経由）と `Generator` 経路が bit 完全一致する」という #2156 の中核
//! 契約と、`normal(mean, std, shape)` が `nn::init::normal(shape, mean,
//! std)` と（`std > 0` の場合に）bit 一致するという横断契約を固定する。
//!
//! プロセスグローバルな決定的 RNG に従属するテスト同士が既定の並列実行で
//! 競合しないよう、ファイル局所 `Mutex` で直列化する
//! （`crates/autodiff/tests/nn_init.rs`・`crates/facade/tests/
//! rng_tensor_generation.rs` と同型のパターン）。

use std::sync::Mutex;

use fandhe_ai_autodiff::nn::init;
use fandhe_ai_autodiff::{Generator, RngError, bernoulli, manual_seed, multinomial, normal};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

// ---------------------------------------------------------------------
// グローバル経路の決定性（同一 seed で再現すること）
// ---------------------------------------------------------------------

#[test]
fn bernoulli_reproducible_under_same_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let probs = Tensor::new(vec![0.5f32; 16], &[16]).unwrap();
    manual_seed(31);
    let a = bernoulli(&probs).unwrap();
    manual_seed(31);
    let b = bernoulli(&probs).unwrap();
    for i in 0..16 {
        assert_eq!(a.get(&[i]), b.get(&[i]));
    }
}

#[test]
fn multinomial_reproducible_under_same_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let weights = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    manual_seed(32);
    let a = multinomial(&weights, 10, true).unwrap();
    manual_seed(32);
    let b = multinomial(&weights, 10, true).unwrap();
    for i in 0..10 {
        assert_eq!(a.get(&[i]), b.get(&[i]));
    }
}

#[test]
fn normal_reproducible_under_same_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    manual_seed(33);
    let a = normal(1.5, 0.5, &[16]).unwrap();
    manual_seed(33);
    let b = normal(1.5, 0.5, &[16]).unwrap();
    for i in 0..16 {
        assert_eq!(a.get(&[i]), b.get(&[i]));
    }
}

#[test]
fn different_seeds_diverge_for_all_three_distributions() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let probs = Tensor::new(vec![0.5f32; 64], &[64]).unwrap();
    manual_seed(1);
    let a = bernoulli(&probs).unwrap();
    manual_seed(2);
    let b = bernoulli(&probs).unwrap();
    let a_vals: Vec<f32> = (0..64).map(|i| a.get(&[i]).unwrap()).collect();
    let b_vals: Vec<f32> = (0..64).map(|i| b.get(&[i]).unwrap()).collect();
    assert_ne!(
        a_vals, b_vals,
        "異なる seed で bernoulli の出力が一致してしまった"
    );

    let weights = Tensor::new(vec![1.0f32; 8], &[8]).unwrap();
    manual_seed(1);
    let a = multinomial(&weights, 32, true).unwrap();
    manual_seed(2);
    let b = multinomial(&weights, 32, true).unwrap();
    let a_vals: Vec<i32> = (0..32).map(|i| a.get(&[i]).unwrap()).collect();
    let b_vals: Vec<i32> = (0..32).map(|i| b.get(&[i]).unwrap()).collect();
    assert_ne!(
        a_vals, b_vals,
        "異なる seed で multinomial の出力が一致してしまった"
    );

    manual_seed(1);
    let a = normal(0.0, 1.0, &[16]).unwrap();
    manual_seed(2);
    let b = normal(0.0, 1.0, &[16]).unwrap();
    let a_vals: Vec<f32> = (0..16).map(|i| a.get(&[i]).unwrap()).collect();
    let b_vals: Vec<f32> = (0..16).map(|i| b.get(&[i]).unwrap()).collect();
    assert_ne!(
        a_vals, b_vals,
        "異なる seed で normal の出力が一致してしまった"
    );
}

// ---------------------------------------------------------------------
// グローバル経路と Generator 経路の bit 一致（#2156 の中核契約）
// ---------------------------------------------------------------------

#[test]
fn bernoulli_global_and_generator_paths_bit_match() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let probs = Tensor::new(vec![0.1, 0.5, 0.9, 0.5], &[4]).unwrap();
    manual_seed(41);
    let global = bernoulli(&probs).unwrap();

    let mut generator = Generator::new(41);
    let via_generator = generator.bernoulli(&probs).unwrap();

    for i in 0..4 {
        assert_eq!(global.get(&[i]), via_generator.get(&[i]));
    }
}

#[test]
fn multinomial_global_and_generator_paths_bit_match() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let weights = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
    manual_seed(42);
    let global = multinomial(&weights, 20, true).unwrap();

    let mut generator = Generator::new(42);
    let via_generator = generator.multinomial(&weights, 20, true).unwrap();

    for i in 0..20 {
        assert_eq!(global.get(&[i]), via_generator.get(&[i]));
    }
}

#[test]
fn normal_global_and_generator_paths_bit_match() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    manual_seed(43);
    let global = normal(2.0, 3.0, &[32]).unwrap();

    let mut generator = Generator::new(43);
    let via_generator = generator.normal(2.0, 3.0, &[32]).unwrap();

    for i in 0..32 {
        assert_eq!(global.get(&[i]), via_generator.get(&[i]));
    }
}

// ---------------------------------------------------------------------
// `normal(mean, std, shape)` と `nn::init::normal(shape, mean, std)` の
// bit 一致（`std > 0` に限る。`std == 0` は消費契約が意図的に異なる
// ——本関数は消費する・`nn::init::normal` は消費しない——ため対象外）。
// ---------------------------------------------------------------------

#[test]
fn normal_matches_nn_init_normal_bit_for_bit_when_std_positive() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    manual_seed(44);
    let a = normal(0.5, 1.5, &[10]).unwrap();
    manual_seed(44);
    let b = init::normal(&[10], 0.5, 1.5).unwrap();
    for i in 0..10 {
        assert_eq!(a.get(&[i]), b.get(&[i]));
    }
}

// ---------------------------------------------------------------------
// 2 つの Generator の相互不干渉
// ---------------------------------------------------------------------

#[test]
fn two_generators_produce_independent_sequences() {
    let mut a = Generator::new(101);
    let mut b = Generator::new(202);

    let weights = Tensor::new(vec![1.0f32, 1.0, 1.0], &[3]).unwrap();
    let a1 = a.multinomial(&weights, 5, true).unwrap();
    let _ = b.multinomial(&weights, 5, true).unwrap();
    let a2 = a.multinomial(&weights, 5, true).unwrap();

    // `b` を挟んでも `a` の系列は `b` の呼び出しに一切影響されない
    // （新規 same-seed の `a_reference` を 2 回連続で呼んだ結果と一致する
    // ことで間接的に確認する）。
    let mut a_reference = Generator::new(101);
    let a1_ref = a_reference.multinomial(&weights, 5, true).unwrap();
    let a2_ref = a_reference.multinomial(&weights, 5, true).unwrap();

    for i in 0..5 {
        assert_eq!(a1.get(&[i]), a1_ref.get(&[i]));
        assert_eq!(a2.get(&[i]), a2_ref.get(&[i]));
    }
}

// ---------------------------------------------------------------------
// エラー系（型のみの軽い回帰。詳細は tensor-core::rng の unit test 側）
// ---------------------------------------------------------------------

#[test]
fn multinomial_rejects_rank_other_than_1_or_2() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let weights = Tensor::new(vec![1.0f32; 8], &[2, 2, 2]).unwrap();
    let err = multinomial(&weights, 1, true).unwrap_err();
    assert!(matches!(
        err,
        RngError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: 3
        })
    ));
}
