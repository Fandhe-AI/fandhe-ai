//! `compat::functional::train`（Functional モデルの `bind`・パラメータ収集・`compile`・`fit`・
//! `evaluate`。イシュー #2667）のクレート内ユニットテスト。
//!
//! 構成: (1) 単一入力・単一ブロック・単一出力のグラフが同条件の `Sequential::fit`／`evaluate` と
//! bit 一致する回帰（本実装の正しさの裁定者。設計記録 §7 の要件）・(2) 拒否系と失敗後の不変条件・
//! (3) 多入力・結合・多出力グラフの学習・(4) パラメータ順序契約と `apply_parameters` の原子性・
//! (5) モード復元。PyTorch 2.14.0 実行値 fixture との照合は `fit_parity_tests.rs`。
//!
//! グローバル RNG（`manual_seed`）を消費する `shuffle(true)` の比較は、同プロセス内の他テストが RNG を
//! 消費して順序がずれる偽陽性を避けるため、`retry_on_global_rng_interference` で再試行する
//! （実装の不具合は全試行で不一致になるため検出力は落ちない）。
//!
//! テスト関数・ヘルパーの名前は `tests/api_surface.rs` の workspace 走査が数える名前
//! （`state_dict` 等）と衝突させない。

use super::{FunctionalBuilder, FunctionalModel};
use crate::compat::{FitConfig, History, Loss, Optimizer, Sequential};
use crate::optim::{
    AdagradConfig, AdamConfig, AdamWConfig, LambConfig, LbfgsConfig, RmsPropConfig, SgdConfig,
};
use crate::{AutodiffError, Tensor};

const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture tensor")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("fixture tensor")
}

/// 決定的な擬似データ（`sin` の位相ずらし。外部 RNG に依存しない）。
fn det_data(rows: usize, cols: usize, salt: f32) -> Tensor<f32> {
    let data = (0..rows * cols)
        .map(|k| ((k as f32) * 0.37 + salt).sin())
        .collect();
    t(data, &[rows, cols])
}

fn bits_of(x: &Tensor<f32>) -> Vec<u32> {
    x.host_slice().iter().map(|v| v.to_bits()).collect()
}

fn vec_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn param_bits(params: &[&Tensor<f32>]) -> Vec<Vec<u32>> {
    params.iter().map(|p| bits_of(p)).collect()
}

fn is_invalid<T>(r: &Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

fn build_seq() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, SEED_L1)
        .expect("linear")
        .add_relu()
        .add_linear(8, 2, SEED_L2)
        .expect("linear")
}

fn single_chain(block: Sequential) -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(block, x).expect("apply");
    b.build(&[x], &[y]).expect("build")
}

fn six_optimizers() -> Vec<(&'static str, Optimizer)> {
    vec![
        (
            "sgd_momentum",
            Optimizer::Sgd(SgdConfig::new(0.05).with_momentum(0.9)),
        ),
        (
            "adamw",
            Optimizer::AdamW(AdamWConfig {
                lr: 0.01,
                ..AdamWConfig::default()
            }),
        ),
        (
            "adam",
            Optimizer::Adam(AdamConfig {
                lr: 0.01,
                ..AdamConfig::default()
            }),
        ),
        (
            "rmsprop",
            Optimizer::RmsProp(RmsPropConfig {
                lr: 0.02,
                alpha: 0.9,
                eps: 1e-8,
                weight_decay: 0.01,
                momentum: 0.1,
                centered: true,
            }),
        ),
        (
            "adagrad",
            Optimizer::Adagrad(AdagradConfig {
                lr: 0.1,
                lr_decay: 0.01,
                weight_decay: 0.01,
                initial_accumulator_value: 0.0,
                eps: 1e-10,
            }),
        ),
        (
            "lamb",
            Optimizer::Lamb(LambConfig {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-6,
                weight_decay: 0.01,
            }),
        ),
    ]
}

/// `manual_seed` を使う比較が他テストの RNG 消費で崩れた場合に限り再試行する。`attempt` が `Ok(true)` なら
/// 一致、`Ok(false)` なら不一致（再試行）。全試行で不一致なら失敗（実装の不具合）。
fn retry_on_global_rng_interference(mut attempt: impl FnMut() -> bool) {
    for _ in 0..8 {
        if attempt() {
            return;
        }
    }
    panic!("8 回の再試行すべてで Sequential と Functional の shuffle 順が一致しなかった");
}

/// `Sequential::fit` と単一チェーン Functional の `fit`／`evaluate` を同条件で走らせ、
/// `History.loss`・`History.lr`・全パラメータ・`evaluate` が bit 一致するかを返す。
fn chain_matches_sequential<T>(
    optimizer: Optimizer,
    loss: Loss,
    config: FitConfig,
    x: &Tensor<f32>,
    y: &Tensor<T>,
    seed: Option<u64>,
) -> bool
where
    T: crate::compat::FitTarget,
{
    let mut seq = build_seq();
    seq.compile(optimizer, loss).expect("compile seq");
    let mut fm = single_chain(build_seq());
    fm.compile(optimizer, loss).expect("compile functional");

    if let Some(s) = seed {
        crate::manual_seed(s);
    }
    let h_seq = seq.fit(x, y, config).expect("seq fit");
    if let Some(s) = seed {
        crate::manual_seed(s);
    }
    let h_fn = fm.fit(&[x], &[y], config).expect("functional fit");

    let same_history = vec_bits(&h_seq.loss) == vec_bits(&h_fn.loss)
        && vec_bits(&h_seq.lr) == vec_bits(&h_fn.lr)
        && h_fn.val_loss.is_empty()
        && h_fn.val_metrics.is_empty();
    let same_params =
        param_bits(&seq.trainable_parameters()) == param_bits(&fm.trainable_parameters());
    let e_seq = seq.evaluate(x, y, 3).expect("seq evaluate");
    let e_fn = fm.evaluate(&[x], &[y], 3).expect("functional evaluate");
    same_history && same_params && e_seq.to_bits() == e_fn.to_bits()
}

// ------------------------------------------------ (1) Sequential との bit 同一回帰

#[test]
fn single_chain_fit_is_bit_identical_to_sequential_for_every_optimizer() {
    // 10 サンプル・バッチ 4 で端数バッチ（2 件）を含む。
    let x = det_data(10, 4, 0.1);
    let y = det_data(10, 2, 1.7);
    for (name, optimizer) in six_optimizers() {
        for drop_last in [false, true] {
            let config = FitConfig::new(3, 4).drop_last(drop_last);
            assert!(
                chain_matches_sequential(optimizer, Loss::Mse, config, &x, &y, None),
                "{name} drop_last={drop_last}: Sequential::fit と bit 一致しない"
            );
        }
    }
}

#[test]
fn single_chain_fit_with_shuffle_is_bit_identical_to_sequential() {
    let x = det_data(10, 4, 0.3);
    let y = det_data(10, 2, 0.9);
    for (name, optimizer) in six_optimizers() {
        retry_on_global_rng_interference(|| {
            let _ = name;
            chain_matches_sequential(
                optimizer,
                Loss::Mse,
                FitConfig::new(3, 4).shuffle(true),
                &x,
                &y,
                Some(42),
            )
        });
    }
}

#[test]
fn single_chain_cross_entropy_with_i32_target_is_bit_identical_to_sequential() {
    let x = det_data(9, 4, 0.2);
    let y = ti(vec![0, 1, 1, 0, 1, 0, 0, 1, 1], &[9]);
    let optimizer = Optimizer::Sgd(SgdConfig::new(0.1));
    assert!(chain_matches_sequential(
        optimizer,
        Loss::CrossEntropy,
        FitConfig::new(3, 4),
        &x,
        &y,
        None,
    ));
}

#[test]
fn fit_twice_equals_fit_once_with_double_epochs() {
    // optimizer 状態が fit 呼び出しをまたいで継続する契約（`Sequential` と同じ）。
    let x = det_data(8, 4, 0.5);
    let y = det_data(8, 2, 0.6);
    let optimizer = Optimizer::Adam(AdamConfig::default());
    let mut twice = single_chain(build_seq());
    twice.compile(optimizer, Loss::Mse).expect("compile");
    twice.fit(&[&x], &[&y], FitConfig::new(1, 4)).expect("fit");
    twice.fit(&[&x], &[&y], FitConfig::new(1, 4)).expect("fit");
    let mut once = single_chain(build_seq());
    once.compile(optimizer, Loss::Mse).expect("compile");
    once.fit(&[&x], &[&y], FitConfig::new(2, 4)).expect("fit");
    assert_eq!(
        param_bits(&twice.trainable_parameters()),
        param_bits(&once.trainable_parameters())
    );
}

#[test]
fn zero_parameter_graph_fit_matches_sequential_result_kind() {
    let x = det_data(6, 2, 0.4);
    let y = det_data(6, 2, 0.8);
    let optimizer = Optimizer::Sgd(SgdConfig::new(0.1));
    let mut seq = Sequential::new().add_relu();
    seq.compile(optimizer, Loss::Mse).expect("compile");
    let mut fm = single_chain(Sequential::new().add_relu());
    fm.compile(optimizer, Loss::Mse).expect("compile");
    let r_seq = seq.fit(&x, &y, FitConfig::new(2, 3));
    let r_fn = fm.fit(&[&x], &[&y], FitConfig::new(2, 3));
    match (r_seq, r_fn) {
        (Ok(a), Ok(b)) => assert_eq!(vec_bits(&a.loss), vec_bits(&b.loss)),
        (Err(a), Err(b)) => assert_eq!(
            std::mem::discriminant(&a),
            std::mem::discriminant(&b),
            "{a} / {b}"
        ),
        (a, b) => panic!(
            "結果種別が一致しない: seq={:?} functional={:?}",
            a.is_ok(),
            b.is_ok()
        ),
    }
}

// ------------------------------------------------------------------ (2) 拒否系

/// パラメータを持つ独自層（`bind` が追跡しないため学習経路は拒否する。#2398）。
struct ParametricCustom(Tensor<f32>);

impl crate::nn::Module for ParametricCustom {
    fn forward<'t>(
        &self,
        _tape: crate::TapeRef<'t>,
        input: &crate::Var<'t>,
    ) -> Result<crate::Var<'t>, AutodiffError> {
        Ok(*input)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("w".into(), &self.0)]
    }
}

/// 失敗後も compile 状態・モード・パラメータが不変であることの共通検査。
fn assert_untouched(model: &FunctionalModel, before: &[Vec<u32>], was_training: bool) {
    assert!(model.is_compiled(), "失敗後も compile 状態を保持する");
    assert_eq!(model.training(), was_training, "モードを復元する");
    assert_eq!(
        param_bits(&model.trainable_parameters()),
        before,
        "パラメータは不変"
    );
}

fn compiled_chain() -> FunctionalModel {
    let mut fm = single_chain(build_seq());
    fm.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .expect("compile");
    fm
}

#[test]
fn fit_and_evaluate_reject_uncompiled_model() {
    let x = det_data(4, 4, 0.1);
    let y = det_data(4, 2, 0.2);
    let mut fm = single_chain(build_seq());
    assert!(!fm.is_compiled());
    assert!(is_invalid(&fm.fit(&[&x], &[&y], FitConfig::new(1, 2))));
    assert!(is_invalid(&fm.evaluate(&[&x], &[&y], 2)));
    assert!(!fm.is_compiled());
}

#[test]
fn compile_rejects_lbfgs_and_keeps_previous_state() {
    let mut fm = compiled_chain();
    let r = fm.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse);
    assert!(is_invalid(&r));
    assert!(fm.is_compiled(), "拒否しても既存の compile 状態を保つ");
    let mut fresh = single_chain(build_seq());
    assert!(is_invalid(
        &fresh.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
    ));
    assert!(!fresh.is_compiled());
}

#[test]
fn fit_rejects_bad_arguments_without_touching_state() {
    let x = det_data(6, 4, 0.1);
    let y = det_data(6, 2, 0.2);
    let short_y = det_data(5, 2, 0.2);
    let wrong_loss_target = ti(vec![0, 1, 0, 1, 0, 1], &[6]);
    let mut fm = compiled_chain();
    fm.set_training(false);
    let before = param_bits(&fm.trainable_parameters());

    // 入力・目標の件数不一致。
    assert!(is_invalid(&fm.fit(&[&x, &x], &[&y], FitConfig::new(1, 2))));
    assert!(is_invalid(&fm.fit(&[&x], &[&y, &y], FitConfig::new(1, 2))));
    assert!(is_invalid(&fm.fit::<f32>(&[], &[], FitConfig::new(1, 2))));
    // epochs・accumulate_steps・batch_size。
    assert!(is_invalid(&fm.fit(&[&x], &[&y], FitConfig::new(0, 2))));
    assert!(is_invalid(&fm.fit(
        &[&x],
        &[&y],
        FitConfig::new(1, 2).accumulate_steps(0)
    )));
    assert!(is_invalid(&fm.fit(
        &[&x],
        &[&y],
        FitConfig::new(1, 2).accumulate_steps(2)
    )));
    assert!(is_invalid(&fm.fit(&[&x], &[&y], FitConfig::new(1, 0))));
    // サンプル数不一致。
    assert!(is_invalid(&fm.fit(
        &[&x],
        &[&short_y],
        FitConfig::new(1, 2)
    )));
    // loss と目標 dtype の不整合（Mse × i32）。
    assert!(is_invalid(&fm.fit(
        &[&x],
        &[&wrong_loss_target],
        FitConfig::new(1, 2)
    )));
    // drop_last で全バッチが落ちる。
    assert!(is_invalid(&fm.fit(
        &[&x],
        &[&y],
        FitConfig::new(1, 100).drop_last(true)
    )));
    assert_untouched(&fm, &before, false);

    // evaluate の拒否。
    assert!(is_invalid(&fm.evaluate(&[&x, &x], &[&y], 2)));
    assert!(is_invalid(&fm.evaluate(&[&x], &[&short_y], 2)));
    assert!(is_invalid(&fm.evaluate(&[&x], &[&y], 0)));
    assert_untouched(&fm, &before, false);
}

#[test]
fn fit_rejects_parametric_custom_layer_and_seq_first_mha() {
    use crate::compat::MultiheadAttentionConfig;
    let x = det_data(4, 4, 0.1);
    let y = det_data(4, 4, 0.2);

    // batch_first=false の MHA を含むブロック。
    let cfg = MultiheadAttentionConfig::new(4, 2).with_batch_first(false);
    let mha = Sequential::new()
        .add_multihead_attention_with_config(cfg, 1)
        .expect("mha");
    let mut fm = single_chain(mha);
    fm.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .expect("compile");
    assert!(is_invalid(&fm.fit(&[&x], &[&y], FitConfig::new(1, 2))));
    assert!(is_invalid(&fm.evaluate(&[&x], &[&y], 2)));
    assert!(fm.is_compiled());

    // パラメータ持ちの `add_module` 層。
    let custom = Sequential::new().add_module(ParametricCustom(t(vec![1.0], &[1])));
    let mut fm = single_chain(custom);
    fm.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .expect("compile");
    assert!(is_invalid(&fm.fit(&[&x], &[&y], FitConfig::new(1, 2))));
    assert!(fm.is_compiled());
}

// ----------------------------------------------- (3) 多入力・結合・多出力グラフ

/// 入力 2・結合 2 種（add／concatenate）・出力 2 のグラフ。パラメータは 8 個。
fn two_in_two_out() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let a = b.input().expect("input a");
    let c = b.input().expect("input b");
    let ba = b
        .apply(
            Sequential::new()
                .add_linear(3, 4, 11)
                .expect("linear")
                .add_tanh(),
            a,
        )
        .expect("apply");
    let bb = b
        .apply(
            Sequential::new()
                .add_linear(2, 4, 12)
                .expect("linear")
                .add_relu(),
            c,
        )
        .expect("apply");
    let sum = b.add(&[ba, bb]).expect("add");
    let cat = b.concatenate(&[ba, bb], 1).expect("concat");
    let o1 = b
        .apply(Sequential::new().add_linear(4, 2, 13).expect("linear"), sum)
        .expect("apply");
    let o2 = b
        .apply(Sequential::new().add_linear(8, 1, 14).expect("linear"), cat)
        .expect("apply");
    b.build(&[a, c], &[o1, o2]).expect("build")
}

#[test]
fn multi_input_multi_output_graph_learns() {
    let xa = det_data(12, 3, 0.1);
    let xb = det_data(12, 2, 0.7);
    let y1 = det_data(12, 2, 1.3);
    let y2 = det_data(12, 1, 1.9);
    let mut fm = two_in_two_out();
    assert_eq!(fm.trainable_parameters().len(), 8);
    fm.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("compile");
    let h = fm
        .fit(&[&xa, &xb], &[&y1, &y2], FitConfig::new(40, 4))
        .expect("fit");
    assert_eq!(h.loss.len(), 40);
    assert_eq!(h.lr.len(), 40);
    assert!(
        h.loss[39] < h.loss[0],
        "損失が減る: {} -> {}",
        h.loss[0],
        h.loss[39]
    );
    assert!(h.loss.iter().all(|v| v.is_finite()));
    let e = fm.evaluate(&[&xa, &xb], &[&y1, &y2], 5).expect("evaluate");
    assert!(e.is_finite() && e < h.loss[0]);
}

#[test]
fn multi_output_loss_is_the_sum_of_per_output_losses() {
    let xa = det_data(6, 3, 0.2);
    let xb = det_data(6, 2, 0.4);
    let y1 = det_data(6, 2, 1.1);
    let y2 = det_data(6, 1, 1.5);
    let mut fm = two_in_two_out();
    fm.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("compile");
    fm.eval();
    let outs = fm.predict(&[&xa, &xb]).expect("predict");
    let mse = |pred: &Tensor<f32>, target: &Tensor<f32>| -> f64 {
        let p = pred.host_slice();
        let q = target.host_slice();
        let n = p.len() as f64;
        p.iter()
            .zip(q.iter())
            .map(|(a, b)| {
                let d = f64::from(*a) - f64::from(*b);
                d * d
            })
            .sum::<f64>()
            / n
    };
    let expected = mse(&outs[0], &y1) + mse(&outs[1], &y2);
    let got = f64::from(fm.evaluate(&[&xa, &xb], &[&y1, &y2], 6).expect("evaluate"));
    assert!(
        (got - expected).abs() <= 1e-5 * expected.abs().max(1.0),
        "evaluate {got} が出力別損失の和 {expected} と一致しない"
    );
}

#[test]
fn fan_out_gradients_merge_and_match_manual_step() {
    // 1 ノード（入力）を 2 ブロックが消費する fan-out。fit 1 ステップの更新が、`bind` 経路で得た
    // 勾配を手で `Sgd` へ渡した結果と bit 一致する（勾配の合流・位置対応の裏取り）。
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let p = b
        .apply(Sequential::new().add_linear(3, 2, 21).expect("linear"), x)
        .expect("apply");
    let q = b
        .apply(Sequential::new().add_linear(3, 2, 22).expect("linear"), x)
        .expect("apply");
    let s = b.add(&[p, q]).expect("add");
    let mut fm = b.build(&[x], &[s]).expect("build");
    let xin = det_data(4, 3, 0.3);
    let target = det_data(4, 2, 0.9);

    // 手計算側: 同じ初期値のモデルで bind → forward → backward → Sgd。
    let mut manual = {
        let mut b = FunctionalBuilder::new();
        let x = b.input().expect("input");
        let p = b
            .apply(Sequential::new().add_linear(3, 2, 21).expect("linear"), x)
            .expect("apply");
        let q = b
            .apply(Sequential::new().add_linear(3, 2, 22).expect("linear"), x)
            .expect("apply");
        let s = b.add(&[p, q]).expect("add");
        b.build(&[x], &[s]).expect("build")
    };
    let updated = {
        let tape = crate::tape();
        let bound = manual.bind(&tape);
        let xv = tape.var(&xin);
        let outs = bound.forward(&tape, &[xv]).expect("forward");
        let tv = tape.var(&target);
        let loss = outs[0].mse_loss(&tv).expect("loss");
        let grads = tape.backward(&loss).expect("backward");
        let grad_refs = bound.trainable_grads(&grads).expect("grads");
        assert_eq!(grad_refs.len(), bound.trainable_vars().len());
        let mut sgd = crate::optim::Sgd::new(SgdConfig::new(0.1)).expect("sgd");
        let params = manual.trainable_parameters();
        sgd.step(&params, &grad_refs).expect("step")
    };
    manual.apply_parameters(updated).expect("apply_parameters");

    fm.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .expect("compile");
    fm.fit(&[&xin], &[&target], FitConfig::new(1, 4))
        .expect("fit");
    assert_eq!(
        param_bits(&fm.trainable_parameters()),
        param_bits(&manual.trainable_parameters())
    );
}

// --------------------------------------------- (4) 順序契約と apply_parameters

#[test]
fn trainable_parameters_follow_named_parameters_order_and_ignore_merge_nodes() {
    let fm = two_in_two_out();
    let named = fm.named_parameters().expect("named");
    let params = fm.trainable_parameters();
    assert_eq!(named.len(), params.len());
    for ((_, a), b) in named.iter().zip(&params) {
        assert_eq!(bits_of(a), bits_of(b));
    }
    // 結合ノードは層を持たないためパラメータ列に現れない（4 ブロック × weight・bias）。
    assert_eq!(params.len(), 8);
}

#[test]
fn bind_forward_matches_plain_forward_bit_for_bit() {
    let fm = two_in_two_out();
    let xa = det_data(5, 3, 0.15);
    let xb = det_data(5, 2, 0.55);
    let plain = fm.predict(&[&xa, &xb]).expect("predict");
    let tape = crate::tape();
    let bound = fm.bind(&tape);
    let vars = [tape.var(&xa), tape.var(&xb)];
    let outs = bound.forward(&tape, &vars).expect("forward");
    assert_eq!(outs.len(), plain.len());
    for (o, p) in outs.iter().zip(&plain) {
        assert_eq!(bits_of(&o.to_tensor()), bits_of(p));
    }
    // 入力件数の不一致は型付きエラー。
    assert!(is_invalid(&bound.forward(&tape, &vars[..1])));
}

#[test]
fn apply_parameters_is_atomic_on_count_or_shape_mismatch() {
    let mut fm = two_in_two_out();
    let before = param_bits(&fm.trainable_parameters());
    let mut good: Vec<Tensor<f32>> = fm.trainable_parameters().into_iter().cloned().collect();

    // 件数不足・過剰。
    let mut fewer = good.clone();
    fewer.pop();
    assert!(is_invalid(&fm.apply_parameters(fewer)));
    let mut more = good.clone();
    more.push(t(vec![0.0], &[1]));
    assert!(is_invalid(&fm.apply_parameters(more)));
    assert_eq!(param_bits(&fm.trainable_parameters()), before);

    // 後ろのブロックの shape 不一致でも、先頭ブロックは書き換わらない。
    if let Some(last) = good.last_mut() {
        *last = t(vec![0.0; 3], &[3]);
    }
    for p in good.iter_mut().take(1) {
        let shape = p.shape().to_vec();
        *p = t(vec![9.0; p.numel()], &shape);
    }
    assert!(is_invalid(&fm.apply_parameters(good)));
    assert_eq!(param_bits(&fm.trainable_parameters()), before);
}

#[test]
fn apply_parameters_updates_every_block_in_order() {
    let mut fm = two_in_two_out();
    let updated: Vec<Tensor<f32>> = fm
        .trainable_parameters()
        .iter()
        .enumerate()
        .map(|(i, p)| t(vec![i as f32 + 0.5; p.numel()], p.shape()))
        .collect();
    fm.apply_parameters(updated.clone()).expect("apply");
    for (got, want) in fm.trainable_parameters().iter().zip(&updated) {
        assert_eq!(bits_of(got), bits_of(want));
    }
}

// ----------------------------------------------------------------- (5) モード

#[test]
fn fit_restores_mode_and_trains_dropout_and_batch_norm_blocks() {
    let x = det_data(8, 4, 0.2);
    let y = det_data(8, 2, 0.6);
    let block = Sequential::new()
        .add_linear(4, 4, 31)
        .expect("linear")
        .add_batch_norm1d(4, 1e-5, 0.1)
        .expect("bn")
        .add_dropout(0.3)
        .expect("dropout")
        .add_linear(4, 2, 32)
        .expect("linear");
    let mut fm = single_chain(block);
    fm.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("compile");

    // eval で呼べば eval へ戻り、train で呼べば train のまま。
    fm.eval();
    fm.fit(&[&x], &[&y], FitConfig::new(2, 4)).expect("fit");
    assert!(!fm.training());
    // fit は train モードで走るため BatchNorm の running stats が初期値から動いている。
    let moved = fm.blocks().any(|(_, b)| {
        b.layers().iter().any(|l| {
            l.as_batch_norm1d()
                .is_some_and(|bn| bn.running_mean().host_slice().iter().any(|v| *v != 0.0))
        })
    });
    assert!(moved, "fit は train モードで BatchNorm を更新する");

    fm.train();
    fm.fit(&[&x], &[&y], FitConfig::new(1, 4)).expect("fit");
    assert!(fm.training());

    // evaluate は eval モードで決定的（Dropout が無効）。モードは呼び出し前へ戻る。
    let a = fm.evaluate(&[&x], &[&y], 4).expect("evaluate");
    let b = fm.evaluate(&[&x], &[&y], 4).expect("evaluate");
    assert_eq!(a.to_bits(), b.to_bits());
    assert!(fm.training());
}

#[test]
fn history_fields_are_filled_as_documented() {
    let x = det_data(6, 4, 0.1);
    let y = det_data(6, 2, 0.2);
    let mut fm = compiled_chain();
    let h: History = fm.fit(&[&x], &[&y], FitConfig::new(3, 2)).expect("fit");
    assert_eq!(h.loss.len(), 3);
    assert_eq!(h.lr.len(), 3);
    assert!(h.val_loss.is_empty() && h.val_metrics.is_empty());
    assert!(h.lr.iter().all(|v| (*v - 0.1).abs() < 1e-7));
}
