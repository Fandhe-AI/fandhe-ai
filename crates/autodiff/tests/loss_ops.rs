//! `crate::loss_ops`（L1 損失・label_smoothing/ignore_index/
//! class_weight 付き CrossEntropy 損失）の受け入れ条件検証（イシュー
//! #2166・親イシュー #2131「PyTorch／TF 置き換えの API 網羅」）。
//!
//! - L1 損失: forward 解析値・`n == 0`・`NaN` 伝播・backward（`sign(0)
//!   = 0`・`dTarget = −dPred`・kink を避けた点での数値微分突合）・
//!   エラー経路（shape 不一致・クロステープ）。
//! - CrossEntropy（オプション付き）: 既定オプションが既存
//!   `Var::cross_entropy_loss` と forward／backward とも bit 完全一致
//!   すること（R3 の機械的検証）・label_smoothing／ignore_index／
//!   class_weight 単体および組み合わせでの数値微分突合・検査の拒否
//!   ケース。
//! - `nn::loss::L1Loss`／`CrossEntropyLoss::forward_with` が自由関数
//!   直接呼び出しと bit 一致する薄いラッパーであること。
//!
//! イシュー #2167（親 #2131）で距離ベースの損失 3 種
//! （`cosine_embedding_loss`・`margin_ranking_loss`・
//! `triplet_margin_loss`）と `poisson_nll_loss` の受け入れ条件検証を
//! 追加した（§9〜§12）。各損失: forward 解析値・`n == 0`・エラー経路
//! （shape 不一致・クロステープ・`y` の `±1` 制約・オプション値検査）・
//! kink を避けた点での数値微分突合。
//!
//! イシュー #2168（親 #2131）で CTC 損失（`ctc_loss`）の受け入れ条件
//! 検証を追加した（§13）: 総当たり整列との forward 一致・reduction・
//! 数値微分突合・フレーム外勾配 0・整列不能（`zero_infinity` 有無）・
//! `NaN` 伝播・空バッチ・target 形式（パディング／連結）の等価性・
//! 入力検査の拒否ケース。
//!
//! 判定基準（backward）: 承認済み複合判定「相対誤差 1e-2 または絶対
//! 誤差 1e-3」＋`τ=1e-4`（`crates/autodiff/tests/nn_cross_entropy.rs`・
//! `nn_loss.rs` と同一パラメータを再利用。新規閾値は導入しない）。

mod common;

use fandhe_ai_autodiff::loss_ops::{
    self, CrossEntropyOptions, CtcLossOptions, PoissonNllOptions, TripletMarginOptions,
};
use fandhe_ai_autodiff::nn::loss::{
    CosineEmbeddingLoss, CrossEntropyLoss, L1Loss, MarginRankingLoss, PoissonNllLoss,
    TripletMarginLoss,
};
use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn f32_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn i32_tensor(data: &[i32], shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous() 後の as_slice() は Some のはず")
        .to_vec()
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    dense(tensor)[0]
}

// 承認済み複合判定（`nn_cross_entropy.rs`／`nn_loss.rs` と同一）。
const H: f64 = 1e-3;
const GC_TAU: f32 = 1e-4;
const GC_REL_TOL: f32 = 1e-2;
const GC_ABS_TOL: f32 = 1e-3;

fn assert_close(label: &str, a: f32, n: f32) {
    let diff = (a - n).abs();
    let rel = diff / a.abs().max(n.abs()).max(GC_TAU);
    assert!(
        rel <= GC_REL_TOL || diff <= GC_ABS_TOL,
        "{label}: analytic={a} numeric={n} diff={diff} rel={rel}"
    );
}

// =====================================================================
// 1. L1 損失
// =====================================================================

// diff = pred − target = [0.5, -1.0, 0.5, -0.5] → |diff| = [0.5, 1.0, 0.5, 0.5]
// sum = 2.5、mean = 2.5 / 4 = 0.625
#[test]
fn l1_loss_forward_mean_and_sum_match_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&f32_tensor(&[0.5, -1.0, 2.5, 1.0], &[2, 2]));

    let mean = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap();
    assert!((scalar(&mean.to_tensor()) - 0.625).abs() < 1e-6);

    let sum = loss_ops::l1_loss(&pred, &target, Reduction::Sum).unwrap();
    assert!((scalar(&sum.to_tensor()) - 2.5).abs() < 1e-6);
}

// `pred=[f32::MAX, f32::MAX]`・`target=[0, 0]` は `|diff|` の和が
// `f32` の範囲を超えて `inf` になるが、平均は `f32::MAX` に収まる
// 有限値（codex-review 指摘・イシュー #2166 PR #2283。`f64` の和を
// `f64` のまま `numel` で割ってから 1 回だけ `f32` へ downcast する
// 契約の回帰テスト）。
#[test]
fn l1_loss_mean_divides_in_f64_before_downcast_to_avoid_overflow() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[f32::MAX, f32::MAX], &[2]));
    let target = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));

    let mean = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap();
    let got = scalar(&mean.to_tensor());
    assert!(got.is_finite(), "mean は有限値であるべき: {got}");
    assert!((got - f32::MAX).abs() < 1e-3 * f32::MAX, "got={got}");

    // Sum は仕様どおり overflow して inf のままである（Mean のみが
    // 除算順序の修正対象）。
    let sum = loss_ops::l1_loss(&pred, &target, Reduction::Sum).unwrap();
    assert!(scalar(&sum.to_tensor()).is_infinite());
}

#[test]
fn l1_loss_empty_numel_returns_zero_for_mean_and_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[], &[0]));
    let target = tape.var(&f32_tensor(&[], &[0]));

    let mean = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap();
    assert_eq!(scalar(&mean.to_tensor()), 0.0);
    let sum = loss_ops::l1_loss(&pred, &target, Reduction::Sum).unwrap();
    assert_eq!(scalar(&sum.to_tensor()), 0.0);
}

#[test]
fn l1_loss_propagates_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[f32::NAN, 1.0], &[2]));
    let target = tape.var(&f32_tensor(&[0.0, 1.0], &[2]));

    let loss = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap();
    assert!(scalar(&loss.to_tensor()).is_nan());
}

#[test]
fn l1_loss_shape_mismatch_is_err() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[1.0, 2.0], &[2]));
    let target = tape.var(&f32_tensor(&[1.0, 2.0, 3.0], &[3]));

    let err = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn l1_loss_cross_tape_is_err() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let pred = tape_a.var(&f32_tensor(&[1.0, 2.0], &[2]));
    let target = tape_b.var(&f32_tensor(&[1.0, 2.0], &[2]));

    let err = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn l1_loss_backward_sign_and_target_symmetry() {
    // d = pred - target = [2.0, -3.0, 0.0, 0.5]（3 番目の要素が kink
    // `d == 0` に厳密一致するケースを含む）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[2.0, -1.0, 5.0, 1.5], &[4]));
    let target = tape.var(&f32_tensor(&[0.0, 2.0, 5.0, 1.0], &[4]));

    let loss = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dpred = dense(grads.get(&pred).unwrap().expect("pred は loss に到達する"));
    let dtarget = dense(
        grads
            .get(&target)
            .unwrap()
            .expect("target は loss に到達する"),
    );

    // scale = 1/4（Mean）。sign = [1, -1, 0, 1]。
    assert_eq!(dpred, vec![0.25, -0.25, 0.0, 0.25]);
    // dTarget = -dPred。
    for (dp, dt) in dpred.iter().zip(dtarget.iter()) {
        assert!((dp + dt).abs() < 1e-7);
    }
}

#[test]
fn l1_loss_grad_matches_numeric_central_difference_away_from_kink() {
    // kink（|d| == 0）を避けた fixture。
    let pred_data = vec![2.0f32, -1.0, 5.0, 1.5];
    let target_data = vec![0.3f32, 2.7, 4.1, 1.0];
    let target = f32_tensor(&target_data, &[4]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let pred_var = tape.var(&f32_tensor(&pred_data, &[4]));
    let target_var = tape.var(&target);
    let loss = loss_ops::l1_loss(&pred_var, &target_var, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(
        grads
            .get(&pred_var)
            .unwrap()
            .expect("pred は loss に到達する"),
    );

    let eval_loss = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let p = t.var(&f32_tensor(data, &[4]));
        let tg = t.var(&target);
        let l = loss_ops::l1_loss(&p, &tg, Reduction::Mean).unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let mut data = pred_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("l1_loss_grad[{i}]"), analytic[i], numeric);
    }
}

// =====================================================================
// 2. CrossEntropy（オプション付き）: 既定オプションは既存経路へ丸ごと
//    委譲する（R3）ことの bit 完全一致検証
// =====================================================================

#[test]
fn cross_entropy_loss_with_default_options_matches_existing_method_bit_exact() {
    let logits_data = [
        1.0f32, 2.0, 0.5, 0.1, -0.5, 2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
    ];
    let targets = i32_tensor(&[1, 2, 0, 1], &[4]);

    for reduction in [Reduction::Mean, Reduction::Sum] {
        let tape_a = Tape::new_with_ops(common::naive_ops());
        let x_a = tape_a.var(&f32_tensor(&logits_data, &[4, 3]));
        let via_existing = x_a.cross_entropy_loss(&targets, 1, reduction).unwrap();
        let grads_a = tape_a.backward(&via_existing).unwrap();
        let grad_a = dense(grads_a.get(&x_a).unwrap().expect("到達する"));

        let tape_b = Tape::new_with_ops(common::naive_ops());
        let x_b = tape_b.var(&f32_tensor(&logits_data, &[4, 3]));
        let via_with = loss_ops::cross_entropy_loss_with(
            &x_b,
            &targets,
            1,
            reduction,
            &CrossEntropyOptions::default(),
        )
        .unwrap();
        let grads_b = tape_b.backward(&via_with).unwrap();
        let grad_b = dense(grads_b.get(&x_b).unwrap().expect("到達する"));

        assert_eq!(
            dense(&via_existing.to_tensor()),
            dense(&via_with.to_tensor()),
            "{reduction:?}: forward が bit 完全一致しない"
        );
        assert_eq!(
            grad_a, grad_b,
            "{reduction:?}: backward が bit 完全一致しない"
        );
    }
}

// =====================================================================
// 3. label_smoothing
// =====================================================================

#[test]
fn cross_entropy_label_smoothing_forward_matches_reference_and_grad_matches_numeric() {
    let logits_data = vec![1.0f32, 2.0, 0.5, -1.0, 0.5, 2.0];
    let targets = i32_tensor(&[2, 0], &[2]);
    let eps = 0.1f32;
    let options = CrossEntropyOptions::default().label_smoothing(eps);

    // 独立参照実装（f64。`w_c = 1` 固定）: L_s = (1-eps)*(-lp_t) +
    // (eps/C)*sum_c(-lp_c)。
    let reference = |data: &[f32]| -> f64 {
        let axis_len = 3usize;
        let target_data = [2i32, 0];
        let mut total = 0.0f64;
        for (o, &t) in target_data.iter().enumerate() {
            let row = &data[o * axis_len..(o + 1) * axis_len];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum_exp: f64 = row.iter().map(|&v| ((v as f64) - m).exp()).sum();
            let lse = m + sum_exp.ln();
            let lp_t = row[t as usize] as f64 - lse;
            let sum_neg_lp: f64 = row.iter().map(|&v| -(v as f64 - lse)).sum();
            let loss_s = (1.0 - eps as f64) * (-lp_t) + (eps as f64 / axis_len as f64) * sum_neg_lp;
            total += loss_s;
        }
        total / target_data.len() as f64
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&logits_data, &[2, 3]));
    let loss =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
    assert_close(
        "label_smoothing forward",
        scalar(&loss.to_tensor()),
        reference(&logits_data) as f32,
    );

    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x).unwrap().expect("到達する"));
    let mut data = logits_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = reference(&data);
        data[i] = (orig - H) as f32;
        let lm = reference(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("label_smoothing_grad[{i}]"), analytic[i], numeric);
    }
}

#[test]
fn cross_entropy_label_smoothing_epsilon_one_endpoint_is_finite() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5], &[1, 3]));
    let targets = i32_tensor(&[1], &[1]);
    let options = CrossEntropyOptions::default().label_smoothing(1.0);
    let loss =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
    assert!(scalar(&loss.to_tensor()).is_finite());
}

// =====================================================================
// 4. ignore_index
// =====================================================================

#[test]
fn cross_entropy_ignore_index_excludes_sample_from_loss_and_grad() {
    let logits_data = vec![1.0f32, 2.0, 0.5, -1.0, 0.5, 2.0, 0.3, -0.2, 1.1];
    let targets = i32_tensor(&[2, 0, 1], &[3]);
    let options = CrossEntropyOptions::default().ignore_index(0);

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&logits_data, &[3, 3]));
    let loss =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let grad = dense(grads.get(&x).unwrap().expect("到達する"));

    // サンプル 1（target == ignore_index 0）の行は勾配 0。
    for v in &grad[3..6] {
        assert_eq!(*v, 0.0);
    }

    // サンプル 0・2（ignore されない）のみの既存 CE と近似一致する。
    let subset_logits: Vec<f32> = logits_data[0..3]
        .iter()
        .chain(logits_data[6..9].iter())
        .cloned()
        .collect();
    let subset_targets = i32_tensor(&[2, 1], &[2]);
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&f32_tensor(&subset_logits, &[2, 3]));
    let loss2 = x2
        .cross_entropy_loss(&subset_targets, 1, Reduction::Mean)
        .unwrap();
    assert_close(
        "ignore_index loss vs subset CE",
        scalar(&loss.to_tensor()),
        scalar(&loss2.to_tensor()),
    );
}

#[test]
fn cross_entropy_all_samples_ignored_returns_zero_loss_and_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.5, 2.0], &[2, 3]));
    let targets = i32_tensor(&[0, 0], &[2]);
    let options = CrossEntropyOptions::default().ignore_index(0);
    let loss =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
    assert_eq!(scalar(&loss.to_tensor()), 0.0);
    let grads = tape.backward(&loss).unwrap();
    let grad = dense(grads.get(&x).unwrap().expect("到達する"));
    assert!(grad.iter().all(|&v| v == 0.0));
}

// =====================================================================
// 5. class_weight
// =====================================================================

#[test]
fn cross_entropy_class_weight_all_ones_matches_existing_ce_via_composite_judgment() {
    let logits_data = [
        1.0f32, 2.0, 0.5, 0.1, -0.5, 2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
    ];
    let targets = i32_tensor(&[1, 2, 0, 1], &[4]);
    let options = CrossEntropyOptions::default().class_weight(f32_tensor(&[1.0, 1.0, 1.0], &[3]));

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&logits_data, &[4, 3]));
    let via_weighted =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&f32_tensor(&logits_data, &[4, 3]));
    let via_existing = x2.cross_entropy_loss(&targets, 1, Reduction::Mean).unwrap();

    assert_close(
        "class_weight all-ones vs existing CE",
        scalar(&via_weighted.to_tensor()),
        scalar(&via_existing.to_tensor()),
    );
}

#[test]
fn cross_entropy_class_weight_weighted_mean_denominator_matches_reference() {
    let logits_data = vec![1.0f32, 2.0, 0.5, -1.0, 0.5, 2.0];
    let targets = i32_tensor(&[0, 1], &[2]);
    let weight_data = [2.0f32, 0.5, 1.0];
    let options = CrossEntropyOptions::default().class_weight(f32_tensor(&weight_data, &[3]));

    let reference = |data: &[f32]| -> f64 {
        let axis_len = 3usize;
        let target_data = [0i32, 1];
        let mut total = 0.0f64;
        let mut denom = 0.0f64;
        for (o, &t) in target_data.iter().enumerate() {
            let row = &data[o * axis_len..(o + 1) * axis_len];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum_exp: f64 = row.iter().map(|&v| ((v as f64) - m).exp()).sum();
            let lse = m + sum_exp.ln();
            let lp_t = row[t as usize] as f64 - lse;
            let w_t = weight_data[t as usize] as f64;
            total += w_t * (-lp_t);
            denom += w_t;
        }
        total / denom
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&logits_data, &[2, 3]));
    let loss =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
    assert_close(
        "class_weight mean denominator",
        scalar(&loss.to_tensor()),
        reference(&logits_data) as f32,
    );

    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x).unwrap().expect("到達する"));
    let mut data = logits_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = reference(&data);
        data[i] = (orig - H) as f32;
        let lm = reference(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("class_weight_grad[{i}]"), analytic[i], numeric);
    }
}

// =====================================================================
// 6. 3 オプション組み合わせ: 数値微分突合
// =====================================================================

#[test]
fn cross_entropy_combined_options_grad_matches_numeric_central_difference() {
    let logits_data = vec![
        1.0f32, 2.0, 0.5, -1.0, 0.5, 2.0, 0.3, -0.2, 1.1, 2.0, -1.0, 0.0, 0.4, -0.6, 1.2, 0.8,
    ];
    let targets = i32_tensor(&[2, 0, 1, 3], &[4]);
    let options = CrossEntropyOptions::default()
        .label_smoothing(0.2)
        .ignore_index(3)
        .class_weight(f32_tensor(&[1.5, 0.5, 1.0, 2.0], &[4]));

    let eval_loss = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let x = t.var(&f32_tensor(data, &[4, 4]));
        let l =
            loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
        dense(&l.to_tensor())[0] as f64
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&logits_data, &[4, 4]));
    let loss =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x).unwrap().expect("到達する"));

    let mut data = logits_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("combined_grad[{i}]"), analytic[i], numeric);
    }
}

// =====================================================================
// 7. 検査の拒否ケース
// =====================================================================

#[test]
fn cross_entropy_label_smoothing_out_of_range_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5], &[1, 3]));
    let targets = i32_tensor(&[0], &[1]);

    for eps in [f32::NAN, -0.1, 1.1] {
        let options = CrossEntropyOptions::default().label_smoothing(eps);
        let err = loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options)
            .unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidArgument(_)),
            "eps={eps}: {err:?}"
        );
    }
}

#[test]
fn cross_entropy_class_weight_shape_mismatch_is_err() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5], &[1, 3]));
    let targets = i32_tensor(&[0], &[1]);
    let options = CrossEntropyOptions::default().class_weight(f32_tensor(&[1.0, 1.0], &[2]));

    let err =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn cross_entropy_class_weight_negative_or_nan_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5], &[1, 3]));
    let targets = i32_tensor(&[0], &[1]);

    for bad in [[-1.0f32, 1.0, 1.0], [f32::NAN, 1.0, 1.0]] {
        let options = CrossEntropyOptions::default().class_weight(f32_tensor(&bad, &[3]));
        let err = loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
    }
}

#[test]
fn cross_entropy_target_out_of_range_not_matching_ignore_index_is_err() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5], &[1, 3]));
    // target 5 は範囲外（[0, 3)）で ignore_index（-1）とも一致しない。
    let targets = i32_tensor(&[5], &[1]);
    let options = CrossEntropyOptions::default().ignore_index(-1);

    let err =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// =====================================================================
// 9. CosineEmbedding 損失（イシュー #2167）
// =====================================================================

// x1=[1,0]・x2=[0,1] は直交（cos=0）。y=1 → loss=1-0=1。
#[test]
fn cosine_embedding_loss_forward_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[0.0, 1.0], &[2]));
    let y = f32_tensor(&[1.0], &[]);

    let loss = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.5, Reduction::Mean).unwrap();
    assert!((scalar(&loss.to_tensor()) - 1.0).abs() < 1e-5);
}

// y=-1・cos=0 < margin(0.5) → hinge 無効 → loss=0。
#[test]
fn cosine_embedding_loss_negative_label_below_margin_is_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[0.0, 1.0], &[2]));
    let y = f32_tensor(&[-1.0], &[]);

    let loss = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.5, Reduction::Mean).unwrap();
    assert!(scalar(&loss.to_tensor()).abs() < 1e-6);
}

// 同方向の極端な大きさのベクトル（`f32::MAX`）は cos ≈ 1 のはずで、
// `y == 1` の損失（`1 - cos`）は 0 に近い有限値になるべき
// （codex-review 指摘・PR #2286。`(m1 * m2).sqrt()` の中間積 overflow
// 経路の回帰）。
#[test]
fn cosine_embedding_loss_extreme_magnitude_same_direction_stays_finite() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[f32::MAX, f32::MAX], &[2]));
    let x2 = tape.var(&f32_tensor(&[f32::MAX, f32::MAX], &[2]));
    let y = f32_tensor(&[1.0], &[]);

    let loss = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap();
    let got = scalar(&loss.to_tensor());
    assert!(got.is_finite(), "got={got}");
    assert!(
        got.abs() < 1e-3,
        "cos は約 1 のはずで loss は約 0: got={got}"
    );
}

// `NaN` を含む入力（`y == -1` の hinge 分岐）は損失も `NaN` を伝播
// すべきで、`f64::max(0.0)` の NaN 消失により黙って 0 に潰れては
// ならない（codex-review 指摘・PR #2286）。
#[test]
fn cosine_embedding_loss_negative_label_nan_input_propagates_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[f32::NAN, 0.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let y = f32_tensor(&[-1.0], &[]);

    let loss = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.5, Reduction::Mean).unwrap();
    assert!(scalar(&loss.to_tensor()).is_nan());
}

#[test]
fn cosine_embedding_loss_empty_batch_is_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[], &[0, 3]));
    let x2 = tape.var(&f32_tensor(&[], &[0, 3]));
    let y = f32_tensor(&[], &[0]);

    let mean = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap();
    assert_eq!(scalar(&mean.to_tensor()), 0.0);
    let sum = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Sum).unwrap();
    assert_eq!(scalar(&sum.to_tensor()), 0.0);
}

#[test]
fn cosine_embedding_loss_rejects_y_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let x2 = tape.var(&f32_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let y = f32_tensor(&[1.0], &[]); // rank 2 入力には [N] が必要

    let err = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn cosine_embedding_loss_rejects_non_pm_one_label() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[0.0, 1.0], &[2]));
    let y = f32_tensor(&[0.5], &[]);

    let err = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn cosine_embedding_loss_rejects_cross_tape() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x1 = tape_a.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let x2 = tape_b.var(&f32_tensor(&[0.0, 1.0], &[2]));
    let y = f32_tensor(&[1.0], &[]);

    let err = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn cosine_embedding_loss_grad_matches_numeric_central_difference() {
    // kink（cos == margin）から離れた fixture、rank 2（N=2, D=3）。
    let x1_data = vec![1.0f32, 2.0, -0.5, 0.3, -1.2, 0.7];
    let x2_data = vec![0.4f32, -0.6, 1.1, -0.9, 0.2, 1.5];
    let x2 = f32_tensor(&x2_data, &[2, 3]);
    let y = f32_tensor(&[1.0, -1.0], &[2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let x1_var = tape.var(&f32_tensor(&x1_data, &[2, 3]));
    let x2_var = tape.var(&x2);
    let loss = loss_ops::cosine_embedding_loss(&x1_var, &x2_var, &y, 0.2, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x1_var).unwrap().expect("x1 は loss に到達する"));

    let eval_loss = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let a = t.var(&f32_tensor(data, &[2, 3]));
        let b = t.var(&x2);
        let l = loss_ops::cosine_embedding_loss(&a, &b, &y, 0.2, Reduction::Mean).unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let mut data = x1_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("cosine_embedding_grad[{i}]"), analytic[i], numeric);
    }
}

// =====================================================================
// 10. MarginRanking 損失（イシュー #2167）
// =====================================================================

// raw = -y*(x1-x2)+margin = -1*(2-1)+0.5 = -0.5 → hinge 無効 → 0。
// raw = -(-1)*(1-3)+0.5 = -1.5 → hinge 無効(<0) → 0。両方 0 で合計 0。
#[test]
fn margin_ranking_loss_forward_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[2.0, 1.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[1.0, 3.0], &[2]));
    let y = f32_tensor(&[1.0, -1.0], &[2]);

    let loss = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.5, Reduction::Sum).unwrap();
    assert!(scalar(&loss.to_tensor()).abs() < 1e-6);
}

// `NaN` を含む入力は `raw` を `NaN` にし、損失も `NaN` を伝播すべきで、
// `f64::max(0.0)` の NaN 消失により黙って 0 に潰れてはならない
// （codex-review 指摘・PR #2286）。
#[test]
fn margin_ranking_loss_nan_input_propagates_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[f32::NAN], &[1]));
    let x2 = tape.var(&f32_tensor(&[1.0], &[1]));
    let y = f32_tensor(&[1.0], &[1]);

    let loss = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.5, Reduction::Mean).unwrap();
    assert!(scalar(&loss.to_tensor()).is_nan());
}

#[test]
fn margin_ranking_loss_empty_is_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[], &[0]));
    let x2 = tape.var(&f32_tensor(&[], &[0]));
    let y = f32_tensor(&[], &[0]);

    let mean = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap();
    assert_eq!(scalar(&mean.to_tensor()), 0.0);
}

#[test]
fn margin_ranking_loss_rejects_non_pm_one_label() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0], &[1]));
    let x2 = tape.var(&f32_tensor(&[0.0], &[1]));
    let y = f32_tensor(&[2.0], &[1]);

    let err = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn margin_ranking_loss_rejects_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0, 2.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[1.0, 2.0, 3.0], &[3]));
    let y = f32_tensor(&[1.0, 1.0], &[2]);

    let err = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.0, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn margin_ranking_loss_grad_matches_numeric_central_difference() {
    // 活性・非活性が混在する fixture（kink から離れた点）。
    let x1_data = vec![2.0f32, 0.0, -1.0, 3.0];
    let x2_data = vec![0.5f32, 1.0, 1.5, -2.0];
    let x2 = f32_tensor(&x2_data, &[4]);
    let y = f32_tensor(&[1.0, -1.0, 1.0, -1.0], &[4]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let x1_var = tape.var(&f32_tensor(&x1_data, &[4]));
    let x2_var = tape.var(&x2);
    let loss = loss_ops::margin_ranking_loss(&x1_var, &x2_var, &y, 0.3, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x1_var).unwrap().expect("x1 は loss に到達する"));

    let eval_loss = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let a = t.var(&f32_tensor(data, &[4]));
        let b = t.var(&x2);
        let l = loss_ops::margin_ranking_loss(&a, &b, &y, 0.3, Reduction::Mean).unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let mut data = x1_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("margin_ranking_grad[{i}]"), analytic[i], numeric);
    }
}

// =====================================================================
// 11. TripletMargin 損失（イシュー #2167）
// =====================================================================

// a=[0,0]・p=[1,0]（d_ap=1）・n=[0,4]（d_an=4）・margin=1 →
// loss = max(0, 1-4+1) = 0。
#[test]
fn triplet_margin_loss_forward_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let n = tape.var(&f32_tensor(&[0.0, 4.0], &[2]));
    let options = TripletMarginOptions::default();

    let loss = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();
    assert!(scalar(&loss.to_tensor()).abs() < 1e-4);
}

#[test]
fn triplet_margin_loss_swap_selects_smaller_negative_distance() {
    // d_ap=1・d_an=10（swap 無効なら loss=max(0,1-10+1)=0）。
    // swap 有効で d_pn=0.5 のとき d_neg=min(10,0.5)=0.5 →
    // loss=max(0,1-0.5+1)=1.5。
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let n = tape.var(&f32_tensor(&[1.0, 0.5], &[2]));
    let options = TripletMarginOptions::default().swap(true);

    let loss = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 1.5).abs() < 1e-3);
}

// `swap` 有効時、`d_an`（anchor-negative 距離）が `inf`・`d_pn`
// （positive-negative 距離）が有限だと、`d_neg` を係数付き和
// `an_coeff·d_an + (1−an_coeff)·d_pn` で求める実装は `an_coeff=0.0`
// でも `0.0 * inf = NaN` になり、有限な `d_pn` を選ぶべき `d_neg` が
// NaN に汚染される欠陥があった（codex-review 指摘・PR #2286）。
// `anchor` に `inf` を置くことで、`diff_an = anchor − negative` は
// `inf` になる一方、`diff_pn = positive − negative`（anchor 非依存）
// は有限のままという状況を作る。修正後は分岐で `d_neg = d_pn`
// （有限）を選び、forward は `NaN` ではなく `inf`（`d_ap` も `inf`
// になるため）を返し、backward も有限勾配を返す。
#[test]
fn triplet_margin_loss_swap_infinite_an_finite_pn_does_not_produce_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[f32::INFINITY, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let n = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let options = TripletMarginOptions::default().swap(true);

    let loss = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();
    let got = scalar(&loss.to_tensor());
    // `d_ap`・`d_an` は共に `inf`・`d_pn` は `0`（有限）。修正前は
    // `d_neg` が `NaN` に汚染され `loss` も `NaN` になっていた。
    assert!(!got.is_nan(), "got={got}（NaN であってはならない）");
    assert!(got.is_infinite() && got > 0.0, "got={got}");

    let grads = tape.backward(&loss).unwrap();
    for (name, var) in [("anchor", &a), ("positive", &p), ("negative", &n)] {
        let g = dense(grads.get(var).unwrap().expect("hinge 有効のため到達する"));
        assert!(
            g.iter().all(|v| !v.is_nan()),
            "{name} の勾配に NaN が含まれる: grad={g:?}"
        );
    }
}

// 有効な `p`（`p >= 1.0` 制約を満たす `p=1024`）と現実的な差分値
// （`|diff_i| <= 3`）でも、素直な `Σ|v_i|^p` の計算は `|v_i|^p` 自体が
// `f64` の範囲を超えて overflow し、`d_ap`/`d_an` が `inf` になって
// `inf - inf = NaN` の損失・勾配を生む欠陥があった（codex-review 指摘・
// PR #2286）。overflow-safe なスケール形（`p_norm_f64`）への修正で
// forward・backward とも有限値を維持することを確認する。
#[test]
fn triplet_margin_loss_large_p_stays_finite_for_forward_and_backward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[3.0, 0.0], &[2]));
    let n = tape.var(&f32_tensor(&[0.0, 1.0], &[2]));
    let options = TripletMarginOptions::default().p(1024.0);

    let loss = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();
    let got = scalar(&loss.to_tensor());
    // p=1024 では p ノルムは実質 L∞（最大要素の絶対値）に収束するため
    // d_ap≈3・d_an≈1・loss=max(0, 3-1+margin(1.0))≈3。
    assert!(got.is_finite(), "got={got}");
    assert!((got - 3.0).abs() < 1e-2, "got={got}");

    let grads = tape.backward(&loss).unwrap();
    for var in [&a, &p, &n] {
        let g = dense(grads.get(var).unwrap().expect("hinge 有効のため到達する"));
        assert!(g.iter().all(|v| v.is_finite()), "grad={g:?}");
    }
}

#[test]
fn triplet_margin_loss_rejects_non_finite_p() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[0.0], &[1]));
    let p = tape.var(&f32_tensor(&[1.0], &[1]));
    let n = tape.var(&f32_tensor(&[2.0], &[1]));
    let options = TripletMarginOptions::default().p(0.5);

    let err = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn triplet_margin_loss_rejects_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[1.0, 0.0, 0.0], &[3]));
    let n = tape.var(&f32_tensor(&[0.0, 4.0], &[2]));
    let options = TripletMarginOptions::default();

    let err = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

// `NaN` を含む入力は距離統計を `NaN` にし、損失も `NaN` を伝播すべきで、
// `f64::max(0.0)` の NaN 消失により黙って 0 に潰れてはならない
// （codex-review 指摘・PR #2286）。
#[test]
fn triplet_margin_loss_nan_input_propagates_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[f32::NAN, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let n = tape.var(&f32_tensor(&[0.0, 4.0], &[2]));
    let options = TripletMarginOptions::default();

    let loss = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();
    assert!(scalar(&loss.to_tensor()).is_nan());
}

#[test]
fn triplet_margin_loss_grad_matches_numeric_central_difference() {
    let a_data = vec![0.2f32, -0.3, 1.1, 0.4, -0.7, 0.9];
    let p_data = vec![1.0f32, 0.5, -0.2, 0.1, 0.3, -0.4];
    let n_data = vec![-0.5f32, 1.2, 0.6, -0.9, 1.0, 0.2];
    let p_t = f32_tensor(&p_data, &[2, 3]);
    let n_t = f32_tensor(&n_data, &[2, 3]);
    let options = TripletMarginOptions::default().margin(0.5);

    let tape = Tape::new_with_ops(common::naive_ops());
    let a_var = tape.var(&f32_tensor(&a_data, &[2, 3]));
    let p_var = tape.var(&p_t);
    let n_var = tape.var(&n_t);
    let loss =
        loss_ops::triplet_margin_loss(&a_var, &p_var, &n_var, &options, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(
        grads
            .get(&a_var)
            .unwrap()
            .expect("anchor は loss に到達する"),
    );

    let eval_loss = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let a = t.var(&f32_tensor(data, &[2, 3]));
        let p = t.var(&p_t);
        let n = t.var(&n_t);
        let l = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Mean).unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let mut data = a_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(
            &format!("triplet_margin_grad_anchor[{i}]"),
            analytic[i],
            numeric,
        );
    }
}

// =====================================================================
// 12. PoissonNLL 損失（イシュー #2167）
// =====================================================================

// log_input=true・x=0・t=1 → loss = exp(0) - 1*0 = 1。
#[test]
fn poisson_nll_loss_log_input_forward_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[0.0], &[1]));
    let target = tape.var(&f32_tensor(&[1.0], &[1]));
    let options = PoissonNllOptions::default();

    let loss = loss_ops::poisson_nll_loss(&input, &target, &options, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 1.0).abs() < 1e-5);
}

// log_input=false・x=1・t=2・eps=0 → loss = 1 - 2*ln(1) = 1。
#[test]
fn poisson_nll_loss_non_log_input_forward_matches_analytic_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[1.0], &[1]));
    let target = tape.var(&f32_tensor(&[2.0], &[1]));
    let options = PoissonNllOptions::default().log_input(false).eps(0.0);

    let loss = loss_ops::poisson_nll_loss(&input, &target, &options, Reduction::Sum).unwrap();
    assert!((scalar(&loss.to_tensor()) - 1.0).abs() < 1e-5);
}

// full=true・t=1 は Stirling 項が寄与しない（t>1 のみ）。
#[test]
fn poisson_nll_loss_full_term_skips_target_equal_one() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[0.0], &[1]));
    let target = tape.var(&f32_tensor(&[1.0], &[1]));
    let base_options = PoissonNllOptions::default();
    let full_options = PoissonNllOptions::default().full(true);

    let base = loss_ops::poisson_nll_loss(&input, &target, &base_options, Reduction::Sum).unwrap();
    let full = loss_ops::poisson_nll_loss(&input, &target, &full_options, Reduction::Sum).unwrap();
    assert!((scalar(&base.to_tensor()) - scalar(&full.to_tensor())).abs() < 1e-6);
}

// full=true・t=e（e>1）で Stirling 項が非ゼロ寄与を持つ。
#[test]
fn poisson_nll_loss_full_term_contributes_when_target_greater_than_one() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[0.0], &[1]));
    let target = tape.var(&f32_tensor(&[2.0], &[1]));
    let base_options = PoissonNllOptions::default();
    let full_options = PoissonNllOptions::default().full(true);

    let base = loss_ops::poisson_nll_loss(&input, &target, &base_options, Reduction::Sum).unwrap();
    let full = loss_ops::poisson_nll_loss(&input, &target, &full_options, Reduction::Sum).unwrap();
    assert!(scalar(&full.to_tensor()) > scalar(&base.to_tensor()) + 1e-3);
}

#[test]
fn poisson_nll_loss_empty_is_zero() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[], &[0]));
    let target = tape.var(&f32_tensor(&[], &[0]));
    let options = PoissonNllOptions::default();

    let mean = loss_ops::poisson_nll_loss(&input, &target, &options, Reduction::Mean).unwrap();
    assert_eq!(scalar(&mean.to_tensor()), 0.0);
}

#[test]
fn poisson_nll_loss_rejects_negative_eps() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[0.0], &[1]));
    let target = tape.var(&f32_tensor(&[1.0], &[1]));
    let options = PoissonNllOptions::default().log_input(false).eps(-1.0);

    let err = loss_ops::poisson_nll_loss(&input, &target, &options, Reduction::Mean).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn poisson_nll_loss_grad_matches_numeric_central_difference_log_input_full() {
    let input_data = vec![0.2f32, -0.3, 0.5, 0.1];
    // `t == 1.0` は `full` 項の Stirling マスク境界（`t > 1` のみ寄与）
    // に一致する kink のため避ける（`l1_loss` の kink 回避方針と同じ）。
    let target_data = vec![2.5f32, 0.5, 3.0, 1.3];
    let target = f32_tensor(&target_data, &[4]);
    let options = PoissonNllOptions::default().full(true);

    let tape = Tape::new_with_ops(common::naive_ops());
    let input_var = tape.var(&f32_tensor(&input_data, &[4]));
    let target_var = tape.var(&target);
    let loss =
        loss_ops::poisson_nll_loss(&input_var, &target_var, &options, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let d_input = dense(
        grads
            .get(&input_var)
            .unwrap()
            .expect("input は loss に到達する"),
    );
    let d_target = dense(
        grads
            .get(&target_var)
            .unwrap()
            .expect("target は loss に到達する（full 項の dTarget を含む）"),
    );

    let eval_loss_input = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let i = t.var(&f32_tensor(data, &[4]));
        let tg = t.var(&target);
        let l = loss_ops::poisson_nll_loss(&i, &tg, &options, Reduction::Mean).unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let mut data = input_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss_input(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss_input(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("poisson_nll_grad_input[{i}]"), d_input[i], numeric);
    }

    let eval_loss_target = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let i = t.var(&f32_tensor(&input_data, &[4]));
        let tg = t.var(&f32_tensor(data, &[4]));
        let l = loss_ops::poisson_nll_loss(&i, &tg, &options, Reduction::Mean).unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let mut tdata = target_data.clone();
    for i in 0..tdata.len() {
        let orig = tdata[i] as f64;
        tdata[i] = (orig + H) as f32;
        let lp = eval_loss_target(&tdata);
        tdata[i] = (orig - H) as f32;
        let lm = eval_loss_target(&tdata);
        tdata[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(
            &format!("poisson_nll_grad_target[{i}]"),
            d_target[i],
            numeric,
        );
    }
}

// =====================================================================
// 13. `nn::loss` の薄いラッパー性
// =====================================================================

#[test]
fn nn_l1_loss_forward_matches_free_function_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let pred = tape.var(&f32_tensor(&[1.0, -2.0, 3.0, 0.5], &[2, 2]));
    let target = tape.var(&f32_tensor(&[0.5, -1.0, 2.5, 1.0], &[2, 2]));

    let module = L1Loss::new(Reduction::Mean);
    let via_module = module.forward(&pred, &target).unwrap();
    let via_free = loss_ops::l1_loss(&pred, &target, Reduction::Mean).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}

#[test]
fn nn_cross_entropy_forward_with_matches_free_function_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.5, 2.0], &[2, 3]));
    let targets = i32_tensor(&[2, 0], &[2]);
    let options = CrossEntropyOptions::default().label_smoothing(0.1);

    let module = CrossEntropyLoss {
        class_dim: 1,
        reduction: Reduction::Mean,
    };
    let via_module = module.forward_with(&x, &targets, &options).unwrap();
    let via_free =
        loss_ops::cross_entropy_loss_with(&x, &targets, 1, Reduction::Mean, &options).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}

#[test]
fn nn_cosine_embedding_loss_forward_matches_free_function_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[0.0, 1.0], &[2]));
    let y = f32_tensor(&[1.0], &[]);

    let module = CosineEmbeddingLoss::new(0.2, Reduction::Mean);
    let via_module = module.forward(&x1, &x2, &y).unwrap();
    let via_free = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.2, Reduction::Mean).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}

#[test]
fn nn_margin_ranking_loss_forward_matches_free_function_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x1 = tape.var(&f32_tensor(&[2.0, 1.0], &[2]));
    let x2 = tape.var(&f32_tensor(&[1.0, 3.0], &[2]));
    let y = f32_tensor(&[1.0, -1.0], &[2]);

    let module = MarginRankingLoss::new(0.5, Reduction::Sum);
    let via_module = module.forward(&x1, &x2, &y).unwrap();
    let via_free = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.5, Reduction::Sum).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}

#[test]
fn nn_triplet_margin_loss_forward_matches_free_function_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&f32_tensor(&[0.0, 0.0], &[2]));
    let p = tape.var(&f32_tensor(&[1.0, 0.0], &[2]));
    let n = tape.var(&f32_tensor(&[0.0, 4.0], &[2]));
    let options = TripletMarginOptions::default();

    let module = TripletMarginLoss::new(TripletMarginOptions::default(), Reduction::Sum);
    let via_module = module.forward(&a, &p, &n).unwrap();
    let via_free = loss_ops::triplet_margin_loss(&a, &p, &n, &options, Reduction::Sum).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}

#[test]
fn nn_poisson_nll_loss_forward_matches_free_function_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = tape.var(&f32_tensor(&[0.1, -0.2], &[2]));
    let target = tape.var(&f32_tensor(&[1.0, 2.0], &[2]));
    let options = PoissonNllOptions::default().full(true);

    let module = PoissonNllLoss::new(PoissonNllOptions::default().full(true), Reduction::Sum);
    let via_module = module.forward(&input, &target).unwrap();
    let via_free = loss_ops::poisson_nll_loss(&input, &target, &options, Reduction::Sum).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}

// =====================================================================
// 13. CTC 損失（イシュー #2168）
// =====================================================================

/// DP（`loss_ops::ctc_loss`）と独立に、全パス `c^t_max` を列挙して
/// collapse-then-remove-blank で `target` に一致するものの確率を `f64`
/// で合計する（実装計画 §5.2「総当たりとの一致」）。`lp` は生の対数値を
/// 返す関数で、正規化されている必要はない（DP 側も同じ値をそのまま
/// 使うため、比較の妥当性は正規化に依存しない）。
fn brute_force_ctc_nll(
    t_max: usize,
    c: usize,
    target: &[i32],
    blank: i32,
    lp: &dyn Fn(usize, usize) -> f64,
) -> f64 {
    let mut total = 0.0f64;
    let mut path = vec![0usize; t_max];
    loop {
        let mut collapsed: Vec<i32> = Vec::new();
        let mut prev: Option<usize> = None;
        for &p in &path {
            if Some(p) != prev {
                collapsed.push(p as i32);
            }
            prev = Some(p);
        }
        let label: Vec<i32> = collapsed.into_iter().filter(|&v| v != blank).collect();
        if label == target {
            let mut log_p = 0.0f64;
            for (t, &k) in path.iter().enumerate() {
                log_p += lp(t, k);
            }
            total += log_p.exp();
        }

        if t_max == 0 {
            break;
        }
        let mut i = t_max;
        let mut done = false;
        loop {
            if i == 0 {
                done = true;
                break;
            }
            i -= 1;
            path[i] += 1;
            if path[i] < c {
                break;
            }
            path[i] = 0;
        }
        if done {
            break;
        }
    }
    if total > 0.0 {
        -total.ln()
    } else {
        f64::INFINITY
    }
}

#[allow(clippy::too_many_arguments)]
fn ctc_forward_sum(
    lp_data: &[f32],
    t_max: usize,
    n: usize,
    c: usize,
    targets: &Tensor<i32>,
    input_lengths: &[usize],
    target_lengths: &[usize],
    options: &CtcLossOptions,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(lp_data, &[t_max, n, c]));
    let loss = loss_ops::ctc_loss(
        &log_probs,
        targets,
        input_lengths,
        target_lengths,
        options,
        Reduction::Sum,
    )
    .unwrap();
    scalar(&loss.to_tensor())
}

#[test]
fn ctc_loss_forward_matches_brute_force_enumeration_with_repeats() {
    // target=[1,1]（連続重複）・blank=0・T=3,C=2。
    let t_max = 3;
    let c = 2;
    let blank = 0i32;
    let target = [1i32, 1];
    let lp_data = [-0.5f32, -1.0, -1.2, -0.4, -0.8, -0.6];
    let lp = |t: usize, k: usize| lp_data[t * c + k] as f64;

    let expected = brute_force_ctc_nll(t_max, c, &target, blank, &lp);
    let targets = i32_tensor(&target, &[1, 2]);
    let options = CtcLossOptions::default();
    let actual = ctc_forward_sum(&lp_data, t_max, 1, c, &targets, &[t_max], &[2], &options);
    assert_close("ctc_brute_force_repeats", actual, expected as f32);
}

#[test]
fn ctc_loss_forward_matches_brute_force_enumeration_empty_target() {
    let t_max = 2;
    let c = 2;
    let blank = 0i32;
    let target: [i32; 0] = [];
    let lp_data = [-0.3f32, -1.4, -0.9, -0.5];
    let lp = |t: usize, k: usize| lp_data[t * c + k] as f64;

    let expected = brute_force_ctc_nll(t_max, c, &target, blank, &lp);
    let targets = i32_tensor(&[], &[1, 0]);
    let options = CtcLossOptions::default();
    let actual = ctc_forward_sum(&lp_data, t_max, 1, c, &targets, &[t_max], &[0], &options);
    assert_close("ctc_brute_force_empty_target", actual, expected as f32);
}

#[test]
fn ctc_loss_forward_matches_brute_force_enumeration_non_zero_blank() {
    // blank=2・C=3・target=[0,1]（target != blank）。
    let t_max = 4;
    let c = 3;
    let blank = 2i32;
    let target = [0i32, 1];
    let lp_data = [
        -0.4f32, -1.1, -0.6, -0.9, -0.3, -1.5, -0.7, -0.8, -0.5, -1.0, -0.6, -0.4,
    ];
    let lp = |t: usize, k: usize| lp_data[t * c + k] as f64;

    let expected = brute_force_ctc_nll(t_max, c, &target, blank, &lp);
    let targets = i32_tensor(&target, &[1, 2]);
    let options = CtcLossOptions::default().blank(2);
    let actual = ctc_forward_sum(&lp_data, t_max, 1, c, &targets, &[t_max], &[2], &options);
    assert_close("ctc_brute_force_non_zero_blank", actual, expected as f32);
}

#[test]
fn ctc_loss_reduction_mean_and_sum_match_definition() {
    let t_max = 3;
    let n = 2;
    let c = 2;
    let lp_data = vec![
        -0.5f32, -1.0, -1.2, -0.4, -0.8, -0.6, -0.3, -1.4, -0.9, -0.5, -0.2, -1.7,
    ];
    let targets = i32_tensor(&[1, 1, 1, 0], &[2, 2]);
    let target_lengths = [2usize, 1];
    let input_lengths = [3usize, 3];
    let options = CtcLossOptions::default();

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&lp_data, &[t_max, n, c]));
    let sum_loss = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &input_lengths,
        &target_lengths,
        &options,
        Reduction::Sum,
    )
    .unwrap();
    let mean_loss = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &input_lengths,
        &target_lengths,
        &options,
        Reduction::Mean,
    )
    .unwrap();

    // 個別サンプルの nll を Sum reduction（N=1 相当）で取り出し、定義
    // どおりの Mean/Sum と突き合わせる。
    let nll = |sample: usize, tl: usize, tgt: &[i32]| -> f32 {
        let t = Tape::new_with_ops(common::naive_ops());
        let mut sample_lp: Vec<f32> = Vec::with_capacity(t_max * c);
        for tt in 0..t_max {
            for k in 0..c {
                sample_lp.push(lp_data[tt * n * c + sample * c + k]);
            }
        }
        let lpv = t.var(&f32_tensor(&sample_lp, &[t_max, 1, c]));
        let tgt_tensor = i32_tensor(tgt, &[1, tl]);
        scalar(
            &loss_ops::ctc_loss(&lpv, &tgt_tensor, &[t_max], &[tl], &options, Reduction::Sum)
                .unwrap()
                .to_tensor(),
        )
    };
    let nll0 = nll(0, 2, &[1, 1]);
    let nll1 = nll(1, 1, &[1]);

    assert_close(
        "ctc_reduction_sum",
        scalar(&sum_loss.to_tensor()),
        nll0 + nll1,
    );
    assert_close(
        "ctc_reduction_mean",
        scalar(&mean_loss.to_tensor()),
        (nll0 / 2.0 + nll1 / 1.0) / 2.0,
    );
}

#[test]
fn ctc_loss_t_zero_target_zero_is_zero_loss() {
    let targets = i32_tensor(&[], &[1, 0]);
    let options = CtcLossOptions::default();
    let actual = ctc_forward_sum(&[], 0, 1, 2, &targets, &[0], &[0], &options);
    assert_eq!(actual, 0.0);
}

#[test]
fn ctc_loss_t_zero_target_nonzero_is_infinite() {
    let lp_data: [f32; 0] = [];
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();
    let actual = ctc_forward_sum(&lp_data, 0, 1, 2, &targets, &[0], &[1], &options);
    assert!(actual.is_infinite() && actual > 0.0);
}

#[test]
fn ctc_loss_unreachable_alignment_zero_infinity_false_is_infinite_and_grad_is_nan() {
    // target=[1,1]・T_n=1 は最短でも 3 フレーム（l, blank, l）必要で
    // 整列不能。
    let t_max = 1;
    let c = 2;
    let lp_data = [-0.3f32, -1.2];
    let targets = i32_tensor(&[1, 1], &[1, 2]);
    let options = CtcLossOptions::default();

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&lp_data, &[t_max, 1, c]));
    let loss = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &[t_max],
        &[2],
        &options,
        Reduction::Sum,
    )
    .unwrap();
    assert!(scalar(&loss.to_tensor()).is_infinite());

    let grads = tape.backward(&loss).unwrap();
    let d = dense(grads.get(&log_probs).unwrap().expect("到達する"));
    assert!(
        d.iter().all(|v| v.is_nan()),
        "整列不能・zero_infinity=false の勾配は NaN: {d:?}"
    );
}

#[test]
fn ctc_loss_unreachable_alignment_zero_infinity_true_is_zero_loss_and_grad() {
    let t_max = 1;
    let c = 2;
    let lp_data = [-0.3f32, -1.2];
    let targets = i32_tensor(&[1, 1], &[1, 2]);
    let options = CtcLossOptions::default().zero_infinity(true);

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&lp_data, &[t_max, 1, c]));
    let loss = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &[t_max],
        &[2],
        &options,
        Reduction::Sum,
    )
    .unwrap();
    assert_eq!(scalar(&loss.to_tensor()), 0.0);

    let grads = tape.backward(&loss).unwrap();
    let d = dense(grads.get(&log_probs).unwrap().expect("到達する"));
    assert!(
        d.iter().all(|&v| v == 0.0),
        "zero_infinity=true の勾配は全 0: {d:?}"
    );
}

#[test]
fn ctc_loss_propagates_nan() {
    let t_max = 2;
    let c = 2;
    let lp_data = [f32::NAN, -1.0, -0.5, -0.2];
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();
    let actual = ctc_forward_sum(&lp_data, t_max, 1, c, &targets, &[t_max], &[1], &options);
    assert!(actual.is_nan());
}

#[test]
fn ctc_loss_unreachable_state_plus_infinite_log_prob_does_not_become_nan() {
    // イシュー #2168 PR #2292 codex-review 指摘の再現ケース。`log_probs`
    // は値検査しない契約（`+inf` を受け付ける）ため、到達不能状態
    // （累積対数確率 `-inf`）に `+inf` の emission が乗ると素朴な加算
    // `acc + lp` は `-inf + inf = NaN` になる。target=[1]・blank=0・
    // T=2（拡張ラベル列 `[blank, 1, blank]` は長さ 3 で `t_n=2` では
    // 整列不能）で、frame0 の label クラスを `-inf`（状態 1 を強制的に
    // 不可能にする）・frame1 の blank クラスを `+inf` にすると、
    // 状態 2（`ext[2]=blank`）の α 遷移が `acc(-inf) + lp(+inf)` を
    // 踏む。`eval::ctc_add_emission`（`crate::grad::ctc_loss_vjp` の
    // α・β も共有）による修正後は到達不能状態を `-inf` のまま維持し
    // `NaN` にならない。
    let t_max = 2;
    let c = 2;
    let lp_data = [
        -0.5f32,
        f32::NEG_INFINITY, // frame0: blank=-0.5, label=-inf
        f32::INFINITY,
        -0.3, // frame1: blank=+inf, label=-0.3
    ];
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();
    let actual = ctc_forward_sum(&lp_data, t_max, 1, c, &targets, &[t_max], &[1], &options);
    assert!(
        !actual.is_nan(),
        "到達不能状態への +inf 加算で NaN になってはならない: {actual}"
    );

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&lp_data, &[t_max, 1, c]));
    let loss = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &[t_max],
        &[1],
        &options,
        Reduction::Sum,
    )
    .unwrap();
    assert!(!scalar(&loss.to_tensor()).is_nan());

    let grads = tape.backward(&loss).unwrap();
    let d = dense(grads.get(&log_probs).unwrap().expect("到達する"));
    assert!(
        d.iter().all(|v| !v.is_nan()),
        "VJP の α・β 計算でも到達不能状態からの NaN は生じない: {d:?}"
    );
}

#[test]
fn ctc_loss_empty_batch_is_zero_and_does_not_panic() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[], &[3, 0, 2]));
    let targets = i32_tensor(&[], &[0, 0]);
    let options = CtcLossOptions::default();

    let mean =
        loss_ops::ctc_loss(&log_probs, &targets, &[], &[], &options, Reduction::Mean).unwrap();
    assert_eq!(scalar(&mean.to_tensor()), 0.0);
    let sum = loss_ops::ctc_loss(&log_probs, &targets, &[], &[], &options, Reduction::Sum).unwrap();
    assert_eq!(scalar(&sum.to_tensor()), 0.0);

    let grads = tape.backward(&sum).unwrap();
    let d = grads
        .get(&log_probs)
        .unwrap()
        .expect("到達する（N=0 でもゼロ勾配を返す）");
    assert_eq!(d.shape(), &[3, 0, 2]);
}

#[test]
fn ctc_loss_padded_and_concatenated_target_formats_agree() {
    let t_max = 3;
    let c = 2;
    let lp_data = [-0.5f32, -1.0, -1.2, -0.4, -0.8, -0.6];
    let options = CtcLossOptions::default();

    let padded = i32_tensor(&[1, 1], &[1, 2]);
    let concatenated = i32_tensor(&[1, 1], &[2]);

    let padded_loss = ctc_forward_sum(&lp_data, t_max, 1, c, &padded, &[t_max], &[2], &options);
    let concat_loss = ctc_forward_sum(
        &lp_data,
        t_max,
        1,
        c,
        &concatenated,
        &[t_max],
        &[2],
        &options,
    );
    assert_eq!(padded_loss, concat_loss);
}

#[test]
fn ctc_loss_frames_beyond_input_length_have_zero_gradient() {
    let t_max = 4;
    let c = 2;
    let lp_data = [-0.5f32, -1.0, -1.2, -0.4, -0.8, -0.6, -0.9, -0.3];
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&lp_data, &[t_max, 1, c]));
    // input_lengths=2 < T=4: t=2,3 は寄与しないので勾配 0。
    let loss =
        loss_ops::ctc_loss(&log_probs, &targets, &[2], &[1], &options, Reduction::Sum).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let d = dense(grads.get(&log_probs).unwrap().expect("到達する"));
    for t in 2..t_max {
        for k in 0..c {
            assert_eq!(
                d[t * c + k],
                0.0,
                "t={t},k={k} はフレーム外のため勾配 0 のはず"
            );
        }
    }
}

#[test]
fn ctc_loss_grad_matches_numeric_central_difference() {
    // blank=2・C=3。sample0: target=[0,1]（tl=2, T_n=4）・
    // sample1: target=[1,0]（tl=2, T_n=3。frame t=3 は寄与しないはず）。
    let t_max = 4;
    let n = 2;
    let c = 3;
    let blank = 2usize;
    let lp_data = vec![
        -0.4f32, -1.1, -0.6, -0.9, -0.3, -1.5, -0.7, -0.8, -0.5, -1.0, -0.6, -0.4, -0.5, -1.2,
        -0.3, -0.6, -0.9, -0.7, -1.3, -0.4, -0.5, -0.8, -0.6, -0.9,
    ];
    let targets = i32_tensor(&[0, 1, 1, 0], &[2, 2]);
    let input_lengths = [4usize, 3];
    let target_lengths = [2usize, 2];
    let options = CtcLossOptions::default().blank(blank);

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&lp_data, &[t_max, n, c]));
    let loss = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &input_lengths,
        &target_lengths,
        &options,
        Reduction::Mean,
    )
    .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let d_analytic = dense(grads.get(&log_probs).unwrap().expect("到達する"));

    let eval_loss = |data: &[f32]| -> f64 {
        let t = Tape::new_with_ops(common::naive_ops());
        let lpv = t.var(&f32_tensor(data, &[t_max, n, c]));
        let l = loss_ops::ctc_loss(
            &lpv,
            &targets,
            &input_lengths,
            &target_lengths,
            &options,
            Reduction::Mean,
        )
        .unwrap();
        dense(&l.to_tensor())[0] as f64
    };

    let mut data = lp_data.clone();
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        let numeric = ((lp - lm) / (2.0 * H)) as f32;
        assert_close(&format!("ctc_loss_grad[{i}]"), d_analytic[i], numeric);
    }
}

#[test]
fn ctc_loss_rejects_rank_2_log_probs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 2]));
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[1], &[1], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_blank_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 1, 2]));
    let targets = i32_tensor(&[], &[1, 0]);
    let options = CtcLossOptions::default().blank(2);
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[1], &[0], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_input_lengths_length_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 1, 2]));
    let targets = i32_tensor(&[], &[1, 0]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(
        &log_probs,
        &targets,
        &[1, 1],
        &[0],
        &options,
        Reduction::Mean,
    )
    .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_input_length_exceeding_t() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 1, 2]));
    let targets = i32_tensor(&[], &[1, 0]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[2], &[0], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_target_length_exceeding_s_in_padded_form() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0, -0.3, -0.4], &[2, 1, 2]));
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[2], &[2], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_concatenated_length_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0, -0.3, -0.4], &[2, 1, 2]));
    let targets = i32_tensor(&[1, 1], &[2]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[2], &[1], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_target_value_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 1, 2]));
    let targets = i32_tensor(&[5], &[1, 1]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[1], &[1], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_target_value_equal_to_blank() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 1, 2]));
    let targets = i32_tensor(&[0], &[1, 1]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[1], &[1], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn ctc_loss_rejects_targets_rank_3() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0], &[1, 1, 2]));
    let targets = i32_tensor(&[1], &[1, 1, 1]);
    let options = CtcLossOptions::default();
    let err = loss_ops::ctc_loss(&log_probs, &targets, &[1], &[1], &options, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn nn_ctc_loss_forward_matches_free_function_directly() {
    use fandhe_ai_autodiff::nn::loss::CtcLoss;

    let tape = Tape::new_with_ops(common::naive_ops());
    let log_probs = tape.var(&f32_tensor(&[-0.5, -1.0, -1.2, -0.4], &[2, 1, 2]));
    let targets = i32_tensor(&[1], &[1, 1]);
    let options = CtcLossOptions::default();

    let module = CtcLoss::new(CtcLossOptions::default(), Reduction::Sum);
    let via_module = module.forward(&log_probs, &targets, &[2], &[1]).unwrap();
    let via_free =
        loss_ops::ctc_loss(&log_probs, &targets, &[2], &[1], &options, Reduction::Sum).unwrap();

    assert_eq!(dense(&via_module.to_tensor()), dense(&via_free.to_tensor()));
}
