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
//! 判定基準（backward）: 承認済み複合判定「相対誤差 1e-2 または絶対
//! 誤差 1e-3」＋`τ=1e-4`（`crates/autodiff/tests/nn_cross_entropy.rs`・
//! `nn_loss.rs` と同一パラメータを再利用。新規閾値は導入しない）。

mod common;

use fandhe_ai_autodiff::loss_ops::{self, CrossEntropyOptions};
use fandhe_ai_autodiff::nn::loss::{CrossEntropyLoss, L1Loss};
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
// 8. `nn::loss` の薄いラッパー性
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
