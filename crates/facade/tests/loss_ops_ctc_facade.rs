//! facade（`fandhe_ai::Var`）経由の `ctc_loss` 利用例と単体テスト
//! （イシュー #2540。実装は #2168・親 #2537・ルート #2499）。
//!
//! `Var::ctc_loss` は `fandhe_ai_autodiff::loss_ops::ctc_loss` への 1 行委譲で、facade は
//! `Var` を再エクスポートするため追加の公開経路を持たない。`Reduction`・`CtcLossOptions` の
//! facade 再エクスポートは一括承認の範囲外（`docs/autodiff-ctc-design.md` §5）のため
//! `fandhe_ai_autodiff` から import する。委譲の同値性は自由関数との bit 一致で、手計算値は
//! REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で確認し、tolerance は
//! 新設・変更しない。CPU（`fandhe_ai::tape()`）のみ。GPU 側は `loss_ops_backend_parity.rs`
//! （`#[ignore]`）。

use fandhe_ai::{AutodiffError, Tape, Tensor, Var};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::loss_ops::{self, CtcLossOptions};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn vals(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn grad_of(tape: &Tape, loss: &Var<'_>, x: &Var<'_>) -> Vec<f32> {
    let grads = tape.backward(loss).expect("backward");
    grads
        .get(x)
        .expect("get")
        .expect("grad")
        .host_slice()
        .into_owned()
}

/// REQ-2 統一複合判定。
fn assert_req2_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(f32::MIN_POSITIVE);
        assert!(
            abs < 1e-5 || rel < 1e-3,
            "REQ-2 判定外: actual={a} expected={e}"
        );
    }
}

/// T=2, N=1, C=3 の log_probs。targets=[1, 2] の唯一の経路の nll は 1 + 2 = 3。
fn lp_unique_path() -> Tensor<f32> {
    t(vec![-0.5, -1.0, -0.7, -0.9, -0.4, -2.0], &[2, 1, 3])
}

#[test]
fn ctc_forward_matches_closed_form_and_free_fn_bitwise() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&lp_unique_path());
    let tg = ti(vec![1, 2], &[2]);
    let opts = CtcLossOptions::default();

    let sum = x
        .ctc_loss(&tg, &[2], &[2], &opts, Reduction::Sum)
        .expect("sum");
    assert_req2_close(&vals(&sum), &[3.0]);
    let mean = x
        .ctc_loss(&tg, &[2], &[2], &opts, Reduction::Mean)
        .expect("mean");
    assert_req2_close(&vals(&mean), &[1.5]);

    for red in [Reduction::Sum, Reduction::Mean] {
        let m = x.ctc_loss(&tg, &[2], &[2], &opts, red).expect("method");
        let f = loss_ops::ctc_loss(&x, &tg, &[2], &[2], &opts, red).expect("free");
        assert_eq!(bits(&vals(&m)), bits(&vals(&f)));
    }
}

#[test]
fn ctc_backward_matches_free_fn_bitwise() {
    // N=2・T=3・C=3。パディング形式 targets [N, S=2]。
    let data: Vec<f32> = (0..18).map(|i| -0.3 - 0.11 * (i as f32)).collect();
    let tg = ti(vec![1, 2, 2, 0], &[2, 2]);
    let opts = CtcLossOptions::default();
    let grads: Vec<Vec<f32>> = [false, true]
        .into_iter()
        .map(|use_free| {
            let tape = fandhe_ai::tape();
            let x = tape.var(&t(data.clone(), &[3, 2, 3]));
            let loss = if use_free {
                loss_ops::ctc_loss(&x, &tg, &[3, 3], &[2, 1], &opts, Reduction::Mean)
            } else {
                x.ctc_loss(&tg, &[3, 3], &[2, 1], &opts, Reduction::Mean)
            }
            .expect("ctc");
            grad_of(&tape, &loss, &x)
        })
        .collect();
    assert_eq!(bits(&grads[0]), bits(&grads[1]));
    assert!(grads[0].iter().any(|g| *g != 0.0));
}

#[test]
fn ctc_padded_and_concatenated_targets_agree_via_facade() {
    let data: Vec<f32> = (0..18).map(|i| -0.3 - 0.11 * (i as f32)).collect();
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(data, &[3, 2, 3]));
    let opts = CtcLossOptions::default();
    let padded = ti(vec![1, 2, 2, 0], &[2, 2]);
    let concat = ti(vec![1, 2, 2], &[3]);
    let a = x
        .ctc_loss(&padded, &[3, 3], &[2, 1], &opts, Reduction::Sum)
        .expect("padded");
    let b = x
        .ctc_loss(&concat, &[3, 3], &[2, 1], &opts, Reduction::Sum)
        .expect("concat");
    assert_eq!(bits(&vals(&a)), bits(&vals(&b)));
}

#[test]
fn ctc_zero_infinity_via_facade() {
    // T=1・target 長 2 は整列不能。
    let data = vec![-1.0_f32, -1.2, -1.4];
    let tg = ti(vec![1, 2], &[1, 2]);

    let tape = fandhe_ai::tape();
    let x = tape.var(&t(data.clone(), &[1, 1, 3]));
    let zi = CtcLossOptions::default().zero_infinity(true);
    let loss = x
        .ctc_loss(&tg, &[1], &[2], &zi, Reduction::Sum)
        .expect("zero_infinity");
    assert_eq!(vals(&loss), [0.0]);
    assert!(grad_of(&tape, &loss, &x).iter().all(|g| *g == 0.0));

    let tape = fandhe_ai::tape();
    let x = tape.var(&t(data, &[1, 1, 3]));
    let loss = x
        .ctc_loss(&tg, &[1], &[2], &CtcLossOptions::default(), Reduction::Sum)
        .expect("no zero_infinity");
    assert_eq!(vals(&loss)[0], f32::INFINITY);
}

#[test]
fn ctc_invalid_inputs_return_typed_errors() {
    let tape = fandhe_ai::tape();
    let opts = CtcLossOptions::default();
    let x = tape.var(&lp_unique_path());
    let tg = ti(vec![1, 2], &[2]);
    let is_invalid =
        |r: Result<Var<'_>, AutodiffError>| matches!(r, Err(AutodiffError::InvalidArgument(_)));

    // rank 2 の log_probs。
    let x2 = tape.var(&t(vec![-1.0; 6], &[2, 3]));
    assert!(is_invalid(x2.ctc_loss(
        &tg,
        &[2],
        &[2],
        &opts,
        Reduction::Sum
    )));
    // blank >= C。
    let bad_blank = CtcLossOptions::default().blank(3);
    assert!(is_invalid(x.ctc_loss(
        &tg,
        &[2],
        &[2],
        &bad_blank,
        Reduction::Sum
    )));
    // input_lengths.len() != N。
    assert!(is_invalid(x.ctc_loss(
        &tg,
        &[2, 2],
        &[2],
        &opts,
        Reduction::Sum
    )));
    // input_lengths[n] > T。
    assert!(is_invalid(x.ctc_loss(
        &tg,
        &[3],
        &[2],
        &opts,
        Reduction::Sum
    )));
    // target 値が blank。
    let tg_blank = ti(vec![0, 2], &[2]);
    assert!(is_invalid(x.ctc_loss(
        &tg_blank,
        &[2],
        &[2],
        &opts,
        Reduction::Sum
    )));
    // target 値が範囲外。
    let tg_oob = ti(vec![1, 3], &[2]);
    assert!(is_invalid(x.ctc_loss(
        &tg_oob,
        &[2],
        &[2],
        &opts,
        Reduction::Sum
    )));
}
