//! facade（`fandhe_ai::Var`）経由の `l1_loss`／`cross_entropy_loss_with` 利用例と
//! 単体テスト（イシュー #2538。実装は #2166・親 #2537・ルート #2499）。
//!
//! `Var::l1_loss` 等は `fandhe_ai_autodiff::loss_ops` の同名自由関数への 1 行委譲
//! メソッドで、facade は `Var` を再エクスポートするため追加の公開経路を持たない。
//! `Reduction`／`CrossEntropyOptions` の facade 再エクスポートは未承認
//! （`docs/autodiff-loss-ops-decision.md` §6）のため `fandhe_ai_autodiff` から import する。
//! 委譲の同値性は自由関数との bit 一致で、手計算値は REQ-2 統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で確認し、tolerance は新設・変更しない。
//! CPU（`fandhe_ai::tape()`）のみ。GPU 側は `loss_ops_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::{AutodiffError, Tape, Tensor, Var};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::loss_ops::{self, CrossEntropyOptions};

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

#[test]
fn l1_loss_forward_mean_and_sum() {
    let tape = fandhe_ai::tape();
    let pred = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let target = tape.var(&t(vec![0.0, 4.0, 3.0], &[3]));
    assert_eq!(
        vals(&pred.l1_loss(&target, Reduction::Mean).unwrap()),
        [1.0]
    );
    assert_eq!(vals(&pred.l1_loss(&target, Reduction::Sum).unwrap()), [3.0]);
}

#[test]
fn l1_loss_backward_is_sign_over_n_with_zero_at_tie() {
    let tape = fandhe_ai::tape();
    let pred = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let target = tape.var(&t(vec![0.0, 4.0, 3.0], &[3]));
    let loss = pred.l1_loss(&target, Reduction::Mean).unwrap();
    let n = 3.0_f32;
    assert_req2_close(&grad_of(&tape, &loss, &pred), &[1.0 / n, -1.0 / n, 0.0]);
    assert_req2_close(&grad_of(&tape, &loss, &target), &[-1.0 / n, 1.0 / n, 0.0]);
}

#[test]
fn l1_loss_matches_free_function_bitwise() {
    let tape = fandhe_ai::tape();
    let pred = tape.var(&t(vec![0.3, -1.2, 2.5, 0.0], &[2, 2]));
    let target = tape.var(&t(vec![0.1, 0.4, 2.0, -0.5], &[2, 2]));
    for red in [Reduction::Mean, Reduction::Sum] {
        let a = pred.l1_loss(&target, red).unwrap();
        let b = loss_ops::l1_loss(&pred, &target, red).unwrap();
        assert_eq!(bits(&vals(&a)), bits(&vals(&b)));
        assert_eq!(
            bits(&grad_of(&tape, &a, &pred)),
            bits(&grad_of(&tape, &b, &pred))
        );
    }
}

#[test]
fn l1_loss_errors_are_typed() {
    let tape = fandhe_ai::tape();
    let pred = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let bad = tape.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        pred.l1_loss(&bad, Reduction::Mean),
        Err(AutodiffError::Shape(_))
    ));
    let other = fandhe_ai::tape();
    let foreign = other.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    assert!(matches!(
        pred.l1_loss(&foreign, Reduction::Mean),
        Err(AutodiffError::TapeMismatch)
    ));
}

fn ce_fixture(tape: &Tape) -> (Var<'_>, Tensor<i32>) {
    let logits = tape.var(&t(vec![2.0, 0.0, 0.0, 0.0, 1.0, -1.0], &[3, 2]));
    (logits, ti(vec![0, 1, 1], &[3]))
}

#[test]
fn cross_entropy_default_options_equal_cross_entropy_loss_bitwise() {
    let tape = fandhe_ai::tape();
    let (logits, targets) = ce_fixture(&tape);
    for red in [Reduction::Mean, Reduction::Sum] {
        let a = logits
            .cross_entropy_loss_with(&targets, 1, red, &CrossEntropyOptions::default())
            .unwrap();
        let b = logits.cross_entropy_loss(&targets, 1, red).unwrap();
        assert_eq!(bits(&vals(&a)), bits(&vals(&b)));
        assert_eq!(
            bits(&grad_of(&tape, &a, &logits)),
            bits(&grad_of(&tape, &b, &logits))
        );
    }
}

#[test]
fn cross_entropy_non_default_options_match_free_function_and_hand_values() {
    let tape = fandhe_ai::tape();
    let ln2 = std::f32::consts::LN_2;

    // label_smoothing: 一様 logits（2 クラス）では損失は ln 2。
    let uni = tape.var(&t(vec![0.0, 0.0], &[1, 2]));
    let tg = ti(vec![0], &[1]);
    let opts = CrossEntropyOptions::default().label_smoothing(0.2);
    let a = uni
        .cross_entropy_loss_with(&tg, 1, Reduction::Mean, &opts)
        .unwrap();
    assert_req2_close(&vals(&a), &[ln2]);

    // ignore_index: 無視行は分母にも入らない（一様 logits の 1 行のみ → ln 2）。
    let two = tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));
    let tg2 = ti(vec![0, 1], &[2]);
    let opts = CrossEntropyOptions::default().ignore_index(1);
    let b = two
        .cross_entropy_loss_with(&tg2, 1, Reduction::Mean, &opts)
        .unwrap();
    assert_req2_close(&vals(&b), &[ln2]);

    // class_weight: 加重平均 (w0*l0 + w1*l1) / (w0 + w1)。
    let lg = tape.var(&t(vec![2.0, 0.0, 0.0, 0.0], &[2, 2]));
    let tg3 = ti(vec![0, 1], &[2]);
    let opts = CrossEntropyOptions::default().class_weight(t(vec![1.0, 3.0], &[2]));
    let c = lg
        .cross_entropy_loss_with(&tg3, 1, Reduction::Mean, &opts)
        .unwrap();
    let l0 = (1.0_f64 + (-2.0_f64).exp()).ln();
    let l1 = std::f64::consts::LN_2;
    let expected = ((l0 + 3.0 * l1) / 4.0) as f32;
    assert_req2_close(&vals(&c), &[expected]);

    // 組み合わせは自由関数と forward・grad で bit 一致。
    let (logits, targets) = ce_fixture(&tape);
    let opts = CrossEntropyOptions::default()
        .label_smoothing(0.1)
        .ignore_index(1)
        .class_weight(t(vec![1.0, 2.0], &[2]));
    for red in [Reduction::Mean, Reduction::Sum] {
        let m = logits
            .cross_entropy_loss_with(&targets, 1, red, &opts)
            .unwrap();
        let f = loss_ops::cross_entropy_loss_with(&logits, &targets, 1, red, &opts).unwrap();
        assert_eq!(bits(&vals(&m)), bits(&vals(&f)));
        assert_eq!(
            bits(&grad_of(&tape, &m, &logits)),
            bits(&grad_of(&tape, &f, &logits))
        );
    }
}

#[test]
fn cross_entropy_errors_are_typed() {
    let tape = fandhe_ai::tape();
    let (logits, targets) = ce_fixture(&tape);
    let run = |opts: &CrossEntropyOptions, tg: &Tensor<i32>, dim: usize| {
        logits
            .cross_entropy_loss_with(tg, dim, Reduction::Mean, opts)
            .map(|_| ())
    };
    for eps in [1.5_f32, -0.1, f32::NAN] {
        let o = CrossEntropyOptions::default().label_smoothing(eps);
        assert!(
            matches!(run(&o, &targets, 1), Err(AutodiffError::InvalidArgument(_))),
            "label_smoothing={eps}"
        );
    }
    let neg = CrossEntropyOptions::default().class_weight(t(vec![1.0, -1.0], &[2]));
    assert!(matches!(
        run(&neg, &targets, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let wrong = CrossEntropyOptions::default().class_weight(t(vec![1.0, 1.0, 1.0], &[3]));
    assert!(run(&wrong, &targets, 1).is_err());
    let oob = ti(vec![0, 2, 1], &[3]);
    let o = CrossEntropyOptions::default().label_smoothing(0.1);
    assert!(matches!(
        run(&o, &oob, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(run(&o, &targets, 5), Err(AutodiffError::Shape(_))));
}
