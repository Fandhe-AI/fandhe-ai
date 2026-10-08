//! `fandhe_ai::nn::loss`（イシュー #2602）の facade 統合テスト。
//!
//! 役割: autodiff 側（`fandhe_ai_autodiff::nn::loss`）の純再エクスポートである損失構造体 14 種が、
//! facade のみの import で構築でき、対応する `Var` 委譲メソッドと forward 値・勾配が bit 一致する
//! （同一コード経路のため厳密一致で足り、tolerance は新設しない）ことを確認する。
//! ホスト側計算のみでバックエンドは関与せず、乱数も使わないため CUDA／Metal 実機・直列化は不要。
//! 決定記録は `docs/facade-nn-loss-structs-exposure-decision.md`。

use fandhe_ai::nn::loss::{
    BceLoss, BceWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss, CrossEntropyOptions,
    CtcLoss, CtcLossOptions, HuberLoss, KlDivLoss, L1Loss, MarginRankingLoss, MseLoss, NllLoss,
    PoissonNllLoss, PoissonNllOptions, Reduction, SmoothL1Loss, TripletMarginLoss,
    TripletMarginOptions,
};
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: tensor")
}

fn bits(v: &Var<'_>) -> Vec<u32> {
    v.to_tensor()
        .host_slice()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn grad_bits(tape: &Tape, loss: &Var<'_>, wrt: &Var<'_>) -> Vec<u32> {
    let g = tape.backward(loss).expect("test fixture: backward");
    g.get(wrt)
        .expect("test fixture: get")
        .expect("test fixture: grad present")
        .host_slice()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// 構造体経由と `Var` メソッド経由の forward 値・第 1 引数の勾配が bit 一致することを確認する。
fn assert_same<'t>(
    tape: &'t Tape,
    wrt: &Var<'t>,
    via_struct: Result<Var<'t>, AutodiffError>,
    via_var: Result<Var<'t>, AutodiffError>,
    what: &str,
) {
    let a = via_struct.unwrap_or_else(|e| panic!("{what}: struct: {e:?}"));
    let b = via_var.unwrap_or_else(|e| panic!("{what}: var: {e:?}"));
    assert_eq!(bits(&a), bits(&b), "{what}: forward 値");
    assert_eq!(
        grad_bits(tape, &a, wrt),
        grad_bits(tape, &b, wrt),
        "{what}: 勾配"
    );
}

#[test]
fn regression_and_binary_losses_match_var_methods() {
    let tape = fandhe_ai::tape();
    let pred = tape.var(&t(&[0.5, -1.0, 2.0, 0.25], &[4]));
    let target = tape.var(&t(&[0.0, 1.0, 1.5, 0.75], &[4]));
    for r in [Reduction::Mean, Reduction::Sum] {
        assert_same(
            &tape,
            &pred,
            MseLoss::new(r).forward(&pred, &target),
            pred.mse_loss_with(&target, r),
            "mse",
        );
        assert_same(
            &tape,
            &pred,
            L1Loss::new(r).forward(&pred, &target),
            pred.l1_loss(&target, r),
            "l1",
        );
        assert_same(
            &tape,
            &pred,
            HuberLoss::new(0.7, r).forward(&pred, &target),
            pred.huber_loss(&target, 0.7, r),
            "huber",
        );
        assert_same(
            &tape,
            &pred,
            SmoothL1Loss::new(0.7, r).forward(&pred, &target),
            pred.smooth_l1_loss(&target, 0.7, r),
            "smooth_l1",
        );
    }
    let p = tape.var(&t(&[0.2, 0.7, 0.9, 0.4], &[4]));
    let y = tape.var(&t(&[0.0, 1.0, 1.0, 0.0], &[4]));
    assert_same(
        &tape,
        &p,
        BceLoss::new(Reduction::Mean).forward(&p, &y),
        p.bce_loss(&y, Reduction::Mean),
        "bce",
    );
    assert_same(
        &tape,
        &pred,
        BceWithLogitsLoss::new(Reduction::Sum).forward(&pred, &y),
        pred.bce_with_logits_loss(&y, Reduction::Sum),
        "bce_with_logits",
    );
}

#[test]
fn class_and_distribution_losses_match_var_methods() {
    let tape = fandhe_ai::tape();
    let logits = tape.var(&t(&[1.0, 2.0, 0.5, -1.0, 0.0, 3.0], &[2, 3]));
    let targets = Tensor::new(vec![1_i32, 2], &[2]).expect("targets");

    // CrossEntropyLoss は new／Default を持たず構造体リテラルで構築する。
    let ce = CrossEntropyLoss {
        class_dim: 1,
        reduction: Reduction::Mean,
    };
    assert_same(
        &tape,
        &logits,
        ce.forward(&logits, &targets),
        logits.cross_entropy_loss(&targets, 1, Reduction::Mean),
        "ce",
    );
    let opts = CrossEntropyOptions::default().label_smoothing(0.1);
    assert_same(
        &tape,
        &logits,
        ce.forward_with(&logits, &targets, &opts),
        logits.cross_entropy_loss_with(&targets, 1, Reduction::Mean, &opts),
        "ce_with",
    );
    assert_same(
        &tape,
        &logits,
        NllLoss::new(1, Reduction::Sum).forward(&logits, &targets),
        logits.nll_loss(&targets, 1, Reduction::Sum),
        "nll",
    );

    let probs = tape.var(&t(&[0.2, 0.3, 0.5, 0.1, 0.6, 0.3], &[2, 3]));
    assert_same(
        &tape,
        &logits,
        KlDivLoss::new(Reduction::Sum).forward(&logits, &probs),
        logits.kl_div_loss(&probs, Reduction::Sum),
        "kl",
    );
    let kl_log = KlDivLoss::new_with_log_target(Reduction::Mean, true);
    assert!(kl_log.log_target);
    assert_same(
        &tape,
        &logits,
        kl_log.forward(&logits, &probs),
        logits.kl_div_loss_with_log_target(&probs, Reduction::Mean),
        "kl_log",
    );
}

#[test]
fn embedding_poisson_and_ctc_losses_match_var_methods() {
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&t(&[1.0, 0.0, 0.5, 0.5, 1.0, -1.0], &[2, 3]));
    let x2 = tape.var(&t(&[0.5, 0.5, 0.0, 1.0, 0.0, 1.0], &[2, 3]));
    let y = t(&[1.0, -1.0], &[2]);
    assert_same(
        &tape,
        &x1,
        CosineEmbeddingLoss::new(0.1, Reduction::Mean).forward(&x1, &x2, &y),
        x1.cosine_embedding_loss(&x2, &y, 0.1, Reduction::Mean),
        "cosine",
    );

    let a = tape.var(&t(&[0.1, 0.9, 0.4], &[3]));
    let b = tape.var(&t(&[0.3, 0.2, 0.8], &[3]));
    let yr = t(&[1.0, -1.0, 1.0], &[3]);
    assert_same(
        &tape,
        &a,
        MarginRankingLoss::new(0.2, Reduction::Sum).forward(&a, &b, &yr),
        a.margin_ranking_loss(&b, &yr, 0.2, Reduction::Sum),
        "margin_ranking",
    );

    let topts = TripletMarginOptions::default().margin(0.5);
    let neg = tape.var(&t(&[0.0, 0.5, 0.5, 1.0, 1.0, 0.0], &[2, 3]));
    assert_same(
        &tape,
        &x1,
        TripletMarginLoss::new(topts.clone(), Reduction::Mean).forward(&x1, &x2, &neg),
        x1.triplet_margin_loss(&x2, &neg, &topts, Reduction::Mean),
        "triplet",
    );

    let rate = tape.var(&t(&[0.5, 1.0, 2.0], &[3]));
    let cnt = tape.var(&t(&[1.0, 0.0, 3.0], &[3]));
    let popts = PoissonNllOptions::default();
    assert_same(
        &tape,
        &rate,
        PoissonNllLoss::new(popts.clone(), Reduction::Mean).forward(&rate, &cnt),
        rate.poisson_nll_loss(&cnt, &popts, Reduction::Mean),
        "poisson",
    );

    // CTC: T=3, N=1, C=3（blank=0）、一様な log 確率。
    let lp = tape.var(&t(&[(1.0f32 / 3.0).ln(); 9], &[3, 1, 3]));
    let tg = Tensor::new(vec![1_i32, 2], &[1, 2]).expect("targets");
    let copts = CtcLossOptions::default();
    assert_same(
        &tape,
        &lp,
        CtcLoss::new(copts.clone(), Reduction::Sum).forward(&lp, &tg, &[3], &[2]),
        lp.ctc_loss(&tg, &[3], &[2], &copts, Reduction::Sum),
        "ctc",
    );
}

#[test]
fn reduction_changes_value_and_default_is_mean() {
    let tape = fandhe_ai::tape();
    let p = tape.var(&t(&[1.0, 2.0], &[2]));
    let z = tape.var(&t(&[0.0, 0.0], &[2]));
    let sum = MseLoss::new(Reduction::Sum).forward(&p, &z).expect("sum");
    let mean = MseLoss::default().forward(&p, &z).expect("mean");
    assert_eq!(sum.to_tensor().host_slice().into_owned(), [5.0]);
    assert_eq!(mean.to_tensor().host_slice().into_owned(), [2.5]);
}

#[test]
fn pub_fields_and_copy_contract() {
    let ce = CrossEntropyLoss {
        class_dim: 1,
        reduction: Reduction::Sum,
    };
    let copy = ce;
    assert_eq!(ce.class_dim, copy.class_dim);
    let cos = CosineEmbeddingLoss::new(0.3, Reduction::Mean);
    let cos2 = cos;
    assert_eq!(cos.margin, cos2.margin);
    let _ = TripletMarginLoss::default().clone();
}

#[test]
fn shape_mismatch_propagates_autodiff_error() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(&[1.0, 2.0, 3.0], &[3]));
    let b = tape.var(&t(&[1.0, 2.0], &[2]));
    assert!(MseLoss::default().forward(&a, &b).is_err());
    assert!(L1Loss::default().forward(&a, &b).is_err());
}
