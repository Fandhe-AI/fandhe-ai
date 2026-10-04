//! facade（`fandhe_ai::Var`）経由の `cosine_embedding_loss`・`margin_ranking_loss`・
//! `triplet_margin_loss`・`poisson_nll_loss` 利用例と単体テスト
//! （イシュー #2539。実装は #2167・親 #2537・ルート #2499）。
//!
//! 各メソッドは `fandhe_ai_autodiff::loss_ops` の同名自由関数への 1 行委譲で、facade は
//! `Var` を再エクスポートするため追加の公開経路を持たない。`Reduction`・
//! `TripletMarginOptions`・`PoissonNllOptions` の facade 再エクスポートは一括承認の範囲外
//! （`docs/autodiff-distance-poisson-loss-ops-decision.md` §6）のため `fandhe_ai_autodiff`
//! から import する。委譲の同値性は自由関数との bit 一致で、手計算値は REQ-2 統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で確認し、tolerance は新設・変更しない。
//! CPU（`fandhe_ai::tape()`）のみ。GPU 側は `loss_ops_backend_parity.rs`（`#[ignore]`）。

use fandhe_ai::{AutodiffError, Tape, Tensor, Var};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::loss_ops::{self, PoissonNllOptions, TripletMarginOptions};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
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
fn cosine_embedding_forward_and_delegation_equivalence() {
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&t(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]));
    let x2 = tape.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));
    let y = t(vec![1.0, -1.0], &[2]);
    let mean = x1
        .cosine_embedding_loss(&x2, &y, 0.5, Reduction::Mean)
        .expect("mean");
    assert_req2_close(&vals(&mean), &[0.75]);
    let sum = x1
        .cosine_embedding_loss(&x2, &y, 0.5, Reduction::Sum)
        .expect("sum");
    assert_req2_close(&vals(&sum), &[1.5]);

    let free = loss_ops::cosine_embedding_loss(&x1, &x2, &y, 0.5, Reduction::Mean).expect("free");
    assert_eq!(bits(&vals(&mean)), bits(&vals(&free)));
    assert_eq!(
        bits(&grad_of(&tape, &mean, &x1)),
        bits(&grad_of(&tape, &free, &x1))
    );
    assert_eq!(
        bits(&grad_of(&tape, &mean, &x2)),
        bits(&grad_of(&tape, &free, &x2))
    );
}

#[test]
fn margin_ranking_forward_gradient_and_equivalence() {
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&t(vec![1.0, 2.0], &[2]));
    let x2 = tape.var(&t(vec![2.0, 1.0], &[2]));
    let y = t(vec![1.0, 1.0], &[2]);
    let mean = x1
        .margin_ranking_loss(&x2, &y, 0.0, Reduction::Mean)
        .expect("mean");
    assert_req2_close(&vals(&mean), &[0.5]);
    let sum = x1
        .margin_ranking_loss(&x2, &y, 0.0, Reduction::Sum)
        .expect("sum");
    assert_req2_close(&vals(&sum), &[1.0]);
    assert_req2_close(&grad_of(&tape, &mean, &x1), &[-0.5, 0.0]);
    assert_req2_close(&grad_of(&tape, &mean, &x2), &[0.5, 0.0]);

    let free = loss_ops::margin_ranking_loss(&x1, &x2, &y, 0.0, Reduction::Mean).expect("free");
    assert_eq!(bits(&vals(&mean)), bits(&vals(&free)));
    assert_eq!(
        bits(&grad_of(&tape, &mean, &x1)),
        bits(&grad_of(&tape, &free, &x1))
    );
}

#[test]
fn triplet_margin_forward_swap_and_equivalence() {
    let tape = fandhe_ai::tape();
    let anchor = tape.var(&t(vec![0.0, 0.0], &[1, 2]));
    let pos = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
    let neg = tape.var(&t(vec![0.0, 1.0], &[1, 2]));
    let opts = TripletMarginOptions::default().eps(0.0);
    let out = anchor
        .triplet_margin_loss(&pos, &neg, &opts, Reduction::Mean)
        .expect("triplet");
    assert_req2_close(&vals(&out), &[5.0]);

    // swap: d_pn = |pos - neg| = sqrt(9 + 9) ≈ 4.2426 > d_an = 1 なので min は d_an のまま → 5。
    let swapped = anchor
        .triplet_margin_loss(&pos, &neg, &opts.clone().swap(true), Reduction::Mean)
        .expect("swap");
    assert_req2_close(&vals(&swapped), &[5.0]);

    // swap が結果を変える入力: neg = (3, 3) とすると d_an = sqrt(18) ≈ 4.2426、d_pn = |pos - neg| = 1 < d_an。
    // swap なし: 5 - 4.2426 + 1 ≈ 1.7574、swap あり: d_an が d_pn = 1 に置換され 5 - 1 + 1 = 5。
    // swap 分岐が無視されると後者が 1.7574 になり、この検証で検出できる。
    let neg2 = tape.var(&t(vec![3.0, 3.0], &[1, 2]));
    let no_swap2 = anchor
        .triplet_margin_loss(&pos, &neg2, &opts, Reduction::Mean)
        .expect("no swap (d_pn < d_an)");
    assert_req2_close(&vals(&no_swap2), &[5.0 - 18.0_f32.sqrt() + 1.0]);
    let swapped2 = anchor
        .triplet_margin_loss(&pos, &neg2, &opts.clone().swap(true), Reduction::Mean)
        .expect("swap (d_pn < d_an)");
    assert_req2_close(&vals(&swapped2), &[5.0]);

    let free =
        loss_ops::triplet_margin_loss(&anchor, &pos, &neg, &opts, Reduction::Mean).expect("free");
    assert_eq!(bits(&vals(&out)), bits(&vals(&free)));
    assert_eq!(
        bits(&grad_of(&tape, &out, &anchor)),
        bits(&grad_of(&tape, &free, &anchor))
    );
}

#[test]
fn poisson_nll_forward_gradient_and_variants() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![0.0, 1.0], &[2]));
    let target = tape.var(&t(vec![1.0, 2.0], &[2]));
    let opts = PoissonNllOptions::default();
    let mean = x
        .poisson_nll_loss(&target, &opts, Reduction::Mean)
        .expect("mean");
    let e = std::f32::consts::E;
    assert_req2_close(&vals(&mean), &[(e - 1.0) / 2.0]);
    assert_req2_close(&grad_of(&tape, &mean, &x), &[0.0, (e - 2.0) / 2.0]);
    assert_req2_close(&grad_of(&tape, &mean, &target), &[0.0, -0.5]);

    // log_input=false・full=true（target > 1 を含む）は自由関数との bit 一致で確認する。
    let xp = tape.var(&t(vec![0.5, 1.5], &[2]));
    for opts in [
        PoissonNllOptions::default().log_input(false),
        PoissonNllOptions::default().full(true),
    ] {
        let m = xp
            .poisson_nll_loss(&target, &opts, Reduction::Sum)
            .expect("method");
        let f = loss_ops::poisson_nll_loss(&xp, &target, &opts, Reduction::Sum).expect("free");
        assert_eq!(bits(&vals(&m)), bits(&vals(&f)));
        assert_eq!(
            bits(&grad_of(&tape, &m, &xp)),
            bits(&grad_of(&tape, &f, &xp))
        );
    }
}

#[test]
fn invalid_inputs_return_typed_errors() {
    let tape = fandhe_ai::tape();
    let other = fandhe_ai::tape();
    let x1 = tape.var(&t(vec![1.0, 0.0, 1.0, 0.0], &[2, 2]));
    let x2 = tape.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));
    let bad_y = t(vec![0.5, -1.0], &[2]);
    assert!(matches!(
        x1.cosine_embedding_loss(&x2, &bad_y, 0.0, Reduction::Mean),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![2.0, 1.0], &[2]));
    assert!(matches!(
        a.margin_ranking_loss(&b, &bad_y, 0.0, Reduction::Mean),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let bad_p = TripletMarginOptions::default().p(0.5);
    assert!(
        x1.triplet_margin_loss(&x2, &x2, &bad_p, Reduction::Mean)
            .is_err()
    );
    // shape 不一致。
    let short = tape.var(&t(vec![1.0], &[1]));
    assert!(
        a.poisson_nll_loss(&short, &PoissonNllOptions::default(), Reduction::Mean)
            .is_err()
    );
    // 別テープの Var 混在。
    let foreign = other.var(&t(vec![2.0, 1.0], &[2]));
    let ones = t(vec![1.0, 1.0], &[2]);
    assert!(
        a.margin_ranking_loss(&foreign, &ones, 0.0, Reduction::Mean)
            .is_err()
    );
}
