//! `Var` の損失委譲メソッド 8 本の facade 利用テスト。
//!
//! - #2677 の 3 本（`hinge_embedding_loss`・`soft_margin_loss`・`multilabel_margin_loss`。ルート #2499 の
//!   2026-10-07 承認コメント `issuecomment-6033824965`・`docs/compat-api-scope.md` §5.1 行 19・20）
//! - #2854 の 5 本（`bce_with_logits_loss_with`・`gaussian_nll_loss`・`multi_margin_loss`・
//!   `multilabel_soft_margin_loss`・`sigmoid_focal_loss`。2026-10-08 承認コメント
//!   `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`・
//!   `docs/facade-nn-loss-structs-exposure-decision.md` §11）。オプション型は
//!   `fandhe_ai::nn::loss` の 1 経路。
//!
//! 利用者が見る面は `fandhe_ai` だけ（`Reduction` は `fandhe_ai::nn::loss::Reduction` の 1 経路）。
//! 委譲メソッドの forward 値と backward（勾配）が、内部クレートの自由関数
//! （`fandhe_ai_autodiff::{elementwise_loss_ops, margin_focal_loss_ops}`。比較の参照としてのみ使う）と
//! bit 一致することを固定する。委譲先は同一コードのため新しい tolerance・baseline は設けない
//! （グローバル RNG は使わず、入力は固定値）。実機（CUDA／Metal）の一致は既存の
//! `elementwise_loss_ops_backend_parity.rs`・`margin_focal_loss_ops_backend_parity.rs` が担う。

use fandhe_ai::nn::loss::{
    BceWithLogitsOptions, GaussianNllOptions, MultiLabelSoftMarginOptions, MultiMarginOptions,
    Reduction, SigmoidFocalLossOptions,
};
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

type R<'t> = Result<Var<'t>, AutodiffError>;

/// シグネチャ固定用（`Var<'t>` の `'t` を名指しするため関数で包む）。
fn sig_hinge<'t>() -> fn(&Var<'t>, &Tensor<f32>, f32, Reduction) -> R<'t> {
    Var::<'t>::hinge_embedding_loss
}
fn sig_soft<'t>() -> fn(&Var<'t>, &Tensor<f32>, Reduction) -> R<'t> {
    Var::<'t>::soft_margin_loss
}
fn sig_multilabel<'t>() -> fn(&Var<'t>, &Tensor<i32>, Reduction) -> R<'t> {
    Var::<'t>::multilabel_margin_loss
}

fn sig_bce_with<'t>() -> fn(&Var<'t>, &Var<'t>, Reduction, &BceWithLogitsOptions) -> R<'t> {
    Var::<'t>::bce_with_logits_loss_with
}
fn sig_gaussian<'t>() -> fn(&Var<'t>, &Var<'t>, &Var<'t>, &GaussianNllOptions, Reduction) -> R<'t> {
    Var::<'t>::gaussian_nll_loss
}
fn sig_multi_margin<'t>() -> fn(&Var<'t>, &Tensor<i32>, &MultiMarginOptions, Reduction) -> R<'t> {
    Var::<'t>::multi_margin_loss
}
fn sig_mlsm<'t>() -> fn(&Var<'t>, &Tensor<f32>, &MultiLabelSoftMarginOptions, Reduction) -> R<'t> {
    Var::<'t>::multilabel_soft_margin_loss
}
fn sig_focal<'t>() -> fn(&Var<'t>, &Tensor<f32>, &SigmoidFocalLossOptions, Reduction) -> R<'t> {
    Var::<'t>::sigmoid_focal_loss
}
fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 負・0・hinge の境界（`x == margin`）・正を含む入力（shape `[2, 3]`）。
fn x_in() -> Tensor<f32> {
    Tensor::new(vec![-2.5, -0.5, 0.0, 1.0, 0.75, 3.0], &[2, 3]).expect("tensor")
}

fn y_pm() -> Tensor<f32> {
    Tensor::new(vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0], &[2, 3]).expect("tensor")
}

/// 行 0: target 列 `[3, 0]`（終端 -1 以降はゴミ）、行 1: target なし。
fn mlm_x() -> Tensor<f32> {
    Tensor::new(vec![0.1, 0.2, 0.4, 0.8, 0.5, -0.3, 0.9, 0.0], &[2, 4]).expect("tensor")
}

fn mlm_target() -> Tensor<i32> {
    Tensor::new(vec![3, 0, -1, 1, -1, 2, 0, 1], &[2, 4]).expect("tensor")
}

/// `(forward 出力, 入力の勾配)` を取り出す。
fn run<'t>(
    tape: &'t Tape,
    x: &Tensor<f32>,
    f: impl Fn(&Var<'t>) -> R<'t>,
) -> (Tensor<f32>, Tensor<f32>) {
    let xv = tape.var(x);
    let loss = f(&xv).expect("loss");
    let out = loss.to_tensor();
    let grads = tape.backward(&loss).expect("backward");
    let g = grads.get(&xv).expect("get").expect("grad").clone();
    (out, g)
}

fn assert_same(a: &(Tensor<f32>, Tensor<f32>), b: &(Tensor<f32>, Tensor<f32>), what: &str) {
    assert_eq!(a.0.shape(), b.0.shape(), "{what}: forward shape");
    assert_eq!(bits(&a.0), bits(&b.0), "{what}: forward bit 一致");
    assert_eq!(bits(&a.1), bits(&b.1), "{what}: backward bit 一致");
}

#[test]
fn signatures_match_approved_form() {
    // 取得できること自体が、引数順・戻り値型の固定になる。
    let _ = sig_hinge();
    let _ = sig_soft();
    let _ = sig_multilabel();
    let _ = sig_bce_with();
    let _ = sig_gaussian();
    let _ = sig_multi_margin();
    let _ = sig_mlsm();
    let _ = sig_focal();
}

#[test]
fn hinge_embedding_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::elementwise_loss_ops::hinge_embedding_loss;
    let tape = fandhe_ai::tape();
    for reduction in [Reduction::Mean, Reduction::Sum] {
        for margin in [0.75f32, 1.0] {
            let via_method = run(&tape, &x_in(), |x| {
                x.hinge_embedding_loss(&y_pm(), margin, reduction)
            });
            let via_free = run(&tape, &x_in(), |x| {
                hinge_embedding_loss(x, &y_pm(), margin, reduction)
            });
            assert_same(&via_method, &via_free, "hinge_embedding_loss");
        }
    }
}

#[test]
fn soft_margin_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::elementwise_loss_ops::soft_margin_loss;
    let tape = fandhe_ai::tape();
    for reduction in [Reduction::Mean, Reduction::Sum] {
        let via_method = run(&tape, &x_in(), |x| x.soft_margin_loss(&y_pm(), reduction));
        let via_free = run(&tape, &x_in(), |x| soft_margin_loss(x, &y_pm(), reduction));
        assert_same(&via_method, &via_free, "soft_margin_loss");
    }
}

#[test]
fn multilabel_margin_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::margin_focal_loss_ops::multilabel_margin_loss;
    let tape = fandhe_ai::tape();
    for reduction in [Reduction::Mean, Reduction::Sum] {
        let via_method = run(&tape, &mlm_x(), |x| {
            x.multilabel_margin_loss(&mlm_target(), reduction)
        });
        let via_free = run(&tape, &mlm_x(), |x| {
            multilabel_margin_loss(x, &mlm_target(), reduction)
        });
        assert_same(&via_method, &via_free, "multilabel_margin_loss");
    }
}

/// 手計算で厳密に決まる値（入口が実際に損失を計算していることの確認）。
#[test]
fn closed_form_values() {
    let tape = fandhe_ai::tape();
    // hinge: y=1 は x、y=-1 は max(0, margin - x)。x=[0.5, 1.5]・y=[1, -1]・margin=1 → 0.5 + 0。
    let x = tape.var(&Tensor::new(vec![0.5f32, 1.5], &[2]).expect("tensor"));
    let y = Tensor::new(vec![1.0f32, -1.0], &[2]).expect("tensor");
    let hinge = x
        .hinge_embedding_loss(&y, 1.0, Reduction::Sum)
        .expect("hinge");
    assert_eq!(hinge.to_tensor().host_slice().into_owned(), [0.5f32]);
    // soft margin: x=0 なら全要素 ln 2。
    let z = tape.var(&Tensor::new(vec![0.0f32, 0.0], &[2]).expect("tensor"));
    let soft = z.soft_margin_loss(&y, Reduction::Mean).expect("soft");
    assert!((soft.to_tensor().host_slice()[0] - std::f32::consts::LN_2).abs() < 1e-6);
    // multilabel margin（PyTorch 公式例）: x=[0.1, 0.2, 0.4, 0.8]・target=[3, 0, -1, 1] → 0.85。
    let m = tape.var(&Tensor::new(vec![0.1f32, 0.2, 0.4, 0.8], &[1, 4]).expect("tensor"));
    let t = Tensor::new(vec![3i32, 0, -1, 1], &[1, 4]).expect("tensor");
    let ml = m.multilabel_margin_loss(&t, Reduction::Mean).expect("mlm");
    assert!((ml.to_tensor().host_slice()[0] - 0.85).abs() < 1e-6);
}

/// 不正入力は型付きエラーで返る（panic しない）。
#[test]
fn invalid_arguments_are_typed_errors() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_in());
    let bad_y = Tensor::new(vec![0.5f32; 6], &[2, 3]).expect("tensor");
    assert!(x.soft_margin_loss(&bad_y, Reduction::Mean).is_err());
    assert!(
        x.hinge_embedding_loss(&bad_y, 1.0, Reduction::Mean)
            .is_err()
    );
    assert!(
        x.hinge_embedding_loss(&y_pm(), f32::NAN, Reduction::Mean)
            .is_err()
    );
    let m = tape.var(&mlm_x());
    let bad_t = Tensor::new(vec![0i32, 1, 2, 9, 0, 0, 0, 0], &[2, 4]).expect("tensor");
    assert!(m.multilabel_margin_loss(&bad_t, Reduction::Mean).is_err());
}
// ---- #2854: オプション型を取る損失 5 本 ----

/// 複数の追跡対象 `Var` を持つ損失用。`(forward 出力, 各入力の勾配)` を返す。
fn run_multi<'t>(
    tape: &'t Tape,
    inputs: &[Tensor<f32>],
    f: impl Fn(&[Var<'t>]) -> R<'t>,
) -> (Tensor<f32>, Vec<Tensor<f32>>) {
    let vars: Vec<Var<'t>> = inputs.iter().map(|t| tape.var(t)).collect();
    let loss = f(&vars).expect("loss");
    let out = loss.to_tensor();
    let grads = tape.backward(&loss).expect("backward");
    let gs = vars
        .iter()
        .map(|v| grads.get(v).expect("get").expect("grad").clone())
        .collect();
    (out, gs)
}

fn assert_same_multi(
    a: &(Tensor<f32>, Vec<Tensor<f32>>),
    b: &(Tensor<f32>, Vec<Tensor<f32>>),
    what: &str,
) {
    assert_eq!(a.0.shape(), b.0.shape(), "{what}: forward shape");
    assert_eq!(bits(&a.0), bits(&b.0), "{what}: forward bit 一致");
    assert_eq!(a.1.len(), b.1.len(), "{what}: 勾配の本数");
    for (i, (ga, gb)) in a.1.iter().zip(&b.1).enumerate() {
        assert_eq!(bits(ga), bits(gb), "{what}: backward[{i}] bit 一致");
    }
}

fn t23(d: [f32; 6]) -> Tensor<f32> {
    Tensor::new(d.to_vec(), &[2, 3]).expect("tensor")
}

fn logits() -> Tensor<f32> {
    t23([-2.5, -0.5, 0.0, 1.0, 0.75, 3.0])
}

fn unit_targets() -> Tensor<f32> {
    t23([1.0, 0.0, 1.0, 0.0, 0.25, 1.0])
}

fn pos_weight() -> Tensor<f32> {
    Tensor::new(vec![2.0f32, 0.5, 1.5], &[3]).expect("tensor")
}

fn class_weight() -> Tensor<f32> {
    Tensor::new(vec![1.0f32, 2.0, 0.5], &[3]).expect("tensor")
}

#[test]
fn bce_with_logits_loss_with_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::elementwise_loss_ops::bce_with_logits_loss_with;
    let tape = fandhe_ai::tape();
    let inputs = [logits(), unit_targets()];
    // 既定オプション（`Var::bce_with_logits_loss` へ短絡する経路）と `pos_weight` 指定。
    for opts in [
        BceWithLogitsOptions::default(),
        BceWithLogitsOptions::default().pos_weight(pos_weight()),
    ] {
        for reduction in [Reduction::Mean, Reduction::Sum] {
            let m = run_multi(&tape, &inputs, |v| {
                v[0].bce_with_logits_loss_with(&v[1], reduction, &opts)
            });
            let f = run_multi(&tape, &inputs, |v| {
                bce_with_logits_loss_with(&v[0], &v[1], reduction, &opts)
            });
            assert_same_multi(&m, &f, "bce_with_logits_loss_with");
        }
    }
}

#[test]
fn gaussian_nll_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::elementwise_loss_ops::gaussian_nll_loss;
    let tape = fandhe_ai::tape();
    let inputs = [
        logits(),
        t23([-2.0, 0.0, 0.5, 1.5, 0.25, 2.0]),
        t23([0.5, 1.0, 2.0, 0.0, 1.5, 0.75]),
    ];
    for opts in [
        GaussianNllOptions::default(),
        GaussianNllOptions::default().full(true).eps(1e-3),
    ] {
        for reduction in [Reduction::Mean, Reduction::Sum] {
            let m = run_multi(&tape, &inputs, |v| {
                v[0].gaussian_nll_loss(&v[1], &v[2], &opts, reduction)
            });
            let f = run_multi(&tape, &inputs, |v| {
                gaussian_nll_loss(&v[0], &v[1], &v[2], &opts, reduction)
            });
            assert_same_multi(&m, &f, "gaussian_nll_loss");
        }
    }
}

#[test]
fn multi_margin_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::margin_focal_loss_ops::multi_margin_loss;
    let tape = fandhe_ai::tape();
    let target = Tensor::new(vec![2i32, 0], &[2]).expect("tensor");
    for opts in [
        MultiMarginOptions::default(),
        MultiMarginOptions::default().p(2).margin(0.5),
        MultiMarginOptions::default().weight(class_weight()),
        MultiMarginOptions::default()
            .p(2)
            .margin(1.5)
            .weight(class_weight()),
    ] {
        for reduction in [Reduction::Mean, Reduction::Sum] {
            let m = run(&tape, &logits(), |x| {
                x.multi_margin_loss(&target, &opts, reduction)
            });
            let f = run(&tape, &logits(), |x| {
                multi_margin_loss(x, &target, &opts, reduction)
            });
            assert_same(&m, &f, "multi_margin_loss");
        }
    }
}

#[test]
fn multilabel_soft_margin_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::margin_focal_loss_ops::multilabel_soft_margin_loss;
    let tape = fandhe_ai::tape();
    let target = unit_targets();
    for opts in [
        MultiLabelSoftMarginOptions::default(),
        MultiLabelSoftMarginOptions::default().weight(class_weight()),
    ] {
        for reduction in [Reduction::Mean, Reduction::Sum] {
            let m = run(&tape, &logits(), |x| {
                x.multilabel_soft_margin_loss(&target, &opts, reduction)
            });
            let f = run(&tape, &logits(), |x| {
                multilabel_soft_margin_loss(x, &target, &opts, reduction)
            });
            assert_same(&m, &f, "multilabel_soft_margin_loss");
        }
    }
}

#[test]
fn sigmoid_focal_loss_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::margin_focal_loss_ops::sigmoid_focal_loss;
    let tape = fandhe_ai::tape();
    let target = unit_targets();
    for opts in [
        SigmoidFocalLossOptions::default(),
        SigmoidFocalLossOptions::default().alpha(None).gamma(0.0),
        SigmoidFocalLossOptions::default()
            .alpha(Some(0.5))
            .gamma(2.0),
        SigmoidFocalLossOptions::default().alpha(None).gamma(2.0),
    ] {
        for reduction in [Reduction::Mean, Reduction::Sum] {
            let m = run(&tape, &logits(), |x| {
                x.sigmoid_focal_loss(&target, &opts, reduction)
            });
            let f = run(&tape, &logits(), |x| {
                sigmoid_focal_loss(x, &target, &opts, reduction)
            });
            assert_same(&m, &f, "sigmoid_focal_loss");
        }
    }
}

/// 手計算で決まる値（#2854 の 5 本が実際に損失を計算していることの確認。許容誤差は既存テストと同じ書き方）。
#[test]
fn closed_form_values_for_option_losses() {
    let tape = fandhe_ai::tape();
    let v = |d: &[f32], s: &[usize]| tape.var(&Tensor::new(d.to_vec(), s).expect("tensor"));
    let ln2 = std::f32::consts::LN_2;

    // BCE-with-logits: logits 0 なら target に依らず ln 2。pos_weight は target=1 の項だけを重くする。
    let z = v(&[0.0, 0.0], &[2]);
    let t = v(&[1.0, 0.0], &[2]);
    let pw = BceWithLogitsOptions::default()
        .pos_weight(Tensor::new(vec![3.0f32, 3.0], &[2]).expect("tensor"));
    let plain = z
        .bce_with_logits_loss_with(&t, Reduction::Mean, &BceWithLogitsOptions::default())
        .expect("bce");
    assert!((plain.to_tensor().host_slice()[0] - ln2).abs() < 1e-6);
    let weighted = z
        .bce_with_logits_loss_with(&t, Reduction::Sum, &pw)
        .expect("bce pos_weight");
    // Sum = 3 ln 2（target=1）+ ln 2（target=0）= 4 ln 2。
    assert!((weighted.to_tensor().host_slice()[0] - 4.0 * ln2).abs() < 1e-5);

    // GaussianNLL: input == target・var == 1 なら 0。full では 0.5 ln(2π) が各要素に加わる。
    let x = v(&[0.5, -1.0], &[2]);
    let tg = v(&[0.5, -1.0], &[2]);
    let one = v(&[1.0, 1.0], &[2]);
    let g0 = x
        .gaussian_nll_loss(&tg, &one, &GaussianNllOptions::default(), Reduction::Mean)
        .expect("gnll");
    assert!(g0.to_tensor().host_slice()[0].abs() < 1e-6);
    let g1 = x
        .gaussian_nll_loss(
            &tg,
            &one,
            &GaussianNllOptions::default().full(true),
            Reduction::Mean,
        )
        .expect("gnll full");
    let half_ln_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
    assert!((g1.to_tensor().host_slice()[0] - half_ln_2pi).abs() < 1e-6);

    // MultiMargin（PyTorch 公式例）: x=[0.1, 0.2, 0.4, 0.8]・target=3・margin=1・p=1 → 1.3 / 4 = 0.325。
    let m = v(&[0.1, 0.2, 0.4, 0.8], &[1, 4]);
    let mt = Tensor::new(vec![3i32], &[1]).expect("tensor");
    let mm = m
        .multi_margin_loss(&mt, &MultiMarginOptions::default(), Reduction::Mean)
        .expect("multi_margin");
    assert!((mm.to_tensor().host_slice()[0] - 0.325).abs() < 1e-6);

    // MultiLabelSoftMargin: logits 0 なら全クラス ln 2（クラス平均）。
    let zz = v(&[0.0, 0.0], &[1, 2]);
    let yy = Tensor::new(vec![1.0f32, 0.0], &[1, 2]).expect("tensor");
    let ml = zz
        .multilabel_soft_margin_loss(
            &yy,
            &MultiLabelSoftMarginOptions::default(),
            Reduction::Mean,
        )
        .expect("mlsm");
    assert!((ml.to_tensor().host_slice()[0] - ln2).abs() < 1e-6);

    // SigmoidFocal: logits 0・target 1・alpha=None・gamma=0 なら係数 1 の BCE = ln 2。
    let f0 = v(&[0.0], &[1]);
    let ft = Tensor::new(vec![1.0f32], &[1]).expect("tensor");
    let fl = f0
        .sigmoid_focal_loss(
            &ft,
            &SigmoidFocalLossOptions::default().alpha(None).gamma(0.0),
            Reduction::Mean,
        )
        .expect("focal");
    assert!((fl.to_tensor().host_slice()[0] - ln2).abs() < 1e-6);
}

/// 不正なオプション・入力は型付きエラーで返る（panic しない。検査は委譲先 autodiff が行う）。
#[test]
fn invalid_option_arguments_are_typed_errors() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&logits());
    let t = tape.var(&unit_targets());
    let r = Reduction::Mean;

    // pos_weight が input に broadcast できない shape。
    let bad_pw = BceWithLogitsOptions::default()
        .pos_weight(Tensor::new(vec![1.0f32; 4], &[4]).expect("tensor"));
    assert!(x.bce_with_logits_loss_with(&t, r, &bad_pw).is_err());
    // 負の pos_weight。
    let neg_pw = BceWithLogitsOptions::default()
        .pos_weight(Tensor::new(vec![-1.0f32, 1.0, 1.0], &[3]).expect("tensor"));
    assert!(x.bce_with_logits_loss_with(&t, r, &neg_pw).is_err());

    // 負の eps・負の var。
    let var_ok = tape.var(&t23([1.0; 6]));
    let neg_eps = GaussianNllOptions::default().eps(-1.0);
    assert!(x.gaussian_nll_loss(&t, &var_ok, &neg_eps, r).is_err());
    let var_neg = tape.var(&t23([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    assert!(
        x.gaussian_nll_loss(&t, &var_neg, &GaussianNllOptions::default(), r)
            .is_err()
    );

    // p = 3・範囲外の target 添字・weight の shape 不一致。
    let tgt = Tensor::new(vec![2i32, 0], &[2]).expect("tensor");
    assert!(
        x.multi_margin_loss(&tgt, &MultiMarginOptions::default().p(3), r)
            .is_err()
    );
    let oob = Tensor::new(vec![2i32, 3], &[2]).expect("tensor");
    assert!(
        x.multi_margin_loss(&oob, &MultiMarginOptions::default(), r)
            .is_err()
    );
    let bad_w =
        MultiMarginOptions::default().weight(Tensor::new(vec![1.0f32; 2], &[2]).expect("tensor"));
    assert!(x.multi_margin_loss(&tgt, &bad_w, r).is_err());
    let bad_mlw = MultiLabelSoftMarginOptions::default()
        .weight(Tensor::new(vec![1.0f32; 2], &[2]).expect("tensor"));
    assert!(
        x.multilabel_soft_margin_loss(&unit_targets(), &bad_mlw, r)
            .is_err()
    );

    // 範囲外の alpha・負の gamma・範囲外の target。
    let bad_alpha = SigmoidFocalLossOptions::default().alpha(Some(1.5));
    assert!(
        x.sigmoid_focal_loss(&unit_targets(), &bad_alpha, r)
            .is_err()
    );
    let bad_gamma = SigmoidFocalLossOptions::default().gamma(-1.0);
    assert!(
        x.sigmoid_focal_loss(&unit_targets(), &bad_gamma, r)
            .is_err()
    );
    let bad_t = t23([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(
        x.sigmoid_focal_loss(&bad_t, &SigmoidFocalLossOptions::default(), r)
            .is_err()
    );
}
