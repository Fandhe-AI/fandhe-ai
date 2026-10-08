//! `Var::hinge_embedding_loss`／`Var::soft_margin_loss`／`Var::multilabel_margin_loss` の
//! facade 利用テスト（イシュー #2677 の承認形。ルート #2499 の 2026-10-07 承認コメント
//! `issuecomment-6033824965`・`docs/compat-api-scope.md` §5.1 行 19・20）。
//!
//! 利用者が見る面は `fandhe_ai` だけ（`Reduction` は `fandhe_ai::nn::loss::Reduction` の 1 経路）。
//! 委譲メソッドの forward 値と backward（勾配）が、内部クレートの自由関数
//! （`fandhe_ai_autodiff::{elementwise_loss_ops, margin_focal_loss_ops}`。比較の参照としてのみ使う）と
//! bit 一致することを固定する。委譲先は同一コードのため新しい tolerance・baseline は設けない
//! （グローバル RNG は使わず、入力は固定値）。実機（CUDA／Metal）の一致は既存の
//! `elementwise_loss_ops_backend_parity.rs`・`margin_focal_loss_ops_backend_parity.rs` が担う。

use fandhe_ai::nn::loss::Reduction;
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
