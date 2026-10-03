//! `compile()` の `Loss` 追加 7 種（L1・Bce・BceWithLogits・Nll・KlDiv・Huber・SmoothL1。
//! イシュー #2509・親 #2500・ルート #2499）の統合テスト。
//!
//! 役割: `fit`／`evaluate` が、同じ初期値・同じバッチで `Var` の損失を直接使う手動学習ループ
//! （`bind → forward → 損失 → backward → Sgd::step → apply_parameters`）と bit 完全一致すること、
//! target dtype の不整合・metrics の可否・`Bce` の範囲外入力が型付きエラーで fail-closed になる
//! ことを検証する。L1 の比較側は内部クレート `fandhe_ai_autodiff::loss_ops::l1_loss` を直接呼ぶ
//! （facade 公開面には `loss_ops` を出さない保留を維持するため、結線は非 `pub` の use のみ）。
//!
//! 決定的にするため重み初期化・データは固定シード／閉形式で生成し、`shuffle=false`（既定）で
//! 学習する。実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use fandhe_ai::compat::{FitConfig, Loss, Metrics, Optimizer, Sequential};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};
use fandhe_ai_autodiff::Reduction;
use fandhe_ai_autodiff::loss_ops::l1_loss;

const N: usize = 8;
const D_IN: usize = 3;
const D_OUT: usize = 2;
const STEPS: usize = 4;
const LR: f32 = 0.1;

fn x_data() -> Tensor<f32> {
    let x: Vec<f32> = (0..N * D_IN)
        .map(|i| ((i as f32) * 0.41 + 0.3).sin())
        .collect();
    Tensor::new(x, &[N, D_IN]).expect("x")
}

fn y_regression() -> Tensor<f32> {
    let y: Vec<f32> = (0..N * D_OUT)
        .map(|i| ((i as f32) * 0.23 + 1.1).cos() * 2.0)
        .collect();
    Tensor::new(y, &[N, D_OUT]).expect("y")
}

/// `[0, 1]` の確率 target。
fn y_prob() -> Tensor<f32> {
    let y: Vec<f32> = (0..N * D_OUT)
        .map(|i| 0.5 + 0.4 * ((i as f32) * 0.7).sin())
        .collect();
    Tensor::new(y, &[N, D_OUT]).expect("y")
}

/// 各行の和が 1 の確率分布 target（KlDiv 用）。
fn y_dist() -> Tensor<f32> {
    let mut y = Vec::with_capacity(N * D_OUT);
    for i in 0..N {
        let p = 0.2 + 0.6 * (((i as f32) * 0.9).sin() * 0.5 + 0.5);
        y.push(p);
        y.push(1.0 - p);
    }
    Tensor::new(y, &[N, D_OUT]).expect("y")
}

fn y_class() -> Tensor<i32> {
    let y: Vec<i32> = (0..N as i32).map(|i| i % D_OUT as i32).collect();
    Tensor::new(y, &[N]).expect("y")
}

#[derive(Clone, Copy)]
enum Head {
    Linear,
    Sigmoid,
    LogSoftmax,
}

fn build(head: Head) -> Sequential {
    let m = Sequential::new()
        .add_linear(D_IN, D_OUT, 7)
        .expect("構築できるはず");
    match head {
        Head::Linear => m,
        Head::Sigmoid => m.add_sigmoid(),
        Head::LogSoftmax => m.add_log_softmax(1),
    }
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn param_bits(m: &Sequential) -> Vec<Vec<u32>> {
    m.trainable_parameters()
        .iter()
        .map(|t| f32_bits(t.as_slice().expect("cpu")))
        .collect()
}

/// 手動ループ。`loss_fn` は `Var` の損失を直接呼ぶ。返り値は (loss 系列, 最終パラメータ)。
fn manual_fit<F>(head: Head, loss_fn: F) -> (Vec<f32>, Vec<Vec<u32>>)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let x = x_data();
    let mut model = build(head);
    let mut sgd = Sgd::new(SgdConfig::new(LR)).expect("sgd");
    let mut losses = Vec::new();
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x_var = tape.var(&x);
            let pred = bound.forward(&tape, &x_var).expect("forward");
            let loss = loss_fn(&tape, &pred).expect("loss");
            losses.push(loss.to_tensor().get(&[]).expect("scalar"));
            let grads = tape.backward(&loss).expect("backward");
            let grad_refs = bound.trainable_grads(&grads).expect("grads");
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).expect("step")
        };
        model.apply_parameters(updated).expect("apply");
    }
    let bits = param_bits(&model);
    (losses, bits)
}

/// `fit`（f32 target）と手動ループが bit 一致し、`evaluate` が直接計算と bit 一致する。
fn check_f32<F>(loss: Loss, head: Head, y: Tensor<f32>, loss_fn: F)
where
    F: for<'t> Fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let x = x_data();
    let (m_losses, m_params) = manual_fit(head, |tape, pred| loss_fn(pred, &tape.var_no_grad(&y)));
    let mut model = build(head);
    model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), loss)
        .expect("compile");
    let h = model.fit(&x, &y, FitConfig::new(STEPS, N)).expect("fit");
    assert_eq!(f32_bits(&h.loss), f32_bits(&m_losses), "{loss:?} loss 系列");
    assert_eq!(param_bits(&model), m_params, "{loss:?} パラメータ");

    // evaluate: 更新後モデルで同じバッチの損失を直接計算した値と bit 一致する。
    let ev = model.evaluate(&x, &y, N).expect("evaluate");
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let pred = bound.forward(&tape, &tape.var(&x)).expect("forward");
    let direct = loss_fn(&pred, &tape.var_no_grad(&y))
        .expect("loss")
        .to_tensor()
        .get(&[])
        .expect("scalar");
    assert_eq!(ev.to_bits(), direct.to_bits(), "{loss:?} evaluate");
}

#[test]
fn f32_target_losses_match_manual_loop_bit_exact() {
    let m = Reduction::Mean;
    check_f32(Loss::L1, Head::Linear, y_regression(), |p, t| {
        l1_loss(p, t, m)
    });
    check_f32(Loss::Huber, Head::Linear, y_regression(), |p, t| {
        p.huber_loss(t, 1.0, m)
    });
    check_f32(Loss::SmoothL1, Head::Linear, y_regression(), |p, t| {
        p.smooth_l1_loss(t, 1.0, m)
    });
    check_f32(Loss::BceWithLogits, Head::Linear, y_prob(), |p, t| {
        p.bce_with_logits_loss(t, m)
    });
    check_f32(Loss::Bce, Head::Sigmoid, y_prob(), |p, t| p.bce_loss(t, m));
    check_f32(Loss::KlDiv, Head::LogSoftmax, y_dist(), |p, t| {
        p.kl_div_loss(t, m)
    });
}

#[test]
fn nll_matches_manual_loop_bit_exact() {
    let x = x_data();
    let y = y_class();
    let (m_losses, m_params) =
        manual_fit(Head::LogSoftmax, |_, p| p.nll_loss(&y, 1, Reduction::Mean));
    let mut model = build(Head::LogSoftmax);
    model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Nll)
        .expect("compile");
    let h = model.fit(&x, &y, FitConfig::new(STEPS, N)).expect("fit");
    assert_eq!(f32_bits(&h.loss), f32_bits(&m_losses));
    assert_eq!(param_bits(&model), m_params);

    let ev = model.evaluate(&x, &y, N).expect("evaluate");
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let pred = bound.forward(&tape, &tape.var(&x)).expect("forward");
    let direct = pred
        .nll_loss(&y, 1, Reduction::Mean)
        .expect("loss")
        .to_tensor()
        .get(&[])
        .expect("scalar");
    assert_eq!(ev.to_bits(), direct.to_bits());
}

fn compiled(loss: Loss, head: Head) -> Sequential {
    let mut m = build(head);
    m.compile(Optimizer::Sgd(SgdConfig::new(LR)), loss)
        .expect("compile");
    m
}

const F32_LOSSES: [Loss; 7] = [
    Loss::Mse,
    Loss::L1,
    Loss::Bce,
    Loss::BceWithLogits,
    Loss::KlDiv,
    Loss::Huber,
    Loss::SmoothL1,
];

#[test]
fn dtype_mismatch_is_invalid_argument() {
    let x = x_data();
    // Nll × f32 target
    let mut m = compiled(Loss::Nll, Head::LogSoftmax);
    let yf = y_dist();
    assert!(matches!(
        m.fit(&x, &yf, FitConfig::new(1, N)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        m.evaluate(&x, &yf, N),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // f32 系 7 種 × i32 target
    let yi = y_class();
    for loss in F32_LOSSES {
        let mut m = compiled(loss, Head::Linear);
        assert!(
            matches!(
                m.fit(&x, &yi, FitConfig::new(1, N)),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "{loss:?} fit"
        );
        assert!(
            matches!(
                m.evaluate(&x, &yi, N),
                Err(AutodiffError::InvalidArgument(_))
            ),
            "{loss:?} evaluate"
        );
    }
}

#[test]
fn metrics_are_allowed_for_nll_and_rejected_for_f32_losses() {
    let x = x_data();
    let yi = y_class();
    let mut m = compiled(Loss::Nll, Head::LogSoftmax);
    m.fit_with_metrics(
        &x,
        &yi,
        FitConfig::new(1, N),
        Some((&x, &yi)),
        &mut [],
        &[Metrics::Accuracy],
    )
    .expect("Nll × i32 は metrics 可");

    let y = y_prob();
    for loss in [Loss::Bce, Loss::L1, Loss::Huber] {
        let mut m = compiled(loss, Head::Sigmoid);
        let r = m.fit_with_metrics(
            &x,
            &y,
            FitConfig::new(1, N),
            Some((&x, &y)),
            &mut [],
            &[Metrics::Accuracy],
        );
        assert!(
            matches!(r, Err(AutodiffError::InvalidArgument(_))),
            "{loss:?}"
        );
    }
}

#[test]
fn bce_out_of_range_pred_is_error_not_panic() {
    // Sigmoid なし: linear 出力は [0,1] を外れうる。
    let x = x_data();
    let mut m = compiled(Loss::Bce, Head::Linear);
    let r = m.fit(&x, &y_prob(), FitConfig::new(1, N));
    assert!(r.is_err());
}
