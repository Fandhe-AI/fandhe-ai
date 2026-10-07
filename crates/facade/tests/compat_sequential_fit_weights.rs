//! `FitWeights`・`Sequential::fit_with_weights`・`FitConfig::validation_split`
//! （イシュー #2564・親 #2562。公開形は `docs/compat-fit-sample-weighting-decision.md`
//! §11.1）の facade 経由の統合テスト。
//!
//! 検証の柱: (1) 重み既定・分割なしで既存 `fit_with_metrics` と bit 完全一致、
//! (2) 重み付き損失が手計算式 `Σ w_i·l_i / N_batch` と一致（REQ-2 統一複合判定。
//! 定数は新設・変更しない）、(3) `validation_split` の分割位置と `evaluate` 一致、
//! (4) 式が未定義の組み合わせ・不正入力の fail-closed 拒否（パラメータ不変・
//! モード復元・compile 維持）。新規 `Op`／カーネルを持たないため実機依存テストは
//! なし（`#[ignore]` 分離なし）。
//!
//! `shuffle(true)` やグローバル RNG を消費するテストは `compat_sequential_fit.rs` と
//! 同型の `test_lock` で直列化する。

use std::collections::HashMap;
use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, EarlyStopping, FitConfig, FitWeights, Loss, Monitor, Optimizer,
    Sequential,
};
use fandhe_ai::optim::{LbfgsConfig, SgdConfig};
use fandhe_ai::{AutodiffError, Tensor};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

const N: usize = 12;
const D_IN: usize = 4;
const C: usize = 3;

fn cls_data() -> (Tensor<f32>, Tensor<i32>) {
    let mut rng = Xorshift64Star::new(0xFEED);
    let x = Tensor::new(rng.fill_vec(N * D_IN), &[N, D_IN]).unwrap();
    let y = Tensor::new((0..N).map(|i| (i % C) as i32).collect(), &[N]).unwrap();
    (x, y)
}

fn reg_data() -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(0xBEEF);
    let x = Tensor::new(rng.fill_vec(N * D_IN), &[N, D_IN]).unwrap();
    let y = Tensor::new(rng.fill_vec(N * 2), &[N, 2]).unwrap();
    (x, y)
}

fn cls_model(loss: Loss) -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, 6, 11)
        .unwrap()
        .add_relu()
        .add_linear(6, C, 12)
        .unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), loss)
        .unwrap();
    m
}

fn reg_model() -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, 6, 21)
        .unwrap()
        .add_relu()
        .add_linear(6, 2, 22)
        .unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    m
}

fn bits(m: &Sequential) -> Vec<Vec<u32>> {
    m.trainable_parameters()
        .iter()
        .map(|t| t.host_slice().iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// REQ-2 統一複合判定（相対 1e-3 未満または絶対 1e-5 未満）。
fn close(a: f32, b: f32) -> bool {
    let d = (a - b).abs();
    d < 1e-5 || d <= 1e-3 * a.abs().max(b.abs())
}

fn all_close(a: &[Vec<u32>], b: &[Vec<u32>]) -> bool {
    a.iter().zip(b).all(|(x, y)| {
        x.iter()
            .zip(y)
            .all(|(p, q)| close(f32::from_bits(*p), f32::from_bits(*q)))
    })
}

// ---------------------------------------------------------------------
// 1. 重み既定は既存経路と bit 完全一致
// ---------------------------------------------------------------------

#[test]
fn default_weights_match_fit_with_metrics_bit_exact() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(FitWeights::new(), FitWeights::default());
    for shuffle in [false, true] {
        let cfg = FitConfig::new(3, 5).shuffle(shuffle);
        let (x, y) = cls_data();
        let (mut a, mut b) = (cls_model(Loss::CrossEntropy), cls_model(Loss::CrossEntropy));
        fandhe_ai::manual_seed(7);
        let ha = a
            .fit_with_metrics(&x, &y, cfg, Some((&x, &y)), &mut [], &[])
            .unwrap();
        fandhe_ai::manual_seed(7);
        let hb = b
            .fit_with_weights(
                &x,
                &y,
                cfg,
                &FitWeights::default(),
                Some((&x, &y)),
                &mut [],
                &[],
            )
            .unwrap();
        assert_eq!(ha, hb);
        assert_eq!(bits(&a), bits(&b));

        let (x, y) = reg_data();
        let (mut a, mut b) = (reg_model(), reg_model());
        fandhe_ai::manual_seed(7);
        let ha = a.fit(&x, &y, cfg).unwrap();
        fandhe_ai::manual_seed(7);
        let hb = b
            .fit_with_weights(&x, &y, cfg, &FitWeights::new(), None, &mut [], &[])
            .unwrap();
        assert_eq!(ha, hb);
        assert_eq!(bits(&a), bits(&b));
    }
}

// ---------------------------------------------------------------------
// 2. 重み付き損失の手計算一致
// ---------------------------------------------------------------------

fn host_log_softmax_nll(logits: &Tensor<f32>, y: &[i32]) -> Vec<f32> {
    let l = logits.host_slice();
    (0..y.len())
        .map(|i| {
            let row = &l[i * C..(i + 1) * C];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + row.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
            lse - row[y[i] as usize]
        })
        .collect()
}

#[test]
fn class_and_sample_weights_match_hand_computed_loss() {
    let (x, y) = cls_data();
    let mut model = cls_model(Loss::CrossEntropy);
    let nll = host_log_softmax_nll(&model.predict(&x).unwrap(), y.host_slice().as_ref());
    let sw: Vec<f32> = (0..N).map(|i| 0.5 + i as f32 * 0.25).collect();
    let cw = HashMap::from([(1u32, 3.0f32), (2u32, 0.0f32)]);
    let ys = y.host_slice();
    let expected: f32 = (0..N)
        .map(|i| sw[i] * cw.get(&(ys[i] as u32)).copied().unwrap_or(1.0) * nll[i])
        .sum::<f32>()
        / N as f32;
    let weights = FitWeights::new().class_weight(cw).sample_weight(&sw);
    let h = model
        .fit_with_weights(&x, &y, FitConfig::new(1, N), &weights, None, &mut [], &[])
        .unwrap();
    assert!(close(h.loss[0], expected), "{} vs {expected}", h.loss[0]);
    assert!(h.val_loss.is_empty());
}

#[test]
fn sample_weight_scaling_and_zero() {
    let (x, y) = cls_data();
    let base = cls_model(Loss::CrossEntropy)
        .fit(&x, &y, FitConfig::new(1, N))
        .unwrap()
        .loss[0];
    let twos = [2.0f32; N];
    let h2 = cls_model(Loss::CrossEntropy)
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, N),
            &FitWeights::new().sample_weight(&twos),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(close(h2.loss[0], 2.0 * base));
    let ones = [1.0f32; N];
    let h1 = cls_model(Loss::CrossEntropy)
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, N),
            &FitWeights::new().sample_weight(&ones),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(close(h1.loss[0], base));

    // 全 0 重み: loss 0・SGD でパラメータ不変。
    let zeros = [0.0f32; N];
    let mut m = cls_model(Loss::CrossEntropy);
    let before = bits(&m);
    let hz = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(2, 4),
            &FitWeights::new().sample_weight(&zeros),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(hz.loss.iter().all(|l| *l == 0.0));
    assert_eq!(bits(&m), before);
}

#[test]
fn mse_sample_weight_matches_unweighted_scaling() {
    let (x, y) = reg_data();
    let base = reg_model().fit(&x, &y, FitConfig::new(1, N)).unwrap().loss[0];
    let ones = [1.0f32; N];
    let twos = [2.0f32; N];
    let run = |w: &[f32]| {
        reg_model()
            .fit_with_weights(
                &x,
                &y,
                FitConfig::new(1, N),
                &FitWeights::new().sample_weight(w),
                None,
                &mut [],
                &[],
            )
            .unwrap()
            .loss[0]
    };
    assert!(close(run(&ones), base));
    assert!(close(run(&twos), 2.0 * base));
}

#[test]
fn weights_work_with_accumulation_and_amp() {
    let (x, y) = cls_data();
    let sw = [1.0f32; N];
    let w = FitWeights::new().sample_weight(&sw);
    let mut m = cls_model(Loss::CrossEntropy);
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(2, 3).accumulate_steps(2),
            &w,
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(h.loss.iter().all(|l| l.is_finite()));

    let mut m = Sequential::new().add_linear(D_IN, C, 5).unwrap();
    m.compile_with_amp(
        Optimizer::Sgd(SgdConfig::new(0.1)),
        Loss::CrossEntropy,
        AmpConfig::new(AmpDType::F16),
    )
    .unwrap();
    let h = m
        .fit_with_weights(&x, &y, FitConfig::new(1, 4), &w, None, &mut [], &[])
        .unwrap();
    assert!(h.loss.iter().all(|l| l.is_finite()));
}

#[test]
fn sample_weight_does_not_change_shuffle_order() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (x, y) = cls_data();
    let ones = [1.0f32; N];
    let cfg = FitConfig::new(2, 5).shuffle(true);
    let (mut a, mut b) = (cls_model(Loss::CrossEntropy), cls_model(Loss::CrossEntropy));
    fandhe_ai::manual_seed(99);
    a.fit(&x, &y, cfg).unwrap();
    fandhe_ai::manual_seed(99);
    b.fit_with_weights(
        &x,
        &y,
        cfg,
        &FitWeights::new().sample_weight(&ones),
        None,
        &mut [],
        &[],
    )
    .unwrap();
    // 重み 1.0 の重み付き経路は数値的にほぼ一致（バッチ順が違えば大きくずれる）。
    assert!(all_close(&bits(&a), &bits(&b)));
}

// ---------------------------------------------------------------------
// 3. validation_split
// ---------------------------------------------------------------------

#[test]
fn validation_split_matches_explicit_tail_validation() {
    let (x, y) = cls_data();
    let at = (N as f64 * (1.0 - 0.25)).floor() as usize; // 9
    let (xt, yt) = (x.narrow(0, 0, at).unwrap(), y.narrow(0, 0, at).unwrap());
    let (xv, yv) = (
        x.narrow(0, at, N - at).unwrap(),
        y.narrow(0, at, N - at).unwrap(),
    );
    let (mut a, mut b) = (cls_model(Loss::CrossEntropy), cls_model(Loss::CrossEntropy));
    let ha = a
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(3, 4).validation_split(0.25),
            &FitWeights::new(),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    let hb = b
        .fit_with_metrics(
            &xt,
            &yt,
            FitConfig::new(3, 4),
            Some((&xv, &yv)),
            &mut [],
            &[],
        )
        .unwrap();
    assert_eq!(ha, hb);
    assert_eq!(ha.val_loss.len(), 3);
    assert_eq!(bits(&a), bits(&b));
    // val_loss は evaluate（重みなし）と一致。
    let ev = a.evaluate(&xv, &yv, 4).unwrap();
    assert_eq!(*ha.val_loss.last().unwrap(), ev);
}

#[test]
fn validation_split_is_effective_for_every_entry_and_enables_val_monitors() {
    let (x, y) = cls_data();
    let cfg = FitConfig::new(2, 4).validation_split(0.25);
    let h = cls_model(Loss::CrossEntropy).fit(&x, &y, cfg).unwrap();
    assert_eq!(h.val_loss.len(), 2);
    // Monitor::ValLoss を要求する callback も満たされる。
    let mut cbs = [Callback::EarlyStopping(
        EarlyStopping::new(5).monitor(Monitor::ValLoss),
    )];
    cls_model(Loss::CrossEntropy)
        .fit_with_callbacks(&x, &y, cfg, None, &mut cbs)
        .unwrap();
    // 0.0 / -0.0 は分割なし（val_loss は空・ValLoss 監視は従来どおり拒否）。
    for z in [0.0f32, -0.0] {
        let cfg0 = FitConfig::new(2, 4).validation_split(z);
        let h = cls_model(Loss::CrossEntropy).fit(&x, &y, cfg0).unwrap();
        assert!(h.val_loss.is_empty());
        let mut cbs = [Callback::EarlyStopping(
            EarlyStopping::new(5).monitor(Monitor::ValLoss),
        )];
        let err = cls_model(Loss::CrossEntropy)
            .fit_with_callbacks(&x, &y, cfg0, None, &mut cbs)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
    assert_ne!(
        FitConfig::new(2, 4).validation_split(0.0),
        FitConfig::new(2, 4)
    );
}

#[test]
fn validation_split_applies_sample_weight_to_train_side_only() {
    let (x, y) = cls_data();
    let at = 9usize;
    let sw: Vec<f32> = (0..N).map(|i| 1.0 + i as f32).collect();
    let (xt, yt) = (x.narrow(0, 0, at).unwrap(), y.narrow(0, 0, at).unwrap());
    let (mut a, mut b) = (cls_model(Loss::CrossEntropy), cls_model(Loss::CrossEntropy));
    let ha = a
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(2, 3).validation_split(0.25),
            &FitWeights::new().sample_weight(&sw),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    let hb = b
        .fit_with_weights(
            &xt,
            &yt,
            FitConfig::new(2, 3),
            &FitWeights::new().sample_weight(&sw[..at]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert_eq!(ha.loss, hb.loss);
    assert_eq!(bits(&a), bits(&b));
}

// ---------------------------------------------------------------------
// 4. fail-closed 拒否（パラメータ不変・モード復元・compile 維持）
// ---------------------------------------------------------------------

fn assert_rejected(
    model: &mut Sequential,
    r: Result<fandhe_ai::compat::History, AutodiffError>,
    before: &[Vec<u32>],
    what: &str,
) {
    match r {
        Err(AutodiffError::InvalidArgument(_)) => {}
        other => panic!("{what}: InvalidArgument のはずが {other:?}"),
    }
    assert_eq!(bits(model), before, "{what}: パラメータが変化した");
    assert!(model.is_compiled(), "{what}");
}

#[test]
fn invalid_inputs_are_rejected_without_side_effects() {
    let (x, y) = cls_data();
    let cfg = FitConfig::new(1, 4);
    let mut m = cls_model(Loss::CrossEntropy);
    let before = bits(&m);
    let mode = m.training();

    let short = [1.0f32; N - 1];
    let neg = {
        let mut v = [1.0f32; N];
        v[3] = -0.1;
        v
    };
    let nan = {
        let mut v = [1.0f32; N];
        v[0] = f32::NAN;
        v
    };
    let inf = {
        let mut v = [1.0f32; N];
        v[5] = f32::INFINITY;
        v
    };
    for (what, w) in [
        ("len", &short[..]),
        ("neg", &neg),
        ("nan", &nan),
        ("inf", &inf),
    ] {
        let r = m.fit_with_weights(
            &x,
            &y,
            cfg,
            &FitWeights::new().sample_weight(w),
            None,
            &mut [],
            &[],
        );
        assert_rejected(&mut m, r, &before, what);
    }
    for (what, cw) in [
        ("cw neg", HashMap::from([(0u32, -1.0f32)])),
        ("cw nan", HashMap::from([(0u32, f32::NAN)])),
        ("cw inf", HashMap::from([(1u32, f32::INFINITY)])),
        ("cw key", HashMap::from([(C as u32, 1.0f32)])),
    ] {
        let r = m.fit_with_weights(
            &x,
            &y,
            cfg,
            &FitWeights::new().class_weight(cw),
            None,
            &mut [],
            &[],
        );
        assert_rejected(&mut m, r, &before, what);
    }
    // validation_split の不正値・併用。
    for (what, s) in [
        ("nan", f32::NAN),
        ("inf", f32::INFINITY),
        ("neg", -0.1),
        ("one", 1.0),
        ("gt", 1.5),
        ("all", 0.999_999),
    ] {
        let r = m.fit_with_weights(
            &x,
            &y,
            cfg.validation_split(s),
            &FitWeights::new(),
            None,
            &mut [],
            &[],
        );
        assert_rejected(&mut m, r, &before, what);
    }
    let r = m.fit_with_weights(
        &x,
        &y,
        cfg.validation_split(0.25),
        &FitWeights::new(),
        Some((&x, &y)),
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "split+explicit");
    assert_eq!(m.training(), mode);

    // target 範囲外（バッチ内で検出。更新前）。
    let bad_y = Tensor::new(
        (0..N).map(|i| if i == 7 { C as i32 } else { 0 }).collect(),
        &[N],
    )
    .unwrap();
    let sw = [1.0f32; N];
    let r = m.fit_with_weights(
        &x,
        &bad_y,
        FitConfig::new(1, N),
        &FitWeights::new().sample_weight(&sw),
        None,
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "target range");
    assert_eq!(m.training(), mode);
}

#[test]
fn undefined_loss_and_optimizer_combinations_are_rejected() {
    let (x, y) = cls_data();
    let sw = [1.0f32; N];
    let cw = || HashMap::from([(0u32, 2.0f32)]);

    // class_weight × Nll／sample_weight × Nll。
    let mut m = cls_model(Loss::Nll);
    let before = bits(&m);
    for w in [
        FitWeights::new().class_weight(cw()),
        FitWeights::new().sample_weight(&sw),
    ] {
        let r = m.fit_with_weights(&x, &y, FitConfig::new(1, 4), &w, None, &mut [], &[]);
        assert_rejected(&mut m, r, &before, "nll");
    }

    // class_weight × Mse、sample_weight × 7 種の Loss。
    let (xr, yr) = reg_data();
    let mut m = reg_model();
    let before = bits(&m);
    let r = m.fit_with_weights(
        &xr,
        &yr,
        FitConfig::new(1, 4),
        &FitWeights::new().class_weight(cw()),
        None,
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "mse class");
    for loss in [
        Loss::L1,
        Loss::Bce,
        Loss::BceWithLogits,
        Loss::KlDiv,
        Loss::Huber,
        Loss::SmoothL1,
    ] {
        let mut m = Sequential::new().add_linear(D_IN, 2, 3).unwrap();
        m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), loss)
            .unwrap();
        let before = bits(&m);
        let r = m.fit_with_weights(
            &xr,
            &yr,
            FitConfig::new(1, 4),
            &FitWeights::new().sample_weight(&sw),
            None,
            &mut [],
            &[],
        );
        assert_rejected(&mut m, r, &before, &format!("{loss:?}"));
        // validation_split 単独は損失に依存せず有効（Bce 等は入力値域の都合で別エラーに
        // なるため、値域制約のない損失だけで確認する）。
        if !matches!(loss, Loss::L1 | Loss::Huber | Loss::SmoothL1) {
            continue;
        }
        let h = m.fit_with_weights(
            &xr,
            &yr,
            FitConfig::new(1, 4).validation_split(0.25),
            &FitWeights::new(),
            None,
            &mut [],
            &[],
        );
        assert!(h.is_ok(), "{loss:?}: {h:?}");
    }

    // Lbfgs: 非既定の重みは拒否・validation_split 単独は有効。
    let mut m = Sequential::new().add_linear(D_IN, C, 3).unwrap();
    m.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::CrossEntropy)
        .unwrap();
    let before = bits(&m);
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, N),
        &FitWeights::new().sample_weight(&sw),
        None,
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "lbfgs");
    assert!(
        m.fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, N).validation_split(0.25),
            &FitWeights::new(),
            None,
            &mut [],
            &[],
        )
        .is_ok()
    );
}
