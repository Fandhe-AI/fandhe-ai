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
//! グローバル RNG（`manual_seed`・shuffle・層初期化）を共有するため、本ファイルの全テストが
//! `test_lock` で直列化する（`compat_sequential_fit.rs` と同型。ロック漏れの shuffle テストが
//! bit 一致テストの RNG 系列を食う不安定を防ぐ）。

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

/// REQ-2 統一複合判定。許容誤差の直書きを避け、`backend-cpu` の `parity::compare`
/// （f64 計算・厳密 `<`・非有限は不合格）へ委譲する。
fn close(a: f32, b: f32) -> bool {
    fandhe_ai_backend_cpu::parity::compare(&[a], &[b])
        .map(|r| r.passes())
        .unwrap_or(false)
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
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
// ---------------------------------------------------------------------
// 5. レビュー指摘の回帰（#2823）
// ---------------------------------------------------------------------

/// 後続バッチの不正 target は、先行バッチの更新より前に拒否される（パラメータ不変）。
#[test]
fn late_batch_invalid_target_is_rejected_before_any_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (x, mut y) = {
        let (x, y) = cls_data();
        (x, y.host_slice().to_vec())
    };
    y[N - 1] = C as i32; // 最終バッチ（batch_size 4・shuffle なし）に範囲外
    let y = Tensor::new(y, &[N]).unwrap();
    let mut m = cls_model(Loss::CrossEntropy);
    let before = bits(&m);
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 4),
        &FitWeights::new().class_weight(HashMap::from([(1u32, 2.0f32)])),
        None,
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "late invalid target");
}

/// sample 1e19 × class 1e20 の積が f32 で overflow せず（f64 で積・除算してから f32 化）、損失が有限。
#[test]
fn huge_weight_product_does_not_overflow_in_f32() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (x, y) = cls_data();
    let mut m = cls_model(Loss::CrossEntropy);
    let mut sw = [0.0f32; N];
    sw[0] = 1e19; // target 0 のサンプルのみ（1e19 × 1e20 = 1e39 は f32 では overflow）
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, N),
            &FitWeights::new()
                .sample_weight(&sw)
                .class_weight(HashMap::from([(0u32, 1e20f32)])),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(h.loss[0].is_finite(), "loss = {}", h.loss[0]);
}

/// 極端な logits でも非 target クラスの `0 × -inf` が NaN を作らない。
#[test]
fn extreme_logits_do_not_produce_nan_loss() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::CrossEntropy)
        .unwrap();
    let x = Tensor::new(vec![f32::MAX], &[1, 1]).unwrap();
    let y = Tensor::new(vec![0i32], &[1]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 1),
            &FitWeights::new().class_weight(HashMap::from([(0u32, 1.0f32)])),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(!h.loss[0].is_nan(), "loss = {}", h.loss[0]);
}

/// 重み 0 のサンプルは `d²` が f32 で overflow しても `inf × 0 = NaN` にならず除外される。
#[test]
fn zero_weight_sample_with_overflowing_square_is_excluded() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 1, 7).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();
    let x = Tensor::new(vec![0.0f32, 0.0], &[2, 1]).unwrap();
    let y = Tensor::new(vec![1e20f32, 0.0], &[2, 1]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 2),
            &FitWeights::new().sample_weight(&[0.0, 1.0]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(h.loss[0].is_finite(), "loss = {}", h.loss[0]);
}

/// CrossEntropy でも重み 0 のサンプルの `-inf` log_softmax が NaN を作らない。
#[test]
fn zero_weight_sample_with_infinite_log_softmax_is_excluded() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::CrossEntropy)
        .unwrap();
    let x = Tensor::new(vec![f32::MAX, 0.0], &[2, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 0], &[2]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 2),
            &FitWeights::new().sample_weight(&[0.0, 1.0]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(!h.loss[0].is_nan(), "loss = {}", h.loss[0]);
}

/// 係数（sample × class / N）が f32 で表現できない場合は更新前に拒否する。
#[test]
fn unrepresentable_coefficient_is_rejected_before_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let before = bits(&m);
    let x = Tensor::new(vec![1.0f32], &[1, 1]).unwrap();
    let y = Tensor::new(vec![0i32], &[1]).unwrap();
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 1),
        &FitWeights::new()
            .sample_weight(&[1e20])
            .class_weight(HashMap::from([(0u32, 1e20f32)])),
        None,
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "unrepresentable coef");
}

/// 後続バッチだけが表現不能な係数になる場合も、先行バッチの更新前に拒否する（レビュー指摘 #2823）。
#[test]
fn unrepresentable_coefficient_in_late_batch_is_rejected_before_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let before = bits(&m);
    let x = Tensor::new(vec![1.0f32, 1.0], &[2, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 0], &[2]).unwrap();
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 1),
        &FitWeights::new()
            .sample_weight(&[1.0, 1e20])
            .class_weight(HashMap::from([(0u32, 1e20f32)])),
        None,
        &mut [],
        &[],
    );
    assert_rejected(&mut m, r, &before, "late unrepresentable coef");
}

/// ゼロ重みの行の logits が inf でも勾配・パラメータが NaN にならない。
#[test]
fn zero_weight_row_with_infinite_logits_keeps_params_finite() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let x = Tensor::new(vec![f32::MAX, 1.0], &[2, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 1], &[2]).unwrap();
    m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 2),
        &FitWeights::new().sample_weight(&[0.0, 1.0]),
        None,
        &mut [],
        &[],
    )
    .unwrap();
    for p in m.trainable_parameters() {
        assert!(
            p.host_slice().iter().all(|v| v.is_finite()),
            "NaN/inf param"
        );
    }
}

/// 差分の二乗が f32 で overflow しても、最終結果が有限なら loss は有限（小さい重み）。
#[test]
fn small_weight_with_large_residual_does_not_overflow() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 1, 7).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();
    let x = Tensor::new(vec![0.0f32], &[1, 1]).unwrap();
    let y = Tensor::new(vec![1e20f32], &[1, 1]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 1),
            &FitWeights::new().sample_weight(&[1e-20]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(
        h.loss[0].is_finite() && h.loss[0] > 0.0,
        "loss = {}",
        h.loss[0]
    );
}

/// shuffle なしでは各サンプルの実バッチサイズで係数を検証する（N=3・batch=2 では先頭バッチの
/// 実サイズは 2。最小バッチサイズ 1 で一律検査すると有限な入力を誤拒否する。レビュー指摘 #2823）。
#[test]
fn no_shuffle_validates_coefficients_with_actual_batch_size() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::CrossEntropy)
        .unwrap();
    let x = Tensor::new(vec![0.0f32, 0.0, 0.0], &[3, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 0, 0], &[3]).unwrap();
    let w = FitWeights::new()
        .sample_weight(&[1e19, 0.0, 0.0])
        .class_weight(HashMap::from([(0u32, 4e19f32)]));
    // shuffle なし: サンプル 0 は実バッチサイズ 2 → 係数 2.5e38 で有限。
    m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 2).shuffle(false),
        &w,
        None,
        &mut [],
        &[],
    )
    .unwrap();
    // shuffle あり: 最小バッチサイズ 1 で評価するため 5e38 が f32 で表現不能 → 拒否。
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 2).shuffle(true),
        &w,
        None,
        &mut [],
        &[],
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
}

/// 重み付き MSE の `pred − target` 自体が f32 で overflow する入力でも loss が有限。
#[test]
fn weighted_mse_subtraction_does_not_overflow() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 1, 7).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();
    let x = Tensor::new(vec![0.0f32], &[1, 1]).unwrap();
    let y = Tensor::new(vec![-3e38f32], &[1, 1]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 1),
            &FitWeights::new().sample_weight(&[1e-40]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(h.loss[0].is_finite(), "loss = {}", h.loss[0]);
}

/// ゼロ重み行の内部層 overflow が非有限勾配になっても、パラメータは汚染されない
/// （更新前に拒否されるか、有限のまま更新される）。
#[test]
fn zero_weight_row_with_internal_overflow_never_pollutes_params() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new()
        .add_linear(1, 4, 5)
        .unwrap()
        .add_linear(4, 2, 6)
        .unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let x = Tensor::new(vec![f32::MAX, 1.0], &[2, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 1], &[2]).unwrap();
    let _ = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 2),
        &FitWeights::new().sample_weight(&[0.0, 1.0]),
        None,
        &mut [],
        &[],
    );
    for p in m.trainable_parameters() {
        assert!(p.host_slice().iter().all(|v| v.is_finite()), "polluted");
    }
}

/// 正の CE 重みが f32 でゼロへアンダーフローする場合は、ゼロ重み扱いにせず更新前に拒否する。
#[test]
fn positive_ce_weight_underflow_is_rejected() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let x = Tensor::new(vec![1.0f32, 1.0], &[2, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 1], &[2]).unwrap();
    let w = FitWeights::new()
        .sample_weight(&[1e-30, 1.0])
        .class_weight(HashMap::from([(0u32, 1e-30f32)]));
    let r = m.fit_with_weights(&x, &y, FitConfig::new(1, 2), &w, None, &mut [], &[]);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
}

/// 残差を先に求めるため、`s_i > 1` で pred・target がともに大きく残差 0 でも NaN にならない。
#[test]
fn weighted_mse_large_equal_operands_and_scale_above_one_is_finite() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 1, 7).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();
    let x = Tensor::new(vec![0.0f32], &[1, 1]).unwrap();
    let pred = m.predict(&x).unwrap();
    let y = Tensor::new(pred.host_slice().to_vec(), &[1, 1]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 1),
            &FitWeights::new().sample_weight(&[1e30]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert_eq!(h.loss[0], 0.0);
}

/// shuffle=true の事前係数検査は最小バッチサイズ（overflow）だけでなく最大バッチサイズ
/// （ゼロへのアンダーフロー）でも行い、更新前に拒否する（レビュー指摘 #2823）。
#[test]
fn shuffle_ce_underflow_at_max_batch_size_is_rejected_before_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let before: Vec<Vec<f32>> = m
        .trainable_parameters()
        .iter()
        .map(|p| p.host_slice().to_vec())
        .collect();
    let x = Tensor::new(vec![1.0f32; 5], &[5, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 1, 0, 1, 0], &[5]).unwrap();
    // N=5・batch_size=2 → バッチサイズ {2, 1}。1.4e-45 / 2 は f32 でゼロへ丸まる。
    let sw = [f32::from_bits(1); 5];
    let w = FitWeights::new().sample_weight(&sw);
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 2).shuffle(true),
        &w,
        None,
        &mut [],
        &[],
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
    let after: Vec<Vec<f32>> = m
        .trainable_parameters()
        .iter()
        .map(|p| p.host_slice().to_vec())
        .collect();
    assert_eq!(before, after, "更新前に拒否されること");
}

/// 別行の残差 overflow が、有限な行（pred=target・係数大）の計算を NaN にしない。
#[test]
fn weighted_mse_overflow_row_does_not_poison_finite_row() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 1, 7).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.0)), Loss::Mse)
        .unwrap();
    let x = Tensor::new(vec![3e38f32, 3e37], &[2, 1]).unwrap();
    let p = m.predict(&x).unwrap();
    let p = p.host_slice().to_vec();
    assert!(p.iter().all(|v| v.is_finite()));
    assert!(p[0].abs() > 1e37, "前提: pred0 が十分大きい: {p:?}");
    let t0 = if p[0] > 0.0 { -3.3e38f32 } else { 3.3e38f32 };
    assert!(
        !(p[0] - t0).is_finite(),
        "前提: 行 0 の残差が overflow: {p:?}"
    );
    let y = Tensor::new(vec![t0, p[1]], &[2, 1]).unwrap();
    let h = m
        .fit_with_weights(
            &x,
            &y,
            FitConfig::new(1, 2),
            &FitWeights::new().sample_weight(&[1e-40, 1e30]),
            None,
            &mut [],
            &[],
        )
        .unwrap();
    assert!(h.loss[0].is_finite(), "loss = {}", h.loss[0]);
}
// ---------------------------------------------------------------------
// 5. 勾配累積で非有限化する重み付き勾配の拒否（レビュー指摘 #2823・P1）
// ---------------------------------------------------------------------

/// 各マイクロバッチの勾配は有限（bias ≈ ±1.5e38）だが f32 累積で ±inf になる最小構成。
/// 入力 0 → logits = bias（同一）→ `softmax − onehot` ≈ ±0.5、係数 3e38 で bias 勾配 ≈ ±1.5e38。
fn overflow_accumulation_model() -> Sequential {
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(
        Optimizer::Adam(fandhe_ai::optim::AdamConfig::default()),
        Loss::CrossEntropy,
    )
    .unwrap();
    m
}

fn run_overflow_accumulation(
    n: usize,
    accumulate_steps: u32,
) -> (
    Sequential,
    Result<fandhe_ai::compat::History, AutodiffError>,
) {
    let mut m = overflow_accumulation_model();
    let x = Tensor::new(vec![0.0f32; n], &[n, 1]).unwrap();
    let y = Tensor::new(vec![0i32; n], &[n]).unwrap();
    let sw = vec![3e38f32; n];
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 1).accumulate_steps(accumulate_steps),
        &FitWeights::new().sample_weight(&sw),
        None,
        &mut [],
        &[],
    );
    (m, r)
}

/// 拒否後に同一の良性 fit を実行し、新品モデルと bit 一致する（optimizer 状態を汚さない）。
fn assert_state_clean_after_reject(mut m: Sequential, what: &str) {
    let mut fresh = overflow_accumulation_model();
    assert!(m.is_compiled(), "{what}");
    let x = Tensor::new(vec![0.5f32, -0.5], &[2, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 1], &[2]).unwrap();
    for model in [&mut m, &mut fresh] {
        model
            .fit_with_weights(
                &x,
                &y,
                FitConfig::new(2, 1),
                &FitWeights::new().sample_weight(&[1.0, 2.0]),
                None,
                &mut [],
                &[],
            )
            .unwrap();
    }
    assert_eq!(bits(&m), bits(&fresh), "{what}: optimizer 状態が汚れた");
}

fn assert_rejected_ref(
    model: &Sequential,
    r: Result<fandhe_ai::compat::History, AutodiffError>,
    before: &[Vec<u32>],
    what: &str,
) {
    match r {
        Err(AutodiffError::InvalidArgument(_)) => {}
        other => panic!("{what}: InvalidArgument のはずが {other:?}"),
    }
    assert_eq!(bits(model), before, "{what}: パラメータが変化した");
}

/// 通常の累積境界（3 バッチ = accumulate_steps）で累積勾配が ±inf になる場合の拒否。
#[test]
fn accumulated_overflow_at_boundary_is_rejected_before_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let fresh = overflow_accumulation_model();
    let (before, mode) = (bits(&fresh), fresh.training());
    let (m, r) = run_overflow_accumulation(3, 3);
    assert_rejected_ref(&m, r, &before, "boundary overflow");
    assert_eq!(m.training(), mode);
    assert_state_clean_after_reject(m, "boundary overflow");
}

/// epoch 末の端数 flush（accumulate_steps=5 に満たない 3 バッチ）で ±inf になる場合の拒否。
#[test]
fn accumulated_overflow_at_epoch_end_flush_is_rejected_before_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let fresh = overflow_accumulation_model();
    let (before, mode) = (bits(&fresh), fresh.training());
    let (m, r) = run_overflow_accumulation(3, 5);
    assert_rejected_ref(&m, r, &before, "flush overflow");
    assert_eq!(m.training(), mode);
    assert_state_clean_after_reject(m, "flush overflow");
}
/// 係数（2.5e38）は f32 で表現可能でも、重み付き損失（係数 × -log p）が f32 で overflow して
/// inf になる場合は、`History` へ inf を記録せず更新前に拒否する（パラメータ不変）。
#[test]
fn weighted_loss_overflow_is_rejected_before_update() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut m = Sequential::new().add_linear(1, 2, 5).unwrap();
    m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let before = bits(&m);
    let x = Tensor::new(vec![1.0f32, 1.0, 1.0], &[3, 1]).unwrap();
    let y = Tensor::new(vec![0i32, 0, 0], &[3]).unwrap();
    let r = m.fit_with_weights(
        &x,
        &y,
        FitConfig::new(1, 2),
        &FitWeights::new()
            .sample_weight(&[1e19, 0.0, 0.0])
            .class_weight(HashMap::from([(0u32, 5e19f32)])),
        None,
        &mut [],
        &[],
    );
    assert_rejected_ref(&m, r, &before, "loss overflow");
}
