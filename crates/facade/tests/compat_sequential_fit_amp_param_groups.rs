//! `compat::Sequential::compile_with_amp_and_param_groups`（イシュー #3832・設計 #3752。
//! 決定記録 `docs/autodiff-param-groups-decision.md` §14・§14.12）の統合テスト。
//!
//! `compat_sequential_fit_amp.rs` と同じく、手書きの参照ループ（低精度 forward →
//! `GradScaler::scale_loss`／`unscale` → `step_with_groups` → `update`）を組むため
//! `fandhe_ai_autodiff`・`fandhe_ai_tensor_core` を直接 import する。skip の fixture は
//! 同ファイルの `init_scale = 2^127`（F16）を再利用する。判定はすべて bit 完全一致で、
//! 許容誤差定数は追加・変更しない。実機（CUDA／Metal）非依存のため `#[ignore]` 分離はしない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, EmaCallback, FitConfig, Loss, LrSchedule, ModelIoError,
    Optimizer, Sequential, SwaCallback, TrainStepFn, TrainStepOptimizer, TrainStepOutput,
    save_model,
};
use fandhe_ai::optim::{
    AdamW, AdamWConfig, GradScaler, GradScalerConfig, LbfgsConfig, ParamGroup, ParamGroupStep as _,
    SgdConfig, StepLr, SwaAnneal,
};
use fandhe_ai::{AutodiffError, Tensor};
use fandhe_ai_autodiff::nn::linear_forward_low_precision;
use fandhe_ai_tensor_core::{Activation, ScalarDType};

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;
const SEED_DATA: u64 = 0xC0FFEE;

fn data(y_scale: f32) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(SEED_DATA);
    let x = rng.fill_vec(N * D_IN);
    let y: Vec<f32> = rng
        .fill_vec(N * D_OUT)
        .iter()
        .map(|v| v * y_scale)
        .collect();
    (
        Tensor::new(x, &[N, D_IN]).expect("test fixture"),
        Tensor::new(y, &[N, D_OUT]).expect("test fixture"),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .expect("test fixture")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .expect("test fixture")
}

/// スロット 0・1 = 1 層目（"0."）、2・3 = 2 層目（"2."）。
fn layer_slots(model: &Sequential, prefix: &str) -> Vec<usize> {
    model
        .named_parameters()
        .iter()
        .enumerate()
        .filter_map(|(i, (name, _))| name.starts_with(prefix).then_some(i))
        .collect()
}

fn snapshot(model: &Sequential) -> Vec<Vec<u32>> {
    model
        .trainable_parameters()
        .iter()
        .map(|t| {
            t.contiguous()
                .as_slice()
                .expect("test fixture: contiguous 化済み")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー")
}

fn is_invalid_arg<T: std::fmt::Debug>(r: Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

fn normal_scaler() -> GradScalerConfig {
    GradScalerConfig {
        init_scale: 256.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1000,
    }
}

/// 最初の step が必ず skip になる fixture（`compat_sequential_fit_amp.rs` と同じ値）。
fn skip_first_scaler() -> GradScalerConfig {
    GradScalerConfig {
        init_scale: 2.0f32.powi(127),
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 2,
    }
}

fn scalar_dtype(d: AmpDType) -> ScalarDType {
    match d {
        AmpDType::F16 => ScalarDType::F16,
        _ => ScalarDType::Bf16,
    }
}

fn adamw_cfg() -> AdamWConfig {
    AdamWConfig::default()
}

/// 手書き参照ループの結果（step ごと）。
struct Manual {
    losses: Vec<f32>,
    scales: Vec<f32>,
    skips: Vec<bool>,
    snapshots: Vec<Vec<Vec<u32>>>,
}

/// `compile_with_amp_and_param_groups` + `fit` が内部で行う演算列の手書き版
/// （AdamW・`step_with_groups`）。1 step = 1 epoch（バッチ = 全件）。
fn manual_loop(
    dtype: AmpDType,
    scaler_config: GradScalerConfig,
    groups: &[ParamGroup],
    steps: usize,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
) -> Manual {
    let sd = scalar_dtype(dtype);
    let mut model = build_model();
    let mut opt = AdamW::new(adamw_cfg()).expect("test fixture");
    let mut scaler = GradScaler::new(scaler_config).expect("test fixture");
    let mut out = Manual {
        losses: vec![],
        scales: vec![],
        skips: vec![],
        snapshots: vec![],
    };
    for _ in 0..steps {
        let outcome = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let xv = tape.var(x);
            let yv = tape.var_no_grad(y);
            let l = bound.linears();
            let h = linear_forward_low_precision(&l[0], &xv, Activation::Relu, sd).expect("fwd1");
            let pred = linear_forward_low_precision(&l[1], &h, Activation::None, sd).expect("fwd2");
            let loss = pred.mse_loss(&yv).expect("mse");
            out.losses.push(scalar(&loss.to_tensor()));
            let scaled = scaler.scale_loss(&loss).expect("scale_loss");
            let grads = tape.backward(&scaled).expect("backward");
            let grad_refs = bound.trainable_grads(&grads).expect("grads");
            let un = scaler.unscale(&grad_refs).expect("unscale");
            if un.should_skip_step() {
                None
            } else {
                let refs: Vec<&Tensor<f32>> = un.grads.iter().collect();
                let params = model.trainable_parameters();
                Some(opt.step_with_groups(&params, &refs, groups).expect("step"))
            }
        };
        let skipped = outcome.is_none();
        if let Some(updated) = outcome {
            model.apply_parameters(updated).expect("apply");
        }
        scaler.update(skipped).expect("update");
        out.scales.push(scaler.scale());
        out.skips.push(skipped);
        out.snapshots.push(snapshot(&model));
    }
    out
}

fn compile_groups(
    model: &mut Sequential,
    dtype: AmpDType,
    scaler: GradScalerConfig,
    groups: &[ParamGroup],
) {
    model
        .compile_with_amp_and_param_groups(
            Optimizer::AdamW(adamw_cfg()),
            Loss::Mse,
            AmpConfig::new(dtype).grad_scaler(scaler),
            groups,
        )
        .expect("test fixture: compile");
}

fn two_groups(model: &Sequential) -> Vec<ParamGroup> {
    // 1 層目 = lr 0.01、2 層目の weight のみ lr 0.002・wd 0.1。bias（スロット 3）はグループ外。
    vec![
        ParamGroup::new(layer_slots(model, "0."), 0.01, 0.0),
        ParamGroup::new(vec![layer_slots(model, "2.")[0]], 0.002, 0.1),
    ]
}

// 1. 空 groups は compile_with_amp と bit 一致する。
fn empty_groups_matches_amp(dtype: AmpDType) {
    let (x, y) = data(1.0);
    let mut a = build_model();
    a.compile_with_amp(
        Optimizer::AdamW(adamw_cfg()),
        Loss::Mse,
        AmpConfig::new(dtype).grad_scaler(normal_scaler()),
    )
    .expect("compile_with_amp");
    let ha = a.fit(&x, &y, FitConfig::new(5, N)).expect("fit");
    let mut b = build_model();
    compile_groups(&mut b, dtype, normal_scaler(), &[]);
    let hb = b.fit(&x, &y, FitConfig::new(5, N)).expect("fit");
    assert_eq!(snapshot(&a), snapshot(&b));
    assert_eq!(a.amp_loss_scale(), b.amp_loss_scale());
    let bits = |h: &[f32]| h.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&ha.loss), bits(&hb.loss));
}

#[test]
fn empty_groups_bit_match_compile_with_amp_f16() {
    empty_groups_matches_amp(AmpDType::F16);
}

#[test]
fn empty_groups_bit_match_compile_with_amp_bf16() {
    empty_groups_matches_amp(AmpDType::Bf16);
}

// 2. config と同じ hparams の全スロット 1 グループは compile_with_amp と bit 一致する。
#[test]
fn all_default_group_bit_matches_compile_with_amp() {
    let (x, y) = data(1.0);
    let mut a = build_model();
    a.compile_with_amp(
        Optimizer::AdamW(adamw_cfg()),
        Loss::Mse,
        AmpConfig::new(AmpDType::F16).grad_scaler(normal_scaler()),
    )
    .expect("compile_with_amp");
    a.fit(&x, &y, FitConfig::new(5, N)).expect("fit");
    let mut b = build_model();
    let n = b.trainable_parameters().len();
    let cfg = adamw_cfg();
    let all = [ParamGroup::new((0..n).collect(), cfg.lr, cfg.weight_decay)];
    compile_groups(&mut b, AmpDType::F16, normal_scaler(), &all);
    b.fit(&x, &y, FitConfig::new(5, N)).expect("fit");
    assert_eq!(snapshot(&a), snapshot(&b));
}

// 3. 空でない groups は手書きループ（step_with_groups）と bit 一致する。
fn layer_groups_match_manual(dtype: AmpDType) {
    let (x, y) = data(1.0);
    let steps = 5;
    let mut m = build_model();
    let groups = two_groups(&m);
    compile_groups(&mut m, dtype, normal_scaler(), &groups);
    let manual = manual_loop(dtype, normal_scaler(), &groups, steps, &x, &y);
    assert!(manual.skips.iter().all(|s| !s));
    let h = m.fit(&x, &y, FitConfig::new(steps, N)).expect("fit");
    assert_eq!(snapshot(&m), *manual.snapshots.last().expect("steps > 0"));
    assert_eq!(m.amp_loss_scale(), manual.scales.last().copied());
    let a: Vec<u32> = h.loss.iter().map(|v| v.to_bits()).collect();
    let b: Vec<u32> = manual.losses.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a, b);
}

#[test]
fn layer_groups_bit_match_manual_gradscaler_step_with_groups_loop_f16() {
    layer_groups_match_manual(AmpDType::F16);
}

#[test]
fn layer_groups_bit_match_manual_gradscaler_step_with_groups_loop_bf16() {
    layer_groups_match_manual(AmpDType::Bf16);
}

// 4. skip は全 group 一括で、optimizer の step_count も進まない。
#[test]
fn skip_step_skips_all_groups_and_keeps_step_count() {
    let (x, y) = data(1.0e3);
    let steps = 8;
    let mut m = build_model();
    let groups = two_groups(&m);
    compile_groups(&mut m, AmpDType::F16, skip_first_scaler(), &groups);
    let manual = manual_loop(AmpDType::F16, skip_first_scaler(), &groups, steps, &x, &y);
    assert!(manual.skips[0], "最初の step は skip のはずの構成");
    assert!(manual.skips.iter().any(|s| !s), "非 skip step が必要");
    let initial = snapshot(&build_model());
    for i in 0..steps {
        m.fit(&x, &y, FitConfig::new(1, N)).expect("fit(1)");
        assert_eq!(m.amp_loss_scale(), Some(manual.scales[i]), "scale step={i}");
        // AdamW の bias 補正は step_count に依存するため、skip で step_count が進めば
        // 以降の step が手書きループ（skip で進めない）と bit 一致しなくなる。
        assert_eq!(snapshot(&m), manual.snapshots[i], "params step={i}");
        if manual.skips[..=i].iter().all(|s| *s) {
            // 先頭からの連続 skip の間は全スロット（グループ内・グループ外）が初期値のまま。
            assert_eq!(snapshot(&m), initial, "skip 中に更新された step={i}");
        }
    }
}

// 5. lr = 0・wd = 0 の group は skip の有無にかかわらず bit 不変。
fn frozen_stays(y_scale: f32, scaler: GradScalerConfig) {
    let (x, y) = data(y_scale);
    let mut m = build_model();
    let first = layer_slots(&m, "0.");
    let initial = snapshot(&m);
    compile_groups(
        &mut m,
        AmpDType::F16,
        scaler,
        &[ParamGroup::new(first.clone(), 0.0, 0.0)],
    );
    m.fit(&x, &y, FitConfig::new(8, N)).expect("fit");
    let after = snapshot(&m);
    for &i in &first {
        assert_eq!(after[i], initial[i], "凍結スロット {i} が変化した");
    }
    let second = layer_slots(&m, "2.");
    assert!(second.iter().any(|&i| after[i] != initial[i]));
}

#[test]
fn frozen_group_stays_bit_unchanged_under_amp() {
    frozen_stays(1.0, normal_scaler());
    frozen_stays(1.0e3, skip_first_scaler());
}

// 6. 不正な hparams は最初の非 skip step で InvalidArgument になる（承認された論点 3 の現行挙動）。
#[test]
fn invalid_group_hparams_rejected_on_first_non_skip_step() {
    let (x, y) = data(1.0e3);
    let skips = manual_loop(AmpDType::F16, skip_first_scaler(), &[], 8, &x, &y).skips;
    let k = skips.iter().take_while(|s| **s).count();
    assert!(k >= 1 && k < skips.len());
    let mut m = build_model();
    let before = snapshot(&m);
    compile_groups(
        &mut m,
        AmpDType::F16,
        skip_first_scaler(),
        &[ParamGroup::new(vec![0], -0.1, 0.0)],
    );
    assert!(is_invalid_arg(m.fit(&x, &y, FitConfig::new(8, N))));
    assert_eq!(snapshot(&m), before, "拒否時にパラメータが変化した");
    // skip した step の scaler.update(true) は残る（k 回の半減）。
    assert_eq!(m.amp_loss_scale(), Some(2.0f32.powi(127 - k as i32)));
}

// 7. 範囲外スロットは fit の入口で、状態を変える前に拒否される。
#[test]
fn out_of_range_slot_rejected_at_fit_entry() {
    let (x, y) = data(1.0);
    let mut m = build_model();
    let before = snapshot(&m);
    compile_groups(
        &mut m,
        AmpDType::F16,
        normal_scaler(),
        &[ParamGroup::new(vec![99], 0.1, 0.0)],
    );
    assert!(is_invalid_arg(m.fit(&x, &y, FitConfig::new(2, N))));
    assert_eq!(snapshot(&m), before);
    assert_eq!(m.amp_loss_scale(), Some(256.0));
}

// 8. compile の拒否は直前の compiled 状態を残す。
#[test]
fn compile_rejections_keep_previous_compiled() {
    let mut m = build_model();
    compile_groups(&mut m, AmpDType::F16, normal_scaler(), &[]);
    assert_eq!(m.amp_loss_scale(), Some(256.0));
    for groups in [vec![], vec![ParamGroup::new(vec![0], 0.1, 0.0)]] {
        assert!(is_invalid_arg(m.compile_with_amp_and_param_groups(
            Optimizer::Lbfgs(LbfgsConfig::default()),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
            &groups,
        )));
        assert_eq!(m.amp_loss_scale(), Some(256.0));
    }
    let bad = GradScalerConfig {
        init_scale: 0.0,
        ..GradScalerConfig::default()
    };
    assert!(is_invalid_arg(m.compile_with_amp_and_param_groups(
        Optimizer::Sgd(SgdConfig::new(0.1)),
        Loss::Mse,
        AmpConfig::new(AmpDType::F16).grad_scaler(bad),
        &[],
    )));
    assert_eq!(m.amp_loss_scale(), Some(256.0));
    assert!(m.is_compiled());
}

// 9. 従来どおり拒否される組み合わせはモデルと compiled 状態を変えない。
//    `LrSchedule`・`save_model` は groups 非空の拒否（#3834／#3835 で変わりうる）を現行契約として固定する。
#[test]
fn fit_rejections_leave_model_untouched() {
    let (x, y) = data(1.0);
    let mut m = build_model();
    let groups = [ParamGroup::new(vec![0], 0.01, 0.0)];
    compile_groups(&mut m, AmpDType::F16, normal_scaler(), &groups);
    let before = snapshot(&m);

    assert!(is_invalid_arg(m.fit(
        &x,
        &y,
        FitConfig::new(2, 4).accumulate_steps(2)
    )));

    let mut step = |_: &Sequential,
                    _: &Tensor<f32>,
                    _: &Tensor<f32>,
                    _: &mut TrainStepOptimizer<'_>|
     -> Result<TrainStepOutput, AutodiffError> { Ok(TrainStepOutput::new(0.0)) };
    let hook: &mut TrainStepFn<'_, f32> = &mut step;
    assert!(is_invalid_arg(m.fit_with_train_step(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut [],
        &[],
        hook
    )));

    let mut swa = [Callback::Swa(
        SwaCallback::new(0, 0.01, 0, SwaAnneal::Cos).expect("test fixture"),
    )];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut swa
    )));

    let mut sched = [Callback::LrSchedule(LrSchedule::per_epoch(
        StepLr::new(0.1, 1, 0.5).expect("test fixture"),
    ))];
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(1, N),
        None,
        &mut sched
    )));

    let dir = std::env::temp_dir().join(format!("fandhe_ai_3832_save_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let err = save_model(&m, &dir).unwrap_err();
    assert!(
        matches!(
            err,
            ModelIoError::Autodiff(AutodiffError::InvalidArgument(_))
        ),
        "{err:?}"
    );
    assert!(!dir.exists());

    assert_eq!(snapshot(&m), before);
    assert_eq!(m.amp_loss_scale(), Some(256.0));
    assert!(m.is_compiled());
    m.fit(&x, &y, FitConfig::new(1, N))
        .expect("拒否後も compile 状態が残る");
}

// 10. EMA × AMP × groups の併用は受理され、skip した step は EMA を更新しない。
#[test]
fn ema_amp_groups_combination_is_accepted() {
    let (x, y) = data(1.0e3);
    let epochs = 8;
    let skips = manual_loop(AmpDType::F16, skip_first_scaler(), &[], epochs, &x, &y).skips;
    let non_skip = skips.iter().filter(|s| !**s).count() as u64;
    assert!(non_skip > 0 && non_skip < epochs as u64);

    let mk = || {
        let mut m = build_model();
        let first = layer_slots(&m, "0.");
        compile_groups(
            &mut m,
            AmpDType::F16,
            skip_first_scaler(),
            &[ParamGroup::new(first, 0.0, 0.0)],
        );
        m
    };
    let mut with_ema = mk();
    let mut cbs = [Callback::Ema(EmaCallback::new(0.5).expect("test fixture"))];
    with_ema
        .fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, &mut cbs)
        .expect("EMA × AMP × groups は受理される");
    let Callback::Ema(e) = &cbs[0] else {
        panic!("test fixture: Callback::Ema のはず");
    };
    assert_eq!(e.num_updates(), non_skip);

    // EMA は生の学習軌道を変えない（shadow の値そのものは丸めがあるため主張しない）。
    let mut plain = mk();
    plain.fit(&x, &y, FitConfig::new(epochs, N)).expect("fit");
    assert_eq!(snapshot(&with_ema), snapshot(&plain));
}
