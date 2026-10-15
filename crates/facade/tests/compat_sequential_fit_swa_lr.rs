//! SWA の学習率アニールと併用拒否条件（イシュー #3059・親 #3057。決定記録
//! `docs/autodiff-swa-decision.md` §16.2・§16.4・§18）の統合テスト。`fandhe_ai` の公開パスのみを
//! import し、内部クレートは直接 import しない。
//!
//! 学習率の比較は `SwaLr`／`SequentialLr` の直接使用との bit 一致を基本とし、PyTorch `SWALR` の
//! fixture（`crates/autodiff/tests/fixtures/swa-pytorch-reference/swa_reference.json`）とは
//! スケジューラ値の検査として相対 1e-6 で突合する（#2176 の先例。REQ-2 の tolerance 定数は
//! 追加・変更しない）。実機非依存のホスト計算のみのため `#[ignore]` 分離は行わない。

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, EarlyStopping, EmaCallback, FitConfig, History, LambdaCallback,
    Loss, LrSchedule, ModelCheckpoint, Monitor, Optimizer, Sequential, SwaCallback, TrainStepFn,
    TrainStepOptimizer, TrainStepOutput,
};
use fandhe_ai::optim::{
    AdagradConfig, CosineAnnealingLr, LambConfig, LbfgsConfig, LrScheduler, ParamGroup,
    RmsPropConfig, SequentialLr, SgdConfig, SwaAnneal, SwaLr,
};
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 16;
const D_IN: usize = 4;
const D_OUT: usize = 2;

fn gen_data() -> (Tensor<f32>, Tensor<f32>) {
    let mut state = 0x3059_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..N * D_IN).map(|_| next()).collect();
    let y: Vec<f32> = (0..N * D_OUT).map(|_| next()).collect();
    (
        Tensor::new(x, &[N, D_IN]).expect("test fixture"),
        Tensor::new(y, &[N, D_OUT]).expect("test fixture"),
    )
}

fn sgd_model(lr: f32) -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, 8, 0x11)
        .expect("test fixture")
        .add_relu()
        .add_linear(8, D_OUT, 0x22)
        .expect("test fixture");
    m.compile(Optimizer::Sgd(SgdConfig::new(lr)), Loss::Mse)
        .expect("test fixture");
    m
}

fn linear_with(opt: Optimizer) -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, D_OUT, 1)
        .expect("fixture");
    m.compile(opt, Loss::Mse).expect("compile");
    m
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("test fixture: contiguous 化済み")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn dict_bits(d: &HashMap<String, Tensor<f32>>) -> BTreeMap<String, Vec<u32>> {
    d.iter().map(|(k, v)| (k.clone(), bits(v))).collect()
}

fn model_bits(m: &Sequential) -> BTreeMap<String, Vec<u32>> {
    dict_bits(&m.state_dict())
}

fn lr_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn swa_cb(start: usize, swa_lr: f32, anneal_epochs: usize, anneal: SwaAnneal) -> Callback {
    Callback::Swa(SwaCallback::new(start, swa_lr, anneal_epochs, anneal).expect("test fixture"))
}

fn swa_of(cb: &Callback) -> &SwaCallback {
    let Callback::Swa(s) = cb else {
        panic!("test fixture: Callback::Swa のはず");
    };
    s
}

fn cos_ls(base: f32, t_max: usize) -> Callback {
    Callback::LrSchedule(LrSchedule::per_epoch(
        CosineAnnealingLr::new(base, t_max, 0.0).expect("test fixture"),
    ))
}

fn is_invalid_arg<T: std::fmt::Debug>(r: Result<T, AutodiffError>) -> bool {
    matches!(r, Err(AutodiffError::InvalidArgument(_)))
}

fn fit(m: &mut Sequential, epochs: usize, cbs: &mut [Callback]) -> Result<History, AutodiffError> {
    let (x, y) = gen_data();
    m.fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, cbs)
}

fn load_fixture() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../autodiff/tests/fixtures/swa-pytorch-reference/swa_reference.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("test fixture: {} が読めない: {e}", path.display()));
    serde_json::from_str(&text).expect("test fixture: JSON")
}

fn anneal_of(name: &str) -> SwaAnneal {
    match name {
        "cos" => SwaAnneal::Cos,
        "linear" => SwaAnneal::Linear,
        other => panic!("test fixture: 未知の strategy {other}"),
    }
}

fn assert_close_rel(a: f32, b: f64, what: &str) {
    let (a, d) = (a as f64, (a as f64 - b).abs());
    assert!(d <= 1e-6 * b.abs() + 1e-9, "{what}: {a} vs {b}");
}

// 1. PyTorch SWALR fixture との突合と、SwaLr 直接使用との bit 一致（AC1）。
#[test]
fn history_lr_matches_swalr_and_pytorch_fixture() {
    let fx = load_fixture();
    let cases = fx["swalr_cases"].as_array().expect("swalr_cases");
    assert_eq!(cases.len(), 10);
    for c in cases {
        let name = c["name"].as_str().expect("name");
        let base = c["base_lr"].as_f64().expect("base_lr") as f32;
        let swa_lr = c["swa_lr"].as_f64().expect("swa_lr") as f32;
        let ae = c["anneal_epochs"].as_u64().expect("anneal_epochs") as usize;
        let anneal = anneal_of(c["strategy"].as_str().expect("strategy"));
        let lrs = c["lrs"].as_array().expect("lrs");
        let epochs = lrs.len().min(31);

        let mut m = sgd_model(base);
        let mut cbs = [swa_cb(0, swa_lr, ae, anneal)];
        let h = fit(&mut m, epochs, &mut cbs).expect("fit");
        assert_eq!(h.lr.len(), epochs, "{name}");

        let direct = SwaLr::new(base, swa_lr, ae, anneal).expect("SwaLr");
        for (k, lr) in h.lr.iter().enumerate() {
            assert_eq!(
                lr.to_bits(),
                direct.lr_at(k).to_bits(),
                "{name} k={k} 直接使用"
            );
            assert_close_rel(*lr, lrs[k].as_f64().expect("lr"), &format!("{name} k={k}"));
        }
    }
}

// 2. LrSchedule 併用と PyTorch 連結ケース（AC1・AC4）。callback の並び順に依らない。
#[test]
fn lr_schedule_chain_matches_sequential_lr_and_fixture_in_both_orders() {
    let fx = load_fixture();
    let c = &fx["swalr_chain_case"];
    let swa_lr = c["swa_lr"].as_f64().expect("swa_lr") as f32;
    let ae = c["anneal_epochs"].as_u64().expect("anneal_epochs") as usize;
    let lrs = c["lrs"].as_array().expect("lrs");
    // PyTorch の `epoch > swa_start`（9）は facade の `epochs_seen >= swa_start`（10）。
    let start = c["switch_step"].as_u64().expect("switch_step") as usize;
    assert_eq!(
        start,
        c["swa_start"].as_u64().expect("swa_start") as usize + 1
    );
    let epochs = lrs.len().min(31);

    let main = || CosineAnnealingLr::new(0.1, 20, 0.0).expect("main");
    let seq = SequentialLr::new(
        vec![
            Box::new(main()) as Box<dyn LrScheduler>,
            Box::new(SwaLr::new(main().lr_at(start), swa_lr, ae, SwaAnneal::Cos).expect("swa"))
                as Box<dyn LrScheduler>,
        ],
        vec![start],
    )
    .expect("SequentialLr");

    for swa_first in [false, true] {
        let mut m = sgd_model(0.1);
        let ls = cos_ls(0.1, 20);
        let sw = swa_cb(start, swa_lr, ae, SwaAnneal::Cos);
        let mut cbs = if swa_first { [sw, ls] } else { [ls, sw] };
        let h = fit(&mut m, epochs, &mut cbs).expect("fit");
        for (k, lr) in h.lr.iter().enumerate() {
            assert_eq!(
                lr.to_bits(),
                seq.lr_at(k).to_bits(),
                "swa_first={swa_first} k={k}"
            );
            assert_close_rel(*lr, lrs[k].as_f64().expect("lr"), &format!("k={k}"));
        }
    }
}

// 3. 区間前の不変（AC3）。lr・loss とも SWA なしと bit 一致する。
#[test]
fn before_swa_start_is_unchanged() {
    let start = 3usize;
    let mut a = sgd_model(0.05);
    let ha = fit(&mut a, 5, &mut [swa_cb(start, 0.01, 2, SwaAnneal::Linear)]).expect("fit");
    let mut b = sgd_model(0.05);
    let hb = fit(&mut b, 5, &mut []).expect("fit");
    assert_eq!(lr_bits(&ha.lr[..start]), lr_bits(&hb.lr[..start]));
    assert!(ha.lr[..start].iter().all(|v| *v == 0.05));
    assert_eq!(lr_bits(&ha.loss[..start]), lr_bits(&hb.loss[..start]));

    let mut a = sgd_model(0.1);
    let ha = fit(
        &mut a,
        5,
        &mut [cos_ls(0.1, 20), swa_cb(start, 0.01, 2, SwaAnneal::Cos)],
    )
    .expect("fit");
    let mut b = sgd_model(0.1);
    let hb = fit(&mut b, 5, &mut [cos_ls(0.1, 20)]).expect("fit");
    assert_eq!(lr_bits(&ha.lr[..start]), lr_bits(&hb.lr[..start]));
    assert_eq!(lr_bits(&ha.loss[..start]), lr_bits(&hb.loss[..start]));
    // 区間に入った epoch は SwaLr の値（基準 lr = 同期後の lr）。
    let base = hb.lr[start];
    let direct = SwaLr::new(base, 0.01, 2, SwaAnneal::Cos).expect("SwaLr");
    assert_eq!(ha.lr[start].to_bits(), direct.lr_at(0).to_bits());
    assert_eq!(ha.lr[start + 1].to_bits(), direct.lr_at(1).to_bits());
}

// 4. 分割 fit の継続。基準 lr が fit をまたいで保持される。
#[test]
fn split_fit_keeps_anneal_base_lr() {
    let mk = || swa_cb(1, 0.01, 3, SwaAnneal::Cos);
    let mut a = sgd_model(0.05);
    let mut cbs_a = [mk()];
    let h1 = fit(&mut a, 2, &mut cbs_a).expect("fit");
    let h2 = fit(&mut a, 2, &mut cbs_a).expect("fit");
    let mut b = sgd_model(0.05);
    let mut cbs_b = [mk()];
    let hb = fit(&mut b, 4, &mut cbs_b).expect("fit");
    let joined: Vec<f32> = h1.lr.iter().chain(&h2.lr).copied().collect();
    assert_eq!(lr_bits(&joined), lr_bits(&hb.lr));
    assert_eq!(
        dict_bits(&swa_of(&cbs_a[0]).averaged_state_dict().expect("avg")),
        dict_bits(&swa_of(&cbs_b[0]).averaged_state_dict().expect("avg"))
    );
    assert_eq!(
        dict_bits(&swa_of(&cbs_a[0]).raw_state_dict().expect("raw")),
        dict_bits(&swa_of(&cbs_b[0]).raw_state_dict().expect("raw"))
    );
}

// 5. 継続 fit が Err で終わっても、次の fit の学習率列は失敗しなかった場合と一致する。
#[test]
fn failed_continuation_does_not_disturb_anneal() {
    let direct = SwaLr::new(0.05, 0.01, 4, SwaAnneal::Linear).expect("SwaLr");
    let mut m = sgd_model(0.05);
    let mut cbs = vec![swa_cb(0, 0.01, 4, SwaAnneal::Linear)];
    fit(&mut m, 2, &mut cbs).expect("fit");
    cbs.push(Callback::Lambda(LambdaCallback::on_epoch_end(|e, _| {
        if e == 1 {
            Err(AutodiffError::InvalidArgument("test: 意図的な失敗".into()))
        } else {
            Ok(())
        }
    })));
    assert!(is_invalid_arg(fit(&mut m, 3, &mut cbs)));
    cbs.pop();
    let h = fit(&mut m, 2, &mut cbs).expect("fit");
    assert_eq!(h.lr[0].to_bits(), direct.lr_at(2).to_bits());
    assert_eq!(h.lr[1].to_bits(), direct.lr_at(3).to_bits());
}

/// step 2 以降で学習率 0 を返す主スケジューラ（基準 lr が 0 になる状況の再現用）。
struct ZeroFrom(usize);
impl LrScheduler for ZeroFrom {
    fn lr_at(&self, step: usize) -> f32 {
        if step >= self.0 { 0.0 } else { 0.1 }
    }
}

// 6. 基準 lr が非正のとき fail-closed（InvalidArgument）。平均化重みで上書きしない。
#[test]
fn non_positive_base_lr_fails_closed() {
    let mut m = sgd_model(0.1);
    let zero = Callback::LrSchedule(LrSchedule::per_epoch(ZeroFrom(2)));
    let mut cbs = [zero, swa_cb(2, 0.01, 2, SwaAnneal::Cos)];
    assert!(matches!(
        fit(&mut m, 4, &mut cbs),
        Err(AutodiffError::InvalidArgument(msg)) if msg.contains("base_lr")
    ));
    assert!(swa_of(&cbs[1]).raw_state_dict().is_none());
}

fn assert_rejected(mut m: Sequential, mut cbs: Vec<Callback>) {
    let (x, y) = gen_data();
    let before = model_bits(&m);
    let mode = m.training();
    assert!(is_invalid_arg(m.fit_with_callbacks(
        &x,
        &y,
        FitConfig::new(2, N),
        None,
        &mut cbs
    )));
    assert_eq!(model_bits(&m), before);
    assert_eq!(m.training(), mode);
    m.evaluate(&x, &y, N).expect("compile 状態が保持されている");
    for cb in &cbs {
        if let Callback::Swa(s) = cb {
            assert_eq!(s.n_averaged(), 0);
            assert!(s.averaged_state_dict().is_none());
        }
    }
}

// 7. 拒否条件（AC2）。
#[test]
fn rejected_combinations_leave_state_untouched() {
    let swa = || swa_cb(0, 0.01, 0, SwaAnneal::Cos);
    assert_rejected(sgd_model(0.05), vec![swa(), swa()]);
    assert_rejected(
        linear_with(Optimizer::Lbfgs(LbfgsConfig::default())),
        vec![swa()],
    );
    let mut amp = Sequential::new()
        .add_linear(D_IN, D_OUT, 1)
        .expect("fixture");
    amp.compile_with_amp(
        Optimizer::Sgd(SgdConfig::new(0.05)),
        Loss::Mse,
        AmpConfig::new(AmpDType::Bf16),
    )
    .expect("compile_with_amp");
    assert_rejected(amp, vec![swa()]);
    assert_rejected(
        sgd_model(0.05),
        vec![swa(), Callback::Ema(EmaCallback::new(0.9).expect("ema"))],
    );
    assert_rejected(
        sgd_model(0.05),
        vec![
            swa(),
            Callback::EarlyStopping(EarlyStopping::new(2).restore_best_weights(true)),
        ],
    );
    for opt in [
        Optimizer::RmsProp(RmsPropConfig::default()),
        Optimizer::Adagrad(AdagradConfig::default()),
        Optimizer::Lamb(LambConfig::default()),
    ] {
        assert_rejected(linear_with(opt), vec![swa()]);
    }
    let mut pg = Sequential::new()
        .add_linear(D_IN, D_OUT, 1)
        .expect("fixture");
    pg.compile_with_param_groups(
        Optimizer::Sgd(SgdConfig::new(0.05)),
        Loss::Mse,
        &[ParamGroup::new(vec![0, 1], 0.05, 0.0)],
    )
    .expect("compile_with_param_groups");
    assert_rejected(pg, vec![swa()]);
}

// 7b. カスタム train_step フックとの併用拒否。
#[test]
fn custom_train_step_is_rejected() {
    let (x, y) = gen_data();
    let mut m = sgd_model(0.05);
    let before = model_bits(&m);
    let mut cbs = [swa_cb(0, 0.01, 0, SwaAnneal::Cos)];
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
        &mut cbs,
        &[],
        hook
    )));
    assert_eq!(model_bits(&m), before);
    assert_eq!(swa_of(&cbs[0]).n_averaged(), 0);
    m.evaluate(&x, &y, N).expect("compile 状態が保持されている");
}

// 7c. 正のケース: ModelCheckpoint（ファイル保存なし）・restore なしの EarlyStopping は併用できる。
#[test]
fn checkpoint_and_plain_early_stopping_are_accepted() {
    let mut m = sgd_model(0.05);
    let mut cbs = [
        swa_cb(0, 0.01, 0, SwaAnneal::Cos),
        Callback::EarlyStopping(EarlyStopping::new(5).monitor(Monitor::Loss)),
        Callback::ModelCheckpoint(ModelCheckpoint::new().monitor(Monitor::Loss)),
    ];
    fit(&mut m, 2, &mut cbs).expect("併用は受理");
    assert_eq!(swa_of(&cbs[0]).n_averaged(), 2);
}
