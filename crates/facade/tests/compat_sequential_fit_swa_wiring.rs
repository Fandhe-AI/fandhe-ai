//! SWA の `fit_with_callbacks` 結線の追加統合テスト（イシュー #3060・親 #3057。決定記録
//! `docs/autodiff-swa-decision.md` §16〜§19）。`fandhe_ai` の公開パスのみを import し、
//! 内部クレートは直接 import しない。
//!
//! #3058（`compat_sequential_fit_swa.rs`）は平均化のみ（`anneal_epochs = 0`）、#3059
//! （`compat_sequential_fit_swa_lr.rs`）は `history.lr` と `n_averaged` のみを検査していた。
//! 本ファイルはその隙間を埋める: 学習率アニールと平均化を**同時に**動かした場合の重みを
//! 手動ループ（`LrSchedule` で同じ lr 列を与える）と突合し、途中終了（`EarlyStopping`）と
//! `ModelCheckpoint` の併用（§16.2・§16.3）も固定する。
//!
//! 判定は bit 完全一致のみで、許容誤差定数は追加・変更しない。実機（CUDA／Metal）非依存の
//! ホスト計算のみ（ファイル I/O なし）のため `#[ignore]` 分離は行わない。

use std::collections::{BTreeMap, HashMap};

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    Callback, EarlyStopping, FitConfig, History, Loss, LrSchedule, ModelCheckpoint, Monitor,
    Optimizer, Sequential, SwaCallback,
};
use fandhe_ai::optim::{AveragedModel, LrScheduler, SgdConfig, SwaAnneal, SwaLr};

const N: usize = 16;
const D_IN: usize = 4;
const D_OUT: usize = 2;
const BASE_LR: f32 = 0.05;
const SWA_LR: f32 = 0.01;
const ANNEAL_EPOCHS: usize = 2;

fn gen_data() -> (Tensor<f32>, Tensor<f32>) {
    // 決定的な疑似データ（シード固定の線形合同法）。
    let mut state = 0x3060_u64;
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

fn build_model() -> Sequential {
    let mut m = Sequential::new()
        .add_linear(D_IN, 8, 0x11)
        .expect("test fixture")
        .add_relu()
        .add_linear(8, D_OUT, 0x22)
        .expect("test fixture");
    m.compile(Optimizer::Sgd(SgdConfig::new(BASE_LR)), Loss::Mse)
        .expect("test fixture");
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

fn f32_bits(v: &[f32]) -> Vec<u32> {
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

/// 参照側の lr 列。epoch < start は基準 lr、以降は `SwaLr` の閉形式（SWA 側の結線と同じ式）。
struct RefLr {
    start: usize,
    anneal: SwaAnneal,
}

impl LrScheduler for RefLr {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.start {
            BASE_LR
        } else {
            SwaLr::new(BASE_LR, SWA_LR, ANNEAL_EPOCHS, self.anneal)
                .expect("test fixture")
                .lr_at(step - self.start)
        }
    }
}

fn fit_n(m: &mut Sequential, epochs: usize, cbs: &mut [Callback]) -> History {
    let (x, y) = gen_data();
    m.fit_with_callbacks(&x, &y, FitConfig::new(epochs, N), None, cbs)
        .expect("test fixture: fit")
}

struct Reference {
    model: Sequential,
    lr: Vec<f32>,
    loss: Vec<f32>,
    avg: AveragedModel,
}

/// 手動ループ: `LrSchedule`（fit をまたいで epoch カウンタを維持）で lr を与え、1 epoch ずつ
/// fit する。epoch >= start のあとで `AveragedModel::update_named` を呼ぶ。
fn reference_run(epochs: usize, start: usize, anneal: SwaAnneal) -> Reference {
    let mut model = build_model();
    let mut avg = AveragedModel::from_named(model.named_parameters()).expect("test fixture");
    let mut cbs = [Callback::LrSchedule(LrSchedule::per_epoch(RefLr {
        start,
        anneal,
    }))];
    let (mut lr, mut loss) = (Vec::new(), Vec::new());
    for e in 0..epochs {
        let h = fit_n(&mut model, 1, &mut cbs);
        lr.extend(h.lr);
        loss.extend(h.loss);
        if e >= start {
            avg.update_named(model.named_parameters())
                .expect("test fixture");
        }
    }
    Reference {
        model,
        lr,
        loss,
        avg,
    }
}

// T1. アニールと平均化を同時に動かした fit が手動ループと bit 一致する。
#[test]
fn fit_swa_with_anneal_matches_manual_loop_bit_exact() {
    let (epochs, start) = (6usize, 2usize);
    for anneal in [SwaAnneal::Cos, SwaAnneal::Linear] {
        let r = reference_run(epochs, start, anneal);
        // lr 列が全て同値ではないこと: アニールが実際に働いている。
        assert!(r.lr.windows(2).any(|w| w[0].to_bits() != w[1].to_bits()));

        let mut model = build_model();
        let mut cbs = [swa_cb(start, SWA_LR, ANNEAL_EPOCHS, anneal)];
        let h = fit_n(&mut model, epochs, &mut cbs);

        assert_eq!(f32_bits(&h.lr), f32_bits(&r.lr), "{anneal:?}: history.lr");
        assert_eq!(f32_bits(&h.loss), f32_bits(&r.loss), "{anneal:?}: loss");
        let s = swa_of(&cbs[0]);
        assert_eq!(s.n_averaged(), r.avg.n_averaged());
        let averaged = s.averaged_state_dict().expect("test fixture");
        assert_eq!(
            dict_bits(&averaged),
            dict_bits(&r.avg.averaged_state_dict()),
            "{anneal:?}: averaged"
        );
        assert_eq!(model_bits(&model), dict_bits(&averaged));
        let raw = s.raw_state_dict().expect("test fixture");
        assert_eq!(dict_bits(&raw), model_bits(&r.model), "{anneal:?}: raw");
        model.load_state_dict(raw).expect("test fixture");
        assert_eq!(model_bits(&model), model_bits(&r.model));
    }
}

// T2. アニール込みの分割 fit（3+3）が一括 fit（6）と bit 一致する。
#[test]
fn fit_swa_with_anneal_split_fit_matches_single_fit() {
    let (start, anneal) = (2usize, SwaAnneal::Cos);

    let mut whole = build_model();
    let mut cbs_w = [swa_cb(start, SWA_LR, ANNEAL_EPOCHS, anneal)];
    let hw = fit_n(&mut whole, 6, &mut cbs_w);

    let mut split = build_model();
    let mut cbs_s = [swa_cb(start, SWA_LR, ANNEAL_EPOCHS, anneal)];
    let h1 = fit_n(&mut split, 3, &mut cbs_s);
    let h2 = fit_n(&mut split, 3, &mut cbs_s);

    let lr_split: Vec<f32> = h1.lr.iter().chain(&h2.lr).copied().collect();
    assert_eq!(f32_bits(&lr_split), f32_bits(&hw.lr));
    // 2 回目の開始時に生の重みへ書き戻されていれば、続きの loss が一括 fit と一致する。
    let loss_split: Vec<f32> = h1.loss.iter().chain(&h2.loss).copied().collect();
    assert_eq!(f32_bits(&loss_split), f32_bits(&hw.loss));
    let (sw, ss) = (swa_of(&cbs_w[0]), swa_of(&cbs_s[0]));
    assert_eq!(sw.n_averaged(), ss.n_averaged());
    assert_eq!(
        dict_bits(&sw.averaged_state_dict().expect("test fixture")),
        dict_bits(&ss.averaged_state_dict().expect("test fixture"))
    );
    assert_eq!(
        dict_bits(&sw.raw_state_dict().expect("test fixture")),
        dict_bits(&ss.raw_state_dict().expect("test fixture"))
    );
    assert_eq!(model_bits(&whole), model_bits(&split));
}

// T3. EarlyStopping による途中終了でも平均化重みが適用される（§16.3）。
#[test]
fn early_stopping_termination_still_applies_averaged_weights() {
    let epochs = 6usize;
    let mut model = build_model();
    let mut cbs = [
        Callback::EarlyStopping(
            EarlyStopping::new(1)
                .monitor(Monitor::Loss)
                .min_delta(1.0e6)
                .expect("test fixture"),
        ),
        swa_cb(0, SWA_LR, ANNEAL_EPOCHS, SwaAnneal::Linear),
    ];
    let h = fit_n(&mut model, epochs, &mut cbs);

    let Callback::EarlyStopping(es) = &cbs[0] else {
        panic!("test fixture");
    };
    assert!(es.stopped_epoch().is_some(), "打ち切られていること");
    assert!(h.loss.len() < epochs);
    let s = swa_of(&cbs[1]);
    assert_eq!(s.n_averaged(), h.loss.len() as u64);
    let averaged = s.averaged_state_dict().expect("test fixture");
    assert_eq!(model_bits(&model), dict_bits(&averaged));
}

// T4. SWA 区間内でも ModelCheckpoint は生の重みを保存する（§16.2）。
// 生の軌跡を揃えるため swa_lr = 基準 lr・anneal_epochs = 0（lr は変化しない）にして、
// SWA なしの checkpoint と bit 一致することを検査する。
#[test]
fn model_checkpoint_inside_swa_window_keeps_raw_weights() {
    let epochs = 5usize;
    let mut plain = build_model();
    let mut cbs_p = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::Loss),
    )];
    fit_n(&mut plain, epochs, &mut cbs_p);

    let mut model = build_model();
    let mut cbs = [
        Callback::ModelCheckpoint(ModelCheckpoint::new().monitor(Monitor::Loss)),
        swa_cb(0, BASE_LR, 0, SwaAnneal::Cos),
    ];
    fit_n(&mut model, epochs, &mut cbs);

    let (Callback::ModelCheckpoint(mc), Callback::ModelCheckpoint(mc_p)) = (&cbs[0], &cbs_p[0])
    else {
        panic!("test fixture");
    };
    let best = mc.best_state_dict().expect("best あり");
    let best_p = mc_p.best_state_dict().expect("best あり");
    assert_eq!(dict_bits(best), dict_bits(best_p));
    let averaged = swa_of(&cbs[1]).averaged_state_dict().expect("test fixture");
    assert_ne!(dict_bits(best), dict_bits(&averaged));
}
