//! `Sgd` の `OptimizerStateDict`（イシュー #2367・親 #2131）の統合テスト。
//! momentum の velocity を保存・復元して bit 一致で再開できること、
//! `step_count` を持たないキー配置、fail-closed な拒否契約を固定する
//! （`nn::optim::state_dict` モジュール冒頭 doc「キー配置」節）。
//! 実機非依存のため `#[ignore]` は付けない。

use std::collections::{BTreeSet, HashMap};

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::optim::{
    Adagrad, AdagradConfig, OptimizerStateDict, RmsProp, RmsPropConfig,
};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn bits(x: &Tensor<f32>) -> (Vec<usize>, Vec<u32>) {
    let v = x
        .as_slice()
        .expect("test fixture: optimizer 出力は contiguous のはず")
        .iter()
        .map(|f| f.to_bits())
        .collect();
    (x.shape().to_vec(), v)
}

fn sd_bits(sd: &HashMap<String, Tensor<f32>>) -> Vec<(String, Vec<usize>, Vec<u32>)> {
    let mut v: Vec<_> = sd
        .iter()
        .map(|(k, x)| {
            let (s, b) = bits(x);
            (k.clone(), s, b)
        })
        .collect();
    v.sort();
    v
}

fn mom(momentum: f32) -> SgdConfig {
    SgdConfig::new(0.05).with_momentum(momentum)
}

fn params() -> Vec<Tensor<f32>> {
    vec![
        t(vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5], &[2, 3]),
        t(vec![0.1, -0.2, 0.3], &[3]),
    ]
}

fn grads(k: usize) -> Vec<Tensor<f32>> {
    let s = 1.0 + k as f32 * 0.37;
    vec![
        t(vec![0.3 * s, -0.7, 0.11 * s, 0.9, -0.2 * s, 0.4], &[2, 3]),
        t(vec![-0.5 * s, 0.6, 0.05], &[3]),
    ]
}

fn run(opt: &mut Sgd, p: &[Tensor<f32>], k: usize) -> Vec<Tensor<f32>> {
    let pr: Vec<&Tensor<f32>> = p.iter().collect();
    let g = grads(k);
    let gr: Vec<&Tensor<f32>> = g.iter().collect();
    opt.step(&pr, &gr).unwrap()
}

fn resume_is_bit_identical(cfg: SgdConfig) {
    let mut a = Sgd::new(cfg).unwrap();
    let mut p = params();
    for k in 0..3 {
        p = run(&mut a, &p, k);
    }
    let sd = a.state_dict().unwrap();
    let mut b = Sgd::new(cfg).unwrap();
    b.load_state_dict(sd.clone()).unwrap();
    assert_eq!(sd_bits(&sd), sd_bits(&b.state_dict().unwrap()));
    let pa = run(&mut a, &p, 3);
    let pb = run(&mut b, &p, 3);
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(bits(x), bits(y));
    }
    assert_eq!(
        sd_bits(&a.state_dict().unwrap()),
        sd_bits(&b.state_dict().unwrap())
    );
}

#[test]
fn resume_plain_momentum() {
    resume_is_bit_identical(mom(0.9));
}

#[test]
fn resume_dampening() {
    resume_is_bit_identical(SgdConfig {
        dampening: 0.3,
        ..mom(0.9)
    });
}

#[test]
fn resume_nesterov() {
    resume_is_bit_identical(SgdConfig {
        nesterov: true,
        ..mom(0.9)
    });
}

#[test]
fn resume_weight_decay() {
    resume_is_bit_identical(SgdConfig {
        weight_decay: 0.01,
        ..mom(0.8)
    });
}

#[test]
fn resume_with_non_contiguous_grad() {
    let cfg = mom(0.9);
    let mut a = Sgd::new(cfg).unwrap();
    let mut pa = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let g = t(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3])
        .transpose_2d()
        .unwrap();
    for _ in 0..2 {
        pa = a.step(&[&pa], &[&g]).unwrap().remove(0);
    }
    let mut b = Sgd::new(cfg).unwrap();
    b.load_state_dict(a.state_dict().unwrap()).unwrap();
    let ra = a.step(&[&pa], &[&g]).unwrap().remove(0);
    let rb = b.step(&[&pa], &[&g]).unwrap().remove(0);
    assert_eq!(bits(&ra), bits(&rb));
}

#[test]
fn key_set_has_no_step_count() {
    let mut a = Sgd::new(mom(0.9)).unwrap();
    let keys =
        |sd: &HashMap<String, Tensor<f32>>| -> BTreeSet<String> { sd.keys().cloned().collect() };
    let s0 = a.state_dict().unwrap();
    assert_eq!(
        keys(&s0),
        ["__optimizer__.sgd", "num_slots.u64_u16x4"]
            .map(String::from)
            .into_iter()
            .collect()
    );
    run(&mut a, &params(), 0);
    let s1 = a.state_dict().unwrap();
    assert_eq!(
        keys(&s1),
        [
            "__optimizer__.sgd",
            "num_slots.u64_u16x4",
            "state.0.momentum_buffer",
            "state.1.momentum_buffer"
        ]
        .map(String::from)
        .into_iter()
        .collect()
    );
}

#[test]
fn momentum_zero_state_is_empty_slots() {
    let mut a = Sgd::new(mom(0.0)).unwrap();
    run(&mut a, &params(), 0);
    assert_eq!(a.state_dict().unwrap().len(), 2);
}

fn valid_state() -> HashMap<String, Tensor<f32>> {
    let mut a = Sgd::new(mom(0.9)).unwrap();
    run(&mut a, &params(), 0);
    a.state_dict().unwrap()
}

/// 拒否されること、かつ失敗後の state_dict が失敗前と bit 一致すること。
fn assert_rejected(target: &mut Sgd, state: HashMap<String, Tensor<f32>>) {
    let before = sd_bits(&target.state_dict().unwrap());
    let r = target.load_state_dict(state);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");
    assert_eq!(before, sd_bits(&target.state_dict().unwrap()));
}

fn warmed() -> Sgd {
    let mut s = Sgd::new(mom(0.9)).unwrap();
    run(&mut s, &params(), 5);
    s
}

#[test]
fn rejects_momentum_state_into_momentum_zero() {
    assert_rejected(&mut Sgd::new(mom(0.0)).unwrap(), valid_state());
}

#[test]
fn rejects_step_count_key() {
    let mut s = valid_state();
    s.insert(
        "step_count.u64_u16x4".to_string(),
        t(vec![1.0, 0.0, 0.0, 0.0], &[4]),
    );
    assert_rejected(&mut warmed(), s);
}

#[test]
fn rejects_missing_num_slots() {
    let mut s = valid_state();
    s.remove("num_slots.u64_u16x4");
    assert_rejected(&mut warmed(), s);
}

#[test]
fn rejects_missing_slot_key() {
    let mut s = valid_state();
    s.remove("state.1.momentum_buffer");
    assert_rejected(&mut warmed(), s);
}

#[test]
fn rejects_non_canonical_index() {
    let mut s = valid_state();
    let v = s.remove("state.1.momentum_buffer").unwrap();
    s.insert("state.01.momentum_buffer".to_string(), v);
    assert_rejected(&mut warmed(), s);
}

#[test]
fn rejects_bad_marker_value() {
    let mut s = valid_state();
    s.insert("__optimizer__.sgd".to_string(), t(vec![2.0], &[1]));
    assert_rejected(&mut warmed(), s);
}

#[test]
fn rejects_rmsprop_and_adagrad_state() {
    let mut r = RmsProp::new(RmsPropConfig::default()).unwrap();
    let p = t(vec![1.0, 2.0], &[2]);
    let g = t(vec![0.1, 0.2], &[2]);
    r.step(&[(&p, &g)]).unwrap();
    assert_rejected(&mut warmed(), r.state_dict().unwrap());

    let mut ag = Adagrad::new(AdagradConfig::default()).unwrap();
    ag.step(&[(&p, &g)]).unwrap();
    assert_rejected(&mut warmed(), ag.state_dict().unwrap());

    // 逆方向: Sgd の state は RmsProp へ入らない。
    let mut r2 = RmsProp::new(RmsPropConfig::default()).unwrap();
    assert!(r2.load_state_dict(valid_state()).is_err());
}

#[test]
fn shape_mismatch_surfaces_at_next_step() {
    let mut b = Sgd::new(mom(0.9)).unwrap();
    b.load_state_dict(valid_state()).unwrap();
    let p = t(vec![1.0, 2.0], &[2]);
    let g = t(vec![0.1, 0.2], &[2]);
    let r = b.step(&[&p], &[&g]);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");
}

#[test]
fn zero_param_step_degenerates_to_uninitialized() {
    let cfg = mom(0.9);
    let mut a = Sgd::new(cfg).unwrap();
    a.step(&[], &[]).unwrap();
    let sd = a.state_dict().unwrap();
    assert_eq!(sd.len(), 2);
    let mut b = Sgd::new(cfg).unwrap();
    b.load_state_dict(sd.clone()).unwrap();
    assert_eq!(sd_bits(&sd), sd_bits(&b.state_dict().unwrap()));
    let p = t(vec![1.0], &[1]);
    let g = t(vec![0.5], &[1]);
    // 元は件数変化エラー、復元側は初回 step として成功する（doc の正規化）。
    assert!(a.step(&[&p], &[&g]).is_err());
    assert!(b.step(&[&p], &[&g]).is_ok());
}
