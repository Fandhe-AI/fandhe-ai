//! `fandhe_ai::optim::OptimizerStateDict`（イシュー #2556・親 #2555）の facade 経由の
//! 単体テスト。
//!
//! 内部クレート `fandhe_ai_autodiff` は import せず、`fandhe_ai` だけで次を固定する
//! （公開形・承認の根拠は `docs/autodiff-optimizer-state-dict-decision.md` §9・§11。
//! 実装本体の網羅検証は `crates/autodiff/tests/nn_optim_state_dict.rs`・
//! `optim_sgd_state_dict.rs`。本ファイルは「facade から同じ契約に到達できる」ことの確認）。
//!
//! 1. 10 型（`AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・`Adadelta`・`Adamax`・
//!    `NAdam`・`RAdam`・`Sgd`）で、数 step 後の `state_dict()` を新規インスタンスへ
//!    `load_state_dict()` すると、続きの `step()` が中断なしの場合と bit 一致する。
//! 2. `Sgd` は momentum あり・なしの両方（なしはスロットキーを持たない）。
//! 3. `load_state_dict` が失敗したとき状態は変わらない（キー欠落・余剰キー・shape 不一致）。
//! 4. 別種の optimizer の状態（`Adam` → `AdamW`）は種別マーカーで拒否される。
//! 5. safetensors のバイト列を経由しても復元できる。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。許容誤差は使わず bit 一致で比較する。

use std::collections::HashMap;

use fandhe_ai::Tensor;
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes,
};
use fandhe_ai::optim::{
    Adadelta, AdadeltaConfig, Adagrad, AdagradConfig, Adam, AdamConfig, AdamW, AdamWConfig, Adamax,
    AdamaxConfig, Lamb, LambConfig, NAdam, NAdamConfig, OptimizerStateDict, RAdam, RAdamConfig,
    RmsProp, RmsPropConfig, Sgd, SgdConfig,
};

type State = HashMap<String, Tensor<f32>>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn initial_params() -> Vec<Tensor<f32>> {
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

fn bits(x: &Tensor<f32>) -> (Vec<usize>, Vec<u32>) {
    let v = x
        .as_slice()
        .expect("test fixture: optimizer 出力は contiguous のはず")
        .iter()
        .map(|f| f.to_bits())
        .collect();
    (x.shape().to_vec(), v)
}

fn state_bits(sd: &State) -> Vec<(String, Vec<usize>, Vec<u32>)> {
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

/// optimizer の種別ごとの差（`step` のシグネチャ）を吸収するドライバ。
trait Drive: OptimizerStateDict {
    fn drive(&mut self, params: &[Tensor<f32>], grads: &[Tensor<f32>]) -> Vec<Tensor<f32>>;
}

macro_rules! drive_pairs {
    ($($ty:ty),* $(,)?) => {$(
        impl Drive for $ty {
            fn drive(&mut self, params: &[Tensor<f32>], grads: &[Tensor<f32>]) -> Vec<Tensor<f32>> {
                let pg: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    params.iter().zip(grads.iter()).collect();
                self.step(&pg).expect("step")
            }
        }
    )*};
}
drive_pairs!(
    AdamW, Adam, RmsProp, Adagrad, Lamb, Adadelta, Adamax, NAdam, RAdam
);

impl Drive for Sgd {
    fn drive(&mut self, params: &[Tensor<f32>], grads: &[Tensor<f32>]) -> Vec<Tensor<f32>> {
        let p: Vec<&Tensor<f32>> = params.iter().collect();
        let g: Vec<&Tensor<f32>> = grads.iter().collect();
        self.step(&p, &g).expect("step")
    }
}

/// `warm` 回 step した状態を新規インスタンスへ移し、続く 3 step が bit 一致することを確認する。
/// `via_bytes` が真なら safetensors のバイト列を経由する。
fn assert_resume_is_bit_exact<O: Drive>(make: impl Fn() -> O, warm: usize, via_bytes: bool) {
    let mut original = make();
    let mut params = initial_params();
    for k in 0..warm {
        params = original.drive(&params, &grads(k));
    }

    let mut sd = original.state_dict().expect("state_dict");
    if via_bytes {
        let bytes = save_safetensors_f32_to_bytes(&sd, None).expect("save");
        sd = load_safetensors_f32_from_bytes(&bytes).expect("load bytes");
    }
    let mut resumed = make();
    resumed.load_state_dict(sd).expect("load_state_dict");
    assert_eq!(
        state_bits(&original.state_dict().expect("state_dict")),
        state_bits(&resumed.state_dict().expect("state_dict")),
        "load 直後の state_dict が元と一致しない"
    );

    let mut p_a = params.clone();
    let mut p_b = params;
    for k in warm..warm + 3 {
        p_a = original.drive(&p_a, &grads(k));
        p_b = resumed.drive(&p_b, &grads(k));
        for (a, b) in p_a.iter().zip(p_b.iter()) {
            assert_eq!(bits(a), bits(b), "step {k} の更新後パラメータが一致しない");
        }
    }
    assert_eq!(
        state_bits(&original.state_dict().expect("state_dict")),
        state_bits(&resumed.state_dict().expect("state_dict")),
    );
}

macro_rules! resume_tests {
    ($($name:ident => $make:expr),* $(,)?) => {$(
        #[test]
        fn $name() {
            assert_resume_is_bit_exact(|| $make, 3, false);
            assert_resume_is_bit_exact(|| $make, 3, true);
        }
    )*};
}

resume_tests! {
    adamw_resumes_bit_exact => AdamW::new(AdamWConfig::default()).unwrap(),
    adam_resumes_bit_exact => Adam::new(AdamConfig::default()).unwrap(),
    rmsprop_resumes_bit_exact => RmsProp::new(RmsPropConfig::default()).unwrap(),
    adagrad_resumes_bit_exact => Adagrad::new(AdagradConfig::default()).unwrap(),
    lamb_resumes_bit_exact => Lamb::new(LambConfig::default()).unwrap(),
    adadelta_resumes_bit_exact => Adadelta::new(AdadeltaConfig::default()).unwrap(),
    adamax_resumes_bit_exact => Adamax::new(AdamaxConfig::default()).unwrap(),
    nadam_resumes_bit_exact => NAdam::new(NAdamConfig::default()).unwrap(),
    radam_resumes_bit_exact => RAdam::new(RAdamConfig::default()).unwrap(),
    sgd_with_momentum_resumes_bit_exact =>
        Sgd::new(SgdConfig::new(0.05).with_momentum(0.9)).unwrap(),
    sgd_without_momentum_resumes_bit_exact => Sgd::new(SgdConfig::new(0.05)).unwrap(),
}

#[test]
fn resume_before_first_step_is_bit_exact() {
    // 初回 step 前（スロットキーなし）の状態を復元しても、続きの step が一致する。
    assert_resume_is_bit_exact(|| AdamW::new(AdamWConfig::default()).unwrap(), 0, false);
}

#[test]
fn sgd_without_momentum_has_no_slot_keys() {
    let mut opt = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let _ = opt.drive(&initial_params(), &grads(0));
    let sd = opt.state_dict().unwrap();
    assert!(
        sd.keys().all(|k| !k.starts_with("state.")),
        "momentum なしの Sgd はスロットキーを持たない: {:?}",
        sd.keys().collect::<Vec<_>>()
    );
}

fn warmed_adamw() -> (AdamW, State) {
    let mut opt = AdamW::new(AdamWConfig::default()).unwrap();
    let mut params = initial_params();
    for k in 0..2 {
        params = opt.drive(&params, &grads(k));
    }
    let sd = opt.state_dict().unwrap();
    (opt, sd)
}

/// 失敗する `load_state_dict` が、対象 optimizer の状態を一切変えないこと。
fn assert_load_fails_without_mutation(mutate: impl FnOnce(&mut State)) {
    let (_, mut bad) = warmed_adamw();
    mutate(&mut bad);

    // 別の状態を持つ optimizer へ失敗する load を試みる。
    let mut target = AdamW::new(AdamWConfig::default()).unwrap();
    let _ = target.drive(&initial_params(), &grads(7));
    let before = state_bits(&target.state_dict().unwrap());

    assert!(target.load_state_dict(bad).is_err());
    assert_eq!(
        before,
        state_bits(&target.state_dict().unwrap()),
        "失敗した load が状態を変えた"
    );
}

#[test]
fn load_rejects_missing_key_without_mutation() {
    assert_load_fails_without_mutation(|sd| {
        let key = sd
            .keys()
            .find(|k| k.starts_with("state.0."))
            .cloned()
            .expect("slot key");
        sd.remove(&key);
    });
}

#[test]
fn load_rejects_extra_key_without_mutation() {
    assert_load_fails_without_mutation(|sd| {
        sd.insert("unexpected".to_string(), t(vec![1.0], &[1]));
    });
}

#[test]
fn load_rejects_shape_mismatch_without_mutation() {
    assert_load_fails_without_mutation(|sd| {
        let key = sd
            .keys()
            .find(|k| k.starts_with("state.0."))
            .cloned()
            .expect("slot key");
        sd.insert(key, t(vec![0.0; 2], &[2]));
    });
}

#[test]
fn load_rejects_state_of_another_optimizer_kind() {
    let mut adam = Adam::new(AdamConfig::default()).unwrap();
    let _ = adam.drive(&initial_params(), &grads(0));
    let adam_state = adam.state_dict().unwrap();

    let mut adamw = AdamW::new(AdamWConfig::default()).unwrap();
    let _ = adamw.drive(&initial_params(), &grads(1));
    let before = state_bits(&adamw.state_dict().unwrap());

    assert!(adamw.load_state_dict(adam_state).is_err());
    assert_eq!(before, state_bits(&adamw.state_dict().unwrap()));
}
