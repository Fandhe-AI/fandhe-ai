//! facade だけで書いたユーザー定義層（residual block）を `fandhe_ai::nn::Sequential` に積む
//! 端から端までの統合テスト（イシュー #2399・親 #2338 受け入れ条件 4）。
//!
//! 役割: `nn::Module`（#2395）・`nn::Sequential`（#2396）・crate 内アダプタ経由の
//! `load_state_dict`（#2397）が、利用者コードから「学習（loss 減少）・`state_dict` 往復・
//! `set_training` 伝播」まで通ることを固定する。
//!
//! **import 契約**: `fandhe_ai` と `bench_harness::rng` 以外は import しない
//! （`grep -n "fandhe_ai_autodiff\|fandhe_ai_tensor_core"` が 0 件。`optim_train_loop.rs` と同型）。
//!
//! **パラメータ勾配の取り出し方（葉プレフィックス方式）**: facade `nn::Module::forward` は
//! パラメータを層内部で `tape.var` 登録するため、公開 API で葉の `Var` を取り戻す経路は
//! `Tape::leaf(i)` だけである。葉プレフィックスは最初の非葉 op で固定されるので、
//! パラメータ持ち層を `Sequential` の index 0 に置き、その `forward` が最初の op より前に
//! 全パラメータを `named_parameters()` と同じ順で登録する。後段は無パラメータ層のみとする。
//! 対応のずれは毎 step の葉数・葉値の bit 一致 assert と、手組み参照との bit 一致で
//! fail-closed に検出する。
//!
//! **決定的シード**: `Xorshift64Star`。新規 tolerance は設けず、bit 一致と既存形式の
//! loss 減少判定（`< 0.5 * initial`）のみを使う。実機非依存のため `#[ignore]` は付けない。

use std::collections::HashMap;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::nn::{Module, Sequential};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, Tape, TapeRef, Tensor, Var};

const BATCH: usize = 4;
const D: usize = 4;
const H: usize = 8;
const PARAM_NAMES: [&str; 4] = ["w1", "b1", "w2", "b2"];

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("test fixture: host tensor のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn rand_tensor(rng: &mut Xorshift64Star, shape: &[usize], scale: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = rng.fill_vec(n).into_iter().map(|v| v * scale).collect();
    Tensor::new(data, shape).expect("test fixture: shape 構築")
}

/// `x + (tanh(x @ w1 + b1) @ w2 + b2)` を計算するユーザー定義 residual block。
struct ResidualBlock {
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
    training: bool,
}

impl ResidualBlock {
    fn new(seed: u64) -> Self {
        let mut rng = Xorshift64Star::new(seed);
        Self {
            w1: rand_tensor(&mut rng, &[D, H], 0.5),
            b1: rand_tensor(&mut rng, &[H], 0.1),
            w2: rand_tensor(&mut rng, &[H, D], 0.5),
            b2: rand_tensor(&mut rng, &[D], 0.1),
            training: true,
        }
    }
}

impl Module for ResidualBlock {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        // 最初の op より前に、named_parameters と同じ順で全パラメータを葉登録する。
        let w1 = tape.var(&self.w1);
        let b1 = tape.var(&self.b1);
        let w2 = tape.var(&self.w2);
        let b2 = tape.var(&self.b2);
        let h = input.matmul(&w1)?.add(&b1)?.tanh();
        input.add(&h.matmul(&w2)?.add(&b2)?)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![
            ("w1".into(), &self.w1),
            ("b1".into(), &self.b1),
            ("w2".into(), &self.w2),
            ("b2".into(), &self.b2),
        ]
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let slot = match name {
            "w1" => &mut self.w1,
            "b1" => &mut self.b1,
            "w2" => &mut self.w2,
            "b2" => &mut self.b2,
            _ => {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ResidualBlock::set_parameter: unknown `{name}`"
                )));
            }
        };
        if slot.shape() != value.shape() {
            return Err(AutodiffError::InvalidArgument(format!(
                "ResidualBlock::set_parameter: shape mismatch for `{name}`"
            )));
        }
        *slot = value;
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }
}

/// 無パラメータのモード依存層。学習時は恒等、評価時は `tanh`。
struct ModeAct {
    training: bool,
}

impl ModeAct {
    fn new() -> Self {
        Self { training: true }
    }
}

impl Module for ModeAct {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(if self.training { *input } else { input.tanh() })
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    fn training(&self) -> bool {
        self.training
    }
}

/// index 0 に residual block、後段は無パラメータ層（入れ子 `Sequential` を含む）。
fn build_model(seed: u64) -> Sequential {
    Sequential::new()
        .add(ResidualBlock::new(seed))
        .add(ModeAct::new())
        .add(Sequential::new().add(ModeAct::new()))
}

fn gen_data() -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    (
        rand_tensor(&mut rng, &[BATCH, D], 1.0),
        rand_tensor(&mut rng, &[BATCH, D], 1.0),
    )
}

/// 葉プレフィックスのインベントリ assert（leaf 0 = x、以降 = パラメータ）。
fn assert_leaf_inventory(tape: &Tape, model: &dyn Module) {
    let params = model.named_parameters();
    assert_eq!(
        tape.leaf_count(),
        1 + params.len(),
        "葉数が 1 + パラメータ数と一致しない（パラメータ持ち層が増えた可能性）"
    );
    for (i, (name, value)) in params.iter().enumerate() {
        let leaf = tape.leaf(i + 1).expect("test fixture: 葉が取得できる");
        assert_eq!(
            bits(&leaf.value()),
            bits(value),
            "葉 {} と named_parameters `{name}` の値が一致しない",
            i + 1
        );
    }
}

/// 1 step 分の (loss, パラメータ勾配) を Sequential 経路で計算する。
fn step_grads(model: &Sequential, x: &Tensor<f32>, y: &Tensor<f32>) -> (f32, Vec<Tensor<f32>>) {
    let tape = fandhe_ai::tape();
    let xv = tape.var(x);
    let pred = model
        .forward(TapeRef::from(&tape), &xv)
        .expect("test fixture: forward");
    // y は forward の後で登録するため、葉プレフィックス（x + パラメータ）の外にある。
    let yv = tape.var(y);
    let loss = pred.mse_loss(&yv).expect("test fixture: mse_loss");
    let loss_value = loss.to_tensor().get(&[]).expect("scalar");
    let grads = tape.backward(&loss).expect("test fixture: backward");
    assert_leaf_inventory(&tape, model);
    let n = model.named_parameters().len();
    let gs = (1..=n)
        .map(|i| {
            let leaf = tape.leaf(i).expect("leaf");
            grads
                .get(&leaf)
                .expect("get")
                .expect("パラメータ勾配が存在する")
                .clone()
        })
        .collect();
    (loss_value, gs)
}

/// SGD 1 step を計算し、`名前 -> 更新後テンソル` を返す。
fn sgd_update(
    model: &Sequential,
    sgd: &mut Sgd,
    grads: &[Tensor<f32>],
) -> HashMap<String, Tensor<f32>> {
    let named = model.named_parameters();
    let params: Vec<&Tensor<f32>> = named.iter().map(|(_, t)| *t).collect();
    let grad_refs: Vec<&Tensor<f32>> = grads.iter().collect();
    let new = sgd
        .step(&params, &grad_refs)
        .expect("test fixture: Sgd::step");
    named.iter().map(|(n, _)| n.clone()).zip(new).collect()
}

#[test]
fn user_defined_block_trains_via_facade_nn_sequential() {
    const STEPS: usize = 100;
    const LR: f32 = 0.05;
    let (x, y) = gen_data();
    let mut model = build_model(0x1111);
    let mut sgd = Sgd::new(SgdConfig::new(LR)).expect("test fixture: Sgd::new");
    let mut log = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let (loss, grads) = step_grads(&model, &x, &y);
        log.push(loss);
        let updated = sgd_update(&model, &mut sgd, &grads);
        // load_state_dict は facade Module 既定実装（crate 内アダプタ経由の two-pass 検証・
        // キー昇順適用）を毎 step 通す公開経路である。
        model
            .load_state_dict(updated)
            .expect("test fixture: load_state_dict");
    }

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let last = *log.last().expect("non-empty");
    assert!(initial.is_finite() && last.is_finite(), "loss は有限");
    assert!(
        last < 0.5 * initial,
        "loss が十分減少しない: initial={initial} final={last}"
    );
}

#[test]
fn leaf_mapping_matches_hand_composition_bit_exact() {
    let (x, y) = gen_data();
    let model = build_model(0x1111);
    let (loss_seq, grads_seq) = step_grads(&model, &x, &y);

    // 手組み参照: パラメータ Var を明示登録して同じ数式を組む。
    let block = ResidualBlock::new(0x1111);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let w1 = tape.var(&block.w1);
    let b1 = tape.var(&block.b1);
    let w2 = tape.var(&block.w2);
    let b2 = tape.var(&block.b2);
    let h = xv.matmul(&w1).unwrap().add(&b1).unwrap().tanh();
    let out = xv.add(&h.matmul(&w2).unwrap().add(&b2).unwrap()).unwrap();
    let yv = tape.var(&y);
    let loss = out.mse_loss(&yv).unwrap();
    let grads = tape.backward(&loss).unwrap();

    assert_eq!(
        loss.to_tensor().get(&[]).unwrap().to_bits(),
        loss_seq.to_bits()
    );
    for (i, v) in [w1, b1, w2, b2].iter().enumerate() {
        let g = grads.get(v).unwrap().expect("勾配あり");
        assert_eq!(
            bits(g),
            bits(&grads_seq[i]),
            "勾配 {} が手組みと一致しない",
            PARAM_NAMES[i]
        );
    }
}

#[test]
fn state_dict_roundtrip_restores_trained_model_bit_exact() {
    let (x, y) = gen_data();
    let mut model = build_model(0x1111);
    let mut sgd = Sgd::new(SgdConfig::new(0.05)).expect("Sgd::new");
    for _ in 0..5 {
        let (_, grads) = step_grads(&model, &x, &y);
        let map = sgd_update(&model, &mut sgd, &grads);
        model.load_state_dict(map).expect("load");
    }

    let sd = model.state_dict();
    let mut keys: Vec<&str> = sd.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["0.b1", "0.b2", "0.w1", "0.w2"]);

    let mut other = build_model(0x9999);
    other.load_state_dict(sd.clone()).expect("往復 load");

    let forward_bits = |m: &Sequential| {
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        bits(&m.forward(TapeRef::from(&tape), &xv).unwrap().value())
    };
    assert_eq!(forward_bits(&model), forward_bits(&other));

    // fail-closed: 欠落・余剰・shape 不一致は Err で状態不変。
    let snapshot = |m: &Sequential| -> HashMap<String, Vec<u32>> {
        m.state_dict()
            .iter()
            .map(|(k, v)| (k.clone(), bits(v)))
            .collect()
    };
    let before = snapshot(&other);
    let mut missing = sd.clone();
    missing.remove("0.w1");
    let mut extra = sd.clone();
    extra.insert("0.zz".into(), sd["0.b1"].clone());
    let mut bad_shape = sd.clone();
    bad_shape.insert("0.b1".into(), sd["0.b2"].clone());
    // 上の b1 は [H]、b2 は [D] のため shape が異なる。
    for bad in [missing, extra, bad_shape] {
        assert!(other.load_state_dict(bad).is_err());
        assert_eq!(before, snapshot(&other), "失敗時に状態が変化した");
    }
}

#[test]
fn set_training_propagates_through_nested_sequential() {
    let (x, _) = gen_data();
    let mut model = build_model(0x1111);
    assert!(model.training());
    assert!(model.layers().iter().all(|l| l.training()));

    let run = |m: &Sequential| {
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        bits(&m.forward(TapeRef::from(&tape), &xv).unwrap().value())
    };
    let train_out = run(&model);

    model.set_training(false);
    assert!(!model.training());
    assert!(model.layers().iter().all(|l| !l.training()));
    let eval_out = run(&model);
    assert_ne!(train_out, eval_out, "モードで出力が変わるはず");

    // 評価時: ModeAct が 2 段（外側 + 入れ子）で tanh を 2 回適用する。
    let block = ResidualBlock::new(0x1111);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let r = block.forward(TapeRef::from(&tape), &xv).unwrap();
    let expected = r.tanh().tanh();
    assert_eq!(eval_out, bits(&expected.value()));

    model.set_training(true);
    assert!(model.layers().iter().all(|l| l.training()));
    assert_eq!(run(&model), train_out);

    // dyn Module としても扱える。
    fn takes_dyn(m: &dyn Module) -> bool {
        m.training()
    }
    assert!(takes_dyn(&model));
}
