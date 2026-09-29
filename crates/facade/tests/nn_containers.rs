//! facade 側コンテナ `fandhe_ai::nn::{ModuleList, Sequential}`（#2396）の統合テスト。
//! facade だけに依存して独自層を積み、forward／backward・`state_dict` 往復・fail-closed・
//! モード伝播が autodiff 側コンテナと同じ契約で動くことを固定する
//! （`fandhe_ai_autodiff` は import しない。tolerance は使わずビット一致で比較する）。

use std::cell::Cell;
use std::rc::Rc;

use fandhe_ai::nn::{Module, ModuleList, Sequential};
use fandhe_ai::{AutodiffError, TapeRef, Tensor, Var, tape};

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, shape).expect("テスト入力の構築")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("host tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `tanh(x @ weight + bias)` を返す独自層。モードは共有フラグで外部から観測できる。
struct Affine {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    training: bool,
    seen: Rc<Cell<bool>>,
}

impl Affine {
    fn new() -> Self {
        Self::observed(Rc::new(Cell::new(true)))
    }
    fn observed(seen: Rc<Cell<bool>>) -> Self {
        Self {
            weight: t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, -0.05], &[2]),
            training: true,
            seen,
        }
    }
}

impl Module for Affine {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let w = tape.var(&self.weight);
        let b = tape.var(&self.bias);
        Ok(input.matmul(&w)?.add(&b)?.tanh())
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("weight".into(), &self.weight), ("bias".into(), &self.bias)]
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let slot = match name {
            "weight" => &mut self.weight,
            "bias" => &mut self.bias,
            _ => return Err(AutodiffError::InvalidArgument(format!("unknown `{name}`"))),
        };
        if slot.shape() != value.shape() {
            return Err(AutodiffError::InvalidArgument("shape".into()));
        }
        *slot = value;
        Ok(())
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.seen.set(training);
    }
    fn training(&self) -> bool {
        self.training
    }
}

/// 無状態の恒等層（既定実装のみ）。
struct Identity;
impl Module for Identity {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
}

fn input() -> Tensor<f32> {
    t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3])
}

/// 手動合成 `tanh(x @ w + b)` の (loss, dx) をビット列で返す。
fn manual_loss_and_dx(x: &Tensor<f32>) -> (Vec<u32>, Vec<u32>) {
    let layer = Affine::new();
    let t = tape();
    let xv = t.var(x);
    let (w, b) = (t.var(&layer.weight), t.var(&layer.bias));
    let y = xv.matmul(&w).expect("mm").add(&b).expect("add").tanh();
    let loss = y.mul(&y).expect("mul").sum(None).expect("sum");
    let g = t.backward(&loss).expect("backward");
    (
        bits(&loss.value()),
        bits(g.get(&xv).expect("get").expect("dx")),
    )
}

fn container_loss_and_dx(m: &dyn Module, x: &Tensor<f32>) -> (Vec<u32>, Vec<u32>) {
    let t = tape();
    let xv = t.var(x);
    let y = m.forward(TapeRef::from(&t), &xv).expect("forward");
    let loss = y.mul(&y).expect("mul").sum(None).expect("sum");
    let g = t.backward(&loss).expect("backward");
    (
        bits(&loss.value()),
        bits(g.get(&xv).expect("get").expect("dx")),
    )
}

#[test]
fn sequential_add_and_push_match_hand_composition() {
    let x = input();
    let want = manual_loss_and_dx(&x);

    let chained = Sequential::new().add(Affine::new()).add(Identity);
    assert_eq!(container_loss_and_dx(&chained, &x), want);

    let mut pushed = Sequential::new();
    pushed.push(Box::new(Affine::new()));
    pushed.push(Box::new(Identity));
    assert_eq!(pushed.len(), 2);
    assert!(!pushed.is_empty());
    assert_eq!(container_loss_and_dx(&pushed, &x), want);
}

#[test]
fn state_dict_uses_index_prefix_and_roundtrips() {
    let mut seq = Sequential::new().add(Affine::new()).add(Identity);
    let names: Vec<String> = seq.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["0.weight", "0.bias"]);

    let mut sd = seq.state_dict();
    let mut keys: Vec<_> = sd.keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["0.bias", "0.weight"]);
    let new_w = t32(&[1.0; 6], &[3, 2]);
    sd.insert("0.weight".into(), new_w.clone());
    seq.load_state_dict(sd).expect("load");
    assert_eq!(bits(&seq.state_dict()["0.weight"]), bits(&new_w));
}

#[test]
fn nested_containers_use_nested_prefix_and_route_set_parameter() {
    let inner: ModuleList = vec![
        Box::new(Identity) as Box<dyn Module>,
        Box::new(Affine::new()),
    ]
    .into_iter()
    .collect();
    let mut outer = Sequential::new().add(inner);
    let names: Vec<String> = outer
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, ["0.1.weight", "0.1.bias"]);
    let new_b = t32(&[9.0, 8.0], &[2]);
    outer
        .set_parameter("0.1.bias", new_b.clone())
        .expect("route");
    assert_eq!(bits(&outer.state_dict()["0.1.bias"]), bits(&new_b));
}

#[test]
fn set_parameter_is_fail_closed_and_state_unchanged() {
    let mut seq = Sequential::new().add(Affine::new());
    let before: Vec<Vec<u32>> = {
        let sd = seq.state_dict();
        let mut ks: Vec<_> = sd.keys().cloned().collect();
        ks.sort();
        ks.iter().map(|k| bits(&sd[k])).collect()
    };
    for key in [
        "weight",
        "x.weight",
        "9.weight",
        "18446744073709551616.weight",
        "0.unknown",
        "",
    ] {
        let e = seq.set_parameter(key, t32(&[0.0; 2], &[2]));
        assert!(e.is_err(), "`{key}` は Err");
    }
    let after: Vec<Vec<u32>> = {
        let sd = seq.state_dict();
        let mut ks: Vec<_> = sd.keys().cloned().collect();
        ks.sort();
        ks.iter().map(|k| bits(&sd[k])).collect()
    };
    assert_eq!(before, after);
}

#[test]
fn load_state_dict_is_fail_closed() {
    let mut seq = Sequential::new().add(Affine::new());
    let good = seq.state_dict();

    let mut missing = good.clone();
    missing.remove("0.bias");
    let e = seq.load_state_dict(missing).expect_err("missing");
    assert!(e.to_string().contains("missing keys"), "{e}");

    let mut extra = good.clone();
    extra.insert("1.weight".into(), t32(&[0.0], &[1]));
    let e = seq.load_state_dict(extra).expect_err("unexpected");
    assert!(e.to_string().contains("unexpected keys"), "{e}");

    let mut bad = good.clone();
    bad.insert("0.weight".into(), t32(&[0.0; 4], &[2, 2]));
    let e = seq.load_state_dict(bad).expect_err("shape");
    assert!(e.to_string().contains("shape mismatch"), "{e}");
    assert_eq!(bits(&seq.state_dict()["0.weight"]), bits(&good["0.weight"]));
}

#[test]
fn set_training_propagates_and_push_does_not_sync() {
    let seen = Rc::new(Cell::new(true));
    let mut seq = Sequential::new().add(Affine::observed(seen.clone()));
    assert!(seq.training());
    seq.set_training(false);
    assert!(!seq.training());
    assert!(!seen.get());

    let late = Rc::new(Cell::new(true));
    seq.push(Box::new(Affine::observed(late.clone())));
    assert!(!seq.training());
    assert!(late.get(), "push した子のモードは同期されない");
    seq.set_training(false);
    assert!(!late.get(), "次の set_training で伝わる");
}

#[test]
fn empty_containers() {
    let list = ModuleList::new();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
    assert!(list.training());
    let t = tape();
    let x = t.var(&input());
    let e = list.forward(TapeRef::from(&t), &x);
    assert!(matches!(e, Err(AutodiffError::InvalidArgument(_))));

    let mut seq = Sequential::new();
    assert!(seq.is_empty());
    let y = seq.forward(TapeRef::from(&t), &x).expect("identity");
    assert_eq!(bits(&y.value()), bits(&input()));
    assert!(seq.named_parameters().is_empty());
    assert!(seq.state_dict().is_empty());
    seq.load_state_dict(Default::default()).expect("empty ok");
    let mut nonempty = seq.state_dict();
    nonempty.insert("0.w".into(), t32(&[0.0], &[1]));
    assert!(seq.load_state_dict(nonempty).is_err());
}

#[test]
fn accessors_conversions_and_dyn_nesting() {
    let mut list: ModuleList = (0..3)
        .map(|_| Box::new(Identity) as Box<dyn Module>)
        .collect();
    assert_eq!(list.len(), 3);
    assert!(list.get(3).is_none());
    assert!(list.get(2).is_some());
    assert!(list.get_mut(0).is_some());
    assert_eq!(list.iter().count(), 3);
    assert_eq!(list.iter_mut().count(), 3);

    list.set_training(false);
    let mut seq: Sequential = list.into();
    assert_eq!(seq.len(), 3);
    assert_eq!(seq.layers().len(), 3);
    assert_eq!(seq.layers_mut().len(), 3);
    assert!(!seq.training(), "From<ModuleList> は training を引き継ぐ");

    let boxed: Vec<Box<dyn Module>> = vec![Box::new(seq), Box::new(ModuleList::default())];
    assert_eq!(boxed.len(), 2);
}
