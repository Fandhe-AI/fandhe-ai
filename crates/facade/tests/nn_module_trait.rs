//! facade 独自 `fandhe_ai::nn::Module`（#2395）の統合テスト。facade だけに依存して
//! 独自層を定義でき、forward／backward・`state_dict` 往復・fail-closed・ロールバックが
//! autodiff 側 trait と同じ契約で動くことを固定する（`fandhe_ai_autodiff` は import しない）。

use std::cell::Cell;
use std::collections::HashMap;

use fandhe_ai::nn::Module;
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

/// `tanh(x @ weight + bias)` を返す独自層。
struct Affine {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    training: bool,
}

impl Affine {
    fn new() -> Self {
        Self {
            weight: t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, -0.05], &[2]),
            training: true,
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
    }
    fn training(&self) -> bool {
        self.training
    }
}

/// 無状態層（既定実装のみ）。
struct Identity;
impl Module for Identity {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
}

/// `set_parameter("b")` が常に失敗する（ロールバック検証用）。
struct FailsOnB {
    a: Tensor<f32>,
    b: Tensor<f32>,
}
impl Module for FailsOnB {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("a".into(), &self.a), ("b".into(), &self.b)]
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        match name {
            "a" => self.a = value,
            _ => return Err(AutodiffError::InvalidArgument("b always fails".into())),
        }
        Ok(())
    }
}

/// `set_parameter` がちょうど 1 回だけ成功する（契約違反のロールバック失敗検証用）。
struct OnceOnly {
    a: Tensor<f32>,
    b: Tensor<f32>,
    calls: Cell<u32>,
}
impl Module for OnceOnly {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("a".into(), &self.a), ("b".into(), &self.b)]
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        if n >= 1 {
            return Err(AutodiffError::InvalidArgument("exhausted".into()));
        }
        if name == "a" {
            self.a = value;
        }
        Ok(())
    }
}

#[test]
fn dyn_module_is_usable() {
    let layers: Vec<Box<dyn Module>> = vec![Box::new(Affine::new()), Box::new(Identity)];
    let t = tape();
    let x = t.var(&t32(&[0.5, -1.0, 2.0], &[1, 3]));
    let y = layers[0].forward(TapeRef::from(&t), &x).expect("forward");
    let z = layers[1].forward(TapeRef::from(&t), &y).expect("forward");
    assert_eq!(z.value().shape(), &[1, 2]);
    fn _takes(_: &dyn Module) {}
    _takes(&Identity);
}

#[test]
fn forward_backward_matches_hand_composition() {
    let layer = Affine::new();
    let x = t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3]);

    let ta = tape();
    let xa = ta.var(&x);
    let ya = layer.forward(TapeRef::from(&ta), &xa).expect("forward");
    let la = ya.mul(&ya).expect("mul").sum(None).expect("sum");
    let ga = ta.backward(&la).expect("backward A");

    let tb = tape();
    let xb = tb.var(&x);
    let (w, b) = (tb.var(&layer.weight), tb.var(&layer.bias));
    let yb = xb.matmul(&w).expect("mm").add(&b).expect("add").tanh();
    let lb = yb.mul(&yb).expect("mul").sum(None).expect("sum");
    let gb = tb.backward(&lb).expect("backward B");

    assert_eq!(bits(&la.value()), bits(&lb.value()));
    let dxa = ga.get(&xa).expect("get").expect("dx");
    let dxb = gb.get(&xb).expect("get").expect("dx");
    assert_eq!(bits(dxa), bits(dxb));
}

#[test]
fn state_dict_roundtrip() {
    let mut layer = Affine::new();
    let mut sd = layer.state_dict();
    assert_eq!(sd.len(), 2);
    let new_w = t32(&[1.0; 6], &[3, 2]);
    sd.insert("weight".into(), new_w.clone());
    layer.load_state_dict(sd).expect("load");
    assert_eq!(bits(&layer.weight), bits(&new_w));
    let back = layer.state_dict();
    assert_eq!(bits(&back["weight"]), bits(&new_w));
}

#[test]
fn load_state_dict_is_fail_closed_and_state_unchanged() {
    let mut layer = Affine::new();
    let before = layer.state_dict();
    let snapshot = |l: &Affine| (bits(&l.weight), bits(&l.bias));
    let want = snapshot(&layer);

    let mut missing = before.clone();
    missing.remove("bias");
    let e = layer.load_state_dict(missing).expect_err("missing");
    assert!(e.to_string().contains("missing keys"), "{e}");

    let mut extra = before.clone();
    extra.insert("unknown".into(), t32(&[0.0], &[1]));
    let e = layer.load_state_dict(extra).expect_err("unexpected");
    assert!(e.to_string().contains("unexpected keys"), "{e}");

    let mut bad_shape = before.clone();
    bad_shape.insert("weight".into(), t32(&[0.0; 4], &[2, 2]));
    let e = layer.load_state_dict(bad_shape).expect_err("shape");
    assert!(e.to_string().contains("shape mismatch"), "{e}");

    assert_eq!(snapshot(&layer), want);

    let e = Identity
        .set_parameter_probe("nope")
        .expect_err("default set_parameter");
    assert!(e.to_string().contains("no parameter named `nope`"), "{e}");
}

trait Probe {
    fn set_parameter_probe(self, name: &str) -> Result<(), AutodiffError>;
}
impl<M: Module> Probe for M {
    fn set_parameter_probe(mut self, name: &str) -> Result<(), AutodiffError> {
        self.set_parameter(name, t32(&[0.0], &[1]))
    }
}

#[test]
fn load_state_dict_rolls_back_applied_keys() {
    let mut layer = FailsOnB {
        a: t32(&[1.0, 1.0], &[2]),
        b: t32(&[2.0, 2.0], &[2]),
    };
    let mut sd = HashMap::new();
    sd.insert("a".to_string(), t32(&[9.0, 9.0], &[2]));
    sd.insert("b".to_string(), t32(&[8.0, 8.0], &[2]));
    let e = layer.load_state_dict(sd).expect_err("b fails");
    assert!(e.to_string().contains("b always fails"), "{e}");
    assert_eq!(bits(&layer.a), bits(&t32(&[1.0, 1.0], &[2])));
}

#[test]
fn rollback_failure_reports_partial_application() {
    let mut layer = OnceOnly {
        a: t32(&[1.0, 1.0], &[2]),
        b: t32(&[2.0, 2.0], &[2]),
        calls: Cell::new(0),
    };
    let mut sd = HashMap::new();
    sd.insert("a".to_string(), t32(&[9.0, 9.0], &[2]));
    sd.insert("b".to_string(), t32(&[8.0, 8.0], &[2]));
    let e = layer.load_state_dict(sd).expect_err("rollback fails");
    assert!(e.to_string().contains("partially applied"), "{e}");
}

#[test]
fn stateless_defaults() {
    let mut l = Identity;
    assert!(l.named_parameters().is_empty());
    assert!(l.state_dict().is_empty());
    l.load_state_dict(HashMap::new()).expect("empty ok");
    let mut extra = HashMap::new();
    extra.insert("k".to_string(), t32(&[0.0], &[1]));
    assert!(l.load_state_dict(extra).is_err());
    assert!(l.training());
    l.set_training(false);
    assert!(l.training());
}

#[test]
fn mode_dependent_layer_overrides_via_dyn() {
    let mut l: Box<dyn Module> = Box::new(Affine::new());
    assert!(l.training());
    l.set_training(false);
    assert!(!l.training());
}
