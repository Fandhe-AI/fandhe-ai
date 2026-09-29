//! facade `nn::Module` の凍結 API（`set_requires_grad`／`freeze`／`requires_grad`。
//! #2400・親 #2338 承認事項 6。#2137 の鏡写し）の統合テスト。
//! facade の公開面だけに依存し（`fandhe_ai_autodiff` は import しない）、tolerance は使わず
//! `to_bits` の bit 一致で比較する。

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

/// `tanh(x @ weight + bias)`。`requires_grad` で葉登録を `var`／`var_no_grad` に切り替える
/// （`Module::set_requires_grad` の実装者契約）。
struct Affine {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    requires_grad: bool,
    training: bool,
}

impl Affine {
    fn new() -> Self {
        Self {
            weight: t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, -0.05], &[2]),
            requires_grad: true,
            training: true,
        }
    }
}

impl Module for Affine {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let reg = |p: &Tensor<f32>| {
            if self.requires_grad {
                tape.var(p)
            } else {
                tape.var_no_grad(p)
            }
        };
        let (w, b) = (reg(&self.weight), reg(&self.bias));
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
        *slot = value;
        Ok(())
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    fn training(&self) -> bool {
        self.training
    }
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.requires_grad = requires_grad;
        Ok(())
    }
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

/// パラメータを持つが `set_requires_grad` を override しない層（fail-closed 既定の検証用）。
struct Forgetful {
    p: Tensor<f32>,
}
impl Module for Forgetful {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.add(&tape.var(&self.p))
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("p".into(), &self.p)]
    }
}

struct Identity;
impl Module for Identity {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
}

fn input() -> Tensor<f32> {
    t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3])
}

/// (forward 出力 bits, dx bits, 葉 1・葉 2 の勾配取得が GradientTrackingDisabled か)。
fn run(m: &dyn Module) -> (Vec<u32>, Vec<u32>, bool, bool) {
    let t = tape();
    let x = t.var(&input());
    let y = m.forward(TapeRef::from(&t), &x).expect("forward");
    let loss = y.sum(None).expect("sum");
    let g = t.backward(&loss).expect("backward");
    let disabled = |i: usize| {
        let v = t.leaf(i).expect("leaf");
        matches!(g.get(&v), Err(AutodiffError::GradientTrackingDisabled))
    };
    (
        bits(&y.value()),
        bits(g.get(&x).expect("get").expect("dx")),
        disabled(1),
        disabled(2),
    )
}

#[test]
fn freeze_keeps_forward_and_dx_bit_identical_and_disables_param_grads() {
    let live = Affine::new();
    let mut frozen = Affine::new();
    frozen.freeze().expect("freeze");
    let (y0, dx0, w0, b0) = run(&live);
    let (y1, dx1, w1, b1) = run(&frozen);
    assert_eq!(y0, y1);
    assert_eq!(dx0, dx1);
    assert!(!w0 && !b0, "非凍結は勾配を持つ");
    assert!(w1 && b1, "凍結側は GradientTrackingDisabled");

    // コンテナ経由の freeze() でも同じ。
    let mut seq = Sequential::new().add(Affine::new());
    seq.freeze().expect("freeze");
    assert!(!seq.requires_grad());
    let (y2, dx2, w2, b2) = run(&seq);
    assert_eq!((y0, dx0), (y2, dx2));
    assert!(w2 && b2);

    seq.set_requires_grad(true).expect("unfreeze");
    assert!(seq.requires_grad());
    let (_, _, w3, b3) = run(&seq);
    assert!(!w3 && !b3);
}

/// パラメータ葉そのものを出力する層（葉 `Var` を直接観測するため）。
struct Probe {
    p: Tensor<f32>,
    requires_grad: bool,
}
impl Module for Probe {
    fn forward<'t>(&self, tape: TapeRef<'t>, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(if self.requires_grad {
            tape.var(&self.p)
        } else {
            tape.var_no_grad(&self.p)
        })
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("p".into(), &self.p)]
    }
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.requires_grad = requires_grad;
        Ok(())
    }
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

#[test]
fn freeze_takes_effect_from_next_forward_only() {
    let mut layer = Probe {
        p: t32(&[1.0, 2.0], &[2]),
        requires_grad: true,
    };
    let t = tape();
    let x = t.var(&input());
    let before = layer.forward(TapeRef::from(&t), &x).expect("forward");
    layer.freeze().expect("freeze");
    let after = layer.forward(TapeRef::from(&t), &x).expect("forward");
    let loss = before
        .sum(None)
        .expect("sum")
        .add(&after.sum(None).expect("sum"))
        .expect("add");
    let g = t.backward(&loss).expect("backward");
    assert!(
        g.get(&before)
            .expect("凍結前に登録した葉は勾配を持つ")
            .is_some()
    );
    assert!(matches!(
        g.get(&after),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
}

#[test]
fn freeze_is_independent_of_training_and_state_dict() {
    let mut layer = Affine::new();
    let mut keys_before: Vec<String> = layer.state_dict().into_keys().collect();
    keys_before.sort();
    layer.freeze().expect("freeze");
    assert!(layer.training(), "freeze は training を変えない");
    layer.set_training(false);
    assert!(
        !layer.requires_grad(),
        "set_training は requires_grad を変えない"
    );
    let sd = layer.state_dict();
    layer.load_state_dict(sd).expect("roundtrip");
    assert!(
        !layer.requires_grad(),
        "load_state_dict 後もフラグは保持される"
    );
    let mut keys_after: Vec<String> = layer.state_dict().into_keys().collect();
    keys_after.sort();
    assert_eq!(keys_before, keys_after);
}

#[test]
fn default_is_fail_closed_for_parameterized_layers() {
    let mut stateless = Identity;
    assert!(stateless.freeze().is_ok());
    assert!(stateless.requires_grad());

    let mut f = Forgetful {
        p: t32(&[1.0, 2.0], &[2]),
    };
    assert!(matches!(f.freeze(), Err(AutodiffError::InvalidArgument(_))));

    let mut seq = Sequential::new().add(Affine::new()).add(Forgetful {
        p: t32(&[1.0, 2.0], &[2]),
    });
    let e = seq.freeze().expect_err("後続が Err");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)));
    assert!(seq.requires_grad(), "手前の子は元へ戻る");
    assert!(seq.layers()[0].requires_grad());

    let mut list = ModuleList::new();
    list.push(Box::new(Affine::new()));
    list.freeze().expect("freeze");
    assert!(!list.requires_grad());
}
