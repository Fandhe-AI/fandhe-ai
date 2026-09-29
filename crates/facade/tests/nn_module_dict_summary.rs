//! facade 側 `fandhe_ai::nn::{ModuleDict, summary}`（イシュー #2402。autodiff #2134 の鏡写し）の
//! 統合テスト。facade だけに依存して独自層を保持し、キー検証・挿入順・`set_parameter` の振り分け・
//! `state_dict` 往復・モード伝播と、`summary` の文字列形式（#2134 が facade 上で満たせなかった
//! 受け入れ条件）を固定する（`fandhe_ai_autodiff` は import しない。ビット一致で比較する）。

use std::cell::Cell;
use std::rc::Rc;

use fandhe_ai::nn::{Module, ModuleDict, Sequential, summary};
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

/// 学習可能パラメータ 8 要素（weight 6 + bias 2）を持つ葉層。
struct Lin {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    seen: Rc<Cell<bool>>,
}

impl Lin {
    fn new(base: f32) -> Self {
        Self::observed(base, Rc::new(Cell::new(true)))
    }
    fn observed(base: f32, seen: Rc<Cell<bool>>) -> Self {
        Self {
            weight: t32(&[base, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, base], &[2]),
            seen,
        }
    }
}

impl Module for Lin {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let w = tape.var(&self.weight);
        let b = tape.var(&self.bias);
        input.matmul(&w)?.add(&b)
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
        self.seen.set(training);
    }
    fn training(&self) -> bool {
        self.seen.get()
    }
}

/// パラメータを持たないゼロサイズの層。
struct Act;

impl Module for Act {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
}

/// ジェネリクス付き型名の短縮確認用ラッパー（葉）。
struct Wrap<T: Module>(T);

impl<T: Module> Module for Wrap<T> {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.0.forward(tape, input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.0.named_parameters()
    }
}

/// `children()` が自分自身を返す循環 `Module`。
struct Cyc;

impl Module for Cyc {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("me".to_string(), self as &dyn Module)]
    }
}

/// 同じ子を 2 か所から返す共有子を持つ `Module`。
struct Twice {
    a: Lin,
}

impl Module for Twice {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.a.forward(tape, input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.a.named_parameters()
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("x".into(), &self.a), ("y".into(), &self.a)]
    }
}

fn dict() -> ModuleDict {
    let mut d = ModuleDict::new();
    d.insert("enc", Box::new(Lin::new(0.1))).expect("insert");
    d.insert("dec", Box::new(Lin::new(0.9))).expect("insert");
    d
}

#[test]
fn rejects_invalid_keys_without_mutation() {
    let mut d = dict();
    for key in ["", "a.b", "."] {
        let err = d
            .insert(key, Box::new(Act))
            .err()
            .expect("不正キーは Err")
            .to_string();
        assert!(err.contains("ModuleDict"), "{err}");
    }
    assert_eq!(d.len(), 2);

    let r = ModuleDict::from_pairs(vec![
        ("ok".to_string(), Box::new(Act) as Box<dyn Module>),
        ("bad.key".to_string(), Box::new(Act)),
    ]);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn preserves_insertion_order_and_replaces_in_place() {
    let mut d = dict();
    d.insert("mid", Box::new(Act)).expect("insert");
    assert_eq!(d.keys().collect::<Vec<_>>(), ["enc", "dec", "mid"]);
    assert_eq!(
        d.iter().map(|(k, _)| k).collect::<Vec<_>>(),
        ["enc", "dec", "mid"]
    );
    let names: Vec<String> = d.children().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["enc", "dec", "mid"]);
    let params: Vec<String> = d.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(params, ["enc.weight", "enc.bias", "dec.weight", "dec.bias"]);

    // 同名キーの挿入は置換して旧値を返し、位置を保つ。
    let old = d.insert("dec", Box::new(Act)).expect("insert");
    assert!(old.is_some());
    assert_eq!(d.keys().collect::<Vec<_>>(), ["enc", "dec", "mid"]);
    assert_eq!(d.parameter_count(), 8);

    assert!(d.remove("enc").is_some());
    assert!(d.remove("enc").is_none());
    assert_eq!(d.keys().collect::<Vec<_>>(), ["dec", "mid"]);
    assert!(d.contains_key("mid") && !d.contains_key("enc"));
    assert!(d.get("mid").is_some() && d.get("enc").is_none());
    assert!(d.get_mut("mid").is_some());
    assert_eq!(d.iter_mut().count(), 2);

    let empty = ModuleDict::default();
    assert!(empty.is_empty());
}

#[test]
fn set_parameter_routes_by_key_and_fails_closed() {
    let mut d = dict();
    let v = t32(&[1.0, 2.0], &[2]);
    d.set_parameter("dec.bias", v.clone()).expect("set");
    let got = d
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "dec.bias")
        .map(|(_, t)| bits(t))
        .expect("dec.bias");
    assert_eq!(got, bits(&v));

    for name in ["bias", "nokey.bias", "enc.nope"] {
        let r = d.set_parameter(name, v.clone());
        assert!(
            matches!(r, Err(AutodiffError::InvalidArgument(_))),
            "name={name}"
        );
    }
}

#[test]
fn state_dict_roundtrip_is_bit_exact_and_strict() {
    let src = dict();
    let sd = src.state_dict();
    assert_eq!(sd.len(), 4);

    let mut dst = ModuleDict::new();
    dst.insert("enc", Box::new(Lin::new(7.0))).expect("insert");
    dst.insert("dec", Box::new(Lin::new(8.0))).expect("insert");
    dst.load_state_dict(sd.clone()).expect("load");
    for (n, t) in src.named_parameters() {
        let other = dst
            .named_parameters()
            .into_iter()
            .find(|(m, _)| *m == n)
            .map(|(_, t)| bits(t))
            .expect("key");
        assert_eq!(bits(t), other, "{n}");
    }

    let mut missing = sd;
    missing.remove("enc.weight");
    assert!(dst.load_state_dict(missing).is_err());
}

#[test]
fn forward_is_rejected() {
    let d = dict();
    let t = tape();
    let x = t.var(&t32(&[0.0; 3], &[1, 3]));
    let r = d.forward(TapeRef::from(&t), &x);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
}

#[test]
fn set_training_propagates_and_own_flag_wins() {
    let seen = Rc::new(Cell::new(true));
    let mut d = ModuleDict::new();
    d.insert("l", Box::new(Lin::observed(0.1, seen.clone())))
        .expect("insert");
    assert!(d.training());
    d.set_training(false);
    assert!(!d.training());
    assert!(!seen.get());
    d.set_training(true);
    assert!(d.training() && seen.get());
}

#[test]
fn nests_in_sequential_and_works_via_dyn() {
    let mut seq = Sequential::new();
    seq.push(Box::new(Lin::new(0.1)));
    seq.push(Box::new(dict()));
    let names: Vec<String> = seq.named_modules().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["0", "1", "1.enc", "1.dec"]);

    let boxed: Box<dyn Module> = Box::new(dict());
    assert_eq!(boxed.parameter_count(), 16);
}

#[test]
fn summary_leaf_root() {
    assert_eq!(
        summary(&Lin::new(0.1)),
        "Lin [params: 8]\nSubmodules: 0\nTotal parameters: 8\n"
    );
}

#[test]
fn summary_nested_format_is_fixed() {
    let mut seq = Sequential::new();
    seq.push(Box::new(Lin::new(0.1)));
    seq.push(Box::new(Act));
    seq.push(Box::new(dict()));
    let expected = "\
Sequential(
  (0): Lin [params: 8]
  (1): Act [params: 0]
  (2): ModuleDict(
    (enc): Lin [params: 8]
    (dec): Lin [params: 8]
  ) [params: 16]
) [params: 24]
Submodules: 5
Total parameters: 24
";
    assert_eq!(summary(&seq), expected);
}

#[test]
fn summary_empty_dict_is_leaf() {
    assert_eq!(
        summary(&ModuleDict::new()),
        "ModuleDict [params: 0]\nSubmodules: 0\nTotal parameters: 0\n"
    );
}

#[test]
fn summary_strips_generics() {
    let out = summary(&Wrap(Lin::new(0.1)));
    assert!(out.starts_with("Wrap [params: 8]\n"), "{out}");
}

#[test]
fn summary_keeps_repeated_zst_children() {
    let mut seq = Sequential::new();
    seq.push(Box::new(Act));
    seq.push(Box::new(Act));
    seq.push(Box::new(Act));
    let expected = "\
Sequential(
  (0): Act [params: 0]
  (1): Act [params: 0]
  (2): Act [params: 0]
) [params: 0]
Submodules: 3
Total parameters: 0
";
    assert_eq!(summary(&seq), expected);
}

#[test]
fn summary_terminates_on_cycle() {
    assert_eq!(
        summary(&Cyc),
        "Cyc(\n) [params: 0]\nSubmodules: 0\nTotal parameters: 0\n"
    );
}

#[test]
fn summary_dedups_shared_child_to_first_path() {
    let m = Twice { a: Lin::new(0.1) };
    let expected = "\
Twice(
  (x): Lin [params: 8]
) [params: 8]
Submodules: 1
Total parameters: 8
";
    assert_eq!(summary(&m), expected);
}
