//! facade `fandhe_ai::nn::Module` の introspection 4 メソッド
//! （`children`・`named_modules`・`parameter_count`・`type_name`。#2401・#2134 の鏡写し）の
//! 統合テスト。facade だけに依存し（`fandhe_ai_autodiff` は import しない）、循環・共有・
//! ZST・offset 0 の子でも autodiff 側と同じ契約で有限に列挙できることを固定する。
//! 数値経路を通らないため実機 `#[ignore]` テストはない。

use std::cell::Cell;

use fandhe_ai::nn::{Module, ModuleList, Sequential};
use fandhe_ai::{AutodiffError, TapeRef, Tensor, Var};

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, shape).expect("テスト入力の構築")
}

/// weight [3,2] + bias [2] = 8 要素。
struct Affine {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl Affine {
    fn new() -> Self {
        Self {
            weight: t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, -0.05], &[2]),
        }
    }
}

impl Module for Affine {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("weight".into(), &self.weight), ("bias".into(), &self.bias)]
    }
}

/// weight のみ [3,2] = 6 要素。
struct NoBias {
    weight: Tensor<f32>,
}

impl Module for NoBias {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("weight".into(), &self.weight)]
    }
}

/// 無状態の ZST 層。
struct Zst;

impl Module for Zst {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
}

fn names(m: &dyn Module) -> Vec<String> {
    m.named_modules().into_iter().map(|(n, _)| n).collect()
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

#[test]
fn leaf_defaults_are_empty() {
    assert!(Zst.children().is_empty());
    assert!(Zst.named_modules().is_empty());
    assert_eq!(Zst.parameter_count(), 0);
}

#[test]
fn parameter_count_sums_numel() {
    assert_eq!(Affine::new().parameter_count(), 8);
    assert_eq!(
        NoBias {
            weight: t32(&[0.0; 6], &[3, 2])
        }
        .parameter_count(),
        6
    );
    let mut seq = Sequential::new();
    seq.push(Box::new(Affine::new()));
    seq.push(Box::new(Zst));
    seq.push(Box::new(Affine::new()));
    let want: usize = seq.named_parameters().iter().map(|(_, t)| t.numel()).sum();
    assert_eq!(seq.parameter_count(), 16);
    assert_eq!(seq.parameter_count(), want);
}

#[test]
fn nested_enumeration_order() {
    let mut inner = Sequential::new();
    inner.push(Box::new(Affine::new()));
    inner.push(Box::new(Zst));
    let mut list = ModuleList::new();
    list.push(Box::new(Affine::new()));
    let mut root = Sequential::new();
    root.push(Box::new(Affine::new()));
    root.push(Box::new(inner));
    root.push(Box::new(list));
    assert_eq!(names(&root), s(&["0", "1", "1.0", "1.1", "2", "2.0"]));
}

#[test]
fn paths_join_with_named_parameters() {
    let mut inner = Sequential::new();
    inner.push(Box::new(Affine::new()));
    let mut root = Sequential::new();
    root.push(Box::new(Affine::new()));
    root.push(Box::new(inner));
    let params: Vec<String> = root
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    for (path, m) in root.named_modules() {
        for (pn, _) in m.named_parameters() {
            let full = format!("{path}.{pn}");
            assert!(params.contains(&full), "{full} not in {params:?}");
        }
    }
}

#[test]
fn module_list_children_names() {
    let mut list = ModuleList::new();
    list.push(Box::new(Affine::new()));
    list.push(Box::new(Zst));
    let c: Vec<String> = list.children().into_iter().map(|(n, _)| n).collect();
    assert_eq!(c, s(&["0", "1"]));
}

#[test]
fn multiple_zst_layers_are_not_dropped() {
    let mut seq = Sequential::new();
    seq.push(Box::new(Affine::new()));
    seq.push(Box::new(Zst));
    seq.push(Box::new(Affine::new()));
    seq.push(Box::new(Zst));
    seq.push(Box::new(Zst));
    seq.push(Box::new(Affine::new()));
    assert_eq!(names(&seq), s(&["0", "1", "2", "3", "4", "5"]));
}

/// 先頭フィールドの子がルートと同一アドレスになる（offset 0）親。
#[repr(C)]
struct Wrapper {
    first: Affine,
    second: Affine,
}

impl Module for Wrapper {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("first".into(), &self.first),
            ("second".into(), &self.second),
        ]
    }
}

#[repr(C)]
struct Outer {
    inner: Wrapper,
}

impl Module for Outer {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("inner".into(), &self.inner)]
    }
}

#[test]
fn offset_zero_first_child_is_not_dropped() {
    let w = Wrapper {
        first: Affine::new(),
        second: Affine::new(),
    };
    assert_eq!(names(&w), s(&["first", "second"]));
    let o = Outer { inner: w };
    assert_eq!(names(&o), s(&["inner", "inner.first", "inner.second"]));
}

struct SelfLoop;

impl Module for SelfLoop {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("me".into(), self)]
    }
}

#[test]
fn self_cycle_terminates_empty() {
    // ZST の自己参照でも祖先スタックで打ち切られる。
    assert!(SelfLoop.named_modules().is_empty());
}

struct Node<'a> {
    pad: u8,
    next: Cell<Option<&'a dyn Module>>,
}

impl Module for Node<'_> {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let _ = self.pad;
        Ok(*input)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        match self.next.get() {
            Some(n) => vec![("b".into(), n)],
            None => Vec::new(),
        }
    }
}

#[test]
fn indirect_cycle_terminates() {
    let a = Node {
        pad: 0,
        next: Cell::new(None),
    };
    let b = Node {
        pad: 1,
        next: Cell::new(None),
    };
    a.next.set(Some(&b));
    b.next.set(Some(&a));
    // a → b → a（a は祖先に既出）。列挙は ["b"] のみ。
    assert_eq!(names(&a), s(&["b"]));
}

struct Shared<'a> {
    child: &'a dyn Module,
}

impl Module for Shared<'_> {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("a".into(), self.child), ("b".into(), self.child)]
    }
}

#[test]
fn shared_child_is_listed_once() {
    let leaf = Affine::new();
    let root = Shared { child: &leaf };
    assert_eq!(names(&root), s(&["a"]));
}

#[test]
fn dyn_dispatch_and_type_name() {
    let mut seq = Sequential::new();
    seq.push(Box::new(Affine::new()));
    let boxed: Box<dyn Module> = Box::new(seq);
    assert_eq!(boxed.parameter_count(), 8);
    assert_eq!(names(boxed.as_ref()), s(&["0"]));
    assert_eq!(boxed.children().len(), 1);
    assert!(boxed.type_name().contains("Sequential"));
    let leaf: Box<dyn Module> = Box::new(Affine::new());
    assert!(leaf.type_name().contains("Affine"));
}
