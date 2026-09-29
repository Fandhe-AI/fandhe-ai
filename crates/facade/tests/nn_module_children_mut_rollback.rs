//! facade `nn::Module::children_mut`（#2400・PR #2426 レビュー是正。2026-09-29 ユーザー承認の
//! 公開面追加）による凍結ロールバックの統合テスト。
//! 利用者定義の複合層（`children`／`children_mut` 実装あり）を `ModuleList`／`Sequential`／
//! `ModuleDict` に積み、後続子の `set_requires_grad` 失敗後に全葉が呼び出し前と完全一致すること、
//! `children_mut` 未実装・不整合の複合層は状態変更前に `InvalidArgument` で拒否されることを
//! 検証する。facade の公開面だけに依存する（外部利用者視点。`fandhe_ai_autodiff` は import しない）。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fandhe_ai::nn::{Module, ModuleDict, ModuleList, Sequential};
use fandhe_ai::{AutodiffError, TapeRef, Tensor, Var};

/// パラメータ持ちの葉。`fail_apply` が真だと `set_requires_grad(false)` で部分適用
/// （`requires_grad` を `false` にした状態）を残して `Err` を返す。`calls` は呼び出し回数。
struct Leaf {
    p: Tensor<f32>,
    rg: bool,
    fail_apply: bool,
    calls: Arc<AtomicUsize>,
}

impl Leaf {
    fn new(rg: bool, calls: &Arc<AtomicUsize>) -> Self {
        Self {
            p: Tensor::from_slice(&[1.0f32, 2.0], &[2]).expect("p"),
            rg,
            fail_apply: false,
            calls: Arc::clone(calls),
        }
    }
    fn failing(calls: &Arc<AtomicUsize>) -> Self {
        Self {
            fail_apply: true,
            ..Self::new(true, calls)
        }
    }
}

impl Module for Leaf {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("p".into(), &self.p)]
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !v && self.fail_apply {
            self.rg = false;
            return Err(AutodiffError::InvalidArgument("leaf failed".into()));
        }
        self.rg = v;
        Ok(())
    }
    fn requires_grad(&self) -> bool {
        self.rg
    }
}

/// 利用者定義の複合層。子 `a`・`b` を `children`／`children_mut` の両方で公開する。
struct Composite {
    a: Box<dyn Module>,
    b: Box<dyn Module>,
}

impl Module for Composite {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        for (prefix, child) in [("a", &self.a), ("b", &self.b)] {
            for (n, t) in child.named_parameters() {
                out.push((format!("{prefix}.{n}"), t));
            }
        }
        out
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.a.set_requires_grad(v)?;
        self.b.set_requires_grad(v)
    }
    fn requires_grad(&self) -> bool {
        self.a.requires_grad() || self.b.requires_grad()
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("a".into(), self.a.as_ref()), ("b".into(), self.b.as_ref())]
    }
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        vec![("a".into(), self.a.as_mut()), ("b".into(), self.b.as_mut())]
    }
}

/// `children` だけ実装し `children_mut` を実装しない複合層（拒否対象）。
struct ChildrenOnly(Composite);

impl Module for ChildrenOnly {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.0.forward(tape, input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.0.named_parameters()
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.0.set_requires_grad(v)
    }
    fn requires_grad(&self) -> bool {
        self.0.requires_grad()
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.0.children()
    }
}

/// `children_mut` の名前が `children` と食い違う複合層（拒否対象）。
struct RenamedMut(Composite);

impl Module for RenamedMut {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.0.forward(tape, input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.0.named_parameters()
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.0.set_requires_grad(v)
    }
    fn requires_grad(&self) -> bool {
        self.0.requires_grad()
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.0.children()
    }
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        self.0
            .children_mut()
            .into_iter()
            .map(|(n, m)| (format!("{n}_renamed"), m))
            .collect()
    }
}

/// `children` は子 `a`、`children_mut` は同名で別フィールドの子 `b` を返す複合層（拒否対象。
/// PR #2426 第 5 回レビュー P1）。`set_requires_grad` は両方へ伝播する。
struct SwappedMut(Composite);

impl Module for SwappedMut {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.0.forward(tape, input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.0.named_parameters()
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.0.set_requires_grad(v)
    }
    fn requires_grad(&self) -> bool {
        self.0.requires_grad()
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("x".into(), self.0.a.as_ref())]
    }
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        vec![("x".into(), self.0.b.as_mut())]
    }
}

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

/// 葉（`children` が空の層）ごとの `requires_grad` を層順に平坦化する。
fn leaf_states(m: &dyn Module) -> Vec<bool> {
    let children = m.children();
    if children.is_empty() {
        vec![m.requires_grad()]
    } else {
        children
            .into_iter()
            .flat_map(|(_, c)| leaf_states(c))
            .collect()
    }
}

/// 混在状態（凍結済み `a`・追跡中 `b`）の複合層。
fn mixed(calls: &Arc<AtomicUsize>) -> Composite {
    Composite {
        a: Box::new(Leaf::new(false, calls)),
        b: Box::new(Leaf::new(true, calls)),
    }
}

/// 2 段ネスト（複合層の中に複合層）。葉は `[false, true, true, false]`。
fn nested(calls: &Arc<AtomicUsize>) -> Composite {
    Composite {
        a: Box::new(mixed(calls)),
        b: Box::new(Composite {
            a: Box::new(Leaf::new(true, calls)),
            b: Box::new(Leaf::new(false, calls)),
        }),
    }
}

#[test]
fn module_list_restores_user_composite_per_leaf_after_later_failure() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(mixed(&calls)));
    list.push(Box::new(Leaf::failing(&calls)));
    let before = leaf_states(&list);
    assert_eq!(before, vec![false, true, true]);
    let e = list.freeze().expect_err("後続子が失敗");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)));
    assert_eq!(leaf_states(&list), before, "全葉が呼び出し前と完全一致");
}

#[test]
fn sequential_restores_two_level_nested_user_composite_per_leaf() {
    let calls = counter();
    let mut seq = Sequential::new()
        .add(Leaf::new(false, &calls))
        .add(nested(&calls))
        .add(Leaf::failing(&calls));
    let before = leaf_states(&seq);
    assert_eq!(before, vec![false, false, true, true, false, true]);
    seq.freeze().expect_err("後続子が失敗");
    assert_eq!(leaf_states(&seq), before);
}

#[test]
fn module_dict_restores_user_composite_per_leaf_after_later_failure() {
    let calls = counter();
    let mut dict = ModuleDict::new();
    dict.insert("c", Box::new(mixed(&calls))).expect("key");
    dict.insert("bad", Box::new(Leaf::failing(&calls)))
        .expect("key");
    let before = leaf_states(&dict);
    dict.freeze().expect_err("後続子が失敗");
    assert_eq!(leaf_states(&dict), before);
}

#[test]
fn composite_failing_inside_restores_its_own_partial_leaves() {
    // 複合層自身の内部で `b` が失敗し `a` が先に適用済みでも、全葉が戻る。
    let calls = counter();
    let c = Composite {
        a: Box::new(Leaf::new(true, &calls)),
        b: Box::new(Leaf::failing(&calls)),
    };
    let mut list = ModuleList::new();
    list.push(Box::new(Leaf::new(false, &calls)));
    list.push(Box::new(c));
    let before = leaf_states(&list);
    list.freeze().expect_err("複合層内部の失敗");
    assert_eq!(leaf_states(&list), before);
}

#[test]
fn success_propagates_to_every_leaf_and_aggregates() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(nested(&calls)));
    list.push(Box::new(Leaf::new(true, &calls)));
    list.freeze().expect("freeze");
    assert!(leaf_states(&list).iter().all(|v| !v));
    assert!(!list.requires_grad());
    list.set_requires_grad(true).expect("unfreeze");
    assert!(leaf_states(&list).iter().all(|v| *v));
    assert!(list.requires_grad());
}

/// `children` だけ実装した複合層は、状態を 1 つも変更する前に `InvalidArgument` で拒否される
/// （先行する子の `set_requires_grad` も呼ばれない）。
#[test]
fn children_only_composite_is_rejected_before_any_mutation() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(Leaf::new(true, &calls)));
    list.push(Box::new(ChildrenOnly(mixed(&calls))));
    let before = leaf_states(&list);
    let e = list.freeze().expect_err("事前検査で拒否");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");
    assert!(e.to_string().contains("children_mut"), "{e}");
    assert_eq!(leaf_states(&list), before);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "どの葉にも触れない");

    // Sequential・ModuleDict・2 段ネスト（拒否対象が奥）でも同じ。
    let mut seq = Sequential::new()
        .add(Leaf::new(true, &calls))
        .add(Composite {
            a: Box::new(Leaf::new(true, &calls)),
            b: Box::new(ChildrenOnly(mixed(&calls))),
        });
    assert!(matches!(
        seq.freeze(),
        Err(AutodiffError::InvalidArgument(_))
    ));
    let mut dict = ModuleDict::new();
    dict.insert("x", Box::new(ChildrenOnly(mixed(&calls))))
        .expect("key");
    assert!(matches!(
        dict.freeze(),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// `children_mut` の名前が `children` と食い違う複合層も状態変更前に拒否される。
#[test]
fn inconsistent_children_mut_names_are_rejected_before_any_mutation() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(Leaf::new(true, &calls)));
    list.push(Box::new(RenamedMut(mixed(&calls))));
    let before = leaf_states(&list);
    let e = list.freeze().expect_err("名前不一致");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");
    assert_eq!(leaf_states(&list), before);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// 同名で別の子を `children_mut` から返す複合層は、状態変更前に `InvalidArgument` で拒否され、
/// a・b ともに状態が変わらない（`ModuleList`／`Sequential`／`ModuleDict` の全経路）。
#[test]
fn different_child_object_with_same_name_is_rejected_before_any_mutation() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(Leaf::new(true, &calls)));
    list.push(Box::new(SwappedMut(mixed(&calls))));
    let e = list.freeze().expect_err("参照先不一致");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let mut seq = Sequential::new();
    seq.push(Box::new(SwappedMut(mixed(&calls))));
    let e = seq.set_requires_grad(false).expect_err("参照先不一致");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");

    let mut dict = ModuleDict::new();
    dict.insert("m", Box::new(SwappedMut(mixed(&calls))))
        .expect("insert");
    let e = dict.set_requires_grad(false).expect_err("参照先不一致");
    assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");

    // 全経路で `set_requires_grad` が 1 度も葉へ届いていない（a・b とも状態不変）。
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// 組込みコンテナ自身の `children_mut` は `children` と同じ名前・順序を返す。
#[test]
fn builtin_containers_children_mut_matches_children() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(Leaf::new(true, &calls)));
    list.push(Box::new(Leaf::new(true, &calls)));
    let a: Vec<String> = list.children().into_iter().map(|(n, _)| n).collect();
    let b: Vec<String> = list.children_mut().into_iter().map(|(n, _)| n).collect();
    assert_eq!(a, b);
    let mut seq = Sequential::new().add(Leaf::new(true, &calls));
    assert_eq!(seq.children().len(), seq.children_mut().len());
    let mut dict = ModuleDict::new();
    dict.insert("k", Box::new(Leaf::new(true, &calls)))
        .expect("key");
    let dn: Vec<String> = dict.children().into_iter().map(|(n, _)| n).collect();
    let dm: Vec<String> = dict.children_mut().into_iter().map(|(n, _)| n).collect();
    assert_eq!(dn, dm);
}

/// 自身の `requires_grad` フラグを持ち、`forward` でそのフラグにより自身のパラメータ葉の
/// 登録（`tape.var`／`tape.var_no_grad`）を切り替える利用者定義複合層（PR #2426 第 4 の P1）。
/// 子 `a`・`b` は `children`／`children_mut` で公開する。`set_requires_grad` は自身のフラグを
/// 更新してから子へ伝播するため、後続子の失敗時に自身だけ更新済みの状態が残りうる。
struct SelfFlagComposite {
    p: Tensor<f32>,
    own: bool,
    a: Box<dyn Module>,
    b: Box<dyn Module>,
}

impl SelfFlagComposite {
    fn new(own: bool, a: Box<dyn Module>, b: Box<dyn Module>) -> Self {
        Self {
            p: Tensor::from_slice(&[1.0f32, 2.0], &[2]).expect("p"),
            own,
            a,
            b,
        }
    }
}

impl Module for SelfFlagComposite {
    fn forward<'t>(&self, tape: TapeRef<'t>, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(if self.own {
            tape.var(&self.p)
        } else {
            tape.var_no_grad(&self.p)
        })
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = vec![("p".to_string(), &self.p)];
        for (prefix, child) in [("a", &self.a), ("b", &self.b)] {
            for (n, t) in child.named_parameters() {
                out.push((format!("{prefix}.{n}"), t));
            }
        }
        out
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.own = v;
        self.a.set_requires_grad(v)?;
        self.b.set_requires_grad(v)
    }
    fn requires_grad(&self) -> bool {
        self.own
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("a".into(), self.a.as_ref()), ("b".into(), self.b.as_ref())]
    }
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        vec![("a".into(), self.a.as_mut()), ("b".into(), self.b.as_mut())]
    }
}

/// `m.forward` が登録する葉が勾配追跡されているか（`Gradients::get` が `Ok(Some(_))` か）。
fn forward_leaf_is_tracked(m: &dyn Module) -> bool {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::from_slice(&[0.0f32, 0.0], &[2]).expect("x"));
    let out = m.forward(TapeRef::from(&tape), &x).expect("forward");
    let loss = out.sum(None).expect("sum");
    let grads = tape.backward(&loss).expect("backward");
    matches!(grads.get(&out), Ok(Some(_)))
}

/// 自身のフラグ `own`・混在子（凍結済み `a`・追跡中 `b`）の複合層を作る。
fn self_flag(calls: &Arc<AtomicUsize>) -> SelfFlagComposite {
    SelfFlagComposite::new(
        true,
        Box::new(Leaf::new(false, calls)),
        Box::new(Leaf::new(true, calls)),
    )
}

/// `ModuleList` に自身のフラグを持つ複合層と失敗する後続子を積み、`freeze` 失敗後に (1) 自身の
/// `requires_grad()`、(2) 全葉の状態、(3) forward で登録される葉の追跡有無が呼び出し前と一致する。
#[test]
fn module_list_restores_composite_own_flag_leaves_and_forward_tracking() {
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(self_flag(&calls)));
    list.push(Box::new(Leaf::failing(&calls)));
    let own_before = list.get(0).expect("0").requires_grad();
    let leaves_before = leaf_states(&list);
    let tracked_before = forward_leaf_is_tracked(list.get(0).expect("0"));
    assert!(own_before && tracked_before);
    assert_eq!(leaves_before, vec![false, true, true]);

    list.freeze().expect_err("後続子が失敗");

    assert_eq!(list.get(0).expect("0").requires_grad(), own_before);
    assert_eq!(leaf_states(&list), leaves_before);
    assert_eq!(
        forward_leaf_is_tracked(list.get(0).expect("0")),
        tracked_before,
        "forward が登録する葉の追跡が止まらない"
    );
}

/// 自身のフラグを持つ複合層を `Sequential`・`ModuleDict` に積んでも同様に戻る。また複合層内部の
/// 子が失敗した場合（自身は更新済み・後続子は未適用）も戻る。
#[test]
fn sequential_and_dict_restore_composite_own_flag_after_failure() {
    let calls = counter();
    let mut seq = Sequential::new()
        .add(self_flag(&calls))
        .add(Leaf::failing(&calls));
    seq.freeze().expect_err("後続子が失敗");
    let first = &seq.children()[0].1;
    assert!(first.requires_grad());
    assert!(forward_leaf_is_tracked(*first));

    let mut dict = ModuleDict::new();
    dict.insert(
        "c",
        Box::new(SelfFlagComposite::new(
            true,
            Box::new(Leaf::new(false, &calls)),
            Box::new(Leaf::failing(&calls)),
        )),
    )
    .expect("key");
    let before = leaf_states(&dict);
    dict.freeze().expect_err("複合層内部の失敗");
    let c = &dict.children()[0].1;
    assert!(c.requires_grad(), "自身のフラグが戻る");
    assert!(forward_leaf_is_tracked(*c));
    assert_eq!(leaf_states(&dict), before);
}

/// 状態を外部から観測できる葉（`rg` を共有する）。
struct SharedLeaf {
    p: Tensor<f32>,
    rg: Arc<AtomicBool>,
}

impl SharedLeaf {
    fn new(rg: &Arc<AtomicBool>) -> Self {
        Self {
            p: Tensor::from_slice(&[1.0f32, 2.0], &[2]).expect("p"),
            rg: Arc::clone(rg),
        }
    }
}

impl Module for SharedLeaf {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("p".into(), &self.p)]
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.rg.store(v, Ordering::SeqCst);
        Ok(())
    }
    fn requires_grad(&self) -> bool {
        self.rg.load(Ordering::SeqCst)
    }
}

/// `children`／`children_mut` は通常子 `a` を返すが、自身の `set_requires_grad` が呼ばれると
/// 以後は同名の別フィールド `b` を返すよう切り替わる利用者定義の複合層（PR #2426 第 6 回
/// レビュー P1）。`false` の伝播は `b`→`a` の順で両方へ、`true` は自身では子へ伝播しない
/// （復元時の自身への呼び出しが子の状態を書き換えず、別の子へ適用されたかを判別できるようにする）。
struct SwitchingComposite {
    a: Box<dyn Module>,
    b: Box<dyn Module>,
    switched: bool,
}

impl Module for SwitchingComposite {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.a.named_parameters()
    }
    fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
        self.switched = true;
        if !v {
            self.b.set_requires_grad(false)?;
            self.a.set_requires_grad(false)?;
        }
        Ok(())
    }
    fn requires_grad(&self) -> bool {
        self.a.requires_grad() || self.b.requires_grad()
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        let c = if self.switched { &self.b } else { &self.a };
        vec![("x".into(), c.as_ref())]
    }
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        let c = if self.switched {
            &mut self.b
        } else {
            &mut self.a
        };
        vec![("x".into(), c.as_mut())]
    }
}

/// 凍結処理中に `children_mut` の参照先が別の子へ切り替わっても、ロールバックは偽の `Ok` や
/// 元エラーだけを返さず、部分適用を示す `Err` を返し、別の子 `b` へスナップショットの値
/// （`a` の取得時の値 `true`）を適用しない。
#[test]
fn rollback_detects_child_swapped_during_freeze() {
    let a_rg = Arc::new(AtomicBool::new(true));
    let b_rg = Arc::new(AtomicBool::new(false));
    let calls = counter();
    let mut list = ModuleList::new();
    list.push(Box::new(SwitchingComposite {
        a: Box::new(SharedLeaf::new(&a_rg)),
        b: Box::new(SharedLeaf::new(&b_rg)),
        switched: false,
    }));
    list.push(Box::new(Leaf::failing(&calls)));

    let e = list
        .freeze()
        .expect_err("後続子が失敗しロールバックも不完全");
    let msg = e.to_string();
    assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{msg}");
    assert!(msg.contains("partially applied"), "部分適用の明示: {msg}");
    assert!(msg.contains("leaf failed"), "元エラーも含む: {msg}");
    assert!(
        !b_rg.load(Ordering::SeqCst),
        "別の子 b へスナップショットの値を適用しない"
    );
}
