//! facade 側コンテナ `nn::ModuleList`／`nn::Sequential`（イシュー #2396・親 #2338）。
//!
//! 役割: facade 利用者が [`crate::nn::Module`] を実装した独自層を、facade だけの依存で
//! 「積んで forward し、state_dict を取る」ための PyTorch `nn.ModuleList`／`nn.Sequential`
//! 相当の薄いコンテナである（#2338 承認事項 3。`docs/facade-nn-module-exposure-decision.md`
//! §6・§10・§14）。保持するのは `Box<dyn crate::nn::Module>` で、公開シグネチャに
//! `fandhe_ai_autodiff` の型は出ない（REQ-12）。
//!
//! 意味論は `crates/autodiff/src/nn/container.rs` の同名コンテナと一致させる
//! （`forward` の fail-closed、`"{index}.{name}"` 命名、`set_parameter` の接頭辞振り分けと
//! 未知キーの `Err`、`set_training`／`training` の伝播）。`state_dict`／`load_state_dict` は
//! autodiff 側と同じく override せず、[`crate::nn::Module`] の既定実装（two-pass 検証・
//! キー昇順適用・逆順ロールバック）に任せる。凍結 API（#2400）は `set_requires_grad` の
//! 子への伝播と失敗時ロールバック（`0..=index` を逆順）、`requires_grad` の集約
//! （パラメータを持つ子の `any`。該当なしは `true`）を autodiff 側と一致させる。
//!
//! autodiff 側との差: 子は facade `Module` の不透明な trait object のため、
//! `Sequential::forward` は autodiff 側の Linear→ReLU 融合を行わず子を順に適用するだけである。
//! また `push` した子のモードはコンテナの現在モードへ同期されない（autodiff 側と同じ。
//! 次の `set_training` で伝わる）。`compat::Sequential`（`crate::compat::sequential`）とは
//! 別パスの別型である。

use crate::nn::Module;
use crate::{AutodiffError, TapeRef, Tensor, Var};

/// 子のパラメータ要素数の合計（autodiff `Module::parameter_count` 既定と同じ定義）。
///
/// `requires_grad` 集約が「パラメータを持たない子」を除外する判定用。facade trait には
/// `parameter_count` がまだ無いためモジュール private に置く（#2401 で追加された場合の
/// 置換は #2401 側で判断する）。`named_parameters().is_empty()` は要素数 0 のテンソルの
/// 扱いが autodiff とずれるため使わない。
fn param_numel(m: &dyn Module) -> usize {
    m.named_parameters()
        .into_iter()
        .fold(0usize, |acc, (_, t)| acc.saturating_add(t.numel()))
}

/// PyTorch `nn.ModuleList` 相当。forward を持たない子 `Module` の保持器。
///
/// `forward` は常に `AutodiffError::InvalidArgument` を返す。モードの正は自身のフラグ
/// （既定 `true`）で、`set_training` は全子へ伝播する。
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleList {
    /// 空の `ModuleList` を作る。
    pub fn new() -> Self {
        ModuleList {
            modules: Vec::new(),
            training: true,
        }
    }

    /// 子 `Module` を末尾へ追加する（子のモードは同期しない）。
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }

    /// 保持している子 `Module` の数。
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// 子 `Module` を 1 つも保持していないか。
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// `index` の子への参照（範囲外は `None`。panic しない）。
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    /// [`Self::get`] の可変版。
    pub fn get_mut(&mut self, index: usize) -> Option<&mut (dyn Module + 'static)> {
        self.modules.get_mut(index).map(|m| m.as_mut())
    }

    /// 子の走査（層順）。
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Module>> {
        self.modules.iter()
    }

    /// [`Self::iter`] の可変版。
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Module>> {
        self.modules.iter_mut()
    }
}

impl FromIterator<Box<dyn Module>> for ModuleList {
    fn from_iter<I: IntoIterator<Item = Box<dyn Module>>>(iter: I) -> Self {
        ModuleList {
            modules: iter.into_iter().collect(),
            training: true,
        }
    }
}

impl Module for ModuleList {
    /// `nn.ModuleList` は forward を持たない。誤用を型付きエラーで検出する。
    fn forward<'t>(&self, _tape: TapeRef<'t>, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "ModuleList has no forward (nn.ModuleList is a holder, not a callable module)"
                .to_string(),
        ))
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in &mut self.modules {
            module.set_training(training);
        }
    }

    fn training(&self) -> bool {
        self.training
    }

    /// 全子へ伝播する。失敗したら適用済みの子と失敗した子自身（`0..=index`）を逆順に
    /// 適用前の値へ必ず戻し（`requires_grad()` が一致して見えても `set_requires_grad` を
    /// 呼ぶ）、復元失敗は打ち切らず集約して `Err` に含める。
    ///
    /// 差分: autodiff 側はネストしたコンテナの混在状態を `as_module_list` 経由で再帰的に
    /// 復元するが、facade trait は内部フックを持たない（REQ-12）ため子単位（集約値 1 つ）で
    /// 復元する。混在状態のネストコンテナはエラー経路でのみ均一化されうる
    /// （`docs/facade-nn-module-exposure-decision.md` §17）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        let previous: Vec<bool> = self.modules.iter().map(|m| m.requires_grad()).collect();
        for index in 0..self.modules.len() {
            if let Err(err) = self.modules[index].set_requires_grad(requires_grad) {
                let mut failures: Vec<String> = Vec::new();
                for i in (0..=index).rev() {
                    if let Err(e) = self.modules[i].set_requires_grad(previous[i]) {
                        failures.push(format!("module {i} ({e})"));
                    }
                }
                if !failures.is_empty() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "ModuleList::set_requires_grad: failed to apply to module {index} \
                         ({err}), and rollback also failed for: {}; the ModuleList may now be \
                         left in a partially applied state",
                        failures.join(", ")
                    )));
                }
                return Err(err);
            }
        }
        Ok(())
    }

    /// パラメータを持つ子のどれかが `true` なら `true`。該当する子がいなければ `true`。
    fn requires_grad(&self) -> bool {
        let mut saw = false;
        for m in &self.modules {
            if param_numel(m.as_ref()) == 0 {
                continue;
            }
            saw = true;
            if m.requires_grad() {
                return true;
            }
        }
        !saw
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        for (index, module) in self.modules.iter().enumerate() {
            for (name, tensor) in module.named_parameters() {
                out.push((format!("{index}.{name}"), tensor));
            }
        }
        out
    }

    /// `"{index}.{name}"` を分解して子へ委譲する。区切りなし・index 非数値・範囲外は
    /// すべて `InvalidArgument`（fail-closed。panic しない）。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let (index_str, rest) = name.split_once('.').ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "ModuleList::set_parameter: no parameter named `{name}` (expected `{{index}}.{{name}}`)"
            ))
        })?;
        let index: usize = index_str.parse().map_err(|_| {
            AutodiffError::InvalidArgument(format!(
                "ModuleList::set_parameter: no parameter named `{name}` (`{index_str}` is not a valid module index)"
            ))
        })?;
        let len = self.modules.len();
        match self.modules.get_mut(index) {
            Some(module) => module.set_parameter(rest, value),
            None => Err(AutodiffError::InvalidArgument(format!(
                "ModuleList::set_parameter: no parameter named `{name}` (index {index} out of range; ModuleList has {len} modules)"
            ))),
        }
    }
}

/// PyTorch `nn.Sequential` 相当。子 `Module` を層順に適用する facade 側コンテナ。
///
/// 空の `Sequential` は恒等写像。内部は [`ModuleList`] へ委譲する。Linear→ReLU 融合は
/// 行わない（子が不透明な facade `Module` のため）。
pub struct Sequential {
    inner: ModuleList,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    /// 空の `Sequential` を作る。
    pub fn new() -> Self {
        Sequential {
            inner: ModuleList::new(),
        }
    }

    /// 子 `Module` を末尾へ追加する（子のモードは同期しない）。
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.inner.push(module);
    }

    /// 所有権を消費するビルダー形式の追加（メソッドチェーン用）。
    // `std::ops::Add::add` ではなく PyTorch 風ビルダーの慣用名のため trait 実装にしない。
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.inner.push(Box::new(module));
        self
    }

    /// 保持している層の数。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 層を 1 つも保持していないか。
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// 層列へのスライス参照。
    pub fn layers(&self) -> &[Box<dyn Module>] {
        &self.inner.modules
    }

    /// [`Self::layers`] の可変版。
    pub fn layers_mut(&mut self) -> &mut [Box<dyn Module>] {
        &mut self.inner.modules
    }
}

impl From<ModuleList> for Sequential {
    fn from(inner: ModuleList) -> Self {
        Sequential { inner }
    }
}

impl Module for Sequential {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let mut current = *input;
        for layer in &self.inner.modules {
            current = layer.forward(tape, &current)?;
        }
        Ok(current)
    }

    fn set_training(&mut self, training: bool) {
        self.inner.set_training(training);
    }

    fn training(&self) -> bool {
        self.inner.training()
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.inner.named_parameters()
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.inner.set_parameter(name, value)
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.inner.set_requires_grad(requires_grad)
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_autodiff::nn::Module as AutodiffModule;

    /// テスト専用: autodiff 組み込み層を facade `Module` として積むための包み型
    /// （#2397 の公開・本番アダプタとは別物）。autodiff 側コンテナとの文言一致検証に使う。
    struct BuiltinLayer<L: AutodiffModule>(L);

    impl<L: AutodiffModule> Module for BuiltinLayer<L> {
        fn forward<'t>(
            &self,
            tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            self.0.forward(tape.0, input)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            self.0.named_parameters()
        }
        fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
            self.0.set_parameter(name, value)
        }
    }

    fn linear() -> fandhe_ai_autodiff::nn::Linear {
        fandhe_ai_autodiff::nn::Linear::new(3, 4, true, 7).expect("linear")
    }

    fn pair() -> (ModuleList, fandhe_ai_autodiff::nn::ModuleList) {
        let mut mine = ModuleList::new();
        mine.push(Box::new(BuiltinLayer(linear())));
        let mut theirs = fandhe_ai_autodiff::nn::ModuleList::new();
        theirs.push(Box::new(linear()));
        (mine, theirs)
    }

    #[test]
    fn names_and_errors_match_autodiff_container() {
        let (mut mine, mut theirs) = pair();
        let n1: Vec<String> = mine
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let n2: Vec<String> = theirs
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(n1, n2);

        let t = crate::tape();
        let x = t.var(&Tensor::from_slice(&[0.0f32; 3], &[1, 3]).expect("x"));
        let e1 = mine
            .forward(TapeRef::from(&t), &x)
            .expect_err("Err")
            .to_string();
        let e2 = theirs.forward(&t.0, &x).expect_err("Err").to_string();
        assert_eq!(e1, e2);

        let v = Tensor::from_slice(&[0.0f32], &[1]).expect("v");
        for key in ["weight", "x.weight", "9.weight", "0.nope"] {
            let a = mine
                .set_parameter(key, v.clone())
                .expect_err("Err")
                .to_string();
            let b = theirs
                .set_parameter(key, v.clone())
                .expect_err("Err")
                .to_string();
            assert_eq!(a, b, "key={key}");
        }
    }
    // ---- #2400: 凍結 API の伝播・集約・ロールバック ----

    use std::cell::RefCell;
    use std::rc::Rc;

    type Calls = Rc<RefCell<Vec<bool>>>;

    /// パラメータ持ちで `requires_grad` を保持する子。`fail_apply`／`fail_restore` が真だと
    /// 対応する呼び出しで部分適用状態を残してから `Err` を返す。
    struct Fz {
        p: Tensor<f32>,
        rg: bool,
        fail_apply: bool,
        fail_restore: bool,
    }
    impl Fz {
        fn new() -> Self {
            Fz {
                p: Tensor::from_slice(&[1.0f32, 2.0], &[2]).expect("p"),
                rg: true,
                fail_apply: false,
                fail_restore: false,
            }
        }
    }
    impl Module for Fz {
        fn forward<'t>(
            &self,
            _tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            Ok(*input)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("p".into(), &self.p)]
        }
        fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
            if (!v && self.fail_apply) || (v && self.fail_restore) {
                self.rg = false;
                return Err(AutodiffError::InvalidArgument("child failed".into()));
            }
            self.rg = v;
            Ok(())
        }
        fn requires_grad(&self) -> bool {
            self.rg
        }
    }

    struct Ident;
    impl Module for Ident {
        fn forward<'t>(
            &self,
            _tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            Ok(*input)
        }
    }

    fn list_of(children: Vec<Box<dyn Module>>) -> ModuleList {
        children.into_iter().collect()
    }

    #[test]
    fn freeze_propagates_and_aggregates() {
        let mut l = list_of(vec![
            Box::new(Fz::new()),
            Box::new(Ident),
            Box::new(Fz::new()),
        ]);
        assert!(l.requires_grad());
        l.freeze().expect("freeze");
        assert!(!l.requires_grad(), "無状態の子は集約から除外");
        assert!(!l.get(0).expect("0").requires_grad());
        assert!(!l.get(2).expect("2").requires_grad());

        // any: 1 つでも true なら true。
        l.get_mut(0)
            .expect("0")
            .set_requires_grad(true)
            .expect("ok");
        assert!(l.requires_grad());

        // 空・無状態のみは true。
        assert!(ModuleList::new().requires_grad());
        assert!(list_of(vec![Box::new(Ident)]).requires_grad());
    }

    #[test]
    fn rollback_restores_failed_child_itself() {
        let mut failing = Fz::new();
        failing.fail_apply = true;
        let mut l = list_of(vec![Box::new(Fz::new()), Box::new(failing)]);
        let e = l.freeze().expect_err("2 番目が失敗");
        assert!(e.to_string().contains("child failed"), "{e}");
        assert!(l.get(0).expect("0").requires_grad());
        assert!(l.get(1).expect("1").requires_grad(), "失敗した子自身も戻る");
    }

    #[test]
    fn rollback_calls_setter_even_when_trait_getter_looks_unchanged() {
        // requires_grad() が常に true の子でも、復元時に set_requires_grad が呼ばれる。
        struct Sticky(Calls, Tensor<f32>, bool);
        impl Module for Sticky {
            fn forward<'t>(
                &self,
                _tape: TapeRef<'t>,
                input: &Var<'t>,
            ) -> Result<Var<'t>, AutodiffError> {
                Ok(*input)
            }
            fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
                vec![("p".into(), &self.1)]
            }
            fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
                self.0.borrow_mut().push(v);
                if self.2 && !v {
                    return Err(AutodiffError::InvalidArgument("fail".into()));
                }
                Ok(())
            }
        }
        let c: Calls = Rc::new(RefCell::new(Vec::new()));
        let p = Tensor::from_slice(&[1.0f32], &[1]).expect("p");
        let mut l = list_of(vec![
            Box::new(Sticky(c.clone(), p.clone(), false)),
            Box::new(Sticky(c.clone(), p, true)),
        ]);
        l.freeze().expect_err("fail");
        // 適用: 0(false)・1(false→Err)。復元: 1(true)・0(true)。
        assert_eq!(*c.borrow(), vec![false, false, true, true]);
    }

    #[test]
    fn rollback_failures_are_aggregated_and_continue() {
        let mut a = Fz::new();
        a.fail_restore = true;
        let mut b = Fz::new();
        b.fail_apply = true;
        let mut l = list_of(vec![Box::new(a), Box::new(Fz::new()), Box::new(b)]);
        let e = l.freeze().expect_err("3 番目が失敗").to_string();
        assert!(e.contains("rollback also failed for"), "{e}");
        assert!(e.contains("partially applied state"), "{e}");
        assert!(e.contains("module 0"), "{e}");
        assert!(
            l.get(1).expect("1").requires_grad(),
            "他の子の復元は続行される"
        );
    }

    #[test]
    fn sequential_delegates_and_nested_uniform_rollback() {
        let mut s = Sequential::new();
        s.push(Box::new(Fz::new()));
        s.freeze().expect("freeze");
        assert!(!s.requires_grad());

        // ネスト（均一状態）: 後続の子が失敗すると内側は true へ戻る。
        let mut failing = Fz::new();
        failing.fail_apply = true;
        let inner = list_of(vec![Box::new(Fz::new()), Box::new(Fz::new())]);
        let mut outer = Sequential::new();
        outer.push(Box::new(inner));
        outer.push(Box::new(failing));
        outer.freeze().expect_err("後続が失敗");
        assert!(outer.requires_grad());
        assert!(outer.layers()[0].requires_grad());
    }
}
