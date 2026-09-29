//! facade 側コンテナ `nn::ModuleList`／`nn::Sequential`（イシュー #2396・親 #2338）と
//! `nn::ModuleDict`／`nn::summary`（イシュー #2402。autodiff #2134 の鏡写し）。
//!
//! 役割: facade 利用者が [`crate::nn::Module`] を実装した独自層を、facade だけの依存で
//! 「積んで forward し、state_dict を取る」ための PyTorch `nn.ModuleList`／`nn.Sequential`
//! 相当の薄いコンテナである（#2338 承認事項 3。`docs/facade-nn-module-exposure-decision.md`
//! §6・§10・§14）。保持するのは `Box<dyn crate::nn::Module>` で、公開シグネチャに
//! `fandhe_ai_autodiff` の型は出ない（REQ-12）。
//!
//! 意味論は `crates/autodiff/src/nn/container.rs` の同名コンテナと一致させる
//! （`forward` の fail-closed、`"{index}.{name}"` 命名、`set_parameter` の接頭辞振り分けと
//! 未知キーの `Err`、`set_training`／`training` の伝播、`children` の `"{index}"` 命名。#2401）。`state_dict`／`load_state_dict` は
//! autodiff 側と同じく override せず、[`crate::nn::Module`] の既定実装（two-pass 検証・
//! キー昇順適用・逆順ロールバック）に任せる。
//!
//! autodiff 側との差: 子は facade `Module` の不透明な trait object のため、
//! `Sequential::forward` は autodiff 側の Linear→ReLU 融合を行わず子を順に適用するだけである。
//! また `push` した子のモードはコンテナの現在モードへ同期されない（autodiff 側と同じ。
//! 次の `set_training` で伝わる）。`compat::Sequential`（`crate::compat::sequential`）とは
//! 別パスの別型である。

use std::collections::HashSet;

use crate::nn::Module;
use crate::nn::module::NodeKey;
use crate::{AutodiffError, TapeRef, Tensor, Var};

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

    /// 子は `"{index}"`（`named_parameters` の接頭辞と一致。#2401）。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .enumerate()
            .map(|(i, m)| (i.to_string(), m.as_ref()))
            .collect()
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

    /// 内側 `ModuleList` へ委譲（`inner` 自体はノードとして出さない。#2401）。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.inner.children()
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
}

/// PyTorch `nn.ModuleDict` 相当。キー付きの子 `Module` 保持器（イシュー #2402。
/// autodiff `ModuleDict`〈#2134〉の鏡写し）。
///
/// 内部は挿入順を保つ `Vec<(String, Box<dyn Module>)>`（列挙順が「登録順」契約のため
/// `HashMap` にしない。検索は O(n) だが層数は高々数十）。`forward` は常に
/// `AutodiffError::InvalidArgument`（保持器のみ）。キーは空文字列と `'.'` 含みを拒否する。
pub struct ModuleDict {
    modules: Vec<(String, Box<dyn Module>)>,
    training: bool,
}

impl Default for ModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

/// `ModuleDict` のキー検証（fail-closed。`.claude/rules/security.md` A03）。空キーは
/// `named_modules` のパス連結で空セグメントを生み、`'.'` 含みは `set_parameter` の
/// `split_once('.')` 解釈を壊すため拒否する。文言は autodiff 側とバイト一致させる。
fn validate_module_dict_key(key: &str) -> Result<(), AutodiffError> {
    if key.is_empty() {
        return Err(AutodiffError::InvalidArgument(
            "ModuleDict: key must not be empty".to_string(),
        ));
    }
    if key.contains('.') {
        return Err(AutodiffError::InvalidArgument(format!(
            "ModuleDict: key `{key}` must not contain `.` (reserved for named_modules/\
             set_parameter path separation)"
        )));
    }
    Ok(())
}

impl ModuleDict {
    /// 空の `ModuleDict` を作る。
    pub fn new() -> Self {
        ModuleDict {
            modules: Vec::new(),
            training: true,
        }
    }

    /// `(key, module)` 列から構築する（挿入順を保持）。不正キーがあれば `Err`（panic しない）。
    pub fn from_pairs(pairs: Vec<(String, Box<dyn Module>)>) -> Result<Self, AutodiffError> {
        let mut dict = ModuleDict::new();
        for (key, module) in pairs {
            dict.insert(key, module)?;
        }
        Ok(dict)
    }

    /// `key` の子を挿入する。同名キーがあれば置換して旧値を返す（位置は保つ）。
    /// 不正キー（空・`'.'` 含み）は `Err`。
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        module: Box<dyn Module>,
    ) -> Result<Option<Box<dyn Module>>, AutodiffError> {
        let key = key.into();
        validate_module_dict_key(&key)?;
        if let Some(slot) = self.modules.iter_mut().find(|(k, _)| *k == key) {
            Ok(Some(std::mem::replace(&mut slot.1, module)))
        } else {
            self.modules.push((key, module));
            Ok(None)
        }
    }

    /// `key` の子を削除して返す（無ければ `None`）。
    pub fn remove(&mut self, key: &str) -> Option<Box<dyn Module>> {
        let index = self.modules.iter().position(|(k, _)| k == key)?;
        Some(self.modules.remove(index).1)
    }

    /// `key` の子への参照（無ければ `None`）。
    pub fn get(&self, key: &str) -> Option<&dyn Module> {
        self.modules
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, m)| m.as_ref())
    }

    /// [`Self::get`] の可変版。
    pub fn get_mut(&mut self, key: &str) -> Option<&mut (dyn Module + 'static)> {
        self.modules
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, m)| m.as_mut())
    }

    /// `key` を保持しているか。
    pub fn contains_key(&self, key: &str) -> bool {
        self.modules.iter().any(|(k, _)| k == key)
    }

    /// キー列の走査（挿入順）。
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.modules.iter().map(|(k, _)| k.as_str())
    }

    /// `(key, &dyn Module)` 列の走査（挿入順）。
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Module)> {
        self.modules.iter().map(|(k, m)| (k.as_str(), m.as_ref()))
    }

    /// [`Self::iter`] の可変版。
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut (dyn Module + 'static))> {
        self.modules
            .iter_mut()
            .map(|(k, m)| (k.as_str(), m.as_mut()))
    }

    /// 保持している子の数。
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// 子を 1 つも保持していないか。
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for ModuleDict {
    fn forward<'t>(&self, _tape: TapeRef<'t>, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "ModuleDict has no forward (nn.ModuleDict is a holder, not a callable module)"
                .to_string(),
        ))
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for (_, module) in &mut self.modules {
            module.set_training(training);
        }
    }

    fn training(&self) -> bool {
        self.training
    }

    /// `"{key}.{name}"` を挿入順で連結する。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        for (key, module) in &self.modules {
            for (name, tensor) in module.named_parameters() {
                out.push((format!("{key}.{name}"), tensor));
            }
        }
        out
    }

    /// 先頭の `'.'` までをキーとして子へ委譲する。区切りなし・未知キーは `InvalidArgument`。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let (key, rest) = name.split_once('.').ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "ModuleDict::set_parameter: no parameter named `{name}` (expected `{{key}}.{{name}}`)"
            ))
        })?;
        match self.get_mut(key) {
            Some(module) => module.set_parameter(rest, value),
            None => Err(AutodiffError::InvalidArgument(format!(
                "ModuleDict::set_parameter: no parameter named `{name}` (no child module with key `{key}`)"
            ))),
        }
    }

    /// 名前はキー、順序は挿入順。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .map(|(key, module)| (key.clone(), module.as_ref()))
            .collect()
    }
}

/// `type_name` の表示用短縮（`<` の手前で切り、最後の `::` 以降を取る）。
fn short_type_name(full: &str) -> &str {
    let without_generics = full.split('<').next().unwrap_or(full);
    without_generics
        .rsplit("::")
        .next()
        .unwrap_or(without_generics)
}

/// `module` の構造を人間可読な文字列へ整形する（PyTorch `print(model)` 相当。
/// autodiff `summary`〈#2134〉の鏡写し。イシュー #2402）。
///
/// 子を持つノードは `"{Type}(\n"` の後に子を 2 スペースずつインデントして並べ
/// `"{indent}) [params: N]\n"` で閉じる。葉は `"{indent}({name}): {Type} [params: N]\n"`
/// （ルートが葉なら `"{Type} [params: N]\n"`）。末尾に `"Submodules: {named_modules 数}\n"`・
/// `"Total parameters: {parameter_count}\n"` を付ける。shape 推定・`extra_repr` は対象外。
///
/// 循環は祖先スタックで打ち切り、ゼロサイズ型を除く共有子は `(データポインタ, type_name)`
/// で dedup する（`named_modules` と同じ同一性キー）。型名は `std::any::type_name` 由来で
/// 安定が保証されないため表示用途に限る。
pub fn summary(module: &dyn Module) -> String {
    let mut out = String::new();
    let mut ancestors: Vec<NodeKey> =
        vec![(module as *const dyn Module as *const (), module.type_name())];
    let mut visited: HashSet<NodeKey> = HashSet::new();
    write_module(&mut out, None, module, 0, &mut ancestors, &mut visited);
    out.push_str(&format!("Submodules: {}\n", module.named_modules().len()));
    out.push_str(&format!("Total parameters: {}\n", module.parameter_count()));
    out
}

/// [`summary`] の再帰本体（`name` は子としての名前。ルートは `None`）。
fn write_module(
    out: &mut String,
    name: Option<&str>,
    module: &dyn Module,
    depth: usize,
    ancestors: &mut Vec<NodeKey>,
    visited: &mut HashSet<NodeKey>,
) {
    let indent = "  ".repeat(depth);
    let type_name = short_type_name(module.type_name());
    let children = module.children();

    if children.is_empty() {
        match name {
            Some(name) => out.push_str(&format!(
                "{indent}({name}): {type_name} [params: {}]\n",
                module.parameter_count()
            )),
            None => out.push_str(&format!(
                "{type_name} [params: {}]\n",
                module.parameter_count()
            )),
        }
        return;
    }

    match name {
        Some(name) => out.push_str(&format!("{indent}({name}): {type_name}(\n")),
        None => out.push_str(&format!("{type_name}(\n")),
    }
    for (child_name, child) in &children {
        let key: NodeKey = (*child as *const dyn Module as *const (), child.type_name());
        if ancestors.contains(&key) {
            continue;
        }
        let is_zst = std::mem::size_of_val(*child) == 0;
        if !is_zst && !visited.insert(key) {
            continue;
        }
        ancestors.push(key);
        write_module(out, Some(child_name), *child, depth + 1, ancestors, visited);
        ancestors.pop();
    }
    out.push_str(&format!(
        "{indent}) [params: {}]\n",
        module.parameter_count()
    ));
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
        fn type_name(&self) -> &'static str {
            self.0.type_name()
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

    fn relu_box() -> Box<dyn Module> {
        Box::new(BuiltinLayer(fandhe_ai_autodiff::nn::activation::Relu))
    }

    fn dict_pair() -> (ModuleDict, fandhe_ai_autodiff::nn::ModuleDict) {
        let mut mine = ModuleDict::new();
        mine.insert("enc", Box::new(BuiltinLayer(linear())))
            .expect("k");
        mine.insert("act", relu_box()).expect("k");
        let mut theirs = fandhe_ai_autodiff::nn::ModuleDict::new();
        theirs.insert("enc", Box::new(linear())).expect("k");
        theirs
            .insert("act", Box::new(fandhe_ai_autodiff::nn::activation::Relu))
            .expect("k");
        (mine, theirs)
    }

    #[test]
    fn module_dict_matches_autodiff_module_dict() {
        let (mut mine, mut theirs) = dict_pair();
        let names = |v: Vec<(String, &Tensor<f32>)>| -> Vec<String> {
            v.into_iter().map(|(n, _)| n).collect()
        };
        assert_eq!(
            names(mine.named_parameters()),
            names(theirs.named_parameters())
        );

        let t = crate::tape();
        let x = t.var(&Tensor::from_slice(&[0.0f32; 3], &[1, 3]).expect("x"));
        let e1 = mine
            .forward(TapeRef::from(&t), &x)
            .expect_err("Err")
            .to_string();
        let e2 = theirs.forward(&t.0, &x).expect_err("Err").to_string();
        assert_eq!(e1, e2);

        let v = Tensor::from_slice(&[0.0f32], &[1]).expect("v");
        for key in ["weight", "x.weight", "enc.nope"] {
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
        for key in ["", "a.b"] {
            let a = mine.insert(key, relu_box()).err().expect("Err").to_string();
            let b = theirs
                .insert(key, Box::new(fandhe_ai_autodiff::nn::activation::Relu))
                .err()
                .expect("Err")
                .to_string();
            assert_eq!(a, b, "key={key:?}");
        }
    }

    #[test]
    fn summary_matches_autodiff_summary() {
        let (mine, theirs) = dict_pair();
        assert_eq!(summary(&mine), fandhe_ai_autodiff::nn::summary(&theirs));

        let mut seq = Sequential::new();
        seq.push(Box::new(BuiltinLayer(linear())));
        seq.push(relu_box());
        seq.push(Box::new(mine));
        let mut aseq = fandhe_ai_autodiff::nn::Sequential::new();
        aseq.push(Box::new(linear()));
        aseq.push(Box::new(fandhe_ai_autodiff::nn::activation::Relu));
        aseq.push(Box::new(theirs));
        assert_eq!(summary(&seq), fandhe_ai_autodiff::nn::summary(&aseq));
    }
}
