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
//! キー昇順適用・逆順ロールバック）に任せる。凍結 API（#2400）は `set_requires_grad` の
//! 子への伝播と失敗時ロールバック（`0..=index` を逆順）、`requires_grad` の集約
//! （パラメータを持つ子の `any`。該当なしは `true`）を autodiff 側と一致させる。
//! ロールバックは子孫を葉（`children` が空の層）単位で復元する（公開の
//! `Module::children`／`children_mut` 経由のため、利用者定義の複合層も対象。PR #2426
//! レビュー指摘・2026-09-29 ユーザー承認）。`children` と `children_mut` が不整合（件数・名前・順序・参照先の不一致）な構成は
//! 状態変更前に `InvalidArgument` で拒否する（fail-closed）。
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

/// `set_requires_grad` ロールバック用の再帰スナップショット（葉単位。イシュー #2400・
/// PR #2426 レビュー指摘）。
///
/// [`Module::children`] が非空の層は子ごと（名前付き）に再帰し、葉（`children` が空）は
/// `requires_grad()` の値 1 つを保持する。集約値 1 つで潰すと、凍結済みと追跡中を併せ持つ
/// 複合層が失敗時に均一化されてしまうため。複合層自身の `requires_grad` は復元対象外
/// （子の葉が正）。`FacadeModuleAdapter` が autodiff の `Module::requires_grad_snapshot` へ
/// `Opaque` として包んで渡す際にも使う（`pub(crate)`）。
pub(crate) enum RequiresGradSnapshot {
    Leaf(bool),
    Nested {
        own: bool,
        children: Vec<(String, RequiresGradSnapshot)>,
    },
}

/// `children` が返す 1 つの子の同一性キー（名前・データアドレス・サイズ）。
/// `children_mut` 側の同位置の子と比べ、同じ子を指すことを確かめるのに使う。
type ChildIdentity = (String, *const (), usize);

/// `module` の凍結状態を葉単位で記録する。状態は変更しない。辿った各ノードで
/// [`Module::children`] と [`Module::children_mut`] の件数・名前・順序・参照先（同一の子）が
/// 食い違う場合は `InvalidArgument`（`children_mut` 未実装の複合層・別の子を返す実装など。
/// 呼び出し側が状態変更前に呼ぶことで fail-closed の事前検査を兼ねる）。
pub(crate) fn snapshot_requires_grad<M: Module + ?Sized>(
    module: &mut M,
) -> Result<RequiresGradSnapshot, AutodiffError> {
    // 共有借用の参照は `children_mut` を呼ぶ前にここで drop し、同一性キーだけ持ち越す。
    let identities: Vec<ChildIdentity> = module
        .children()
        .into_iter()
        .map(|(n, c)| {
            (
                n,
                c as *const dyn Module as *const (),
                std::mem::size_of_val(c),
            )
        })
        .collect();
    let type_name = module.type_name();
    let nested = {
        let children = module.children_mut();
        check_children_consistent(type_name, &identities, &children)?;
        children
            .into_iter()
            .map(|(name, child)| Ok((name, snapshot_requires_grad(child)?)))
            .collect::<Result<Vec<_>, AutodiffError>>()?
    };
    if nested.is_empty() {
        Ok(RequiresGradSnapshot::Leaf(module.requires_grad()))
    } else {
        Ok(RequiresGradSnapshot::Nested {
            own: module.requires_grad(),
            children: nested,
        })
    }
}

/// `children` の同一性キー列と `children_mut` の結果が位置ごとに一致することを検査する
/// （名前に加え、データアドレスと `size_of_val` が同じ = 同一の子を指すこと。不一致は
/// `InvalidArgument`）。vtable 比較は codegen unit をまたぐと値がずれ偽陰性になりうるため
/// 行わない。サイズも比べるのは、ZST が後続フィールドとアドレスを共有しうるため
/// （サイズまで同じ ZST 同士は状態を持たず区別不要）。
fn check_children_consistent(
    type_name: &str,
    expected: &[ChildIdentity],
    children_mut: &[(String, &mut dyn Module)],
) -> Result<(), AutodiffError> {
    let consistent = expected.len() == children_mut.len()
        && expected
            .iter()
            .zip(children_mut)
            .all(|((n, addr, size), (m, c))| {
                n == m
                    && *addr == (&**c as *const dyn Module as *const ())
                    && *size == std::mem::size_of_val(&**c)
            });
    if consistent {
        Ok(())
    } else {
        Err(AutodiffError::InvalidArgument(format!(
            "set_requires_grad: `{type_name}` exposes children {:?} via `children` but {:?} \
             via `children_mut` (or refers to different child objects); implement \
             `children_mut` consistently with `children` (same count, names, order and the \
             identical child references) so that freezing can be rolled back per leaf",
            expected.iter().map(|(n, _, _)| n).collect::<Vec<_>>(),
            children_mut.iter().map(|(n, _)| n).collect::<Vec<_>>()
        )))
    }
}

/// スナップショットへ葉単位で復元する。葉は `requires_grad()` の一致で早期 `Ok` にせず
/// 必ず `set_requires_grad` を呼び（getter が既定 `true` のままの外部実装対策）、`Err` は
/// そのまま返す。入れ子は先に自身へ `set_requires_grad(own)` を呼び、その後に子を復元する
/// （自身の呼び出しが子へ伝播した値は子の復元で上書きされる）。自身・子のどれかが失敗しても
/// 残りの復元を続行し、失敗を集約する。
/// 子の件数・名前が食い違う場合は fail-closed の `InvalidArgument`。
pub(crate) fn restore_requires_grad<M: Module + ?Sized>(
    module: &mut M,
    snapshot: &RequiresGradSnapshot,
) -> Result<(), AutodiffError> {
    match snapshot {
        RequiresGradSnapshot::Leaf(value) => module.set_requires_grad(*value),
        RequiresGradSnapshot::Nested {
            own,
            children: expected,
        } => {
            let mut failures: Vec<String> = Vec::new();
            // 自身 → 子の順。自身の呼び出しは子へ伝播しうるが、直後の子の復元で上書きされる。
            if let Err(e) = module.set_requires_grad(*own) {
                failures.push(format!("self ({e})"));
            }
            let children = module.children_mut();
            if children.len() != expected.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "restore_requires_grad: child count mismatch (snapshot {}, actual {})",
                    expected.len(),
                    children.len()
                )));
            }
            for ((name, child), (expected_name, snap)) in children.into_iter().zip(expected) {
                if &name != expected_name {
                    failures.push(format!(
                        "child `{name}` (snapshot expected `{expected_name}`; structure changed)"
                    ));
                    continue;
                }
                if let Err(e) = restore_requires_grad(child, snap) {
                    failures.push(format!("child `{name}` ({e})"));
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                Err(AutodiffError::InvalidArgument(failures.join(", ")))
            }
        }
    }
}

/// `ModuleList`／`ModuleDict` 共通の `set_requires_grad` 本体（伝播＋失敗時ロールバック）。
///
/// 適用前に各子を [`snapshot_requires_grad`] で葉単位に記録し、失敗したら失敗した子自身を
/// 含む `0..=index` を逆順に葉単位で復元する（部分適用を残しうる外部実装対策）。復元失敗は
/// 打ち切らず集約して部分適用を明示した `Err` を返す。`owner` はエラー文言の型名。
fn set_requires_grad_with_rollback(
    owner: &str,
    mut children: Vec<&mut dyn Module>,
    requires_grad: bool,
) -> Result<(), AutodiffError> {
    // 状態変更前の事前検査を兼ねる: `children`／`children_mut` が不整合な子孫が 1 つでも
    // あれば、ここで `InvalidArgument` を返し何も変更しない。
    let previous: Vec<RequiresGradSnapshot> = children
        .iter_mut()
        .map(|m| snapshot_requires_grad(&mut **m))
        .collect::<Result<_, _>>()?;
    for index in 0..children.len() {
        if let Err(err) = children[index].set_requires_grad(requires_grad) {
            let mut failures: Vec<String> = Vec::new();
            for i in (0..=index).rev() {
                if let Err(e) = restore_requires_grad(&mut *children[i], &previous[i]) {
                    failures.push(format!("module {i} ({e})"));
                }
            }
            if !failures.is_empty() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "{owner}::set_requires_grad: failed to apply to module {index} \
                     ({err}), and rollback also failed for: {}; the {owner} may now be \
                     left in a partially applied state",
                    failures.join(", ")
                )));
            }
            return Err(err);
        }
    }
    Ok(())
}

/// 子の `requires_grad` 集約（パラメータを持つ子の `any`。該当なしは `true`）。
fn aggregate_requires_grad<'a>(children: impl Iterator<Item = &'a dyn Module>) -> bool {
    let mut saw = false;
    for m in children {
        if m.parameter_count() == 0 {
            continue;
        }
        saw = true;
        if m.requires_grad() {
            return true;
        }
    }
    !saw
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

    /// 全子へ伝播する。失敗したら適用済みの子と失敗した子自身（`0..=index`）を逆順に
    /// 適用前の値へ必ず戻し（`requires_grad()` が一致して見えても `set_requires_grad` を
    /// 呼ぶ）、復元失敗は打ち切らず集約して `Err` に含める。子孫は `children`／`children_mut`
    /// 経由で葉単位に復元するため、凍結済みと追跡中が混在する内側の複合層も
    /// 呼び出し前と完全一致に戻る（`children_mut` 不整合な構成は変更前に `InvalidArgument`）（autodiff 側と同じ契約。PR #2426 レビュー指摘）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        let children: Vec<&mut dyn Module> = self
            .modules
            .iter_mut()
            .map(|m| -> &mut dyn Module { m.as_mut() })
            .collect();
        set_requires_grad_with_rollback("ModuleList", children, requires_grad)
    }

    /// パラメータを持つ子のどれかが `true` なら `true`。該当する子がいなければ `true`。
    fn requires_grad(&self) -> bool {
        aggregate_requires_grad(self.modules.iter().map(|m| m.as_ref()))
    }

    /// `children` の可変版（名前・順序は同一。#2400）。
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        self.modules
            .iter_mut()
            .enumerate()
            .map(|(i, m)| -> (String, &mut dyn Module) { (i.to_string(), m.as_mut()) })
            .collect()
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

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.inner.set_requires_grad(requires_grad)
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        self.inner.children_mut()
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

    /// 挿入順に全子へ伝播する。ロールバック・集約は [`ModuleList`] と同一
    /// （`set_requires_grad_with_rollback`。#2402 の申し送りを #2400 で実施）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        let children: Vec<&mut dyn Module> = self
            .modules
            .iter_mut()
            .map(|(_, m)| -> &mut dyn Module { m.as_mut() })
            .collect();
        set_requires_grad_with_rollback("ModuleDict", children, requires_grad)
    }

    /// パラメータを持つ子のどれかが `true` なら `true`。該当する子がいなければ `true`。
    fn requires_grad(&self) -> bool {
        aggregate_requires_grad(self.modules.iter().map(|(_, m)| m.as_ref()))
    }

    /// `children` の可変版（キー・挿入順は同一。#2400）。
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        self.modules
            .iter_mut()
            .map(|(key, module)| -> (String, &mut dyn Module) { (key.clone(), module.as_mut()) })
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

    /// 葉ごとの `requires_grad` を層順に平坦化して返す（入れ子は `children` で辿る）。
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

    fn fz(rg: bool) -> Fz {
        let mut f = Fz::new();
        f.rg = rg;
        f
    }

    fn failing_fz() -> Fz {
        let mut f = Fz::new();
        f.fail_apply = true;
        f
    }

    /// PR #2426 P1: 内側が「凍結済み＋追跡中」の混在でも、外側の後続子が失敗したとき
    /// 各葉が呼び出し前と完全一致する（ModuleList ネスト）。
    #[test]
    fn nested_module_list_mixed_state_restored_per_leaf_on_failure() {
        let inner = list_of(vec![Box::new(fz(false)), Box::new(fz(true))]);
        let mut outer = list_of(vec![Box::new(inner), Box::new(failing_fz())]);
        let before = leaf_states(&outer);
        assert_eq!(before, vec![false, true, true]);
        outer.freeze().expect_err("後続が失敗");
        assert_eq!(leaf_states(&outer), before);
    }

    /// 同上（Sequential ネスト・2 段ネスト）。
    #[test]
    fn nested_sequential_mixed_state_restored_per_leaf_on_failure() {
        let mut deep = Sequential::new();
        deep.push(Box::new(fz(true)));
        deep.push(Box::new(fz(false)));
        let inner = list_of(vec![Box::new(fz(false)), Box::new(deep)]);
        let mut outer = Sequential::new();
        outer.push(Box::new(inner));
        outer.push(Box::new(failing_fz()));
        let before = leaf_states(&outer);
        assert_eq!(before, vec![false, true, false, true]);
        outer.freeze().expect_err("後続が失敗");
        assert_eq!(leaf_states(&outer), before);
    }

    /// 同上（ModuleDict ネスト。外側が ModuleDict・内側が ModuleDict の両方）。
    #[test]
    fn nested_module_dict_mixed_state_restored_per_leaf_on_failure() {
        let mut inner = ModuleDict::new();
        inner.insert("a", Box::new(fz(false))).expect("k");
        inner.insert("b", Box::new(fz(true))).expect("k");
        let mut outer = ModuleDict::new();
        outer.insert("in", Box::new(inner)).expect("k");
        outer.insert("bad", Box::new(failing_fz())).expect("k");
        let before = leaf_states(&outer);
        assert_eq!(before, vec![false, true, true]);
        outer.freeze().expect_err("後続が失敗");
        assert_eq!(leaf_states(&outer), before);
    }

    /// ModuleDict の伝播・集約（#2402 申し送り）と、成功時は入れ子も全葉へ伝播する。
    #[test]
    fn module_dict_freeze_propagates_and_aggregates() {
        let mut d = ModuleDict::new();
        d.insert("a", Box::new(Fz::new())).expect("k");
        d.insert("n", Box::new(Ident)).expect("k");
        d.insert("l", Box::new(list_of(vec![Box::new(fz(true))])))
            .expect("k");
        assert!(d.requires_grad());
        d.freeze().expect("freeze");
        assert_eq!(leaf_states(&d), vec![false, true, false]);
        assert!(!d.requires_grad());
        d.set_requires_grad(true).expect("unfreeze");
        assert!(d.requires_grad());
        assert!(ModuleDict::new().requires_grad());
    }

    /// アダプタが返す `Opaque` スナップショットから facade 側の葉状態を平坦化して読む。
    fn adapter_leaf_states(m: &mut dyn fandhe_ai_autodiff::nn::Module) -> Vec<bool> {
        fn flat(s: &super::RequiresGradSnapshot, out: &mut Vec<bool>) {
            match s {
                super::RequiresGradSnapshot::Leaf(v) => out.push(*v),
                super::RequiresGradSnapshot::Nested { children: c, .. } => {
                    c.iter().for_each(|(_, x)| flat(x, out))
                }
            }
        }
        let snap = m.requires_grad_snapshot().expect("snapshot");
        let fandhe_ai_autodiff::nn::RequiresGradSnapshot::Opaque(any) = snap else {
            panic!("アダプタは Opaque を返すはず");
        };
        let inner = any
            .downcast_ref::<super::RequiresGradSnapshot>()
            .expect("facade スナップショット");
        let mut out = Vec::new();
        flat(inner, &mut out);
        out
    }

    fn adapted(
        m: impl Module + 'static,
    ) -> crate::nn::module::FacadeModuleAdapter<Box<dyn Module>> {
        crate::nn::module::FacadeModuleAdapter(Box::new(m) as Box<dyn Module>)
    }

    fn mixed_facade_container() -> ModuleList {
        let inner = list_of(vec![Box::new(fz(false)), Box::new(fz(true))]);
        list_of(vec![Box::new(fz(false)), Box::new(inner)])
    }

    /// PR #2426 P1（アダプタ境界）: 混在状態の facade コンテナをアダプタで包んで autodiff の
    /// `ModuleList` に積み、後続の子が失敗しても全葉が呼び出し前と完全一致する。
    #[test]
    fn adapter_in_autodiff_module_list_restores_mixed_facade_container_per_leaf() {
        let mut outer = fandhe_ai_autodiff::nn::ModuleList::new();
        outer.push(Box::new(adapted(mixed_facade_container())));
        outer.push(Box::new(adapted(failing_fz())));
        let before = adapter_leaf_states(outer.get_mut(0).expect("0"));
        assert_eq!(before, vec![false, false, true]);
        fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false).expect_err("失敗");
        assert_eq!(adapter_leaf_states(outer.get_mut(0).expect("0")), before);
    }

    /// 同上（autodiff `Sequential`・入れ子 autodiff `ModuleList` 経由）。
    #[test]
    fn adapter_in_autodiff_sequential_restores_mixed_facade_container_per_leaf() {
        let mut inner = fandhe_ai_autodiff::nn::ModuleList::new();
        inner.push(Box::new(adapted(mixed_facade_container())));
        let mut outer = fandhe_ai_autodiff::nn::Sequential::new();
        outer.push(Box::new(inner));
        outer.push(Box::new(adapted(failing_fz())));
        let list = outer.layers_mut()[0]
            .as_module_list_mut()
            .expect("ModuleList");
        let before = adapter_leaf_states(list.get_mut(0).expect("0"));
        fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false).expect_err("失敗");
        let list = outer.layers_mut()[0]
            .as_module_list_mut()
            .expect("ModuleList");
        assert_eq!(adapter_leaf_states(list.get_mut(0).expect("0")), before);
    }

    /// 同上（autodiff `ModuleDict`）。
    #[test]
    fn adapter_in_autodiff_module_dict_restores_mixed_facade_container_per_leaf() {
        let mut outer = fandhe_ai_autodiff::nn::ModuleDict::new();
        outer
            .insert("m", Box::new(adapted(mixed_facade_container())))
            .expect("k");
        outer
            .insert("bad", Box::new(adapted(failing_fz())))
            .expect("k");
        let before = adapter_leaf_states(outer.get_mut("m").expect("m"));
        fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false).expect_err("失敗");
        assert_eq!(adapter_leaf_states(outer.get_mut("m").expect("m")), before);
    }

    /// 自身の `requires_grad` フラグも持つ利用者定義の複合層（PR #2426 第 4 の P1）。
    /// `own` は外部から観測するため共有セルで保持する。
    struct SelfFlagComp {
        own: Rc<std::cell::Cell<bool>>,
        a: Fz,
        b: Fz,
    }
    impl Module for SelfFlagComp {
        fn forward<'t>(
            &self,
            _tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            Ok(*input)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("a.p".into(), &self.a.p), ("b.p".into(), &self.b.p)]
        }
        fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
            self.own.set(v);
            self.a.set_requires_grad(v)?;
            self.b.set_requires_grad(v)
        }
        fn requires_grad(&self) -> bool {
            self.own.get()
        }
        fn children(&self) -> Vec<(String, &dyn Module)> {
            vec![("a".into(), &self.a), ("b".into(), &self.b)]
        }
        fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
            vec![("a".into(), &mut self.a), ("b".into(), &mut self.b)]
        }
    }

    /// PR #2426 第 4 の P1（アダプタ境界）: 自身のフラグを持つ複合層をアダプタで包んで autodiff の
    /// `ModuleList` に積み、後続の子が失敗しても自身のフラグと全葉が呼び出し前へ戻る。
    #[test]
    fn adapter_restores_composite_own_flag_after_later_failure() {
        let own = Rc::new(std::cell::Cell::new(true));
        let mut a = Fz::new();
        a.rg = false;
        let comp = SelfFlagComp {
            own: Rc::clone(&own),
            a,
            b: Fz::new(),
        };
        let mut outer = fandhe_ai_autodiff::nn::ModuleList::new();
        outer.push(Box::new(adapted(comp)));
        outer.push(Box::new(adapted(failing_fz())));
        let before = adapter_leaf_states(outer.get_mut(0).expect("0"));
        assert_eq!(before, vec![false, true]);
        fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false).expect_err("失敗");
        assert!(own.get(), "自身のフラグが呼び出し前へ戻る");
        assert_eq!(adapter_leaf_states(outer.get_mut(0).expect("0")), before);
    }

    /// facade コンテナ経路: 複合層自身のフラグと全葉が失敗後に戻る（子の失敗が複合層内部でも）。
    #[test]
    fn facade_list_restores_composite_own_flag_after_failure() {
        let own = Rc::new(std::cell::Cell::new(true));
        let mut a = Fz::new();
        a.rg = false;
        let mut bad = Fz::new();
        bad.fail_apply = true;
        let comp = SelfFlagComp {
            own: Rc::clone(&own),
            a,
            b: bad,
        };
        let mut l = list_of(vec![Box::new(comp)]);
        l.freeze().expect_err("複合層内部の失敗");
        assert!(own.get(), "自身のフラグが呼び出し前へ戻る");
        assert_eq!(leaf_states_of(&l), vec![false, true]);
    }

    fn leaf_states_of(m: &dyn Module) -> Vec<bool> {
        let c = m.children();
        if c.is_empty() {
            vec![m.requires_grad()]
        } else {
            c.into_iter().flat_map(|(_, x)| leaf_states_of(x)).collect()
        }
    }

    /// 利用者定義の複合層（`children`／`children_mut` 実装あり。PR #2426 第 3 の P1）。
    struct Comp {
        a: Fz,
        b: Fz,
    }
    impl Module for Comp {
        fn forward<'t>(
            &self,
            _tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            Ok(*input)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("a.p".into(), &self.a.p), ("b.p".into(), &self.b.p)]
        }
        fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
            self.a.set_requires_grad(v)?;
            self.b.set_requires_grad(v)
        }
        fn requires_grad(&self) -> bool {
            self.a.requires_grad() || self.b.requires_grad()
        }
        fn children(&self) -> Vec<(String, &dyn Module)> {
            vec![("a".into(), &self.a), ("b".into(), &self.b)]
        }
        fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
            vec![("a".into(), &mut self.a), ("b".into(), &mut self.b)]
        }
    }

    /// `children` だけ実装し `children_mut` を実装しない複合層（拒否対象）。
    struct ChildrenOnly(Comp);
    impl Module for ChildrenOnly {
        fn forward<'t>(
            &self,
            tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
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

    fn mixed_comp() -> Comp {
        Comp {
            a: fz(false),
            b: fz(true),
        }
    }

    /// PR #2426 第 3 の P1: 混在状態の利用者定義複合層を autodiff `ModuleList` へアダプタ経由で
    /// 積み、後続の失敗後も全葉が呼び出し前と完全一致する。
    #[test]
    fn adapter_in_autodiff_module_list_restores_user_composite_per_leaf() {
        let mut outer = fandhe_ai_autodiff::nn::ModuleList::new();
        outer.push(Box::new(adapted(mixed_comp())));
        outer.push(Box::new(adapted(failing_fz())));
        let before = adapter_leaf_states(outer.get_mut(0).expect("0"));
        assert_eq!(before, vec![false, true]);
        fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false).expect_err("失敗");
        assert_eq!(adapter_leaf_states(outer.get_mut(0).expect("0")), before);
    }

    /// `children_mut` 未実装の複合層をアダプタ経由で積むと、状態を変更する前に
    /// `InvalidArgument` で拒否され、全葉が不変（autodiff `ModuleList`）。
    #[test]
    fn adapter_in_autodiff_module_list_rejects_children_only_composite_before_mutation() {
        let mut outer = fandhe_ai_autodiff::nn::ModuleList::new();
        outer.push(Box::new(adapted(fz(true))));
        outer.push(Box::new(adapted(ChildrenOnly(mixed_comp()))));
        let e = fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false)
            .expect_err("事前検査で拒否");
        assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");
        assert!(e.to_string().contains("children_mut"), "{e}");
        assert_eq!(
            adapter_leaf_states(outer.get_mut(0).expect("0")),
            vec![true]
        );
    }

    /// 同名で別の子を `children_mut` から返す複合層をアダプタ経由で積むと、状態変更前に
    /// `InvalidArgument` で拒否され全葉が不変（PR #2426 第 5 回レビュー P1。autodiff `ModuleList`）。
    #[test]
    fn adapter_in_autodiff_module_list_rejects_swapped_children_mut_before_mutation() {
        struct Swapped(Comp);
        impl Module for Swapped {
            fn forward<'t>(
                &self,
                tape: TapeRef<'t>,
                input: &Var<'t>,
            ) -> Result<Var<'t>, AutodiffError> {
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
                vec![("x".into(), &self.0.a)]
            }
            fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
                vec![("x".into(), &mut self.0.b)]
            }
        }
        let mut outer = fandhe_ai_autodiff::nn::ModuleList::new();
        outer.push(Box::new(adapted(fz(true))));
        outer.push(Box::new(adapted(Swapped(mixed_comp()))));
        let e = fandhe_ai_autodiff::nn::Module::set_requires_grad(&mut outer, false)
            .expect_err("事前検査で拒否");
        assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");
        assert!(e.to_string().contains("children_mut"), "{e}");
        assert_eq!(
            adapter_leaf_states(outer.get_mut(0).expect("0")),
            vec![true]
        );
    }

    /// アダプタの復元は `Opaque` 以外・型違いを fail-closed で拒否する。
    #[test]
    fn adapter_restore_rejects_foreign_snapshot() {
        use fandhe_ai_autodiff::nn::{Module as Ad, RequiresGradSnapshot as Snap};
        let mut a = adapted(fz(true));
        assert!(a.restore_requires_grad_snapshot(&Snap::Leaf(true)).is_err());
        let foreign = Snap::Opaque(Box::new(7u8));
        let e = a
            .restore_requires_grad_snapshot(&foreign)
            .expect_err("型違い");
        assert!(e.to_string().contains("not produced by"), "{e}");
        let _ = Ad::requires_grad(&a);
    }

    /// 復元中の入れ子が構造変化（子数不一致）した場合は fail-closed の `Err`。
    #[test]
    fn restore_shape_mismatch_is_fail_closed() {
        let mut l = list_of(vec![Box::new(fz(true))]);
        let snap = RequiresGradSnapshot::Nested {
            own: true,
            children: vec![
                ("0".to_string(), RequiresGradSnapshot::Leaf(true)),
                ("1".to_string(), RequiresGradSnapshot::Leaf(true)),
            ],
        };
        let e = restore_requires_grad(&mut l, &snap).expect_err("不一致");
        assert!(e.to_string().contains("count mismatch"), "{e}");
        let mut leaf = fz(true);
        let e = restore_requires_grad(&mut leaf, &snap).expect_err("葉");
        assert!(e.to_string().contains("count mismatch"), "{e}");
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
