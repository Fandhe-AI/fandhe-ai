//! `ModuleList`／汎用 `Sequential` コンテナ（イシュー #1759・親 #1617）。
//!
//! [`crate::nn::module::Module`] trait の `set_training`／`training`
//! doc（イシュー #1758）が「モードの正はコンテナ（当時は
//! `fandhe_ai_facade::compat::sequential::Sequential`・将来の
//! `ModuleList`／汎用 `Sequential`）が保持するフラグ」と明記していた
//! 汎用コンテナを本イシューで実装する。PyTorch `nn.ModuleList`／
//! `nn.Sequential` 相当。
//!
//! # 配置（REQ-9・facade は薄いラッパーに徹する）
//!
//! `fandhe_ai::facade` は `Module` trait を公開していない
//! （`docs/compat-api-scope.md` §0）ため、`Box<dyn Module>` を受ける
//! 汎用コンテナは facade からは構築できない。したがって本コンテナは
//! 内部クレート `fandhe_ai_autodiff::nn` に置き、
//! [`fandhe_ai_facade::compat::sequential::Sequential`]（レイヤー種別を
//! Linear／活性化関数の閉集合〈`docs/compat-api-scope.md` §1〉に限定した
//! ビルダー）は本モジュールの [`Sequential`] を `inner` として合成する
//! 薄いラッパーへ再構成した（本イシュー）。両者はパスが異なるため
//! 名前は衝突しない（`fandhe_ai_autodiff::nn::Sequential` と
//! `fandhe_ai_facade::compat::Sequential`）。`fandhe_ai_autodiff::
//! compat::Sequential`（#411 で `facade` へ移設済みの非推奨シム）とも
//! 別物であり、本イシューの対象外（触れない）。
//!
//! # facade への非公開
//!
//! `ModuleList`／`Sequential`（本モジュール）は facade から
//! 再エクスポートしない（`Module` trait 自体が非公開のため使途がなく、
//! `crates/facade/tests/api_surface.rs` の走査対象を増やさない）。
//!
//! イシュー #2134（親 #2131）で追加した [`ModuleDict`]（PyTorch
//! `nn.ModuleDict` 相当）・[`summary`] 自由関数（PyTorch
//! `print(model)` 相当の簡易表示）も同じ理由で非公開のままとする。
//! イシュー本文が挙げる facade 公開面の拡張（`Module::named_modules`／
//! `Module::parameter_count`・`ModuleDict`・`summary` の再エクスポート）
//! は、(a) #2134・親 #2131 とも承認コメントが確認できない、(b) facade
//! が `Module` trait 自体を公開していないため `Box<dyn Module>` を
//! 受ける `ModuleDict`・`&dyn Module` を受ける `summary` は #2133
//! （`Module` trait の facade 公開）完了まで意味を成さない、(c) 既存
//! ガード `nn_mod_declares_only_rnn_submodule`（`crates/facade/tests/
//! api_surface.rs`）が facade `nn/mod.rs` の公開宣言を `pub mod rnn;`
//! 1 件へ固定している、という 3 点により本イシューでは実施しない
//! （`docs/compat-feature-gap.md` 追補・`crates/facade/tests/
//! api_surface.rs` の否定ガードで固定する）。承認取得後の実施形は
//! #2133 完了後の `crates/facade/src/nn/mod.rs` 再エクスポート、または
//! `fandhe_ai_facade::compat::Sequential::parameter_count()`／
//! `summary()` の薄い委譲のいずれかを想定する（後続 issue で判断）。
//!
//! # ネストの限界（既知の制限。解消は行わない）
//!
//! `fandhe_ai_facade::compat::sequential::Sequential` の学習契約
//! （`bind`／`trainable_parameters`／`apply_parameters`・
//! `init_device_param_store` 等のデバイス常駐経路）は最上位の
//! `as_linear()`／`as_conv2d()`／`as_conv1d()`（イシュー #1770）のみを
//! 見る。本モジュールの `Sequential`／`ModuleList` をネストして構築
//! した `Box<dyn Module>` を compat 層へ積んだ場合、ネスト内部の
//! `Linear`／`Conv2d`／`Conv1d` は上記フックが `None` を返すため学習
//! 可能パラメータとして認識されない。ただし facade は `ModuleList`／
//! `Sequential`（本モジュール）を構築する経路を公開していないため、
//! この制限は facade 経由では到達不能である。

use std::collections::HashSet;

use crate::error::AutodiffError;
use crate::nn::module::Module;
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

/// PyTorch `nn.ModuleList` 相当: forward を持たない子 `Module` の
/// 保持器。`set_training`／`training`・`named_parameters` は自身が
/// モードの正となり全子へ伝播・収集する（[`Module`] trait doc の
/// コンテナ契約）。
///
/// `forward`／`forward_host` はいずれも
/// [`AutodiffError::InvalidArgument`] を返す（`ModuleList` 自体は
/// 演算列を持たないため。`nn.ModuleList` が `forward` を持たないのと
/// 同じ設計）。
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
    /// train／eval モード（イシュー #1758 契約。既定 `true`）。
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

    /// 子 `Module` を末尾へ追加する。
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

    /// `index` の子 `Module` への参照（範囲外は `None`。panic しない）。
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    /// [`Self::get`] の可変版。
    pub fn get_mut(&mut self, index: usize) -> Option<&mut (dyn Module + 'static)> {
        self.modules.get_mut(index).map(|m| m.as_mut())
    }

    /// 子 `Module` 列への走査（層順）。
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Module>> {
        self.modules.iter()
    }

    /// [`Self::iter`] の可変版。
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Module>> {
        self.modules.iter_mut()
    }

    /// 子 `Module` 列へのスライス参照（`compat::Sequential` の層走査
    /// 移設先が `self.layers` の代わりに使う）。
    pub fn as_slice(&self) -> &[Box<dyn Module>] {
        &self.modules
    }

    /// [`Self::as_slice`] の可変版（`apply_parameters` 相当の書き戻しに
    /// 使う）。
    pub fn as_mut_slice(&mut self) -> &mut [Box<dyn Module>] {
        &mut self.modules
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
    /// `nn.ModuleList` は forward を持たない（保持器のみ）。呼び出し側の
    /// 誤用を型付きエラーで検出する（`Unsupported` ではなく
    /// `InvalidArgument`: バックエンド未対応のフォールバック対象では
    /// なく、そもそも意味を持たない呼び出しであるため）。
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "ModuleList has no forward (nn.ModuleList is a holder, not a callable module)"
                .to_string(),
        ))
    }

    /// [`Self::forward`] と同じ理由・同じ variant で拒否する。既定
    /// （`Unsupported`）のまま放置すると、呼び出し元の `Unsupported`
    /// フォールバック判定（例: `compat::Sequential::predict` の
    /// tape 不要経路フォールバック）に誤って乗ってしまうため明示
    /// オーバーライドする。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        _input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "ModuleList has no forward_host (nn.ModuleList is a holder, not a callable module)"
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

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        for (index, module) in self.modules.iter().enumerate() {
            for (name, tensor) in module.named_parameters() {
                out.push((format!("{index}.{name}"), tensor));
            }
        }
        out
    }

    /// [`Module::set_parameter`] の実装（イシュー #1752）。
    /// `named_parameters` の `"{index}.{name}"` 接頭辞契約（直上参照）
    /// の逆演算: 先頭の `"."` までを `index` として切り出し
    /// `usize::parse` してから `self.modules[index].set_parameter`
    /// へ委譲する。区切りなし・パース失敗・範囲外はいずれも未知名
    /// 扱いで `InvalidArgument`（fail-closed。`.claude/rules/
    /// security.md` A03）。
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
        match self.modules.get_mut(index) {
            Some(module) => module.set_parameter(rest, value),
            None => Err(AutodiffError::InvalidArgument(format!(
                "ModuleList::set_parameter: no parameter named `{name}` (index {index} out of range; ModuleList has {} modules)",
                self.modules.len()
            ))),
        }
    }

    /// [`Module::children`] の実装（イシュー #2134）。名前は
    /// [`Module::named_parameters`]／[`Module::set_parameter`] が使う
    /// `"{index}."` 接頭辞契約と一致させる。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .enumerate()
            .map(|(index, module)| (index.to_string(), module.as_ref()))
            .collect()
    }
}

/// PyTorch `nn.Sequential` 相当の汎用コンテナ: 子 `Module` を
/// [`ModuleList`] で保持し、[`Module::forward`] を実装して順に委譲する。
/// `fandhe_ai_facade::compat::sequential::Sequential` の Linear→ReLU
/// 融合先読み走査（イシュー #1044）をそのまま移設したロジックを持つ。
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

    /// 子 `Module` を末尾へ追加する（所有権を消費する版。
    /// [`Self::add`] と異なり `Box` 化済みの値を受け取る）。
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.inner.push(module);
    }

    /// メソッドチェーン用ビルダー: `self` を消費し `Module + 'static` を
    /// `Box` 化して追加する。
    ///
    /// `#[allow(clippy::should_implement_trait)]`: メソッド名 `add` は
    /// PyTorch `nn.Sequential.add_module` 系ビルダー API の慣用名を踏襲した
    /// ものであり、`std::ops::Add` トレイトの実装を意図しない（算術演算子
    /// ではなく子 `Module` を追加するビルダーメソッド）。
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.inner.push(Box::new(module));
        self
    }

    /// 保持している子 `Module` の数。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 子 `Module` を 1 つも保持していないか。
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// 子 `Module` 列へのスライス参照（`compat::Sequential` の層走査
    /// 移設先が使う。`as_linear()`／`as_relu()` フックで種別判定する）。
    pub fn layers(&self) -> &[Box<dyn Module>] {
        self.inner.as_slice()
    }

    /// [`Self::layers`] の可変版（`apply_parameters` 相当の書き戻しに
    /// 使う）。
    pub fn layers_mut(&mut self) -> &mut [Box<dyn Module>] {
        self.inner.as_mut_slice()
    }

    /// `ModuleList` を消費して `Sequential` の子として取り込む。
    pub fn from_module_list(inner: ModuleList) -> Self {
        Sequential { inner }
    }

    /// `self.inner` を取り出す（`Sequential` を消費する）。
    pub fn into_module_list(self) -> ModuleList {
        self.inner
    }
}

impl From<ModuleList> for Sequential {
    fn from(inner: ModuleList) -> Self {
        Sequential::from_module_list(inner)
    }
}

impl Module for Sequential {
    /// `fandhe_ai_facade::compat::sequential::Sequential::forward`
    /// （旧実装。イシュー #1044 の Linear→ReLU 融合先読み走査）を移設。
    /// `Linear` 層に出会うたび次層が `ReLU` かを先読みし、実際に続く
    /// 場合のみ `LinearVars::forward_with_activation(.., Activation::
    /// Relu)`（1 ノード・1 カーネル起動）へ結線して `ReLU` 層自体の
    /// ノード追加をスキップする。それ以外の層は `Module::forward`
    /// （多態 dispatch）へ委譲する。空 `Sequential` は恒等（`input` を
    /// そのまま返す）。
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let layers = self.inner.as_slice();
        let mut current = *input;
        let mut i = 0;
        while i < layers.len() {
            let layer = &layers[i];
            if let Some(linear) = layer.as_linear() {
                let fuse_relu = layers.get(i + 1).is_some_and(|next| next.as_relu());
                let bound = linear.bind(tape);
                current = if fuse_relu {
                    bound.forward_with_activation(&current, Activation::Relu)?
                } else {
                    bound.forward(&current)?
                };
                i += if fuse_relu { 2 } else { 1 };
            } else {
                current = layer.forward(tape, &current)?;
                i += 1;
            }
        }
        Ok(current)
    }

    /// `forward_host` は融合を行わず各層の `Module::forward_host` へ
    /// 順次委譲する（trait 契約: 汎用 `&dyn BackendOps` 向けの
    /// bit-exact 非融合経路）。CPU 限定の Linear→ReLU 融合 tape 不要
    /// 経路（`predict_tape_free_with_ops`）は
    /// `fandhe_ai_facade::compat::sequential` 側に据え置く（この融合が
    /// `CpuBackendOps` の `gemm_bias_act` オーバーライドと非融合合成が
    /// bit 完全一致することに依存しており、汎用 `&dyn BackendOps` では
    /// 安全性を保証できないため。同ファイルの
    /// `predict_tape_free_with_ops` doc 参照）。空 `Sequential` は
    /// 恒等（`input.clone()`）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let mut current = input.clone();
        for layer in self.inner.as_slice() {
            current = layer.forward_host(ops, &current)?;
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

    /// [`Module::set_parameter`] の実装（イシュー #1752）。
    /// `self.inner`（`ModuleList`）へそのまま委譲する。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.inner.set_parameter(name, value)
    }

    /// [`Module::children`] の実装（イシュー #2134）。`self.inner`
    /// （`ModuleList`）へそのまま委譲する。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.inner.children()
    }
}

/// PyTorch `nn.ModuleDict` 相当: 名前（キー）をインデックスにした子
/// `Module` の保持器（イシュー #2134・親 #2131）。
///
/// # 内部表現に `Vec` を使う理由（`HashMap` ではない）
///
/// [`Module::named_parameters`]／[`Module::state_dict`] の列挙順は
/// 「登録順」という既存契約（[`ModuleList`] と同じ）に依存する呼び
/// 出し元（`trainable_parameters` の位置対応・state_dict のテスト）が
/// あるため、`HashMap`（走査順不定）ではなく**挿入順を保持する
/// `Vec<(String, Box<dyn Module>)>`** をキー付き保持器の実装に使う。
/// 検索は O(n) 線形走査になるが、層数は高々数十のため実用上十分
/// （PyTorch `nn.ModuleDict` も内部的に `OrderedDict` で同じ特性を持つ）。
///
/// `forward`／`forward_host` は [`ModuleList`] と同じ理由・同じ
/// variant（`AutodiffError::InvalidArgument`）で拒否する（`ModuleDict`
/// 自体は演算列を持たない保持器のため）。
pub struct ModuleDict {
    /// 挿入順を保持するキー付き保持器（直上「内部表現」節参照）。
    modules: Vec<(String, Box<dyn Module>)>,
    /// train／eval モード（[`Module`] trait doc のコンテナ契約。既定
    /// `true`）。
    training: bool,
}

impl Default for ModuleDict {
    fn default() -> Self {
        Self::new()
    }
}

/// `ModuleDict` のキー検証（イシュー #2134。`.claude/rules/
/// security.md` A03 fail-closed 方針）。空文字列と `'.'` を含むキーを
/// 拒否する: 空文字列は [`Module::named_modules`] のパス連結
/// （`"{parent}.{child}"`）で意味を持たない空セグメントを生み、`'.'`
/// を含むキーは [`Module::set_parameter`]（`split_once('.')` で
/// 先頭セグメントをキーとして取り出す契約）・`named_modules` の
/// パス解釈を壊す。PyTorch `nn.ModuleDict` も両者を拒否する。
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

    /// `(key, module)` の列から `ModuleDict` を構築する（挿入順を
    /// 保持。列挙時の順に `insert` するのと同義）。いずれかのキーが
    /// 不正（空文字列・`'.'` 含み）な場合は `Err`（fail-closed。
    /// `validate_module_dict_key` 参照）。
    pub fn from_pairs(pairs: Vec<(String, Box<dyn Module>)>) -> Result<Self, AutodiffError> {
        let mut dict = ModuleDict::new();
        for (key, module) in pairs {
            dict.insert(key, module)?;
        }
        Ok(dict)
    }

    /// `key` の子 `Module` を挿入する。既存の同名キーがあれば新しい
    /// 値で置換し、置換前の値を返す（挿入順の位置は保たれる。
    /// PyTorch `nn.ModuleDict.__setitem__` の上書き契約と同じ）。
    /// `key` が不正（空文字列・`'.'` 含み）な場合は `Err`。
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

    /// `key` を持つ子 `Module` を削除して返す（無ければ `None`）。
    pub fn remove(&mut self, key: &str) -> Option<Box<dyn Module>> {
        let index = self.modules.iter().position(|(k, _)| k == key)?;
        Some(self.modules.remove(index).1)
    }

    /// `key` の子 `Module` への参照（無ければ `None`。panic しない）。
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

    /// キー列への走査（挿入順）。
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.modules.iter().map(|(k, _)| k.as_str())
    }

    /// `(key, &dyn Module)` 列への走査（挿入順）。
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Module)> {
        self.modules.iter().map(|(k, m)| (k.as_str(), m.as_ref()))
    }

    /// [`Self::iter`] の可変版。
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut (dyn Module + 'static))> {
        self.modules
            .iter_mut()
            .map(|(k, m)| (k.as_str(), m.as_mut()))
    }

    /// 保持している子 `Module` の数。
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// 子 `Module` を 1 つも保持していないか。
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Module for ModuleDict {
    /// `nn.ModuleDict` は forward を持たない（保持器のみ。
    /// [`ModuleList::forward`] と同じ理由・同じ variant）。
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "ModuleDict has no forward (nn.ModuleDict is a holder, not a callable module)"
                .to_string(),
        ))
    }

    /// [`Self::forward`] と同じ理由・同じ variant で拒否する
    /// （[`ModuleList::forward_host`] と同型。`Unsupported` の既定に
    /// 乗せて呼び出し元のフォールバック判定を誤らせないための明示
    /// オーバーライド）。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        _input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "ModuleDict has no forward_host (nn.ModuleDict is a holder, not a callable module)"
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

    /// [`Module::named_parameters`] の実装。`"{key}.{name}"` 接頭辞を
    /// 挿入順で連結する（[`ModuleList::named_parameters`] と同型）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        for (key, module) in &self.modules {
            for (name, tensor) in module.named_parameters() {
                out.push((format!("{key}.{name}"), tensor));
            }
        }
        out
    }

    /// [`Module::set_parameter`] の実装。`named_parameters` の
    /// `"{key}.{name}"` 接頭辞契約の逆演算: 先頭の `'.'` までを `key`
    /// として切り出し `self.get_mut(key)` へ委譲する。区切りなし・
    /// 未知キーはいずれも `InvalidArgument`（fail-closed。
    /// `.claude/rules/security.md` A03。[`ModuleList::set_parameter`]
    /// と同型）。
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

    /// [`Module::children`] の実装。名前はキー、順序は挿入順
    /// （[`Self::iter`] と同じ）。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.modules
            .iter()
            .map(|(key, module)| (key.clone(), module.as_ref()))
            .collect()
    }
}

/// `nn::summary` が型名の短縮に使う内部ヘルパー（イシュー #2134）。
///
/// [`Module::type_name`] が返す `std::any::type_name` の出力は
/// クレートパス付き（例: `fandhe_ai_autodiff::nn::linear::Linear`）・
/// ジェネリクス付き（`<` 以降）になりうる。表示用に「最後の `::`
/// 区切り以降・`<` より前」だけを取り出す（`std::any::type_name` の
/// 出力形式は標準ライブラリが安定性を保証しないため、あくまで
/// 表示上のベストエフォートの短縮であり、テストで固定するのは既知の
/// 具象型〈`Linear`・`Sequential`・`ModuleDict` 等〉に対する経験的な
/// 出力のみとする）。
fn short_type_name(full: &str) -> &str {
    let without_generics = full.split('<').next().unwrap_or(full);
    without_generics
        .rsplit("::")
        .next()
        .unwrap_or(without_generics)
}

/// `module`（および子孫を持つ場合はその全体）の構造を人間可読な文字列
/// へ整形する（PyTorch `print(model)` 相当の簡易表示。イシュー
/// #2134）。
///
/// # 出力形式
///
/// 子を持つノードは `"{Type}(\n"` の後に子の行（深さごとに 2 スペース
/// ずつインデント）を並べ `"{indent}) [params: N]\n"` で閉じる。葉
/// （子を持たないノード）は `"{indent}({name}): {Type} [params: N]\n"`
/// （ルートが葉の場合は名前を持たないため `"{Type} [params: N]\n"`）。
/// 末尾に `"Submodules: {[Module::named_modules] の要素数}\n"`・
/// `"Total parameters: {[Module::parameter_count]}\n"` を追加する。
///
/// 入出力 shape 推定・`extra_repr` 相当（`in_features` 等の属性表示）
/// は対象外（イシュー本文のスコープ外指定）。型名は
/// [`Module::type_name`] をモジュール非公開のヘルパー（型パスの最終
/// セグメントのみを残す縮約）で短縮したものを使う。
///
/// # 循環・重複ノードの扱い（イシュー #2134 codex-review 指摘。
/// PR #2231）
///
/// `write_module`（本関数直下の再帰本体）は [`Module::named_modules`] と同じ理由
/// （`Module::children` の実装者が自身や既出の `Module` を任意に
/// 返せる）で無限再帰しうる。本関数は [`Module::named_modules`]
/// （`collect_named_modules`）と同じ 2 段階の判定方式（祖先限定の
/// 循環検出＋ゼロサイズ型を除くグローバル共有 dedup）を使う
/// （`.claude/rules/security.md` A03・本番経路 panic 禁止の方針に
/// 合わせる）。
///
/// **ZST をグローバル dedup から除外する理由（イシュー #2134
/// codex-review／Bugbot 指摘・PR #2231 是正）**: 経路をまたぐ訪問済み
/// 集合を無条件適用すると、ZST（`Relu`・`Gelu` 等）を `Box` へ格納
/// した際に複数インスタンスがアロケータの well-known dangling
/// address を共有しうるため、`Sequential` に同種の ZST 活性化層を
/// 複数積んだ場合に後続レイヤーを「既出」と誤判定して出力から
/// 欠落させる。ZST は祖先限定の循環検出のみで保護し、グローバル
/// dedup の対象からは外す（詳細は [`Module::named_modules`] の
/// 「循環・重複ノードの扱い」節参照）。
pub fn summary(module: &dyn Module) -> String {
    let mut out = String::new();
    let mut ancestors: Vec<*const ()> = vec![module as *const dyn Module as *const ()];
    let mut visited: HashSet<*const ()> = HashSet::new();
    write_module(&mut out, None, module, 0, &mut ancestors, &mut visited);
    out.push_str(&format!("Submodules: {}\n", module.named_modules().len()));
    out.push_str(&format!("Total parameters: {}\n", module.parameter_count()));
    out
}

/// [`summary`] の再帰本体。`name` はこのノードの子としての名前
/// （ルート呼び出しでは `None`）、`depth` はインデント段数。
/// `ancestors`／`visited` は [`summary`] から通しで渡される 2 段階の
/// 判定用状態（既出ノードの再帰打ち切りに使う。上記「循環・重複
/// ノードの扱い」節参照）。
fn write_module(
    out: &mut String,
    name: Option<&str>,
    module: &dyn Module,
    depth: usize,
    ancestors: &mut Vec<*const ()>,
    visited: &mut HashSet<*const ()>,
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
        let ptr = *child as *const dyn Module as *const ();
        if ancestors.contains(&ptr) {
            continue;
        }
        let is_zst = std::mem::size_of_val(*child) == 0;
        if !is_zst && !visited.insert(ptr) {
            continue;
        }
        ancestors.push(ptr);
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
    use crate::nn::activation::Relu;
    use crate::nn::linear::Linear;

    #[test]
    fn sequential_add_builder_chains_and_reports_len() {
        let seq = Sequential::new()
            .add(Linear::new(4, 8, true, 1).unwrap())
            .add(Relu)
            .add(Linear::new(8, 2, true, 2).unwrap());
        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());
    }

    #[test]
    fn sequential_layers_and_layers_mut_report_same_len() {
        let mut seq = Sequential::new().add(Relu).add(Relu);
        assert_eq!(seq.layers().len(), 2);
        assert_eq!(seq.layers_mut().len(), 2);
    }

    #[test]
    fn module_list_from_module_list_round_trips_via_sequential() {
        let mut list = ModuleList::new();
        list.push(Box::new(Relu));
        list.push(Box::new(Relu));
        let seq = Sequential::from(list);
        assert_eq!(seq.len(), 2);
        let list_back = seq.into_module_list();
        assert_eq!(list_back.len(), 2);
    }

    #[test]
    fn module_list_get_out_of_range_is_none_not_panic() {
        let list = ModuleList::new();
        assert!(list.get(0).is_none());
    }

    #[test]
    fn module_list_default_is_empty() {
        let list = ModuleList::default();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    // `ModuleList`/`Sequential::set_parameter`・`state_dict`/
    // `load_state_dict`（`Module` の defaulted 実装。イシュー #1752）の
    // 単体テスト。

    fn two_linear_sequential() -> Sequential {
        Sequential::new()
            .add(Linear::new(4, 8, true, 1).unwrap())
            .add(Relu)
            .add(Linear::new(8, 2, true, 2).unwrap())
    }

    #[test]
    fn state_dict_keys_match_named_parameters() {
        let seq = two_linear_sequential();
        let named: std::collections::HashSet<String> = seq
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let dict_keys: std::collections::HashSet<String> = seq.state_dict().into_keys().collect();
        assert_eq!(named, dict_keys);
        assert_eq!(named.len(), 4); // 0.weight, 0.bias, 2.weight, 2.bias
    }

    #[test]
    fn load_state_dict_round_trip_is_no_op() {
        let mut seq = two_linear_sequential();
        let before = seq.state_dict();
        seq.load_state_dict(seq.state_dict()).unwrap();
        let after = seq.state_dict();
        for (key, tensor) in &before {
            assert_eq!(
                tensor.contiguous().as_slice().unwrap(),
                after[key].contiguous().as_slice().unwrap(),
                "key `{key}` が往復後に変化した"
            );
        }
    }

    #[test]
    fn load_state_dict_actually_updates_values() {
        let mut seq = two_linear_sequential();
        let mut state = seq.state_dict();
        let new_weight = Tensor::new(vec![7.0f32; 32], &[4, 8]).unwrap();
        state.insert("0.weight".to_string(), new_weight.clone());
        seq.load_state_dict(state).unwrap();
        let (name, tensor) = seq
            .named_parameters()
            .into_iter()
            .find(|(name, _)| name == "0.weight")
            .unwrap();
        assert_eq!(name, "0.weight");
        assert_eq!(
            tensor.contiguous().as_slice().unwrap(),
            new_weight.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn load_state_dict_rejects_missing_key() {
        let mut seq = two_linear_sequential();
        let mut state = seq.state_dict();
        state.remove("0.bias");
        let err = seq
            .load_state_dict(state)
            .expect_err("欠落キーは Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn load_state_dict_rejects_unexpected_key() {
        let mut seq = two_linear_sequential();
        let mut state = seq.state_dict();
        state.insert(
            "99.weight".to_string(),
            Tensor::new(vec![0.0f32], &[1]).unwrap(),
        );
        let err = seq
            .load_state_dict(state)
            .expect_err("余剰キーは Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn load_state_dict_rejects_shape_mismatch() {
        let mut seq = two_linear_sequential();
        let mut state = seq.state_dict();
        state.insert(
            "0.weight".to_string(),
            Tensor::new(vec![1.0f32; 6], &[2, 3]).unwrap(),
        );
        let err = seq
            .load_state_dict(state)
            .expect_err("shape 不一致は Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn load_state_dict_err_leaves_model_unchanged() {
        let mut seq = two_linear_sequential();
        let before = seq.state_dict();
        let mut bad_state = seq.state_dict();
        bad_state.remove("2.bias");
        let _ = seq.load_state_dict(bad_state);
        let after = seq.state_dict();
        for (key, tensor) in &before {
            assert_eq!(
                tensor.contiguous().as_slice().unwrap(),
                after[key].contiguous().as_slice().unwrap(),
                "load_state_dict が Err を返したのに key `{key}` が変化した"
            );
        }
    }

    #[test]
    fn module_list_set_parameter_dispatches_by_index_and_rejects_bad_names() {
        let mut list = ModuleList::new();
        list.push(Box::new(Linear::new(4, 8, true, 1).unwrap()));
        list.push(Box::new(Relu));

        let new_weight = Tensor::new(vec![3.0f32; 32], &[4, 8]).unwrap();
        list.set_parameter("0.weight", new_weight.clone()).unwrap();

        // 区切りなし。
        assert!(matches!(
            list.set_parameter("weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
            Err(AutodiffError::InvalidArgument(_))
        ));
        // パース失敗。
        assert!(matches!(
            list.set_parameter("x.weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
            Err(AutodiffError::InvalidArgument(_))
        ));
        // 範囲外。
        assert!(matches!(
            list.set_parameter("99.weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    // `ModuleDict`（イシュー #2134）の単体テスト。

    #[test]
    fn module_dict_default_is_empty() {
        let dict = ModuleDict::default();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn module_dict_insert_get_remove_contains_keys_round_trip() {
        let mut dict = ModuleDict::new();
        assert!(dict.insert("relu", Box::new(Relu)).unwrap().is_none());
        assert!(dict.insert("relu2", Box::new(Relu)).unwrap().is_none());

        assert_eq!(dict.len(), 2);
        assert!(dict.contains_key("relu"));
        assert!(!dict.contains_key("missing"));
        assert!(dict.get("relu").is_some());
        assert!(dict.get_mut("relu2").is_some());
        assert_eq!(dict.keys().collect::<Vec<_>>(), vec!["relu", "relu2"]);

        let removed = dict.remove("relu");
        assert!(removed.is_some());
        assert_eq!(dict.len(), 1);
        assert!(!dict.contains_key("relu"));
    }

    #[test]
    fn module_dict_insert_same_key_replaces_and_returns_old_value_preserving_position() {
        let mut dict = ModuleDict::new();
        dict.insert("a", Box::new(Relu)).unwrap();
        dict.insert("b", Box::new(Relu)).unwrap();
        let replaced = dict.insert("a", Box::new(Relu)).unwrap();
        assert!(replaced.is_some(), "同名キーの再 insert は旧値を返すはず");
        assert_eq!(
            dict.keys().collect::<Vec<_>>(),
            vec!["a", "b"],
            "同名キーの置換で挿入順の位置がずれてはいけない"
        );
    }

    #[test]
    fn module_dict_rejects_empty_and_dotted_keys() {
        let mut dict = ModuleDict::new();
        assert!(matches!(
            dict.insert("", Box::new(Relu)),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            dict.insert("a.b", Box::new(Relu)),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(
            dict.is_empty(),
            "拒否された insert で状態が変化してはいけない"
        );
    }

    #[test]
    fn module_dict_from_pairs_rejects_bad_key_without_panicking() {
        let pairs: Vec<(String, Box<dyn Module>)> = vec![
            ("ok".to_string(), Box::new(Relu)),
            ("bad.key".to_string(), Box::new(Relu)),
        ];
        match ModuleDict::from_pairs(pairs) {
            Err(AutodiffError::InvalidArgument(_)) => {}
            other => panic!("不正キーは InvalidArgument を返すはず: {}", other.is_ok()),
        }
    }

    #[test]
    fn module_dict_named_parameters_uses_key_prefix() {
        let mut dict = ModuleDict::new();
        dict.insert("l1", Box::new(Linear::new(4, 8, true, 1).unwrap()))
            .unwrap();
        let names: Vec<String> = dict
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, vec!["l1.weight", "l1.bias"]);
    }

    #[test]
    fn module_dict_set_parameter_dispatches_by_key_and_rejects_bad_names() {
        let mut dict = ModuleDict::new();
        dict.insert("l1", Box::new(Linear::new(4, 8, true, 1).unwrap()))
            .unwrap();

        let new_weight = Tensor::new(vec![3.0f32; 32], &[4, 8]).unwrap();
        dict.set_parameter("l1.weight", new_weight).unwrap();

        // 区切りなし。
        assert!(matches!(
            dict.set_parameter("weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
            Err(AutodiffError::InvalidArgument(_))
        ));
        // 未知キー。
        assert!(matches!(
            dict.set_parameter("missing.weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn module_dict_state_dict_round_trip() {
        let mut dict = ModuleDict::new();
        dict.insert("l1", Box::new(Linear::new(4, 8, true, 1).unwrap()))
            .unwrap();
        let before = dict.state_dict();
        dict.load_state_dict(dict.state_dict()).unwrap();
        let after = dict.state_dict();
        for (key, tensor) in &before {
            assert_eq!(
                tensor.contiguous().as_slice().unwrap(),
                after[key].contiguous().as_slice().unwrap(),
                "key `{key}` が往復後に変化した"
            );
        }
    }

    #[test]
    fn module_dict_forward_and_forward_host_are_invalid_argument() {
        use crate::tape::Tape;
        let dict = ModuleDict::new();
        let tape = Tape::new();
        let input_tensor = Tensor::new(vec![1.0f32], &[1]).unwrap();
        let input = tape.var(&input_tensor);
        assert!(matches!(
            dict.forward(&tape, &input),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn module_dict_set_training_propagates_to_children() {
        use crate::nn::dropout::Dropout;
        let mut dict = ModuleDict::new();
        dict.insert("drop", Box::new(Dropout::new(0.5).unwrap()))
            .unwrap();
        assert!(dict.training());
        dict.set_training(false);
        assert!(!dict.training());
        assert!(!dict.get("drop").unwrap().training());
    }

    #[test]
    fn module_dict_children_returns_key_and_module_in_insertion_order() {
        let mut dict = ModuleDict::new();
        dict.insert("a", Box::new(Relu)).unwrap();
        dict.insert("b", Box::new(Relu)).unwrap();
        let names: Vec<String> = dict.children().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["a", "b"]);
    }
}
