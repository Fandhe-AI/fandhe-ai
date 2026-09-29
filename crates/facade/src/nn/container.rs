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
//! キー昇順適用・逆順ロールバック）に任せる。
//!
//! autodiff 側との差: 子は facade `Module` の不透明な trait object のため、
//! `Sequential::forward` は autodiff 側の Linear→ReLU 融合を行わず子を順に適用するだけである。
//! また `push` した子のモードはコンテナの現在モードへ同期されない（autodiff 側と同じ。
//! 次の `set_training` で伝わる）。`compat::Sequential`（`crate::compat::sequential`）とは
//! 別パスの別型である。

use crate::nn::Module;
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
}
