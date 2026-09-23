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

    /// [`Module::set_requires_grad`] の実装（イシュー #2137）。全子
    /// `Module` へ層順に伝播する。`Module::load_state_dict`（本ファイル
    /// 冒頭 doc が参照する `module.rs`）と同型の**ベストエフォート・
    /// ロールバック**: 子の 1 つが `Err` を返した場合、それより前に
    /// 適用済みの子を逆順に元の `requires_grad()` へ戻す（`freeze()`
    /// が途中で失敗しても「一部の層だけ凍結された」状態を残さない
    /// 意図。`Module::set_requires_grad` 既定実装 doc「fail-closed」節
    /// 参照）。ロールバック自体が失敗した場合は、その旨を明示した
    /// `InvalidArgument` を返す（`load_state_dict` と同じ方針）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        // 適用前の値を層順に記録する（ロールバック用）。`Module::
        // requires_grad`（既定 `true`）は無状態層でも呼べるため、この
        // スナップショットは全子に対して失敗しない。
        let previous: Vec<bool> = self.modules.iter().map(|m| m.requires_grad()).collect();

        for (index, module) in self.modules.iter_mut().enumerate() {
            if let Err(err) = module.set_requires_grad(requires_grad) {
                // 適用済みの子（`0..index`）を逆順に元の値へ戻す。
                for rollback_index in (0..index).rev() {
                    let Some(rollback_module) = self.modules.get_mut(rollback_index) else {
                        continue;
                    };
                    if let Err(rollback_err) =
                        rollback_module.set_requires_grad(previous[rollback_index])
                    {
                        return Err(AutodiffError::InvalidArgument(format!(
                            "ModuleList::set_requires_grad: failed to apply to module {index} \
                             ({err}), and rollback of already-applied module {rollback_index} \
                             also failed ({rollback_err}); the ModuleList may now be left in a \
                             partially applied state"
                        )));
                    }
                }
                return Err(err);
            }
        }
        Ok(())
    }

    /// 子が 1 つも `false` を返さなければ `true`（`Module::
    /// requires_grad` の複合層契約「子が 1 つでも `true` を返せば
    /// `true`」の否定形。子が 1 つも無い空 `ModuleList` は `true`
    /// を返す——`Module::requires_grad` 既定と同じ「パラメータを
    /// 持たなければ凍結状態を持たない」契約に揃える）。
    fn requires_grad(&self) -> bool {
        self.modules.iter().all(|m| m.requires_grad())
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

    /// [`Module::set_requires_grad`] の実装（イシュー #2137）。
    /// `self.inner`（`ModuleList`）へそのまま委譲する（ベストエフォート・
    /// ロールバックも `ModuleList::set_requires_grad` の実装に従う）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        Module::set_requires_grad(&mut self.inner, requires_grad)
    }

    fn requires_grad(&self) -> bool {
        Module::requires_grad(&self.inner)
    }
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
}
