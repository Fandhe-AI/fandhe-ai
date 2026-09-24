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
use crate::nn::module::{Module, NodeKey};
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

    /// [`Module::set_requires_grad`] の実装（イシュー #2137。レビュー
    /// 是正 PR #2234: 集約判定を `any` へ修正し、ロールバックを
    /// 再帰的な子別スナップショット方式へ変更）。全子 `Module` へ
    /// 層順に伝播する。`Module::load_state_dict`（本ファイル冒頭 doc が
    /// 参照する `module.rs`）と同型の**ベストエフォート・ロールバック**:
    /// 子の 1 つが `Err` を返した場合、それより前に適用済みの子
    /// **と失敗した子自身**（`0..=index`。逆順。P1 是正・#2234 レビュー
    /// 指摘 `PRRT_kwDOTuUCJc6lE8Gn`）を元の状態へ戻す（`freeze()` が
    /// 途中で失敗しても「一部の層だけ凍結された」状態を残さない意図。
    /// `Module::set_requires_grad` 既定実装 doc「fail-closed」節参照）。
    /// 個々のロールバックが失敗しても打ち切らず、残りの子のロール
    /// バックは続行する（1 子が復元不能でも、それより前の適用済みの
    /// 子まで未復元のまま諦めない）。
    ///
    /// # 入れ子コンテナの混在状態（P1 是正・#2234 レビュー指摘）
    ///
    /// 子が入れ子の `ModuleList`／`Sequential`（[`Module::
    /// as_module_list`] が `Some` を返す）または `ModuleDict`
    /// （[`Module::as_module_dict`] が `Some` を返す。第 2 ラウンドの
    /// レビュー是正・cursor\[bot\] `PRRT_kwDOTuUCJc6lE80z`）である場合、
    /// その子は内部に混在状態（一部凍結・一部解凍）を持ちうる。単純に
    /// 集約 bool 1 つをスナップショットして `set_requires_grad(bool)` で
    /// 復元すると、復元時に子の全孫へ同一値が強制され混在状態を破壊
    /// してしまう（旧実装の fail-closed 契約違反）。このため
    /// `snapshot_requires_grad` で子孫の bool を末端層単位まで再帰的に
    /// スナップショットし、`restore_requires_grad` で同じ構造をたどって
    /// 1 つずつ復元する（`RequiresGradSnapshot::Nested`／`NestedDict` の
    /// 子数が実行時の子数と一致しない場合は形状不一致として
    /// `InvalidArgument` を返す。ロールバック中に構造が変わることは
    /// 通常起こらないが、fail-closed のため検査する）。ロールバック
    /// 自体が失敗した場合は、その旨を明示した `InvalidArgument` を
    /// 返す（`load_state_dict` と同じ方針）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        // 適用前の状態を層順に再帰的スナップショットとして記録する
        // （ロールバック用）。`Module::requires_grad`（既定 `true`）は
        // 無状態層でも呼べるため、このスナップショットは全子に対して
        // 失敗しない。
        let previous: Vec<RequiresGradSnapshot> = self
            .modules
            .iter()
            .map(|m| snapshot_requires_grad(m.as_ref()))
            .collect();

        for (index, module) in self.modules.iter_mut().enumerate() {
            if let Err(err) = module.set_requires_grad(requires_grad) {
                // 適用済みの子（`0..index`）に加え、失敗した子自身
                // （`index`）も逆順に元の状態へ戻す（P1 是正・#2234
                // レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gn`）。`Module` は外部
                // 実装可能で「エラーを返したら状態を変更しない」契約を
                // 持たないため、複数内部パラメータを順に変更するカスタム
                // 複合層では、エラーを返した子自身が部分適用状態
                // （一部パラメータだけ切り替わった状態）を残している
                // おそれがある。`previous[index]`（適用前スナップショット）
                // で必ず復元し、fail-closed／全体ロールバック契約を守る。
                //
                // 1 子のロールバックが失敗しても、そこで打ち切らず
                // 残りの子（`0..rollback_index`）のロールバックは続行
                // する（是正。復元不能な子が 1 つあるからといって、
                // それより前の適用済みの子——是正前の実装が確実に
                // 復元していた範囲——まで未復元のまま諦めるのは
                // 「全体ロールバック」契約に反する）。失敗した
                // ロールバックはすべて集約してエラーメッセージへ含める。
                let mut rollback_failures: Vec<String> = Vec::new();
                for rollback_index in (0..=index).rev() {
                    let Some(rollback_module) = self.modules.get_mut(rollback_index) else {
                        continue;
                    };
                    if let Err(rollback_err) =
                        restore_requires_grad(rollback_module.as_mut(), &previous[rollback_index])
                    {
                        rollback_failures.push(format!("module {rollback_index} ({rollback_err})"));
                    }
                }
                if !rollback_failures.is_empty() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "ModuleList::set_requires_grad: failed to apply to module {index} \
                         ({err}), and rollback also failed for: {}; the ModuleList may now be \
                         left in a partially applied state",
                        rollback_failures.join(", ")
                    )));
                }
                return Err(err);
            }
        }
        Ok(())
    }

    /// パラメータを持つ子が 1 つでも `true` を返せば `true`
    /// （[`Module::requires_grad`] の複合層契約「子が 1 つでも `true`
    /// なら `true`」どおり）。子が 1 つも無い、またはどの子も
    /// パラメータを持たない場合は `true` を返す（`any` の空列に対する
    /// 数学的な既定は `false` だが、[`Module::requires_grad`] 既定実装
    /// doc の「既定 `true`。パラメータを持たない層は状態を保持しない
    /// ため `freeze()` 後も `true` のまま」契約に合わせる——`ModuleList`
    /// 自体もパラメータを持たない空コンテナのときは同じ契約に従うべき
    /// であり、素の `any` の空列既定 `false` を採用すると、空の
    /// `ModuleList`／`Sequential` だけ「常に凍結扱い」という他の無状態層
    /// と矛盾する挙動になる）。
    ///
    /// # 是正記録（P1・#2234 レビュー指摘）
    ///
    /// 旧実装は `all`（全子が `true` の場合のみ `true`。空列は `true`）
    /// だったが、上記 doc の公開契約（「子が 1 つでも `true` なら
    /// `true`」）と逆であり、一部凍結・一部解凍の `ModuleList`／
    /// `Sequential` が誤って `false` を返す不具合だった。`any` へ修正
    /// する際、空列既定は `Module::requires_grad` の無状態層契約を
    /// 保つため `all` と同じ `true` のまま変更していない。
    ///
    /// # 是正記録（P1・Cursor Bugbot 指摘・PR #2234 review thread
    /// `PRRT_kwDOTuUCJc6lEbAQ`）
    ///
    /// 単純な `any` 集約は、`Relu` 等パラメータを持たない層
    /// （[`Module::requires_grad`] 既定実装により常に `true` を返す）を
    /// 無条件に集約対象へ含めてしまうため、典型的な
    /// `Linear → Relu → Linear` 構成では、両 `Linear` を `freeze()` して
    /// 全パラメータ leaf が凍結済みでも `Relu` の既定 `true` に引きずら
    /// れて全体が `true`（trainable）と誤って報告される不具合があった。
    /// 是正: [`Module::parameter_count`]（`named_parameters` を再帰的に
    /// 数える）でパラメータを 1 つも持たない子（無状態層。入れ子コン
    /// テナの場合は子孫すべてが無状態のケースを含む）を集約対象から
    /// 除外し、パラメータを持つ子のみで `any` を取る。パラメータを持つ
    /// 子が 1 つも無ければ（無状態層のみ、または空列）上記「無状態層は
    /// 既定 `true`」契約に従い `true` を返す。
    fn requires_grad(&self) -> bool {
        let mut saw_param_bearing_child = false;
        for module in &self.modules {
            if module.parameter_count() == 0 {
                // パラメータを持たない層（Relu 等）は「常に true」の
                // 既定契約を持つだけで凍結状態を表さないため、集約から
                // 除外する（直上是正記録参照）。
                continue;
            }
            saw_param_bearing_child = true;
            if module.requires_grad() {
                return true;
            }
        }
        !saw_param_bearing_child
    }

    fn as_module_list(&self) -> Option<&ModuleList> {
        Some(self)
    }

    fn as_module_list_mut(&mut self) -> Option<&mut ModuleList> {
        Some(self)
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

/// [`ModuleList::set_requires_grad`] のロールバック用、子 1 体分の
/// 再帰的 `requires_grad` スナップショット（P1 是正・#2234 レビュー
/// 指摘）。末端層（[`Module::as_module_list`] が `None` を返す層）は
/// 集約 bool 1 つ（`Leaf`）で表現できるが、入れ子コンテナ
/// （`ModuleList`／`Sequential`）は孫の混在状態を保持するため、孫ごとに
/// 再帰した `Nested` で表現する。
enum RequiresGradSnapshot {
    /// 末端層（非コンテナ）の集約 `requires_grad()` 値。
    Leaf(bool),
    /// 入れ子 `ModuleList`／`Sequential`（[`Module::as_module_list`] が
    /// `Some` を返す）の子 1 体ごとのスナップショット（層順）。
    Nested(Vec<RequiresGradSnapshot>),
    /// 入れ子 `ModuleDict`（[`Module::as_module_dict`] が `Some` を返す。
    /// P1 是正・#2234 レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gg`／cursor\[bot\]
    /// `PRRT_kwDOTuUCJc6lE80z`）の子 1 体ごとのスナップショット（挿入
    /// 順。`ModuleDict::keys`／`iter` と同じ順序）。`ModuleList` の
    /// `Nested` と別 variant にする理由は `restore_requires_grad` 側で
    /// `as_module_list_mut`／`as_module_dict_mut` のどちらへ委譲するかを
    /// スナップショット自体から判別するため（`ModuleList` と
    /// `ModuleDict` は別の内部表現を持つ別コンテナ型であり、取り違えると
    /// 復元が `as_module_list_mut`（`None` を返す）に落ちて fail-closed
    /// エラーになる）。
    NestedDict(Vec<RequiresGradSnapshot>),
}

/// `module` の現在の `requires_grad` 状態を再帰的にスナップショットする
/// （[`RequiresGradSnapshot`] 参照）。`ModuleList`／`Sequential`
/// （[`Module::as_module_list`]）と `ModuleDict`（[`Module::
/// as_module_dict`]）の両方を入れ子コンテナとして認識する（どちらか
/// 一方しか見ないと、他方がネストした場合に末端層として単一 bool へ
/// 潰され、混在状態（一部凍結・一部解凍）がロールバックで破壊される。
/// P1 是正・#2234 レビュー指摘）。
fn snapshot_requires_grad(module: &dyn Module) -> RequiresGradSnapshot {
    if let Some(list) = module.as_module_list() {
        return RequiresGradSnapshot::Nested(
            list.modules
                .iter()
                .map(|m| snapshot_requires_grad(m.as_ref()))
                .collect(),
        );
    }
    if let Some(dict) = module.as_module_dict() {
        return RequiresGradSnapshot::NestedDict(
            dict.modules
                .iter()
                .map(|(_, m)| snapshot_requires_grad(m.as_ref()))
                .collect(),
        );
    }
    RequiresGradSnapshot::Leaf(module.requires_grad())
}

/// `snapshot_requires_grad` で取得したスナップショットへ `module` の
/// 状態を復元する。`Nested`／`NestedDict` の子数が実行時の子数と
/// 一致しない場合は fail-closed で `InvalidArgument` を返す（通常の
/// ロールバック経路では構造が変わらないため起こらないはずだが、想定外の
/// 構成変化を静かに無視しないため検査する）。
///
/// `Nested`／`NestedDict` は `?` による早期 return を使わず、子の 1 つが
/// 復元に失敗しても**必ず残り全ての子を処理してから**集約エラーを返す
/// （P1 是正・codex-review 指摘 `PRRT_kwDOTuUCJc6lFh9N`。詳細は
/// [`collect_restore_errors`] doc 参照）。
fn restore_requires_grad(
    module: &mut dyn Module,
    snapshot: &RequiresGradSnapshot,
) -> Result<(), AutodiffError> {
    match snapshot {
        RequiresGradSnapshot::Leaf(value) => {
            // `0..=index`（失敗した子自身を含むロールバック範囲。P1
            // 是正・#2234 レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gn`）により、
            // ここで復元しようとしている子は「もともと失敗した
            // `set_requires_grad` 呼び出しの当事者」でありうる。
            //
            // 呼び出し**前**に「すでに一致しているか」を見て早期 `Ok`
            // にする最適化は行わない（`Module` は外部実装可能で
            // `requires_grad()` が既定 `true` のまま・`set_requires_grad`
            // だけを正しくオーバーライドする実装もありうるため。この
            // 場合スナップショット `Leaf(true)` と実際の呼び出し前の値
            // `true` が一致していても、実際に `set_requires_grad(*value)`
            // を呼ばなければ復元されない）。したがって常に
            // `set_requires_grad` を実際に呼ぶ。
            //
            // `Err` を返した場合は、その `Err` をそのまま伝播する
            // （P1 是正・codex-review 指摘 `PRRT_kwDOTuUCJc6lFaVh`）。
            // 旧実装は「呼び出し後の `requires_grad()`（集約 bool）が
            // `value` と一致するか」で復元成功を判定していたが、`Leaf`
            // はこの関数の再帰から見た粒度であり、`Module` は外部実装
            // 可能で内部に複数の独立した子状態を持つ「複合 Module」
            // でありうる（`as_module_list`／`as_module_dict` を実装せず
            // `Leaf` 扱いされる不透明な実装）。そのような実装の
            // `requires_grad()` が例えば「子のいずれかが `true` なら
            // `true`」という集約契約を持つ場合、一部の子だけ `value` へ
            // 書き換えに成功し残りは失敗した部分適用状態でも、集約結果が
            // たまたま `value` と一致してしまい「復元成功」と誤判定
            // （false positive）しうる。`Leaf` のスナップショットには
            // 集約 bool 1 つしか無く、これ以上の粒度で内部状態を検証
            // する手段が無い（`Module::children` は `&dyn` のみで
            // `_mut` を持たず、複合実装が `as_module_list_mut`／
            // `as_module_dict_mut` を実装しない限り再帰スナップショット
            // 化できない）ため、検証不能な場合は fail-closed に「`Err`
            // をそのまま呼び出し元へ返す」を採用し、集約値の一致を
            // 復元成功の根拠にしない。
            module.set_requires_grad(*value)
        }
        RequiresGradSnapshot::Nested(children) => match module.as_module_list_mut() {
            Some(list) => {
                if list.modules.len() != children.len() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "restore_requires_grad: snapshot has {} child module(s) but the \
                         container now has {} (structure changed during rollback)",
                        children.len(),
                        list.modules.len()
                    )));
                }
                // `?` による即時伝播は使わない（P1 是正・codex-review 指摘
                // `PRRT_kwDOTuUCJc6lFh9N`）。最初の子の復元が `Err` を返した
                // 時点で打ち切ると、残りの兄弟（1 度も `restore_requires_grad`
                // が呼ばれない）が部分適用状態のまま放置され、fail-closed
                // 契約（失敗した子自身も含め全子を元へ戻す）を破る。全子を
                // 必ず 1 回ずつ処理してから、集約したエラーの有無で結果を返す。
                collect_restore_errors(list.modules.iter_mut().zip(children.iter()))
            }
            None => Err(AutodiffError::InvalidArgument(
                "restore_requires_grad: snapshot is Nested but the module is no longer a \
                 ModuleList/Sequential container (structure changed during rollback)"
                    .to_string(),
            )),
        },
        RequiresGradSnapshot::NestedDict(children) => match module.as_module_dict_mut() {
            Some(dict) => {
                if dict.modules.len() != children.len() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "restore_requires_grad: snapshot has {} child module(s) but the \
                         ModuleDict now has {} (structure changed during rollback)",
                        children.len(),
                        dict.modules.len()
                    )));
                }
                // 同上（`Nested` 分岐のコメント参照）。`ModuleDict` 側も同じ
                // fail-closed 契約を守るため全子を処理してからエラーを集約する。
                collect_restore_errors(dict.modules.iter_mut().map(|(_, m)| m).zip(children.iter()))
            }
            None => Err(AutodiffError::InvalidArgument(
                "restore_requires_grad: snapshot is NestedDict but the module is no longer a \
                 ModuleDict container (structure changed during rollback)"
                    .to_string(),
            )),
        },
    }
}

/// `restore_requires_grad` の `Nested`／`NestedDict` 分岐が共有する集約
/// ロジック（P1 是正・codex-review 指摘 `PRRT_kwDOTuUCJc6lFh9N`）: 子を
/// 1 体ずつ `restore_requires_grad` へ委譲し、`Err` が出ても即座に伝播
/// せず**必ず残り全ての子も処理してから**、失敗した子すべてを集約した
/// 単一の `InvalidArgument` を返す（fail-closed。個々の子の復元失敗が
/// 兄弟の復元機会を奪わないことを保証する。集約メッセージの形式は
/// `ModuleList::set_requires_grad`／`ModuleDict::set_requires_grad` の
/// ロールバック失敗集約〈`"module {index} ({err})"` を `", "` で連結〉と
/// 揃える）。
fn collect_restore_errors<'a, I>(children: I) -> Result<(), AutodiffError>
where
    I: Iterator<Item = (&'a mut Box<dyn Module>, &'a RequiresGradSnapshot)>,
{
    let mut failures: Vec<String> = Vec::new();
    for (index, (m, s)) in children.enumerate() {
        if let Err(err) = restore_requires_grad(m.as_mut(), s) {
            failures.push(format!("child {index} ({err})"));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(AutodiffError::InvalidArgument(format!(
            "restore_requires_grad: failed to restore {} of the child module(s): {}",
            failures.len(),
            failures.join(", ")
        )))
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

    /// `self.inner`（`ModuleList`）を返す（P1 是正・#2234 レビュー指摘。
    /// `ModuleList::as_module_list` doc 参照）。`Sequential` は `inner`
    /// の薄いラッパーであり、混在状態のスナップショット・ロールバック
    /// 対象は `inner` 側に存在するため、こちらへそのまま委譲する。
    fn as_module_list(&self) -> Option<&ModuleList> {
        Some(&self.inner)
    }

    fn as_module_list_mut(&mut self) -> Option<&mut ModuleList> {
        Some(&mut self.inner)
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

    /// [`Module::set_requires_grad`] の実装（P1 是正・#2234 レビュー
    /// 指摘 `PRRT_kwDOTuUCJc6lE8Gg`／cursor\[bot\] `PRRT_kwDOTuUCJc6lE80z`。
    /// イシュー #2137）。`ModuleList::set_requires_grad` と同じ伝播・
    /// 集約・ロールバック処理（`snapshot_requires_grad`／
    /// `restore_requires_grad`）を再利用する: 挿入順に全子 `Module` へ
    /// 伝播するベストエフォート・ロールバック方式で、子の 1 つが `Err`
    /// を返した場合はそれより前に適用済みの子 **と失敗した子自身**
    /// （`0..=index`。`ModuleList::set_requires_grad` の同ロールバック
    /// 範囲是正〈review thread `PRRT_kwDOTuUCJc6lE8Gn`〉と同じ理由）を
    /// 逆順に元の状態へ戻す。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        // 適用前の状態を層順（挿入順）に再帰的スナップショットとして
        // 記録する（ロールバック用）。
        let previous: Vec<RequiresGradSnapshot> = self
            .modules
            .iter()
            .map(|(_, m)| snapshot_requires_grad(m.as_ref()))
            .collect();

        for (index, (_, module)) in self.modules.iter_mut().enumerate() {
            if let Err(err) = module.set_requires_grad(requires_grad) {
                // 適用済みの子（`0..index`）に加え、失敗した子自身
                // （`index`）も逆順に元の状態へ戻す
                // （`ModuleList::set_requires_grad` と同じ理由）。
                //
                // 1 子のロールバックが失敗しても打ち切らず、残りの子の
                // ロールバックは続行する（`ModuleList::set_requires_grad`
                // と同じ是正。理由も同じ）。
                let mut rollback_failures: Vec<String> = Vec::new();
                for rollback_index in (0..=index).rev() {
                    let Some((_, rollback_module)) = self.modules.get_mut(rollback_index) else {
                        continue;
                    };
                    if let Err(rollback_err) =
                        restore_requires_grad(rollback_module.as_mut(), &previous[rollback_index])
                    {
                        rollback_failures.push(format!("module {rollback_index} ({rollback_err})"));
                    }
                }
                if !rollback_failures.is_empty() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "ModuleDict::set_requires_grad: failed to apply to module {index} \
                         ({err}), and rollback also failed for: {}; the ModuleDict may now be \
                         left in a partially applied state",
                        rollback_failures.join(", ")
                    )));
                }
                return Err(err);
            }
        }
        Ok(())
    }

    /// パラメータを持つ子が 1 つでも `true` を返せば `true`
    /// （[`ModuleList::requires_grad`] と同じ複合層契約・同じ理由の
    /// `parameter_count() == 0` 除外〈パラメータを持たない子は既定
    /// `true` に引きずられて誤判定するのを防ぐ〉。子が 1 つも無い、
    /// またはどの子もパラメータを持たない場合は `true`（`ModuleList::
    /// requires_grad` の doc「無状態層は既定 `true`」契約と同じ）。
    fn requires_grad(&self) -> bool {
        let mut saw_param_bearing_child = false;
        for (_, module) in &self.modules {
            if module.parameter_count() == 0 {
                continue;
            }
            saw_param_bearing_child = true;
            if module.requires_grad() {
                return true;
            }
        }
        !saw_param_bearing_child
    }

    fn as_module_dict(&self) -> Option<&ModuleDict> {
        Some(self)
    }

    fn as_module_dict_mut(&mut self) -> Option<&mut ModuleDict> {
        Some(self)
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
/// **ZST をグローバル dedup から除外する理由・型名を同一性キーに
/// 含める理由（イシュー #2134 codex-review／Bugbot 指摘・PR #2231
/// 是正）**: データポインタ単独のグローバル訪問済み集合を無条件適用
/// すると、ZST（`Relu`・`Gelu` 等）を `Box` へ格納した際に複数
/// インスタンスがアロケータの well-known dangling address を共有
/// しうるため、`Sequential` に同種の ZST 活性化層を複数積んだ場合に
/// 後続レイヤーを「既出」と誤判定して出力から欠落させる。さらに、
/// データポインタ単独では `MultiheadAttention { q_proj: Linear,
/// ... }` のような複合 `Module` で、ルート自身と最初の子フィールド
/// （先頭に配置されうる）のデータポインタが異なる型でありながら
/// 数値としては一致しうるため、祖先限定の循環検出単独でも最初の子を
/// 「ルート自身の既出」と誤判定して子孫ごと欠落させる。本関数は
/// [`Module::named_modules`]（`collect_named_modules`）と同じ
/// `(データポインタ, 型名)` の組（`crate::nn::module::NodeKey`）を
/// 同一性キーとして使い、ZST は祖先限定の循環検出のみで保護し
/// グローバル dedup の対象からは外す（詳細は [`Module::named_modules`]
/// の「循環・重複ノードの扱い」節参照）。
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

    // `ModuleList::requires_grad`／`set_requires_grad`（イシュー #2137・
    // PR #2234 レビュー是正）の単体テスト。

    /// [`Module::named_parameters`] を持つが [`Module::set_requires_grad`]
    /// をオーバーライドしない「半分だけ実装した」`Module`
    /// （`module.rs` の `HalfImplementedModule` と同型・同意図。ロール
    /// バック経路を決定的に踏むための失敗注入用。既定実装が常に
    /// `Err(AutodiffError::InvalidArgument)` を返す）。
    struct FailingSetRequiresGradModule {
        param: Tensor<f32>,
    }

    impl Module for FailingSetRequiresGradModule {
        fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
            unreachable!("本テストでは forward は呼ばれない")
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("param".to_string(), &self.param)]
        }

        // `set_requires_grad` は意図的にオーバーライドしない（既定実装の
        // まま。常に `AutodiffError::InvalidArgument` を返す）。
    }

    /// `ModuleList::set_requires_grad`／`ModuleDict::set_requires_grad`
    /// ロールバック範囲の回帰テスト用モジュール（P1 是正・#2234
    /// レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gn`／cursor\[bot\]
    /// `PRRT_kwDOTuUCJc6lE80z` の「複数内部パラメータを順に変更する
    /// カスタム複合層で部分適用状態が残る」再現）。`set_requires_grad`
    /// は要求された値へ**先に自身の内部状態を書き換えてから** `Err`
    /// を返す。ロールバックが失敗した子自身（`index`）を含めて
    /// 復元しなければ、この部分適用状態（`state` だけが書き変わった
    /// 中途半端な状態）が残ってしまう。
    struct PartiallyMutatingFailingModule {
        param: Tensor<f32>,
        state: bool,
    }

    impl Module for PartiallyMutatingFailingModule {
        fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
            unreachable!("本テストでは forward は呼ばれない")
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("param".to_string(), &self.param)]
        }

        fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
            // 複数内部パラメータを順に変更してから失敗する複合層を
            // 模擬: 先に自身の状態を書き換えてから `Err` を返す。
            self.state = requires_grad;
            Err(AutodiffError::InvalidArgument(
                "PartiallyMutatingFailingModule: 意図的に失敗する".to_string(),
            ))
        }

        fn requires_grad(&self) -> bool {
            self.state
        }
    }

    /// `restore_requires_grad` の `Leaf` 分岐の false positive 回帰
    /// テスト用モジュール（P1 是正・codex-review 指摘
    /// `PRRT_kwDOTuUCJc6lFaVh`）: `as_module_list`／`as_module_dict` を
    /// 実装せず `Leaf` 扱いされる不透明な「複合 Module」を模擬する。
    /// 内部に独立した 2 つのフラグ（`a`／`b`）を持ち、公開する
    /// `requires_grad()` は「どちらかが `true` なら `true`」という
    /// 集約 bool のみ（`ModuleList::requires_grad` の `any` 契約と同種）。
    /// 順方向（`false` への設定）は両フラグを正しく更新して成功するが、
    /// 復元方向（`true` への設定＝ロールバック経路）は `a` だけ書き換え
    /// `b` を放置したまま `Err` を返す——`a || b` が偶然 desired value
    /// （`true`）と一致してしまうため、旧実装の「呼び出し後の集約
    /// `requires_grad()` が一致するか」判定では `b` が未復元のまま
    /// 「復元成功」と誤判定されていた（false positive）。
    struct AggregateLeafPartialRestoreFailure {
        param: Tensor<f32>,
        a: bool,
        b: bool,
    }

    impl Module for AggregateLeafPartialRestoreFailure {
        fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
            unreachable!("本テストでは forward は呼ばれない")
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("param".to_string(), &self.param)]
        }

        fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
            if requires_grad {
                // 復元呼び出し（ロールバックで `true` へ戻す）を模擬:
                // `a` だけ書き換えて `b` を放置したまま失敗する。
                self.a = true;
                Err(AutodiffError::InvalidArgument(
                    "AggregateLeafPartialRestoreFailure: 復元時は意図的に失敗する".to_string(),
                ))
            } else {
                // 順方向の適用（`false` への設定）は両フラグとも正しく
                // 更新して成功する。
                self.a = false;
                self.b = false;
                Ok(())
            }
        }

        fn requires_grad(&self) -> bool {
            self.a || self.b
        }
    }

    /// 外部実装 `Module` の回帰テスト用モジュール（`advisor` レビュー
    /// 指摘: 「`set_requires_grad` を正しくオーバーライドしつつ
    /// `requires_grad()` は既定 `true` のまま公開する」実装も `Module`
    /// が外部実装可能な trait である以上ありうる）。`restore_requires_grad`
    /// の `Leaf` 分岐が「呼び出し前にすでに一致しているか」を見て
    /// `set_requires_grad` の呼び出し自体を省略する実装だと、この
    /// モジュールは `requires_grad()` が常に `true` を返すため
    /// スナップショットと常に「一致している」ように見え、実際には
    /// ロールバックが必要でも `set_requires_grad` が一度も呼ばれず
    /// 凍結状態のまま放置されてしまう（是正前の不具合の再現）。
    /// `requires_grad()` の戻り値だけでは内部状態を観測できないため、
    /// `Rc<Cell<bool>>` を共有し、`set_requires_grad` が実際に呼ばれた
    /// かどうかをテスト側から独立に検証する。
    struct ExternalModuleReportingDefaultRequiresGrad {
        param: Tensor<f32>,
        state: std::rc::Rc<std::cell::Cell<bool>>,
    }

    impl Module for ExternalModuleReportingDefaultRequiresGrad {
        fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
            unreachable!("本テストでは forward は呼ばれない")
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("param".to_string(), &self.param)]
        }

        fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
            self.state.set(requires_grad);
            Ok(())
        }

        // `requires_grad()` は意図的にオーバーライドしない（既定 `true`
        // のまま）。
    }

    /// 子が 1 つでも `true` を返せば `true`（[`Module::requires_grad`]
    /// の公開契約どおり。P1 是正・#2234 レビュー指摘: 旧実装は `all`
    /// だったため一部凍結・一部解凍の `ModuleList` が誤って `false` を
    /// 返していた）。
    #[test]
    fn module_list_requires_grad_is_any_not_all() {
        let mut list = ModuleList::new();
        list.push(Box::new(Linear::new(2, 2, true, 21).unwrap()));
        list.push(Box::new(Linear::new(2, 2, true, 22).unwrap()));

        // 初期状態は両方 `true`。
        assert!(list.requires_grad());

        // 片方だけ凍結（混在状態）。「子が 1 つでも `true`」なので
        // 全体は引き続き `true` のはず（旧 `all` 実装なら誤って
        // `false` になる）。
        list.get_mut(0).unwrap().set_requires_grad(false).unwrap();
        assert!(
            list.requires_grad(),
            "一部解凍の ModuleList は any 契約で true を返すはず"
        );

        // 両方凍結すれば `false`。
        list.get_mut(1).unwrap().set_requires_grad(false).unwrap();
        assert!(!list.requires_grad());
    }

    /// 空 `ModuleList` は `true`（素の `any` の空列既定は `false` だが、
    /// [`Module::requires_grad`] の「パラメータを持たない層は既定
    /// `true` のまま」契約に合わせて `true` を返す。`ModuleList::
    /// requires_grad` doc 参照）。
    #[test]
    fn module_list_requires_grad_is_true_for_empty_list() {
        let list = ModuleList::new();
        assert!(list.requires_grad());
    }

    /// P1 是正・#2234 レビュー指摘: 入れ子 `ModuleList` が混在状態
    /// （一部凍結・一部解凍）を持つ状態で、外側の `set_requires_grad`
    /// 呼び出しが途中失敗した場合、ロールバックは混在状態を保ったまま
    /// 各子ごとに元の値へ復元しなければならない（集約 bool 1 つでの
    /// 復元は入れ子の孫全員へ同一値を強制し混在状態を破壊してしまう
    /// ——旧実装の fail-closed 契約違反）。
    #[test]
    fn module_list_set_requires_grad_rollback_preserves_nested_mixed_state() {
        // 子 0: 入れ子 ModuleList（孫 0 は解凍のまま・孫 1 は事前に
        // 凍結済み——混在状態）。
        let mut nested = ModuleList::new();
        nested.push(Box::new(Linear::new(2, 2, true, 31).unwrap()));
        nested.push(Box::new(Linear::new(2, 2, true, 32).unwrap()));
        nested.get_mut(1).unwrap().set_requires_grad(false).unwrap();

        let mut outer = ModuleList::new();
        outer.push(Box::new(nested));
        // 子 1: 必ず失敗する Module（`set_requires_grad(true)` 適用時に
        // 子 0〈入れ子 ModuleList〉はすでに適用済みという状況を作る）。
        outer.push(Box::new(FailingSetRequiresGradModule {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
        }));

        let err = outer
            .set_requires_grad(true)
            .expect_err("子 1 の既定 set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        // ロールバック後、子 0（入れ子 ModuleList）の孫別の混在状態が
        // 元どおり復元されているはず（孫 0 = 解凍・孫 1 = 凍結のまま）。
        let restored_nested = outer
            .get(0)
            .unwrap()
            .as_module_list()
            .expect("子 0 は ModuleList のはず");
        assert!(
            restored_nested
                .get(0)
                .unwrap()
                .as_linear()
                .unwrap()
                .requires_grad(),
            "孫 0 はロールバック前から解凍のままだったはず"
        );
        assert!(
            !restored_nested
                .get(1)
                .unwrap()
                .as_linear()
                .unwrap()
                .requires_grad(),
            "孫 1 はロールバック前から凍結済みだったはず（集約 bool 1 つでの \
             復元だと解凍されてしまう不具合の再現テスト）"
        );
    }

    /// Cursor Bugbot 指摘の再現・是正確認テスト（PR #2234 review thread
    /// `PRRT_kwDOTuUCJc6lEbAQ`）: `Linear → Relu → Linear` 構成で両
    /// `Linear` を `freeze()` すると、パラメータを持たない `Relu`
    /// （`requires_grad()` の既定 `true`）に引きずられず、全体の
    /// `requires_grad()` が `false`（凍結済み）を正しく報告すること。
    #[test]
    fn module_list_requires_grad_ignores_stateless_layers_when_frozen() {
        let mut list = ModuleList::new();
        list.push(Box::new(Linear::new(2, 2, true, 41).unwrap()));
        list.push(Box::new(Relu));
        list.push(Box::new(Linear::new(2, 2, true, 42).unwrap()));

        // 初期状態: 両 Linear とも解凍済みのため `true`。
        assert!(list.requires_grad());

        // 両 Linear を凍結。無状態の Relu（既定 `true`）が挟まっていても
        // 全パラメータ leaf が凍結済みなら全体は `false` を返すはず。
        list.get_mut(0).unwrap().set_requires_grad(false).unwrap();
        list.get_mut(2).unwrap().set_requires_grad(false).unwrap();
        assert!(
            !list.requires_grad(),
            "パラメータを持たない Relu の既定 true に引きずられて \
             trainable と誤報告してはいけない"
        );

        // 片方だけ解凍すれば `true`（混在状態は any 契約どおり）。
        list.get_mut(0).unwrap().set_requires_grad(true).unwrap();
        assert!(list.requires_grad());
    }

    /// 無状態層のみ（パラメータを持つ子が 1 つも無い）の `ModuleList`
    /// は、空 `ModuleList` と同じ契約で `true` を返すこと。
    #[test]
    fn module_list_requires_grad_is_true_when_no_child_has_parameters() {
        let mut list = ModuleList::new();
        list.push(Box::new(Relu));
        list.push(Box::new(Relu));
        assert!(list.requires_grad());
    }

    /// P1 是正回帰テスト（#2234 レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gn`）:
    /// ロールバック範囲は `0..index` ではなく `0..=index`（失敗した子
    /// 自身を含む）でなければならない。`PartiallyMutatingFailingModule`
    /// は `set_requires_grad` 内で先に自身の状態を書き換えてから失敗
    /// するため、`0..index` のみのロールバックだと失敗した子自身
    /// （`index`）の部分適用状態が残ってしまう（是正前の実装で失敗
    /// する再現テスト）。
    #[test]
    fn module_list_set_requires_grad_rollback_restores_failed_child_itself() {
        let mut list = ModuleList::new();
        list.push(Box::new(Linear::new(2, 2, true, 51).unwrap()));
        list.push(Box::new(PartiallyMutatingFailingModule {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
            state: true,
        }));

        let err = list
            .set_requires_grad(false)
            .expect_err("子 1 の set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        assert!(
            list.get(0).unwrap().requires_grad(),
            "適用済みの子 0（Linear）はロールバックで元の true に \
             戻っているはず"
        );
        assert!(
            list.get(1).unwrap().requires_grad(),
            "失敗した子自身（index）の部分適用状態がロールバックされて \
             いない（0..index のみのロールバックだと index 自身が \
             false のまま残る不具合の再現テスト）"
        );
    }

    /// P1 是正回帰テスト（codex-review 指摘 `PRRT_kwDOTuUCJc6lFaVh`）:
    /// `restore_requires_grad` の `Leaf` 分岐は「呼び出し後の集約
    /// `requires_grad()` が desired value と一致するか」を復元成功の
    /// 根拠にしてはならない。`AggregateLeafPartialRestoreFailure`
    /// （`a || b` の集約のみを公開する複合 Leaf）で、復元呼び出しが
    /// `a` だけ書き換えて `b` を放置したまま `Err` を返しても、`a || b`
    /// が偶然 desired value と一致するため、旧実装は「復元成功」と
    /// 誤判定していた（`b` は実際には復元されていない）。是正後は
    /// `Err` をそのまま伝播し、外側のロールバック失敗として報告
    /// されなければならない。
    #[test]
    fn module_list_set_requires_grad_rollback_leaf_with_aggregate_requires_grad_reports_partial_restore_failure()
     {
        let mut list = ModuleList::new();
        list.push(Box::new(AggregateLeafPartialRestoreFailure {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
            a: true,
            b: true,
        }));
        list.push(Box::new(FailingSetRequiresGradModule {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
        }));

        let err = list
            .set_requires_grad(false)
            .expect_err("子 1 の既定 set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let message = err.to_string();
        assert!(
            message.contains("rollback also failed for"),
            "子 0（AggregateLeafPartialRestoreFailure）の復元呼び出し \
             自体が `Err` を返した以上、集約 `requires_grad()`（`a || \
             b`）がたまたま desired value と一致していても復元失敗と \
             して報告されなければならない（旧実装は集約値の一致だけで \
             復元成功と誤判定していた false positive の再現・是正 \
             確認）: {message}"
        );
    }

    /// `advisor` レビュー指摘の回帰テスト: `restore_requires_grad` の
    /// `Leaf` 分岐が「呼び出し前にすでに `requires_grad()` の戻り値が
    /// スナップショット値と一致しているか」を見て `set_requires_grad`
    /// の呼び出し自体を省略する実装だと、`requires_grad()` を
    /// オーバーライドせず既定 `true` を返し続ける外部実装
    /// （`set_requires_grad` 自体は正しく内部状態を更新する）の子が
    /// ロールバックから漏れる。`Rc<Cell<bool>>` で `set_requires_grad`
    /// が実際に呼ばれたかどうかを trait の戻り値と独立に検証する。
    #[test]
    fn module_list_set_requires_grad_rollback_calls_set_requires_grad_even_when_trait_requires_grad_looks_unchanged()
     {
        let state = std::rc::Rc::new(std::cell::Cell::new(true));
        let mut list = ModuleList::new();
        list.push(Box::new(ExternalModuleReportingDefaultRequiresGrad {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
            state: state.clone(),
        }));
        list.push(Box::new(FailingSetRequiresGradModule {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
        }));

        let err = list
            .set_requires_grad(false)
            .expect_err("子 1 の set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        // 子 0 の set_requires_grad は成功して state=false になった
        // はずなので、ロールバックが実際に子 0 の set_requires_grad(true)
        // を呼び直していなければ state は false のまま残る。
        assert!(
            state.get(),
            "requires_grad() が既定 true のまま変化しない外部実装でも、 \
             ロールバックは set_requires_grad を実際に呼び直して \
             内部状態を復元しなければならない（呼び出し前の \
             `requires_grad() == value` チェックだけで早期 Ok にする \
             実装だと見逃す）"
        );
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

    // `ModuleDict::set_requires_grad`／`requires_grad`（P1 是正・#2234
    // レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gg`／cursor\[bot\]
    // `PRRT_kwDOTuUCJc6lE80z`: `ModuleList`／`Sequential` には実装した
    // 凍結操作が同じ公開 `Module` である `ModuleDict` に未実装で
    // `freeze()` が `InvalidArgument` を返していた不具合の是正確認）。

    /// `ModuleList::set_requires_grad`（`ModuleList::requires_grad_is_any_not_all`
    /// と対）と同じ挙動: 挿入順に全子へ伝播し、`requires_grad()` は
    /// パラメータを持つ子の any 集約。
    #[test]
    fn module_dict_set_requires_grad_propagates_and_requires_grad_is_any() {
        let mut dict = ModuleDict::new();
        dict.insert("l1", Box::new(Linear::new(2, 2, true, 61).unwrap()))
            .unwrap();
        dict.insert("l2", Box::new(Linear::new(2, 2, true, 62).unwrap()))
            .unwrap();

        // 初期状態は両方 `true`。
        assert!(dict.requires_grad());

        // 片方だけ凍結（混在状態）。any 契約なので全体は引き続き
        // `true` のはず。
        dict.get_mut("l1")
            .unwrap()
            .set_requires_grad(false)
            .unwrap();
        assert!(
            dict.requires_grad(),
            "一部解凍の ModuleDict は any 契約で true を返すはず"
        );

        // 両方凍結すれば `false`。
        dict.get_mut("l2")
            .unwrap()
            .set_requires_grad(false)
            .unwrap();
        assert!(!dict.requires_grad());

        // `ModuleDict::set_requires_grad` 自体も挿入順に全子へ伝播する
        // ことを確認する（`freeze()` が `InvalidArgument` を返していた
        // 是正前の不具合が解消されたことの直接確認）。
        dict.set_requires_grad(true).unwrap();
        assert!(dict.get("l1").unwrap().requires_grad());
        assert!(dict.get("l2").unwrap().requires_grad());

        dict.freeze().unwrap();
        assert!(!dict.get("l1").unwrap().requires_grad());
        assert!(!dict.get("l2").unwrap().requires_grad());
    }

    /// 空 `ModuleDict`／無状態層のみの `ModuleDict` は `true`
    /// （`ModuleList::requires_grad` と同じ「パラメータを持たない層は
    /// 既定 `true`」契約）。
    #[test]
    fn module_dict_requires_grad_is_true_for_empty_or_stateless_only() {
        let empty = ModuleDict::new();
        assert!(empty.requires_grad());

        let mut stateless = ModuleDict::new();
        stateless.insert("r1", Box::new(Relu)).unwrap();
        stateless.insert("r2", Box::new(Relu)).unwrap();
        assert!(stateless.requires_grad());
    }

    /// P1 是正回帰テスト（#2234 レビュー指摘 `PRRT_kwDOTuUCJc6lE8Gg`。
    /// `ModuleList::set_requires_grad_rollback_restores_failed_child_itself`
    /// と対）: `ModuleDict::set_requires_grad` のロールバックも
    /// 失敗した子自身（キー）を含めて復元しなければならない。
    #[test]
    fn module_dict_set_requires_grad_rollback_restores_failed_child_itself() {
        let mut dict = ModuleDict::new();
        dict.insert("l1", Box::new(Linear::new(2, 2, true, 71).unwrap()))
            .unwrap();
        dict.insert(
            "bad",
            Box::new(PartiallyMutatingFailingModule {
                param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
                state: true,
            }),
        )
        .unwrap();

        let err = dict
            .set_requires_grad(false)
            .expect_err("`bad` の set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        assert!(
            dict.get("l1").unwrap().requires_grad(),
            "適用済みの子（l1）はロールバックで元の true に戻っている \
             はず"
        );
        assert!(
            dict.get("bad").unwrap().requires_grad(),
            "失敗した子自身（bad）の部分適用状態がロールバックされて \
             いない"
        );
    }

    /// P1 是正回帰テスト（cursor\[bot\] 指摘 `PRRT_kwDOTuUCJc6lE80z`:
    /// 「`as_module_list` が dict を leaf 扱いしロールバックで混在状態を
    /// 保持できない」の再現・是正確認）: 親 `ModuleList` に入れ子の
    /// `ModuleDict`（混在状態）を持たせ、外側の `set_requires_grad` が
    /// 途中失敗した場合でも、`ModuleDict` 側の混在状態（キーごとの
    /// 個別 `requires_grad`）を保ったままロールバックできること。
    #[test]
    fn module_list_set_requires_grad_rollback_preserves_nested_module_dict_mixed_state() {
        // 子 0: 入れ子 ModuleDict（キー a は解凍のまま・キー b は事前に
        // 凍結済み——混在状態）。
        let mut nested = ModuleDict::new();
        nested
            .insert("a", Box::new(Linear::new(2, 2, true, 81).unwrap()))
            .unwrap();
        nested
            .insert("b", Box::new(Linear::new(2, 2, true, 82).unwrap()))
            .unwrap();
        nested
            .get_mut("b")
            .unwrap()
            .set_requires_grad(false)
            .unwrap();

        let mut outer = ModuleList::new();
        outer.push(Box::new(nested));
        // 子 1: 必ず失敗する Module（子 0〈入れ子 ModuleDict〉が適用済み
        // という状況を作る）。
        outer.push(Box::new(FailingSetRequiresGradModule {
            param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
        }));

        let err = outer
            .set_requires_grad(true)
            .expect_err("子 1 の既定 set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let restored_nested = outer
            .get(0)
            .unwrap()
            .as_module_dict()
            .expect("子 0 は ModuleDict のはず（as_module_list では None のはず）");
        assert!(
            restored_nested.get("a").unwrap().requires_grad(),
            "キー a はロールバック前から解凍のままだったはず"
        );
        assert!(
            !restored_nested.get("b").unwrap().requires_grad(),
            "キー b はロールバック前から凍結済みだったはず（dict を \
             leaf 扱いすると孫の混在状態が破壊される不具合の再現 \
             テスト）"
        );
        // `as_module_list` 経由では入れ子 ModuleDict を検出できない
        // （別コンテナ型のため）ことも合わせて確認する。
        assert!(outer.get(0).unwrap().as_module_list().is_none());
    }

    /// 上記の逆方向（cursor\[bot\] 指摘のネスト方向カバレッジ）: 親
    /// `ModuleDict` に入れ子の `ModuleList`（混在状態）を持たせた場合も
    /// 同様にロールバックが混在状態を保つこと。
    #[test]
    fn module_dict_set_requires_grad_rollback_preserves_nested_module_list_mixed_state() {
        let mut nested = ModuleList::new();
        nested.push(Box::new(Linear::new(2, 2, true, 91).unwrap()));
        nested.push(Box::new(Linear::new(2, 2, true, 92).unwrap()));
        nested.get_mut(1).unwrap().set_requires_grad(false).unwrap();

        let mut outer = ModuleDict::new();
        outer.insert("nested", Box::new(nested)).unwrap();
        outer
            .insert(
                "bad",
                Box::new(FailingSetRequiresGradModule {
                    param: Tensor::new(vec![1.0f32], &[1]).unwrap(),
                }),
            )
            .unwrap();

        let err = outer
            .set_requires_grad(true)
            .expect_err("`bad` の既定 set_requires_grad 失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let restored_nested = outer
            .get("nested")
            .unwrap()
            .as_module_list()
            .expect("`nested` は ModuleList のはず");
        assert!(restored_nested.get(0).unwrap().requires_grad());
        assert!(!restored_nested.get(1).unwrap().requires_grad());
    }
}
