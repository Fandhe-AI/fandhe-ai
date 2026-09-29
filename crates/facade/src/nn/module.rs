//! facade 独自の `nn::Module` trait（イシュー #2395・親 #2338）。
//!
//! 役割: facade 利用者が `Var` 演算を自由に組み合わせた独自層を、facade だけの依存で
//! 定義するための土台である（`fandhe_ai_autodiff::nn::Module::forward` は生の
//! `fandhe_ai_autodiff::Tape` を引数に取るため facade 利用者は名指しできない。
//! `docs/facade-nn-module-exposure-decision.md` §1.3）。#2338 承認事項 1（案 B）・
//! 2（required `forward` ＋ defaulted 6 件。#2400 で凍結 API 3 件・#2401 で introspection 4 件・#2400 の葉単位ロールバック用に `children_mut` を追加し 14 件）・4（`forward` 第 1 引数は
//! [`crate::TapeRef`]）に従う。
//!
//! REQ-12: 生の `Tape`・`BackendOps`・内部層型（`as_*`・`forward_host`・
//! `is_pooling` 等の内部フック）は一切載せない。open trait（sealed にしない）で
//! `dyn Module` が成り立つ。後方互換規則: defaulted メソッドの追加は非破壊、
//! required メソッドの追加は破壊的変更である。
//!
//! `named_parameters`／`set_parameter`／`state_dict`／`load_state_dict`／
//! `set_training`／`training`／`children`／`named_modules`／`parameter_count`／`type_name`
//! （後ろ 4 件は #2134 の鏡写し。イシュー #2401）と、凍結 API の `set_requires_grad`／`freeze`／
//! `requires_grad`（#2137 の鏡写し。#2400）の意味論・命名契約・fail-closed 検証は
//! `crates/autodiff/src/nn/module.rs` の同名メソッドと同一である。
//! `load_state_dict` は autodiff 側の単一実装を crate 内アダプタ（`FacadeModuleAdapter`。
//! autodiff コンテナへ積む橋渡しを兼ねる。#2397）経由で再利用し、
//! two-pass 検証・キー昇順適用・逆順ロールバックの一致を構造的に保証する
//! （コピーによるドリフトを避ける）。

use std::collections::{HashMap, HashSet};

use fandhe_ai_autodiff::nn::Module as AutodiffModule;
use fandhe_ai_autodiff::nn::RequiresGradSnapshot;

use crate::{AutodiffError, TapeRef, Tensor, Var};

/// facade 利用者が独自層を定義するための共通 forward シグネチャ。
///
/// required は [`Self::forward`] の 1 件、defaulted は 14 件
/// （[`Self::named_parameters`]・[`Self::set_parameter`]・[`Self::state_dict`]・
/// [`Self::load_state_dict`]・[`Self::set_training`]・[`Self::training`]・
/// [`Self::set_requires_grad`]・[`Self::freeze`]・[`Self::requires_grad`]・
/// [`Self::children`]・[`Self::children_mut`]・[`Self::named_modules`]・
/// [`Self::parameter_count`]・[`Self::type_name`]。`children`・`named_modules`・
/// `parameter_count`・`type_name` の 4 件は autodiff `Module`（#2134）の同名メソッドの
/// 鏡写し〈イシュー #2401〉、`children_mut` は `set_requires_grad` の葉単位ロールバック用に
/// `children` と対で実装する facade 独自のメソッド〈#2400〉）。
/// 出典は `crates/autodiff/src/nn/module.rs` で、`tests/api_surface.rs` が
/// この集合を機械的に固定する。
pub trait Module {
    /// このステップの `tape` 上で 1 回分の forward を計算する。
    ///
    /// `tape` は [`crate::TapeRef`]（`TapeRef::from(&fandhe_ai::tape())` 等）。
    /// パラメータは `tape.var(&tensor)` で葉ノード登録してから使う。
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>;

    /// 学習可能パラメータの「名前, 参照」列。名前は struct フィールド名／accessor 名
    /// を正とし、順序は登録順（weight → bias）。既定は空（無状態層向け）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        Vec::new()
    }

    /// [`Self::named_parameters`] が返す名前のパラメータを書き戻す（shape 保存置換のみ）。
    ///
    /// 該当名がなければ `AutodiffError::InvalidArgument`（fail-closed）。既定は常に `Err`
    /// （無状態層向け）。[`Self::named_parameters`] を override する層は必ず本メソッドも
    /// 対で override すること（片方だけだと [`Self::load_state_dict`] が `Err` になる）。
    fn set_parameter(&mut self, name: &str, _value: Tensor<f32>) -> Result<(), AutodiffError> {
        Err(AutodiffError::InvalidArgument(format!(
            "Module::set_parameter: no parameter named `{name}` (default fail-safe; this \
             Module does not override set_parameter)"
        )))
    }

    /// [`Self::named_parameters`] のキー付きビュー（各値は `clone`）。順序契約は持たない。
    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect()
    }

    /// [`Self::state_dict`] の逆（strict 限定）。
    ///
    /// パス 1 でキー集合の完全一致（欠落・余剰を昇順列挙）と shape 完全一致を検証し
    /// （無変更）、パス 2 でキー名昇順に [`Self::set_parameter`] を適用する。途中で
    /// 失敗したら適用済みキーを逆順に元の値へ戻す。ロールバック自体が失敗した場合は
    /// 部分適用の可能性を明示した `Err` を返す（完全な原子性は保証しない）。
    fn load_state_dict(
        &mut self,
        state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        AutodiffModule::load_state_dict(&mut FacadeModuleAdapter(self), state)
    }

    /// 学習／評価モードの切替。既定は no-op（無状態層向け）。モード依存層は
    /// [`Self::training`] と必ず両方を override して自層に保持すること。
    fn set_training(&mut self, _training: bool) {}

    /// 現在のモード。既定は `true`（PyTorch `Module.training` の初期値と同じ）。
    fn training(&self) -> bool {
        true
    }

    /// 層単位の `requires_grad` 凍結（PyTorch `module.requires_grad_(bool)` 相当。
    /// #2137 の鏡写し・#2400）。
    ///
    /// 実装者契約: パラメータを持つ層は `requires_grad: bool` を自層に保持し、`forward`
    /// 内でパラメータ葉を `if self.requires_grad { tape.var(&p) } else { tape.var_no_grad(&p) }`
    /// のように切り替えて登録する。本メソッドと [`Self::requires_grad`] は必ず対で
    /// override すること。
    ///
    /// 複合層（[`Self::children`] が非空）が自身にも `requires_grad` フラグを持つ場合、
    /// [`Self::requires_grad`] はそのフラグを返すこと。凍結が失敗したときのロールバックは
    /// 複合層ごとに「自身へ `set_requires_grad(呼び出し前の自身の `requires_grad()`)` →
    /// 子を葉単位で再帰復元」の順に呼んで戻す（自身の呼び出しが子へ伝播した値は直後の子の
    /// 復元で上書きされる）。したがって自身のフラグと各子の状態が共に呼び出し前へ戻る。
    ///
    /// - 反映は次の `forward`（葉登録）から。登録済みの `Var` は変わらない。
    /// - [`Self::set_training`]／[`Self::training`] とは独立の軸（BatchNorm 系の統計は
    ///   凍結後も training のまま更新される）。
    /// - [`Self::state_dict`]／[`Self::load_state_dict`]／[`Self::set_parameter`] の対象外で、
    ///   パラメータを差し替えてもフラグは保持される。
    /// - 粒度は層単位（パラメータ名指定の個別凍結は範囲外）。
    ///
    /// 既定は fail-closed: パラメータ 0 件なら `Ok(())`（無状態層）、1 件以上なら
    /// `InvalidArgument`。override 忘れで「凍結したつもりが学習が続く」黙った事故を
    /// 防ぐ（security.md A08。no-op 既定は採らない）。文言は autodiff 側と同一。
    fn set_requires_grad(&mut self, _requires_grad: bool) -> Result<(), AutodiffError> {
        let param_count = self.named_parameters().len();
        if param_count == 0 {
            Ok(())
        } else {
            Err(AutodiffError::InvalidArgument(format!(
                "Module::set_requires_grad: this Module has {param_count} named parameter(s) \
                 but does not override set_requires_grad (fail-closed default; freezing would \
                 silently be a no-op)"
            )))
        }
    }

    /// [`Self::set_requires_grad`]`(false)` の別名（転移学習で backbone を固定する典型呼び出し）。
    fn freeze(&mut self) -> Result<(), AutodiffError> {
        self.set_requires_grad(false)
    }

    /// この層が現在追跡対象か。既定は `true`（パラメータを持たない層は状態を持たない）。
    /// パラメータを持つ層は [`Self::set_requires_grad`] と対で override する。自身のフラグを
    /// 持つ複合層はそのフラグを返すこと（ロールバックが自身→子の順に復元するため。
    /// [`Self::set_requires_grad`] 参照）。
    fn requires_grad(&self) -> bool {
        true
    }

    /// この層が直接内包する子 `Module` の「名前, 参照」列を登録順で返す
    /// （autodiff `Module::children`・PyTorch `Module.children()` 相当。#2134・#2401）。
    ///
    /// 名前は [`Self::named_parameters`] の接頭辞と完全一致させる（`ModuleList`／
    /// `Sequential` は `"{index}"`）。既定は空（葉モジュール向け）。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        Vec::new()
    }

    /// 子孫を深さ優先（子自身 → その子孫）・登録順で再帰列挙する
    /// （autodiff `Module::named_modules` の鏡写し。#2134・#2401）。
    ///
    /// ルート自身は含めない（`Self: Sized` 境界が object safety を壊すため。PyTorch の
    /// `""` エントリとは異なる）。パスは `"{parent}.{child}"`。
    ///
    /// 循環・重複: 同一性は `(データポインタ, type_name)` の組で判定する。(1) 祖先スタックに
    /// 既出なら常に打ち切る（循環でも panic しない）、(2) ゼロサイズ型を除き訪問済み集合でも
    /// dedup する（共有子は最初の経路のみ。PyTorch の memo 相当）。ZST は複数インスタンスが
    /// dangling address を共有しうるため (2) から除外する。型名をキーに含めるのは、先頭
    /// フィールドの子がルートと同一アドレスになる（offset 0）場合の誤判定を避けるため。
    fn named_modules(&self) -> Vec<(String, &dyn Module)> {
        let mut out = Vec::new();
        let root_key: NodeKey = (self as *const Self as *const (), self.type_name());
        let mut ancestors: Vec<NodeKey> = vec![root_key];
        let mut visited: HashSet<NodeKey> = HashSet::new();
        for (name, child) in self.children() {
            collect_named_modules(name, child, &mut ancestors, &mut visited, &mut out);
        }
        out
    }

    /// [`Self::named_parameters`] が公開する学習可能パラメータの総要素数
    /// （autodiff `Module::parameter_count` の鏡写し。#2134・#2401）。
    /// `saturating_add` で合計するためオーバーフローでも panic せず `usize::MAX` に飽和する。
    fn parameter_count(&self) -> usize {
        self.named_parameters()
            .into_iter()
            .fold(0usize, |acc, (_, tensor)| {
                acc.saturating_add(tensor.numel())
            })
    }

    /// 実装型名（`std::any::type_name::<Self>()`）。出力形式は標準ライブラリが安定を
    /// 保証しないため表示用途に限る（autodiff `Module::type_name` の鏡写し。#2134・#2401）。
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// [`Self::children`] の可変版（同じ「名前, 参照」列を同じ順序・同じ名前で返す。
    /// #2400・PR #2426 レビュー是正。2026-09-29 ユーザー承認済みの公開面追加）。
    ///
    /// 用途: 凍結（[`Self::set_requires_grad`]）が失敗したとき、コンテナが子孫を**葉単位**
    /// （葉 = [`Self::children`] が空の層）で呼び出し前の状態へ戻すための可変アクセス。
    /// 子を持つ利用者定義の複合層は `children` と対で必ず実装すること。件数・名前・順序が
    /// `children` と食い違う（既定の空のまま `children` だけ実装した場合を含む）層を
    /// 含む構成は、`ModuleList`／`Sequential`／`ModuleDict` の `set_requires_grad` が状態を
    /// 変更する前に `InvalidArgument` で拒否する（fail-closed）。復元は複合層自身へ
    /// `set_requires_grad` を呼んだ後に本メソッドの子を葉単位で戻す順（[`Self::set_requires_grad`]
    /// 参照）。既定は空（葉モジュール向け）。
    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        Vec::new()
    }
}

/// `named_modules` 用のノード同一性キー（データポインタ, 型名）。非公開。
pub(super) type NodeKey = (*const (), &'static str);

/// `Module::named_modules` の再帰本体（autodiff 側 `collect_named_modules` と同一意味論）。
/// 祖先スタックで循環を常に打ち切り、非 ZST のみ訪問済み集合で共有子を dedup する。
fn collect_named_modules<'a>(
    name: String,
    child: &'a dyn Module,
    ancestors: &mut Vec<NodeKey>,
    visited: &mut HashSet<NodeKey>,
    out: &mut Vec<(String, &'a dyn Module)>,
) {
    let key: NodeKey = (child as *const dyn Module as *const (), child.type_name());
    if ancestors.contains(&key) {
        return;
    }
    let is_zst = std::mem::size_of_val(child) == 0;
    if !is_zst && !visited.insert(key) {
        return;
    }
    out.push((name.clone(), child));
    ancestors.push(key);
    for (descendant_name, descendant) in child.children() {
        collect_named_modules(
            format!("{name}.{descendant_name}"),
            descendant,
            ancestors,
            visited,
            out,
        );
    }
    ancestors.pop();
}

/// facade の [`Module`] を autodiff 側 `Module` として扱う crate 内専用アダプタ
/// （イシュー #2397・親 #2338 承認事項 4「借用ハンドル型による橋渡し」。
/// `docs/facade-nn-module-exposure-decision.md` §6・§10・§14）。
///
/// 役割は 2 つ。`P = &mut M`（`M: Module + ?Sized`）では facade `Module::load_state_dict`
/// 既定実装が autodiff 側の単一実装を再利用するための内部ブリッジ、`P = Box<dyn Module>`
/// では autodiff コンテナ（`Box<dyn fandhe_ai_autodiff::nn::Module>`。`'static` 必須）へ
/// facade 層を積むアダプタ（公開入口は #2398 で `compat::Sequential` に足す。それまで
/// `nn/mod.rs` からは再エクスポートせず、compat からは到達しない）。
///
/// `pub(crate)` で公開面（REQ-12）には出ない。`unsafe` は使わず、autodiff の
/// `&Tape` は `TapeRef::from_autodiff` の安全な借用変換で facade 層へ渡す。
///
/// 委譲: `forward`・`named_parameters`・`set_parameter`・`set_training`・`training`・
/// `set_requires_grad`・`requires_grad`（#2400）・葉単位の凍結スナップショット
/// `requires_grad_snapshot`／`restore_requires_grad_snapshot`（autodiff コンテナの
/// ロールバックで内側の複合層の混在状態を `children`／`children_mut` 経由で葉単位に保つ。PR #2426 P1）。`freeze` は autodiff 既定が
/// `set_requires_grad(false)` を経由するため委譲せず（facade 層が独自に `freeze` を
/// override していても本アダプタ経由では迂回される）、
/// `state_dict`／`load_state_dict` は autodiff 既定のまま（委譲済みの
/// `named_parameters`／`set_parameter` の上で動く。facade 層が独自に
/// `load_state_dict` を override していても本アダプタ経由では迂回される。
/// `ModuleList` が子を扱うのと同じ意味論）。`as_*`／`is_pooling` は既定（`None`／`false`）。
/// `type_name` も委譲する（autodiff 側の表示に利用者の層の型名を出すため。#2401）。
/// `children` は委譲しない（facade の `&dyn Module` を借用のまま autodiff の
/// `&dyn Module` へ変換できず、autodiff の `named_modules` はアダプタ内へ降りない）。
///
/// ホスト推論経路は非対応: `supports_forward_host` を `false` へ override し
/// （autodiff 既定は `true`。既定のままだと `compat::Sequential::predict` の事前判定を
/// 通過し、途中層の `Err` で手前層の副作用〈Dropout の RNG 消費・BatchNorm の running
/// stats 更新〉が tape 経路再実行と二重化する）、`forward_host` は
/// `InvalidArgument` を返す（`BackendError::Unsupported` は「フォールバックの合図」で
/// `predict_recorded` が捕捉して再実行するため使わない。前例は `ModuleList::forward_host`。
/// `Embedding` は `Unsupported` と `supports_forward_host() == false` の組だが、本アダプタは
/// 直接呼び出しやネスト経由でも黙ってフォールバックさせない方針を優先する。security.md A04）。
pub(crate) struct FacadeModuleAdapter<P>(pub(crate) P);

impl<P> AutodiffModule for FacadeModuleAdapter<P>
where
    P: std::ops::DerefMut,
    P::Target: Module,
{
    fn forward<'t>(
        &self,
        tape: &'t fandhe_ai_autodiff::Tape,
        input: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        Module::forward(&*self.0, TapeRef::from_autodiff(tape), input)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        Module::named_parameters(&*self.0)
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        Module::set_parameter(&mut *self.0, name, value)
    }

    fn set_training(&mut self, training: bool) {
        Module::set_training(&mut *self.0, training);
    }

    fn training(&self) -> bool {
        Module::training(&*self.0)
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        Module::set_requires_grad(&mut *self.0, requires_grad)
    }

    fn requires_grad(&self) -> bool {
        Module::requires_grad(&*self.0)
    }

    /// facade 側の葉単位スナップショット（`nn::container::snapshot_requires_grad`。
    /// `children`／`children_mut` 経由）を `Opaque` に包んで返す。autodiff のコンテナが
    /// アダプタを単一の葉として集約値で保存すると、内側の複合層の混在状態が復元時に
    /// 均一化されるため（PR #2426 P1）。`children`／`children_mut` が不整合な構成は
    /// 状態変更前に `InvalidArgument` で拒否する。
    fn requires_grad_snapshot(&mut self) -> Result<RequiresGradSnapshot, AutodiffError> {
        crate::nn::container::snapshot_requires_grad(&mut *self.0)
            .map(|snap| RequiresGradSnapshot::Opaque(Box::new(snap)))
    }

    /// `Opaque` を facade 側のスナップショット型へ戻して葉単位に復元する。`Opaque` 以外・
    /// 型違いは fail-closed の `InvalidArgument`。
    fn restore_requires_grad_snapshot(
        &mut self,
        snapshot: &RequiresGradSnapshot,
    ) -> Result<(), AutodiffError> {
        let inner = match snapshot {
            RequiresGradSnapshot::Opaque(any) => {
                any.downcast_ref::<crate::nn::container::RequiresGradSnapshot>()
            }
            _ => None,
        };
        match inner {
            Some(snap) => crate::nn::container::restore_requires_grad(&mut *self.0, snap),
            None => Err(AutodiffError::InvalidArgument(
                "FacadeModuleAdapter::restore_requires_grad_snapshot: snapshot was not \
                 produced by a FacadeModuleAdapter (structure changed during rollback)"
                    .to_string(),
            )),
        }
    }

    fn type_name(&self) -> &'static str {
        Module::type_name(&*self.0)
    }

    fn supports_forward_host(&self) -> bool {
        false
    }

    fn forward_host(
        &self,
        _ops: &dyn fandhe_ai_tensor_core::BackendOps,
        _input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Err(AutodiffError::InvalidArgument(
            "FacadeModuleAdapter::forward_host: facade nn::Module does not support the host \
             (tape-free) inference path; supports_forward_host() is false and must be honored"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 パラメータ層。`fail_b` が真なら `b` の `set_parameter` は常に失敗する。
    struct TwoParam {
        a: Tensor<f32>,
        b: Tensor<f32>,
        fail_b: bool,
    }

    impl Module for TwoParam {
        fn forward<'t>(
            &self,
            tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            let a = tape.var(&self.a);
            input.add(&a)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("a".into(), &self.a), ("b".into(), &self.b)]
        }
        fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
            match name {
                "a" => self.a = value,
                "b" if self.fail_b => {
                    return Err(AutodiffError::InvalidArgument("b failed".into()));
                }
                "b" => self.b = value,
                _ => return Err(AutodiffError::InvalidArgument(format!("unknown {name}"))),
            }
            Ok(())
        }
    }

    struct Stateless;
    impl Module for Stateless {
        fn forward<'t>(
            &self,
            _tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            Ok(*input)
        }
    }

    fn t(v: f32) -> Tensor<f32> {
        Tensor::from_slice(&[v, v], &[2]).expect("tensor")
    }

    fn bits(x: &Tensor<f32>) -> Vec<u32> {
        x.as_slice()
            .expect("host tensor")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn bridge_forward_matches_direct_forward() {
        let mut layer = TwoParam {
            a: t(1.0),
            b: t(2.0),
            fail_b: false,
        };
        let ta = crate::tape();
        let xa = ta.var(&t(3.0));
        let direct = layer.forward(TapeRef::from(&ta), &xa).expect("direct");

        let tb = crate::tape();
        let xb = tb.var(&t(3.0));
        let bridge = FacadeModuleAdapter(&mut layer);
        let via = AutodiffModule::forward(&bridge, &tb.0, &xb).expect("bridge");
        assert_eq!(bits(&direct.value()), bits(&via.value()));
    }

    #[test]
    fn default_set_parameter_message_matches_autodiff() {
        let mine = Stateless
            .set_parameter("w", t(0.0))
            .expect_err("default is Err")
            .to_string();
        let mut relu = fandhe_ai_autodiff::nn::activation::Relu;
        let theirs = AutodiffModule::set_parameter(&mut relu, "w", t(0.0))
            .expect_err("default is Err")
            .to_string();
        assert_eq!(mine, theirs);
    }

    #[test]
    fn load_state_dict_errors_and_rollback_via_bridge() {
        let mut layer = TwoParam {
            a: t(1.0),
            b: t(2.0),
            fail_b: true,
        };
        let mut good = layer.state_dict();
        good.insert("a".into(), t(9.0));
        let err = layer.load_state_dict(good).expect_err("b fails");
        assert!(err.to_string().contains("b failed"), "{err}");
        assert_eq!(bits(&layer.a), bits(&t(1.0)), "a is rolled back");

        let mut missing = layer.state_dict();
        missing.remove("b");
        let e = layer.load_state_dict(missing).expect_err("missing");
        assert!(e.to_string().contains("missing keys"), "{e}");
    }

    // ---- #2397: autodiff コンテナへ積むアダプタ（P = Box<dyn Module>）の検証 ----

    use fandhe_ai_autodiff::nn::{Linear, Sequential};

    /// パラメータ `a`（`[3]`）と train/eval フラグを実保持する facade 層。
    struct Scale {
        a: Tensor<f32>,
        training: bool,
    }

    impl Scale {
        fn new() -> Self {
            Self {
                a: Tensor::from_slice(&[0.5, -1.5, 2.0], &[3]).expect("tensor"),
                training: true,
            }
        }
    }

    impl Module for Scale {
        fn forward<'t>(
            &self,
            tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            let a = tape.var(&self.a);
            input.mul(&a)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("a".into(), &self.a)]
        }
        fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
            if name != "a" {
                return Err(AutodiffError::InvalidArgument(format!("unknown {name}")));
            }
            self.a = value;
            Ok(())
        }
        fn set_training(&mut self, training: bool) {
            self.training = training;
        }
        fn training(&self) -> bool {
            self.training
        }
    }

    fn input() -> Tensor<f32> {
        Tensor::from_slice(&[0.1, 0.2, -0.3, 0.4, -0.5, 0.6], &[2, 3]).expect("tensor")
    }

    fn adapter() -> FacadeModuleAdapter<Box<dyn Module>> {
        FacadeModuleAdapter(Box::new(Scale::new()) as Box<dyn Module>)
    }

    /// forward 値・入力勾配・葉勾配・ノード数のスナップショット（bit 比較用）。
    type Snapshot = (Vec<u32>, Option<Vec<u32>>, Vec<Option<Vec<u32>>>, usize);

    fn snapshot<'t>(tape: &'t crate::Tape, x: &Var<'t>, out: &Var<'t>) -> Snapshot {
        let loss = out.sum(None).expect("sum");
        let grads = tape.backward(&loss).expect("backward");
        let leaves = (0..tape.leaf_count())
            .map(|i| {
                let v = tape.leaf(i).expect("leaf");
                grads.get(&v).expect("get").map(bits)
            })
            .collect();
        (
            bits(&out.value()),
            grads.get(x).expect("get").map(bits),
            leaves,
            tape.0.len(),
        )
    }

    #[test]
    fn adapter_in_autodiff_sequential_matches_manual_chain() {
        for linear_first in [true, false] {
            let linear = Linear::new(3, 3, true, 7).expect("linear");
            let reference = Scale::new();

            let tape = crate::tape();
            let x = tape.var(&input());
            let mut h = x;
            if linear_first {
                h = linear.bind(&tape.0).forward(&h).expect("linear");
                h = Module::forward(&reference, TapeRef::from(&tape), &h).expect("scale");
            } else {
                h = Module::forward(&reference, TapeRef::from(&tape), &h).expect("scale");
                h = linear.bind(&tape.0).forward(&h).expect("linear");
            }
            let expected = snapshot(&tape, &x, &h);

            let mut seq = Sequential::new();
            let linear2 = Linear::new(3, 3, true, 7).expect("linear");
            if linear_first {
                seq.push(Box::new(linear2));
                seq.push(Box::new(adapter()));
            } else {
                seq.push(Box::new(adapter()));
                seq.push(Box::new(linear2));
            }
            let tape2 = crate::tape();
            let x2 = tape2.var(&input());
            let out = AutodiffModule::forward(&seq, &tape2.0, &x2).expect("seq forward");
            let actual = snapshot(&tape2, &x2, &out);
            assert_eq!(expected, actual, "linear_first={linear_first}");
        }
    }

    #[test]
    fn adapter_state_dict_semantics_match_facade_layer() {
        let mut ad = adapter();
        let direct = Scale::new();
        let via = AutodiffModule::state_dict(&ad);
        let want = Module::state_dict(&direct);
        assert_eq!(via.len(), want.len());
        assert_eq!(bits(&via["a"]), bits(&want["a"]));

        let mut seq = Sequential::new();
        seq.push(Box::new(Linear::new(3, 3, false, 1).expect("linear")));
        seq.push(Box::new(adapter()));
        let mut sd = AutodiffModule::state_dict(&seq);
        let new_a = Tensor::from_slice(&[9.0, 8.0, 7.0], &[3]).expect("tensor");
        sd.insert("1.a".into(), new_a.clone());
        AutodiffModule::load_state_dict(&mut seq, sd).expect("load");
        assert_eq!(bits(&AutodiffModule::state_dict(&seq)["1.a"]), bits(&new_a));

        // 欠落キーの Err 文言は facade 層直接と同一。
        let mut direct_mut = Scale::new();
        let e_direct = Module::load_state_dict(&mut direct_mut, HashMap::new())
            .expect_err("missing")
            .to_string();
        let e_adapter = AutodiffModule::load_state_dict(&mut ad, HashMap::new())
            .expect_err("missing")
            .to_string();
        assert_eq!(e_direct, e_adapter);
    }

    #[test]
    fn adapter_propagates_training_flag() {
        let mut seq = Sequential::new();
        seq.push(Box::new(adapter()));
        assert!(AutodiffModule::training(&seq));
        AutodiffModule::set_training(&mut seq, false);
        assert!(!AutodiffModule::training(&seq));
        assert!(!seq.layers()[0].training());
        AutodiffModule::set_training(&mut seq, true);
        assert!(seq.layers()[0].training());
    }

    #[test]
    fn default_set_requires_grad_message_matches_autodiff() {
        // autodiff 側は named_parameters だけを override した同数パラメータの型で比較する。
        struct AdParams(Tensor<f32>);
        impl AutodiffModule for AdParams {
            fn forward<'t>(
                &self,
                _tape: &'t fandhe_ai_autodiff::Tape,
                input: &Var<'t>,
            ) -> Result<Var<'t>, AutodiffError> {
                Ok(*input)
            }
            fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
                vec![("p".into(), &self.0)]
            }
        }
        struct FacParams(Tensor<f32>);
        impl Module for FacParams {
            fn forward<'t>(
                &self,
                _tape: TapeRef<'t>,
                input: &Var<'t>,
            ) -> Result<Var<'t>, AutodiffError> {
                Ok(*input)
            }
            fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
                vec![("p".into(), &self.0)]
            }
        }
        let mine = FacParams(t(1.0))
            .set_requires_grad(false)
            .expect_err("fail-closed")
            .to_string();
        let theirs = AutodiffModule::set_requires_grad(&mut AdParams(t(1.0)), false)
            .expect_err("fail-closed")
            .to_string();
        assert_eq!(mine, theirs);
    }

    #[test]
    fn default_requires_grad_for_stateless_layer() {
        let mut s = Stateless;
        assert!(s.set_requires_grad(false).is_ok());
        assert!(s.freeze().is_ok());
        assert!(s.requires_grad());
        let mut p = TwoParam {
            a: t(1.0),
            b: t(2.0),
            fail_b: false,
        };
        assert!(matches!(p.freeze(), Err(AutodiffError::InvalidArgument(_))));
    }

    /// `requires_grad` を実保持する facade 層（アダプタ委譲の検証用）。
    struct Freezable {
        a: Tensor<f32>,
        rg: bool,
    }
    impl Module for Freezable {
        fn forward<'t>(
            &self,
            tape: TapeRef<'t>,
            input: &Var<'t>,
        ) -> Result<Var<'t>, AutodiffError> {
            let a = if self.rg {
                tape.var(&self.a)
            } else {
                tape.var_no_grad(&self.a)
            };
            input.mul(&a)
        }
        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("a".into(), &self.a)]
        }
        fn set_requires_grad(&mut self, v: bool) -> Result<(), AutodiffError> {
            self.rg = v;
            Ok(())
        }
        fn requires_grad(&self) -> bool {
            self.rg
        }
    }

    #[test]
    fn adapter_delegates_set_requires_grad_and_requires_grad() {
        let mut seq = Sequential::new();
        seq.push(Box::new(FacadeModuleAdapter(Box::new(Freezable {
            a: Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).expect("tensor"),
            rg: true,
        }) as Box<dyn Module>)));
        assert!(AutodiffModule::requires_grad(&seq));
        AutodiffModule::freeze(&mut seq).expect("freeze");
        assert!(!AutodiffModule::requires_grad(&seq));
        assert!(!seq.layers()[0].requires_grad());
        AutodiffModule::set_requires_grad(&mut seq, true).expect("unfreeze");
        assert!(AutodiffModule::requires_grad(&seq));

        // 未 override のパラメータ持ち facade 層は fail-closed で Err。
        let mut bad = Sequential::new();
        bad.push(Box::new(adapter()));
        assert!(AutodiffModule::freeze(&mut bad).is_err());
    }
    #[test]
    fn adapter_delegates_type_name_but_not_children() {
        let ad = adapter();
        assert_eq!(
            AutodiffModule::type_name(&ad),
            Module::type_name(&Scale::new())
        );
        assert!(AutodiffModule::children(&ad).is_empty());
        assert_eq!(
            AutodiffModule::parameter_count(&ad),
            Module::parameter_count(&Scale::new())
        );

        let mut seq = Sequential::new();
        seq.push(Box::new(adapter()));
        seq.push(Box::new(adapter()));
        let names: Vec<String> = AutodiffModule::named_modules(&seq)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn adapter_host_inference_is_fail_closed() {
        let ad = adapter();
        assert!(!AutodiffModule::supports_forward_host(&ad));
        let ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
        let e = AutodiffModule::forward_host(&ad, &ops, &input()).expect_err("must be Err");
        assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");

        let mut seq = Sequential::new();
        seq.push(Box::new(adapter()));
        let e = AutodiffModule::forward_host(&seq, &ops, &input()).expect_err("must be Err");
        assert!(matches!(e, AutodiffError::InvalidArgument(_)), "{e}");

        assert!(AutodiffModule::as_linear(&ad).is_none());
        assert!(!AutodiffModule::is_pooling(&ad));
    }
}
