//! facade 独自の `nn::Module` trait（イシュー #2395・親 #2338）。
//!
//! 役割: facade 利用者が `Var` 演算を自由に組み合わせた独自層を、facade だけの依存で
//! 定義するための土台である（`fandhe_ai_autodiff::nn::Module::forward` は生の
//! `fandhe_ai_autodiff::Tape` を引数に取るため facade 利用者は名指しできない。
//! `docs/facade-nn-module-exposure-decision.md` §1.3）。#2338 承認事項 1（案 B）・
//! 2（required `forward` ＋ defaulted 6 件）・4（`forward` 第 1 引数は
//! [`crate::TapeRef`]）に従う。
//!
//! REQ-12: 生の `Tape`・`BackendOps`・内部層型（`as_*`・`forward_host`・
//! `is_pooling` 等の内部フック）は一切載せない。open trait（sealed にしない）で
//! `dyn Module` が成り立つ。後方互換規則: defaulted メソッドの追加は非破壊、
//! required メソッドの追加は破壊的変更である。
//!
//! `named_parameters`／`set_parameter`／`state_dict`／`load_state_dict`／
//! `set_training`／`training` の意味論・命名契約・fail-closed 検証は
//! `crates/autodiff/src/nn/module.rs` の同名メソッドと同一である。
//! `load_state_dict` は autodiff 側の単一実装を crate 内ブリッジ経由で再利用し、
//! two-pass 検証・キー昇順適用・逆順ロールバックの一致を構造的に保証する
//! （コピーによるドリフトを避ける）。

use std::collections::HashMap;

use fandhe_ai_autodiff::nn::Module as AutodiffModule;

use crate::{AutodiffError, TapeRef, Tensor, Var};

/// facade 利用者が独自層を定義するための共通 forward シグネチャ。
///
/// required は [`Self::forward`] の 1 件、defaulted は 6 件
/// （[`Self::named_parameters`]・[`Self::set_parameter`]・[`Self::state_dict`]・
/// [`Self::load_state_dict`]・[`Self::set_training`]・[`Self::training`]）。
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
        AutodiffModule::load_state_dict(&mut ParamBridge(self), state)
    }

    /// 学習／評価モードの切替。既定は no-op（無状態層向け）。モード依存層は
    /// [`Self::training`] と必ず両方を override して自層に保持すること。
    fn set_training(&mut self, _training: bool) {}

    /// 現在のモード。既定は `true`（PyTorch `Module.training` の初期値と同じ）。
    fn training(&self) -> bool {
        true
    }
}

/// facade の [`Module`] を autodiff 側 `Module` として借用し、`load_state_dict` の
/// 既定実装（two-pass・ロールバック）を再利用するための crate 内専用ブリッジ。
///
/// 非公開（公開面に出ない）。`forward` は実際に委譲する（スタブにしない）。
struct ParamBridge<'a, M: ?Sized>(&'a mut M);

impl<M: Module + ?Sized> AutodiffModule for ParamBridge<'_, M> {
    fn forward<'t>(
        &self,
        tape: &'t fandhe_ai_autodiff::Tape,
        input: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.0.forward(TapeRef::from_autodiff(tape), input)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.0.named_parameters()
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.0.set_parameter(name, value)
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
        let bridge = ParamBridge(&mut layer);
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
}
