//! Dropout（`nn.Dropout` 相当。PyTorch `torch.nn.Dropout` 準拠。
//! イシュー #1603・親 #1601）。
//!
//! `Var::dropout(p, training)`（`var.rs`）の薄いラッパーだが、
//! `Linear`・活性化関数群等の**モード非依存**な既存実装と異なり、
//! [`Module::set_training`]／[`Module::training`]（イシュー #1758）を
//! 実際にオーバーライドしフィールド（`training: bool`）へ保持する
//! **最初の実装**である（`nn/module.rs` の trait doc「今後 Dropout・
//! BatchNorm 等のモード依存層を追加する際は…」を本 issue で解消する）。
//!
//! # train／eval と `predict`／`forward_host` の整合
//!
//! `fandhe_ai_facade::compat::sequential::Sequential::predict`
//! （tape 不要経路）はモードを無視しない: [`Module::forward_host`]
//! （下記実装）が `self.training` を読み、eval（`false`）なら
//! `input.clone()` の恒等写像、train（`true`）ならマスクを新規に
//! 抽選して適用する。これは PyTorch `model(x)` が呼び出し時点の
//! `model.training` に従うのと同じ意味論であり、決定的な推論が
//! 必要な場合は呼び出し側が先に `eval()` を呼ぶ（`Sequential::
//! predict` doc 参照。Keras `predict()` の「常に推論モード」意味論は
//! 採らない）。
//!
//! この設計は [`Module::forward`]（tape 経路）と
//! [`Module::forward_host`]（tape 不要経路）を同一の分岐
//! （`training`／`p` の早期リターン判定 → [`crate::grad::
//! dropout_mask`] → [`crate::grad::dropout_with_fallback`]）で実装する
//! ことで、`fandhe_ai_facade::compat::sequential::Sequential::
//! predict`（`predict_tape_free ≡ predict_via_tape` の bit 完全一致
//! 不変条件に依存する。`sequential.rs::predict` doc 参照）が train
//! モードでも成立するために必須である——`forward_host` だけが
//! モードを無視すると、後段の層が `Unsupported` を返し tape 経路へ
//! フォールバックした場合に「`forward_host` は eval 相当・`forward`
//! は train 相当」という食い違いが生じてしまう
//! （`.claude/rules/security.md` A08「判定迂回経路を作らない」規律）。
//!
//! # RNG 消費の注記（フォールバック時の二重抽選）
//!
//! `predict_tape_free_with_ops`（`sequential.rs`）が後段の層で
//! `Unsupported` を返し `predict` が旧経路（`predict_via_tape`）へ
//! 全体フォールバックした場合、`forward_host` は既に 1 回マスクを
//! 抽選した後に呼び出し元が計算結果を破棄し、`forward`（tape 経路）が
//! 改めてマスクを抽選し直す（RNG 消費が 2 回になる）。現状
//! `docs/compat-api-scope.md` §1 対象の全層が `forward_host` 対応済み
//! のため通常はこの経路に到達しない（到達した場合の出力自体は
//! 正しい——単に決定性のための RNG 消費回数が変わるだけ）。この既知の
//! 特性は是正しない（スコープ外）。

use crate::error::AutodiffError;
use crate::nn::module::Module;
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// PyTorch `torch.nn.Dropout(p=0.5)` 相当。`p`（drop 確率）と
/// `training`（PyTorch `Module.training` 相当。イシュー #1758 の
/// コンテナ伝播契約に従い `Module::set_training` で更新される）を
/// 保持する。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    /// `p`（drop 確率。`[0, 1]` の範囲かつ有限であること）を指定して
    /// 構築する。`training` は `true`（PyTorch `Module.training` の
    /// 初期値と揃える。`nn/activation.rs::Softplus::new` と同じ
    /// 「層構築の時点で早期に検査を弾く」規律のため `Result` を返す）。
    pub fn new(p: f32) -> Result<Self, AutodiffError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Dropout::new: p must be finite and in [0, 1], got {p}"
            )));
        }
        Ok(Self { p, training: true })
    }

    /// 構築済みの `p`（drop 確率）。
    pub fn p(&self) -> f32 {
        self.p
    }

    /// `self.p`／`self.training` を用いて [`Var::dropout`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.dropout(self.p, self.training)
    }
}

impl Default for Dropout {
    /// PyTorch `torch.nn.Dropout` の既定値（`p=0.5`）。`new` の検査
    /// （有限性・範囲）を通る既知の定数のため本番経路で `unwrap` する
    /// （`nn/activation.rs::Softplus` の `Default` 実装と同じ規律）。
    fn default() -> Self {
        #[allow(clippy::unwrap_used)]
        Self::new(0.5).unwrap()
    }
}

impl Module for Dropout {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Dropout::forward(self, input)
    }

    /// [`Var::dropout`]（`var.rs`）と**同一の関数列**
    /// （[`crate::grad::dropout_mask`] → [`crate::grad::
    /// dropout_with_fallback`]）を `tape` 不要経路で再現する
    /// （bit-exactness が構造的に成立する。モジュール doc「train／eval
    /// と `predict`／`forward_host` の整合」参照）。早期リターン条件
    /// （`!self.training || self.p == 0.0`）も `Var::dropout` と同一。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        let mask = crate::grad::dropout_mask(input.shape(), self.p)?;
        crate::grad::dropout_with_fallback(ops, input, &mask)
    }

    /// モジュール doc「最初の実装」節参照: `self.training` を実際に
    /// 更新する（`nn/module.rs::Module::set_training` の trait doc が
    /// モード依存層へ要求する契約）。
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_out_of_range_p() {
        assert!(Dropout::new(-0.1).is_err());
        assert!(Dropout::new(1.1).is_err());
        assert!(Dropout::new(f32::NAN).is_err());
        assert!(Dropout::new(f32::INFINITY).is_err());
    }

    #[test]
    fn new_accepts_boundary_p() {
        assert!(Dropout::new(0.0).is_ok());
        assert!(Dropout::new(1.0).is_ok());
    }

    #[test]
    fn default_has_p_half_and_training_true() {
        let d = Dropout::default();
        assert_eq!(d.p(), 0.5);
        assert!(d.training());
    }

    #[test]
    fn set_training_updates_field_and_training_accessor() {
        let mut d = Dropout::new(0.3).unwrap();
        assert!(d.training());
        d.set_training(false);
        assert!(!d.training());
        d.set_training(true);
        assert!(d.training());
    }

    #[test]
    fn forward_host_eval_mode_is_identity() {
        let d = {
            let mut d = Dropout::new(0.9).unwrap();
            d.set_training(false);
            d
        };
        let ops = crate::default_ops::NaiveOps;
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let out = d.forward_host(&ops, &input).unwrap();
        assert_eq!(out.as_slice().unwrap(), input.as_slice().unwrap());
    }

    #[test]
    fn forward_host_zero_p_is_identity_even_in_training() {
        let d = Dropout::new(0.0).unwrap();
        assert!(d.training());
        let ops = crate::default_ops::NaiveOps;
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let out = d.forward_host(&ops, &input).unwrap();
        assert_eq!(out.as_slice().unwrap(), input.as_slice().unwrap());
    }
}
