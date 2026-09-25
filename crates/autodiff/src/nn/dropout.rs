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
//! （`training`／`p` の早期リターン判定 → `crate::grad::
//! dropout_mask` → `crate::grad::dropout_with_fallback`）で実装する
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
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

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
    /// PyTorch `torch.nn.Dropout` の既定値（`p=0.5`）。`p=0.5` は
    /// `new` の検査（有限性・`[0, 1]` の範囲）を自明に満たすため、
    /// `new` を経由して `unwrap` するのではなく構造体リテラルを直接
    /// 構築する（本番経路での `unwrap`／`panic` を避ける
    /// `.claude/rules/coding-rust.md`「本番経路で `unwrap()` /
    /// `expect()` を使わない」規律）。
    fn default() -> Self {
        Self {
            p: 0.5,
            training: true,
        }
    }
}

impl Module for Dropout {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Dropout::forward(self, input)
    }

    /// [`Var::dropout`]（`var.rs`）と**同一の関数列**
    /// （`crate::grad::dropout_mask` → `crate::grad::
    /// dropout_with_fallback`）を `tape` 不要経路で再現する
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

/// PyTorch `torch.nn.Dropout2d(p=0.5)` 相当（イシュー #2161・親
/// #2131）。要素単位の [`Dropout`] と異なり、`[N, C, H, W]` 入力の
/// `(n, c)` チャネルをまとめて 0 に落とすか `1/(1-p)` 倍する
/// （`at::native::feature_dropout` 相当。チャネル内の空間的相関を
/// 考慮した正則化）。`p`／`training` の保持・検査規律は [`Dropout`] と
/// 同一。
///
/// # rank 4 限定（スコープ）
///
/// PyTorch の `Dropout2d` はバージョンにより 3 次元入力（暗黙の
/// バッチ次元なし `[C, H, W]`）の扱いが変遷し非推奨のため、本実装は
/// **rank 4（`[N, C, H, W]`）限定**とする。それ以外の rank は
/// [`AutodiffError::Shape`]（[`ShapeError::RankMismatch`]）で
/// fail-closed に拒否する（`.claude/rules/security.md` A03）。
///
/// # マスク生成と `Op::Dropout` の再利用
///
/// `crate::grad::feature_dropout_mask`（`pub(crate)` のためコード
/// スパン表記で参照しリンク化しない。`nn/embedding.rs` の
/// `ids_from_f32` 等と同じ規約）がチャネル単位で抽選したマスクを
/// `[N, C, H, W]` へ展開してから `crate::var::Var::dropout_with_mask`
/// （同様に `pub(crate)`。[`Dropout`] と共有する forward 入口）へ渡す。
/// forward（`ops.mul` 1 回）・backward
/// （`Op::Dropout` の VJP。`vjp_elementwise_mul` 1 回）とも [`Dropout`]
/// と全く同じ演算列を通るため、3 バックエンド間の bit 一致が構造的に
/// 成り立つ（`crate::grad::dropout_with_fallback` doc 参照）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dropout2d {
    p: f32,
    training: bool,
}

impl Dropout2d {
    /// [`Dropout::new`] と同じ検査（`p` は有限かつ `[0, 1]`）。
    pub fn new(p: f32) -> Result<Self, AutodiffError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Dropout2d::new: p must be finite and in [0, 1], got {p}"
            )));
        }
        Ok(Self { p, training: true })
    }

    /// 構築済みの `p`（drop 確率）。
    pub fn p(&self) -> f32 {
        self.p
    }

    /// rank 4 検査 → 早期リターン判定（`!training || p == 0.0`）→
    /// `crate::grad::feature_dropout_mask` → `crate::var::Var::
    /// dropout_with_mask`（いずれも `pub(crate)`）の順に実行する
    /// （モジュール doc 参照）。
    /// 早期リターンは新しいノードを積まず RNG も消費しない
    /// （[`Dropout::forward`] と同じ規律）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: shape.len(),
            }));
        }
        if !self.training || self.p == 0.0 {
            return Ok(*input);
        }
        let mask = crate::grad::feature_dropout_mask(&shape, self.p)?;
        input.dropout_with_mask(mask)
    }
}

impl Default for Dropout2d {
    /// PyTorch `torch.nn.Dropout2d` の既定値（`p=0.5`）。[`Dropout::
    /// default`] と同じく構造体リテラルで直接構築する（本番経路
    /// panic 禁止）。
    fn default() -> Self {
        Self {
            p: 0.5,
            training: true,
        }
    }
}

impl Module for Dropout2d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Dropout2d::forward(self, input)
    }

    /// [`Self::forward`] と**同一の関数列**（rank 検査 → 早期リターン
    /// 判定 → `crate::grad::feature_dropout_mask` → `crate::grad::
    /// dropout_with_fallback`）を `tape` 不要経路で再現する（[`Dropout::
    /// forward_host`] と同型。bit-exactness が構造的に成立する）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: shape.len(),
            }));
        }
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        let mask = crate::grad::feature_dropout_mask(shape, self.p)?;
        crate::grad::dropout_with_fallback(ops, input, &mask)
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }
}

/// PyTorch `torch.nn.AlphaDropout(p=0.5)` 相当（イシュー #2161・親
/// #2131）。SELU 系ネットワーク向けに、drop 後の出力が入力（自己
/// 正規化前提の `N(0, 1)`）の平均・分散を保つようアフィン補正する
/// （`at::native::_dropout_impl` の alpha 分岐相当）。`p`／`training`
/// の保持・検査規律は [`Dropout`] と同一で、入力の rank は任意
/// （要素単位の変換のため）。
///
/// # 数式（`crate::grad::alpha_dropout_mask_and_bias`〈`pub(crate)`。
/// コードスパン表記で参照しリンク化しない〉doc 参照）
///
/// `alpha = 1.7580993408473766`（SELU 定数）・
/// `a = 1 / sqrt((alpha^2 * p + 1) * (1 - p))`（`f64` で計算し 1 回
/// `f32` へ narrow）として、要素ごとに
/// `out = x * noise + b`（`noise` は keep 位置で `a`・drop 位置で
/// `0.0`。`b` は keep 位置で `alpha * a * p`・drop 位置で
/// `-(alpha * a) + alpha * a * p`）。
///
/// forward は `crate::var::Var::dropout_with_mask`（`pub(crate)`。
/// `x * noise`。[`Dropout`] と共有）→ [`crate::var::Var::add`]（`+ b`。`b` は
/// [`crate::tape::Tape::var_no_grad`] で勾配を持たない定数として登録）
/// の 2 段構成。`b` への勾配は流さない（`noise` に対する定数バイアスの
/// ため、`x` への勾配は `upstream * noise` のみで `AlphaDropout` 固有の
/// VJP は不要）。
///
/// # `p == 1.0` の特例
///
/// `a` の分母 `(1 - p)` がゼロになり `a = inf` から `inf * 0 = NaN` が
/// 生じるため、`crate::grad::alpha_dropout_mask_and_bias` を呼ばず
/// 全ゼロマスク（`b` は加えない）で `x * 0` を返す
/// （`at::native::_dropout_impl` の `p == 1` 特例と同じ）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaDropout {
    p: f32,
    training: bool,
}

impl AlphaDropout {
    /// [`Dropout::new`] と同じ検査（`p` は有限かつ `[0, 1]`）。
    pub fn new(p: f32) -> Result<Self, AutodiffError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(AutodiffError::InvalidArgument(format!(
                "AlphaDropout::new: p must be finite and in [0, 1], got {p}"
            )));
        }
        Ok(Self { p, training: true })
    }

    /// 構築済みの `p`（drop 確率）。
    pub fn p(&self) -> f32 {
        self.p
    }

    /// モジュール doc「数式」節・「`p == 1.0` の特例」節参照。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        if !self.training || self.p == 0.0 {
            return Ok(*input);
        }
        let shape = input.shape();
        if self.p == 1.0 {
            let zeros = Tensor::<f32>::zeros(&shape).map_err(AutodiffError::Shape)?;
            return input.dropout_with_mask(zeros);
        }
        let (noise, bias) = crate::grad::alpha_dropout_mask_and_bias(&shape, self.p)?;
        let y = input.dropout_with_mask(noise)?;
        let b = y.tape().var_no_grad(&bias);
        y.add(&b)
    }
}

impl Default for AlphaDropout {
    /// PyTorch `torch.nn.AlphaDropout` の既定値（`p=0.5`）。[`Dropout::
    /// default`] と同じく構造体リテラルで直接構築する（本番経路
    /// panic 禁止）。
    fn default() -> Self {
        Self {
            p: 0.5,
            training: true,
        }
    }
}

impl Module for AlphaDropout {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        AlphaDropout::forward(self, input)
    }

    /// [`Self::forward`] と**同一の関数列**（早期リターン判定 →
    /// `p == 1.0` 特例 → `crate::grad::alpha_dropout_mask_and_bias` →
    /// `crate::grad::dropout_with_fallback` → `crate::grad::
    /// alpha_dropout_bias_add_with_fallback`）を `tape` 不要経路で
    /// 再現する（[`Dropout::forward_host`] と同型。bit-exactness が
    /// 構造的に成立する）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        let shape = input.shape();
        if self.p == 1.0 {
            let zeros = Tensor::<f32>::zeros(shape).map_err(AutodiffError::Shape)?;
            return crate::grad::dropout_with_fallback(ops, input, &zeros);
        }
        let (noise, bias) = crate::grad::alpha_dropout_mask_and_bias(shape, self.p)?;
        let y = crate::grad::dropout_with_fallback(ops, input, &noise)?;
        crate::grad::alpha_dropout_bias_add_with_fallback(ops, &y, &bias)
    }

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
