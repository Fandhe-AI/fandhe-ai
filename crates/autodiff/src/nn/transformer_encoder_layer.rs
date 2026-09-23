//! `TransformerEncoderLayer` Module（イシュー #2068。親 #2059）。
//!
//! **設計方針（新規 `Op`／`BackendOps` メソッド／VJP／カーネルを追加
//! しない。`nn::attention`（#1640）・`nn::normalization`（#2066）と
//! 同型）**: self-attention（[`crate::nn::MultiheadAttention`]）・
//! LayerNorm（[`crate::nn::LayerNorm`]）・FFN（`nn::Linear` 2 層 +
//! 活性化）・residual 加算（[`crate::var::Var::add`]）のすべてを既存の
//! `Var` 演算・既存 `nn` 部品の合成として組み立てる。VJP は各構成演算
//! の VJP 合成として自動的に成立し、バックエンド間数値一致もこれら
//! 既存演算の parity 契約にそのまま帰着する。
//!
//! **forward 順序（post-norm 固定。PyTorch `nn.TransformerEncoderLayer`
//! の既定 `norm_first=False` と同じ）**:
//! `x1 = norm1(x + self_attn(x))` → `x2 = norm2(x1 + ffn(x1))`。
//! `norm_first=True`（pre-norm）は対象外（`out-of-scope-tracking.md`）。
//!
//! **対象外（PR 本文に記載・`out-of-scope-tracking.md`）**: pre-norm・
//! Dropout 結線（層をモード非依存に保つため。PyTorch 既定 `dropout=0.1`
//! を省略）・`batch_first=False`・FFN 活性化の facade 公開（GELU
//! 選択。本層自体は [`FeedForwardActivation`] で両対応）・
//! `forward_host`（tape 不要推論経路。trait 既定のまま `Unsupported`
//! fail-safe）・AMP 低精度（#2071）・複数層スタック／Encoder-Decoder・
//! KV キャッシュ（#2083／#2084）。
//!
//! **入出力契約（rank-3・batch_first 固定）**: `input: [B, L, E]` →
//! 出力 `[B, L, E]`（[`crate::nn::MultiheadAttention`] の既存契約を継承）。
//!
//! **`Module` trait との関係**: `forward` は self-attention（mask
//! なし・非 causal）を表す。`supports_forward_host` は `false`
//! （`nn::attention`・`nn::embedding` と同型。`compat::Sequential::
//! predict` の tape 不要経路が本層で `Unsupported` に当たる前に全層を
//! 事前判定できるようにする）。学習可能パラメータの認識には専用の
//! [`crate::nn::module::Module::as_transformer_encoder_layer`] フックを使う。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::attention::{MultiheadAttention, MultiheadAttentionVars, project};
use crate::nn::init::{
    ENC_ATTN_SEED_SALT, ENC_LINEAR1_SEED_SALT, ENC_LINEAR2_SEED_SALT, derive_seed,
};
use crate::nn::linear::{Linear, LinearVars};
use crate::nn::module::{Module, prefixed, strip_child_prefix};
use crate::nn::norm::{LayerNorm, LayerNormVars};
use crate::tape::Tape;
use crate::var::Var;

/// FFN 中間層の活性化関数（PyTorch `nn.TransformerEncoderLayer` の
/// `activation` 引数のうち `"relu"`／`"gelu"` に相当する 2 択。既定は
/// `Relu`（PyTorch 既定と同じ）。`Var::relu` は infallible（`Result` を
/// 返さない）・`Var::gelu` は `Result` を返すため、[`TransformerEncoderLayerVars::forward`]
/// 側で両者の呼び出し形の違いを吸収する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeedForwardActivation {
    #[default]
    Relu,
    Gelu,
}

/// q/k/v/out・`linear1`／`linear2`・`norm1`／`norm2` の 5 子層をまたぐ
/// 整合性検証（[`TransformerEncoderLayer::from_parameters`] 向け）。
/// `d_model`（`self_attn.embed_dim()` == `linear1` の in_features ==
/// `linear2` の out_features）・`linear1` の out_features == `linear2`
/// の in_features（`dim_feedforward`）・`norm1`／`norm2` の `weight`／
/// `bias`（`Some` の場合）が `[d_model]` であることを検査する。
fn validate_layer_parameters(
    self_attn: &MultiheadAttention,
    linear1: &Linear,
    linear2: &Linear,
    norm1: &LayerNorm,
    norm2: &LayerNorm,
) -> Result<(usize, usize), AutodiffError> {
    let d_model = self_attn.embed_dim();

    let l1_shape = linear1.weight().shape();
    if l1_shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: l1_shape.len(),
        }));
    }
    if l1_shape[0] != d_model {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: l1_shape.to_vec(),
            rhs: vec![d_model, l1_shape.get(1).copied().unwrap_or(0)],
        }));
    }
    let dim_feedforward = l1_shape[1];

    let l2_shape = linear2.weight().shape();
    if l2_shape != [dim_feedforward, d_model] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: l2_shape.to_vec(),
            rhs: vec![dim_feedforward, d_model],
        }));
    }

    for norm in [norm1, norm2] {
        if let Some(w) = norm.weight()
            && w.shape() != [d_model]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: w.shape().to_vec(),
                rhs: vec![d_model],
            }));
        }
        if let Some(b) = norm.bias()
            && b.shape() != [d_model]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: b.shape().to_vec(),
                rhs: vec![d_model],
            }));
        }
    }

    Ok((d_model, dim_feedforward))
}

/// `validate_layer_parameters` の `*Vars`（テープ登録済み）版。加えて
/// 5 子層すべてが同一 `Tape` に属することを検査する
/// （`Var::check_same_tape`）。
fn validate_layer_vars<'t>(
    self_attn: &MultiheadAttentionVars<'t>,
    linear1: &LinearVars<'t>,
    linear2: &LinearVars<'t>,
    norm1: &LayerNormVars<'t>,
    norm2: &LayerNormVars<'t>,
) -> Result<(usize, usize), AutodiffError> {
    self_attn.q.weight.check_same_tape(&linear1.weight)?;
    self_attn.q.weight.check_same_tape(&linear2.weight)?;
    if let Some(w) = &norm1.weight {
        self_attn.q.weight.check_same_tape(w)?;
    }
    if let Some(b) = &norm1.bias {
        self_attn.q.weight.check_same_tape(b)?;
    }
    if let Some(w) = &norm2.weight {
        self_attn.q.weight.check_same_tape(w)?;
    }
    if let Some(b) = &norm2.bias {
        self_attn.q.weight.check_same_tape(b)?;
    }

    let d_model = self_attn.embed_dim();

    let l1_shape = linear1.weight.shape();
    if l1_shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: l1_shape.len(),
        }));
    }
    if l1_shape[0] != d_model {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: l1_shape,
            rhs: vec![d_model, 0],
        }));
    }
    let dim_feedforward = l1_shape[1];

    let l2_shape = linear2.weight.shape();
    if l2_shape.as_slice() != [dim_feedforward, d_model] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: l2_shape,
            rhs: vec![dim_feedforward, d_model],
        }));
    }

    for norm in [norm1, norm2] {
        if let Some(w) = &norm.weight
            && w.shape().as_slice() != [d_model]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: w.shape(),
                rhs: vec![d_model],
            }));
        }
        if let Some(b) = &norm.bias
            && b.shape().as_slice() != [d_model]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: b.shape(),
                rhs: vec![d_model],
            }));
        }
    }

    Ok((d_model, dim_feedforward))
}

/// `TransformerEncoderLayer` のパラメータ本体。PyTorch
/// `nn.TransformerEncoderLayer` と同じフィールド名
/// （`self_attn`／`linear1`／`linear2`／`norm1`／`norm2`）を使う
/// （`Module::named_parameters` の命名契約・#2080 の HF safetensors
/// 復元との整合）。
pub struct TransformerEncoderLayer {
    self_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    activation: FeedForwardActivation,
    d_model: usize,
    num_heads: usize,
    dim_feedforward: usize,
}

impl TransformerEncoderLayer {
    /// 決定的シードで `self_attn`・`linear1`・`linear2`・`norm1`・
    /// `norm2` を構築する。単一の呼び出し `seed` から `nn/init.rs` の
    /// 3 ソルト（`ENC_ATTN_SEED_SALT`〜`ENC_LINEAR2_SEED_SALT`）で
    /// 3 系統の独立した構築シードを導出する（`MultiheadAttention::new`
    /// と同じ「2 段の `derive_seed` 合成」構造）。`bias=true` 固定
    /// （PyTorch 既定と同じ）。
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        activation: FeedForwardActivation,
        eps: f32,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        if d_model == 0 {
            return Err(AutodiffError::InvalidArgument(
                "TransformerEncoderLayer::new: d_model must be > 0".to_string(),
            ));
        }
        if dim_feedforward == 0 {
            return Err(AutodiffError::InvalidArgument(
                "TransformerEncoderLayer::new: dim_feedforward must be > 0".to_string(),
            ));
        }
        let self_attn = MultiheadAttention::new(
            d_model,
            num_heads,
            true,
            derive_seed(seed, ENC_ATTN_SEED_SALT),
        )?;
        let linear1 = Linear::new(
            d_model,
            dim_feedforward,
            true,
            derive_seed(seed, ENC_LINEAR1_SEED_SALT),
        )?;
        let linear2 = Linear::new(
            dim_feedforward,
            d_model,
            true,
            derive_seed(seed, ENC_LINEAR2_SEED_SALT),
        )?;
        let norm1 = LayerNorm::new(d_model, eps)?;
        let norm2 = LayerNorm::new(d_model, eps)?;
        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            activation,
            d_model,
            num_heads,
            dim_feedforward,
        })
    }

    /// 明示的な 5 子層から構築する（テスト・safetensors 等の外部由来
    /// パラメータロード経路向けの入口。[`MultiheadAttention::
    /// from_parameters`] と同じ位置づけ）。子層間の横断整合性
    /// （`validate_layer_parameters`）を検証する。
    pub fn from_parameters(
        self_attn: MultiheadAttention,
        linear1: Linear,
        linear2: Linear,
        norm1: LayerNorm,
        norm2: LayerNorm,
        activation: FeedForwardActivation,
    ) -> Result<Self, AutodiffError> {
        let (d_model, dim_feedforward) =
            validate_layer_parameters(&self_attn, &linear1, &linear2, &norm1, &norm2)?;
        let num_heads = self_attn.num_heads();
        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            activation,
            d_model,
            num_heads,
            dim_feedforward,
        })
    }

    /// [`Self::new`]／[`Self::from_parameters`] に渡した `d_model`
    /// （= `self_attn.embed_dim()`）。
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// [`Self::new`]／[`Self::from_parameters`] に渡した `num_heads`。
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// [`Self::new`]／[`Self::from_parameters`] に渡した
    /// `dim_feedforward`。
    pub fn dim_feedforward(&self) -> usize {
        self.dim_feedforward
    }

    /// FFN 中間層の活性化関数。
    pub fn activation(&self) -> FeedForwardActivation {
        self.activation
    }

    /// self-attention 部（`nn::MultiheadAttention`）への参照。
    pub fn self_attn(&self) -> &MultiheadAttention {
        &self.self_attn
    }

    /// FFN 第 1 層（`d_model -> dim_feedforward`）への参照。
    pub fn linear1(&self) -> &Linear {
        &self.linear1
    }

    /// FFN 第 2 層（`dim_feedforward -> d_model`）への参照。
    pub fn linear2(&self) -> &Linear {
        &self.linear2
    }

    /// self-attention 後の LayerNorm への参照。
    pub fn norm1(&self) -> &LayerNorm {
        &self.norm1
    }

    /// FFN 後の LayerNorm への参照。
    pub fn norm2(&self) -> &LayerNorm {
        &self.norm2
    }

    /// このステップの `tape` へ 5 子層すべての学習可能パラメータを
    /// 葉ノードとして登録し、`forward` を呼べる
    /// [`TransformerEncoderLayerVars`] を返す（`MultiheadAttention::bind`
    /// と同じ「毎ステップ作り直す」契約）。`new`／`from_parameters` が
    /// 構築時に不変条件を検証済みのため、`bind` 自体は再検証しない。
    pub fn bind<'t>(&self, tape: &'t Tape) -> TransformerEncoderLayerVars<'t> {
        TransformerEncoderLayerVars {
            self_attn: self.self_attn.bind(tape),
            linear1: self.linear1.bind(tape),
            linear2: self.linear2.bind(tape),
            norm1: self.norm1.bind(tape),
            norm2: self.norm2.bind(tape),
            activation: self.activation,
            d_model: self.d_model,
            dim_feedforward: self.dim_feedforward,
        }
    }
}

/// `TransformerEncoderLayer::bind` が返す、1 ステップ分のテープに
/// 登録済みパラメータ。各フィールドを `pub` にする理由は
/// [`MultiheadAttentionVars`] と同じ（`Tape::backward` 後に
/// `Gradients::get` で勾配を取り出すのは呼び出し側の責務）に加え、
/// facade 横断 parity テストが [`Self::new`]（下記）経由で `tape.var(&tensor)`
/// から組み立てた子 `*Vars` を直接渡せるようにするため。
pub struct TransformerEncoderLayerVars<'t> {
    pub self_attn: MultiheadAttentionVars<'t>,
    pub linear1: LinearVars<'t>,
    pub linear2: LinearVars<'t>,
    pub norm1: LayerNormVars<'t>,
    pub norm2: LayerNormVars<'t>,
    activation: FeedForwardActivation,
    d_model: usize,
    dim_feedforward: usize,
}

impl<'t> TransformerEncoderLayerVars<'t> {
    /// `LinearVars`／`MultiheadAttentionVars`／`LayerNormVars` 5 個から
    /// 直接構築する。[`TransformerEncoderLayer::bind`] を経由しない
    /// 到達経路（facade 横断 parity テスト向け）のため、`bind` が
    /// 省略していた不変条件の検証（`validate_layer_vars`。子層間の
    /// shape 整合・同一 `Tape`）をここで行う。
    pub fn new(
        self_attn: MultiheadAttentionVars<'t>,
        linear1: LinearVars<'t>,
        linear2: LinearVars<'t>,
        norm1: LayerNormVars<'t>,
        norm2: LayerNormVars<'t>,
        activation: FeedForwardActivation,
    ) -> Result<Self, AutodiffError> {
        let (d_model, dim_feedforward) =
            validate_layer_vars(&self_attn, &linear1, &linear2, &norm1, &norm2)?;
        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            activation,
            d_model,
            dim_feedforward,
        })
    }

    /// `new` に渡した `d_model`。
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// `new` に渡した `dim_feedforward`。
    pub fn dim_feedforward(&self) -> usize {
        self.dim_feedforward
    }

    /// FFN 中間層の活性化関数。
    pub fn activation(&self) -> FeedForwardActivation {
        self.activation
    }

    /// `y = TransformerEncoderLayer(input)`。`input: [B, L, E]` →
    /// `[B, L, E]`（モジュール doc「forward 順序」参照）。
    ///
    /// 処理順序: ①shape 検査（rank・最終軸 == `d_model`）→ ②self-attn
    /// （`self_attn.forward(input, input, input, attn_mask, is_causal)`）
    /// → ③residual + `norm1` → ④FFN（`linear1` → activation →
    /// `linear2`。`nn::attention::project` で `[B, L, E]` <->
    /// `[B*L, E]` の reshape を挟む。`gemm_out_shape` が rank 2 を要求
    /// するため）→ ⑤residual + `norm2`。
    ///
    /// # Errors
    ///
    /// `input` の rank が 3 でない・最終軸が `d_model` と不一致の場合は
    /// `AutodiffError::Shape` を返す。`attn_mask`／`is_causal` の検証は
    /// `MultiheadAttentionVars::forward` へ委譲する。テープ不一致は
    /// `AutodiffError::TapeMismatch`（`self_attn.forward` の
    /// `check_same_tape` 経由）。
    pub fn forward(
        &self,
        input: &Var<'t>,
        attn_mask: Option<&Tensor<bool>>,
        is_causal: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: shape.len(),
            }));
        }
        let (b, l, e) = (shape[0], shape[1], shape[2]);
        if e != self.d_model {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: shape,
                rhs: vec![b, l, self.d_model],
            }));
        }

        // ①self-attention → residual → norm1。
        let attn_out = self
            .self_attn
            .forward(input, input, input, attn_mask, is_causal)?;
        let x1 = self.norm1.forward(&input.add(&attn_out)?)?;

        // ②FFN: [B, L, E] -> [B*L, E] -> linear1 -> activation ->
        // linear2 -> [B, L, E]（`attention::project` を再利用。
        // モジュール doc・`attention.rs::project` doc 参照）。
        let hidden = project(&x1, &self.linear1, b, l, e, self.dim_feedforward, None)?;
        let activated = match self.activation {
            FeedForwardActivation::Relu => hidden.relu(),
            FeedForwardActivation::Gelu => hidden.gelu()?,
        };
        let ffn_out = project(
            &activated,
            &self.linear2,
            b,
            l,
            self.dim_feedforward,
            e,
            None,
        )?;

        // ③residual + norm2。
        self.norm2.forward(&x1.add(&ffn_out)?)
    }
}

/// self-attention（`q=k=v=input`・mask なし・非 causal）として
/// `Module::forward` を定義する（モジュール doc「`Module` trait との
/// 関係」参照）。`forward_host` は trait 既定のまま
/// オーバーライドしない。
impl Module for TransformerEncoderLayer {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input, None, false)
    }

    fn as_transformer_encoder_layer(&self) -> Option<&TransformerEncoderLayer> {
        Some(self)
    }

    fn as_transformer_encoder_layer_mut(&mut self) -> Option<&mut TransformerEncoderLayer> {
        Some(self)
    }

    /// `forward_host` は trait 既定のまま（常に `Unsupported`）。
    /// [`Module::supports_forward_host`] を `false` へオーバーライド
    /// し、`compat::Sequential::predict` の tape 不要経路が本層で
    /// `Unsupported` に当たる前に全層を事前判定できるようにする
    /// （`nn::attention`・`nn::embedding` と同型）。
    fn supports_forward_host(&self) -> bool {
        false
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `self_attn.*` → `linear1.*` → `linear2.*` → `norm1.*` →
    /// `norm2.*` の順で、各子 `Module::named_parameters()` に接頭辞を
    /// 連結する（`module::prefixed` ヘルパー参照）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = prefixed("self_attn", self.self_attn.named_parameters());
        out.extend(prefixed("linear1", self.linear1.named_parameters()));
        out.extend(prefixed("linear2", self.linear2.named_parameters()));
        out.extend(prefixed("norm1", self.norm1.named_parameters()));
        out.extend(prefixed("norm2", self.norm2.named_parameters()));
        out
    }

    /// [`Module::set_parameter`] の実装。`strip_child_prefix` で
    /// `self_attn.`／`linear1.`／`linear2.`／`norm1.`／`norm2.` の
    /// いずれかを剥がし、対応する子の `set_parameter` へ委譲する
    /// （`named_parameters` の接頭辞契約の逆演算）。該当する接頭辞が
    /// ない名前は未知名として拒否する。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        if let Some(rest) = strip_child_prefix(name, "self_attn") {
            return Module::set_parameter(&mut self.self_attn, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "linear1") {
            return Linear::set_parameter(&mut self.linear1, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "linear2") {
            return Linear::set_parameter(&mut self.linear2, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "norm1") {
            return LayerNorm::set_parameter(&mut self.norm1, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "norm2") {
            return LayerNorm::set_parameter(&mut self.norm2, rest, value);
        }
        Err(AutodiffError::InvalidArgument(format!(
            "TransformerEncoderLayer::set_parameter: no parameter named `{name}`"
        )))
    }

    /// [`Module::children`] の実装（イシュー #2134）。順序・名前は
    /// [`Self::named_parameters`] の接頭辞契約（`self_attn`→
    /// `linear1`→`linear2`→`norm1`→`norm2`）と一致させる。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("self_attn".to_string(), &self.self_attn as &dyn Module),
            ("linear1".to_string(), &self.linear1 as &dyn Module),
            ("linear2".to_string(), &self.linear2 as &dyn Module),
            ("norm1".to_string(), &self.norm1 as &dyn Module),
            ("norm2".to_string(), &self.norm2 as &dyn Module),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::norm::LAYER_NORM_DEFAULT_EPS;
    use crate::tape::Tape;

    const D_MODEL: usize = 4;
    const NUM_HEADS: usize = 2;
    const DIM_FF: usize = 8;

    fn build(seed: u64) -> TransformerEncoderLayer {
        TransformerEncoderLayer::new(
            D_MODEL,
            NUM_HEADS,
            DIM_FF,
            FeedForwardActivation::Relu,
            LAYER_NORM_DEFAULT_EPS,
            seed,
        )
        .unwrap()
    }

    #[test]
    fn new_rejects_zero_d_model() {
        assert!(
            TransformerEncoderLayer::new(
                0,
                NUM_HEADS,
                DIM_FF,
                FeedForwardActivation::Relu,
                LAYER_NORM_DEFAULT_EPS,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn new_rejects_zero_dim_feedforward() {
        assert!(
            TransformerEncoderLayer::new(
                D_MODEL,
                NUM_HEADS,
                0,
                FeedForwardActivation::Relu,
                LAYER_NORM_DEFAULT_EPS,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn new_rejects_embed_dim_not_divisible_by_num_heads() {
        assert!(
            TransformerEncoderLayer::new(
                5,
                NUM_HEADS,
                DIM_FF,
                FeedForwardActivation::Relu,
                LAYER_NORM_DEFAULT_EPS,
                1
            )
            .is_err()
        );
    }

    #[test]
    fn from_parameters_rejects_cross_layer_shape_mismatch() {
        let self_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, 1).unwrap();
        let linear1 = Linear::new(D_MODEL, DIM_FF, true, 2).unwrap();
        // linear2 の in_features が dim_feedforward と食い違う。
        let linear2 = Linear::new(DIM_FF + 1, D_MODEL, true, 3).unwrap();
        let norm1 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        let norm2 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        let result = TransformerEncoderLayer::from_parameters(
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            FeedForwardActivation::Relu,
        );
        // `TransformerEncoderLayer`（`MultiheadAttention` を内包）は
        // `Debug` を導出していないため `unwrap_err` は使えない
        // （`Result::unwrap_err` は `T: Debug` を要求する）。
        match result {
            Err(AutodiffError::Shape(_)) => {}
            Err(other) => panic!("Shape エラーを期待したが別の Err: {other:?}"),
            Ok(_) => panic!("linear2 の in_features 不一致は Err を返すはず"),
        }
    }

    #[test]
    fn from_parameters_accepts_consistent_layers() {
        let self_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, 1).unwrap();
        let linear1 = Linear::new(D_MODEL, DIM_FF, true, 2).unwrap();
        let linear2 = Linear::new(DIM_FF, D_MODEL, true, 3).unwrap();
        let norm1 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        let norm2 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        let layer = TransformerEncoderLayer::from_parameters(
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            FeedForwardActivation::Relu,
        )
        .unwrap();
        assert_eq!(layer.d_model(), D_MODEL);
        assert_eq!(layer.dim_feedforward(), DIM_FF);
    }

    /// `TransformerEncoderLayerVars::new`（公開 `Result` コンストラクタ）
    /// は `linear1.weight` の rank を検証してから軸へアクセスすべきで
    /// あり、rank-1／rank-0／rank-3 のいずれを渡しても panic せず
    /// `Err` を返す必要がある（codex-review／Cursor Bugbot 指摘。PR #2211）。
    #[test]
    fn new_vars_rejects_non_rank2_linear1_weight() {
        for bad_shape in [vec![D_MODEL], vec![], vec![D_MODEL, DIM_FF, 1]] {
            let tape = Tape::new();
            let self_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, 1)
                .unwrap()
                .bind(&tape);
            let bad_weight =
                Tensor::new(vec![0.0f32; bad_shape.iter().product()], &bad_shape).unwrap();
            let linear1 = LinearVars {
                weight: tape.var(&bad_weight),
                bias: None,
            };
            let linear2 = Linear::new(DIM_FF, D_MODEL, true, 3).unwrap().bind(&tape);
            let norm1 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS)
                .unwrap()
                .bind(&tape);
            let norm2 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS)
                .unwrap()
                .bind(&tape);
            let result = TransformerEncoderLayerVars::new(
                self_attn,
                linear1,
                linear2,
                norm1,
                norm2,
                FeedForwardActivation::Relu,
            );
            match result {
                Err(AutodiffError::Shape(ShapeError::RankMismatch { expected, actual })) => {
                    assert_eq!(expected, 2);
                    assert_eq!(actual, bad_shape.len());
                }
                Err(other) => {
                    panic!("RankMismatch を期待したが別の Err（shape={bad_shape:?}）: {other:?}")
                }
                Ok(_) => {
                    panic!("rank != 2 の linear1.weight は Err を返すはず（shape={bad_shape:?}）")
                }
            }
        }
    }

    #[test]
    fn named_parameters_has_expected_names_and_order() {
        let layer = build(7);
        let names: Vec<String> = layer
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            names,
            vec![
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias",
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
            ]
        );
    }

    #[test]
    fn set_parameter_delegates_to_correct_child() {
        let mut layer = build(7);
        let new_bias = Tensor::new(vec![9.0f32; D_MODEL], &[D_MODEL]).unwrap();
        layer.set_parameter("norm1.bias", new_bias.clone()).unwrap();
        assert_eq!(
            layer
                .norm1()
                .bias()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            new_bias.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn set_parameter_rejects_unknown_prefix() {
        let mut layer = build(7);
        let dummy = layer.norm1().bias().unwrap().clone();
        let err = layer
            .set_parameter("bogus.weight", dummy)
            .expect_err("未知の接頭辞は Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_output_shape_matches_input() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(11);
        let x = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let out = layer.bind(&tape).forward(&x, None, false).unwrap();
        assert_eq!(out.shape(), vec![2, 3, D_MODEL]);
    }

    #[test]
    fn forward_rejects_rank_mismatch() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(11);
        let x = tape.var(&Tensor::new(vec![0.1f32; 3 * D_MODEL], &[3, D_MODEL]).unwrap());
        let err = layer.bind(&tape).forward(&x, None, false).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn forward_matches_manual_composition_bit_exact() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(13);
        let x_tensor = Tensor::new(vec![0.05f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap();
        let x = tape.var(&x_tensor);

        let via_layer = layer.bind(&tape).forward(&x, None, false).unwrap();

        // 手動合成（`Module::forward` の実装と同一の演算列）。
        let bound = layer.bind(&tape);
        let attn = bound.self_attn.forward(&x, &x, &x, None, false).unwrap();
        let x1 = bound.norm1.forward(&x.add(&attn).unwrap()).unwrap();
        let hidden = project(&x1, &bound.linear1, 2, 3, D_MODEL, DIM_FF, None).unwrap();
        let activated = hidden.relu();
        let ffn_out = project(&activated, &bound.linear2, 2, 3, DIM_FF, D_MODEL, None).unwrap();
        let manual = bound.norm2.forward(&x1.add(&ffn_out).unwrap()).unwrap();

        assert_eq!(
            via_layer.to_tensor().contiguous().as_slice().unwrap(),
            manual.to_tensor().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn backward_reaches_all_sixteen_parameters() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(17);
        let x = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let bound = layer.bind(&tape);
        let out = bound.forward(&x, None, false).unwrap();
        let loss = out.mean(None).unwrap();
        let grads = tape.backward(&loss).unwrap();

        assert!(grads.get(&bound.self_attn.q.weight).unwrap().is_some());
        assert!(
            grads
                .get(&bound.self_attn.q.bias.unwrap())
                .unwrap()
                .is_some()
        );
        assert!(grads.get(&bound.self_attn.k.weight).unwrap().is_some());
        assert!(grads.get(&bound.self_attn.v.weight).unwrap().is_some());
        assert!(grads.get(&bound.self_attn.out.weight).unwrap().is_some());
        assert!(grads.get(&bound.linear1.weight).unwrap().is_some());
        assert!(grads.get(&bound.linear1.bias.unwrap()).unwrap().is_some());
        assert!(grads.get(&bound.linear2.weight).unwrap().is_some());
        assert!(grads.get(&bound.linear2.bias.unwrap()).unwrap().is_some());
        assert!(
            grads
                .get(bound.norm1.weight.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
        assert!(
            grads
                .get(bound.norm1.bias.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
        assert!(
            grads
                .get(bound.norm2.weight.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
        assert!(
            grads
                .get(bound.norm2.bias.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
    }
}
