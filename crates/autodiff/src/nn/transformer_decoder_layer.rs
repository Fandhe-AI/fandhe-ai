//! `TransformerDecoderLayer` Module（イシュー #2165。親 #2131。#2068
//! の対）。
//!
//! **設計方針（新規 `Op`／`BackendOps` メソッド／VJP／カーネルを追加
//! しない。`nn::transformer_encoder_layer`（#2068）と同型）**:
//! self-attention・cross-attention（いずれも
//! [`crate::nn::MultiheadAttention`]）・LayerNorm（[`crate::nn::LayerNorm`]）・
//! FFN（`nn::Linear` 2 層 + 活性化）・residual 加算（[`crate::var::Var::add`]）
//! のすべてを既存の `Var` 演算・既存 `nn` 部品の合成として組み立てる。
//! VJP は各構成演算の VJP 合成として自動的に成立し、バックエンド間
//! 数値一致もこれら既存演算の parity 契約にそのまま帰着する。
//!
//! **forward 順序（post-norm 固定。PyTorch `nn.TransformerDecoderLayer`
//! の既定 `norm_first=False` と同じ）**:
//! `x1 = norm1(tgt + self_attn(tgt, tgt, tgt))` →
//! `x2 = norm2(x1 + multihead_attn(x1, memory, memory))` →
//! `y = norm3(x2 + ffn(x2))`。`norm_first=True`（pre-norm）は対象外
//! （`out-of-scope-tracking.md`）。
//!
//! **mask 極性の注意**: `tgt_mask`／`memory_mask` は本クレートの
//! `MultiheadAttention` と同じ `true` = attend 規約（`nn/attention.rs`
//! モジュール doc「mask 極性の注意」参照）であり、PyTorch
//! `nn.TransformerDecoderLayer` の bool mask（`True` = blocked）とは
//! **極性が逆**である。
//!
//! **対象外（PR 本文に記載・`out-of-scope-tracking.md`）**: pre-norm・
//! Dropout 結線・`batch_first=False`・`tgt_key_padding_mask`／
//! `memory_key_padding_mask`・facade 公開（`compat::Sequential::
//! add_transformer_decoder_layer` 相当。承認事項。`crates/facade/src/lib.rs`
//! の `TransformerDecoderHoldDoctestGuard` で保留固定）・
//! `forward_host`（tape 不要推論経路。trait 既定のまま `Unsupported`
//! fail-safe）・AMP 低精度（#2071）・KV キャッシュ（#2083／#2084）。
//!
//! **入出力契約（rank-3・batch_first 固定）**: `tgt: [B, L, E]`・
//! `memory: [B, S, E]`（`S` は `L` と異なってよい） → 出力 `[B, L, E]`
//! （[`crate::nn::MultiheadAttention`] の既存契約を継承）。
//!
//! **`Module` trait との関係**: `forward` は `tgt = memory = input`・
//! mask なし・非 causal を表す（`nn/transformer_encoder_layer.rs` の
//! 単一入力慣習を踏襲。§「単一入力の `Module::forward` の意味論」参照）。
//! `supports_forward_host` は `false`。学習可能パラメータの認識には
//! 専用の [`crate::nn::module::Module::as_transformer_decoder_layer`]
//! フックを使う。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::attention::{MultiheadAttention, MultiheadAttentionVars, project};
use crate::nn::init::{
    DEC_CROSS_ATTN_SEED_SALT, DEC_LINEAR1_SEED_SALT, DEC_LINEAR2_SEED_SALT,
    DEC_SELF_ATTN_SEED_SALT, derive_seed,
};
use crate::nn::linear::{Linear, LinearVars};
use crate::nn::module::{Module, prefixed, strip_child_prefix};
use crate::nn::norm::{LayerNorm, LayerNormVars};
use crate::nn::transformer_encoder_layer::FeedForwardActivation;
use crate::tape::Tape;
use crate::var::Var;

/// `self_attn`／`multihead_attn` の非既定 `MultiheadAttentionConfig`
/// （`batch_first == false`・`kdim`/`vdim != d_model`）を拒否する
/// 共通ゲート（イシュー #2163 の fail-closed 化と同じ理由:
/// `residual = x + attn(..)` は `[B, L, E]` 同士の加算を前提とする）。
/// `who` はエラーメッセージ用のラベル（`"self_attn"`／`"multihead_attn"`）。
fn reject_non_default_mha_config(
    attn: &MultiheadAttention,
    d_model: usize,
    who: &str,
) -> Result<(), AutodiffError> {
    if !attn.batch_first() || attn.kdim() != d_model || attn.vdim() != d_model {
        return Err(AutodiffError::InvalidArgument(format!(
            "TransformerDecoderLayer: {who} must have batch_first=true and kdim=vdim=d_model \
             （イシュー #2163 スコープ外。非既定 MultiheadAttentionConfig は未対応）"
        )));
    }
    Ok(())
}

/// [`reject_non_default_mha_config`] の `*Vars` 版。
fn reject_non_default_mha_config_vars(
    attn: &MultiheadAttentionVars<'_>,
    d_model: usize,
    who: &str,
) -> Result<(), AutodiffError> {
    if !attn.batch_first() || attn.kdim() != d_model || attn.vdim() != d_model {
        return Err(AutodiffError::InvalidArgument(format!(
            "TransformerDecoderLayerVars: {who} must have batch_first=true and kdim=vdim=d_model \
             （イシュー #2163 スコープ外。非既定 MultiheadAttentionConfig は未対応）"
        )));
    }
    Ok(())
}

/// [`validate_ffn_and_norms`] の `norm_shapes` 引数の型（clippy
/// `type_complexity` 回避。各要素は 1 個の LayerNorm の
/// `(weight_shape, bias_shape)`）。
type NormShapeTriple = [(Option<Vec<usize>>, Option<Vec<usize>>); 3];

/// `linear1`/`linear2`/`norm1`/`norm2`/`norm3` の shape 整合を検証する
/// （`self_attn`／`multihead_attn` の `embed_dim` 一致検査後に呼ばれる
/// 共通処理。`validate_layer_parameters`／`validate_layer_vars` から
/// 使う）。`(dim_feedforward)` を返す。
fn validate_ffn_and_norms(
    d_model: usize,
    l1_shape: &[usize],
    l2_shape: &[usize],
    norm_shapes: NormShapeTriple,
) -> Result<usize, AutodiffError> {
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

    if l2_shape != [dim_feedforward, d_model] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: l2_shape.to_vec(),
            rhs: vec![dim_feedforward, d_model],
        }));
    }

    for (weight, bias) in norm_shapes {
        if let Some(w) = weight
            && w != [d_model]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: w,
                rhs: vec![d_model],
            }));
        }
        if let Some(b) = bias
            && b != [d_model]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: b,
                rhs: vec![d_model],
            }));
        }
    }

    Ok(dim_feedforward)
}

/// `self_attn`／`multihead_attn`／`linear1`／`linear2`／`norm1`／
/// `norm2`／`norm3` の 7 子層をまたぐ整合性検証
/// （[`TransformerDecoderLayer::from_parameters`] 向け）。
fn validate_layer_parameters(
    self_attn: &MultiheadAttention,
    multihead_attn: &MultiheadAttention,
    linear1: &Linear,
    linear2: &Linear,
    norm1: &LayerNorm,
    norm2: &LayerNorm,
    norm3: &LayerNorm,
) -> Result<(usize, usize), AutodiffError> {
    let d_model = self_attn.embed_dim();
    reject_non_default_mha_config(self_attn, d_model, "self_attn")?;
    reject_non_default_mha_config(multihead_attn, d_model, "multihead_attn")?;
    if multihead_attn.embed_dim() != d_model {
        return Err(AutodiffError::InvalidArgument(format!(
            "TransformerDecoderLayer: self_attn.embed_dim ({d_model}) must equal \
             multihead_attn.embed_dim ({})",
            multihead_attn.embed_dim()
        )));
    }

    let dim_feedforward = validate_ffn_and_norms(
        d_model,
        linear1.weight().shape(),
        linear2.weight().shape(),
        [
            (
                norm1.weight().map(|t| t.shape().to_vec()),
                norm1.bias().map(|t| t.shape().to_vec()),
            ),
            (
                norm2.weight().map(|t| t.shape().to_vec()),
                norm2.bias().map(|t| t.shape().to_vec()),
            ),
            (
                norm3.weight().map(|t| t.shape().to_vec()),
                norm3.bias().map(|t| t.shape().to_vec()),
            ),
        ],
    )?;

    Ok((d_model, dim_feedforward))
}

/// [`validate_layer_parameters`] の `*Vars`（テープ登録済み）版。加えて
/// 7 子層すべてが同一 `Tape` に属することを検査する
/// （`Var::check_same_tape`）。
fn validate_layer_vars<'t>(
    self_attn: &MultiheadAttentionVars<'t>,
    multihead_attn: &MultiheadAttentionVars<'t>,
    linear1: &LinearVars<'t>,
    linear2: &LinearVars<'t>,
    norm1: &LayerNormVars<'t>,
    norm2: &LayerNormVars<'t>,
    norm3: &LayerNormVars<'t>,
) -> Result<(usize, usize), AutodiffError> {
    let d_model = self_attn.embed_dim();
    reject_non_default_mha_config_vars(self_attn, d_model, "self_attn")?;
    reject_non_default_mha_config_vars(multihead_attn, d_model, "multihead_attn")?;
    if multihead_attn.embed_dim() != d_model {
        return Err(AutodiffError::InvalidArgument(format!(
            "TransformerDecoderLayerVars: self_attn.embed_dim ({d_model}) must equal \
             multihead_attn.embed_dim ({})",
            multihead_attn.embed_dim()
        )));
    }

    self_attn
        .q
        .weight
        .check_same_tape(&multihead_attn.q.weight)?;
    self_attn.q.weight.check_same_tape(&linear1.weight)?;
    self_attn.q.weight.check_same_tape(&linear2.weight)?;
    for norm in [norm1, norm2, norm3] {
        if let Some(w) = &norm.weight {
            self_attn.q.weight.check_same_tape(w)?;
        }
        if let Some(b) = &norm.bias {
            self_attn.q.weight.check_same_tape(b)?;
        }
    }

    let dim_feedforward = validate_ffn_and_norms(
        d_model,
        &linear1.weight.shape(),
        &linear2.weight.shape(),
        [
            (
                norm1.weight.as_ref().map(|t| t.shape()),
                norm1.bias.as_ref().map(|t| t.shape()),
            ),
            (
                norm2.weight.as_ref().map(|t| t.shape()),
                norm2.bias.as_ref().map(|t| t.shape()),
            ),
            (
                norm3.weight.as_ref().map(|t| t.shape()),
                norm3.bias.as_ref().map(|t| t.shape()),
            ),
        ],
    )?;

    Ok((d_model, dim_feedforward))
}

/// [`TransformerDecoderLayer::from_parameters`] へ渡す 7 子層をまとめる
/// 構造体。関連関数の引数が 8 個（self を含めない）になり clippy の
/// 既定閾値（7）を超えるため、`#[allow(clippy::too_many_arguments)]`
/// ではなくこの構造体で引数をまとめる（実装計画 §2.1）。フィールドは
/// すべて `pub`（構築時にリテラルで組み立てられるようにするための
/// 素朴なデータ運搬用構造体であり、独自の不変条件は持たない——
/// 不変条件の検証は `TransformerDecoderLayer::from_parameters` 側が行う）。
pub struct TransformerDecoderLayerParts {
    pub self_attn: MultiheadAttention,
    pub multihead_attn: MultiheadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub activation: FeedForwardActivation,
}

/// `TransformerDecoderLayer` のパラメータ本体。PyTorch
/// `nn.TransformerDecoderLayer` と同じフィールド名（`self_attn`／
/// `multihead_attn`／`linear1`／`linear2`／`norm1`／`norm2`／`norm3`）を
/// 使う（`Module::named_parameters` の命名契約・HF safetensors 復元
/// との整合。`nn/transformer_encoder_layer.rs` と同方針）。
pub struct TransformerDecoderLayer {
    self_attn: MultiheadAttention,
    multihead_attn: MultiheadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    activation: FeedForwardActivation,
    d_model: usize,
    num_heads: usize,
    dim_feedforward: usize,
}

impl TransformerDecoderLayer {
    /// 決定的シードで 7 子層を構築する。単一の呼び出し `seed` から
    /// `nn/init.rs` の 4 ソルト（`DEC_SELF_ATTN_SEED_SALT`〜
    /// `DEC_LINEAR2_SEED_SALT`）で 4 系統の独立した構築シードを導出する
    /// （`MultiheadAttention::new`・`TransformerEncoderLayer::new` と
    /// 同じ「2 段の `derive_seed` 合成」構造）。`bias=true` 固定
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
                "TransformerDecoderLayer::new: d_model must be > 0".to_string(),
            ));
        }
        if dim_feedforward == 0 {
            return Err(AutodiffError::InvalidArgument(
                "TransformerDecoderLayer::new: dim_feedforward must be > 0".to_string(),
            ));
        }
        let self_attn = MultiheadAttention::new(
            d_model,
            num_heads,
            true,
            derive_seed(seed, DEC_SELF_ATTN_SEED_SALT),
        )?;
        let multihead_attn = MultiheadAttention::new(
            d_model,
            num_heads,
            true,
            derive_seed(seed, DEC_CROSS_ATTN_SEED_SALT),
        )?;
        let linear1 = Linear::new(
            d_model,
            dim_feedforward,
            true,
            derive_seed(seed, DEC_LINEAR1_SEED_SALT),
        )?;
        let linear2 = Linear::new(
            dim_feedforward,
            d_model,
            true,
            derive_seed(seed, DEC_LINEAR2_SEED_SALT),
        )?;
        let norm1 = LayerNorm::new(d_model, eps)?;
        let norm2 = LayerNorm::new(d_model, eps)?;
        let norm3 = LayerNorm::new(d_model, eps)?;
        Ok(Self {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            activation,
            d_model,
            num_heads,
            dim_feedforward,
        })
    }

    /// 明示的な 7 子層（[`TransformerDecoderLayerParts`] 経由）から
    /// 構築する（テスト・safetensors 等の外部由来パラメータロード経路
    /// 向けの入口。[`MultiheadAttention::from_parameters`] と同じ
    /// 位置づけ）。子層間の横断整合性（`validate_layer_parameters`）
    /// を検証する。
    pub fn from_parameters(parts: TransformerDecoderLayerParts) -> Result<Self, AutodiffError> {
        let TransformerDecoderLayerParts {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            activation,
        } = parts;
        let (d_model, dim_feedforward) = validate_layer_parameters(
            &self_attn,
            &multihead_attn,
            &linear1,
            &linear2,
            &norm1,
            &norm2,
            &norm3,
        )?;
        let num_heads = self_attn.num_heads();
        Ok(Self {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
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

    /// cross-attention 部（`nn::MultiheadAttention`）への参照。
    pub fn multihead_attn(&self) -> &MultiheadAttention {
        &self.multihead_attn
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

    /// cross-attention 後の LayerNorm への参照。
    pub fn norm2(&self) -> &LayerNorm {
        &self.norm2
    }

    /// FFN 後の LayerNorm への参照。
    pub fn norm3(&self) -> &LayerNorm {
        &self.norm3
    }

    /// このステップの `tape` へ 7 子層すべての学習可能パラメータを
    /// 葉ノードとして登録し、`forward` を呼べる
    /// [`TransformerDecoderLayerVars`] を返す（`MultiheadAttention::bind`
    /// と同じ「毎ステップ作り直す」契約）。`new`／`from_parameters` が
    /// 構築時に不変条件を検証済みのため、`bind` 自体は再検証しない。
    pub fn bind<'t>(&self, tape: &'t Tape) -> TransformerDecoderLayerVars<'t> {
        TransformerDecoderLayerVars {
            self_attn: self.self_attn.bind(tape),
            multihead_attn: self.multihead_attn.bind(tape),
            linear1: self.linear1.bind(tape),
            linear2: self.linear2.bind(tape),
            norm1: self.norm1.bind(tape),
            norm2: self.norm2.bind(tape),
            norm3: self.norm3.bind(tape),
            activation: self.activation,
            d_model: self.d_model,
            dim_feedforward: self.dim_feedforward,
        }
    }
}

/// [`TransformerDecoderLayerVars::new`] へ渡す 7 子 `*Vars` をまとめる
/// 構造体（[`TransformerDecoderLayerParts`] の `*Vars` 版。同じ
/// clippy 閾値超過の回避理由）。
pub struct TransformerDecoderLayerVarsParts<'t> {
    pub self_attn: MultiheadAttentionVars<'t>,
    pub multihead_attn: MultiheadAttentionVars<'t>,
    pub linear1: LinearVars<'t>,
    pub linear2: LinearVars<'t>,
    pub norm1: LayerNormVars<'t>,
    pub norm2: LayerNormVars<'t>,
    pub norm3: LayerNormVars<'t>,
    pub activation: FeedForwardActivation,
}

/// `TransformerDecoderLayer::bind` が返す、1 ステップ分のテープに
/// 登録済みパラメータ。各フィールドを `pub` にする理由は
/// [`MultiheadAttentionVars`] と同じ（`Tape::backward` 後に
/// `Gradients::get` で勾配を取り出すのは呼び出し側の責務）に加え、
/// facade 横断 parity テストが [`Self::new`]（下記）経由で `tape.var(&tensor)`
/// から組み立てた子 `*Vars` を直接渡せるようにするため。
pub struct TransformerDecoderLayerVars<'t> {
    pub self_attn: MultiheadAttentionVars<'t>,
    pub multihead_attn: MultiheadAttentionVars<'t>,
    pub linear1: LinearVars<'t>,
    pub linear2: LinearVars<'t>,
    pub norm1: LayerNormVars<'t>,
    pub norm2: LayerNormVars<'t>,
    pub norm3: LayerNormVars<'t>,
    activation: FeedForwardActivation,
    d_model: usize,
    dim_feedforward: usize,
}

impl<'t> TransformerDecoderLayerVars<'t> {
    /// [`TransformerDecoderLayerVarsParts`] から直接構築する
    /// （[`TransformerDecoderLayer::bind`] を経由しない到達経路。
    /// facade 横断 parity テスト向け）。[`TransformerDecoderLayer::bind`]
    /// が省略していた不変条件の検証（`validate_layer_vars`。子層間の
    /// shape 整合・同一 `Tape`）をここで行う。
    pub fn new(parts: TransformerDecoderLayerVarsParts<'t>) -> Result<Self, AutodiffError> {
        let TransformerDecoderLayerVarsParts {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            activation,
        } = parts;
        let (d_model, dim_feedforward) = validate_layer_vars(
            &self_attn,
            &multihead_attn,
            &linear1,
            &linear2,
            &norm1,
            &norm2,
            &norm3,
        )?;
        Ok(Self {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
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

    /// `y = TransformerDecoderLayer(tgt, memory)`。`tgt: [B, L, E]`・
    /// `memory: [B, S, E]`（`S` は `L` と異なってよい） → `[B, L, E]`
    /// （モジュール doc「forward 順序」参照）。
    ///
    /// 処理順序: ①shape 検査（`tgt`／`memory` ともに rank 3・最終軸
    /// == `d_model`・バッチ次元 `B` 一致） → ②self-attention
    /// （`self_attn.forward(tgt, tgt, tgt, tgt_mask, tgt_is_causal)`）
    /// → ③residual + `norm1` → ④cross-attention
    /// （`multihead_attn.forward(x1, memory, memory, memory_mask,
    /// memory_is_causal)`） → ⑤residual + `norm2` → ⑥FFN（`linear1` →
    /// activation → `linear2`。`nn::attention::project` で `[B, L, E]`
    /// <-> `[B*L, E]` の reshape を挟む） → ⑦residual + `norm3`。
    ///
    /// mask 極性・`attn_mask`／`is_causal` の同時指定拒否は
    /// [`MultiheadAttentionVars::forward`] へ委譲する（モジュール doc
    /// 「mask 極性の注意」参照）。
    ///
    /// # Errors
    ///
    /// `tgt`／`memory` の rank が 3 でない・最終軸が `d_model` と
    /// 不一致・バッチ次元 `B` が不一致の場合は `AutodiffError::Shape`
    /// を返す。mask／causal の検証は
    /// `MultiheadAttentionVars::forward` へ委譲する。テープ不一致は
    /// `AutodiffError::TapeMismatch`（`self_attn.forward`／
    /// `multihead_attn.forward` の `check_same_tape` 経由）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tgt: &Var<'t>,
        memory: &Var<'t>,
        tgt_mask: Option<&Tensor<bool>>,
        memory_mask: Option<&Tensor<bool>>,
        tgt_is_causal: bool,
        memory_is_causal: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        let tgt_shape = tgt.shape();
        if tgt_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: tgt_shape.len(),
            }));
        }
        let (b, l, e) = (tgt_shape[0], tgt_shape[1], tgt_shape[2]);
        if e != self.d_model {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: tgt_shape,
                rhs: vec![b, l, self.d_model],
            }));
        }

        let mem_shape = memory.shape();
        if mem_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: mem_shape.len(),
            }));
        }
        let s = mem_shape[1];
        if mem_shape[0] != b || mem_shape[2] != self.d_model {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: mem_shape,
                rhs: vec![b, s, self.d_model],
            }));
        }

        // ①self-attention → residual → norm1。
        let self_attn_out = self
            .self_attn
            .forward(tgt, tgt, tgt, tgt_mask, tgt_is_causal)?;
        let x1 = self.norm1.forward(&tgt.add(&self_attn_out)?)?;

        // ②cross-attention（query=x1・key=value=memory） → residual →
        // norm2。
        let cross_attn_out =
            self.multihead_attn
                .forward(&x1, memory, memory, memory_mask, memory_is_causal)?;
        let x2 = self.norm2.forward(&x1.add(&cross_attn_out)?)?;

        // ③FFN: [B, L, E] -> [B*L, E] -> linear1 -> activation ->
        // linear2 -> [B, L, E]（`attention::project` を再利用。
        // `nn/transformer_encoder_layer.rs` と同型）。
        let hidden = project(&x2, &self.linear1, b, l, e, self.dim_feedforward, None)?;
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

        // ④residual + norm3。
        self.norm3.forward(&x2.add(&ffn_out)?)
    }
}

/// `tgt = memory = input`（mask なし・非 causal）として
/// `Module::forward` を定義する（モジュール doc「`Module` trait との
/// 関係」参照。`TransformerEncoderLayer` の単一入力慣習を踏襲）。
/// `forward_host` は trait 既定のままオーバーライドしない。
impl Module for TransformerDecoderLayer {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape)
            .forward(input, input, None, None, false, false)
    }

    fn as_transformer_decoder_layer(&self) -> Option<&TransformerDecoderLayer> {
        Some(self)
    }

    fn as_transformer_decoder_layer_mut(&mut self) -> Option<&mut TransformerDecoderLayer> {
        Some(self)
    }

    /// `forward_host` は trait 既定のまま（常に `Unsupported`）。
    /// [`Module::supports_forward_host`] を `false` へオーバーライド
    /// する（`nn::transformer_encoder_layer` と同型）。
    fn supports_forward_host(&self) -> bool {
        false
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `self_attn.*` → `multihead_attn.*` → `linear1.*` → `linear2.*` →
    /// `norm1.*` → `norm2.*` → `norm3.*` の順で、各子
    /// `Module::named_parameters()` に接頭辞を連結する（`module::prefixed`
    /// ヘルパー参照）。合計 26 テンソル（8 + 8 + 4 + 6）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = prefixed("self_attn", self.self_attn.named_parameters());
        out.extend(prefixed(
            "multihead_attn",
            self.multihead_attn.named_parameters(),
        ));
        out.extend(prefixed("linear1", self.linear1.named_parameters()));
        out.extend(prefixed("linear2", self.linear2.named_parameters()));
        out.extend(prefixed("norm1", self.norm1.named_parameters()));
        out.extend(prefixed("norm2", self.norm2.named_parameters()));
        out.extend(prefixed("norm3", self.norm3.named_parameters()));
        out
    }

    /// [`Module::set_parameter`] の実装。`strip_child_prefix` で
    /// `self_attn.`／`multihead_attn.`／`linear1.`／`linear2.`／
    /// `norm1.`／`norm2.`／`norm3.` のいずれかを剥がし、対応する子の
    /// `set_parameter` へ委譲する（`named_parameters` の接頭辞契約の
    /// 逆演算）。該当する接頭辞がない名前は未知名として拒否する。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        if let Some(rest) = strip_child_prefix(name, "self_attn") {
            return Module::set_parameter(&mut self.self_attn, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "multihead_attn") {
            return Module::set_parameter(&mut self.multihead_attn, rest, value);
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
        if let Some(rest) = strip_child_prefix(name, "norm3") {
            return LayerNorm::set_parameter(&mut self.norm3, rest, value);
        }
        Err(AutodiffError::InvalidArgument(format!(
            "TransformerDecoderLayer::set_parameter: no parameter named `{name}`"
        )))
    }

    /// [`Module::set_requires_grad`] の実装。7 子層すべてへ伝播する。
    /// 子はいずれも本クレート内の層（fail-closed 既定の対象外）のため
    /// 実際には常に `Ok` を返すが、`Module` trait の汎用契約に従い `?`
    /// で伝播する（`nn/transformer_encoder_layer.rs` と同方針）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        Module::set_requires_grad(&mut self.self_attn, requires_grad)?;
        Module::set_requires_grad(&mut self.multihead_attn, requires_grad)?;
        Module::set_requires_grad(&mut self.linear1, requires_grad)?;
        Module::set_requires_grad(&mut self.linear2, requires_grad)?;
        Module::set_requires_grad(&mut self.norm1, requires_grad)?;
        Module::set_requires_grad(&mut self.norm2, requires_grad)?;
        Module::set_requires_grad(&mut self.norm3, requires_grad)?;
        Ok(())
    }

    /// 7 子層はすべて private フィールドのため常に揃った値を返す
    /// （`self_attn` の値を代表として返す。`TransformerEncoderLayer::
    /// requires_grad` と同方針）。
    fn requires_grad(&self) -> bool {
        Module::requires_grad(&self.self_attn)
    }

    /// [`Module::children`] の実装。順序・名前は
    /// [`Self::named_parameters`] の接頭辞契約と一致させる。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("self_attn".to_string(), &self.self_attn as &dyn Module),
            (
                "multihead_attn".to_string(),
                &self.multihead_attn as &dyn Module,
            ),
            ("linear1".to_string(), &self.linear1 as &dyn Module),
            ("linear2".to_string(), &self.linear2 as &dyn Module),
            ("norm1".to_string(), &self.norm1 as &dyn Module),
            ("norm2".to_string(), &self.norm2 as &dyn Module),
            ("norm3".to_string(), &self.norm3 as &dyn Module),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::attention::MultiheadAttentionConfig;
    use crate::nn::norm::LAYER_NORM_DEFAULT_EPS;
    use crate::tape::Tape;

    const D_MODEL: usize = 4;
    const NUM_HEADS: usize = 2;
    const DIM_FF: usize = 8;

    fn build(seed: u64) -> TransformerDecoderLayer {
        TransformerDecoderLayer::new(
            D_MODEL,
            NUM_HEADS,
            DIM_FF,
            FeedForwardActivation::Relu,
            LAYER_NORM_DEFAULT_EPS,
            seed,
        )
        .unwrap()
    }

    fn parts_from(
        self_attn: MultiheadAttention,
        multihead_attn: MultiheadAttention,
        linear1: Linear,
        linear2: Linear,
        norm1: LayerNorm,
        norm2: LayerNorm,
        norm3: LayerNorm,
    ) -> TransformerDecoderLayerParts {
        TransformerDecoderLayerParts {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            activation: FeedForwardActivation::Relu,
        }
    }

    fn default_parts(seeds: [u64; 3]) -> TransformerDecoderLayerParts {
        let self_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, seeds[0]).unwrap();
        let multihead_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, seeds[1]).unwrap();
        let linear1 = Linear::new(D_MODEL, DIM_FF, true, seeds[2]).unwrap();
        let linear2 = Linear::new(DIM_FF, D_MODEL, true, seeds[2] + 1).unwrap();
        let norm1 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        let norm2 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        let norm3 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS).unwrap();
        parts_from(
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
        )
    }

    #[test]
    fn new_rejects_zero_d_model() {
        assert!(
            TransformerDecoderLayer::new(
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
            TransformerDecoderLayer::new(
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
            TransformerDecoderLayer::new(
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
    fn from_parameters_accepts_consistent_layers() {
        let parts = default_parts([1, 2, 3]);
        let layer = TransformerDecoderLayer::from_parameters(parts).unwrap();
        assert_eq!(layer.d_model(), D_MODEL);
        assert_eq!(layer.dim_feedforward(), DIM_FF);
    }

    #[test]
    fn from_parameters_rejects_cross_layer_shape_mismatch() {
        let mut parts = default_parts([1, 2, 3]);
        // linear2 の in_features が dim_feedforward と食い違う。
        parts.linear2 = Linear::new(DIM_FF + 1, D_MODEL, true, 9).unwrap();
        let result = TransformerDecoderLayer::from_parameters(parts);
        match result {
            Err(AutodiffError::Shape(_)) => {}
            Err(other) => panic!("Shape エラーを期待したが別の Err: {other:?}"),
            Ok(_) => panic!("linear2 の in_features 不一致は Err を返すはず"),
        }
    }

    /// `TransformerDecoderLayerVars::new`（公開 `Result` コンストラクタ）
    /// は `linear1.weight` の rank を検証してから軸へアクセスすべきで
    /// あり、rank-1／rank-0／rank-3 のいずれを渡しても panic せず
    /// `Err` を返す必要がある（`nn/transformer_encoder_layer.rs::
    /// new_vars_rejects_non_rank2_linear1_weight` と同型の回帰。
    /// codex-review／Cursor Bugbot 指摘・PR #2211）。
    #[test]
    fn new_vars_rejects_non_rank2_linear1_weight() {
        for bad_shape in [vec![D_MODEL], vec![], vec![D_MODEL, DIM_FF, 1]] {
            let tape = Tape::new();
            let self_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, 1)
                .unwrap()
                .bind(&tape);
            let multihead_attn = MultiheadAttention::new(D_MODEL, NUM_HEADS, true, 2)
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
            let norm3 = LayerNorm::new(D_MODEL, LAYER_NORM_DEFAULT_EPS)
                .unwrap()
                .bind(&tape);
            let result = TransformerDecoderLayerVars::new(TransformerDecoderLayerVarsParts {
                self_attn,
                multihead_attn,
                linear1,
                linear2,
                norm1,
                norm2,
                norm3,
                activation: FeedForwardActivation::Relu,
            });
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
    fn from_parameters_rejects_non_default_self_attn_batch_first() {
        let mut parts = default_parts([1, 2, 3]);
        let cfg = MultiheadAttentionConfig::new(D_MODEL, NUM_HEADS).with_batch_first(false);
        parts.self_attn = MultiheadAttention::from_config(&cfg, 1).unwrap();
        let result = TransformerDecoderLayer::from_parameters(parts);
        match result {
            Err(AutodiffError::InvalidArgument(_)) => {}
            other => panic!(
                "非既定 config（batch_first=false）は Err を返すはず（is_err={})",
                other.is_ok()
            ),
        }
    }

    #[test]
    fn from_parameters_rejects_non_default_multihead_attn_kdim() {
        let mut parts = default_parts([1, 2, 3]);
        let cfg = MultiheadAttentionConfig::new(D_MODEL, NUM_HEADS).with_kdim(D_MODEL + 2);
        parts.multihead_attn = MultiheadAttention::from_config(&cfg, 1).unwrap();
        let result = TransformerDecoderLayer::from_parameters(parts);
        match result {
            Err(AutodiffError::InvalidArgument(_)) => {}
            other => panic!(
                "非既定 config（kdim != d_model）は Err を返すはず（is_err={})",
                other.is_ok()
            ),
        }
    }

    #[test]
    fn from_parameters_rejects_embed_dim_mismatch_between_attentions() {
        let mut parts = default_parts([1, 2, 3]);
        parts.multihead_attn = MultiheadAttention::new(D_MODEL + 2, NUM_HEADS, true, 5).unwrap();
        let result = TransformerDecoderLayer::from_parameters(parts);
        match result {
            Err(AutodiffError::InvalidArgument(_)) => {}
            other => panic!(
                "self_attn/multihead_attn の embed_dim 不一致は Err を返すはず（is_err={})",
                other.is_ok()
            ),
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
                "multihead_attn.q_proj.weight",
                "multihead_attn.q_proj.bias",
                "multihead_attn.k_proj.weight",
                "multihead_attn.k_proj.bias",
                "multihead_attn.v_proj.weight",
                "multihead_attn.v_proj.bias",
                "multihead_attn.out_proj.weight",
                "multihead_attn.out_proj.bias",
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias",
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "norm3.weight",
                "norm3.bias",
            ]
        );
        assert_eq!(names.len(), 26);
    }

    #[test]
    fn set_parameter_delegates_to_correct_child() {
        let mut layer = build(7);
        let new_bias = Tensor::new(vec![9.0f32; D_MODEL], &[D_MODEL]).unwrap();
        layer.set_parameter("norm3.bias", new_bias.clone()).unwrap();
        assert_eq!(
            layer
                .norm3()
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
    fn forward_output_shape_allows_memory_len_ne_tgt_len() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(11);
        let bound = layer.bind(&tape);
        let tgt = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let memory =
            tape.var(&Tensor::new(vec![0.2f32; 2 * 5 * D_MODEL], &[2, 5, D_MODEL]).unwrap());
        let out = bound
            .forward(&tgt, &memory, None, None, false, false)
            .unwrap();
        assert_eq!(out.shape(), vec![2, 3, D_MODEL]);
    }

    #[test]
    fn forward_rejects_tgt_rank_mismatch() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(11);
        let bound = layer.bind(&tape);
        let tgt = tape.var(&Tensor::new(vec![0.1f32; 3 * D_MODEL], &[3, D_MODEL]).unwrap());
        let memory = tape.var(&Tensor::new(vec![0.1f32; 3 * D_MODEL], &[1, 3, D_MODEL]).unwrap());
        let err = bound
            .forward(&tgt, &memory, None, None, false, false)
            .unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::RankMismatch { .. })
        ));
    }

    #[test]
    fn forward_rejects_memory_batch_mismatch() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(11);
        let bound = layer.bind(&tape);
        let tgt = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let memory = tape.var(&Tensor::new(vec![0.1f32; 3 * D_MODEL], &[1, 3, D_MODEL]).unwrap());
        let err = bound
            .forward(&tgt, &memory, None, None, false, false)
            .unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn forward_tape_mismatch_between_tgt_and_memory() {
        let tape1 = Tape::new_with_ops(crate::default_ops::naive_ops());
        let tape2 = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(11);
        let bound = layer.bind(&tape1);
        let tgt = tape1.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let memory =
            tape2.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let err = bound
            .forward(&tgt, &memory, None, None, false, false)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::TapeMismatch));
    }

    #[test]
    fn forward_matches_manual_composition_bit_exact() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(13);
        let tgt_tensor = Tensor::new(vec![0.05f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap();
        let mem_tensor = Tensor::new(vec![0.02f32; 2 * 5 * D_MODEL], &[2, 5, D_MODEL]).unwrap();
        let tgt = tape.var(&tgt_tensor);
        let memory = tape.var(&mem_tensor);

        let bound = layer.bind(&tape);
        let via_layer = bound
            .forward(&tgt, &memory, None, None, false, false)
            .unwrap();

        // 手動合成（`forward` の実装と同一の演算列）。
        let bound2 = layer.bind(&tape);
        let self_attn_out = bound2
            .self_attn
            .forward(&tgt, &tgt, &tgt, None, false)
            .unwrap();
        let x1 = bound2
            .norm1
            .forward(&tgt.add(&self_attn_out).unwrap())
            .unwrap();
        let cross_attn_out = bound2
            .multihead_attn
            .forward(&x1, &memory, &memory, None, false)
            .unwrap();
        let x2 = bound2
            .norm2
            .forward(&x1.add(&cross_attn_out).unwrap())
            .unwrap();
        let hidden = project(&x2, &bound2.linear1, 2, 3, D_MODEL, DIM_FF, None).unwrap();
        let activated = hidden.relu();
        let ffn_out = project(&activated, &bound2.linear2, 2, 3, DIM_FF, D_MODEL, None).unwrap();
        let manual = bound2.norm3.forward(&x2.add(&ffn_out).unwrap()).unwrap();

        assert_eq!(
            via_layer.to_tensor().contiguous().as_slice().unwrap(),
            manual.to_tensor().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn forward_with_causal_and_tgt_mask_matches_manual_composition() {
        use fandhe_ai_tensor_core::Tensor as BoolTensor;
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(17);
        let tgt_tensor = Tensor::new(vec![0.05f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap();
        let mem_tensor = Tensor::new(vec![0.02f32; 2 * 4 * D_MODEL], &[2, 4, D_MODEL]).unwrap();
        let tgt = tape.var(&tgt_tensor);
        let memory = tape.var(&mem_tensor);

        let mem_mask = BoolTensor::new(vec![true; 2 * 3 * 4], &[2, 3, 4]).unwrap();

        let bound = layer.bind(&tape);
        let out = bound
            .forward(&tgt, &memory, None, Some(&mem_mask), true, false)
            .unwrap();

        let bound2 = layer.bind(&tape);
        let self_attn_out = bound2
            .self_attn
            .forward(&tgt, &tgt, &tgt, None, true)
            .unwrap();
        let x1 = bound2
            .norm1
            .forward(&tgt.add(&self_attn_out).unwrap())
            .unwrap();
        let cross_attn_out = bound2
            .multihead_attn
            .forward(&x1, &memory, &memory, Some(&mem_mask), false)
            .unwrap();
        let x2 = bound2
            .norm2
            .forward(&x1.add(&cross_attn_out).unwrap())
            .unwrap();
        let hidden = project(&x2, &bound2.linear1, 2, 3, D_MODEL, DIM_FF, None).unwrap();
        let activated = hidden.relu();
        let ffn_out = project(&activated, &bound2.linear2, 2, 3, DIM_FF, D_MODEL, None).unwrap();
        let manual = bound2.norm3.forward(&x2.add(&ffn_out).unwrap()).unwrap();

        assert_eq!(
            out.to_tensor().contiguous().as_slice().unwrap(),
            manual.to_tensor().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn attn_mask_and_is_causal_together_is_rejected() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(19);
        let bound = layer.bind(&tape);
        let tgt = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let memory =
            tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let tgt_mask = Tensor::new(vec![true; 3 * 3], &[3, 3]).unwrap();
        let err = bound
            .forward(&tgt, &memory, Some(&tgt_mask), None, true, false)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn module_forward_matches_bind_forward_with_input_as_tgt_and_memory() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(23);
        let x = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());

        let via_module = Module::forward(&layer, &tape, &x).unwrap();

        let bound = layer.bind(&tape);
        let via_bind = bound.forward(&x, &x, None, None, false, false).unwrap();

        assert_eq!(
            via_module.to_tensor().contiguous().as_slice().unwrap(),
            via_bind.to_tensor().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn backward_reaches_all_twenty_six_parameters() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let layer = build(29);
        let tgt = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * D_MODEL], &[2, 3, D_MODEL]).unwrap());
        let memory =
            tape.var(&Tensor::new(vec![0.1f32; 2 * 4 * D_MODEL], &[2, 4, D_MODEL]).unwrap());
        let bound = layer.bind(&tape);
        let out = bound
            .forward(&tgt, &memory, None, None, false, false)
            .unwrap();
        let loss = out.mean(None).unwrap();
        let grads = tape.backward(&loss).unwrap();

        assert!(grads.get(&bound.self_attn.q.weight).unwrap().is_some());
        assert!(grads.get(&bound.multihead_attn.q.weight).unwrap().is_some());
        assert!(grads.get(&bound.multihead_attn.k.weight).unwrap().is_some());
        assert!(grads.get(&bound.multihead_attn.v.weight).unwrap().is_some());
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
                .get(bound.norm3.bias.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn as_transformer_decoder_layer_hooks_recognize_this_layer_only() {
        let mut layer = build(31);
        assert!(Module::as_transformer_decoder_layer(&layer).is_some());
        assert!(Module::as_transformer_decoder_layer_mut(&mut layer).is_some());

        let linear = Linear::new(D_MODEL, D_MODEL, true, 1).unwrap();
        assert!(Module::as_transformer_decoder_layer(&linear).is_none());
    }
}
