//! `Transformer` Module（イシュー #2165。親 #2131。PyTorch
//! `nn.Transformer` 相当の encoder-decoder 全体構築層）。
//!
//! **設計方針（新規 `Op`／`BackendOps` メソッド／VJP／カーネルを追加
//! しない）**: [`crate::nn::TransformerEncoderLayer`]・
//! [`crate::nn::TransformerDecoderLayer`]（いずれも既存部品の合成の
//! みで実装済み）・[`crate::nn::LayerNorm`]（終端正規化）の合成として
//! 組み立てる。中間コンテナ型（`TransformerEncoder`／
//! `TransformerDecoder` スタック）は公開せず、`Vec<TransformerEncoderLayer>`／
//! `Vec<TransformerDecoderLayer>` を直接フィールドとして持つ
//! （対象外。実装計画 §2.2「中間型の公開はスコープ外」参照）。
//!
//! **構成（PyTorch `nn.Transformer` と同じ。最終 LayerNorm を 2 つ
//! 含む）**: `encoder_layers: Vec<TransformerEncoderLayer>` +
//! `encoder_norm: LayerNorm` → `decoder_layers: Vec<TransformerDecoderLayer>`
//! + `decoder_norm: LayerNorm`。
//!
//! **forward 順序**: ①`src` を `encoder_layers` の各層へ順に通す
//! （mask は `src_mask`・非 causal） → ②`encoder_norm` → `memory` →
//! ③`tgt` を `decoder_layers` の各層へ順に通す（`tgt_mask`・
//! `memory_mask`・`tgt_is_causal`・`memory_is_causal = false` 固定） →
//! ④`decoder_norm` → 出力 `[B, T, E]`。
//!
//! **対象外（PR 本文に記載・`out-of-scope-tracking.md`）**:
//! `src_is_causal`／`memory_is_causal`（`Transformer::forward` 引数
//! として。明示 mask で代替できる）・`src_key_padding_mask` 等・
//! カスタム encoder／decoder の注入・facade 公開（承認事項。
//! `crates/facade/src/lib.rs` の `TransformerDecoderHoldDoctestGuard`
//! で保留固定）。
//!
//! **`Module` trait との関係**: `forward` は `src = tgt = input`・
//! mask なし・非 causal を表す（`nn/transformer_encoder_layer.rs`・
//! `nn/transformer_decoder_layer.rs` の単一入力慣習を踏襲）。
//! `supports_forward_host` は `false`。学習可能パラメータの認識には
//! 専用の [`crate::nn::module::Module::as_transformer`] フックを使う。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::nn::init::{
    TRANSFORMER_DEC_STACK_SEED_SALT, TRANSFORMER_ENC_STACK_SEED_SALT, derive_seed,
};
use crate::nn::module::{Module, prefixed, strip_child_prefix};
use crate::nn::norm::{LAYER_NORM_DEFAULT_EPS, LayerNorm, LayerNormVars};
use crate::nn::transformer_decoder_layer::{TransformerDecoderLayer, TransformerDecoderLayerVars};
use crate::nn::transformer_encoder_layer::{
    FeedForwardActivation, TransformerEncoderLayer, TransformerEncoderLayerVars,
};
use crate::tape::Tape;
use crate::var::Var;

/// [`Transformer::new`] の構築時オプション。関連関数の引数が 8 個
/// （self を含めない）になり clippy の既定閾値（7）を超えるため、
/// `#[allow(clippy::too_many_arguments)]` ではなく本ビルダー構造体で
/// 引数をまとめる（実装計画 §2.2。`MultiheadAttentionConfig` と同じ
/// ビルダー方針。フィールドは private で `#[non_exhaustive]` 相当の
/// 非破壊拡張性を持たせる）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TransformerConfig {
    d_model: usize,
    num_heads: usize,
    num_encoder_layers: usize,
    num_decoder_layers: usize,
    dim_feedforward: usize,
    activation: FeedForwardActivation,
    eps: f32,
}

impl TransformerConfig {
    /// PyTorch `nn.Transformer` の既定値（`num_encoder_layers=6`・
    /// `num_decoder_layers=6`・`dim_feedforward=2048`・
    /// `activation="relu"`）で初期化する。`eps` は
    /// [`crate::nn::norm::LAYER_NORM_DEFAULT_EPS`]（`1e-5`）。
    /// `with_*` メソッドで個別に上書きする。
    pub fn new(d_model: usize, num_heads: usize) -> TransformerConfig {
        TransformerConfig {
            d_model,
            num_heads,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            dim_feedforward: 2048,
            activation: FeedForwardActivation::Relu,
            eps: LAYER_NORM_DEFAULT_EPS,
        }
    }

    /// encoder 層数を上書きする（既定 6）。
    pub fn with_num_encoder_layers(mut self, num_encoder_layers: usize) -> TransformerConfig {
        self.num_encoder_layers = num_encoder_layers;
        self
    }

    /// decoder 層数を上書きする（既定 6）。
    pub fn with_num_decoder_layers(mut self, num_decoder_layers: usize) -> TransformerConfig {
        self.num_decoder_layers = num_decoder_layers;
        self
    }

    /// FFN 中間次元を上書きする（既定 2048）。
    pub fn with_dim_feedforward(mut self, dim_feedforward: usize) -> TransformerConfig {
        self.dim_feedforward = dim_feedforward;
        self
    }

    /// FFN 活性化関数を上書きする（既定 `Relu`）。
    pub fn with_activation(mut self, activation: FeedForwardActivation) -> TransformerConfig {
        self.activation = activation;
        self
    }

    /// LayerNorm の `eps` を上書きする（既定 `LAYER_NORM_DEFAULT_EPS`）。
    pub fn with_eps(mut self, eps: f32) -> TransformerConfig {
        self.eps = eps;
        self
    }

    /// `new` に渡した `d_model`。
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// `new` に渡した `num_heads`。
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// encoder 層数。
    pub fn num_encoder_layers(&self) -> usize {
        self.num_encoder_layers
    }

    /// decoder 層数。
    pub fn num_decoder_layers(&self) -> usize {
        self.num_decoder_layers
    }

    /// FFN 中間次元。
    pub fn dim_feedforward(&self) -> usize {
        self.dim_feedforward
    }

    /// FFN 活性化関数。
    pub fn activation(&self) -> FeedForwardActivation {
        self.activation
    }

    /// LayerNorm の `eps`。
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

/// `Transformer` のパラメータ本体（モジュール doc「構成」参照）。
pub struct Transformer {
    encoder_layers: Vec<TransformerEncoderLayer>,
    encoder_norm: LayerNorm,
    decoder_layers: Vec<TransformerDecoderLayer>,
    decoder_norm: LayerNorm,
    d_model: usize,
}

impl Transformer {
    /// `config` と単一の呼び出し `seed` から encoder／decoder 各層を
    /// 決定的に構築する。層 `i` の構築シードは
    /// `derive_seed(derive_seed(seed, TRANSFORMER_*_STACK_SEED_SALT), i)`
    /// （`nn/init.rs` の `RNN_STACK_SEED_SALT` と同型の「2 段の
    /// `derive_seed` 合成」。`i` は encoder／decoder それぞれ独立に
    /// `0` から始まる）。`num_encoder_layers == 0`／
    /// `num_decoder_layers == 0` は拒否する。
    pub fn new(config: &TransformerConfig, seed: u64) -> Result<Self, AutodiffError> {
        if config.num_encoder_layers == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Transformer::new: num_encoder_layers must be > 0".to_string(),
            ));
        }
        if config.num_decoder_layers == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Transformer::new: num_decoder_layers must be > 0".to_string(),
            ));
        }

        let enc_stack_seed = derive_seed(seed, TRANSFORMER_ENC_STACK_SEED_SALT);
        let mut encoder_layers = Vec::with_capacity(config.num_encoder_layers);
        for i in 0..config.num_encoder_layers {
            encoder_layers.push(TransformerEncoderLayer::new(
                config.d_model,
                config.num_heads,
                config.dim_feedforward,
                config.activation,
                config.eps,
                derive_seed(enc_stack_seed, i as u64),
            )?);
        }
        let encoder_norm = LayerNorm::new(config.d_model, config.eps)?;

        let dec_stack_seed = derive_seed(seed, TRANSFORMER_DEC_STACK_SEED_SALT);
        let mut decoder_layers = Vec::with_capacity(config.num_decoder_layers);
        for i in 0..config.num_decoder_layers {
            decoder_layers.push(TransformerDecoderLayer::new(
                config.d_model,
                config.num_heads,
                config.dim_feedforward,
                config.activation,
                config.eps,
                derive_seed(dec_stack_seed, i as u64),
            )?);
        }
        let decoder_norm = LayerNorm::new(config.d_model, config.eps)?;

        Ok(Self {
            encoder_layers,
            encoder_norm,
            decoder_layers,
            decoder_norm,
            d_model: config.d_model,
        })
    }

    /// `new` に渡した `d_model`。
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// encoder 層数。
    pub fn num_encoder_layers(&self) -> usize {
        self.encoder_layers.len()
    }

    /// decoder 層数。
    pub fn num_decoder_layers(&self) -> usize {
        self.decoder_layers.len()
    }

    /// encoder 層への参照列。
    pub fn encoder_layers(&self) -> &[TransformerEncoderLayer] {
        &self.encoder_layers
    }

    /// encoder 出力の最終 LayerNorm への参照。
    pub fn encoder_norm(&self) -> &LayerNorm {
        &self.encoder_norm
    }

    /// decoder 層への参照列。
    pub fn decoder_layers(&self) -> &[TransformerDecoderLayer] {
        &self.decoder_layers
    }

    /// decoder 出力の最終 LayerNorm への参照。
    pub fn decoder_norm(&self) -> &LayerNorm {
        &self.decoder_norm
    }

    /// このステップの `tape` へ全層の学習可能パラメータを葉ノードと
    /// して登録し、`forward` を呼べる [`TransformerVars`] を返す
    /// （`TransformerEncoderLayer::bind` と同じ「毎ステップ作り直す」
    /// 契約）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> TransformerVars<'t> {
        TransformerVars {
            encoder_layers: self.encoder_layers.iter().map(|l| l.bind(tape)).collect(),
            encoder_norm: self.encoder_norm.bind(tape),
            decoder_layers: self.decoder_layers.iter().map(|l| l.bind(tape)).collect(),
            decoder_norm: self.decoder_norm.bind(tape),
            d_model: self.d_model,
        }
    }
}

/// `Transformer::bind` が返す、1 ステップ分のテープに登録済み
/// パラメータ。フィールドは `pub` だが、構築は [`Transformer::bind`]
/// からのみ行う契約（`encoder_layers`／`decoder_layers` の要素数を
/// 独自に変える構築経路を作らないため）。`d_model` は private のため
/// 呼び出し側はリテラル構築できず、`bind` 経由の構築のみが可能。
pub struct TransformerVars<'t> {
    pub encoder_layers: Vec<TransformerEncoderLayerVars<'t>>,
    pub encoder_norm: LayerNormVars<'t>,
    pub decoder_layers: Vec<TransformerDecoderLayerVars<'t>>,
    pub decoder_norm: LayerNormVars<'t>,
    d_model: usize,
}

impl<'t> TransformerVars<'t> {
    /// `Transformer::bind` に渡した `d_model`。
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// `y = Transformer(src, tgt)`。`src: [B, S, E]`・`tgt: [B, T, E]`
    /// → `[B, T, E]`（モジュール doc「forward 順序」参照）。
    ///
    /// `memory_is_causal` は常に `false` 固定（PyTorch
    /// `nn.Transformer.forward` の既定と同じ）。`src_is_causal` は
    /// 対象外（モジュール doc「対象外」参照）。
    ///
    /// # Errors
    ///
    /// `src`／`tgt` の rank・`d_model` 不一致は各 `TransformerEncoderLayerVars::forward`／
    /// `TransformerDecoderLayerVars::forward` が検査する
    /// `AutodiffError::Shape` をそのまま伝播する。テープ不一致は
    /// `AutodiffError::TapeMismatch`。
    pub fn forward(
        &self,
        src: &Var<'t>,
        tgt: &Var<'t>,
        src_mask: Option<&Tensor<bool>>,
        tgt_mask: Option<&Tensor<bool>>,
        memory_mask: Option<&Tensor<bool>>,
        tgt_is_causal: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        let mut x = *src;
        for layer in &self.encoder_layers {
            x = layer.forward(&x, src_mask, false)?;
        }
        let memory = self.encoder_norm.forward(&x)?;

        let mut y = *tgt;
        for layer in &self.decoder_layers {
            y = layer.forward(&y, &memory, tgt_mask, memory_mask, tgt_is_causal, false)?;
        }
        self.decoder_norm.forward(&y)
    }
}

/// `src = tgt = input`（mask なし・非 causal）として `Module::forward`
/// を定義する（モジュール doc「`Module` trait との関係」参照）。
/// `forward_host` は trait 既定のままオーバーライドしない。
impl Module for Transformer {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape)
            .forward(input, input, None, None, None, false)
    }

    fn as_transformer(&self) -> Option<&Transformer> {
        Some(self)
    }

    fn as_transformer_mut(&mut self) -> Option<&mut Transformer> {
        Some(self)
    }

    /// `forward_host` は trait 既定のまま（常に `Unsupported`）。
    fn supports_forward_host(&self) -> bool {
        false
    }

    /// 命名契約: `encoder.layers.{i}.*` → `encoder.norm.*` →
    /// `decoder.layers.{i}.*` → `decoder.norm.*` の順（PyTorch
    /// `nn.Transformer` の `state_dict` キー体系に揃える）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            out.extend(prefixed(
                &format!("encoder.layers.{i}"),
                layer.named_parameters(),
            ));
        }
        out.extend(prefixed(
            "encoder.norm",
            self.encoder_norm.named_parameters(),
        ));
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            out.extend(prefixed(
                &format!("decoder.layers.{i}"),
                layer.named_parameters(),
            ));
        }
        out.extend(prefixed(
            "decoder.norm",
            self.decoder_norm.named_parameters(),
        ));
        out
    }

    /// [`Module::set_parameter`] の実装。`encoder.layers.`／
    /// `decoder.layers.` を剥がしたあと `ModuleList::set_parameter`
    /// と同型で index を parse し、範囲外なら `InvalidArgument` を
    /// 返す。`encoder.norm`／`decoder.norm` は `strip_child_prefix` で
    /// 扱う。未知の名前は `InvalidArgument` で拒否する（fail-closed。
    /// `.claude/rules/security.md` A03）。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        if let Some(rest) = strip_child_prefix(name, "encoder.norm") {
            return LayerNorm::set_parameter(&mut self.encoder_norm, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "decoder.norm") {
            return LayerNorm::set_parameter(&mut self.decoder_norm, rest, value);
        }
        if let Some(rest) = name.strip_prefix("encoder.layers.") {
            return set_parameter_in_layers(
                &mut self.encoder_layers,
                "encoder.layers",
                rest,
                value,
            );
        }
        if let Some(rest) = name.strip_prefix("decoder.layers.") {
            return set_parameter_in_layers(
                &mut self.decoder_layers,
                "decoder.layers",
                rest,
                value,
            );
        }
        Err(AutodiffError::InvalidArgument(format!(
            "Transformer::set_parameter: no parameter named `{name}`"
        )))
    }

    /// [`Module::set_requires_grad`] の実装。全層（encoder／decoder）
    /// と両終端 LayerNorm へ伝播する。子はいずれも本クレート内の層
    /// （fail-closed 既定の対象外）のため実際には常に `Ok` を返すが、
    /// `Module` trait の汎用契約に従い `?` で伝播する
    /// （`TransformerEncoderLayer::set_requires_grad` と同方針）。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        for layer in &mut self.encoder_layers {
            Module::set_requires_grad(layer, requires_grad)?;
        }
        Module::set_requires_grad(&mut self.encoder_norm, requires_grad)?;
        for layer in &mut self.decoder_layers {
            Module::set_requires_grad(layer, requires_grad)?;
        }
        Module::set_requires_grad(&mut self.decoder_norm, requires_grad)?;
        Ok(())
    }

    /// 全子層はすべて private フィールドのため常に揃った値を返す
    /// （`encoder_norm` の値を代表として返す。`num_encoder_layers`／
    /// `num_decoder_layers` はいずれも `>= 1` を構築時に検証済みで、
    /// `encoder_layers`／`decoder_layers` は空にならない）。
    fn requires_grad(&self) -> bool {
        Module::requires_grad(&self.encoder_norm)
    }

    /// [`Module::children`] の実装。順序・名前は
    /// [`Self::named_parameters`] の接頭辞契約と一致させる。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        let mut out: Vec<(String, &dyn Module)> = Vec::new();
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            out.push((format!("encoder.layers.{i}"), layer as &dyn Module));
        }
        out.push((
            "encoder.norm".to_string(),
            &self.encoder_norm as &dyn Module,
        ));
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            out.push((format!("decoder.layers.{i}"), layer as &dyn Module));
        }
        out.push((
            "decoder.norm".to_string(),
            &self.decoder_norm as &dyn Module,
        ));
        out
    }
}

/// [`Module::set_parameter`]（`Transformer` 実装）が `encoder.layers.`／
/// `decoder.layers.` の接頭辞を剥がした後の共通処理（`ModuleList::
/// set_parameter` と同型の index-parse + range-check）。`stack_label`
/// はエラーメッセージ用のラベル（`"encoder.layers"`／
/// `"decoder.layers"`）。
fn set_parameter_in_layers<M: Module>(
    layers: &mut [M],
    stack_label: &str,
    rest: &str,
    value: Tensor<f32>,
) -> Result<(), AutodiffError> {
    let (index_str, param_name) = rest.split_once('.').ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "Transformer::set_parameter: no parameter named `{stack_label}.{rest}` (expected \
             `{stack_label}.{{index}}.{{name}}`)"
        ))
    })?;
    let index: usize = index_str.parse().map_err(|_| {
        AutodiffError::InvalidArgument(format!(
            "Transformer::set_parameter: no parameter named `{stack_label}.{rest}` \
             (`{index_str}` is not a valid layer index)"
        ))
    })?;
    match layers.get_mut(index) {
        Some(layer) => layer.set_parameter(param_name, value),
        None => Err(AutodiffError::InvalidArgument(format!(
            "Transformer::set_parameter: no parameter named `{stack_label}.{rest}` (index \
             {index} out of range; {stack_label} has {} layers)",
            layers.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    fn config() -> TransformerConfig {
        TransformerConfig::new(4, 2)
            .with_num_encoder_layers(2)
            .with_num_decoder_layers(2)
            .with_dim_feedforward(8)
    }

    #[test]
    fn config_defaults_match_pytorch() {
        let c = TransformerConfig::new(8, 4);
        assert_eq!(c.num_encoder_layers(), 6);
        assert_eq!(c.num_decoder_layers(), 6);
        assert_eq!(c.dim_feedforward(), 2048);
        assert_eq!(c.activation(), FeedForwardActivation::Relu);
        assert_eq!(c.eps(), LAYER_NORM_DEFAULT_EPS);
    }

    #[test]
    fn new_rejects_zero_num_encoder_layers() {
        let c = config().with_num_encoder_layers(0);
        assert!(Transformer::new(&c, 1).is_err());
    }

    #[test]
    fn new_rejects_zero_num_decoder_layers() {
        let c = config().with_num_decoder_layers(0);
        assert!(Transformer::new(&c, 1).is_err());
    }

    #[test]
    fn named_parameters_has_expected_prefixes_and_total_count() {
        let model = Transformer::new(&config(), 7).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        // encoder: 2 層 * 16 テンソル + encoder.norm 2 = 34
        // decoder: 2 層 * 26 テンソル + decoder.norm 2 = 54
        assert_eq!(names.len(), 2 * 16 + 2 + 2 * 26 + 2);
        assert_eq!(names[0], "encoder.layers.0.self_attn.q_proj.weight");
        assert!(names.contains(&"encoder.norm.weight".to_string()));
        assert!(names.contains(&"decoder.layers.1.multihead_attn.out_proj.bias".to_string()));
        assert!(names.contains(&"decoder.norm.bias".to_string()));
    }

    #[test]
    fn set_parameter_delegates_to_indexed_layer_and_norm() {
        let mut model = Transformer::new(&config(), 7).unwrap();
        let new_bias = Tensor::new(vec![9.0f32; 4], &[4]).unwrap();
        model
            .set_parameter("decoder.norm.bias", new_bias.clone())
            .unwrap();
        assert_eq!(
            model
                .decoder_norm()
                .bias()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            new_bias.contiguous().as_slice().unwrap()
        );

        let new_norm1_bias = Tensor::new(vec![3.0f32; 4], &[4]).unwrap();
        model
            .set_parameter("encoder.layers.1.norm1.bias", new_norm1_bias.clone())
            .unwrap();
        assert_eq!(
            model.encoder_layers()[1]
                .norm1()
                .bias()
                .unwrap()
                .contiguous()
                .as_slice()
                .unwrap(),
            new_norm1_bias.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn set_parameter_rejects_out_of_range_layer_index() {
        let mut model = Transformer::new(&config(), 7).unwrap();
        let dummy = Tensor::new(vec![0.0f32; 4], &[4]).unwrap();
        let err = model
            .set_parameter("encoder.layers.99.norm1.bias", dummy)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn set_parameter_rejects_non_numeric_layer_index() {
        let mut model = Transformer::new(&config(), 7).unwrap();
        let dummy = Tensor::new(vec![0.0f32; 4], &[4]).unwrap();
        let err = model
            .set_parameter("decoder.layers.bogus.norm1.bias", dummy)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn set_parameter_rejects_unknown_name() {
        let mut model = Transformer::new(&config(), 7).unwrap();
        let dummy = Tensor::new(vec![0.0f32; 4], &[4]).unwrap();
        let err = model.set_parameter("bogus.weight", dummy).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_output_shape_matches_manual_composition_bit_exact() {
        let tape = Tape::new_with_ops(naive_ops());
        let model = Transformer::new(&config(), 13).unwrap();
        let src = tape.var(&Tensor::new(vec![0.05f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        let tgt = tape.var(&Tensor::new(vec![0.02f32; 2 * 5 * 4], &[2, 5, 4]).unwrap());

        let bound = model.bind(&tape);
        let out = bound.forward(&src, &tgt, None, None, None, false).unwrap();
        assert_eq!(out.shape(), vec![2, 5, 4]);

        // 手動合成（`TransformerVars::forward` の実装と同一の演算列）。
        let bound2 = model.bind(&tape);
        let mut x = src;
        for layer in &bound2.encoder_layers {
            x = layer.forward(&x, None, false).unwrap();
        }
        let memory = bound2.encoder_norm.forward(&x).unwrap();
        let mut y = tgt;
        for layer in &bound2.decoder_layers {
            y = layer
                .forward(&y, &memory, None, None, false, false)
                .unwrap();
        }
        let manual = bound2.decoder_norm.forward(&y).unwrap();

        assert_eq!(
            out.to_tensor().contiguous().as_slice().unwrap(),
            manual.to_tensor().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn module_forward_matches_bind_forward_with_input_as_src_and_tgt() {
        let tape = Tape::new_with_ops(naive_ops());
        let model = Transformer::new(&config(), 17).unwrap();
        let x = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());

        let via_module = Module::forward(&model, &tape, &x).unwrap();
        let bound = model.bind(&tape);
        let via_bind = bound.forward(&x, &x, None, None, None, false).unwrap();

        assert_eq!(
            via_module.to_tensor().contiguous().as_slice().unwrap(),
            via_bind.to_tensor().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn backward_reaches_all_leaves() {
        let tape = Tape::new_with_ops(naive_ops());
        let model = Transformer::new(&config(), 19).unwrap();
        let src = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        let tgt = tape.var(&Tensor::new(vec![0.1f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        let bound = model.bind(&tape);
        let out = bound.forward(&src, &tgt, None, None, None, false).unwrap();
        let loss = out.mean(None).unwrap();
        let grads = tape.backward(&loss).unwrap();

        assert!(
            grads
                .get(&bound.encoder_layers[0].self_attn.q.weight)
                .unwrap()
                .is_some()
        );
        assert!(
            grads
                .get(bound.encoder_norm.weight.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
        assert!(
            grads
                .get(&bound.decoder_layers[1].multihead_attn.k.weight)
                .unwrap()
                .is_some()
        );
        assert!(
            grads
                .get(bound.decoder_norm.bias.as_ref().unwrap())
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn different_layers_get_independent_seeds() {
        let model = Transformer::new(&config(), 23).unwrap();
        let l0 = model.encoder_layers()[0]
            .linear1()
            .weight()
            .contiguous()
            .as_slice()
            .unwrap()
            .to_vec();
        let l1 = model.encoder_layers()[1]
            .linear1()
            .weight()
            .contiguous()
            .as_slice()
            .unwrap()
            .to_vec();
        assert_ne!(l0, l1);
    }

    #[test]
    fn same_seed_reconstructs_bit_identical_model() {
        let model_a = Transformer::new(&config(), 29).unwrap();
        let model_b = Transformer::new(&config(), 29).unwrap();
        assert_eq!(
            model_a.encoder_layers()[0]
                .linear1()
                .weight()
                .contiguous()
                .as_slice()
                .unwrap(),
            model_b.encoder_layers()[0]
                .linear1()
                .weight()
                .contiguous()
                .as_slice()
                .unwrap()
        );
        assert_eq!(
            model_a.decoder_layers()[1]
                .linear2()
                .weight()
                .contiguous()
                .as_slice()
                .unwrap(),
            model_b.decoder_layers()[1]
                .linear2()
                .weight()
                .contiguous()
                .as_slice()
                .unwrap()
        );
    }

    #[test]
    fn as_transformer_hooks_recognize_this_layer_only() {
        let mut model = Transformer::new(&config(), 31).unwrap();
        assert!(Module::as_transformer(&model).is_some());
        assert!(Module::as_transformer_mut(&mut model).is_some());

        let layer = TransformerEncoderLayer::new(
            4,
            2,
            8,
            FeedForwardActivation::Relu,
            LAYER_NORM_DEFAULT_EPS,
            1,
        )
        .unwrap();
        assert!(Module::as_transformer(&layer).is_none());
    }
}
