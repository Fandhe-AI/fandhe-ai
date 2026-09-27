//! 参照モデル定義: `Transformer`（イシュー #2202・親 #2190）。
//!
//! ## 位置づけ
//!
//! `resnet.rs` と同じ理由・同じ制約で本ファイルも**単独で完結する**
//! （`super::reference_module` 参照のみ許容。`resnet.rs` モジュール
//! doc「位置づけ」節参照）。`facade` の公開面は変更していない
//! （`ResNetBlock`／`ResNet` と同様、`Transformer` の `pub use` も
//! 承認事項として保留する）。
//!
//! ## 構成
//!
//! ```text
//! embed = Linear(in_features, embed_dim)             # [N, L, in] -> [N, L, E]
//! x = embed(x) + positional_encoding                 # 固定 sinusoidal（学習しない）
//! encoder = TransformerEncoderLayer x num_layers      # 内部 FFN が MLP 層に相当
//! pooled = mean(x, dim=1)                             # [N, L, E] -> [N, E]
//! head  = Linear(E, mlp_hidden) -> ReLU -> Linear(mlp_hidden, num_classes)
//! ```
//!
//! **ViT ではない**（patch embedding・cls token・学習可能な位置埋め込み
//! を持たない。Vision Transformer は本イシューのスコープ外）。
//! `TransformerConfig::cifar10` は画像の行をトークンとして扱う
//! （`seq_len=32`・`in_features=96=3*32`。`synthetic_cifar.rs::
//! to_row_tokens` が `[N,3,32,32]` を `[N,32,96]` へ並べ替える）。
//!
//! ## 重みレイアウトの契約
//!
//! `embed`／`head` の `Linear.weight` は fandhe 側 `[in, out]`
//! （PyTorch `[out, in]` と転置の関係。`mlp.rs` と同じ契約）。
//! `TransformerEncoderLayer` 内部の重みは
//! `add_transformer_encoder`（`crates/facade/src/compat/sequential.rs`）
//! の契約（post-norm・`activation="relu"`・dropout 結線なし）に従う。

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::Adam;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

use super::reference_module::{ReferenceModule, Trainable, cross_entropy_mean, scalar_of};

/// `Transformer::new` の構成値。
#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    /// 系列長（トークン数）。
    pub seq_len: usize,
    /// 入力特徴次元（embed 前）。
    pub in_features: usize,
    /// 埋め込み次元（`d_model`）。
    pub embed_dim: usize,
    /// self-attention のヘッド数（`embed_dim % num_heads == 0` 必須）。
    pub num_heads: usize,
    /// `TransformerEncoderLayer` の積層数。
    pub num_layers: usize,
    /// 各層 FFN の中間次元。
    pub dim_feedforward: usize,
    /// 分類 head（MLP）の中間次元。
    pub mlp_hidden: usize,
    /// 分類クラス数。
    pub num_classes: usize,
}

impl TransformerConfig {
    /// 合成 CIFAR-10 相当（行トークン化。`seq_len=32`・
    /// `in_features=96`）向けプリセット。`dim_feedforward`／
    /// `mlp_hidden` は `embed_dim * 2` に固定する（学習可能パラメータ数を
    /// 小さく保ち、debug ビルドでのテスト所要時間予算に収める狙い。
    /// `docs/reference-models-decision.md` #2202 節参照）。
    pub fn cifar10(
        embed_dim: usize,
        num_heads: usize,
        num_layers: usize,
        num_classes: usize,
    ) -> Self {
        TransformerConfig {
            seq_len: 32,
            in_features: 96,
            embed_dim,
            num_heads,
            num_layers,
            dim_feedforward: embed_dim * 2,
            mlp_hidden: embed_dim * 2,
            num_classes,
        }
    }
}

/// 固定 sinusoidal 位置符号（`[seq_len, embed_dim]`。学習しない定数。
/// Vaswani et al. 2017 と同じ式: 偶数次元は `sin`、奇数次元は `cos`）。
fn sinusoidal_positional_encoding(seq_len: usize, embed_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * embed_dim];
    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let exponent = 2.0 * (i / 2) as f32 / embed_dim as f32;
            let denom = 10000f32.powf(exponent);
            let angle = pos as f32 / denom;
            out[pos * embed_dim + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    out
}

/// Transformer 参照実装（`compat::Sequential` 3 部品のラッパー:
/// `embed`・`encoder`・`head`。位置符号は非学習の定数 `Tensor`）。
pub struct Transformer {
    config: TransformerConfig,
    embed: Sequential,
    pos_encoding: Tensor<f32>,
    encoder: Sequential,
    head: Sequential,
}

impl Transformer {
    /// `config` から Transformer を構築する。`seed` は各層の初期化基準値
    /// （`wrapping_add` でずらす）。
    pub fn new(config: TransformerConfig, seed: u64) -> Result<Self, AutodiffError> {
        if config.seq_len == 0
            || config.in_features == 0
            || config.embed_dim == 0
            || config.num_heads == 0
            || config.num_layers == 0
            || config.dim_feedforward == 0
            || config.mlp_hidden == 0
            || config.num_classes == 0
        {
            return Err(AutodiffError::InvalidArgument(
                "Transformer::new: config の各フィールドはいずれも 0 より大きい必要がある"
                    .to_string(),
            ));
        }
        if !config.embed_dim.is_multiple_of(config.num_heads) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Transformer::new: embed_dim（{}）は num_heads（{}）で割り切れる必要がある",
                config.embed_dim, config.num_heads
            )));
        }

        let embed = Sequential::new().add_linear(config.in_features, config.embed_dim, seed)?;

        let pos_encoding_data = sinusoidal_positional_encoding(config.seq_len, config.embed_dim);
        let pos_encoding = Tensor::new(pos_encoding_data, &[config.seq_len, config.embed_dim])
            .map_err(|e| {
                AutodiffError::InvalidArgument(format!(
                    "Transformer::new: 位置符号テンソル構築に失敗: {e}"
                ))
            })?;

        let mut encoder = Sequential::new();
        for i in 0..config.num_layers {
            encoder = encoder.add_transformer_encoder(
                config.embed_dim,
                config.num_heads,
                config.dim_feedforward,
                seed.wrapping_add(1000 + i as u64),
            )?;
        }

        let head = Sequential::new()
            .add_linear(config.embed_dim, config.mlp_hidden, seed.wrapping_add(2000))?
            .add_relu()
            .add_linear(
                config.mlp_hidden,
                config.num_classes,
                seed.wrapping_add(2001),
            )?;

        Ok(Transformer {
            config,
            embed,
            pos_encoding,
            encoder,
            head,
        })
    }

    /// 推論の入口（`ResNet::predict` と同型）。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let tape = fandhe_ai::tape();
        let xv = tape.var(x);
        let y = ReferenceModule::forward(self, &tape, &xv)?;
        Ok(y.to_tensor())
    }

    /// 構成値（example・テストの表示用）。
    pub fn config(&self) -> TransformerConfig {
        self.config
    }
}

impl ReferenceModule for Transformer {
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let shape = x.to_tensor().shape().to_vec();
        if shape.len() != 3
            || shape[1] != self.config.seq_len
            || shape[2] != self.config.in_features
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Transformer::forward: 入力 shape は [N, {}, {}] である必要がある \
                 （実際: {shape:?}）",
                self.config.seq_len, self.config.in_features
            )));
        }
        let n = shape[0];

        let flat = x.reshape(&[n * self.config.seq_len, self.config.in_features])?;
        let embedded = self.embed.forward(tape, &flat)?;
        let embedded = embedded.reshape(&[n, self.config.seq_len, self.config.embed_dim])?;

        let pos_var = tape.var(&self.pos_encoding);
        let with_pos = embedded.add(&pos_var)?;

        let encoded = self.encoder.forward(tape, &with_pos)?;
        let pooled = encoded.mean(Some(1))?;
        self.head.forward(tape, &pooled)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out: Vec<(String, &Tensor<f32>)> = self
            .embed
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("embed.{k}"), v))
            .collect();
        out.extend(
            self.encoder
                .named_parameters()
                .into_iter()
                .map(|(k, v)| (format!("encoder.{k}"), v)),
        );
        out.extend(
            self.head
                .named_parameters()
                .into_iter()
                .map(|(k, v)| (format!("head.{k}"), v)),
        );
        out
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.embed.train();
            self.encoder.train();
            self.head.train();
        } else {
            self.embed.eval();
            self.encoder.eval();
            self.head.eval();
        }
    }
}

impl Trainable for Transformer {
    /// `embed`・`encoder`・`head` を同一 `tape` に `bind` し、
    /// `SequentialVars::forward`（train forward）で合成 forward を組む
    /// （`ResNet::train_step` と同じ設計）。
    fn train_step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<i32>,
        opt: &mut Adam,
    ) -> Result<f32, AutodiffError> {
        let shape = x.shape().to_vec();
        if shape.len() != 3
            || shape[1] != self.config.seq_len
            || shape[2] != self.config.in_features
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Transformer::train_step: 入力 shape は [N, {}, {}] である必要がある \
                 （実際: {shape:?}）",
                self.config.seq_len, self.config.in_features
            )));
        }
        let n = shape[0];

        let (loss_value, updated) = {
            let tape = fandhe_ai::tape();
            let xv = tape.var(x);

            let embed_bound = self.embed.bind(&tape);
            let encoder_bound = self.encoder.bind(&tape);
            let head_bound = self.head.bind(&tape);

            let flat = xv.reshape(&[n * self.config.seq_len, self.config.in_features])?;
            let embedded = embed_bound.forward(&tape, &flat)?;
            let embedded = embedded.reshape(&[n, self.config.seq_len, self.config.embed_dim])?;
            let pos_var = tape.var(&self.pos_encoding);
            let with_pos = embedded.add(&pos_var)?;
            let encoded = encoder_bound.forward(&tape, &with_pos)?;
            let pooled = encoded.mean(Some(1))?;
            let logits = head_bound.forward(&tape, &pooled)?;

            let loss = cross_entropy_mean(&tape, &logits, y, self.config.num_classes)?;
            let loss_value = scalar_of(&loss.to_tensor())?;
            let grads = tape.backward(&loss)?;

            let mut param_refs: Vec<&Tensor<f32>> = self.embed.trainable_parameters();
            param_refs.extend(self.encoder.trainable_parameters());
            param_refs.extend(self.head.trainable_parameters());

            let mut grad_refs: Vec<&Tensor<f32>> = embed_bound.trainable_grads(&grads)?;
            grad_refs.extend(encoder_bound.trainable_grads(&grads)?);
            grad_refs.extend(head_bound.trainable_grads(&grads)?);

            if param_refs.len() != grad_refs.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Transformer::train_step: パラメータ数（{}）と勾配数（{}）が不一致",
                    param_refs.len(),
                    grad_refs.len()
                )));
            }
            let pairs: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                param_refs.into_iter().zip(grad_refs).collect();
            let updated = opt.step(&pairs)?;
            (loss_value, updated)
        };

        let mut idx = 0usize;
        let embed_n = self.embed.trainable_parameters().len();
        self.embed
            .apply_parameters(updated[idx..idx + embed_n].to_vec())?;
        idx += embed_n;
        let encoder_n = self.encoder.trainable_parameters().len();
        self.encoder
            .apply_parameters(updated[idx..idx + encoder_n].to_vec())?;
        idx += encoder_n;
        let head_n = self.head.trainable_parameters().len();
        self.head
            .apply_parameters(updated[idx..idx + head_n].to_vec())?;
        idx += head_n;
        debug_assert_eq!(idx, updated.len());

        Ok(loss_value)
    }
}
