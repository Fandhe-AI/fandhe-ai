//! 参照モデル `TransformerClassifier`（行トークンのエンコーダ分類器）の実体（イシュー
//! #2202 の examples 実装 `Transformer` を #2975〈親 #2541・Phase 11-2〉で
//! `fandhe_ai::models` として公開したもの）。
//!
//! 公開パスは `models/mod.rs` の `pub use` のみ（本モジュール自体は非公開）。公開済みの
//! `fandhe_ai::nn::Transformer`（エンコーダ・デコーダ）とは別の型で、名前の衝突を避けるため
//! `TransformerClassifier`／`TransformerClassifierConfig` と改名している。承認範囲は
//! `docs/reference-models-decision.md` §11.8。保存・読み込み経路は持たない。
//!
//! # 構成
//!
//! ```text
//! embed = Linear(in_features, embed_dim)             # [N, L, in] -> [N, L, E]
//! x = embed(x) + positional_encoding                 # 固定 sinusoidal（学習しない）
//! encoder = TransformerEncoderLayer x num_layers
//! pooled = mean(x, dim=1)                            # [N, L, E] -> [N, E]
//! head  = Linear(E, mlp_hidden) -> ReLU -> Linear(mlp_hidden, num_classes)
//! ```
//!
//! ViT ではない（patch embedding・cls token・学習可能な位置埋め込みを持たない）。
//! `Linear.weight` は `[in, out]` で PyTorch の `[out, in]` と転置の関係。
//! `TransformerEncoderLayer` は `compat::Sequential::add_transformer_encoder` の契約
//! （post-norm・`activation="relu"`・dropout 結線なし）に従う。

use crate::compat::Sequential;
use crate::optim::Adam;
use crate::{AutodiffError, Tape, Tensor, Var};

use super::train_support::{check_model_size, cross_entropy_mean, scalar_of, take_updated};

/// [`TransformerClassifier::new`] の構成値。
///
/// フィールドは 0.11 系の非破壊契約として固定する（追加・削除は破壊的変更）。
#[derive(Debug, Clone, Copy)]
pub struct TransformerClassifierConfig {
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

impl TransformerClassifierConfig {
    /// 画像の行をトークンとして扱う CIFAR-10 相当入力（`[N, 32, 96]`。`seq_len=32`・
    /// `in_features=96=3*32`）向けプリセット。`dim_feedforward`／`mlp_hidden` は
    /// `embed_dim * 2` に固定する。
    ///
    /// # Errors
    ///
    /// `embed_dim * 2` が `usize` を超える場合は `AutodiffError::InvalidArgument`。
    pub fn cifar10(
        embed_dim: usize,
        num_heads: usize,
        num_layers: usize,
        num_classes: usize,
    ) -> Result<Self, AutodiffError> {
        let ffn_dim = embed_dim.checked_mul(2).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "TransformerClassifierConfig::cifar10: embed_dim（{embed_dim}）* 2 が usize の範囲を超える"
            ))
        })?;
        Ok(TransformerClassifierConfig {
            seq_len: 32,
            in_features: 96,
            embed_dim,
            num_heads,
            num_layers,
            dim_feedforward: ffn_dim,
            mlp_hidden: ffn_dim,
            num_classes,
        })
    }
}

/// 固定 sinusoidal 位置符号（`[seq_len, embed_dim]`。学習しない定数。Vaswani et al. 2017 と
/// 同じ式: 偶数次元は `sin`、奇数次元は `cos`）。
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

/// エンコーダだけの Transformer 分類器（`compat::Sequential` 3 部品のラッパー:
/// `embed`・`encoder`・`head`。位置符号は非学習の定数 `Tensor`）。
///
/// `fandhe_ai::nn::Transformer`（エンコーダ・デコーダ）とは別の型。保存・読み込み
/// （`compat::save_model`／`load_model`）の対象ではない。
pub struct TransformerClassifier {
    config: TransformerClassifierConfig,
    embed: Sequential,
    pos_encoding: Tensor<f32>,
    encoder: Sequential,
    head: Sequential,
}

impl TransformerClassifier {
    /// `config` から分類器を構築する。`seed` は各層の初期化基準値（`wrapping_add` でずらす）。
    ///
    /// # Errors
    ///
    /// `config` のいずれかのフィールドが 0、`embed_dim` が `num_heads` で割り切れない、
    /// または `seq_len * embed_dim` が `usize` を超える場合は
    /// `AutodiffError::InvalidArgument`。
    pub fn new(config: TransformerClassifierConfig, seed: u64) -> Result<Self, AutodiffError> {
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
                "TransformerClassifier::new: config の各フィールドはいずれも 0 より大きい必要がある"
                    .to_string(),
            ));
        }
        if !config.embed_dim.is_multiple_of(config.num_heads) {
            return Err(AutodiffError::InvalidArgument(format!(
                "TransformerClassifier::new: embed_dim（{}）は num_heads（{}）で割り切れる必要がある",
                config.embed_dim, config.num_heads
            )));
        }
        // `config` は pub フィールドで任意値を構築できるため、位置符号テンソルの要素数
        // `seq_len * embed_dim` の overflow を先に検査する（wrap-around すると確保サイズと
        // 後段のインデックス計算が食い違う）。
        let pos_elems = config
            .seq_len
            .checked_mul(config.embed_dim)
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "TransformerClassifier::new: seq_len（{}）* embed_dim（{}）が usize の範囲を超える",
                    config.seq_len, config.embed_dim
                ))
            })?;
        // 確保前に総要素数の上限を検証する（巨大 config で `vec!` が capacity overflow で
        // panic／OOM するのを防ぐ）。位置符号 + embed + 各層 + head の飽和算術による見積もり。
        // 各層は重み（attention 4*e^2 + FFN 2*e*ff）に加え、projection の bias
        // （4*e + ff + e）・2 つの LayerNorm（gamma/beta 計 4*e）・層ごとのオブジェクト
        // コスト（`PER_LAYER_OVERHEAD_ELEMS`）を数える。極小次元かつ層数が巨大な config が
        // 重み項だけの見積もりをすり抜けて数百万層を確保するのを防ぐ。
        const PER_LAYER_OVERHEAD_ELEMS: usize = 4096;
        let e = config.embed_dim;
        let per_layer = e
            .saturating_mul(e)
            .saturating_mul(4)
            .saturating_add(e.saturating_mul(config.dim_feedforward).saturating_mul(2))
            .saturating_add(e.saturating_mul(9))
            .saturating_add(config.dim_feedforward)
            .saturating_add(PER_LAYER_OVERHEAD_ELEMS);
        let estimated = pos_elems
            .saturating_add(config.in_features.saturating_mul(e))
            .saturating_add(per_layer.saturating_mul(config.num_layers))
            .saturating_add(e.saturating_mul(config.mlp_hidden))
            .saturating_add(config.mlp_hidden.saturating_mul(config.num_classes));
        check_model_size("TransformerClassifier::new", estimated)?;

        let embed = Sequential::new().add_linear(config.in_features, config.embed_dim, seed)?;

        let pos_encoding_data = sinusoidal_positional_encoding(config.seq_len, config.embed_dim);
        let pos_encoding = Tensor::new(pos_encoding_data, &[config.seq_len, config.embed_dim])
            .map_err(|e| {
                AutodiffError::InvalidArgument(format!(
                    "TransformerClassifier::new: 位置符号テンソル構築に失敗: {e}"
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

        Ok(TransformerClassifier {
            config,
            embed,
            pos_encoding,
            encoder,
            head,
        })
    }

    /// 外部 `Tape` 上で推論用 forward を計算する（入力 `[N, seq_len, in_features]`、出力
    /// `[N, num_classes]`）。
    ///
    /// # Errors
    ///
    /// 入力 shape が `[N, seq_len, in_features]` でない場合は
    /// `AutodiffError::InvalidArgument`。
    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let shape = x.to_tensor().shape().to_vec();
        if shape.len() != 3
            || shape[1] != self.config.seq_len
            || shape[2] != self.config.in_features
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "TransformerClassifier::forward: 入力 shape は [N, {}, {}] である必要がある \
                 （実際: {shape:?}）",
                self.config.seq_len, self.config.in_features
            )));
        }
        let n = shape[0];
        let flat_rows = n.checked_mul(self.config.seq_len).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "TransformerClassifier::forward: n（{n}）* seq_len（{}）が usize の範囲を超える",
                self.config.seq_len
            ))
        })?;

        let flat = x.reshape(&[flat_rows, self.config.in_features])?;
        let embedded = self.embed.forward(tape, &flat)?;
        let embedded = embedded.reshape(&[n, self.config.seq_len, self.config.embed_dim])?;

        let pos_var = tape.var(&self.pos_encoding);
        let with_pos = embedded.add(&pos_var)?;

        let encoded = self.encoder.forward(tape, &with_pos)?;
        let pooled = encoded.mean(Some(1))?;
        self.head.forward(tape, &pooled)
    }

    /// `Tensor` 入出力の推論。モードは切り替えない（`compat::Sequential::predict` と同じ契約）。
    ///
    /// # Errors
    ///
    /// [`TransformerClassifier::forward`] と同じ。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let tape = crate::tape();
        let xv = tape.var(x);
        let y = self.forward(&tape, &xv)?;
        Ok(y.to_tensor())
    }

    /// 階層名付きのパラメータ一覧（`embed.`・`encoder.`・`head.` 接頭辞）。
    pub fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
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

    /// 全部品へ train/eval モードを伝える。エンコーダ層は post-norm・dropout 結線なしで
    /// mode 依存の層を持たないため、数値挙動は変わらない。
    pub fn set_training(&mut self, training: bool) {
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

    /// 現在 train モードなら `true`（`embed` の値を代表として返す）。
    pub fn training(&self) -> bool {
        self.embed.training()
    }

    /// 1 学習ステップ（tape 構築 → forward → cross entropy → backward → optimizer 適用 →
    /// パラメータ書き戻し）を実行し、損失値を返す。冒頭で train モードへ切り替える。
    /// `x` は `[N, seq_len, in_features]`、`y` は `[N]` のクラス index。
    ///
    /// # Errors
    ///
    /// 入力 shape・ラベルの不正、勾配とパラメータの数の不一致は
    /// `AutodiffError::InvalidArgument`。
    pub fn train_step(
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
                "TransformerClassifier::train_step: 入力 shape は [N, {}, {}] である必要がある \
                 （実際: {shape:?}）",
                self.config.seq_len, self.config.in_features
            )));
        }
        self.set_training(true);
        let n = shape[0];
        let flat_rows = n.checked_mul(self.config.seq_len).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "TransformerClassifier::train_step: n（{n}）* seq_len（{}）が usize の範囲を超える",
                self.config.seq_len
            ))
        })?;

        let (loss_value, updated) = {
            let tape = crate::tape();
            let xv = tape.var(x);

            let embed_bound = self.embed.bind(&tape);
            let encoder_bound = self.encoder.bind(&tape);
            let head_bound = self.head.bind(&tape);

            let flat = xv.reshape(&[flat_rows, self.config.in_features])?;
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
                    "TransformerClassifier::train_step: パラメータ数（{}）と勾配数（{}）が不一致",
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
            .apply_parameters(take_updated(&updated, &mut idx, embed_n)?)?;
        let encoder_n = self.encoder.trainable_parameters().len();
        self.encoder
            .apply_parameters(take_updated(&updated, &mut idx, encoder_n)?)?;
        let head_n = self.head.trainable_parameters().len();
        self.head
            .apply_parameters(take_updated(&updated, &mut idx, head_n)?)?;
        if idx != updated.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "TransformerClassifier::train_step: 更新後パラメータ数（{}）が消費数（{idx}）と不一致",
                updated.len()
            )));
        }

        Ok(loss_value)
    }

    /// 構成値の参照用ゲッター。
    pub fn config(&self) -> TransformerClassifierConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seq_len: usize) -> TransformerClassifierConfig {
        TransformerClassifierConfig {
            seq_len,
            in_features: 1,
            embed_dim: 1,
            num_heads: 1,
            num_layers: 1,
            dim_feedforward: 1,
            mlp_hidden: 1,
            num_classes: 1,
        }
    }

    /// 巨大 config でも panic／abort せず `InvalidArgument` を返す（正のプローブ）。
    #[test]
    fn new_rejects_huge_config_without_panic() {
        assert!(matches!(
            TransformerClassifier::new(cfg(usize::MAX), 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            TransformerClassifier::new(cfg(1 << 40), 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let mut c = cfg(4);
        c.num_layers = usize::MAX;
        assert!(matches!(
            TransformerClassifier::new(c, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        // 極小次元 + 巨大層数（重み項のみの見積もりでは上限未満になる境界）も拒否する。
        let mut c = cfg(4);
        c.num_layers = 1 << 27;
        assert!(matches!(
            TransformerClassifier::new(c, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(TransformerClassifier::new(cfg(4), 0).is_ok());
    }
}
