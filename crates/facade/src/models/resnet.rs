//! 参照モデル `ResNet`（CIFAR 版 `6n+2`・BasicBlock・option B shortcut）の実体（イシュー
//! #2202 の examples 実装を #2975〈親 #2541・Phase 11-2〉で `fandhe_ai::models` として公開
//! したもの）。
//!
//! 公開パスは `models/mod.rs` の `pub use` のみ（本モジュール自体は非公開）。承認範囲は
//! `docs/reference-models-decision.md` §11.8: `ResNet` は `new`・`forward`・`predict`・
//! `named_parameters`・`set_training`・`training`・`train_step` とゲッター、`ResNetBlock` は
//! 型と `has_projection_shortcut` だけを公開する（ブロックの `new`／`forward` 等は非公開）。
//! 保存・読み込み経路は持たない。
//!
//! # 構成（He et al. 2016 §4.2。CIFAR 版 `6n+2` 構成）
//!
//! ```text
//! stem  = Conv2d(3, w, 3, pad=1) -> BN(w) -> ReLU
//! stage1: BasicBlock(w -> w)        x n（stride 1）
//! stage2: BasicBlock(w -> 2w, s=2)  + BasicBlock(2w -> 2w) x (n-1)
//! stage3: BasicBlock(2w -> 4w, s=2) + BasicBlock(4w -> 4w) x (n-1)
//! head  = AdaptiveAvgPool2d(1,1) -> Flatten -> Linear(4w, num_classes)
//! ```
//!
//! `depth = 6n + 2`（`n >= 1`）に限る。ブロックの main 経路は
//! `conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN`、shortcut は stride≠1 またはチャネル数
//! 変化時のみ `conv1x1(stride) -> BN`（option B。それ以外は identity）。出力は
//! `relu(main + shortcut)`。`Conv2d.weight` は PyTorch と同一 shape、`Linear.weight` は
//! `[in, out]` で PyTorch の `[out, in]` と転置の関係。

use crate::compat::Sequential;
use crate::optim::Adam;
use crate::{AutodiffError, Tape, Tensor, Var};

use super::train_support::{check_model_size, cross_entropy_mean, scalar_of, take_updated};

/// BatchNorm2d の `eps`（PyTorch `nn.BatchNorm2d` 既定）。
const BN_EPS: f32 = 1e-5;
/// BatchNorm2d の `momentum`（PyTorch 既定）。
const BN_MOMENTUM: f32 = 0.1;

/// 1 つの BasicBlock（`main` 経路 + 任意の `shortcut` 経路）。
///
/// [`ResNet::blocks`] から観察するだけの型で、公開コンストラクタは持たない
/// （公開するのは `has_projection_shortcut` のみ）。residual 加算だけを手組みし、
/// 各経路は直列専用の `compat::Sequential` で組む。
pub struct ResNetBlock {
    main: Sequential,
    shortcut: Option<Sequential>,
}

impl ResNetBlock {
    /// `in_channels -> out_channels`（`stride` 付き）の BasicBlock を構築する。
    /// `in_channels != out_channels` または `stride != 1` のときのみ shortcut に
    /// `conv1x1(stride) + BN` を持つ（option B）。`ResNet::new` からのみ呼ばれる。
    fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        if in_channels == 0 || out_channels == 0 || stride == 0 {
            return Err(AutodiffError::InvalidArgument(
                "ResNetBlock::new: in_channels・out_channels・stride はいずれも 0 より \
                 大きい必要がある"
                    .to_string(),
            ));
        }

        let main = Sequential::new()
            .add_conv2d(
                in_channels,
                out_channels,
                [3, 3],
                [stride, stride],
                [1, 1],
                [1, 1],
                1,
                seed,
            )?
            .add_batch_norm2d(out_channels, BN_EPS, BN_MOMENTUM)?
            .add_relu()
            .add_conv2d(
                out_channels,
                out_channels,
                [3, 3],
                [1, 1],
                [1, 1],
                [1, 1],
                1,
                seed.wrapping_add(1),
            )?
            .add_batch_norm2d(out_channels, BN_EPS, BN_MOMENTUM)?;

        let shortcut = if stride != 1 || in_channels != out_channels {
            Some(
                Sequential::new()
                    .add_conv2d(
                        in_channels,
                        out_channels,
                        [1, 1],
                        [stride, stride],
                        [0, 0],
                        [1, 1],
                        1,
                        seed.wrapping_add(2),
                    )?
                    .add_batch_norm2d(out_channels, BN_EPS, BN_MOMENTUM)?,
            )
        } else {
            None
        };

        Ok(ResNetBlock { main, shortcut })
    }

    /// shortcut が projection（`conv1x1 + BN`）なら `true`、identity なら `false`。
    pub fn has_projection_shortcut(&self) -> bool {
        self.shortcut.is_some()
    }

    /// `relu(main(x) + shortcut(x))`（`shortcut` が `None` のときは identity）。
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let main_out = self.main.forward(tape, x)?;
        let shortcut_out = match &self.shortcut {
            Some(s) => s.forward(tape, x)?,
            None => *x,
        };
        let summed = main_out.add(&shortcut_out)?;
        Ok(summed.relu())
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out: Vec<(String, &Tensor<f32>)> = self
            .main
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("main.{k}"), v))
            .collect();
        if let Some(shortcut) = &self.shortcut {
            out.extend(
                shortcut
                    .named_parameters()
                    .into_iter()
                    .map(|(k, v)| (format!("shortcut.{k}"), v)),
            );
        }
        out
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.main.train();
            if let Some(shortcut) = &mut self.shortcut {
                shortcut.train();
            }
        } else {
            self.main.eval();
            if let Some(shortcut) = &mut self.shortcut {
                shortcut.eval();
            }
        }
    }
}

/// CIFAR 版 ResNet（`depth = 6n + 2`）。`stem` → `blocks`（3 ステージ合計 `3n` block）→
/// `head` の合成モデル。residual 加算があるため `compat::Sequential` 単体では表現できず、
/// 複数の `Sequential` を持つラッパーとして構成する。
///
/// 保存・読み込み（`compat::save_model`／`load_model`）の対象ではない。
pub struct ResNet {
    stem: Sequential,
    blocks: Vec<ResNetBlock>,
    head: Sequential,
    depth: usize,
    width: usize,
    num_classes: usize,
}

impl ResNet {
    /// `depth`（`6n+2`。`n >= 1`）・`width`（stage1 の出力チャネル数。stage2/3 はそれぞれ
    /// `2*width`／`4*width`）・`num_classes` から ResNet を構築する。`seed` は各層の初期化
    /// 基準値（層ごとに `wrapping_add` でずらす）。入力は `[N, 3, H, W]` 前提。
    ///
    /// # Errors
    ///
    /// `width`・`num_classes` が 0、`depth` が `6n+2`（`n >= 1`）でない、またはチャネル数の
    /// 乗算が `usize` を超える場合は `AutodiffError::InvalidArgument`。
    pub fn new(
        depth: usize,
        width: usize,
        num_classes: usize,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        if width == 0 || num_classes == 0 {
            return Err(AutodiffError::InvalidArgument(
                "ResNet::new: width・num_classes はいずれも 0 より大きい必要がある".to_string(),
            ));
        }
        if depth < 8 || !(depth - 2).is_multiple_of(6) {
            return Err(AutodiffError::InvalidArgument(format!(
                "ResNet::new: depth は 6n+2（n >= 1。8, 14, 20, ...）である必要がある \
                 （実際: {depth}）"
            )));
        }
        let n = (depth - 2) / 6;
        // 公開引数 `width` は任意の usize を取りうるため、チャネル数の乗算は
        // `checked_mul` で検証する（wrap-around したまま構築が進むのを防ぐ）。
        let width_x2 = width.checked_mul(2).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "ResNet::new: width（{width}）* 2 が usize の範囲を超える"
            ))
        })?;
        let width_x4 = width.checked_mul(4).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "ResNet::new: width（{width}）* 4 が usize の範囲を超える"
            ))
        })?;
        let total_blocks = 3usize.checked_mul(n).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "ResNet::new: block 総数（3 * n。n={n}）が usize の範囲を超える"
            ))
        })?;

        // 確保前に総パラメータ数の上限を検証する（巨大 depth／width で `Vec::with_capacity`
        // や各層の確保が panic／OOM するのを防ぐ）。block あたり 3x3 conv 2 枚の最大
        // チャネル（4*width）基準の保守的な上限見積もり。
        let per_block = width_x4
            .saturating_mul(width_x4)
            .saturating_mul(18)
            .saturating_add(width_x4.saturating_mul(8));
        let estimated = total_blocks
            .saturating_mul(per_block)
            .saturating_add(width_x4.saturating_mul(num_classes))
            .saturating_add(width.saturating_mul(27));
        check_model_size("ResNet::new", estimated)?;

        let stem = Sequential::new()
            .add_conv2d(3, width, [3, 3], [1, 1], [1, 1], [1, 1], 1, seed)?
            .add_batch_norm2d(width, BN_EPS, BN_MOMENTUM)?
            .add_relu();

        let mut blocks = Vec::with_capacity(total_blocks);
        let mut in_channels = width;
        let mut seed_ctr = seed.wrapping_add(10);
        for (stage_idx, &out_channels) in [width, width_x2, width_x4].iter().enumerate() {
            for block_idx in 0..n {
                let stride = if stage_idx > 0 && block_idx == 0 {
                    2
                } else {
                    1
                };
                blocks.push(ResNetBlock::new(
                    in_channels,
                    out_channels,
                    stride,
                    seed_ctr,
                )?);
                seed_ctr = seed_ctr.wrapping_add(10);
                in_channels = out_channels;
            }
        }

        let head = Sequential::new()
            .add_adaptive_avg_pool2d([1, 1])?
            .add_flatten(1, 3)
            .add_linear(width_x4, num_classes, seed_ctr)?;

        Ok(ResNet {
            stem,
            blocks,
            head,
            depth,
            width,
            num_classes,
        })
    }

    /// 外部 `Tape` 上で推論用 forward を計算する（入力 `[N, 3, H, W]`、出力
    /// `[N, num_classes]`）。現在の train/eval モードをそのまま使う（`BatchNorm` の
    /// running stats は train モードの forward で更新される）。
    ///
    /// # Errors
    ///
    /// 入力 shape が `[N, 3, H, W]` でない場合は `AutodiffError::InvalidArgument`。
    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let input_shape = x.to_tensor().shape().to_vec();
        if input_shape.len() != 4 || input_shape[1] != 3 {
            return Err(AutodiffError::InvalidArgument(format!(
                "ResNet::forward: 入力 shape は [N, 3, H, W] である必要がある（実際: \
                 {input_shape:?}）"
            )));
        }

        let mut current = self.stem.forward(tape, x)?;
        for block in &self.blocks {
            current = block.forward(tape, &current)?;
        }
        self.head.forward(tape, &current)
    }

    /// `Tensor` 入出力の推論。モードは切り替えない（`compat::Sequential::predict` と同じ
    /// 契約）。構築直後は train モードのため、推論前に `set_training(false)` を呼ぶこと
    /// （呼ばないと `BatchNorm` の running stats が更新される）。
    ///
    /// # Errors
    ///
    /// [`ResNet::forward`] と同じ。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let tape = crate::tape();
        let xv = tape.var(x);
        let y = self.forward(&tape, &xv)?;
        Ok(y.to_tensor())
    }

    /// 階層名付きのパラメータ一覧（`stem.`・`layer.{i}.main.`／`layer.{i}.shortcut.`・
    /// `head.` 接頭辞。`i` は全 block を通した 0 起点連番）。
    pub fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out: Vec<(String, &Tensor<f32>)> = self
            .stem
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("stem.{k}"), v))
            .collect();
        for (i, block) in self.blocks.iter().enumerate() {
            out.extend(
                block
                    .named_parameters()
                    .into_iter()
                    .map(|(k, v)| (format!("layer.{i}.{k}"), v)),
            );
        }
        out.extend(
            self.head
                .named_parameters()
                .into_iter()
                .map(|(k, v)| (format!("head.{k}"), v)),
        );
        out
    }

    /// 全部品へ train/eval モードを伝える（`BatchNorm` の running stats 更新可否を切り替える）。
    pub fn set_training(&mut self, training: bool) {
        if training {
            self.stem.train();
            self.head.train();
        } else {
            self.stem.eval();
            self.head.eval();
        }
        for block in &mut self.blocks {
            block.set_training(training);
        }
    }

    /// 現在 train モードなら `true`。`set_training` が全部品へ一様に伝播するため
    /// `stem` の値を代表として返す。
    pub fn training(&self) -> bool {
        self.stem.training()
    }

    /// 1 学習ステップ（tape 構築 → forward → cross entropy → backward → optimizer 適用 →
    /// パラメータ書き戻し）を実行し、損失値を返す。冒頭で train モードへ切り替える。
    /// `x` は `[N, 3, H, W]`、`y` は `[N]` のクラス index（`0..num_classes`）。
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
        let input_shape = x.shape();
        if input_shape.len() != 4 || input_shape[1] != 3 {
            return Err(AutodiffError::InvalidArgument(format!(
                "ResNet::train_step: 入力 shape は [N, 3, H, W] である必要がある（実際: \
                 {input_shape:?}）"
            )));
        }

        // eval モードのまま backward すると BatchNorm が running stats のみを使う eval 演算に
        // なり学習として意味をなさないため、冒頭で train モードを明示する。
        self.set_training(true);

        let (loss_value, updated) = {
            let tape = crate::tape();
            let xv = tape.var(x);

            let stem_bound = self.stem.bind(&tape);
            let mut block_bounds = Vec::with_capacity(self.blocks.len());
            for block in &self.blocks {
                let main_bound = block.main.bind(&tape);
                let shortcut_bound = block.shortcut.as_ref().map(|s| s.bind(&tape));
                block_bounds.push((main_bound, shortcut_bound));
            }
            let head_bound = self.head.bind(&tape);

            let mut current = stem_bound.forward(&tape, &xv)?;
            for (main_bound, shortcut_bound) in &block_bounds {
                let main_out = main_bound.forward(&tape, &current)?;
                let shortcut_out = match shortcut_bound {
                    Some(s) => s.forward(&tape, &current)?,
                    None => current,
                };
                current = main_out.add(&shortcut_out)?.relu();
            }
            let logits = head_bound.forward(&tape, &current)?;

            let loss = cross_entropy_mean(&tape, &logits, y, self.num_classes)?;
            let loss_value = scalar_of(&loss.to_tensor())?;
            let grads = tape.backward(&loss)?;

            let mut param_refs: Vec<&Tensor<f32>> = self.stem.trainable_parameters();
            for block in &self.blocks {
                param_refs.extend(block.main.trainable_parameters());
                if let Some(shortcut) = &block.shortcut {
                    param_refs.extend(shortcut.trainable_parameters());
                }
            }
            param_refs.extend(self.head.trainable_parameters());

            let mut grad_refs: Vec<&Tensor<f32>> = stem_bound.trainable_grads(&grads)?;
            for (main_bound, shortcut_bound) in &block_bounds {
                grad_refs.extend(main_bound.trainable_grads(&grads)?);
                if let Some(s) = shortcut_bound {
                    grad_refs.extend(s.trainable_grads(&grads)?);
                }
            }
            grad_refs.extend(head_bound.trainable_grads(&grads)?);

            if param_refs.len() != grad_refs.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ResNet::train_step: パラメータ数（{}）と勾配数（{}）が不一致",
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
        let stem_n = self.stem.trainable_parameters().len();
        self.stem
            .apply_parameters(take_updated(&updated, &mut idx, stem_n)?)?;
        for block in &mut self.blocks {
            let main_n = block.main.trainable_parameters().len();
            block
                .main
                .apply_parameters(take_updated(&updated, &mut idx, main_n)?)?;
            if let Some(shortcut) = &mut block.shortcut {
                let shortcut_n = shortcut.trainable_parameters().len();
                shortcut.apply_parameters(take_updated(&updated, &mut idx, shortcut_n)?)?;
            }
        }
        let head_n = self.head.trainable_parameters().len();
        self.head
            .apply_parameters(take_updated(&updated, &mut idx, head_n)?)?;
        if idx != updated.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "ResNet::train_step: 更新後パラメータ数（{}）が消費数（{idx}）と不一致",
                updated.len()
            )));
        }

        Ok(loss_value)
    }

    /// 構成値 `depth`（`6n+2`）。
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// 構成値 `width`（stage1 の出力チャネル数）。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 分類クラス数。
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// 全 stage 合計の block 数（`3n`）。
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// 個々の block への参照（構造の観察用）。
    pub fn blocks(&self) -> &[ResNetBlock] {
        &self.blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 巨大 depth／width でも panic／abort せず `InvalidArgument` を返す（正のプローブ）。
    #[test]
    fn new_rejects_huge_arguments_without_panic() {
        assert!(matches!(
            ResNet::new(usize::MAX - 1, 1, 1, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            ResNet::new(8, 1 << 20, 10, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            ResNet::new(8, 4, 1 << 40, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        // 小さな正常系は上限検証を通る。
        assert!(ResNet::new(8, 4, 10, 0).is_ok());
    }

    #[test]
    fn block_rejects_zero_arguments() {
        assert!(matches!(
            ResNetBlock::new(0, 4, 1, 1),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            ResNetBlock::new(4, 4, 0, 1),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn projection_block_has_twelve_parameters_with_main_and_shortcut_prefixes() {
        let block = ResNetBlock::new(4, 8, 2, 0x5555_6666).unwrap();
        assert!(block.has_projection_shortcut());
        // main（conv + BN の weight・bias を 2 段で 8）+ shortcut（conv + BN で 4）= 12。
        let params = block.named_parameters();
        assert_eq!(params.len(), 12);
        assert!(params.iter().any(|(n, _)| n.starts_with("main.")));
        assert!(params.iter().any(|(n, _)| n.starts_with("shortcut.")));
    }

    #[test]
    fn identity_block_has_no_shortcut_parameters() {
        let block = ResNetBlock::new(4, 4, 1, 1).unwrap();
        assert!(!block.has_projection_shortcut());
        assert!(
            block
                .named_parameters()
                .iter()
                .all(|(n, _)| n.starts_with("main."))
        );
    }
}
