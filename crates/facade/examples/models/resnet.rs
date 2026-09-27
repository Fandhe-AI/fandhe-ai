//! 参照モデル定義: `ResNet`（CIFAR 版。イシュー #2202・親 #2190）。
//!
//! ## 位置づけ
//!
//! `mlp.rs`／`lenet.rs`（イシュー #2201）と同じ理由・同じ制約で本
//! ファイルも**単独で完結する**（`super::reference_module` への参照
//! のみ許容——取り込み側 crate root が `#[path]` で `reference_module`
//! を同名モジュールとして宣言し、`super::reference_module` がどの
//! 取り込み先でも同じ場所を指すようにする契約。
//! `docs/reference-models-decision.md` #2202 節参照）。
//!
//! `facade` の公開面（`crates/facade/src/`）は変更していない
//! （`mlp.rs` モジュール doc「位置づけ（重要）」節と同じ保留理由。
//! `ResNetBlock`／`ResNet` の `pub use` は承認事項として保留する）。
//!
//! ## 構成（He et al. 2016 §4.2。CIFAR 版 `6n+2` 構成）
//!
//! ```text
//! stem  = Conv2d(3, w, 3, pad=1) -> BN(w) -> ReLU
//! stage1: BasicBlock(w -> w)       x n（stride 1）
//! stage2: BasicBlock(w -> 2w, s=2) + BasicBlock(2w -> 2w) x (n-1)
//! stage3: BasicBlock(2w -> 4w, s=2) + BasicBlock(4w -> 4w) x (n-1)
//! head  = AdaptiveAvgPool2d(1,1) -> Flatten -> Linear(4w, num_classes)
//! ```
//!
//! `depth = 6n + 2`（`n >= 1`）に限る。`BasicBlock`（`ResNetBlock`）の
//! main 経路は `conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN`、
//! shortcut は stride≠1 またはチャネル数変化時のみ
//! `conv1x1(stride) -> BN`（option B。それ以外は identity）。
//! 出力は `relu(main + shortcut)`。
//!
//! ## 重みレイアウトの契約
//!
//! `Conv2d`／`BatchNorm2d`／`Linear` の shape 契約は `lenet.rs`
//! モジュール doc と同一（`Conv2d.weight` は PyTorch と同一 shape、
//! `Linear.weight` は `[in, out]` で PyTorch `[out, in]` と転置の
//! 関係）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::Adam;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

use super::reference_module::{ReferenceModule, Trainable, cross_entropy_mean, scalar_of};

/// BatchNorm2d の `eps`（PyTorch `nn.BatchNorm2d` 既定）。
const BN_EPS: f32 = 1e-5;
/// BatchNorm2d の `momentum`（PyTorch 既定）。
const BN_MOMENTUM: f32 = 0.1;

/// 1 つの BasicBlock（`main` 経路 + 任意の `shortcut` 経路）。
///
/// `main`・`shortcut` はいずれも `compat::Sequential`（直列専用。
/// `docs/reference-models-decision.md` #2202 節「`compat::Sequential`
/// は直列専用」）で組み、residual 加算（`Var::add`）だけを
/// [`ResNetBlock::forward`] で手組みする。
pub struct ResNetBlock {
    main: Sequential,
    shortcut: Option<Sequential>,
}

impl ResNetBlock {
    /// `in_channels -> out_channels`（`stride` 付き）の BasicBlock を
    /// 構築する。`in_channels != out_channels` または `stride != 1` の
    /// ときのみ shortcut に `conv1x1(stride) + BN` を持つ（option B）。
    pub fn new(
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

    /// shortcut が projection（`conv1x1 + BN`）か identity かを返す
    /// （テストの構造検証・example の表示用）。
    pub fn has_projection_shortcut(&self) -> bool {
        self.shortcut.is_some()
    }
}

impl ReferenceModule for ResNetBlock {
    /// `relu(main(x) + shortcut(x))`（`shortcut` が `None` のときは
    /// identity）。`Var` は `Copy`（`SequentialVars::forward_with_precision`
    /// の `let mut current = *input;` と同じ前提）のため、identity 分岐は
    /// 単純な値コピーで表現できる。
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

    /// `main`（常に存在）の `training` フラグを代表値として返す
    /// （`set_training` が `main`／`shortcut` を常に同時に切り替える
    /// ため、両者は一致する契約。`ReferenceModule::is_training` doc
    /// 参照）。
    fn is_training(&self) -> bool {
        self.main.training()
    }
}

/// CIFAR 版 ResNet（`depth = 6n + 2`）。`stem` → `blocks`（3 ステージ
/// 合計 `3n` block）→ `head` の合成モデル。residual 加算があるため
/// `compat::Sequential` 単体では表現できず、複数の `Sequential` を
/// 持つラッパーとして構成する（モジュール doc 参照）。
pub struct ResNet {
    stem: Sequential,
    blocks: Vec<ResNetBlock>,
    head: Sequential,
    depth: usize,
    width: usize,
    num_classes: usize,
}

impl ResNet {
    /// `depth`（`6n+2`。`n >= 1`）・`width`（stage1 の出力チャネル数。
    /// stage2/3 はそれぞれ `2*width`／`4*width`）・`num_classes` から
    /// ResNet を構築する。`seed` は各層の初期化基準値
    /// （`wrapping_add` でずらす。`mlp.rs`／`lenet.rs` と同じ方式）。
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
        // `width` は `ResNet::new` の公開引数（呼び出し元が任意の usize
        // を渡せる）であり、`width * 2`・`width * 4`（stage2/3 の出力
        // チャネル数）は debug では panic・release では wrap-around して
        // 誤ったチャネル数のまま構築が進みうる。`checked_mul` で表現
        // できない場合は `InvalidArgument` にする（Codex レビュー
        // 指摘・イシュー #2202 PR #2325）。`3 * n`（block 総数。`Vec::
        // with_capacity` の容量ヒント）も同じ理由で検査する。
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

    /// 推論の入口（`compat::Sequential::predict` と同じ eval モード
    /// 前提。`ReferenceModule::forward` と同じ演算列を `Tensor` 直接
    /// 入出力で呼びたい場合に使う）。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        let tape = fandhe_ai::tape();
        let xv = tape.var(x);
        let y = ReferenceModule::forward(self, &tape, &xv)?;
        Ok(y.to_tensor())
    }

    /// 構成値（example・テストの表示用）。
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
    /// stage 合計の block 数（`3n`。テストの構造検証用）。
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
    /// 個々の block への参照（テストの構造検証用）。
    pub fn blocks(&self) -> &[ResNetBlock] {
        &self.blocks
    }
}

impl ReferenceModule for ResNet {
    fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
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

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out: Vec<(String, &Tensor<f32>)> = self
            .stem
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("stem.{k}"), v))
            .collect();
        for (i, block) in self.blocks.iter().enumerate() {
            out.extend(
                ReferenceModule::named_parameters(block)
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

    fn set_training(&mut self, training: bool) {
        if training {
            self.stem.train();
            self.head.train();
        } else {
            self.stem.eval();
            self.head.eval();
        }
        for block in &mut self.blocks {
            ReferenceModule::set_training(block, training);
        }
    }

    /// `stem`（常に存在）の `training` フラグを代表値として返す
    /// （`ResNetBlock::is_training` と同じ考え方）。
    fn is_training(&self) -> bool {
        self.stem.training()
    }
}

impl Trainable for ResNet {
    /// `stem`・各 block の `main`／`shortcut`・`head` を同一 `tape` に
    /// `bind` し、`SequentialVars::forward`（train forward。`BatchNorm`
    /// の running stats を `RefCell` 越しに更新する経路）で合成 forward
    /// を組む。`ReferenceModule::forward`（推論用。生 `Sequential::
    /// forward`）とは別経路（`docs/reference-models-decision.md`
    /// #2202 節「`compat::Sequential` は直列専用」参照）。
    fn train_step(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<i32>,
        opt: &mut Adam,
    ) -> Result<f32, AutodiffError> {
        // `ResNet::forward` と同じ入力 shape 契約をここでも検証する
        // （`train_step` は `forward` を経由せず直接 `stem_bound.forward`
        // を呼ぶため、検証しないと不正な shape が bind 後の演算列の
        // どこかで初めて失敗し、エラーメッセージが分かりにくくなる。
        // `Transformer::train_step` は同型の検証を持つが `ResNet::
        // train_step` は欠けていた非対称性。Codex レビュー指摘・
        // イシュー #2202 PR #2325）。
        let input_shape = x.shape();
        if input_shape.len() != 4 || input_shape[1] != 3 {
            return Err(AutodiffError::InvalidArgument(format!(
                "ResNet::train_step: 入力 shape は [N, 3, H, W] である必要がある（実際: \
                 {input_shape:?}）"
            )));
        }

        // `train_step` は `fit_epochs` の内部ループからだけでなく trait
        // メソッドとして直接呼び出しうる（`Trainable` は pub trait）。
        // eval モードのモデルへ直接呼ぶと BatchNorm が eval math（running
        // stats のみ使用・更新なし）のまま backward することになり、
        // 学習として意味をなさない。冒頭で train モードを明示する
        // （`fit_epochs` はループ前に一度呼ぶだけのため、`train_step`
        // 単体呼び出しの契約としてここでも明示する。Codex レビュー
        // 指摘・イシュー #2202 PR #2325）。
        ReferenceModule::set_training(self, true);

        let (loss_value, updated) = {
            let tape = fandhe_ai::tape();
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
            .apply_parameters(updated[idx..idx + stem_n].to_vec())?;
        idx += stem_n;
        for block in &mut self.blocks {
            let main_n = block.main.trainable_parameters().len();
            block
                .main
                .apply_parameters(updated[idx..idx + main_n].to_vec())?;
            idx += main_n;
            if let Some(shortcut) = &mut block.shortcut {
                let shortcut_n = shortcut.trainable_parameters().len();
                shortcut.apply_parameters(updated[idx..idx + shortcut_n].to_vec())?;
                idx += shortcut_n;
            }
        }
        let head_n = self.head.trainable_parameters().len();
        self.head
            .apply_parameters(updated[idx..idx + head_n].to_vec())?;
        idx += head_n;
        debug_assert_eq!(idx, updated.len());

        Ok(loss_value)
    }
}
