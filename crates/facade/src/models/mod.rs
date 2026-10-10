//! 参照モデル（PyTorch の定番モデル）の公開モジュール。
//!
//! `compat::Sequential::add_*` だけを組み合わせた薄いラッパー `Mlp`（全結合分類器）と
//! `LeNet`（Conv2d 2 層 + Dense 2 層の CNN）に加え、残差加算を手組みする `ResNet`
//! （CIFAR 版 `6n+2`）・`ResNetBlock` と、エンコーダだけの分類器 `TransformerClassifier`
//! （構成値 `TransformerClassifierConfig`）を公開する。実体は `examples/models/` の利用者コード
//! （イシュー #2201・#2202）で、イシュー #2974（`Mlp`／`LeNet`。親 #2541・Phase 11-1）と
//! #2975（`ResNet` 系。Phase 11-2）で本モジュールへ移した。承認の根拠はイシュー #2499 の
//! 所有者コメント（issuecomment-6097478475）と `docs/reference-models-decision.md` §11。
//!
//! # 公開する型
//!
//! [`Mlp`]・[`LeNet`]・[`ResNet`]・[`ResNetBlock`]・[`TransformerClassifier`]・
//! [`TransformerClassifierConfig`] の 6 名だけをこのモジュールから公開する（クレートルートへは
//! 再エクスポートしない。サブモジュールも非公開）。`Mlp`／`LeNet` の公開メソッドは `new`・
//! `forward`・`predict`・`sequential`・`sequential_mut` と、`Mlp::with_seed`・`Mlp::dropout`・
//! `LeNet::num_classes`。`ResNet`／`TransformerClassifier` は `new`・`forward`・`predict`・
//! `named_parameters`・`set_training`・`training`・`train_step` とゲッター
//! （`ResNet::{depth, width, num_classes, num_blocks, blocks}`・
//! `TransformerClassifier::config`・`TransformerClassifierConfig::cifar10`）、`ResNetBlock` は
//! `has_projection_shortcut` のみ。`fandhe_ai::nn::Transformer`（エンコーダ・デコーダ）とは
//! 別の型で、`use fandhe_ai::nn::*; use fandhe_ai::models::*;` を併用しても衝突しない。
//! PyTorch との層対応表は公開せず examples 側に置く。
//!
//! # 保存と読み込み
//!
//! `Mlp`／`LeNet` は `sequential()` が返す `compat::Sequential` を既存の
//! `compat::save_model`／`compat::load_model` に渡す。`ResNet`／`TransformerClassifier` は
//! 保存・読み込み経路を持たない（`compat::Sequential` を外へ出さず、`nn::Module` も実装しない）。
//!
//! # 範囲
//!
//! `ReferenceModule`／`Trainable` のような共通 trait や、`fit_epochs`・`accuracy` 等の学習
//! ユーティリティは公開しない（examples に残す）。
//!
//! # 使用例
//!
//! 評価モードにして推論する（`Mlp` と `LeNet`）。
//!
//! ```
//! use fandhe_ai::models::{LeNet, Mlp};
//! use fandhe_ai::Tensor;
//!
//! let mut mlp = Mlp::new(8, &[16, 4], 3, 0.1).unwrap();
//! mlp.sequential_mut().eval();
//! let x = Tensor::<f32>::new(vec![0.5; 2 * 8], &[2, 8]).unwrap();
//! let y = mlp.predict(&x).unwrap();
//! assert_eq!(y.shape(), &[2, 3]);
//!
//! // LeNet の入力は `[N, 1, 28, 28]`。
//! let mut lenet = LeNet::new(10, 1).unwrap();
//! lenet.sequential_mut().eval();
//! assert_eq!(lenet.num_classes(), 10);
//! let x = Tensor::<f32>::new(vec![0.1; 28 * 28], &[1, 1, 28, 28]).unwrap();
//! let y = lenet.predict(&x).unwrap();
//! assert_eq!(y.shape(), &[1, 10]);
//! ```
//!
//! `ResNet` と `TransformerClassifier` の推論と 1 学習ステップ。
//!
//! ```
//! use fandhe_ai::models::{ResNet, TransformerClassifier, TransformerClassifierConfig};
//! use fandhe_ai::optim::{Adam, AdamConfig};
//! use fandhe_ai::Tensor;
//!
//! // ResNet（depth=8 は 6n+2 の最小構成）。BatchNorm を汚さないよう eval にして推論する。
//! let mut resnet = ResNet::new(8, 2, 10, 0).unwrap();
//! resnet.set_training(false);
//! assert!(!resnet.training());
//! assert_eq!(resnet.num_blocks(), 3);
//! assert!(!resnet.blocks()[0].has_projection_shortcut());
//! let x = Tensor::<f32>::new(vec![0.1; 3 * 8 * 8], &[1, 3, 8, 8]).unwrap();
//! assert_eq!(resnet.predict(&x).unwrap().shape(), &[1, 10]);
//!
//! // TransformerClassifier（pub フィールドの構成値から構築）。
//! let cfg = TransformerClassifierConfig {
//!     seq_len: 4,
//!     in_features: 6,
//!     embed_dim: 8,
//!     num_heads: 2,
//!     num_layers: 1,
//!     dim_feedforward: 16,
//!     mlp_hidden: 16,
//!     num_classes: 3,
//! };
//! let mut tc = TransformerClassifier::new(cfg, 0).unwrap();
//! let x = Tensor::<f32>::new(vec![0.1; 2 * 4 * 6], &[2, 4, 6]).unwrap();
//! assert_eq!(tc.predict(&x).unwrap().shape(), &[2, 3]);
//! let preset = TransformerClassifierConfig::cifar10(16, 2, 1, 10).unwrap();
//! assert_eq!((preset.seq_len, preset.in_features), (32, 96));
//!
//! // 1 学習ステップ（損失は有限値）。
//! let y = Tensor::<i32>::new(vec![0, 2], &[2]).unwrap();
//! let mut opt = Adam::new(AdamConfig::default()).unwrap();
//! let loss = tc.train_step(&x, &y, &mut opt).unwrap();
//! assert!(loss.is_finite());
//! ```

mod lenet;
mod mlp;
mod resnet;
mod train_support;
mod transformer_classifier;

pub use lenet::LeNet;
pub use mlp::Mlp;
pub use resnet::ResNet;
pub use resnet::ResNetBlock;
pub use transformer_classifier::TransformerClassifier;
pub use transformer_classifier::TransformerClassifierConfig;
