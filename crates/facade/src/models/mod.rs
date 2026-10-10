//! 参照モデル（PyTorch の定番モデル）の公開モジュール。
//!
//! `compat::Sequential::add_*` だけを組み合わせた薄いラッパー `Mlp`（全結合分類器）と
//! `LeNet`（Conv2d 2 層 + Dense 2 層の CNN）を公開する。実体は `examples/models/` の
//! 利用者コード（イシュー #2201）で、イシュー #2974（親 #2541・Phase 11-1）で本モジュールへ
//! 移した。承認の根拠はイシュー #2499 の所有者コメント（issuecomment-6097478475）と
//! `docs/reference-models-decision.md` §11。
//!
//! # 公開する型
//!
//! [`Mlp`]・[`LeNet`] の 2 名だけをこのモジュールから公開する（クレートルートへは
//! 再エクスポートしない。サブモジュールも非公開）。公開メソッドは `new`・`forward`・
//! `predict`・`sequential`・`sequential_mut` と、`Mlp::with_seed`・`Mlp::dropout`・
//! `LeNet::num_classes`。PyTorch との層対応表は公開せず examples 側に置く。
//!
//! # 保存と読み込み
//!
//! 専用の保存経路は持たない。`sequential()` が返す `compat::Sequential` を既存の
//! `compat::save_model`／`compat::load_model` に渡す。
//!
//! # 範囲
//!
//! `ResNet`・`TransformerClassifier` は後続のイシューで扱う（本モジュールでは公開しない）。
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

mod lenet;
mod mlp;

pub use lenet::LeNet;
pub use mlp::Mlp;
