//! PyTorch `nn.MSELoss`／`nn.CrossEntropyLoss` 等に相当する損失構造体 14 種と、
//! その引数型 5 種を `fandhe_ai_autodiff::nn::loss` から再エクスポートする
//! **純再エクスポートモジュール**（`crate::nn::init`・`crate::nn::kv_cache` と同型。
//! facade 独自の型・関数は持ち込まない）。
//!
//! イシュー #2602（親 #2600・ルート #2499 の一括承認。
//! `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`。
//! `docs/compat-api-scope.md` §5 経路 2。決定記録は
//! `docs/facade-nn-loss-structs-exposure-decision.md` §4）。
//!
//! # 公開する 19 名
//!
//! - 損失構造体 14: `MseLoss`・`L1Loss`・`HuberLoss`・`SmoothL1Loss`・`BceLoss`・
//!   `BceWithLogitsLoss`・`CrossEntropyLoss`・`NllLoss`・`KlDivLoss`・
//!   `CosineEmbeddingLoss`・`MarginRankingLoss`・`TripletMarginLoss`・
//!   `PoissonNllLoss`・`CtcLoss`
//! - 引数型 5: `Reduction`・`CrossEntropyOptions`・`TripletMarginOptions`・
//!   `PoissonNllOptions`・`CtcLossOptions`
//!
//! 引数型は構造体のコンストラクタや `Var` の公開済みメソッドが引数に取るため、
//! 同じ経路で出さないと facade 単独では呼べない。`Reduction` とオプション型は
//! **この経路のみ**で公開し、crate ルートや `nn` 直下には出さない。
//!
//! `CrossEntropyLoss` は `new`／`Default` を持たず、pub フィールドの構造体リテラルで
//! 構築する。`compat::Loss`（`compile()`／`fit` 用）とは別レイヤである。入力の形状・値域の
//! 検査は委譲先（autodiff）が担う。公開後は pub フィールドを持つ構造体と `Copy` 実装が
//! 契約として固定される（決定記録 §5）。
//!
//! # 利用例
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::nn::loss::{CrossEntropyLoss, MseLoss, Reduction};
//!
//! let tape = fandhe_ai::tape();
//! let pred = tape.var(&Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap());
//! let target = tape.var(&Tensor::new(vec![0.0f32, 0.0], &[2]).unwrap());
//! let loss = MseLoss::new(Reduction::Sum).forward(&pred, &target).unwrap();
//! assert_eq!(loss.to_tensor().host_slice().into_owned(), [5.0f32]);
//!
//! let _ce = CrossEntropyLoss { class_dim: 1, reduction: Reduction::Mean };
//! ```

pub use fandhe_ai_autodiff::nn::loss::{
    BceLoss, BceWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss, CtcLoss, HuberLoss,
    KlDivLoss, L1Loss, MarginRankingLoss, MseLoss, NllLoss, PoissonNllLoss, SmoothL1Loss,
    TripletMarginLoss,
};
pub use fandhe_ai_autodiff::nn::loss::{
    CrossEntropyOptions, CtcLossOptions, PoissonNllOptions, Reduction, TripletMarginOptions,
};
