//! PyTorch `nn.MSELoss`／`nn.CrossEntropyLoss` 等に相当する損失構造体 14 種と、
//! その引数型 10 種（`Reduction`＋オプション型 9）を `fandhe_ai_autodiff::nn::loss` から再エクスポートする
//! **純再エクスポートモジュール**（`crate::nn::init`・`crate::nn::kv_cache` と同型。
//! facade 独自の型・関数は持ち込まない）。
//!
//! イシュー #2602（親 #2600・ルート #2499 の一括承認。
//! `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`。
//! `docs/compat-api-scope.md` §5 経路 2。決定記録は
//! `docs/facade-nn-loss-structs-exposure-decision.md` §4）。
//!
//! # 公開する 24 名
//!
//! - 損失構造体 14: `MseLoss`・`L1Loss`・`HuberLoss`・`SmoothL1Loss`・`BceLoss`・
//!   `BceWithLogitsLoss`・`CrossEntropyLoss`・`NllLoss`・`KlDivLoss`・
//!   `CosineEmbeddingLoss`・`MarginRankingLoss`・`TripletMarginLoss`・
//!   `PoissonNllLoss`・`CtcLoss`
//! - 引数型 10: `Reduction`・オプション型 9 = `CrossEntropyOptions`・`TripletMarginOptions`・
//!   `PoissonNllOptions`・`CtcLossOptions`（#2602）と、`BceWithLogitsOptions`・
//!   `GaussianNllOptions`・`MultiMarginOptions`・`MultiLabelSoftMarginOptions`・
//!   `SigmoidFocalLossOptions`（#2854）
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
//!
//! # `Var` の損失メソッド 3 本（イシュー #2677 の承認形）
//!
//! `Var::hinge_embedding_loss`・`Var::soft_margin_loss`・`Var::multilabel_margin_loss` は
//! 引数の `Reduction` を本モジュールの `Reduction` で名指しする（root には出さない）。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::nn::loss::Reduction;
//!
//! let tape = fandhe_ai::tape();
//! let x = tape.var(&Tensor::new(vec![0.5f32, 1.5], &[2]).unwrap());
//! let y = Tensor::new(vec![1.0f32, -1.0], &[2]).unwrap();
//! // hinge: y=1 は x、y=-1 は max(0, margin - x)。Sum = 0.5 + 0 = 0.5。
//! let hinge = x.hinge_embedding_loss(&y, 1.0, Reduction::Sum).unwrap();
//! assert_eq!(hinge.to_tensor().host_slice().into_owned(), [0.5f32]);
//! // soft margin は ln(1 + exp(-y*x)) の和（有限・正）。
//! let soft = x.soft_margin_loss(&y, Reduction::Mean).unwrap();
//! assert!(soft.to_tensor().host_slice()[0] > 0.0);
//! // multilabel margin: 行 [0.1, 0.2, 0.4, 0.8]、target [3, 0, -1, 1]（クラス 3 と 0 が正例）。
//! let z = tape.var(&Tensor::new(vec![0.1f32, 0.2, 0.4, 0.8], &[1, 4]).unwrap());
//! let t = Tensor::new(vec![3i32, 0, -1, 1], &[1, 4]).unwrap();
//! let ml = z.multilabel_margin_loss(&t, Reduction::Mean).unwrap();
//! assert!((ml.to_tensor().host_slice()[0] - 0.85).abs() < 1e-6);
//! ```

//! # `Var` の損失メソッド 5 本（イシュー #2854 の承認形）
//!
//! `Var::bce_with_logits_loss_with`・`gaussian_nll_loss`・`multi_margin_loss`・
//! `multilabel_soft_margin_loss`・`sigmoid_focal_loss` が取るオプション型 5 つは、本モジュールの
//! 1 経路だけで公開する（承認根拠: ルート #2499 の
//! `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`。
//! 形の正は `docs/facade-nn-loss-structs-exposure-decision.md` §11）。オプション型は
//! `XOptions::default()` にビルダーを連鎖して構築し、値の妥当性検査は呼び出し時に型付きエラーで行う。
//! `bce_with_logits_loss_with` だけ引数順が `(target, reduction, options)` である。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::nn::loss::{
//!     BceWithLogitsOptions, GaussianNllOptions, MultiLabelSoftMarginOptions, MultiMarginOptions,
//!     Reduction, SigmoidFocalLossOptions,
//! };
//!
//! let tape = fandhe_ai::tape();
//! let x = tape.var(&Tensor::new(vec![0.5f32, -0.5, 1.0, 0.2], &[2, 2]).unwrap());
//! let t = tape.var(&Tensor::new(vec![1.0f32, 0.0, 1.0, 0.0], &[2, 2]).unwrap());
//! let pw = Tensor::new(vec![2.0f32, 1.0], &[2]).unwrap();
//! let opts = BceWithLogitsOptions::default().pos_weight(pw);
//! assert!(x.bce_with_logits_loss_with(&t, Reduction::Mean, &opts).is_ok());
//!
//! let v = tape.var(&Tensor::new(vec![1.0f32, 2.0, 1.5, 0.5], &[2, 2]).unwrap());
//! let g = GaussianNllOptions::default().full(true).eps(1e-5);
//! assert!(x.gaussian_nll_loss(&t, &v, &g, Reduction::Sum).is_ok());
//!
//! let idx = Tensor::new(vec![0i32, 1], &[2]).unwrap();
//! let mm = MultiMarginOptions::default().p(2).margin(0.5);
//! assert!(x.multi_margin_loss(&idx, &mm, Reduction::Mean).is_ok());
//!
//! let tgt = Tensor::new(vec![1.0f32, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
//! let ml = MultiLabelSoftMarginOptions::default();
//! assert!(x.multilabel_soft_margin_loss(&tgt, &ml, Reduction::Mean).is_ok());
//!
//! let fl = SigmoidFocalLossOptions::default().alpha(None).gamma(0.0);
//! assert!(x.sigmoid_focal_loss(&tgt, &fl, Reduction::Sum).is_ok());
//! ```

pub use fandhe_ai_autodiff::nn::loss::{
    BceLoss, BceWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss, CtcLoss, HuberLoss,
    KlDivLoss, L1Loss, MarginRankingLoss, MseLoss, NllLoss, PoissonNllLoss, SmoothL1Loss,
    TripletMarginLoss,
};
pub use fandhe_ai_autodiff::nn::loss::{
    CrossEntropyOptions, CtcLossOptions, PoissonNllOptions, Reduction, TripletMarginOptions,
};
// #2854: `Var` の損失メソッド 5 本が引数に取るオプション型 5 つ。承認はルート #2499 の
// https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061
// （決定記録 `docs/facade-nn-loss-structs-exposure-decision.md` §11）。接頭辞は他と同じ。
pub use fandhe_ai_autodiff::nn::loss::{
    BceWithLogitsOptions, GaussianNllOptions, MultiLabelSoftMarginOptions, MultiMarginOptions,
    SigmoidFocalLossOptions,
};
