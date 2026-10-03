//! PyTorch `torch.nn.init.*` 相当の重み初期化関数群（`fandhe_ai_autodiff::
//! nn::init`）を facade へ再エクスポートする**純再エクスポートモジュール**
//! （`crate::nn::rnn`・`crate::data`・`crate::optim` と同型。facade 独自の
//! 型・関数は持ち込まない）。
//!
//! イシュー #2504（親 #2500・ルート #2499 の一括承認〈2026-10-04〉。
//! `docs/compat-api-scope.md` §5 経路 2。決定記録は
//! `docs/facade-nn-init-exposure-decision.md`）。autodiff 側の実装は
//! イシュー #2140。
//!
//! # 公開する 13 名
//!
//! - 初期化関数 9 個: `uniform`／`normal`／`constant`／`xavier_uniform`／
//!   `xavier_normal`／`kaiming_uniform`／`kaiming_normal`／`orthogonal`／
//!   `trunc_normal`
//! - 補助 4 個: `FanMode`／`Nonlinearity`（`#[non_exhaustive]`）／
//!   `calculate_gain`／`calculate_fan_in_and_fan_out`
//!
//! `kaiming_*` は引数に `FanMode`・`Nonlinearity` を取るため、これらを
//! 公開しないと facade から呼べない。戻り値型の [`crate::Tensor`]・
//! [`crate::AutodiffError`] は facade ルートで再エクスポート済みで、内部型は
//! 露出しない。
//!
//! # 乱数源と決定性
//!
//! 乱数は**プロセスグローバル RNG**（[`crate::manual_seed`]）に従い、
//! `compat::Sequential::add_linear(.., seed)` 等の個別シード API とは独立
//! である。同一シード・同一プロセス・同一プラットフォーム内で再現可能
//! （Box–Muller 経由の関数は超越関数の実装差でプラットフォーム間の bit 同一
//! を保証しない。整数演算のみの関数はプラットフォームをまたいで bit 同一）。
//! 乱数源は xorshift64* で**暗号論的に安全ではない**ため、鍵・トークン生成
//! に使ってはならない。
//!
//! # 重みレイアウトの注意（決定記録 §2.2）
//!
//! `compat::Sequential::add_linear` の `weight` は `[in_features,
//! out_features]` で PyTorch（`[out, in]`）と転置の関係にある。そのため
//! `kaiming_*`／`xavier_*`／`calculate_fan_in_and_fan_out` に `[in, out]`
//! の形状を渡すと fan_in と fan_out の意味が入れ替わる。Conv 系は PyTorch
//! と同じレイアウトである。
//!
//! # 利用例
//!
//! ```
//! use fandhe_ai::compat::Sequential;
//! use fandhe_ai::nn::init::{constant, uniform};
//!
//! fandhe_ai::manual_seed(42);
//! let w = uniform(&[2, 3], -0.1, 0.1).unwrap();
//! let b = constant(&[3], 0.0).unwrap();
//! let mut seq = Sequential::new().add_linear(2, 3, 0).unwrap();
//! let mut sd = seq.state_dict();
//! sd.insert("0.weight".to_string(), w);
//! sd.insert("0.bias".to_string(), b);
//! seq.load_state_dict(sd).unwrap();
//! ```

pub use fandhe_ai_autodiff::nn::init::{
    FanMode, Nonlinearity, calculate_fan_in_and_fan_out, calculate_gain,
};
pub use fandhe_ai_autodiff::nn::init::{constant, normal, orthogonal, trunc_normal, uniform};
pub use fandhe_ai_autodiff::nn::init::{
    kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform,
};
