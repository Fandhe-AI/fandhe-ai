//! optimizer 公開面（イシュー #961・親 #960。設計判断
//! `docs/facade-optimizer-promotion-decision.md` §4 案 A「素の再エクスポート」）。
//!
//! `facade` が唯一のサポート対象公開 API 面（`docs/compat-api-scope.md` §0）
//! であるにもかかわらず、これまで optimizer（SGD・AdamW・gradient
//! clipping・LR スケジューラ）は内部クレート `fandhe_ai_autodiff` にしか
//! 公開されておらず、利用者は `fandhe_ai_autodiff` へ直接依存するか
//! 手動 SGD（`examples/training_loop.rs`）を書くしかなかった。本モジュール
//! はその欠落を埋め、内部で配置が不統一な 2 か所
//! （`fandhe_ai_autodiff::optim::{Sgd, SgdConfig}` と
//! `fandhe_ai_autodiff::nn::optim::{AdamW, ClipGradResult, clip_grad_norm,
//! global_grad_norm, ConstantLr, LrScheduler, StepLr}`）を `fandhe_ai::optim`
//! という単一の入口へ吸収する。
//!
//! `fandhe_ai::optim` は REQ-9 の 2026-08-29 追記（正本 spec
//! `docs/spec/04-requirements.md:211-212`。実装リポ #984／#986）で、
//! `tape()`系・`compat` と並ぶ確定入口となった（`docs/compat-api-scope.md` §0）。
//!
//! # 呼び出し文脈（`compat::Sequential` との位置対応契約）
//!
//! [`crate::optim::Sgd::step`]／[`crate::optim::AdamW::step`] が受け取る `params`／`grads` の順序は、
//! [`crate::compat::Sequential::trainable_parameters`]（更新前パラメータ
//! 列）と [`crate::compat::SequentialVars::trainable_grads`]（対応する
//! 勾配列）が返す列の位置に対応させる契約になっている（`Sequential` 側の
//! doc 参照）。本モジュールはこの契約を変更せず、値型・純関数をそのまま
//! 再エクスポートするだけの薄い層である。
//!
//! # 適用順序契約
//!
//! 1 学習ステップは必ず
//! `backward → (AMP 導入後の unscale) → clip → optimizer step`
//! の順で実行する（`fandhe_ai_autodiff::nn::optim` モジュール doc から転記）。
//! 損失スケーリング（AMP）は現時点で未実装のため unscale ステップは
//! 存在しないが、将来 AMP を導入する際も「clip は unscale 後の生勾配に
//! 対してのみ適用する」契約を崩さない（clip 前に scale が残っていると
//! `max_norm` の意味が変わり、意図しない過剰クリップ・過小クリップを
//! 招くため）。
//!
//! # REQ-12 との整合（`Tape`／`BackendOps` 非依存）
//!
//! 本モジュールが再エクスポートする型・関数はいずれも `Tape`／`Var`／
//! `BackendOps` に依存しない値型・純関数である（`params`／`grads` を
//! `&Tensor<f32>` の参照列として受け取り、更新後 `Tensor<f32>` の列を
//! 返す関数型 API）。newtype でラップせず素の再エクスポートに留めるのは、
//! ラップしても迂回経路を持たない値型には `BackendOps` 注入の懸念が
//! 生じないため（`crate::Tape` のように `new_with_ops` を隠す必要がない。
//! `docs/facade-optimizer-promotion-decision.md` §4.2）。この構造は
//! `tests/api_surface.rs` の optim 固有検査（純再エクスポートであること・
//! 昇格元公開面と 1 対 1 であること）で機械的に固定する。
//!
//! # 内部配置の不統一・シグネチャ差異について
//!
//! [`crate::optim::Sgd`] は `fandhe_ai_autodiff::optim`、[`crate::optim::AdamW`] 等は
//! `fandhe_ai_autodiff::nn::optim` と、内部クレート側の配置は歴史的経緯
//! （親 #192 の並行実装）により不統一だが、本モジュールでは単一の
//! `fandhe_ai::optim` 入口へ吸収し利用者からは意識させない。一方で
//! [`crate::optim::Sgd::step`] は `&[&Tensor<f32>]` 2 本（`params`・`grads`）を、
//! [`crate::optim::AdamW::step`] は `&[(&Tensor<f32>, &Tensor<f32>)]`（tuple 列）を
//! 引数に取るというシグネチャ形の相違は**本モジュールでは統一しない**
//! （親 #192 の統合判断待ち。`docs/facade-optimizer-promotion-decision.md`
//! §4.3）。将来統一する場合は破壊的変更になる。
//!
//! # デバイス常駐更新との違い（誤認防止）
//!
//! 本モジュールの再エクスポートはホスト側 `Tensor<f32>` を介した
//! optimizer step であり、ステップごとのホスト⇔デバイス往復コストは
//! 本再エクスポートでは解消しない。デバイス常駐のパラメータ更新経路は
//! 別に存在する（[`crate::DeviceParamStore`]／
//! [`crate::Tape::step_device_param_store`]。イシュー #935／#954・
//! `docs/device-resident-update-design.md`）。`DeviceParamStore` は
//! `Tape` を引数に取る状態機械であり本モジュールの値型群とは性質が
//! 異なるため、意図的に本モジュールへは含めない（root 再エクスポート
//! のまま）。

// `pub use` は 1 文 1 行を維持する（複数行折返し禁止。`tests/api_surface.rs`
// が `pub use` を行単位（`trimmed.starts_with("pub use")`）で走査する
// 契約に合わせる。`src/lib.rs` 冒頭コメントと同じ理由）。
pub use fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig};
pub use fandhe_ai_autodiff::nn::optim::{ClipGradResult, clip_grad_norm, global_grad_norm};
pub use fandhe_ai_autodiff::nn::optim::{ConstantLr, LrScheduler, StepLr};
pub use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
