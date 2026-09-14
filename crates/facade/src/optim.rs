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
//! clip_grad_value, global_grad_norm, ConstantLr, LrScheduler, StepLr}`。
//! `clip_grad_value` は #1753・親 #1631 で追加した value 方式 gradient
//! clipping）を `fandhe_ai::optim`
//! という単一の入口へ吸収する。
//!
//! **Adam（coupled L2 weight decay。イシュー #1742・親 #1610）**:
//! [`crate::optim::Adam`]／[`crate::optim::AdamConfig`] を
//! `fandhe_ai_autodiff::nn::optim`（実体は `nn::optim::adam` モジュール）
//! から同じく素の再エクスポートで公開する。`AdamW`（decoupled）と
//! `weight_decay > 0` のとき異なる更新値を生む点は `nn::optim::adam`
//! モジュール doc・`docs/compat-feature-gap.md` §2.9 を参照。
//!
//! **LR スケジューラ拡充（式ベース 3 種。イシュー #1745・親 #1611）**:
//! [`crate::optim::CosineAnnealingLr`]／[`crate::optim::ExponentialLr`]／
//! [`crate::optim::LinearWarmupLr`] を [`crate::optim::ConstantLr`]／
//! [`crate::optim::StepLr`] と同じく `fandhe_ai_autodiff::nn::optim`
//! （実体は `nn::optim::lr_scheduler` モジュール）から素の再エクスポート
//! で公開する。いずれも [`crate::optim::LrScheduler::lr_at`] のみを持つ
//! stateless な純関数であり、状態保持型の `ReduceLROnPlateau`／
//! `OneCycleLR` は対象外（兄弟イシュー #1746／#1747 が担当）。
//!
//! `fandhe_ai::optim` は REQ-9 の 2026-08-29 追記（正本 spec
//! `docs/spec/04-requirements.md:211-212`。実装リポ #984／#986）で、
//! `tape()`系・`compat` と並ぶ確定入口となった（`docs/compat-api-scope.md` §0）。
//!
//! **AMP（損失スケーリング。イシュー #1625・#1721・本イシュー #1722）**:
//! [`crate::optim::GradScaler`]／[`crate::optim::GradScalerConfig`]／
//! [`crate::optim::UnscaleResult`]／[`crate::optim::scale_loss`]／
//! [`crate::optim::scale_grads`]／[`crate::optim::unscale_grads`]／
//! [`crate::optim::has_non_finite`] を
//! `fandhe_ai_autodiff::nn::optim`（実体は `nn::optim::amp` モジュール）から
//! 同じく素の再エクスポートで公開する。「# 適用順序契約」節を参照。
//!
//! # 呼び出し文脈（`compat::Sequential` との位置対応契約）
//!
//! [`crate::optim::Sgd::step`]／[`crate::optim::AdamW::step`]／
//! [`crate::optim::Adam::step`] が受け取る `params`／`grads` の順序は、
//! [`crate::compat::Sequential::trainable_parameters`]（更新前パラメータ
//! 列）と [`crate::compat::SequentialVars::trainable_grads`]（対応する
//! 勾配列）が返す列の位置に対応させる契約になっている（`Sequential` 側の
//! doc 参照）。本モジュールはこの契約を変更せず、値型・純関数をそのまま
//! 再エクスポートするだけの薄い層である。
//!
//! # 適用順序契約
//!
//! ## AMP を使わない場合（既存経路。無変更で動作する）
//!
//! 1 学習ステップは
//! `backward → clip → optimizer step`
//! の順で実行する。
//!
//! ## AMP（[`crate::optim::GradScaler`]）を使う場合
//!
//! 1 学習ステップは必ず次の順で実行する（`fandhe_ai_autodiff::nn::optim::amp`
//! モジュール doc から転記）:
//!
//! 1. [`crate::optim::GradScaler::scale_loss`] で loss をスケールする
//! 2. `Tape::backward` でスケール済み loss を逆伝播する
//! 3. [`crate::optim::GradScaler::unscale`]（内部で [`crate::optim::unscale_grads`]
//!    を呼ぶ）で勾配をスケールで割り戻しつつ非有限値の有無を検出する
//! 4. [`crate::optim::UnscaleResult::should_skip_step`] が `true` の場合、
//!    この step の clip・optimizer step を**両方**スキップする
//!    （`clip_grad_norm` は非有限勾配で `Err` を返す fail-closed 契約の
//!    ため、非有限検出より前に clip を呼ぶと学習ループ全体が失敗する）
//! 5. `false` の場合のみ `clip_grad_norm` → optimizer step の順に進める
//!    （「clip は unscale 後の生勾配に対してのみ適用する」契約は不変。
//!    clip 前にスケールが残っていると `max_norm` の意味が変わり、意図
//!    しない過剰クリップ・過小クリップを招くため）
//! 6. スキップした step でも必ず最後に
//!    [`crate::optim::GradScaler::update`] を呼びスケールを更新する
//!    （backoff のため）
//!
//! AMP を使わない既存ループはこの節の変更の影響を受けず、そのまま
//! `backward → clip → optimizer step` で動作し続ける。
//!
//! # AMP の適用範囲（対象外の明記）
//!
//! - **真の混合精度（f16 forward・f32 master weight）は対象外**。
//!   `Var`／`Tape` の dtype 一般化は
//!   `docs/backend-dtype-dispatch-design.md` §8 で明示的にスコープ外と
//!   されている別軸の変更であり、本モジュールが提供するのは **ホスト
//!   `Tensor<f32>` へ実体化済みの勾配**に対するスケーリング／unscale／
//!   非有限検出のみである
//! - **デバイス常駐更新経路（[`crate::DeviceParamStore`]／
//!   [`crate::Tape::step_device_param_store`]）には unscale／非有限検出が
//!   結線されていない**。AMP はホスト `Tensor<f32>` 勾配（`Gradients::get`／
//!   [`crate::compat::SequentialVars::trainable_grads`]／
//!   `Tape::param_grads_to_host` 経由）にのみ適用できる
//! - [`crate::optim::scale_loss`] は呼び出しごとにスカラー葉を 1 個
//!   tape へ登録する（`Tape::leaf_count` に影響。`Tape::reset` をまたいで
//!   蓄積しない契約は `nn::optim::amp::scale_loss` doc を参照）
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
//! 昇格元公開面と 1 対 1 であること）で機械的に固定する。**AMP の
//! [`crate::optim::GradScaler::scale_loss`] のみ `&Var` を受け取るが、これは既存
//! [`crate::Var::mul`] の合成のみで実装され（`nn::optim::amp` モジュール
//! doc）、新規 `Op`／`BackendOps` メソッド／VJP を追加しない**。
//!
//! # 内部配置の不統一・シグネチャ差異について
//!
//! [`crate::optim::Sgd`] は `fandhe_ai_autodiff::optim`、[`crate::optim::AdamW`] 等は
//! `fandhe_ai_autodiff::nn::optim` と、内部クレート側の配置は歴史的経緯
//! （親 #192 の並行実装）により不統一だが、本モジュールでは単一の
//! `fandhe_ai::optim` 入口へ吸収し利用者からは意識させない。一方で
//! [`crate::optim::Sgd::step`] は `&[&Tensor<f32>]` 2 本（`params`・`grads`）を、
//! [`crate::optim::AdamW::step`]／[`crate::optim::Adam::step`]／
//! [`crate::optim::Lamb::step`] は
//! `&[(&Tensor<f32>, &Tensor<f32>)]`（tuple 列）を
//! 引数に取るというシグネチャ形の相違は**本モジュールでは統一しない**
//! （親 #192 の統合判断待ち。`docs/facade-optimizer-promotion-decision.md`
//! §4.3）。将来統一する場合は破壊的変更になる。
//!
//! # RMSprop／Adagrad（イシュー #1743・親 #1610）
//!
//! [`crate::optim::RmsProp`]／[`crate::optim::RmsPropConfig`]・
//! [`crate::optim::Adagrad`]／[`crate::optim::AdagradConfig`] を
//! `fandhe_ai_autodiff::nn::optim`（`adamw.rs` を鏡写しにした別実装
//! `rmsprop.rs`／`adagrad.rs`）から同じく素の再エクスポートで公開
//! する。`AdamW`・[`crate::optim::Sgd`] と同じく `Tape`／`Var`／
//! `BackendOps` に一切依存しない値型・純関数であり、新規 `Op`／
//! `BackendOps` メソッド／`Var` メソッド／VJP は追加していない
//! （カーネルなし）。位置対応契約（「呼び出し文脈」節）はそのまま
//! 適用される。**`crate::DeviceParamStore` には未結線**（「デバイス
//! 常駐更新との違い」節参照。RMSprop・Adagrad とも本 issue では対応
//! する `BackendOps` メソッドを追加していないため非対応）。
//!
//! # ReduceLrOnPlateau（イシュー #1746・親 #1611）
//!
//! [`crate::optim::ReduceLrOnPlateau`]／[`crate::optim::ReduceLrOnPlateauConfig`]・
//! [`crate::optim::PlateauMode`]／[`crate::optim::ThresholdMode`] を
//! `fandhe_ai_autodiff::nn::optim`（実体は `nn::optim::reduce_lr_on_plateau`
//! モジュール）から同じく素の再エクスポートで公開する。PyTorch
//! `torch.optim.lr_scheduler.ReduceLROnPlateau` 相当で、既存
//! [`crate::optim::ConstantLr`]／[`crate::optim::StepLr`]（stateless
//! 純関数）とは異なり、検証指標の観測に応じて内部状態
//! （patience／best／cooldown カウンタ）を進める **状態保持型**である
//! （詳細は `nn::optim::reduce_lr_on_plateau` モジュール doc）。
//! [`crate::optim::LrScheduler`] は実装するが、状態を進める入口は
//! [`crate::optim::ReduceLrOnPlateau::step`]（検証指標を受け取る）のみで
//! `lr_at` は現在値を返すだけ（`ConstantLr` と同型）。他の optim 型と
//! 同じく `Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数で
//! あり、新規 `Op`／`BackendOps` メソッド／`Var` メソッド／VJP は
//! 追加していない（カーネルなし）。`crate::DeviceParamStore` には
//! 未結線（「デバイス常駐更新との違い」節参照）。
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
//! のまま）。AMP（[`crate::optim::GradScaler`]）もこの経路へは未結線（上記「AMP の
//! 適用範囲」節参照）。[`crate::optim::Adam`]（coupled L2 weight decay。
//! イシュー #1742）も同様に `DeviceParamStore` へは未結線であり、本
//! モジュールの他の optimizer と同じくホスト `Tensor<f32>` を介した
//! optimizer step のみを提供する（`nn::optim::adam` モジュール doc
//! 「`DeviceParamStore` 非対応」節）。[`crate::optim::Lamb`]（layer-wise
//! trust ratio。イシュー #1744）も同様に `DeviceParamStore` へは未結線
//! （パラメータテンソルごとの L2 norm reduction カーネルが未実装の
//! ため。`nn::optim::lamb` モジュール doc「`DeviceParamStore` 非対応」節）。
//!
//! **LAMB（イシュー #1744・親 #1610）**: [`crate::optim::Lamb`]／
//! [`crate::optim::LambConfig`] を `fandhe_ai_autodiff::nn::optim`
//! （実体は `nn::optim::lamb` モジュール）から素の再エクスポートで
//! 公開する。weight decay は paper 定義どおり更新方向 `u` へ coupled
//! で織り込む（`AdamW` の decoupled 乗算減衰とは構造が異なる。
//! `nn::optim::lamb` モジュール doc 参照）。`step()` シグネチャは
//! `AdamW::step`／`Adam::step` と同一（`&[(&Tensor<f32>, &Tensor<f32>)]`
//! を受け取り更新後 `Tensor<f32>` の列を返す）。

// `pub use` は 1 文 1 行を維持する（複数行折返し禁止。`tests/api_surface.rs`
// が `pub use` を行単位（`trimmed.starts_with("pub use")`）で走査する
// 契約に合わせる。`src/lib.rs` 冒頭コメントと同じ理由）。
pub use fandhe_ai_autodiff::nn::optim::{Adagrad, AdagradConfig};
pub use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig};
pub use fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig};
pub use fandhe_ai_autodiff::nn::optim::{ClipGradResult, clip_grad_value};
pub use fandhe_ai_autodiff::nn::optim::{ConstantLr, LrScheduler, StepLr};
pub use fandhe_ai_autodiff::nn::optim::{CosineAnnealingLr, ExponentialLr, LinearWarmupLr};
pub use fandhe_ai_autodiff::nn::optim::{GradScaler, GradScalerConfig, UnscaleResult};
pub use fandhe_ai_autodiff::nn::optim::{Lamb, LambConfig};
pub use fandhe_ai_autodiff::nn::optim::{PlateauMode, ThresholdMode};
pub use fandhe_ai_autodiff::nn::optim::{ReduceLrOnPlateau, ReduceLrOnPlateauConfig};
pub use fandhe_ai_autodiff::nn::optim::{RmsProp, RmsPropConfig};
pub use fandhe_ai_autodiff::nn::optim::{clip_grad_norm, global_grad_norm};
pub use fandhe_ai_autodiff::nn::optim::{has_non_finite, scale_grads, scale_loss, unscale_grads};
pub use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
