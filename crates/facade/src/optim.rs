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
//! stateless な純関数であり、状態保持型の `ReduceLROnPlateau` は対象外
//! （兄弟イシュー #1746 が担当）。
//!
//! **OneCycleLr（PyTorch `OneCycleLR` 相当。イシュー #1747・親 #1611）**:
//! [`crate::optim::OneCycleLr`]／[`crate::optim::OneCycleLrConfig`]／
//! [`crate::optim::OneCycleAnneal`] を同じく `fandhe_ai_autodiff::nn::optim`
//! （実体は `nn::optim::lr_scheduler` モジュール）から素の再エクスポート
//! で公開する。`new` 構築時にフェーズ境界を事前計算して保持することで
//! `lr_at` 自体は参照のみの stateless 純関数として実装されており（`nn::
//! optim::lr_scheduler::OneCycleLr` doc 参照）、他のスケジューラと同じ
//! [`crate::optim::LrScheduler`] trait を実装する。momentum cycling
//! （`cycle_momentum` 等）は対象外。状態保持型で残る対象外は
//! `ReduceLROnPlateau`（兄弟イシュー #1746）のみ。
//!
//! **LR スケジューラ拡張 5 種（イシュー #2176 実装・#2503 公開。親 #2499）**:
//! [`crate::optim::MultiStepLr`]／[`crate::optim::CosineAnnealingWarmRestarts`]／
//! [`crate::optim::CyclicLr`]／[`crate::optim::LambdaLr`]／
//! [`crate::optim::SequentialLr`] を `fandhe_ai_autodiff::nn::optim`
//! （実体は `nn::optim::lr_scheduler` モジュール）から素の再エクスポートで
//! 公開する（`docs/autodiff-lr-scheduler-ext-decision.md` §2・§8。2026-10-04
//! ルート #2499 の一括承認）。いずれも既存 6 種と同じ
//! [`crate::optim::LrScheduler::lr_at`] のみを持つ stateless な値型で、
//! `compat::Sequential` とは識別子単位で別物である。
//!
//! ```
//! use fandhe_ai::optim::{LrScheduler, MultiStepLr, SequentialLr, StepLr};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ms = MultiStepLr::new(0.1, &[2, 4], 0.5)?;
//! assert_eq!(ms.lr_at(0), 0.1);
//! assert_eq!(ms.lr_at(2), 0.05);
//! assert_eq!(ms.lr_at(4), 0.025);
//!
//! let seq = SequentialLr::new(
//!     vec![
//!         Box::new(StepLr::new(0.1, 1, 0.5)?) as Box<dyn LrScheduler>,
//!         Box::new(MultiStepLr::new(0.1, &[2], 0.5)?) as Box<dyn LrScheduler>,
//!     ],
//!     vec![2],
//! )?;
//! assert!(seq.lr_at(0).is_finite());
//! # Ok(())
//! # }
//! ```
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
//! - **デバイス常駐更新経路（[`crate::DeviceParamStore`]）への結線は
//!   イシュー #2181 で完了済み**（[`crate::Tape::
//!   step_device_param_store_amp`]／[`crate::Tape::
//!   step_device_param_store_adam_amp`]／[`crate::Tape::
//!   step_device_param_store_adamw_amp`]。「デバイス常駐更新との違い」節
//!   参照）。実装は既存 `Tape::param_grads_to_host`（イシュー #1479）で
//!   スケール済み勾配をホストへ実体化してから unscale・非有限検出する
//!   （CUDA／Metal 専用のデバイス側 unscale カーネルは持たない。ホスト
//!   計算フォールバック）。SGD／Adam／AdamW の 3 optimizer 限定
//!   （RmsProp／Adagrad／LAMB は未結線。デバイス常駐 step 自体の対応
//!   状況は本 doc「RMSprop／Adagrad」節参照）
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
//! `Var` メソッド／VJP は追加していない（カーネルなし）。位置対応契約
//! （「呼び出し文脈」節）はそのまま適用される。**`crate::
//! DeviceParamStore` へは CPU 限定で結線済み**（イシュー #2175。
//! [`crate::Tape::step_device_param_store_rmsprop`]／[`crate::Tape::
//! step_device_param_store_adagrad`]。「デバイス常駐更新との違い」節
//! 参照）。
//!
//! # Adadelta／Adamax／NAdam／RAdam（イシュー #2171・#2501・親 #2499）
//!
//! [`crate::optim::Adadelta`]／[`crate::optim::AdadeltaConfig`]・
//! [`crate::optim::Adamax`]／[`crate::optim::AdamaxConfig`]・
//! [`crate::optim::NAdam`]／[`crate::optim::NAdamConfig`]・
//! [`crate::optim::RAdam`]／[`crate::optim::RAdamConfig`] を
//! `fandhe_ai_autodiff::nn::optim` から素の再エクスポートで公開する
//! （`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md`
//! §8 の推奨形。ルート #2499 本文「承認範囲」節のユーザー一括承認）。
//! `step()` シグネチャは [`crate::optim::AdamW::step`] と同一
//! （`&[(&Tensor<f32>, &Tensor<f32>)]`）で、`Tape`／`Var`／`BackendOps`
//! に依存しない値型・純関数。位置対応契約（「呼び出し文脈」節）が
//! そのまま適用される。**`crate::DeviceParamStore` 非対応**（決定記録
//! §7）、**`crate::compat::Optimizer`（`compile()`）にも未統合**（同 §9）。
//! 状態保存用の `OptimizerStateDict` trait は facade 非公開のままで、facade
//! のみの import ではそのメソッドに到達しない（`ParamGroupStep` は #2553 で
//! 公開済み。下記「param groups」節）。
//!
//! # param groups（イシュー #2553・親 #2499）
//!
//! 層別の学習率・weight decay（PyTorch の `param_groups` 相当）を
//! [`crate::optim::ParamGroup`]／[`crate::optim::ParamGroupStep`] として公開する
//! （`docs/autodiff-param-groups-decision.md` §9.2・§9.3。承認はルート #2499 の
//! コメント〈issuecomment-6033824965〉）。
//!
//! - グループは**スロット添字**（`step` に渡す `params` の位置。
//!   [`crate::compat::Sequential::named_parameters`] の列挙位置と同じ）で
//!   パラメータを指す。どのグループにも属さないスロットは optimizer の config
//!   （`lr`・`weight_decay`）を使う。
//! - `ParamGroupStep::step_with_groups` を持つのは `Sgd`・`Adam`・`AdamW`・
//!   `RmsProp`・`Adagrad`・`Lamb`・`Adadelta`・`Adamax`・`NAdam`・`RAdam` の 10 種。
//!   `Lbfgs`・Rprop・ASGD・Adafactor・Lion は対象外。メソッドを呼ぶには
//!   `ParamGroupStep` の import が要る。
//! - [`crate::compat::Sequential::compile_with_param_groups`]（`compile()` 経路）で
//!   使えるのは `Optimizer` enum にある 6 種（`Lbfgs` は拒否）。それ以外の種は
//!   手動ループで `step_with_groups` を呼ぶ。
//! - `set_lr` は既定グループ（グループ外スロット）にだけ効く。
//! - glob import（`use fandhe_ai::optim::*;`）の利用者が同名の型を自前で定義して
//!   いると衝突しうる。
//! - 公開面は `crates/facade/tests/api_surface.rs` の正ガード（
//!   `facade_param_groups_public_surface_matches_approved_contract`・
//!   `param_groups_types_match_approved_shape`・
//!   `param_groups_usage_doctests_are_present_and_compiled` ほか。#2554）で固定している。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::optim::{ParamGroup, ParamGroupStep, Sgd, SgdConfig};
//!
//! let mut opt = Sgd::new(SgdConfig::new(0.1)).unwrap();
//! let p0 = Tensor::new(vec![1.0f32, 1.0], &[2]).unwrap();
//! let p1 = Tensor::new(vec![1.0f32, 1.0], &[2]).unwrap();
//! let g = Tensor::new(vec![1.0f32, 1.0], &[2]).unwrap();
//! // スロット 1 だけを lr = 0.5 のグループへ入れる（スロット 0 は config の lr = 0.1）。
//! let groups = [ParamGroup::new(vec![1], 0.5, 0.0)];
//! let out = opt
//!     .step_with_groups(&[&p0, &p1], &[&g, &g], &groups)
//!     .unwrap();
//! assert!((out[0].as_slice().unwrap()[0] - 0.9).abs() < 1e-6);
//! assert!((out[1].as_slice().unwrap()[0] - 0.5).abs() < 1e-6);
//! ```
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
//! のまま）。AMP（[`crate::optim::GradScaler`]）はイシュー #2181 で
//! `step_device_param_store_amp`／`_adam_amp`／`_adamw_amp` として
//! 結線済み（上記「AMP の適用範囲」節参照）。[`crate::optim::Adam`]
//! （coupled L2 weight decay。
//! イシュー #1742）・[`crate::optim::AdamW`] は **イシュー #1959 で
//! `DeviceParamStore` への結線を完了済み**（[`crate::Tape::
//! step_device_param_store_adam`]／[`crate::Tape::
//! step_device_param_store_adamw`]。本モジュールの [`AdamConfig`]／
//! [`AdamWConfig`] をそのまま渡せる。CPU 実装のみ・CUDA／Metal は
//! `Unsupported` のまま。`nn::optim::adam` モジュール doc
//! 「`DeviceParamStore` 結線済み」節）。[`crate::optim::RmsProp`]／
//! [`crate::optim::Adagrad`]／[`crate::optim::Lamb`]（layer-wise
//! trust ratio。イシュー #1744）も **イシュー #2175 で `DeviceParamStore`
//! への結線を完了済み**（[`crate::Tape::step_device_param_store_rmsprop`]／
//! [`crate::Tape::step_device_param_store_adagrad`]／[`crate::Tape::
//! step_device_param_store_lamb`]。CPU 実装のみ・CUDA／Metal のネイティブ
//! カーネルは後続イシューの対象。`nn::optim::{rmsprop,adagrad,lamb}`
//! モジュール doc「`DeviceParamStore` 結線済み」節）。
//!
//! **LAMB（イシュー #1744・親 #1610）**: [`crate::optim::Lamb`]／
//! [`crate::optim::LambConfig`] を `fandhe_ai_autodiff::nn::optim`
//! （実体は `nn::optim::lamb` モジュール）から素の再エクスポートで
//! 公開する。weight decay は paper 定義どおり更新方向 `u` へ coupled
//! で織り込む（`AdamW` の decoupled 乗算減衰とは構造が異なる。
//! `nn::optim::lamb` モジュール doc 参照）。`step()` シグネチャは
//! `AdamW::step`／`Adam::step` と同一（`&[(&Tensor<f32>, &Tensor<f32>)]`
//! を受け取り更新後 `Tensor<f32>` の列を返す）。
//!
//! # L-BFGS（イシュー #2172 コメント・#2502）
//!
//! [`crate::optim::Lbfgs`]（closure 駆動の optimizer 本体）・
//! [`crate::optim::LbfgsConfig`]（ハイパーパラメータ）・
//! [`crate::optim::LbfgsLineSearch`]（line search 方式選択）を
//! `fandhe_ai_autodiff::nn::optim` から素の再エクスポートで公開する
//! （`LbfgsConfig` は #2198・2026-09-27 承認、残る 2 型は #2502・
//! ルート #2499 本文「承認範囲」節の一括承認。形は
//! `docs/autodiff-lbfgs-decision.md` §8）。
//!
//! - [`crate::compat::Optimizer::Lbfgs`] へ `LbfgsConfig` を渡すと
//!   [`crate::compat::Sequential::compile`]／[`crate::compat::Sequential::fit`]
//!   が内部で `Lbfgs::try_step_closure` を駆動する（`compat::training`
//!   モジュール doc「L-BFGS（closure 駆動 optimizer）」節）。
//! - `torch.optim.LBFGS.step(closure)` 相当の手動 closure ループは
//!   `Lbfgs::new` → `step_closure`／`try_step_closure` で書ける。
//! - strong Wolfe は
//!   `LbfgsConfig { line_search: LbfgsLineSearch::StrongWolfe, ..Default::default() }`
//!   で指定できる。
//! - `Lbfgs` の inherent `state_dict`／`load_state_dict`／`history_len`
//!   （#2366）も到達可能になる。`OptimizerStateDict` trait の facade 公開は
//!   本節の対象外（#2555）。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
//!
//! let cfg = LbfgsConfig {
//!     line_search: LbfgsLineSearch::StrongWolfe,
//!     ..LbfgsConfig::default()
//! };
//! let mut opt = Lbfgs::new(cfg).unwrap();
//! let params = vec![Tensor::new(vec![0.0_f32; 2], &[2]).unwrap()];
//! // f(x) = Σ (x - 1)^2 を最小化する。
//! let updated = opt
//!     .step_closure(&params, |p| {
//!         let x = p[0].contiguous();
//!         let x = x.as_slice().unwrap();
//!         let loss = x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum::<f32>();
//!         let g = x.iter().map(|v| 2.0 * (v - 1.0)).collect::<Vec<f32>>();
//!         (loss, vec![Tensor::new(g, &[2]).unwrap()])
//!     })
//!     .unwrap();
//! let x = updated[0].contiguous();
//! assert!(x.as_slice().unwrap().iter().all(|v| (v - 1.0).abs() < 1e-3));
//! ```

// `pub use` は 1 文 1 行を維持する（複数行折返し禁止。`tests/api_surface.rs`
// が `pub use` を行単位（`trimmed.starts_with("pub use")`）で走査する
// 契約に合わせる。`src/lib.rs` 冒頭コメントと同じ理由）。
pub use fandhe_ai_autodiff::nn::optim::{Adadelta, AdadeltaConfig};
pub use fandhe_ai_autodiff::nn::optim::{Adagrad, AdagradConfig};
pub use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig};
pub use fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig};
pub use fandhe_ai_autodiff::nn::optim::{Adamax, AdamaxConfig};
pub use fandhe_ai_autodiff::nn::optim::{ClipGradResult, clip_grad_value};
pub use fandhe_ai_autodiff::nn::optim::{ConstantLr, LrScheduler, StepLr};
pub use fandhe_ai_autodiff::nn::optim::{CosineAnnealingLr, ExponentialLr, LinearWarmupLr};
pub use fandhe_ai_autodiff::nn::optim::{CosineAnnealingWarmRestarts, CyclicLr};
pub use fandhe_ai_autodiff::nn::optim::{GradScaler, GradScalerConfig, UnscaleResult};
pub use fandhe_ai_autodiff::nn::optim::{Lamb, LambConfig};
// イシュー #2502（親 #2500・ルート #2499 本文「承認範囲」節の一括承認）: L-BFGS
// の 3 型を `docs/autodiff-lbfgs-decision.md` §8 の波括弧形で公開する
// （`LbfgsConfig` は #2198 で公開済み。`Lbfgs`〈closure 駆動の本体〉・
// `LbfgsLineSearch`〈line search 方式選択〉を追加）。
pub use fandhe_ai_autodiff::nn::optim::{LambdaLr, MultiStepLr, SequentialLr};
pub use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
pub use fandhe_ai_autodiff::nn::optim::{NAdam, NAdamConfig};
pub use fandhe_ai_autodiff::nn::optim::{OneCycleAnneal, OneCycleLr, OneCycleLrConfig};
// イシュー #2553（親 #2551・ルート #2499 の承認コメント）: param groups の 2 名を
// `docs/autodiff-param-groups-decision.md` §9.2 項目 2 の素の再エクスポートで公開する。
pub use fandhe_ai_autodiff::nn::optim::{ParamGroup, ParamGroupStep};
pub use fandhe_ai_autodiff::nn::optim::{PlateauMode, ThresholdMode};
pub use fandhe_ai_autodiff::nn::optim::{RAdam, RAdamConfig};
pub use fandhe_ai_autodiff::nn::optim::{ReduceLrOnPlateau, ReduceLrOnPlateauConfig};
pub use fandhe_ai_autodiff::nn::optim::{RmsProp, RmsPropConfig};
pub use fandhe_ai_autodiff::nn::optim::{clip_grad_norm, global_grad_norm};
pub use fandhe_ai_autodiff::nn::optim::{has_non_finite, scale_grads, scale_loss, unscale_grads};
pub use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
