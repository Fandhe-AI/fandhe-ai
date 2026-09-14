//! optimizer 周辺の最小構成部品（親イシュー #192「optimizer（SGD・AdamW）・
//! gradient clipping の実装」・REQ-9・M3）。
//!
//! `nn`（`Linear`・活性化・損失。`nn/mod.rs` 参照）で組んだ計算グラフの
//! 逆伝播結果（`Tape::backward` が返す `Gradients`）を消費し、
//! パラメータの `Tensor<f32>` を更新後の値へ差し替える入口を置く。
//! 既存の学習ループ（`tests/nn_train_convergence.rs::sgd_step`・
//! `tests/poc_v2_2_parity.rs::sgd_step`）が採る不変更新パターン
//! （`Linear` はパラメータを不変に保持し、`Linear::from_parameters` で
//! 更新後の値を持つ新しい `Linear` に差し替える）にそのまま接続できる
//! よう、optimizer の `step()` は `(param, grad)` の参照列を受け取り
//! 更新後 `Tensor<f32>` の列を返す形にする（呼び出し元が層を再構築
//! する。`adamw.rs` の doc 参照）。
//!
//! 本イシュー（#194）で AdamW（[`AdamW`]・[`AdamWConfig`]）を実装した。
//! SGD（momentum・dampening・weight decay・nesterov 対応。PyTorch
//! `torch.optim.SGD` 準拠）は `crate::optim`（`Tape`/`Var` から独立した
//! 純粋な optimizer 群を置く別モジュール。#193）に実装済みで、AdamW
//! （本モジュール）とはモジュールの置き場所が異なる。統合は親 #192
//! 完了時に判断する。
//!
//! **適用順序契約**（#1721 で AMP 損失スケーリングのコア関数
//! （[`amp`]）を追加したため更新）: 1 学習ステップは必ず
//! `scale_loss（AMP 使用時）→ backward → unscale＋非有限検出
//! （AMP 使用時）→ 非有限なら clip・optimizer step の両方をスキップ
//! → 非有限でなければ clip → optimizer step → GradScaler::update
//! （AMP 使用時）` の順で実行する。
//!
//! 非有限検出（[`amp::UnscaleResult::should_skip_step`]）が **clip より
//! 先** に判定される理由: [`clip::clip_grad_norm`]／
//! [`clip::global_grad_norm`]／[`clip::clip_grad_value`]（#1753・親 #1631。
//! value 方式も同じ fail-closed 契約）は非有限勾配に対し `Err` を返す契約
//! （fail-closed）であり、非有限検出前に clip を呼ぶと overflow が
//! 起きただけで学習ループ全体が失敗してしまう（AMP では overflow に
//! よる非有限勾配の出現自体は正常な運用パスであり、その step を
//! スキップして `GradScaler` 側のスケールを backoff させるのが正しい
//! 扱い）。また「clip は unscale 後の生勾配に対してのみ適用する」
//! 契約も崩さない（clip 前に scale が残っていると `max_norm` の意味が
//! 変わり、意図しない過剰クリップ・過小クリップを招くため。仕様突合
//! 2026-08-06・#192 本文）。本モジュールはこの契約を doc として固定し、
//! [`clip`] にテスト（`nn_optim_clip.rs`）で正順・逆順の不一致を回帰化する。
//!
//! AMP（損失スケーリング・unscale・inf/nan 検出）を使わない既存の
//! 学習ループは、`scale_loss`／`unscale`／`GradScaler` を一切呼ばずに
//! `backward → clip → optimizer step` のまま変更なしで動作する
//! （[`amp`] は既存経路に割り込まない独立モジュール）。
//!
//! gradient clipping・LR スケジューラ（本イシュー・#195）を追加した。
//! [`clip::clip_grad_norm`]／[`lr_scheduler::LrScheduler`] は
//! `Gradients`/`Var` に依存しない純関数・純データ構造として実装し、
//! `crate::optim::Sgd`（#193）・[`AdamW`]（#194）等の optimizer 実装
//! からそのまま呼び出せる形にする（疎結合設計）。
//! `nn/mod.rs` の `Module` trait 未定義方針と同様、共通 `Optimizer`
//! trait の定義は本イシューでは行わない（並行実装される #193/#194 と
//! 一方的に API を固定しないため。親 #192 の統合時に判断する）。

mod adagrad;
mod adam;
mod adamw;
mod lamb;
mod rmsprop;

pub mod amp;
pub mod clip;
pub mod lr_scheduler;
pub mod reduce_lr_on_plateau;

pub use adagrad::{Adagrad, AdagradConfig};
pub use adam::{Adam, AdamConfig};
pub use adamw::{AdamW, AdamWConfig};
pub use amp::{
    GradScaler, GradScalerConfig, UnscaleResult, has_non_finite, scale_grads, scale_loss,
    unscale_grads,
};
pub use clip::{ClipGradResult, clip_grad_norm, clip_grad_value, global_grad_norm};
pub use lamb::{Lamb, LambConfig};
pub use lr_scheduler::{
    ConstantLr, CosineAnnealingLr, ExponentialLr, LinearWarmupLr, LrScheduler, StepLr,
};
pub use reduce_lr_on_plateau::{
    PlateauMode, ReduceLrOnPlateau, ReduceLrOnPlateauConfig, ThresholdMode,
};
pub use rmsprop::{RmsProp, RmsPropConfig};

// イシュー #1721: 損失スケーリング（`amp::scale_loss`/`amp::GradScaler::
// scale_loss`）・unscale＋非有限検出（`amp::unscale_grads`/
// `amp::GradScaler::unscale`）・スケール更新契約（`amp::GradScaler::
// update`）のコア関数を追加した。新規 `Op`／`BackendOps` メソッド／VJP
// は追加していない（`amp` モジュール冒頭 doc 参照。`scale_loss` は
// 既存 `Var::mul` の合成のみ）。facade（`fandhe_ai::optim`）への公開・
// `crates/facade/tests/api_surface.rs` の期待集合更新・
// `docs/compat-api-scope.md` §1.3 AMP 行の更新はイシュー #1722 で
// 完了済み（純再エクスポート。`crates/facade/src/optim.rs` 参照）。
// 真の混合精度（f16 forward／f32 master weight）は対象外
// （`docs/backend-dtype-dispatch-design.md` §8）。

// イシュー #1743（親 #1610「optimizer（Adam／RMSprop／Adagrad／
// LAMB）」）: RMSprop（[`RmsProp`]）・Adagrad（[`Adagrad`]）を追加した。
// `AdamW`（#194）・`crate::optim::Sgd`（#193）と同じく `Tape`／`Var`／
// `BackendOps` に一切依存しない値型・純関数の optimizer であり、新規
// `Op`／`BackendOps` メソッド／`Var` メソッド／VJP は追加していない
// （カーネルなし。詳細は `rmsprop.rs`／`adagrad.rs` の冒頭 doc）。
// facade（`fandhe_ai::optim`）への公開は `crates/facade/src/optim.rs`
// の素の再エクスポート（純再エクスポート契約は
// `docs/facade-optimizer-promotion-decision.md` §4 案 A）。
// `crate::optim::device_store::DeviceParamStore::step` は
// `BackendOps::sgd_step_device` 専用のデバイス常駐更新経路であり、
// 本イシューでは対応する `BackendOps` メソッドを追加していないため
// RMSprop・Adagrad とも **`DeviceParamStore` 非対応**。LAMB は #1744
// が対象のまま。
//
// イシュー #1742（親 #1610）: Adam（coupled L2 weight decay。PyTorch
// `torch.optim.Adam(weight_decay>0)` 相当）を追加した（`adam` モジュール
// doc 参照）。`AdamW`（decoupled）とはハイパーパラメータの構造・
// `step()` シグネチャは同一だが、decay を勾配へ加算するか（coupled・
// `Adam`）パラメータへ直接乗算するか（decoupled・`AdamW`）が異なる。
// `weight_decay == 0` では両者は bit 完全一致する
// （`crates/autodiff/tests/nn_optim_adam.rs::
// adam_wd_zero_bit_matches_adamw_wd_zero`）。新規 `Op`／`BackendOps`
// メソッド／VJP は追加していない。facade（`fandhe_ai::optim`）への
// 公開・`crates/facade/tests/api_surface.rs` の期待集合更新・
// `docs/compat-api-scope.md` §1.3 optimizer 行の更新も本イシューで完了
// 済み（`Adam`／`AdamConfig` の純再エクスポート。`crates/facade/src/
// optim.rs` 参照）。`DeviceParamStore` への結線は非対応のまま
// （`adam` モジュール doc「`DeviceParamStore` 非対応」節）。

// イシュー #1744（親 #1610）: LAMB（layer-wise adaptive trust ratio。
// You et al., 2019）を追加した（`lamb` モジュール doc 参照）。`AdamW`・
// `Adam` と同じく `(param, grad)` の参照列を受け取り更新後
// `Tensor<f32>` の列を返す値型・純関数。weight decay は paper 定義
// どおり更新方向 `u` へ coupled で織り込む（`AdamW` の decoupled 乗算
// 減衰とは構造が異なる）。新規 `Op`／`BackendOps`／`Var` メソッド／
// VJP は追加していない。facade（`fandhe_ai::optim`）への公開・
// `crates/facade/tests/api_surface.rs` の期待集合更新・
// `docs/compat-api-scope.md` §1.3 optimizer 行の更新も本イシューで完了
// 済み（`Lamb`／`LambConfig` の純再エクスポート。`crates/facade/src/
// optim.rs` 参照）。`DeviceParamStore` への結線は非対応のまま
// （`lamb` モジュール doc「`DeviceParamStore` 非対応」節。テンソル
// ごとの L2 norm reduction カーネルが未実装のため）。#1610 配下の
// optimizer sub-issue のうち LAMB（本イシュー）で Adam〈#1742〉に
// 続き完了する（RMSprop／Adagrad は #1743 が別途対応）。

// イシュー #1745（親 #1611）: CosineAnnealingLr／ExponentialLr／
// LinearWarmupLr（式ベース・stateless 純関数の LR スケジューラ 3 種）を
// 追加した（`lr_scheduler` モジュール doc 参照）。新規 `Op`／
// `BackendOps` メソッド／`Var`／VJP は追加していない（テンソル演算では
// なくホスト側 `f32` 純関数のため）。facade（`fandhe_ai::optim`）への
// 公開・`crates/facade/tests/api_surface.rs` の期待集合更新・
// `docs/compat-api-scope.md` §1.2 scheduler 行の更新も本イシューで
// 完了済み（純再エクスポート。`crates/facade/src/optim.rs` 参照）。
// 状態保持型の `ReduceLROnPlateau`／`OneCycleLR` は対象外（兄弟
// イシュー #1746／#1747 が担当）。
//
// イシュー #1746（親 #1611）: 検証指標の停滞を検知して学習率を下げる
// 状態保持型スケジューラ（`ReduceLrOnPlateau`）を追加した。既存
// `ConstantLr`／`StepLr`（stateless 純関数）とは異なり、内部に
// patience／best／cooldown カウンタを持つ唯一の例外である
// （詳細は `reduce_lr_on_plateau` モジュール冒頭 doc）。`LrScheduler`
// は実装するが、状態を進める入口は `ReduceLrOnPlateau::step(metric)`
// のみで `lr_at` は現在値を返すだけ（`ConstantLr` と同型）。新規
// `Op`／`BackendOps`／`Var` メソッド／VJP は追加していない
// （`Tape`／`Var` に一切依存しない値型・純関数）。facade
// （`fandhe_ai::optim`）への公開は `crates/facade/src/optim.rs` の
// 素の再エクスポート。
