//! 勾配平均を optimizer ステップへ結線した分散ステップ（フリー関数）。
//!
//! 役割: autodiff 内部の `distributed::step::step_*` への委譲とエラー写像だけを行う薄い入口。
//! フリー関数にするのは、`DeviceParamStore` の inherent method に `&mut Communicator` を取らせると
//! 内部型が公開面へ漏れるため（決定記録 §15・§16）。
//!
//! # 呼び出し契約
//!
//! - 呼び出し窓は「backward の後・pending 消費前」（`Tape::step_device_param_store*` と同じ）。
//!   全ランクが同じ config・同じパラメータ並び・同じ総要素数で呼ぶこと。
//! - 損失は平均縮約（`mse_loss` 等）を前提にする。各ランクのバッチサイズが等しいとき、平均勾配は
//!   全バッチの勾配と一致する。
//! - world_size 1 は無通信の素通しで、既存の `Tape::step_device_param_store*` と bit 一致する。
//! - all-reduce が失敗したらどのパラメータも optimizer 状態も更新せず（fail-closed）、`pending` は
//!   残り、`Communicator` は poisoned になる。
//! - **認証も暗号化も無い**。信頼できるネットワーク内でだけ使う。

use crate::optim::{AdamConfig, AdamWConfig};
use crate::{DeviceParamStore, Gradients, SgdConfig, Tape};

use super::communicator::Communicator;
use super::error::DistributedError;

/// SGD の分散ステップ（勾配を全ランクで平均してから更新する）。
///
/// # Errors
///
/// all-reduce の失敗・バックエンドエラー・勾配長の不一致を [`DistributedError`] で返す。
/// 失敗時はパラメータを更新しない。
pub fn step_sgd(
    tape: &Tape,
    store: &mut DeviceParamStore,
    grads: &Gradients,
    config: &SgdConfig,
    comm: &mut Communicator,
) -> Result<(), DistributedError> {
    fandhe_ai_autodiff::distributed::step::step_sgd(store, &tape.0, grads, config, &mut comm.inner)
        .map_err(DistributedError::from_step)
}

/// Adam の分散ステップ。意味論・エラーは [`step_sgd`] と同じ。
pub fn step_adam(
    tape: &Tape,
    store: &mut DeviceParamStore,
    grads: &Gradients,
    config: &AdamConfig,
    comm: &mut Communicator,
) -> Result<(), DistributedError> {
    fandhe_ai_autodiff::distributed::step::step_adam(store, &tape.0, grads, config, &mut comm.inner)
        .map_err(DistributedError::from_step)
}

/// AdamW の分散ステップ。意味論・エラーは [`step_sgd`] と同じ。
pub fn step_adamw(
    tape: &Tape,
    store: &mut DeviceParamStore,
    grads: &Gradients,
    config: &AdamWConfig,
    comm: &mut Communicator,
) -> Result<(), DistributedError> {
    fandhe_ai_autodiff::distributed::step::step_adamw(
        store,
        &tape.0,
        grads,
        config,
        &mut comm.inner,
    )
    .map_err(DistributedError::from_step)
}
