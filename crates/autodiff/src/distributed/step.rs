//! 勾配平均を optimizer ステップへ結線する分散ステップの公開入口（イシュー #3146）。
//!
//! 役割: 各ランクが自分のバッチで backward した後、勾配を `RingCommunicator` で全ランク平均し
//! （`DeviceParamStore::mean_reduce_grads`。`optim/device_store/distributed.rs`）、平均勾配で既存の
//! `DeviceParamStore::step`／`step_adam`／`step_adamw` を呼ぶ。順序は
//! backward → `param_grads_to_host`（登録順）→ 平坦化 → all-reduce（平均）→ 分割 → optimizer ステップ。
//!
//! なぜフリー関数か: facade は `DeviceParamStore` を再エクスポートしているため、inherent method に
//! `&mut RingCommunicator` を取らせると内部型が facade の公開面へ漏れる。本モジュールは facade が
//! 再エクスポートしないので公開面は変わらない。facade への公開は #3147、CPU 複数プロセス検証は #3148。
//!
//! # 呼び出し契約
//!
//! - 呼び出し窓は「backward の後・pending 消費前」（`step*` と同じ）。全ランクが同じ config・同じ
//!   パラメータ並び・同じ総要素数で呼ぶこと。
//! - 損失は平均縮約（`mse_loss` 等）を前提にする。ランクのバッチサイズが等しいとき、平均勾配は
//!   全バッチの勾配と一致する（記録 §6）。
//! - world_size 1 は無通信の素通しで、既存 `step*` と bit 一致する。
//! - all-reduce が失敗したらどのパラメータも optimizer 状態も更新せず（fail-closed）、store は
//!   poisoned にならず、`pending` は残る（`abandon_pending_forward` で明示的にクリアする）。
//!   communicator は poisoned になる。
//! - 通信前にローカルで失敗した場合、他ランクは `op_timeout` で `Timeout`／`PeerDisconnected` を受けて
//!   更新せずに戻る（ハングしない既知の挙動。記録 §15）。
//! - 対象 optimizer は SGD／Adam／AdamW（AMP 結線と同じ）。RmsProp／Adagrad／LAMB は範囲外。
//! - **認証も暗号化も無い**。信頼できるネットワーク内でだけ使う（記録 §7）。

use crate::backward::Gradients;
use crate::nn::optim::{AdamConfig, AdamWConfig};
use crate::optim::{DeviceParamStore, SgdConfig};
use crate::tape::Tape;

use super::error::DistributedStepError;
use super::ring::RingCommunicator;

use fandhe_ai_tensor_core::device::BackendError;

/// SGD の分散ステップ。通信前に `config.validate()` を呼び、不正な config を早期に拒否する。
///
/// # Errors
///
/// all-reduce の失敗は [`DistributedStepError::Collective`]、読み出しや既存 `step` の失敗は
/// [`DistributedStepError::Backend`]。
pub fn step_sgd(
    store: &mut DeviceParamStore,
    tape: &Tape,
    grads: &Gradients,
    config: &SgdConfig,
    comm: &mut RingCommunicator,
) -> Result<(), DistributedStepError> {
    config
        .validate()
        .map_err(|e| BackendError::InvalidArgument(e.to_string()))?;
    match store.mean_reduce_grads(tape, grads, comm)? {
        Some(mean) => store.step(tape, &mean, config)?,
        None => store.step(tape, grads, config)?,
    }
    Ok(())
}

/// Adam の分散ステップ。意味論・エラーは [`step_sgd`] と同じ。
pub fn step_adam(
    store: &mut DeviceParamStore,
    tape: &Tape,
    grads: &Gradients,
    config: &AdamConfig,
    comm: &mut RingCommunicator,
) -> Result<(), DistributedStepError> {
    match store.mean_reduce_grads(tape, grads, comm)? {
        Some(mean) => store.step_adam(tape, &mean, config)?,
        None => store.step_adam(tape, grads, config)?,
    }
    Ok(())
}

/// AdamW の分散ステップ。意味論・エラーは [`step_sgd`] と同じ。
pub fn step_adamw(
    store: &mut DeviceParamStore,
    tape: &Tape,
    grads: &Gradients,
    config: &AdamWConfig,
    comm: &mut RingCommunicator,
) -> Result<(), DistributedStepError> {
    match store.mean_reduce_grads(tape, grads, comm)? {
        Some(mean) => store.step_adamw(tape, &mean, config)?,
        None => store.step_adamw(tape, grads, config)?,
    }
    Ok(())
}
