//! DDP 初期形の勾配平均を [`DeviceParamStore`] の既存 `step*` へ結線する内部ヘルパ
//! （イシュー #3146。設計の正本は `docs/facade-distributed-host-collective-decision.md` §6・§14・§15）。
//!
//! `device_store` の子モジュール（`device_store.rs` の `mod distributed;`）。private フィールド
//! （`layout`・`total_numel`・`pending`）に触るためここに置く。公開入口は
//! [`crate::distributed::step`] のフリー関数で、本ヘルパはそこからだけ呼ばれる（facade は
//! `DeviceParamStore` を再エクスポートしているため、inherent method にすると `RingCommunicator` が
//! facade の公開面へ漏れる。公開は #3147）。
//!
//! # 設計方針: AMP 結線（`amp.rs`）と同じ形
//!
//! `param_grads_to_host` でホストへ実体化 → 勾配を変換（ここでは全ランク平均） →
//! [`Gradients::synthetic`] で包み直し → 既存 `step*` へ委譲する。更新フェーズ（CUDA Graph capture・
//! `pending_backup` 復元・`poisoned` 遷移・optimizer 状態の確定）は 1 バイトも変更・複製しない。
//! 合成 `Gradients` は `resident_fingerprint = None` なので `any_resident == false` 経路が強制される。
//!
//! # 状態遷移の契約
//!
//! - **world_size 1**: `Ok(None)`。読み出しも通信も行わず、呼び出し側は元の `grads` で既存 `step*`
//!   を呼ぶ（resident 経路を含め構造的に bit 一致）。
//! - **all-reduce が `Err`**: `step*` を呼ばないのでパラメータも optimizer 状態も不変。store は
//!   poisoned にしない（デバイス側は未変更）。`pending` は消費しない（`&self` 限定）ので、呼び出し側は
//!   `abandon_pending_forward` で明示的にクリアする。communicator は `RingCommunicator` の契約どおり
//!   poisoned になる（復旧は範囲外）。
//! - **all-reduce より前のローカル失敗**（勾配欠落・tape 不一致・store poisoned など）: 通信に入らず
//!   `Backend` を返す。他ランクは all-reduce の中で `op_timeout` により `Timeout`／`PeerDisconnected`
//!   を受け、更新しないまま戻る（ハングしない。既知の挙動で、早期通知の仕組みは範囲外。記録 §15）。
//! - **数値**: 平均はランク方向（W ≤ 64）の短い軸で `f64` アキュムレータの対象外（記録 §6）。

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::device::BackendError;

use crate::backward::Gradients;
use crate::distributed::{DistributedStepError, ReduceOp, RingCommunicator};
use crate::tape::Tape;

use super::DeviceParamStore;

impl DeviceParamStore {
    /// 各ランクの勾配をホストへ読み出し、全ランクで平均した合成 [`Gradients`] を返す。
    /// world_size 1 は `Ok(None)`（無通信の素通し。モジュール doc 参照）。
    ///
    /// 全ランクが同じパラメータ並び・同じ総要素数で呼ぶこと（`all_reduce` が総要素数の不一致を
    /// `Protocol(TotalElementsMismatch)` で検出する）。`pending` は消費しない。
    pub(crate) fn mean_reduce_grads(
        &self,
        tape: &Tape,
        grads: &Gradients,
        comm: &mut RingCommunicator,
    ) -> Result<Option<Gradients>, DistributedStepError> {
        if comm.world_size() == 1 {
            return Ok(None);
        }
        let pending = self.pending_for(tape)?;
        let tape_id = pending.tape_id;
        let epoch = pending.epoch;
        let node_ids = pending.node_ids.clone();

        let host = self.param_grads_to_host(tape, grads)?;

        // 登録順（= `layout[i].offset` 順）に 1 本の連続バッファへ平坦化する。
        let mut buf: Vec<f32> = Vec::with_capacity(self.total_numel);
        for t in &host {
            buf.extend_from_slice(&t.host_slice());
        }
        // 通信前のローカル検証（fail-closed。ピアの検出に頼らない）。
        if buf.len() != self.total_numel || host.len() != self.layout.len() {
            return Err(DistributedStepError::GradientLengthMismatch {
                expected: self.total_numel,
                got: buf.len(),
            });
        }

        // `Err` なら即 return（書き戻し・step に進まない）。
        comm.all_reduce(&mut buf, ReduceOp::Mean)?;

        let max_id = node_ids.iter().map(|n| n.0).max();
        let mut grads_vec: Vec<Option<Tensor<f32>>> = match max_id {
            Some(m) => vec![None; m + 1],
            None => Vec::new(),
        };
        for (node_id, slot) in node_ids.iter().zip(self.layout.iter()) {
            let end = slot.offset.checked_add(slot.numel).ok_or(
                DistributedStepError::GradientLengthMismatch {
                    expected: self.total_numel,
                    got: buf.len(),
                },
            )?;
            let part = buf
                .get(slot.offset..end)
                .ok_or(DistributedStepError::GradientLengthMismatch {
                    expected: self.total_numel,
                    got: buf.len(),
                })?
                .to_vec();
            let tensor = Tensor::new(part, &slot.shape).map_err(BackendError::ShapeMismatch)?;
            grads_vec[node_id.0] = Some(tensor);
        }
        Ok(Some(Gradients::synthetic(tape_id, epoch, grads_vec)))
    }
}
