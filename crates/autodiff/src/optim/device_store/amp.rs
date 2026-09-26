//! AMP（自動混合精度。`crate::nn::optim::amp`）を [`DeviceParamStore`] の
//! 常駐 step へ結線する（イシュー #2181・`docs/device-resident-update-
//! design.md` 追補）。
//!
//! `device_store` モジュールの子モジュールとして定義する（`device_store.rs`
//! 冒頭の `mod amp;` 参照）。子モジュールは親モジュールで定義された
//! private フィールド・private fn（[`super::DeviceParamStore::pending`]・
//! [`super::DeviceParamStore::pending_for`] 等）へアクセスできる——Rust の
//! 可視性は「定義モジュールとその子孫モジュール」単位で決まるため
//! （`.claude/rules/code-comment-style.md` の「非自明な前提」に対応）。
//!
//! # 設計方針: 既存 `step`／`step_adam`／`step_adamw` の演算列を一切変えない
//!
//! AMP は「scale 済み loss を backward → 勾配を unscale → 非有限なら
//! skip、そうでなければ optimizer step」という**既存 optimizer の外側**の
//! 前処理・後処理であり、`step()`／`step_adam_impl()` 自体（CUDA Graph
//! capture 分岐・`pending_backup` 復元・`poisoned` 遷移を含む複雑な
//! 状態機械）を変更・複製する必要はない。本モジュールは
//! [`DeviceParamStore::param_grads_to_host`]（既存公開 API。イシュー
//! #1479）でスケール済み勾配をホストへ実体化し、
//! [`crate::nn::optim::amp::unscale_grads`] で unscale・非有限検出した
//! うえで、その結果を [`crate::backward::Gradients::synthetic`]
//! （`pub(crate)`。本イシューで新設）を使い通常の `Tape::backward` が
//! 返す形と同じ `Gradients` へ包み直し、**既存の `step`／`step_adam`／
//! `step_adamw` へそのまま渡す**。これにより:
//!
//! - 更新フェーズ（flat grad アップロード・カーネル起動・CUDA Graph
//!   capture・`poisoned` 遷移・`step_count`／`sgd_used`／`adam_state` の
//!   確定）は既存コードを 1 バイトも変更せずに再利用でき、bit 完全一致
//!   （AC4）が構造的に保証される
//! - 合成 `Gradients` の `resident_fingerprint` は常に `None` にする
//!   （`Gradients::synthetic` doc 参照）ため、既存の
//!   `resident_filled_slots` 判定は常に「resident 経由なし」を返し、
//!   `any_resident == false`（単一連結バッファの新規アップロード）
//!   経路が強制される。これは実装計画 §4.1 の「AMP では
//!   `any_resident` を forced false にする」契約そのもの——AMP は
//!   unscale のためにどのみち一度ホストへ実体化するので、resident
//!   直接書き込み経路（デバイス上のまま素通り）を再利用する意味がない
//!
//! # skip 時の状態遷移契約
//!
//! 非有限を検出した場合、`step`／`step_adam`／`step_adamw` を**一切
//! 呼ばない**（カーネル起動 0 回）。[`super::DeviceParamStore::
//! abandon_pending_forward`] で forward 登録（`pending`）のみを消費し、
//! `step_count`／`sgd_used`／`velocity`／`adam_state`／`adam_m`／`adam_v`
//! は変更しない（Adam の bias correction が `t` に依存するため、skip
//! した step は `t` を進めてはならない）。`velocity`／`adam_m`／`adam_v`
//! の遅延確保も既存 `step`／`step_adam_impl` 内部でしか行われないため、
//! skip 時はどのバッファも確保しない（advisor レビュー指摘: skip は
//! alloc してはならない）。
//!
//! # 勾配の鮮度（`resident_filled_slots`）への影響（A08）
//!
//! `resident_filled_slots` の鮮度判定は `grads.resident_fingerprint()`
//! と `self.backward_serial`／`pending.generation` の**その場の突合**で
//! 決まり、`step()` 呼び出しが状態を消費してフラグを立てる方式ではない
//! （`resident_filled_slots` doc 参照）。そのため AMP で skip した後の
//! 次回 backward・`param_grads_to_host`／`step*` 呼び出しは、新しい
//! `grads`（新しい `backward_serial`）と新しい `pending` を使って毎回
//! 再判定されるため、skip によって鮮度判定が stale になることはない
//! （追加のフラグクリアは不要。単体テスト
//! `amp_skip_then_next_backward_step_still_fresh` で固定する）。

use fandhe_ai_tensor_core::Tensor;

use crate::backward::Gradients;
use crate::error::AutodiffError;
use crate::nn::optim::amp::{GradScaler, unscale_grads};
use crate::nn::optim::{AdamConfig, AdamWConfig};
use crate::optim::sgd::SgdConfig;
use crate::tape::Tape;

use super::DeviceParamStore;

impl DeviceParamStore {
    /// [`super::DeviceParamStore::step`]（SGD）の AMP 版。1 step の使い方:
    /// `scaler.scale_loss(&loss)` → `tape.backward`／`DeviceParamStore::
    /// backward` → 本メソッド（内部で unscale・非有限検出・
    /// `scaler.update` まで完結する）。
    ///
    /// 戻り値 `Ok(true)` はこの step が非有限勾配により **skip** された
    /// ことを示す（`crate::nn::optim::amp::UnscaleResult::
    /// should_skip_step` と同じ極性）。skip 時はどのパラメータも
    /// 更新されず、forward 登録（`pending`）のみが消費される（モジュール
    /// doc「skip 時の状態遷移契約」参照）。
    ///
    /// CUDA／Metal はデバイス側 unscale カーネルを持たないため、
    /// `param_grads_to_host` によるホスト計算フォールバック（D2H →
    /// ホスト unscale → 必要なら H2D）を経由する（実装計画 §4.1
    /// 「新規の `BackendOps` を追加しない理由」）。性能目標は置かない。
    ///
    /// # Errors
    ///
    /// 勾配読み出し・unscale・`step()` 本体・`scaler.update` のいずれかが
    /// 失敗した場合（`AutodiffError::Backend` でラップされた
    /// [`fandhe_ai_tensor_core::device::BackendError`] を含む）。
    pub fn step_amp(
        &mut self,
        tape: &Tape,
        grads: &Gradients,
        config: &SgdConfig,
        scaler: &mut GradScaler,
    ) -> Result<bool, AutodiffError> {
        let (synthetic, found_non_finite) = self.amp_unscale_pending(tape, grads, scaler)?;
        if found_non_finite {
            self.abandon_pending_forward();
            scaler.update(true)?;
            return Ok(true);
        }
        self.step(tape, &synthetic, config)?;
        scaler.update(false)?;
        Ok(false)
    }

    /// [`super::DeviceParamStore::step_adam`] の AMP 版。意味論・
    /// エラー・戻り値の極性は [`Self::step_amp`] と同一（[`super::
    /// DeviceParamStore::step`] を呼ぶか [`super::DeviceParamStore::
    /// step_adam`] を呼ぶかのみが異なる）。
    pub fn step_adam_amp(
        &mut self,
        tape: &Tape,
        grads: &Gradients,
        config: &AdamConfig,
        scaler: &mut GradScaler,
    ) -> Result<bool, AutodiffError> {
        let (synthetic, found_non_finite) = self.amp_unscale_pending(tape, grads, scaler)?;
        if found_non_finite {
            self.abandon_pending_forward();
            scaler.update(true)?;
            return Ok(true);
        }
        self.step_adam(tape, &synthetic, config)?;
        scaler.update(false)?;
        Ok(false)
    }

    /// [`super::DeviceParamStore::step_adamw`] の AMP 版。意味論・
    /// エラー・戻り値の極性は [`Self::step_amp`] と同一。
    pub fn step_adamw_amp(
        &mut self,
        tape: &Tape,
        grads: &Gradients,
        config: &AdamWConfig,
        scaler: &mut GradScaler,
    ) -> Result<bool, AutodiffError> {
        let (synthetic, found_non_finite) = self.amp_unscale_pending(tape, grads, scaler)?;
        if found_non_finite {
            self.abandon_pending_forward();
            scaler.update(true)?;
            return Ok(true);
        }
        self.step_adamw(tape, &synthetic, config)?;
        scaler.update(false)?;
        Ok(false)
    }

    /// [`Self::step_amp`]／[`Self::step_adam_amp`]／[`Self::
    /// step_adamw_amp`] の共有ヘルパ: pending 順にスケール済み勾配を
    /// ホストへ読み出し（[`super::DeviceParamStore::param_grads_to_host`]
    /// を再利用。既存の poisoned／device／pending・tape・epoch・shape
    /// 検査を一切変更せずそのまま通す）、`scaler` の現在の scale で
    /// unscale する。`pending` は**消費しない**（`&self` 限定。呼び出し元
    /// が skip／非 skip を判定してから `abandon_pending_forward` または
    /// `step*` の pending 消費を選ぶ）。
    ///
    /// 戻り値は `(合成 Gradients, 非有限検出フラグ)`。合成
    /// `Gradients` の `tape_id`／`epoch` は今回消費対象の `pending` から
    /// 取る（`param_grads_to_host` 自身が「`pending.tape_id == tape.id`
    /// かつ `pending.epoch == tape.epoch()`」を検証済みのため、`tape.id`／
    /// `tape.epoch()` を直接使うのと等価）。
    fn amp_unscale_pending(
        &self,
        tape: &Tape,
        grads: &Gradients,
        scaler: &GradScaler,
    ) -> Result<(Gradients, bool), AutodiffError> {
        let pending = self.pending_for(tape)?;
        let tape_id = pending.tape_id;
        let epoch = pending.epoch;
        // `param_grads_to_host` の呼び出しは `pending` を消費しないため
        // （`&self` 限定・`pending_for` は参照のみを返す契約。`step()` の
        // ような `take()` は行わない）、後段の `node_ids` 参照は今回の
        // `pending` と同一のものを指し続ける。
        let node_ids = pending.node_ids.clone();

        let scaled = self.param_grads_to_host(tape, grads)?;
        let refs: Vec<&Tensor<f32>> = scaled.iter().collect();
        let unscale_result = unscale_grads(&refs, scaler.scale())?;

        // `Gradients::grads` は `NodeId` を添字とする密な `Vec` のため、
        // pending の各パラメータが刺す `NodeId` の最大値まで確保する
        // （通常のテープ走査で生じる `Gradients` の形と同じ——`backward.rs`
        // の `accumulate` も同様に `tape.nodes.len()` 分を確保する）。
        let max_id = node_ids.iter().map(|n| n.0).max();
        let mut grads_vec: Vec<Option<Tensor<f32>>> = match max_id {
            Some(m) => vec![None; m + 1],
            None => Vec::new(),
        };
        for (node_id, tensor) in node_ids.iter().zip(unscale_result.grads) {
            grads_vec[node_id.0] = Some(tensor);
        }

        let synthetic = Gradients::synthetic(tape_id, epoch, grads_vec);
        Ok((synthetic, unscale_result.found_non_finite))
    }
}
