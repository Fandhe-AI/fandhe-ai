//! 学習 1 step 全体（forward・MSE・backward・SGD）を encode-only で 1 区間に積む内部チェーン
//! （イシュー #3251。公開形の決定は `docs/compat-resident-train-step-decision.md` §4・§11）。
//!
//! # 役割
//! 現状の学習 1 step は同期（`waitUntilCompleted`）を 7 回起こす（Metal の実測は
//! `docs/backend-metal-command-batching-design.md` §7.5.4・§7.7）。本モジュールは
//! 入力 `x`・目標 `y` を常駐入力として受け、`BackendOps::*_tracked`（encode のみ・待たない）
//! の 6 部品（#3218〜#3220・#3224 で実装済み）だけで 1 step を積み、同期を
//! 「loss を読む 1 回」（[`DeviceParamStore::read_train_step_loss`]）へ集約する。
//!
//! # バックエンド非依存
//! チェーンのループは `BackendOps` の trait メソッドだけを呼ぶ。Metal 固有の処理は無い
//! （#3251 の Metal 固有部分は実機テストと記録だけ）。CUDA（#3252）・CPU（#3253）は部品を
//! 実装すれば同じループを再利用する。部品が `Unsupported` のバックエンドでは本メソッドは
//! `Unsupported` を返し、黙ってホスト経路へ落とさない。
//!
//! # 呼び出し元・境界
//! - 呼ぶ: facade の `#[cfg(test)]` 補助（#3251）。公開入口 `Sequential::train_step_resident`・
//!   `ResidentLoss::read` は #3254 の担当で、本 PR では公開面を増やさない
//! - 生の autodiff `Tape` を取るため facade の利用者からは到達できない
//!   （`predict_device_chain_resident` と同じ前例）
//!
//! # 対応範囲（これ以外は encode 前に型付きエラー）
//! - 全層が `Linear`（`steps` の並び）で、**最後の層以外は `Activation::Relu`・最後の層は
//!   `Activation::None`**。ReLU のない中間 Linear では勾配が転置格納のまま dW に渡り、
//!   Metal の dW デバイス版が `Unsupported` になる（Tape 経路もホスト経路へ落ちて bit 一致を
//!   保証できない）ため
//! - 損失は MSE の mean 固定。SGD のみ（`step()` と同じ前提条件を通す）
//! - `steps` が store の全パラメータをちょうど 1 回ずつ覆うこと（勾配バッファの
//!   取りこぼし・weight tying を拒否する）
//! - バッチが空（`m == 0`）は `Unsupported`
//!
//! # 失敗の境界（決定記録 §5 の 2 段目）
//! - 最初の dW を勾配バッファへ書く前に部品が `Unsupported` を返した場合は、そのまま返し
//!   poison しない。params・velocity・`step_count`・`sgd_used` は不変
//! - 書き込み前の `DeviceContextCaptureInProgress`（一時的な capture 衝突）も params 未変更のため
//!   poison せずそのまま返す（`predict_device_chain` と同じ扱い）
//! - それ以外の `Err`（書き込み後の失敗・SGD の失敗）は `step()` と同じく store を poison する
//!   （SGD 段の `DeviceContextCaptureInProgress` だけは params 未変更のため poison しない）
//! - どの失敗経路でも、encode 済みの中間バッファは drop せず `train_chain_hold` へ退避する
//!   （encode-only の GPU 処理が実行中でも同じデバイスメモリを再利用させないため）
//! - 成功時は encode 成功の時点で `step_count`・`sgd_used`・世代を確定する
//!
//! # 失敗トークン（A08）
//! 全部品を `_tracked` 版で呼び、store の `failure_token` へ登録する。loss を読んだ後に
//! トークンを再検査し、セット済みなら poison して `StorePoisoned` を返す。
//!
//! # 確保について
//! 勾配連結バッファ・loss スカラー・中間バッファは store が次の step まで保持する。
//! Metal の部品は新しいバッファを返すため step ごとの確保はゼロにならない（pool 再利用で
//! 同期は起きない）。確保ゼロ化（`_into` 版の部品）は本 PR の範囲外。

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    Activation, BackendOps, DeviceMatrixLayout, MseReduction, SgdStepConfig,
};
use std::sync::atomic::Ordering;

use crate::grad::mse_loss_scale;
use crate::optim::resident_input::{ResidentInput, TrainStepLoss};
use crate::optim::sgd::SgdConfig;
use crate::tape::Tape;
use crate::var::Reduction;

use super::{ChainStep, DeviceParamStore};

/// 検証済みの 1 層分の計画。
#[derive(Debug, Clone, Copy)]
struct ChainLayer {
    w_slot: usize,
    b_slot: Option<usize>,
    act: Activation,
}

/// encode 中の失敗。`written` は勾配バッファへ 1 回以上書いたか。
type EncodeFailure = (BackendError, bool);

/// encode 中に生成した GPU バッファの置き場。成功・失敗どちらの経路でも、encode-only の GPU 処理が
/// 完了するまでバッファを解放しないよう、呼び出し側が [`Self::into_held`] で store へ退避する。
#[derive(Default)]
struct ChainScratch {
    outs: Vec<DeviceBuffer<f32>>,
    loss: Option<DeviceBuffer<f32>>,
    g_cur: Option<DeviceBuffer<f32>>,
    hold: Vec<DeviceBuffer<f32>>,
}

impl ChainScratch {
    /// loss を除く保持対象を 1 本の Vec へ集約し、loss と共に返す。
    fn into_held(self) -> (Option<DeviceBuffer<f32>>, Vec<DeviceBuffer<f32>>) {
        let mut hold = self.hold;
        hold.extend(self.g_cur);
        hold.extend(self.outs);
        (self.loss, hold)
    }
}

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidArgument(format!("DeviceParamStore::train_step_device_chain: {msg}"))
}

impl DeviceParamStore {
    /// forward・MSE（mean）・backward（dW・bias 勾配・ReLU マスク・L2 以降の d_input）・
    /// SGD を encode-only で 1 区間に積み、loss ハンドルを返す。この区間に同期は無い
    /// （モジュール doc 参照）。
    ///
    /// `x`・`y` は [`Self::upload_resident_input`] が窓の外で upload したハンドル。
    /// `steps` は `predict_device_chain` と同じ `(weight, bias, act)` の列。
    pub fn train_step_device_chain<'t>(
        &mut self,
        tape: &'t Tape,
        x: &ResidentInput,
        y: &ResidentInput,
        steps: &[ChainStep<'_, 't>],
        config: &SgdConfig,
    ) -> Result<TrainStepLoss, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if self.pending.is_some() {
            return Err(BackendError::PendingForwardUnconsumed);
        }
        for (name, r) in [("x", x), ("y", y)] {
            if r.store_id() != self.store_id {
                return Err(invalid(format!(
                    "{name} belongs to a different DeviceParamStore (store_id mismatch)"
                )));
            }
            if r.device() != self.device {
                return Err(BackendError::DeviceMismatch);
            }
        }
        // `step()` と同じ SGD の前提条件（有限値・非負・nesterov 条件、他 optimizer 状態の拒否、
        // momentum 構成の途中変更の拒否）を encode より前に通す。
        config
            .validate()
            .map_err(|e| BackendError::InvalidArgument(e.to_string()))?;
        if self.rmsprop_state.is_some() || self.adagrad_state.is_some() || self.lamb_state.is_some()
        {
            return Err(invalid(
                "this store already has RmsProp/Adagrad/LAMB step history; reconstruct the \
                 store to switch optimizers"
                    .to_string(),
            ));
        }
        let use_momentum = config.momentum != 0.0;
        let momentum_state_exists = self.velocity.is_some();
        if self.sgd_used && momentum_state_exists != use_momentum {
            return Err(invalid(format!(
                "momentum configuration changed mid-training (previously \
                 {momentum_state_exists}, now {use_momentum}); reconstruct the store to change \
                 momentum settings"
            )));
        }
        let is_first_step = !self.sgd_used;

        let layers = self.plan_train_chain(tape, x, y, steps)?;
        let ops = tape.ops();
        let mem = ops.memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::train_step_device_chain: backend does not implement MemoryOps"
                    .to_string(),
            )
        })?;
        let m = x.shape()[0];
        let n_last = y.shape()[1];
        // 式は Tape 経路（`grad.rs` の `Op::MseLoss` VJP）と同じ関数を共有する（二重に書かない）。
        let scale = mse_loss_scale(1.0, m * n_last, Reduction::Mean);

        // 勾配バッファを先に確保し、そのあとで velocity を確定する。この呼び出しで新規確保した
        // velocity は、poison せず戻る失敗経路（velocity 自身・勾配バッファの確保失敗を含む）で
        // `None` へ戻す（残すと `sgd_used == false` のまま velocity だけ在り、次回以降の
        // momentum 構成検査が「途中変更」と誤判定してストアが固着する）。
        let mut grad = match self.train_chain_grad.take() {
            Some(g) if g.shape() == [self.total_numel] => g,
            _ => mem.alloc_zeroed(&[self.total_numel])?,
        };
        let velocity_is_new = use_momentum && self.velocity.is_none();
        if velocity_is_new {
            match mem.alloc_zeroed(&[self.total_numel]) {
                Ok(v) => self.velocity = Some(v),
                Err(e) => {
                    self.train_chain_grad = Some(grad);
                    return Err(e);
                }
            }
        }

        let mut scratch = ChainScratch::default();
        let encoded = self.encode_train_chain(ops, (x, y), &layers, &mut grad, scale, &mut scratch);
        self.train_chain_grad = Some(grad);
        let (loss_opt, mut hold) = scratch.into_held();
        if let Err((e, written)) = encoded {
            // 成功経路と同様、中間バッファは GPU 処理の完了まで store が保持する（loss も含む）。
            let mut keep = hold;
            keep.extend(loss_opt);
            self.train_chain_hold = keep;
            // 書き込み前の `Unsupported`・一時的な capture 衝突だけが「何も変わっていない」ことを
            // 保証できる。
            let unchanged = !written
                && matches!(
                    e,
                    BackendError::Unsupported(_)
                        | BackendError::DeviceContextCaptureInProgress { .. }
                );
            if !unchanged {
                self.poisoned.store(true, Ordering::SeqCst);
            } else if velocity_is_new {
                self.velocity = None;
            }
            return Err(e);
        }
        let Some(loss) = loss_opt else {
            self.train_chain_hold = hold;
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(invalid(
                "loss buffer is missing (internal error)".to_string(),
            ));
        };

        let step_config = SgdStepConfig {
            lr: config.lr,
            momentum: config.momentum,
            dampening: config.dampening,
            weight_decay: config.weight_decay,
            nesterov: config.nesterov,
            is_first_step,
        };
        let velocity_ref = if use_momentum {
            self.velocity.as_mut()
        } else {
            None
        };
        // `train_chain_grad` は直前に `Some` を入れたため常に在る。None は内部矛盾として拒否する。
        let grad_ref = self
            .train_chain_grad
            .as_ref()
            .ok_or_else(|| invalid("gradient workspace is missing (internal error)".to_string()))?;
        if let Err(e) = ops.sgd_step_device_tracked(
            &mut self.params,
            grad_ref,
            velocity_ref,
            &step_config,
            &self.failure_token,
        ) {
            // `step()` と同じ: capture 競合は params 未変更の一過性エラーで poison しない。
            // encode 済みの GPU 処理が実行中でありうるため、中間バッファ・loss は解放せず退避する。
            hold.push(loss);
            self.train_chain_hold = std::mem::take(&mut hold);
            if !matches!(e, BackendError::DeviceContextCaptureInProgress { .. }) {
                self.poisoned.store(true, Ordering::SeqCst);
            } else if velocity_is_new {
                self.velocity = None;
            }
            return Err(e);
        }

        self.step_count += 1;
        self.sgd_used = true;
        self.train_chain_generation = self.train_chain_generation.wrapping_add(1);
        self.train_chain_loss = Some(loss);
        // 新しい step の中間バッファで置き換える（前 step のものはここで解放される。
        // 解放は backend 側で deferred のため同期しない）。
        self.train_chain_hold = std::mem::take(&mut hold);
        Ok(TrainStepLoss::new(
            self.store_id,
            self.train_chain_generation,
        ))
    }

    /// [`Self::train_step_device_chain`] が積んだ step の loss を `f32` で読む。D2H と同期は
    /// ここ 1 回だけ。ハンドルが別 store・古い世代なら `InvalidArgument`。download の後に
    /// 失敗トークンを再検査し、セット済みなら poison して `StorePoisoned`。
    pub fn read_train_step_loss(
        &self,
        tape: &Tape,
        loss: &TrainStepLoss,
    ) -> Result<f32, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if loss.store_id() != self.store_id {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore::read_train_step_loss: loss belongs to a different \
                 DeviceParamStore (store_id mismatch)"
                    .to_string(),
            ));
        }
        if loss.generation() != self.train_chain_generation {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore::read_train_step_loss: stale loss handle (a newer train step \
                 has replaced this loss)"
                    .to_string(),
            ));
        }
        let buf = self.train_chain_loss.as_ref().ok_or_else(|| {
            BackendError::InvalidArgument(
                "DeviceParamStore::read_train_step_loss: no train step has been issued".to_string(),
            )
        })?;
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::read_train_step_loss: backend does not implement MemoryOps"
                    .to_string(),
            )
        })?;
        let host = match mem.download(buf) {
            Ok(t) => t,
            Err(e) => {
                // 積んだ GPU 仕事の失敗がここで表面化しうるため、store を隔離する。
                self.poisoned.store(true, Ordering::SeqCst);
                return Err(e);
            }
        };
        if self.failure_token.is_set() {
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(BackendError::StorePoisoned);
        }
        let host = host.contiguous();
        host.as_slice()
            .and_then(|s| s.first())
            .copied()
            .ok_or_else(|| {
                BackendError::InvalidArgument(
                    "DeviceParamStore::read_train_step_loss: loss buffer is empty".to_string(),
                )
            })
    }

    /// encode より前の事前検証。`self` を変更しない。
    fn plan_train_chain<'t>(
        &self,
        tape: &Tape,
        x: &ResidentInput,
        y: &ResidentInput,
        steps: &[ChainStep<'_, 't>],
    ) -> Result<Vec<ChainLayer>, BackendError> {
        if steps.is_empty() {
            return Err(invalid("steps must not be empty".to_string()));
        }
        let mut seen = vec![false; self.layout.len()];
        let mut layers = Vec::with_capacity(steps.len());
        let mut prev_n: Option<usize> = None;
        let last = steps.len() - 1;
        for (i, (weight, bias, act)) in steps.iter().enumerate() {
            if weight.tape_id != tape.id {
                return Err(BackendError::TapeMismatch);
            }
            if let Some(b) = bias
                && b.tape_id != tape.id
            {
                return Err(BackendError::TapeMismatch);
            }
            let w = self.checked_resident_buffer(weight.store_id, weight.slot)?;
            let (k, n) = match w.shape() {
                [k, n] => (*k, *n),
                other => {
                    return Err(invalid(format!(
                        "layer {i}: weight must be rank 2, got shape {other:?}"
                    )));
                }
            };
            if let Some(b) = bias {
                let bv = self.checked_resident_buffer(b.store_id, b.slot)?;
                if bv.shape() != [n] {
                    return Err(BackendError::ShapeMismatch(
                        fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                            lhs: bv.shape().to_vec(),
                            rhs: vec![n],
                        },
                    ));
                }
            }
            if let Some(pn) = prev_n
                && pn != k
            {
                return Err(BackendError::ShapeMismatch(
                    fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                        lhs: vec![pn],
                        rhs: vec![k, n],
                    },
                ));
            }
            prev_n = Some(n);
            let relu = matches!(act, Activation::Relu);
            let none = matches!(act, Activation::None);
            if (i < last && !relu) || (i == last && !none) {
                return Err(BackendError::Unsupported(format!(
                    "DeviceParamStore::train_step_device_chain: layer {i} activation {act:?} is \
                     unsupported (non-final layers must be Relu and the final layer None)"
                )));
            }
            let slots = std::iter::once(weight.slot).chain(bias.map(|b| b.slot));
            for slot in slots {
                match seen.get_mut(slot) {
                    Some(flag) if !*flag => *flag = true,
                    Some(_) => {
                        return Err(invalid(format!(
                            "parameter slot {slot} is used more than once (weight tying is \
                             unsupported)"
                        )));
                    }
                    None => {
                        return Err(invalid(format!("parameter slot {slot} is out of range")));
                    }
                }
            }
            layers.push(ChainLayer {
                w_slot: weight.slot,
                b_slot: bias.map(|b| b.slot),
                act: *act,
            });
        }
        if seen.iter().any(|s| !s) {
            return Err(BackendError::Unsupported(
                "DeviceParamStore::train_step_device_chain: steps do not cover every parameter \
                 of the store (the gradient workspace would keep stale values)"
                    .to_string(),
            ));
        }
        // 入力の形状: x は rank 2 で列数が第 1 層の行数、y は [m, n_last]。
        let first_k = self
            .layout
            .get(layers[0].w_slot)
            .and_then(|l| l.shape.first().copied())
            .unwrap_or(0);
        let (xs, ys) = (x.shape(), y.shape());
        if xs.len() != 2 {
            return Err(invalid(format!("x must be rank 2, got shape {xs:?}")));
        }
        if xs[1] != first_k {
            return Err(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: xs.to_vec(),
                    rhs: vec![first_k],
                },
            ));
        }
        let n_last = prev_n.unwrap_or(0);
        if ys != [xs[0], n_last] {
            return Err(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: ys.to_vec(),
                    rhs: vec![xs[0], n_last],
                },
            ));
        }
        if xs[0] == 0 {
            return Err(BackendError::Unsupported(
                "DeviceParamStore::train_step_device_chain: empty batch (m == 0) is unsupported"
                    .to_string(),
            ));
        }
        Ok(layers)
    }

    /// forward → loss → MSE backward → 層ごとの dW（と L2 以降の d_input・ReLU マスク）を
    /// encode のみで積む。SGD は呼ばない。生成したバッファ（loss 含む）は成否に依らず `scratch` へ置く。
    fn encode_train_chain(
        &self,
        ops: &dyn BackendOps,
        io: (&ResidentInput, &ResidentInput),
        layers: &[ChainLayer],
        grad: &mut DeviceBuffer<f32>,
        scale: f32,
        scratch: &mut ChainScratch,
    ) -> Result<(), EncodeFailure> {
        let (x, y) = io;
        let token = &self.failure_token;
        let mut written = false;
        let n_layers = layers.len();
        let x_buf = x.buffer();
        let y_buf = y.buffer();

        // forward: 層出力は ReLU マスク元・次層の dW 入力として全部保持する。
        scratch.outs.reserve(n_layers);
        for (i, l) in layers.iter().enumerate() {
            let w = self
                .checked_resident_buffer(self.store_id, l.w_slot)
                .map_err(|e| (e, written))?;
            let b = match l.b_slot {
                Some(s) => Some(
                    self.checked_resident_buffer(self.store_id, s)
                        .map_err(|e| (e, written))?,
                ),
                None => None,
            };
            let out = {
                let src = if i == 0 { x_buf } else { &scratch.outs[i - 1] };
                ops.linear_forward_device_tracked(src, w, b, l.act, token)
                    .map_err(|e| (e, written))?
            };
            scratch.outs.push(out);
        }

        let pred = &scratch.outs[n_layers - 1];
        scratch.loss = Some(
            ops.mse_loss_device_tracked(pred, y_buf, MseReduction::Mean, token)
                .map_err(|e| (e, written))?,
        );
        scratch.g_cur = Some(
            ops.mse_loss_backward_device_tracked(pred, y_buf, scale, token)
                .map_err(|e| (e, written))?,
        );

        for i in (0..n_layers).rev() {
            let l = &layers[i];
            let src = if i == 0 { x_buf } else { &scratch.outs[i - 1] };
            let g_cur = scratch.g_cur.as_ref().ok_or_else(|| {
                (
                    invalid("gradient chain buffer is missing (internal error)".to_string()),
                    written,
                )
            })?;
            let w_off = self.layout[l.w_slot].offset;
            let n = self.layout[l.w_slot].shape.get(1).copied().unwrap_or(0);
            let bias = l.b_slot.map(|s| (self.layout[s].offset, n));

            let g_prev = {
                let src_view =
                    DeviceBufferView::new(src, 0, src.shape()).map_err(|e| (e, written))?;
                let g_view =
                    DeviceBufferView::new(g_cur, 0, g_cur.shape()).map_err(|e| (e, written))?;
                let bias_filled = ops
                    .gemm_fp32_strict_into_device_tracked(
                        src_view,
                        DeviceMatrixLayout::Transposed,
                        g_view,
                        DeviceMatrixLayout::RowMajor,
                        grad,
                        w_off,
                        bias,
                        token,
                    )
                    .map_err(|e| (e, written))?;
                written = true;
                if bias.is_some() && !bias_filled {
                    return Err((
                        invalid(
                            "backend did not fuse the bias gradient reduction (contract \
                             violation: bias gradient would be missing)"
                                .to_string(),
                        ),
                        written,
                    ));
                }
                if i == 0 {
                    // 第 1 層の d_input は使わないので計算しない（#3245）。
                    None
                } else {
                    let w = self
                        .checked_resident_buffer(self.store_id, l.w_slot)
                        .map_err(|e| (e, written))?;
                    // 格納 `[k, m]` で中身は d_out_{i-1} の転置。
                    let t = ops
                        .gemm_resident_lhs_device_tracked(
                            w,
                            g_view,
                            DeviceMatrixLayout::Transposed,
                            token,
                        )
                        .map_err(|e| (e, written))?;
                    let prev_out = &scratch.outs[i - 1];
                    let prev_view = DeviceBufferView::new(prev_out, 0, prev_out.shape())
                        .map_err(|e| (e, written))?;
                    let t_view =
                        DeviceBufferView::new(&t, 0, t.shape()).map_err(|e| (e, written))?;
                    let g_prev = ops
                        .relu_backward_mask_device_tracked(
                            t_view,
                            DeviceMatrixLayout::Transposed,
                            prev_view,
                            token,
                        )
                        .map_err(|e| (e, written))?;
                    scratch.hold.push(t);
                    Some(g_prev)
                }
            };
            if let Some(g_prev) = g_prev
                && let Some(old) = scratch.g_cur.replace(g_prev)
            {
                scratch.hold.push(old);
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    //! 記録型モックによる CI テスト（イシュー #3251）。数値は検証しない（Metal 実機の bit 一致は
    //! facade の `#[ignore]` テストが担う）。ここでは呼び出しの構造・失敗トークンの受け渡し・
    //! 失敗時の store 状態・事前検証を固定する。
    use super::*;
    use fandhe_ai_tensor_core::buffer::{BufferHandle, MemoryOps};
    use fandhe_ai_tensor_core::{Device, DispatchFailureCell, Tensor};
    use std::any::Any;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Counters {
        forward: AtomicUsize,
        mse: AtomicUsize,
        mse_bwd: AtomicUsize,
        dw: AtomicUsize,
        lhs: AtomicUsize,
        mask: AtomicUsize,
        sgd: AtomicUsize,
        download: AtomicUsize,
        upload: AtomicUsize,
        /// 各 `_tracked` 呼び出しが受け取ったトークンのアドレス。
        tokens: Mutex<Vec<usize>>,
        /// `sgd_step_device_tracked` が受けた `is_first_step` の列。
        first_flags: Mutex<Vec<bool>>,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fault {
        None,
        /// forward で `Unsupported`（どの書き込みよりも前）。
        ForwardUnsupported,
        /// 最初の dW が `Unsupported`（書き込み前）。
        FirstDwUnsupported,
        /// d_input（`gemm_resident_lhs`）が `Unsupported`（最初の dW の後）。
        LhsUnsupported,
        SgdFails,
        /// dW が `bias_filled == false` を返す。
        BiasNotFilled,
        /// forward が一時的な capture 衝突を返す（どの書き込みよりも前）。
        ForwardCaptureBusy,
        /// 最初の dW が一時的な capture 衝突を返す（書き込み前・forward 出力は生成済み）。
        FirstDwCaptureBusy,
    }

    struct Rec {
        c: Arc<Counters>,
        fault: Fault,
    }

    struct H(std::cell::RefCell<Vec<f32>>);
    impl std::fmt::Debug for H {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("H")
        }
    }
    impl BufferHandle for H {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn buf(shape: &[usize], fill: f32) -> DeviceBuffer<f32> {
        let numel: usize = shape.iter().product();
        let h: Box<dyn BufferHandle> = Box::new(H(std::cell::RefCell::new(vec![fill; numel])));
        DeviceBuffer::new(Device::Cpu, shape.to_vec(), h)
    }

    impl Rec {
        fn note(&self, t: &DispatchFailureCell) {
            self.c
                .tokens
                .lock()
                .unwrap()
                .push(t as *const DispatchFailureCell as usize);
        }
    }

    impl MemoryOps for Rec {
        fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
            Ok(buf(shape, 0.0))
        }
        fn upload(&self, t: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
            self.c.upload.fetch_add(1, Ordering::SeqCst);
            let d = t.contiguous().as_slice().unwrap_or(&[]).to_vec();
            let h: Box<dyn BufferHandle> = Box::new(H(std::cell::RefCell::new(d)));
            Ok(DeviceBuffer::new(Device::Cpu, t.shape().to_vec(), h))
        }
        fn download(&self, b: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
            self.c.download.fetch_add(1, Ordering::SeqCst);
            let h = b
                .downcast_handle::<H>()
                .ok_or(BackendError::DeviceMismatch)?;
            let d = h.0.borrow().clone();
            Tensor::new(d, b.shape()).map_err(BackendError::ShapeMismatch)
        }
    }

    impl BackendOps for Rec {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn memory_ops(&self) -> Option<&dyn MemoryOps> {
            Some(self)
        }
        fn sgd_step_device(
            &self,
            _p: &mut DeviceBuffer<f32>,
            _g: &DeviceBuffer<f32>,
            _v: Option<&mut DeviceBuffer<f32>>,
            _c: &SgdStepConfig,
        ) -> Result<(), BackendError> {
            Ok(())
        }
        fn sgd_step_device_tracked(
            &self,
            _p: &mut DeviceBuffer<f32>,
            _g: &DeviceBuffer<f32>,
            _v: Option<&mut DeviceBuffer<f32>>,
            c: &SgdStepConfig,
            t: &DispatchFailureCell,
        ) -> Result<(), BackendError> {
            self.c.sgd.fetch_add(1, Ordering::SeqCst);
            self.c.first_flags.lock().unwrap().push(c.is_first_step);
            self.note(t);
            if self.fault == Fault::SgdFails {
                return Err(BackendError::KernelLaunchFailed("mock sgd".into()));
            }
            Ok(())
        }
        fn linear_forward_device_tracked(
            &self,
            a: &DeviceBuffer<f32>,
            w: DeviceBufferView<'_>,
            _b: Option<DeviceBufferView<'_>>,
            _act: Activation,
            t: &DispatchFailureCell,
        ) -> Result<DeviceBuffer<f32>, BackendError> {
            self.c.forward.fetch_add(1, Ordering::SeqCst);
            self.note(t);
            if self.fault == Fault::ForwardUnsupported {
                return Err(BackendError::Unsupported("mock fwd".into()));
            }
            // 一時的な衝突は最初の 1 回だけ（再試行で解消する想定）。
            if self.fault == Fault::ForwardCaptureBusy && self.c.forward.load(Ordering::SeqCst) == 1
            {
                return Err(BackendError::DeviceContextCaptureInProgress { ordinal: 0 });
            }
            Ok(buf(&[a.shape()[0], w.shape()[1]], 1.0))
        }
        fn mse_loss_device_tracked(
            &self,
            _p: &DeviceBuffer<f32>,
            _t: &DeviceBuffer<f32>,
            _r: MseReduction,
            t: &DispatchFailureCell,
        ) -> Result<DeviceBuffer<f32>, BackendError> {
            self.c.mse.fetch_add(1, Ordering::SeqCst);
            self.note(t);
            Ok(buf(&[], 0.5))
        }
        fn mse_loss_backward_device_tracked(
            &self,
            p: &DeviceBuffer<f32>,
            _t: &DeviceBuffer<f32>,
            _s: f32,
            t: &DispatchFailureCell,
        ) -> Result<DeviceBuffer<f32>, BackendError> {
            self.c.mse_bwd.fetch_add(1, Ordering::SeqCst);
            self.note(t);
            Ok(buf(p.shape(), 1.0))
        }
        fn gemm_fp32_strict_into_device_tracked(
            &self,
            _a: DeviceBufferView<'_>,
            _al: DeviceMatrixLayout,
            _b: DeviceBufferView<'_>,
            _bl: DeviceMatrixLayout,
            _out: &mut DeviceBuffer<f32>,
            _off: usize,
            bias: Option<(usize, usize)>,
            t: &DispatchFailureCell,
        ) -> Result<bool, BackendError> {
            self.c.dw.fetch_add(1, Ordering::SeqCst);
            self.note(t);
            if self.fault == Fault::FirstDwUnsupported {
                return Err(BackendError::Unsupported("mock dw".into()));
            }
            if self.fault == Fault::FirstDwCaptureBusy && self.c.dw.load(Ordering::SeqCst) == 1 {
                return Err(BackendError::DeviceContextCaptureInProgress { ordinal: 0 });
            }
            Ok(bias.is_some() && self.fault != Fault::BiasNotFilled)
        }
        fn gemm_resident_lhs_device_tracked(
            &self,
            w: DeviceBufferView<'_>,
            b: DeviceBufferView<'_>,
            _l: DeviceMatrixLayout,
            t: &DispatchFailureCell,
        ) -> Result<DeviceBuffer<f32>, BackendError> {
            self.c.lhs.fetch_add(1, Ordering::SeqCst);
            self.note(t);
            if self.fault == Fault::LhsUnsupported {
                return Err(BackendError::Unsupported("mock lhs".into()));
            }
            Ok(buf(&[w.shape()[0], b.shape()[0]], 1.0))
        }
        fn relu_backward_mask_device_tracked(
            &self,
            _g: DeviceBufferView<'_>,
            _l: DeviceMatrixLayout,
            m: DeviceBufferView<'_>,
            t: &DispatchFailureCell,
        ) -> Result<DeviceBuffer<f32>, BackendError> {
            self.c.mask.fetch_add(1, Ordering::SeqCst);
            self.note(t);
            Ok(buf(m.shape(), 1.0))
        }
        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(crate::eval::matmul(a, b))
        }
        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(crate::eval::add(a, b))
        }
        fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(crate::eval::mul(a, b))
        }
        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(crate::eval::relu(a))
        }
        fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(crate::eval::exp(a))
        }
        fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(crate::eval::tanh(a))
        }
        fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            let s = fandhe_ai_tensor_core::reduce_out_shape(a.shape(), dim)
                .map_err(BackendError::ShapeMismatch)?;
            Ok(crate::eval::sum(a, dim, &s))
        }
        fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            let s = fandhe_ai_tensor_core::reduce_out_shape(a.shape(), dim)
                .map_err(BackendError::ShapeMismatch)?;
            Ok(crate::eval::max(a, dim, &s))
        }
    }

    const M: usize = 4;
    const D_IN: usize = 3;
    const D_HID: usize = 5;
    const D_OUT: usize = 2;

    fn t(shape: &[usize]) -> Tensor<f32> {
        let numel: usize = shape.iter().product();
        Tensor::new(vec![0.25; numel], shape).unwrap()
    }

    /// 3 層 MLP（bias あり）の tape・store・カウンタ。
    fn setup(fault: Fault) -> (Tape, DeviceParamStore, Arc<Counters>) {
        let c = Arc::new(Counters::default());
        let tape = Tape::new_with_ops(Box::new(Rec {
            c: Arc::clone(&c),
            fault,
        }));
        let ps = [
            t(&[D_IN, D_HID]),
            t(&[D_HID]),
            t(&[D_HID, D_HID]),
            t(&[D_HID]),
            t(&[D_HID, D_OUT]),
            t(&[D_OUT]),
        ];
        let refs: Vec<&Tensor<f32>> = ps.iter().collect();
        let store = DeviceParamStore::new(&tape, &refs).unwrap();
        (tape, store, c)
    }

    fn cfg(momentum: f32) -> SgdConfig {
        SgdConfig {
            lr: 0.1,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    const ACTS: [Activation; 3] = [Activation::Relu, Activation::Relu, Activation::None];

    fn run_with(
        tape: &Tape,
        store: &mut DeviceParamStore,
        x: &ResidentInput,
        y: &ResidentInput,
        acts: &[Activation],
        config: &SgdConfig,
    ) -> Result<TrainStepLoss, BackendError> {
        let leaves = store.snapshot_resident_params(tape).unwrap();
        let steps: Vec<ChainStep<'_, '_>> = acts
            .iter()
            .enumerate()
            .map(|(i, a)| (&leaves[2 * i], Some(&leaves[2 * i + 1]), *a))
            .collect();
        store.train_step_device_chain(tape, x, y, &steps, config)
    }

    fn run(
        tape: &Tape,
        store: &mut DeviceParamStore,
        acts: &[Activation],
        config: &SgdConfig,
    ) -> Result<TrainStepLoss, BackendError> {
        let x = store.upload_resident_input(tape, &t(&[M, D_IN])).unwrap();
        let y = store.upload_resident_input(tape, &t(&[M, D_OUT])).unwrap();
        run_with(tape, store, &x, &y, acts, config)
    }

    fn count(c: &AtomicUsize) -> usize {
        c.load(Ordering::SeqCst)
    }

    #[test]
    fn chain_issues_every_part_once_per_layer_and_reads_loss_with_one_download() {
        let (tape, mut store, c) = setup(Fault::None);
        let x = store.upload_resident_input(&tape, &t(&[M, D_IN])).unwrap();
        let y = store.upload_resident_input(&tape, &t(&[M, D_OUT])).unwrap();
        let downloads_before = count(&c.download);
        let uploads_before = count(&c.upload);
        let loss = run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        // チェーン区間は download 0・upload 0（x・y は窓の外で upload 済み）。
        assert_eq!(count(&c.download), downloads_before);
        assert_eq!(count(&c.upload), uploads_before);
        assert_eq!(count(&c.forward), 3);
        assert_eq!(count(&c.mse), 1);
        assert_eq!(count(&c.mse_bwd), 1);
        assert_eq!(count(&c.dw), 3);
        // L1 の d_input は計算しない（層数 − 1 回）。
        assert_eq!(count(&c.lhs), 2);
        assert_eq!(count(&c.mask), 2);
        assert_eq!(count(&c.sgd), 1);
        // loss を読む D2H が step 内で唯一。
        assert_eq!(store.read_train_step_loss(&tape, &loss).unwrap(), 0.5);
        assert_eq!(count(&c.download), downloads_before + 1);
        // 全部品が同じ store の failure_token を受け取る。
        let toks = c.tokens.lock().unwrap();
        let expected = &store.failure_token as *const DispatchFailureCell as usize;
        assert!(!toks.is_empty());
        assert!(toks.iter().all(|&a| a == expected));
        assert_eq!(store.step_count, 1);
        assert!(store.sgd_used);
    }

    #[test]
    fn first_step_flag_follows_sgd_used_and_momentum_change_is_rejected() {
        let (tape, mut store, c) = setup(Fault::None);
        run(&tape, &mut store, &ACTS, &cfg(0.9)).unwrap();
        run(&tape, &mut store, &ACTS, &cfg(0.9)).unwrap();
        assert_eq!(*c.first_flags.lock().unwrap(), vec![true, false]);
        let err = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
        assert_eq!(store.step_count, 2);
        store.check_not_poisoned().unwrap();
    }

    #[test]
    fn stale_or_foreign_loss_handles_are_rejected() {
        let (tape, mut store, _c) = setup(Fault::None);
        let l1 = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        let l2 = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        assert!(matches!(
            store.read_train_step_loss(&tape, &l1),
            Err(BackendError::InvalidArgument(_))
        ));
        store.read_train_step_loss(&tape, &l2).unwrap();
        let (tape2, mut store2, _c2) = setup(Fault::None);
        let other = run(&tape2, &mut store2, &ACTS, &cfg(0.0)).unwrap();
        assert!(matches!(
            store.read_train_step_loss(&tape, &other),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[test]
    fn token_set_after_the_chain_poisons_the_store_on_read() {
        let (tape, mut store, _c) = setup(Fault::None);
        let loss = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        store
            .failure_token
            .set(BackendError::KernelLaunchFailed("late".into()));
        // `check_not_poisoned` が遅延トークンを検出して自己 poison する前に読むと
        // 入口で StorePoisoned になる。いずれにせよ以後の入口はすべて拒否される。
        let err = store.read_train_step_loss(&tape, &loss).unwrap_err();
        assert!(matches!(err, BackendError::StorePoisoned), "{err:?}");
        assert!(matches!(
            store.check_not_poisoned(),
            Err(BackendError::StorePoisoned)
        ));
        // 入口（poison の検査は steps の検証より先）も拒否する。
        let x = ResidentInput::new(buf(&[M, D_IN], 0.0), store.store_id, Device::Cpu);
        let y = ResidentInput::new(buf(&[M, D_OUT], 0.0), store.store_id, Device::Cpu);
        let err = store
            .train_step_device_chain(&tape, &x, &y, &[], &cfg(0.0))
            .unwrap_err();
        assert!(matches!(err, BackendError::StorePoisoned), "{err:?}");
    }

    #[test]
    fn unsupported_before_any_gradient_write_does_not_poison_or_change_state() {
        for fault in [Fault::ForwardUnsupported, Fault::FirstDwUnsupported] {
            let (tape, mut store, c) = setup(fault);
            let err = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap_err();
            assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
            store.check_not_poisoned().unwrap();
            assert_eq!(store.step_count, 0);
            assert!(!store.sgd_used);
            assert_eq!(count(&c.sgd), 0);
            assert!(store.train_chain_loss.is_none());
        }
    }

    #[test]
    fn unsupported_before_write_with_momentum_leaves_no_velocity_behind() {
        for fault in [Fault::ForwardUnsupported, Fault::FirstDwUnsupported] {
            let (tape, mut store, _c) = setup(fault);
            let err = run(&tape, &mut store, &ACTS, &cfg(0.9)).unwrap_err();
            assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
            assert!(store.velocity.is_none(), "新規 velocity は残さない");
            assert!(!store.sgd_used);
            // momentum なしへ切り替えて繰り返しても「途中変更」で拒否されず、
            // 毎回部品の Unsupported がそのまま返る（ストアは固着しない）。
            for _ in 0..2 {
                let err = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap_err();
                assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
                assert!(store.velocity.is_none());
            }
            store.check_not_poisoned().unwrap();
        }
    }

    #[test]
    fn transient_capture_conflict_before_write_does_not_poison_the_store() {
        for fault in [Fault::ForwardCaptureBusy, Fault::FirstDwCaptureBusy] {
            let (tape, mut store, c) = setup(fault);
            let err = run(&tape, &mut store, &ACTS, &cfg(0.9)).unwrap_err();
            assert!(
                matches!(err, BackendError::DeviceContextCaptureInProgress { .. }),
                "{err:?}"
            );
            store.check_not_poisoned().unwrap();
            assert_eq!(store.step_count, 0);
            assert!(!store.sgd_used);
            assert!(store.velocity.is_none(), "新規 velocity は残さない");
            assert_eq!(count(&c.sgd), 0);
            // 衝突が解消した後の step は成功する（同じストアを使い続けられる）。
            let loss = run(&tape, &mut store, &ACTS, &cfg(0.9)).unwrap();
            assert_eq!(store.step_count, 1);
            assert!(store.sgd_used);
            assert_eq!(count(&c.sgd), 1);
            store.read_train_step_loss(&tape, &loss).unwrap();
        }
    }

    #[test]
    fn failure_paths_keep_in_flight_buffers_alive_in_the_store() {
        // 成功経路と同様、Err でも encode 済みの中間バッファを drop せず store が保持する。
        for fault in [
            Fault::FirstDwUnsupported,
            Fault::FirstDwCaptureBusy,
            Fault::LhsUnsupported,
            Fault::SgdFails,
        ] {
            let (tape, mut store, _c) = setup(fault);
            assert!(run(&tape, &mut store, &ACTS, &cfg(0.0)).is_err());
            assert!(
                !store.train_chain_hold.is_empty(),
                "Err 経路でも中間バッファを train_chain_hold へ退避するはず"
            );
        }
    }

    #[test]
    fn failure_after_a_gradient_write_poisons_the_store() {
        for fault in [Fault::LhsUnsupported, Fault::SgdFails, Fault::BiasNotFilled] {
            let (tape, mut store, _c) = setup(fault);
            assert!(run(&tape, &mut store, &ACTS, &cfg(0.0)).is_err());
            assert!(
                matches!(store.check_not_poisoned(), Err(BackendError::StorePoisoned)),
                "書き込み後の失敗は poison するはず"
            );
            assert_eq!(store.step_count, 0);
        }
    }

    #[test]
    fn invalid_configurations_are_rejected_before_encode_and_keep_the_store_usable() {
        let (tape, mut store, c) = setup(Fault::None);
        // 中間層に ReLU が無い・最終層が ReLU は Unsupported。
        for bad in [
            [Activation::Relu, Activation::None, Activation::None],
            [Activation::Relu, Activation::Relu, Activation::Relu],
        ] {
            let err = run(&tape, &mut store, &bad, &cfg(0.0)).unwrap_err();
            assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
        }
        let x = store.upload_resident_input(&tape, &t(&[M, D_IN])).unwrap();
        let y = store.upload_resident_input(&tape, &t(&[M, D_OUT])).unwrap();
        // y の形状不一致。
        let y_bad = store
            .upload_resident_input(&tape, &t(&[M, D_OUT + 1]))
            .unwrap();
        let err = run_with(&tape, &mut store, &x, &y_bad, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)), "{err:?}");
        // x の列数不一致。
        let x_bad = store
            .upload_resident_input(&tape, &t(&[M, D_IN + 1]))
            .unwrap();
        let err = run_with(&tape, &mut store, &x_bad, &y, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)), "{err:?}");
        // 空バッチ。
        let x0 = store.upload_resident_input(&tape, &t(&[0, D_IN])).unwrap();
        let y0 = store.upload_resident_input(&tape, &t(&[0, D_OUT])).unwrap();
        let err = run_with(&tape, &mut store, &x0, &y0, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
        // 別 store のハンドル。
        let (tape2, store2, _c2) = setup(Fault::None);
        let foreign = store2
            .upload_resident_input(&tape2, &t(&[M, D_IN]))
            .unwrap();
        let err = run_with(&tape, &mut store, &foreign, &y, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
        // 層がパラメータを覆わない（先頭 2 層だけ。最終層が ReLU でも先に拒否される）。
        assert!(run(&tape, &mut store, &ACTS[..2], &cfg(0.0)).is_err());
        // どれも encode には進まず、store は使い続けられる。
        assert_eq!(count(&c.forward), 0);
        store.check_not_poisoned().unwrap();
        run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(store.step_count, 1);
    }

    #[test]
    fn pending_forward_blocks_the_chain() {
        let (tape, mut store, _c) = setup(Fault::None);
        let _leaves = store.register_resident_params(&tape).unwrap();
        let err = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(
            matches!(err, BackendError::PendingForwardUnconsumed),
            "{err:?}"
        );
    }
}
