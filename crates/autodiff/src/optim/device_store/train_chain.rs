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
//! 実装済みで同じループを再利用する（CPU の部品は `backend-cpu::ops_train_chain_parts`）。部品が `Unsupported` のバックエンドでは本メソッドは
//! `Unsupported` を返し、黙ってホスト経路へ落とさない。
//!
//! # 呼び出し元・境界
//! - 呼ぶ: facade の公開入口 `Sequential::train_step_resident`・`ResidentLoss::read`
//!   （#3254）。入口は先に [`DeviceParamStore::validate_train_step_device_chain`] で範囲外を
//!   型付きエラーにし、チェーンが書き込み前に `Unsupported` を返した場合だけ Tape 経路へ全体
//!   フォールバックする（`download_resident_input`・`record_host_train_step_loss` が内部入口）
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
//! - バッチが空（`m == 0`）・最終層の出力幅 0（`n == 0`）は `Unsupported`
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
//! # 確保について（イシュー #3252）
//! 勾配連結バッファ・loss スカラー・層出力・dPred・層ごとの d_input と mask 後勾配は store が
//! step を跨いで保持する（[`TrainChainBufs`]。鍵は `(m, 各層の出力幅)`。変わったら作り直す）。
//! 部品ごとに `BackendOps::*_into_tracked`（出力先を受け取る版）を先に試し、`Unsupported` を返す
//! 部品だけが確保版の `*_tracked` を使う（`Unsupported` はここで消費し、チェーン全体の
//! `Unsupported` としては表に出さない。能力は [`TrainChainCaps`] へラッチする）。
//! - CUDA: 全部品が `_into` 対応のため、定常状態（2 step 目以降）の区間内の確保は 0
//!   （永続ワークスペースは初回だけ grow）。ストリーム順序契約により、前 step の読み出しより
//!   今回の書き込みが後になるためホスト同期なしで再利用しても競合しない
//! - Metal・CPU: `_into` 未対応のため従来どおり step ごとに確保する（pool 再利用で同期は起きない）
//! - 初回 step は確保した出力へ `_into` を 1 回試して能力を確定する（同じ結果の上書き）
//! - 失敗経路のバッファは再利用せず `train_chain_hold` へ退避する
//! - 古い loss ハンドルは世代検査が拒否する（loss バッファを再利用する step の開始時に世代を進める）
//!
//! # 一括再生機構（Graph capture。イシュー #3260。opt-in 機構・#3594 で CUDA は既定 ON）
//! 2 step 目以降で、部品ごとの encode の**前**に `BackendOps::train_chain_captured` を 1 回試す
//! （[`DeviceParamStore::try_captured_train_step`]）。CUDA は opt-in（環境変数
//! `FANDHE_AI_CUDA_GRAPH_TRAIN`）が ON のとき step 全体を 1 本の graph へ capture・再生し、それ以外の
//! バックエンドと opt-out（`=0`）の CUDA は `Ok(None)` で、従来のチェーンがそのまま走る（出力も経路も不変）。
//! - 適用条件: 成功 step を 1 回以上経ている・前 step の永続バッファ一式を鍵つきで再利用できる・
//!   勾配バッファと velocity を新規確保していない・使う部品の `_into` 対応が確定済み
//! - `Ok(None)` なら store もバッファも変えていないので、同じ step を従来のチェーンで実行する
//!   （Tape 経路へは落とさない。`Unsupported` はバックエンドが外へ返さない契約）
//! - `Err(DeviceContextCaptureInProgress)` は launch 前の一過性の衝突で poison しない。外側の
//!   [`DeviceParamStore::train_step_device_chain`] が待ち上限つきで step 全体をやり直す
//! - それ以外の `Err` は書き込み後の可能性があるため store を poison する
//! - **非 capture チェーンの排他（イシュー #3596）**: capture 試行が `Ok(false)` のあと、チェーン
//!   （encode ＋ SGD）の間 `BackendOps::begin_device_chain_scope` のスコープを関数末尾まで持つ。
//!   CUDA では別スレッドの capture 開始がチェーンの完走を待ち、capture 開始後の新しいチェーンは
//!   **書き込み前**（スコープ取得）に `DeviceContextCaptureInProgress` を受けるため、書き込み後に
//!   この衝突で poison される経路はない（SGD 段の同エラー分岐は防御として残す）。スコープは capture
//!   試行・待機の外で取る（持ったまま capture すると開始が拒否される）
//! - capture 区間の本体は `backend-cuda/src/graph.rs::launch_train_chain` にあり、**`encode_train_chain`
//!   ＋ SGD と同じ順・同じ引数**で部品を呼ぶ（bit 一致のため）。ここの順序を変えたらそちらも直す

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    Activation, BackendOps, DeviceMatrixLayout, MseReduction, SgdStepConfig,
    TrainChainCaptureLayer, TrainChainCapturePlan,
};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;

use crate::grad::mse_loss_scale;
use crate::optim::resident_input::{ResidentInput, TrainStepLoss};
use crate::optim::sgd::SgdConfig;
use crate::tape::Tape;
use crate::var::Reduction;

use super::{ChainStep, DeviceParamStore};

/// 別スレッドの Graph capture と衝突した step を待って再試行する際の待ち上限（全体）。
/// `predict_device_chain` の `PREDICT_CAPTURE_WAIT_TIMEOUT` と同じ値（イシュー #3205・#3260）。
/// テストでは mock の待機が即時に返るため、busy-loop を避けて短くする。
#[cfg(not(test))]
const TRAIN_CAPTURE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1000);
#[cfg(test)]
const TRAIN_CAPTURE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(30);

/// 検証済みの 1 層分の計画。
#[derive(Debug, Clone, Copy)]
struct ChainLayer {
    w_slot: usize,
    b_slot: Option<usize>,
    act: Activation,
}

/// encode 中の失敗。`written` は勾配バッファへ 1 回以上書いたか。
type EncodeFailure = (BackendError, bool);

/// 部品の番号（[`TrainChainCaps`] の添字）。
const PART_FORWARD: usize = 0;
const PART_MSE: usize = 1;
const PART_MSE_BWD: usize = 2;
const PART_MASK: usize = 3;
const PART_LHS: usize = 4;
const N_PARTS: usize = 5;

/// 部品ごとの `_into_tracked` 対応状況（バックエンド固有。一度確定したら変わらない）。
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TrainChainCaps {
    /// `Unsupported` を返した部品。以後その部品は確保版だけを使う。
    unsupported: [bool; N_PARTS],
    /// `_into` が成功した部品（初回の能力判定の省略に使う）。
    probed: [bool; N_PARTS],
}

/// step を跨いで保持する学習チェーンの出力バッファ群。`key`（`m` と各層の出力幅）が変わったら
/// 破棄して作り直す。成功した step の分だけを戻す（失敗した step のバッファは残さない）。
#[derive(Debug)]
pub(super) struct TrainChainBufs {
    key: (usize, Vec<usize>),
    outs: Vec<DeviceBuffer<f32>>,
    dpred: Option<DeviceBuffer<f32>>,
    /// 層 `n-1`, `n-2`, .., `1` の順（生成順）の d_input（`[k, m]`）。
    t: Vec<DeviceBuffer<f32>>,
    /// 同順の mask 後勾配（`[m, k]`）。
    gm: Vec<DeviceBuffer<f32>>,
}

impl TrainChainBufs {
    fn into_vec(self) -> Vec<DeviceBuffer<f32>> {
        let mut v = self.outs;
        v.extend(self.dpred);
        v.extend(self.t);
        v.extend(self.gm);
        v
    }
}

/// 前 step から引き継いだ再利用バッファ（生成順に取り出す）。
#[derive(Default)]
struct ReuseQueues {
    outs: VecDeque<DeviceBuffer<f32>>,
    dpred: Option<DeviceBuffer<f32>>,
    t: VecDeque<DeviceBuffer<f32>>,
    gm: VecDeque<DeviceBuffer<f32>>,
    loss: Option<DeviceBuffer<f32>>,
}

/// encode 中に生成した GPU バッファの置き場。成功・失敗どちらの経路でも、encode-only の GPU 処理が
/// 完了するまでバッファを解放しないよう、呼び出し側が [`Self::into_held`]／[`Self::finish`] で
/// store へ退避する。
#[derive(Default)]
struct ChainScratch {
    outs: Vec<DeviceBuffer<f32>>,
    loss: Option<DeviceBuffer<f32>>,
    dpred: Option<DeviceBuffer<f32>>,
    t: Vec<DeviceBuffer<f32>>,
    gm: Vec<DeviceBuffer<f32>>,
    /// 再利用できず捨てるバッファ（世代切れ・`Unsupported` 確定後・未消費分）。実行中の GPU 処理が
    /// 触れうるため解放せず退避する。
    hold: Vec<DeviceBuffer<f32>>,
    reuse: ReuseQueues,
    /// 部品ごとの `_into` 対応状況（encode 中に更新し、呼び出し側が store へ書き戻す）。
    caps: TrainChainCaps,
}

impl ChainScratch {
    fn with_reuse(
        bufs: Option<TrainChainBufs>,
        loss: Option<DeviceBuffer<f32>>,
        caps: TrainChainCaps,
    ) -> Self {
        let mut reuse = ReuseQueues {
            loss,
            ..ReuseQueues::default()
        };
        if let Some(b) = bufs {
            reuse.outs = b.outs.into();
            reuse.dpred = b.dpred;
            reuse.t = b.t.into();
            reuse.gm = b.gm.into();
        }
        Self {
            reuse,
            caps,
            ..Self::default()
        }
    }

    /// 未消費の再利用バッファを `hold` へ移す。
    fn drain_reuse_into_hold(&mut self) {
        let r = std::mem::take(&mut self.reuse);
        self.hold.extend(r.outs);
        self.hold.extend(r.dpred);
        self.hold.extend(r.t);
        self.hold.extend(r.gm);
        self.hold.extend(r.loss);
    }

    /// 失敗経路: loss を除く全バッファを 1 本の Vec へ集約し、loss と共に返す。
    fn into_held(mut self) -> (Option<DeviceBuffer<f32>>, Vec<DeviceBuffer<f32>>) {
        self.drain_reuse_into_hold();
        let mut hold = self.hold;
        hold.extend(self.dpred);
        hold.extend(self.outs);
        hold.extend(self.t);
        hold.extend(self.gm);
        (self.loss, hold)
    }

    /// 成功経路: loss・次 step へ引き継ぐバッファ群・捨てるバッファに分けて返す。
    fn finish(
        mut self,
        key: (usize, Vec<usize>),
    ) -> (
        Option<DeviceBuffer<f32>>,
        TrainChainBufs,
        Vec<DeviceBuffer<f32>>,
    ) {
        self.drain_reuse_into_hold();
        (
            self.loss,
            TrainChainBufs {
                key,
                outs: self.outs,
                dpred: self.dpred,
                t: self.t,
                gm: self.gm,
            },
            self.hold,
        )
    }
}

/// 1 部品の出力を得る。再利用バッファがあれば `into`（`_into_tracked`）で書き込み、なければ
/// `alloc`（確保版 `_tracked`）で確保する。`into` の `Unsupported` は部品単位のフォールバックの
/// 合図として消費し（チェーン全体には伝播させない）、`caps` へラッチして以後は `_into` を試さない。
/// 世代切れ（古いバッファ）は捨てて確保し直す。初回（再利用バッファなし）は確保した出力へ
/// `into` を 1 回試して能力を確定する（同じ結果の上書き。`predict_device_chain` と同じ方式）。
/// 捨てるバッファは実行中の GPU 処理が触れうるため `hold` へ退避する（drop しない）。
fn produce_part(
    part: usize,
    caps: &mut TrainChainCaps,
    reuse: Option<DeviceBuffer<f32>>,
    hold: &mut Vec<DeviceBuffer<f32>>,
    into: &mut dyn FnMut(&mut DeviceBuffer<f32>) -> Result<(), BackendError>,
    alloc: &mut dyn FnMut() -> Result<DeviceBuffer<f32>, BackendError>,
) -> Result<DeviceBuffer<f32>, BackendError> {
    let non_empty = |b: &DeviceBuffer<f32>| b.shape().iter().product::<usize>() != 0;
    if let Some(mut out) = reuse {
        if caps.unsupported[part] || !non_empty(&out) {
            hold.push(out);
        } else {
            match into(&mut out) {
                Ok(()) => return Ok(out),
                Err(BackendError::Unsupported(_)) => {
                    caps.unsupported[part] = true;
                    hold.push(out);
                }
                Err(BackendError::StaleDeviceGeneration { .. }) => hold.push(out),
                Err(e) => {
                    hold.push(out);
                    return Err(e);
                }
            }
        }
    }
    let mut out = alloc()?;
    if !caps.unsupported[part] && !caps.probed[part] && non_empty(&out) {
        match into(&mut out) {
            Ok(()) => caps.probed[part] = true,
            Err(BackendError::Unsupported(_)) => caps.unsupported[part] = true,
            Err(BackendError::StaleDeviceGeneration { .. }) => {}
            Err(e) => {
                hold.push(out);
                return Err(e);
            }
        }
    }
    Ok(out)
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
    ///
    /// 別スレッドの Graph capture と衝突した場合（`DeviceContextCaptureInProgress`。書き込み前、
    /// または SGD 段の衝突で params は未変更）は、待ち上限（`TRAIN_CAPTURE_WAIT_TIMEOUT`）の範囲で
    /// capture の終了を待って step 全体をやり直す（`predict_device_chain` と同型。イシュー #3260・
    /// 決定記録 `docs/backend-cuda-graph-step-capture-design.md` §8.5）。やり直しで部分結果は
    /// 漏れない（勾配バッファと loss は次の試行が全要素を上書きする）。capture が 1 つも無ければ
    /// このエラーは起きず、ループは 1 回で抜ける。
    pub fn train_step_device_chain<'t>(
        &mut self,
        tape: &'t Tape,
        x: &ResidentInput,
        y: &ResidentInput,
        steps: &[ChainStep<'_, 't>],
        config: &SgdConfig,
    ) -> Result<TrainStepLoss, BackendError> {
        let deadline = std::time::Instant::now() + TRAIN_CAPTURE_WAIT_TIMEOUT;
        loop {
            match self.train_step_device_chain_once(tape, x, y, steps, config) {
                Err(BackendError::DeviceContextCaptureInProgress { ordinal })
                    if !self.poisoned.load(Ordering::SeqCst) =>
                {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(BackendError::DeviceContextCaptureInProgress { ordinal });
                    }
                    tape.ops().wait_device_capture_idle(remaining)?;
                }
                other => return other,
            }
        }
    }

    /// [`Self::train_step_device_chain`] の 1 回分（待機・再試行なし）。
    fn train_step_device_chain_once<'t>(
        &mut self,
        tape: &'t Tape,
        x: &ResidentInput,
        y: &ResidentInput,
        steps: &[ChainStep<'_, 't>],
        config: &SgdConfig,
    ) -> Result<TrainStepLoss, BackendError> {
        let layers = self.precheck_train_chain(tape, x, y, steps, config)?;
        let use_momentum = config.momentum != 0.0;
        let is_first_step = !self.sgd_used;
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
        let (mut grad, grad_is_new) = match self.train_chain_grad.take() {
            Some(g) if g.shape() == [self.total_numel] => (g, false),
            _ => (mem.alloc_zeroed(&[self.total_numel])?, true),
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

        // 前 step のバッファを引き継ぐ（鍵が違えば破棄。解放は backend 側で deferred）。loss バッファを
        // 再利用する step では、書き込み後の失敗で古いハンドルが新 step の途中値を読めないよう、
        // 取り出した時点で世代を進める。
        let widths: Vec<usize> = layers
            .iter()
            .map(|l| self.layout[l.w_slot].shape.get(1).copied().unwrap_or(0))
            .collect();
        let key = (m, widths);
        let step_config = SgdStepConfig {
            lr: config.lr,
            momentum: config.momentum,
            dampening: config.dampening,
            weight_decay: config.weight_decay,
            nesterov: config.nesterov,
            is_first_step,
        };

        // opt-in の一括再生機構（CUDA Graph。イシュー #3260）。バックエンドが `Ok(None)` を返す
        // （既定・不適用・capture 失敗の抑止）か、適用条件を満たさなければ、以降の既存チェーンへ進む。
        let steady = !grad_is_new && !velocity_is_new;
        match self.try_captured_train_step(
            ops,
            (x, y),
            &layers,
            &key,
            scale,
            step_config,
            &mut grad,
            steady,
        ) {
            Ok(false) => {}
            Ok(true) => {
                self.train_chain_grad = Some(grad);
                return Ok(TrainStepLoss::new(
                    self.store_id,
                    self.train_chain_generation,
                ));
            }
            Err(e) => {
                self.train_chain_grad = Some(grad);
                // launch 前の capture 衝突だけが「何も変わっていない」ことを保証できる。
                if !matches!(e, BackendError::DeviceContextCaptureInProgress { .. }) {
                    self.poisoned.store(true, Ordering::SeqCst);
                }
                return Err(e);
            }
        }

        // 非 capture チェーン（encode ＋ SGD）の実行区間を、同じ ordinal の別スレッド capture と
        // 排他する（イシュー #3596。決定記録 `docs/backend-cuda-graph-step-capture-design.md`
        // §8.11）。**capture 試行（上の `try_captured_train_step`）の後**で取る: スコープを持つ
        // スレッドは capture を開始できない。関数を抜けるまで保持する（RAII。`?`・panic でも解放）。
        // 取得は書き込み前なので、一過性の衝突・未対応は何も変えずに返す（再試行は呼び出し元の
        // `train_step_device_chain` が待機して行う）。それ以外の失敗は従来どおり poison する。
        let _chain_scope = match ops.begin_device_chain_scope() {
            Ok(scope) => scope,
            Err(e) => {
                self.train_chain_grad = Some(grad);
                if matches!(
                    e,
                    BackendError::Unsupported(_)
                        | BackendError::DeviceContextCaptureInProgress { .. }
                ) {
                    if velocity_is_new {
                        self.velocity = None;
                    }
                } else {
                    self.poisoned.store(true, Ordering::SeqCst);
                }
                return Err(e);
            }
        };

        let reuse_bufs = self.train_chain_bufs.take().filter(|b| b.key == key);
        let reuse_loss = self.train_chain_loss.take();
        if reuse_loss.is_some() {
            self.train_chain_generation = self.train_chain_generation.wrapping_add(1);
        }
        let mut scratch = ChainScratch::with_reuse(reuse_bufs, reuse_loss, self.train_chain_caps);
        let encoded = self.encode_train_chain(ops, (x, y), &layers, &mut grad, scale, &mut scratch);
        self.train_chain_caps = scratch.caps;
        self.train_chain_grad = Some(grad);
        if let Err((e, written)) = encoded {
            let (loss_opt, hold) = scratch.into_held();
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
        let (loss_opt, bufs, mut hold) = scratch.finish(key);
        let Some(loss) = loss_opt else {
            hold.extend(bufs.into_vec());
            self.train_chain_hold = hold;
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(invalid(
                "loss buffer is missing (internal error)".to_string(),
            ));
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
            hold.extend(bufs.into_vec());
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
        self.train_chain_host_loss = None;
        // 成功した step のバッファを次 step の再利用元として保持する。再利用できず捨てた分だけを
        // hold へ残す（解放は backend 側で deferred のため同期しない）。
        self.train_chain_bufs = Some(bufs);
        self.train_chain_hold = std::mem::take(&mut hold);
        Ok(TrainStepLoss::new(
            self.store_id,
            self.train_chain_generation,
        ))
    }

    /// opt-in の一括再生機構（`BackendOps::train_chain_captured`。CUDA では学習 step 全体の
    /// Graph capture・replay。イシュー #3260）で 1 step を実行する。`Ok(true)` は 1 step 分
    /// （SGD の更新を含む）を実行済みで、`step_count`・`sgd_used`・世代・loss を成功 step と
    /// 同じに確定済み。`Ok(false)` は不適用で **store もバッファも変えていない**ので、呼び出し元は
    /// 既存の encode ループを実行する。`Err` はバックエンドの契約どおり（launch 前の
    /// `DeviceContextCaptureInProgress` か、書き込み後の可能性がある失敗）。
    ///
    /// 適用条件（決定記録 §8.2 のウォームアップ）: 既存チェーンの成功 step を 1 回以上経ており
    /// （`!is_first_step`。モジュールロード・永続ワークスペースの grow は済んでいる）、前 step の
    /// 永続バッファ一式を鍵つきで再利用でき、勾配バッファと velocity を新規確保しておらず、
    /// 使う部品の `_into` 対応が確定済みで、配列の本数が層数とそろっていること。
    #[allow(clippy::too_many_arguments)]
    fn try_captured_train_step(
        &mut self,
        ops: &dyn BackendOps,
        io: (&ResidentInput, &ResidentInput),
        layers: &[ChainLayer],
        key: &(usize, Vec<usize>),
        scale: f32,
        sgd: SgdStepConfig,
        grad: &mut DeviceBuffer<f32>,
        steady: bool,
    ) -> Result<bool, BackendError> {
        let n_layers = layers.len();
        let caps = &self.train_chain_caps;
        let needed: &[usize] = if n_layers > 1 {
            &[PART_FORWARD, PART_MSE, PART_MSE_BWD, PART_MASK, PART_LHS]
        } else {
            &[PART_FORWARD, PART_MSE, PART_MSE_BWD]
        };
        let caps_ready = needed
            .iter()
            .all(|&p| caps.probed[p] && !caps.unsupported[p]);
        let use_momentum = sgd.momentum != 0.0;
        let shapes_fit = self.train_chain_bufs.as_ref().is_some_and(|b| {
            b.key == *key
                && b.outs.len() == n_layers
                && b.dpred.is_some()
                && b.t.len() == n_layers - 1
                && b.gm.len() == n_layers - 1
        });
        if sgd.is_first_step
            || !steady
            || !caps_ready
            || !shapes_fit
            || self.train_chain_loss.is_none()
            || self.total_numel == 0
            || (use_momentum && self.velocity.is_none())
        {
            return Ok(false);
        }
        let descs: Vec<TrainChainCaptureLayer> = layers
            .iter()
            .map(|l| {
                let w = &self.layout[l.w_slot];
                TrainChainCaptureLayer {
                    w_offset: w.offset,
                    k: w.shape.first().copied().unwrap_or(0),
                    n: w.shape.get(1).copied().unwrap_or(0),
                    bias_offset: l.b_slot.map(|s| self.layout[s].offset),
                    act: l.act,
                }
            })
            .collect();

        // loss バッファを書き換える step では、書き込み後の失敗で古いハンドルが新 step の途中値を
        // 読めないよう、先に世代を進める（既存チェーンの `reuse_loss` と同じ）。`Ok(None)` で従来の
        // チェーンへ進む場合は世代がもう一度進むが、古いハンドルの拒否は変わらず観測差は無い。
        self.train_chain_generation = self.train_chain_generation.wrapping_add(1);
        let (Some(bufs), Some(loss)) = (
            self.train_chain_bufs.as_mut(),
            self.train_chain_loss.as_mut(),
        ) else {
            return Ok(false);
        };
        let Some(dpred) = bufs.dpred.as_mut() else {
            return Ok(false);
        };
        let plan = TrainChainCapturePlan {
            x: io.0.buffer(),
            y: io.1.buffer(),
            params: &mut self.params,
            layers: &descs,
            grad,
            velocity: if use_momentum {
                self.velocity.as_mut()
            } else {
                None
            },
            outs: &mut bufs.outs,
            dpred,
            d_input: &mut bufs.t,
            masked: &mut bufs.gm,
            loss,
            mse_scale: scale,
            sgd,
            token: &self.failure_token,
        };
        match ops.train_chain_captured(plan)? {
            None => Ok(false),
            Some(_) => {
                self.step_count += 1;
                self.sgd_used = true;
                self.train_chain_generation = self.train_chain_generation.wrapping_add(1);
                self.train_chain_host_loss = None;
                Ok(true)
            }
        }
    }

    /// [`Self::train_step_device_chain`] の事前検証だけを行う（副作用なし。イシュー #3254）。
    ///
    /// facade の `Sequential::train_step_resident` が「モデル構成・損失・最適化が範囲外（型付き
    /// エラーで止める）」と「バックエンドが部品を持たない（Tape 経路へ全体フォールバックしてよい）」を
    /// 判別するために、チェーンを積む前に呼ぶ。ここを通った後にチェーンが返す `Unsupported` は
    /// 後者（最初の dW の前の失敗は store を変えない。モジュール doc「失敗の境界」）に限られる。
    /// 生の `&Tape` を取るため facade の利用者からは到達できない。
    pub fn validate_train_step_device_chain<'t>(
        &self,
        tape: &'t Tape,
        x: &ResidentInput,
        y: &ResidentInput,
        steps: &[ChainStep<'_, 't>],
        config: &SgdConfig,
    ) -> Result<(), BackendError> {
        self.precheck_train_chain(tape, x, y, steps, config)
            .map(|_| ())
    }

    /// 常駐入力をホストへ download する（イシュー #3254。Tape 経路フォールバック専用。D2H が入る
    /// ため通常のチェーン経路では呼ばない）。別 store・別デバイスのハンドルは拒否する。
    pub fn download_resident_input(
        &self,
        tape: &Tape,
        input: &ResidentInput,
    ) -> Result<fandhe_ai_tensor_core::Tensor<f32>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if input.store_id() != self.store_id {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore::download_resident_input: input belongs to a different \
                 DeviceParamStore (store_id mismatch)"
                    .to_string(),
            ));
        }
        if input.device() != self.device {
            return Err(BackendError::DeviceMismatch);
        }
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::download_resident_input: backend does not implement MemoryOps"
                    .to_string(),
            )
        })?;
        mem.download(input.buffer())
    }

    /// Tape 経路へフォールバックした step の loss（ホスト `f32`）を記録し、新しい世代の
    /// ハンドルを返す（イシュー #3254）。世代を進めるため、それ以前のハンドル（チェーン・
    /// フォールバックのどちらのものも）は `InvalidArgument` になる。読み出し
    /// （[`Self::read_train_step_loss`]）は同期なしでこの値を返す。
    pub fn record_host_train_step_loss(
        &mut self,
        tape: &Tape,
        loss: f32,
    ) -> Result<TrainStepLoss, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        self.train_chain_generation = self.train_chain_generation.wrapping_add(1);
        self.train_chain_host_loss = Some(loss);
        Ok(TrainStepLoss::new(
            self.store_id,
            self.train_chain_generation,
        ))
    }

    /// このバックエンドへ積まれた仕事の完了を D2H なしで待つ（イシュー #3254 P7。facade の
    /// `Tape::wait_device_param_store` の実体）。poison 済み・別デバイスは拒否し、待機後に失敗
    /// トークンを再検査してセット済みなら poison して `StorePoisoned`（A08）。ops が同期の部品を
    /// 持たない（`Unsupported`）場合は store の状態を変えずにそのまま返す。待機自体が失敗した
    /// 場合は積んだ GPU 仕事の失敗が表面化したとみなして poison する。
    pub fn wait_device(&self, tape: &Tape) -> Result<(), BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        match tape.ops().synchronize_device() {
            Ok(()) => {}
            Err(e @ BackendError::Unsupported(_)) => return Err(e),
            // 別スレッドの capture 中は driver に触れる前に返る一過性エラー。積んだ GPU 仕事の
            // 失敗ではないため poison せず、store を変えずに返す（step 系と同じ再試行可能な契約）。
            Err(e @ BackendError::DeviceContextCaptureInProgress { .. }) => return Err(e),
            Err(e) => {
                self.poisoned.store(true, Ordering::SeqCst);
                return Err(e);
            }
        }
        if self.failure_token.is_set() {
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(BackendError::StorePoisoned);
        }
        Ok(())
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
        // Tape 経路フォールバックの step（`record_host_train_step_loss`）はホストで確定済みの値を
        // 同期なしで返す（poison・失敗トークンは上の `check_not_poisoned` が検査済み）。
        if let Some(v) = self.train_chain_host_loss {
            return Ok(v);
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
            // capture 競合は GPU 仕事の失敗ではない一過性エラー。poison せず再試行可能にする。
            Err(e @ BackendError::DeviceContextCaptureInProgress { .. }) => return Err(e),
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

    /// encode より前の事前検証一式（poison・デバイス・pending・入力の束縛・SGD 設定・optimizer
    /// 状態・[`Self::plan_train_chain`]）。`self` を変更しない。[`Self::train_step_device_chain`] と
    /// [`Self::validate_train_step_device_chain`] が共有する（検査を二重に書かない）。
    fn precheck_train_chain<'t>(
        &self,
        tape: &'t Tape,
        x: &ResidentInput,
        y: &ResidentInput,
        steps: &[ChainStep<'_, 't>],
        config: &SgdConfig,
    ) -> Result<Vec<ChainLayer>, BackendError> {
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
        self.plan_train_chain(tape, x, y, steps)
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
        // 最終層の出力幅 0 は `mse_loss_scale(1.0, 0, Mean)` が inf になり、backward / SGD が
        // 非有限の更新値を書き込むため、scale 計算より前に型付きエラーで拒否する。
        if n_last == 0 {
            return Err(BackendError::Unsupported(
                "DeviceParamStore::train_step_device_chain: zero-width output (n == 0) is unsupported"
                    .to_string(),
            ));
        }
        Ok(layers)
    }

    /// forward → loss → MSE backward → 層ごとの dW（と L2 以降の d_input・ReLU マスク）を
    /// encode のみで積む。SGD は呼ばない。生成したバッファ（loss 含む）は成否に依らず `scratch` へ置く。
    ///
    /// 部品を呼ぶ順・引数は `backend-cuda/src/graph.rs::launch_train_chain`（Graph capture の本体。
    /// イシュー #3260）と同一に保つこと（capture あり／なしの bit 一致のため）。
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
            let reuse = scratch.reuse.outs.pop_front();
            let out = {
                let src = if i == 0 { x_buf } else { &scratch.outs[i - 1] };
                produce_part(
                    PART_FORWARD,
                    &mut scratch.caps,
                    reuse,
                    &mut scratch.hold,
                    &mut |o| ops.linear_forward_device_into_tracked(src, w, b, l.act, o, token),
                    &mut || ops.linear_forward_device_tracked(src, w, b, l.act, token),
                )
                .map_err(|e| (e, written))?
            };
            scratch.outs.push(out);
        }

        let pred = &scratch.outs[n_layers - 1];
        let reuse_loss = scratch.reuse.loss.take();
        scratch.loss = Some(
            produce_part(
                PART_MSE,
                &mut scratch.caps,
                reuse_loss,
                &mut scratch.hold,
                &mut |o| {
                    ops.mse_loss_device_into_tracked(pred, y_buf, MseReduction::Mean, o, token)
                },
                &mut || ops.mse_loss_device_tracked(pred, y_buf, MseReduction::Mean, token),
            )
            .map_err(|e| (e, written))?,
        );
        let reuse_dpred = scratch.reuse.dpred.take();
        scratch.dpred = Some(
            produce_part(
                PART_MSE_BWD,
                &mut scratch.caps,
                reuse_dpred,
                &mut scratch.hold,
                &mut |o| ops.mse_loss_backward_device_into_tracked(pred, y_buf, scale, o, token),
                &mut || ops.mse_loss_backward_device_tracked(pred, y_buf, scale, token),
            )
            .map_err(|e| (e, written))?,
        );

        for i in (0..n_layers).rev() {
            let l = &layers[i];
            let src = if i == 0 { x_buf } else { &scratch.outs[i - 1] };
            // 現在の勾配: 最後に作った mask 後勾配、なければ dPred。
            let g_cur = scratch
                .gm
                .last()
                .or(scratch.dpred.as_ref())
                .ok_or_else(|| {
                    (
                        invalid("gradient chain buffer is missing (internal error)".to_string()),
                        written,
                    )
                })?;
            let w_off = self.layout[l.w_slot].offset;
            let n = self.layout[l.w_slot].shape.get(1).copied().unwrap_or(0);
            let bias = l.b_slot.map(|s| (self.layout[s].offset, n));

            let src_view = DeviceBufferView::new(src, 0, src.shape()).map_err(|e| (e, written))?;
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
                continue;
            }
            let w = self
                .checked_resident_buffer(self.store_id, l.w_slot)
                .map_err(|e| (e, written))?;
            // 格納 `[k, m]` で中身は d_out_{i-1} の転置。
            let reuse_t = scratch.reuse.t.pop_front();
            let t = produce_part(
                PART_LHS,
                &mut scratch.caps,
                reuse_t,
                &mut scratch.hold,
                &mut |o| {
                    ops.gemm_resident_lhs_device_into_tracked(
                        w,
                        g_view,
                        DeviceMatrixLayout::Transposed,
                        o,
                        token,
                    )
                },
                &mut || {
                    ops.gemm_resident_lhs_device_tracked(
                        w,
                        g_view,
                        DeviceMatrixLayout::Transposed,
                        token,
                    )
                },
            )
            .map_err(|e| (e, written))?;
            // gemm は `t` へ書く GPU 作業を積み済み。後続の view 設定・mask encode が失敗
            // しても完了前に drop されプールが再利用しないよう、成功直後に scratch へ移す。
            scratch.t.push(t);
            let t = scratch.t.last().ok_or_else(|| {
                (
                    invalid("d_input buffer is missing (internal error)".to_string()),
                    written,
                )
            })?;
            let t_view = DeviceBufferView::new(t, 0, t.shape()).map_err(|e| (e, written))?;
            let prev_out = &scratch.outs[i - 1];
            let prev_view =
                DeviceBufferView::new(prev_out, 0, prev_out.shape()).map_err(|e| (e, written))?;
            let reuse_g = scratch.reuse.gm.pop_front();
            let g_prev = produce_part(
                PART_MASK,
                &mut scratch.caps,
                reuse_g,
                &mut scratch.hold,
                &mut |o| {
                    ops.relu_backward_mask_device_into_tracked(
                        t_view,
                        DeviceMatrixLayout::Transposed,
                        prev_view,
                        o,
                        token,
                    )
                },
                &mut || {
                    ops.relu_backward_mask_device_tracked(
                        t_view,
                        DeviceMatrixLayout::Transposed,
                        prev_view,
                        token,
                    )
                },
            )
            .map_err(|e| (e, written))?;
            scratch.gm.push(g_prev);
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
    use fandhe_ai_tensor_core::{Device, DispatchFailureCell, SegmentRun, Tensor};
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
        /// `synchronize_device` の呼び出し回数（イシュー #3254）。
        sync: AtomicUsize,
        /// `MemoryOps::alloc_zeroed` の呼び出し回数。
        alloc: AtomicUsize,
        /// `*_into_tracked` が成功した回数（部品の合計）。
        into_ok: AtomicUsize,
        /// `*_into_tracked` の呼び出し回数（`Unsupported` を返した分を含む）。
        into_calls: AtomicUsize,
        /// 各 `_tracked` 呼び出しが受け取ったトークンのアドレス。
        tokens: Mutex<Vec<usize>>,
        /// `sgd_step_device_tracked` が受けた `is_first_step` の列。
        first_flags: Mutex<Vec<bool>>,
        /// `train_chain_captured` の挙動（`CAP_*`）。
        cap_mode: AtomicUsize,
        /// `train_chain_captured` の呼び出し回数。
        cap_calls: AtomicUsize,
        /// 各 `train_chain_captured` 呼び出しが受けた永続バッファのアドレス（params・grad・loss・outs 先頭）。
        cap_ptrs: Mutex<Vec<[usize; 4]>>,
        /// `begin_device_chain_scope` の呼び出し回数（イシュー #3596）。
        scope_opens: AtomicUsize,
        /// いま保持中のスコープの個数（drop で減る）。
        scope_live: Arc<AtomicUsize>,
        /// スコープ取得の挙動（`SCOPE_*`）。
        scope_mode: AtomicUsize,
        /// 各 `train_chain_captured` 呼び出し時点の保持中スコープ数（0 でなければならない）。
        live_at_cap: Mutex<Vec<usize>>,
        /// 各 `sgd_step_device_tracked` 呼び出し時点の保持中スコープ数（1 でなければならない）。
        live_at_sgd: Mutex<Vec<usize>>,
    }

    // スコープ取得は既定（0）で常に成功する。
    /// 最初の 1 回だけ capture 衝突（書き込み前）。
    const SCOPE_BUSY_ONCE: usize = 1;
    /// 常に実行時エラー。
    const SCOPE_FAILS: usize = 2;

    /// 保持中の個数を drop で戻すモックのスコープ実体。
    struct ScopeGuard(Arc<AtomicUsize>);
    impl Drop for ScopeGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// `train_chain_captured` を実装しない（既定の `Ok(None)`）。
    const CAP_NONE: usize = 0;
    /// 常に `Ok(Some(Replayed))`（部品も SGD も積まない）。
    const CAP_REPLAY: usize = 1;
    /// 最初の 1 回だけ capture 衝突、以後 `Replayed`。
    const CAP_BUSY_ONCE: usize = 2;
    /// 常に capture 衝突。
    const CAP_BUSY_ALWAYS: usize = 3;
    /// 常に実行時エラー（書き込み後の失敗を模擬）。
    const CAP_FAILS: usize = 4;

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
        /// `synchronize_device` が一時的な capture 衝突を返す。
        SyncCaptureBusy,
        /// dW が `bias_filled == false` を返す。
        BiasNotFilled,
        /// forward が一時的な capture 衝突を返す（どの書き込みよりも前）。
        ForwardCaptureBusy,
        /// 最初の dW が一時的な capture 衝突を返す（書き込み前・forward 出力は生成済み）。
        FirstDwCaptureBusy,
        /// ReLU マスク encode が失敗（d_input の `gemm_resident_lhs` 出力は生成・encode 済み）。
        MaskUnsupported,
        /// `synchronize_device` が `Unsupported`（既定実装と同じ。イシュー #3254）。
        SyncUnsupported,
        /// `synchronize_device` が実行時エラー。
        SyncFails,
    }

    struct Rec {
        c: Arc<Counters>,
        fault: Fault,
        /// `true` なら `*_into_tracked` を実装する（CUDA 相当）。`false` は既定の `Unsupported`。
        into_supported: bool,
        /// `true` なら `*_into_tracked` が世代切れ（`StaleDeviceGeneration`）を返す。
        into_stale: bool,
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
        /// 全 `_into` 部品の共通挙動（呼び出し回数を数え、設定に応じて成功・`Unsupported`・世代切れ）。
        fn run_into_part(&self, t: &DispatchFailureCell) -> Result<(), BackendError> {
            self.c.into_calls.fetch_add(1, Ordering::SeqCst);
            if !self.into_supported {
                return Err(BackendError::Unsupported("mock: no into".into()));
            }
            self.note(t);
            if self.into_stale {
                return Err(BackendError::StaleDeviceGeneration {
                    ordinal: 0,
                    resource_generation: 0,
                    current_generation: 1,
                });
            }
            self.c.into_ok.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

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
            self.c.alloc.fetch_add(1, Ordering::SeqCst);
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
        fn train_chain_captured(
            &self,
            plan: TrainChainCapturePlan<'_>,
        ) -> Result<Option<SegmentRun>, BackendError> {
            let calls = self.c.cap_calls.fetch_add(1, Ordering::SeqCst) + 1;
            self.c
                .live_at_cap
                .lock()
                .unwrap()
                .push(self.c.scope_live.load(Ordering::SeqCst));
            self.note(plan.token);
            self.c.cap_ptrs.lock().unwrap().push([
                &*plan.params as *const DeviceBuffer<f32> as usize,
                &*plan.grad as *const DeviceBuffer<f32> as usize,
                &*plan.loss as *const DeviceBuffer<f32> as usize,
                plan.outs.as_ptr() as usize,
            ]);
            match self.c.cap_mode.load(Ordering::SeqCst) {
                CAP_REPLAY => Ok(Some(SegmentRun::Replayed)),
                CAP_BUSY_ONCE if calls == 1 => {
                    Err(BackendError::DeviceContextCaptureInProgress { ordinal: 0 })
                }
                CAP_BUSY_ONCE => Ok(Some(SegmentRun::Replayed)),
                CAP_BUSY_ALWAYS => Err(BackendError::DeviceContextCaptureInProgress { ordinal: 0 }),
                CAP_FAILS => Err(BackendError::KernelLaunchFailed("mock capture".into())),
                _ => Ok(None),
            }
        }
        fn begin_device_chain_scope(
            &self,
        ) -> Result<fandhe_ai_tensor_core::DeviceChainScope, BackendError> {
            let opens = self.c.scope_opens.fetch_add(1, Ordering::SeqCst) + 1;
            match self.c.scope_mode.load(Ordering::SeqCst) {
                SCOPE_BUSY_ONCE if opens == 1 => {
                    Err(BackendError::DeviceContextCaptureInProgress { ordinal: 0 })
                }
                SCOPE_FAILS => Err(BackendError::KernelLaunchFailed("mock scope".into())),
                _ => {
                    self.c.scope_live.fetch_add(1, Ordering::SeqCst);
                    Ok(fandhe_ai_tensor_core::DeviceChainScope::holding(Box::new(
                        ScopeGuard(Arc::clone(&self.c.scope_live)),
                    )))
                }
            }
        }
        fn synchronize_device(&self) -> Result<(), BackendError> {
            self.c.sync.fetch_add(1, Ordering::SeqCst);
            match self.fault {
                Fault::SyncUnsupported => Err(BackendError::Unsupported("mock sync".into())),
                Fault::SyncFails => Err(BackendError::KernelLaunchFailed("mock sync".into())),
                Fault::SyncCaptureBusy => {
                    Err(BackendError::DeviceContextCaptureInProgress { ordinal: 0 })
                }
                _ => Ok(()),
            }
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
            self.c
                .live_at_sgd
                .lock()
                .unwrap()
                .push(self.c.scope_live.load(Ordering::SeqCst));
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
            if self.fault == Fault::MaskUnsupported {
                return Err(BackendError::Unsupported("mock mask".into()));
            }
            Ok(buf(m.shape(), 1.0))
        }
        fn linear_forward_device_into_tracked(
            &self,
            _a: &DeviceBuffer<f32>,
            _w: DeviceBufferView<'_>,
            _b: Option<DeviceBufferView<'_>>,
            _act: Activation,
            _out: &mut DeviceBuffer<f32>,
            t: &DispatchFailureCell,
        ) -> Result<(), BackendError> {
            self.run_into_part(t)
        }
        fn mse_loss_device_into_tracked(
            &self,
            _p: &DeviceBuffer<f32>,
            _t: &DeviceBuffer<f32>,
            _r: MseReduction,
            _out: &mut DeviceBuffer<f32>,
            t: &DispatchFailureCell,
        ) -> Result<(), BackendError> {
            self.run_into_part(t)
        }
        fn mse_loss_backward_device_into_tracked(
            &self,
            _p: &DeviceBuffer<f32>,
            _t: &DeviceBuffer<f32>,
            _s: f32,
            _out: &mut DeviceBuffer<f32>,
            t: &DispatchFailureCell,
        ) -> Result<(), BackendError> {
            self.run_into_part(t)
        }
        fn relu_backward_mask_device_into_tracked(
            &self,
            _g: DeviceBufferView<'_>,
            _l: DeviceMatrixLayout,
            _m: DeviceBufferView<'_>,
            _out: &mut DeviceBuffer<f32>,
            t: &DispatchFailureCell,
        ) -> Result<(), BackendError> {
            self.run_into_part(t)
        }
        fn gemm_resident_lhs_device_into_tracked(
            &self,
            _w: DeviceBufferView<'_>,
            _b: DeviceBufferView<'_>,
            _l: DeviceMatrixLayout,
            _out: &mut DeviceBuffer<f32>,
            t: &DispatchFailureCell,
        ) -> Result<(), BackendError> {
            self.run_into_part(t)
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
        setup_with(fault, false, false)
    }

    fn setup_with(
        fault: Fault,
        into_supported: bool,
        into_stale: bool,
    ) -> (Tape, DeviceParamStore, Arc<Counters>) {
        let c = Arc::new(Counters::default());
        let tape = Tape::new_with_ops(Box::new(Rec {
            c: Arc::clone(&c),
            fault,
            into_supported,
            into_stale,
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
        store.train_step_device_chain_once(tape, x, y, &steps, config)
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
    fn mask_failure_keeps_d_input_buffer_held_until_gpu_work_completes() {
        let (tape, mut store, c) = setup(Fault::MaskUnsupported);
        let err = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
        assert_eq!(count(&c.lhs), 1);
        assert_eq!(count(&c.mask), 1);
        // forward 出力 3 + g_cur 1 + gemm_resident_lhs の出力 1 + loss 1。
        // gemm の出力を mask 成功後にしか hold へ置かないと 1 本欠けて GPU 実行中に解放される。
        assert_eq!(store.train_chain_hold.len(), 6);
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
    fn wait_device_capture_conflict_does_not_poison_the_store() {
        let (tape, store, _c) = setup(Fault::SyncCaptureBusy);
        let err = store.wait_device(&tape).unwrap_err();
        assert!(
            matches!(err, BackendError::DeviceContextCaptureInProgress { .. }),
            "{err:?}"
        );
        store.check_not_poisoned().unwrap();
        // 衝突解消後の再試行は poison されずに到達する（失敗トークンも未設定）。
        let err2 = store.wait_device(&tape).unwrap_err();
        assert!(matches!(
            err2,
            BackendError::DeviceContextCaptureInProgress { .. }
        ));
        store.check_not_poisoned().unwrap();
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

    /// 3 層（forward 3・MSE 1・dPred 1・d_input 2・mask 2）の `_into` 部品の合計回数。
    const INTO_PARTS_PER_STEP: usize = 3 + 1 + 1 + 2 + 2;

    #[test]
    fn into_supported_backend_allocates_nothing_in_steady_state() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let x = store.upload_resident_input(&tape, &t(&[M, D_IN])).unwrap();
        let y = store.upload_resident_input(&tape, &t(&[M, D_OUT])).unwrap();
        // 初回 step: 確保版で作り、能力判定のため `_into` を 1 回ずつ試す。
        let l1 = run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.9)).unwrap();
        assert_eq!(count(&c.forward), 3);
        // 能力判定は部品ごとに 1 回（5 部品）。
        assert_eq!(count(&c.into_ok), N_PARTS);
        let (fwd, mse, bwd, lhs, mask) = (
            count(&c.forward),
            count(&c.mse),
            count(&c.mse_bwd),
            count(&c.lhs),
            count(&c.mask),
        );
        let (alloc, up, down) = (count(&c.alloc), count(&c.upload), count(&c.download));
        let into_before = count(&c.into_ok);
        // 2・3 step 目（定常状態）: 確保版の部品呼び出し 0・alloc_zeroed 0・upload／download 0。
        for _ in 0..2 {
            run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.9)).unwrap();
        }
        assert_eq!(count(&c.forward), fwd);
        assert_eq!(count(&c.mse), mse);
        assert_eq!(count(&c.mse_bwd), bwd);
        assert_eq!(count(&c.lhs), lhs);
        assert_eq!(count(&c.mask), mask);
        assert_eq!(count(&c.alloc), alloc);
        assert_eq!(count(&c.upload), up);
        assert_eq!(count(&c.download), down);
        assert_eq!(count(&c.into_ok), into_before + 2 * INTO_PARTS_PER_STEP);
        // dW・SGD は従来どおり step ごとに 1 セット。L1 の d_input は計算しない（#3245）。
        assert_eq!(count(&c.dw), 9);
        assert_eq!(count(&c.sgd), 3);
        // 古いハンドルは世代検査が拒否し、最新は読める。
        assert!(store.read_train_step_loss(&tape, &l1).is_err());
        let l4 = run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.9)).unwrap();
        assert_eq!(store.read_train_step_loss(&tape, &l4).unwrap(), 0.5);
    }

    #[test]
    fn into_unsupported_backend_falls_back_per_part_without_surfacing_unsupported() {
        let (tape, mut store, c) = setup_with(Fault::None, false, false);
        for _ in 0..3 {
            run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        }
        // 毎 step 確保版の部品を使う（結果は従来と同じ）。`_into` は初回の能力判定で 1 回ずつ
        // 試されて以後ラッチされる（5 部品）。
        assert_eq!(count(&c.forward), 9);
        assert_eq!(count(&c.mse), 3);
        assert_eq!(count(&c.lhs), 6);
        assert_eq!(count(&c.mask), 6);
        assert_eq!(count(&c.into_calls), 5);
        assert_eq!(count(&c.into_ok), 0);
        store.check_not_poisoned().unwrap();
        assert_eq!(store.step_count, 3);
    }

    #[test]
    fn changed_batch_or_width_rebuilds_the_scratch() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        let into_before = count(&c.into_ok);
        let fwd_before = count(&c.forward);
        // バッチ行数が変わる → 鍵不一致で再利用せず確保版で作り直す（以後はまた再利用）。
        let x2 = store
            .upload_resident_input(&tape, &t(&[M + 2, D_IN]))
            .unwrap();
        let y2 = store
            .upload_resident_input(&tape, &t(&[M + 2, D_OUT]))
            .unwrap();
        run_with(&tape, &mut store, &x2, &y2, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.forward), fwd_before + 3);
        // 能力は確定済み（probed）のため再判定の `_into` は呼ばれない。loss（shape `[]`）だけは
        // 鍵に依らず再利用される。
        assert_eq!(count(&c.into_ok), into_before + 1);
        run_with(&tape, &mut store, &x2, &y2, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.forward), fwd_before + 3);
        assert_eq!(count(&c.into_ok), into_before + 1 + INTO_PARTS_PER_STEP);
    }

    #[test]
    fn stale_scratch_is_dropped_and_reallocated() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        // 世代切れを返す設定へ差し替える代わりに、新しい mock で 2 step 目から stale を返す。
        let (tape2, mut store2, c2) = setup_with(Fault::None, true, true);
        run(&tape2, &mut store2, &ACTS, &cfg(0.0)).unwrap();
        run(&tape2, &mut store2, &ACTS, &cfg(0.0)).unwrap();
        // 初回は能力判定の `_into` が世代切れ（確定せず）、2 step 目も再利用に失敗して確保版へ進む。
        assert_eq!(count(&c2.forward), 6);
        assert_eq!(count(&c2.mse), 2);
        assert_eq!(count(&c2.into_ok), 0);
        store2.check_not_poisoned().unwrap();
        assert_eq!(store2.step_count, 2);
        // 対照: 世代切れでない mock は 2 step 目で確保版を呼ばない。
        run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.forward), 3);
    }

    #[test]
    fn failed_step_buffers_are_held_and_not_reused() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        assert!(store.train_chain_bufs.is_some());
        // 2 step 目を最初の dW 後の失敗にする（Rec の fault は固定のため新ストアで検証）。
        let (tape2, mut store2, _c2) = setup_with(Fault::LhsUnsupported, true, false);
        assert!(run(&tape2, &mut store2, &ACTS, &cfg(0.0)).is_err());
        assert!(
            store2.train_chain_bufs.is_none(),
            "失敗 step のバッファは再利用しない"
        );
        assert!(!store2.train_chain_hold.is_empty());
        let _ = c;
    }

    // ---- イシュー #3254: facade 公開入口のための内部入口 ----

    /// 事前検証は store を変えない（pending・世代・step_count・params・確保・encode なし）。
    #[test]
    fn validate_does_not_touch_store_or_encode() {
        let (tape, store, c) = setup(Fault::None);
        let x = store.upload_resident_input(&tape, &t(&[M, D_IN])).unwrap();
        let y = store.upload_resident_input(&tape, &t(&[M, D_OUT])).unwrap();
        let leaves = store.snapshot_resident_params(&tape).unwrap();
        let steps: Vec<ChainStep<'_, '_>> = ACTS
            .iter()
            .enumerate()
            .map(|(i, a)| (&leaves[2 * i], Some(&leaves[2 * i + 1]), *a))
            .collect();
        let gen_before = store.train_chain_generation;
        store
            .validate_train_step_device_chain(&tape, &x, &y, &steps, &cfg(0.9))
            .unwrap();
        assert_eq!(store.step_count, 0);
        assert!(!store.sgd_used);
        assert!(store.velocity.is_none());
        assert_eq!(store.train_chain_generation, gen_before);
        assert_eq!(count(&c.forward) + count(&c.sgd) + count(&c.dw), 0);
        // 範囲外（最終層 ReLU）は encode 前に Unsupported。
        let bad: Vec<ChainStep<'_, '_>> = [Activation::Relu; 3]
            .iter()
            .enumerate()
            .map(|(i, a)| (&leaves[2 * i], Some(&leaves[2 * i + 1]), *a))
            .collect();
        let r = store.validate_train_step_device_chain(&tape, &x, &y, &bad, &cfg(0.0));
        assert!(matches!(r, Err(BackendError::Unsupported(_))), "{r:?}");
        assert_eq!(count(&c.forward), 0);
    }

    /// フォールバックの loss は同期なしで読め、世代が進むと古いデバイスハンドルは拒否される。
    #[test]
    fn host_loss_handle_reads_without_download_and_invalidates_older_handles() {
        let (tape, mut store, c) = setup(Fault::None);
        let dev = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        let host = store.record_host_train_step_loss(&tape, 1.25).unwrap();
        let before = count(&c.download);
        assert_eq!(store.read_train_step_loss(&tape, &host).unwrap(), 1.25);
        assert_eq!(count(&c.download), before, "host loss は download しない");
        let r = store.read_train_step_loss(&tape, &dev);
        assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    }

    /// チェーンが成功すると古い host ハンドルは拒否され、新しいデバイスハンドルが読める。
    #[test]
    fn chain_after_host_loss_invalidates_host_handle() {
        let (tape, mut store, _c) = setup(Fault::None);
        let host = store.record_host_train_step_loss(&tape, 9.0).unwrap();
        let dev = run(&tape, &mut store, &ACTS, &cfg(0.0)).unwrap();
        let r = store.read_train_step_loss(&tape, &host);
        assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
        assert_eq!(store.read_train_step_loss(&tape, &dev).unwrap(), 0.5);
    }

    #[test]
    fn download_resident_input_checks_store_and_returns_host_copy() {
        let (tape, store, _c) = setup(Fault::None);
        let x = store.upload_resident_input(&tape, &t(&[M, D_IN])).unwrap();
        let host = store.download_resident_input(&tape, &x).unwrap();
        assert_eq!(host.shape(), &[M, D_IN]);
        let (tape2, store2, _c2) = setup(Fault::None);
        let other = store2
            .upload_resident_input(&tape2, &t(&[M, D_IN]))
            .unwrap();
        let r = store.download_resident_input(&tape, &other);
        assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    }

    #[test]
    fn wait_device_syncs_without_download_and_rejects_poisoned_store() {
        let (tape, store, c) = setup(Fault::None);
        let before = count(&c.download);
        store.wait_device(&tape).unwrap();
        assert_eq!(count(&c.sync), 1);
        assert_eq!(count(&c.download), before, "D2H なし");
        store.poisoned.store(true, Ordering::SeqCst);
        let r = store.wait_device(&tape);
        assert!(matches!(r, Err(BackendError::StorePoisoned)), "{r:?}");
        assert_eq!(count(&c.sync), 1, "poison 済みは待たずに拒否");
    }

    #[test]
    fn wait_device_rejects_set_failure_token_and_poisons() {
        let (tape, store, _c) = setup(Fault::None);
        store
            .failure_token
            .set(BackendError::KernelLaunchFailed("mock".into()));
        let r = store.wait_device(&tape);
        assert!(matches!(r, Err(BackendError::StorePoisoned)), "{r:?}");
        assert!(store.poisoned.load(Ordering::SeqCst));
    }

    /// `Unsupported` は状態を変えない（poison しない）。実行時エラーは poison する。
    #[test]
    fn wait_device_unsupported_keeps_store_usable_and_runtime_error_poisons() {
        let (tape, store, _c) = setup(Fault::SyncUnsupported);
        let r = store.wait_device(&tape);
        assert!(matches!(r, Err(BackendError::Unsupported(_))), "{r:?}");
        assert!(!store.poisoned.load(Ordering::SeqCst));
        let (tape, store, _c) = setup(Fault::SyncFails);
        assert!(store.wait_device(&tape).is_err());
        assert!(store.poisoned.load(Ordering::SeqCst));
    }

    /// 待機・再試行つきの公開入口（`train_step_device_chain`）で 1 step 実行する。
    fn run_retrying(
        tape: &Tape,
        store: &mut DeviceParamStore,
        x: &ResidentInput,
        y: &ResidentInput,
    ) -> Result<TrainStepLoss, BackendError> {
        let leaves = store.snapshot_resident_params(tape).unwrap();
        let steps: Vec<ChainStep<'_, '_>> = ACTS
            .iter()
            .enumerate()
            .map(|(i, a)| (&leaves[2 * i], Some(&leaves[2 * i + 1]), *a))
            .collect();
        store.train_step_device_chain(tape, x, y, &steps, &cfg(0.0))
    }

    fn io(tape: &Tape, store: &DeviceParamStore) -> (ResidentInput, ResidentInput) {
        (
            store.upload_resident_input(tape, &t(&[M, D_IN])).unwrap(),
            store.upload_resident_input(tape, &t(&[M, D_OUT])).unwrap(),
        )
    }

    /// 既定（`Ok(None)`）では capture 経路を 2 step 目以降だけ試し、結果は従来と同じ（R1・D3）。
    #[test]
    fn default_none_tries_capture_from_the_second_step_and_keeps_every_part_count() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        for step in 1..=3usize {
            run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
            // 1 step 目はウォームアップ（試さない）。2 step 目以降は 1 回ずつ試す。
            assert_eq!(count(&c.cap_calls), step - 1);
            // `Ok(None)` なら部品・SGD は従来どおり step ごとに全部積む。
            assert_eq!(count(&c.sgd), step);
            assert_eq!(store.step_count as usize, step);
        }
    }

    /// `into` 未対応のバックエンド（Metal・CPU 相当）では capture 経路を呼ばない。
    #[test]
    fn capture_is_not_tried_without_into_support() {
        let (tape, mut store, c) = setup_with(Fault::None, false, false);
        let (x, y) = io(&tape, &store);
        for _ in 0..3 {
            run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        }
        assert_eq!(count(&c.cap_calls), 0);
    }

    /// `Replayed` なら部品も SGD も積まず、状態・世代・loss 読み出しは成功 step と同じ。
    /// 渡す永続バッファは step をまたいで同じアドレス。
    #[test]
    fn replayed_step_skips_parts_and_sgd_and_updates_state() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        c.cap_mode.store(CAP_REPLAY, Ordering::SeqCst);
        let (fwd, sgd, gen_before) = (
            count(&c.forward),
            count(&c.sgd),
            store.train_chain_generation,
        );
        let mut losses = Vec::new();
        for _ in 0..2 {
            let downloads = count(&c.download);
            let allocs = count(&c.alloc);
            let loss = run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
            // 区間内の確保・download は 0。
            assert_eq!(count(&c.alloc), allocs);
            assert_eq!(count(&c.download), downloads);
            losses.push(loss);
        }
        assert_eq!(count(&c.forward), fwd);
        assert_eq!(count(&c.sgd), sgd);
        assert_eq!(store.step_count, 3);
        assert!(store.sgd_used);
        assert!(store.train_chain_generation > gen_before);
        // 古いハンドルは拒否され、最新は download 1 回で読める。
        assert!(store.read_train_step_loss(&tape, &losses[0]).is_err());
        let downloads = count(&c.download);
        assert_eq!(store.read_train_step_loss(&tape, &losses[1]).unwrap(), 0.5);
        assert_eq!(count(&c.download), downloads + 1);
        let ptrs = c.cap_ptrs.lock().unwrap();
        assert_eq!(ptrs.len(), 2);
        assert_eq!(
            ptrs[0], ptrs[1],
            "永続バッファは step をまたいで同じアドレス"
        );
        // 全バッファが store の failure_token を受ける。
        let expected = &store.failure_token as *const DispatchFailureCell as usize;
        assert!(c.tokens.lock().unwrap().contains(&expected));
    }

    /// バッチ形状が変わった直後は鍵が合わないので capture 経路を呼ばない。
    #[test]
    fn changed_batch_does_not_try_capture() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        let x2 = store
            .upload_resident_input(&tape, &t(&[M + 2, D_IN]))
            .unwrap();
        let y2 = store
            .upload_resident_input(&tape, &t(&[M + 2, D_OUT]))
            .unwrap();
        run_with(&tape, &mut store, &x2, &y2, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.cap_calls), 0);
        // 同じ形状が続けば次の step から試す。
        run_with(&tape, &mut store, &x2, &y2, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.cap_calls), 1);
    }

    /// 一過性の capture 衝突は poison せず、待って再試行して成功する（R6・D6）。
    #[test]
    fn capture_conflict_is_retried_without_poisoning() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        c.cap_mode.store(CAP_BUSY_ONCE, Ordering::SeqCst);
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        assert_eq!(count(&c.cap_calls), 2, "衝突 1 回 + 再試行 1 回");
        store.check_not_poisoned().unwrap();
        assert_eq!(store.step_count, 2);
    }

    /// 衝突が続けば待ち上限でエラーを返すが、poison せず状態も変えない。解消後は続けられる。
    #[test]
    fn persistent_capture_conflict_returns_error_without_poison_or_state_change() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        c.cap_mode.store(CAP_BUSY_ALWAYS, Ordering::SeqCst);
        let err = run_retrying(&tape, &mut store, &x, &y).unwrap_err();
        assert!(
            matches!(err, BackendError::DeviceContextCaptureInProgress { .. }),
            "{err:?}"
        );
        store.check_not_poisoned().unwrap();
        assert_eq!(store.step_count, 1);
        c.cap_mode.store(CAP_NONE, Ordering::SeqCst);
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        assert_eq!(store.step_count, 2);
    }

    /// 排他スコープは 1 step に 1 回だけ、capture 試行の後・SGD の前から関数を抜けるまで保持し、
    /// 成功でも各エラー分岐でも解放される（イシュー #3596）。
    #[test]
    fn chain_scope_is_held_once_across_encode_and_sgd_and_released_on_every_path() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.scope_opens), 1);
        assert_eq!(count(&c.scope_live), 0);
        assert_eq!(*c.live_at_sgd.lock().unwrap(), vec![1]);
        // 2 step 目は capture 試行（スコープの外）→ 不適用なので通常チェーン（スコープ内）。
        run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.scope_opens), 2);
        assert_eq!(count(&c.scope_live), 0);
        assert!(c.live_at_cap.lock().unwrap().iter().all(|&n| n == 0));
        for fault in [
            Fault::ForwardUnsupported,
            Fault::LhsUnsupported,
            Fault::SgdFails,
        ] {
            let (tape, mut store, c) = setup_with(fault, true, false);
            let (x, y) = io(&tape, &store);
            let _ = run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0));
            assert_eq!(
                count(&c.scope_live),
                0,
                "エラー分岐でもスコープは解放される"
            );
        }
    }

    /// capture が成功した step（`Replayed`）ではスコープを取らない。
    #[test]
    fn replayed_step_does_not_open_a_chain_scope() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        let opens = count(&c.scope_opens);
        c.cap_mode.store(CAP_REPLAY, Ordering::SeqCst);
        run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).unwrap();
        assert_eq!(count(&c.scope_opens), opens);
    }

    /// スコープ取得が capture 衝突を返しても（書き込み前）store は poison されず、新規確保した
    /// velocity は戻り、再試行で成功する。
    #[test]
    fn scope_acquisition_conflict_does_not_poison_and_is_retried() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        c.scope_mode.store(SCOPE_BUSY_ONCE, Ordering::SeqCst);
        let err = run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.9)).unwrap_err();
        assert!(matches!(
            err,
            BackendError::DeviceContextCaptureInProgress { .. }
        ));
        store.check_not_poisoned().unwrap();
        assert!(store.velocity.is_none(), "新規確保した velocity は戻る");
        assert_eq!(count(&c.forward), 0, "書き込み前に拒否される");
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        assert_eq!(store.step_count, 1);
        assert_eq!(count(&c.scope_live), 0);
    }

    /// スコープ取得の実行時エラー（poison 済み等）は従来どおり store を poison する。
    #[test]
    fn scope_acquisition_hard_failure_poisons_the_store() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        c.scope_mode.store(SCOPE_FAILS, Ordering::SeqCst);
        assert!(run_with(&tape, &mut store, &x, &y, &ACTS, &cfg(0.0)).is_err());
        assert!(matches!(
            store.check_not_poisoned(),
            Err(BackendError::StorePoisoned)
        ));
    }

    /// capture 経路の実行時エラーは書き込み後の可能性があるため store を poison する。
    #[test]
    fn capture_runtime_failure_poisons_the_store() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        c.cap_mode.store(CAP_FAILS, Ordering::SeqCst);
        assert!(run_retrying(&tape, &mut store, &x, &y).is_err());
        assert!(matches!(
            store.check_not_poisoned(),
            Err(BackendError::StorePoisoned)
        ));
    }

    /// capture 経路の後でも、失敗トークンがセット済みなら読み出し時に poison になる（A08）。
    #[test]
    fn token_set_after_a_replayed_step_poisons_the_store_on_read() {
        let (tape, mut store, c) = setup_with(Fault::None, true, false);
        let (x, y) = io(&tape, &store);
        run_retrying(&tape, &mut store, &x, &y).unwrap();
        c.cap_mode.store(CAP_REPLAY, Ordering::SeqCst);
        let loss = run_retrying(&tape, &mut store, &x, &y).unwrap();
        store
            .failure_token
            .set(BackendError::KernelLaunchFailed("late".into()));
        let err = store.read_train_step_loss(&tape, &loss).unwrap_err();
        assert!(matches!(err, BackendError::StorePoisoned), "{err:?}");
    }
}
