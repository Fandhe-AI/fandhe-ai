//! デバイス上パラメータ更新の状態機械（イシュー #935・
//! `docs/device-resident-update-design.md`。以下「設計文書」）。
//!
//! [`Tape`] はステップごとに生成・破棄される運用（`tape.rs`「学習ループ
//! での運用」節）だが、[`DeviceParamStore`] は `Tape` とは独立した寿命で
//! パラメータ（`weight`／`bias` 等）と momentum の velocity バッファを
//! **デバイス上に常駐**させ続ける。これにより学習ループは「パラメータの
//! 再アップロード」を学習開始時 1 回に縮退できる（設計文書 §3・§7「#935
//! への引き渡し事項」）。
//!
//! # 簡略化（設計文書からの意図的な逸脱）
//!
//! 設計文書 §3.3a は `TrainableParam { weight, bias }`／
//! `ResidentLeafVars { trainable_idx, weight, bias }` という層構造を持つ
//! 型を想定していたが、本実装は
//! `fandhe_ai::compat::Sequential::trainable_parameters()`／
//! `SequentialVars::trainable_vars()`／`trainable_grads()`（`facade` クレート。
//! 既に「層順に weight → bias」のフラットな `Vec` として位置対応契約を
//! 確立済み）にそのまま乗せられるよう、`DeviceParamStore` はパラメータを
//! 層構造を持たない**フラットな列**として扱う。呼び出し元（`facade`）が
//! `Sequential::trainable_parameters()` の並びをそのまま渡せば、既存の
//! 位置対応契約（`Sgd::step`／`AdamW::step` と同じ）を再利用できるため、
//! 層構造の受け渡しを二重実装しない（REQ-9「薄いラッパーに徹する」）。
//!
//! # パラメータ横断の単一連結バッファ化（イシュー #1023）
//!
//! 当初実装は `params: Vec<DeviceBuffer<f32>>`／
//! `velocities: Vec<Option<DeviceBuffer<f32>>>` として**パラメータごとに
//! 個別の `DeviceBuffer`** を保持し、`step()` の更新フェーズは
//! パラメータ数だけ「grad の upload → `sgd_step_device_tracked` 起動」を
//! 繰り返していた。Metal 実機（M4 Max）実測（`docs/perf/
//! device-resident-update-bench.md` §3〜§5）で resident update が legacy
//! 比 132〜152 倍遅いという後退が判明し、原因は「ディスパッチ単位あたり
//! 固定オーバーヘッド（Metal の command buffer commit +
//! `waitUntilCompleted`、CUDA の `launch` + `stream.synchronize()`）×
//! パラメータ数」と診断された（同 doc §4）。
//!
//! 本イシューでは、全パラメータを**単一の連結（フラット）
//! `DeviceBuffer<f32>`**（shape `[total_numel]`）としてデバイス上に常駐
//! させる方式（`ParamLayout` が各パラメータの元 shape・オフセット・
//! 要素数を保持し、往復時にのみ分割・復元する）へ再構成した。これにより
//! `step()` の更新フェーズは grad の upload・カーネル起動とも
//! **パラメータ数に依らず 1 回**になる（`BackendOps::sgd_step_device`／
//! `sgd_step_device_tracked` は要素単位で shape 非依存に定義されている
//! ため、連結しても各要素の入出力は per-param 呼び出しと bit 同一。
//! `BackendOps`／`MemoryOps` trait 自体・3 バックエンドのカーネル実装は
//! 無変更）。副次効果として forward 用の D2H download
//! （`register_resident_leaves`／`snapshot_resident_leaves`／
//! `sync_to_host`）も N 回 → 1 回に縮退する。
//!
//! Metal 実機・DGX Spark GB10 実機での再計測は本変更を実装したセッション
//! では未実施（`docs/perf/device-resident-update-bench.md` §6 追補・PR
//! 本文参照）。
//!
//! # 状態機械
//!
//! ```text
//! new() ──▶ [register_resident_leaves] ──▶ pending ──▶ [step] ──▶ (pending 消費)
//!              ▲                              │
//!              └──────── [abandon_pending_forward] ────┘
//! ```
//!
//! - `pending`（`Option<PendingForward>`）: forward で登録した葉ノードが
//!   まだ `step()` で消費されていない状態。2 回連続で
//!   `register_resident_leaves` を呼ぶと
//!   [`BackendError::PendingForwardUnconsumed`] で拒否する（設計文書
//!   §3.3a）。
//! - `poisoned`: `sgd_step_device` の実行時エラー（GPU 起動失敗等）後に
//!   遷移する。以降 `step`／`sync_to_host`／`register_resident_leaves`／
//!   `snapshot_resident_leaves` の 4 経路すべてを
//!   [`BackendError::StorePoisoned`] で拒否する（`.claude/rules/
//!   security.md` A08。部分的に更新されたデバイス側パラメータをそのまま
//!   学習継続・推論に使わせない）。回復は新しい `DeviceParamStore` の
//!   再構築のみ。単一連結バッファ化（#1023）後は「一部だけ更新された
//!   状態」自体が構造的に発生しなくなった（更新は 1 バッファへの単一
//!   カーネル起動のため部分適用がない）が、起動自体の失敗（バッファ
//!   全体が不定になりうる）に備えて poison 遷移は維持する。
//!
//! # 遅延失敗トークン経由の poison（イシュー #1017）
//!
//! Metal のコマンドバッファ共有バッチ（`backend-metal::context::
//! MetalContext::encode`／`synchronize`。`docs/backend-metal-command-
//! batching-design.md`）では、`step()` 呼び出し中の `sgd_step_device`
//! 自体は encode（バッファ結線のみ）で即座に成功を返し、実行時エラー
//! （GPU fault 等）はホスト実体化（`download`／`zero_fill`／`Drop`）まで
//! 判明しない。このため `step()` 呼び出し時点では検出できない失敗が
//! ある。`failure_token`（[`fandhe_ai_tensor_core::DispatchFailureCell`]）
//! を `ops.sgd_step_device_tracked` へ同一ロック区間で渡しておき、
//! `check_not_poisoned` が 4 つの状態機械エントリいずれかへの
//! 入口で `failure_token.is_set()` を検査して `poisoned` へ自己遷移する
//! （`ops` 側からの能動的な通知を待たない。CPU／CUDA は同期実行のため
//! `failure_token` を使わず、`step()` 内の即時エラーでこれまでどおり
//! `poisoned` へ遷移する）。

use std::sync::atomic::{AtomicBool, Ordering};

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{DispatchFailureCell, SgdStepConfig, Tensor};

use crate::backward::Gradients;
use crate::optim::sgd::SgdConfig;
use crate::tape::{NodeId, Tape, TapeId};
use crate::var::Var;

/// forward で登録済みだが `step()` にまだ消費されていない葉ノード列
/// （モジュール冒頭「状態機械」参照）。
#[derive(Debug)]
struct PendingForward {
    tape_id: TapeId,
    node_ids: Vec<NodeId>,
}

/// 連結フラットバッファ内での 1 パラメータ分の位置情報（イシュー #1023
/// 「パラメータ横断の単一連結バッファ化」）。`DeviceParamStore` は
/// パラメータを個別の `DeviceBuffer` としては保持せず、全パラメータを
/// 1 本の `DeviceBuffer<f32>`（shape `[total_numel]`）へ連結して常駐
/// させる。`shape`（download 時の復元・grad の shape 検証に使う）・
/// `offset`（フラットバッファ内の先頭要素位置）・`numel`（要素数）を
/// 元パラメータの登録順に保持する。
#[derive(Debug, Clone)]
struct ParamLayout {
    shape: Vec<usize>,
    offset: usize,
    numel: usize,
}

/// デバイス常駐パラメータ更新の状態機械本体（モジュール冒頭コメント
/// 参照）。単一デバイス固定（`new()` 時に `tape.ops().device()` から
/// 決定し、以後変更不可）。`Debug` は `unwrap_err()`（テストコード）が
/// `Result<DeviceParamStore, _>` を要求するため導出する（フィールドは
/// いずれも `Debug` 実装済み: `Device`／`DeviceBuffer<f32>`（`tensor-core`
/// 側で導出済み）／`ParamLayout`／`PendingForward`〈上記〉／
/// `AtomicBool`／`DispatchFailureCell`）。
///
/// `poisoned` は `AtomicBool`（`Cell<bool>` ではない）とする: `&self`
/// メソッド（`sync_to_host`／`snapshot_resident_leaves`）からも
/// `check_not_poisoned` が「遅延失敗トークンを検出して自己
/// poison する」ために書き込みが必要であり、`&self` からの内部可変性を
/// 要求するため（モジュール冒頭「遅延失敗トークン経由の poison」参照）。
#[derive(Debug)]
pub struct DeviceParamStore {
    device: Device,
    /// 各パラメータの元 shape・フラットバッファ内オフセット・要素数
    /// （登録順。#1023）。パラメータ件数は `layout.len()` から得る
    /// （`len()`／`is_empty()` はこちらを参照する。`params`／`velocity`
    /// の要素数〈`total_numel`〉とは意味が異なる）。
    layout: Vec<ParamLayout>,
    /// `layout` の `numel` 合計（`params`／`velocity` の shape）。
    total_numel: usize,
    /// 全パラメータを連結した単一バッファ（shape `[total_numel]`。
    /// #1023）。
    params: DeviceBuffer<f32>,
    /// momentum の velocity（`params` と同形。momentum 有効時に初回
    /// `step()` で遅延確保する。#1023 で `Vec<Option<DeviceBuffer<f32>>>`
    /// から単一連結バッファへ変更）。
    velocity: Option<DeviceBuffer<f32>>,
    step_count: u64,
    poisoned: AtomicBool,
    pending: Option<PendingForward>,
    /// `ops.sgd_step_device_tracked` へ渡す共有失敗トークン
    /// （モジュール冒頭「遅延失敗トークン経由の poison」参照）。
    failure_token: DispatchFailureCell,
}

impl DeviceParamStore {
    /// `params`（呼び出し元の位置対応契約に従うホスト常駐パラメータ列。
    /// 典型的には `fandhe_ai::compat::Sequential::trainable_parameters()`
    /// の戻り値）を 1 本の連結バッファへまとめ、`tape` のバックエンドへ
    /// **1 回だけ** アップロードして `DeviceParamStore` を構築する
    /// （#1023）。
    ///
    /// `tape.ops().memory_ops()` が `None`（`MemoryOps` 未実装バックエンド）
    /// の場合は [`BackendError::Unsupported`] を返す（`memory_ops()` を
    /// 呼ぶフォールバック合成は設けない。設計文書 §3.2 改訂）。momentum の
    /// velocity バッファはここでは確保しない（初回 `step()` で遅延確保。
    /// PyTorch `torch.optim.SGD` の `state['momentum_buffer']` 初期化
    /// タイミングと同じ。`fandhe_ai_autodiff::optim::sgd::Sgd` の
    /// `velocity` フィールド doc 参照）。
    pub fn new(tape: &Tape, params: &[&Tensor<f32>]) -> Result<DeviceParamStore, BackendError> {
        let device = tape.ops().device();
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::new: backend does not implement MemoryOps".to_string(),
            )
        })?;

        // 連結: 各パラメータの contiguous 要素列を登録順に 1 本の
        // `Vec<f32>` へ積み、`layout` に offset/numel を記録する
        // （要素の並び替えは行わないため、per-param upload と同じ
        // 要素順序を保つ）。
        let mut layout = Vec::with_capacity(params.len());
        let mut flat: Vec<f32> = Vec::new();
        let mut total_numel: usize = 0;
        for p in params {
            let contiguous = p.contiguous();
            let data = contiguous.as_slice().unwrap_or(&[]);
            let numel = data.len();
            let offset = total_numel;
            total_numel = total_numel.checked_add(numel).ok_or_else(|| {
                BackendError::InvalidArgument(
                    "DeviceParamStore::new: total parameter element count overflows usize"
                        .to_string(),
                )
            })?;
            layout.push(ParamLayout {
                shape: p.shape().to_vec(),
                offset,
                numel,
            });
            flat.extend_from_slice(data);
        }
        let flat_tensor = Tensor::new(flat, &[total_numel]).map_err(BackendError::ShapeMismatch)?;
        let params_buf = mem.upload(&flat_tensor)?;

        Ok(DeviceParamStore {
            device,
            layout,
            total_numel,
            params: params_buf,
            velocity: None,
            step_count: 0,
            poisoned: AtomicBool::new(false),
            pending: None,
            failure_token: DispatchFailureCell::new(),
        })
    }

    /// このストアが常駐するデバイス。
    pub fn device(&self) -> Device {
        self.device
    }

    /// このストアが保持するパラメータ件数（連結バッファの要素数
    /// `total_numel` ではなく `layout` の登録件数。#1023）。
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    /// パラメータ件数が 0 か判定する（`clippy::len_without_is_empty` 対応）。
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// 4 つの状態機械エントリ（`step`／`sync_to_host`／
    /// `register_resident_leaves`／`snapshot_resident_leaves`）が共通で
    /// 呼ぶ poison 検査。既に `poisoned` なら即座に拒否する。まだ
    /// `poisoned` でなくても `failure_token`（イシュー #1017）が
    /// 設定済み（Metal バッチの遅延実行時エラーが `synchronize` 時点で
    /// 判明した）なら、ここで `poisoned` へ自己遷移してから拒否する
    /// （モジュール冒頭「遅延失敗トークン経由の poison」参照。`ops` 側
    /// からの能動的な通知を待たず、次にストアへアクセスした経路が
    /// 検出する設計）。
    fn check_not_poisoned(&self) -> Result<(), BackendError> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(BackendError::StorePoisoned);
        }
        if self.failure_token.is_set() {
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(BackendError::StorePoisoned);
        }
        Ok(())
    }

    fn check_device(&self, tape: &Tape) -> Result<(), BackendError> {
        if tape.ops().device() != self.device {
            return Err(BackendError::DeviceMismatch);
        }
        Ok(())
    }

    /// 連結バッファを **1 回だけ** ダウンロードし、`layout` に従って
    /// 元の shape ごとの `Tensor<f32>` 列（登録順）へ分割する
    /// （`register_resident_leaves`／`snapshot_resident_leaves`／
    /// `sync_to_host` の 3 メソッドが共有する。#1023 で連結バッファ化
    /// した際に、パラメータごとに download を繰り返していた旧実装から
    /// 切り出した）。`mem` は呼び出し元が `check_device` 済みの
    /// `tape.ops().memory_ops()` を渡す契約。
    fn download_split(&self, mem: &dyn MemoryOps) -> Result<Vec<Tensor<f32>>, BackendError> {
        let flat = mem.download(&self.params)?;
        let contiguous = flat.contiguous();
        let data = contiguous.as_slice().unwrap_or(&[]);
        self.layout
            .iter()
            .map(|l| {
                let slice = data[l.offset..l.offset + l.numel].to_vec();
                Tensor::new(slice, &l.shape).map_err(BackendError::ShapeMismatch)
            })
            .collect()
    }

    /// forward 用: 現在デバイス上にある各パラメータを `tape` の葉ノードと
    /// して**毎回新規登録**し（`tape.var(&tensor)`。既存 `Tape::var` と
    /// 同じ「使い捨て」契約）、`step()` が消費するまで
    /// [`BackendError::PendingForwardUnconsumed`] で以後の再登録を拒否する
    /// （モジュール冒頭「状態機械」参照）。
    ///
    /// ダウンロードした `Tensor<f32>` は forward が消費するために必要な
    /// ホスト側実体化であり、param のホスト**再アップロード**（本イシュー
    /// が排除する対象）とは別（設計文書 §3.3b: 「削減は param 再 upload
    /// のみ。VJP 用 weight/bias download と grad upload は本段階で残る」）。
    /// #1023 により download 自体も連結バッファ 1 回（`download_split`）
    /// へ縮退した。
    pub fn register_resident_leaves<'t>(
        &mut self,
        tape: &'t Tape,
    ) -> Result<Vec<Var<'t>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if self.pending.is_some() {
            return Err(BackendError::PendingForwardUnconsumed);
        }
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::register_resident_leaves: backend does not implement \
                 MemoryOps"
                    .to_string(),
            )
        })?;
        let tensors = self.download_split(mem)?;
        // Cursor Bugbot 指摘（PR #1057）対応: `download` 呼び出し中に
        // 別スレッドが先に `synchronize` して GPU fault を検出し
        // `failure_token` をセットした場合、この関数冒頭の
        // `check_not_poisoned` はその発生前に通過済みのため見逃す
        // （`MetalContext::synchronize` は commit 済みバッチを一度だけ
        // drain するため、後続の `download` 内 `synchronize` 呼び出しは
        // 既に空の committed リストに対して成功扱いで返り、実際は無効な
        // バッファ内容を読んだ可能性がある）。`download` 完了直後にも
        // 再検査し、その間に poison 化していれば結果を破棄して拒否する。
        self.check_not_poisoned()?;
        let mut vars = Vec::with_capacity(tensors.len());
        let mut node_ids = Vec::with_capacity(tensors.len());
        for tensor in &tensors {
            let var = tape.var(tensor);
            node_ids.push(var.node_id());
            vars.push(var);
        }
        self.pending = Some(PendingForward {
            tape_id: tape.id,
            node_ids,
        });
        Ok(vars)
    }

    /// 推論用の読み取り専用版（`register_resident_leaves` と異なり
    /// `pending` 状態を変化させない。`step()` で消費する必要がないため）。
    pub fn snapshot_resident_leaves<'t>(
        &self,
        tape: &'t Tape,
    ) -> Result<Vec<Var<'t>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::snapshot_resident_leaves: backend does not implement \
                 MemoryOps"
                    .to_string(),
            )
        })?;
        let tensors = self.download_split(mem)?;
        // `register_resident_leaves` と同じレース（Cursor Bugbot 指摘・
        // PR #1057）への対処: `download` の間に別スレッドが GPU fault を
        // 検出し `failure_token` を設定した可能性があるため、返却直前に
        // 再検査する。
        self.check_not_poisoned()?;
        Ok(tensors.iter().map(|t| tape.var(t)).collect())
    }

    /// `pending`（未消費の forward 登録）を副作用なくクリアする冪等
    /// メソッド。呼び出し元が forward の結果を使わずに中断する場合
    /// （例: 例外的なバッチスキップ）に呼ぶ。`poisoned` は解除しない
    /// （モジュール冒頭「状態機械」参照）。戻り値はクリア前に `pending`
    /// が存在したか（冪等性の可視化。2 回目以降の呼び出しは `false`）。
    pub fn abandon_pending_forward(&mut self) -> bool {
        self.pending.take().is_some()
    }

    /// `grads`（`tape.backward(...)` の結果）を使って SGD の 1 ステップを
    /// **デバイス上で in-place に**適用する（本イシューの受け入れ条件の
    /// 本体）。
    ///
    /// 処理は「①`TapeId` 一致検査 → ②事前検証（勾配存在・shape）→
    /// ③更新（grad を連結して 1 回 upload → `sgd_step_device_tracked` を
    /// 1 回起動）」の順（設計文書 §3.3a・#1023 で③をパラメータ横断の
    /// 単一カーネル起動へバッチ化）。②で 1 件でも失敗すれば**どの
    /// パラメータも更新せずに** `Err` を返す（fail-closed。
    /// `.claude/rules/security.md` A03）。③の実行時エラー（GPU 起動失敗
    /// 等）は最初のエラーを返しつつ `poisoned` へ遷移する（モジュール
    /// 冒頭「状態機械」参照。連結バッファは単一カーネル起動で更新される
    /// ため「一部だけ更新された」状態は構造的に発生しないが、起動失敗時の
    /// バッファ内容は不定になりうるため隔離する）。
    pub fn step(
        &mut self,
        tape: &Tape,
        grads: &Gradients,
        config: &SgdConfig,
    ) -> Result<(), BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        // `Sgd::new` を経由しないため（本 `step` は毎呼び出しで `SgdConfig`
        // を直接適用する。モジュール冒頭「簡略化」節）、`Sgd::new` と
        // 同基準の検証（有限値・非負・nesterov 条件）をここで明示的に
        // 通す（Review 指摘: 検証なしに不正な `SgdConfig` がデバイス側
        // パラメータへ in-place 適用されてしまう抜け穴があった）。事前
        // 検証フェーズの一部として、どのパラメータも更新する前に検査する
        // （fail-closed。`.claude/rules/security.md` A03）。
        config
            .validate()
            .map_err(|e| BackendError::InvalidArgument(e.to_string()))?;

        // `pending` はこの事前検証フェーズ全体を通じて `take()` せず参照
        // だけで検査する（Review 指摘: 全事前検証より先に `take()` すると、
        // 勾配欠落・shape 不一致・momentum 構成不一致・`MemoryOps` 未提供・
        // velocity 確保失敗のいずれでもパラメータ未更新のまま pending
        // 登録が失われ、`register_resident_leaves` からやり直す必要が
        // 生じてしまう）。実際に `take()` するのは②更新フェーズへ入る
        // 直前（＝以降のエラーはどのみち `poisoned` へ遷移し `pending` の
        // 意味を失う地点）のみ。
        let Some(pending) = self.pending.as_ref() else {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore::step: no pending forward registration (call \
                 register_resident_leaves first)"
                    .to_string(),
            ));
        };
        if pending.tape_id != tape.id {
            // 呼び出し元の誤り（別 Tape を渡した）であり、ストア自体は
            // 壊れていない。`take()` していないため `pending` は復元不要
            // でそのまま残る。
            return Err(BackendError::TapeMismatch);
        }

        // ① 事前検証フェーズ: 全パラメータの勾配が揃っているか・shape が
        // 一致するかを、いずれのバッファも更新する前に検査する
        // （fail-closed。`SequentialVars::trainable_grads` と同じ方針）。
        // 検証を通った要素のみを連結先バッファ `flat_grad` へ積む
        // （#1023: どの要素も upload される前に全件検証済みであることは
        // 変わらない。`flat_grad` への追記は upload そのものではない）。
        let vars: Vec<Var<'_>> = pending
            .node_ids
            .iter()
            .map(|&id| Var::from_raw(tape, id))
            .collect();
        let mut flat_grad: Vec<f32> = Vec::with_capacity(self.total_numel);
        for (i, var) in vars.iter().enumerate() {
            let grad = grads.get(var).map_err(|_| BackendError::TapeMismatch)?;
            let grad = grad.ok_or_else(|| {
                BackendError::MissingGradient(format!(
                    "DeviceParamStore::step: parameter {i} has no gradient (loss unreachable)"
                ))
            })?;
            if grad.shape() != self.layout[i].shape.as_slice() {
                return Err(BackendError::InvalidArgument(format!(
                    "DeviceParamStore::step: gradient shape {:?} does not match parameter {i} \
                     shape {:?}",
                    grad.shape(),
                    self.layout[i].shape
                )));
            }
            let contiguous = grad.contiguous();
            flat_grad.extend_from_slice(contiguous.as_slice().unwrap_or(&[]));
        }

        let use_momentum = config.momentum != 0.0;

        // momentum 構成の途中変更を拒否する（設計文書 §3.3 事前検証
        // フェーズ「件数・shape・device・momentum 構成」・
        // `BackendError::InvalidArgument` の doc コメント「件数・momentum
        // 構成不一致」）。`step_count > 0` かつ velocity バッファの有無
        // （`momentum_state_exists`）と今回の `use_momentum` が食い違う
        // 場合、`is_first_step` の意味が曖昧になる: momentum を途中から
        // 有効化すると velocity は未確保のままここへ来て遅延確保される
        // が、`is_first_step` は false のため PyTorch の
        // `b ← g`（初回ルール）ではなく `b ← μ・0 + (1-τ)・g` を通って
        // しまい、`dampening != 0` のとき数値が食い違う（ホスト参照実装
        // `Sgd::step` にはこの経路自体が存在しない）。途中で無効化する
        // 場合も既存 velocity を握ったまま無視することになり、後で
        // 再度有効化した際の意味が不定になるため同様に拒否する。
        let momentum_state_exists = self.velocity.is_some();
        if self.step_count > 0 && momentum_state_exists != use_momentum {
            return Err(BackendError::InvalidArgument(format!(
                "DeviceParamStore::step: momentum configuration changed mid-training \
                 (previously {momentum_state_exists}, now {use_momentum}); reconstruct the \
                 store to change momentum settings"
            )));
        }

        let is_first_step = self.step_count == 0;
        let ops = tape.ops();
        let mem = ops.memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::step: backend does not implement MemoryOps".to_string(),
            )
        })?;

        // momentum が有効かつ未確保なら、連結バッファと同形（shape
        // `[total_numel]`）の velocity をここで遅延確保する（PyTorch の
        // 「初回は state 未生成、初回 step でゼロ初期化バッファを生成」
        // タイミングと同じ）。確保自体は「更新フェーズ」より前の準備
        // 段階であり、確保失敗はどのパラメータも更新しないまま `Err` を
        // 返す（fail-closed。poisoned へは遷移しない: 確保失敗は事前検証
        // と同種の「実行前に判明した失敗」であり、デバイス側パラメータは
        // 未変更のまま安全なため）。
        if use_momentum && self.velocity.is_none() {
            self.velocity = Some(mem.alloc_zeroed(&[self.total_numel])?);
        }

        // ここまでの事前検証（勾配存在・shape・momentum 構成・
        // `MemoryOps` 提供・velocity 確保）を全て通過し、以降は実際に
        // デバイス側パラメータを更新するフェーズへ入る。ここで初めて
        // `pending` を消費する（以降のエラーは `poisoned` 遷移で `pending`
        // の意味自体が失われるため、これより手前で `take()` しない）。
        self.pending = None;

        // ② 更新フェーズ（#1023）: 連結済み grad を **1 回だけ** upload
        // し、`sgd_step_device_tracked` を **1 回だけ** 起動する
        // （旧実装はパラメータ数だけ upload・起動を繰り返していた。
        // モジュール冒頭「パラメータ横断の単一連結バッファ化」参照）。
        // 実行時エラーは最初のエラーを返しつつ `poisoned` へ遷移する
        // （モジュール冒頭「状態機械」参照）。
        let grad_tensor = match Tensor::new(flat_grad, &[self.total_numel])
            .map_err(BackendError::ShapeMismatch)
        {
            Ok(t) => t,
            Err(e) => {
                self.poisoned.store(true, Ordering::SeqCst);
                return Err(e);
            }
        };
        let grad_buf = match mem.upload(&grad_tensor) {
            Ok(buf) => buf,
            Err(e) => {
                self.poisoned.store(true, Ordering::SeqCst);
                return Err(e);
            }
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
        // `sgd_step_device_tracked`（イシュー #1017）: Metal は encode
        // と同一ロック区間で `self.failure_token` をバッチへ登録する
        // ため、`step()` 自身の即時エラーに加え `check_not_poisoned`
        // 経由の遅延検出（モジュール冒頭コメント）でも取りこぼさない。
        // デフォルト実装（CPU／CUDA）は `sgd_step_device` へそのまま
        // 委譲するため、この呼び出し自体の挙動は変わらない。
        if let Err(e) = ops.sgd_step_device_tracked(
            &mut self.params,
            &grad_buf,
            velocity_ref,
            &step_config,
            &self.failure_token,
        ) {
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(e);
        }

        self.step_count += 1;
        Ok(())
    }

    /// 現在のデバイス上パラメータをホストへダウンロードする（明示同期
    /// 点。`fandhe_ai::compat::Sequential::apply_parameters` へそのまま
    /// 渡せる並び順で返す）。#1023 により連結バッファの download も
    /// 1 回（`download_split`）へ縮退した。
    pub fn sync_to_host(&self, tape: &Tape) -> Result<Vec<Tensor<f32>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::sync_to_host: backend does not implement MemoryOps".to_string(),
            )
        })?;
        let tensors = self.download_split(mem)?;
        // `register_resident_leaves`／`snapshot_resident_leaves` と同じ
        // レース（Cursor Bugbot 指摘・PR #1057）への対処。
        self.check_not_poisoned()?;
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use fandhe_ai_tensor_core::buffer::{BufferHandle, MemoryOps};
    use fandhe_ai_tensor_core::{Activation, BackendOps, Device, FusionPlan};

    /// `crate::test_support::TestOps` と同型（`eval.rs` へ委譲する naive
    /// 参照実装）だが、`MemoryOps`／`sgd_step_device` を追加実装した
    /// テスト専用モック。`autodiff` は `backend-cpu` 等の具体クレートに
    /// 依存できない（`tests/architecture_boundaries.rs`）ため、
    /// `DeviceParamStore` の状態機械（poisoned 遷移・`TapeId` 検査等）を
    /// 検証するにはこのモックが唯一の手段となる。
    ///
    /// `fail_after`（`Some(n)`）が設定されている場合、`sgd_step_device`
    /// の呼び出し累計が `n` 回目で `Err(BackendError::KernelLaunchFailed)`
    /// を返す（poisoned 遷移の検証用）。#1023 でパラメータ横断の単一
    /// 起動へバッチ化されたため、`step()` 1 回につき `sgd_step_device`
    /// 呼び出しはちょうど 1 回になる（`fail_after` はもはや「何個目の
    /// パラメータで失敗させるか」ではなく「何回目の `step()` 呼び出しで
    /// 失敗させるか」を意味する）。
    ///
    /// `call_count`／`upload_count`（#1023 追加）は `Arc<AtomicUsize>` と
    /// する: `MockDeviceOps` は `Box<dyn BackendOps + Send>` として
    /// `Tape` の内部（`tape.ops()` は `pub(crate)` かつ `&dyn BackendOps`
    /// を返すのみで `Any` へのダウンキャスト手段を持たない）へ move
    /// されるため、呼び出し回数をテスト側から観測するには構築前に
    /// `Arc` を clone して手元に残す必要がある
    /// （`step_launches_sgd_kernel_exactly_once_regardless_of_param_count`
    /// が「起動・grad upload ともパラメータ数に依らず 1 回／step」で
    /// あることを検査するために使う）。
    struct MockDeviceOps {
        fail_after: Option<usize>,
        call_count: Arc<AtomicUsize>,
        upload_count: Arc<AtomicUsize>,
    }

    impl MockDeviceOps {
        fn new() -> Self {
            Self {
                fail_after: None,
                call_count: Arc::new(AtomicUsize::new(0)),
                upload_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn failing_after(n: usize) -> Self {
            Self {
                fail_after: Some(n),
                call_count: Arc::new(AtomicUsize::new(0)),
                upload_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    struct MockHandle {
        data: std::cell::RefCell<Vec<f32>>,
    }

    impl std::fmt::Debug for MockHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockHandle").finish_non_exhaustive()
        }
    }

    impl BufferHandle for MockHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl MemoryOps for MockDeviceOps {
        fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
            let numel: usize = shape.iter().product();
            let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
                data: RefCell::new(vec![0.0; numel]),
            });
            Ok(DeviceBuffer::new(Device::Cpu, shape.to_vec(), handle))
        }

        fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
            self.upload_count.fetch_add(1, Ordering::SeqCst);
            let data = tensor.contiguous().as_slice().unwrap_or(&[]).to_vec();
            let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
                data: RefCell::new(data),
            });
            Ok(DeviceBuffer::new(
                Device::Cpu,
                tensor.shape().to_vec(),
                handle,
            ))
        }

        fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
            let handle = buffer
                .downcast_handle::<MockHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            let data = handle.data.borrow().clone();
            Tensor::new(data, buffer.shape()).map_err(BackendError::ShapeMismatch)
        }
    }

    impl BackendOps for MockDeviceOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn memory_ops(&self) -> Option<&dyn MemoryOps> {
            Some(self)
        }

        fn sgd_step_device(
            &self,
            param: &mut DeviceBuffer<f32>,
            grad: &DeviceBuffer<f32>,
            velocity: Option<&mut DeviceBuffer<f32>>,
            config: &SgdStepConfig,
        ) -> Result<(), BackendError> {
            let call_index = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_after == Some(call_index) {
                return Err(BackendError::KernelLaunchFailed(
                    "MockDeviceOps: simulated sgd_step_device failure".into(),
                ));
            }
            let grad_handle = grad
                .downcast_handle::<MockHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            let grad_data = grad_handle.data.borrow();

            let use_momentum = config.momentum != 0.0;
            let velocity_handle = velocity
                .as_ref()
                .map(|v| {
                    v.downcast_handle::<MockHandle>()
                        .ok_or(BackendError::DeviceMismatch)
                })
                .transpose()?;

            let param_handle = param
                .downcast_handle::<MockHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            let mut param_data = param_handle.data.borrow_mut();
            let mut velocity_data = velocity_handle.map(|h| h.data.borrow_mut());

            for j in 0..param_data.len() {
                let p = param_data[j];
                let mut g = grad_data[j];
                if config.weight_decay != 0.0 {
                    g = config.weight_decay.mul_add(p, g);
                }
                if use_momentum {
                    let v = velocity_data.as_mut().expect("momentum enabled");
                    let prev = v[j];
                    let b = if config.is_first_step {
                        g
                    } else {
                        config.momentum.mul_add(prev, (1.0 - config.dampening) * g)
                    };
                    v[j] = b;
                    g = if config.nesterov {
                        config.momentum.mul_add(b, g)
                    } else {
                        b
                    };
                }
                param_data[j] = p - config.lr * g;
            }
            Ok(())
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
            let out_shape = fandhe_ai_tensor_core::reduce_out_shape(a.shape(), dim)
                .map_err(BackendError::ShapeMismatch)?;
            Ok(crate::eval::sum(a, dim, &out_shape))
        }

        fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            let out_shape = fandhe_ai_tensor_core::reduce_out_shape(a.shape(), dim)
                .map_err(BackendError::ShapeMismatch)?;
            Ok(crate::eval::max(a, dim, &out_shape))
        }
    }

    fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    fn simple_tape(fail_after: Option<usize>) -> Tape {
        let ops: Box<dyn BackendOps + Send> = match fail_after {
            Some(n) => Box::new(MockDeviceOps::failing_after(n)),
            None => Box::new(MockDeviceOps::new()),
        };
        Tape::new_with_ops(ops)
    }

    /// `simple_tape` と同じだが、`MockDeviceOps` の呼び出し回数カウンタ
    /// （#1023 追加）を `Tape` へ move する前に `Arc::clone` して手元に
    /// 残す。`tape.ops()` は `pub(crate)` かつ `&dyn BackendOps` を返す
    /// のみで `Any` ダウンキャスト手段を持たないため、これが呼び出し
    /// 回数をテスト側から観測する唯一の経路となる。
    fn tape_with_counters(fail_after: Option<usize>) -> (Tape, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let mock = match fail_after {
            Some(n) => MockDeviceOps::failing_after(n),
            None => MockDeviceOps::new(),
        };
        let call_count = Arc::clone(&mock.call_count);
        let upload_count = Arc::clone(&mock.upload_count);
        let ops: Box<dyn BackendOps + Send> = Box::new(mock);
        (Tape::new_with_ops(ops), call_count, upload_count)
    }

    #[test]
    fn new_uploads_params_and_sync_to_host_roundtrips() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0, 2.0], &[2]);
        let b = tensor(vec![0.5, -0.5], &[2]);
        let store = DeviceParamStore::new(&tape, &[&w, &b]).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.device(), Device::Cpu);

        let synced = store.sync_to_host(&tape).unwrap();
        assert_eq!(synced[0].get(&[0]).unwrap(), 1.0);
        assert_eq!(synced[1].get(&[1]).unwrap(), -0.5);
    }

    /// エンドツーエンド: register → forward → backward → step → sync が
    /// vanilla SGD の手計算と一致することを検証する。
    fn train_one_step(momentum: f32) -> (Tensor<f32>, Tensor<f32>) {
        let tape = simple_tape(None);
        let w_init = tensor(vec![1.0, 2.0], &[2]);
        let b_init = tensor(vec![0.5, -0.5], &[2]);
        let mut store = DeviceParamStore::new(&tape, &[&w_init, &b_init]).unwrap();

        let vars = store.register_resident_leaves(&tape).unwrap();
        let x = tape.var(&tensor(vec![2.0, 3.0], &[2]));
        let target = tape.var(&tensor(vec![10.0, 10.0], &[2]));

        let pred = vars[0].mul(&x).unwrap().add(&vars[1]).unwrap();
        let loss = pred.mse_loss(&target).unwrap();
        let grads = tape.backward(&loss).unwrap();

        let mut config = SgdConfig::new(0.1);
        if momentum != 0.0 {
            config = config.with_momentum(momentum);
        }
        store.step(&tape, &grads, &config).unwrap();

        let synced = store.sync_to_host(&tape).unwrap();
        (synced[0].clone(), synced[1].clone())
    }

    #[test]
    fn full_pipeline_updates_parameters() {
        let (w, b) = train_one_step(0.0);
        // w/b は更新前から変化しているはず（勾配がゼロでない限り）。
        assert_ne!(w.get(&[0]).unwrap(), 1.0);
        assert_ne!(b.get(&[0]).unwrap(), 0.5);
    }

    #[test]
    fn full_pipeline_with_momentum_updates_parameters() {
        let (w, _b) = train_one_step(0.9);
        assert_ne!(w.get(&[0]).unwrap(), 1.0);
    }

    /// momentum を途中で有効化すると `InvalidArgument` で拒否される
    /// （`step()` の momentum 構成検査。レビュー指摘対応）。
    #[test]
    fn enabling_momentum_mid_training_is_rejected() {
        let tape = simple_tape(None);
        let w_init = tensor(vec![1.0, 2.0], &[2]);
        let mut store = DeviceParamStore::new(&tape, &[&w_init]).unwrap();

        // 1 ステップ目は momentum 無効で成功させる（step_count が 0 → 1）。
        let vars = store.register_resident_leaves(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0, 10.0], &[2]));
        let loss = vars[0].mse_loss(&target).unwrap();
        let grads = tape.backward(&loss).unwrap();
        store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();

        // 2 ステップ目で momentum を有効化すると拒否される。
        let vars = store.register_resident_leaves(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0, 10.0], &[2]));
        let loss = vars[0].mse_loss(&target).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let err = store
            .step(&tape, &grads, &SgdConfig::new(0.1).with_momentum(0.9))
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));

        // momentum 構成拒否は `poisoned` へは遷移せず（実行前に判明した
        // 失敗であり、デバイス側パラメータは未変更のまま安全なため）、
        // `pending` も失われない（codex-review PR #954 P2 是正: `step()` は
        // 事前検証フェーズ全体を `pending.take()` せず参照だけで通し、
        // ②更新フェーズへ入る直前にのみ消費する）。そのため
        // `register_resident_leaves` を呼び直さず、同じ `grads` で正しい
        // 設定の `step` を再試行するだけで成功する。
        store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();
    }

    #[test]
    fn step_without_pending_forward_is_rejected() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1]));
        let loss = x.mse_loss(&x).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let err = store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn registering_resident_leaves_twice_is_rejected() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();
        store.register_resident_leaves(&tape).unwrap();
        let err = store.register_resident_leaves(&tape).unwrap_err();
        assert!(matches!(err, BackendError::PendingForwardUnconsumed));
    }

    #[test]
    fn abandon_pending_forward_allows_reregistration() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();
        store.register_resident_leaves(&tape).unwrap();
        assert!(store.abandon_pending_forward());
        assert!(!store.abandon_pending_forward(), "2 回目は冪等に false");
        // 破棄後は再登録できる。
        store.register_resident_leaves(&tape).unwrap();
    }

    #[test]
    fn step_with_mismatched_tape_is_rejected_and_pending_is_restored() {
        let tape1 = simple_tape(None);
        let tape2 = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape1, &[&w]).unwrap();
        let vars1 = store.register_resident_leaves(&tape1).unwrap();

        let x2 = tape2.var(&tensor(vec![1.0], &[1]));
        let loss2 = x2.mse_loss(&x2).unwrap();
        let grads2 = tape2.backward(&loss2).unwrap();
        let err = store
            .step(&tape2, &grads2, &SgdConfig::new(0.1))
            .unwrap_err();
        assert!(matches!(err, BackendError::TapeMismatch));

        // pending は復元されているため、正しい tape・正しい grads を
        // 与えれば `step` は成功する（`register_resident_leaves` を
        // 呼び直す必要はない）。
        let loss1 = vars1[0]
            .mse_loss(&tape1.var(&tensor(vec![2.0], &[1])))
            .unwrap();
        let grads1 = tape1.backward(&loss1).unwrap();
        store.step(&tape1, &grads1, &SgdConfig::new(0.1)).unwrap();
    }

    #[test]
    fn step_with_missing_gradient_is_rejected() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let unused = tensor(vec![2.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &unused]).unwrap();
        let vars = store.register_resident_leaves(&tape).unwrap();
        // loss は vars[0] のみに依存し、vars[1] へは勾配が流れない。
        let loss = vars[0].mse_loss(&vars[0]).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let err = store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap_err();
        assert!(matches!(err, BackendError::MissingGradient(_)));
    }

    #[test]
    fn poisoned_after_failing_step_blocks_all_four_entry_points() {
        // #1023 でパラメータ横断の単一起動へバッチ化されたため、1 個目の
        // `step()` 呼び出し（内部の `sgd_step_device` 起動 1 回）自体を
        // 失敗させる（`fail_after` は「何個目のパラメータ」ではなく
        // 「何回目の `step()` 呼び出し」を指す。単一バッチ起動の失敗で
        // poisoned へ遷移することを検証する）。
        let tape = simple_tape(Some(1));
        let w = tensor(vec![1.0], &[1]);
        let b = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &b]).unwrap();
        let vars = store.register_resident_leaves(&tape).unwrap();
        let loss = vars[0].mul(&vars[1]).unwrap().mse_loss(&vars[0]).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let err = store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap_err();
        assert!(matches!(err, BackendError::KernelLaunchFailed(_)));

        // 4 経路すべてが StorePoisoned で拒否される。
        assert!(matches!(
            store.step(&tape, &grads, &SgdConfig::new(0.1)),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.sync_to_host(&tape),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.register_resident_leaves(&tape),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.snapshot_resident_leaves(&tape),
            Err(BackendError::StorePoisoned)
        ));
    }

    /// イシュー #1017: Metal のコマンドバッファ共有バッチでは
    /// `sgd_step_device_tracked` 自体は即座に成功を返し、実行時エラーは
    /// `synchronize`（ホスト実体化時）まで判明しない。本テストは
    /// `step()` が正常終了した**後**に外部（Metal 側の `synchronize`
    /// 相当）から `failure_token` へエラーを設定し、`step()` を含む
    /// 4 つの状態機械エントリすべてが次回アクセス時に
    /// `check_not_poisoned` 経由で `StorePoisoned` へ遷移することを
    /// 検証する（`poisoned_after_failing_step_blocks_all_four_entry_points`
    /// の「即時エラー」経路に対する「遅延エラー」経路の対）。
    #[test]
    fn deferred_failure_token_poisons_store_on_next_entry() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();

        let vars = store.register_resident_leaves(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0], &[1]));
        let loss = vars[0].mse_loss(&target).unwrap();
        let grads = tape.backward(&loss).unwrap();
        // 1 ステップ目は成功する（`sgd_step_device_tracked` のデフォルト
        // 委譲は `MockDeviceOps::sgd_step_device` へそのまま流れる）。
        store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();

        // Metal の `MetalContext::synchronize` が実行時エラーを検出し
        // `propagate_failure` でトークンへ書き込む経路を模擬する
        // （`failure_token` は `pub(crate)` ではなくプライベートフィールド
        // だが、`tests` は `device_store` の子モジュールのため直接
        // アクセスできる）。
        store.failure_token.set(BackendError::KernelLaunchFailed(
            "simulated deferred Metal command buffer failure".into(),
        ));

        // 4 経路すべてが StorePoisoned で拒否される
        // （即時エラー経路と同一の受け入れ条件）。
        assert!(matches!(
            store.step(&tape, &grads, &SgdConfig::new(0.1)),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.sync_to_host(&tape),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.register_resident_leaves(&tape),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.snapshot_resident_leaves(&tape),
            Err(BackendError::StorePoisoned)
        ));
    }

    #[test]
    fn new_on_backend_without_memory_ops_returns_unsupported() {
        // `crate::test_support::TestOps` は `memory_ops()` をオーバーライド
        // しない（既定 `None`）ため、`DeviceParamStore::new` は
        // `Unsupported` を返す（`memory_ops()` を呼ぶフォールバック合成は
        // 設けない設計。設計文書 §3.2 改訂）。
        let tape = Tape::new_with_ops(Box::new(crate::test_support::TestOps));
        let w = tensor(vec![1.0], &[1]);
        let err = DeviceParamStore::new(&tape, &[&w]).unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn empty_param_list_is_valid() {
        let tape = simple_tape(None);
        let store = DeviceParamStore::new(&tape, &[]).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    /// R3（#1023）の CI 上の機械検査: `step()` を複数回呼んでも
    /// `sgd_step_device`（デフォルト委譲経由）の起動回数・`upload` の
    /// 呼び出し回数がいずれもパラメータ件数に依らず「1 回／step」に
    /// なることを検証する（3 パラメータで 2 step 実行し、起動回数・
    /// grad upload 回数がともに 2 であることを確認）。
    #[test]
    fn step_launches_sgd_kernel_exactly_once_regardless_of_param_count() {
        let (tape, call_count, upload_count) = tape_with_counters(None);
        let w1 = tensor(vec![1.0, 2.0], &[2]);
        let w2 = tensor(vec![3.0], &[1]);
        let w3 = tensor(vec![4.0, 5.0, 6.0], &[3]);
        let mut store = DeviceParamStore::new(&tape, &[&w1, &w2, &w3]).unwrap();

        for _ in 0..2 {
            let vars = store.register_resident_leaves(&tape).unwrap();
            let target = tape.var(&tensor(vec![0.0, 0.0], &[2]));
            let loss0 = vars[0].mse_loss(&target).unwrap();
            let loss1 = vars[1].mse_loss(&vars[1]).unwrap();
            let loss2 = vars[2]
                .mse_loss(&tape.var(&tensor(vec![0.0, 0.0, 0.0], &[3])))
                .unwrap();
            let loss = loss0.add(&loss1).unwrap().add(&loss2).unwrap();
            let grads = tape.backward(&loss).unwrap();
            store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();
        }

        // `Tape::backward` 自体は `MockDeviceOps` の `MemoryOps` を経由
        // しないため、upload 回数の内訳は「`new()` で 1 回 + `step()` 毎に
        // 1 回（grad）」＝ 1 + 2 = 3 回のはず。
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "sgd_step_device の起動回数はパラメータ件数(3)に依らず step 回数(2)と一致するはず"
        );
        assert_eq!(
            upload_count.load(Ordering::SeqCst),
            3,
            "upload 回数は new() の 1 回 + step() 毎の grad upload 1 回 x 2 = 3 のはず"
        );
    }

    /// #1023: 異なる shape の複数パラメータを連結しても、`sync_to_host`
    /// （`download_split` 経由）で登録順どおりの shape・値へ正しく復元
    /// できることを検証する（連結・分割の往復整合性）。numel 0 の
    /// パラメータを混ぜても他パラメータのオフセット計算が壊れないことも
    /// あわせて確認する。
    #[test]
    fn flat_layout_roundtrips_shapes_and_values() {
        let tape = simple_tape(None);
        let a = tensor(vec![1.0, 2.0, 3.0], &[3]);
        let empty = Tensor::<f32>::new(vec![], &[0]).unwrap();
        let b = tensor(vec![10.0, 20.0], &[1, 2]);
        let store = DeviceParamStore::new(&tape, &[&a, &empty, &b]).unwrap();
        assert_eq!(store.len(), 3);

        let synced = store.sync_to_host(&tape).unwrap();
        assert_eq!(synced[0].shape(), &[3]);
        assert_eq!(synced[0].as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(synced[1].shape(), &[0]);
        assert_eq!(synced[2].shape(), &[1, 2]);
        assert_eq!(synced[2].as_slice().unwrap(), &[10.0, 20.0]);
    }

    /// #1023: `step()` の①事前検証フェーズ（勾配欠落）で拒否される場合、
    /// 連結先バッファへの `upload` が一切発生しないことを検証する
    /// （fail-closed。`step_with_missing_gradient_is_rejected` と同じ
    /// シナリオに `upload_count` の不変を追加した版）。
    #[test]
    fn missing_gradient_is_rejected_before_any_upload() {
        let (tape, _call_count, upload_count) = tape_with_counters(None);
        let w = tensor(vec![1.0], &[1]);
        let unused = tensor(vec![2.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &unused]).unwrap();
        let vars = store.register_resident_leaves(&tape).unwrap();
        // loss は vars[0] のみに依存し、vars[1] へは勾配が流れない。
        let loss = vars[0].mse_loss(&vars[0]).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let upload_before = upload_count.load(Ordering::SeqCst);

        let err = store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap_err();
        assert!(matches!(err, BackendError::MissingGradient(_)));
        assert_eq!(
            upload_count.load(Ordering::SeqCst),
            upload_before,
            "勾配欠落は upload 前に拒否されるはず"
        );
    }

    /// `BackendOps` の default メソッド（`gemm_bias_act`／`run_fused`）が
    /// `MockDeviceOps` に影響しないことのコンパイル時確認（object safety・
    /// `Activation`／`FusionPlan` の import が未使用にならないための
    /// 明示的な参照）。
    #[allow(dead_code)]
    fn assert_default_methods_exist(ops: &dyn BackendOps) {
        let a = Tensor::<f32>::scalar(1.0);
        let _ = ops.gemm_bias_act(&a, &a, None, Activation::None);
        let plan = FusionPlan::from_ops(
            vec![
                fandhe_ai_tensor_core::FusedOpKind::Input { leaf_index: 0 },
                fandhe_ai_tensor_core::FusedOpKind::Relu { input: 0 },
            ],
            vec![],
            fandhe_ai_tensor_core::DType::F32,
            1,
        );
        if let Ok(plan) = plan {
            let _ = ops.run_fused(&plan, &[&a]);
        }
    }
}
