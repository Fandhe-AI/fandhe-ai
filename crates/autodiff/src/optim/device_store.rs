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
//! （`register_resident_params`／`snapshot_resident_params`／
//! `sync_to_host`）も N 回 → 1 回に縮退する。
//!
//! Metal 実機・DGX Spark GB10 実機での再計測は本変更を実装したセッション
//! では未実施（`docs/perf/device-resident-update-bench.md` §6 追補・PR
//! 本文参照）。
//!
//! # 状態機械
//!
//! ```text
//! new() ──▶ [register_resident_params] ──▶ pending ──▶ [step] ──▶ (pending 消費)
//!              ▲                              │
//!              └──────── [abandon_pending_forward] ────┘
//! ```
//!
//! - `pending`（`Option<PendingForward>`）: forward で登録した葉ノードが
//!   まだ `step()` で消費されていない状態。2 回連続で
//!   `register_resident_params` を呼ぶと
//!   [`BackendError::PendingForwardUnconsumed`] で拒否する（設計文書
//!   §3.3a）。
//! - `poisoned`: `sgd_step_device` の実行時エラー（GPU 起動失敗等）後に
//!   遷移する。以降 `step`／`sync_to_host`／`register_resident_params`／
//!   `snapshot_resident_params` の 4 経路すべてを
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
//! （`ops` 側からの能動的な通知を待たない。CPU は同期実行のため
//! `failure_token` を使わず、`step()` 内の即時エラーでこれまでどおり
//! `poisoned` へ遷移する。CUDA はイシュー #1013（`docs/backend-cuda-
//! async-execution-design.md` §5）でカーネル起動直後の都度
//! `synchronize()` を除去し非同期実行契約へ移行しており、`failure_token`
//! は使わない（Metal 専用の機構のまま）。代わりに
//! `backend-cuda::context_cache` の ordinal 単位 poison 状態機械
//! （`begin_driver_call`／`observe_driver_result`／`observe_cuda_result`）
//! を PR #1064（イシュー #1013 の codex-review P0 指摘への対応）で
//! `backend-cuda::ops`／`backend-cuda::memory` の演算入口へ結線した
//! （同設計文書 §12）。sticky エラー（illegal address 等）は、それを
//! 最初に観測した同一 ordinal 上の driver 呼び出し時点で ordinal を
//! poison 化し、以降の `sgd_step_device` 呼び出しは `begin_driver_call`
//! の拒否により `Err` を返す。この `Err` は `step()` の
//! `if let Err(e) = ops.sgd_step_device_tracked(..) { self.poisoned.store
//! (true, ..); .. }` を経て必ず `StorePoisoned` への自己遷移につながる
//! （本ファイル下部 `step` 実装参照）。ただし検出は「エラー発生後、
//! 同一 ordinal 上で次に driver 呼び出しが起きた時点」に限られる
//! （sticky エラーの発生自体を即座に検出する仕組みではない。単一
//! ストリームの FIFO 順序保証を前提に、当該演算自身の次の driver
//! 呼び出し、または起動直後に同期を要求しない限り、その呼び出し自体で
//! 検出されることが多い）。poison からの**回復**（`context_cache::
//! invalidate_with` の呼び出し）は #1062 へ引き継いだままである。

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{Activation, DispatchFailureCell, SgdStepConfig, Tensor};

use crate::backward::Gradients;
use crate::error::AutodiffError;
use crate::optim::sgd::SgdConfig;
use crate::tape::{NodeId, Op, ResidentResolver, Tape, TapeId};
use crate::var::Var;

/// forward で登録済みだが `step()` にまだ消費されていない葉ノード列
/// （モジュール冒頭「状態機械」参照）。
#[derive(Debug)]
struct PendingForward {
    tape_id: TapeId,
    node_ids: Vec<NodeId>,
}

/// プロセス全体で共有する `store_id` 発行カウンタ（イシュー #1022）。
/// `tape::TapeId`（`tape.rs::NEXT_TAPE_ID`）と同じ理由（単調増加 ID は
/// プロセス生存中に衝突しない）で `DeviceParamStore::new` からのみ
/// インクリメントされる。`Op::ResidentLeaf { store_id, .. }` が
/// [`ResidentResolver::resident_buffer`] 経由で「どのストアの葉か」を
/// 一意に識別するための鍵になる。
static NEXT_STORE_ID: AtomicU64 = AtomicU64::new(0);

/// 連結フラットバッファ内での 1 パラメータ分の位置情報（イシュー #1023
/// 「パラメータ横断の単一連結バッファ化」）。`DeviceParamStore` は
/// パラメータを個別の `DeviceBuffer` としては保持せず、全パラメータを
/// 1 本の `DeviceBuffer<f32>`（shape `[total_numel]`）へ連結して常駐
/// させる。`shape`（download 時の復元・grad の shape 検証・
/// [`DeviceBufferView`] 構築に使う）・`offset`（フラットバッファ内の
/// 先頭要素位置）・`numel`（要素数）を元パラメータの登録順に保持する。
#[derive(Debug, Clone)]
struct ParamLayout {
    shape: Vec<usize>,
    offset: usize,
    numel: usize,
}

/// [`DeviceParamStore::register_resident_params`]／
/// [`DeviceParamStore::snapshot_resident_params`] が返す、テープ上の
/// `Op::ResidentLeaf` ノードへの不透明なハンドル（イシュー #1022・
/// #1023「R3: 要素オフセット付き常駐ビュー」）。
///
/// **`Var` ではなく専用の不透明型にする理由**: `Var::value()`／
/// `to_tensor()`（`var.rs`）は非 fallible な API であり、ホスト値を
/// 持たない `Op::ResidentLeaf` に対して呼ばれると「panic させる」か
/// 「黙示的にゼロを返す」のいずれかしか選べない。本型は `shape()` の
/// みを公開し、値アクセサ（`value()`／`to_tensor()` 相当）を持たない
/// ことで、呼び出し元がこの罠に到達する経路自体を型で塞ぐ
/// （`tape::Op::ResidentLeaf` doc 参照）。
///
/// `slot` は #1023 の連結バッファ化後も意味を変えず、`DeviceParamStore`
/// 内の登録順インデックス（`layout` の添字）を指す（連結バッファ内の
/// 生の要素オフセットではない。オフセットへの変換は
/// `DeviceParamStore::checked_resident_buffer` が `layout[slot]` から
/// 行う）。
///
/// ライフタイム `'t` は `Var<'t>` と同じく「元となった `Tape` の生存
/// 期間」を表す（このハンドル自体は `Tape` への参照を保持しないが、
/// `DeviceParamStore::linear_forward`／`backward` に渡す際に同じ `'t` の
/// `Tape`／`Var<'t>` としか組み合わせられないよう型で拘束する）。
#[derive(Debug, Clone)]
pub struct ResidentLeaf<'t> {
    node_id: NodeId,
    store_id: u64,
    slot: usize,
    shape: Vec<usize>,
    /// このリーフを発行した `Tape` の識別子（イシュー #1022 P1 是正・
    /// codex-review 指摘）。`'t` はライフタイムのみを拘束し `Tape` の
    /// 同一性までは保証しないため（同じ `'t` を持つ複数の `Tape` を
    /// 区別できない）、`linear_forward` はここを検査して `tape` 引数と
    /// 異なる `Tape` から発行された葉の混入を fail-closed に拒否する
    /// （`Var::tape_id()` と同じ設計。`.claude/rules/security.md` A08）。
    tape_id: TapeId,
    _marker: PhantomData<&'t Tape>,
}

impl<'t> ResidentLeaf<'t> {
    /// このパラメータのデバイス上 shape（実体化不要。`TapeNode.shape`
    /// と同じく構造的に確定済みの値をそのまま保持する）。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
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
/// メソッド（`sync_to_host`／`snapshot_resident_params`）からも
/// `check_not_poisoned` が「遅延失敗トークンを検出して自己
/// poison する」ために書き込みが必要であり、`&self` からの内部可変性を
/// 要求するため（モジュール冒頭「遅延失敗トークン経由の poison」参照）。
#[derive(Debug)]
pub struct DeviceParamStore {
    device: Device,
    /// このストアの一意識別子（イシュー #1022）。`Op::ResidentLeaf`／
    /// `ResidentLeaf` が保持し、`ResidentResolver::resident_buffer` が
    /// 「別ストアの葉が誤って混入していないか」を検査する鍵になる
    /// （`NEXT_STORE_ID` のドキュメンテーションコメント参照）。
    store_id: u64,
    /// 各パラメータの元 shape・フラットバッファ内オフセット・要素数
    /// （登録順。#1023）。パラメータ件数は `layout.len()` から得る
    /// （`len()`／`is_empty()` はこちらを参照する。`params`／`velocity`
    /// の要素数〈`total_numel`〉とは意味が異なる）。`ResidentLeaf::slot`
    /// はこの `Vec` の添字（イシュー #1022・#1023「R3」。
    /// `Self::checked_resident_buffer` が `layout[slot]` の
    /// `offset`／`shape` から [`DeviceBufferView`] を構築する）。
    layout: Vec<ParamLayout>,
    /// `layout` の `numel` 合計（`params`／`velocity` の shape）。
    total_numel: usize,
    /// 全パラメータを連結した単一バッファ（shape `[total_numel]`。
    /// #1023）。forward（`linear_forward`）は本バッファを download
    /// せず、`layout` のオフセットから作った [`DeviceBufferView`] 経由で
    /// 直接参照する（#1022 の D2H 排除と #1023 の単一連結バッファ化を
    /// 両立させる「R3」設計。`docs/device-resident-update-design.md`
    /// 追補参照）。
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

/// `AutodiffError` を `BackendError` へ変換する（イシュー #1022）。
/// [`DeviceParamStore::linear_forward`] は `Result<_, BackendError>` を
/// 返す契約（`BackendOps::gemm_resident_rhs` 等と同じエラー型）だが、
/// `materialize_fallible`（`tape.rs`）は `AutodiffError` を返すため橋渡し
/// が要る。`AutodiffError::Backend`（既に `BackendError` を包んでいる
/// 場合）はそのまま unwrap し、それ以外（`Shape`／`TapeMismatch`／
/// `InvalidArgument`／`Backward`）は `Display` 実装（`error.rs`）の
/// 文字列を保持したまま `BackendError::Unsupported` へ包む（エラー種別
/// を誤って別カテゴリへすり替えない範囲での最善の橋渡し。厳密な 1:1
/// 対応が必要になった場合は将来 `BackendError` 側に variant を追加する）。
fn autodiff_err_to_backend(err: AutodiffError) -> BackendError {
    match err {
        AutodiffError::Backend(be) => be,
        other => BackendError::Unsupported(other.to_string()),
    }
}

impl ResidentResolver for DeviceParamStore {
    /// `grad::vjp`（`Op::LinearResident` の VJP）から
    /// `Tape::backward_with_resident` 経由で呼ばれる（イシュー #1022）。
    /// `DeviceParamStore::checked_resident_buffer` へ委譲する薄い実装
    /// （`linear_forward` と同じ検証を共有する）。
    fn resident_buffer(
        &self,
        store_id: u64,
        slot: usize,
    ) -> Result<DeviceBufferView<'_>, AutodiffError> {
        self.checked_resident_buffer(store_id, slot)
            .map_err(AutodiffError::Backend)
    }
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
            store_id: NEXT_STORE_ID.fetch_add(1, Ordering::Relaxed),
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
    /// `register_resident_params`／`snapshot_resident_params`）が共通で
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
    /// （`sync_to_host` が使う。#1023 で連結バッファ化した際に、
    /// パラメータごとに download を繰り返していた旧実装から切り出した。
    /// 「R3」設計〈モジュール冒頭「パラメータ横断の単一連結バッファ化」
    /// 参照〉により `register_resident_params`／
    /// `snapshot_resident_params` は forward 用の葉登録に本メソッドを
    /// 使わなくなった——それらは download を伴わない
    /// `Self::checked_resident_buffer` 経由の [`DeviceBufferView`] を
    /// 使う）。`mem` は呼び出し元が `check_device` 済みの
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

    /// forward 用: 現在デバイス上にある各パラメータを `tape` の
    /// `Op::ResidentLeaf` ノードとして**毎回新規登録**し（`tape.
    /// push_resident_leaf(...)`。ホストへの download を伴わない。本
    /// イシュー #1022 の中核）、`step()` が消費するまで
    /// [`BackendError::PendingForwardUnconsumed`] で以後の再登録を拒否する
    /// （モジュール冒頭「状態機械」参照）。
    ///
    /// **#1022 による変更**: 旧実装（`register_resident_params`。現在は
    /// `#[deprecated]`）は `mem.download(buf)` でホストへ落としてから
    /// `tape.var(&tensor)`（`Op::Leaf`）に登録していた（設計文書 §3.3b が
    /// 「本段階で残る転送」と明記していたもの）。本メソッドはこの
    /// download を排除し、代わりに `Op::ResidentLeaf`（ホスト値を持たない
    /// テープノード）を登録して不透明型 [`ResidentLeaf`] を返す（`Var` を
    /// 返さない理由は同型の doc 参照）。呼び出し元は返った `ResidentLeaf`
    /// を [`Self::linear_forward`] へ渡して forward する。
    ///
    /// **#1023「R3」による変更**: `DeviceParamStore` はパラメータを
    /// 個別の `DeviceBuffer` としてではなく単一の連結バッファ
    /// （`self.params`）として保持するため、各パラメータは `self.layout`
    /// の添字（`slot`）で識別する。`shape` は `layout[slot].shape` から
    /// 取得し、実際のバッファ範囲（オフセット）へのアクセスは
    /// forward 時（[`Self::linear_forward`]）に
    /// `Self::checked_resident_buffer` が解決する。
    ///
    /// **命名（codex-review PR #1059 P1 是正）**: `DeviceParamStore` は
    /// `facade` の公開 API（crates.io `fandhe-ai` 0.4.0 で公開済み。
    /// `crates/facade/src/lib.rs` の再エクスポート）であり、旧
    /// `register_resident_params`（`Result<Vec<Var<'t>>, _>` を返す）の
    /// 戻り型を本メソッドへ直接差し替えると SemVer 破壊的変更になる。
    /// そのため常駐 forward（D2H を伴わない）経路は新規メソッド名
    /// `register_resident_params` として追加し、旧名はホスト
    /// download を伴う旧挙動のまま `#[deprecated]` として維持する
    /// （下記 [`Self::register_resident_params`] 参照）。
    pub fn register_resident_params<'t>(
        &mut self,
        tape: &'t Tape,
    ) -> Result<Vec<ResidentLeaf<'t>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if self.pending.is_some() {
            return Err(BackendError::PendingForwardUnconsumed);
        }
        let mut leaves = Vec::with_capacity(self.layout.len());
        let mut node_ids = Vec::with_capacity(self.layout.len());
        for (slot, layout) in self.layout.iter().enumerate() {
            let shape = layout.shape.clone();
            let node_id = tape.push_resident_leaf(shape.clone(), self.store_id, slot);
            node_ids.push(node_id);
            leaves.push(ResidentLeaf {
                node_id,
                store_id: self.store_id,
                slot,
                shape,
                tape_id: tape.id,
                _marker: PhantomData,
            });
        }
        self.pending = Some(PendingForward {
            tape_id: tape.id,
            node_ids,
        });
        Ok(leaves)
    }

    /// 推論用の読み取り専用版（[`Self::register_resident_params`] と
    /// 異なり `pending` 状態を変化させない。`step()` で消費する必要が
    /// ないため）。D2H を伴わず [`ResidentLeaf`] を返す（`Self::
    /// register_resident_params` と同じ理由・命名判断。上記 doc 参照）。
    pub fn snapshot_resident_params<'t>(
        &self,
        tape: &'t Tape,
    ) -> Result<Vec<ResidentLeaf<'t>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        Ok(self
            .layout
            .iter()
            .enumerate()
            .map(|(slot, layout)| {
                let shape = layout.shape.clone();
                let node_id = tape.push_resident_leaf(shape.clone(), self.store_id, slot);
                ResidentLeaf {
                    node_id,
                    store_id: self.store_id,
                    slot,
                    shape,
                    tape_id: tape.id,
                    _marker: PhantomData,
                }
            })
            .collect())
    }

    /// forward 用（**非推奨・互換維持版**）: 現在デバイス上にある各
    /// パラメータをホストへ **`download` してから**（D2H を伴う）
    /// `tape.var(&tensor)`（`Op::Leaf`）へ登録し、`step()` が消費するまで
    /// [`BackendError::PendingForwardUnconsumed`] で以後の再登録を拒否する
    /// （モジュール冒頭「状態機械」参照。イシュー #1022 以前の挙動を
    /// そのまま維持する）。
    ///
    /// **非推奨の理由（codex-review PR #1059 P1 是正）**: `DeviceParamStore`
    /// は `facade` の公開 API（crates.io `fandhe-ai` 0.4.0 で公開済み）
    /// であり、本メソッドの戻り型 `Result<Vec<Var<'t>>, BackendError>` は
    /// SemVer 契約の一部である。イシュー #1022 の D2H 排除（forward の
    /// たびに weight/bias をホストへ download しない）を導入する際、
    /// 戻り型を非破壊に変更できなかったため（`Var<'t>` と
    /// `ResidentLeaf<'t>` は異なる型）、D2H を伴わない新経路は
    /// [`Self::register_resident_params`] という別名で追加し、本メソッド
    /// は旧シグネチャ・旧挙動（download を伴う）のまま維持して既存
    /// 呼び出し元の互換性を壊さない。新規コードは
    /// [`Self::register_resident_params`]（`ResidentLeaf` を
    /// [`Self::linear_forward`] へ渡す設計）を使うこと。
    #[deprecated(
        since = "0.5.0",
        note = "毎 step の D2H（ホスト download）を伴う。D2H を排除した \
                register_resident_params を使うこと"
    )]
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

    /// 推論用の読み取り専用版（**非推奨・互換維持版**）。
    /// [`Self::register_resident_leaves`]（非推奨版）と異なり `pending`
    /// 状態を変化させない。非推奨の理由・移行先は同メソッドの doc を
    /// 参照（D2H を伴わない新経路は [`Self::snapshot_resident_params`]）。
    #[deprecated(
        since = "0.5.0",
        note = "毎回の D2H（ホスト download）を伴う。D2H を排除した \
                snapshot_resident_params を使うこと"
    )]
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
        // `register_resident_leaves`（非推奨版）と同じレース（Cursor
        // Bugbot 指摘・PR #1057）への対処: `download` の間に別スレッドが
        // GPU fault を検出し `failure_token` を設定した可能性があるため、
        // 返却直前に再検査する。
        self.check_not_poisoned()?;
        Ok(tensors.iter().map(|t| tape.var(t)).collect())
    }

    /// `store_id`／`slot` を検証したうえで、連結バッファ（`self.params`）
    /// 内の対応する要素範囲を指す [`DeviceBufferView`] を返す（イシュー
    /// #1022・#1023「R3: 要素オフセット付き常駐ビュー」）。
    /// [`Self::linear_forward`]・[`ResidentResolver::resident_buffer`] の
    /// 共通実装。別ストアの葉が混入した場合・`slot` が範囲外の場合は
    /// fail-closed に拒否する（`.claude/rules/security.md` A08）。範囲
    /// （`offset + numel <= self.params.numel()`）自体は
    /// [`DeviceBufferView::new`] が構築時に検証する（`layout` の不変条件
    /// より通常は必ず成功するが、本番経路で panic させない方針
    /// （`.claude/rules/coding-rust.md`）に従い `?` でそのまま伝播する）。
    fn checked_resident_buffer(
        &self,
        store_id: u64,
        slot: usize,
    ) -> Result<DeviceBufferView<'_>, BackendError> {
        if store_id != self.store_id {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore: resident leaf belongs to a different DeviceParamStore \
                 (store_id mismatch)"
                    .to_string(),
            ));
        }
        let layout = self.layout.get(slot).ok_or_else(|| {
            BackendError::InvalidArgument(format!(
                "DeviceParamStore: resident leaf slot {slot} is out of range (store has {} \
                 parameters)",
                self.layout.len()
            ))
        })?;
        DeviceBufferView::new(&self.params, layout.offset, &layout.shape)
    }

    /// forward 用: `weight`（・`bias`）をデバイス常駐のまま
    /// `BackendOps::gemm_resident_rhs`（`tensor-core`）へ渡し、
    /// `y = input.matmul(weight) (+ bias)` を計算してテープへ
    /// `Op::LinearResident` として記録する（イシュー #1022 の中核・
    /// #1023「R3」で連結バッファ内のオフセットビュー経由に変更）。
    ///
    /// `input`（活性化値）はホスト常駐のまま渡してよい（本イシューが
    /// 排除する D2H は weight／bias のものに限る。§1.2 の解釈）。
    /// `weight`／`bias` はいずれも `self`（同一 `DeviceParamStore`）が
    /// **同じ `tape` に対して** [`Self::register_resident_params`]／
    /// [`Self::snapshot_resident_params`] で返した [`ResidentLeaf`] で
    /// なければならない（`Self::checked_resident_buffer` が
    /// `store_id`／`slot` を検証し、本メソッドが `tape_id` を検証する。
    /// イシュー #1022 P1 是正: ライフタイム `'t` のみでは `Tape` の
    /// 同一性を保証しないため、`store_id`／`slot` の検証だけでは
    /// 「別の `Tape` 上で発行された同じストアの葉」を通してしまう
    /// 抜け穴があった。`grad::vjp` の `Op::LinearResident` 分岐が
    /// `nodes[weight.0]` を当該 `tape` の `nodes` に対して引くため、
    /// 別テープのノード ID が混入すると範囲外添字・無関係ノード誤読の
    /// 余地がある。ここで fail-closed に拒否することで、`grad.rs` 側の
    /// `nodes.get(...)` 検査と合わせた縦深防御とする）。
    ///
    /// `bias` は `Some` の場合 `[n]`（`weight` の列数）への厳密一致のみ
    /// 対応する（`BackendOps::gemm_resident_rhs` の融合カーネル契約と
    /// 同じ。ブロードキャスト全般は非対応のため、この形状検査はカーネル
    /// 本体へ触れる前にここで行う。REQ-8・OWASP A03）。
    pub fn linear_forward<'t>(
        &self,
        tape: &'t Tape,
        input: &Var<'t>,
        weight: &ResidentLeaf<'t>,
        bias: Option<&ResidentLeaf<'t>>,
    ) -> Result<Var<'t>, BackendError> {
        self.linear_forward_with_activation(tape, input, weight, bias, Activation::None)
    }

    /// [`Self::linear_forward`] の activation 融合版（イシュー #1044・
    /// `docs/kernel-fusion.md` §2.2「学習経路への結線」）。次層が
    /// `ReLU` の場合、呼び出し元（`fandhe_ai_facade::compat::sequential::
    /// Sequential::forward_from_flat_leaves`）が `act: Activation::Relu`
    /// を渡し、`ReLU` 層自体のノード追加をスキップする。層あたりの
    /// カーネル起動数を「gemm+bias（`Op::LinearResident`）→ 別ノードの
    /// `relu`」の 2 起動から「gemm+bias+act」の 1 起動へ減らす。
    ///
    /// `act` 以外の契約（`ResidentLeaf` の tape 一致検査・bias shape
    /// 厳密一致・D2H 排除）は [`Self::linear_forward`] と同一。
    pub fn linear_forward_with_activation<'t>(
        &self,
        tape: &'t Tape,
        input: &Var<'t>,
        weight: &ResidentLeaf<'t>,
        bias: Option<&ResidentLeaf<'t>>,
        act: Activation,
    ) -> Result<Var<'t>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if input.tape_id() != tape.id {
            return Err(BackendError::TapeMismatch);
        }
        if weight.tape_id != tape.id {
            return Err(BackendError::TapeMismatch);
        }
        if let Some(b) = bias
            && b.tape_id != tape.id
        {
            return Err(BackendError::TapeMismatch);
        }

        let w_buf = self.checked_resident_buffer(weight.store_id, weight.slot)?;
        let b_buf = match bias {
            Some(b) => Some(self.checked_resident_buffer(b.store_id, b.slot)?),
            None => None,
        };
        if let (Some(b), Some(n)) = (&b_buf, w_buf.shape().get(1))
            && b.shape() != [*n]
        {
            return Err(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![*n],
                },
            ));
        }

        // `input` のホスト値は `materialize_fallible`（層 1。`Var::
        // matmul` 等と同じ経路）で取得する。`Var::to_tensor()`（層 2・
        // 非 fallible）ではなくこちらを使う理由: forward 演算の失敗
        // （`Unsupported` 以外の `run_fused` エラー等）を黙示的に吸収
        // させず、本メソッドの `Result` へそのまま伝播させるため
        // （`docs/fusion-graph-design.md` §3.5.2 の層 1 契約）。
        let x_val = {
            // `nodes`（`pub(crate) RefCell<Vec<TapeNode>>`）はクレート内
            // 限定公開のため同一クレート（`autodiff`）内の本ファイルから
            // 直接借用できる（`var.rs::Var::matmul` と同じ借用パターン）。
            let nodes = tape.nodes.borrow();
            crate::tape::materialize_fallible(&nodes, tape.ops(), input.node_id())
                .map_err(autodiff_err_to_backend)?
                .clone()
        };

        let y = tape
            .ops()
            .gemm_resident_rhs_act(&x_val, w_buf, b_buf, act)?;
        let node_id = tape.push_eager(
            Op::LinearResident {
                input: input.node_id(),
                weight: weight.node_id,
                bias: bias.map(|b| b.node_id),
                act,
            },
            y,
        );
        Ok(Var::from_raw(tape, node_id))
    }

    /// `loss` から逆伝播し、常駐 weight／bias（`Op::ResidentLeaf`・
    /// `Op::LinearResident`）を含むグラフの勾配を計算する（イシュー
    /// #1022）。`self` を `ResidentResolver` として `tape.
    /// backward_with_resident` へ渡す薄いラッパー。
    ///
    /// **素の `tape.backward(loss)` との違い**: `Op::LinearResident` の
    /// VJP は weight のデバイスバッファ範囲を解決する手段
    /// （`ResidentResolver`）を要求するため、素の `backward` は
    /// `AutodiffError::InvalidArgument` で拒否する（`tape::Op::
    /// LinearResident` doc 参照）。`Sequential::forward_resident` で
    /// forward したグラフは必ず本メソッドで backward すること。
    pub fn backward(&self, tape: &Tape, loss: &Var<'_>) -> Result<Gradients, AutodiffError> {
        self.check_not_poisoned().map_err(AutodiffError::Backend)?;
        self.check_device(tape).map_err(AutodiffError::Backend)?;
        tape.backward_with_resident(loss, self)
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
        // 登録が失われ、`register_resident_params` からやり直す必要が
        // 生じてしまう）。実際に `take()` するのは②更新フェーズへ入る
        // 直前（＝以降のエラーはどのみち `poisoned` へ遷移し `pending` の
        // 意味を失う地点）のみ。
        let Some(pending) = self.pending.as_ref() else {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore::step: no pending forward registration (call \
                 register_resident_params first)"
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
        // `register_resident_params`／`snapshot_resident_params` と同じ
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
        /// `MemoryOps::download` の累計呼び出し回数（イシュー #1022 の
        /// 受け入れ条件 1「reuse 学習 1 step 内の D2H が loss 実体化以外
        /// 0 回」を機械検証するためのカウンタ）。`gemm_resident_rhs`／
        /// `gemm_resident_lhs`（下記）は本カウンタを増やさない実装
        /// （`downcast_handle` で直接読む。`backend-cpu::ops::
        /// CpuBackendOps` の「ゼロコピー」実装と同じモデル）とすることで、
        /// forward／backward が実際に `download()` を呼んでいないことを
        /// 区別して検証できる。
        download_count: Arc<AtomicUsize>,
    }

    impl MockDeviceOps {
        fn new() -> Self {
            Self {
                fail_after: None,
                call_count: Arc::new(AtomicUsize::new(0)),
                upload_count: Arc::new(AtomicUsize::new(0)),
                download_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn failing_after(n: usize) -> Self {
            Self {
                fail_after: Some(n),
                call_count: Arc::new(AtomicUsize::new(0)),
                upload_count: Arc::new(AtomicUsize::new(0)),
                download_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// カウンタの共有ハンドルを複製する（`Tape::new_with_ops` へ
        /// `self` の所有権を渡した後もテスト側から読み出せるようにする。
        /// `resident_forward_backward_has_zero_param_download` 参照）。
        fn download_counter(&self) -> std::sync::Arc<AtomicUsize> {
            self.download_count.clone()
        }

        /// [`DeviceBufferView`] が指す範囲の中身をコピー取得する
        /// （`download()` を経由しないため `download_count` を増やさない。
        /// `gemm_resident_rhs`／`gemm_resident_lhs` 専用のヘルパー。#1023
        /// 「R3」により連結バッファ内のオフセット範囲を読む形へ変更）。
        fn read_resident(view: DeviceBufferView<'_>) -> Result<Tensor<f32>, BackendError> {
            let handle = view
                .buffer()
                .downcast_handle::<MockHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            let data = handle.data.borrow();
            let slice = data[view.offset()..view.offset() + view.numel()].to_vec();
            Tensor::new(slice, view.shape()).map_err(BackendError::ShapeMismatch)
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
            self.download_count.fetch_add(1, Ordering::SeqCst);
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

        /// `w`（デバイス常駐）を [`MockDeviceOps::read_resident`] で直接
        /// 読み取り（`download()` を経由しないため `download_count` は
        /// 増えない。イシュー #1022 の受け入れ条件 1 を機械検証するテスト
        /// の前提）、`a @ w (+ bias)` をホスト側 `eval` で計算する。
        fn gemm_resident_rhs(
            &self,
            a: &Tensor<f32>,
            w: DeviceBufferView<'_>,
            bias: Option<DeviceBufferView<'_>>,
        ) -> Result<Tensor<f32>, BackendError> {
            let w_tensor = Self::read_resident(w)?;
            let mut y = crate::eval::matmul(a, &w_tensor);
            if let Some(bias) = bias {
                let b_tensor = Self::read_resident(bias)?;
                y = crate::eval::add(&y, &b_tensor);
            }
            Ok(y)
        }

        /// [`Self::gemm_resident_rhs`] と同じく `download()` を経由しない
        /// `w @ b` の計算（`Op::LinearResident` の VJP が `d_input` を
        /// 求めるために使う）。
        fn gemm_resident_lhs(
            &self,
            w: DeviceBufferView<'_>,
            b: &Tensor<f32>,
        ) -> Result<Tensor<f32>, BackendError> {
            let w_tensor = Self::read_resident(w)?;
            Ok(crate::eval::matmul(&w_tensor, b))
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

    /// 互換維持版（`#[deprecated]`。イシュー #1022 P1 是正・codex-review
    /// PR #1059 指摘）: `register_resident_leaves`／`snapshot_resident_leaves`
    /// が旧シグネチャ（`Result<Vec<Var<'t>>, _>`）・旧挙動（D2H を伴う）の
    /// まま維持されており、返った `Var` へ既存の演算（`mse_loss` 等）を
    /// 直接呼べることを検証する（`ResidentLeaf` を返す新経路
    /// `register_resident_params`／`snapshot_resident_params` とは異なり、
    /// `linear_forward` を経由せず素の `Var` 演算がそのまま使えるのが
    /// 旧経路の契約）。`#[allow(deprecated)]` はテスト関数単位で付与し、
    /// 非推奨警告（`-D warnings`）を抑制する。
    #[test]
    #[allow(deprecated)]
    fn deprecated_register_and_snapshot_resident_leaves_return_usable_vars() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0, 2.0], &[2]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();

        let vars = store.register_resident_leaves(&tape).unwrap();
        assert_eq!(vars.len(), 1);
        // 旧経路は download 済みホスト値を持つ `Var` を返すため、
        // `ResidentLeaf` と異なりそのまま演算できる（`Op::Leaf` 経由）。
        let target = tape.var(&tensor(vec![0.0, 0.0], &[2]));
        let loss = vars[0].mse_loss(&target).unwrap();
        let grads = tape.backward(&loss).unwrap();
        store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();

        let snapshot = store.snapshot_resident_leaves(&tape).unwrap();
        assert_eq!(snapshot.len(), 1);
        assert_ne!(snapshot[0].to_tensor().get(&[0]).unwrap(), 1.0);
    }

    /// エンドツーエンド: register → forward（`linear_forward`） →
    /// backward（`DeviceParamStore::backward`） → step → sync が vanilla
    /// SGD の手計算と一致することを検証する（#1022 で `Op::LinearResident`
    /// 経路へ書き換え）。`w_init` は `[2, 2]` の 2 次元行列（`linear_forward`
    /// の `matmul` 契約に合わせる。旧テストの `[2]` 要素ごとの `mul` とは
    /// 異なる形状だが、「forward→backward→step でパラメータが変化する」
    /// という検証意図は変わらない）。
    fn train_one_step(momentum: f32) -> (Tensor<f32>, Tensor<f32>) {
        let tape = simple_tape(None);
        let w_init = tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b_init = tensor(vec![0.5, -0.5], &[2]);
        let mut store = DeviceParamStore::new(&tape, &[&w_init, &b_init]).unwrap();

        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![2.0, 3.0], &[1, 2]));
        let target = tape.var(&tensor(vec![10.0, 10.0], &[1, 2]));

        let pred = store
            .linear_forward(&tape, &x, &leaves[0], Some(&leaves[1]))
            .unwrap();
        let loss = pred.mse_loss(&target).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();

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
        assert_ne!(w.get(&[0, 0]).unwrap(), 1.0);
        assert_ne!(b.get(&[0]).unwrap(), 0.5);
    }

    #[test]
    fn full_pipeline_with_momentum_updates_parameters() {
        let (w, _b) = train_one_step(0.9);
        assert_ne!(w.get(&[0, 0]).unwrap(), 1.0);
    }

    /// momentum を途中で有効化すると `InvalidArgument` で拒否される
    /// （`step()` の momentum 構成検査。レビュー指摘対応）。
    #[test]
    fn enabling_momentum_mid_training_is_rejected() {
        let tape = simple_tape(None);
        let w_init = tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let mut store = DeviceParamStore::new(&tape, &[&w_init]).unwrap();
        let x = tape.var(&tensor(vec![1.0, 1.0], &[1, 2]));

        // 1 ステップ目は momentum 無効で成功させる（step_count が 0 → 1）。
        let leaves = store.register_resident_params(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0, 10.0], &[1, 2]));
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let loss = pred.mse_loss(&target).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
        store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();

        // 2 ステップ目で momentum を有効化すると拒否される。
        let leaves = store.register_resident_params(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0, 10.0], &[1, 2]));
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let loss = pred.mse_loss(&target).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
        let err = store
            .step(&tape, &grads, &SgdConfig::new(0.1).with_momentum(0.9))
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));

        // momentum 構成拒否は `poisoned` へは遷移せず（実行前に判明した
        // 失敗であり、デバイス側パラメータは未変更のまま安全なため）、
        // `pending` も失われない（codex-review PR #954 P2 是正: `step()` は
        // 事前検証フェーズ全体を `pending.take()` せず参照だけで通し、
        // ②更新フェーズへ入る直前にのみ消費する）。そのため
        // `register_resident_params` を呼び直さず、同じ `grads` で正しい
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
        store.register_resident_params(&tape).unwrap();
        let err = store.register_resident_params(&tape).unwrap_err();
        assert!(matches!(err, BackendError::PendingForwardUnconsumed));
    }

    #[test]
    fn abandon_pending_forward_allows_reregistration() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();
        store.register_resident_params(&tape).unwrap();
        assert!(store.abandon_pending_forward());
        assert!(!store.abandon_pending_forward(), "2 回目は冪等に false");
        // 破棄後は再登録できる。
        store.register_resident_params(&tape).unwrap();
    }

    #[test]
    fn step_with_mismatched_tape_is_rejected_and_pending_is_restored() {
        let tape1 = simple_tape(None);
        let tape2 = simple_tape(None);
        let w = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape1, &[&w]).unwrap();
        let leaves1 = store.register_resident_params(&tape1).unwrap();
        let x1 = tape1.var(&tensor(vec![1.0], &[1, 1]));

        let x2 = tape2.var(&tensor(vec![1.0], &[1]));
        let loss2 = x2.mse_loss(&x2).unwrap();
        let grads2 = tape2.backward(&loss2).unwrap();
        let err = store
            .step(&tape2, &grads2, &SgdConfig::new(0.1))
            .unwrap_err();
        assert!(matches!(err, BackendError::TapeMismatch));

        // pending は復元されているため、正しい tape・正しい grads を
        // 与えれば `step` は成功する（`register_resident_params` を
        // 呼び直す必要はない）。
        let pred1 = store
            .linear_forward(&tape1, &x1, &leaves1[0], None)
            .unwrap();
        let loss1 = pred1
            .mse_loss(&tape1.var(&tensor(vec![2.0], &[1, 1])))
            .unwrap();
        let grads1 = store.backward(&tape1, &loss1).unwrap();
        store.step(&tape1, &grads1, &SgdConfig::new(0.1)).unwrap();
    }

    #[test]
    fn step_with_missing_gradient_is_rejected() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1, 1]);
        let unused = tensor(vec![2.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &unused]).unwrap();
        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));
        // loss は leaves[0] のみに依存し、leaves[1] へは勾配が流れない。
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let loss = pred.mse_loss(&pred).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
        let err = store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap_err();
        assert!(matches!(err, BackendError::MissingGradient(_)));
    }

    #[test]
    fn poisoned_after_failing_step_blocks_all_four_entry_points() {
        // #1023 でパラメータ横断の単一起動へバッチ化されたため（R3 設計
        // でも `step()` の更新フェーズは連結バッファへの単一カーネル
        // 起動のまま維持する）、1 個目の `step()` 呼び出し（内部の
        // `sgd_step_device` 起動 1 回）自体を失敗させる（`fail_after` は
        // 「何個目のパラメータ」ではなく「何回目の `step()` 呼び出し」を
        // 指す。単一バッチ起動の失敗で poisoned へ遷移することを検証
        // する）。2 層の `linear_forward` を連鎖させ、両パラメータへ
        // 勾配が流れる形にする（#1022 で `Op::LinearResident` 経路へ
        // 書き換え。旧テストの `vars[0].mul(&vars[1])` は `ResidentLeaf`
        // 同士の直接演算ができなくなったため代替）。`linear_forward` は
        // `weight` が 2 次元 shape を要求するため `w`／`b` は `[1, 1]`。
        let tape = simple_tape(Some(1));
        let w = tensor(vec![1.0], &[1, 1]);
        let b = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &b]).unwrap();
        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));
        let y1 = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let y2 = store.linear_forward(&tape, &y1, &leaves[1], None).unwrap();
        let loss = y2.mse_loss(&y1).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
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
            store.register_resident_params(&tape),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.snapshot_resident_params(&tape),
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
        let w = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();

        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let target = tape.var(&tensor(vec![10.0], &[1, 1]));
        let loss = pred.mse_loss(&target).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
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
            store.register_resident_params(&tape),
            Err(BackendError::StorePoisoned)
        ));
        assert!(matches!(
            store.snapshot_resident_params(&tape),
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

    /// イシュー #1022 受け入れ条件 1: reuse 学習 1 step 内（forward の
    /// `register_resident_params`／`linear_forward`・backward の
    /// `DeviceParamStore::backward`）で `MemoryOps::download`（D2H）が
    /// 0 回であることを機械検証する。`MockDeviceOps::gemm_resident_rhs`／
    /// `gemm_resident_lhs` は `downcast_handle` で直接読む実装
    /// （`download()` を経由しない）であり、`register_resident_params`／
    /// `snapshot_resident_params` 自体も本イシューで download を撤去
    /// 済みのため、`step()`／`sync_to_host()` を呼ばない本テストの範囲
    /// では `download_count()` が終始 0 のまま推移するはずである。
    #[test]
    fn resident_forward_backward_has_zero_param_download() {
        let ops = MockDeviceOps::new();
        let download_count = ops.download_counter();
        let tape = Tape::new_with_ops(Box::new(ops));
        let w_init = tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b_init = tensor(vec![0.5, -0.5], &[2]);
        let mut store = DeviceParamStore::new(&tape, &[&w_init, &b_init]).unwrap();

        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![2.0, 3.0], &[1, 2]));
        let target = tape.var(&tensor(vec![10.0, 10.0], &[1, 2]));
        let pred = store
            .linear_forward(&tape, &x, &leaves[0], Some(&leaves[1]))
            .unwrap();
        let loss = pred.mse_loss(&target).unwrap();
        let _grads = store.backward(&tape, &loss).unwrap();

        assert_eq!(
            download_count.load(Ordering::SeqCst),
            0,
            "register_resident_params/linear_forward/backward の 1 step 内で \
             MemoryOps::download が呼ばれてはならない（#1022 受け入れ条件 1）"
        );
    }

    /// 素の `Tape::backward`（`DeviceParamStore::backward` を経由しない）
    /// を `Op::LinearResident` を含むグラフへ適用すると、weight のデバイス
    /// バッファを解決する手段（`ResidentResolver`）がないため型付き
    /// エラーで拒否される（fail-closed。`tape::Op::LinearResident` doc
    /// 参照）。
    #[test]
    fn plain_backward_on_resident_graph_is_rejected() {
        let tape = simple_tape(None);
        let w = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w]).unwrap();
        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let loss = pred.mse_loss(&pred).unwrap();

        let err = tape.backward(&loss).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    /// 別ストアの `ResidentLeaf` を `linear_forward` に渡すと
    /// `store_id` 不一致で fail-closed に拒否される（`.claude/rules/
    /// security.md` A08）。
    #[test]
    fn linear_forward_rejects_leaf_from_a_different_store() {
        let tape = simple_tape(None);
        let w1 = tensor(vec![1.0], &[1, 1]);
        let w2 = tensor(vec![2.0], &[1, 1]);
        let store1 = DeviceParamStore::new(&tape, &[&w1]).unwrap();
        let mut store2 = DeviceParamStore::new(&tape, &[&w2]).unwrap();
        let leaves2 = store2.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));

        let err = store1
            .linear_forward(&tape, &x, &leaves2[0], None)
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
        // store2 側の pending は消費されていないため、store2 の
        // register_resident_params を呼び直す必要はなく引き続き有効。
        store2.abandon_pending_forward();
        let _ = store1;
    }

    /// 別 `Tape` で発行された `ResidentLeaf`（同じ `DeviceParamStore`）を
    /// `linear_forward` に渡すと `tape_id` 不一致で fail-closed に拒否
    /// される（イシュー #1022 P1 是正・codex-review 指摘）。`store_id`／
    /// `slot` の一致検査（`linear_forward_rejects_leaf_from_a_different_
    /// store`）だけでは `Tape` の同一性まで保証しないため、
    /// `linear_forward` 自身が `ResidentLeaf::tape_id` を検証する必要が
    /// あることを確認する。
    #[test]
    fn linear_forward_rejects_leaf_from_a_different_tape() {
        let tape1 = simple_tape(None);
        let tape2 = simple_tape(None);
        let w = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape1, &[&w]).unwrap();
        let leaves1 = store.register_resident_params(&tape1).unwrap();
        // `tape2` 上の `Var`（`input` 自体は `tape` 引数と一致させる）に
        // 対し、`tape1` で発行された `ResidentLeaf` を渡す。
        let x2 = tape2.var(&tensor(vec![1.0], &[1, 1]));

        let err = store
            .linear_forward(&tape2, &x2, &leaves1[0], None)
            .unwrap_err();
        assert!(matches!(err, BackendError::TapeMismatch));

        // `tape1` 側の pending は消費されていないため、正しい `tape1` で
        // forward すれば引き続き成功する。
        let pred = store
            .linear_forward(
                &tape1,
                &tape1.var(&tensor(vec![1.0], &[1, 1])),
                &leaves1[0],
                None,
            )
            .unwrap();
        let _ = pred;
    }

    /// R3（#1022・#1023 統合設計）の CI 上の機械検査: `step()` を複数回
    /// 呼んでも `sgd_step_device`（デフォルト委譲経由）の起動回数・
    /// `upload` の呼び出し回数がいずれもパラメータ件数に依らず
    /// 「1 回／step」になることを検証する（3 パラメータ〈2 層の
    /// `linear_forward` を連鎖させ全パラメータへ勾配を流す〉で 2 step
    /// 実行し、起動回数・grad upload 回数がともに 2 であることを確認）。
    /// forward／backward は `MockDeviceOps::gemm_resident_rhs`／
    /// `gemm_resident_lhs`（`downcast_handle` 直読み）経由のため
    /// `upload` を発生させない（`resident_forward_backward_has_zero_
    /// param_download` と同じ理由）。
    #[test]
    fn step_launches_sgd_kernel_exactly_once_regardless_of_param_count() {
        let (tape, call_count, upload_count) = tape_with_counters(None);
        let w1 = tensor(vec![1.0], &[1, 1]);
        let b1 = tensor(vec![1.0], &[1]);
        let w2 = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w1, &b1, &w2]).unwrap();

        for _ in 0..2 {
            let leaves = store.register_resident_params(&tape).unwrap();
            let x = tape.var(&tensor(vec![1.0], &[1, 1]));
            let y1 = store
                .linear_forward(&tape, &x, &leaves[0], Some(&leaves[1]))
                .unwrap();
            let y2 = store.linear_forward(&tape, &y1, &leaves[2], None).unwrap();
            let target = tape.var(&tensor(vec![0.0], &[1, 1]));
            let loss = y2.mse_loss(&target).unwrap();
            let grads = store.backward(&tape, &loss).unwrap();
            store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();
        }

        // upload 回数の内訳: `new()` で連結パラメータ upload 1 回 +
        // `step()` 毎の grad upload 1 回 x 2 = 3 のはず。
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
    /// シナリオに `upload_count` の不変を追加した版。#1022 統合により
    /// `linear_forward` 経由で `Op::LinearResident` を forward してから
    /// 検証する形へ調整）。
    #[test]
    fn missing_gradient_is_rejected_before_any_upload() {
        let (tape, _call_count, upload_count) = tape_with_counters(None);
        let w = tensor(vec![1.0], &[1, 1]);
        let unused = tensor(vec![2.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &unused]).unwrap();
        let leaves = store.register_resident_params(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));
        // loss は leaves[0] のみに依存し、leaves[1]（`unused`）へは勾配が
        // 流れない。
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let loss = pred.mse_loss(&pred).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
        let upload_before = upload_count.load(Ordering::SeqCst);

        let err = store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap_err();
        assert!(matches!(err, BackendError::MissingGradient(_)));
        assert_eq!(
            upload_count.load(Ordering::SeqCst),
            upload_before,
            "勾配欠落は upload 前に拒否されるはず"
        );
    }

    /// イシュー #1044: `linear_forward_with_activation(..., Activation::
    /// Relu)`（デバイス常駐 epilogue 融合）の勾配が、`linear_forward`
    /// （bias のみ融合）+ `Var::relu()`（別ノード）という非融合合成の
    /// 勾配とビット一致することを検証する（`Op::LinearResident` の VJP
    /// が `act == Relu` の場合に適用する `out_value > 0` マスクが、
    /// 非融合 `Op::Relu` の VJP と同じ勾配を返すことの実測確認。
    /// `MockDeviceOps::gemm_resident_rhs_act` は `BackendOps` の既定
    /// メソッド〈`gemm_resident_rhs` → `act == Relu` なら `self.relu`〉
    /// をそのまま使うため、フォワード値自体も両経路で一致する）。
    #[test]
    fn linear_forward_with_activation_relu_grad_matches_manual_relu_composition() {
        let fused_tape = simple_tape(None);
        let w = tensor(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]);
        let b = tensor(vec![0.1, -0.1], &[2]);
        let x_data = tensor(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);

        let mut fused_store = DeviceParamStore::new(&fused_tape, &[&w, &b]).unwrap();
        let fused_leaves = fused_store.register_resident_params(&fused_tape).unwrap();
        let fused_x = fused_tape.var(&x_data);
        let fused_out = fused_store
            .linear_forward_with_activation(
                &fused_tape,
                &fused_x,
                &fused_leaves[0],
                Some(&fused_leaves[1]),
                Activation::Relu,
            )
            .unwrap();
        let fused_loss = fused_out.sum(None).unwrap();
        let fused_grads = fused_store.backward(&fused_tape, &fused_loss).unwrap();

        let manual_tape = simple_tape(None);
        let mut manual_store = DeviceParamStore::new(&manual_tape, &[&w, &b]).unwrap();
        let manual_leaves = manual_store.register_resident_params(&manual_tape).unwrap();
        let manual_x = manual_tape.var(&x_data);
        let manual_pre = manual_store
            .linear_forward(
                &manual_tape,
                &manual_x,
                &manual_leaves[0],
                Some(&manual_leaves[1]),
            )
            .unwrap();
        let manual_out = manual_pre.relu();
        let manual_loss = manual_out.sum(None).unwrap();
        let manual_grads = manual_store.backward(&manual_tape, &manual_loss).unwrap();

        assert_eq!(
            crate::eval::dense_vec(&fused_out.to_tensor()),
            crate::eval::dense_vec(&manual_out.to_tensor()),
            "融合・非融合で forward 出力がビット一致しない"
        );

        // `ResidentLeaf` は値アクセサを持たない不透明型（doc 参照）の
        // ため、勾配取得には `node_id` から `Var::from_raw` で `Var` を
        // 組み立てて `Gradients::get` に渡す（`node_id` は private
        // フィールドだが本 `mod tests` は同一モジュールの子孫として
        // アクセス可能）。
        let fused_w_grad = fused_grads
            .get(&Var::from_raw(&fused_tape, fused_leaves[0].node_id))
            .unwrap()
            .unwrap();
        let manual_w_grad = manual_grads
            .get(&Var::from_raw(&manual_tape, manual_leaves[0].node_id))
            .unwrap()
            .unwrap();
        assert_eq!(
            crate::eval::dense_vec(fused_w_grad),
            crate::eval::dense_vec(manual_w_grad),
            "weight 勾配が融合・非融合でビット一致しない"
        );

        let fused_b_grad = fused_grads
            .get(&Var::from_raw(&fused_tape, fused_leaves[1].node_id))
            .unwrap()
            .unwrap();
        let manual_b_grad = manual_grads
            .get(&Var::from_raw(&manual_tape, manual_leaves[1].node_id))
            .unwrap()
            .unwrap();
        assert_eq!(
            crate::eval::dense_vec(fused_b_grad),
            crate::eval::dense_vec(manual_b_grad),
            "bias 勾配が融合・非融合でビット一致しない"
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
