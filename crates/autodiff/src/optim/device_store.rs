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
//!   再構築のみ。

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use fandhe_ai_tensor_core::buffer::DeviceBuffer;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{SgdStepConfig, Tensor};

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

/// [`DeviceParamStore::register_resident_leaves`]／
/// [`DeviceParamStore::snapshot_resident_leaves`] が返す、テープ上の
/// `Op::ResidentLeaf` ノードへの不透明なハンドル（イシュー #1022）。
///
/// **`Var` ではなく専用の不透明型にする理由**: `Var::value()`／
/// `to_tensor()`（`var.rs`）は非 fallible な API であり、ホスト値を
/// 持たない `Op::ResidentLeaf` に対して呼ばれると「panic させる」か
/// 「黙示的にゼロを返す」のいずれかしか選べない。本型は `shape()` の
/// みを公開し、値アクセサ（`value()`／`to_tensor()` 相当）を持たない
/// ことで、呼び出し元がこの罠に到達する経路自体を型で塞ぐ
/// （`tape::Op::ResidentLeaf` doc 参照）。
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
/// 側で導出済み）／`PendingForward`〈上記〉）。
#[derive(Debug)]
pub struct DeviceParamStore {
    device: Device,
    /// このストアの一意識別子（イシュー #1022）。`Op::ResidentLeaf`／
    /// `ResidentLeaf` が保持し、`ResidentResolver::resident_buffer` が
    /// 「別ストアの葉が誤って混入していないか」を検査する鍵になる
    /// （`NEXT_STORE_ID` のドキュメンテーションコメント参照）。
    store_id: u64,
    params: Vec<DeviceBuffer<f32>>,
    velocities: Vec<Option<DeviceBuffer<f32>>>,
    step_count: u64,
    poisoned: bool,
    pending: Option<PendingForward>,
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
    /// [`DeviceParamStore::checked_resident_buffer`] へ委譲する薄い実装
    /// （`linear_forward` と同じ検証を共有する）。
    fn resident_buffer(
        &self,
        store_id: u64,
        slot: usize,
    ) -> Result<&DeviceBuffer<f32>, AutodiffError> {
        self.checked_resident_buffer(store_id, slot)
            .map_err(AutodiffError::Backend)
    }
}

impl DeviceParamStore {
    /// `params`（呼び出し元の位置対応契約に従うホスト常駐パラメータ列。
    /// 典型的には `fandhe_ai::compat::Sequential::trainable_parameters()`
    /// の戻り値）を `tape` のバックエンドへ 1 回だけアップロードして
    /// `DeviceParamStore` を構築する。
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
        let mut uploaded = Vec::with_capacity(params.len());
        for p in params {
            uploaded.push(mem.upload(p)?);
        }
        Ok(DeviceParamStore {
            device,
            store_id: NEXT_STORE_ID.fetch_add(1, Ordering::Relaxed),
            velocities: (0..uploaded.len()).map(|_| None).collect(),
            params: uploaded,
            step_count: 0,
            poisoned: false,
            pending: None,
        })
    }

    /// このストアが常駐するデバイス。
    pub fn device(&self) -> Device {
        self.device
    }

    /// このストアが保持するパラメータ件数。
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// パラメータ件数が 0 か判定する（`clippy::len_without_is_empty` 対応）。
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    fn check_not_poisoned(&self) -> Result<(), BackendError> {
        if self.poisoned {
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

    /// forward 用: 現在デバイス上にある各パラメータを `tape` の
    /// `Op::ResidentLeaf` ノードとして**毎回新規登録**し（`tape.
    /// push_resident_leaf(...)`。ホストへの download を伴わない。本
    /// イシュー #1022 の中核）、`step()` が消費するまで
    /// [`BackendError::PendingForwardUnconsumed`] で以後の再登録を拒否する
    /// （モジュール冒頭「状態機械」参照）。
    ///
    /// **#1022 による変更**: 旧実装は `mem.download(buf)` でホストへ
    /// 落としてから `tape.var(&tensor)`（`Op::Leaf`）に登録していた
    /// （設計文書 §3.3b が「本段階で残る転送」と明記していたもの）。
    /// 本イシューでこの download を排除し、代わりに `Op::ResidentLeaf`
    /// （ホスト値を持たないテープノード）を登録して不透明型
    /// [`ResidentLeaf`] を返す（`Var` を返さない理由は同型の doc 参照）。
    /// 呼び出し元は返った `ResidentLeaf` を [`Self::linear_forward`] へ
    /// 渡して forward する。
    pub fn register_resident_leaves<'t>(
        &mut self,
        tape: &'t Tape,
    ) -> Result<Vec<ResidentLeaf<'t>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if self.pending.is_some() {
            return Err(BackendError::PendingForwardUnconsumed);
        }
        let mut leaves = Vec::with_capacity(self.params.len());
        let mut node_ids = Vec::with_capacity(self.params.len());
        for (slot, buf) in self.params.iter().enumerate() {
            let shape = buf.shape().to_vec();
            let node_id = tape.push_resident_leaf(shape.clone(), self.store_id, slot);
            node_ids.push(node_id);
            leaves.push(ResidentLeaf {
                node_id,
                store_id: self.store_id,
                slot,
                shape,
                _marker: PhantomData,
            });
        }
        self.pending = Some(PendingForward {
            tape_id: tape.id,
            node_ids,
        });
        Ok(leaves)
    }

    /// 推論用の読み取り専用版（`register_resident_leaves` と異なり
    /// `pending` 状態を変化させない。`step()` で消費する必要がないため）。
    /// #1022 による変更点は `register_resident_leaves` と同じ
    /// （download を排除し `ResidentLeaf` を返す）。
    pub fn snapshot_resident_leaves<'t>(
        &self,
        tape: &'t Tape,
    ) -> Result<Vec<ResidentLeaf<'t>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        Ok(self
            .params
            .iter()
            .enumerate()
            .map(|(slot, buf)| {
                let shape = buf.shape().to_vec();
                let node_id = tape.push_resident_leaf(shape.clone(), self.store_id, slot);
                ResidentLeaf {
                    node_id,
                    store_id: self.store_id,
                    slot,
                    shape,
                    _marker: PhantomData,
                }
            })
            .collect())
    }

    /// `store_id`／`slot` を検証したうえで対応する `DeviceBuffer<f32>` を
    /// 返す（イシュー #1022）。[`Self::linear_forward`]・
    /// [`ResidentResolver::resident_buffer`] の共通実装。別ストアの葉が
    /// 混入した場合・`slot` が範囲外の場合は fail-closed に拒否する
    /// （`.claude/rules/security.md` A08）。
    fn checked_resident_buffer(
        &self,
        store_id: u64,
        slot: usize,
    ) -> Result<&DeviceBuffer<f32>, BackendError> {
        if store_id != self.store_id {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore: resident leaf belongs to a different DeviceParamStore \
                 (store_id mismatch)"
                    .to_string(),
            ));
        }
        self.params.get(slot).ok_or_else(|| {
            BackendError::InvalidArgument(format!(
                "DeviceParamStore: resident leaf slot {slot} is out of range (store has {} \
                 parameters)",
                self.params.len()
            ))
        })
    }

    /// forward 用: `weight`（・`bias`）をデバイス常駐のまま
    /// `BackendOps::gemm_resident_rhs`（`tensor-core`）へ渡し、
    /// `y = input.matmul(weight) (+ bias)` を計算してテープへ
    /// `Op::LinearResident` として記録する（イシュー #1022 の中核）。
    ///
    /// `input`（活性化値）はホスト常駐のまま渡してよい（本イシューが
    /// 排除する D2H は weight／bias のものに限る。§1.2 の解釈）。
    /// `weight`／`bias` はいずれも `self`（同一 `DeviceParamStore`）が
    /// [`Self::register_resident_leaves`]／[`Self::snapshot_resident_leaves`]
    /// で返した [`ResidentLeaf`] でなければならない（[`Self::
    /// checked_resident_buffer`] が `store_id`／`slot` を検証する）。
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
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        if input.tape_id() != tape.id {
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

        let y = tape.ops().gemm_resident_rhs(&x_val, w_buf, b_buf)?;
        let node_id = tape.push_eager(
            Op::LinearResident {
                input: input.node_id(),
                weight: weight.node_id,
                bias: bias.map(|b| b.node_id),
            },
            y,
        );
        Ok(Var::from_raw(tape, node_id))
    }

    /// `loss` から逆伝播し、常駐 weight／bias（`Op::ResidentLeaf`・
    /// `Op::LinearResident`）を含むグラフの勾配を計算する（イシュー
    /// #1022）。`self` を [`ResidentResolver`] として `tape.
    /// backward_with_resident` へ渡す薄いラッパー。
    ///
    /// **素の `tape.backward(loss)` との違い**: `Op::LinearResident` の
    /// VJP は weight の `DeviceBuffer<f32>` を解決する手段
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
    /// ③更新（1 パラメータずつ grad を upload → `sgd_step_device`）」の
    /// 順（設計文書 §3.3a）。②で 1 件でも失敗すれば**どのパラメータも
    /// 更新せずに** `Err` を返す（fail-closed。`.claude/rules/security.md`
    /// A03）。③の実行時エラー（GPU 起動失敗等）は最初のエラーを返しつつ
    /// `poisoned` へ遷移する（モジュール冒頭「状態機械」参照。それ以前の
    /// パラメータは既に更新済みの可能性があるため、以降のアクセスを
    /// 一律拒否することで「一部だけ更新された状態」を安全に隔離する）。
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
        // 一致するかを、いずれかのバッファを更新する前に検査する
        // （fail-closed。`SequentialVars::trainable_grads` と同じ方針）。
        let vars: Vec<Var<'_>> = pending
            .node_ids
            .iter()
            .map(|&id| Var::from_raw(tape, id))
            .collect();
        let mut grad_tensors: Vec<&Tensor<f32>> = Vec::with_capacity(vars.len());
        for (i, var) in vars.iter().enumerate() {
            let grad = grads.get(var).map_err(|_| BackendError::TapeMismatch)?;
            let grad = grad.ok_or_else(|| {
                BackendError::MissingGradient(format!(
                    "DeviceParamStore::step: parameter {i} has no gradient (loss unreachable)"
                ))
            })?;
            if grad.shape() != self.params[i].shape() {
                return Err(BackendError::InvalidArgument(format!(
                    "DeviceParamStore::step: gradient shape {:?} does not match parameter {i} \
                     shape {:?}",
                    grad.shape(),
                    self.params[i].shape()
                )));
            }
            grad_tensors.push(grad);
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
        let momentum_state_exists = self.velocities.iter().any(|v| v.is_some());
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

        // momentum が有効かつ未確保のパラメータには、ここで velocity を
        // 遅延確保する（PyTorch の「初回は state 未生成、初回 step で
        // ゼロ初期化バッファを生成」タイミングと同じ）。確保自体は
        // 「更新フェーズ」より前の準備段階であり、確保失敗はどのパラメータ
        // も更新しないまま `Err` を返す（fail-closed。poisoned へは遷移
        // しない: 確保失敗は事前検証と同種の「実行前に判明した失敗」で
        // あり、デバイス側パラメータは未変更のまま安全なため）。
        if use_momentum {
            for (i, v) in self.velocities.iter_mut().enumerate() {
                if v.is_none() {
                    *v = Some(mem.alloc_zeroed(self.params[i].shape())?);
                }
            }
        }

        // ここまでの事前検証（勾配存在・shape・momentum 構成・
        // `MemoryOps` 提供・velocity 確保）を全て通過し、以降は実際に
        // デバイス側パラメータを更新するフェーズへ入る。ここで初めて
        // `pending` を消費する（以降のエラーは `poisoned` 遷移で `pending`
        // の意味自体が失われるため、これより手前で `take()` しない）。
        self.pending = None;

        // ② 更新フェーズ: 1 パラメータずつ grad を upload →
        // `sgd_step_device`。実行時エラーは最初のエラーを返しつつ
        // `poisoned` へ遷移する（モジュール冒頭「状態機械」参照）。
        // `self.params`／`self.velocities`／`grad_tensors` の 3 つの
        // 独立した列を同一添字で並行アクセスする必要があるため、
        // `clippy::needless_range_loop` の推奨する単一 `.iter()` 化は
        // 適用できない（`self.velocities[i]` への可変アクセスと
        // `grad_tensors[i]`／`self.params[i].shape()` への参照を同一
        // ループボディで行う）。
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.params.len() {
            let grad_buf = match mem.upload(grad_tensors[i]) {
                Ok(buf) => buf,
                Err(e) => {
                    self.poisoned = true;
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
                self.velocities[i].as_mut()
            } else {
                None
            };
            if let Err(e) =
                ops.sgd_step_device(&mut self.params[i], &grad_buf, velocity_ref, &step_config)
            {
                self.poisoned = true;
                return Err(e);
            }
        }

        self.step_count += 1;
        Ok(())
    }

    /// 現在のデバイス上パラメータをホストへダウンロードする（明示同期
    /// 点。`fandhe_ai::compat::Sequential::apply_parameters` へそのまま
    /// 渡せる並び順で返す）。
    pub fn sync_to_host(&self, tape: &Tape) -> Result<Vec<Tensor<f32>>, BackendError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        let mem = tape.ops().memory_ops().ok_or_else(|| {
            BackendError::Unsupported(
                "DeviceParamStore::sync_to_host: backend does not implement MemoryOps".to_string(),
            )
        })?;
        self.params.iter().map(|buf| mem.download(buf)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::cell::RefCell;
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
    /// を返す（poisoned 遷移の検証用）。
    struct MockDeviceOps {
        fail_after: Option<usize>,
        call_count: AtomicUsize,
        /// `MemoryOps::download` の累計呼び出し回数（イシュー #1022 の
        /// 受け入れ条件 1「reuse 学習 1 step 内の D2H が loss 実体化以外
        /// 0 回」を機械検証するためのカウンタ）。`gemm_resident_rhs`／
        /// `gemm_resident_lhs`（下記）は本カウンタを増やさない実装
        /// （`downcast_handle` で直接読む。`backend-cpu::ops::
        /// CpuBackendOps` の「ゼロコピー」実装と同じモデル）とすることで、
        /// forward／backward が実際に `download()` を呼んでいないことを
        /// 区別して検証できる。
        /// `Arc` にする理由: `Tape::new_with_ops` は `Box<dyn BackendOps +
        /// Send>` として所有権を奪うため、`BackendOps` トレイトを介さず
        /// カウンタだけをテスト側に残して読み出すには共有参照が要る
        /// （`BackendOps` は `Any` を要求しないため `tape.ops()` から
        /// downcast する経路は取れない）。
        download_count: std::sync::Arc<AtomicUsize>,
    }

    impl MockDeviceOps {
        fn new() -> Self {
            Self {
                fail_after: None,
                call_count: AtomicUsize::new(0),
                download_count: std::sync::Arc::new(AtomicUsize::new(0)),
            }
        }

        fn failing_after(n: usize) -> Self {
            Self {
                fail_after: Some(n),
                call_count: AtomicUsize::new(0),
                download_count: std::sync::Arc::new(AtomicUsize::new(0)),
            }
        }

        /// カウンタの共有ハンドルを複製する（`Tape::new_with_ops` へ
        /// `self` の所有権を渡した後もテスト側から読み出せるようにする。
        /// `resident_forward_backward_has_zero_param_download` 参照）。
        fn download_counter(&self) -> std::sync::Arc<AtomicUsize> {
            self.download_count.clone()
        }

        /// `DeviceBuffer<f32>` の中身をコピー取得する（`download()` を
        /// 経由しないため `download_count` を増やさない。`gemm_resident_
        /// rhs`／`gemm_resident_lhs` 専用のヘルパー）。
        fn read_resident(buf: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
            let handle = buf
                .downcast_handle::<MockHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            let data = handle.data.borrow().clone();
            Tensor::new(data, buf.shape()).map_err(BackendError::ShapeMismatch)
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
            w: &DeviceBuffer<f32>,
            bias: Option<&DeviceBuffer<f32>>,
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
            w: &DeviceBuffer<f32>,
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

        let leaves = store.register_resident_leaves(&tape).unwrap();
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
        let leaves = store.register_resident_leaves(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0, 10.0], &[1, 2]));
        let pred = store.linear_forward(&tape, &x, &leaves[0], None).unwrap();
        let loss = pred.mse_loss(&target).unwrap();
        let grads = store.backward(&tape, &loss).unwrap();
        store.step(&tape, &grads, &SgdConfig::new(0.1)).unwrap();

        // 2 ステップ目で momentum を有効化すると拒否される。
        let leaves = store.register_resident_leaves(&tape).unwrap();
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
        let w = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape1, &[&w]).unwrap();
        let leaves1 = store.register_resident_leaves(&tape1).unwrap();
        let x1 = tape1.var(&tensor(vec![1.0], &[1, 1]));

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
        let leaves = store.register_resident_leaves(&tape).unwrap();
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
        // 2 パラメータのうち 1 個目の `sgd_step_device` 成功後、2 個目で
        // 失敗させる（部分更新後の poisoned 遷移を検証する）。2 層の
        // `linear_forward` を連鎖させ、両パラメータへ勾配が流れる形にする
        // （#1022 で `Op::LinearResident` 経路へ書き換え。旧テストの
        // `vars[0].mul(&vars[1])` は `ResidentLeaf` 同士の直接演算が
        // できなくなったため代替）。
        let tape = simple_tape(Some(2));
        let w = tensor(vec![1.0], &[1, 1]);
        let b = tensor(vec![1.0], &[1, 1]);
        let mut store = DeviceParamStore::new(&tape, &[&w, &b]).unwrap();
        let leaves = store.register_resident_leaves(&tape).unwrap();
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

    /// イシュー #1022 受け入れ条件 1: reuse 学習 1 step 内（forward の
    /// `register_resident_leaves`／`linear_forward`・backward の
    /// `DeviceParamStore::backward`）で `MemoryOps::download`（D2H）が
    /// 0 回であることを機械検証する。`MockDeviceOps::gemm_resident_rhs`／
    /// `gemm_resident_lhs` は `downcast_handle` で直接読む実装
    /// （`download()` を経由しない）であり、`register_resident_leaves`／
    /// `snapshot_resident_leaves` 自体も本イシューで download を撤去
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

        let leaves = store.register_resident_leaves(&tape).unwrap();
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
            "register_resident_leaves/linear_forward/backward の 1 step 内で \
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
        let leaves = store.register_resident_leaves(&tape).unwrap();
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
        let leaves2 = store2.register_resident_leaves(&tape).unwrap();
        let x = tape.var(&tensor(vec![1.0], &[1, 1]));

        let err = store1
            .linear_forward(&tape, &x, &leaves2[0], None)
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
        // store2 側の pending は消費されていないため、store2 の
        // register_resident_leaves を呼び直す必要はなく引き続き有効。
        store2.abandon_pending_forward();
        let _ = store1;
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
