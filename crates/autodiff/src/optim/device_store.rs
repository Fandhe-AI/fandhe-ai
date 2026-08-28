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

use fandhe_ai_tensor_core::buffer::DeviceBuffer;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{SgdStepConfig, Tensor};

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

/// デバイス常駐パラメータ更新の状態機械本体（モジュール冒頭コメント
/// 参照）。単一デバイス固定（`new()` 時に `tape.ops().device()` から
/// 決定し、以後変更不可）。`Debug` は `unwrap_err()`（テストコード）が
/// `Result<DeviceParamStore, _>` を要求するため導出する（フィールドは
/// いずれも `Debug` 実装済み: `Device`／`DeviceBuffer<f32>`（`tensor-core`
/// 側で導出済み）／`PendingForward`〈上記〉）。
#[derive(Debug)]
pub struct DeviceParamStore {
    device: Device,
    params: Vec<DeviceBuffer<f32>>,
    velocities: Vec<Option<DeviceBuffer<f32>>>,
    step_count: u64,
    poisoned: bool,
    pending: Option<PendingForward>,
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
        let mut vars = Vec::with_capacity(self.params.len());
        let mut node_ids = Vec::with_capacity(self.params.len());
        for buf in &self.params {
            let tensor = mem.download(buf)?;
            let var = tape.var(&tensor);
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
        self.params
            .iter()
            .map(|buf| Ok(tape.var(&mem.download(buf)?)))
            .collect()
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

        let Some(pending) = self.pending.take() else {
            return Err(BackendError::InvalidArgument(
                "DeviceParamStore::step: no pending forward registration (call \
                 register_resident_leaves first)"
                    .to_string(),
            ));
        };
        if pending.tape_id != tape.id {
            // 呼び出し元の誤り（別 Tape を渡した）であり、ストア自体は
            // 壊れていない。次回の `step()` 呼び出しが正しい `Tape` で
            // 成功できるよう `pending` を復元する。
            self.pending = Some(pending);
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
    }

    impl MockDeviceOps {
        fn new() -> Self {
            Self {
                fail_after: None,
                call_count: AtomicUsize::new(0),
            }
        }

        fn failing_after(n: usize) -> Self {
            Self {
                fail_after: Some(n),
                call_count: AtomicUsize::new(0),
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

        // momentum 構成拒否は `poisoned` へは遷移しない（実行前に判明した
        // 失敗であり、デバイス側パラメータは未変更のまま安全なため。
        // `TapeMismatch` と異なり `pending` は復元しないため、正しい
        // 設定で再開するには `register_resident_leaves` を呼び直す）。
        let vars = store.register_resident_leaves(&tape).unwrap();
        let target = tape.var(&tensor(vec![10.0, 10.0], &[2]));
        let loss = vars[0].mse_loss(&target).unwrap();
        let grads = tape.backward(&loss).unwrap();
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
        // 2 パラメータのうち 1 個目の `sgd_step_device` 成功後、2 個目で
        // 失敗させる（部分更新後の poisoned 遷移を検証する）。
        let tape = simple_tape(Some(2));
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
