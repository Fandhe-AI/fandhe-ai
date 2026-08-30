//! Metal バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `fandhe_ai_tensor_core::buffer::MemoryOps` の Metal 実装。既存の
//! [`crate::buffer::MetalBuffer`]（TASK-1.8a・#38。`new_with_data`／
//! `new_zeroed`／`read_to_vec`）をそのまま再利用する。本モジュール
//! 自体は新規 `unsafe` を追加しない（`.claude/rules/security.md` の
//! 「unsafe は必要最小限」方針。FFI 境界の safety 根拠は `buffer.rs`
//! 側に集約済み）。TASK-#201（REQ-14 14-3）で追加した
//! `MetalBuffer::zero_fill` のみ、`read_to_vec` と対になる書き込み版
//! FFI アクセスとして `buffer.rs` 側に 1 箇所追加している
//! （`buffer.rs` モジュールコメント「Safety 境界」参照）。
//!
//! `StorageModeShared`（Apple Silicon の UMA。`buffer.rs` モジュール
//! コメント参照）のため CUDA のような明示的な非同期**転送**は不要だが、
//! イシュー #1017（コマンドバッファ共有バッチ）導入後は
//! **`context.rs::MetalContext::synchronize` を経由する**（`download_inner`／
//! `PoolZeroFill::zero_fill` 双方）。`sgd.rs::MetalSgd::run` 等が
//! `ctx.encode`（バッファ結線のみ・待たない）でバッチへ積んだ dispatch
//! は GPU 完了前に `contents()` のバイト列を確定させない: `synchronize`
//! を挟まないまま `read_to_vec`／`zero_fill` を呼ぶと、GPU がまだ書き
//! 込み中のバッファを読む・GPU 実行中のバッファへ上書きする未定義動作の
//! 危険がある（UMA は物理メモリ共有を保証するのみで、CPU/GPU 間の
//! **実行順序**の保証はコマンドバッファの完了同期に依存する）。

use std::any::Any;
use std::mem::size_of;
use std::sync::Arc;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
use fandhe_ai_tensor_core::pool::PoolZeroFill;

/// Metal バッファの具体ハンドル。
///
/// `numel == 0`（空テンソルの契約。`fandhe_ai_tensor_core::buffer` モジュール
/// コメント参照）では `buffer` を `None` とする。`MetalBuffer::new_with_data`／
/// `new_zeroed` はいずれも長さ 0 を `MetalError::ZeroLengthAllocation`
/// として FFI 呼び出し前に拒否するため（`buffer.rs::checked_byte_len`）、
/// 空テンソルはこのハンドル自体を経由して Metal 側の拒否を回避する。
///
/// `_alloc`（[`TrackedAllocation`]。TASK-14.1b・#175）は `buffer` より後に
/// 宣言しており、フィールドは宣言順に drop される Rust の規則により
/// `buffer`（`MetalBuffer` 内部の `Retained<MtlBuffer>` の解放）の後に
/// drop される。`TrackedAllocation::drop` は `buffer` の中身を参照せず
/// 確保時に記録したバイト数を `AllocationTracker` へ返すだけ
/// （`backend-cpu::CpuBufferHandle`／`backend-cuda::CudaBufferHandle` の
/// `_alloc` と同型の契約）であるため、drop 順は計測上問題にならない。
/// `pub(crate)`（イシュー #935・`docs/device-resident-update-design.md`
/// §3.2 で `ops.rs::MetalBackendOps::sgd_step_device`／`sgd.rs::MetalSgd::run`
/// が `DeviceBuffer::downcast_handle_mut` 経由で in-place 書き換えを行う
/// ために `crate::memory::MetalBufferHandle` として参照する必要があり、
/// 可視性を crate 内に広げた。`backend-cpu::CpuBufferHandle`／
/// `backend-cuda::CudaBufferHandle` と同じ判断）。
#[derive(Debug)]
pub(crate) struct MetalBufferHandle {
    pub(crate) buffer: Option<MetalBuffer>,
    _alloc: TrackedAllocation,
}

impl BufferHandle for MetalBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// `MemoryOps` の Metal 実装。[`MetalContext`] を保持し、確保・転送の
/// たびに `MetalBuffer::new_with_data`／`new_zeroed` へ委譲する。
///
/// `tracker`（TASK-14.1b・#175）は `backend-cpu::CpuMemory`／
/// `backend-cuda::CudaMemory` と同型の計測フックだが、`MetalContext` が
/// `Clone` を導出していないため（`context.rs` 参照。`Retained<MtlDevice>`／
/// `Retained<MtlQueue>` の複製可否をこのイシューでは検証しない安全側
/// 判断）、`MetalMemory` 自体にも `Clone` は付与しない
/// （out-of-scope。受け入れ条件「3 バックエンドで同一 API からピーク値が
/// 取得できる」は `MemoryStats` 実装のみを要求し `Clone` を要求しない）。
pub struct MetalMemory {
    context: Arc<MetalContext>,
    tracker: Arc<AllocationTracker>,
}

impl MetalMemory {
    /// 初期化済みの [`MetalContext`] から `MetalMemory` を構築する。
    /// 新規の計測系列を持つトラッカーを生成する
    /// （`backend-cpu::CpuMemory::new` と同型）。
    ///
    /// 内部で [`Self::from_shared`] へ委譲する（`Arc::new` で包むだけの
    /// 薄いラッパー）。呼び出し元が単発で所有権を持つ `MetalContext` を
    /// そのまま渡せるよう、シグネチャは変更しない（非破壊）。
    ///
    /// **注意（イシュー #1017）**: `context` は呼び出し元が新規構築した
    /// 専用インスタンスであり、`ops::MetalBackendOps`（`sgd_step_device`
    /// 等）が内部で使う `context_cache::cached_context()`（プロセス全体
    /// シングルトン）とは**別のバッチ状態**を持つ。本コンストラクタ経由の
    /// `MetalMemory`（例: `bench-harness::peak_memory`・
    /// `tests/memory_roundtrip.rs`）で `download`／`zero_fill` を呼んでも、
    /// `MetalBackendOps` 経由でディスパッチされた別コンテキストのバッチ
    /// （SGD 等）は同期しない。この経路を SGD デバイス常駐更新
    /// （`fandhe_ai_autodiff::optim::device_store::DeviceParamStore`）と
    /// 混在させない（同ストアは `tape.ops().memory_ops()` →
    /// `MetalBackendOps::memory_ops()` → [`Self::from_shared`]
    /// （プロセス共有コンテキスト）経由に固定されており、この懸念は
    /// 生じない）。
    pub fn new(context: MetalContext) -> Self {
        Self::from_shared(Arc::new(context))
    }

    /// 既に `Arc` 共有されている [`MetalContext`] から `MetalMemory` を
    /// 構築する（イシュー #935 レビュー対応）。
    ///
    /// `ops.rs::static_metal_memory` が `context_cache::cached_context`
    /// （`Arc<MetalContext>` を返す既存のプロセス全体シングルトン）と
    /// 同一の `MetalContext` を共有するために追加した。これにより
    /// バッファ確保（本モジュール）とカーネルディスパッチ
    /// （`sgd.rs::MetalSgd::run` 等）が同一の `MTLDevice`／
    /// `MTLCommandQueue` を経由することを構造的に保証する
    /// （`context_cache` 経由取得の一本化。`docs/device-resident-
    /// update-design.md` §3.3d「独自初期化禁止」）。
    ///
    /// フィールド型（`context: Arc<MetalContext>`）は非公開のため、この
    /// コンストラクタ追加は `MetalMemory` の公開 API（`new` の
    /// シグネチャ）を変更しない SemVer 非破壊な拡張である。
    pub fn from_shared(context: Arc<MetalContext>) -> Self {
        Self {
            context,
            tracker: Arc::new(AllocationTracker::new()),
        }
    }
}

/// [`MemoryStats`] の Metal 実装（TASK-14.1b・#175）。`backend-cpu::
/// CpuMemory`／`backend-cuda::CudaMemory` と同一シグネチャで `tracker` へ
/// 委譲する。REQ-14 の受け入れ条件（CPU/CUDA/Metal で同一 API からピーク
/// 値が取得できる）を満たす。
impl MemoryStats for MetalMemory {
    fn allocated_bytes(&self) -> u64 {
        self.tracker.allocated_bytes()
    }

    fn peak_allocated_bytes(&self) -> u64 {
        self.tracker.peak_allocated_bytes()
    }

    fn reset_peak(&self) {
        self.tracker.reset_peak();
    }
}

/// `MetalError` を `BackendError` へ変換する。
///
/// `ZeroLengthAllocation`/`AllocationSizeOverflow` は形状検証系の失敗
/// （呼び出し前の shape 由来）のため `DeviceAllocationFailed` に、
/// `CommandBufferExecutionFailed` 等の実行時失敗は 4.4 の
/// `KernelLaunchFailed` に寄せる（本モジュールは GEMM ディスパッチを
/// 行わないため実際には到達しないが、`MetalError` は `#[non_exhaustive]`
/// であり網羅的 match が書けないため wildcard の受け皿として用意する）。
///
/// `crate::ops`（GEMM ディスパッチ）からも `MetalContext::new` の
/// エラー変換に再利用される（`pub(crate)`）。`DeviceUnavailable` を
/// `MetalDeviceProvider::select`（`device.rs`）と同一の
/// `BackendError::DeviceUnavailable` に統一するため（Bugbot 指摘対応。
/// PR #262 レビュースレッド）。
pub(crate) fn map_metal_error(err: MetalError) -> BackendError {
    match err {
        MetalError::ZeroLengthAllocation => {
            BackendError::DeviceAllocationFailed("zero-length allocation requested".to_string())
        }
        MetalError::AllocationSizeOverflow { len } => BackendError::DeviceAllocationFailed(
            format!("buffer byte length overflows usize for len={len} elements"),
        ),
        MetalError::BufferAllocation { bytes } => {
            BackendError::DeviceAllocationFailed(format!("allocation failed for {bytes} bytes"))
        }
        MetalError::DeviceUnavailable | MetalError::CommandQueueCreation => {
            BackendError::DeviceUnavailable(err.to_string())
        }
        // CUDA 実装（`backend-cuda/src/memory.rs::map_cuda_error`）の
        // `CudaError::InvalidShape { detail } => BackendError::
        // DeviceAllocationFailed(detail)` と同じ変換先に揃える
        // （レビュー指摘対応。detail を保持したまま伝播する）。
        MetalError::ShapeMismatch { detail } => BackendError::DeviceAllocationFailed(detail),
        other => BackendError::KernelLaunchFailed(other.to_string()),
    }
}

/// shape の要素数積を検査付きで計算する（`checked_byte_len`
/// （`buffer.rs`）が要素数からバイト長を検査するのに対し、本関数は
/// その手前の shape → 要素数の積を検証する。外部由来の shape がこの
/// 経路へ流入しうるための前段検証。OWASP A03）。
fn checked_numel(shape: &[usize]) -> Result<usize, MetalError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            // エラー値構築のために `shape.iter().product()`（checked 乗算）を
            // 再計算しない。`try_fold` が既に検知したのと同じオーバーフロー
            // 乗算を `product()` で踏むと、debug プロファイル
            // （overflow-checks 既定 ON）ではここ自体が
            // `attempt to multiply with overflow` で panic し、本関数が
            // 防御対象とする入力（shape 由来のオーバーフロー）で目的を
            // 果たせなくなる（CPU/CUDA 版の `ok_or_else` + 遅延評価パターンに
            // 揃える）。`wrapping_mul` は overflow-checks の影響を受けず
            // panic しないため、参考値としての近似 len を安全に算出できる。
            MetalError::AllocationSizeOverflow {
                len: shape.iter().fold(1usize, |acc, &dim| acc.wrapping_mul(dim)),
            }
        })
}

/// `numel` 分の `f32` 確保が消費するバイト数を検査付きで計算する
/// （TASK-14.1b・#175。計測専用ヘルパー）。
///
/// `buffer.rs::checked_byte_len`（`pub(crate)` ではなく private）とは
/// 名前が同じだが、意図的に**挙動が異なる**: `buffer.rs` 側は
/// `len == 0` を `MetalError::ZeroLengthAllocation` として拒否する
/// （FFI 呼び出し前の防御）のに対し、本関数は `numel == 0` を `Ok(0)` で
/// 通す（空テンソル契約における `TrackedAllocation::new(tracker, 0)` の
/// no-op 計上に使うため。`buffer.rs` 側の呼び出しは空テンソル経路では
/// 到達しない〈`alloc_zeroed_inner`／`upload_inner` が `numel == 0` を
/// FFI 呼び出し前に分岐で回避する〉。`checked_numel` の後段検証として
/// 配置する点は CPU/CUDA 実装と同型。外部由来の shape がこの経路へ
/// 流入しうるための OWASP A03 対策）。
fn checked_byte_len(numel: usize) -> Result<u64, MetalError> {
    let bytes = numel
        .checked_mul(size_of::<f32>())
        .ok_or(MetalError::AllocationSizeOverflow { len: numel })?;
    Ok(bytes as u64)
}

impl MetalMemory {
    fn alloc_zeroed_inner(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, MetalError> {
        let numel = checked_numel(shape)?;
        // 計測（`TrackedAllocation::new`）は確保成功後に行う。確保が
        // 失敗しうる `?` の前でカウントすると、失敗した確保が一時的に
        // ピークへ計上されてしまう（`backend-cpu::CpuMemory`／
        // `backend-cuda::CudaMemory` と同じ順序契約。TASK-14.1b・#175）。
        let handle: Box<dyn BufferHandle> = if numel == 0 {
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            Box::new(MetalBufferHandle {
                buffer: None,
                _alloc: alloc,
            })
        } else {
            let buf = MetalBuffer::alloc_zeroed_pooled(&self.context, numel)?;
            // `alloc_zeroed_pooled` は `self.context` がプロセスワイド
            // singleton と一致する場合のみプール経由（`Backing::Pooled`）
            // になり、一致しない場合（`MetalMemory::new` 経由の専有
            // コンテキスト。`tests/memory_roundtrip.rs`・`bench-harness::
            // peak_memory` が使う経路）は `new_zeroed` と同一の専有確保
            // （`Backing::Owned`）へフォールバックする
            // （`buffer.rs::MetalBuffer::singleton_context_matching`）。
            //
            // **常に論理バイト数（`checked_byte_len(numel)`）を
            // `self.tracker` へ計上する（codex-review P1 是正。PR #1063）**:
            // 旧稿はプール経由（`MetalBuffer::is_pooled() == true`。
            // 二重計上防止のためだけの用途を失ったため本 PR で
            // 削除済み）の場合に `0` バイトを積んでいた。理由付けは
            // 「`crate::pool::
            // MetalAllocator::tracker`（`RawMetalBuffer::_alloc`）に
            // 既に計上されているため二重計上を避ける」だったが、
            // `MetalAllocator::tracker` と `self.tracker`（`MetalMemory`
            // 自身のフィールド）は**別々の `Arc<AllocationTracker>`
            // インスタンス**であり、両者を合算する経路はどこにも
            // 存在しない（`MetalAllocator` 自体は `MemoryStats` を
            // 実装しておらず、REQ-14 の公開契約〈`allocated_bytes`／
            // `peak_allocated_bytes`〉から到達可能なのは `self.tracker`
            // のみ）。つまり「二重計上」は実際には発生しておらず、
            // `0` を積む分岐は本節冒頭（上記コメント §139-142 相当。
            // `impl MemoryStats for MetalMemory`）が明記する「CPU/CUDA/
            // Metal で同一契約」を破る過小計上バグだった
            // （`static_metal_memory()`〈`ops.rs`〉経由の cached
            // singleton context は常にプール経由になるため、通常の
            // `MemoryOps::alloc_zeroed` 呼び出し経路で実害が生じていた。
            // codex-review 指摘）。CUDA 側（`backend-cuda::memory::
            // CudaMemory::alloc_zeroed_inner`）は `self.stream.
            // alloc_zeros` を直接呼び `backend-cuda::pool::
            // CudaAllocator`（GEMM 等ホットパス専用。REQ-14 の
            // `MemoryOps` 経路とは別系統）を一切経由しないため同種の
            // 問題を構造的に持たない。CUDA 側の契約（`MemoryStats` は
            // 常に論理確保量を反映する）に Metal 側を揃えるため、
            // 上記の判定分岐は廃し常に `checked_byte_len` を計上する
            // （プール経由の物理容量〈サイズクラス丸め後の
            // capacity〉ではなく、他バックエンドと同じ論理要求量を
            // 計上する契約は維持する）。プール実装自体が持つ物理確保量
            // の計測（`crate::pool::MetalAllocator::stats()`〈`PoolStats::
            // cached_bytes`〉との統合）は `docs/device-memory-pool-
            // design.md` §8 スコープ外〈bench-harness の独自計測経路
            // との統合〉として申し送り済みのまま変更しない。
            let tracked_bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), tracked_bytes);
            Box::new(MetalBufferHandle {
                buffer: Some(buf),
                _alloc: alloc,
            })
        };
        Ok(DeviceBuffer::new(Device::Metal, shape.to_vec(), handle))
    }

    fn upload_inner(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, MetalError> {
        let shape = tensor.shape().to_vec();
        if tensor.numel() == 0 {
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(MetalBufferHandle {
                buffer: None,
                _alloc: alloc,
            });
            return Ok(DeviceBuffer::new(Device::Metal, shape, handle));
        }
        // 非 contiguous な入力は実体化してから転送する（`MemoryOps::upload`
        // の契約。`fandhe_ai_tensor_core::buffer` モジュールコメント参照）。
        let contiguous = tensor.contiguous();
        let data = contiguous.as_slice().ok_or(MetalError::BufferAllocation {
            bytes: 0, // contiguous() 直後に as_slice が None を返す到達不能パス。
        })?;
        let buf = MetalBuffer::new_with_data(&self.context, data)?;
        let bytes = checked_byte_len(data.len())?;
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(MetalBufferHandle {
            buffer: Some(buf),
            _alloc: alloc,
        });
        Ok(DeviceBuffer::new(Device::Metal, shape, handle))
    }

    fn download_inner(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, MetalError> {
        let handle = buffer
            .downcast_handle::<MetalBufferHandle>()
            .ok_or(MetalError::DeviceUnavailable)?;
        // ホスト実体化の同期点（イシュー #1017）: `read_to_vec` の前に
        // `context.rs::MetalContext::synchronize` を挟み、コマンドバッファ
        // 共有バッチに積まれた dispatch（`sgd.rs::MetalSgd::run` 等）の
        // GPU 完了を待つ。UMA（`StorageModeShared`）は物理メモリ共有のみを
        // 保証し、GPU 実行完了前に `contents()` を読む未定義動作を防ぐ
        // ためには本呼び出しが必須（モジュール冒頭コメント参照）。
        self.context.synchronize()?;
        let data = match &handle.buffer {
            None => Vec::new(),
            Some(buf) => buf.read_to_vec(),
        };
        // shape 不整合（通常到達しない防御的経路）を `BufferAllocation
        // { bytes: 0 }` のような実態と異なる variant に化けさせず、
        // 元の `ShapeError` の詳細を `MetalError::ShapeMismatch` として
        // 保持する（CUDA 実装の `CudaError::InvalidShape { detail:
        // format!(...) }` と同型の防御的経路。レビュー指摘対応）。
        Tensor::new(data, buffer.shape()).map_err(|err| MetalError::ShapeMismatch {
            detail: format!("download produced a shape-inconsistent tensor: {err}"),
        })
    }
}

impl MemoryOps for MetalMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        self.alloc_zeroed_inner(shape).map_err(map_metal_error)
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        self.upload_inner(tensor).map_err(map_metal_error)
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        // ハンドル型不一致（他バックエンドの `DeviceBuffer` を誤って
        // 渡した場合）は、CPU 実装（`backend-cpu/src/memory.rs`）と
        // 同じ `BackendError::DeviceMismatch` に統一する。`MetalError`
        // には「デバイス確保失敗」「デバイス利用不可」等の既存 variant
        // しかなく、`map_metal_error` を経由すると実態と異なるエラー種別
        // （`DeviceUnavailable`）に化けてしまうため、ここで直接判定する
        // （3 バックエンド共通のハンドル型不一致検出。レビュー指摘対応）。
        if buffer.downcast_handle::<MetalBufferHandle>().is_none() {
            return Err(BackendError::DeviceMismatch);
        }
        self.download_inner(buffer).map_err(map_metal_error)
    }
}

/// `fandhe_ai_tensor_core::pool::PooledMemory<MetalMemory>`（TASK-#201・REQ-14
/// 14-3）が再利用バッファを返す前に呼ぶゼロ初期化フック。
/// `MetalBuffer::zero_fill`（`buffer.rs`）へ委譲する（`StorageModeShared`
/// の CPU 可視アドレスへの直接書き込み。モジュール冒頭コメント
/// 「既存バッファ書き込みパターン踏襲」参照）。プール保持中も
/// `MetalBufferHandle::_alloc`（`TrackedAllocation`。TASK-14.1b・#175）は
/// 生存し続けるため、「返却されたが未解放のバッファ」も
/// `allocated_bytes()` に自然に計上され続ける（リークではなく意図した
/// 挙動。`backend-cuda::memory` の同型コメント参照）。実機でのピーク
/// 計測の裏取りは TASK-14.2（#177）で実施する。
impl PoolZeroFill for MetalMemory {
    fn zero_fill(&self, handle: &mut dyn BufferHandle) -> Result<(), BackendError> {
        let Some(metal_handle) = handle.as_any_mut().downcast_mut::<MetalBufferHandle>() else {
            return Err(BackendError::DeviceMismatch);
        };
        // 空ハンドル（`numel == 0`）は `pool.rs::PooledMemory::alloc_zeroed`
        // が空テンソル契約によりそもそもプールを介さない経路で扱うため
        // 到達しない想定だが、`buffer` が `None` の場合に備えて no-op
        // として安全に振る舞う（CUDA 実装と同じ防御的分岐）。
        if let Some(buf) = metal_handle.buffer.as_ref() {
            // ホスト実体化の同期点（イシュー #1017。`download_inner` と
            // 同じ理由）: プールから再利用したバッファへ書き込む前に、
            // そのバッファを最後に読み書きした dispatch の GPU 完了を
            // 待つ。バッチに未完了の dispatch が残ったまま
            // `contents()` へ直接ゼロ書き込みすると、GPU 実行中の
            // バッファを CPU 側から上書きする未定義動作になりうる。
            self.context.synchronize().map_err(map_metal_error)?;
            buf.zero_fill();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, MetalError::AllocationSizeOverflow { .. }));
    }

    #[test]
    fn checked_numel_accepts_ordinary_shape() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel(&[0, 3]).unwrap(), 0);
    }

    #[test]
    fn map_metal_error_covers_zero_length_allocation() {
        let err = map_metal_error(MetalError::ZeroLengthAllocation);
        assert!(matches!(err, BackendError::DeviceAllocationFailed(_)));
    }

    #[test]
    fn map_metal_error_covers_device_unavailable() {
        let err = map_metal_error(MetalError::DeviceUnavailable);
        assert!(matches!(err, BackendError::DeviceUnavailable(_)));
    }

    #[test]
    fn checked_byte_len_rejects_overflow() {
        let err = checked_byte_len(usize::MAX).unwrap_err();
        assert!(matches!(err, MetalError::AllocationSizeOverflow { .. }));
    }

    #[test]
    fn checked_byte_len_accepts_ordinary_numel_including_zero() {
        // `buffer.rs::checked_byte_len` とは異なり、本モジュールの
        // 計測用 `checked_byte_len` は `numel == 0` を `Ok(0)` で通す
        // （関数 doc コメント「意図的に挙動が異なる」参照）。
        assert_eq!(checked_byte_len(1024).unwrap(), 4096);
        assert_eq!(checked_byte_len(0).unwrap(), 0);
    }

    /// コンパイル時の静的検査。`fn(): T where T: MemoryStats` が
    /// `MetalMemory`／`PooledMemory<MetalMemory>` に対して呼び出せること
    /// 自体が、「CPU/CUDA/Metal で同一 API（同一シグネチャの trait）から
    /// ピーク値が取得できる」という REQ-14 の受け入れ条件を Linux
    /// self-hosted CI（Metal 非搭載）でも `aarch64-apple-darwin` クロス
    /// ビルド経由で機械検証する（TASK-14.1b・#175。実機でのピーク実測は
    /// TASK-14.2・#177 で裏取りする）。
    fn assert_memory_stats<T: MemoryStats>() {}

    #[test]
    fn metal_memory_and_pooled_metal_memory_implement_memory_stats() {
        assert_memory_stats::<MetalMemory>();
        assert_memory_stats::<fandhe_ai_tensor_core::pool::PooledMemory<MetalMemory>>();
    }

    /// pooled 経路（`static_metal_memory()` が使う singleton
    /// `MetalContext`）でも `MemoryStats::allocated_bytes`／
    /// `peak_allocated_bytes` が論理要求バイト数を反映することを確認
    /// する（codex-review P1 是正の回帰テスト。PR #1063）。
    ///
    /// `context_cache::cached_context()` は `pub(crate)` のため、この
    /// 検証は外部統合テスト（`tests/memory_roundtrip.rs`）ではなく
    /// クレート内部の `#[cfg(test)]` からのみ行える。`MetalMemory::new`
    /// （専有コンテキスト）は `singleton_context_matching` により常に
    /// プール非経由（`Backing::Owned`）へフォールバックするため、この
    /// バグ（`Backing::Pooled` の場合に `0` を計上していた）を再現・
    /// 検証できない。`MetalMemory::from_shared(context_cache::
    /// cached_context()?)` で singleton と同一の `Arc<MetalContext>` を
    /// 明示的に共有し、プール経由分岐を確実に踏む。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn pooled_alloc_zeroed_records_logical_bytes_via_shared_context() {
        let ctx = crate::context_cache::cached_context()
            .expect("singleton MetalContext の取得に失敗した（実機依存）");
        let mem = MetalMemory::from_shared(ctx);

        let numel = 4096usize;
        let expected_bytes = (numel * size_of::<f32>()) as u64;

        let buf = mem
            .alloc_zeroed(&[numel])
            .expect("pooled alloc_zeroed は成功するはず");
        assert_eq!(
            mem.allocated_bytes(),
            expected_bytes,
            "pooled 経路（Backing::Pooled）でも 0 ではなく論理要求バイト数を              計上するはず（codex-review P1 是正）"
        );
        assert_eq!(mem.peak_allocated_bytes(), expected_bytes);

        drop(buf);
        assert_eq!(mem.allocated_bytes(), 0, "貸出終了後は current が 0 に戻る");
        assert_eq!(
            mem.peak_allocated_bytes(),
            expected_bytes,
            "peak は過去最大値を保持する"
        );
    }
}
