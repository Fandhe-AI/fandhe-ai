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
//! コメント参照）を用いるため、CUDA のような明示的な非同期転送・
//! `synchronize()` は不要（`MetalBuffer::read_to_vec` は `contents()` の
//! CPU 可視アドレスを直接読むため、呼び出し復帰時点でホストデータは
//! 既に確定している）。

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
#[derive(Debug)]
struct MetalBufferHandle {
    buffer: Option<MetalBuffer>,
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
    context: MetalContext,
    tracker: Arc<AllocationTracker>,
}

impl MetalMemory {
    /// 初期化済みの [`MetalContext`] から `MetalMemory` を構築する。
    /// 新規の計測系列を持つトラッカーを生成する
    /// （`backend-cpu::CpuMemory::new` と同型）。
    pub fn new(context: MetalContext) -> Self {
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
            let buf = MetalBuffer::new_zeroed(&self.context, numel)?;
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
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
        let data = match &handle.buffer {
            // StorageModeShared のため read_to_vec 復帰時点でホストデータは
            // 既に確定している（同期不要。モジュール冒頭コメント参照）。
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
}
