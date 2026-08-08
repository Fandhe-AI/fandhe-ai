//! CPU バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `tensor_core::buffer::MemoryOps` の CPU 実装。CPU は「デバイス」が
//! ホストメモリそのものであるため、`upload`/`download` は FFI を伴わず
//! `Vec<f32>` の複製のみで完結する。`backend-cuda::CudaMemory`／
//! `backend-metal::MetalMemory` の数値一致の参照点（`.claude/rules/coding-rust.md`
//! の「CPU 参照実装」方針）であり、`CpuDeviceProvider`（`device.rs`・#44）
//! と同種の位置付けを `MemoryOps` 側で担う。
//!
//! TASK-14.1a（#174）で `tensor_core::memory_stats::MemoryStats` を実装し、
//! `alloc_zeroed`/`upload` の確保バイト数を [`AllocationTracker`] へ計上する
//! ようにした。トラッカーは `CpuMemory` インスタンス間で `Arc` 共有する
//! （`memory_stats` モジュールコメント「トラッカーの共有範囲」参照。
//! CUDA/Metal への同フック組み込みは #175・TASK-14.1b で完了済み。
//! `backend-cuda::memory::CudaMemory`／`backend-metal::memory::MetalMemory`
//! も本ファイルと同型の `tracker: Arc<AllocationTracker>` パターンで
//! `MemoryStats` を実装する）。

use std::any::Any;
use std::mem::size_of;
use std::sync::Arc;

use tensor_core::Tensor;
use tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use tensor_core::device::{BackendError, Device};
use tensor_core::memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
use tensor_core::pool::PoolZeroFill;

/// CPU バッファの具体ハンドル。ホスト常駐そのものなので `Vec<f32>` を
/// 直接保持する。`Drop` は `Vec<f32>` の既定 `Drop` に委ねる（RAII 一本化
/// 方針。`tensor_core::buffer` モジュールコメント参照）。
///
/// `_alloc`（[`TrackedAllocation`]）はフィールド順により `data` より後に
/// drop される（Rust の構造体フィールドは宣言順に drop される）ため、
/// 計測上は問題にならない（`TrackedAllocation::drop` は `data` の中身を
/// 参照せず、確保時に記録したバイト数を `AllocationTracker` へ返すだけ）。
/// 明示的に保持しているだけで、`data` の `Drop` と同時に解放計上される
/// （RAII 一本化。`buffer.rs` モジュールコメント参照）。
#[derive(Debug)]
struct CpuBufferHandle {
    data: Vec<f32>,
    _alloc: TrackedAllocation,
}

impl BufferHandle for CpuBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// `MemoryOps`／`MemoryStats` の CPU 実装。`tracker` は `Arc` で共有される
/// ため `Clone` は「同一計測系列への参照複製」を意味する（`derive(Copy)` を
/// 外したのは TASK-14.1a による破壊的変更。ワークスペース内に `CpuMemory`
/// の `Copy` へ依存する呼び出し箇所は無いことを確認済み〈grep 済み〉。
/// `Device::Cpu` は単一デバイスであり、`CpuDeviceProvider` と同じく構築
/// 自体が失敗しない点は変更していない）。
#[derive(Debug, Default, Clone)]
pub struct CpuMemory {
    tracker: Arc<AllocationTracker>,
}

impl CpuMemory {
    /// 新規 `CpuMemory` を構築する（新規の計測系列を持つトラッカーを
    /// 生成する）。同一プロセス内でピークを集約したい場合は `clone()` で
    /// トラッカーを共有する（`memory_stats` モジュールコメント参照）。
    pub fn new() -> Self {
        Self {
            tracker: Arc::new(AllocationTracker::new()),
        }
    }
}

impl MemoryStats for CpuMemory {
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

/// `tensor_core::pool::PooledMemory<CpuMemory>`（TASK-#201・REQ-14 14-3）が
/// プールから再利用したバッファを返す前に呼ぶゼロ初期化フック。
/// `handle` を `CpuBufferHandle` へダウンキャストし、`data`（`Vec<f32>`）を
/// `fill(0.0)` で上書きする（`alloc_zeroed` の「全要素 0」契約を再利用時
/// にも維持する。前利用データの残留はプロセス内情報漏えいリスクでもある。
/// `.claude/rules/security.md` A02/A04）。
impl PoolZeroFill for CpuMemory {
    fn zero_fill(&self, handle: &mut dyn BufferHandle) -> Result<(), BackendError> {
        // `&mut dyn BufferHandle`（`PoolZeroFill::zero_fill` のシグネチャ。
        // `pool.rs` モジュールコメント「ゼロ初期化契約の維持」参照）を
        // 経由することで、`unsafe` な生ポインタ書き込みなしに安全な
        // `downcast_mut` + `Vec::fill` だけで完結する（`.claude/rules/
        // coding-rust.md`「`unsafe` は FFI 境界等の必要最小限に留める」
        // 方針。ダウンキャスト失敗＝他バックエンドのハンドルが誤って
        // 渡された等は `PooledMemory` 側の不変条件違反であり通常到達
        // しないが、`unwrap`/`expect` は使わず型付きエラーで返す）。
        let Some(cpu_handle) = handle.as_any_mut().downcast_mut::<CpuBufferHandle>() else {
            return Err(BackendError::DeviceMismatch);
        };
        cpu_handle.data.fill(0.0);
        Ok(())
    }
}

/// `numel` 分の `f32` 確保が消費するバイト数を検査付きで計算する。
/// `checked_numel` の後段に置く checked 演算（`.claude/rules/security.md`
/// A03: shape は safetensors/ONNX 経由で外部入力が流入しうる経路のため、
/// バイト数換算でもオーバーフローを型付きエラーとして拒否する）。
fn checked_byte_len(numel: usize) -> Result<u64, BackendError> {
    let bytes = numel.checked_mul(size_of::<f32>()).ok_or_else(|| {
        BackendError::DeviceAllocationFailed(format!(
            "allocation byte length overflows usize: numel={numel}"
        ))
    })?;
    Ok(bytes as u64)
}

/// shape の要素数積を検査付きで計算する。`Tensor::zeros` 等が内部で行う
/// 検証と同種だが、`tensor_core::tensor::checked_numel` はクレート内
/// 非公開（`pub(crate)`）のため、`backend-cpu` 側で同じ検証をここでも
/// 独立して行う（`.claude/rules/security.md` A03: 外部由来の shape が
/// この経路へ流入しうるための前段検証）。
fn checked_numel(shape: &[usize]) -> Result<usize, BackendError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            BackendError::DeviceAllocationFailed(format!(
                "shape element count overflows usize: {shape:?}"
            ))
        })
}

impl MemoryOps for CpuMemory {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        let numel = checked_numel(shape)?;
        // numel == 0 でも `vec![0.0f32; 0]` は素通りする（CPU には
        // MetalBuffer::checked_byte_len のような FFI 前ゼロ長拒否は
        // 不要）が、他バックエンドと同じ「空ハンドル」契約
        // （`tensor_core::buffer` モジュールコメント）に揃えて明示的に
        // 分岐しておく（将来 CPU 側にプール確保等を導入した際に契約が
        // 崩れないようにするための保険）。numel == 0 は checked_byte_len も
        // 0 を返すため TrackedAllocation::new への計上は自然な no-op になる
        // （`memory_stats` モジュールコメント参照）。
        let bytes = checked_byte_len(numel)?;
        let data = vec![0.0f32; numel];
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle {
            data,
            _alloc: alloc,
        });
        Ok(DeviceBuffer::new(Device::Cpu, shape.to_vec(), handle))
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        let shape = tensor.shape().to_vec();
        if tensor.numel() == 0 {
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle {
                data: Vec::new(),
                _alloc: alloc,
            });
            return Ok(DeviceBuffer::new(Device::Cpu, shape, handle));
        }
        // 非 contiguous な入力（transpose/narrow 後の view）は実体化して
        // から複製する（`MemoryOps::upload` の契約。`tensor_core::buffer`
        // モジュールコメント「非 contiguous テンソルの upload」参照）。
        let contiguous = tensor.contiguous();
        let data = contiguous
            .as_slice()
            .ok_or_else(|| {
                BackendError::DeviceAllocationFailed(
                    "contiguous() の直後にもかかわらず as_slice が None を返した \
                     （tensor-core 側のロジック不整合。到達しないはずの防御経路）"
                        .to_string(),
                )
            })?
            .to_vec();
        let bytes = checked_byte_len(data.len())?;
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle {
            data,
            _alloc: alloc,
        });
        Ok(DeviceBuffer::new(Device::Cpu, shape, handle))
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        if buffer.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let handle = buffer
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if buffer.numel() == 0 {
            return Tensor::new(Vec::new(), buffer.shape()).map_err(BackendError::ShapeMismatch);
        }
        Tensor::new(handle.data.clone(), buffer.shape()).map_err(BackendError::ShapeMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_download_roundtrip_is_bit_exact() {
        let mem = CpuMemory::new();
        let data = vec![1.0f32, -2.5, 3.25, f32::MIN_POSITIVE, f32::MAX];
        let tensor = Tensor::<f32>::new(data.clone(), &[5]).unwrap();

        let buf = mem.upload(&tensor).unwrap();
        assert_eq!(buf.shape(), &[5]);
        assert_eq!(buf.device(), Device::Cpu);

        let back = mem.download(&buf).unwrap();
        assert_eq!(back.shape(), &[5]);
        for (i, &expected) in data.iter().enumerate() {
            let actual = back.get(&[i]).unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "roundtrip must be bit exact at index {i}"
            );
        }
    }

    #[test]
    fn upload_non_contiguous_tensor_materializes_before_transfer() {
        let mem = CpuMemory::new();
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        // transpose 直後は非 contiguous（as_slice は None）。upload は
        // これを内部で contiguous() 実体化してから転送できることを確認する。
        assert!(!tt.is_contiguous());
        assert!(tt.as_slice().is_none());

        let buf = mem.upload(&tt).unwrap();
        let back = mem.download(&buf).unwrap();
        assert_eq!(back.shape(), &[3, 2]);
        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(back.get(&[i, j]).unwrap(), tt.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn zero_numel_tensor_roundtrips_without_error() {
        let mem = CpuMemory::new();
        let empty = Tensor::<f32>::zeros(&[0, 3]).unwrap();

        let buf = mem.upload(&empty).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.shape(), &[0, 3]);

        let back = mem.download(&buf).unwrap();
        assert!(back.is_empty());
        assert_eq!(back.shape(), &[0, 3]);
    }

    #[test]
    fn alloc_zeroed_returns_all_zero_buffer() {
        let mem = CpuMemory::new();
        let buf = mem.alloc_zeroed(&[2, 2]).unwrap();
        let tensor = mem.download(&buf).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(tensor.get(&[i, j]).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn alloc_zeroed_element_count_overflow_is_rejected() {
        let mem = CpuMemory::new();
        let err = mem.alloc_zeroed(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, BackendError::DeviceAllocationFailed(_)));
    }

    #[test]
    fn download_rejects_mismatched_handle_type() {
        // 他バックエンドのハンドル型を模した downcast 失敗経路の検証。
        #[derive(Debug)]
        struct OtherHandle;
        impl BufferHandle for OtherHandle {
            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }
        let mem = CpuMemory::new();
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![1], Box::new(OtherHandle));
        let err = mem.download(&buf).unwrap_err();
        assert!(matches!(err, BackendError::DeviceMismatch));
    }

    #[test]
    fn repeated_upload_alloc_cycles_do_not_leak_across_backend_error_reuse() {
        // リークなく動作する（受け入れ条件）の CPU 側検証: 大量サイクルで
        // panic せず・値が破壊されないことを確認する（`Vec<f32>` の
        // `Drop` に委ねる RAII 方式のため実際のメモリ解放そのものは
        // tensor-core 側モックテストで検証済み。ここでは CPU 経路が
        // 繰り返し呼び出しに対して安定して動作することを確認する）。
        let mem = CpuMemory::new();
        for i in 0..200 {
            let shape = [((i % 8) + 1)];
            let buf = mem.alloc_zeroed(&shape).unwrap();
            let back = mem.download(&buf).unwrap();
            assert_eq!(back.numel(), shape[0]);
        }
    }

    /// 受け入れ条件「CPU バックエンドでピーク値が取得できる」の直接検証。
    /// `[1024]` の f32 確保は `1024 * size_of::<f32>() == 4096` バイトと
    /// 期待値が一致することを確認する（TASK-14.1a・#174）。
    #[test]
    fn peak_allocated_bytes_matches_known_allocation_size() {
        let mem = CpuMemory::new();
        assert_eq!(mem.allocated_bytes(), 0);
        assert_eq!(mem.peak_allocated_bytes(), 0);

        let buf = mem.alloc_zeroed(&[1024]).unwrap();
        assert_eq!(mem.allocated_bytes(), 4096);
        assert_eq!(mem.peak_allocated_bytes(), 4096);

        drop(buf);
        assert_eq!(
            mem.allocated_bytes(),
            0,
            "解放後は allocated_bytes が 0 に戻る"
        );
        assert_eq!(
            mem.peak_allocated_bytes(),
            4096,
            "peak は解放後も過去最大値を保持する"
        );
    }

    /// 2 本同時保持のピークは合計値になる（tensor-core 側の
    /// `AllocationTracker` テストの CPU 経路での再検証）。
    #[test]
    fn peak_allocated_bytes_tracks_sum_of_concurrent_buffers() {
        let mem = CpuMemory::new();
        let a = mem.alloc_zeroed(&[256]).unwrap(); // 1024 バイト
        let b = mem.alloc_zeroed(&[256]).unwrap(); // 1024 バイト

        assert_eq!(mem.allocated_bytes(), 2048);
        assert_eq!(mem.peak_allocated_bytes(), 2048);

        drop(a);
        drop(b);
        assert_eq!(mem.allocated_bytes(), 0);
        assert_eq!(mem.peak_allocated_bytes(), 2048);
    }

    /// `reset_peak` 後は peak が現在値まで引き下がり、以降の確保で
    /// 新しいピーク区間として計測できることを確認する。
    #[test]
    fn reset_peak_starts_a_new_measurement_window() {
        let mem = CpuMemory::new();
        let a = mem.alloc_zeroed(&[1024]).unwrap(); // 4096 バイト
        drop(a);
        assert_eq!(mem.peak_allocated_bytes(), 4096);

        mem.reset_peak();
        assert_eq!(mem.peak_allocated_bytes(), mem.allocated_bytes());

        let b = mem.alloc_zeroed(&[128]).unwrap(); // 512 バイト
        assert_eq!(mem.peak_allocated_bytes(), 512);
        drop(b);
    }

    /// 空テンソル（numel 0）の確保が計数へ影響しないことを確認する
    /// （`memory_stats` モジュールコメント「0 バイト加算は no-op」参照）。
    #[test]
    fn empty_allocation_does_not_affect_peak() {
        let mem = CpuMemory::new();
        let empty = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let buf = mem.upload(&empty).unwrap();

        assert_eq!(mem.allocated_bytes(), 0);
        assert_eq!(mem.peak_allocated_bytes(), 0);
        drop(buf);
        assert_eq!(mem.allocated_bytes(), 0);
    }

    /// `CpuMemory::clone()` はトラッカーを共有する（`Arc` 経由）ため、
    /// clone された側から確保しても元のインスタンス経由で同じピークが
    /// 観測できることを確認する（`memory_stats` モジュールコメント
    /// 「トラッカーの共有範囲」参照）。
    #[test]
    fn cloned_cpu_memory_shares_the_same_tracker() {
        let mem = CpuMemory::new();
        let mem_clone = mem.clone();

        let buf = mem_clone.alloc_zeroed(&[64]).unwrap(); // 256 バイト
        assert_eq!(mem.allocated_bytes(), 256, "clone 元からも計上が見える");
        assert_eq!(mem.peak_allocated_bytes(), 256);
        drop(buf);
    }
}
