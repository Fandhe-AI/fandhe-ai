//! CPU バックエンドのメモリ操作（TASK-1.9b・#45）。
//!
//! `fandhe_ai_tensor_core::buffer::MemoryOps` の CPU 実装。CPU は「デバイス」が
//! ホストメモリそのものであるため、`upload`/`download` は FFI を伴わず
//! `Vec<f32>` の複製のみで完結する。`backend-cuda::CudaMemory`／
//! `backend-metal::MetalMemory` の数値一致の参照点（`.claude/rules/coding-rust.md`
//! の「CPU 参照実装」方針）であり、`CpuDeviceProvider`（`device.rs`・#44）
//! と同種の位置付けを `MemoryOps` 側で担う。
//!
//! TASK-14.1a（#174）で `fandhe_ai_tensor_core::memory_stats::MemoryStats` を実装し、
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

use fandhe_ai_tensor_core::ShapeError;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
use fandhe_ai_tensor_core::pool::PoolZeroFill;

/// CPU バッファの具体ハンドル。ホスト常駐そのものなので `Vec<f32>` を
/// 直接保持する。`Drop` は `Vec<f32>` の既定 `Drop` に委ねる（RAII 一本化
/// 方針。`fandhe_ai_tensor_core::buffer` モジュールコメント参照）。
///
/// `_alloc`（[`TrackedAllocation`]）はフィールド順により `data` より後に
/// drop される（Rust の構造体フィールドは宣言順に drop される）ため、
/// 計測上は問題にならない（`TrackedAllocation::drop` は `data` の中身を
/// 参照せず、確保時に記録したバイト数を `AllocationTracker` へ返すだけ）。
/// 明示的に保持しているだけで、`data` の `Drop` と同時に解放計上される
/// （RAII 一本化。`buffer.rs` モジュールコメント参照）。
/// `pub(crate)`（本来は module-private でよいが、イシュー #935・
/// `docs/device-resident-update-design.md` §3.2 で `ops.rs::CpuBackendOps::
/// sgd_step_device` が `DeviceBuffer::downcast_handle_mut` 経由で in-place
/// 書き換えを行うために `crate::memory::CpuBufferHandle` として参照する
/// 必要があり、可視性を crate 内に広げた）。
mod handle {
    use super::TrackedAllocation;
    use crate::gemm_blis::prepack::PrepackCache;

    /// 本体。`data` は private で、書き込み口は [`CpuBufferHandle::data_mut`] だけにする
    /// （#3200。常駐重みの pack 済み panel キャッシュ `prepack` を、書き込みのたびに必ず
    /// 無効化するための型による強制。読み手は `&CpuBufferHandle` を GEMM の間じゅう保持するため、
    /// `&mut self` を要求する `data_mut` と重なることは借用規則上起きない）。
    #[derive(Debug)]
    pub(crate) struct CpuBufferHandle {
        data: Vec<f32>,
        prepack: PrepackCache,
        _alloc: TrackedAllocation,
    }

    impl CpuBufferHandle {
        /// 構築口（`memory.rs` の 4 か所の確保経路が共有する）。
        pub(super) fn new(data: Vec<f32>, alloc: TrackedAllocation) -> Self {
            Self {
                data,
                prepack: PrepackCache::default(),
                _alloc: alloc,
            }
        }

        /// 読み取り専用のホストデータ。
        pub(crate) fn data(&self) -> &[f32] {
            &self.data
        }

        /// 書き込み口。pack 済み panel キャッシュを先に破棄してから返す
        /// （`Mutex::get_mut` でロック不要）。
        pub(crate) fn data_mut(&mut self) -> &mut [f32] {
            self.prepack.clear_mut();
            &mut self.data
        }

        /// ホストデータを所有権ごと取り出す（`download_owned`。#3672）。
        /// `data_mut` と同じく pack 済み panel キャッシュを先に破棄して #3200 の不変条件を保つ。
        /// 抜け殻のハンドルは空の `Vec` だけを持つ（`TrackedAllocation` は drop で解放計上される）。
        pub(crate) fn take_data(&mut self) -> Vec<f32> {
            self.prepack.clear_mut();
            std::mem::take(&mut self.data)
        }

        /// テスト用: 要素数を 1 つ減らす（長さ不一致の拒否経路の検証用。キャッシュも破棄する）。
        #[cfg(test)]
        pub(crate) fn pop_for_test(&mut self) {
            self.prepack.clear_mut();
            self.data.pop();
        }

        /// pack 済み B panel キャッシュ（`ops.rs` の forward が引き当てに使う）。
        pub(crate) fn prepack_cache(&self) -> &PrepackCache {
            &self.prepack
        }
    }
}

pub(crate) use handle::CpuBufferHandle;

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

    /// 既に計算済みのホスト常駐 `data`（`shape` に整合する要素数）を
    /// コピーなしで [`DeviceBuffer`] としてラップする（イシュー #1028）。
    ///
    /// `ops.rs::CpuBackendOps::linear_forward_device` が推論チェーンの
    /// 中間出力（`gemm_blis_bias_act_parallel` の出力バッファ）を、`upload`
    /// （`Tensor` 経由の複製）を経由せずそのまま次層の入力
    /// `DeviceBuffer` へ仕立てるために使う（CPU は「デバイス」が
    /// ホストメモリそのものであるため、`data` の所有権移動だけで
    /// 「デバイス常駐」契約を満たせる）。`alloc_zeroed`/`upload` と同じ
    /// `AllocationTracker` 計上を行う（計測系列の一貫性を保つ。
    /// `memory_stats` モジュールコメント参照）。
    ///
    /// `data.len()` が `shape` の要素数積と一致しない場合は呼び出し元
    /// （`ops.rs`）のロジック不整合であり、fail-closed に
    /// [`BackendError::ShapeMismatch`] で拒否する（本番経路で
    /// `unwrap()`/`expect()` を使わない方針。`.claude/rules/
    /// coding-rust.md`）。
    pub(crate) fn wrap_vec(
        &self,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        let expected = checked_numel(&shape)?;
        if data.len() != expected {
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected,
                    actual: data.len(),
                },
            ));
        }
        let bytes = checked_byte_len(data.len())?;
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle::new(data, alloc));
        Ok(DeviceBuffer::new(Device::Cpu, shape, handle))
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

/// `fandhe_ai_tensor_core::pool::PooledMemory<CpuMemory>`（TASK-#201・REQ-14 14-3）が
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
        cpu_handle.data_mut().fill(0.0);
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
/// 検証と同種だが、`fandhe_ai_tensor_core::tensor::checked_numel` はクレート内
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
        // （`fandhe_ai_tensor_core::buffer` モジュールコメント）に揃えて明示的に
        // 分岐しておく（将来 CPU 側にプール確保等を導入した際に契約が
        // 崩れないようにするための保険）。numel == 0 は checked_byte_len も
        // 0 を返すため TrackedAllocation::new への計上は自然な no-op になる
        // （`memory_stats` モジュールコメント参照）。
        let bytes = checked_byte_len(numel)?;
        let data = vec![0.0f32; numel];
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle::new(data, alloc));
        Ok(DeviceBuffer::new(Device::Cpu, shape.to_vec(), handle))
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        let shape = tensor.shape().to_vec();
        if tensor.numel() == 0 {
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), 0);
            let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle::new(Vec::new(), alloc));
            return Ok(DeviceBuffer::new(Device::Cpu, shape, handle));
        }
        // 非 contiguous な入力（transpose/narrow 後の view）は実体化して
        // から複製する（`MemoryOps::upload` の契約。`fandhe_ai_tensor_core::buffer`
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
        let handle: Box<dyn BufferHandle> = Box::new(CpuBufferHandle::new(data, alloc));
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
        Tensor::new(handle.data().to_vec(), buffer.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`MemoryOps::download_owned`] の CPU 実装（イシュー #3672）。
    /// `handle.data` の `Vec` を `Tensor` へ移すだけで、N² の新規確保と memcpy を行わない
    /// （`unsafe` なし）。device・ダウンキャスト・長さの検査は `download` と同じで、
    /// 異種ハンドルは `DeviceMismatch` で拒否する（コピー版へ黙って落とさない）。
    fn download_owned(&self, buffer: DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        if buffer.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        // プール付きバッファは drop で内部ハンドルがプールへ戻るため、Vec を奪うと
        // 抜け殻が再利用される（codex P1）。コピー版へ退いて実体を保つ。
        if buffer.is_pool_owned() {
            return self.download(&buffer);
        }
        let shape = buffer.shape().to_vec();
        let numel = buffer.numel();
        // 不変ダウンキャストで型を先に確認する。`as_any` のみ実装した外部ハンドルに
        // 可変ダウンキャスト（`as_any_mut` 既定実装は panic）を行うと、
        // `DeviceMismatch` の前に panic するため（codex P1）。
        if buffer.downcast_handle::<CpuBufferHandle>().is_none() {
            return Err(BackendError::DeviceMismatch);
        }
        let mut buffer = buffer;
        let handle = buffer
            .downcast_handle_mut::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if numel == 0 {
            return Tensor::new(Vec::new(), &shape).map_err(BackendError::ShapeMismatch);
        }
        let data = handle.take_data();
        Tensor::new(data, &shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`MemoryOps::upload_into`] の CPU 実装（イシュー #1212）。
    /// `crate::ops::CpuBackendOps`（ホットパス。`memory_ops()` が
    /// `Some(self)` を返す実装）と契約を共有するため、実体は
    /// `crate::ops::upload_into_cpu_buffer` へ委譲する（`ops.rs`
    /// モジュール doc「ホットパス」参照。本実装からは通常到達しない
    /// が、契約の一貫性のため鏡像実装を置く）。
    fn upload_into(
        &self,
        tensor: &Tensor<f32>,
        dst: &mut DeviceBuffer<f32>,
        dst_offset: usize,
    ) -> Result<(), BackendError> {
        crate::ops::upload_into_cpu_buffer(tensor, dst, dst_offset)
    }

    /// [`MemoryOps::with_host_view`] の CPU 実装（イシュー #1335）。
    ///
    /// CPU バックエンドはそもそもホスト常駐（`CpuBufferHandle::data`）
    /// のため、既定の `download` 経由実装（`Tensor::new(handle.data
    /// .clone(), ...)` を経て `as_slice()` する）にある clone コピーを
    /// 経由せず、`handle.data` を直接借用する（コピーなし）。
    fn with_host_view(
        &self,
        buffer: &DeviceBuffer<f32>,
        f: &mut dyn FnMut(&[f32]),
    ) -> Result<(), BackendError> {
        if buffer.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let handle = buffer
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        f(handle.data());
        Ok(())
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

    /// `with_host_view` と `download().as_slice()` が bit 同一であること
    /// を確認する（イシュー #1335・AC2）。
    #[test]
    fn with_host_view_matches_download_bit_exact() {
        let mem = CpuMemory::new();
        let data = vec![1.0f32, -2.5, 3.25, f32::MIN_POSITIVE, f32::MAX];
        let tensor = Tensor::<f32>::new(data.clone(), &[5]).unwrap();
        let buf = mem.upload(&tensor).unwrap();

        let via_download = mem.download(&buf).unwrap();
        let via_download_bits: Vec<u32> = via_download
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();

        let mut via_view_bits = Vec::new();
        mem.with_host_view(&buf, &mut |slice| {
            via_view_bits = slice.iter().map(|v| v.to_bits()).collect();
        })
        .unwrap();

        assert_eq!(
            via_view_bits, via_download_bits,
            "with_host_view は download().as_slice() と bit 同一のはず"
        );
    }

    /// 空バッファ（`numel == 0`）は FFI を経由せず空スライスを渡す
    /// （モジュール冒頭「空テンソルの契約」）。
    #[test]
    fn with_host_view_on_empty_buffer_yields_empty_slice() {
        let mem = CpuMemory::new();
        let empty = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let buf = mem.upload(&empty).unwrap();

        let mut seen_len = usize::MAX;
        mem.with_host_view(&buf, &mut |slice| {
            seen_len = slice.len();
        })
        .unwrap();
        assert_eq!(seen_len, 0);
    }

    #[test]
    fn with_host_view_rejects_mismatched_handle_type() {
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
        let err = mem.with_host_view(&buf, &mut |_| {}).unwrap_err();
        assert!(matches!(err, BackendError::DeviceMismatch));
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

    /// `download_owned`（#3672）は `download` と bit 一致し、Vec を移す（ポインタ一致）。
    #[test]
    fn download_owned_is_bit_exact_and_moves_vec() {
        for shape in [vec![3usize, 5], vec![1, 1], vec![64, 128]] {
            let n: usize = shape.iter().product();
            let mut data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let specials = [
                f32::NAN,
                -0.0,
                0.0,
                f32::MIN_POSITIVE / 4.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ];
            for (i, v) in specials.iter().enumerate() {
                if i < n {
                    data[i] = *v;
                }
            }
            let mem = CpuMemory::new();
            let buf = mem
                .upload(&Tensor::<f32>::new(data, &shape).unwrap())
                .unwrap();
            let base = mem.allocated_bytes();
            assert!(base > 0);
            let copied = mem.download(&buf).unwrap();
            let mut ptr = std::ptr::null();
            mem.with_host_view(&buf, &mut |s| ptr = s.as_ptr()).unwrap();
            let owned = mem.download_owned(buf).unwrap();
            assert_eq!(owned.as_slice().unwrap().as_ptr(), ptr, "Vec が移るはず");
            assert_eq!(owned.shape(), copied.shape());
            let a: Vec<u32> = owned
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let b: Vec<u32> = copied
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(a, b);
            assert_eq!(mem.allocated_bytes(), 0, "バッファ解放が計上されるはず");
        }
    }

    #[test]
    fn download_owned_empty_and_mismatch() {
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
        let empty = mem.alloc_zeroed(&[0, 4]).unwrap();
        let t = mem.download_owned(empty).unwrap();
        assert_eq!(t.shape(), &[0, 4]);
        assert!(t.as_slice().unwrap().is_empty());

        let other: DeviceBuffer<f32> =
            DeviceBuffer::new(Device::Cpu, vec![1], Box::new(OtherHandle));
        assert!(matches!(
            mem.download_owned(other).unwrap_err(),
            BackendError::DeviceMismatch
        ));
        let other: DeviceBuffer<f32> =
            DeviceBuffer::new(Device::Cuda(0), vec![1], Box::new(OtherHandle));
        assert!(matches!(
            mem.download_owned(other).unwrap_err(),
            BackendError::DeviceMismatch
        ));
    }

    /// `as_any` のみ実装した外部ハンドルでも panic せず `DeviceMismatch` を返す（codex P1）。
    #[test]
    fn download_owned_foreign_handle_without_as_any_mut_is_mismatch() {
        #[derive(Debug)]
        struct ForeignHandle;
        impl BufferHandle for ForeignHandle {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let foreign: DeviceBuffer<f32> =
            DeviceBuffer::new(Device::Cpu, vec![1], Box::new(ForeignHandle));
        assert!(matches!(
            CpuMemory::new().download_owned(foreign).unwrap_err(),
            BackendError::DeviceMismatch
        ));
    }

    /// プール付きバッファは `download_owned` で Vec を奪わず、同サイズ再確保が健全（codex P1）。
    #[test]
    fn download_owned_pooled_buffer_keeps_shell_intact() {
        use fandhe_ai_tensor_core::pool::{PoolConfig, PooledMemory};
        let mem = PooledMemory::new(CpuMemory::new(), Device::Cpu, PoolConfig::default());
        let mut buf = mem.alloc_zeroed(&[4, 4]).unwrap();
        buf.downcast_handle_mut::<CpuBufferHandle>()
            .unwrap()
            .data_mut()
            .fill(3.0);
        let t = CpuMemory::new().download_owned(buf).unwrap();
        assert_eq!(t.as_slice().unwrap(), &[3.0f32; 16][..]);
        let again = mem.alloc_zeroed(&[4, 4]).unwrap();
        let t2 = mem.download(&again).unwrap();
        assert_eq!(t2.as_slice().unwrap(), &[0.0f32; 16][..]);
    }

    /// `take_data` は prepack キャッシュ破棄経路（`data_mut` と同じ）を通る。
    #[test]
    fn take_data_leaves_empty_shell() {
        let mem = CpuMemory::new();
        let mut buf = mem.alloc_zeroed(&[2, 2]).unwrap();
        let h = buf.downcast_handle_mut::<CpuBufferHandle>().unwrap();
        assert_eq!(h.take_data().len(), 4);
        assert!(h.data().is_empty());
    }
}
