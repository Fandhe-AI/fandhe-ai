//! Metal のサイズクラス別プールアロケータ（イシュー #1021。設計は
//! `docs/device-memory-pool-design.md` §3.1〜3.6）。
//!
//! `fandhe_ai_tensor_core::pool_core::SizeClassPool<H>`（ハンドル型非依存の
//! 汎用プール本体）を、Metal 固有の生ハンドル型（[`RawMetalBuffer`]）で
//! 具体化する。本モジュールは `tensor-core` のどの公開 trait
//! （`BackendOps`／`MemoryOps`）にも属さない `pub(crate)` 面であり、
//! `crate::buffer::MetalBuffer`（`alloc_zeroed_pooled`／
//! `alloc_uninit_pooled`）からのみ到達する（設計文書 §3.1「各バックエンド
//! クレート内に閉じる `pub(crate)` 面」）。
//!
//! # 命名の差異（イシュー #1021 実装確定・PR 本文に記録）
//!
//! 設計文書は例示名 `MetalBufferHandle` を使うが、`memory.rs` の既存
//! `pub(crate) struct MetalBufferHandle`（`BufferHandle` 実装。
//! `DeviceBuffer` の中身）と衝突するため、本実装は生ハンドル型を
//! [`RawMetalBuffer`] と命名する（#1020 側にも共有する命名）。
//!
//! # Metal の GPU 完了待ち返却（設計文書 §3.3「Metal」）
//!
//! [`PooledMetalHandle::drop`] は `crate::context::MetalContext::
//! defer_pool_return` へ委譲する。同メソッドが `BatchSlots` の同一ロック
//! 区間内で `open`／`committed` バッチの有無を検査し、in-flight であれば
//! `crate::pool_pending::PendingReturns` へ返却を委譲する（即座に
//! `SizeClassPool::put` しない）。判定ロジック自体は `pool_pending.rs`
//! （`objc2` 系 FFI に触れない純粋ロジック。Linux でも単体テストが回る）
//! に切り出し済みであり、本モジュールは配線のみを担う。

use std::sync::Arc;

use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use fandhe_ai_tensor_core::memory_stats::{AllocationTracker, TrackedAllocation};
use fandhe_ai_tensor_core::pool_core::{
    PoolStats, SizeClassPool, SizeClassPoolConfig, size_class_for,
};

use crate::buffer::MtlBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pool_pending::PendingReturn;

/// [`SizeClassPool`] のフリーリストに格納される Metal の生ハンドル型
/// （設計文書 §3.1「各バックエンドクレート内に閉じる `pub(crate)` 面」の
/// 「生ハンドル型」）。`tensor-core` のどの公開シグネチャにも現れない。
///
/// `capacity_bytes`（サイズクラス丸め後の実確保量）は
/// `crate::buffer::MetalBuffer` が保持する論理長 `len` とは別物であり、
/// [`crate::buffer::MetalBuffer::raw`]／`read_to_vec`／`zero_fill` は
/// 常に論理長 `len` のみを使う契約を維持する（設計文書 §3.1「capacity と
/// 論理長の分離」。`buffer.rs` 側は変更しない）。
///
/// `_alloc`（[`TrackedAllocation`]）は本ハンドルの**物理確保**の生存期間
/// （フリーリストで保持中も含む）を計測する。`memory.rs::MetalBufferHandle`
/// 側の `_alloc` はプール経路では `TrackedAllocation::new(tracker, 0)`
/// （二重計上防止。`memory.rs::alloc_zeroed_inner` コメント参照）にする
/// ため、実バイト数の唯一の計測系列は本フィールドが指す
/// [`MetalAllocator::tracker`] になる。
pub(crate) struct RawMetalBuffer {
    buffer: Retained<MtlBuffer>,
    capacity_bytes: u64,
    _alloc: TrackedAllocation,
}

impl RawMetalBuffer {
    fn raw(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// サイズクラス丸め後の実確保バイト数（`crate::buffer::MetalBuffer`
    /// 側の論理長 `len` の上限チェック用。設計文書 §3.1「capacity と
    /// 論理長の分離」）。
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// 論理要素数 `logical_len`（capacity 以下であることは呼び出し元
    /// `MetalAllocator::alloc_inner` が保証する）分だけをゼロで上書き
    /// する（`buffer.rs::MetalBuffer::zero_fill` の capacity 対応版。
    /// 再利用ヒット時の「全要素 0」契約再適用に使う。設計文書 §6「A02」
    /// 参照: 再利用バッファに前利用者のデータが残留するのを防ぐ）。
    ///
    /// # Safety 境界
    /// `contents()` は `StorageModeShared` バッファの CPU 可視アドレスを
    /// 返す。書き込む要素数 `logical_len` は `capacity_bytes /
    /// size_of::<f32>()` 以下であることを呼び出し元が保証する
    /// （`MetalAllocator::alloc_inner` が `class_bytes >= logical_bytes`
    /// を `size_class_for` の丸め規則により常に満たす）ため、確保
    /// バイト数を超えて書くことはない。
    fn zero_fill_logical(&self, logical_len: usize) {
        let ptr = self.buffer.contents();
        // SAFETY: 上記コメント参照。
        let slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut f32, logical_len) };
        slice.fill(0.0);
    }
}

// SAFETY: 設計文書 §3.5「`Send`/`Sync` 方針の更新」は「Metal の
// `MTLBuffer` protocol も objc2-metal 0.3.2 で `Send + Sync` を
// supertrait に持つ」と記述しているが、実装時に本クレートが依存する
// objc2-metal 0.3.2 の生成コード（`MTLBuffer: MTLResource` →
// `MTLResource: MTLAllocation` のいずれも `Send`/`Sync` を課さない。
// `MTLDevice: NSObjectProtocol + Send + Sync` とは異なる）を実測した
// ところ、この記述は実態と異なることが判明した（イシュー #1021 実装
// 確定・PR 本文と `docs/device-memory-pool-design.md` §9 実装記録に
// 記録する）。よって `context.rs::Batch` と同じ正当化ロジックで
// `unsafe impl Send` を付与する: `RawMetalBuffer` へのアクセスは
// (a) `SizeClassPool<H>` のフリーリスト内（`Mutex<PoolCore<H>>` が
// 直列化）、(b) `PooledMetalHandle`（`Drop` まで単一スレッドが排他
// 所有し他スレッドと共有されない）、(c) `BatchSlots::
// pending_pool_returns`（`Batch` と同じ `Mutex<BatchSlots>` が直列化）
// のいずれかの経路に限られ、複数スレッドから同時にアクセスされること
// はない。よって `Send` の付与は安全（`Sync` は付与しない:
// `SizeClassPool<H>: Send + Sync where H: Send` の成立に `H: Sync` は
// 不要であり、`RawMetalBuffer` 自体を複数スレッドから同時に `&` 参照
// する経路は設計上存在しない）。
unsafe impl Send for RawMetalBuffer {}

/// 貸出中のハンドルを RAII で管理するラッパー型（設計文書 §3.1「RAII
/// 貸出ラッパー型」）。`crate::buffer::MetalBuffer::Backing::Pooled` が
/// 保持し、`raw()`／`len()`／`read_to_vec()`／`zero_fill()` の実体は
/// `MetalBuffer` 側が論理長 `len` を使って提供する（本型自体は生バッファ
/// への到達経路のみを提供する）。
///
/// 二重返却・use-after-return の構造的防止は `Option<H>::take()` 方式
/// （設計文書 §3.1）: `Drop` は必ず `self.handle.take()` で所有権を奪って
/// から後段処理へ渡す。
pub(crate) struct PooledMetalHandle {
    handle: Option<RawMetalBuffer>,
    class_bytes: u64,
    logical_bytes: u64,
    pool: Arc<SizeClassPool<RawMetalBuffer>>,
    ctx: Arc<MetalContext>,
}

impl PooledMetalHandle {
    /// `crate::buffer::MetalBuffer::raw`（`Backing::Pooled` 分岐）から
    /// 呼ばれる。`handle` は `Drop` の `take()` 実行中の一瞬を除き常に
    /// `Some` であり（本型が生存している間はその不変条件が構造的に
    /// 保たれる。`Drop` 中は他コードから本メソッドが呼ばれることは
    /// あり得ない）、到達不能パスは `unreachable!()`（本番経路で
    /// panic しない `.claude/rules/coding-rust.md` の対象は
    /// `unwrap`/`expect` によるフォールブル値の握り潰しであり、構造的に
    /// 到達しない不変条件違反の検出は本クレート `gemm.rs::pipeline_for`
    /// と同じ扱い）。
    pub(crate) fn raw(&self) -> &MtlBuffer {
        match &self.handle {
            Some(raw) => raw.raw(),
            None => unreachable!(
                "PooledMetalHandle::raw: called while the handle was taken during Drop"
            ),
        }
    }

    /// `crate::buffer::MetalBuffer` が構築直後に「論理長が capacity を
    /// 超えていない」不変条件を `debug_assert!` で固定するためのアクセサ
    /// （設計文書 §3.1「capacity と論理長の分離」）。
    pub(crate) fn capacity_bytes(&self) -> u64 {
        match &self.handle {
            Some(raw) => raw.capacity_bytes(),
            None => unreachable!(
                "PooledMetalHandle::capacity_bytes: called while the handle was taken during Drop"
            ),
        }
    }
}

impl Drop for PooledMetalHandle {
    /// 設計文書 §3.1「RAII 貸出ラッパー型」・§3.3「Metal」。
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        // `record_loan_end` は `put()` の呼び出しタイミングとは独立に、
        // `Drop` の時点で常に即座に呼ぶ（CUDA・Metal 共通。設計文書
        // §3.1「フィールド更新契約」`capacity_waste_bytes` 行）。
        self.pool
            .record_loan_end(self.logical_bytes, self.class_bytes);
        self.ctx.defer_pool_return(PendingReturn {
            class_bytes: self.class_bytes,
            handle,
            pool: Arc::clone(&self.pool),
        });
    }
}

/// `f32` 要素数から論理バイト数を検査付きで算出する（`buffer.rs::
/// checked_byte_len` と同じ意図の防御だが、本モジュールは `len == 0` を
/// 拒否しない: 呼び出し元 `buffer.rs::alloc_zeroed_pooled`／
/// `alloc_uninit_pooled` が `len == 0` の拒否を既に担っているため
/// 二重検証にせず、ここではオーバーフロー検査のみ行う）。
fn logical_bytes_for(len: usize) -> Result<u64, MetalError> {
    len.checked_mul(std::mem::size_of::<f32>())
        .map(|bytes| bytes as u64)
        .ok_or(MetalError::AllocationSizeOverflow { len })
}

/// device 単位のプロセスワイド singleton アロケータ本体（設計文書
/// §3.1「プールは device 単位のプロセスワイド singleton とする」）。
/// `crate::context_cache::cached_allocator` から取得する。
pub(crate) struct MetalAllocator {
    context: Arc<MetalContext>,
    pool: Arc<SizeClassPool<RawMetalBuffer>>,
    tracker: Arc<AllocationTracker>,
}

impl MetalAllocator {
    pub(crate) fn new(context: Arc<MetalContext>) -> Self {
        Self {
            context,
            pool: Arc::new(SizeClassPool::new(SizeClassPoolConfig::default())),
            tracker: Arc::new(AllocationTracker::new()),
        }
    }

    /// `len` 要素分（ゼロ初期化）の Metal バッファを確保する
    /// （`crate::buffer::MetalBuffer::alloc_zeroed_pooled` から呼ばれる。
    /// `MetalBuffer::new_zeroed` の pooled 版）。
    pub(crate) fn alloc_zeroed(&self, len: usize) -> Result<PooledMetalHandle, MetalError> {
        self.alloc_inner(len, true)
    }

    /// `len` 要素分（未初期化）の Metal バッファを確保する
    /// （`crate::buffer::MetalBuffer::alloc_uninit_pooled` から呼ばれる。
    /// カーネルが全要素を書き切る出力専用の確保に限定して使う契約は
    /// 呼び出し元〈`gemm.rs`〉のコメントが担う。設計文書 §6「A02」）。
    pub(crate) fn alloc_uninit(&self, len: usize) -> Result<PooledMetalHandle, MetalError> {
        self.alloc_inner(len, false)
    }

    fn alloc_inner(
        &self,
        len: usize,
        zero_on_reuse: bool,
    ) -> Result<PooledMetalHandle, MetalError> {
        let logical_bytes = logical_bytes_for(len)?;
        let cfg = self.pool.config();
        let class_bytes = size_class_for(logical_bytes, &cfg)
            .map_err(|_| MetalError::AllocationSizeOverflow { len })?;

        if let Some(raw) = self.pool.take(class_bytes) {
            // 再利用貸出の統計記帳（codex P2・Cursor Bugbot 指摘対応。
            // `pool_core.rs::SizeClassPool::take` の doc comment が定める
            // 契約: `Some` を受け取った直後に必ず `record_reuse` を呼び、
            // `reuse_count`／`capacity_waste_bytes`（貸出中ストック量）を
            // まとめて更新する。新規確保経路の `record_allocation`
            // 〈下記 `wrap_fresh`〉と対になる。これを怠ると `Drop` 時の
            // `record_loan_end` の減算に対応する加算が存在せず、
            // `capacity_waste_bytes` が過小表示・`debug_assert!` 失敗に
            // なる）。
            self.pool.record_reuse(logical_bytes, class_bytes);
            if zero_on_reuse {
                // 再利用時のゼロ初期化契約（設計文書 §3.3「ゼロ初期化は
                // ホスト書き込み」・§6「A02」）: 前利用者のバイト残留を
                // 防ぐため、`synchronize()` で GPU 完了を待ってから
                // `zero_fill_logical` する。フレッシュ確保
                // （`newBufferWithLength_options` 直後）はこの限りでなく
                // `buffer.rs::MetalBuffer::new_zeroed` と同じ「OS 新規
                // ページはゼロ初期化済み」という既存の暗黙契約に乗る
                // （本関数はフレッシュ経路〈下記 `alloc_fresh`〉で明示
                // ゼロクリアを行わない）。
                self.context.synchronize()?;
                raw.zero_fill_logical(len);
            }
            return Ok(PooledMetalHandle {
                handle: Some(raw),
                class_bytes,
                logical_bytes,
                pool: Arc::clone(&self.pool),
                ctx: Arc::clone(&self.context),
            });
        }

        match self.alloc_fresh(class_bytes, logical_bytes) {
            Ok(raw) => Ok(self.wrap_fresh(raw, class_bytes, logical_bytes)),
            Err(MetalError::BufferAllocation { .. }) => {
                // OOM フォールバック（設計文書 §3.4）: `release_cached()`
                // を 1 回実行してから再試行する。それでも失敗すれば
                // fail-closed に `Err` を返す（無限リトライしない）。
                self.release_cached()?;
                let raw = self.alloc_fresh(class_bytes, logical_bytes)?;
                Ok(self.wrap_fresh(raw, class_bytes, logical_bytes))
            }
            Err(other) => Err(other),
        }
    }

    fn wrap_fresh(
        &self,
        raw: RawMetalBuffer,
        class_bytes: u64,
        logical_bytes: u64,
    ) -> PooledMetalHandle {
        self.pool.record_allocation(logical_bytes, class_bytes);
        PooledMetalHandle {
            handle: Some(raw),
            class_bytes,
            logical_bytes,
            pool: Arc::clone(&self.pool),
            ctx: Arc::clone(&self.context),
        }
    }

    /// 新規物理確保（`newBufferWithLength_options`）を行う。統計
    /// （`record_allocation`）は成功後に呼び出し元が更新する（確保が
    /// 失敗しうる箇所より後で計上する順序契約は `memory.rs::
    /// alloc_zeroed_inner` と同じ）。
    fn alloc_fresh(
        &self,
        class_bytes: u64,
        logical_bytes: u64,
    ) -> Result<RawMetalBuffer, MetalError> {
        let capacity = class_bytes as usize;
        let buffer = self
            .context
            .device()
            .newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocation { bytes: capacity })?;
        let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), class_bytes);
        debug_assert!(class_bytes >= logical_bytes);
        Ok(RawMetalBuffer {
            buffer,
            capacity_bytes: class_bytes,
            _alloc: alloc,
        })
    }

    /// REQ-14 の解放 API 本体（内部メソッド。公開 `BackendOps::
    /// release_cached_device_memory()` から `crate::ops` 経由で呼ばれる。
    /// 設計文書 §3.6 (2)「バックエンド別の該当フェーズ」表の Metal 行）。
    ///
    /// フェーズ (i): `MetalContext::synchronize()`（`pending_pool_
    /// returns` の全件合流を含む。成否に関わらず合流する契約は
    /// `context.rs::synchronize` 側が担う）。失敗時はここで打ち切り、
    /// `take_one_for_release` を一切呼ばない（フリーリストへ触れない。
    /// 設計文書 §3.6 (2)「フェーズ (i) 失敗」）。
    ///
    /// フェーズ (ii): フリーリストを走査し `take_one_for_release` で
    /// 1 件ずつ取り出して drop する（`Retained` の ObjC 参照カウント
    /// 減算。解放 FFI が存在せず失敗しない設計。設計文書の表「Metal」
    /// 列）。Metal には driver トリム相当の機構がないためフェーズ
    /// (iii)/(iv) は存在しない（同表参照）。
    pub(crate) fn release_cached(&self) -> Result<u64, MetalError> {
        self.context.synchronize()?;

        let mut released = 0u64;
        while let Some((class_bytes, handle)) = self.pool.take_one_for_release() {
            drop(handle);
            self.pool.record_release(class_bytes);
            released = released.saturating_add(class_bytes);
        }
        Ok(released)
    }

    /// 統計スナップショット（`crate::ops::MetalBackendOps::
    /// device_memory_pool_stats` から呼ばれる）。
    pub(crate) fn stats(&self) -> PoolStats {
        self.pool.stats()
    }
}

#[cfg(test)]
mod tests {
    // Metal 実機（`MetalContext::new`）が必要なテストは
    // `tests/pool_real_device.rs`（`#[ignore]`）へ分離する（本モジュールの
    // 純粋ロジック部分〈丸め・所有権遷移〉は `fandhe_ai_tensor_core::
    // pool_core` 側と `pool_pending.rs`（Linux 実行可能）が既にカバー
    // 済みのため、ここでは `MetalAllocator` 固有の配線
    // 〈`logical_bytes_for`〉のみを Linux で検証する）。
    use super::*;

    #[test]
    fn logical_bytes_for_computes_f32_byte_length() {
        assert_eq!(logical_bytes_for(1024).unwrap(), 4096);
        assert_eq!(logical_bytes_for(0).unwrap(), 0);
    }

    #[test]
    fn logical_bytes_for_rejects_overflow() {
        let err = logical_bytes_for(usize::MAX).unwrap_err();
        assert!(matches!(err, MetalError::AllocationSizeOverflow { .. }));
    }
}
