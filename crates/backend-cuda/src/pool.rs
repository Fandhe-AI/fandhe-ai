//! CUDA 出力バッファのサイズクラス別プール（イシュー #1020・REQ-14）。
//!
//! `fandhe_ai_tensor_core::pool_core::SizeClassPool<H>`（ハンドル非依存の
//! プール本体。`crates/tensor-core/src/pool_core.rs`）を `H =
//! `[`CudaSliceHandle`]（`CudaSlice<f32>` を包む）として具体化し、
//! `gemm.rs`／`elementwise.rs`／`softmax.rs` の出力バッファ確保
//! （`stream.alloc_zeros::<f32>(numel)` の直接呼び出し）を置き換える。
//!
//! # 採用方式（`docs/backend-cuda-pool-allocator-decision.md` の要約）
//!
//! cudarc の stream-ordered 確保（driver プール。`cuMemAllocAsync`/
//! `cuMemFreeAsync`）は本モジュールでは**能動的に使わない**（既存の
//! `stream.alloc`/`stream.alloc_zeros` は同期 `cuMemAlloc` 系のまま）。
//! 代わりに自作の [`SizeClassPool`](fandhe_ai_tensor_core::pool_core::SizeClassPool)
//! （案 B）をアプリケーション層のキャッシュとして使い、[`CudaAllocator::
//! release_cached`] のフェーズ (iv) でのみ driver プール（存在する環境
//! では cudarc が内部的に `cuMemAllocAsync` を使いうる）を
//! `cuMemPoolTrimTo` でトリムする（driver 側が保持している分の解放を
//! 「面倒を見る」位置づけ。`has_async_alloc` が偽の環境ではこのフェーズを
//! スキップする）。
//!
//! # 単一ストリームモデル
//!
//! [`CudaAllocator`] は `(ordinal, 既定 stream)` 単位のプロセスワイド
//! singleton（`crate::context_cache::cached_allocator` 経由）。複数
//! CUDA ストリームをまたぐ貸し出しは対象外（イシュー #1012/#1013 の
//! 確定後にスコープ外事項として引き継ぐ）。
//!
//! # 対象は f32 出力バッファのみ（v1 スコープ）
//!
//! f16 出力（`gemm_wmma.rs`／`gemm_mma.rs`／`gemm_auto.rs` の
//! `alloc_zeros::<f16>`）・`MemoryOps`（`memory.rs::CudaMemory`）経由の
//! `DeviceBuffer` はスコープ外（`CudaBufferHandle.slice.len()==numel`
//! 前提の既存検証への影響範囲が広いため。out-of-scope-tracking.md 対象）。
//!
//! # `pool_core` API 統合（PR #1063 マージコンフリクト解消）
//!
//! 本モジュールは元々（#1020・PR #1061）独自の `pool_core` 実装
//! （`PoolConfig`／`Vec<H>` を返す `put`／引数無し `record_loan_end` 等）
//! と組んでいたが、#1021（Metal）が独立に実装した `pool_core.rs` との
//! main 統合時に、巨大帯クラス別上限・オクターブ境界切り上げ・再利用
//! 貸出時の waste 計上という 3 件の是正が反映済みの Metal 側 API
//! （[`SizeClassPoolConfig`]・`record_reuse`・`(logical_bytes,
//! class_bytes)` の引数順・`Vec<(u64, H)>` を返す `put`）へ合わせて
//! 書き換えた（詳細は `crates/tensor-core/src/pool_core.rs` モジュール
//! doc「#1020／#1021 統合時の経緯」参照）。

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut};

use fandhe_ai_tensor_core::pool_core::{
    PoolStats, SizeClassPool, SizeClassPoolConfig, size_class_for,
};

use crate::device::CudaDevice;
use crate::error::CudaError;

/// [`SizeClassPool`] が保持する CUDA 側ハンドル。`CudaSlice<f32>` は
/// cudarc 側で `unsafe impl Send + Sync` 済み（`driver/safe/core.rs`）の
/// ため、本ラッパーも追加の `unsafe` なしで `Send` になる
/// （`SizeClassPool<H: Send>` の境界を満たす）。
pub(crate) struct CudaSliceHandle(CudaSlice<f32>);

// `SizeClassPool<CudaSliceHandle>::put`／`release_cached` フェーズ (ii)
// が返す破棄対象は本ハンドルの `Drop`（cudarc 側の `cuMemFree` 相当）で
// 実解放される。CUDA は「即時返却」（`fandhe_ai_tensor_core::pool_core`
// モジュール冒頭「統計契約」の `pending_return_bytes` は常に 0）のため、
// `record_pending_return`／`put_merged`（旧 `record_pending_merge`。
// codex P2 最終指摘対応で本番呼び出し元ゼロとなり削除済み）のいずれも
// 呼ばない。

/// プールから貸し出された出力バッファの RAII ハンドル。
///
/// `handle` を `ManuallyDrop` で保持する設計は
/// `fandhe_ai_tensor_core::pool::PooledBufferHandle` と同型（`Drop::drop`
/// の 1 箇所でのみ所有権を取り出し、`as_view`/`as_view_mut` は借用のみで
/// 完結させることで borrow checker の E0499 を避ける）。`logical_numel`
/// は要求時の要素数（サイズクラス丸めによる `class_bytes` 側の余剰容量は
/// `as_view`/`as_view_mut` の対象範囲に含めない。呼び出し元
/// 〈`gemm.rs`／`elementwise.rs`／`softmax.rs`〉は常にこのビュー経由で
/// アクセスするため、余剰領域を誤って読み書きすることはない）。
pub(crate) struct PooledCudaHandle {
    handle: std::mem::ManuallyDrop<CudaSliceHandle>,
    class_bytes: u64,
    // `record_allocation`／`record_reuse` に渡した値と同一でなければ
    // ならない（`pool_core.rs` 契約。`record_loan_end` 呼び出し時に
    // 再計算せず、貸出開始時の値をそのまま保持する）。
    logical_bytes: u64,
    logical_numel: usize,
    pool: Arc<SizeClassPool<CudaSliceHandle>>,
}

impl PooledCudaHandle {
    /// カーネル起動の読み取り専用引数・D2H 転送（`clone_dtoh` 相当）に
    /// 使う論理長ビュー（`class_bytes` 側の余剰容量は含まない）。
    pub(crate) fn as_view(&self) -> CudaView<'_, f32> {
        self.handle.0.slice(0..self.logical_numel)
    }

    /// カーネル起動の書き込み引数に使う論理長の可変ビュー。
    pub(crate) fn as_view_mut(&mut self) -> CudaViewMut<'_, f32> {
        self.handle.0.slice_mut(0..self.logical_numel)
    }
}

impl Drop for PooledCudaHandle {
    /// 内部ハンドルをプールへ返却する（CUDA は即時返却。モジュール冒頭
    /// 「統計契約」参照）。`class_bytes == 0`（空バッファ。`numel == 0`
    /// 契約）はプールを介さず直接解放する
    /// （`fandhe_ai_tensor_core::pool` の「空テンソル契約」と同型）。
    fn drop(&mut self) {
        // SAFETY: `ManuallyDrop::take` は `Drop::drop` 内のこの 1 箇所
        // でのみ呼ばれ、`Drop::drop` は言語仕様上インスタンスごとに
        // ちょうど 1 回だけ呼ばれるため、`self.handle` への二重 take は
        // 構造的に発生しない（`as_view`/`as_view_mut` は参照を返すのみで
        // 所有権を奪わない。`fandhe_ai_tensor_core::pool::
        // PooledBufferHandle::drop` と同一根拠）。
        let inner = unsafe { std::mem::ManuallyDrop::take(&mut self.handle) };
        if self.class_bytes == 0 {
            drop(inner);
            return;
        }
        self.pool
            .record_loan_end(self.logical_bytes, self.class_bytes);
        // `put` が返す破棄対象（LRU 超過分）はロック解放後にここで drop
        // される（`pool_core.rs` モジュール冒頭「所有権契約 (b)」。ロック
        // 内で FFI 解放を伴いうる `Drop` を実行しない契約）。
        drop(self.pool.put(self.class_bytes, inner));
    }
}

/// 要求要素数 `numel` の `f32` バッファが消費するバイト数を検査付きで
/// 計算する（`fandhe_ai_tensor_core::pool::checked_byte_len` と同種の
/// OWASP A03 前段検証）。
fn checked_byte_len(numel: usize) -> Result<u64, CudaError> {
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("pool: allocation byte length overflows usize: numel={numel}"),
        })?;
    Ok(bytes as u64)
}

/// `ctx` が指すデバイスが CUDA driver 側のメモリプール
/// （`cuMemAllocAsync`/`cuMemFreeAsync` の基盤）をサポートするかを
/// `CudaContext::has_async_alloc()`（cudarc 0.19.8 公開 API。内部で
/// `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` を `CudaContext::new`
/// 構築時に 1 回だけクエリしキャッシュ済みの値を返す。
/// `cudarc-0.19.8/src/driver/safe/core.rs` の `has_async_alloc` フィールド・
/// 同名メソッド doc「Memory allocations performed through the default
/// CudaStream will use cuMemAllocAsync over cuMemAlloc if this method
/// returns true」参照）で判定する。実装開始時点では本メソッドの存在に
/// 気づかず独自実装を検討したが、cudarc 側に既に同等の判定 API が
/// 存在することを確認したため、それをそのまま使う
/// （`docs/backend-cuda-pool-allocator-decision.md` 参照）。
fn has_async_alloc(ctx: &CudaContext) -> bool {
    ctx.has_async_alloc()
}

/// [`CudaAllocator::release_cached`] のどのフェーズで失敗したかを表す
/// （`BackendError::DeviceAllocationFailed` の理由文字列へ埋め込む
/// フェーズ識別子。設計判断: 新しい `BackendError`/`CudaError` variant
/// は追加せず、既存 `DeviceAllocationFailed(String)` の文字列内で
/// フェーズを区別する。`crate::ops::CudaBackendOps::
/// release_cached_device_memory` が本型を消費して整形する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReleasePhase {
    /// (i) 破棄前の `stream.synchronize()`。
    PreFreeSync,
    /// (iii) 破棄後の `stream.synchronize()`。
    PostFreeSync,
    /// (iv) driver プールのトリム（`cuMemPoolTrimTo`）。
    DriverTrim,
}

// 注記: (ii)〈`take_one_for_release` ループでの実解放〉に対応する
// variant は持たない。CUDA の `CudaSlice::Drop` は `Result` を返さない
// （フォールブルな解放 API を持たない）ため、本バックエンドではこの
// フェーズが失敗しうる構造になっていない。フォールブルな明示解放 API を
// 持つバックエンド（`backend-metal`〈#1021〉等）を実装する際、実際に
// エラーを返しうることを確認したうえで variant を追加する
// （使われない speculative な variant を残さない。`.claude/rules/
// coding-rust.md` の `#[allow]` 安易追加禁止と同じ「未検証の拡張を
// 先回りしない」判断）。

impl ReleasePhase {
    fn as_str(self) -> &'static str {
        match self {
            ReleasePhase::PreFreeSync => "pre-free sync",
            ReleasePhase::PostFreeSync => "post-free sync",
            ReleasePhase::DriverTrim => "driver trim",
        }
    }
}

/// [`CudaAllocator::release_cached`] の失敗（フェーズ識別子付き）。
#[derive(Debug)]
pub(crate) struct ReleaseCacheError {
    pub(crate) phase: ReleasePhase,
    pub(crate) detail: String,
}

impl std::fmt::Display for ReleaseCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "phase={}: {}", self.phase.as_str(), self.detail)
    }
}

/// [`CudaAllocator::release_cached`] の実体。`sync_pre`／`sync_post`／
/// `trim` をクロージャとして注入することで、GPU 実機なしでも各フェーズの
/// 失敗時にフリーリスト状態・再試行対象が正しいことをテストできる
/// （`#[cfg(test)]` のフォールト注入テスト参照。design 実装計画
/// 「フォールト注入テスト」節）。
///
/// フェーズ: (i) `sync_pre` → (ii) `pool.take_one_for_release()` を
/// 空になるまでループし `record_release` してロック外で `drop`
/// （`H::Drop` は本関数内では失敗しない契約。CUDA では `ReleasePhase` に
/// このフェーズ専用の variant を持たない。上記 `ReleasePhase` 定義直後の
/// 注記参照）→ (iii) `sync_post` → (iv) `trim`。
fn release_cached_with<H: Send>(
    pool: &SizeClassPool<H>,
    mut sync_pre: impl FnMut() -> Result<(), String>,
    mut sync_post: impl FnMut() -> Result<(), String>,
    mut trim: impl FnMut() -> Result<(), String>,
) -> Result<u64, ReleaseCacheError> {
    sync_pre().map_err(|detail| ReleaseCacheError {
        phase: ReleasePhase::PreFreeSync,
        detail,
    })?;

    let mut freed_bytes = 0u64;
    while let Some((class_bytes, handle)) = pool.take_one_for_release() {
        pool.record_release(class_bytes);
        freed_bytes = freed_bytes.saturating_add(class_bytes);
        // ロックは `take_one_for_release` の呼び出し内でのみ保持され、
        // ここでの `drop`（実解放）はロック外（design §3.5「ロック粒度」）。
        drop(handle);
    }

    sync_post().map_err(|detail| ReleaseCacheError {
        phase: ReleasePhase::PostFreeSync,
        detail,
    })?;

    trim().map_err(|detail| ReleaseCacheError {
        phase: ReleasePhase::DriverTrim,
        detail,
    })?;

    Ok(freed_bytes)
}

/// `(ordinal, 既定 stream)` 単位のサイズクラス別プールアロケータ
/// （モジュール冒頭「単一ストリームモデル」参照）。
///
/// `crate::context_cache::cached_allocator` がプロセスワイドにキャッシュ
/// する。ただし `cached_gemm`／`cached_elementwise` 等（eternal に
/// `Arc` を保持する）とは異なり `Weak` 保持＋死んだエントリの刈り取り
/// 方式のため、本当の意味での singleton であり続けるのは「`CudaGemm`
/// 等のスイートが `allocator` フィールド経由で参照を保持し続ける限り」
/// （正準経路。`ops.rs` 経由の呼び出しでは事実上常にこれに該当する）
/// という条件付きである。全ハンドルが drop されればキャッシュからも
/// 消え、次回構築時は新しいインスタンスになる（`context_cache.rs`
/// モジュール冒頭「`cached_allocator` のみ Weak 参照＋刈り取り」参照。
/// codex-review 指摘。イシュー #1020 PR #1061）。
pub(crate) struct CudaAllocator {
    pool: Arc<SizeClassPool<CudaSliceHandle>>,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
}

impl CudaAllocator {
    /// `device` の既定ストリーム・コンテキストを共有した新規アロケータを
    /// 構築する（`SizeClassPoolConfig::default()`。既定 128MiB 上限は
    /// REQ-14 の係数上限〈2 倍以内〉を侵さない安全側の値。
    /// `pool_core.rs` 参照）。
    pub(crate) fn new(device: &CudaDevice) -> Self {
        Self {
            pool: Arc::new(SizeClassPool::new(SizeClassPoolConfig::default())),
            stream: Arc::clone(device.stream()),
            ctx: Arc::clone(device.context()),
        }
    }

    /// 構築時に共有した `CudaContext` を返す（`context_cache.rs` の
    /// `ContextKey` 回帰テスト専用のアクセサ。コード本体はこの
    /// アロケータが持つ `stream`/`ctx` の同一性を直接検証しないため、
    /// テスト以外の呼び出しは想定しない）。
    #[cfg(test)]
    pub(crate) fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// サイズクラス丸め後のクラスバイト数を計算する共通処理
    /// （`alloc_zeroed_f32`/`alloc_uninit_f32` 共用）。
    ///
    /// `size_class_for` は `bytes == 0` を `Ok(0)` として返す契約
    /// （`pool_core.rs` doc comment）のため、呼び出し元は戻り値の
    /// `class_bytes == 0` で「プール非経由（空バッファ）」を判定する。
    fn class_bytes_for(&self, numel: usize) -> Result<(u64, u64), CudaError> {
        let requested_bytes = checked_byte_len(numel)?;
        let class_bytes = size_class_for(requested_bytes, &self.pool.config()).map_err(|e| {
            CudaError::InvalidShape {
                detail: format!("pool: size_class_for failed: {e:?}"),
            }
        })?;
        Ok((requested_bytes, class_bytes))
    }

    /// `numel` 要素・ゼロ初期化済みの `f32` 出力バッファを確保する
    /// （`stream.alloc_zeros::<f32>(numel)` の直接呼び出しの置換）。
    ///
    /// プールヒット時は再利用バッファの論理範囲のみを `memset_zeros` で
    /// 上書きする（OWASP A02: 前利用データの残留を防ぐ。
    /// `fandhe_ai_tensor_core::pool::PoolZeroFill` と同じ契約）。ミス時は
    /// `stream.alloc_zeros`（クラス容量全体を確保・ゼロ初期化）で新規
    /// 確保する。OOM 時は [`Self::release_cached`] を 1 回試みてから
    /// 再試行し、それでも失敗すれば `Err` を返す（無限リトライしない。
    /// OWASP A04）。
    pub(crate) fn alloc_zeroed_f32(&self, numel: usize) -> Result<PooledCudaHandle, CudaError> {
        let (requested_bytes, class_bytes) = self.class_bytes_for(numel)?;
        if class_bytes == 0 {
            // numel == 0: プールを介さず直接確保する（空テンソル契約。
            // `PooledCudaHandle::drop` は `class_bytes == 0` を見て
            // プールへ返却せず直接解放する）。
            let raw = self.stream.alloc_zeros::<f32>(0)?;
            return Ok(self.wrap_handle(raw, 0, 0, 0));
        }
        let class_numel = (class_bytes / std::mem::size_of::<f32>() as u64) as usize;

        if let Some(mut handle) = self.pool.take(class_bytes) {
            self.pool.record_reuse(requested_bytes, class_bytes);
            // `record_reuse` 済みの貸出は、この `memset_zeros` が失敗すると
            // `PooledCudaHandle` が構築されず `Drop` 経由の `record_loan_end`
            // も走らないため、統計を明示的に巻き戻してからエラーを返す
            // （Cursor Bugbot Low 指摘。Metal 再利用経路の synchronize 失敗
            // 時の巻き戻し〈`backend-metal::pool`〉と同一契約。ハンドルは
            // drop してデバイスメモリを解放する）。
            if let Err(e) = self.stream.memset_zeros(&mut handle.0.slice_mut(0..numel)) {
                self.pool.record_loan_end(requested_bytes, class_bytes);
                return Err(e.into());
            }
            return Ok(self.wrap_handle_from(handle, class_bytes, requested_bytes, numel));
        }

        match self.stream.alloc_zeros::<f32>(class_numel) {
            Ok(raw) => {
                self.pool.record_allocation(requested_bytes, class_bytes);
                Ok(self.wrap_handle(raw, class_bytes, requested_bytes, numel))
            }
            Err(_) => {
                // OOM フォールバック: キャッシュ解放を 1 回試み再試行する
                // （OWASP A04。無限リトライはしない）。
                let _ = self.release_cached();
                let raw = self.stream.alloc_zeros::<f32>(class_numel)?;
                self.pool.record_allocation(requested_bytes, class_bytes);
                Ok(self.wrap_handle(raw, class_bytes, requested_bytes, numel))
            }
        }
    }

    /// `numel` 要素・未初期化の `f32` 出力バッファを確保する。
    ///
    /// **呼び出し元は、対象カーネルが確保した全要素（`0..numel`）を
    /// 必ず書き切ることをカーネルソースで確認済みでなければならない**
    /// （`docs/backend-cuda-pool-allocator-decision.md` §「`alloc_uninit`
    /// の適用」・OWASP A02。確認できない場合は
    /// [`Self::alloc_zeroed_f32`] を使う）。プールヒット時は前利用データを
    /// そのまま返す（ゼロクリアしない。これが `alloc_zeroed_f32` との
    /// 唯一の差）。ミス時は `stream.alloc`（`unsafe`。境界チェックなしの
    /// 生確保）で新規確保する。
    pub(crate) fn alloc_uninit_f32(&self, numel: usize) -> Result<PooledCudaHandle, CudaError> {
        let (requested_bytes, class_bytes) = self.class_bytes_for(numel)?;
        if class_bytes == 0 {
            let raw = self.stream.alloc_zeros::<f32>(0)?;
            return Ok(self.wrap_handle(raw, 0, 0, 0));
        }
        let class_numel = (class_bytes / std::mem::size_of::<f32>() as u64) as usize;

        if let Some(handle) = self.pool.take(class_bytes) {
            self.pool.record_reuse(requested_bytes, class_bytes);
            return Ok(self.wrap_handle_from(handle, class_bytes, requested_bytes, numel));
        }

        // SAFETY: `alloc::<f32>` は指定要素数分のデバイスメモリを未初期化
        // のまま確保する（`cuMemAlloc` 相当）。呼び出し元契約（本メソッド
        // ドキュメンテーションコメント）により、返却するビュー
        // （`0..numel`）は起動するカーネルが必ず全要素へ書き込むため、
        // 未初期化領域が読み出される経路は生じない。`numel <= class_numel`
        // （サイズクラス丸めは要求量以上にしか切り上げない。
        // `pool_core::size_class_for` 契約）により `0..numel` は確保済み
        // 範囲内に収まる。
        let alloc_result = unsafe { self.stream.alloc::<f32>(class_numel) };
        match alloc_result {
            Ok(raw) => {
                self.pool.record_allocation(requested_bytes, class_bytes);
                Ok(self.wrap_handle(raw, class_bytes, requested_bytes, numel))
            }
            Err(_) => {
                let _ = self.release_cached();
                // SAFETY: 上記と同一根拠。
                let raw = unsafe { self.stream.alloc::<f32>(class_numel) }?;
                self.pool.record_allocation(requested_bytes, class_bytes);
                Ok(self.wrap_handle(raw, class_bytes, requested_bytes, numel))
            }
        }
    }

    fn wrap_handle(
        &self,
        raw: CudaSlice<f32>,
        class_bytes: u64,
        logical_bytes: u64,
        logical_numel: usize,
    ) -> PooledCudaHandle {
        self.wrap_handle_from(
            CudaSliceHandle(raw),
            class_bytes,
            logical_bytes,
            logical_numel,
        )
    }

    fn wrap_handle_from(
        &self,
        handle: CudaSliceHandle,
        class_bytes: u64,
        logical_bytes: u64,
        logical_numel: usize,
    ) -> PooledCudaHandle {
        PooledCudaHandle {
            handle: std::mem::ManuallyDrop::new(handle),
            class_bytes,
            logical_bytes,
            logical_numel,
            pool: Arc::clone(&self.pool),
        }
    }

    /// プールがアイドル保持している分を即座に実解放する
    /// （[`fandhe_ai_tensor_core::BackendOps::release_cached_device_memory`]
    /// の CUDA 実装本体。`crate::ops::CudaBackendOps::
    /// release_cached_device_memory` の唯一の呼び出し先）。
    ///
    /// フェーズ (i)〜(iv) は [`release_cached_with`] 参照。フェーズ (iv)
    /// は `has_async_alloc(&self.ctx)` が真の環境でのみ実行する
    /// （偽の環境では driver プール自体が存在しないため `cuDeviceGetMemPool`
    /// を呼ばずスキップして `Ok`。`docs/backend-cuda-pool-allocator-decision.md`
    /// 参照）。
    pub(crate) fn release_cached(&self) -> Result<u64, ReleaseCacheError> {
        let stream_pre = Arc::clone(&self.stream);
        let stream_post = Arc::clone(&self.stream);
        let ctx = Arc::clone(&self.ctx);
        release_cached_with(
            &self.pool,
            move || stream_pre.synchronize().map_err(|e| format!("{e:?}")),
            move || stream_post.synchronize().map_err(|e| format!("{e:?}")),
            move || {
                if !has_async_alloc(&ctx) {
                    return Ok(());
                }
                // SAFETY: `ctx.cu_device()` は `CudaContext::new` が既に
                // 構築済みの有効なデバイスハンドルを返す（cudarc の
                // `get_mem_pool`/`trim_to` はいずれも「`get` から得た
                // device／有効な pool」を事前条件とする。本関数は同一
                // `ctx` から得た device を `get_mem_pool` に渡し、その
                // 戻り値をそのまま `trim_to` に渡すため事前条件を満たす）。
                unsafe {
                    let mem_pool = cudarc::driver::result::device::get_mem_pool(ctx.cu_device())
                        .map_err(|e| format!("{e:?}"))?;
                    cudarc::driver::result::mem_pool::trim_to(mem_pool, 0)
                        .map_err(|e| format!("{e:?}"))?;
                }
                Ok(())
            },
        )
    }

    /// 現在のプール利用状況（[`PoolStats`]）を返す。
    pub(crate) fn stats(&self) -> PoolStats {
        self.pool.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // release_cached_with は GPU 非依存の純粋なロジックのため、実 CUDA
    // 型を要求しない汎用 `H`（ここでは `u32`）でフォールト注入する
    // （`context_cache.rs::tests::fresh_cache` と同じ「GPU 不要ロジックは
    // 実カーネル型に依存しない形でテストする」方針）。

    fn cfg() -> SizeClassPoolConfig {
        SizeClassPoolConfig::default()
    }

    #[test]
    fn release_cached_with_drains_pool_and_reports_freed_bytes() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        drop(pool.put(256, 1));
        drop(pool.put(512, 2));

        let freed =
            release_cached_with(&pool, || Ok(()), || Ok(()), || Ok(())).expect("release succeeds");

        assert_eq!(freed, 256 + 512);
        assert_eq!(pool.stats().cached_bytes, 0);
        assert_eq!(pool.stats().released_bytes, 256 + 512);
    }

    #[test]
    fn release_cached_with_pre_free_sync_failure_leaves_pool_untouched() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        drop(pool.put(256, 1));

        let err = release_cached_with(
            &pool,
            || Err("pre-free sync boom".to_string()),
            || Ok(()),
            || Ok(()),
        )
        .expect_err("pre-free sync failure propagates");

        assert_eq!(err.phase, ReleasePhase::PreFreeSync);
        // pre-free sync 失敗時はプールへ手を付けていない
        // （フリーリストが手つかずのまま残ることを検証）。
        assert_eq!(pool.stats().cached_bytes, 256);
        assert_eq!(pool.stats().released_bytes, 0);
    }

    #[test]
    fn release_cached_with_post_free_sync_failure_still_drains_and_records() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        drop(pool.put(256, 1));

        let err = release_cached_with(
            &pool,
            || Ok(()),
            || Err("post-free sync boom".to_string()),
            || Ok(()),
        )
        .expect_err("post-free sync failure propagates");

        assert_eq!(err.phase, ReleasePhase::PostFreeSync);
        // (ii) の解放自体は (iii) の前に完了しているため、実解放・統計
        // 更新は post-free sync の失敗に関わらず反映済みである。
        assert_eq!(pool.stats().cached_bytes, 0);
        assert_eq!(pool.stats().released_bytes, 256);
    }

    #[test]
    fn release_cached_with_driver_trim_failure_still_drains_and_records() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        drop(pool.put(256, 1));
        drop(pool.put(512, 2));

        let err = release_cached_with(
            &pool,
            || Ok(()),
            || Ok(()),
            || Err("driver trim boom".to_string()),
        )
        .expect_err("driver trim failure propagates");

        assert_eq!(err.phase, ReleasePhase::DriverTrim);
        // (ii)(iii) は成功済み: フリーリストは空・released_bytes は加算済み。
        assert_eq!(pool.stats().cached_bytes, 0);
        assert_eq!(pool.stats().released_bytes, 256 + 512);
    }

    #[test]
    fn release_cached_with_empty_pool_reports_zero_freed_bytes() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let freed = release_cached_with(&pool, || Ok(()), || Ok(()), || Ok(()))
            .expect("release succeeds on empty pool");
        assert_eq!(freed, 0);
    }

    #[test]
    fn release_phase_display_matches_identifier() {
        let err = ReleaseCacheError {
            phase: ReleasePhase::DriverTrim,
            detail: "boom".to_string(),
        };
        let rendered = err.to_string();
        assert!(rendered.contains("driver trim"));
        assert!(rendered.contains("boom"));
    }
}
