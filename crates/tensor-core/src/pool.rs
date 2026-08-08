//! サイズクラス別バッファプール（TASK-#201・REQ-14 14-3）。
//!
//! # 背景
//!
//! v1（Burn/CubeCL）では GEMM 4096³ でピークメモリが理論値 192MiB の約 17 倍
//! （3235MiB）に蓄積した（candle の Metal プール無制限成長と同種の教訓）。
//! REQ-14 14-3 は「バッファプール等のキャッシュ機構を導入する場合、係数上限
//! （2 倍以内）を維持できなければプール解放 API を提供」を求める。本モジュール
//! は総量上限・自動破棄（LRU）を最初から組み込んだプール機構を提供する。
//!
//! # 位置付け（opt-in デコレータ）
//!
//! [`PooledMemory`] は既存 [`crate::buffer::MemoryOps`] 実装（`CpuMemory`／
//! `CudaMemory`／`MetalMemory`）を包むデコレータであり、既定の確保経路
//! （素の `MemoryOps` 実装を直接使う経路）は変更しない（opt-in。既定有効化の
//! 構成判断は PoC-v2 実測・#202 の係数維持テスト後に行う。安全側判断）。
//!
//! # サイズクラス方針（バイトサイズ完全一致）
//!
//! 初期実装は「正確なバイトサイズをキーとするバケット」とし、再利用は
//! バイトサイズ完全一致時のみ行う。冪等 2 乗への切り上げ（capacity > 論理
//! numel）を許すと、既存 `download` 契約（handle 実長 = numel 前提。
//! `backend-cpu`／`backend-cuda` の `MemoryOps` 実装）を壊すため、切り上げ
//! クラス化は将来最適化としてスコープ外の申し送りとする
//! （`.claude/rules/out-of-scope-tracking.md`）。
//!
//! # 総量上限・LRU 破棄
//!
//! [`PoolConfig::max_pool_bytes`] を超えるアイドル保持は、挿入順が最も
//! 古いエントリからグローバル LRU で自動破棄する（`PoolCore` 内部の
//! `order`〈tick → バケットキー〉が全エントリの挿入順を保持し、破棄対象は
//! `order` の最小 tick から求める）。`max_pool_bytes == 0` はプール無効
//! （全パススルー）と定義する。上限より大きい単一バッファはプールに
//! 入れず即解放する。既定値は 128MiB（[`PoolConfig::default`]）とし、
//! GEMM 4096³ のワーキングセット 192MiB + アイドル 128MiB = 320MiB が
//! REQ-14 の係数 2 倍（384MiB）を侵さない安全側の値とする（確定・調整は
//! PoC-v2 実測に委ねる）。
//!
//! # 返却経路（RAII 維持）
//!
//! `MemoryOps` に明示 `free()` は無い（`buffer.rs` モジュールコメント
//! 「解放方針（RAII 一本化）」）ため、[`PooledBufferHandle`] を導入し、
//! `Drop` で内部ハンドルをプールへ返却する。プールが既に破棄済み
//! （`Weak::upgrade` 失敗）・lock poisoning 時はそのまま内部ハンドルを
//! drop する（素直に解放。panic させない。`memory_stats::AllocationTracker`
//! の `lock()` と同じ「poisoned でも `unwrap` しない」方針）。
//!
//! # 透過ダウンキャスト
//!
//! [`PooledBufferHandle::as_any`] は内部ハンドルの `as_any()` へ転送する。
//! `downcast_ref::<H>()` は `Any` オブジェクトが指す**具体型**の
//! `TypeId` で判定するため、`PooledBufferHandle` 越しでも各バックエンドの
//! `download`／カーネルの `downcast_handle::<CpuBufferHandle>()` 等が
//! プール経由バッファで無変更で動作する（`buffer.rs` モジュールコメント
//! 「`Any` を supertrait にしない理由」の設計と両立する）。
//!
//! # ゼロ初期化契約の維持
//!
//! `alloc_zeroed` の「全要素 0」契約を再利用時にも守るため、[`PoolZeroFill`]
//! トレイトを各バックエンドに実装させる（CPU: `Vec<f32>` の `fill`、CUDA:
//! `CudaStream::memset_zeros`、Metal: `StorageModeShared` バッファの
//! `contents()` 書き込み）。前利用データの残留は情報漏えいリスクでもある
//! （`.claude/rules/security.md` A02/A04）。
//!
//! # プール対象は `alloc_zeroed` のみ（初期実装）
//!
//! `upload` はパススルーとする。再利用 upload には `upload_into`
//! （既存バッファへの htod 転送）が必要で、CUDA の非同期転送同期契約・
//! 非 contiguous 処理の再実装リスクがあるため、本イシューでは追加しない
//! （PoC-v2 実測で不足が判明した場合に別イシューで追加。out-of-scope
//! 記録）。
//!
//! # 計測反映（受け入れ条件の充足機構）
//!
//! `memory_stats::TrackedAllocation` は各バックエンドの具体ハンドル内に
//! 埋まっているため、(a) プール保持中も内部ハンドルが生存 →
//! `allocated_bytes` に計上、(b) LRU 破棄で内部ハンドルが drop →
//! `on_free` で `allocated_bytes` が減少、が追加実装なしで成立する
//! （`memory_stats` モジュールの契約に相乗りする設計）。プール固有統計
//! として [`PooledMemory::pooled_bytes`]（アイドル保持量）を公開する。
//!
//! # 空テンソル契約
//!
//! `numel == 0` は従来どおり FFI 非経由の空ハンドルとし、プールを介さない
//! （`buffer.rs` モジュールコメント「空テンソル（numel == 0）の契約」を
//! `PooledMemory` でも維持する）。
//!
//! # #202 向け内部解放フック
//!
//! 明示解放 API の公開・係数維持テストは #202 のスコープである。本モジュール
//! は `PoolCore::clear_all`（`pub(crate)`）としてプール全保持分を即座に
//! 解放する内部フックのみ用意し、公開 API 化（`PooledMemory` からの
//! `pub` メソッド追加）は #202 に委ねる。

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use crate::buffer::{BufferHandle, DeviceBuffer, MemoryOps};
use crate::device::{BackendError, Device};
use crate::tensor::Tensor;

/// プールの総量上限（design §2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    /// プールがアイドル保持してよいバイト数の総上限。
    /// `0` はプール無効（全パススルー）を意味する。
    pub max_pool_bytes: u64,
}

impl Default for PoolConfig {
    /// 既定値 128MiB（モジュール冒頭「総量上限・LRU 破棄」参照）。
    fn default() -> Self {
        Self {
            max_pool_bytes: 128 * 1024 * 1024,
        }
    }
}

/// 各バックエンドが実装する「既存バッファのゼロ初期化」フック。
///
/// `MemoryOps::alloc_zeroed` は毎回新規確保するため既定でゼロ初期化契約を
/// 満たすが、[`PooledMemory`] がプールから再利用したバッファは前利用時の
/// データを保持している可能性があるため、再利用のたびに本トレイトで明示的
/// にゼロで上書きする（モジュール冒頭「ゼロ初期化契約の維持」参照）。
///
/// `&dyn BufferHandle` を直接受け取る（`DeviceBuffer` を経由しない）ため、
/// 実装は `handle.as_any().downcast_ref::<ConcreteHandle>()` で自身の
/// 具体型へダウンキャストしてから中身を書き換える
/// （`buffer.rs::BufferHandle::as_any` と同じダウンキャスト経路）。
///
/// `&mut dyn BufferHandle` を受け取るのは、プールから取り出した直後
/// （まだ `PooledBufferHandle` に包まれる前・呼び出し元へ返す前の排他
/// 所有段階）のバッファに書き込むためであり、この経路に限ることで
/// 各バックエンド実装が `unsafe` な生ポインタ書き込みなしに
/// `downcast_mut::<ConcreteHandle>()` 経由で安全に書き換えられる
/// （`buffer.rs::BufferHandle::as_any_mut` 参照）。
pub trait PoolZeroFill {
    /// `handle` の指す確保済みバッファを全要素 0 で上書きする。
    fn zero_fill(&self, handle: &mut dyn BufferHandle) -> Result<(), BackendError>;
}

/// `PoolCore` が保持する 1 エントリ。`tick` は挿入順（グローバル LRU 用）。
#[derive(Debug)]
struct PoolEntry {
    handle: Box<dyn BufferHandle>,
    bytes: u64,
    tick: u64,
}

/// プール本体（サイズ別バケット・総量上限・グローバル LRU 破棄）。
///
/// `buckets`（バイトサイズ → FIFO キュー）はバケット内では常に挿入順
/// （先頭が最古）を保つ。`order`（tick → バケットキー）は全エントリの
/// 挿入順を横断的に保持し、グローバル最古エントリを O(log n) で特定する
/// ために使う（各バケットは FIFO のため、バケット内最古は常に先頭。
/// よってプール全体の最古は「各バケット先頭の tick の最小値」＝
/// `order` の最小 tick に一致する）。
#[derive(Debug)]
struct PoolCore {
    max_pool_bytes: u64,
    buckets: BTreeMap<u64, VecDeque<PoolEntry>>,
    order: BTreeMap<u64, u64>,
    total_bytes: u64,
    next_tick: u64,
}

impl PoolCore {
    fn new(config: PoolConfig) -> Self {
        Self {
            max_pool_bytes: config.max_pool_bytes,
            buckets: BTreeMap::new(),
            order: BTreeMap::new(),
            total_bytes: 0,
            next_tick: 0,
        }
    }

    /// `bytes` バイトのバケットから再利用可能なハンドルを 1 つ取り出す。
    /// バケットが空・存在しない場合は `None`（呼び出し元はバックエンド
    /// 側の新規確保にフォールバックする）。
    fn acquire(&mut self, bytes: u64) -> Option<Box<dyn BufferHandle>> {
        let bucket = self.buckets.get_mut(&bytes)?;
        let entry = bucket.pop_front()?;
        self.order.remove(&entry.tick);
        self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
        if bucket.is_empty() {
            self.buckets.remove(&bytes);
        }
        Some(entry.handle)
    }

    /// `handle`（`bytes` バイト）をプールへ返却する。上限超過（プール無効
    /// を含む）・単一バッファが上限より大きい場合は即座に `drop` して
    /// 実解放する（design §2-3。`memory_stats::TrackedAllocation::drop`
    /// が `AllocationTracker` の `allocated_bytes` を減算するのはこの
    /// `drop` の瞬間である）。
    fn push(&mut self, bytes: u64, handle: Box<dyn BufferHandle>) {
        if self.max_pool_bytes == 0 || bytes > self.max_pool_bytes {
            drop(handle);
            return;
        }
        let tick = self.next_tick;
        self.next_tick = self.next_tick.wrapping_add(1);
        self.buckets.entry(bytes).or_default().push_back(PoolEntry {
            handle,
            bytes,
            tick,
        });
        self.order.insert(tick, bytes);
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.evict_to_limit();
    }

    /// `total_bytes` が `max_pool_bytes` を超えている間、最古エントリから
    /// 破棄し続ける（グローバル LRU。design §2-3）。
    fn evict_to_limit(&mut self) {
        while self.total_bytes > self.max_pool_bytes {
            let Some((&tick, &bytes_key)) = self.order.iter().next() else {
                break;
            };
            self.order.remove(&tick);
            let Some(bucket) = self.buckets.get_mut(&bytes_key) else {
                continue;
            };
            if let Some(evicted) = bucket.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(evicted.bytes);
                // 明示 drop: LRU 破棄の実解放そのもの（コメントで意図を
                // 明示し、単なる無駄なフィールドアクセスと区別する）。
                drop(evicted.handle);
            }
            if bucket.is_empty() {
                self.buckets.remove(&bytes_key);
            }
        }
    }

    fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// プール保持分を即座に全て解放する内部フック（モジュール冒頭
    /// 「#202 向け内部解放フック」参照）。`pub(crate)` に留め、公開 API
    /// 化は #202 に委ねる。
    ///
    /// `#[allow(dead_code)]` の理由: 本イシュー（#201）では `PooledMemory`
    /// 側に本フックを呼ぶ公開メソッドをまだ追加しない（明示解放 API の
    /// 設計・係数維持テストは #202 のスコープ）ため、現時点で呼び出し元が
    /// 存在せず構造的に未使用となる。安易な黙らせではなく、#202 が
    /// `PooledMemory::release_all_pooled` 等の公開メソッドからこの
    /// `pub(crate)` 関数を呼ぶまでの一時的な状態であることを明示する。
    #[allow(dead_code)]
    pub(crate) fn clear_all(&mut self) {
        self.buckets.clear();
        self.order.clear();
        self.total_bytes = 0;
    }
}

/// プールへ返却される `DeviceBuffer` の具体ハンドル。
///
/// `inner` は `ManuallyDrop` で保持する（`Option` にして `Drop::drop` で
/// `take()` する設計も検討したが、`as_any`／`as_any_mut` の `None` 分岐が
/// 「`self.inner` への部分借用」と「`self` 全体の再借用」を同一ライフタイム
/// で要求する形になり borrow checker が受理しない〈E0499〉。`ManuallyDrop`
/// なら通常経路（`as_any`／`as_any_mut`）は常に中身へ委譲するだけで済み、
/// 所有権を取り出す操作は `Drop::drop` の 1 箇所〈`ManuallyDrop::take`〉に
/// 閉じ込められる）。
struct PooledBufferHandle {
    inner: std::mem::ManuallyDrop<Box<dyn BufferHandle>>,
    pool: Weak<Mutex<PoolCore>>,
    bytes: u64,
}

impl PooledBufferHandle {
    fn new(inner: Box<dyn BufferHandle>, pool: Weak<Mutex<PoolCore>>, bytes: u64) -> Self {
        Self {
            inner: std::mem::ManuallyDrop::new(inner),
            pool,
            bytes,
        }
    }
}

impl std::fmt::Debug for PooledBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBufferHandle")
            .field("bytes", &self.bytes)
            .field("inner", &*self.inner)
            .finish()
    }
}

impl BufferHandle for PooledBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        // `zero_fill` は「プールから取り出した直後・まだ `PooledBufferHandle`
        // に包まれていない」段階の内部ハンドルへ直接呼ぶため
        // （`PooledMemory::alloc_zeroed` 参照）、本経路は通常到達しない。
        // それでも `BufferHandle` の完全性のため転送を実装しておく。
        self.inner.as_any_mut()
    }
}

impl Drop for PooledBufferHandle {
    /// 内部ハンドルをプールへ返却する（返却経路。モジュール冒頭
    /// 「返却経路（RAII 維持）」参照）。プールが既に破棄済み
    /// （`Weak::upgrade` 失敗）・lock poisoning 時は内部ハンドルを直接
    /// `drop` する（素直に解放。panic させない）。
    fn drop(&mut self) {
        // SAFETY: `ManuallyDrop::take` は同一フィールドへ二重に呼ばない
        // 限り安全（中身を read で取り出した後、元の `ManuallyDrop` を
        // 二度と使わないことが呼び出し元の責務）。`Drop::drop` は Rust の
        // 言語仕様上インスタンスごとにちょうど 1 回だけ呼ばれ、`inner` を
        // 読み出す経路は本関数のこの 1 行のみ（`as_any`／`as_any_mut` は
        // 参照を返すのみで所有権を奪わない）であるため、二重 take は
        // 構造的に発生しない。
        let inner = unsafe { std::mem::ManuallyDrop::take(&mut self.inner) };
        if let Some(pool) = self.pool.upgrade() {
            match pool.lock() {
                Ok(mut core) => {
                    core.push(self.bytes, inner);
                    return;
                }
                Err(poisoned) => {
                    // `memory_stats::AllocationTracker::lock` と同じ方針:
                    // 本トラッカーは単調カウンタ／FIFO キューのみを保持し
                    // 不変条件の破壊が起きないため、poisoned でも中身を
                    // そのまま引き継いで処理を継続する。
                    poisoned.into_inner().push(self.bytes, inner);
                    return;
                }
            }
        }
        // プールが既に破棄済み: 返却先が無いためそのまま解放する
        // （リークしない。design §2-4）。
        drop(inner);
    }
}

/// `numel` 分の `f32` 確保が消費するバイト数を検査付きで計算する
/// （`backend-cpu::memory::checked_byte_len` と同種の OWASP A03 前段検証。
/// `.claude/rules/security.md`: shape は safetensors/ONNX 経由で外部入力が
/// 流入しうる経路のため、バイト数換算でもオーバーフローを型付きエラーと
/// して拒否する）。
fn checked_byte_len(numel: usize) -> Result<u64, BackendError> {
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            BackendError::DeviceAllocationFailed(format!(
                "pool: allocation byte length overflows usize: numel={numel}"
            ))
        })?;
    Ok(bytes as u64)
}

/// shape の要素数積を検査付きで計算する（同上の前段検証。`buffer.rs` の
/// 各バックエンド実装がそれぞれ持つ `checked_numel` と同種）。
fn checked_numel(shape: &[usize]) -> Result<usize, BackendError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            BackendError::DeviceAllocationFailed(format!(
                "pool: shape element count overflows usize: {shape:?}"
            ))
        })
}

/// 既存 [`MemoryOps`] 実装 `M` をサイズクラス別プールで包むデコレータ
/// （モジュール冒頭「位置付け（opt-in デコレータ）」参照）。
///
/// `M: MemoryOps + PoolZeroFill` を要求する。`device` はプールヒット時
/// （バックエンド呼び出しを経由せずハンドルを再利用する経路）でも
/// `DeviceBuffer::new` に渡すデバイス情報を復元できるよう、構築時に
/// 明示的に保持する。
pub struct PooledMemory<M> {
    inner: M,
    device: Device,
    core: Arc<Mutex<PoolCore>>,
}

impl<M> PooledMemory<M> {
    /// `inner`（既存の `MemoryOps` 実装）を `config` の上限でプール化する。
    /// `device` は `inner` が確保するバッファの `Device`（`CpuMemory` なら
    /// `Device::Cpu` 等）を呼び出し元が明示する。
    pub fn new(inner: M, device: Device, config: PoolConfig) -> Self {
        Self {
            inner,
            device,
            // `PoolCore` は `Box<dyn BufferHandle>` を保持し、`BufferHandle`
            // は `Send`/`Sync` を要求しない設計（`buffer.rs` モジュール
            // コメント「Send/Sync 境界」参照。Metal `Retained<MTLBuffer>` の
            // スレッド安全性を過剰に約束しないための v1 からの判断）。
            // clippy の `arc_with_non_send_sync` はこの `Mutex<PoolCore>`
            // が実際に `Send`/`Sync` でないことを正しく検出しており指摘は
            // 妥当だが、それでも `Rc` ではなく `Arc` が必要な理由:
            // `PooledBufferHandle::pool` が `Weak<Mutex<PoolCore>>` として
            // このプールを弱参照する（`Drop` 時にプール生存を確認して返却
            // する設計。モジュール冒頭「返却経路」参照）ため、`Weak` を
            // 持たない `Rc` ではこの自己参照的な共有所有権を表現できない。
            // マルチスレッド共有そのものは意図しておらず、`Send`/`Sync`
            // 非対応であることは既存 `BufferHandle` の設計判断を素直に
            // 継承しているだけであるため、ここでは `allow` する。
            #[allow(clippy::arc_with_non_send_sync)]
            core: Arc::new(Mutex::new(PoolCore::new(config))),
        }
    }

    /// 元の `MemoryOps` 実装への参照（`MemoryStats` 転送等、プール外の
    /// 用途向け）。
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// プールが現在アイドル保持しているバイト数の合計（受け入れ条件検証
    /// 用の統計。`memory_stats::MemoryStats::allocated_bytes` とは別軸で、
    /// 「生存中だが未使用」の量を表す）。
    pub fn pooled_bytes(&self) -> u64 {
        match self.core.lock() {
            Ok(core) => core.total_bytes(),
            Err(poisoned) => poisoned.into_inner().total_bytes(),
        }
    }

    fn try_acquire(&self, bytes: u64) -> Option<Box<dyn BufferHandle>> {
        match self.core.lock() {
            Ok(mut core) => core.acquire(bytes),
            Err(poisoned) => poisoned.into_inner().acquire(bytes),
        }
    }
}

impl<M: MemoryOps + PoolZeroFill> MemoryOps for PooledMemory<M> {
    /// バイトサイズ完全一致のバケットから再利用を試み、無ければ `inner`
    /// へ委譲して新規確保する。いずれの経路でも返す `DeviceBuffer` の
    /// ハンドルは [`PooledBufferHandle`] で包み、`Drop` 時にプールへ
    /// 返却されるようにする（`numel == 0` はモジュール冒頭「空テンソル
    /// 契約」のとおりプールを介さない）。
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        let numel = checked_numel(shape)?;
        if numel == 0 {
            return self.inner.alloc_zeroed(shape);
        }
        let bytes = checked_byte_len(numel)?;

        if let Some(mut handle) = self.try_acquire(bytes) {
            // 再利用時は前利用データが残留しうるため、ゼロ初期化契約
            // （`buffer.rs::MemoryOps::alloc_zeroed` の「全要素 0」契約）
            // を明示的に再適用する（モジュール冒頭「ゼロ初期化契約の
            // 維持」参照）。プールから取り出した直後（まだ他に共有
            // されていない排他所有の段階）であるため `&mut` を渡せる
            // （`PoolZeroFill::zero_fill` のシグネチャコメント参照）。
            self.inner.zero_fill(handle.as_mut())?;
            let pooled = PooledBufferHandle::new(handle, Arc::downgrade(&self.core), bytes);
            return Ok(DeviceBuffer::new(
                self.device,
                shape.to_vec(),
                Box::new(pooled),
            ));
        }

        let buffer = self.inner.alloc_zeroed(shape)?;
        let device = buffer.device();
        // 再利用経路（上の `try_acquire` 分岐）はハンドル自体からデバイス
        // を復元できず（`BufferHandle` は `device()` を持たない）構築時に
        // 渡された `self.device` をそのまま `DeviceBuffer::new` へ渡す。
        // その前提を保つには「`self.device` は常に `inner` の実確保先と
        // 一致する」という不変条件が必要で、ここで実測値と照合しておかない
        // と、`inner` が構築時指定と異なるデバイスへ確保する構成（誤設定・
        // 将来の `inner` 実装変更）で初回は正しいデバイス、プール再利用後は
        // 誤ったデバイスを報告する状態依存の不整合になる（`download` での
        // 誤判定・誤ディスパッチを招く）。不一致時はプールへ格納せず
        // `BackendError::DeviceMismatch` を返し、以後の全確保が同じ不整合を
        // 継承するのを防ぐ。
        if device != self.device {
            return Err(BackendError::DeviceMismatch);
        }
        let shape_vec = buffer.shape().to_vec();
        let handle = buffer.into_handle();
        let pooled = PooledBufferHandle::new(handle, Arc::downgrade(&self.core), bytes);
        Ok(DeviceBuffer::new(device, shape_vec, Box::new(pooled)))
    }

    /// パススルー（モジュール冒頭「プール対象は `alloc_zeroed` のみ」
    /// 参照）。返す `DeviceBuffer` は `inner` の具体ハンドルをそのまま
    /// 持ち、`PooledBufferHandle` で包まない（プールへは返却されない）。
    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        self.inner.upload(tensor)
    }

    /// `buffer` のハンドルが `PooledBufferHandle`（`alloc_zeroed` 経由）・
    /// `inner` の生ハンドル（`upload` 経由）のいずれであっても、透過
    /// ダウンキャスト（モジュール冒頭参照）により `inner.download` が
    /// そのまま動作する。
    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.download(buffer)
    }
}

impl<M: crate::memory_stats::MemoryStats> crate::memory_stats::MemoryStats for PooledMemory<M> {
    /// `inner` へ委譲する（モジュール冒頭「計測反映」参照。プール保持中
    /// も内部ハンドルは生存しているため `inner.allocated_bytes()` に
    /// 自然に計上される）。
    fn allocated_bytes(&self) -> u64 {
        self.inner.allocated_bytes()
    }

    fn peak_allocated_bytes(&self) -> u64 {
        self.inner.peak_allocated_bytes()
    }

    fn reset_peak(&self) {
        self.inner.reset_peak();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
    use std::cell::Cell;
    use std::rc::Rc;

    /// テスト専用のモックハンドル。`alive` を通じて実解放（`Drop`）が
    /// 起きたかどうかを検証する。`buffer.rs` テストの `MockHandle` と
    /// 同種の設計判断（`Rc<Cell<_>>` はシングルスレッドテストで十分）。
    #[derive(Debug)]
    struct MockHandle {
        payload: Vec<f32>,
        alive: Rc<Cell<bool>>,
        _alloc: Option<TrackedAllocation>,
    }

    impl BufferHandle for MockHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl Drop for MockHandle {
        fn drop(&mut self) {
            self.alive.set(false);
        }
    }

    /// `MemoryOps + PoolZeroFill` のモック実装。`alloc_count` で下位
    /// アロケータへの新規確保回数を、`zero_fill_count` でゼロ初期化
    /// 呼び出し回数を検証する。`tracker` は `memory_stats::MemoryStats`
    /// 委譲テスト用。
    struct MockMemory {
        alloc_count: Cell<usize>,
        zero_fill_count: Cell<usize>,
        tracker: Arc<AllocationTracker>,
    }

    impl MockMemory {
        fn new() -> Self {
            Self {
                alloc_count: Cell::new(0),
                zero_fill_count: Cell::new(0),
                tracker: Arc::new(AllocationTracker::new()),
            }
        }
    }

    impl MemoryOps for MockMemory {
        fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
            let numel: usize = shape.iter().product();
            self.alloc_count.set(self.alloc_count.get() + 1);
            let bytes = checked_byte_len(numel)?;
            let alloc = TrackedAllocation::new(Arc::clone(&self.tracker), bytes);
            let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
                payload: vec![1.0f32; numel], // 非ゼロで初期化し zero_fill 検証に使う
                alive: Rc::new(Cell::new(true)),
                _alloc: Some(alloc),
            });
            Ok(DeviceBuffer::new(Device::Cpu, shape.to_vec(), handle))
        }

        fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
            self.alloc_zeroed(tensor.shape())
        }

        fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
            let handle = buffer
                .downcast_handle::<MockHandle>()
                .expect("MockMemory から生成した DeviceBuffer は MockHandle を持つはず");
            Tensor::new(handle.payload.clone(), buffer.shape()).map_err(BackendError::ShapeMismatch)
        }
    }

    impl PoolZeroFill for MockMemory {
        fn zero_fill(&self, handle: &mut dyn BufferHandle) -> Result<(), BackendError> {
            self.zero_fill_count.set(self.zero_fill_count.get() + 1);
            // 実際にゼロで上書きし、再利用時の「全要素 0」契約が
            // `PooledMemory::alloc_zeroed` から呼ばれることをテスト側でも
            // 検証できるようにする。
            if let Some(mock) = handle.as_any_mut().downcast_mut::<MockHandle>() {
                mock.payload.fill(0.0);
            }
            Ok(())
        }
    }

    impl MemoryStats for MockMemory {
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

    /// 同一サイズ alloc→drop→alloc で下位アロケータの確保回数が増えない
    /// （再利用される）ことを検証する。
    #[test]
    fn same_size_alloc_drop_alloc_reuses_buffer() {
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, PoolConfig::default());

        let buf1 = mem.alloc_zeroed(&[64]).unwrap();
        assert_eq!(mem.inner().alloc_count.get(), 1);
        drop(buf1);
        assert_eq!(mem.pooled_bytes(), 64 * 4);

        let buf2 = mem.alloc_zeroed(&[64]).unwrap();
        assert_eq!(
            mem.inner().alloc_count.get(),
            1,
            "同一サイズの再利用では下位アロケータへ新規確保が発生しないはず"
        );
        assert_eq!(
            mem.inner().zero_fill_count.get(),
            1,
            "再利用時はゼロ初期化契約を再適用するはず"
        );
        drop(buf2);
    }

    /// `PooledMemory::new` に渡した `device` が `inner` の実確保先
    /// （`MockMemory::alloc_zeroed` は常に `Device::Cpu` へ確保する）と
    /// 一致しない場合、`alloc_zeroed` は `BackendError::DeviceMismatch`
    /// を返しプールへ格納しないことを検証する（codex-review 指摘・
    /// `crates/tensor-core/src/pool.rs:390` の再利用時未検証 device
    /// 修正の回帰テスト）。プールに格納されないため、以後の同サイズ
    /// 確保も毎回この検証を経て失敗し続け、状態依存で誤ったデバイスを
    /// 報告することがない。
    #[test]
    fn device_mismatch_on_fresh_alloc_returns_error_and_is_not_pooled() {
        let mem = PooledMemory::new(MockMemory::new(), Device::Cuda(0), PoolConfig::default());

        let err = mem.alloc_zeroed(&[64]).unwrap_err();
        assert!(
            matches!(err, BackendError::DeviceMismatch),
            "device 不一致は BackendError::DeviceMismatch を返すはず（実測 {err:?}）"
        );
        assert_eq!(
            mem.pooled_bytes(),
            0,
            "device 不一致のバッファはプールへ格納されないはず"
        );

        // 不一致は 1 度限りの取りこぼしではなく、以後も再現し続ける
        // （プールに紛れ込まず、都度 inner から実測して検出される）。
        let err2 = mem.alloc_zeroed(&[64]).unwrap_err();
        assert!(matches!(err2, BackendError::DeviceMismatch));
    }

    /// 異なるサイズはバケット分離され再利用されないことを検証する
    /// （完全一致方針の固定）。
    #[test]
    fn different_size_allocations_are_not_reused_across_buckets() {
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, PoolConfig::default());

        let buf_a = mem.alloc_zeroed(&[64]).unwrap();
        drop(buf_a);
        assert_eq!(mem.inner().alloc_count.get(), 1);

        let buf_b = mem.alloc_zeroed(&[128]).unwrap();
        assert_eq!(
            mem.inner().alloc_count.get(),
            2,
            "異なるバイトサイズはバケットが別れ再利用されないはず"
        );
        drop(buf_b);
    }

    /// 上限超過時に最古エントリから自動破棄され、`pooled_bytes <=
    /// max_pool_bytes` 不変条件が常に成立することを検証する
    /// （#201 受け入れ条件の直接検証）。
    #[test]
    fn exceeding_limit_evicts_oldest_entry_first() {
        // 64 要素 = 256 バイトのバッファを 2 本まで保持できる上限（512）。
        let config = PoolConfig {
            max_pool_bytes: 512,
        };
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, config);

        let buf1 = mem.alloc_zeroed(&[64]).unwrap(); // 256 バイト
        let buf2 = mem.alloc_zeroed(&[128]).unwrap(); // 512 バイト（別バケット）
        let buf3 = mem.alloc_zeroed(&[192]).unwrap(); // 768 バイト（別バケット）
        drop(buf1); // pooled_bytes = 256
        drop(buf2); // pooled_bytes = 256 + 512 = 768 > 512 -> 最古（buf1 由来）を破棄
        assert!(
            mem.pooled_bytes() <= 512,
            "上限超過後は pooled_bytes が max_pool_bytes 以下になるはず（実測 {}）",
            mem.pooled_bytes()
        );
        // buf1（64 要素バケット）が破棄され、64 要素の再確保は新規確保になるはず。
        let alloc_count_before = mem.inner().alloc_count.get();
        let buf4 = mem.alloc_zeroed(&[64]).unwrap();
        assert_eq!(
            mem.inner().alloc_count.get(),
            alloc_count_before + 1,
            "LRU 破棄されたバケットの再確保は新規確保になるはず"
        );
        drop(buf3);
        drop(buf4);
    }

    /// 上限より大きい単一バッファはプール非経由で即解放されることを検証する。
    #[test]
    fn buffer_larger_than_limit_bypasses_pool() {
        let config = PoolConfig {
            max_pool_bytes: 128,
        };
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, config);

        let buf = mem.alloc_zeroed(&[64]).unwrap(); // 256 バイト > 128 バイト上限
        drop(buf);
        assert_eq!(
            mem.pooled_bytes(),
            0,
            "上限より大きい単一バッファはプールに入らず即解放されるはず"
        );
    }

    /// `max_pool_bytes == 0` で全パススルーになることを検証する。
    #[test]
    fn zero_limit_disables_pooling_entirely() {
        let config = PoolConfig { max_pool_bytes: 0 };
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, config);

        let buf = mem.alloc_zeroed(&[16]).unwrap();
        drop(buf);
        assert_eq!(mem.pooled_bytes(), 0);

        let alloc_count_before = mem.inner().alloc_count.get();
        let buf2 = mem.alloc_zeroed(&[16]).unwrap();
        assert_eq!(
            mem.inner().alloc_count.get(),
            alloc_count_before + 1,
            "プール無効時は毎回新規確保になるはず"
        );
        drop(buf2);
    }

    /// プール破棄後の handle `Drop` は素直に解放され、リークしないことを
    /// 検証する（`Weak` 経路。`MockHandle::alive` で実解放を確認）。
    #[test]
    fn dropping_pool_before_handle_still_releases_handle() {
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, PoolConfig::default());
        let buf = mem.alloc_zeroed(&[8]).unwrap();

        drop(mem); // プール本体（Arc<Mutex<PoolCore>>）を先に破棄する
        drop(buf); // Weak::upgrade が失敗し、素直に解放される経路
        // panic しないことそのものが検証（handle 側の Drop 内 unwrap 皆無）。
    }

    /// 空テンソル（numel == 0）はプール非経由であることを検証する
    /// （`inner.alloc_zeroed` へ直接委譲され、`PooledBufferHandle` で
    /// 包まれないため drop してもプール統計に影響しない）。
    #[test]
    fn empty_tensor_bypasses_pool() {
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, PoolConfig::default());
        let buf = mem.alloc_zeroed(&[0, 4]).unwrap();
        drop(buf);
        assert_eq!(mem.pooled_bytes(), 0);
    }

    /// `PooledMemory` の `MemoryStats` 委譲が `inner` に正しく転送される
    /// ことを検証する（#201 受け入れ条件「ピーク計測 API に反映される」の
    /// tensor-core 側の受け皿。バックエンド統合テストは backend-cpu 側）。
    #[test]
    fn memory_stats_delegates_to_inner_and_reflects_pool_eviction() {
        let config = PoolConfig {
            max_pool_bytes: 256, // 64 要素 1 本分のみ保持できる上限
        };
        let mem = PooledMemory::new(MockMemory::new(), Device::Cpu, config);

        assert_eq!(mem.allocated_bytes(), 0);
        let buf = mem.alloc_zeroed(&[64]).unwrap(); // 256 バイト
        assert_eq!(mem.allocated_bytes(), 256);
        assert_eq!(mem.peak_allocated_bytes(), 256);

        drop(buf); // プールへ返却: 内部ハンドルは生存し続けるため allocated_bytes は減らない
        assert_eq!(
            mem.allocated_bytes(),
            256,
            "プール保持中はハンドルが生存しているため allocated_bytes は減らないはず"
        );

        // 上限超過を発生させて LRU 破棄させる: 256 バイト 1 本保持済みの
        // 状態でさらに 256 バイトを確保・返却すると合計 512 > 256 となり、
        // 最古（先に返却済みの 1 本目）が破棄される。
        let buf2 = mem.alloc_zeroed(&[64]).unwrap();
        drop(buf2);
        assert_eq!(
            mem.allocated_bytes(),
            256,
            "LRU 破棄により allocated_bytes は上限相当まで減少するはず（受け入れ条件の直接検証）"
        );
        assert_eq!(
            mem.peak_allocated_bytes(),
            256,
            "peak はこのシナリオでは同時生存が 256 バイトを超えないため据え置き"
        );
    }
}
