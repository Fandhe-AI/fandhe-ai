//! ハンドル非依存のサイズクラス別プール本体（イシュー #1020・REQ-14）。
//!
//! # 位置付け・既存 [`crate::pool`] との違い
//!
//! [`crate::pool::PooledMemory`] はバイトサイズ**完全一致**のバケット方式
//! （opt-in デコレータ）であり、どの本番実行経路にも接続されていない
//! （`docs/device-memory-pool-design.md` §3.1・`docs/backend-cuda-pool-allocator-decision.md`）。
//! 本モジュールはそれとは別に、`backend-cuda`（後続 #1021 の `backend-metal`
//! も同様に予定）のホットパス（GEMM／elementwise／softmax の出力バッファ確保）
//! へ実際に接続するための**サイズクラス丸め**（近い要求サイズを同一クラスへ
//! 集約し再利用率を上げる）プール本体 [`SizeClassPool`] を提供する。
//!
//! ハンドル型 `H` を型パラメータとする（`Send` のみを要求）ことで、
//! `backend-cuda::pool::CudaSliceHandle`（`CudaSlice<f32>` を包む）・
//! 将来の `backend-metal` ハンドル型のいずれでも同じプール機構を再利用できる
//! （`backend-cpu` は cudarc/objc2 のような確保コストが小さいため対象外）。
//! `crate::pool::PooledMemory` は `MemoryOps` デコレータとして残し、本モジュール
//! と統合・置換はしない（既存公開 API `PoolConfig`/`PooledMemory` 非破壊。
//! 設計判断は `docs/backend-cuda-pool-allocator-decision.md` を参照）。
//!
//! # 型名衝突の回避
//!
//! 本モジュールの [`PoolConfig`] はクレートルートへ再エクスポート**しない**
//! （`crate::pool::PoolConfig` が crates.io 0.4.0 で公開済みのため、同名を
//! root で再エクスポートすると衝突する）。呼び出し元は
//! `fandhe_ai_tensor_core::pool_core::PoolConfig` のフルパスで参照する。
//! ルートへ再エクスポートするのは [`PoolStats`] のみ（`backend_ops::BackendOps`
//! のデフォルトメソッド `device_memory_pool_stats` の戻り値型のため）。
//!
//! # サイズクラス丸め規則（design 文書 §3.2 を実装）
//!
//! [`size_class_for`] は要求バイト数を以下の帯へ丸める:
//!
//! - `0` バイト → `None`（プール対象外。空バッファは呼び出し元がプールを
//!   介さず直接ハンドリングする契約。`crate::pool` の「空テンソル契約」と同型）
//! - `1..=255` バイト → `256` バイト固定（極小帯）
//! - `256 バイト以上 1MiB（[`PoolConfig::small_band_max`]）以下` →
//!   4 段階（256B の 2 のべき乗切り上げ: 256, 1KiB, 4KiB, ... と同種の
//!   処理を簡略化し、`next_power_of_two` で切り上げる小帯）
//! - `1MiB 超 64MiB（[`PoolConfig::huge_threshold`]）以下` →
//!   [`PoolConfig::large_granularity`]（既定 2MiB）単位への切り上げ（大帯）
//! - `64MiB 超` → 完全一致（巨大帯。丸めない。バケット当たり
//!   [`PoolConfig::huge_max_entries_per_class`] 件までしか保持しない）
//!
//! 計算はすべて `checked_*` を用い、オーバーフロー時は
//! `BackendError::DeviceAllocationFailed` を返す（OWASP A03。
//! `.claude/rules/security.md`）。
//!
//! # 所有権契約（design 文書 §3.5 を実装）
//!
//! - (a) `take` はヒット時 `Some(H)` を返しプール内部の該当バケットから
//!   除去する（二重貸出はしない）
//! - (b) `put` は返却されたハンドルをバケットへ戻し、総量上限
//!   （[`PoolConfig::max_pool_bytes`]）超過分をグローバル LRU で破棄する。
//!   破棄対象は `Vec<H>` として**呼び出し元へ返す**（ロック内で `H` の
//!   `Drop`〈CUDA では FFI 解放を伴いうる〉を実行しない契約。§3.5「ロック
//!   粒度」）
//! - (c) 巨大帯は 1 クラスあたり [`PoolConfig::huge_max_entries_per_class`]
//!   件（既定 1）までしか保持しない（`docs/backend-cuda-pool-allocator-decision.md`
//!   の「巨大確保は再利用機会が乏しく専有コストが高い」判断）
//!
//! # 統計契約
//!
//! [`PoolStats`] は `alloc_count`（`take` 呼び出し総数）・`reuse_count`
//! （`take` がヒットした回数）・`cached_bytes`（現在アイドル保持中の総
//! バイト数。恒等式 `cached_bytes == Σ class_bytes` を維持）・
//! `pending_return_bytes`（返却待ち。CUDA は即時返却のため常に 0 だが、
//! 将来の非同期返却バックエンド向けに予約）・`capacity_waste_bytes`
//! （丸めによる `class_bytes - requested_bytes` の累積。実質未使用の
//! 確保 API サポート値として保持）・`released_bytes`（LRU 破棄・明示解放で
//! 実解放した累積バイト数）を保持する。

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use crate::device::BackendError;

/// サイズクラス丸めパラメータ（モジュール冒頭「サイズクラス丸め規則」参照）。
///
/// `crate::pool::PoolConfig`（バイトサイズ完全一致方式）とは別の型
/// （モジュール冒頭「型名衝突の回避」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    /// プールがアイドル保持してよいバイト数の総上限（既定 128MiB。
    /// `crate::pool::PoolConfig::default` と同値。REQ-14 14-3 の係数
    /// 上限〈2 倍以内〉を侵さない安全側の値として据え置く）。
    pub max_pool_bytes: u64,
    /// 小帯（`next_power_of_two` 切り上げ）の上限バイト数（既定 1MiB）。
    pub small_band_max: u64,
    /// 大帯（`1MiB 超 huge_threshold 以下`）の切り上げ粒度（既定 2MiB）。
    pub large_granularity: u64,
    /// これを超えるバイト数は完全一致の巨大帯として扱う（既定 64MiB）。
    pub huge_threshold: u64,
    /// 巨大帯 1 クラスあたりの最大保持エントリ数（既定 1）。
    pub huge_max_entries_per_class: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_bytes: 128 * 1024 * 1024,
            small_band_max: 1024 * 1024,
            large_granularity: 2 * 1024 * 1024,
            huge_threshold: 64 * 1024 * 1024,
            huge_max_entries_per_class: 1,
        }
    }
}

/// プール利用状況の統計（モジュール冒頭「統計契約」参照）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// `take` の呼び出し総数。
    pub alloc_count: u64,
    /// `take` がヒット（プールから再利用）した回数。
    pub reuse_count: u64,
    /// 現在アイドル保持中の総バイト数。
    pub cached_bytes: u64,
    /// 返却待ちバイト数（CUDA は即時返却のため常に 0）。
    pub pending_return_bytes: u64,
    /// サイズクラス丸めによる確保過多バイト数の累積。
    pub capacity_waste_bytes: u64,
    /// LRU 破棄・明示解放で実解放した累積バイト数。
    pub released_bytes: u64,
}

/// 要求バイト数 `bytes` をサイズクラスへ丸める（モジュール冒頭「サイズ
/// クラス丸め規則」参照）。`bytes == 0` は `Ok(None)`（プール対象外）。
/// 演算はすべて `checked_*` で行い、オーバーフローは
/// `BackendError::DeviceAllocationFailed` として拒否する。
pub fn size_class_for(bytes: u64, config: &PoolConfig) -> Result<Option<u64>, BackendError> {
    if bytes == 0 {
        return Ok(None);
    }
    if bytes > config.huge_threshold {
        // 巨大帯: 丸めない（完全一致）。
        return Ok(Some(bytes));
    }
    if bytes <= config.small_band_max {
        // 小帯: 2 のべき乗切り上げ。256 未満は 256 に底上げする
        // （極小確保のクラス数爆発を防ぐ。design §3.2）。
        let floored = bytes.max(256);
        let rounded = floored.checked_next_power_of_two().ok_or_else(|| {
            BackendError::DeviceAllocationFailed(format!(
                "pool_core: size class rounding overflows u64: bytes={bytes}"
            ))
        })?;
        return Ok(Some(rounded));
    }
    // 大帯: large_granularity 単位への切り上げ。
    let granularity = config.large_granularity.max(1);
    let quotient = bytes.checked_add(granularity - 1).ok_or_else(|| {
        BackendError::DeviceAllocationFailed(format!(
            "pool_core: size class rounding overflows u64: bytes={bytes}"
        ))
    })? / granularity;
    let rounded = quotient.checked_mul(granularity).ok_or_else(|| {
        BackendError::DeviceAllocationFailed(format!(
            "pool_core: size class rounding overflows u64: bytes={bytes}"
        ))
    })?;
    Ok(Some(rounded))
}

/// `PoolCore` が保持する 1 エントリ。`tick` は挿入順（グローバル LRU 用。
/// `crate::pool::PoolEntry` と同型の設計）。
struct Entry<H> {
    handle: H,
    tick: u64,
}

/// プール本体の内部状態（`Mutex` で保護される。ロック内で `H` の `Drop`
/// を実行しないため `H` に追加の trait 境界を要求しない）。
struct PoolCore<H> {
    config: PoolConfig,
    buckets: BTreeMap<u64, VecDeque<Entry<H>>>,
    order: BTreeMap<u64, u64>,
    cached_bytes: u64,
    next_tick: u64,
    stats: PoolStats,
}

impl<H> PoolCore<H> {
    fn new(config: PoolConfig) -> Self {
        Self {
            config,
            buckets: BTreeMap::new(),
            order: BTreeMap::new(),
            cached_bytes: 0,
            next_tick: 0,
            stats: PoolStats::default(),
        }
    }

    fn take(&mut self, class_bytes: u64) -> Option<H> {
        self.stats.alloc_count = self.stats.alloc_count.saturating_add(1);
        let bucket = self.buckets.get_mut(&class_bytes)?;
        let entry = bucket.pop_front()?;
        self.order.remove(&entry.tick);
        self.cached_bytes = self.cached_bytes.saturating_sub(class_bytes);
        self.stats.cached_bytes = self.cached_bytes;
        self.stats.reuse_count = self.stats.reuse_count.saturating_add(1);
        if bucket.is_empty() {
            self.buckets.remove(&class_bytes);
        }
        Some(entry.handle)
    }

    /// `handle`（`class_bytes` バイトのクラス）をプールへ返却する。
    /// 上限超過分・巨大帯の保持数超過分は `Vec<H>` として呼び出し元へ返す
    /// （ロック内で `Drop` しない契約。design §3.5）。
    fn put(&mut self, class_bytes: u64, handle: H) -> Vec<H> {
        let mut evicted = Vec::new();
        if self.config.max_pool_bytes == 0 || class_bytes > self.config.max_pool_bytes {
            evicted.push(handle);
            return evicted;
        }
        let is_huge = class_bytes > self.config.huge_threshold;
        if is_huge {
            let cap = self.config.huge_max_entries_per_class.max(1);
            if let Some(bucket) = self.buckets.get(&class_bytes)
                && bucket.len() >= cap
            {
                // 巨大帯は保持上限を超える追加返却をそのまま解放対象にする
                // （モジュール冒頭 (c)）。
                evicted.push(handle);
                return evicted;
            }
        }
        let tick = self.next_tick;
        self.next_tick = self.next_tick.wrapping_add(1);
        self.buckets
            .entry(class_bytes)
            .or_default()
            .push_back(Entry { handle, tick });
        self.order.insert(tick, class_bytes);
        self.cached_bytes = self.cached_bytes.saturating_add(class_bytes);
        self.stats.cached_bytes = self.cached_bytes;
        evicted.extend(self.evict_to_limit());
        evicted
    }

    /// `cached_bytes` が上限を超えている間、グローバル最古エントリから
    /// 破棄対象へ積む（`crate::pool::PoolCore::evict_to_limit` と同型）。
    fn evict_to_limit(&mut self) -> Vec<H> {
        let mut evicted = Vec::new();
        while self.cached_bytes > self.config.max_pool_bytes {
            let Some((&tick, &class_bytes)) = self.order.iter().next() else {
                break;
            };
            self.order.remove(&tick);
            let Some(bucket) = self.buckets.get_mut(&class_bytes) else {
                continue;
            };
            if let Some(entry) = bucket.pop_front() {
                self.cached_bytes = self.cached_bytes.saturating_sub(class_bytes);
                self.stats.cached_bytes = self.cached_bytes;
                self.stats.released_bytes = self.stats.released_bytes.saturating_add(class_bytes);
                evicted.push(entry.handle);
            }
            if bucket.is_empty() {
                self.buckets.remove(&class_bytes);
            }
        }
        evicted
    }

    /// `release_cached` フェーズ (ii) 用: バケットから 1 件取り出す
    /// （呼び出し元がロック解放後に drop する。ロック内で `H::Drop` を
    /// 呼ばない契約を維持するため、複数件の取り出しはこのメソッドを
    /// ループで呼び出す〈`backend-cuda::pool::release_cached_with`〉）。
    fn take_one_for_release(&mut self) -> Option<(u64, H)> {
        let (&tick, &class_bytes) = self.order.iter().next()?;
        self.order.remove(&tick);
        let bucket = self.buckets.get_mut(&class_bytes)?;
        let entry = bucket.pop_front()?;
        self.cached_bytes = self.cached_bytes.saturating_sub(class_bytes);
        self.stats.cached_bytes = self.cached_bytes;
        if bucket.is_empty() {
            self.buckets.remove(&class_bytes);
        }
        Some((class_bytes, entry.handle))
    }

    fn record_release(&mut self, class_bytes: u64) {
        self.stats.released_bytes = self.stats.released_bytes.saturating_add(class_bytes);
    }

    fn record_capacity_waste(&mut self, class_bytes: u64, requested_bytes: u64) {
        self.stats.capacity_waste_bytes = self
            .stats
            .capacity_waste_bytes
            .saturating_add(class_bytes.saturating_sub(requested_bytes));
    }

    fn record_pending_return(&mut self, delta: i64) {
        if delta >= 0 {
            self.stats.pending_return_bytes =
                self.stats.pending_return_bytes.saturating_add(delta as u64);
        } else {
            self.stats.pending_return_bytes = self
                .stats
                .pending_return_bytes
                .saturating_sub(delta.unsigned_abs());
        }
    }
}

/// ハンドル非依存のサイズクラス別プール（`backend-cuda::pool::CudaAllocator`
/// が `SizeClassPool<CudaSliceHandle>` として利用する。モジュール冒頭参照）。
///
/// `H: Send` のみを要求する（`Sync` は要求しない。`Mutex` 越しの排他アクセス
/// のみで共有するため `H` 自体の `Sync` は不要。`SizeClassPool<H>` 自身が
/// `Send + Sync` になることは `#[cfg(test)]` のコンパイル時アサーションで
/// 固定する）。
pub struct SizeClassPool<H> {
    core: Mutex<PoolCore<H>>,
    config: PoolConfig,
}

impl<H: Send> SizeClassPool<H> {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            core: Mutex::new(PoolCore::new(config)),
            config,
        }
    }

    pub fn config(&self) -> PoolConfig {
        self.config
    }

    /// `class_bytes` のバケットから再利用可能なハンドルを取り出す
    /// （ヒットしなければ `None`。呼び出し元は新規確保にフォールバック
    /// する。`crate::pool::PoolCore::acquire` と同型）。
    pub fn take(&self, class_bytes: u64) -> Option<H> {
        match self.core.lock() {
            Ok(mut core) => core.take(class_bytes),
            // poisoned でも継続する方針は `crate::pool`／`memory_stats` と
            // 同一（単調カウンタ／FIFO キューのみで不変条件破壊が起きない）。
            Err(poisoned) => poisoned.into_inner().take(class_bytes),
        }
    }

    /// `handle`（`class_bytes` バイトのクラス）をプールへ返却する。
    /// 上限超過・巨大帯保持数超過分は `Vec<H>` として返す（呼び出し元が
    /// ロック非保持の状態で `drop` する契約。design §3.5「ロック粒度」）。
    #[must_use = "破棄対象ハンドルを drop し忘れるとメモリがリークする"]
    pub fn put(&self, class_bytes: u64, handle: H) -> Vec<H> {
        match self.core.lock() {
            Ok(mut core) => core.put(class_bytes, handle),
            Err(poisoned) => poisoned.into_inner().put(class_bytes, handle),
        }
    }

    /// 確保が発生した（プールミス）ことを記録する。`take` がヒットした
    /// 場合はこのメソッドを呼ばない（`alloc_count`/`reuse_count` は
    /// `take` 自体が計上するため）。
    pub fn record_allocation(&self, class_bytes: u64, requested_bytes: u64) {
        match self.core.lock() {
            Ok(mut core) => core.record_capacity_waste(class_bytes, requested_bytes),
            Err(poisoned) => poisoned
                .into_inner()
                .record_capacity_waste(class_bytes, requested_bytes),
        }
    }

    /// 貸出終了（`Drop` 開始）を記録するフック。現状 `SizeClassPool` 自体
    /// は明示的な貸出カウントを持たないため no-op だが、将来の貸出数
    /// 統計拡張に備えて呼び出し元（`backend-cuda::pool::PooledCudaHandle`）
    /// からの呼び出し点を先に固定しておく。
    pub fn record_loan_end(&self) {}

    /// `release_cached` フェーズ (ii) 用: 1 件取り出す（ループ呼び出し）。
    pub fn take_one_for_release(&self) -> Option<(u64, H)> {
        match self.core.lock() {
            Ok(mut core) => core.take_one_for_release(),
            Err(poisoned) => poisoned.into_inner().take_one_for_release(),
        }
    }

    pub fn record_release(&self, class_bytes: u64) {
        match self.core.lock() {
            Ok(mut core) => core.record_release(class_bytes),
            Err(poisoned) => poisoned.into_inner().record_release(class_bytes),
        }
    }

    pub fn record_pending_merge(&self, delta: i64) {
        match self.core.lock() {
            Ok(mut core) => core.record_pending_return(delta),
            Err(poisoned) => poisoned.into_inner().record_pending_return(delta),
        }
    }

    pub fn stats(&self) -> PoolStats {
        match self.core.lock() {
            Ok(core) => core.stats,
            Err(poisoned) => poisoned.into_inner().stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PoolConfig {
        PoolConfig::default()
    }

    // --- size_class_for 丸め規則 ---

    #[test]
    fn size_class_zero_bytes_is_none() {
        assert_eq!(size_class_for(0, &cfg()).unwrap(), None);
    }

    #[test]
    fn size_class_small_band_rounds_to_power_of_two() {
        assert_eq!(size_class_for(1, &cfg()).unwrap(), Some(256));
        assert_eq!(size_class_for(200, &cfg()).unwrap(), Some(256));
        assert_eq!(size_class_for(255, &cfg()).unwrap(), Some(256));
        assert_eq!(size_class_for(256, &cfg()).unwrap(), Some(256));
        assert_eq!(size_class_for(257, &cfg()).unwrap(), Some(512));
    }

    #[test]
    fn size_class_small_band_boundary_1mib() {
        let one_mib = 1024 * 1024;
        assert_eq!(size_class_for(one_mib, &cfg()).unwrap(), Some(one_mib));
        // 1MiB + 1 は大帯（large_granularity=2MiB 切り上げ）に入る。
        assert_eq!(
            size_class_for(one_mib + 1, &cfg()).unwrap(),
            Some(2 * 1024 * 1024)
        );
    }

    #[test]
    fn size_class_large_band_rounds_to_granularity() {
        // 200,704 バイト → 大帯（1MiB 超えていないので実際は小帯側）
        // ではなく明示的に大帯範囲の値で検証する。
        let two_mib = 2 * 1024 * 1024;
        let three_mib = 3 * 1024 * 1024;
        assert_eq!(
            size_class_for(two_mib + 1, &cfg()).unwrap(),
            Some(2 * two_mib)
        );
        assert_eq!(
            size_class_for(three_mib, &cfg()).unwrap(),
            Some(2 * two_mib)
        );
    }

    #[test]
    fn size_class_huge_band_is_exact_match() {
        let c = cfg();
        assert_eq!(
            size_class_for(c.huge_threshold, &c).unwrap(),
            Some(c.huge_threshold)
        );
        assert_eq!(
            size_class_for(c.huge_threshold + 1, &c).unwrap(),
            Some(c.huge_threshold + 1)
        );
        // 巨大帯は要求ごとに異なる値でも丸めない（完全一致）。
        let odd = c.huge_threshold + 12345;
        assert_eq!(size_class_for(odd, &c).unwrap(), Some(odd));
    }

    #[test]
    fn size_class_overflow_is_err() {
        // 既定 `PoolConfig`（huge_threshold=64MiB）では `u64::MAX` は巨大帯
        // （完全一致・丸め演算なし）に入るためオーバーフローしない。
        // 大帯の `checked_add` オーバーフロー経路を検証するため、
        // `huge_threshold` を `u64::MAX` に広げ、大帯の切り上げ演算
        // （`bytes + (granularity - 1)`）が確実にオーバーフローする値
        // （`u64::MAX`）を渡す。
        let config = PoolConfig {
            huge_threshold: u64::MAX,
            ..cfg()
        };
        assert!(size_class_for(u64::MAX, &config).is_err());
    }

    // --- 所有権契約 (a)(b)(c) ---

    #[test]
    fn take_miss_then_put_then_take_hit() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        assert_eq!(pool.take(256), None);
        let evicted = pool.put(256, 42u32);
        assert!(evicted.is_empty());
        assert_eq!(pool.take(256), Some(42));
        // 一度取り出したら同じクラスから二重に取れない。
        assert_eq!(pool.take(256), None);
    }

    #[test]
    fn put_over_limit_evicts_lru_as_vec() {
        let config = PoolConfig {
            max_pool_bytes: 512,
            ..cfg()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(config);
        assert!(pool.put(256, 1).is_empty());
        assert!(pool.put(256, 2).is_empty());
        // 3 件目（256*3=768 > 512）で最古（1）が破棄される。
        let evicted = pool.put(256, 3);
        assert_eq!(evicted, vec![1]);
        assert!(pool.stats().cached_bytes <= 512);
    }

    #[test]
    fn huge_band_respects_max_entries_per_class() {
        let config = PoolConfig {
            huge_max_entries_per_class: 1,
            ..cfg()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(config);
        let huge = config.huge_threshold + 1;
        assert!(pool.put(huge, 1).is_empty());
        // 2 件目は保持上限超過のためそのまま破棄対象として返る。
        let evicted = pool.put(huge, 2);
        assert_eq!(evicted, vec![2]);
    }

    // --- 統計契約 ---

    #[test]
    fn stats_track_alloc_and_reuse_counts() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        assert_eq!(pool.take(256), None); // alloc_count=1, reuse_count=0
        let _ = pool.put(256, 1);
        assert_eq!(pool.take(256), Some(1)); // alloc_count=2, reuse_count=1
        let stats = pool.stats();
        assert_eq!(stats.alloc_count, 2);
        assert_eq!(stats.reuse_count, 1);
    }

    #[test]
    fn stats_cached_bytes_matches_sum_of_class_bytes() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let _ = pool.put(256, 1);
        let _ = pool.put(512, 2);
        assert_eq!(pool.stats().cached_bytes, 256 + 512);
        pool.take(256);
        assert_eq!(pool.stats().cached_bytes, 512);
    }

    #[test]
    fn stats_pending_return_defaults_zero() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        assert_eq!(pool.stats().pending_return_bytes, 0);
        pool.record_pending_merge(100);
        assert_eq!(pool.stats().pending_return_bytes, 100);
        pool.record_pending_merge(-100);
        assert_eq!(pool.stats().pending_return_bytes, 0);
    }

    #[test]
    fn stats_released_bytes_accumulates_on_eviction() {
        let config = PoolConfig {
            max_pool_bytes: 256,
            ..cfg()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(config);
        let _ = pool.put(256, 1);
        let _ = pool.put(256, 2); // 1 を破棄
        assert_eq!(pool.stats().released_bytes, 256);
    }

    #[test]
    fn take_one_for_release_drains_lru_order() {
        let pool: SizeClassPool<u32> = SizeClassPool::new(cfg());
        let _ = pool.put(256, 1);
        let _ = pool.put(512, 2);
        let (bytes, handle) = pool.take_one_for_release().unwrap();
        assert_eq!(bytes, 256);
        assert_eq!(handle, 1);
        let (bytes2, handle2) = pool.take_one_for_release().unwrap();
        assert_eq!(bytes2, 512);
        assert_eq!(handle2, 2);
        assert!(pool.take_one_for_release().is_none());
        assert_eq!(pool.stats().cached_bytes, 0);
    }

    #[test]
    fn zero_max_pool_bytes_disables_pool_passthrough() {
        let config = PoolConfig {
            max_pool_bytes: 0,
            ..cfg()
        };
        let pool: SizeClassPool<u32> = SizeClassPool::new(config);
        let evicted = pool.put(256, 1);
        assert_eq!(evicted, vec![1]);
        assert_eq!(pool.stats().cached_bytes, 0);
    }

    // --- Send + Sync コンパイル時アサーション ---

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn size_class_pool_is_send_sync() {
        assert_send_sync::<SizeClassPool<u32>>();
    }
}
