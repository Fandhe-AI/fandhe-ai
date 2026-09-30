//! Metal `tape_build` 固定費の opt-in 削減機構と infer GPU 起動固定費の診断カウンタ
//! （イシュー #2114。設計・仮説・判定規則は `docs/perf/metal-tape-build-infer-fixedcost.md`）。
//!
//! # 役割
//!
//! - **デバイス存在確認キャッシュ**（opt-in・既定 OFF）: `facade::resolve_ops` の Metal 分岐は
//!   `tape_for(Device::Metal)`・`Sequential::predict_resident` の呼び出しごとに
//!   `MetalDeviceProvider::select`（`probe_all` = IOKit の GPU コア数取得 + `MTLCopyAllDevices`
//!   + `name`／`recommendedMaxWorkingSetSize`）を実行し、得た `DeviceInfo` を捨てて存在確認にしか
//!   使わない。v0.9.0 の M4 Max 実測で tape_build が 14.9〜15.7 µs（CPU 0.1 µs）を占める主因と
//!   推定される（**仮説**。フェーズ分解診断で検証する）。ON の間は「一度成功した存在確認」を
//!   プロセス内で再利用し 2 回目以降の probe を省く
//! - **診断カウンタ**（`Relaxed` の `fetch_add` のみ）: 存在確認 probe の実行・キャッシュヒット回数と
//!   ホスト⇔デバイス転送の回数・バイト数。infer の GPU 起動固定費（encode／command buffer／wait は
//!   既存の `__diagnostic_batch_counters_snapshot`〈#1099〉が担う）を切り分ける材料
//!
//! # 呼び出し元と配置理由
//!
//! 呼び出し元は `facade::resolve_ops`（[`verify_device_cached`]）と `memory.rs`
//! （`record_upload`／`record_download`）。objc2 に依存しない純 Rust の atomics／thread_local
//! だけで書き、`cfg(target_os = "macos")` を付けない（`batch_state`／`*_model.rs` と同じ判断。
//! Linux CI で状態機械の単体テストが回る）。
//!
//! # 契約
//!
//! - 既定 OFF（`METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED`）。OFF の間は従来どおり毎回 probe し
//!   `verified` フラグを読まない・書かない。A/B の after 側ではこの定数だけを反転する
//!   （`tensor-core::alloc::HOST_ARENA_DEFAULT_ENABLED` と同型の運用）
//! - **失敗はキャッシュしない**（fail-closed）: デバイス不在は毎回 probe し直して `Err` を返す
//! - スイッチは thread-local のスコープ付きガード（`!Send`）。環境変数などの隠れた有効化経路は
//!   作らない。`#[doc(hidden)] pub` は crate 間結線のための内部面で、facade からは再公開しない
//!   （`docs/compat-api-scope.md` §0）
//! - 残留リスク: `resolve_ops` を経由する `release_cached_memory`／`memory_pool_stats` も ON の間は
//!   2 回目以降の probe を省く。Apple Silicon の統合 GPU は消えないが、Intel Mac の eGPU 取り外しには
//!   追従しない。既定 OFF で、既定 ON 化は別イシューで A/B の後に判断する

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// キャッシュの既定有効フラグ。**既定 OFF**。A/B が ADOPT となった後の別 PR でのみ `true` に
/// 切り替える（after 側 worktree ではこの定数だけを反転する）。
pub(crate) const METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED: bool = false;

thread_local! {
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// スコープ限定の有効／無効上書きを元に戻す RAII ガード。
#[must_use = "ガードを drop すると上書きが元に戻る"]
pub struct DeviceVerifyCacheOverrideGuard {
    prev: Option<bool>,
    /// `!Send` 化。`OVERRIDE` は thread_local のため、別スレッドで drop されると作成元スレッドの
    /// 上書きが復元されない（`tensor-core::alloc::HostArenaOverrideGuard` と同じ理由）。
    _not_send: PhantomData<*const ()>,
}

impl Drop for DeviceVerifyCacheOverrideGuard {
    fn drop(&mut self) {
        OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// 現スレッドのキャッシュ有効／無効をスコープ限定で上書きする（テスト・診断用。ネスト可）。
#[doc(hidden)]
pub fn override_device_verify_cache_for_scope(enabled: bool) -> DeviceVerifyCacheOverrideGuard {
    let prev = OVERRIDE.with(|c| c.replace(Some(enabled)));
    DeviceVerifyCacheOverrideGuard {
        prev,
        _not_send: PhantomData,
    }
}

/// 現スレッドでキャッシュが有効か。TLS 破棄後は既定値へ倒す（`try_with`）。
#[doc(hidden)]
pub fn is_device_verify_cache_enabled() -> bool {
    OVERRIDE
        .try_with(Cell::get)
        .ok()
        .flatten()
        .unwrap_or(METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED)
}

/// 存在確認の成功済みフラグと診断カウンタを持つ状態機械。プロセスワイドの実体は
/// [`verify_device_cached`] が使う内部 static。テスト用にローカルインスタンスを作れる。
#[derive(Debug, Default)]
pub struct VerifyCache {
    verified: AtomicBool,
    probe_calls: AtomicU64,
    cache_hits: AtomicU64,
}

impl VerifyCache {
    /// 未確認状態の新規インスタンス。
    pub const fn new() -> Self {
        Self {
            verified: AtomicBool::new(false),
            probe_calls: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    /// `enabled` に従って `probe` を実行またはスキップする。
    ///
    /// - OFF: 常に `probe()` を実行する（`verified` は読まない・書かない）
    /// - ON: `verified` なら `Ok(())`（ヒット計上）。未確認なら `probe()` を実行し、`Ok` のときだけ
    ///   `verified` を立てる（失敗はキャッシュしない）
    pub fn verify_with<E>(
        &self,
        enabled: bool,
        probe: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        if enabled && self.verified.load(Ordering::Acquire) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.probe_calls.fetch_add(1, Ordering::Relaxed);
        let result = probe();
        if enabled && result.is_ok() {
            self.verified.store(true, Ordering::Release);
        }
        result
    }

    /// 実際に probe を実行した回数。
    pub fn probe_calls(&self) -> u64 {
        self.probe_calls.load(Ordering::Relaxed)
    }

    /// キャッシュで probe を省いた回数。
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// 成功済みフラグが立っているか。
    pub fn is_verified(&self) -> bool {
        self.verified.load(Ordering::Acquire)
    }
}

static GLOBAL_VERIFY_CACHE: VerifyCache = VerifyCache::new();

/// `facade::resolve_ops` の Metal 分岐が呼ぶデバイス存在確認。現スレッドの有効／無効
/// （[`is_device_verify_cache_enabled`]）に従い、プロセスワイドのキャッシュ越しに `probe` を実行する。
/// 既定 OFF では従来と同じく毎回 `probe` が実行される。
#[doc(hidden)]
pub fn verify_device_cached<E>(probe: impl FnOnce() -> Result<(), E>) -> Result<(), E> {
    GLOBAL_VERIFY_CACHE.verify_with(is_device_verify_cache_enabled(), probe)
}

static HOST_UPLOADS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_DOWNLOADS: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_BYTES: AtomicU64 = AtomicU64::new(0);

/// `memory.rs` の実アップロード点から呼ぶ（`Relaxed` 加算のみ。振る舞いは変えない）。
// 呼び出し元 `memory.rs` は macOS 限定モジュールのため、非 macOS では単体テストからのみ使われる。
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn record_upload(bytes: u64) {
    HOST_UPLOADS.fetch_add(1, Ordering::Relaxed);
    UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// `memory.rs` の実ダウンロード点から呼ぶ（`Relaxed` 加算のみ）。
// 呼び出し元 `memory.rs` は macOS 限定モジュールのため、非 macOS では単体テストからのみ使われる。
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn record_download(bytes: u64) {
    HOST_DOWNLOADS.fetch_add(1, Ordering::Relaxed);
    DOWNLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// 固定費カウンタのスナップショット（POD）。差分は [`Self::delta_since`]。
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FixedCostCountersSnapshot {
    /// デバイス存在確認で実際に probe を実行した回数。
    pub verify_probe_calls: u64,
    /// キャッシュで probe を省いた回数。
    pub verify_cache_hits: u64,
    /// ホスト→デバイスの実アップロード回数。
    pub host_uploads: u64,
    /// 同バイト数。
    pub upload_bytes: u64,
    /// デバイス→ホストの実ダウンロード回数。
    pub host_downloads: u64,
    /// 同バイト数。
    pub download_bytes: u64,
}

impl FixedCostCountersSnapshot {
    /// `earlier` からの増分（カウンタは単調増加。逆転時は 0 に飽和）。
    pub fn delta_since(&self, earlier: &Self) -> Self {
        Self {
            verify_probe_calls: self
                .verify_probe_calls
                .saturating_sub(earlier.verify_probe_calls),
            verify_cache_hits: self
                .verify_cache_hits
                .saturating_sub(earlier.verify_cache_hits),
            host_uploads: self.host_uploads.saturating_sub(earlier.host_uploads),
            upload_bytes: self.upload_bytes.saturating_sub(earlier.upload_bytes),
            host_downloads: self.host_downloads.saturating_sub(earlier.host_downloads),
            download_bytes: self.download_bytes.saturating_sub(earlier.download_bytes),
        }
    }
}

/// 現在の固定費カウンタを返す（テスト・ベンチ専用。encode／command buffer／wait 回数は
/// 既存の `__diagnostic_batch_counters_snapshot` を併用する）。
#[doc(hidden)]
pub fn __diagnostic_fixed_cost_counters_snapshot() -> FixedCostCountersSnapshot {
    FixedCostCountersSnapshot {
        verify_probe_calls: GLOBAL_VERIFY_CACHE.probe_calls(),
        verify_cache_hits: GLOBAL_VERIFY_CACHE.cache_hits(),
        host_uploads: HOST_UPLOADS.load(Ordering::Relaxed),
        upload_bytes: UPLOAD_BYTES.load(Ordering::Relaxed),
        host_downloads: HOST_DOWNLOADS.load(Ordering::Relaxed),
        download_bytes: DOWNLOAD_BYTES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        assert!(!is_device_verify_cache_enabled());
    }

    #[test]
    fn off_probes_every_time_and_never_sets_verified() {
        let c = VerifyCache::new();
        let mut n = 0;
        for _ in 0..3 {
            c.verify_with(false, || -> Result<(), ()> {
                n += 1;
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(n, 3);
        assert_eq!(c.probe_calls(), 3);
        assert_eq!(c.cache_hits(), 0);
        assert!(!c.is_verified());
    }

    #[test]
    fn on_probes_once_then_hits() {
        let c = VerifyCache::new();
        let mut n = 0;
        for _ in 0..4 {
            c.verify_with(true, || -> Result<(), ()> {
                n += 1;
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(n, 1);
        assert_eq!(c.probe_calls(), 1);
        assert_eq!(c.cache_hits(), 3);
        assert!(c.is_verified());
    }

    #[test]
    fn on_does_not_cache_failure() {
        let c = VerifyCache::new();
        let mut n = 0;
        for _ in 0..2 {
            let r = c.verify_with(true, || {
                n += 1;
                Err("no device")
            });
            assert_eq!(r, Err("no device"));
        }
        assert_eq!(n, 2);
        assert!(!c.is_verified());
        assert_eq!(c.cache_hits(), 0);
        // 復旧後の成功は記録され、以降はヒットする。
        c.verify_with(true, || -> Result<(), &str> { Ok(()) })
            .unwrap();
        c.verify_with(true, || -> Result<(), &str> { Err("unreachable") })
            .unwrap();
        assert_eq!(c.cache_hits(), 1);
    }

    #[test]
    fn off_success_does_not_prime_later_on() {
        let c = VerifyCache::new();
        c.verify_with(false, || -> Result<(), ()> { Ok(()) })
            .unwrap();
        let mut n = 0;
        c.verify_with(true, || -> Result<(), ()> {
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 1, "OFF 中の成功は ON 初回の probe を省かせない");
    }

    #[test]
    fn override_guard_nests_and_restores() {
        assert!(!is_device_verify_cache_enabled());
        {
            let _a = override_device_verify_cache_for_scope(true);
            assert!(is_device_verify_cache_enabled());
            {
                let _b = override_device_verify_cache_for_scope(false);
                assert!(!is_device_verify_cache_enabled());
            }
            assert!(is_device_verify_cache_enabled());
        }
        assert!(!is_device_verify_cache_enabled());
    }

    #[test]
    fn counters_are_monotonic_and_delta_works() {
        let before = __diagnostic_fixed_cost_counters_snapshot();
        record_upload(16);
        record_upload(8);
        record_download(4);
        let after = __diagnostic_fixed_cost_counters_snapshot();
        let d = after.delta_since(&before);
        // 他テストの並行加算がありうるため下限で検証する。
        assert!(d.host_uploads >= 2 && d.upload_bytes >= 24);
        assert!(d.host_downloads >= 1 && d.download_bytes >= 4);
        assert_eq!(
            before.delta_since(&after),
            FixedCostCountersSnapshot::default()
        );
    }
}
