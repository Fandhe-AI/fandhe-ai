//! readback 宛先ポリシー（イシュー #2108）。`memory::readback`（全 D2H readback が通る唯一の
//! 同期点）が返すホスト `Vec` を「毎回 fresh + 事前タッチ（`PretouchedFresh`）」で作るか、
//! 「pinned（page-locked）staging を ordinal・要素数ごとに再利用して copy-out（`PinnedStagingReuse`）」
//! で作るかを選ぶ純ロジックと、env 切替・staging プールをまとめる。
//!
//! # 位置づけ
//! - 呼び出し元: `memory::readback`（`current_dest()` で宛先方式を解決）と
//!   `memory::ReadbackSentinel::readback_pinned_reuse`（f32 のみ `ReadbackStagingPool` を使う）。
//! - staging バッファ実体・キャッシュ契約（世代不一致の fail-closed 破棄・cap 超過で登録しない）は
//!   `host_staging::{HostStaging, HostStagingCache}` を流用する。pinned 確保の `unsafe` は既存の
//!   `HostStaging::alloc` 1 箇所のみで、本モジュールは `unsafe` を持たない（新規 `unsafe` なし）。
//!
//! # 仮説と非目標
//! - 仮説 H-reuse: pageable 宛先への D2H（driver 内部 bounce・ページフォールト）と事前フィルを、
//!   pinned への直接 DMA と CPU の copy-out 1 回へ置き換えれば `matmul` 区間の露出コストが減る。
//!   #2107 の帰属判定（`docs/perf/logs/cuda-gemm-readback-attribution-2107`）が前提ゲート。
//! - #1436 の `PretouchedReusedDest`（pageable 再利用宛先）とは別の実験（宛先が pinned）。
//! - 真の宛先再利用（copy-out なし）・ゼロコピーは `Tensor` の `Arc<Storage{Vec<T>}>` に drop フックが
//!   なく返却できないため対象外（tensor-core のストレージ抽象変更と新規 `unsafe` が要りユーザー承認事項）。
//!
//! # 既定 OFF・数値契約
//! 既定は `PretouchedFresh`（`memory::READBACK_DEST`）。`PinnedStagingReuse` も D2H が要素を全て
//! 上書きしてから `to_vec()` するため出力は bit 同一で、tolerance・REQ-2 判定には影響しない。
//! ADOPT 後の既定値化は別 PR・ユーザー承認（pinned 確保の到達範囲が全 f32 readback に広がるため）。

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::host_staging::{HostStaging, HostStagingCache, HostStagingKind};
use crate::memory::{READBACK_DEST, ReadbackDest};

/// readback 宛先ポリシーを選ぶ env 変数名。許容値は `pretouched`／`pinned-reuse`（完全一致）のみ。
pub(crate) const READBACK_DEST_ENV: &str = "FANDHE_AI_CUDA_READBACK_DEST";

/// env 値の解釈。完全一致の allowlist のみ受理し、未知値は `None`（呼び出し側が既定へ倒す）。
/// 値はログ・エラーへエコーしない（A03）。
pub(crate) fn parse_env_value(raw: &str) -> Option<ReadbackDest> {
    match raw {
        "pretouched" => Some(ReadbackDest::PretouchedFresh),
        "pinned-reuse" => Some(ReadbackDest::PinnedStagingReuse),
        _ => None,
    }
}

fn env_mode() -> ReadbackDest {
    static MODE: OnceLock<ReadbackDest> = OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var(READBACK_DEST_ENV)
            .ok()
            .and_then(|v| parse_env_value(&v))
            .unwrap_or(READBACK_DEST)
    })
}

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<ReadbackDest>> =
        const { std::cell::Cell::new(None) };
}

/// テスト用 scoped override（thread-local・RAII。入れ子は drop で直前値へ復元する）。
#[cfg(test)]
pub(crate) struct DestOverrideGuard {
    prev: Option<ReadbackDest>,
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(test)]
impl DestOverrideGuard {
    pub(crate) fn new(dest: ReadbackDest) -> Self {
        let prev = TEST_OVERRIDE.with(|c| c.replace(Some(dest)));
        Self {
            prev,
            _not_send: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for DestOverrideGuard {
    fn drop(&mut self) {
        TEST_OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// 現在有効なポリシー（テスト override > env > `memory::READBACK_DEST`）。
pub(crate) fn current_dest() -> ReadbackDest {
    #[cfg(test)]
    if let Some(d) = TEST_OVERRIDE.with(|c| c.get()) {
        return d;
    }
    env_mode()
}

/// ordinal ごとの staging プール（要素数ごとに 1 バッファ。各 ordinal の総量は
/// `HOST_STAGING_CAP_BYTES` で頭打ち）。ロック保持は `take`／`put` の間のみで、
/// pinned 確保・D2H・synchronize はロック外で行う。
pub(crate) struct ReadbackStagingPool {
    kind: HostStagingKind,
    per_ordinal: Mutex<HashMap<usize, HostStagingCache>>,
    #[cfg(test)]
    cap_bytes: Option<u64>,
}

impl ReadbackStagingPool {
    pub(crate) fn new(kind: HostStagingKind) -> Self {
        Self {
            kind,
            per_ordinal: Mutex::new(HashMap::new()),
            #[cfg(test)]
            cap_bytes: None,
        }
    }

    /// テスト専用: ordinal ごとの cap を小さく差し替える（巨大確保を避けるため）。
    #[cfg(test)]
    fn with_cap(kind: HostStagingKind, cap_bytes: u64) -> Self {
        Self {
            cap_bytes: Some(cap_bytes),
            ..Self::new(kind)
        }
    }

    fn new_cache(&self) -> HostStagingCache {
        #[cfg(test)]
        if let Some(cap) = self.cap_bytes {
            return HostStagingCache::with_cap(self.kind, cap);
        }
        HostStagingCache::new(self.kind)
    }

    /// 世代・要素数が一致する staging を取り出す。poison 時は miss 扱い（新規確保へ倒す。
    /// D2H が全要素を上書きするため正しさには影響しない）。
    pub(crate) fn take(
        &self,
        ordinal: usize,
        generation: u64,
        numel: usize,
    ) -> Option<HostStaging> {
        let mut guard = self.per_ordinal.lock().ok()?;
        guard.get_mut(&ordinal)?.take(numel, generation)
    }

    /// 使用済み staging を返却する（cap 超過は登録せず破棄）。poison 後も `into_inner` で回復する。
    pub(crate) fn put(&self, ordinal: usize, generation: u64, numel: usize, buf: HostStaging) {
        let mut guard = match self.per_ordinal.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard
            .entry(ordinal)
            .or_insert_with(|| self.new_cache())
            .put(numel, generation, buf);
    }

    /// 全 ordinal の staging を破棄し、解放したバイト数を返す（本番用の明示解放経路。
    /// `CudaMemory::release_readback_staging` から呼ばれる。REQ-14 `release_cached` 系と同型）。
    pub(crate) fn release_all(&self) -> u64 {
        let mut guard = match self.per_ordinal.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.values_mut().map(|c| c.release_all()).sum()
    }

    /// hit／miss／evicted／保持バイト数の合計。
    #[cfg(any(test, feature = "internal-diagnostics"))]
    pub(crate) fn stats(&self) -> crate::host_staging::HostStagingStats {
        let guard = match self.per_ordinal.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.values().map(|c| c.stats()).fold(
            crate::host_staging::HostStagingStats::default(),
            |a, s| crate::host_staging::HostStagingStats {
                hits: a.hits + s.hits,
                misses: a.misses + s.misses,
                evicted: a.evicted + s.evicted,
                cached_bytes: a.cached_bytes + s.cached_bytes,
            },
        )
    }
}

/// 本番のプロセスワイド pool（pinned 種別）。`PinnedStagingReuse` 選択時のみ確保が発生する。
pub(crate) fn readback_staging_pool() -> &'static ReadbackStagingPool {
    static POOL: OnceLock<ReadbackStagingPool> = OnceLock::new();
    POOL.get_or_init(|| ReadbackStagingPool::new(crate::host_staging::HOST_STAGING_KIND))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_allowlist_is_exact_match_only() {
        assert_eq!(
            parse_env_value("pretouched"),
            Some(ReadbackDest::PretouchedFresh)
        );
        assert_eq!(
            parse_env_value("pinned-reuse"),
            Some(ReadbackDest::PinnedStagingReuse)
        );
        for bad in [
            "",
            "Pinned-Reuse",
            " pinned-reuse",
            "pinned-reuse ",
            "1",
            "true",
            "pinned",
            "fresh",
        ] {
            assert_eq!(parse_env_value(bad), None, "{bad:?}");
        }
    }

    /// 既定定数を直接検査するドリフトガード（setter・env を経由しない。#1585 の教訓）。
    #[test]
    fn default_dest_is_pretouched_fresh() {
        assert_eq!(READBACK_DEST, ReadbackDest::PretouchedFresh);
    }

    #[test]
    fn override_guard_nests_and_restores() {
        let base = current_dest();
        {
            let _a = DestOverrideGuard::new(ReadbackDest::PinnedStagingReuse);
            assert_eq!(current_dest(), ReadbackDest::PinnedStagingReuse);
            {
                let _b = DestOverrideGuard::new(ReadbackDest::PretouchedFresh);
                assert_eq!(current_dest(), ReadbackDest::PretouchedFresh);
            }
            assert_eq!(current_dest(), ReadbackDest::PinnedStagingReuse);
        }
        assert_eq!(current_dest(), base);
    }

    fn pageable(n: usize) -> HostStaging {
        HostStaging::Pageable(vec![0.0f32; n])
    }

    #[test]
    fn pool_separates_ordinals_and_reuses() {
        let pool = ReadbackStagingPool::new(HostStagingKind::Pageable);
        assert!(pool.take(0, 0, 8).is_none());
        pool.put(0, 0, 8, pageable(8));
        assert!(pool.take(1, 0, 8).is_none(), "ordinal 別に分離");
        assert!(pool.take(0, 0, 8).is_some());
        let s = pool.stats();
        assert_eq!(s.hits, 1);
    }

    #[test]
    fn pool_drops_stale_generation() {
        let pool = ReadbackStagingPool::new(HostStagingKind::Pageable);
        pool.put(0, 0, 8, pageable(8));
        assert!(pool.take(0, 1, 8).is_none(), "世代不一致は破棄");
        assert_eq!(pool.stats().evicted, 1);
        assert_eq!(pool.stats().cached_bytes, 0);
    }

    #[test]
    fn pool_release_all_reports_bytes() {
        let pool = ReadbackStagingPool::new(HostStagingKind::Pageable);
        pool.put(0, 0, 8, pageable(8));
        pool.put(1, 0, 4, pageable(4));
        assert_eq!(pool.release_all(), 48);
        assert_eq!(pool.stats().cached_bytes, 0);
    }

    #[test]
    fn pool_cap_overflow_is_not_registered() {
        let pool = ReadbackStagingPool::with_cap(HostStagingKind::Pageable, 16);
        let n = 16 / 4 + 1;
        pool.put(0, 0, n, pageable(n));
        assert_eq!(pool.stats().cached_bytes, 0);
        assert_eq!(pool.stats().evicted, 1);
    }
}
