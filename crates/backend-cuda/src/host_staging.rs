//! `CudaMemory::with_host_view`（[`MemoryOps::with_host_view`] の CUDA
//! 実装。イシュー #1336）が使うホストステージングバッファのキャッシュ。
//!
//! ## 背景
//!
//! `MemoryOps::with_host_view` の既定実装（`tensor_core::buffer` モジュール
//! コメント）は毎回 `download`（`readback` 経由の `clone_dtoh` が**都度新規
//! `Vec<f32>` を確保**）→ `Tensor::new` → `as_slice()` という経路を辿る。
//! `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
//! （§4.2・§6）は、D2H の宛先を毎回新規確保する構成（P4）のみが 31→32 MiB
//! で 24〜33 倍の段差（glibc `M_MMAP_THRESHOLD` 動的上限）を示す一方、
//! **事前タッチ済み再利用 `Vec`（P5）は線形・段差なし**であることを実測
//! 済みである。本モジュールは形状（要素数）ごとに再利用するホスト
//! ステージングバッファへ `memcpy_dtoh` 1 回で D2H し、そのスライスを
//! `with_host_view` の呼び出し元クロージャへ借用として渡すことで、この
//! 段差を機構として回避する。
//!
//! ## 種別（[`HostStagingKind`]）
//!
//! `Pinned`（cudarc `CudaContext::alloc_pinned`。page-locked・
//! `CU_MEMHOSTALLOC_WRITECOMBINED` 固定・**unsafe 1 箇所**）と
//! `Pageable`（事前タッチ済み `Vec<f32>`。unsafe なし）の 2 種を実装し、
//! 実測（`docs/perf/cuda-host-view-staging-readout.md`）で本番種を選ぶ
//! 設計とした。WRITECOMBINED メモリは GPU 側の DMA 書き込み先としては
//! 有利だが CPU 側の読み出しが著しく遅いことが知られており（cudarc-0.19.8
//! `core.rs:1406-1427`）、D2H 自体は速くても「D2H＋ホスト読み出し」の
//! 合計では `Pinned` が `Pageable` に劣る可能性がある（決め打ちしない）。
//!
//! **GB10 実機実測（2026-09-08。`docs/perf/cuda-host-view-staging-readout.md`
//! §5）で `Pinned` が全計測形状（N=1024/2048/4096）で `Pageable` を一貫して
//! 上回ることを確認したが、既定 [`HOST_STAGING_KIND`] は unsafe 経路を
//! 通さない安全側（`Pageable`）に固定したまま維持している**（同 doc
//! 「採否」節。unsafe 経路の既定化はユーザー承認事項のため、性能が
//! 上回るだけでは切り替えない方針）。`Pinned` 経路自体は実装・GPU 非依存
//! 単体テスト・`#[ignore]` 実機テストとも整備済みで、ユーザー承認を
//! 経て切り替える想定。
//!
//! ## 同期契約
//!
//! `MemoryOps::with_host_view` の呼び出し元契約（`tensor_core::buffer`
//! モジュールの同トレイトのドキュメンテーションコメント）と同一:
//! 「復帰時点でデバイス側の書き込みが完了していること」。実際の
//! `memcpy_dtoh`／`synchronize` 呼び出しは `memory.rs::CudaMemory::
//! with_host_view` が `with_driver_call`（poison／世代状態機械）経由で
//! 行う。本モジュール自身は driver を直接呼ばない（`HostStaging::alloc`
//! の `Pinned` 確保を除く）。
//!
//! ## ロック方針（take/put 方式）
//!
//! [`HostStagingCache`] の `Mutex` は `take`／`put` の間（データ構造操作
//! のみ）だけ保持し、`memcpy_dtoh`・`synchronize`・呼び出し元クロージャ
//! `f` の実行中は保持しない（`memory.rs::CudaMemory::with_host_view` の
//! 実装参照）。これにより `f` の内部で別バッファの `with_host_view` を
//! 再入しても deadlock しない（同一 numel の再入は新規確保で対応する）。

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, HostSlice, PinnedHostSlice};

use crate::error::CudaError;

/// 本番経路が使うステージング種別（実測前は unsafe 経路を通さない
/// `Pageable` を既定とする。モジュール冒頭コメント「種別」節）。
pub(crate) const HOST_STAGING_KIND: HostStagingKind = HostStagingKind::Pageable;

/// [`HostStagingCache`] が保持するホストバッファ合計の確保上限
/// （バイト）。超過分は使用後に破棄する（`.claude/rules/security.md`
/// 「リソース枯渇（DoS）」対策。page-locked メモリはホスト RAM を固定
/// するため特に重要）。4096×4096 の f32（64 MiB）を複数形状ぶん保持
/// できる程度の余裕を持たせた既定値。
pub(crate) const HOST_STAGING_CAP_BYTES: u64 = 256 * 1024 * 1024;

/// ホストステージングバッファの実装種別。`Pageable`・`Pinned` いずれも
/// `pub`（`pub(crate)` から変更。codex-review 指摘: 本番既定
/// （[`HOST_STAGING_KIND`]）は `Pageable` に固定されているため、`Pinned`
/// 経路を実機で検証する・両者を A/B 比較する手段が診断入口から
/// 提供されていなかった）で、`internal-diagnostics` feature（既定
/// off）限定の `memory::CudaMemory::with_host_view_using_kind`
/// （`lib.rs` の `pub use host_staging::HostStagingKind` re-export と
/// 対）から crate 外部（実機 `#[ignore]` テスト）が明示的に選べる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostStagingKind {
    /// cudarc `alloc_pinned`（page-locked・WRITECOMBINED）。unsafe 1 箇所。
    ///
    /// GB10 実機実測（`docs/perf/cuda-host-view-staging-readout.md`
    /// §5・2026-09-08）で本 variant が全計測形状（N=1024/2048/4096）で
    /// `Pageable` を一貫して上回ることを確認済みだが、本番既定
    /// [`HOST_STAGING_KIND`] は unsafe 経路の既定化がユーザー承認事項
    /// であるため本 variant を選ばない（常に `Pageable`）。
    /// `memory::CudaMemory::with_host_view_using_kind`（`internal-
    /// diagnostics` feature 限定）を介して明示的に選べば `HostStaging::
    /// alloc` の `match` アームへ到達し、`tests/host_view_real_device.rs`
    /// の実機 `#[ignore]` テストが実際に `Pinned` 経路を通す（ユーザー
    /// 承認のうえ本番既定へ切り替える際は本コメントを更新する）。
    Pinned,
    /// 事前タッチ済み pageable `Vec<f32>`（unsafe なし）。
    Pageable,
}

/// ホストステージングバッファ本体（種別ごとの実データを保持する）。
pub(crate) enum HostStaging {
    Pinned(PinnedHostSlice<f32>),
    Pageable(Vec<f32>),
}

impl HostStaging {
    fn len(&self) -> usize {
        match self {
            HostStaging::Pinned(p) => p.len(),
            HostStaging::Pageable(v) => v.len(),
        }
    }

    fn byte_len(&self) -> u64 {
        (self.len() * std::mem::size_of::<f32>()) as u64
    }

    /// `CudaStream::memcpy_dtoh` の `dst: &mut Dst where Dst: HostSlice<T>
    /// + ?Sized` へそのまま渡せるトレイトオブジェクト参照（両 variant を
    /// 統一的に扱う。`cudarc::driver::HostSlice` は object-safe な設計
    /// のため `dyn HostSlice<f32>` として扱える）。
    pub(crate) fn as_host_slice_mut(&mut self) -> &mut dyn HostSlice<f32> {
        match self {
            HostStaging::Pinned(p) => p,
            HostStaging::Pageable(v) => v,
        }
    }

    /// 転送後のホスト側読み出し（`with_host_view` の呼び出し元クロージャ
    /// へ渡すスライス）。呼び出し元は `memcpy_dtoh` 直後に必ず
    /// `stream.synchronize()` を経由済みであること（`memory.rs::
    /// CudaMemory::with_host_view` の実装が保証する呼び出し順）。`Pinned`
    /// 側の `as_slice()` 内部 `event.synchronize()` は、その `synchronize`
    /// 後は既に完了済みの event を待つだけの no-op に近い二重待ちであり
    /// 追加コストは無視できる。
    pub(crate) fn as_slice(&self) -> Result<&[f32], CudaError> {
        match self {
            HostStaging::Pinned(p) => Ok(p.as_slice()?),
            HostStaging::Pageable(v) => Ok(v.as_slice()),
        }
    }

    /// `kind` に応じて `numel` 要素ぶんのステージングバッファを新規確保
    /// する（キャッシュ miss 時。`HostStagingCache::take` が `None` を
    /// 返した場合に `memory.rs` から呼ばれる）。
    pub(crate) fn alloc(
        kind: HostStagingKind,
        ctx: &Arc<CudaContext>,
        numel: usize,
    ) -> Result<Self, CudaError> {
        match kind {
            HostStagingKind::Pageable => {
                // 事前タッチ（ゼロ初期化確保）: #1146 P5「事前タッチ済み
                // 再利用 Vec」相当。確保直後に全要素へ書き込むため、初回
                // ページフォールトを本関数内で消化してから返す。
                Ok(HostStaging::Pageable(vec![0.0f32; numel]))
            }
            HostStagingKind::Pinned => {
                // SAFETY: `CudaContext::alloc_pinned` が unsafe な唯一の
                // 理由（cudarc-0.19.8 `core.rs:1410-1411` "この呼び出しの
                // 後、メモリは未初期化"）は、返す `PinnedHostSlice` の
                // 内容が確保直後は不定であること。本関数はキャッシュ
                // miss 時にのみ呼ばれ、呼び出し元
                // （`memory.rs::CudaMemory::with_host_view`）は返った
                // バッファを `f`（呼び出し元クロージャ）へ渡す前に必ず
                // `CudaStream::memcpy_dtoh` で要素数ぶん全域を D2H 上書き
                // してから `as_slice()` で読み出す（未初期化内容が外部へ
                // 露出する経路は存在しない）。要素型は `f32` であり
                // 全ビットパターンが有効な浮動小数点表現になるため
                // （`cudarc::driver::ValidAsZeroBits` が `f32` に実装
                // 済み。`memory.rs::alloc_zeroed_inner` の managed 確保
                // 分岐と同種の安全性根拠）、未初期化ビット列を無効値と
                // して解釈する余地もない。
                let pinned = unsafe { ctx.alloc_pinned::<f32>(numel)? };
                Ok(HostStaging::Pinned(pinned))
            }
        }
    }
}

/// [`HostStagingCache::take`]／[`put`](HostStagingCache::put) の呼び出し
/// 回数・キャッシュ状態を集計する診断用スナップショット（テスト・
/// 実機診断向け。`pub` だが値の生成は本モジュール限定）。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HostStagingStats {
    /// `take` が既存エントリを返せた回数。
    pub hits: u64,
    /// `take` が新規確保を要求した回数（世代・長さ不一致による破棄も
    /// 含む。`fandhe_ai_backend_cuda` クレート外からは判別しない）。
    pub misses: u64,
    /// 世代・長さ不一致またはキャッシュ上限超過により破棄したエントリ数。
    pub evicted: u64,
    /// 現在キャッシュが保持しているバイト数の合計。
    pub cached_bytes: u64,
}

/// キャッシュが保持する 1 エントリ（バッファ本体＋確保時点の ordinal 世代）。
struct StagingEntry {
    buf: HostStaging,
    generation: u64,
}

/// 形状（要素数）ごとに再利用するホストステージングバッファのキャッシュ
/// 本体。`CudaMemory::host_staging`（`Arc<Mutex<Self>>`。モジュール冒頭
/// 「ロック方針」節）が保持し、`with_host_view` から `take`／`put` される。
pub(crate) struct HostStagingCache {
    entries: HashMap<usize, StagingEntry>,
    cached_bytes: u64,
    cap_bytes: u64,
    kind: HostStagingKind,
    stats: HostStagingStats,
}

impl HostStagingCache {
    pub(crate) fn new(kind: HostStagingKind) -> Self {
        Self {
            entries: HashMap::new(),
            cached_bytes: 0,
            cap_bytes: HOST_STAGING_CAP_BYTES,
            kind,
            stats: HostStagingStats::default(),
        }
    }

    #[cfg(test)]
    fn with_cap(kind: HostStagingKind, cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            ..Self::new(kind)
        }
    }

    pub(crate) fn kind(&self) -> HostStagingKind {
        self.kind
    }

    /// 統計スナップショットを返す。`cached_bytes` は `take`／`put` の
    /// たびに `self.stats.cached_bytes` を個別更新するのではなく、
    /// 読み出し時に `self.cached_bytes`（唯一の真実源）から合成する
    /// （2 箇所を独立更新して drift させない設計）。
    pub(crate) fn stats(&self) -> HostStagingStats {
        HostStagingStats {
            cached_bytes: self.cached_bytes,
            ..self.stats
        }
    }

    /// `numel`・`generation` に一致する既存エントリを取り出す
    /// （世代不一致・長さ不一致は fail-closed に破棄して miss 扱いに
    /// する。`context_cache::invalidate` はストリームの世代を進めるが
    /// `CudaMemory` 自体は再生成されないため、旧世代のステージング
    /// バッファを新世代の D2H へ誤って使い回すことを防ぐ。実装計画
    /// 「スコープ上の重要事実」item 7）。
    pub(crate) fn take(&mut self, numel: usize, generation: u64) -> Option<HostStaging> {
        match self.entries.remove(&numel) {
            Some(entry) if entry.generation == generation && entry.buf.len() == numel => {
                self.cached_bytes = self.cached_bytes.saturating_sub(entry.buf.byte_len());
                self.stats.hits += 1;
                Some(entry.buf)
            }
            Some(stale) => {
                self.cached_bytes = self.cached_bytes.saturating_sub(stale.buf.byte_len());
                self.stats.misses += 1;
                self.stats.evicted += 1;
                None
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// 使用済みバッファをキャッシュへ返却する。`cap_bytes` を超える
    /// 場合は登録せず破棄する（DoS 対策。モジュール冒頭コメント参照）。
    ///
    /// 同一 `numel` キーへ既存エントリがある状態（再入・並行
    /// `with_host_view` により同一形状のバッファが `take` されずに
    /// 複数回 `put` されうる）で `HashMap::insert` が既存エントリを
    /// 置換する場合、置換前の既存分バイト数を `cached_bytes` から
    /// 差し引いてから新規分を加算する。差し引かずに加算のみを行うと
    /// `cached_bytes` が実際の保持量（`entries` の総バイト数）より
    /// 過大計上され、以降の上限判定（本メソッド冒頭の cap 比較）が
    /// 不当に返却を拒否し、`release_all`（`CudaMemory::
    /// release_host_staging` 経由）の返却値も過大になる
    /// （codex-review 指摘 P2・Cursor Bugbot Medium 指摘。両者は
    /// 同一箇所・同一問題）。
    fn put(&mut self, numel: usize, generation: u64, buf: HostStaging) {
        let bytes = buf.byte_len();
        let existing_bytes = self
            .entries
            .get(&numel)
            .map(|entry| entry.buf.byte_len())
            .unwrap_or(0);
        let projected_cached_bytes = self
            .cached_bytes
            .saturating_sub(existing_bytes)
            .saturating_add(bytes);
        if projected_cached_bytes > self.cap_bytes {
            self.stats.evicted += 1;
            return;
        }
        self.cached_bytes = projected_cached_bytes;
        self.entries.insert(numel, StagingEntry { buf, generation });
    }

    /// 全エントリを破棄し解放したバイト数を返す（REQ-14
    /// `release_cached` 系と同型の明示解放 API。
    /// `CudaMemory::release_host_staging` から呼ばれる）。
    pub(crate) fn release_all(&mut self) -> u64 {
        let bytes = self.cached_bytes;
        self.entries.clear();
        self.cached_bytes = 0;
        bytes
    }
}

/// 使用済みバッファをキャッシュへ返却する（`memory.rs::CudaMemory::
/// return_staging` から呼ばれる。poison 後も `into_inner` で返却を試み
/// panic しない一方、以降の新規取得〈`memory.rs::CudaMemory::
/// take_cached_staging` が直接 `lock()` する経路〉は通常どおり poison
/// エラーとして fail-closed に拒否される）。
pub(crate) fn put_back(
    cache: &std::sync::Mutex<HostStagingCache>,
    numel: usize,
    generation: u64,
    buf: HostStaging,
) {
    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.put(numel, generation, buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    // `HostStaging::alloc(Pinned, ..)` の miss→alloc 経路そのものは
    // `CudaContext`（実 driver 初期化）を要求するため CUDA 実機テスト側
    // （`tests/host_view_real_device.rs`）で検証する。本モジュールの
    // GPU 非依存テストは `HostStaging::Pageable` の直接構築・
    // `HostStagingCache` の `take`／`put` ロジック・poison 処理に限定
    // する。

    #[test]
    fn host_staging_kind_variants_are_distinct() {
        // `HostStagingKind::Pinned` は実機（`CudaContext`）を要求する
        // 経路でのみ構築されるため（`HostStaging::alloc`・`tests/
        // host_view_real_device.rs`）、GPU 非依存の通常ビルドでは
        // このテストが唯一の構築箇所になる（dead_code 検査対策では
        // なく、`Eq`／`Clone` 実装そのものの契約検証を兼ねる）。
        assert_ne!(HostStagingKind::Pinned, HostStagingKind::Pageable);
        assert_eq!(HostStagingKind::Pinned, HostStagingKind::Pinned.clone());
    }

    #[test]
    fn pageable_staging_is_zero_filled_and_correct_length() {
        let staging = HostStaging::Pageable(vec![0.0f32; 8]);
        assert_eq!(staging.len(), 8);
        assert_eq!(staging.byte_len(), 32);
        assert_eq!(staging.as_slice().unwrap(), &[0.0f32; 8]);
    }

    #[test]
    fn cache_take_miss_then_put_then_take_hit() {
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        assert!(cache.take(4, 0).is_none());
        assert_eq!(cache.stats().misses, 1);

        cache.put(4, 0, HostStaging::Pageable(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(cache.stats().cached_bytes, 16);

        let hit = cache.take(4, 0).expect("同一 numel・世代は hit するはず");
        assert_eq!(hit.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(
            cache.stats().cached_bytes,
            0,
            "take 後は cached_bytes から差し引かれる"
        );
    }

    #[test]
    fn cache_take_discards_entry_with_mismatched_generation() {
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));

        // 世代 1 で取り出そうとすると、格納時の世代 0 と不一致のため
        // 破棄されて miss 扱いになる（fail-closed。`invalidate` 後の
        // 旧世代バッファ誤使用防止）。
        let result = cache.take(4, 1);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().evicted, 1);
        assert_eq!(cache.stats().cached_bytes, 0);

        // 破棄済みのため、同じ numel を世代 0 で再度取り出しても miss。
        assert!(cache.take(4, 0).is_none());
    }

    #[test]
    fn cache_take_discards_entry_with_mismatched_length() {
        // `HashMap` のキーは numel のため通常長さは一致するが、内部
        // 不整合（本来到達しない防御的経路）を模して直接検証する。
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.entries.insert(
            4,
            StagingEntry {
                buf: HostStaging::Pageable(vec![0.0; 3]),
                generation: 0,
            },
        );
        assert!(cache.take(4, 0).is_none());
        assert_eq!(cache.stats().evicted, 1);
    }

    #[test]
    fn cache_put_discards_entries_exceeding_cap() {
        let mut cache = HostStagingCache::with_cap(HostStagingKind::Pageable, 16);
        // 16 バイト（numel=4）はちょうど収まる。
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        // 追加で 8 バイト（numel=2）を put すると cap（16 バイト）を
        // 超過するため破棄される（既存エントリは維持）。
        cache.put(2, 0, HostStaging::Pageable(vec![0.0; 2]));
        assert_eq!(cache.stats().cached_bytes, 16, "cap 超過分は登録されない");
        assert_eq!(cache.stats().evicted, 1);
        assert!(cache.take(2, 0).is_none(), "cap 超過で破棄されたため miss");
    }

    #[test]
    fn cache_put_same_numel_twice_does_not_overcount_cached_bytes() {
        // 回帰テスト（codex-review P2 指摘・Cursor Bugbot Medium 指摘。
        // 同一箇所・同一問題）: 同一 numel キーへ `take` を挟まず 2 回
        // `put` すると、`HashMap::insert` が既存エントリを置換する際に
        // 旧エントリ分のバイト数を差し引かずに新規分だけ加算していた
        // ため `cached_bytes` が実体（`entries` の総バイト数）より過大
        // 計上されていた。修正後は置換時に旧分を差し引くため、2 回目の
        // `put` 後も `cached_bytes` は最新エントリの実バイト数と一致
        // する。
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        // take を挟まず同一 numel へ再入で put（並行 with_host_view を
        // 模す）。旧エントリが破棄され新エントリに置換される。
        cache.put(4, 1, HostStaging::Pageable(vec![1.0; 4]));
        assert_eq!(
            cache.stats().cached_bytes,
            16,
            "同一 numel の置換では合計バイト数は変わらないはず（過大計上の回帰検知）"
        );

        // 実体との整合を release_all の返却値でも確認する。
        let freed = cache.release_all();
        assert_eq!(freed, 16, "release_all の返却値も実体と一致するはず");
    }

    #[test]
    fn cache_put_same_numel_twice_respects_cap_with_replacement() {
        // 置換時に旧分を差し引いたうえで cap 判定することを検証する
        // （旧分を差し引かずに加算のみだと、実際には cap 内に収まる
        // 置換でも不当に拒否されうる）。
        let mut cache = HostStagingCache::with_cap(HostStagingKind::Pageable, 16);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        assert_eq!(cache.stats().cached_bytes, 16);

        // 同一 numel（同じ 16 バイト）への置換は cap ちょうどのため
        // 受理されるはず（旧分を差し引かず加算のみだと 32 > 16 で
        // 不当に拒否されていた）。
        cache.put(4, 1, HostStaging::Pageable(vec![1.0; 4]));
        assert_eq!(
            cache.stats().cached_bytes,
            16,
            "置換は cap 内で受理されるはず"
        );
        assert_eq!(
            cache.stats().evicted,
            0,
            "置換は cap 超過ではないため evicted は増えない"
        );

        let hit = cache.take(4, 1).expect("置換後の世代 1 で take できるはず");
        assert_eq!(hit.as_slice().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn release_all_clears_cache_and_returns_freed_bytes() {
        let mut cache = HostStagingCache::new(HostStagingKind::Pageable);
        cache.put(4, 0, HostStaging::Pageable(vec![0.0; 4]));
        cache.put(8, 0, HostStaging::Pageable(vec![0.0; 8]));
        assert_eq!(cache.stats().cached_bytes, 48);

        let freed = cache.release_all();
        assert_eq!(freed, 48);
        assert_eq!(cache.stats().cached_bytes, 0);
        assert!(cache.take(4, 0).is_none());
        assert!(cache.take(8, 0).is_none());
    }

    #[test]
    fn take_or_alloc_and_put_back_roundtrip_via_mutex() {
        let cache = std::sync::Mutex::new(HostStagingCache::new(HostStagingKind::Pageable));

        // `Pageable` 種は `ctx` を参照しないため、実機なしでも
        // `take_or_alloc` の miss→alloc 経路を検証できる。`CudaContext`
        // を構築せずダミーの `Arc` を渡すことはできない（型が要求する）
        // ため、本経路の GPU 非依存検証は `HostStagingCache` 単体の
        // `take`／`put`（上記テスト群）で行い、`take_or_alloc`／
        // `put_back` の poison 処理のみを別途検証する。
        {
            let mut guard = cache.lock().unwrap();
            assert!(guard.take(4, 0).is_none());
            guard.put(4, 0, HostStaging::Pageable(vec![9.0; 4]));
        }
        {
            let mut guard = cache.lock().unwrap();
            let hit = guard.take(4, 0).unwrap();
            assert_eq!(hit.as_slice().unwrap(), &[9.0f32; 4]);
        }
    }

    #[test]
    fn put_back_recovers_from_poisoned_mutex() {
        // Mutex poison 後も `into_inner` で内部データへアクセスし続け、
        // 呼び出し元（`memory.rs`）の `lock()` 直接呼び出しが poison を
        // 検出して `BackendError` を返す一方、`put_back`／`take_or_alloc`
        // 自体は panic しないことを確認する（`.claude/rules/coding-rust.md`
        // 「本番経路で unwrap/expect を使わない」の対偶: poison からの
        // 復旧経路が panic しないこと自体は許容される設計上の緩和策）。
        let cache = std::sync::Mutex::new(HostStagingCache::new(HostStagingKind::Pageable));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = cache.lock().unwrap();
            panic!("deliberately poison the mutex");
        }));
        assert!(result.is_err());
        assert!(cache.is_poisoned());

        // poison 後でも put_back は panic せずに完了する。
        put_back(&cache, 4, 0, HostStaging::Pageable(vec![1.0; 4]));

        // 直接 `lock()` を呼ぶ経路（`memory.rs` が実際に使う経路）は
        // `Err(PoisonError)` を観測できる（呼び出し元が
        // `BackendError::DeviceUnavailable` へ変換する契約）。
        assert!(cache.lock().is_err());
    }
}
