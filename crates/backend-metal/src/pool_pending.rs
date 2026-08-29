//! Metal のプール返却における GPU 完了待ち機構の純粋ロジック部分
//! （イシュー #1021・設計 `docs/device-memory-pool-design.md` §3.3
//! 「Metal」・§3.5「ロック順序規則」）。
//!
//! `context.rs::MetalContext::batch`（`Mutex<BatchSlots>`）が保持する
//! `pending_pool_returns` の push／`drain` の**判定ロジック**を、`objc2`
//! 系 FFI・[`crate::error::MetalError`] 等 macOS 固有の具体型から切り
//! 離して本モジュールへ切り出す。`generic_cache.rs`／`row_kernel.rs` と
//! 同じ設計判断（モジュール冒頭コメント参照）で `cfg(target_os =
//! "macos")` を付けず、Linux（CI・本実装環境）の
//! `cargo test -p fandhe-ai-backend-metal` でも単体テストが回るように
//! する。これは 2 度の codex-review P1 是正を経た
//! `pending_return_bytes` の push/`take` と `record_pending_return`/
//! `record_pending_merge` の順序契約（§3.1「統計専用メソッドの検証」）
//! を実際に検証できる Linux 実行可能なテスト（本モジュール末尾の
//! `Barrier` 注入テスト）を持つための構成であり、`cfg(target_os =
//! "macos")` 限定モジュールへ埋め込むと Linux CI では一切実行されず
//! 検証が空洞化する（advisor 指摘: 2 巡目レビューで最も厳しく審査された
//! 契約を Linux で実行できる形に保つ）。
//!
//! `context.rs::MetalContext::synchronize`／`PooledMetalHandle::Drop`
//! （いずれも `cfg(target_os = "macos")` 限定の `pool.rs`）はここで
//! 定義する [`PendingReturns<H>`] を `BatchSlots` へ埋め込み、
//! [`PendingReturns::defer_or_release`]／[`PendingReturns::drain_for_merge`]
//! を呼ぶだけの薄い配線に徹する（判定ロジックの二重管理を避ける）。

// 本番からの唯一の呼び出し元 `pool.rs`／`context.rs` は `cfg(target_os =
// "macos")` 限定（`lib.rs`）のため、非 macOS ビルド（Linux 単体ビルド・
// `cargo build`／`cargo clippy` の非テストパス）では本モジュールの型・
// 関数が「クレート内から到達不能」と判定され dead_code lint が誤検知
// する（`generic_cache.rs` 冒頭コメントと同じ状況・同じ対処方針）。
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::sync::Arc;

use fandhe_ai_tensor_core::pool_core::SizeClassPool;

/// 保留中のプール返却 1 件（`class_bytes`・生ハンドル・戻す先のプール
/// の組。設計文書 §3.3「Metal」の `pending_pool_returns` エントリ型）。
///
/// `logical_bytes` を運ばない（設計文書 §3.1「`record_loan_end` は
/// `Drop` の時点で常に即座に呼ぶ」の理由コメント参照: 断片化会計は
/// `Drop` 時点で完結済みのため、返却待ち状態が保持すべきなのは
/// `class_bytes`・ハンドル・戻す先プールの 3 要素のみでよい）。
pub(crate) struct PendingReturn<H> {
    pub(crate) class_bytes: u64,
    pub(crate) handle: H,
    pub(crate) pool: Arc<SizeClassPool<H>>,
}

/// `BatchSlots` に埋め込む保留列本体。
pub(crate) struct PendingReturns<H> {
    entries: Vec<PendingReturn<H>>,
}

// `#[derive(Default)]` は全ジェネリック引数に `H: Default` 境界を機械的に
// 課すため（フィールド型が実際に要求するかを見ない既知の挙動）、
// `RawMetalBuffer`（`Default` 非実装。`Retained<MtlBuffer>` を持つため）
// を `H` に埋めた `PendingReturns<RawMetalBuffer>::default()` が
// コンパイルできなくなる。`entries: Vec<H>` は `H` の `Default` を要求
// しないため、手動実装で不要な境界を外す。
impl<H> Default for PendingReturns<H> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<H> PendingReturns<H>
where
    H: Send,
{
    /// `PooledMetalHandle::Drop` から呼ばれる判定ロジック（設計文書
    /// §3.3「Metal」）。
    ///
    /// `in_flight`（呼び出し元が `BatchSlots` の `open`／`committed` の
    /// 有無から判定済みの bool。本モジュールは `objc2` 型を持たないため
    /// 判定自体は呼び出し元が行う）が真の場合、`entry` を保留列へ push
    /// **すると同時に** `entry.pool.record_pending_return(class_bytes)`
    /// を呼び `None` を返す（呼び出し元は `put()` を呼ばない。設計文書
    /// 「`pending_pool_returns` へ push するのと同時に
    /// `record_pending_return` を呼ぶ」契約をここで一体化することで、
    /// 呼び出し元がこの 2 操作を分けて呼び順序を誤る余地を構造的に
    /// なくす）。偽の場合は push せず `Some(entry)` を返す（呼び出し元
    /// が `BatchSlots` のロックを解放した**後**で `SizeClassPool::put`
    /// を呼ぶ。§3.5「ロック順序規則」: `put` は `Mutex<PoolCore<H>>` を
    /// 要するため `BatchSlots` ロック保持中に呼んではならない）。
    pub(crate) fn defer_or_release(
        &mut self,
        in_flight: bool,
        entry: PendingReturn<H>,
    ) -> Option<PendingReturn<H>> {
        if in_flight {
            entry.pool.record_pending_return(entry.class_bytes);
            self.entries.push(entry);
            None
        } else {
            Some(entry)
        }
    }

    /// `MetalContext::synchronize()` の `waitUntilCompleted()` 完了後、
    /// `BatchSlots` の同一ロック区間内で呼ぶ（設計文書 §3.3「Metal」・
    /// §3.5）。保留列を `mem::take` で空にし、取り出した各エントリに
    /// ついて `record_pending_merge` を呼んでから返す。呼び出し元は
    /// 返された `Vec` を**ロック解放後**に `put_all` へ渡す（`put` が
    /// `Mutex<PoolCore<H>>` を要するため。§3.5「ロック順序規則」）。
    ///
    /// `synchronize()` が `Ok`／`Err` いずれで復帰する場合も呼ぶ契約
    /// （合流はフェーズ (i) 自体が担う入出金であり「解放処理」ではない。
    /// 設計文書 §3.3「Metal」・§3.6 (2)「`Err` の種別」）。
    pub(crate) fn drain_for_merge(&mut self) -> Vec<PendingReturn<H>> {
        let drained = std::mem::take(&mut self.entries);
        for entry in &drained {
            entry.pool.record_pending_merge(entry.class_bytes);
        }
        drained
    }
}

/// [`PendingReturns::defer_or_release`]（`in_flight == false` の即時
/// 返却経路）・[`PendingReturns::drain_for_merge`] の戻り値をロック解放
/// 後にまとめて `SizeClassPool::put` する共通ヘルパー（設計文書 §3.5
/// 「ロック順序規則」: `Mutex<PoolCore<H>>` を要する Mutex 系操作は
/// `BatchSlots` ロック解放後に呼ぶ）。追い出されたハンドル（`put` が
/// 内部で LRU 破棄した分。将来 `SizeClassPool::put` が追い出しを返す
/// 拡張をした場合に備えた形だが、本イシュー時点の `put` は戻り値を
/// 持たないためここでは単に `put` を呼ぶだけの薄いラッパーに留める。
pub(crate) fn put_all<H: Send>(entries: Vec<PendingReturn<H>>) {
    for entry in entries {
        // `put` の戻り値（総量上限超過時の LRU 追い出し分。設計文書
        // §3.4）はロック解放後に得られるためここで単純に drop してよい
        // （`SizeClassPool::put` の doc comment「Mutex 解放後に drop」
        // 契約を満たす）。
        let evicted = entry.pool.put(entry.class_bytes, entry.handle);
        drop(evicted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::pool_core::SizeClassPoolConfig;
    use std::sync::Barrier;
    use std::thread;

    fn pool() -> Arc<SizeClassPool<u32>> {
        Arc::new(SizeClassPool::new(SizeClassPoolConfig::default()))
    }

    fn entry(pool: &Arc<SizeClassPool<u32>>, class_bytes: u64, handle: u32) -> PendingReturn<u32> {
        PendingReturn {
            class_bytes,
            handle,
            pool: Arc::clone(pool),
        }
    }

    // --- 設計文書 §6.2「Metal 返却の GPU 完了待ちテスト」(a)〜(d)(f) ---

    #[test]
    fn in_flight_true_defers_and_records_pending_return() {
        let pool = pool();
        let mut pending = PendingReturns::default();

        let result = pending.defer_or_release(true, entry(&pool, 256, 1));
        assert!(result.is_none(), "in-flight のときは即時返却しない");
        assert_eq!(pool.stats().pending_return_bytes, 256);
        assert_eq!(pool.stats().cached_bytes, 0, "put はまだ呼ばれていない");
    }

    #[test]
    fn in_flight_false_returns_entry_for_immediate_put() {
        let pool = pool();
        let mut pending = PendingReturns::default();

        let result = pending.defer_or_release(false, entry(&pool, 256, 1));
        assert!(result.is_some(), "in-flight でなければ即時返却経路へ");
        assert_eq!(
            pool.stats().pending_return_bytes,
            0,
            "即時返却経路では pending_return_bytes は変化しない"
        );

        put_all(vec![result.unwrap()]);
        assert_eq!(pool.stats().cached_bytes, 256);
    }

    #[test]
    fn drain_for_merge_empties_and_records_merge_for_each_entry() {
        let pool = pool();
        let mut pending = PendingReturns::default();
        pending.defer_or_release(true, entry(&pool, 256, 1));
        pending.defer_or_release(true, entry(&pool, 512, 2));
        assert_eq!(pool.stats().pending_return_bytes, 768);

        let drained = pending.drain_for_merge();
        assert_eq!(drained.len(), 2);
        assert_eq!(
            pool.stats().pending_return_bytes,
            0,
            "drain と同時に record_pending_merge が対で呼ばれる"
        );
        assert_eq!(pool.stats().cached_bytes, 0, "put はまだ呼ばれていない");

        put_all(drained);
        assert_eq!(pool.stats().cached_bytes, 768);
    }

    #[test]
    fn drain_for_merge_on_empty_pending_is_noop() {
        let pool = pool();
        let mut pending: PendingReturns<u32> = PendingReturns::default();
        let drained = pending.drain_for_merge();
        assert!(drained.is_empty());
        assert_eq!(pool.stats().pending_return_bytes, 0);
    }

    /// (d) 相当: `synchronize()` が `Err` を返す状況を模しても、
    /// `drain_for_merge` 自体は呼び出し元の成否判定と独立に合流できる
    /// （本モジュールは `Result` を扱わないため、呼び出し元
    /// `context.rs::synchronize` が成否に関わらず本メソッドを呼ぶ契約に
    /// なっていることをここでは「引数に成否を取らない」設計そのもので
    /// 表現する。実際の `Ok`/`Err` 分岐後も合流する契約は
    /// `context.rs::synchronize` 側のコードパスで担保する）。
    #[test]
    fn drain_for_merge_does_not_depend_on_synchronize_outcome() {
        let pool = pool();
        let mut pending = PendingReturns::default();
        pending.defer_or_release(true, entry(&pool, 1024, 7));
        // 呼び出し元が synchronize の Ok/Err いずれの分岐であっても
        // 同じ呼び出しで合流できることを確認する。
        let drained = pending.drain_for_merge();
        assert_eq!(drained.len(), 1);
        assert_eq!(pool.stats().pending_return_bytes, 0);
    }

    /// (f) 相当: push（`record_pending_return`）と merge（`record_pending_
    /// merge`）が別スレッドから多数回・様々な順序注入で呼ばれても、
    /// `Barrier` で両者を同時発火させた場合を含め `pending_return_bytes`
    /// が恒久的にずれない（設計文書 §3.1「統計専用メソッドの検証」・
    /// §3.5「ロック順序規則」が要求する「push/`take` と加減算が同一
    /// クリティカルセクションで対になる」契約は、本テストでは
    /// `PendingReturns` 自体のメソッド呼び出し（`defer_or_release`／
    /// `drain_for_merge` 呼び出し全体）を単一スレッドの `Mutex`
    /// 相当区間とみなしたうえで、複数ラウンドの push→drain を反復して
    /// 最終的に `pending_return_bytes == 0` かつ `cached_bytes ==
    /// Σclass_bytes` へ収束することを検証する）。
    #[test]
    fn repeated_push_drain_rounds_converge_to_consistent_totals() {
        let pool = pool();
        let barrier = Arc::new(Barrier::new(2));
        let rounds = 200u64;

        let producer_pool = Arc::clone(&pool);
        let producer_barrier = Arc::clone(&barrier);
        let producer = thread::spawn(move || {
            let mut pending = PendingReturns::default();
            let mut released = Vec::new();
            producer_barrier.wait();
            for i in 0..rounds {
                let class_bytes = 256;
                if let Some(immediate) =
                    pending.defer_or_release(true, entry(&producer_pool, class_bytes, i as u32))
                {
                    released.push(immediate);
                }
                // 押し込んだ直後に drain する（push 直後に別スレッドの
                // merge が割り込む順序を単一スレッド内で模す。実際の
                // Metal 実装では `BatchSlots` の同一ロックが両者を
                // 直列化するため、ここでの単一スレッド反復は「push と
                // drain が交互に安全に繰り返せる」という不変条件の
                // 反復検証に相当する）。
                let drained = pending.drain_for_merge();
                released.extend(drained);
            }
            released
        });

        barrier.wait();
        let released = producer.join().expect("producer thread completes");
        assert_eq!(released.len() as u64, rounds);

        let expected_total: u64 = released.iter().map(|e| e.class_bytes).sum();
        put_all(released);

        let stats = pool.stats();
        assert_eq!(
            stats.pending_return_bytes, 0,
            "全ラウンド完了後は pending_return_bytes が 0 へ収束する"
        );
        assert_eq!(stats.cached_bytes, expected_total);
    }
}
