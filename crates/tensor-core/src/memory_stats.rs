//! アロケータ計測フック（TASK-14.1a・#174）。
//!
//! REQ-14（`docs/spec/04-requirements.md`）は、v1 で Rust 側の内部計測手段が
//! なく外部計測（`nvidia-smi`）に頼らざるを得なかった教訓から、プロセス内の
//! 確保済みバイト数のピーク値を返す内部計測 API を CPU/CUDA/Metal で**同一
//! シグネチャ**（バックエンド共通）で提供することを求める。本モジュールは
//! その共通シグネチャ（[`MemoryStats`]）と、3 バックエンド共通で使う計測実装
//! （[`AllocationTracker`]・[`TrackedAllocation`]）を提供する。
//!
//! # イシュー分担
//!
//! - 本イシュー（#174・TASK-14.1a）: 本モジュールの新設 ＋ `backend-cpu`
//!   （`CpuMemory`）への組み込み。受け入れ条件は「CPU バックエンドで
//!   ピーク値が取得できる」
//! - #175（TASK-14.1b）: CUDA/Metal のメモリ確保経路（`backend-cuda::CudaMemory`／
//!   `backend-metal::MetalMemory`）への同フック組み込み（同一シグネチャ維持）
//! - #176（TASK-14.1c）: 既知確保パターンの期待値一致テストの本格整備
//!
//! `buffer::MemoryOps` に必須メソッドとして追加すると既存の `CudaMemory`／
//! `MetalMemory` 実装（未実装のため）を壊し #175 のスコープを先食いするため、
//! `MemoryOps` とは独立したトレイトとして新設する（`device.rs`・`buffer.rs`
//! と同じ依存逆転構成: trait 定義は `tensor-core`、実装は各バックエンド）。
//!
//! # 計測対象の粒度（スコープ外の申し送り）
//!
//! 本フックが計測するのは `MemoryOps`（`alloc_zeroed`／`upload`）経由の
//! デバイスバッファ確保のみである。`BackendOps` 演算内部が一時的に確保する
//! `Vec<f32>`（例: `backend-cpu::ops::CpuBackendOps::gemm` の出力バッファ）は
//! 対象外とする。計測要否は TASK-14.2（GEMM 4096³ 係数上限の実測）で判断し、
//! 必要であれば別イシューで追跡する（`.claude/rules/out-of-scope-tracking.md`）。
//!
//! # トラッカーの共有範囲（プロセスグローバルにしない理由）
//!
//! [`AllocationTracker`] は `Arc` で `CpuMemory`／将来の `CudaMemory`／
//! `MetalMemory` インスタンス間に共有させる設計とし、`static` グローバルには
//! しない。理由は 2 点:
//! (a) 並列実行される単体テスト間で計数が混線しフレーキーテストの原因になる
//!     （REQ-4 の偽陽性防止方針と整合）
//! (b) グローバル可変状態を避ける安全側判断
//!
//! spec が言う「プロセス内のピーク値」は、計測対象プロセスがバックエンド
//! 入口（`CpuMemory` 等）を単一インスタンスで共有する運用（TASK-14.2 の
//! ベンチハーネスがこの形を取る想定）で満たせる。

use std::sync::{Arc, Mutex};

/// バックエンド共通のアロケータ計測 API（PyTorch の `memory_allocated`／
/// `max_memory_allocated`／`reset_peak_memory_stats` 相当）。
///
/// CPU/CUDA/Metal で同一シグネチャを実装させることが TASK-14.1 の受け入れ
/// 基準そのものである。object-safe に設計している（`buffer::MemoryOps`・
/// `device::DeviceProvider` と同じく `&dyn MemoryStats` として扱える）。
pub trait MemoryStats {
    /// 現在の確保済みバイト数（この瞬間に生存しているアロケーションの合計）。
    fn allocated_bytes(&self) -> u64;

    /// このトラッカーを共有しているインスタンス群が観測してきた
    /// `allocated_bytes()` のピーク値（直近の `reset_peak` 以降の区間での
    /// 最大値）。`reset_peak` を挟まない限り単調非減少。`peak >=
    /// allocated_bytes()` は `reset_peak` 呼び出しの有無によらず常時成立する
    /// 強不変条件である（[`AllocationTracker`] の同期方式を参照）。
    fn peak_allocated_bytes(&self) -> u64;

    /// ピーク値を現在値へリセットする（以降のピーク計測区間を区切る用途。
    /// PyTorch の `reset_peak_memory_stats` 相当）。
    fn reset_peak(&self);
}

/// [`MemoryStats`] を実装するバックエンド入口型（`CpuMemory` 等）が内部に
/// 保持する計測本体。`current`／`peak` を単一の [`Mutex`] の下に同居させ、
/// 確保・解放・リセットのすべての操作をそのロックで直列化する。
/// [`TrackedAllocation`] 経由で `current`・`peak` が更新される。
///
/// # 同期方式（Mutex 一本化。PR #359 codex-review 指摘 P1 の修正）
///
/// 旧実装は `current`／`peak` をそれぞれ独立の `AtomicU64` として持ち、
/// `reset_peak()` は `peak.fetch_update` の CAS ループで `current` を
/// 都度再読込する方式だった。しかしこれは TOCTOU を解消できていなかった:
/// CAS は「`peak` の**値**が読み取り時から変化していないか」しか検査
/// できないため、並行する `on_alloc()`（`fetch_max`）が `peak` を
/// **同じ値へ**更新した場合（新たな確保後の `current` がちょうど既存の
/// `peak` と同値になるケース。例: `peak == 200` の状態で `current` が
/// 100 → 200 に増えて `fetch_max(200)` が実行される場合）、`peak` 自体の
/// ビット列は変化しないため reset 側の CAS は「変化なし」と誤認して成功
/// し、その確保より前に読んだ古い `current`（例: 100）で `peak` を
/// 100 に引き下げてしまう。結果、確保が生存中にもかかわらず
/// `peak_allocated_bytes() < allocated_bytes()` という契約違反が生じうる
/// （並行確保の値がたまたま既存ピークと一致する場合に限り顕在化するため
/// 単体テストでの再現率が低く、コードレビューで検出された）。
///
/// `current`・`peak` を 1 つの `Mutex` の背後に置き、確保
/// （`current` 加算 → `peak` 更新の比較）・解放（`current` 減算）・
/// リセット（`peak = current`）のいずれも「ロック取得 → 両フィールドを
/// 一括更新 → 解放」という単一の原子操作として行うことで、値の一致に
/// 依存する検出漏れを構造的に排除する。本トラッカーはホットパス
/// （GEMM 等の演算カーネル内部）ではなく統計フックの用途に限られるため
/// （モジュールコメント「計測対象の粒度」参照）、`Mutex` のオーバーヘッドは
/// 許容する。
#[derive(Debug, Default, Clone, Copy)]
struct MemoryCounters {
    current: u64,
    peak: u64,
}

#[derive(Debug, Default)]
pub struct AllocationTracker {
    counters: Mutex<MemoryCounters>,
}

impl AllocationTracker {
    /// ゼロ初期化された新規トラッカーを構築する。
    pub fn new() -> Self {
        Self::default()
    }

    /// `counters` のロックを取得する。他スレッドがロック保持中に panic
    /// した場合（poisoned）でも、本トラッカーが保持するのは単調カウンタの
    /// みで不変条件の破壊が起きないため、`unwrap()` で連鎖 panic させず
    /// `into_inner()` で中身をそのまま引き継ぐ（本番経路で `unwrap()` を
    /// 使わない方針。`coding-rust.md`）。
    fn lock(&self) -> std::sync::MutexGuard<'_, MemoryCounters> {
        self.counters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `bytes` 分の確保を計上する。[`TrackedAllocation::new`] からのみ
    /// 呼ばれる（減算経路〈`on_free`〉との対称性を `TrackedAllocation` の
    /// RAII に閉じ込めるため、本メソッド自体は non-pub のまま維持する）。
    fn on_alloc(&self, bytes: u64) {
        let mut counters = self.lock();
        // saturating を使う理由: 旧 `AtomicU64::fetch_add` はオーバーフロー
        // 時に暗黙のラップアラウンドをしていたが、`u64` 同士の `+=` は
        // debug ビルド（`cargo test` 既定）でオーバーフロー panic する。
        // 統計フックが確保経路を panic させることは避けたいため
        // `saturating_add` で上限に飽和させる（本番経路で panic しない方針。
        // `coding-rust.md`）。
        counters.current = counters.current.saturating_add(bytes);
        if counters.current > counters.peak {
            counters.peak = counters.current;
        }
    }

    /// `bytes` 分の解放を計上する。[`TrackedAllocation::drop`] からのみ
    /// 呼ばれる（公開 API にはしない。`on_alloc` と 1:1 で対応する呼び出しを
    /// `TrackedAllocation` の構築・破棄に構造的に紐付けることで、減算過多
    /// による整数アンダーフローを防ぐ）。`saturating_sub` は上記 `on_alloc`
    /// と同じ理由（panic 回避）で用いる保険であり、正常経路では
    /// `on_alloc`/`on_free` の 1:1 対応により減算過多は起きない想定。
    fn on_free(&self, bytes: u64) {
        let mut counters = self.lock();
        counters.current = counters.current.saturating_sub(bytes);
    }

    /// 現在の確保済みバイト数。[`MemoryStats::allocated_bytes`] の実体
    /// （バックエンド入口型が委譲実装するための公開メソッド）。
    pub fn allocated_bytes(&self) -> u64 {
        self.lock().current
    }

    /// ピーク確保済みバイト数。[`MemoryStats::peak_allocated_bytes`] の実体。
    pub fn peak_allocated_bytes(&self) -> u64 {
        self.lock().peak
    }

    /// ピーク値を現在値へリセットする。[`MemoryStats::reset_peak`] の実体。
    pub fn reset_peak(&self) {
        // 「以降の区間のピーク」を求める `reset_peak` の意図どおり、
        // 現在値まで引き下げる（0 に戻すと生存中のアロケーションが
        // 未計上のピークとして扱われてしまい、直後に allocated_bytes()
        // > peak_allocated_bytes() という矛盾した観測が生じるため避ける）。
        //
        // `current` の読み取りと `peak` への書き込みを同一ロック区間で
        // 行うため、他スレッドの `on_alloc()` はこの区間の前後どちらかで
        // 完全に直列化される（構造体コメント「同期方式」参照）。CAS の
        // 値一致による検出漏れが原理的に発生しない。
        let mut counters = self.lock();
        counters.peak = counters.current;
    }
}

/// 1 回の確保に対応する RAII ガード。構築時に `tracker.on_alloc(bytes)`、
/// `Drop` 時に `tracker.on_free(bytes)` を呼ぶことで、確保・解放の対応漏れ
/// （計上漏れ・二重減算）を構造的に排除する。
///
/// バックエンドの具体ハンドル型（`backend-cpu::memory::CpuBufferHandle` 等）
/// にフィールドとして埋め込み、ハンドル本体（`Vec<f32>` 等）の `Drop` と
/// 同時に解放計上されるようにする想定（`buffer.rs` モジュールコメント
/// 「解放方針（RAII 一本化）」と同じ設計判断）。
#[derive(Debug)]
pub struct TrackedAllocation {
    tracker: Arc<AllocationTracker>,
    bytes: u64,
}

impl TrackedAllocation {
    /// `bytes` バイトの確保を `tracker` に計上し、対応する RAII ガードを返す。
    /// `bytes == 0`（空ハンドル契約。`buffer.rs` モジュールコメント参照）でも
    /// 呼び出し自体は許容する（0 バイト加算は現在値・ピークいずれも変化
    /// させない no-op として自然に振る舞う）。
    pub fn new(tracker: Arc<AllocationTracker>, bytes: u64) -> Self {
        tracker.on_alloc(bytes);
        Self { tracker, bytes }
    }
}

impl Drop for TrackedAllocation {
    fn drop(&mut self) {
        self.tracker.on_free(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn alloc_increases_current_and_peak() {
        let tracker = Arc::new(AllocationTracker::new());
        let guard = TrackedAllocation::new(Arc::clone(&tracker), 1024);

        assert_eq!(tracker.allocated_bytes(), 1024);
        assert_eq!(tracker.peak_allocated_bytes(), 1024);
        drop(guard);
    }

    #[test]
    fn peak_persists_after_free_while_current_drops() {
        let tracker = Arc::new(AllocationTracker::new());
        let guard = TrackedAllocation::new(Arc::clone(&tracker), 4096);
        drop(guard);

        assert_eq!(tracker.allocated_bytes(), 0, "解放後は current が 0 に戻る");
        assert_eq!(
            tracker.peak_allocated_bytes(),
            4096,
            "peak は解放後も過去最大値を保持する"
        );
    }

    #[test]
    fn peak_tracks_sum_of_concurrently_live_allocations() {
        let tracker = Arc::new(AllocationTracker::new());
        let a = TrackedAllocation::new(Arc::clone(&tracker), 100);
        let b = TrackedAllocation::new(Arc::clone(&tracker), 200);

        assert_eq!(tracker.allocated_bytes(), 300);
        assert_eq!(tracker.peak_allocated_bytes(), 300);

        drop(a);
        assert_eq!(tracker.allocated_bytes(), 200);
        assert_eq!(
            tracker.peak_allocated_bytes(),
            300,
            "1 本解放してもピークは同時生存時の合計を保つ"
        );
        drop(b);
    }

    #[test]
    fn reset_peak_rebases_to_current_value() {
        let tracker = Arc::new(AllocationTracker::new());
        let a = TrackedAllocation::new(Arc::clone(&tracker), 500);
        let b = TrackedAllocation::new(Arc::clone(&tracker), 500);
        drop(b);
        assert_eq!(tracker.peak_allocated_bytes(), 1000);

        tracker.reset_peak();
        assert_eq!(
            tracker.peak_allocated_bytes(),
            tracker.allocated_bytes(),
            "reset_peak 直後は peak == current"
        );

        let c = TrackedAllocation::new(Arc::clone(&tracker), 100);
        assert_eq!(tracker.peak_allocated_bytes(), 600);
        drop(a);
        drop(c);
    }

    #[test]
    fn zero_byte_allocation_is_a_no_op_for_counters() {
        let tracker = Arc::new(AllocationTracker::new());
        let guard = TrackedAllocation::new(Arc::clone(&tracker), 0);
        assert_eq!(tracker.allocated_bytes(), 0);
        assert_eq!(tracker.peak_allocated_bytes(), 0);
        drop(guard);
        assert_eq!(tracker.allocated_bytes(), 0);
    }

    #[test]
    fn drop_deducts_exactly_once() {
        let tracker = Arc::new(AllocationTracker::new());
        let guard = TrackedAllocation::new(Arc::clone(&tracker), 64);
        assert_eq!(tracker.allocated_bytes(), 64);
        drop(guard);
        assert_eq!(
            tracker.allocated_bytes(),
            0,
            "TrackedAllocation は Drop 経由でのみ減算されるため二重減算は起きない"
        );
    }

    /// 複数スレッドから同時に確保・解放しても panic せず、最終的な
    /// `current` が理論値（0）に一致することを確認するスモークテスト
    /// （`Mutex` による直列化がカウンタの一貫性を損なわないことの
    /// 実行時裏付け）。
    #[test]
    fn concurrent_alloc_free_smoke_test_converges_to_zero() {
        let tracker = Arc::new(AllocationTracker::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let guard = TrackedAllocation::new(Arc::clone(&tracker), 8);
                        drop(guard);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("計測スレッドが panic せずに完了する");
        }

        assert_eq!(tracker.allocated_bytes(), 0);
        assert!(tracker.peak_allocated_bytes() >= 8);
    }

    /// PR #359 codex-review 指摘 P1 の再発防止テスト。`current`／`peak` を
    /// 独立の `AtomicU64` として持ち `reset_peak()` を `peak.fetch_update`
    /// の CAS ループで実装していた旧実装では、並行する `on_alloc()`
    /// （`fetch_max`）が `peak` を**既存の値と同じ値へ**更新するケース
    /// （新たな確保後の `current` がちょうど既存の `peak` と一致する場合）
    /// で CAS が「変化なし」と誤認して成功し、reset 側が読んだ古い
    /// `current` で `peak` を引き下げてしまう TOCTOU が存在した
    /// （`AllocationTracker` の doc コメント「同期方式」参照）。値の一致に
    /// 依存する再現条件のため単体テストでは検出困難だったが、Mutex 一本化
    /// により構造的に解消済みである。スレッド A が確保を繰り返し生存させ
    /// 続け、スレッド B が並行して `reset_peak()` を連打し、全スレッド
    /// join 後（並行中の確保がすべて確定した時点）に `peak >= current` を
    /// 検査することで、退行がないことを確認する。
    ///
    /// PR #359 Bugbot 指摘（Low）の修正: 旧実装は alloc スレッド側で各
    /// `TrackedAllocation` を確保直後に `drop` していたため、join 後の
    /// `current` はベースライン（1 バイト）のみに縮退し、`peak >= current`
    /// が競合の有無によらず自明に成立してしまい、並行ピークの取りこぼしを
    /// 実質検証できていなかった。ここでは各確保を `Vec` に集めて join まで
    /// 生存させ、`current` が確保スレッドの活動を反映した非自明な値になる
    /// ようにすることで、`reset_peak()` 実行中の実確保状態に対して
    /// アサーションが意味を持つようにする。
    #[test]
    fn reset_peak_does_not_lose_concurrent_allocations() {
        let tracker = Arc::new(AllocationTracker::new());
        // ベースライン確保: テスト全体を通じて生存させ、current > 0 を保つ
        // （reset_peak が 0 側に潰れていないかも合わせて検査できる）。
        let baseline = TrackedAllocation::new(Arc::clone(&tracker), 1);

        let alloc_thread = {
            let tracker = Arc::clone(&tracker);
            thread::spawn(move || {
                // join まで生存させる: 即 drop すると current が
                // ベースラインのみに縮退し reset_peak との競合を
                // 検証できなくなる（PR #359 Bugbot 指摘）。
                let mut guards = Vec::with_capacity(2000);
                for _ in 0..2000 {
                    guards.push(TrackedAllocation::new(Arc::clone(&tracker), 64));
                }
                guards
            })
        };
        let reset_thread = {
            let tracker = Arc::clone(&tracker);
            thread::spawn(move || {
                for _ in 0..2000 {
                    tracker.reset_peak();
                }
            })
        };

        let guards = alloc_thread
            .join()
            .expect("確保スレッドが panic せずに完了する");
        reset_thread
            .join()
            .expect("reset_peak スレッドが panic せずに完了する");

        assert!(
            tracker.peak_allocated_bytes() >= tracker.allocated_bytes(),
            "全スレッド join 後は peak >= current が成立するはず \
             （peak={}, current={}）",
            tracker.peak_allocated_bytes(),
            tracker.allocated_bytes()
        );
        // guards 生存中（drop 前）の current は「ベースライン + 2000 件分」
        // に達しているはずであり、alloc スレッドの確保が実際に join まで
        // 生き続けていたことを裏付ける（このアサーションがないと Vec に
        // 集めた意図がテストコード上で検証されないまま暗黙の前提になる）。
        assert_eq!(
            tracker.allocated_bytes(),
            1 + 2000 * 64,
            "join 直後・drop 前は全確保が生存しているはず"
        );
        drop(guards);
        drop(baseline);
    }
}
