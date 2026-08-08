//! TASK-14.1c（#176・REQ-14）: 既知確保パターンに対する `AllocationTracker`
//! の解析的期待値一致テスト。
//!
//! `memory_stats.rs` の `#[cfg(test)]`（#174・TASK-14.1a 同梱分）は Mutex
//! 一本化・`u128` 内部表現・並行アクセスといった**トラッカー内部実装の
//! 退行防止**が主眼であり、「既知サイズの確保を N 個積んだ場合のピーク値」
//! を体系的なパターン列（N 個同時生存・鋸歯・階段増減・reset_peak 区間）
//! として検証する責務は本ファイル（#176）に申し送りされている
//! （`memory_stats.rs` モジュールコメント「イシュー分担」参照）。
//!
//! サイズクラス別プール経由の計測反映（保持中の計上維持・LRU 破棄時の
//! 減算・peak 保持）は `crates/backend-cpu/tests/pooled_memory_integration.rs`
//! （#201）が既に検証済みのため、本ファイルはプールを介さない素の
//! `AllocationTracker`／`TrackedAllocation` の確保パターンに限定し、
//! 重複実装しない。
//!
//! 各テストは独立した `AllocationTracker` を用いる（`memory_stats.rs`
//! モジュールコメント「トラッカーの共有範囲」の設計意図どおり、並列実行
//! される単体テスト間で計数が混線しないようにするため）。

use std::sync::Arc;

use tensor_core::memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};

/// N 個の同一サイズ確保を生存させたまま積み、各ステップで
/// `current == i * size`・全確保後 `peak == n * size` となることを検証する。
/// 全 drop 後は `current == 0` かつ `peak` は最大値を保持し続ける。
#[test]
fn n_concurrent_allocations_of_known_size_track_exact_running_total() {
    const N: usize = 16;
    const SIZE: u64 = 4096;

    let tracker = Arc::new(AllocationTracker::new());
    let mut guards = Vec::with_capacity(N);

    for i in 1..=N {
        guards.push(TrackedAllocation::new(Arc::clone(&tracker), SIZE));
        assert_eq!(
            tracker.allocated_bytes(),
            (i as u64) * SIZE,
            "{i} 本確保後の current は i * SIZE と一致するはず"
        );
    }
    assert_eq!(tracker.peak_allocated_bytes(), (N as u64) * SIZE);

    drop(guards);
    assert_eq!(
        tracker.allocated_bytes(),
        0,
        "全 drop 後は current が 0 に戻る"
    );
    assert_eq!(
        tracker.peak_allocated_bytes(),
        (N as u64) * SIZE,
        "peak は全 drop 後も過去最大値（N * SIZE）を維持する"
    );
}

/// 鋸歯パターン（確保 → 即解放を N 回繰り返す）: 同時生存が常に 1 本
/// なので `peak == SIZE`（N によらず一定）、最終 `current == 0` となる。
#[test]
fn sawtooth_alloc_then_free_cycles_keep_peak_at_single_allocation_size() {
    const N: usize = 32;
    const SIZE: u64 = 777;

    let tracker = Arc::new(AllocationTracker::new());
    for _ in 0..N {
        let guard = TrackedAllocation::new(Arc::clone(&tracker), SIZE);
        assert_eq!(tracker.allocated_bytes(), SIZE);
        drop(guard);
        assert_eq!(tracker.allocated_bytes(), 0);
    }
    assert_eq!(
        tracker.peak_allocated_bytes(),
        SIZE,
        "同時生存が常に 1 本のためピークは単一確保サイズに留まるはず"
    );
    assert_eq!(tracker.allocated_bytes(), 0);
}

/// 階段増減パターン: 異種サイズを確保・一部を段階的に drop しながら、
/// 各ステップの `current` が「その時点の同時生存合計」と一致し、
/// `peak` が全ステップを通じたプレフィックス最大と一致することを検証する。
#[test]
fn staircase_pattern_peak_matches_prefix_maximum_of_live_totals() {
    let tracker = Arc::new(AllocationTracker::new());

    // ステップ 1: 100 バイト確保 → current=100
    let a = TrackedAllocation::new(Arc::clone(&tracker), 100);
    assert_eq!(tracker.allocated_bytes(), 100);
    assert_eq!(tracker.peak_allocated_bytes(), 100);

    // ステップ 2: 200 バイト追加確保 → current=300（プレフィックス最大値）
    let b = TrackedAllocation::new(Arc::clone(&tracker), 200);
    assert_eq!(tracker.allocated_bytes(), 300);
    assert_eq!(tracker.peak_allocated_bytes(), 300);

    // ステップ 3: a（100）を解放 → current=200 に減るが peak は 300 を保持
    drop(a);
    assert_eq!(tracker.allocated_bytes(), 200);
    assert_eq!(tracker.peak_allocated_bytes(), 300);

    // ステップ 4: 400 バイト追加確保 → current=600（新しいプレフィックス最大）
    let c = TrackedAllocation::new(Arc::clone(&tracker), 400);
    assert_eq!(tracker.allocated_bytes(), 600);
    assert_eq!(
        tracker.peak_allocated_bytes(),
        600,
        "600（200+400）が新たなプレフィックス最大になるはず"
    );

    // ステップ 5: b（200）・c（400）を解放 → current=0、peak は 600 を維持
    drop(b);
    drop(c);
    assert_eq!(tracker.allocated_bytes(), 0);
    assert_eq!(
        tracker.peak_allocated_bytes(),
        600,
        "全解放後もピークはプレフィックス最大値 600 を維持するはず"
    );
}

/// `reset_peak` を挟んだ複数区間パターン: 区間 1 で大きいピークを形成した
/// あと `reset_peak()` し、区間 2 では小さい確保のみを行う。区間 2 の
/// ピークは「reset 時点の current 基点 + 区間 2 の増分」の期待値と一致し、
/// 全ステップで不変条件 `peak >= current` を保つことを確認する。
#[test]
fn reset_peak_partitions_measurement_into_independent_windows() {
    let tracker = Arc::new(AllocationTracker::new());

    // 区間 1: 500 + 500 = 1000 でピーク形成後、500 を解放して基点 500 を残す。
    let a = TrackedAllocation::new(Arc::clone(&tracker), 500);
    let b = TrackedAllocation::new(Arc::clone(&tracker), 500);
    assert!(tracker.peak_allocated_bytes() >= tracker.allocated_bytes());
    assert_eq!(tracker.peak_allocated_bytes(), 1000);
    drop(b);
    assert_eq!(
        tracker.allocated_bytes(),
        500,
        "区間 2 の基点として 500 が残る"
    );
    assert!(tracker.peak_allocated_bytes() >= tracker.allocated_bytes());

    // 区間の切り替え: reset 直後は peak == current（1000 の記憶は破棄される）。
    tracker.reset_peak();
    assert_eq!(
        tracker.peak_allocated_bytes(),
        tracker.allocated_bytes(),
        "reset_peak 直後は peak == current"
    );
    let baseline = tracker.allocated_bytes();

    // 区間 2: 基点（500）+ 増分（150）= 650 が新ピークの期待値。
    let c = TrackedAllocation::new(Arc::clone(&tracker), 150);
    assert!(tracker.peak_allocated_bytes() >= tracker.allocated_bytes());
    assert_eq!(
        tracker.peak_allocated_bytes(),
        baseline + 150,
        "区間 2 のピークは reset 基点 + 区間 2 の増分と一致するはず"
    );

    drop(a);
    drop(c);
    assert!(tracker.peak_allocated_bytes() >= tracker.allocated_bytes());
}

/// `AllocationTracker` 自体は `MemoryStats` を実装しない（実装するのは
/// `backend-cpu::CpuMemory` 等のバックエンド入口型。`memory_stats.rs`
/// モジュールコメント「トラッカーの共有範囲」参照）。object-safe 契約
/// （`MemoryStats` トレイト doc コメント参照）を tensor-core 側単独でも
/// 検証できるよう、トラッカーへ委譲するだけの最小ラッパーをテスト内に
/// 用意する。
struct TrackerAsMemoryStats(Arc<AllocationTracker>);

impl MemoryStats for TrackerAsMemoryStats {
    fn allocated_bytes(&self) -> u64 {
        self.0.allocated_bytes()
    }

    fn peak_allocated_bytes(&self) -> u64 {
        self.0.peak_allocated_bytes()
    }

    fn reset_peak(&self) {
        self.0.reset_peak();
    }
}

/// object-safe 契約（`memory_stats.rs` の `MemoryStats` トレイト doc
/// コメント参照）の検証: `&dyn MemoryStats` として trait オブジェクト
/// 越しに読み取っても、具象型経由と同じ期待値が観測できることを確認する。
#[test]
fn dyn_memory_stats_trait_object_observes_the_same_expected_values() {
    let tracker = Arc::new(AllocationTracker::new());
    let guard = TrackedAllocation::new(Arc::clone(&tracker), 2048);

    let wrapper = TrackerAsMemoryStats(Arc::clone(&tracker));
    let as_dyn: &dyn MemoryStats = &wrapper;
    assert_eq!(as_dyn.allocated_bytes(), 2048);
    assert_eq!(as_dyn.peak_allocated_bytes(), 2048);

    drop(guard);
    assert_eq!(as_dyn.allocated_bytes(), 0);
    assert_eq!(as_dyn.peak_allocated_bytes(), 2048);

    as_dyn.reset_peak();
    assert_eq!(as_dyn.peak_allocated_bytes(), as_dyn.allocated_bytes());
}
