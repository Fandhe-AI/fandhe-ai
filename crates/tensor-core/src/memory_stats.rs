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

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// `allocated_bytes()` のピーク値（過去最大値。単調非減少）。
    fn peak_allocated_bytes(&self) -> u64;

    /// ピーク値を現在値へリセットする（以降のピーク計測区間を区切る用途。
    /// PyTorch の `reset_peak_memory_stats` 相当）。
    fn reset_peak(&self);
}

/// [`MemoryStats`] を実装するバックエンド入口型（`CpuMemory` 等）が内部に
/// 保持する計測本体。`current`／`peak` をそれぞれ `AtomicU64` で持ち、
/// 確保・解放のたびに [`TrackedAllocation`] 経由で更新される。
///
/// メモリオーダリングは全て `Relaxed` を用いる。理由: 本トラッカーは単調
/// カウンタの読み書きのみを行い、他の共有データへの happens-before 関係を
/// 要求しない（`current`/`peak` 自体の値の一貫性は各アトミック操作の
/// atomicity のみで足りる）。`peak` は「`current` に加算した直後の値」で
/// `fetch_max` するため、並行アクセス下でも常に「ある時点で実在した
/// `current` 値以上」を保つ（単調非減少）。
#[derive(Debug, Default)]
pub struct AllocationTracker {
    current: AtomicU64,
    peak: AtomicU64,
}

impl AllocationTracker {
    /// ゼロ初期化された新規トラッカーを構築する。
    pub fn new() -> Self {
        Self {
            current: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }
    }

    /// `bytes` 分の確保を計上する。[`TrackedAllocation::new`] からのみ
    /// 呼ばれる（減算経路〈`on_free`〉との対称性を `TrackedAllocation` の
    /// RAII に閉じ込めるため、本メソッド自体は non-pub のまま維持する）。
    fn on_alloc(&self, bytes: u64) {
        let new_current = self.current.fetch_add(bytes, Ordering::Relaxed) + bytes;
        // fetch_max は複数スレッドが同時に加算しても「実在した current 値の
        // 最大」を取り漏らさない（CAS ループ相当を標準ライブラリが内包）。
        self.peak.fetch_max(new_current, Ordering::Relaxed);
    }

    /// `bytes` 分の解放を計上する。[`TrackedAllocation::drop`] からのみ
    /// 呼ばれる（公開 API にはしない。`on_alloc` と 1:1 で対応する呼び出しを
    /// `TrackedAllocation` の構築・破棄に構造的に紐付けることで、減算過多
    /// による `fetch_sub` のラップアラウンド〈整数アンダーフロー〉を防ぐ）。
    fn on_free(&self, bytes: u64) {
        self.current.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// 現在の確保済みバイト数。[`MemoryStats::allocated_bytes`] の実体
    /// （バックエンド入口型が委譲実装するための公開メソッド）。
    pub fn allocated_bytes(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// ピーク確保済みバイト数。[`MemoryStats::peak_allocated_bytes`] の実体。
    pub fn peak_allocated_bytes(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// ピーク値を現在値へリセットする。[`MemoryStats::reset_peak`] の実体。
    pub fn reset_peak(&self) {
        // 「以降の区間のピーク」を求める `reset_peak` の意図どおり、
        // 現在値まで引き下げる（0 に戻すと生存中のアロケーションが
        // 未計上のピークとして扱われてしまい、直後に allocated_bytes()
        // > peak_allocated_bytes() という矛盾した観測が生じるため避ける）。
        let current = self.current.load(Ordering::Relaxed);
        self.peak.store(current, Ordering::Relaxed);
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
    /// （`Relaxed` オーダリングの選択がカウンタの atomicity を損なわない
    /// ことの実行時裏付け）。
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
}
