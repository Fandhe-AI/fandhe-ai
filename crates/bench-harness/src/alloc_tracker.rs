//! プロセス全体のヒープ確保量を実測する `#[global_allocator]` 用トラッカー
//! （PR #370 codex-review 指摘 P1 対応・イシュー #178）。
//!
//! [`crate::peak_memory`] の `MemoryOps`（`upload`／`alloc_zeroed`）経由の
//! 計測は、`BackendOps::gemm` が内部で確保する一時バッファ（CPU:
//! `CpuBackendOps::gemm` の出力 `Vec<f32>`、`gemm_blis_parallel` の
//! パッキングバッファ）を一切計上しない（`crates/backend-cpu/src/ops.rs`
//! 参照。`gemm` は `&Tensor<f32>` を直接受け取り `DeviceBuffer` を経由
//! しないため）。本モジュールは Rust の `GlobalAlloc` フックを用いて
//! プロセス全体の生存中バイト数とその最大値を実測することで、この計測
//! できない領域（GEMM 実行が実際に確保するバイト数）を埋める。
//!
//! ## 適用範囲（CPU バックエンド限定）
//!
//! CUDA（`cudarc` 経由の driver allocation）・Metal（`objc2-metal` 経由の
//! `MTLBuffer`）は Rust の `GlobalAlloc` を経由しない別ヒープ（デバイス
//! メモリ）に確保するため、本トラッカーでは観測できない
//! （`docs/peak-memory-measurement-methods.md`「計測手段の環境差」と同型の
//! 制約）。CUDA/Metal 側の真のピーク計測は別イシューのスコープとし
//! （`.claude/rules/out-of-scope-tracking.md`）、[`crate::peak_memory`] は
//! 両バックエンドについて本トラッカー由来の値を `None` として扱う。
//!
//! ## スレッド安全性
//!
//! `CpuBackendOps::gemm`（`gemm_blis_parallel`）は `rayon` のワーカー
//! スレッドから並列にヒープ確保を行う（許容依存 8 区分「CPU 並列」
//! `.claude/rules/deps-policy.md`）。本トラッカーはスレッドローカルでは
//! なく [`AtomicUsize`] によるプロセス全体のカウンタを用い、どのスレッドの
//! 確保も取りこぼさない。
//!
//! ## 新規依存の追加なし
//!
//! `std::alloc::{GlobalAlloc, System}` のみを用いる自作ラッパーであり、
//! 許容依存 8 区分外の新規クレート追加（`.claude/rules/deps-policy.md`）
//! には該当しない。

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// プロセス全体で現在生存しているバイト数（`alloc` で加算・`dealloc` で減算）。
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
/// 直近の [`reset_peak`] 以降に観測された `CURRENT_BYTES` の最大値。
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
/// [`reset_peak`] 呼び出し時点の `CURRENT_BYTES`（計測区間の起点）。
/// [`peak_since_reset_bytes`] はこの値を差し引いて区間中の純増分ピークを返す。
static BASELINE_BYTES: AtomicUsize = AtomicUsize::new(0);
/// `alloc`/`dealloc` フックが実際に一度でも呼ばれたか（＝本トラッカーが
/// プロセスの `#[global_allocator]` として実際に有効かの実行時検出）。
/// 未インストールの文脈（本クレートを通常の依存として利用する側の
/// ビルド等）で [`peak_since_reset_bytes`] が誤って `Some(0)`（「確保 0
/// バイトだった」）を返さないよう、`None`（未計測）と区別するために使う。
static HOOK_ACTIVE: AtomicBool = AtomicBool::new(false);

/// `System` アロケータへ処理を委譲しつつ、確保・解放バイト数を
/// [`CURRENT_BYTES`]／[`PEAK_BYTES`] へ記録する `GlobalAlloc` 実装。
///
/// `peak_memory_bench` バイナリ（`src/bin/peak_memory_bench.rs`）・
/// `peak_memory` モジュールの単体テスト（`#[cfg(test)] mod tests`）でのみ
/// `#[global_allocator]` として宣言する。ライブラリクレート本体
/// （`lib.rs`）では宣言しない: `#[global_allocator]` はプロセス全体に
/// 1 つしか持てず、本クレートを `dev-dependencies` として参照する
/// `backend-cpu`／`backend-cuda`（`crates/bench-harness/Cargo.toml` 冒頭
/// コメント参照）のテストバイナリへ意図せず伝播させないため。
pub struct TrackingAllocator;

// SAFETY: `System`（標準ライブラリのシステムアロケータ）への単純な委譲に
// 加え、`Layout` から求まる確保サイズをアトミックカウンタへ加減算する
// だけであり、確保されたメモリ領域自体には一切触れない。`alloc`/`dealloc`
// は仕様どおり対になって呼ばれる前提（`GlobalAlloc` トレイト契約）に従う。
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` は呼び出し元（Rust allocator API）が構築した
        // 有効な `Layout` をそのまま `System::alloc` へ委譲するのみ。
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            HOOK_ACTIVE.store(true, Ordering::Relaxed);
            let new_current =
                CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_BYTES.fetch_max(new_current, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // `GlobalAlloc::alloc_zeroed` の既定実装（`self.alloc` を呼んで
        // 手動でゼロ埋めする）に委ねず、`System::alloc_zeroed`（`calloc`
        // 等 OS のゼロ埋め済みページ割り当てを活かせる経路）へ明示的に
        // 委譲したうえでカウンタを更新する。`vec![0u8; N]` 等はこの経路を
        // 通るため、`alloc` 側の override のみでは計上漏れが起こる
        // （PR #370 codex-review 指摘 P1 対応の実装過程で実測確認）。
        // SAFETY: `layout` は呼び出し元が構築した有効な `Layout` を
        // `System::alloc_zeroed` へそのまま委譲するのみ。
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            HOOK_ACTIVE.store(true, Ordering::Relaxed);
            let new_current =
                CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_BYTES.fetch_max(new_current, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HOOK_ACTIVE.store(true, Ordering::Relaxed);
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr`／`layout` は対応する `alloc` 呼び出しのものを
        // そのまま `System::dealloc` へ委譲するのみ（呼び出し元契約）。
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`／`layout`／`new_size` は呼び出し元契約どおり
        // `System::realloc` へそのまま委譲するのみ。
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            HOOK_ACTIVE.store(true, Ordering::Relaxed);
            CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            let new_current = CURRENT_BYTES.fetch_add(new_size, Ordering::Relaxed) + new_size;
            PEAK_BYTES.fetch_max(new_current, Ordering::Relaxed);
        }
        new_ptr
    }
}

/// 計測区間の起点を記録する。`PEAK_BYTES` を現在の生存バイト数まで
/// 引き下げ、以降の増加分のみを次回の [`peak_since_reset_bytes`] が
/// 報告するようにする（`MemoryStats::reset_peak()` と同型の意味論）。
pub fn reset_peak() {
    let current = CURRENT_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(current, Ordering::Relaxed);
    BASELINE_BYTES.store(current, Ordering::Relaxed);
}

/// 直近の [`reset_peak`] 以降に観測された、生存バイト数の純増分ピークを
/// 返す。`TrackingAllocator` が実際にプロセスの `#[global_allocator]`
/// として有効化されていない文脈（`HOOK_ACTIVE` が一度も立っていない）
/// では `None` を返し、「確保 0 バイトだった」との誤った断定を避ける。
pub fn peak_since_reset_bytes() -> Option<u64> {
    if !HOOK_ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    let baseline = BASELINE_BYTES.load(Ordering::Relaxed);
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    Some(peak.saturating_sub(baseline) as u64)
}

// `cargo test --lib`（本クレートの単体テストバイナリ）に限り
// `TrackingAllocator` をプロセスの `#[global_allocator]` として有効化する
// （`#[cfg(test)]` ゲートのため通常のライブラリビルド・本クレートを
// `dev-dependencies` として参照する `backend-cpu`／`backend-cuda` 等の
// ビルドには一切影響しない）。`peak_memory::tests` を含む本クレートの
// 全単体テストが同一バイナリで実行されるため（Rust の単体テストは
// クレート単位で 1 バイナリに集約される）、宣言はここ 1 箇所のみで足り、
// `peak_memory::run_cpu_trial` が呼ぶ `reset_peak`／`peak_since_reset_bytes`
// もこの宣言により実際に計測できる状態になる（PR #370 codex-review 指摘
// P1 対応）。
#[cfg(test)]
#[global_allocator]
static TEST_GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[cfg(test)]
mod tests {
    use super::*;

    /// 既知サイズのヒープ確保を挟むことで、`reset_peak`／
    /// `peak_since_reset_bytes` が実際のアロケータイベントを反映すること
    /// を確認する（PR #370 codex-review 指摘 P1 の回帰テスト）。
    ///
    /// 本カウンタはプロセス全体で共有されるため、`cargo test` のデフォルト
    /// 並列実行下では他の単体テストの確保・解放も同一カウンタへ計上され
    /// うる（`.claude/rules/coding-rust.md` はテストの並列実行自体を禁じて
    /// いない）。厳密な「ちょうど N バイト」判定は他テストの雑音でフレーキー
    /// 化するため避け、以下の 2 点で頑健化する: (1) 判定閾値 `LEN` を本
    /// クレートの他の単体テストが単発で確保する量（数百 KB〜数 MB 台。
    /// `run_peak_memory_cpu_matches_theoretical_minimum_for_small_size` の
    /// 256KB 規模等）を大きく上回る 64MiB とし、雑音による誤検出の確率を
    /// 実務上無視できる水準まで下げる、(2) 「少なくとも LEN バイト」という
    /// 下限のみを主張し、上限は主張しない。
    #[test]
    fn peak_since_reset_bytes_reflects_real_heap_allocation() {
        const LEN: usize = 64 * 1024 * 1024;
        reset_peak();
        let v: Vec<u8> = vec![0u8; LEN];

        let peak = peak_since_reset_bytes().expect(
            "TEST_GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず",
        );
        assert!(
            peak >= LEN as u64,
            "{LEN} バイト確保後のピークは少なくとも {LEN} バイトのはず（実測: {peak}）"
        );

        drop(v);
    }

    /// `reset_peak` が計測区間を正しく区切ることを確認する: 大きな確保
    /// （解放済み）の直後に `reset_peak` した場合、新規区間のピークは
    /// 前区間の確保量を大きく下回る（雑音耐性については上のテストの
    /// コメント参照。厳密な 0 判定は並列実行下でフレーキー化するため
    /// 「前区間の確保量の半分未満」という緩い上限のみを主張する）。
    #[test]
    fn reset_peak_clears_previous_interval() {
        const LEN: usize = 64 * 1024 * 1024;
        reset_peak();
        let v: Vec<u8> = vec![0u8; LEN];
        let first_peak = peak_since_reset_bytes().unwrap();
        assert!(first_peak >= LEN as u64);
        drop(v);

        reset_peak();
        let second_peak = peak_since_reset_bytes().unwrap();
        assert!(
            second_peak < first_peak / 2,
            "reset_peak 後は前区間の確保量を大きく下回るはず（前区間: {first_peak}・実測: {second_peak}）"
        );
    }
}
