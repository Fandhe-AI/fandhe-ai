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
//! `PEAK_BYTES`／`BASELINE_BYTES` 自体は個々の読み書きこそアトミックだが、
//! 「区間の起点を記録する（`reset_peak`）→ 計測対象区間を実行する →
//! 純増分ピークを読み出す（`peak_since_reset_bytes`）」という一連の
//! 手順全体は 1 つのアトミック操作ではない。同一プロセス内で計測区間が
//! 並行・重複すると、一方の `reset_peak` が他方の計測途中の
//! `BASELINE_BYTES`／`PEAK_BYTES` を上書きし、`gemm_alloc_peak_bytes` が
//! 過小・過大な値でも「正常」として返ってしまう（PR #370 codex-review
//! 指摘 P1・alloc_tracker.rs:173 付近）。これを防ぐため、計測区間全体を
//! `MEASUREMENT_LOCK`（`Mutex<()>`）で直列化する [`measure`] を計測の
//! 唯一の入口とし、`reset_peak`／`peak_since_reset_bytes` は本モジュール
//! 内部専用（非 `pub`）に格下げする。呼び出し側（`peak_memory::run_cpu_trial`
//! 等）は `measure` の返す `(戻り値, Option<u64>)` のみを扱う。
//!
//! ## テストでの計測検証（プロセス分離。イシュー #161 PR #357 codex-review
//! 再指摘 P1 対応）
//!
//! [`measure`] 自身の直列化（`MEASUREMENT_LOCK`）は「他の `measure`
//! 呼び出し」同士の重なりだけを防ぐものであり、`TrackingAllocator` を
//! `#[global_allocator]` として共有する**同一テストバイナリ内の他
//! `#[test]` 関数**からの干渉は防げない。過去（イシュー #161 PR #357
//! 当初）は各 `#[test]` 関数本体の先頭でクレート全体共有の
//! `Mutex<()>`（`TEST_SERIAL_LOCK`）を取得する規約で対処しようとしたが、
//! libtest はロックの外側（各テストスレッドの起動・終了・結果処理）を
//! 並行して行うため、あるテストが `measure` 中でも、別テストの本体開始前
//! または終了後に生じる確保・解放が同じ `CURRENT_BYTES`／`PEAK_BYTES` へ
//! 干渉できてしまい（codex-review 再指摘 P1）、根本対処にならなかった。
//! rayon ワーカースレッド経由の確保も計上する必要がある（上記
//! 「スレッド安全性」）ため、確保元スレッドで計測対象を判別するような
//! フィルタリング（例: 呼び出しスレッド ID 限定）も採用できない。
//!
//! そのため、`TrackingAllocator` の実測値（`gemm_alloc_peak_bytes` の非
//! `None` 値）に依存するテストは、`cargo test --lib`（本クレートの通常の
//! 単体テストバイナリ）からは完全に排除し、`tests/alloc_tracker_serial.rs`
//! （`Cargo.toml` の `[[test]] harness = false` 指定）という**専用の
//! 単一スレッド・単一プロセス**へ集約する。`harness = false` は libtest の
//! スレッドプールそのものを使わず、`fn main()` が各検査関数を直接・順番に
//! 呼ぶ構成のため、他のテストコード・libtest 自身のスレッド起動処理が
//! `TrackingAllocator` のカウンタへ混入する余地が構造的に存在しない
//! （プロセス分離こそが唯一の根本対処であり、ロック方式では原理的に
//! 塞げない。詳細は当該ファイル冒頭コメント参照）。この結果、本クレートの
//! `#[cfg(test)] #[global_allocator]` 宣言（`TEST_GLOBAL_ALLOCATOR`）は
//! 不要になったため削除した: `cargo test --lib` は `System`（既定
//! アロケータ）のまま実行され、`gemm_alloc_peak_bytes` を実測しない
//! テスト（手動で値を上書きする回帰テスト等）には影響しない。
//!
//! ## 新規依存の追加なし
//!
//! `std::alloc::{GlobalAlloc, System}` と `std::sync::Mutex`（標準ライブラリ）
//! のみを用いる自作ラッパーであり、許容依存 8 区分外の新規クレート追加
//! （`.claude/rules/deps-policy.md`）には該当しない。

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
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
/// `CURRENT_BYTES`／`PEAK_BYTES` へ記録する `GlobalAlloc` 実装。
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
///
/// プロセス全体で共有される [`PEAK_BYTES`]／[`BASELINE_BYTES`] への
/// 書き込みであり、計測区間が並行・重複すると他区間の値を破壊しうる
/// （PR #370 codex-review 指摘 P1）。そのため本関数は非公開とし、
/// [`MEASUREMENT_LOCK`] で直列化された [`measure`] の内部でのみ呼ぶ。
fn reset_peak() {
    let current = CURRENT_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(current, Ordering::Relaxed);
    BASELINE_BYTES.store(current, Ordering::Relaxed);
}

/// 直近の [`reset_peak`] 以降に観測された、生存バイト数の純増分ピークを
/// 返す。`TrackingAllocator` が実際にプロセスの `#[global_allocator]`
/// として有効化されていない文脈（`HOOK_ACTIVE` が一度も立っていない）
/// では `None` を返し、「確保 0 バイトだった」との誤った断定を避ける。
///
/// [`reset_peak`] と同様、非公開かつ [`measure`] の内部専用（理由は
/// [`reset_peak`] のドキュメント参照）。
fn peak_since_reset_bytes() -> Option<u64> {
    if !HOOK_ACTIVE.load(Ordering::Relaxed) {
        return None;
    }
    let baseline = BASELINE_BYTES.load(Ordering::Relaxed);
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    Some(peak.saturating_sub(baseline) as u64)
}

/// `PEAK_BYTES`／`BASELINE_BYTES`（プロセス全体で共有）への計測区間
/// アクセスを直列化する排他ロック。[`measure`] だけがこれを獲得し、
/// 「起点記録 → 計測対象実行 → ピーク読み出し」を単一の臨界区間として
/// 保護する（PR #370 codex-review 指摘 P1・並行するピーク計測がグローバル
/// 状態を上書きする問題への対応。alloc_tracker.rs:173 付近）。
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

/// `f` の実行区間中に生じたプロセス全体のヒープ確保ピーク（純増分。
/// バイト単位）を計測し、`f` の戻り値と併せて返す唯一の公開計測 API。
///
/// `MEASUREMENT_LOCK` を `f` の実行完了までホールドし続けることで、
/// 複数スレッド・複数テストから本関数が並行呼び出しされても
/// `reset_peak`（起点記録）・`peak_since_reset_bytes`（読み出し）の
/// ペアが他の計測区間と重ならないことを保証する（PR #370 codex-review
/// 指摘 P1）。`TrackingAllocator` が有効化されていない文脈では
/// ピーク値は `None` になる（`peak_since_reset_bytes` 参照）。
///
/// ロック汚染（`f` 内での panic）時は `Mutex::lock` の `Err` から
/// 内部ガードを取り出して継続する: 本ロックが保護するのは単純な
/// アトミックカウンタの読み書き手順のみであり、`f` の panic によって
/// カウンタ自体が不変条件を破ることはないため（`into_inner()` で
/// poison を解除しても安全）。
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, Option<u64>) {
    let _guard = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_peak();
    let result = f();
    let peak = peak_since_reset_bytes();
    (result, peak)
}
