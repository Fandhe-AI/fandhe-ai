//! `bench_harness::alloc_tracker::TrackingAllocator` の実測値
//! （`gemm_alloc_peak_bytes`／`measure()` の戻り値）に依存する検証を、
//! libtest の既定ハーネスを一切経由しない単一プロセス・単一スレッドで
//! 実行する専用テストターゲット（イシュー #161 PR #357 codex-review 再指摘
//! P1 対応）。
//!
//! ## なぜ `harness = false` か（`crates/bench-harness/Cargo.toml` の
//! `[[test]] name = "alloc_tracker_serial"` 参照）
//!
//! `TrackingAllocator` はプロセス全体で共有される `CURRENT_BYTES`／
//! `PEAK_BYTES`（`bench_harness::alloc_tracker`）を持つ。libtest の既定
//! ハーネスは各 `#[test]` をワーカースレッドへ並列ディスパッチし、
//! テスト関数の**本体を** `Mutex` 等でロック直列化しても、libtest 自身が
//! 行うスレッド起動・終了・結果処理はそのロックの外側で並行に走る
//! （イシュー #161 PR #357 codex-review 再指摘 P1 の指摘そのもの）。
//! したがって「同一プロセス内に `TrackingAllocator` を共有する他の
//! `#[test]` が存在する」こと自体が干渉源であり、ロックでは原理的に
//! 塞げない。本ファイルは `harness = false`（`fn main()` が libtest を
//! 介さず直接エントリポイントになる）とし、`main()` が各検査関数を
//! スレッドを一切生成せず順番に呼ぶ。これにより「`TrackingAllocator` を
//! 唯一の `#[global_allocator]` として持つプロセス中で、計測対象の
//! コード以外は何も並行実行されない」ことを構造的に保証する。
//!
//! integration test（`tests/` 配下）としてライブラリ本体（`cargo test
//! --lib`）とは別プロセス・別バイナリでコンパイル・実行されるため、
//! `bench_harness::peak_memory::tests`（`cargo test --lib`。`System` が
//! 既定アロケータ）とは独立して安全に共存する。

use bench_harness::alloc_tracker::TrackingAllocator;
use bench_harness::alloc_tracker::measure;
use bench_harness::peak_memory::{
    PeakMemoryBackend, PeakMemoryConfig, emit_peak_memory_report, run_peak_memory,
};

/// 本バイナリ限定で `TrackingAllocator` をプロセスの `#[global_allocator]`
/// として有効化する。本ファイルは `harness = false` かつ単一スレッドで
/// 検査関数を順番に呼ぶため、libtest の既定並列実行が引き起こす干渉
/// （モジュール冒頭参照）が構造的に発生しない。
#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// `LEN` バイト確保後のピーク下限判定に許容する雑音マージン。
///
/// 本バイナリは `harness = false` により他コードと並行実行されないため、
/// `cargo test --lib` の並列実行下で問題になっていた「他テストの確保・
/// 解放による下振れ」は原理的に発生しない。それでも OS・ランタイムの
/// 内部確保（スレッドスタック等。本バイナリは単一スレッドだが `println!`
/// のバッファリング等、細かな確保はありうる）による無視できる程度の
/// 雑音を見込み、`.claude/rules/coding-rust.md`「バックエンド間数値一致
/// テストの許容誤差を単独で緩和しない」の精神に倣い、値そのもの
/// （99.9%）は変更せず据え置く（緩和ではなく維持）。
const NOISE_TOLERANCE_RATIO: f64 = 0.999;

/// 既知サイズのヒープ確保を挟むことで、`measure` が実際のアロケータ
/// イベントを反映することを確認する（PR #370 codex-review 指摘 P1 の
/// 回帰テスト。旧 `alloc_tracker::tests::measure_reflects_real_heap_allocation`
/// を本バイナリへ移設）。
fn check_measure_reflects_real_heap_allocation() {
    const LEN: usize = 64 * 1024 * 1024;
    let (_, peak) = measure(|| {
        let v: Vec<u8> = vec![0u8; LEN];
        drop(v);
    });

    let peak =
        peak.expect("GLOBAL_ALLOCATOR がテストバイナリの #[global_allocator] のため Some のはず");
    let floor = (LEN as f64 * NOISE_TOLERANCE_RATIO) as u64;
    assert!(
        peak >= floor,
        "{LEN} バイト確保後のピークは少なくともその {NOISE_TOLERANCE_RATIO} 倍 \
         （{floor} バイト）のはず（実測: {peak}）"
    );
}

/// `measure` が計測区間を正しく区切ることを確認する（旧
/// `alloc_tracker::tests::measure_clears_previous_interval` を本バイナリへ
/// 移設）。
fn check_measure_clears_previous_interval() {
    const LEN: usize = 64 * 1024 * 1024;
    let (_, first_peak) = measure(|| {
        let v: Vec<u8> = vec![0u8; LEN];
        drop(v);
    });
    let first_peak = first_peak.unwrap();
    let floor = (LEN as f64 * NOISE_TOLERANCE_RATIO) as u64;
    assert!(
        first_peak >= floor,
        "{LEN} バイト確保後のピークは少なくともその {NOISE_TOLERANCE_RATIO} 倍 \
         （{floor} バイト）のはず（実測: {first_peak}）"
    );

    let (_, second_peak) = measure(|| {});
    let second_peak = second_peak.unwrap();
    assert!(
        second_peak < first_peak / 2,
        "measure 後は前区間の確保量を大きく下回るはず（前区間: {first_peak}・実測: {second_peak}）"
    );
}

/// `MEASUREMENT_LOCK`（`bench_harness::alloc_tracker`）が並行呼び出しを
/// 直列化することの回帰テスト（旧
/// `alloc_tracker::tests::measure_serializes_concurrent_callers` を本
/// バイナリへ移設）。本検査自体は複数スレッドを spawn するが、本バイナリ
/// 内に他の検査関数が並行実行されない（`main()` が順番に呼ぶ）ため、
/// スレッド起動・終了ノイズが他検査の計測区間へ混入する心配はない。
fn check_measure_serializes_concurrent_callers() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
    const THREADS: usize = 8;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            std::thread::spawn(|| {
                measure(|| {
                    let concurrent = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
                    // 他スレッドが割り込む猶予を与えてレースを検出しやすくする。
                    std::thread::yield_now();
                    IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
                    assert_eq!(
                        concurrent, 1,
                        "MEASUREMENT_LOCK が直列化していれば closure 実行中の \
                         同時実行数は常に 1 のはず（実測: {concurrent}）"
                    );
                });
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("直列化が破れていれば closure 内の assert_eq! で panic し検出される");
    }
}

/// PR #370 codex-review 指摘 P1 の回帰テスト: `gemm_alloc_peak_bytes` が
/// `BackendOps::gemm` の実ヒープ確保（出力 `Vec<f32>`・BLIS パッキング
/// バッファ）を実測しており、`MemoryOps` 経由の理論値（`peak_bytes`）から
/// 独立していることを確認する（旧 `peak_memory::tests::
/// run_peak_memory_cpu_gemm_alloc_peak_bytes_reflects_real_gemm_allocation`
/// を本バイナリへ移設。`gemm_alloc_peak_bytes` は `TrackingAllocator` が
/// `#[global_allocator]` として有効化されている場合のみ `Some` になる契約
/// のため、`cargo test --lib` では検証できない）。
fn check_gemm_alloc_peak_bytes_reflects_real_gemm_allocation() {
    let size = 256;
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, size, 2).unwrap();
    let report = run_peak_memory(&config).unwrap();

    let output_buffer_bytes = (size * size * std::mem::size_of::<f32>()) as u64;
    for trial in &report.samples {
        let observed = trial.gemm_alloc_peak_bytes.expect(
            "CPU バックエンドは TrackingAllocator が本テストバイナリの \
             #[global_allocator] のため Some のはず",
        );
        assert!(
            observed >= output_buffer_bytes,
            "gemm_alloc_peak_bytes は少なくとも出力バッファ分（{output_buffer_bytes} \
             バイト）は観測されるはず（実測: {observed}）"
        );
    }
}

/// [`bench_harness::peak_memory::PeakMemoryReport::require_gemm_alloc_tracked`]
/// の正常系: `TrackingAllocator` が有効化されている本バイナリでは
/// `gemm_alloc_peak_bytes` が `Some` になるため、CPU の公式実測記録と同じ
/// 形のレポートを素通しできることを確認する（旧
/// `peak_memory::tests::require_gemm_alloc_tracked_accepts_cpu_report_with_tracking_active`
/// を本バイナリへ移設）。
fn check_require_gemm_alloc_tracked_accepts_cpu_report_with_tracking_active() {
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 2).unwrap();
    let report = run_peak_memory(&config).unwrap();
    assert!(report.require_gemm_alloc_tracked().is_ok());
}

/// [`bench_harness::peak_memory::emit_peak_memory_report`] の正常系:
/// `TrackingAllocator` が有効化されている本バイナリでは通常の CPU 計測結果が
/// 標準出力経路（`out: None`）で成功することを確認する（旧
/// `peak_memory_bench.rs::tests::emit_report_accepts_tracked_cpu_report_via_stdout`
/// を本バイナリへ移設。イシュー #161 PR #357 CI 失敗の原因テストそのもの）。
fn check_emit_peak_memory_report_accepts_tracked_cpu_report_via_stdout() {
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 2).unwrap();
    let report = run_peak_memory(&config).unwrap();
    assert!(emit_peak_memory_report(&report, None).is_ok());
}

/// 検査関数 1 件のエントリ（名前・実行本体）。`main()` が順番に呼び、
/// panic を捕捉して合否レポートに集約する。
struct Check {
    name: &'static str,
    run: fn(),
}

/// 全検査を単一スレッドで順番に実行する（`harness = false` の `fn main()`）。
///
/// libtest の代替として: 各検査を `std::panic::catch_unwind` で個別に実行し
/// 失敗を集約する。1 件でも失敗すれば非 0 で終了する（`harness = false` は
/// 「何もしなければ成功で終了する」ため、検査を誤って空にした場合に
/// サイレント成功しないよう、実行した検査数もログへ出す）。
fn main() {
    let checks: &[Check] = &[
        Check {
            name: "measure_reflects_real_heap_allocation",
            run: check_measure_reflects_real_heap_allocation,
        },
        Check {
            name: "measure_clears_previous_interval",
            run: check_measure_clears_previous_interval,
        },
        Check {
            name: "measure_serializes_concurrent_callers",
            run: check_measure_serializes_concurrent_callers,
        },
        Check {
            name: "gemm_alloc_peak_bytes_reflects_real_gemm_allocation",
            run: check_gemm_alloc_peak_bytes_reflects_real_gemm_allocation,
        },
        Check {
            name: "require_gemm_alloc_tracked_accepts_cpu_report_with_tracking_active",
            run: check_require_gemm_alloc_tracked_accepts_cpu_report_with_tracking_active,
        },
        Check {
            name: "emit_peak_memory_report_accepts_tracked_cpu_report_via_stdout",
            run: check_emit_peak_memory_report_accepts_tracked_cpu_report_via_stdout,
        },
    ];

    let mut failures: Vec<&'static str> = Vec::new();
    for check in checks {
        print!("test alloc_tracker_serial::{} ... ", check.name);
        let result = std::panic::catch_unwind(check.run);
        match result {
            Ok(()) => println!("ok"),
            Err(_) => {
                println!("FAILED");
                failures.push(check.name);
            }
        }
    }

    println!(
        "\nalloc_tracker_serial: {} 件実行・{} 件失敗（single-threaded, no libtest harness）",
        checks.len(),
        failures.len()
    );

    if !failures.is_empty() {
        eprintln!("失敗した検査: {failures:?}");
        std::process::exit(1);
    }
}
