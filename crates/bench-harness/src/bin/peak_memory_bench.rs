//! GEMM ピークメモリ計測を駆動する CLI（TASK-14.2a・イシュー #178）。
//!
//! `bench_harness::peak_memory::run_peak_memory` を呼び出し、
//! [`bench_harness::peak_memory::PeakMemoryReport`] を JSON として標準出力
//! （または `--out` 指定先）へ書き出す。REQ-14 の代表ワークロード
//! （M=N=K=4096, f32）は既定値（[`bench_harness::peak_memory::
//! DEFAULT_GEMM_SIZE`]・[`bench_harness::peak_memory::
//! DEFAULT_PEAK_MEMORY_TRIALS`]）のみで到達できる（`--backend` のみ必須指定）。
//! `Makefile` の `peak-memory-bench` ターゲットから同じコマンドで再現できる。
//!
//! 引数パース（[`bench_harness::peak_memory::parse_peak_memory_cli_args`]）・
//! 出力経路（[`bench_harness::peak_memory::emit_peak_memory_report`]）は
//! ライブラリ側（`peak_memory.rs`）へ実装を寄せ、本バイナリは `main()` のみを
//! 持つ薄いラッパーとする（イシュー #161 PR #357 codex-review 再指摘 P1
//! 対応）。理由: 本バイナリは下記 `TrackingAllocator` を `#[cfg(test)]` なしで
//! 常時 `#[global_allocator]` 宣言するため、`cargo test --bin
//! peak_memory_bench` は全 `#[test]` がこの単一アロケータのカウンタを共有する
//! テストバイナリになる。テスト関数内ロック（旧 `serial_guard()`）は
//! libtest が `#[test]` 本体の外側（スレッド起動・終了）で行う並行処理を
//! 直列化できないため、干渉を根本排除するには「本バイナリに `#[test]` を
//! 一切持たせない」しかない（`bench_harness::alloc_tracker` モジュール冒頭
//! 「スレッド安全性」参照）。ロジックをライブラリへ移したことで、CLI 引数
//! パース・出力経路の回帰テストは `cargo test --lib`（`TrackingAllocator` を
//! 宣言しない通常のテストバイナリ）側でテストできる
//! （`peak_memory::tests` 参照）。実測 `gemm_alloc_peak_bytes` に依存する
//! テストは `tests/alloc_tracker_serial.rs`（`harness = false` の専用
//! プロセス）に集約する。

use bench_harness::alloc_tracker::TrackingAllocator;
use bench_harness::peak_memory::{
    PeakMemoryConfig, emit_peak_memory_report, parse_peak_memory_cli_args, run_peak_memory,
};
use std::process::ExitCode;

/// 本バイナリ限定で `TrackingAllocator` をプロセスの `#[global_allocator]`
/// として有効化する（`bench_harness::alloc_tracker` モジュール冒頭
/// 「適用範囲」参照。PR #370 codex-review 指摘 P1 対応）。`peak_memory::
/// run_cpu_trial` が呼ぶ `alloc_tracker::reset_peak`／
/// `peak_since_reset_bytes` は、実際にこの宣言があるバイナリでのみ
/// `Some` を返す。ライブラリクレート側（`bench_harness::lib.rs`）では
/// 宣言しないため、本クレートを `dev-dependencies` として参照する他クレート
/// のテストバイナリには影響しない。
#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_peak_memory_cli_args(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("引数エラー: {e}");
            eprintln!(
                "使用法: peak_memory_bench --backend <cpu|cuda|metal> [--size N] [--trials N] \
                 [--out path.json]"
            );
            return ExitCode::from(2);
        }
    };

    let config = match PeakMemoryConfig::new(args.backend, args.size, args.trials) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("設定エラー: {e}");
            return ExitCode::FAILURE;
        }
    };

    let report = match run_peak_memory(&config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("計測失敗: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = emit_peak_memory_report(&report, args.out.as_deref()) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

// 本バイナリは意図的に `#[cfg(test)] mod tests` を持たない（モジュール冒頭
// doc comment 参照）。引数パース・出力経路のテストは
// `bench_harness::peak_memory::tests`、`TrackingAllocator` 実測依存のテストは
// `tests/alloc_tracker_serial.rs` を参照。
