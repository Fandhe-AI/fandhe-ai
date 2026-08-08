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
//! 引数パースは外部 CLI クレートを追加せず `std` のみで実装する（許容依存 8 区分外の
//! 新規追加を避けるため。`.claude/rules/deps-policy.md`。`startup_bench.rs` と同型の方針）。

use bench_harness::alloc_tracker::TrackingAllocator;
use bench_harness::peak_memory::{
    DEFAULT_GEMM_SIZE, DEFAULT_PEAK_MEMORY_TRIALS, PeakMemoryBackend, PeakMemoryConfig,
    PeakMemoryError, PeakMemoryReport, run_peak_memory,
};
use std::path::{Path, PathBuf};
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

#[derive(Debug)]
struct Args {
    backend: PeakMemoryBackend,
    size: usize,
    trials: usize,
    out: Option<PathBuf>,
}

/// `--backend <cpu|cuda|metal>`（必須）・`--size <N>`（既定 [`DEFAULT_GEMM_SIZE`]）・
/// `--trials <N>`（既定 [`DEFAULT_PEAK_MEMORY_TRIALS`]）・`--out <path>`
/// （省略時は標準出力）を許可リスト方式で解釈する（`startup_bench.rs::parse_args` と同型）。
fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut backend: Option<PeakMemoryBackend> = None;
    let mut size = DEFAULT_GEMM_SIZE;
    let mut trials = DEFAULT_PEAK_MEMORY_TRIALS;
    let mut out: Option<PathBuf> = None;

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--backend" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--backend には値が必要".to_string())?;
                backend = Some(PeakMemoryBackend::parse(value).map_err(|e| e.to_string())?);
                i += 2;
            }
            "--size" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--size には値が必要".to_string())?;
                size = value
                    .parse::<usize>()
                    .map_err(|e| format!("--size の値が不正（{value:?}）: {e}"))?;
                i += 2;
            }
            "--trials" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--trials には値が必要".to_string())?;
                trials = value
                    .parse::<usize>()
                    .map_err(|e| format!("--trials の値が不正（{value:?}）: {e}"))?;
                i += 2;
            }
            "--out" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--out には値が必要".to_string())?;
                out = Some(PathBuf::from(value));
                i += 2;
            }
            other => {
                return Err(format!(
                    "未知の引数 {other:?}（--backend / --size / --trials / --out のいずれかを指定）"
                ));
            }
        }
    }

    let backend = backend.ok_or_else(|| "--backend <cpu|cuda|metal> は必須".to_string())?;
    Ok(Args {
        backend,
        size,
        trials,
        out,
    })
}

/// 公式出力（標準出力または `--out` 先ファイル）を確定する唯一の経路。
///
/// 出力先の分岐（`out`）に関わらず、シリアライズより前に必ず
/// [`PeakMemoryReport::require_gemm_alloc_tracked`] を通す。`--out` 経路は
/// `write_to_file()` が内部で同メソッドを呼ぶため元々 fail-closed だったが、
/// 標準出力（`out: None`。`make peak-memory-bench` の既定経路）は `to_json()` の
/// みで検証をバイパスできてしまい、CPU の `gemm_alloc_peak_bytes` が `None` または
/// 過小でも「公式記録」として出力されてしまっていた（PR #370 codex-review 再指摘
/// P1 対応）。検証を `main()` の分岐の外側・本関数の入口に集約することで、将来
/// 出力経路を追加しても検証バイパスの抜け道にならない構造にする。
fn emit_report(report: &PeakMemoryReport, out: Option<&Path>) -> Result<(), PeakMemoryError> {
    report.require_gemm_alloc_tracked()?;
    match out {
        Some(path) => report.write_to_file(path),
        None => {
            let json = report.to_json()?;
            println!("{json}");
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
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

    if let Err(e) = emit_report(&report, args.out.as_deref()) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_requires_backend() {
        let err = parse_args(&[]).unwrap_err();
        assert!(err.contains("--backend"));
    }

    #[test]
    fn parse_args_accepts_full_form() {
        let raw: Vec<String> = [
            "--backend",
            "cpu",
            "--size",
            "128",
            "--trials",
            "3",
            "--out",
            "out.json",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let args = parse_args(&raw).unwrap();
        assert_eq!(args.backend, PeakMemoryBackend::Cpu);
        assert_eq!(args.size, 128);
        assert_eq!(args.trials, 3);
        assert_eq!(args.out, Some(PathBuf::from("out.json")));
    }

    #[test]
    fn parse_args_uses_defaults_when_omitted() {
        let raw: Vec<String> = ["--backend", "cpu"].into_iter().map(String::from).collect();
        let args = parse_args(&raw).unwrap();
        assert_eq!(args.size, DEFAULT_GEMM_SIZE);
        assert_eq!(args.trials, DEFAULT_PEAK_MEMORY_TRIALS);
        assert_eq!(args.out, None);
    }

    #[test]
    fn parse_args_rejects_unknown_flag() {
        let raw: Vec<String> = ["--bogus", "x"].into_iter().map(String::from).collect();
        assert!(parse_args(&raw).is_err());
    }

    #[test]
    fn parse_args_rejects_invalid_backend() {
        let raw: Vec<String> = ["--backend", "gpu"].into_iter().map(String::from).collect();
        assert!(parse_args(&raw).is_err());
    }

    /// [`emit_report`] の標準出力経路（`out: None`）が `gemm_alloc_peak_bytes` 欠落を
    /// fail-closed で拒否することの回帰テスト（PR #370 codex-review 再指摘 P1: `--out`
    /// 省略時は `to_json()` のみが呼ばれ検証をバイパスできていた欠陥そのものの再現）。
    /// 本バイナリはファイル冒頭で `TrackingAllocator` を `#[global_allocator]` 宣言する
    /// ため `run_peak_memory` の結果は通常 `Some` になるが、旧スキーマ実測データの
    /// 紛れ込みを模して明示的に `None` へ書き換えて検証する。
    #[test]
    fn emit_report_rejects_missing_gemm_alloc_tracking_via_stdout() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = None;

        assert!(matches!(
            emit_report(&report, None),
            Err(PeakMemoryError::ProtocolViolation(_))
        ));
    }

    /// [`emit_report`] の `--out` 経路でも同じ欠陥形（`gemm_alloc_peak_bytes` 欠落）が
    /// fail-closed で拒否されることを確認する（標準出力経路と対称であることの確認。
    /// `write_to_file()` 単体の回帰テストは `peak_memory.rs` 側にもあるが、CLI の
    /// 共通入口である本関数からも同じ契約が効くことをここで固定する）。
    #[test]
    fn emit_report_rejects_missing_gemm_alloc_tracking_via_out_file() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 1).unwrap();
        let mut report = run_peak_memory(&config).unwrap();
        report.samples[0].gemm_alloc_peak_bytes = None;

        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "emit_report_rejects_missing_gemm_alloc_tracking_via_out_file_{}.json",
            std::process::id()
        ));
        let result = emit_report(&report, Some(&path));

        assert!(matches!(result, Err(PeakMemoryError::ProtocolViolation(_))));
        assert!(!path.exists(), "検証失敗時にファイルを書き出してはならない");
    }

    /// [`emit_report`] の正常系: 本テストバイナリの `#[global_allocator]`
    /// （`TrackingAllocator`）により `gemm_alloc_peak_bytes` が `Some` になる通常の
    /// CPU 計測結果は、標準出力経路（`out: None`）で成功することを確認する。
    #[test]
    fn emit_report_accepts_tracked_cpu_report_via_stdout() {
        let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 32, 2).unwrap();
        let report = run_peak_memory(&config).unwrap();
        assert!(emit_report(&report, None).is_ok());
    }
}
