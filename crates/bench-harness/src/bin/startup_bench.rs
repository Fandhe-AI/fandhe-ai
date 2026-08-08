//! 起動コスト計測を駆動する CLI（TASK-13.1a・イシュー #170）。
//!
//! `bench_harness::startup::run_phase` を呼び出し、コールド／ウォーム双方の
//! [`bench_harness::startup::StartupReport`] を JSON として標準出力（または `--out` 指定先）へ
//! 書き出す。実測の実施・v1（PoC-5）との差分記録は兄弟イシュー #171（TASK-13.1b）が
//! 本バイナリを再利用する想定であり、本イシューでは導線の整備のみを行う
//! （`Makefile` の `startup-bench` ターゲットから同じコマンドで再現できる）。
//!
//! 引数パースは外部 CLI クレートを追加せず `std` のみで実装する（許容依存 8 区分外の
//! 新規追加を避けるため。`.claude/rules/deps-policy.md`）。

use bench_harness::startup::{
    DEFAULT_STARTUP_TRIALS, StartupBackend, StartupConfig, StartupPhase, run_phase,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Args {
    backend: StartupBackend,
    trials: usize,
    out: Option<PathBuf>,
}

/// `--backend <cpu|cuda|metal>`（必須）・`--trials <N>`（既定 [`DEFAULT_STARTUP_TRIALS`]）・
/// `--out <path>`（省略時は標準出力）を許可リスト方式で解釈する。
fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut backend: Option<StartupBackend> = None;
    let mut trials = DEFAULT_STARTUP_TRIALS;
    let mut out: Option<PathBuf> = None;

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--backend" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| "--backend には値が必要".to_string())?;
                backend = Some(StartupBackend::parse(value).map_err(|e| e.to_string())?);
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
                    "未知の引数 {other:?}（--backend / --trials / --out のいずれかを指定）"
                ));
            }
        }
    }

    let backend = backend.ok_or_else(|| "--backend <cpu|cuda|metal> は必須".to_string())?;
    Ok(Args {
        backend,
        trials,
        out,
    })
}

/// `startup_probe` バイナリのパスを解決する。
///
/// 同一ビルドディレクトリ内で `startup_bench` と `startup_probe` は必ず並んで生成される
/// （両者とも本クレートの `src/bin/*.rs`。`cargo build -p bench-harness --bins` で同時に
/// ビルドされる契約）ため、`current_exe()` の親ディレクトリから解決する。
fn resolve_probe_path() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("自身の実行パス取得失敗: {e}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| format!("実行パスに親ディレクトリがない: {exe:?}"))?;
    let probe_name = if cfg!(windows) {
        "startup_probe.exe"
    } else {
        "startup_probe"
    };
    let probe_path = dir.join(probe_name);
    if !probe_path.is_file() {
        return Err(format!(
            "startup_probe が見つからない（{probe_path:?}）。`cargo build -p bench-harness --bins` で \
             startup_bench と同時にビルドされているか確認する"
        ));
    }
    Ok(probe_path)
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("引数エラー: {e}");
            eprintln!(
                "使用法: startup_bench --backend <cpu|cuda|metal> [--trials N] [--out path.json]"
            );
            return ExitCode::from(2);
        }
    };

    let probe_path = match resolve_probe_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let config = match StartupConfig::new(args.backend, args.trials, probe_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("設定エラー: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut phase_reports = Vec::new();
    for phase in [StartupPhase::Cold, StartupPhase::Warm] {
        match run_phase(&config, phase) {
            Ok(report) => phase_reports.push(report),
            Err(e) => {
                eprintln!("{} フェーズの計測失敗: {e}", phase.as_str());
                return ExitCode::FAILURE;
            }
        }
    }

    let json_reports: Result<Vec<String>, _> = phase_reports.iter().map(|r| r.to_json()).collect();
    let json_reports = match json_reports {
        Ok(v) => v,
        Err(e) => {
            eprintln!("レポート JSON エンコード失敗: {e}");
            return ExitCode::FAILURE;
        }
    };
    let combined = format!("[{}]", json_reports.join(","));

    match args.out {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &combined) {
                eprintln!("出力ファイル書き込み失敗（{path:?}）: {e}");
                return ExitCode::FAILURE;
            }
        }
        None => println!("{combined}"),
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
        let raw: Vec<String> = ["--backend", "cpu", "--trials", "3", "--out", "out.json"]
            .into_iter()
            .map(String::from)
            .collect();
        let args = parse_args(&raw).unwrap();
        assert_eq!(args.backend, StartupBackend::Cpu);
        assert_eq!(args.trials, 3);
        assert_eq!(args.out, Some(PathBuf::from("out.json")));
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
}
