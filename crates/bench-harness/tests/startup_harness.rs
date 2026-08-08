//! 起動コスト計測ハーネスの統合テスト（TASK-13.1a・イシュー #170）。
//!
//! CPU バックエンド経路は self-hosted CI で実行可能な通常テストとして検証する
//! （常に利用可能なため。`.claude/rules/coding-rust.md`）。CUDA 実機（DGX Spark GB10）
//! 依存の cold/warm E2E は `#[ignore]` で分離し、通常 CI では実行しない
//! （`.claude/rules/ci.md`「実機依存」節）。実行コマンド:
//! `cargo test -p bench-harness --release -- --ignored`（実機で TASK-13.1b・#171 が使用）。
//!
//! probe バイナリのパス解決には `CARGO_BIN_EXE_<name>`（Cargo が同一パッケージの
//! `[[bin]]` ターゲットに対して自動設定する環境変数）を使う。`startup_bench` CLI
//! （本番導線）が使う `current_exe()` 相対解決とは異なる経路だが、いずれも
//! 「`startup_probe` と同時にビルドされた実体を指す」という契約は同じ。

use bench_harness::startup::{StartupBackend, StartupConfig, StartupPhase, run_phase};

fn probe_path() -> &'static str {
    env!("CARGO_BIN_EXE_startup_probe")
}

/// 受け入れ条件「コールド／ウォーム双方の計測が再現可能」を検証する:
/// 同一設定で 2 回計測しても、いずれもプロトコル遵守済み（`validate()` 済み）の
/// `StartupReport` が得られることを確認する（値そのものの再現一致までは求めない。
/// タイミング計測は環境ノイズを含むため。TASK-13.1b の実測比較で厳密な再現性を扱う）。
#[test]
fn cpu_cold_and_warm_phases_are_reproducibly_measurable() {
    let config = StartupConfig::new(StartupBackend::Cpu, 3, probe_path())
        .expect("trials=3 は下限（1 以上）を満たすため成功するはず");

    for phase in [StartupPhase::Cold, StartupPhase::Warm] {
        for attempt in 0..2 {
            let report = run_phase(&config, phase).unwrap_or_else(|e| {
                panic!(
                    "CPU {:?} フェーズの計測に失敗（attempt={attempt}）: {e}",
                    phase
                )
            });
            report
                .validate()
                .expect("run_phase が返すレポートは常にプロトコル遵守済みのはず");
            assert_eq!(report.backend, "cpu");
            assert_eq!(report.trials, 3);
            assert_eq!(report.samples.len(), 3);
        }
    }
}

/// [`bench_harness::startup::StartupReport`] の JSON シリアライズ往復が
/// 実際の計測結果に対しても成立することを確認する（`startup.rs` 単体テストは
/// 手動構築したサンプルのみを対象とするため、実計測結果での往復も別途検証する）。
#[test]
fn cpu_warm_report_json_roundtrip_matches_original() {
    let config = StartupConfig::new(StartupBackend::Cpu, 2, probe_path()).unwrap();
    let report = run_phase(&config, StartupPhase::Warm).expect("CPU 計測は常に成功するはず");

    let json = report
        .to_json()
        .expect("プロトコル遵守済みのため成功するはず");
    let round_tripped = bench_harness::startup::StartupReport::from_json(&json)
        .expect("往復デコードは成功するはず");
    assert_eq!(report, round_tripped);
}

/// CUDA 実機（DGX Spark GB10）依存の cold/warm E2E。
///
/// `CudaDevice::is_available()` に相当する可用性はハーネス内部（`startup_probe`
/// バイナリの CUDA 経路）で判定するため、本テストは非実機環境では
/// `run_phase` がエラーを返すことのみを期待する（実機では成功する想定。
/// `#[ignore]` 分離により通常 CI（self-hosted・実機なし）では実行されない）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。cargo test -p bench-harness --release -- --ignored"]
fn cuda_cold_and_warm_phases_are_reproducibly_measurable_on_real_hardware() {
    let config = StartupConfig::new(StartupBackend::Cuda, 3, probe_path()).unwrap();

    for phase in [StartupPhase::Cold, StartupPhase::Warm] {
        let report = run_phase(&config, phase)
            .unwrap_or_else(|e| panic!("CUDA {:?} フェーズの計測に失敗: {e}", phase));
        report.validate().expect("プロトコル遵守済みのはず");
        assert_eq!(report.backend, "cuda");
        assert_eq!(report.trials, 3);
    }
}
