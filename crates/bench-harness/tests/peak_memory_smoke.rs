//! GEMM ピークメモリ計測ハーネス（TASK-14.2a・イシュー #178）の CI 実行可能な
//! スモークテスト。
//!
//! REQ-14 の代表ワークロード自体（M=N=K=4096）は self-hosted runner 上でも
//! 数百 MB〜数 GB の確保・GEMM 実行を伴い CI の通常テストとしては重いため、
//! 本ファイルは小サイズ（256³）で `bench_harness::peak_memory` の公開契約
//! （内部計測 API 値の決定性・リーク検査・JSON ラウンドトリップ）のみを
//! 検証する。4096³ 実測本体は `docs/perf/gemm-peak-memory-measurement.md`
//! （#178 実測記録。手動実行 `make peak-memory-bench` 経由）が担う。
//!
//! CUDA/Metal の実機依存経路は `#[ignore]` で分離する
//! （`.claude/rules/coding-rust.md`「実機依存テストは #[ignore] で分離」）。

use bench_harness::peak_memory::{
    PeakMemoryBackend, PeakMemoryConfig, PeakMemoryReport, run_peak_memory,
};

/// 受け入れ条件「CPU バックエンドでピーク値が取得できる」の統合テスト版:
/// `MemoryOps` 経由の確保（A・B・C の 3 バッファ）のみが計上され、理論最小
/// ワーキングセット（3 × size² × 4 バイト）とちょうど一致することを確認する
/// （`crates/bench-harness/src/peak_memory.rs` 単体テストと同じ主張を、
/// クレート公開 API 経由の統合テストとして再検証する）。
#[test]
fn cpu_peak_memory_matches_theoretical_minimum() {
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 256, 5).unwrap();
    let report = run_peak_memory(&config).unwrap();

    let expected_bytes: u64 = 3 * 256 * 256 * 4;
    assert_eq!(report.theoretical_min_bytes, expected_bytes);
    assert_eq!(report.backend, "cpu");
    assert_eq!(report.trials, 5);
    assert_eq!(report.m, 256);
    assert_eq!(report.n, 256);
    assert_eq!(report.k, 256);
    assert_eq!(report.dtype, "f32");

    for trial in &report.samples {
        assert_eq!(trial.peak_bytes, expected_bytes);
        assert_eq!(
            trial.allocated_after_drop_bytes, 0,
            "drop 後にリークがあってはならない"
        );
    }
    assert_eq!(report.peak_bytes.median, expected_bytes);
    assert_eq!(report.peak_bytes.q1, expected_bytes);
    assert_eq!(report.peak_bytes.q3, expected_bytes);
}

/// JSON ラウンドトリップが [`PeakMemoryReport::validate`] を通過し続けることを
/// 確認する（`.claude/rules/security.md` A08「検証済み DTO」方針の統合テスト）。
#[test]
fn report_json_roundtrip_is_stable() {
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Cpu, 128, 3).unwrap();
    let report = run_peak_memory(&config).unwrap();

    let json = report.to_json().unwrap();
    let restored = PeakMemoryReport::from_json(&json).unwrap();
    assert_eq!(report, restored);
}

/// CUDA 実機（driver/NVRTC 搭載環境）依存の経路。本セッションの実行環境
/// （libnvrtc 未導入）では `DeviceUnavailable` で fail-closed に終わることを
/// 実地確認済み（`docs/perf/gemm-peak-memory-measurement.md` 参照）。
/// 通常 CI（self-hosted・CUDA 非搭載）では `#[ignore]` により実行しない。
#[test]
#[ignore = "CUDA 実機（driver/NVRTC 搭載）依存。DGX Spark GB10 等でのみ実行する"]
fn cuda_peak_memory_matches_theoretical_minimum() {
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Cuda, 256, 5).unwrap();
    let report = run_peak_memory(&config).expect("CUDA 実機環境でのみ成功する");
    let expected_bytes: u64 = 3 * 256 * 256 * 4;
    assert_eq!(report.theoretical_min_bytes, expected_bytes);
}

/// Metal 実機（Apple Silicon）依存の経路。`cfg(target_os = "macos")` 限定
/// （非 macOS では `PeakMemoryError::DeviceUnavailable` を返す契約。
/// `crates/bench-harness/src/peak_memory.rs::run_metal_trial` 参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存"]
fn metal_peak_memory_matches_theoretical_minimum() {
    let config = PeakMemoryConfig::new(PeakMemoryBackend::Metal, 256, 5).unwrap();
    let report = run_peak_memory(&config).expect("Metal 実機環境でのみ成功する");
    let expected_bytes: u64 = 3 * 256 * 256 * 4;
    assert_eq!(report.theoretical_min_bytes, expected_bytes);
}
