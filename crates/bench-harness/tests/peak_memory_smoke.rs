//! GEMM ピークメモリ計測ハーネス（TASK-14.2a・イシュー #178）の CI 実行可能な
//! スモークテスト。
//!
//! REQ-14 の代表ワークロード自体（M=N=K=4096）は self-hosted runner 上でも
//! 数百 MB〜数 GB の確保・GEMM 実行を伴い CI の通常テストとしては重いため、
//! 本ファイルは小サイズ（256³）で `bench_harness::peak_memory` の公開契約
//! （内部計測 API 値の決定性・リーク検査・JSON ラウンドトリップ）のみを
//! 検証する。4096³ 実測本体は `docs/perf/gemm-peak-memory-measurement.md`
//! （#178 実測記録。手動実行 `make peak-memory-bench` 経由）が担う。本ファイルは
//! それとは別に、当該実測記録の生データ（`docs/perf/peak-memory/cpu-run{1,2}.json`）を
//! 実際に読み込み、現行スキーマ（`gemm_alloc_peak_bytes` 込み）を満たしていることを
//! 自己保証する（`committed_cpu_peak_memory_reports_have_gemm_alloc_peak_bytes_tracked`。
//! PR #370 codex-review 指摘 P1 再指摘対応）。
//!
//! CUDA/Metal の実機依存経路は `#[ignore]` で分離する
//! （`.claude/rules/coding-rust.md`「実機依存テストは #[ignore] で分離」）。

use bench_harness::peak_memory::{
    PeakMemoryBackend, PeakMemoryConfig, PeakMemoryReport, run_peak_memory,
};
use std::path::Path;

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

/// `docs/perf/peak-memory/cpu-run1.json`・`cpu-run2.json`（#178 実測記録本体。
/// `docs/perf/gemm-peak-memory-measurement.md` が参照する生データ）を実際に読み込み、
/// 現行スキーマ（`gemm_alloc_peak_bytes` を含む）の実測結果として自己保証する。
///
/// PR #370 codex-review 指摘 P1 の回帰テスト: 「`gemm_alloc_peak_bytes` フィールド追加後も
/// 旧スキーマ（当該フィールド欠落）の JSON が `PeakMemoryReport::from_json`（`Option` の
/// 欠落は serde により `None` として受理される）を素通ししてしまい、`validate` も CPU 側の
/// `None` を拒否していなかった」ことを直接検証する。ファイル不在・パース失敗・
/// `require_gemm_alloc_tracked` 失敗のいずれも本テストの失敗として扱う（skip しない。
/// スキップは検査の迂回になるため許容しない）。
#[test]
fn committed_cpu_peak_memory_reports_have_gemm_alloc_peak_bytes_tracked() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR は cargo test 実行時に必ず設定される");
    let peak_memory_dir = Path::new(&manifest_dir)
        .parent()
        .and_then(Path::parent)
        .expect("crates/bench-harness からリポジトリルートへ 2 階層上れるはず")
        .join("docs/perf/peak-memory");

    for filename in ["cpu-run1.json", "cpu-run2.json"] {
        let path = peak_memory_dir.join(filename);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{path:?} の読み込みに失敗: {e}"));
        let report = PeakMemoryReport::from_json(&json)
            .unwrap_or_else(|e| panic!("{path:?} は現行スキーマの検証を通過するはず: {e}"));

        assert_eq!(report.backend, "cpu", "{path:?} は CPU 実測記録のはず");
        report.require_gemm_alloc_tracked().unwrap_or_else(|e| {
            panic!("{path:?} は gemm_alloc_peak_bytes が実測された現行の実測データのはず: {e}")
        });
        for (i, trial) in report.samples.iter().enumerate() {
            assert!(
                trial.gemm_alloc_peak_bytes.is_some(),
                "{path:?} の samples[{i}] は gemm_alloc_peak_bytes が Some のはず"
            );
        }
    }
}

/// `docs/perf/peak-memory/metal-run1.json`・`metal-run2.json`（イシュー #385
/// 実測記録本体。`docs/perf/gemm-peak-memory-measurement.md` の Metal 実機実測結果
/// 節が参照する生データ）を実際に読み込み、Metal 契約（`gemm_alloc_peak_bytes` が
/// 常に `None`。`GlobalAlloc` フックは CPU 専用で Metal の `MTLBuffer` 確保は
/// これを経由しない。`crates/bench-harness/src/peak_memory.rs` モジュールコメント
/// 「計測対象の粒度」参照）を満たすスキーマとして自己保証する。
///
/// `committed_cpu_peak_memory_reports_have_gemm_alloc_peak_bytes_tracked` の Metal
/// ミラー。JSON の読み込みと DTO 検証のみで GPU を必要としないため `#[ignore]` に
/// せず、`cfg(target_os = "macos")` でも囲まない（Linux の通常 CI でも実行し、
/// コミット済み実測データの改ざん・スキーマ退行を継続的に検出する）。ファイル不在・
/// パース失敗・`require_gemm_alloc_tracked` 失敗のいずれも本テストの失敗として扱う
/// （skip しない。CPU 版と同じ fail-closed 方針）。
#[test]
fn committed_metal_peak_memory_reports_have_theoretical_minimum_peak_bytes() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR は cargo test 実行時に必ず設定される");
    let peak_memory_dir = Path::new(&manifest_dir)
        .parent()
        .and_then(Path::parent)
        .expect("crates/bench-harness からリポジトリルートへ 2 階層上れるはず")
        .join("docs/perf/peak-memory");

    for filename in ["metal-run1.json", "metal-run2.json"] {
        let path = peak_memory_dir.join(filename);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{path:?} の読み込みに失敗: {e}"));
        let report = PeakMemoryReport::from_json(&json)
            .unwrap_or_else(|e| panic!("{path:?} は現行スキーマの検証を通過するはず: {e}"));

        assert_eq!(report.backend, "metal", "{path:?} は Metal 実測記録のはず");
        assert_eq!(
            report.theoretical_min_bytes, 201_326_592,
            "{path:?} は REQ-14 代表ワークロード（M=N=K=4096, f32）の理論最小ワーキングセットのはず"
        );
        report.require_gemm_alloc_tracked().unwrap_or_else(|e| {
            panic!(
                "{path:?} は Metal 契約（gemm_alloc_peak_bytes が全 trial で None）を満たすはず: {e}"
            )
        });
        for (i, trial) in report.samples.iter().enumerate() {
            assert!(
                trial.gemm_alloc_peak_bytes.is_none(),
                "{path:?} の samples[{i}] は Metal が GlobalAlloc 非経由のため gemm_alloc_peak_bytes が None のはず"
            );
        }
    }
}

/// `docs/perf/peak-memory/cuda-run1.json`・`cuda-run2.json`（イシュー #392
/// 実測記録本体。`docs/perf/gemm-peak-memory-measurement.md` の CUDA 実機実測結果
/// 節が参照する生データ）を実際に読み込み、CUDA 契約（`gemm_alloc_peak_bytes` が
/// 常に `None`。`cudarc` の driver 確保は Rust の `GlobalAlloc` を経由しない。
/// `crates/bench-harness/src/peak_memory.rs::run_cuda_trial` 参照）を満たす
/// スキーマとして自己保証する。
///
/// `committed_metal_peak_memory_reports_have_theoretical_minimum_peak_bytes` の
/// 直接ミラー（CUDA 版）。JSON の読み込みと DTO 検証のみで GPU を必要としないため
/// `#[ignore]` にせず、`cfg` でも囲まない（Linux の通常 CI・CUDA 非搭載環境でも実行し、
/// コミット済み実測データの改ざん・スキーマ退行を継続的に検出する）。ファイル不在・
/// パース失敗・`require_gemm_alloc_tracked` 失敗のいずれも本テストの失敗として扱う
/// （skip しない。CPU・Metal 版と同じ fail-closed 方針）。
#[test]
fn committed_cuda_peak_memory_reports_have_theoretical_minimum_peak_bytes() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR は cargo test 実行時に必ず設定される");
    let peak_memory_dir = Path::new(&manifest_dir)
        .parent()
        .and_then(Path::parent)
        .expect("crates/bench-harness からリポジトリルートへ 2 階層上れるはず")
        .join("docs/perf/peak-memory");

    for filename in ["cuda-run1.json", "cuda-run2.json"] {
        let path = peak_memory_dir.join(filename);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{path:?} の読み込みに失敗: {e}"));
        let report = PeakMemoryReport::from_json(&json)
            .unwrap_or_else(|e| panic!("{path:?} は現行スキーマの検証を通過するはず: {e}"));

        assert_eq!(report.backend, "cuda", "{path:?} は CUDA 実測記録のはず");
        assert_eq!(
            report.theoretical_min_bytes, 201_326_592,
            "{path:?} は REQ-14 代表ワークロード（M=N=K=4096, f32）の理論最小ワーキングセットのはず"
        );
        report.require_gemm_alloc_tracked().unwrap_or_else(|e| {
            panic!(
                "{path:?} は CUDA 契約（gemm_alloc_peak_bytes が全 trial で None）を満たすはず: {e}"
            )
        });
        for (i, trial) in report.samples.iter().enumerate() {
            assert!(
                trial.gemm_alloc_peak_bytes.is_none(),
                "{path:?} の samples[{i}] は CUDA が GlobalAlloc 非経由のため gemm_alloc_peak_bytes が None のはず"
            );
        }
    }
}
