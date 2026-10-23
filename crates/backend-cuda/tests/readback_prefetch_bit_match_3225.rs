//! 背景事前タッチプール（`ReadbackDest::PrefetchedPretouched`、イシュー #3225・PR #3540）が
//! `PretouchedFresh` と bit 完全一致の出力を返し、プールが命中・解放されることを GB10 実機で
//! 確認する `#[ignore]` テスト。A/B 計測（`run_ab_readback_prefetch_cuda.sh`）の前提ゲート。
//!
//! CUDA driver のみで動く（NVRTC 不要）。`internal-diagnostics` feature 必須（無効時は本ファイル
//! 全体が空になる。`Cargo.toml` を変えないための file-level cfg）。統計はプロセス共有のため
//! `--test-threads=1` 前提で差分比較する。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test readback_prefetch_bit_match_3225 -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "internal-diagnostics")]

use std::thread::sleep;
use std::time::Duration;

use fandhe_ai_backend_cuda::CudaDevice;
use fandhe_ai_backend_cuda::memory::{
    readback_f32_diag, readback_f32_prefetched_diag, readback_prefetch_stats_diag,
    release_readback_prefetch,
};

/// NaN payload・±0・inf・subnormal を含む決定的パターン（#2108 と同一）。
fn pattern(numel: usize) -> Vec<f32> {
    (0..numel)
        .map(|i| match i % 11 {
            0 => f32::from_bits(0x7fc0_1234),
            1 => -0.0,
            2 => 0.0,
            3 => f32::INFINITY,
            4 => f32::NEG_INFINITY,
            5 => f32::from_bits(1),
            _ => (i as f32) * 0.125 - 12345.0,
        })
        .collect()
}

#[test]
#[ignore = "CUDA 実機必須"]
fn prefetched_is_bit_identical_and_hits_pool_f32() {
    let device = CudaDevice::new(0).expect("CUDA device 0 required");
    let stream = device.stream();
    let _ = release_readback_prefetch();
    // 1 MiB 未満は従来経路へ落ちる（37・4099）。512*512 f32 = ちょうど 1 MiB から対象。
    for &(numel, eligible) in &[
        (0usize, false),
        (1, false),
        (37, false),
        (4099, false),
        (512 * 512, true),
        (1024 * 1024, true),
        (2048 * 2048, true),
        (4096 * 4096, true),
    ] {
        let data = pattern(numel);
        let dev = stream.clone_htod(&data).expect("H2D");
        stream.synchronize().expect("sync");
        let base = readback_f32_diag(stream, &dev, true).expect("pretouched-fresh");
        let (h0, _, _, _) = readback_prefetch_stats_diag();
        for _ in 0..5 {
            let got = readback_f32_prefetched_diag(stream, &dev).expect("prefetched");
            assert_eq!(got.len(), numel);
            for i in 0..numel {
                assert_eq!(got[i].to_bits(), base[i].to_bits(), "numel={numel} i={i}");
                assert_eq!(got[i].to_bits(), data[i].to_bits());
            }
            // 非同期 worker の補充完了を待つ（wait_timeout 20 ms より長く取る）。
            sleep(Duration::from_millis(50));
        }
        let (h1, _, _, _) = readback_prefetch_stats_diag();
        if eligible {
            assert!(
                h1 - h0 >= 1,
                "対象 numel ではプール命中が出る (numel={numel})"
            );
        } else {
            assert_eq!(h1, h0, "対象外 numel はプールを使わない (numel={numel})");
        }
        let freed = release_readback_prefetch();
        if eligible {
            assert!(freed > 0, "対象 numel の後は解放バイトが正 (numel={numel})");
        }
        println!(
            "numel={numel} eligible={eligible} hits_delta={} freed={freed}",
            h1 - h0
        );
    }
    assert_eq!(readback_prefetch_stats_diag().3, 0);
}
