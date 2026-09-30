//! `ReadbackDest::PinnedStagingReuse`（イシュー #2108）が `PretouchedFresh` と bit 完全一致の出力を
//! 返し、pinned staging が再利用・世代破棄されることを実機で確認する `#[ignore]` テスト。
//!
//! CUDA driver のみで動く（NVRTC 不要）。`internal-diagnostics` feature 必須（feature 無効時は
//! 本ファイル全体が空になる。`Cargo.toml` を変えないための file-level cfg）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test readback_reuse_bit_match_2108 -- --ignored --nocapture --test-threads=1
//! ```
#![cfg(feature = "internal-diagnostics")]

use fandhe_ai_backend_cuda::CudaDevice;
use fandhe_ai_backend_cuda::memory::{
    readback_f16_policy_diag, readback_f32_policy_diag, readback_staging_stats_diag,
    release_readback_staging_diag,
};
use half::f16;

/// NaN payload・±0・inf・subnormal を含む決定的パターン。
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
fn pinned_reuse_is_bit_identical_and_reuses_staging_f32() {
    let device = CudaDevice::new(0).expect("CUDA device 0 required");
    let stream = device.stream();
    release_readback_staging_diag();
    for &numel in &[0usize, 1, 37, 4099, 1024 * 1024, 2048 * 2048] {
        let data = pattern(numel);
        let dev = stream.clone_htod(&data).expect("H2D");
        stream.synchronize().expect("sync");
        let base = readback_f32_policy_diag(stream, &dev, false).expect("pretouched");
        let (h0, _, _, _) = readback_staging_stats_diag();
        for round in 0..3 {
            let got = readback_f32_policy_diag(stream, &dev, true).expect("pinned-reuse");
            assert_eq!(got.len(), numel);
            for i in 0..numel {
                assert_eq!(got[i].to_bits(), base[i].to_bits(), "numel={numel} i={i}");
                assert_eq!(got[i].to_bits(), data[i].to_bits());
            }
            let _ = round;
        }
        let (h1, _, _, _) = readback_staging_stats_diag();
        if numel > 0 {
            assert!(h1 - h0 >= 2, "2 回目以降は staging 再利用 (numel={numel})");
        }
    }
    let freed = release_readback_staging_diag();
    assert!(freed > 0);
    assert_eq!(readback_staging_stats_diag().3, 0);
}

#[test]
#[ignore = "CUDA 実機必須"]
fn non_f32_falls_back_to_pretouched_f16() {
    let device = CudaDevice::new(0).expect("CUDA device 0 required");
    let stream = device.stream();
    for &numel in &[0usize, 1, 37, 65536] {
        let data: Vec<f16> = (0..numel)
            .map(|i| f16::from_f32((i as f32) * 0.03125 - 512.0))
            .collect();
        let dev = stream.clone_htod(&data).expect("H2D");
        stream.synchronize().expect("sync");
        let a = readback_f16_policy_diag(stream, &dev, false).expect("pretouched");
        let b = readback_f16_policy_diag(stream, &dev, true).expect("pinned-reuse");
        assert_eq!(a.len(), b.len());
        for i in 0..numel {
            assert_eq!(a[i].to_bits(), b[i].to_bits());
            assert_eq!(a[i].to_bits(), data[i].to_bits());
        }
    }
}
