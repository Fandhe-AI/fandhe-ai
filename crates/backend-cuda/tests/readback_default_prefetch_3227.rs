//! 既定の `CudaMemory::download`（本番経路）が背景事前タッチプール（`ReadbackDest::PrefetchedPretouched`）
//! を通ることを GB10 実機で確かめる `#[ignore]` テスト（イシュー #3227・既定反転 PR #3552 の確認）。
//!
//! `readback_prefetch_bit_match_3225.rs` は宛先を診断入口で固定するため本番の既定経路を検証しない。
//! 本テストは `MemoryOps::download` を直接呼び、`FANDHE_AI_CUDA_READBACK_DEST` の有無で 2 腕を切り替える:
//! 未設定 = 既定腕（対象 numel でプール命中が出る）、`pretouched` = opt-out 腕（全 numel で命中 0）。
//! env の値はプロセス内で 1 回だけ解決される（OnceLock）ため、腕ごとに別プロセスで実行する。
//! 統計はプロセス共有のため `--test-threads=1` 前提。`internal-diagnostics` feature 必須。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test readback_default_prefetch_3227 -- --ignored --nocapture --test-threads=1
//! FANDHE_AI_CUDA_READBACK_DEST=pretouched cargo test ...（同上）
//! ```
#![cfg(feature = "internal-diagnostics")]

use std::thread::sleep;
use std::time::Duration;

use fandhe_ai_backend_cuda::memory::{readback_prefetch_stats_diag, release_readback_prefetch};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice, CudaMemory};
use fandhe_ai_tensor_core::{BackendOps, MemoryOps, Tensor};

/// NaN payload・±0・inf・subnormal を含む決定的パターン（#3225 と同一）。
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
fn default_readback_goes_through_prefetch_pool() {
    // 腕の取り違えを fail-closed で検出する（未設定 or "pretouched" 以外は失敗）。
    let default_arm = match std::env::var("FANDHE_AI_CUDA_READBACK_DEST") {
        Err(std::env::VarError::NotPresent) => true,
        Ok(v) if v == "pretouched" => false,
        other => panic!("FANDHE_AI_CUDA_READBACK_DEST は未設定か pretouched のみ許可: {other:?}"),
    };
    println!("arm={}", if default_arm { "default" } else { "optout" });

    let device = CudaDevice::new(0).expect("CUDA device 0 required");
    let mem = CudaMemory::new(&device);
    let _ = release_readback_prefetch();

    let mut max_cached = 0u64;
    for &(numel, eligible) in &[
        (37usize, false),
        (4099, false),
        (512 * 512, true),
        (1024 * 1024, true),
        (2048 * 2048, true),
        (4096 * 4096, true),
    ] {
        let data = pattern(numel);
        let tensor = Tensor::<f32>::new(data.clone(), &[numel]).expect("tensor");
        let buf = mem.upload(&tensor).expect("upload");
        let (h0, _, _, _) = readback_prefetch_stats_diag();
        for _ in 0..5 {
            let back = mem.download(&buf).expect("download");
            let got = back.as_slice().expect("contiguous host tensor");
            assert_eq!(got.len(), numel);
            for i in 0..numel {
                assert_eq!(got[i].to_bits(), data[i].to_bits(), "numel={numel} i={i}");
            }
            // 非同期 worker の補充完了を待つ（wait_timeout 20 ms より長く取る）。
            sleep(Duration::from_millis(50));
        }
        let (h1, _, _, cached) = readback_prefetch_stats_diag();
        max_cached = max_cached.max(cached);
        let hits = h1 - h0;
        if default_arm && eligible {
            assert!(
                hits >= 1,
                "既定腕の対象 numel はプール命中が出る (numel={numel})"
            );
        } else {
            assert_eq!(
                hits, 0,
                "対象外 numel／opt-out 腕はプールを使わない (numel={numel})"
            );
        }
        println!("numel={numel} eligible={eligible} hits_delta={hits} cached_bytes={cached}");
    }
    if default_arm {
        assert!(max_cached > 0, "既定腕では解放前に cached_bytes > 0");
    }

    // 本番の REQ-14 解放経路で保持が 0 に戻る。
    CudaBackendOps::new(device.ordinal())
        .release_cached_device_memory()
        .expect("release_cached_device_memory");
    let (_, _, _, after) = readback_prefetch_stats_diag();
    println!("max_cached_bytes={max_cached} cached_bytes_after_release={after}");
    assert_eq!(after, 0);
}
