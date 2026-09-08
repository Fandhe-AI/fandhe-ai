//! `crate::memory::readback` の宛先確保方式（イシュー #1437。
//! `ReadbackDest::Fresh`／`PretouchedFresh`）が bit 完全一致の出力を
//! 返すことを実機（DGX Spark GB10 等）で確認する `#[ignore]` テスト。
//!
//! `docs/perf/cuda-host-view-readout-small-shape-regression.md`（#1436）
//! の診断が示した後退機構（宛先 `Vec` の未タッチ mmap ページ由来の
//! D2H 中ページフォールト）は宛先バッファの**内容**を変えない
//! （`memcpy_dtoh` は全バイトを上書きするコピーであり、事前タッチの
//! 値は D2H 完了後には残らない）。この不変条件を実機の実際の GPU
//! D2H 経路（cudarc の `cuMemcpyDtoHAsync`）で直接検証する。
//!
//! `readback_f32_diag`／`readback_f16_diag`（`internal-diagnostics`
//! feature 限定・診断専用公開入口。`crate::memory` ドキュメンテーション
//! コメント参照）を経由する。`internal-diagnostics` feature 必須
//! （`Cargo.toml` の `[[test]]` エントリ参照）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test readback_pretouched_bit_match_1437 -- --ignored --nocapture --test-threads=1
//! ```

use fandhe_ai_backend_cuda::CudaDevice;
use fandhe_ai_backend_cuda::memory::{readback_f16_diag, readback_f32_diag};
use half::f16;

/// `Fresh`（現行 `clone_dtoh`）と `PretouchedFresh`（#1437 是正候補）が
/// 同一デバイスバッファに対して byte 単位で完全一致することを、複数の
/// 形状・複数の入力パターンで確認する（f32 経路。`gemm.rs` 等大半の
/// GEMM 出力読み出しが通る型）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn readback_fresh_and_pretouched_fresh_are_bit_identical_f32() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let stream = device.stream();

    // 0 要素・小要素数・大要素数（数十 MiB。glibc mmap 閾値付近）を
    // 横断し、`pretouched_host_vec` の境界条件（numel==0 を含む）を
    // 実機の `memcpy_dtoh` 経路でも確認する。
    for &numel in &[0usize, 1, 37, 4096, 1_048_576, 16 * 1024 * 1024] {
        // 決定的な非対称パターン（全要素同値だと「宛先が上書きされず
        // 事前タッチ値のまま」という不具合を誤って見逃しうるため、
        // インデックス依存の値にする）。
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.125 - 12345.0).collect();

        let dev_buf = stream.clone_htod(&data).expect("H2D upload must succeed");
        stream
            .synchronize()
            .expect("post-H2D synchronize must succeed");

        let fresh =
            readback_f32_diag(stream, &dev_buf, false).expect("Fresh readback must succeed");
        let pretouched = readback_f32_diag(stream, &dev_buf, true)
            .expect("PretouchedFresh readback must succeed");

        assert_eq!(
            fresh.len(),
            numel,
            "Fresh readback length mismatch (numel={numel})"
        );
        assert_eq!(
            pretouched.len(),
            numel,
            "PretouchedFresh readback length mismatch (numel={numel})"
        );
        for i in 0..numel {
            assert_eq!(
                fresh[i].to_bits(),
                pretouched[i].to_bits(),
                "byte-level mismatch between Fresh and PretouchedFresh at index {i} (numel={numel})"
            );
            assert_eq!(
                fresh[i].to_bits(),
                data[i].to_bits(),
                "Fresh readback must round-trip the uploaded value exactly at index {i} \
                 (numel={numel})"
            );
        }
    }
}

/// f16 版（`gemm_mma.rs::download_f16` が経由する `T = f16` 経路。#1191
/// 本番結線済みの MMA f16 GEMM がこの型で `readback` を呼ぶため、
/// `ReadbackSentinel` の型ごとの impl 漏れがないことを実機で保証する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn readback_fresh_and_pretouched_fresh_are_bit_identical_f16() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let stream = device.stream();

    for &numel in &[0usize, 1, 37, 4096, 65536] {
        let data: Vec<f16> = (0..numel)
            .map(|i| f16::from_f32((i as f32) * 0.03125 - 512.0))
            .collect();

        let dev_buf = stream.clone_htod(&data).expect("H2D upload must succeed");
        stream
            .synchronize()
            .expect("post-H2D synchronize must succeed");

        let fresh =
            readback_f16_diag(stream, &dev_buf, false).expect("Fresh readback must succeed");
        let pretouched = readback_f16_diag(stream, &dev_buf, true)
            .expect("PretouchedFresh readback must succeed");

        assert_eq!(fresh.len(), numel);
        assert_eq!(pretouched.len(), numel);
        for i in 0..numel {
            assert_eq!(
                fresh[i].to_bits(),
                pretouched[i].to_bits(),
                "byte-level mismatch between Fresh and PretouchedFresh at index {i} (numel={numel})"
            );
            assert_eq!(fresh[i].to_bits(), data[i].to_bits());
        }
    }
}
