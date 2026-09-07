//! `MemoryOps::with_host_view`（[`crate::host_staging`]。イシュー #1336）の
//! D2H＋ホスト読み出し時間 before/after を GB10 実機で計測する A/B
//! ハーネス（**計測専用・本番非変更**。`large_buffer_percall_alloc_
//! ab_1149.rs` と同じ位置づけ）。
//!
//! ## 3 系列（同一プロセス内・固定順序）
//!
//! - `before`: `mem.download(&buf)`（毎回新規 `Vec<f32>` 確保。#1336 導入
//!   前の既定実装相当）→ 全要素読み出し
//! - `after_pageable`: 本番 `CudaMemory::new`（`HOST_STAGING_KIND` 既定
//!   ＝ `Pageable`）の `with_host_view` でクロージャ内に同じ全要素読み出し
//! - `after_pinned`: `CudaMemory::new_with_host_staging_kind(Pinned)`
//!   （イシュー #1336 実機実測フェーズで追加。`with_host_view` 経由・
//!   本番と同じキャッシュ hit 条件で `Pageable` と公平に比較できる。
//!   `memory.rs::CudaMemory::new_with_host_staging_kind` ドキュメンテー
//!   ションコメント参照）
//!
//! 各系列は `bench_harness::MeasurementConfig::new(20, 20)`（TASK-8.1
//! 下限）＋ cold 1 回目を個別記録する（`large_buffer_percall_alloc_
//! transfer_triage.rs::measure_with_cold` と同型）。3 系列とも読み出し
//! 結果の畳み込み値（`to_bits()` の XOR 畳み込み）が一致することを
//! `assert_eq!` で確認し、bit 同一の副次検証を兼ねる。
//!
//! ## 実機前提・実行コマンド
//!
//! 通常 CI（GitHub ホステッド・CUDA 実機なし）では実行されない
//! `#[ignore]` 分離テスト。`internal-diagnostics` feature 必須
//! （`new_with_host_staging_kind`／`host_staging_stats` が同 feature 限定。
//! `Cargo.toml` の `[[test]]` エントリ参照）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test host_view_staging_readout_ab_1336 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release` 必須（他 A/B ハーネスと同じく最適化なしのビルドは
//! 相対比較の意味を失わせる）。`--test-threads=1` 必須（`host_staging`
//! キャッシュはプロセス内の `CudaMemory` インスタンスごとに独立する
//! ため直接の競合はないが、driver ストリームの計測ノイズを避けるため
//! 他実機テストと同じ直列化方針を踏襲する）。

use std::time::Instant;

use bench_harness::MeasurementConfig;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaMemory, HostStagingKind};
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::MemoryOps;

/// 計測対象の正方形状（N×N f32）。イシュー本文・計画の N=1024/2048/4096
/// （4/16/64 MiB）。64 MiB は `HostStagingCache` の既定上限
/// （`host_staging::HOST_STAGING_CAP_BYTES` = 256 MiB）内に収まる。
const SIZES_N: [u64; 3] = [1024, 2048, 4096];

/// `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md` が
/// 特定した glibc mmap 閾値（32 MiB）。N=4096（64 MiB）のみこれを超える
/// ため、`after_pageable` の改善が期待できるのは当該形状のみである
/// （N=1024/2048 は差なしでも失敗と読まない。呼び出し元 doc §5 のゲート
/// B 参照）。
const MMAP_THRESHOLD_MIB: u64 = 32;

fn bytes_mib(n: u64) -> f64 {
    (n * n * std::mem::size_of::<f32>() as u64) as f64 / (1024.0 * 1024.0)
}

/// 決定的シード（xorshift64*）で N×N の f32 データを生成する（`.claude/
/// rules/coding-rust.md` の学習系回帰テスト向け決定的シード方針を、
/// 本ハーネスの入力生成にも同様に適用する）。
fn deterministic_data(n: u64, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    let numel = (n * n) as usize;
    rng.fill_vec(numel)
}

/// 読み出し結果の畳み込み値（`to_bits()` の XOR 畳み込み）。3 系列の
/// bit 同一性を安価に確認するための副次チェックであり、
/// `host_view_real_device.rs` の厳密な `assert_eq!(Vec<u32>, ..)` 比較
/// （実機 `#[ignore]` テスト）を代替するものではない。
fn fold_bits(slice: &[f32]) -> u32 {
    slice.iter().fold(0u32, |acc, v| acc ^ v.to_bits())
}

/// [`bench_harness::Measurement`] から本テストが出力する要約統計。
struct Summary {
    min_ms: f64,
    q1_ms: f64,
    median_ms: f64,
    q3_ms: f64,
    max_ms: f64,
}

fn summarize(measurement: &bench_harness::Measurement) -> Summary {
    let min_secs = measurement
        .samples_secs
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_secs = measurement
        .samples_secs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    Summary {
        min_ms: min_secs * 1000.0,
        q1_ms: measurement.q1_secs * 1000.0,
        median_ms: measurement.median_secs * 1000.0,
        q3_ms: measurement.q3_secs * 1000.0,
        max_ms: max_secs * 1000.0,
    }
}

/// `workload` を cold 1 回計測してから `bench_harness::run`（20/20 下限）
/// で計測する（`large_buffer_percall_alloc_transfer_triage.rs::
/// measure_with_cold` と同型。畳み込み値も cold 実行から取得して返す）。
fn measure_with_cold<F: FnMut() -> u32>(
    config: &MeasurementConfig,
    mut workload: F,
) -> (f64, u32, bench_harness::Measurement) {
    let cold_start = Instant::now();
    let cold_folded = workload();
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    let measurement = bench_harness::run(config, || {
        let _ = workload();
    })
    .expect("phase measurement must satisfy TASK-8.1 protocol");
    (cold_ms, cold_folded, measurement)
}

fn print_row(phase: &str, n: u64, cold_ms: f64, s: &Summary) {
    println!(
        "{phase},{n},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
        bytes_mib(n),
        cold_ms,
        s.min_ms,
        s.q1_ms,
        s.median_ms,
        s.q3_ms,
        s.max_ms,
    );
}

/// N=1024/2048/4096 の 3 形状で `before`／`after_pageable`／`after_pinned`
/// を計測し、CSV 風ログを stdout へ出力する。畳み込み値が 3 系列とも
/// 一致することを都度確認する（bit 同一の副次検証）。
///
/// `docs/perf/cuda-host-view-staging-readout.md` §5 のゲート A〜C・
/// `docs/perf/logs/cuda-host-view-staging-1336/aggregate.py` が本テスト
/// の 5 プロセス起動出力を集計する前提の CSV ヘッダ:
/// `phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms`
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn host_view_staging_readout_ab() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem_pageable = CudaMemory::new(&device);
    let mem_pinned = CudaMemory::new_with_host_staging_kind(&device, HostStagingKind::Pinned);
    let config = MeasurementConfig::new(20, 20).expect("20/20 は TASK-8.1 下限を満たす");

    println!("phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms");

    for &n in &SIZES_N {
        let data = deterministic_data(n, 0x1336_0000_0000_0000 ^ n);
        let tensor = Tensor::<f32>::new(data, &[n as usize, n as usize])
            .expect("Tensor::new must succeed for deterministic data");
        // `before`／`after_*` いずれも同じアップロード元バッファを使い回す
        // （アップロード自体は計測対象外。D2H＋読み出しのみを比較する）。
        let buf_before = mem_pageable
            .upload(&tensor)
            .expect("upload must succeed on real hardware");
        let buf_pinned = mem_pinned
            .upload(&tensor)
            .expect("upload must succeed on real hardware (pinned-kind CudaMemory)");

        // `before`: `download()` が毎回新規 `Vec<f32>` を確保する経路
        // （#1336 導入前の既定実装相当）。
        let (cold_ms, cold_folded, m) = measure_with_cold(&config, || {
            let downloaded = mem_pageable
                .download(&buf_before)
                .expect("download must succeed on real hardware");
            fold_bits(
                downloaded
                    .as_slice()
                    .expect("download returns a contiguous tensor"),
            )
        });
        let before_folded = cold_folded;
        print_row("before", n, cold_ms, &summarize(&m));

        // `after_pageable`: 本番既定経路（`HOST_STAGING_KIND` = Pageable）。
        let (cold_ms, cold_folded, m) = measure_with_cold(&config, || {
            let mut folded = 0u32;
            mem_pageable
                .with_host_view(&buf_before, &mut |slice| folded = fold_bits(slice))
                .expect("with_host_view must succeed on real hardware");
            folded
        });
        assert_eq!(
            cold_folded, before_folded,
            "after_pageable の読み出し結果は before と bit 同一のはず（N={n}）"
        );
        print_row("after_pageable", n, cold_ms, &summarize(&m));

        // `after_pinned`: キャッシュ経由 Pinned（`new_with_host_staging_
        // kind` 経由。`with_host_view_using_kind` は毎回新規確保のため
        // ここでは使わない。モジュール冒頭コメント参照）。
        let (cold_ms, cold_folded, m) = measure_with_cold(&config, || {
            let mut folded = 0u32;
            mem_pinned
                .with_host_view(&buf_pinned, &mut |slice| folded = fold_bits(slice))
                .expect("with_host_view (pinned-kind CudaMemory) must succeed on real hardware");
            folded
        });
        assert_eq!(
            cold_folded, before_folded,
            "after_pinned の読み出し結果は before と bit 同一のはず（N={n}）"
        );
        print_row("after_pinned", n, cold_ms, &summarize(&m));

        if n * n * std::mem::size_of::<f32>() as u64 <= MMAP_THRESHOLD_MIB * 1024 * 1024 {
            eprintln!(
                "note: N={n}（{:.1} MiB）は glibc mmap 閾値（{MMAP_THRESHOLD_MIB} MiB）未満のため、\
                 after 系列の改善が観測されなくても病態不成立とは判断しない（doc §5 ゲート B 参照）",
                bytes_mib(n)
            );
        }
    }

    // `release_host_staging` で明示解放し、他テストへ影響を残さない
    // （`release_host_staging_clears_cache_and_frees_reported_bytes` と
    // 同じ後始末方針）。
    let freed_pageable = mem_pageable.release_host_staging();
    let freed_pinned = mem_pinned.release_host_staging();
    println!("release_host_staging,pageable,{freed_pageable}");
    println!("release_host_staging,pinned,{freed_pinned}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU 非依存の単体テスト: `bytes_mib` の換算が正しいこと
    /// （N=4096 の f32 正方行列は 64 MiB のはず）。
    #[test]
    fn bytes_mib_converts_n_to_expected_mebibytes() {
        assert!((bytes_mib(1024) - 4.0).abs() < 1e-9);
        assert!((bytes_mib(2048) - 16.0).abs() < 1e-9);
        assert!((bytes_mib(4096) - 64.0).abs() < 1e-9);
    }

    /// `fold_bits` は同一内容のスライスに対し決定的に同じ値を返す
    /// （XOR 畳み込みの基本性質。実機なしで検証可能な範囲）。
    #[test]
    fn fold_bits_is_deterministic_for_same_input() {
        let data = vec![1.0f32, -2.5, f32::NAN, 0.0];
        assert_eq!(fold_bits(&data), fold_bits(&data));
    }

    /// `deterministic_data` は同一シードで同一系列を返す（決定的シード
    /// 方針の自己検証。`.claude/rules/coding-rust.md`）。
    #[test]
    fn deterministic_data_is_reproducible_for_same_seed() {
        let a = deterministic_data(8, 42);
        let b = deterministic_data(8, 42);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
