//! H2D pinned staging（[`fandhe_ai_backend_cuda::set_pinned_h2d_enabled`]。
//! イシュー #1585・`docs/perf/cuda-h2d-pinned-staging.md` §3「Layer B
//! （改善根拠）」）の H2D 単体マイクロ A/B ハーネス（**計測専用・本番
//! 非変更**。`host_view_staging_readout_ab_1336.rs`〈D2H 側の対称〉と
//! 同じ位置づけ）。
//!
//! ## 2 腕（同一プロセス内・N ごとに固定順序）
//!
//! - `pageable`: [`set_pinned_h2d_enabled`]`(false)`。`MemoryOps::upload`
//!   が `host_staging::upload_new` の早期分岐経由で `stream.clone_htod`
//!   をそのまま呼ぶ経路（導入前と経路・出力とも bit 同一）。
//! - `pinned_staged`: [`set_pinned_h2d_enabled`]`(true)`。同じ
//!   `MemoryOps::upload` が `H2dStagingCache` 経由の pinned ステージング
//!   バッファ（`PinnedHostSlice`）へ同期コピーしてから `clone_htod` へ
//!   渡す経路（`crate::host_staging` モジュール「H2D 用ステージング」
//!   節）。
//!
//! いずれも **「upload 発行 → `stream.synchronize()`」を 1 反復**として
//! 計測する（`docs/perf/cuda-h2d-pinned-staging.md` §3 の規則「H2D 単体
//! （発行＋`synchronize`）」）。`MemoryOps::upload` 自体はホストを
//! ブロックしない非同期投入契約（`docs/backend-cuda-async-execution-
//! design.md` §4 の表「`MemoryOps::upload`」行）のため、投入だけでは
//! デバイス側の完了を計測できない。本ハーネスは
//! `CudaDevice::stream()`（`internal-diagnostics` feature 限定の診断
//! 専用アクセサ）経由で明示的に `synchronize()` を呼び、投入から完了
//! までを 1 反復の所要時間とする。
//!
//! ウォームアップは [`bench_harness::run`] 自身が内部で行う
//! （`config.warmup` 回。呼び出しが返った時点で計測対象処理が完了して
//! いることを前提とするプロトコル）。ただし `pinned_staged` 腕の
//! **最初の 1 回**（`H2dStagingCache` の miss・新規 `cuMemHostAlloc`）は
//! `bench_harness::run` 自身の内部ウォームアップよりも前に個別の
//! cold 計測として切り出す（`host_view_staging_readout_ab_1336.rs::
//! measure_with_cold` と同型）。これにより `bench_harness::run` が実行
//! する warmup 回・計測回はすべて同一 numel の 2 回目以降の呼び出しと
//! なり、`H2dStagingCache` の再利用（hit）経路に確実に乗る。
//!
//! 各腕・各 N で読み戻し結果（`mem.upload` → `mem.download`）が
//! bit 同一であることを畳み込み値（`to_bits()` の XOR 畳み込み）で
//! 確認する（`fold_bits`。`host_view_staging_readout_ab_1336.rs` と同じ
//! 副次検証で、厳密な要素ごと比較は `pinned_h2d_real_device.rs` が担う）。
//! `pinned_staged` 腕では `H2dStagingCache` の統計（`h2d_staging_stats`。
//! `internal-diagnostics` feature 限定）から、計測区間中に hit が
//! 発生していること（キャッシュ再利用経路に実際に乗ったこと）も検証
//! する。
//!
//! ## 実機前提・実行コマンド
//!
//! 通常 CI（GitHub ホステッド・CUDA 実機なし）では実行されない
//! `#[ignore]` 分離テスト。`internal-diagnostics` feature 必須
//! （`CudaDevice::stream`／`CudaMemory::h2d_staging_stats` が同 feature
//! 限定。`Cargo.toml` の `[[test]]` エントリ参照）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics \
//!     --test pinned_h2d_upload_ab_1585 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--release` 必須（他 A/B ハーネスと同じく最適化なしのビルドは
//! 相対比較の意味を失わせる）。`--test-threads=1` 必須（`set_pinned_
//! h2d_enabled` はプロセスグローバル `AtomicBool` のため、`cargo test`
//! の既定並列実行下では他テストと相互干渉しうる。`FlagGuard` が
//! テスト内での直列化・原状復帰を担うが、プロセス内の他バイナリ実行
//! （同時に走る別 `#[ignore]` テスト）とは独立しないため直列実行を
//! 前提とする）。

use std::time::Instant;

use bench_harness::MeasurementConfig;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaMemory, pinned_h2d_enabled, set_pinned_h2d_enabled};
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::MemoryOps;

/// 計測対象の正方形状（N×N f32）。イシュー本文・`docs/perf/cuda-h2d-
/// pinned-staging.md` §3 が指定する N=1024/2048/4096（4/16/64 MiB）。
const SIZES_N: [u64; 3] = [1024, 2048, 4096];

/// `PINNED_H2D_ENABLED` はプロセスグローバル（`AtomicBool`）のため、
/// `cargo test` の既定並列実行下での相互干渉を避けて直列化・原状復帰
/// する RAII ガード（`crates/backend-cuda/tests/pinned_h2d_real_device.rs::
/// FlagGuard`・`host_view_real_device.rs::PlacementFlagGuard` と同型。
/// 統合テストはクレート内 `#[cfg(test)]` 限定の `pinned_h2d_test_support`
/// へ到達できないため、本ファイル専用のロックを持つ）。
static PINNED_H2D_AB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct FlagGuard {
    original: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl FlagGuard {
    fn acquire(enabled: bool) -> Self {
        let lock = PINNED_H2D_AB_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let original = pinned_h2d_enabled();
        set_pinned_h2d_enabled(enabled);
        Self {
            original,
            _lock: lock,
        }
    }
}

impl Drop for FlagGuard {
    fn drop(&mut self) {
        set_pinned_h2d_enabled(self.original);
    }
}

fn bytes_mib(n: u64) -> f64 {
    (n * n * std::mem::size_of::<f32>() as u64) as f64 / (1024.0 * 1024.0)
}

/// 決定的シード（xorshift64*）で N×N の f32 データを生成する（`.claude/
/// rules/coding-rust.md` の学習系回帰テスト向け決定的シード方針を、
/// 本ハーネスの入力生成にも同様に適用する。`host_view_staging_readout_
/// ab_1336.rs::deterministic_data` と同型）。
fn deterministic_data(n: u64, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    let numel = (n * n) as usize;
    rng.fill_vec(numel)
}

/// 読み戻し結果の畳み込み値（`to_bits()` の XOR 畳み込み）。2 腕の
/// bit 同一性を安価に確認するための副次チェック（`pinned_h2d_real_
/// device.rs` の厳密な要素ごと比較を代替するものではない）。
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

/// 「upload 発行 → `stream.synchronize()`」1 反復を計測する。`mem` は
/// 呼び出し元が腕ごとに構築済みのものを使う（`H2dStagingCache` は
/// `CudaMemory` インスタンスごとに独立するため、腕をまたいだキャッシュ
/// 汚染は生じない）。
///
/// cold 計測（`bench_harness::run` の内部ウォームアップより前の 1 回）
/// と、`bench_harness::run`（`config.warmup` 回のウォームアップ→
/// `config.iters` 回の計測）を分けて返す（`host_view_staging_readout_
/// ab_1336.rs::measure_with_cold` と同型）。
fn measure_with_cold(
    device: &CudaDevice,
    mem: &CudaMemory,
    tensor: &Tensor<f32>,
    config: &MeasurementConfig,
) -> (f64, bench_harness::Measurement) {
    let cold_start = Instant::now();
    let cold_buf = mem
        .upload(tensor)
        .expect("upload must succeed on real hardware");
    device
        .stream()
        .synchronize()
        .expect("stream.synchronize() must succeed after upload on real hardware");
    let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
    // `cold_buf` はスコープ終了時に drop（デバイスメモリ解放）される。
    // `black_box` で保護し、cold 計測自体が最適化で消えないようにする
    // （`bench_harness::run` 側と同じ保護方針。モジュール冒頭コメント
    // 「`bench_harness::run` は `workload()` の呼び出しを `black_box` で
    // 包む」を参照。cold 計測は `run` の外側で手動計測するためここでも
    // 明示的に保護する）。
    std::hint::black_box(&cold_buf);

    let measurement = bench_harness::run(config, || {
        let buf = mem
            .upload(tensor)
            .expect("upload must succeed on real hardware");
        device
            .stream()
            .synchronize()
            .expect("stream.synchronize() must succeed after upload on real hardware");
        // 戻り値を捨てると release 最適化で「発行＋synchronize」全体が
        // 除去されうる（`host_view_staging_readout_ab_1336.rs` codex-review
        // P1・Cursor Bugbot 指摘の再発防止。`bench_harness::run` 自身は
        // クロージャの呼び出しのみを `black_box` で保護し、クロージャ
        // 内部の計算過程までは保護しない契約——`protocol.rs::run` の
        // ドキュメンテーションコメント参照）。呼び出し側の責務として
        // `buf` を `black_box` に渡し、計測対象（デバイス確保・H2D・
        // synchronize）を保護する。`buf` はクロージャのスコープ終了時に
        // drop（デバイスメモリ解放）される。次反復の `upload` は新規
        // デバイス確保を伴うため「発行（alloc+H2D）＋synchronize」という
        // 規則どおり計測対象に含める。
        std::hint::black_box(buf);
    })
    .expect("phase measurement must satisfy TASK-8.1 protocol");
    (cold_ms, measurement)
}

/// N=1024/2048/4096 の 3 形状で `pageable`／`pinned_staged` を計測し、
/// CSV 風ログを stdout へ出力する。
///
/// `docs/perf/cuda-h2d-pinned-staging.md` §3・
/// `docs/perf/logs/cuda-h2d-pinned-staging-1585/aggregate.py` が本テスト
/// の 5 プロセス起動出力を集計する前提の CSV ヘッダ:
/// `phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms`
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn pinned_h2d_upload_ab() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let config = MeasurementConfig::new(20, 20).expect("20/20 は TASK-8.1 下限を満たす");

    println!("phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms");

    for &n in &SIZES_N {
        let data = deterministic_data(n, 0x1585_0000_0000_0000 ^ n);
        let tensor = Tensor::<f32>::new(data.clone(), &[n as usize, n as usize])
            .expect("Tensor::new must succeed for deterministic data");

        // `pageable` 腕: フラグ OFF（`clone_htod` 直呼び経路）。
        let pageable_folded;
        {
            let _guard = FlagGuard::acquire(false);
            let mem = CudaMemory::new(&device);
            let (cold_ms, m) = measure_with_cold(&device, &mem, &tensor, &config);
            print_row("pageable", n, cold_ms, &summarize(&m));

            // bit 同一検証用の読み戻し（計測対象外の 1 回）。
            let buf = mem
                .upload(&tensor)
                .expect("verification upload must succeed on real hardware");
            let back = mem
                .download(&buf)
                .expect("verification download must succeed on real hardware");
            pageable_folded = fold_bits(
                back.as_slice()
                    .expect("download returns a contiguous tensor"),
            );
        }

        // `pinned_staged` 腕: フラグ ON（`H2dStagingCache` 経由）。
        let pinned_folded;
        let hits_before;
        let hits_after;
        {
            let _guard = FlagGuard::acquire(true);
            let mem = CudaMemory::new(&device);
            hits_before = mem.h2d_staging_stats().hits;
            let (cold_ms, m) = measure_with_cold(&device, &mem, &tensor, &config);
            hits_after = mem.h2d_staging_stats().hits;
            print_row("pinned_staged", n, cold_ms, &summarize(&m));

            // bit 同一検証用の読み戻し（計測対象外の 1 回）。
            let buf = mem
                .upload(&tensor)
                .expect("verification upload must succeed on real hardware");
            let back = mem
                .download(&buf)
                .expect("verification download must succeed on real hardware");
            pinned_folded = fold_bits(
                back.as_slice()
                    .expect("download returns a contiguous tensor"),
            );
        }

        assert_eq!(
            pinned_folded, pageable_folded,
            "N={n}: pinned_staged と pageable の読み戻し結果は bit 同一のはず"
        );

        // cold（miss・1 回）を除く `config.warmup + config.iters` 回は
        // 同一 numel の 2 回目以降の呼び出しであり、`H2dStagingCache`
        // の再利用（hit）経路に乗るはず（モジュール冒頭コメント参照）。
        let expected_min_hits = (config.warmup + config.iters) as u64;
        let observed_hits = hits_after.saturating_sub(hits_before);
        assert!(
            observed_hits >= expected_min_hits,
            "N={n}: H2dStagingCache は計測区間中に再利用（hit）されるはず \
             (observed={observed_hits}, expected_min={expected_min_hits}, \
             hits_before={hits_before}, hits_after={hits_after})"
        );
    }
}
