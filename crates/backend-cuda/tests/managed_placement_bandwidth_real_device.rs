//! イシュー #1353: CUDA managed memory 配置（#1352。`crate::placement`）
//! 有無での「CPU 側ページ経由アクセスの帯域」実機計測。
//!
//! `#1352` の性能実測は #1353 が引き継ぐ契約（`docs/backend-cuda-managed-
//! placement-decision.md`「実機実測」節）のうち、本ファイルは受入基準
//! 「CPU 側ページ経由アクセスの帯域低下の有無を記録する」を担う。
//! GB10（ホスト・GPU 物理統合メモリ）では managed 配置はホスト側 memcpy
//! （`cuMemAllocManaged` 経由。`crate::memory::UnifiedSlice`）で
//! upload・`host_readback`（synchronize 後の `as_slice`）で download する
//! 一方、device-only 配置は通常の H2D/D2H DMA 転送を使う
//! （`crate::memory` モジュール冒頭コメント「配置（managed 拡張）」）。
//! 本ファイルはこの 2 経路を同一サイズスイープで比較し GB/s を出力する。
//!
//! サイズは #1146/#1149 が確認した 32→33 MiB 段差（glibc mmap しきい値・
//! `cuMemPool` release threshold）を含む
//! {4, 16, 32, 33, 64} MiB（f32 要素数へ換算）を使う。
//!
//! 計測する 3 区間（各 5 run 中央値）:
//! 1. upload（device-only: H2D DMA／managed: ホスト → `UnifiedSlice` memcpy）
//! 2. download（device-only: D2H DMA／managed: `synchronize` + `as_slice`
//!    による host_readback。managed は `to_vec()` でホスト `Vec` へ
//!    コピーする分、本区間自体は managed ページを直接読む区間ではない）
//! 3. ページ経由アクセスの実効帯域（`Tensor` の要素を単純合計して読み
//!    飛ばしを防ぐ）。**device-only は区間 2 が返した通常ホスト `Tensor`
//!    を読むが、managed は区間 2 とは別に `CudaMemory::
//!    measure_managed_direct_read_seconds`（`internal-diagnostics`
//!    feature 限定の診断専用入口）でアップロード直後の `UnifiedSlice` を
//!    コピーせず直接読む**（codex-review 指摘の是正: 区間 2 の
//!    `host_readback` が既にホスト `Vec` へコピー済みのため、そこから
//!    読むだけでは managed ページへの CPU 直接アクセス帯域を計測でき
//!    ない。`crate::memory::CudaMemory::measure_managed_direct_read_
//!    seconds` ドキュメンテーションコメント参照）
//!
//! 出力に先立ち各サイズで upload → download の bit 一致（`to_bits`）を
//! 検証する（性能値のみを出力せず、数値契約〈配置に依らず bit 同一〉を
//! 本ファイル内でも裏取りする）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test managed_placement_bandwidth_real_device -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::placement::{managed_placement_enabled, set_managed_placement_enabled};
use fandhe_ai_backend_cuda::{CudaDevice, CudaMemory};
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::buffer::MemoryOps;
use std::time::Instant;

/// フラグはプロセスグローバル（`crate::placement`）のため、`cargo test`
/// の既定並列実行下での相互干渉を避けて直列化・原状復帰する RAII ガード
/// （`managed_placement_real_device.rs::PlacementFlagGuard` と同型。本
/// ファイル専用のロックを持つ理由も同じ: 統合テストは
/// `crate::placement::test_support`〈`pub(crate)` 限定〉へ到達できない）。
static PLACEMENT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct PlacementFlagGuard {
    original: bool,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl PlacementFlagGuard {
    fn acquire(enabled: bool) -> Self {
        let lock = PLACEMENT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let original = managed_placement_enabled();
        set_managed_placement_enabled(enabled);
        Self {
            original,
            _lock: lock,
        }
    }
}

impl Drop for PlacementFlagGuard {
    fn drop(&mut self) {
        set_managed_placement_enabled(self.original);
    }
}

const RUNS: usize = 5;

/// MiB を f32 要素数へ換算する（4 バイト/要素）。
fn mib_to_elements(mib: usize) -> usize {
    (mib * 1024 * 1024) / std::mem::size_of::<f32>()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("no NaN durations"));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

fn gib_per_s(bytes: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return f64::INFINITY;
    }
    (bytes as f64) / seconds / (1024.0 * 1024.0 * 1024.0)
}

fn assert_bit_exact(actual: &Tensor<f32>, expected: &Tensor<f32>, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{ctx}: shape mismatch");
    let a = actual.contiguous();
    let e = expected.contiguous();
    let a_data = a.as_slice().expect("contiguous tensor exposes a slice");
    let e_data = e.as_slice().expect("contiguous tensor exposes a slice");
    assert_eq!(a_data.len(), e_data.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a_data.iter().zip(e_data.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} not bit-exact (actual={x} expected={y})"
        );
    }
}

/// 1 サイズ分の upload/download/host-readback を `RUNS` 回計測し中央値
/// GB/s を返す。呼び出し前に `PlacementFlagGuard` で配置を確定しておく
/// こと（本関数はフラグを変更しない）。`managed` は現在の配置が managed
/// かどうかを呼び出し元から明示的に渡す（フラグは
/// `crate::placement::managed_placement_enabled()` で読めるが、本関数は
/// 純粋にバッファの実配置に基づいて読み取り経路を切り替えたいため引数化
/// する）。
///
/// 読み取り経路（区間 3・codex-review 指摘の是正）: managed 配置では
/// `download()` が既にホスト `Vec` へコピー済みの `Tensor` を返すため、
/// そこから読んでも managed ページへの CPU 直接アクセス帯域は測れない。
/// managed の場合は upload 直後の `buf`（`UnifiedSlice` を保持したまま）
/// を `CudaMemory::measure_managed_direct_read_seconds`
/// （`internal-diagnostics` feature 限定）へ渡し、コピーなしで直接読む。
/// device-only は従来どおり `download()` が返した通常ホスト `Tensor` を
/// 読む（これは実際に device-only 配置が使う唯一のホスト側読み取り経路
/// であり、コピー後の読み取りで正しい）。
fn measure_one(
    mem: &CudaMemory,
    device: &CudaDevice,
    tensor: &Tensor<f32>,
    managed: bool,
) -> (f64, f64, f64) {
    let contiguous = tensor.contiguous();
    let bytes = std::mem::size_of_val(contiguous.as_slice().unwrap());

    let mut upload_s = Vec::with_capacity(RUNS);
    let mut download_s = Vec::with_capacity(RUNS);
    let mut readback_s = Vec::with_capacity(RUNS);

    for _ in 0..RUNS {
        let t0 = Instant::now();
        let buf = mem
            .upload(tensor)
            .expect("upload must succeed on real hardware");
        if !managed {
            // device-only: `upload_inner` の `clone_htod` は `cuMemcpyHtoDAsync`
            // を発行する非同期コピー（`memory.rs::readback` ドキュメンテーション
            // コメント「唯一の同期点」参照）のため、`upload()` の呼び出し復帰は
            // 転送完了を意味しない。managed 側は同一位置でホスト → `UnifiedSlice`
            // の同期 memcpy（`upload_inner` 内 `copy_from_slice`）が完了してから
            // 復帰するため、この非対称のまま両者を同じ区間として計測すると比較
            // 範囲がずれる（未完了 H2D の残り待機時間が後続の区間 2 `download()`
            // 側へ混入する。codex-review 指摘）。ストリーム同期をここで挟み、
            // upload 完了までを upload_s に含める。
            device
                .stream()
                .synchronize()
                .expect("stream synchronize must succeed on real hardware");
        }
        upload_s.push(t0.elapsed().as_secs_f64());

        if managed {
            // managed ページ自体への CPU 直接アクセス帯域（コピーなし）。
            // `download()`（区間 2）より先に計測する: `download()` 自体は
            // 副作用として `buf` の中身を変更しないため順序は結果に影響
            // しないが、「アップロード直後の生きた `UnifiedSlice` を読む」
            // という意図を明確にするため upload の直後に置く。
            let direct_read_s = mem
                .measure_managed_direct_read_seconds(&buf)
                .expect("managed direct read must succeed on real hardware");
            readback_s.push(direct_read_s);
        }

        let t1 = Instant::now();
        let back = mem
            .download(&buf)
            .expect("download must succeed on real hardware");
        download_s.push(t1.elapsed().as_secs_f64());

        assert_bit_exact(&back, tensor, "upload/download roundtrip");

        if !managed {
            // device-only: `download()` が返した通常ホスト `Tensor` を
            // 逐次読んで合計する（コンパイラによる読み飛ばし最適化を
            // 防ぐため合計を `std::hint::black_box` へ渡す）。これが
            // device-only 配置における唯一のホスト側読み取り経路。
            let t2 = Instant::now();
            let contiguous = back.contiguous();
            let data = contiguous.as_slice().expect("contiguous tensor slice");
            let mut acc = 0.0f64;
            for &v in data {
                acc += v as f64;
            }
            std::hint::black_box(acc);
            readback_s.push(t2.elapsed().as_secs_f64());
        }
    }

    (
        gib_per_s(bytes, median(upload_s)),
        gib_per_s(bytes, median(download_s)),
        gib_per_s(bytes, median(readback_s)),
    )
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn managed_vs_device_only_bandwidth_sweep_on_real_hardware() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let mem = CudaMemory::new(&device);

    // #1146/#1149 が確認した 32→33 MiB 段差（glibc mmap しきい値・
    // cuMemPool release threshold）を挟む形状を含める。
    let sizes_mib = [4usize, 16, 32, 33, 64];

    println!(
        "{:>6} | {:>10} | {:>12} {:>12} {:>12} | {:>12} {:>12} {:>12}",
        "MiB",
        "elements",
        "off upload",
        "off download",
        "off readback",
        "on upload",
        "on download",
        "on readback",
    );
    println!("(単位: GiB/s. off=device-only 配置・on=managed 配置)");

    for &mib in &sizes_mib {
        let n = mib_to_elements(mib);
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 100.0).collect();
        let tensor = Tensor::<f32>::new(data, &[n]).expect("tensor construction");

        let (off_up, off_down, off_read) = {
            let _guard = PlacementFlagGuard::acquire(false);
            measure_one(&mem, &device, &tensor, false)
        };
        let (on_up, on_down, on_read) = {
            let _guard = PlacementFlagGuard::acquire(true);
            measure_one(&mem, &device, &tensor, true)
        };

        println!(
            "{mib:>6} | {n:>10} | {off_up:>12.3} {off_down:>12.3} {off_read:>12.3} | \
             {on_up:>12.3} {on_down:>12.3} {on_read:>12.3}"
        );

        // 「明確な低下」の一次スクリーニング（判定は docs 側で人間が
        // 最終確認する。ここでは実機実行のたびに壊滅的な後退〈半分以下〉
        // が無音で見逃されないよう、緩い閾値で警告のみ出す。テスト自体は
        // 失敗させない: 統合メモリ環境の測定ノイズは大きく、単発の
        // ばらつきでゲート失敗にすると fail-open ではなく別の意味の
        // 不安定化を招くため）。
        if on_read < off_read * 0.5 {
            eprintln!(
                "warning: {mib} MiB で managed readback 帯域が device-only の半分未満 \
                 (on={on_read:.3} GiB/s off={off_read:.3} GiB/s)"
            );
        }
    }
}
