//! `host-view-readout` cargo feature（`scripts/bench/framework-compare`
//! `bench-fandhe`。#1335／#1336／#1337）有効時の CUDA reuse N=1024/2048
//! 後退（15.04 倍・1.20 倍。`docs/perf/cuda-gemm-candle-gate-remeasurement.md`
//! §12.3／§12.4／§13）を D2H 読み出し方式別に再現・分解する診断テスト
//! （イシュー #1436・親 #1435）。
//!
//! # 背景
//!
//! `readout_var`（off 腕）は `Var::to_tensor()`（`Arc` 複製）→
//! `.contiguous().as_slice().to_vec()` で **反復ごとに新規 `Vec` を確保・
//! 反復末尾で free する**。一方 `host-view-readout` 有効時（on 腕）は
//! `Var::host_view()`（`Arc` 複製のみ）を `Deref` で借用するだけで、
//! **追加確保・free が一切無い**。いずれの腕も出力元は同一
//! （`Var::matmul` → `CudaBackendOps::gemm_fp32_strict` → `run_f32_kernel`
//! → `memory::readback` = `clone_dtoh` + `synchronize`。`gemm.rs`／
//! `memory.rs` 参照）であり、`MemoryOps::with_host_view`
//! （`HostStagingCache`・`Pinned`／`Pageable`・世代検査。#1336）は
//! この経路に到達しない（`docs/perf/cuda-host-view-staging-readout.md`
//! §7・`cuda-gemm-candle-gate-remeasurement.md` §13.3 で確認済み）。
//!
//! off/on の唯一の実質差は「`clone_dtoh` の宛先 `Vec<f32>` を反復ごとに
//! 確保・free するか、確保だけして keep-alive するか」に集約される。
//! glibc malloc は free された領域のサイズへ動的 mmap 閾値
//! （`M_MMAP_THRESHOLD`）を適応させるため、off 腕は定常状態で
//! brk ヒープ（既タッチ・再利用ページ）から確保される一方、on 腕は
//! free が無いため `clone_dtoh` の宛先が毎回 fresh mmap ページになり
//! うる（先行知見: `docs/perf/cuda-large-buffer-percall-alloc-transfer-
//! threshold.md` #1146 §4。ただし #1146 の P4/P5 は「未タッチ」と
//! 「区間内新規確保」が未分離だった）。本ファイルはこの 2 要因
//! （新規確保の有無・宛先ページの事前タッチ有無）を分離した 4 腕で
//! `d2h`／`host_read` 区間を実測し、後退フェーズを特定する。
//!
//! # 配置理由（`gemm_reuse_phase_diag_tests.rs` と同じ判断）
//!
//! `context_cache::{cached_device, cached_gemm, cached_allocator}`・
//! `CudaGemm::launch_tiled_f32_pooled`（いずれも `pub(crate)`）へ
//! 到達するため、integration test ではなく `lib.rs` の兄弟モジュール
//! として配置する。本ファイルはあくまで**診断専用**であり、
//! `memory::readback`／`gemm.rs`／`host_staging.rs` 等の本番経路は
//! 一切変更しない。
//!
//! # 4 腕の定義
//!
//! - `LegacyToVec`（off 腕の再現）: `clone_dtoh` で 1 本目を受け取り、
//!   `to_vec()` で 2 本目を確保して読み出し、2 本目を drop する。1 本目は
//!   `keep_alive` で保持する（`gemm_reuse_phase_diag_tests.rs` の
//!   `host_copy` 区間と同じ構成）
//! - `BorrowedKeepAlive`（on 腕の再現）: `clone_dtoh` が返す `Vec` を
//!   直接読み出し、追加確保・free なしで `keep_alive` へ積む
//! - `BorrowedWithDummyAllocFree`: `BorrowedKeepAlive` に加え、読み出し
//!   後に同サイズ `Vec` を確保・全ページへ書き込み・即 drop する
//!   （「2 本目のコピー」と「アロケータへの free 副作用」を分離する
//!   対照腕）
//! - `PretouchedReusedDest`: 計測外で 1 回だけ確保し全要素を明示
//!   書き込み（事前タッチ）した宛先 `Vec` へ `stream.memcpy_dtoh` し、
//!   読み出し後にその内容を `keep_alive` 用の別 `Vec` へ `copy_from_slice`
//!   する（reuse tape がノード storage として D2H 結果を所有する契約を
//!   模した copy-out込み）。#1437 の是正候補 A（`readback` の事前タッチ
//!   済みステージング化）を先取りする対照腕
//!
//! # gating しない方針（`gemm_reuse_phase_diag_tests.rs` と同じ理由）
//!
//! 本ファイルの `#[test]` は実行が成功すること（各腕が例外なく完了し
//! 有限・非ゼロの出力を返すこと）のみを検証条件とし、腕間の大小関係・
//! 絶対値への `assert!` は行わない（環境揺らぎによる flaky 化防止）。
//! 数値は `println!` に残し `docs/perf/cuda-host-view-readout-small-
//! shape-regression.md` へ転記する一次情報とする。腕間の checksum
//! 一致（同一入力・同一カーネルなら同一出力になるはずという sanity）
//! のみ `assert!` する。
//!
//! # 実行時は必ず `--test-threads=1`（同一 GPU 上の競合を避ける。
//! `gemm_reuse_phase_diag_tests.rs` と同じ理由）
//!
//! # プロセス分離実行（#1442 レビュー指摘対応）
//!
//! `*_all_arms`／`*_n1024`／`*_n2048`／`*_n4096` は 4 腕を同一プロセス
//! 内で順に実行するため、glibc の動的 mmap 閾値適応・`keep_alive` の
//! 一括解放が後続の腕へアロケータ状態として引き継がれうる。腕間の
//! 独立性を保証した比較が必要な場合はファイル末尾の単一腕専用テスト
//! （`readout_regression_diag_n{1024,2048,4096}_{legacy_to_vec,
//! borrowed_keep_alive,borrowed_with_dummy_alloc_free,
//! pretouched_reused_dest}`）を使う。cargo test は単一テスト名で
//! filter すると新規プロセスを起動するため、これらを個別に呼べば
//! 腕ごとに独立したプロセス状態で計測できる。
//!
//! # メモリ使用量
//!
//! keep-alive は N=4096・1 腕あたり (20 warmup + 20 測定) ×
//! 4096² × 4 bytes ≈ 2.7 GiB。腕ごとに次の腕へ進む前に `keep_alive` を
//! drop する（`gemm_reuse_phase_diag_tests.rs` と同型の方針）。

use std::time::Instant;

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};

use crate::context_cache::{cached_allocator, cached_device, cached_gemm};
use crate::gemm::DiagTiledF32Kernel;

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §12/§13 が後退を
/// 報告した対象形状（N=1024/2048）に、比較対照として改善方向の
/// N=4096 を加える。
const SIZES: [usize; 3] = [1024, 2048, 4096];

/// 読み出し方式 4 腕（本ファイル冒頭「4 腕の定義」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadoutArm {
    LegacyToVec,
    BorrowedKeepAlive,
    BorrowedWithDummyAllocFree,
    PretouchedReusedDest,
}

impl ReadoutArm {
    const ALL: [ReadoutArm; 4] = [
        ReadoutArm::LegacyToVec,
        ReadoutArm::BorrowedKeepAlive,
        ReadoutArm::BorrowedWithDummyAllocFree,
        ReadoutArm::PretouchedReusedDest,
    ];

    fn label(self) -> &'static str {
        match self {
            ReadoutArm::LegacyToVec => "LegacyToVec",
            ReadoutArm::BorrowedKeepAlive => "BorrowedKeepAlive",
            ReadoutArm::BorrowedWithDummyAllocFree => "BorrowedWithDummyAllocFree",
            ReadoutArm::PretouchedReusedDest => "PretouchedReusedDest",
        }
    }
}

fn gen_square_ab(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let a = rng.fill_vec(n * n);
    let b = rng.fill_vec(n * n);
    (a, b)
}

fn median_of(samples: &[f64]) -> Quartiles {
    median_q1_q3(samples)
        .expect("samples collected from successful trials must be non-empty and NaN-free")
}

fn print_quartiles_ms(label: &str, q: Quartiles) {
    println!(
        "    {label}: median={:.4} ms  q1={:.4} ms  q3={:.4} ms  (q3-q1)={:.4} ms",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3,
        (q.q3 - q.q1) * 1e3
    );
}

fn checksum_f64(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum()
}

/// `d2h`（`clone_dtoh`／`memcpy_dtoh` 発行 + 完了同期）・`host_read`
/// （腕別の読み出し・追加確保・copy-out）の 2 区間を計測した 1 反復分。
struct ArmSample {
    d2h_secs: f64,
    host_read_secs: f64,
    checksum: f64,
}

/// 1 反復分を計測する。`c_dev`（プール確保済みデバイス出力）は呼び出し
/// 元がカーネル投入・同期まで済ませた状態で渡す（D2H 以降のみを本関数
/// が計測する。H2D／カーネルの計測は `gemm_reuse_phase_diag_tests.rs`
/// が既に行っているため本ファイルでは対象外）。
///
/// `pretouched_dest` は `ReadoutArm::PretouchedReusedDest` 専用の
/// 事前タッチ済み再利用宛先（呼び出し元がループ外で 1 回だけ確保する）。
#[allow(clippy::too_many_arguments)]
fn measure_one_readout_trial(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    c_dev: &crate::pool::PooledCudaHandle<f32>,
    arm: ReadoutArm,
    keep_alive: &mut Vec<Vec<f32>>,
    pretouched_dest: &mut Vec<f32>,
) -> ArmSample {
    match arm {
        ReadoutArm::LegacyToVec => {
            // off 腕（`readout_var` 既定経路）の再現: `clone_dtoh` で
            // 1 本目を受け取り keep-alive し、`to_vec()` の 2 本目を
            // host_read 区間として計測してから即 drop する。
            let t = Instant::now();
            let out = stream
                .clone_dtoh(&c_dev.as_view())
                .expect("D2H download must succeed");
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let copied = out.to_vec();
            let checksum = checksum_f64(&copied);
            let host_read_secs = t.elapsed().as_secs_f64();
            drop(copied);

            keep_alive.push(out);
            ArmSample {
                d2h_secs,
                host_read_secs,
                checksum,
            }
        }
        ReadoutArm::BorrowedKeepAlive => {
            // on 腕（`host-view-readout` 有効時の `Var::host_view()`）の
            // 再現: `clone_dtoh` の `Vec` を直接読み出し、追加確保・free
            // なしで keep-alive へ積む。
            let t = Instant::now();
            let out = stream
                .clone_dtoh(&c_dev.as_view())
                .expect("D2H download must succeed");
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&out);
            let host_read_secs = t.elapsed().as_secs_f64();

            keep_alive.push(out);
            ArmSample {
                d2h_secs,
                host_read_secs,
                checksum,
            }
        }
        ReadoutArm::BorrowedWithDummyAllocFree => {
            // `BorrowedKeepAlive` と同じ D2H・読み出しに加え、読み出し後
            // に同サイズ `Vec` を確保・全ページへ書き込み・即 drop する。
            // アロケータへ free を発生させる副作用のみを追加した対照腕。
            let t = Instant::now();
            let out = stream
                .clone_dtoh(&c_dev.as_view())
                .expect("D2H download must succeed");
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&out);
            // ゼロ初期化（`vec![0.0f32; n]`）だけでは、glibc の mmap
            // 経由確保がカーネルのゼロページ（COW）で応答しうるため
            // 物理ページへの実書き込みを保証できない。全要素へ非ゼロ
            // 値を明示的に書き込むことで実際のページフォールト・
            // コミットを発生させる。加えて、書き込み結果を一度も
            // 読み出さない dead store は最適化で除去されうるため、
            // 確保・書き込みの両方を `std::hint::black_box` で観測
            // 境界に包み、コンパイラによる除去を防ぐ（#1442 レビュー
            // 指摘対応）。
            let mut dummy: Vec<f32> = std::hint::black_box(vec![0.0f32; out.len()]);
            for (i, x) in dummy.iter_mut().enumerate() {
                *x = std::hint::black_box(i as f32 + 1.0);
            }
            std::hint::black_box(&dummy);
            let host_read_secs = t.elapsed().as_secs_f64();
            drop(dummy);

            keep_alive.push(out);
            ArmSample {
                d2h_secs,
                host_read_secs,
                checksum,
            }
        }
        ReadoutArm::PretouchedReusedDest => {
            // #1437 是正候補 A の先取り対照腕: 事前タッチ済み宛先へ
            // `memcpy_dtoh` し、読み出し後に keep-alive 用の別 `Vec` へ
            // copy-out する（reuse tape がノード storage として D2H
            // 結果を所有する契約を模す）。
            let t = Instant::now();
            stream
                .memcpy_dtoh(&c_dev.as_view(), pretouched_dest)
                .expect("D2H download into pretouched dest must succeed");
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(pretouched_dest);
            let mut owned = Vec::with_capacity(pretouched_dest.len());
            owned.extend_from_slice(pretouched_dest);
            let host_read_secs = t.elapsed().as_secs_f64();

            keep_alive.push(owned);
            ArmSample {
                d2h_secs,
                host_read_secs,
                checksum,
            }
        }
    }
}

fn run_size_arm(n: usize, arm: ReadoutArm) {
    let device = cached_device(0).expect("CUDA device (ordinal 0) must be available");
    let gemm = cached_gemm(&device).expect("CudaGemm construction must succeed");
    let allocator = cached_allocator(&device).expect("CudaAllocator construction must succeed");
    let stream = device.stream().clone();

    let (a, b) = gen_square_ab(0x1436_a000 ^ (n as u64), n);
    let numel = n * n;
    let a_dev = stream.clone_htod(&a).expect("H2D A upload must succeed");
    let b_dev = stream.clone_htod(&b).expect("H2D B upload must succeed");

    // 腕間・プロセス分離実行間で共通に使う参照 checksum（#1442 レビュー
    // 指摘対応）。計測ループ（warmup／測定とも）に含めない独立実行で
    // 1 回だけ求める。`a`／`b` は `n` から決定的に生成されるため
    // （`gen_square_ab` の seed が `n` 依存）、この参照値は「どの腕を
    // どのプロセスで単体実行しても同じ入力に対する同じカーネル出力」
    // という不変量に基づき、腕間の checksum 一致・非ゼロを独立に
    // 検証できる基準になる（同一腕内の先頭値との比較だけでは、ある腕が
    // 一貫してゼロ・誤値を返しても自己無矛盾のため検出できない、という
    // 指摘への対応）。
    let reference_checksum = {
        let mut c_dev = allocator
            .alloc_uninit_f32(numel)
            .expect("pooled output buffer allocation must succeed (reference run)");
        gemm.launch_tiled_f32_pooled(
            &a_dev,
            &b_dev,
            &mut c_dev,
            n as u32,
            n as u32,
            n as u32,
            DiagTiledF32Kernel::Select,
        )
        .expect("launch_tiled_f32_pooled must succeed (reference run)");
        stream
            .synchronize()
            .expect("stream synchronize (kernel completion wait) must succeed (reference run)");
        let out = stream
            .clone_dtoh(&c_dev.as_view())
            .expect("D2H download must succeed (reference run)");
        stream
            .synchronize()
            .expect("stream synchronize after D2H must succeed (reference run)");
        drop(c_dev);
        checksum_f64(&out)
    };
    assert!(
        reference_checksum.is_finite() && reference_checksum != 0.0,
        "reference checksum must be finite and non-zero (n={n}, arm={arm:?})"
    );

    // `PretouchedReusedDest` 専用の事前タッチ済み再利用宛先（ループの
    // 外で 1 回だけ確保・全要素書き込み。§冒頭「4 腕の定義」参照）。
    let mut pretouched_dest = vec![0.0f32; numel];

    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

    let mut run_trial = |keep_alive: &mut Vec<Vec<f32>>| -> ArmSample {
        let mut c_dev = allocator
            .alloc_uninit_f32(numel)
            .expect("pooled output buffer allocation must succeed");
        gemm.launch_tiled_f32_pooled(
            &a_dev,
            &b_dev,
            &mut c_dev,
            n as u32,
            n as u32,
            n as u32,
            DiagTiledF32Kernel::Select,
        )
        .expect("launch_tiled_f32_pooled must succeed");
        stream
            .synchronize()
            .expect("stream synchronize (kernel completion wait) must succeed");
        let sample =
            measure_one_readout_trial(&stream, &c_dev, arm, keep_alive, &mut pretouched_dest);
        drop(c_dev);
        sample
    };

    for _ in 0..WARMUP_TRIALS {
        let _ = run_trial(&mut keep_alive);
    }

    let mut d2h = Vec::with_capacity(MEASURED_TRIALS);
    let mut host_read = Vec::with_capacity(MEASURED_TRIALS);
    let mut checksums = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let s = run_trial(&mut keep_alive);
        d2h.push(s.d2h_secs);
        host_read.push(s.host_read_secs);
        checksums.push(s.checksum);
    }

    // sanity: 全反復で有限・非ゼロ・かつ計測外で独立に求めた
    // `reference_checksum`（腕間で共通）と一致すること（大小関係への
    // assert は行わない。gating しない方針参照）。同一腕内の先頭値との
    // 比較のみでは、ある腕が一貫してゼロ・誤値を返しても自己無矛盾の
    // ため検出できないため、腕間・プロセス分離実行間でも共通に働く
    // 外部参照（`reference_checksum`）との比較へ統一する（#1442 レビュー
    // 指摘対応）。
    assert!(
        checksums.iter().all(|c| c.is_finite() && *c != 0.0),
        "checksum must be finite and non-zero (n={n}, arm={arm:?})"
    );
    assert!(
        checksums
            .iter()
            .all(|&c| (c - reference_checksum).abs() <= reference_checksum.abs() * 1e-9 + 1e-6),
        "checksum must match the arm-independent reference checksum \
         (n={n}, arm={arm:?}, reference={reference_checksum})"
    );
    let first = checksums[0];

    let total: f64 = [&d2h, &host_read].iter().map(|v| median_of(v).median).sum();

    println!(
        "  N={n} arm={} (median over {MEASURED_TRIALS} trials, {WARMUP_TRIALS} warmup) checksum={:.6}:",
        arm.label(),
        first
    );
    print_quartiles_ms("d2h", median_of(&d2h));
    print_quartiles_ms("host_read", median_of(&host_read));
    println!("    sum of medians: {:.4} ms", total * 1e3);

    let _ = allocator.release_cached();
}

/// 実機（CUDA）依存の診断テスト。全 4 腕 × N=1024/2048/4096 を順に計測
/// する（`--test-threads=1` 必須。ファイル冒頭コメント参照）。
#[test]
#[ignore]
fn readout_regression_diag_all_arms() {
    for n in SIZES {
        for arm in ReadoutArm::ALL {
            run_size_arm(n, arm);
        }
    }
}

/// N=1024 単体（15.04 倍後退の対象形状）を素早く再実行するための分割
/// テスト（フル計測が長時間になるため個別再計測を可能にする）。
#[test]
#[ignore]
fn readout_regression_diag_n1024() {
    for arm in ReadoutArm::ALL {
        run_size_arm(1024, arm);
    }
}

/// N=2048 単体（1.20 倍後退・135 ms スパイクの対象形状）。
#[test]
#[ignore]
fn readout_regression_diag_n2048() {
    for arm in ReadoutArm::ALL {
        run_size_arm(2048, arm);
    }
}

/// N=4096 単体（改善方向の対照形状）。
#[test]
#[ignore]
fn readout_regression_diag_n4096() {
    for arm in ReadoutArm::ALL {
        run_size_arm(4096, arm);
    }
}

/// 単一腕・単一サイズ限定の分離実行エントリ（#1442 レビュー指摘対応）。
///
/// 上記の `*_n1024`／`*_n2048`／`*_n4096` は 4 腕を同一プロセス内で順に
/// 実行するため、glibc の動的 mmap 閾値適応・`keep_alive` の一括解放が
/// 後続の腕へ状態として引き継がれうる（例: `LegacyToVec` が先に free を
/// 発生させて以降の腕の確保が既タッチページを再利用できてしまう等）。
/// 「free がないため閾値が適応しない」という H1 の機構は本来腕ごとに
/// 独立したプロセス状態で検証すべきであり、この汚染の有無自体を切り
/// 分けるため、腕単体を新規プロセスとして起動できる入口を用意する。
/// `cargo test --release -p fandhe-ai-backend-cuda --lib
/// readout_regression_diag_n2048_borrowed_keep_alive -- --ignored
/// --test-threads=1` のように単一テスト名で filter すれば、cargo test
/// が腕ごとに新規プロセスを起動するため、他腕のアロケータ状態を一切
/// 引き継がない計測になる。
macro_rules! single_arm_test {
    ($fn_name:ident, $n:expr, $arm:expr) => {
        #[test]
        #[ignore]
        fn $fn_name() {
            run_size_arm($n, $arm);
        }
    };
}

single_arm_test!(
    readout_regression_diag_n1024_legacy_to_vec,
    1024,
    ReadoutArm::LegacyToVec
);
single_arm_test!(
    readout_regression_diag_n1024_borrowed_keep_alive,
    1024,
    ReadoutArm::BorrowedKeepAlive
);
single_arm_test!(
    readout_regression_diag_n1024_borrowed_with_dummy_alloc_free,
    1024,
    ReadoutArm::BorrowedWithDummyAllocFree
);
single_arm_test!(
    readout_regression_diag_n1024_pretouched_reused_dest,
    1024,
    ReadoutArm::PretouchedReusedDest
);

single_arm_test!(
    readout_regression_diag_n2048_legacy_to_vec,
    2048,
    ReadoutArm::LegacyToVec
);
single_arm_test!(
    readout_regression_diag_n2048_borrowed_keep_alive,
    2048,
    ReadoutArm::BorrowedKeepAlive
);
single_arm_test!(
    readout_regression_diag_n2048_borrowed_with_dummy_alloc_free,
    2048,
    ReadoutArm::BorrowedWithDummyAllocFree
);
single_arm_test!(
    readout_regression_diag_n2048_pretouched_reused_dest,
    2048,
    ReadoutArm::PretouchedReusedDest
);

single_arm_test!(
    readout_regression_diag_n4096_legacy_to_vec,
    4096,
    ReadoutArm::LegacyToVec
);
single_arm_test!(
    readout_regression_diag_n4096_borrowed_keep_alive,
    4096,
    ReadoutArm::BorrowedKeepAlive
);
single_arm_test!(
    readout_regression_diag_n4096_borrowed_with_dummy_alloc_free,
    4096,
    ReadoutArm::BorrowedWithDummyAllocFree
);
single_arm_test!(
    readout_regression_diag_n4096_pretouched_reused_dest,
    4096,
    ReadoutArm::PretouchedReusedDest
);

#[cfg(test)]
mod pure_unit_tests {
    use super::*;

    #[test]
    fn arm_labels_are_distinct() {
        let labels: Vec<&str> = ReadoutArm::ALL.iter().map(|a| a.label()).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            labels.len(),
            sorted.len(),
            "arm labels must be pairwise distinct for log parsing"
        );
    }

    #[test]
    fn checksum_f64_matches_manual_sum_for_small_vector() {
        let v = vec![1.0f32, 2.0f32, 3.5f32];
        assert!((checksum_f64(&v) - 6.5).abs() < 1e-9);
    }
}
