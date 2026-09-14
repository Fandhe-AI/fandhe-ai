//! Metal 版 readout legacy 後退の 4 腕診断ハーネス（イシュー #1695。
//! CUDA 側 `crates/backend-cuda/src/readout_regression_diag_tests_1436.rs`
//! （イシュー #1436）と同型の構成を Metal へ移植する。腕の定義・純関数
//! ヘルパは `crate::readout_regression_diag_arms` を参照）。
//!
//! # 背景
//!
//! `#1520`（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §17）で、
//! bench-fandhe の Metal `--readout borrowed`（`Var::host_view()` 借用）
//! が `--readout legacy`（`to_tensor()` + `.to_vec()`）に対し N=1024
//! fresh/reuse で 1.21／1.11 倍後退し REJECT（legacy フォールバック維持）
//! となった。`#1574` の低レイヤー診断（`docs/perf/lowlayer-diagnosis-
//! 2026-09-12.md` §5）で、その後退は `matmul` 区間（GPU 待ち＋ホストへの
//! 読み出し）に閉じており `host_copy` は 0 ms、checksum は同一と局所化
//! されたが、**機構は未特定**のまま残っている。本ファイルはその機構を
//! 切り分けるための 4 腕診断ハーネスであり、実機実測・機構記録は
//! 兄弟イシュー #1696 が担う（本ファイルは実機なしでもコンパイル
//! できることのみを担保する）。
//!
//! # `bench-fandhe`／本番 `dispatch_auto` の読み出し実体と本ファイルの
//! 対応
//!
//! | 項目 | 実体 |
//! | --- | --- |
//! | 本番 reuse の `matmul` 区間 | `Var::matmul` → `MetalBackendOps::gemm`（`ops.rs`）→ `MetalGemm::dispatch_auto` → `alloc_uninit_pooled(C)` → encode → `synchronize`（commit + `waitUntilCompleted`）→ `read_to_vec()` → `Tensor::new` |
//! | legacy readout（`bench-fandhe` off 腕） | `to_tensor()`（`Arc` 複製）→ `.to_vec()` で 2 本目確保 → 反復末尾で free |
//! | borrowed readout（`bench-fandhe` on 腕。#1438 で既定経路） | `host_view()`（`Arc` 複製のみ）。追加確保・free なし |
//! | 両者の唯一の実質差 | `read_to_vec` が返す 1 本目を「2 本目へ複製して free するか」「そのまま keep-alive するか」 |
//! | 「D2H」に相当する区間 | UMA のため明示転送は存在せず、`read_to_vec`（`contents()` からの memcpy）が相当する（本ファイルでは `readback` と呼ぶ。`gemm_reuse_phase_diag_tests.rs` の区間定義と同じ） |
//!
//! # 配置理由（`gemm_reuse_phase_diag_tests.rs` と同じ判断）
//!
//! `context_cache::{cached_context, cached_gemm}`（`pub(crate)`）・
//! `MetalGemm::diag_encode_tiled_nn`（`#[cfg(test)] pub(crate)`）・
//! `MetalBuffer::read_into_slice`（`#[cfg(test)] pub(crate)`。本イシューで
//! `buffer.rs` に追加）へ到達するため、integration test ではなく
//! `lib.rs` の兄弟モジュールとして配置する。本ファイルは**診断専用**で
//! あり、`ops.rs`／`gemm.rs`（`dispatch_auto` 系）／`memory.rs` 等の本番
//! 経路は一切変更しない。
//!
//! # 4 腕の定義
//!
//! [`crate::readout_regression_diag_arms::ReadoutArm`] 参照。CUDA 版の
//! 5 腕目 `PretouchedFreshDest`（CUDA 本番の是正 `ReadbackDest::
//! PretouchedFresh`〈#1437〉に対応する腕）は、Metal 側に対応する本番
//! 機構が存在しないため本イシューでは移植しない（意図的なスコープ
//! 判断）。
//!
//! # `readback`／`host_read` 区間定義
//!
//! - `readback`: `MetalBuffer::read_to_vec()`（または `PretouchedReusedDest`
//!   腕では `read_into_slice`）。CUDA の `d2h` に相当するが、UMA のため
//!   明示転送ではなく `contents()` からの memcpy である点が異なる
//! - `host_read`: 腕別の読み出し・追加確保・copy-out（CUDA の
//!   `host_read` と同一の位置づけ）
//!
//! # gating しない方針（`gemm_reuse_phase_diag_tests.rs`／CUDA 側
//! `readout_regression_diag_tests_1436.rs` と同じ理由）
//!
//! 本ファイルの `#[test]` は実行が成功すること（各腕が例外なく完了し
//! 有限・非ゼロの出力を返すこと）のみを検証条件とし、腕間の大小関係・
//! 絶対値への `assert!` は行わない（環境揺らぎによる flaky 化防止）。
//! 数値は `println!` に残し、実機実測記録（イシュー #1696 が担う
//! `docs/perf/` への転記）の一次情報とする。腕間の checksum 一致
//! （同一入力・同一カーネルなら同一出力になるはずという sanity）のみ
//! `assert!` する。この sanity は REQ-2 の複合判定とは無関係。
//!
//! # 機構仮説は中立表現に留める
//!
//! `crate::readout_regression_diag_arms` モジュール冒頭コメント参照。
//! CUDA 側の glibc 固有機構説明（`M_MMAP_THRESHOLD` 動的適応）を Metal
//! （macOS・libmalloc）へそのまま当てはめず、機構の確認・記録は実機実測
//! （イシュー #1696）へ委ねる。
//!
//! # 実行時は必ず `--test-threads=1`（同一 GPU 上の競合を避ける。
//! `gemm_reuse_phase_diag_tests.rs` と同じ理由）
//!
//! # プロセス分離実行
//!
//! `*_all_arms`／`*_n{1024,2048,4096}` は 4 腕を同一プロセス内で順に
//! 実行するため、アロケータの動的な閾値適応・`keep_alive` の一括解放が
//! 後続の腕へ状態として引き継がれうる（CUDA 側 #1442 レビュー指摘と同じ
//! 懸念）。腕間の独立性を保証した比較が必要な場合はファイル末尾の
//! 単一腕専用テスト（`readout_regression_diag_n{1024,2048,4096}_
//! {legacy_to_vec,borrowed_keep_alive,borrowed_with_dummy_alloc_free,
//! pretouched_reused_dest}`）を使う。単一テスト名で `cargo test` を
//! filter すると新規プロセスを起動するため、これらを個別に呼べば
//! 腕ごとに独立したプロセス状態で計測できる。
//!
//! # メモリ使用量
//!
//! keep-alive は N=4096・1 腕あたり (20 warmup + 20 測定) ×
//! 4096² × 4 bytes ≈ 2.7 GiB（統合メモリ上。`gemm_reuse_phase_diag_
//! tests.rs`・CUDA 側 #1436 と同じオーダー）。腕ごとに次の腕へ進む前に
//! `keep_alive` を drop する。
//!
//! # プロダクションコード不変
//!
//! `gemm.rs`／`ops.rs`（`dispatch_auto` 系）／`memory.rs`／`context.rs`／
//! `pool.rs` への変更はない。`buffer.rs` への変更は新規 `#[cfg(test)]`
//! 限定ヘルパ `read_into_slice` の追加のみに限定する。

use std::time::Instant;

use bench_harness::{Quartiles, median_q1_q3};

use crate::buffer::MetalBuffer;
use crate::context_cache::{cached_context, cached_gemm};
use crate::gemm_reuse_phase_diag_tests::{MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab};
use crate::readout_regression_diag_arms::{
    ReadoutArm, checksum_f64, dummy_alloc_touch_free, pretouched_host_vec_f32,
};
use crate::tile;

/// `docs/perf/metal-gemm-candle-gate-remeasurement.md` §17 が後退を
/// 報告した対象形状（N=1024）に、比較対照として N=2048／4096 を加える
/// （CUDA 側 #1436 の `SIZES` と同一の 3 形状）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

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

/// `readback`（`MetalBuffer::read_to_vec`／`read_into_slice`）・
/// `host_read`（腕別の読み出し・追加確保・copy-out）の 2 区間を計測した
/// 1 反復分。
struct ArmSample {
    readback_secs: f64,
    host_read_secs: f64,
    checksum: f64,
}

/// 1 反復分の読み出しを計測する。`c_buf` は呼び出し元が encode ＋
/// `ctx.synchronize()` まで済ませた状態で渡す（アップロード・エンコード・
/// commit_wait の計測は `gemm_reuse_phase_diag_tests.rs` が既に行って
/// いるため本ファイルでは対象外。readback 以降のみを計測する）。
///
/// `pretouched_dest` は `ReadoutArm::PretouchedReusedDest` 専用の事前
/// タッチ済み再利用宛先（呼び出し元がループ外で 1 回だけ確保する）。
fn measure_one_readout_trial(
    c_buf: &MetalBuffer,
    arm: ReadoutArm,
    keep_alive: &mut Vec<Vec<f32>>,
    pretouched_dest: &mut Vec<f32>,
) -> ArmSample {
    match arm {
        ReadoutArm::LegacyToVec => {
            // legacy readout（`bench-fandhe` off 腕）の再現: `read_to_vec`
            // で 1 本目を受け取り keep-alive し、`to_vec()` の 2 本目を
            // host_read 区間として計測してから即 drop する。
            let t = Instant::now();
            let out = c_buf.read_to_vec();
            let readback_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let copied = out.to_vec();
            let checksum = checksum_f64(&copied);
            let host_read_secs = t.elapsed().as_secs_f64();
            drop(copied);

            keep_alive.push(out);
            ArmSample {
                readback_secs,
                host_read_secs,
                checksum,
            }
        }
        ReadoutArm::BorrowedKeepAlive => {
            // borrowed readout（`bench-fandhe` on 腕。#1438 で既定経路）の
            // 再現: `read_to_vec` の `Vec` を直接読み出し、追加確保・free
            // なしで keep-alive へ積む。
            let t = Instant::now();
            let out = c_buf.read_to_vec();
            let readback_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&out);
            let host_read_secs = t.elapsed().as_secs_f64();

            keep_alive.push(out);
            ArmSample {
                readback_secs,
                host_read_secs,
                checksum,
            }
        }
        ReadoutArm::BorrowedWithDummyAllocFree => {
            // `BorrowedKeepAlive` と同じ readback・読み出しに加え、読み出し
            // 後に同サイズ `Vec` を確保・全ページへ書き込み・即 drop する。
            // アロケータへ free を発生させる副作用のみを追加した対照腕。
            let t = Instant::now();
            let out = c_buf.read_to_vec();
            let readback_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&out);
            dummy_alloc_touch_free(out.len());
            let host_read_secs = t.elapsed().as_secs_f64();

            keep_alive.push(out);
            ArmSample {
                readback_secs,
                host_read_secs,
                checksum,
            }
        }
        ReadoutArm::PretouchedReusedDest => {
            // 事前タッチ済み宛先へ `read_into_slice`（確保を伴わない
            // memcpy）し、読み出し後に keep-alive 用の別 `Vec` へ
            // copy-out する（reuse tape がノード storage として readback
            // 結果を所有する契約を模す。CUDA 側 `PretouchedReusedDest`
            // 腕と同型）。
            let t = Instant::now();
            c_buf.read_into_slice(pretouched_dest);
            let readback_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(pretouched_dest);
            let mut owned = Vec::with_capacity(pretouched_dest.len());
            owned.extend_from_slice(pretouched_dest);
            let host_read_secs = t.elapsed().as_secs_f64();

            keep_alive.push(owned);
            ArmSample {
                readback_secs,
                host_read_secs,
                checksum,
            }
        }
    }
}

fn run_size_arm(n: usize, arm: ReadoutArm) {
    let ctx = cached_context().expect("Metal device (system default) must be available");
    let gemm = cached_gemm(&ctx).expect("MetalGemm construction must succeed");
    let cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());

    let (a, b) = gen_square_ab(0x1695_a000 ^ (n as u64), n);
    let numel = n * n;
    let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A upload must succeed");
    let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B upload must succeed");

    // 腕間・プロセス分離実行間で共通に使う参照 checksum（CUDA 側 #1442
    // レビュー指摘対応と同じ判断）。計測ループ（warmup／測定とも）に
    // 含めない独立実行で 1 回だけ求める。`reference_out` はこの関数の
    // 末尾まで drop しない（先行 free が「free が一切ない」対照条件
    // 〈`BorrowedKeepAlive`〉を汚染しないようにするため。CUDA 側と同じ
    // 理由）。
    let (reference_checksum, reference_out) = {
        let c_buf = MetalBuffer::alloc_uninit_pooled(&ctx, numel)
            .expect("pooled output buffer allocation must succeed (reference run)");
        gemm.diag_encode_tiled_nn(&ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
            .expect("diag_encode_tiled_nn (reference run) must succeed");
        ctx.synchronize()
            .expect("synchronize (reference run) must succeed");
        let out = c_buf.read_to_vec();
        drop(c_buf);
        let checksum = checksum_f64(&out);
        (checksum, out)
    };
    assert!(
        reference_checksum.is_finite() && reference_checksum != 0.0,
        "reference checksum must be finite and non-zero (n={n}, arm={arm:?})"
    );

    // `PretouchedReusedDest` 専用の事前タッチ済み再利用宛先（ループの
    // 外で 1 回だけ確保）。
    let mut pretouched_dest = pretouched_host_vec_f32(numel);

    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);

    let run_trial = |keep_alive: &mut Vec<Vec<f32>>, pretouched_dest: &mut Vec<f32>| -> ArmSample {
        let c_buf = MetalBuffer::alloc_uninit_pooled(&ctx, numel)
            .expect("pooled output buffer allocation must succeed");
        gemm.diag_encode_tiled_nn(&ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
            .expect("diag_encode_tiled_nn must succeed");
        ctx.synchronize().expect("synchronize must succeed");
        let sample = measure_one_readout_trial(&c_buf, arm, keep_alive, pretouched_dest);
        drop(c_buf);
        sample
    };

    for _ in 0..WARMUP_TRIALS {
        let _ = run_trial(&mut keep_alive, &mut pretouched_dest);
    }

    let mut readback = Vec::with_capacity(MEASURED_TRIALS);
    let mut host_read = Vec::with_capacity(MEASURED_TRIALS);
    let mut checksums = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let s = run_trial(&mut keep_alive, &mut pretouched_dest);
        readback.push(s.readback_secs);
        host_read.push(s.host_read_secs);
        checksums.push(s.checksum);
    }

    // sanity: 全反復で有限・非ゼロ・かつ計測外で独立に求めた
    // `reference_checksum`（腕間で共通）と一致すること（大小関係への
    // assert は行わない。gating しない方針参照）。
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

    let total: f64 = [&readback, &host_read]
        .iter()
        .map(|v| median_of(v).median)
        .sum();

    println!(
        "  N={n} arm={} requested_tile={cfg:?} (median over {MEASURED_TRIALS} trials, {WARMUP_TRIALS} warmup) checksum={:.6}:",
        arm.label(),
        first
    );
    print_quartiles_ms("readback", median_of(&readback));
    print_quartiles_ms("host_read", median_of(&host_read));
    println!("    sum of medians: {:.4} ms", total * 1e3);

    // `reference_out` をここまで明示的に生かす（測定ループ全体を通じて
    // free させない。CUDA 側と同じ理由・同じ固定位置での `drop`）。
    drop(reference_out);
}

/// 実機（Metal）依存の診断テスト。全 4 腕 × N=1024/2048/4096 を順に計測
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

/// N=1024 単体（#1520 が後退を報告した対象形状）を素早く再実行するための
/// 分割テスト（フル計測が長時間になるため個別再計測を可能にする）。
#[test]
#[ignore]
fn readout_regression_diag_n1024() {
    for arm in ReadoutArm::ALL {
        run_size_arm(1024, arm);
    }
}

/// N=2048 単体。
#[test]
#[ignore]
fn readout_regression_diag_n2048() {
    for arm in ReadoutArm::ALL {
        run_size_arm(2048, arm);
    }
}

/// N=4096 単体（比較対照形状）。
#[test]
#[ignore]
fn readout_regression_diag_n4096() {
    for arm in ReadoutArm::ALL {
        run_size_arm(4096, arm);
    }
}

/// 単一腕・単一サイズ限定の分離実行エントリ（CUDA 側 #1442 レビュー
/// 指摘対応と同じ判断）。
///
/// 上記の `*_n1024`／`*_n2048`／`*_n4096` は 4 腕を同一プロセス内で順に
/// 実行するため、アロケータの状態・`keep_alive` の一括解放が後続の腕へ
/// 引き継がれうる。腕ごとに独立したプロセス状態で計測したい場合は本
/// マクロが生成する単一テストを使う:
///
/// ```text
/// cargo test --release -p fandhe-ai-backend-metal --lib \
///     readout_regression_diag_n1024_borrowed_keep_alive \
///     -- --ignored --nocapture --test-threads=1
/// ```
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
