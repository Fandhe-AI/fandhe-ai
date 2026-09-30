//! CUDA GEMM reuse の readback（D2H）区間を「宛先確保・事前タッチ・D2H 発行・
//! 同期・ホスト読み出し」へ分解し、`docs/perf/cuda-gemm-reuse-phase-breakdown.md`
//! §12.5 の未説明分（Layer A `matmul` − Σ Layer B）のうち宛先確保・事前タッチが
//! 占める割合を定量化するための診断テスト（イシュー #2107・Layer B 計装）。
//!
//! # 位置づけ
//!
//! - §12.5 は未説明分を本番 `memory::readback`（既定 `ReadbackDest::
//!   PretouchedFresh`。#1437）の「D2H 宛先 `Vec` の新規確保と非ゼロ sentinel に
//!   よる事前タッチ」へ帰属する候補としたが、既存 Layer B
//!   （`gemm_reuse_phase_diag_tests.rs`）の `d2h` は `clone_dtoh` を使い本番と
//!   宛先確保方式が異なるため**未検証**だった。本ファイルはこの帰属を検証する。
//! - 4 腕は `readout_regression_diag_tests_1436.rs` の `ReadoutArm` と意味を
//!   1:1 で揃える（`LegacyToVec`／`BorrowedKeepAlive`／`BorrowedWithDummyAllocFree`
//!   ／`PretouchedFreshDest`。腕の再定義はしない。`PretouchedReusedDest` は対象外）。
//!   1436 が `d2h`／`host_read` の 2 区間・各腕 1 run・ms 単位だったのに対し、
//!   本ファイルは (a) readback 内部のサブフェーズ分解、(b) 各試行に H2D・確保・
//!   投入・カーネル待ちを含めた Σ(matmul 相当) の算出（Layer A `matmul` との
//!   直接突合用）、(c) 腕単位のプロセス分離・μs・JSONL 出力を加える。
//! - 判定規則は `docs/perf/logs/cuda-gemm-readback-attribution-2107/RULE.txt`
//!   （実測前に固定）が正。集計は同ディレクトリの `aggregate.py`、起動は
//!   `orchestrate.sh` が担う。本ファイルは計測値の出力のみを担い、腕間の大小
//!   関係・絶対値へは `assert!` しない（gating しない方針。`gemm_reuse_phase_
//!   diag_tests.rs` と同じ理由で flaky 化を避ける）。
//!
//! # 配置理由
//!
//! `context_cache::{cached_device, cached_gemm, cached_allocator}`・
//! `CudaGemm::launch_tiled_f32_pooled`・`crate::memory::{readback,
//! ReadbackSentinel}` はいずれも `pub(crate)` のため、1436／1182 と同様に
//! クレートルートの兄弟モジュールとして配置する。本番経路（`memory.rs` 等）は
//! 変更せず、本モジュールは `#[cfg(test)]` に閉じる。
//!
//! # 分解方式と `unsafe` を使わない理由
//!
//! 宛先確保と事前タッチは `Vec::with_capacity(n)`（確保のみ。大サイズでは
//! mmap の遅延確保）と `resize(n, SENTINEL)`（全要素への明示書き込み＝ページ
//! フォールトを含む事前タッチ）で分離する。本番の `pretouched_host_vec` は
//! `vec![SENTINEL; n]` で、非ゼロ要素のため `alloc_zeroed` を経由せず全要素を
//! 書き込むので、両者は費用として等価である。この等価性は補助腕
//! `pretouched_fresh_production`（本番 `readback` を分解せず 1 区間で計測）の
//! 中央値と分解腕のサブフェーズ和の一致で実測でも裏付ける（RULE.txt の
//! 計装健全性）。`clone_dtoh` 系は内部確保と D2H 発行を分離できない（未初期化
//! `Vec` を得るには `unsafe` の `set_len` が必要になるため）ので、両者を合わせた
//! 1 区間 `clone_dtoh` として記録する。本ファイルは新規 `unsafe` を持たない。
//!
//! # 実行時は必ず `--test-threads=1`
//!
//! 同一 GPU 上での複数テストスレッド競合を避けるため（1436／1182 と同じ）。
//! 腕ごとに新規プロセスで起動できるよう、(N, 腕) 単位の `#[ignore]` テストを
//! `single_arm_test!` で生成する（allocator の状態を腕間で持ち越さない）。
//!
//! # メモリ使用量
//!
//! ホスト側キープアライブは N=4096 で 1 腕あたり約 (20 warmup + 20 測定) ×
//! 4096² × 4 bytes ≈ 2.7 GiB（`run_gemm_reuse` の reuse tape 蓄積と同型）。
//! 腕ごとにプロセスが終了して解放される。

use std::hint::black_box;
use std::time::Instant;

use bench_harness::{median_q1_q3, rng::Xorshift64Star};

use crate::context_cache::{cached_allocator, cached_device, cached_gemm};
use crate::gemm::DiagTiledF32Kernel;
use crate::memory::ReadbackSentinel;

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// 全腕共通の前半 5 区間（`gemm_reuse_phase_diag_tests.rs` と同じ定義）。
const FRONT_PHASES: [&str; 5] = ["h2d_a", "h2d_b", "alloc_c", "launch_issue", "kernel_wait"];

/// readback 宛先確保方式の腕（1436 の `ReadoutArm` との対応はファイル冒頭参照）。
/// `PretouchedFreshProduction` は 4 腕に数えない補助対照。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    CloneDtohLegacyToVec,
    CloneDtohBorrowedKeepAlive,
    CloneDtohBorrowedDummyAllocFree,
    PretouchedFreshSplit,
    PretouchedFreshProduction,
}

impl Arm {
    const ALL: [Arm; 5] = [
        Arm::CloneDtohLegacyToVec,
        Arm::CloneDtohBorrowedKeepAlive,
        Arm::CloneDtohBorrowedDummyAllocFree,
        Arm::PretouchedFreshSplit,
        Arm::PretouchedFreshProduction,
    ];

    /// JSONL の `arm` 値（`orchestrate.sh`／`aggregate.py`／RULE.txt と共通）。
    fn label(self) -> &'static str {
        match self {
            Arm::CloneDtohLegacyToVec => "clone_dtoh_legacy_to_vec",
            Arm::CloneDtohBorrowedKeepAlive => "clone_dtoh_borrowed_keep_alive",
            Arm::CloneDtohBorrowedDummyAllocFree => "clone_dtoh_borrowed_dummy_alloc_free",
            Arm::PretouchedFreshSplit => "pretouched_fresh_split",
            Arm::PretouchedFreshProduction => "pretouched_fresh_production",
        }
    }

    /// 腕別 readback サブフェーズ名（計測順）。
    fn readback_parts(self) -> &'static [&'static str] {
        match self {
            Arm::CloneDtohLegacyToVec
            | Arm::CloneDtohBorrowedKeepAlive
            | Arm::CloneDtohBorrowedDummyAllocFree => &["clone_dtoh", "d2h_sync"],
            Arm::PretouchedFreshSplit => &["dest_alloc", "pretouch_fill", "d2h_issue", "d2h_sync"],
            Arm::PretouchedFreshProduction => &["readback_total"],
        }
    }
}

/// Layer A（`bench-fandhe` の `gemm --mode reuse`）と同一の入力シード。
/// `scripts/bench/framework-compare/bench-common/src/lib.rs` の `SEED_A`／`SEED_B`
/// と一致させる（Layer A−ΣLayer B の残差へデータ差を混入させないため。#2107）。
const SEED_A: u64 = 0xA11CE;
const SEED_B: u64 = 0xB0B;

/// Layer A と同一入力（A は `SEED_A`、B は `SEED_B` の別ストリーム、各 `n*n` 要素）を生成する。
fn gen_square_ab(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = Xorshift64Star::new(SEED_A).fill_vec(n * n);
    let b = Xorshift64Star::new(SEED_B).fill_vec(n * n);
    (a, b)
}

fn checksum_f64(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum()
}

/// 1 試行分の計測結果（秒）。
struct TrialSample {
    front: [f64; 5],
    readback_parts: Vec<f64>,
    host_read: f64,
    checksum: f64,
}

/// 1 試行を計測する。前半 5 区間は `gemm_reuse_phase_diag_tests.rs::
/// measure_one_phase_trial` と同じ構成（H2D は試行ごとに行う）で、後半は腕別。
#[allow(clippy::too_many_arguments)]
fn measure_one_trial(
    device: &crate::device::CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    allocator: &crate::pool::CudaAllocator,
    a: &[f32],
    b: &[f32],
    n: u32,
    arm: Arm,
    keep_alive: &mut Vec<Vec<f32>>,
) -> TrialSample {
    let stream = device.stream().clone();
    let len = (n as usize) * (n as usize);

    let t = Instant::now();
    let a_dev = stream.clone_htod(a).expect("H2D A upload must succeed");
    let h2d_a = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let b_dev = stream.clone_htod(b).expect("H2D B upload must succeed");
    let h2d_b = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut c_dev = allocator
        .alloc_uninit_f32(len)
        .expect("pooled output buffer allocation must succeed");
    let alloc_c = t.elapsed().as_secs_f64();

    let t = Instant::now();
    gemm.launch_tiled_f32_pooled(
        &a_dev,
        &b_dev,
        &mut c_dev,
        n,
        n,
        n,
        DiagTiledF32Kernel::Select,
    )
    .expect("launch_tiled_f32_pooled (issue only) must succeed");
    let launch_issue = t.elapsed().as_secs_f64();

    let t = Instant::now();
    stream
        .synchronize()
        .expect("stream synchronize (kernel completion wait) must succeed");
    let kernel_wait = t.elapsed().as_secs_f64();

    let (readback_parts, host_read, checksum) = match arm {
        Arm::CloneDtohLegacyToVec
        | Arm::CloneDtohBorrowedKeepAlive
        | Arm::CloneDtohBorrowedDummyAllocFree => {
            let t = Instant::now();
            let out = stream
                .clone_dtoh(&c_dev.as_view())
                .expect("D2H download must succeed");
            let clone_dtoh = t.elapsed().as_secs_f64();
            let t = Instant::now();
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_sync = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&out);
            match arm {
                Arm::CloneDtohLegacyToVec => {
                    // `readout_var` 既定経路の再現: `to_vec` の 2 本目を作り、
                    // 直後に drop する（1 本目は keep-alive）。
                    let copied = out.to_vec();
                    black_box(&copied);
                    drop(copied);
                }
                Arm::CloneDtohBorrowedDummyAllocFree => {
                    // 借用経路に「同サイズの確保・全書き込み・解放」だけを足す
                    // 対照（最適化除去を `black_box` で防ぐ）。
                    let dummy = vec![1.0f32; len];
                    black_box(&dummy);
                    drop(dummy);
                }
                _ => {}
            }
            let host_read = t.elapsed().as_secs_f64();
            keep_alive.push(out);
            (vec![clone_dtoh, d2h_sync], host_read, checksum)
        }
        Arm::PretouchedFreshSplit => {
            let t = Instant::now();
            let mut dest: Vec<f32> = Vec::with_capacity(len);
            let dest_alloc = t.elapsed().as_secs_f64();

            let t = Instant::now();
            dest.resize(len, <f32 as ReadbackSentinel>::SENTINEL);
            let pretouch_fill = t.elapsed().as_secs_f64();

            let t = Instant::now();
            stream
                .memcpy_dtoh(&c_dev.as_view(), &mut dest)
                .expect("D2H download into pretouched-fresh dest must succeed");
            let d2h_issue = t.elapsed().as_secs_f64();

            let t = Instant::now();
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_sync = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&dest);
            let host_read = t.elapsed().as_secs_f64();
            keep_alive.push(dest);
            (
                vec![dest_alloc, pretouch_fill, d2h_issue, d2h_sync],
                host_read,
                checksum,
            )
        }
        Arm::PretouchedFreshProduction => {
            let t = Instant::now();
            let dest = crate::memory::readback::<f32, _>(&stream, &c_dev.as_view())
                .expect("production readback must succeed");
            let readback_total = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let checksum = checksum_f64(&dest);
            let host_read = t.elapsed().as_secs_f64();
            keep_alive.push(dest);
            (vec![readback_total], host_read, checksum)
        }
    };

    drop(c_dev);
    TrialSample {
        front: [h2d_a, h2d_b, alloc_c, launch_issue, kernel_wait],
        readback_parts,
        host_read,
        checksum,
    }
}

/// 1 区間の統計（μs）。
struct Stat {
    median: f64,
    q1: f64,
    q3: f64,
    min: f64,
    max: f64,
}

fn stat_us(samples_secs: &[f64]) -> Stat {
    let q = median_q1_q3(samples_secs)
        .expect("samples collected from successful trials must be non-empty and NaN-free");
    let min = samples_secs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples_secs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Stat {
        median: q.median * 1e6,
        q1: q.q1 * 1e6,
        q3: q.q3 * 1e6,
        min: min * 1e6,
        max: max * 1e6,
    }
}

fn stat_json(s: &Stat) -> String {
    format!(
        "{{\"median_us\":{:.3},\"q1_us\":{:.3},\"q3_us\":{:.3},\"min_us\":{:.3},\"max_us\":{:.3}}}",
        s.median, s.q1, s.q3, s.min, s.max
    )
}

/// 区間中央値（μs）の単純和。
fn sum_medians_us(stats: &[&Stat]) -> f64 {
    stats.iter().map(|s| s.median).sum()
}

/// JSONL 1 行を組み立てる（`serde_json` なしの手組み。ラベルは英数字と `_`
/// のみで、エスケープ不要であることをこのモジュールの定数集合が保証する）。
fn build_jsonl(
    n: usize,
    arm: Arm,
    front: &[Stat],
    parts: &[Stat],
    host_read: &Stat,
    checksum: f64,
) -> String {
    let mut phases: Vec<String> = Vec::new();
    for (name, s) in FRONT_PHASES.iter().zip(front) {
        phases.push(format!("\"{name}\":{}", stat_json(s)));
    }
    for (name, s) in arm.readback_parts().iter().zip(parts) {
        phases.push(format!("\"{name}\":{}", stat_json(s)));
    }
    phases.push(format!("\"host_read\":{}", stat_json(host_read)));
    let sum_readback = sum_medians_us(&parts.iter().collect::<Vec<_>>());
    let sum_front = sum_medians_us(&front.iter().collect::<Vec<_>>());
    format!(
        "{{\"issue\":2107,\"n\":{n},\"arm\":\"{}\",\"warmup\":{WARMUP_TRIALS},\"measured\":{MEASURED_TRIALS},\
\"phases\":{{{}}},\"checksum\":{checksum:.6},\"checksum_bits\":\"{:016x}\",\
\"sum_matmul_equiv_us\":{:.3},\"sum_readback_us\":{:.3}}}",
        arm.label(),
        phases.join(","),
        checksum.to_bits(),
        sum_front + sum_readback,
        sum_readback,
    )
}

fn run_size_arm(n: usize, arm: Arm) {
    let device = cached_device(0).expect("CUDA device (ordinal 0) must be available");
    let gemm = cached_gemm(&device).expect("CudaGemm construction must succeed");
    let allocator = cached_allocator(&device).expect("CudaAllocator construction must succeed");
    let stream = device.stream().clone();

    let (a, b) = gen_square_ab(n);
    let numel = n * n;

    // 参照 checksum。計測ループの外で 1 回だけ求め、`Vec` は関数末尾まで
    // drop しない（1436 の #1442 対応と同じ理由: 参照側の free が glibc の
    // mmap 閾値を適応させ、keep-alive 腕の「free なし」条件を崩すのを防ぐ）。
    let a_dev = stream.clone_htod(&a).expect("H2D A upload must succeed");
    let b_dev = stream.clone_htod(&b).expect("H2D B upload must succeed");
    let mut c_ref = allocator
        .alloc_uninit_f32(numel)
        .expect("pooled output buffer allocation must succeed (reference run)");
    gemm.launch_tiled_f32_pooled(
        &a_dev,
        &b_dev,
        &mut c_ref,
        n as u32,
        n as u32,
        n as u32,
        DiagTiledF32Kernel::Select,
    )
    .expect("launch_tiled_f32_pooled must succeed (reference run)");
    stream
        .synchronize()
        .expect("stream synchronize must succeed (reference run)");
    let reference_out = stream
        .clone_dtoh(&c_ref.as_view())
        .expect("D2H download must succeed (reference run)");
    stream
        .synchronize()
        .expect("stream synchronize after D2H must succeed (reference run)");
    drop(c_ref);
    drop(a_dev);
    drop(b_dev);
    let reference_checksum = checksum_f64(&reference_out);
    assert!(
        reference_checksum.is_finite() && reference_checksum != 0.0,
        "reference checksum must be finite and non-zero (n={n}, arm={arm:?})"
    );

    let mut keep_alive: Vec<Vec<f32>> = Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS);
    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_trial(
            &device,
            &gemm,
            &allocator,
            &a,
            &b,
            n as u32,
            arm,
            &mut keep_alive,
        );
    }

    let parts_len = arm.readback_parts().len();
    let mut front: Vec<Vec<f64>> = vec![Vec::with_capacity(MEASURED_TRIALS); FRONT_PHASES.len()];
    let mut parts: Vec<Vec<f64>> = vec![Vec::with_capacity(MEASURED_TRIALS); parts_len];
    let mut host_read: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
    let mut checksums: Vec<f64> = Vec::with_capacity(MEASURED_TRIALS);
    for _ in 0..MEASURED_TRIALS {
        let s = measure_one_trial(
            &device,
            &gemm,
            &allocator,
            &a,
            &b,
            n as u32,
            arm,
            &mut keep_alive,
        );
        for (col, v) in front.iter_mut().zip(s.front) {
            col.push(v);
        }
        for (col, v) in parts.iter_mut().zip(s.readback_parts) {
            col.push(v);
        }
        host_read.push(s.host_read);
        checksums.push(s.checksum);
    }

    // sanity のみ（gating しない方針。大小関係への assert は行わない）。
    assert!(
        checksums
            .iter()
            .all(|&c| (c - reference_checksum).abs() <= reference_checksum.abs() * 1e-9 + 1e-6),
        "all trial checksums must match the reference checksum \
         (n={n}, arm={arm:?}, reference={reference_checksum})"
    );

    let front_stats: Vec<Stat> = front.iter().map(|v| stat_us(v)).collect();
    let part_stats: Vec<Stat> = parts.iter().map(|v| stat_us(v)).collect();
    let host_stat = stat_us(&host_read);
    let line = build_jsonl(
        n,
        arm,
        &front_stats,
        &part_stats,
        &host_stat,
        reference_checksum,
    );
    println!("DIAG_JSON {line}");
    println!(
        "HUMAN N={n} arm={} sum_readback_median={:.1} us",
        arm.label(),
        sum_medians_us(&part_stats.iter().collect::<Vec<_>>())
    );

    drop(reference_out);
    drop(keep_alive);
    let _ = allocator.release_cached();
}

/// 腕単体を新規プロセスとして起動できる入口（1436 の `single_arm_test!` と
/// 同型）。`orchestrate.sh` が `--ignored --exact <name>` で 1 テストずつ起動する。
macro_rules! single_arm_test {
    ($fn_name:ident, $n:expr, $arm:expr) => {
        #[test]
        #[ignore]
        fn $fn_name() {
            run_size_arm($n, $arm);
        }
    };
}

macro_rules! arms_for_n {
    ($n:expr, $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        single_arm_test!($a, $n, Arm::CloneDtohLegacyToVec);
        single_arm_test!($b, $n, Arm::CloneDtohBorrowedKeepAlive);
        single_arm_test!($c, $n, Arm::CloneDtohBorrowedDummyAllocFree);
        single_arm_test!($d, $n, Arm::PretouchedFreshSplit);
        single_arm_test!($e, $n, Arm::PretouchedFreshProduction);
    };
}

arms_for_n!(
    1024,
    readback_attribution_2107_n1024_clone_dtoh_legacy_to_vec,
    readback_attribution_2107_n1024_clone_dtoh_borrowed_keep_alive,
    readback_attribution_2107_n1024_clone_dtoh_borrowed_dummy_alloc_free,
    readback_attribution_2107_n1024_pretouched_fresh_split,
    readback_attribution_2107_n1024_pretouched_fresh_production
);
arms_for_n!(
    2048,
    readback_attribution_2107_n2048_clone_dtoh_legacy_to_vec,
    readback_attribution_2107_n2048_clone_dtoh_borrowed_keep_alive,
    readback_attribution_2107_n2048_clone_dtoh_borrowed_dummy_alloc_free,
    readback_attribution_2107_n2048_pretouched_fresh_split,
    readback_attribution_2107_n2048_pretouched_fresh_production
);
arms_for_n!(
    4096,
    readback_attribution_2107_n4096_clone_dtoh_legacy_to_vec,
    readback_attribution_2107_n4096_clone_dtoh_borrowed_keep_alive,
    readback_attribution_2107_n4096_clone_dtoh_borrowed_dummy_alloc_free,
    readback_attribution_2107_n4096_pretouched_fresh_split,
    readback_attribution_2107_n4096_pretouched_fresh_production
);

#[cfg(test)]
mod pure_unit_tests {
    use super::*;

    fn stat(m: f64) -> Stat {
        Stat {
            median: m,
            q1: m,
            q3: m,
            min: m,
            max: m,
        }
    }

    #[test]
    fn arm_labels_are_distinct() {
        let mut labels: Vec<&str> = Arm::ALL.iter().map(|a| a.label()).collect();
        let n = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(n, labels.len(), "arm labels must be pairwise distinct");
    }

    #[test]
    fn checksum_f64_matches_manual_sum_for_small_vector() {
        assert!((checksum_f64(&[1.0f32, 2.0, 3.5]) - 6.5).abs() < 1e-9);
    }

    #[test]
    fn sum_medians_is_simple_sum() {
        let (a, b) = (stat(1.5), stat(2.25));
        assert!((sum_medians_us(&[&a, &b]) - 3.75).abs() < 1e-12);
    }

    #[test]
    fn jsonl_line_has_required_keys_and_balanced_braces() {
        for arm in Arm::ALL {
            let front: Vec<Stat> = (0..5).map(|i| stat(i as f64)).collect();
            let parts: Vec<Stat> = arm.readback_parts().iter().map(|_| stat(2.0)).collect();
            let line = build_jsonl(1024, arm, &front, &parts, &stat(1.0), -1855.597736);
            for key in [
                "\"issue\":2107",
                "\"n\":1024",
                "\"arm\":",
                "\"phases\":",
                "\"checksum_bits\":",
                "\"sum_matmul_equiv_us\":",
                "\"sum_readback_us\":",
                "\"host_read\":",
                "\"median_us\":",
            ] {
                assert!(line.contains(key), "missing {key} in {line}");
            }
            for name in FRONT_PHASES.iter().chain(arm.readback_parts()) {
                assert!(line.contains(&format!("\"{name}\":")), "missing {name}");
            }
            assert!(!line.contains('\n'));
            let open = line.matches('{').count();
            let close = line.matches('}').count();
            assert_eq!(open, close, "unbalanced braces: {line}");
        }
    }

    #[test]
    fn jsonl_sum_matmul_equiv_excludes_host_read() {
        let front: Vec<Stat> = (0..5).map(|_| stat(10.0)).collect();
        let parts = vec![stat(3.0), stat(4.0)];
        let line = build_jsonl(
            2048,
            Arm::CloneDtohBorrowedKeepAlive,
            &front,
            &parts,
            &stat(999.0),
            1.0,
        );
        assert!(line.contains("\"sum_matmul_equiv_us\":57.000"), "{line}");
        assert!(line.contains("\"sum_readback_us\":7.000"), "{line}");
    }
}
