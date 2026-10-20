//! CUDA GEMM の reuse が fresh の 2 倍以上遅い原因（イシュー #3186）を切り分ける
//! 診断テスト。`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §14 の Layer B 計装。
//!
//! # 位置づけ
//!
//! - 仮説: bench-fandhe の reuse 窓は結果ノード（readout のホスト `Vec`）を tape に
//!   溜め続けるため free が起きず、毎回まだタッチしていない新規ページへ事前タッチ
//!   と D2H を行う。fresh 窓は毎反復で宛先を解放するため glibc の動的 mmap 閾値が
//!   上がり（32 MiB 以下）、タッチ済みの brk ヒープを再利用できる。
//! - 本ファイルは「窓パターン（`drop`＝反復ごとに宛先解放／`keep`＝宛先を溜める）」×
//!   「宛先方式（`pretouched`／`untouched`／`reused`／`pinned`）」を本番順序
//!   （起動後に同期せず、宛先確保・事前タッチ・D2H・同期の順）で計測し、
//!   ページフォールト数（`/proc/self/stat` の minflt/majflt）を併記する。
//! - 判定規則は `docs/perf/logs/cuda-gemm-reuse-fresh-gap-3186/RULE.txt`（実測前に
//!   固定）が正。起動は `orchestrate.sh`、集計は `aggregate.py`。本ファイルは計測値
//!   の出力のみを担い、腕間の大小へ `assert!` しない（gating しない方針。#2107 と同じ）。
//!
//! # 配置理由と `unsafe` 方針
//!
//! `context_cache`・`launch_tiled_f32_pooled`・`memory::{readback_with,
//! ReadbackDest, ReadbackSentinel}` が `pub(crate)` のためクレートルートの兄弟として
//! 置く。`#[cfg(test)]` に閉じ、本番経路は変更しない。新規 `unsafe` なし。
//! ページフォールト計数は libc（deps-policy 第 10 区分の用途外）を使わず `/proc` の
//! std 読み取りだけで行い、THP 設定などシステムの状態は読むだけで変更しない。
//!
//! # 実行時は必ず `--test-threads=1`、腕ごとに新規プロセス
//!
//! allocator／glibc ヒープの状態を腕間で持ち越さないため、(N, 腕) 単位の `#[ignore]`
//! テストを `orchestrate.sh` が 1 件ずつ独立プロセスで起動する。

use std::time::Instant;

use bench_harness::median_q1_q3;

use crate::context_cache::{cached_allocator, cached_device, cached_gemm};
use crate::gemm::DiagTiledF32Kernel;
use crate::memory::{ReadbackDest, ReadbackSentinel, readback_with};
use crate::readback_attribution_diag_tests_2107::{
    checksum_f64, gen_square_ab, stat_json, stat_us,
};

const WARMUP_TRIALS: usize = 20;
const MEASURED_TRIALS: usize = 20;

/// 前半 4 区間（全腕共通。本番順序のため `kernel_wait` は持たない）。
const FRONT_PHASES: [&str; 4] = ["h2d_a", "h2d_b", "alloc_c", "launch_issue"];

/// 窓パターン。`Drop` は反復ごとに宛先を解放（fresh 窓相当）、`Keep` は溜める（reuse 窓相当）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Window {
    Drop,
    Keep,
}

/// readback 宛先方式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dest {
    /// 本番 `readback_with(PretouchedFresh)` を 1 区間で計測。
    Pretouched,
    /// 本番 `readback_with(PretouchedFresh)` を確保・事前タッチ・D2H 発行・同期へ分解。
    PretouchedSplit,
    /// `stream.clone_dtoh`（未タッチ宛先。safe API）。
    Untouched,
    /// 計測ループ外で 1 回だけ確保・タッチした宛先へ `memcpy_dtoh`。
    Reused,
    /// `readback_with(PinnedStagingReuse)`（copy-out 先 `Vec` は窓パターンに従う）。
    Pinned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Arm {
    window: Window,
    dest: Dest,
}

impl Arm {
    const ALL: [Arm; 9] = [
        Arm::new(Window::Drop, Dest::Pretouched),
        Arm::new(Window::Keep, Dest::Pretouched),
        Arm::new(Window::Drop, Dest::Untouched),
        Arm::new(Window::Keep, Dest::Untouched),
        Arm::new(Window::Drop, Dest::Reused),
        Arm::new(Window::Drop, Dest::Pinned),
        Arm::new(Window::Keep, Dest::Pinned),
        Arm::new(Window::Drop, Dest::PretouchedSplit),
        Arm::new(Window::Keep, Dest::PretouchedSplit),
    ];

    const fn new(window: Window, dest: Dest) -> Self {
        Self { window, dest }
    }

    /// JSONL の `arm` 値（`orchestrate.sh`／`aggregate.py`／RULE.txt と共通）。
    fn label(self) -> &'static str {
        match (self.window, self.dest) {
            (Window::Drop, Dest::Pretouched) => "drop_pretouched",
            (Window::Keep, Dest::Pretouched) => "keep_pretouched",
            (Window::Drop, Dest::Untouched) => "drop_untouched",
            (Window::Keep, Dest::Untouched) => "keep_untouched",
            (Window::Drop, Dest::Reused) => "drop_reused",
            (_, Dest::Reused) => "keep_reused",
            (Window::Drop, Dest::Pinned) => "drop_pinned",
            (Window::Keep, Dest::Pinned) => "keep_pinned",
            (Window::Drop, Dest::PretouchedSplit) => "drop_pretouched_split",
            (Window::Keep, Dest::PretouchedSplit) => "keep_pretouched_split",
        }
    }

    /// readback 区間のサブフェーズ名（計測順）。`d2h` は D2H 発行と同期の和。
    fn readback_parts(self) -> &'static [&'static str] {
        match self.dest {
            Dest::PretouchedSplit => &["dest_alloc", "pretouch_fill", "d2h_issue", "d2h_sync"],
            Dest::Reused => &["d2h_issue", "d2h_sync"],
            _ => &["readback_total"],
        }
    }
}

/// `/proc/self/stat` から (minflt, majflt) を取り出す。comm に空白や括弧を含んでも
/// 最後の `)` 以降だけを見るので壊れない。失敗時は `None`（panic させない）。
fn parse_proc_stat_faults(stat: &str) -> Option<(u64, u64)> {
    let rest = &stat[stat.rfind(')')? + 1..];
    // rest の先頭トークンは state（第 3 フィールド）。minflt は第 10、majflt は第 12。
    let f: Vec<&str> = rest.split_whitespace().collect();
    let minflt = f.get(7)?.parse().ok()?;
    let majflt = f.get(9)?.parse().ok()?;
    Some((minflt, majflt))
}

/// 現在のページフォールト累計。Linux 以外・読み取り失敗は `None`（JSONL では `null`）。
fn read_faults() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/stat").ok()?;
        parse_proc_stat_faults(&s)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// 1 行目の文字列（読み取り失敗は `null` 相当の空）。THP 設定の記録用。
fn read_trimmed(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|s| s.trim().replace('"', "'"))
        .unwrap_or_default()
}

/// `/proc/self/smaps_rollup` の `AnonHugePages`（kB）。取得不能は `None`。
fn anon_huge_pages_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    s.lines()
        .find(|l| l.starts_with("AnonHugePages:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

fn opt_u64_json(v: Option<u64>) -> String {
    v.map_or_else(|| "null".to_string(), |x| x.to_string())
}

struct TrialSample {
    front: Vec<f64>,
    parts: Vec<f64>,
    host_read: f64,
    checksum: f64,
    minflt: Option<u64>,
    majflt: Option<u64>,
}

/// 保持するホスト側宛先。`Keep` 窓では試行をまたいで溜める。
#[derive(Default)]
struct Held {
    keep_alive: Vec<Vec<f32>>,
    reused_dest: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn measure_one_trial(
    device: &crate::device::CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    allocator: &crate::pool::CudaAllocator,
    a: &[f32],
    b: &[f32],
    n: u32,
    arm: Arm,
    held: &mut Held,
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

    // フォールト計数は readback 区間の前後（タイマー区間の外）で読む。
    let faults_before = read_faults();
    let (out, parts): (Option<Vec<f32>>, Vec<f64>) = match arm.dest {
        Dest::Pretouched | Dest::Pinned => {
            let which = if arm.dest == Dest::Pinned {
                ReadbackDest::PinnedStagingReuse
            } else {
                ReadbackDest::PretouchedFresh
            };
            let t = Instant::now();
            let v = readback_with::<f32, _>(&stream, &c_dev.as_view(), which)
                .expect("readback_with must succeed");
            (Some(v), vec![t.elapsed().as_secs_f64()])
        }
        Dest::Untouched => {
            let t = Instant::now();
            let v = stream
                .clone_dtoh(&c_dev.as_view())
                .expect("D2H download must succeed");
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            (Some(v), vec![t.elapsed().as_secs_f64()])
        }
        Dest::PretouchedSplit => {
            let t = Instant::now();
            let mut dest: Vec<f32> = Vec::with_capacity(len);
            let dest_alloc = t.elapsed().as_secs_f64();
            let t = Instant::now();
            dest.resize(len, <f32 as ReadbackSentinel>::SENTINEL);
            let pretouch_fill = t.elapsed().as_secs_f64();
            let t = Instant::now();
            stream
                .memcpy_dtoh(&c_dev.as_view(), &mut dest)
                .expect("D2H into pretouched dest must succeed");
            let d2h_issue = t.elapsed().as_secs_f64();
            let t = Instant::now();
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            let d2h_sync = t.elapsed().as_secs_f64();
            (
                Some(dest),
                vec![dest_alloc, pretouch_fill, d2h_issue, d2h_sync],
            )
        }
        Dest::Reused => {
            let t = Instant::now();
            stream
                .memcpy_dtoh(&c_dev.as_view(), &mut held.reused_dest)
                .expect("D2H into reused dest must succeed");
            let d2h_issue = t.elapsed().as_secs_f64();
            let t = Instant::now();
            stream
                .synchronize()
                .expect("stream synchronize after D2H must succeed");
            (None, vec![d2h_issue, t.elapsed().as_secs_f64()])
        }
    };
    let faults_after = read_faults();

    let t = Instant::now();
    let checksum = match &out {
        Some(v) => checksum_f64(v),
        None => checksum_f64(&held.reused_dest),
    };
    let host_read = t.elapsed().as_secs_f64();

    if let Some(v) = out {
        match arm.window {
            Window::Keep => held.keep_alive.push(v),
            Window::Drop => drop(v),
        }
    }
    drop(c_dev);

    let (minflt, majflt) = match (faults_before, faults_after) {
        (Some((a0, b0)), Some((a1, b1))) => (Some(a1 - a0), Some(b1 - b0)),
        _ => (None, None),
    };
    TrialSample {
        front: vec![h2d_a, h2d_b, alloc_c, launch_issue],
        parts,
        host_read,
        checksum,
        minflt,
        majflt,
    }
}

fn arr_json(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{:.3}", x * 1e6)).collect();
    format!("[{}]", items.join(","))
}

fn opt_arr_json(v: &[Option<u64>]) -> String {
    let items: Vec<String> = v.iter().map(|x| opt_u64_json(*x)).collect();
    format!("[{}]", items.join(","))
}

/// JSONL 1 行を組み立てる。試行ごとの生値（μs）と minflt/majflt 差分を含む。
#[allow(clippy::too_many_arguments)]
fn build_jsonl(
    n: usize,
    arm: Arm,
    front: &[Vec<f64>],
    parts: &[Vec<f64>],
    host_read: &[f64],
    checksum: f64,
    minflt: &[Option<u64>],
    majflt: &[Option<u64>],
    env: &str,
) -> String {
    let mut phases: Vec<String> = Vec::new();
    for (name, v) in FRONT_PHASES.iter().zip(front) {
        phases.push(format!(
            "\"{name}\":{{\"stat\":{},\"trials_us\":{}}}",
            stat_json(&stat_us(v)),
            arr_json(v)
        ));
    }
    for (name, v) in arm.readback_parts().iter().zip(parts) {
        phases.push(format!(
            "\"{name}\":{{\"stat\":{},\"trials_us\":{}}}",
            stat_json(&stat_us(v)),
            arr_json(v)
        ));
    }
    phases.push(format!(
        "\"host_read\":{{\"stat\":{},\"trials_us\":{}}}",
        stat_json(&stat_us(host_read)),
        arr_json(host_read)
    ));
    format!(
        "{{\"issue\":3186,\"n\":{n},\"arm\":\"{}\",\"warmup\":{WARMUP_TRIALS},\"measured\":{MEASURED_TRIALS},\
\"phases\":{{{}}},\"checksum\":{checksum:.6},\"checksum_bits\":\"{:016x}\",\
\"minflt_trials\":{},\"majflt_trials\":{},\"env\":{env}}}",
        arm.label(),
        phases.join(","),
        checksum.to_bits(),
        opt_arr_json(minflt),
        opt_arr_json(majflt),
    )
}

fn env_json() -> String {
    format!(
        "{{\"thp_enabled\":\"{}\",\"thp_defrag\":\"{}\",\"anon_huge_pages_kb\":{}}}",
        read_trimmed("/sys/kernel/mm/transparent_hugepage/enabled"),
        read_trimmed("/sys/kernel/mm/transparent_hugepage/defrag"),
        opt_u64_json(anon_huge_pages_kb()),
    )
}

fn run_size_arm(n: usize, arm: Arm) {
    let device = cached_device(0).expect("CUDA device (ordinal 0) must be available");
    let gemm = cached_gemm(&device).expect("CudaGemm construction must succeed");
    let allocator = cached_allocator(&device).expect("CudaAllocator construction must succeed");
    let stream = device.stream().clone();

    let (a, b) = gen_square_ab(n);
    let numel = n * n;

    // 参照 checksum。参照用 Vec は関数末尾まで drop しない（free が glibc の mmap
    // 閾値を適応させ、`keep` 窓の「free なし」条件を崩すのを防ぐ。#2107 と同じ）。
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

    let mut held = Held {
        keep_alive: Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS),
        reused_dest: if arm.dest == Dest::Reused {
            vec![<f32 as ReadbackSentinel>::SENTINEL; numel]
        } else {
            Vec::new()
        },
    };
    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_trial(&device, &gemm, &allocator, &a, &b, n as u32, arm, &mut held);
    }

    let mut front: Vec<Vec<f64>> = vec![Vec::new(); FRONT_PHASES.len()];
    let mut parts: Vec<Vec<f64>> = vec![Vec::new(); arm.readback_parts().len()];
    let mut host_read = Vec::new();
    let mut checksums = Vec::new();
    let mut minflt = Vec::new();
    let mut majflt = Vec::new();
    for _ in 0..MEASURED_TRIALS {
        let s = measure_one_trial(&device, &gemm, &allocator, &a, &b, n as u32, arm, &mut held);
        for (col, v) in front.iter_mut().zip(s.front) {
            col.push(v);
        }
        for (col, v) in parts.iter_mut().zip(s.parts) {
            col.push(v);
        }
        host_read.push(s.host_read);
        checksums.push(s.checksum);
        minflt.push(s.minflt);
        majflt.push(s.majflt);
    }

    assert!(
        checksums
            .iter()
            .all(|&c| (c - reference_checksum).abs() <= reference_checksum.abs() * 1e-9 + 1e-6),
        "all trial checksums must match the reference checksum \
         (n={n}, arm={arm:?}, reference={reference_checksum})"
    );

    let line = build_jsonl(
        n,
        arm,
        &front,
        &parts,
        &host_read,
        reference_checksum,
        &minflt,
        &majflt,
        &env_json(),
    );
    println!("DIAG_JSON {line}");
    let readback_med: f64 = parts
        .iter()
        .map(|v| median_q1_q3(v).map_or(0.0, |q| q.median))
        .sum();
    println!(
        "HUMAN N={n} arm={} sum_readback_median={:.1} us",
        arm.label(),
        readback_med * 1e6
    );

    drop(reference_out);
    drop(held);
    let _ = allocator.release_cached();
}

macro_rules! single_arm_test {
    ($fn_name:ident, $n:expr, $arm:expr) => {
        #[test]
        #[ignore]
        fn $fn_name() {
            run_size_arm($n, $arm);
        }
    };
}

// 腕名は固定のため、1 N あたり 9 テストを `gen_tests!` で生成する（関数名は N ごとに
// 手書きの識別子を渡す。`paste` 依存は使わない）。
macro_rules! gen_tests {
    ($n:expr; $dp:ident, $kp:ident, $du:ident, $ku:ident, $dr:ident, $dn:ident, $kn:ident, $ds:ident, $ks:ident) => {
        single_arm_test!($dp, $n, Arm::new(Window::Drop, Dest::Pretouched));
        single_arm_test!($kp, $n, Arm::new(Window::Keep, Dest::Pretouched));
        single_arm_test!($du, $n, Arm::new(Window::Drop, Dest::Untouched));
        single_arm_test!($ku, $n, Arm::new(Window::Keep, Dest::Untouched));
        single_arm_test!($dr, $n, Arm::new(Window::Drop, Dest::Reused));
        single_arm_test!($dn, $n, Arm::new(Window::Drop, Dest::Pinned));
        single_arm_test!($kn, $n, Arm::new(Window::Keep, Dest::Pinned));
        single_arm_test!($ds, $n, Arm::new(Window::Drop, Dest::PretouchedSplit));
        single_arm_test!($ks, $n, Arm::new(Window::Keep, Dest::PretouchedSplit));
    };
}

gen_tests!(512;
    window_gap_3186_n512_drop_pretouched,
    window_gap_3186_n512_keep_pretouched,
    window_gap_3186_n512_drop_untouched,
    window_gap_3186_n512_keep_untouched,
    window_gap_3186_n512_drop_reused,
    window_gap_3186_n512_drop_pinned,
    window_gap_3186_n512_keep_pinned,
    window_gap_3186_n512_drop_pretouched_split,
    window_gap_3186_n512_keep_pretouched_split
);
gen_tests!(1024;
    window_gap_3186_n1024_drop_pretouched,
    window_gap_3186_n1024_keep_pretouched,
    window_gap_3186_n1024_drop_untouched,
    window_gap_3186_n1024_keep_untouched,
    window_gap_3186_n1024_drop_reused,
    window_gap_3186_n1024_drop_pinned,
    window_gap_3186_n1024_keep_pinned,
    window_gap_3186_n1024_drop_pretouched_split,
    window_gap_3186_n1024_keep_pretouched_split
);
gen_tests!(2048;
    window_gap_3186_n2048_drop_pretouched,
    window_gap_3186_n2048_keep_pretouched,
    window_gap_3186_n2048_drop_untouched,
    window_gap_3186_n2048_keep_untouched,
    window_gap_3186_n2048_drop_reused,
    window_gap_3186_n2048_drop_pinned,
    window_gap_3186_n2048_keep_pinned,
    window_gap_3186_n2048_drop_pretouched_split,
    window_gap_3186_n2048_keep_pretouched_split
);
gen_tests!(4096;
    window_gap_3186_n4096_drop_pretouched,
    window_gap_3186_n4096_keep_pretouched,
    window_gap_3186_n4096_drop_untouched,
    window_gap_3186_n4096_keep_untouched,
    window_gap_3186_n4096_drop_reused,
    window_gap_3186_n4096_drop_pinned,
    window_gap_3186_n4096_keep_pinned,
    window_gap_3186_n4096_drop_pretouched_split,
    window_gap_3186_n4096_keep_pretouched_split
);

#[cfg(test)]
mod pure_unit_tests {
    use super::*;

    #[test]
    fn arm_labels_are_distinct() {
        let mut labels: Vec<&str> = Arm::ALL.iter().map(|a| a.label()).collect();
        let n = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(n, labels.len());
    }

    #[test]
    fn proc_stat_parser_handles_comm_with_spaces_and_parens() {
        // pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt ...
        let s = "123 (my (weird) comm) S 1 2 3 4 5 6 777 8 9 10 11";
        assert_eq!(parse_proc_stat_faults(s), Some((777, 9)));
        assert_eq!(parse_proc_stat_faults("garbage"), None);
        assert_eq!(parse_proc_stat_faults("1 (x) S 1 2"), None);
    }

    #[test]
    fn readback_parts_match_arm_definition() {
        for arm in Arm::ALL {
            let parts = arm.readback_parts();
            match arm.dest {
                Dest::PretouchedSplit => assert_eq!(parts.len(), 4),
                Dest::Reused => assert_eq!(parts, &["d2h_issue", "d2h_sync"]),
                _ => assert_eq!(parts, &["readback_total"]),
            }
        }
    }

    #[test]
    fn jsonl_line_has_required_keys_and_balanced_braces() {
        for arm in Arm::ALL {
            let front = vec![vec![1e-6, 2e-6]; FRONT_PHASES.len()];
            let parts = vec![vec![3e-6, 4e-6]; arm.readback_parts().len()];
            let line = build_jsonl(
                1024,
                arm,
                &front,
                &parts,
                &[1e-6, 1e-6],
                -1855.597736,
                &[Some(1), None],
                &[Some(0), None],
                "{}",
            );
            for key in [
                "\"issue\":3186",
                "\"n\":1024",
                "\"checksum_bits\":",
                "\"minflt_trials\":[1,null]",
                "\"trials_us\":",
                "\"host_read\":",
            ] {
                assert!(line.contains(key), "missing {key} in {line}");
            }
            assert!(!line.contains('\n'));
            assert_eq!(line.matches('{').count(), line.matches('}').count());
        }
    }
}
