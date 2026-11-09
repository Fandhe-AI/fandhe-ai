//! イシュー #4152 計測バイナリ: 細い形状で NR タイルの端数（n mod NR）が無駄になっていないかを、
//! n を掃引した単スレッド GEMM の時間から診断する（MM-177 `docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md` §3
//! の「L2 の細い形状が低 GFLOPS」の原因候補の切り分け）。
//!
//! 役割: `docs/perf/logs/diag-0140-cpu-gemm-nr-tail-4152/run_round.sh` から 1 round に 1 回ずつ
//! （`t1`／`native` の 2 系列）呼ばれ、標準出力の 1 行 1 計測を `raw/*.txt` に保存する。
//! 判定式は RULE（測る前に単独コミットして固定）に置き、本バイナリは計時と bit 記録だけを担う。
//! 計時は `cpu_jobgrid_bench`（#3177）と同じ `MeasurementConfig::default()`＋`bench_harness::run` の
//! `median_secs` で、n=10 の点を MM-177 §3 の値と比べられる。本番コード・カーネルは変更しない。
//!
//! 使い方: `cpu_gemm_nr_tail_4152 <arm-label> <t1|native> <asc|desc>`
//! - `t1`: 公開入口 `gemm_blis_parallel{,_tn}`。`RAYON_NUM_THREADS=1` でなければ終了コード 2 で止める
//! - `native`: 同じ入口を既定のプールで実行（record_only。n で job grid の列帯数が変わり交絡する）
//! - `asc|desc`: ファミリ内の n の計測順（round の奇偶でドリフトの偏りを抑える）
//! - `arm-label`: `[A-Za-z0-9_-]{1,32}` のみ（出力行へそのまま載るため。A03）
//!
//! 出力: `arm= series= family= trans= m= n= k= isa= mr= nr= rayon_threads= median_secs= gflops= bits_hash=0x..`
//! `bits_hash` は計時の外で、ゼロ初期化した C に 1 回だけ呼んだ結果の全要素 `f32::to_bits()` を
//! FNV-1a 64 で畳んだ値（#3614 と同方式）。`isa`／`mr`／`nr` は実際に駆動経路が選ぶ ISA のカーネル定数。
//! aarch64 以外では NR が 12 にならないため、集計側（`aggregate_4152.py`）が `nr` 不一致として止める。
//! kernel 腕（1 タイル単体の計時）は実装しない（区間分解能の問題。README に未実施と記す）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cpu::gemm_blis::microkernel::Isa;
use fandhe_ai_backend_cpu::{gemm_blis_parallel, gemm_blis_parallel_tn};

const SEED: u64 = 0xC0FFEE;

/// `(ファミリ, 転置, m, k, n の掃引)`。判定対象は各ファミリの n=10・12・24（RULE の PARAMS が正）。
/// 8・11・13 と l1_fwd の 256・264・276 は端数の曲線を見るだけの record_only。
const FAMILIES: &[(&str, &str, usize, usize, &[usize])] = &[
    ("l2_fwd", "nn", 64, 256, &[8, 10, 11, 12, 13, 24]),
    ("l2_dw", "tn", 256, 64, &[8, 10, 11, 12, 13, 24]),
    ("l1_fwd", "nn", 64, 784, &[256, 264, 276]),
];

/// 正方 512 の参照点（GFLOPS の分母に使う）。`(ファミリ, 転置, m, n, k)`。
const REF_SQUARE: (&str, &str, usize, usize, usize) = ("ref_square_512", "nn", 512, 512, 512);

/// 検出された ISA のレジスタタイル定数 `(mr, nr)`。
fn tile_of(isa: Isa) -> (usize, usize) {
    match isa {
        Isa::Scalar => (4, 4),
        Isa::Neon => (8, 12),
        Isa::Avx2 => (6, 16),
        Isa::Avx512 => (8, 32),
    }
}

fn bits_hash(c: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in c {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn usage() -> ! {
    eprintln!("usage: cpu_gemm_nr_tail_4152 <arm-label> <t1|native> <asc|desc>");
    std::process::exit(2);
}

fn measure(
    arm: &str,
    series: &str,
    family: &str,
    trans: &str,
    (m, n, k): (usize, usize, usize),
    isa: Isa,
    threads: usize,
) {
    let mut rng = Xorshift64Star::new(SEED);
    // 転置格納オペランドも要素数は同じなので同一の入力列で足りる。
    let a = rng.fill_vec(m * k);
    let b = rng.fill_vec(k * n);
    let call = |c: &mut [f32]| {
        let r = if trans == "tn" {
            gemm_blis_parallel_tn(&a, &b, c, m, n, k)
        } else {
            gemm_blis_parallel(&a, &b, c, m, n, k)
        };
        if let Err(e) = r {
            eprintln!("gemm failed: {e:?}");
            std::process::exit(1);
        }
    };
    // bits_hash は計時クロージャの外で、ゼロ初期化した C への 1 回呼びから作る。
    let mut c0 = vec![0.0f32; m * n];
    call(&mut c0);
    let hash = bits_hash(&c0);
    let mut c = vec![0.0f32; m * n];
    let measurement = match bench_run(&MeasurementConfig::default(), || call(&mut c)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("measurement failed: {e:?}");
            std::process::exit(1);
        }
    };
    let gflops = 2.0 * (m * n * k) as f64 / measurement.median_secs / 1e9;
    let (mr, nr) = tile_of(isa);
    println!(
        "arm={arm} series={series} family={family} trans={trans} m={m} n={n} k={k} isa={isa:?} \
mr={mr} nr={nr} rayon_threads={threads} median_secs={:.9} gflops={gflops:.3} bits_hash=0x{hash:016x}",
        measurement.median_secs
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        usage();
    }
    let (arm, series, order) = (args[1].as_str(), args[2].as_str(), args[3].as_str());
    let label_ok = !arm.is_empty()
        && arm.len() <= 32
        && arm
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !label_ok || !matches!(series, "t1" | "native") || !matches!(order, "asc" | "desc") {
        usage();
    }
    let threads = rayon::current_num_threads();
    if series == "t1" && threads != 1 {
        eprintln!("t1 系列は RAYON_NUM_THREADS=1 が必要（実際: {threads}）");
        std::process::exit(2);
    }
    let isa = Isa::detect();

    let (rl, rt, rm, rn, rk) = REF_SQUARE;
    measure(arm, series, rl, rt, (rm, rn, rk), isa, threads);
    for &(family, trans, m, k, ns) in FAMILIES {
        let mut ns: Vec<usize> = ns.to_vec();
        if order == "desc" {
            ns.reverse();
        }
        for n in ns {
            measure(arm, series, family, trans, (m, n, k), isa, threads);
        }
    }
}
