//! イシュー #4151 計測バイナリ（base 腕）: M4（`<metal-node>`）の CPU GEMM N=256（m=n=k=256・`nn`）の
//! 1 回あたり wall 時間を、系列ごとに記録する。
//!
//! 役割: スコアボード 0.14.0 の M-CPU-G256 の分類（現「ハードの上限」）が成り立つかを、GEMM 駆動部の
//! 固定費（worker の起床・fork-join・pack・C staging・確保・SME の `smstart`／`smstop`）の
//! 実測で確かめる診断の土台。区間の内訳は本番コードに入れず、計測用の複製ツリーにだけ当てる
//! `docs/perf/logs/diag-0140-m-cpu-g256-4151/diag_timing.patch` が本ファイルのフック行を差し替えて
//! 追加出力を足す。SME ON の腕は同ディレクトリの `on-arm.patch`（const を 1 行 true にする）で作る。
//! 事前登録規則は同ディレクトリの `RULE.txt`（PR-1 時点は雛形 `RULE.template.txt`）。
//!
//! 使い方: `cpu_gemm_n256_fixed_cost_4151 <arm-label>`（`off-base`／`on-base`／`off-timed`／`on-timed`。
//! ラベルは記録するだけで挙動は変えない）。スレッド数は機体の既定プール（`RAYON_NUM_THREADS` 禁止）。
//! 呼び出し元: `run_round.sh`。出力の集計先: `aggregate_4151.py`（`ROW4151`／`ENV4151` を厳格にパース）。
//!
//! 出力は 1 行 1 レコード、`key=value` を空白区切りで並べる。
//! `ROW4151 arm= series= m= n= k= rayon_threads= calls= median_ns= mean_ns= bits_hash=0x..` に、
//! timed 腕では区間の内訳（`diag_fixed_cost_timer`）が後ろに続く。
//! `bits_hash` は全要素 `to_bits()` を FNV-1a 64 で畳んだ値。GEMM 系列（hot・gap・serial_t1）は、
//! 計測ループの最後の 1 回の前に C をゼロへ戻し（計時の外）、その呼び出し（計時フック有効のまま）の
//! 出力を計測窓の後で畳む。cold_first_call と forkjoin 系（GEMM 出力なし）は起動直後の 1 回の出力の値。
//! 腕・系列間の一致で、計時フックと SME の有無が数値を変えないことを確かめる。
//!
//! 公開入口 `gemm_blis_parallel`／`gemm_blis` と公開の診断 API だけを呼ぶ。`unsafe`・`asm!`・依存の追加はない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{
    gemm_blis, gemm_blis_parallel, small_shape_cap_report, sme_report, thread_limit_report,
};
use rayon::prelude::*;
use std::cell::RefCell;
use std::time::{Duration, Instant};

const SEED: u64 = 0xC0FFEE;
const DIM: usize = 256;
const N_WARMUP: usize = 50;
const N_MEASURE: usize = 500;
/// `series=gap` で各呼び出しの前にスピン待ちする時間（µs）。RULE の PARAMS `gap_us` と一致させる。
const GAP_US: u64 = 50;
/// 本番の `TWO_D_JOBS_PER_WORKER`（`gemm_blis/mod.rs`）。forkjoin の床の job 数を揃えるため。
const JOBS_PER_WORKER: usize = 2;

fn fnv1a_bits(c: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in c {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// 昇順に並べ替え済みの標本の中央値（偶数個なら上側の要素。500 個固定のため区別は影響しない）。
fn median_sorted(sorted: &[u128]) -> u128 {
    sorted[sorted.len() / 2]
}

fn mean(samples: &[u128]) -> f64 {
    samples.iter().sum::<u128>() as f64 / samples.len() as f64
}

/// `Instant` の分解能（連続呼び出しで観測した最小の非ゼロ差）と空区間 1 回のコストを返す。
fn calibrate_timer() -> (u128, f64) {
    let mut min_tick = u128::MAX;
    let mut prev = Instant::now();
    for _ in 0..100_000 {
        let now = Instant::now();
        let d = now.duration_since(prev).as_nanos();
        if d > 0 && d < min_tick {
            min_tick = d;
        }
        prev = now;
    }
    let t0 = Instant::now();
    for _ in 0..100_000 {
        let s = Instant::now();
        std::hint::black_box(s.elapsed());
    }
    let pair = t0.elapsed().as_nanos() as f64 / 100_000.0;
    (min_tick, pair)
}

/// 指定時間だけスピン待ちする（sleep はスケジューラ起床の揺らぎを持ち込むため使わない）。
fn spin_for(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

/// 1 系列を測って `ROW4151` を 1 行出力する。`gap` が `Some` なら各呼び出しの前にスピン待ちする。
/// フック行は診断パッチが差し替える（base 腕では空）。
fn run_series(
    arm: &str,
    series: &str,
    rayon_threads: usize,
    fallback_hash: u64,
    out: Option<&RefCell<Vec<f32>>>,
    gap: Option<Duration>,
    f: &mut dyn FnMut(),
) {
    for _ in 0..N_WARMUP {
        f();
    }
    // DIAG_RESET_HOOK（診断パッチが warmup 後の積算リセットを差し込む）
    let mut samples = Vec::with_capacity(N_MEASURE);
    for i in 0..N_MEASURE {
        if let Some(g) = gap {
            spin_for(g);
        }
        if let (true, Some(o)) = (i + 1 == N_MEASURE, out) {
            // 最後の 1 回だけ C をゼロへ戻す（計時の外）。出力をゼロ初期化 C 起点の値にそろえる。
            o.borrow_mut().fill(0.0);
        }
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos());
    }
    // 計時有効のまま走った最後の 1 回の実出力（GEMM 出力のない系列は起動直後の値）。
    let bits_hash = out.map_or(fallback_hash, |o| fnv1a_bits(&o.borrow()));
    // 診断パッチがこの行を内訳文字列の生成へ置き換える。base 腕では空。
    let diag_suffix = String::new(); // DIAG_READ_HOOK
    samples.sort_unstable();
    println!(
        "ROW4151 arm={arm} series={series} m={DIM} n={DIM} k={DIM} rayon_threads={rayon_threads} \
calls={N_MEASURE} median_ns={} mean_ns={:.1} bits_hash=0x{bits_hash:016x}{diag_suffix}",
        median_sorted(&samples),
        mean(&samples)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2
        || !matches!(
            args[1].as_str(),
            "off-base" | "on-base" | "off-timed" | "on-timed"
        )
    {
        eprintln!("usage: cpu_gemm_n256_fixed_cost_4151 <off-base|on-base|off-timed|on-timed>");
        std::process::exit(2);
    }
    let arm = args[1].as_str();

    // 起動直後の最初の 1 回（グローバルプール生成を含む）。計測窓の外の量なので記録のみ。
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(DIM * DIM);
    let b = rng.fill_vec(DIM * DIM);
    let mut c0 = vec![0.0f32; DIM * DIM];
    let t_cold = Instant::now();
    gemm_blis_parallel(&a, &b, &mut c0, DIM, DIM, DIM).expect("gemm");
    let cold_ns = t_cold.elapsed().as_nanos();
    let bits_hash = fnv1a_bits(&c0);

    let rayon_threads = rayon::current_num_threads();
    let (tick, pair) = calibrate_timer();
    let sme = sme_report();
    println!(
        "ENV4151 arm={arm} rayon_threads={rayon_threads} sme_os_flag={} sme_svl_bytes={} \
sme_kernel_enabled={} thread_limit_effective={} thread_limit_env_override={} \
small_shape_cap_eligible={} timer_tick_ns={tick} timer_pair_ns={pair:.2} gap_us={GAP_US}",
        sme.os_flag,
        sme.svl_bytes.unwrap_or(0),
        sme.kernel_enabled,
        thread_limit_report().effective,
        thread_limit_report().env_override,
        small_shape_cap_report().eligible,
    );
    println!(
        "ROW4151 arm={arm} series=cold_first_call m={DIM} n={DIM} k={DIM} rayon_threads={rayon_threads} \
calls=1 median_ns={cold_ns} mean_ns={cold_ns}.0 bits_hash=0x{bits_hash:016x}"
    );

    let c = RefCell::new(vec![0.0f32; DIM * DIM]);
    run_series(
        arm,
        "hot",
        rayon_threads,
        bits_hash,
        Some(&c),
        None,
        &mut || {
            gemm_blis_parallel(&a, &b, &mut c.borrow_mut(), DIM, DIM, DIM).expect("gemm");
        },
    );
    run_series(
        arm,
        "gap",
        rayon_threads,
        bits_hash,
        Some(&c),
        Some(Duration::from_micros(GAP_US)),
        &mut || {
            gemm_blis_parallel(&a, &b, &mut c.borrow_mut(), DIM, DIM, DIM).expect("gemm");
        },
    );
    run_series(
        arm,
        "serial_t1",
        rayon_threads,
        bits_hash,
        Some(&c),
        None,
        &mut || {
            gemm_blis(&a, &b, &mut c.borrow_mut(), DIM, DIM, DIM).expect("gemm");
        },
    );

    // fork-join の床: 本番と同じ job 数の「ほぼ空」の job を `par_iter_mut().try_for_each` で配る。
    let jobs_n = JOBS_PER_WORKER * rayon_threads;
    let mut slots = vec![0u64; jobs_n];
    run_series(
        arm,
        "forkjoin_floor_hot",
        rayon_threads,
        bits_hash,
        None,
        None,
        &mut || forkjoin_once(&mut slots),
    );
    run_series(
        arm,
        "forkjoin_floor_gap",
        rayon_threads,
        bits_hash,
        None,
        Some(Duration::from_micros(GAP_US)),
        &mut || forkjoin_once(&mut slots),
    );
    println!("END4151 arm={arm} forkjoin_jobs={jobs_n}");
}

fn forkjoin_once(slots: &mut [u64]) {
    let r: Result<(), ()> = slots.par_iter_mut().try_for_each(|s| {
        *s = std::hint::black_box(*s).wrapping_add(1);
        Ok(())
    });
    r.expect("forkjoin");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_is_order_and_bit_sensitive() {
        assert_ne!(fnv1a_bits(&[1.0, 2.0]), fnv1a_bits(&[2.0, 1.0]));
        assert_ne!(fnv1a_bits(&[0.0]), fnv1a_bits(&[-0.0]));
        assert_eq!(fnv1a_bits(&[]), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn median_and_mean() {
        assert_eq!(median_sorted(&[1, 2, 9]), 2);
        assert!((mean(&[1, 2, 6]) - 3.0).abs() < 1e-12);
    }
}
