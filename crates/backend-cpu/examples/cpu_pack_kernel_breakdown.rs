//! イシュー #3614 計測バイナリ（base 腕）: MLP（784→256→10・batch 64）の 6 形状について、
//! 公開入口 `gemm_blis_parallel{,_nt,_tn}` の wall 時間と出力の bit ハッシュを記録する。
//!
//! 役割: A pack・B pack・kernel の時間内訳（pack が主因か。NEON 直接カーネル (A) 再検討の
//! 前提。`docs/perf/cpu-gemm-neon-direct-decision.md` §7.1）を測る計測の土台。内訳の区間計時は
//! 本番コードに入れず、計測用の複製ツリーにだけ当てる診断パッチ
//! （`docs/perf/logs/cpu-pack-kernel-breakdown-3614/diag_timing.patch`）が本ファイルへ追加出力を
//! 足す。base 腕（本ファイルそのまま）と timed 腕（パッチ適用）の `bits_hash` 一致で、区間計時が
//! 数値を変えていないことを確かめる。事前登録規則は同ディレクトリの `RULE.txt`。
//!
//! 使い方: `cpu_pack_kernel_breakdown <arm-label>`。スレッド数は `RAYON_NUM_THREADS` の有無で
//! native（機体の既定プール）と単スレッドを切り替え、出力の `rayon_threads=` を正とする。
//!
//! 出力: `arm= mode= shape= trans= m= n= k= rayon_threads= calls= wall_median_ns=
//! wall_mean_ns= bits_hash=0x..`。`bits_hash` はゼロ初期化した C へ 1 回だけ呼んだ結果の
//! 全要素 `f32::to_bits()` を添字順に FNV-1a 64 で畳み込んだ値（計時の外で計算する。C は
//! 累積型のため計時ループ後の C では再現性がない）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{gemm_blis_parallel, gemm_blis_parallel_nt, gemm_blis_parallel_tn};
use std::time::Instant;

const SEED: u64 = 0xC0FFEE;
const N_WARMUP: usize = 50;
const N_MEASURE: usize = 500;

/// `(ラベル, 転置, m, n, k)`。正方 512 は参照で判定には使わない（RULE §3）。
const SHAPES: &[(&str, &str, usize, usize, usize)] = &[
    ("l1_fwd", "nn", 64, 256, 784),
    ("l1_dw", "tn", 784, 256, 64),
    ("l1_dx", "nt", 64, 784, 256),
    ("l2_fwd", "nn", 64, 10, 256),
    ("l2_dw", "tn", 256, 10, 64),
    ("l2_dx", "nt", 64, 256, 10),
    ("ref_square_512", "nn", 512, 512, 512),
];

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

fn run_once(trans: &str, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let r = match trans {
        "nn" => gemm_blis_parallel(a, b, c, m, n, k),
        "nt" => gemm_blis_parallel_nt(a, b, c, m, n, k),
        _ => gemm_blis_parallel_tn(a, b, c, m, n, k),
    };
    r.expect("gemm");
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: cpu_pack_kernel_breakdown <arm-label>");
        std::process::exit(2);
    }
    let arm = args[1].as_str();
    let rayon_threads = rayon::current_num_threads();
    let mode = if rayon_threads == 1 { "t1" } else { "native" };
    let (tick, pair) = calibrate_timer();
    println!("arm={arm} mode={mode} timer_tick_ns={tick} timer_pair_ns={pair:.2}");

    for &(label, trans, m, n, k) in SHAPES {
        let mut rng = Xorshift64Star::new(SEED);
        // 転置格納オペランドも要素数は同じなので同一の入力列で足りる。
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        let mut c0 = vec![0.0f32; m * n];
        run_once(trans, &a, &b, &mut c0, m, n, k);
        let bits_hash = fnv1a_bits(&c0);

        let mut c = vec![0.0f32; m * n];
        for _ in 0..N_WARMUP {
            run_once(trans, &a, &b, &mut c, m, n, k);
        }
        // DIAG_RESET_HOOK（診断パッチが warmup 後の積算リセットを差し込む）
        let mut samples = Vec::with_capacity(N_MEASURE);
        for _ in 0..N_MEASURE {
            let t = Instant::now();
            run_once(trans, &a, &b, &mut c, m, n, k);
            samples.push(t.elapsed().as_nanos());
        }
        // 診断パッチがこの行を内訳文字列の生成へ置き換える。base 腕では空。
        let diag_suffix = String::new(); // DIAG_READ_HOOK
        samples.sort_unstable();
        let median = samples[N_MEASURE / 2];
        let mean = samples.iter().sum::<u128>() as f64 / N_MEASURE as f64;
        println!(
            "arm={arm} mode={mode} shape={label} trans={trans} m={m} n={n} k={k} \
rayon_threads={rayon_threads} calls={N_MEASURE} wall_median_ns={median} wall_mean_ns={mean:.1} \
bits_hash=0x{bits_hash:016x}{diag_suffix}"
        );
    }
}
