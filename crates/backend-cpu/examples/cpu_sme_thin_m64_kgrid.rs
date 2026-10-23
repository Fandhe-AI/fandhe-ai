//! イシュー #3236 計測バイナリ: m=64 専用 SME 薄型カーネル（#3235）の k 下限を決めるための K 格子。
//!
//! `(m, n) = (64, 256)` の Nn を `k ∈ {10, 16, 32, 64, 128, 256, 784}` で計時する。
//! 腕（`SME_THIN_M64_ENABLED` の false／true）は計測専用 worktree に当てた patch でビルドし分けるため、
//! 本バイナリは腕ラベルを出力に記録するだけで挙動は変えない（`cpu_jobgrid_bench` と同じ作法）。
//! 呼ぶのは本番の入口 `gemm_blis_parallel` だけで、`unsafe`・新しい依存・`cfg(test)` を使わない。
//! 事前登録規則（ADOPT-k の K 決定）は `docs/perf/logs/sme-thin-m64-ab-3236/RULE.txt`、
//! 設計の正は `docs/perf/cpu-gemm-sme-thin-m64-decision.md`。
//!
//! 使い方: `cpu_sme_thin_m64_kgrid <arm>`
//! 出力: `arm= shape= m= n= k= rayon_threads= median_secs= gflops= checksum=0x..`

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cpu::gemm_blis_parallel;

const SEED: u64 = 0xC0FFEE;
const M: usize = 64;
const N: usize = 256;
const KS: &[usize] = &[10, 16, 32, 64, 128, 256, 784];

/// 全要素の f64 和の bit 列。計時クロージャの外で 1 回だけ計算する。
fn checksum_bits(c: &[f32]) -> u64 {
    c.iter().map(|&v| v as f64).sum::<f64>().to_bits()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: cpu_sme_thin_m64_kgrid <arm>");
        std::process::exit(2);
    }
    let arm = args[1].as_str();
    let rayon_threads = rayon::current_num_threads();

    for &k in KS {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(M * k);
        let b = rng.fill_vec(k * N);
        let mut c = vec![0.0f32; M * N];
        let config = MeasurementConfig::default();
        let measurement = bench_run(&config, || {
            gemm_blis_parallel(&a, &b, &mut c, M, N, k).expect("gemm");
        })
        .expect("MeasurementConfig::default は下限を満たすため失敗しない");
        let gflops = 2.0 * (M * N * k) as f64 / measurement.median_secs / 1e9;
        println!(
            "arm={arm} shape=kgrid_k{k} m={M} n={N} k={k} rayon_threads={rayon_threads} \
median_secs={:.9} gflops={gflops:.3} checksum=0x{:016x}",
            measurement.median_secs,
            checksum_bits(&c)
        );
    }
}
