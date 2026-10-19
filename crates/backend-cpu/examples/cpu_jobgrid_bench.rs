//! イシュー #3177 計測バイナリ: MLP（784→256→10・batch 64）の 6 形状を計時し、
//! 単スレッドの実効 GFLOPS と job 分割 A/B（`jpw1`／`cap4`／`cap8` 腕）の比較に使う。
//!
//! `cfg(test)` を含まない通常ビルドのバイナリなので、`gemm_blis::jobgrid_probe`
//! のフックはコンパイルされず計時に混入しない（計数は `cargo test --lib` 側）。
//! 腕の違い（定数差分）は計測専用 worktree に当てた patch でビルドし分けるため、
//! 本バイナリは腕ラベルを出力に記録するだけで挙動は変えない。
//!
//! 使い方: `cpu_jobgrid_bench <arm> <threads|serial_ref>`
//! - `threads`: 公開入口 `gemm_blis_parallel{,_nt,_tn}`（`RAYON_NUM_THREADS=1` で単スレッド）
//! - `serial_ref`: 直列 `gemm_blis`（Nn の形状のみ。staging・確保の固定費の切り出し用参照）
//!
//! 出力: `arm= series= shape= trans= m= n= k= rayon_threads= median_secs= gflops= checksum=0x..`
//! 事前登録規則は `docs/perf/logs/cpu-jobgrid-3177/RULE.txt`。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cpu::{
    gemm_blis, gemm_blis_parallel, gemm_blis_parallel_nt, gemm_blis_parallel_tn,
};

const SEED: u64 = 0xC0FFEE;

/// `(ラベル, 転置, m, n, k)`。
const SHAPES: &[(&str, &str, usize, usize, usize)] = &[
    ("l1_fwd", "nn", 64, 256, 784),
    ("l1_dw", "tn", 784, 256, 64),
    ("l1_dx", "nt", 64, 784, 256),
    ("l2_fwd", "nn", 64, 10, 256),
    ("l2_dw", "tn", 256, 10, 64),
    ("l2_dx", "nt", 64, 256, 10),
    ("ref_square_512", "nn", 512, 512, 512),
];

fn checksum_bits(c: &[f32]) -> u64 {
    c.iter().map(|&v| v as f64).sum::<f64>().to_bits()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || !matches!(args[2].as_str(), "threads" | "serial_ref") {
        eprintln!("usage: cpu_jobgrid_bench <arm> <threads|serial_ref>");
        std::process::exit(2);
    }
    let (arm, series) = (args[1].as_str(), args[2].as_str());
    let rayon_threads = rayon::current_num_threads();

    for &(label, trans, m, n, k) in SHAPES {
        if series == "serial_ref" && trans != "nn" {
            continue;
        }
        let mut rng = Xorshift64Star::new(SEED);
        // 転置格納オペランドも要素数は同じなので同一の入力列で足りる。
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);
        let mut c = vec![0.0f32; m * n];
        let config = MeasurementConfig::default();
        // checksum は計時クロージャの外で 1 回だけ計算する（#1575 の指摘を踏襲）。
        let measurement = bench_run(&config, || {
            let r = match (series, trans) {
                ("serial_ref", _) => gemm_blis(&a, &b, &mut c, m, n, k),
                (_, "nn") => gemm_blis_parallel(&a, &b, &mut c, m, n, k),
                (_, "nt") => gemm_blis_parallel_nt(&a, &b, &mut c, m, n, k),
                _ => gemm_blis_parallel_tn(&a, &b, &mut c, m, n, k),
            };
            r.expect("gemm");
        })
        .expect("MeasurementConfig::default は下限を満たすため失敗しない");
        let gflops = 2.0 * (m * n * k) as f64 / measurement.median_secs / 1e9;
        println!(
            "arm={arm} series={series} shape={label} trans={trans} m={m} n={n} k={k} \
rayon_threads={rayon_threads} median_secs={:.9} gflops={gflops:.3} checksum=0x{:016x}",
            measurement.median_secs,
            checksum_bits(&c)
        );
    }
}
