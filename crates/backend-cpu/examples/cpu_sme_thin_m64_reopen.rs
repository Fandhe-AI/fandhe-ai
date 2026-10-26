//! イシュー #3615 計測バイナリ: #3236 の A/B で undetermined だった `l2_dx`（64x256x10 Nt）を、
//! 反復数を増やして測り直すための計測専用 example（RULE-reopen-1。`docs/perf/logs/sme-thin-m64-ab-3236/RULE.txt`）。
//!
//! `cpu_jobgrid_bench`（#3177 の RULE が依存）と `cpu_sme_thin_m64_kgrid`（#3236 の記録が依存）は変えず、
//! 本ファイルで warmup／iters だけを引き上げる（`MeasurementConfig::new(WARMUP, ITERS)`）。
//! 腕（`SME_THIN_M64_ENABLED` の false／true）は計測専用 worktree に当てた patch でビルドし分けるため、
//! 本バイナリは腕ラベルを出力に記録するだけで挙動は変えない。
//! 呼ぶのは本番の入口 `gemm_blis_parallel{,_nt}` だけで、`unsafe`・新しい依存・`cfg(test)` を使わない。
//! 集計は `aggregate_reopen.py` が出力の `warmup=`／`iters=` を RULE 値と突き合わせて fail-closed で検査する。
//!
//! 使い方: `cpu_sme_thin_m64_reopen <before|after> <gemm|kgrid>`
//! - `gemm`: l2_dx（判定対象・必須）と、比較のための l1_fwd・l1_dx
//! - `kgrid`: `(64, 256, k)` Nn を `k ∈ {10, 16, 32, 64, 128, 256, 784}`
//!
//! 出力（1 行 1 セル）: `arm= series=threads shape= trans= m= n= k= rayon_threads= warmup= iters= median_secs= gflops= checksum=0x..`
//! （kgrid は `series`・`trans` を持たない）

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cpu::{gemm_blis_parallel, gemm_blis_parallel_nt};

const SEED: u64 = 0xC0FFEE;
/// RULE-reopen-1 の値。`aggregate_reopen.py` の WARMUP／ITERS と一致させる。
const WARMUP: usize = 200;
const ITERS: usize = 2000;

/// `(ラベル, 転置, m, n, k)`。`cpu_jobgrid_bench` の到達 3 形状と同一。
const GEMM_SHAPES: &[(&str, &str, usize, usize, usize)] = &[
    ("l1_fwd", "nn", 64, 256, 784),
    ("l1_dx", "nt", 64, 784, 256),
    ("l2_dx", "nt", 64, 256, 10),
];
const KGRID_KS: &[usize] = &[10, 16, 32, 64, 128, 256, 784];

/// 全要素の f64 和の bit 列。計時クロージャの外で 1 回だけ計算する。
fn checksum_bits(c: &[f32]) -> u64 {
    c.iter().map(|&v| v as f64).sum::<f64>().to_bits()
}

/// 1 セルを計時し、`(median_secs, checksum)` を返す。失敗は文字列で返し main が非 0 終了する。
fn measure(trans: &str, m: usize, n: usize, k: usize) -> Result<(f64, u64), String> {
    let mut rng = Xorshift64Star::new(SEED);
    // 転置格納オペランドも要素数は同じなので同一の入力列で足りる。
    let a = rng.fill_vec(m * k);
    let b = rng.fill_vec(k * n);
    let mut c = vec![0.0f32; m * n];
    let config =
        MeasurementConfig::new(WARMUP, ITERS).map_err(|e| format!("MeasurementConfig: {e:?}"))?;
    let mut err: Option<String> = None;
    let measurement = bench_run(&config, || {
        let r = match trans {
            "nt" => gemm_blis_parallel_nt(&a, &b, &mut c, m, n, k),
            _ => gemm_blis_parallel(&a, &b, &mut c, m, n, k),
        };
        if let Err(e) = r {
            err = Some(format!("gemm: {e:?}"));
        }
    })
    .map_err(|e| format!("bench_run: {e:?}"))?;
    if let Some(e) = err {
        return Err(e);
    }
    Ok((measurement.median_secs, checksum_bits(&c)))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3
        || !matches!(args[1].as_str(), "before" | "after")
        || !matches!(args[2].as_str(), "gemm" | "kgrid")
    {
        eprintln!("usage: cpu_sme_thin_m64_reopen <before|after> <gemm|kgrid>");
        std::process::exit(2);
    }
    let (arm, mode) = (args[1].as_str(), args[2].as_str());
    let rayon_threads = rayon::current_num_threads();

    let cells: Vec<(String, &str, usize, usize, usize)> = if mode == "gemm" {
        GEMM_SHAPES
            .iter()
            .map(|&(l, t, m, n, k)| (l.to_string(), t, m, n, k))
            .collect()
    } else {
        KGRID_KS
            .iter()
            .map(|&k| (format!("kgrid_k{k}"), "nn", 64, 256, k))
            .collect()
    };
    for (label, trans, m, n, k) in cells {
        let (median, ck) = match measure(trans, m, n, k) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {label}: {e}");
                std::process::exit(1);
            }
        };
        let gflops = 2.0 * (m * n * k) as f64 / median / 1e9;
        let head = if mode == "gemm" {
            format!("arm={arm} series=threads shape={label} trans={trans}")
        } else {
            format!("arm={arm} shape={label}")
        };
        println!(
            "{head} m={m} n={n} k={k} rayon_threads={rayon_threads} warmup={WARMUP} iters={ITERS} \
median_secs={median:.9} gflops={gflops:.3} checksum=0x{ck:016x}"
        );
    }
}
