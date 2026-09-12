//! イシュー #1575 Phase 0（M4 Max・定数確定スイープ）計測バイナリ。
//!
//! `docs/perf/cpu-gemm-small-shape-thread-cap.md`・イシュー #1575 コメント
//! （事前登録規則）に定めた形状・腕を、単一プロセス起動ごとに 1 腕ずつ
//! 計測する（`RAYON_NUM_THREADS`〈プロセス全体〉腕はプロセス起動時に固定
//! されるため、複数腕を 1 プロセス内で切り替えられない。オーケストレー
//! ション側〈`docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/
//! orchestrate_m4max.sh`〉が本バイナリをプロセスごとに再起動して腕を
//! 切り替える設計とする。`examples/gemm_bench.rs` と同じ理由で
//! `examples/` に置く: `dev-dependencies`〈`bench-harness`〉を使いつつ
//! 通常の `cargo test`／CI では実行されず self-hosted runner を占有しない）。
//!
//! ## 使い方
//!
//! `cargo run --release -p fandhe-ai-backend-cpu --example small_shape_cap_sweep -- <arm>`
//!
//! `<arm>`:
//! - `global` — 現行プール（`RAYON_NUM_THREADS` 環境変数に従う。既定 16）
//!   でそのまま `gemm_blis_parallel` を呼ぶ（off 腕・参照腕を兼ねる。
//!   オーケストレーション側が `RAYON_NUM_THREADS=4` 等を設定して起動すれば
//!   「プロセス全体 RAYON_NUM_THREADS=N」腕になる）
//! - `dedicated:<N>` — 専用の `rayon::ThreadPool`（`N` スレッド）を構築し
//!   `pool.install(|| gemm_blis_parallel(...))` で実行する（本機構
//!   `run_capped` が有効時に選ぶ経路そのものと同一挙動の計測用複製）
//!
//! 全対象形状（学習 5 形状＋交差確認用正方 128/256/512）を 1 回の起動で
//! 順に計測し、`arm=<arm> shape=<label> m=.. n=.. k=.. median_secs=..
//! checksum=<f32 全要素和の bit 表現 16 進>` を 1 行ずつ標準出力へ書く
//! （`checksum` は全腕・全 run で bit 完全一致するはず。本機構は並列度
//! のみを変え GEMM 本体を変えないため。`aggregate.py` が一致検証に使う）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cpu::gemm_blis_parallel;

const SEED: u64 = 0xC0FFEE;

/// 事前登録形状（イシュー #1575 コメント「Phase 0」節）。
/// `(ラベル, m, n, k)`。
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("train_64x256x784_nn", 64, 256, 784),
    ("train_64x10x256_nn", 64, 10, 256),
    ("train_784x256x64_tn_shape", 784, 256, 64),
    ("train_64x256x10_nt_shape", 64, 256, 10),
    ("train_256x10x64_tn_shape", 256, 10, 64),
    ("square_128", 128, 128, 128),
    ("square_256", 256, 256, 256),
    ("square_512", 512, 512, 512),
];

fn checksum_bits(c: &[f32]) -> u64 {
    // f64 逐次和のビット表現を 16 進で出す（`device-checksum` feature と
    // 同種の「全要素和」だが、本バイナリは診断用のためホスト f64 逐次和
    // で十分。bit 完全一致検証が目的であり性能計測対象ではない）。
    let sum: f64 = c.iter().map(|&v| v as f64).sum();
    sum.to_bits()
}

fn run_shape(arm_label: &str, label: &str, m: usize, n: usize, k: usize, dedicated: Option<usize>) {
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(m * k);
    let b = rng.fill_vec(k * n);
    let mut c = vec![0.0f32; m * n];

    let config = MeasurementConfig::default();

    // `bench_run` のクロージャは GEMM 本体（`gemm_blis_parallel`）のみを計時
    // 対象とする。checksum（f64 逐次和）の計算をクロージャ内に含めると、
    // 全 run の計時に checksum 計算コストが混入し、幾何平均によるスレッド数
    // 選択（Phase 0 事前登録規則）の根拠が歪む（codex-review 指摘。イシュー
    // #1575）。checksum は bit 完全一致検証専用の診断値であり性能計測対象で
    // はないため、`bench_run` 完了後に最終状態の `c` から 1 回だけ計算する。
    let measurement = match dedicated {
        Some(threads) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build dedicated pool for Phase 0 sweep");
            bench_run(&config, || {
                pool.install(|| {
                    gemm_blis_parallel(&a, &b, &mut c, m, n, k)
                        .expect("gemm_blis_parallel (dedicated pool)");
                });
            })
        }
        None => bench_run(&config, || {
            gemm_blis_parallel(&a, &b, &mut c, m, n, k).expect("gemm_blis_parallel (global pool)");
        }),
    }
    .expect("MeasurementConfig::default は下限を満たすため失敗しない");

    let checksum = checksum_bits(&c);

    println!(
        "arm={arm_label} shape={label} m={m} n={n} k={k} median_secs={:.9} checksum=0x{checksum:016x}",
        measurement.median_secs
    );
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: small_shape_cap_sweep <global|dedicated:N>");
        std::process::exit(2);
    });

    let (arm_label, dedicated) = if arg == "global" {
        (arg.clone(), None)
    } else if let Some(n_str) = arg.strip_prefix("dedicated:") {
        let n: usize = n_str.parse().unwrap_or_else(|_| {
            eprintln!("invalid dedicated thread count: {n_str}");
            std::process::exit(2);
        });
        (arg.clone(), Some(n))
    } else {
        eprintln!("usage: small_shape_cap_sweep <global|dedicated:N>");
        std::process::exit(2);
    };

    // rayon 環境変数の実効値を記録に残す（`RAYON_NUM_THREADS` 未設定なら
    // rayon 既定＝物理コア数）。
    eprintln!(
        "rayon_current_num_threads={} RAYON_NUM_THREADS={:?}",
        rayon::current_num_threads(),
        std::env::var("RAYON_NUM_THREADS").ok()
    );

    for (label, m, n, k) in SHAPES {
        run_shape(&arm_label, label, *m, *n, *k, dedicated);
    }
}
