//! CPU GEMM 計測バイナリ（#24・TASK-1.6d）。
//!
//! `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/rust/src/bin/gemm_bench.rs`
//! の productize 版。PoC バイナリとの違いは、計測コアを自前の
//! `median_q1_q3` から `bench-harness::protocol::run`（warmup/計測とも
//! 20 回以上・中央値/Q1/Q3 集計。TASK-8.1）へ置き換えた点、および
//! `backend_cpu::gemm::BlockSizes`／`gemm_parallel_tuned` の
//! オーバーサブスクリプション係数を実測スイープできるようにした点。
//!
//! `examples/` に置くのは、`dev-dependencies`（`bench-harness`）を
//! 利用しつつ、通常の `cargo test`／CI では実行されず、ビルド検証
//! （`cargo build --workspace --all-targets`）のみが CI で走るようにする
//! ためである（self-hosted runner をベンチ実行で占有しない。`ci.md`）。
//!
//! ## 使い方
//!
//! - `cargo run --release -p backend-cpu --example gemm_bench` — 既定サイズ
//!   （512/2048/4096）で naive/blocked/parallel を計測し、改善比・並列効率を表示する
//!   （naive@4096 は所要時間過大のため計測せず、blocked@4096 を分母に使う。
//!   本ファイル内コメント参照）
//! - `cargo run --release -p backend-cpu --example gemm_bench -- sweep` —
//!   M=N=K=2048 での MC/KC/NC 座標降下法スイープと、512/2048 での
//!   オーバーサブスクリプション係数（1/2/4）スイープを実行する
//!
//! いずれも `MeasurementConfig::default()`（warmup 20・iters 20）を使う。

use backend_cpu::gemm::{BlockSizes, gemm_blocked, gemm_naive, gemm_parallel_tuned};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{Measurement, MeasurementConfig, run as bench_run};
use std::fmt::Write as _;

/// 決定的シード（PoC-v2-1・PoC-v2-5 と同一値。`rng.rs` の xorshift64* 系列を
/// 揃えることで、本計測と過去 PoC 実測が同じ入力分布に基づくことを保証する）。
const SEED: u64 = 0xC0FFEE;

/// `/proc/loadavg` の 1 分平均値を読む（Linux 限定。読み取り失敗時は計測を
/// 止めずに `None` を返す）。並列実行される他エージェントのビルド負荷が
/// 計測に混入していないかを記録に残すための補助情報（advisor 指摘。
/// `docs/perf/cpu-gemm-rayon-tuning.md` に転記する）。
fn loadavg_1min() -> Option<f64> {
    let content = std::fs::read_to_string("/proc/loadavg").ok()?;
    content.split_whitespace().next()?.parse().ok()
}

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

fn measure(
    name: &str,
    size: usize,
    config: &MeasurementConfig,
    kernel: impl FnMut(&[f32], &[f32], &mut [f32], usize, usize, usize),
) -> Measurement {
    let mut kernel = kernel;
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(size * size);
    let b = rng.fill_vec(size * size);
    let mut c = vec![0.0f32; size * size];

    let measurement = bench_run(config, || {
        c.iter_mut().for_each(|v| *v = 0.0);
        kernel(&a, &b, &mut c, size, size, size);
    })
    .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

    // `Measurement::q1_secs`／`q3_secs` は所要時間（秒）の四分位であり、
    // TFLOPS は時間の逆数のため大小関係が反転する（時間が短い q1_secs
    // ほど TFLOPS は高い）。Cursor Bugbot 指摘 #231: 変換前の変数名
    // （q1_secs/q3_secs）のまま `tflops_q1`/`tflops_q3` として表示すると
    // 「q1 は低い方の四分位」という一般的な期待に反し、記録済みの表
    // （`docs/perf/cpu-gemm-rayon-tuning.md`）でも逆転して見えていた。
    // ここでは表示ラベルどおり tflops_q1 ≦ tflops_q3（スループットの
    // 四分位として昇順）になるよう、変換後に値を入れ替えて渡す。
    let tflops_median = tflops(size, measurement.median_secs);
    let tflops_q1 = tflops(size, measurement.q3_secs);
    let tflops_q3 = tflops(size, measurement.q1_secs);
    println!(
        "kernel={name} size={size} warmup={} iters={} median_tflops={:.4} tflops_q1={:.4} tflops_q3={:.4} loadavg={:?}",
        measurement.warmup,
        measurement.iters,
        tflops_median,
        tflops_q1,
        tflops_q3,
        loadavg_1min(),
    );
    measurement
}

/// 既定モード: naive/blocked/parallel を同一 `MeasurementConfig` で計測し
/// 改善比・並列効率を表示する（受け入れ条件「PoC-v2-1 比の性能改善比が
/// 再現される」の直接確認）。
///
/// naive@2048 以上は 1 回あたり数秒〜十数秒かかり 40 回（warmup20+iters20）
/// では数分〜十数分に達する（本環境は QEMU Virtual CPU であり PoC-v2-1
/// 実測環境の Apple M4 Max より実測 TFLOPS が 1 桁以上小さいことが 512 での
/// 事前計測で判明した。当初計画は 4096 のみ naive を省く想定だったが、
/// 本環境の実測に基づき閾値を 2048 に前倒しした）。プロトコル遵守
/// （20/20 必須。TASK-8.1）を保ったまま実行時間を抑えるため、2048 以上では
/// naive を計測せず blocked を単一スレッド基準（分母）として使う
/// （PoC README「計測結果」節が「blocked は naive とほぼ同等、2048/4096
/// ではむしろ naive をやや下回る」と記録しており、blocked を分母にする
/// ことは改善比を過大評価しない安全側の選択である）。
fn run_default(sizes: &[usize]) {
    let config = MeasurementConfig::default();
    println!("threads={}", rayon::current_num_threads());
    println!("loadavg_before={:?}", loadavg_1min());

    for &size in sizes {
        let blocked = measure("blocked", size, &config, |a, b, c, m, n, k| {
            gemm_blocked(a, b, c, m, n, k).unwrap();
        });
        // baseline_name は実際に分母として使った median_secs の由来
        // カーネル名と一致させる（Cursor Bugbot 指摘 #231: 従来は
        // size<2048 のとき常に "naive" 固定だったが、`.min()` で選ぶのは
        // naive・blocked のうち実測が速かった方であり、512 の実測では
        // blocked（0.0012 TFLOPS）が naive（0.0011 TFLOPS）を上回り実際の
        // 分母だった。ラベルと実値の不一致を防ぐため、比較結果に応じて
        // 明示的にラベルを選ぶ）。
        let (baseline_name, baseline_median) = if size >= 2048 {
            ("blocked", blocked.median_secs)
        } else {
            let naive = measure("naive", size, &config, |a, b, c, m, n, k| {
                gemm_naive(a, b, c, m, n, k).unwrap();
            });
            if naive.median_secs <= blocked.median_secs {
                ("naive", naive.median_secs)
            } else {
                ("blocked", blocked.median_secs)
            }
        };
        let parallel = measure("parallel", size, &config, |a, b, c, m, n, k| {
            gemm_parallel_tuned(a, b, c, m, n, k, BlockSizes::poc_v2_1_default(), 1).unwrap();
        });

        let ratio = baseline_median / parallel.median_secs;
        let threads = rayon::current_num_threads().max(1) as f64;
        let efficiency = ratio / threads;
        println!(
            "size={size} improvement_ratio(parallel/{baseline_name})={ratio:.3}x parallel_efficiency={efficiency:.3} (poc_range=0.37-0.53)"
        );
    }
    println!("loadavg_after={:?}", loadavg_1min());
}

/// `sweep` モード: MC/KC/NC の座標降下法スイープ（M=N=K=2048）と、
/// オーバーサブスクリプション係数（1/2/4）スイープ（512・2048）。
///
/// 27（=3^3）通りの全数探索は 1 点あたり計測 20/20 回で数十秒かかり
/// 非現実的なため、「MC → KC → NC の順に、直前で選んだ最良値を固定して
/// 次のパラメータを振る」座標降下法（9 点）に縮小した（全数探索ではない
/// 旨を記録に明記する）。各段の median_secs 最小の候補を `best` に反映し
/// 次段へ引き継ぐ（Cursor Bugbot 指摘 #231 の修正。「最良」の採否自体は
/// median_secs の単純比較であり、Q1〜Q3 幅を考慮した統計的有意差判定は
/// 行わない。最終的な採否記録は出力された中央値を目視で
/// `docs/perf/cpu-gemm-rayon-tuning.md` へ転記する運用とする）。
fn run_sweep() {
    let config = MeasurementConfig::default();
    println!("threads={}", rayon::current_num_threads());
    println!("loadavg_before={:?}", loadavg_1min());

    let size = 2048usize;
    // 座標降下法の「直前で選んだ最良値を固定」を実際に実装する（Cursor
    // Bugbot 指摘 #231: 従来は `best` を `poc_v2_1_default()` に束縛した
    // まま各段で `{ mc, ..best }` のように定数側のフィールドしか参照して
    // おらず、KC・NC 段が MC 段の最良値ではなく常に既定値の周辺を振る
    // だけになっていた。各段の計測後に `median_secs` 最小の候補で `best`
    // を更新し、次段へ引き継ぐ）。
    let mut best = BlockSizes::poc_v2_1_default();

    let mut best_mc_secs = f64::INFINITY;
    for &mc in &[64usize, 128, 256] {
        let blocks = BlockSizes { mc, ..best };
        let m = measure(
            &format!("sweep_mc={mc}"),
            size,
            &config,
            move |a, b, c, m, n, k| {
                gemm_parallel_tuned(a, b, c, m, n, k, blocks, 1).unwrap();
            },
        );
        if m.median_secs < best_mc_secs {
            best_mc_secs = m.median_secs;
            best.mc = mc;
        }
    }

    let mut best_kc_secs = f64::INFINITY;
    for &kc in &[128usize, 256, 512] {
        let blocks = BlockSizes { kc, ..best };
        let m = measure(
            &format!("sweep_kc={kc}"),
            size,
            &config,
            move |a, b, c, m, n, k| {
                gemm_parallel_tuned(a, b, c, m, n, k, blocks, 1).unwrap();
            },
        );
        if m.median_secs < best_kc_secs {
            best_kc_secs = m.median_secs;
            best.kc = kc;
        }
    }

    let mut best_nc_secs = f64::INFINITY;
    for &nc in &[256usize, 512, 1024] {
        let blocks = BlockSizes { nc, ..best };
        let m = measure(
            &format!("sweep_nc={nc}"),
            size,
            &config,
            move |a, b, c, m, n, k| {
                gemm_parallel_tuned(a, b, c, m, n, k, blocks, 1).unwrap();
            },
        );
        if m.median_secs < best_nc_secs {
            best_nc_secs = m.median_secs;
            best.nc = nc;
        }
    }
    println!("sweep_best mc={} kc={} nc={}", best.mc, best.kc, best.nc);

    for &osize in &[512usize, 2048] {
        for &oversub in &[1usize, 2, 4] {
            measure(
                &format!("sweep_oversub={oversub}"),
                osize,
                &config,
                move |a, b, c, m, n, k| {
                    gemm_parallel_tuned(a, b, c, m, n, k, BlockSizes::poc_v2_1_default(), oversub)
                        .unwrap();
                },
            );
        }
    }

    println!("loadavg_after={:?}", loadavg_1min());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("sweep") {
        run_sweep();
        return;
    }

    let sizes: Vec<usize> = args
        .get(1)
        .map(|s| {
            s.split(',')
                .map(|part| {
                    part.parse().unwrap_or_else(|e| {
                        let mut msg = String::new();
                        let _ = write!(msg, "不正なサイズ指定 {part:?}: {e}");
                        panic!("{msg}");
                    })
                })
                .collect()
        })
        .unwrap_or_else(|| vec![512, 2048, 4096]);

    run_default(&sizes);
}
