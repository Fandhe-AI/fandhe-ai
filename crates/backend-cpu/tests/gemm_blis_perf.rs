//! `gemm_blis_parallel`（TASK-1.6f・#184）と `gemm_parallel`（TASK-1.6a）の
//! 性能比較ハーネス。`bench_harness::protocol::run`（warmup 20 回以上・
//! 計測 20 回以上・中央値／Q1/Q3 記録。TASK-8.1 準拠）を用いる。
//!
//! `#[ignore]` として通常 CI から除外する（`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値を採用し」の趣旨と、TASK-8.1 の 20/20
//! プロトコルとの整合は `bench-harness` クレートドキュメント参照。本
//! ハーネスは受け入れ条件「TASK-1.6a 比の相対改善を実測・記録する」を
//! 満たすための性能計測であり、pass/fail の自動判定は行わない
//! （性能下限〈REQ-8 の 20%〉達成の可否判定自体は #24 のスコープ）。
//!
//! 実行例（AVX2+FMA を有効化してビルド）:
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p fandhe-ai-backend-cpu \
//!     --release -- --ignored gemm_blis_perf
//! ```

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};
use fandhe_ai_backend_cpu::{gemm_blis_parallel, gemm_parallel};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// M=N=K の正方形状で `gemm_parallel`（TASK-1.6a）・`gemm_blis_parallel`
/// （TASK-1.6f）を計測し、中央値・改善比を標準出力へ記録する。
fn measure_square(size: usize) {
    let m = size;
    let n = size;
    let k = size;
    let a = random_matrix(1000 + size as u64, m * k);
    let b = random_matrix(2000 + size as u64, k * n);

    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

    let mut c_blocked = vec![0.0f32; m * n];
    let blocked = run(&config, || {
        c_blocked.iter_mut().for_each(|v| *v = 0.0);
        gemm_parallel(&a, &b, &mut c_blocked, m, n, k).unwrap();
    })
    .expect("gemm_parallel の計測に失敗");

    let mut c_blis = vec![0.0f32; m * n];
    let blis = run(&config, || {
        c_blis.iter_mut().for_each(|v| *v = 0.0);
        gemm_blis_parallel(&a, &b, &mut c_blis, m, n, k).unwrap();
    })
    .expect("gemm_blis_parallel の計測に失敗");

    let speedup = blocked.median_secs / blis.median_secs;

    println!(
        "M=N=K={size}: gemm_parallel median={:.6}s (q1={:.6}, q3={:.6}) / \
         gemm_blis_parallel median={:.6}s (q1={:.6}, q3={:.6}) / speedup={speedup:.3}x",
        blocked.median_secs,
        blocked.q1_secs,
        blocked.q3_secs,
        blis.median_secs,
        blis.q1_secs,
        blis.q3_secs,
    );

    // 数値も一致することを併せて確認する（性能計測が誤った実装を比較して
    // いないことの保険。bit 完全一致契約は `tests/gemm_blis_parity.rs` で
    // 別途網羅的に検証済みのため、ここでは計測に使った具体的な入力に
    // 限定した簡易チェックに留める）。
    assert_eq!(
        c_blocked, c_blis,
        "計測対象の 2 実装が bit 一致しない（M=N=K={size}）"
    );
}

#[test]
#[ignore = "性能計測ハーネス。--release かつ RUSTFLAGS で AVX2+FMA を有効化して個別実行する想定"]
fn gemm_blis_perf_square_512_1024_2048() {
    for size in [512usize, 1024, 2048] {
        measure_square(size);
    }
}

/// イシュー #488（A-8）: 現行本番演算経路 `gemm_blis_parallel` 単体の
/// 対 PyTorch ベースライン再計測ハーネス。
///
/// `docs/perf/gemm-optimization-baseline.md` §1 CPU 行が指摘する通り、
/// REQ-8 実測比率 5.3%（`docs/performance-targets.md` §2）は PoC-v2-1
/// 旧経路（SIMD 未適用）の値であり、現行の本番演算経路（aarch64 では
/// `dispatch_region` が無条件に `NeonKernel` を選択する。
/// `crates/backend-cpu/src/gemm_blis/mod.rs:360-365`）の対 PyTorch 比は
/// 別途確定が必要というのが本イシューのスコープ。既存
/// `gemm_blis_perf_square_512_1024_2048`（TASK-1.6a 比の相対改善計測）
/// とは目的が異なるため独立関数として追加する（`gemm_parallel` との
/// 比較・512/1024/2048 固定という既存関数の形を変えない）。
///
/// 対 PyTorch 比較は本テストの計測値（Rust 側）と、同一プロトコル
/// （warmup 20・iters 20）で実行した PyTorch 側スクリプト
/// （`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/gemm_bench_torch_cpu.py`。
/// 読み取り実行のみ・submodule は編集しない）の出力を、
/// `docs/perf/cpu-gemm-baseline-remeasurement.md` 側で人手で突合する
/// （本テストは Rust 側単体の計測のみを担い、PyTorch 実行はテストの
/// 責務外）。
///
/// 出力は 1 行形式（`kernel=gemm_blis_parallel size=<n>
/// median_tflops=<v> q1_tflops=<v> q3_tflops=<v> median_secs=<v>`）とし、
/// PyTorch スクリプト出力との突合を容易にする。TFLOPS = 2·N³ /
/// median_secs / 1e12（正方形状 GEMM の浮動小数点演算数）。
///
/// 判定・Phase E（#564 等）改善率の分母は 2048/4096 を主対象とする
/// （512・1024 は起動オーバーヘッド支配・中間参考値。
/// `docs/perf/gemm-optimization-baseline.md` §1 表と同方針）。
#[test]
#[ignore = "性能計測ハーネス。--release で個別実行する想定（M4 Max 実機で 5 回実行し中央値の中央値を採る運用は呼び出し側の責務）"]
fn gemm_blis_baseline_pytorch_square_512_to_4096() {
    let arch = std::env::consts::ARCH;
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    println!("arch={arch} logical_cores={logical_cores}");

    for size in [512usize, 1024, 2048, 4096] {
        let m = size;
        let n = size;
        let k = size;
        let a = random_matrix(3000 + size as u64, m * k);
        let b = random_matrix(4000 + size as u64, k * n);

        let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

        // `gemm_blis_parallel`（`dispatch_region` → `gemm_blis_region`）は
        // C タイルの現在値をロードしてから `kernel.run` で加算する
        // 「C += A@B」契約であり、出力バッファ全体を上書きする契約では
        // ない（`crates/backend-cpu/src/gemm_blis/mod.rs` の C タイル
        // load/store 部を参照）。そのため計測クロージャ内では `c` を
        // ゼロクリアしない（warmup・計測の反復を経るたびに前回の
        // 結果へ積算されていくだけで、計測対象はあくまで
        // `gemm_blis_parallel` 呼び出し自体の所要時間であり `c` の
        // 最終値は計測目的では使わない）。ゼロクリアの O(N²) メモリ
        // 書き込み時間が `median_secs`・そこから算出する TFLOPS
        // （Phase E の対 PyTorch 比率の分子）へ混入するのを避ける
        // （codex-review 指摘・PR #650）。
        let mut c = vec![0.0f32; m * n];
        let measurement = run(&config, || {
            gemm_blis_parallel(&a, &b, &mut c, m, n, k).unwrap();
        })
        .expect("gemm_blis_parallel の計測に失敗");

        let flops = 2.0 * (size as f64).powi(3);
        let median_tflops = flops / measurement.median_secs / 1e12;
        let q1_tflops = flops / measurement.q3_secs / 1e12; // 所要時間が短いほど TFLOPS は高いため q3_secs が q1_tflops に対応する
        let q3_tflops = flops / measurement.q1_secs / 1e12;

        // 計測済み TFLOPS は正当性確認より先に出力する。以降の
        // `assert_eq!` が失敗して panic しても、Phase E の分母となる
        // この形状の計測値が出力されないまま失われることはない
        // （cursor[bot]・github-actions 指摘・PR #650）。
        println!(
            "kernel=gemm_blis_parallel size={size} median_tflops={median_tflops:.6} \
             q1_tflops={q1_tflops:.6} q3_tflops={q3_tflops:.6} median_secs={:.6}",
            measurement.median_secs,
        );

        // 数値正当性は計測区間の外側で独立して確認する（計測ループ中の
        // `c` は複数回の `gemm_blis_parallel` 呼び出しで積算された値の
        // ため、そのままでは参照実装と比較できない）。`c` を明示的に
        // ゼロクリアしてから単回のみ再実行し、その結果を
        // `gemm_parallel` と突き合わせる（bit 完全一致契約の網羅検証は
        // `tests/gemm_blis_parity.rs` が別途担うため、ここでは計測対象
        // 取り違えの検出に限定する）。
        c.iter_mut().for_each(|v| *v = 0.0);
        gemm_blis_parallel(&a, &b, &mut c, m, n, k).unwrap();
        let mut c_ref = vec![0.0f32; m * n];
        gemm_parallel(&a, &b, &mut c_ref, m, n, k).unwrap();
        assert_eq!(
            c, c_ref,
            "計測対象 gemm_blis_parallel が参照実装と bit 一致しない（M=N=K={size}）"
        );
    }
}
