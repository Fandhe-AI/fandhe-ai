//! `CpuBackendOps::gemm_bias_act`（融合カーネル。TASK-12.1f・#203）と
//! `fandhe_ai_tensor_core::BackendOps::gemm_bias_act` の**デフォルト実装**（非融合
//! `gemm` → `add` → `relu` の 3 段合成。利用者が現在得られる経路）の
//! 性能比較ハーネス。`bench_harness::protocol::run`（warmup 20 回以上・
//! 計測 20 回以上・中央値／Q1/Q3 記録。TASK-8.1 準拠。
//! `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用し」の
//! 下限を包含する）を用いる。
//!
//! ## 計測を `BackendOps` トレイト経由に統一する理由
//!
//! 当初 `gemm_blis_parallel` を直接呼び逐次 `for` ループで bias 加算・
//! `relu` を模した baseline を使っていたが、実際の非融合経路
//! （`fandhe_ai_tensor_core::backend_ops::BackendOps::gemm_bias_act` のデフォルト
//! 実装が呼ぶ `elementwise::add`／`elementwise::relu`）は
//! `PARALLEL_THRESHOLD`（`1<<15` 要素。`crates/backend-cpu/src/elementwise.rs`）
//! 以上で rayon 並列化される。本ハーネスの全形状（最小 512×512=262144 要素）
//! はこの閾値を超えるため、逐次 baseline は実際に利用者が得る経路より
//! 大幅に遅く、改善比を過大評価していた（レビュー指摘）。本版は両側とも
//! `CpuBackendOps`（`ops::CpuBackendOps`）を経由させ、GEMM・bias 加算・
//! activation いずれも実カーネル（並列 elementwise・`Tensor` 出力割当を
//! 含む）で計測することで、融合による純粋な削減効果（中間 `Tensor` 2 個
//! から 1 個への削減・C の再読み出しパス削減）のみを反映させる。
//!
//! `#[ignore]` として通常 CI から除外する（`tests/gemm_blis_perf.rs` と
//! 同方針）。受け入れ条件「Linear+bias+ReLU 相当で非融合比の性能向上を
//! 実測（5 回中央値）」の実測記録は `docs/perf/cpu-gemm-epilogue-fusion.md`。
//!
//! 実行例（AVX2+FMA を有効化してビルド）:
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p fandhe-ai-backend-cpu \
//!     --release -- --ignored gemm_epilogue_perf
//! ```

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// Linear+bias+ReLU 相当の M×K×N 形状で非融合デフォルト実装・融合カーネルを
/// `BackendOps` トレイト経由で計測し、中央値・改善比を標準出力へ記録する。
fn measure(m: usize, n: usize, k: usize) {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(random_matrix(3000 + m as u64, m * k), &[m, k]).unwrap();
    let b = Tensor::new(random_matrix(4000 + n as u64, k * n), &[k, n]).unwrap();
    let bias = Tensor::new(random_matrix(5000 + n as u64, n), &[n]).unwrap();

    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）

    // 非融合 baseline: `gemm_bias_act` のデフォルト実装を明示的に組み立てる
    // （`ops.gemm` → `ops.add` → `ops.relu`。デフォルト実装の実体と同一で
    // あることは `tests/gemm_epilogue_parity.rs`
    // `gemm_bias_act_default_matches_manual_composition`〈`tensor-core`
    // 側〉で担保済み）。両ステップとも `elementwise` 実カーネル（rayon
    // 並列・`Tensor` 出力割当込み）を通る。
    let composed = run(&config, || {
        let c = ops.gemm(&a, &b).expect("gemm");
        let c = ops.add(&c, &bias).expect("add");
        std::hint::black_box(ops.relu(&c).expect("relu"));
    })
    .expect("非融合合成の計測に失敗");

    let fused = run(&config, || {
        std::hint::black_box(
            ops.gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
                .expect("gemm_bias_act"),
        );
    })
    .expect("融合カーネルの計測に失敗");

    let speedup = composed.median_secs / fused.median_secs;

    println!(
        "M={m},N={n},K={k}: composed(gemm+add+relu) median={:.6}s (q1={:.6}, q3={:.6}) / \
         fused(gemm_bias_act) median={:.6}s (q1={:.6}, q3={:.6}) / speedup={speedup:.3}x",
        composed.median_secs,
        composed.q1_secs,
        composed.q3_secs,
        fused.median_secs,
        fused.q1_secs,
        fused.q3_secs,
    );
}

#[test]
#[ignore = "性能計測ハーネス。--release かつ RUSTFLAGS で AVX2+FMA を有効化して個別実行する想定"]
fn gemm_epilogue_perf_linear_shapes() {
    // Linear 層相当（M=バッチ、K=in_features、N=out_features）の代表形状。
    for (m, k, n) in [(256usize, 1024usize, 1024usize), (1024, 1024, 1024)] {
        measure(m, n, k);
    }
    // 正方形状（メモリパス削減効果が現れやすい大きめ形状を含める）。
    for size in [512usize, 1024, 2048] {
        measure(size, size, size);
    }
}
