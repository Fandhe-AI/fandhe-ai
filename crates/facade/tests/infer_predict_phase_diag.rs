//! イシュー #1218（`docs/perf/cpu-infer-predict-profile.md`）: CPU 推論
//! `Sequential::predict` 経路のフェーズ分解 record-only 診断ベンチ
//! （`infer_fixed_cost_bench.rs`・`device_param_store_bench.rs` と同じ
//! 方針。hard assert しない。`.claude/rules/coding-rust.md`「ベンチは
//! 5 回計測の中央値」）。
//!
//! `--task infer --phases`（`bench-fandhe`）はハーネス制約
//! （`MEASURE_ERROR`）で predict 内部の内訳を取れない
//! （`docs/perf/train-step-phase-breakdown.md` §15.5）ため、本ベンチは
//! `fandhe_ai_backend_cpu::CpuBackendOps` を直接呼び、`Sequential::
//! predict`（イシュー #1218 で `Linear`→`ReLU` を `gemm_bias_act` へ
//! 結線済み）を構成する各フェーズ（1 個目の `Linear`→`ReLU` の融合
//! カーネル・2 個目の `Linear` の非融合合成〈`gemm`→`add`〉）と全体を
//! 個別に計測し、どのフェーズが `predict` 全体に対して支配的かを
//! 記録する。
//!
//! **形状**: `infer_fixed_cost_bench.rs` と同じ framework-compare 推論
//! プロトコル（batch 64・784→256→ReLU→10）。
//!
//! **検証する仮説**（`docs/perf/cpu-infer-predict-profile.md` 参照）:
//! - (i) `CpuBackendOps::new()` は ZST（`crates/backend-cpu/src/ops.rs`）
//!   のため計測に現れないはず（本ベンチはループ内で毎回 `new()` する
//!   構成と 1 回だけ構築する構成を両方計測して比較する）
//! - (ii) 融合 `gemm_bias_act`（L1）と非融合合成（L2 の `gemm`→`add`）の
//!   相対コスト
//! - (iii) `predict` 全体 ≈ L1 フェーズ + L2 フェーズの和に近いか
//!   （乖離が大きい場合、層間の中間 `Tensor` 割当等の固定費が別途
//!   支配的である可能性を示す）

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{Activation, BackendOps};

const WARMUP: usize = 20;
const ITERS: usize = 20;
const TRIALS: usize = 5;
const BATCH: usize = 64;
const IN_FEATURES: usize = 784;
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;

fn random_input(seed: u64) -> Tensor<f32> {
    let data = Xorshift64Star::new(seed).fill_vec(BATCH * IN_FEATURES);
    Tensor::new(data, &[BATCH, IN_FEATURES]).unwrap()
}

/// 中央値・q1/q3 を計測して 1 行フォーマットする（各フェーズ共通の
/// 計測ループ。`warmup` 回で結線コストを除いてから `TRIALS` 回
/// 中央値を取る。`infer_fixed_cost_bench.rs` と同じ方式）。
fn measure(label: &str, mut f: impl FnMut()) {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let start = Instant::now();
        for _ in 0..ITERS {
            f();
        }
        samples.push(start.elapsed().as_secs_f64() / ITERS as f64);
    }
    let q = median_q1_q3(&samples).expect("TRIALS 個の non-NaN サンプル");
    println!(
        "[infer_predict_phase_diag:cpu] {label} median_s={:.9} (q1={:.9}, q3={:.9}) \
         — record only, non-gating（本ファイル冒頭コメント参照。実測値は \
         docs/perf/cpu-infer-predict-profile.md へ転記する）",
        q.median, q.q1, q.q3,
    );
}

#[test]
fn infer_predict_phase_diag_cpu() {
    // rayon のスレッド数は `RAYON_NUM_THREADS` 環境変数で確認する
    // （`rayon` は facade の直接依存でないため、統合テストからは
    // `rayon::current_num_threads()` を呼べない。`backend-cpu` 内部の
    // 並列化構成に依存する値のため facade 側では公開していない）。
    println!(
        "[infer_predict_phase_diag:cpu] RAYON_NUM_THREADS={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(unset)".to_string())
    );

    let l1 = Linear::new(IN_FEATURES, HIDDEN, true, 42).unwrap();
    let l2 = Linear::new(HIDDEN, OUT_FEATURES, true, 43).unwrap();
    let input = random_input(0x9000);
    let ops = CpuBackendOps::new();

    // 仮説 (i): `CpuBackendOps::new()` が ZST であることの直接計測
    // （毎回 `new()` する版と 1 回だけ構築する版の差が計測誤差内で
    // あれば、`CpuBackendOps::new()` 自体は predict のホットスポットで
    // ないことの追認になる）。
    measure("cpu_backend_ops_new_only (baseline noise floor)", || {
        std::hint::black_box(CpuBackendOps::new());
    });

    // L1: Linear(784->256) -> ReLU を融合 gemm_bias_act で計算する
    // （`Sequential::predict` の Linear→ReLU 先読み結線と同じ呼び出し。
    // `nn/linear.rs::Linear::forward_host_with_activation` 参照）。
    measure("l1_linear_relu_fused_gemm_bias_act", || {
        std::hint::black_box(
            l1.forward_host_with_activation(&ops, &input, Activation::Relu)
                .unwrap(),
        );
    });

    // L1 の非融合合成（gemm -> add -> relu）。融合との相対コスト比較用。
    let h_fixture = l1
        .forward_host_with_activation(&ops, &input, Activation::Relu)
        .unwrap();
    measure("l1_linear_relu_unfused_gemm_add_relu", || {
        let y = ops.gemm(&input, l1.weight()).unwrap();
        let y = ops.add(&y, l1.bias().unwrap()).unwrap();
        std::hint::black_box(ops.relu(&y).unwrap());
    });

    // L2: Linear(256->10)（ReLU が続かないため非融合合成のまま。
    // `Sequential::predict` と同じ経路）。
    measure("l2_linear_unfused_gemm_add", || {
        let y = ops.gemm(&h_fixture, l2.weight()).unwrap();
        std::hint::black_box(ops.add(&y, l2.bias().unwrap()).unwrap());
    });

    // predict 全体（Sequential::predict。本イシューで Linear→ReLU を
    // gemm_bias_act へ結線した後の経路）。
    let model = Sequential::new()
        .add_linear(IN_FEATURES, HIDDEN, 42)
        .unwrap()
        .add_relu()
        .add_linear(HIDDEN, OUT_FEATURES, 43)
        .unwrap();
    measure("predict_full", || {
        std::hint::black_box(model.predict(&input).unwrap());
    });
}
