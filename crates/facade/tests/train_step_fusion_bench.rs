//! イシュー #1044 受け入れ条件 2（改善前後の実測記録）: 学習 forward の
//! epilogue 融合（`Linear`＋`ReLU` を `gemm_bias_act` へ結線）が
//! forward+backward の 1 ステップ時間にもたらす効果を計測する record-only
//! ベンチ（`infer_fixed_cost_bench.rs`・`device_param_store_bench.rs` と
//! 同じ方針。hard assert しない。`.claude/rules/coding-rust.md`「ベンチは
//! 5 回計測の中央値」）。
//!
//! **旧経路（本ベンチにおける参照点。#1044 適用前の演算列を手組みで
//! 再現）**: `SequentialVars::linears()`（`Linear` 層のみを層順に抽出した
//! `LinearVars` 列。公開 API）を使い、`LinearVars::forward`（非融合の
//! `matmul` → `add`）→ `Var::relu()`（別ノード）→ `LinearVars::forward`
//! と手動で合成する。`Sequential::forward`／`SequentialVars::forward` は
//! #1044 で融合済みのため、旧経路の計測には使えない（同じ理由で
//! `bound.forward(...)` は「新経路」側にのみ使う）。
//!
//! **新経路**: `SequentialVars::forward`（#1044 で `Linear`→`ReLU` を
//! 1 ノード〈`Op::LinearAct`〉・1 カーネル起動〈`gemm_bias_act`〉へ結線
//! 済み）。
//!
//! **形状**: `infer_fixed_cost_bench.rs` と同じ framework-compare 推論
//! プロトコル（batch 64・784→256→ReLU→10）を学習 forward+backward へ
//! 転用する（`docs/inference-forward-fixed-cost-design.md` §1 参照）。
//!
//! CPU は常時利用可能なため通常テストとして実行する（Metal／DGX Spark
//! GB10 実機は未実施。`docs/perf/train-linear-epilogue-fusion.md` に
//! その旨を明記する。両バックエンドは `gemm_resident_rhs_act` を
//! オーバーライドしないため resident forward の融合効果は host 常駐
//! forward〈本ベンチの対象〉のみに限られる。`docs/kernel-fusion.md`
//! §2.2.1「スコープ外」参照）。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

const WARMUP: usize = 20;
const ITERS: usize = 20;
const TRIALS: usize = 5;
const BATCH: usize = 64;
const IN_FEATURES: usize = 784;
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(IN_FEATURES, HIDDEN, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_linear(HIDDEN, OUT_FEATURES, /* seed = */ 43)
        .unwrap()
}

fn random_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).unwrap()
}

/// 旧経路: `LinearVars::forward`（非融合 `matmul`→`add`）→ `Var::relu()`
/// （別ノード）→ `LinearVars::forward` を手動合成し、forward+backward を
/// `iters` 回実行した平均秒を返す。
fn run_manual_composition(
    model: &Sequential,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    iters: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let linears = bound.linears();
        let x = tape.var(x_data);
        let y = tape.var(y_data);
        let h = linears[0].forward(&x).unwrap();
        let h = h.relu();
        let pred = linears[1].forward(&h).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward(&loss).unwrap();
        std::hint::black_box(bound.trainable_grads(&grads).unwrap());
    }
    start.elapsed().as_secs_f64() / iters as f64
}

/// 新経路: `SequentialVars::forward`（#1044 で `Linear`＋`ReLU` を
/// `gemm_bias_act` へ融合済み）で forward+backward を `iters` 回実行した
/// 平均秒を返す。
fn run_fused(model: &Sequential, x_data: &Tensor<f32>, y_data: &Tensor<f32>, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(x_data);
        let y = tape.var(y_data);
        let pred = bound.forward(&tape, &x).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward(&loss).unwrap();
        std::hint::black_box(bound.trainable_grads(&grads).unwrap());
    }
    start.elapsed().as_secs_f64() / iters as f64
}

#[test]
fn train_step_epilogue_fusion_cpu() {
    let model = build_model();
    let x_data = random_tensor(0x9001, &[BATCH, IN_FEATURES]);
    let y_data = random_tensor(0x9002, &[BATCH, OUT_FEATURES]);

    // warmup: 初回呼び出しの結線コストを本計測から除く。
    let _ = run_manual_composition(&model, &x_data, &y_data, WARMUP);
    let _ = run_fused(&model, &x_data, &y_data, WARMUP);

    let mut manual_samples = Vec::with_capacity(TRIALS);
    let mut fused_samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        manual_samples.push(run_manual_composition(&model, &x_data, &y_data, ITERS));
        fused_samples.push(run_fused(&model, &x_data, &y_data, ITERS));
    }

    let manual_q = median_q1_q3(&manual_samples).expect("TRIALS 個の non-NaN サンプル");
    let fused_q = median_q1_q3(&fused_samples).expect("TRIALS 個の non-NaN サンプル");

    println!(
        "[train_step_fusion_bench:cpu] batch={BATCH} {IN_FEATURES}->{HIDDEN}->ReLU->{OUT_FEATURES} \
         manual_median_s={:.9} (q1={:.9}, q3={:.9}) \
         fused_median_s={:.9} (q1={:.9}, q3={:.9}) \
         speedup_x={:.3} fused_faster={} \
         — record only, non-gating（本ファイル冒頭コメント参照。実測値は \
         docs/perf/train-linear-epilogue-fusion.md へ転記する）",
        manual_q.median,
        manual_q.q1,
        manual_q.q3,
        fused_q.median,
        fused_q.q1,
        fused_q.q3,
        manual_q.median / fused_q.median.max(f64::EPSILON),
        fused_q.median < manual_q.median,
    );

    // 数値結果は不変のはず（bit 一致は
    // `compat::sequential::tests::sequential_vars_forward_with_activation_grad_matches_manual_composition`
    // が検証済み）。ここでは shape のみ再確認する。
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let x = tape.var(&x_data);
    let fused_out = bound.forward(&tape, &x).unwrap().to_tensor();
    assert_eq!(fused_out.shape(), &[BATCH, OUT_FEATURES]);
}
