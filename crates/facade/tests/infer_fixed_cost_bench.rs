//! イシュー #1028 受け入れ条件 1（改善前後の中央値記録）: 推論 forward の
//! 固定費削減を計測する record-only ベンチ（`device_param_store_bench.rs`
//! と同じ方針。hard assert しない。`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値」）。
//!
//! **旧経路（本ベンチにおける参照点）**: `Sequential::predict` が本
//! イシュー以前に使っていた実装と同一（`tape.var(input)` → `Sequential::
//! forward` → `to_tensor()`。`compat/sequential.rs::predict_via_tape` の
//! 内部実装と同じ組み立てだが、`predict_via_tape` はクレート内
//! `pub(crate)` 未満のプライベートメソッドのため、ここでは公開 API
//! （[`fandhe_ai::tape`]・[`Sequential::forward`]）で同じ組み立てを
//! 再現する）。層ごとに `Linear::bind` が `weight`／`bias` を
//! `Tape::var`（内部で `tensor.clone()`）としてテープへ登録し直す
//! （`nn/linear.rs`）ため、forward 1 回あたり全パラメータのホスト
//! クローンとノード記録のアロケーションを払う。
//!
//! **新経路**: [`Sequential::predict`]（イシュー #1028 で内部実装を
//! tape 不要経路へ差し替え済み。`compat/sequential.rs::predict_tape_free`。
//! 公開シグネチャ・戻り値・数値結果は不変）。`Tape`／`Var` を構築せず
//! `Module::forward_host` を直接呼ぶため、パラメータクローン・ノード
//! 記録のアロケーションがない。
//!
//! **形状**: `docs/inference-forward-fixed-cost-design.md` §1 が引用する
//! framework-compare の推論プロトコル（batch 64・784→256→ReLU→10・
//! warmup 20・iters 20・中央値）に合わせる。
//!
//! CPU は常時利用可能なため通常テストとして実行する（CUDA/Metal は
//! 別イシュー〈#1028 out-of-scope〉で実機セッションが追加する）。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;

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

fn random_input(seed: u64) -> Tensor<f32> {
    let data = Xorshift64Star::new(seed).fill_vec(BATCH * IN_FEATURES);
    Tensor::new(data, &[BATCH, IN_FEATURES]).unwrap()
}

/// 旧経路（`tape.var` → `Sequential::forward` → `to_tensor()`）。
/// `predict_tape_free` 導入前の `Sequential::predict` 実装と同一の組み立て。
fn run_via_tape(model: &Sequential, input: &Tensor<f32>, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        let tape = fandhe_ai::tape();
        let input_var = tape.var(input);
        let output = model.forward(&tape, &input_var).unwrap();
        std::hint::black_box(output.to_tensor());
    }
    start.elapsed().as_secs_f64() / iters as f64
}

/// 新経路（`Sequential::predict`。イシュー #1028 で tape 不要化）。
fn run_predict(model: &Sequential, input: &Tensor<f32>, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.predict(input).unwrap());
    }
    start.elapsed().as_secs_f64() / iters as f64
}

#[test]
fn infer_forward_fixed_cost_cpu() {
    let model = build_model();
    let input = random_input(0x9000);

    // warmup: 初回呼び出しの結線コストを本計測から除く
    // （`device_param_store_bench.rs` と同じ方針）。
    let _ = run_via_tape(&model, &input, WARMUP);
    let _ = run_predict(&model, &input, WARMUP);

    let mut via_tape_samples = Vec::with_capacity(TRIALS);
    let mut predict_samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        via_tape_samples.push(run_via_tape(&model, &input, ITERS));
        predict_samples.push(run_predict(&model, &input, ITERS));
    }

    let via_tape_q = median_q1_q3(&via_tape_samples).expect("TRIALS 個の non-NaN サンプル");
    let predict_q = median_q1_q3(&predict_samples).expect("TRIALS 個の non-NaN サンプル");

    println!(
        "[infer_fixed_cost_bench:cpu] batch={BATCH} {IN_FEATURES}->{HIDDEN}->ReLU->{OUT_FEATURES} \
         via_tape_median_s={:.9} (q1={:.9}, q3={:.9}) \
         predict_median_s={:.9} (q1={:.9}, q3={:.9}) \
         speedup_x={:.3} predict_faster={} \
         — record only, non-gating（本ファイル冒頭コメント参照。実測値は \
         docs/inference-forward-fixed-cost-design.md §実測記録へ転記する）",
        via_tape_q.median,
        via_tape_q.q1,
        via_tape_q.q3,
        predict_q.median,
        predict_q.q1,
        predict_q.q3,
        via_tape_q.median / predict_q.median.max(f64::EPSILON),
        predict_q.median < via_tape_q.median,
    );

    // 数値結果は不変のはず（新旧経路 bit 完全一致は
    // `compat::sequential::tests::sequential_predict_tape_free_matches_via_tape_bit_exact`
    // が検証済み）。ここでは shape のみ再確認する。
    let via_tape_out = {
        let tape = fandhe_ai::tape();
        let input_var = tape.var(&input);
        model.forward(&tape, &input_var).unwrap().to_tensor()
    };
    let predict_out = model.predict(&input).unwrap();
    assert_eq!(via_tape_out.shape(), predict_out.shape());
}
