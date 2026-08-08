//! `cargo bench` による性能ベンチマーク。TASK-4.2a 検証題材（D3・性能回帰）
//! の検出ゲート。`cargo bench -- --save-baseline main` でベースラインを
//! 保存し、変更後に `cargo bench -- --baseline main` で比較することで
//! 劣化を検出する。

use autodiff::Tape;
use criterion::{Criterion, criterion_group, criterion_main};
use guardrail_labeled_changes_baseline::model::Mlp;
use guardrail_labeled_changes_baseline::train::xor_dataset;
use std::hint::black_box;

fn forward_benchmark(c: &mut Criterion) {
    const SEED: u64 = 0x5EED_0001;
    let model = Mlp::new(SEED).expect("bench fixture: shape は事前に妥当");
    let (x, _y) = xor_dataset(8);

    c.bench_function("mlp_forward", |b| {
        b.iter(|| {
            let tape = Tape::new(Box::new(backend_cpu::CpuBackendOps::new()));
            let x_var = tape.var(black_box(&x));
            let (out, _, _, _) = model
                .forward(&tape, &x_var)
                .expect("bench: forward は失敗しない");
            black_box(out.to_tensor())
        })
    });
}

criterion_group!(benches, forward_benchmark);
criterion_main!(benches);
