//! イシュー #1583 の事前登録マイクロベンチ（`#[ignore]`。実機実測専用）。
//!
//! `grad::ELEMENTWISE_VJP_VIA_BACKEND_OPS` の切替対象経路（`Op::Mul`／
//! `Op::Tanh`／`Op::Sigmoid` の乗算・`backward.rs::accumulate` の
//! fan-out 勾配合算）へ直接到達するケース (A)〜(F) の backward 単体
//! 時間を計測する。事前登録規則（比較腕・判定規則）はイシュー #1583 の
//! コメント <https://github.com/Fandhe-AI/fandhe-ai/issues/1583#issuecomment-5646324585>
//! に固定済み（`docs/perf/elementwise-vjp-backend-ops.md` 参照）。
//!
//! `FANDHE_BENCH_DEVICE` 環境変数（`cpu`〈既定〉／`metal`／`cuda`）で
//! 対象デバイスを切り替える（`infer_fixed_cost_bench.rs` 等、既存
//! 実機ベンチと同じ方式）。出力は before/after で diff 可能な固定形式
//! （`bench[<case>][<numel>].median_s=`・`grad[<case>][<numel>].fold_bits=`・
//! `grad[<case>][<numel>][<i>].bits=`）とする。
//!
//! forward は `.to_tensor()` で計測前に実体化し、計測区間は
//! `tape.backward(&loss)` のみに限定する（forward 側のコストを
//! backward 単体の A/B から除外するため）。

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::{Device, Tape, Tensor};
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS: usize = 20;
const SIZES: [usize; 3] = [16384, 65536, 1048576];

fn random_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).unwrap()
}

/// `FANDHE_BENCH_DEVICE`（`cpu`〈既定〉／`metal`／`cuda`）からデバイスを
/// 解決する。未対応値・実機非到達（provider 不在等）は `panic!`（本
/// ベンチは実機専用の `#[ignore]` テストのため、判定不能を静かに丸めず
/// 即座に失敗させる方が診断しやすい）。
fn device_from_env() -> Device {
    match std::env::var("FANDHE_BENCH_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        #[cfg(target_os = "macos")]
        Ok("metal") => Device::Metal,
        Ok("cuda") => Device::Cuda(0),
        Ok(other) => panic!("FANDHE_BENCH_DEVICE: 未対応の値 '{other}'"),
    }
}

fn new_tape(device: Device) -> Tape {
    fandhe_ai::tape_for(device).unwrap_or_else(|err| {
        panic!("tape_for({device:?}) に失敗した（実機非到達の可能性）: {err}")
    })
}

/// 2D 形状（`rows × cols`）を `numel` の近似平方分解で決める（正方に
/// 近い形状を優先し、割り切れない場合は `cols` 側へ余りを寄せる）。
fn shape_2d(numel: usize) -> [usize; 2] {
    let rows = (numel as f64).sqrt().round() as usize;
    let rows = rows.max(1);
    let cols = numel.div_ceil(rows);
    [rows, cols]
}

/// 勾配テンソルの `to_bits()` を fold した診断用チェックサム
/// （bit 完全一致比較用。before/after で同一入力・同一 shape なら
/// 完全一致するはず）。
fn fold_bits(t: &Tensor<f32>) -> u64 {
    let mut acc: u64 = 0xcbf29ce484222325; // FNV-1a 相当の固定初期値（診断専用・暗号用途ではない）
    for &v in t.host_slice().iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    acc
}

/// 1 ケース分の backward 計測 + 勾配ダンプ出力。
fn run_case(
    label: &str,
    numel: usize,
    device: Device,
    mut build: impl FnMut(&Tape) -> fandhe_ai::Var<'_>,
) {
    let mut samples = Vec::with_capacity(ITERS);
    let mut last_grad_fold: Option<u64> = None;
    let mut last_grad_head: Vec<f32> = Vec::new();

    for i in 0..(WARMUP + ITERS) {
        let tape = new_tape(device);
        let loss = build(&tape);
        // 計測前に forward を実体化し、backward 単体の時間だけを測る。
        std::hint::black_box(loss.to_tensor());

        let start = Instant::now();
        let grads = tape.backward(&loss).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        std::hint::black_box(&grads);

        if i >= WARMUP {
            samples.push(elapsed);
        }
        if i == WARMUP + ITERS - 1 {
            // 最終反復の leaf 0 の勾配を記録（bit 一致確認用）。
            if let Some(leaf0) = tape.leaf(0)
                && let Ok(Some(g)) = grads.get(&leaf0)
            {
                last_grad_fold = Some(fold_bits(g));
                last_grad_head = g.host_slice().iter().take(8).copied().collect();
            }
        }
    }

    let q = median_q1_q3(&samples).unwrap();
    println!("bench[{label}][{numel}].median_s={:.9}", q.median);
    if let Some(fold) = last_grad_fold {
        println!("grad[{label}][{numel}].fold_bits={fold:#018x}");
        for (i, v) in last_grad_head.iter().enumerate() {
            println!("grad[{label}][{numel}][{i}].bits={:#010x}", v.to_bits());
        }
    }
}

#[test]
#[ignore = "実機実測専用（FANDHE_BENCH_DEVICE で対象デバイス指定。イシュー #1583）"]
fn elementwise_vjp_backward_cases() {
    let device = device_from_env();

    for &numel in &SIZES {
        let [rows, cols] = shape_2d(numel);

        // (A) mul 連続×連続
        run_case("mul_contig", numel, device, |tape| {
            let a = random_tensor(1, &[rows, cols]);
            let b = random_tensor(2, &[rows, cols]);
            let a = tape.var(&a);
            let b = tape.var(&b);
            a.mul(&b).unwrap()
        });

        // (B) mul の upstream が transpose view
        run_case("mul_transpose_upstream", numel, device, |tape| {
            let a = random_tensor(3, &[rows, cols]);
            let b = random_tensor(4, &[rows, cols]);
            let a = tape.var(&a);
            let b = tape.var(&b);
            a.mul(&b).unwrap().transpose(0, 1).unwrap()
        });

        // (C) mul broadcast（x[rows, cols] * bias[cols]）
        run_case("mul_broadcast", numel, device, |tape| {
            let x = random_tensor(5, &[rows, cols]);
            let bias = random_tensor(6, &[cols]);
            let x = tape.var(&x);
            let bias = tape.var(&bias);
            x.mul(&bias).unwrap()
        });

        // (D) tanh
        run_case("tanh", numel, device, |tape| {
            let x = random_tensor(7, &[rows, cols]);
            let x = tape.var(&x);
            x.tanh()
        });

        // (E) sigmoid
        run_case("sigmoid", numel, device, |tape| {
            let x = random_tensor(8, &[rows, cols]);
            let x = tape.var(&x);
            x.sigmoid()
        });

        // (F) fan-out（x.mul(y) + x.tanh()。backward.rs::accumulate の
        // fan-out 勾配合算〈同 shape の `ops.add`〉に到達させる）
        run_case("fan_out_accumulate", numel, device, |tape| {
            let x = random_tensor(9, &[rows, cols]);
            let y = random_tensor(10, &[rows, cols]);
            let x = tape.var(&x);
            let y = tape.var(&y);
            let m = x.mul(&y).unwrap();
            let t = x.tanh();
            m.add(&t).unwrap()
        });
    }
}
