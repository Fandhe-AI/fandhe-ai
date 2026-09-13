//! イシュー #1692 の事前登録マイクロベンチ（`#[ignore]`。実機実測専用）。
//!
//! CUDA 側 `mse_loss_backward`（`backend-cuda::mse::CudaMse::
//! run_mse_backward_f32`）がストリーム順序契約（`docs/backend-cuda-
//! async-execution-design.md` §2.3・§16）に既に準拠していることをコード
//! 読解で確認したうえで、`Var::mse_loss` の backward 単体時間を計測する
//! 事前登録マイクロベンチ。参照実装は `elementwise_vjp_bench.rs`
//! （イシュー #1583）と同型構成。事前登録規則（比較腕・判定規則）は
//! イシュー #1692 のコメント
//! <https://github.com/Fandhe-AI/fandhe-ai/issues/1692#issuecomment-5654656690>
//! に固定済み（`docs/perf/cuda-mse-backward-stream-contract.md` 参照）。
//!
//! `FANDHE_BENCH_DEVICE` 環境変数（`cpu`〈既定〉／`metal`／`cuda`）で
//! 対象デバイスを切り替える（既存実機ベンチと同じ方式）。出力は
//! before/after で diff 可能な固定形式（`bench[<case>][<numel>].median_s=`・
//! `grad[<case>][<numel>].fold_bits=`・`grad[<case>][<numel>][<i>].bits=`）
//! とする。
//!
//! forward は `.to_tensor()` で計測前に実体化し、計測区間は
//! `tape.backward(&loss)` のみに限定する（`mse_loss` の VJP 呼び出し時間
//! だけを before/after 比較の対象にするため）。

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::{Device, Tape, Tensor};
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS: usize = 20;
/// 一般形状スイープ（`elementwise_vjp_bench.rs::SIZES` と同一値）。
const SIZES: [usize; 3] = [16384, 65536, 1048576];

fn random_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).unwrap()
}

/// `FANDHE_BENCH_DEVICE`（`cpu`〈既定〉／`metal`／`cuda`）からデバイスを
/// 解決する。未対応値・実機非到達（provider 不在等）は `panic!`（本
/// ベンチは実機専用の `#[ignore]` テストのため、判定不能を静かに丸めず
/// 即座に失敗させる方が診断しやすい。`elementwise_vjp_bench.rs::
/// device_from_env` と同一方針）。
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

/// 2D 形状（`rows × cols`）を `numel` の近似平方分解で決める
/// （`elementwise_vjp_bench.rs::shape_2d` と同一方針）。
fn shape_2d(numel: usize) -> [usize; 2] {
    let rows = (numel as f64).sqrt().round() as usize;
    let rows = rows.max(1);
    let cols = numel.div_ceil(rows);
    [rows, cols]
}

/// 勾配テンソルの `to_bits()` を fold した診断用チェックサム
/// （bit 完全一致比較用。before/after で同一入力・同一 shape なら
/// 完全一致するはず。`elementwise_vjp_bench.rs::fold_bits` と同一実装）。
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
            // 最終反復の leaf 0（pred）の勾配を記録（bit 一致確認用）。
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
#[ignore = "実機実測専用（FANDHE_BENCH_DEVICE で対象デバイス指定。イシュー #1692）"]
fn mse_backward_cases() {
    let device = device_from_env();

    // (train) 訓練ハーネス実形状（`[64, 10]`。640 要素・固定費支配）。
    // `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 が指摘する
    // 「MSE backward の固定費」を代表する形状として最優先で計測する。
    {
        let numel = 64 * 10;
        run_case("train_shape", numel, device, |tape| {
            let pred = random_tensor(101, &[64, 10]);
            let target = random_tensor(102, &[64, 10]);
            let pred = tape.var(&pred);
            let target = tape.var(&target);
            pred.mse_loss(&target).unwrap()
        });
    }

    // (general) 一般形状スイープ（`elementwise_vjp_bench.rs` と同じ
    // 3 サイズ。大形状での傾向確認用）。
    for &numel in &SIZES {
        let [rows, cols] = shape_2d(numel);
        run_case("general_shape", numel, device, |tape| {
            let pred = random_tensor(201, &[rows, cols]);
            let target = random_tensor(202, &[rows, cols]);
            let pred = tape.var(&pred);
            let target = tape.var(&target);
            pred.mse_loss(&target).unwrap()
        });
    }
}
