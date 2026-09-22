//! イシュー #2085（区分 B-1）の事前登録マイクロベンチ（`#[ignore]`。
//! 実機実測専用）。
//!
//! GPU `run_fused` の elementwise allowlist 融合（既定 OFF・opt-in
//! ゲート）の A/B 計測を行う。事前登録規則（比較腕・判定規則）は
//! `docs/perf/gpu-elementwise-fusion-b1.md`「事前登録判定規則」に
//! 固定済み。`elementwise_vjp_bench.rs`（#1583）と同じ構成方針
//! （`FANDHE_BENCH_DEVICE` でデバイス切替・固定形式出力・計測前に
//! forward を実体化）を踏襲するが、本ベンチは**計測区間に forward
//! 実体化（`to_tensor()`）も含める**（融合カーネルの効果は forward
//! 実体化時の `run_fused` 呼び出しに現れるため。設計判断
//! `docs/perf/gpu-elementwise-fusion-b1.md` 参照）。
//!
//! ゲートは `FANDHE_BENCH_GPU_EW_FUSION`（`0`〈既定〉／`1`）で切り替える。
//! ライブラリ本体は環境変数を読まないため（`fused_elementwise.rs`
//! モジュール冒頭コメント参照）、本ベンチ自身が起動時に 1 回だけ読み、
//! 対象デバイスに対応するバックエンドクレートの `pub` setter
//! （`backend-cuda::fused_elementwise::set_gpu_elementwise_fusion_enabled`／
//! `backend-metal::fused_elementwise::set_gpu_elementwise_fusion_enabled`。
//! `facade` からは再公開されていないため直接呼ぶ）を呼ぶ。

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

fn device_from_env() -> Device {
    match std::env::var("FANDHE_BENCH_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        #[cfg(target_os = "macos")]
        Ok("metal") => Device::Metal,
        Ok("cuda") => Device::Cuda(0),
        Ok(other) => panic!("FANDHE_BENCH_DEVICE: 未対応の値 '{other}'"),
    }
}

/// `FANDHE_BENCH_GPU_EW_FUSION`（`0`〈既定〉／`1`）を読み、`device` に
/// 対応するバックエンドクレートのゲートへ反映する。`Device::Cpu` は
/// ゲート自体を持たない（CPU は常時融合。#163 で結線済み）ため
/// no-op（control 腕）。
fn apply_gate_from_env(device: Device) {
    let enabled = match std::env::var("FANDHE_BENCH_GPU_EW_FUSION").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(other) => panic!("FANDHE_BENCH_GPU_EW_FUSION: 未対応の値 '{other}'"),
    };
    match device {
        Device::Cpu => {}
        Device::Cuda(_) => {
            fandhe_ai_backend_cuda::fused_elementwise::set_gpu_elementwise_fusion_enabled(enabled);
        }
        #[cfg(target_os = "macos")]
        Device::Metal => {
            fandhe_ai_backend_metal::fused_elementwise::set_gpu_elementwise_fusion_enabled(enabled);
        } // `Device`（`tensor-core::device::Device`）は `Cpu`／`Cuda(_)`／
          // （macOS 限定の）`Metal` の variant のみを持つ。非 macOS では
          // `Cpu`／`Cuda(_)` の 2 アームで、macOS では上記 3 アームで既に
          // 網羅的（exhaustive）であるため、フォールバック `_` アームは
          // いずれのターゲットでも到達しえず `unreachable_patterns`
          // warning の原因になる。ワイルドカードは追加せず、`Device` に
          // 将来 variant が増えた場合は非網羅 match のコンパイルエラーで
          // 検知させる（安全側）。
    }
}

fn new_tape(device: Device) -> Tape {
    fandhe_ai::tape_for(device).unwrap_or_else(|err| {
        panic!("tape_for({device:?}) に失敗した（実機非到達の可能性）: {err}")
    })
}

fn fold_bits(t: &Tensor<f32>) -> u64 {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in t.host_slice().iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    acc
}

/// 1 ケース分の計測（forward 実体化 + backward）+ 出力・勾配ダンプ。
fn run_case(
    label: &str,
    numel: usize,
    device: Device,
    mut build: impl FnMut(&Tape) -> fandhe_ai::Var<'_>,
) {
    let mut samples = Vec::with_capacity(ITERS);
    let mut last_out_fold: Option<u64> = None;
    let mut last_grad_fold: Option<u64> = None;

    for i in 0..(WARMUP + ITERS) {
        let tape = new_tape(device);
        let loss = build(&tape);

        let start = Instant::now();
        let out = loss.to_tensor();
        let grads = tape.backward(&loss).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        std::hint::black_box((&out, &grads));

        if i >= WARMUP {
            samples.push(elapsed);
        }
        if i == WARMUP + ITERS - 1 {
            last_out_fold = Some(fold_bits(&out));
            if let Some(leaf0) = tape.leaf(0)
                && let Ok(Some(g)) = grads.get(&leaf0)
            {
                last_grad_fold = Some(fold_bits(g));
            }
        }
    }

    let q = median_q1_q3(&samples).unwrap();
    println!("bench[{label}][{numel}].median_s={:.9}", q.median);
    if let Some(fold) = last_out_fold {
        println!("out[{label}][{numel}].fold_bits={fold:#018x}");
    }
    if let Some(fold) = last_grad_fold {
        println!("grad[{label}][{numel}].fold_bits={fold:#018x}");
    }
}

fn shape_2d(numel: usize) -> [usize; 2] {
    let rows = (numel as f64).sqrt().round() as usize;
    let rows = rows.max(1);
    let cols = numel.div_ceil(rows);
    [rows, cols]
}

#[test]
#[ignore = "実機実測専用（FANDHE_BENCH_DEVICE／FANDHE_BENCH_GPU_EW_FUSION で指定。イシュー #2085）"]
fn gpu_elementwise_fusion_cases() {
    let device = device_from_env();
    apply_gate_from_env(device);

    for &numel in &SIZES {
        let [rows, cols] = shape_2d(numel);

        // (A) 4 段連鎖 add → relu → exp → tanh
        run_case("chain4", numel, device, |tape| {
            let a = random_tensor(101, &[rows, cols]);
            let b = random_tensor(102, &[rows, cols]);
            let a = tape.var(&a);
            let b = tape.var(&b);
            a.add(&b).unwrap().relu().exp().tanh()
        });

        // (B) 6 段連鎖 add → mul → relu → exp → tanh → add
        run_case("chain6", numel, device, |tape| {
            let a = random_tensor(103, &[rows, cols]);
            let b = random_tensor(104, &[rows, cols]);
            let c = random_tensor(105, &[rows, cols]);
            let a = tape.var(&a);
            let b = tape.var(&b);
            let c = tape.var(&c);
            let h = a.add(&b).unwrap().mul(&c).unwrap().relu().exp().tanh();
            h.add(&c).unwrap()
        });

        // (C) fan-out（1 入力を 2 連鎖で共有してから加算）
        run_case("fan_out", numel, device, |tape| {
            let x = random_tensor(106, &[rows, cols]);
            let x = tape.var(&x);
            let left = x.relu().exp();
            let right = x.tanh();
            left.add(&right).unwrap()
        });

        // (D) fan-in（2 連鎖が同一ノードへ合流）
        run_case("fan_in", numel, device, |tape| {
            let a = random_tensor(107, &[rows, cols]);
            let b = random_tensor(108, &[rows, cols]);
            let a = tape.var(&a);
            let b = tape.var(&b);
            let left = a.relu().exp();
            let right = b.tanh().exp();
            left.add(&right).unwrap().tanh()
        });

        // (E) matmul 境界を挟む 2 連鎖（融合対象は matmul 前後の
        // elementwise 区間のみ。matmul 自体は融合境界ノード）
        run_case("matmul_boundary", numel, device, |tape| {
            let x = random_tensor(109, &[rows, cols]);
            let w = random_tensor(110, &[cols, cols]);
            let x = tape.var(&x);
            let w = tape.var(&w);
            let pre = x.relu().exp();
            let mm = pre.matmul(&w).unwrap();
            mm.tanh()
        });
    }
}
