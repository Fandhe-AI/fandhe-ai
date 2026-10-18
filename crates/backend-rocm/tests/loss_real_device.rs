//! ROCm 実機（MI300X・gfx942）での loss 10 メソッドを CPU 参照と突き合わせる実機 parity テスト
//! （イシュー #3436・R5）。実測は #3159 段 B（MI300X・gfx942）へ申し送り。
//!
//! - 役割: `RocmBackendOps::new(Gfx942)` の mse・huber・bce・nll・kl_div の forward／backward を
//!   `CpuBackendOps` と REQ-2 統一複合判定（`assert_parity`。定数は変更しない）で照合する。
//!   同じ入力を 2 回実行して bit 一致すること（決定性。`atomicAdd` 不使用）も確認する。
//! - 実機で確認する点: HIPRTC での `logf`・`log1pf`・`expf` 等の数学関数の解決と精度、
//!   `__builtin_fmaf`・`#pragma clang fp contract(off)`・`__launch_bounds__` の受理、
//!   wave64 での縮約順序の差（`WARPS_PER_BLOCK = 4`）。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。サイズは `src/loss_model.rs` の
//!   CPU 照合テストと同じ集合（`LOSS_MAX_BLOCKS` 上限で grid-stride が 2 周以上する最後の
//!   サイズを含む）。

use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
use fandhe_ai_backend_rocm::{GfxTarget, RocmBackendOps};
use fandhe_ai_tensor_core::{BackendOps, BceKind, HuberKind, KlDivTarget, MseReduction, Tensor};

const SIZES: [usize; 7] = [1, 7, 255, 256, 257, 4097, 256 * 1024 + 3];
const REDUCTIONS: [MseReduction; 2] = [MseReduction::Mean, MseReduction::Sum];

fn lcg(len: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            lo + (hi - lo) * ((s >> 8) as f32 / (1u32 << 24) as f32)
        })
        .collect()
}

fn t1(v: Vec<f32>) -> Tensor<f32> {
    let n = v.len();
    Tensor::new(v, &[n]).unwrap()
}

fn check(label: &str, got: &Tensor<f32>, want: &Tensor<f32>) {
    assert_eq!(got.shape(), want.shape(), "{label}");
    assert_parity(
        label,
        got.contiguous().as_slice().unwrap(),
        want.contiguous().as_slice().unwrap(),
    );
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
#[ignore = "ROCm 実機（MI300X・gfx942）専用。実測は #3159 段 B"]
fn elementwise_loss_family_matches_cpu_and_is_deterministic() {
    let (dev, cpu) = (RocmBackendOps::new(GfxTarget::Gfx942), CpuBackendOps);
    for n in SIZES {
        let p = t1(lcg(n, 1, -3.0, 3.0));
        let tg = t1(lcg(n, 2, -3.0, 3.0));
        let probs = t1(lcg(n, 3, 0.01, 0.99));
        let tprob = t1(lcg(n, 4, 0.0, 1.0));
        for red in REDUCTIONS {
            let got = dev.mse_loss(&p, &tg, red).unwrap();
            check(
                &format!("mse n{n}"),
                &got,
                &cpu.mse_loss(&p, &tg, red).unwrap(),
            );
            assert_eq!(bits(&got), bits(&dev.mse_loss(&p, &tg, red).unwrap()));
            for (k, d) in [(HuberKind::Huber, 0.7f32), (HuberKind::SmoothL1, 1.3)] {
                check(
                    &format!("huber n{n}"),
                    &dev.huber_loss(&p, &tg, k, d, red).unwrap(),
                    &cpu.huber_loss(&p, &tg, k, d, red).unwrap(),
                );
            }
            for k in [BceKind::Probabilities, BceKind::Logits] {
                let x = if matches!(k, BceKind::Probabilities) {
                    &probs
                } else {
                    &p
                };
                check(
                    &format!("bce n{n}"),
                    &dev.bce_loss(x, &tprob, k, red).unwrap(),
                    &cpu.bce_loss(x, &tprob, k, red).unwrap(),
                );
            }
            for k in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
                check(
                    &format!("kl_div n{n}"),
                    &dev.kl_div_loss(&p, &tprob, k, red).unwrap(),
                    &cpu.kl_div_loss(&p, &tprob, k, red).unwrap(),
                );
            }
        }
        check(
            &format!("mse bwd n{n}"),
            &dev.mse_loss_backward(&p, &tg, 0.3).unwrap(),
            &cpu.mse_loss_backward(&p, &tg, 0.3).unwrap(),
        );
        for (k, d) in [(HuberKind::Huber, 0.7f32), (HuberKind::SmoothL1, 1.3)] {
            check(
                &format!("huber bwd n{n}"),
                &dev.huber_loss_backward(&p, &tg, k, d, 0.5).unwrap(),
                &cpu.huber_loss_backward(&p, &tg, k, d, 0.5).unwrap(),
            );
        }
        for k in [BceKind::Probabilities, BceKind::Logits] {
            let x = if matches!(k, BceKind::Probabilities) {
                &probs
            } else {
                &p
            };
            check(
                &format!("bce bwd n{n}"),
                &dev.bce_loss_backward(x, &tprob, k, 0.5).unwrap(),
                &cpu.bce_loss_backward(x, &tprob, k, 0.5).unwrap(),
            );
        }
        for k in [KlDivTarget::Probabilities, KlDivTarget::LogProbabilities] {
            check(
                &format!("kl_div bwd n{n}"),
                &dev.kl_div_loss_backward(&p, &tprob, k, 0.5).unwrap(),
                &cpu.kl_div_loss_backward(&p, &tprob, k, 0.5).unwrap(),
            );
        }
    }
}

#[test]
#[ignore = "ROCm 実機（MI300X・gfx942）専用。実測は #3159 段 B"]
fn nll_matches_cpu_and_is_deterministic() {
    let (dev, cpu) = (RocmBackendOps::new(GfxTarget::Gfx942), CpuBackendOps);
    let cases: [(&[usize], usize); 5] = [
        (&[5], 0),
        (&[4, 6], 0),
        (&[3, 5, 4], 1),
        (&[300, 9, 2], 1),
        (&[256 * 1024 + 3, 3], 1),
    ];
    for (shape, cd) in cases {
        let numel: usize = shape.iter().product();
        let c = shape[cd];
        let rest: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != cd)
            .map(|(_, &d)| d)
            .collect();
        let n: usize = rest.iter().product();
        let input = Tensor::new(lcg(numel, 11, -5.0, 0.0), shape).unwrap();
        let targets: Vec<i32> = (0..n).map(|s| ((s * 7 + 3) % c) as i32).collect();
        let tt = Tensor::new(targets, &rest).unwrap();
        for red in REDUCTIONS {
            let got = dev.nll_loss(&input, &tt, cd, red).unwrap();
            check(
                &format!("nll {shape:?}"),
                &got,
                &cpu.nll_loss(&input, &tt, cd, red).unwrap(),
            );
            assert_eq!(
                bits(&got),
                bits(&dev.nll_loss(&input, &tt, cd, red).unwrap())
            );
        }
        check(
            &format!("nll bwd {shape:?}"),
            &dev.nll_loss_backward(shape, &tt, cd, 0.125).unwrap(),
            &cpu.nll_loss_backward(shape, &tt, cd, 0.125).unwrap(),
        );
    }
}
