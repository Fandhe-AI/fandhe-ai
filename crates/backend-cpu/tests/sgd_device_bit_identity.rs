//! `CpuBackendOps::sgd_step_device` の zip 化（イシュー #3193）が、
//! #3193 以前の添字ループ実装と bit 単位で同一であることの検証。
//!
//! `reference_sgd_step_indexed` は #3193 以前の本番実装の逐語の写しで、
//! 変更禁止（回帰検出の基準）。FMA 契約・tolerance は変えない。
//! NaN は payload 非依存でクラス一致として比較する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, SgdStepConfig, Tensor};

/// #3193 以前の添字ループ（変更禁止）。
fn reference_sgd_step_indexed(
    param: &mut [f32],
    grad: &[f32],
    mut velocity: Option<&mut [f32]>,
    config: &SgdStepConfig,
) {
    let use_momentum = config.momentum != 0.0;
    for j in 0..param.len() {
        let p = param[j];
        let mut g = grad[j];
        if config.weight_decay != 0.0 {
            g = config.weight_decay.mul_add(p, g);
        }
        if use_momentum {
            let v = velocity.as_deref_mut().expect("velocity");
            let prev = v[j];
            let b = if config.is_first_step {
                g
            } else {
                config.momentum.mul_add(prev, (1.0 - config.dampening) * g)
            };
            v[j] = b;
            g = if config.nesterov {
                config.momentum.mul_add(b, g)
            } else {
                b
            };
        }
        param[j] = p - config.lr * g;
    }
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// 決定的 LCG 入力に NaN・±inf・-0.0・サブノーマル・MAX・大値を差し込む。
fn gen_data(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed);
    let mut v: Vec<f32> = (0..n).map(|_| r.next_f32()).collect();
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        1.0e-40,
        f32::MAX,
        1.0e30,
        -1.0e30,
    ];
    if n > 0 {
        for (k, s) in specials.iter().enumerate() {
            if n >= 16 || k < 2 {
                let idx = (k * 7 + seed as usize) % n;
                v[idx] = *s;
            }
        }
    }
    v
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn assert_same(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: len");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            same(*x, *y),
            "{ctx}[{i}]: {x} ({:#x}) vs {y} ({:#x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

fn run(n: usize, cfg0: &SgdStepConfig, steps: usize, nonzero_v: bool) {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut ref_p = gen_data(n, 1);
    let mut ref_v = if nonzero_v {
        gen_data(n, 3)
    } else {
        vec![0.0; n]
    };
    let up = |v: &[f32]| {
        mem.upload(&Tensor::new(v.to_vec(), &[v.len()]).unwrap())
            .unwrap()
    };
    let mut p_buf = up(&ref_p);
    let mut v_buf = up(&ref_v);
    for step in 0..steps {
        let g = gen_data(n, 10 + step as u64);
        let g_buf = up(&g);
        let mut cfg = *cfg0;
        if steps > 1 {
            cfg.is_first_step = step == 0;
        }
        reference_sgd_step_indexed(&mut ref_p, &g, Some(&mut ref_v[..]), &cfg);
        ops.sgd_step_device(&mut p_buf, &g_buf, Some(&mut v_buf), &cfg)
            .unwrap();
        let got_p = mem.download(&p_buf).unwrap();
        let got_v = mem.download(&v_buf).unwrap();
        let ctx = format!("n={n} step={step} cfg={cfg:?}");
        assert_same(got_p.as_slice().unwrap(), &ref_p, &format!("param {ctx}"));
        assert_same(
            got_v.as_slice().unwrap(),
            &ref_v,
            &format!("velocity {ctx}"),
        );
    }
}

#[test]
fn bit_identical_to_indexed_reference_matrix() {
    let lens = [0usize, 1, 3, 7, 8, 13, 16, 17, 31, 33, 1023, 203_530];
    for &n in &lens {
        // 大きい n は組合せを絞って実行時間を抑える。
        let wds: &[f32] = if n > 2000 {
            &[0.0, 0.01]
        } else {
            &[0.0, -0.0, 0.01]
        };
        for &momentum in &[0.0f32, 0.9] {
            for &dampening in &[0.0f32, 0.2] {
                for &weight_decay in wds {
                    for &nesterov in &[false, true] {
                        let cfg = SgdStepConfig {
                            lr: 0.1,
                            momentum,
                            dampening,
                            weight_decay,
                            nesterov,
                            is_first_step: true,
                        };
                        // 連続 3 ステップ（1 ステップ目のみ first）。
                        run(n, &cfg, 3, false);
                        // 非ゼロ初期 velocity・非 first の単発。
                        let mut c2 = cfg;
                        c2.is_first_step = false;
                        run(n, &c2, 1, true);
                    }
                }
            }
        }
    }
}

#[test]
fn velocity_untouched_when_momentum_is_zero() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let n = 33;
    let v0 = gen_data(n, 5);
    let up = |v: &[f32]| {
        mem.upload(&Tensor::new(v.to_vec(), &[v.len()]).unwrap())
            .unwrap()
    };
    let mut p = up(&gen_data(n, 1));
    let g = up(&gen_data(n, 2));
    let mut v = up(&v0);
    let cfg = SgdStepConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.2,
        weight_decay: 0.01,
        nesterov: true,
        is_first_step: false,
    };
    ops.sgd_step_device(&mut p, &g, Some(&mut v), &cfg).unwrap();
    let got = mem.download(&v).unwrap();
    assert_same(got.as_slice().unwrap(), &v0, "velocity");
}
