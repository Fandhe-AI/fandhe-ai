//! `MetalBackendOps::{sgd,adam}_step_device_grouped`（イシュー #3837・設計
//! `docs/autodiff-param-groups-decision.md` §19）の実機テスト。Metal 実機必須のため
//! `cfg(target_os = "macos")` かつ `#[ignore]`。実測は #3863 が持つ（本ファイルは未実測）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test grouped_optimizer_step_parity -- --ignored --nocapture
//! ```
//!
//! 合否は CPU の同一 grouped 呼び出しとの統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）。

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{
    AdamStepConfig, AdamStepKind, AdamStepSegment, BackendOps, LambMoments, SgdStepConfig,
    SgdStepSegment, Tensor,
};

const N: usize = 40;
const SPLIT: [usize; 3] = [13, 0, 27];

fn close(a: f32, b: f32) -> bool {
    let d = (a - b).abs();
    d < 1e-5 || d / b.abs().max(1e-12) < 1e-3
}

fn vals(seed: f32) -> Vec<f32> {
    (0..N).map(|i| (seed + i as f32 * 0.37).sin()).collect()
}

#[test]
#[ignore = "Metal 実機（#3863 で実測）"]
fn sgd_grouped_matches_cpu_grouped() {
    let metal = MetalBackendOps::new();
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let (mm, pm) = (metal.memory_ops().unwrap(), cpu.memory_ops().unwrap());
    assert!(metal.supports_grouped_optimizer_step());
    let segs: Vec<SgdStepSegment> = SPLIT
        .iter()
        .enumerate()
        .map(|(i, &n)| SgdStepSegment {
            numel: n,
            lr: [0.1f32, 0.5, 0.0][i],
            weight_decay: [0.01f32, 0.0, 0.05][i],
        })
        .collect();
    let cfg = SgdStepConfig {
        lr: 0.3,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.02,
        nesterov: true,
        is_first_step: true,
    };
    let (p0, v0) = (vals(1.0), vals(3.0));
    let up = |m: &dyn fandhe_ai_tensor_core::MemoryOps, v: &[f32]| {
        m.upload(&Tensor::new(v.to_vec(), &[N]).unwrap()).unwrap()
    };
    let (mut mp, mut mv) = (up(mm, &p0), up(mm, &v0));
    let (mut pp, mut pv) = (up(pm, &p0), up(pm, &v0));
    for step in 0..3 {
        let g = Tensor::new(vals(2.0 + step as f32), &[N]).unwrap();
        let (mg, pg) = (mm.upload(&g).unwrap(), pm.upload(&g).unwrap());
        let cfg = SgdStepConfig {
            is_first_step: step == 0,
            ..cfg
        };
        metal
            .sgd_step_device_grouped(&mut mp, &mg, Some(&mut mv), &cfg, &segs)
            .unwrap();
        cpu.sgd_step_device_grouped(&mut pp, &pg, Some(&mut pv), &cfg, &segs)
            .unwrap();
    }
    let (a, b) = (mm.download(&mp).unwrap(), pm.download(&pp).unwrap());
    for (x, y) in a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()) {
        assert!(close(*x, *y), "{x} vs {y}");
    }
}

#[test]
#[ignore = "Metal 実機（#3863 で実測）"]
fn adam_grouped_matches_cpu_grouped() {
    let metal = MetalBackendOps::new();
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let (mm, pm) = (metal.memory_ops().unwrap(), cpu.memory_ops().unwrap());
    for kind in [AdamStepKind::Coupled, AdamStepKind::Decoupled] {
        let cfg = AdamStepConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            decay_factor: 0.999,
            step_size: 0.05,
            bias_correction2_sqrt: 0.7,
            kind,
        };
        let segs: Vec<AdamStepSegment> = SPLIT
            .iter()
            .enumerate()
            .map(|(i, &n)| AdamStepSegment {
                numel: n,
                step_size: [0.05f32, 0.0, 0.2][i],
                weight_decay: [0.01f32, 0.0, 0.1][i],
                decay_factor: [0.999f32, 1.0, 0.99][i],
            })
            .collect();
        let init = vals(1.0);
        let g = Tensor::new(vals(2.0), &[N]).unwrap();
        let up = |m: &dyn fandhe_ai_tensor_core::MemoryOps, v: &[f32]| {
            m.upload(&Tensor::new(v.to_vec(), &[N]).unwrap()).unwrap()
        };
        let (mut mp, mut mm1, mut mv1) = (
            up(mm, &init),
            mm.alloc_zeroed(&[N]).unwrap(),
            mm.alloc_zeroed(&[N]).unwrap(),
        );
        let (mut pp, mut pm1, mut pv1) = (
            up(pm, &init),
            pm.alloc_zeroed(&[N]).unwrap(),
            pm.alloc_zeroed(&[N]).unwrap(),
        );
        let (mg, pg) = (mm.upload(&g).unwrap(), pm.upload(&g).unwrap());
        metal
            .adam_step_device_grouped(
                &mut mp,
                &mg,
                LambMoments {
                    m: &mut mm1,
                    v: &mut mv1,
                },
                &cfg,
                &segs,
            )
            .unwrap();
        cpu.adam_step_device_grouped(
            &mut pp,
            &pg,
            LambMoments {
                m: &mut pm1,
                v: &mut pv1,
            },
            &cfg,
            &segs,
        )
        .unwrap();
        let (a, b) = (mm.download(&mp).unwrap(), pm.download(&pp).unwrap());
        for (x, y) in a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()) {
            assert!(close(*x, *y), "{kind:?}: {x} vs {y}");
        }
    }
}
