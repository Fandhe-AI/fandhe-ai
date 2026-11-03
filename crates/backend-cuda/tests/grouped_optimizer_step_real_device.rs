//! `CudaBackendOps::{sgd,adam}_step_device_grouped`（param groups 向け区間テーブル更新。
//! イシュー #3837・設計 `docs/autodiff-param-groups-decision.md` §19）の実機テスト。
//!
//! 実機（CUDA）依存のため `#[ignore]` 分離する。実測は #3862 が持つ（本ファイルは
//! 未実測のまま。Linux CI では実行しない）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test grouped_optimizer_step_real_device -- --ignored --nocapture
//! ```
//!
//! 合否は CPU の同一 grouped 呼び出しとの統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）。run-to-run の bit 同一は観測記録のみ（合否の基準にしない）。
//! CUDA の grouped は既存カーネルを区間ごとに部分ビューへ起動する構造のため、区間値で
//! 上書きしたスカラー版と構造的に同一の演算列になる。

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::{
    AdamStepConfig, AdamStepKind, AdamStepSegment, BackendError, BackendOps, LambMoments,
    SgdStepConfig, SgdStepSegment, Tensor,
};

const N: usize = 40;
const SPLIT: [usize; 3] = [13, 0, 27];

fn close(a: f32, b: f32) -> bool {
    fandhe_ai_tensor_core::tolerance::composite_close_f32(a, b)
}

fn vals(seed: f32) -> Vec<f32> {
    (0..N).map(|i| (seed + i as f32 * 0.37).sin()).collect()
}

/// capability は実機なしでも判定できる（ドライバに触れない）。
#[test]
fn cuda_reports_grouped_capability() {
    assert!(CudaBackendOps::new(0).supports_grouped_optimizer_step());
}

#[test]
#[ignore = "CUDA 実機（#3862 で実測）"]
fn sgd_grouped_matches_cpu_grouped() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let cuda = CudaBackendOps::new(dev.ordinal());
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let (cm, pm) = (cuda.memory_ops().unwrap(), cpu.memory_ops().unwrap());
    let lrs = [0.1f32, 0.5, 0.0];
    let wds = [0.01f32, 0.0, 0.05];
    let segs: Vec<SgdStepSegment> = SPLIT
        .iter()
        .enumerate()
        .map(|(i, &n)| SgdStepSegment {
            numel: n,
            lr: lrs[i],
            weight_decay: wds[i],
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
    let (mut cp, mut cv) = (
        cm.upload(&Tensor::new(p0.clone(), &[N]).unwrap()).unwrap(),
        cm.upload(&Tensor::new(v0.clone(), &[N]).unwrap()).unwrap(),
    );
    let (mut pp, mut pv) = (
        pm.upload(&Tensor::new(p0, &[N]).unwrap()).unwrap(),
        pm.upload(&Tensor::new(v0, &[N]).unwrap()).unwrap(),
    );
    for step in 0..3 {
        let g = Tensor::new(vals(2.0 + step as f32), &[N]).unwrap();
        let (cg, pg) = (cm.upload(&g).unwrap(), pm.upload(&g).unwrap());
        let cfg = SgdStepConfig {
            is_first_step: step == 0,
            ..cfg
        };
        cuda.sgd_step_device_grouped(&mut cp, &cg, Some(&mut cv), &cfg, &segs)
            .unwrap();
        cpu.sgd_step_device_grouped(&mut pp, &pg, Some(&mut pv), &cfg, &segs)
            .unwrap();
    }
    let (a, b) = (cm.download(&cp).unwrap(), pm.download(&pp).unwrap());
    for (x, y) in a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()) {
        assert!(close(*x, *y), "{x} vs {y}");
    }
}

#[test]
#[ignore = "CUDA 実機（#3862 で実測）"]
fn adam_grouped_matches_cpu_grouped_and_rejects_bad_totals() {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let cuda = CudaBackendOps::new(dev.ordinal());
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let (cm, pm) = (cuda.memory_ops().unwrap(), cpu.memory_ops().unwrap());
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
        let (mut cp, mut cmb, mut cvb) = (
            up(cm, &init),
            cm.alloc_zeroed(&[N]).unwrap(),
            cm.alloc_zeroed(&[N]).unwrap(),
        );
        let (mut pp, mut pmb, mut pvb) = (
            up(pm, &init),
            pm.alloc_zeroed(&[N]).unwrap(),
            pm.alloc_zeroed(&[N]).unwrap(),
        );
        let (cg, pg) = (cm.upload(&g).unwrap(), pm.upload(&g).unwrap());
        cuda.adam_step_device_grouped(
            &mut cp,
            &cg,
            LambMoments {
                m: &mut cmb,
                v: &mut cvb,
            },
            &cfg,
            &segs,
        )
        .unwrap();
        cpu.adam_step_device_grouped(
            &mut pp,
            &pg,
            LambMoments {
                m: &mut pmb,
                v: &mut pvb,
            },
            &cfg,
            &segs,
        )
        .unwrap();
        let (a, b) = (cm.download(&cp).unwrap(), pm.download(&pp).unwrap());
        for (x, y) in a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()) {
            assert!(close(*x, *y), "{kind:?}: {x} vs {y}");
        }
        // 合計長の不一致は driver 接触前に InvalidArgument（どの要素も更新しない）。
        let bad = [AdamStepSegment {
            numel: N - 1,
            step_size: 0.1,
            weight_decay: 0.0,
            decay_factor: 1.0,
        }];
        let before = cm.download(&cp).unwrap();
        let r = cuda.adam_step_device_grouped(
            &mut cp,
            &cg,
            LambMoments {
                m: &mut cmb,
                v: &mut cvb,
            },
            &cfg,
            &bad,
        );
        assert!(matches!(r, Err(BackendError::InvalidArgument(_))));
        assert_eq!(cm.download(&cp).unwrap().as_slice(), before.as_slice());
    }
}
