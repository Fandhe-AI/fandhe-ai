//! `CpuBackendOps::*_step_device_grouped`（param groups 向け区間テーブル更新。
//! イシュー #3837・設計 `docs/autodiff-param-groups-decision.md` §19）の
//! 数値契約テスト。
//!
//! 契約: 区間 s の結果は「既存スカラーカーネルを、config の該当フィールドを
//! 区間の値で上書きして区間の部分範囲へ適用した結果」と **bit 完全一致**する
//! （CPU 実装は同じスライス関数を区間ごとに呼ぶ構造のため）。ここでは
//! 区間ごとに別バッファへ切り出したスカラー呼び出しを参照として突合する。

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{
    AdagradStepConfig, AdagradStepSegment, AdamStepConfig, AdamStepKind, AdamStepSegment,
    BackendError, BackendOps, DeviceBuffer, LambMoments, LambStepConfig, LambStepSegment,
    MemoryOps, RmsPropOptionalBuffers, RmsPropStepConfig, RmsPropStepSegment, SgdStepConfig,
    SgdStepSegment, Tensor,
};

const N: usize = 10;
const SPLIT: [usize; 3] = [3, 0, 7];

fn vals(seed: f32) -> Vec<f32> {
    (0..N).map(|i| (seed + i as f32 * 0.37).sin()).collect()
}

fn up(mem: &dyn MemoryOps, v: &[f32]) -> DeviceBuffer<f32> {
    mem.upload(&Tensor::new(v.to_vec(), &[v.len()]).unwrap())
        .unwrap()
}

fn down(mem: &dyn MemoryOps, b: &DeviceBuffer<f32>) -> Vec<f32> {
    mem.download(b).unwrap().as_slice().unwrap().to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn ranges() -> Vec<(usize, usize)> {
    let mut off = 0;
    SPLIT
        .iter()
        .map(|&n| {
            let r = (off, off + n);
            off += n;
            r
        })
        .collect()
}

#[test]
fn sgd_grouped_matches_per_segment_scalar_bits() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let lrs = [0.1f32, 0.5, 0.0];
    let wds = [0.01f32, 0.0, -0.0];
    for (first, momentum, nesterov) in [
        (true, 0.9f32, true),
        (false, 0.9, false),
        (true, 0.0, false),
    ] {
        let base = SgdStepConfig {
            lr: 0.3,
            momentum,
            dampening: 0.0,
            weight_decay: 0.02,
            nesterov,
            is_first_step: first,
        };
        let (p0, g0, v0) = (vals(1.0), vals(2.0), vals(3.0));
        let mut p = up(mem, &p0);
        let g = up(mem, &g0);
        let mut v = up(mem, &v0);
        let segs: Vec<SgdStepSegment> = SPLIT
            .iter()
            .enumerate()
            .map(|(i, &n)| SgdStepSegment {
                numel: n,
                lr: lrs[i],
                weight_decay: wds[i],
            })
            .collect();
        let vel = if momentum != 0.0 { Some(&mut v) } else { None };
        ops.sgd_step_device_grouped(&mut p, &g, vel, &base, &segs)
            .unwrap();
        let mut expect_p = p0.clone();
        let mut expect_v = v0.clone();
        for (i, (a, b)) in ranges().into_iter().enumerate() {
            if a == b {
                continue;
            }
            let cfg = SgdStepConfig {
                lr: lrs[i],
                weight_decay: wds[i],
                ..base
            };
            let mut sp = up(mem, &p0[a..b]);
            let sg = up(mem, &g0[a..b]);
            let mut sv = up(mem, &v0[a..b]);
            let svel = if momentum != 0.0 { Some(&mut sv) } else { None };
            ops.sgd_step_device(&mut sp, &sg, svel, &cfg).unwrap();
            expect_p[a..b].copy_from_slice(&down(mem, &sp));
            if momentum != 0.0 {
                expect_v[a..b].copy_from_slice(&down(mem, &sv));
            }
        }
        assert_eq!(bits(&down(mem, &p)), bits(&expect_p));
        assert_eq!(bits(&down(mem, &v)), bits(&expect_v));
    }
}

#[test]
fn sgd_grouped_single_segment_equals_scalar() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let cfg = SgdStepConfig {
        lr: 0.2,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.1,
        nesterov: false,
        is_first_step: true,
    };
    let (p0, g0) = (vals(1.0), vals(2.0));
    let (mut a, mut b) = (up(mem, &p0), up(mem, &p0));
    let g = up(mem, &g0);
    ops.sgd_step_device(&mut a, &g, None, &cfg).unwrap();
    let seg = [SgdStepSegment {
        numel: N,
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
    }];
    ops.sgd_step_device_grouped(&mut b, &g, None, &cfg, &seg)
        .unwrap();
    assert_eq!(bits(&down(mem, &a)), bits(&down(mem, &b)));
}

#[test]
fn adam_grouped_matches_per_segment_scalar_bits() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    for kind in [AdamStepKind::Coupled, AdamStepKind::Decoupled] {
        let base = AdamStepConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            decay_factor: 0.999,
            step_size: 0.05,
            bias_correction2_sqrt: 0.7,
            kind,
        };
        let sizes = [0.05f32, 0.0, 0.2];
        let wds = [0.01f32, 0.0, 0.1];
        let dfs = [0.999f32, 1.0, 0.99];
        let (p0, g0, m0, v0) = (
            vals(1.0),
            vals(2.0),
            vals(3.0),
            vals(4.0).iter().map(|x| x.abs()).collect::<Vec<_>>(),
        );
        let mut p = up(mem, &p0);
        let g = up(mem, &g0);
        let (mut m, mut v) = (up(mem, &m0), up(mem, &v0));
        let segs: Vec<AdamStepSegment> = SPLIT
            .iter()
            .enumerate()
            .map(|(i, &n)| AdamStepSegment {
                numel: n,
                step_size: sizes[i],
                weight_decay: wds[i],
                decay_factor: dfs[i],
            })
            .collect();
        ops.adam_step_device_grouped(
            &mut p,
            &g,
            LambMoments {
                m: &mut m,
                v: &mut v,
            },
            &base,
            &segs,
        )
        .unwrap();
        let (mut ep, mut em, mut ev) = (p0.clone(), m0.clone(), v0.clone());
        for (i, (a, b)) in ranges().into_iter().enumerate() {
            if a == b {
                continue;
            }
            let cfg = AdamStepConfig {
                step_size: sizes[i],
                weight_decay: wds[i],
                decay_factor: dfs[i],
                ..base
            };
            let (mut sp, sg) = (up(mem, &p0[a..b]), up(mem, &g0[a..b]));
            let (mut sm, mut sv) = (up(mem, &m0[a..b]), up(mem, &v0[a..b]));
            ops.adam_step_device(&mut sp, &sg, &mut sm, &mut sv, &cfg)
                .unwrap();
            ep[a..b].copy_from_slice(&down(mem, &sp));
            em[a..b].copy_from_slice(&down(mem, &sm));
            ev[a..b].copy_from_slice(&down(mem, &sv));
        }
        assert_eq!(bits(&down(mem, &p)), bits(&ep));
        assert_eq!(bits(&down(mem, &m)), bits(&em));
        assert_eq!(bits(&down(mem, &v)), bits(&ev));
    }
}

#[test]
fn rmsprop_grouped_matches_per_segment_scalar_bits() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let lrs = [0.01f32, 0.1, 0.0];
    let wds = [0.0f32, 0.05, 0.01];
    let base = RmsPropStepConfig {
        lr: 0.5,
        alpha: 0.99,
        eps: 1e-8,
        weight_decay: 0.3,
        momentum: 0.9,
        centered: true,
    };
    let abs = |s: f32| vals(s).iter().map(|x| x.abs() + 0.5).collect::<Vec<_>>();
    let (p0, g0, sq0, ga0, mo0) = (vals(1.0), vals(2.0), abs(3.0), vals(4.0), vals(5.0));
    let mut p = up(mem, &p0);
    let g = up(mem, &g0);
    let (mut sq, mut ga, mut mo) = (up(mem, &sq0), up(mem, &ga0), up(mem, &mo0));
    let segs: Vec<RmsPropStepSegment> = SPLIT
        .iter()
        .enumerate()
        .map(|(i, &n)| RmsPropStepSegment {
            numel: n,
            lr: lrs[i],
            weight_decay: wds[i],
        })
        .collect();
    ops.rmsprop_step_device_grouped(
        &mut p,
        &g,
        &mut sq,
        RmsPropOptionalBuffers {
            grad_avg: Some(&mut ga),
            momentum_buf: Some(&mut mo),
        },
        &base,
        &segs,
    )
    .unwrap();
    let mut ep = p0.clone();
    for (i, (a, b)) in ranges().into_iter().enumerate() {
        if a == b {
            continue;
        }
        let cfg = RmsPropStepConfig {
            lr: lrs[i],
            weight_decay: wds[i],
            ..base
        };
        let (mut sp, sg, mut ssq) = (up(mem, &p0[a..b]), up(mem, &g0[a..b]), up(mem, &sq0[a..b]));
        let (mut sga, mut smo) = (up(mem, &ga0[a..b]), up(mem, &mo0[a..b]));
        ops.rmsprop_step_device(
            &mut sp,
            &sg,
            &mut ssq,
            RmsPropOptionalBuffers {
                grad_avg: Some(&mut sga),
                momentum_buf: Some(&mut smo),
            },
            &cfg,
        )
        .unwrap();
        ep[a..b].copy_from_slice(&down(mem, &sp));
    }
    assert_eq!(bits(&down(mem, &p)), bits(&ep));
}

#[test]
fn adagrad_grouped_matches_per_segment_scalar_bits() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let clrs = [0.01f32, 0.1, 0.5];
    let wds = [0.0f32, 0.05, 0.01];
    let base = AdagradStepConfig {
        clr: 0.2,
        eps: 1e-10,
        weight_decay: 0.3,
    };
    let (p0, g0) = (vals(1.0), vals(2.0));
    let ss0: Vec<f32> = vals(3.0).iter().map(|x| x.abs()).collect();
    let mut p = up(mem, &p0);
    let g = up(mem, &g0);
    let mut ss = up(mem, &ss0);
    let segs: Vec<AdagradStepSegment> = SPLIT
        .iter()
        .enumerate()
        .map(|(i, &n)| AdagradStepSegment {
            numel: n,
            clr: clrs[i],
            weight_decay: wds[i],
        })
        .collect();
    ops.adagrad_step_device_grouped(&mut p, &g, &mut ss, &base, &segs)
        .unwrap();
    let mut ep = p0.clone();
    for (i, (a, b)) in ranges().into_iter().enumerate() {
        if a == b {
            continue;
        }
        let cfg = AdagradStepConfig {
            clr: clrs[i],
            weight_decay: wds[i],
            ..base
        };
        let (mut sp, sg, mut sss) = (up(mem, &p0[a..b]), up(mem, &g0[a..b]), up(mem, &ss0[a..b]));
        ops.adagrad_step_device(&mut sp, &sg, &mut sss, &cfg)
            .unwrap();
        ep[a..b].copy_from_slice(&down(mem, &sp));
    }
    assert_eq!(bits(&down(mem, &p)), bits(&ep));
}

#[test]
fn lamb_grouped_matches_per_layer_scalar_bits() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let lrs = [0.01f32, 0.1, 0.05];
    let sizes = [0.01f32, 0.1, 0.05];
    let wds = [0.0f32, 0.05, 0.01];
    let base = LambStepConfig {
        lr: 0.5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.3,
        step_size: 0.5,
        bias_correction2_sqrt: 0.6,
    };
    let (p0, g0, m0) = (vals(1.0), vals(2.0), vals(3.0));
    let v0: Vec<f32> = vals(4.0).iter().map(|x| x.abs()).collect();
    let mut p = up(mem, &p0);
    let g = up(mem, &g0);
    let (mut m, mut v) = (up(mem, &m0), up(mem, &v0));
    let segs: Vec<LambStepSegment> = SPLIT
        .iter()
        .enumerate()
        .map(|(i, &n)| LambStepSegment {
            numel: n,
            lr: lrs[i],
            step_size: sizes[i],
            weight_decay: wds[i],
        })
        .collect();
    ops.lamb_step_device_grouped(
        &mut p,
        &g,
        LambMoments {
            m: &mut m,
            v: &mut v,
        },
        &base,
        &segs,
    )
    .unwrap();
    let mut ep = p0.clone();
    for (i, (a, b)) in ranges().into_iter().enumerate() {
        if a == b {
            continue;
        }
        let cfg = LambStepConfig {
            lr: lrs[i],
            step_size: sizes[i],
            weight_decay: wds[i],
            ..base
        };
        let (mut sp, sg) = (up(mem, &p0[a..b]), up(mem, &g0[a..b]));
        let (mut sm, mut sv) = (up(mem, &m0[a..b]), up(mem, &v0[a..b]));
        ops.lamb_step_device(
            &mut sp,
            &sg,
            LambMoments {
                m: &mut sm,
                v: &mut sv,
            },
            &[b - a],
            &cfg,
        )
        .unwrap();
        ep[a..b].copy_from_slice(&down(mem, &sp));
    }
    assert_eq!(bits(&down(mem, &p)), bits(&ep));
}

#[test]
fn invalid_segment_totals_are_rejected_before_any_write() {
    let ops = CpuBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let cfg = SgdStepConfig {
        lr: 0.2,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    let p0 = vals(1.0);
    let mut p = up(mem, &p0);
    let g = up(mem, &vals(2.0));
    let seg = |n: usize| SgdStepSegment {
        numel: n,
        lr: 1.0,
        weight_decay: 0.0,
    };
    for segs in [
        vec![seg(N - 1)],
        vec![seg(N + 1)],
        vec![seg(usize::MAX), seg(2)],
    ] {
        let r = ops.sgd_step_device_grouped(&mut p, &g, None, &cfg, &segs);
        assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
        assert_eq!(bits(&down(mem, &p)), bits(&p0));
    }
    assert!(ops.supports_grouped_optimizer_step());
}
