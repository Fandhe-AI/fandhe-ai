//! `RocmBackendOps` の `sgd_step_device`／`adam_step_device` の結線（イシュー #3440・分解案 R9）。
//!
//! 役割: `ops.rs` の `BackendOps` impl から 1 行で呼ばれ、(1) ハンドル型・shape・要素数・
//! momentum と velocity の組・未知の `AdamStepKind`・`i32` 収容の検証、(2) engine（Hip／テストの
//! HostModel）の分岐、を担う。カーネル起動そのものは `optim.rs` の `RocmOptim`。
//! `*_tracked` は trait 既定の委譲（非 tracked 版を呼ぶ）のままにする（被覆計画 §4。同期実行の
//! ため `DispatchFailureCell` の結線は不要）。
//!
//! # 検証の順序（fail-closed・部分更新を残さない）
//! すべての検証を **どの要素も書き換える前・HIP に触れる前** に終える。`device()` は #3157 まで
//! `Device::Cpu` の仮置きで判別に使えないため、`ops/ew.rs` と同じくハンドル型の downcast で判別する。
//! - 手順: downcast（失敗は `DeviceMismatch`）、shape 一致、ハンドルの要素数 = `numel`、
//!   SGD の momentum 有効かつ velocity なし・Adam の未知 kind は `Unsupported`、`numel` の `i32` 収容、
//!   `numel == 0` は HIP に触れず `Ok`、最後に engine とハンドル種別の組み合わせ
//!   （取り違えは `DeviceMismatch`）。
//!
//! # デバイス常駐 API の到達性
//! `memory_ops()` が `None` のままなので crate の外から ROCm の `DeviceBuffer` を作れない
//! （`ops/ew.rs` と同じ制約。被覆計画 §9）。テストは `upload_resident`／`download_resident` を使う。

use fandhe_ai_tensor_core::{
    AdamStepConfig, BackendError, DeviceBuffer, SgdStepConfig, ShapeError,
};

use super::{Engine, OPTIM_CACHE, RocmBackendOps, cached};
use crate::buffer_handle::RocmBufferHandle;
use crate::elementwise::fit_i32;
use crate::error::to_backend_error;
use crate::optim::{AdamKernelParams, RocmOptim, SgdKernelParams};

const OP_SGD: &str = "sgd_step";
const OP_ADAM: &str = "adam_step";

fn handle_of(buf: &DeviceBuffer<f32>) -> Result<&RocmBufferHandle, BackendError> {
    buf.downcast_handle::<RocmBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)
}

fn handle_mut_of(buf: &mut DeviceBuffer<f32>) -> Result<&mut RocmBufferHandle, BackendError> {
    buf.downcast_handle_mut::<RocmBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)
}

fn shape_mismatch(lhs: &[usize], rhs: &[usize]) -> BackendError {
    BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    })
}

/// ハンドル型の確認と shape・要素数の検証（読み取りのみ）。`param` を基準にする。
fn check_buffer(param: &DeviceBuffer<f32>, other: &DeviceBuffer<f32>) -> Result<(), BackendError> {
    let h = handle_of(other)?;
    if other.shape() != param.shape() {
        return Err(shape_mismatch(param.shape(), other.shape()));
    }
    if h.len() != param.numel() {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: param.numel(),
                actual: h.len(),
            },
        ));
    }
    Ok(())
}

impl RocmBackendOps {
    /// `sgd_step_device`。CPU 参照（`CpuBackendOps::sgd_step_device`）と同じ演算列で in-place 更新する。
    pub(super) fn optim_sgd_step(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        velocity: Option<&mut DeviceBuffer<f32>>,
        config: &SgdStepConfig,
    ) -> Result<(), BackendError> {
        // --- 検証（読み取りのみ。書き換え・HIP 呼び出しより前） ---
        handle_of(param)?;
        check_buffer(param, grad)?;
        if let Some(v) = velocity.as_deref() {
            check_buffer(param, v)?;
        }
        // 長さの照合は param のハンドルにも行う（shape と要素数の整合）。
        check_buffer(param, param)?;
        let params = SgdKernelParams::from_config(config);
        if params.use_momentum && velocity.is_none() {
            return Err(BackendError::Unsupported(
                "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
            ));
        }
        let numel = param.numel();
        fit_i32(OP_SGD, numel, "numel").map_err(to_backend_error)?;
        if numel == 0 {
            return Ok(()); // HIP を触らない。
        }

        // --- 更新（ここから書き込み。engine とハンドル種別の取り違えは書く前に DeviceMismatch） ---
        let vh = match velocity {
            Some(v) => Some(handle_mut_of(v)?),
            None => None,
        };
        let ph = handle_mut_of(param)?;
        let gh = handle_of(grad)?;
        match (self.engine, ph, gh) {
            (Engine::Hip, RocmBufferHandle::Hip(p), RocmBufferHandle::Hip(g)) => {
                let v = match vh {
                    Some(RocmBufferHandle::Hip(b)) => Some(b),
                    Some(_) => return Err(BackendError::DeviceMismatch),
                    None => None,
                };
                let o =
                    cached(&OPTIM_CACHE, self.target, RocmOptim::new).map_err(to_backend_error)?;
                o.sgd_step_resident(p, g, v, &params)
                    .map_err(to_backend_error)
            }
            #[cfg(test)]
            (Engine::HostModel, RocmBufferHandle::Host(p), RocmBufferHandle::Host(g)) => {
                let v = match vh {
                    Some(RocmBufferHandle::Host(b)) => Some(b.as_mut_slice()),
                    Some(_) => return Err(BackendError::DeviceMismatch),
                    None => None,
                };
                crate::optim_model::sgd_step(p, g, v, &params);
                Ok(())
            }
            _ => Err(BackendError::DeviceMismatch),
        }
    }

    /// `adam_step_device`。CPU 参照（`CpuBackendOps::adam_step_device`）と同じ演算列で in-place 更新する。
    pub(super) fn optim_adam_step(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        m: &mut DeviceBuffer<f32>,
        v: &mut DeviceBuffer<f32>,
        config: &AdamStepConfig,
    ) -> Result<(), BackendError> {
        // --- 検証（読み取りのみ） ---
        check_buffer(param, param)?;
        check_buffer(param, grad)?;
        check_buffer(param, m)?;
        check_buffer(param, v)?;
        // `AdamStepKind` は `#[non_exhaustive]`。未知 variant は書き換え前に fail-closed で拒否する。
        let params = AdamKernelParams::from_config(config).ok_or_else(|| {
            BackendError::Unsupported("adam_step_device: unknown AdamStepKind variant".into())
        })?;
        let numel = param.numel();
        fit_i32(OP_ADAM, numel, "numel").map_err(to_backend_error)?;
        if numel == 0 {
            return Ok(()); // HIP を触らない。
        }

        // --- 更新 ---
        let mh = handle_mut_of(m)?;
        let vh = handle_mut_of(v)?;
        let ph = handle_mut_of(param)?;
        let gh = handle_of(grad)?;
        match (self.engine, ph, gh, mh, vh) {
            (
                Engine::Hip,
                RocmBufferHandle::Hip(p),
                RocmBufferHandle::Hip(g),
                RocmBufferHandle::Hip(mm),
                RocmBufferHandle::Hip(vv),
            ) => {
                let o =
                    cached(&OPTIM_CACHE, self.target, RocmOptim::new).map_err(to_backend_error)?;
                o.adam_step_resident(p, g, mm, vv, &params)
                    .map_err(to_backend_error)
            }
            #[cfg(test)]
            (
                Engine::HostModel,
                RocmBufferHandle::Host(p),
                RocmBufferHandle::Host(g),
                RocmBufferHandle::Host(mm),
                RocmBufferHandle::Host(vv),
            ) => {
                crate::optim_model::adam_step(p, g, mm, vv, &params);
                Ok(())
            }
            _ => Err(BackendError::DeviceMismatch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::GfxTarget;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{AdamStepKind, BackendOps, MemoryOps, Tensor};

    fn ops() -> RocmBackendOps {
        RocmBackendOps::with_host_model(GfxTarget::Gfx942)
    }

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let r = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 10_001;
                -1.0 + 2.0 * (r as f32 / 10_000.0)
            })
            .collect()
    }

    fn t(len: usize, seed: u32) -> Tensor<f32> {
        Tensor::new(data(len, seed), &[len]).unwrap()
    }

    fn bits(x: &Tensor<f32>) -> Vec<u32> {
        x.as_slice().unwrap().iter().map(|v| v.to_bits()).collect()
    }

    fn cpu_mem() -> (&'static CpuBackendOps, &'static dyn MemoryOps) {
        // テスト専用: `memory_ops` の借用を返すため、CPU 参照 ops を leak して 'static にする。
        let cpu: &'static CpuBackendOps = Box::leak(Box::new(CpuBackendOps::new()));
        let mem = cpu.memory_ops().expect("CpuBackendOps は MemoryOps を持つ");
        (cpu, mem)
    }

    fn sgd_base(momentum: f32, dampening: f32, weight_decay: f32, nesterov: bool) -> SgdStepConfig {
        SgdStepConfig {
            lr: 0.05,
            momentum,
            dampening,
            weight_decay,
            nesterov,
            is_first_step: true,
        }
    }

    /// ROCm（ホストモデル）と CPU 参照を同じ入力で `steps` 回進め、param・velocity をビット比較する。
    fn run_sgd(n: usize, base: SgdStepConfig, pass_velocity: bool, steps: usize) {
        let (o, (cpu, mem)) = (ops(), cpu_mem());
        let mut rp = o.upload_resident(&t(n, 1)).unwrap();
        let mut cp = mem.upload(&t(n, 1)).unwrap();
        let mut rv = o.upload_resident(&t(n, 2)).unwrap();
        let mut cv = mem.upload(&t(n, 2)).unwrap();
        for s in 0..steps {
            let g = t(n, 10 + s as u32);
            let (rg, cg) = (o.upload_resident(&g).unwrap(), mem.upload(&g).unwrap());
            let cfg = SgdStepConfig {
                is_first_step: s == 0,
                ..base
            };
            o.sgd_step_device(&mut rp, &rg, pass_velocity.then_some(&mut rv), &cfg)
                .unwrap();
            cpu.sgd_step_device(&mut cp, &cg, pass_velocity.then_some(&mut cv), &cfg)
                .unwrap();
            let (a, b) = (
                o.download_resident(&rp).unwrap(),
                mem.download(&cp).unwrap(),
            );
            assert_eq!(bits(&a), bits(&b), "param step {s} n={n} {base:?}");
        }
        let (a, b) = (
            o.download_resident(&rv).unwrap(),
            mem.download(&cv).unwrap(),
        );
        assert_eq!(bits(&a), bits(&b), "velocity n={n} {base:?}");
    }

    #[test]
    fn sgd_matches_cpu_bitwise() {
        for n in [1usize, 255, 256, 257, 1000] {
            for momentum in [0.0f32, 0.9] {
                for nesterov in [false, true] {
                    for dampening in [0.0f32, 0.1] {
                        for wd in [0.0f32, 1e-2] {
                            run_sgd(n, sgd_base(momentum, dampening, wd, nesterov), true, 4);
                        }
                    }
                }
            }
        }
    }

    /// momentum 0 のときは velocity が渡されても読まず書かない（CUDA の `velocity.is_some()` 基準ではなく
    /// CPU と trait doc に合わせる。退行防止）。dampening != 0 で食い違いが顕在化する。
    #[test]
    fn sgd_zero_momentum_ignores_velocity_even_with_dampening() {
        let n = 257;
        run_sgd(n, sgd_base(0.0, 0.3, 1e-2, false), true, 3);
        // velocity が書き換わっていないことの直接確認。
        let o = ops();
        let mut p = o.upload_resident(&t(n, 1)).unwrap();
        let g = o.upload_resident(&t(n, 3)).unwrap();
        let mut v = o.upload_resident(&t(n, 2)).unwrap();
        o.sgd_step_device(&mut p, &g, Some(&mut v), &sgd_base(0.0, 0.3, 0.0, false))
            .unwrap();
        assert_eq!(bits(&o.download_resident(&v).unwrap()), bits(&t(n, 2)));
    }

    #[test]
    fn sgd_without_velocity_matches_cpu_when_momentum_off() {
        run_sgd(300, sgd_base(0.0, 0.0, 1e-2, false), false, 3);
    }

    fn adam_cfg(kind: AdamStepKind, wd: f32, lr: f32, t_step: i32) -> AdamStepConfig {
        let (beta1, beta2) = (0.9f32, 0.999f32);
        let bc1 = 1.0 - (beta1 as f64).powi(t_step);
        let bc2 = 1.0 - (beta2 as f64).powi(t_step);
        AdamStepConfig {
            beta1,
            beta2,
            eps: 1e-8,
            weight_decay: wd,
            decay_factor: match kind {
                AdamStepKind::Decoupled => 1.0 - lr * wd,
                _ => 1.0,
            },
            step_size: (lr as f64 / bc1) as f32,
            bias_correction2_sqrt: bc2.sqrt() as f32,
            kind,
        }
    }

    #[test]
    fn adam_matches_cpu_bitwise() {
        let (o, (cpu, mem)) = (ops(), cpu_mem());
        for n in [1usize, 255, 256, 257, 1000] {
            for kind in [AdamStepKind::Coupled, AdamStepKind::Decoupled] {
                for wd in [0.0f32, 1e-2] {
                    let mut rb = [1u32, 2, 3].map(|s| o.upload_resident(&t(n, s)).unwrap());
                    let mut cb = [1u32, 2, 3].map(|s| mem.upload(&t(n, s)).unwrap());
                    // m・v は非負にする（v < 0 の sqrt が NaN になるのを避ける）。
                    let pos = |s: u32| {
                        Tensor::new(data(n, s).iter().map(|x| x.abs()).collect(), &[n]).unwrap()
                    };
                    rb[1] = o.upload_resident(&pos(2)).unwrap();
                    rb[2] = o.upload_resident(&pos(3)).unwrap();
                    cb[1] = mem.upload(&pos(2)).unwrap();
                    cb[2] = mem.upload(&pos(3)).unwrap();
                    for s in 0..4 {
                        let g = t(n, 20 + s as u32);
                        let (rg, cg) = (o.upload_resident(&g).unwrap(), mem.upload(&g).unwrap());
                        let cfg = adam_cfg(kind, wd, 1e-3, s + 1);
                        let [rp, rm, rv] = &mut rb;
                        o.adam_step_device(rp, &rg, rm, rv, &cfg).unwrap();
                        let [cp, cm, cv] = &mut cb;
                        cpu.adam_step_device(cp, &cg, cm, cv, &cfg).unwrap();
                        for (r, c, name) in [
                            (&rb[0], &cb[0], "param"),
                            (&rb[1], &cb[1], "m"),
                            (&rb[2], &cb[2], "v"),
                        ] {
                            assert_eq!(
                                bits(&o.download_resident(r).unwrap()),
                                bits(&mem.download(c).unwrap()),
                                "{name} step {s} n={n} {kind:?} wd={wd}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn errors_are_typed_and_leave_buffers_untouched() {
        let o = ops();
        let n = 8;
        let (cpu, mem) = cpu_mem();
        let _ = cpu;
        let mut p = o.upload_resident(&t(n, 1)).unwrap();
        let g = o.upload_resident(&t(n, 2)).unwrap();
        let before = bits(&o.download_resident(&p).unwrap());

        // shape 不一致。
        let g_bad = o.upload_resident(&t(n + 1, 2)).unwrap();
        assert!(matches!(
            o.sgd_step_device(&mut p, &g_bad, None, &sgd_base(0.0, 0.0, 0.0, false)),
            Err(BackendError::ShapeMismatch(_))
        ));
        // momentum 有効で velocity なし。
        assert!(matches!(
            o.sgd_step_device(&mut p, &g, None, &sgd_base(0.9, 0.0, 0.0, false)),
            Err(BackendError::Unsupported(_))
        ));
        // CPU 由来のハンドル。
        let g_cpu = mem.upload(&t(n, 2)).unwrap();
        assert!(matches!(
            o.sgd_step_device(&mut p, &g_cpu, None, &sgd_base(0.0, 0.0, 0.0, false)),
            Err(BackendError::DeviceMismatch)
        ));
        let mut m = o.upload_resident(&t(n, 3)).unwrap();
        let mut v_cpu = mem.upload(&t(n, 4)).unwrap();
        assert!(matches!(
            o.adam_step_device(
                &mut p,
                &g,
                &mut m,
                &mut v_cpu,
                &adam_cfg(AdamStepKind::Coupled, 0.0, 1e-3, 1)
            ),
            Err(BackendError::DeviceMismatch)
        ));
        // shape 不一致（m）。
        let mut m_bad = o.upload_resident(&t(n + 1, 3)).unwrap();
        let mut v = o.upload_resident(&t(n, 4)).unwrap();
        assert!(matches!(
            o.adam_step_device(
                &mut p,
                &g,
                &mut m_bad,
                &mut v,
                &adam_cfg(AdamStepKind::Coupled, 0.0, 1e-3, 1)
            ),
            Err(BackendError::ShapeMismatch(_))
        ));
        // 未知の `AdamStepKind` は `#[non_exhaustive]` のため crate の外・内とも値を作れず、
        // `_ =>` 分岐（`AdamKernelParams::from_config` の `None`）はテストできない。コードで残す。

        assert_eq!(bits(&o.download_resident(&p).unwrap()), before);
    }

    #[test]
    fn hip_engine_with_host_handles_is_device_mismatch_and_empty_is_ok() {
        let o = RocmBackendOps::new(GfxTarget::Gfx942);
        let h = ops(); // Host ハンドルを作る用。
        let mut p = h.upload_resident(&t(4, 1)).unwrap();
        let g = h.upload_resident(&t(4, 2)).unwrap();
        assert!(matches!(
            o.sgd_step_device(&mut p, &g, None, &sgd_base(0.0, 0.0, 0.0, false)),
            Err(BackendError::DeviceMismatch)
        ));
        // numel == 0 は Hip engine（ROCm 非搭載でも）で HIP に触れず Ok。
        let e = Tensor::<f32>::zeros(&[0]).unwrap();
        // Hip engine の upload は空なら Empty ハンドルを作る（HIP に触れない）。
        let mut ep = o.upload_resident(&e).unwrap();
        let eg = o.upload_resident(&e).unwrap();
        let mut em = o.upload_resident(&e).unwrap();
        let mut ev = o.upload_resident(&e).unwrap();
        o.sgd_step_device(&mut ep, &eg, None, &sgd_base(0.0, 0.0, 0.0, false))
            .unwrap();
        o.adam_step_device(
            &mut ep,
            &eg,
            &mut em,
            &mut ev,
            &adam_cfg(AdamStepKind::Decoupled, 1e-2, 1e-3, 1),
        )
        .unwrap();
    }

    #[test]
    fn tracked_variants_delegate_to_the_untracked_path() {
        use fandhe_ai_tensor_core::DispatchFailureCell;
        let (o, n) = (ops(), 33);
        let cell = DispatchFailureCell::new();
        let (mut a, mut b) = (
            o.upload_resident(&t(n, 1)).unwrap(),
            o.upload_resident(&t(n, 1)).unwrap(),
        );
        let g = o.upload_resident(&t(n, 2)).unwrap();
        let cfg = sgd_base(0.0, 0.0, 1e-2, false);
        o.sgd_step_device(&mut a, &g, None, &cfg).unwrap();
        o.sgd_step_device_tracked(&mut b, &g, None, &cfg, &cell)
            .unwrap();
        assert_eq!(
            bits(&o.download_resident(&a).unwrap()),
            bits(&o.download_resident(&b).unwrap())
        );
        let (mut m1, mut m2) = (
            o.upload_resident(&t(n, 3)).unwrap(),
            o.upload_resident(&t(n, 3)).unwrap(),
        );
        let (mut pa, mut pb) = (
            o.upload_resident(&t(n, 1)).unwrap(),
            o.upload_resident(&t(n, 1)).unwrap(),
        );
        // v は非負。
        let pos = Tensor::new(data(n, 3).iter().map(|x| x.abs()).collect(), &[n]).unwrap();
        let mut v1 = o.upload_resident(&pos).unwrap();
        let mut v2 = o.upload_resident(&pos).unwrap();
        let acfg = adam_cfg(AdamStepKind::Coupled, 1e-2, 1e-3, 1);
        o.adam_step_device(&mut pa, &g, &mut m1, &mut v1, &acfg)
            .unwrap();
        o.adam_step_device_tracked(&mut pb, &g, &mut m2, &mut v2, &acfg, &cell)
            .unwrap();
        assert_eq!(
            bits(&o.download_resident(&pa).unwrap()),
            bits(&o.download_resident(&pb).unwrap())
        );
    }
}
