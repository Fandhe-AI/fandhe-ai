//! イシュー #3405（Stage D 第 2 段）: facade の公開 `Var` から Metal GPU 線形代数カーネル
//! （`inv`・`solve`・`det`・`cholesky`・`qr`）へ届くことの実機 parity テスト
//! （CUDA 版 `linalg_gpu_optin_cuda_parity.rs` の対）。
//!
//! 全テストが `#[ignore]`（Apple Silicon 実機用。CI では実行しない）。macOS 限定。ゲートは
//! facade の `fandhe_ai::set_metal_gpu_linalg_enabled`／`metal_gpu_linalg_enabled` 経由で操作する。
//! 判定は REQ-2 統一複合判定（`assert_parity`）を厳密ゼロ fail で適用する
//! （`docs/backend-gpu-linalg-decision.md` §5。tolerance は変えない）。入力集合は
//! backend-metal の正本 `linalg_parity_inputs.rs` を `#[path]` で共有し、新しい入力は作らない
//! （`RULE.txt` とログ README も触らない）。facade 経由の実機実測は #3866 の範囲。
//!
//! ゲート ON でも `Unsupported` なら `Var` は静かにホスト f64 へフォールバックして pass して
//! しまうため、各ケースは先に `MetalBackendOps` を直接呼び `Unsupported` でないことを確認する。

#![cfg(target_os = "macos")]

use std::sync::Mutex;

use fandhe_ai::Device;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

#[path = "../../backend-metal/tests/linalg_parity_inputs.rs"]
mod inputs;

use inputs::*;

static GATE_LOCK: Mutex<()> = Mutex::new(());

/// ゲートを ON にし、drop で OFF へ戻す RAII ガード。
struct GateOn {
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl GateOn {
    fn new() -> Self {
        let g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        fandhe_ai::set_metal_gpu_linalg_enabled(true);
        assert!(
            fandhe_ai::metal_gpu_linalg_enabled(),
            "ゲートが ON にならない"
        );
        GateOn { _guard: g }
    }
}

impl Drop for GateOn {
    fn drop(&mut self) {
        fandhe_ai::set_metal_gpu_linalg_enabled(false);
    }
}

fn assert_device_path<T>(ctx: &str, r: Result<T, BackendError>) {
    assert!(
        !matches!(r, Err(BackendError::Unsupported(_))),
        "{ctx}: Unsupported（デバイス経路に入っていない）"
    );
}

fn tapes() -> (fandhe_ai::Tape, fandhe_ai::Tape) {
    let metal = fandhe_ai::tape_for(Device::Metal).expect("Metal 実機が必要");
    (fandhe_ai::tape(), metal)
}

fn check(ctx: &str, gpu: &Tensor<f32>, cpu: &Tensor<f32>) {
    let (g, c) = (s(gpu), s(cpu));
    assert!(
        g.iter().all(|e| e.is_finite()),
        "{ctx}: 非有限値が出力された"
    );
    fandhe_ai_backend_cpu::parity::assert_parity(ctx, &g, &c);
}

/// LU 系（inv・solve・det）の入力: 対角優位を `SCALES` 倍したものと `1e20 × I₂`。
fn lu_cases() -> Vec<(String, Tensor<f32>)> {
    let mut cases = vec![("1e20*I2".to_string(), eye2_scaled(EYE2_SCALE))];
    for &n in &SIZES {
        for &f in &SCALES {
            cases.push((format!("n={n} scale={f:e}"), scaled(&diag_dominant(n), f)));
        }
    }
    cases
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）が必要。facade 経由の実測は #3866"]
fn facade_inv_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, gpu_t) = tapes();
    let ops = MetalBackendOps::new();
    for (tag, a) in &lu_cases() {
        let ctx = format!("inv {tag}");
        assert_device_path(&ctx, ops.linalg_inv(a));
        let c = cpu_t.var(a).inv().expect("cpu inv").to_tensor();
        let g = gpu_t.var(a).inv().expect("metal inv").to_tensor();
        check(&ctx, &g, &c);
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）が必要。facade 経由の実測は #3866"]
fn facade_solve_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, gpu_t) = tapes();
    let ops = MetalBackendOps::new();
    for (tag, a) in &lu_cases() {
        let n = a.shape()[0];
        let b = t(
            (0..n * 2).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect(),
            &[n, 2],
        );
        let ctx = format!("solve {tag}");
        assert_device_path(&ctx, ops.linalg_solve(a, &b));
        let c = cpu_t
            .var(a)
            .solve(&cpu_t.var(&b))
            .expect("cpu solve")
            .to_tensor();
        let g = gpu_t
            .var(a)
            .solve(&gpu_t.var(&b))
            .expect("metal solve")
            .to_tensor();
        check(&ctx, &g, &c);
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）が必要。facade 経由の実測は #3866"]
fn facade_det_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, gpu_t) = tapes();
    let ops = MetalBackendOps::new();
    // det は等倍かつ n <= DET_MAX_N のみ。スケール倍・1e20*I2 は行列式が f32 範囲を超えて inf となり
    // check() の有限値要件に反する（backend-metal/tests/linalg_parity.rs と同根拠）。
    for n in SIZES.into_iter().filter(|&n| n <= DET_MAX_N) {
        let tag = format!("n={n}");
        let a = &diag_dominant(n);
        let ctx = format!("det {tag}");
        assert_device_path(&ctx, ops.linalg_det(a));
        let c = cpu_t.var(a).det().expect("cpu det").to_tensor();
        let g = gpu_t.var(a).det().expect("metal det").to_tensor();
        check(&ctx, &g, &c);
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）が必要。facade 経由の実測は #3866"]
fn facade_cholesky_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, gpu_t) = tapes();
    let ops = MetalBackendOps::new();
    let mut cases = vec![("1e20*I2".to_string(), eye2_scaled(EYE2_SCALE))];
    for &n in &SIZES {
        for &f in &SCALES {
            cases.push((format!("n={n} scale={f:e}"), scaled(&spd(n), f)));
        }
    }
    for (tag, a) in &cases {
        let ctx = format!("cholesky {tag}");
        assert_device_path(&ctx, ops.linalg_cholesky(a));
        let c = cpu_t.var(a).cholesky().expect("cpu cholesky").to_tensor();
        let g = gpu_t.var(a).cholesky().expect("metal cholesky").to_tensor();
        check(&ctx, &g, &c);
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）が必要。facade 経由の実測は #3866"]
fn facade_qr_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, gpu_t) = tapes();
    let ops = MetalBackendOps::new();
    for &(m, n) in &QR_SHAPES {
        for &f in &SCALES {
            let a = scaled(&qr_input(m, n), f);
            let ctx = format!("qr {m}x{n} scale={f:e}");
            assert_device_path(&ctx, ops.linalg_qr(&a));
            let c = cpu_t.var(&a).qr().expect("cpu qr");
            let g = gpu_t.var(&a).qr().expect("metal qr");
            check(&format!("{ctx} Q"), &g.q.to_tensor(), &c.q.to_tensor());
            check(&format!("{ctx} R"), &g.r.to_tensor(), &c.r.to_tensor());
        }
    }
}
