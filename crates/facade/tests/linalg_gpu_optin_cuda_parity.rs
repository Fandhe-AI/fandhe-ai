//! イシュー #3138（Stage C）: facade の公開 `Var` から CUDA GPU 線形代数カーネル
//! （`inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`）へ届くことの実機 parity テスト。
//!
//! 全テストが `#[ignore]`（DGX Spark GB10 実機用。CI では実行しない）。判定は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）を厳密ゼロ fail で適用する
//! （`docs/backend-gpu-linalg-decision.md` §5。tolerance・入力集合は変えない）。
//! 事前登録規則・測定コマンドは `docs/perf/logs/cuda-linalg-parity-3138/`。
//!
//! ゲート（`fandhe_ai_backend_cuda::linalg::set_gpu_linalg_enabled`）は facade 非公開
//! （同 §7。`api_surface.rs::facade_does_not_expose_gpu_linalg_optin` は `crates/facade/src`
//! のみ走査し `tests/` は対象外）のため、本ファイルが backend-cuda を直接参照して操作する。
//! ゲートはプロセス全体のグローバル状態なので、ゲート OFF のフォールバック経路を確かめ続ける
//! 既存の `linalg_ops_backend_parity.rs` とはファイルを分け、本ファイル内では `GATE_LOCK` で直列化し
//! drop で OFF へ戻す。
//!
//! ゲート ON でも `Unsupported` なら `Var` は静かにホスト f64 へフォールバックして pass してしまうため、
//! 各テストは `CudaBackendOps` を直接呼び `Unsupported` でないこと（デバイス経路に入ること）を先に確認する。

use std::sync::Mutex;

use fandhe_ai::Device;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_backend_cuda::linalg::{gpu_linalg_enabled, set_gpu_linalg_enabled};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

static GATE_LOCK: Mutex<()> = Mutex::new(());

/// ゲートを ON にし、drop で OFF へ戻す RAII ガード。
struct GateOn {
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl GateOn {
    fn new() -> Self {
        let g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_gpu_linalg_enabled(true);
        assert!(gpu_linalg_enabled(), "ゲートが ON にならない");
        GateOn { _guard: g }
    }
}

impl Drop for GateOn {
    fn drop(&mut self) {
        set_gpu_linalg_enabled(false);
    }
}

/// デバイス経路プローブ: `Unsupported`（= 静かなホストフォールバック）でないことを確認する。
fn assert_device_path<T>(ctx: &str, r: Result<T, BackendError>) {
    assert!(
        !matches!(r, Err(BackendError::Unsupported(_))),
        "{ctx}: Unsupported（デバイス経路に入っていない）"
    );
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn v(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().expect("slice").to_vec()
}

/// 決定的 LCG（`[-1, 1)`）。backend テストと同型。
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// 対角優位の `n×n` 行列（κ ≤ 1e3 の範囲。入力集合は設計記録 §5 のとおり広げない）を `scale` 倍する。
fn dominant(n: usize, seed: u64, scale: f32) -> Tensor<f32> {
    let mut rng = Lcg(seed);
    let mut d = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let x = rng.next();
            d.push(if i == j {
                (n as f32 + 1.0 + x.abs()) * scale
            } else {
                x * scale
            });
        }
    }
    t(d, &[n, n])
}

/// 対角優位の対称正定値行列（κ が小さい。κ ≤ 1e3）を `scale` 倍する。
fn spd(n: usize, scale: f32) -> Tensor<f32> {
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = if i == j {
                (n as f32 + 1.0) * scale
            } else {
                scale / (1.0 + (i as f32 - j as f32).abs())
            };
        }
    }
    t(d, &[n, n])
}

const SIZES: [usize; 5] = [2, 3, 8, 17, 64];
const SCALES: [f32; 3] = [1.0, 1e20, 1e-20];

fn i2_1e20() -> Tensor<f32> {
    t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2])
}

fn tapes() -> (fandhe_ai::Tape, fandhe_ai::Tape) {
    let cuda = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA 実機が必要");
    (fandhe_ai::tape(), cuda)
}

fn assert_finite(ctx: &str, x: &[f32]) {
    assert!(
        x.iter().all(|e| e.is_finite()),
        "{ctx}: 非有限値が出力された"
    );
}

fn check(ctx: &str, cuda: &Tensor<f32>, cpu: &Tensor<f32>) {
    let (c, r) = (v(cuda), v(cpu));
    assert_finite(ctx, &c);
    fandhe_ai_backend_cpu::parity::assert_parity(ctx, &c, &r);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138（docs/perf/logs/cuda-linalg-parity-3138/）"]
fn facade_inv_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, cuda_t) = tapes();
    let ops = CudaBackendOps::new(0);
    let mut cases = vec![("inv 1e20*I2".to_string(), i2_1e20())];
    for &n in &SIZES {
        for &s in &SCALES {
            cases.push((
                format!("inv n={n} scale={s:e}"),
                dominant(n, 0x5eed + n as u64, s),
            ));
        }
    }
    for (ctx, a) in &cases {
        assert_device_path(ctx, ops.linalg_inv(a));
        let c = cpu_t.var(a).inv().expect("cpu inv").to_tensor();
        let g = cuda_t.var(a).inv().expect("cuda inv").to_tensor();
        check(ctx, &g, &c);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138（docs/perf/logs/cuda-linalg-parity-3138/）"]
fn facade_solve_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, cuda_t) = tapes();
    let ops = CudaBackendOps::new(0);
    for &n in &SIZES {
        for &s in &SCALES {
            let a = dominant(n, 0x5eed + n as u64, s);
            let mut rng = Lcg(0xb + n as u64);
            let b = t((0..n * 3).map(|_| rng.next()).collect(), &[n, 3]);
            let ctx = format!("solve n={n} scale={s:e}");
            assert_device_path(&ctx, ops.linalg_solve(&a, &b));
            let c = cpu_t
                .var(&a)
                .solve(&cpu_t.var(&b))
                .expect("cpu solve")
                .to_tensor();
            let g = cuda_t
                .var(&a)
                .solve(&cuda_t.var(&b))
                .expect("cuda solve")
                .to_tensor();
            check(&ctx, &g, &c);
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138（docs/perf/logs/cuda-linalg-parity-3138/）"]
fn facade_det_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, cuda_t) = tapes();
    let ops = CudaBackendOps::new(0);
    let mut cases = vec![("det 1e20*I2".to_string(), i2_1e20())];
    for &n in &SIZES {
        for &s in &SCALES {
            cases.push((
                format!("det n={n} scale={s:e}"),
                dominant(n, 0x5eed + n as u64, s),
            ));
        }
    }
    for (ctx, a) in &cases {
        assert_device_path(ctx, ops.linalg_det(a));
        let c = v(&cpu_t.var(a).det().expect("cpu det").to_tensor());
        let g = v(&cuda_t.var(a).det().expect("cuda det").to_tensor());
        // det は f32 範囲外になりうる（例: 1e20*I2 → 1e40）。非有限の期待値は bit 一致で確認する
        // （判定式は変えない。backend の `assert_det_parity` と同方針）。
        if c.iter().all(|e| e.is_finite()) {
            assert_finite(ctx, &g);
            fandhe_ai_backend_cpu::parity::assert_parity(ctx, &g, &c);
        } else {
            let gb: Vec<u32> = g.iter().map(|e| e.to_bits()).collect();
            let cb: Vec<u32> = c.iter().map(|e| e.to_bits()).collect();
            assert_eq!(gb, cb, "{ctx}: 非有限の期待値と一致しない");
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138（docs/perf/logs/cuda-linalg-parity-3138/）"]
fn facade_cholesky_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, cuda_t) = tapes();
    let ops = CudaBackendOps::new(0);
    let mut cases = vec![("cholesky 1e20*I2".to_string(), i2_1e20())];
    for &n in &SIZES {
        for &s in &SCALES {
            cases.push((format!("cholesky n={n} scale={s:e}"), spd(n, s)));
        }
    }
    for (ctx, a) in &cases {
        assert_device_path(ctx, ops.linalg_cholesky(a));
        let c = cpu_t.var(a).cholesky().expect("cpu cholesky").to_tensor();
        let g = cuda_t.var(a).cholesky().expect("cuda cholesky").to_tensor();
        check(ctx, &g, &c);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138（docs/perf/logs/cuda-linalg-parity-3138/）"]
fn facade_qr_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, cuda_t) = tapes();
    let ops = CudaBackendOps::new(0);
    let mut cases = vec![("qr 1e20*I2".to_string(), i2_1e20())];
    for &n in &SIZES {
        for &s in &SCALES {
            cases.push((
                format!("qr n={n} scale={s:e}"),
                dominant(n, 0x5eed + n as u64, s),
            ));
        }
    }
    for (ctx, a) in &cases {
        assert_device_path(ctx, ops.linalg_qr(a));
        let c = cpu_t.var(a).qr().expect("cpu qr");
        let g = cuda_t.var(a).qr().expect("cuda qr");
        check(&format!("{ctx} Q"), &g.q.to_tensor(), &c.q.to_tensor());
        check(&format!("{ctx} R"), &g.r.to_tensor(), &c.r.to_tensor());
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138（docs/perf/logs/cuda-linalg-parity-3138/）"]
fn facade_eigh_gate_on_matches_cpu() {
    let _g = GateOn::new();
    let (cpu_t, cuda_t) = tapes();
    let ops = CudaBackendOps::new(0);
    let mut cases = vec![("eigh 1e20*I2".to_string(), i2_1e20())];
    for &n in &SIZES {
        for &s in &SCALES {
            cases.push((format!("eigh n={n} scale={s:e}"), spd(n, s)));
        }
    }
    for (ctx, a) in &cases {
        assert_device_path(ctx, ops.linalg_eigh(a));
        let c = cpu_t.var(a).eigh().expect("cpu eigh");
        let g = cuda_t.var(a).eigh().expect("cuda eigh");
        check(
            &format!("{ctx} eigenvalues"),
            &g.eigenvalues.to_tensor(),
            &c.eigenvalues.to_tensor(),
        );
        check(
            &format!("{ctx} eigenvectors"),
            &g.eigenvectors.to_tensor(),
            &c.eigenvectors.to_tensor(),
        );
    }
}
