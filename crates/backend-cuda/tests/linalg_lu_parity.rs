//! イシュー #2674: CUDA LU 系線形代数（`linalg_inv`／`linalg_solve`／`linalg_det`）の
//! ゲート・検証・CPU 比較テスト。
//!
//! 属性なしテスト（通常 CI で実行。CUDA 非搭載でも panic しない）はゲート書き換えを
//! `static Mutex` で直列化し RAII で原状復帰する。`#[ignore]` テストは DGX Spark GB10 実機用:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test linalg_lu_parity -- --ignored --nocapture
//! ```
//!
//! Stage B（#3135）でカーネル起動を実装済み。ゲート ON・非空入力は `Unsupported` を返さない
//! （CI では `CudaUnavailable` 等の型付きエラー）。実機での一致（`assert_parity`＝厳密ゼロ
//! fail）と特異写像の実測は #3138 の担当で、本ファイルの `#[ignore]` テストがその入口になる。

use std::sync::Mutex;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::linalg::{CudaLinalg, set_gpu_linalg_enabled};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

static GATE_LOCK: Mutex<()> = Mutex::new(());

/// ゲートを ON にし、drop で OFF へ戻す。
struct GateOn<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
}

impl<'a> GateOn<'a> {
    fn new(on: bool) -> Self {
        let g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_gpu_linalg_enabled(on);
        Self { _guard: g }
    }
}

impl Drop for GateOn<'_> {
    fn drop(&mut self) {
        set_gpu_linalg_enabled(false);
    }
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("valid tensor")
}

#[test]
fn gate_off_returns_unsupported() {
    let _g = GateOn::new(false);
    let cuda = CudaBackendOps::new(0);
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let b = t(vec![1.0, 2.0], &[2, 1]);
    assert!(matches!(
        cuda.linalg_inv(&a),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.linalg_solve(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        cuda.linalg_det(&a),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn gate_on_validates_shapes_without_touching_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let rect = t(vec![0.0; 6], &[2, 3]);
    let rank1 = t(vec![0.0; 2], &[2]);
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let b_bad = t(vec![0.0; 3], &[3, 1]);
    let b_rank1 = t(vec![0.0; 2], &[2]);
    assert!(matches!(
        cuda.linalg_inv(&rect),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_det(&rect),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_inv(&rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        cuda.linalg_solve(&a, &b_bad),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        cuda.linalg_solve(&a, &b_rank1),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn gate_on_empty_matrix_matches_cpu() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = t(vec![], &[0, 0]);
    let b = t(vec![], &[0, 3]);
    let inv = cuda.linalg_inv(&a).expect("n=0 inv");
    assert_eq!(inv.shape(), cpu.linalg_inv(&a).expect("cpu").shape());
    let sol = cuda.linalg_solve(&a, &b).expect("n=0 solve");
    assert_eq!(sol.shape(), cpu.linalg_solve(&a, &b).expect("cpu").shape());
    let det = cuda.linalg_det(&a).expect("n=0 det");
    assert_eq!(det.shape(), cpu.linalg_det(&a).expect("cpu").shape());
    assert_eq!(det.contiguous().as_slice().expect("slice"), &[1.0f32]);
}

/// ゲート ON・非空入力は 3 演算とも `Unsupported` を返さない（`Unsupported` は
/// ゲート OFF のときだけ）。CUDA 非搭載の CI では `CudaUnavailable` 等の型付きエラー、
/// 実機では `Ok`。panic しない。
#[test]
fn gate_on_non_empty_does_not_panic() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let b = t(vec![1.0, 2.0], &[2, 1]);
    let b0 = t(vec![], &[2, 0]);
    let not_unsupported = |name: &str, r: Result<Tensor<f32>, BackendError>| {
        assert!(
            !matches!(r, Err(BackendError::Unsupported(_))),
            "{name}: ゲート ON の非空入力で Unsupported を返した"
        );
    };
    not_unsupported("inv", cuda.linalg_inv(&a));
    not_unsupported("solve", cuda.linalg_solve(&a, &b));
    not_unsupported("solve k=0", cuda.linalg_solve(&a, &b0));
    not_unsupported("det", cuda.linalg_det(&a));
}

/// 実機: 2 カーネルが GB10 で NVRTC コンパイル・ロードできる（Stage A で検証可能な項目）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn nvrtc_compiles_and_loads_lu_kernels_on_device() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let lin = CudaLinalg::new(&device).expect("NVRTC compile/load");
    assert_eq!(lin.ordinal(), 0);
}

// ---------------------------------------------------------------------------
// 実機 `#[ignore]` テスト（DGX Spark GB10）。実測の実施は #3138。
// 入力は設計記録 `docs/backend-gpu-linalg-decision.md` §5 の集合（決定的 LCG の
// 対角優位行列〈κ≤1e3 目安〉・全体 1e±20 倍・`1e20 × I₂`）。比較は
// `fandhe_ai_backend_cpu::assert_parity`（厳密ゼロ fail。tolerance は変えない）。
// ---------------------------------------------------------------------------

/// 決定的 LCG（シード固定。外部 RNG 依存なし）。`[-1, 1)` の `f32` を返す。
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

/// 対角優位の `n×n` 行列（対角 `n + 1 + |u|`・非対角 `[-1, 1)`）を `scale` 倍する。
fn dominant(n: usize, seed: u64, scale: f32) -> Tensor<f32> {
    let mut rng = Lcg(seed);
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let v = rng.next();
            data.push(if i == j {
                (n as f32 + 1.0 + v.abs()) * scale
            } else {
                v * scale
            });
        }
    }
    t(data, &[n, n])
}

fn rhs(n: usize, k: usize, seed: u64) -> Tensor<f32> {
    let mut rng = Lcg(seed);
    t((0..n * k).map(|_| rng.next()).collect(), &[n, k])
}

fn slice(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().expect("slice").to_vec()
}

/// 期待値が非有限（f32 範囲外）の det は `assert_parity` が Inf 同士を fail にするため、
/// 非有限のときは「GPU も同じ非有限値（`to_bits` 一致）」を別途確認する。判定式は不変。
fn assert_det_parity(ctx: &str, got: &[f32], want: &[f32]) {
    if want.iter().all(|v| v.is_finite()) {
        fandhe_ai_backend_cpu::assert_parity(ctx, got, want);
    } else {
        let g: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let w: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(g, w, "{ctx}: 非有限の期待値と一致しない");
    }
}

fn scaled(a: &Tensor<f32>, s: f32) -> Tensor<f32> {
    let n = a.shape()[0];
    t(slice(a).iter().map(|v| v * s).collect(), &[n, n])
}

/// 実機: ゲート ON の det を CPU と比較する（2×2 の基本ケース）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138"]
fn lu_det_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = t(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
    let got = cuda.linalg_det(&a).expect("linalg_det");
    let want = cpu.linalg_det(&a).expect("cpu det");
    fandhe_ai_backend_cpu::assert_parity("linalg_det 2x2", &slice(&got), &slice(&want));
}

/// 実機: det の入力集合（n 掃引・スケール・`1e20 × I₂`・置換が奇数回になる行列）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138"]
fn lu_det_matches_cpu_on_device_set() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let mut cases: Vec<(String, Tensor<f32>)> = Vec::new();
    for &n in &[1usize, 2, 3, 8, 17, 64] {
        let a = dominant(n, 0x5eed + n as u64, 1.0);
        for &s in &[1.0f32, 1e20, 1e-20] {
            cases.push((format!("det n={n} scale={s:e}"), scaled(&a, s)));
        }
    }
    cases.push(("det 1e20*I2".into(), t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2])));
    // 行入替えが必ず起きる（先頭ピボットが 0）ケースで符号を確認する。
    cases.push(("det swap".into(), t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2])));
    for (ctx, a) in &cases {
        let got = cuda
            .linalg_det(a)
            .unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
        let want = cpu.linalg_det(a).expect("cpu det");
        assert_det_parity(ctx, &slice(&got), &slice(&want));
    }
}

/// 実機: solve を CPU と比較する（n・k〈block 境界 128/129 をまたぐ〉・スケール）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138"]
fn lu_solve_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    for &n in &[1usize, 2, 3, 8, 17, 64] {
        for &k in &[0usize, 1, 5, 128, 129] {
            for &s in &[1.0f32, 1e20, 1e-20] {
                let a = scaled(&dominant(n, 0xa11ce + n as u64, 1.0), s);
                let b = rhs(n, k, 0xb0b + (n * 1000 + k) as u64);
                let ctx = format!("solve n={n} k={k} scale={s:e}");
                let got = cuda
                    .linalg_solve(&a, &b)
                    .unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
                let want = cpu.linalg_solve(&a, &b).expect("cpu solve");
                assert_eq!(got.shape(), want.shape(), "{ctx}: shape");
                fandhe_ai_backend_cpu::assert_parity(&ctx, &slice(&got), &slice(&want));
            }
        }
    }
}

/// 実機: inv を CPU と比較する（`1e20 × I₂` を含む）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138"]
fn lu_inv_matches_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let mut cases: Vec<(String, Tensor<f32>)> = Vec::new();
    for &n in &[1usize, 2, 3, 8, 17, 64] {
        for &s in &[1.0f32, 1e20, 1e-20] {
            cases.push((
                format!("inv n={n} scale={s:e}"),
                scaled(&dominant(n, 0x1eaf + n as u64, 1.0), s),
            ));
        }
    }
    cases.push(("inv 1e20*I2".into(), t(vec![1e20, 0.0, 0.0, 1e20], &[2, 2])));
    for (ctx, a) in &cases {
        let got = cuda
            .linalg_inv(a)
            .unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
        let want = cpu.linalg_inv(a).expect("cpu inv");
        assert_eq!(got.shape(), want.shape(), "{ctx}: shape");
        fandhe_ai_backend_cpu::assert_parity(ctx, &slice(&got), &slice(&want));
    }
}

/// 実機: 特異行列の写像が CPU と一致する（solve／inv は `InvalidArgument`、det は `0.0`）。
/// `n > 0` かつ `k == 0` の solve も特異なら `InvalidArgument`（CPU と同じ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。実測は #3138"]
fn lu_singular_maps_like_cpu_on_device() {
    let _g = GateOn::new(true);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let singulars = [
        ("[[1,2],[2,4]]", t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2])),
        ("zero 2x2", t(vec![0.0; 4], &[2, 2])),
        (
            "same rows 3x3",
            t(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 3]),
        ),
    ];
    for (ctx, a) in &singulars {
        let n = a.shape()[0];
        let b = rhs(n, 2, 7);
        let b0 = t(vec![], &[n, 0]);
        assert!(
            matches!(cuda.linalg_inv(a), Err(BackendError::InvalidArgument(_))),
            "{ctx}: inv"
        );
        assert!(
            matches!(cpu.linalg_inv(a), Err(BackendError::InvalidArgument(_))),
            "{ctx}: cpu inv"
        );
        assert!(
            matches!(
                cuda.linalg_solve(a, &b),
                Err(BackendError::InvalidArgument(_))
            ),
            "{ctx}: solve"
        );
        assert!(
            matches!(
                cuda.linalg_solve(a, &b0),
                Err(BackendError::InvalidArgument(_))
            ),
            "{ctx}: solve k=0"
        );
        let got = cuda.linalg_det(a).expect("det");
        let want = cpu.linalg_det(a).expect("cpu det");
        assert_eq!(slice(&got), vec![0.0f32], "{ctx}: det");
        assert_eq!(slice(&want), vec![0.0f32], "{ctx}: cpu det");
    }
}
