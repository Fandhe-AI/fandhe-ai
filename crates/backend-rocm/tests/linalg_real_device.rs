//! ROCm 実機（MI300X・gfx942）での線形代数 6 演算を CPU 参照と突き合わせる実機 parity テスト
//! （イシュー #3444。実測は #3159 段 B）。
//!
//! - 役割: `RocmLinalgLu`・`RocmLinalgFactor` の出力を `CpuBackendOps::linalg_*` と
//!   `fandhe_ai_backend_cpu::compare`（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5
//!   未満。定数は変更しない）で判定する。内部は全経路 `f64` で、CPU 逐次実装との bit 一致は
//!   観測値（`bit_mismatch`）として出力するだけで assert しない。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。`run_rocm_parity.sh` はクレートの `--ignored`
//!   テストを全件実行するため本ファイルも自動で含まれ、**本ファイルの失敗は run の終了コードに影響する**。
//!   記録は `docs/perf/logs/rocm-parity-3159/README.md` の「linalg（#3444）」節。
//! - ログ行は GEMM の `ROCM_PARITY` 行と同じ形（`ROCM_PARITY_LINALG` で始まる）。

use fandhe_ai_backend_cpu::{CpuBackendOps, compare};
use fandhe_ai_backend_rocm::{
    GfxTarget, LuSolveOutcome, RocmError, RocmLinalgFactor, RocmLinalgLu,
};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn diag_dominant(n: usize, seed: u64) -> Vec<f32> {
    let mut a = data(n * n, seed);
    for i in 0..n {
        a[i * n + i] += n as f32 + 1.0;
    }
    a
}

/// 対称正定値（`M Mᵀ/n + I`）。
fn spd(n: usize, seed: u64) -> Vec<f32> {
    let m = data(n * n, seed);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let acc: f64 = (0..n)
                .map(|t| f64::from(m[i * n + t]) * f64::from(m[j * n + t]))
                .sum();
            let v = (acc / n as f64 + if i == j { 1.0 } else { 0.0 }) as f32;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

fn sym(n: usize, seed: u64) -> Vec<f32> {
    let mut a = data(n * n, seed);
    for i in 0..n {
        for j in 0..i {
            a[j * n + i] = a[i * n + j];
        }
    }
    a
}

fn t(d: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(d.to_vec(), shape).unwrap()
}

fn v(x: &Tensor<f32>) -> Vec<f32> {
    x.as_slice().unwrap().to_vec()
}

/// 1 件ぶんの記録行を出力し、複合判定の fail 数を返す。
fn record(name: &str, got: &[f32], want: &[f32]) -> usize {
    let r = compare(got, want).unwrap_or_else(|e| panic!("compare {name}: {e:?}"));
    let bits = got
        .iter()
        .zip(want)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    eprintln!(
        "ROCM_PARITY_LINALG\t{name}\ttotal={}\tfail={}\tmax_abs={:e}\tmax_rel={:e}\tmean_abs={:e}\tp99_abs={:e}\tbit_mismatch={bits}",
        r.total, r.fail_count, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.p99_abs_diff
    );
    r.fail_count
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn lu_family_matches_cpu_reference() {
    let lu = RocmLinalgLu::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let cpu = CpuBackendOps::new();
    let mut fails = 0;
    for &n in &[1usize, 2, 3, 17, 64] {
        for scale in [1.0f32, 1e20, 1e-20] {
            let a: Vec<f32> = diag_dominant(n, 10 + n as u64)
                .iter()
                .map(|x| x * scale)
                .collect();
            let b = data(n * 2, 20 + n as u64);
            let name = format!("solve n={n} scale={scale:e}");
            match lu.run_lu_solve_f32(&a, &b, n, 2).unwrap() {
                LuSolveOutcome::Solved(x) => {
                    let want = cpu.linalg_solve(&t(&a, &[n, n]), &t(&b, &[n, 2])).unwrap();
                    fails += record(&name, &x, &v(&want));
                }
                LuSolveOutcome::Singular => panic!("{name}: 特異と判定された"),
            }
            let name = format!("inv n={n} scale={scale:e}");
            match lu.run_lu_inv_f32(&a, n).unwrap() {
                LuSolveOutcome::Solved(x) => {
                    let want = cpu.linalg_inv(&t(&a, &[n, n])).unwrap();
                    fails += record(&name, &x, &v(&want));
                }
                LuSolveOutcome::Singular => panic!("{name}: 特異と判定された"),
            }
        }
        let mut a: Vec<f32> = data(n * n, 50 + n as u64)
            .iter()
            .map(|x| x * 0.01)
            .collect();
        for i in 0..n {
            a[i * n + i] += 1.0;
        }
        let got = lu.run_lu_det_f32(&a, n).unwrap();
        let want = cpu.linalg_det(&t(&a, &[n, n])).unwrap();
        fails += record(&format!("det n={n}"), &[got], &v(&want));
    }
    // 特異: solve・inv は Singular、det は 0.0。
    let sing = [1.0f32, 2.0, 2.0, 4.0];
    assert_eq!(
        lu.run_lu_solve_f32(&sing, &[1.0, 1.0], 2, 1).unwrap(),
        LuSolveOutcome::Singular
    );
    assert_eq!(
        lu.run_lu_inv_f32(&sing, 2).unwrap(),
        LuSolveOutcome::Singular
    );
    assert_eq!(lu.run_lu_det_f32(&sing, 2).unwrap(), 0.0);
    assert_eq!(
        fails, 0,
        "複合判定 fail の要素がある（上の ROCM_PARITY_LINALG 行を参照）"
    );
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn factor_family_matches_cpu_reference() {
    let f =
        RocmLinalgFactor::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let cpu = CpuBackendOps::new();
    let mut fails = 0;
    for &n in &[1usize, 2, 3, 17, 64] {
        for scale in [1.0f32, 1e20, 1e-20] {
            let a: Vec<f32> = spd(n, 60 + n as u64).iter().map(|x| x * scale).collect();
            let got = f.run_cholesky_f32(&a, n).unwrap();
            let want = cpu.linalg_cholesky(&t(&a, &[n, n])).unwrap();
            fails += record(&format!("cholesky n={n} scale={scale:e}"), &got, &v(&want));
        }
        let a = sym(n, 80 + n as u64);
        let (ev, vv) = f.run_eigh_f32(&a, n).unwrap();
        let want = cpu.linalg_eigh(&t(&a, &[n, n])).unwrap();
        fails += record(&format!("eigh evals n={n}"), &ev, &v(&want.eigenvalues));
        fails += record(&format!("eigh vecs n={n}"), &vv, &v(&want.eigenvectors));
    }
    for (m, n) in [
        (1usize, 1usize),
        (5, 3),
        (3, 5),
        (17, 17),
        (64, 64),
        (65, 17),
        (17, 65),
    ] {
        let a = data(m * n, 70 + (m * 31 + n) as u64);
        let (r, q) = f.run_qr_f32(&a, m, n).unwrap();
        let want = cpu.linalg_qr(&t(&a, &[m, n])).unwrap();
        fails += record(&format!("qr R {m}x{n}"), &r, &v(&want.r));
        fails += record(&format!("qr Q {m}x{n}"), &q, &v(&want.q));
    }
    // 非正定値・未収束の写像（CPU と同じ InvalidArgument 分類）。
    assert!(matches!(
        f.run_cholesky_f32(&[1.0, 2.0, 2.0, 1.0], 2),
        Err(RocmError::LinalgInvalidArgument { .. })
    ));
    assert!(matches!(
        f.run_eigh_f32(&[f32::NAN, 1.0, 1.0, 2.0], 2),
        Err(RocmError::LinalgInvalidArgument { .. })
    ));
    assert_eq!(
        fails, 0,
        "複合判定 fail の要素がある（上の ROCM_PARITY_LINALG 行を参照）"
    );
}
