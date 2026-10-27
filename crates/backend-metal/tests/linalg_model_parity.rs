//! イシュー #2676: `shaders/linalg.metal` の f32 ホスト逐語モデル
//! （`fandhe_ai_backend_metal::linalg_model`）と CPU 参照実装（f64）の突合
//! （設計記録 `docs/backend-gpu-linalg-decision.md` §5）。
//!
//! カーネル起動は新規 `unsafe` の承認待ちのため、「カーネルに写像した演算列
//! （k 昇順の逐次ループ・FMA 縮約なし）が統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満。`assert_parity`）を厳密ゼロ fail で満たす」ことを Linux でも
//! 検証する。tolerance は変更しない。入力は κ が小さい（κ ≈ 10 の対角優位／SPD）
//! 行列から始める（κ ≤ 1e3 は上限であって目標ではない。決定記録 §5）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
use fandhe_ai_backend_metal::linalg::{lu_diag_product, permutation_sign, qr_finalize};
use fandhe_ai_backend_metal::linalg_model::{cholesky_f32, lu_factor_f32, lu_solve_f32, qr_f32};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

// #3139 の記録専用診断が入力の正本（実機テストと同一）と判定出力ヘルパーを共有する。
#[path = "linalg_parity_inputs.rs"]
mod inputs;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("valid tensor")
}

fn s(x: &Tensor<f32>) -> Vec<f32> {
    x.contiguous().as_slice().expect("slice").to_vec()
}

/// 対角優位（κ ≈ 10 オーダー）の n×n 行列。
fn diag_dominant(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    let mut a = rng.fill_vec(n * n);
    for i in 0..n {
        a[i * n + i] += n as f32 * 0.8 + 2.0;
    }
    a.iter().map(|v| v * scale).collect()
}

/// 対称正定値（`B Bᵀ/n + I` ではなく対角優位対称化で κ を抑える）。
fn spd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let base = diag_dominant(n, seed, 1.0);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = 0.5 * (base[i * n + j] + base[j * n + i]) * scale;
        }
    }
    a
}

fn rhs(n: usize, k: usize, seed: u64) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(n * k)
}

fn model_solve(a: &[f32], b: &[f32], n: usize, k: usize) -> Vec<f32> {
    let lu = lu_factor_f32(a, n).expect("dims");
    lu_solve_f32(&lu, b, n, k).expect("dims").expect("非特異")
}

fn model_inv(a: &[f32], n: usize) -> Vec<f32> {
    let mut eye = vec![0.0f32; n * n];
    for i in 0..n {
        eye[i * n + i] = 1.0;
    }
    model_solve(a, &eye, n, n)
}

fn model_det(a: &[f32], n: usize) -> f32 {
    let lu = lu_factor_f32(a, n).expect("dims");
    if lu.singular {
        return 0.0;
    }
    let diag: Vec<f64> = (0..n).map(|i| f64::from(lu.lu[i * n + i])).collect();
    lu_diag_product(&diag, permutation_sign(lu.swap_count)) as f32
}

fn check_lu_family(n: usize, seed: u64, scale: f32, with_det: bool) {
    let cpu = CpuBackendOps::new();
    let a = diag_dominant(n, seed, scale);
    let at = t(a.clone(), &[n, n]);
    let k = 3;
    let b = rhs(n, k, seed ^ 0x55);
    let bt = t(b.clone(), &[n, k]);
    assert_parity(
        &format!("solve n={n} scale={scale}"),
        &model_solve(&a, &b, n, k),
        &s(&cpu.linalg_solve(&at, &bt).expect("cpu solve")),
    );
    assert_parity(
        &format!("inv n={n} scale={scale}"),
        &model_inv(&a, n),
        &s(&cpu.linalg_inv(&at).expect("cpu inv")),
    );
    if with_det {
        assert_parity(
            &format!("det n={n}"),
            &[model_det(&a, n)],
            &s(&cpu.linalg_det(&at).expect("cpu det")),
        );
    }
}

#[test]
fn lu_family_matches_cpu_on_diagonally_dominant() {
    for (i, n) in [1usize, 2, 3, 8, 17, 64].into_iter().enumerate() {
        // det は n=64 で f32 範囲外（inf）になるため n ≤ 17 に限る。
        check_lu_family(n, 100 + i as u64, 1.0, n <= 17);
    }
}

#[test]
fn lu_family_matches_cpu_at_extreme_scales() {
    for scale in [1e20f32, 1e-20f32] {
        for n in [2usize, 8, 17] {
            check_lu_family(n, 7 + n as u64, scale, false);
        }
    }
    // 1e20 × I₂。
    let cpu = CpuBackendOps::new();
    let a = vec![1e20f32, 0.0, 0.0, 1e20];
    let at = t(a.clone(), &[2, 2]);
    assert_parity(
        "inv 1e20*I2",
        &model_inv(&a, 2),
        &s(&cpu.linalg_inv(&at).expect("cpu inv")),
    );
}

#[test]
fn pivoting_matches_cpu_when_row_swap_required() {
    // 先頭ピボットが 0 で行入替が必須（swap_count が奇数になる）。
    let cpu = CpuBackendOps::new();
    let a = vec![0.0f32, 2.0, 1.0, 3.0];
    let at = t(a.clone(), &[2, 2]);
    assert_parity(
        "det swap",
        &[model_det(&a, 2)],
        &s(&cpu.linalg_det(&at).expect("cpu det")),
    );
    assert_eq!(model_det(&a, 2), -2.0);
}

#[test]
fn singular_matrix_is_flagged_and_det_is_zero() {
    let cpu = CpuBackendOps::new();
    let a = vec![1.0f32, 2.0, 2.0, 4.0];
    let lu = lu_factor_f32(&a, 2).expect("dims");
    assert!(lu.singular);
    assert!(
        lu_solve_f32(&lu, &[1.0, 1.0], 2, 1)
            .expect("dims")
            .is_none()
    );
    assert_eq!(model_det(&a, 2), 0.0);
    let at = t(a, &[2, 2]);
    assert!(cpu.linalg_inv(&at).is_err());
    assert_eq!(s(&cpu.linalg_det(&at).expect("cpu det")), vec![0.0]);
}

#[test]
fn nan_input_passes_through_lu_but_cholesky_rejects() {
    let lu = lu_factor_f32(&[f32::NAN, 1.0, 1.0, 2.0], 2).expect("dims");
    assert!(!lu.singular);
    let cpu = CpuBackendOps::new();
    let a = vec![f32::NAN, 1.0, 1.0, 2.0];
    assert!(cholesky_f32(&a, 2).expect("dims").is_none());
    assert!(cpu.linalg_cholesky(&t(a, &[2, 2])).is_err());
}

#[test]
fn cholesky_matches_cpu_and_rejects_non_spd() {
    let cpu = CpuBackendOps::new();
    for (i, n) in [1usize, 2, 3, 8, 17, 64].into_iter().enumerate() {
        for scale in [1.0f32, 1e20, 1e-20] {
            let a = spd(n, 300 + i as u64, scale);
            let model = cholesky_f32(&a, n).expect("dims").expect("SPD");
            let reference = s(&cpu.linalg_cholesky(&t(a, &[n, n])).expect("cpu"));
            assert_parity(&format!("cholesky n={n} scale={scale}"), &model, &reference);
        }
    }
    let not_spd = vec![1.0f32, 2.0, 2.0, 1.0];
    assert!(cholesky_f32(&not_spd, 2).expect("dims").is_none());
    assert!(cpu.linalg_cholesky(&t(not_spd, &[2, 2])).is_err());
}

fn model_qr(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (r_work, q_work) = qr_f32(a, m, n).expect("dims");
    qr_finalize(&r_work, &q_work, m, n).expect("finalize")
}

#[test]
fn qr_matches_cpu_on_rectangular_and_scaled() {
    let cpu = CpuBackendOps::new();
    for (i, (m, n)) in [(8usize, 3usize), (3, 8), (64, 17), (1, 5), (5, 1), (6, 6)]
        .into_iter()
        .enumerate()
    {
        for scale in [1.0f32, 1e20, 1e-20] {
            let a: Vec<f32> = Xorshift64Star::new(500 + i as u64)
                .fill_vec(m * n)
                .into_iter()
                .map(|v| v * scale)
                .collect();
            let (r, q) = model_qr(&a, m, n);
            let reference = cpu.linalg_qr(&t(a, &[m, n])).expect("cpu qr");
            assert_parity(&format!("qr R {m}x{n} s={scale}"), &r, &s(&reference.r));
            assert_parity(&format!("qr Q {m}x{n} s={scale}"), &q, &s(&reference.q));
        }
    }
}

#[test]
fn qr_rank_deficient_zero_column_matches_cpu() {
    // 先頭列がゼロ（norm_x == 0 の分岐＝反射を適用しない）。
    let cpu = CpuBackendOps::new();
    let mut a = Xorshift64Star::new(900).fill_vec(5 * 3);
    for row in 0..5 {
        a[row * 3] = 0.0;
    }
    let (r, q) = model_qr(&a, 5, 3);
    let reference = cpu.linalg_qr(&t(a, &[5, 3])).expect("cpu qr");
    assert_parity("qr R rank-deficient", &r, &s(&reference.r));
    assert_parity("qr Q rank-deficient", &q, &s(&reference.q));
}

#[test]
fn model_rejects_bad_dims_without_panic() {
    // イシュー #2676 codex P1: 公開モデル関数は無検証の添字アクセスで panic しない。
    assert!(lu_factor_f32(&[], 1).is_err());
    assert!(cholesky_f32(&[], 1).is_err());
    assert!(qr_f32(&[1.0], 2, 2).is_err());
    assert!(lu_factor_f32(&[], 0).is_ok());
    assert!(lu_factor_f32(&[1.0], usize::MAX).is_err());
    let lu = lu_factor_f32(&[1.0, 0.0, 0.0, 1.0], 2).expect("dims");
    assert!(lu_solve_f32(&lu, &[1.0], 2, 1).is_err());
    let mut bad = lu.clone();
    bad.perm = vec![0, 7];
    assert!(lu_solve_f32(&bad, &[1.0, 1.0], 2, 1).is_err());
    let mut short = lu;
    short.lu.truncate(1);
    assert!(lu_solve_f32(&short, &[1.0, 1.0], 2, 1).is_err());
}

#[test]
fn qr_householder_does_not_overflow_near_f32_max() {
    // イシュー #2676 codex P2: A=[2e38] で vv[0]-=alpha が overflow し NaN になっていた。
    for m in [1usize, 3] {
        let a: Vec<f32> = (0..m).map(|i| if i == 0 { 2e38 } else { 1e38 }).collect();
        let (r, q) = model_qr(&a, m, 1);
        assert!(
            r.iter().chain(q.iter()).all(|v| v.is_finite()),
            "m={m}: {r:?} {q:?}"
        );
        assert!(r[0] > 0.0);
    }
}
/// 記録専用診断（#3139。assert しない・判定に含めない）: 実機テスト `qr_matches_cpu_on_device` と
/// 同じ入力（正本 `linalg_parity_inputs.rs`）で f32 ホストモデルを CPU 参照と突合し、判定を外れる要素を出力する。
/// モデルが実機と同形の不一致を出すなら精度クラス（f32 対 f64）による差で、カーネルのずれではないと判別できる。
#[test]
#[ignore = "記録専用診断（#3139）"]
fn qr_input_model_vs_cpu_diagnostic() {
    let cpu = CpuBackendOps::new();
    let mut cases: Vec<(String, _)> = Vec::new();
    for (m, n) in inputs::QR_SHAPES {
        for factor in inputs::SCALES {
            cases.push((
                format!("{m}x{n} scale={factor:e}"),
                inputs::scaled(&inputs::qr_input(m, n), factor),
            ));
        }
    }
    cases.push(("eye2 1e20".into(), inputs::eye2_scaled(inputs::EYE2_SCALE)));
    for (label, a) in cases {
        let sh = a.shape().to_vec();
        let (m, n) = (sh[0], sh[1]);
        let k = m.min(n);
        let (mr, mq) = model_qr(&s(&a), m, n);
        let reference = cpu.linalg_qr(&a).expect("cpu qr");
        inputs::diag_compare(
            &format!("R {label}"),
            "model",
            &mr,
            "cpu",
            &s(&reference.r),
            n,
        );
        inputs::diag_compare(
            &format!("Q {label}"),
            "model",
            &mq,
            "cpu",
            &s(&reference.q),
            k,
        );
    }
}
