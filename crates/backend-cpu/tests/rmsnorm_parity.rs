//! `fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32` の受け入れ基準対応テスト
//! （イシュー #607）。
//!
//! 形状網羅（cols 端数〈NEON 4 要素幅の非倍数〉含む）・weight/eps 有無・
//! `run_fused`（canonical プラン）vs per-op 合成・NEON/スカラー同値
//! （aarch64 限定）を検証する。判定式・許容誤差は `fandhe_ai_backend_cpu::parity`
//! を唯一の参照とし再定義しない（`.claude/rules/coding-rust.md`）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32;
use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

/// テスト専用の素朴 CPU 参照実装（`f32::mul_add` を使い FMA 契約を揃える。
/// `run_rmsnorm_f32` と数学的に同一だが、独立した実装で突き合わせることで
/// カーネル実装自体のバグを検出できるようにする）。
fn naive_rmsnorm(x: &[f32], w: Option<&[f32]>, eps: f32, rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    let inv_n = 1.0f32 / hidden as f32;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut acc = 0.0f32;
        for &v in row {
            acc = v.mul_add(v, acc);
        }
        let rstd = 1.0f32 / acc.mul_add(inv_n, eps).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut normed = row[i] * rstd;
            if let Some(w) = w {
                normed *= w[i];
            }
            out_row[i] = normed;
        }
    }
    out
}

fn assert_rmsnorm_matches_naive(
    seed: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    eps: f32,
) {
    let x = Xorshift64Star::new(seed).fill_vec(rows * hidden);
    let w = if with_weight {
        Some(Xorshift64Star::new(seed + 500).fill_vec(hidden))
    } else {
        None
    };
    let actual = run_rmsnorm_f32(&x, w.as_deref(), eps, rows, hidden).unwrap();
    let expected = naive_rmsnorm(&x, w.as_deref(), eps, rows, hidden);
    assert_parity(
        &format!("rmsnorm rows={rows} hidden={hidden} with_weight={with_weight} eps={eps}"),
        &actual,
        &expected,
    );
}

/// 形状網羅: cols（hidden）∈ {0,1,4,8,1023,1024,4097} × rows ∈ {0,1,3,33}。
/// 4097 は NEON 端要素（`hidden % 4 != 0`）検証。
#[test]
fn rmsnorm_matches_naive_across_shapes() {
    let hiddens: &[usize] = &[0, 1, 4, 8, 1023, 1024, 4097];
    let rows_cases: &[usize] = &[0, 1, 3, 33];
    let mut seed = 2000u64;
    for &hidden in hiddens {
        for &rows in rows_cases {
            for &with_weight in &[false, true] {
                seed += 1;
                assert_rmsnorm_matches_naive(seed, rows, hidden, with_weight, 1e-5);
            }
        }
    }
    // eps=0.0（`run_fused` 経由の canonical プランと同じ eps）。
    assert_rmsnorm_matches_naive(9001, 4, 256, false, 0.0);
}

/// 数値安定性: 極値入力で NaN/inf を出さない。
#[test]
fn rmsnorm_extreme_values_no_nan_inf() {
    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let out = run_rmsnorm_f32(&x, None, 1e-5, 2, 4).unwrap();
    for &v in &out {
        assert!(v.is_finite(), "expected finite rmsnorm output, got {v}");
    }
}

/// `run_fused`（canonical プラン: `x * rsqrt(sum(x^2))`。mean 化・eps・
/// weight なし）を per-op 合成と突き合わせる（`backend-cuda::
/// rmsnorm_run_fused_matches_cpu_composed_env_adaptive` と同型）。
#[test]
fn rmsnorm_run_fused_matches_per_op_composed() {
    let hidden = 37usize; // NEON 4 要素幅の非倍数を含める。
    let x_data = Xorshift64Star::new(4242).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[hidden]).unwrap();

    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Mul { lhs: 0, rhs: 0 },
        FusedOpKind::Sum {
            input: 1,
            axis: None,
        },
        FusedOpKind::Rsqrt { input: 2 },
        FusedOpKind::Broadcast {
            input: 3,
            axis: None,
        },
        FusedOpKind::Mul { lhs: 4, rhs: 0 },
    ];
    let plan = FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1).unwrap();

    let cpu = CpuBackendOps::new();
    let fused_out = cpu.run_fused(&plan, &[&x]).unwrap();

    let sq: Vec<f32> = x_data.iter().map(|v| v * v).collect();
    let sum: f32 = sq.iter().sum();
    let rstd = 1.0f32 / sum.sqrt();
    let composed: Vec<f32> = x_data.iter().map(|v| v * rstd).collect();

    assert_eq!(fused_out.shape(), &[hidden]);
    assert_parity(
        "rmsnorm run_fused vs per-op composed (canonical plan)",
        fused_out.as_slice().unwrap(),
        &composed,
    );
}

// NEON/スカラー A/B 同値テストは `fandhe_ai_backend_cpu::rmsnorm` 側の
// `#[cfg(test)] mod tests`（`pub(crate)` 関数への直接アクセスが必要な
// ため、統合テストクレートからは呼べない）に置く
// （`crates/backend-cpu/src/rmsnorm.rs::tests::neon_matches_scalar_various_hidden`）。
