//! `backend_cpu::softmax::run_softmax_f32` の受け入れ基準対応テスト
//! （イシュー #607）。
//!
//! 形状網羅（cols 端数〈NEON 4 要素幅の非倍数〉含む）・極値安定性・行和
//! 1・`run_fused`（canonical プラン。axis None／最終軸 rows>1）を検証
//! する。判定式・許容誤差は `backend_cpu::parity` を唯一の参照とし
//! 再定義しない（`.claude/rules/coding-rust.md`）。NEON/スカラー A/B
//! 同値テストは `backend_cpu::softmax` 側の `#[cfg(test)] mod tests`
//! （`pub(crate)` 関数への直接アクセスが必要なため）に置く。

use backend_cpu::CpuBackendOps;
use backend_cpu::parity::assert_parity;
use backend_cpu::softmax::run_softmax_f32;
use bench_harness::rng::Xorshift64Star;
use tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

/// テスト専用の素朴 CPU 参照実装（`run_softmax_f32` と数学的に同一だが、
/// 独立した実装で突き合わせる）。
fn naive_softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row {
            if v > max_v {
                max_v = v;
            }
        }
        let out_row = &mut out[r * cols..(r + 1) * cols];
        let mut sum = 0.0f32;
        for (o, &v) in out_row.iter_mut().zip(row.iter()) {
            let e = (v - max_v).exp();
            *o = e;
            sum += e;
        }
        for o in out_row.iter_mut() {
            *o /= sum;
        }
    }
    out
}

fn assert_softmax_matches_naive(seed: u64, rows: usize, cols: usize) {
    let x = Xorshift64Star::new(seed).fill_vec(rows * cols);
    let actual = run_softmax_f32(&x, rows, cols).unwrap();
    let expected = naive_softmax(&x, rows, cols);
    assert_parity(
        &format!("softmax rows={rows} cols={cols}"),
        &actual,
        &expected,
    );
}

/// 形状網羅: cols ∈ {0,1,4,8,1023,1024,4097} × rows ∈ {0,1,3,33}。
/// 4097 は NEON 端要素（`cols % 4 != 0`）検証。
#[test]
fn softmax_matches_naive_across_shapes() {
    let cols_cases: &[usize] = &[0, 1, 4, 8, 1023, 1024, 4097];
    let rows_cases: &[usize] = &[0, 1, 3, 33];
    let mut seed = 3000u64;
    for &cols in cols_cases {
        for &rows in rows_cases {
            seed += 1;
            assert_softmax_matches_naive(seed, rows, cols);
        }
    }
}

/// 行和が 1 になること（`cols > 0` の全ケース）。
#[test]
fn softmax_rows_sum_to_one() {
    let rows = 5usize;
    let cols = 37usize; // NEON 端要素を含む。
    let x = Xorshift64Star::new(555).fill_vec(rows * cols);
    let out = run_softmax_f32(&x, rows, cols).unwrap();
    for row in out.chunks(cols) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row sum={sum}");
    }
}

/// 数値安定性: 極値入力で NaN/inf を出さない。
#[test]
fn softmax_extreme_values_no_nan_inf() {
    let x = vec![1e30f32, -1e30, 1e30, -1e30, f32::MAX, -f32::MAX, 0.0, 0.0];
    let out = run_softmax_f32(&x, 2, 4).unwrap();
    for &v in &out {
        assert!(v.is_finite(), "expected finite softmax output, got {v}");
    }
}

fn canonical_softmax_ops(axis: Option<usize>) -> Vec<FusedOpKind> {
    vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Max { input: 0, axis },
        FusedOpKind::Broadcast { input: 1, axis },
        FusedOpKind::Sub { lhs: 0, rhs: 2 },
        FusedOpKind::Exp { input: 3 },
        FusedOpKind::Sum { input: 4, axis },
        FusedOpKind::Broadcast { input: 5, axis },
        FusedOpKind::Div { lhs: 4, rhs: 6 },
    ]
}

/// `run_fused`（canonical プラン。`axis: None`・rank-1）を per-op 合成と
/// 突き合わせる。
#[test]
fn softmax_run_fused_matches_per_op_composed_rank1() {
    let cols = 41usize; // NEON 4 要素幅の非倍数を含める。
    let x_data = Xorshift64Star::new(6161).fill_vec(cols);
    let x = Tensor::new(x_data.clone(), &[cols]).unwrap();

    let plan =
        FusionPlan::from_ops(canonical_softmax_ops(None), vec![cols], DType::F32, 1).unwrap();

    let cpu = CpuBackendOps::new();
    let fused_out = cpu.run_fused(&plan, &[&x]).unwrap();
    let expected = naive_softmax(&x_data, 1, cols);

    assert_eq!(fused_out.shape(), &[cols]);
    assert_parity(
        "softmax run_fused vs per-op composed (canonical plan, axis=None)",
        fused_out.as_slice().unwrap(),
        &expected,
    );
}

/// `run_fused`（canonical プラン。最終軸・rows > 1）を per-op 合成と
/// 突き合わせる。
#[test]
fn softmax_run_fused_matches_per_op_composed_last_axis() {
    let rows = 3usize;
    let cols = 17usize;
    let x_data = Xorshift64Star::new(7171).fill_vec(rows * cols);
    let x = Tensor::new(x_data.clone(), &[rows, cols]).unwrap();

    let plan = FusionPlan::from_ops(
        canonical_softmax_ops(Some(1)),
        vec![rows, cols],
        DType::F32,
        1,
    )
    .unwrap();

    let cpu = CpuBackendOps::new();
    let fused_out = cpu.run_fused(&plan, &[&x]).unwrap();
    let expected = naive_softmax(&x_data, rows, cols);

    assert_eq!(fused_out.shape(), &[rows, cols]);
    assert_parity(
        "softmax run_fused vs per-op composed (canonical plan, last axis)",
        fused_out.as_slice().unwrap(),
        &expected,
    );
}
