//! `TypedOps<half::f16>` の受け入れ基準対応テスト（イシュー #1698・親 #1649）。
//!
//! 設計方針（`crates/backend-cpu/src/typed_f16.rs` モジュール doc）は
//! 「f16 をソフトウェア変換で f32 へ昇格 → 既存 f32 `BackendOps` カーネル
//! （`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`）へ委譲 →
//! 出力を f16 へ 1 回丸め」。本ファイルはクレート境界外（統合テスト）
//! から次を検証する:
//!
//! 1. `&dyn BackendOps` 経由で `typed_ops_f16()` が `Some` を返すこと
//!    （`typed_ops_f64`／`typed_ops_bf16` の状態は assert しない）
//! 2. `gemm` が `matmul_reference_fma`（p 昇順 `mul_add`）を f32 昇格→f16
//!    丸めした値と bit 完全一致すること（複数形状）
//! 3. elementwise（`add`／`mul`／`relu`／`exp`／`tanh`）がスカラー参照
//!    実装（f32 昇格→f16 丸め）と bit 完全一致すること
//! 4. reduction（`sum`／`max`）が f64 逐次和／`f32::max` 参照と
//!    複合判定（`assert_parity`）で一致すること
//! 5. 非 contiguous view（`transpose_2d`）が contiguous 実体化後と bit 一致
//!    すること
//! 6. エラー経路（shape 不整合・空縮約）が型付きエラーで返ること
//!    （panic しない）
//!
//! 実機依存なし（CPU のみ）のため `#[ignore]` 分離は不要。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};
use half::f16;

fn to_f16_vec(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn to_f32_vec(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// f16 精度で正確に表現できる範囲（丸め誤差を持ち込まないための
/// スケール）に収めた乱数入力を生成する。
fn random_f16_matrix(seed: u64, len: usize, scale: f32) -> Vec<f16> {
    let f32_vals: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(len)
        .iter()
        .map(|&v| v * scale)
        .collect();
    to_f16_vec(&f32_vals)
}

fn gemm_reference_f16(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let a32 = to_f32_vec(a);
    let b32 = to_f32_vec(b);
    let mut c = vec![0.0f32; m * n];
    fandhe_ai_backend_cpu::parity::matmul_reference_fma(&a32, &b32, &mut c, m, n, k).unwrap();
    // 参照値自体も出力 dtype（f16）へ 1 回丸めてから f32 へ戻し、実装
    // 出力（f16 → f32）と bit 完全一致する形に揃える。
    c.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

#[test]
fn typed_ops_f16_accessor_is_some() {
    let ops = CpuBackendOps::new();
    let dyn_ops: &dyn BackendOps = &ops;
    assert!(dyn_ops.typed_ops_f16().is_some());
}

// --- gemm: bit 完全一致 ---

#[test]
fn gemm_bit_exact_against_reference_various_shapes() {
    let ops = CpuBackendOps::new();
    for &(m, n, k) in &[
        (1, 1, 1),
        (3, 5, 7),
        (64, 64, 64),
        (129, 65, 33),
        (2, 2, 256),
    ] {
        let a = random_f16_matrix(m as u64 * 31 + 1, m * k, 0.05);
        let b = random_f16_matrix(k as u64 * 17 + 2, k * n, 0.05);
        let a_t = Tensor::new(a.clone(), &[m, k]).unwrap();
        let b_t = Tensor::new(b.clone(), &[k, n]).unwrap();

        let out = TypedOps::<f16>::gemm(&ops, &a_t, &b_t).unwrap();
        let actual = to_f32_vec(&out.host_slice());
        let expected = gemm_reference_f16(&a, &b, m, n, k);
        assert_eq!(actual, expected, "gemm mismatch for m={m} n={n} k={k}");
    }
}

// --- elementwise: bit 完全一致 ---

#[test]
fn add_bit_exact_same_shape_and_broadcast() {
    let ops = CpuBackendOps::new();

    // 同形状。
    let a = to_f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = to_f16_vec(&[0.5, -1.0, 2.0, 0.0, 3.5, -2.5]);
    let a_t = Tensor::new(a.clone(), &[2, 3]).unwrap();
    let b_t = Tensor::new(b.clone(), &[2, 3]).unwrap();
    let out = TypedOps::<f16>::add(&ops, &a_t, &b_t).unwrap();
    let expected: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f16::from_f32(x.to_f32() + y.to_f32()).to_f32())
        .collect();
    assert_eq!(to_f32_vec(&out.host_slice()), expected);

    // ブロードキャスト（[2,3] + [3]）。
    let c = to_f16_vec(&[10.0, 20.0, 30.0]);
    let c_t = Tensor::new(c.clone(), &[3]).unwrap();
    let out2 = TypedOps::<f16>::add(&ops, &a_t, &c_t).unwrap();
    let expected2: Vec<f32> = (0..2)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j): (usize, usize)| f16::from_f32(a[i * 3 + j].to_f32() + c[j].to_f32()).to_f32())
        .collect();
    assert_eq!(to_f32_vec(&out2.host_slice()), expected2);
}

#[test]
fn mul_bit_exact_broadcast() {
    let ops = CpuBackendOps::new();
    let a = to_f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = to_f16_vec(&[2.0, 3.0, 4.0]);
    let a_t = Tensor::new(a.clone(), &[2, 3]).unwrap();
    let b_t = Tensor::new(b.clone(), &[3]).unwrap();
    let out = TypedOps::<f16>::mul(&ops, &a_t, &b_t).unwrap();
    let expected: Vec<f32> = (0..2)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j): (usize, usize)| f16::from_f32(a[i * 3 + j].to_f32() * b[j].to_f32()).to_f32())
        .collect();
    assert_eq!(to_f32_vec(&out.host_slice()), expected);
}

#[test]
fn relu_exp_tanh_bit_exact_large_and_small() {
    let ops = CpuBackendOps::new();
    for &n in &[8usize, (1 << 15) + 3] {
        let raw = random_f16_matrix(n as u64 + 7, n, 0.02);
        let a_t = Tensor::new(raw.clone(), &[n]).unwrap();

        let relu_out = TypedOps::<f16>::relu(&ops, &a_t).unwrap();
        let relu_expected: Vec<f32> = raw
            .iter()
            .map(|&x| f16::from_f32(x.to_f32().max(0.0)).to_f32())
            .collect();
        assert_eq!(to_f32_vec(&relu_out.host_slice()), relu_expected);

        let exp_out = TypedOps::<f16>::exp(&ops, &a_t).unwrap();
        let exp_expected: Vec<f32> = raw
            .iter()
            .map(|&x| f16::from_f32(x.to_f32().exp()).to_f32())
            .collect();
        assert_eq!(to_f32_vec(&exp_out.host_slice()), exp_expected);

        let tanh_out = TypedOps::<f16>::tanh(&ops, &a_t).unwrap();
        let tanh_expected: Vec<f32> = raw
            .iter()
            .map(|&x| f16::from_f32(x.to_f32().tanh()).to_f32())
            .collect();
        assert_eq!(to_f32_vec(&tanh_out.host_slice()), tanh_expected);
    }
}

// --- reduction: 複合判定（f64 逐次和・CHUNK 境界を跨ぐサイズ） ---

#[test]
fn sum_matches_f64_sequential_reference_across_chunk_boundary() {
    let ops = CpuBackendOps::new();
    // CHUNK(4096) を跨ぐサイズ。
    for &n in &[100usize, 4096, 4096 * 2 + 17] {
        let raw = random_f16_matrix(n as u64 + 3, n, 0.01);
        let a_t = Tensor::new(raw.clone(), &[n]).unwrap();
        let out = TypedOps::<f16>::sum(&ops, &a_t, None).unwrap();

        let mut acc = 0.0f64;
        for &v in &raw {
            acc += v.to_f32() as f64;
        }
        let expected = f16::from_f32(acc as f32).to_f32();
        assert_parity(
            &format!("sum n={n}"),
            &to_f32_vec(&out.host_slice()),
            &[expected],
        );
    }
}

#[test]
fn sum_axis_matches_reference() {
    let ops = CpuBackendOps::new();
    let raw = to_f16_vec(&[1.0, 2.0, 3.0, 4.0]);
    let a_t = Tensor::new(raw, &[2, 2]).unwrap();
    let out = TypedOps::<f16>::sum(&ops, &a_t, Some(0)).unwrap();
    assert_parity("sum axis0", &to_f32_vec(&out.host_slice()), &[4.0, 6.0]);
}

#[test]
fn max_matches_f32_reference() {
    let ops = CpuBackendOps::new();
    let raw = random_f16_matrix(11, 1000, 0.03);
    let a_t = Tensor::new(raw.clone(), &[1000]).unwrap();
    let out = TypedOps::<f16>::max(&ops, &a_t, None).unwrap();
    let expected = raw
        .iter()
        .map(|v| v.to_f32())
        .fold(f32::NEG_INFINITY, f32::max);
    let expected = f16::from_f32(expected).to_f32();
    assert_parity("max all", &to_f32_vec(&out.host_slice()), &[expected]);
}

#[test]
fn max_axis_matches_reference() {
    let ops = CpuBackendOps::new();
    let raw = to_f16_vec(&[1.0, 5.0, 3.0, 2.0]);
    let a_t = Tensor::new(raw, &[2, 2]).unwrap();
    let out = TypedOps::<f16>::max(&ops, &a_t, Some(1)).unwrap();
    assert_parity("max axis1", &to_f32_vec(&out.host_slice()), &[5.0, 3.0]);
}

// --- 非 contiguous view ---

#[test]
fn non_contiguous_transpose_view_matches_contiguous_gemm_sum() {
    let ops = CpuBackendOps::new();
    let raw = to_f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let a_t = Tensor::new(raw, &[2, 3]).unwrap();
    let a_transposed = a_t.transpose_2d().unwrap();
    let a_transposed_contig = a_transposed.contiguous();

    let sum_view = TypedOps::<f16>::sum(&ops, &a_transposed, None).unwrap();
    let sum_contig = TypedOps::<f16>::sum(&ops, &a_transposed_contig, None).unwrap();
    assert_eq!(
        to_f32_vec(&sum_view.host_slice()),
        to_f32_vec(&sum_contig.host_slice())
    );

    // gemm: 転置 view（shape [3,2]）を A 側に渡し、事前に contiguous 化した
    // 同値と一致することを確認する（B は k=2 に合わせ shape [2,2]）。
    let b = to_f16_vec(&[1.0, 0.0, 0.0, 1.0]);
    let b_t = Tensor::new(b, &[2, 2]).unwrap();
    let gemm_view = TypedOps::<f16>::gemm(&ops, &a_transposed, &b_t).unwrap();
    let gemm_contig = TypedOps::<f16>::gemm(&ops, &a_transposed_contig, &b_t).unwrap();
    assert_eq!(
        to_f32_vec(&gemm_view.host_slice()),
        to_f32_vec(&gemm_contig.host_slice())
    );
}

// --- エラー経路 ---

#[test]
fn add_shape_mismatch_returns_typed_error() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(to_f16_vec(&[1.0, 2.0]), &[2]).unwrap();
    let b = Tensor::new(to_f16_vec(&[1.0, 2.0, 3.0]), &[3]).unwrap();
    let err = TypedOps::<f16>::add(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn max_empty_reduction_is_typed_error_not_panic() {
    let ops = CpuBackendOps::new();
    let empty = Tensor::new(Vec::<f16>::new(), &[0]).unwrap();
    let err = TypedOps::<f16>::max(&ops, &empty, None).unwrap_err();
    assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
}

#[test]
fn sum_empty_tensor_is_zero() {
    let ops = CpuBackendOps::new();
    let empty = Tensor::new(Vec::<f16>::new(), &[0]).unwrap();
    let out = TypedOps::<f16>::sum(&ops, &empty, None).unwrap();
    assert_eq!(to_f32_vec(&out.host_slice()), vec![0.0]);
}

// --- 端点 ---

#[test]
fn single_element_and_empty_tensor_endpoints() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(to_f16_vec(&[3.0]), &[1, 1]).unwrap();
    let b = Tensor::new(to_f16_vec(&[4.0]), &[1, 1]).unwrap();
    let gemm_out = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap();
    assert_eq!(to_f32_vec(&gemm_out.host_slice()), vec![12.0]);

    let empty_a = Tensor::new(Vec::<f16>::new(), &[0, 3]).unwrap();
    let empty_b = Tensor::new(Vec::<f16>::new(), &[3, 0]).unwrap();
    let empty_gemm = TypedOps::<f16>::gemm(&ops, &empty_a, &empty_b).unwrap();
    assert_eq!(empty_gemm.shape(), &[0, 0]);

    let empty_relu = Tensor::new(Vec::<f16>::new(), &[0]).unwrap();
    let relu_out = TypedOps::<f16>::relu(&ops, &empty_relu).unwrap();
    assert_eq!(relu_out.shape(), &[0]);
}
