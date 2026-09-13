//! `TypedOps<f64>`（イシュー #1697・親 #1649）の統合テスト。
//!
//! `CpuBackendOps` を `&dyn BackendOps` として保持したまま
//! `BackendOps::typed_ops_f64()` accessor を経由して `TypedOps<f64>` を
//! 取得できること（capability accessor パターンの受け入れ条件）・
//! 各演算 8 種が [`fandhe_ai_backend_cpu::parity::matmul_reference_fma_f64`]
//! 等の逐次参照実装と bit 完全一致すること・空縮約／shape 不整合エラー
//! 経路・f32 版との REQ-2 複合判定一致を検証する。
//!
//! `typed_f64` 自体のクレート内単体テスト（`crates/backend-cpu/src/
//! typed_f64.rs` の `#[cfg(test)] mod tests`）は手計算値中心のため、本
//! ファイルは決定的乱数入力（`bench_harness::rng::Xorshift64Star`）による
//! より大きな形状・`PARALLEL_THRESHOLD` を跨ぐサイズでの並列経路検証を担う。
//!
//! 実機依存なし（CPU 参照実装のみ）のため `#[ignore]` 分離は不要。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, compare_f64, matmul_reference_fma_f64};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

/// `elementwise.rs::PARALLEL_THRESHOLD`（1 << 15）と同じ値。`typed_f64`
/// 側の閾値定数は `pub(crate)` のため統合テストからは参照できず、並列
/// 経路を確実に踏むテストサイズの根拠としてここに複製する（値のみの
/// 複製であり判定ロジックの複製ではない）。
const PARALLEL_THRESHOLD: usize = 1 << 15;

/// `[-0.5, 0.5)` の範囲に収まる決定的 f64 乱数列を生成する。
/// `Xorshift64Star::next_f32()`（`[-1.0, 1.0)`）を f64 へ昇格してから
/// 0.5 倍する（GEMM K=64 程度の累積で桁あふれ・打ち切り誤差の偏りを
/// 避けるための入力レンジ限定。PoC-v2-1/3/5 の `next_f32` 方針を踏襲）。
fn next_f64_half(rng: &mut Xorshift64Star) -> f64 {
    rng.next_f32() as f64 * 0.5
}

fn fill_vec_f64(rng: &mut Xorshift64Star, len: usize) -> Vec<f64> {
    (0..len).map(|_| next_f64_half(rng)).collect()
}

#[test]
fn typed_ops_f64_accessor_is_some_through_dyn_backend_ops() {
    let ops = CpuBackendOps::new();
    let dyn_ops: &dyn BackendOps = &ops;
    assert!(dyn_ops.typed_ops_f64().is_some());
}

#[test]
fn gemm_f64_matches_sequential_reference_bit_exact() {
    let shapes = [
        (1usize, 1usize, 1usize),
        (3, 5, 7),
        (64, 64, 64),
        (129, 65, 33),
    ];
    let ops = CpuBackendOps::new();
    for (m, n, k) in shapes {
        let mut rng =
            Xorshift64Star::new(0xC0FFEE ^ (m as u64) ^ ((n as u64) << 8) ^ ((k as u64) << 16));
        let a_data = fill_vec_f64(&mut rng, m * k);
        let b_data = fill_vec_f64(&mut rng, k * n);
        let a = Tensor::new(a_data.clone(), &[m, k]).unwrap();
        let b = Tensor::new(b_data.clone(), &[k, n]).unwrap();

        let actual = TypedOps::<f64>::gemm(&ops, &a, &b).unwrap();

        let mut expected = vec![0.0f64; m * n];
        matmul_reference_fma_f64(&a_data, &b_data, &mut expected, m, n, k).unwrap();

        assert_eq!(
            actual.as_slice().unwrap(),
            expected.as_slice(),
            "shape ({m},{n},{k}) で bit 不一致"
        );
    }
}

#[test]
fn gemm_f64_non_contiguous_inputs() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0xABCDEF);
    let m = 5;
    let k = 4;
    let n = 3;
    // a は [k, m] を transpose_2d して [m, k] view にする（非 contiguous）。
    let a_data = fill_vec_f64(&mut rng, m * k);
    let a_km = Tensor::new(a_data, &[k, m]).unwrap();
    let a_view = a_km.transpose_2d().unwrap();
    let a_contig = a_view.contiguous();

    let b_data = fill_vec_f64(&mut rng, k * n);
    let b = Tensor::new(b_data, &[k, n]).unwrap();

    let via_view = TypedOps::<f64>::gemm(&ops, &a_view, &b).unwrap();
    let via_contig = TypedOps::<f64>::gemm(&ops, &a_contig, &b).unwrap();
    assert_eq!(via_view.as_slice().unwrap(), via_contig.as_slice().unwrap());
}

#[test]
fn gemm_f64_shape_mismatch_is_shape_error() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let err = TypedOps::<f64>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn elementwise_f64_bit_exact_vs_std() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0x1234_5678);

    // contiguous・PARALLEL_THRESHOLD 超のサイズ（並列経路を踏む）。
    let big_len = PARALLEL_THRESHOLD + 17;
    let a_big = fill_vec_f64(&mut rng, big_len);
    let b_big = fill_vec_f64(&mut rng, big_len);
    let a = Tensor::new(a_big.clone(), &[big_len]).unwrap();
    let b = Tensor::new(b_big.clone(), &[big_len]).unwrap();

    let add_out = TypedOps::<f64>::add(&ops, &a, &b).unwrap();
    let expected_add: Vec<f64> = a_big.iter().zip(&b_big).map(|(&x, &y)| x + y).collect();
    assert_eq!(add_out.as_slice().unwrap(), expected_add.as_slice());

    let mul_out = TypedOps::<f64>::mul(&ops, &a, &b).unwrap();
    let expected_mul: Vec<f64> = a_big.iter().zip(&b_big).map(|(&x, &y)| x * y).collect();
    assert_eq!(mul_out.as_slice().unwrap(), expected_mul.as_slice());

    let relu_out = TypedOps::<f64>::relu(&ops, &a).unwrap();
    let expected_relu: Vec<f64> = a_big.iter().map(|&x| x.max(0.0)).collect();
    assert_eq!(relu_out.as_slice().unwrap(), expected_relu.as_slice());

    let exp_out = TypedOps::<f64>::exp(&ops, &a).unwrap();
    let expected_exp: Vec<f64> = a_big.iter().map(|&x| x.exp()).collect();
    assert_eq!(exp_out.as_slice().unwrap(), expected_exp.as_slice());

    let tanh_out = TypedOps::<f64>::tanh(&ops, &a).unwrap();
    let expected_tanh: Vec<f64> = a_big.iter().map(|&x| x.tanh()).collect();
    assert_eq!(tanh_out.as_slice().unwrap(), expected_tanh.as_slice());

    // broadcast（`[2,3] + [3]`・`[1]`）。
    let a2 = Tensor::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b_row = Tensor::new(vec![10.0f64, 20.0, 30.0], &[3]).unwrap();
    let broadcast_out = TypedOps::<f64>::add(&ops, &a2, &b_row).unwrap();
    assert_eq!(
        broadcast_out.as_slice().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
    let scalar = Tensor::new(vec![2.0f64], &[1]).unwrap();
    let broadcast_scalar = TypedOps::<f64>::mul(&ops, &a2, &scalar).unwrap();
    assert_eq!(
        broadcast_scalar.as_slice().unwrap(),
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );

    // 非 contiguous view（transpose_2d）。
    let a3 = Tensor::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let a3_t = a3.transpose_2d().unwrap(); // [3, 2]
    let b3 = Tensor::new(vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();
    let via_view = TypedOps::<f64>::add(&ops, &a3_t, &b3).unwrap();
    let via_contig = TypedOps::<f64>::add(&ops, &a3_t.contiguous(), &b3).unwrap();
    assert_eq!(via_view.as_slice().unwrap(), via_contig.as_slice().unwrap());
}

#[test]
fn reduction_f64_bit_exact_vs_sequential() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(0x0BAD_F00D);

    // 全縮約: CHUNK（4096）を跨ぐ 10_000 要素。
    let n = 10_000;
    let data = fill_vec_f64(&mut rng, n);
    let a = Tensor::new(data.clone(), &[n]).unwrap();
    let sum_out = TypedOps::<f64>::sum(&ops, &a, None).unwrap();
    let expected_sum: f64 = data.iter().sum();
    assert_eq!(sum_out.as_slice().unwrap(), &[expected_sum]);

    let max_out = TypedOps::<f64>::max(&ops, &a, None).unwrap();
    let expected_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(max_out.as_slice().unwrap(), &[expected_max]);

    // 軸指定 rank 2: dim=Some(0)・dim=Some(1)。
    let m2 = 4;
    let n2 = 5;
    let data2 = fill_vec_f64(&mut rng, m2 * n2);
    let a2 = Tensor::new(data2.clone(), &[m2, n2]).unwrap();

    let sum_axis0 = TypedOps::<f64>::sum(&ops, &a2, Some(0)).unwrap();
    let mut expected_axis0 = vec![0.0f64; n2];
    for i in 0..m2 {
        for j in 0..n2 {
            expected_axis0[j] += data2[i * n2 + j];
        }
    }
    assert_eq!(sum_axis0.as_slice().unwrap(), expected_axis0.as_slice());

    let sum_axis1 = TypedOps::<f64>::sum(&ops, &a2, Some(1)).unwrap();
    let mut expected_axis1 = vec![0.0f64; m2];
    for i in 0..m2 {
        for j in 0..n2 {
            expected_axis1[i] += data2[i * n2 + j];
        }
    }
    assert_eq!(sum_axis1.as_slice().unwrap(), expected_axis1.as_slice());

    // 軸指定 rank 3: dim=Some(1)。
    let (d0, d1, d2) = (2, 3, 4);
    let data3 = fill_vec_f64(&mut rng, d0 * d1 * d2);
    let a3 = Tensor::new(data3.clone(), &[d0, d1, d2]).unwrap();
    let sum3 = TypedOps::<f64>::sum(&ops, &a3, Some(1)).unwrap();
    let mut expected3 = vec![0.0f64; d0 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for l in 0..d2 {
                expected3[i * d2 + l] += data3[i * d1 * d2 + j * d2 + l];
            }
        }
    }
    assert_eq!(sum3.as_slice().unwrap(), expected3.as_slice());

    // 非 contiguous view（transpose_2d）の全縮約。
    let a4 = Tensor::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let a4_t = a4.transpose_2d().unwrap();
    let sum_view = TypedOps::<f64>::sum(&ops, &a4_t, None).unwrap();
    let sum_contig = TypedOps::<f64>::sum(&ops, &a4_t.contiguous(), None).unwrap();
    assert_eq!(sum_view.as_slice().unwrap(), sum_contig.as_slice().unwrap());
}

#[test]
fn reduction_f64_empty_semantics() {
    let ops = CpuBackendOps::new();
    let empty = Tensor::new(Vec::<f64>::new(), &[0, 3]).unwrap();

    let sum_out = TypedOps::<f64>::sum(&ops, &empty, None).unwrap();
    assert_eq!(sum_out.as_slice().unwrap(), &[0.0]);

    let max_err = TypedOps::<f64>::max(&ops, &empty, None).unwrap_err();
    assert!(matches!(max_err, BackendError::KernelLaunchFailed(_)));

    // dim 範囲外は ShapeMismatch。
    let a = Tensor::new(vec![1.0f64, 2.0, 3.0], &[3]).unwrap();
    let err = TypedOps::<f64>::sum(&ops, &a, Some(5)).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn cross_dtype_f64_vs_f32_composite_parity() {
    // 同一入力（f32 表現可能値）で TypedOps<f64> と BackendOps（f32）の
    // 8 演算を実行し、REQ-2 複合判定（`compare_f64`。f32 出力は f64 へ
    // 昇格）で一致することを確認する。tolerance 定数は f32 版・f64 版で
    // 完全に共有（`crate::parity`）。
    let ops = CpuBackendOps::new();
    let k = 64;
    let mut rng = Xorshift64Star::new(0x5EED);

    let a32: Vec<f32> = (0..k * k).map(|_| rng.next_f32() * 0.5).collect();
    let b32: Vec<f32> = (0..k * k).map(|_| rng.next_f32() * 0.5).collect();
    let a64: Vec<f64> = a32.iter().map(|&v| v as f64).collect();
    let b64: Vec<f64> = b32.iter().map(|&v| v as f64).collect();

    let a32_t = Tensor::new(a32.clone(), &[k, k]).unwrap();
    let b32_t = Tensor::new(b32.clone(), &[k, k]).unwrap();
    let a64_t = Tensor::new(a64.clone(), &[k, k]).unwrap();
    let b64_t = Tensor::new(b64.clone(), &[k, k]).unwrap();

    let gemm32 = BackendOps::gemm(&ops, &a32_t, &b32_t).unwrap();
    let gemm64 = TypedOps::<f64>::gemm(&ops, &a64_t, &b64_t).unwrap();
    let gemm32_as_f64: Vec<f64> = gemm32
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let report = compare_f64(&gemm32_as_f64, gemm64.as_slice().unwrap()).unwrap();
    assert!(report.passes(), "gemm 複合判定 FAIL: {report:?}");

    macro_rules! assert_unary_composite {
        ($f32_method:ident, $f64_method:ident) => {{
            let out32 = BackendOps::$f32_method(&ops, &a32_t).unwrap();
            let out64 = TypedOps::<f64>::$f64_method(&ops, &a64_t).unwrap();
            let out32_as_f64: Vec<f64> = out32
                .as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as f64)
                .collect();
            let report = compare_f64(&out32_as_f64, out64.as_slice().unwrap()).unwrap();
            assert!(
                report.passes(),
                "{} 複合判定 FAIL: {:?}",
                stringify!($f32_method),
                report
            );
        }};
    }
    assert_unary_composite!(relu, relu);
    assert_unary_composite!(exp, exp);
    assert_unary_composite!(tanh, tanh);

    let add32 = BackendOps::add(&ops, &a32_t, &b32_t).unwrap();
    let add64 = TypedOps::<f64>::add(&ops, &a64_t, &b64_t).unwrap();
    let add32_as_f64: Vec<f64> = add32
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let report = compare_f64(&add32_as_f64, add64.as_slice().unwrap()).unwrap();
    assert!(report.passes(), "add 複合判定 FAIL: {report:?}");

    let mul32 = BackendOps::mul(&ops, &a32_t, &b32_t).unwrap();
    let mul64 = TypedOps::<f64>::mul(&ops, &a64_t, &b64_t).unwrap();
    let mul32_as_f64: Vec<f64> = mul32
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let report = compare_f64(&mul32_as_f64, mul64.as_slice().unwrap()).unwrap();
    assert!(report.passes(), "mul 複合判定 FAIL: {report:?}");

    let sum32 = BackendOps::sum(&ops, &a32_t, None).unwrap();
    let sum64 = TypedOps::<f64>::sum(&ops, &a64_t, None).unwrap();
    let sum32_as_f64: Vec<f64> = sum32
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let report = compare_f64(&sum32_as_f64, sum64.as_slice().unwrap()).unwrap();
    assert!(report.passes(), "sum 複合判定 FAIL: {report:?}");

    let max32 = BackendOps::max(&ops, &a32_t, None).unwrap();
    let max64 = TypedOps::<f64>::max(&ops, &a64_t, None).unwrap();
    let max32_as_f64: Vec<f64> = max32
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect();
    let report = compare_f64(&max32_as_f64, max64.as_slice().unwrap()).unwrap();
    assert!(report.passes(), "max 複合判定 FAIL: {report:?}");
}
