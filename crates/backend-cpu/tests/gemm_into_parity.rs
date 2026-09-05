//! `CpuBackendOps::gemm_fp32_strict_into`／`upload_into` の受け入れ基準
//! 対応テスト（イシュー #1212）。
//!
//! `fandhe_ai_tensor_core` に非破壊追加した 2 メソッド（`BackendOps::
//! gemm_fp32_strict_into`・`MemoryOps::upload_into`。デフォルトは
//! `BackendError::Unsupported`）の CPU 実装が、既存の
//! `gemm_fp32_strict`／`upload` と bit 完全一致すること、および NaN
//! 事前充填・オフセット非 0・範囲外拒否の各契約を満たすことを検証する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// `gemm_fp32_strict_into` が `out_offset` から書き込む結果は、同じ
/// `a`／`b` に対する `gemm_fp32_strict` と bit 完全一致する（NaN で
/// 事前充填した永続バッファに対しても、対象範囲以外は変更されず、
/// 対象範囲は上書き契約〈累積ではない〉を満たすことを併せて確認する）。
#[test]
fn gemm_fp32_strict_into_matches_gemm_fp32_strict_with_offset() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();

    for &(m, k, n, prefix, suffix) in &[
        (1usize, 1usize, 1usize, 0usize, 0usize),
        (4, 8, 4, 3, 5),
        (37, 65, 33, 11, 0),
        (128, 129, 96, 0, 7),
    ] {
        let a = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
        let b = tensor(random_matrix(0x2000 + n as u64, k * n), &[k, n]);
        let expected = ops.gemm_fp32_strict(&a, &b).unwrap();
        let expected_slice = expected.contiguous();
        let expected_data = expected_slice.as_slice().unwrap();

        // 永続バッファを模した「対象範囲以外は NaN 事前充填」の
        // `DeviceBuffer`。前方 `prefix` 要素・後方 `suffix` 要素が
        // 破壊されないことを検証する（`docs/perf/
        // train-resident-grad-device-update.md` の NaN 事前充填契約）。
        let total = prefix + m * n + suffix;
        let seed = tensor(vec![f32::NAN; total], &[total]);
        let mut staging = mem.upload(&seed).unwrap();

        ops.gemm_fp32_strict_into(&a, &b, &mut staging, prefix)
            .unwrap();

        let readback = mem.download(&staging).unwrap();
        let readback = readback.contiguous();
        let readback_data = readback.as_slice().unwrap();

        assert_eq!(
            &readback_data[prefix..prefix + m * n],
            expected_data,
            "gemm_fp32_strict_into は gemm_fp32_strict と bit 完全一致するはず（m={m} k={k} \
             n={n} prefix={prefix}）"
        );
        assert!(
            readback_data[..prefix].iter().all(|v| v.is_nan()),
            "対象範囲より前の NaN 事前充填領域が変更された（m={m} k={k} n={n}）"
        );
        assert!(
            readback_data[prefix + m * n..].iter().all(|v| v.is_nan()),
            "対象範囲より後の NaN 事前充填領域が変更された（m={m} k={k} n={n}）"
        );
    }
}

/// 範囲外書き込み（`out_offset + m*n > out.numel()`）は `InvalidArgument`
/// で拒否される（REQ-8「カーネル側の手動境界チェックを省略しない」・
/// OWASP A03）。
#[test]
fn gemm_fp32_strict_into_rejects_out_of_range_offset() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // out は 4 要素分（2x2 の結果ちょうど）しかないため offset=1 は
    // 範囲外になる。
    let seed = tensor(vec![0.0f32; 4], &[4]);
    let mut staging = mem.upload(&seed).unwrap();

    let err = ops
        .gemm_fp32_strict_into(&a, &b, &mut staging, 1)
        .unwrap_err();
    assert!(
        matches!(err, BackendError::InvalidArgument(_)),
        "範囲外オフセットは InvalidArgument であるべき: {err:?}"
    );
}

/// `upload_into` が `upload` と bit 完全一致し、NaN 事前充填した対象外
/// 領域を破壊しないことを確認する。
#[test]
fn upload_into_matches_upload_with_offset() {
    let mem = CpuMemory::new();
    let data = vec![1.0f32, -2.5, 3.25, f32::MIN_POSITIVE, f32::MAX];
    let src = tensor(data.clone(), &[5]);

    let expected = mem.upload(&src).unwrap();
    let expected_readback = mem.download(&expected).unwrap();

    let seed = tensor(vec![f32::NAN; 8], &[8]);
    let mut staging = mem.upload(&seed).unwrap();
    mem.upload_into(&src, &mut staging, 2).unwrap();

    let readback = mem.download(&staging).unwrap();
    let readback = readback.contiguous();
    let readback_data = readback.as_slice().unwrap();
    let expected_data = expected_readback.contiguous();
    let expected_data = expected_data.as_slice().unwrap();

    assert_eq!(&readback_data[2..7], expected_data);
    assert!(readback_data[..2].iter().all(|v| v.is_nan()));
    assert!(readback_data[7..].iter().all(|v| v.is_nan()));
}

/// `upload_into` の範囲外書き込みは `InvalidArgument` で拒否される。
#[test]
fn upload_into_rejects_out_of_range_offset() {
    let mem = CpuMemory::new();
    let src = tensor(vec![1.0, 2.0, 3.0], &[3]);
    let seed = tensor(vec![0.0f32; 4], &[4]);
    let mut staging = mem.upload(&seed).unwrap();

    let err = mem.upload_into(&src, &mut staging, 2).unwrap_err();
    assert!(
        matches!(err, BackendError::InvalidArgument(_)),
        "範囲外オフセットは InvalidArgument であるべき: {err:?}"
    );
}
