//! `CpuBackendOps::gemm_resident_rhs`／`gemm_resident_lhs` の受け入れ基準
//! 対応テスト（イシュー #1022）。
//!
//! `fandhe_ai_tensor_core::BackendOps` に非破壊追加した 2 メソッド
//! （`crates/tensor-core/src/backend_ops.rs`。デフォルトは
//! `BackendError::Unsupported`）の CPU 実装が、既存の `gemm`／
//! `gemm_bias_act`（ホスト常駐オペランド版）と同一の計算結果を返す
//! ことを検証する（`w`／`bias` をデバイス常駐バッファへ upload してから
//! 呼ぶ点のみが異なり、数式は同一のため REQ-2 統一複合判定ではなく
//! bit 完全一致で比較する）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// `gemm_resident_rhs(a, w_dev, bias_dev)` が `gemm_bias_act(a, w, bias,
/// None)`（ホスト常駐版）と bit 完全一致することを、MR/NR/MC/KC/NC
/// 境界を跨ぐ複数形状で確認する（`gemm_epilogue_parity.rs` と同じ形状
/// グリッド方針）。
#[test]
fn gemm_resident_rhs_matches_host_gemm_bias_act_across_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();

    for &(m, k, n) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33), (128, 129, 96)] {
        for has_bias in [false, true] {
            let a = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
            let w = tensor(random_matrix(0x2000 + n as u64, k * n), &[k, n]);
            let bias = has_bias.then(|| tensor(random_matrix(0x3000 + n as u64, n), &[n]));

            let expected = ops
                .gemm_bias_act(&a, &w, bias.as_ref(), Activation::None)
                .unwrap();

            let w_dev = mem.upload(&w).unwrap();
            let bias_dev = bias.as_ref().map(|b| mem.upload(b).unwrap());
            let actual = ops
                .gemm_resident_rhs(&a, &w_dev, bias_dev.as_ref())
                .unwrap();

            assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
            let a_data = actual.contiguous();
            let e_data = expected.contiguous();
            assert_eq!(
                a_data.as_slice().unwrap(),
                e_data.as_slice().unwrap(),
                "gemm_resident_rhs は gemm_bias_act と bit 完全一致するはず（m={m} k={k} n={n} \
                 has_bias={has_bias}）"
            );
        }
    }
}

/// `gemm_resident_lhs(w_dev, b)` が `gemm(w, b)`（ホスト常駐版）と
/// bit 完全一致することを確認する。
#[test]
fn gemm_resident_lhs_matches_host_gemm_across_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();

    for &(p, q, r) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33)] {
        let w = tensor(random_matrix(0x4000 + p as u64, p * q), &[p, q]);
        let b = tensor(random_matrix(0x5000 + r as u64, q * r), &[q, r]);

        let expected = ops.gemm(&w, &b).unwrap();

        let w_dev = mem.upload(&w).unwrap();
        let actual = ops.gemm_resident_lhs(&w_dev, &b).unwrap();

        assert_eq!(actual.shape(), expected.shape(), "p={p} q={q} r={r}");
        let a_data = actual.contiguous();
        let e_data = expected.contiguous();
        assert_eq!(
            a_data.as_slice().unwrap(),
            e_data.as_slice().unwrap(),
            "gemm_resident_lhs は gemm と bit 完全一致するはず（p={p} q={q} r={r}）"
        );
    }
}

/// `w`（デバイス常駐）の shape が `a` の列数と噛み合わない場合は
/// カーネル本体へ触れる前に `ShapeMismatch` を返す（REQ-8・OWASP A03）。
#[test]
fn gemm_resident_rhs_rejects_incompatible_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    let w = tensor(vec![1.0, 2.0], &[2, 1]);
    let w_dev = mem.upload(&w).unwrap();
    let err = ops.gemm_resident_rhs(&a, &w_dev, None).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::ShapeMismatch(_)
    ));
}

/// `bias` の shape が `[n]` 厳密一致でない場合も同様に拒否する。
#[test]
fn gemm_resident_rhs_rejects_bias_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0], &[1, 2]);
    let w = tensor(vec![1.0, 2.0], &[2, 1]);
    let bias = tensor(vec![1.0, 2.0], &[2]);
    let w_dev = mem.upload(&w).unwrap();
    let bias_dev = mem.upload(&bias).unwrap();
    let err = ops
        .gemm_resident_rhs(&a, &w_dev, Some(&bias_dev))
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::ShapeMismatch(_)
    ));
}
