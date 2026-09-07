//! `fandhe_ai::ChecksumReadout`／`GemmChecksum`（イシュー #1339）が
//! facade 経由で CPU バックエンドまで結線されていることを検証する
//! （受入条件: composition root から `Var::matmul_checksum` へ到達
//! できる。`tape_construction.rs` と同じ既定 CPU 経路のみを対象とし、
//! CUDA／Metal（本イシュー時点で `BackendOps::gemm_checksum` 未実装。
//! `crates/tensor-core/src/backend_ops.rs` デフォルト実装 doc 参照）は
//! 対象外）。

use fandhe_ai::ChecksumReadout;
use fandhe_ai_tensor_core::Tensor;

fn sample_tensor() -> Tensor<f32> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("sample tensor は shape が一致する")
}

#[test]
fn matmul_checksum_reaches_cpu_backend_via_facade_tape() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&sample_tensor());
    let b = tape.var(&sample_tensor());

    let reference = a.matmul(&b).expect("matmul は成功する");
    let expected: f64 = reference
        .value()
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| x as f64)
        .sum();

    let result = a
        .matmul_checksum(&b, ChecksumReadout::ChecksumOnly)
        .expect("CPU バックエンドは gemm_checksum を実装済み");

    assert_eq!(result.checksum, expected);
    assert!(result.output.is_none());
}
