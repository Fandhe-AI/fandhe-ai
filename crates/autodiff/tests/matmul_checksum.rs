//! `Var::matmul_checksum`（イシュー #1339）の統合テスト。
//!
//! `common::naive_ops()`（`NaiveOps::gemm_checksum`）を経由し、
//! (1) `ChecksumOnly` が `matmul` と同じ数値の `f64` 和を返し `output`
//! を持たないこと、(2) `WithOutput` が `matmul` と bit 同一の `output`
//! を返すこと、(3) `matmul` と同じ検証順序（クロステープ検査・shape
//! 検査）を守ることを検証する。tape への非記録契約（`Var::matmul_checksum`
//! doc comment 参照）自体は `tape.rs::nodes` が `pub(crate)` であり
//! 統合テスト（別クレート扱い）から直接観測できないため、本ファイルでは
//! 検証しない。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{ChecksumReadout, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[test]
fn matmul_checksum_only_matches_matmul_host_f64_sum_and_has_no_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a_val = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b_val = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let a = tape.var(&a_val);
    let b = tape.var(&b_val);

    let reference = a.matmul(&b).unwrap();
    let expected: f64 = reference
        .value()
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| x as f64)
        .sum();

    let result = a
        .matmul_checksum(&b, ChecksumReadout::ChecksumOnly)
        .unwrap();

    assert_eq!(result.checksum, expected);
    assert!(result.output.is_none());
}

#[test]
fn matmul_checksum_with_output_is_bit_identical_to_matmul() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a_val = t(vec![1.0, -2.5, 3.25, 0.5, -1.0, 2.0], &[2, 3]);
    let b_val = t(vec![0.5, 1.5, -1.0, 2.0, 3.0, -2.0], &[3, 2]);
    let a = tape.var(&a_val);
    let b = tape.var(&b_val);

    let reference = a.matmul(&b).unwrap();
    let result = a.matmul_checksum(&b, ChecksumReadout::WithOutput).unwrap();

    let output = result.output.expect("WithOutput は output を返すはず");
    assert_eq!(
        output
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        reference
            .value()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "matmul_checksum(WithOutput).output は matmul と bit 同一のはず"
    );
}

#[test]
fn matmul_checksum_rejects_cross_tape_var() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let a_val = t(vec![1.0, 2.0], &[1, 2]);
    let b_val = t(vec![1.0, 2.0], &[2, 1]);
    let a = tape_a.var(&a_val);
    let b = tape_b.var(&b_val);

    let result = a.matmul_checksum(&b, ChecksumReadout::ChecksumOnly);

    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn matmul_checksum_rejects_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a_val = t(vec![1.0, 2.0, 3.0], &[1, 3]);
    let b_val = t(vec![1.0, 2.0], &[2, 1]);
    let a = tape.var(&a_val);
    let b = tape.var(&b_val);

    let result = a.matmul_checksum(&b, ChecksumReadout::ChecksumOnly);

    assert!(matches!(result, Err(AutodiffError::Shape(_))));
}
