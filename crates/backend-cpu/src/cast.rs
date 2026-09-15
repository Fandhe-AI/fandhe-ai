//! `CastOps` の CPU 実装（イシュー #1750・親 #1613）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::cast_ops`] accessor（`ops.rs`
//! 参照）経由でのみ到達する dtype 変換カーネル本体。8 方向すべてを
//! `fandhe_ai_tensor_core::cast::{cast_from_f32, cast_to_f32}`（ホスト
//! 参照実装。`tensor-core::cast` モジュール doc「単一情報源」参照）へ
//! 委譲する——`gather_scatter.rs`／`unique.rs` のように独自アルゴリズムを
//! 複製する必要がなく、算術を含まない純粋な選択・変換のため乖離の
//! リスクがない（`crate::interpolate` のような座標計算の複製とは異なる
//! 判断）。並列化しない（性能最適化は
//! `.claude/rules/out-of-scope-tracking.md` 対象）。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{CastOps, Tensor, cast_from_f32, cast_to_f32};

/// `ShapeError` → `BackendError::ShapeMismatch` への変換
/// （他の CPU カーネル実装〈`unique::unique` の呼び出し元〉と同じ
/// 写像規約）。
fn to_backend_error(err: fandhe_ai_tensor_core::ShapeError) -> BackendError {
    BackendError::ShapeMismatch(err)
}

impl CastOps for crate::ops::CpuBackendOps {
    fn cast_f32_to_f64(&self, x: &Tensor<f32>) -> Result<Tensor<f64>, BackendError> {
        cast_from_f32(x).map_err(to_backend_error)
    }

    fn cast_f32_to_i32(&self, x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> {
        cast_from_f32(x).map_err(to_backend_error)
    }

    fn cast_f32_to_i64(&self, x: &Tensor<f32>) -> Result<Tensor<i64>, BackendError> {
        cast_from_f32(x).map_err(to_backend_error)
    }

    fn cast_f32_to_bool(&self, x: &Tensor<f32>) -> Result<Tensor<bool>, BackendError> {
        cast_from_f32(x).map_err(to_backend_error)
    }

    fn cast_f64_to_f32(&self, x: &Tensor<f64>) -> Result<Tensor<f32>, BackendError> {
        cast_to_f32(x).map_err(to_backend_error)
    }

    fn cast_i32_to_f32(&self, x: &Tensor<i32>) -> Result<Tensor<f32>, BackendError> {
        cast_to_f32(x).map_err(to_backend_error)
    }

    fn cast_i64_to_f32(&self, x: &Tensor<i64>) -> Result<Tensor<f32>, BackendError> {
        cast_to_f32(x).map_err(to_backend_error)
    }

    fn cast_bool_to_f32(&self, x: &Tensor<bool>) -> Result<Tensor<f32>, BackendError> {
        cast_to_f32(x).map_err(to_backend_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::CpuBackendOps;
    use fandhe_ai_tensor_core::BackendOps;

    fn t<T: fandhe_ai_tensor_core::Element>(data: Vec<T>, shape: &[usize]) -> Tensor<T> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    /// `cast_ops` accessor が `Some(self)` を返し、`&dyn BackendOps`
    /// 経由でも到達できることを確認する（`typed_ops_f64` の同型テスト
    /// と同じ意図）。
    #[test]
    fn cast_ops_accessor_is_some() {
        let ops = CpuBackendOps;
        let backend_ops: &dyn BackendOps = &ops;
        assert!(backend_ops.cast_ops().is_some());
    }

    /// 8 方向すべてが `tensor_core::cast` のホスト参照実装と bit 一致
    /// することを確認する（`CpuBackendOps` は委譲するだけのため、実質
    /// 「委譲が正しく配線されている」ことの検証。数値契約の詳細検証は
    /// `tensor-core::cast::tests` を正とする）。
    #[test]
    fn cpu_cast_matches_host_reference_all_directions() {
        let ops = CpuBackendOps;

        let f32_in = t(vec![1.5f32, -2.5, 0.0, f32::NAN], &[4]);
        // f64_out/f64_ref は NaN を含みうるため `to_bits()` で比較する
        // （`NaN != NaN` のため `assert_eq!` 直接比較は使えない。
        // `unique.rs` の NaN テストと同じ理由）。
        let f64_out = ops.cast_f32_to_f64(&f32_in).unwrap();
        let f64_ref: Tensor<f64> = cast_from_f32(&f32_in).unwrap();
        let f64_out_bits: Vec<u64> = f64_out.host_slice().iter().map(|v| v.to_bits()).collect();
        let f64_ref_bits: Vec<u64> = f64_ref.host_slice().iter().map(|v| v.to_bits()).collect();
        assert_eq!(f64_out_bits, f64_ref_bits);

        let i32_out = ops.cast_f32_to_i32(&f32_in).unwrap();
        let i32_ref: Tensor<i32> = cast_from_f32(&f32_in).unwrap();
        assert_eq!(
            i32_out.host_slice().into_owned(),
            i32_ref.host_slice().into_owned()
        );

        let i64_out = ops.cast_f32_to_i64(&f32_in).unwrap();
        let i64_ref: Tensor<i64> = cast_from_f32(&f32_in).unwrap();
        assert_eq!(
            i64_out.host_slice().into_owned(),
            i64_ref.host_slice().into_owned()
        );

        let bool_out = ops.cast_f32_to_bool(&f32_in).unwrap();
        let bool_ref: Tensor<bool> = cast_from_f32(&f32_in).unwrap();
        assert_eq!(
            bool_out.host_slice().into_owned(),
            bool_ref.host_slice().into_owned()
        );

        let f64_in = t(vec![1.0f64, f64::MAX], &[2]);
        let back_f32 = ops.cast_f64_to_f32(&f64_in).unwrap();
        let back_f32_ref = cast_to_f32(&f64_in).unwrap();
        assert_eq!(
            back_f32.host_slice().into_owned(),
            back_f32_ref.host_slice().into_owned()
        );

        let i32_in = t(vec![i32::MIN, i32::MAX], &[2]);
        let from_i32 = ops.cast_i32_to_f32(&i32_in).unwrap();
        let from_i32_ref = cast_to_f32(&i32_in).unwrap();
        assert_eq!(
            from_i32.host_slice().into_owned(),
            from_i32_ref.host_slice().into_owned()
        );

        let i64_in = t(vec![i64::MIN, i64::MAX], &[2]);
        let from_i64 = ops.cast_i64_to_f32(&i64_in).unwrap();
        let from_i64_ref = cast_to_f32(&i64_in).unwrap();
        assert_eq!(
            from_i64.host_slice().into_owned(),
            from_i64_ref.host_slice().into_owned()
        );

        let bool_in = t(vec![true, false], &[2]);
        let from_bool = ops.cast_bool_to_f32(&bool_in).unwrap();
        let from_bool_ref = cast_to_f32(&bool_in).unwrap();
        assert_eq!(
            from_bool.host_slice().into_owned(),
            from_bool_ref.host_slice().into_owned()
        );
    }

    /// 非 contiguous view（transpose 済み）を渡しても正しく稠密化される
    /// ことを確認する（`tensor_core::cast::cast_from_f32` の `host_slice`
    /// 経由の性質をそのまま引き継ぐ）。
    #[test]
    fn cpu_cast_handles_non_contiguous_view() {
        let ops = CpuBackendOps;
        let base = t(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let transposed = base
            .permute(&[1, 0])
            .expect("test fixture: rank 2 の permute は常に妥当");
        let out = ops.cast_f32_to_i32(&transposed).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.host_slice().into_owned(), vec![1, 4, 2, 5, 3, 6]);
    }
}
