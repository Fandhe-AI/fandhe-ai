//! numpy `np.array` 慣習のテンソル生成入口（TASK-9.2a・#95）。
//!
//! **非推奨シム（TASK-9.4・#411）**: 唯一のサポート対象実装は
//! `fandhe_ai::compat::array`（`crates/facade/src/compat/array.rs`）。
//! 本ファイルは移設前の実装を移行期間中のソース互換のためだけに残す
//! （`compat/mod.rs` モジュール doc 参照）。
//!
//! `compat::array()` は 1-D/2-D の Rust ネイティブ構造から
//! `fandhe_ai_tensor_core::Tensor<f32>` を組み立てるだけの薄いラッパーで、
//! 数値計算・shape 検査ロジックは持たない（すべて `Tensor::new` へ
//! 委譲する。REQ-9）。ネストが不揃い（jagged）な 2-D 入力だけは
//! `Tensor::new` に到達する前に検出する必要がある（各行を平坦化して
//! 連結した時点で「どの行が短い／長いか」という情報が失われ、
//! `Tensor::new` の要素数積検査だけでは jagged 入力を個別に指摘できない
//! ため。A03: 外部由来入力を計算前に検証する契約。`.claude/rules/
//! security.md`）。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;

/// `compat::array()` が受理する入力の変換 trait。numpy `np.array` の
/// 「ネストしたリスト/配列から shape を推論する」慣習を模す。
///
/// **非推奨シム（TASK-9.4・#411）**: `fandhe_ai::compat::array::ArrayData` が
/// 唯一のサポート対象実装（`compat/mod.rs` モジュール doc 参照）。
#[deprecated(
    since = "0.0.0",
    note = "fandhe_ai::compat::array::ArrayData へ移設済み（TASK-9.4・#411）。移行期間中の非推奨シム"
)]
pub trait ArrayData {
    /// 行優先で平坦化したデータ列と shape を返す。jagged 入力はここで
    /// 検証し `AutodiffError::InvalidArgument` を返す（モジュール doc
    /// 参照）。
    fn into_array_data(self) -> Result<(Vec<f32>, Vec<usize>), AutodiffError>;
}

/// 1-D: `Vec<f32>` → shape `[n]`。
#[allow(deprecated)] // ArrayData 自体が非推奨シム（本ファイル冒頭 doc 参照）。
impl ArrayData for Vec<f32> {
    fn into_array_data(self) -> Result<(Vec<f32>, Vec<usize>), AutodiffError> {
        let n = self.len();
        Ok((self, vec![n]))
    }
}

/// 1-D: `&[f32]` → shape `[n]`（借用入力向け。複製して所有データにする）。
#[allow(deprecated)]
impl ArrayData for &[f32] {
    fn into_array_data(self) -> Result<(Vec<f32>, Vec<usize>), AutodiffError> {
        Ok((self.to_vec(), vec![self.len()]))
    }
}

/// 1-D 配列リテラル `[f32; N]` → shape `[N]`。
#[allow(deprecated)]
impl<const N: usize> ArrayData for [f32; N] {
    fn into_array_data(self) -> Result<(Vec<f32>, Vec<usize>), AutodiffError> {
        Ok((self.to_vec(), vec![N]))
    }
}

/// 2-D: `Vec<Vec<f32>>` → 行優先で平坦化し shape `[rows, cols]`。
/// 行長不一致（jagged）は計算前に検出する（モジュール doc 参照）。
#[allow(deprecated)]
impl ArrayData for Vec<Vec<f32>> {
    fn into_array_data(self) -> Result<(Vec<f32>, Vec<usize>), AutodiffError> {
        let rows = self.len();
        let cols = self.first().map_or(0, |row| row.len());
        // jagged 検証を `rows * cols` の確保より先に行う（A03: 外部由来
        // 入力を計算前に検証する契約。`.claude/rules/security.md`）。
        // 先頭行だけが極端に長く残りの行が短い jagged 入力に対し、
        // 検証前の `with_capacity(rows * cols)` が巨大確保でメモリ確保
        // 異常終了しうるため、全行の長さを先に走査してから確保する。
        for (i, row) in self.iter().enumerate() {
            if row.len() != cols {
                return Err(AutodiffError::InvalidArgument(format!(
                    "compat::array: jagged 2-D 入力（行 {i} の長さ {} が行 0 の長さ {cols} と不一致）",
                    row.len()
                )));
            }
        }
        let mut data = Vec::with_capacity(rows * cols);
        for row in self {
            data.extend(row);
        }
        Ok((data, vec![rows, cols]))
    }
}

/// 2-D 配列リテラル `[[f32; N]; M]` → shape `[M, N]`（固定長のため
/// 行長は型で保証され jagged になりえない。検証不要）。
#[allow(deprecated)]
impl<const M: usize, const N: usize> ArrayData for [[f32; N]; M] {
    fn into_array_data(self) -> Result<(Vec<f32>, Vec<usize>), AutodiffError> {
        let mut data = Vec::with_capacity(M * N);
        for row in self {
            data.extend(row);
        }
        Ok((data, vec![M, N]))
    }
}

/// numpy `np.array` 慣習のテンソル生成。1-D（`Vec<f32>`/`&[f32]`/
/// `[f32; N]`）・2-D（`Vec<Vec<f32>>`/`[[f32; N]; M]`）を受理し、
/// ネストから shape を推論して `fandhe_ai_tensor_core::Tensor<f32>` を返す
/// （`Tensor::new` への委譲のみ。モジュール doc 参照）。
///
/// **非推奨シム（TASK-9.4・#411）**: `fandhe_ai::compat::array` へ移行する
/// こと（`compat/mod.rs` モジュール doc 参照）。
#[deprecated(
    since = "0.0.0",
    note = "fandhe_ai::compat::array へ移設済み（TASK-9.4・#411）。移行期間中の非推奨シム"
)]
#[allow(deprecated)] // ArrayData 境界自体が非推奨のため。
pub fn array<A: ArrayData>(data: A) -> Result<Tensor<f32>, AutodiffError> {
    let (flat, shape) = data.into_array_data()?;
    Ok(Tensor::new(flat, &shape)?)
}

#[cfg(test)]
#[allow(deprecated)] // 非推奨シム自身のテスト（本ファイル冒頭 doc 参照）。
mod tests {
    use super::*;
    use crate::eval::dense_vec;

    #[test]
    fn array_1d_vec_infers_shape() {
        let t = array(vec![1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(dense_vec(&t), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn array_1d_slice_infers_shape() {
        let src = [1.0_f32, 2.0, 3.0];
        let t = array(src.as_slice()).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(dense_vec(&t), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn array_1d_literal_infers_shape() {
        let t = array([1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(dense_vec(&t), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn array_2d_vec_infers_shape_row_major() {
        let t = array(vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(dense_vec(&t), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn array_2d_literal_infers_shape_row_major() {
        let t = array([[1.0_f32, 2.0], [3.0, 4.0]]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(dense_vec(&t), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn array_2d_jagged_input_is_rejected() {
        let err = array(vec![vec![1.0_f32, 2.0], vec![3.0]]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn array_empty_1d_is_valid_zero_size_axis() {
        let t = array(Vec::<f32>::new()).unwrap();
        assert_eq!(t.shape(), &[0]);
    }
}
