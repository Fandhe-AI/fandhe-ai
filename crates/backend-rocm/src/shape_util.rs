//! 添字系演算（gather/scatter・interpolate）のホスト側 shape 補助（イシュー #3439・R8）。
//!
//! 役割: shape の要素数・行優先ストライドを overflow なしで求め、カーネル引数型（`int`）へ
//! 収まる形で返す。`gather_scatter.rs`・`interpolate.rs` が起動前の fail-closed 検証に使う
//! （`backend-cuda::gather_scatter` の `checked_numel`／`shape_to_i32`／`row_major_strides_i32`
//! 相当。エラーは本クレートの `RocmError`）。

use crate::error::RocmError;

fn fit_i32(op: &'static str, value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

/// 要素数（積）。overflow は `SizeLimitExceeded`。
pub(crate) fn checked_numel(op: &'static str, shape: &[usize]) -> Result<usize, RocmError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &s| acc.checked_mul(s))
        .ok_or(RocmError::SizeLimitExceeded {
            op,
            n: usize::MAX,
            limit: usize::MAX,
        })
}

/// 各軸サイズを `i32` 配列へ。
pub(crate) fn shape_to_i32(op: &'static str, shape: &[usize]) -> Result<Vec<i32>, RocmError> {
    shape.iter().map(|&s| fit_i32(op, s, "shape 軸")).collect()
}

/// 行優先ストライド（要素単位）を `i32` 配列で返す。0 サイズ軸があっても積は 0 になるだけで
/// overflow しない範囲に限る（呼び出し側は numel が 0 のとき起動しない）。
pub(crate) fn row_major_strides_i32(
    op: &'static str,
    shape: &[usize],
) -> Result<Vec<i32>, RocmError> {
    let mut strides = vec![0i32; shape.len()];
    let mut acc: usize = 1;
    for (a, &s) in shape.iter().enumerate().rev() {
        strides[a] = fit_i32(op, acc, "stride")?;
        acc = acc.checked_mul(s).ok_or(RocmError::SizeLimitExceeded {
            op,
            n: usize::MAX,
            limit: usize::MAX,
        })?;
    }
    Ok(strides)
}

/// `i32` の flat 添字として使える最大要素数の検証。
pub(crate) fn fit_numel_i32(op: &'static str, numel: usize, name: &str) -> Result<i32, RocmError> {
    fit_i32(op, numel, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_strides() {
        assert_eq!(checked_numel("t", &[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel("t", &[]).unwrap(), 1);
        assert!(checked_numel("t", &[usize::MAX, 2]).is_err());
        assert_eq!(row_major_strides_i32("t", &[2, 3, 4]).unwrap(), [12, 4, 1]);
        assert_eq!(row_major_strides_i32("t", &[5]).unwrap(), [1]);
    }

    #[test]
    fn i32_overflow_is_rejected() {
        let big = i32::MAX as usize + 1;
        assert!(shape_to_i32("t", &[2, big]).is_err());
        assert!(row_major_strides_i32("t", &[2, big, 2]).is_err());
        assert!(fit_numel_i32("t", big, "numel").is_err());
        assert_eq!(fit_numel_i32("t", 7, "numel").unwrap(), 7);
    }
}
