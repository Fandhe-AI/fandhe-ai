//! `BackendOps::relu_backward_mask_device`（イシュー #3221・CUDA 実装）の
//! GPU 非依存な純関数群（形状・rank・layout の検証と、カーネルの逐語ホスト
//! モデル）。`backend-metal::relu_mask_model` と同じ設計判断で、CUDA ドライバ
//! に触れないため Linux の CI（GPU なし）でも単体テストが回る。
//!
//! 内部用・安定性保証なし（backend-cuda は facade の公開面に出ないサポート外の
//! 内部クレート。`docs/compat-api-scope.md` §0）。integration test から使う
//! ために `pub` としている。
//!
//! # 責務境界
//! - 本モジュール: shape／rank／layout の検証と `i32` 上限の検査
//!   ([`plan_relu_mask`])、`kernels_elementwise::EW_RELU_MASK_SELECT_F32` の
//!   添字計算と選択式を逐語再現するホストモデル
//!   ([`relu_mask_select_host_model`])。
//! - `crate::ops::CudaBackendOps::relu_backward_mask_device_impl`: device 一致
//!   検査・view の切り出し・確保・起動。
//!
//! # 意味論（設計 `docs/train-step-resident-parts-design.md` §3 行 3）
//! `out[i] = (mask[i] > 0.0) ? g[i] : +0.0`。乗算ではなく**選択**で、`g` の
//! NaN／inf／−0.0 は mask が真ならビットのまま通り、mask が NaN なら `+0.0`。
//! CPU 参照は `autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)` と同一。
//! view の起点は `CudaStorage::view` のスライスで渡すため offset は plan に
//! 含めない（カーネルは先頭からの相対添字だけを見る）。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{DeviceMatrixLayout, ShapeError};

/// 検証済みの起動パラメータ。`ops.rs` がカーネル引数（`int`）へ写す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReluMaskPlan {
    /// 論理形の行数（`RowMajor` なら格納 `r`、`Transposed` なら格納 `c`）。
    pub rows: usize,
    /// 論理形の列数。
    pub cols: usize,
    /// 論理要素数（`rows * cols`）。
    pub numel: usize,
    /// `g` を転置して読むか。
    pub g_transposed: bool,
}

impl ReluMaskPlan {
    /// 論理形（出力 shape）。
    pub fn logical_shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
}

/// 形状・layout を検証して [`ReluMaskPlan`] を返す（driver 呼び出しや
/// `CudaStorage::view`〈範囲外で panic する〉より前に呼ぶ。A03・REQ-8）。
///
/// - `g_shape` は格納形で rank 2 必須（`RankMismatch`）。
/// - `mask_shape` は論理形と完全一致（`ShapeMismatch`）。
/// - カーネル引数は `int` のため `rows`・`cols`・`numel` が `i32::MAX` 以下で
///   あることを要求する（超過・`rows * cols` の overflow は `InvalidArgument`）。
/// - `DeviceMatrixLayout` は `#[non_exhaustive]` のため未知 variant は
///   `Unsupported` で拒否する（fail-closed）。
pub fn plan_relu_mask(
    g_shape: &[usize],
    layout: DeviceMatrixLayout,
    mask_shape: &[usize],
) -> Result<ReluMaskPlan, BackendError> {
    if g_shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: g_shape.len(),
        }));
    }
    let (stored_r, stored_c) = (g_shape[0], g_shape[1]);
    let (g_transposed, rows, cols) = match layout {
        DeviceMatrixLayout::RowMajor => (false, stored_r, stored_c),
        DeviceMatrixLayout::Transposed => (true, stored_c, stored_r),
        _ => {
            return Err(BackendError::Unsupported(format!(
                "relu_backward_mask_device: unsupported layout {layout:?}"
            )));
        }
    };
    if mask_shape != [rows, cols] {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: mask_shape.to_vec(),
        }));
    }
    let numel = rows.checked_mul(cols).ok_or_else(|| {
        BackendError::InvalidArgument(
            "relu_backward_mask_device: rows * cols overflows usize".to_string(),
        )
    })?;
    let fits_i32 = |v: usize| i32::try_from(v).is_ok();
    if !(fits_i32(rows) && fits_i32(cols) && fits_i32(numel)) {
        return Err(BackendError::InvalidArgument(format!(
            "relu_backward_mask_device: rows={rows}, cols={cols}, numel={numel} must fit in i32 \
             (kernel argument type)"
        )));
    }
    Ok(ReluMaskPlan {
        rows,
        cols,
        numel,
        g_transposed,
    })
}

/// `ew_relu_mask_select_f32` の添字計算と選択式を逐語再現するホストモデル。
/// `g_view`／`mask_view` は各 view の先頭を 0 とするスライス（`numel` 要素以上）。
/// 出力は論理形の行優先。
///
/// `ReluMaskPlan` は全フィールド公開で外部から組み立てられるため、`plan` の
/// 整合（`numel == rows * cols`）と両スライスの長さ（`numel` 以上）を検証し、
/// 違反は `InvalidArgument` で返す（本番経路で panic しない）。カーネルと同じく
/// `gi >= numel` は読まず `+0.0` のまま残す。
pub fn relu_mask_select_host_model(
    plan: &ReluMaskPlan,
    g_view: &[f32],
    mask_view: &[f32],
) -> Result<Vec<f32>, BackendError> {
    let invalid =
        |msg: &str| BackendError::InvalidArgument(format!("relu_mask_select_host_model: {msg}"));
    let numel = plan
        .rows
        .checked_mul(plan.cols)
        .ok_or_else(|| invalid("rows * cols overflows usize"))?;
    if numel != plan.numel {
        return Err(invalid("plan.numel != rows * cols"));
    }
    if g_view.len() < numel {
        return Err(invalid("g_view is shorter than numel"));
    }
    if mask_view.len() < numel {
        return Err(invalid("mask_view is shorter than numel"));
    }
    let mut out = vec![0.0f32; numel];
    // numel > 0 なら rows・cols ともに 1 以上なので除算の分母は 0 にならない。
    for (idx, slot) in out.iter_mut().enumerate() {
        let gi = if plan.g_transposed {
            let r_l = idx / plan.cols;
            let c_l = idx - r_l * plan.cols;
            c_l * plan.rows + r_l
        } else {
            idx
        };
        if gi < numel {
            let m = mask_view[idx];
            // 乗算ではなく選択（NaN×0・−x×0 で bit がずれるのを避ける）。
            *slot = if m > 0.0 { g_view[gi] } else { 0.0 };
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU 参照の式（論理形の連続配列 2 本から計算）。
    fn reference(g_logical: &[f32], mask: &[f32]) -> Vec<f32> {
        g_logical
            .iter()
            .zip(mask)
            .map(|(&g, &m)| if m > 0.0 { g } else { 0.0 })
            .collect()
    }

    fn assert_bits_or_nan_class(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            if e.is_nan() {
                assert!(a.is_nan(), "idx {i}: expected NaN, got {a}");
            } else {
                assert_eq!(a.to_bits(), e.to_bits(), "idx {i}: {a} vs {e}");
            }
        }
    }

    /// 特殊値（NaN・±inf・±0.0・正負の非正規化数・通常値）を巡回させた決定的データ。
    fn special_cycle(n: usize, salt: usize) -> Vec<f32> {
        let table = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1e-40,
            -1e-40,
            1.5,
            -2.5,
            3.0e30,
        ];
        (0..n)
            .map(|i| table[(i * 7 + salt) % table.len()])
            .collect()
    }

    /// 格納 `[r, c]` を転置して論理 `[c, r]` の行優先へ並べ替える。
    fn transpose(stored: &[f32], r: usize, c: usize) -> Vec<f32> {
        let mut out = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = stored[i * c + j];
            }
        }
        out
    }

    #[test]
    fn row_major_matches_reference() {
        for &(r, c) in &[(1usize, 1usize), (1, 13), (13, 1), (3, 5), (17, 9)] {
            let g = special_cycle(r * c, 1);
            let m = special_cycle(r * c, 4);
            let plan = plan_relu_mask(&[r, c], DeviceMatrixLayout::RowMajor, &[r, c]).unwrap();
            let got = relu_mask_select_host_model(&plan, &g, &m).unwrap();
            assert_bits_or_nan_class(&got, &reference(&g, &m));
        }
    }

    #[test]
    fn transposed_matches_reference() {
        for &(r, c) in &[(1usize, 1usize), (1, 13), (13, 1), (3, 5), (17, 9)] {
            let g_stored = special_cycle(r * c, 2);
            let m = special_cycle(r * c, 5);
            let plan = plan_relu_mask(&[r, c], DeviceMatrixLayout::Transposed, &[c, r]).unwrap();
            assert_eq!(plan.logical_shape(), [c, r]);
            assert!(plan.g_transposed);
            let got = relu_mask_select_host_model(&plan, &g_stored, &m).unwrap();
            let g_logical = transpose(&g_stored, r, c);
            assert_bits_or_nan_class(&got, &reference(&g_logical, &m));
        }
    }

    #[test]
    fn denormal_positive_mask_selects_and_nan_mask_zeroes() {
        let plan = plan_relu_mask(&[1, 3], DeviceMatrixLayout::RowMajor, &[1, 3]).unwrap();
        let got =
            relu_mask_select_host_model(&plan, &[f32::NAN, -0.0, 7.0], &[1e-40, f32::NAN, -1e-40])
                .unwrap();
        assert!(got[0].is_nan());
        assert_eq!(got[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(got[2].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn rank_violation_is_rank_mismatch() {
        for shape in [&[4usize][..], &[2, 2, 2][..]] {
            let err = plan_relu_mask(shape, DeviceMatrixLayout::RowMajor, shape).unwrap_err();
            assert!(matches!(
                err,
                BackendError::ShapeMismatch(ShapeError::RankMismatch { expected: 2, .. })
            ));
        }
    }

    #[test]
    fn mask_shape_must_equal_logical_shape() {
        let err = plan_relu_mask(&[2, 3], DeviceMatrixLayout::RowMajor, &[3, 2]).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ShapeMismatch { .. })
        ));
        let err = plan_relu_mask(&[2, 3], DeviceMatrixLayout::Transposed, &[2, 3]).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn i32_limit_and_overflow_are_invalid_argument() {
        let big = i32::MAX as usize + 1;
        let err = plan_relu_mask(&[big, 1], DeviceMatrixLayout::RowMajor, &[big, 1]).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
        // rows と cols は i32 に収まるが積が超える。
        let half = 1usize << 20;
        let err =
            plan_relu_mask(&[half, half], DeviceMatrixLayout::RowMajor, &[half, half]).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
        // usize の積が overflow する。
        let huge = usize::MAX / 2 + 1;
        let err = plan_relu_mask(&[huge, 4], DeviceMatrixLayout::RowMajor, &[huge, 4]).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn empty_shapes_yield_empty_plan_and_output() {
        for shape in [[0usize, 5usize], [5, 0], [0, 0]] {
            let plan = plan_relu_mask(&shape, DeviceMatrixLayout::RowMajor, &shape).unwrap();
            assert_eq!(plan.numel, 0);
            assert!(
                relu_mask_select_host_model(&plan, &[], &[])
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn host_model_rejects_short_slices_and_inconsistent_plan() {
        let plan = plan_relu_mask(&[2, 2], DeviceMatrixLayout::RowMajor, &[2, 2]).unwrap();
        assert!(matches!(
            relu_mask_select_host_model(&plan, &[1.0; 3], &[1.0; 4]),
            Err(BackendError::InvalidArgument(_))
        ));
        assert!(matches!(
            relu_mask_select_host_model(&plan, &[1.0; 4], &[1.0; 3]),
            Err(BackendError::InvalidArgument(_))
        ));
        let bad = ReluMaskPlan { numel: 5, ..plan };
        assert!(matches!(
            relu_mask_select_host_model(&bad, &[1.0; 8], &[1.0; 8]),
            Err(BackendError::InvalidArgument(_))
        ));
    }
}
