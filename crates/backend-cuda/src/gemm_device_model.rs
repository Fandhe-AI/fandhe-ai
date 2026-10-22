//! `BackendOps::gemm_fp32_strict_into_device`／`gemm_resident_lhs_device`
//! （イシュー #3223・CUDA 実装）の GPU 非依存な純関数群（形状・rank・layout の
//! 検証、dW の TN／NT 判定、cp.async 整列のための staging 要否）。
//! `relu_mask_model` と同じ設計判断で、CUDA ドライバに触れないため Linux の CI
//! （GPU なし）でも単体テストが回る。
//!
//! 内部用・安定性保証なし（backend-cuda は facade の公開面に出ないサポート外の
//! 内部クレート。`docs/compat-api-scope.md` §0）。
//!
//! # 責務境界
//! - 本モジュール: `plan_dw_device`／`plan_dinput_device`（検証と寸法導出）と
//!   [`needs_staging`]。
//! - `crate::ops::CudaBackendOps::gemm_fp32_strict_into_device_impl`／
//!   `gemm_resident_lhs_device_impl`: device 一致・範囲検査・世代検査・起動。
//! - `crate::gemm::CudaGemm::launch_gemm_dw_device`／`launch_gemm_dinput_device`:
//!   転置・staging・GEMM・bias 縮約の起動。
//!
//! # layout の意味（`DeviceMatrixLayout` と同じ）
//! `Transposed` は格納 `[r, c]` を論理 `[c, r]` として読む。ホスト版の
//! `dense_transposed_view`（`Tensor::transpose_2d()` の zero-copy view）と同じ意味で、
//! 同じ論理形・同じカーネルへ渡すことでホスト版との bit 一致を構造で保つ。
//! 設計は `docs/train-step-resident-parts-design.md` §3 行 4・5。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{DeviceMatrixLayout, ShapeError};

/// dW の GEMM がどちらのオペランドを転置格納として受けるか。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwKind {
    /// `a` が転置格納・`b` が行優先（`xᵀ @ g` の標準形）。
    Tn,
    /// `a` が行優先・`b` が転置格納。
    Nt,
}

/// 検証済みの dW 寸法（論理 `a: [m, k]`・`b: [k, n]`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub kind: DwKind,
}

/// 検証済みの d_input 寸法（`w: [p, q]`・論理 `b: [q, r]`・出力 `[p, r]`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DinputPlan {
    pub p: usize,
    pub q: usize,
    pub r: usize,
    /// `b` が転置格納（格納 `[r, q]`）か。
    pub b_transposed: bool,
}

fn logical(
    op: &str,
    shape: &[usize],
    layout: DeviceMatrixLayout,
) -> Result<(usize, usize, bool), BackendError> {
    if shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    let (r, c) = (shape[0], shape[1]);
    match layout {
        DeviceMatrixLayout::RowMajor => Ok((r, c, false)),
        DeviceMatrixLayout::Transposed => Ok((c, r, true)),
        _ => Err(BackendError::Unsupported(format!(
            "{op}: unsupported layout {layout:?}"
        ))),
    }
}

/// dW の形状・layout を検証して [`DwPlan`] を返す（driver 呼び出しや範囲外で
/// panic する `CudaStorage::view` より前に呼ぶ。A03・REQ-8）。
///
/// - rank 2 必須（`RankMismatch`）、内側次元の一致（`ShapeMismatch`）。
/// - NN／TT（`a_layout == b_layout`）は `Unsupported`（ホスト版は別の経路で
///   bit 一致が保証できず、ホストへの黙示フォールバックも禁止 = A08）。
/// - `m`・`n`・`k` のいずれかが 0 は `Unsupported`（Metal 版と trait doc の一致を
///   優先。0 次元は呼び出し元がホスト版を使う）。
pub fn plan_dw_device(
    a_shape: &[usize],
    a_layout: DeviceMatrixLayout,
    b_shape: &[usize],
    b_layout: DeviceMatrixLayout,
) -> Result<DwPlan, BackendError> {
    let op = "gemm_fp32_strict_into_device";
    let (m, k, a_t) = logical(op, a_shape, a_layout)?;
    let (kb, n, b_t) = logical(op, b_shape, b_layout)?;
    if k != kb {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: vec![m, k],
            rhs: vec![kb, n],
        }));
    }
    if a_t == b_t {
        return Err(BackendError::Unsupported(format!(
            "{op}: NN/TT cannot be encoded without a wait (host path uses gemm_fp32_strict \
             with different accumulation order; bit parity not guaranteed)"
        )));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err(BackendError::Unsupported(format!(
            "{op}: zero-sized dimension is not handled by the device-input path"
        )));
    }
    Ok(DwPlan {
        m,
        n,
        k,
        kind: if a_t { DwKind::Tn } else { DwKind::Nt },
    })
}

/// d_input の形状・layout を検証して [`DinputPlan`] を返す。`w` は格納 `[p, q]`
/// の行優先、`b` は `b_layout` に従う。`p`・`q`・`r` の 0 は退化形状として plan
/// には載せる（呼び出し元が空／全 0 を返す）。
pub fn plan_dinput_device(
    w_shape: &[usize],
    b_shape: &[usize],
    b_layout: DeviceMatrixLayout,
) -> Result<DinputPlan, BackendError> {
    let op = "gemm_resident_lhs_device";
    let (p, q, _) = logical(op, w_shape, DeviceMatrixLayout::RowMajor)?;
    let (qb, r, b_t) = logical(op, b_shape, b_layout)?;
    if qb != q {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: vec![p, q],
            rhs: vec![qb, r],
        }));
    }
    Ok(DinputPlan {
        p,
        q,
        r,
        b_transposed: b_t,
    })
}

/// 直接 GEMM へ渡すオペランドの view offset（要素単位）が cp.async の 16 バイト
/// 境界（4 要素）を外れ、staging が要るか。`tiled_f32_kernel_kind` の整列ゲートは
/// A の offset しか見ず、既存の呼び出し元は B を常に offset 0 で渡してきたため、
/// 任意 offset の view を B に直接渡す新入口でだけ必要になる。基底確保が 256 バイト
/// 整列である前提のもと、offset が 4 の倍数なら 16 バイト整列になる。
pub fn needs_staging(offset_elems: usize) -> bool {
    !offset_elems.is_multiple_of(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use DeviceMatrixLayout::{RowMajor, Transposed};

    #[test]
    fn dw_tn_is_standard_form() {
        // x: 格納 [k=64, m=128]（Transposed）、g: [k=64, n=10]
        let p = plan_dw_device(&[64, 128], Transposed, &[64, 10], RowMajor).unwrap();
        assert_eq!(
            p,
            DwPlan {
                m: 128,
                n: 10,
                k: 64,
                kind: DwKind::Tn
            }
        );
    }

    #[test]
    fn dw_nt_uses_transposed_b() {
        // a: [m=3, k=4]、b: 格納 [n=5, k=4]（Transposed）
        let p = plan_dw_device(&[3, 4], RowMajor, &[5, 4], Transposed).unwrap();
        assert_eq!(
            p,
            DwPlan {
                m: 3,
                n: 5,
                k: 4,
                kind: DwKind::Nt
            }
        );
    }

    #[test]
    fn dw_rejects_rank_and_inner_mismatch() {
        assert!(matches!(
            plan_dw_device(&[2, 3, 4], Transposed, &[3, 4], RowMajor),
            Err(BackendError::ShapeMismatch(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            plan_dw_device(&[3, 4], RowMajor, &[5, 5], Transposed),
            Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn dw_rejects_nn_tt_and_zero_dims() {
        assert!(matches!(
            plan_dw_device(&[3, 4], RowMajor, &[4, 5], RowMajor),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            plan_dw_device(&[4, 3], Transposed, &[5, 4], Transposed),
            Err(BackendError::Unsupported(_))
        ));
        for (a, b) in [
            ([0usize, 4], [5usize, 4]),
            ([3, 4], [0, 4]),
            ([3, 0], [5, 0]),
        ] {
            assert!(matches!(
                plan_dw_device(&a, RowMajor, &b, Transposed),
                Err(BackendError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn dinput_transposed_and_row_major() {
        let t = plan_dinput_device(&[6, 4], &[5, 4], Transposed).unwrap();
        assert_eq!(
            t,
            DinputPlan {
                p: 6,
                q: 4,
                r: 5,
                b_transposed: true
            }
        );
        let r = plan_dinput_device(&[6, 4], &[4, 5], RowMajor).unwrap();
        assert_eq!(
            r,
            DinputPlan {
                p: 6,
                q: 4,
                r: 5,
                b_transposed: false
            }
        );
    }

    #[test]
    fn dinput_rejects_rank_and_mismatch_but_allows_degenerate() {
        assert!(matches!(
            plan_dinput_device(&[6], &[4, 5], RowMajor),
            Err(BackendError::ShapeMismatch(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            plan_dinput_device(&[6, 4], &[3, 5], RowMajor),
            Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch { .. }
            ))
        ));
        assert!(plan_dinput_device(&[0, 4], &[4, 5], RowMajor).is_ok());
        assert!(plan_dinput_device(&[6, 0], &[0, 5], RowMajor).is_ok());
    }

    #[test]
    fn staging_is_needed_only_off_16_byte_boundary() {
        assert!(!needs_staging(0));
        assert!(!needs_staging(4));
        assert!(!needs_staging(1024));
        assert!(needs_staging(1));
        assert!(needs_staging(3));
        assert!(needs_staging(6));
    }
}
