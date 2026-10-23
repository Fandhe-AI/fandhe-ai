//! `BackendOps::relu_backward_mask_device`（イシュー #3218）の GPU 非依存な
//! 純関数群（形状・rank・layout・offset の検証と、カーネルの逐語ホストモデル）。
//! `crate::adam_model` と同じ設計判断で `objc2` 系 FFI に触れないため
//! `cfg(target_os = "macos")` を付けず、Linux（CI）でも単体テストが回る。
//!
//! # 責務境界
//! - 本モジュール: shape／rank／layout／view offset の u32 範囲の検証
//!   ([`plan_relu_mask`])、`shaders/elementwise.metal::ew_relu_mask_select_f32`
//!   の添字計算と選択式を逐語再現するホストモデル
//!   ([`relu_mask_select_host_model`])。
//! - `crate::ops`（macOS 限定）: デバイス一致検査（`Device::Metal` は Linux に
//!   存在しないため本モジュールでは扱えない）・確保・encode。
//!
//! # 意味論（設計 `docs/train-step-resident-parts-design.md` §3 行 3）
//! Metal は float 比較で非正規化数を 0 へ flush する（FTZ）ため、`> 0.0` は
//! マスクのビット列で評価する（[`relu_mask_is_positive_bits`]。#3218 reopen）。
//! `out[i] = (mask[i] > 0.0) ? g[i] : +0.0`。乗算ではなく**選択**で、`g` の
//! NaN／inf／−0.0 は mask が真ならビットのまま通り、mask が NaN なら `+0.0`。
//! CPU 参照は `autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)` の式と
//! 同一（選択は走査順に依存しないため stride view 経路とも bit 同一）。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{DeviceMatrixLayout, ShapeError};

/// 検証済みの起動パラメータ。`crate::ops::relu_backward_mask_device_impl` が
/// カーネル引数（`constant uint&`）へそのまま写す。
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
    /// `g` の連結バッファ内先頭要素位置（要素単位）。
    pub g_off: usize,
    /// `mask_src` の連結バッファ内先頭要素位置（要素単位）。
    pub mask_off: usize,
}

impl ReluMaskPlan {
    /// 論理形（出力 shape）。
    pub fn logical_shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
}

/// 形状・layout・offset を検証して [`ReluMaskPlan`] を返す（encode より前に
/// 呼ぶ。A03 入力検証・REQ-8）。
///
/// - `g_shape` は格納形で rank 2 必須（`RankMismatch`）。
/// - `mask_shape` は論理形と完全一致（`ShapeMismatch`）。
/// - カーネル引数は `uint` のため `offset + numel`・論理 `rows`・`cols` が
///   `u32::MAX` 以下であることを要求する（超過は `InvalidArgument`。fail-closed）。
/// - `DeviceMatrixLayout` は `#[non_exhaustive]` のため未知 variant は
///   `Unsupported` で拒否する。
pub fn plan_relu_mask(
    g_shape: &[usize],
    g_off: usize,
    layout: DeviceMatrixLayout,
    mask_shape: &[usize],
    mask_off: usize,
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
    let fits_u32 = |v: usize| u32::try_from(v).is_ok();
    let g_end = g_off.checked_add(numel);
    let mask_end = mask_off.checked_add(numel);
    let in_range = fits_u32(rows)
        && fits_u32(cols)
        && fits_u32(numel)
        && g_end.is_some_and(fits_u32)
        && mask_end.is_some_and(fits_u32);
    if !in_range {
        return Err(BackendError::InvalidArgument(format!(
            "relu_backward_mask_device: rows={rows}, cols={cols}, g_offset={g_off}, \
             mask_offset={mask_off} exceed the u32 kernel-argument range (offset + numel must \
             be <= u32::MAX)"
        )));
    }
    Ok(ReluMaskPlan {
        rows,
        cols,
        numel,
        g_transposed,
        g_off,
        mask_off,
    })
}

/// マスクのビット列が「`f32::from_bits(bits) > 0.0`（FTZ なし）」と同値かを返す。
/// `ew_relu_mask_select_f32` の判定式の逐語再現（#3218 reopen）。
/// 非ゼロ・符号ビット 0・`+inf` 以下（正の NaN を除く）なら真で、正の非正規化数も真。
pub fn relu_mask_is_positive_bits(bits: u32) -> bool {
    bits != 0 && bits <= 0x7F80_0000
}

/// `shaders/elementwise.metal::ew_relu_mask_select_f32` の添字計算と選択式を
/// 逐語再現するホストモデル。`g_storage`／`mask_storage` は連結バッファ全体
/// （`plan` の offset を起点に読む）。出力は論理形の行優先。
///
/// `ReluMaskPlan` は全フィールド公開で外部から組み立てられるため、`plan` の
/// 整合（`numel == rows * cols`、`cols == 0` で `numel > 0` でない）と、
/// 両入力スライスが `offset + numel` を覆う長さであることを検証し、違反は
/// `InvalidArgument` で返す（本番経路で panic しない）。
///
/// カーネルと同じく view 終端（`g_off + numel`／`mask_off + numel`）を超える
/// 添字は読まず `+0.0` のまま残す。
pub fn relu_mask_select_host_model(
    plan: &ReluMaskPlan,
    g_storage: &[f32],
    mask_storage: &[f32],
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
    let g_end = plan
        .g_off
        .checked_add(numel)
        .ok_or_else(|| invalid("g_off + numel overflows usize"))?;
    let mask_end = plan
        .mask_off
        .checked_add(numel)
        .ok_or_else(|| invalid("mask_off + numel overflows usize"))?;
    if g_storage.len() < g_end {
        return Err(invalid("g_storage is shorter than g_off + numel"));
    }
    if mask_storage.len() < mask_end {
        return Err(invalid("mask_storage is shorter than mask_off + numel"));
    }
    let mut out = vec![0.0f32; numel];
    // numel > 0 なら rows・cols ともに 1 以上なので除算の分母は 0 にならない。
    for (idx, slot) in out.iter_mut().enumerate() {
        let gi = if plan.g_transposed {
            let r_l = idx / plan.cols;
            let c_l = idx % plan.cols;
            plan.g_off + c_l * plan.rows + r_l
        } else {
            plan.g_off + idx
        };
        let mi = plan.mask_off + idx;
        if gi < g_end && mi < mask_end {
            // 乗算ではなく選択（NaN×0・−x×0 で bit がずれるのを避ける）。
            // 判定はカーネルと同じくビット列で行う（FTZ 非依存）。
            *slot = if relu_mask_is_positive_bits(mask_storage[mi].to_bits()) {
                g_storage[gi]
            } else {
                0.0
            };
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU 参照の式（`autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)` と
    /// 同一）。論理形の連続配列 2 本から計算する。
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

    /// 特殊値を巡回させた決定的データ。
    fn pattern(n: usize, salt: usize) -> Vec<f32> {
        const SPECIAL: [f32; 9] = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.5,
            -2.25,
            1e-30,
            3.0e38,
        ];
        (0..n)
            .map(|i| {
                if (i + salt).is_multiple_of(3) {
                    SPECIAL[(i * 7 + salt) % SPECIAL.len()]
                } else {
                    ((i * 31 + salt * 17) % 23) as f32 - 11.0
                }
            })
            .collect()
    }

    fn transpose(stored: &[f32], r: usize, c: usize) -> Vec<f32> {
        let mut out = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = stored[i * c + j];
            }
        }
        out
    }

    const SHAPES: [(usize, usize); 8] = [
        (1, 1),
        (1, 13),
        (13, 1),
        (7, 13),
        (63, 65),
        (64, 128),
        (64, 257),
        (2, 3),
    ];

    #[test]
    fn row_major_matches_reference_with_offsets() {
        for &(r, c) in &SHAPES {
            let n = r * c;
            for (g_off, m_off) in [(0usize, 0usize), (5, 3), (0, 9)] {
                let g_store = pattern(g_off + n + 4, 1);
                let m_store = pattern(m_off + n + 4, 2);
                let plan =
                    plan_relu_mask(&[r, c], g_off, DeviceMatrixLayout::RowMajor, &[r, c], m_off)
                        .unwrap();
                let got = relu_mask_select_host_model(&plan, &g_store, &m_store).unwrap();
                let want = reference(&g_store[g_off..g_off + n], &m_store[m_off..m_off + n]);
                assert_bits_or_nan_class(&got, &want);
            }
        }
    }

    #[test]
    fn transposed_matches_reference_with_offsets() {
        for &(r, c) in &SHAPES {
            let n = r * c;
            for (g_off, m_off) in [(0usize, 0usize), (4, 6)] {
                let g_store = pattern(g_off + n + 2, 3);
                let m_store = pattern(m_off + n + 2, 4);
                // 格納 [r, c] → 論理 [c, r]
                let plan = plan_relu_mask(
                    &[r, c],
                    g_off,
                    DeviceMatrixLayout::Transposed,
                    &[c, r],
                    m_off,
                )
                .unwrap();
                assert_eq!(plan.logical_shape(), [c, r]);
                let got = relu_mask_select_host_model(&plan, &g_store, &m_store).unwrap();
                let g_logical = transpose(&g_store[g_off..g_off + n], r, c);
                let want = reference(&g_logical, &m_store[m_off..m_off + n]);
                assert_bits_or_nan_class(&got, &want);
            }
        }
    }

    /// 選択であって乗算ではないこと: mask 偽位置の NaN／inf は `+0.0`
    /// （乗算なら NaN）、−0.0 は mask 真ならビット保存。
    #[test]
    fn selection_not_multiplication_semantics() {
        let g = [f32::NAN, f32::INFINITY, -0.0, -3.0, f32::NAN];
        let m = [0.0, -1.0, 1.0, f32::NAN, 2.0];
        let plan = plan_relu_mask(&[1, 5], 0, DeviceMatrixLayout::RowMajor, &[1, 5], 0).unwrap();
        let out = relu_mask_select_host_model(&plan, &g, &m).unwrap();
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[2].to_bits(), (-0.0f32).to_bits());
        assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
        assert!(out[4].is_nan());
    }

    #[test]
    fn rejects_non_rank2() {
        let e = plan_relu_mask(&[4], 0, DeviceMatrixLayout::RowMajor, &[4], 0).unwrap_err();
        assert!(matches!(
            e,
            BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn rejects_mask_shape_mismatch() {
        let e = plan_relu_mask(&[2, 3], 0, DeviceMatrixLayout::RowMajor, &[3, 2], 0).unwrap_err();
        assert!(matches!(
            e,
            BackendError::ShapeMismatch(ShapeError::ShapeMismatch { .. })
        ));
        // Transposed は論理形 [c, r] を要求する。
        let e = plan_relu_mask(&[2, 3], 0, DeviceMatrixLayout::Transposed, &[2, 3], 0).unwrap_err();
        assert!(matches!(
            e,
            BackendError::ShapeMismatch(ShapeError::ShapeMismatch { .. })
        ));
        assert!(plan_relu_mask(&[2, 3], 0, DeviceMatrixLayout::Transposed, &[3, 2], 0).is_ok());
    }

    #[test]
    fn rejects_offset_beyond_u32() {
        let big = u32::MAX as usize;
        let e =
            plan_relu_mask(&[2, 2], big - 3, DeviceMatrixLayout::RowMajor, &[2, 2], 0).unwrap_err();
        assert!(matches!(e, BackendError::InvalidArgument(_)));
        let e =
            plan_relu_mask(&[2, 2], 0, DeviceMatrixLayout::RowMajor, &[2, 2], big - 3).unwrap_err();
        assert!(matches!(e, BackendError::InvalidArgument(_)));
        // ちょうど u32::MAX に収まる境界は許可。
        assert!(plan_relu_mask(&[2, 2], big - 4, DeviceMatrixLayout::RowMajor, &[2, 2], 0).is_ok());
        // usize 加算 overflow も fail-closed。
        let e = plan_relu_mask(
            &[2, 2],
            usize::MAX,
            DeviceMatrixLayout::RowMajor,
            &[2, 2],
            0,
        )
        .unwrap_err();
        assert!(matches!(e, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn zero_numel_is_planned_as_empty() {
        let plan = plan_relu_mask(&[0, 5], 0, DeviceMatrixLayout::RowMajor, &[0, 5], 0).unwrap();
        assert_eq!(plan.numel, 0);
        assert!(
            relu_mask_select_host_model(&plan, &[], &[])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_short_slices_and_inconsistent_plan() {
        let plan = plan_relu_mask(&[1, 1], 0, DeviceMatrixLayout::RowMajor, &[1, 1], 0).unwrap();
        assert!(relu_mask_select_host_model(&plan, &[1.0], &[]).is_err());
        assert!(relu_mask_select_host_model(&plan, &[], &[1.0]).is_err());
        let mut bad = plan;
        bad.cols = 0;
        bad.g_transposed = true;
        assert!(relu_mask_select_host_model(&bad, &[1.0], &[1.0]).is_err());
        let mut bad = plan;
        bad.g_off = usize::MAX;
        assert!(relu_mask_select_host_model(&bad, &[1.0], &[1.0]).is_err());
    }

    #[test]
    fn bits_predicate_matches_float_compare_on_boundaries() {
        for b in [
            0u32,
            1,
            0x007F_FFFF,
            0x0080_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7F80_0001,
            0x7FC0_0000,
            0x7FFF_FFFF,
            0x8000_0000,
            0x8000_0001,
            0x8080_0000,
            0xFF80_0000,
            0xFFC0_0000,
            0xFFFF_FFFF,
        ] {
            assert_eq!(
                relu_mask_is_positive_bits(b),
                f32::from_bits(b) > 0.0,
                "bits {b:#010x}"
            );
        }
    }

    #[test]
    fn bits_predicate_matches_float_compare_on_sweep() {
        let mut b = 0u32;
        loop {
            assert_eq!(relu_mask_is_positive_bits(b), f32::from_bits(b) > 0.0);
            match b.checked_add(4099) {
                Some(n) => b = n,
                None => break,
            }
        }
    }

    #[test]
    fn denormal_mask_matches_reference_both_layouts() {
        let (r, c) = (7usize, 13usize);
        let n = r * c;
        let m: Vec<f32> = (0..n)
            .map(|i| match i % 6 {
                0 => 1e-40,
                1 => -1e-40,
                2 => 1e-45,
                3 => f32::from_bits(1),
                4 => f32::MIN_POSITIVE,
                _ => 1.5,
            })
            .collect();
        let g: Vec<f32> = (0..n).map(|i| (i % 23) as f32 - 11.0).collect();
        let plan = plan_relu_mask(&[r, c], 0, DeviceMatrixLayout::RowMajor, &[r, c], 0).unwrap();
        let got = relu_mask_select_host_model(&plan, &g, &m).unwrap();
        assert_bits_or_nan_class(&got, &reference(&g, &m));
        let plan = plan_relu_mask(&[r, c], 0, DeviceMatrixLayout::Transposed, &[c, r], 0).unwrap();
        let got = relu_mask_select_host_model(&plan, &g, &m).unwrap();
        assert_bits_or_nan_class(&got, &reference(&transpose(&g, r, c), &m));
    }
}
