//! `lp_pool1d`／`lp_pool2d`／`lp_pool3d`（PyTorch `F.lp_pool1d`／`lp_pool2d`／`lp_pool3d`、
//! `nn.LPPool1d`／`LPPool2d`／`LPPool3d` 相当）の自由関数（イシュー #3825・親 #3768。設計記録
//! `docs/autodiff-pool3d-ops-decision.md` §19）。
//!
//! **新規 `Op`・`BackendOps`・カーネル・`unsafe`・依存はゼロ**。既存の `Var::pow_scalar`・
//! `Var::avg_pool{1,2,3}d`・`Var::sign`・`Var::abs`・`Var::relu`・`Var::mul` の合成のみで、CPU・CUDA・
//! Metal の全バックエンドに既存経路で到達する。呼び出し元は `Var::{lp_pool1d, lp_pool2d, lp_pool3d}`
//! （1 行委譲）と `nn::{LpPool1d, LpPool2d, LpPool3d}` 層。本モジュールは facade から再エクスポートしない。
//!
//! **式**（PyTorch `F.lp_pool*d` と同じ演算順）: `s = avg_pool(x^p)`、`z = sign(s) * relu(|s|)`、
//! `y = (z * K)^(1/p)`（`K = Π kernel_size`）。窓和は `avg_pool` の契約（`f64` 蓄積・`K` で割って 1 回だけ
//! `f32` へ戻す）に従い、`×K` と `pow(1/p)` は `f32` で行う（`mul_add` は使わず FMA 契約は不変）。
//! `padding = 0` のため divisor は常に `K` で、`count_include_pad` は結果に影響しない。
//!
//! **NaN・勾配**: 負の入力と非整数 `p` では `x^p` が NaN になり出力も NaN。全零窓の勾配は `Op::Relu` の
//! VJP が要素選択のため 0（NaN にならない）。`0 < p < 1` かつ窓内に 0 がある場合は `x^p` の勾配が
//! 非有限になりうる（PyTorch も同様）。
//!
//! **`create_graph` 非対応**: `Sign` が replay できないため、二階微分は型付きエラーになる（#3768 Q3 (a)）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: `norm_type` は有限かつ `> 0`、`ceil_mode=true` は
//! 拒否、rank・`kernel_size`・出力 shape を検査し、`K = Π kernel_size` は `checked_mul` で計算して
//! `2^24` 以下に限る（`f32` で厳密に表せる範囲。fail-closed）。保存ファイル経由の値も `Sequential::add_*`
//! を通るため同じ検査を受ける。検査は tape 操作より前に行い、エラー時に孤児ノードを残さない。

use fandhe_ai_tensor_core::pool3d::{self, Pool3dParams};
use fandhe_ai_tensor_core::{Pool2dParams, ShapeError, Tensor, pool2d_out_shape};

use crate::error::AutodiffError;
use crate::var::Var;

/// `K = Π kernel_size` の上限（`f32` で厳密に表せる 2^24）。
const MAX_WINDOW_ELEMS: usize = 1 << 24;

/// `norm_type` の検査（有限かつ `> 0`）。層の構築時と自由関数の先頭から呼ぶ。
pub(crate) fn validate_norm_type(op: &str, norm_type: f32) -> Result<(), AutodiffError> {
    if !norm_type.is_finite() || norm_type <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "lp_pool_ops::{op}: norm_type must be finite and > 0, got {norm_type}"
        )));
    }
    Ok(())
}

/// `ceil_mode=true` の拒否（v1 未対応）。
fn reject_ceil_mode(op: &str, ceil_mode: bool) -> Result<(), AutodiffError> {
    if ceil_mode {
        return Err(AutodiffError::InvalidArgument(format!(
            "lp_pool_ops::{op}: ceil_mode=true は v1 で未対応"
        )));
    }
    Ok(())
}

/// 窓要素数 `K` を `checked_mul` で求め、`1..=2^24` に限る。
pub(crate) fn window_elems(op: &str, kernel_size: &[usize]) -> Result<usize, AutodiffError> {
    let mut k: usize = 1;
    for &v in kernel_size {
        k = k.checked_mul(v).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "lp_pool_ops::{op}: kernel_size の積がオーバーフロー"
            ))
        })?;
    }
    if k == 0 || k > MAX_WINDOW_ELEMS {
        return Err(AutodiffError::InvalidArgument(format!(
            "lp_pool_ops::{op}: kernel_size の積は 1..=2^24 に限る, got {k}"
        )));
    }
    Ok(k)
}

fn check_rank(x: &Var<'_>, expected: usize) -> Result<Vec<usize>, AutodiffError> {
    let shape = x.shape();
    if shape.len() != expected {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected,
            actual: shape.len(),
        }));
    }
    Ok(shape)
}

/// 合成本体。`pool` は `x^p` を `padding=0`・`ceil_mode=false` で平均プーリングする closure。
fn compose<'t>(
    x: &Var<'t>,
    norm_type: f32,
    k: usize,
    pool: impl FnOnce(&Var<'t>) -> Result<Var<'t>, AutodiffError>,
) -> Result<Var<'t>, AutodiffError> {
    let s = pool(&x.pow_scalar(norm_type)?)?;
    let z = s.sign()?.mul(&s.abs()?.relu())?;
    let kc = x.tape().var_no_grad(&Tensor::scalar(k as f32));
    z.mul(&kc)?.pow_scalar(1.0 / norm_type)
}

/// 1D LPPool（`F.lp_pool1d`）。`x`: `[N, C, L]`。
pub fn lp_pool1d<'t>(
    x: &Var<'t>,
    norm_type: f32,
    kernel_size: usize,
    stride: Option<usize>,
    ceil_mode: bool,
) -> Result<Var<'t>, AutodiffError> {
    validate_norm_type("lp_pool1d", norm_type)?;
    reject_ceil_mode("lp_pool1d", ceil_mode)?;
    let shape = check_rank(x, 3)?;
    let params = Pool2dParams::new([1, kernel_size], stride.map(|s| [1, s]), [0, 0], [1, 1])
        .map_err(AutodiffError::Backend)?;
    pool2d_out_shape(&[shape[0], shape[1], 1, shape[2]], &params).map_err(AutodiffError::Shape)?;
    let k = window_elems("lp_pool1d", &[kernel_size])?;
    compose(x, norm_type, k, |p| {
        p.avg_pool1d(kernel_size, stride, 0, false, true)
    })
}

/// 2D LPPool（`F.lp_pool2d`）。`x`: `[N, C, H, W]`。
pub fn lp_pool2d<'t>(
    x: &Var<'t>,
    norm_type: f32,
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    ceil_mode: bool,
) -> Result<Var<'t>, AutodiffError> {
    validate_norm_type("lp_pool2d", norm_type)?;
    reject_ceil_mode("lp_pool2d", ceil_mode)?;
    let shape = check_rank(x, 4)?;
    let params =
        Pool2dParams::new(kernel_size, stride, [0, 0], [1, 1]).map_err(AutodiffError::Backend)?;
    pool2d_out_shape(&shape, &params).map_err(AutodiffError::Shape)?;
    let k = window_elems("lp_pool2d", &kernel_size)?;
    compose(x, norm_type, k, |p| {
        p.avg_pool2d(kernel_size, stride, [0, 0], false, true)
    })
}

/// 3D LPPool（`F.lp_pool3d`）。`x`: `[N, C, D, H, W]`。
pub fn lp_pool3d<'t>(
    x: &Var<'t>,
    norm_type: f32,
    kernel_size: [usize; 3],
    stride: Option<[usize; 3]>,
    ceil_mode: bool,
) -> Result<Var<'t>, AutodiffError> {
    validate_norm_type("lp_pool3d", norm_type)?;
    reject_ceil_mode("lp_pool3d", ceil_mode)?;
    let shape = check_rank(x, 5)?;
    let params = Pool3dParams::new(kernel_size, stride, [0, 0, 0], [1, 1, 1])
        .map_err(AutodiffError::Backend)?;
    pool3d::pool3d_layout(&shape, &params).map_err(AutodiffError::Shape)?;
    let k = window_elems("lp_pool3d", &kernel_size)?;
    compose(x, norm_type, k, |p| {
        p.avg_pool3d(kernel_size, stride, [0, 0, 0], false, true)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn p1_is_window_sum_and_p2_is_l2_norm_1d() {
        let tape = Tape::new();
        let x = tape.var(&t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]));
        let y1 = lp_pool1d(&x, 1.0, 2, None, false).unwrap();
        assert_eq!(y1.value().host_slice().to_vec(), vec![3.0, 7.0]);
        let y2 = lp_pool1d(&x, 2.0, 2, None, false).unwrap();
        let v = y2.value().host_slice().to_vec();
        assert!((v[0] - 5.0f32.sqrt()).abs() < 1e-6);
        assert!((v[1] - 25.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn p2_2d_and_3d_closed_form() {
        let tape = Tape::new();
        let x2 = tape.var(&t(&[1.0, 2.0, 2.0, 4.0], &[1, 1, 2, 2]));
        let y = lp_pool2d(&x2, 2.0, [2, 2], None, false).unwrap();
        assert!((y.value().host_slice()[0] - 5.0).abs() < 1e-6);
        let x3 = tape.var(&t(&[1.0; 8], &[1, 1, 2, 2, 2]));
        let y = lp_pool3d(&x3, 1.0, [2, 2, 2], None, false).unwrap();
        assert!((y.value().host_slice()[0] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn zero_window_gradient_is_exactly_zero() {
        for p in [2.0f32, 1.5] {
            let tape = Tape::new();
            let x = tape.var(&t(&[0.0, 0.0, 1.0, 2.0], &[1, 1, 4]));
            let y = lp_pool1d(&x, p, 2, None, false).unwrap();
            let loss = y.sum(None).unwrap();
            let g = tape.backward(&loss).unwrap();
            let gv = g.get(&x).unwrap().unwrap().host_slice().to_vec();
            assert_eq!(gv[0], 0.0);
            assert_eq!(gv[1], 0.0);
            assert!(gv[2].is_finite() && gv[3].is_finite());
        }
    }

    #[test]
    fn negative_input_with_fractional_p_is_nan() {
        let tape = Tape::new();
        let x = tape.var(&t(&[-1.0, 2.0], &[1, 1, 2]));
        let y = lp_pool1d(&x, 1.5, 2, None, false).unwrap();
        assert!(y.value().host_slice()[0].is_nan());
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let tape = Tape::new();
        let x = tape.var(&t(&[1.0; 16], &[1, 1, 4, 4]));
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                lp_pool2d(&x, bad, [2, 2], None, false),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
        assert!(lp_pool2d(&x, 2.0, [2, 2], None, true).is_err());
        assert!(lp_pool2d(&x, 2.0, [0, 2], None, false).is_err());
        assert!(lp_pool2d(&x, 2.0, [5, 5], None, false).is_err());
        assert!(lp_pool1d(&x, 2.0, 2, None, false).is_err());
        assert!(lp_pool3d(&x, 2.0, [2, 2, 2], None, false).is_err());
        assert!(matches!(
            window_elems("t", &[usize::MAX, 2]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(window_elems("t", &[1 << 13, 1 << 12]).is_err());
        assert!(window_elems("t", &[1 << 12, 1 << 12]).is_ok());
    }
}
