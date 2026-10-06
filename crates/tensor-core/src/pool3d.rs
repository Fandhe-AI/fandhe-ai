//! 3D プーリング（`max_pool3d`／`avg_pool3d`）のホスト参照カーネルの**単一情報源**
//! （イシュー #2643・親 #2625「Phase 4」・ルート #2499。実装記録は
//! `docs/autodiff-pool3d-ops-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::pool3d_ops`（`BackendOps` が `Unsupported` のときのホストフォールバック）・
//!   `autodiff::grad`（`Op::AvgPool3d` の VJP）・`backend-cpu` の
//!   `CpuBackendOps::pool3d_max`／`pool3d_avg` がいずれも本モジュールの関数を直接呼ぶ。
//!   窓の走査順・タイ／NaN 規則・アキュムレータ契約をクレート間で複製せず、乖離を構造的に
//!   排除する（`cumulative.rs` と同じ「共有可能な層に一本化する」方式）。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の
//!   `Unsupported` から autodiff 側がこのホスト実装へフォールバックする。
//! - 2D 版（[`crate::Pool2dParams`]・`docs/pooling-ops-design.md`）の決定を 3 軸へそのまま
//!   一般化する。レイアウトは NCDHW 固定・rank 5 のみ。
//!
//! # 数値契約
//!
//! - **Max**: 窓内を `kd` 外側 → `kh` → `kw` 内側の row-major で走査し、更新条件は
//!   `v > best || (v.is_nan() && !best.is_nan())`（**タイ先勝ち**・NaN 伝播・最初の NaN の
//!   索引が残る）。padding 位置は走査対象外。値は入力要素そのもの（bit 一致）。索引は
//!   `(n, c)` 平面内の flat 添字 `d·H·W + h·W + w`（`i32`）。
//! - **Avg**: 窓内（padding を除く）を row-major で `f64` へ昇格して逐次加算し、`f64` で
//!   除算してから 1 回だけ `f32` へ downcast する（正規化統計・長軸縮約の `f64`
//!   アキュムレータ契約。`.claude/rules/coding-rust.md`）。divisor は
//!   `count_include_pad = true` なら `kD·kH·kW`、`false` なら有効要素数。
//!   [`avg_pool3d_vjp_host`] は出力 major 走査で `f64` 配列へ加算し 1 回 downcast する。
//! - 非有限入力は事前に拒否せず伝播する。`mul_add` は使わない（matmul 系 FMA 契約には
//!   触れない）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状・パラメータの検査は [`Pool3dParams::new`] と [`pool3d_layout`] に集約し、確保より前に
//! rank・空間軸 0・負分子・空窓・要素数／バイト数の `checked_*` を検査して型付きエラーで
//! 拒否する。カーネルは入力スライス長を再検査し、窓座標を `checked_*` で逆算して
//! `< in_len` を手動検査する。索引は `i32::try_from`（無検査 `as i32` を使わない）で変換する。
//! `unsafe`／`get_unchecked` は使わない。

use crate::device::BackendError;
use crate::error::ShapeError;
use crate::ops_shape::pool_out_len;
use crate::tensor::checked_numel_for;

/// 3D プーリングのパラメータ記述子（[`crate::Pool2dParams`] の 3 軸版）。
///
/// `ceil_mode`／`count_include_pad` は本構造体に含めない（`ceil_mode` は autodiff 入口で
/// `true` を拒否する契約フラグ、`count_include_pad` は呼び出しごとの引数）。
/// `#[non_exhaustive]` で将来のフィールド追加を非破壊にし、フィールドは
/// [`Pool3dParams::new`] 経由でのみ設定する。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Pool3dParams {
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
}

impl Pool3dParams {
    /// `kernel_size`／`stride`／`dilation` の 0 を [`BackendError::InvalidArgument`] で拒否する。
    /// `stride = None` は `kernel_size` と同値（PyTorch の既定意味論）。
    ///
    /// 各軸 **`padding <= kernel_size / 2`**（`dilation` 非依存の整数除算・floor。
    /// `docs/pooling-ops-design.md` §3）を要求し、`2 * padding` の `usize` オーバーフローも
    /// `checked_mul` で拒否する。
    pub fn new(
        kernel_size: [usize; 3],
        stride: Option<[usize; 3]>,
        padding: [usize; 3],
        dilation: [usize; 3],
    ) -> Result<Self, BackendError> {
        if kernel_size.contains(&0) {
            return Err(BackendError::InvalidArgument(
                "Pool3dParams::new: kernel_size の各軸は 1 以上である必要がある".into(),
            ));
        }
        let stride = stride.unwrap_or(kernel_size);
        if stride.contains(&0) {
            return Err(BackendError::InvalidArgument(
                "Pool3dParams::new: stride の各軸は 1 以上である必要がある".into(),
            ));
        }
        if dilation.contains(&0) {
            return Err(BackendError::InvalidArgument(
                "Pool3dParams::new: dilation の各軸は 1 以上である必要がある".into(),
            ));
        }
        for i in 0..3 {
            let two_p = padding[i].checked_mul(2).ok_or_else(|| {
                BackendError::InvalidArgument(
                    "Pool3dParams::new: 2 * padding が usize の範囲でオーバーフローする".into(),
                )
            })?;
            if two_p > kernel_size[i] {
                return Err(BackendError::InvalidArgument(format!(
                    "Pool3dParams::new: padding[{i}]（{}）は kernel_size[{i}]（{}）の半分以下である必要がある",
                    padding[i], kernel_size[i]
                )));
            }
        }
        Ok(Self {
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// カーネル空間サイズ `[kD, kH, kW]`。
    pub fn kernel_size(&self) -> [usize; 3] {
        self.kernel_size
    }

    /// ストライド `[sD, sH, sW]`。
    pub fn stride(&self) -> [usize; 3] {
        self.stride
    }

    /// パディング `[pD, pH, pW]`（各軸の前後同一幅）。
    pub fn padding(&self) -> [usize; 3] {
        self.padding
    }

    /// dilation `[dD, dH, dW]`。
    pub fn dilation(&self) -> [usize; 3] {
        self.dilation
    }
}

/// 検査済みの 3D プーリングレイアウト（[`pool3d_layout`] の戻り値）。
///
/// 入力 `[N, C, D, H, W]` と出力 `[N, C, Dout, Hout, Wout]` の shape と、検査に使った
/// [`Pool3dParams`] を保持する。フィールドは非公開で [`pool3d_layout`] だけが生成し、
/// 公開カーネルが外部から改変された値で範囲外アクセスしないようにする（REQ-8）。
#[derive(Debug, Clone, PartialEq)]
pub struct Pool3dLayout {
    in_shape: [usize; 5],
    out_shape: [usize; 5],
    params: Pool3dParams,
}

impl Pool3dLayout {
    /// 入力 shape `[N, C, D, H, W]`。
    pub fn in_shape(&self) -> [usize; 5] {
        self.in_shape
    }

    /// 出力 shape `[N, C, Dout, Hout, Wout]`。
    pub fn out_shape(&self) -> [usize; 5] {
        self.out_shape
    }

    /// 検査に使ったパラメータ。
    pub fn params(&self) -> &Pool3dParams {
        &self.params
    }

    /// 入力の総要素数（[`pool3d_layout`] が `checked_mul` 検査済みのためオーバーフローしない）。
    pub fn in_numel(&self) -> usize {
        self.in_shape.iter().product()
    }

    /// 出力の総要素数（同上）。
    pub fn out_numel(&self) -> usize {
        self.out_shape.iter().product()
    }

    fn in_plane(&self) -> usize {
        self.in_shape[2] * self.in_shape[3] * self.in_shape[4]
    }

    fn out_plane(&self) -> usize {
        self.out_shape[2] * self.out_shape[3] * self.out_shape[4]
    }

    /// Max の索引（`(n, c)` 平面内 flat 添字の最大値 `D·H·W - 1`）が `i32` で表現できる
    /// ことを検査する。`N`／`C`／出力が空かどうかに依存させず、実体化より前に autodiff
    /// 入口が呼ぶ（`Var::max_pool2d` の `H·W <= i32::MAX` 検査と同じ位置づけ）。
    pub fn check_i32_indices(&self) -> Result<(), ShapeError> {
        // N／C が 0 の空バッチでは `pool3d_layout` の要素数検査を `D·H·W` が超えていても通る
        // ため、ここで `checked_mul` する（panic／wrap ではなく型付きエラーにする。REQ-8）。
        let plane = self.in_shape[2]
            .checked_mul(self.in_shape[3])
            .and_then(|p| p.checked_mul(self.in_shape[4]))
            .ok_or(ShapeError::ElementCountOverflow)?;
        if plane > i32::MAX as usize {
            return Err(ShapeError::IndexRangeOverflow { index: plane });
        }
        Ok(())
    }
}

/// 入力 shape とパラメータを検査して [`Pool3dLayout`] を返す。
///
/// 検査順（`pool2d_out_shape` の 3 軸版）: rank（5）→ 空間軸 `D`／`H`／`W == 0` 拒否
/// （`N`／`C == 0` は受理し空出力）→ 軸ごとに `pool_out_len`（負分子拒否ゲート）→ 軸ごとに
/// 空窓拒否（`kernel == 2` かつ `dilation > in_len`。`padding <= kernel/2` 契約の下で
/// 空窓が起こりうる唯一の構成）→ 入出力の要素数積オーバーフロー。
pub fn pool3d_layout(
    in_shape: &[usize],
    params: &Pool3dParams,
) -> Result<Pool3dLayout, ShapeError> {
    let in_shape: [usize; 5] = in_shape.try_into().map_err(|_| ShapeError::RankMismatch {
        expected: 5,
        actual: in_shape.len(),
    })?;
    if in_shape[2] == 0 || in_shape[3] == 0 || in_shape[4] == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: in_shape[2..].to_vec(),
            rhs: vec![1, 1, 1],
        });
    }
    let mut out_shape = [in_shape[0], in_shape[1], 0, 0, 0];
    for a in 0..3 {
        let in_len = in_shape[2 + a];
        out_shape[2 + a] = pool_out_len(
            in_len,
            params.kernel_size[a],
            params.stride[a],
            params.padding[a],
            params.dilation[a],
        )?;
    }
    for a in 0..3 {
        let in_len = in_shape[2 + a];
        if params.kernel_size[a] == 2 && params.dilation[a] > in_len {
            return Err(ShapeError::ShapeMismatch {
                lhs: vec![in_len],
                rhs: vec![params.dilation[a]],
            });
        }
    }
    checked_numel_for::<f32>(&in_shape)?;
    checked_numel_for::<f32>(&out_shape)?;
    Ok(Pool3dLayout {
        in_shape,
        out_shape,
        params: params.clone(),
    })
}

/// 出力位置 `out_idx`・カーネル位置 `k` に対応する入力座標を逆算する
/// （`out_idx*stride + k*dilation - padding`。`checked_*` で逆算し、負・オーバーフローは
/// `None`）。呼び出し側が `< in_len` を手動検査する（REQ-8）。
#[inline]
fn window_pos(
    out_idx: usize,
    stride: usize,
    k: usize,
    dilation: usize,
    padding: usize,
) -> Option<usize> {
    out_idx
        .checked_mul(stride)?
        .checked_add(k.checked_mul(dilation)?)?
        .checked_sub(padding)
}

fn check_input_len(layout: &Pool3dLayout, actual: usize) -> Result<(), ShapeError> {
    let expected = layout.in_numel();
    if actual == expected {
        Ok(())
    } else {
        Err(ShapeError::ElementCountMismatch { expected, actual })
    }
}

/// 出力位置 `(od, oh, ow)` の窓内有効入力座標を row-major（`kd` → `kh` → `kw`）で列挙する。
fn for_each_window<F>(layout: &Pool3dLayout, o: [usize; 3], mut f: F) -> Result<(), ShapeError>
where
    F: FnMut(usize, usize, usize) -> Result<(), ShapeError>,
{
    let p = &layout.params;
    let [d_in, h_in, w_in] = [layout.in_shape[2], layout.in_shape[3], layout.in_shape[4]];
    for kd in 0..p.kernel_size[0] {
        let Some(d) =
            window_pos(o[0], p.stride[0], kd, p.dilation[0], p.padding[0]).filter(|&d| d < d_in)
        else {
            continue;
        };
        for kh in 0..p.kernel_size[1] {
            let Some(h) = window_pos(o[1], p.stride[1], kh, p.dilation[1], p.padding[1])
                .filter(|&h| h < h_in)
            else {
                continue;
            };
            for kw in 0..p.kernel_size[2] {
                let Some(w) = window_pos(o[2], p.stride[2], kw, p.dilation[2], p.padding[2])
                    .filter(|&w| w < w_in)
                else {
                    continue;
                };
                f(d, h, w)?;
            }
        }
    }
    Ok(())
}

/// 3D max pooling（値と索引）。`torch.nn.functional.max_pool3d(..., return_indices=True)` 相当。
///
/// 戻り値は `([N,C,Dout,Hout,Wout] の値, 同 shape の索引)`。索引は `(n, c)` 平面内 flat 添字
/// `d·H·W + h·W + w`。タイ先勝ち・NaN 伝播（最初の NaN の索引が残る。PyTorch は最後の NaN で、
/// この差は `docs/autodiff-pool3d-ops-decision.md` §5 に記録している）。
pub fn max_pool3d_host(
    input: &[f32],
    layout: &Pool3dLayout,
) -> Result<(Vec<f32>, Vec<i32>), ShapeError> {
    check_input_len(layout, input.len())?;
    layout.check_i32_indices()?;
    let out_numel = layout.out_numel();
    if out_numel == 0 || input.is_empty() {
        return Ok((vec![0.0; out_numel], vec![0; out_numel]));
    }
    let [n, c] = [layout.in_shape[0], layout.in_shape[1]];
    let [h_in, w_in] = [layout.in_shape[3], layout.in_shape[4]];
    let [d_out, h_out, w_out] = [
        layout.out_shape[2],
        layout.out_shape[3],
        layout.out_shape[4],
    ];
    let (in_plane, out_plane) = (layout.in_plane(), layout.out_plane());
    let mut values = Vec::with_capacity(out_numel);
    let mut indices = Vec::with_capacity(out_numel);
    for nc in 0..n * c {
        let plane = &input[nc * in_plane..(nc + 1) * in_plane];
        debug_assert_eq!(values.len(), nc * out_plane);
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best: Option<(f32, usize)> = None;
                    for_each_window(layout, [od, oh, ow], |d, h, w| {
                        let flat = (d * h_in + h) * w_in + w;
                        let v = *plane
                            .get(flat)
                            .ok_or(ShapeError::IndexRangeOverflow { index: flat })?;
                        let take = match best {
                            None => true,
                            Some((b, _)) => v > b || (v.is_nan() && !b.is_nan()),
                        };
                        if take {
                            best = Some((v, flat));
                        }
                        Ok(())
                    })?;
                    // 空窓は `pool3d_layout` が拒否済み。到達した場合は fail-closed で拒否する。
                    let (v, flat) = best.ok_or(ShapeError::ShapeMismatch {
                        lhs: vec![od, oh, ow],
                        rhs: vec![0],
                    })?;
                    values.push(v);
                    indices.push(
                        i32::try_from(flat)
                            .map_err(|_| ShapeError::IndexRangeOverflow { index: flat })?,
                    );
                }
            }
        }
    }
    Ok((values, indices))
}

/// 窓内の有効要素数と divisor を求める（`count_include_pad` 規約）。
fn avg_divisor(
    layout: &Pool3dLayout,
    o: [usize; 3],
    count_include_pad: bool,
) -> Result<(f64, usize), ShapeError> {
    let mut count = 0usize;
    for_each_window(layout, o, |_, _, _| {
        count += 1;
        Ok(())
    })?;
    let [kd, kh, kw] = layout.params.kernel_size;
    let divisor = if count_include_pad {
        kd.checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .ok_or(ShapeError::ElementCountOverflow)?
    } else {
        count
    };
    if divisor == 0 {
        return Err(ShapeError::ShapeMismatch {
            lhs: o.to_vec(),
            rhs: vec![0],
        });
    }
    Ok((divisor as f64, count))
}

/// 3D average pooling。`torch.nn.functional.avg_pool3d`（`ceil_mode=False`・
/// `divisor_override=None`）相当。`dilation` は常に `[1, 1, 1]` の呼び出しを想定する
/// （呼び出し側が [`Pool3dParams`] を `dilation = [1,1,1]` で構築する。ここでは窓走査が
/// `dilation` を尊重するだけで、PyTorch に無い dilation 付き平均は呼び出し側が拒否する）。
pub fn avg_pool3d_host(
    input: &[f32],
    layout: &Pool3dLayout,
    count_include_pad: bool,
) -> Result<Vec<f32>, ShapeError> {
    check_input_len(layout, input.len())?;
    let out_numel = layout.out_numel();
    if out_numel == 0 || input.is_empty() {
        return Ok(vec![0.0; out_numel]);
    }
    let [n, c] = [layout.in_shape[0], layout.in_shape[1]];
    let [h_in, w_in] = [layout.in_shape[3], layout.in_shape[4]];
    let [d_out, h_out, w_out] = [
        layout.out_shape[2],
        layout.out_shape[3],
        layout.out_shape[4],
    ];
    let in_plane = layout.in_plane();
    let mut out = Vec::with_capacity(out_numel);
    for nc in 0..n * c {
        let plane = &input[nc * in_plane..(nc + 1) * in_plane];
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let (divisor, _) = avg_divisor(layout, [od, oh, ow], count_include_pad)?;
                    let mut acc = 0.0_f64;
                    for_each_window(layout, [od, oh, ow], |d, h, w| {
                        let flat = (d * h_in + h) * w_in + w;
                        let v = *plane
                            .get(flat)
                            .ok_or(ShapeError::IndexRangeOverflow { index: flat })?;
                        acc += f64::from(v);
                        Ok(())
                    })?;
                    out.push((acc / divisor) as f32);
                }
            }
        }
    }
    Ok(out)
}

/// [`avg_pool3d_host`] の VJP（入力勾配）。
///
/// 出力を row-major（`N`・`C`・`od`・`oh`・`ow`）で走査し、各出力位置の窓に属する入力位置へ
/// `upstream / divisor` を `f64` アキュムレータ配列（入力要素数）へ加算し、最後に 1 回だけ
/// `f32` へ downcast する。特定の入力位置への加算列は常に出力 row-major 順になる。
/// `upstream` は出力 shape（`layout.out_shape()`）の連続配置スライス。
pub fn avg_pool3d_vjp_host(
    upstream: &[f32],
    layout: &Pool3dLayout,
    count_include_pad: bool,
) -> Result<Vec<f32>, ShapeError> {
    let out_numel = layout.out_numel();
    if upstream.len() != out_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: out_numel,
            actual: upstream.len(),
        });
    }
    let in_numel = layout.in_numel();
    let acc_bytes = in_numel
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if acc_bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    if in_numel == 0 || out_numel == 0 {
        return Ok(vec![0.0; in_numel]);
    }
    let [n, c] = [layout.in_shape[0], layout.in_shape[1]];
    let [h_in, w_in] = [layout.in_shape[3], layout.in_shape[4]];
    let [d_out, h_out, w_out] = [
        layout.out_shape[2],
        layout.out_shape[3],
        layout.out_shape[4],
    ];
    let (in_plane, out_plane) = (layout.in_plane(), layout.out_plane());
    let mut acc = vec![0.0_f64; in_numel];
    for nc in 0..n * c {
        let acc_plane = &mut acc[nc * in_plane..(nc + 1) * in_plane];
        let up_plane = &upstream[nc * out_plane..(nc + 1) * out_plane];
        for od in 0..d_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let (divisor, _) = avg_divisor(layout, [od, oh, ow], count_include_pad)?;
                    let g = up_plane[(od * h_out + oh) * w_out + ow];
                    let contrib = f64::from(g) / divisor;
                    for_each_window(layout, [od, oh, ow], |d, h, w| {
                        let flat = (d * h_in + h) * w_in + w;
                        let slot = acc_plane
                            .get_mut(flat)
                            .ok_or(ShapeError::IndexRangeOverflow { index: flat })?;
                        *slot += contrib;
                        Ok(())
                    })?;
                }
            }
        }
    }
    Ok(acc.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(k: [usize; 3], s: Option<[usize; 3]>, p: [usize; 3], d: [usize; 3]) -> Pool3dParams {
        Pool3dParams::new(k, s, p, d).unwrap()
    }

    fn seq(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn max_basic_values_and_indices() {
        // [1,1,2,2,2] を kernel 2・stride 既定（=kernel）→ 出力 1 要素（最大 7・索引 7）。
        let p = params([2, 2, 2], None, [0; 3], [1; 3]);
        let layout = pool3d_layout(&[1, 1, 2, 2, 2], &p).unwrap();
        let (v, i) = max_pool3d_host(&seq(8), &layout).unwrap();
        assert_eq!((v, i), (vec![7.0], vec![7]));
    }

    #[test]
    fn max_overlapping_windows_and_anisotropic() {
        // [1,1,3,2,2]・kernel [2,1,2]・stride [1,1,1]
        let p = params([2, 1, 2], Some([1, 1, 1]), [0; 3], [1; 3]);
        let layout = pool3d_layout(&[1, 1, 3, 2, 2], &p).unwrap();
        assert_eq!(layout.out_shape(), [1, 1, 2, 2, 1]);
        let (v, i) = max_pool3d_host(&seq(12), &layout).unwrap();
        // 出力 (od,oh): 窓 d∈{od,od+1}, w∈{0,1} の最大 = d=od+1, w=1。
        assert_eq!(v, vec![5.0, 7.0, 9.0, 11.0]);
        assert_eq!(i, vec![5, 7, 9, 11]);
    }

    #[test]
    fn max_tie_first_wins_and_nan_propagates_first_index() {
        let p = params([2, 2, 2], None, [0; 3], [1; 3]);
        let layout = pool3d_layout(&[1, 1, 2, 2, 2], &p).unwrap();
        let (v, i) = max_pool3d_host(&[3.0; 8], &layout).unwrap();
        assert_eq!((v, i), (vec![3.0], vec![0]));
        let mut x = vec![1.0_f32; 8];
        x[3] = f32::NAN;
        x[5] = f32::NAN;
        x[6] = f32::INFINITY;
        let (v, i) = max_pool3d_host(&x, &layout).unwrap();
        assert!(v[0].is_nan());
        assert_eq!(i, vec![3]);
        let x = vec![f32::NEG_INFINITY; 8];
        let (v, i) = max_pool3d_host(&x, &layout).unwrap();
        assert_eq!((v, i), (vec![f32::NEG_INFINITY], vec![0]));
    }

    #[test]
    fn max_padding_positions_are_skipped_and_dilation() {
        // 1 軸相当: [1,1,1,1,4]・kernel [1,1,2]・padding [0,0,1]・stride [1,1,1]。
        let p = params([1, 1, 2], Some([1, 1, 1]), [0, 0, 1], [1; 3]);
        let layout = pool3d_layout(&[1, 1, 1, 1, 4], &p).unwrap();
        assert_eq!(layout.out_shape(), [1, 1, 1, 1, 5]);
        let x = [-5.0_f32, -4.0, -3.0, -2.0];
        let (v, i) = max_pool3d_host(&x, &layout).unwrap();
        // 負値でも padding(0) が勝たない。
        assert_eq!(v, vec![-5.0, -4.0, -3.0, -2.0, -2.0]);
        assert_eq!(i, vec![0, 1, 2, 3, 3]);
        // dilation 2: 窓 {0,2},{1,3}。
        let p = params([1, 1, 2], Some([1, 1, 1]), [0; 3], [1, 1, 2]);
        let layout = pool3d_layout(&[1, 1, 1, 1, 4], &p).unwrap();
        let (v, i) = max_pool3d_host(&[1.0, 9.0, 3.0, 2.0], &layout).unwrap();
        assert_eq!((v, i), (vec![3.0, 9.0], vec![2, 1]));
    }

    #[test]
    fn avg_count_include_pad_both_modes() {
        let p = params([1, 1, 2], Some([1, 1, 1]), [0, 0, 1], [1; 3]);
        let layout = pool3d_layout(&[1, 1, 1, 1, 2], &p).unwrap();
        let x = [2.0_f32, 4.0];
        // 窓: {pad,2},{2,4},{4,pad}
        assert_eq!(
            avg_pool3d_host(&x, &layout, true).unwrap(),
            vec![1.0, 3.0, 2.0]
        );
        assert_eq!(
            avg_pool3d_host(&x, &layout, false).unwrap(),
            vec![2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn avg_vjp_matches_independent_oracle() {
        let p = params([2, 2, 2], Some([1, 2, 1]), [1, 0, 1], [1; 3]);
        let shape = [2, 2, 3, 4, 3];
        let layout = pool3d_layout(&shape, &p).unwrap();
        let n_in = layout.in_numel();
        for cip in [true, false] {
            let g: Vec<f32> = (0..layout.out_numel())
                .map(|i| (i % 7) as f32 - 3.0)
                .collect();
            let got = avg_pool3d_vjp_host(&g, &layout, cip).unwrap();
            // 独立オラクル: 線形写像の転置を基底ベクトルで構成する（forward を e_j に適用）。
            let mut want = vec![0.0_f64; n_in];
            for j in 0..n_in {
                let mut e = vec![0.0_f32; n_in];
                e[j] = 1.0;
                let y = avg_pool3d_host(&e, &layout, cip).unwrap();
                want[j] = y
                    .iter()
                    .zip(&g)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
            }
            for (a, b) in got.iter().zip(&want) {
                assert!((f64::from(*a) - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn empty_batch_and_channel_give_empty_output() {
        let p = params([2, 2, 2], None, [0; 3], [1; 3]);
        for shape in [[0, 3, 2, 2, 2], [2, 0, 2, 2, 2]] {
            let layout = pool3d_layout(&shape, &p).unwrap();
            assert_eq!(layout.out_numel(), 0);
            let (v, i) = max_pool3d_host(&[], &layout).unwrap();
            assert!(v.is_empty() && i.is_empty());
            assert!(avg_pool3d_host(&[], &layout, true).unwrap().is_empty());
            assert!(avg_pool3d_vjp_host(&[], &layout, true).unwrap().is_empty());
        }
    }

    #[test]
    fn layout_rejections() {
        let p = params([2, 2, 2], None, [0; 3], [1; 3]);
        assert!(matches!(
            pool3d_layout(&[1, 1, 2, 2], &p),
            Err(ShapeError::RankMismatch {
                expected: 5,
                actual: 4
            })
        ));
        assert!(pool3d_layout(&[1, 1, 0, 2, 2], &p).is_err());
        assert!(pool3d_layout(&[1, 1, 2, 0, 2], &p).is_err());
        assert!(pool3d_layout(&[1, 1, 2, 2, 0], &p).is_err());
        // 負分子（カーネルが入力より大きい）。
        assert!(pool3d_layout(&[1, 1, 1, 2, 2], &p).is_err());
        // 空窓: kernel=2・dilation > in_len。
        let pd = params([2, 1, 1], Some([1; 3]), [0; 3], [3, 1, 1]);
        assert!(pool3d_layout(&[1, 1, 2, 2, 2], &pd).is_err());
        // 巨大 shape は確保前に拒否される（panic しない）。
        let pk = params([1; 3], None, [0; 3], [1; 3]);
        assert!(pool3d_layout(&[usize::MAX, 2, 2, 2, 2], &pk).is_err());
    }

    #[test]
    fn params_rejections() {
        assert!(Pool3dParams::new([0, 1, 1], None, [0; 3], [1; 3]).is_err());
        assert!(Pool3dParams::new([1; 3], Some([1, 0, 1]), [0; 3], [1; 3]).is_err());
        assert!(Pool3dParams::new([1; 3], None, [0; 3], [1, 1, 0]).is_err());
        // padding > kernel/2。
        assert!(Pool3dParams::new([2, 2, 2], None, [0, 2, 0], [1; 3]).is_err());
        assert!(Pool3dParams::new([2, 2, 2], None, [usize::MAX, 0, 0], [1; 3]).is_err());
        assert!(Pool3dParams::new([2, 2, 2], None, [1, 1, 1], [1; 3]).is_ok());
    }

    #[test]
    fn i32_index_limit_is_checked_independent_of_batch() {
        let p = params([1; 3], None, [0; 3], [1; 3]);
        // D*H*W = 2^31（i32::MAX 超）。N=0 でも拒否される。
        let layout = pool3d_layout(&[0, 1, 1 << 15, 1 << 15, 2], &p).unwrap();
        assert!(matches!(
            layout.check_i32_indices(),
            Err(ShapeError::IndexRangeOverflow { .. })
        ));
        assert!(max_pool3d_host(&[], &layout).is_err());
    }

    #[test]
    fn overflowing_plane_with_empty_batch_is_typed_error() {
        let p = params([1; 3], None, [0; 3], [1; 3]);
        for shape in [[0, 1, usize::MAX, 2, 1], [1, 0, usize::MAX, 2, 1]] {
            let layout = pool3d_layout(&shape, &p).unwrap();
            assert!(matches!(
                layout.check_i32_indices(),
                Err(ShapeError::ElementCountOverflow)
            ));
            assert!(max_pool3d_host(&[], &layout).is_err());
        }
    }

    #[test]
    fn slice_length_mismatch_is_rejected() {
        let p = params([2, 2, 2], None, [0; 3], [1; 3]);
        let layout = pool3d_layout(&[1, 1, 2, 2, 2], &p).unwrap();
        assert!(max_pool3d_host(&[0.0; 7], &layout).is_err());
        assert!(avg_pool3d_host(&[0.0; 9], &layout, true).is_err());
        assert!(avg_pool3d_vjp_host(&[0.0; 2], &layout, true).is_err());
    }

    #[test]
    fn deterministic_across_runs() {
        let p = params([2, 3, 2], Some([1, 1, 1]), [1, 1, 1], [1; 3]);
        let layout = pool3d_layout(&[2, 3, 4, 5, 4], &p).unwrap();
        let x: Vec<f32> = (0..layout.in_numel())
            .map(|i| ((i * 37) % 11) as f32 * 0.25)
            .collect();
        let a = max_pool3d_host(&x, &layout).unwrap();
        let b = max_pool3d_host(&x, &layout).unwrap();
        assert_eq!(a, b);
        let a = avg_pool3d_host(&x, &layout, false).unwrap();
        let b = avg_pool3d_host(&x, &layout, false).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
