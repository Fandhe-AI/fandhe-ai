//! FractionalMaxPool2d／3d の窓表とホスト参照カーネルの**単一情報源**（イシュー #3826・親 #3749。
//! 設計・実装記録は `docs/autodiff-pool3d-ops-decision.md` §20・§24）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::fractional_max_pool_ops`（標本の検証・窓表の構築・`BackendOps` が `Unsupported`
//!   のときのホストフォールバック）と `backend-cpu` の
//!   `CpuBackendOps::fractional_max_pool{2,3}d_windows` が本モジュールの関数を直接呼ぶ。
//! - 窓は乱数標本から**ホストで 1 回だけ**整数の start 配列へ確定し、`BackendOps` には整数の
//!   [`FractionalMaxPoolWindows`] だけを渡す。GPU 側で窓の式を計算し直さないため、同じ標本なら
//!   バックエンド間で値と索引が bit 一致する（§20.5）。標本の生成そのもの（`rng::rand`）は
//!   本モジュールの責務外で、parity の軸にならない。
//! - CUDA／Metal の専用カーネルは対象外（`BackendOps` 既定の `Unsupported` から autodiff 側が
//!   本ホスト実装へフォールバックする。GPU 実機 parity は #3860／#3861）。
//!
//! # 数値契約
//!
//! - **窓表**: PyTorch ATen の `generate_intervals` と同じ式を **f32 演算**で評価する
//!   （Q1(a)）。`alpha = (in - k) / (out - 1)`（`out > 1` のときだけ）、
//!   `seq[i] = (int)((i + sample) · alpha) - (int)(sample · alpha)`（`i < out - 1`）、
//!   `seq[out - 1] = in - k`。`int` 化は 0 方向への切り捨て。加算と乗算は別々に丸め、
//!   `mul_add` は使わない（FMA 契約は不変）。
//! - **選択**: 窓内を row-major で走査し、更新条件は `v > best || (v.is_nan() && !best.is_nan())`
//!   （タイ先勝ち・窓内で最初の NaN の索引が残る。PyTorch は最後の NaN）。値は入力要素と bit 一致。
//!   索引は `(n, c)` 平面内 flat 添字（`i32`）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! rank・空間軸 0・`kernel`／`output` 0・実行可能性・出力要素数・索引上限・標本の shape と範囲を
//! 確保より前に検査し、各 start は `start + k <= in` を `checked_add` で再検査する。
//! 座標は手動で範囲検査し、`unsafe`／`get_unchecked` は使わない。

use std::fmt;

use crate::error::ShapeError;

/// 窓表の構築・検査で返すエラー。形状違反は [`ShapeError`]、引数の値不正は `InvalidArgument`。
/// autodiff 側が `AutodiffError::Shape`／`InvalidArgument` へ写す。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FractionalMaxPoolError {
    /// shape（rank・軸・要素数）の違反。
    Shape(ShapeError),
    /// `kernel_size`／`output_size`／標本の値が不正（0・実行不能・範囲外・非有限）。
    InvalidArgument(String),
}

impl fmt::Display for FractionalMaxPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(e) => write!(f, "fractional_max_pool: {e}"),
            Self::InvalidArgument(m) => write!(f, "fractional_max_pool: invalid argument: {m}"),
        }
    }
}

impl std::error::Error for FractionalMaxPoolError {}

impl From<ShapeError> for FractionalMaxPoolError {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}

/// 検査済みの窓表（`R` は空間軸数。2d は 2、3d は 3）。フィールドは非公開で、
/// [`fractional_max_pool_windows`] を通った値だけがカーネルへ渡る。
#[derive(Debug, Clone, PartialEq)]
pub struct FractionalMaxPoolWindows<const R: usize> {
    n: usize,
    c: usize,
    in_dims: [usize; R],
    kernel: [usize; R],
    out_dims: [usize; R],
    /// 軸ごとの start 配列。長さは `N·C·out_dims[a]`（`(n·C + c)` 主・出力位置従の並び）。
    starts: [Vec<usize>; R],
}

impl<const R: usize> FractionalMaxPoolWindows<R> {
    /// 入力 shape（`[N, C, *spatial]`）。
    pub fn in_shape(&self) -> Vec<usize> {
        let mut s = vec![self.n, self.c];
        s.extend_from_slice(&self.in_dims);
        s
    }

    /// 出力 shape（`[N, C, *out]`）。
    pub fn out_shape(&self) -> Vec<usize> {
        let mut s = vec![self.n, self.c];
        s.extend_from_slice(&self.out_dims);
        s
    }

    /// 入力の総要素数（検査済みのため桁あふれしない）。
    pub fn in_numel(&self) -> usize {
        self.n * self.c * self.in_dims.iter().product::<usize>()
    }

    /// 出力の総要素数。
    pub fn out_numel(&self) -> usize {
        self.n * self.c * self.out_dims.iter().product::<usize>()
    }

    /// 窓の start（テスト・診断用）。`axis` 軸の `plane`（`n·C + c`）・出力位置 `o`。
    pub fn start(&self, axis: usize, plane: usize, o: usize) -> Option<usize> {
        let len = *self.out_dims.get(axis)?;
        self.starts
            .get(axis)?
            .get(plane.checked_mul(len)?.checked_add(o)?)
            .copied()
    }
}

/// 標本の軸並び（PyTorch と同じ）。空間軸 `a` が `samples[.., idx]` のどれを使うか。
/// 2d は `[0]=W・[1]=H`（空間軸 H=0, W=1 → idx 1, 0）、3d は `[0]=T・[1]=H・[2]=W`。
fn sample_axis_index<const R: usize>(a: usize) -> usize {
    if R == 2 { 1 - a } else { a }
}

fn invalid(msg: String) -> FractionalMaxPoolError {
    FractionalMaxPoolError::InvalidArgument(msg)
}

/// 形状・`kernel`・`output` の検査（§20.3 の 1〜6。確保・RNG 消費・tape 操作より前に呼ぶ）。
///
/// 順序: rank（`R + 2`）→ 空間軸 0 → `kernel`／`output` 0 → 実行可能性（2d は
/// `out + k - 1 <= in`、3d は `out + k - 1 < in`。Q6(a)）→ 出力要素数 →
/// 平面の要素数 `<= i32::MAX`（`N`／`C` が 0 でも検査）。
pub fn fractional_max_pool_check<const R: usize>(
    in_shape: &[usize],
    kernel: [usize; R],
    output: [usize; R],
) -> Result<(), FractionalMaxPoolError> {
    if in_shape.len() != R + 2 {
        return Err(ShapeError::RankMismatch {
            expected: R + 2,
            actual: in_shape.len(),
        }
        .into());
    }
    let spatial = &in_shape[2..];
    if spatial.contains(&0) {
        return Err(ShapeError::ShapeMismatch {
            lhs: spatial.to_vec(),
            rhs: vec![1; R],
        }
        .into());
    }
    if kernel.contains(&0) || output.contains(&0) {
        return Err(invalid(format!(
            "kernel_size {kernel:?} / output_size {output:?} must be >= 1"
        )));
    }
    for a in 0..R {
        let (inp, k, out) = (spatial[a], kernel[a], output[a]);
        let need = out
            .checked_add(k)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| invalid("output_size + kernel_size overflows".into()))?;
        // ATen: 2d は `out + k - 1 <= in`、3d は `out + k - 1 < in`（式が異なる。Q6(a)）。
        let feasible = if R == 2 { need <= inp } else { need < inp };
        if !feasible || k > inp {
            return Err(invalid(format!(
                "axis {a}: output_size {out} + kernel_size {k} - 1 is infeasible for input size {inp}"
            )));
        }
    }
    let mut out_shape = in_shape[..2].to_vec();
    out_shape.extend_from_slice(&output);
    crate::tensor::checked_numel_for::<f32>(&out_shape)?;
    crate::tensor::checked_numel_for::<f32>(in_shape)?;
    check_plane_fits_i32(spatial)?;
    Ok(())
}

fn check_plane_fits_i32(spatial: &[usize]) -> Result<(), ShapeError> {
    let mut plane = 1usize;
    for &d in spatial {
        plane = plane
            .checked_mul(d)
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    if plane > i32::MAX as usize {
        return Err(ShapeError::IndexRangeOverflow { index: plane });
    }
    Ok(())
}

/// ATen `generate_intervals` 相当。f32 演算・0 方向への切り捨て・最後の窓は `in - k` に固定。
/// 各 start を `start + k <= in` で再検査し、違反は fail-closed で拒否する（A03 二重検査）。
fn generate_intervals(
    sample: f32,
    inp: usize,
    k: usize,
    out: usize,
) -> Result<Vec<usize>, FractionalMaxPoolError> {
    let last = inp
        .checked_sub(k)
        .ok_or_else(|| invalid("kernel_size exceeds input size".into()))?;
    let mut seq = Vec::with_capacity(out);
    if out > 1 {
        let alpha = last as f32 / (out - 1) as f32;
        let base = (sample * alpha) as i64;
        for i in 0..out - 1 {
            let v = ((i as f32 + sample) * alpha) as i64 - base;
            let s = usize::try_from(v).map_err(|_| invalid("window start is negative".into()))?;
            seq.push(s);
        }
    }
    if out > 0 {
        seq.push(last);
    }
    for &s in &seq {
        match s.checked_add(k) {
            Some(end) if end <= inp => {}
            _ => {
                return Err(invalid(format!(
                    "window start {s} + kernel {k} exceeds {inp}"
                )));
            }
        }
    }
    Ok(seq)
}

/// 標本から窓表を作る。`samples` は shape `[N, C, R]` の row-major 値、`samples_shape` はその shape。
///
/// 検査: [`fractional_max_pool_check`] → 標本 shape が `[N, C, R]` と完全一致 → 長さ一致 →
/// 全要素が有限かつ `0 <= s < 1`（範囲外の標本は窓を入力の外へ出すため拒否。§20.3 の 7）→
/// 窓表の構築（各 start を再検査。§20.3 の 9）。
pub fn fractional_max_pool_windows<const R: usize>(
    in_shape: &[usize],
    kernel: [usize; R],
    output: [usize; R],
    samples: &[f32],
    samples_shape: &[usize],
) -> Result<FractionalMaxPoolWindows<R>, FractionalMaxPoolError> {
    fractional_max_pool_check::<R>(in_shape, kernel, output)?;
    let (n, c) = (in_shape[0], in_shape[1]);
    if samples_shape != [n, c, R] {
        return Err(ShapeError::ShapeMismatch {
            lhs: samples_shape.to_vec(),
            rhs: vec![n, c, R],
        }
        .into());
    }
    let planes = n * c;
    if samples.len() != planes * R {
        return Err(ShapeError::ElementCountMismatch {
            expected: planes * R,
            actual: samples.len(),
        }
        .into());
    }
    if let Some(bad) = samples
        .iter()
        .find(|s| !(s.is_finite() && **s >= 0.0 && **s < 1.0))
    {
        return Err(invalid(format!(
            "sample {bad} must be finite and in [0, 1)"
        )));
    }
    let mut in_dims = [0usize; R];
    in_dims.copy_from_slice(&in_shape[2..]);
    let mut starts: [Vec<usize>; R] = std::array::from_fn(|_| Vec::new());
    for a in 0..R {
        let si = sample_axis_index::<R>(a);
        let mut v = Vec::with_capacity(planes * output[a]);
        for p in 0..planes {
            v.extend(generate_intervals(
                samples[p * R + si],
                in_dims[a],
                kernel[a],
                output[a],
            )?);
        }
        starts[a] = v;
    }
    Ok(FractionalMaxPoolWindows {
        n,
        c,
        in_dims,
        kernel,
        out_dims: output,
        starts,
    })
}

/// 窓表に従う Fractional max pooling（値と索引）。2d・3d 共通。`input` は論理順の連続スライス。
///
/// 戻り値は `(out_shape の値, 同 shape の索引)`。タイ先勝ち・最初の NaN の索引が残る。
pub fn fractional_max_pool_host<const R: usize>(
    input: &[f32],
    w: &FractionalMaxPoolWindows<R>,
) -> Result<(Vec<f32>, Vec<i32>), ShapeError> {
    if input.len() != w.in_numel() {
        return Err(ShapeError::ElementCountMismatch {
            expected: w.in_numel(),
            actual: input.len(),
        });
    }
    check_plane_fits_i32(&w.in_dims)?;
    let out_numel = w.out_numel();
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let in_plane: usize = w.in_dims.iter().product();
    let out_plane: usize = w.out_dims.iter().product();
    let mut values = Vec::with_capacity(out_numel);
    let mut indices = Vec::with_capacity(out_numel);
    for p in 0..w.n * w.c {
        let plane = &input[p * in_plane..(p + 1) * in_plane];
        for flat_out in 0..out_plane {
            // 出力位置 flat_out -> 軸ごとの座標（row-major）
            let mut rem = flat_out;
            let mut o = [0usize; R];
            for a in (0..R).rev() {
                o[a] = rem % w.out_dims[a];
                rem /= w.out_dims[a];
            }
            let mut begin = [0usize; R];
            for a in 0..R {
                let s = w
                    .starts
                    .get(a)
                    .and_then(|v| v.get(p * w.out_dims[a] + o[a]))
                    .copied()
                    .ok_or(ShapeError::IndexRangeOverflow { index: p })?;
                match s.checked_add(w.kernel[a]) {
                    Some(e) if e <= w.in_dims[a] => {}
                    _ => return Err(ShapeError::IndexRangeOverflow { index: s }),
                }
                begin[a] = s;
            }
            let win: usize = w.kernel.iter().product();
            let mut best: Option<(f32, usize)> = None;
            for t in 0..win {
                let mut rem = t;
                let mut flat = 0usize;
                let mut coord = [0usize; R];
                for a in (0..R).rev() {
                    coord[a] = begin[a] + rem % w.kernel[a];
                    rem /= w.kernel[a];
                }
                for (a, &cv) in coord.iter().enumerate() {
                    flat = flat * w.in_dims[a] + cv;
                }
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
            }
            let (v, flat) = best.ok_or(ShapeError::ElementCountOverflow)?;
            values.push(v);
            indices.push(
                i32::try_from(flat).map_err(|_| ShapeError::IndexRangeOverflow { index: flat })?,
            );
        }
    }
    Ok((values, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BELOW_ONE: f32 = 1.0 - 1.0 / 16_777_216.0;

    fn w2(
        in_shape: &[usize],
        k: [usize; 2],
        o: [usize; 2],
        s: &[f32],
    ) -> Result<FractionalMaxPoolWindows<2>, FractionalMaxPoolError> {
        fractional_max_pool_windows(in_shape, k, o, s, &[in_shape[0], in_shape[1], 2])
    }

    #[test]
    fn intervals_formula_and_last_window_fixed() {
        // in=10,k=2,out=4: alpha=8/3
        let s = generate_intervals(0.5, 10, 2, 4).unwrap();
        // i=0: (0.5*alpha)=1.333->1 -1 =0; i=1: 1.5*alpha=4->4-1=3; i=2: 2.5*alpha=6.667->6-1=5; last=8
        assert_eq!(s, vec![0, 3, 5, 8]);
        assert_eq!(*s.last().unwrap(), 10 - 2);
    }

    #[test]
    fn intervals_out_one_and_contiguous() {
        assert_eq!(generate_intervals(0.7, 5, 3, 1).unwrap(), vec![2]);
        // out = in-k+1 => alpha=1 -> consecutive windows regardless of sample
        // 標本が 1 に極めて近いと f32 の丸めで `i + s` が整数へ繰り上がる（ATen と同じ挙動）ため除く
        for s in [0.0, 0.3, 0.9] {
            assert_eq!(generate_intervals(s, 6, 3, 4).unwrap(), vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn intervals_sample_extremes_stay_in_bounds() {
        for s in [0.0, BELOW_ONE] {
            for (inp, k, out) in [(7, 2, 3), (13, 3, 5), (100, 7, 9)] {
                let v = generate_intervals(s, inp, k, out).unwrap();
                assert!(v.iter().all(|&x| x + k <= inp));
                assert!(v.windows(2).all(|p| p[0] <= p[1]));
            }
        }
    }

    #[test]
    fn check_feasibility_boundaries_differ_between_2d_and_3d() {
        // 2d: out + k - 1 == in は受理
        assert!(fractional_max_pool_check::<2>(&[1, 1, 4, 4], [2, 2], [3, 3]).is_ok());
        assert!(fractional_max_pool_check::<2>(&[1, 1, 4, 4], [2, 2], [4, 3]).is_err());
        // 3d: 等号は拒否、1 つ小さければ受理
        assert!(fractional_max_pool_check::<3>(&[1, 1, 4, 4, 4], [2, 2, 2], [3, 3, 3]).is_err());
        assert!(fractional_max_pool_check::<3>(&[1, 1, 4, 4, 4], [2, 2, 2], [2, 2, 2]).is_ok());
    }

    #[test]
    fn check_rejects_bad_rank_zero_axes_and_huge() {
        assert!(fractional_max_pool_check::<2>(&[1, 4, 4], [1, 1], [1, 1]).is_err());
        assert!(fractional_max_pool_check::<2>(&[1, 1, 0, 4], [1, 1], [1, 1]).is_err());
        assert!(fractional_max_pool_check::<2>(&[1, 1, 4, 4], [0, 1], [1, 1]).is_err());
        assert!(fractional_max_pool_check::<2>(&[1, 1, 4, 4], [1, 1], [1, 0]).is_err());
        assert!(fractional_max_pool_check::<2>(&[1, 1, 4, 4], [5, 1], [1, 1]).is_err());
        // N=0 でも平面が i32::MAX を超えれば拒否
        let big = 1usize << 16;
        assert!(fractional_max_pool_check::<2>(&[0, 1, big, big], [1, 1], [1, 1]).is_err());
        assert!(fractional_max_pool_check::<2>(&[1, 1, usize::MAX, 2], [1, 1], [1, 1]).is_err());
    }

    #[test]
    fn samples_are_validated() {
        let ok = [0.1, 0.2];
        assert!(w2(&[1, 1, 4, 4], [2, 2], [2, 2], &ok).is_ok());
        for bad in [
            [1.0, 0.0],
            [-0.1, 0.0],
            [f32::NAN, 0.0],
            [f32::INFINITY, 0.0],
        ] {
            assert!(w2(&[1, 1, 4, 4], [2, 2], [2, 2], &bad).is_err());
        }
        // shape 違い・長さ違い
        assert!(
            fractional_max_pool_windows::<2>(&[1, 1, 4, 4], [2, 2], [2, 2], &ok, &[1, 2, 1])
                .is_err()
        );
        assert!(
            fractional_max_pool_windows::<2>(&[1, 1, 4, 4], [2, 2], [2, 2], &[0.1], &[1, 1, 2])
                .is_err()
        );
    }

    #[test]
    fn sample_axis_order_is_w_then_h_for_2d() {
        // samples[0]=W, samples[1]=H。H 軸は sample 0.9、W 軸は 0.0 を使う
        let w = w2(&[1, 1, 10, 10], [2, 2], [4, 4], &[0.0, 0.5]).unwrap();
        let h_starts: Vec<_> = (0..4).map(|o| w.start(0, 0, o).unwrap()).collect();
        let w_starts: Vec<_> = (0..4).map(|o| w.start(1, 0, o).unwrap()).collect();
        assert_eq!(h_starts, generate_intervals(0.5, 10, 2, 4).unwrap());
        assert_eq!(w_starts, generate_intervals(0.0, 10, 2, 4).unwrap());
    }

    #[test]
    fn host_kernel_picks_max_with_flat_plane_index_and_ties_first() {
        // 4x4, k=2, out=3 (alpha=1): 重なり窓。タイは先勝ち
        #[rustfmt::skip]
        let x = [
            1.0, 5.0, 2.0, 0.0,
            3.0, 5.0, 9.0, 1.0,
            0.0, 0.0, 2.0, 2.0,
            7.0, 1.0, 2.0, 8.0,
        ];
        let w = w2(&[1, 1, 4, 4], [2, 2], [3, 3], &[0.3, 0.6]).unwrap();
        let (v, i) = fractional_max_pool_host(&x, &w).unwrap();
        assert_eq!(v, vec![5.0, 9.0, 9.0, 5.0, 9.0, 9.0, 7.0, 2.0, 8.0]);
        assert_eq!(i, vec![1, 6, 6, 5, 6, 6, 12, 10, 15]);
        // 決定性
        assert_eq!(fractional_max_pool_host(&x, &w).unwrap(), (v, i));
    }

    #[test]
    fn host_kernel_nan_first_and_neg_inf() {
        let x = [f32::NEG_INFINITY, f32::NAN, f32::NAN, 1.0];
        let w = w2(&[1, 1, 2, 2], [2, 2], [1, 1], &[0.0, 0.0]).unwrap();
        let (v, i) = fractional_max_pool_host(&x, &w).unwrap();
        assert!(v[0].is_nan());
        assert_eq!(i, vec![1]);
        let x = [f32::NEG_INFINITY; 4];
        let (v, i) = fractional_max_pool_host(&x, &w).unwrap();
        assert_eq!((v[0], i[0]), (f32::NEG_INFINITY, 0));
    }

    #[test]
    fn host_kernel_3d_and_empty_batch() {
        let w = fractional_max_pool_windows::<3>(
            &[1, 1, 2, 3, 3],
            [1, 2, 2],
            [1, 2, 2],
            &[0.0, 0.0, 0.0],
            &[1, 1, 3],
        );
        // 3d は out + k - 1 < in を要求: D 軸 1+1-1=1 < 2 OK, H 軸 2+2-1=3 < 3 NG
        assert!(w.is_err());
        let x: Vec<f32> = (0..3 * 4 * 4).map(|v| v as f32).collect();
        let w = fractional_max_pool_windows::<3>(
            &[1, 1, 3, 4, 4],
            [1, 2, 2],
            [2, 2, 2],
            &[0.0, 0.0, 0.0],
            &[1, 1, 3],
        )
        .unwrap();
        let (v, i) = fractional_max_pool_host(&x, &w).unwrap();
        assert_eq!(v.len(), 8);
        assert!(i.iter().all(|&f| (0..48).contains(&f)));
        let w = fractional_max_pool_windows::<2>(&[0, 3, 4, 4], [2, 2], [2, 2], &[], &[0, 3, 2])
            .unwrap();
        assert_eq!(fractional_max_pool_host(&[], &w).unwrap(), (vec![], vec![]));
    }

    #[test]
    fn host_kernel_rejects_wrong_input_len() {
        let w = w2(&[1, 1, 4, 4], [2, 2], [2, 2], &[0.1, 0.2]).unwrap();
        assert!(fractional_max_pool_host(&[0.0; 15], &w).is_err());
    }
}
