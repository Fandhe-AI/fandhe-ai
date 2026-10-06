//! 重み再パラメータ化（`weight_norm`・`spectral_norm`）のホスト参照カーネルの**単一情報源**
//! （イシュー #2646・親 #2625「Phase 4」・ルート #2499。実装記録は
//! `docs/autodiff-lrn-weight-reparam-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::weight_reparam_ops`（`BackendOps::{weight_norm_forward, spectral_norm_forward}` が
//!   `Unsupported` のときのホストフォールバック・`SpectralNormState` の power iteration）・
//!   `autodiff::grad`（`Op::WeightNorm`／`Op::SpectralNorm` の VJP）・`backend-cpu` の
//!   `CpuBackendOps` の override がいずれも本モジュールの関数を直接呼ぶ。軸の扱い・アキュムレータ
//!   契約をクレート間で複製しない。
//! - CUDA／Metal の専用カーネルは本イシューの対象外（`BackendOps` 既定の `Unsupported` から
//!   ホストへフォールバックする）。
//!
//! # weight_norm
//!
//! `w = v · (g / ‖v‖)`。`‖v‖` は `dim` 以外の全軸の L2（`dim = None` はテンソル全体。PyTorch の
//! `dim=-1`）。グループ（`dim` の各添字）ごとに `n = sqrt(Σ (f64)v²)`（昇格してから二乗・index 順）、
//! `w = v · (g / n)` を `f64` で計算して 1 回 downcast する。式順は PyTorch（`v * (g / norm)`）に揃え、
//! `‖v‖ = 0` 時の非有限クラスを一致させる（拒否せず伝播）。`g` の shape は [`norm_except_dim_shape`]
//! と完全一致のみ受理する（`Some(dim)` は keepdim 形・`None` は rank 0）。
//! VJP は `dot = Σ up·v`（要素積は `f32` で確定してから `f64` へ昇格・index 順）、`dg = dot/n`、
//! `dv = (g/n)·up − (g·dot/n³)·v`（`f64`・1 回 downcast）。
//! 軸は permute コピーせず `(outer, axis_len, inner)` 分解でグループ番号を求める。
//!
//! # spectral_norm
//!
//! `W_mat` は「`dim` を先頭へ出して残りを元の順で平坦化」した `[h, w]`（行 `i` = `dim` 添字、列
//! `j = outer·inner_len + inner`。permute コピー不要）。power iteration は `u ← normalize(W v)` →
//! `v ← normalize(Wᵀ u)` の順（**u が先**。PyTorch `_power_method`）で、行列ベクトル積・ノルムは
//! `f64` 蓄積・各反復の `u`／`v` は `f32` で保持する。`normalize(x) = x / max(‖x‖, eps)`
//! （`F.normalize`。`+ eps` ではない）。`σ = uᵀ W_mat v` は `f64` 蓄積、`out = W / σ` は `f64` 除算・
//! 1 回 downcast。`σ = 0` は拒否せず伝播。VJP は `s = Σ up·W`（要素積 `f32` 確定 → `f64` 蓄積）、
//! `dW_ij = up_ij/σ − (s/σ²)·u_i·v_j`（`u`／`v` は定数。PyTorch も buffer の clone で勾配を流さない）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状は各 `*_layout` が確保前に `checked_numel_for` で検査し、スライス長・`u`／`v`／`g` の長さを
//! レイアウトと照合して型付きエラーで返す。`unsafe`／`get_unchecked` は使わない。

use crate::error::ShapeError;
use crate::tensor::checked_numel_for;

/// `(outer, axis_len, inner)` 分解。`dim = None` は 1 グループ。
fn decompose(shape: &[usize], dim: Option<usize>) -> Result<(usize, usize, usize), ShapeError> {
    match dim {
        None => Ok((1, shape.iter().product(), 1)),
        Some(d) => {
            if d >= shape.len() {
                return Err(ShapeError::AxisOutOfRange {
                    axis: d,
                    rank: shape.len(),
                });
            }
            Ok((
                shape[..d].iter().product(),
                shape[d],
                shape[d + 1..].iter().product(),
            ))
        }
    }
}

/// `norm_except_dim`／`weight_norm` が返す（受理する）ノルム shape。`Some(dim)` は `dim` 以外が 1 の
/// keepdim 形、`None` は rank 0（`[]`）。rank 0 の `v` は拒否する。
pub fn norm_except_dim_shape(
    v_shape: &[usize],
    dim: Option<usize>,
) -> Result<Vec<usize>, ShapeError> {
    if v_shape.is_empty() {
        return Err(ShapeError::RankMismatch {
            expected: 1,
            actual: 0,
        });
    }
    match dim {
        None => Ok(Vec::new()),
        Some(d) => {
            if d >= v_shape.len() {
                return Err(ShapeError::AxisOutOfRange {
                    axis: d,
                    rank: v_shape.len(),
                });
            }
            let mut s = vec![1; v_shape.len()];
            s[d] = v_shape[d];
            Ok(s)
        }
    }
}

/// 検査済みの weight_norm レイアウト。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightNormLayout {
    shape: Vec<usize>,
    g_shape: Vec<usize>,
    outer: usize,
    axis_len: usize,
    inner: usize,
    groups: usize,
}

impl WeightNormLayout {
    /// `v` の shape。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// `g`（および `norm_except_dim` の出力）の shape。
    pub fn g_shape(&self) -> &[usize] {
        &self.g_shape
    }

    /// グループ数（`g` の要素数）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    fn numel(&self) -> usize {
        self.outer * self.axis_len * self.inner
    }

    fn group_of(&self, flat: usize) -> usize {
        if self.groups <= 1 || self.inner == 0 {
            return 0;
        }
        (flat / self.inner) % self.axis_len
    }
}

/// `v` だけから作るレイアウト（`norm_except_dim` 用）。rank >= 1・`dim < rank`・確保前検査。
pub fn norm_except_dim_layout(
    v_shape: &[usize],
    dim: Option<usize>,
) -> Result<WeightNormLayout, ShapeError> {
    let g_shape = norm_except_dim_shape(v_shape, dim)?;
    checked_numel_for::<f32>(v_shape)?;
    let (outer, axis_len, inner) = decompose(v_shape, dim)?;
    let groups = if dim.is_some() { axis_len } else { 1 };
    Ok(WeightNormLayout {
        shape: v_shape.to_vec(),
        g_shape,
        outer,
        axis_len,
        inner,
        groups,
    })
}

/// `g` の shape が [`norm_except_dim_shape`] と完全一致することまで検査したレイアウト。
pub fn weight_norm_layout(
    v_shape: &[usize],
    g_shape: &[usize],
    dim: Option<usize>,
) -> Result<WeightNormLayout, ShapeError> {
    let layout = norm_except_dim_layout(v_shape, dim)?;
    if layout.g_shape != g_shape {
        return Err(ShapeError::ShapeMismatch {
            lhs: g_shape.to_vec(),
            rhs: layout.g_shape,
        });
    }
    Ok(layout)
}

fn check_len(len: usize, expected: usize) -> Result<(), ShapeError> {
    if len != expected {
        return Err(ShapeError::ElementCountMismatch {
            expected,
            actual: len,
        });
    }
    Ok(())
}

fn group_norms(v: &[f32], layout: &WeightNormLayout) -> Vec<f64> {
    let mut sq = vec![0.0_f64; layout.groups];
    for (i, &x) in v.iter().enumerate() {
        let xd = f64::from(x);
        sq[layout.group_of(i)] += xd * xd;
    }
    sq.into_iter().map(f64::sqrt).collect()
}

/// `torch.norm_except_dim(v, 2, dim)`。長さ `groups` の `Vec`（shape は `layout.g_shape()`）。
pub fn norm_except_dim_host(v: &[f32], layout: &WeightNormLayout) -> Result<Vec<f32>, ShapeError> {
    check_len(v.len(), layout.numel())?;
    Ok(group_norms(v, layout)
        .into_iter()
        .map(|n| n as f32)
        .collect())
}

/// `w = v · (g / ‖v‖)`。
pub fn weight_norm_host(
    v: &[f32],
    g: &[f32],
    layout: &WeightNormLayout,
) -> Result<Vec<f32>, ShapeError> {
    check_len(v.len(), layout.numel())?;
    check_len(g.len(), layout.groups)?;
    let norms = group_norms(v, layout);
    let scale: Vec<f64> = g
        .iter()
        .zip(&norms)
        .map(|(&gv, &n)| f64::from(gv) / n)
        .collect();
    Ok(v.iter()
        .enumerate()
        .map(|(i, &x)| (f64::from(x) * scale[layout.group_of(i)]) as f32)
        .collect())
}

/// `(dv, dg)`。`dg` の長さは `groups`（shape は `layout.g_shape()`）。
pub fn weight_norm_vjp_host(
    v: &[f32],
    g: &[f32],
    upstream: &[f32],
    layout: &WeightNormLayout,
) -> Result<(Vec<f32>, Vec<f32>), ShapeError> {
    check_len(v.len(), layout.numel())?;
    check_len(upstream.len(), layout.numel())?;
    check_len(g.len(), layout.groups)?;
    let norms = group_norms(v, layout);
    let mut dot = vec![0.0_f64; layout.groups];
    for (i, (&x, &u)) in v.iter().zip(upstream).enumerate() {
        // 要素積は f32 で確定してから f64 へ昇格する（dw の行方向蓄積と同じ契約形）。
        let term = u * x;
        dot[layout.group_of(i)] += f64::from(term);
    }
    let dg: Vec<f32> = dot
        .iter()
        .zip(&norms)
        .map(|(&d, &n)| (d / n) as f32)
        .collect();
    let dv: Vec<f32> = v
        .iter()
        .zip(upstream)
        .enumerate()
        .map(|(i, (&x, &u))| {
            let k = layout.group_of(i);
            let n = norms[k];
            let gk = f64::from(g[k]);
            let a = gk / n;
            let b = gk * dot[k] / (n * n * n);
            (a * f64::from(u) - b * f64::from(x)) as f32
        })
        .collect();
    Ok((dv, dg))
}

/// 検査済みの spectral_norm レイアウト。`W_mat` は `[h, w]`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectralLayout {
    shape: Vec<usize>,
    dim: usize,
    outer: usize,
    h: usize,
    inner: usize,
}

impl SpectralLayout {
    /// 重みの shape。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// `dim`。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 行数 `h = shape[dim]`。
    pub fn rows(&self) -> usize {
        self.h
    }

    /// 列数 `w = numel / h`。
    pub fn cols(&self) -> usize {
        self.outer * self.inner
    }

    fn numel(&self) -> usize {
        self.outer * self.h * self.inner
    }

    /// `W_mat[i][j]` の flat 添字。
    fn at(&self, i: usize, j: usize) -> usize {
        let (o, t) = (j / self.inner, j % self.inner);
        (o * self.h + i) * self.inner + t
    }
}

/// rank 2 以上・`dim < rank`・要素数 0 の拒否・確保前検査。
pub fn spectral_norm_layout(shape: &[usize], dim: usize) -> Result<SpectralLayout, ShapeError> {
    if shape.len() < 2 {
        return Err(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        });
    }
    if dim >= shape.len() {
        return Err(ShapeError::AxisOutOfRange {
            axis: dim,
            rank: shape.len(),
        });
    }
    let numel = checked_numel_for::<f32>(shape)?;
    if numel == 0 {
        return Err(ShapeError::ElementCountMismatch {
            expected: 1,
            actual: 0,
        });
    }
    let (outer, h, inner) = decompose(shape, Some(dim))?;
    Ok(SpectralLayout {
        shape: shape.to_vec(),
        dim,
        outer,
        h,
        inner,
    })
}

/// `F.normalize` 相当（`x / max(‖x‖, eps)`）。NaN は `clamp_min` と同じく伝播させる。
fn normalize_into(tmp: &[f64], eps: f32, out: &mut [f32]) {
    let norm = tmp.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = if norm.is_nan() {
        norm
    } else {
        norm.max(f64::from(eps))
    };
    for (o, &x) in out.iter_mut().zip(tmp) {
        *o = (x / denom) as f32;
    }
}

/// `u`／`v` を正規化して返す（`SpectralNormState::from_vectors` 用。`x / max(‖x‖, eps)`）。
pub fn normalize_vector_host(x: &[f32], eps: f32) -> Vec<f32> {
    let tmp: Vec<f64> = x.iter().map(|&a| f64::from(a)).collect();
    let mut out = vec![0.0_f32; x.len()];
    normalize_into(&tmp, eps, &mut out);
    out
}

/// power iteration を `n` 回進める（**u が先**）。`u`（長さ `h`）・`v`（長さ `w`）を更新する。
pub fn spectral_power_iterate_host(
    weight: &[f32],
    layout: &SpectralLayout,
    u: &mut [f32],
    v: &mut [f32],
    n: usize,
    eps: f32,
) -> Result<(), ShapeError> {
    check_len(weight.len(), layout.numel())?;
    check_len(u.len(), layout.h)?;
    check_len(v.len(), layout.cols())?;
    let (h, w) = (layout.h, layout.cols());
    let mut tu = vec![0.0_f64; h];
    let mut tv = vec![0.0_f64; w];
    for _ in 0..n {
        for (i, slot) in tu.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (j, &vj) in v.iter().enumerate() {
                acc += f64::from(weight[layout.at(i, j)]) * f64::from(vj);
            }
            *slot = acc;
        }
        normalize_into(&tu, eps, u);
        for (j, slot) in tv.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (i, &ui) in u.iter().enumerate() {
                acc += f64::from(weight[layout.at(i, j)]) * f64::from(ui);
            }
            *slot = acc;
        }
        normalize_into(&tv, eps, v);
    }
    Ok(())
}

/// `σ = uᵀ W_mat v`（`f64` 蓄積）。
pub fn spectral_sigma_host(
    weight: &[f32],
    layout: &SpectralLayout,
    u: &[f32],
    v: &[f32],
) -> Result<f64, ShapeError> {
    check_len(weight.len(), layout.numel())?;
    check_len(u.len(), layout.h)?;
    check_len(v.len(), layout.cols())?;
    let mut sigma = 0.0_f64;
    for (i, &ui) in u.iter().enumerate() {
        let mut row = 0.0_f64;
        for (j, &vj) in v.iter().enumerate() {
            row += f64::from(weight[layout.at(i, j)]) * f64::from(vj);
        }
        sigma += f64::from(ui) * row;
    }
    Ok(sigma)
}

/// `out = W / σ`。
pub fn spectral_norm_host(
    weight: &[f32],
    layout: &SpectralLayout,
    u: &[f32],
    v: &[f32],
) -> Result<Vec<f32>, ShapeError> {
    let sigma = spectral_sigma_host(weight, layout, u, v)?;
    Ok(weight
        .iter()
        .map(|&x| (f64::from(x) / sigma) as f32)
        .collect())
}

/// VJP。`dW_ij = up_ij/σ − (s/σ²)·u_i·v_j`、`s = Σ up·W`。
pub fn spectral_norm_vjp_host(
    weight: &[f32],
    upstream: &[f32],
    layout: &SpectralLayout,
    u: &[f32],
    v: &[f32],
) -> Result<Vec<f32>, ShapeError> {
    check_len(upstream.len(), layout.numel())?;
    let sigma = spectral_sigma_host(weight, layout, u, v)?;
    let mut s = 0.0_f64;
    for (&up, &x) in upstream.iter().zip(weight) {
        // 要素積は f32 で確定してから f64 へ昇格する。
        let term = up * x;
        s += f64::from(term);
    }
    let coeff = s / (sigma * sigma);
    let w = layout.cols();
    let mut out = vec![0.0_f32; layout.numel()];
    for (i, &ui) in u.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate().take(w) {
            let idx = layout.at(i, j);
            let val = f64::from(upstream[idx]) / sigma - coeff * f64::from(ui) * f64::from(vj);
            out[idx] = val as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_except_dim_dim0_dim1_and_whole() {
        // v = [[3, 0], [0, 4]]。
        let v = [3.0_f32, 0.0, 0.0, 4.0];
        let l0 = norm_except_dim_layout(&[2, 2], Some(0)).unwrap();
        assert_eq!(l0.g_shape(), &[2, 1]);
        assert_eq!(norm_except_dim_host(&v, &l0).unwrap(), vec![3.0, 4.0]);
        let l1 = norm_except_dim_layout(&[2, 2], Some(1)).unwrap();
        assert_eq!(l1.g_shape(), &[1, 2]);
        assert_eq!(norm_except_dim_host(&v, &l1).unwrap(), vec![3.0, 4.0]);
        let ln = norm_except_dim_layout(&[2, 2], None).unwrap();
        assert!(ln.g_shape().is_empty());
        assert_eq!(norm_except_dim_host(&v, &ln).unwrap(), vec![5.0]);
        // rank 1。
        let l = norm_except_dim_layout(&[3], Some(0)).unwrap();
        assert_eq!(l.g_shape(), &[3]);
    }

    #[test]
    fn weight_norm_roundtrip_with_own_norm_is_identity() {
        let v: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.37).collect();
        for dim in [Some(0), Some(1), Some(2), None] {
            let shape = [2, 3, 4];
            let l = norm_except_dim_layout(&shape, dim).unwrap();
            let g = norm_except_dim_host(&v, &l).unwrap();
            let wl = weight_norm_layout(&shape, l.g_shape(), dim).unwrap();
            let w = weight_norm_host(&v, &g, &wl).unwrap();
            for (a, b) in w.iter().zip(&v) {
                assert!((a - b).abs() < 1e-5, "{dim:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn weight_norm_hand_computed_and_zero_norm_propagates() {
        let l = weight_norm_layout(&[2, 2], &[2, 1], Some(0)).unwrap();
        // 行 0: v=(3,4) n=5 g=10 → (6,8)。行 1: ゼロノルム → 0·inf = NaN。
        let w = weight_norm_host(&[3.0, 4.0, 0.0, 0.0], &[10.0, 1.0], &l).unwrap();
        assert_eq!(&w[..2], &[6.0, 8.0]);
        assert!(w[2].is_nan() && w[3].is_nan());
    }

    #[test]
    fn weight_norm_squares_do_not_overflow_in_f32() {
        let l = weight_norm_layout(&[2], &[], None).unwrap();
        let w = weight_norm_host(&[2e20, 0.0], &[1.0], &l).unwrap();
        assert!((w[0] - 1.0).abs() < 1e-6 && w[1] == 0.0, "{w:?}");
    }

    #[test]
    fn weight_norm_rejects_bad_shapes() {
        assert!(weight_norm_layout(&[], &[], None).is_err());
        assert!(weight_norm_layout(&[2, 2], &[2], Some(0)).is_err());
        assert!(weight_norm_layout(&[2, 2], &[2, 1], Some(2)).is_err());
        assert!(weight_norm_layout(&[2, 2], &[2, 1], None).is_err());
        assert!(norm_except_dim_layout(&[usize::MAX, 2], Some(0)).is_err());
        let l = weight_norm_layout(&[2, 2], &[2, 1], Some(0)).unwrap();
        assert!(weight_norm_host(&[0.0; 3], &[1.0, 1.0], &l).is_err());
        assert!(weight_norm_host(&[0.0; 4], &[1.0], &l).is_err());
    }

    #[test]
    fn weight_norm_vjp_matches_central_difference() {
        let shape = [3, 4];
        let v: Vec<f32> = (0..12)
            .map(|i| ((i * 5 % 7) as f32 - 3.0) * 0.4 + 0.1)
            .collect();
        let up: Vec<f32> = (0..12).map(|i| ((i * 3 % 5) as f32 - 2.0) * 0.5).collect();
        for dim in [Some(0), Some(1), None] {
            let l = norm_except_dim_layout(&shape, dim).unwrap();
            let g: Vec<f32> = (0..l.groups()).map(|k| 0.5 + k as f32 * 0.3).collect();
            let (dv, dg) = weight_norm_vjp_host(&v, &g, &up, &l).unwrap();
            let loss = |vs: &[f32], gs: &[f32]| -> f64 {
                weight_norm_host(vs, gs, &l)
                    .unwrap()
                    .iter()
                    .zip(&up)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum()
            };
            let h = 1e-2_f32;
            for i in 0..v.len() {
                let (mut p, mut m) = (v.clone(), v.clone());
                p[i] += h;
                m[i] -= h;
                let num = (loss(&p, &g) - loss(&m, &g)) / (2.0 * f64::from(h));
                assert!((num - f64::from(dv[i])).abs() < 5e-3, "{dim:?} dv[{i}]");
            }
            for k in 0..g.len() {
                let (mut p, mut m) = (g.clone(), g.clone());
                p[k] += h;
                m[k] -= h;
                let num = (loss(&v, &p) - loss(&v, &m)) / (2.0 * f64::from(h));
                assert!((num - f64::from(dg[k])).abs() < 5e-3, "{dim:?} dg[{k}]");
            }
        }
    }

    #[test]
    fn spectral_known_singular_value_of_diagonal_matrix() {
        // diag(3, 1)。最大特異値 3。
        let w = [3.0_f32, 0.0, 0.0, 1.0];
        let l = spectral_norm_layout(&[2, 2], 0).unwrap();
        let mut u = normalize_vector_host(&[1.0, 1.0], 1e-12);
        let mut v = normalize_vector_host(&[1.0, 1.0], 1e-12);
        spectral_power_iterate_host(&w, &l, &mut u, &mut v, 40, 1e-12).unwrap();
        let sigma = spectral_sigma_host(&w, &l, &u, &v).unwrap();
        assert!((sigma - 3.0).abs() < 1e-5, "{sigma}");
        let out = spectral_norm_host(&w, &l, &u, &v).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn spectral_dim1_uses_transposed_matrix() {
        // 重み [2, 3]・dim=1 → W_mat = Wᵀ（[3, 2]）。特異値は dim=0 と一致する。
        let w = [1.0_f32, 2.0, 0.5, -1.0, 0.25, 3.0];
        let l0 = spectral_norm_layout(&[2, 3], 0).unwrap();
        let l1 = spectral_norm_layout(&[2, 3], 1).unwrap();
        assert_eq!((l0.rows(), l0.cols()), (2, 3));
        assert_eq!((l1.rows(), l1.cols()), (3, 2));
        let sig = |l: &SpectralLayout| {
            let mut u = vec![1.0_f32; l.rows()];
            let mut v = vec![1.0_f32; l.cols()];
            u = normalize_vector_host(&u, 1e-12);
            v = normalize_vector_host(&v, 1e-12);
            spectral_power_iterate_host(&w, l, &mut u, &mut v, 200, 1e-12).unwrap();
            spectral_sigma_host(&w, l, &u, &v).unwrap()
        };
        assert!((sig(&l0) - sig(&l1)).abs() < 1e-4);
    }

    #[test]
    fn spectral_vjp_matches_central_difference() {
        let shape = [3, 2, 2];
        let w: Vec<f32> = (0..12)
            .map(|i| ((i * 7 % 5) as f32 - 2.0) * 0.6 + 0.2)
            .collect();
        let up: Vec<f32> = (0..12).map(|i| ((i * 5 % 7) as f32 - 3.0) * 0.3).collect();
        for dim in [0_usize, 1, 2] {
            let l = spectral_norm_layout(&shape, dim).unwrap();
            let u = normalize_vector_host(&vec![1.0; l.rows()], 1e-12);
            let v = normalize_vector_host(&vec![0.5; l.cols()], 1e-12);
            let dw = spectral_norm_vjp_host(&w, &up, &l, &u, &v).unwrap();
            let loss = |ws: &[f32]| -> f64 {
                spectral_norm_host(ws, &l, &u, &v)
                    .unwrap()
                    .iter()
                    .zip(&up)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum()
            };
            let h = 1e-2_f32;
            for i in 0..w.len() {
                let (mut p, mut m) = (w.clone(), w.clone());
                p[i] += h;
                m[i] -= h;
                let num = (loss(&p) - loss(&m)) / (2.0 * f64::from(h));
                assert!((num - f64::from(dw[i])).abs() < 5e-3, "dim={dim} dw[{i}]");
            }
        }
    }

    #[test]
    fn spectral_rejects_bad_inputs() {
        assert!(spectral_norm_layout(&[3], 0).is_err());
        assert!(spectral_norm_layout(&[2, 2], 2).is_err());
        assert!(spectral_norm_layout(&[0, 2], 0).is_err());
        assert!(spectral_norm_layout(&[usize::MAX, 2], 0).is_err());
        let l = spectral_norm_layout(&[2, 2], 0).unwrap();
        assert!(spectral_sigma_host(&[0.0; 3], &l, &[0.0; 2], &[0.0; 2]).is_err());
        assert!(spectral_sigma_host(&[0.0; 4], &l, &[0.0; 3], &[0.0; 2]).is_err());
        assert!(spectral_norm_vjp_host(&[0.0; 4], &[0.0; 3], &l, &[0.0; 2], &[0.0; 2]).is_err());
    }

    #[test]
    fn spectral_zero_sigma_propagates_non_finite() {
        let l = spectral_norm_layout(&[2, 2], 0).unwrap();
        let out = spectral_norm_host(&[1.0, 0.0, 0.0, 0.0], &l, &[0.0, 1.0], &[0.0, 1.0]).unwrap();
        assert!(out[0].is_infinite() && out[1].is_nan());
    }
}
