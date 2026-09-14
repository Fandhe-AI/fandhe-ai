//! `TypedOps<f64>` の CPU 実装（イシュー #1697・親 #1649）。
//!
//! `crate::ops::CpuBackendOps` から `BackendOps::typed_ops_f64()`（accessor。
//! `ops.rs` 参照）経由でのみ到達する f64 演算本体。`docs/backend-dtype-dispatch-design.md`
//! §4 案 D（型パラメータ trait 1 本への集約）に従い、対象は最小集合 8 演算
//! （`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`）に固定する
//! （設計 §4.2・本イシューの承認事項 6）。
//!
//! # f32 ホットパスとの関係（自己完結モジュール方針）
//!
//! 本モジュールは `crate::elementwise`／`crate::reduction`／`crate::gemm`／
//! `crate::gemm_blis` の f32 実装本体を一切変更せず、走査構造（2 層構成の
//! 「スライスカーネル層」「Tensor 入口層」・`ElementwiseReadOperand` 相当の
//! stride 読み・`sum`/`max` の `CHUNK` 単位分割）を f64 向けに書き写す形で
//! 自己完結させる。理由:
//!
//! 1. `.claude/rules/coding-rust.md` の「f32 経路 bit 同一」契約を
//!    `git diff --stat`（f32 実装ファイルへの変更が可視性変更〈`pub(crate)`
//!    化〉のみであること）で構造的に示せる
//! 2. #1698（f16）／#1699（bf16）が同じ base から並走しうるため、共有
//!    ファイルの汎用化（`T: Element` ジェネリック化）はコンフリクト源になる
//! 3. `docs/backend-dtype-dispatch-design.md` §4.1 は共通境界の追加を
//!    「複数 impl で実際に共通化が必要になった時点」と定めている
//!
//! `crate::reduction` の `CHUNK`（決定性契約の固定チャンクサイズ）・
//! `unravel`（線形 index の多次元展開）・`checked_product`（オーバーフロー
//! 検査つき次元積）は `pub(crate)` 化して再利用し、チャンク分割・展開
//! ロジックの二重管理を避ける。`crate::elementwise::increment_index` も
//! 同様に再利用する。
//!
//! # 数値契約
//!
//! - `gemm`: 累算は `f64::mul_add`（FMA 契約統一の f64 版）。ikj 順・
//!   出力要素ごとの累積順序は逐次と同一。並列化は C の行（i）単位のみ
//!   分割し、各出力要素の累積順序をスレッド数非依存にする（BLIS 型
//!   packing・NT/TN fast path は対象外。f64 に性能目標がないため）
//! - `sum`（全縮約）: `CHUNK` 単位の `par_chunks` でチャンク内逐次・
//!   チャンク間固定順序結合（`crate::reduction::sum_slice` と同型の決定性
//!   契約）。アキュムレータは f64（出力 dtype と同一のため downcast なし）
//! - `sum`（軸指定）: 出力要素側のみ並列・縮約軸は昇順逐次
//! - `max`: 単位元 `f64::NEG_INFINITY`・`f64::max`（NaN 非伝播は f32 版と
//!   同じ既知事項。スコープ外）。空縮約は
//!   `BackendError::KernelLaunchFailed`（`sum` の空縮約は `0.0`）
//! - `relu`/`exp`/`tanh`: `x.max(0.0)`／`f64::exp`／`f64::tanh`
//!
//! tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）は
//! 変更しない（`crate::parity`。ユーザー承認範囲外）。

use rayon::prelude::*;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    Element, ShapeError, Tensor, TypedOps, elementwise_out_shape, gemm_out_shape, reduce_out_shape,
};

use crate::elementwise::{PARALLEL_THRESHOLD, increment_index};
use crate::ops::CpuBackendOps;
use crate::reduction::{CHUNK, ReduceError, checked_product, unravel};

// ---------------------------------------------------------------------
// gemm
// ---------------------------------------------------------------------

/// `m*k`／`k*n`／`m*n` を `checked_mul` で算出し、オーバーフローと
/// スライス長不整合を本体アクセス前に拒否する（`crate::gemm::validate_dims`
/// の f64 版。`validate_dims` 自体は `&[f32]` 固定のため呼べず、同じ検査を
/// f64 向けに複製する。OWASP A03・`.claude/rules/security.md`）。
fn validate_gemm_dims_f64(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError> {
    let overflow = || BackendError::KernelLaunchFailed("m*k, k*n or m*n overflows usize".into());
    let mk = m.checked_mul(k).ok_or_else(overflow)?;
    let kn = k.checked_mul(n).ok_or_else(overflow)?;
    let mn = m.checked_mul(n).ok_or_else(overflow)?;
    if a.len() != mk {
        return Err(BackendError::KernelLaunchFailed(format!(
            "a length mismatch: expected {mk}, actual {}",
            a.len()
        )));
    }
    if b.len() != kn {
        return Err(BackendError::KernelLaunchFailed(format!(
            "b length mismatch: expected {kn}, actual {}",
            b.len()
        )));
    }
    if c.len() != mn {
        return Err(BackendError::KernelLaunchFailed(format!(
            "c length mismatch: expected {mn}, actual {}",
            c.len()
        )));
    }
    Ok(())
}

/// `Tensor::contiguous()` 実体化後もなお `as_slice()` が `None` を返す
/// （契約上到達しないはずだが、`Tensor` 実装のバグに対する fail-safe）
/// 場合の変換ヘルパー。`crate::ops::gemm_contiguity_fail_safe` と同じ
/// 設計判断（`pub(crate)` ではないためここに複製する）。
fn contiguity_fail_safe(msg: impl std::fmt::Display) -> BackendError {
    BackendError::KernelLaunchFailed(msg.to_string())
}

/// `C の行 i` 単位で rayon 並列化する f64 GEMM 本体（`crate::parity::
/// matmul_reference_fma_f64` と bit 完全一致する契約。各行の出力要素の
/// 累積順序は逐次時と同一のため、並列度に関わらず決定的）。
///
/// `c` はゼロ初期化済みの前提（呼び出し元 [`gemm_f64`] が `vec![0.0f64; m*n]`
/// を渡す）。
fn gemm_row_parallel_f64(a: &[f64], b: &[f64], c: &mut [f64], n: usize, k: usize) {
    // `n == 0`（出力列数 0）の場合、`rayon::par_chunks_mut(0)` は
    // `chunk_size must not be zero` で panic する。`c` はこのとき必ず
    // 空スライス（`m * 0 == 0`）であり書き込むべき要素も存在しないため、
    // ここで早期リターンして panic を回避する（`BackendOps::gemm`
    // 〈f32 版〉が `n == 0` を `Ok([m, 0])` として正しく扱う契約・
    // `crate::parity::matmul_reference_fma_f64` が `n == 0` を単に
    // 0 回ループとして扱う契約と揃える。イシュー #1697 レビュー指摘）。
    if n == 0 {
        return;
    }
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k..i * k + k];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b[p * n..p * n + n];
            for j in 0..n {
                c_row[j] = a_ip.mul_add(b_row[j], c_row[j]);
            }
        }
    });
}

/// 行列積（`m×k` × `k×n` → `m×n`）の f64 版（[`TypedOps::gemm`]）。
pub(crate) fn gemm_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
    let out_shape = gemm_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
    let a_c = a.contiguous();
    let b_c = b.contiguous();
    let a_slice = a_c
        .as_slice()
        .ok_or_else(|| contiguity_fail_safe("gemm_f64: a not contiguous after contiguous()"))?;
    let b_slice = b_c
        .as_slice()
        .ok_or_else(|| contiguity_fail_safe("gemm_f64: b not contiguous after contiguous()"))?;

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let mn = m
        .checked_mul(n)
        .ok_or_else(|| BackendError::KernelLaunchFailed("m*n overflows usize".into()))?;
    let mut out = vec![0.0f64; mn];
    // `out` をゼロ確保した後にまとめて a/b/c 3 者の長さ整合を検査する
    // （`crate::gemm::validate_dims` と同じ「本体アクセス前に検査」契約。
    // 事前に a/b だけを検査してから改めて c 込みで検査する二度手間を避ける）。
    validate_gemm_dims_f64(a_slice, b_slice, &out, m, n, k)?;
    gemm_row_parallel_f64(a_slice, b_slice, &mut out, n, k);
    Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
}

// ---------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------

/// [`crate::elementwise::ElementwiseReadOperand`] の f64 版（イシュー
/// #1697。同型の非負 stride 読み抽象。`crate::elementwise` 側は f32 固定の
/// ため再利用できず、同じ設計を複製する）。
enum ReadOperandF64<'a> {
    Contig(&'a [f64]),
    View {
        span: &'a [f64],
        strides: Vec<usize>,
    },
}

impl<'a> ReadOperandF64<'a> {
    fn classify(t: &'a Tensor<f64>) -> Option<Self> {
        if let Some(s) = t.as_slice() {
            return Some(Self::Contig(s));
        }
        let span = t.as_view_slice()?;
        let strides: Vec<usize> = t
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).ok())
            .collect::<Option<_>>()?;
        Some(Self::View { span, strides })
    }

    #[inline]
    fn read(&self, idx: &[usize], flat: usize) -> Option<f64> {
        match self {
            Self::Contig(s) => s.get(flat).copied(),
            Self::View { span, strides } => {
                let mut off = 0usize;
                for (&i, &st) in idx.iter().zip(strides.iter()) {
                    off = off.checked_add(i.checked_mul(st)?)?;
                }
                span.get(off).copied()
            }
        }
    }
}

fn add_slice_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = x + y);
    } else {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x + y;
        }
    }
}

fn mul_slice_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .for_each(|((o, &x), &y)| *o = x * y);
    } else {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            *o = x * y;
        }
    }
}

fn relu_slice_f64(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.max(0.0));
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.max(0.0);
        }
    }
}

fn exp_slice_f64(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.exp());
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.exp();
        }
    }
}

fn tanh_slice_f64(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    if a.len() >= PARALLEL_THRESHOLD {
        out.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(o, &x)| *o = x.tanh());
    } else {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = x.tanh();
        }
    }
}

/// 二項 elementwise 演算の Tensor 入口共通処理の f64 版
/// （`crate::elementwise::binary_elementwise` を鏡写し）。
fn binary_elementwise_f64(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    slice_kernel: fn(&[f64], &[f64], &mut [f64]),
    scalar_kernel: fn(f64, f64) -> f64,
) -> Result<Tensor<f64>, ShapeError> {
    let out_shape = elementwise_out_shape(a.shape(), b.shape())?;
    let ba = a.broadcast_to(&out_shape)?;
    let bb = b.broadcast_to(&out_shape)?;

    if let (Some(sa), Some(sb)) = (ba.as_slice(), bb.as_slice()) {
        let mut out = vec![0.0f64; sa.len()];
        slice_kernel(sa, sb, &mut out);
        return Tensor::new(out, &out_shape);
    }

    let numel = out_shape.iter().product::<usize>();
    let mut index = vec![0usize; out_shape.len()];

    if let (Some(a_op), Some(b_op)) = (ReadOperandF64::classify(&ba), ReadOperandF64::classify(&bb))
    {
        let mut out = Vec::with_capacity(numel);
        for flat in 0..numel {
            let x = a_op.read(&index, flat);
            let y = b_op.read(&index, flat);
            debug_assert!(
                x.is_some() && y.is_some(),
                "binary_elementwise_f64: as_view_slice の span 保証が破れ、境界外アクセスを検知した (index {index:?})"
            );
            out.push(scalar_kernel(
                x.unwrap_or_else(Element::zero),
                y.unwrap_or_else(Element::zero),
            ));
            increment_index(&mut index, &out_shape);
        }
        return Tensor::new(out, &out_shape);
    }

    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let x = ba.get(&index);
        let y = bb.get(&index);
        debug_assert!(
            x.is_some() && y.is_some(),
            "binary_elementwise_f64: shape 走査ロジックのバグにより index {index:?} が範囲外になった"
        );
        out.push(scalar_kernel(
            x.unwrap_or_else(Element::zero),
            y.unwrap_or_else(Element::zero),
        ));
        increment_index(&mut index, &out_shape);
    }
    Tensor::new(out, &out_shape)
}

/// 単項 elementwise 演算（活性化）の Tensor 入口共通処理の f64 版。
fn unary_elementwise_f64(
    a: &Tensor<f64>,
    slice_kernel: fn(&[f64], &mut [f64]),
    scalar_kernel: fn(f64) -> f64,
) -> Result<Tensor<f64>, ShapeError> {
    if let Some(sa) = a.as_slice() {
        let mut out = vec![0.0f64; sa.len()];
        slice_kernel(sa, &mut out);
        return Tensor::new(out, a.shape());
    }

    let shape = a.shape();
    let numel = a.numel();
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        let x = a.get(&index);
        debug_assert!(
            x.is_some(),
            "unary_elementwise_f64: shape 走査ロジックのバグにより index {index:?} が範囲外になった"
        );
        out.push(scalar_kernel(x.unwrap_or_else(Element::zero)));
        increment_index(&mut index, shape);
    }
    Tensor::new(out, shape)
}

pub(crate) fn add_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
    binary_elementwise_f64(a, b, add_slice_f64, |x, y| x + y).map_err(BackendError::ShapeMismatch)
}

pub(crate) fn mul_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
    binary_elementwise_f64(a, b, mul_slice_f64, |x, y| x * y).map_err(BackendError::ShapeMismatch)
}

pub(crate) fn relu_f64(a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
    unary_elementwise_f64(a, relu_slice_f64, |x: f64| x.max(0.0))
        .map_err(BackendError::ShapeMismatch)
}

pub(crate) fn exp_f64(a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
    unary_elementwise_f64(a, exp_slice_f64, f64::exp).map_err(BackendError::ShapeMismatch)
}

pub(crate) fn tanh_f64(a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
    unary_elementwise_f64(a, tanh_slice_f64, f64::tanh).map_err(BackendError::ShapeMismatch)
}

// ---------------------------------------------------------------------
// reduction
// ---------------------------------------------------------------------

/// 非 contiguous な入力を行優先順で走査し `Vec<f64>` へ収集する
/// （`crate::reduction::gather_elements` の f64 版）。
fn gather_elements_f64(a: &Tensor<f64>) -> Vec<f64> {
    let shape = a.shape();
    let numel = a.numel();
    let mut out = Vec::with_capacity(numel);
    for flat in 0..numel {
        let idx = unravel(flat, shape);
        let value = a.get(&idx);
        debug_assert!(
            value.is_some(),
            "gather_elements_f64: 走査ロジックのバグにより index {idx:?} が範囲外になった"
        );
        out.push(value.unwrap_or(0.0));
    }
    out
}

/// `data` を [`CHUNK`] 単位に分割し、決定性契約に従って `sum` を計算する
/// （`crate::reduction::sum_slice` の f64 版。出力 dtype 自体が f64 のため
/// アキュムレータの downcast は発生しない）。
fn sum_slice_f64(data: &[f64]) -> f64 {
    data.par_chunks(CHUNK)
        .map(|chunk| chunk.iter().fold(0.0f64, |acc, &v| acc + v))
        .collect::<Vec<f64>>()
        .into_iter()
        .fold(0.0f64, |acc, v| acc + v)
}

/// `data` を [`CHUNK`] 単位に分割し、決定性契約に従って `max` を計算する
/// （`crate::reduction::max_slice` の f64 版）。
fn max_slice_f64(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let result = data
        .par_chunks(CHUNK)
        .map(|chunk| chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect::<Vec<f64>>()
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    Some(result)
}

/// 軸指定 reduction の出力要素ごとの畳み込みを行う共通駆動関数の f64 版
/// （`crate::reduction::axis_reduce` を鏡写し。`sum`／`max` 双方が使う）。
fn axis_reduce_f64<F>(a: &Tensor<f64>, axis: usize, identity: f64, op: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    let compute = |flat: usize| -> f64 {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut acc = identity;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx);
            debug_assert!(
                value.is_some(),
                "axis_reduce_f64: 走査ロジックのバグにより index {full_idx:?} が範囲外になった"
            );
            acc = op(acc, value.unwrap_or(identity));
        }
        acc
    };

    (0..total_out).into_par_iter().map(compute).collect()
}

pub(crate) fn sum_f64(a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let data = match dim {
        None => {
            let total = match a.as_slice() {
                Some(slice) => sum_slice_f64(slice),
                None => sum_slice_f64(&gather_elements_f64(a)),
            };
            vec![total]
        }
        Some(axis) => {
            let shape = a.shape();
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            axis_reduce_f64(a, axis, 0.0, |acc, v| acc + v)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

pub(crate) fn max_f64(a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let data = match dim {
        None => {
            let result = match a.as_slice() {
                Some(slice) => max_slice_f64(slice),
                None => max_slice_f64(&gather_elements_f64(a)),
            };
            match result {
                Some(v) => vec![v],
                None => return Err(ReduceError::EmptyReduction { op: "max" }),
            }
        }
        Some(axis) => {
            let shape = a.shape();
            let axis_len = shape[axis];
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            let total_out = outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            if axis_len == 0 && total_out > 0 {
                return Err(ReduceError::EmptyReduction { op: "max" });
            }
            axis_reduce_f64(a, axis, f64::NEG_INFINITY, f64::max)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

// ---------------------------------------------------------------------
// TypedOps<f64> 実装
// ---------------------------------------------------------------------

impl TypedOps<f64> for CpuBackendOps {
    fn gemm(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        gemm_f64(a, b)
    }

    fn add(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        add_f64(a, b)
    }

    fn mul(&self, a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        mul_f64(a, b)
    }

    fn relu(&self, a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        relu_f64(a)
    }

    fn exp(&self, a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        exp_f64(a)
    }

    fn tanh(&self, a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
        tanh_f64(a)
    }

    fn sum(&self, a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        sum_f64(a, dim).map_err(crate::ops::reduce_error_to_backend_error)
    }

    fn max(&self, a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
        max_f64(a, dim).map_err(crate::ops::reduce_error_to_backend_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_f64_matches_hand_computed_2x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = gemm_f64(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_f64_rejects_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).unwrap();
        // a: [1,3], b: [2,1] -> k mismatch (3 != 2)。
        let err = gemm_f64(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn gemm_f64_handles_zero_output_columns() {
        // n == 0 (b: [k, 0]) の場合、rayon `par_chunks_mut(0)` の panic
        // （`chunk_size must not be zero`）を回避しつつ `Ok([m, 0])` を
        // 返すことを確認する（`BackendOps::gemm`〈f32 版〉と揃える契約。
        // イシュー #1697 レビュー指摘）。
        let a = Tensor::new(vec![1.0; 6], &[2, 3]).unwrap();
        let b = Tensor::new(vec![], &[3, 0]).unwrap();
        let c = gemm_f64(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 0]);
        assert_eq!(c.as_slice().unwrap(), &[] as &[f64]);
    }

    #[test]
    fn gemm_f64_handles_zero_output_rows() {
        // m == 0 の場合も同様に empty 出力を返すことを確認する。
        let a = Tensor::new(vec![], &[0, 3]).unwrap();
        let b = Tensor::new(vec![1.0; 6], &[3, 2]).unwrap();
        let c = gemm_f64(&a, &b).unwrap();
        assert_eq!(c.shape(), &[0, 2]);
        assert_eq!(c.as_slice().unwrap(), &[] as &[f64]);
    }

    #[test]
    fn gemm_f64_handles_zero_contraction_dim() {
        // k == 0 の場合は出力が全てゼロ（contraction 次元が空のため）。
        let a = Tensor::new(vec![], &[2, 0]).unwrap();
        let b = Tensor::new(vec![], &[0, 3]).unwrap();
        let c = gemm_f64(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_slice().unwrap(), &[0.0; 6]);
    }

    #[test]
    fn add_f64_broadcasts() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![10.0, 20.0], &[2]).unwrap();
        let c = add_f64(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn mul_f64_elementwise() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = mul_f64(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn relu_f64_clamps_negative() {
        let a = Tensor::new(vec![-1.0, 0.0, 2.0], &[3]).unwrap();
        let c = relu_f64(&a).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[0.0, 0.0, 2.0]);
    }

    #[test]
    fn exp_f64_matches_std() {
        let a = Tensor::new(vec![0.0, 1.0], &[2]).unwrap();
        let c = exp_f64(&a).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[0.0f64.exp(), 1.0f64.exp()]);
    }

    #[test]
    fn tanh_f64_matches_std() {
        let a = Tensor::new(vec![0.0, 1.0], &[2]).unwrap();
        let c = tanh_f64(&a).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[0.0f64.tanh(), 1.0f64.tanh()]);
    }

    #[test]
    fn sum_f64_full_reduction() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = sum_f64(&a, None).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[10.0]);
    }

    #[test]
    fn sum_f64_axis_reduction() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = sum_f64(&a, Some(0)).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn sum_f64_empty_reduction_is_zero() {
        let a = Tensor::new(Vec::<f64>::new(), &[0, 3]).unwrap();
        let c = sum_f64(&a, None).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[0.0]);
    }

    #[test]
    fn max_f64_full_reduction() {
        let a = Tensor::new(vec![1.0, 5.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = max_f64(&a, None).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0]);
    }

    #[test]
    fn max_f64_empty_reduction_is_error() {
        let a = Tensor::new(Vec::<f64>::new(), &[0, 3]).unwrap();
        let err = max_f64(&a, None).unwrap_err();
        assert!(matches!(err, ReduceError::EmptyReduction { op: "max" }));
    }

    #[test]
    fn typed_ops_gemm_dispatches_through_trait() {
        let ops = CpuBackendOps::new();
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = TypedOps::<f64>::gemm(&ops, &a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
    }
}
