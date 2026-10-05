//! `histc`／`bincount`／`searchsorted`／`bucketize` のホスト参照カーネルの
//! **単一情報源**（イシュー #2638・親 #2625。実装記録は
//! `docs/autodiff-binning-ops-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::binning_ops`（`BackendOps` が `Unsupported` のときのホスト
//!   フォールバック）と `backend-cpu` の `CpuBackendOps::binning_*` がいずれも本
//!   モジュールの関数を直接呼ぶ。範囲決定・ビン添字の算術・探索手順・アキュムレータ
//!   契約をクレート間で複製せず、乖離を構造的に排除する（`stat_reduce.rs`・
//!   `cumulative.rs` と同じ「共有可能な層に一本化する」方式）。
//! - 4 演算はすべて**非微分**（出力は整数のカウント・索引、または勾配を持たない
//!   ヒストグラム）で、VJP・tape ノードを持たない（決定記録 §5）。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の
//!   `Unsupported` から autodiff 側がこのホスト実装へフォールバックする。
//!
//! # 数値契約
//!
//! - `histc` は ATen（`aten/src/ATen/native/cpu/HistogramKernel.cpp`・`histc` の
//!   `LINEAR_INTERPOLATION`。局所探索なし）と同じ手順を **`f32` のまま同じ演算順**で
//!   再現する: ① 範囲決定（`min == max` なら入力の最小・最大、なお等しければ ±1。`f64`）→
//!   ② 左右端を `torch.linspace` の端点と同じ `f32` 算術で確定（`start + step * 0`・
//!   `end - step * 0`。幅が `f32` で溢れると `step = inf` で両端が NaN になり全要素が
//!   無視される。PyTorch の実測と一致）→ ③ 要素ごとに
//!   `trunc((x - left) * bins / (right - left))` でビンを決める → ④ `x == right` は最終ビン。
//!   `f64` で添字を計算すると境界要素のビンがずれる。`mul_add` は使わず matmul 系 FMA 契約には
//!   触れない。カウントは `u64` で数え最後に 1 回だけ `f32` へ変換する。
//! - `bincount`（重みなし）は整数カウント。重みあり版は **`f64` アキュムレータ**へ入力順に
//!   加算して最後に 1 回 `f32` へ downcast する（長軸縮約の `f64` 契約。PyTorch は `f32`
//!   逐次加算の見込みで、意図的な差。比較は REQ-2 統一複合判定）。非有限重みは伝播する。
//! - `searchsorted`／`bucketize` は PyTorch の二分探索手順（`cus_lower_bound`／
//!   `cus_upper_bound`: 中点 `start + (end - start) / 2`・`!(mid >= v)`／`!(mid > v)` で右へ）を
//!   そのまま再現する。重複・NaN・未ソート列でも結果が PyTorch と一致し、結果は常に
//!   `[0, len]` 内に収まる（未ソート列は検査しない）。
//!
//! # 境界検査（REQ-8・OWASP A03／A04）
//!
//! 形状・要素数・バイト数は `checked_mul`／`isize::MAX` で確保前に検査する。出力長が入力値
//! や引数に依存する箇所（`bincount` の最大値・`minlength`、`histc` の `bins`）は、全要素の
//! 検証 → `checked_add`／`usize::try_from` → バイト数検査 → `Vec::try_reserve_exact` の順で、
//! 確保失敗を abort ではなく型付きエラーにする。索引は `i32::try_from`（無検査 `as` を
//! 使わない）。`unsafe`／`get_unchecked`／本番経路の `unwrap`／`expect` は使わない。

use std::cmp::Ordering;

use crate::device::BackendError;
use crate::error::ShapeError;

/// 形状・引数検査の失敗（型付き）。`autodiff` は `AutodiffError` へ、バックエンドは
/// [`BackendError`] へ写像する（`StatReduceError` と同じ置き方）。
#[derive(Debug, Clone, PartialEq)]
pub enum BinningError {
    /// rank 不整合・要素数／バイト数オーバーフロー・索引幅超過等の形状違反。
    Shape(ShapeError),
    /// `bins`・範囲・負の索引・重み長不一致など、引数の不正。
    InvalidArgument(String),
}

impl std::fmt::Display for BinningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinningError::Shape(e) => write!(f, "binning: shape error: {e:?}"),
            BinningError::InvalidArgument(msg) => write!(f, "binning: invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for BinningError {}

impl From<ShapeError> for BinningError {
    fn from(err: ShapeError) -> Self {
        BinningError::Shape(err)
    }
}

impl From<BinningError> for BackendError {
    fn from(err: BinningError) -> Self {
        match err {
            BinningError::Shape(e) => BackendError::ShapeMismatch(e),
            BinningError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
        }
    }
}

fn invalid(msg: impl Into<String>) -> BinningError {
    BinningError::InvalidArgument(msg.into())
}

/// `count * elem_bytes` の `checked_mul` と `isize::MAX` 上限を検査する。
fn check_alloc_bytes(count: usize, elem_bytes: usize) -> Result<(), ShapeError> {
    let bytes = count
        .checked_mul(elem_bytes)
        .ok_or(ShapeError::ElementCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(())
}

/// shape の要素数積（`checked_mul`）を返す。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d).ok_or(ShapeError::ElementCountOverflow)
    })
}

/// shape の要素数を求め、`f32` 実体のバイト数が `isize::MAX` を超えないことを検査する。
fn checked_numel_f32(shape: &[usize]) -> Result<usize, ShapeError> {
    let n = checked_numel(shape)?;
    check_alloc_bytes(n, std::mem::size_of::<f32>())?;
    Ok(n)
}

/// 長さ `len`・値 `fill` の `Vec` を、確保失敗を型付きエラーにして作る
/// （巨大な `bins`／`minlength` で abort しない。OWASP A04）。
fn try_filled<T: Clone>(len: usize, fill: T) -> Result<Vec<T>, BinningError> {
    check_alloc_bytes(len, std::mem::size_of::<T>())?;
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| BinningError::Shape(ShapeError::ElementCountOverflow))?;
    v.resize(len, fill);
    Ok(v)
}

// ---------------------------------------------------------------------------
// histc
// ---------------------------------------------------------------------------

/// `histc` の引数・形状の事前検査（実体化・確保より前。autodiff／CPU 実装が呼ぶ）。
///
/// `bins == 0`・`min > max`・要素数／バイト数オーバーフロー・出力・カウントの確保サイズ
/// 超過は型付きエラー。範囲の有限性は `min == max` のとき入力の最小・最大を採るため
/// データ依存で、[`histc_host`] が検査する。
pub fn histc_check(shape: &[usize], bins: usize, min: f32, max: f32) -> Result<(), BinningError> {
    checked_numel_f32(shape)?;
    if bins == 0 {
        return Err(invalid("histc: bins must be > 0"));
    }
    if f64::from(min) > f64::from(max) {
        return Err(invalid(format!(
            "histc: max must be larger than min (min={min}, max={max})"
        )));
    }
    // 出力 `f32` とカウント用 `u64` バッファ。
    check_alloc_bytes(bins, std::mem::size_of::<f32>())?;
    check_alloc_bytes(bins, std::mem::size_of::<u64>())?;
    Ok(())
}

/// `torch.linspace(start, end, bins + 1)` の両端点（`f32`）。ATen は線形補間でビンを決める
/// 際の左右端をこのエッジ列の先頭・末尾から取るため、同じ算術（前半の先頭は
/// `start + step * 0`・後半の末尾は `end - step * 0`）で再現する。
fn linspace_ends(start: f32, end: f32, bins: usize) -> (f32, f32) {
    // `bins` は整数のまま `f32` へ昇格して割る（ATen: `step_t / int64_t`）。
    let step = (end - start) / (bins as f32);
    (start + step * 0.0, end - step * 0.0)
}

/// 非負の有限 `f32` を 0 方向へ切り捨てて `i64` へ変換する。NaN・負・`i64` 範囲外は `None`。
fn trunc_to_i64(v: f32) -> Option<i64> {
    // `2^63` は `f32` で正確に表せる。範囲内に限り `as` は切り捨て変換で情報落ちを起こさない。
    if (0.0..9.223_372e18).contains(&v) {
        Some(v as i64)
    } else {
        None
    }
}

/// `torch.histc(x, bins, min, max)` のホスト参照実装。入力は平坦化され、出力は長さ `bins` の
/// カウント（`f32`）。
///
/// 範囲外・NaN 要素は無視する（`x == right` は最終ビン）。範囲が非有限なら型付きエラー
/// （PyTorch と同じ）。範囲の幅が `f32` で溢れてエッジが NaN になる入力は、PyTorch と同じく
/// 全要素が無視され全ビン 0 になる。
pub fn histc_host(x: &[f32], bins: usize, min: f32, max: f32) -> Result<Vec<f32>, BinningError> {
    histc_check(&[x.len()], bins, min, max)?;
    let (mut left, mut right) = (f64::from(min), f64::from(max));
    if left == right && !x.is_empty() {
        // `aminmax`: NaN を含めば最小・最大とも NaN。
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut has_nan = false;
        for &v in x {
            if v.is_nan() {
                has_nan = true;
            } else {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if has_nan {
            left = f64::NAN;
            right = f64::NAN;
        } else {
            left = f64::from(lo);
            right = f64::from(hi);
        }
    }
    if left == right {
        left -= 1.0;
        right += 1.0;
    }
    if !left.is_finite() || !right.is_finite() {
        return Err(invalid(format!(
            "histc: range of [{left}, {right}] is not finite"
        )));
    }
    // 以降は `f32`（ATen は入力 dtype で計算する）。`f64 -> f32` の `as` は丸め変換。
    let (lo, hi) = linspace_ends(left as f32, right as f32, bins);
    let bins_f = bins as f32;
    let mut counts = try_filled(bins, 0u64)?;
    for &v in x {
        // NaN 要素・範囲外は比較が偽になり無視される。
        if !(v >= lo && v <= hi) {
            continue;
        }
        let Some(pos) = trunc_to_i64(((v - lo) * bins_f) / (hi - lo)) else {
            continue;
        };
        let Ok(mut p) = usize::try_from(pos) else {
            continue;
        };
        // 最右ビンだけ右端を含む。
        if p == bins {
            p -= 1;
        }
        if let Some(c) = counts.get_mut(p) {
            *c += 1;
        }
    }
    let mut out = try_filled(bins, 0.0f32)?;
    for (o, &c) in out.iter_mut().zip(counts.iter()) {
        *o = c as f32;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// bincount
// ---------------------------------------------------------------------------

/// `bincount` の入力 shape 検査（rank 1 のみ）。重みなし版のカウントは `i32` なので入力長も
/// `i32::MAX` 以下に制限する（`weighted = false`）。
pub fn bincount_check(shape: &[usize], weighted: bool) -> Result<usize, BinningError> {
    if shape.len() != 1 {
        return Err(invalid(
            "bincount: only 1-d non-negative integral inputs are supported",
        ));
    }
    let n = checked_numel(shape)?;
    check_alloc_bytes(n, std::mem::size_of::<i32>())?;
    if !weighted && i32::try_from(n).is_err() {
        return Err(BinningError::Shape(ShapeError::IndexRangeOverflow {
            index: n,
        }));
    }
    Ok(n)
}

/// 全要素を検証して出力長 `max(max + 1, minlength)` を返す（確保より前。負値は拒否）。
///
/// 公開 API 層（`autodiff::binning_ops`）がバックエンド呼び出し前の負値拒否と、バックエンド
/// 出力長の完全一致検証（契約どおりの長さ）に使う。
pub fn bincount_out_len(input: &[i32], minlength: usize) -> Result<usize, BinningError> {
    let mut max_v: Option<i32> = None;
    for &v in input {
        if v < 0 {
            return Err(invalid(format!("bincount: negative input {v}")));
        }
        max_v = Some(max_v.map_or(v, |m| m.max(v)));
    }
    let needed = match max_v {
        None => 0,
        Some(m) => usize::try_from(m)
            .ok()
            .and_then(|m| m.checked_add(1))
            .ok_or(ShapeError::ElementCountOverflow)?,
    };
    let out_len = needed.max(minlength);
    // 出力（i32／f32 とも 4 バイト）のバイト数を確保前に検査する。巨大 `minlength` は
    // バックエンド呼び出し前に型付きエラーへ変換する（OWASP A04）。
    check_alloc_bytes(out_len, std::mem::size_of::<f32>())?;
    Ok(out_len)
}

/// 長さ `len` の零 `f32` ベクタを、バイト数検査と `try_reserve_exact` を通して作る
/// （空入力の重み付き `bincount` が巨大 `minlength` で abort しないための確保口）。
pub fn bincount_zeros_f32(len: usize) -> Result<Vec<f32>, BinningError> {
    try_filled(len, 0.0f32)
}

/// 重み付き `bincount` の重み shape 検査。rank 1 で入力と同長であること。
/// 呼び出し側は空入力のとき本検査を行わない（PyTorch と同じく重みを見ない）。
pub fn bincount_weights_check(
    input_len: usize,
    weights_shape: &[usize],
) -> Result<(), BinningError> {
    if weights_shape.len() != 1 || weights_shape[0] != input_len {
        return Err(invalid(
            "bincount: weights should be 1-d and have the same length as input",
        ));
    }
    Ok(())
}

/// 重みなし `torch.bincount` のホスト参照実装。出力長 `max(max(input) + 1, minlength)`・
/// 要素は `i32` のカウント。空入力は長さ `minlength` の零。
pub fn bincount_host(input: &[i32], minlength: usize) -> Result<Vec<i32>, BinningError> {
    if i32::try_from(input.len()).is_err() {
        return Err(BinningError::Shape(ShapeError::IndexRangeOverflow {
            index: input.len(),
        }));
    }
    let out_len = bincount_out_len(input, minlength)?;
    let mut out = try_filled(out_len, 0i32)?;
    for &v in input {
        let idx = usize::try_from(v).map_err(|_| invalid("bincount: negative input"))?;
        let slot = out
            .get_mut(idx)
            .ok_or_else(|| invalid("bincount: index out of output range"))?;
        *slot = slot
            .checked_add(1)
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    Ok(out)
}

/// 重みあり `torch.bincount` のホスト参照実装。ビンごとに `f64` アキュムレータへ入力順に
/// 加算し、最後に 1 回 `f32` へ downcast する。`weights` は入力と同長（空入力は PyTorch と
/// 同じく重みを見ずに長さ `minlength` の零を返す）。
pub fn bincount_weighted_host(
    input: &[i32],
    weights: &[f32],
    minlength: usize,
) -> Result<Vec<f32>, BinningError> {
    if !input.is_empty() && weights.len() != input.len() {
        return Err(invalid(
            "bincount: weights should be 1-d and have the same length as input",
        ));
    }
    let out_len = bincount_out_len(input, minlength)?;
    let mut acc = try_filled(out_len, 0.0f64)?;
    for (&v, &w) in input.iter().zip(weights.iter()) {
        let idx = usize::try_from(v).map_err(|_| invalid("bincount: negative input"))?;
        let slot = acc
            .get_mut(idx)
            .ok_or_else(|| invalid("bincount: index out of output range"))?;
        *slot += f64::from(w);
    }
    let mut out = try_filled(out_len, 0.0f32)?;
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = a as f32;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// searchsorted / bucketize
// ---------------------------------------------------------------------------

/// 解決済みの `searchsorted` レイアウト（[`searchsorted_layout`] の戻り値）。
///
/// `sorted_sequence` を最内軸 1 本ずつの lane（`batch` 本・各 `seq_len` 要素）、`values` を
/// lane ごとに `per_batch` 個として扱う。フィールドは非公開で、
/// [`searchsorted_layout`] だけが生成する（REQ-8。公開カーネルが外部から改変された値で
/// 範囲外アクセスしない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchSortedLayout {
    batch: usize,
    seq_len: usize,
    per_batch: usize,
    out_shape: Vec<usize>,
}

impl SearchSortedLayout {
    /// 出力 shape（= `values` の shape）。
    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }

    /// 出力要素数（= `values` の要素数）。
    pub fn out_numel(&self) -> usize {
        self.batch * self.per_batch
    }
}

/// `searchsorted(sorted_sequence, values)` の shape 規則を検査してレイアウトを返す
/// （実体化・確保より前）。
///
/// - `sorted_sequence` は rank ≥ 1。rank 1 なら `values` は任意 shape（0 次元可）。
/// - rank N ≥ 2 なら `values` は同 rank で先頭 N−1 軸が一致する。
/// - 列長は `i32::MAX` 以下（索引を `i32` で返す）。要素数・バイト数は確保前に検査する。
pub fn searchsorted_layout(
    seq_shape: &[usize],
    values_shape: &[usize],
) -> Result<SearchSortedLayout, BinningError> {
    let Some((&seq_len, seq_lead)) = seq_shape.split_last() else {
        return Err(invalid(
            "searchsorted: sorted_sequence must have positive dimension",
        ));
    };
    let (batch, per_batch) = if seq_lead.is_empty() {
        (1, checked_numel(values_shape)?)
    } else {
        let Some((&last, values_lead)) = values_shape.split_last() else {
            return Err(invalid(
                "searchsorted: input value can be a scalar only when sorted_sequence is 1-d",
            ));
        };
        if values_shape.len() != seq_shape.len() || values_lead != seq_lead {
            return Err(invalid(
                "searchsorted: sorted_sequence should be 1-d or the first N-1 dims must match values",
            ));
        }
        (checked_numel(seq_lead)?, last)
    };
    checked_numel_f32(seq_shape)?;
    let out_numel = batch
        .checked_mul(per_batch)
        .ok_or(ShapeError::ElementCountOverflow)?;
    check_alloc_bytes(out_numel, std::mem::size_of::<f32>())?;
    if i32::try_from(seq_len).is_err() {
        return Err(BinningError::Shape(ShapeError::IndexRangeOverflow {
            index: seq_len,
        }));
    }
    Ok(SearchSortedLayout {
        batch,
        seq_len,
        per_batch,
        out_shape: values_shape.to_vec(),
    })
}

/// `bucketize(input, boundaries)` の shape 規則（`boundaries` は 1 次元必須）。
/// 検査後は `searchsorted_layout(boundaries_shape, input_shape)` と同じレイアウトを返す。
pub fn bucketize_layout(
    boundaries_shape: &[usize],
    input_shape: &[usize],
) -> Result<SearchSortedLayout, BinningError> {
    if boundaries_shape.len() != 1 {
        return Err(invalid(format!(
            "bucketize: boundaries tensor must be 1 dimension, but got dim({})",
            boundaries_shape.len()
        )));
    }
    searchsorted_layout(boundaries_shape, input_shape)
}

/// 1 lane の二分探索（PyTorch `cus_lower_bound`／`cus_upper_bound` と同じ手順）。
fn search_lane(seq: &[f32], v: f32, right: bool) -> usize {
    let (mut start, mut end) = (0usize, seq.len());
    while start < end {
        let mid = start + ((end - start) >> 1);
        let mid_val = seq[mid];
        // PyTorch は `!(mid_val > v)`／`!(mid_val >= v)` で右へ進む。NaN（非順序）も右へ進む
        // ため `partial_cmp` の `None` を明示的に含める。
        let go_right = match mid_val.partial_cmp(&v) {
            None | Some(Ordering::Less) => true,
            Some(Ordering::Equal) => right,
            Some(Ordering::Greater) => false,
        };
        if go_right {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    start
}

/// `torch.searchsorted`／`torch.bucketize` のホスト参照実装。`right = false` は下限、
/// `true` は上限。出力は `values` と同順の `i32` 索引（`layout.out_shape()` の shape）。
///
/// 入力スライス長は `layout` と再照合する（直接呼び出しでも範囲外アクセスしない）。
pub fn searchsorted_host(
    seq: &[f32],
    values: &[f32],
    layout: &SearchSortedLayout,
    right: bool,
) -> Result<Vec<i32>, BinningError> {
    let seq_total = layout
        .batch
        .checked_mul(layout.seq_len)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let values_total = layout.out_numel();
    if seq.len() != seq_total {
        return Err(BinningError::Shape(ShapeError::ElementCountMismatch {
            expected: seq_total,
            actual: seq.len(),
        }));
    }
    if values.len() != values_total {
        return Err(BinningError::Shape(ShapeError::ElementCountMismatch {
            expected: values_total,
            actual: values.len(),
        }));
    }
    let mut out = try_filled(values_total, 0i32)?;
    if layout.per_batch == 0 || layout.batch == 0 {
        return Ok(out);
    }
    for b in 0..layout.batch {
        let lane = &seq[b * layout.seq_len..(b + 1) * layout.seq_len];
        let vals = &values[b * layout.per_batch..(b + 1) * layout.per_batch];
        let outs = &mut out[b * layout.per_batch..(b + 1) * layout.per_batch];
        for (o, &v) in outs.iter_mut().zip(vals.iter()) {
            let idx = search_lane(lane, v, right);
            *o = i32::try_from(idx)
                .map_err(|_| BinningError::Shape(ShapeError::IndexRangeOverflow { index: idx }))?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 素朴な線形走査オラクル（探索手順を共有しない）。昇順列に対する下限／上限。
    fn linear_oracle(seq: &[f32], v: f32, right: bool) -> usize {
        seq.iter()
            .filter(|&&s| if right { s <= v } else { s < v })
            .count()
    }

    #[test]
    fn histc_counts_grid_and_last_bin_includes_right_edge() {
        let x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let h = histc_host(&x, 4, 0.0, 4.0).unwrap();
        assert_eq!(h, vec![2.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn histc_default_range_uses_data_extrema_and_expands_equal() {
        assert_eq!(
            histc_host(&[1.0, 1.0, 1.0], 4, 0.0, 0.0).unwrap(),
            vec![0.0, 0.0, 3.0, 0.0]
        );
        assert_eq!(histc_host(&[], 4, 0.0, 0.0).unwrap(), vec![0.0; 4]);
    }

    #[test]
    fn histc_rejects_bad_arguments() {
        assert!(histc_host(&[1.0], 0, 0.0, 1.0).is_err());
        assert!(histc_host(&[1.0], 2, 2.0, 1.0).is_err());
        assert!(histc_host(&[f32::NAN, 1.0], 2, 0.0, 0.0).is_err());
        assert!(histc_host(&[f32::INFINITY, 1.0], 2, 0.0, 0.0).is_err());
        assert!(histc_host(&[1.0], 2, 0.0, f32::INFINITY).is_err());
        // 巨大 bins は確保前に型付きエラー。
        assert!(histc_check(&[1], usize::MAX, 0.0, 1.0).is_err());
        assert!(histc_host(&[1.0], usize::MAX / 2, 0.0, 1.0).is_err());
    }

    #[test]
    fn histc_ignores_nan_and_out_of_range_with_explicit_range() {
        let h = histc_host(&[0.0, f32::NAN, 2.0, 9.0, -1.0, f32::INFINITY], 4, 0.0, 2.0).unwrap();
        assert_eq!(h, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn histc_total_equals_in_range_count_and_is_deterministic() {
        let x: Vec<f32> = (0..2000)
            .map(|i| ((i * 37 % 1000) as f32) / 100.0)
            .collect();
        let a = histc_host(&x, 13, 1.0, 8.0).unwrap();
        let b = histc_host(&x, 13, 1.0, 8.0).unwrap();
        assert_eq!(a, b);
        let in_range = x.iter().filter(|&&v| (1.0..=8.0).contains(&v)).count();
        assert_eq!(a.iter().sum::<f32>() as usize, in_range);
    }

    #[test]
    fn bincount_basic_minlength_and_empty() {
        assert_eq!(bincount_host(&[0, 1, 1, 3], 0).unwrap(), vec![1, 2, 0, 1]);
        assert_eq!(bincount_host(&[0, 1], 4).unwrap(), vec![1, 1, 0, 0]);
        assert_eq!(bincount_host(&[0, 4], 2).unwrap(), vec![1, 0, 0, 0, 1]);
        assert_eq!(bincount_host(&[], 3).unwrap(), vec![0, 0, 0]);
        assert_eq!(bincount_host(&[], 0).unwrap(), Vec::<i32>::new());
    }

    #[test]
    fn bincount_rejects_negative_rank_and_huge_output() {
        assert!(bincount_host(&[0, -1], 0).is_err());
        assert!(bincount_check(&[2, 2], false).is_err());
        assert!(bincount_check(&[], false).is_err());
        assert!(bincount_host(&[0], usize::MAX).is_err());
        // 巨大な最大値は確保前の検査（または確保失敗）で型付きエラー。
        assert!(bincount_host(&[i32::MAX], usize::MAX / 8).is_err());
    }

    #[test]
    fn bincount_weighted_uses_f64_accumulator_and_checks_length() {
        let out = bincount_weighted_host(&[0, 0, 0], &[1e8, 1.0, -1e8], 0).unwrap();
        assert_eq!(out, vec![1.0]);
        assert!(bincount_weighted_host(&[0, 1], &[1.0], 0).is_err());
        assert_eq!(
            bincount_weighted_host(&[], &[1.0], 2).unwrap(),
            vec![0.0; 2]
        );
        let nf = bincount_weighted_host(&[0, 0], &[f32::INFINITY, f32::NEG_INFINITY], 0).unwrap();
        assert!(nf[0].is_nan());
    }

    #[test]
    fn searchsorted_matches_linear_oracle_on_sorted_input() {
        let seq: Vec<f32> = vec![1.0, 2.0, 2.0, 2.0, 5.0, 7.0];
        let vals: Vec<f32> = vec![0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 8.0];
        let layout = searchsorted_layout(&[6], &[8]).unwrap();
        for right in [false, true] {
            let got = searchsorted_host(&seq, &vals, &layout, right).unwrap();
            let want: Vec<i32> = vals
                .iter()
                .map(|&v| linear_oracle(&seq, v, right) as i32)
                .collect();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn searchsorted_layout_rules() {
        assert!(searchsorted_layout(&[], &[1]).is_err());
        assert!(searchsorted_layout(&[2, 2], &[3, 1]).is_err());
        assert!(searchsorted_layout(&[2, 2], &[2]).is_err());
        assert!(searchsorted_layout(&[2, 2], &[]).is_err());
        assert_eq!(searchsorted_layout(&[3], &[]).unwrap().out_numel(), 1);
        assert_eq!(
            searchsorted_layout(&[2, 3], &[2, 4]).unwrap().out_numel(),
            8
        );
        assert!(bucketize_layout(&[2, 2], &[1]).is_err());
        assert!(bucketize_layout(&[], &[1]).is_err());
        // 巨大 shape は確保前に拒否される。
        assert!(searchsorted_layout(&[1], &[usize::MAX, 2]).is_err());
        assert!(searchsorted_layout(&[1], &[usize::MAX / 2]).is_err());
    }

    #[test]
    fn searchsorted_nan_and_unsorted_stay_in_range_and_batched() {
        let seq = [1.0, f32::NAN, 3.0];
        let layout = searchsorted_layout(&[3], &[3]).unwrap();
        let got = searchsorted_host(&seq, &[f32::NAN, 2.0, 5.0], &layout, false).unwrap();
        assert_eq!(got, vec![3, 2, 3]);
        let unsorted = [3.0, 1.0, 2.0, 0.0, 5.0];
        let l2 = searchsorted_layout(&[5], &[2]).unwrap();
        let g = searchsorted_host(&unsorted, &[0.0, 6.0], &l2, true).unwrap();
        assert!(g.iter().all(|&i| (0..=5).contains(&i)));
        let bl = searchsorted_layout(&[2, 3], &[2, 2]).unwrap();
        let got = searchsorted_host(
            &[1.0, 3.0, 5.0, 2.0, 2.0, 8.0],
            &[3.0, 0.0, 2.0, 9.0],
            &bl,
            false,
        )
        .unwrap();
        assert_eq!(got, vec![1, 0, 0, 3]);
    }

    #[test]
    fn searchsorted_host_rechecks_slice_lengths() {
        let layout = searchsorted_layout(&[3], &[2]).unwrap();
        assert!(searchsorted_host(&[1.0, 2.0], &[1.0, 2.0], &layout, false).is_err());
        assert!(searchsorted_host(&[1.0, 2.0, 3.0], &[1.0], &layout, false).is_err());
    }

    #[test]
    fn searchsorted_empty_cases() {
        let l = searchsorted_layout(&[0], &[2]).unwrap();
        assert_eq!(
            searchsorted_host(&[], &[1.0, 2.0], &l, true).unwrap(),
            vec![0, 0]
        );
        let l = searchsorted_layout(&[3], &[0]).unwrap();
        assert!(
            searchsorted_host(&[1.0, 2.0, 3.0], &[], &l, false)
                .unwrap()
                .is_empty()
        );
    }
}
