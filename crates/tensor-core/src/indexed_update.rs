//! `scatter_reduce` のホスト参照カーネルの**単一情報源**（イシュー #2641・
//! 親 #2625。実装記録は `docs/autodiff-indexed-update-ops-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::indexed_update_ops::scatter_reduce`（`BackendOps::
//!   indexed_scatter_reduce` が `Unsupported` のときのホストフォールバック）・
//!   `autodiff::grad`（`Op::IndexedScatterReduce` の VJP）・`backend-cpu` の
//!   `CpuBackendOps::indexed_scatter_reduce` がいずれも本モジュールの関数を直接
//!   呼ぶ。走査順・タイ／NaN 規則・アキュムレータ契約をクレート間で複製せず、
//!   乖離を構造的に排除する（`cumulative.rs` と同じ方式）。
//! - 既存の [`crate::ScatterReduce`]（`Overwrite`／`Add`）は拡張しない。その全
//!   消費者（`eval::scatter`・CPU／CUDA／Metal の scatter 実装）は未知 variant を
//!   release では黙って `Overwrite` として処理するため、`Prod` 等を足すと CUDA／
//!   Metal 利用者がエラーなしで誤った数値を受け取りうる。本モジュールは別 enum
//!   [`ScatterReduceMode`] と別 trait メソッドで縮約を表す（決定記録 §2）。
//!
//! # 数値契約
//!
//! 「触れられた位置」とは 1 個以上の `src` が書き込まれた出力位置。**触れられて
//! いない位置は入力のビット列をそのまま写す**（アキュムレータを通さない）。
//! `include_self == false` のとき、触れられた位置は入力値を寄与から外す。
//!
//! - `Sum`: `(include_self ? input : +0.0) + Σ src` を位置ごとの `f64` に走査順
//!   （`index`／`src` の行優先）で加算し 1 回だけ `f32` へ downcast。
//!   `include_self == true` は [`crate::ScatterReduce::Add`] と bit 一致する。
//! - `Mean`: 上の和を寄与数（`src` の個数＋`include_self` なら 1）で割る（`f64`、
//!   1 回 downcast）。
//! - `Prod`: `(include_self ? input : 1.0) × Π src` を `f64` で走査順に乗算し
//!   1 回 downcast。`mul_add` は使わない（matmul 系 FMA 契約には触れない）。
//! - `Amax`／`Amin`: 比較と選択のみ（値は寄与のいずれかと bit 一致）。種は
//!   `include_self` なら入力、そうでなければ最初の `src`。以降は**厳密に大きい
//!   （小さい）ときのみ更新（タイは先勝ち）**、NaN は伝播する（PyTorch 2.14.0 の
//!   実測。`tests/fixtures/indexed-update-pytorch-reference/`）。
//!
//! VJP は上流勾配を `f64` で割り戻して各要素 1 回だけ `f32` へ downcast する。
//! `Prod` は `result / 値` を使わず各寄与を除いた積を前置積×後置積で直接求める
//! （0・inf を含む lane でも他寄与の積がそのまま得られ、inf/inf の NaN を生まない）。`Amax`／`Amin` は結果と等しい寄与の
//! 個数で上流勾配を均等に分ける。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状検査は [`scatter_reduce_layout`] に集約し、確保より前に `dim` 範囲・
//! rank・`index`/`src` の shape 一致・要素数とバイト数の `checked_mul` を検査する。
//! カーネルは入力スライス長と `index` の値域（負値を含む）も再検査し、範囲外は
//! 型付きエラー（[`ShapeError`]）で拒否する。`unsafe`／`get_unchecked` は使わない。

use crate::error::ShapeError;
use crate::ops_shape::scatter_out_shape;

/// `scatter_reduce` の縮約種別（`torch.scatter_reduce` の `reduce` 相当）。
///
/// `#[non_exhaustive]`。[`crate::ScatterReduce`] とは別 enum（モジュール doc 参照）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScatterReduceMode {
    /// 総和（`"sum"`）。
    Sum,
    /// 総積（`"prod"`）。
    Prod,
    /// 平均（`"mean"`）。
    Mean,
    /// 最大値（`"amax"`）。
    Amax,
    /// 最小値（`"amin"`）。
    Amin,
}

/// 解決済みの scatter_reduce レイアウト（[`scatter_reduce_layout`] の戻り値）。
///
/// フィールドは非公開で、[`scatter_reduce_layout`] だけが生成する。生成時の検査で
/// 整合性が保証され、公開カーネルが改変された値で範囲外アクセスしないようにする。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScatterReduceLayout {
    dim: usize,
    in_shape: Vec<usize>,
    idx_shape: Vec<usize>,
    in_numel: usize,
    idx_numel: usize,
    in_strides: Vec<usize>,
}

impl ScatterReduceLayout {
    /// 入力（＝出力）の shape。
    pub fn shape(&self) -> &[usize] {
        &self.in_shape
    }

    /// 入力（＝出力）の総要素数。
    pub fn numel(&self) -> usize {
        self.in_numel
    }

    /// `index`／`src` の総要素数。
    pub fn index_numel(&self) -> usize {
        self.idx_numel
    }

    /// 入力・`index`・`src`・上流勾配のスライス長を検査する。
    fn check_lens(
        &self,
        input: Option<usize>,
        index: usize,
        src: Option<usize>,
        upstream: Option<usize>,
    ) -> Result<(), ShapeError> {
        let mismatch =
            |expected: usize, actual: usize| ShapeError::ElementCountMismatch { expected, actual };
        if let Some(n) = input
            && n != self.in_numel
        {
            return Err(mismatch(self.in_numel, n));
        }
        if index != self.idx_numel {
            return Err(mismatch(self.idx_numel, index));
        }
        if let Some(n) = src
            && n != self.idx_numel
        {
            return Err(mismatch(self.idx_numel, n));
        }
        if let Some(n) = upstream
            && n != self.in_numel
        {
            return Err(mismatch(self.in_numel, n));
        }
        Ok(())
    }

    /// `index`／`src` の各要素 `p`（行優先）の書き込み先（入力の行優先線形添字）を
    /// 求める。`index` の値が `[0, shape[dim])` を外れる（負値を含む）場合は
    /// [`ShapeError::IndexOutOfRange`]。
    fn positions(&self, index: &[i32]) -> Result<Vec<usize>, ShapeError> {
        let dim_size = self.in_shape[self.dim];
        let rank = self.idx_shape.len();
        let mut coords = vec![0usize; rank];
        let mut out = Vec::with_capacity(self.idx_numel);
        for &raw in index {
            let idx = usize::try_from(raw).ok().filter(|&v| v < dim_size).ok_or(
                ShapeError::IndexOutOfRange {
                    dim: self.dim,
                    index: i64::from(raw),
                    dim_size,
                },
            )?;
            let mut pos = 0usize;
            for (axis, &stride) in self.in_strides.iter().enumerate() {
                let c = if axis == self.dim { idx } else { coords[axis] };
                pos += c * stride;
            }
            out.push(pos);
            // オドメータ式で `coords` を進める（要素ごとの Vec 確保を避ける）。
            for axis in (0..rank).rev() {
                coords[axis] += 1;
                if coords[axis] < self.idx_shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        Ok(out)
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// shape を検査して [`ScatterReduceLayout`] を返す。
///
/// `dim`・rank・`index == src` の shape・`dim` 以外の軸で `index <= input` を
/// [`scatter_out_shape`] で検査し、要素数と `f32`／`f64`／`i32`／`usize` の
/// バイト数の `checked_mul` および `isize::MAX` 超過を確保より前に拒否する
/// （[`ShapeError::ElementCountOverflow`]）。
pub fn scatter_reduce_layout(
    in_shape: &[usize],
    index_shape: &[usize],
    src_shape: &[usize],
    dim: usize,
) -> Result<ScatterReduceLayout, ShapeError> {
    scatter_out_shape(in_shape, index_shape, src_shape, dim)?;
    let in_numel = checked_numel(in_shape)?;
    let idx_numel = checked_numel(index_shape)?;
    // 最大の要素幅（f64／usize＝8 バイト）で確保サイズの上限を確認する。
    for n in [in_numel, idx_numel] {
        let bytes = n
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(ShapeError::ElementCountOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }
    let mut in_strides = vec![1usize; in_shape.len()];
    for axis in (0..in_shape.len().saturating_sub(1)).rev() {
        in_strides[axis] = in_strides[axis + 1]
            .checked_mul(in_shape[axis + 1].max(1))
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    Ok(ScatterReduceLayout {
        dim,
        in_shape: in_shape.to_vec(),
        idx_shape: index_shape.to_vec(),
        in_numel,
        idx_numel,
        in_strides,
    })
}

/// 位置ごとの寄与数（触れられた回数）。
fn touch_counts(pos: &[usize], n: usize) -> Vec<usize> {
    let mut cnt = vec![0usize; n];
    for &p in pos {
        cnt[p] += 1;
    }
    cnt
}

/// `Amax`／`Amin` の更新規則: 現在値が NaN なら保持、`v` が NaN なら採用、
/// それ以外は厳密に大きい（小さい）ときのみ採用（タイは先勝ち）。
fn extremum_take(cur: f32, v: f32, is_max: bool) -> bool {
    if cur.is_nan() {
        false
    } else if v.is_nan() {
        true
    } else if is_max {
        v > cur
    } else {
        v < cur
    }
}

fn extremum_forward(
    input: &[f32],
    src: &[f32],
    pos: &[usize],
    include_self: bool,
    is_max: bool,
) -> Vec<f32> {
    let mut out = input.to_vec();
    let mut seeded = vec![false; input.len()];
    for (&p, &v) in pos.iter().zip(src) {
        if !seeded[p] {
            seeded[p] = true;
            if !include_self {
                out[p] = v;
                continue;
            }
        }
        if extremum_take(out[p], v, is_max) {
            out[p] = v;
        }
    }
    out
}

/// `scatter_reduce` の順伝播（モジュール doc の数値契約）。
///
/// `input`／`index`／`src` は連続配置の行優先スライス。戻り値は入力と同 shape の
/// 連続配置。長さ不一致・`index` の範囲外（負値を含む）は型付きエラー。
pub fn scatter_reduce_host(
    input: &[f32],
    index: &[i32],
    src: &[f32],
    layout: &ScatterReduceLayout,
    mode: ScatterReduceMode,
    include_self: bool,
) -> Result<Vec<f32>, ShapeError> {
    layout.check_lens(Some(input.len()), index.len(), Some(src.len()), None)?;
    let pos = layout.positions(index)?;
    match mode {
        ScatterReduceMode::Amax | ScatterReduceMode::Amin => Ok(extremum_forward(
            input,
            src,
            &pos,
            include_self,
            mode == ScatterReduceMode::Amax,
        )),
        ScatterReduceMode::Sum | ScatterReduceMode::Prod | ScatterReduceMode::Mean => {
            let cnt = touch_counts(&pos, input.len());
            let is_prod = mode == ScatterReduceMode::Prod;
            let neutral = if is_prod { 1.0_f64 } else { 0.0_f64 };
            let mut acc: Vec<f64> = input
                .iter()
                .zip(&cnt)
                .map(|(&x, &c)| {
                    if c > 0 && !include_self {
                        neutral
                    } else {
                        f64::from(x)
                    }
                })
                .collect();
            for (&p, &v) in pos.iter().zip(src) {
                if is_prod {
                    acc[p] *= f64::from(v);
                } else {
                    acc[p] += f64::from(v);
                }
            }
            Ok(acc
                .iter()
                .zip(input)
                .zip(&cnt)
                .map(|((&a, &x), &c)| {
                    if c == 0 {
                        x
                    } else if mode == ScatterReduceMode::Mean {
                        let n = c + usize::from(include_self);
                        (a / n as f64) as f32
                    } else {
                        a as f32
                    }
                })
                .collect())
        }
    }
}

/// [`scatter_reduce_host`] の VJP。`(d_input, d_src)` を返す。
///
/// 入力値は forward と同じ規則で再計算に使う（`Op::Logcumsumexp` と同じ「入力から
/// 再計算」方式）。`upstream` は入力と同 shape（要素数を検査する）。
pub fn scatter_reduce_vjp_host(
    input: &[f32],
    index: &[i32],
    src: &[f32],
    upstream: &[f32],
    layout: &ScatterReduceLayout,
    mode: ScatterReduceMode,
    include_self: bool,
) -> Result<(Vec<f32>, Vec<f32>), ShapeError> {
    layout.check_lens(
        Some(input.len()),
        index.len(),
        Some(src.len()),
        Some(upstream.len()),
    )?;
    let pos = layout.positions(index)?;
    let n = input.len();
    let cnt = touch_counts(&pos, n);
    let g = |q: usize| f64::from(upstream[q]);
    let mut d_input = vec![0.0_f32; n];
    let mut d_src = vec![0.0_f32; src.len()];
    // 触れられていない位置の d_input は常に上流勾配（全 mode 共通）。
    for q in 0..n {
        if cnt[q] == 0 {
            d_input[q] = upstream[q];
        }
    }
    match mode {
        ScatterReduceMode::Sum => {
            for (p, &q) in pos.iter().enumerate() {
                d_src[p] = upstream[q];
            }
            if include_self {
                d_input.copy_from_slice(upstream);
            }
        }
        ScatterReduceMode::Mean => {
            for (p, &q) in pos.iter().enumerate() {
                let m = cnt[q] + usize::from(include_self);
                d_src[p] = (g(q) / m as f64) as f32;
            }
            if include_self {
                for q in 0..n {
                    if cnt[q] > 0 {
                        d_input[q] = (g(q) / (cnt[q] + 1) as f64) as f32;
                    }
                }
            }
        }
        ScatterReduceMode::Prod => {
            // 位置ごとの寄与列（src は走査順、include_self の自己寄与は末尾）を集め、
            // 各寄与を除いた積を「前置積 × 後置積」で直接求める。総積を各値で割る方式は
            // inf 寄与で inf/inf = NaN になるため使わない（0・inf を含む lane でも
            // 他寄与の積がそのまま得られる）。slot は src の位置 p、自己寄与は usize::MAX。
            let mut lanes: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
            for (p, (&q, &v)) in pos.iter().zip(src).enumerate() {
                lanes[q].push((p, f64::from(v)));
            }
            if include_self {
                for q in 0..n {
                    if cnt[q] > 0 {
                        lanes[q].push((usize::MAX, f64::from(input[q])));
                    }
                }
            }
            for (q, lane) in lanes.iter().enumerate() {
                let k = lane.len();
                if k == 0 {
                    continue;
                }
                // NaN 寄与を含む lane は全寄与の勾配を NaN とする（PyTorch と同じ。NaN 自身の
                // 排他的積は有限になりうるが、NaN の伝播を隠さない）。inf は割り戻さず直接積を取る。
                if lane.iter().any(|&(_, v)| v.is_nan()) {
                    for &(slot, _) in lane {
                        if slot == usize::MAX {
                            d_input[q] = f32::NAN;
                        } else {
                            d_src[slot] = f32::NAN;
                        }
                    }
                    continue;
                }
                // 各寄与を除いた積（排他的積）。0 と inf／f64 overflow した積を直接乗算すると
                // 0 × inf = NaN になるため、先に 0・inf の個数を数えて分岐する:
                //  - 他寄与に 0 が残る → 0（inf が混在しても 0。NaN にしない）
                //  - 他寄与に 0 が無く inf が残る → 符号付き inf
                //  - それ以外（有限・非 0 のみ）→ (仮数, 2 の指数) 対で積を取り overflow／underflow
                //    を避け、最後に 1 回だけ f64 へ戻す（途中で inf・0 を作らない）。
                let zeros = lane.iter().filter(|&&(_, v)| v == 0.0).count();
                let infs = lane.iter().filter(|&&(_, v)| v.is_infinite()).count();
                let neg_infs = lane
                    .iter()
                    .filter(|&&(_, v)| v.is_infinite() && v < 0.0)
                    .count();
                // 有限・非 0 の寄与のみ (仮数, 指数) へ分解（0・inf は単位元 (1, 0)）。
                let fac: Vec<(f64, i64)> = lane
                    .iter()
                    .map(|&(_, v)| {
                        if v == 0.0 || v.is_infinite() {
                            (1.0, 0)
                        } else {
                            frexp_normal(v)
                        }
                    })
                    .collect();
                let mut suffix = vec![(1.0_f64, 0_i64); k + 1];
                for i in (0..k).rev() {
                    suffix[i] = scaled_mul(suffix[i + 1], fac[i]);
                }
                let mut prefix = (1.0_f64, 0_i64);
                for (i, &(slot, v)) in lane.iter().enumerate() {
                    let zeros_other = zeros - usize::from(v == 0.0);
                    let infs_other = infs - usize::from(v.is_infinite());
                    let excl = scaled_mul(prefix, suffix[i + 1]);
                    let g_excl = if zeros_other > 0 {
                        0.0_f32
                    } else if infs_other > 0 {
                        let neg_inf_other = neg_infs - usize::from(v.is_infinite() && v < 0.0);
                        let neg = (neg_inf_other % 2 == 1) != (excl.0 < 0.0);
                        let inf = if neg {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        };
                        (g(q) * inf) as f32
                    } else {
                        (scaled_apply(g(q), excl)) as f32
                    };
                    if slot == usize::MAX {
                        d_input[q] = g_excl;
                    } else {
                        d_src[slot] = g_excl;
                    }
                    prefix = scaled_mul(prefix, fac[i]);
                }
            }
        }
        ScatterReduceMode::Amax | ScatterReduceMode::Amin => {
            let is_max = mode == ScatterReduceMode::Amax;
            let result = extremum_forward(input, src, &pos, include_self, is_max);
            // 結果と等しい寄与の個数（NaN 結果は等しいものが無く 0）。
            let mut ties = vec![0usize; n];
            for (&q, &v) in pos.iter().zip(src) {
                if v == result[q] {
                    ties[q] += 1;
                }
            }
            if include_self {
                for q in 0..n {
                    if cnt[q] > 0 && input[q] == result[q] {
                        ties[q] += 1;
                    }
                }
            }
            for (p, &q) in pos.iter().enumerate() {
                if src[p] == result[q] && ties[q] > 0 {
                    d_src[p] = (g(q) / ties[q] as f64) as f32;
                }
            }
            if include_self {
                for q in 0..n {
                    if cnt[q] > 0 && input[q] == result[q] && ties[q] > 0 {
                        d_input[q] = (g(q) / ties[q] as f64) as f32;
                    }
                }
            }
        }
    }
    Ok((d_input, d_src))
}

/// 有限・非 0 の正規化済み `f64` を `(仮数, 2 の指数)` へ分解する（`m × 2^e`、`|m|` は `[0.5, 1)`）。
/// `f32` から昇格した値は常に `f64` の正規数のため部分正規数は扱わない。
fn frexp_normal(v: f64) -> (f64, i64) {
    let bits = v.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1022;
    let m = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1022_u64 << 52));
    (m, e)
}

/// `(仮数, 指数)` 同士の積。仮数積を `[0.5, 1)` へ再正規化し、指数側へ繰り上げる
/// （仮数は有限・非 0 のみ。仮数積は `[0.25, 1)` で常に正規数）。
fn scaled_mul(a: (f64, i64), b: (f64, i64)) -> (f64, i64) {
    let (m, e) = frexp_normal(a.0 * b.0);
    (m, a.1 + b.1 + e)
}

/// `g × m × 2^e` を中間で不要に inf／0 を作らず求める（指数を 2 分割して適用）。
fn scaled_apply(g: f64, x: (f64, i64)) -> f64 {
    let e = x.1.clamp(-2040, 2040);
    let h1 = (e / 2) as i32;
    let h2 = (e - i64::from(h1)) as i32;
    g * x.0 * 2.0_f64.powi(h1) * 2.0_f64.powi(h2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay(inp: &[usize], idx: &[usize], dim: usize) -> ScatterReduceLayout {
        scatter_reduce_layout(inp, idx, idx, dim).expect("layout")
    }

    fn run(mode: ScatterReduceMode, inc: bool, x: &[f32], idx: &[i32], s: &[f32]) -> Vec<f32> {
        let l = lay(&[x.len()], &[idx.len()], 0);
        scatter_reduce_host(x, idx, s, &l, mode, inc).expect("host")
    }

    #[test]
    fn each_mode_matches_hand_values() {
        use ScatterReduceMode::*;
        let x = [1.0, 2.0, 3.0];
        let idx = [0, 0, 2];
        let s = [4.0, 5.0, 6.0];
        assert_eq!(run(Sum, true, &x, &idx, &s), vec![10.0, 2.0, 9.0]);
        assert_eq!(run(Sum, false, &x, &idx, &s), vec![9.0, 2.0, 6.0]);
        assert_eq!(run(Prod, true, &x, &idx, &s), vec![20.0, 2.0, 18.0]);
        assert_eq!(run(Prod, false, &x, &idx, &s), vec![20.0, 2.0, 6.0]);
        assert_eq!(run(Mean, true, &x, &idx, &s), vec![10.0 / 3.0, 2.0, 4.5]);
        assert_eq!(run(Mean, false, &x, &idx, &s), vec![4.5, 2.0, 6.0]);
        assert_eq!(run(Amax, true, &x, &idx, &s), vec![5.0, 2.0, 6.0]);
        assert_eq!(
            run(Amax, false, &x, &idx, &[0.5, 0.25, 0.0]),
            vec![0.5, 2.0, 0.0]
        );
        assert_eq!(run(Amin, true, &x, &idx, &s), vec![1.0, 2.0, 3.0]);
        assert_eq!(run(Amin, false, &x, &idx, &s), vec![4.0, 2.0, 6.0]);
    }

    #[test]
    fn untouched_positions_keep_input_bits() {
        let x = [f32::NAN, -0.0, f32::INFINITY];
        for mode in [
            ScatterReduceMode::Sum,
            ScatterReduceMode::Prod,
            ScatterReduceMode::Mean,
            ScatterReduceMode::Amax,
            ScatterReduceMode::Amin,
        ] {
            let out = run(mode, false, &x, &[1], &[7.0]);
            assert_eq!(out[0].to_bits(), x[0].to_bits());
            assert_eq!(out[2].to_bits(), x[2].to_bits());
            assert_eq!(out[1], 7.0);
        }
    }

    #[test]
    fn sum_include_self_is_f64_sequential_sum() {
        let x = [16_777_216.0_f32];
        let s = [1.0_f32; 4];
        let out = run(ScatterReduceMode::Sum, true, &x, &[0, 0, 0, 0], &s);
        assert_eq!(out[0], 16_777_220.0);
    }

    #[test]
    fn non_trailing_dim_and_empty_index() {
        let l = scatter_reduce_layout(&[2, 3], &[2, 2], &[2, 2], 1).expect("layout");
        let x = [0.0; 6];
        let out = scatter_reduce_host(
            &x,
            &[2, 2, 0, 1],
            &[1.0, 2.0, 3.0, 4.0],
            &l,
            ScatterReduceMode::Sum,
            true,
        )
        .expect("host");
        assert_eq!(out, vec![0.0, 0.0, 3.0, 3.0, 4.0, 0.0]);
        let l0 = scatter_reduce_layout(&[3], &[0], &[0], 0).expect("layout");
        let same = scatter_reduce_host(
            &[1.0, 2.0, 3.0],
            &[],
            &[],
            &l0,
            ScatterReduceMode::Prod,
            false,
        )
        .expect("host");
        assert_eq!(same, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn layout_and_kernel_reject_invalid_inputs() {
        assert!(matches!(
            scatter_reduce_layout(&[3], &[2], &[2], 1),
            Err(ShapeError::AxisOutOfRange { .. })
        ));
        assert!(scatter_reduce_layout(&[3, 3], &[2], &[2], 0).is_err());
        assert!(scatter_reduce_layout(&[3], &[2], &[3], 0).is_err());
        assert!(scatter_reduce_layout(&[3, 2], &[1, 3], &[1, 3], 0).is_err());
        assert!(matches!(
            scatter_reduce_layout(&[usize::MAX, usize::MAX, 0], &[1, 1, 0], &[1, 1, 0], 0),
            Err(ShapeError::ElementCountOverflow)
        ));
        let l = lay(&[3], &[2], 0);
        let m = ScatterReduceMode::Sum;
        assert!(matches!(
            scatter_reduce_host(&[0.0; 3], &[0, 3], &[1.0; 2], &l, m, true),
            Err(ShapeError::IndexOutOfRange { .. })
        ));
        assert!(matches!(
            scatter_reduce_host(&[0.0; 3], &[0, -1], &[1.0; 2], &l, m, true),
            Err(ShapeError::IndexOutOfRange { .. })
        ));
        assert!(matches!(
            scatter_reduce_host(&[0.0; 2], &[0, 1], &[1.0; 2], &l, m, true),
            Err(ShapeError::ElementCountMismatch { .. })
        ));
        assert!(matches!(
            scatter_reduce_vjp_host(&[0.0; 3], &[0, 1], &[1.0; 2], &[1.0; 2], &l, m, true),
            Err(ShapeError::ElementCountMismatch { .. })
        ));
    }

    /// 全 mode・`include_self` の VJP を f64 中心差分と突合する（タイ・0 なしの入力）。
    #[test]
    fn vjp_matches_central_difference() {
        let x = [1.5_f32, -2.0, 0.75, 3.0];
        let idx = [0, 0, 2, 2, 2];
        let s = [2.5_f32, -1.25, 4.0, 0.5, -3.5];
        let g = [0.3_f32, -0.7, 1.1, 0.9];
        let l = lay(&[4], &[5], 0);
        for mode in [
            ScatterReduceMode::Sum,
            ScatterReduceMode::Prod,
            ScatterReduceMode::Mean,
            ScatterReduceMode::Amax,
            ScatterReduceMode::Amin,
        ] {
            for inc in [true, false] {
                let (dx, ds) =
                    scatter_reduce_vjp_host(&x, &idx, &s, &g, &l, mode, inc).expect("vjp");
                let loss = |xx: &[f32], ss: &[f32]| -> f64 {
                    scatter_reduce_host(xx, &idx, ss, &l, mode, inc)
                        .expect("host")
                        .iter()
                        .zip(&g)
                        .map(|(&o, &w)| f64::from(o) * f64::from(w))
                        .sum()
                };
                let h = 1e-2_f32;
                for i in 0..x.len() {
                    let (mut a, mut b) = (x, x);
                    a[i] += h;
                    b[i] -= h;
                    let num = (loss(&a, &s) - loss(&b, &s)) / (2.0 * f64::from(h));
                    assert!(
                        (num - f64::from(dx[i])).abs() < 5e-3,
                        "{mode:?} inc={inc} dx[{i}]: num={num} got={}",
                        dx[i]
                    );
                }
                for i in 0..s.len() {
                    let (mut a, mut b) = (s, s);
                    a[i] += h;
                    b[i] -= h;
                    let num = (loss(&x, &a) - loss(&x, &b)) / (2.0 * f64::from(h));
                    assert!(
                        (num - f64::from(ds[i])).abs() < 5e-3,
                        "{mode:?} inc={inc} ds[{i}]: num={num} got={}",
                        ds[i]
                    );
                }
            }
        }
    }

    #[test]
    fn prod_vjp_is_finite_with_zero_contributions() {
        let l = lay(&[1], &[3], 0);
        let g = [1.0_f32];
        // 0 が 1 個: 0 の寄与だけが非ゼロ勾配（残りの積）。
        let (dx, ds) = scatter_reduce_vjp_host(
            &[2.0],
            &[0, 0, 0],
            &[0.0, 3.0, 5.0],
            &g,
            &l,
            ScatterReduceMode::Prod,
            true,
        )
        .expect("vjp");
        assert_eq!((dx[0], ds.clone()), (0.0, vec![30.0, 0.0, 0.0]));
        // 0 が 2 個以上: 全勾配 0。
        let (dx2, ds2) = scatter_reduce_vjp_host(
            &[0.0],
            &[0, 0, 0],
            &[0.0, 3.0, 5.0],
            &g,
            &l,
            ScatterReduceMode::Prod,
            true,
        )
        .expect("vjp");
        assert_eq!((dx2[0], ds2), (0.0, vec![0.0, 0.0, 0.0]));
    }

    #[test]
    fn prod_vjp_does_not_divide_out_infinite_contributions() {
        // include_self=false・src=[inf, 2]: src[0] の勾配は他寄与の積 2.0（inf/inf の NaN にしない）。
        let l = lay(&[1], &[2], 0);
        let (dx, ds) = scatter_reduce_vjp_host(
            &[7.0],
            &[0, 0],
            &[f32::INFINITY, 2.0],
            &[1.0],
            &l,
            ScatterReduceMode::Prod,
            false,
        )
        .expect("vjp");
        assert_eq!(ds, vec![2.0, f32::INFINITY]);
        assert_eq!(dx, vec![0.0]);
    }

    #[test]
    fn prod_vjp_zero_with_infinite_contribution_is_not_nan() {
        // src=[0, inf, 2]・include_self=false: 他に 0 が残る要素は inf が混在しても 0。
        let l = lay(&[1], &[3], 0);
        let (_, ds) = scatter_reduce_vjp_host(
            &[7.0],
            &[0, 0, 0],
            &[0.0, f32::INFINITY, 2.0],
            &[1.0],
            &l,
            ScatterReduceMode::Prod,
            false,
        )
        .expect("vjp");
        assert_eq!(ds[1], 0.0);
        assert_eq!(ds[2], 0.0);
        assert!(!ds[0].is_nan());
    }

    #[test]
    fn prod_vjp_zero_with_overflowing_finite_products_is_not_nan() {
        // 0 と 1e38 を多数: 総積は f64 でも overflow する。0 の勾配は残りの積（f32 で inf）、
        // 他は 0。NaN にしない。
        let l = lay(&[1], &[40], 0);
        let mut src = vec![1.0e38_f32; 40];
        src[0] = 0.0;
        let idx = vec![0_i32; 40];
        let (_, ds) = scatter_reduce_vjp_host(
            &[1.0],
            &idx,
            &src,
            &[1.0],
            &l,
            ScatterReduceMode::Prod,
            false,
        )
        .expect("vjp");
        assert_eq!(ds[0], f32::INFINITY);
        assert!(ds[1..].iter().all(|&d| d == 0.0));
        // 0 無し: 大小混在で途中 overflow／underflow しても正しい有限積（1e38^20 × 1e-38^20 = 1）。
        let mut src2 = vec![1.0e38_f32; 20];
        src2.extend(vec![1.0e-38_f32; 21]);
        let idx2 = vec![0_i32; 41];
        let l2 = lay(&[1], &[41], 0);
        let (_, ds2) = scatter_reduce_vjp_host(
            &[1.0],
            &idx2,
            &src2,
            &[1.0],
            &l2,
            ScatterReduceMode::Prod,
            false,
        )
        .expect("vjp");
        // src2[0] を除く積 = 1e38^19 × 1e-38^21 = 1e-76 → f32 では 0。NaN でないことを確認。
        assert!(ds2.iter().all(|d| !d.is_nan()));
    }

    #[test]
    fn extremum_vjp_splits_ties_evenly_and_nan_is_zero() {
        let l = lay(&[1], &[3], 0);
        let (dx, ds) = scatter_reduce_vjp_host(
            &[1.0],
            &[0, 0, 0],
            &[2.0, 2.0, 1.0],
            &[6.0],
            &l,
            ScatterReduceMode::Amax,
            true,
        )
        .expect("vjp");
        assert_eq!((dx[0], ds), (0.0, vec![3.0, 3.0, 0.0]));
        let (dx, ds) = scatter_reduce_vjp_host(
            &[1.0],
            &[0, 0],
            &[f32::NAN, 2.0],
            &[6.0],
            &l_one(2),
            ScatterReduceMode::Amax,
            true,
        )
        .expect("vjp");
        assert_eq!((dx[0], ds), (0.0, vec![0.0, 0.0]));
    }

    fn l_one(n: usize) -> ScatterReduceLayout {
        lay(&[1], &[n], 0)
    }

    #[test]
    fn vjp_is_bit_deterministic() {
        let l = lay(&[3], &[4], 0);
        let call = || {
            scatter_reduce_vjp_host(
                &[1.0, 2.0, 3.0],
                &[0, 1, 0, 2],
                &[0.1, 0.2, 0.3, 0.4],
                &[1.0, 1.0, 1.0],
                &l,
                ScatterReduceMode::Prod,
                true,
            )
            .expect("vjp")
        };
        let (a, b) = (call(), call());
        let bits = |v: &(Vec<f32>, Vec<f32>)| -> Vec<u32> {
            v.0.iter().chain(&v.1).map(|x| x.to_bits()).collect()
        };
        assert_eq!(bits(&a), bits(&b));
    }
}
