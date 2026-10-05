//! `median`／`kthvalue`／`quantile`／`nansum`／`nanmean` のホスト参照カーネルの
//! **単一情報源**（イシュー #2637・親 #2625。実装記録は
//! `docs/autodiff-stat-reduce-ops-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::stat_reduce_ops`（`BackendOps` が `Unsupported` のときの
//!   ホストフォールバック）・`autodiff::grad`（`median(None)`・`quantile`・
//!   `nansum`・`nanmean` の VJP）・`backend-cpu` の
//!   `CpuBackendOps::stat_*` がいずれも本モジュールの関数を直接呼ぶ。順序規則・
//!   NaN 規則・アキュムレータ契約をクレート間で複製せず、乖離を構造的に排除する
//!   （`cumulative.rs`・`fft.rs` と同じ「共有可能な層に一本化する」方式）。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の
//!   `Unsupported` から autodiff 側がこのホスト実装へフォールバックする。
//!
//! # 順序の契約（独自契約）
//!
//! lane（縮約軸 1 本）内の順序は `BackendOps::sort` の順序契約と同じ: 昇順・
//! **安定**（同値は元添字の昇順）・NaN は任意の非 NaN より大きい・`±0` は同値。
//! 実装は lane ごとの添字バッファの安定ソート（`sort_by`）で、
//! `select_nth_unstable` は使わない（タイの索引が非決定的になるため）。PyTorch は
//! `kthvalue`／`median` のタイ時の索引を規定しておらず、本実装は上記の安定順で
//! 決定的に返す（PyTorch の実測とタイ時の索引が一致することは保証しない）。
//!
//! # 数値契約
//!
//! - 選択系（`kthvalue`・`median`・`quantile` の `Lower`／`Higher`／`Nearest`）は
//!   算術を含まず、値は入力要素と bit 一致する。FMA 契約・`f64` 契約は非該当。
//! - `nansum`／`nanmean` は lane ごとに要素を `f64` へ昇格して添字昇順に逐次加算し、
//!   最後に 1 回だけ `f32` へ downcast する（長軸縮約の `f64` アキュムレータ契約）。
//!   `nanmean` は `f64` のまま `和 ÷ 非 NaN 個数` を計算してから 1 回 downcast する。
//! - `quantile` の rank は PyTorch 2.14.0 の実測に合わせ **`f64` で
//!   `f64::from(q) * (n - 1)`** を計算する（`f32` の積に丸めると離散モードで選ぶ要素が
//!   ずれる。決定記録 §3）。補間は `f64` で PyTorch
//!   の `lerp` と同じ 2 分岐を計算し 1 回 downcast する。`mul_add` は使わず、
//!   matmul 系 FMA 契約には触れない。
//! - 非有限入力は事前に拒否せず伝播する（`cumsum`・FFT と同じ）。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! 形状・`dim` の検査は [`stat_layout`] に集約し、確保より前に `dim` 範囲・要素数・
//! バイト数・作業バッファの `checked_mul` を検査して型付きエラーで拒否する。
//! カーネルは入力スライス長も再検査し、索引は `i32::try_from`（無検査 `as i32` を
//! 使わない）で変換する。`unsafe`／`get_unchecked` は使わない。

use std::cmp::Ordering;

use crate::device::BackendError;
use crate::error::ShapeError;
use crate::ops_shape::reduce_out_shape;

/// `quantile` の補間方式（`torch.quantile` の `interpolation` 相当）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantileInterpolation {
    /// 下側と上側の線形補間（PyTorch 既定）。
    #[default]
    Linear,
    /// 下側の要素。
    Lower,
    /// 上側の要素。
    Higher,
    /// 下側と上側の中点。
    Midpoint,
    /// 最近傍の要素（`.5` は偶数丸め）。
    Nearest,
}

/// 形状・引数検査の失敗（型付き）。`autodiff` は `AutodiffError` へ、
/// バックエンドは [`BackendError`] へ写像する（`FftError` と同じ置き方）。
#[derive(Debug, Clone, PartialEq)]
pub enum StatReduceError {
    /// `dim` 範囲外・要素数／バイト数オーバーフロー・索引幅超過等の形状違反。
    Shape(ShapeError),
    /// `k`・`q`・空 lane など、引数の不正。
    InvalidArgument(String),
}

impl std::fmt::Display for StatReduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatReduceError::Shape(e) => write!(f, "stat_reduce: shape error: {e:?}"),
            StatReduceError::InvalidArgument(msg) => {
                write!(f, "stat_reduce: invalid argument: {msg}")
            }
        }
    }
}

impl std::error::Error for StatReduceError {}

impl From<ShapeError> for StatReduceError {
    fn from(err: ShapeError) -> Self {
        StatReduceError::Shape(err)
    }
}

impl From<StatReduceError> for BackendError {
    fn from(err: StatReduceError) -> Self {
        match err {
            StatReduceError::Shape(e) => BackendError::ShapeMismatch(e),
            StatReduceError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
        }
    }
}

/// 解決済みの縮約レイアウト（[`stat_layout`] の戻り値）。
///
/// 入力は連続配置の行優先。縮約軸の前を `outer`・後ろを `inner` として、
/// 1 lane = `(outer 添字, inner 添字)`。`dim = None` は全要素を 1 lane
/// （`outer = inner = 1`）として扱い、出力は 0 次元になる。
///
/// フィールドは非公開で、[`stat_layout`] だけが生成する。生成時の検査で整合性が
/// 保証され、公開カーネルが外部から改変された値で範囲外アクセスしないようにする
/// （REQ-8）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatLayout {
    outer: usize,
    axis_len: usize,
    inner: usize,
    numel: usize,
    shape: Vec<usize>,
    out_shape: Vec<usize>,
}

impl StatLayout {
    /// 縮約軸（`dim = None` は全要素）の長さ。
    pub fn axis_len(&self) -> usize {
        self.axis_len
    }

    /// 入力 shape。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 縮約後（縮約軸を落とした）shape。`dim = None` は空（0 次元）。
    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }

    /// 入力の総要素数。
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// 出力の総要素数（lane 数）。
    pub fn out_numel(&self) -> usize {
        self.outer * self.inner
    }

    /// 索引（`i32`）で表現できることを検査する。軸長 0 は常に成功する。
    /// 最大索引 `axis_len - 1` が `i32::MAX` を超える場合は
    /// [`ShapeError::IndexRangeOverflow`]（`sort`／`topk` と同じ契約）。
    pub fn check_i32_indices(&self) -> Result<(), ShapeError> {
        if self.axis_len == 0 {
            return Ok(());
        }
        let max_index = self.axis_len - 1;
        i32::try_from(max_index)
            .map(|_| ())
            .map_err(|_| ShapeError::IndexRangeOverflow { index: max_index })
    }

    fn check_len(&self, actual: usize) -> Result<(), ShapeError> {
        if actual == self.numel {
            Ok(())
        } else {
            Err(ShapeError::ElementCountMismatch {
                expected: self.numel,
                actual,
            })
        }
    }

    fn check_out_len(&self, actual: usize) -> Result<(), ShapeError> {
        let expected = self.out_numel();
        if actual == expected {
            Ok(())
        } else {
            Err(ShapeError::ElementCountMismatch { expected, actual })
        }
    }

    #[inline]
    fn pos(&self, o: usize, a: usize, i: usize) -> usize {
        (o * self.axis_len + a) * self.inner + i
    }

    /// lane `(o, i)` を `buf` へ集める（長さ `axis_len`）。
    fn gather(&self, x: &[f32], o: usize, i: usize, buf: &mut Vec<f32>) {
        buf.clear();
        for a in 0..self.axis_len {
            buf.push(x[self.pos(o, a, i)]);
        }
    }
}

/// `dim`（`None` は全要素）と shape を検査して [`StatLayout`] を返す。
///
/// `dim >= rank`（rank 0 に `Some(_)` を含む）は [`ShapeError::AxisOutOfRange`]、
/// 要素数・バイト数・作業バッファの `checked_mul` オーバーフローや `isize::MAX`
/// 超過は [`ShapeError::ElementCountOverflow`]。要素数 0 の入力は、入力側の部分積を
/// 計算する前に空レイアウトへ倒す（出力 shape の積だけを検査する）。
pub fn stat_layout(shape: &[usize], dim: Option<usize>) -> Result<StatLayout, ShapeError> {
    let out_shape = reduce_out_shape(shape, dim)?;
    if shape.contains(&0) {
        let axis_len = dim.map_or(0, |d| shape[d]);
        let out_numel = checked_numel(&out_shape)?;
        // 空入力でも出力 `Vec<f32>` のバイト数と lane 作業バッファのサイズを確保前に検査する
        // （`[0, usize::MAX]` の軸 0 縮約等で巨大確保 panic させない。REQ-8）。
        check_alloc_bytes(out_numel, std::mem::size_of::<f32>())?;
        check_alloc_bytes(
            axis_len,
            std::mem::size_of::<usize>() + std::mem::size_of::<f32>(),
        )?;
        // 空入力。縮約軸が長さ 0 のときだけ出力 lane が残る（空 lane）。
        let (outer, inner) = if axis_len == 0 {
            (out_numel, 1)
        } else {
            (0, 0)
        };
        return Ok(StatLayout {
            outer,
            axis_len,
            inner,
            numel: 0,
            shape: shape.to_vec(),
            out_shape,
        });
    }
    let numel = checked_numel(shape)?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    let (outer, axis_len, inner) = match dim {
        None => (1, numel, 1),
        Some(d) => (
            checked_numel(&shape[..d])?,
            shape[d],
            checked_numel(&shape[d + 1..])?,
        ),
    };
    // lane ソート用の作業バッファ（`f32` 値 + `usize` 添字）の確保サイズ。
    let work = axis_len
        .checked_mul(std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if work > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(StatLayout {
        outer,
        axis_len,
        inner,
        numel,
        shape: shape.to_vec(),
        out_shape,
    })
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

fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// 昇順・NaN は最大・`±0` は同値（`sort_by` の安定性で同値は元添字順）。
fn nan_last_cmp(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// `lane` の安定昇順の添字列を `order` へ作る。
fn sort_order(lane: &[f32], order: &mut Vec<usize>) {
    order.clear();
    order.extend(0..lane.len());
    order.sort_by(|&a, &b| nan_last_cmp(lane[a], lane[b]));
}

fn to_i32(index: usize) -> Result<i32, StatReduceError> {
    i32::try_from(index)
        .map_err(|_| StatReduceError::Shape(ShapeError::IndexRangeOverflow { index }))
}

fn require_nonempty_lane(layout: &StatLayout, who: &str) -> Result<(), StatReduceError> {
    if layout.axis_len == 0 {
        return Err(StatReduceError::InvalidArgument(format!(
            "{who}: reduction axis has zero size"
        )));
    }
    Ok(())
}

/// `kthvalue`（`k` は 1 始まり。値と索引）。安定昇順の `k - 1` 番目を返す。
///
/// `k == 0`・`k > 軸長`・軸長 0 は [`StatReduceError::InvalidArgument`]。
/// NaN は最大として並ぶだけで特別扱いしない（`k` が NaN の位置に当たれば NaN）。
pub fn kthvalue_host(
    x: &[f32],
    layout: &StatLayout,
    k: usize,
) -> Result<(Vec<f32>, Vec<i32>), StatReduceError> {
    layout.check_len(x.len())?;
    layout.check_i32_indices()?;
    require_nonempty_lane(layout, "kthvalue")?;
    if k == 0 || k > layout.axis_len {
        return Err(StatReduceError::InvalidArgument(format!(
            "kthvalue: k={k} is out of range for an axis of length {}",
            layout.axis_len
        )));
    }
    select_lanes(x, layout, |_, _| k - 1)
}

/// `median`（軸指定。値と索引）。**下側中央値**（ソート位置 `(n - 1) / 2`。偶数個でも
/// 平均しない）。lane に NaN があれば値は NaN・索引は最初の NaN の位置。軸長 0 は
/// [`StatReduceError::InvalidArgument`]。
pub fn median_dim_host(
    x: &[f32],
    layout: &StatLayout,
) -> Result<(Vec<f32>, Vec<i32>), StatReduceError> {
    layout.check_len(x.len())?;
    layout.check_i32_indices()?;
    require_nonempty_lane(layout, "median")?;
    select_lanes(x, layout, |nan_count, n| {
        if nan_count > 0 {
            n - nan_count
        } else {
            (n - 1) / 2
        }
    })
}

/// lane ごとに安定ソートし `choose(nan_count, n)` のソート位置の要素（値・元添字）を返す。
fn select_lanes(
    x: &[f32],
    layout: &StatLayout,
    choose: impl Fn(usize, usize) -> usize,
) -> Result<(Vec<f32>, Vec<i32>), StatReduceError> {
    let lanes = layout.out_numel();
    let mut values = vec![0.0_f32; lanes];
    let mut indices = vec![0_i32; lanes];
    let mut lane = Vec::with_capacity(layout.axis_len);
    let mut order = Vec::with_capacity(layout.axis_len);
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            layout.gather(x, o, i, &mut lane);
            sort_order(&lane, &mut order);
            let nan_count = lane.iter().filter(|v| v.is_nan()).count();
            let sorted_pos = choose(nan_count, lane.len());
            let orig = order[sorted_pos];
            let out = o * layout.inner + i;
            values[out] = lane[orig];
            indices[out] = to_i32(orig)?;
        }
    }
    Ok((values, indices))
}

/// `median`（全要素。値のみ）。要素数 0 は NaN（PyTorch 2.14.0 の実測）。
/// `layout` は `stat_layout(shape, None)` で作ったものを渡す。
pub fn median_all_host(x: &[f32], layout: &StatLayout) -> Result<f32, StatReduceError> {
    layout.check_len(x.len())?;
    if layout.axis_len == 0 {
        return Ok(f32::NAN);
    }
    if x.iter().any(|v| v.is_nan()) {
        return Ok(f32::NAN);
    }
    let mut order = Vec::with_capacity(x.len());
    sort_order(x, &mut order);
    Ok(x[order[(x.len() - 1) / 2]])
}

/// [`median_all_host`] の VJP。中央値と**等しい要素（値の等価 `==`。`-0.0` と `0.0` は
/// 同じ組）**へ upstream を均等分配する（`g / 個数`）。中央値が NaN のときは NaN 要素へ
/// 均等分配する（PyTorch 2.14.0 の実測）。軸指定版（1 要素へ全量）と規則が異なる。
pub fn median_all_vjp_host(
    x: &[f32],
    upstream: &[f32],
    layout: &StatLayout,
) -> Result<Vec<f32>, StatReduceError> {
    layout.check_len(x.len())?;
    layout.check_out_len(upstream.len())?;
    if layout.numel == 0 {
        return Ok(Vec::new());
    }
    let m = median_all_host(x, layout)?;
    let matches = |v: f32| if m.is_nan() { v.is_nan() } else { v == m };
    let count = x.iter().filter(|&&v| matches(v)).count();
    let share = f64::from(upstream[0]) / count as f64;
    Ok(x.iter()
        .map(|&v| if matches(v) { share as f32 } else { 0.0 })
        .collect())
}

fn validate_q(q: f32) -> Result<(), StatReduceError> {
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(StatReduceError::InvalidArgument(format!(
            "quantile: q must be in [0, 1] and finite, got {q}"
        )));
    }
    Ok(())
}

/// 1 lane の `quantile` の選択位置: `(下側ソート位置, 上側ソート位置, 重み)`。
/// rank は `f64` で `f64::from(q) * (n - 1)`（PyTorch 2.14.0 の実測。`f32` の積に丸めると
/// `q = 0.1`・`n = 11` の `Higher` などで選ぶ要素がずれる）。位置は `n - 1` へクランプする。
fn quantile_positions(q: f32, n: usize, interp: QuantileInterpolation) -> (usize, usize, f64) {
    let rank = f64::from(q) * ((n - 1) as f64);
    let lo = (rank.floor() as usize).min(n - 1);
    let hi = (rank.ceil() as usize).min(n - 1);
    let w = rank - rank.floor();
    match interp {
        QuantileInterpolation::Linear => (lo, hi, w),
        QuantileInterpolation::Lower => (lo, lo, 0.0),
        QuantileInterpolation::Higher => (hi, hi, 0.0),
        QuantileInterpolation::Midpoint => (lo, hi, 0.5),
        QuantileInterpolation::Nearest => {
            let r = (rank.round_ties_even() as usize).min(n - 1);
            (r, r, 0.0)
        }
    }
}

/// PyTorch `lerp` と同じ 2 分岐（`w < 0.5` なら `a + w(b-a)`、それ以外は `b - (b-a)(1-w)`）。
fn lerp(a: f64, b: f64, w: f64) -> f64 {
    if w < 0.5 {
        a + w * (b - a)
    } else {
        b - (b - a) * (1.0 - w)
    }
}

fn quantile_precheck(x: &[f32], layout: &StatLayout, q: f32) -> Result<(), StatReduceError> {
    validate_q(q)?;
    layout.check_len(x.len())?;
    require_nonempty_lane(layout, "quantile")
}

/// `quantile`（スカラー `q`。`torch.quantile` 相当）。lane に NaN があれば NaN。
///
/// `q` が NaN／`[0, 1]` 外・空 lane は
/// [`StatReduceError::InvalidArgument`]。
pub fn quantile_host(
    x: &[f32],
    layout: &StatLayout,
    q: f32,
    interp: QuantileInterpolation,
) -> Result<Vec<f32>, StatReduceError> {
    quantile_precheck(x, layout, q)?;
    let mut out = vec![0.0_f32; layout.out_numel()];
    let mut lane = Vec::with_capacity(layout.axis_len);
    let mut order = Vec::with_capacity(layout.axis_len);
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            layout.gather(x, o, i, &mut lane);
            let slot = o * layout.inner + i;
            if lane.iter().any(|v| v.is_nan()) {
                out[slot] = f32::NAN;
                continue;
            }
            sort_order(&lane, &mut order);
            let (lo, hi, w) = quantile_positions(q, lane.len(), interp);
            let a = f64::from(lane[order[lo]]);
            let discrete = matches!(
                interp,
                QuantileInterpolation::Lower
                    | QuantileInterpolation::Higher
                    | QuantileInterpolation::Nearest
            );
            let value = if discrete {
                // 離散選択（`Lower`／`Higher`／`Nearest`・rank が整数）は算術を通さない。
                lane[order[lo]]
            } else {
                lerp(a, f64::from(lane[order[hi]]), w) as f32
            };
            out[slot] = value;
        }
    }
    Ok(out)
}

/// [`quantile_host`] の VJP。下側へ `g·(1-w)`・上側へ `g·w`（`Lower`／`Higher`／
/// `Nearest` は選択 1 要素へ `g`・`Midpoint` は `0.5` ずつ）。`lo == hi` のときは同じ
/// 要素へ加算する。NaN を含む lane は、安定順で最後の NaN 要素へ `g` を流す（本実装の
/// 独自契約。NaN が 1 つなら PyTorch と同じ）。入力から lane を再ソートして導出する。
pub fn quantile_vjp_host(
    x: &[f32],
    upstream: &[f32],
    layout: &StatLayout,
    q: f32,
    interp: QuantileInterpolation,
) -> Result<Vec<f32>, StatReduceError> {
    quantile_precheck(x, layout, q)?;
    layout.check_out_len(upstream.len())?;
    let mut d_x = vec![0.0_f32; layout.numel];
    let mut lane = Vec::with_capacity(layout.axis_len);
    let mut order = Vec::with_capacity(layout.axis_len);
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            layout.gather(x, o, i, &mut lane);
            sort_order(&lane, &mut order);
            let g = f64::from(upstream[o * layout.inner + i]);
            let n = lane.len();
            if lane.iter().any(|v| v.is_nan()) {
                d_x[layout.pos(o, order[n - 1], i)] = g as f32;
                continue;
            }
            let (lo, hi, w) = quantile_positions(q, n, interp);
            if lo == hi {
                // 単一要素選択は g を直接代入する（g=inf・w=0 で inf + inf*0 = NaN を避ける）。
                d_x[layout.pos(o, order[lo], i)] = g as f32;
            } else {
                d_x[layout.pos(o, order[lo], i)] = (g * (1.0 - w)) as f32;
                d_x[layout.pos(o, order[hi], i)] = (g * w) as f32;
            }
        }
    }
    Ok(d_x)
}

/// `nansum`／`nanmean` の共通走査。`f64` アキュムレータへ非 NaN 要素を添字昇順に
/// 逐次加算し、`finish(和, 個数)` を 1 回だけ `f32` へ downcast する。
fn nan_reduce(
    x: &[f32],
    layout: &StatLayout,
    finish: impl Fn(f64, usize) -> f64,
) -> Result<Vec<f32>, StatReduceError> {
    layout.check_len(x.len())?;
    let mut out = vec![0.0_f32; layout.out_numel()];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            let mut acc = 0.0_f64;
            let mut count = 0_usize;
            for a in 0..layout.axis_len {
                let v = x[layout.pos(o, a, i)];
                if !v.is_nan() {
                    acc += f64::from(v);
                    count += 1;
                }
            }
            out[o * layout.inner + i] = finish(acc, count) as f32;
        }
    }
    Ok(out)
}

/// `nansum`（NaN を 0 とみなした和。全 NaN・空 lane は `0.0`）。
pub fn nansum_host(x: &[f32], layout: &StatLayout) -> Result<Vec<f32>, StatReduceError> {
    nan_reduce(x, layout, |sum, _| sum)
}

/// `nanmean`（非 NaN の和 ÷ 非 NaN の個数。全 NaN・空 lane は NaN）。
pub fn nanmean_host(x: &[f32], layout: &StatLayout) -> Result<Vec<f32>, StatReduceError> {
    nan_reduce(x, layout, |sum, count| sum / count as f64)
}

/// `nansum`（`mean = false`）／`nanmean`（`mean = true`）の VJP。非 NaN 位置へ
/// `g`（`nanmean` は `g / 個数`）、NaN 位置へ `0 × (その値)` を流す。PyTorch 2.14.0 は
/// **乗算**で NaN 位置を 0 にするため、upstream が非有限（`inf`／NaN）や全 NaN lane
/// （`g / 0`）では NaN 位置が NaN になる（実測。決定記録 §5）。`f64` で計算し 1 回だけ
/// downcast する。
pub fn nan_reduce_vjp_host(
    x: &[f32],
    upstream: &[f32],
    layout: &StatLayout,
    mean: bool,
) -> Result<Vec<f32>, StatReduceError> {
    layout.check_len(x.len())?;
    layout.check_out_len(upstream.len())?;
    let mut d_x = vec![0.0_f32; layout.numel];
    for o in 0..layout.outer {
        for i in 0..layout.inner {
            let count = (0..layout.axis_len)
                .filter(|&a| !x[layout.pos(o, a, i)].is_nan())
                .count();
            let g = f64::from(upstream[o * layout.inner + i]);
            let t = if mean { g / count as f64 } else { g };
            for a in 0..layout.axis_len {
                let p = layout.pos(o, a, i);
                d_x[p] = if x[p].is_nan() {
                    (0.0 * t) as f32
                } else {
                    t as f32
                };
            }
        }
    }
    Ok(d_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lay(shape: &[usize], dim: Option<usize>) -> StatLayout {
        stat_layout(shape, dim).expect("test layout")
    }

    #[test]
    fn median_is_lower_median_and_stable_for_ties() {
        let l = lay(&[4], Some(0));
        let (v, i) = median_dim_host(&[4.0, 1.0, 3.0, 2.0], &l).unwrap();
        assert_eq!((v, i), (vec![2.0], vec![3]));
        // タイは元添字昇順（安定順）: [2,1,2,2,3] の昇順は 1,2(1),2(2),2(3),3 → 位置 2 は添字 2。
        let l = lay(&[5], Some(0));
        let (v, i) = median_dim_host(&[2.0, 1.0, 2.0, 2.0, 3.0], &l).unwrap();
        assert_eq!((v, i), (vec![2.0], vec![2]));
        let (v, i) = kthvalue_host(&[2.0, 1.0, 2.0, 2.0, 3.0], &l, 2).unwrap();
        assert_eq!((v, i), (vec![2.0], vec![0]));
    }

    #[test]
    fn nan_rules() {
        let l = lay(&[5], Some(0));
        let x = [3.0, f32::NAN, 1.0, 2.0, f32::NAN];
        let (v, i) = median_dim_host(&x, &l).unwrap();
        assert!(v[0].is_nan());
        assert_eq!(i, vec![1]); // 最初の NaN
        assert!(median_all_host(&x, &lay(&[5], None)).unwrap().is_nan());
        // kthvalue は NaN を最大として並べるだけ。
        let (v, _) = kthvalue_host(&x, &l, 3).unwrap();
        assert_eq!(v, vec![3.0]);
        let (v, i) = kthvalue_host(&x, &l, 4).unwrap();
        assert!(v[0].is_nan());
        assert_eq!(i, vec![1]);
        let q = quantile_host(&x, &l, 0.5, QuantileInterpolation::Linear).unwrap();
        assert!(q[0].is_nan());
    }

    #[test]
    fn signed_zero_is_tie() {
        let l = lay(&[3], Some(0));
        let (v, i) = median_dim_host(&[0.0, -0.0, 0.0], &l).unwrap();
        assert_eq!(i, vec![1]);
        assert_eq!(v[0].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn non_last_dim_and_dim_none() {
        // [[1,5],[3,2],[2,9]] を dim 0 で中央値: 列 0 = {1,3,2} → 2、列 1 = {5,2,9} → 5。
        let x = [1.0, 5.0, 3.0, 2.0, 2.0, 9.0];
        let (v, i) = median_dim_host(&x, &lay(&[3, 2], Some(0))).unwrap();
        assert_eq!((v, i), (vec![2.0, 5.0], vec![2, 0]));
        assert_eq!(median_all_host(&x, &lay(&[3, 2], None)).unwrap(), 2.0);
        let (v, i) = kthvalue_host(&x, &lay(&[3, 2], Some(1)), 2).unwrap();
        assert_eq!((v, i), (vec![5.0, 3.0, 9.0], vec![1, 0, 1]));
    }

    #[test]
    fn single_element_axis() {
        let (v, i) = median_dim_host(&[7.0, 8.0], &lay(&[2, 1], Some(1))).unwrap();
        assert_eq!((v, i), (vec![7.0, 8.0], vec![0, 0]));
    }

    #[test]
    fn empty_handling() {
        // 空軸: 順序統計は拒否、nansum は 0、nanmean は NaN、median_all は NaN。
        let l = lay(&[0], Some(0));
        assert!(matches!(
            median_dim_host(&[], &l),
            Err(StatReduceError::InvalidArgument(_))
        ));
        assert!(matches!(
            kthvalue_host(&[], &l, 1),
            Err(StatReduceError::InvalidArgument(_))
        ));
        assert!(matches!(
            quantile_host(&[], &l, 0.5, QuantileInterpolation::Linear),
            Err(StatReduceError::InvalidArgument(_))
        ));
        assert_eq!(nansum_host(&[], &l).unwrap(), vec![0.0]);
        assert!(nanmean_host(&[], &l).unwrap()[0].is_nan());
        assert!(median_all_host(&[], &lay(&[0], None)).unwrap().is_nan());
        let l = lay(&[0, 2], Some(0));
        assert_eq!(nansum_host(&[], &l).unwrap(), vec![0.0, 0.0]);
        // 他の軸が 0: 出力も空。
        let l = lay(&[0, 3], Some(1));
        assert_eq!(l.out_numel(), 0);
        assert_eq!(nansum_host(&[], &l).unwrap(), Vec::<f32>::new());
        assert_eq!(median_dim_host(&[], &l).unwrap().0, Vec::<f32>::new());
    }

    #[test]
    fn k_and_q_validation() {
        let l = lay(&[3], Some(0));
        let x = [1.0, 2.0, 3.0];
        assert!(kthvalue_host(&x, &l, 0).is_err());
        assert!(kthvalue_host(&x, &l, 4).is_err());
        for q in [-0.1_f32, 1.1, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    quantile_host(&x, &l, q, QuantileInterpolation::Linear),
                    Err(StatReduceError::InvalidArgument(_))
                ),
                "q={q}"
            );
        }
    }

    #[test]
    fn quantile_interpolations_and_endpoints() {
        let l = lay(&[6], Some(0));
        let x = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0]; // 昇順 1,2,3,5,8,9
        let q = |interp| quantile_host(&x, &l, 0.3, interp).unwrap()[0];
        assert_eq!(q(QuantileInterpolation::Linear), 2.5);
        assert_eq!(q(QuantileInterpolation::Lower), 2.0);
        assert_eq!(q(QuantileInterpolation::Higher), 3.0);
        assert_eq!(q(QuantileInterpolation::Midpoint), 2.5);
        assert_eq!(q(QuantileInterpolation::Nearest), 3.0);
        for interp in [
            QuantileInterpolation::Linear,
            QuantileInterpolation::Lower,
            QuantileInterpolation::Higher,
            QuantileInterpolation::Midpoint,
            QuantileInterpolation::Nearest,
        ] {
            let lo = quantile_host(&x, &l, 0.0, interp).unwrap()[0];
            let hi = quantile_host(&x, &l, 1.0, interp).unwrap()[0];
            assert_eq!(lo.to_bits(), 1.0_f32.to_bits());
            assert_eq!(hi.to_bits(), 9.0_f32.to_bits());
        }
        // Nearest の `.5` は偶数丸め（rank 0.5 → 0・1.5 → 2・2.5 → 2・3.5 → 4）。
        let l5 = lay(&[5], Some(0));
        let y = [0.0, 1.0, 2.0, 3.0, 4.0];
        let near = |q| quantile_host(&y, &l5, q, QuantileInterpolation::Nearest).unwrap()[0];
        assert_eq!(
            [near(0.125), near(0.375), near(0.625), near(0.875)],
            [0.0, 2.0, 2.0, 4.0]
        );
    }

    #[test]
    fn quantile_rank_uses_f64() {
        // n = 11, q = 0.1: f64 の rank は 1.0000000149 で `Higher` は 2（f32 の積なら 1.0
        // ちょうどで 1 になる）。PyTorch 2.14.0 の実測（`Higher` → 2.0）。
        let x: Vec<f32> = (0..11).map(|v| v as f32).collect();
        let l = lay(&[11], Some(0));
        let v = quantile_host(&x, &l, 0.1, QuantileInterpolation::Higher).unwrap()[0];
        assert_eq!(v, 2.0);
        // n = 11, q = 0.7: rank 6.99999988 → `Lower` は 6。
        let v = quantile_host(&x, &l, 0.7, QuantileInterpolation::Lower).unwrap()[0];
        assert_eq!(v, 6.0);
        // n = 6, q = 0.1: rank 0.5000000075 は `.5` を超えるため `Nearest` は 1（偶数丸めの 0 ではない）。
        let y: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let v =
            quantile_host(&y, &lay(&[6], Some(0)), 0.1, QuantileInterpolation::Nearest).unwrap()[0];
        assert_eq!(v, 1.0);
    }

    #[test]
    fn nan_reductions_use_f64_accumulator() {
        // f32 逐次和では 1e8 + 1 - 1e8 が 0 になる並び。f64 なら 1。
        let x = [1.0e8_f32, 1.0, f32::NAN, -1.0e8];
        let l = lay(&[4], None);
        assert_eq!(nansum_host(&x, &l).unwrap(), vec![1.0]);
        let m = nanmean_host(&x, &l).unwrap()[0];
        assert_eq!(m.to_bits(), ((1.0_f64 / 3.0) as f32).to_bits());
        assert!(nanmean_host(&[f32::NAN, f32::NAN], &lay(&[2], None)).unwrap()[0].is_nan());
        assert_eq!(
            nansum_host(&[f32::NAN], &lay(&[1], None)).unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn layout_errors_do_not_panic_or_allocate() {
        assert!(matches!(
            stat_layout(&[usize::MAX, usize::MAX, 0], Some(2)),
            Err(ShapeError::ElementCountOverflow)
        ));
        assert!(matches!(
            stat_layout(&[usize::MAX, 2], Some(0)),
            Err(ShapeError::ElementCountOverflow)
        ));
        assert!(matches!(
            stat_layout(&[2], Some(1)),
            Err(ShapeError::AxisOutOfRange { .. })
        ));
        assert!(matches!(
            stat_layout(&[], Some(0)),
            Err(ShapeError::AxisOutOfRange { .. })
        ));
        assert!(stat_layout(&[], None).is_ok());
        // 空入力でも出力バイト数・作業バッファを確保前に検査する。
        assert!(matches!(
            stat_layout(&[0, usize::MAX], Some(0)),
            Err(ShapeError::ElementCountOverflow)
        ));
        assert!(matches!(
            stat_layout(&[0, usize::MAX], Some(1)),
            Err(ShapeError::ElementCountOverflow)
        ));
        let l = stat_layout(&[usize::MAX / 8], Some(0));
        assert!(matches!(l, Err(ShapeError::ElementCountOverflow)));
    }

    #[test]
    fn index_range_and_slice_length_checks() {
        let l = lay(&[3], Some(0));
        assert!(matches!(
            median_dim_host(&[1.0, 2.0], &l),
            Err(StatReduceError::Shape(
                ShapeError::ElementCountMismatch { .. }
            ))
        ));
        assert!(matches!(
            nansum_host(&[1.0], &l),
            Err(StatReduceError::Shape(
                ShapeError::ElementCountMismatch { .. }
            ))
        ));
        let big = StatLayout {
            outer: 1,
            axis_len: i32::MAX as usize + 2,
            inner: 1,
            numel: 0,
            shape: vec![],
            out_shape: vec![],
        };
        assert!(big.check_i32_indices().is_err());
    }

    #[test]
    fn median_all_vjp_distributes_over_equal_elements() {
        let l = lay(&[5], None);
        let g = median_all_vjp_host(&[1.0, 2.0, 2.0, 2.0, 3.0], &[3.0], &l).unwrap();
        assert_eq!(g, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
        // ±0 は同じ組（== 判定）。
        let g =
            median_all_vjp_host(&[0.0, -0.0, 1.0, -0.0, 0.0], &[1.0], &lay(&[5], None)).unwrap();
        assert_eq!(g, vec![0.25, 0.25, 0.0, 0.25, 0.25]);
        // NaN 中央値は NaN 要素へ均等分配。
        let g = median_all_vjp_host(&[1.0, f32::NAN, f32::NAN], &[1.0], &lay(&[3], None)).unwrap();
        assert_eq!(g, vec![0.0, 0.5, 0.5]);
    }

    #[test]
    fn quantile_vjp_hand_values() {
        let l = lay(&[6], Some(0));
        let x = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0];
        // `f64::from(0.3_f32) * 5 = 1.50000006` のため重みは 0.5 ちょうどではない。
        let g = quantile_vjp_host(&x, &[1.0], &l, 0.3, QuantileInterpolation::Linear).unwrap();
        let want = [0.0_f32, 0.0, 0.5, 0.0, 0.5, 0.0];
        for (a, e) in g.iter().zip(want) {
            assert!((a - e).abs() < 1e-6, "{g:?}");
        }
        // 重みが 0.5 ちょうどになる q（rank 2.5）では厳密に一致する。
        let g = quantile_vjp_host(&x, &[1.0], &l, 0.5, QuantileInterpolation::Linear).unwrap();
        assert_eq!(g, vec![0.0, 0.5, 0.0, 0.0, 0.5, 0.0]);
        let g = quantile_vjp_host(&x, &[2.0], &l, 0.0, QuantileInterpolation::Linear).unwrap();
        assert_eq!(g, vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let g = quantile_vjp_host(&x, &[2.0], &l, 0.3, QuantileInterpolation::Higher).unwrap();
        assert_eq!(g, vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        // 単一要素選択で g=inf でも NaN にならない（inf + inf*0 を避ける）。
        let g = quantile_vjp_host(&x, &[f32::INFINITY], &l, 0.0, QuantileInterpolation::Linear)
            .unwrap();
        assert_eq!(g[0], f32::INFINITY);
        assert!(g.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn nan_reduce_vjp_rules() {
        let l = lay(&[3], None);
        let x = [f32::NAN, 2.0, 4.0];
        assert_eq!(
            nan_reduce_vjp_host(&x, &[3.0], &l, false).unwrap(),
            vec![0.0, 3.0, 3.0]
        );
        assert_eq!(
            nan_reduce_vjp_host(&x, &[3.0], &l, true).unwrap(),
            vec![0.0, 1.5, 1.5]
        );
        // 非有限の upstream・全 NaN lane では NaN 位置が NaN になる（乗算）。
        let g = nan_reduce_vjp_host(&x, &[f32::INFINITY], &l, false).unwrap();
        assert!(g[0].is_nan() && g[1] == f32::INFINITY);
        let g = nan_reduce_vjp_host(&[f32::NAN, f32::NAN], &[1.0], &lay(&[2], None), true).unwrap();
        assert!(g.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn deterministic_run_to_run() {
        let x: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 17) as f32) * 0.3 - 2.0)
            .collect();
        let l = lay(&[8, 8], Some(1));
        let a = quantile_host(&x, &l, 0.37, QuantileInterpolation::Linear).unwrap();
        let b = quantile_host(&x, &l, 0.37, QuantileInterpolation::Linear).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let g1 = nan_reduce_vjp_host(&x, &[0.7; 8], &l, true).unwrap();
        let g2 = nan_reduce_vjp_host(&x, &[0.7; 8], &l, true).unwrap();
        assert_eq!(g1, g2);
    }
}
