//! `median`・`kthvalue`・`quantile`・`nanmean`・`nansum` の自由関数（イシュー
//! #2637・親 #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::median` 等の委譲メソッドと引数型
//! `QuantileInterpolation` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は
//! 承認後の #2678）。本モジュールは内部クレート限定の入口で、`Var` に inherent
//! メソッドを足さない。保留は `crates/facade/src/lib.rs` の
//! `StatReduceOpsHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の
//! 否定ガードが機械的に固定する（`docs/autodiff-stat-reduce-ops-decision.md`）。
//!
//! **PyTorch 相当**:
//!
//! | 関数 | PyTorch 相当 | 出力 |
//! |---|---|---|
//! | [`median`]（`dim = None`） | `torch.median(x)` | 0 次元の値 `Var` |
//! | [`median`]（`dim = Some(d)`） | `torch.median(x, d).values` | 縮約後 shape の値 `Var` |
//! | [`median_with_indices`] | `torch.median(x, d)` | 値 `Var` と索引 `Tensor<i32>` |
//! | [`kthvalue`] | `torch.kthvalue(x, k, d)`（`k` は 1 始まり） | 値 `Var` と索引 |
//! | [`quantile`] | `torch.quantile(x, q, dim, interpolation=…)`（スカラー `q`） | 値 `Var` |
//! | [`nansum`] | `torch.nansum(x[, dim])` | 値 `Var` |
//! | [`nanmean`] | `torch.nanmean(x[, dim])` | 値 `Var` |
//!
//! `dim` は `usize`（`Var::sum` と同じく負の添字は受けず、`keepdim` もなく軸を落とす）。
//! 索引は既存慣例（`Var::sort`／`topk`）に合わせ `Tensor<i32>` で、勾配は流れない。
//!
//! **経路**: ① `dim`・形状・確保サイズ・索引上限・`k`／`q` の検査
//! （`fandhe_ai_tensor_core::stat_reduce::stat_layout`。実体化より前）→ ② 入力の実体化 →
//! ③ `BackendOps::stat_*`（`Unsupported` のときだけ共有ホストカーネル
//! `stat_reduce::*_host` へフォールバックし、他のエラーは伝播する。戻り値 shape も
//! 検証する）→ ④ 専用 `Op`（`Op::OrderSelect`／`Op::MedianAll`／`Op::Quantile`／
//! `Op::Nansum`／`Op::Nanmean`）を積む。VJP は `grad.rs`。
//!
//! **数値契約・順序規則**: 規則の正は `fandhe_ai_tensor_core::stat_reduce`
//! （安定昇順・NaN 最大・下側中央値・`f64` アキュムレータ・`quantile` の `f64` rank）。
//! 非有限入力は拒否せず伝播する。高階微分（`create_graph`）・activation
//! checkpoint・f64 自動微分経路・`nanmedian`／`nanquantile`・1 次元 `q` は対象外。
//!
//! **公開状況（イシュー #2678）**: 承認形どおり公開済み: `Var::{median,median_with_indices,kthvalue,quantile,nanmean,nansum}` と `QuantileInterpolation`（クレートルート）。本モジュール自体は facade から再エクスポートしない。
//! 上の「未承認」「保留」「承認依頼は #2677」の記述は #2677 時点のもので、承認形の公開は #2678 で行った
//! （ルート #2499 の承認コメント issuecomment-6033824965・`docs/compat-api-scope.md` §5.1）。

use fandhe_ai_tensor_core::stat_reduce::{self, QuantileInterpolation, StatLayout};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

fn tensor_of<T: fandhe_ai_tensor_core::Element>(
    data: Vec<T>,
    layout: &StatLayout,
) -> Result<Tensor<T>, AutodiffError> {
    Tensor::new(data, layout.out_shape()).map_err(AutodiffError::Shape)
}

/// 値のみを返す演算の共通後半。バックエンドの戻り値を検証し、`Unsupported` のときだけ
/// `host` で計算し、専用 `Op` を 1 ノード積む。
fn finish_value<'t>(
    x: &Var<'t>,
    layout: &StatLayout,
    backend: Result<Tensor<f32>, BackendError>,
    host: impl FnOnce() -> Result<Vec<f32>, AutodiffError>,
    node: Op,
) -> Result<Var<'t>, AutodiffError> {
    let value = match backend {
        Ok(v) => {
            verify_shape(v.shape(), layout.out_shape())?;
            v
        }
        Err(BackendError::Unsupported(_)) => tensor_of(host()?, layout)?,
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = x.tape().push_eager(node, value);
    Ok(Var::from_raw(x.tape(), id))
}

/// 値と索引を返す演算（`kthvalue`／軸指定 `median`）の共通後半。
fn finish_selected<'t>(
    x: &Var<'t>,
    layout: &StatLayout,
    dim: usize,
    backend: Result<(Tensor<f32>, Tensor<i32>), BackendError>,
    host: impl FnOnce() -> Result<(Vec<f32>, Vec<i32>), AutodiffError>,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let (value, index) = match backend {
        Ok((v, i)) => {
            verify_shape(v.shape(), layout.out_shape())?;
            verify_shape(i.shape(), layout.out_shape())?;
            (v, i)
        }
        Err(BackendError::Unsupported(_)) => {
            let (v, i) = host()?;
            (tensor_of(v, layout)?, tensor_of(i, layout)?)
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = x.tape().push_eager(
        Op::OrderSelect {
            input: x.node_id(),
            dim,
            index: index.clone(),
        },
        value,
    );
    Ok((Var::from_raw(x.tape(), id), index))
}

/// `kthvalue`（`k` は 1 始まり。`torch.kthvalue` 相当）。値と索引（`dim` 軸上の位置）を返す。
///
/// `k == 0`・`k > 軸長`・軸長 0・`dim` 範囲外は型付きエラー。
pub fn kthvalue<'t>(
    x: &Var<'t>,
    k: usize,
    dim: usize,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let layout = stat_reduce::stat_layout(&x.shape(), Some(dim)).map_err(AutodiffError::Shape)?;
    layout.check_i32_indices().map_err(AutodiffError::Shape)?;
    if k == 0 || k > layout.axis_len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "kthvalue: k={k} is out of range for an axis of length {}",
            layout.axis_len()
        )));
    }
    let input = materialize_one(x)?;
    let backend = x.tape().ops().stat_kthvalue(&input, k, dim);
    finish_selected(x, &layout, dim, backend, || {
        Ok(stat_reduce::kthvalue_host(
            &input.contiguous().host_slice(),
            &layout,
            k,
        )?)
    })
}

/// 軸指定の下側中央値（`torch.median(x, dim)` 相当）。値と索引を返す。
///
/// 偶数個でも平均せず下側（ソート位置 `(n - 1) / 2`）を返す。lane に NaN があれば値は NaN・
/// 索引は最初の NaN の位置。軸長 0・`dim` 範囲外は型付きエラー。
pub fn median_with_indices<'t>(
    x: &Var<'t>,
    dim: usize,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let layout = stat_reduce::stat_layout(&x.shape(), Some(dim)).map_err(AutodiffError::Shape)?;
    layout.check_i32_indices().map_err(AutodiffError::Shape)?;
    if layout.axis_len() == 0 {
        return Err(AutodiffError::InvalidArgument(
            "median: reduction axis has zero size".into(),
        ));
    }
    let input = materialize_one(x)?;
    let backend = x.tape().ops().stat_median_dim(&input, dim);
    finish_selected(x, &layout, dim, backend, || {
        Ok(stat_reduce::median_dim_host(
            &input.contiguous().host_slice(),
            &layout,
        )?)
    })
}

/// 下側中央値（値のみ）。`dim = None` は全要素（`torch.median(x)`。要素数 0 は NaN。
/// VJP は中央値と等しい要素へ均等分配）、`dim = Some(d)` は
/// `torch.median(x, d).values`（[`median_with_indices`] の値側。VJP は選ばれた 1 要素へ全量）。
pub fn median<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    match dim {
        Some(d) => Ok(median_with_indices(x, d)?.0),
        None => {
            let layout =
                stat_reduce::stat_layout(&x.shape(), None).map_err(AutodiffError::Shape)?;
            let input = materialize_one(x)?;
            let backend = x.tape().ops().stat_median_all(&input);
            finish_value(
                x,
                &layout,
                backend,
                || {
                    let v =
                        stat_reduce::median_all_host(&input.contiguous().host_slice(), &layout)?;
                    Ok(vec![v])
                },
                Op::MedianAll { input: x.node_id() },
            )
        }
    }
}

/// スカラー `q` の分位数（`torch.quantile` 相当）。`dim = None` は全要素。
///
/// `q` が NaN／`[0, 1]` 外・空 lane・`dim` 範囲外は型付きエラー
/// （実体化・確保より前）。lane に NaN があれば NaN。
pub fn quantile<'t>(
    x: &Var<'t>,
    q: f32,
    dim: Option<usize>,
    interpolation: QuantileInterpolation,
) -> Result<Var<'t>, AutodiffError> {
    let layout = stat_reduce::stat_layout(&x.shape(), dim).map_err(AutodiffError::Shape)?;
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(AutodiffError::InvalidArgument(format!(
            "quantile: q must be in [0, 1] and finite, got {q}"
        )));
    }
    if layout.axis_len() == 0 {
        return Err(AutodiffError::InvalidArgument(
            "quantile: reduction axis has zero size".into(),
        ));
    }
    let input = materialize_one(x)?;
    let backend = x.tape().ops().stat_quantile(&input, q, dim, interpolation);
    finish_value(
        x,
        &layout,
        backend,
        || {
            Ok(stat_reduce::quantile_host(
                &input.contiguous().host_slice(),
                &layout,
                q,
                interpolation,
            )?)
        },
        Op::Quantile {
            input: x.node_id(),
            q,
            dim,
            interpolation,
        },
    )
}

/// NaN を 0 とみなした和（`torch.nansum` 相当）。`dim = None` は全要素。全 NaN・空 lane は 0。
pub fn nansum<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let layout = stat_reduce::stat_layout(&x.shape(), dim).map_err(AutodiffError::Shape)?;
    let input = materialize_one(x)?;
    let backend = x.tape().ops().stat_nansum(&input, dim);
    finish_value(
        x,
        &layout,
        backend,
        || {
            Ok(stat_reduce::nansum_host(
                &input.contiguous().host_slice(),
                &layout,
            )?)
        },
        Op::Nansum {
            input: x.node_id(),
            dim,
        },
    )
}

/// NaN を無視した平均（`torch.nanmean` 相当）。`dim = None` は全要素。全 NaN・空 lane は NaN。
/// `f64` のまま `和 ÷ 非 NaN 個数` を計算して 1 回だけ `f32` へ downcast する（丸め 1 回）。
pub fn nanmean<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let layout = stat_reduce::stat_layout(&x.shape(), dim).map_err(AutodiffError::Shape)?;
    let input = materialize_one(x)?;
    let backend = x.tape().ops().stat_nanmean(&input, dim);
    finish_value(
        x,
        &layout,
        backend,
        || {
            Ok(stat_reduce::nanmean_host(
                &input.contiguous().host_slice(),
                &layout,
            )?)
        },
        Op::Nanmean {
            input: x.node_id(),
            dim,
        },
    )
}
