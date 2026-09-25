//! `topk`・`unique` のオプション拡張（イシュー #2153・親 #2131
//! 「5-B 演算」）。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/reduce_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。承認が
//! 取れるまでは `Var` の外に自由関数として置き到達不能にする
//! （`docs/autodiff-topk-unique-ops-decision.md` §0）。
//!
//! **PyTorch 相当・出力型**（詳細は `docs/autodiff-topk-unique-ops-
//! decision.md` §1 の表を参照）:
//!
//! | 関数 | 相当 | 出力 | 微分 |
//! |---|---|---|---|
//! | [`topk_with_options`] | `torch.topk(k, dim, largest, sorted)` | `(Var, Tensor<i32>)` | 可 |
//! | [`unique_with_options`] | `torch.unique(sorted=True, return_inverse, return_counts, dim)` | [`UniqueOutput`]（detached） | 不可 |
//! | [`unique_consecutive`] | `torch.unique_consecutive(return_inverse, return_counts, dim)` | [`UniqueOutput`]（detached） | 不可 |
//!
//! **既存 API 非破壊**: 本モジュールは `Var::topk`（`sorted=True`
//! 固定・非負 `dim` のみ。イシュー #1733）・`Var::unique`（values
//! のみ。イシュー #1734）のシグネチャ・意味論を一切変更しない。
//! [`topk_with_options`] は `sorted=true` の場合、正規化済み `dim` で
//! 既存 `Var::topk` へそのまま委譲する。
//!
//! **`sorted=false` の決定的契約**: PyTorch は `sorted=False` の順序を
//! 未規定とするが、本リポでは 3 バックエンド bit 一致のため
//! 「既存 `topk`（`sorted=True` 相当）で選んだ `k` 個を、`dim` 軸上の
//! 元添字の昇順に並べ替えた順」と定義する（`docs/autodiff-topk-
//! unique-ops-decision.md` §2.3）。`grad::topk_with_fallback` は
//! `largest` に応じた値の大小順で `(values, index)` を返す契約
//! （[`fandhe_ai_tensor_core::BackendOps::topk`] doc）のため、これを
//! そのまま呼んだ後にホスト側で `index` 昇順の置換を `values`／
//! `index` の双方へ適用する（新規 `Op`・新規 `BackendOps` メソッドは
//! 不要——`Op::Topk` の VJP は scatter ベースで index 順序に依存
//! しないため、置換後の組をそのまま記録すれば勾配は正しい）。
//!
//! **unique 拡張の契約**（詳細は `docs/autodiff-topk-unique-ops-
//! decision.md` §2.4）: [`unique_with_options`]・[`unique_consecutive`]
//! は非微分・tape 非記録（既存 `Var::unique` と同型。出力形状が入力値
//! に依存して動的に決まるため）。`unique_with_options` は
//! `dim=None` かつ `return_inverse`／`return_counts` を要求しない
//! 場合のみ既存 `unique_with_fallback`（CUDA／Metal の既存カーネル
//! 経路をそのまま使う）へ委譲し、出力は既存 `Var::unique` と bit
//! 同一になる。それ以外は
//! [`fandhe_ai_tensor_core::BackendOps::unique_ext`] 経由（CUDA／
//! Metal は override しない——既定 `Unsupported` からホスト
//! フォールバックへ到達する。GPU 専用カーネルは別イシュー）。
//!
//! **確保前検査（REQ-8・`.claude/rules/security.md` A03）**: 3 つの
//! 公開入口すべての冒頭・あらゆる分岐（`sorted=true` の委譲経路・
//! `dim=None` かつ追加出力なしの既存 `unique` 委譲経路を含む）や
//! 実体化（`materialize_one`・既存 API への委譲）より前に、共有
//! ヘルパ `ensure_alloc_fits_f32` で入力 shape の要素数積
//! オーバーフロー・`isize::MAX` バイト超過を検査する（検査を迂回する
//! 分岐を作らない）。`unique` 系はさらに対象要素数
//! （`inverse`／`counts` の `i32` 上限）を dispatch 前に検査する。

use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor, topk_out_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::grad::{topk_with_fallback, unique_ext_with_fallback, unique_with_fallback};
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// [`topk_with_options`] のオプション（`torch.topk` の `dim`／
/// `largest`／`sorted` 引数相当）。PyTorch 既定（`dim=-1`・
/// `largest=true`・`sorted=true`）を [`Default`] とする。将来の
/// 非破壊拡張に備え `#[non_exhaustive]`・`with_*` ビルダ方式を採る。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TopkOptions {
    /// 対象軸（負値は末尾からの相対指定。`torch.topk` の `dim` 相当）。
    pub dim: isize,
    /// `true` なら上位 `k` 個、`false` なら下位 `k` 個。
    pub largest: bool,
    /// `true` なら（値の大小順で）ソート済み、`false` なら
    /// [モジュール doc]「`sorted=false` の決定的契約」を参照。
    pub sorted: bool,
}

impl Default for TopkOptions {
    fn default() -> Self {
        Self {
            dim: -1,
            largest: true,
            sorted: true,
        }
    }
}

impl TopkOptions {
    /// 対象軸を設定する（`torch.topk` の `dim` 相当。負値可）。
    pub fn with_dim(mut self, dim: isize) -> Self {
        self.dim = dim;
        self
    }
    /// 上位／下位のいずれを選ぶかを設定する。
    pub fn with_largest(mut self, largest: bool) -> Self {
        self.largest = largest;
        self
    }
    /// ソート有無を設定する。
    pub fn with_sorted(mut self, sorted: bool) -> Self {
        self.sorted = sorted;
        self
    }
}

/// [`unique_with_options`]／[`unique_consecutive`] のオプション
/// （`torch.unique`／`torch.unique_consecutive` の `dim`／
/// `return_inverse`／`return_counts` 引数相当）。既定（`dim=None`・
/// 追加出力なし）を [`Default`] とする。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct UniqueOptions {
    /// 対象軸（`None` は入力全体を平坦化。負値は末尾からの相対指定）。
    pub dim: Option<isize>,
    /// `true` なら [`UniqueOutput::inverse`] を計算する。
    pub return_inverse: bool,
    /// `true` なら [`UniqueOutput::counts`] を計算する。
    pub return_counts: bool,
}

impl UniqueOptions {
    /// 対象軸を設定する（`torch.unique`／`torch.unique_consecutive` の
    /// `dim` 相当。負値可）。
    pub fn with_dim(mut self, dim: isize) -> Self {
        self.dim = Some(dim);
        self
    }
    /// `return_inverse` を設定する。
    pub fn with_return_inverse(mut self, return_inverse: bool) -> Self {
        self.return_inverse = return_inverse;
        self
    }
    /// `return_counts` を設定する。
    pub fn with_return_counts(mut self, return_counts: bool) -> Self {
        self.return_counts = return_counts;
        self
    }
}

/// [`unique_with_options`]／[`unique_consecutive`] の戻り値。要求
/// されなかった出力（`return_inverse`／`return_counts` が `false`）は
/// `None`。**非微分・detached**（`Var::unique` と同型の理由。モジュール
/// doc 参照）。
#[derive(Debug, Clone)]
pub struct UniqueOutput {
    /// 一意値集合（`sorted=True`）。
    pub values: Tensor<f32>,
    /// 各入力要素（`dim` 指定時は各スライス）が属する `values` の
    /// 添字（`return_inverse=true` のときのみ `Some`）。
    pub inverse: Option<Tensor<i32>>,
    /// `values` の各要素（スライス）の出現回数
    /// （`return_counts=true` のときのみ `Some`）。
    pub counts: Option<Tensor<i32>>,
}

/// 負 dim を正規化する（`torch` 慣習: `[-rank, rank)` の範囲で末尾
/// からの相対指定を許す）。範囲外は `dim >= rank`（非負）なら
/// [`fandhe_ai_tensor_core::ops_shape::topk_out_shape`] 等の既存軸
/// 検査と同じ `AutodiffError::Shape(ShapeError::AxisOutOfRange)`、
/// 負で範囲外（`usize` で表現できない）は `AutodiffError::
/// InvalidArgument` を返す。`rank == 0` は常に拒否する（`Var::topk`
/// の既存 `topk_out_shape` と同じく 0-d 入力は対象外——PyTorch は
/// 0-d を許容するが本リポの既存契約に合わせる。`docs/autodiff-
/// topk-unique-ops-decision.md` §3「PyTorch との差分」）。
fn normalize_dim(dim: isize, rank: usize) -> Result<usize, AutodiffError> {
    if dim >= 0 {
        let d = dim as usize;
        if d >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: d,
                rank,
            }));
        }
        Ok(d)
    } else {
        let steps = match dim.checked_neg() {
            Some(v) if v > 0 => v as usize,
            _ => {
                return Err(AutodiffError::InvalidArgument(format!(
                    "topk_unique_ops::normalize_dim: dim={dim} は許容範囲 \
                     [-{rank}, {rank}) を外れている"
                )));
            }
        };
        if steps > rank {
            return Err(AutodiffError::InvalidArgument(format!(
                "topk_unique_ops::normalize_dim: dim={dim} は許容範囲 \
                 [-{rank}, {rank}) を外れている"
            )));
        }
        Ok(rank - steps)
    }
}

/// `rank == 0`（0-d スカラー）入力を拒否する（`docs/autodiff-
/// topk-unique-ops-decision.md` §3「PyTorch との差分」：「`rank == 0`
/// （0-d）入力の `topk`／`unique` 拡張はすべて拒否する」契約の
/// 単一実装）。
///
/// [`normalize_dim`] は `dim` 引数を受け取る経路（`dim=Some(_)`）では
/// `d >= rank`（`rank == 0` なら常に真）により自然に 0-d を拒否するが、
/// `unique_with_options`／`unique_consecutive` は `dim=None`（軸非指定・
/// 平坦化）を許す API であり、その経路は `normalize_dim` を一切呼ばない
/// ため 0-d 拒否契約から漏れる（`unique_with_options` の
/// `dim=None && !return_inverse && !return_counts` 早期委譲分岐は
/// なおさら——既存 `Var::unique`〈#1734〉と bit 同一にするため
/// `materialize_one` へ直行し軸検査を一切経由しない）。codex-review
/// P2 是正（PR #2270・イシュー #2153）: 両公開入口の冒頭・
/// `dim` 分岐より前で本関数を呼び、`dim` の有無に関わらず 0-d 入力を
/// 一律拒否することで契約どおりの動作にする（既存 `Var::unique`
/// 自体〈`unique_with_fallback` が直接ラップする既存 #1734 API〉の
/// 0-d 許容契約は本 PR のスコープ外のため変更しない——変更対象は
/// あくまで本モジュールが追加する `unique_with_options`／
/// `unique_consecutive` の新規公開入口のみ）。
///
/// `topk_with_options` は常に `dim` を要求する API のため
/// `normalize_dim` の自然な拒否のみで契約を満たし、本関数の呼び出しは
/// 不要。
fn reject_rank_zero(rank: usize) -> Result<(), AutodiffError> {
    if rank == 0 {
        Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: 0,
        }))
    } else {
        Ok(())
    }
}

/// 本モジュールの全公開入口（[`topk_with_options`]・
/// [`unique_with_options`]・[`unique_consecutive`]）が冒頭で呼ぶ、
/// 確保前バイト数上限検査ヘルパ（`crate::reduce_ops::
/// ensure_alloc_fits_f32` と同型。同一クレート内で共有はせず個別に
/// 持つ——`out_shape` の意味論がモジュールごとに異なるため独立に
/// 保守する）。
fn ensure_alloc_fits_f32(
    input_shape: &[usize],
    out_shape: Option<&[usize]>,
) -> Result<(), AutodiffError> {
    checked_bytes_for::<f32>(input_shape)?;
    if let Some(out_shape) = out_shape {
        checked_bytes_for::<f32>(out_shape)?;
    }
    Ok(())
}

/// `unique` 系（[`unique_with_options`]・[`unique_consecutive`]）が
/// dispatch 前に検査する、`inverse`／`counts` の `i32` 上限（対象
/// 要素数が `i32::MAX` を超えると添字を表現できない）。
fn ensure_target_len_fits_i32(target_len: usize) -> Result<(), AutodiffError> {
    if target_len > i32::MAX as usize {
        Err(AutodiffError::InvalidArgument(format!(
            "topk_unique_ops: 対象要素数 {target_len} が i32::MAX を超えている \
             （inverse/counts を i32 で表現できない）"
        )))
    } else {
        Ok(())
    }
}

/// `x` を層 1 で実体化した `Tensor<f32>` を返す（`crate::reduce_ops::
/// materialize_one` と同型の複製）。
fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// `dim` 軸に沿って上位／下位 `k` 個を選ぶ（`torch.topk(k, dim,
/// largest, sorted)` 相当。イシュー #2153）。`opts.sorted == true` は
/// 正規化済み `dim` で既存 [`Var::topk`] へそのまま委譲し、挙動・
/// ノード構成とも完全同一。`opts.sorted == false` は
/// [モジュール doc]「`sorted=false` の決定的契約」を参照。
pub fn topk_with_options<'t>(
    x: &Var<'t>,
    k: usize,
    opts: TopkOptions,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let in_shape = x.shape();
    // 確保前検査（あらゆる分岐・委譲より前。モジュール doc 参照）。
    ensure_alloc_fits_f32(&in_shape, None)?;
    let rank = in_shape.len();
    let dim = normalize_dim(opts.dim, rank)?;

    if opts.sorted {
        return x.topk(k, dim, opts.largest);
    }

    let out_shape = topk_out_shape(&in_shape, dim, k).map_err(AutodiffError::Shape)?;
    ensure_alloc_fits_f32(&in_shape, Some(&out_shape))?;

    let input_val = materialize_one(x)?;
    let (value, index) =
        topk_with_fallback(x.tape().ops(), &input_val, dim, k, opts.largest, &out_shape)?;
    if value.shape() != out_shape.as_slice() || index.shape() != out_shape.as_slice() {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: value.shape().to_vec(),
                rhs: out_shape,
            },
        )));
    }

    let (value2, index2) = resort_topk_by_index(&value, &index, dim);
    if value2.shape() != out_shape.as_slice() || index2.shape() != out_shape.as_slice() {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: value2.shape().to_vec(),
                rhs: out_shape,
            },
        )));
    }

    let id = x.tape().push_eager(
        Op::Topk {
            input: x.node_id(),
            dim,
            index: index2.clone(),
        },
        value2,
    );
    Ok((Var::from_raw(x.tape(), id), index2))
}

/// [`topk_with_options`] の `sorted=false` 経路が使うヘルパー:
/// `topk_with_fallback` が返す「`largest` に応じた値の大小順」の
/// `(values, index)` を、`dim` 軸上の各レーンごとに `index` 昇順へ
/// 並べ替える（`values`／`index` の双方へ同じ置換を適用するため、
/// 選択演算〈丸めなし〉として bit 一致が構成的に保たれる）。
fn resort_topk_by_index(
    values: &Tensor<f32>,
    index: &Tensor<i32>,
    dim: usize,
) -> (Tensor<f32>, Tensor<i32>) {
    let shape = values.shape().to_vec();
    let dim_size = shape[dim];
    let numel: usize = shape.iter().product();
    if dim_size == 0 || numel == 0 {
        return (values.clone(), index.clone());
    }
    let data_v = eval::dense_vec(values);
    let data_i = eval::dense_vec_i32(index);
    let strides = eval::row_major_strides(&shape);
    let mut out_v = data_v.clone();
    let mut out_i = data_i.clone();

    for flat in 0..numel {
        let coords = eval::unravel(flat, &shape);
        if coords[dim] != 0 {
            continue;
        }
        let mut lane: Vec<(i32, f32)> = Vec::with_capacity(dim_size);
        for k in 0..dim_size {
            let mut pos = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { k } else { coords[axis] };
                pos += coord * stride;
            }
            lane.push((data_i[pos], data_v[pos]));
        }
        let mut order: Vec<usize> = (0..dim_size).collect();
        order.sort_by_key(|&k| lane[k].0);
        for (out_k, &src_k) in order.iter().enumerate() {
            let mut pos = 0usize;
            for (axis, &stride) in strides.iter().enumerate() {
                let coord = if axis == dim { out_k } else { coords[axis] };
                pos += coord * stride;
            }
            out_v[pos] = lane[src_k].1;
            out_i[pos] = lane[src_k].0;
        }
    }
    (
        eval::build_tensor(out_v, &shape),
        eval::build_index_tensor(out_i, &shape),
    )
}

/// 一意値集合（＋任意で `inverse`／`counts`）を返す（`torch.unique(
/// sorted=True, return_inverse, return_counts, dim)` 相当。イシュー
/// #2153）。`dim=None` かつ追加出力を要求しない場合は既存
/// [`Var::unique`] と同じ経路（`unique_with_fallback`）へ委譲し、
/// `values` は bit 同一になる。**非微分・detached**
/// （[モジュール doc] 参照）。
pub fn unique_with_options<'t>(
    x: &Var<'t>,
    opts: UniqueOptions,
) -> Result<UniqueOutput, AutodiffError> {
    let in_shape = x.shape();
    // 確保前検査（あらゆる分岐・委譲より前。モジュール doc 参照）。
    ensure_alloc_fits_f32(&in_shape, None)?;
    // 0-d 拒否契約（`dim=None` の早期委譲分岐は `normalize_dim` を
    // 経由しないため、ここで明示的に検査する。codex-review P2 是正・
    // `reject_rank_zero` doc 参照）。
    reject_rank_zero(in_shape.len())?;

    if opts.dim.is_none() && !opts.return_inverse && !opts.return_counts {
        let input_val = materialize_one(x)?;
        let values = unique_with_fallback(x.tape().ops(), &input_val)?;
        return Ok(UniqueOutput {
            values,
            inverse: None,
            counts: None,
        });
    }

    let rank = in_shape.len();
    let dim = match opts.dim {
        Some(d) => Some(normalize_dim(d, rank)?),
        None => None,
    };
    let target_len = unique_target_len(&in_shape, dim)?;
    ensure_target_len_fits_i32(target_len)?;

    let input_val = materialize_one(x)?;
    let out = unique_ext_with_fallback(x.tape().ops(), &input_val, dim, false)?;
    Ok(UniqueOutput {
        values: out.values,
        inverse: opts.return_inverse.then_some(out.inverse),
        counts: opts.return_counts.then_some(out.counts),
    })
}

/// 隣接要素（`dim` 指定時は隣接スライス）のみを群化する
/// （`torch.unique_consecutive(return_inverse, return_counts, dim)`
/// 相当。イシュー #2153）。ソートを行わないため常に
/// [`fandhe_ai_tensor_core::BackendOps::unique_ext`] 経由（`unique`
/// の既存カーネルへは委譲しない——`sorted=True` 固定の既存経路とは
/// 意味論が異なるため）。**非微分・detached**（[モジュール doc]
/// 参照）。
pub fn unique_consecutive<'t>(
    x: &Var<'t>,
    opts: UniqueOptions,
) -> Result<UniqueOutput, AutodiffError> {
    let in_shape = x.shape();
    // 確保前検査（あらゆる分岐・委譲より前。モジュール doc 参照）。
    ensure_alloc_fits_f32(&in_shape, None)?;
    // 0-d 拒否契約（`dim=None` 経路は `normalize_dim` を経由しないため
    // ここで明示的に検査する。codex-review P2 是正・`reject_rank_zero`
    // doc 参照）。
    reject_rank_zero(in_shape.len())?;

    let rank = in_shape.len();
    let dim = match opts.dim {
        Some(d) => Some(normalize_dim(d, rank)?),
        None => None,
    };
    let target_len = unique_target_len(&in_shape, dim)?;
    ensure_target_len_fits_i32(target_len)?;

    let input_val = materialize_one(x)?;
    let out = unique_ext_with_fallback(x.tape().ops(), &input_val, dim, true)?;
    Ok(UniqueOutput {
        values: out.values,
        inverse: opts.return_inverse.then_some(out.inverse),
        counts: opts.return_counts.then_some(out.counts),
    })
}

/// [`unique_with_options`]・[`unique_consecutive`] が共有する
/// 「対象要素数」の算出（`dim=None` は要素数積・`dim=Some(d)` は
/// `shape[d]`）。要素数積は `checked_mul` で再検査する（`in_shape`
/// 自体は `ensure_alloc_fits_f32` で確保上限を検査済みだが、
/// バイト数上限〈`isize::MAX` バイト〉と要素数積の `usize`
/// オーバーフローは独立の検査軸のため、ここでも明示的に検査する）。
fn unique_target_len(in_shape: &[usize], dim: Option<usize>) -> Result<usize, AutodiffError> {
    match dim {
        None => in_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow)),
        Some(d) => Ok(in_shape[d]),
    }
}
