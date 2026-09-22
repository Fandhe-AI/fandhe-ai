//! 低精度 Linear forward（`half::f16`／`half::bf16`。イシュー #1960・親
//! #1626／#1648）。
//!
//! `crate::typed_ops::TypedOps<T>`（capability accessor
//! `BackendOps::typed_ops_f16`／`typed_ops_bf16`。イシュー #1687）は
//! `gemm`／`add`／`relu` 等の低レベル演算のみを提供し、`autodiff` から
//! 直接利用する箇所はこれまで存在しなかった（`docs/backend-dtype-
//! dispatch-design.md` §8「`Var`／`Tape` の dtype 一般化はスコープ外」）。
//! 本モジュールは `fandhe_ai_autodiff::nn::linear` から呼ばれる 1 本の
//! 関数（[`linear_forward_low_precision`]）として、Linear 層限定の
//! **opt-in** 低精度 forward（f32 master weight・f32 backward・forward
//! のみ低精度）を提供する。`Var`／`Tape` 自体の dtype は f32 のまま
//! 不変（`autodiff` 側は forward 値を f32 昇格済みで `Tensor<f32>`
//! として受け取り、通常の `Op::LinearAct`〈`compute_dtype` フィールド〉
//! ノードへそのまま記録する）。
//!
//! # 数値方式
//!
//! `input`／`weight`／`bias`（いずれも f32）を要素ごとに低精度へ降格
//! （`half::f16::from_f32`／`half::bf16::from_f32`。IEEE 754 最近接偶数
//! 丸め）し、`TypedOps<T>::gemm` → （`bias` があれば）`TypedOps<T>::add`
//! （NumPy 互換 broadcast。3 バックエンドとも f32 版と同一規則）→
//! （`Activation::Relu` なら）`TypedOps<T>::relu` の順に低精度カーネルへ
//! 委譲し、結果を 1 回だけ f32 へ昇格して返す。低精度の表現範囲を
//! 超える中間値は ±inf／非正規化数へ丸められる（IEEE 754 挙動をそのまま
//! 伝播させる。`crates/backend-cpu/src/typed_f16.rs` と同じ契約）。
//!
//! # fail-closed 方針
//!
//! - `BackendOps::typed_ops_f16`／`typed_ops_bf16` が `None`（低精度
//!   カーネル未実装のバックエンド）の場合は f32 へ静かにフォール
//!   バックせず [`BackendError::Unsupported`] を返す（opt-in が精度に
//!   ついて嘘をつかないため。`.claude/rules/security.md` A04）
//! - `dtype` が `ScalarDType::F16`／`Bf16` 以外、`act` が
//!   `Activation::None`／`Relu` 以外（[`crate::backend_ops::Activation`]
//!   は `#[non_exhaustive]`）は [`BackendError::InvalidArgument`] で
//!   拒否する
//!
//! # スコープ外
//!
//! `Var`／`Tape` の dtype 一般化・backward の低精度化・`DeviceParamStore`
//! 常駐経路・facade 公開面の拡張は対象外（`docs/
//! autodiff-low-precision-linear-design.md` 参照）。
//!
//! # Conv2d・MatMul（attention）への拡張（イシュー #2071）
//!
//! [`matmul_low_precision`]・[`conv2d_forward_low_precision`] は上記
//! Linear 限定の方式を横展開したもの。数値方式・fail-closed 方針は
//! Linear 版と同一（低精度へ丸め → `TypedOps<T>` の演算 → f32 へ 1 回
//! 昇格。accessor 不在は `Unsupported`、対応外 dtype は
//! `InvalidArgument`）。`matmul_low_precision` はバッチ行列積
//! （`crate::ops_shape::batched_matmul_plan` で正規化。rank 2 は
//! `TypedOps::gemm` へ直接委譲・rank≥3 は per-batch ループで合成する
//! 点も `tensor-core::backend_ops::default_gemm_batched` と同型）。
//! `conv2d_forward_low_precision` は im2col 済みの `col`
//! （算術を伴わない bit 完全一致コピーのため f32 のまま呼び出し元
//! 〈`fandhe_ai_autodiff::var::Var::conv2d_low_precision`〉が計算する）
//! を受け取り、GEMM＋bias 加算のみを低精度化する。

use half::{bf16, f16};

use crate::backend_ops::{Activation, BackendOps};
use crate::broadcast::broadcast_shape;
use crate::device::BackendError;
use crate::element::{Element, ScalarDType};
use crate::error::ShapeError;
use crate::ops_shape::{batched_matmul_plan, gemm_out_shape};
use crate::tensor::{Tensor, checked_numel, checked_numel_for};
use crate::typed_ops::TypedOps;

/// `f16`／`bf16` を低精度 Linear forward の対象型として封印する
/// sealed trait。`f32`/`f64` はこの trait を実装しない（`Scalar` 自体は
/// 4 型あるが、本モジュールが扱うのは低精度 2 型のみ）。
mod private {
    pub trait Sealed {}
}

/// 低精度 Linear forward の対象型が持つべき f32 往復変換
/// （`half::f16`／`half::bf16` それぞれの `from_f32`／`to_f32` を
/// 共通シグネチャへ揃えるための内部 trait。`crate::element::Scalar`
/// の封印を再利用せず本モジュール専用に封印する）。
trait LowPrecisionScalar: crate::element::Scalar + private::Sealed {
    /// IEEE 754 最近接偶数丸めで f32 から降格する。
    fn from_f32_round(v: f32) -> Self;
    /// f32 へ昇格する（`half::{f16,bf16}::to_f32` そのもの）。
    fn to_f32_value(self) -> f32;
}

impl private::Sealed for f16 {}
impl LowPrecisionScalar for f16 {
    fn from_f32_round(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn to_f32_value(self) -> f32 {
        self.to_f32()
    }
}

impl private::Sealed for bf16 {}
impl LowPrecisionScalar for bf16 {
    fn from_f32_round(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn to_f32_value(self) -> f32 {
        self.to_f32()
    }
}

/// `Tensor<f32>` を低精度 `Tensor<T>` へ降格する（要素ごとの丸め。
/// `Tensor::host_slice` は contiguous なら借用・非 contiguous な view は
/// ここで 1 回だけ実体化する。shape 自体は不変のため `Tensor::new` の
/// 失敗は契約上到達しないはずだが、`Tensor` 実装の不変条件違反に対する
/// fail-safe として型付きエラーで受ける
/// `crates/backend-cpu/src/typed_f16.rs::upcast_f16` と同型の設計）。
fn downcast<T: LowPrecisionScalar>(t: &Tensor<f32>) -> Result<Tensor<T>, BackendError> {
    let rounded: Vec<T> = t
        .host_slice()
        .iter()
        .map(|&v| T::from_f32_round(v))
        .collect();
    Tensor::new(rounded, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "low_precision::downcast: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// 低精度 `Tensor<T>` を `Tensor<f32>` へ昇格する（[`downcast`] の対）。
fn upcast<T: LowPrecisionScalar>(t: &Tensor<T>) -> Result<Tensor<f32>, BackendError> {
    let promoted: Vec<f32> = t.host_slice().iter().map(|&v| v.to_f32_value()).collect();
    Tensor::new(promoted, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "low_precision::upcast: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// `TypedOps<T>` を型消去せずに直接呼ぶ本体（`T` が確定した後の
/// ジェネリック実装。呼び出し元の [`linear_forward_low_precision`] が
/// `ScalarDType` から `T` を選び本関数へディスパッチする）。
fn linear_forward_typed<T: LowPrecisionScalar>(
    ops: &dyn TypedOps<T>,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Result<Tensor<f32>, BackendError> {
    let x = downcast::<T>(input)?;
    let w = downcast::<T>(weight)?;
    let mut y = TypedOps::gemm(ops, &x, &w)?;
    if let Some(b) = bias {
        let b_t = downcast::<T>(b)?;
        y = TypedOps::add(ops, &y, &b_t)?;
    }
    // `Activation` は `#[non_exhaustive]`（外部クレート向けの非破壊拡張
    // マーカー）だが、本クレート（`tensor-core`）自身が定義元のため
    // ここでは網羅 match が成立する（`non_exhaustive` はクレート境界
    // 越しの match にのみ `_` 分岐を強制する）。`grad.rs::vjp` の
    // `Op::LinearAct` 分岐（`autodiff` クレート側・外部）はワイルド
    // カードで同じ contract を fail-closed に守る。
    let y = match act {
        Activation::None => y,
        Activation::Relu => TypedOps::relu(ops, &y)?,
    };
    upcast::<T>(&y)
}

/// Linear 層限定の opt-in 低精度 forward（イシュー #1960）。
///
/// `input`／`weight`／`bias`（すべて f32 のホスト常駐テンソル）を
/// `dtype`（[`ScalarDType::F16`]／[`ScalarDType::Bf16`] のみ）で
/// 指定した低精度へ丸めて `gemm`（＋`add`＋`act`）を計算し、結果を f32
/// へ昇格して返す。`fandhe_ai_autodiff::nn::linear::
/// linear_forward_low_precision`（自由関数。`LinearVars` へのフィールド
/// 追加を避けるための配置。設計 doc 参照）の唯一の呼び出し元。
///
/// shape 検査は `gemm_out_shape`／`broadcast_shape`（既存 f32 経路と
/// 同一の判定関数）で計算前に行う。`ops.typed_ops_f16()`／
/// `typed_ops_bf16()` が `None`（バックエンド未実装）の場合は
/// [`BackendError::Unsupported`] を返す（f32 へのフォールバックはしない。
/// モジュール doc「fail-closed 方針」参照）。
pub fn linear_forward_low_precision(
    ops: &dyn BackendOps,
    dtype: ScalarDType,
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Result<Tensor<f32>, BackendError> {
    let out_shape =
        gemm_out_shape(input.shape(), weight.shape()).map_err(BackendError::ShapeMismatch)?;
    if let Some(b) = bias {
        // `broadcast_shape` の成功のみでは bias が `out_shape` を
        // 「拡張」する場合（例: out_shape `[1,1]`・bias `[2,1]`）を
        // 誤って受理してしまう。`fandhe_ai_autodiff::grad::vjp` の
        // `Op::LinearAct` 分岐は forward の実際の出力 shape（`add` が
        // bias 方向へ拡張していればそちら）を `matmul_vjp` へそのまま
        // 渡すため、`weight` との縮約次元が食い違い K 不一致で失敗する。
        // 呼び出し元（`autodiff::var.rs::linear_act_low_precision`）も
        // 同型の検査を行うが、本関数は `tensor-core` の公開 API として
        // 直接呼ばれうるため独立に検査する（判定迂回経路を作らない。
        // `.claude/rules/security.md` A08。codex-review 指摘・PR #2000）。
        let broadcast_result =
            broadcast_shape(&out_shape, b.shape()).map_err(BackendError::ShapeMismatch)?;
        if broadcast_result != out_shape {
            return Err(BackendError::ShapeMismatch(
                ShapeError::BroadcastIncompatible {
                    lhs: out_shape,
                    rhs: b.shape().to_vec(),
                },
            ));
        }
    }
    match dtype {
        ScalarDType::F16 => {
            let typed = ops.typed_ops_f16().ok_or_else(|| {
                BackendError::Unsupported(
                    "linear_forward_low_precision: typed_ops_f16 unavailable on this backend"
                        .into(),
                )
            })?;
            linear_forward_typed(typed, input, weight, bias, act)
        }
        ScalarDType::Bf16 => {
            let typed = ops.typed_ops_bf16().ok_or_else(|| {
                BackendError::Unsupported(
                    "linear_forward_low_precision: typed_ops_bf16 unavailable on this backend"
                        .into(),
                )
            })?;
            linear_forward_typed(typed, input, weight, bias, act)
        }
        other => Err(BackendError::InvalidArgument(format!(
            "linear_forward_low_precision: unsupported dtype ({other:?}); only F16/Bf16 are \
             accepted"
        ))),
    }
}

/// [`default_gemm_batched`](crate::backend_ops)（f32 専用）の低精度版
/// 正規化ヘルパー。`operand`（`[..., rows, cols]`。先頭バッチ軸は
/// `out_batch_shape` へ broadcast 可能）を `[flat_len, rows, cols]` の
/// contiguous 3 次元へ正規化する。`normalize_batched_operand`
/// （`backend_ops.rs`）と同じ正規化規則を `T: Element` へ一般化した
/// もの（`Tensor<f32>` 専用の同関数は再利用できないため本モジュール
/// 専用に持つ。二重管理を避けるため規則のみ揃える）。
fn normalize_batched_operand_typed<T: Element>(
    operand: &Tensor<T>,
    out_batch_shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<Tensor<T>, BackendError> {
    let operand_rank = operand.shape().len();
    if operand_rank < 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: operand_rank,
        }));
    }
    let operand_batch_shape = operand.shape()[..operand_rank - 2].to_vec();
    let operand_tail = operand.shape()[operand_rank - 2..].to_vec();
    if operand_tail != [rows, cols] {
        return Err(BackendError::ShapeMismatch(ShapeError::MatmulDimMismatch {
            lhs: operand.shape().to_vec(),
            rhs: vec![rows, cols],
        }));
    }

    // `checked_numel`／`checked_numel_for::<T>` は `tensor.rs::
    // default_gemm_batched` 前段と同じ overflow-safe な検査
    // （`usize` 積オーバーフロー・`Vec` allocation 上限〈`isize::MAX`
    // バイト〉）を、低精度要素サイズ（`f16`/`bf16` は 2 バイト）で行う
    // （`.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
    let flat_len = checked_numel(out_batch_shape).map_err(BackendError::ShapeMismatch)?;
    let mut full_shape = Vec::with_capacity(out_batch_shape.len() + 2);
    full_shape.extend_from_slice(out_batch_shape);
    full_shape.push(rows);
    full_shape.push(cols);
    checked_numel_for::<T>(&full_shape).map_err(BackendError::ShapeMismatch)?;

    let normalized = if operand_batch_shape == out_batch_shape {
        operand.contiguous()
    } else {
        operand
            .broadcast_to(&full_shape)
            .map_err(BackendError::ShapeMismatch)?
            .contiguous()
    };
    let flat_shape = [flat_len, rows, cols];
    normalized
        .reshape(&flat_shape)
        .map_err(BackendError::ShapeMismatch)
}

/// `TypedOps<T>` を型消去せずに直接呼ぶバッチ行列積本体（`T` が
/// 確定した後のジェネリック実装。[`matmul_low_precision`] が
/// `ScalarDType` から `T` を選び本関数へディスパッチする）。
///
/// rank 2 同士（`plan.batch_shape()` が空）は `TypedOps::gemm` へ直接
/// 委譲する（`tensor-core::backend_ops::default_gemm_batched` と同じ
/// 「バッチをほどく正規化・ループを経由しない」構造保証）。rank≥3 は
/// 各オペランドを `[B, m, k]`／`[B, k, n]` へ正規化してから per-batch
/// `TypedOps::gemm` ループで合成する。
fn matmul_typed<T: LowPrecisionScalar>(
    ops: &dyn TypedOps<T>,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<Tensor<T>, BackendError> {
    let plan =
        batched_matmul_plan(lhs.shape(), rhs.shape()).map_err(BackendError::ShapeMismatch)?;

    if plan.batch_shape().is_empty() {
        return TypedOps::gemm(ops, lhs, rhs);
    }

    let m = plan.m();
    let k = plan.k();
    let n = plan.n();
    let batch_len: usize = plan.batch_shape().iter().product();

    // 出力全体（`batch_shape ++ [m, n]`）の要素数・バイトサイズを
    // バッファ確保より前に検証する（`default_gemm_batched` と同じ
    // 順序。`.claude/rules/coding-rust.md`）。
    let total = checked_numel_for::<T>(&plan.out_shape()).map_err(BackendError::ShapeMismatch)?;
    if total == 0 {
        return Tensor::new(Vec::new(), &plan.out_shape()).map_err(BackendError::ShapeMismatch);
    }

    let lhs_norm = normalize_batched_operand_typed(lhs, plan.batch_shape(), m, k)?;
    let rhs_norm = normalize_batched_operand_typed(rhs, plan.batch_shape(), k, n)?;

    let mut out_data: Vec<T> = Vec::with_capacity(total);
    for i in 0..batch_len {
        let lhs_i = lhs_norm
            .narrow(0, i, 1)
            .and_then(|t| t.reshape(&[m, k]))
            .map_err(BackendError::ShapeMismatch)?;
        let rhs_i = rhs_norm
            .narrow(0, i, 1)
            .and_then(|t| t.reshape(&[k, n]))
            .map_err(BackendError::ShapeMismatch)?;
        let out_i = TypedOps::gemm(ops, &lhs_i, &rhs_i)?;
        // `TypedOps::gemm` は shape `[m, n]` の新規確保テンソルを返す
        // 契約（f32 版 `BackendOps::gemm` と同型）であり常に contiguous
        // のため `as_slice` は必ず `Some` を返す。`None`（契約違反）は
        // fail-closed で拒否する（`default_gemm_batched` と同型）。
        let out_slice = out_i.as_slice().ok_or_else(|| {
            BackendError::InvalidArgument(
                "low_precision::matmul_typed: per-batch gemm returned a non-contiguous tensor \
                 (contract violation)"
                    .into(),
            )
        })?;
        out_data.extend_from_slice(out_slice);
    }

    Tensor::new(out_data, &plan.out_shape()).map_err(BackendError::ShapeMismatch)
}

/// バッチ行列積の opt-in 低精度 forward（イシュー #2071）。
///
/// `lhs`／`rhs`（f32 のホスト常駐テンソル。rank≥2・NumPy 互換バッチ
/// broadcast）を `dtype`（[`ScalarDType::F16`]／[`ScalarDType::Bf16`]
/// のみ）で指定した低精度へ丸めて `TypedOps<T>::gemm`（バッチは
/// per-batch ループで合成）を計算し、結果を f32 へ昇格して返す。
/// `fandhe_ai_autodiff::var::Var::matmul_low_precision`（`crate::
/// attention` の低精度 SDPA・`nn::attention` の低精度 MHA が経由する）
/// の唯一の呼び出し元。
///
/// shape 検査（`batched_matmul_plan`）を accessor 取得より先に行う。
/// `ops.typed_ops_f16()`／`typed_ops_bf16()` が `None`（バックエンド
/// 未実装）の場合は [`BackendError::Unsupported`] を返す（f32 への
/// フォールバックはしない。モジュール doc「fail-closed 方針」参照）。
pub fn matmul_low_precision(
    ops: &dyn BackendOps,
    dtype: ScalarDType,
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError> {
    batched_matmul_plan(lhs.shape(), rhs.shape()).map_err(BackendError::ShapeMismatch)?;
    match dtype {
        ScalarDType::F16 => {
            let typed = ops.typed_ops_f16().ok_or_else(|| {
                BackendError::Unsupported(
                    "matmul_low_precision: typed_ops_f16 unavailable on this backend".into(),
                )
            })?;
            let lhs_t = downcast::<f16>(lhs)?;
            let rhs_t = downcast::<f16>(rhs)?;
            let out = matmul_typed(typed, &lhs_t, &rhs_t)?;
            upcast::<f16>(&out)
        }
        ScalarDType::Bf16 => {
            let typed = ops.typed_ops_bf16().ok_or_else(|| {
                BackendError::Unsupported(
                    "matmul_low_precision: typed_ops_bf16 unavailable on this backend".into(),
                )
            })?;
            let lhs_t = downcast::<bf16>(lhs)?;
            let rhs_t = downcast::<bf16>(rhs)?;
            let out = matmul_typed(typed, &lhs_t, &rhs_t)?;
            upcast::<bf16>(&out)
        }
        other => Err(BackendError::InvalidArgument(format!(
            "matmul_low_precision: unsupported dtype ({other:?}); only F16/Bf16 are accepted"
        ))),
    }
}

/// Conv2d の opt-in 低精度 forward（イシュー #2071）。
///
/// `col`（im2col 済み。`[N, G, K_g, P]`。呼び出し元
/// 〈`fandhe_ai_autodiff::var::Var::conv2d_low_precision`〉が f32 の
/// まま im2col を計算する——算術を伴わない bit 完全一致コピーのため
/// 低精度化の対象外）・`weight`（`[Cout, Cin_g, kH, kW]`）・`bias`
/// （`Some` なら `[Cout]`）を `dtype` で指定した低精度へ丸めて
/// バッチ GEMM（`matmul_typed`）＋（`bias` があれば）`TypedOps::add`
/// を計算し、`out_shape` へ reshape してから f32 へ 1 回昇格して返す。
/// `fandhe_ai_autodiff::var::Var::conv2d_low_precision` の唯一の
/// 呼び出し元。
///
/// # 検査順序
///
/// ①`col`／`weight` の rank・`groups`（`col.shape()[1]`）・
/// `Cout % groups`・`K_g` 一致・`out_shape` の要素数一致・`bias`
/// `[Cout]` 一致（いずれも accessor 取得より先）→ ②accessor 取得
/// （`None` は [`BackendError::Unsupported`]）→ ③降格 → ④
/// `matmul_typed`（`[N, G, Cout_g, P]`）→ ⑤`out_shape` へ reshape →
/// ⑥bias 加算（`TypedOps::add`。結果 shape が `out_shape` と一致する
/// ことを検証）→ ⑦昇格。
#[allow(clippy::too_many_arguments)] // linear_forward_low_precision と同じ理由（im2col 済み col・weight・bias・out_shape をすべて明示する必要がある）。
pub fn conv2d_forward_low_precision(
    ops: &dyn BackendOps,
    dtype: ScalarDType,
    col: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, BackendError> {
    let col_shape = col.shape();
    if col_shape.len() != 4 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 4,
            actual: col_shape.len(),
        }));
    }
    let weight_shape = weight.shape();
    if weight_shape.len() != 4 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 4,
            actual: weight_shape.len(),
        }));
    }
    let n = col_shape[0];
    let groups = col_shape[1];
    let k_g = col_shape[2];
    let p = col_shape[3];
    let cout = weight_shape[0];
    if groups == 0 || !cout.is_multiple_of(groups) {
        return Err(BackendError::InvalidArgument(format!(
            "conv2d_forward_low_precision: out_channels ({cout}) must be divisible by groups \
             ({groups})"
        )));
    }
    let cout_g = cout / groups;
    let weight_k_g = weight_shape[1]
        .checked_mul(weight_shape[2])
        .and_then(|v| v.checked_mul(weight_shape[3]))
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(BackendError::ShapeMismatch)?;
    if weight_k_g != k_g {
        return Err(BackendError::ShapeMismatch(ShapeError::MatmulDimMismatch {
            lhs: vec![groups, k_g],
            rhs: vec![groups, weight_k_g],
        }));
    }
    let expected_out_numel = n
        .checked_mul(cout)
        .and_then(|v| v.checked_mul(p))
        .ok_or(ShapeError::ElementCountOverflow)
        .map_err(BackendError::ShapeMismatch)?;
    let out_numel = checked_numel_for::<f32>(out_shape).map_err(BackendError::ShapeMismatch)?;
    if out_numel != expected_out_numel {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: out_shape.to_vec(),
            rhs: vec![n, cout, p],
        }));
    }
    if let Some(b) = bias
        && b.shape() != [cout]
    {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: b.shape().to_vec(),
            rhs: vec![cout],
        }));
    }

    match dtype {
        ScalarDType::F16 => {
            let typed = ops.typed_ops_f16().ok_or_else(|| {
                BackendError::Unsupported(
                    "conv2d_forward_low_precision: typed_ops_f16 unavailable on this backend"
                        .into(),
                )
            })?;
            conv2d_forward_typed(typed, col, weight, bias, groups, cout_g, k_g, out_shape)
        }
        ScalarDType::Bf16 => {
            let typed = ops.typed_ops_bf16().ok_or_else(|| {
                BackendError::Unsupported(
                    "conv2d_forward_low_precision: typed_ops_bf16 unavailable on this backend"
                        .into(),
                )
            })?;
            conv2d_forward_typed(typed, col, weight, bias, groups, cout_g, k_g, out_shape)
        }
        other => Err(BackendError::InvalidArgument(format!(
            "conv2d_forward_low_precision: unsupported dtype ({other:?}); only F16/Bf16 are \
             accepted"
        ))),
    }
}

/// [`conv2d_forward_low_precision`] の `T` 確定後の本体。
#[allow(clippy::too_many_arguments)]
fn conv2d_forward_typed<T: LowPrecisionScalar>(
    ops: &dyn TypedOps<T>,
    col: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    groups: usize,
    cout_g: usize,
    k_g: usize,
    out_shape: &[usize],
) -> Result<Tensor<f32>, BackendError> {
    let col_t = downcast::<T>(col)?;
    let w_mat = weight
        .contiguous()
        .reshape(&[groups, cout_g, k_g])
        .map_err(BackendError::ShapeMismatch)?;
    let w_mat_t = downcast::<T>(&w_mat)?;
    // `w_mat_t` `[G, Cout_g, K_g]` × `col_t` `[N, G, K_g, P]` の
    // per-group バッチ GEMM。`batched_matmul_plan` が両者を NumPy
    // 互換 broadcast し `[N, G, Cout_g, P]`（要素数は `out_shape` と
    // 同一）を返す（`grad::conv2d_with_fallback` の f32
    // `ops.gemm_batched` 呼び出しと同じ broadcast 規則）。
    let out_mat = matmul_typed(ops, &w_mat_t, &col_t)?;
    let cout = groups * cout_g;
    let out_no_bias = out_mat
        .reshape(out_shape)
        .map_err(BackendError::ShapeMismatch)?;

    let out_t = match bias {
        Some(b) => {
            let bias_reshaped = b
                .contiguous()
                .reshape(&[1, cout, 1, 1])
                .map_err(BackendError::ShapeMismatch)?;
            let bias_t = downcast::<T>(&bias_reshaped)?;
            let out_biased = TypedOps::add(ops, &out_no_bias, &bias_t)?;
            if out_biased.shape() != out_shape {
                return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                    lhs: out_biased.shape().to_vec(),
                    rhs: out_shape.to_vec(),
                }));
            }
            out_biased
        }
        None => out_no_bias,
    };
    upcast::<T>(&out_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    /// `typed_ops_f16`／`typed_ops_bf16` とも `None` を返すだけの
    /// モック（accessor 不在の fail-closed 経路を検証するためのもの。
    /// `BackendOps` の他メソッドは到達しない想定のため `unimplemented!`
    /// で構わない）。
    struct NoTypedOpsBackend;

    impl BackendOps for NoTypedOpsBackend {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: gemm should not be reached")
        }
        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: add should not be reached")
        }
        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: mul should not be reached")
        }
        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: relu should not be reached")
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: exp should not be reached")
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: tanh should not be reached")
        }
        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: sum should not be reached")
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unimplemented!("test: max should not be reached")
        }
    }

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn f16_accessor_none_returns_unsupported() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F16, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn bf16_accessor_none_returns_unsupported() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::Bf16, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn f32_dtype_is_rejected_as_invalid_argument() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F32, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn f64_dtype_is_rejected_as_invalid_argument() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F64, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn shape_mismatch_is_rejected_before_accessor_lookup() {
        // 出力 shape がそもそも成立しない（k 不一致）ケースは
        // `typed_ops_f16()` を呼ぶ前に `ShapeMismatch` で拒否される
        // （accessor が `None` の `NoTypedOpsBackend` でも到達できる
        // ことで、shape 検査が accessor 取得より先に走る契約を確認）。
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0, 3.0], &[1, 3]);
        let w = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err =
            linear_forward_low_precision(&ops, ScalarDType::F16, &x, &w, None, Activation::None)
                .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    /// codex-review 指摘（PR #2000）の回帰テスト: `bias` が `out_shape`
    /// を「拡張」するブロードキャスト（out_shape `[1,1]`・bias `[2,1]`。
    /// NumPy 互換規則では両立するが結果 shape は `[2,1]` へ拡張される）
    /// は、`typed_ops_f16()` を呼ぶ前に fail-closed で拒否される
    /// （accessor が `None` の `NoTypedOpsBackend` でも到達できることで
    /// 拒否が shape 検査段階で完結することを確認する）。
    #[test]
    fn expanding_bias_broadcast_is_rejected_before_accessor_lookup() {
        let ops = NoTypedOpsBackend;
        let x = t(&[1.0, 2.0], &[1, 2]);
        let w = t(&[1.0, 1.0], &[2, 1]);
        let bias = t(&[0.0, 0.0], &[2, 1]);
        let err = linear_forward_low_precision(
            &ops,
            ScalarDType::F16,
            &x,
            &w,
            Some(&bias),
            Activation::None,
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    // --- matmul_low_precision（イシュー #2071） ---

    #[test]
    fn matmul_low_precision_f16_accessor_none_returns_unsupported() {
        let ops = NoTypedOpsBackend;
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err = matmul_low_precision(&ops, ScalarDType::F16, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn matmul_low_precision_f32_dtype_is_rejected_as_invalid_argument() {
        let ops = NoTypedOpsBackend;
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let err = matmul_low_precision(&ops, ScalarDType::F32, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn matmul_low_precision_shape_mismatch_is_rejected_before_accessor_lookup() {
        let ops = NoTypedOpsBackend;
        // rank 3 バッチ・K 不一致（[2,1,3] x [2,2,4] は K=3 vs K=2）。
        let a = t(&[1.0; 6], &[2, 1, 3]);
        let b = t(&[1.0; 16], &[2, 2, 4]);
        let err = matmul_low_precision(&ops, ScalarDType::F16, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    // --- conv2d_forward_low_precision（イシュー #2071） ---

    #[test]
    fn conv2d_low_precision_accessor_none_returns_unsupported() {
        let ops = NoTypedOpsBackend;
        // col: [N=1, G=1, K_g=4, P=1]・weight: [Cout=1, Cin_g=1, kH=2, kW=2]（K_g = 1*2*2 = 4）。
        let col = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let weight = t(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let err = conv2d_forward_low_precision(
            &ops,
            ScalarDType::F16,
            &col,
            &weight,
            None,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn conv2d_low_precision_k_g_mismatch_is_rejected_before_accessor_lookup() {
        let ops = NoTypedOpsBackend;
        // K_g = col.shape()[2] = 3 だが weight は Cin_g*kH*kW = 1*2*2 = 4 で不一致。
        let col = t(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
        let weight = t(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let err = conv2d_forward_low_precision(
            &ops,
            ScalarDType::F16,
            &col,
            &weight,
            None,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn conv2d_low_precision_out_channels_not_divisible_by_groups_is_rejected() {
        let ops = NoTypedOpsBackend;
        // groups = col.shape()[1] = 2 だが cout = weight.shape()[0] = 3（3 % 2 != 0）。
        let col = t(&[1.0; 8], &[1, 2, 4, 1]);
        let weight = t(&[1.0; 12], &[3, 1, 2, 2]);
        let err = conv2d_forward_low_precision(
            &ops,
            ScalarDType::F16,
            &col,
            &weight,
            None,
            &[1, 3, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }

    #[test]
    fn conv2d_low_precision_bias_shape_mismatch_is_rejected_before_accessor_lookup() {
        let ops = NoTypedOpsBackend;
        let col = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let weight = t(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let bias = t(&[0.0, 0.0], &[2]);
        let err = conv2d_forward_low_precision(
            &ops,
            ScalarDType::F16,
            &col,
            &weight,
            Some(&bias),
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn conv2d_low_precision_f32_dtype_is_rejected_as_invalid_argument() {
        let ops = NoTypedOpsBackend;
        let col = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let weight = t(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);
        let err = conv2d_forward_low_precision(
            &ops,
            ScalarDType::F32,
            &col,
            &weight,
            None,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::InvalidArgument(_)));
    }
}
