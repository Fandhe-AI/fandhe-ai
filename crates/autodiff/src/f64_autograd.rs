//! f64 専用の独立自動微分グラフ（イシュー #2195・親 #2142「f64 autograd
//! の最小集合」の第 1 段）。
//!
//! # 設計の位置づけ（`docs/autodiff-var-dtype-multiplexing-design.md` との関係）
//!
//! 同 doc は §5 で「案 A（`Var<'t, T>` へのフル一般化）に着手しない」、
//! 「案 B（1 本のテープ内で dtype が混在する型消去ノード）は不採用」と
//! 結論しており、§10 の承認事項 1〜5（`TypedOps` への演算追加・`Var`
//! への inherent メソッド追加・facade 再エクスポート・`Var<T>` 一般化・
//! ホスト参照実装の扱い）はいずれも未承認のまま確定していない。本
//! モジュールは**そのどちらでもない第 3 の形**として実装する:
//! f32 の [`crate::tape::Tape`]／[`crate::var::Var`] とは完全に独立した、
//! f64 専用・dtype 混在なしの小さなグラフ（[`TapeF64`]／[`VarF64`]）を
//! 新規ファイルのみで構成する。既存の `Var`（`var.rs`）へ inherent
//! メソッドを追加せず、`Var` の型エイリアスも作らない。`facade` は本
//! モジュールを一切再エクスポートしない（内部クレート限定 `pub` API。
//! `docs/compat-api-scope.md` §0 の「`facade` が唯一のサポートされる
//! 公開 API 面」という前提のもと、§10 のどの承認事項も消費しない）。
//!
//! # cast との関係
//!
//! [`crate::var::Var::cast`]`::<f64>()` は既存どおり**勾配の切れた
//! （detached な）`Tensor<f64>`** を返す（挙動は変更しない）。f32 の
//! グラフから f64 のグラフへは、この cast を経由して
//! [`TapeF64::var`]／[`TapeF64::var_no_grad`] へ渡す「勾配の切れた経路」
//! だけで渡る。f64 側の [`TapeF64::backward`] は f32 テープへは一切
//! 勾配を流さない（両グラフは `NodeIdF64`／`NodeId` という別々の
//! 添字空間を持ち、相互参照する構造を持たないため構造的に不可能）。
//!
//! # バックエンド別 dispatch
//!
//! `add`／`mul` は [`crate::tape::Tape::typed_ops_f64`]（`BackendOps` の
//! capability accessor）が `Some` を返せばそちらへ委譲し、`None`
//! または `BackendError::Unsupported` が返った場合はホスト上の f64
//! 参照実装へフォールバックする。CPU（#1697）・CUDA（#2060）はいずれも
//! ネイティブ実装を返す。Metal は MSL の `double` 非対応が恒久的な
//! ため常に `None` を返し、ホスト経路へ到達する。**`div`／`pow` は
//! `TypedOps<f64>` に演算が存在しない**（trait は `gemm`／`add`／`mul`／
//! `relu`／`exp`／`tanh`／`sum`／`max` の 8 演算に固定済み。
//! `crates/tensor-core/src/typed_ops.rs` 参照）ため、バックエンドに
//! 依らず常にホスト上の f64 参照実装で計算する。f32 へ黙って
//! フォールバックすることは決してしない（`Unsupported` 以外のバック
//! エンドエラーはそのまま伝播する）。
//!
//! # 数値契約
//!
//! すべての値は eager に `f64` のまま計算する（融合・遅延実行は行わ
//! ない。`FusionPlan` は f32 固定のまま不変）。ブロードキャストの逆
//! 演算（[`reduce_to_shape_f64`]）はホスト上の `f64` 逐次和（row-major
//! index 順）で実装し、出力 dtype が既に `f64` であるため
//! `.claude/rules/coding-rust.md` の f64 アキュムレータ契約を自明に
//! 満たす。勾配の蓄積（fan-out 合流）も [`TapeF64::backward`] 内で
//! add と同じ dispatch 規則（native → host fallback）を通すため、CPU
//! ネイティブ経路とホスト経路で bit が一致する。
//!
//! # #2196 への申し送り
//!
//! [`OpF64`] は `pub(crate)` の非 `#[non_exhaustive]` enum とし、
//! 後続イシュー #2196（matmul・sum・mean・max・facade への到達）が
//! variant を追加する前提で本モジュール内の `match` を網羅形にして
//! いる（variant 追加時にコンパイルエラーで追従漏れを検知できる）。

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor, TypedOps, elementwise_out_shape};

use crate::error::AutodiffError;
use crate::tape::Tape;

/// プロセス全体で共有する [`TapeF64`] 識別子発行カウンタ。`crate::tape::
/// TapeId`（`tape.rs` の `NEXT_TAPE_ID`）と同じ理由（ポインタ比較は
/// 破棄されたテープのメモリ領域再利用による誤判定の余地があるため
/// 使わない。`docs/public-api-design.md` §3.1）で単調増加 ID を使う。
static NEXT_TAPE_ID_F64: AtomicU64 = AtomicU64::new(0);

/// [`TapeF64`] の一意識別子（`crate::tape::TapeId` の f64 版）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TapeIdF64(u64);

impl TapeIdF64 {
    fn fresh() -> Self {
        TapeIdF64(NEXT_TAPE_ID_F64.fetch_add(1, Ordering::Relaxed))
    }
}

/// [`TapeF64`] 内ノードの識別子（`nodes: Vec<NodeF64>` への添字）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeIdF64(usize);

/// f64 グラフのノードが表す演算の種別。`Leaf` は
/// [`TapeF64::var`]／[`TapeF64::var_no_grad`] が登録する葉ノード。
/// 二項演算 4 種はそれぞれ入力 2 ノードの [`NodeIdF64`] を保持する。
///
/// `#[non_exhaustive]` を付けない（クレート内限定 `pub(crate)` の enum
/// であり、公開 API 非破壊契約の対象外。#2196 が variant を追加する際に
/// 本モジュール内の `match` を非網羅として検出できるようにするため）。
pub(crate) enum OpF64 {
    Leaf,
    Add(NodeIdF64, NodeIdF64),
    Mul(NodeIdF64, NodeIdF64),
    Div(NodeIdF64, NodeIdF64),
    Pow(NodeIdF64, NodeIdF64),
}

/// 二項演算 4 種の種別のみを表す軽量タグ（[`OpF64`] から演算の種類だけ
/// を取り出して forward dispatch・VJP 係数計算へ渡すために使う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOpF64 {
    Add,
    Mul,
    Div,
    Pow,
}

/// f64 グラフの 1 ノード。値はすべて eager に保持する（遅延実行・融合は
/// 行わない）。
struct NodeF64 {
    op: OpF64,
    shape: Vec<usize>,
    value: Tensor<f64>,
    /// `Op::for_each_input` で列挙した全入力ノードの `requires_grad` の
    /// OR（`crate::tape::TapeNode::requires_grad` doc と同じ伝播規則）。
    requires_grad: bool,
}

/// f64 専用の独立自動微分グラフ本体。
///
/// `tape`（f32 の [`Tape`]）への `&'t` 借用は
/// [`Tape::typed_ops_f64`]（`BackendOps` capability accessor）を経由して
/// バックエンド実装へ到達するためだけに保持し、f32 側のノード列
/// （`Tape::nodes`）には一切触れない。`TapeF64` 自身は独自の
/// `nodes: RefCell<Vec<NodeF64>>` を持つ。
pub struct TapeF64<'t> {
    tape: &'t Tape,
    id: TapeIdF64,
    nodes: RefCell<Vec<NodeF64>>,
}

impl<'t> TapeF64<'t> {
    /// `tape`（f32 グラフの `BackendOps` 経由でバックエンドへ到達する
    /// ためだけに使う）を借用して新しい f64 グラフを構築する。
    pub fn new(tape: &'t Tape) -> Self {
        TapeF64 {
            tape,
            id: TapeIdF64::fresh(),
            nodes: RefCell::new(Vec::new()),
        }
    }

    /// `requires_grad = true` の葉ノードを登録する（PyTorch
    /// `requires_grad=True` 相当。既定）。
    pub fn var(&self, value: &Tensor<f64>) -> VarF64<'_, 't> {
        self.push_leaf(value.clone(), true)
    }

    /// `requires_grad = false` の葉ノードを登録する（PyTorch
    /// `requires_grad=False`／`torch.no_grad()` で作った葉相当）。
    pub fn var_no_grad(&self, value: &Tensor<f64>) -> VarF64<'_, 't> {
        self.push_leaf(value.clone(), false)
    }

    fn push_leaf(&self, value: Tensor<f64>, requires_grad: bool) -> VarF64<'_, 't> {
        let shape = value.shape().to_vec();
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeIdF64(nodes.len());
        nodes.push(NodeF64 {
            op: OpF64::Leaf,
            shape,
            value,
            requires_grad,
        });
        drop(nodes);
        VarF64 { graph: self, id }
    }

    /// `loss` を起点に逆伝播し、各ノードへ流入した勾配を
    /// [`GradientsF64`] へまとめて返す（`crate::tape::Tape::backward`
    /// と同じ「①クロステープ検査 → ②シード設定 → ③逆走査 → ④蓄積」の
    /// 構成）。
    ///
    /// 非スカラー `loss` のセマンティクスも `Tape::backward` と同じ:
    /// シードは全要素 1 の同 shape テンソル（`sum(loss).backward()` と
    /// 数学的に等価な暗黙の総和射影）。
    pub fn backward(&self, loss: &VarF64<'_, 't>) -> Result<GradientsF64, AutodiffError> {
        if loss.graph.id != self.id {
            return Err(AutodiffError::TapeMismatch);
        }

        let n = {
            let nodes = self.nodes.borrow();
            if !nodes[loss.id.0].requires_grad {
                return Err(AutodiffError::Backward(
                    "loss は勾配追跡対象の祖先を持たない（f64 グラフの \
                     requires_grad が false。var_no_grad の葉のみで構成 \
                     されている）"
                        .into(),
                ));
            }
            nodes.len()
        };

        let mut grads: Vec<Option<Tensor<f64>>> = vec![None; n];
        let loss_shape = self.nodes.borrow()[loss.id.0].shape.clone();
        let seed = Tensor::full(&loss_shape, 1.0f64).map_err(|err| {
            AutodiffError::Backward(format!(
                "loss 自身の shape での f64 シードテンソル構築に失敗した（契約違反）: {err}"
            ))
        })?;
        grads[loss.id.0] = Some(seed);

        // 発生順とは逆順に走査する（`crate::backward::Tape::backward` と
        // 同じ Wengert list の逆伝播）。
        for id in (0..n).rev() {
            // `grads[id]` は `GradientsF64::get()` が返す最終値そのもの
            // のため `take()` せず複製する（`crate::backward::Tape::
            // backward` と同じ理由。取り除くと非葉ノードの `get()` が
            // 常に `None` になる）。
            let Some(grad_g) = grads[id].clone() else {
                continue;
            };

            // 走査中に必要な情報だけを 1 回の借用で読み出し、VJP 計算
            // （借用を必要としない純粋関数）へ移る前に `nodes` の借用を
            // 閉じる（`RefCell` の二重可変借用 panic を避ける規律。
            // `var.rs` モジュール doc と同じ実装規律）。
            let bin = {
                let nodes = self.nodes.borrow();
                match nodes[id].op {
                    OpF64::Leaf => None,
                    OpF64::Add(lhs, rhs) => Some((BinOpF64::Add, lhs, rhs)),
                    OpF64::Mul(lhs, rhs) => Some((BinOpF64::Mul, lhs, rhs)),
                    OpF64::Div(lhs, rhs) => Some((BinOpF64::Div, lhs, rhs)),
                    OpF64::Pow(lhs, rhs) => Some((BinOpF64::Pow, lhs, rhs)),
                }
            };
            let Some((op_kind, lhs, rhs)) = bin else {
                // 葉ノードへは入力がないため寄与を戻す先がない
                // （`grads[id]` は上で複製済みのため、このノード自身の
                // 最終値は `grads` に残ったまま）。
                continue;
            };

            let (a_val, b_val, y_val) = {
                let nodes = self.nodes.borrow();
                (
                    nodes[lhs.0].value.clone(),
                    nodes[rhs.0].value.clone(),
                    nodes[id].value.clone(),
                )
            };

            let (da, db) = binary_vjp_f64(op_kind, &a_val, &b_val, &y_val, &grad_g)?;

            let lhs_requires_grad = self.nodes.borrow()[lhs.0].requires_grad;
            let rhs_requires_grad = self.nodes.borrow()[rhs.0].requires_grad;

            // イシュー #1748 と同じ規約: `requires_grad == false` の
            // ノードへの寄与は `accumulate` へ渡さず捨てる。
            if lhs_requires_grad {
                let existing = grads[lhs.0].take();
                grads[lhs.0] = Some(accumulate_f64(self.tape, existing, da)?);
            }
            if rhs_requires_grad {
                let existing = grads[rhs.0].take();
                grads[rhs.0] = Some(accumulate_f64(self.tape, existing, db)?);
            }
        }

        Ok(GradientsF64 {
            tape_id: self.id,
            grads,
        })
    }
}

/// [`TapeF64::backward`] が返す勾配の入れ物。[`VarF64`] 単位で
/// [`GradientsF64::get`] から引ける（`crate::backward::Gradients` の
/// f64 版・同じ意味論）。
#[derive(Debug)]
pub struct GradientsF64 {
    tape_id: TapeIdF64,
    grads: Vec<Option<Tensor<f64>>>,
}

impl GradientsF64 {
    /// `var` に対応する勾配を取得する。`var` が別 [`TapeF64`] に属する
    /// 場合は `Err(TapeMismatch)`。対象ノードが
    /// `requires_grad == false`（[`TapeF64::var_no_grad`] の葉、または
    /// それのみを祖先に持つ非葉ノード）の場合は
    /// `Err(GradientTrackingDisabled)`（`crate::backward::Gradients::get`
    /// と同じ「構造的に勾配を持ちえない」ことを表す型区別）。loss から
    /// 未到達なノードは `Ok(None)`。
    pub fn get(&self, var: &VarF64<'_, '_>) -> Result<Option<&Tensor<f64>>, AutodiffError> {
        if var.graph.id != self.tape_id {
            return Err(AutodiffError::TapeMismatch);
        }
        let requires_grad = var.graph.nodes.borrow()[var.id.0].requires_grad;
        if !requires_grad {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        Ok(self.grads.get(var.id.0).and_then(|g| g.as_ref()))
    }
}

/// f64 グラフ上の 1 ノードを指す追跡対象値（`crate::var::Var` の f64
/// 版）。値そのものではなく [`NodeIdF64`] + [`TapeF64`] への共有参照を
/// 保持する。`graph`（`'g`）と、その先の f32 [`Tape`]（`'t`）の 2 つの
/// ライフタイムを別々に持つことで、`TapeF64::var` が返す借用の寿命
/// （`'g`）と、`TapeF64` 自身が構築時に借用した f32 `Tape` の寿命
/// （`'t`）を混同しない。
#[derive(Clone, Copy)]
pub struct VarF64<'g, 't> {
    graph: &'g TapeF64<'t>,
    id: NodeIdF64,
}

impl<'g, 't> VarF64<'g, 't> {
    /// 現在の値を複製して返す（`Tensor` の `Clone` は内部 `Arc` の
    /// ポインタ複製のみで安価。`crate::var::Var::to_tensor` と同じ
    /// 契約）。
    pub fn value(&self) -> Tensor<f64> {
        self.graph.nodes.borrow()[self.id.0].value.clone()
    }

    /// このノードの出力 shape。
    pub fn shape(&self) -> Vec<usize> {
        self.graph.nodes.borrow()[self.id.0].shape.clone()
    }

    /// 要素ごとの加算（NumPy 互換ブロードキャスト）。
    pub fn add(&self, other: &Self) -> Result<Self, AutodiffError> {
        self.binary_op(other, BinOpF64::Add)
    }

    /// 要素ごとの乗算（NumPy 互換ブロードキャスト）。
    pub fn mul(&self, other: &Self) -> Result<Self, AutodiffError> {
        self.binary_op(other, BinOpF64::Mul)
    }

    /// 要素ごとの除算（NumPy 互換ブロードキャスト）。0 除算は `inf`／
    /// `NaN` を返し panic しない（IEEE 754 のまま扱う）。
    pub fn div(&self, other: &Self) -> Result<Self, AutodiffError> {
        self.binary_op(other, BinOpF64::Div)
    }

    /// 要素ごとの冪乗 `self ^ other`（NumPy 互換ブロードキャスト）。
    pub fn pow(&self, other: &Self) -> Result<Self, AutodiffError> {
        self.binary_op(other, BinOpF64::Pow)
    }

    fn binary_op(&self, other: &Self, op: BinOpF64) -> Result<Self, AutodiffError> {
        if self.graph.id != other.graph.id {
            return Err(AutodiffError::TapeMismatch);
        }

        let (a_val, b_val, requires_grad) = {
            let nodes = self.graph.nodes.borrow();
            let a_node = &nodes[self.id.0];
            let b_node = &nodes[other.id.0];
            (
                a_node.value.clone(),
                b_node.value.clone(),
                a_node.requires_grad || b_node.requires_grad,
            )
        };

        let (value, out_shape) = forward_binary(self.graph.tape, op, &a_val, &b_val)?;
        if value.shape() != out_shape.as_slice() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }

        let op_variant = match op {
            BinOpF64::Add => OpF64::Add(self.id, other.id),
            BinOpF64::Mul => OpF64::Mul(self.id, other.id),
            BinOpF64::Div => OpF64::Div(self.id, other.id),
            BinOpF64::Pow => OpF64::Pow(self.id, other.id),
        };

        let mut nodes = self.graph.nodes.borrow_mut();
        let id = NodeIdF64(nodes.len());
        nodes.push(NodeF64 {
            op: op_variant,
            shape: out_shape,
            value,
            requires_grad,
        });
        drop(nodes);
        Ok(VarF64 {
            graph: self.graph,
            id,
        })
    }
}

// ---------------------------------------------------------------------
// forward dispatch
// ---------------------------------------------------------------------

/// 出力 shape の確保前サイズ検査（`crate::bool_ops::checked_bytes_for`
/// と同型の独立複製。要素数積の `usize` オーバーフローに加え、`f64`
/// 換算のバイトサイズが `Vec` の allocation 上限（`isize::MAX` バイト）
/// に収まるかも検査する。要素数 1 の `VarF64` を巨大 shape へ
/// ブロードキャストした場合に `Vec::with_capacity`／`Tensor::contiguous`
/// が無検査確保で panic するのを、確保前に型付きエラーで拒否する
/// （本番経路 panic 禁止規約 `.claude/rules/coding-rust.md`。OWASP A03）。
fn checked_bytes_for_f64(shape: &[usize]) -> Result<(), ShapeError> {
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)?;
    let elem_size = std::mem::size_of::<f64>();
    let bytes = numel
        .checked_mul(elem_size)
        .ok_or(ShapeError::ElementCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(())
}

/// `a`／`b`（NumPy 互換ブロードキャスト可能な任意 shape）に `f` を
/// 要素ごとに適用したホスト参照実装。`div`／`pow` の唯一の forward
/// 経路（`TypedOps<f64>` に演算が存在しないため）であり、`add`／`mul`
/// の `typed_ops_f64()` が `None`／`Unsupported` のときのフォール
/// バック先でもある。
fn host_binary_elementwise(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Tensor<f64>, ShapeError> {
    let out_shape = elementwise_out_shape(a.shape(), b.shape())?;
    checked_bytes_for_f64(&out_shape)?;
    let a_bc = a.broadcast_to(&out_shape)?.contiguous();
    let b_bc = b.broadcast_to(&out_shape)?.contiguous();
    let a_slice = a_bc.host_slice();
    let b_slice = b_bc.host_slice();
    let data: Vec<f64> = a_slice
        .iter()
        .zip(b_slice.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Tensor::new(data, &out_shape)
}

/// `add`／`mul` の共通 dispatch: `tape.typed_ops_f64()` が `Some` なら
/// ネイティブ実装（CPU／CUDA）を呼び、`Err(Unsupported)` または `None`
/// ならホスト参照実装へフォールバックする。`Unsupported` 以外の
/// バックエンドエラーはそのまま `AutodiffError::Backend` として伝播し
/// 握り潰さない（f32 へ黙ってフォールバックしない契約）。
fn native_or_host_binary(
    tape: &Tape,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    native: impl FnOnce(&dyn TypedOps<f64>) -> Result<Tensor<f64>, BackendError>,
    host_fn: impl Fn(f64, f64) -> f64,
) -> Result<Tensor<f64>, AutodiffError> {
    if let Some(ops) = tape.typed_ops_f64() {
        match native(ops) {
            Ok(value) => return Ok(value),
            Err(BackendError::Unsupported(_)) => {}
            Err(err) => return Err(AutodiffError::Backend(err)),
        }
    }
    host_binary_elementwise(a, b, host_fn).map_err(AutodiffError::Shape)
}

/// 二項演算 4 種の forward 本体。出力 shape を先に確定・検査してから
/// （shape 検証と実行の分離。`docs/fusion-graph-design.md` §3.5.1 と
/// 同じ設計方針）演算を実行する。戻り値は `(計算結果, 出力 shape)`。
fn forward_binary(
    tape: &Tape,
    op: BinOpF64,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
) -> Result<(Tensor<f64>, Vec<usize>), AutodiffError> {
    let out_shape = elementwise_out_shape(a.shape(), b.shape()).map_err(AutodiffError::Shape)?;
    checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
    let value = match op {
        BinOpF64::Add => native_or_host_binary(tape, a, b, |ops| ops.add(a, b), |x, y| x + y)?,
        BinOpF64::Mul => native_or_host_binary(tape, a, b, |ops| ops.mul(a, b), |x, y| x * y)?,
        BinOpF64::Div => {
            host_binary_elementwise(a, b, |x, y| x / y).map_err(AutodiffError::Shape)?
        }
        BinOpF64::Pow => {
            host_binary_elementwise(a, b, |x, y| x.powf(y)).map_err(AutodiffError::Shape)?
        }
    };
    Ok((value, out_shape))
}

/// 勾配の蓄積（fan-out 合流）。`add` と同じ dispatch 規則
/// （native → host fallback）を使うため、CPU ネイティブ経路とホスト
/// 経路で bit が一致する（`.claude/rules/coding-rust.md` の f64
/// アキュムレータ契約と同じ「経路によらず同じ結合順序」という趣旨を、
/// 2 項の単純加算という最小形で満たす）。
fn accumulate_f64(
    tape: &Tape,
    existing: Option<Tensor<f64>>,
    contribution: Tensor<f64>,
) -> Result<Tensor<f64>, AutodiffError> {
    match existing {
        None => Ok(contribution),
        Some(acc) => native_or_host_binary(
            tape,
            &acc,
            &contribution,
            |ops| ops.add(&acc, &contribution),
            |x, y| x + y,
        ),
    }
}

// ---------------------------------------------------------------------
// VJP（backward 側）
// ---------------------------------------------------------------------

/// [`BinOpF64`] の VJP 係数 `(da, db)`（`d/da[op(a,b)]`・`d/db[op(a,b)]`）。
/// `crate::eval::scalar::binary_partials`（f32 版）と同じ式を `f64` で
/// 書き写したもの（PR #1686 codex-review 指摘の overflow/underflow
/// 耐性のある変形・`b == 0`／`a == 0` マスクを含む）。`y` は forward
/// 出力値（`Pow` の `db` が再計算を避けて再利用する）。
fn binary_partials_f64(op: BinOpF64, a: f64, b: f64, y: f64) -> (f64, f64) {
    match op {
        BinOpF64::Add => (1.0, 1.0),
        BinOpF64::Mul => (b, a),
        // `db = -a/b^2` を `-(a/b)/b` へ変形（overflow/underflow 耐性。
        // `eval::scalar::binary_partials` の `Div` と同じ理由）。
        BinOpF64::Div => (1.0 / b, -(a / b) / b),
        BinOpF64::Pow => {
            // `b == 0.0` は forward が定数関数（`a^0 = 1`）になるため da
            // は常に 0（ガードなしだと `a == 0` かつ `b == 0` で
            // `0 * inf = NaN`）。`a == 0.0` も同型の理由で db を 0 に
            // マスクする（`eval::scalar::binary_partials` の `Pow` と
            // 同じ規約）。
            let da = if b == 0.0 { 0.0 } else { b * a.powf(b - 1.0) };
            let db = if a == 0.0 { 0.0 } else { y * a.ln() };
            (da, db)
        }
    }
}

/// ブロードキャストの逆演算（`crate::grad::reduce_to_shape` の f64
/// 版）。`add`／`mul`／`div`／`pow` の VJP が返す勾配は forward 出力の
/// shape（ブロードキャスト後）を持つため、元の入力 shape
/// （`target_shape`）へ縮約する。NumPy 風ブロードキャストが複製した
/// 軸集合（先頭に新設された軸・入力側が size 1 だった軸）を、ホスト上
/// の `f64` 逐次和（row-major index 順）で合計して潰す。出力 dtype が
/// 既に `f64` であるため、`.claude/rules/coding-rust.md` の f64
/// アキュムレータ契約を昇格なしに満たす。
fn reduce_to_shape_f64(
    g: &Tensor<f64>,
    target_shape: &[usize],
) -> Result<Tensor<f64>, AutodiffError> {
    let g_shape = g.shape().to_vec();
    if g_shape == target_shape {
        return Ok(g.clone());
    }
    debug_assert!(
        g_shape.len() >= target_shape.len(),
        "reduce_to_shape_f64: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = g_shape.len() - target_shape.len();
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let g_c = g.contiguous();
    let mut data: Vec<f64> = g_c.host_slice().into_owned();
    let mut cur_shape = g_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            let mut reduced = vec![0f64; outer * inner];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + a) * inner + i;
                        reduced[o * inner + i] += data[src];
                    }
                }
            }
            data = reduced;
            cur_shape[axis] = 1;
        }
    }
    // `data.len()` は上記の縮約ロジックにより常に
    // `target_shape.iter().product()` と一致する構成（`crate::grad::
    // reduce_to_shape` と同型）だが、本番経路 panic 禁止規約
    // （`.claude/rules/coding-rust.md`）に従い `unwrap`／`unreachable!`
    // で握り潰さず、`Tensor::new` の検査結果をそのまま型付きエラーとして
    // 呼び出し元（`binary_vjp_f64`）へ伝播する。
    Tensor::new(data, target_shape).map_err(AutodiffError::Shape)
}

/// 二項演算 4 種の VJP 本体。`a`／`b`（forward 入力値。元の shape）・
/// `y`（forward 出力値。`out_shape`）・`g`（upstream 勾配。`out_shape`）
/// を受け取り、`a`／`b` それぞれの元 shape へ縮約済みの勾配を返す。
fn binary_vjp_f64(
    op: BinOpF64,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    y: &Tensor<f64>,
    g: &Tensor<f64>,
) -> Result<(Tensor<f64>, Tensor<f64>), AutodiffError> {
    let out_shape = g.shape().to_vec();
    let a_bc = a
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let b_bc = b
        .broadcast_to(&out_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let y_c = y.contiguous();
    let g_c = g.contiguous();
    let a_slice = a_bc.host_slice();
    let b_slice = b_bc.host_slice();
    let y_slice = y_c.host_slice();
    let g_slice = g_c.host_slice();

    let n = a_slice.len();
    let mut da_full = Vec::with_capacity(n);
    let mut db_full = Vec::with_capacity(n);
    for i in 0..n {
        let (da_coeff, db_coeff) = binary_partials_f64(op, a_slice[i], b_slice[i], y_slice[i]);
        da_full.push(g_slice[i] * da_coeff);
        db_full.push(g_slice[i] * db_coeff);
    }
    let da_full_t = Tensor::new(da_full, &out_shape).map_err(AutodiffError::Shape)?;
    let db_full_t = Tensor::new(db_full, &out_shape).map_err(AutodiffError::Shape)?;

    Ok((
        reduce_to_shape_f64(&da_full_t, a.shape())?,
        reduce_to_shape_f64(&db_full_t, b.shape())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    fn new_tape() -> Tape {
        Tape::new_with_ops(naive_ops())
    }

    fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
        Tensor::new(data, shape).expect("test fixture: shape 構築に失敗した")
    }

    #[test]
    fn leaf_preserves_value_and_shape() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        assert_eq!(x.shape(), vec![3]);
        assert_eq!(x.value().host_slice().into_owned(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn var_no_grad_rejects_get() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var_no_grad(&t(vec![1.0], &[]));
        let y = graph.var(&t(vec![2.0], &[]));
        let z = x.add(&y).expect("add は成功するはず");
        let grads = graph.backward(&z).expect("backward は成功するはず");
        assert!(matches!(
            grads.get(&x),
            Err(AutodiffError::GradientTrackingDisabled)
        ));
        // `x` は追跡なしだが `y` は追跡ありのため、`z` の
        // `requires_grad` は OR で true になり `y` 側は取得できる。
        assert!(grads.get(&y).expect("y は追跡対象のはず").is_some());
    }

    #[test]
    fn backward_add_mul_div_pow_matches_host_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![2.0], &[]));
        let w = graph.var(&t(vec![3.0], &[]));

        // y = x*w + x/w
        let mul = x.mul(&w).expect("mul");
        let div = x.div(&w).expect("div");
        let y = mul.add(&div).expect("add");
        let grads = graph.backward(&y).expect("backward");

        // dy/dx = w + 1/w, dy/dw = x - x/w^2
        let dx = grads.get(&x).unwrap().unwrap();
        let dw = grads.get(&w).unwrap().unwrap();
        let expected_dx = 3.0 + 1.0 / 3.0;
        let expected_dw = 2.0 - 2.0 / (3.0 * 3.0);
        assert_eq!(dx.host_slice().into_owned(), vec![expected_dx]);
        assert_eq!(dw.host_slice().into_owned(), vec![expected_dw]);
    }

    #[test]
    fn pow_zero_base_and_exponent_masks_gradient() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![0.0], &[]));
        let b = graph.var(&t(vec![0.0], &[]));
        let y = a.pow(&b).expect("pow");
        assert_eq!(y.value().host_slice().into_owned(), vec![1.0]);
        let grads = graph.backward(&y).expect("backward");
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        assert_eq!(da.host_slice().into_owned(), vec![0.0]);
        assert_eq!(db.host_slice().into_owned(), vec![0.0]);
    }

    #[test]
    fn div_by_zero_forward_is_inf_not_panic() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![1.0], &[]));
        let b = graph.var(&t(vec![0.0], &[]));
        let y = a.div(&b).expect("div は panic せず成功するはず");
        assert!(y.value().host_slice()[0].is_infinite());
    }

    #[test]
    fn broadcast_add_reduces_gradient_by_sequential_sum() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = graph.var(&t(vec![10.0, 20.0, 30.0], &[3]));
        let y = a.add(&b).expect("add");
        let grads = graph.backward(&y).expect("backward");
        let db = grads.get(&b).unwrap().unwrap();
        // `[2,3]` の全 1 勾配を軸 0（長さ 2）で縮約すると各要素 2.0。
        assert_eq!(db.host_slice().into_owned(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn fan_out_accumulates_gradient() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![2.0], &[]));
        // y = x + x + x → dy/dx = 3
        let y = x.add(&x).expect("add").add(&x).expect("add");
        let grads = graph.backward(&y).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![3.0]);
    }

    #[test]
    fn cross_graph_operands_are_rejected() {
        let tape = new_tape();
        let graph_a = TapeF64::new(&tape);
        let graph_b = TapeF64::new(&tape);
        let x = graph_a.var(&t(vec![1.0], &[]));
        let y = graph_b.var(&t(vec![1.0], &[]));
        assert!(matches!(x.add(&y), Err(AutodiffError::TapeMismatch)));
    }

    #[test]
    fn backward_rejects_loss_without_requires_grad() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var_no_grad(&t(vec![1.0], &[]));
        let b = graph.var_no_grad(&t(vec![2.0], &[]));
        let loss = a.add(&b).expect("add");
        assert!(matches!(
            graph.backward(&loss),
            Err(AutodiffError::Backward(_))
        ));
    }

    #[test]
    fn shape_mismatch_is_rejected_before_allocation() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![1.0, 2.0], &[2]));
        let b = graph.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        assert!(matches!(a.add(&b), Err(AutodiffError::Shape(_))));
    }
}
