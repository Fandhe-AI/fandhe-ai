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
//! `add`／`mul`／`matmul`／`sum`／`max` は
//! [`crate::tape::Tape::typed_ops_f64`]（`BackendOps` の capability
//! accessor）が `Some` を返せばそちらへ委譲し、`None` または
//! `BackendError::Unsupported` が返った場合はホスト上の f64 参照実装へ
//! フォールバックする（イシュー #2196 で `matmul`／`sum`／`max` を追加。
//! CPU（#1697）・CUDA（#2060）はいずれもネイティブ実装を返す。Metal は
//! MSL の `double` 非対応が恒久的なため常に `None` を返し、ホスト経路へ
//! 到達する）。**`div`／`pow`／`mean` は `TypedOps<f64>` に演算が存在
//! しない**（trait は `gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／
//! `max` の 8 演算に固定済み。`crates/tensor-core/src/typed_ops.rs`
//! 参照）ため、`div`／`pow` はバックエンドに依らず常にホスト上の f64
//! 参照実装で計算し、`mean` は `sum` の dispatch 結果をホスト上で
//! 縮約対象要素数 `n` により 1 回だけ除算する合成実装とする（`Var::mean`
//! と同じ「sum の後に 1 回だけ除算する」丸め規律）。f32 へ黙って
//! フォールバックすることは決してしない（`Unsupported` 以外のバック
//! エンドエラーはそのまま伝播する）。
//!
//! 演算 × バックエンドの dispatch 表（イシュー #2196）:
//!
//! | 演算 | CPU／CUDA | Metal |
//! |------|-----------|-------|
//! | `add`／`mul`／`matmul`／`sum`／`max` | ネイティブ（`TypedOps<f64>`） | ホスト（`typed_ops_f64()` が常に `None`） |
//! | `div`／`pow` | ホスト（`TypedOps<f64>` に演算なし） | ホスト |
//! | `mean` | ネイティブ `sum` + ホスト除算 1 回 | ホスト `sum` + ホスト除算 1 回 |
//!
//! # 数値契約
//!
//! すべての値は eager に `f64` のまま計算する（融合・遅延実行は行わ
//! ない。`FusionPlan` は f32 固定のまま不変）。ブロードキャストの逆
//! 演算（`reduce_to_shape_f64`・`unreduce_broadcast_f64`）はホスト上の
//! `f64` 逐次和（row-major index 順）で実装し、出力 dtype が既に `f64`
//! であるため `.claude/rules/coding-rust.md` の f64 アキュムレータ契約
//! を自明に満たす。勾配の蓄積（fan-out 合流）も [`TapeF64::backward`]
//! 内で add と同じ dispatch 規則（native → host fallback）を通すため、
//! CPU ネイティブ経路とホスト経路で bit が一致する。
//!
//! # bit 一致の境界（イシュー #2196）
//!
//! - `matmul`（rank 2 限定）: ホスト参照実装（`host_gemm_f64`）は出力
//!   要素ごとに `p`（縮約軸）昇順で `f64::mul_add` を適用する ikj 順の
//!   累積であり、CPU ネイティブ実装（`crate::tape::Tape::
//!   typed_ops_f64` 経由。CPU バックエンドクレートの `gemm_row_parallel_f64`）・
//!   同クレートの `matmul_reference_fma_f64`（`fandhe-ai-backend-cpu`
//!   の parity ユーティリティ）の 3 者は常に bit 完全一致する
//!   （`autodiff` クレート自体はバックエンド crate へ直接依存しない
//!   設計上の不変条件のため、本 doc も crate 識別子を直接書かない）。
//!   rank≥3 のバッチ matmul は本イシューの対象外（`gemm_out_shape` が
//!   `ShapeError::RankMismatch` で拒否する）。
//! - `sum`（軸指定 `Some(axis)`）: ホスト・CPU ネイティブとも縮約軸を
//!   昇順で逐次和する構造が同一のため、任意の要素数で bit 一致する。
//! - `sum`（全軸 `None`）: CPU ネイティブは `CHUNK = 4096`
//!   （`crates/backend-cpu/src/reduction.rs`）単位の 2 段構成（チャンク
//!   内逐次・チャンク間固定順結合）である一方、ホスト参照実装は単純な
//!   単一逐次和である。**両者の bit 一致は縮約対象要素数が
//!   `CHUNK`（4096）以下の場合に限って成立する**（`CHUNK` は
//!   `pub(crate)` で本モジュールから参照できないためハードコードで
//!   再現せず、境界を doc に明記するに留める）。
//! - `max`: NaN は伝播しない（`f64::max` の既知の挙動。CPU ネイティブと
//!   同じ既知事項）。同値タイは「走査順で最初に一致した要素」へ勾配を
//!   伝播する（`max_first_match_vjp_f64`。`grad.rs::
//!   extremum_first_match_vjp` の f64 版・PyTorch `max(dim)` と同じ
//!   決定的方式）。`max` の値そのものは走査順に依らないため（NaN を
//!   除き）常に bit 一致する。
//!
//! # 次段への申し送り
//!
//! 本イシュー（#2196）のスコープ外（`docs/autodiff-var-dtype-multiplexing-
//! design.md` §10 承認事項が未承認のまま）:
//!
//! - CUDA／Metal のネイティブ f64 カーネル追加・`TypedOps<f64>` への
//!   演算追加（バッチ matmul・`mean`・`div`・`pow`）
//! - facade から本モジュールを直接使えるようにする再エクスポート
//!   （`TapeF64`／`VarF64` は内部クレート限定 `pub` のまま。facade の
//!   `typed_ops_f64()` accessor 経由での到達確認のみが本イシューの
//!   「facade 到達」の範囲）
//! - rank≥3 のバッチ matmul・`keepdim`・多軸縮約（`sum_dims` 相当）・
//!   `min`

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use fandhe_ai_tensor_core::{
    BackendError, ShapeError, Tensor, TypedOps, elementwise_out_shape, gemm_out_shape,
    reduce_out_shape,
};

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
    /// 行列積（rank 2 限定。イシュー #2196）。`gemm_out_shape` が
    /// rank≠2 を `ShapeError::RankMismatch` で拒否するため、本 variant
    /// が記録するノードは常に rank 2 オペランドを持つ。
    MatMul(NodeIdF64, NodeIdF64),
    /// 縮約和（`dim: None` は全軸縮約。イシュー #2196）。
    Sum {
        input: NodeIdF64,
        dim: Option<usize>,
    },
    /// 縮約平均（`sum` の除算 1 回による合成。イシュー #2196）。
    Mean {
        input: NodeIdF64,
        dim: Option<usize>,
    },
    /// 縮約最大値（イシュー #2196）。
    Max {
        input: NodeIdF64,
        dim: Option<usize>,
    },
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

/// [`TapeF64::backward`] の逆走査 1 ステップが扱う演算種別（イシュー
/// #2196）。[`OpF64`] から VJP 計算に必要な最小情報だけを取り出した
/// 軽量タグであり、寄与先ノード数（二項演算・`MatMul` は 2、`Sum`／
/// `Mean`／`Max` は 1）に依らず `backward` ループを 1 つの `match` で
/// 統一的に扱うために使う。
enum OpKindF64 {
    Binary(BinOpF64, NodeIdF64, NodeIdF64),
    MatMul(NodeIdF64, NodeIdF64),
    Sum {
        input: NodeIdF64,
        dim: Option<usize>,
    },
    Mean {
        input: NodeIdF64,
        dim: Option<usize>,
    },
    Max {
        input: NodeIdF64,
        dim: Option<usize>,
    },
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
            // `var.rs` モジュール doc と同じ実装規律）。演算種別に応じて
            // 寄与先ノード数が異なる（二項演算・`MatMul` は 2、`Sum`／
            // `Mean`／`Max` は 1）ため、`(寄与先, 寄与勾配)` の可変長
            // ベクトルへ統一する（イシュー #2196 で `Op` 種別を拡張した
            // 際の一般化）。
            let kind = {
                let nodes = self.nodes.borrow();
                match nodes[id].op {
                    OpF64::Leaf => None,
                    OpF64::Add(lhs, rhs) => Some(OpKindF64::Binary(BinOpF64::Add, lhs, rhs)),
                    OpF64::Mul(lhs, rhs) => Some(OpKindF64::Binary(BinOpF64::Mul, lhs, rhs)),
                    OpF64::Div(lhs, rhs) => Some(OpKindF64::Binary(BinOpF64::Div, lhs, rhs)),
                    OpF64::Pow(lhs, rhs) => Some(OpKindF64::Binary(BinOpF64::Pow, lhs, rhs)),
                    OpF64::MatMul(lhs, rhs) => Some(OpKindF64::MatMul(lhs, rhs)),
                    OpF64::Sum { input, dim } => Some(OpKindF64::Sum { input, dim }),
                    OpF64::Mean { input, dim } => Some(OpKindF64::Mean { input, dim }),
                    OpF64::Max { input, dim } => Some(OpKindF64::Max { input, dim }),
                }
            };
            let Some(kind) = kind else {
                // 葉ノードへは入力がないため寄与を戻す先がない
                // （`grads[id]` は上で複製済みのため、このノード自身の
                // 最終値は `grads` に残ったまま）。
                continue;
            };

            let contributions: Vec<(NodeIdF64, Tensor<f64>)> = match kind {
                OpKindF64::Binary(op_kind, lhs, rhs) => {
                    let (a_val, b_val, y_val) = {
                        let nodes = self.nodes.borrow();
                        (
                            nodes[lhs.0].value.clone(),
                            nodes[rhs.0].value.clone(),
                            nodes[id].value.clone(),
                        )
                    };
                    let (da, db) = binary_vjp_f64(op_kind, &a_val, &b_val, &y_val, &grad_g)?;
                    vec![(lhs, da), (rhs, db)]
                }
                OpKindF64::MatMul(lhs, rhs) => {
                    let (a_val, b_val) = {
                        let nodes = self.nodes.borrow();
                        (nodes[lhs.0].value.clone(), nodes[rhs.0].value.clone())
                    };
                    let (da, db) = matmul_vjp_f64(self.tape, &a_val, &b_val, &grad_g)?;
                    vec![(lhs, da), (rhs, db)]
                }
                OpKindF64::Sum { input, dim } => {
                    let input_shape = self.nodes.borrow()[input.0].shape.clone();
                    let da = unreduce_broadcast_f64(&grad_g, &input_shape, dim)?;
                    vec![(input, da)]
                }
                OpKindF64::Mean { input, dim } => {
                    let input_shape = self.nodes.borrow()[input.0].shape.clone();
                    let da = mean_vjp_f64(&grad_g, &input_shape, dim)?;
                    vec![(input, da)]
                }
                OpKindF64::Max { input, dim } => {
                    let (input_val, out_val) = {
                        let nodes = self.nodes.borrow();
                        (nodes[input.0].value.clone(), nodes[id].value.clone())
                    };
                    let da = max_first_match_vjp_f64(&input_val, dim, &out_val, &grad_g)?;
                    vec![(input, da)]
                }
            };

            // イシュー #1748 と同じ規約: `requires_grad == false` の
            // ノードへの寄与は `accumulate` へ渡さず捨てる。同一ノードへ
            // 複数回寄与する場合（例: `x.matmul(&x)`）も、二項演算と同じ
            // 逐次 `accumulate_f64` 呼び出しで扱われるため fan-out 蓄積の
            // 意味論は変わらない。
            for (target_id, contribution) in contributions {
                let requires_grad = self.nodes.borrow()[target_id.0].requires_grad;
                if requires_grad {
                    let existing = grads[target_id.0].take();
                    grads[target_id.0] = Some(accumulate_f64(self.tape, existing, contribution)?);
                }
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
/// 版）。値そのものではなく `NodeIdF64` + [`TapeF64`] への共有参照を
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

    /// 行列積 `self @ other`（rank 2 限定。イシュー #2196）。`Var::matmul`
    /// （f32 版）と同じ「①クロステープ検査 → ②入力値取得 → ③forward
    /// dispatch → ④出力 shape 再検査 → ⑤ノード記録」の順で処理する。
    /// rank≠2 は `forward_matmul` 内の `gemm_out_shape` が
    /// `ShapeError::RankMismatch` で拒否する（バッチ matmul は本
    /// イシューの対象外。§10 承認事項 2 未承認のため `TypedOps<f64>` に
    /// バッチ版がない）。
    pub fn matmul(&self, other: &Self) -> Result<Self, AutodiffError> {
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
        let (value, out_shape) = forward_matmul(self.graph.tape, &a_val, &b_val)?;
        self.push_node(
            OpF64::MatMul(self.id, other.id),
            value,
            out_shape,
            requires_grad,
        )
    }

    /// 縮約和（`dim: None` は全軸縮約。イシュー #2196）。空縮約
    /// （要素数 0）は `0.0` を返す（`TypedOps<f64>::sum` の空縮約契約
    /// と同じ。`max` とは異なり単位元 `0.0` を持つため拒否しない）。
    pub fn sum(&self, dim: Option<usize>) -> Result<Self, AutodiffError> {
        let (a_val, requires_grad) = {
            let nodes = self.graph.nodes.borrow();
            let node = &nodes[self.id.0];
            (node.value.clone(), node.requires_grad)
        };
        let (value, out_shape) = forward_sum(self.graph.tape, &a_val, dim)?;
        self.push_node(
            OpF64::Sum {
                input: self.id,
                dim,
            },
            value,
            out_shape,
            requires_grad,
        )
    }

    /// 縮約平均（`dim: None` は全軸縮約。イシュー #2196）。`n == 0`
    /// （縮約対象の要素数が 0）は [`AutodiffError::InvalidArgument`] で
    /// 拒否する（`Var::mean`〈f32 版〉と同じ「値が定義されない縮約は
    /// 安全側で明示的に拒否する」方針）。`sum` の dispatch 結果を
    /// ホスト上で `n` により 1 回だけ除算する合成実装（`TypedOps<f64>`
    /// に `mean` 演算はない）。
    pub fn mean(&self, dim: Option<usize>) -> Result<Self, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim).map_err(AutodiffError::Shape)?;
        let n: usize = match dim {
            None => shape.iter().product(),
            Some(axis) => shape[axis],
        };
        if n == 0 {
            return Err(AutodiffError::InvalidArgument(
                "VarF64::mean: 縮約対象の要素数が 0 です".to_string(),
            ));
        }
        let (a_val, requires_grad) = {
            let nodes = self.graph.nodes.borrow();
            let node = &nodes[self.id.0];
            (node.value.clone(), node.requires_grad)
        };
        let (sum_value, out_shape) = forward_sum(self.graph.tape, &a_val, dim)?;
        let sum_c = sum_value.contiguous();
        let divided: Vec<f64> = sum_c.host_slice().iter().map(|&v| v / n as f64).collect();
        let value = Tensor::new(divided, &out_shape).map_err(AutodiffError::Shape)?;
        self.push_node(
            OpF64::Mean {
                input: self.id,
                dim,
            },
            value,
            out_shape,
            requires_grad,
        )
    }

    /// 縮約最大値（`dim: None` は全軸縮約。イシュー #2196）。空縮約
    /// （`dim: None` で全体が空、または `Some(axis)` で当該軸が長さ 0
    /// かつ出力要素数 > 0）は単位元を持たないため
    /// [`AutodiffError::InvalidArgument`] で拒否する（`TypedOps<f64>::
    /// max`〈CPU ネイティブ〉が `BackendError::KernelLaunchFailed` を
    /// 返す同じ状況を、native／host いずれの経路でも同一のエラー型
    /// として拒否するために dispatch 前に検査する）。NaN は伝播しない
    /// （`f64::max` の既知の挙動）。同値タイは走査順で最初に一致した
    /// 要素へ勾配が伝播する（`max_first_match_vjp_f64`）。
    pub fn max(&self, dim: Option<usize>) -> Result<Self, AutodiffError> {
        let shape = self.shape();
        let out_shape = reduce_out_shape(&shape, dim).map_err(AutodiffError::Shape)?;
        if is_empty_reduction_f64(&shape, dim, &out_shape) {
            return Err(AutodiffError::InvalidArgument(
                "VarF64::max: 縮約対象の要素数が 0 です".to_string(),
            ));
        }
        let (a_val, requires_grad) = {
            let nodes = self.graph.nodes.borrow();
            let node = &nodes[self.id.0];
            (node.value.clone(), node.requires_grad)
        };
        let (value, out_shape) = forward_max(self.graph.tape, &a_val, dim)?;
        self.push_node(
            OpF64::Max {
                input: self.id,
                dim,
            },
            value,
            out_shape,
            requires_grad,
        )
    }

    /// 演算実行結果をノードとして記録する共通末尾処理（`matmul`／
    /// `sum`／`mean`／`max` が共有。`binary_op` は二項演算固有の
    /// `op_variant` 分岐を持つため独立のまま維持する）。forward
    /// dispatch が返した `value` の shape が `out_shape`（呼び出し元が
    /// 事前に検証した期待 shape）と一致するかを再検査してから push する
    /// （`binary_op` と同じ「実行結果の shape 再検査」規律）。
    fn push_node(
        &self,
        op: OpF64,
        value: Tensor<f64>,
        out_shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<Self, AutodiffError> {
        if value.shape() != out_shape.as_slice() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let mut nodes = self.graph.nodes.borrow_mut();
        let id = NodeIdF64(nodes.len());
        nodes.push(NodeF64 {
            op,
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

/// 部分次元列（`outer`／`inner` 等）の要素数積を overflow 検査付きで
/// 計算する共通ヘルパー（`host_sum_f64`／`host_max_f64`／
/// `max_first_match_vjp_f64` 共用。codex-review／Cursor Bugbot 指摘是正。
/// イシュー #2196）。`out_shape`（縮約軸を除いた shape）全体の積が
/// `checked_bytes_for_f64` を通過していても、`outer`（縮約軸より前の
/// 次元列）と `inner`（縮約軸より後ろの次元列）は互いに独立な
/// `.iter().product()` として再計算されるため、`out_shape` 側の検査
/// （0 次元を経由すると overflow を素通りしうる逐次 `checked_mul` の
/// 累積）では守れない。例えば `out_shape = [0, usize::MAX, usize::MAX]`
/// は全体積が 0 で `checked_bytes_for_f64` を通過するが、`inner`（`0`
/// を含まない後半部分列 `[usize::MAX, usize::MAX]`）の素の `.product()`
/// は単独で overflow する。本関数は `outer`／`inner` それぞれを
/// `checked_mul` で独立に検査することで、この 0 次元の手前の部分列に
/// 依存しない overflow 検出を行う（本番経路 panic 禁止規約
/// `.claude/rules/coding-rust.md`。OWASP A03）。
fn checked_product(dims: &[usize]) -> Result<usize, ShapeError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
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

/// `matmul` の共通 dispatch（`native_or_host_binary` の gemm 版。イシュー
/// #2196）。`tape.typed_ops_f64()` が `Some` なら `ops.gemm` を呼び、
/// `Err(Unsupported)` または `None` ならホスト参照実装
/// （[`host_gemm_f64`]）へフォールバックする。`Unsupported` 以外の
/// バックエンドエラーはそのまま `AutodiffError::Backend` として伝播する
/// （f32 へ黙ってフォールバックしない契約は `add`／`mul` と同一）。
fn native_or_host_gemm(
    tape: &Tape,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
) -> Result<Tensor<f64>, AutodiffError> {
    if let Some(ops) = tape.typed_ops_f64() {
        match ops.gemm(a, b) {
            Ok(value) => return Ok(value),
            Err(BackendError::Unsupported(_)) => {}
            Err(err) => return Err(AutodiffError::Backend(err)),
        }
    }
    host_gemm_f64(a, b).map_err(AutodiffError::Shape)
}

/// `matmul`（rank 2 限定）のホスト参照実装（イシュー #2196）。出力要素
/// ごとに縮約軸 `p` を昇順で `f64::mul_add` により累積する ikj 順（`i`
/// 行固定・`p` 縮約軸・`j` 列の順で走査）で、CPU ネイティブ
/// （`crates/backend-cpu/src/typed_f64.rs::gemm_row_parallel_f64`）・
/// 同クレートの `matmul_reference_fma_f64`（parity ユーティリティ）と
/// bit 完全一致する（モジュール doc「bit 一致の境界」参照）。`m == 0`／`n == 0`／
/// `k == 0` はループが自然に空になり panic しない。ただし `n == 0`
/// （出力が空）のとき `m` 自体が巨大（`usize::MAX` 等）だと、`i` 行
/// ループは panic しないまま `m` 回逐次走査してしまい実質ハングする
/// （`gemm_row_parallel_f64` が `n == 0` を早期 return で弾く契約と
/// 揃っていなかった。codex-review 指摘是正。イシュー #2196）。出力
/// 要素数（`= m * n`）が 0 の時点で走査すべき要素が存在しないため、
/// 確保済みの空 `data` をそのまま返す早期 return で対処する。
fn host_gemm_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, ShapeError> {
    let out_shape = gemm_out_shape(a.shape(), b.shape())?;
    checked_bytes_for_f64(&out_shape)?;
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    // `out_shape` のバイトサイズは `checked_bytes_for_f64` で確保前検査
    // 済みのため、`m * n`（`out_shape` の要素数積そのもの）は
    // overflow しない（`gemm_out_shape` が `checked_numel` で同じ検査を
    // 行っている二重の裏付けでもある）。
    let mut data = vec![0.0f64; m * n];
    // `m == 0` または `n == 0`（出力要素数 0）の場合、後続の `i` 行
    // ループを回すべき理由が無い。`m` が巨大（`usize::MAX` 等）で
    // `n == 0` の組合せは、走査対象が無いにもかかわらず `i` ループが
    // `m` 回逐次実行されて実質ハングする（`gemm_row_parallel_f64` が
    // `n == 0` を早期 return する契約と揃える。codex-review 指摘是正）。
    if data.is_empty() {
        return Tensor::new(data, &out_shape);
    }
    let a_c = a.contiguous();
    let b_c = b.contiguous();
    let a_slice = a_c.host_slice();
    let b_slice = b_c.host_slice();
    for i in 0..m {
        let a_row = &a_slice[i * k..i * k + k];
        let c_row = &mut data[i * n..i * n + n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b_slice[p * n..p * n + n];
            for j in 0..n {
                c_row[j] = a_ip.mul_add(b_row[j], c_row[j]);
            }
        }
    }
    Tensor::new(data, &out_shape)
}

/// `matmul` の forward 本体。出力 shape を先に確定・検査してから
/// （`forward_binary` と同じ「shape 検証と実行の分離」方針）
/// [`native_or_host_gemm`] へ委譲する。戻り値は `(計算結果, 出力 shape)`。
fn forward_matmul(
    tape: &Tape,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
) -> Result<(Tensor<f64>, Vec<usize>), AutodiffError> {
    let out_shape = gemm_out_shape(a.shape(), b.shape()).map_err(AutodiffError::Shape)?;
    checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
    let value = native_or_host_gemm(tape, a, b)?;
    Ok((value, out_shape))
}

/// `sum`／`max`（`dim: Option<usize>` を取る縮約 2 演算）の共通
/// dispatch（`native_or_host_binary` の縮約版。イシュー #2196）。
fn native_or_host_reduce(
    tape: &Tape,
    a: &Tensor<f64>,
    dim: Option<usize>,
    native: impl FnOnce(&dyn TypedOps<f64>) -> Result<Tensor<f64>, BackendError>,
    host_fn: impl Fn(&Tensor<f64>, Option<usize>) -> Result<Tensor<f64>, ShapeError>,
) -> Result<Tensor<f64>, AutodiffError> {
    if let Some(ops) = tape.typed_ops_f64() {
        match native(ops) {
            Ok(value) => return Ok(value),
            Err(BackendError::Unsupported(_)) => {}
            Err(err) => return Err(AutodiffError::Backend(err)),
        }
    }
    host_fn(a, dim).map_err(AutodiffError::Shape)
}

/// `sum` のホスト参照実装（イシュー #2196）。`dim: Some(axis)` は縮約軸
/// を昇順で逐次和するため CPU ネイティブと任意要素数で bit 一致する。
/// `dim: None` は単純な単一逐次和で、CPU ネイティブ（`CHUNK = 4096`
/// 単位の 2 段構成）との bit 一致は縮約対象要素数が `CHUNK` 以下の場合
/// に限る（モジュール doc「bit 一致の境界」参照）。
///
/// `outer`／`inner` の overflow 検査は `checked_product`（本モジュール
/// 冒頭寄り。`checked_bytes_for_f64` 直後）に委譲する。`out_shape`
/// （`= in_shape[..axis] ++ in_shape[axis+1..]`）全体の積が
/// `checked_bytes_for_f64` を通過していても、`outer`（縮約軸より前の
/// 部分列）と `inner`（縮約軸より後ろの部分列）を素の `.iter().product()`
/// で独立に再計算すると、0 次元を含む shape（例
/// `[0, usize::MAX, usize::MAX]`）で `inner` 単独の積が overflow しうる
/// （codex-review／Cursor Bugbot 指摘是正。`checked_product` の doc
/// comment 参照）。
fn host_sum_f64(a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, ShapeError> {
    let out_shape = reduce_out_shape(a.shape(), dim)?;
    checked_bytes_for_f64(&out_shape)?;
    let in_shape = a.shape().to_vec();
    let a_c = a.contiguous();
    let a_slice = a_c.host_slice();
    match dim {
        None => {
            let total = a_slice.iter().fold(0.0f64, |acc, &v| acc + v);
            Tensor::new(vec![total], &out_shape)
        }
        Some(axis) => {
            let outer = checked_product(&in_shape[..axis])?;
            let axis_len = in_shape[axis];
            let inner = checked_product(&in_shape[axis + 1..])?;
            let data_len = outer
                .checked_mul(inner)
                .ok_or(ShapeError::ElementCountOverflow)?;
            let mut data = vec![0.0f64; data_len];
            // 出力要素数（`data_len = outer * inner`）が 0 なら走査すべき
            // 出力が存在しない。走査順が `o -> i -> x`（`inner` が
            // `axis_len` より内側）のため `inner == 0` はここで自然に
            // 空になるが、`axis_reduce_f64`（`crates/backend-cpu/src/
            // typed_f64.rs`）が出力要素数 0 を明示的に走査しない契約と
            // 揃えるため、[`host_max_f64`] と同じ早期 return で明示する
            // （codex-review 指摘是正。イシュー #2196）。
            if data_len == 0 {
                return Tensor::new(data, &out_shape);
            }
            for o in 0..outer {
                for i in 0..inner {
                    let mut acc = 0.0f64;
                    for x in 0..axis_len {
                        let src = (o * axis_len + x) * inner + i;
                        acc += a_slice[src];
                    }
                    data[o * inner + i] = acc;
                }
            }
            Tensor::new(data, &out_shape)
        }
    }
}

/// `max` のホスト参照実装（イシュー #2196）。単位元 `f64::NEG_INFINITY`
/// から `f64::max` で fold する（NaN 非伝播は CPU ネイティブと同じ既知
/// 事項）。`outer`／`inner` の overflow 検査は [`host_sum_f64`] と同じ
/// `checked_product` に委譲する（0 次元を含む shape での独立部分積
/// overflow を個別検査する理由も同一。codex-review／Cursor Bugbot
/// 指摘是正）。呼び出し元（[`forward_max`]／[`VarF64::max`]）が
/// **縮約対象**の要素数 0（`axis_len == 0`）を dispatch 前に拒否する
/// 契約のため、本関数はその意味での空縮約を想定しない。ただし**出力**
/// 要素数 0（`outer == 0` または `inner == 0`。`axis_len` 自体は非 0
/// のまま巨大でもよい）は別の軸で発生しうる契約違反ではない正常系
/// であり、下記の早期 return で扱う（codex-review 指摘是正）。
fn host_max_f64(a: &Tensor<f64>, dim: Option<usize>) -> Result<Tensor<f64>, ShapeError> {
    let out_shape = reduce_out_shape(a.shape(), dim)?;
    checked_bytes_for_f64(&out_shape)?;
    let in_shape = a.shape().to_vec();
    let a_c = a.contiguous();
    let a_slice = a_c.host_slice();
    match dim {
        None => {
            let m = a_slice.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
            Tensor::new(vec![m], &out_shape)
        }
        Some(axis) => {
            let outer = checked_product(&in_shape[..axis])?;
            let axis_len = in_shape[axis];
            let inner = checked_product(&in_shape[axis + 1..])?;
            let data_len = outer
                .checked_mul(inner)
                .ok_or(ShapeError::ElementCountOverflow)?;
            let mut data = vec![f64::NEG_INFINITY; data_len];
            // 出力要素数（`data_len = outer * inner`）が 0 なら走査すべき
            // 出力が存在しない。走査順は `o -> x -> i`（`axis_len` が
            // `inner` より外側）のため、`inner == 0` かつ `axis_len` が
            // 巨大（例 `usize::MAX`）な shape（`[usize::MAX, 0]` に
            // `max(Some(0))` 等）では、早期 return が無いと出力へ書く
            // べき要素が 0 個にもかかわらず `x` ループを `axis_len` 回
            // 逐次走査してしまい実質ハングする。`axis_reduce_f64`
            // （`crates/backend-cpu/src/typed_f64.rs`。出力要素数を
            // `total_out` として走査対象を決めるため `total_out == 0`
            // では走査自体が発生しない）と挙動を揃える（codex-review
            // 指摘是正。イシュー #2196）。
            if data_len == 0 {
                return Tensor::new(data, &out_shape);
            }
            for o in 0..outer {
                for x in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + x) * inner + i;
                        let dst = o * inner + i;
                        data[dst] = data[dst].max(a_slice[src]);
                    }
                }
            }
            Tensor::new(data, &out_shape)
        }
    }
}

/// `sum` の forward 本体（[`forward_binary`]／[`forward_matmul`] と同じ
/// 「shape 検証と実行の分離」方針。イシュー #2196）。
fn forward_sum(
    tape: &Tape,
    a: &Tensor<f64>,
    dim: Option<usize>,
) -> Result<(Tensor<f64>, Vec<usize>), AutodiffError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(AutodiffError::Shape)?;
    checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
    let value = native_or_host_reduce(tape, a, dim, |ops| ops.sum(a, dim), host_sum_f64)?;
    Ok((value, out_shape))
}

/// `max` の forward 本体（[`forward_sum`] と同型）。空縮約の事前拒否は
/// 呼び出し元 [`VarF64::max`] の責務（[`is_empty_reduction_f64`]）。
fn forward_max(
    tape: &Tape,
    a: &Tensor<f64>,
    dim: Option<usize>,
) -> Result<(Tensor<f64>, Vec<usize>), AutodiffError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(AutodiffError::Shape)?;
    checked_bytes_for_f64(&out_shape).map_err(AutodiffError::Shape)?;
    let value = native_or_host_reduce(tape, a, dim, |ops| ops.max(a, dim), host_max_f64)?;
    Ok((value, out_shape))
}

/// `max` の空縮約判定（イシュー #2196）。CPU ネイティブ
/// （`crates/backend-cpu/src/typed_f64.rs::max_f64`）が
/// `axis_len == 0 && total_out > 0`（`dim: Some`）または全要素縮約で
/// 単位元が取れない（`dim: None`）場合に `ReduceError::EmptyReduction`
/// を返すのと同じ条件を、dispatch 前に検査してネイティブ／ホスト
/// いずれの経路でも同一のエラー型（`AutodiffError::InvalidArgument`）
/// で拒否できるようにする（判定迂回経路を作らない。
/// `.claude/rules/security.md` A08）。`out_shape` に 0 要素の軸が残る
/// 場合（出力自体が空）は書き込むべき要素がないため拒否しない。
fn is_empty_reduction_f64(in_shape: &[usize], dim: Option<usize>, out_shape: &[usize]) -> bool {
    let out_has_elements = !out_shape.contains(&0);
    if !out_has_elements {
        return false;
    }
    match dim {
        None => in_shape.contains(&0),
        Some(axis) => in_shape.get(axis).copied() == Some(0),
    }
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

    // 空テンソル（`numel == 0`）は縮約すべき要素が存在しないため、
    // 対象 shape のゼロ勾配として早期に返す（codex-review P1 是正・
    // PR #2253）。空でない場合、`outer * axis_len * inner` は常に
    // `data.len()`（縮約前の総要素数。`Tensor::new` で既に確保検査済み
    // で usize に収まることが保証済み）と一致するため、`outer * inner`
    // は `data.len()` 以下に収まり overflow しない。しかし空テンソル
    // では他の軸（今見ている `axis` を含まない軸）が 0 のために総要素数
    // が 0 になり得る一方、`axis` 自体の縮約対象次元やその他の非ゼロ
    // 次元は `usize::MAX` 級の値を取り得る（例: shape
    // `[0, usize::MAX, usize::MAX]`）。この場合 `outer * inner` の通常
    // 乗算は本番経路 panic 禁止規約（`.claude/rules/coding-rust.md`）に
    // 違反して debug build で overflow panic し、release build でも
    // wrap した誤った値で巨大確保・誤走査に進みかねない。したがって
    // 空テンソルは縮約ループへ入る前に切り離す。
    if data.is_empty() {
        return Tensor::zeros(target_shape).map_err(AutodiffError::Shape);
    }

    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            // `data` が空でないことは上記の早期 return により保証済みで、
            // `outer * axis_len * inner == data.len()`（各軸の積で全要素数
            // を再構成する構造上の不変条件）が成り立つため、`outer * inner`
            // は必ず `data.len()` 以下に収まり数学的には overflow しない。
            // それでも「本番経路 panic 禁止規約」を実装として自明に満たす
            // ため、想定外の不変条件破れに備えて `checked_mul` で明示的に
            // 検査し、破れていれば `unreachable!` ではなく型付きエラー
            // （`ShapeError::ElementCountOverflow`）を返す（codex-review
            // P1 是正・PR #2253）。
            let reduced_len = outer
                .checked_mul(inner)
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
            let mut reduced = vec![0f64; reduced_len];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        // `src`／`dst` も同じ不変条件（`outer * axis_len *
                        // inner == data.len()`）の下で構築される添字であり
                        // 数学的には `data.len()` を超えないが、同じ理由で
                        // `checked_mul`／`checked_add` により明示検査する。
                        let src = o
                            .checked_mul(axis_len)
                            .and_then(|v| v.checked_add(a))
                            .and_then(|v| v.checked_mul(inner))
                            .and_then(|v| v.checked_add(i))
                            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
                        let dst = o
                            .checked_mul(inner)
                            .and_then(|v| v.checked_add(i))
                            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
                        reduced[dst] += data[src];
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

/// `matmul`（rank 2 限定）の VJP（イシュー #2196）: `da = g @ bᵀ`・
/// `db = aᵀ @ g`。転置は `Tensor::transpose` の後に `.contiguous()` で
/// 実体化してから、forward と同じ gemm dispatch
/// （[`native_or_host_gemm`]。native → host fallback）に通す。f64 には
/// CUDA TF32 の精度低下問題がないため `matmul_fp32_strict` 相当の別経路
/// は不要（`grad.rs::matmul_vjp` との相違点）。
fn matmul_vjp_f64(
    tape: &Tape,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    g: &Tensor<f64>,
) -> Result<(Tensor<f64>, Tensor<f64>), AutodiffError> {
    let b_t = b
        .transpose(0, 1)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let a_t = a
        .transpose(0, 1)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let da = native_or_host_gemm(tape, g, &b_t)?;
    let db = native_or_host_gemm(tape, &a_t, g)?;
    Ok((da, db))
}

/// `Sum{input, dim}` の VJP（`crate::grad::unreduce_broadcast` の f64
/// 版。イシュー #2196）。`grad.rs` の「契約違反時は `debug_assert!` +
/// 無加工の `g` を返す」フォールバック方針は真似ず（モジュール doc
/// 「#2196 への申し送り」旧節参照）、失敗は型付きエラーとして伝播する。
fn unreduce_broadcast_f64(
    g: &Tensor<f64>,
    input_shape: &[usize],
    dim: Option<usize>,
) -> Result<Tensor<f64>, AutodiffError> {
    match dim {
        None => {
            let g_c = g.contiguous();
            let value = g_c.host_slice().first().copied().unwrap_or(0.0);
            Tensor::full(input_shape, value).map_err(AutodiffError::Shape)
        }
        Some(axis) => {
            let mut inserted_shape = g.shape().to_vec();
            inserted_shape.insert(axis, 1);
            let reshaped = g
                .contiguous()
                .reshape(&inserted_shape)
                .map_err(AutodiffError::Shape)?;
            let broadcasted = reshaped
                .broadcast_to(input_shape)
                .map_err(AutodiffError::Shape)?;
            Ok(broadcasted.contiguous())
        }
    }
}

/// `Mean{input, dim}` の VJP（`grad.rs::mean_vjp` の f64 版。イシュー
/// #2196）。`d(mean)/d(x_i) = 1/n` のため、上流勾配 `g` の各要素を `n`
/// （forward〈`VarF64::mean`〉と同じ縮約対象要素数）で割ってから
/// [`unreduce_broadcast_f64`]（`Sum` の VJP＝複製）へ渡す。`n == 0` は
/// forward が事前に `AutodiffError::InvalidArgument` で拒否しているため
/// backward 側へは構造的に到達しない契約——到達した場合は
/// `AutodiffError::Backward` で明示的に拒否する（`grad.rs::mean_vjp` の
/// 「0 除算を避け無加工で複製する」フォールバックは f64 版では採用しない。
/// モジュール doc 「#4.4」参照）。
fn mean_vjp_f64(
    g: &Tensor<f64>,
    input_shape: &[usize],
    dim: Option<usize>,
) -> Result<Tensor<f64>, AutodiffError> {
    let n: usize = match dim {
        None => input_shape.iter().product(),
        Some(axis) => input_shape.get(axis).copied().unwrap_or(0),
    };
    if n == 0 {
        return Err(AutodiffError::Backward(
            "VarF64 mean backward: 縮約対象の要素数が 0（forward が事前拒否 \
             しているはずの契約違反）"
                .into(),
        ));
    }
    let g_c = g.contiguous();
    let scaled: Vec<f64> = g_c.host_slice().iter().map(|&v| v / n as f64).collect();
    let scaled_t = Tensor::new(scaled, g.shape()).map_err(AutodiffError::Shape)?;
    unreduce_broadcast_f64(&scaled_t, input_shape, dim)
}

/// `Max{input, dim}` の VJP（`grad.rs::extremum_first_match_vjp` の f64
/// 版。イシュー #2196）。`out_value`（forward 記録済みの縮約後最大値）
/// と `==` 一致する `dim` 軸上の最初の要素へ上流勾配 `g` を伝播する
/// （PyTorch `torch.max(input, dim)` と同じ先勝ち決定的方式。イシュー
/// #1718 で確定した f32 側の方針をそのまま踏襲する）。`out_value`／`g`
/// の要素数が `reduce_out_shape` の想定と不一致（契約違反）の場合、
/// `grad.rs` 版は `debug_assert!` + 当該要素 0 のまま継続するが、本関数
/// は型付きエラー（`AutodiffError::Backward`）で拒否する。
fn max_first_match_vjp_f64(
    input: &Tensor<f64>,
    dim: Option<usize>,
    out_value: &Tensor<f64>,
    g: &Tensor<f64>,
) -> Result<Tensor<f64>, AutodiffError> {
    let in_shape = input.shape().to_vec();
    let in_c = input.contiguous();
    let g_c = g.contiguous();
    let out_c = out_value.contiguous();
    let in_data = in_c.host_slice();
    let g_data = g_c.host_slice();
    let out_data = out_c.host_slice();
    let mut grad = vec![0.0f64; in_data.len()];
    match dim {
        None => {
            if let (Some(&target), Some(&gv)) = (out_data.first(), g_data.first())
                && let Some(idx) = in_data.iter().position(|&v| v == target)
            {
                grad[idx] = gv;
            }
        }
        Some(axis) => {
            // `outer`／`inner` の overflow 検査は `host_sum_f64` と同じ
            // `checked_product` に委譲する（0 次元を含む shape での
            // 独立部分積 overflow を個別検査する理由も同一。
            // codex-review／Cursor Bugbot 指摘是正）。
            let outer = checked_product(&in_shape[..axis]).map_err(AutodiffError::Shape)?;
            let axis_len = in_shape[axis];
            let inner = checked_product(&in_shape[axis + 1..]).map_err(AutodiffError::Shape)?;
            // 出力要素数（`outer * inner`）が 0 なら伝播すべき出力が
            // 存在しない。`inner == 0` かつ `outer` が巨大（例
            // `usize::MAX`。axis=1 に沿った `[usize::MAX, 1, 0]` 等）な
            // shape では、この早期 return が無いと内側 `i` ループが
            // 0 回で実質何もしないにもかかわらず外側 `o` ループを
            // `outer` 回逐次走査してしまいハングする。`host_max_f64`
            // の同型早期 return と挙動を揃える（Cursor Bugbot 指摘是正。
            // イシュー #2196）。
            if outer == 0 || inner == 0 {
                return Tensor::new(grad, &in_shape).map_err(AutodiffError::Shape);
            }
            for o in 0..outer {
                for i in 0..inner {
                    let out_idx = o * inner + i;
                    let (Some(&target), Some(&gv)) = (out_data.get(out_idx), g_data.get(out_idx))
                    else {
                        return Err(AutodiffError::Backward(
                            "VarF64 max backward: out_value/g の要素数が \
                             reduce_out_shape の想定と不一致（契約違反）"
                                .into(),
                        ));
                    };
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        if let Some(&v) = in_data.get(src)
                            && v == target
                        {
                            grad[src] = gv;
                            break;
                        }
                    }
                }
            }
        }
    }
    Tensor::new(grad, &in_shape).map_err(AutodiffError::Shape)
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

    /// codex-review P1 是正の回帰テスト（PR #2253）: 空テンソル（要素数
    /// 0）だが個々の次元は `usize::MAX` を含む shape
    /// （`[0, usize::MAX, usize::MAX]`）を `reduce_to_shape_f64` へ渡し
    /// ても、`outer * inner` の通常乗算 overflow で panic せず・
    /// wrap による誤確保にも進まず、対象 shape のゼロ勾配を返すことを
    /// 確認する。`checked_numel`（`crates/tensor-core`）は
    /// `0 * usize::MAX * usize::MAX` を `Some(0)` として構築を許すため、
    /// この shape の `Tensor` 自体は正当に構築できる（本テストが再現
    /// する状況は実在する）。
    #[test]
    fn reduce_to_shape_f64_handles_empty_tensor_with_huge_dims_without_overflow() {
        let g = Tensor::<f64>::new(Vec::new(), &[0, usize::MAX, usize::MAX])
            .expect("checked_numel は 0 * MAX * MAX を Some(0) とするため構築できるはず");
        let target_shape = [0usize, 1, 1];
        let reduced =
            reduce_to_shape_f64(&g, &target_shape).expect("空テンソルはゼロ勾配へ縮約できるはず");
        assert_eq!(reduced.shape(), target_shape);
        assert_eq!(reduced.numel(), 0);
    }

    /// codex-review P1 是正の回帰テスト（PR #2255・イシュー #2196）:
    /// `n == 0`（出力が空）だが `m` 側の次元が `usize::MAX` 級の
    /// `host_gemm_f64` を、早期 return 無しでは `i` 行ループが `m` 回
    /// 逐次走査して実質ハングする形状で呼び出し、即座に空出力
    /// `[usize::MAX, 0]` を返すことを確認する（テスト自体が有限時間で
    /// 終わることが早期 return の直接的な証拠になる。CI
    /// `test-timeout-minutes: 20` がこの回帰を検出する設計）。
    #[test]
    fn host_gemm_f64_skips_huge_row_scan_when_output_is_empty() {
        let a = Tensor::<f64>::new(Vec::new(), &[usize::MAX, 0])
            .expect("checked_numel は usize::MAX * 0 を Some(0) とするため構築できるはず");
        let b = Tensor::<f64>::new(Vec::new(), &[0, 0])
            .expect("checked_numel は 0 * 0 を Some(0) とするため構築できるはず");
        let out = host_gemm_f64(&a, &b).expect("空出力の gemm は成功するはず");
        assert_eq!(out.shape(), vec![usize::MAX, 0]);
        assert_eq!(out.numel(), 0);
    }

    /// codex-review P1 是正の回帰テスト（PR #2255・イシュー #2196）:
    /// `axis_len`（縮約軸の要素数）が `usize::MAX` 級でも、出力側の
    /// `inner` が 0（`shape [usize::MAX, 0]` に `max(Some(0))`）であれば
    /// 出力要素数は 0 であり、早期 return 無しでは `o -> x -> i` の
    /// 走査順のため `x` ループが `axis_len` 回逐次走査して実質ハング
    /// する形状で `host_max_f64` を呼び出し、即座に空出力 `[0]` を
    /// 返すことを確認する（`axis_reduce_f64`〈`backend-cpu`〉が出力要素数
    /// 0 で走査自体を行わない契約と揃える）。
    #[test]
    fn host_max_f64_skips_huge_axis_scan_when_output_is_empty() {
        let a = Tensor::<f64>::new(Vec::new(), &[usize::MAX, 0])
            .expect("checked_numel は usize::MAX * 0 を Some(0) とするため構築できるはず");
        let out = host_max_f64(&a, Some(0)).expect("空出力の max は成功するはず");
        assert_eq!(out.shape(), vec![0]);
        assert_eq!(out.numel(), 0);
    }

    /// Cursor Bugbot 指摘（High Severity。PR #2255・イシュー #2196）の
    /// 回帰テスト: `max_first_match_vjp_f64` は `forward` 側
    /// （`host_max_f64`）と異なり `outer` を最外周に置く走査順
    /// （`o -> i -> a`）のため、`inner == 0` で出力要素数が 0 でも
    /// `outer` 自体が巨大（`usize::MAX` 級）だと早期 return 無しでは
    /// 外側 `o` ループを `outer` 回逐次走査して実質ハングする
    /// （`axis_len` の内側ループは `inner == 0` のため到達しないにも
    /// 関わらず、である）。`axis=1` に沿った `[usize::MAX, 1, 0]`
    /// （`outer = usize::MAX`・`axis_len = 1`・`inner = 0`）で
    /// 即座に空勾配 `[usize::MAX, 1, 0]` を返すことを確認する
    /// （テスト自体が有限時間で終わることが早期 return の直接的な
    /// 証拠になる。CI `test-timeout-minutes: 20` がこの回帰を検出する
    /// 設計）。
    #[test]
    fn max_first_match_vjp_f64_skips_huge_outer_scan_when_output_is_empty() {
        let in_shape = [usize::MAX, 1, 0usize];
        let input = Tensor::<f64>::new(Vec::new(), &in_shape)
            .expect("checked_numel は usize::MAX * 1 * 0 を Some(0) とするため構築できるはず");
        let out_shape = [usize::MAX, 0usize];
        let out_value = Tensor::<f64>::new(Vec::new(), &out_shape)
            .expect("out_value は空出力 shape で構築できるはず");
        let g =
            Tensor::<f64>::new(Vec::new(), &out_shape).expect("g は空出力 shape で構築できるはず");
        let grad = max_first_match_vjp_f64(&input, Some(1), &out_value, &g)
            .expect("空出力の max backward は成功するはず");
        assert_eq!(grad.shape(), in_shape);
        assert_eq!(grad.numel(), 0);
    }

    /// codex-review 指摘の類型調査（PR #2255・イシュー #2196）:
    /// `unreduce_broadcast_f64`（`Sum`／`Mean` の VJP）は
    /// `host_gemm_f64`／`host_max_f64` と異なり `outer`／`axis_len`／
    /// `inner` の手書き三重ループを持たず、`reshape` + `broadcast_to` +
    /// `contiguous`（`tensor-core` 側の numel 駆動実装）へ委譲する。
    /// 縮約後 `g`（`[0]`。numel 0）を巨大次元を含む `input_shape`
    /// （`[usize::MAX, 0]`。numel 0）へ broadcast する経路が、同じ
    /// ハング類型を持ち込んでいないことを回帰確認する。
    #[test]
    fn unreduce_broadcast_f64_handles_huge_input_shape_without_hang() {
        let g = Tensor::<f64>::new(Vec::new(), &[0])
            .expect("g は縮約後 shape [0]（軸長 0 のダミー）で構築できるはず");
        let input_shape = [usize::MAX, 0usize];
        let out =
            unreduce_broadcast_f64(&g, &input_shape, Some(0)).expect("空 broadcast は成功するはず");
        assert_eq!(out.shape(), input_shape);
        assert_eq!(out.numel(), 0);
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

    // --- matmul（イシュー #2196） ---

    #[test]
    fn matmul_forward_and_backward_match_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        // a = [[1,2,3],[4,5,6]]（[2,3]）、b = [[1,0],[0,1],[1,1]]（[3,2]）。
        let a = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = graph.var(&t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]));
        let c = a.matmul(&b).expect("matmul");
        assert_eq!(c.shape(), vec![2, 2]);
        assert_eq!(
            c.value().host_slice().into_owned(),
            vec![4.0, 5.0, 10.0, 11.0]
        );

        let loss = c.sum(None).expect("sum");
        let grads = graph.backward(&loss).expect("backward");
        let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();
        let db = grads.get(&b).unwrap().unwrap().host_slice().into_owned();
        // da = g(全 1) @ bᵀ、db = aᵀ @ g(全 1)。
        assert_eq!(da, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0]);
        assert_eq!(db, vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn matmul_rejects_rank_other_than_two() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]));
        let b = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert!(matches!(
            a.matmul(&b),
            Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
        ));
    }

    #[test]
    fn matmul_rejects_inner_dimension_mismatch() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        assert!(matches!(
            a.matmul(&b),
            Err(AutodiffError::Shape(ShapeError::MatmulDimMismatch { .. }))
        ));
    }

    #[test]
    fn matmul_with_zero_inner_dimension_yields_zero_output() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let a = graph.var(&t(vec![], &[2, 0]));
        let b = graph.var(&t(vec![], &[0, 3]));
        let c = a.matmul(&b).expect("k=0 は panic せずゼロ出力になるはず");
        assert_eq!(c.shape(), vec![2, 3]);
        assert_eq!(c.value().host_slice().into_owned(), vec![0.0; 6]);
    }

    // --- sum（イシュー #2196） ---

    #[test]
    fn sum_none_backward_is_ones() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let y = x.sum(None).expect("sum");
        assert_eq!(y.value().host_slice().into_owned(), vec![21.0]);
        let grads = graph.backward(&y).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![1.0; 6]);
    }

    #[test]
    fn sum_axis0_forward_and_backward_match_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let y = x.sum(Some(0)).expect("sum axis 0");
        assert_eq!(y.value().host_slice().into_owned(), vec![5.0, 7.0, 9.0]);
        let w = graph.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let z = y.mul(&w).expect("mul");
        let loss = z.sum(None).expect("sum");
        let grads = graph.backward(&loss).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn sum_axis1_forward_and_backward_match_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let y = x.sum(Some(1)).expect("sum axis 1");
        assert_eq!(y.value().host_slice().into_owned(), vec![6.0, 15.0]);
        let w = graph.var(&t(vec![10.0, 20.0], &[2]));
        let z = y.mul(&w).expect("mul");
        let loss = z.sum(None).expect("sum");
        let grads = graph.backward(&loss).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn sum_axis_out_of_range_is_rejected() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        assert!(matches!(
            x.sum(Some(5)),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
    }

    #[test]
    fn sum_of_empty_tensor_is_zero() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![], &[0]));
        let y = x.sum(None).expect("空縮約は 0.0 を返すはず");
        assert_eq!(y.value().host_slice().into_owned(), vec![0.0]);
    }

    // --- mean（イシュー #2196） ---

    #[test]
    fn mean_none_backward_matches_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![2.0, 4.0, 6.0, 8.0], &[4]));
        let y = x.mean(None).expect("mean");
        assert_eq!(y.value().host_slice().into_owned(), vec![5.0]);
        let grads = graph.backward(&y).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.25; 4]);
    }

    #[test]
    fn mean_axis0_forward_and_backward_match_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let y = x.mean(Some(0)).expect("mean axis 0");
        assert_eq!(y.value().host_slice().into_owned(), vec![2.5, 3.5, 4.5]);
        let w = graph.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let z = y.mul(&w).expect("mul");
        let loss = z.sum(None).expect("sum");
        let grads = graph.backward(&loss).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.5, 1.0, 1.5, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn mean_with_zero_elements_is_rejected() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![], &[0]));
        assert!(matches!(
            x.mean(None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    // --- max（イシュー #2196） ---

    #[test]
    fn max_none_first_match_tie_breaking() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 5.0, 3.0, 5.0], &[4]));
        let y = x.max(None).expect("max");
        assert_eq!(y.value().host_slice().into_owned(), vec![5.0]);
        let grads = graph.backward(&y).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        // タイ（idx 1・3 とも 5.0）は最初に一致した idx 1 のみへ伝播する。
        assert_eq!(dx, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn max_axis1_forward_and_backward_match_closed_form() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![1.0, 5.0, 3.0, 5.0, 2.0, 6.0], &[2, 3]));
        let y = x.max(Some(1)).expect("max axis 1");
        assert_eq!(y.value().host_slice().into_owned(), vec![5.0, 6.0]);
        let w = graph.var(&t(vec![1.0, 10.0], &[2]));
        let z = y.mul(&w).expect("mul");
        let loss = z.sum(None).expect("sum");
        let grads = graph.backward(&loss).expect("backward");
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 1.0, 0.0, 0.0, 0.0, 10.0]);
    }

    #[test]
    fn max_of_empty_tensor_is_rejected() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        let x = graph.var(&t(vec![], &[0]));
        assert!(matches!(
            x.max(None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn max_axis_with_zero_length_is_rejected() {
        let tape = new_tape();
        let graph = TapeF64::new(&tape);
        // shape [2, 0]: axis 1 の長さが 0 だが出力（軸 0 が残る）は
        // 要素数 2 > 0 のため空縮約として拒否される。
        let x = graph.var(&t(vec![], &[2, 0]));
        assert!(matches!(
            x.max(Some(1)),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}
