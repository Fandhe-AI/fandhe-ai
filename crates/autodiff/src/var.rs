//! テープ上の 1 ノードを指す追跡対象値 `Var` と、その forward 演算群。
//!
//! `tensor_core::Tensor<f32>` に対する演算は一切テープを構築しない
//! （非追跡）。`Var` に対する演算のみが `Tape::push`（`tape.rs`）を
//! 経由してテープへ記録される。この「型分離」により、勾配追跡の
//! ON/OFF がコンパイル時に保証される（`docs/public-api-design.md`
//! §3.1「型分離方式」）。
//!
//! 各演算メソッドは
//! 「①クロステープ検査 → ②shape 検査 → ③forward 値計算（`eval.rs`）
//! → ④ノード記録（`Tape::push`）」の順で処理する。値計算の借用
//! （`Ref`）はスコープを閉じてから `push`（`borrow_mut`）を呼ぶ
//! （`RefCell` の二重可変借用 panic を避けるための実装規律。
//! `.claude/rules/coding-rust.md` の本番経路 panic 禁止方針）。

use std::cell::Ref;

use tensor_core::{
    Tensor, broadcast_shape, matmul_out_shape, reduce_out_shape, require_same_shape,
};

use crate::error::AutodiffError;
use crate::eval;
use crate::tape::{NodeId, Op, Tape, materialize_fallible, materialize_non_fallible};

/// `Var::mse_loss_with` の縮約種別（#190・TASK-9.1c 相当。親イシュー
/// #189「損失関数（MSE・CrossEntropy）の実装」）。PyTorch
/// `nn.MSELoss(reduction=...)` の `mean`/`sum` に対応する。
///
/// `#[non_exhaustive]` とする理由: 将来 `none`（要素ごと損失。PyTorch
/// `reduction='none'` 相当）を追加しうるが、本イシューでは #190 実装
/// 計画のスコープ外（out-of-scope-tracking.md 準拠でユーザー承認後に
/// 別途追加）としたため、追加時に呼び出し側の非網羅的 `match` を破壊
/// しないようにする。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Reduction {
    /// 全要素平均（`Σ(pred−target)² / n`）。`Var::mse_loss` の既定
    /// （PyTorch `nn.MSELoss` の既定 `reduction='mean'` と一致）。
    Mean,
    /// 全要素総和（`Σ(pred−target)²`）。
    Sum,
}

/// テープ上の 1 ノードを指す追跡対象値。値そのものではなく `NodeId` +
/// テープへの共有参照を保持し、演算のたびにテープへ新しいノードを
/// 追加する（`docs/public-api-design.md` §3.1）。
///
/// **クロステープ安全性**: ライフタイム `'t` の一致は同一 `Tape` を
/// 指す証明にはならない（同一スコープに複数の `Tape` が存在する場合、
/// それぞれの `Var<'t>` は同一の `'t` を持ちうる）。そのため二項演算
/// （`matmul`/`add`/`mul`/`mse_loss`）は入口で `self.tape.id` と相手側
/// `Var` が保持する `TapeId` の一致を実行時検査し、不一致なら
/// `AutodiffError::TapeMismatch` を返す。
#[derive(Debug, Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    id: NodeId,
}

impl<'t> Var<'t> {
    /// `Tape::var()` からのみ呼ばれる内部コンストラクタ。
    pub(crate) fn from_raw(tape: &'t Tape, id: NodeId) -> Var<'t> {
        Var { tape, id }
    }

    /// 追跡を外し、現在の値を非追跡の `Tensor<f32>` の借用として取り出す。
    ///
    /// **TASK-12.1d（#164）**: 対象ノードが未実体化（elementwise の遅延
    /// グラフの一部）であれば [`materialize_non_fallible`]（層 2。融合を
    /// 試み、失敗すれば `ops` の per-op メソッド → `eval.rs` の順に必ず
    /// 値を返す）経由で実体化する。`matmul`/`sum`/`max`・`Tape::backward`
    /// が使う層 1（[`crate::tape::materialize_fallible`]）とは異なる
    /// エラー処理契約を持つ（`docs/fusion-graph-design.md` §3.5.3）。
    ///
    /// **借用注意**: この `Ref` を保持したまま、同じ `Tape` に対して
    /// `borrow_mut()` を要する演算（`matmul`/`add` 等のノード追加）を
    /// 呼ぶと `RefCell` の二重可変借用で実行時 panic になる。値をその場
    /// の参照ではなく所有値として持ち出したい場合は `to_tensor()` を
    /// 使うこと（`docs/public-api-design.md` §3.1）。
    pub fn value(&self) -> Ref<'_, Tensor<f32>> {
        Ref::map(self.tape.nodes.borrow(), |nodes| {
            materialize_non_fallible(nodes, self.tape.ops(), self.id)
        })
    }

    /// `value()` の所有値版。`Tensor<f32>` へ複製して返すため `Ref` を
    /// 持ち越さず、直後に同じ `Tape` へノード追加演算を呼んでも借用
    /// エラー・panic が起きない。
    pub fn to_tensor(&self) -> Tensor<f32> {
        let nodes = self.tape.nodes.borrow();
        materialize_non_fallible(&nodes, self.tape.ops(), self.id).clone()
    }

    /// 実体化なしに読める構造的な出力 shape（`TapeNode.shape`。
    /// TASK-12.1d・#164）。演算入口の shape 検査は本メソッドを使い、
    /// `value()`/`materialize_fallible` を呼ばない（`docs/
    /// fusion-graph-design.md` §3.5.1「shape 検証と実行を分離する」）。
    fn shape(&self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.id.0].shape.clone()
    }

    /// 演算入口で必ず shape 検査より前に呼ぶクロステープ検査
    /// （`docs/public-api-design.md` §3.1「クロステープ安全性」）。
    fn check_same_tape(&self, other: &Var<'t>) -> Result<(), AutodiffError> {
        if self.tape.id != other.tape.id {
            return Err(AutodiffError::TapeMismatch);
        }
        Ok(())
    }

    /// この `Var` が属する `Tape` の識別子。`backward.rs`（TASK-1.5c・
    /// #18）は別モジュールのため `tape` フィールド（private）へ直接
    /// 触れられず、`Tape::backward`/`Gradients::get` のクロステープ検査
    /// （`check_same_tape` と同じ「入口で必ず shape・NodeId 解決より前に
    /// 検査する」契約）にこのアクセサを使う。
    pub(crate) fn tape_id(&self) -> crate::tape::TapeId {
        self.tape.id
    }

    /// この `Var` が指すテープ内ノードの識別子。`backward.rs` が
    /// `Gradients` から当該ノードの勾配を引くための添字として使う
    /// （`tape_id()` と同じくクレート内限定公開）。
    pub(crate) fn node_id(&self) -> NodeId {
        self.id
    }

    /// 2 次元 `matmul`（`docs/public-api-design.md` §3.2）。
    ///
    /// **TASK-12.1d（#164）**: 非 elementwise のため常に実体化済みで
    /// 返る（`push_eager`）。実行は `eval.rs` 直接呼び出しから
    /// `self.tape.ops().gemm`（`BackendOps` 経由）へ置き換えた
    /// （TASK-1.9「backend 経由実行への置き換え」・設計書 §3.5.2）。
    /// 入力が elementwise の遅延グラフであった場合は
    /// [`materialize_fallible`]（層 1）で自身の実行の一部として実体化
    /// する。
    pub fn matmul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        matmul_out_shape(&lhs_shape, &rhs_shape)?;
        let (lhs_val, rhs_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), other.id)?.clone();
            (lhs_val, rhs_val)
        };
        let value = self.tape.ops().gemm(&lhs_val, &rhs_val)?;
        let id = self.tape.push_eager(Op::MatMul(self.id, other.id), value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// bias broadcast を含む要素ごとの加算（`docs/public-api-design.md`
    /// §3.2）。
    ///
    /// **TASK-12.1d（#164）**: elementwise 5 演算の 1 つ。shape 検証
    /// （①クロステープ検査・②shape 検査）のみ即時実行し、値計算
    /// （③）は実体化境界まで遅延させる（`push_lazy`。`Ok` を返すことは
    /// 「shape が妥当でノードが記録された」ことのみを意味し「加算が
    /// 計算済み」であることを意味しない。設計書 §3.5.1）。
    pub fn add(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        let id = self.tape.push_lazy(Op::Add(self.id, other.id), out_shape);
        Ok(Var::from_raw(self.tape, id))
    }

    /// ブロードキャスト付き要素ごとの乗算。elementwise 5 演算の 1 つ
    /// （`add` と同じ遅延契約。TASK-12.1d・#164）。
    pub fn mul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        let id = self.tape.push_lazy(Op::Mul(self.id, other.id), out_shape);
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約和。`dim: None` は全軸縮約（スカラー）。
    /// 非 elementwise のため常に実体化済みで返る（`matmul` と同じ
    /// TASK-12.1d の置き換え方針。実行は `self.tape.ops().sum` 経由）。
    pub fn sum(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = self.tape.ops().sum(&input_val, dim)?;
        let id = self.tape.push_eager(
            Op::Sum {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約最大値。`dim: None` は全軸縮約（スカラー）。
    /// `sum` と同じ置き換え方針（`self.tape.ops().max` 経由）。
    pub fn max(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = self.tape.ops().max(&input_val, dim)?;
        let id = self.tape.push_eager(
            Op::Max {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値。全要素平均・
    /// PyTorch `nn.MSELoss` の既定 `reduction='mean'` 相当）。
    /// `mse_loss_with(target, Reduction::Mean)` への委譲（#190）。
    /// 既存呼び出し元（`nn::activation` 系テスト・`tests/backward.rs`
    /// 等）のシグネチャ・意味を変えないため本メソッドは維持する。
    pub fn mse_loss(&self, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.mse_loss_with(target, Reduction::Mean)
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値）。`reduction`
    /// で mean/sum の縮約種別を選べる（#190。親イシュー #189「損失関数
    /// （MSE・CrossEntropy）の実装」）。`nn::loss::MseLoss`（`nn/loss.rs`）
    /// はこのメソッドを呼ぶだけの薄いラッパー（REQ-9）。
    ///
    /// **TASK-12.1d（#164）**: `BackendOps` に対応メソッドがないため
    /// 融合対象外とし、入力を層 1（[`materialize_fallible`]）で実体化
    /// したうえで従来どおり `eval::mse_loss` を直接呼ぶ（設計書 §3.5.1
    /// 「`sigmoid`/`mse_loss`/`cross_entropy_loss` は従来どおり `eval.rs`
    /// で即時計算し `OnceCell` を充填して返す」）。
    pub fn mse_loss_with(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.shape();
        let rhs_shape = target.shape();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        let (pred_val, target_val) = {
            let nodes = self.tape.nodes.borrow();
            let pred_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let target_val = materialize_fallible(&nodes, self.tape.ops(), target.id)?.clone();
            (pred_val, target_val)
        };
        let value = eval::mse_loss(&pred_val, &target_val, reduction);
        let id = self.tape.push_eager(
            Op::MseLoss {
                pred: self.id,
                target: target.id,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191・
    /// 親イシュー #189）。`self` = logits（追跡対象）、`targets` = 正解
    /// クラス添字（非追跡・`Tensor<i32>`。勾配は定義されないため
    /// `Var` にしない。`tape::Op::CrossEntropyLoss` doc 参照）。
    ///
    /// 検査順序（本メソッド冒頭 doc の演算メソッド規律に、targets
    /// 範囲検査〈REQ-8 趣旨の境界外アクセス防止・A03 対策〉を追加）:
    /// ①`class_dim` 範囲・targets shape 一致（`reduce_out_shape` を
    /// 再利用。`class_dim >= rank` は `ShapeError::AxisOutOfRange`）
    /// → ②targets 全添字が `0 <= t < C`（違反は
    /// `AutodiffError::InvalidArgument`）→ ③実体化（層 1）→ ④forward
    /// 値計算（`eval::cross_entropy_loss`。`mse_loss_with` と同じく
    /// `BackendOps` に対応メソッドがないため融合対象外）→ ⑤ノード記録。
    pub fn cross_entropy_loss(
        &self,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        let logits_shape = self.shape();
        let expected_targets_shape = reduce_out_shape(&logits_shape, Some(class_dim))?;
        require_same_shape(targets.shape(), &expected_targets_shape)?;

        // `reduce_out_shape` が成功した時点で `class_dim < logits_shape.len()`
        // が保証されるため、この添字アクセスは安全（`.claude/rules/
        // coding-rust.md` REQ-8「境界検査を省略しない」の趣旨に沿い、
        // 検査済みの添字のみでアクセスする）。
        let num_classes = logits_shape[class_dim];
        for t in eval::dense_vec_i32(targets) {
            if t < 0 || (t as usize) >= num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "cross_entropy_loss: target 添字 {t} が範囲 [0, {num_classes}) を外れている"
                )));
            }
        }

        let logits_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = eval::cross_entropy_loss(&logits_val, targets, class_dim, reduction);
        let id = self.tape.push_eager(
            Op::CrossEntropyLoss {
                logits: self.id,
                targets: targets.clone(),
                class_dim,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// ReLU。shape を変えない要素ごとの演算のため構造的に失敗しえない
    /// （`docs/public-api-design.md` §3.2）。elementwise 5 演算の 1 つ
    /// （`add`/`mul` と同じ遅延契約。TASK-12.1d・#164）。
    pub fn relu(&self) -> Var<'t> {
        let shape = self.shape();
        let id = self.tape.push_lazy(Op::Relu(self.id), shape);
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの指数関数。elementwise 5 演算の 1 つ（遅延契約）。
    pub fn exp(&self) -> Var<'t> {
        let shape = self.shape();
        let id = self.tape.push_lazy(Op::Exp(self.id), shape);
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの双曲線正接。elementwise 5 演算の 1 つ（遅延契約）。
    pub fn tanh(&self) -> Var<'t> {
        let shape = self.shape();
        let id = self.tape.push_lazy(Op::Tanh(self.id), shape);
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとのシグモイド（`1 / (1 + exp(-x))`）。`relu`/`exp`/`tanh`
    /// と同じく shape 不変の単項演算のため構造的に失敗しえない
    /// （TASK-9.1b・#92。`nn::activation::Sigmoid` の薄いラッパーが
    /// このメソッドを呼ぶ）。forward は `eval::sigmoid`（数値安定形）
    /// を使う。
    ///
    /// **TASK-12.1d（#164）**: `BackendOps` に対応メソッドがないため
    /// 融合対象外とし、常に実体化済みで返る（`push_eager`）。入力読み
    /// 出しは非 fallible な本メソッド自身の契約に合わせ `value()`
    /// （層 2）経由とする（設計書 §3.5.1）。
    pub fn sigmoid(&self) -> Var<'t> {
        let value = eval::sigmoid(&self.value());
        let id = self.tape.push_eager(Op::Sigmoid(self.id), value);
        Var::from_raw(self.tape, id)
    }
}
