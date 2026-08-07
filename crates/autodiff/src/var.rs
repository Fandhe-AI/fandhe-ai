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
use crate::nn::loss::Reduction;
use crate::tape::{NodeId, Op, Tape};

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
    /// **借用注意**: この `Ref` を保持したまま、同じ `Tape` に対して
    /// `borrow_mut()` を要する演算（`matmul`/`add` 等のノード追加）を
    /// 呼ぶと `RefCell` の二重可変借用で実行時 panic になる。値をその場
    /// の参照ではなく所有値として持ち出したい場合は `to_tensor()` を
    /// 使うこと（`docs/public-api-design.md` §3.1）。
    pub fn value(&self) -> Ref<'_, Tensor<f32>> {
        Ref::map(self.tape.nodes.borrow(), |nodes| &nodes[self.id.0].value)
    }

    /// `value()` の所有値版。`Tensor<f32>` へ複製して返すため `Ref` を
    /// 持ち越さず、直後に同じ `Tape` へノード追加演算を呼んでも借用
    /// エラー・panic が起きない。
    pub fn to_tensor(&self) -> Tensor<f32> {
        self.tape.nodes.borrow()[self.id.0].value.clone()
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
    pub fn matmul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.value().shape().to_vec();
        let rhs_shape = other.value().shape().to_vec();
        matmul_out_shape(&lhs_shape, &rhs_shape)?;
        let value = eval::matmul(&self.value(), &other.value());
        let id = self.tape.push(Op::MatMul(self.id, other.id), value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// bias broadcast を含む要素ごとの加算（`docs/public-api-design.md` §3.2）。
    pub fn add(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.value().shape().to_vec();
        let rhs_shape = other.value().shape().to_vec();
        broadcast_shape(&lhs_shape, &rhs_shape)?;
        let value = eval::add(&self.value(), &other.value());
        let id = self.tape.push(Op::Add(self.id, other.id), value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// ブロードキャスト付き要素ごとの乗算。
    pub fn mul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.value().shape().to_vec();
        let rhs_shape = other.value().shape().to_vec();
        broadcast_shape(&lhs_shape, &rhs_shape)?;
        let value = eval::mul(&self.value(), &other.value());
        let id = self.tape.push(Op::Mul(self.id, other.id), value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約和。`dim: None` は全軸縮約（スカラー）。
    pub fn sum(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.value().shape().to_vec();
        let out_shape = reduce_out_shape(&shape, dim)?;
        let value = eval::sum(&self.value(), dim, &out_shape);
        let id = self.tape.push(
            Op::Sum {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約最大値。`dim: None` は全軸縮約（スカラー）。
    pub fn max(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.value().shape().to_vec();
        let out_shape = reduce_out_shape(&shape, dim)?;
        let value = eval::max(&self.value(), dim, &out_shape);
        let id = self.tape.push(
            Op::Max {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値）。
    pub fn mse_loss(&self, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.value().shape().to_vec();
        let rhs_shape = target.value().shape().to_vec();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        let value = eval::mse_loss(&self.value(), &target.value());
        let id = self.tape.push(Op::MseLoss(self.id, target.id), value);
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
    /// `AutodiffError::InvalidArgument`）→ ③forward 値計算
    /// （`eval::cross_entropy_loss`）→ ④ノード記録。
    pub fn cross_entropy_loss(
        &self,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        let logits_shape = self.value().shape().to_vec();
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

        let value = eval::cross_entropy_loss(&self.value(), targets, class_dim, reduction);
        let id = self.tape.push(
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
    /// （`docs/public-api-design.md` §3.2）。
    pub fn relu(&self) -> Var<'t> {
        let value = eval::relu(&self.value());
        let id = self.tape.push(Op::Relu(self.id), value);
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの指数関数。
    pub fn exp(&self) -> Var<'t> {
        let value = eval::exp(&self.value());
        let id = self.tape.push(Op::Exp(self.id), value);
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの双曲線正接。
    pub fn tanh(&self) -> Var<'t> {
        let value = eval::tanh(&self.value());
        let id = self.tape.push(Op::Tanh(self.id), value);
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとのシグモイド（`1 / (1 + exp(-x))`）。`relu`/`exp`/`tanh`
    /// と同じく shape 不変の単項演算のため構造的に失敗しえない
    /// （TASK-9.1b・#92。`nn::activation::Sigmoid` の薄いラッパーが
    /// このメソッドを呼ぶ）。forward は `eval::sigmoid`（数値安定形）
    /// を使う。
    pub fn sigmoid(&self) -> Var<'t> {
        let value = eval::sigmoid(&self.value());
        let id = self.tape.push(Op::Sigmoid(self.id), value);
        Var::from_raw(self.tape, id)
    }
}
