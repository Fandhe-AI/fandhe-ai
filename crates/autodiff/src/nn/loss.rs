//! 損失関数（親イシュー #189）のうち CrossEntropy を担当する
//! （#191。兄弟イシュー #190 は MSE の reduction 対応・`nn` ラッパー
//! 側が主スコープと推測されるが、着手時点で `origin/main` に
//! `Reduction` 型は未定義だったため本イシューで定義する）。
//!
//! `nn::activation`（`nn/activation.rs`）と同じく、各構造体は
//! `Tape`/`Var`（`crate::tape`/`crate::var`）の内部契約を直接扱う
//! `Var::cross_entropy_loss`（`crate::var`）の薄いラッパーに徹する
//! （REQ-9「互換 API 層は自作コアの上の薄いラッパーに徹する」の
//! 精神を `nn` モジュールにも適用。`nn/mod.rs` の境界説明参照）。
//!
//! **Softmax 単体の公開活性化は本イシューのスコープ外**のまま維持する
//! （`nn/activation.rs` の「Softmax は CE と密結合のため対象外」判断を
//! 踏襲。CE は融合オペ〈`tape::Op::CrossEntropyLoss`〉として実装する
//! ため、独立した Softmax プリミティブは不要）。

use tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::var::Var;

/// 損失の集約方式。`CrossEntropyLoss`（本モジュール）・MSE（#190）で
/// 共有されうる値のため `nn::loss` 直下に置く（実装計画 §3.2「#190 が
/// 先行して `Reduction` を定義済みなら再利用、未定義なら本イシューで
/// 定義する」の判断に従う。着手時点で他の `Reduction` 定義は
/// `crates/autodiff/src` に存在しなかった）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// 全サンプルの平均（PyTorch `F.cross_entropy` の既定）。
    Mean,
    /// 全サンプルの総和。
    Sum,
}

/// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191）。
/// `Var::cross_entropy_loss`（`crate::var`）の薄いラッパー。
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    /// クラス次元（PyTorch の `[N, C, d1..]` 形状は `class_dim = 1` に
    /// 相当する）。
    pub class_dim: usize,
    pub reduction: Reduction,
}

impl CrossEntropyLoss {
    /// `logits`（予測値・追跡対象）と `targets`（正解クラス添字・
    /// 非追跡）から損失を計算する。検査・数値安定化の実体は
    /// `Var::cross_entropy_loss` 側にあり、ここでは呼び出すだけ
    /// （「薄いラッパー性」は `tests/nn_cross_entropy.rs` で検証する）。
    pub fn forward<'t>(
        &self,
        logits: &Var<'t>,
        targets: &Tensor<i32>,
    ) -> Result<Var<'t>, AutodiffError> {
        logits.cross_entropy_loss(targets, self.class_dim, self.reduction)
    }
}
