//! elementwise カーネル融合機構（REQ-12・TASK-12.1）の中間表現・連鎖検出。
//!
//! REQ-12（カーネル融合の自動活用・Should）を、Burn／CubeCL に依存しない
//! 自作 elementwise 融合機構として読み替えたもの（TASK-12.1、
//! `docs/spec/05-tasks.md:370`）。設計正本は
//! `docs/fusion-graph-design.md`（TASK-12.1a・#161）であり、本モジュールは
//! 同文書 §2（グラフ表現）・§3.2（実体化条件）を実装する。
//!
//! - `graph`（TASK-12.1a 設計の実装。#162）: 融合グラフの中間表現
//!   （[`graph::FusionOp`]／[`graph::FusionNodeId`]／[`graph::NodeMeta`]／
//!   [`graph::FusionNode`]／[`graph::FusionGraph`]）。検証を先行させる
//!   構築 API（`FusionGraph::push`）を提供する（OWASP A03 観点。
//!   `.claude/rules/security.md`）。
//! - `detect`（TASK-12.1b・#162 本体）: elementwise 連鎖検出（融合判定）
//!   アルゴリズム（[`detect::detect_fusion`]・[`detect::FusionDecision`]・
//!   [`detect::FusionSegment`]）。副作用なしの純関数として実装する
//!   （`dispatch::select_gemm_kernel` と同方針。設計書 §3.4）。
//! - `plan`（TASK-12.1c 本体・#163）: 融合カーネル生成向け公開 DTO
//!   （[`plan::FusionPlan`]・[`plan::FusedOpKind`]・
//!   [`plan::FusedNodeIndex`]・[`plan::FusionPlanError`]）。
//!   `FusionOp`／`FusionNode`／`FusionGraph`（`graph` モジュール）は
//!   `pub(crate)` のまま変更しない設計判断（設計書 §2.5）のため、
//!   `backend-cpu`／`backend-cuda`／`backend-metal` が融合グラフの内容を
//!   読み取る唯一の経路が `plan` モジュールの公開 DTO である
//!   （設計書 §3.4「外部 backend が `run_fused` 内で融合グラフの演算
//!   内容を読み取る手段」）。
//!
//! **後続イシューとの責務分界**（設計書 §6.1 対応表・#163 実装で更新）:
//! 本モジュールは融合可否の**判定**（`detect`）と融合対象区間の**公開
//! DTO 化・CPU カーネル生成**（`plan`・`backend-cpu::fused_elementwise`）
//! までを担う。`FusionSession`／`FusionValue`／`BackendOps::run_fused`
//! trait メソッド追加・`autodiff` 側の遅延評価統合（`Tape::new(ops)` へ
//! の結線）は #164（TASK-12.1d）が担当し、本イシュー（#163）のスコープ
//! 外である。`graph`／`detect` モジュールの型はすべて `pub(crate)`
//! （設計書 §2.5「配置は `tensor-core` の 1 か所に閉じる」）のまま。
//! `plan` モジュールの [`plan::FusionPlan`]・[`plan::FusedOpKind`]・
//! [`plan::FusedNodeIndex`]・[`plan::FusionPlanError`] のみ `pub`
//! （クレートルートから re-export。設計書 §3.4 の privacy 制約）。
//!
//! `#![allow(dead_code)]`（本ファイルおよび配下の `graph`／`detect`
//! モジュールへ再帰的に適用される lint スコープ）: `graph::FusionGraph`
//! の構築 API（`push`）・`detect::detect_fusion` は、#164 が
//! `FusionSession`／`autodiff` 側の遅延評価統合で実際の融合実行経路へ
//! 結線するまでの間、本クレート内のどこからも（テスト以外では）使用
//! されず `-D warnings`（`.claude/rules/coding-rust.md`）下で dead_code
//! 警告となる。`crates/backend-cuda/src/kernels_wmma_opt.rs:116` 以降が
//! 採る既存プラクティス（結線待ちコードへの理由付き
//! `#[allow(dead_code)]`）と同型であり、結線完了（#164 のマージ）時に
//! 撤去する。`plan` モジュールの公開 DTO・`FusionPlan::from_ops` は
//! `backend-cpu` から実際に使用されるため dead_code 対象外だが、
//! `FusionPlan::from_segment`（`pub(crate)`）は #164 の `FusionSession`
//! 結線までの間、本クレート内では `plan.rs` 自身の `#[cfg(test)]`
//! からのみ使用されるため、引き続き本 allow の対象に含める。

#![allow(dead_code)]
// re-export 自体も #164 が FusionSession を結線するまで
// `FusionGraph`／`detect_fusion` 側は未使用（上記と同じ撤去条件）。
#![allow(unused_imports)]

mod detect;
mod graph;
mod plan;

pub(crate) use detect::{
    FallbackReason, FusionDecision, FusionSegment, MAX_FUSED_CHAIN_LEN, MIN_FUSED_CHAIN_LEN,
    detect_fusion,
};
pub(crate) use graph::{
    FusionGraph, FusionGraphError, FusionNode, FusionNodeId, FusionOp, NodeMeta,
};
pub use plan::{FusedNodeIndex, FusedOpKind, FusionPlan, FusionPlanError};
