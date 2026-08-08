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
//!
//! **後続イシューとの責務分界**（設計書 §6.1 対応表）: `graph`／`detect`
//! （#162）は融合可否の**判定**までを担う。`plan`（本イシュー・#164）は
//! `FusionPlan` 公開 DTO・`FusionSession`／`FusionValue`・
//! `BackendOps::run_fused` デフォルト実装を提供し、`autodiff` 側の
//! 遅延評価統合（`crates/autodiff/src/tape.rs` の `materialize_fallible`／
//! `materialize_non_fallible`）から `FusionPlan::from_ops` 経由で使われる。
//! 融合カーネル生成本体（CPU 融合実行器・`backend-cpu` 側の `run_fused`
//! オーバーライド）は #163（TASK-12.1c）のスコープであり、本イシュー
//! 時点では #163 が未マージのため `run_fused` はデフォルト実装
//! （`BackendError::Unsupported`）のまま結線される（`docs/
//! fusion-graph-design.md` §3.4）。本モジュールの型はすべて `pub(crate)`
//! （設計書 §2.5「配置は `tensor-core` の 1 か所に閉じる」）だが、
//! `FusionPlan`／`FusedOpKind`／`FusedNodeIndex`（`plan.rs`）のみ `pub`
//! （`BackendOps::run_fused` のシグネチャに現れる公開 DTO。privacy
//! 制約。`plan.rs` 冒頭コメント参照）。
//!
//! `#![allow(dead_code)]`（`detect`／`FusionSession` に限り残存。下記）:
//! `detect_fusion`（#162 の連鎖検出アルゴリズム）・`FusionSession`
//! （`tensor-core` 内で `FusionGraph` が既に存在する場合のための内部
//! 機構）はいずれも #163（融合カーネル生成本体・実際の連鎖検出結線）が
//! マージされるまでクレート内のどこからも使用されない
//! （`autodiff` は `detect_fusion`／`FusionSession` を経由せず、自身の
//! `TapeNode`／`Op` 遅延連鎖を直接 `FusedOpKind` へ変換して
//! `FusionPlan::from_ops` を呼ぶ。設計書 §3.4「`autodiff::Tape` の
//! 実体化はこの `FusionSession` を経由しない」）。
//! `crates/backend-cuda/src/kernels_wmma_opt.rs:116` 以降が採る既存
//! プラクティス（結線待ちコードへの理由付き `#[allow(dead_code)]`）と
//! 同型であり、結線完了（#163 のマージ）時に撤去する。
//!
//! `#![allow(unused_imports)]`: `plan`／`graph` サブモジュールは互いに
//! `super::graph::X` の形で直接参照する（本 `mod.rs` の再エクスポートを
//! 経由しない）ため、本ファイルの `pub(crate) use` 群は #163 が
//! `detect_fusion`／`FusionSession` を実際に結線するまでの間、クレート内
//! のどこからも `crate::fusion::X` の形では参照されない。将来の結線時に
//! 参照経路が生まれた際にすぐ使える形（型の一覧性）を保つため、
//! 撤去はせず本 allow で抑制する（上記 `#[allow(dead_code)]` と同じ
//! 撤去条件）。

#![allow(dead_code)]
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
pub use plan::{FusedNodeIndex, FusedOpKind, FusionPlan};
pub(crate) use plan::{FusionSession, FusionValue};
