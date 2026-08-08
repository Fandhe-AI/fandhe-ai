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
//! **後続イシューとの責務分界**（設計書 §6.1 対応表）: 本モジュールは
//! 融合可否の**判定**までを担う。融合カーネル生成・`FusionPlan` 公開
//! DTO アクセサは #163（TASK-12.1c）、`FusionSession`／`FusionValue`／
//! `BackendOps::run_fused`・`autodiff` 側の遅延評価統合は #164
//! （TASK-12.1d）が担当し、いずれも本イシュー（#162）のスコープ外
//! である。本モジュールの型はすべて `pub(crate)`（設計書 §2.5「配置は
//! `tensor-core` の 1 か所に閉じる」）。
//!
//! `#![allow(dead_code)]`（本ファイルおよび配下の `graph`／`detect`
//! モジュールへ再帰的に適用される lint スコープ）: #163／#164 が
//! `FusionGraph`／`detect_fusion` を実際の融合実行経路へ結線するまでの
//! 間、本モジュールの型・関数はクレート内のどこからも使用されず
//! `-D warnings`（`.claude/rules/coding-rust.md`）下で dead_code 警告と
//! なる。`crates/backend-cuda/src/kernels_wmma_opt.rs:116` 以降が採る
//! 既存プラクティス（結線待ちコードへの理由付き `#[allow(dead_code)]`）
//! と同型であり、結線完了（#163／#164 のマージ）時に撤去する。

#![allow(dead_code)]
// re-export 自体も #163/#164 が結線するまで未使用（上記と同じ撤去条件）。
#![allow(unused_imports)]

mod detect;
mod graph;

pub(crate) use detect::{
    FallbackReason, FusionDecision, FusionSegment, MAX_FUSED_CHAIN_LEN, MIN_FUSED_CHAIN_LEN,
    detect_fusion,
};
pub(crate) use graph::{
    FusionGraph, FusionGraphError, FusionNode, FusionNodeId, FusionOp, NodeMeta,
};
