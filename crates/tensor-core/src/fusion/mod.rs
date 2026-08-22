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
//!   （`dispatch::select_gemm_kernel` と同方針。設計書 §3.4）。イシュー
//!   #586 で融合境界を再定義: reduction（`Sum`／`Max`）は「常に境界」
//!   ではなく、セグメント軸（`dim`）が一致する限り融合セグメントへ
//!   組み込める（`graph::FusionOp::Rsqrt` を含む elementwise 8 演算に
//!   加えて reduction もセグメント対象になる）。`Gemm`・`Input` のみが
//!   常に境界のまま（`graph.rs`・`detect.rs` の doc 参照）。イシュー
//!   #588 でさらに拡張: 縮約済みテンソルを元の行 shape へ論理拡張する
//!   `graph::FusionOp::Broadcast` を追加し、reduction と同じセグメント
//!   軸一致判定でセグメントへ組み込めるようにした（「行方向 reduction →
//!   派生スカラー → 同一行へ broadcast 適用」という RMSNorm／softmax 型
//!   の 2 パス構造を単一セグメントとして表現できる）。softmax に必須の
//!   `graph::FusionOp::Sub`／`Div` も追加し、`MAX_FUSED_CHAIN_LEN`
//!   （elementwise 連鎖長上限）は elementwise ノード数のみに適用する
//!   意味論へ精密化した（総数の暴走防止は新設の
//!   [`MAX_FUSED_SEGMENT_NODES`] が担う）。
//! - `plan`（TASK-12.1c 本体・#163）: 融合カーネル生成向け公開 DTO
//!   （[`plan::FusionPlan`]・[`plan::FusedOpKind`]・
//!   [`plan::FusedNodeIndex`]・[`plan::FusionPlanError`]・#588 で追加した
//!   [`plan::RowFusionMeta`]。1 パス／2 パス判定の閾値定数は codex-review
//!   PR #687 P2 是正で backend 非依存層から削除し、閾値判定は各
//!   バックエンドの責務とした〈`plan::RowFusionMeta` doc 参照〉）。
//!   `FusionOp`／`FusionNode`／`FusionGraph`（`graph` モジュール）は
//!   `pub(crate)` のまま変更しない設計判断（設計書 §2.5）のため、
//!   `backend-cpu`／`backend-cuda`／`backend-metal` が融合グラフの内容を
//!   読み取る唯一の経路が `plan` モジュールの公開 DTO である
//!   （設計書 §3.4「外部 backend が `run_fused` 内で融合グラフの演算
//!   内容を読み取る手段」）。
//!
//! **後続イシューとの責務分界**（設計書 §6.1 対応表・#164 実装で更新）:
//! `graph`／`detect` は融合可否の**判定**を担い、`plan` は融合対象区間の
//! **公開 DTO 化・CPU カーネル生成**（`backend-cpu::fused_elementwise`）を
//! 担う（#163）。`BackendOps::run_fused` trait メソッド追加・`autodiff`
//! 側の遅延評価統合は #164（TASK-12.1d）で実装した——`autodiff`
//! （`crates/autodiff/src/tape.rs` の `Tape::push_lazy`／
//! `materialize_fallible`／`materialize_non_fallible`）は本クレート内部の
//! `pub(crate)` 型（`graph`／`detect` モジュール）を一切経由せず、自身が
//! 保持する遅延ノード連鎖を直接 [`plan::FusedOpKind`] 列へ変換して
//! [`plan::FusionPlan::from_ops`] を呼ぶ（`tensor-core` → `autodiff` の
//! 逆依存を作れないため。設計書 §3.4「`autodiff` クレート専用の構築経路」）。
//! `graph`／`detect` モジュールの型はすべて `pub(crate)`（設計書 §2.5
//! 「配置は `tensor-core` の 1 か所に閉じる」）のまま。`plan` モジュールの
//! [`plan::FusionPlan`]・[`plan::FusedOpKind`]・[`plan::FusedNodeIndex`]・
//! [`plan::FusionPlanError`] のみ `pub`（クレートルートから re-export。
//! 設計書 §3.4 の privacy 制約）。
//!
//! `#![allow(dead_code)]`（本ファイルおよび配下の `graph`／`detect`
//! モジュールへ再帰的に適用される lint スコープ）: `graph::FusionGraph`
//! の構築 API（`push`）・`detect::detect_fusion`・`plan::FusionPlan::
//! from_segment` は、上記のとおり `autodiff` 側の統合（#164）が
//! `FusionGraph`／`detect_fusion` を経由しない構成を採ったため、本クレート
//! 内では（テスト以外では）どこからも使用されず `-D warnings`
//! （`.claude/rules/coding-rust.md`）下で dead_code 警告となる。
//! `crates/backend-cuda/src/kernels_wmma_opt.rs:116` 以降が採る既存
//! プラクティス（結線待ちコードへの理由付き `#[allow(dead_code)]`）と
//! 同型であり、`tensor-core` 内で `FusionGraph` が既に存在する場合の
//! 構築経路（`from_segment`）を実際に呼ぶ将来の利用者が現れるまで残す
//! 設計判断とする。`plan` モジュールの公開 DTO・`FusionPlan::from_ops` は
//! `backend-cpu`（`run_fused` 経由）・`autodiff`（`tape.rs`）から実際に
//! 使用されるため dead_code 対象外。`detect::MAX_FUSED_CHAIN_LEN` も
//! `fandhe_ai_autodiff::tape` の push 時上限適用（#404）が参照する単一真実源と
//! なったため dead_code 対象外（`detect_fusion` 自体は未結線のまま
//! dead_code スコープに残る）。

#![allow(dead_code)]
// `FusionGraph`／`detect_fusion`／`FusionPlan::from_segment` は上記のとおり
// `autodiff` 統合（#164）が経由しない構成のため未使用のまま残る
// （同じ撤去条件）。
#![allow(unused_imports)]

mod detect;
mod graph;
mod plan;

pub(crate) use detect::{
    FallbackReason, FusionDecision, FusionSegment, MIN_FUSED_CHAIN_LEN, detect_fusion,
};
// `fandhe_ai_autodiff::tape` の push 時上限適用（#404）が参照する単一真実源
// として `pub` 昇格（`detect.rs` の doc comment 参照）。クレートルート
// （`lib.rs`）で再 re-export する。`MAX_FUSED_SEGMENT_NODES`（#588）は
// 現状クレート内利用者を持たないが、`MAX_FUSED_CHAIN_LEN` と対の
// 実装判断の定数として同じ可視性（`pub`）で揃える。
pub use detect::{MAX_FUSED_CHAIN_LEN, MAX_FUSED_SEGMENT_NODES};
pub(crate) use graph::{
    FusionGraph, FusionGraphError, FusionNode, FusionNodeId, FusionOp, NodeMeta,
};
pub use plan::{FusedNodeIndex, FusedOpKind, FusionPlan, FusionPlanError, RowFusionMeta};
