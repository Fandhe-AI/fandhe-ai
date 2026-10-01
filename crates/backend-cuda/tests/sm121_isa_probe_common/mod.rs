//! sm_121 の使える命令とアーキ固有機能のプローブ基盤（イシュー #2122・PR-A）。
//!
//! 3 つのテストバイナリ（`sm121_isa_probe_registry`・`sm121_isa_probe_compile`・
//! `sm121_isa_probe_exec_real_device`）が
//! `#[path = "sm121_isa_probe_common/mod.rs"] mod common;` で取り込む
//! （`tests/setmaxnreg_common/mod.rs` と同じ流儀。本ディレクトリ自体は独立の
//! cargo test バイナリにならない）。
//!
//! 判定規則の正は `docs/perf/logs/sm121-isa-probe-2122/RULE.txt`、設計と使い方は
//! `docs/cuda-sm121-isa-probe.md`。レジストリ（`registry.rs`）・RULE.txt・
//! `aggregate.py` の 3 か所で条項 ID・プローブ ID を突き合わせる。
//!
//! # `#![allow(dead_code)]` の理由
//!
//! 取り込む 3 バイナリはそれぞれ本モジュールの一部しか使わない（registry テストは
//! GPU 経路の `runner` を、compile テストは `run_pipeline` の S3 以降を使わない）。
//! バイナリごとの未使用側が `dead_code` の対象になるが、共有ロジックを分割すると
//! 二重管理になる（`setmaxnreg_common/mod.rs` と同じトレードオフ。
//! `.claude/rules/code-comment-style.md`）。
//!
//! # A03（インジェクション）対応
//!
//! カーネルソースは `&'static str` のコンパイル時定数のみ。アーキ名は
//! `types.rs` のホワイトリストからのみ取り、環境変数・外部入力をソースや
//! NVRTC オプションへ連結しない。

#![allow(dead_code)]

pub mod jsonl;
pub mod kernels_arch;
pub mod kernels_cluster;
pub mod kernels_mma;
pub mod model;
pub mod registry;
pub mod runner;
pub mod types;
