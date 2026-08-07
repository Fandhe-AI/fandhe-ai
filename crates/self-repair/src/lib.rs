//! 自己修復ループの種別非依存な共通骨格（TASK-3.1a・イシュー #132・REQ-3）。
//!
//! AI が生成した変更（コード修正・パラメータ調整等）を試行し、`guardrail` クレートの
//! 3 分岐判定を必ず経由させたうえで取り込み可否を決める（REQ-3。判定の迂回経路を作らない。
//! `.claude/rules/security.md`）。ループ試行ログは改竄検知可能な形式で記録し、
//! 取り込み判断の根拠を追跡可能にする。
//!
//! REQ-3（`docs/spec/04-requirements.md`）が要求する
//! 「検出 → 可否判断 → 修正生成 → 検証 → 取り込み/却下」の 1 ループを、
//! バグ修正・性能回帰・機能追加のいずれの種別にも共通するインターフェース
//! （[`stages`] の trait 群）と状態遷移（[`runner::SelfRepairLoop`]）として
//! 実装する。設計は PoC-2 の 3 題材（
//! `docs/spec/03-poc/poc-2-ai-self-maintenance/README.md`）が示すループ構造
//! 「ベースライン確認 → 検出 → 修正試行（複数回。失敗時は却下して再試行）→
//! 4 ゲート検証 → 取り込み/却下」をそのまま写像したものであり、新規の概念を
//! 持ち込まない。検証フェーズのベンチゲート（[`verify_bench`]・TASK-3.2a・#137）は
//! この 4 ゲート検証のうち計測系を担う [`VerificationGate`] 実装として、決定的シード
//! 再輸出（TASK-4.4b・#113）と同じ理由で `guardrail` 経由に依存を一本化している
//! （モジュール末尾 `verify_bench` のドキュメント参照）。
//!
//! 移植元は `Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/`
//! （`docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//! 本イシュー（TASK-3.1a）のスコープはループの**制御構造・状態遷移**の移植
//! のみであり、種別ごとの検出・修正生成ロジックの実装は含まない
//! （受け入れ条件は「ループ骨格が新 workspace でビルドできる」こと）。
//!
//! # モジュール構成
//! - [`kind`][]: 種別軸 [`kind::RepairKind`]（`BugFix`/`PerfRegression`/
//!   `FeatureAddition`）。種別ごとの検出・修正生成ロジック実装時（後続イシュー）の
//!   判別子として使う。
//! - [`stages`][]: 種別非依存の段階インターフェース（`Detector`/`FixGenerator`/
//!   `VerificationGate`/`AdoptionJudge`）。
//! - [`outcome`][]: ループ全体の結論 [`outcome::LoopOutcome`] と、検証通過を
//!   型で保証する [`outcome::VerifiedEvidence`]（typestate）。本イシューでは
//!   guardrail 非依存の v1 S1 形で移植する（`outcome` モジュールコメント参照）。
//! - [`report`][]: 試行回数・所要時間・判断根拠を保持する [`report::LoopReport`]
//!   （構造化ログ出力 seam）。段階の実行自体が失敗した場合は
//!   [`report::LoopFailure`] がそれまでの試行記録ごとエラーを保持する。
//! - [`runner`][]: 上記 trait を組み合わせて 1 ループを実行するオーケストレータ
//!   [`runner::SelfRepairLoop`]。
//! - [`error`][]: 型付きエラー [`error::SelfRepairError`]。
//! - [`exec`][]: コマンド実行抽象（[`exec::CommandRunner`]・
//!   [`exec::SystemCommandRunner`]）。[`verify_gates::CargoVerificationGate`]
//!   が `cargo build`/`test`/`clippy` を起動するのに使う（TASK-3.1c・#134）。
//! - [`candidate`][]: 修正生成フェーズの候補適用基盤（[`candidate::CandidateFix`]・
//!   [`candidate::CandidateFixGenerator`]。種別非依存。TASK-3.1c・#134）。
//! - [`verify_gates`][]: 検証フェーズ 3 ゲート（build/test/clippy）の実実行
//!   [`verify_gates::CargoVerificationGate`]（TASK-3.1c・#134）。
//!
//! # 本クレートが担わない責務（TASK-3.1c 完了時点でのスコープ・
//! `.claude/rules/out-of-scope-tracking.md` 準拠）
//! - 種別別 Detector・実候補列の供給（`bug_fix`/`perf_regression`/
//!   `feature_addition` 相当の検出・候補選定ロジック本体） → イシュー #133
//!   （TASK-3.1b）・TASK-3.3（再実証）
//! - 検証フェーズ 4 ゲートのうちベンチゲート（[`verify_bench::SelfRepairBenchGate`]）
//!   の [`stages::VerificationGate`] への結線（4 ゲート合成） → イシュー #136 系
//!   （TASK-3.2）
//! - guardrail 3 分岐判定との統合（`judge` 相当・[`outcome::VerifiedEvidence`]
//!   への guardrail シグナル拡張） → イシュー #135（TASK-3.1d。`guardrail`
//!   クレート自体の CLI 移植〈TASK-4.1〉はイシュー #103 が別途追跡）
//! - `exec`（コマンド実行抽象）の `guardrail` 側への共通化（`guardrail check`
//!   実シグナル計測経路・TASK-6.1c・#199 との統合時に検討） → 未起票
//!   （本イシュー〈#134〉の PR 本文に記録）
//! - ログ形式（`logging` 相当・SHA-256 ハッシュチェーン・`sha2` 依存追加。
//!   依存追加はユーザー承認事項） → イシュー #145（TASK-3.4）
//! - CLI バイナリ（`self-repair run`/`verify-log`。
//!   `docs/guardrail-self-repair-cli.md` 3 節） → 後続タスク（既存イシューで
//!   追跡済み）

pub mod candidate;
pub mod error;
pub mod exec;
pub mod kind;
pub mod outcome;
pub mod report;
pub mod runner;
pub mod stages;
pub mod verify_gates;

pub use candidate::{CandidateFix, CandidateFixGenerator};
pub use error::SelfRepairError;
pub use exec::{CommandRunner, SystemCommandRunner};
pub use kind::RepairKind;
pub use outcome::{AdoptionVerdict, LoopOutcome, VerifiedEvidence};
pub use report::{LoopFailure, LoopReport};
pub use runner::SelfRepairLoop;
pub use stages::{AdoptionJudge, Detector, FixGenerator, VerificationGate};
pub use verify_gates::CargoVerificationGate;

/// 決定的シード設定ユーティリティ（TASK-4.4b・イシュー #113）。
///
/// 学習を伴う回帰テストがモデル初期化前に決定的シードを設定するための入口を、
/// `guardrail::determinism`（PRNG 本体の実体・`.claude/rules/deps-policy.md`
/// 準拠の path 依存経由）から再輸出する。`self-repair` が取り込む AI 生成
/// 変更の検証（取り込み判断に先立つテスト実行。#134 のスコープ）でも同一の
/// 決定性契約（同一シード → 同一系列）を使えるようにするため、
/// `bench-harness` へ直接依存を重ねず `guardrail` 経由に一本化する
/// （3 分岐判定を必ず経由させる依存方向と揃える）。
pub use guardrail::determinism;

/// ベンチゲート（TASK-3.2a・イシュー #137）。
///
/// 検証フェーズ 4 ゲートのうちベンチゲートの計測系を `guardrail::bench_gate`
/// （`bench-harness` 付け替え済み・TASK-4.1d）経由で実行する [`SelfRepairBenchGate`][sbg] を
/// 提供する。決定的シードと同じ理由で `bench-harness` への直接依存は持たず `guardrail`
/// 経由に一本化する（モジュール冒頭ドキュメント参照）。
///
/// [sbg]: verify_bench::SelfRepairBenchGate
pub mod verify_bench;
