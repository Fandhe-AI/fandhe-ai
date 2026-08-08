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
//! - [`exec`][]・[`candidate`][]・[`bug_fix`][]・[`perf_regression`][]・
//!   [`feature_addition`][]（TASK-3.1b・イシュー #133）: 検出・可否判断
//!   フェーズの種別ごとの実装。[`exec::CommandRunner`] はコマンド実行 seam
//!   （v2 `guardrail` に `exec` モジュールが未移植のため本クレート内に新設。
//!   `exec` モジュールの doc 参照）、[`candidate::apply_candidate`] は
//!   `bug_fix`/`feature_addition` 共通の「候補存在確認 → baseline 復元 →
//!   候補適用」ロジック、[`perf_regression`] は
//!   `guardrail::median_gate`/`guardrail::Thresholds` に一本化した性能回帰
//!   検出を提供する。
//! - [`outcome`][]: ループ全体の結論 [`outcome::LoopOutcome`] と、検証通過を
//!   型で保証する [`outcome::VerifiedEvidence`]（typestate）。本イシューでは
//!   guardrail 非依存の v1 S1 形で移植する（`outcome` モジュールコメント参照）。
//! - [`report`][]: 試行回数・所要時間・判断根拠を保持する [`report::LoopReport`]
//!   （構造化ログ出力 seam）。段階の実行自体が失敗した場合は
//!   [`report::LoopFailure`] がそれまでの試行記録ごとエラーを保持する。
//! - [`runner`][]: 上記 trait を組み合わせて 1 ループを実行するオーケストレータ
//!   [`runner::SelfRepairLoop`]。
//! - [`judge`][]: guardrail 3 分岐判定を [`stages::AdoptionJudge`] として接続する
//!   [`judge::GuardrailAdoptionJudge`]（TASK-3.1d・#135）。`evidence` の 6 シグナルを
//!   `guardrail::DecisionInput::new` へそのまま渡し `guardrail::decide` を呼ぶだけの
//!   薄いアダプタであり、`decide` を経由しない [`outcome::AdoptionVerdict`] 生成経路を
//!   持たない（A08: 判定の迂回経路を作らない）。
//! - [`error`][]: 型付きエラー [`error::SelfRepairError`]。
//! - [`exec`][]: コマンド実行抽象（[`exec::CommandRunner`]・
//!   [`exec::SystemCommandRunner`]）。[`verify_gates::CargoVerificationGate`]
//!   が `cargo build`/`test`/`clippy` を起動するのに使う（TASK-3.1c・#134）。
//! - [`candidate`][]: 修正生成フェーズの候補適用基盤（[`candidate::CandidateFix`]・
//!   [`candidate::CandidateFixGenerator`]。種別非依存。TASK-3.1c・#134）。
//! - [`verify_gates`][]: 検証フェーズ 3 ゲート（build/test/clippy）の実実行
//!   [`verify_gates::CargoVerificationGate`]（TASK-3.1c・#134）。
//! - [`verify_composite`][]: 3 ゲート（`verify_gates`）とベンチゲート
//!   （`verify_bench`）を合成した [`verify_composite::FeatureAdditionCompositeGate`]
//!   （TASK-3.3c・#142）。TASK-3.2 がスコープ外としていた「4 ゲート合成」の
//!   結線点を、機能追加種別のループ完走実証のために満たす（`verify_composite`
//!   モジュール冒頭ドキュメント参照）。
//! - [`diff_signals`][]: `lines_changed`／`api_broken`／`gaming_suspect`／
//!   `exclusion_rule_ids` の**試行ごとの**実測（TASK-3.2a・#137）。
//!   [`verify_gates::CargoVerificationGate`] が構築時固定の必須引数として
//!   受け取る契約はそのままに、この実測ロジック自体を `tests/` 専用実装から
//!   `src/` 本体へ昇格する。
//! - [`verify_bench_direct`][]: ベンチゲートの「候補 diff に対する直接実測」
//!   （TASK-3.2a・#137）。[`verify_bench_direct::DirectBenchRunner`] が
//!   baseline commit・候補適用済み作業木の双方を release ビルドし、外部
//!   タイミング方式で [`verify_bench::SelfRepairBenchGate`]（判定ロジック
//!   自体は変更しない）へ委譲する。
//! - [`verify_direct_composite`][]: `diff_signals`・`verify_bench_direct`・
//!   `verify_gates` を合成した真の 4 ゲート合成
//!   [`verify_direct_composite::RepairCompositeGate`]（TASK-3.2a・#137）。
//!   `verify_composite::FeatureAdditionCompositeGate`（合成ワークロード版・
//!   構築時固定シグナル版）とは別モジュールとして共存する（`verify_direct_composite`
//!   モジュール冒頭ドキュメント参照）。
//! - [`sha256`][]: FIPS 180-4 準拠 SHA-256 の自作実装（TASK-3.4・#145）。`sha2`
//!   クレートは許容依存 8 区分外・依存追加はユーザー承認事項のため、
//!   `logging` のハッシュチェーン計算専用に std のみで実装する
//!   （`sha256` モジュール冒頭ドキュメント参照）。
//! - [`logging`][]: 試行ログの JSON Lines・SHA-256 ハッシュチェーン出力
//!   （TASK-3.4・#145）。[`report::LoopReport`]/[`report::LoopFailure`] を
//!   受け取り、追記専用ファイルへ改竄検知可能な形式で記録する
//!   [`logging::LogWriter`] と、記録済みログの整合性を検証する
//!   [`logging::verify_chain`] を提供する（`.claude/rules/security.md`:
//!   ループ試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を
//!   追跡可能にする、という要求への対応。詳細は `docs/self-repair-log-format.md`）。
//! - [`cli`][]: `self-repair` バイナリ（`src/main.rs`）向けの自作コマンドライン
//!   引数パーサ（TASK-3.4 残作業・#145）。`verify-log` サブコマンドが
//!   [`logging::verify_chain`] を CLI から結線し、監査担当者が `cargo test`
//!   経由でなく直接ログを検証できるようにする（`docs/guardrail-self-repair-cli.md`
//!   3.2 節。詳細は `cli` モジュール冒頭ドキュメント参照）。
//!
//! # 本クレートが担わない責務（TASK-3.1c 完了時点でのスコープ・
//! `.claude/rules/out-of-scope-tracking.md` 準拠）
//! - 種別別の実候補列の供給（`bug_fix`/`perf_regression`/`feature_addition` の
//!   `Detector`・`FixGenerator` 骨格自体は TASK-3.1b・イシュー #133 で実装済み
//!   だが、実 AI 生成修正の動的取得・実運用題材での再実証は TASK-3.3 のスコープ）
//! - 検証フェーズ 4 ゲートのうちベンチゲート（[`verify_bench::SelfRepairBenchGate`]）
//!   の [`stages::VerificationGate`] への結線（4 ゲート合成） → イシュー #136 系
//!   （TASK-3.2）
//! - `lines_changed`/`api_broken`/`gaming_suspect`/`exclusion_rule_ids`
//!   の実測（diff 解析・ポリシー除外リスト評価）
//!   → [`verify_gates::CargoVerificationGate`] は構築時にこれらを呼び出し元
//!   から必須引数として受け取るのみで、自ら計測しない（未計測値を fail-open
//!   な既定値で埋めない。モジュール末尾 `verify_gates` のドキュメント参照）。
//!   実測経路の配線は #133・TASK-3.3（再実証）のスコープ
//! - `exec`（コマンド実行抽象）の `guardrail` 側への共通化（`guardrail check`
//!   実シグナル計測経路・TASK-6.1c・#199 との統合時に検討） → 未起票
//!   （本イシュー〈#134〉の PR 本文に記録）
//! - CLI バイナリ `self-repair run`（`docs/guardrail-self-repair-cli.md`
//!   3.1 節）・`verify-log`（3.2 節）はいずれも [`cli`]／`src/main.rs` として
//!   実装済み（`verify-log` は #145 差し戻し分、`run` は #142 差し戻し分。
//!   `run` は `--kind bug-fix`／`--kind feature-addition` に完全対応するが、
//!   `--kind perf-regression` は `PerfRegressionDetector`/
//!   `PerfRegressionFixGenerator` が他 2 種別と非対称な構築契約
//!   〈`BenchMeasurer`・戦略リスト〉を持ち #141／#142 いずれも本種別を
//!   必要としないため実行時未対応〈内部エラー扱い〉のまま。
//!   `docs/guardrail-self-repair-cli.md` 3.1 節参照）
//! - guardrail クレート自体の CLI 移植（TASK-4.1）→ イシュー #103 が別途追跡

pub mod bug_fix;
pub mod candidate;
pub mod cli;
pub mod diff_signals;
pub mod error;
pub mod exec;
pub mod feature_addition;
pub mod judge;
pub mod kind;
pub mod logging;
pub mod outcome;
pub mod perf_regression;
pub mod report;
pub mod runner;
pub mod sha256;
pub mod stages;
pub mod verify_bench_direct;
pub mod verify_composite;
pub mod verify_direct_composite;
pub mod verify_gates;

#[cfg(test)]
pub(crate) mod test_support;

pub use bug_fix::{BugFixDetector, BugFixFixGenerator};
pub use candidate::{CandidateFix, CandidateFixGenerator};
pub use error::SelfRepairError;
pub use exec::{CommandOutput, CommandRunner, SystemCommandRunner};
pub use feature_addition::{FeatureAdditionDetector, FeatureAdditionFixGenerator};
pub use judge::GuardrailAdoptionJudge;
pub use kind::RepairKind;
pub use logging::{LogError, LogWriter, VerifyChainSummary, verify_chain};
pub use outcome::{AdoptionVerdict, LoopOutcome, VerifiedEvidence};
pub use perf_regression::{BenchMeasurer, PerfRegressionDetector, PerfRegressionFixGenerator};
pub use report::{LoopFailure, LoopReport};
pub use runner::SelfRepairLoop;
pub use stages::{AdoptionJudge, Detector, FixGenerator, VerificationGate};
pub use verify_composite::FeatureAdditionCompositeGate;
pub use verify_direct_composite::{RepairCompositeGate, RepairCompositeGateSpec};
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
