//! 自己修復ループ。
//!
//! AI が生成した変更（コード修正・パラメータ調整等）を試行し、`guardrail` クレートの
//! 3 分岐判定を必ず経由させたうえで取り込み可否を決める（REQ-3。判定の迂回経路を作らない。
//! `.claude/rules/security.md`）。ループ試行ログは改竄検知可能な形式で記録し、
//! 取り込み判断の根拠を追跡可能にする。
//!
//! 雛形段階（TASK-1.1 部分実装。許容依存の `Cargo.toml` 反映を除く。反映はユーザー承認を
//! 要するため別イシューで対応する）に、決定的シード再輸出（TASK-4.4b・#113）・検証
//! フェーズのベンチゲート（[`verify_bench`]・TASK-3.2a・#137）を順次追加している段階
//! である（spec 根拠: `docs/spec/05-tasks.md` TASK-1.1、REQ-3）。検証フェーズ骨格
//! （Detector／FixGenerator／VerificationGate／AdoptionJudge の 4 trait 構成。
//! TASK-3.1・#132〜#135）は本クレート実装時点で未着手であり、[`verify_bench`] は
//! その `VerificationGate` 実装への結線を待つ独立モジュールとして存在する。
//!
//! # 決定的シード設定ユーティリティ（TASK-4.4b・イシュー #113）
//!
//! 学習を伴う回帰テストがモデル初期化前に決定的シードを設定するための入口を、
//! `guardrail::determinism`（PRNG 本体の実体・`.claude/rules/deps-policy.md` 準拠の
//! path 依存経由）から再輸出する。`self-repair` が取り込む AI 生成変更の検証（3 分岐
//! 判定に先立つテスト実行）でも同一の決定性契約（同一シード → 同一系列）を使えるように
//! するため、`bench-harness` へ直接依存を重ねず `guardrail` 経由に一本化する
//! （3 分岐判定を必ず経由させる依存方向と揃える）。
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
