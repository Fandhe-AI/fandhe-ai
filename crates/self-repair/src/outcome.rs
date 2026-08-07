//! ループ全体の結論と、検証通過を型で保証する typestate（TASK-3.1a・イシュー #132・REQ-3）。
//!
//! [`crate::runner::SelfRepairLoop`] が返す最終結果 [`LoopOutcome`] と、
//! [`crate::stages::AdoptionJudge`] への入力を検証済みに限定する
//! [`VerifiedEvidence`] を定義する。
//!
//! v1（`Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/outcome.rs`）の
//! S2 版は `VerifiedEvidence` が `guardrail::GateSignals`/`guardrail::BenchSignal`
//! 等の guardrail 型を保持するが、v2 の `guardrail` クレートはまだ移植されて
//! いない（イシュー #103 が open）。本イシューでは guardrail 非依存の v1 S1
//! 形（`attempt`・`proposal_summary`・`gate_report` の 3 フィールド）で移植し、
//! guardrail シグナル（`gates`/`bench`/`lines_changed`/`api_broken`/
//! `gaming_suspect`/`exclusion_rule_ids`）の追加は guardrail 統合を担う
//! 後続イシュー（#135）のスコープとする。

/// 自己修復ループ 1 回分の最終結論。
///
/// PoC-2 の「取り込み/却下」の 2 分岐に加え、`Exhausted`（試行上限到達）と
/// `NoActionNeeded`（検出段階で修正不要と判断され、そもそもループが
/// 開始されなかった）を明示的に区別する。いずれも「取り込まれない」点は
/// 共通するが、理由（検証・判断の否定 / 試行回数の枯渇 / そもそも不要）を
/// 混同すると自己修復ループの監査可能性（`.claude/rules/security.md`:
/// 試行ログから取り込み判断の根拠を追跡可能にする）を損なうため、variant を
/// 分けている。
///
/// match は網羅列挙とし `_ =>` を使わない（guardrail と同じ fail-closed
/// 設計。variant 追加時に既存の呼び出し側が黙って誤った分岐へ落ちるのを防ぐ。
/// `.claude/rules/security.md` A05）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopOutcome {
    /// 検出段階で修正不要と判断され、ループを開始しなかった。
    NoActionNeeded,
    /// 検証・取り込み判断を経て取り込まれた。
    Adopted,
    /// 取り込み判断が人間レビューへ回すべきと判定した（この試行で確定はしない。
    /// 本イシューでは即座にループを終了する。エスカレーション後の扱いは
    /// guardrail 統合〔#135〕以降のスコープ）。`reason` は取り込み判断が返した
    /// エスカレーション理由（`Rejected` と同様、`Escalated` でも理由を保持し
    /// `LoopReport` から追跡できるようにする。security.md の「取り込み判断の
    /// 根拠を追跡可能にする」要求に対応）。
    Escalated { reason: String },
    /// 再試行の余地なく却下が確定した（[`AdoptionVerdict::Reject`] の
    /// `retryable = false` に対応。`stage` は却下が確定した段階名、`reason`
    /// は理由）。
    Rejected { stage: &'static str, reason: String },
    /// 試行上限に達し、いずれの試行も検証・取り込み判断を通過しなかった。
    Exhausted,
}

/// 取り込み判断（[`crate::stages::AdoptionJudge::judge`]）の結論。
///
/// guardrail 統合（#135）後は `guardrail::decision::Verdict`（却下 >
/// エスカレーション > 自動適用の優先順序）と同型・同順序に揃える想定
/// （TASK-3.1 実装計画セクション 3。現時点では guardrail クレート未移植の
/// ため参照しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdoptionVerdict {
    /// 取り込みを承認する。
    Adopt,
    /// 人間によるレビューへ回す。`reason` はエスカレーション理由（`Reject` の
    /// `reason` と対称に持たせ、`LoopReport` へ経路を確保する）。
    Escalate { reason: String },
    /// 取り込みを拒否する。`retryable = true` なら
    /// [`crate::runner::SelfRepairLoop`] は次の試行へ進む（PoC-2:
    /// 「失敗時は却下して再試行」）。`false` ならその場でループを終了し
    /// [`LoopOutcome::Rejected`] を返す（再試行しても結論が変わらないことが
    /// 自明な却下。例: 検出された事象自体が対応不能と判明した場合）。
    Reject { retryable: bool, reason: String },
}

/// 検証（[`crate::stages::VerificationGate::verify`]）を通過した証跡。
///
/// コンストラクタを `pub(crate)` に限定している。これが型として保証するのは
/// 「クレート外からは構築不能」という境界であり、本クレート内での呼び出し元を
/// `VerificationGate` 実装に強制するものではない（本クレート内の任意の
/// コードから呼び出せる。運用上は `VerificationGate::verify` 内でのみ構築する
/// 契約をレビューで担保する）。これにより
/// [`crate::stages::AdoptionJudge::judge`] は「クレート外から検証迂回で
/// 構築された証跡」を受け取れず、クレート境界を越えて検証を迂回し取り込み
/// 判断へ到達する経路をコンパイル時に封じる（`.claude/rules/security.md`
/// A08: 自己修復ループが取り込む変更はガードレール判定を必ず経由し、判定の
/// 迂回経路を作らない）。
///
/// guardrail 統合（#135）で `gates`/`bench`/`lines_changed`/`api_broken`/
/// `gaming_suspect`/`exclusion_rule_ids` の構造化シグナルを追加する想定
/// （v1 `outcome.rs` S2 版参照）。本イシューでは guardrail 非依存の 3
/// フィールドのみを保持する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEvidence {
    attempt: u32,
    proposal_summary: String,
    gate_report: String,
}

impl VerifiedEvidence {
    /// `pub(crate)` 構築子。可視性が保証するのは「クレート外からは到達不能」
    /// という境界のみであり、本クレート内の任意のコード（`VerificationGate`
    /// 実装に限らない）から呼び出せる。型システムが強制するのはこのクレート
    /// 境界までであり、「`VerificationGate::verify` 内でのみ構築する」という
    /// 運用上の契約（stages.rs モジュールコメント参照）はレビューで担保する。
    /// 本 typestate の安全性の根拠は「クレート外から検証迂回で構築できない」
    /// ことであるため、可視性を `pub` へ緩めない。
    ///
    /// 検証ゲート実実行（イシュー #134）が本クレート内に `VerificationGate`
    /// 実装を追加するまでは、本コンストラクタの呼び出し元は
    /// `runner.rs` の `#[cfg(test)]` テストダブルのみである。`cargo clippy`
    /// の non-test ビルドでは呼び出し元が存在せず `dead_code` と判定される
    /// ため、意図的に `#[allow]` する（#134 で実装が追加され次第、外す）。
    #[allow(dead_code)]
    pub(crate) fn new(
        attempt: u32,
        proposal_summary: impl Into<String>,
        gate_report: impl Into<String>,
    ) -> Self {
        VerifiedEvidence {
            attempt,
            proposal_summary: proposal_summary.into(),
            gate_report: gate_report.into(),
        }
    }

    /// この証跡が対応する試行番号（1 始まり。[`crate::stages::Proposal::attempt`]
    /// と同じ値）。取り込み判断の実装が
    /// [`crate::error::SelfRepairError::Judgement`] を組み立てる際に使う。
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// 検証対象となった修正案の要約（[`crate::stages::AdoptionJudge`] や
    /// ログ出力が参照する）。
    pub fn proposal_summary(&self) -> &str {
        &self.proposal_summary
    }

    /// 検証ゲートが出力した根拠（build/test/clippy 等の結果要約。人間可読・
    /// ログ埋め込み用）。
    pub fn gate_report(&self) -> &str {
        &self.gate_report
    }
}
