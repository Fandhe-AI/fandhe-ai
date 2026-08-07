//! 種別非依存の段階インターフェース（TASK-3.1a・イシュー #132・REQ-3）。
//!
//! PoC-2 のループ構造「ベースライン確認 → 検出 → 修正試行（複数回。失敗時は
//! 却下して再試行）→ 4 ゲート検証 → 取り込み/却下」を 4 つの trait に写像する。
//! [`crate::runner::SelfRepairLoop`] がこれらを組み合わせて 1 ループを実行する
//! オーケストレータであり、各 trait の実装は種別（[`crate::kind::RepairKind`]）
//! ごとに後続イシュー（#133: 検出・可否判断、#134: 検証 3 ゲート実実行）で
//! 用意される想定（本イシューでは実装を持たずシグネチャのみ確定する）。
//!
//! # 呼び出し順序の契約（変更禁止・runner.rs が前提とする）
//! 1. [`Detector::detect`] — 1 回。`NoActionNeeded` なら以降の段階へ進まない
//!    （PoC-2 の「要修正判断」を検出結果に含める設計。独立 trait を起こさず
//!    [`DetectionOutcome`] の variant で表現し、新規概念を増やさない）。
//! 2. [`FixGenerator::generate`] — 試行ごとに 1 回。
//! 3. [`VerificationGate::verify`] — 修正生成の直後に 1 回。実装は
//!    「引数配列で起動しシェル経由の文字列展開をしない」方針
//!    （`.claude/rules/security.md` A03）を踏襲すること（#134 のスコープ）。
//! 4. [`AdoptionJudge::judge`] — 検証 `Passed`（[`crate::outcome::VerifiedEvidence`]
//!    取得）の場合のみ 1 回。検証 `Failed` の場合は判断段階を呼ばずに次の試行へ
//!    進む（PoC-2: 検証落ちは再試行対象であり、取り込み判断の対象にすらしない）。

use crate::error::SelfRepairError;
use crate::kind::RepairKind;
use crate::outcome::VerifiedEvidence;

/// 検出段階の結果。
///
/// `NoActionNeeded` は PoC-2 の「要修正判断」でベースラインとの差分が
/// 修正を要しないと判断されたケースに対応する（[`crate::runner::SelfRepairLoop`]
/// はこの場合ループを開始せず即座に完了する）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionOutcome {
    /// 検出されたが修正不要（例: ベンチ劣化が閾値未満）。
    NoActionNeeded,
    /// 修正が必要な事象を検出した。
    Finding(Finding),
}

/// 検出された「修正すべき事象」（PoC-2 の検出結果に対応）。
///
/// `kind` と `summary` で可視性の方針を意図的に分けている。`kind` は
/// `Finding::new` 経由でのみ設定され、構築後は読み取り専用（外部コードから
/// 書き換えられない）としたいフィールドのため private + アクセサとする。
/// `summary` は人間可読なログ用の自由文字列であり、そのような読み取り専用
/// 制約を必要としないため `pub` フィールドで直接公開する
/// （[`Proposal::description`] と同じ方針）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    kind: RepairKind,
    /// 検出内容の要約（人間可読・ログ埋め込み用）。
    pub summary: String,
}

impl Finding {
    pub fn new(kind: RepairKind, summary: impl Into<String>) -> Self {
        Finding {
            kind,
            summary: summary.into(),
        }
    }

    /// この Finding がどの種別に属するか（種別ごとの検出・修正生成実装〔#133〕
    /// の選択、種別非依存性テストの検証に使う）。
    pub fn kind(&self) -> RepairKind {
        self.kind
    }
}

/// 検出段階の抽象（種別非依存）。
///
/// 実装は種別ごとに後続イシュー（#133）が用意する想定（例: `BugFix` 用は
/// テスト失敗差分、`PerfRegression` 用はベンチ中央値比較、`FeatureAddition`
/// 用は受け入れ基準差分から検出する）。本 trait 自体は種別に依存しない
/// シグネチャのみを定める。
pub trait Detector {
    /// `kind` に対応する事象を検出する。
    ///
    /// 実装がベースライン取得・差分計算そのものに失敗した場合は
    /// [`SelfRepairError::Detection`] を返す（fail-closed。判定不能を
    /// `NoActionNeeded` に丸めてはならない）。
    fn detect(&self, kind: RepairKind) -> Result<DetectionOutcome, SelfRepairError>;
}

/// 修正案（PoC-2 の「修正試行」の 1 回分に対応）。
///
/// `attempt`・`description` とも不変条件を持たない値のため `pub`
/// フィールドとする（[`Finding`] の可視性方針の説明を参照）。ただし
/// `attempt` は [`FixGenerator::generate`] の呼び出し元（`runner.rs`）が
/// 渡した試行番号と一致するはずの値であり、[`crate::runner::SelfRepairLoop::run`]
/// はループ側の試行カウンタを単一の真実源として、返された `Proposal.attempt`
/// との不一致を検査する（レビュー指摘: attempt 番号に単一の真実源がないと
/// 監査ログ〈`LoopReport`〉とエラー詳細〈`SelfRepairError::Judgement`〉の
/// attempt 番号が食い違いうる）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proposal {
    /// 何回目の試行で生成されたか（1 始まり）。
    pub attempt: u32,
    /// 修正内容の要約（人間可読・ログ埋め込み用）。
    pub description: String,
}

/// 修正生成段階の抽象（種別非依存）。
///
/// `finding` と現在の試行回数 `attempt` を受け取り、[`Proposal`] を生成する。
/// 前回試行の失敗理由を踏まえた再試行ロジック（例: 別アプローチを試す）は
/// 実装側（#133）の責務であり、本 trait はシグネチャのみを定める。
pub trait FixGenerator {
    fn generate(&self, finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError>;
}

/// 検証段階の結果。
///
/// `Passed` は [`VerifiedEvidence`] を伴う。`VerifiedEvidence` は
/// `pub(crate)` コンストラクタしか持たないため、クレート外からは構築できず
/// [`VerificationGate::verify`] を経由せずに構築することはできない（A08
/// 対応。取り込み判断 [`AdoptionJudge::judge`] をクレート外から検証迂回で
/// 呼び出す経路を型レベルで封じる。ただし本クレート内では
/// `VerificationGate` 実装以外からも呼び出し可能であり、`verify` 内でのみ
/// 構築する契約は運用上のものである。`outcome.rs` の `VerifiedEvidence` の
/// doc 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationOutcome {
    Passed(VerifiedEvidence),
    /// 検証不合格（PoC-2: 却下して次の試行へ進む対象）。
    Failed {
        reason: String,
    },
}

/// 検証段階の抽象（種別非依存）。
///
/// 本イシュー（TASK-3.1a）では trait（seam）のみを定義する。実際の
/// build/test/clippy ゲート実行・`guardrail` 3 分岐判定との統合は #134
/// （検証ゲート実実行）・#135（guardrail 統合）の責務。
pub trait VerificationGate {
    fn verify(&self, proposal: &Proposal) -> Result<VerificationOutcome, SelfRepairError>;
}

/// 取り込み判断段階の抽象（種別非依存）。
///
/// 検証済み証跡 [`VerifiedEvidence`] のみを入力に取る（型レベルで検証迂回を
/// 封じる契約。stages.rs モジュールコメント参照）。出力
/// [`crate::outcome::AdoptionVerdict`] の 3 値・優先順序は、guardrail 統合後
/// （#135）に `guardrail::decision::Verdict`（却下 > エスカレーション >
/// 自動適用）と同型・同順序になる想定である（現時点では guardrail クレートは
/// 未移植〈イシュー #103〉のため参照しない）。
///
/// 実装が `Err(SelfRepairError::Judgement { attempt, .. })` を返す場合は
/// 固定値・推測値ではなく `evidence.attempt()` を使うこと（v1 PR #170 での
/// Bugbot 指摘対応を踏襲）。`VerifiedEvidence` は検証対象となった試行の
/// 番号を保持しており、`judge` はそれ以外から試行番号を知る手段を持たない
/// （リトライ後のエラー文言・監査ログが誤った attempt を指すのを防ぐ）。
pub trait AdoptionJudge {
    fn judge(
        &self,
        evidence: &VerifiedEvidence,
    ) -> Result<crate::outcome::AdoptionVerdict, SelfRepairError>;
}
