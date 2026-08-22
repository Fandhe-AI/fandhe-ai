//! 3 分岐判定ロジック本体（TASK-4.1b・イシュー #105・REQ-4）。
//!
//! `guardrail` が収集する 5 条件（変更行数・ベンチ劣化・build/test/clippy
//! ゲート・公開 API 非破壊・ゲーミング疑い）の評価済みシグナルを受け取り、
//! **却下（Reject）> エスカレーション（Escalate）> 自動適用（AutoApply）** の
//! 優先順序で判定する（`docs/spec/04-requirements.md` REQ-4）。判定順序の正本は
//! PoC-3 の `docs/spec/03-poc/poc-3-guardrail-validity/code/guardrail.sh:167-210`
//! （判定セクション）であり、本モジュールはこれを Rust で productize したもの。
//! v1（`rust-ai-library-v1/crates/guardrail/src/decision.rs`）からの移植で
//! あり、受け入れ条件「5 条件の判定が v1 と同一結果を返す」を満たすため
//! 判定順序・型・定数を変更せず移植する。
//!
//! # 本モジュールが担う責務・境界
//! - **担う**: 評価済みシグナル（[`DecisionInput`]）から [`Decision`] を導出する
//!   純粋関数 [`decide`]。判定根拠は型付き [`Reason`]（[`Reason::condition`] が
//!   CI・自己修復ループ向けの機械可読 ID を返す）。分岐種別の機械可読 ID は
//!   [`Verdict::as_machine_id`]。
//! - **担わない**: 変更行数・公開 API 破壊・ゲーミング疑い・ゲート実行の実測
//!   （シグナル計測。#104/#106 の管轄。本モジュールは実測手段を持たず、呼び
//!   出し元が実測値または契約検証用の値を注入する）。ポリシー除外リストの
//!   評価自体も本モジュールの管轄外（TASK-5.x 系）であり、評価済みの
//!   `exclusion_rule_ids`（match したルール `id` 一覧）を [`DecisionInput`]
//!   経由で受け取るのみ。v1 が持つ `gates::GateStatus`/`bench::BenchOutcome`
//!   からの `From` 変換 impl は、移植元モジュール（`gates.rs`/`bench.rs`）が
//!   v2 に未移植のため本イシューには含めない（#104/#107 が各モジュール移植時
//!   に追加する）。
//!
//! # 判定順序の契約（変更禁止）
//! 1. **却下（最優先）**: `build`/`test`/`clippy` のいずれかが [`GateSignal::Failed`]。
//!    ポリシー除外リストの match は却下を格下げしない（却下時も match の記録
//!    自体は行う。下記「除外リスト match の記録」節参照）。
//! 2. **除外リスト match（無条件エスカレーション）**: 却下でなく、
//!    `exclusion_rule_ids` が 1 件以上ある場合。機械的な自動適用条件（3.）を
//!    一切参照せず、機械判定の結果によらず無条件でエスカレーションへ回す
//!    （REQ-5。security.md「除外リストは判定を迂回させないための最後の砦」）。
//! 3. **機械エスカレーション**: 却下・除外リスト match のいずれでもなく、
//!    次のいずれかが成立する場合。
//!    - `lines_changed > thresholds.lines_max`
//!    - `api_broken == true`
//!    - `gaming_suspect == true`
//!    - [`BenchSignal::Measured`] かつ `median_pct > thresholds.bench_median_max_pct`
//!      （正方向の劣化のみを罰する。改善（負値）・境界値ちょうどは罰しない）
//! 4. **ゲート未全通過エスカレーション**: 却下でも上記 2./3. でもないが、
//!    `gates.all_passed()` が `false`（一部 `Skipped`）の場合。
//! 5. **自動適用**: 上記いずれにも該当しない場合のみ。判定根拠＝逸脱条件と
//!    いう意味論により `reasons` は空。
//!
//! # 除外リスト match の記録（結果によらず必ず行う）
//! `exclusion_rule_ids` が非空の場合、[`Decision::exclusion_rule_ids`] へ
//! そのまま記録する。これは判定順序 1.（却下）が確定した場合も含む —
//! 「除外リストへの match は判定結果に関わらず評価・記録される」という
//! security.md の追跡可能性要求（「取り込み判断の根拠を追跡可能にする」）に
//! 対応するためであり、却下優先の判定順序契約自体は変更しない。
//!
//! `match` は網羅列挙とし `_ =>` ワイルドカードを使わない（fail-closed 設計。
//! variant 追加時に自動適用へ黙って落ちるのを防ぐ。security.md A05）。

use crate::config::Thresholds;
use crate::error::GuardrailError;

/// 3 分岐判定の結論（REQ-4）。severity の高さは `Reject` > `Escalate` >
/// `AutoApply` の順（[`decide`] のモジュールコメント参照）。
///
/// `Serialize`/`Deserialize` は判定レポート JSON（`crate::report::Report`。
/// TASK-4.1a・イシュー #104 管轄）が `verdict` フィールドとして本型を直接
/// シリアライズするために付与する（`snake_case` は `docs/guardrail-self-repair-cli.md`
/// §2.1 の `"auto_apply"`/`"escalate"`/`"reject"` 表記と一致させる）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// 逸脱なし。全ゲート通過・全指標が閾値内。
    AutoApply,
    /// 人間によるレビューへ回す。
    Escalate,
    /// 取り込みを拒否する（build/test/clippy のいずれかが失敗）。
    Reject,
}

impl Verdict {
    /// CLI 出力・判定レポートの表示用（後方互換）の日本語表記。`guardrail.sh`
    /// （PoC-3）互換の表示を踏襲する。機械可読な分岐識別子は
    /// [`Verdict::as_machine_id`] を正とする。
    pub fn as_ja(self) -> &'static str {
        match self {
            Verdict::AutoApply => "自動適用",
            Verdict::Escalate => "エスカレーション",
            Verdict::Reject => "却下",
        }
    }

    /// CI・自己修復ループが分岐に使う機械可読識別子。
    ///
    /// `auto_apply`/`escalate`/`reject` の 3 値を正とする（v1 の
    /// `Report::verdict_id` が使う値と同一の文字列。#106 が接続する）。
    pub fn as_machine_id(self) -> &'static str {
        match self {
            Verdict::AutoApply => "auto_apply",
            Verdict::Escalate => "escalate",
            Verdict::Reject => "reject",
        }
    }
}

/// 判定根拠（逸脱した条件）の型付き表現。
///
/// 判定根拠が機械可読になっていることで、CI・自己修復ループの取り込み判断
/// ログ（REQ-6・spec データ要件、`docs/spec/04-requirements.md`）が条件名で
/// 照合できるようにする。
///
/// `ExclusionMatch` のみ動的な `rule_id`（ポリシー除外リスト設定由来。
/// A03: 設定ファイル自体の内容であり外部から任意注入される文字列ではない）を
/// 保持するため `Copy` は付けず `Clone` のみを derive する。
///
/// `match` は網羅列挙とし `_ =>` を使わない（fail-closed 契約を
/// [`Reason::condition`] / [`std::fmt::Display`] にもそのまま引き継ぐ。
/// variant 追加時に新条件が黙って欠落するのを防ぐ）。
#[derive(Debug, Clone, PartialEq)]
pub enum Reason {
    /// build/test/clippy のいずれかが `Failed`。`gate` は `"build"`/`"test"`/
    /// `"clippy"` のいずれか（`GateSignals::failed_names` が列挙する名前）。
    GateFailed { gate: &'static str },
    /// 一部のゲートが `Skipped` のまま自動適用の前提条件
    /// （`GateSignals::all_passed`）を満たさない（`Failed` は 1 件もないが
    /// `Passed` でもないゲートが残っている状態。Escalate 側の fail-closed 分岐）。
    GateSkipped,
    /// ポリシー除外リスト設定のルール `rule_id` に match（無条件人間承認。REQ-5）。
    ExclusionMatch { rule_id: String },
    /// 変更行数が `Thresholds::lines_max` を超過。
    LinesMaxExceeded { lines_changed: u64, lines_max: u64 },
    /// 公開 API の破壊的変更を検出。
    ApiBroken,
    /// ゲーミング（判定回避）の疑いを検出。
    GamingSuspect,
    /// ベンチ劣化の中央値が `Thresholds::bench_median_max_pct` を超過。
    BenchMedianExceeded { median_pct: f64, bench_max_pct: f64 },
    /// ベンチ計測値が非有限（NaN/inf）。閾値比較を経ずに無条件でエスカレーション
    /// へ回す fail-closed 分岐（NaN が `>` 比較を素通りする回帰を防ぐ）。
    BenchNonFinite { median_pct: f64 },
}

impl Reason {
    /// CI・自己修復ループが照合する機械可読の条件名。`&'static str` に固定し、
    /// 外部由来の任意文字列を混入させないことで理由文言のインジェクションを
    /// 防ぐ。`rule_id` の具体的な値は `Reason::ExclusionMatch` 自体
    /// （[`Decision::exclusion_rule_ids`] 経由でも参照可能）に保持し、
    /// `condition()` は固定文字列を返す（security.md A03）。
    pub fn condition(&self) -> &'static str {
        match self {
            Reason::GateFailed { gate: "build" } => "gate_build_failed",
            Reason::GateFailed { gate: "test" } => "gate_test_failed",
            Reason::GateFailed { gate: "clippy" } => "gate_clippy_failed",
            Reason::GateFailed { .. } => "gate_failed",
            Reason::GateSkipped => "gate_skipped",
            Reason::ExclusionMatch { .. } => "policy_exclusion_match",
            Reason::LinesMaxExceeded { .. } => "lines_max_exceeded",
            Reason::ApiBroken => "api_broken",
            Reason::GamingSuspect => "gaming_suspect",
            Reason::BenchMedianExceeded { .. } => "bench_median_exceeded",
            Reason::BenchNonFinite { .. } => "bench_non_finite",
        }
    }
}

impl std::fmt::Display for Reason {
    /// 判定根拠の日本語詳細文言（テキスト出力・ログ・判定レポート表示の
    /// 後方互換。実測値・閾値を埋め込む）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reason::GateFailed { gate } => write!(f, "ゲート `{gate}` が失敗しました"),
            Reason::GateSkipped => write!(
                f,
                "一部のゲートが Skipped のため自動適用の前提条件（全ゲート Passed）を満たしません"
            ),
            Reason::ExclusionMatch { rule_id } => write!(
                f,
                "ポリシー除外リストのルール `{rule_id}` に一致しました（無条件人間承認。REQ-5）"
            ),
            Reason::LinesMaxExceeded {
                lines_changed,
                lines_max,
            } => write!(
                f,
                "変更行数が上限を超過しました（{lines_changed} > {lines_max}）"
            ),
            Reason::ApiBroken => write!(f, "公開 API の破壊的変更が検出されました"),
            Reason::GamingSuspect => write!(f, "ゲーミング（判定回避）の疑いが検出されました"),
            Reason::BenchMedianExceeded {
                median_pct,
                bench_max_pct,
            } => write!(
                f,
                "ベンチ劣化の中央値が上限を超過しました（{median_pct:.2}% > {bench_max_pct:.2}%）"
            ),
            Reason::BenchNonFinite { median_pct } => write!(
                f,
                "ベンチ計測値が非有限（NaN/inf）です（median_pct = {median_pct}）"
            ),
        }
    }
}

/// build/test/clippy ゲート 1 個の状態。
///
/// `Skipped` は「先行ゲートの失敗により本ゲートが実行されなかった」ことを表す
/// （PoC-3: build 失敗時 test/clippy はスキップされる実行順序契約）。`Passed`
/// でない以上、[`decide`] は `Skipped` を自動適用の根拠には決して使わない
/// （fail-closed。security.md A05）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateSignal {
    Passed,
    Failed,
    Skipped,
}

/// build/test/clippy の 3 ゲート状態一式（REQ-4 条件 (3)）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateSignals {
    pub build: GateSignal,
    pub test: GateSignal,
    pub clippy: GateSignal,
}

impl GateSignals {
    /// いずれかのゲートが `Failed` であれば `true`（却下条件そのもの）。
    fn any_failed(&self) -> bool {
        self.build == GateSignal::Failed
            || self.test == GateSignal::Failed
            || self.clippy == GateSignal::Failed
    }

    /// 全ゲートが `Passed` であれば `true`（自動適用の前提条件の一つ）。
    fn all_passed(&self) -> bool {
        self.build == GateSignal::Passed
            && self.test == GateSignal::Passed
            && self.clippy == GateSignal::Passed
    }

    /// 却下となったゲート名（`build`/`test`/`clippy`）を理由文言用に列挙する。
    /// `Skipped` は先行失敗の帰結であり、それ自体は理由に含めない
    /// （モジュールコメントの契約どおり）。
    fn failed_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.build == GateSignal::Failed {
            names.push("build");
        }
        if self.test == GateSignal::Failed {
            names.push("test");
        }
        if self.clippy == GateSignal::Failed {
            names.push("clippy");
        }
        names
    }
}

/// ベンチ計測結果（REQ-4 条件 (2)。5 回以上計測の中央値。`Thresholds::bench_runs_min`
/// が下限を強制する契約は `config` モジュール側で担保済み）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BenchSignal {
    /// ゲート未通過等により計測していない（PoC-3 準拠: ゲート全通過時のみ
    /// ベンチを計測する運用のため、これ自体は矛盾ではない）。
    NotRun,
    /// 計測済み。`median_pct` は変化率の中央値（正 = 劣化、負 = 改善）。
    Measured { median_pct: f64 },
}

/// 判定入力。5 条件の評価済みシグナル一式と検証済み閾値、およびポリシー
/// 除外リストの評価結果をまとめて受け取る。
///
/// フィールドは非公開とし、[`DecisionInput::new`] を経由した構築のみを許す
/// （`config::Thresholds` と同じく「不変条件を型で強制する」流儀。矛盾入力
/// （ゲート未全通過なのに `BenchSignal::Measured`）は `new` の時点で
/// [`GuardrailError::InconsistentDecisionInput`] として拒否する）。
///
/// `exclusion_rule_ids`（match したポリシー除外リストのルール `id` 一覧。
/// 空 = match なし）は省略可能なデフォルト値を持たない必須引数とする。
/// ビルダー既定値で暗黙に空を許すと「評価忘れ」が「除外リスト素通り」という
/// fail-open な経路になり、REQ-5・security.md A08（判定の迂回経路を作らない）
/// に反するため、全呼び出し元にコンパイルエラー駆動で明示的に渡させる。
#[derive(Debug)]
pub struct DecisionInput<'a> {
    thresholds: &'a Thresholds,
    lines_changed: u64,
    gates: GateSignals,
    api_broken: bool,
    gaming_suspect: bool,
    bench: BenchSignal,
    exclusion_rule_ids: Vec<String>,
}

impl<'a> DecisionInput<'a> {
    /// 各シグナルを検証したうえで判定入力を構築する。
    ///
    /// `exclusion_rule_ids` はポリシー除外リスト評価結果（match したルール
    /// `id` 一覧）をそのまま渡す（match なしは空 `Vec` を明示的に渡す）。
    ///
    /// # エラー
    /// ゲートが全通過（`GateSignals::all_passed`）でないにもかかわらず
    /// `BenchSignal::Measured` が渡された場合、PoC-3 の実行順序契約
    /// （「ゲート全通過時のみベンチを計測する」）に反する呼び出し側バグと
    /// みなし、判定を続行せず fail-closed で拒否する（security.md A08:
    /// 誤った自動適用を出すより実行失敗させる）。
    pub fn new(
        thresholds: &'a Thresholds,
        lines_changed: u64,
        gates: GateSignals,
        api_broken: bool,
        gaming_suspect: bool,
        bench: BenchSignal,
        exclusion_rule_ids: Vec<String>,
    ) -> Result<Self, GuardrailError> {
        // edition 2024 の let-chains で collapsible_if を解消する（挙動は
        // 従来の入れ子 if と同一: ベンチ計測時のみゲート全通過を検査する）。
        if let BenchSignal::Measured { .. } = bench
            && !gates.all_passed()
        {
            return Err(GuardrailError::InconsistentDecisionInput {
                reason:
                    "build/test/clippy が全通過していないにもかかわらずベンチ計測結果が渡されました\
                         （PoC-3 の実行順序契約: ベンチはゲート全通過時のみ計測する）"
                        .to_string(),
            });
        }

        Ok(DecisionInput {
            thresholds,
            lines_changed,
            gates,
            api_broken,
            gaming_suspect,
            bench,
            exclusion_rule_ids,
        })
    }
}

/// `Verdict::AutoApply` 時に [`Decision::reasons`] が空である場合に表示層
/// （CLI・判定レポート）が補うフォールバック文言。「判定根拠＝逸脱条件」
/// （下記 [`Decision`] 型のドキュメント参照）という意味論のもとでも、
/// `guardrail.sh`（PoC-3）互換の出力・CLI 標準出力の双方で空文字列を出さない
/// ための共有定数。
pub const AUTO_APPLY_FALLBACK_REASON: &str = "全ゲート green・全指標が閾値内です";

/// 判定結果。`reasons` は「逸脱した条件」の一覧であり、自動適用
/// （[`Verdict::AutoApply`]）の場合は空（判定根拠＝逸脱条件という意味論を
/// 型から明示する。「全ゲート green・全指標が閾値内」の表示文言は表示層
/// （[`Verdict::as_ja`] と組み合わせて呼び出し元が付与する）に移し、型からは
/// 自由文の既定値を排除した。フォールバック文言の実体は
/// [`AUTO_APPLY_FALLBACK_REASON`] を参照）。
///
/// `exclusion_rule_ids` は判定順序（却下含む）によらず、match したポリシー
/// 除外リストのルール `id` をそのまま保持する（security.md「取り込み判断の
/// 根拠を追跡可能にする」）。
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    verdict: Verdict,
    reasons: Vec<Reason>,
    exclusion_rule_ids: Vec<String>,
}

impl Decision {
    pub fn verdict(&self) -> Verdict {
        self.verdict
    }

    /// 逸脱した条件の一覧（型付き）。`AutoApply` では空。
    pub fn reasons(&self) -> &[Reason] {
        &self.reasons
    }

    /// `reasons()` の機械可読 ID 一覧。判定レポートの `reason_conditions`
    /// フィールドとして CI・自己修復ループへ出力される契約（#106 が接続する）。
    pub fn reason_conditions(&self) -> Vec<&'static str> {
        self.reasons.iter().map(Reason::condition).collect()
    }

    /// match したポリシー除外リストのルール `id` 一覧（空 = match なし）。
    /// 判定順序（却下を含む）によらず常に記録される（モジュールコメント
    /// 「除外リスト match の記録」節参照）。
    pub fn exclusion_rule_ids(&self) -> &[String] {
        &self.exclusion_rule_ids
    }
}

/// 判定本体（純粋関数）。§モジュールコメントの判定順序契約
/// （却下 > 除外リスト match > 機械エスカレーション > ゲート未全通過 >
/// 自動適用の 5 段階）をそのまま実装する。
///
/// `gates`/`bench` の整合性は [`DecisionInput::new`] が構築時点で検証済みの
/// ため、ここでは判定順序のみを扱う。
pub fn decide(input: &DecisionInput) -> Result<Decision, GuardrailError> {
    // 1. 却下（最優先）: build/test/clippy のいずれかが Failed。
    //    除外リスト match の記録はここでも省略しない（モジュールコメント
    //    「除外リスト match の記録」節: 判定結果によらず評価される）。
    if input.gates.any_failed() {
        let reasons = input
            .gates
            .failed_names()
            .into_iter()
            .map(|gate| Reason::GateFailed { gate })
            .collect();
        return Ok(Decision {
            verdict: Verdict::Reject,
            reasons,
            exclusion_rule_ids: input.exclusion_rule_ids.clone(),
        });
    }

    // 2. 除外リスト match（無条件エスカレーション）: 却下でなく、
    //    `exclusion_rule_ids` が 1 件以上あれば、以降の機械判定条件を一切
    //    参照せずエスカレーションを確定する（REQ-5・security.md A08）。
    if !input.exclusion_rule_ids.is_empty() {
        let reasons = input
            .exclusion_rule_ids
            .iter()
            .map(|id| Reason::ExclusionMatch {
                rule_id: id.clone(),
            })
            .collect();
        return Ok(Decision {
            verdict: Verdict::Escalate,
            reasons,
            exclusion_rule_ids: input.exclusion_rule_ids.clone(),
        });
    }

    // 3. 機械エスカレーション: 却下・除外リスト match のいずれでもない場合に、
    //    残り 4 条件の逸脱を全て収集する。reasons はスーパーセットとして
    //    透明性を上げる（該当した逸脱を全て列挙）。
    let mut escalation_reasons: Vec<Reason> = Vec::new();

    if input.lines_changed > input.thresholds.lines_max {
        escalation_reasons.push(Reason::LinesMaxExceeded {
            lines_changed: input.lines_changed,
            lines_max: input.thresholds.lines_max,
        });
    }

    if input.api_broken {
        escalation_reasons.push(Reason::ApiBroken);
    }

    if input.gaming_suspect {
        escalation_reasons.push(Reason::GamingSuspect);
    }

    if let BenchSignal::Measured { median_pct } = input.bench {
        // fail-closed（security.md A08）: NaN は `>` 比較が常に false を返す
        // ため、閾値超過を素通りして誤って自動適用の根拠になり得る。
        // 非有限値（NaN・±inf）はそもそも「計測に失敗した」ことを意味する
        // ため、比較の可否によらず無条件でエスカレーションへ回す。
        if !median_pct.is_finite() {
            escalation_reasons.push(Reason::BenchNonFinite { median_pct });
        } else if median_pct > input.thresholds.bench_median_max_pct {
            // 正方向の劣化のみを罰する。改善（負値）・境界値ちょうど（`==`）は
            // 閾値内として扱う（PoC-3 準拠。比較演算子は `>` を用いる）。
            escalation_reasons.push(Reason::BenchMedianExceeded {
                median_pct,
                bench_max_pct: input.thresholds.bench_median_max_pct,
            });
        }
    }

    if !escalation_reasons.is_empty() {
        return Ok(Decision {
            verdict: Verdict::Escalate,
            reasons: escalation_reasons,
            exclusion_rule_ids: input.exclusion_rule_ids.clone(),
        });
    }

    // 4. ゲート未全通過エスカレーション: 上記いずれにも該当しないが、
    //
    // fail-closed 契約（モジュールコメント参照）: 自動適用の根拠は
    // `gates.all_passed()`（3 ゲート全てが `Passed`）でなければならない。
    // ここまでに `any_failed()` は false と確定しているが、それだけでは
    // `Skipped` が混在している可能性を排除できない。`Skipped` は
    // 「実行されなかった」ことを意味し「合格した」ことを意味しないため、
    // `Passed` でないゲートが 1 つでもあれば自動適用してはならず、
    // エスカレーションへ回す。
    if !input.gates.all_passed() {
        return Ok(Decision {
            verdict: Verdict::Escalate,
            reasons: vec![Reason::GateSkipped],
            exclusion_rule_ids: input.exclusion_rule_ids.clone(),
        });
    }

    // 5. 自動適用: 上記いずれにも該当しない場合のみ。
    //
    // 判定根拠＝逸脱条件という意味論のため、自動適用時の reasons は空。
    // 表示文言「全ゲート green・全指標が閾値内」は呼び出し側が
    // `Verdict::AutoApply` から導出する。
    Ok(Decision {
        verdict: Verdict::AutoApply,
        reasons: Vec::new(),
        exclusion_rule_ids: input.exclusion_rule_ids.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{self, PresetName};

    /// テスト全体で使い回す既定プリセットの検証済み閾値
    /// （`lines_max=200`・`bench_max_pct=5.0`・`bench_runs=5`）。
    fn default_thresholds() -> Thresholds {
        config::Thresholds::builtin(PresetName::Default)
    }

    fn all_passed_gates() -> GateSignals {
        GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Passed,
            clippy: GateSignal::Passed,
        }
    }

    #[test]
    fn all_clean_yields_auto_apply() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::AutoApply);
        // 判定根拠＝逸脱条件の意味論に伴い、自動適用時は reasons が空になる
        // （表示文言は Verdict::as_ja 側）。
        assert!(decision.reasons().is_empty());
        assert!(decision.reason_conditions().is_empty());
        assert_eq!(decision.verdict().as_machine_id(), "auto_apply");
        assert_eq!(decision.verdict().as_ja(), "自動適用");
    }

    #[test]
    fn build_failed_alone_yields_reject() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let input = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Reject);
        assert_eq!(decision.verdict().as_machine_id(), "reject");
        assert_eq!(decision.reason_conditions(), vec!["gate_build_failed"]);
        // Skipped は理由に含めない契約。
        assert!(!decision.reason_conditions().contains(&"gate_test_failed"));
        assert!(!decision.reason_conditions().contains(&"gate_clippy_failed"));
    }

    #[test]
    fn test_failed_alone_yields_reject() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Failed,
            clippy: GateSignal::Skipped,
        };
        let input = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Reject);
    }

    #[test]
    fn clippy_failed_alone_yields_reject() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Passed,
            clippy: GateSignal::Failed,
        };
        let input = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Reject);
    }

    /// 却下優先の確定: build 失敗かつ行数超過・API 破壊が併発しても
    /// エスカレーションへは落ちず却下となること。
    #[test]
    fn reject_takes_priority_over_escalation_conditions() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let input = DecisionInput::new(
            &thresholds,
            thresholds.lines_max + 1000,
            gates,
            true,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Reject);
    }

    #[test]
    fn lines_exceeded_alone_yields_escalate() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            thresholds.lines_max + 1,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["lines_max_exceeded"]);
    }

    #[test]
    fn api_broken_alone_yields_escalate() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            true,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["api_broken"]);
    }

    #[test]
    fn gaming_suspect_alone_yields_escalate() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            true,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["gaming_suspect"]);
    }

    #[test]
    fn bench_exceeded_alone_yields_escalate() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::Measured {
                median_pct: thresholds.bench_median_max_pct + 0.01,
            },
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["bench_median_exceeded"]);
    }

    /// エスカレーション優先: 却下条件がなく複数逸脱が併発した場合、
    /// 判定は Escalate で確定し、理由は全逸脱のスーパーセットになる。
    #[test]
    fn multiple_escalation_conditions_are_all_listed() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            thresholds.lines_max + 1,
            all_passed_gates(),
            true,
            true,
            BenchSignal::Measured {
                median_pct: thresholds.bench_median_max_pct + 1.0,
            },
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(
            decision.reason_conditions(),
            vec![
                "lines_max_exceeded",
                "api_broken",
                "gaming_suspect",
                "bench_median_exceeded",
            ]
        );
    }

    #[test]
    fn bench_at_exact_threshold_is_within_limit() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::Measured {
                median_pct: thresholds.bench_median_max_pct,
            },
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::AutoApply);
    }

    #[test]
    fn bench_improvement_negative_is_auto_apply() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::Measured { median_pct: -3.0 },
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::AutoApply);
    }

    /// 矛盾入力の拒否: ゲートが全通過していないのに `Measured` が渡された場合、
    /// 判定を続行せず `InconsistentDecisionInput` で構築時に拒否する。
    #[test]
    fn inconsistent_input_with_failed_gate_and_measured_bench_is_rejected() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let err = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::Measured { median_pct: 1.0 },
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            GuardrailError::InconsistentDecisionInput { .. }
        ));
    }

    #[test]
    fn inconsistent_input_with_skipped_gate_and_measured_bench_is_rejected() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let err = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::Measured { median_pct: 1.0 },
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            GuardrailError::InconsistentDecisionInput { .. }
        ));
    }

    /// Skipped を含むゲート状態では、他の逸脱条件が一切なくとも自動適用には
    /// ならない（fail-closed 契約: 自動適用の前提条件は `all_passed()` であり、
    /// `any_failed()` が false というだけでは不十分）。
    #[test]
    fn skipped_gate_without_failure_does_not_force_reject_and_is_not_auto_applied() {
        let thresholds = default_thresholds();
        // build は Passed だが test が Skipped（実運用ではこの組み合わせは
        // 通常発生しないが、`any_failed` が false であることの確認として有効）。
        let gates = GateSignals {
            build: GateSignal::Passed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Passed,
        };
        let input = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗（NotRun なら Skipped でも許容される）");

        let decision = decide(&input).expect("判定に失敗");
        // Failed が無いため Reject ではないが、all_passed() でもないため
        // AutoApply にもならず Escalate へ回る。
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["gate_skipped"]);
    }

    /// NaN が閾値比較を素通りする回帰への防止テスト。`median_pct` が NaN の
    /// 場合、`>` 比較は常に false を返すため閾値超過判定を素通りしうるが、
    /// `is_finite()` チェックにより無条件でエスカレーションへ回ることを確認する。
    #[test]
    fn nan_bench_median_yields_escalate_not_auto_apply() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::Measured {
                median_pct: f64::NAN,
            },
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["bench_non_finite"]);
    }

    /// 正の無限大も同様に非有限としてエスカレーションへ回ることを確認する。
    #[test]
    fn positive_infinity_bench_median_yields_escalate() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::Measured {
                median_pct: f64::INFINITY,
            },
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reason_conditions(), vec!["bench_non_finite"]);
    }

    /// [`Reason::condition`]・[`std::fmt::Display`] の対応表が網羅されている
    /// ことの回帰テスト（受け入れ条件「出力に分岐種別と判定根拠（逸脱した
    /// 条件名）が含まれる」）。
    #[test]
    fn reason_condition_and_display_cover_every_variant() {
        let cases = vec![
            (Reason::GateFailed { gate: "build" }, "gate_build_failed"),
            (Reason::GateFailed { gate: "test" }, "gate_test_failed"),
            (Reason::GateFailed { gate: "clippy" }, "gate_clippy_failed"),
            (Reason::GateSkipped, "gate_skipped"),
            (
                Reason::ExclusionMatch {
                    rule_id: "arch-hyperparameter-change".to_string(),
                },
                "policy_exclusion_match",
            ),
            (
                Reason::LinesMaxExceeded {
                    lines_changed: 250,
                    lines_max: 200,
                },
                "lines_max_exceeded",
            ),
            (Reason::ApiBroken, "api_broken"),
            (Reason::GamingSuspect, "gaming_suspect"),
            (
                Reason::BenchMedianExceeded {
                    median_pct: 6.0,
                    bench_max_pct: 5.0,
                },
                "bench_median_exceeded",
            ),
            (
                Reason::BenchNonFinite {
                    median_pct: f64::NAN,
                },
                "bench_non_finite",
            ),
        ];

        for (reason, expected_condition) in cases {
            assert_eq!(reason.condition(), expected_condition);
            // Display は判定根拠の人間可読文言（テキスト出力用）。空文字列や
            // プレースホルダのままではないことを最低限確認する。
            assert!(!reason.to_string().is_empty());
        }
    }

    /// [`Verdict::as_machine_id`] / [`Verdict::as_ja`] の網羅テスト
    /// （CI・自己修復ループが分岐に使う機械可読 ID の確定確認）。
    #[test]
    fn verdict_machine_id_and_ja_cover_every_variant() {
        assert_eq!(Verdict::AutoApply.as_machine_id(), "auto_apply");
        assert_eq!(Verdict::Escalate.as_machine_id(), "escalate");
        assert_eq!(Verdict::Reject.as_machine_id(), "reject");
        assert_eq!(Verdict::AutoApply.as_ja(), "自動適用");
        assert_eq!(Verdict::Escalate.as_ja(), "エスカレーション");
        assert_eq!(Verdict::Reject.as_ja(), "却下");
    }

    // --- ポリシー除外リスト match の無条件エスカレーション統合 ---

    /// 受け入れ条件 1: 全ゲート通過・全指標が閾値内（機械判定は AutoApply 相当）
    /// でも、`exclusion_rule_ids` が非空なら無条件で `Escalate` になり、
    /// `reasons` にルール `id` が反映されること（機械判定より除外リスト match
    /// が優先される）。
    #[test]
    fn exclusion_match_yields_escalate_even_when_all_signals_clean() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            vec!["arch-hyperparameter-change".to_string()],
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert!(
            decision
                .reasons()
                .iter()
                .any(|r| r.to_string().contains("arch-hyperparameter-change"))
        );
        assert_eq!(decision.reason_conditions(), vec!["policy_exclusion_match"]);
        assert_eq!(
            decision.exclusion_rule_ids(),
            &["arch-hyperparameter-change".to_string()]
        );
    }

    /// 受け入れ条件 2 の裏面: 却下（ゲート失敗）と除外リスト match が同時
    /// 成立する場合、判定順序契約 1. > 2. により `Reject` が確定する
    /// （却下優先）。ただし match した事実自体は `Decision` に記録され続ける
    /// （`exclusion_rule_ids()` が失われない。security.md「取り込み判断の
    /// 根拠を追跡可能にする」）。
    #[test]
    fn reject_takes_priority_but_still_records_exclusion_match() {
        let thresholds = default_thresholds();
        let gates = GateSignals {
            build: GateSignal::Failed,
            test: GateSignal::Skipped,
            clippy: GateSignal::Skipped,
        };
        let input = DecisionInput::new(
            &thresholds,
            10,
            gates,
            false,
            false,
            BenchSignal::NotRun,
            vec!["test-tolerance-loosening".to_string()],
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Reject);
        assert!(
            decision
                .reasons()
                .iter()
                .any(|r| r.to_string().contains("build"))
        );
        assert_eq!(
            decision.exclusion_rule_ids(),
            &["test-tolerance-loosening".to_string()]
        );
    }

    /// 受け入れ条件 2（既存 3 分岐判定テストとの後方互換）: `exclusion_rule_ids`
    /// が空の場合、判定結果は本統合前の挙動と完全に同一になること
    /// （`all_clean_yields_auto_apply` と同一シナリオでの二重確認。
    /// `exclusion_rule_ids()` は必ず空を返す）。
    #[test]
    fn empty_exclusion_rule_ids_does_not_affect_verdict() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            Vec::new(),
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::AutoApply);
        assert!(decision.exclusion_rule_ids().is_empty());
    }

    /// 複数ルールが同時に match した場合、全ルール `id` が理由・記録の両方に
    /// 反映されること（1 件のみを代表させて後続ルールを黙って捨てない）。
    #[test]
    fn multiple_exclusion_matches_are_all_recorded() {
        let thresholds = default_thresholds();
        let input = DecisionInput::new(
            &thresholds,
            10,
            all_passed_gates(),
            false,
            false,
            BenchSignal::NotRun,
            vec![
                "arch-hyperparameter-change".to_string(),
                "test-tolerance-loosening".to_string(),
            ],
        )
        .expect("矛盾なし入力の構築に失敗");

        let decision = decide(&input).expect("判定に失敗");
        assert_eq!(decision.verdict(), Verdict::Escalate);
        assert_eq!(decision.reasons().len(), 2);
        assert_eq!(decision.exclusion_rule_ids().len(), 2);
    }
}
