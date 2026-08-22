//! ループ試行の構造化ログ出力機構（TASK-3.4・イシュー #145・REQ-3）。
//!
//! [`crate::report::LoopReport`] / [`crate::report::LoopFailure`]
//! （`runner::SelfRepairLoop::run` が返す「入力となる seam」。
//! `report.rs` のモジュールコメント参照）を受け取り、ループ各段階
//! （検出開始 → 検出結果 → 各試行 → 最終結論／失敗）を JSON Lines
//! （1 行 1 レコード）として追記専用ファイルへ書き出す。
//!
//! `.claude/rules/security.md`「ループ試行ログは改竄検知可能な形式で記録し、
//! 取り込み判断の根拠を追跡可能にする」に対応するため、各レコードは直前
//! レコードのハッシュを含む SHA-256 ハッシュチェーンを構成する
//! （[`verify_chain`] で改変・削除・並べ替えを検知できる）。
//!
//! v1（`Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/logging.rs`。
//! TASK-3.2-S1・イシュー #47）からの移植（`docs/spec/v1-assets-inventory.md`
//! L17「改修して再利用」判定）。v1 は `sha2`（`Sha256`）と `thiserror` に
//! 依存するが、いずれも v2 の許容依存 8 区分
//! （`.claude/rules/deps-policy.md`）に含まれず依存追加はユーザー承認事項の
//! ため、`sha2` は [`crate::sha256`]（本クレート内の自作 FIPS 180-4 実装。
//! `crates/guardrail/src/toml_lite.rs` が `toml` クレートを自作パーサで
//! 代替したのと同じ方針）へ、`thiserror::Error` 派生は `crates/guardrail/src/error.rs`・
//! `crate::error::SelfRepairError` と同じ手書き `Display`/`Error` 実装へ
//! 差し替える。ハッシュ計算式・`DOMAIN_PREFIX`・genesis 値・JSON Lines
//! フォーマット自体は v1 と同一に保つ（`docs/self-repair-log-format.md` が
//! 転記元）。
//!
//! ログフォーマットの詳細仕様書化・監査手順・外部アンカー運用の文書化は
//! `docs/self-repair-log-format.md` が担う。CLI バイナリ
//! （`self-repair run`/`verify-log`。`docs/guardrail-self-repair-cli.md` 3 節）
//! の実装は本イシューのスコープ外（`.claude/rules/out-of-scope-tracking.md`）。

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write as _};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::kind::RepairKind;
use crate::outcome::LoopOutcome;
use crate::report::{AttemptOutcome, AttemptRecord, LoopFailure, LoopReport};
use crate::sha256::{sha256, to_hex};

/// ハッシュチェーンのドメイン分離プレフィックス。
///
/// `hash = SHA-256(DOMAIN_PREFIX || seq(LE8) || 0x00 || recorded_at_unix_ms(LE16)
/// || 0x00 || stage || 0x00 || prev_hash || canonical_payload_json)`
/// （[`compute_hash`] のドキュメンテーションコメント・
/// `docs/self-repair-log-format.md` 参照）の形でハッシュ計算に混入させ、
/// 本ログ以外の用途で計算された SHA-256 値を「有効な直前ハッシュ」として
/// 誤って受理する衝突・流用を防ぐ（security.md A08: 改竄検知の実効性を
/// 高める）。値自体は v1 と同一（`v1` はログフォーマットの版数であり
/// リポジトリ世代ではないため、移植にあたり変更しない）。
const DOMAIN_PREFIX: &str = "rust-ai-library/self-repair/log/v1";

/// 先頭レコードの `prev_hash` に使うジェネシス値。
///
/// 固定文字列 `"self-repair-log-v1"` の SHA-256（16 進小文字）。ログファイルが
/// 空の状態から最初のレコードを書く際、「直前レコードが存在しない」ことを
/// 検証可能な形で表現するために使う（`None` や空文字列だと改竄側が
/// 同じ値を用意しやすく検知力が弱まるため、ハッシュ値で固定する）。
/// この既知値（`be26b2311e026d01ceabc4dc7b360f8583ee203c4e4a858aef631c698c4b1a28`。
/// [`crate::sha256`] のテストにも同じ値を持つ）が v1 と一致することが、
/// 自作 SHA-256 実装が v1 の `sha2` 出力と互換であることの実証になる。
fn genesis_hash() -> String {
    to_hex(&sha256(b"self-repair-log-v1"))
}

/// ループの各段階を表す判別子（[`LogRecord::stage`]）。
///
/// `_ =>` を使わず全 variant を明示する方針（`report.rs`・`outcome.rs` と
/// 同じ fail-closed 設計）に揃え、[`stage_name`] で機械可読な文字列へ写像する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    LoopStart,
    Detection,
    Attempt,
    LoopOutcome,
    LoopFailure,
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::LoopStart => "loop_start",
        Stage::Detection => "detection",
        Stage::Attempt => "attempt",
        Stage::LoopOutcome => "loop_outcome",
        Stage::LoopFailure => "loop_failure",
    }
}

/// JSON Lines 1 行分のレコード（追記専用ログの最小単位）。
///
/// `payload` は段階ごとに内容が異なるため `serde_json::Value` で保持する
/// （段階別に別の Rust 型を要求すると [`verify_chain`] の読み戻し処理が
/// 段階ごとの分岐を必要とし、fail-closed な一括検証が複雑化するため）。
///
/// `deny_unknown_fields` を付ける（トップレベルの未知フィールド注入で
/// [`verify_chain`] の検証をすり抜けさせない。ハッシュは既知フィールドのみを
/// 対象に再計算するため、`deny_unknown_fields` がないと JSON 行に未知の
/// プロパティを追加してもチェーン検証が成功してしまい、生の JSON リーダーに
/// とって誤解を招く監査ログを許してしまう。security.md A03/A08・fail-closed
/// 方針に対応）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogRecord {
    /// 0 始まりの連番。並べ替え・欠番を検知するための補助情報
    /// （ハッシュチェーン自体が本質的な改竄検知手段であり、`seq` は
    /// 人間が目視で連続性を確認するための冗長な補助にすぎない）。
    seq: u64,
    /// レコード書き込み時刻（UNIX ミリ秒）。
    recorded_at_unix_ms: u128,
    /// この段階の識別子（[`stage_name`] が返す機械可読文字列）。
    stage: String,
    /// 段階別の内容（下記 `*_payload` 群が構築する）。
    payload: serde_json::Value,
    /// 直前レコードの `hash`（先頭レコードは [`genesis_hash`]）。
    prev_hash: String,
    /// `DOMAIN_PREFIX || seq || recorded_at_unix_ms || stage || prev_hash ||
    /// canonical_payload_json` の SHA-256（[`compute_hash`] 参照）。
    hash: String,
}

/// ログ入出力で発生しうる失敗。
///
/// `.claude/rules/coding-rust.md`・`.claude/rules/security.md` の方針により、
/// 本番経路（[`LogWriter`] / [`verify_chain`]）は `unwrap()` / `expect()` を
/// 使わず、失敗理由をここに定義した variant で呼び出し元へ返す
/// （fail-closed。書き込み失敗を黙殺しない）。`thiserror` は許容依存 8 区分
/// 外のため、`crate::error::SelfRepairError` と同じ手書き `Display`/`Error`
/// 実装で代替する。
#[derive(Debug)]
pub enum LogError {
    /// ログファイルの追記オープン・書き込み・読み戻しでの I/O 失敗。
    Io {
        path: String,
        source: std::io::Error,
    },
    /// レコードの JSON シリアライズ・デシリアライズ失敗。
    Serialization(serde_json::Error),
    /// [`verify_chain`] がハッシュ不一致・欠落を検出した（改竄・破損の疑い）。
    ChainViolation { seq: u64, reason: String },
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::Io { path, source } => {
                write!(
                    f,
                    "ログファイル I/O でエラーが発生しました（path={path}）: {source}"
                )
            }
            LogError::Serialization(source) => {
                write!(
                    f,
                    "ログレコードの JSON 変換でエラーが発生しました: {source}"
                )
            }
            LogError::ChainViolation { seq, reason } => {
                write!(
                    f,
                    "ログの改竄またはレコード欠落を検知しました（seq={seq}）: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for LogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LogError::Io { source, .. } => Some(source),
            LogError::Serialization(source) => Some(source),
            LogError::ChainViolation { .. } => None,
        }
    }
}

/// 1 レコードの `hash` を計算する（[`LogWriter::append_stages`] /
/// [`verify_chain`] の双方から呼ばれ、書き込み時と検証時で同一の計算式を
/// 保証する）。
///
/// `seq`・`recorded_at_unix_ms`・`stage` も `payload` と同様にハッシュへ
/// 混入させる（`LogRecord` の全フィールドを対象に含めないと、`payload` を
/// 変えずに `stage` だけ書き換える改竄――例えば `loop_outcome` を
/// `detection` に偽装する――を [`verify_chain`] が見逃す。この 3 値は
/// フィールド境界に `\0` セパレータを挟んで結合し、`"a"+"bc"` と `"ab"+"c"`
/// のような異なる値の組が同一バイト列に潰れて衝突する余地を防ぐ）。
///
/// `payload` は [`serde_json::to_vec`] のキー順序（`serde_json::Value` の
/// 内部表現に従う。`preserve_order` feature を有効化していないため
/// アルファベット順で安定する）でシリアライズし、同一 payload からは常に
/// 同一バイト列が得られることを前提にする。
fn compute_hash(
    seq: u64,
    recorded_at_unix_ms: u128,
    stage: &str,
    prev_hash: &str,
    payload: &serde_json::Value,
) -> Result<String, LogError> {
    let canonical = serde_json::to_vec(payload).map_err(LogError::Serialization)?;
    let mut buf = Vec::with_capacity(
        DOMAIN_PREFIX.len() + 8 + 1 + 16 + 1 + stage.len() + 1 + prev_hash.len() + canonical.len(),
    );
    buf.extend_from_slice(DOMAIN_PREFIX.as_bytes());
    buf.extend_from_slice(&seq.to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(&recorded_at_unix_ms.to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(stage.as_bytes());
    buf.push(0);
    buf.extend_from_slice(prev_hash.as_bytes());
    buf.extend_from_slice(&canonical);
    Ok(to_hex(&sha256(&buf)))
}

/// [`AttemptOutcome`] をログ用 JSON へ写像する。
///
/// ドメイン型（`report.rs`）へ serde derive を追加せず、ログ用の変換を
/// ここに閉じる（`report.rs`・`outcome.rs` は本イシューのスコープ外＝編集禁止
/// 対象ではないが、ログ表現の都合でドメイン型の形を歪めない方針を維持する）。
/// `_ =>` は使わず全 variant を明示する（受け入れ基準「`AttemptOutcome` 全 5
/// variant の判断根拠がログから復元できる」に対応）。
fn attempt_outcome_payload(outcome: &AttemptOutcome) -> serde_json::Value {
    match outcome {
        AttemptOutcome::VerificationFailed { reason } => serde_json::json!({
            "kind": "verification_failed",
            "reason": reason,
        }),
        AttemptOutcome::AdoptionRejectedRetryable { reason } => serde_json::json!({
            "kind": "adoption_rejected_retryable",
            "reason": reason,
        }),
        AttemptOutcome::Adopted => serde_json::json!({
            "kind": "adopted",
        }),
        AttemptOutcome::Escalated { reason } => serde_json::json!({
            "kind": "escalated",
            "reason": reason,
        }),
        AttemptOutcome::RejectedFinal { reason } => serde_json::json!({
            "kind": "rejected_final",
            "reason": reason,
        }),
    }
}

/// [`LoopOutcome`] をログ用 JSON へ写像する（`attempt_outcome_payload` と同じ
/// 理由でドメイン型に derive を追加せずここで変換する）。
fn loop_outcome_payload(outcome: &LoopOutcome) -> serde_json::Value {
    match outcome {
        LoopOutcome::NoActionNeeded => serde_json::json!({ "kind": "no_action_needed" }),
        LoopOutcome::Adopted => serde_json::json!({ "kind": "adopted" }),
        LoopOutcome::Escalated { reason } => serde_json::json!({
            "kind": "escalated",
            "reason": reason,
        }),
        LoopOutcome::Rejected { stage, reason } => serde_json::json!({
            "kind": "rejected",
            "stage": stage,
            "reason": reason,
        }),
        LoopOutcome::Exhausted => serde_json::json!({ "kind": "exhausted" }),
    }
}

fn repair_kind_payload(kind: RepairKind) -> serde_json::Value {
    serde_json::json!({ "kind": kind.as_machine_id() })
}

fn attempt_record_payload(record: &AttemptRecord) -> serde_json::Value {
    serde_json::json!({
        "attempt": record.attempt,
        "duration_ms": record.duration.as_millis(),
        "outcome": attempt_outcome_payload(&record.outcome),
    })
}

/// [`LoopReport`] / [`LoopFailure`] から「ループ各段階」のレコード列を
/// （`stage`, `payload`) の組として導出する。
///
/// [`LogWriter::append_report`] / [`LogWriter::append_failure`] が本関数の
/// 出力をハッシュチェーンへ連結する。段階順序は
/// `loop_start → detection → attempt × n → loop_outcome/loop_failure` に従う
/// （実装計画セクション 3）。
fn stages_for_report(report: &LoopReport) -> Vec<(Stage, serde_json::Value)> {
    let mut stages = vec![
        (Stage::LoopStart, repair_kind_payload(report.kind)),
        (
            Stage::Detection,
            serde_json::json!({
                "action_needed": !matches!(report.outcome, LoopOutcome::NoActionNeeded),
            }),
        ),
    ];
    for attempt in &report.attempts {
        stages.push((Stage::Attempt, attempt_record_payload(attempt)));
    }
    stages.push((
        Stage::LoopOutcome,
        serde_json::json!({
            "outcome": loop_outcome_payload(&report.outcome),
            "total_duration_ms": report.total_duration.as_millis(),
            "attempt_count": report.attempt_count(),
        }),
    ));
    stages
}

/// [`LoopFailure`] 用の段階列（`loop_start`/`detection` を省き、それまでの
/// 試行と失敗理由のみを記録する）。
///
/// `LoopFailure` は「段階の実行自体が失敗した」ケース（`report.rs` 参照）
/// であり、`kind` を保持しないため `loop_start`/`detection` は出力しない
/// （存在しない情報を偽装しない。security.md A05 の fail-closed 方針）。
fn stages_for_failure(failure: &LoopFailure) -> Vec<(Stage, serde_json::Value)> {
    let mut stages = Vec::new();
    for attempt in &failure.attempts {
        stages.push((Stage::Attempt, attempt_record_payload(attempt)));
    }
    stages.push((
        Stage::LoopFailure,
        serde_json::json!({
            "error": failure.error.to_string(),
            "attempt_count": failure.attempts.len(),
        }),
    ));
    stages
}

fn now_unix_ms() -> u128 {
    // `SystemTime::now()` は `UNIX_EPOCH` 以降である前提が崩れる状況
    // （システム時計がエポック以前に設定されている等）は本ツールの
    // 通常運用では想定しないが、`unwrap` は使わず 0 にフォールバックする
    // （`.claude/rules/coding-rust.md`: 本番経路で unwrap/expect を使わない）。
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// 追記専用のループ試行ログライタ。
///
/// [`crate::runner::SelfRepairLoop::run`] の呼び出し元（CLI バイナリ想定。
/// `docs/guardrail-self-repair-cli.md` 3 節）が、得られた [`LoopReport`] /
/// [`LoopFailure`] をそのまま渡して永続化するために使う。ファイルは
/// 既存内容を保持したまま追記するため、複数回のループ実行を同一ログへ
/// 連続して積み重ねられる（[`LogWriter::open`] のたびに直前レコードの
/// `hash` を読み戻し、チェーンを継続する）。
pub struct LogWriter {
    path: std::path::PathBuf,
    next_seq: u64,
    last_hash: String,
}

impl LogWriter {
    /// `path` を追記専用で開く（存在しなければ新規作成）。既存レコードが
    /// あれば末尾まで読み、`next_seq` / `last_hash` を復元する。
    ///
    /// 既存ログの整合性チェックは行わない（意図的に検証しない。書き込み
    /// 継続のためだけに末尾ハッシュが必要なため）。整合性を確認したい
    /// 場合は事前に [`verify_chain`] を呼ぶこと。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LogError> {
        let path = path.as_ref().to_path_buf();
        let (next_seq, last_hash) = match std::fs::File::open(&path) {
            Ok(file) => read_tail_state(&path, file)?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => (0, genesis_hash()),
            Err(err) => {
                return Err(LogError::Io {
                    path: path.display().to_string(),
                    source: err,
                });
            }
        };
        Ok(LogWriter {
            path,
            next_seq,
            last_hash,
        })
    }

    /// [`LoopReport`]（正常終了）由来の段階レコード列を追記する。
    pub fn append_report(&mut self, report: &LoopReport) -> Result<(), LogError> {
        self.append_stages(stages_for_report(report))
    }

    /// [`LoopFailure`]（段階の実行自体が失敗）由来の段階レコード列を追記する。
    pub fn append_failure(&mut self, failure: &LoopFailure) -> Result<(), LogError> {
        self.append_stages(stages_for_failure(failure))
    }

    fn append_stages(&mut self, stages: Vec<(Stage, serde_json::Value)>) -> Result<(), LogError> {
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .map_err(|source| LogError::Io {
                path: self.path.display().to_string(),
                source,
            })?;
        for (stage, payload) in stages {
            let recorded_at_unix_ms = now_unix_ms();
            let stage_name = stage_name(stage);
            let hash = compute_hash(
                self.next_seq,
                recorded_at_unix_ms,
                stage_name,
                &self.last_hash,
                &payload,
            )?;
            let record = LogRecord {
                seq: self.next_seq,
                recorded_at_unix_ms,
                stage: stage_name.to_string(),
                payload,
                prev_hash: self.last_hash.clone(),
                hash: hash.clone(),
            };
            let mut line = serde_json::to_vec(&record).map_err(LogError::Serialization)?;
            line.push(b'\n');
            file.write_all(&line).map_err(|source| LogError::Io {
                path: self.path.display().to_string(),
                source,
            })?;
            self.next_seq += 1;
            self.last_hash = hash;
        }
        file.flush().map_err(|source| LogError::Io {
            path: self.path.display().to_string(),
            source,
        })
    }
}

/// 既存ログを末尾まで読み、次に書くべき `seq` と直前 `hash` を復元する。
///
/// [`LogWriter::open`] が既存ファイルを引き継ぐ際に使う。パース不能な行が
/// あれば [`LogError::Serialization`] として fail-closed に扱う（黙って
/// 読み飛ばさない。改行途中で壊れたログを正常とみなさないため）。
fn read_tail_state(path: &Path, file: std::fs::File) -> Result<(u64, String), LogError> {
    let reader = BufReader::new(file);
    let mut next_seq = 0u64;
    let mut last_hash = genesis_hash();
    for line in reader.lines() {
        let line = line.map_err(|source| LogError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record: LogRecord = serde_json::from_str(&line).map_err(LogError::Serialization)?;
        next_seq = record.seq + 1;
        last_hash = record.hash;
    }
    Ok((next_seq, last_hash))
}

/// [`verify_chain`] の検証結果サマリ。
///
/// CLI（`self-repair verify-log`）が成功メッセージに含めることで、監査担当者が
/// 外部アンカー運用（`docs/self-repair-log-format.md` §7）の記録値と突合できる
/// ようにする（Review #145 指摘対応。`.claude/rules/security.md` A08「ループ
/// 試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を追跡可能にする」の
/// 意図に沿い、無条件の「整合性を確認しました」だけで終わらせない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyChainSummary {
    /// 検証で走査した有効レコード件数（空行は数えない）。
    pub record_count: u64,
    /// 最終レコードの `seq`。レコードが 1 件もない場合は `None`
    /// （`record_count == 0` と等価。呼び出し元が「空ログ」を区別するための
    /// フィールド）。
    pub last_seq: Option<u64>,
    /// 最終レコードの `hash`（16 進文字列）。レコードが 1 件もない場合は
    /// チェーン起点ハッシュ（`genesis_hash`）のままとなる。
    pub last_hash: String,
}

/// ログファイル全体のハッシュチェーンを再計算し、改竄（フィールド改変・
/// レコード削除・順序入れ替え）を検知する。
///
/// 1 箇所でも不一致・パース不能なレコードがあれば直ちに `Err` を返す
/// （fail-closed。部分的に正しい範囲だけを認める緩い検証はしない。
/// security.md A08「判定の迂回経路を作らない」と同じ思想）。成功時は
/// [`VerifyChainSummary`]（レコード件数・最終 `seq`・最終 `hash`）を返し、
/// 呼び出し元（CLI 等）が突合材料として提示できるようにする。
///
/// 末尾切り詰め（ファイル末尾の正当なレコードをまるごと削除する改竄）は、
/// 削除後の最終レコードまでのチェーンが自己無矛盾になってしまうため本関数
/// 単体では検知できない（レコード 0 件のファイル・切り詰め後のファイルの
/// どちらも `Ok` を返しうる）。検知には別途「最終 `hash` の外部アンカー」
/// （書き込み直後に別経路へ記録する等）が必要であり、その運用は
/// `docs/self-repair-log-format.md` の外部アンカー運用節に委ねる。
pub fn verify_chain(path: impl AsRef<Path>) -> Result<VerifyChainSummary, LogError> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|source| LogError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut expected_seq = 0u64;
    let mut expected_prev_hash = genesis_hash();
    let mut record_count = 0u64;
    let mut last_seq: Option<u64> = None;
    for line in reader.lines() {
        let line = line.map_err(|source| LogError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record: LogRecord = serde_json::from_str(&line).map_err(LogError::Serialization)?;
        if record.seq != expected_seq {
            return Err(LogError::ChainViolation {
                seq: record.seq,
                reason: format!(
                    "seq の連続性が崩れています（期待値={expected_seq}、実際={}）",
                    record.seq
                ),
            });
        }
        if record.prev_hash != expected_prev_hash {
            return Err(LogError::ChainViolation {
                seq: record.seq,
                reason: "prev_hash が直前レコードのハッシュと一致しません".to_string(),
            });
        }
        let recomputed = compute_hash(
            record.seq,
            record.recorded_at_unix_ms,
            &record.stage,
            &record.prev_hash,
            &record.payload,
        )?;
        if recomputed != record.hash {
            return Err(LogError::ChainViolation {
                seq: record.seq,
                reason: "レコードからの再計算ハッシュが記録済み hash と一致しません（seq/recorded_at_unix_ms/stage/payload いずれかの改竄の疑い）"
                    .to_string(),
            });
        }
        record_count += 1;
        last_seq = Some(record.seq);
        expected_seq += 1;
        expected_prev_hash = record.hash;
    }
    Ok(VerifyChainSummary {
        record_count,
        last_seq,
        last_hash: expected_prev_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SelfRepairError;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    /// テストごとに衝突しない一時ファイルパスを作る。`tempfile` クレート
    /// （許容依存 8 区分外）を使わず、`crate::test_support::unique_temp_dir`
    /// と同じ `std::env::temp_dir()` + プロセス ID + 単調増加カウンタ方式で
    /// 代替する（実装計画セクション 2。同モジュールを再利用しない理由は
    /// `test_support` がディレクトリ単位のヘルパーであり、本テストが必要な
    /// のは単一ファイルパスのみのため）。
    fn unique_log_path(test_name: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "self-repair-logging-test-{}-{test_name}-{seq}.jsonl",
            std::process::id()
        ))
    }

    fn sample_report(outcome: LoopOutcome, attempts: Vec<AttemptRecord>) -> LoopReport {
        LoopReport {
            kind: RepairKind::BugFix,
            outcome,
            attempts,
            total_duration: Duration::from_millis(123),
        }
    }

    fn attempt(n: u32, outcome: AttemptOutcome) -> AttemptRecord {
        AttemptRecord {
            attempt: n,
            duration: Duration::from_millis(10 * u64::from(n)),
            outcome,
        }
    }

    /// genesis 値が v1 実装（`sha2` クレート採用）と同一の既知値を返すこと。
    /// 自作 SHA-256（`crate::sha256`）が v1 と互換であることの実証。
    #[test]
    fn genesis_hash_matches_v1_known_value() {
        assert_eq!(
            genesis_hash(),
            "be26b2311e026d01ceabc4dc7b360f8583ee203c4e4a858aef631c698c4b1a28"
        );
    }

    /// 受け入れ条件: 採用（Adopted）ケースで段階レコード列
    /// （loop_start → detection → attempt → loop_outcome）が出力される。
    #[test]
    fn append_report_adopted_writes_expected_stage_sequence() {
        let log_path = unique_log_path("adopted_stage_sequence");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );

        let mut writer = LogWriter::open(&log_path).expect("新規ログを開けること");
        writer
            .append_report(&report)
            .expect("正常ケースの追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("ログを読めること");
        let stages: Vec<String> = content
            .lines()
            .map(|line| {
                let record: LogRecord =
                    serde_json::from_str(line).expect("各行が JSON として読めること");
                record.stage
            })
            .collect();
        assert_eq!(
            stages,
            vec!["loop_start", "detection", "attempt", "loop_outcome"]
        );
        verify_chain(&log_path).expect("直後の verify_chain が成功すること");
        let _ = std::fs::remove_file(&log_path);
    }

    /// [`VerifyChainSummary`] がレコード件数・最終 `seq`・最終 `hash` を
    /// 正しく報告すること（Review #145 指摘対応: CLI 成功メッセージが
    /// これらの値を突合材料として提示できるようにするための実測根拠）。
    #[test]
    fn verify_chain_summary_reports_record_count_and_last_seq_hash() {
        let log_path = unique_log_path("summary_record_count_and_last_seq_hash");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );
        LogWriter::open(&log_path)
            .expect("新規ログを開けること")
            .append_report(&report)
            .expect("正常ケースの追記に失敗しないこと");

        // loop_start → detection → attempt → loop_outcome の 4 レコード
        // （このヘルパーが生成する段階列。上のテストで実測済み）。
        let content = std::fs::read_to_string(&log_path).expect("ログを読めること");
        let last_line = content
            .lines()
            .next_back()
            .expect("少なくとも 1 行はあること");
        let last_record: LogRecord =
            serde_json::from_str(last_line).expect("最終行が JSON として読めること");

        let summary = verify_chain(&log_path).expect("verify_chain が成功すること");
        assert_eq!(summary.record_count, 4);
        assert_eq!(summary.last_seq, Some(last_record.seq));
        assert_eq!(summary.last_hash, last_record.hash);
        let _ = std::fs::remove_file(&log_path);
    }

    /// 空（0 バイト）ログに対する `verify_chain` は `Err` にはならないが
    /// （fail-closed の対象は「壊れている」ことが判定できる場合のみ）、
    /// `record_count == 0`・`last_seq == None`・`last_hash == genesis_hash()`
    /// を返すこと。CLI 側（`main.rs::run_verify_log`）はこの値を見て、既定では
    /// fail-closed に exit 1 とし、`--allow-empty-log` 明示指定時のみ `OK:` では
    /// なく `WARN:` メッセージ付きで exit 0 に切り替える（PR #356 codex-review
    /// P1 指摘対応。当初〈Review #145〉は無条件 exit 0 だったが、終了コードのみ
    /// 見る監査自動化がログ全削除による改竄を見逃す経路だったため変更した）。
    #[test]
    fn verify_chain_on_empty_file_returns_zero_record_summary() {
        let log_path = unique_log_path("empty_file_zero_record_summary");
        std::fs::write(&log_path, b"").expect("空ファイルを作成できること");

        let summary = verify_chain(&log_path).expect("空ログはチェーン違反ではなく Ok を返すこと");
        assert_eq!(summary.record_count, 0);
        assert_eq!(summary.last_seq, None);
        assert_eq!(summary.last_hash, genesis_hash());
        let _ = std::fs::remove_file(&log_path);
    }

    /// 却下・エスカレーション・NoActionNeeded の各ケースでも段階列が導出され、
    /// チェーン検証が通ることを確認する。
    #[test]
    fn append_report_covers_rejected_escalated_no_action_needed() {
        let rejected_path = unique_log_path("rejected");
        let rejected = sample_report(
            LoopOutcome::Rejected {
                stage: "verification",
                reason: "再現しない".to_string(),
            },
            vec![attempt(
                1,
                AttemptOutcome::RejectedFinal {
                    reason: "再現しない".to_string(),
                },
            )],
        );
        LogWriter::open(&rejected_path)
            .expect("開けること")
            .append_report(&rejected)
            .expect("却下ケースの追記に失敗しないこと");
        verify_chain(&rejected_path).expect("却下ケースの検証が成功すること");
        let _ = std::fs::remove_file(&rejected_path);

        let escalated_path = unique_log_path("escalated");
        let escalated = sample_report(
            LoopOutcome::Escalated {
                reason: "人間レビューへ回す".to_string(),
            },
            vec![attempt(
                1,
                AttemptOutcome::Escalated {
                    reason: "人間レビューへ回す".to_string(),
                },
            )],
        );
        LogWriter::open(&escalated_path)
            .expect("開けること")
            .append_report(&escalated)
            .expect("エスカレーションケースの追記に失敗しないこと");
        verify_chain(&escalated_path).expect("エスカレーションケースの検証が成功すること");
        let _ = std::fs::remove_file(&escalated_path);

        let no_action_path = unique_log_path("no_action");
        let no_action = sample_report(LoopOutcome::NoActionNeeded, vec![]);
        LogWriter::open(&no_action_path)
            .expect("開けること")
            .append_report(&no_action)
            .expect("NoActionNeeded ケースの追記に失敗しないこと");
        verify_chain(&no_action_path).expect("NoActionNeeded ケースの検証が成功すること");
        let _ = std::fs::remove_file(&no_action_path);
    }

    /// `LoopFailure` 由来のログでも attempt → loop_failure の段階列が出力され、
    /// チェーン検証が通る。
    #[test]
    fn append_failure_writes_attempts_then_loop_failure_stage() {
        let log_path = unique_log_path("failure_stage_sequence");
        let failure = LoopFailure {
            error: SelfRepairError::Verification {
                attempt: 2,
                reason: "gate 実行に失敗".to_string(),
            },
            attempts: vec![attempt(
                1,
                AttemptOutcome::VerificationFailed {
                    reason: "初回不合格".to_string(),
                },
            )],
        };

        let mut writer = LogWriter::open(&log_path).expect("新規ログを開けること");
        writer
            .append_failure(&failure)
            .expect("失敗ケースの追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("ログを読めること");
        let stages: Vec<String> = content
            .lines()
            .map(|line| {
                let record: LogRecord =
                    serde_json::from_str(line).expect("各行が JSON として読めること");
                record.stage
            })
            .collect();
        assert_eq!(stages, vec!["attempt", "loop_failure"]);
        verify_chain(&log_path).expect("失敗ケースの検証が成功すること");
        let _ = std::fs::remove_file(&log_path);
    }

    /// `stage` フィールドの改変（`payload` はそのまま）を検知する
    /// （改竄側が `payload` を変えず `stage` だけ書き換えて監査記録の意味を
    /// 偽装する経路――例えば「取り込み確定」の記録を「単なる検出メモ」に
    /// 見せかける――を、`compute_hash` が `stage` もハッシュへ混入させる
    /// ことで塞いでいることを確認する）。
    #[test]
    fn verify_chain_detects_stage_field_tampering() {
        let log_path = unique_log_path("stage_tampering");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let tampered = content.replacen("\"stage\":\"loop_outcome\"", "\"stage\":\"detection\"", 1);
        assert_ne!(content, tampered, "置換対象が実在すること");
        std::fs::write(&log_path, tampered).expect("改変後の内容を書き戻せること");

        let result = verify_chain(&log_path);
        assert!(
            matches!(result, Err(LogError::ChainViolation { .. })),
            "stage のみの改変（payload 不変）が ChainViolation として検知されること"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// `recorded_at_unix_ms` フィールドの改変を検知する（書き込み時刻の
    /// 偽装で「いつ判断されたか」の監査証跡を崩す経路を塞ぐ）。
    #[test]
    fn verify_chain_detects_recorded_at_field_tampering() {
        let log_path = unique_log_path("recorded_at_tampering");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let first_line = content.lines().next().expect("最低 1 行は存在すること");
        let first_record: LogRecord =
            serde_json::from_str(first_line).expect("先頭行がパースできること");
        let forged_timestamp = first_record.recorded_at_unix_ms + 1;
        let tampered = content.replacen(
            &format!(
                "\"recorded_at_unix_ms\":{}",
                first_record.recorded_at_unix_ms
            ),
            &format!("\"recorded_at_unix_ms\":{forged_timestamp}"),
            1,
        );
        assert_ne!(content, tampered, "置換対象が実在すること");
        std::fs::write(&log_path, tampered).expect("改変後の内容を書き戻せること");

        let result = verify_chain(&log_path);
        assert!(
            matches!(result, Err(LogError::ChainViolation { .. })),
            "recorded_at_unix_ms のみの改変が ChainViolation として検知されること"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// フィールド改変（`payload` 内部）を検知する。
    #[test]
    fn verify_chain_detects_field_tampering() {
        let log_path = unique_log_path("payload_field_tampering");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let tampered = content.replace("\"attempt\":1", "\"attempt\":999");
        assert_ne!(content, tampered, "置換対象が実在すること");
        std::fs::write(&log_path, tampered).expect("改変後の内容を書き戻せること");

        let result = verify_chain(&log_path);
        assert!(
            matches!(result, Err(LogError::ChainViolation { .. })),
            "フィールド改変が ChainViolation として検知されること"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// `LogRecord` にトップレベルの未知フィールドを注入した JSON 行は、
    /// `deny_unknown_fields` によりデシリアライズ段階で拒否されるべきである
    /// （既知フィールドのみをハッシュ化する [`verify_chain`] が、未知の
    /// プロパティ混入を「検証済み」として見逃さないようにする。
    /// security.md A03/A08・fail-closed 方針に対応。v1 PR #175 review
    /// comment 3700045168 の指摘に対する回帰テストを移植）。
    #[test]
    fn verify_chain_rejects_unknown_top_level_field_injection() {
        let log_path = unique_log_path("unknown_field_injection");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let mut lines: Vec<String> = content.lines().map(str::to_string).collect();
        assert!(!lines.is_empty(), "検証対象となる行が存在すること");
        // 先頭レコードの末尾 `}` の直前に未知のトップレベルフィールドを注入する。
        // ハッシュは既知フィールドのみを対象に再計算されるため、
        // `deny_unknown_fields` がなければこの注入はチェーン検証をすり抜けてしまう。
        let injected = {
            let original = &lines[0];
            let insert_at = original.rfind('}').expect("JSON オブジェクトであること");
            format!(
                "{}{}{}",
                &original[..insert_at],
                r#","injected_unknown_field":"malicious""#,
                &original[insert_at..]
            )
        };
        assert_ne!(
            &injected, &lines[0],
            "未知フィールドが実際に追加されていること"
        );
        lines[0] = injected;
        let tampered = lines
            .into_iter()
            .map(|l| format!("{l}\n"))
            .collect::<String>();
        std::fs::write(&log_path, tampered).expect("改変後の内容を書き戻せること");

        let result = verify_chain(&log_path);
        assert!(
            matches!(result, Err(LogError::Serialization(_))),
            "未知フィールド注入がデシリアライズ段階で拒否されること（fail-closed）"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// レコード削除を検知する。
    #[test]
    fn verify_chain_detects_record_deletion() {
        let log_path = unique_log_path("record_deletion");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![
                attempt(
                    1,
                    AttemptOutcome::VerificationFailed {
                        reason: "1 回目不合格".to_string(),
                    },
                ),
                attempt(2, AttemptOutcome::Adopted),
            ],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() > 2, "削除対象となる中間行が存在すること");
        let truncated: String = lines
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 2)
            .map(|(_, l)| format!("{l}\n"))
            .collect();
        std::fs::write(&log_path, truncated).expect("削除後の内容を書き戻せること");

        let result = verify_chain(&log_path);
        assert!(
            matches!(result, Err(LogError::ChainViolation { .. })),
            "レコード削除が ChainViolation として検知されること"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// レコード順序入れ替えを検知する。
    #[test]
    fn verify_chain_detects_record_reordering() {
        let log_path = unique_log_path("record_reordering");
        let report = sample_report(
            LoopOutcome::Adopted,
            vec![
                attempt(
                    1,
                    AttemptOutcome::VerificationFailed {
                        reason: "1 回目不合格".to_string(),
                    },
                ),
                attempt(2, AttemptOutcome::Adopted),
            ],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let mut lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() > 3, "入れ替え対象の行が十分にあること");
        lines.swap(2, 3);
        let reordered: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(&log_path, reordered).expect("入れ替え後の内容を書き戻せること");

        let result = verify_chain(&log_path);
        assert!(
            matches!(result, Err(LogError::ChainViolation { .. })),
            "順序入れ替えが ChainViolation として検知されること"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// `AttemptOutcome` 全 5 variant の判断根拠（reason 含む）がログから
    /// 復元できる。
    #[test]
    fn attempt_outcomes_round_trip_through_log() {
        let log_path = unique_log_path("attempt_outcomes_round_trip");
        let attempts = vec![
            attempt(
                1,
                AttemptOutcome::VerificationFailed {
                    reason: "検証不合格".to_string(),
                },
            ),
            attempt(
                2,
                AttemptOutcome::AdoptionRejectedRetryable {
                    reason: "再試行可能な却下".to_string(),
                },
            ),
            attempt(
                3,
                AttemptOutcome::RejectedFinal {
                    reason: "再試行不能な却下".to_string(),
                },
            ),
            attempt(
                4,
                AttemptOutcome::Escalated {
                    reason: "人間レビューへ回す".to_string(),
                },
            ),
            attempt(5, AttemptOutcome::Adopted),
        ];
        let report = sample_report(LoopOutcome::Adopted, attempts);
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("追記に失敗しないこと");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let attempt_payloads: Vec<serde_json::Value> = content
            .lines()
            .map(|line| serde_json::from_str::<LogRecord>(line).expect("パースできること"))
            .filter(|record| record.stage == "attempt")
            .map(|record| record.payload)
            .collect();

        assert_eq!(attempt_payloads.len(), 5);
        assert_eq!(
            attempt_payloads[0]["outcome"]["kind"],
            "verification_failed"
        );
        assert_eq!(attempt_payloads[0]["outcome"]["reason"], "検証不合格");
        assert_eq!(
            attempt_payloads[1]["outcome"]["kind"],
            "adoption_rejected_retryable"
        );
        assert_eq!(attempt_payloads[2]["outcome"]["kind"], "rejected_final");
        assert_eq!(attempt_payloads[3]["outcome"]["kind"], "escalated");
        assert_eq!(attempt_payloads[4]["outcome"]["kind"], "adopted");
        let _ = std::fs::remove_file(&log_path);
    }

    /// A03 回帰: reason に JSON 特殊文字（`"` / 改行等）を含んでも整形式
    /// JSONL として出力・再パースでき、値が失われないこと。
    #[test]
    fn special_characters_in_reason_round_trip_safely() {
        let log_path = unique_log_path("special_characters");
        let tricky_reason = "改行\nとダブルクォート\"と\\バックスラッシュを含む理由";
        let report = sample_report(
            LoopOutcome::Rejected {
                stage: "verification",
                reason: tricky_reason.to_string(),
            },
            vec![attempt(
                1,
                AttemptOutcome::RejectedFinal {
                    reason: tricky_reason.to_string(),
                },
            )],
        );
        LogWriter::open(&log_path)
            .expect("開けること")
            .append_report(&report)
            .expect("特殊文字を含む追記に失敗しないこと");

        // 各行が改行を含まない整形式 JSON であること（1 行 1 レコードが崩れていない）。
        let content = std::fs::read_to_string(&log_path).expect("読めること");
        for line in content.lines() {
            let _: LogRecord =
                serde_json::from_str(line).expect("特殊文字を含んでも各行が単独でパースできること");
        }
        verify_chain(&log_path).expect("特殊文字を含んでもチェーン検証が成功すること");

        let outcome_record: LogRecord = content
            .lines()
            .map(|line| serde_json::from_str::<LogRecord>(line).expect("パースできること"))
            .find(|record| record.stage == "loop_outcome")
            .expect("loop_outcome レコードが存在すること");
        assert_eq!(
            outcome_record.payload["outcome"]["reason"], tricky_reason,
            "reason の内容が JSON エスケープを経ても保持されること"
        );
        let _ = std::fs::remove_file(&log_path);
    }

    /// 追記オープンにより既存ログへ追記してもチェーンが連続すること。
    #[test]
    fn reopening_writer_continues_the_chain() {
        let log_path = unique_log_path("reopen_continues_chain");
        let first = sample_report(
            LoopOutcome::Adopted,
            vec![attempt(1, AttemptOutcome::Adopted)],
        );
        LogWriter::open(&log_path)
            .expect("1 回目のオープンに失敗しないこと")
            .append_report(&first)
            .expect("1 回目の追記に失敗しないこと");

        let second = sample_report(LoopOutcome::NoActionNeeded, Vec::new());
        LogWriter::open(&log_path)
            .expect("2 回目のオープン（既存ファイルへの追記）に失敗しないこと")
            .append_report(&second)
            .expect("2 回目の追記に失敗しないこと");

        verify_chain(&log_path).expect("2 回に分けて追記してもチェーンが連続していること");

        let content = std::fs::read_to_string(&log_path).expect("読めること");
        let seqs: Vec<u64> = content
            .lines()
            .map(|line| {
                serde_json::from_str::<LogRecord>(line)
                    .expect("パースできること")
                    .seq
            })
            .collect();
        let expected: Vec<u64> = (0..seqs.len() as u64).collect();
        assert_eq!(seqs, expected, "seq が両方の書き込みを跨いで連番であること");
        let _ = std::fs::remove_file(&log_path);
    }
}
