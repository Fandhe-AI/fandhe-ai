//! 候補修正（[`CandidateFix`]）の表現と、attempt 順適用の共通ロジック
//! （TASK-3.1b・イシュー #133、TASK-3.1c・イシュー #134・REQ-3。移植元は v1
//! `Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/candidate.rs`。
//! `docs/spec/v1-assets-inventory.md` L17「改修して再利用」判定）。
//!
//! [`crate::bug_fix::BugFixFixGenerator`]・
//! [`crate::feature_addition::FeatureAdditionFixGenerator`] の双方が「候補存在
//! 確認 → baseline 復元 → 候補適用」という同一の適用契約（候補枯渇の
//! hard-error 経路では baseline 復元によるファイル書き換えを発生させない）を
//! 持つ。両者から [`apply_candidate`] として共通利用する（構築時検証は種別
//! ごとに異なるため各モジュールに残す。`feature_addition.rs` の
//! `is_manifest_path` 等）。
//!
//! 種別を持たない汎用の [`crate::stages::FixGenerator`] 実装
//! [`CandidateFixGenerator`] も本モジュールが提供する（TASK-3.1c・#134）。
//! v1 の種別別 `FixGenerator`（`bug_fix.rs`・`feature_addition.rs`）は検出器
//! （種別ごとの `Detector`）と一体だったのに対し、決定的な候補列
//! （[`CandidateFix`]）さえ構築時に注入できれば種別非依存に組み立てられる
//! ループ利用者（`verify_gates`・`runner` の新 API 経路）向けに、[`apply_candidate`]
//! を直接ラップする薄い実装として追加する。既存の種別別 `FixGenerator`
//! （`bug_fix.rs`・`feature_addition.rs`）を置き換えるものではなく、
//! いずれも本モジュールの [`apply_candidate`]／[`validate_relative_path`] を
//! 共通基盤として利用する。
//!
//! # A03（インジェクション）対応
//! [`validate_relative_path`] は絶対パス・`..` 成分を含むパスを構築時に
//! 拒否する。これにより候補修正が workspace 外のファイルへ書き込む経路を
//! 閉じる（`.claude/rules/security.md` A03）。

use std::collections::HashMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::error::SelfRepairError;
use crate::stages::{Finding, FixGenerator, Proposal};

/// workspace 相対パスの検証（絶対パス・`..` 成分を拒否）。
///
/// `path` が絶対パスの場合、または `..`（親ディレクトリ参照）成分を含む
/// 場合は `Err` を返す。workspace 外への書き込みを構築時に封じる
/// （`apply_candidate` がファイルシステムへ触れる前に必ず経由させる。
/// `bug_fix.rs`/`feature_addition.rs` の `new` も構築時検証として呼ぶ）。
///
/// これは字句検査（パス文字列の構造のみを見る）であり、workspace 内に
/// 実在する symlink による脱出は検出しない（fd 走査ベースの symlink 検証は
/// `crate::fd_walk`（非公開モジュール）が担う）。両者は独立した防御であり、`apply_candidate`
/// は書き込み直前に双方を経由する。
pub fn validate_relative_path(path: &Path) -> Result<(), String> {
    if path.is_absolute() {
        return Err(format!(
            "候補修正のパスは workspace 相対パスである必要があります（絶対パス: {}）",
            path.display()
        ));
    }
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(format!(
                "候補修正のパスに親ディレクトリ参照（..）を含めることはできません: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// 候補の書き換え対象がガードレール判定に使う設定ファイル
/// （`policy-exclusion.toml`／`guardrail.toml`）かどうかを、ファイル名の
/// 大文字小文字を区別せず判定する（`is_manifest_path`〈`feature_addition.rs`〉
/// と同じ理由。macOS 既定の APFS は大文字小文字を区別しない）。
///
/// [`apply_candidate`] がこれを理由に候補適用を無条件拒否する
/// （fail-closed。多重防御のもう一方は `crate::diff_signals::
/// load_policy_exclusion_config` を候補適用前に一度だけ呼び、結果を
/// 不変値として使い回す設計。`diff_signals.rs` モジュール冒頭「ポリシー
/// 除外設定の信頼境界」参照）。候補がこれらのファイルを書き換えられると、
/// 判定に使われる除外ルール・閾値そのものが候補由来の内容へ差し替わり、
/// ガードレール判定を迂回しうる（PR #361 codex-review P1 指摘。
/// `.claude/rules/security.md` A08「判定の迂回経路を作らない」・
/// 「ガードレール閾値・ポリシー除外リストの変更は必ず人間の承認を経る」）。
///
/// # `CurDir`（`.`）コンポーネントを含むパスの扱い（PR #361 codex-review
/// High 指摘の検証結果）
/// `policy-exclusion.toml/.` のような末尾 `CurDir` パスで `Path::file_name()`
/// が `None` を返し本判定をすり抜けるとの指摘があったが、実測では
/// `std::path::Components` が末尾の `.` を正規化して読み飛ばすため
/// （`file_name()` は最後の非 `CurDir` コンポーネントを返す）発生しない
/// （`is_guardrail_config_path_normalizes_trailing_and_embedded_cur_dir`
/// 参照）。
fn is_guardrail_config_path(rel: &Path) -> bool {
    const PROTECTED_FILE_NAMES: [&str; 2] = ["policy-exclusion.toml", "guardrail.toml"];
    rel.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            PROTECTED_FILE_NAMES
                .iter()
                .any(|protected| name.eq_ignore_ascii_case(protected))
        })
}

// symlink TOCTOU 対策の実体は crate::fd_walk へ移した（PR #361
// codex-review 第 4 波 P0 指摘: 旧 reject_symlink_escape（fs::symlink_metadata
// による逐次「検査」）と fs::write/fs::OpenOptions::open（パス文字列を
// 再解決する「利用」）が別 syscall だったため、両者の間で途中ディレクトリが
// symlink へ差し替えられると（TOCTOU）追跡してしまう余地が残っていた。
// crate::fd_walk::{read_via_fd_walk, write_via_fd_walk, probe} は
// workspace の dir-fd を起点に openat(O_NOFOLLOW|O_DIRECTORY) で
// 1 段ずつ辿り、検証済み fd に対して直接読み書きするため、検査と利用が
// 単一の syscall に統合され TOCTOU window が構造的に存在しない
// （crate::fd_walk モジュール冒頭 doc 参照）。

/// 1 候補修正 = workspace 相対パスと置換後内容の組。
///
/// `description`・`files` とも [`crate::stages::Proposal::description`] と
/// 同じ理由（不変条件を持たない値）で `pub` フィールドとする。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateFix {
    /// 人間可読な修正内容の要約（[`Proposal::description`] へそのまま渡る）。
    pub description: String,
    /// (workspace 相対パス, 置換後の全内容) の列。
    pub files: Vec<(PathBuf, String)>,
}

/// `self-repair run --candidates <path>` の JSON 外部入力境界（イシュー #142
/// 差し戻し分・実装計画 §3.1「追加引数」）。
///
/// `--candidates` は事前生成済みの候補列（AI 生成・人手作成いずれも想定。
/// 生成手段自体は本 CLI のスコープ外）を JSON で受け取る唯一の経路であり、
/// [`load_candidates_from_json`] が [`CandidateFix`] へ変換したうえで
/// [`crate::bug_fix::BugFixFixGenerator::new`]／
/// [`crate::feature_addition::FeatureAdditionFixGenerator::new`] へそのまま渡す。
/// workspace 外パス・`Cargo.toml` 書き換え等の検証は変換後にこれら `new` が
/// 既存の構築時検証（`validate_relative_path`・`is_manifest_path`）で行うため
/// 本モジュールでは再実装しない（A03: 外部入力はまず構造検証してから既存の
/// 検証済み経路へ渡す）。
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateFileDto {
    /// workspace 相対パス（`CandidateFix::files` のパス側）。
    path: String,
    /// 置換後の全内容。
    content: String,
}

/// `--candidates` JSON のトップレベル配列要素 1 件分。
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateFixDto {
    description: String,
    files: Vec<CandidateFileDto>,
}

/// `--candidates` JSON 外部入力の明示上限（PR #361 codex-review 第 3 波 P1
/// 指摘: `fs::read_to_string` の全量読み込みにファイルサイズ・候補数・
/// ファイル数・content 長のいずれも上限がなかった。ガードレール判定閾値では
/// なくコード内定数〈OWASP A03 対応の入力境界〉であり、値の変更自体は
/// ユーザー承認を要さないが、既存の再実証入力
/// （`docs/self-repair-revalidation/*/loop-report.json` 等。最大でも
/// 数十 KiB・候補数は片手で数えられる程度）を壊さない範囲で余裕を持たせた
/// 値とする。
///
/// `--candidates` JSON ファイル自体のサイズ上限。
const MAX_CANDIDATES_JSON_BYTES: u64 = 8 * 1024 * 1024;
/// トップレベル配列（候補数）の上限。
const MAX_CANDIDATES: usize = 64;
/// 候補 1 件あたりのファイル数の上限。
const MAX_FILES_PER_CANDIDATE: usize = 256;
/// ファイル 1 件あたりの `content` 長（バイト数）の上限。
const MAX_CONTENT_BYTES: usize = 1024 * 1024;
/// 全候補・全ファイルの `content` 長の合計上限（`MAX_CANDIDATES` ×
/// `MAX_FILES_PER_CANDIDATE` × `MAX_CONTENT_BYTES` の最悪ケースは
/// `MAX_CANDIDATES_JSON_BYTES` を大きく超えるため、JSON 本体のサイズ上限とは
/// 独立に総量も別途制限する）。
const MAX_TOTAL_CONTENT_BYTES: usize = 16 * 1024 * 1024;

/// `path` の JSON（`CandidateFixDto` の配列）を読み込み [`CandidateFix`] の列へ
/// 変換する（`self-repair run` の `main.rs::run_run` から呼ばれる）。
///
/// # 読み込みサイズの境界
/// `fs::metadata` によるサイズ確認は高速パス（明らかな超過を早期検出し
/// エラーメッセージへ実サイズを含める）に過ぎない。FIFO・procfs 等
/// `metadata().len()` が信頼できない特殊ファイルもありうるため、実効的な
/// 境界は実読み込みを `Read::take(MAX_CANDIDATES_JSON_BYTES + 1)` で
/// 制限し、返ってきた長さが上限を超えていれば拒否する処理が担う
/// （`.claude/rules/security.md` A03: 外部入力は構造検証の前にサイズを
/// 境界づける）。
///
/// # Errors
/// ファイル読み込み失敗・サイズ上限超過・JSON パース失敗
/// （`deny_unknown_fields` によりタイポフィールドも検出する。
/// `docs/guardrail-self-repair-cli.md` 2.5 節と同じ方針）・候補列が空また
/// は候補数／ファイル数／content 長の上限超過、のいずれかで
/// [`SelfRepairError::FixGeneration`]（`attempt: 0` は「試行開始前の候補
/// 読み込み段階」を示す）を返す。
pub fn load_candidates_from_json(path: &Path) -> Result<Vec<CandidateFix>, SelfRepairError> {
    let metadata = fs::metadata(path).map_err(|source| SelfRepairError::FixGeneration {
        attempt: 0,
        reason: format!(
            "候補 JSON のメタデータ取得に失敗しました（path={}）: {source}",
            path.display()
        ),
    })?;
    if metadata.len() > MAX_CANDIDATES_JSON_BYTES {
        return Err(SelfRepairError::FixGeneration {
            attempt: 0,
            reason: format!(
                "候補 JSON がサイズ上限（{MAX_CANDIDATES_JSON_BYTES} バイト）を超えています（path={}, size={}）",
                path.display(),
                metadata.len()
            ),
        });
    }

    let file = fs::File::open(path).map_err(|source| SelfRepairError::FixGeneration {
        attempt: 0,
        reason: format!(
            "候補 JSON の読み込みに失敗しました（path={}）: {source}",
            path.display()
        ),
    })?;
    let mut text = String::new();
    file.take(MAX_CANDIDATES_JSON_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|source| SelfRepairError::FixGeneration {
            attempt: 0,
            reason: format!(
                "候補 JSON の読み込みに失敗しました（path={}）: {source}",
                path.display()
            ),
        })?;
    if text.len() as u64 > MAX_CANDIDATES_JSON_BYTES {
        return Err(SelfRepairError::FixGeneration {
            attempt: 0,
            reason: format!(
                "候補 JSON がサイズ上限（{MAX_CANDIDATES_JSON_BYTES} バイト）を超えています（path={}）",
                path.display()
            ),
        });
    }

    let dtos: Vec<CandidateFixDto> =
        serde_json::from_str(&text).map_err(|error| SelfRepairError::FixGeneration {
            attempt: 0,
            reason: format!(
                "候補 JSON のパースに失敗しました（path={}）: {error}",
                path.display()
            ),
        })?;
    if dtos.is_empty() {
        return Err(SelfRepairError::FixGeneration {
            attempt: 0,
            reason: format!("候補 JSON が空です（path={}）", path.display()),
        });
    }
    if dtos.len() > MAX_CANDIDATES {
        return Err(SelfRepairError::FixGeneration {
            attempt: 0,
            reason: format!(
                "候補数が上限（{MAX_CANDIDATES}）を超えています（path={}, 候補数={}）",
                path.display(),
                dtos.len()
            ),
        });
    }

    // 上限検査（ファイル数・content 長・総量）は serde_json のパース後に
    // 行うため、JSON 本体のサイズ上限（上記 `Read::take`）が実質的なメモリ
    // 確保量の境界であり、以下は構造上の妥当性チェックに位置づけられる。
    let mut total_content_bytes: usize = 0;
    for dto in &dtos {
        if dto.files.len() > MAX_FILES_PER_CANDIDATE {
            return Err(SelfRepairError::FixGeneration {
                attempt: 0,
                reason: format!(
                    "候補あたりのファイル数が上限（{MAX_FILES_PER_CANDIDATE}）を超えています（path={}, ファイル数={}）",
                    path.display(),
                    dto.files.len()
                ),
            });
        }
        for file in &dto.files {
            if file.content.len() > MAX_CONTENT_BYTES {
                return Err(SelfRepairError::FixGeneration {
                    attempt: 0,
                    reason: format!(
                        "content 長が上限（{MAX_CONTENT_BYTES} バイト）を超えています（path={}, file={}）",
                        path.display(),
                        file.path
                    ),
                });
            }
            total_content_bytes = total_content_bytes.saturating_add(file.content.len());
            if total_content_bytes > MAX_TOTAL_CONTENT_BYTES {
                return Err(SelfRepairError::FixGeneration {
                    attempt: 0,
                    reason: format!(
                        "候補 JSON 全体の content 合計サイズが上限（{MAX_TOTAL_CONTENT_BYTES} バイト）を超えています（path={}）",
                        path.display()
                    ),
                });
            }
        }
    }

    Ok(dtos
        .into_iter()
        .map(|dto| CandidateFix {
            description: dto.description,
            files: dto
                .files
                .into_iter()
                .map(|file| (PathBuf::from(file.path), file.content))
                .collect(),
        })
        .collect())
}

/// `candidates[attempt - 1]` を `workspace` へ適用し [`Proposal`] を返す共通本体。
///
/// 手順（[`crate::bug_fix::BugFixFixGenerator::generate`]・
/// [`crate::feature_addition::FeatureAdditionFixGenerator::generate`]・
/// [`CandidateFixGenerator::generate`] が委譲する）:
/// 1. 候補（`attempt` 番目、1 始まり）の存在確認 — ファイルシステム副作用
///    より前に行う。候補枯渇（`attempt` が候補数を超える）を検出した場合は
///    baseline 復元を一切行わずに `Err` を返す。これを候補確認より後に
///    baseline 復元を行う実装にすると、候補枯渇時にも復元処理が実行され
///    「本来何もしないはずの hard-error 経路でファイルが書き換わる」
///    副作用が生じる（v1 PR #172 の指摘）。
/// 2. baseline 復元（前回試行の書き換えを白紙化）
/// 3. 候補適用（今回の書き換え）
pub fn apply_candidate(
    workspace: &Path,
    baseline: &HashMap<PathBuf, String>,
    candidates: &[CandidateFix],
    attempt: u32,
) -> Result<Proposal, SelfRepairError> {
    // 1. 候補存在確認（副作用より前）。
    let index =
        (attempt as usize)
            .checked_sub(1)
            .ok_or_else(|| SelfRepairError::FixGeneration {
                attempt,
                reason: "attempt は 1 始まりである必要があります".to_string(),
            })?;
    let candidate = candidates
        .get(index)
        .ok_or_else(|| SelfRepairError::FixGeneration {
            attempt,
            reason: format!(
                "候補修正が尽きました（attempt={attempt}・候補数={}）",
                candidates.len()
            ),
        })?;

    // 2. baseline 復元。
    //
    // `apply_candidate` は `pub`（crate 外からも呼べる公開エントリポイント）
    // であり、`CandidateFixGenerator::new` の構築時検証（呼び出し元限定の
    // 経路）だけに依存すると、この関数を直接呼ぶ経路では
    // `validate_relative_path` を経由せずファイルシステムへ書き込みうる
    // （絶対パス・`..` セグメントが `workspace.join(path)` 経由で workspace
    // 外へ脱出する。A03 対応）。書き込みの直前ではなく、他方の走査より
    // 前に全パスを検証し尽くすことで、一部のファイルだけが書き換わった
    // 状態で `Err` を返す事態を避ける。字句検査（`validate_relative_path`）
    // に加え、`crate::fd_walk::probe` で fd 走査による symlink 検証も upfront
    // に済ませる（[`crate::fd_walk::probe`] doc 参照。PR #361 codex-review
    // 第 4 波 P0 指摘対応。実際の書き込み時にも [`crate::fd_walk::write_via_fd_walk`]
    // が同じ検証を行うため、本ループは「部分適用の防止」のための早期失敗
    // 判定であり、symlink 拒否そのものの安全性は書き込み時の検証が担保する）。
    // この「部分適用の防止」保証は `probe` が「中間ディレクトリの `NotFound`
    // を拒否し、末端ファイルの `NotFound` のみ許容する」契約を守ることに
    // 依存する（PR #361 codex-review 追加指摘 P1 対応。`crate::fd_walk::probe`
    // doc 参照）。`probe` があらゆる `NotFound` を一律許容すると、`write_via_fd_walk`
    // が新規作成できない中間ディレクトリ不在の候補が事前検証を通過してしまい、
    // 先行する書き込み可能な候補だけが適用された状態で後続の書き込みが失敗
    // しうる（本コメント冒頭の「一部だけ書き換わった状態で `Err` を返さない」
    // 契約に違反する）。

    for relative_path in baseline.keys() {
        validate_relative_path(relative_path)
            .map_err(|reason| SelfRepairError::FixGeneration { attempt, reason })?;
        crate::fd_walk::probe(workspace, relative_path).map_err(|error| {
            SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "候補修正パスの検証に失敗しました（{}）: {error}",
                    relative_path.display()
                ),
            }
        })?;
    }
    for (relative_path, _content) in &candidate.files {
        validate_relative_path(relative_path)
            .map_err(|reason| SelfRepairError::FixGeneration { attempt, reason })?;
        // ガードレール判定に使う設定ファイル自体の書き換えは無条件拒否する
        // （`is_guardrail_config_path` doc 参照。ファイルシステムへ触れる前の
        // 早期拒否であり、他候補ファイルの部分適用も発生させない）。
        if is_guardrail_config_path(relative_path) {
            return Err(SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "ガードレール設定ファイルの書き換えは対象外です（判定迂回防止。\
                     .claude/rules/security.md「ガードレール閾値・ポリシー除外リストの\
                     変更は必ず人間の承認を経る」）: {}",
                    relative_path.display()
                ),
            });
        }
        crate::fd_walk::probe(workspace, relative_path).map_err(|error| {
            SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "候補修正パスの検証に失敗しました（{}）: {error}",
                    relative_path.display()
                ),
            }
        })?;
    }
    for (relative_path, original_content) in baseline {
        crate::fd_walk::write_via_fd_walk(workspace, relative_path, original_content).map_err(
            |error| SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "baseline 復元に失敗しました（{}）: {error}",
                    relative_path.display()
                ),
            },
        )?;
    }

    // 3. 候補適用。
    for (relative_path, content) in &candidate.files {
        crate::fd_walk::write_via_fd_walk(workspace, relative_path, content).map_err(|error| {
            SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "候補修正の適用に失敗しました（{}）: {error}",
                    relative_path.display()
                ),
            }
        })?;
    }

    Ok(Proposal {
        attempt,
        description: candidate.description.clone(),
    })
}

/// [`FixGenerator`] の種別非依存な実装。
///
/// 構築時に決定的な候補列（`candidates`）を受け取り、`generate` 呼び出しごと
/// に `attempt` 番目の候補を [`apply_candidate`] 経由でファイルシステムへ
/// 適用する。種別別の候補**選定**ロジック（バグ修正ならどのテスト失敗から
/// どんな差分を作るか等）は持たず、あくまで「候補列 → 適用」の機械的な
/// 変換のみを担う（モジュール冒頭ドキュメント参照）。
pub struct CandidateFixGenerator {
    workspace: PathBuf,
    baseline: HashMap<PathBuf, String>,
    candidates: Vec<CandidateFix>,
}

impl CandidateFixGenerator {
    /// `workspace` 配下で `candidates` を試行順に適用する
    /// [`CandidateFixGenerator`] を構築する。
    ///
    /// 構築時に全候補パスを [`validate_relative_path`] で検証し
    /// （A03 対応。ファイルシステムへ触れる前に構築自体を失敗させる）、
    /// 候補が参照する全ファイルの現内容を baseline としてスナップショット
    /// する（`apply_candidate` の再試行時復元に使う）。
    pub fn new(workspace: PathBuf, candidates: Vec<CandidateFix>) -> Result<Self, SelfRepairError> {
        let mut baseline = HashMap::new();
        for candidate in &candidates {
            for (relative_path, _content) in &candidate.files {
                validate_relative_path(relative_path)
                    .map_err(|reason| SelfRepairError::FixGeneration { attempt: 0, reason })?;
                if baseline.contains_key(relative_path) {
                    continue;
                }
                // baseline スナップショット読み込みは `crate::fd_walk::read_via_fd_walk`
                // が fd 走査（`openat(O_NOFOLLOW)`）で symlink 追跡を拒否しつつ
                // 直接読み込む（`crate::fd_walk` モジュール冒頭 doc 参照）。
                let original_content = crate::fd_walk::read_via_fd_walk(&workspace, relative_path)
                    .map_err(|error| SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "baseline スナップショット取得に失敗しました（{}）: {error}",
                            relative_path.display()
                        ),
                    })?;
                baseline.insert(relative_path.clone(), original_content);
            }
        }
        Ok(CandidateFixGenerator {
            workspace,
            baseline,
            candidates,
        })
    }
}

impl FixGenerator for CandidateFixGenerator {
    fn generate(&self, _finding: &Finding, attempt: u32) -> Result<Proposal, SelfRepairError> {
        // `Proposal.attempt` にループから渡された `attempt` をそのまま
        // 設定する（`apply_candidate` が返す `Proposal` がこの契約を満たす。
        // `runner.rs` の fail-closed 検査契約〈attempt 番号の単一の真実源〉
        // を満たすため、ここで値をすり替えない）。
        apply_candidate(&self.workspace, &self.baseline, &self.candidates, attempt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::RepairKind;
    use std::fs;

    fn write_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create_dir_all should succeed in test setup");
        }
        fs::write(path, content).expect("write should succeed in test setup");
    }

    fn temp_workspace(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "self-repair-candidate-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create_dir_all should succeed in test setup");
        dir
    }

    #[test]
    fn validate_relative_path_rejects_absolute_path() {
        let result = validate_relative_path(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_relative_path_rejects_parent_dir_traversal() {
        let result = validate_relative_path(Path::new("../outside/file.rs"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_relative_path_accepts_plain_relative_path() {
        let result = validate_relative_path(Path::new("src/lib.rs"));
        assert!(result.is_ok());
    }

    /// PR #361 codex-review High 指摘（`CurDir bypasses config write block`）の
    /// 検証テスト。指摘は `Path::file_name()` が末尾コンポーネントが `.`
    /// （`CurDir`）の場合に `None` を返すため
    /// `policy-exclusion.toml/.` のようなパスが `is_guardrail_config_path` を
    /// すり抜けると主張していたが、`std::path::Components` は末尾以外の `.`
    /// と同様に末尾の `.` も正規化して読み飛ばすため（`file_name()` の実装は
    /// `components().next_back()` が最後の非 `CurDir` コンポーネントを返す）、
    /// 実測では発生しない（本テストの各 assert が実測確認）。回帰防止として
    /// 固定する。
    #[test]
    fn is_guardrail_config_path_normalizes_trailing_and_embedded_cur_dir() {
        assert!(is_guardrail_config_path(Path::new(
            "policy-exclusion.toml/."
        )));
        assert!(is_guardrail_config_path(Path::new(
            "./policy-exclusion.toml"
        )));
        assert!(is_guardrail_config_path(Path::new(
            "nested/./policy-exclusion.toml"
        )));
        assert!(is_guardrail_config_path(Path::new(
            "policy-exclusion.toml/./."
        )));
        assert!(is_guardrail_config_path(Path::new("guardrail.toml/.")));
    }

    #[test]
    fn is_guardrail_config_path_ignores_case_and_directory() {
        assert!(is_guardrail_config_path(Path::new("policy-exclusion.toml")));
        assert!(is_guardrail_config_path(Path::new("Policy-Exclusion.TOML")));
        assert!(is_guardrail_config_path(Path::new("guardrail.toml")));
        assert!(is_guardrail_config_path(Path::new("GUARDRAIL.toml")));
        assert!(is_guardrail_config_path(Path::new(
            "nested/dir/policy-exclusion.toml"
        )));
        assert!(!is_guardrail_config_path(Path::new("Cargo.toml")));
        assert!(!is_guardrail_config_path(Path::new("src/lib.rs")));
    }

    #[cfg(unix)]
    #[test]
    fn apply_candidate_rejects_candidate_via_symlink_file_without_touching_target() {
        // `apply_candidate` レベルでも symlink 経由の候補パスを拒否し、
        // symlink の指す先（workspace 外）を書き換えないことを確認する
        // （PR #361 codex-review P0 指摘の回帰防止）。
        let dir = temp_workspace("apply-symlink-file");
        let outside_dir = temp_workspace("apply-symlink-file-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "do-not-overwrite").expect("write should succeed in test setup");

        std::os::unix::fs::symlink(&outside_file, dir.join("target.txt"))
            .expect("symlink creation should succeed in test setup");

        let baseline = HashMap::new();
        let candidates = vec![CandidateFix {
            description: "malicious-symlink".to_string(),
            files: vec![(PathBuf::from("target.txt"), "pwned".to_string())],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(result.is_err());
        assert_eq!(
            fs::read_to_string(&outside_file).expect("read should succeed"),
            "do-not-overwrite"
        );

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[cfg(unix)]
    #[test]
    fn apply_candidate_rejects_candidate_via_symlink_directory_without_touching_target() {
        let dir = temp_workspace("apply-symlink-dir");
        let outside_dir = temp_workspace("apply-symlink-dir-outside");
        fs::create_dir_all(&outside_dir).expect("create_dir_all should succeed in test setup");

        std::os::unix::fs::symlink(&outside_dir, dir.join("sub"))
            .expect("symlink creation should succeed in test setup");

        let baseline = HashMap::new();
        let candidates = vec![CandidateFix {
            description: "malicious-symlink-dir".to_string(),
            files: vec![(PathBuf::from("sub/target.txt"), "pwned".to_string())],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(result.is_err());
        assert!(!outside_dir.join("target.txt").exists());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn apply_candidate_applies_in_attempt_order() {
        let dir = temp_workspace("order");
        write_file(&dir, "target.txt", "original");

        let mut baseline = HashMap::new();
        baseline.insert(PathBuf::from("target.txt"), "original".to_string());

        let candidates = vec![
            CandidateFix {
                description: "attempt 1".to_string(),
                files: vec![(PathBuf::from("target.txt"), "fix-1".to_string())],
            },
            CandidateFix {
                description: "attempt 2".to_string(),
                files: vec![(PathBuf::from("target.txt"), "fix-2".to_string())],
            },
        ];

        let proposal = apply_candidate(&dir, &baseline, &candidates, 1)
            .expect("attempt 1 should apply successfully");
        assert_eq!(proposal.attempt, 1);
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "fix-1"
        );

        let proposal = apply_candidate(&dir, &baseline, &candidates, 2)
            .expect("attempt 2 should apply successfully");
        assert_eq!(proposal.attempt, 2);
        // baseline 復元後に候補 2 が適用されるため "fix-1" は残らない。
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "fix-2"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_exhaustion_does_not_touch_filesystem() {
        // 候補枯渇時（attempt が候補数を超える）は baseline 復元すら
        // 発生しないことを確認する（v1 PR #172 指摘の回帰防止）。
        let dir = temp_workspace("exhaustion");
        write_file(&dir, "target.txt", "untouched");

        let mut baseline = HashMap::new();
        baseline.insert(
            PathBuf::from("target.txt"),
            "should-not-be-written".to_string(),
        );

        let candidates = vec![CandidateFix {
            description: "only attempt".to_string(),
            files: vec![(PathBuf::from("target.txt"), "fix-1".to_string())],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 2);
        assert!(result.is_err());
        // baseline 復元が実行されていれば "should-not-be-written" になる。
        // 候補確認が先に走り Err を返すため、ファイル内容は変化しない。
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "untouched"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_rejects_missing_intermediate_directory_without_partial_write() {
        // PR #361 codex-review 追加指摘 P1 回帰テスト。候補の先頭ファイルは
        // 既存の書き込み可能パス、後続ファイルは中間ディレクトリが未実在の
        // パス（`missing/target.rs`）とする。旧実装ではあらゆる `NotFound`
        // を probe が許容していたため、この事前検証を通過したうえで先頭
        // ファイルが書き換わってから後続の書き込みで失敗し、「一部だけ
        // 書き換わった状態で `Err` を返さない」契約に違反していた。
        let dir = temp_workspace("missing-intermediate-dir");
        write_file(&dir, "target.txt", "original");

        let mut baseline = HashMap::new();
        baseline.insert(PathBuf::from("target.txt"), "original".to_string());

        let candidates = vec![CandidateFix {
            description: "partial apply attempt".to_string(),
            files: vec![
                (
                    PathBuf::from("target.txt"),
                    "should-not-be-written".to_string(),
                ),
                (
                    PathBuf::from("missing/target.rs"),
                    "should-not-be-written-either".to_string(),
                ),
            ],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(
            result.is_err(),
            "中間ディレクトリ不在の候補を含む場合は事前検証で拒否されるべきです"
        );
        assert_eq!(
            fs::read_to_string(dir.join("target.txt")).expect("read should succeed"),
            "original",
            "後続候補の中間ディレクトリ不在により、先頭候補も書き換わってはいけません"
        );
        assert!(!dir.join("missing").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_rejects_attempt_zero() {
        let dir = temp_workspace("attempt-zero");
        let baseline = HashMap::new();
        let candidates: Vec<CandidateFix> = Vec::new();
        let result = apply_candidate(&dir, &baseline, &candidates, 0);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_fix_generator_retains_loop_attempt_number() {
        let dir = temp_workspace("generator");
        write_file(&dir, "target.txt", "original");

        let candidates = vec![CandidateFix {
            description: "fix".to_string(),
            files: vec![(PathBuf::from("target.txt"), "fixed".to_string())],
        }];

        let generator = CandidateFixGenerator::new(dir.clone(), candidates)
            .expect("generator construction should succeed");
        let finding = Finding::new(RepairKind::BugFix, "dummy finding");
        let proposal = generator
            .generate(&finding, 1)
            .expect("generate should succeed");
        assert_eq!(proposal.attempt, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_rejects_unsafe_candidate_path_without_touching_filesystem() {
        // `apply_candidate` は `pub`（crate 外から `CandidateFixGenerator::new`
        // の構築時検証を経由せず直接呼べる）であるため、この関数自身が
        // `validate_relative_path` を経由することを確認する（A03 対応。
        // Cursor Bugbot review id 4885516474 指摘の回帰防止）。
        let dir = temp_workspace("apply-unsafe-candidate-path");
        let baseline = HashMap::new();
        let candidates = vec![CandidateFix {
            description: "malicious".to_string(),
            files: vec![(PathBuf::from("../outside.txt"), "pwned".to_string())],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(result.is_err());
        assert!(!dir.parent().unwrap().join("outside.txt").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    /// PR #361 codex-review P1 指摘の回帰防止（要求 (2)）: 候補が
    /// `policy-exclusion.toml` を書き換えようとした場合、`apply_candidate`
    /// がファイルシステムへ触れる前に無条件拒否することを確認する
    /// （`is_guardrail_config_path` doc・`diff_signals.rs` モジュール冒頭
    /// 「ポリシー除外設定の信頼境界」参照）。
    #[test]
    fn apply_candidate_rejects_policy_exclusion_toml_rewrite_candidate() {
        let dir = temp_workspace("apply-rejects-policy-exclusion-rewrite");
        write_file(&dir, "policy-exclusion.toml", "[[exclusion]]\n");
        let baseline = HashMap::new();
        let candidates = vec![CandidateFix {
            description: "malicious exclusion widening".to_string(),
            files: vec![(
                PathBuf::from("policy-exclusion.toml"),
                "[[exclusion]]\nid = \"self-immunize\"\n".to_string(),
            )],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(result.is_err());
        let content =
            fs::read_to_string(dir.join("policy-exclusion.toml")).expect("read should succeed");
        assert_eq!(
            content, "[[exclusion]]\n",
            "拒否された候補が policy-exclusion.toml を書き換えてはならない"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// 上記と同じ契約を `guardrail.toml`（`Category`・大文字小文字混在パス）
    /// でも確認する。
    #[test]
    fn apply_candidate_rejects_guardrail_toml_rewrite_candidate_case_insensitive() {
        let dir = temp_workspace("apply-rejects-guardrail-toml-rewrite");
        write_file(&dir, "GuardRail.TOML", "[thresholds]\n");
        let baseline = HashMap::new();
        let candidates = vec![CandidateFix {
            description: "malicious threshold widening".to_string(),
            files: vec![(
                PathBuf::from("GuardRail.TOML"),
                "[thresholds]\nmax_lines_changed = 999999\n".to_string(),
            )],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_candidate_rejects_unsafe_baseline_path_without_touching_filesystem() {
        // baseline 側（前回試行の復元対象）も同様に検証対象とする
        // （baseline は `CandidateFixGenerator::new` が構築するため通常は
        // 安全だが、`apply_candidate` を直接呼ぶ経路では任意の
        // `HashMap` を渡せるため、baseline 側の脱出も塞ぐ）。
        let dir = temp_workspace("apply-unsafe-baseline-path");
        let mut baseline = HashMap::new();
        baseline.insert(
            PathBuf::from("../outside.txt"),
            "should-not-write".to_string(),
        );
        let candidates = vec![CandidateFix {
            description: "harmless".to_string(),
            files: vec![(PathBuf::from("target.txt"), "fix".to_string())],
        }];

        let result = apply_candidate(&dir, &baseline, &candidates, 1);
        assert!(result.is_err());
        assert!(!dir.parent().unwrap().join("outside.txt").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_fix_generator_construction_rejects_unsafe_path() {
        let dir = temp_workspace("unsafe-path");
        let candidates = vec![CandidateFix {
            description: "malicious".to_string(),
            files: vec![(PathBuf::from("../outside.txt"), "pwned".to_string())],
        }];
        let result = CandidateFixGenerator::new(dir.clone(), candidates);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    /// `--candidates` JSON 読み込み専用の一時ファイルパス（`temp_workspace`
    /// と同じ `temp_dir() + process::id()` 方式だがディレクトリではなく単一
    /// ファイルを扱う）。
    fn temp_json_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "self-repair-candidate-json-{name}-{}.json",
            std::process::id()
        ))
    }

    #[test]
    fn load_candidates_from_json_parses_valid_input() {
        let path = temp_json_path("valid");
        fs::write(
            &path,
            r#"[
                {"description": "試行1", "files": [{"path": "src/lib.rs", "content": "fn a() {}"}]},
                {"description": "試行2", "files": [{"path": "src/lib.rs", "content": "fn b() {}"}]}
            ]"#,
        )
        .expect("一時 JSON の書き込みに失敗");

        let candidates = load_candidates_from_json(&path).expect("パースに成功するはず");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].description, "試行1");
        assert_eq!(
            candidates[1].files,
            vec![(PathBuf::from("src/lib.rs"), "fn b() {}".to_string())]
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_candidates_from_json_rejects_missing_file() {
        let path = temp_json_path("missing");
        let _ = fs::remove_file(&path);
        let result = load_candidates_from_json(&path);
        assert!(result.is_err());
    }

    #[test]
    fn load_candidates_from_json_rejects_empty_array() {
        let path = temp_json_path("empty");
        fs::write(&path, "[]").expect("一時 JSON の書き込みに失敗");
        let result = load_candidates_from_json(&path);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_candidates_from_json_rejects_unknown_field() {
        let path = temp_json_path("unknown-field");
        fs::write(
            &path,
            r#"[{"description": "x", "files": [], "bogus": true}]"#,
        )
        .expect("一時 JSON の書き込みに失敗");
        let result = load_candidates_from_json(&path);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }

    /// PR #361 codex-review 第 3 波 P1 指摘の回帰防止（ファイルサイズ上限）。
    /// `MAX_CANDIDATES_JSON_BYTES` を超えるファイルは（JSON として妥当か
    /// どうかに関わらず）拒否されることを確認する。
    #[test]
    fn load_candidates_from_json_rejects_file_size_over_limit() {
        let path = temp_json_path("oversize-file");
        // JSON としての妥当性は問わない（サイズ検査がパース前に効くことを
        // 確認するテストのため、パディングの中身は任意）。
        let padding = "x".repeat((MAX_CANDIDATES_JSON_BYTES + 1) as usize);
        fs::write(&path, padding).expect("一時 JSON の書き込みに失敗");
        let result = load_candidates_from_json(&path);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }

    /// 候補数上限（`MAX_CANDIDATES`）超過を拒否することを確認する。
    #[test]
    fn load_candidates_from_json_rejects_candidate_count_over_limit() {
        let path = temp_json_path("too-many-candidates");
        let candidates: Vec<String> = (0..(MAX_CANDIDATES + 1))
            .map(|i| format!(r#"{{"description": "c{i}", "files": []}}"#))
            .collect();
        fs::write(&path, format!("[{}]", candidates.join(",")))
            .expect("一時 JSON の書き込みに失敗");
        let result = load_candidates_from_json(&path);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }

    /// 候補数が上限ちょうど（`MAX_CANDIDATES`）の場合は受理されることを
    /// 確認する（上限超過テストと対になる境界確認）。
    #[test]
    fn load_candidates_from_json_accepts_candidate_count_at_limit() {
        let path = temp_json_path("candidates-at-limit");
        let candidates: Vec<String> = (0..MAX_CANDIDATES)
            .map(|i| format!(r#"{{"description": "c{i}", "files": []}}"#))
            .collect();
        fs::write(&path, format!("[{}]", candidates.join(",")))
            .expect("一時 JSON の書き込みに失敗");
        let result = load_candidates_from_json(&path);
        assert_eq!(
            result.expect("上限ちょうどの候補数は受理されるはず").len(),
            MAX_CANDIDATES
        );
        let _ = fs::remove_file(&path);
    }

    /// content 長上限（`MAX_CONTENT_BYTES`）超過を拒否することを確認する
    /// （ファイルサイズ自体は `MAX_CANDIDATES_JSON_BYTES` 未満に収める。
    /// ファイルサイズ検査とは独立の検査であることを確認するため）。
    #[test]
    fn load_candidates_from_json_rejects_content_over_limit() {
        let path = temp_json_path("oversize-content");
        let oversized_content = "x".repeat(MAX_CONTENT_BYTES + 1);
        let json = format!(
            r#"[{{"description": "c", "files": [{{"path": "a.txt", "content": "{oversized_content}"}}]}}]"#
        );
        assert!(
            (json.len() as u64) < MAX_CANDIDATES_JSON_BYTES,
            "このテストはファイルサイズ上限とは独立の content 長検査を確認するためのもの"
        );
        fs::write(&path, json).expect("一時 JSON の書き込みに失敗");
        let result = load_candidates_from_json(&path);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }
}
