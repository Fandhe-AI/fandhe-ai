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
/// 実在する symlink による脱出（下記 [`reject_symlink_escape`]）は検出
/// しない。両者は独立した防御であり、`apply_candidate` は書き込み直前に
/// 双方を経由する。
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

/// `workspace` 配下に実在する symlink 経由の脱出を拒否する（OWASP A03。
/// `.claude/rules/security.md`）。
///
/// [`validate_relative_path`] はパス文字列の字句（絶対パス・`..`）しか
/// 見ないため、`--candidates` の外部入力が指す相対パスの途中に
/// workspace 外を指す symlink（ファイル・ディレクトリいずれも）が実在
/// すると、`fs::write` がその symlink を追跡して sandbox 外の任意ファイル
/// を上書きしうる（PR #361 codex-review P0 指摘: `apply_candidate` の
/// `fs::write(workspace.join(relative_path), ...)` がパストラバーサルの
/// 経路になる）。
///
/// `workspace` から `relative_path` の各コンポーネントを 1 段ずつ
/// `symlink_metadata`（`fs::metadata` と異なり symlink 自体を追跡しない）
/// で検査し、途中経路・書き込み先本体のいずれかが symlink であれば
/// `Err` で fail-closed に拒否する。symlink の指す先が workspace 内か外か
/// を判定せず一律拒否する（AI 生成候補が正当に symlink を作る必要は
/// なく、判定を「解決先が workspace 内か」に緩めると
/// TOCTOU（検査後に指す先が変わる）の余地を残すため）。まだ存在しない
/// 中間ディレクトリ・新規作成予定のファイル（`NotFound`）は許容する
/// （通常の新規ファイル書き込みを妨げないため）。
///
/// `pub(crate)`: `apply_candidate`（書き込み経路）に加え、
/// `BugFixFixGenerator::new`／`FeatureAdditionFixGenerator::new`
/// （baseline スナップショット読み込み経路。`bug_fix.rs`/`feature_addition.rs`）
/// からも同じ検査を利用する（読み込みも symlink を追跡するため、書き込み
/// 経路だけを塞いでも読み込み経路から任意ファイルの内容を baseline へ
/// 取り込みうる余地が残る）。
pub(crate) fn reject_symlink_escape(workspace: &Path, relative_path: &Path) -> Result<(), String> {
    let mut current = workspace.to_path_buf();
    for component in relative_path.components() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(format!(
                        "候補修正のパスに symlink が含まれています（workspace 外への書き込みを許すため拒否します）: {}",
                        current.display()
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // 新規作成予定のパス（未実在の中間ディレクトリ・末端
                // ファイル）は symlink になりようがないため許容する。
                continue;
            }
            Err(error) => {
                return Err(format!(
                    "候補修正パスのメタデータ取得に失敗しました（{}）: {error}",
                    current.display()
                ));
            }
        }
    }
    Ok(())
}

/// `O_NOFOLLOW`（open 対象の末尾コンポーネントが symlink の場合に open 自体を
/// 失敗させるフラグ）の値。`libc` 依存は禁止（deps-policy.md 許容依存 8 区分
/// 外）のため、各 OS の `fcntl.h` 定義値をローカル定数として複製する。
/// 出典: Linux `include/uapi/asm-generic/fcntl.h`（`0o400000` = `0x20000`）・
/// macOS `sys/fcntl.h`（`0x0100`）。
#[cfg(target_os = "linux")]
const O_NOFOLLOW: i32 = 0o400000;
#[cfg(target_os = "macos")]
const O_NOFOLLOW: i32 = 0x0100;

/// [`reject_symlink_escape`] は「検査」であり、検査後 [`write_no_follow_symlink`]／
/// [`read_no_follow_symlink`] が実際に呼ばれるまでの間にパスが symlink へ
/// 差し替えられると（TOCTOU）追跡してしまう（PR #361 codex-review 第 3 波 P0
/// 指摘）。本関数は書き込み時の open 自体に `O_NOFOLLOW` を付けることで
/// 「検査」と「利用」を単一の syscall へ統合し、末尾コンポーネントに関する
/// TOCTOU の隙をなくす。
///
/// 制約: `O_NOFOLLOW` は open 対象パスの末尾コンポーネントにのみ効き、途中
/// ディレクトリの symlink 差し替えは防がない（`openat`/dir-fd チェーンで
/// 塞ぐ手段があるが `libc` 依存になるためポリシー上不可。
/// [`reject_symlink_escape`] の逐次コンポーネント検査が主防御であり、本関数は
/// 末尾コンポーネントに対する第 2 層の防御に留まる）。
///
/// `O_NOFOLLOW` の値定義は Linux・macOS のみ持つ（[`O_NOFOLLOW`] doc 参照）。
/// これ以外の全ターゲット（FreeBSD 等の他 Unix・Windows）は
/// [`write_via_temp_rename`] へフォールバックする（PR #361 codex-review
/// 指摘 1・2: 旧実装は `#[cfg(not(unix))]` で `O_NOFOLLOW` 定義のないターゲット
/// を含む Unix 全般をこの cfg で誤って捕捉しビルド不能にしていた上、
/// フォールバック自体も `remove_file` → `create_new` の順で失敗時に元ファイル
/// を失う実装だった）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn write_no_follow_symlink(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)?;
    file.write_all(content.as_bytes())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn write_no_follow_symlink(path: &Path, content: &str) -> std::io::Result<()> {
    write_via_temp_rename(path, content)
}

/// [`write_no_follow_symlink`] の Linux/macOS 以外向けフォールバック本体
/// （PR #361 codex-review 指摘 2 対応）。
///
/// `path` と同一ディレクトリに一意名の一時ファイルを `create_new`
/// （`O_CREAT|O_EXCL` 相当。既存パスがあれば必ず失敗する）で作成し全量書き込み
/// ＋ flush した後、`path` を `fs::rename` で置換する。書き込み・flush・
/// rename 前 symlink 再検査のいずれかが失敗した場合は一時ファイルを削除し
/// `path` には一切触れないため、旧実装（`remove_file` 先行）と異なり失敗時に
/// 元ファイルを喪失しない。
///
/// rename 直前の `symlink_metadata` 再検査は
/// [`reject_symlink_escape`] の事前検査と本関数呼び出しの間の TOCTOU window
/// を縮小する防御であり、`fs::rename` 自体が対象パスの symlink を追跡しない
/// （symlink エントリ自体を置き換える）ため、この再検査を通過せず symlink の
/// 指す先が書き換わることはそもそもない。再検査は fail-closed の早期拒否に
/// 過ぎない。
///
/// 注意: `fs::rename` は新しい inode で置換するため、`path` の既存パーミッション・
/// 所有者は引き継がれない。self-repair の候補適用はユーザー所有の workspace
/// 内ファイルへの書き込みに限られ、パーミッション・所有権の保持は要件外
/// （baseline 復元・候補適用いずれも内容の一致のみを検証する）のため許容する。
///
/// `#[cfg(test)]` でも Linux/macOS 上でコンパイル対象に含める。フォールバック
/// 経路は通常ビルドでは非対象 OS 上でしか使われず CI（Linux self-hosted）では
/// 検証されないため、テストのみこの関数を直接呼び動作を固定する
/// （下記 `write_via_temp_rename_*` テスト参照）。
#[cfg(any(test, not(any(target_os = "linux", target_os = "macos"))))]
fn write_via_temp_rename(path: &Path, content: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::other(format!(
            "書き込み先パスにファイル名がありません: {}",
            path.display()
        ))
    })?;

    // 一意な一時ファイル名を生成する。プロセス ID だけでは同一プロセス内の
    // 複数呼び出し（`apply_candidate` の baseline 復元・候補適用の連続呼び出し
    // 等）で衝突しうるため単調カウンタも混ぜ、`create_new` が
    // `AlreadyExists` を返した場合はカウンタを進めて再試行する。
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut last_error = None;
    for _ in 0..8 {
        let counter = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp_path = parent.join(format!(
            ".{}.tmp-{}-{counter}",
            file_name.to_string_lossy(),
            std::process::id()
        ));

        let mut file = match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_error = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };

        if let Err(error) = file
            .write_all(content.as_bytes())
            .and_then(|()| file.flush())
        {
            drop(file);
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }
        drop(file);

        // rename 直前の symlink 再検査（TOCTOU window 縮小。上記 doc 参照）。
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let _ = fs::remove_file(&tmp_path);
                return Err(std::io::Error::other(format!(
                    "書き込み先が symlink です（追跡を拒否します）: {}",
                    path.display()
                )));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                let _ = fs::remove_file(&tmp_path);
                return Err(error);
            }
        }

        return fs::rename(&tmp_path, path).inspect_err(|_| {
            let _ = fs::remove_file(&tmp_path);
        });
    }
    Err(last_error.unwrap_or_else(|| {
        std::io::Error::other("一時ファイル名の生成に失敗しました（再試行上限超過）")
    }))
}

/// [`write_no_follow_symlink`] の読み込み版（baseline スナップショット取得に
/// 使う。`bug_fix.rs`／`feature_addition.rs`／[`CandidateFixGenerator::new`] の
/// 全 baseline 読み込み経路から呼ばれる）。
///
/// Linux・macOS（`O_NOFOLLOW` の値定義を持つターゲット。[`O_NOFOLLOW`] doc
/// 参照）は `O_NOFOLLOW` 付き open で末尾コンポーネントの symlink 追跡を拒否
/// する。それ以外（FreeBSD 等の他 Unix・Windows）は読み込み対象を消すわけに
/// いかないため [`write_via_temp_rename`] と同じ一時ファイル方式は使えず、
/// [`reject_symlink_escape`] の事前検査のみに依拠する残存 TOCTOU window が
/// ある。この window は「symlink の指す先の内容が baseline へ取り込まれる」
/// （workspace 内へ content が入ってくる方向）に留まり、書き込み経路
/// （workspace 外への書き込み）より影響が限定的なため許容する。
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn read_no_follow_symlink(path: &Path) -> std::io::Result<String> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn read_no_follow_symlink(path: &Path) -> std::io::Result<String> {
    fs::read_to_string(path)
}

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
    // に加え、workspace 内に実在する symlink 経由の脱出も同じ upfront
    // 検証で塞ぐ（`reject_symlink_escape` doc 参照。PR #361 codex-review
    // P0 指摘対応）。
    for relative_path in baseline.keys() {
        validate_relative_path(relative_path)
            .map_err(|reason| SelfRepairError::FixGeneration { attempt, reason })?;
        reject_symlink_escape(workspace, relative_path)
            .map_err(|reason| SelfRepairError::FixGeneration { attempt, reason })?;
    }
    for (relative_path, _content) in &candidate.files {
        validate_relative_path(relative_path)
            .map_err(|reason| SelfRepairError::FixGeneration { attempt, reason })?;
        reject_symlink_escape(workspace, relative_path)
            .map_err(|reason| SelfRepairError::FixGeneration { attempt, reason })?;
    }
    for (relative_path, original_content) in baseline {
        let absolute_path = workspace.join(relative_path);
        write_no_follow_symlink(&absolute_path, original_content).map_err(|error| {
            SelfRepairError::FixGeneration {
                attempt,
                reason: format!(
                    "baseline 復元に失敗しました（{}）: {error}",
                    relative_path.display()
                ),
            }
        })?;
    }

    // 3. 候補適用。
    for (relative_path, content) in &candidate.files {
        let absolute_path = workspace.join(relative_path);
        write_no_follow_symlink(&absolute_path, content).map_err(|error| {
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
                // symlink 経由の脱出は書き込み（`apply_candidate`）だけで
                // なく、ここでの baseline スナップショット読み込み
                // （`fs::read_to_string`）も symlink を追跡するため同じ
                // 検査を構築時にも通す（`reject_symlink_escape` doc 参照）。
                reject_symlink_escape(&workspace, relative_path)
                    .map_err(|reason| SelfRepairError::FixGeneration { attempt: 0, reason })?;
                if baseline.contains_key(relative_path) {
                    continue;
                }
                let absolute_path = workspace.join(relative_path);
                let original_content = read_no_follow_symlink(&absolute_path).map_err(|error| {
                    SelfRepairError::FixGeneration {
                        attempt: 0,
                        reason: format!(
                            "baseline スナップショット取得に失敗しました（{}）: {error}",
                            relative_path.display()
                        ),
                    }
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

    #[test]
    fn reject_symlink_escape_accepts_plain_path() {
        let dir = temp_workspace("symlink-plain");
        write_file(&dir, "target.txt", "original");

        let result = reject_symlink_escape(&dir, Path::new("target.txt"));
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn write_no_follow_symlink_rejects_symlink_target_at_open_time() {
        // `reject_symlink_escape` の事前検査を経由せず `write_no_follow_symlink`
        // を直接呼ぶことで、TOCTOU 対策（open 自体が symlink を拒否する）が
        // 事前検査に依存せず単独で機能することを固定する（PR #361
        // codex-review 第 3 波 P0 指摘の回帰防止）。
        let dir = temp_workspace("write-no-follow-reject");
        let outside_dir = temp_workspace("write-no-follow-reject-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "do-not-overwrite").expect("write should succeed in test setup");

        let target = dir.join("target.txt");
        std::os::unix::fs::symlink(&outside_file, &target)
            .expect("symlink creation should succeed in test setup");

        let result = write_no_follow_symlink(&target, "pwned");
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
    fn write_no_follow_symlink_writes_plain_file() {
        let dir = temp_workspace("write-no-follow-plain");
        let target = dir.join("target.txt");

        write_no_follow_symlink(&target, "content").expect("write should succeed");
        assert_eq!(
            fs::read_to_string(&target).expect("read should succeed"),
            "content"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    // Linux/macOS 以外向けフォールバック（`write_via_temp_rename`）の動作固定
    // （PR #361 codex-review 指摘 2 の回帰防止）。フォールバックは通常ビルド
    // では非対象 OS 上でしか使われないため、`#[cfg(test)]` により Linux
    // 上でも直接呼んで検証する（`write_via_temp_rename` の doc 参照）。
    #[test]
    fn write_via_temp_rename_replaces_existing_file_and_cleans_up_temp() {
        let dir = temp_workspace("temp-rename-replace");
        let target = dir.join("target.txt");
        fs::write(&target, "original").expect("write should succeed in test setup");

        write_via_temp_rename(&target, "replaced").expect("write should succeed");
        assert_eq!(
            fs::read_to_string(&target).expect("read should succeed"),
            "replaced"
        );

        // 一時ファイルが残存していないことを確認する（cleanup バグの回帰防止）。
        let leftover_tmp_entries: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir should succeed")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .contains(".target.txt.tmp-")
            })
            .collect();
        assert!(
            leftover_tmp_entries.is_empty(),
            "一時ファイルが残存しています: {leftover_tmp_entries:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn write_via_temp_rename_rejects_symlink_target_without_touching_outside_file() {
        // `fs::rename` 自体は symlink を追跡しないため書き込み経路として
        // 安全だが、rename 直前の `symlink_metadata` 再検査により早期に
        // fail-closed で拒否することを確認する（`write_via_temp_rename` doc
        // 参照）。
        let dir = temp_workspace("temp-rename-reject-symlink");
        let outside_dir = temp_workspace("temp-rename-reject-symlink-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "do-not-overwrite").expect("write should succeed in test setup");

        let target = dir.join("target.txt");
        std::os::unix::fs::symlink(&outside_file, &target)
            .expect("symlink creation should succeed in test setup");

        let result = write_via_temp_rename(&target, "pwned");
        assert!(result.is_err());
        assert_eq!(
            fs::read_to_string(&outside_file).expect("read should succeed"),
            "do-not-overwrite"
        );
        // symlink 自体もこの経路で書き換わっていないことを確認する。
        assert!(
            fs::symlink_metadata(&target)
                .expect("symlink_metadata should succeed")
                .file_type()
                .is_symlink()
        );

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn write_via_temp_rename_on_rename_failure_keeps_destination_and_removes_temp() {
        // `write_via_temp_rename` doc の「失敗時は一時ファイルを削除し `path`
        // には一切触れない」を固定する（PR #361 codex-review 指摘 2 の
        // 核心）。書き込み先を空でないディレクトリにすると `fs::rename`
        // （通常ファイル → ディレクトリ）が root 権限下でも決定的に失敗する
        // （`EISDIR` 相当）ため、chmod によるパーミッション操作
        // （root では無視され得るため不採用）より確実にロールバック経路を
        // 駆動できる。
        let dir = temp_workspace("temp-rename-failure");
        let dest = dir.join("target");
        fs::create_dir_all(&dest).expect("create_dir_all should succeed in test setup");
        fs::write(dest.join("keep.txt"), "keep").expect("write should succeed in test setup");

        let result = write_via_temp_rename(&dest, "pwned");
        assert!(result.is_err());
        // 置換先ディレクトリの既存内容が無傷であることを確認する。
        assert_eq!(
            fs::read_to_string(dest.join("keep.txt")).expect("read should succeed"),
            "keep"
        );
        // 一時ファイルが残存していないことを確認する。
        let leftover_tmp_entries: Vec<_> = fs::read_dir(&dir)
            .expect("read_dir should succeed")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains(".target.tmp-"))
            .collect();
        assert!(
            leftover_tmp_entries.is_empty(),
            "一時ファイルが残存しています: {leftover_tmp_entries:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn read_no_follow_symlink_rejects_symlink_target_at_open_time() {
        let dir = temp_workspace("read-no-follow-reject");
        let outside_dir = temp_workspace("read-no-follow-reject-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "secret-content").expect("write should succeed in test setup");

        let target = dir.join("target.txt");
        std::os::unix::fs::symlink(&outside_file, &target)
            .expect("symlink creation should succeed in test setup");

        let result = read_no_follow_symlink(&target);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // `std::os::unix::fs::symlink` を使うため unix 限定
    // （`tests/cli_run.rs::run_with_non_utf8_kind_value_does_not_panic` と
    // 同じ方針）。
    #[cfg(unix)]
    #[test]
    fn reject_symlink_escape_rejects_symlink_file_pointing_outside_workspace() {
        // 候補が指すファイル自体が workspace 外を指す symlink の場合
        // （codex-review 指摘: `fs::write` が symlink を追跡し sandbox 外の
        // 任意ファイルを上書きしうる）。
        let dir = temp_workspace("symlink-file-target");
        let outside_dir = temp_workspace("symlink-file-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "do-not-overwrite").expect("write should succeed in test setup");

        std::os::unix::fs::symlink(&outside_file, dir.join("target.txt"))
            .expect("symlink creation should succeed in test setup");

        let result = reject_symlink_escape(&dir, Path::new("target.txt"));
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
    fn reject_symlink_escape_rejects_symlink_directory_pointing_outside_workspace() {
        // 途中ディレクトリが workspace 外を指す symlink の場合
        // （`sub/target.txt` の `sub` が symlink ディレクトリ）。
        let dir = temp_workspace("symlink-dir-target");
        let outside_dir = temp_workspace("symlink-dir-outside");
        fs::create_dir_all(&outside_dir).expect("create_dir_all should succeed in test setup");

        std::os::unix::fs::symlink(&outside_dir, dir.join("sub"))
            .expect("symlink creation should succeed in test setup");

        let result = reject_symlink_escape(&dir, Path::new("sub/target.txt"));
        assert!(result.is_err());
        assert!(!outside_dir.join("target.txt").exists());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
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
