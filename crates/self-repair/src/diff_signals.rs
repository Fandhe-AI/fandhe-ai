//! 試行ごとの diff 由来シグナル実測（TASK-3.2a・イシュー #137）。
//!
//! [`crate::verify_gates::CargoVerificationGate`] は `lines_changed`／
//! `api_broken`／`gaming_suspect`／`exclusion_rule_ids` を構築時の必須引数として
//! 受け取るのみで自ら計測しない（`verify_gates.rs` モジュール冒頭「スコープ境界」
//! 参照）。この実測ロジックはこれまで `tests/revalidation_bug_fix.rs`
//! （`diff_numstat`／`api_signature_touched`／`gaming_suspect`／
//! `evaluate_exclusion_rules`）にのみ存在し、テスト専用の使い捨て実装だった。
//!
//! 本モジュールはそのロジックを `crates/self-repair/src/` 本体へ昇格し、
//! [`crate::verify_direct_composite::RepairCompositeGate`] が**試行ごとに**
//! 呼び出せるようにする（実装計画 #137 §3.3・§4）。これにより
//! `crate::verify_composite::FeatureAdditionCompositeGate`（構築時にシグナルを
//! 固定する既存合成ゲート）が持つ「試行ごとの再計測ができない」という設計制約
//! （`verify_composite.rs` モジュール冒頭ドキュメント参照）を、新設の合成ゲート側
//! では解消する。
//!
//! `revalidation_bug_fix.rs` 自体は編集しない（#141 との並行編集衝突を避ける。
//! `.claude/rules/delegation-impl.md`）。ロジックの複製元として参照するのみ。
//!
//! # fail-closed 契約（A08）
//! git 呼び出し・ポリシー除外リストのロード/評価がいずれか失敗した場合、
//! 未計測値を fail-open な既定値（0 行・破壊なし・ゲーミング疑いなし）で
//! 埋めず [`Err`] を返す（`.claude/rules/security.md` A08「判定の迂回経路を
//! 作らない」。`verify_gates.rs` の契約と同じ）。
//!
//! 変更ファイル一覧は `git diff --name-status -z`（[`list_changed_files`]）で
//! 構造化取得する。`git diff --numstat`（`-z` なし）の 3 列目は rename を
//! `src/{old.rs => new.rs}` という単一フィールドの人間可読表記で返すため、
//! これをそのまま `git show <baseline>:<file>` のパス引数に使うと必ず失敗し、
//! その失敗を「baseline に存在しない新規ファイル」へ丸めると rename と同時の
//! 破壊的変更（既存 `pub fn` 削除等）を見逃す（PR #361 Codex レビュー P1）。
//! `git show` の失敗は、`list_changed_files` が返すステータスで新規追加
//! （`ChangedFile::is_newly_added()`）と確認できた場合のみ許容し、それ以外は
//! [`Err`]（fail-closed）とする（[`show_file_at_baseline`] ドキュメント参照）。
//!
//! # A03（インジェクション）対応
//! `baseline_commit` は git コマンドライン引数へ渡す前に 7〜40 桁の 16 進文字列
//! であることを検証する（[`validate_commit_ref`]）。検証しない場合、先頭に
//! `-` を含む値（例: `--upload-pack=...`）を git のオプションとして誤解釈させる
//! 攻撃（コマンドラインオプション偽装）が成立しうる
//! （`.claude/rules/security.md` A03）。パス引数は `--` 区切りで分離し、
//! シェル（`sh -c` 等）は一切経由しない（`crate::exec` モジュール冒頭ドキュメント
//! と同じ設計）。
//!
//! # ポリシー除外設定の信頼境界（PR #361 codex-review P1 指摘対応）
//! [`load_policy_exclusion_config`] は候補適用前に一度だけ呼び出し、返る
//! [`guardrail::PolicyExclusionConfig`] を試行ループ全体で不変値として使い回す
//! 契約とする（同関数 doc 参照）。`policy-exclusion.toml` の既定パスは
//! sandbox 内（候補適用対象そのもの）を指すため、[`measure_diff_signals`]
//! が呼ばれるたびファイルを再読込する実装だと、候補が自身の diff に本来
//! match するはずの除外ルールを**削除・縮小**したうえで書き換え、以降の
//! `verify` にそのルール抜きの状態を読ませて match を回避できてしまう
//! （攻撃方向は「ルールの追加」ではない: `guardrail::decision::decide` は
//! `exclusion_rule_ids` が 1 件以上あれば機械判定の結果によらず無条件で
//! エスカレーションへ回す「安全側にしか作用しない」設計〈`crates/guardrail/
//! src/decision.rs` モジュール冒頭「判定順序の契約」参照〉であるため、
//! ルールを増やしても判定は緩まない。緩む方向は「本来 match すべきルールを
//! 消し match させない」ことだけである。A08「判定の迂回経路を作らない」
//! 違反。`crate::verify_direct_composite::RepairCompositeGate` が本モジュール
//! の契約をどう守るかは同モジュールのドキュメント参照）。

use std::path::Path;

use crate::exec::CommandRunner;

/// 試行ごとに実測した diff 由来 4 シグナル（`guardrail::DecisionInput::new` の
/// `lines_changed`／`api_broken`／`gaming_suspect`／`exclusion_rule_ids` 引数に
/// 1 対 1 対応する。`outcome.rs` の `VerifiedEvidence` フィールドと同じ意味）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSignals {
    /// `baseline_commit` から現在の作業木までの追加行数＋削除行数の合計
    /// （`git diff --numstat` の実測値）。
    pub lines_changed: u64,
    /// baseline 時点に存在した公開シグネチャ（`pub fn`／`pub struct`／
    /// `pub enum` の宣言行）が現在の作業木から消失しているか（新規追加のみの
    /// 変更は破壊とみなさない）。意味論は `guardrail::checks::api_stability::
    /// api_broken`（`crates/guardrail/src/checks/api_stability.rs`）と同一。
    /// 当該関数は `guardrail` クレートの非公開実装（`mod checks;`。`lib.rs`
    /// 参照）のため呼び出せず、本モジュールが同じ意味論を独立して複製する
    /// （イシュー #142 差し戻し分: 旧実装は追加・削除いずれの `pub fn` 行も
    /// 一律 `api_broken=true` としており、新規公開関数の追加という機能追加
    /// 種別の候補が原理的に必ずエスカレーションされる誤った意味論だった。
    /// `checks/api_stability.rs` の `adding_new_pub_fn_is_not_broken` が
    /// 検出器の既存仕様〈追加は破壊ではない〉を証明する）。
    pub api_broken: bool,
    /// 変更ファイル一覧に本番コードとテストコードの双方が同時に含まれるか
    /// （ゲーミング疑いの簡易ヒューリスティック）。
    pub gaming_suspect: bool,
    /// match したポリシー除外リストのルール `id` 一覧（空 = match なし）。
    pub exclusion_rule_ids: Vec<String>,
}

/// `git diff --name-status -z` の 1 レコード（構造化ステータス＋パス）。
///
/// [`list_changed_files`] が構築する。`status` の先頭 1 文字が `'R'`／`'C'`
/// （rename／copy）の場合のみ `old_path` を `Some` にする（`git` の
/// `-z` 出力はこの場合のみ旧パス・新パスを別々の NUL 区切りフィールドとして
/// 出す。それ以外の `A`／`M`／`D`／`T` 等は単一パスであり baseline 側・
/// 現作業木側で同じパスを指す）。`path` は常に現作業木側のパス（`A` の
/// 場合のみ baseline 側に対応物がない新規パス）。
///
/// # 修正の経緯（PR #361 Codex レビュー P1）
/// これ以前は `git diff --numstat` の 3 列目をそのままファイルパスとして
/// 扱っていた。しかし `--numstat`（`-z` なし）は rename を `src/{old.rs =>
/// new.rs}` という**単一フィールド中の人間可読表記**で返すため、この文字列を
/// そのまま `git show <baseline>:<file>` のパス引数へ渡すと必ず失敗する。
/// [`show_file_at_baseline_if_present`] がこの失敗を一律「baseline に存在
/// しない新規ファイル」（`Ok(None)`）に丸めていたため、rename と同時に
/// 既存 `pub fn` を削除しても `api_broken=false` になり得た（fail-closed
/// 違反・A08）。`--name-status -z` は rename／copy を旧パス・新パスの組で
/// 構造化して返すため、この曖昧さが生じない。
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChangedFile {
    /// `git diff --name-status` のステータス文字列（例: `"A"`／`"M"`／`"D"`／
    /// `"R100"`／`"C100"`）。先頭 1 文字のみを判定に使う。
    status: String,
    /// rename／copy の場合のみ `Some`（baseline 側のパス）。
    old_path: Option<String>,
    /// 現作業木側のパス（`status` が `"A"` の場合は baseline に対応物がない）。
    path: String,
}

impl ChangedFile {
    /// baseline 側のパス（rename／copy なら旧パス、それ以外は `path` と同一）。
    fn baseline_path(&self) -> &str {
        self.old_path.as_deref().unwrap_or(&self.path)
    }

    /// baseline に対応物を持たない新規追加ファイルか（`status` 先頭が `'A'`）。
    fn is_newly_added(&self) -> bool {
        self.status.starts_with('A')
    }
}

/// [`measure_diff_signals`] の実測時エラー。
///
/// `crate::error::SelfRepairError` は `attempt` を必須フィールドとして要求する
/// （`error.rs` 参照）が、本関数は `attempt` を知らない（呼び出し元
/// `RepairCompositeGate::verify` が `Proposal::attempt` を知る）ため、独立した
/// エラー型として返し、呼び出し元で `SelfRepairError::Verification` へ変換する
/// （`crate::verify_gates::CargoVerificationGate::run_gate` が `ExecError` を
/// 変換する契約と同じ設計）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSignalsError {
    message: String,
}

impl DiffSignalsError {
    fn new(message: impl Into<String>) -> Self {
        DiffSignalsError {
            message: message.into(),
        }
    }

    /// 人間可読なエラー内容（`SelfRepairError::Verification.reason` へ埋め込む
    /// 用途）。
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for DiffSignalsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "diff 由来シグナルの実測に失敗しました: {}", self.message)
    }
}

impl std::error::Error for DiffSignalsError {}

/// `baseline_commit` が git のコマンドラインオプションとして誤解釈されない
/// 16 進 commit sha（短縮形含む）であることを検証する（モジュール冒頭
/// 「A03（インジェクション）対応」参照）。
fn validate_commit_ref(baseline_commit: &str) -> Result<(), DiffSignalsError> {
    let is_valid = (7..=40).contains(&baseline_commit.len())
        && baseline_commit
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && (b.is_ascii_digit() || b.is_ascii_lowercase()));
    if is_valid {
        Ok(())
    } else {
        Err(DiffSignalsError::new(format!(
            "baseline_commit は 7〜40 桁の小文字 16 進文字列である必要があります: {baseline_commit:?}"
        )))
    }
}

/// `runner` で `git` を `sandbox_root` を cwd として起動し、標準出力（UTF-8
/// lossy 変換済み）を返す。非 0 終了・spawn 失敗はいずれも [`Err`]（fail-closed。
/// stderr は診断用にメッセージへ含める）。
///
/// # ログ切り詰めの扱い（Codex レビュー #137 指摘）
/// [`crate::exec::CommandOutput::log_tail`] は 256 KiB 超過時に**先頭側**を
/// 切り詰める（`exec.rs` の `MAX_CAPTURED_LOG_BYTES` 参照）。本関数の戻り値は
/// `diff_numstat`／`api_signature_touched` が `git diff --numstat`／
/// `git diff -U0` の構造化出力として逐次パースするため、先頭側が欠落した
/// ログをそのまま解析すると `lines_changed` の過少計上・`api_broken` の
/// 見逃しに繋がる（fail-closed 契約違反。モジュール冒頭ドキュメント参照）。
/// `output.truncated()` が真の場合は解析を試みず [`Err`] を返す。
fn run_git<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    args: &[&str],
) -> Result<String, DiffSignalsError> {
    let output = runner
        .run("git", args, sandbox_root)
        .map_err(|error| DiffSignalsError::new(format!("git {args:?} の起動に失敗: {error}")))?;
    if !output.success() {
        return Err(DiffSignalsError::new(format!(
            "git {args:?} が失敗しました: {}",
            output.log_tail()
        )));
    }
    if output.truncated() {
        return Err(DiffSignalsError::new(format!(
            "git {args:?} の出力が 256 KiB 上限で切り詰められました。構造化出力の \
             先頭側欠落により lines_changed／api_broken を誤計測しうるため、\
             fail-open な部分解析はせず拒否します（fail-closed。\
             .claude/rules/security.md A08）"
        )));
    }
    Ok(output.log_tail().to_string())
}

/// `sandbox_root` の作業木で未追跡ファイルを含む全ファイルを index へ反映する
/// （`git add -A -- .`）。
///
/// `diff_numstat`／`api_signature_touched`（本モジュール）と
/// `guardrail::EvaluationContext::from_repo`（`evaluate_exclusion_rules` が
/// 内部で呼ぶ。`git diff <baseline_commit>` を独自に起動する既存契約であり
/// `CommandRunner` 注入が効かない点はモジュール冒頭ドキュメント参照）はいずれも
/// `git diff <baseline_commit> --` 系のコマンドで差分を取得するが、この形式は
/// **未追跡（新規追加）ファイルを出力に含めない**（Codex レビュー #137 指摘）。
/// 候補が新規 `.rs` ファイルを追加した場合、`lines_changed`／`api_broken`／
/// `gaming_suspect` の計測対象から外れ、変更規模・公開 API ガードを迂回しうる
/// （fail-open）。
///
/// `sandbox_root` は `git worktree add --detach` で作られた候補適用専用の
/// 隔離作業木（呼び出し元 `RepairCompositeGate::verify` の契約。
/// `crate::verify_bench_direct` モジュール冒頭ドキュメント参照）であり、実リポの
/// 作業木・index には影響しない。ここで index へ反映しておけば、以降に走る
/// 全ての `git diff <baseline_commit>` 系コマンド（本モジュール内・guardrail
/// 内部呼び出しの双方）が新規ファイルを diff 対象に含めるようになる。
/// staging 失敗（git 未初期化・権限エラー等）は fail-open で無視せず [`Err`]
/// を返す。
fn stage_untracked_files<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
) -> Result<(), DiffSignalsError> {
    let output = runner
        .run("git", &["add", "-A", "--", "."], sandbox_root)
        .map_err(|error| {
            DiffSignalsError::new(format!(
                "git add -A の起動に失敗（未追跡ファイル反映）: {error}"
            ))
        })?;
    if !output.success() {
        return Err(DiffSignalsError::new(format!(
            "git add -A が失敗しました（未追跡ファイル反映）: {}",
            output.log_tail()
        )));
    }
    Ok(())
}

/// `baseline_commit` と現在の作業木の diff から `lines_changed`（追加行数＋
/// 削除行数の合計）を実測する。移植元: `tests/revalidation_bug_fix.rs::
/// diff_numstat`（モジュール冒頭ドキュメント参照）。バイナリファイル等で
/// `added`/`deleted` 列の解析に失敗した場合、fail-open な 0 加算にせず
/// [`Err`] を返す（移植元コメントの fail-closed 方針をそのまま踏襲。本番
/// 経路のため `panic!` ではなく型付きエラーにする）。
///
/// # rename 表記と行数計測の独立性（PR #361 Codex レビュー P1）
/// `--numstat`（`-z` なし）は rename を 3 列目に `src/{old.rs => new.rs}`
/// という人間可読の単一フィールドで返すが、1・2 列目（追加行数・削除行数）
/// は rename の有無に関わらずタブ区切りの数値のまま変わらない。本関数は
/// 3 列目（パス）を読み捨て 1・2 列目のみを合算するため、rename 表記の
/// 曖昧さに影響されない。ファイルパスが必要な処理（`api_broken`・
/// `gaming_suspect` の実測）は構造化された [`list_changed_files`]
/// （`git diff --name-status -z`）側に一本化する。
fn diff_numstat<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
) -> Result<u64, DiffSignalsError> {
    let stdout = run_git(
        runner,
        sandbox_root,
        &["diff", "--numstat", baseline_commit, "--"],
    )?;
    let mut lines_changed: u64 = 0;
    for line in stdout.lines() {
        let mut cols = line.splitn(3, '\t');
        let added = cols.next().ok_or_else(|| {
            DiffSignalsError::new(format!("git diff --numstat の行が想定外です: {line:?}"))
        })?;
        let deleted = cols.next().ok_or_else(|| {
            DiffSignalsError::new(format!("git diff --numstat の行が想定外です: {line:?}"))
        })?;
        lines_changed += added.parse::<u64>().map_err(|error| {
            DiffSignalsError::new(format!(
                "added 列の解析に失敗しました（fail-open で 0 に丸めない）: {added:?}: {error}"
            ))
        })?;
        lines_changed += deleted.parse::<u64>().map_err(|error| {
            DiffSignalsError::new(format!(
                "deleted 列の解析に失敗しました（fail-open で 0 に丸めない）: {deleted:?}: {error}"
            ))
        })?;
    }
    Ok(lines_changed)
}

/// トークンが `git diff --name-status` のステータス列として妥当な形式か
/// （先頭 1 文字が `A`／`M`／`D`／`T`／`U`／`X`／`B`／`R`／`C` のいずれかで、
/// 残りは数字のみ〈rename／copy の類似度。例: `R100`〉）。
///
/// # 存在理由（PR #361 Codex レビュー フォローアップ）
/// [`run_git`]（延いては [`list_changed_files`]）は stdout と stderr を
/// 結合したログを返す（`exec.rs::SystemCommandRunner::run` 参照）。
/// `stage_untracked_files` が直前に走らせる `git add -A` 等は環境によって
/// stderr へ警告（CRLF 変換警告等）を出しうる。この警告テキストが `-z`
/// 出力の NUL 区切りストリームへ（NUL 終端なしで）連結されると、警告文の
/// 断片が「ステータストークン」として読まれ、後続の本来のパストークンを
/// 巻き込んで誤ってペアリングしうる（レコードのズレ）。本関数でステータス
/// トークンの形式を検証し、想定外の文字列は「解析失敗」として拒否する
/// ことで、この巻き込みを fail-open な誤ペアリングではなく fail-closed な
/// エラーに変える。
fn is_plausible_status_token(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !matches!(first, 'A' | 'M' | 'D' | 'T' | 'U' | 'X' | 'B' | 'R' | 'C') {
        return false;
    }
    chars.all(|c| c.is_ascii_digit())
}

/// `baseline_commit` と現在の作業木の diff から変更ファイル一覧を
/// **構造化**して実測する（`git diff --name-status -z`）。
///
/// `-z` は各レコードを NUL 区切りで返す（パス自体に含まれ得るタブ・改行・
/// クォートの曖昧さを避けるため）。rename／copy（ステータス先頭が `'R'`／
/// `'C'`）のみ、旧パス・新パスが別々の NUL 区切りフィールドとして続く
/// （実測確認済み: `git diff --name-status -z <baseline> --` は類似度が
/// rename 検出閾値〈既定 50%〉を上回る変更を `R050\0old\0new\0` の形で返す。
/// 閾値未満の変更は `D\0old\0` と `A\0new\0` の 2 レコードに分かれ、この
/// 場合も本関数の対応で問題なく扱える——`D` は baseline 側 = 現作業木側
/// パスとして扱われ、[`api_signature_touched`] が baseline 内容と「削除
/// 済み＝空文字列」を比較して破壊を検出する）。他のステータス
/// （`A`／`M`／`D`／`T` 等）は単一パスのみ続く。各ステータストークンは
/// [`is_plausible_status_token`] で形式検証する（同関数ドキュメント参照。
/// stdout/stderr 結合ログでの誤ペアリング対策）。
///
/// # 非 UTF-8 パスの扱い
/// `run_git`（延いては本関数）は `String::from_utf8_lossy` を経由するため
/// （`exec.rs::CommandOutput::from_captured`）、非 UTF-8 バイト列を含む
/// パスは置換文字 U+FFFD へ変換される。この場合、後段の処理は fail-open に
/// ならない: 現作業木側パスが破損すれば `std::fs::read_to_string` が失敗し
/// 「削除済み＝空文字列」扱い（[`api_signature_touched`]）で安全側（破壊
/// あり方向）に倒れ、baseline 側パス（`ChangedFile::baseline_path()`）が
/// 破損すれば `git show` が失敗し、新規追加以外は [`Err`]（fail-closed。
/// [`show_file_at_baseline`] 参照）になる。
fn list_changed_files<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
) -> Result<Vec<ChangedFile>, DiffSignalsError> {
    let stdout = run_git(
        runner,
        sandbox_root,
        &["diff", "--name-status", "-z", baseline_commit, "--"],
    )?;
    let mut tokens = stdout.split('\0');
    let mut files = Vec::new();
    while let Some(status) = tokens.next() {
        if status.is_empty() {
            // `-z` 出力末尾の NUL によって生じる空トークン（split の仕様上、
            // 末尾に必ず 1 つ現れる）。ステータス文字列が空になることは
            // 他になく安全に読み飛ばせる。
            continue;
        }
        if !is_plausible_status_token(status) {
            return Err(DiffSignalsError::new(format!(
                "git diff --name-status -z の出力にステータストークンとして不正な \
                 値が含まれています（stdout/stderr 結合ログへの警告混入等による \
                 レコードのズレの可能性があるため fail-closed に拒否します。\
                 .claude/rules/security.md A08）: {status:?}"
            )));
        }
        let is_rename_or_copy = status.starts_with('R') || status.starts_with('C');
        if is_rename_or_copy {
            let old_path = tokens.next().ok_or_else(|| {
                DiffSignalsError::new(format!(
                    "git diff --name-status -z の rename/copy レコードに旧パスがありません: status={status:?}"
                ))
            })?;
            let new_path = tokens.next().ok_or_else(|| {
                DiffSignalsError::new(format!(
                    "git diff --name-status -z の rename/copy レコードに新パスがありません: status={status:?}"
                ))
            })?;
            files.push(ChangedFile {
                status: status.to_string(),
                old_path: Some(old_path.to_string()),
                path: new_path.to_string(),
            });
        } else {
            let path = tokens.next().ok_or_else(|| {
                DiffSignalsError::new(format!(
                    "git diff --name-status -z のレコードにパスがありません: status={status:?}"
                ))
            })?;
            files.push(ChangedFile {
                status: status.to_string(),
                old_path: None,
                path: path.to_string(),
            });
        }
    }
    Ok(files)
}

/// シグネチャ行として扱う接頭辞（先頭空白除去後）。`guardrail::checks::
/// api_stability::PUBLIC_SIGNATURE_PREFIXES` と同一の判定基準（意味論を
/// 独立複製する理由は [`DiffSignals::api_broken`] のドキュメント参照）。
const PUBLIC_SIGNATURE_PREFIXES: [&str; 3] = ["pub fn ", "pub struct ", "pub enum "];

/// `content` から公開 API シグネチャ行（トリム済み）を抽出する。移植元:
/// `guardrail::checks::api_stability::extract_public_signatures`。
fn extract_public_signatures(content: &str) -> Vec<String> {
    content
        .lines()
        .map(str::trim)
        .filter(|line| {
            PUBLIC_SIGNATURE_PREFIXES
                .iter()
                .any(|prefix| line.starts_with(prefix))
        })
        .map(str::to_string)
        .collect()
}

/// `baseline_commit:file` の内容を取得する。呼び出し元（[`api_signature_touched`]）
/// は `changed_files`（[`list_changed_files`] の構造化ステータス）で
/// `ChangedFile::is_newly_added()` が真と確認できたファイルについてのみ本関数を
/// **呼ばない**契約とする。したがって本関数へ渡される `file` は baseline 時点に
/// 存在するはずであり、`git show` の非 0 終了は「baseline に存在しない」への
/// fail-open な丸めではなく実測異常として [`Err`]（fail-closed）を返す。
///
/// # 修正の経緯（PR #361 Codex レビュー P1）
/// 旧実装（`show_file_at_baseline_if_present`）は `git show` の非 0 終了を
/// 一律「baseline に存在しない新規ファイル」（`Ok(None)`）に丸めていた。
/// [`list_changed_files`] 導入以前は変更ファイル一覧が `--numstat` の
/// 人間可読パス（rename 時は `src/{old.rs => new.rs}`）由来だったため、
/// rename されたファイルへの `git show` は常にこの経路で失敗し「新規
/// ファイル」に丸められていた。rename と同時に既存 `pub fn` を削除した
/// 変更が `api_broken=false` へすり抜けうる欠陥だった（fail-closed 違反・
/// A08）。現在は [`list_changed_files`]（`git diff --name-status -z`）が
/// rename／copy を旧パス・新パスの組として構造化するため、baseline 側の
/// パス（`ChangedFile::baseline_path()`）を渡せば `git show` は新規追加
/// 以外で失敗しないはずであり、失敗は実測異常として拒否してよい。
/// `output.truncated()` の場合も同様に「存在しない」へ丸めず [`Err`]
/// （fail-closed。[`run_git`] と同じ理由）。
fn show_file_at_baseline<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
    file: &str,
) -> Result<String, DiffSignalsError> {
    let path_arg = format!("{baseline_commit}:{file}");
    let output = runner
        .run("git", &["show", &path_arg], sandbox_root)
        .map_err(|error| {
            DiffSignalsError::new(format!("git show {path_arg:?} の起動に失敗: {error}"))
        })?;
    if !output.success() {
        return Err(DiffSignalsError::new(format!(
            "git show {path_arg:?} が失敗しました（name-status では新規追加以外と \
             判定されたファイルのため、baseline に存在するはずでした。fail-open な \
             『新規ファイル』への丸めはしません。.claude/rules/security.md A08）: {}",
            output.log_tail()
        )));
    }
    if output.truncated() {
        return Err(DiffSignalsError::new(format!(
            "git show {path_arg:?} の出力が 256 KiB 上限で切り詰められました。構造化出力の \
             先頭側欠落により api_broken を見逃しうるため、fail-open な部分解析はせず拒否します \
             （fail-closed。.claude/rules/security.md A08）"
        )));
    }
    Ok(output.log_tail().to_string())
}

/// `changed_files`（[`list_changed_files`] が返す構造化された変更ファイル
/// 一覧）のうち `.rs` ファイルについて、baseline 時点に存在した公開
/// シグネチャ行（`pub fn`／`pub struct`／`pub enum`）が現在の作業木
/// （`sandbox_root` 配下の実ファイル）から消失していないかを検査する。
/// 新規追加ファイル（`ChangedFile::is_newly_added()`）・新規追加シグネチャ
/// のみの変更は破壊とみなさない（[`DiffSignals::api_broken`] ドキュメント
/// 参照。移植元: `guardrail::checks::api_stability::api_broken`）。
///
/// rename／copy は `ChangedFile::baseline_path()`（旧パス）で baseline
/// 内容を取得し、`ChangedFile::path`（新パス）で現作業木の内容を読む
/// （PR #361 Codex レビュー P1 修正。[`show_file_at_baseline`] ドキュメント
/// 参照）。
///
/// # `.rs` → 非 `.rs` rename の扱い（PR #361 codex-review Medium 指摘対応）
/// 走査対象は「baseline 側パス（`ChangedFile::baseline_path()`）が `.rs`」を
/// 基準にする（新パスのみを見る `filter(|f| f.path.ends_with(".rs"))` では
/// なくなった）。baseline 側が `.rs` で新パスが `.rs` でない rename（例:
/// `src/lib.rs` → `src/lib.txt`）は、新パスがもはやコンパイル対象外である
/// 以上、baseline に公開シグネチャが 1 つでも存在すれば内容を問わず無条件で
/// 破壊とみなす（新パスの内容を読んで比較しない。仮に新パスへ同一テキストを
/// 書き込んだ「内容が同一の rename」であっても、クレートの公開 API 面からは
/// 消失している）。旧実装（`filter(|f| f.path.ends_with(".rs"))`）は新パスが
/// `.rs` でないレコードをそもそも走査対象から除外していたため、この rename
/// で `pub fn` が削除されても `api_broken=false` にすり抜けていた。逆に
/// baseline 側が `.rs` でない（`.rs` → `.rs` の一部を含む新規追加や、
/// 非 Rust ファイルからの rename）場合は従来どおりスキップする
/// （非 Rust ファイルの内容に偶然 `pub fn ` で始まる行があっても誤検知しない）。
///
/// copy（`status` 先頭が `'C'`）で baseline 側 `.rs` を非 `.rs` パスへ複製した
/// 場合も同じ分岐に入り `Ok(true)` になるが、これは複製元 `.rs` 自体は消えず
/// 公開シグネチャも消失していないため理論上は誤検知（false positive）である。
/// [`list_changed_files`] は `git diff --name-status -z`（`-C`／
/// `--find-copies` 未指定）を呼ぶため `'C'` は現状発生しない到達不能ケースだが
/// （`is_plausible_status_token` が `'C'` を形式上許容しているのは将来
/// `-C` 系オプションを追加する余地を残すため）、万一発生しても安全側
/// （破壊あり＝エスカレーション方向）に倒れるだけで A08 違反にはならない。
fn api_signature_touched<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
    changed_files: &[ChangedFile],
) -> Result<bool, DiffSignalsError> {
    for file in changed_files
        .iter()
        .filter(|file| file.baseline_path().ends_with(".rs"))
    {
        if file.is_newly_added() {
            // baseline に対応物がなく、消失し得るシグネチャがない。
            continue;
        }

        let baseline_content =
            show_file_at_baseline(runner, sandbox_root, baseline_commit, file.baseline_path())?;
        let baseline_sigs = extract_public_signatures(&baseline_content);
        if baseline_sigs.is_empty() {
            continue;
        }

        if !file.path.ends_with(".rs") {
            // 新パスが `.rs` でない rename/copy: 新パスはもはやコンパイル
            // 対象外であり、baseline に存在した公開シグネチャは内容を問わず
            // 全て API 面から消失する（本関数ドキュメント「`.rs` → 非 `.rs`
            // rename の扱い」参照）。新パスの内容は読まない。
            return Ok(true);
        }

        // ファイルが現作業木から削除されている場合は空文字列扱い（全シグネ
        // チャが消失＝破壊）。読み取り失敗（権限等）は「消えた」と区別せず
        // 安全側（破壊あり方向）に倒す。`guardrail::checks::api_stability::
        // api_broken` と同一方針。
        let current_content =
            std::fs::read_to_string(sandbox_root.join(&file.path)).unwrap_or_default();
        let current_sigs = extract_public_signatures(&current_content);

        if baseline_sigs.iter().any(|sig| !current_sigs.contains(sig)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// パス文字列が「テストコード」を指すかを判定する。`Path::components()` で
/// `tests` コンポーネントの有無を見るため、`tests/foo.rs`（リポジトリ直下）・
/// `crates/self-repair/tests/foo.rs`（中間パス）のいずれの相対パス表現でも
/// 一致する（`path.contains("/tests/")` は先頭が `/` を含まない `tests/foo.rs`
/// を取りこぼすため使わない）。
fn is_test_path(path: &str) -> bool {
    Path::new(path)
        .components()
        .any(|component| component.as_os_str() == "tests")
        || path.ends_with("_test.rs")
}

/// 変更ファイル一覧に「本番コード」と「テストコード」の双方が同時に含まれる
/// かを検査する（ゲーミング疑いの簡易ヒューリスティック）。移植元:
/// `tests/revalidation_bug_fix.rs::gaming_suspect`。`touches_test` /
/// `touches_prod` は `is_test_path` を共通の判定基準として使うため、
/// ルート直下 `tests/foo.rs` のようなパスが誤って両方 `false`（または
/// `touches_prod` 側で誤って `true`）になるブラインドスポットを持たない。
///
/// rename／copy は現パス（`ChangedFile::path`）と旧パス
/// （`ChangedFile::baseline_path()`）の**両方**を判定対象にする。新パスの
/// みで判定すると、`tests/foo.rs` を `src/foo.rs` へ rename して本番コードを
/// 変更したケースが「テストコードを触れていない」に見えてしまう
/// （逆方向も同様）。
fn gaming_suspect_from_files(changed_files: &[ChangedFile]) -> bool {
    let touches_test = changed_files
        .iter()
        .any(|file| is_test_path(&file.path) || is_test_path(file.baseline_path()));
    let touches_prod = changed_files.iter().any(|file| {
        (file.path.ends_with(".rs") && !is_test_path(&file.path))
            || (file.baseline_path().ends_with(".rs") && !is_test_path(file.baseline_path()))
    });
    touches_test && touches_prod
}

/// `policy_exclusion_path` の `policy-exclusion.toml` を読み込み・パースする。
///
/// # 呼び出し契約（PR #361 codex-review P1 指摘対応）
/// **呼び出し元は候補適用前に一度だけ本関数を呼び、返る [`guardrail::
/// PolicyExclusionConfig`] を試行ループ全体で使い回さなければならない**
/// （[`crate::verify_direct_composite::RepairCompositeGate`] が保持する）。
/// 以前の実装は [`evaluate_exclusion_rules`] 内でこのロードを**試行ごとに**
/// （`RepairCompositeGate::verify` が呼ばれるたび）行っていた。
/// `policy_exclusion_path` の既定値（`main.rs::run_run`）は
/// sandbox 内の書き込み可能なパス（`sandbox_root/policy-exclusion.toml`。
/// `RunSandbox::create` が `git clone --local` で `baseline_commit` 時点の
/// 内容を反映した直後の状態）であり、`sandbox_root` は候補適用対象
/// そのものである。候補が `policy-exclusion.toml` を書き換えて、本来
/// 自身の diff に match して人間承認（エスカレーション）へ回すはずだった
/// ルールを削除・縮小すれば、次回以降の `verify` 呼び出しがその改変済み
/// ルールを読み込み、match を回避してガードレール判定
/// （`GuardrailAdoptionJudge`）を迂回できてしまう（`guardrail::decision::
/// decide` は `exclusion_rule_ids` が 1 件以上あれば無条件でエスカレー
/// ションへ回す「安全側にしか作用しない」設計のため、ルールの**追加**は
/// 判定を緩めない。A08「判定の迂回経路を作らない」違反となるのは削除・
/// 縮小方向のみ）。本関数を「sandbox 構築直後・候補未適用の状態」で
/// 一度だけ呼び、結果を不変値として保持することで、候補による自己免除を
/// 構造的に防ぐ（多重防御のもう一方は `crate::candidate::apply_candidate`
/// によるガードレール設定ファイル自体の書き換え拒否）。
pub fn load_policy_exclusion_config(
    policy_exclusion_path: &Path,
) -> Result<guardrail::PolicyExclusionConfig, DiffSignalsError> {
    let toml = std::fs::read_to_string(policy_exclusion_path).map_err(|error| {
        DiffSignalsError::new(format!(
            "{} の読み込みに失敗しました: {error}",
            policy_exclusion_path.display()
        ))
    })?;
    guardrail::load_policy_exclusion(&toml).map_err(|error| {
        DiffSignalsError::new(format!("policy-exclusion.toml のパースに失敗: {error}"))
    })
}

/// `guardrail` のポリシー除外リスト評価（REQ-5）を `sandbox_root` 上で実行する。
/// `policy_exclusion`（[`load_policy_exclusion_config`] が候補適用前に一度だけ
/// ロードした不変値。[`load_policy_exclusion_config`] doc「呼び出し契約」
/// 参照）と `baseline_commit` から現作業木までの diff を突き合わせて評価する。
/// 移植元: `tests/revalidation_bug_fix.rs::evaluate_exclusion_rules`
/// （ロード部分は [`load_policy_exclusion_config`] へ分離済み。
/// `guardrail::EvaluationContext::from_repo` 自体が内部で git を起動するため、
/// `CommandRunner` 注入はここでは効かない——これは guardrail 側公開 API の
/// 既存契約であり、本モジュールで変更しない）。
fn evaluate_exclusion_rules(
    sandbox_root: &Path,
    baseline_commit: &str,
    policy_exclusion: &guardrail::PolicyExclusionConfig,
) -> Result<Vec<String>, DiffSignalsError> {
    let ctx = guardrail::EvaluationContext::from_repo(sandbox_root, baseline_commit).map_err(
        |error| DiffSignalsError::new(format!("EvaluationContext の構築に失敗: {error}")),
    )?;
    let evaluation = guardrail::ExclusionEvaluation::evaluate(&policy_exclusion.rules, &ctx)
        .map_err(|error| DiffSignalsError::new(format!("除外リスト評価に失敗: {error}")))?;
    Ok(evaluation.effective_rule_ids())
}

/// `sandbox_root` 上で `baseline_commit` からの diff を**試行ごとに**実測し、
/// 4 シグナルをまとめて返す（[`crate::verify_direct_composite::
/// RepairCompositeGate::verify`] から候補適用直後の作業木に対して毎試行
/// 呼ばれる想定。モジュール冒頭ドキュメント参照）。
///
/// `policy_exclusion` はポリシー除外リストの評価にのみ使う不変値であり、
/// 本関数はこれをロードし直さない（[`load_policy_exclusion_config`] doc
/// 「呼び出し契約」参照。呼び出し元が候補適用前に一度だけロードした値を
/// 試行ごとに使い回す）。
///
/// # Errors
///
/// `baseline_commit` の形式検証・git 呼び出し・ポリシー除外リストの評価の
/// いずれかに失敗した場合 [`DiffSignalsError`]（fail-closed。未計測値を
/// 既定値で埋めない）。
pub fn measure_diff_signals<R: CommandRunner>(
    runner: &R,
    sandbox_root: &Path,
    baseline_commit: &str,
    policy_exclusion: &guardrail::PolicyExclusionConfig,
) -> Result<DiffSignals, DiffSignalsError> {
    validate_commit_ref(baseline_commit)?;
    // 未追跡（新規追加）ファイルも `git diff <baseline_commit>` の対象に含める
    // ため、以降の全 diff 計測に先立って index へ反映する（`stage_untracked_files`
    // ドキュメント参照。Codex レビュー #137 指摘）。
    stage_untracked_files(runner, sandbox_root)?;
    let lines_changed = diff_numstat(runner, sandbox_root, baseline_commit)?;
    let changed_files = list_changed_files(runner, sandbox_root, baseline_commit)?;
    let api_broken = api_signature_touched(runner, sandbox_root, baseline_commit, &changed_files)?;
    let gaming_suspect = gaming_suspect_from_files(&changed_files);
    let exclusion_rule_ids =
        evaluate_exclusion_rules(sandbox_root, baseline_commit, policy_exclusion)?;
    Ok(DiffSignals {
        lines_changed,
        api_broken,
        gaming_suspect,
        exclusion_rule_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{CommandOutput, ExecError};
    use std::cell::RefCell;

    /// テスト用に [`ChangedFile`] を組み立てる補助関数。
    fn cf(status: &str, old_path: Option<&str>, path: &str) -> ChangedFile {
        ChangedFile {
            status: status.to_string(),
            old_path: old_path.map(str::to_string),
            path: path.to_string(),
        }
    }

    /// スクリプト化した `CommandRunner` テストダブル。`args` の先頭から
    /// `git diff --numstat ...` / `git diff --name-status -z ...` /
    /// `git show <baseline>:<file>` 等を区別して固定応答を返す
    /// （`verify_gates.rs` のテストダブルと同種の設計）。`show_stdout`／
    /// `show_success` は `git show` 呼び出し全件に共通で返す単一の固定応答
    /// （本モジュールのテストは 1 ファイルのみを対象とするため十分。
    /// `show_success = false` は「baseline にファイルが存在しない」を模擬する
    /// 実測異常ケース用）。
    struct ScriptedGit {
        numstat_stdout: String,
        name_status_stdout: String,
        show_stdout: String,
        show_success: bool,
        calls: RefCell<Vec<Vec<String>>>,
    }

    impl ScriptedGit {
        fn new(numstat_stdout: &str, show_stdout: &str) -> Self {
            ScriptedGit {
                numstat_stdout: numstat_stdout.to_string(),
                name_status_stdout: String::new(),
                show_stdout: show_stdout.to_string(),
                show_success: true,
                calls: RefCell::new(Vec::new()),
            }
        }

        /// `git show` が非 0 終了する（`api_signature_touched` が新規追加と
        /// 誤認せずに実測異常として拒否すべきケース）テストダブルを構築する。
        fn new_show_fails(numstat_stdout: &str) -> Self {
            ScriptedGit {
                numstat_stdout: numstat_stdout.to_string(),
                name_status_stdout: String::new(),
                show_stdout: String::new(),
                show_success: false,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl CommandRunner for ScriptedGit {
        fn run(
            &self,
            _program: &str,
            args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            self.calls
                .borrow_mut()
                .push(args.iter().map(|s| s.to_string()).collect());
            if args.contains(&"--numstat") {
                Ok(CommandOutput::from_captured(
                    true,
                    self.numstat_stdout.clone().into_bytes(),
                ))
            } else if args.contains(&"--name-status") {
                Ok(CommandOutput::from_captured(
                    true,
                    self.name_status_stdout.clone().into_bytes(),
                ))
            } else if args.first() == Some(&"show") {
                Ok(CommandOutput::from_captured(
                    self.show_success,
                    self.show_stdout.clone().into_bytes(),
                ))
            } else {
                Ok(CommandOutput::from_captured(true, Vec::new()))
            }
        }
    }

    /// `git diff --name-status ...` 呼び出しにのみ固定応答を返すテストダブル
    /// （[`list_changed_files`] 単体のパース確認用。他の git 呼び出しは
    /// 到達しない想定のため空成功を返す）。
    struct NameStatusOnly(String);
    impl CommandRunner for NameStatusOnly {
        fn run(
            &self,
            _program: &str,
            args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            if args.contains(&"--name-status") {
                Ok(CommandOutput::from_captured(
                    true,
                    self.0.clone().into_bytes(),
                ))
            } else {
                Ok(CommandOutput::from_captured(true, Vec::new()))
            }
        }
    }

    /// spawn 自体が失敗するテストダブル。
    struct FailingRunner;
    impl CommandRunner for FailingRunner {
        fn run(
            &self,
            _program: &str,
            _args: &[&str],
            _cwd: &Path,
        ) -> Result<CommandOutput, ExecError> {
            Err(ExecError::new("spawn failed (scripted)"))
        }
    }

    #[test]
    fn validate_commit_ref_accepts_valid_short_and_full_sha() {
        assert!(validate_commit_ref("abc1234").is_ok());
        assert!(validate_commit_ref(&"a".repeat(40)).is_ok());
    }

    #[test]
    fn validate_commit_ref_rejects_option_like_value() {
        let err = validate_commit_ref("--upload-pack=evil").expect_err("拒否されるはず");
        assert!(err.message().contains("baseline_commit"));
    }

    #[test]
    fn validate_commit_ref_rejects_uppercase_and_short_values() {
        assert!(validate_commit_ref("ABCDEFA").is_err(), "大文字は拒否");
        assert!(
            validate_commit_ref("abc123").is_err(),
            "6 桁は下限未満で拒否"
        );
    }

    #[test]
    fn diff_numstat_parses_added_and_deleted_columns() {
        let runner = ScriptedGit::new("3\t1\tsrc/lib.rs\n0\t2\ttests/foo.rs\n", "");
        let lines_changed =
            diff_numstat(&runner, Path::new("/sandbox"), "abc1234").expect("解析成功");
        assert_eq!(lines_changed, 6);
    }

    /// rename 表記（`src/{old.rs => new.rs}`）を含む `--numstat` 出力でも、
    /// 1・2 列目（追加行数・削除行数）は rename の影響を受けず正しく合算
    /// されることを確認する（[`diff_numstat`] ドキュメント「rename 表記と
    /// 行数計測の独立性」参照。PR #361 Codex レビュー P1 の修正方針の前提
    /// 確認）。
    #[test]
    fn diff_numstat_sums_added_and_deleted_regardless_of_rename_notation_in_path_column() {
        let runner = ScriptedGit::new("1\t1\tsrc/{old.rs => new.rs}\n", "");
        let lines_changed =
            diff_numstat(&runner, Path::new("/sandbox"), "abc1234").expect("解析成功");
        assert_eq!(lines_changed, 2);
    }

    #[test]
    fn diff_numstat_fails_closed_on_unparseable_column() {
        let runner = ScriptedGit::new("not-a-number\t1\tsrc/lib.rs\n", "");
        let err = diff_numstat(&runner, Path::new("/sandbox"), "abc1234")
            .expect_err("fail-open で 0 に丸めず拒否されるはず");
        assert!(err.message().contains("added"));
    }

    /// 意味論は `guardrail::checks::api_stability::api_broken` と同一（同モジュール
    /// の `removing_pub_fn_is_detected_as_broken`／`adding_new_pub_fn_is_not_broken`
    /// に対応。イシュー #142 差し戻し分の回帰テスト: 旧実装は追加・削除いずれの
    /// `pub fn` 行も一律 `true` としており、機能追加種別の候補が新規 `pub fn` を
    /// 追加しただけで無条件エスカレーションされる誤りがあった）。
    #[test]
    fn api_signature_touched_flags_removed_pub_fn_as_broken() {
        let runner = ScriptedGit::new(
            "1\t1\tsrc/lib.rs\n",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
        );
        let touched = api_signature_touched(
            &runner,
            Path::new("/sandbox"),
            "abc1234",
            &[cf("M", None, "src/lib.rs")],
        )
        .expect("成功");
        assert!(
            touched,
            "baseline の pub fn sub が消えている場合は破壊として検出されるはず"
        );
    }

    #[test]
    fn api_signature_touched_does_not_flag_pure_addition_as_broken() {
        // baseline の `git show` 応答（`pub fn add` のみ）を返し、現作業木
        // （`/sandbox/src/lib.rs`）は実在しないパスのため
        // `std::fs::read_to_string` が失敗し空文字列扱いになる。これでは
        // 「baseline のシグネチャが全て消えた」ことになり破壊として誤検出
        // されてしまうため、`sandbox_root` を実ディレクトリにし、現作業木の
        // 内容（追加後）を実ファイルとして書き込んで検証する。
        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-pure-addition-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        std::fs::create_dir_all(sandbox.join("src")).expect("src ディレクトリ作成に失敗");
        std::fs::write(
            sandbox.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
        )
        .expect("src/lib.rs 書き込み失敗");

        let runner = ScriptedGit::new(
            "1\t0\tsrc/lib.rs\n",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        );
        let touched =
            api_signature_touched(&runner, &sandbox, "abc1234", &[cf("M", None, "src/lib.rs")])
                .expect("成功");
        assert!(
            !touched,
            "既存 pub fn を維持したまま新規 pub fn を追加しただけでは破壊とみなさないはず \
             （guardrail::checks::api_stability::adding_new_pub_fn_is_not_broken と同一意味論）"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }

    #[test]
    fn api_signature_touched_treats_newly_added_status_as_new_file_without_calling_git_show() {
        // status="A"（新規追加）と確認できる場合、baseline にシグネチャを
        // 持ちようがないため `git show` を呼ばずスキップする。
        // `ScriptedGit::new_show_fails` は `git show` が呼ばれると非 0 終了を
        // 返すテストダブルであり、[`show_file_at_baseline`] は非 0 終了を
        // 一律 `Err` にする（PR #361 修正後の契約）。したがって、もし
        // `api_signature_touched` が誤って `git show` を呼び出せば
        // 本テストの `.expect("成功（git show を呼ばずスキップされるはず）")`
        // が `Err` を受け取って panic する。この `.expect()` が通ること自体が
        // 「`git show` を呼んでいない」ことの直接証明になる。
        let runner = ScriptedGit::new_show_fails("5\t0\tsrc/new_api.rs\n");
        let touched = api_signature_touched(
            &runner,
            Path::new("/sandbox"),
            "abc1234",
            &[cf("A", None, "src/new_api.rs")],
        )
        .expect("成功（git show を呼ばずスキップされるはず）");
        assert!(
            !touched,
            "status=A（新規追加）のファイルは破壊とみなさないはず"
        );
    }

    #[test]
    fn api_signature_touched_fails_closed_when_git_show_fails_for_non_added_status() {
        // status="M"（変更）にも関わらず `git show` が非 0 終了する異常系は、
        // 「新規ファイル」への fail-open な丸めをせず [`DiffSignalsError`]
        // として拒否する（PR #361 Codex レビュー P1 修正の中核契約）。
        let runner = ScriptedGit::new_show_fails("1\t0\tsrc/lib.rs\n");
        let err = api_signature_touched(
            &runner,
            Path::new("/sandbox"),
            "abc1234",
            &[cf("M", None, "src/lib.rs")],
        )
        .expect_err("status=M での git show 失敗は fail-closed に拒否されるはず");
        assert!(err.message().contains("git show"));
    }

    /// PR #361 Codex レビュー P1 の中核回帰テスト（要求 (a)）: rename と同時に
    /// baseline に存在した `pub fn` を削除した場合、baseline 側のパス
    /// （`old_path`）から `git show` した内容と比較して破壊として検出される。
    #[test]
    fn api_signature_touched_flags_pub_fn_removed_during_rename_as_broken() {
        let runner = ScriptedGit::new(
            "1\t1\tsrc/old.rs\tsrc/new.rs\n",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
        );
        let touched = api_signature_touched(
            &runner,
            Path::new("/sandbox"),
            "abc1234",
            &[cf("R050", Some("src/old.rs"), "src/new.rs")],
        )
        .expect("成功");
        assert!(
            touched,
            "rename と同時に baseline の pub fn sub が消えている場合は破壊として検出 \
             されるはず（rename 表記を git show のパスへ誤って渡していた旧実装では \
             この破壊が見逃されていた）"
        );
    }

    /// PR #361 Codex レビュー P1 の回帰テスト（要求 (b)）: 内容が同一の純粋な
    /// rename は破壊として誤検出されない。
    #[test]
    fn api_signature_touched_does_not_flag_pure_rename_without_content_change_as_broken() {
        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-pure-rename-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        std::fs::create_dir_all(sandbox.join("src")).expect("src ディレクトリ作成に失敗");
        std::fs::write(
            sandbox.join("src/new.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .expect("src/new.rs 書き込み失敗");

        let runner = ScriptedGit::new(
            "0\t0\tsrc/old.rs\tsrc/new.rs\n",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        );
        let touched = api_signature_touched(
            &runner,
            &sandbox,
            "abc1234",
            &[cf("R100", Some("src/old.rs"), "src/new.rs")],
        )
        .expect("成功");
        assert!(
            !touched,
            "内容が同一の純粋な rename は破壊として誤検出されないはず"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }

    /// PR #361 codex-review Medium 指摘（`Rename API check skips non-rs
    /// paths`）の回帰防止: `.rs` → 非 `.rs` の rename で baseline の `pub fn`
    /// が失われる場合、新パスの内容に関わらず破壊として検出されることを
    /// 確認する。`sandbox` を実ディレクトリにして新パス
    /// （`src/lib.txt`）へ baseline と**同一のテキスト**を書き込む
    /// （advisor 指摘: `Path::new("/sandbox")` のまま `read_to_string` を
    /// 失敗させると「新パスを読んでいないこと」の証明にならない。本テストは
    /// 新パスに `pub fn` を含む同一内容を置いてもなお破壊と判定される
    /// ことを検証することで、「新パスの内容を読まずに無条件で破壊とみなす」
    /// 分岐を通っていることを保証する）。
    #[test]
    fn api_signature_touched_flags_rename_from_rs_to_non_rs_as_broken_even_with_identical_content()
    {
        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-rename-rs-to-non-rs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        std::fs::create_dir_all(sandbox.join("src")).expect("src ディレクトリ作成に失敗");
        std::fs::write(
            sandbox.join("src/lib.txt"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .expect("src/lib.txt 書き込み失敗");

        let runner = ScriptedGit::new(
            "0\t0\tsrc/lib.rs\tsrc/lib.txt\n",
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        );
        let touched = api_signature_touched(
            &runner,
            &sandbox,
            "abc1234",
            &[cf("R100", Some("src/lib.rs"), "src/lib.txt")],
        )
        .expect("成功");
        assert!(
            touched,
            "`.rs` から非 `.rs` への rename は、新パスの内容が baseline と \
             同一であっても公開シグネチャが API 面から消失するため破壊として \
             検出されるはず（新パスの内容を読んで比較する旧実装では見逃されて \
             いた）"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }

    /// 上記と対になる確認: 非 `.rs` → `.rs` の rename は baseline 側
    /// （非 Rust ファイル）に走査対象外として扱われ、破壊として誤検出
    /// されない。
    #[test]
    fn api_signature_touched_does_not_flag_rename_from_non_rs_to_rs() {
        let runner = ScriptedGit::new("0\t0\tsrc/lib.txt\tsrc/lib.rs\n", "");
        let touched = api_signature_touched(
            &runner,
            Path::new("/sandbox"),
            "abc1234",
            &[cf("R100", Some("src/lib.txt"), "src/lib.rs")],
        )
        .expect("成功（baseline 側が非 .rs のため git show を呼ばずスキップされるはず）");
        assert!(
            !touched,
            "非 .rs から .rs への rename は baseline 側が走査対象外のため破壊とみなさないはず"
        );
    }

    // 要求 (c)（新規追加ファイルは従来どおり非破壊扱い）は
    // `api_signature_touched_treats_newly_added_status_as_new_file_without_calling_git_show`
    // が同一シナリオでより強く（`git show` を呼んでいないこと自体を）検証
    // 済みのため、重複するテストはここに置かない。

    #[test]
    fn api_signature_touched_ignores_non_rs_files() {
        let runner = ScriptedGit::new("3\t0\tCargo.toml\n", "");
        let touched = api_signature_touched(
            &runner,
            Path::new("/sandbox"),
            "abc1234",
            &[cf("M", None, "Cargo.toml")],
        )
        .expect("成功");
        assert!(!touched, ".rs 以外のファイルは走査対象外のはず");
    }

    #[test]
    fn gaming_suspect_from_files_true_when_prod_and_test_both_touched() {
        let files = vec![
            cf("M", None, "src/lib.rs"),
            cf("M", None, "tests/foo_test.rs"),
        ];
        assert!(gaming_suspect_from_files(&files));
    }

    #[test]
    fn gaming_suspect_from_files_false_when_only_prod_touched() {
        let files = vec![cf("M", None, "src/lib.rs")];
        assert!(!gaming_suspect_from_files(&files));
    }

    #[test]
    fn gaming_suspect_from_files_true_when_root_level_tests_dir_touched() {
        // リポジトリ直下の `tests/foo.rs`（先頭に `/` が付かない相対パス）は
        // `path.contains("/tests/")` では取りこぼし、かつ `.rs` 拡張子ゆえに
        // `touches_prod` 側で誤って本番コード扱いされていた回帰ケース
        // （PR #355 codex-review 指摘。P1）。
        let files = vec![cf("M", None, "src/lib.rs"), cf("M", None, "tests/foo.rs")];
        assert!(gaming_suspect_from_files(&files));
    }

    #[test]
    fn gaming_suspect_from_files_false_when_only_root_level_tests_dir_touched() {
        let files = vec![cf("M", None, "tests/foo.rs")];
        assert!(!gaming_suspect_from_files(&files));
    }

    /// rename の**旧パス**がテストディレクトリを指す場合も、新パス側のみでは
    /// 判定漏れするブラインドスポットを持たない（[`gaming_suspect_from_files`]
    /// ドキュメント参照）。
    #[test]
    fn gaming_suspect_from_files_true_when_rename_moves_file_out_of_tests_dir() {
        let files = vec![
            cf("R100", Some("tests/foo.rs"), "src/foo.rs"),
            cf("M", None, "src/lib.rs"),
        ];
        assert!(
            gaming_suspect_from_files(&files),
            "tests/ から src/ への rename は touches_test 側で検出されるはず"
        );
    }

    /// [`list_changed_files`]（`git diff --name-status -z`）の rename/copy
    /// レコード（旧パス・新パスの 2 フィールド）のパースを確認する。
    #[test]
    fn list_changed_files_parses_rename_record_with_old_and_new_path() {
        let stdout = "R050\0src/old.rs\0src/new.rs\0";
        let runner = NameStatusOnly(stdout.to_string());
        let files =
            list_changed_files(&runner, Path::new("/sandbox"), "abc1234").expect("解析成功");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].status, "R050");
        assert_eq!(files[0].old_path.as_deref(), Some("src/old.rs"));
        assert_eq!(files[0].path, "src/new.rs");
    }

    #[test]
    fn list_changed_files_parses_non_rename_records_with_single_path() {
        let stdout = "A\0src/new_api.rs\0M\0src/lib.rs\0D\0src/removed.rs\0";
        let runner = NameStatusOnly(stdout.to_string());
        let files =
            list_changed_files(&runner, Path::new("/sandbox"), "abc1234").expect("解析成功");
        assert_eq!(files.len(), 3);
        assert_eq!(files[0].status, "A");
        assert!(files[0].old_path.is_none());
        assert_eq!(files[0].path, "src/new_api.rs");
        assert_eq!(files[1].status, "M");
        assert_eq!(files[1].path, "src/lib.rs");
        assert_eq!(files[2].status, "D");
        assert_eq!(files[2].path, "src/removed.rs");
    }

    #[test]
    fn is_plausible_status_token_accepts_known_statuses_and_rejects_others() {
        assert!(is_plausible_status_token("A"));
        assert!(is_plausible_status_token("M"));
        assert!(is_plausible_status_token("D"));
        assert!(is_plausible_status_token("R050"));
        assert!(is_plausible_status_token("C100"));
        assert!(!is_plausible_status_token(""));
        assert!(!is_plausible_status_token("src/lib.rs"));
        assert!(!is_plausible_status_token("warning:"));
    }

    /// `run_git`（延いては [`list_changed_files`]）は stdout/stderr を結合した
    /// ログを返す（`exec.rs::SystemCommandRunner::run`）。`git add -A`
    /// （`stage_untracked_files`）等が出す警告テキストが `-z` 出力へ NUL
    /// 終端なしで連結されると、警告文の断片がステータストークンとして誤って
    /// 読まれレコードがズレうる。[`is_plausible_status_token`] による検証で
    /// これを fail-closed な `Err` に変えることを確認する（PR #361 Codex
    /// レビュー フォローアップ）。
    #[test]
    fn list_changed_files_fails_closed_on_stderr_warning_mistaken_for_status_token() {
        let stdout = "A\0src/new_api.rs\0warning: LF will be replaced by CRLF\0src/lib.rs\0";
        let runner = NameStatusOnly(stdout.to_string());
        let err = list_changed_files(&runner, Path::new("/sandbox"), "abc1234")
            .expect_err("警告文の混入は解析失敗として拒否されるはず");
        assert!(err.message().contains("warning"));
    }

    /// diff 計測に失敗する経路（不正な `baseline_commit`／spawn 失敗）は
    /// ポリシー除外リスト評価まで到達しないため、ダミーの空ルール
    /// （`std::fs::read_to_string` を経由しない値）で十分（`evaluate_exclusion_rules`
    /// が `policy_exclusion` を使うのは全 diff 計測が成功した最後の手順のみ。
    /// `measure_diff_signals` 本体参照）。
    fn empty_policy_exclusion() -> guardrail::PolicyExclusionConfig {
        guardrail::PolicyExclusionConfig { rules: Vec::new() }
    }

    #[test]
    fn measure_diff_signals_rejects_invalid_baseline_commit_before_spawning() {
        let runner = FailingRunner;
        let err = measure_diff_signals(
            &runner,
            Path::new("/sandbox"),
            "--evil",
            &empty_policy_exclusion(),
        )
        .expect_err("不正な baseline_commit は git 起動前に拒否されるはず");
        assert!(err.message().contains("baseline_commit"));
    }

    #[test]
    fn measure_diff_signals_propagates_spawn_failure() {
        let err = measure_diff_signals(
            &FailingRunner,
            Path::new("/sandbox"),
            "abc1234",
            &empty_policy_exclusion(),
        )
        .expect_err("spawn 失敗は Err として伝播するはず");
        assert!(err.message().contains("起動に失敗"));
    }

    /// `stage_untracked_files` → `diff_numstat`／`api_signature_touched` の
    /// 一連が、実 git リポジトリ上で**未追跡（新規追加）ファイル**を diff
    /// 計測対象に含めることを確認する（Codex レビュー #137 指摘「未追跡
    /// ファイルが差分シグナルから完全に除外される」の再発防止）。
    /// `ScriptedGit` はスクリプト化した固定応答を返すのみで、`git diff
    /// --numstat` が実際に未追跡ファイルを出力しないという実 git の挙動
    /// そのものは検証できないため、`SystemCommandRunner` を使った実 git
    /// リポジトリで検証する。
    #[test]
    fn stage_untracked_files_makes_new_file_visible_to_diff_numstat_and_api_signature_touched() {
        use crate::exec::SystemCommandRunner;

        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-untracked-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(&sandbox).expect("sandbox ディレクトリ作成に失敗");

        let runner = SystemCommandRunner::new();
        let run_ok = |args: &[&str]| {
            let output = runner
                .run("git", args, &sandbox)
                .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗: {error}"));
            assert!(
                output.success(),
                "git {args:?} が失敗しました: {}",
                output.log_tail()
            );
        };

        // baseline commit（追跡ファイルを 1 つ持つ）。
        std::fs::write(sandbox.join("tracked.rs"), "// tracked\n")
            .expect("tracked.rs 書き込み失敗");
        run_ok(&["init", "-q"]);
        run_ok(&["add", "-A"]);
        run_ok(&[
            "-c",
            "user.email=self-repair-137-diff-signals@example.invalid",
            "-c",
            "user.name=self-repair-137-diff-signals",
            "commit",
            "-q",
            "-m",
            "baseline",
        ]);
        let baseline_output = runner
            .run("git", &["rev-parse", "HEAD"], &sandbox)
            .expect("git rev-parse HEAD の起動に失敗");
        assert!(
            baseline_output.success(),
            "git rev-parse HEAD が失敗しました"
        );
        let baseline_commit = baseline_output.log_tail().trim().to_string();

        // 未追跡（新規追加）ファイルを作成する。コミットにもステージングにも
        // 含めない。`pub fn` を含み `api_signature_touched` の検出対象にもする。
        std::fs::create_dir_all(sandbox.join("src")).expect("src ディレクトリ作成に失敗");
        std::fs::write(
            sandbox.join("src/new_api.rs"),
            "pub fn something() -> i32 {\n    1\n}\n",
        )
        .expect("src/new_api.rs 書き込み失敗");

        // stage 前: 実 git の既知の挙動として未追跡ファイルは diff --numstat／
        // --name-status に現れない（このアサーションはテスト対象コードの
        // 前提確認であり、本テストの主眼は stage 後の挙動）。
        let lines_changed_before =
            diff_numstat(&runner, &sandbox, &baseline_commit).expect("diff_numstat 実行に失敗");
        let files_before = list_changed_files(&runner, &sandbox, &baseline_commit)
            .expect("list_changed_files 実行に失敗");
        assert_eq!(
            lines_changed_before, 0,
            "stage 前は未追跡ファイルが diff --numstat に現れないはず（前提確認）"
        );
        assert!(
            files_before.is_empty(),
            "stage 前の変更ファイル一覧は空のはず（前提確認）"
        );

        // `stage_untracked_files` 適用後は新規ファイルが diff 計測対象に入る。
        stage_untracked_files(&runner, &sandbox).expect("stage_untracked_files に失敗");
        let lines_changed_after =
            diff_numstat(&runner, &sandbox, &baseline_commit).expect("diff_numstat 実行に失敗");
        let files_after = list_changed_files(&runner, &sandbox, &baseline_commit)
            .expect("list_changed_files 実行に失敗");
        assert!(
            lines_changed_after > 0,
            "stage 後は新規ファイルの追加行が lines_changed に計上されるはず"
        );
        assert!(
            files_after.iter().any(|file| file.path == "src/new_api.rs"),
            "stage 後は新規ファイルが変更ファイル一覧に含まれるはず: {files_after:?}"
        );
        let api_broken = api_signature_touched(&runner, &sandbox, &baseline_commit, &files_after)
            .expect("api_signature_touched 実行に失敗");
        assert!(
            !api_broken,
            "baseline に存在しない新規ファイルの pub fn 追加は破壊とみなさないはず \
             （guardrail::checks::api_stability::api_broken と同一意味論。イシュー #142 \
             差し戻し分: 本アサーションは旧実装の誤った意味論〈新規ファイルの pub fn も \
             一律検出〉を固定していたため反転した）"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }

    /// PR #361 Codex レビュー P1 の回帰テスト（要求 (a)）: rename と同時に
    /// baseline に存在した `pub fn` を削除した場合、実 git リポジトリ上で
    /// `measure_diff_signals`（公開エントリポイント全体）を通しても
    /// `api_broken=true` になることを確認する。
    ///
    /// 修正前コード（コミット 62815cf）では本テストと同一の操作列で
    /// `api_broken=false`（fail-closed 違反）になることを実行して確認済み
    /// （rename 表記が `git show` のパス引数に渡され、非 0 終了が一律
    /// 「新規ファイル」に丸められていたため）。
    #[test]
    fn measure_diff_signals_detects_pub_fn_removal_hidden_behind_rename() {
        use crate::exec::SystemCommandRunner;

        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-proof-rename-removal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(sandbox.join("src")).expect("sandbox ディレクトリ作成に失敗");

        let runner = SystemCommandRunner::new();
        let run_ok = |args: &[&str]| {
            let output = runner
                .run("git", args, &sandbox)
                .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗: {error}"));
            assert!(
                output.success(),
                "git {args:?} が失敗しました: {}",
                output.log_tail()
            );
        };

        std::fs::write(
            sandbox.join("src/old.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub fn sub(a: i32, b: i32) -> i32 { a - b }\n",
        )
        .expect("src/old.rs 書き込み失敗");
        run_ok(&["init", "-q"]);
        run_ok(&["add", "-A"]);
        run_ok(&[
            "-c",
            "user.email=self-repair-361-diff-signals@example.invalid",
            "-c",
            "user.name=self-repair-361-diff-signals",
            "commit",
            "-q",
            "-m",
            "baseline",
        ]);
        let baseline_output = runner
            .run("git", &["rev-parse", "HEAD"], &sandbox)
            .expect("git rev-parse HEAD の起動に失敗");
        assert!(baseline_output.success(), "git rev-parse HEAD が失敗");
        let baseline_commit = baseline_output.log_tail().trim().to_string();

        // rename しつつ、rename 前に存在した `pub fn sub` を削除する
        // （類似度が rename 検出閾値〈既定 50%〉を上回る範囲で本文を維持）。
        std::fs::remove_file(sandbox.join("src/old.rs")).expect("src/old.rs 削除に失敗");
        std::fs::write(
            sandbox.join("src/new.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .expect("src/new.rs 書き込み失敗");

        let policy_exclusion_path = std::env::current_dir()
            .expect("current_dir 取得に失敗")
            .ancestors()
            .find(|dir| dir.join("policy-exclusion.toml").is_file())
            .map(|dir| dir.join("policy-exclusion.toml"))
            .expect("リポジトリルートの policy-exclusion.toml が見つからないはず");
        let policy_exclusion = load_policy_exclusion_config(&policy_exclusion_path)
            .expect("policy-exclusion.toml のロードに失敗");

        let signals = measure_diff_signals(&runner, &sandbox, &baseline_commit, &policy_exclusion)
            .expect("measure_diff_signals 実行に失敗");
        assert!(
            signals.api_broken,
            "rename と同時に baseline の pub fn sub を削除した場合は api_broken=true \
             になるはず（PR #361 Codex レビュー P1: rename 表記が git show に渡され \
             非 0 終了が一律『新規ファイル』に丸められると、この破壊が見逃される）"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }

    /// PR #361 codex-review P1 の回帰テスト（本体）: `policy_exclusion` を
    /// 候補適用前（本テストでは sandbox 構築直後）に一度だけロードした値は、
    /// sandbox 内の `policy-exclusion.toml` がその後（例えば候補による
    /// 書き換えで）縮小・削除されても `measure_diff_signals` の判定には反映
    /// されないことを確認する。
    ///
    /// 実際の迂回方向はルールの**追加**ではなく**削除・縮小**である
    /// （`guardrail::decision::decide` の判定順序契約: `exclusion_rule_ids`
    /// が 1 件以上あれば機械判定の結果によらず無条件でエスカレーションへ回る
    /// 「安全側にしか作用しない」設計。`crates/guardrail/src/decision.rs`
    /// モジュール冒頭「判定順序の契約」・`crates/guardrail/src/
    /// policy_exclusion/mod.rs` モジュール冒頭参照）。したがって候補にとって
    /// 都合が良い迂回は「本来 match して自身をエスカレーションさせるはずの
    /// ルールを、適用後に消してしまい match させない」方向であり、本テストは
    /// この方向を検証する（ルール追加は match 件数が増えるだけで、エスカレー
    /// ションを強めることはあっても弱めない。「追加すれば自分の diff を除外
    /// できる」という記述は誤りであり、本テストのタイトル・アサーションで
    /// 訂正する）。
    #[test]
    fn measure_diff_signals_uses_policy_exclusion_loaded_before_candidate_application_even_after_sandbox_file_is_narrowed()
     {
        use crate::exec::SystemCommandRunner;

        let sandbox = std::env::temp_dir().join(format!(
            "self-repair-diff-signals-policy-exclusion-fixation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("システム時刻は UNIX_EPOCH 以降のはず")
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(sandbox.join("src")).expect("sandbox ディレクトリ作成に失敗");

        let runner = SystemCommandRunner::new();
        let run_ok = |args: &[&str]| {
            let output = runner
                .run("git", args, &sandbox)
                .unwrap_or_else(|error| panic!("git {args:?} の起動に失敗: {error}"));
            assert!(
                output.success(),
                "git {args:?} が失敗しました: {}",
                output.log_tail()
            );
        };

        // baseline: モデル定義ファイル（除外ルール `arch-hyperparameter-change`
        // 相当の対象パス）を含む。
        std::fs::write(sandbox.join("src/model.rs"), "pub struct Model;\n")
            .expect("src/model.rs 書き込み失敗");
        run_ok(&["init", "-q"]);
        run_ok(&["add", "-A"]);
        run_ok(&[
            "-c",
            "user.email=self-repair-361-policy-exclusion-fixation@example.invalid",
            "-c",
            "user.name=self-repair-361-policy-exclusion-fixation",
            "commit",
            "-q",
            "-m",
            "baseline",
        ]);
        let baseline_output = runner
            .run("git", &["rev-parse", "HEAD"], &sandbox)
            .expect("git rev-parse HEAD の起動に失敗");
        assert!(baseline_output.success(), "git rev-parse HEAD が失敗");
        let baseline_commit = baseline_output.log_tail().trim().to_string();

        // sandbox 内に、モデル定義ファイルへの変更を match する除外ルールを
        // 持つ `policy-exclusion.toml` を置く（候補適用前の信頼済み状態を
        // 模す）。`load_policy_exclusion_config`（→ `guardrail::
        // load_policy_exclusion`）は 3 カテゴリ（architecture_change／
        // test_tolerance_loosening／dependency_change）それぞれ最低 1 件の
        // ルールを要求する（`ensure_required_category_coverage`。#330）ため、
        // 他 2 カテゴリも本テストの diff には match しないパスで満たす。
        let policy_exclusion_path = sandbox.join("policy-exclusion.toml");
        let toml_with_rule = r#"
schema_version = 1

[[exclusion]]
id = "arch-hyperparameter-change"
category = "architecture_change"
description = "model 定義ファイルへの変更"
rationale = "回帰テスト用"
paths = ["**/src/model*.rs"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"

[[exclusion]]
id = "test-tolerance-loosening"
category = "test_tolerance_loosening"
description = "テスト許容誤差の単独緩和（本テストの diff には match しない）"
rationale = "回帰テスト用（schema 要求のダミー）"
paths = ["**/*.rs"]
action = "human_approval"

[exclusion.match]
type = "test_assertion_relaxation_without_prod_change"
assertion_patterns = ["assert!"]

[[exclusion]]
id = "dependency-change"
category = "dependency_change"
description = "依存管理ファイルへの変更（本テストの diff には match しない）"
rationale = "回帰テスト用（schema 要求のダミー）"
paths = ["**/Cargo.toml"]
action = "human_approval"

[exclusion.match]
type = "any_diff_in_paths"
"#;
        std::fs::write(&policy_exclusion_path, toml_with_rule)
            .expect("policy-exclusion.toml（ルールあり）書き込み失敗");

        // 「候補適用前に一度だけロード」を模す: ここで読み込んだ値を以降
        // 不変のまま使い回す（`load_policy_exclusion_config` doc「呼び出し
        // 契約」参照）。
        let policy_exclusion_loaded_before_candidate =
            load_policy_exclusion_config(&policy_exclusion_path)
                .expect("policy-exclusion.toml（ルールあり）のロードに失敗");

        // モデル定義ファイルを変更する（本来なら `arch-hyperparameter-change`
        // に match しエスカレーションされるべき変更）。
        std::fs::write(sandbox.join("src/model.rs"), "pub struct Model2;\n")
            .expect("src/model.rs 変更に失敗");

        // 悪意ある候補が sandbox 内の `policy-exclusion.toml` を書き換え、
        // 自身の変更に match していた `arch-hyperparameter-change` の `paths`
        // を match しないパスへ縮小したことを模す（schema 自体は有効なまま
        // ——3 カテゴリの要求を満たす——にする点が重要: スキーマ違反で
        // `Err` になる「安全に失敗する」書き換えではなく、`Ok` のまま
        // `exclusion_rule_ids` が静かに空になる「危険な」書き換えを再現する）。
        let toml_with_narrowed_rule = toml_with_rule.replace(
            r#"paths = ["**/src/model*.rs"]"#,
            r#"paths = ["**/src/nowhere*.rs"]"#,
        );
        assert_ne!(
            toml_with_narrowed_rule, toml_with_rule,
            "置換対象の paths 行が見つからないはず（テスト前提の確認）"
        );
        std::fs::write(&policy_exclusion_path, &toml_with_narrowed_rule)
            .expect("policy-exclusion.toml（ルール縮小後）への書き換えに失敗");

        // 事前ロード済みの値（ルールあり）を渡して測定する。
        let signals = measure_diff_signals(
            &runner,
            &sandbox,
            &baseline_commit,
            &policy_exclusion_loaded_before_candidate,
        )
        .expect("measure_diff_signals 実行に失敗");

        assert_eq!(
            signals.exclusion_rule_ids,
            vec!["arch-hyperparameter-change".to_string()],
            "sandbox 内の policy-exclusion.toml が縮小されても、候補適用前に \
             ロードした除外ルール（match 済み・本来エスカレーションされる \
             べきルール）がそのまま反映されなければならない（PR #361 \
             codex-review P1 指摘の回帰防止: ファイルパスを渡して verify \
             のたび再読込する実装だと、この時点で exclusion_rule_ids が \
             空になり、候補が自身の変更をエスカレーションから逃れさせられる）"
        );

        let _ = std::fs::remove_dir_all(&sandbox);
    }
}
