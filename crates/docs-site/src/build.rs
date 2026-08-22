//! `nav.toml` の読み込み・検証から HTML ページ・テーマ CSS の書き出しまでの
//! ビルドパイプライン本体。
//!
//! # 呼び出し文脈
//!
//! `main.rs`（CLI）から [`build_site`] が呼ばれる。本モジュールは
//! `crate::nav`（[`crate::nav::parse_nav`] / [`crate::nav::validate_sources`]）・
//! `crate::markdown`（[`crate::markdown::markdown_to_nodes`]）・
//! `crate::layout`（[`crate::layout::docs_page`]）・`crate::theme`
//! （[`crate::theme::SITE_CSS`]）に依存し、`<root>/site/nav.toml` の読み込みから
//! `<out>` 配下への実 HTML・CSS 書き出しまでを結線する（実装計画 §2.6・#870）。

use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::html;
use crate::layout;
use crate::markdown;
use crate::nav::{self, Nav, NavError};

/// `nav.toml` の相対配置（`repo_root` からの相対パス）。
const NAV_TOML_RELATIVE_PATH: &str = "site/nav.toml";

/// 生成物 HTML の先頭に前置する doctype 宣言（`layout::docs_page` は `<html>` の
/// 中身のみを返すため、書き出し時にここで前置する。実装計画 §2.4 注記）。
const DOCTYPE: &str = "<!DOCTYPE html>\n";

/// `assets/site.css` の出力先（`out` からの相対パス）。
const SITE_CSS_RELATIVE_PATH: &str = "assets/site.css";

/// `page.source`（Markdown 原稿）の入力サイズ上限。[`nav::MAX_INPUT_BYTES`] と
/// 同値を採用する（DoS 抑止の読み込み前実効化。実装計画 §2.6 手順 2）。
const MAX_SOURCE_BYTES: u64 = nav::MAX_INPUT_BYTES as u64;

/// [`build_site`] の失敗理由。[`NavError`] と I/O エラーを包む型付き enum。
/// `Display` に機微情報（入力全文・絶対パス・環境変数）を載せない方針は
/// [`NavError`] 側の契約をそのまま引き継ぐ（`page.source` 等の相対パス文字列は
/// 診断に必要な最小限として含める）。
#[derive(Debug)]
pub enum BuildError {
    /// `<root>/site/nav.toml` の読み込みに失敗した（不在・権限不足等）。
    ReadNavToml(std::io::Error),
    /// [`nav::parse_nav`] / [`nav::validate_sources`] のいずれかが失敗した。
    Nav(NavError),
    /// 出力ディレクトリ（`--out`）の作成に失敗した。
    CreateOutDir(std::io::Error),
    /// `page.source` が `MAX_SOURCE_BYTES`（private 定数）を超過した（読み込み前にサイズ確認で検出）。
    SourceTooLarge(String),
    /// `page.source` の読み込みに失敗した（`validate_sources` 通過後の再読込エラー。
    /// TOCTOU・権限変更等の稀な経路）。
    ReadSource {
        source: String,
        error: std::io::Error,
    },
    /// ページ HTML の書き出し（親ディレクトリ作成含む）に失敗した。
    WritePage { path: String, error: std::io::Error },
    /// `assets/site.css` の書き出しに失敗した。
    WriteAsset(std::io::Error),
    /// 出力先パスがシンボリックリンク経由で `out` 配下の外を指す、または
    /// 既存のシンボリックリンクを上書きしようとした（TOCTOU・意図しない上書き
    /// 対策。レビュー指摘）。
    UnsafeOutputPath(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ReadNavToml(err) => {
                write!(f, "failed to read {NAV_TOML_RELATIVE_PATH}: {err}")
            }
            BuildError::Nav(err) => write!(f, "{err}"),
            BuildError::CreateOutDir(err) => {
                write!(f, "failed to create output directory: {err}")
            }
            BuildError::SourceTooLarge(source) => {
                write!(
                    f,
                    "page.source `{source}` exceeds the {MAX_SOURCE_BYTES} byte size limit"
                )
            }
            BuildError::ReadSource { source, error } => {
                write!(f, "failed to read page.source `{source}`: {error}")
            }
            BuildError::WritePage { path, error } => {
                write!(f, "failed to write page output `{path}`: {error}")
            }
            BuildError::WriteAsset(err) => {
                write!(f, "failed to write {SITE_CSS_RELATIVE_PATH}: {err}")
            }
            BuildError::UnsafeOutputPath(path) => {
                write!(
                    f,
                    "output path `{path}` escapes the output directory or overwrites an existing symlink"
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}

impl From<NavError> for BuildError {
    fn from(err: NavError) -> Self {
        BuildError::Nav(err)
    }
}

/// [`build_site`] 成功時のレポート。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildReport {
    /// `nav.toml` 内で検証済みのページ総数（全セクション合算）。
    pub pages: usize,
    /// 出力先ディレクトリ（`--out` の値をそのまま保持する）。
    pub out_dir: PathBuf,
    /// 実際に書き出した生成物の一覧（`out_dir` からの相対パス。各ページの
    /// `index.html` + `assets/site.css`）。
    pub written: Vec<PathBuf>,
}

/// `page.path`（`/` 始まり・`/` 終わりが [`nav::parse_nav`] で保証済み）から
/// `<out>` 配下の出力ファイルパスを組み立てる。
///
/// `page.path` は常に `/` 始まりであり、`Path::join` は絶対パス相当の
/// コンポーネントを渡すと受け側（`out`）を丸ごと破棄してしまう
/// （`nav.rs` の `looks_like_windows_drive_path` と同種の `Path::join`
/// セマンティクスの罠。レビュー指摘）。よって結合前に必ず先頭の `/` を取り除く。
fn page_output_path(out: &Path, page_path: &str) -> PathBuf {
    let relative = page_path.trim_start_matches('/');
    if relative.is_empty() {
        out.join("index.html")
    } else {
        out.join(relative).join("index.html")
    }
}

/// `path` を開いたうえで、実際に開いた fd（`Metadata` の `(dev, ino)`）が
/// `canonical_path` を検証した際のファイルと同一であることを確認する。
///
/// unix では `std::os::unix::fs::MetadataExt` の `dev()`/`ino()` で fd 起点の
/// アイデンティティを取れる（`fstat` 相当。パス文字列の再解決を経ないため
/// TOCTOU に強い）。`canonicalize` はパス解決のみで実ファイルを開かないため、
/// 「検証したパスの実体」と「実際に読む fd」が同一かどうかまでは保証しない
/// （`canonicalize` の直後に同じ絶対パスがシンボリックリンクへ差し替えられる
/// race）。この関数は開いた fd 側から `(dev, ino)` を取り、`canonical_path` を
/// 独立に再 `canonicalize` した結果と一致することを検査することで、その race を
/// 検知して fail-closed に倒す（レビュー指摘・PR #899）。unix 以外（Windows 等）
/// では `MetadataExt` が使えないため検査を no-op にする（`docs-site` は CI
///〈GitHub ホステッド ubuntu-latest〉・開発者ローカル実行が前提で、Windows は
/// 現状 CI 対象外。`.claude/rules/ci.md` 実機依存節参照）。
#[cfg(unix)]
fn verify_opened_file_matches_canonical_path(
    file: &fs::File,
    canonical_path: &Path,
) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let opened = file.metadata()?;
    let reresolved = canonical_path.canonicalize()?;
    let reresolved_meta = fs::symlink_metadata(&reresolved)?;
    Ok(opened.dev() == reresolved_meta.dev() && opened.ino() == reresolved_meta.ino())
}

#[cfg(not(unix))]
fn verify_opened_file_matches_canonical_path(
    _file: &fs::File,
    _canonical_path: &Path,
) -> std::io::Result<bool> {
    Ok(true)
}

/// `page.source` を `repo_root` 配下であることを再検証したうえで、単一の
/// `File` ハンドルからサイズ確認・内容読み込みの両方を行う。
///
/// [`nav::validate_sources`] は全ページの一括事前検証（早期にわかりやすい
/// エラーを返すため）だが、実際に読み込む直前にも同じ経路のパスを
/// `canonicalize` して `repo_root` 配下であることを**再検証**し、その
/// canonicalize 結果のパスを 1 回だけ `File::open` してサイズ確認・内容読み込み
/// の両方をその同一ハンドルから行う。
///
/// 単一ハンドルへ収斂させても、なお 2 つの残存 race がある（レビュー指摘・
/// PR #899）。それぞれ次の手当てで塞ぐ:
/// 1. `canonicalize` から `File::open` までの間に最終コンポーネントがシンボリック
///    リンクへ差し替えられる race → 開いた fd の `(dev, ino)` を
///    [`verify_opened_file_matches_canonical_path`] で再検証し、不一致なら
///    fail-closed に拒否する。
/// 2. `metadata().len()` でのサイズ確認後、同じハンドルから EOF まで無制限に
///    読み切っていたため、確認後にファイルが増加すると `MAX_SOURCE_BYTES` を
///    実効的に迂回できた → `Read::take` で同一ハンドルから
///    `MAX_SOURCE_BYTES + 1` バイトに上限した bounded read に変更し、
///    上限超過を拒否する。
///
/// 中間コンポーネント（`repo_root` から `canonical_path` までの途中の
/// ディレクトリ）がリンクへ差し替えられる race は、`std` のみでは
/// `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS)` 相当の API がなく解消でき
/// ない。これは `libc`/`nix` 等の追加依存が必要な既知のプラットフォーム制約
/// であり（deps-policy.md: 追加依存はユーザー承認必須）、対応を省略した判断
/// ではなく現状の制約として残す。
fn read_verified_source(
    canonical_root: &Path,
    repo_root: &Path,
    source: &str,
) -> Result<String, BuildError> {
    let candidate = repo_root.join(source);
    let canonical_path = candidate
        .canonicalize()
        .map_err(|error| BuildError::ReadSource {
            source: source.to_string(),
            error,
        })?;
    if !canonical_path.starts_with(canonical_root) {
        return Err(BuildError::Nav(NavError::UnsafeSource(source.to_string())));
    }

    let file = fs::File::open(&canonical_path).map_err(|error| BuildError::ReadSource {
        source: source.to_string(),
        error,
    })?;

    let matches =
        verify_opened_file_matches_canonical_path(&file, &canonical_path).map_err(|error| {
            BuildError::ReadSource {
                source: source.to_string(),
                error,
            }
        })?;
    if !matches {
        return Err(BuildError::Nav(NavError::UnsafeSource(source.to_string())));
    }

    let size = file
        .metadata()
        .map_err(|error| BuildError::ReadSource {
            source: source.to_string(),
            error,
        })?
        .len();
    if size > MAX_SOURCE_BYTES {
        return Err(BuildError::SourceTooLarge(source.to_string()));
    }

    // bounded read: 確認済みサイズを超えてハンドルから読み進めないよう、同一
    // ハンドルからの読み取りそのものに上限 `MAX_SOURCE_BYTES + 1` を課す
    // （fstat 後の追記で無制限読み込みが上限を迂回する経路を塞ぐ）。
    let mut contents = String::new();
    file.take(MAX_SOURCE_BYTES + 1)
        .read_to_string(&mut contents)
        .map_err(|error| BuildError::ReadSource {
            source: source.to_string(),
            error,
        })?;
    if contents.len() as u64 > MAX_SOURCE_BYTES {
        return Err(BuildError::SourceTooLarge(source.to_string()));
    }
    Ok(contents)
}

/// `out` 配下のディレクトリを 1 階層ずつ作成する。`target`（`out.join(...)` で
/// 組み立てられた絶対パス）へ向けて、`canonical_out` を起点に相対コンポーネント
/// を 1 つずつ push・作成・`canonicalize` 再検証してから次の階層へ進む。
///
/// **`fs::create_dir_all` を一括で呼んでから事後に `canonicalize` するのでは
/// 不十分**（レビュー指摘・回帰テストで確認済み）: 中間コンポーネントが `out`
/// 外を指すシンボリックリンクの場合、一括 `create_dir_all` は検証前にそのリンク
/// 先へ実際にディレクトリを作成してしまう（副作用が検査より先に発生する）。
/// 1 階層ごとに「シンボリックリンクでないか確認 → 作成 → canonicalize して
/// `canonical_out` 配下か確認」の順で進めることで、脱出を検知した時点で
/// それ以上ディレクトリを作らずに止められる。
fn create_dir_all_verified(
    out: &Path,
    canonical_out: &Path,
    target: &Path,
) -> Result<(), BuildError> {
    let target_display = target.display().to_string();
    let relative = target.strip_prefix(out).unwrap_or(target);
    let mut current = canonical_out.to_path_buf();
    for component in relative.components() {
        current.push(component);
        if let Ok(existing) = fs::symlink_metadata(&current)
            && existing.file_type().is_symlink()
        {
            return Err(BuildError::UnsafeOutputPath(target_display));
        }
        match fs::create_dir(&current) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(BuildError::WritePage {
                    path: target_display,
                    error,
                });
            }
        }
        current = current
            .canonicalize()
            .map_err(|error| BuildError::WritePage {
                path: target_display.clone(),
                error,
            })?;
        if !current.starts_with(canonical_out) {
            return Err(BuildError::UnsafeOutputPath(target_display));
        }
    }
    Ok(())
}

/// `path` の親ディレクトリを作成してからファイルへ書き出す。
///
/// 書き出し前に 2 点を検査する（レビュー指摘: 出力先の既存シンボリックリンク
/// を検査せず `create_dir_all`/`fs::write` していたため、`out` 配下外への
/// 書き込み・既存ファイルの意図しない上書きが可能だった）。
/// - 親ディレクトリの作成を [`create_dir_all_verified`] へ委ね、1 階層ごとに
///   `canonical_out`（`out` の正規化パス）配下であることを確認する（中間
///   コンポーネントがシンボリックリンクで外を指す場合を、実際に作成する前に
///   拒否する）
/// - 書き込み先が既存のシンボリックリンクでないことを `fs::symlink_metadata`
///   （symlink を辿らない）で確認する（既存シンボリックリンク経由の意図しない
///   上書きを拒否する）
///
/// `page_output_path` は `page.path` の安全な形式検証（`nav::validate_page_path`。
/// 英数字・`-`・`_` のみのセグメント）済みの値のみを結合するため通常は `out`
/// 内に収まるが、`out` が使い回されるビルド環境等で事前に配置されたシンボリック
/// リンクによる脱出をここで別途防ぐ。
fn write_file_creating_parent(
    out: &Path,
    canonical_out: &Path,
    path: &Path,
    contents: &str,
) -> Result<(), BuildError> {
    let path_display = path.display().to_string();
    if let Some(parent) = path.parent() {
        create_dir_all_verified(out, canonical_out, parent)?;
    }
    if let Ok(existing) = fs::symlink_metadata(path)
        && existing.file_type().is_symlink()
    {
        return Err(BuildError::UnsafeOutputPath(path_display));
    }

    // 上の `symlink_metadata` 検査から実際の書き込みまでの間に、当該パスへ
    // シンボリックリンクを差し替えられる race がなお残る（レビュー指摘・
    // PR #899）。`fs::write` はリンクを辿って追従先を上書きしてしまうため、
    // 検査と書き込みを不可分にする代わりに次の手順で closed に倒す:
    // 1. `remove_file`（`NotFound` は無視）でそのパス上のエントリ自体を
    //    unlink する（再ビルドで既存の通常ファイルを上書きする通常経路を
    //    保つ。symlink であっても unlink はリンク自体を消すだけで追従先には
    //    触れない）
    // 2. `OpenOptions::create_new(true)` で新規作成する。POSIX の
    //    `O_CREAT|O_EXCL` は対象パスが（シンボリックリンクを含め）既に何かを
    //    指している場合 `EEXIST` で失敗するため、手順 1 と 2 の間で誰かが
    //    そのパスへシンボリックリンクを再作成しても、その追従先へは絶対に
    //    書き込まず fail-closed にエラーを返す（`std` のみで実現できる
    //    atomic な原始命令。`custom_flags(O_NOFOLLOW)` のような手書き
    //    プラットフォーム定数を要らない）。
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(BuildError::WritePage {
                path: path_display,
                error,
            });
        }
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| BuildError::WritePage {
            path: path_display.clone(),
            error,
        })?;
    std::io::Write::write_all(&mut file, contents.as_bytes()).map_err(|error| {
        BuildError::WritePage {
            path: path_display,
            error,
        }
    })
}

/// `<root>/site/nav.toml` を読み込み・パース・検証し、各ページを Markdown→HTML
/// 変換したうえで `out` 配下へ実ファイルとして書き出すビルドパイプライン。
///
/// 手順:
/// 1. `<root>/site/nav.toml` を読む（不在・読み込み失敗はエラー）。
///    [`nav::MAX_INPUT_BYTES`] 超過は、DoS 抑止の実効性を保つため
///    `fs::metadata` でファイルサイズを見てから `fs::read_to_string` する
///    （超過ファイル全体をメモリに読み切ってから [`nav::parse_nav`] 内で
///    拒否する経路だと、この唯一の FS アクセス経路では「読み込み前」の
///    抑止が効かない。レビュー指摘）
/// 2. [`nav::parse_nav`] でスキーマ検証 → [`nav::validate_sources`] で
///    `page.source` の実ファイル存在検証
/// 3. 各 `page.source` を（読み込み前サイズ検査つきで）読み、
///    [`markdown::markdown_to_nodes`] → [`layout::docs_page`] → [`html::render`]
///    の順で HTML 文字列へ変換する。**全ページの変換をメモリ上で終えてから**
///    書き出しを開始する（I/O エラー発生時に部分生成物を減らす安全側の順序。
///    完全な fail-closed 原子性は #872 linkcheck の責務）
/// 4. `out` ディレクトリを作成し、各ページを `<out>{page.path}index.html` へ、
///    テーマ CSS を `<out>/assets/site.css` へ書き出す
/// 5. 書き出し件数を含む [`BuildReport`] を返す
///
/// # Errors
///
/// `nav.toml` の読み込み失敗・スキーマ／source 検証失敗・`page.source` の
/// サイズ超過／読み込み失敗・出力ディレクトリ作成失敗・書き出し失敗のいずれかで
/// [`BuildError`] を返す。
pub fn build_site(root: &Path, out: &Path) -> Result<BuildReport, BuildError> {
    let nav_toml_path = root.join(NAV_TOML_RELATIVE_PATH);

    // DoS 抑止（`nav::MAX_INPUT_BYTES`）をこの唯一の FS アクセス経路で実効化する
    // ため、ファイル全体を読み切る前にサイズを確認する。`parse_nav` 内の
    // `input.len()` 検査（`nav.rs` 側）は既にメモリに載った文字列に対する検査で
    // あり、この読み込み前チェックとは独立に維持する（`parse_nav` 単体テストの
    // FS 非依存性を壊さないため）。
    //
    // `fs::metadata(path)` と `fs::read_to_string(path)` は同じパスを別々に
    // 再オープンする 2 回の FS アクセスであり、その間にパスがシンボリック
    // リンクへ差し替えられる・確認後にファイルが増加するという 2 種の TOCTOU が
    // 残っていた（レビュー指摘・PR #899）。単一の `File` ハンドルへ開き直し、
    // 同一ハンドルの `metadata()`（早期棄却用）→ `Read::take` による
    // `MAX_INPUT_BYTES + 1` 上限の bounded read（確認後の増加を無視できない
    // ようにする）の順に変更する。
    let nav_toml_file = fs::File::open(&nav_toml_path).map_err(BuildError::ReadNavToml)?;
    let metadata = nav_toml_file.metadata().map_err(BuildError::ReadNavToml)?;
    if metadata.len() > nav::MAX_INPUT_BYTES as u64 {
        return Err(BuildError::Nav(NavError::TooLarge));
    }
    let mut input = String::new();
    nav_toml_file
        .take(nav::MAX_INPUT_BYTES as u64 + 1)
        .read_to_string(&mut input)
        .map_err(BuildError::ReadNavToml)?;
    if input.len() as u64 > nav::MAX_INPUT_BYTES as u64 {
        return Err(BuildError::Nav(NavError::TooLarge));
    }

    let parsed: Nav = nav::parse_nav(&input)?;
    nav::validate_sources(&parsed, root)?;

    // 手順 3 の読み込み直前の再検証（TOCTOU 対策。`read_verified_source`
    // モジュールコメント参照）に使う `root` の正規化パス。`validate_sources` が
    // 直前に同じ `canonicalize` に成功しているため、通常経路でここが失敗する
    // ことはない。
    let canonical_root = root
        .canonicalize()
        .map_err(|_| BuildError::Nav(NavError::MissingSource(root.display().to_string())))?;

    // 手順 3: 全ページを先にメモリ上でレンダリングし切ってから書き出す。
    let mut rendered_pages: Vec<(PathBuf, String)> = Vec::new();
    for section in &parsed.sections {
        for page in &section.pages {
            let markdown_input = read_verified_source(&canonical_root, root, &page.source)?;

            let body = markdown::markdown_to_nodes(&markdown_input);
            let page_node = layout::docs_page(&parsed, &page.title, &page.path, body);
            let html_out = format!("{DOCTYPE}{}", html::render(&page_node));

            let out_path = page_output_path(out, &page.path);
            rendered_pages.push((out_path, html_out));
        }
    }

    // 手順 4: 出力ディレクトリ作成 → ページ書き出し → テーマ CSS 書き出し。
    fs::create_dir_all(out).map_err(BuildError::CreateOutDir)?;
    let canonical_out = out.canonicalize().map_err(BuildError::CreateOutDir)?;

    let mut written = Vec::with_capacity(rendered_pages.len() + 1);
    for (out_path, html_out) in &rendered_pages {
        write_file_creating_parent(out, &canonical_out, out_path, html_out)?;
        if let Ok(relative) = out_path.strip_prefix(out) {
            written.push(relative.to_path_buf());
        }
    }

    let css_path = out.join(SITE_CSS_RELATIVE_PATH);
    write_file_creating_parent(out, &canonical_out, &css_path, crate::theme::SITE_CSS).map_err(
        |error| match error {
            // ページ書き出し（`WritePage`）と同じヘルパーを使うため、この呼び出し
            // 元だけは `assets/site.css` 固有の `WriteAsset` へ詰め替える
            // （Bugbot 指摘・PR #899: 詰め替えないとアセット I/O 失敗がページ
            // 書き込みエラーとして誤報告され、`BuildError::WriteAsset` が
            // 到達不能なデッドコードのままになる）。
            BuildError::WritePage { error, .. } => BuildError::WriteAsset(error),
            other => other,
        },
    )?;
    written.push(PathBuf::from(SITE_CSS_RELATIVE_PATH));

    let pages = parsed.sections.iter().map(|s| s.pages.len()).sum();
    Ok(BuildReport {
        pages,
        out_dir: out.to_path_buf(),
        written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用の一時ディレクトリ。`Drop` でベストエフォート削除する。
    /// 外部クレート（`tempfile` 等）を追加せず `std::env::temp_dir()` +
    /// プロセス固有サフィックスで代用する（REQ-1 v2: 外部依存ゼロを維持する）。
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "rust-ai-library-docs-site-build-test-{tag}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temp dir for build.rs test");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// レビュー指摘（PR #899）の回帰テスト: `read_verified_source` 自体が、
    /// 読み込み直前の再検証で `repo_root` 配下外へのシンボリックリンク脱出を
    /// 拒否することを確認する。`build_site` 経由では [`nav::validate_sources`]
    /// が先に同じ脱出を検出するため（`nav::tests::validate_sources_rejects_symlink_escape`
    /// が既にその経路をカバー済み）、`read_verified_source` の再検証ロジック
    /// 自体は `build_site` からは到達できない。直接呼び出して単体で確認する
    /// （symlink を使わない環境〈Windows 等〉では対象外のため unix 限定）。
    #[cfg(unix)]
    #[test]
    fn read_verified_source_rejects_symlink_escaping_repo_root() {
        let root = TempDir::new("read-verified-escape-root");
        let outside = TempDir::new("read-verified-escape-outside");
        let secret = outside.0.join("secret.md");
        fs::write(&secret, b"outside repo_root").expect("write fixture outside repo_root");
        std::os::unix::fs::symlink(&secret, root.0.join("linked.md"))
            .expect("create symlink escaping repo_root for test fixture");

        let canonical_root = root.0.canonicalize().expect("canonicalize repo_root");
        match read_verified_source(&canonical_root, &root.0, "linked.md") {
            Err(BuildError::Nav(NavError::UnsafeSource(source))) => {
                assert_eq!(source, "linked.md");
            }
            other => panic!("expected UnsafeSource for symlink escape, got {other:?}"),
        }
    }

    #[test]
    fn build_site_succeeds_for_valid_fixture_root() {
        let root = TempDir::new("build-valid");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/intro/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/intro.md"), b"# Intro").unwrap();

        let out = TempDir::new("build-valid-out");
        let report = build_site(&root.0, &out.0).expect("build should succeed");
        assert_eq!(report.pages, 1);
        assert!(out.0.is_dir());
        assert!(out.0.join("intro/index.html").is_file());
        assert!(out.0.join("assets/site.css").is_file());
        let html = fs::read_to_string(out.0.join("intro/index.html")).unwrap();
        assert!(html.starts_with("<!DOCTYPE html>\n"));
        assert!(html.contains("<h1>Intro</h1>"));
        assert!(html.contains("assets/site.css"));
    }

    #[test]
    fn build_site_writes_root_page_path_directly_under_out_without_escaping() {
        // レビュー指摘の回帰テスト: `page.path` は必ず `/` 始まりのため、
        // `Path::join` の絶対パスセマンティクスに任せると `out` を丸ごと破棄して
        // ファイルシステムルートへ書き出してしまう（`page_output_path` が
        // 先頭 `/` を除去してから結合することの検証）。
        let root = TempDir::new("build-root-path");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Top"
source = "site/index.md"
path = "/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/index.md"), b"# Top").unwrap();

        let out = TempDir::new("build-root-path-out");
        let report = build_site(&root.0, &out.0).expect("build should succeed");
        let written_path = out.0.join("index.html");
        assert!(written_path.is_file());
        assert!(written_path.starts_with(&out.0));
        assert!(report.written.contains(&PathBuf::from("index.html")));
    }

    #[test]
    fn build_site_reports_all_written_page_files_and_theme_css() {
        let root = TempDir::new("build-report-written");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/guide/intro/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/intro.md"), b"# Intro").unwrap();

        let out = TempDir::new("build-report-written-out");
        let report = build_site(&root.0, &out.0).expect("build should succeed");
        assert!(
            report
                .written
                .contains(&PathBuf::from("guide/intro/index.html"))
        );
        assert!(report.written.contains(&PathBuf::from("assets/site.css")));
    }

    #[test]
    fn build_site_rejects_oversized_page_source_without_reading_it_fully() {
        let root = TempDir::new("build-oversized-source");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/intro/"
"#,
        )
        .unwrap();
        let oversized = "a".repeat((MAX_SOURCE_BYTES + 1) as usize);
        fs::write(root.0.join("site/intro.md"), oversized).unwrap();

        let out = TempDir::new("build-oversized-source-out");
        match build_site(&root.0, &out.0) {
            Err(BuildError::SourceTooLarge(source)) => assert_eq!(source, "site/intro.md"),
            other => panic!("expected SourceTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn build_site_fails_when_nav_toml_missing() {
        let root = TempDir::new("build-missing-nav");
        let out = TempDir::new("build-missing-nav-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::ReadNavToml(_))
        ));
    }

    #[test]
    fn build_site_fails_when_source_missing() {
        let root = TempDir::new("build-missing-source");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/does-not-exist.md"
path = "/intro/"
"#,
        )
        .unwrap();

        let out = TempDir::new("build-missing-source-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::Nav(NavError::MissingSource(_)))
        ));
    }

    #[test]
    fn build_site_fails_on_invalid_nav_schema() {
        let root = TempDir::new("build-invalid-schema");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(root.0.join("site/nav.toml"), "not valid toml subset\n").unwrap();

        let out = TempDir::new("build-invalid-schema-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::Nav(NavError::Parse { .. }))
        ));
    }

    #[test]
    fn build_site_rejects_oversized_nav_toml_without_reading_it_fully() {
        // `fs::metadata` によるサイズ確認が `fs::read_to_string` の前段で効いて
        // いることの回帰テスト（レビュー指摘: DoS 抑止の「読み込み前」実効化）。
        let root = TempDir::new("build-oversized-nav");
        fs::create_dir_all(root.0.join("site")).unwrap();
        let mut oversized = String::from("[site]\ntitle = \"");
        oversized.push_str(&"a".repeat(nav::MAX_INPUT_BYTES + 1));
        oversized.push_str("\"\nbase_path = \"\"\n");
        fs::write(root.0.join("site/nav.toml"), oversized).unwrap();

        let out = TempDir::new("build-oversized-nav-out");
        assert!(matches!(
            build_site(&root.0, &out.0),
            Err(BuildError::Nav(NavError::TooLarge))
        ));
    }

    /// レビュー指摘（PR #899）の回帰テスト: 出力先の親ディレクトリコンポーネント
    /// がシンボリックリンクで `out` 外を指す場合、`write_file_creating_parent` が
    /// `UnsafeOutputPath` で fail-closed に拒否し、リンク先へ書き込まないことを
    /// 確認する（symlink を使わない環境〈Windows 等〉では対象外のため unix 限定）。
    #[cfg(unix)]
    #[test]
    fn build_site_rejects_output_path_escaping_via_symlinked_parent() {
        let root = TempDir::new("build-out-escape-root");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/guide/intro/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/intro.md"), b"# Intro").unwrap();

        let out = TempDir::new("build-out-escape-out");
        let outside = TempDir::new("build-out-escape-outside");
        // `out/guide` を `out` の外（`outside`）を指すシンボリックリンクへ差し
        // 替える。`page_output_path` 自体は安全な `page.path` からしか組み立てら
        // れないが、`out` が使い回されるビルド環境で事前に配置されたリンクを
        // 想定した攻撃シナリオを再現する。
        fs::create_dir_all(&out.0).unwrap();
        std::os::unix::fs::symlink(&outside.0, out.0.join("guide"))
            .expect("create symlinked parent directory escaping out");

        match build_site(&root.0, &out.0) {
            Err(BuildError::UnsafeOutputPath(_)) => {}
            other => panic!("expected UnsafeOutputPath for symlinked parent, got {other:?}"),
        }
        assert!(
            !outside.0.join("intro/index.html").exists(),
            "must not write through the symlinked parent into the outside directory"
        );
        // `create_dir_all(parent)` 自体がリンク先へディレクトリを作ってしまう
        // （書き込みより前に副作用が発生する）ケースも合わせて拒否できている
        // ことを確認する: `intro/` サブディレクトリの作成もリンク先で起きては
        // ならない（レビュー指摘の再発防止。advisor 指摘）。
        assert!(
            !outside.0.join("intro").exists(),
            "must not create directories through the symlinked parent into the outside directory"
        );
    }

    /// レビュー指摘（PR #899）の回帰テスト: 出力先ファイル自体が既存の
    /// シンボリックリンクの場合、`write_file_creating_parent` が
    /// `UnsafeOutputPath` で拒否し、リンク先を意図せず上書きしないことを確認する
    /// （unix 限定）。
    #[cfg(unix)]
    #[test]
    fn build_site_rejects_overwriting_existing_symlink_at_output_path() {
        let root = TempDir::new("build-out-symlink-file-root");
        fs::create_dir_all(root.0.join("site")).unwrap();
        fs::write(
            root.0.join("site/nav.toml"),
            r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/intro/"
"#,
        )
        .unwrap();
        fs::write(root.0.join("site/intro.md"), b"# Intro").unwrap();

        let out = TempDir::new("build-out-symlink-file-out");
        let victim_dir = TempDir::new("build-out-symlink-file-victim");
        let victim = victim_dir.0.join("victim.html");
        fs::write(&victim, b"do not overwrite me").unwrap();

        fs::create_dir_all(out.0.join("intro")).unwrap();
        std::os::unix::fs::symlink(&victim, out.0.join("intro/index.html"))
            .expect("create symlink at the exact output file path");

        match build_site(&root.0, &out.0) {
            Err(BuildError::UnsafeOutputPath(_)) => {}
            other => panic!("expected UnsafeOutputPath for existing symlink, got {other:?}"),
        }
        assert_eq!(
            fs::read_to_string(&victim).unwrap(),
            "do not overwrite me",
            "the symlink target must not be overwritten"
        );
    }
}
