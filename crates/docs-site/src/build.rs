//! `nav.toml` の読み込み・検証から出力ディレクトリ作成までのビルドパイプライン枠。
//!
//! # 呼び出し文脈
//!
//! `main.rs`（CLI）から [`build_site`] が呼ばれる。本モジュールは
//! `crate::nav`（[`crate::nav::parse_nav`] / [`crate::nav::validate_sources`]）に
//! 依存し、`<root>/site/nav.toml` を読んでスキーマ検証・source 存在検証まで行う。
//!
//! # スコープ境界（イシュー #869 時点）
//!
//! 本イシューではページの Markdown→HTML 変換・実ファイル書き出しは行わない
//! （検証済みページ数を返すのみ）。実ページ生成・layout・テーマ CSS は兄弟イシュー
//! #870 が [`build_site`] の拡張点として実装する（実装計画 §4「build.rs（拡張点）」）。
//! リポジトリルート実体の `site/nav.toml`・Markdown 原稿は #873 のスコープであり、
//! 本クレート自体は `site/` をリポジトリルートに作らない（テストは `tests/fixtures/`
//! を `--root` で渡す。実装計画 §1）。

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::nav::{self, Nav, NavError};

/// `nav.toml` の相対配置（`repo_root` からの相対パス）。
const NAV_TOML_RELATIVE_PATH: &str = "site/nav.toml";

/// [`build_site`] の失敗理由。[`NavError`] と I/O エラーを包む型付き enum。
/// `Display` に機微情報（入力全文・絶対パス・環境変数）を載せない方針は
/// [`NavError`] 側の契約をそのまま引き継ぐ。
#[derive(Debug)]
pub enum BuildError {
    /// `<root>/site/nav.toml` の読み込みに失敗した（不在・権限不足等）。
    ReadNavToml(std::io::Error),
    /// [`nav::parse_nav`] / [`nav::validate_sources`] のいずれかが失敗した。
    Nav(NavError),
    /// 出力ディレクトリ（`--out`）の作成に失敗した。
    CreateOutDir(std::io::Error),
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
}

/// `<root>/site/nav.toml` を読み込み・パース・検証し、出力ディレクトリ `out` を
/// 作成するビルドパイプラインの枠組み。
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
/// 3. `out` ディレクトリを作成（[`fs::create_dir_all`]。既存でも成功として扱う）
/// 4. 検証済みページ数を含む [`BuildReport`] を返す
///
/// ページ本文の HTML 書き出し・アセット生成は行わない
/// （モジュール冒頭のスコープ境界コメント・#870 参照）。
///
/// # Errors
///
/// `nav.toml` の読み込み失敗・スキーマ／source 検証失敗・出力ディレクトリ作成失敗の
/// いずれかで [`BuildError`] を返す。
pub fn build_site(root: &Path, out: &Path) -> Result<BuildReport, BuildError> {
    let nav_toml_path = root.join(NAV_TOML_RELATIVE_PATH);

    // DoS 抑止（`nav::MAX_INPUT_BYTES`）をこの唯一の FS アクセス経路で実効化する
    // ため、ファイル全体を `fs::read_to_string` で読み切る前に `fs::metadata` で
    // サイズを確認する。`parse_nav` 内の `input.len()` 検査（`nav.rs` 側）は
    // 既にメモリに載った文字列に対する検査であり、この読み込み前チェックとは
    // 独立に維持する（`parse_nav` 単体テストの FS 非依存性を壊さないため）。
    let metadata = fs::metadata(&nav_toml_path).map_err(BuildError::ReadNavToml)?;
    if metadata.len() > nav::MAX_INPUT_BYTES as u64 {
        return Err(BuildError::Nav(NavError::TooLarge));
    }

    let input = fs::read_to_string(&nav_toml_path).map_err(BuildError::ReadNavToml)?;

    let parsed: Nav = nav::parse_nav(&input)?;
    nav::validate_sources(&parsed, root)?;

    fs::create_dir_all(out).map_err(BuildError::CreateOutDir)?;

    let pages = parsed.sections.iter().map(|s| s.pages.len()).sum();
    Ok(BuildReport {
        pages,
        out_dir: out.to_path_buf(),
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
}
