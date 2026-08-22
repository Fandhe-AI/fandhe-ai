//! `nav.toml` の読み込み・検証から HTML ページ・テーマ CSS の書き出しまでの
//! ビルドパイプライン本体。
//!
//! # 呼び出し文脈
//!
//! `main.rs`（CLI）から [`build_site`] が呼ばれる。本モジュールは
//! `crate::nav`（[`crate::nav::parse_nav`] / [`crate::nav::validate_sources`]）・
//! `crate::markdown`（`crate::markdown::markdown_to_nodes`）・
//! `crate::layout`（`crate::layout::docs_page`）・`crate::theme`
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

/// fd 相対（`openat`/`mkdirat`/`renameat`、いずれも `O_NOFOLLOW`）でのファイル
/// システム走査。`canonicalize` 後に同じパス文字列を再解決する経路をすべて
/// 排除し、検証済みディレクトリ fd から一切名前解決をやり直さないことで、
/// 中間コンポーネント（`repo_root`/`out` から目的のパスまでの途中の
/// ディレクトリ）のシンボリックリンク差し替え TOCTOU を構造的に閉じる
/// （codex-review 追加ラウンド指摘・PR #899: `canonicalize` + `starts_with` に
/// よる事前検証は、その後の `File::open`/`fs::rename` がパス文字列で経路を
/// 再解決するため、検証と実際のアクセスの間で中間ディレクトリが symlink へ
/// 差し替えられる race を検知できなかった）。
///
/// - `openat(dirfd, name, O_NOFOLLOW, ...)` はカーネル側で `name` がその
///   ディレクトリ直下でシンボリックリンクなら `ELOOP` で拒否する
///   （`fd_walk::is_symlink_rejection` が errno を見て判定する。このツール
///   チェーンでは `std::io::ErrorKind::FilesystemLoop` が `io_error_more`
///   〈issue #86442〉未安定のため使えない）。「シンボリックリンクでないか
///   確認 → 開く」を名前の再解決を挟まない 1 回のシステムコールへ不可分化
///   できる
/// - `renameat(dirfd, old, dirfd, new)` は同一の検証済み dirfd を指定するため、
///   置換先の親ディレクトリを名前で再解決しない（`fs::rename` は絶対/相対
///   パス文字列を毎回解決し直すため、置換先の親が検証後に差し替えられていて
///   も検知できなかった）
/// - Windows 等 unix 以外は `openat` 相当の fd 相対 API がなく本ウォークを
///   実施できないため fail-closed で拒否する（レビュー指摘のとおり、検証不能
///   なプラットフォームでは許可しない）。`docs-site` は CI（GitHub ホステッド
///   ubuntu-latest）・開発者ローカル実行（Linux/macOS）が前提のため実害はない
///   （`.claude/rules/ci.md` 実機依存節）
///
/// 追加 Cargo 依存は発生しない: `libc`/`nix` 等の crate を追加せず、`std` が
/// 既にリンクする system libc の安定 C ABI 関数（`openat`・`mkdirat`・
/// `renameat`）を直接 `extern "C"` 宣言する。`docs-site` の「外部依存ゼロ」
/// 方針（`Cargo.toml` コメント参照）・deps-policy.md の許容依存 9 区分に対する
/// 新規追加のいずれにも該当しない。
// SAFETY（モジュール全体）: 本モジュールは `docs-site` 唯一の FFI 境界であり、
// `lib.rs` の `#![deny(unsafe_code)]` に対してここだけ `#[allow(unsafe_code)]`
// する（PR #899 codex-review 追加ラウンド P0 x2: `canonicalize` 等のパス文字列
// ベースの検証では中間ディレクトリのシンボリックリンク差し替え TOCTOU を
// 防げないため、`std` にない fd 相対 `openat`/`mkdirat`/`renameat`/`unlinkat`
// を直接 FFI 宣言する。`lib.rs` の「unsafe の使用範囲」節参照）。
#[cfg(unix)]
#[allow(unsafe_code)]
mod fd_walk {
    use std::ffi::{CString, OsStr};
    use std::fs::File;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::raw::{c_char, c_int};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Component, Path};

    // `mkdirat` の `mode` は固定引数であり、`mode_t` の実体幅は OS ごとに
    // 異なる（Linux は `unsigned int`〈32 bit〉、macOS/Darwin は
    // `unsigned short`〈16 bit〉）ため、固定引数の呼び出し規約不一致を避ける
    // べく OS ごとに正しい幅で宣言する。本モジュールで渡す値（`0o755` 等）は
    // 16 bit に収まるため実害はない。
    //
    // 対して `open`/`openat` は POSIX 上 `mode` を**可変長引数**として取る
    // （`int open(const char *path, int oflag, ... /* mode_t mode */);`）。
    // 可変長引数は C の既定引数昇格により常に `int` 幅へ昇格されて渡される
    // （実際の `mode_t` 幅に関係なく）。Rust 側が `mode` を固定引数として
    // `mode_t`〈macOS では 16 bit〉幅で宣言すると、Apple の AArch64 ABI は
    // 固定引数をレジスタ渡し・可変長引数をスタック渡しで扱う点が標準 AAPCS64
    // と異なるため、実体は可変長引数として読み出す `openat` 側の実装と
    // 呼び出し規約が食い違い、渡した mode が破壊される（実測: Apple Silicon
    // 上で `0o644` を渡したはずが生成ファイルが `----r-----` になる実害を
    // 確認済み）。よって `openat` は Rust の可変長引数 extern 宣言
    // （`...`）で宣言し、呼び出し側は `mode` を `c_uint`〈常に int 幅〉で渡す
    // ことで C の既定引数昇格と一致させる。
    #[cfg(target_os = "linux")]
    type ModeT = u32;
    #[cfg(target_os = "macos")]
    type ModeT = u16;

    // edition 2024 は `extern "C"` ブロック自体に `unsafe` 修飾を要求する
    // （宣言した関数シグネチャの正しさを保証しないため）。
    //
    // SAFETY: 以下は POSIX/glibc・macOS libSystem が安定 ABI として提供する
    // 標準 C 関数（`openat(2)`・`mkdirat(2)`・`renameat(2)`・`unlinkat(2)`）の
    // 宣言であり、`std` が既にリンクする system libc 以外の依存を要求しない
    // （追加 Cargo 依存なし。上記コメント参照）。安全性の根拠:
    // - シグネチャは各関数の POSIX 宣言と一致させている。`openat` のみ
    //   `mode` を可変長引数（`...`）として宣言し、呼び出し側（`openat_raw`）は
    //   C の既定引数昇格と同じ `c_uint`（int 幅）で渡す。`mkdirat`/`renameat`/
    //   `unlinkat` は固定引数のみで可変長引数を持たないため通常の Rust
    //   extern 宣言で ABI が一致する（可変長引数と固定引数を取り違えた場合の
    //   実害〈Apple AArch64 での `mode` 破壊〉は上記コメントで実測済み）
    // - `ModeT`（`mkdirat` の `mode` 型）は OS ごとの実体幅（Linux
    //   `unsigned int`・macOS `unsigned short`）で個別に宣言しており、本
    //   モジュールが渡す値（`0o755`・`0o644`）はいずれの幅にも収まる
    // - 呼び出し側（`openat_raw`・`mkdirat_if_missing`・`rename_beneath`・
    //   `remove_beneath`）は常に呼び出し元が所有する有効な dirfd と、
    //   `component_cstring` で NUL 終端検証済みの単一パスコンポーネント
    //   （`/` を含まない）のみを渡す（各関数呼び出し箇所の SAFETY コメント
    //   参照）。関数宣言自体はこれらの契約を型で強制できないため、契約の
    //   遵守は呼び出し側の責務とする
    unsafe extern "C" {
        fn openat(dirfd: c_int, pathname: *const c_char, flags: c_int, ...) -> c_int;
        fn mkdirat(dirfd: c_int, pathname: *const c_char, mode: ModeT) -> c_int;
        fn renameat(
            olddirfd: c_int,
            oldpath: *const c_char,
            newdirfd: c_int,
            newpath: *const c_char,
        ) -> c_int;
        fn unlinkat(dirfd: c_int, pathname: *const c_char, flags: c_int) -> c_int;
    }

    /// 対象 OS の `ELOOP`（symlink 解決を `O_NOFOLLOW` が拒否した際の errno）。
    /// `std::io::ErrorKind::FilesystemLoop` はこのツールチェーン（`rust-toolchain.toml`
    /// の stable）では `io_error_more`（issue #86442）が未安定のため使えず、
    /// `raw_os_error()` を直接比較する。
    #[cfg(target_os = "linux")]
    const ELOOP: i32 = 40;
    #[cfg(target_os = "macos")]
    const ELOOP: i32 = 62;

    /// `ENOTDIR`。`open_subdir`（`O_DIRECTORY | O_NOFOLLOW`）で対象が
    /// シンボリックリンクの場合、`O_DIRECTORY` は「最終コンポーネントは
    /// ディレクトリでなければならない」という制約を課すため、`O_NOFOLLOW` に
    /// よってシンボリックリンクとして解決されない実体を「ディレクトリでは
    /// ない」とみなし `ENOTDIR` を返す。これは Darwin だけでなく Linux でも
    /// 同様（実測確認済み: `build_site_rejects_output_path_escaping_via_symlinked_parent`・
    /// `read_verified_source_rejects_symlinked_intermediate_directory` の
    /// 回帰テストが GitHub ホステッド `ubuntu-latest`〈PR #899 CI 実行結果〉・
    /// Apple Silicon 実機の両方で `ENOTDIR` を観測しており、`ELOOP` 一貫という
    /// 当初の想定〈Linux は `open(2)` の `O_NOFOLLOW` 単体の記述どおり `ELOOP`
    /// を返すという誤った類推〉は `O_DIRECTORY` 併用時には成立しない）。
    /// Linux・Darwin ともに `ENOTDIR` の数値は `20` で一致するため単一定数で
    /// 扱う。よって全 unix プラットフォームで `ENOTDIR` も symlink 拒否として
    /// 扱う（実際に「ディレクトリを期待した位置に通常ファイルがある」設定
    /// ミスも同じ errno になり得るが、いずれの場合も fail-closed に拒否する
    /// 点は変わらないため安全側の近似として許容する）。
    const ENOTDIR_AS_SYMLINK_REJECTION: i32 = 20;

    /// `error` が「`O_NOFOLLOW` によるシンボリックリンク拒否」であるかを
    /// 判定する。呼び出し元（`build.rs` の `classify_*_io_error`・
    /// `probe_non_symlink_entry` 呼び出し側）がこれを見て `UnsafeSource`/
    /// `UnsafeOutputPath` へ詰め替える。
    pub fn is_symlink_rejection(error: &io::Error) -> bool {
        let raw = error.raw_os_error();
        raw == Some(ELOOP) || raw == Some(ENOTDIR_AS_SYMLINK_REJECTION)
    }

    #[cfg(target_os = "linux")]
    mod flags {
        use std::os::raw::c_int;
        pub const O_RDONLY: c_int = 0o0;
        pub const O_WRONLY: c_int = 0o1;
        pub const O_CREAT: c_int = 0o100;
        pub const O_EXCL: c_int = 0o200;
        pub const O_DIRECTORY: c_int = 0o200_000;
        pub const O_NOFOLLOW: c_int = 0o400_000;
        pub const O_CLOEXEC: c_int = 0o2_000_000;
        pub const O_NONBLOCK: c_int = 0o4_000;
    }

    #[cfg(target_os = "macos")]
    mod flags {
        use std::os::raw::c_int;
        pub const O_RDONLY: c_int = 0x0000;
        pub const O_WRONLY: c_int = 0x0001;
        pub const O_CREAT: c_int = 0x0200;
        pub const O_EXCL: c_int = 0x0800;
        pub const O_NOFOLLOW: c_int = 0x0100;
        pub const O_DIRECTORY: c_int = 0x10_0000;
        pub const O_CLOEXEC: c_int = 0x100_0000;
        pub const O_NONBLOCK: c_int = 0x0004;
    }

    use flags::*;

    /// コンポーネント（1 セグメント分の `OsStr`）を NUL 終端 C 文字列へ変換
    /// する。内部 NUL を含む異常な入力は `CString::new` が失敗するため
    /// fail-closed に拒否する。
    fn component_cstring(component: &OsStr) -> io::Result<CString> {
        CString::new(component.as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "path component contains an interior NUL byte",
            )
        })
    }

    /// `relative` を `Normal` コンポーネントのみからなる非空列として検証する。
    /// `ParentDir`（`..`）・`RootDir`（絶対パス）・`Prefix`（Windows ドライブ）・
    /// `CurDir`（`.`）はすべて拒否する。呼び出し元（`nav::parse_nav` の
    /// `validate_source_shape`／`validate_page_path`）側で `..`・絶対パスは既に
    /// 拒否済みだが、本ウォーク自体を自己完結した安全条件として多層防御で
    /// 再検査する。
    fn normalize_components(relative: &Path) -> io::Result<Vec<&OsStr>> {
        let mut out = Vec::new();
        for component in relative.components() {
            match component {
                Component::Normal(name) => out.push(name),
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "path contains a non-normal component (`..`, an absolute root, or `.`)",
                    ));
                }
            }
        }
        if out.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "relative path has no components",
            ));
        }
        Ok(out)
    }

    /// `openat` 1 回分の薄いラッパー。返る fd は呼び出し元が直ちに
    /// `File::from_raw_fd` で RAII 管理へ委ねる（fd リーク防止）。`mode` は
    /// 常に `c_uint`（= 昇格後の `int` 幅）で可変長引数として渡す（`openat`
    /// 宣言のドキュメント参照: 固定引数として渡すと Apple AArch64 ABI 上で
    /// 破壊される）。
    fn openat_raw(
        dirfd: c_int,
        name: &OsStr,
        extra_flags: c_int,
        mode: std::os::raw::c_uint,
    ) -> io::Result<File> {
        let cname = component_cstring(name)?;
        // SAFETY: `dirfd` は呼び出し元が所有する有効なディレクトリ fd、
        // `cname` は NUL 終端済みの単一パスコンポーネント（`/` を含まない）。
        // `mode` は C の可変長引数既定昇格と同じ `c_uint`（int 幅）で渡す。
        // FFI 境界（`.claude/rules/coding-rust.md` unsafe 方針）。返る fd の
        // 所有権は本関数が引き継ぎ、直後に `File::from_raw_fd` で RAII へ渡す。
        let fd = unsafe { openat(dirfd, cname.as_ptr(), extra_flags | O_CLOEXEC, mode) };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            // SAFETY: 直前の `openat` が返した、他に所有者のいない新規 fd。
            Ok(unsafe { File::from_raw_fd(fd) })
        }
    }

    /// `dir` 直下の 1 コンポーネントをディレクトリとして `O_DIRECTORY |
    /// O_NOFOLLOW` で開く（symlink なら `ELOOP`、非ディレクトリなら
    /// `ENOTDIR` でカーネルが拒否する）。
    fn open_subdir(dir: &File, name: &OsStr) -> io::Result<File> {
        openat_raw(dir.as_raw_fd(), name, O_DIRECTORY | O_NOFOLLOW, 0)
    }

    /// `dir` 直下の 1 コンポーネントを `O_RDONLY | O_NOFOLLOW` で開く
    /// （symlink なら `ELOOP` で拒否）。
    fn open_leaf_readonly(dir: &File, name: &OsStr) -> io::Result<File> {
        openat_raw(dir.as_raw_fd(), name, O_RDONLY | O_NOFOLLOW, 0)
    }

    /// `dir` 直下に `name` というサブディレクトリを作る（既存なら成功扱い）。
    /// symlink への差し替えは呼び出し元が続けて行う `open_subdir` の
    /// `O_NOFOLLOW` オープンで検出する（ここでは作成のみ担う）。
    fn mkdirat_if_missing(dir: &File, name: &OsStr, mode: ModeT) -> io::Result<()> {
        let cname = component_cstring(name)?;
        // SAFETY: `dir` は呼び出し元が所有する有効なディレクトリ fd。
        let result = unsafe { mkdirat(dir.as_raw_fd(), cname.as_ptr(), mode) };
        if result == 0 {
            Ok(())
        } else {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::AlreadyExists {
                Ok(())
            } else {
                Err(error)
            }
        }
    }

    /// `base` を起点に、`relative` の各コンポーネントを 1 つずつ既存の
    /// ディレクトリとして `O_NOFOLLOW` で開き進める（作成はしない）。
    /// `page.source` の親ディレクトリ列（`nav::validate_sources` が実在済みで
    /// あることを事前検証している）を辿る用途。
    pub fn open_dir_beneath(base: &File, relative: &Path) -> io::Result<File> {
        let components = normalize_components(relative)?;
        let mut current = base.try_clone()?;
        for name in components {
            current = open_subdir(&current, name)?;
        }
        Ok(current)
    }

    /// `base` を起点に、`relative` の各コンポーネントを 1 つずつ「なければ
    /// 作成 → `O_NOFOLLOW` で開く」の順で辿り、最終ディレクトリの fd を返す。
    /// 出力先（`out` 配下）のディレクトリ列を、シンボリックリンク差し替えを
    /// 検出しながら作成する用途。
    pub fn open_or_create_dir_chain(base: &File, relative: &Path) -> io::Result<File> {
        let components = normalize_components(relative)?;
        let mut current = base.try_clone()?;
        for name in components {
            mkdirat_if_missing(&current, name, 0o755)?;
            current = open_subdir(&current, name)?;
        }
        Ok(current)
    }

    /// `dir` 直下の `name` を読み取り専用で開く（symlink なら `ELOOP`）。
    /// `page.source` の最終コンポーネント（ファイル本体）の読み込み用。
    pub fn open_file_beneath(dir: &File, name: &OsStr) -> io::Result<File> {
        open_leaf_readonly(dir, name)
    }

    /// `dir` 直下に `name` を新規作成する（`O_CREAT | O_EXCL | O_NOFOLLOW`）。
    /// 既存エントリ（通常ファイル・symlink 問わず）があれば `EEXIST` で失敗する
    /// （一時ファイル名の衝突検出・symlink 追従防止の両方を兼ねる）。
    pub fn create_new_file_beneath(dir: &File, name: &OsStr, mode: u32) -> io::Result<File> {
        openat_raw(
            dir.as_raw_fd(),
            name,
            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW,
            mode,
        )
    }

    /// `dir` 直下に既存の `name` があるかを `O_NOFOLLOW` で確認する。
    /// - `Ok(true)`: 既存の非 symlink エントリ（通常ファイル等）
    /// - `Ok(false)`: 存在しない
    /// - `Err` かつ `kind() == FilesystemLoop`: 既存の symlink（呼び出し元が
    ///   `UnsafeOutputPath` へ詰め替える）
    ///
    /// **`O_NONBLOCK` を併用する**（`open_leaf_readonly` を使い回さない
    /// 理由）: `out` 配下は再ビルドで使い回されるディレクトリのため、
    /// 既存生成物の名前を先客の FIFO（named pipe）に差し替えられている
    /// 可能性を排除できない。`O_NONBLOCK` なしの `open(O_RDONLY)` は FIFO に
    /// 対して書き込み側が現れるまで無期限にブロックしうる（POSIX の規定
    /// 動作）ため、存在確認だけのこの用途では `O_NONBLOCK` を付けて即時に
    /// 復帰させる（Cursor Bugbot 指摘・PR #899）。読み取り専用オープンな
    /// ので通常ファイル・ディレクトリへの挙動には影響しない。
    pub fn probe_non_symlink_entry(dir: &File, name: &OsStr) -> io::Result<bool> {
        match openat_raw(dir.as_raw_fd(), name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK, 0) {
            Ok(_file) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// `dir` 直下の `name` をベストエフォートで削除する（一時ファイルの
    /// 書き込み・rename 失敗時の後始末専用。失敗しても呼び出し元へは伝播しない
    /// 前提で使う）。
    pub fn remove_beneath(dir: &File, name: &OsStr) -> io::Result<()> {
        let cname = component_cstring(name)?;
        // SAFETY: `dir` は呼び出し元が所有する有効なディレクトリ fd。
        let result = unsafe { unlinkat(dir.as_raw_fd(), cname.as_ptr(), 0) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// 同一の検証済み `dir`（dirfd）内で `old_name` → `new_name` へ不可分に
    /// rename する。親ディレクトリを名前で再解決しないため、検証後に親が
    /// 差し替えられていても影響を受けない（`fs::rename` はパス文字列で親を
    /// 毎回再解決するため対策になっていなかった。レビュー追加ラウンド指摘）。
    pub fn rename_beneath(dir: &File, old_name: &OsStr, new_name: &OsStr) -> io::Result<()> {
        let old_c = component_cstring(old_name)?;
        let new_c = component_cstring(new_name)?;
        let dirfd = dir.as_raw_fd();
        // SAFETY: `dir` は検証済みディレクトリの fd。old/new とも `/` を
        // 含まない単一コンポーネント名の NUL 終端 C 文字列。
        let result = unsafe { renameat(dirfd, old_c.as_ptr(), dirfd, new_c.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

/// unix 以外（Windows 等）は `openat` 相当の fd 相対 API がなく、
/// [`fd_walk`]（unix 版）による TOCTOU 対策を実施できない。レビュー指摘の
/// とおり検証不能なプラットフォームでは許可せず fail-closed に拒否する
/// （`docs-site` は CI・開発者ローカル実行とも Linux/macOS が前提。
/// `.claude/rules/ci.md` 実機依存節）。
#[cfg(not(unix))]
mod fd_walk {
    use std::ffi::OsStr;
    use std::fs::File;
    use std::io;
    use std::path::Path;

    fn unsupported() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "fd-relative symlink-safe filesystem access is not implemented on this platform",
        )
    }

    pub fn open_dir_beneath(_base: &File, _relative: &Path) -> io::Result<File> {
        Err(unsupported())
    }
    pub fn open_or_create_dir_chain(_base: &File, _relative: &Path) -> io::Result<File> {
        Err(unsupported())
    }
    pub fn open_file_beneath(_dir: &File, _name: &OsStr) -> io::Result<File> {
        Err(unsupported())
    }
    pub fn create_new_file_beneath(_dir: &File, _name: &OsStr, _mode: u32) -> io::Result<File> {
        Err(unsupported())
    }
    pub fn probe_non_symlink_entry(_dir: &File, _name: &OsStr) -> io::Result<bool> {
        Err(unsupported())
    }
    pub fn rename_beneath(_dir: &File, _old_name: &OsStr, _new_name: &OsStr) -> io::Result<()> {
        Err(unsupported())
    }
    pub fn remove_beneath(_dir: &File, _name: &OsStr) -> io::Result<()> {
        Err(unsupported())
    }
    pub fn is_symlink_rejection(_error: &io::Error) -> bool {
        false
    }
}

/// [`read_verified_source`] の I/O エラーを、fd ウォークが検出したシンボリック
/// リンク差し替え（[`fd_walk::is_symlink_rejection`]）とそれ以外に振り分ける。
fn classify_source_io_error(source: &str, error: std::io::Error) -> BuildError {
    if fd_walk::is_symlink_rejection(&error) {
        BuildError::Nav(NavError::UnsafeSource(source.to_string()))
    } else {
        BuildError::ReadSource {
            source: source.to_string(),
            error,
        }
    }
}

/// `page.source` を `repo_root_dir`（`repo_root` を検証済みで開いた fd）配下
/// から fd 相対（[`fd_walk`]）で辿って読み込む。単一の `File` ハンドルから
/// サイズ確認・内容読み込みの両方を行う。
///
/// [`nav::validate_sources`] は全ページの一括事前検証（早期にわかりやすい
/// エラーを返すため）だが、実際に読み込む直前にも [`fd_walk::open_dir_beneath`]
/// / [`fd_walk::open_file_beneath`] で `repo_root_dir` 起点の fd 相対アクセスへ
/// 収斂させる。`canonicalize` によるパス文字列ベースの再検証（旧実装）は
/// `File::open` がパスを再解決するため、検証と実際のアクセスの間で中間
/// ディレクトリ（`source` 中の途中のディレクトリコンポーネント）がシンボリック
/// リンクへ差し替えられる race を検知できなかった（codex-review 追加ラウンド
/// 指摘・PR #899）。fd ウォークは検証済み dirfd からしか名前解決しないため、
/// この race が構造的に発生しない。
///
/// 残る対策（[`fd_walk`] のドキュメント参照）:
/// - 各コンポーネントの `openat(..., O_NOFOLLOW)` がシンボリックリンクを
///   `ELOOP`（`ErrorKind::FilesystemLoop`）で拒否する
/// - `metadata().len()` でのサイズ確認後、同じハンドルから `Read::take` で
///   `MAX_SOURCE_BYTES + 1` バイトに上限した bounded read を行い、確認後の
///   追記による上限迂回を防ぐ
fn read_verified_source(repo_root_dir: &fs::File, source: &str) -> Result<String, BuildError> {
    let relative = Path::new(source);
    let (parent, file_name) = match (relative.parent(), relative.file_name()) {
        (Some(parent), Some(file_name)) => (parent, file_name),
        _ => return Err(BuildError::Nav(NavError::UnsafeSource(source.to_string()))),
    };

    let parent_dir = if parent.as_os_str().is_empty() {
        repo_root_dir
            .try_clone()
            .map_err(|error| classify_source_io_error(source, error))?
    } else {
        fd_walk::open_dir_beneath(repo_root_dir, parent)
            .map_err(|error| classify_source_io_error(source, error))?
    };

    let file = fd_walk::open_file_beneath(&parent_dir, file_name)
        .map_err(|error| classify_source_io_error(source, error))?;

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

/// [`fd_walk::open_or_create_dir_chain`] / [`fd_walk::open_dir_beneath`] の
/// I/O エラーを、fd ウォークが検出したシンボリックリンク差し替え
/// （[`fd_walk::is_symlink_rejection`]）とそれ以外に振り分ける。
fn classify_output_dir_io_error(path_display: &str, error: std::io::Error) -> BuildError {
    if fd_walk::is_symlink_rejection(&error) {
        BuildError::UnsafeOutputPath(path_display.to_string())
    } else {
        BuildError::WritePage {
            path: path_display.to_string(),
            error,
        }
    }
}

/// `out_root_dir`（`out` を検証済みで開いた fd）配下へ、`relative_path`
/// （`out` からの相対パス。末尾セグメントがファイル名）の親ディレクトリを
/// [`fd_walk::open_or_create_dir_chain`] で作成しつつ辿り、書き込み先の
/// ディレクトリ fd を得たうえで一時ファイル＋`renameat` の atomic replace で
/// 書き出す。
///
/// **`fs::create_dir_all` を一括で呼んでから事後に `canonicalize` するのでは
/// 不十分**（レビュー指摘・回帰テストで確認済み）: 中間コンポーネントが `out`
/// 外を指すシンボリックリンクの場合、一括 `create_dir_all` は検証前にそのリンク
/// 先へ実際にディレクトリを作成してしまう（副作用が検査より先に発生する）。
/// さらに `canonicalize` によるパス文字列ベースの事後検証（旧実装）は、検証と
/// 実際の `open`/`rename` の間に中間ディレクトリを symlink へ差し替えられる
/// race を検知できなかった（codex-review 追加ラウンド指摘・PR #899）。
/// [`fd_walk`] は検証済み dirfd からしか名前解決しないため、この race が
/// 構造的に発生しない。
///
/// 書き込み本体（`codex-review` 指摘・PR #899・CI ブロック要因の P0 だった
/// `remove_file` → `create_new` の危険な順序の修正）:
/// - 一時ファイル名は `.{ファイル名}.tmp-{pid}-{カウンタ}-{nanos}` とし、
///   同一プロセス内の並行呼び出し・過去の残骸との衝突を避ける
///   （`AlreadyExists` は限られた回数だけ次の候補へ retry し、それでも衝突
///   する場合は fail-closed にエラーを返す）
/// - 一時ファイルは検証済みディレクトリ fd に対する
///   [`fd_walk::create_new_file_beneath`]（`O_CREAT | O_EXCL | O_NOFOLLOW`）で
///   作成するため、事前に何か（symlink 含む）がそのパスを指していれば
///   `EEXIST` で失敗し、その追従先へは書き込まない
/// - 書き込み・`sync_all` 完了後に [`fd_walk::rename_beneath`]（同一 dirfd 内の
///   `renameat`）で本来の名前へ不可分に置換する。**親ディレクトリを名前で
///   再解決しない**ため、検証後に親が symlink へ差し替えられていても影響を
///   受けない（`fs::rename` はパス文字列で親を毎回再解決するため対策に
///   なっていなかった。レビュー追加ラウンド指摘）
/// - 置換先の既存エントリが symlink かどうかは
///   [`fd_walk::probe_non_symlink_entry`] で事前確認し、symlink なら
///   `UnsafeOutputPath` で拒否する（既存シンボリックリンク経由の意図しない
///   上書き対策）。既存の通常ファイルはそのまま `rename` で置換してよい
///   （リビルド時の正常な更新経路）
/// - 書き込み・`rename` のいずれかが失敗した場合は一時ファイルをベストエフォート
///   で削除し、元の生成物・エラー内容はそのまま呼び出し元へ返す
fn write_file_creating_parent(
    out: &Path,
    out_root_dir: &fs::File,
    relative_path: &Path,
    contents: &str,
) -> Result<(), BuildError> {
    let path_display = out.join(relative_path).display().to_string();

    let parent_relative = relative_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let dir = match parent_relative {
        Some(parent) => fd_walk::open_or_create_dir_chain(out_root_dir, parent)
            .map_err(|error| classify_output_dir_io_error(&path_display, error))?,
        None => out_root_dir
            .try_clone()
            .map_err(|error| classify_output_dir_io_error(&path_display, error))?,
    };

    let file_name = relative_path
        .file_name()
        .ok_or_else(|| BuildError::UnsafeOutputPath(path_display.clone()))?;

    match fd_walk::probe_non_symlink_entry(&dir, file_name) {
        Ok(_existing_or_missing) => {}
        Err(error) if fd_walk::is_symlink_rejection(&error) => {
            return Err(BuildError::UnsafeOutputPath(path_display));
        }
        Err(error) => {
            return Err(BuildError::WritePage {
                path: path_display,
                error,
            });
        }
    }

    let file_name_lossy = file_name.to_string_lossy().into_owned();
    static TMP_NAME_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    const MAX_TMP_NAME_ATTEMPTS: u32 = 8;

    let mut attempt = 0u32;
    let (tmp_name, mut tmp_file) = loop {
        let counter = TMP_NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let tmp_name = format!(
            ".{file_name_lossy}.tmp-{}-{counter}-{nanos}",
            std::process::id()
        );
        match fd_walk::create_new_file_beneath(&dir, std::ffi::OsStr::new(&tmp_name), 0o644) {
            Ok(file) => break (tmp_name, file),
            Err(error)
                if error.kind() == std::io::ErrorKind::AlreadyExists
                    && attempt < MAX_TMP_NAME_ATTEMPTS =>
            {
                attempt += 1;
            }
            Err(error) => {
                return Err(BuildError::WritePage {
                    path: path_display,
                    error,
                });
            }
        }
    };

    let write_result = std::io::Write::write_all(&mut tmp_file, contents.as_bytes())
        .and_then(|()| tmp_file.sync_all());
    drop(tmp_file);
    if let Err(error) = write_result {
        let _ = fd_walk::remove_beneath(&dir, std::ffi::OsStr::new(&tmp_name));
        return Err(BuildError::WritePage {
            path: path_display,
            error,
        });
    }

    if let Err(error) = fd_walk::rename_beneath(&dir, std::ffi::OsStr::new(&tmp_name), file_name) {
        let _ = fd_walk::remove_beneath(&dir, std::ffi::OsStr::new(&tmp_name));
        return Err(BuildError::WritePage {
            path: path_display,
            error,
        });
    }

    Ok(())
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
///    `markdown::markdown_to_nodes` → `layout::docs_page` → `html::render`
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

    // 手順 3 の読み込み直前の再検証（TOCTOU 対策。[`fd_walk`]・
    // [`read_verified_source`] のモジュールコメント参照）に使う `root` の
    // dirfd。以降の全ページ読み込みをこの単一の検証済み fd から fd 相対で
    // 辿るため、`validate_sources` 通過後に `root` 自体が別の実体へ差し替え
    // られない限り（`root` は呼び出し元が制御する起点であり、その先の途中
    // ディレクトリの差し替えのみを本ウォークで防ぐ）安全である。
    let root_dir = fs::File::open(root)
        .map_err(|_| BuildError::Nav(NavError::MissingSource(root.display().to_string())))?;

    // 手順 3: 全ページを先にメモリ上でレンダリングし切ってから書き出す。
    // `relative_out_path` は `out` からの相対パス（`page_output_path` に空の
    // ベースを渡して算出する）。書き出しは fd 相対（[`write_file_creating_parent`]）
    // で行うため絶対パスは不要で、`out` は表示用にのみ渡す。
    let mut rendered_pages: Vec<(PathBuf, String)> = Vec::new();
    for section in &parsed.sections {
        for page in &section.pages {
            let markdown_input = read_verified_source(&root_dir, &page.source)?;

            let body = markdown::markdown_to_nodes(&markdown_input);
            let page_node = layout::docs_page(&parsed, &page.title, &page.path, body);
            let html_out = format!("{DOCTYPE}{}", html::render(&page_node));

            let relative_out_path = page_output_path(Path::new(""), &page.path);
            rendered_pages.push((relative_out_path, html_out));
        }
    }

    // 手順 4: 出力ディレクトリ作成 → ページ書き出し → テーマ CSS 書き出し。
    fs::create_dir_all(out).map_err(BuildError::CreateOutDir)?;
    let out_root_dir = fs::File::open(out).map_err(BuildError::CreateOutDir)?;

    let mut written = Vec::with_capacity(rendered_pages.len() + 1);
    for (relative_out_path, html_out) in &rendered_pages {
        write_file_creating_parent(out, &out_root_dir, relative_out_path, html_out)?;
        written.push(relative_out_path.clone());
    }

    let css_path = PathBuf::from(SITE_CSS_RELATIVE_PATH);
    write_file_creating_parent(out, &out_root_dir, &css_path, crate::theme::SITE_CSS).map_err(
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

        let root_dir = fs::File::open(&root.0).expect("open repo_root as dirfd");
        match read_verified_source(&root_dir, "linked.md") {
            Err(BuildError::Nav(NavError::UnsafeSource(source))) => {
                assert_eq!(source, "linked.md");
            }
            other => panic!("expected UnsafeSource for symlink escape, got {other:?}"),
        }
    }

    /// [`fd_walk`] 追加ラウンドの回帰テスト（PR #899 追加レビュー指摘）:
    /// `page.source` の**中間ディレクトリ**（`repo_root` 直下ではなく、その
    /// 内側のディレクトリ）がシンボリックリンクで `repo_root` の外を指す場合に
    /// `read_verified_source` が拒否することを確認する。旧実装
    /// （`canonicalize` + `starts_with` によるパス文字列ベースの検証）は
    /// `canonicalize` がシンボリックリンクを解決してしまうため、この中間
    /// ディレクトリ差し替え自体は素通りしていた（`canonicalize` 結果が
    /// たまたま `repo_root` 外を指せば旧実装でも拒否できたが、本テストは
    /// 「`repo_root` 内側の別ディレクトリへの中間シンボリックリンク」という、
    /// 旧実装が拒否できなかった具体的な回帰ケースを再現する）。
    #[cfg(unix)]
    #[test]
    fn read_verified_source_rejects_symlinked_intermediate_directory() {
        let root = TempDir::new("read-verified-intermediate-root");
        let real_dir = root.0.join("real");
        fs::create_dir_all(&real_dir).expect("create real intermediate directory");
        fs::write(real_dir.join("doc.md"), b"# Real").expect("write fixture under real dir");
        // `linked` は `repo_root` 内側の別ディレクトリ（`real`）を指す中間
        // シンボリックリンク。`linked/doc.md` という `page.source` はこの
        // シンボリックリンクを経由する。
        std::os::unix::fs::symlink(&real_dir, root.0.join("linked"))
            .expect("create symlinked intermediate directory");

        let root_dir = fs::File::open(&root.0).expect("open repo_root as dirfd");
        match read_verified_source(&root_dir, "linked/doc.md") {
            Err(BuildError::Nav(NavError::UnsafeSource(source))) => {
                assert_eq!(source, "linked/doc.md");
            }
            other => {
                panic!("expected UnsafeSource for symlinked intermediate directory, got {other:?}")
            }
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

    /// レビュー指摘の回帰テスト（PR #899・codex-review P0・CI ブロック要因）:
    /// `write_file_creating_parent` は既存生成物を `remove_file` で消してから
    /// 書き込む代わりに、一時ファイル＋`rename` の atomic replace で置換する。
    /// 同一 `out` への再ビルドが（1）内容を正しく更新し、（2）一時ファイルの
    /// 残骸（`.{name}.tmp-*`）を残さないことを確認する。
    #[test]
    fn build_site_rebuild_replaces_existing_output_without_tmp_residue() {
        let root = TempDir::new("build-rebuild-root");
        fs::create_dir_all(root.0.join("site")).unwrap();
        let nav_toml = r#"
[site]
title = "Docs"
base_path = ""

[[section]]
title = "Guide"

[[section.page]]
title = "Intro"
source = "site/intro.md"
path = "/intro/"
"#;
        fs::write(root.0.join("site/nav.toml"), nav_toml).unwrap();
        fs::write(root.0.join("site/intro.md"), b"# First").unwrap();

        let out = TempDir::new("build-rebuild-out");
        build_site(&root.0, &out.0).expect("first build should succeed");
        let first_html = fs::read_to_string(out.0.join("intro/index.html")).unwrap();
        assert!(first_html.contains("<h1>First</h1>"));

        // 同じ `out` へ内容を変えて再ビルドする。
        fs::write(root.0.join("site/intro.md"), b"# Second").unwrap();
        build_site(&root.0, &out.0).expect("second build should succeed");
        let second_html = fs::read_to_string(out.0.join("intro/index.html")).unwrap();
        assert!(
            second_html.contains("<h1>Second</h1>"),
            "rebuild must replace the previous generated content"
        );

        // `out` 配下（`assets/` 含む）のどこにも一時ファイルの残骸が残っていない
        // ことを確認する。
        fn assert_no_tmp_residue(dir: &Path) {
            for entry in fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                assert!(
                    !name.contains(".tmp-"),
                    "leftover temp file found: {}",
                    path.display()
                );
                if entry.file_type().unwrap().is_dir() {
                    assert_no_tmp_residue(&path);
                }
            }
        }
        assert_no_tmp_residue(&out.0);
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
