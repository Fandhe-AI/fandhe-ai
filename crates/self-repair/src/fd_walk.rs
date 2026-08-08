//! パスコンポーネントを `openat`（`O_NOFOLLOW`）で 1 段ずつ辿り、検証済み
//! fd に対して直接読み書きするユーティリティ（PR #361 codex-review 第 4 波
//! P0 指摘: [`crate::candidate`] の途中ディレクトリ symlink 差し替え TOCTOU
//! の構造的排除）。
//!
//! # 旧実装（`reject_symlink_escape` + `fs::write`/`fs::read_to_string`）の限界
//! 旧実装は「検査」（`fs::symlink_metadata` によるコンポーネント逐次検査）と
//! 「利用」（パス文字列を再解決する `fs::write`/`fs::OpenOptions::open`）が
//! 別々の syscall 列に分かれていた。両者の間（TOCTOU window）に途中
//! ディレクトリが symlink へ差し替えられると、「利用」側がその symlink を
//! 追跡してしまう余地が残った（末尾コンポーネントは `O_NOFOLLOW` 付き open で
//! 塞いでいたが、途中ディレクトリは字句検査後の `stat` に依存していた）。
//!
//! # 本モジュールの設計
//! `workspace` を dir-fd として開き、そこを起点に相対パスの各コンポーネントを
//! `openat(dirfd, name, O_NOFOLLOW | O_DIRECTORY, ...)` で 1 段ずつ辿る。
//! 各コンポーネントの「symlink でないことの検証」と「そのディレクトリを開いて
//! 次段の起点にすること」が単一の syscall に統合されるため、検査後に別の
//! syscall で再解決する隙（TOCTOU window）が構造的に存在しない。パス文字列
//! ベースの API（`fs::write`・`fs::read_to_string`・`Path::join` を経由した
//! 全体パス解決）は一切使わない。
//!
//! # プラットフォーム
//! Linux・macOS のみ対応する。`openat`/`O_NOFOLLOW`/`O_DIRECTORY` 等のフラグ
//! 値は各 OS の `fcntl.h` を出典として本ファイル内にローカル複製する
//! （`libc` 依存は禁止。deps-policy.md 許容依存 8 区分外）。self-repair CLI の
//! 実行環境は self-hosted の Linux/macOS runner のみであり、それ以外の
//! ターゲットでは値の出典を保証できないため、限定的なフォールバック
//! （旧: 一時ファイル + rename 方式）は用意せず `io::Error` で fail-closed に
//! 候補適用そのものを拒否する（ユーザー指示。安全側に倒す。下記
//! `unsupported_platform_error` 参照）。
//!
//! # 呼び出し元
//! [`crate::candidate::apply_candidate`]（baseline 復元・候補適用の書き込み
//! 経路）・[`crate::candidate::CandidateFixGenerator::new`]・
//! [`crate::bug_fix::BugFixFixGenerator::new`]・
//! [`crate::feature_addition::FeatureAdditionFixGenerator::new`]（baseline
//! スナップショット読み込み経路）から利用する。呼び出し元は本モジュールの
//! 前に必ず [`crate::candidate::validate_relative_path`]（絶対パス・`..`
//! 成分の字句拒否）を経由させる（本モジュールは字句検査を行わず、`..` 成分を
//! 渡されると `openat` がそのまま親ディレクトリへ遡ってしまう）。

// `io`・`Path` は fail-closed フォールバック（下記
// `cfg(not(any(target_os = "linux", target_os = "macos")))` 側）のシグネチャ
// にも必要なため無条件 import とする。それ以外（`fs`・`CString`・fd 型・
// C 型・Unix 拡張 trait）は Linux/macOS 専用実装でのみ使うため cfg で
// 絞り込む。無条件 import すると Windows では `std::os::fd`/`std::os::unix`
// 自体が存在せずビルド不能になり、他の Unix 系（FreeBSD 等）では未使用
// import が `-D warnings` で拒否される（PR #361 codex-review 指摘 1 と同型の
// 不具合を fd_walk 側で再発させないための cfg 分離）。
use std::io;
use std::path::Path;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::ffi::CString;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::fs;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::io::{Read as _, Write as _};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd, RawFd};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::raw::{c_char, c_int, c_uint};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::ffi::OsStrExt as _;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::path::Component;

/// 各 OS の `fcntl.h` 定義値のローカル複製（`libc` 依存禁止のため）。
/// 出典: Linux `include/uapi/asm-generic/fcntl.h`（x86_64/aarch64 共通の
/// generic 値。mips/parisc/sparc/alpha 等の非 asm-generic アーキテクチャでは
/// 値が異なるが、self-repair CLI の実行対象アーキテクチャ外のため未対応
/// ＝ 下記 `cfg(not(...))` フォールバック側で fail-closed に拒否する）。
#[cfg(target_os = "linux")]
mod raw {
    pub const O_RDONLY: i32 = 0o0;
    pub const O_WRONLY: i32 = 0o1;
    pub const O_CREAT: i32 = 0o100;
    pub const O_TRUNC: i32 = 0o1000;
    pub const O_DIRECTORY: i32 = 0o200000;
    pub const O_NOFOLLOW: i32 = 0o400000;
}

/// 出典: macOS `sys/fcntl.h`。
#[cfg(target_os = "macos")]
mod raw {
    pub const O_RDONLY: i32 = 0x0000;
    pub const O_WRONLY: i32 = 0x0001;
    pub const O_CREAT: i32 = 0x0200;
    pub const O_TRUNC: i32 = 0x0400;
    pub const O_DIRECTORY: i32 = 0x100000;
    pub const O_NOFOLLOW: i32 = 0x0100;
}

// `openat(2)` の FFI 宣言。C 側のシグネチャは
// `int openat(int dirfd, const char *pathname, int flags, ...)` で
// `mode_t` 引数（`O_CREAT` 使用時のみ意味を持つ）は可変長引数側にある。
// 固定 4 引数として宣言すると呼び出し規約が可変長引数と異なる ABI
// （例: `aarch64-apple-darwin` は可変長引数をスタック経由で渡す）で
// `mode` が読めない不整合を起こすため、Rust 側も `...` 可変長引数として
// 宣言し、`O_CREAT` を渡す呼び出しでのみ `mode` 実引数を追加する
// （下記 openat_dir/openat_final 参照）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
unsafe extern "C" {
    fn openat(dirfd: c_int, pathname: *const c_char, flags: c_int, ...) -> c_int;
}

/// 相対パスをコンポーネント単位の NUL 終端文字列列へ分解する。
///
/// 呼び出し元は事前に [`crate::candidate::validate_relative_path`] を経由
/// させる契約のため、ここに到達するのは通常 `Normal`（1 コンポーネント
/// あたり 1 段のディレクトリ／ファイル名）のみである。それ以外
/// （`RootDir`・`Prefix`・`ParentDir`）が渡された場合は契約違反として
/// fail-closed に拒否する（`openat` へ `..` をそのまま渡すと親ディレクトリへ
/// 遡ってしまうため、ここでの防御は多重化の意味を持つ）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn split_components(relative_path: &Path) -> io::Result<Vec<CString>> {
    let mut parts = Vec::new();
    for component in relative_path.components() {
        match component {
            Component::Normal(part) => {
                let cstring = CString::new(part.as_bytes()).map_err(|_| {
                    io::Error::other(format!(
                        "候補修正のパスに NUL バイトを含むコンポーネントがあります: {}",
                        relative_path.display()
                    ))
                })?;
                parts.push(cstring);
            }
            Component::CurDir => continue,
            other => {
                return Err(io::Error::other(format!(
                    "候補修正のパスに許可されないコンポーネント（{other:?}）が含まれます: {}",
                    relative_path.display()
                )));
            }
        }
    }
    if parts.is_empty() {
        return Err(io::Error::other(format!(
            "候補修正のパスが空です: {}",
            relative_path.display()
        )));
    }
    Ok(parts)
}

/// `workspace` を dir-fd として開く（fd 走査チェーンの起点）。
///
/// `workspace` 自体は sandbox.rs が管理する信頼済みルート（self-repair が
/// 自ら作成した一時ディレクトリ）であり、途中ディレクトリの symlink
/// 差し替え防御の対象は `workspace` からの相対パス側（[`walk_to_final`]）
/// である。ルート自体は std の安全な API（`fs::OpenOptions`）で開けるため
/// `unsafe` を使わない（`O_DIRECTORY` を `custom_flags` で付与し、
/// ディレクトリでなければ open 自体を失敗させる）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_root_dir(workspace: &Path) -> io::Result<OwnedFd> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(raw::O_DIRECTORY)
        .open(workspace)?;
    Ok(OwnedFd::from(file))
}

/// 途中ディレクトリを 1 段開く（`O_NOFOLLOW | O_DIRECTORY`）。
///
/// `name` が symlink または非ディレクトリであれば `openat` 自体が
/// `ELOOP`/`ENOTDIR` で失敗する。「symlink でないことの検証」と
/// 「次段の起点として開くこと」が単一の syscall に統合されるため、
/// モジュール冒頭 doc が説明する TOCTOU window がここには存在しない。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn openat_dir(parent: RawFd, name: &CString) -> io::Result<OwnedFd> {
    let flags = raw::O_DIRECTORY | raw::O_NOFOLLOW | raw::O_RDONLY;
    // SAFETY: FFI 境界（TOCTOU 排除のための openat 呼び出し。モジュール冒頭
    // doc 参照）。`parent` は呼び出し元（[`walk_to_final`]）が直前の
    // `open_root_dir`/`openat_dir` 呼び出しで取得した有効な dir-fd であり、
    // `name` は [`split_components`] が生成した NUL 終端済み `CString` から
    // 得た有効なポインタである。`O_CREAT` を含まない固定 3 引数呼び出しの
    // ため可変長引数側の `mode` は渡さない（[`openat`] doc 参照）。
    let fd = unsafe { openat(parent, name.as_ptr(), flags) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` は直前の `openat(2)` 呼び出しが返した新規かつ有効な
    // 所有 fd（他に所有者はいない）。`OwnedFd` でラップし、以後の close(2)
    // を std の `Drop` 実装に委譲することで close 漏れ（fd リーク）を防ぐ。
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// 末尾コンポーネント（ファイル本体）を開く（`O_NOFOLLOW` 必須）。
///
/// `extra_flags` に `O_CREAT` を含む場合のみ `mode`（`0o600`）を可変長引数
/// として渡す（[`openat`] doc 参照。C 側で `mode_t` は `O_CREAT` 未指定時は
/// 未評価だが、可変長引数の有無自体が呼び出し規約に影響するため実引数の
/// 個数を一致させる）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn openat_final(parent: RawFd, name: &CString, extra_flags: i32) -> io::Result<OwnedFd> {
    let flags = extra_flags | raw::O_NOFOLLOW;
    let fd = if extra_flags & raw::O_CREAT != 0 {
        let mode: c_uint = 0o600;
        // SAFETY: openat_dir と同じ契約（parent は検証済み dir-fd・name は
        // NUL 終端済み）。O_CREAT を渡すため mode を可変長引数側に追加する。
        unsafe { openat(parent, name.as_ptr(), flags, mode) }
    } else {
        // SAFETY: 同上。O_CREAT を含まないため mode は渡さない。
        unsafe { openat(parent, name.as_ptr(), flags) }
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: openat_dir と同じ契約（新規かつ有効な所有 fd を OwnedFd で
    // ラップし close を Drop に委譲する）。
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// `workspace` から `relative_path` を fd 走査で辿り、末尾コンポーネントを
/// `final_flags`（`O_RDONLY` または `O_WRONLY | O_CREAT | O_TRUNC`）で開く。
///
/// [`read_via_fd_walk`]・[`write_via_fd_walk`]・[`probe`] の共通実体。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn walk_to_final(workspace: &Path, relative_path: &Path, final_flags: i32) -> io::Result<OwnedFd> {
    let parts = split_components(relative_path)?;
    // `split_last` は `(末尾要素, 残り全部)` を返す。ここでの末尾要素が
    // ファイル本体（`file_part`）、残りが先頭からの中間ディレクトリ列
    // （`dir_parts`）である。
    let (file_part, dir_parts) = parts
        .split_last()
        .expect("split_components は空 Vec を返さない（上の空チェック参照）");
    let mut current = open_root_dir(workspace)?;
    for part in dir_parts {
        current = openat_dir(current.as_raw_fd(), part)?;
    }
    openat_final(current.as_raw_fd(), file_part, final_flags)
}

/// baseline 復元・候補適用が使う書き込み経路。
///
/// `O_WRONLY | O_CREAT | O_TRUNC` で開く。生成 AI 候補は既存モジュール内の
/// 合成実装に限定され（`feature_addition.rs` の新規ファイル拒否）、
/// baseline はいずれのコンストラクタも「読み込みに成功した」パスからのみ
/// 構築される（＝親ディレクトリは常に実在する）ため、本番経路で `O_CREAT`
/// が実際に新規ファイルを作る場面はない。新規作成の動作自体は
/// [`crate::candidate`] レベルの `apply_candidate` を直接呼ぶ経路（`pub`
/// API）でも成立しうるため、汎用性のためのサポートとして残している
/// （`mkdirat` による中間ディレクトリの新規作成はスコープ外 — 現状の呼び出し
/// 元にその要求がない）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn write_via_fd_walk(
    workspace: &Path,
    relative_path: &Path,
    content: &str,
) -> io::Result<()> {
    let owned = walk_to_final(
        workspace,
        relative_path,
        raw::O_WRONLY | raw::O_CREAT | raw::O_TRUNC,
    )?;
    let mut file = fs::File::from(owned);
    file.write_all(content.as_bytes())
}

/// baseline スナップショット取得が使う読み込み経路。
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn read_via_fd_walk(workspace: &Path, relative_path: &Path) -> io::Result<String> {
    let owned = walk_to_final(workspace, relative_path, raw::O_RDONLY)?;
    let mut file = fs::File::from(owned);
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

/// `apply_candidate` の upfront 一括検証専用の「歩くだけ」プローブ。
///
/// [`crate::candidate::apply_candidate`] は「候補が複数ファイルにまたがる
/// 場合、一部だけ書き換わった状態で `Err` を返さない」契約を持つ（doc
/// 参照）。これを実現するには実際の書き込みより前に全パスを検証し尽くす
/// 必要がある。本関数は [`walk_to_final`] と同じ fd 走査を行うが、開いた
/// fd は即座に破棄し内容の読み書きは行わない（検証のみ）。
///
/// 旧 `reject_symlink_escape` と同じ許容規則を踏襲する: 未実在
/// （`NotFound`）の中間ディレクトリ・末端ファイルは「まだ作成されていない
/// だけ」として許容し、実在する symlink（`ELOOP`）や非ディレクトリ
/// （`ENOTDIR`）等の異常のみを拒否する。
///
/// 本関数の判定と実際の書き込み（[`write_via_fd_walk`]）呼び出しとの間には
/// 検査後の再解決を挟む余地（TOCTOU window）が理論上存在するが、
/// [`write_via_fd_walk`] 自身が呼び出し時に改めて `O_NOFOLLOW` 付きで
/// fd 走査するため、この window を悪用しても最終的な書き込みは拒否される
/// （本関数は「早期失敗によるファイル書き換えの部分適用防止」が目的であり、
/// symlink 拒否そのものの安全性は保証しない・保証する必要もない）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn probe(workspace: &Path, relative_path: &Path) -> io::Result<()> {
    match walk_to_final(workspace, relative_path, raw::O_RDONLY) {
        Ok(_owned) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Linux・macOS 以外向けの fail-closed フォールバック（モジュール冒頭 doc
/// 参照）。`O_NOFOLLOW`/`O_DIRECTORY` の値の出典を持たないターゲットでは
/// 候補適用そのものを一律拒否する。
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn unsupported_platform_error() -> io::Error {
    io::Error::other(
        "候補修正の適用は Linux・macOS のみサポートします（fd 走査ベースの \
         symlink TOCTOU 対策〈openat の O_NOFOLLOW〉が定義された値を持つ \
         プラットフォームに限定されるため、それ以外では fail-closed に \
         拒否します）",
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn write_via_fd_walk(
    _workspace: &Path,
    _relative_path: &Path,
    _content: &str,
) -> io::Result<()> {
    Err(unsupported_platform_error())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn read_via_fd_walk(_workspace: &Path, _relative_path: &Path) -> io::Result<String> {
    Err(unsupported_platform_error())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn probe(_workspace: &Path, _relative_path: &Path) -> io::Result<()> {
    Err(unsupported_platform_error())
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;

    fn temp_workspace(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "self-repair-fd-walk-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create_dir_all should succeed in test setup");
        dir
    }

    #[test]
    fn write_then_read_plain_path_round_trips() {
        let dir = temp_workspace("plain-round-trip");
        fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");
        let relative = Path::new("src/lib.rs");

        write_via_fd_walk(&dir, relative, "original")
            .expect("write_via_fd_walk should succeed for a plain existing parent dir");
        let content =
            read_via_fd_walk(&dir, relative).expect("read_via_fd_walk should succeed after write");
        assert_eq!(content, "original");

        write_via_fd_walk(&dir, relative, "updated")
            .expect("write_via_fd_walk should overwrite existing file content");
        let content = read_via_fd_walk(&dir, relative)
            .expect("read_via_fd_walk should see overwritten content");
        assert_eq!(content, "updated");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_creates_new_file_via_existing_parent_dir() {
        let dir = temp_workspace("new-file-via-parent");
        fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");
        let relative = Path::new("src/new_module.rs");
        assert!(!dir.join(relative).exists());

        write_via_fd_walk(&dir, relative, "pub fn added() {}")
            .expect("write_via_fd_walk should create a new file under an existing parent dir");
        let content = read_via_fd_walk(&dir, relative)
            .expect("read_via_fd_walk should read the newly created file");
        assert_eq!(content, "pub fn added() {}");

        // `openat_final` の `O_CREAT` 呼び出しは可変長引数側に `mode = 0o600`
        // を渡す（[`openat`] doc 参照）。ここで実際に owner rw のみが立ち
        // group/other は 0 であることを確認することで、固定 4 引数宣言に
        // よる ABI 不整合（advisor 指摘: `aarch64-apple-darwin` では可変長
        // 引数がスタック経由になり、固定引数宣言だと `mode` が読めず不定値に
        // なりうる）が紛れ込んでいないことをこのテストで検出できるようにする。
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(dir.join(relative))
                .expect("metadata should succeed")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(
                mode, 0o600,
                "作成されたファイルの権限が想定外です: {mode:o}"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_leaf_symlink_without_touching_outside_target() {
        let dir = temp_workspace("leaf-symlink-reject");
        let outside_dir = temp_workspace("leaf-symlink-reject-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "do-not-overwrite").expect("write should succeed in test setup");

        let relative = Path::new("target.txt");
        std::os::unix::fs::symlink(&outside_file, dir.join(relative))
            .expect("symlink creation should succeed in test setup");

        let write_result = write_via_fd_walk(&dir, relative, "pwned");
        assert!(write_result.is_err());
        assert_eq!(
            fs::read_to_string(&outside_file).expect("read should succeed"),
            "do-not-overwrite"
        );

        let read_result = read_via_fd_walk(&dir, relative);
        assert!(read_result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_intermediate_directory_symlink_without_touching_outside_target() {
        let dir = temp_workspace("dir-symlink-reject");
        let outside_dir = temp_workspace("dir-symlink-reject-outside");
        fs::create_dir_all(&outside_dir).expect("create_dir_all should succeed in test setup");

        std::os::unix::fs::symlink(&outside_dir, dir.join("sub"))
            .expect("symlink creation should succeed in test setup");
        let relative = Path::new("sub/target.txt");

        let write_result = write_via_fd_walk(&dir, relative, "pwned");
        assert!(write_result.is_err());
        assert!(!outside_dir.join("target.txt").exists());

        let read_result = read_via_fd_walk(&dir, relative);
        assert!(read_result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn probe_allows_not_yet_existing_path() {
        let dir = temp_workspace("probe-not-found");
        fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");

        let result = probe(&dir, Path::new("src/not_created_yet.rs"));
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn probe_rejects_existing_symlink() {
        let dir = temp_workspace("probe-symlink-reject");
        let outside_dir = temp_workspace("probe-symlink-reject-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "secret").expect("write should succeed in test setup");

        let relative = Path::new("target.txt");
        std::os::unix::fs::symlink(&outside_file, dir.join(relative))
            .expect("symlink creation should succeed in test setup");

        let result = probe(&dir, relative);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }
}
