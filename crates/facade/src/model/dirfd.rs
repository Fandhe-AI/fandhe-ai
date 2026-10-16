//! dirfd 相対のキャッシュ書き込み（イシュー #3125。`cfg(unix)`。libc の薄いラッパー）。
//!
//! `ModelRegistry::download` が `<root>/<name>/<version>/` を 1 段ずつ `O_NOFOLLOW |
//! O_DIRECTORY` で開き（無ければ `mkdirat`）、以後の一時ファイル作成・公開（`renameat`）・
//! 後始末（`unlinkat`）をすべて保持した fd 基準で行う。パス文字列を再解決しないため、
//! 作成中に `<name>`／`<version>` をシンボリックリンクへ差し替えられても書き込みはルート外へ
//! 出ない（A01。`docs/model-download-design.md` §6 (a)(b)）。`openat2` は使わず
//! `openat` + `O_NOFOLLOW | O_DIRECTORY` の 1 段ずつの経路を採る（Linux／macOS 共通・unsafe
//! 最小。§6 (a) が同格と認める経路）。`unsafe` は本ファイルの libc 呼び出しに限る
//! （`.claude/rules/deps-policy.md` 第 10 区分 `libc`。定数は libc のものを使い手書きしない）。

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use crate::fs_write::{MAX_TMP_NAME_ATTEMPTS, name_entropy};

fn cstr(name: &str) -> io::Result<CString> {
    CString::new(name)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "名前に NUL が含まれています"))
}

/// `openat` の薄いラッパー。成功時は所有 fd を返す。
fn openat_owned(
    dir: &OwnedFd,
    name: &str,
    flags: libc::c_int,
    mode: libc::c_uint,
) -> io::Result<OwnedFd> {
    let c = cstr(name)?;
    // SAFETY: `dir` は呼び出し中生存する有効な fd。`c` は NUL 終端の `CString` で呼び出し中
    // 生存する。`mode` は `O_CREAT` 時のみ参照される variadic 引数で `c_uint` を渡している。
    let fd = unsafe { libc::openat(dir.as_raw_fd(), c.as_ptr(), flags, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `openat` が成功して返した fd はこのプロセスだけが所有し、他に所有者がいない。
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// ルートを（無ければ作成して）ディレクトリ fd として開く。ルート自体がシンボリックリンク
/// でも許容する（利用者が選んだ配置。`save`／`load` と同じ扱い）。
pub(super) fn open_root(root: &Path) -> io::Result<OwnedFd> {
    std::fs::create_dir_all(root)?;
    // `O_DIRECTORY | O_NONBLOCK`: 作成後に FIFO 等へ差し替えられても writer 待ちで open が
    // ブロックしない（DoS 回避。`save` は `O_DIRECTORY`、`load` は `O_NONBLOCK` で同趣旨）。
    // `O_NOFOLLOW` は付けない（ルート自体のシンボリックリンクは許容する）。
    let c = CString::new(root.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "パスに NUL が含まれています"))?;
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NONBLOCK | libc::O_CLOEXEC;
    // SAFETY: `c` は NUL 終端の `CString` で呼び出し中生存する。`O_CREAT` なしのため mode 不要。
    let fd = unsafe { libc::open(c.as_ptr(), flags) };
    if fd < 0 {
        let e = io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::ENOTDIR) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "キャッシュルートがディレクトリではありません",
            ));
        }
        return Err(e);
    }
    // SAFETY: `open` が成功して返した fd はこのプロセスだけが所有する。
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// `parent` 直下の `component` を `O_NOFOLLOW | O_DIRECTORY` で開く。無ければ `mkdirat` で
/// 作って開き直す。シンボリックリンク・非ディレクトリは `InvalidInput`（パスは含めない）。
pub(super) fn open_or_create_dir(parent: &OwnedFd, component: &str) -> io::Result<OwnedFd> {
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    let try_open = || openat_owned(parent, component, flags, 0);
    let reject = |e: io::Error| match e.raw_os_error() {
        Some(code) if code == libc::ELOOP || code == libc::ENOTDIR => io::Error::new(
            io::ErrorKind::InvalidInput,
            "中間ディレクトリがシンボリックリンクまたは非ディレクトリです",
        ),
        _ => e,
    };
    match try_open() {
        Ok(fd) => return Ok(fd),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(reject(e)),
    }
    let c = cstr(component)?;
    // SAFETY: `parent` は有効な fd、`c` は NUL 終端で呼び出し中生存する。
    let rc = unsafe { libc::mkdirat(parent.as_raw_fd(), c.as_ptr(), 0o755) };
    if rc != 0 {
        let e = io::Error::last_os_error();
        // 競合で先に作られた場合は許容し、開き直して種別を検証する。
        if e.kind() != io::ErrorKind::AlreadyExists {
            return Err(e);
        }
    }
    try_open().map_err(reject)
}

/// ルートを作成せずにディレクトリ fd として開く（キャッシュ参照段用。#3126）。存在しなければ
/// `None`。ルート自体のシンボリックリンクは [`open_root`] と同じく許容する。
pub(super) fn open_root_existing(root: &Path) -> io::Result<Option<OwnedFd>> {
    let c = CString::new(root.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "パスに NUL が含まれています"))?;
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NONBLOCK | libc::O_CLOEXEC;
    // SAFETY: `c` は NUL 終端の `CString` で呼び出し中生存する。`O_CREAT` なしのため mode 不要。
    let fd = unsafe { libc::open(c.as_ptr(), flags) };
    if fd < 0 {
        let e = io::Error::last_os_error();
        return match e.raw_os_error() {
            Some(code) if code == libc::ENOENT => Ok(None),
            Some(code) if code == libc::ENOTDIR => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "キャッシュルートがディレクトリではありません",
            )),
            _ => Err(e),
        };
    }
    // SAFETY: `open` が成功して返した fd はこのプロセスだけが所有する。
    Ok(Some(unsafe { OwnedFd::from_raw_fd(fd) }))
}

/// `parent` 直下の `component` を作成せずに `O_NOFOLLOW | O_DIRECTORY` で開く（#3126）。
/// 存在しなければ `None`。シンボリックリンク・非ディレクトリは [`open_or_create_dir`] と同じく
/// `InvalidInput`。
pub(super) fn open_existing_dir(parent: &OwnedFd, component: &str) -> io::Result<Option<OwnedFd>> {
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    match openat_owned(parent, component, flags, 0) {
        Ok(fd) => Ok(Some(fd)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(match e.raw_os_error() {
            Some(code) if code == libc::ELOOP || code == libc::ENOTDIR => io::Error::new(
                io::ErrorKind::InvalidInput,
                "中間ディレクトリがシンボリックリンクまたは非ディレクトリです",
            ),
            _ => e,
        }),
    }
}

/// `dir` 直下の通常ファイル `name` を読み取り専用で開く（#3126。`download.json` と既存
/// `model.safetensors` の再検証用）。`O_NOFOLLOW`（リンクは拒否）かつ `O_NONBLOCK`（FIFO で
/// open がブロックしない）で開き、`fstat` で通常ファイルであることと `st_size <= max` を確認する。
/// 不在は `None`、非通常ファイル・リンクは `InvalidInput`、上限超過は `InvalidData`。
pub(super) fn open_file_no_follow(dir: &OwnedFd, name: &str, max: u64) -> io::Result<Option<File>> {
    let flags = libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
    let fd = match openat_owned(dir, name, flags, 0) {
        Ok(fd) => fd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "シンボリックリンクは受け付けません",
            ));
        }
        Err(e) => return Err(e),
    };
    // SAFETY: `libc::stat` は全ビット 0 が有効な POD。
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `fd` は有効な fd、`st` は書き込み可能な有効領域。
    if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if (st.st_mode & libc::S_IFMT) != libc::S_IFREG {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "通常ファイルではありません",
        ));
    }
    if u64::try_from(st.st_size).map_or(true, |len| len > max) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ファイルサイズが上限を超えています",
        ));
    }
    Ok(Some(File::from(fd)))
}

/// `dir` 直下に `next_name()` の名前で `O_CREAT | O_EXCL | O_NOFOLLOW`（0600）作成する。
/// `AlreadyExists` は既存に触れず名前を作り直し、`MAX_TMP_NAME_ATTEMPTS` 回で諦める。
pub(super) fn create_tmp(
    dir: &OwnedFd,
    mut next_name: impl FnMut() -> String,
) -> io::Result<(File, String)> {
    let flags = libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    for _ in 0..MAX_TMP_NAME_ATTEMPTS {
        let name = next_name();
        match openat_owned(dir, &name, flags, 0o600) {
            Ok(fd) => return Ok((File::from(fd), name)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "一意な一時ファイル名を確保できませんでした",
    ))
}

/// 本番の一時ファイル名（先頭 `.` のため `validate_component` を通らず、`load`／
/// `available_models` からは見えない）。
pub(super) fn production_tmp_name(leaf: &str) -> String {
    let (pid, nanos, counter) = name_entropy();
    format!(".{leaf}.dl-{pid}-{counter}-{nanos}")
}

/// 公開前の検証済みファイルを置く private staging ディレクトリ名（先頭 `.` のため
/// `validate_component` を通らず `load`／`available_models` からは見えない）。
pub(super) fn production_staging_name(leaf: &str) -> String {
    let (pid, nanos, counter) = name_entropy();
    format!(".{leaf}.stg-{pid}-{counter}-{nanos}")
}

/// `parent` 直下に `next_name()` の名前で 0700 の private staging ディレクトリを新規作成して
/// 開く。共有キャッシュを書き換えられる第三者が一時ファイルを差し替えられないよう、検証済み
/// 実体は常にこのディレクトリ内に置く（A08。作成者のみが入れるため、検証した実体と公開する
/// 実体の一致が名前の再照合なしで構造的に保証される）。`EEXIST` は名前を作り直す。開いた
/// 実体が自分の所有・group/other アクセス不可であることを `fstat` で確認し、作成〜open の
/// 隙間に他者のディレクトリへ差し替えられた場合は fail-closed にする。
pub(super) fn create_staging(
    parent: &OwnedFd,
    mut next_name: impl FnMut() -> String,
) -> io::Result<(OwnedFd, String)> {
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    for _ in 0..MAX_TMP_NAME_ATTEMPTS {
        let name = next_name();
        let c = cstr(&name)?;
        // SAFETY: `parent` は有効な fd、`c` は NUL 終端で呼び出し中生存する。
        let rc = unsafe { libc::mkdirat(parent.as_raw_fd(), c.as_ptr(), 0o700) };
        if rc != 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::AlreadyExists {
                continue;
            }
            return Err(e);
        }
        let fd = openat_owned(parent, &name, flags, 0)?;
        // SAFETY: `libc::stat` は全ビット 0 が有効な POD。
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `fd` は有効な fd、`st` は書き込み可能な有効領域。
        let stat_ok = unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } == 0;
        // SAFETY: 引数なし・副作用なしの libc 呼び出し。
        let euid = unsafe { libc::geteuid() };
        if !stat_ok || st.st_uid != euid || (st.st_mode & 0o077) != 0 {
            return Err(io::Error::other(
                "staging ディレクトリが他者のものに差し替えられたため使用しません",
            ));
        }
        return Ok((fd, name));
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "一意な staging ディレクトリ名を確保できませんでした",
    ))
}

/// `renameat`（`from_dir` の `from` を `to_dir` の `to` へ atomic replace。fd 基準）。
pub(super) fn rename_across(
    from_dir: &OwnedFd,
    from: &str,
    to_dir: &OwnedFd,
    to: &str,
) -> io::Result<()> {
    let f = cstr(from)?;
    let t = cstr(to)?;
    // SAFETY: 両 dir は有効な fd。`f`／`t` は NUL 終端で呼び出し中生存する。
    let rc = unsafe {
        libc::renameat(
            from_dir.as_raw_fd(),
            f.as_ptr(),
            to_dir.as_raw_fd(),
            t.as_ptr(),
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `dir` 直下の `name` を `unlinkat` する（best-effort）。`dir` は private staging のみで使い、
/// 共有ディレクトリの公開名には使わない（別処理の成果物を消さないため）。
pub(super) fn unlink_in(dir: &OwnedFd, name: &str) {
    if let Ok(c) = cstr(name) {
        // SAFETY: `dir` は有効な fd、`c` は NUL 終端で呼び出し中生存する。
        let _ = unsafe { libc::unlinkat(dir.as_raw_fd(), c.as_ptr(), 0) };
    }
}

/// private staging ディレクトリ自体の削除（best-effort。空でなければ `rmdir` が失敗して残す）。
pub(super) fn remove_staging(parent: &OwnedFd, name: &str) {
    if let Ok(c) = cstr(name) {
        // SAFETY: `parent` は有効な fd、`c` は NUL 終端で呼び出し中生存する。
        let _ = unsafe { libc::unlinkat(parent.as_raw_fd(), c.as_ptr(), libc::AT_REMOVEDIR) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::symlink;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let (pid, nanos, counter) = name_entropy();
        let p = std::env::temp_dir().join(format!("fandhe-dirfd-{tag}-{pid}-{counter}-{nanos}"));
        std::fs::create_dir_all(&p).expect("mkdir");
        p
    }

    #[test]
    fn creates_chain_and_tolerates_existing() {
        let root = tmpdir("chain");
        let r = open_root(&root).unwrap();
        let _n = open_or_create_dir(&r, "name").unwrap();
        // 既存（EEXIST 競合後の開き直しと同じ経路）。
        let _n2 = open_or_create_dir(&r, "name").unwrap();
        assert!(root.join("name").is_dir());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_symlinked_component_without_following() {
        let root = tmpdir("symlink");
        let outside = tmpdir("outside");
        symlink(&outside, root.join("name")).unwrap();
        let r = open_root(&root).unwrap();
        let err = open_or_create_dir(&r, "name").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(std::fs::read_dir(&outside).unwrap().next().is_none());
        // 非ディレクトリ。
        std::fs::write(root.join("file"), b"x").unwrap();
        assert_eq!(
            open_or_create_dir(&r, "file").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn writes_stay_inside_held_dir_after_swap() {
        let root = tmpdir("swap");
        let outside = tmpdir("swap-out");
        let r = open_root(&root).unwrap();
        let v = open_or_create_dir(&r, "v").unwrap();
        // fd を保持したまま実ディレクトリをリンクへ差し替える。
        std::fs::rename(root.join("v"), root.join("v.moved")).unwrap();
        symlink(&outside, root.join("v")).unwrap();
        let (mut f, name) = create_tmp(&v, || production_tmp_name("m")).unwrap();
        f.write_all(b"data").unwrap();
        rename_across(&v, &name, &v, "model.safetensors").unwrap();
        assert!(std::fs::read_dir(&outside).unwrap().next().is_none());
        assert!(root.join("v.moved/model.safetensors").is_file());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn create_tmp_retries_on_collision_then_gives_up() {
        let root = tmpdir("collide");
        let r = open_root(&root).unwrap();
        std::fs::write(root.join("taken"), b"x").unwrap();
        let mut calls = 0;
        let (_f, name) = create_tmp(&r, || {
            calls += 1;
            if calls < 3 {
                "taken".into()
            } else {
                "fresh".into()
            }
        })
        .unwrap();
        assert_eq!(name, "fresh");
        assert_eq!(std::fs::read(root.join("taken")).unwrap(), b"x");
        let err = create_tmp(&r, || "taken".into()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_root_does_not_block_on_fifo() {
        let root = tmpdir("fifo");
        let fifo = root.join("f");
        let c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: `c` は NUL 終端で呼び出し中生存する。
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        // writer 不在でもブロックせずエラーで返る（ハングすればテストがタイムアウトする）。
        assert!(open_root(&fifo).is_err());
        // create_dir_all 通過後に FIFO へ差し替わる競合は、fd 取得を直接 O_NONBLOCK で再現できない
        // ため、ディレクトリ以外が open_root に渡らないことを上記で担保する。
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn staging_is_private_and_removable() {
        use std::os::unix::fs::PermissionsExt;
        let root = tmpdir("staging");
        let r = open_root(&root).unwrap();
        let (sfd, sname) = create_staging(&r, || ".s1".into()).unwrap();
        let mode = std::fs::metadata(root.join(".s1"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0);
        // 名前衝突は既存に触れず作り直す。
        let mut c = 0;
        let (_s2, n2) = create_staging(&r, || {
            c += 1;
            if c == 1 { ".s1".into() } else { ".s2".into() }
        })
        .unwrap();
        assert_eq!(n2, ".s2");
        let (mut f, tname) = create_tmp(&sfd, || "t".into()).unwrap();
        f.write_all(b"x").unwrap();
        rename_across(&sfd, &tname, &r, "model.safetensors").unwrap();
        assert!(root.join("model.safetensors").is_file());
        remove_staging(&r, &sname);
        assert!(!root.join(".s1").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn existing_dir_helpers_do_not_create_and_reject_links() {
        let root = tmpdir("existing");
        let missing = root.join("nope");
        assert!(open_root_existing(&missing).unwrap().is_none());
        assert!(!missing.exists());
        let r = open_root_existing(&root).unwrap().unwrap();
        assert!(open_existing_dir(&r, "a").unwrap().is_none());
        assert!(!root.join("a").exists());
        std::fs::create_dir(root.join("a")).unwrap();
        assert!(open_existing_dir(&r, "a").unwrap().is_some());
        let outside = tmpdir("existing-out");
        symlink(&outside, root.join("l")).unwrap();
        assert_eq!(
            open_existing_dir(&r, "l").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn open_file_no_follow_checks_kind_and_size() {
        let root = tmpdir("nofollow");
        let r = open_root(&root).unwrap();
        assert!(open_file_no_follow(&r, "f", 10).unwrap().is_none());
        std::fs::write(root.join("f"), b"12345").unwrap();
        assert!(open_file_no_follow(&r, "f", 5).unwrap().is_some());
        assert_eq!(
            open_file_no_follow(&r, "f", 4).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        symlink(root.join("f"), root.join("l")).unwrap();
        assert_eq!(
            open_file_no_follow(&r, "l", 10).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        std::fs::create_dir(root.join("d")).unwrap();
        assert!(open_file_no_follow(&r, "d", 10).is_err());
        let fifo = CString::new(root.join("p").as_os_str().as_bytes()).unwrap();
        // SAFETY: `fifo` は NUL 終端で呼び出し中生存する。
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        // writer 不在でもブロックせず、通常ファイルでないため拒否される。
        assert_eq!(
            open_file_no_follow(&r, "p", 10).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
