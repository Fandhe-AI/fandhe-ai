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

/// `dir` 直下に `next_name()` の名前で `O_CREAT | O_EXCL | O_NOFOLLOW`（0600）作成する。
/// `AlreadyExists` は既存に触れず名前を作り直し、`MAX_TMP_NAME_ATTEMPTS` 回で諦める。
pub(super) fn create_tmp(
    dir: &OwnedFd,
    mut next_name: impl FnMut() -> String,
) -> io::Result<(File, String)> {
    let flags = libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC;
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

/// `renameat`（同一ディレクトリ fd 内の atomic replace）。
pub(super) fn rename_in(dir: &OwnedFd, from: &str, to: &str) -> io::Result<()> {
    let f = cstr(from)?;
    let t = cstr(to)?;
    // SAFETY: `dir` は有効な fd で新旧とも同一 fd 基準。`f`／`t` は NUL 終端で呼び出し中生存する。
    let rc = unsafe { libc::renameat(dir.as_raw_fd(), f.as_ptr(), dir.as_raw_fd(), t.as_ptr()) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `dir` 直下の `name`（`AT_SYMLINK_NOFOLLOW`）が保持 fd `held` と同一の通常ファイル
/// （`(dev, ino)` 一致）かを返す。名前が存在しない・stat 失敗は `false`。
pub(super) fn names_same_entity(dir: &OwnedFd, name: &str, held: &File) -> bool {
    let Ok(c) = cstr(name) else {
        return false;
    };
    // SAFETY: `libc::stat` は全ビット 0 が有効な POD。以後 libc が書き込む。
    let mut on_disk: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `dir` は有効な fd、`c` は NUL 終端、`on_disk` は書き込み可能な有効領域。
    let rc = unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            c.as_ptr(),
            &mut on_disk,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        return false;
    }
    // SAFETY: 同上（POD のゼロ初期化）。
    let mut held_st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `held` は有効な fd、`held_st` は書き込み可能な有効領域。
    if unsafe { libc::fstat(held.as_raw_fd(), &mut held_st) } != 0 {
        return false;
    }
    (on_disk.st_mode & libc::S_IFMT) == libc::S_IFREG
        && on_disk.st_dev == held_st.st_dev
        && on_disk.st_ino == held_st.st_ino
}

/// `dir` 直下の `name` を無条件に `unlinkat` する（best-effort）。公開後の再照合で未検証の
/// 実体が公開名に居ると分かった場合の撤去用。
pub(super) fn unlink_name(dir: &OwnedFd, name: &str) {
    if let Ok(c) = cstr(name) {
        // SAFETY: `dir` は有効な fd、`c` は NUL 終端で呼び出し中生存する。
        let _ = unsafe { libc::unlinkat(dir.as_raw_fd(), c.as_ptr(), 0) };
    }
}

/// 自分が作って fd を保持している一時ファイルの削除（best-effort）。保持 fd と同一実体
/// （`names_same_entity`）のときだけ `unlinkat` する（差し替え検出時は削除しない）。
pub(super) fn unlink_own(dir: &OwnedFd, name: &str, held: &File) {
    if names_same_entity(dir, name, held) {
        unlink_name(dir, name);
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
        rename_in(&v, &name, "model.safetensors").unwrap();
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
    fn unlink_own_removes_only_matching_entity() {
        let root = tmpdir("unlink");
        let r = open_root(&root).unwrap();
        let (f, name) = create_tmp(&r, || "t1".into()).unwrap();
        unlink_own(&r, &name, &f);
        assert!(!root.join("t1").exists());
        // 別の実体に差し替えられた場合は消さない。
        let (f2, name2) = create_tmp(&r, || "t2".into()).unwrap();
        std::fs::remove_file(root.join("t2")).unwrap();
        std::fs::write(root.join("t2"), b"other").unwrap();
        unlink_own(&r, &name2, &f2);
        assert_eq!(std::fs::read(root.join("t2")).unwrap(), b"other");
        let _ = std::fs::remove_dir_all(&root);
    }
}
