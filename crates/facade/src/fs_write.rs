//! 書き込み側の共有ヘルパー（std のみ・unix 限定）。
//!
//! `compat::model_io`（`save_model`／Functional 保存）と `crate::model`
//! （`ModelRegistry::save`）が共有する、`create_new` による一時ファイル確保（衝突時の
//! 名前再生成）・自己所有の一時ファイルだけを消す後始末・一時名の素材生成を持つ。
//! `crate::fs_guard`（読み取り側）の対になる書き込み側の置き場で、公開面ではない。
//!
//! 出典: `docs/facade-model-registry-decision.md` §16.4、
//! `docs/compat-model-io-decision.md` §12.3・§13.6。`compat::model_io` から挙動を変えず
//! に移設した（戻り値型のみ `ModelIoError` から `io::Result` へ変更。型付きエラーへの
//! 変換は各呼び出し元が行う）。

#![cfg(unix)]

use std::io;
use std::path::Path;

/// `create_new` が `AlreadyExists` を返したときの名前再生成の試行上限。
/// `crates/docs-site/src/build.rs` の同名定数と同値（決定記録 §12.3 手順 1）。
pub(crate) const MAX_TMP_NAME_ATTEMPTS: usize = 8;

/// `dir` 直下に `next_name()` の名前で `create_new` する。`AlreadyExists`（既存エントリの
/// 種類は問わない）は既存に一切触れずに名前を作り直し、`max_attempts` 回で諦める
/// （`Io(AlreadyExists)`。決定記録 §12.3 手順 1・2）。衝突を注入できるよう名前生成を引数にする。
pub(crate) fn create_new_with_retry(
    dir: &Path,
    mut next_name: impl FnMut() -> String,
    max_attempts: usize,
) -> io::Result<(std::fs::File, String)> {
    for _ in 0..max_attempts {
        let name = next_name();
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join(&name))
        {
            Ok(file) => return Ok((file, name)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!("{max_attempts} 回試行しても一意なファイル名を確保できませんでした"),
    ))
}

/// 自分が作って fd を保持している一時ファイルの削除（決定記録 §13.6。best-effort）。
/// 保持ハンドルの `fstat` と削除直前の `symlink_metadata` の `(dev, ino)` が一致し、
/// かつ通常ファイルのときだけ `remove_file` する（差し替え検出時は削除しない）。
pub(crate) fn remove_own_tmp(file: &std::fs::File, path: &Path) {
    use std::os::unix::fs::MetadataExt;
    let Ok(held) = file.metadata() else {
        return;
    };
    let Ok(on_disk) = std::fs::symlink_metadata(path) else {
        return;
    };
    if on_disk.file_type().is_file() && on_disk.dev() == held.dev() && on_disk.ino() == held.ino() {
        let _ = std::fs::remove_file(path);
    }
}

static NAME_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// pid・ナノ秒・プロセス内カウンタの組（一時名・世代 ID の素材）。
pub(crate) fn name_entropy() -> (u32, u128, u64) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let counter = NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (std::process::id(), nanos, counter)
}
