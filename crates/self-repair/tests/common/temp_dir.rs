//! 結合テスト（`tests/*.rs`）共通の一時パスヘルパ。src 側の `#[cfg(test)]` の `test_support` と
//! 同一方式（結合テストは別バイナリのため crate 内部を参照できない）。
//! イシュー #2382（親 #2363）。参照実装は facade の `TempDirGuard::new`（6130760f）・#2378・#2380。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 同一プロセス内の連番。pid・時刻が同一でも名前が衝突しないようにする。
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// `fandhe-ai-self-repair-it-{pid}-{nanos}-{seq}-{label}` 形式の一意な候補パスを返す（作成はしない）。
/// `TempDirGuard::new`・`temp_file::TempFileGuard::new` が呼ぶ。呼ぶたびに nanos・seq を取り直す。
pub fn unique_temp_candidate(label: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "fandhe-ai-self-repair-it-{pid}-{nanos}-{seq}-{label}"
    ))
}

/// 一時ディレクトリを一意名で**排他作成**し、Drop で自身のパスだけを削除する RAII ガード
/// （panic しても一時物を残さない）。
///
/// - `std::fs::create_dir` は既存パスに失敗するため、第三者が先に置いたディレクトリ・symlink を
///   再利用しない（`AlreadyExists` は再試行）。作成前の `remove_dir_all` は行わない。
/// - **注意**: 名前付き変数へ束縛して保持すること。`let _ = ..` や `.path().to_path_buf()` の
///   一時値化は即 drop されてディレクトリが削除される。
pub struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    /// `label` を名前末尾に含む一時ディレクトリを排他作成する。失敗時は panic（テスト専用）。
    pub fn new(label: &str) -> Self {
        for _ in 0..64 {
            let candidate = unique_temp_candidate(label);
            match std::fs::create_dir(&candidate) {
                Ok(()) => return Self { path: candidate },
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("一時ディレクトリの作成に失敗: {candidate:?}: {e}"),
            }
        }
        panic!("一時ディレクトリの一意名を確保できなかった（64 回衝突）: label={label}");
    }

    /// 作成済みディレクトリのパス。
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        // std の remove_dir_all は symlink を辿らない。自身が作成したパスのみ削除する。
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
