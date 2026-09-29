//! self-repair 単体テスト（`#[cfg(test)]`）共通の一時ディレクトリ RAII ガード。公開 API ではない。
//!
//! lib（`crate::test_support` 経由で再輸出）と bin（`main.rs` が `#[path]` で取り込む）の両方から使う。
//! bin のテストビルドは非テスト版 lib をリンクするため `crate::test_support` に到達できず、
//! `test_support.rs` は `crate::exec` に依存して `#[path]` 取り込みできないため、std のみに依存する
//! 本ファイルへ分離した（dead_code 回避のため lib・bin 双方で使う項目だけを置く）。
//!
//! 参照実装: `crates/facade/tests/compat_sequential_model_io_manual.rs::TempDirGuard::new`（6130760f）・
//! `crates/guardrail/src/test_support.rs`（#2409）。イシュー #2381（親 #2363）。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 同一プロセス内の連番。pid・時刻が同一でも名前が衝突しないようにする。
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 一意名で**排他作成**した一時ディレクトリを保持し、Drop で自身のパスだけを削除する RAII ガード。
///
/// - 名前は `fandhe-ai-self-repair-unit-{pid}-{nanos}-{seq}-{label}`。`create_dir` は既存パスに失敗するため
///   第三者が先に置いたディレクトリ・symlink を再利用しない（`AlreadyExists` のみ再試行）。
/// - 作成前の `remove_dir_all` は行わない（無関係な既存パスを消さない）。
/// - **注意**: 名前付き変数へ束縛して保持すること。`let _ = ..`・`let (_, p) = ..`・
///   `unique_temp_dir(..).path().to_path_buf()` は即 drop されディレクトリが削除される。
pub(crate) struct TempDirGuard {
    path: PathBuf,
}

/// `label` を名前末尾に含む一時ディレクトリを排他作成する。失敗時は panic（テスト専用）。
pub(crate) fn unique_temp_dir(label: &str) -> TempDirGuard {
    let pid = std::process::id();
    for _ in 0..64 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = std::env::temp_dir().join(format!(
            "fandhe-ai-self-repair-unit-{pid}-{nanos}-{seq}-{label}"
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return TempDirGuard { path: candidate },
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("一時ディレクトリの作成に失敗: {candidate:?}: {e}"),
        }
    }
    panic!("一時ディレクトリの一意名を確保できなかった（64 回衝突）: label={label}");
}

impl TempDirGuard {
    /// 作成済みディレクトリのパス。
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        // std の remove_dir_all は symlink を辿らない。自身が作成したパスのみ削除する。
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
