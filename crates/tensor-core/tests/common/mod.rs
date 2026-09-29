//! 統合テスト用一時ディレクトリ RAII ガード（イシュー #2385・親 #2363）。
//!
//! `tests/io_npy.rs` と `tests/io_npz.rs` から `mod common;` 経由で使われる。
//! pid・ナノ秒時刻・プロセス内カウンタ・ラベルによる一意名を `create_dir` で排他作成し、
//! 既存パスを再利用せず作成前の削除もしない。`Drop` は作成に成功したパスだけを
//! `remove_dir_all` する（std のそれは symlink を辿らない）。参照実装: コミット 6130760f の
//! `compat_sequential_model_io_manual.rs::TempDirGuard::new`。
//!
//! 公開 API は `new`・`path`・`Drop` のみ（Windows の `--tests` clippy で使われない
//! メソッドが `dead_code` になるのを避ける。`#[allow]` で黙らせない）。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 排他作成した一時ディレクトリの所有者。drop 時に自身のパスだけを削除する。
/// 名前付き変数へ束縛して保持すること（`let _ = ...` は即 drop される）。
pub struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    /// `label` を含む一意名のディレクトリを `create_dir` で排他作成する。
    pub fn new(label: &str) -> Self {
        let pid = std::process::id();
        for _ in 0..64 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-ai-tensor-core-test-{pid}-{nanos}-{seq}-{label}"
            ));
            match std::fs::create_dir(&candidate) {
                Ok(()) => return Self { path: candidate },
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("一時ディレクトリを作成できない: {e}"),
            }
        }
        panic!("一意な一時ディレクトリ名を 64 回試行しても確保できない");
    }

    /// 一時ディレクトリのパス。
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
