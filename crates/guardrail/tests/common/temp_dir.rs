//! 結合テスト（`tests/*.rs`）共通の一時ディレクトリヘルパ。src 側の
//! `src/test_support.rs` と同一方式（結合テストは別バイナリのため crate 内部を参照できない）。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 同一プロセス内の連番。pid・時刻が同一でも名前が衝突しないようにする。
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// `tests/*.rs` の結合テスト用の一時ディレクトリを一意名で**排他作成**し、
/// Drop で自身のパスだけを削除する RAII ガード。
///
/// - 名前は `fandhe-ai-guardrail-it-{pid}-{nanos}-{seq}-{label}`。`std::fs::create_dir` は既存パスに
///   失敗するため、第三者が先に置いたディレクトリ・symlink を再利用しない（`AlreadyExists` は再試行）。
/// - 作成前の `remove_dir_all` は行わない（無関係な既存ディレクトリを消さない）。
/// - 参照実装: `crates/facade/tests/compat_sequential_model_io_manual.rs::TempDirGuard::new`（6130760f）・
///   #2378 の `crates/facade/tests/common/temp_dir.rs`。イシュー #2380（親 #2363）。
/// - **注意**: 名前付き変数へ束縛して保持すること。`let _ = ..` や `.path().to_path_buf()` の
///   一時値化は即 drop されてディレクトリが削除される。
pub struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    /// `label` を名前末尾に含む一時ディレクトリを排他作成する。失敗時は panic（テスト専用）。
    pub fn new(label: &str) -> Self {
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for _ in 0..64 {
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-ai-guardrail-it-{pid}-{nanos}-{seq}-{label}"
            ));
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
