//! docs-site 統合テスト（`tests/*.rs`）共通の一時出力ディレクトリヘルパー
//! （イシュー #2386・親 #2363）。`cli_fail_closed.rs`・`site_nav.rs`・
//! `real_site_search_index.rs` が `mod common;` で取り込む。
//!
//! - root はプロセス ID・ナノ秒・プロセス内カウンタ・タグで一意名を作り、
//!   `std::fs::create_dir` で**排他作成**する（既存パスは `AlreadyExists` で
//!   失敗し再試行する。他テストや無関係なディレクトリ・symlink を自分の
//!   ものとして扱わず、`Drop` で消すのも自分が作成した root だけ）。
//! - `std::env::temp_dir()` は macOS では `/var/folders/...`（`/var` は
//!   `/private/var` への symlink）を返す。`build_site` の `open_out_root_dir`
//!   は `out` 配下の全コンポーネントを fd 相対 `O_NOFOLLOW` で辿り symlink を
//!   拒否する（`src/build.rs`・PR #899）ため、実在する `temp_dir()` を先に
//!   `canonicalize` してから root を作る（フィクスチャ自身の正規化であり
//!   本番の symlink 拒否を弱めるものではない）。
//! - 出力先 `out` は root 配下の**未作成**の子パス。作成は `build_site`／CLI の
//!   責務であり、失敗時に作成されないこと（linkcheck の fail-closed）も
//!   従来どおり検証できる。
//! - 外部クレート（`tempfile` 等）は追加しない（deps-policy.md）。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// テストが排他作成した一意 root と、その配下の未作成の出力先 `out`。
pub struct TempOutDir {
    root: PathBuf,
    out: PathBuf,
}

impl TempOutDir {
    pub fn new(tag: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let base = std::fs::canonicalize(std::env::temp_dir())
            .expect("canonicalize std::env::temp_dir() for docs-site test");
        let pid = std::process::id();
        for _ in 0..64u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = base.join(format!(
                "rust-ai-library-docs-site-test-{tag}-{pid}-{nanos}-{seq}"
            ));
            match std::fs::create_dir(&root) {
                Ok(()) => {
                    let out = root.join("out");
                    return Self { root, out };
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create temp dir {}: {e}", root.display()),
            }
        }
        panic!("failed to create a unique temp dir within 64 attempts");
    }

    /// `build_site` / `--out` に渡す出力先（root 配下・未作成）。
    pub fn out(&self) -> &Path {
        &self.out
    }
}

impl Drop for TempOutDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
