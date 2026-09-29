//! 結合テスト共通の一時ファイルヘルパ（イシュー #2382、親 #2363）。
//!
//! 使うバイナリだけが `mod common;` と `#[path = "common/temp_file.rs"] mod temp_file;` で読み込む
//! （`#[path]` 読み込みのため crate root 直下に置かれ、`crate::common::temp_dir` を参照する。
//! 未使用バイナリでの dead_code を避けるため `common/mod.rs` では宣言しない）。

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::common::temp_dir::unique_temp_candidate;

/// 一時ファイルを一意名で `create_new(true)` により**排他作成**し、内容を書き込んで返す RAII ガード。
/// Drop で自身のパスだけを削除する（panic しても残さない）。
///
/// - 作成前の `remove_file` は行わず、既存パス（symlink 含む）には `AlreadyExists` で失敗して再試行する。
/// - 作成と書き込みを 1 回で済ませ、後から `fs::write` で上書きする経路を作らない。
/// - **注意**: 名前付き変数へ束縛して保持すること（一時値化は即 drop されて削除される）。
pub struct TempFileGuard {
    path: PathBuf,
}

impl TempFileGuard {
    /// `label` を名前末尾に含む一時ファイルを排他作成し `contents` を書き込む。失敗時は panic（テスト専用）。
    pub fn new(label: &str, contents: &[u8]) -> Self {
        for _ in 0..64 {
            let candidate = unique_temp_candidate(label);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(mut file) => {
                    // 書き込み失敗時も作成済みファイルを残さないよう、先にガードを確保する。
                    let guard = Self { path: candidate };
                    if let Err(e) = file.write_all(contents) {
                        panic!("一時ファイルへの書き込みに失敗: {:?}: {e}", guard.path);
                    }
                    return guard;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("一時ファイルの作成に失敗: {candidate:?}: {e}"),
            }
        }
        panic!("一時ファイルの一意名を確保できなかった（64 回衝突）: label={label}");
    }

    /// 作成済みファイルのパス。
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
