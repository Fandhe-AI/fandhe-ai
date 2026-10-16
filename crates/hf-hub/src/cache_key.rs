//! repo_id と facade キャッシュキーの写像（イシュー #3131・設計 §9.4）。
//!
//! `HfHub::download` が facade の `download_with(name, version)` へ渡す `name` を作る。
//! `name = hf--<namespace>--<repo>`、`version` = 解決済み commit sha。
//! 単射性の根拠: `HfRepoId` の各 segment は `--` を含まず先頭末尾が `-` でないため、
//! `hf--` の後の最初の `--` が namespace と repo の境界として一意に決まる。
//! 先頭が `h` なので facade の `validate_component`（先頭 `.` 不可・`[A-Za-z0-9._-]+`）にも適合する。

use crate::HfRepoId;

const PREFIX: &str = "hf--";

/// repo_id から facade の `name` を作る。
pub(crate) fn encode(repo: &HfRepoId) -> String {
    format!("{PREFIX}{}--{}", repo.namespace(), repo.name())
}

/// 逆写像（単射性の検証用。本番経路では使わない）。
#[cfg(test)]
pub(crate) fn decode(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix(PREFIX)?;
    let (ns, repo) = rest.split_once("--")?;
    Some((ns.to_owned(), repo.to_owned()))
}
