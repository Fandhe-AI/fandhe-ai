//! HF の URL 組み立て（純関数。イシュー #3131・設計 §3.5・§9.6）。
//!
//! 接続先は `https://huggingface.co` 固定でエンドポイント上書き口を持たない（A05・A10）。
//! `ApiUrl` はこのモジュールの関数からしか作れない newtype で、転送層へ未検証文字列が渡る
//! 経路を型で閉じる。URL エンコードは依存を増やさず自作する（設計 §9.5）。

use crate::{CommitSha, HfRepoId, Revision};

const BASE: &str = "https://huggingface.co";

/// 検証済みコンポーネントだけから組み立てた HF API URL。
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ApiUrl(String);

impl ApiUrl {
    /// 転送層が接続に使う文字列。本番の転送層 `UreqJsonTransport`（#3393）から使われるが、
    /// 呼び出し元の公開化は #3394 のため非 test ビルドでは未使用になりうる。
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    /// host 検査の拒否テスト用に任意文字列から作る。本番ビルドには存在しない。
    #[cfg(test)]
    pub(crate) fn for_test(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl std::fmt::Debug for ApiUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiUrl(..)")
    }
}

/// RFC 3986 の unreserved 以外を `%XX` へ変換する（path 要素 1 個分）。
pub(crate) fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// ファイル取得 URL: `https://huggingface.co/<ns>/<repo>/resolve/<sha>/<filename>`。
/// `filename` は呼び出し側で検証済みの定数のみ（`filename::validate` 通過後）。
pub(crate) fn resolve_url(repo: &HfRepoId, sha: &CommitSha, filename: &str) -> String {
    format!(
        "{BASE}/{}/{}/resolve/{}/{}",
        repo.namespace(),
        repo.name(),
        sha.as_str(),
        filename
    )
}

/// revision 情報 API URL。ref 名は `/` を含みうるため path 要素としてエンコードする。
/// エンドポイント形式は一次資料で未確認（設計 §9.12）。
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn revision_api_url(repo: &HfRepoId, rev: &Revision) -> ApiUrl {
    ApiUrl(format!(
        "{BASE}/api/models/{}/{}/revision/{}",
        repo.namespace(),
        repo.name(),
        percent_encode(rev.as_str())
    ))
}
