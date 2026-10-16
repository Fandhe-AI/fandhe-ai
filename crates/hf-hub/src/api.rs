//! HF JSON API の解析ロジック（イシュー #3131・A03）。
//!
//! `resolve_revision_via`（branch／tag → commit sha）と `list_files_via` を [`JsonTransport`]
//! 越しに実装する。応答は解析前にサイズを検査し、型付き構造体で解析し、sha は 40 桁小文字
//! 16 進で再検証する。etag／LFS oid は信頼根にしない（真正性は呼び出し側の sha256 pin）。
//! 本番の転送層が未結線のため crate 内限定（`pub(crate)`）。公開化は非破壊で後から行える。
//! 3xx は追従せず `HfError::Http` として拒否する（リダイレクト先ホストの検証経路を持たないため）。

use serde::Deserialize;

use crate::transport::JsonTransport;
use crate::url::revision_api_url;
use crate::{CommitSha, HfError, HfRepoId, Revision};

/// JSON 応答のサイズ上限（1 MiB。revision 情報は通常数 KiB〜数十 KiB のため十分に余裕を見た値）。
pub(crate) const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
/// ファイル一覧の件数上限。
const MAX_FILES: usize = 10_000;
/// 各パスの長さ上限。
const MAX_PATH_BYTES: usize = 1024;

#[derive(Deserialize)]
struct RevisionInfo {
    sha: String,
    /// 欠落は「空一覧の成功」と区別するため `Option`（一覧取得時のみ必須）。
    siblings: Option<Vec<Sibling>>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

/// リポジトリ内のファイル 1 件。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    path: String,
}

impl RepoFile {
    /// リポジトリ内パス。
    pub fn path(&self) -> &str {
        &self.path
    }
}

fn fetch_info(
    t: &mut dyn JsonTransport,
    repo: &HfRepoId,
    rev: &Revision,
) -> Result<RevisionInfo, HfError> {
    let url = revision_api_url(repo, rev);
    let resp = t.get_json(&url, MAX_RESPONSE_BYTES)?;
    match resp.status {
        200 => {}
        401 => return Err(HfError::Unauthorized),
        403 => return Err(HfError::Gated),
        404 => return Err(HfError::NotFound),
        s => return Err(HfError::Http { status: s }),
    }
    if resp.body.len() > MAX_RESPONSE_BYTES {
        return Err(HfError::ResponseTooLarge {
            max: MAX_RESPONSE_BYTES,
        });
    }
    serde_json::from_slice(&resp.body).map_err(|_| HfError::InvalidResponse {
        reason: "JSON の形状が想定と異なる",
    })
}

/// revision を commit sha へ固定する。既に sha ならネットワークを使わない。
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_revision_via(
    t: &mut dyn JsonTransport,
    repo: &HfRepoId,
    rev: &Revision,
) -> Result<CommitSha, HfError> {
    if let Some(sha) = rev.to_commit_sha() {
        return Ok(sha);
    }
    let info = fetch_info(t, repo, rev)?;
    CommitSha::parse(&info.sha).map_err(|_| HfError::InvalidResponse {
        reason: "sha が 40 桁小文字 16 進ではない",
    })
}

/// 指定 commit のファイル一覧を返す。
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn list_files_via(
    t: &mut dyn JsonTransport,
    repo: &HfRepoId,
    sha: &CommitSha,
) -> Result<Vec<RepoFile>, HfError> {
    let rev = Revision::from_commit_sha(sha);
    let info = fetch_info(t, repo, &rev)?;
    // 応答の sha は要求した commit と一致しなければならない（別 commit の一覧を混入させない）。
    let got = CommitSha::parse(&info.sha).map_err(|_| HfError::InvalidResponse {
        reason: "sha が 40 桁小文字 16 進ではない",
    })?;
    if got.as_str() != sha.as_str() {
        return Err(HfError::InvalidResponse {
            reason: "応答の sha が要求した commit と一致しない",
        });
    }
    let siblings = info.siblings.ok_or(HfError::InvalidResponse {
        reason: "siblings が欠落している",
    })?;
    if siblings.len() > MAX_FILES {
        return Err(HfError::InvalidResponse {
            reason: "ファイル数が上限を超える",
        });
    }
    let mut out = Vec::with_capacity(siblings.len());
    for s in siblings {
        let p = s.rfilename;
        if p.is_empty() || p.len() > MAX_PATH_BYTES || p.chars().any(char::is_control) {
            return Err(HfError::InvalidResponse {
                reason: "パスが不正",
            });
        }
        out.push(RepoFile { path: p });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::JsonResponse;
    use crate::url::ApiUrl;

    struct Mock {
        status: u16,
        body: Vec<u8>,
        calls: usize,
        last_url: String,
    }
    impl Mock {
        fn new(status: u16, body: &str) -> Self {
            Self {
                status,
                body: body.as_bytes().to_vec(),
                calls: 0,
                last_url: String::new(),
            }
        }
    }
    impl JsonTransport for Mock {
        fn get_json(&mut self, url: &ApiUrl, _max: usize) -> Result<JsonResponse, HfError> {
            self.calls += 1;
            self.last_url = url.as_str().to_owned();
            Ok(JsonResponse {
                status: self.status,
                body: self.body.clone(),
            })
        }
    }

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
    fn repo() -> HfRepoId {
        HfRepoId::parse("org/model").unwrap()
    }

    #[test]
    fn resolves_ref_to_sha() {
        let mut m = Mock::new(200, &format!(r#"{{"sha":"{SHA}","extra":1}}"#));
        let sha = resolve_revision_via(&mut m, &repo(), &Revision::main()).unwrap();
        assert_eq!(sha.as_str(), SHA);
        assert_eq!(
            m.last_url,
            "https://huggingface.co/api/models/org/model/revision/main"
        );
    }

    #[test]
    fn ref_with_slash_is_percent_encoded() {
        let mut m = Mock::new(200, &format!(r#"{{"sha":"{SHA}"}}"#));
        let rev = Revision::parse("refs/pr/1").unwrap();
        resolve_revision_via(&mut m, &repo(), &rev).unwrap();
        assert!(m.last_url.ends_with("/revision/refs%2Fpr%2F1"));
    }

    #[test]
    fn sha_revision_skips_network() {
        let mut m = Mock::new(500, "");
        let rev = Revision::parse(SHA).unwrap();
        let got = resolve_revision_via(&mut m, &repo(), &rev).unwrap();
        assert_eq!(got.as_str(), SHA);
        assert_eq!(m.calls, 0);
    }

    #[test]
    fn rejects_bad_sha_and_shapes() {
        let upper = SHA.to_uppercase();
        for body in [
            r#"{"sha":"abc"}"#.to_owned(),
            format!(r#"{{"sha":"{upper}"}}"#),
            r#"{}"#.to_owned(),
            r#"{"sha":1}"#.to_owned(),
            "not json".to_owned(),
        ] {
            let mut m = Mock::new(200, &body);
            let r = resolve_revision_via(&mut m, &repo(), &Revision::main());
            assert!(matches!(r, Err(HfError::InvalidResponse { .. })), "{body}");
        }
    }

    #[test]
    fn rejects_oversized_body() {
        let mut m = Mock::new(200, "");
        m.body = vec![b' '; MAX_RESPONSE_BYTES + 1];
        let r = resolve_revision_via(&mut m, &repo(), &Revision::main());
        assert!(matches!(r, Err(HfError::ResponseTooLarge { .. })));
    }

    #[test]
    fn maps_status_codes() {
        let cases = [
            (401, "Unauthorized"),
            (403, "Gated"),
            (404, "NotFound"),
            (302, "Http { status: 302 }"),
            (503, "Http { status: 503 }"),
        ];
        for (status, expect) in cases {
            let mut m = Mock::new(status, "{}");
            let e = resolve_revision_via(&mut m, &repo(), &Revision::main()).unwrap_err();
            assert_eq!(format!("{e:?}"), expect, "{status}");
        }
    }

    #[test]
    fn lists_files_and_validates_paths() {
        let sha = CommitSha::parse(SHA).unwrap();
        let body = format!(
            r#"{{"sha":"{SHA}","siblings":[{{"rfilename":"model.safetensors"}},{{"rfilename":"config.json"}}]}}"#
        );
        let mut m = Mock::new(200, &body);
        let files = list_files_via(&mut m, &repo(), &sha).unwrap();
        let paths: Vec<_> = files.iter().map(RepoFile::path).collect();
        assert_eq!(paths, ["model.safetensors", "config.json"]);

        for bad in [r#""a\u0000b""#, r#""""#] {
            let body = format!(r#"{{"sha":"{SHA}","siblings":[{{"rfilename":{bad}}}]}}"#);
            let mut m = Mock::new(200, &body);
            assert!(list_files_via(&mut m, &repo(), &sha).is_err(), "{bad}");
        }
        // siblings 欠落・sha 不一致・sha 不正は fail-closed。
        let other = "f".repeat(40);
        for body in [
            format!(r#"{{"sha":"{SHA}"}}"#),
            format!(r#"{{"sha":"{other}","siblings":[]}}"#),
            r#"{"sha":"abc","siblings":[]}"#.to_owned(),
            r#"{"siblings":[]}"#.to_owned(),
        ] {
            let mut m = Mock::new(200, &body);
            let r = list_files_via(&mut m, &repo(), &sha);
            assert!(matches!(r, Err(HfError::InvalidResponse { .. })), "{body}");
        }
        let many = vec![r#"{"rfilename":"a"}"#; MAX_FILES + 1].join(",");
        let mut m = Mock::new(200, &format!(r#"{{"sha":"{SHA}","siblings":[{many}]}}"#));
        assert!(m.body.len() < MAX_RESPONSE_BYTES);
        assert!(list_files_via(&mut m, &repo(), &sha).is_err());
    }
}
