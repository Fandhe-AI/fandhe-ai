//! ダウンロード URL の検証（イシュー #3125。std のみ・全プラットフォームでビルド）。
//!
//! `ModelRegistry::download` が接続前に呼ぶ fail-closed の入口。`ValidatedUrl` は検証関数
//! からしか作れない newtype で、転送層（`transport.rs`）は `&ValidatedUrl` しか受け取らない
//! ため、未検証の文字列が接続へ渡る経路を型で閉じる（`docs/model-download-design.md` §6）。
//! エラーは静的な理由文字列だけを持ち、入力 URL（クエリ・資格情報を含みうる）は返さない（A09）。

use super::ModelError;

/// URL 全体の長さ上限。
const MAX_URL_LEN: usize = 8 * 1024;

/// 検証済み `https` URL（フラグメント除去済み）。
#[derive(Clone, PartialEq, Eq)]
pub(super) struct ValidatedUrl {
    text: String,
    /// `https://` 直後から authority 終端までの範囲。
    authority_end: usize,
}

impl ValidatedUrl {
    pub(super) fn as_str(&self) -> &str {
        &self.text
    }

    fn authority(&self) -> &str {
        &self.text["https://".len()..self.authority_end]
    }
}

fn invalid(reason: &'static str) -> ModelError {
    ModelError::InvalidUrl { reason }
}

fn all_visible_ascii(s: &str) -> bool {
    s.bytes().all(|b| (0x21..=0x7e).contains(&b)) && !s.contains('\\')
}

/// `https` 絶対 URL を検証する。資格情報付き・非 `https`・制御文字・非 ASCII・`\` を拒否する。
pub(super) fn validate_https_url(input: &str) -> Result<ValidatedUrl, ModelError> {
    if input.len() > MAX_URL_LEN {
        return Err(invalid("URL が長すぎます"));
    }
    if !all_visible_ascii(input) {
        return Err(invalid("URL に使えない文字が含まれています"));
    }
    let Some((scheme, rest)) = input.split_once("://") else {
        return Err(invalid("https:// で始まる絶対 URL が必要です"));
    };
    if !scheme.eq_ignore_ascii_case("https") {
        return Err(invalid("https 以外のスキームは許可されません"));
    }
    // フラグメントは送信しないので除去する。
    let rest = rest.split('#').next().unwrap_or("");
    let auth_len = rest.find(['/', '?']).unwrap_or(rest.len());
    let authority = &rest[..auth_len];
    if authority.contains('@') {
        return Err(invalid("資格情報付き URL は許可されません"));
    }
    let (host, port) = split_host_port(authority)?;
    validate_host(host)?;
    if let Some(p) = port {
        let ok = !p.is_empty()
            && p.len() <= 5
            && p.bytes().all(|b| b.is_ascii_digit())
            && p.parse::<u32>().is_ok_and(|n| (1..=65535).contains(&n));
        if !ok {
            return Err(invalid("ポート番号が不正です"));
        }
    }
    Ok(ValidatedUrl {
        text: format!("https://{rest}"),
        authority_end: "https://".len() + auth_len,
    })
}

fn split_host_port(authority: &str) -> Result<(&str, Option<&str>), ModelError> {
    if let Some(stripped) = authority.strip_prefix('[') {
        let Some(close) = stripped.find(']') else {
            return Err(invalid("IPv6 リテラルが閉じていません"));
        };
        let host = &authority[..close + 2];
        let after = &stripped[close + 1..];
        if after.is_empty() {
            return Ok((host, None));
        }
        return match after.strip_prefix(':') {
            Some(p) => Ok((host, Some(p))),
            None => Err(invalid("ホストの後ろに不正な文字があります")),
        };
    }
    match authority.split_once(':') {
        Some((h, p)) => Ok((h, Some(p))),
        None => Ok((authority, None)),
    }
}

fn validate_host(host: &str) -> Result<(), ModelError> {
    if let Some(inner) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        let ok = !inner.is_empty()
            && inner
                .bytes()
                .all(|b| b.is_ascii_hexdigit() || matches!(b, b':' | b'.'));
        return if ok {
            Ok(())
        } else {
            Err(invalid("IPv6 リテラルが不正です"))
        };
    }
    let ok = !host.is_empty()
        && host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-'))
        && !host.starts_with(['.', '-'])
        && !host.ends_with(['.', '-']);
    if ok {
        Ok(())
    } else {
        Err(invalid("ホスト名が不正です"))
    }
}

/// リダイレクト先 `Location` を解決して再検証する。受理するのは絶対 `https://`・
/// スキーム相対 `//host/…`・絶対パス `/…` のみ（相対参照 `../x`・`x` と `http://` は拒否）。
pub(super) fn resolve_location(
    base: &ValidatedUrl,
    location: &str,
) -> Result<ValidatedUrl, ModelError> {
    if let Some(rest) = location.strip_prefix("//") {
        return validate_https_url(&format!("https://{rest}"));
    }
    if location.starts_with('/') {
        if !all_visible_ascii(location) {
            return Err(invalid("Location に使えない文字が含まれています"));
        }
        return validate_https_url(&format!("https://{}{location}", base.authority()));
    }
    if location.contains("://") {
        return validate_https_url(location);
    }
    Err(invalid("相対参照の Location は許可されません"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> ValidatedUrl {
        validate_https_url(s).unwrap_or_else(|_| panic!("accept: {s}"))
    }

    #[test]
    fn accepts_plain_https_and_strips_fragment() {
        assert_eq!(
            ok("https://example.com/a/b?x=1#frag").as_str(),
            "https://example.com/a/b?x=1"
        );
        assert_eq!(
            ok("HTTPS://Example.com:8443/m").as_str(),
            "https://Example.com:8443/m"
        );
        ok("https://[::1]:443/x");
        ok("https://example.com");
        ok("https://example.com?q=1");
    }

    #[test]
    fn rejects_unsafe_urls() {
        let bad = [
            "http://example.com/",
            "HTTP://example.com/",
            "file:///etc/passwd",
            "ftp://example.com/",
            "example.com/model",
            "https:/example.com",
            "https://user:pass@example.com/",
            "https://user@example.com/",
            "https://@example.com/",
            "https://example.com@evil.com/",
            "https:///path",
            "https://",
            "https://exa mple.com/",
            "https://example.com/\n",
            "https://example.com/a\\b",
            "https://exämple.com/",
            "https://example.com:0/",
            "https://example.com:65536/",
            "https://example.com:abc/",
            "https://example.com:/",
            "https://-bad.com/",
            "https://[::1/",
            "https://[::1]x/",
        ];
        for s in bad {
            assert!(validate_https_url(s).is_err(), "reject: {s:?}");
        }
        let long = format!("https://example.com/{}", "a".repeat(MAX_URL_LEN));
        assert!(validate_https_url(&long).is_err());
    }

    #[test]
    fn resolve_location_accepts_https_forms() {
        let base = ok("https://a.example/x/y?q=1");
        assert_eq!(
            resolve_location(&base, "https://b.example/z")
                .unwrap()
                .as_str(),
            "https://b.example/z"
        );
        assert_eq!(
            resolve_location(&base, "//c.example/z").unwrap().as_str(),
            "https://c.example/z"
        );
        assert_eq!(
            resolve_location(&base, "/abs?x=1").unwrap().as_str(),
            "https://a.example/abs?x=1"
        );
    }

    #[test]
    fn resolve_location_rejects_downgrade_credentials_and_relative() {
        let base = ok("https://a.example/x");
        for loc in [
            "http://b.example/z",
            "https://u:p@b.example/z",
            "//u@b.example/z",
            "../z",
            "z",
            "",
            "ftp://b.example/",
            "/a\\b",
        ] {
            assert!(resolve_location(&base, loc).is_err(), "reject: {loc:?}");
        }
    }
}
