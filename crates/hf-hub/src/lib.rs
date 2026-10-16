//! Hugging Face Hub 連携の補助クレート（イシュー #3130 の雛形。実装本体は #3131）。
//!
//! # 役割と責務境界
//!
//! - 本クレートは HF Hub の `repo_id`・revision の検証と、将来の解決・取得処理を担う
//!   （`docs/hf-hub-integration-design.md` §2.1・§9.2）。
//! - キャッシュへの書き込み・sha256 pin 検証・safetensors 読込は facade の
//!   [`fandhe_ai::model`] に任せ、本クレートは独自にファイル I/O をしない。
//!   ファイル本体の取得は将来 `ModelRegistry::download_with` と呼び出し側の sha256 pin に委ねる。
//! - 依存方向は hf-hub → facade の一方向のみ。`docs/compat-api-scope.md` §0 の (b) に従い、
//!   facade の安定性保証の外にある補助クレートである。
//! - 1 段目は `publish = false`。private／gated repo は初期状態で非対応とする（§4.1）。
//!
//! # 未実装範囲
//!
//! `resolve_revision`・`list_files`・`download` と HTTP／TLS は #3131 で実装する（第 11 区分の
//! HF クレートでの利用承認が前提）。導入時は `TlsProvider::NativeTls` の明示・`https_only(true)`・
//! タイムアウトとサイズ上限・リダイレクト先の https／ドメイン allowlist を必須とする。
//! `download` では facade の既存 `fandhe_ai::model::DownloadOptions` を再利用し、同名の型を作らない。
//!
//! # 検証規則の暫定性
//!
//! [`HfRepoId`]・[`Revision`] の文字集合・長さ上限は外部仕様確認（§9.6）前の暫定値で、
//! fail-closed に厳しめへ倒している（後から緩めるのは非破壊、厳しくするのは破壊的なため）。
//! エラーには入力値を含めない（資格情報付き URL 等の誤入力を漏らさないため）。

use std::fmt;

use fandhe_ai::model::{ModelError, ModelRegistry};

/// repo_id の各 segment の暫定上限バイト数（外部仕様・要確認）。
const MAX_SEGMENT_BYTES: usize = 96;
/// revision（ref 名）の暫定上限バイト数（外部仕様・要確認）。
const MAX_REVISION_BYTES: usize = 255;
/// コミット SHA（小文字 16 進）の長さ。
const COMMIT_SHA_LEN: usize = 40;

/// 本クレートのエラー。入力値そのものは保持しない（reason は固定文字列）。
#[non_exhaustive]
#[derive(Debug)]
pub enum HfError {
    /// repo_id が検証規則に違反した。
    InvalidRepoId {
        /// 違反理由（固定文字列）。
        reason: &'static str,
    },
    /// revision が検証規則に違反した。
    InvalidRevision {
        /// 違反理由（固定文字列）。
        reason: &'static str,
    },
    /// facade の `ModelRegistry` 側のエラー。
    Model(ModelError),
}

impl fmt::Display for HfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HfError::InvalidRepoId { reason } => write!(f, "repo_id が不正です: {reason}"),
            HfError::InvalidRevision { reason } => write!(f, "revision が不正です: {reason}"),
            HfError::Model(e) => write!(f, "モデルレジストリのエラー: {e}"),
        }
    }
}

impl std::error::Error for HfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HfError::Model(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ModelError> for HfError {
    fn from(e: ModelError) -> Self {
        HfError::Model(e)
    }
}

/// 検証済みの HF repo_id（`namespace/name`）。フィールドは非公開で、未検証の値は作れない。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HfRepoId {
    namespace: String,
    name: String,
}

/// repo_id の 1 segment を allowlist で検査する（`HfRepoId::parse` から呼ばれる）。
fn check_repo_segment(seg: &str) -> Result<(), HfError> {
    let bad = |reason| Err(HfError::InvalidRepoId { reason });
    if seg.is_empty() {
        return bad("segment が空");
    }
    if seg.len() > MAX_SEGMENT_BYTES {
        return bad("segment が長すぎる");
    }
    if !seg
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return bad("許可されない文字を含む");
    }
    if seg.starts_with(['.', '-']) || seg.ends_with(['.', '-']) {
        return bad("segment の先頭または末尾が . か -");
    }
    // `--` はキャッシュキー `hf--<ns>--<name>` の区切りと衝突するため拒否する。
    if seg.contains("..") || seg.contains("--") {
        return bad(".. または -- を含む");
    }
    Ok(())
}

impl HfRepoId {
    /// `namespace/name` 形式を検証して構築する。
    pub fn parse(s: &str) -> Result<Self, HfError> {
        let mut parts = s.split('/');
        let (Some(ns), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(HfError::InvalidRepoId {
                reason: "namespace/name 形式ではない",
            });
        };
        check_repo_segment(ns)?;
        check_repo_segment(name)?;
        Ok(Self {
            namespace: ns.to_owned(),
            name: name.to_owned(),
        })
    }

    /// namespace 部。
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// name 部。
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for HfRepoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RevisionKind {
    CommitSha(String),
    Ref(String),
}

/// 検証済みの revision（コミット SHA または ref 名）。内部表現は非公開（不透明型）。
///
/// variant を公開しないのは、公開 enum の String から未検証値を作らせないため。
/// 後から公開するのは非破壊、逆は破壊的変更になる。ref の `/` の URL エンコードは #3131 で決める。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Revision(RevisionKind);

impl Revision {
    /// 既定の ref `main`。
    pub fn main() -> Self {
        Self(RevisionKind::Ref("main".to_owned()))
    }

    /// revision 文字列を検証して構築する。小文字 16 進 40 文字はコミット SHA、他は ref 名。
    pub fn parse(s: &str) -> Result<Self, HfError> {
        let bad = |reason| Err(HfError::InvalidRevision { reason });
        let is_hex40 = s.len() == COMMIT_SHA_LEN && s.bytes().all(|b| b.is_ascii_hexdigit());
        if is_hex40 {
            if s.bytes().any(|b| b.is_ascii_uppercase()) {
                return bad("大文字を含む 40 桁 16 進は曖昧なため拒否");
            }
            return Ok(Self(RevisionKind::CommitSha(s.to_owned())));
        }
        if s.is_empty() {
            return bad("空");
        }
        if s.len() > MAX_REVISION_BYTES {
            return bad("長すぎる");
        }
        if !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'-'))
        {
            return bad("許可されない文字を含む");
        }
        if s.starts_with(['/', '-', '.']) || s.ends_with('/') {
            return bad("先頭または末尾が不正");
        }
        if s.contains("//") || s.contains("..") {
            return bad("// または .. を含む");
        }
        Ok(Self(RevisionKind::Ref(s.to_owned())))
    }

    /// コミット SHA のとき `Some`。
    pub fn commit_sha(&self) -> Option<&str> {
        match &self.0 {
            RevisionKind::CommitSha(s) => Some(s),
            RevisionKind::Ref(_) => None,
        }
    }

    /// ref 名のとき `Some`。
    pub fn ref_name(&self) -> Option<&str> {
        match &self.0 {
            RevisionKind::Ref(s) => Some(s),
            RevisionKind::CommitSha(_) => None,
        }
    }

    /// 文字列表現。
    pub fn as_str(&self) -> &str {
        match &self.0 {
            RevisionKind::CommitSha(s) | RevisionKind::Ref(s) => s,
        }
    }
}

impl Default for Revision {
    fn default() -> Self {
        Self::main()
    }
}

/// HF Hub 連携の入口。キャッシュは facade の [`ModelRegistry`] に委ねる。
///
/// 解決・取得メソッドは #3131 で追加する（未実装スタブは置かない）。
pub struct HfHub {
    registry: ModelRegistry,
}

impl HfHub {
    /// facade の `ModelRegistry` を受け取って構築する。
    pub fn new(registry: ModelRegistry) -> Self {
        Self { registry }
    }

    /// 内部の `ModelRegistry`。
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn repo_id_accepts_valid() {
        let id = HfRepoId::parse("org/model").unwrap();
        assert_eq!((id.namespace(), id.name()), ("org", "model"));
        assert_eq!(id.to_string(), "org/model");
        assert!(HfRepoId::parse("a.b_c-d/x1").is_ok());
    }

    #[test]
    fn repo_id_rejects_invalid() {
        let long = format!("a/{}", "x".repeat(97));
        let cases = [
            "",
            "a",
            "a/b/c",
            "/b",
            "a/",
            ".a/b",
            "-a/b",
            "a/b.",
            "a/b-",
            "a/..",
            "a/b..c",
            "a--b/c",
            "a/b c",
            "a/b\nc",
            "a/b\0c",
            "a/bé",
            "a\\b/c",
            "https://u:p@x/y",
            &long,
        ];
        for c in cases {
            assert!(
                matches!(HfRepoId::parse(c), Err(HfError::InvalidRepoId { .. })),
                "{c:?}"
            );
        }
    }

    #[test]
    fn revision_classification() {
        let sha = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(Revision::parse(sha).unwrap().commit_sha(), Some(sha));
        assert!(Revision::parse(&sha.to_uppercase()).is_err());
        for s in [&sha[..39], &format!("{sha}0"), "main", "v1.0", "refs/pr/1"] {
            let r = Revision::parse(s).unwrap();
            assert_eq!(r.ref_name(), Some(s));
            assert_eq!(r.as_str(), s);
        }
        assert_eq!(Revision::default().as_str(), "main");
    }

    #[test]
    fn revision_rejects_invalid() {
        let long = "a".repeat(256);
        for s in [
            "", "/a", "-a", ".a", "a/", "a//b", "a..b", "a\u{1}b", "a b", &long,
        ] {
            assert!(
                matches!(Revision::parse(s), Err(HfError::InvalidRevision { .. })),
                "{s:?}"
            );
        }
    }

    #[test]
    fn errors_do_not_echo_input() {
        let secret = "https://user:s3cr3tTOKEN@host/x";
        let e1 = HfRepoId::parse(secret).unwrap_err().to_string();
        let e2 = Revision::parse(secret).unwrap_err().to_string();
        assert!(!e1.contains("s3cr3t") && !e2.contains("s3cr3t"));
    }

    #[test]
    fn hub_holds_registry_and_wraps_model_error() {
        let dir = std::env::temp_dir().join("fandhe-ai-hf-hub-test");
        let hub = HfHub::new(ModelRegistry::with_cache_dir(dir.clone()));
        assert_eq!(hub.registry().cache_dir(), dir.as_path());
        let e = HfError::from(ModelError::CacheDirUnavailable);
        assert!(matches!(e, HfError::Model(_)));
        assert!(e.source().is_some());
    }
}
