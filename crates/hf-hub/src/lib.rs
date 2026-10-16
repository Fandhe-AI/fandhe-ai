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
//! # 実装範囲（イシュー #3131）
//!
//! - [`HfHub::download`]: commit sha 固定の `model.safetensors` を facade の `download_with` で
//!   取得する。sha256 pin（[`fandhe_ai::model::Sha256Pin`]）は必須引数で、キャッシュ書き込み・
//!   pin 検証は facade が担う。`name` = `hf--<namespace>--<repo>`・`version` = commit sha。
//! - revision 解決（`resolve_revision`）とファイル一覧（`list_files`）は crate 内に実装済みだが、
//!   HTTP／TLS（第 11 区分 `ureq`）を HF クレートで使う承認が未取得のため転送層を結線しておらず
//!   公開していない（`docs/hf-hub-integration-design.md` §9.12）。
//!
//! # 残留リスク
//!
//! facade の `download_with` はリダイレクトを `https` 限定で追従するがホスト allowlist は持たない
//! （設計 §4.2 (b-1)／(b-2)。facade 公開面の拡張は別承認）。真正性は必須の sha256 pin で担保する。
//!
//! # dtype
//!
//! bf16／f16 のチェックポイントも `download` 自体は成功しうる（facade の公開前検証は dtype 非依存）が、
//! `ModelRegistry::load` は F32 限定契約のため読込時に fail-closed になる（設計 §3.4・
//! `docs/facade-safetensors-exposure-decision.md` §11）。非 unix では facade 側が `Unsupported` を返す。
//!
//! # 検証規則の暫定性
//!
//! [`HfRepoId`]・[`Revision`] の文字集合・長さ上限は外部仕様確認（§9.6）前の暫定値で、
//! fail-closed に厳しめへ倒している（後から緩めるのは非破壊、厳しくするのは破壊的なため）。
//! エラーには入力値を含めない（資格情報付き URL 等の誤入力を漏らさないため）。

use std::fmt;

use fandhe_ai::model::{DownloadOptions, ModelError, ModelRegistry, Sha256Pin};

mod api;
mod cache_key;
mod filename;
mod transport;
mod url;

pub use api::RepoFile;

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
    /// filename が検証規則に違反した（`model.safetensors` 以外）。
    InvalidFilename {
        /// 違反理由（固定文字列）。
        reason: &'static str,
    },
    /// サーバー応答の形状・内容が想定と異なる。
    InvalidResponse {
        /// 違反理由（固定文字列）。
        reason: &'static str,
    },
    /// 応答が上限サイズを超えた。
    ResponseTooLarge {
        /// 上限バイト数。
        max: usize,
    },
    /// repo または revision が見つからない（HTTP 404）。private repo も同じ応答になりうる。
    NotFound,
    /// 認証が必要（HTTP 401）。本クレートは認証を持たず fail-closed で拒否する。
    Unauthorized,
    /// gated／アクセス拒否（HTTP 403）。fail-closed で拒否する。
    Gated,
    /// 上記以外の HTTP ステータス（リダイレクトは追従しない）。
    Http {
        /// ステータスコード。
        status: u16,
    },
    /// facade の `ModelRegistry` 側のエラー。
    Model(ModelError),
}

impl fmt::Display for HfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HfError::InvalidRepoId { reason } => write!(f, "repo_id が不正です: {reason}"),
            HfError::InvalidRevision { reason } => write!(f, "revision が不正です: {reason}"),
            HfError::InvalidFilename { reason } => write!(f, "filename が不正です: {reason}"),
            HfError::InvalidResponse { reason } => write!(f, "サーバー応答が不正です: {reason}"),
            HfError::ResponseTooLarge { max } => write!(f, "応答が上限 {max} バイトを超えました"),
            HfError::NotFound => write!(f, "repo または revision が見つかりません"),
            HfError::Unauthorized => write!(f, "認証が必要です（非対応）"),
            HfError::Gated => write!(f, "アクセスが拒否されました（gated／private は非対応）"),
            HfError::Http { status } => write!(f, "HTTP ステータス {status}"),
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

/// 検証済みの commit sha（小文字 16 進 40 文字）。リビジョン固定済みの値で、キャッシュの
/// `version` と resolve URL に使う。構築は検証経由のみ（サーバー応答の sha も再検証する）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommitSha(String);

fn is_commit_sha(s: &str) -> bool {
    s.len() == COMMIT_SHA_LEN && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl CommitSha {
    /// 小文字 16 進 40 文字のみ受理する（大文字は曖昧なため拒否）。
    pub fn parse(s: &str) -> Result<Self, HfError> {
        if is_commit_sha(s) {
            Ok(Self(s.to_owned()))
        } else {
            Err(HfError::InvalidRevision {
                reason: "commit sha は小文字 16 進 40 文字",
            })
        }
    }

    /// 文字列表現。
    pub fn as_str(&self) -> &str {
        &self.0
    }
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
            if !is_commit_sha(s) {
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

    /// コミット SHA のとき [`CommitSha`] を返す（ネットワーク不要で固定済み）。
    pub fn to_commit_sha(&self) -> Option<CommitSha> {
        self.commit_sha().map(|s| CommitSha(s.to_owned()))
    }

    /// 検証済み [`CommitSha`] から作る（`api` の一覧取得用）。
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_commit_sha(sha: &CommitSha) -> Self {
        Self(RevisionKind::CommitSha(sha.0.clone()))
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
/// 取得は [`HfHub::download`]。revision 解決・一覧は転送層の承認待ちで未公開（クレート doc 参照）。
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

    /// commit sha 固定の `model.safetensors` を取得し facade のキャッシュへ配置する。
    ///
    /// - `filename` は `model.safetensors` のみ（是正案 A）。不正な入力は接続前に `Err`。
    /// - `pin` は必須。`options` に別の pin があっても本引数の pin が優先される。
    /// - 保存先は `<cache_dir>/<name>/<version>/model.safetensors`（[`CachedModel`] 参照）。
    ///   本クレートは自らファイルを書かず、書き込み・検証はすべて facade が行う。
    pub fn download(
        &self,
        repo: &HfRepoId,
        rev: &CommitSha,
        filename: &str,
        pin: Sha256Pin,
        options: DownloadOptions<'_>,
    ) -> Result<CachedModel, HfError> {
        filename::validate(filename)?;
        let name = cache_key::encode(repo);
        let version = rev.as_str().to_owned();
        let url = url::resolve_url(repo, rev, filename);
        self.registry
            .download_with(&url, &name, &version, options.expected_sha256(pin))?;
        Ok(CachedModel { name, version })
    }
}

/// 取得済みモデルの識別子。`ModelRegistry::load(name(), version())` で読める。
///
/// 実ファイルは `<cache_dir>/<name>/<version>/model.safetensors`。同一 repo_id・commit sha は
/// 常に同じ位置を指し、異なる repo_id・sha は異なる位置になる（filename は固定のため衝突しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModel {
    name: String,
    version: String,
}

impl CachedModel {
    /// facade のキャッシュ上の `name`（`hf--<namespace>--<repo>`）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// facade のキャッシュ上の `version`（commit sha）。
    pub fn version(&self) -> &str {
        &self.version
    }

    /// ファイル名（常に `model.safetensors`）。
    pub fn filename(&self) -> &'static str {
        filename::MODEL_FILENAME
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

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn scratch_hub(tag: &str) -> (HfHub, std::path::PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("fandhe-ai-hf-hub-{tag}-{}", std::process::id()));
        (HfHub::new(ModelRegistry::with_cache_dir(dir.clone())), dir)
    }

    fn pin() -> Sha256Pin {
        Sha256Pin::from_hex(&"0".repeat(64)).unwrap()
    }

    #[test]
    fn commit_sha_validation() {
        assert!(CommitSha::parse(SHA).is_ok());
        for bad in [
            "",
            "main",
            &SHA[..39],
            &SHA.to_uppercase(),
            &format!("{SHA}0"),
        ] {
            assert!(CommitSha::parse(bad).is_err(), "{bad}");
        }
        let rev = Revision::parse(SHA).unwrap();
        assert_eq!(rev.to_commit_sha().unwrap().as_str(), SHA);
        assert!(Revision::main().to_commit_sha().is_none());
    }

    #[test]
    fn download_rejects_bad_filename_before_connecting() {
        let (hub, dir) = scratch_hub("fname");
        let repo = HfRepoId::parse("org/model").unwrap();
        let sha = CommitSha::parse(SHA).unwrap();
        for f in [
            "pytorch_model.bin",
            "model-00001-of-00002.safetensors",
            "sub/model.safetensors",
            "../model.safetensors",
            "/model.safetensors",
            "Model.safetensors",
            "model.safetensors\0",
            "model.safetensors\n",
            "",
        ] {
            let r = hub.download(&repo, &sha, f, pin(), DownloadOptions::default());
            assert!(matches!(r, Err(HfError::InvalidFilename { .. })), "{f:?}");
        }
        assert!(
            !dir.exists(),
            "接続前の拒否でキャッシュに何も作られないこと"
        );
    }

    #[test]
    fn cache_key_is_injective_and_accepted_by_facade() {
        let ids = [
            "a/b",
            "a-b/c",
            "a/b-c",
            "a.b/c_d",
            "org/model",
            "org/model.v2",
        ];
        let mut seen = std::collections::HashSet::new();
        let dir = std::env::temp_dir().join(format!("fandhe-ai-hf-hub-key-{}", std::process::id()));
        let reg = ModelRegistry::with_cache_dir(dir);
        for id in ids {
            let repo = HfRepoId::parse(id).unwrap();
            let name = cache_key::encode(&repo);
            assert!(seen.insert(name.clone()), "{id}");
            assert_eq!(
                cache_key::decode(&name),
                Some((repo.namespace().to_owned(), repo.name().to_owned()))
            );
            // facade の validate_component に受理される（InvalidComponent ではなく NotFound）。
            let e = reg.load(&name, SHA).map(|_| ()).unwrap_err();
            assert!(matches!(e, ModelError::NotFound { .. }), "{id}: {e}");
        }
    }

    #[test]
    fn resolve_url_shape_is_accepted_form() {
        let repo = HfRepoId::parse("org/model").unwrap();
        let sha = CommitSha::parse(SHA).unwrap();
        let u = url::resolve_url(&repo, &sha, filename::MODEL_FILENAME);
        assert_eq!(
            u,
            format!("https://huggingface.co/org/model/resolve/{SHA}/model.safetensors")
        );
        let visible = u.bytes().all(|b| (0x21..=0x7e).contains(&b));
        assert!(u.len() < 8 * 1024 && visible && !u.contains('@'));
    }

    #[test]
    fn new_errors_do_not_echo_input() {
        let secret = "s3cr3tTOKEN";
        let (hub, _d) = scratch_hub("echo");
        let repo = HfRepoId::parse("org/model").unwrap();
        let sha = CommitSha::parse(SHA).unwrap();
        let e = hub
            .download(&repo, &sha, secret, pin(), DownloadOptions::default())
            .unwrap_err()
            .to_string();
        assert!(!e.contains(secret));
        assert!(
            !CommitSha::parse(secret)
                .unwrap_err()
                .to_string()
                .contains(secret)
        );
    }
}
