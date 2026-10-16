//! HTTP 転送層（イシュー #3125。`cfg(unix)`。ureq の型はこのファイルの外へ出さない）。
//!
//! `download.rs` が内部 trait [`Transport`] 越しに呼ぶ。本番実装 `UreqTransport` は TLS を
//! `NativeTls`＋OS ルート証明書に固定し、`https_only`・手動リダイレクト（`max_redirects(0)`）・
//! 全タイムアウト明示・環境変数 proxy 無効を `Agent` に焼き込む（A02／A05。
//! `docs/model-download-design.md` §6・§15〜17）。`disable_verification` はどこにも公開しない。
//! ureq のエラー文字列は URL を含みうる（`RequireHttpsOnly(String)` 等）ため `Display` を
//! 通さず、静的メッセージの `io::Error`／`InvalidUrl` へ変換する（A09）。

use std::io::{self, Read};
use std::time::{Duration, Instant};

use ureq::tls::{RootCerts, TlsConfig, TlsProvider};

use super::ModelError;
use super::url::ValidatedUrl;

/// 名前解決・接続（TLS ハンドシェイク含む）・リクエスト送信の各上限。
const TIMEOUT_PHASE: Duration = Duration::from_secs(30);
/// 応答ヘッダ受信の上限。
const TIMEOUT_RECV_RESPONSE: Duration = Duration::from_secs(60);
/// 全ホップ合算の期限（`download.rs` が呼び出し開始時に決める）。
pub(super) const OVERALL_DEADLINE: Duration = Duration::from_secs(3600);

/// 転送層の応答（ヘッダの必要部分とボディのストリーム）。
pub(super) struct RawResponse {
    pub(super) status: u16,
    pub(super) location: Option<String>,
    pub(super) content_length: Option<u64>,
    pub(super) content_encoding: Option<String>,
    pub(super) body: Box<dyn Read>,
}

/// GET を 1 ホップ分実行する抽象。`download.rs` の単体テストはこれをプロセス内モックで差し替える。
/// 受け取るのは検証済み URL のみ（未検証文字列が転送層へ渡る経路を型で閉じる）。
pub(super) trait Transport {
    fn get(&mut self, url: &ValidatedUrl, deadline: Instant) -> Result<RawResponse, ModelError>;
}

fn net(kind: io::ErrorKind, msg: &'static str) -> ModelError {
    ModelError::Network(io::Error::new(kind, msg))
}

/// ureq のエラーを URL を含まない静的メッセージへ変換する。
fn map_ureq_error(e: ureq::Error) -> ModelError {
    match e {
        ureq::Error::RequireHttpsOnly(_) => ModelError::InvalidUrl {
            reason: "https 以外への接続は許可されません",
        },
        ureq::Error::Timeout(_) => net(io::ErrorKind::TimedOut, "HTTP 転送がタイムアウトしました"),
        ureq::Error::HostNotFound => net(io::ErrorKind::NotFound, "ホストを解決できません"),
        ureq::Error::ConnectionFailed => {
            net(io::ErrorKind::ConnectionRefused, "接続に失敗しました")
        }
        ureq::Error::Io(io_err) => ModelError::Network(io::Error::new(
            io_err.kind(),
            "HTTP 転送の I/O に失敗しました",
        )),
        _ => net(io::ErrorKind::Other, "HTTP 転送に失敗しました"),
    }
}

/// ボディ読み取り中のエラーも URL を含みうるため、種別だけを残して静的メッセージへ置き換える。
struct SanitizedReader<R: Read>(R);

impl<R: Read> Read for SanitizedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0
            .read(buf)
            .map_err(|e| io::Error::new(e.kind(), "HTTP ボディの受信に失敗しました"))
    }
}

/// 固定構成の `Agent` を作り、不変条件を読み戻して照合する。ureq は provider と feature の
/// 不整合を構築時でなくリクエスト時に panic で報告しうる（設計 §15.4）ため、ここでの照合は
/// 構成の取り違え検出であり、feature 整合は単体テストが実リクエストで固定する。
pub(super) fn build_agent() -> Result<ureq::Agent, ModelError> {
    let tls = TlsConfig::builder()
        .provider(TlsProvider::NativeTls)
        .root_certs(RootCerts::PlatformVerifier)
        .build();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .max_redirects(0)
        .http_status_as_error(false)
        .proxy(None)
        .user_agent(concat!("fandhe-ai/", env!("CARGO_PKG_VERSION")))
        .accept_encoding("identity")
        .tls_config(tls)
        .timeout_resolve(Some(TIMEOUT_PHASE))
        .timeout_connect(Some(TIMEOUT_PHASE))
        .timeout_send_request(Some(TIMEOUT_PHASE))
        .timeout_recv_response(Some(TIMEOUT_RECV_RESPONSE))
        .timeout_recv_body(Some(OVERALL_DEADLINE))
        .timeout_global(Some(OVERALL_DEADLINE))
        .build()
        .into();
    check_invariants(&agent)?;
    Ok(agent)
}

fn check_invariants(agent: &ureq::Agent) -> Result<(), ModelError> {
    let c = agent.config();
    let t = c.timeouts();
    let ok = c.tls_config().provider() == TlsProvider::NativeTls
        && !c.tls_config().disable_verification()
        && c.https_only()
        && c.max_redirects() == 0
        && c.proxy().is_none()
        && t.global.is_some()
        && t.resolve.is_some()
        && t.connect.is_some()
        && t.send_request.is_some()
        && t.recv_response.is_some()
        && t.recv_body.is_some();
    if ok {
        Ok(())
    } else {
        Err(net(io::ErrorKind::InvalidInput, "TLS／HTTP 構成が不正です"))
    }
}

/// ureq による本番の転送実装。
pub(super) struct UreqTransport {
    agent: ureq::Agent,
}

impl UreqTransport {
    pub(super) fn new() -> Result<Self, ModelError> {
        Ok(Self {
            agent: build_agent()?,
        })
    }
}

/// `Content-Encoding` の全ヘッダ行を `,` 連結で 1 本の文字列にする。`HeaderMap::get` は最初の
/// 値しか返さず `to_str` 失敗を不存在扱いにすると、別行の `gzip` を見落として圧縮ボディを
/// 受理しうるため、`get_all` で全値を集め、非 ASCII 等の変換失敗は明示エラーにする（設計 §19.2）。
/// 値の許否判定（`identity` 以外を拒否）は `download.rs` が行う。
fn collect_content_encoding(headers: &ureq::http::HeaderMap) -> Result<Option<String>, ModelError> {
    let mut parts: Vec<&str> = Vec::new();
    for v in headers.get_all("content-encoding") {
        parts.push(v.to_str().map_err(|_| {
            net(
                io::ErrorKind::InvalidData,
                "Content-Encoding ヘッダを解釈できません",
            )
        })?);
    }
    Ok(if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    })
}

impl Transport for UreqTransport {
    fn get(&mut self, url: &ValidatedUrl, deadline: Instant) -> Result<RawResponse, ModelError> {
        // 多層防御: 検証済み URL を `Uri` として再解析し、https・資格情報なしを照合する。
        let uri = ureq::http::Uri::try_from(url.as_str()).map_err(|_| ModelError::InvalidUrl {
            reason: "URL を解析できません",
        })?;
        let authority_ok = uri.authority().is_some_and(|a| !a.as_str().contains('@'));
        if uri.scheme_str() != Some("https") || !authority_ok {
            return Err(ModelError::InvalidUrl {
                reason: "https 以外または資格情報付きの URL です",
            });
        }
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| net(io::ErrorKind::TimedOut, "HTTP 転送がタイムアウトしました"))?;
        let resp = self
            .agent
            .get(uri)
            .config()
            .timeout_global(Some(remaining))
            .build()
            .call()
            .map_err(map_ureq_error)?;
        let status = resp.status().as_u16();
        let header = |name: &str| -> Option<String> {
            resp.headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        let location = header("location");
        let content_encoding = collect_content_encoding(resp.headers())?;
        let (_, body) = resp.into_parts();
        let content_length = body.content_length();
        let reader = body.into_with_config().limit(u64::MAX).reader();
        Ok(RawResponse {
            status,
            location,
            content_length,
            content_encoding,
            body: Box::new(SanitizedReader(reader)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn content_encoding_collects_all_header_lines() {
        let mut h = ureq::http::HeaderMap::new();
        assert!(matches!(collect_content_encoding(&h), Ok(None)));
        h.append("content-encoding", "identity".parse().unwrap());
        h.append("content-encoding", "gzip".parse().unwrap());
        let joined = collect_content_encoding(&h).unwrap().unwrap();
        assert_eq!(joined, "identity,gzip");
    }

    #[test]
    fn content_encoding_non_ascii_value_is_explicit_error() {
        let mut h = ureq::http::HeaderMap::new();
        h.append(
            "content-encoding",
            ureq::http::HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
        );
        assert!(matches!(
            collect_content_encoding(&h),
            Err(ModelError::Network(_))
        ));
    }

    #[test]
    fn build_agent_satisfies_invariants() {
        let agent = build_agent().expect("agent");
        assert!(check_invariants(&agent).is_ok());
    }

    #[test]
    fn https_connection_failure_is_err_not_panic() {
        // 閉じたポートへの https 要求が panic せず `Err(Network)` になる
        // （NativeTls の feature 整合を外部通信なしで固定する）。
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").expect("bind");
            l.local_addr().expect("addr").port()
        };
        let url = super::super::url::validate_https_url(&format!("https://127.0.0.1:{port}/m"))
            .expect("url");
        let mut t = UreqTransport::new().expect("transport");
        let r = t.get(&url, Instant::now() + Duration::from_secs(20));
        assert!(
            matches!(r, Err(ModelError::Network(_))),
            "got Err? {}",
            r.is_err()
        );
    }

    #[test]
    fn tls_handshake_failure_on_plain_listener_is_err_not_panic() {
        // TLS を話さないサーバ（accept して即 close）へ要求しても panic しない。
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = l.local_addr().expect("addr").port();
        let h = std::thread::spawn(move || {
            let _ = l.accept();
        });
        let url = super::super::url::validate_https_url(&format!("https://127.0.0.1:{port}/m"))
            .expect("url");
        let mut t = UreqTransport::new().expect("transport");
        let r = t.get(&url, Instant::now() + Duration::from_secs(20));
        assert!(r.is_err());
        let _ = h.join();
    }

    #[test]
    fn expired_deadline_is_timed_out_without_connecting() {
        let url = super::super::url::validate_https_url("https://127.0.0.1:1/m").expect("url");
        let mut t = UreqTransport::new().expect("transport");
        let r = t.get(&url, Instant::now() - Duration::from_secs(1));
        match r {
            Err(ModelError::Network(e)) => assert_eq!(e.kind(), io::ErrorKind::TimedOut),
            _ => panic!("expected TimedOut"),
        }
    }

    #[test]
    fn ureq_https_only_rejects_http_scheme_without_connecting() {
        let agent = build_agent().expect("agent");
        let e = agent
            .get("http://127.0.0.1:1/")
            .call()
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(map_ureq_error(e), ModelError::InvalidUrl { .. }));
    }
}
