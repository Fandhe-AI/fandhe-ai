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
    /// `ETag` ヘッダ。解釈できない・`valid_etag` を満たさない値は `None`（保存しない。#3126）。
    pub(super) etag: Option<String>,
    pub(super) body: Box<dyn Read>,
}

/// GET を 1 ホップ分実行する抽象。`download.rs` の単体テストはこれをプロセス内モックで差し替える。
/// 受け取るのは検証済み URL のみ（未検証文字列が転送層へ渡る経路を型で閉じる）。
pub(super) trait Transport {
    /// `if_none_match` が `Some` のときだけ `If-None-Match` を付ける（#3126。呼び出し側が
    /// `cache::valid_etag` で検証済みの値のみ渡す）。
    fn get(
        &mut self,
        url: &ValidatedUrl,
        deadline: Instant,
        if_none_match: Option<&str>,
    ) -> Result<RawResponse, ModelError>;
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

/// 本番専用の前段。検証済み URL を `Uri` として再解析し、https・資格情報なしを照合する
/// （多層防御）。テスト専用の平文ループバック転送も同じ照合を通してから `perform` へ進む。
fn checked_https_uri(url: &ValidatedUrl) -> Result<ureq::http::Uri, ModelError> {
    let uri = ureq::http::Uri::try_from(url.as_str()).map_err(|_| ModelError::InvalidUrl {
        reason: "URL を解析できません",
    })?;
    let authority_ok = uri.authority().is_some_and(|a| !a.as_str().contains('@'));
    if uri.scheme_str() != Some("https") || !authority_ok {
        return Err(ModelError::InvalidUrl {
            reason: "https 以外または資格情報付きの URL です",
        });
    }
    Ok(uri)
}

/// 共有の後段。agent で 1 ホップ GET し、応答を `RawResponse` へ変換する。残り時間の算出・
/// `If-None-Match` 付与・`timeout_global(remaining)`・ヘッダ解釈・ボディのサニタイズを受け持つ。
/// 本番 `UreqTransport` とテスト専用の `loopback::PlaintextLoopbackTransport`（#3127）が共用し、
/// スキーム・TLS の判断は呼び出し側（前段）に残す。
fn perform(
    agent: &ureq::Agent,
    uri: ureq::http::Uri,
    deadline: Instant,
    if_none_match: Option<&str>,
) -> Result<RawResponse, ModelError> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| net(io::ErrorKind::TimedOut, "HTTP 転送がタイムアウトしました"))?;
    let mut req = agent.get(uri);
    // 多層防御: 送信直前にも etag を再確認する（ヘッダインジェクション対策）。違反値は
    // 条件なし GET に倒す（呼び出し側 `cache.rs` で検証済みのため通常は到達しない）。
    if let Some(v) = if_none_match.filter(|v| super::cache::valid_etag(v)) {
        req = req.header("If-None-Match", v);
    }
    let resp = req
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
    let etag = header("etag").filter(|v| super::cache::valid_etag(v));
    let (_, body) = resp.into_parts();
    let content_length = body.content_length();
    let reader = body.into_with_config().limit(u64::MAX).reader();
    Ok(RawResponse {
        status,
        location,
        content_length,
        content_encoding,
        etag,
        body: Box::new(SanitizedReader(reader)),
    })
}

impl Transport for UreqTransport {
    fn get(
        &mut self,
        url: &ValidatedUrl,
        deadline: Instant,
        if_none_match: Option<&str>,
    ) -> Result<RawResponse, ModelError> {
        let uri = checked_https_uri(url)?;
        perform(&self.agent, uri, deadline, if_none_match)
    }
}

/// テスト専用の平文ループバック転送（#3127。`docs/model-download-design.md` §6「テスト経路の
/// 分離」）。`download_via` へ差し込み、ローカルの平文 HTTP モックへ実ソケットで接続する。
/// 本番 `get` と同じ前段（https・資格情報の再照合）を通し、ホストがループバックの場合に限り
/// スキームだけ `http://` へ置き換えて共有後段 `perform` を呼ぶ。`https_only(false)` はこの
/// モジュールの agent にだけ存在し、本番 `build_agent` には現れない（api_surface が固定）。
#[cfg(test)]
pub(super) mod loopback {
    use super::*;

    pub(in crate::model) struct PlaintextLoopbackTransport {
        agent: ureq::Agent,
    }

    impl PlaintextLoopbackTransport {
        pub(in crate::model) fn new() -> Self {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .https_only(false)
                .max_redirects(0)
                .http_status_as_error(false)
                .proxy(None)
                .accept_encoding("identity")
                .timeout_connect(Some(Duration::from_secs(2)))
                .timeout_recv_response(Some(Duration::from_secs(2)))
                .build()
                .into();
            Self { agent }
        }
    }

    impl Transport for PlaintextLoopbackTransport {
        fn get(
            &mut self,
            url: &ValidatedUrl,
            deadline: Instant,
            if_none_match: Option<&str>,
        ) -> Result<RawResponse, ModelError> {
            let uri = checked_https_uri(url)?;
            let host = uri.host().unwrap_or("");
            if host != "127.0.0.1" && host != "[::1]" {
                return Err(ModelError::InvalidUrl {
                    reason: "ループバック以外のホストは許可されません",
                });
            }
            let pq = uri.path_and_query().map_or("/", |p| p.as_str());
            let authority = uri.authority().map_or("", |a| a.as_str());
            let plain =
                ureq::http::Uri::try_from(format!("http://{authority}{pq}")).map_err(|_| {
                    ModelError::InvalidUrl {
                        reason: "URL を解析できません",
                    }
                })?;
            perform(&self.agent, plain, deadline, if_none_match)
        }
    }
}

// 本ファイルでは `#[cfg(test)]` のモジュールを末尾にまとめる。api_surface.rs の
// `model_tls_posture_violations` は最初の `#[cfg(test)]` より前を本番部分として検査するため。
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
        let r = t.get(&url, Instant::now() + Duration::from_secs(20), None);
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
        let r = t.get(&url, Instant::now() + Duration::from_secs(20), None);
        assert!(r.is_err());
        let _ = h.join();
    }

    #[test]
    fn expired_deadline_is_timed_out_without_connecting() {
        let url = super::super::url::validate_https_url("https://127.0.0.1:1/m").expect("url");
        let mut t = UreqTransport::new().expect("transport");
        let r = t.get(&url, Instant::now() - Duration::from_secs(1), None);
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

    /// 無応答の相手に対し、全体期限（`timeout_global(remaining)`）が 30 秒のフェーズ上限より
    /// 先に効いて `TimedOut` になる（本番 `UreqTransport`。TLS ハンドシェイク待ちで停止）。
    /// SYN 段階の純粋な接続タイムアウトはループバックで決定的に再現できないため、接続失敗
    /// （既存テスト）と deadline による打ち切りで代替する（#3127・設計 §21）。
    #[test]
    fn production_transport_times_out_on_unresponsive_peer_by_deadline() {
        use super::super::test_http_server::{MockReply, MockServer};
        let srv = MockServer::start(vec![MockReply::Stall]);
        let url =
            super::super::url::validate_https_url(&format!("https://127.0.0.1:{}/m", srv.port()))
                .expect("url");
        let mut t = UreqTransport::new().expect("transport");
        let started = Instant::now();
        let r = t.get(&url, Instant::now() + Duration::from_millis(1500), None);
        match r {
            Err(ModelError::Network(e)) => assert_eq!(e.kind(), io::ErrorKind::TimedOut),
            _ => panic!("expected TimedOut"),
        }
        assert!(started.elapsed() < Duration::from_secs(20));
    }

    #[test]
    fn loopback_transport_times_out_waiting_for_response_headers() {
        use super::super::test_http_server::{MockReply, MockServer};
        let srv = MockServer::start(vec![MockReply::Stall]);
        let url =
            super::super::url::validate_https_url(&format!("https://127.0.0.1:{}/m", srv.port()))
                .expect("url");
        let mut t = loopback::PlaintextLoopbackTransport::new();
        let started = Instant::now();
        let r = t.get(&url, Instant::now() + Duration::from_secs(5), None);
        match r {
            Err(ModelError::Network(e)) => assert_eq!(e.kind(), io::ErrorKind::TimedOut),
            _ => panic!("expected TimedOut"),
        }
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn loopback_transport_times_out_while_receiving_body() {
        use super::super::test_http_server::{MockReply, MockServer};
        let srv = MockServer::start(vec![MockReply::HeadersThenStall {
            declared_len: 4096,
            body: vec![1u8; 16],
        }]);
        let url =
            super::super::url::validate_https_url(&format!("https://127.0.0.1:{}/m", srv.port()))
                .expect("url");
        let mut t = loopback::PlaintextLoopbackTransport::new();
        let resp = t
            .get(&url, Instant::now() + Duration::from_millis(1500), None)
            .unwrap_or_else(|_| panic!("headers should arrive"));
        let mut body = resp.body;
        let mut buf = Vec::new();
        // 本文受信中の期限切れは ureq が `io::Error` へ包んで返し、種別は `Other`（実測）。
        // 種別には依存せず「ハングせず Err になる」ことだけを固定する。
        let started = Instant::now();
        assert!(body.read_to_end(&mut buf).is_err());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn loopback_transport_refuses_non_loopback_hosts_without_connecting() {
        let url = super::super::url::validate_https_url("https://example.com/m").expect("url");
        let mut t = loopback::PlaintextLoopbackTransport::new();
        let r = t.get(&url, Instant::now() + Duration::from_secs(5), None);
        assert!(matches!(r, Err(ModelError::InvalidUrl { .. })));
    }
}
