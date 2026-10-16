//! `JsonTransport` の本番実装 `UreqJsonTransport`（イシュー #3393。`cfg(unix)`）。
//!
//! 役割: HF Hub API（`https://huggingface.co/api/...`）の JSON 取得だけを担う転送層。
//! `api.rs` の `resolve_revision_via`／`list_files_via` が `JsonTransport` 越しに呼ぶ。
//! ファイル本体の取得とキャッシュ書き込みは facade の `download_with` に任せ、ここでは行わない。
//!
//! 固定設定（facade の `model/transport.rs` と同方針。deps-policy.md 第 11 区分）:
//! `TlsProvider::NativeTls`＋`PlatformVerifier`・`https_only(true)`・`proxy(None)`・
//! `max_redirects(0)`（リダイレクトは追従せず 3xx を呼び出し側へ返して拒否させる）・全タイムアウト明示・
//! `accept_encoding("identity")`。認証ヘッダは付けない（A01）。証明書検証を無効化する手段は置かない
//! （A02）。ureq のエラー文字列は URL を含みうるため `Display` を通さず静的メッセージへ変換する（A09）。
//!
//! `ureq` のトークンはこのファイルに閉じる（`tests/api_surface.rs` で機械検査）。テスト専用の
//! 平文ループバック転送は末尾の `#[cfg(test)] mod loopback` にだけ存在する（本番部分に
//! `https_only(false)` を置かない）。

use std::io::{self, Read};
use std::time::Duration;

use ureq::http::Uri;
use ureq::tls::{RootCerts, TlsConfig, TlsProvider};

use super::{JsonResponse, JsonTransport};
use crate::HfError;
use crate::url::ApiUrl;

/// 名前解決・接続（TLS ハンドシェイク含む）・リクエスト送信の各上限。
const TIMEOUT_PHASE: Duration = Duration::from_secs(30);
/// 応答ヘッダ受信の上限。
const TIMEOUT_RECV_RESPONSE: Duration = Duration::from_secs(30);
/// ボディ受信の上限（JSON は 1 MiB 以下のため短く取る）。
const TIMEOUT_RECV_BODY: Duration = Duration::from_secs(60);
/// リクエスト全体の期限。
const TIMEOUT_GLOBAL: Duration = Duration::from_secs(120);
/// 接続先として許す唯一のホスト（`url.rs` の BASE と一致。多層防御）。
const ALLOWED_HOST: &str = "huggingface.co";

fn net(kind: io::ErrorKind, msg: &'static str) -> HfError {
    HfError::Network(io::Error::new(kind, msg))
}

/// ureq のエラーを URL を含まない静的メッセージへ変換する。
fn map_ureq_error(e: ureq::Error, max: usize) -> HfError {
    match e {
        ureq::Error::RequireHttpsOnly(_) => net(
            io::ErrorKind::InvalidInput,
            "https 以外への接続は許可されません",
        ),
        ureq::Error::Timeout(_) => net(io::ErrorKind::TimedOut, "HTTP 転送がタイムアウトしました"),
        ureq::Error::HostNotFound => net(io::ErrorKind::NotFound, "ホストを解決できません"),
        ureq::Error::ConnectionFailed => {
            net(io::ErrorKind::ConnectionRefused, "接続に失敗しました")
        }
        ureq::Error::BodyExceedsLimit(_) => HfError::ResponseTooLarge { max },
        ureq::Error::Io(io_err) => HfError::Network(io::Error::new(
            io_err.kind(),
            "HTTP 転送の I/O に失敗しました",
        )),
        _ => net(io::ErrorKind::Other, "HTTP 転送に失敗しました"),
    }
}

/// 固定構成の `Agent` を作り、不変条件を読み戻して照合する。ureq は provider と feature の
/// 不整合をリクエスト時に panic で報告しうるため、feature 整合は単体テストが実リクエストで固定する。
fn build_agent() -> Result<ureq::Agent, HfError> {
    let tls = TlsConfig::builder()
        .provider(TlsProvider::NativeTls)
        .root_certs(RootCerts::PlatformVerifier)
        .build();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .max_redirects(0)
        .http_status_as_error(false)
        .proxy(None)
        .user_agent(concat!("fandhe-ai-hf-hub/", env!("CARGO_PKG_VERSION")))
        .accept_encoding("identity")
        .tls_config(tls)
        .timeout_resolve(Some(TIMEOUT_PHASE))
        .timeout_connect(Some(TIMEOUT_PHASE))
        .timeout_send_request(Some(TIMEOUT_PHASE))
        .timeout_recv_response(Some(TIMEOUT_RECV_RESPONSE))
        .timeout_recv_body(Some(TIMEOUT_RECV_BODY))
        .timeout_global(Some(TIMEOUT_GLOBAL))
        .build()
        .into();
    check_invariants(&agent)?;
    Ok(agent)
}

fn check_invariants(agent: &ureq::Agent) -> Result<(), HfError> {
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

/// 多層防御（A10）: `ApiUrl` を `Uri` として再解析し、https・資格情報なし・host 完全一致・
/// ポート指定なしを接続前に照合する。
fn checked_api_uri(url: &ApiUrl) -> Result<Uri, HfError> {
    let bad = || net(io::ErrorKind::InvalidInput, "接続先 URL が許可されません");
    let uri = Uri::try_from(url.as_str()).map_err(|_| bad())?;
    let ok = uri.scheme_str() == Some("https")
        && uri.authority().is_some_and(|a| !a.as_str().contains('@'))
        && uri.host() == Some(ALLOWED_HOST)
        && uri.port().is_none();
    if ok { Ok(uri) } else { Err(bad()) }
}

/// `Content-Encoding` の全ヘッダ行を集め、`identity` 以外が 1 つでもあれば拒否する。
/// `get` は最初の値しか返さないため `get_all` を使い、変換失敗も拒否にする。
fn ensure_identity_encoding(headers: &ureq::http::HeaderMap) -> Result<(), HfError> {
    for v in headers.get_all("content-encoding") {
        let s = v.to_str().map_err(|_| HfError::InvalidResponse {
            reason: "Content-Encoding を解釈できない",
        })?;
        let non_identity = s.split(',').any(|p| {
            let p = p.trim();
            !p.is_empty() && !p.eq_ignore_ascii_case("identity")
        });
        if non_identity {
            return Err(HfError::InvalidResponse {
                reason: "圧縮された応答は許可されない",
            });
        }
    }
    Ok(())
}

/// 共有後段: GET を 1 回実行し、上限付きでボディを読む。本番と（テストの）ループバックが共有する。
/// 200 以外は本体も `Location` も読まずにステータスだけ返す（非追従と情報非露出の両立）。
fn perform(agent: &ureq::Agent, uri: Uri, max_bytes: usize) -> Result<JsonResponse, HfError> {
    let resp = agent
        .get(uri)
        .call()
        .map_err(|e| map_ureq_error(e, max_bytes))?;
    let status = resp.status().as_u16();
    if status != 200 {
        return Ok(JsonResponse {
            status,
            body: Vec::new(),
        });
    }
    ensure_identity_encoding(resp.headers())?;
    let (_, body) = resp.into_parts();
    let too_large = HfError::ResponseTooLarge { max: max_bytes };
    if body.content_length().is_some_and(|n| n > max_bytes as u64) {
        return Err(too_large);
    }
    // ureq の `limit(n)` は「ちょうど n バイト」の本体まで失敗させるため、上限判定は
    // `take(max + 1)` で自前に行い、ureq 側は実質無制限にする。
    let mut reader = body
        .into_with_config()
        .limit(u64::MAX)
        .reader()
        .take(max_bytes as u64 + 1);
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map_err(|e| {
        HfError::Network(io::Error::new(e.kind(), "HTTP ボディの受信に失敗しました"))
    })?;
    if buf.len() > max_bytes {
        return Err(too_large);
    }
    Ok(JsonResponse { status, body: buf })
}

/// ureq による本番の JSON 転送実装。
pub(crate) struct UreqJsonTransport {
    agent: ureq::Agent,
}

impl UreqJsonTransport {
    /// 固定構成の転送層を作る。公開化（#3394）まで本番コードからは呼ばれない。
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new() -> Result<Self, HfError> {
        Ok(Self {
            agent: build_agent()?,
        })
    }
}

impl JsonTransport for UreqJsonTransport {
    fn get_json(&mut self, url: &ApiUrl, max_bytes: usize) -> Result<JsonResponse, HfError> {
        let uri = checked_api_uri(url)?;
        perform(&self.agent, uri, max_bytes)
    }
}

#[cfg(test)]
mod loopback {
    //! テスト専用の平文ループバック転送。本番と同じ前段（`checked_api_uri`）・後段（`perform`）を
    //! 通し、agent の `https_only` とタイムアウトだけを変える。
    use super::*;
    use crate::api::{MAX_RESPONSE_BYTES, list_files_via, resolve_revision_via};
    use crate::test_http_server::{MockReply, MockServer};
    use crate::{CommitSha, HfRepoId, Revision};

    pub(super) struct PlaintextLoopbackJsonTransport {
        agent: ureq::Agent,
        port: u16,
    }

    impl PlaintextLoopbackJsonTransport {
        pub(super) fn new(port: u16) -> Self {
            let t = Some(Duration::from_secs(2));
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .https_only(false)
                .max_redirects(0)
                .http_status_as_error(false)
                .proxy(None)
                .user_agent(concat!("fandhe-ai-hf-hub/", env!("CARGO_PKG_VERSION")))
                .accept_encoding("identity")
                .timeout_resolve(t)
                .timeout_connect(t)
                .timeout_send_request(t)
                .timeout_recv_response(t)
                .timeout_recv_body(t)
                .timeout_global(Some(Duration::from_secs(10)))
                .build()
                .into();
            Self { agent, port }
        }
    }

    impl JsonTransport for PlaintextLoopbackJsonTransport {
        fn get_json(&mut self, url: &ApiUrl, max_bytes: usize) -> Result<JsonResponse, HfError> {
            let uri = checked_api_uri(url)?;
            let pq = uri.path_and_query().map(|p| p.as_str()).unwrap_or("/");
            let local = Uri::try_from(format!("http://127.0.0.1:{}{pq}", self.port))
                .map_err(|_| net(io::ErrorKind::InvalidInput, "ループバック URL が不正"))?;
            perform(&self.agent, local, max_bytes)
        }
    }

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn repo() -> HfRepoId {
        HfRepoId::parse("org/model").unwrap()
    }

    fn json_ok(body: &str) -> MockReply {
        MockReply::Full {
            status: 200,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body.as_bytes().to_vec(),
        }
    }

    fn call(reply: MockReply) -> (Result<JsonResponse, HfError>, MockServer) {
        let srv = MockServer::start(vec![reply]);
        let mut t = PlaintextLoopbackJsonTransport::new(srv.port());
        let url = crate::url::revision_api_url(&repo(), &Revision::main());
        let r = t.get_json(&url, MAX_RESPONSE_BYTES);
        (r, srv)
    }

    #[test]
    fn ok_200_returns_body_and_sends_expected_request() {
        let (r, srv) = call(json_ok(&format!(r#"{{"sha":"{SHA}"}}"#)));
        let r = r.unwrap();
        assert_eq!(r.status, 200);
        assert!(String::from_utf8(r.body).unwrap().contains(SHA));
        let rec = srv.recorded();
        assert_eq!(rec.len(), 1);
        assert!(
            rec[0]
                .request_line
                .starts_with("GET /api/models/org/model/revision/main "),
            "{}",
            rec[0].request_line
        );
        let h = &rec[0].headers;
        assert!(!h.iter().any(|(k, _)| k == "authorization"));
        assert!(
            h.iter()
                .any(|(k, v)| k == "accept-encoding" && v == "identity")
        );
        assert!(
            h.iter()
                .any(|(k, v)| k == "user-agent" && v.starts_with("fandhe-ai-hf-hub/"))
        );
    }

    #[test]
    fn resolve_and_list_work_through_loopback() {
        let srv = MockServer::start(vec![
            json_ok(&format!(r#"{{"sha":"{SHA}"}}"#)),
            json_ok(&format!(
                r#"{{"sha":"{SHA}","siblings":[{{"rfilename":"model.safetensors"}}]}}"#
            )),
        ]);
        let mut t = PlaintextLoopbackJsonTransport::new(srv.port());
        let sha = resolve_revision_via(&mut t, &repo(), &Revision::main()).unwrap();
        assert_eq!(sha.as_str(), SHA);
        let sha = CommitSha::parse(SHA).unwrap();
        let files = list_files_via(&mut t, &repo(), &sha).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path(), "model.safetensors");
    }

    #[test]
    fn status_codes_map_through_api() {
        for status in [401u16, 403, 404, 503] {
            let srv = MockServer::start(vec![MockReply::Full {
                status,
                headers: vec![],
                body: b"{}".to_vec(),
            }]);
            let mut t = PlaintextLoopbackJsonTransport::new(srv.port());
            let e = resolve_revision_via(&mut t, &repo(), &Revision::main()).unwrap_err();
            match (status, &e) {
                (401, HfError::Unauthorized)
                | (403, HfError::Gated)
                | (404, HfError::NotFound)
                | (503, HfError::Http { status: 503 }) => {}
                _ => panic!("status {status}: {e:?}"),
            }
        }
    }

    #[test]
    fn redirects_are_not_followed_and_location_is_not_exposed() {
        for status in [301u16, 302, 303, 307, 308] {
            for loc in [
                "https://evil.example/x",
                "https://cdn-lfs.huggingface.co/y",
                "http://127.0.0.1:1/z",
            ] {
                let srv = MockServer::start(vec![MockReply::Full {
                    status,
                    headers: vec![("Location".into(), loc.into())],
                    body: Vec::new(),
                }]);
                let mut t = PlaintextLoopbackJsonTransport::new(srv.port());
                let e = resolve_revision_via(&mut t, &repo(), &Revision::main()).unwrap_err();
                assert!(
                    matches!(e, HfError::Http { status: s } if s == status),
                    "{e:?}"
                );
                assert_eq!(srv.connection_count(), 1);
                let shown = format!("{e} {e:?}");
                assert!(
                    !shown.contains("evil") && !shown.contains("cdn-lfs"),
                    "{shown}"
                );
            }
        }
    }

    #[test]
    fn exactly_max_bytes_is_accepted() {
        let mut body = br#"{"sha":""#.to_vec();
        body.extend_from_slice(SHA.as_bytes());
        body.extend_from_slice(br#""}"#);
        body.resize(MAX_RESPONSE_BYTES, b' ');
        let (r, _srv) = call(MockReply::Full {
            status: 200,
            headers: vec![],
            body,
        });
        assert_eq!(r.unwrap().body.len(), MAX_RESPONSE_BYTES);
    }

    #[test]
    fn content_length_over_max_is_rejected_without_reading() {
        let (r, _srv) = call(MockReply::Full {
            status: 200,
            headers: vec![],
            body: vec![b' '; MAX_RESPONSE_BYTES + 1],
        });
        assert!(matches!(r, Err(HfError::ResponseTooLarge { max }) if max == MAX_RESPONSE_BYTES));
    }

    #[test]
    fn chunked_over_max_is_rejected() {
        let (r, _srv) = call(MockReply::Chunked {
            body: vec![b' '; MAX_RESPONSE_BYTES + 1],
        });
        assert!(
            matches!(r, Err(HfError::ResponseTooLarge { .. })),
            "{:?}",
            r.err()
        );
    }

    #[test]
    fn truncated_body_is_network_error() {
        let (r, _srv) = call(MockReply::Truncated {
            declared_len: 100,
            body: b"{\"sha\"".to_vec(),
        });
        assert!(matches!(r, Err(HfError::Network(_))), "{:?}", r.err());
    }

    #[test]
    fn non_identity_content_encoding_is_rejected() {
        for headers in [
            vec![("Content-Encoding".to_owned(), "gzip".to_owned())],
            vec![
                ("Content-Encoding".to_owned(), "identity".to_owned()),
                ("Content-Encoding".to_owned(), "gzip".to_owned()),
            ],
        ] {
            let (r, _srv) = call(MockReply::Full {
                status: 200,
                headers,
                body: b"{}".to_vec(),
            });
            assert!(
                matches!(r, Err(HfError::InvalidResponse { .. })),
                "{:?}",
                r.err()
            );
        }
    }

    #[test]
    fn stall_times_out() {
        let (r, _srv) = call(MockReply::Stall);
        match r {
            Err(HfError::Network(e)) => assert_eq!(e.kind(), io::ErrorKind::TimedOut),
            other => panic!("expected TimedOut: {:?}", other.err()),
        }
    }

    #[test]
    fn headers_then_stall_does_not_hang() {
        let (r, _srv) = call(MockReply::HeadersThenStall {
            declared_len: 100,
            body: b"{".to_vec(),
        });
        assert!(r.is_err());
    }

    #[test]
    fn build_agent_satisfies_invariants() {
        let agent = build_agent().expect("agent");
        assert!(check_invariants(&agent).is_ok());
    }

    #[test]
    fn plain_http_is_rejected_by_https_only_without_connecting() {
        let agent = build_agent().expect("agent");
        let e = agent
            .get("http://127.0.0.1:1/")
            .call()
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(map_ureq_error(e, 1), HfError::Network(_)));
    }

    fn closed_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr").port()
    }

    #[test]
    fn https_connection_failure_is_err_not_panic() {
        // 閉じたポートへの https 要求が panic せず `Err(Network)` になる
        // （NativeTls の feature 整合を外部通信なしで固定する）。
        let agent = build_agent().expect("agent");
        let uri = Uri::try_from(format!("https://127.0.0.1:{}/", closed_port())).unwrap();
        let r = perform(&agent, uri, 1024);
        assert!(matches!(r, Err(HfError::Network(_))), "{:?}", r.err());
    }

    #[test]
    fn tls_handshake_failure_on_plain_listener_is_err_not_panic() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = l.local_addr().expect("addr").port();
        let h = std::thread::spawn(move || {
            let _ = l.accept();
        });
        let agent = build_agent().expect("agent");
        let uri = Uri::try_from(format!("https://127.0.0.1:{port}/")).unwrap();
        assert!(perform(&agent, uri, 1024).is_err());
        let _ = h.join();
    }

    #[test]
    fn non_hf_hosts_are_rejected_before_connecting() {
        for u in [
            "https://evil.example/api/models/a/b/revision/main",
            "https://huggingface.co@evil.example/x",
            "https://user:s3cr3tTOKEN@huggingface.co/x",
            "http://huggingface.co/x",
            "https://huggingface.co:8443/x",
            "https://huggingface.co.evil.example/x",
        ] {
            let mut t = UreqJsonTransport::new().unwrap();
            let e = t.get_json(&ApiUrl::for_test(u), 1024).unwrap_err();
            assert!(matches!(e, HfError::Network(_)), "{u}: {e:?}");
            let shown = format!("{e} {e:?}");
            assert!(
                !shown.contains("s3cr3tTOKEN") && !shown.contains("://"),
                "{shown}"
            );
        }
    }

    #[test]
    fn valid_api_url_passes_precheck() {
        let url = crate::url::revision_api_url(&repo(), &Revision::main());
        assert!(checked_api_uri(&url).is_ok());
    }
}
