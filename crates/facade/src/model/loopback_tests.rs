//! ループバック HTTP モックによる `download` 経路の結合テスト（イシュー #3127。
//! `cfg(all(test, unix))`）。
//!
//! `ModelRegistry::download_via` へテスト専用の `PlaintextLoopbackTransport` を差し込み、
//! `std::net::TcpListener` のモックサーバ（`test_http_server.rs`）へ実ソケットで接続する。
//! プロセス内モック（`download.rs::tests`・`model.rs::tests`）が通らない実コード経路
//! （送信された `If-None-Match`・ヘッダ実値・チャンク・途中切断）を CI で外部通信なしに固定する
//! （`docs/model-download-design.md` §6・§21）。接続先は 127.0.0.1 のみ。URL は常に https
//! 検証済みの `ValidatedUrl` として渡り、`Location: http://...` は本番と同じ経路で拒否される。

use std::collections::HashMap;
use std::path::PathBuf;

use super::test_http_server::{MockReply, MockServer};
use super::transport::loopback::PlaintextLoopbackTransport;
use super::url::validate_https_url;
use super::*;

/// テストごとの一時キャッシュルート（Drop で削除）。
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let (pid, nanos, counter) = crate::fs_write::name_entropy();
        Self(std::env::temp_dir().join(format!("fandhe-loop-{tag}-{pid}-{counter}-{nanos}")))
    }
    fn registry(&self) -> ModelRegistry {
        ModelRegistry::with_cache_dir(&self.0)
    }
    fn model_path(&self) -> PathBuf {
        self.0.join("m/v1").join(MODEL_FILE_NAME)
    }
    fn leaf_names(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.0.join("m/v1"))
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn state(seed: f32) -> HashMap<String, Tensor<f32>> {
    let mut m = HashMap::new();
    m.insert(
        "fc.weight".to_string(),
        Tensor::new(vec![seed, -0.2, 0.3, -0.4, 0.5, 1.5], &[2, 3]).unwrap(),
    );
    m
}

fn model_bytes(seed: f32) -> Vec<u8> {
    crate::interop::safetensors::save_safetensors_f32_to_bytes(&state(seed), None).unwrap()
}

fn pin_of(data: &[u8]) -> Sha256Pin {
    let mut h = sha256::Sha256::new();
    h.update(data);
    Sha256Pin(h.finalize())
}

fn full(status: u16, headers: Vec<(&'static str, String)>, body: Vec<u8>) -> MockReply {
    MockReply::Full {
        status,
        headers,
        body,
    }
}

fn ok_etag(body: &[u8], etag: &str) -> MockReply {
    full(200, vec![("ETag", etag.to_string())], body.to_vec())
}

fn not_modified() -> MockReply {
    full(304, vec![], Vec::new())
}

fn redirect(loc: String) -> MockReply {
    full(302, vec![("Location", loc)], Vec::new())
}

fn dl_max(
    reg: &ModelRegistry,
    port: u16,
    path: &str,
    opts: DownloadOptions<'_>,
    max: u64,
) -> Result<(), ModelError> {
    let url = validate_https_url(&format!("https://127.0.0.1:{port}{path}")).unwrap();
    let mut t = PlaintextLoopbackTransport::new();
    reg.download_via(&mut t, url, "m", "v1", opts, max)
}

fn dl(reg: &ModelRegistry, srv: &MockServer, opts: DownloadOptions<'_>) -> Result<(), ModelError> {
    dl_max(reg, srv.port(), "/m", opts, MAX_MODEL_FILE_BYTES)
}

#[test]
fn ok_200_publishes_and_loads_back_without_leftovers() {
    let root = Root::new("ok");
    let body = model_bytes(0.1);
    let srv = MockServer::start(vec![ok_etag(&body, "\"e1\"")]);
    let reg = root.registry();
    dl(&reg, &srv, DownloadOptions::default()).unwrap();
    assert_eq!(std::fs::read(root.model_path()).unwrap(), body);
    assert_eq!(
        root.leaf_names(),
        vec!["download.json".to_string(), MODEL_FILE_NAME.to_string()]
    );
    assert_eq!(reg.load("m", "v1").unwrap().len(), 1);
    assert_eq!(srv.recorded()[0].header("if-none-match"), None);
}

#[test]
fn second_fetch_sends_if_none_match_and_reuses_on_304() {
    let root = Root::new("304");
    let body = model_bytes(0.1);
    let srv = MockServer::start(vec![ok_etag(&body, "\"e1\""), not_modified()]);
    let reg = root.registry();
    dl(&reg, &srv, DownloadOptions::default()).unwrap();
    let meta1 = std::fs::read(root.0.join("m/v1/download.json")).unwrap();
    dl(&reg, &srv, DownloadOptions::default()).unwrap();
    let rec = srv.recorded();
    assert_eq!(rec.len(), 2);
    assert_eq!(rec[0].header("if-none-match"), None);
    assert_eq!(rec[1].header("if-none-match"), Some("\"e1\""));
    assert_eq!(std::fs::read(root.model_path()).unwrap(), body);
    // 本体は不変で、download.json（fetched_at）だけが更新対象。ファイル集合は増えない。
    assert!(!meta1.is_empty());
    assert_eq!(root.leaf_names().len(), 2);
}

#[test]
fn tampered_cache_with_304_refetches_unconditionally() {
    let root = Root::new("tamper");
    let body = model_bytes(0.1);
    let srv = MockServer::start(vec![
        ok_etag(&body, "\"e1\""),
        not_modified(),
        ok_etag(&body, "\"e1\""),
    ]);
    let reg = root.registry();
    dl(&reg, &srv, DownloadOptions::default()).unwrap();
    std::fs::write(root.model_path(), model_bytes(9.0)).unwrap();
    dl(&reg, &srv, DownloadOptions::default()).unwrap();
    let rec = srv.recorded();
    assert_eq!(rec.len(), 3);
    assert_eq!(rec[1].header("if-none-match"), Some("\"e1\""));
    assert_eq!(rec[2].header("if-none-match"), None);
    assert_eq!(std::fs::read(root.model_path()).unwrap(), body);
}

#[test]
fn not_found_is_http_404_and_creates_no_directory() {
    let root = Root::new("404");
    let srv = MockServer::start(vec![full(404, vec![], b"nope".to_vec())]);
    let e = dl(&root.registry(), &srv, DownloadOptions::default()).unwrap_err();
    assert!(matches!(e, ModelError::Http { status: 404 }), "{e:?}");
    assert!(!root.0.join("m").exists());
}

#[test]
fn https_redirect_is_followed_on_the_new_path() {
    let root = Root::new("redir");
    let body = model_bytes(0.1);
    let srv = MockServer::start(vec![redirect("/other".into()), ok_etag(&body, "\"e1\"")]);
    dl(&root.registry(), &srv, DownloadOptions::default()).unwrap();
    let paths: Vec<String> = srv.recorded().into_iter().map(|r| r.path).collect();
    assert_eq!(paths, vec!["/m".to_string(), "/other".to_string()]);
    assert_eq!(std::fs::read(root.model_path()).unwrap(), body);
}

#[test]
fn redirect_to_plain_http_is_rejected_without_connecting() {
    let root = Root::new("redir-http");
    let other = MockServer::start(vec![full(200, vec![], Vec::new())]);
    let srv = MockServer::start(vec![redirect(format!(
        "http://127.0.0.1:{}/m",
        other.port()
    ))]);
    let e = dl(&root.registry(), &srv, DownloadOptions::default()).unwrap_err();
    assert!(matches!(e, ModelError::InvalidUrl { .. }), "{e:?}");
    assert_eq!(other.connection_count(), 0);
    assert!(!root.0.join("m").exists());
}

#[test]
fn redirect_chain_over_limit_is_too_many_redirects() {
    let root = Root::new("redir-many");
    let srv = MockServer::start((0..6).map(|_| redirect("/m".into())).collect());
    let e = dl(&root.registry(), &srv, DownloadOptions::default()).unwrap_err();
    assert!(
        matches!(e, ModelError::TooManyRedirects { max: 5 }),
        "{e:?}"
    );
}

#[test]
fn truncated_body_publishes_nothing_and_keeps_existing_model() {
    let root = Root::new("short");
    let good = model_bytes(0.1);
    let reg = root.registry();
    let first = MockServer::start(vec![ok_etag(&good, "\"e1\"")]);
    dl(&reg, &first, DownloadOptions::default()).unwrap();
    let srv = MockServer::start(vec![MockReply::ShortBody {
        declared_len: 4096,
        body: vec![0u8; 100],
    }]);
    let e = dl(&reg, &srv, DownloadOptions::default()).unwrap_err();
    // 実測: `Network(UnexpectedEof)`（ureq が自前の長さ照合より先に切断を検出する。
    // `ContentLengthMismatch` は転送層が切断を報告しないモック経路でのみ到達）。
    assert!(
        matches!(&e, ModelError::Network(io) if io.kind() == std::io::ErrorKind::UnexpectedEof),
        "{e:?}"
    );
    assert_eq!(std::fs::read(root.model_path()).unwrap(), good);
    assert_eq!(
        root.leaf_names(),
        vec!["download.json".to_string(), MODEL_FILE_NAME.to_string()]
    );
}

#[test]
fn declared_length_over_limit_is_too_large_before_directory_creation() {
    let root = Root::new("big-decl");
    let srv = MockServer::start(vec![full(200, vec![], vec![0u8; 8192])]);
    let url_port = srv.port();
    let e = dl_max(
        &root.registry(),
        url_port,
        "/m",
        DownloadOptions::default(),
        4096,
    )
    .unwrap_err();
    assert!(matches!(e, ModelError::TooLarge { .. }), "{e:?}");
    assert!(!root.0.join("m").exists());
}

#[test]
fn chunked_body_over_limit_is_cut_off_without_leftovers() {
    let root = Root::new("big-chunk");
    let srv = MockServer::start(vec![MockReply::Chunked {
        headers: vec![],
        chunks: vec![vec![0u8; 3000], vec![0u8; 3000], vec![0u8; 3000]],
    }]);
    let e = dl_max(
        &root.registry(),
        srv.port(),
        "/m",
        DownloadOptions::default(),
        4096,
    )
    .unwrap_err();
    assert!(matches!(e, ModelError::TooLarge { .. }), "{e:?}");
    assert!(!root.model_path().exists());
    assert!(root.leaf_names().is_empty());
}

#[test]
fn pin_match_publishes_and_mismatch_keeps_existing_model() {
    let root = Root::new("pin");
    let body = model_bytes(0.1);
    let reg = root.registry();
    let srv = MockServer::start(vec![ok_etag(&body, "\"e1\"")]);
    dl(
        &reg,
        &srv,
        DownloadOptions::default().expected_sha256(pin_of(&body)),
    )
    .unwrap();
    // 不一致: 既存は不変。保存値と pin が食い違うため最初から無条件 GET。
    let other = model_bytes(2.0);
    let srv2 = MockServer::start(vec![ok_etag(&other, "\"e2\"")]);
    let e = dl(
        &reg,
        &srv2,
        DownloadOptions::default().expected_sha256(pin_of(&body)),
    )
    .unwrap_err();
    assert!(matches!(e, ModelError::HashMismatch { .. }), "{e:?}");
    assert_eq!(std::fs::read(root.model_path()).unwrap(), body);
    assert_eq!(
        root.leaf_names(),
        vec!["download.json".to_string(), MODEL_FILE_NAME.to_string()]
    );
}

#[test]
fn pin_on_304_path_matches_stored_or_forces_unconditional_get() {
    let root = Root::new("pin304");
    let body = model_bytes(0.1);
    let reg = root.registry();
    let srv = MockServer::start(vec![
        ok_etag(&body, "\"e1\""),
        not_modified(),
        ok_etag(&body, "\"e1\""),
    ]);
    let pin = pin_of(&body);
    dl(&reg, &srv, DownloadOptions::default().expected_sha256(pin)).unwrap();
    // 保存値と pin が一致: 条件付き GET の 304 を再ハッシュで確認して再利用。
    dl(&reg, &srv, DownloadOptions::default().expected_sha256(pin)).unwrap();
    // pin が保存値と異なる: 条件なし GET になり、内容が pin と不一致なら HashMismatch。
    let wrong = pin_of(b"other");
    let e = dl(
        &reg,
        &srv,
        DownloadOptions::default().expected_sha256(wrong),
    )
    .unwrap_err();
    assert!(matches!(e, ModelError::HashMismatch { .. }), "{e:?}");
    let rec = srv.recorded();
    assert_eq!(rec[1].header("if-none-match"), Some("\"e1\""));
    assert_eq!(rec[2].header("if-none-match"), None);
    assert_eq!(std::fs::read(root.model_path()).unwrap(), body);
}

#[test]
fn content_encoding_on_second_header_line_is_rejected() {
    let root = Root::new("enc");
    let srv = MockServer::start(vec![full(
        200,
        vec![
            ("Content-Encoding", "identity".into()),
            ("Content-Encoding", "gzip".into()),
        ],
        model_bytes(0.1),
    )]);
    let e = dl(&root.registry(), &srv, DownloadOptions::default()).unwrap_err();
    assert!(matches!(e, ModelError::Network(_)), "{e:?}");
    assert!(!root.0.join("m").exists());
}

#[test]
fn progress_is_monotonic_with_total_for_length_and_none_for_chunked() {
    let body = model_bytes(0.1);
    let root = Root::new("prog");
    let srv = MockServer::start(vec![ok_etag(&body, "\"e1\"")]);
    let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
    let mut cb = |p: DownloadProgress| seen.push((p.received, p.total));
    dl(
        &root.registry(),
        &srv,
        DownloadOptions::default().progress(&mut cb),
    )
    .unwrap();
    assert!(!seen.is_empty());
    assert!(seen.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(seen.last().unwrap().0, body.len() as u64);
    assert!(seen.iter().all(|(_, t)| *t == Some(body.len() as u64)));

    let root2 = Root::new("prog-chunk");
    let half = body.len() / 2;
    let srv2 = MockServer::start(vec![MockReply::Chunked {
        headers: vec![],
        chunks: vec![body[..half].to_vec(), body[half..].to_vec()],
    }]);
    let mut seen2: Vec<(u64, Option<u64>)> = Vec::new();
    let mut cb2 = |p: DownloadProgress| seen2.push((p.received, p.total));
    dl(
        &root2.registry(),
        &srv2,
        DownloadOptions::default().progress(&mut cb2),
    )
    .unwrap();
    assert!(seen2.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(seen2.last().unwrap().0, body.len() as u64);
    assert!(seen2.iter().all(|(_, t)| t.is_none()));
}
