//! ループバック用の最小 HTTP/1.1 モックサーバ（イシュー #3127。`cfg(all(test, unix))`）。
//!
//! `loopback_tests.rs`・`model.rs::tests` が transport 層の実ソケット経路（リクエスト送信内容・
//! ヘッダ実値・チャンク・途中切断・無応答）を CI で外部通信なしに確認するための足場。std のみで
//! `127.0.0.1:0` に bind し、1 接続 1 リクエストで必ず `Connection: close` を返す
//! （keep-alive の解析を避ける）。外部ホストへは一切接続しない。

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// サーバが接続ごとに返す応答の台本。
pub(super) enum MockReply {
    /// ステータス・追加ヘッダ（`Content-Length` は自動付与）・本文。
    Full {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: Vec<u8>,
    },
    /// `Transfer-Encoding: chunked`（`Content-Length` なし）。
    Chunked {
        headers: Vec<(&'static str, String)>,
        chunks: Vec<Vec<u8>>,
    },
    /// 宣言長より短い本文を送って切断する。
    ShortBody { declared_len: usize, body: Vec<u8> },
    /// 要求を読んだ後、無応答で保持する。
    Stall,
    /// ヘッダと本文の一部を送った後に停止する。
    HeadersThenStall { declared_len: usize, body: Vec<u8> },
}

/// 受信したリクエストの記録。
#[derive(Clone, Debug)]
pub(super) struct Recorded {
    pub(super) path: String,
    pub(super) headers: Vec<(String, String)>,
}

impl Recorded {
    pub(super) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

pub(super) struct MockServer {
    addr: SocketAddr,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    connections: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

/// 無応答・途中停止の保持時間（クライアントのタイムアウトより長く、Drop が待てる上限内）。
const STALL_HOLD: Duration = Duration::from_secs(6);
/// サーバ全体の寿命上限（ハング防止）。
const SERVER_LIFETIME: Duration = Duration::from_secs(60);

impl MockServer {
    pub(super) fn start(script: Vec<MockReply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let addr = listener.local_addr().expect("addr");
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let connections = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (r, c, s) = (recorded.clone(), connections.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            let started = Instant::now();
            let mut script = script.into_iter();
            while !s.load(Ordering::SeqCst) && started.elapsed() < SERVER_LIFETIME {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let Some(reply) = script.next() else { break };
                        c.fetch_add(1, Ordering::SeqCst);
                        serve(stream, reply, &r, &s);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            addr,
            recorded,
            connections,
            stop,
            handle: Some(handle),
        }
    }

    pub(super) fn port(&self) -> u16 {
        self.addr.port()
    }

    /// 受け付けた接続数（「接続されなかった」ことの確認に使う）。
    pub(super) fn connection_count(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    pub(super) fn recorded(&self) -> Vec<Recorded> {
        self.recorded.lock().expect("lock").clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => return None,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
        if buf.len() > 64 * 1024 {
            return None;
        }
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let mut lines = text.split("\r\n");
    let first = lines.next()?;
    let path = first.split(' ').nth(1)?.to_string();
    let headers = lines
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    Some(Recorded { path, headers })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        304 => "Not Modified",
        404 => "Not Found",
        301 | 302 | 303 | 307 | 308 => "Redirect",
        _ => "Status",
    }
}

fn head(status: u16, headers: &[(&'static str, String)]) -> String {
    let mut h = format!(
        "HTTP/1.1 {status} {}\r\nConnection: close\r\n",
        reason(status)
    );
    for (k, v) in headers {
        h.push_str(&format!("{k}: {v}\r\n"));
    }
    h
}

fn hold(stop: &AtomicBool) {
    let t = Instant::now();
    while t.elapsed() < STALL_HOLD && !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn serve(
    mut stream: TcpStream,
    reply: MockReply,
    recorded: &Mutex<Vec<Recorded>>,
    stop: &AtomicBool,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let Some(req) = read_request(&mut stream) else {
        return;
    };
    recorded.lock().expect("lock").push(req);
    match reply {
        MockReply::Full {
            status,
            headers,
            body,
        } => {
            let mut h = head(status, &headers);
            if status != 304 {
                h.push_str(&format!("Content-Length: {}\r\n", body.len()));
            }
            h.push_str("\r\n");
            let _ = stream.write_all(h.as_bytes());
            let _ = stream.write_all(&body);
        }
        MockReply::Chunked { headers, chunks } => {
            let mut h = head(200, &headers);
            h.push_str("Transfer-Encoding: chunked\r\n\r\n");
            let _ = stream.write_all(h.as_bytes());
            for c in chunks {
                let _ = stream.write_all(format!("{:x}\r\n", c.len()).as_bytes());
                let _ = stream.write_all(&c);
                let _ = stream.write_all(b"\r\n");
            }
            let _ = stream.write_all(b"0\r\n\r\n");
        }
        MockReply::ShortBody { declared_len, body } => {
            let mut h = head(200, &[]);
            h.push_str(&format!("Content-Length: {declared_len}\r\n\r\n"));
            let _ = stream.write_all(h.as_bytes());
            let _ = stream.write_all(&body);
        }
        MockReply::Stall => hold(stop),
        MockReply::HeadersThenStall { declared_len, body } => {
            let mut h = head(200, &[]);
            h.push_str(&format!("Content-Length: {declared_len}\r\n\r\n"));
            let _ = stream.write_all(h.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
            hold(stop);
        }
    }
    let _ = stream.flush();
}
