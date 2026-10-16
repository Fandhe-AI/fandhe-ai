//! テスト専用のスクリプト式 HTTP サーバ（イシュー #3393。`#[cfg(test)]`）。
//!
//! 役割: `UreqJsonTransport` のループバックテストが使う、`127.0.0.1` だけに bind する最小の
//! HTTP/1.1 サーバ。外部へは一切接続しない。台本（`MockReply`）を接続順に 1 件ずつ消費する。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// 1 接続分の応答の台本。
pub(crate) enum MockReply {
    /// 通常応答（`Content-Length` 付き）。
    Full {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    },
    /// `Transfer-Encoding: chunked` 応答。
    Chunked { body: Vec<u8> },
    /// ヘッダを送らず待つ。
    Stall,
    /// ヘッダ送信後、ボディを途中まで送って待つ。
    HeadersThenStall { declared_len: usize, body: Vec<u8> },
    /// 宣言長未満のボディを送って close する。
    Truncated { declared_len: usize, body: Vec<u8> },
}

/// 受信したリクエストの記録（ヘッダ名は小文字化）。
#[derive(Clone)]
pub(crate) struct Recorded {
    pub(crate) request_line: String,
    pub(crate) headers: Vec<(String, String)>,
}

pub(crate) struct MockServer {
    port: u16,
    stop: Arc<AtomicBool>,
    conns: Arc<AtomicUsize>,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    handle: Option<JoinHandle<()>>,
}

impl MockServer {
    pub(crate) fn start(script: Vec<MockReply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let port = listener.local_addr().expect("addr").port();
        let stop = Arc::new(AtomicBool::new(false));
        let conns = Arc::new(AtomicUsize::new(0));
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let (s, c, r) = (stop.clone(), conns.clone(), recorded.clone());
        let handle = std::thread::spawn(move || {
            let mut script = script.into_iter();
            while !s.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        c.fetch_add(1, Ordering::SeqCst);
                        let Some(reply) = script.next() else { continue };
                        serve(stream, reply, &r, &s);
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        });
        Self {
            port,
            stop,
            conns,
            recorded,
            handle: Some(handle),
        }
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.conns.load(Ordering::SeqCst)
    }

    pub(crate) fn recorded(&self) -> Vec<Recorded> {
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
    stream.set_nonblocking(false).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let mut lines = text.split("\r\n");
    let request_line = lines.next()?.to_owned();
    let headers = lines
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    Some(Recorded {
        request_line,
        headers,
    })
}

/// クライアントが切断するかサーバが停止するまで（上限 10 秒）接続を保持する。
fn hold(stream: &TcpStream, stop: &AtomicBool) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(50)));
    let start = Instant::now();
    let mut tmp = [0u8; 16];
    let mut s = stream;
    while !stop.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(10) {
        if let Ok(0) = s.read(&mut tmp) {
            break;
        }
    }
}

fn serve(mut stream: TcpStream, reply: MockReply, rec: &Mutex<Vec<Recorded>>, stop: &AtomicBool) {
    let Some(req) = read_request(&mut stream) else {
        return;
    };
    rec.lock().expect("lock").push(req);
    // 書き込み失敗（クライアントが先に切断）は想定内なので無視する。
    match reply {
        MockReply::Full {
            status,
            headers,
            body,
        } => {
            let mut head = format!("HTTP/1.1 {status} X\r\nConnection: close\r\n");
            for (k, v) in &headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
        MockReply::Chunked { body } => {
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n",
            );
            for c in body.chunks(8192) {
                let _ = stream.write_all(format!("{:x}\r\n", c.len()).as_bytes());
                let _ = stream.write_all(c);
                let _ = stream.write_all(b"\r\n");
            }
            let _ = stream.write_all(b"0\r\n\r\n");
        }
        MockReply::Stall => hold(&stream, stop),
        MockReply::HeadersThenStall { declared_len, body } => {
            let head = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {declared_len}\r\n\r\n"
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
            hold(&stream, stop);
        }
        MockReply::Truncated { declared_len, body } => {
            let head = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {declared_len}\r\n\r\n"
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    }
}
