//! 期限つきのソケット I/O（connect・accept・read・write）。
//!
//! 役割: `std::net` の素のブロッキング操作を、すべて**絶対期限**（`Instant`）で包む。1 回ごとの
//! timeout だけだと、少しずつ送るピアが期限を延ばし続けられるため（OWASP A04）。
//! `frame.rs` は `Read`／`Write` に対して汎用で、期限は [`DeadlineStream`] 経由で与える。
//! `rendezvous.rs` と後続 #3145 が呼ぶ。根拠: `docs/facade-distributed-host-collective-decision.md` §5。

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use super::error::{DistributedError, Result};

/// accept のポーリング間隔の上限。
const ACCEPT_POLL: Duration = Duration::from_millis(5);
/// connect 再試行の初期・最大間隔。
const RETRY_INITIAL: Duration = Duration::from_millis(50);
const RETRY_MAX: Duration = Duration::from_millis(500);

/// `TcpStream` を絶対期限つきの `Read`／`Write` として見せる薄いラッパー。
///
/// 呼ばれるたびに残り時間（と 1 回あたりの上限 `io_timeout` の小さい方）を
/// `set_read_timeout`／`set_write_timeout` へ設定する。残りが 0 なら `TimedOut`。
pub struct DeadlineStream<'a> {
    stream: &'a TcpStream,
    deadline: Instant,
    io_timeout: Duration,
}

impl<'a> DeadlineStream<'a> {
    /// 期限と 1 回あたりの上限を指定して包む。
    pub fn new(stream: &'a TcpStream, deadline: Instant, io_timeout: Duration) -> Self {
        Self {
            stream,
            deadline,
            io_timeout,
        }
    }

    fn slice(&self) -> io::Result<Duration> {
        let rem = self.deadline.saturating_duration_since(Instant::now());
        if rem.is_zero() {
            return Err(io::Error::from(io::ErrorKind::TimedOut));
        }
        Ok(rem.min(self.io_timeout))
    }
}

impl Read for DeadlineStream<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stream.set_read_timeout(Some(self.slice()?))?;
        self.stream.read(buf)
    }
}

impl Write for DeadlineStream<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream.set_write_timeout(Some(self.slice()?))?;
        self.stream.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

/// 期限までに connect する。集合点がまだ待受を始めていない状況（`ConnectionRefused`）では
/// 短い間隔で再試行する。接続後は `TCP_NODELAY` を設定する。
pub fn connect_with_retry(
    addr: SocketAddr,
    connect_timeout: Duration,
    deadline: Instant,
) -> Result<TcpStream> {
    let mut backoff = RETRY_INITIAL;
    loop {
        let rem = deadline.saturating_duration_since(Instant::now());
        if rem.is_zero() {
            return Err(DistributedError::Timeout { phase: "connect" });
        }
        match TcpStream::connect_timeout(&addr, connect_timeout.min(rem)) {
            Ok(s) => {
                s.set_nodelay(true)?;
                return Ok(s);
            }
            Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => {
                std::thread::sleep(backoff.min(rem));
                backoff = (backoff * 2).min(RETRY_MAX);
            }
            Err(e) if e.kind() == io::ErrorKind::TimedOut => {
                return Err(DistributedError::Timeout { phase: "connect" });
            }
            Err(e) => return Err(DistributedError::Io(e)),
        }
    }
}

/// 非ブロッキングの listener を bind する（accept のタイムアウトが std に無いためポーリングする）。
pub fn bind_nonblocking(addr: SocketAddr) -> Result<TcpListener> {
    let l = TcpListener::bind(addr)?;
    l.set_nonblocking(true)?;
    Ok(l)
}

/// 期限つき accept。`listener` は非ブロッキングであること。受理したソケットは必ずブロッキングへ
/// 戻す（macOS／BSD では listener の非ブロッキング属性が継承されるため）。
///
/// 1 本の不正な接続で集合全体を失敗させない: accept 前に RST した接続（BSD／macOS は
/// `ConnectionAborted` を返す）や、受理直後にソケット設定が失敗した接続は捨てて受付を続ける。
/// どの分岐でもループ先頭で絶対期限を先に検査するため、期限は延びない。
pub fn accept_by(
    listener: &TcpListener,
    deadline: Instant,
    phase: &'static str,
) -> Result<(TcpStream, SocketAddr)> {
    loop {
        // accept が成功し続けても絶対期限を超えない（接続を送り続けるピアが期限を延ばせない）。
        if deadline.saturating_duration_since(Instant::now()).is_zero() {
            return Err(DistributedError::Timeout { phase });
        }
        match listener.accept() {
            Ok((s, peer)) => {
                if s.set_nonblocking(false).is_err() || s.set_nodelay(true).is_err() {
                    continue;
                }
                return Ok((s, peer));
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let rem = deadline.saturating_duration_since(Instant::now());
                if rem.is_zero() {
                    return Err(DistributedError::Timeout { phase });
                }
                std::thread::sleep(ACCEPT_POLL.min(rem));
            }
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::Interrupted
                        | io::ErrorKind::ConnectionAborted
                        | io::ErrorKind::ConnectionReset
                ) => {}
            Err(e) => return Err(DistributedError::Io(e)),
        }
    }
}

/// 双方向 shutdown（エラーを隣のランクへ早く伝える。失敗は無視する）。`ring::RingCommunicator` が
/// 失敗時（poisoned 遷移）に保持する全ソケットへ使う。
pub fn shutdown_both(stream: &TcpStream) {
    let _ = stream.shutdown(Shutdown::Both);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        (c, s)
    }

    #[test]
    fn silent_peer_read_times_out() {
        let (_c, s) = pair();
        let t = Instant::now();
        let mut ds = DeadlineStream::new(
            &s,
            Instant::now() + Duration::from_secs(5),
            Duration::from_millis(300),
        );
        let mut b = [0u8; 4];
        let e: DistributedError = ds.read_exact(&mut b).unwrap_err().into();
        assert!(matches!(e, DistributedError::Timeout { .. }), "{e}");
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn trickling_peer_cannot_extend_deadline() {
        let (mut c, s) = pair();
        let t = Instant::now();
        let h = std::thread::spawn(move || {
            for _ in 0..40 {
                if c.write_all(&[1]).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let mut ds = DeadlineStream::new(
            &s,
            Instant::now() + Duration::from_millis(400),
            Duration::from_secs(5),
        );
        let mut b = [0u8; 1000];
        let e: DistributedError = ds.read_exact(&mut b).unwrap_err().into();
        assert!(matches!(e, DistributedError::Timeout { .. }), "{e}");
        assert!(t.elapsed() < Duration::from_secs(10));
        drop(s);
        h.join().unwrap();
    }

    #[test]
    fn shutdown_write_is_peer_disconnected() {
        let (c, s) = pair();
        c.shutdown(Shutdown::Write).unwrap();
        let mut ds = DeadlineStream::new(
            &s,
            Instant::now() + Duration::from_secs(5),
            Duration::from_secs(5),
        );
        let mut b = [0u8; 4];
        let e: DistributedError = ds.read_exact(&mut b).unwrap_err().into();
        assert!(matches!(e, DistributedError::PeerDisconnected), "{e}");
    }

    #[test]
    fn accept_flood_does_not_extend_the_deadline() {
        // 接続が途切れず届き続けても accept_by は絶対期限で終わる（指摘 1 の類型）。
        let l = bind_nonblocking("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = l.local_addr().unwrap();
        let stop = std::sync::atomic::AtomicBool::new(false);
        let t = Instant::now();
        std::thread::scope(|sc| {
            sc.spawn(|| {
                let mut keep = Vec::new();
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    match TcpStream::connect(addr) {
                        Ok(c) => keep.push(c),
                        Err(_) => std::thread::sleep(Duration::from_millis(1)),
                    }
                    if keep.len() > 256 {
                        keep.clear();
                    }
                }
            });
            let deadline = Instant::now() + Duration::from_millis(300);
            let mut accepted = 0usize;
            let r = loop {
                match accept_by(&l, deadline, "test accept") {
                    Ok(_) => accepted += 1,
                    Err(e) => break e,
                }
            };
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            assert!(matches!(r, DistributedError::Timeout { .. }), "{r}");
            assert!(accepted > 0, "the flood must actually reach accept");
        });
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn accept_times_out_without_peer() {
        let l = bind_nonblocking("127.0.0.1:0".parse().unwrap()).unwrap();
        let r = accept_by(
            &l,
            Instant::now() + Duration::from_millis(100),
            "test accept",
        );
        assert!(matches!(r, Err(DistributedError::Timeout { .. })));
    }
}
