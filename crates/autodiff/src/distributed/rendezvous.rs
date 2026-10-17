//! ランク発見（ランク 0 の集合点）とリング接続の確立。
//!
//! 役割: 起動情報（`RendezvousConfig`）から、各ランクが次のランク `(r+1)%W` へ送る接続
//! （`to_next`）と前のランク `(r-1)%W` から受ける接続（`from_prev`）を確立して [`RingLinks`]
//! を返す。all-reduce 本体（#3145）はこの `RingLinks` と `frame.rs` の送受信 API を使う。
//! 根拠: `docs/facade-distributed-host-collective-decision.md` §3・§5・§7・§13（spec `:386-393`）。
//!
//! # リング待受アドレスの導出（記録 §13）
//!
//! ランク表に載る IP は、ランク 0 が集合点で観測した接続元 IP（記録 §3 手順 3）である。これと
//! リング待受の bind IP を別々の規則で決めると、IPv6・`127.0.0.x`・IPv4-mapped・非 loopback の
//! 組み合わせごとに「配布したアドレスに待受が無い」ずれが生じる。そこで待受 IP は設定値に
//! せず、観測される IP そのものから導く:
//!
//! - ランク 0 以外: 先に集合点へ接続し、その接続ソケットの `local_addr().ip()`（集合点が
//!   `peer_addr` として観測する IP と同一）にリング待受を bind してから、そのポートを Hello で
//!   申告する。ランク表を受けたら自分のエントリが `(その IP, 待受ポート)` と一致すること、
//!   ランク 0 のエントリの IP が設定した集合点の IP と一致することを検証する（NAT 等で一致
//!   しなければ、後の connect をタイムアウトさせず確立前に型付きエラーで止める）。
//! - ランク 0: 集合点の待受 IP（ピアが実際に到達した IP）にリング待受を bind する。
//!
//! IPv4-mapped は設定時と観測時の双方で `to_canonical` により IPv4 へ正規化する。前提は、ランク
//! 間に NAT が無い（loopback か信頼できる同一ネットワーク。記録 §1・§7 の範囲）こと。
//!
//! 手順は記録 §3 のとおり（手順 2 の順序は §13 で「接続 → 待受 → Hello」に改めた）。形式不正・
//! magic／版／セッション違い・無応答の接続は閉じて受付を続け（期限は延ばさない）、範囲外・
//! 重複ランクと world_size 不一致は設定誤りとして確立を失敗させる。確立後は集合点とリング用の
//! listener を閉じる（以降の接続は拒否される）。認証・暗号化は無い。セッション識別子は
//! 混線防止用で認証ではない。

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use super::config::{RendezvousConfig, check_peer_allowed};
use super::error::{DistributedError, ProtocolViolation, Result};
use super::frame::{
    Hello, RankTable, read_header, read_hello, read_rank_table, write_hello, write_rank_table,
};
use super::transport::{DeadlineStream, accept_by, bind_nonblocking, connect_with_retry};

/// 確立済みのリング接続。`world_size == 1` のときは両方 `None`（ソケットを開かない）。
#[derive(Debug)]
pub struct RingLinks {
    /// 自ランク。
    pub rank: usize,
    /// world_size。
    pub world_size: usize,
    /// 次のランク `(r+1)%W` への接続（送信用）。
    pub to_next: Option<TcpStream>,
    /// 前のランク `(r-1)%W` からの接続（受信用）。
    pub from_prev: Option<TcpStream>,
}

/// bind 済みでまだ集合していない状態。`bind` → `establish` の 2 段にするのは、ポート 0 で
/// bind した集合点の実アドレスをピア待ちの前に取得できるようにするため（記録 §3 手順 1。
/// TOCTOU 回避）。ランク 0 以外のリング待受は集合点への接続 IP から導くため `establish` で開く。
#[derive(Debug)]
pub struct Rendezvous {
    cfg: RendezvousConfig,
    /// ランク 0 だけが持つ集合点とリング用の待受（リングは集合点と同じ IP）。
    rank0: Option<(TcpListener, TcpListener)>,
}

/// IPv4-mapped を IPv4 へ寄せた IP（設定時の正規化と観測側を揃える）。
fn canonical_ip(ip: IpAddr) -> IpAddr {
    ip.to_canonical()
}

impl Rendezvous {
    /// 設定を検証し、ランク 0 は集合点と（同じ IP の）リング用待受を開く。ランク 0 以外と
    /// `world_size == 1` はここではソケットを開かない。
    pub fn bind(cfg: RendezvousConfig) -> Result<Self> {
        cfg.validate()?;
        if cfg.world_size() == 1 || cfg.rank() != 0 {
            return Ok(Self { cfg, rank0: None });
        }
        let rendezvous = bind_nonblocking(cfg.rendezvous_addr())?;
        // ピアが到達するのは集合点の待受 IP なので、リングもその IP に bind する
        // （`SocketAddr` を引き継ぎ、ポートだけ 0 にする）。
        let mut ring_addr = rendezvous.local_addr()?;
        ring_addr.set_port(0);
        let ring = bind_nonblocking(ring_addr)?;
        Ok(Self {
            cfg,
            rank0: Some((rendezvous, ring)),
        })
    }

    /// 集合点の実際の待受アドレス（ランク 0 のみ `Some`）。
    pub fn rendezvous_local_addr(&self) -> Option<SocketAddr> {
        self.rank0.as_ref().and_then(|(l, _)| l.local_addr().ok())
    }

    /// 集合を確立する。全体を `establish_timeout` の期限 1 つで包む。
    /// 失敗時は保持していた全接続を drop して閉じる（相手は切断を検出して早く終了する）。
    pub fn establish(self) -> Result<RingLinks> {
        let cfg = &self.cfg;
        let (w, rank) = (cfg.world_size(), cfg.rank());
        if w == 1 {
            return Ok(RingLinks {
                rank,
                world_size: w,
                to_next: None,
                from_prev: None,
            });
        }
        // Duration::MAX 等で Instant を表現できない設定は panic させず型付きエラーにする。
        let deadline = Instant::now()
            .checked_add(cfg.establish_timeout_value())
            .ok_or(DistributedError::InvalidConfig {
                reason: "establish_timeout is too large",
            })?;
        match self.rank0.as_ref() {
            Some((rv, ring)) => {
                let ring_port = ring.local_addr()?.port();
                let table = collect_as_rank0(cfg, rv, ring, deadline)?;
                connect_ring(cfg, ring, &table, ring_port, deadline)
            }
            None => {
                let (table, ring) = fetch_table(cfg, deadline)?;
                let ring_port = ring.local_addr()?.port();
                connect_ring(cfg, &ring, &table, ring_port, deadline)
            }
        }
    }
}

fn my_hello(cfg: &RendezvousConfig, ring_port: u16) -> Hello {
    Hello {
        world_size: cfg.world_size() as u32,
        rank: cfg.rank() as u32,
        session_id: cfg.session(),
        ring_port,
    }
}

/// Hello を 1 通読む。期限は `min(io_timeout, 全体期限)`（何も送らない接続に足止めされない）。
fn read_hello_by(s: &TcpStream, deadline: Instant, io_timeout: Duration) -> Result<Hello> {
    // io_timeout が Instant に加算できない巨大値なら全体期限だけで打ち切る（panic しない）。
    let own = Instant::now()
        .checked_add(io_timeout)
        .map_or(deadline, |t| deadline.min(t));
    let mut ds = DeadlineStream::new(s, own, io_timeout);
    let h = read_header(&mut ds)?;
    read_hello(&mut ds, &h)
}

/// ランク 0: W−1 本の正当な Hello を集め、ランク表を全ランクへ送って集合点を閉じる。
///
/// 各ランクのエントリは「観測した接続元 IP（正規化）＋申告ポート」。ピアはその接続の
/// `local_addr` IP にリングを bind 済みなので、配布アドレスには必ず待受がある（モジュール冒頭）。
fn collect_as_rank0(
    cfg: &RendezvousConfig,
    listener: &TcpListener,
    ring: &TcpListener,
    deadline: Instant,
) -> Result<RankTable> {
    let w = cfg.world_size();
    let io = cfg.io_timeout_value();
    let allow = cfg.is_non_loopback_allowed();
    let mut peers: Vec<Option<(TcpStream, SocketAddr)>> = (0..w).map(|_| None).collect();
    let mut got = 0usize;
    while got < w - 1 {
        let (s, peer) = accept_by(listener, deadline, "rendezvous accept")?;
        if check_peer_allowed(peer, allow).is_err() {
            continue;
        }
        // 形式不正・無応答・magic／版違いはその接続だけ閉じて続ける。
        let Ok(hello) = read_hello_by(&s, deadline, io) else {
            continue;
        };
        if hello.session_id != cfg.session() || hello.ring_port == 0 {
            continue;
        }
        let hello_world = hello.world_size as usize;
        if hello_world != w {
            return Err(DistributedError::WorldSizeMismatch {
                expected: w,
                got: hello_world,
            });
        }
        let rank = hello.rank as usize;
        if rank >= w {
            return Err(DistributedError::RankOutOfRange {
                rank,
                world_size: w,
            });
        }
        if rank == 0 || peers[rank].is_some() {
            return Err(DistributedError::DuplicateRank { rank });
        }
        // 観測 IP をそのまま使う（loopback への読み替え等をすると、ピアの待受 IP とずれる）。
        let ip = canonical_ip(peer.ip());
        peers[rank] = Some((s, SocketAddr::new(ip, hello.ring_port)));
        got += 1;
    }
    // 自分のエントリはリング待受そのもの（集合点と同じ IP。`Rendezvous::bind`）。
    let own = ring.local_addr()?;
    let mut entries = Vec::with_capacity(w);
    entries.push(SocketAddr::new(canonical_ip(own.ip()), own.port()));
    for p in peers.iter().skip(1) {
        match p {
            Some((_, a)) => entries.push(*a),
            None => return Err(DistributedError::DuplicateRank { rank: 0 }),
        }
    }
    let table = RankTable { entries };
    for (s, _) in peers.iter().skip(1).flatten() {
        let mut ds = DeadlineStream::new(s, deadline, io);
        write_rank_table(&mut ds, &table)?;
    }
    Ok(table)
}

/// ランク 0 以外: 集合点へ接続し、その接続の `local_addr` IP にリング待受を bind してから
/// Hello を送り、ランク表を受けて検証する。リング待受を返す（`connect_ring` が accept に使う）。
fn fetch_table(cfg: &RendezvousConfig, deadline: Instant) -> Result<(RankTable, TcpListener)> {
    let (w, rank) = (cfg.world_size(), cfg.rank());
    let allow = cfg.is_non_loopback_allowed();
    let s = connect_with_retry(cfg.rendezvous_addr(), cfg.connect_timeout_value(), deadline)?;
    // 集合点が `peer_addr` として観測する IP と同一（NAT が無い前提）。
    let local = s.local_addr()?;
    let ring_ip = canonical_ip(local.ip());
    if ring_ip.is_unspecified() {
        return Err(DistributedError::InvalidConfig {
            reason: "rendezvous connection has an unspecified local address",
        });
    }
    check_peer_allowed(SocketAddr::new(ring_ip, 0), allow)?;
    // 正規化で族が変わらないときは `SocketAddr` を引き継ぐ（IPv6 のスコープ等を落とさない）。
    let ring_bind = if ring_ip == local.ip() {
        let mut a = local;
        a.set_port(0);
        a
    } else {
        SocketAddr::new(ring_ip, 0)
    };
    let ring = bind_nonblocking(ring_bind)?;
    let ring_port = ring.local_addr()?.port();
    let io = cfg.io_timeout_value();
    write_hello(
        &mut DeadlineStream::new(&s, deadline, io),
        &my_hello(cfg, ring_port),
    )?;
    // 全ランクが揃うまで待つため、1 回あたりの上限は確立の期限そのもの（全体期限は有効）。
    let mut ds = DeadlineStream::new(&s, deadline, cfg.establish_timeout_value());
    let h = read_header(&mut ds)?;
    let table = read_rank_table(&mut ds, &h, w)?;
    validate_table(cfg, &table, rank, SocketAddr::new(ring_ip, ring_port))?;
    Ok((table, ring))
}

/// 受け取ったランク表を接続前に検証する。opt-in の迂回（非 loopback エントリ）・unspecified・
/// 重複エントリを拒否し、自分で確かめられる 2 エントリ（自ランク＝自分のリング待受、
/// ランク 0＝設定した集合点の IP）が実際の待受と一致することを確かめる。
fn validate_table(
    cfg: &RendezvousConfig,
    table: &RankTable,
    rank: usize,
    own_ring: SocketAddr,
) -> Result<()> {
    let mut seen = HashSet::with_capacity(table.entries.len());
    for a in &table.entries {
        check_peer_allowed(*a, cfg.is_non_loopback_allowed())?;
        let canon = SocketAddr::new(canonical_ip(a.ip()), a.port());
        if canon.ip().is_unspecified() || !seen.insert(canon) {
            return Err(ProtocolViolation::MalformedRankTable.into());
        }
    }
    let entry = |r: usize| -> Result<SocketAddr> {
        table
            .entries
            .get(r)
            .map(|a| SocketAddr::new(canonical_ip(a.ip()), a.port()))
            .ok_or_else(|| ProtocolViolation::MalformedRankTable.into())
    };
    if entry(rank)? != own_ring {
        return Err(ProtocolViolation::RankTableAddressMismatch { rank }.into());
    }
    if entry(0)?.ip() != canonical_ip(cfg.rendezvous_addr().ip()) {
        return Err(ProtocolViolation::RankTableAddressMismatch { rank: 0 }.into());
    }
    Ok(())
}

/// 全ランク共通: 次のランクへ接続して Hello を交わし、前のランクからの接続を受ける。
fn connect_ring(
    cfg: &RendezvousConfig,
    ring: &TcpListener,
    table: &RankTable,
    ring_port: u16,
    deadline: Instant,
) -> Result<RingLinks> {
    let (w, rank) = (cfg.world_size(), cfg.rank());
    let (next, prev) = ((rank + 1) % w, (rank + w - 1) % w);
    let io = cfg.io_timeout_value();
    let allow = cfg.is_non_loopback_allowed();
    let next_addr = *table
        .entries
        .get(next)
        .ok_or(DistributedError::RankOutOfRange {
            rank: next,
            world_size: w,
        })?;
    check_peer_allowed(next_addr, allow)?;
    let to_next = connect_with_retry(next_addr, cfg.connect_timeout_value(), deadline)?;
    let hello = my_hello(cfg, ring_port);
    write_hello(&mut DeadlineStream::new(&to_next, deadline, io), &hello)?;
    // 小さな Hello は送信バッファに収まるため、全ランクが connect と送信を済ませてから accept
    // してもデッドロックしない。W=2 でも接続は別々の 2 本で、相手は accept 順ではなく Hello で識別する。
    let from_prev = loop {
        let (s, peer) = accept_by(ring, deadline, "ring accept")?;
        if check_peer_allowed(peer, allow).is_err() {
            continue;
        }
        let Ok(h) = read_hello_by(&s, deadline, io) else {
            continue;
        };
        if h.session_id != cfg.session() {
            continue;
        }
        if h.world_size as usize != w {
            return Err(DistributedError::WorldSizeMismatch {
                expected: w,
                got: h.world_size as usize,
            });
        }
        if h.rank as usize != prev {
            return Err(DistributedError::UnexpectedRank {
                expected: prev,
                got: h.rank as usize,
            });
        }
        write_hello(&mut DeadlineStream::new(&s, deadline, io), &hello)?;
        break s;
    };
    let reply = read_hello_by(&to_next, deadline, io)?;
    if reply.session_id != cfg.session() {
        return Err(DistributedError::InvalidConfig {
            reason: "ring peer session id mismatch",
        });
    }
    if reply.world_size as usize != w {
        return Err(DistributedError::WorldSizeMismatch {
            expected: w,
            got: reply.world_size as usize,
        });
    }
    if reply.rank as usize != next {
        return Err(DistributedError::UnexpectedRank {
            expected: next,
            got: reply.rank as usize,
        });
    }
    Ok(RingLinks {
        rank,
        world_size: w,
        to_next: Some(to_next),
        from_prev: Some(from_prev),
    })
}

#[cfg(test)]
mod tests {
    use super::super::frame::{
        FrameKind, SegmentExpect, SegmentHeader, recv_segment_into, send_segment,
    };
    use super::*;
    use std::io::Write;
    use std::net::Shutdown;
    use std::thread;

    fn cfg(w: usize, rank: usize, addr: SocketAddr) -> RendezvousConfig {
        RendezvousConfig::new(w, rank, addr)
            .unwrap()
            .session_id(42)
            .establish_timeout(Duration::from_secs(20))
            .io_timeout(Duration::from_secs(5))
    }

    fn lo0() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    fn fake_hello(world: u32, rank: u32, session: u64, port: u16) -> Hello {
        Hello {
            world_size: world,
            rank,
            session_id: session,
            ring_port: port,
        }
    }

    /// ランク 0 を bind して集合点アドレスを得て、ランク 0 を別スレッドで establish する。
    fn spawn_rank0<'s>(
        scope: &'s thread::Scope<'s, '_>,
        c: RendezvousConfig,
    ) -> (SocketAddr, thread::ScopedJoinHandle<'s, Result<RingLinks>>) {
        let r0 = Rendezvous::bind(c).unwrap();
        let addr = r0.rendezvous_local_addr().unwrap();
        (addr, scope.spawn(move || r0.establish()))
    }

    /// ランク 0 を `rank0_addr` で bind し、ランク 1.. は実際の集合点アドレスを `peer_addr` で
    /// 写したアドレス（mapped 表記等）へ接続させて、全ランクのリングを確立する。
    fn establish_group_with(
        w: usize,
        rank0_addr: SocketAddr,
        allow: bool,
        peer_addr: impl Fn(SocketAddr) -> SocketAddr + Sync,
    ) -> Vec<RingLinks> {
        thread::scope(|s| {
            let (addr, h0) = spawn_rank0(s, cfg(w, 0, rank0_addr).allow_non_loopback(allow));
            let target = peer_addr(addr);
            let hs: Vec<_> = (1..w)
                .map(|r| {
                    s.spawn(move || {
                        Rendezvous::bind(cfg(w, r, target).allow_non_loopback(allow))?.establish()
                    })
                })
                .collect();
            let mut v = vec![h0.join().unwrap().unwrap()];
            for h in hs {
                v.push(h.join().unwrap().unwrap());
            }
            v
        })
    }

    fn establish_group(w: usize) -> Vec<RingLinks> {
        establish_group_with(w, lo0(), false, |a| a)
    }

    /// 確立したリングの正のプローブ: ランクと world_size、配布アドレスと待受の一致
    /// （ランク r の `to_next` の接続先 ＝ ランク r+1 の `from_prev` を受けた待受）、
    /// 各ランクが隣接へフレームを送り、前のランクのフレームを bit 一致で受けられること。
    fn assert_ring_round_trip(links: &[RingLinks]) {
        let w = links.len();
        for (r, l) in links.iter().enumerate() {
            assert_eq!((l.rank, l.world_size), (r, w));
            let dst = l.to_next.as_ref().unwrap().peer_addr().unwrap();
            let next_listen = links[(r + 1) % w]
                .from_prev
                .as_ref()
                .unwrap()
                .local_addr()
                .unwrap();
            assert_eq!(dst, next_listen, "rank {r} -> next");
        }
        // 各ランクが to_next へ送り、from_prev から受ける（送信は小さく、バッファに収まる）。
        for (r, l) in links.iter().enumerate() {
            let data = [r as f32, -0.0, f32::NAN, 1.25 + r as f32];
            let hdr = SegmentHeader {
                step_seq: 1,
                total_elems: 8,
                chunk: r as u32,
                seg_offset: 0,
            };
            let mut tx = l.to_next.as_ref().unwrap();
            send_segment(&mut tx, FrameKind::ReduceScatterSegment, &hdr, &data).unwrap();
        }
        for (r, l) in links.iter().enumerate() {
            let prev = (r + w - 1) % w;
            let rx = l.from_prev.as_ref().unwrap();
            rx.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut rx = rx;
            let mut dst = [0.0f32; 4];
            recv_segment_into(&mut rx, &expect(prev as u32), &mut dst).unwrap();
            let want = [prev as f32, -0.0, f32::NAN, 1.25 + prev as f32];
            for (a, b) in want.iter().zip(dst.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    fn expect(chunk: u32) -> SegmentExpect {
        SegmentExpect {
            kind: FrameKind::ReduceScatterSegment,
            step_seq: 1,
            total_elems: 8,
            chunk,
            seg_offset: 0,
        }
    }

    fn to_mapped(a: SocketAddr) -> SocketAddr {
        match a.ip() {
            IpAddr::V4(v4) => SocketAddr::new(IpAddr::V6(v4.to_ipv6_mapped()), a.port()),
            IpAddr::V6(_) => a,
        }
    }

    #[test]
    fn ring_round_trip_for_several_world_sizes() {
        for w in [2usize, 3, 4] {
            assert_ring_round_trip(&establish_group(w));
        }
    }

    #[test]
    fn ring_established_on_ipv4_mapped_rendezvous() {
        // ランク 0 も mapped 表記で起動し、ピアも mapped 表記の集合点へ接続する（指摘 7）。
        let mapped0: SocketAddr = "[::ffff:127.0.0.1]:0".parse().unwrap();
        for w in [2usize, 3] {
            assert_ring_round_trip(&establish_group_with(w, mapped0, false, to_mapped));
        }
    }

    #[test]
    fn ring_established_on_ipv6_loopback_rendezvous() {
        // 指摘 3・4: 全ランクが [::1] を集合点に指定してもリングが確立すること。
        let v6: SocketAddr = "[::1]:0".parse().unwrap();
        match TcpListener::bind(v6) {
            Ok(probe) => {
                drop(probe);
                for w in [2usize, 3] {
                    assert_ring_round_trip(&establish_group_with(w, v6, false, |a| a));
                }
            }
            Err(e) => {
                // IPv6 が無効化された環境（一部のコンテナ・CI）では ::1 に bind できず、リングの
                // 到達性は検証できない。代わりに、その環境で集合点の bind が panic せず型付きの
                // I/O エラーで失敗することを確かめる（黙ってスキップしない）。
                eprintln!(
                    "IPv6 loopback unavailable ({:?}); checking the typed error",
                    e.kind()
                );
                assert!(matches!(
                    Rendezvous::bind(cfg(2, 0, v6)),
                    Err(DistributedError::Io(_))
                ));
            }
        }
    }

    /// Linux は 127.0.0.0/8 全体が lo に載るため 127.0.0.2 で待受できる。macOS は lo0 に
    /// 127.0.0.1 しか載らない（別名の追加には管理者操作が要る）ため Linux に限る。
    #[cfg(target_os = "linux")]
    #[test]
    fn ring_established_on_non_canonical_ipv4_loopback_rendezvous() {
        // 指摘 5・6: 集合点 127.0.0.2 への接続元は通常 127.0.0.1 になる。ピアのリング待受は
        // その接続元 IP に、ランク 0 のリング待受は 127.0.0.2 に bind されて一致する。
        let a: SocketAddr = "127.0.0.2:0".parse().unwrap();
        for w in [2usize, 3] {
            let links = establish_group_with(w, a, false, |x| x);
            assert_eq!(
                links[0]
                    .from_prev
                    .as_ref()
                    .unwrap()
                    .local_addr()
                    .unwrap()
                    .ip(),
                a.ip(),
                "rank 0 listens for the ring on its rendezvous IP"
            );
            assert_ring_round_trip(&links);
        }
    }

    /// このホストの非 loopback ユニキャスト IP を、パケットを送らずに得る（UDP の connect は
    /// 経路表を引くだけで送信しない）。経路やインターフェースが無い環境では `None`。
    fn local_non_loopback_ip() -> Option<IpAddr> {
        let u = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
        u.connect("192.0.2.1:9").ok()?;
        let ip = u.local_addr().ok()?.ip();
        if ip.is_loopback() || ip.is_unspecified() {
            return None;
        }
        TcpListener::bind(SocketAddr::new(ip, 0)).ok()?;
        Some(ip)
    }

    #[test]
    fn non_loopback_rendezvous_needs_opt_in_and_then_reaches_the_ring() {
        // opt-in なしは常にソケット作成前に拒否される（TEST-NET の固定アドレスで検査）。
        let ext: SocketAddr = "192.0.2.1:0".parse().unwrap();
        assert!(matches!(
            Rendezvous::bind(cfg(2, 0, ext)),
            Err(DistributedError::NonLoopbackNotAllowed { .. })
        ));
        assert!(matches!(
            Rendezvous::bind(cfg(2, 1, "192.0.2.1:7".parse().unwrap())),
            Err(DistributedError::NonLoopbackNotAllowed { .. })
        ));
        match local_non_loopback_ip() {
            Some(ip) => {
                // 指摘 8: 非 loopback 集合点でも opt-in だけでリングが確立し、配布した LAN IP に
                // 待受がある。同一ホスト内の接続で、外部へは出ない。
                let a = SocketAddr::new(ip, 0);
                assert!(matches!(
                    Rendezvous::bind(cfg(2, 0, a)),
                    Err(DistributedError::NonLoopbackNotAllowed { .. })
                ));
                for w in [2usize, 3] {
                    assert_ring_round_trip(&establish_group_with(w, a, true, |x| x));
                }
            }
            None => {
                // 非 loopback の IP を持たない CI・サンドボックスでは、到達性を検証する宛先が
                // 無い（外部ホストへの通信は CI の前提〈外部通信なし〉に反する）。設定段の判定
                // だけを確かめる: opt-in があれば検証を通り、リング待受の追加設定は要らない。
                eprintln!("no local non-loopback IP; reachability is not checked");
                assert!(
                    RendezvousConfig::new(2, 0, "192.0.2.1:7000".parse().unwrap())
                        .unwrap()
                        .allow_non_loopback(true)
                        .validate()
                        .is_ok()
                );
            }
        }
    }

    #[test]
    fn huge_timeouts_are_invalid_config_not_panic() {
        let c = cfg(2, 0, lo0()).establish_timeout(Duration::MAX);
        assert!(matches!(
            Rendezvous::bind(c),
            Err(DistributedError::InvalidConfig { .. })
        ));
        let c = cfg(2, 0, lo0()).io_timeout(Duration::MAX);
        assert!(matches!(
            Rendezvous::bind(c),
            Err(DistributedError::InvalidConfig { .. })
        ));
        let c = cfg(2, 0, lo0()).connect_timeout(Duration::MAX);
        assert!(matches!(
            Rendezvous::bind(c),
            Err(DistributedError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn world_size_one_opens_no_sockets() {
        let r = Rendezvous::bind(cfg(1, 0, lo0())).unwrap();
        assert!(r.rendezvous_local_addr().is_none());
        let l = r.establish().unwrap();
        assert!(l.to_next.is_none() && l.from_prev.is_none());
    }

    #[test]
    fn non_loopback_bind_is_rejected_before_opening() {
        let c = RendezvousConfig::new(2, 0, "192.0.2.1:9".parse().unwrap()).unwrap();
        assert!(matches!(
            Rendezvous::bind(c),
            Err(DistributedError::NonLoopbackNotAllowed { .. })
        ));
    }

    fn rank0_with_fakes(w: usize, fakes: &[Hello], c0_timeout: Duration) -> Result<RingLinks> {
        thread::scope(|s| {
            let c = cfg(w, 0, lo0()).establish_timeout(c0_timeout);
            let (addr, h0) = spawn_rank0(s, c);
            let mut keep = Vec::new();
            for h in fakes {
                let mut st = TcpStream::connect(addr).unwrap();
                write_hello(&mut st, h).unwrap();
                keep.push(st);
            }
            let r = h0.join().unwrap();
            drop(keep);
            r
        })
    }

    #[test]
    fn out_of_range_rank_is_rejected() {
        let r = rank0_with_fakes(2, &[fake_hello(2, 2, 42, 5000)], Duration::from_secs(20));
        assert!(matches!(
            r,
            Err(DistributedError::RankOutOfRange {
                rank: 2,
                world_size: 2
            })
        ));
    }

    #[test]
    fn duplicate_ranks_are_rejected() {
        let r = rank0_with_fakes(
            3,
            &[fake_hello(3, 1, 42, 5000), fake_hello(3, 1, 42, 5001)],
            Duration::from_secs(20),
        );
        assert!(matches!(
            r,
            Err(DistributedError::DuplicateRank { rank: 1 })
        ));
        let r = rank0_with_fakes(2, &[fake_hello(2, 0, 42, 5000)], Duration::from_secs(20));
        assert!(matches!(
            r,
            Err(DistributedError::DuplicateRank { rank: 0 })
        ));
    }

    #[test]
    fn world_size_mismatch_is_rejected() {
        let r = rank0_with_fakes(2, &[fake_hello(5, 1, 42, 5000)], Duration::from_secs(20));
        assert!(matches!(
            r,
            Err(DistributedError::WorldSizeMismatch {
                expected: 2,
                got: 5
            })
        ));
    }

    #[test]
    fn bad_connections_are_closed_and_accepting_continues() {
        thread::scope(|s| {
            let (addr, h0) = spawn_rank0(s, cfg(2, 0, lo0()));
            // magic 違い（ヘッダ長以上のゴミ）と、セッション違いの Hello を先に送る。
            let mut junk = TcpStream::connect(addr).unwrap();
            junk.write_all(&[0xff; 40]).unwrap();
            let mut other = TcpStream::connect(addr).unwrap();
            write_hello(&mut other, &fake_hello(2, 1, 999, 5000)).unwrap();
            let h1 = s.spawn(move || Rendezvous::bind(cfg(2, 1, addr))?.establish());
            let l0 = h0.join().unwrap().unwrap();
            let l1 = h1.join().unwrap().unwrap();
            assert!(l0.to_next.is_some() && l1.from_prev.is_some());
        });
    }

    #[test]
    fn unexpected_ring_peer_rank_is_rejected() {
        thread::scope(|s| {
            let (addr, h0) = spawn_rank0(s, cfg(2, 0, lo0()));
            // ランク 1 を名乗る偽ピア: 集合点で Hello を送ってランク表を受け、リングでは別ランクを名乗る。
            let fake_ring = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut st = TcpStream::connect(addr).unwrap();
            let port = fake_ring.local_addr().unwrap().port();
            write_hello(&mut st, &fake_hello(2, 1, 42, port)).unwrap();
            let h = read_header(&mut st).unwrap();
            let table = read_rank_table(&mut st, &h, 2).unwrap();
            let mut ring = TcpStream::connect(table.entries[0]).unwrap();
            write_hello(&mut ring, &fake_hello(2, 0, 42, port)).unwrap();
            let r = h0.join().unwrap();
            assert!(matches!(
                r,
                Err(DistributedError::UnexpectedRank {
                    expected: 1,
                    got: 0
                })
            ));
        });
    }

    #[test]
    fn non_loopback_rank_table_entry_is_rejected() {
        thread::scope(|s| {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            let h1 = s.spawn(move || Rendezvous::bind(cfg(2, 1, addr))?.establish());
            let (mut st, _) = l.accept().unwrap();
            let h = read_header(&mut st).unwrap();
            read_hello(&mut st, &h).unwrap();
            let t = RankTable {
                entries: vec![
                    "127.0.0.1:1001".parse().unwrap(),
                    "192.0.2.5:1002".parse().unwrap(),
                ],
            };
            write_rank_table(&mut st, &t).unwrap();
            assert!(matches!(
                h1.join().unwrap(),
                Err(DistributedError::NonLoopbackNotAllowed { .. })
            ));
        });
    }

    /// 偽の集合点でランク 1 の Hello を受け、`make` で作ったランク表を返して結果を得る。
    /// `make` には観測した接続元 IP とランク 1 が申告したリング待受ポートを渡す。
    fn rank1_with_fake_table(
        make: impl FnOnce(IpAddr, u16) -> Vec<SocketAddr> + Send,
    ) -> Result<RingLinks> {
        rank1_with_fake_table_opt(false, make)
    }

    fn rank1_with_fake_table_opt(
        allow: bool,
        make: impl FnOnce(IpAddr, u16) -> Vec<SocketAddr> + Send,
    ) -> Result<RingLinks> {
        thread::scope(|s| {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            let h1 = s.spawn(move || {
                Rendezvous::bind(cfg(2, 1, addr).allow_non_loopback(allow))?.establish()
            });
            let (mut st, peer) = l.accept().unwrap();
            let h = read_header(&mut st).unwrap();
            let hello = read_hello(&mut st, &h).unwrap();
            let t = RankTable {
                entries: make(peer.ip(), hello.ring_port),
            };
            write_rank_table(&mut st, &t).unwrap();
            h1.join().unwrap()
        })
    }

    #[test]
    fn rank_table_own_entry_mismatch_is_rejected_before_connecting() {
        // 自分のエントリが自分のリング待受と違う（観測 IP と待受 IP のずれ・別ポート）なら、
        // 配布先へ connect してタイムアウトする前に型付きエラーで止まる。
        let r = rank1_with_fake_table(|ip, port| {
            let other = if port == u16::MAX { port - 1 } else { port + 1 };
            vec![
                "127.0.0.1:1001".parse().unwrap(),
                SocketAddr::new(ip, other),
            ]
        });
        assert!(matches!(
            r,
            Err(DistributedError::Protocol(
                ProtocolViolation::RankTableAddressMismatch { rank: 1 }
            ))
        ));
        let r = rank1_with_fake_table(|_, port| {
            vec![
                "127.0.0.1:1001".parse().unwrap(),
                SocketAddr::new("127.0.0.9".parse().unwrap(), port),
            ]
        });
        assert!(matches!(
            r,
            Err(DistributedError::Protocol(
                ProtocolViolation::RankTableAddressMismatch { rank: 1 }
            ))
        ));
    }

    #[test]
    fn rank_table_rank0_entry_off_the_rendezvous_ip_is_rejected() {
        let r = rank1_with_fake_table(|ip, port| {
            vec!["127.0.0.3:1001".parse().unwrap(), SocketAddr::new(ip, port)]
        });
        assert!(matches!(
            r,
            Err(DistributedError::Protocol(
                ProtocolViolation::RankTableAddressMismatch { rank: 0 }
            ))
        ));
    }

    #[test]
    fn rank_table_duplicate_or_unspecified_entries_are_rejected() {
        let r = rank1_with_fake_table(|ip, port| {
            vec![SocketAddr::new(ip, port), SocketAddr::new(ip, port)]
        });
        assert!(matches!(
            r,
            Err(DistributedError::Protocol(
                ProtocolViolation::MalformedRankTable
            ))
        ));
        // unspecified は opt-in なしなら非 loopback として、opt-in ありでも形式不正として拒否する。
        let unspec = |ip, port| vec!["0.0.0.0:1001".parse().unwrap(), SocketAddr::new(ip, port)];
        assert!(matches!(
            rank1_with_fake_table(unspec),
            Err(DistributedError::NonLoopbackNotAllowed { .. })
        ));
        assert!(matches!(
            rank1_with_fake_table_opt(true, unspec),
            Err(DistributedError::Protocol(
                ProtocolViolation::MalformedRankTable
            ))
        ));
    }

    #[test]
    fn rendezvous_connection_flood_does_not_extend_the_deadline() {
        // 指摘 1: 不正な接続が途切れず届き続けても、ランク 0 は確立の期限で終わる。
        let stop = std::sync::atomic::AtomicBool::new(false);
        let t = Instant::now();
        thread::scope(|s| {
            let c = cfg(2, 0, lo0())
                .establish_timeout(Duration::from_millis(800))
                .io_timeout(Duration::from_millis(200));
            let (addr, h0) = spawn_rank0(s, c);
            let stop = &stop;
            s.spawn(move || {
                let mut keep = Vec::new();
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Ok(mut c) = TcpStream::connect(addr) {
                        let _ = c.write_all(&[0xff; 40]);
                        keep.push(c);
                    } else {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    if keep.len() > 256 {
                        keep.clear();
                    }
                }
            });
            let r = h0.join().unwrap();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            assert!(matches!(r, Err(DistributedError::Timeout { .. })), "{r:?}");
        });
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn silent_connection_does_not_extend_the_deadline() {
        let t = Instant::now();
        thread::scope(|s| {
            let c = cfg(2, 0, lo0())
                .establish_timeout(Duration::from_millis(800))
                .io_timeout(Duration::from_millis(300));
            let (addr, h0) = spawn_rank0(s, c);
            let _silent = TcpStream::connect(addr).unwrap();
            assert!(matches!(
                h0.join().unwrap(),
                Err(DistributedError::Timeout { .. })
            ));
        });
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn missing_rank_times_out_on_rank0() {
        thread::scope(|s| {
            let c = cfg(3, 0, lo0()).establish_timeout(Duration::from_millis(500));
            let (addr, h0) = spawn_rank0(s, c);
            let h1 = s.spawn(move || {
                Rendezvous::bind(cfg(3, 1, addr).establish_timeout(Duration::from_millis(500)))?
                    .establish()
            });
            assert!(matches!(
                h0.join().unwrap(),
                Err(DistributedError::Timeout { .. })
            ));
            assert!(matches!(
                h1.join().unwrap(),
                Err(DistributedError::Timeout { .. } | DistributedError::PeerDisconnected)
            ));
        });
    }

    #[test]
    fn non_accepting_rendezvous_times_out_on_peer() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let c = cfg(2, 1, addr).establish_timeout(Duration::from_millis(500));
        let r = Rendezvous::bind(c).unwrap().establish();
        assert!(matches!(r, Err(DistributedError::Timeout { .. })));
        drop(l);
    }

    #[test]
    fn truncated_hello_on_socket_is_peer_disconnected() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        let mut bytes = Vec::new();
        write_hello(&mut bytes, &fake_hello(2, 1, 42, 5000)).unwrap();
        c.write_all(&bytes[..7]).unwrap();
        c.shutdown(Shutdown::Write).unwrap();
        let r = read_hello_by(
            &s,
            Instant::now() + Duration::from_secs(5),
            Duration::from_secs(5),
        );
        assert!(matches!(r, Err(DistributedError::PeerDisconnected)));
    }
}
