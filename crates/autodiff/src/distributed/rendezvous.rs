//! ランク発見（ランク 0 の集合点）とリング接続の確立。
//!
//! 役割: 起動情報（`RendezvousConfig`）から、各ランクが次のランク `(r+1)%W` へ送る接続
//! （`to_next`）と前のランク `(r-1)%W` から受ける接続（`from_prev`）を確立して [`RingLinks`]
//! を返す。all-reduce 本体（#3145）はこの `RingLinks` と `frame.rs` の送受信 API を使う。
//! 根拠: `docs/facade-distributed-host-collective-decision.md` §3・§5・§7（spec `:386-393`）。
//!
//! 手順は記録 §3 のとおり。形式不正・magic／版／セッション違い・無応答の接続は閉じて受付を続け
//! （期限は延ばさない）、範囲外・重複ランクと world_size 不一致は設定誤りとして確立を失敗させる。
//! 確立後は集合点とリング用の listener を閉じる（以降の接続は拒否される）。
//! 認証・暗号化は無い。セッション識別子は混線防止用で認証ではない。

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use super::config::{RendezvousConfig, check_peer_allowed};
use super::error::{DistributedError, Result};
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
/// bind した実アドレスをピア待ちの前に取得できるようにするため（記録 §3 手順 1。TOCTOU 回避）。
#[derive(Debug)]
pub struct Rendezvous {
    cfg: RendezvousConfig,
    rendezvous: Option<TcpListener>,
    ring: Option<TcpListener>,
}

impl Rendezvous {
    /// 設定を検証し、待受を開く。ランク 0 は集合点とリング用、それ以外はリング用だけ。
    /// `world_size == 1` はソケットを一切開かない。
    pub fn bind(cfg: RendezvousConfig) -> Result<Self> {
        cfg.validate()?;
        if cfg.world_size() == 1 {
            return Ok(Self {
                cfg,
                rendezvous: None,
                ring: None,
            });
        }
        let rendezvous = if cfg.rank() == 0 {
            Some(bind_nonblocking(cfg.rendezvous_addr())?)
        } else {
            None
        };
        let ring = bind_nonblocking(SocketAddr::new(cfg.ring_bind_ip_value(), 0))?;
        Ok(Self {
            cfg,
            rendezvous,
            ring: Some(ring),
        })
    }

    /// 集合点の実際の待受アドレス（ランク 0 のみ `Some`）。
    pub fn rendezvous_local_addr(&self) -> Option<SocketAddr> {
        self.rendezvous.as_ref().and_then(|l| l.local_addr().ok())
    }

    /// 集合を確立する。全体を `establish_timeout` の期限 1 つで包む。
    /// 失敗時は保持していた全接続を drop して閉じる（相手は切断を検出して早く終了する）。
    pub fn establish(self) -> Result<RingLinks> {
        let cfg = &self.cfg;
        let (w, rank) = (cfg.world_size(), cfg.rank());
        let Some(ring) = self.ring.as_ref() else {
            return Ok(RingLinks {
                rank,
                world_size: w,
                to_next: None,
                from_prev: None,
            });
        };
        let deadline = Instant::now() + cfg.establish_timeout_value();
        let ring_port = ring.local_addr()?.port();
        let table = match self.rendezvous.as_ref() {
            Some(l) => collect_as_rank0(cfg, l, ring, ring_port, deadline)?,
            None => fetch_table(cfg, ring_port, deadline)?,
        };
        connect_ring(cfg, ring, &table, ring_port, deadline)
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
    let own = deadline.min(Instant::now() + io_timeout);
    let mut ds = DeadlineStream::new(s, own, io_timeout);
    let h = read_header(&mut ds)?;
    read_hello(&mut ds, &h)
}

/// ランク 0: W−1 本の正当な Hello を集め、ランク表を全ランクへ送って集合点を閉じる。
fn collect_as_rank0(
    cfg: &RendezvousConfig,
    listener: &TcpListener,
    ring: &TcpListener,
    ring_port: u16,
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
        peers[rank] = Some((s, SocketAddr::new(peer.ip(), hello.ring_port)));
        got += 1;
    }
    // 自分のエントリは集合点 listener の IP（ring_bind_ip が unspecified のとき）か ring の IP。
    let rv_ip = listener.local_addr()?.ip();
    let ring_ip = ring.local_addr()?.ip();
    let own_ip = if ring_ip.is_unspecified() {
        rv_ip
    } else {
        ring_ip
    };
    let mut entries = Vec::with_capacity(w);
    entries.push(SocketAddr::new(own_ip, ring_port));
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

/// ランク 0 以外: 集合点へ Hello を送り、ランク表を受けて検証する。
fn fetch_table(cfg: &RendezvousConfig, ring_port: u16, deadline: Instant) -> Result<RankTable> {
    let w = cfg.world_size();
    let s = connect_with_retry(cfg.rendezvous_addr(), cfg.connect_timeout_value(), deadline)?;
    let io = cfg.io_timeout_value();
    write_hello(
        &mut DeadlineStream::new(&s, deadline, io),
        &my_hello(cfg, ring_port),
    )?;
    // 全ランクが揃うまで待つため、1 回あたりの上限は確立の期限そのもの（全体期限は有効）。
    let mut ds = DeadlineStream::new(&s, deadline, cfg.establish_timeout_value());
    let h = read_header(&mut ds)?;
    let table = read_rank_table(&mut ds, &h, w)?;
    for a in &table.entries {
        check_peer_allowed(*a, cfg.is_non_loopback_allowed())?;
    }
    Ok(table)
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

    fn establish_group(w: usize) -> Vec<RingLinks> {
        thread::scope(|s| {
            let (addr, h0) = spawn_rank0(s, cfg(w, 0, lo0()));
            let hs: Vec<_> = (1..w)
                .map(|r| s.spawn(move || Rendezvous::bind(cfg(w, r, addr))?.establish()))
                .collect();
            let mut v = vec![h0.join().unwrap().unwrap()];
            for h in hs {
                v.push(h.join().unwrap().unwrap());
            }
            v
        })
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

    #[test]
    fn ring_round_trip_for_several_world_sizes() {
        for w in [2usize, 3, 4] {
            let links = establish_group(w);
            for (r, l) in links.iter().enumerate() {
                assert_eq!((l.rank, l.world_size), (r, w));
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
