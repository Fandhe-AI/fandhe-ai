//! `distributed` 配下のテスト共通補助（`#[cfg(test)]` 限定）。
//!
//! 役割: loopback 上で W ランクのリングを確立する `establish_group` を、`ring` のテストと
//! `step`／`optim::device_store` の DDP 結線テスト（イシュー #3146）で共有する。本番コードからは
//! 参照されない。

use std::net::SocketAddr;
use std::thread;
use std::time::Duration;

use super::config::RendezvousConfig;
use super::rendezvous::{Rendezvous, RingLinks};

pub(crate) fn lo0() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

pub(crate) fn cfg(w: usize, rank: usize, addr: SocketAddr) -> RendezvousConfig {
    RendezvousConfig::new(w, rank, addr)
        .unwrap()
        .session_id(7)
        .establish_timeout(Duration::from_secs(20))
        .io_timeout(Duration::from_secs(5))
}

/// loopback で W ランクのリングを確立し、ランク順の `RingLinks` を返す。
pub(crate) fn establish_group(w: usize) -> Vec<RingLinks> {
    if w == 1 {
        return vec![
            Rendezvous::bind(cfg(1, 0, lo0()))
                .unwrap()
                .establish()
                .unwrap(),
        ];
    }
    thread::scope(|s| {
        let r0 = Rendezvous::bind(cfg(w, 0, lo0())).unwrap();
        let addr = r0.rendezvous_local_addr().unwrap();
        let h0 = s.spawn(move || r0.establish());
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
