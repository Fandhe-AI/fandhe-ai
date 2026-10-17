//! ランク発見の起動情報（設定値）と loopback 検査。
//!
//! 役割: world_size・rank・集合点アドレス・opt-in・タイムアウトを受け、ソケットを開く前に
//! 検証を一か所に集める。`rendezvous.rs` の `Rendezvous::bind` が冒頭で `validate` を呼ぶ。
//! 環境変数の読み取りは初期形では持たない（記録 §3）。アドレスは `SocketAddr` のみ受け、
//! 文字列解釈と DNS 解決の経路を作らない（OWASP A03）。
//! 根拠: `docs/facade-distributed-host-collective-decision.md` §3・§5・§7。

use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use super::error::{DistributedError, Result};
use super::frame::MAX_WORLD_SIZE;

/// 同一アドレス族の正規 loopback（`127.0.0.1`／`::1`）。ランク 0 の接続元 IP 正規化にも使う。
pub(crate) fn canonical_loopback(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => IpAddr::from([127, 0, 0, 1]),
        IpAddr::V6(_) => IpAddr::from(std::net::Ipv6Addr::LOCALHOST),
    }
}

/// IPv4-mapped IPv6（`::ffff:127.0.0.0/8`）も含めた loopback 判定。
pub fn is_loopback_ip(ip: IpAddr) -> bool {
    if ip.is_loopback() {
        return true;
    }
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback()),
        IpAddr::V4(_) => false,
    }
}

/// opt-in が無いとき loopback 以外のアドレスを拒否する。bind 先・集合点・ランク表の接続先・
/// accept した接続元の 4 か所から呼ばれる（記録 §7: opt-in の迂回を防ぐ）。
pub fn check_peer_allowed(addr: SocketAddr, allow_non_loopback: bool) -> Result<()> {
    if allow_non_loopback || is_loopback_ip(addr.ip()) {
        Ok(())
    } else {
        Err(DistributedError::NonLoopbackNotAllowed { addr })
    }
}

/// 集合の確立に必要な起動情報。
#[derive(Debug, Clone)]
pub struct RendezvousConfig {
    world_size: usize,
    rank: usize,
    rendezvous_addr: SocketAddr,
    allow_non_loopback: bool,
    ring_bind_ip: IpAddr,
    session_id: u64,
    connect_timeout: Duration,
    establish_timeout: Duration,
    io_timeout: Duration,
}

impl RendezvousConfig {
    /// 既定値（loopback・接続 10 秒・確立 60 秒・読み書き 30 秒）で構築する。
    /// world_size と rank の範囲だけをここで検証し、残りは `validate`（`bind` 冒頭）で検証する。
    pub fn new(world_size: usize, rank: usize, rendezvous_addr: SocketAddr) -> Result<Self> {
        let cfg = Self {
            world_size,
            rank,
            rendezvous_addr,
            allow_non_loopback: false,
            // 既定は集合点と同じアドレス族の loopback（::1 の集合点で IPv4 リングにならない）。
            ring_bind_ip: if rendezvous_addr.is_ipv6() {
                IpAddr::from(std::net::Ipv6Addr::LOCALHOST)
            } else {
                IpAddr::from([127, 0, 0, 1])
            },
            session_id: 0,
            connect_timeout: Duration::from_secs(10),
            establish_timeout: Duration::from_secs(60),
            io_timeout: Duration::from_secs(30),
        };
        cfg.validate_size_and_rank()?;
        Ok(cfg)
    }

    /// 非 loopback の bind・接続を許可する（なりすまし・盗聴のリスクを利用者が受け入れる）。
    pub fn allow_non_loopback(mut self, allow: bool) -> Self {
        self.allow_non_loopback = allow;
        self
    }

    /// リング用待受の bind IP（既定 `127.0.0.1`）。
    pub fn ring_bind_ip(mut self, ip: IpAddr) -> Self {
        self.ring_bind_ip = ip;
        self
    }

    /// 別ジョブとの混線防止用の識別子（認証ではない）。
    pub fn session_id(mut self, id: u64) -> Self {
        self.session_id = id;
        self
    }

    /// 1 回の connect の上限。
    pub fn connect_timeout(mut self, d: Duration) -> Self {
        self.connect_timeout = d;
        self
    }

    /// 集合の確立全体の期限。
    pub fn establish_timeout(mut self, d: Duration) -> Self {
        self.establish_timeout = d;
        self
    }

    /// 1 回の読み書きの上限。
    pub fn io_timeout(mut self, d: Duration) -> Self {
        self.io_timeout = d;
        self
    }

    /// world_size。
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// 自ランク。
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// 集合点アドレス。
    pub fn rendezvous_addr(&self) -> SocketAddr {
        self.rendezvous_addr
    }

    /// 非 loopback の opt-in 有無。
    pub fn is_non_loopback_allowed(&self) -> bool {
        self.allow_non_loopback
    }

    /// セッション識別子。
    pub fn session(&self) -> u64 {
        self.session_id
    }

    /// 1 回の connect の上限。
    pub fn connect_timeout_value(&self) -> Duration {
        self.connect_timeout
    }

    /// 確立全体の期限。
    pub fn establish_timeout_value(&self) -> Duration {
        self.establish_timeout
    }

    /// 1 回の読み書きの上限。
    pub fn io_timeout_value(&self) -> Duration {
        self.io_timeout
    }

    /// リング用待受の bind IP。
    pub fn ring_bind_ip_value(&self) -> IpAddr {
        self.ring_bind_ip
    }

    fn validate_size_and_rank(&self) -> Result<()> {
        if self.world_size == 0 || self.world_size > MAX_WORLD_SIZE {
            return Err(DistributedError::WorldSizeOutOfRange {
                world_size: self.world_size,
                max: MAX_WORLD_SIZE,
            });
        }
        if self.rank >= self.world_size {
            return Err(DistributedError::RankOutOfRange {
                rank: self.rank,
                world_size: self.world_size,
            });
        }
        Ok(())
    }

    /// すべての検証をソケットを開く前に行う。
    pub fn validate(&self) -> Result<()> {
        self.validate_size_and_rank()?;
        // set_read_timeout(Some(0)) は Err になるため 0 を拒否する。
        if self.connect_timeout.is_zero()
            || self.establish_timeout.is_zero()
            || self.io_timeout.is_zero()
        {
            return Err(DistributedError::InvalidConfig {
                reason: "timeouts must be non-zero",
            });
        }
        // Instant に加算できない巨大 timeout は panic ではなく設定誤りとして拒否する。
        let now = Instant::now();
        if now.checked_add(self.connect_timeout).is_none()
            || now.checked_add(self.establish_timeout).is_none()
            || now.checked_add(self.io_timeout).is_none()
        {
            return Err(DistributedError::InvalidConfig {
                reason: "timeout is too large",
            });
        }
        // リングは集合点で観測したピア IP を使うため、アドレス族が異なると接続できない。
        if self.rendezvous_addr.is_ipv4() != self.ring_bind_ip.is_ipv4() {
            return Err(DistributedError::InvalidConfig {
                reason: "ring_bind_ip must be the same address family as the rendezvous address",
            });
        }
        // ループバック集合点では全ランクが同一ホストにいるため、ランク 0 が観測した接続元 IP
        // （127.0.0.1 か集合点と同じ IP かは OS 依存）を正規 loopback（127.0.0.1／::1）へ
        // 正規化して配布する。よって ring_bind_ip は正規 loopback か unspecified に限る
        // （127.0.0.2 のような別 loopback IP はリング待受に届かないため早期に拒否する）。
        if self.rendezvous_addr.ip().is_loopback()
            && !self.ring_bind_ip.is_unspecified()
            && self.ring_bind_ip != canonical_loopback(self.rendezvous_addr.ip())
        {
            return Err(DistributedError::InvalidConfig {
                reason: "ring_bind_ip must be the canonical loopback address (127.0.0.1 or ::1) or unspecified for a loopback rendezvous",
            });
        }
        if self.rendezvous_addr.ip().is_unspecified() {
            return Err(DistributedError::InvalidConfig {
                reason: "rendezvous address must not be unspecified",
            });
        }
        if self.rank != 0 && self.rendezvous_addr.port() == 0 {
            return Err(DistributedError::InvalidConfig {
                reason: "non-zero rank needs a concrete rendezvous port",
            });
        }
        check_peer_allowed(self.rendezvous_addr, self.allow_non_loopback)?;
        check_peer_allowed(
            SocketAddr::new(self.ring_bind_ip, 0),
            self.allow_non_loopback,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    #[test]
    fn loopback_table() {
        for s in ["127.0.0.1", "127.1.2.3", "::1", "::ffff:127.0.0.1"] {
            assert!(is_loopback_ip(s.parse().unwrap()), "{s}");
        }
        for s in ["192.0.2.1", "::ffff:192.0.2.1", "0.0.0.0"] {
            assert!(!is_loopback_ip(s.parse().unwrap()), "{s}");
        }
    }

    #[test]
    fn world_size_bounds() {
        let a = addr("127.0.0.1:1234");
        assert!(matches!(
            RendezvousConfig::new(0, 0, a),
            Err(DistributedError::WorldSizeOutOfRange { .. })
        ));
        assert!(matches!(
            RendezvousConfig::new(65, 0, a),
            Err(DistributedError::WorldSizeOutOfRange { .. })
        ));
        assert!(RendezvousConfig::new(64, 63, a).unwrap().validate().is_ok());
        assert!(matches!(
            RendezvousConfig::new(4, 4, a),
            Err(DistributedError::RankOutOfRange { .. })
        ));
    }

    #[test]
    fn invalid_config_cases() {
        let a = addr("127.0.0.1:1234");
        let c = RendezvousConfig::new(2, 0, a).unwrap();
        assert!(matches!(
            c.clone().io_timeout(Duration::ZERO).validate(),
            Err(DistributedError::InvalidConfig { .. })
        ));
        let u = RendezvousConfig::new(2, 0, addr("0.0.0.0:1234")).unwrap();
        assert!(matches!(
            u.validate(),
            Err(DistributedError::InvalidConfig { .. })
        ));
        let p0 = RendezvousConfig::new(2, 1, addr("127.0.0.1:0")).unwrap();
        assert!(matches!(
            p0.validate(),
            Err(DistributedError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn non_loopback_requires_opt_in() {
        let ext = addr("192.0.2.1:1234");
        let c = RendezvousConfig::new(2, 0, ext).unwrap();
        assert!(matches!(
            c.clone().validate(),
            Err(DistributedError::NonLoopbackNotAllowed { .. })
        ));
        assert!(c.allow_non_loopback(true).validate().is_ok());
        let c = RendezvousConfig::new(2, 0, addr("127.0.0.1:1234"))
            .unwrap()
            .ring_bind_ip("0.0.0.0".parse().unwrap());
        assert!(matches!(
            c.clone().validate(),
            Err(DistributedError::NonLoopbackNotAllowed { .. })
        ));
        assert!(c.allow_non_loopback(true).validate().is_ok());
    }

    #[test]
    fn loopback_rendezvous_rejects_mismatched_ring_bind_ip() {
        let c = RendezvousConfig::new(2, 1, addr("127.0.0.1:1234"))
            .unwrap()
            .ring_bind_ip("127.0.0.2".parse().unwrap());
        assert!(matches!(
            c.validate(),
            Err(DistributedError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn non_canonical_loopback_rendezvous_accepts_default_ring_bind_ip() {
        let c = RendezvousConfig::new(2, 1, addr("127.0.0.2:1234")).unwrap();
        assert!(c.validate().is_ok());
    }
}
