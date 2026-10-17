//! 分散学習の設定（world_size・rank・集合点アドレス・opt-in・タイムアウト）。
//!
//! 役割: autodiff 内部の `RendezvousConfig`／`CollectiveOptions` を facade の builder で包み、
//! `build()` の時点でソケットを開く前にすべて検証する（OWASP A03。決定記録 §16）。
//! 既定は loopback。非 loopback は `allow_non_loopback(true)` の明示 opt-in だけで許す。
//! タイムアウトの既定値は内部の既定（接続 10 秒・確立 60 秒・io 30 秒・all-reduce 1 回 120 秒）を
//! そのまま使い、facade では独自の既定値を持たない。

use std::net::SocketAddr;
use std::time::Duration;

use fandhe_ai_autodiff::distributed::{CollectiveOptions, RendezvousConfig};

use super::error::DistributedError;

/// 検証済みの分散学習設定。[`DistributedConfig::builder`] から作る。
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub(crate) rendezvous: RendezvousConfig,
    pub(crate) collective: CollectiveOptions,
}

impl DistributedConfig {
    /// builder を作る（world_size・rank・集合点アドレスは `build` で検証する）。
    ///
    /// 集合点アドレスはランク 0 が bind し、他ランクが接続する先。`SocketAddr` だけを受け、
    /// ホスト名解決（DNS）は行わない。
    pub fn builder(
        world_size: usize,
        rank: usize,
        rendezvous_addr: SocketAddr,
    ) -> DistributedConfigBuilder {
        DistributedConfigBuilder {
            world_size,
            rank,
            rendezvous_addr,
            allow_non_loopback: false,
            session_id: None,
            connect_timeout: None,
            establish_timeout: None,
            io_timeout: None,
            op_timeout: None,
        }
    }

    /// world_size（全ランク数）。
    pub fn world_size(&self) -> usize {
        self.rendezvous.world_size()
    }

    /// 自ランク。
    pub fn rank(&self) -> usize {
        self.rendezvous.rank()
    }

    /// 集合点アドレス（IPv4-mapped IPv6 は IPv4 へ正規化済み）。
    pub fn rendezvous_addr(&self) -> SocketAddr {
        self.rendezvous.rendezvous_addr()
    }

    /// 非 loopback が明示 opt-in されているか。
    pub fn is_non_loopback_allowed(&self) -> bool {
        self.rendezvous.is_non_loopback_allowed()
    }
}

/// [`DistributedConfig`] の builder。
#[derive(Debug, Clone)]
#[must_use]
pub struct DistributedConfigBuilder {
    world_size: usize,
    rank: usize,
    rendezvous_addr: SocketAddr,
    allow_non_loopback: bool,
    session_id: Option<u64>,
    connect_timeout: Option<Duration>,
    establish_timeout: Option<Duration>,
    io_timeout: Option<Duration>,
    op_timeout: Option<Duration>,
}

impl DistributedConfigBuilder {
    /// 非 loopback の bind・接続を許可する。**認証も暗号化も無い**ため、なりすまし・盗聴の
    /// リスクを受け入れられる信頼できるネットワークでだけ使うこと。
    pub fn allow_non_loopback(mut self, allow: bool) -> Self {
        self.allow_non_loopback = allow;
        self
    }

    /// セッション識別子。誤接続の検出用で、**認証ではない**。
    pub fn session_id(mut self, id: u64) -> Self {
        self.session_id = Some(id);
        self
    }

    /// 接続 1 回の上限。
    pub fn connect_timeout(mut self, d: Duration) -> Self {
        self.connect_timeout = Some(d);
        self
    }

    /// 集合の確立全体の期限。
    pub fn establish_timeout(mut self, d: Duration) -> Self {
        self.establish_timeout = Some(d);
        self
    }

    /// 1 回の読み書きの上限（確立時と all-reduce の両方に適用する）。
    pub fn io_timeout(mut self, d: Duration) -> Self {
        self.io_timeout = Some(d);
        self
    }

    /// all-reduce 1 回の全体の絶対期限。
    pub fn op_timeout(mut self, d: Duration) -> Self {
        self.op_timeout = Some(d);
        self
    }

    /// すべてをソケットを開く前に検証して確定する。
    ///
    /// # Errors
    ///
    /// world_size・rank の範囲外、タイムアウトの 0／巨大値、unspecified・マルチキャスト・
    /// ブロードキャスト・リンクローカルのアドレス、ランク 1 以降のポート 0、opt-in 無しの
    /// 非 loopback を [`DistributedError`] で拒否する。
    pub fn build(self) -> Result<DistributedConfig, DistributedError> {
        let mut rendezvous =
            RendezvousConfig::new(self.world_size, self.rank, self.rendezvous_addr)
                .map_err(DistributedError::from_inner)?
                .allow_non_loopback(self.allow_non_loopback);
        let mut collective = CollectiveOptions::default();
        if let Some(id) = self.session_id {
            rendezvous = rendezvous.session_id(id);
        }
        if let Some(d) = self.connect_timeout {
            rendezvous = rendezvous.connect_timeout(d);
        }
        if let Some(d) = self.establish_timeout {
            rendezvous = rendezvous.establish_timeout(d);
        }
        if let Some(d) = self.io_timeout {
            rendezvous = rendezvous.io_timeout(d);
            collective = collective.io_timeout(d);
        }
        if let Some(d) = self.op_timeout {
            collective = collective.op_timeout(d);
        }
        rendezvous
            .validate()
            .map_err(DistributedError::from_inner)?;
        collective
            .validate()
            .map_err(DistributedError::from_inner)?;
        Ok(DistributedConfig {
            rendezvous,
            collective,
        })
    }
}
