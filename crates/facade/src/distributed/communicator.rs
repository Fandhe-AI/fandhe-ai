//! 集合の確立（`Rendezvous`）と集合通信ハンドル（`Communicator`）。
//!
//! 役割: autodiff 内部の `Rendezvous`／`RingCommunicator` を facade の不透明な型で包む
//! （`docs/facade-distributed-host-collective-decision.md` §3・§16）。内部型は公開シグネチャに出さない。
//! 検証は内部の検証済み API（`bind` 冒頭の `validate`・`RingCommunicator::new`）に任せ、迂回しない。

use std::net::SocketAddr;

use fandhe_ai_autodiff::distributed as inner;

use super::config::DistributedConfig;
use super::error::DistributedError;

/// bind 済みでまだ集合していない状態。
///
/// ポート 0 で bind した集合点の実アドレスを、ピア待ちの前に [`Rendezvous::local_addr`] で取得
/// できるようにする 2 段構成（TOCTOU 回避）。
#[derive(Debug)]
pub struct Rendezvous {
    inner: inner::Rendezvous,
    collective: inner::CollectiveOptions,
}

impl Rendezvous {
    /// 設定を検証し、ランク 0（world_size 2 以上）は集合点とリング用の待受を開く。
    /// それ以外はここではソケットを開かない。
    pub fn bind(config: &DistributedConfig) -> Result<Rendezvous, DistributedError> {
        let inner = inner::Rendezvous::bind(config.rendezvous.clone())
            .map_err(DistributedError::from_inner)?;
        Ok(Rendezvous {
            inner,
            collective: config.collective,
        })
    }

    /// 集合点の実際の待受アドレス（ランク 0 かつ world_size 2 以上のときだけ `Some`）。
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.inner.rendezvous_local_addr()
    }

    /// 全ランクの集合を確立して [`Communicator`] を返す。全体を `establish_timeout` で包む。
    pub fn establish(self) -> Result<Communicator, DistributedError> {
        let links = self
            .inner
            .establish()
            .map_err(DistributedError::from_inner)?;
        let ring = inner::RingCommunicator::new(links, self.collective)
            .map_err(DistributedError::from_inner)?;
        Ok(Communicator { inner: ring })
    }
}

/// 確立済みのホスト側 ring 集合通信ハンドル。
///
/// 勾配平均付きの分散ステップ（`step_sgd`／`step_adam`／`step_adamw`）に渡して使う。
/// all-reduce が失敗すると poisoned になり、以後の集合通信は拒否される。
#[derive(Debug)]
pub struct Communicator {
    pub(crate) inner: inner::RingCommunicator,
}

impl Communicator {
    /// `Rendezvous::bind` と `establish` を続けて行う便宜関数。
    ///
    /// world_size 2 以上でランク 0 の集合点ポートが 0 の構成は、ピアが実ポートを知る手段が無く
    /// 必ずタイムアウトするため `InvalidConfig` で拒否する。ポート 0 を使うなら
    /// [`Rendezvous`] で `bind` → `local_addr` を先に取得すること。world_size 1 は無通信。
    pub fn init(config: &DistributedConfig) -> Result<Communicator, DistributedError> {
        if config.world_size() >= 2 && config.rank() == 0 && config.rendezvous_addr().port() == 0 {
            return Err(DistributedError::InvalidConfig {
                reason: "rank 0 with world_size >= 2 needs a concrete rendezvous port (use Rendezvous for port 0)",
            });
        }
        Rendezvous::bind(config)?.establish()
    }

    /// 自ランク。
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// world_size（全ランク数）。
    pub fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    /// all-reduce の失敗で poisoned になっているか。
    pub fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }
}
