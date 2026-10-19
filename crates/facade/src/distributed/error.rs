//! 分散学習（`fandhe_ai::distributed`）の型付きエラー。
//!
//! 役割: autodiff 内部の `distributed::DistributedError`／`DistributedStepError` を、内部 enum を
//! ペイロードに持たない自己完結の [`DistributedError`] へ写して facade の公開面に出す
//! （`docs/facade-distributed-host-collective-decision.md` §5・§16。`OnnxError` と同じ方針）。
//! 写像は `pub(crate)` の `from_inner`／`from_step` に置き、`From` impl は公開しない
//! （内部型が公開 trait impl として現れるのを避ける）。
//!
//! `Display` は固定文・数値・内側エラーの `Display` だけで、ピアから受け取ったバイト列は出さない
//! （OWASP A09）。

use std::fmt;
use std::net::SocketAddr;

use fandhe_ai_autodiff::distributed as inner;

use crate::BackendError;

/// 分散学習の設定・集合の確立・集合通信・分散ステップが返す型付きエラー。
#[derive(Debug)]
#[non_exhaustive]
pub enum DistributedError {
    /// 設定値が不正（固定文の理由）。
    InvalidConfig {
        /// 固定文の理由。
        reason: &'static str,
    },
    /// world_size が `1..=max` の範囲外。
    WorldSizeOutOfRange {
        /// 指定値。
        world_size: usize,
        /// 上限。
        max: usize,
    },
    /// ランク番号が `0..world_size` の範囲外。
    RankOutOfRange {
        /// 指定値。
        rank: usize,
        /// world_size。
        world_size: usize,
    },
    /// 非 loopback のアドレスが opt-in 無しで指定された。
    NonLoopbackNotAllowed {
        /// 該当アドレス。
        addr: SocketAddr,
    },
    /// 同じランク番号が重複した。
    DuplicateRank {
        /// 重複したランク。
        rank: usize,
    },
    /// ピアの world_size が自分と違う。
    WorldSizeMismatch {
        /// 自分の world_size。
        expected: usize,
        /// ピアの申告値。
        got: usize,
    },
    /// リングの相手が想定したランクと違う。
    UnexpectedRank {
        /// 想定したランク。
        expected: usize,
        /// ピアの申告値。
        got: usize,
    },
    /// ワイヤ上のプロトコル違反（固定文と数値だけの説明。ピアのバイト列は含まない）。
    Protocol {
        /// 説明文。
        message: String,
    },
    /// タイムアウト。
    Timeout {
        /// 固定文の段階名。
        phase: &'static str,
    },
    /// ピアが切断した。
    PeerDisconnected,
    /// その他の I/O エラー。
    Io(std::io::Error),
    /// communicator が poisoned（以後の集合通信は拒否される。復旧は範囲外）。
    Poisoned,
    /// 勾配の読み出しや optimizer ステップが返したバックエンドエラー。
    Backend(BackendError),
    /// 平坦化した勾配の長さがストアの総要素数と一致しない。
    GradientLengthMismatch {
        /// ストアの総要素数。
        expected: usize,
        /// 平坦化後の長さ。
        got: usize,
    },
    /// 将来 autodiff 側に増える variant の受け皿（現行の全 variant はこれ以外へ写る）。
    Other {
        /// 説明文。
        message: String,
    },
}

impl DistributedError {
    /// 内部の集合通信エラーから写す。
    pub(crate) fn from_inner(e: inner::DistributedError) -> Self {
        use inner::DistributedError as E;
        match e {
            E::InvalidConfig { reason } => Self::InvalidConfig { reason },
            E::WorldSizeOutOfRange { world_size, max } => {
                Self::WorldSizeOutOfRange { world_size, max }
            }
            E::RankOutOfRange { rank, world_size } => Self::RankOutOfRange { rank, world_size },
            E::DuplicateRank { rank } => Self::DuplicateRank { rank },
            E::WorldSizeMismatch { expected, got } => Self::WorldSizeMismatch { expected, got },
            E::UnexpectedRank { expected, got } => Self::UnexpectedRank { expected, got },
            E::NonLoopbackNotAllowed { addr } => Self::NonLoopbackNotAllowed { addr },
            E::Protocol(v) => Self::Protocol {
                message: v.to_string(),
            },
            E::Timeout { phase } => Self::Timeout { phase },
            E::PeerDisconnected => Self::PeerDisconnected,
            E::Io(e) => Self::Io(e),
            E::Poisoned => Self::Poisoned,
            other => Self::Other {
                message: other.to_string(),
            },
        }
    }

    /// 内部の分散ステップエラーから写す（`Collective` は `from_inner` を再帰適用する）。
    pub(crate) fn from_step(e: inner::DistributedStepError) -> Self {
        use inner::DistributedStepError as S;
        match e {
            S::Collective(c) => Self::from_inner(c),
            S::Backend(b) => Self::Backend(b),
            S::GradientLengthMismatch { expected, got } => {
                Self::GradientLengthMismatch { expected, got }
            }
            other => Self::Other {
                message: other.to_string(),
            },
        }
    }
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid distributed config: {reason}"),
            Self::WorldSizeOutOfRange { world_size, max } => {
                write!(f, "world_size {world_size} is out of range 1..={max}")
            }
            Self::RankOutOfRange { rank, world_size } => {
                write!(f, "rank {rank} is out of range 0..{world_size}")
            }
            Self::NonLoopbackNotAllowed { addr } => write!(
                f,
                "non-loopback address {addr} requires explicit opt-in (allow_non_loopback)"
            ),
            Self::DuplicateRank { rank } => write!(f, "duplicate rank {rank}"),
            Self::WorldSizeMismatch { expected, got } => {
                write!(f, "world_size mismatch (expected {expected}, got {got})")
            }
            Self::UnexpectedRank { expected, got } => {
                write!(f, "unexpected peer rank (expected {expected}, got {got})")
            }
            Self::Protocol { message } => write!(f, "protocol violation: {message}"),
            Self::Timeout { phase } => write!(f, "timed out during {phase}"),
            Self::PeerDisconnected => write!(f, "peer disconnected"),
            Self::Io(e) => write!(f, "i/o error: {:?}", e.kind()),
            Self::Poisoned => write!(f, "communicator is poisoned"),
            Self::Backend(e) => write!(f, "backend error during distributed step: {e}"),
            Self::GradientLengthMismatch { expected, got } => write!(
                f,
                "flattened gradient length mismatch (expected {expected}, got {got})"
            ),
            Self::Other { message } => write!(f, "distributed error: {message}"),
        }
    }
}

impl std::error::Error for DistributedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Backend(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inner::DistributedError as E;
    use inner::DistributedStepError as S;

    fn addr() -> SocketAddr {
        "192.0.2.1:1".parse().unwrap()
    }

    #[test]
    fn every_inner_collective_variant_maps_to_a_dedicated_variant() {
        type Check = fn(&DistributedError) -> bool;
        let cases: Vec<(E, Check)> = vec![
            (E::InvalidConfig { reason: "x" }, |d| {
                matches!(d, DistributedError::InvalidConfig { reason: "x" })
            }),
            (
                E::WorldSizeOutOfRange {
                    world_size: 0,
                    max: 64,
                },
                |d| {
                    matches!(
                        d,
                        DistributedError::WorldSizeOutOfRange {
                            world_size: 0,
                            max: 64
                        }
                    )
                },
            ),
            (
                E::RankOutOfRange {
                    rank: 2,
                    world_size: 2,
                },
                |d| {
                    matches!(
                        d,
                        DistributedError::RankOutOfRange {
                            rank: 2,
                            world_size: 2
                        }
                    )
                },
            ),
            (E::DuplicateRank { rank: 1 }, |d| {
                matches!(d, DistributedError::DuplicateRank { rank: 1 })
            }),
            (
                E::WorldSizeMismatch {
                    expected: 2,
                    got: 3,
                },
                |d| {
                    matches!(
                        d,
                        DistributedError::WorldSizeMismatch {
                            expected: 2,
                            got: 3
                        }
                    )
                },
            ),
            (
                E::UnexpectedRank {
                    expected: 1,
                    got: 2,
                },
                |d| {
                    matches!(
                        d,
                        DistributedError::UnexpectedRank {
                            expected: 1,
                            got: 2
                        }
                    )
                },
            ),
            (E::NonLoopbackNotAllowed { addr: addr() }, |d| {
                matches!(d, DistributedError::NonLoopbackNotAllowed { .. })
            }),
            (
                E::Protocol(inner::ProtocolViolation::BadMagic),
                |d| matches!(d, DistributedError::Protocol { message } if message == "protocol violation: bad magic" || message == "bad magic"),
            ),
            (E::Timeout { phase: "p" }, |d| {
                matches!(d, DistributedError::Timeout { phase: "p" })
            }),
            (E::PeerDisconnected, |d| {
                matches!(d, DistributedError::PeerDisconnected)
            }),
            (E::Io(std::io::Error::other("x")), |d| {
                matches!(d, DistributedError::Io(_))
            }),
            (E::Poisoned, |d| matches!(d, DistributedError::Poisoned)),
        ];
        for (inner_err, check) in cases {
            let label = format!("{inner_err:?}");
            let mapped = DistributedError::from_inner(inner_err);
            assert!(check(&mapped), "{label} の写像が想定外: {mapped:?}");
            assert!(!matches!(mapped, DistributedError::Other { .. }), "{label}");
        }
    }

    #[test]
    fn step_variants_map_and_collective_recurses() {
        let m = DistributedError::from_step(S::Collective(E::PeerDisconnected));
        assert!(matches!(m, DistributedError::PeerDisconnected));
        let m = DistributedError::from_step(S::GradientLengthMismatch {
            expected: 3,
            got: 2,
        });
        assert!(matches!(
            m,
            DistributedError::GradientLengthMismatch {
                expected: 3,
                got: 2
            }
        ));
        let m = DistributedError::from_step(S::Backend(BackendError::InvalidArgument("x".into())));
        assert!(matches!(m, DistributedError::Backend(_)));
        assert!(std::error::Error::source(&m).is_some());
    }

    #[test]
    fn display_has_no_debug_dump_of_io_error() {
        let e = DistributedError::Io(std::io::Error::other("secret-detail"));
        assert!(!e.to_string().contains("secret-detail"));
    }
}
