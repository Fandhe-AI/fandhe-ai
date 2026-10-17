//! ホスト側 collective（DDP 初期形）の型付きエラー。
//!
//! 役割: `distributed` モジュール全体（フレーム・トランスポート・ランク発見、後続 #3145 の
//! ring all-reduce）が返すエラーの単一の型。`AutodiffError` とは別の型にする
//! （`docs/facade-distributed-host-collective-decision.md` §5）。facade 公開用の自己完結型は #3147。
//!
//! `Display` は固定文と数値だけを出し、ピアが送ったバイト列・文字列は出さない
//! （OWASP A09。facade `model/transport.rs` の前例と同方針）。

use std::fmt;
use std::io;
use std::net::SocketAddr;

use fandhe_ai_tensor_core::device::BackendError;

/// ワイヤ上のプロトコル違反の種別（`DistributedError::Protocol` のペイロード）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProtocolViolation {
    /// magic が一致しない。
    BadMagic,
    /// 未対応のプロトコル版。
    UnsupportedVersion {
        /// ピアが申告した版。
        got: u16,
    },
    /// 未知、または期待と異なるフレーム種別。
    UnexpectedKind {
        /// ピアが申告した種別。
        got: u8,
    },
    /// 予約領域が 0 でない。
    ReservedNonZero,
    /// 本体長が絶対上限を超える。
    LengthTooLarge {
        /// 申告された長さ。
        len: u64,
        /// 上限。
        max: u64,
    },
    /// 本体長が期待長と一致しない。
    LengthMismatch {
        /// 期待長。
        expected: u64,
        /// 申告された長さ。
        got: u64,
    },
    /// ステップ連番が一致しない。
    SequenceMismatch,
    /// 総要素数が一致しない。
    TotalElementsMismatch,
    /// チャンク番号が一致しない。
    ChunkMismatch,
    /// セグメントのオフセットが一致しない。
    OffsetMismatch,
    /// ランク表の形式が不正。
    MalformedRankTable,
    /// ランク表のエントリが、受信側が自分で確かめられる待受アドレスと一致しない。
    /// 自ランクのエントリ（自分のリング待受）とランク 0 のエントリ（集合点の IP）を検査する。
    /// 配布アドレスに待受が無いまま後の connect がタイムアウトするのを、確立前に止めるため。
    RankTableAddressMismatch {
        /// 不一致だったエントリのランク。
        rank: usize,
    },
}

/// ホスト側 collective が返す型付きエラー。
#[derive(Debug)]
#[non_exhaustive]
pub enum DistributedError {
    /// 設定値が不正。
    InvalidConfig {
        /// 固定文の理由。
        reason: &'static str,
    },
    /// world_size が `1..=MAX_WORLD_SIZE` の範囲外。
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
    /// 非 loopback のアドレスが opt-in 無しで指定された。
    NonLoopbackNotAllowed {
        /// 該当アドレス。
        addr: SocketAddr,
    },
    /// ワイヤ上のプロトコル違反。
    Protocol(ProtocolViolation),
    /// タイムアウト。
    Timeout {
        /// 固定文の段階名。
        phase: &'static str,
    },
    /// ピアが切断した（途中で切れたフレームを含む）。
    PeerDisconnected,
    /// その他の I/O エラー。
    Io(io::Error),
    /// communicator が poisoned（`ring::RingCommunicator` が all-reduce の失敗時に遷移し、以後の呼び出しを拒否する。復旧は範囲外）。
    Poisoned,
}

impl fmt::Display for ProtocolViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic"),
            Self::UnsupportedVersion { got } => write!(f, "unsupported protocol version {got}"),
            Self::UnexpectedKind { got } => write!(f, "unexpected frame kind {got}"),
            Self::ReservedNonZero => write!(f, "reserved field is not zero"),
            Self::LengthTooLarge { len, max } => write!(f, "length {len} exceeds limit {max}"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch (expected {expected}, got {got})")
            }
            Self::SequenceMismatch => write!(f, "step sequence mismatch"),
            Self::TotalElementsMismatch => write!(f, "total element count mismatch"),
            Self::ChunkMismatch => write!(f, "chunk index mismatch"),
            Self::OffsetMismatch => write!(f, "segment offset mismatch"),
            Self::MalformedRankTable => write!(f, "malformed rank table"),
            Self::RankTableAddressMismatch { rank } => write!(
                f,
                "rank table entry for rank {rank} does not match its listening address"
            ),
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
            Self::DuplicateRank { rank } => write!(f, "duplicate rank {rank}"),
            Self::WorldSizeMismatch { expected, got } => {
                write!(f, "world_size mismatch (expected {expected}, got {got})")
            }
            Self::UnexpectedRank { expected, got } => {
                write!(f, "unexpected peer rank (expected {expected}, got {got})")
            }
            Self::NonLoopbackNotAllowed { addr } => write!(
                f,
                "non-loopback address {addr} requires explicit opt-in (allow_non_loopback)"
            ),
            Self::Protocol(v) => write!(f, "protocol violation: {v}"),
            Self::Timeout { phase } => write!(f, "timed out during {phase}"),
            Self::PeerDisconnected => write!(f, "peer disconnected"),
            Self::Io(e) => write!(f, "i/o error: {:?}", e.kind()),
            Self::Poisoned => write!(f, "communicator is poisoned"),
        }
    }
}

impl std::error::Error for DistributedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ProtocolViolation> for DistributedError {
    fn from(v: ProtocolViolation) -> Self {
        Self::Protocol(v)
    }
}

/// `io::Error` の写像。Unix は `WouldBlock`・Windows は `TimedOut` でタイムアウトを返すため両方を
/// `Timeout` に寄せる。切断系は `PeerDisconnected`（途中で切れたフレームの検出に使う）。
impl From<io::Error> for DistributedError {
    fn from(e: io::Error) -> Self {
        use io::ErrorKind::*;
        match e.kind() {
            WouldBlock | TimedOut => Self::Timeout { phase: "io" },
            UnexpectedEof | ConnectionReset | ConnectionAborted | BrokenPipe => {
                Self::PeerDisconnected
            }
            _ => Self::Io(e),
        }
    }
}

/// 本モジュールの `Result` 別名。
pub type Result<T> = std::result::Result<T, DistributedError>;

/// 勾配平均を optimizer ステップへ結線した分散ステップ（`distributed::step`。#3146）の型付きエラー。
///
/// `AutodiffError` とは別の型にする（記録 §5。`AutodiffError` は facade から再エクスポートされるため、
/// `DistributedError` を載せると内部 enum が公開面へ漏れる）。facade 向けの自己完結型は #3147。
/// `Display` は固定文・数値・内側エラーの `Display` だけで、ピアのバイト列は出さない（OWASP A09）。
#[derive(Debug)]
#[non_exhaustive]
pub enum DistributedStepError {
    /// all-reduce の失敗。このときどのパラメータも optimizer 状態も更新されていない。
    Collective(DistributedError),
    /// 勾配の読み出し、または既存 `step*` が返したバックエンドエラー。
    Backend(BackendError),
    /// 平坦化した勾配の長さが `total_numel` と一致しない（通信前のローカル検証）。
    GradientLengthMismatch {
        /// ストアの総要素数。
        expected: usize,
        /// 平坦化後の長さ。
        got: usize,
    },
}

impl fmt::Display for DistributedStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collective(e) => write!(f, "gradient all-reduce failed: {e}"),
            Self::Backend(e) => write!(f, "backend error during distributed step: {e}"),
            Self::GradientLengthMismatch { expected, got } => write!(
                f,
                "flattened gradient length mismatch (expected {expected}, got {got})"
            ),
        }
    }
}

impl std::error::Error for DistributedStepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Collective(e) => Some(e),
            Self::Backend(e) => Some(e),
            Self::GradientLengthMismatch { .. } => None,
        }
    }
}

impl From<DistributedError> for DistributedStepError {
    fn from(e: DistributedError) -> Self {
        Self::Collective(e)
    }
}

impl From<BackendError> for DistributedStepError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}
