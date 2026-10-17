//! DDP 初期形のホスト側 collective のトランスポート層とランク発見（イシュー #3144）。
//!
//! 役割: `std::net` のみ（依存追加なし）で、ランク 0 を集合点にしたランク発見と、リング上の
//! 隣接ランクとの TCP 接続確立・長さ前置きフレームの送受信・受信検証・タイムアウトを提供する。
//! all-reduce 本体と communicator の poisoned 状態機械は #3145、optimizer への結線は #3146、
//! facade 公開は #3147（本モジュールは autodiff の内部 API で、サポートする公開面は facade のみ）。
//!
//! 設計の正本: `docs/facade-distributed-host-collective-decision.md`（§2〜§7・§13）。
//! spec: `docs/spec/04-requirements.md` `:386-393`。
//!
//! # セキュリティ上の前提
//!
//! - 既定は loopback のみ。非 loopback は `RendezvousConfig::allow_non_loopback(true)` の明示
//!   opt-in でだけ有効になる。リング待受の IP は設定値にせず、ランク 0 は集合点の待受 IP、
//!   それ以外は集合点への接続の `local_addr` IP から導く（配布アドレスに必ず待受がある。
//!   `rendezvous.rs` 冒頭・記録 §13）。
//! - 受信長・ランク・world_size・連番は確保や処理の前に検証する（OWASP A03）。
//! - **認証も暗号化も無い**。信頼できるネットワークの中でだけ使うこと。セッション識別子は
//!   別ジョブとの混線防止用で、認証ではない。

pub mod config;
pub mod error;
pub mod frame;
pub mod rendezvous;
pub mod transport;

pub use config::{RendezvousConfig, check_peer_allowed, is_loopback_ip};
pub use error::{DistributedError, ProtocolViolation};
pub use rendezvous::{Rendezvous, RingLinks};
