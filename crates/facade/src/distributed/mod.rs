//! 分散学習（DDP の初期形）の公開モジュール。
//!
//! std のみの TCP によるホスト側 ring all-reduce で、各ランクの勾配を平均してから optimizer を
//! 更新する（`docs/facade-distributed-host-collective-decision.md` §16。実装は autodiff 内部の
//! `distributed`。イシュー #3147・親 #3143）。承認の根拠は同記録 §11 の D-A・D-B。
//!
//! # 公開する名前
//!
//! 設定 [`DistributedConfig`]（builder は [`DistributedConfigBuilder`]）、集合の確立
//! [`Rendezvous`]・集合通信ハンドル [`Communicator`]、エラー [`DistributedError`]、分散ステップの
//! フリー関数 [`step_sgd`]・[`step_adam`]・[`step_adamw`] だけを、このモジュールから公開する
//! （クレートルートへは再エクスポートしない。サブモジュールも非公開）。汎用の `all_reduce`
//! 等は公開しない。
//!
//! # セキュリティ前提
//!
//! - **認証も暗号化も無い**。信頼できるネットワーク内でだけ使う。`session_id` は誤接続の検出用で
//!   認証ではない。
//! - 既定は loopback。非 loopback は `allow_non_loopback(true)` の明示 opt-in だけで許す。
//! - 設定は `build()` でソケットを開く前に検証し、範囲外は型付きエラーで拒否する。
//! - all-reduce が失敗したらパラメータを更新しない（fail-closed）。
//!
//! # 使用例
//!
//! world_size 1 は無通信で、既存の単一プロセス学習と同じ結果になる。
//!
//! ```
//! use fandhe_ai::distributed::{Communicator, DistributedConfig, DistributedError};
//!
//! let addr = "127.0.0.1:29500".parse().unwrap();
//! let config = DistributedConfig::builder(1, 0, addr).build().unwrap();
//! let comm = Communicator::init(&config).unwrap();
//! assert_eq!((comm.rank(), comm.world_size()), (0, 1));
//! assert!(!comm.is_poisoned());
//!
//! // 範囲外の rank は型付きエラーで拒否される。
//! let err = DistributedConfig::builder(2, 2, addr).build().unwrap_err();
//! assert!(matches!(err, DistributedError::RankOutOfRange { rank: 2, world_size: 2 }));
//! ```
//!
//! world_size 2 以上では、ランク 0 が先に `Rendezvous::bind` して実アドレスを取り、他ランクへ
//! 渡してから `establish` する（外部通信を伴うため実行しない例）。
//!
//! ```no_run
//! use fandhe_ai::distributed::{Communicator, DistributedConfig, Rendezvous};
//!
//! // ランク 0（ポート 0 で bind し、実アドレスを他ランクへ伝える）。
//! let addr = "127.0.0.1:0".parse().unwrap();
//! let config = DistributedConfig::builder(2, 0, addr).session_id(7).build().unwrap();
//! let rdv = Rendezvous::bind(&config).unwrap();
//! let _actual = rdv.local_addr().unwrap(); // 他ランクへ渡す
//! let mut comm: Communicator = rdv.establish().unwrap();
//! assert_eq!(comm.world_size(), 2);
//! let _ = &mut comm; // `step_sgd(&tape, &mut store, &grads, &cfg, &mut comm)` に渡す
//! ```

mod communicator;
mod config;
mod error;
mod step;

pub use communicator::{Communicator, Rendezvous};
pub use config::{DistributedConfig, DistributedConfigBuilder};
pub use error::DistributedError;
pub use step::{step_adam, step_adamw, step_sgd};
