//! 互換 API 層（REQ-9）が積む「レイヤー」相当のモジュール群の入口。
//!
//! `Var`（`crate::var`）の演算メソッドは値・shape のみを扱う低レベル
//! API であり、`compat::Sequential`（TASK-9.2・#94/#95）はこの上に
//! 「モジュールを並べて `forward` を連鎖させる」高レベル API を積む
//! 想定である。本モジュールはその中間に位置する薄いレイヤー実装群
//! （TASK-9.1・REQ-9）を集約する。
//!
//! 現時点（TASK-9.1b・#92）は活性化関数（[`activation`]）のみを提供
//! する。共通 `Module` trait の定義は本イシューでは行わない
//! （Linear・#91 と合わせて `compat::Sequential` 設計時に確定する）。

pub mod activation;
