//! 語彙 lookup 型のテキスト変換（Keras `TextVectorization` 相当）の内部実装。
//!
//! 役割: spec REQ-9（2026-10-08 追記で範囲承認済み）の語彙 lookup 型テキスト
//! 変換を、facade 内部の非公開モジュールとして段階的に作る。本イシュー
//! （#2897）は骨格のみで、後続の標準化・分割・n-gram・語彙・adapt・transform が
//! 共通で使う `limits`（非信頼入力の上限）と `error`（型付きエラー）を置く。
//! 配置・既定値・検査時機の正は `docs/facade-text-vectorization-design.md`
//! §5・§6・§8。#2898 で標準化（`standardize`）と分割（`split`）の内部実装を
//! 追加した（論点 3 は未承認のため設計記録 §3 の推奨定義に従う）。#2899 で
//! n-gram 生成（`ngram`）を、#2900 で語彙の直接指定と lookup（`vocab`）を
//! 追加した。
//!
//! 範囲外: サブワード分割・Unicode 正規化・ファイル形式の読み書き。ファイル・
//! ネットワーク・スレッドは使わない（設計記録 §8 の機械検査契約。
//! `tests/text_module_hygiene.rs` が常時検査する）。
//!
//! 公開形・配置は未承認のため facade へは公開しない（公開は設計記録 §11 の 9）。

// 呼び出し元がまだ無い骨格段階のため、非テストビルドでのみ dead_code を許す。
// テストビルドでは有効のままなので、単体テストが全項目を使うことを強制する。
#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "イシュー #2897 の骨格段階。facade 公開（設計記録 §11 の 9）で呼び出し元が結線されたら撤去する"
    )
)]

pub(crate) mod error;
pub(crate) mod limits;
pub(crate) mod ngram;
pub(crate) mod split;
pub(crate) mod standardize;
pub(crate) mod vocab;
