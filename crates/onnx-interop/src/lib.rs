//! ONNX / safetensors 相互運用層。
//!
//! `safetensors` はワイヤフォーマットの読み書きのみに用い、`tensor-core` のテンソル型への
//! マッピングは自作する。ONNX の protobuf デコードは `prost` を用いるが、`prost-build`
//! （`protoc` へのビルド時依存）は使わず手書き derive で取り込む（PoC-v2-6。
//! `.claude/rules/deps-policy.md`）。外部フォーマットのパースは長さ・形状の検証を先に行い
//! 不正入力を弾く（OWASP A03。`.claude/rules/security.md`）。
//!
//! ## モジュール構成
//!
//! - [`st_load`]: safetensors パース → `tensor-core::Tensor<f32>` へのマッピング
//!   （TASK-7.1a・#73・REQ-7）。PyTorch で保存した重みファイルのロード経路を提供する。
//!   キー不足時は [`st_load::LoadError::MissingKeys`] で報告する。
//! - `require_keys`（非公開。TASK-7.1b・イシュー #74）は safetensors ロード結果の
//!   キーマップに対する期待キー集合の充足検査を提供する。キー不足を無言 skip せず
//!   型付きエラー（[`LoadError`]）で報告する（v1 PoC-6 詰まりポイント #2 対策。詳細は
//!   `require_keys` モジュールのドキュメンテーションコメント参照）。
//! - [`ops`]（TASK-7.2c・#79 / TASK-7.3a・#82 / TASK-7.3b・#83）: ONNX オペを
//!   `tensor-core::Tensor<f32>` 上の純粋関数として提供する。8 オペ（`Gemm`／`Relu`／
//!   `Sigmoid`／`Shape`／`Gather`／`Unsqueeze`／`Concat`／`Slice`）に加え MVP 算術オペ
//!   （`Add`／`Mul`／`Div`／`Mod`／`Sqrt`／`Constant`）・MVP 形状操作オペ（`Cast`／
//!   `Reshape`／`Squeeze`／`Transpose`）を含む。ONNX proto デコード（TASK-7.2a）・
//!   グラフ実行エンジンは別イシューの担当であり、本モジュールは「入力テンソル＋属性 →
//!   出力テンソル」の単体演算のみを扱う（decode → 属性値 → 本モジュール呼び出し、の結線は
//!   後続タスクで行う）。属性は proto 由来の型に依存しないプレーンな Rust 構造体・
//!   スライスとして受け取り、デコード層の実装順序に依存しない。
//! - ONNX proto デコード経路（TASK-7.2a）は本イシューのスコープ外で未実装。
//!
//! `require_keys` モジュールは非公開（`mod`）とし、型・関数のみ `pub use` で
//! クレートルートへ再エクスポートする。モジュールを `pub mod` にすると
//! `onnx_interop::require_keys` がモジュール（型名前空間）と関数（値名前空間）の
//! 同名で並存し、`use onnx_interop::require_keys;` が両方を意図せずインポートして
//! 読み手（Claude を含む）を混乱させるため（イシュー #74 レビュー指摘）。
//!
//! `require_keys::LoadError`（クレートルートへ [`LoadError`] として再エクスポート）と
//! [`st_load::LoadError`] は #73／#74 それぞれの独立実装として並存する（別名前空間なので
//! 衝突はしない）。統合整理の要否はスコープ外事項としてイシュー #74 側で追跡する
//! （`.claude/rules/out-of-scope-tracking.md`）。

pub mod ops;
pub mod st_load;

mod require_keys;
pub use require_keys::{LoadError, require_keys};
