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
//! - [`ops`]: ONNX 8 オペ（`Gemm`／`Relu`／`Sigmoid`／`Shape`／`Gather`／`Unsqueeze`／
//!   `Concat`／`Slice`。TASK-7.2c・#79）を `tensor-core::Tensor<f32>` 上の純粋関数として提供する。
//!   ONNX proto デコード（TASK-7.2a）・グラフ実行エンジンは別イシューの担当であり、本モジュールは
//!   「入力テンソル＋属性 → 出力テンソル」の単体演算のみを扱う（decode → 属性値 →
//!   本モジュール呼び出し、の結線は後続タスクで行う）。属性は proto 由来の型に依存しない
//!   プレーンな Rust 構造体・スライスとして受け取り、デコード層の実装順序に依存しない。

pub mod ops;
pub mod st_load;
