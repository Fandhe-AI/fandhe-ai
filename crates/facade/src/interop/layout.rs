//! safetensors レイアウト検証の crate 内ラッパー（イシュー #3126。`cfg(unix)`）。
//!
//! 呼び出し元は `model.rs` の公開前検証（`ModelRegistry::download`。設計
//! `docs/model-download-design.md` §5 方式 (ii)）。dtype 非依存のレイアウト検証（ヘッダ長・
//! ヘッダ JSON・オフセット連続性・全体長）を `onnx-interop` へ素通しで委譲する。facade の
//! 公開面ではない（`pub(crate)`・`pub use` なし）。`onnx-interop` の参照を `src/interop/`
//! 配下に閉じ込める設計（#2017。`tests/api_surface.rs` が機械検査）を保つため、`model.rs` は
//! `onnx-interop` を名指さず本関数経由で呼ぶ。`safetensors.rs` は純再エクスポート契約
//! （#2019）のため、関数を足さず別ファイルにした。

use fandhe_ai_onnx_interop::st_load::LoadError;

/// dtype 非依存で safetensors バイト列のレイアウトだけを検証する（テンソルは確保しない）。
pub(crate) fn validate_from_bytes(bytes: &[u8]) -> Result<(), LoadError> {
    fandhe_ai_onnx_interop::st_load::validate_safetensors_layout_from_bytes(bytes)
}
