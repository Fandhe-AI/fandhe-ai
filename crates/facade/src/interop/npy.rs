//! NumPy 互換 `.npy`／`.npz` 読み書きの公開面（イシュー #2590・親 #2588・
//! ルート #2499 Phase 3。承認形は `docs/tensor-core-npy-npz-io-decision.md`
//! §10.4 の P1〜P7）。
//!
//! `fandhe_ai_tensor_core::io`（内部クレート。実装は #2189）が提供する
//! `Tensor<f32>` 向けのパス版 npy／npz 読み書きを、ロジックを一切複製せず
//! `fandhe_ai::interop::npy` として **純再エクスポート**する
//! （`interop::safetensors` と同型。facade 側に型・関数・`impl` を定義しない）。
//!
//! ## 公開する 5 名（別名なし）
//!
//! | 名前 | 役割 |
//! |------|------|
//! | `NpyError` | npy／npz 共通のエラー型 |
//! | `load_npy`／`save_npy` | 単一配列の読み書き |
//! | `load_npz`／`save_npz` | `HashMap<String, Tensor<f32>>` の読み書き |
//!
//! バイト列版（`read_npy_bytes` 等 4 関数）と `Tensor` へのメソッド追加は
//! 承認範囲外のため公開しない（`tests/api_surface.rs` が機械的に固定）。
//!
//! ## 対応範囲
//!
//! - 対象は `Tensor<f32>` のみ（`<f4`／`>f4`）。他 dtype は
//!   `NpyError::UnsupportedDtype` で拒否される。
//! - npz の読み込みは STORED／DEFLATE、書き出しは STORED のみ（zip64 書き出しなし）。
//!   書き出しはキー昇順で決定的。
//! - `NpyError` は `#[non_exhaustive]`（`match` にはワイルドカード腕が必要）で、
//!   `Clone`／`PartialEq` を持たない。`Display` の文言は契約にしない。
//!
//! ## 注意（承認時点の現状維持事項）
//!
//! - **書き込みは原子的ではない**（P5）: `save_npy`／`save_npz` は
//!   `std::fs::write` を用いる。途中クラッシュで不完全なファイルが残りうる。
//!   一時ファイル + rename の `interop::safetensors::save_safetensors_f32` とは異なる。
//! - **読み込みのパス扱い**（P6）: symlink を辿り、ファイルサイズ上限は 1 GiB。
//!   信頼できないパスを渡してよいかは呼び出し側の責任である。ヘッダ長・npz の
//!   伸長後サイズ・CRC-32 等の検証は `tensor-core` 実装を継承し、迂回しない。
//!
//! ## 例
//!
//! `compat::Sequential::state_dict` の `HashMap<String, Tensor<f32>>` は
//! そのまま `save_npz` に渡せる。単一配列の往復は次のとおり。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::interop::npy::{load_npy, load_npz, save_npy, save_npz};
//! use std::collections::HashMap;
//!
//! let dir = std::env::temp_dir().join(format!("fandhe-ai-npy-doc-{}", std::process::id()));
//! std::fs::create_dir_all(&dir).unwrap();
//!
//! let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
//! let npy = dir.join("a.npy");
//! save_npy(&t, &npy).unwrap();
//! let back = load_npy(&npy).unwrap();
//! assert_eq!(back.shape(), t.shape());
//! assert_eq!(back.as_slice().unwrap(), t.as_slice().unwrap());
//!
//! let mut map = HashMap::new();
//! map.insert("w".to_string(), t.clone());
//! let npz = dir.join("a.npz");
//! save_npz(&map, &npz).unwrap();
//! let loaded = load_npz(&npz).unwrap();
//! assert_eq!(loaded["w"].as_slice().unwrap(), t.as_slice().unwrap());
//!
//! std::fs::remove_dir_all(&dir).unwrap();
//! ```

// `pub use` は 1 文 1 行を維持する（`tests/api_surface.rs` が行単位で走査する契約。
// `interop/safetensors.rs` と同じ理由）。
pub use fandhe_ai_tensor_core::io::NpyError;
pub use fandhe_ai_tensor_core::io::npy::{load_npy, save_npy};
pub use fandhe_ai_tensor_core::io::npz::{load_npz, save_npz};
