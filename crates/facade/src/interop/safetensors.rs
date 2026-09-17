//! safetensors save／load 公開面（イシュー #2019・
//! `docs/facade-safetensors-exposure-decision.md` §11 案 A「素の
//! 再エクスポート」の実装）。
//!
//! `fandhe_ai_onnx_interop`（内部クレート。crates.io 公開名
//! `fandhe-ai-onnx-interop`）の `st_load`／`st_save` モジュールが提供
//! する safetensors ワイヤフォーマットの読み書き
//! （`tensor-core::Tensor<f32>` へのマッピングは自作。`safetensors`
//! クレートはワイヤフォーマット処理のみに用いる。
//! `.claude/rules/deps-policy.md`）を、ロジックを一切複製せずそのまま
//! `fandhe_ai::interop::safetensors` として facade から公開する。
//! `onnx.rs`（[`crate::interop::onnx`]）が薄いラッパー型で包むのに対し、
//! 本モジュールは型・関数の定義を一切持たない **純再エクスポート**
//! である（`fandhe_ai::optim` と同型。`src/optim.rs` モジュール doc
//! 参照）。
//!
//! ## 2 系統ある `LoadError`／`require_keys` の区別（罠）
//!
//! `fandhe_ai_onnx_interop` のクレートルートにも同名の別実装
//! （`onnx::interp` 用。private `require_keys.rs` 由来）が存在するが、
//! **本モジュールが再エクスポートするのは `st_load`／`st_save`
//! モジュール配下のもののみ**である。クレートルート直下の
//! `LoadError`／`require_keys` は safetensors の型（`Result<HashMap<..>,
//! st_load::LoadError>`）とは異なる型であり、両者を混同しないこと
//! （`tests/api_surface.rs::interop_safetensors_reexports_exactly_expected_surface`
//! が `st_load::`／`st_save::` 接頭辞のみを許可することで機械的に固定
//! する）。
//!
//! ## REQ-7 契約（変更なし。`crates/onnx-interop/src/{st_load,st_save}.rs`
//! モジュール doc が正）
//!
//! 1. **暗黙アダプタなし**: PyTorch `nn.Linear.weight`
//!    （`[out_features, in_features]`）等の転置は一切行わない。呼び出し
//!    側が必要に応じて [`crate::Tensor::transpose_2d`] を明示的に呼ぶ。
//! 2. **無言 skip 禁止**: [`require_keys`] は不足キーを**全件**収集して
//!    返す。
//! 3. **決定的出力**: [`save_safetensors_f32_to_bytes`] はキーを昇順
//!    ソートしてから書き出すため、同一マップから常に同一バイト列を
//!    生成する。
//! 4. **一時ファイル + rename**: [`save_safetensors_f32`] は同一
//!    ディレクトリへ一時ファイルを書いてから `rename`（POSIX 上
//!    atomic）することで、途中クラッシュでも正規パスには完全な
//!    ファイルか元のファイルのいずれかのみが存在する。
//!
//! ## 非信頼入力の扱い（OWASP A03。`.claude/rules/security.md`）
//!
//! safetensors バイト列は非信頼な外部フォーマットである。本モジュール
//! は検証を一切迂回・複製しない。`load_safetensors_f32_from_bytes` は
//! `SafeTensors::deserialize` によるヘッダ・レイアウト整合検査 →
//! dtype 検査（F32 以外は [`LoadError::UnsupportedDtype`]） → データ長
//! 整合検査 → `Tensor::new` の shape 検査（要素数積オーバーフローは
//! [`crate::ShapeError::ElementCountOverflow`] へ委譲）の順で行う
//! （検証順序の詳細は `crates/onnx-interop/src/st_load.rs` モジュール
//! doc 参照）。`load_safetensors_f32` はパスを `std::fs::read` へそのまま
//! 渡すのみでシェル展開・パス連結は行わない。入力総バイト数の明示
//! 上限は導入していない（`onnx.rs` と同じ扱い。値の決定にはユーザー
//! 承認が要るため本モジュールのスコープ外）。
//!
//! ## `compat::Sequential::state_dict` との組み合わせ例
//!
//! ```
//! use fandhe_ai::compat::Sequential;
//! use fandhe_ai::interop::safetensors::{
//!     load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes,
//! };
//!
//! let model = Sequential::new().add_linear(4, 8, /* seed = */ 1).unwrap();
//!
//! let state = model.state_dict();
//! let bytes = save_safetensors_f32_to_bytes(&state, None).unwrap();
//! let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();
//!
//! let mut other = Sequential::new().add_linear(4, 8, /* seed = */ 2).unwrap();
//! other.load_state_dict(loaded).unwrap();
//! ```
//!
//! ## 対象外
//!
//! `compat::Sequential`／`compat::callbacks::ModelCheckpoint` にファイル
//! 保存の薄いラッパーを追加すること（`Sequential::save`／`load` 等）は
//! 本モジュールのスコープ外（案 A は素の再エクスポートのみ）。
//! `docs/compat-callbacks-design.md` §8 参照。F32 以外の dtype・入力
//! サイズ上限の導入・`st_load`／`st_save` 本体ロジックの変更も対象外。

// `pub use` は 1 文 1 行を維持する（複数行折返し禁止。
// `tests/api_surface.rs::interop_safetensors_reexports_exactly_expected_surface`
// が `pub use` を行単位で走査する契約に合わせる。`src/optim.rs` と
// 同じ理由）。
pub use fandhe_ai_onnx_interop::st_load::{LoadError, load_safetensors_f32};
pub use fandhe_ai_onnx_interop::st_load::{load_safetensors_f32_from_bytes, require_keys};
pub use fandhe_ai_onnx_interop::st_save::save_safetensors_f32_to_bytes;
pub use fandhe_ai_onnx_interop::st_save::{SaveError, save_safetensors_f32};
