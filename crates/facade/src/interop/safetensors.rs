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
//! ## dtype 混在ファイルの読込（イシュー #3845・設計記録 §13 案 C・読込側）
//!
//! HF Hub の bf16／f16 チェックポイントや f32 重み + i64 buffer の混在ファイルは
//! 既存の `*_f32` では [`LoadError::UnsupportedDtype`] で拒否される（不変）。
//! 別名の [`load_safetensors_typed_from_bytes`]／[`load_safetensors_typed`] は
//! F32／F16／BF16／F64／I32／I64／BOOL の 7 種を [`SafetensorsTensor`] として bit 単位で
//! 読み、それ以外の dtype と BOOL の 0／1 以外は拒否する。f32 への昇格は明示 opt-in の
//! [`SafetensorsTensor::to_f32`]（`half` は再エクスポートしないため dtype 判定は
//! [`SafetensorsTensor::dtype`] を使う）。書出側は #3846。
//!
//! ```
//! use fandhe_ai::interop::safetensors::{
//!     SafetensorsDtype, SafetensorsTensor, load_safetensors_typed_from_bytes,
//! };
//!
//! // F64 [1.5] + I64 [7] + BOOL [1, 0] を手組みした safetensors バイト列。
//! let header = br#"{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"b":{"dtype":"I64","shape":[1],"data_offsets":[8,16]},"c":{"dtype":"BOOL","shape":[2],"data_offsets":[16,18]}}"#;
//! let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
//! bytes.extend_from_slice(header);
//! bytes.extend_from_slice(&1.5f64.to_le_bytes());
//! bytes.extend_from_slice(&7i64.to_le_bytes());
//! bytes.extend_from_slice(&[1, 0]);
//!
//! let m = load_safetensors_typed_from_bytes(&bytes).unwrap();
//! assert_eq!(m["a"].dtype(), SafetensorsDtype::F64);
//! assert!(matches!(m["b"], SafetensorsTensor::I64(_)));
//! assert_eq!(m["a"].to_f32().unwrap().host_slice()[0], 1.5);
//! assert_eq!(m["c"].to_f32().unwrap().host_slice().to_vec(), vec![1.0, 0.0]);
//!
//! let t = fandhe_ai::Tensor::<f32>::new(vec![0.5, 2.0], &[2]).unwrap();
//! let bf = SafetensorsTensor::from_f32(&t, SafetensorsDtype::BF16).unwrap();
//! assert_eq!(bf.to_f32().unwrap().host_slice().to_vec(), vec![0.5, 2.0]);
//! ```
//!
//! ## 対象外
//!
//! `compat::callbacks::ModelCheckpoint` からの薄いラッパー結線
//! （[`crate::compat::ModelCheckpoint::to_file`]）は #2073 で実装済み。
//! `compat::Sequential` の層構成シリアライズ（manifest.json ＋
//! safetensors の組み合わせによる `save_model`／`load_model`）は、
//! イシュー #2369（親 #2362）で `compat::save_model`／`compat::load_model` として
//! 公開済み（全 50 層・BatchNorm の buffer・compile 状態まで #2373 で実装完了。
//! 設計は `docs/compat-model-io-decision.md`。本モジュールの
//! `save_safetensors_f32_to_bytes`／`load_safetensors_f32_from_bytes` を再利用する。
//! 本モジュール自体への変更はなく、案 A の素の再エクスポートのみのまま）。F32 以外の
//! dtype・入力サイズ上限の導入・`st_load`／`st_save` 本体ロジックの
//! 変更も対象外。

// `pub use` は 1 文 1 行を維持する（複数行折返し禁止。
// `tests/api_surface.rs::interop_safetensors_reexports_exactly_expected_surface`
// が `pub use` を行単位で走査する契約に合わせる。`src/optim.rs` と
// 同じ理由）。
pub use fandhe_ai_onnx_interop::st_load::load_safetensors_typed;
pub use fandhe_ai_onnx_interop::st_load::load_safetensors_typed_from_bytes;
pub use fandhe_ai_onnx_interop::st_load::require_keys_typed;
pub use fandhe_ai_onnx_interop::st_load::{LoadError, load_safetensors_f32};
pub use fandhe_ai_onnx_interop::st_load::{SafetensorsDtype, SafetensorsTensor};
pub use fandhe_ai_onnx_interop::st_load::{load_safetensors_f32_from_bytes, require_keys};
pub use fandhe_ai_onnx_interop::st_save::save_safetensors_f32_to_bytes;
pub use fandhe_ai_onnx_interop::st_save::{SaveError, save_safetensors_f32};
