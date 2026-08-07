//! safetensors ワイヤフォーマットのパースと自作 `Tensor<f32>` へのマッピング
//! （TASK-7.1a・#73・REQ-7）。
//!
//! `onnx-interop` クレート（本クレート）は ONNX / safetensors 相互運用層
//! （`crate` ルート `lib.rs` 参照）を担い、本モジュールはその safetensors
//! 経路を提供する。`safetensors` クレートはワイヤフォーマットの読み書き
//! のみに用い、`tensor-core::Tensor<f32>` へのマッピングは自作する
//! （`.claude/rules/deps-policy.md`）。呼び出し元は PyTorch で保存した
//! 重みファイル（`.safetensors`）を本モジュール経由でロードし、
//! `tensor-core::Tensor::transpose_2d()` 等の後続 API へ渡す想定である。
//!
//! ## REQ-7 が明示する 2 つの契約（`docs/spec/04-requirements.md`）
//!
//! 1. **暗黙アダプタを設けない**: PyTorch `nn.Linear.weight` は
//!    `[out_features, in_features]` で保存される。matmul の右辺
//!    （`[in_features, out_features]`）へ変換する転置は本モジュールでは
//!    一切行わない。呼び出し側が `Tensor::transpose_2d()` を明示的に
//!    呼ぶ（PoC-v2-6 `mlp.rs` の `Mlp::from_safetensors` と同型の設計）。
//! 2. **無言 skip 禁止**: 期待するキー集合に対する充足検査
//!    （[`require_keys`]）は不足キーを**全件**収集して返す。呼び出し元が
//!    最初の 1 件だけを見て見落とすことを避ける。
//!
//! ## 検証順序（OWASP A03。`.claude/rules/security.md`）
//!
//! 各テンソルについて、データ変換（バイト列 → `f32`）より**前**に
//! (1) `SafeTensors::deserialize` によるヘッダ・レイアウト整合検査 →
//! (2) dtype 検査（F32 以外は即エラー） → (3) データ長が shape の要素数積
//! と一致するかの検査 → (4) `Tensor::new` の shape 検査（要素数積の
//! オーバーフローは `tensor-core::ShapeError::ElementCountOverflow` に
//! 委譲） の順で行う。

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::SafeTensors;
use tensor_core::{ShapeError, Tensor};

/// safetensors ロード経路の型付きエラー。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスク（ONNX 経路等）で
/// variant が増えても呼び出し側の網羅的 match を破壊しないため
/// （`tensor_core::ShapeError` と同じ方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum LoadError {
    /// ファイル読み込み（`load_safetensors_f32`）の I/O エラー。
    Io(std::io::Error),
    /// `SafeTensors::deserialize`／`tensor()` が返すヘッダ・レイアウト
    /// 不整合（バイト列が safetensors フォーマットとして壊れている）。
    SafetensorsFormat(String),
    /// [`require_keys`] が検出した不足キー（**全件**。無言 skip 禁止の
    /// 契約を診断性の面から強化する）。
    MissingKeys(Vec<String>),
    /// F32 以外の dtype（本イシューのスコープ外。明示エラーとして扱う）。
    UnsupportedDtype { key: String, dtype: String },
    /// テンソルの生データ長（バイト数 / 4）が safetensors ヘッダの shape
    /// が要求する要素数積と一致しない（ヘッダとデータ本体の不整合）。
    ///
    /// 注記: safetensors 0.7.0 は `deserialize` 内部の
    /// `Metadata::validate` で同じ不変条件（`data_offsets` 区間長と
    /// shape 要素数積 × dtype サイズの一致）を検査するため、通常この
    /// variant に到達する前に [`LoadError::SafetensorsFormat`] として
    /// 弾かれる（`tests/st_load.rs` の
    /// `truncated_data_is_rejected_before_conversion` で確認済み）。
    /// 本 variant は将来の safetensors バージョン変更や `SafeTensors`
    /// 以外の入力経路が追加された場合に備えた多層防御として保持する
    /// （OWASP A03「外部入力を信頼しない」。`.claude/rules/security.md`）。
    DataLengthMismatch {
        key: String,
        expected: usize,
        actual: usize,
    },
    /// `Tensor::new` が返す shape 検査エラー（要素数積オーバーフロー等）。
    Shape(ShapeError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "IO エラー: {e}"),
            LoadError::SafetensorsFormat(e) => write!(f, "safetensors 形式エラー: {e}"),
            LoadError::MissingKeys(keys) => {
                write!(f, "必須テンソルキーが見つかりません: {}", keys.join(", "))
            }
            LoadError::UnsupportedDtype { key, dtype } => {
                write!(f, "未対応の dtype（key={key}）: {dtype}")
            }
            LoadError::DataLengthMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "テンソルデータ長不整合（key={key}）: shape が要求する要素数 {expected}、実データ要素数 {actual}"
            ),
            LoadError::Shape(e) => write!(f, "shape エラー: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// safetensors バイト列を `name -> Tensor<f32>` のマップへ変換する。
///
/// [`load_safetensors_f32`] から呼ばれる本体実装。バイト列を直接受け取る
/// 形にしてあるのは、テスト（`tests/st_load.rs`）が破損データ・
/// 未対応 dtype 等の異常系をファイル I/O なしに構築できるようにするため。
///
/// dtype が F32 以外のテンソルは [`LoadError::UnsupportedDtype`] として
/// 報告する（無言でスキップしない。呼び出し元がキー不足を検知できる
/// ようにするため）。転置・キーリネームは一切行わない
/// （モジュール冒頭ドキュメント参照）。
pub fn load_safetensors_f32_from_bytes(
    bytes: &[u8],
) -> Result<HashMap<String, Tensor<f32>>, LoadError> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| LoadError::SafetensorsFormat(format!("{e:?}")))?;

    let mut out = HashMap::new();
    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|e| LoadError::SafetensorsFormat(format!("{name}: {e:?}")))?;

        // (2) dtype 検査をデータ変換より先に行う（無言 skip 禁止）。
        if view.dtype() != Dtype::F32 {
            return Err(LoadError::UnsupportedDtype {
                key: name.to_string(),
                dtype: format!("{:?}", view.dtype()),
            });
        }

        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();

        // (3) データ長検査をバイト変換より先に行う。F32 は 4 バイト固定
        // （`chunks_exact(4)` は端数バイトを黙って切り捨てるため、事前に
        // 長さの整合を確認しないと末尾破損データを見逃す）。
        let expected_elems: usize = shape.iter().product();
        if raw.len() % 4 != 0 || raw.len() / 4 != expected_elems {
            return Err(LoadError::DataLengthMismatch {
                key: name.to_string(),
                expected: expected_elems,
                actual: raw.len() / 4,
            });
        }

        // safetensors のテンソルデータは仕様上常にリトルエンディアン。
        // `unsafe` は使わず `from_le_bytes` で変換する。
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // (4) shape 検査（要素数積オーバーフロー等）は Tensor::new に委譲。
        let tensor = Tensor::new(data, &shape).map_err(LoadError::Shape)?;
        out.insert(name.to_string(), tensor);
    }
    Ok(out)
}

/// `path` の safetensors ファイルを読み込み、`name -> Tensor<f32>` の
/// マップを返す（`std::fs::read` + [`load_safetensors_f32_from_bytes`] の
/// 薄い合成）。
pub fn load_safetensors_f32(path: &Path) -> Result<HashMap<String, Tensor<f32>>, LoadError> {
    let bytes = std::fs::read(path).map_err(LoadError::Io)?;
    load_safetensors_f32_from_bytes(&bytes)
}

/// 期待するキー集合がすべて `map` に存在するか検査する
/// （REQ-7「無言 skip 禁止」契約の実体）。
///
/// 不足キーは**全件**収集して [`LoadError::MissingKeys`] で返す
/// （最初の 1 件のみ報告すると、複数キーが不足している場合に
/// 呼び出し元が 1 件ずつしか気づけず修正が後手に回るため）。
pub fn require_keys(map: &HashMap<String, Tensor<f32>>, keys: &[&str]) -> Result<(), LoadError> {
    let missing: Vec<String> = keys
        .iter()
        .filter(|k| !map.contains_key(**k))
        .map(|k| (*k).to_string())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(LoadError::MissingKeys(missing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_keys_reports_all_missing_not_just_the_first() {
        let map: HashMap<String, Tensor<f32>> = HashMap::new();
        let err = require_keys(&map, &["fc1.weight", "fc1.bias"]).unwrap_err();
        match err {
            LoadError::MissingKeys(keys) => {
                assert_eq!(keys, vec!["fc1.weight".to_string(), "fc1.bias".to_string()]);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn require_keys_ok_when_all_present() {
        let mut map: HashMap<String, Tensor<f32>> = HashMap::new();
        map.insert("fc1.weight".to_string(), Tensor::zeros(&[1]).unwrap());
        require_keys(&map, &["fc1.weight"]).unwrap();
    }
}
