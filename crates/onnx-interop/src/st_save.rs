//! `tensor-core::Tensor<f32>` から safetensors ワイヤフォーマットへの書き出し
//! （TASK-7.1c・#197・REQ-7）。[`st_load`](crate::st_load) と対になる保存経路。
//!
//! 親イシュー #196（学習最小構成：重み保存・チェックポイント）の受け入れ条件
//! 「save→load ラウンドトリップで bit 一致」の基盤となる。ラウンドトリップ・
//! チェックポイントのテストスイート整備本体は兄弟イシュー #198 の担当であり、
//! 本モジュールは書き出し実装とその単体・統合テストまでを扱う
//! （`.claude/rules/out-of-scope-tracking.md`）。
//!
//! ## REQ-7 が明示する契約（`st_load` と対称）
//!
//! 1. **暗黙アダプタを設けない**: キー・形状・値を渡されたまま書き出す。転置・
//!    キーリネームは一切行わない。PyTorch 慣習キー（`<module>.weight` は
//!    `[out_features, in_features]` 等）に合わせるかどうかは呼び出し側の責務
//!    （[`st_load`](crate::st_load) モジュール冒頭ドキュメント参照）。
//! 2. **dtype は F32 のみ**: 本関数群は `Tensor<f32>` にのみ型付けされるため
//!    型レベルで保証される。f16 等の追加 dtype 対応はイシュー #274 で追跡済み
//!    （本イシューのスコープ外）。
//!
//! ## 決定的出力
//!
//! `HashMap` の反復順は非決定的なため、キーを**昇順ソート**してから
//! `safetensors::tensor::serialize` へ渡す。同一の入力マップは常に同一の
//! バイト列を生成する（#198 の bit 一致検証・sha256 改竄検知の前提となる。
//! `.claude/rules/security.md` A08）。
//!
//! ## エンディアン
//!
//! safetensors 仕様上データは常にリトルエンディアンである。`unsafe` は
//! 使わず `f32::to_le_bytes` で変換する（`st_load` の `from_le_bytes` と対称）。
//!
//! ## ファイル書き込みの整合性（OWASP A08）
//!
//! [`save_safetensors_f32`] は同一ディレクトリの一時ファイルへ書き込んでから
//! `rename` する。書き込み途中でクラッシュしても、壊れたチェックポイントが
//! 正規パス（呼び出し元が期待するファイル名）に露出しないようにするため
//! （チェックポイント用途での整合性確保。`.claude/rules/security.md`）。
//!
//! ## PyTorch 側での手動検証手順
//!
//! CI（self-hosted・Python/PyTorch 非搭載）では実行できないため、生成物の
//! 実機検証は手動で行う。
//!
//! ```text
//! from safetensors.torch import load_file
//! tensors = load_file("output.safetensors")
//! print({k: v.shape for k, v in tensors.items()})
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize};
use tensor_core::Tensor;

/// safetensors 書き出し経路の型付きエラー。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスクで variant が増えても
/// 呼び出し側の網羅的 match を破壊しないため（[`st_load::LoadError`]
/// (crate::st_load::LoadError) と同じ方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum SaveError {
    /// ファイル書き込み（一時ファイル書き出し・`rename` を含む）の I/O エラー。
    Io(std::io::Error),
    /// `safetensors::tensor::serialize` が返すエラー（キー・レイアウト不整合等）。
    SafetensorsFormat(String),
    /// `contiguous()` 化後もデータスライスを取得できない内部不整合。
    ///
    /// 通常到達しない防御的 variant であり、本番経路で `unwrap()` / `expect()`
    /// を使わない規約（`.claude/rules/coding-rust.md`）のための受け皿として
    /// 保持する。
    DataUnavailable { key: String },
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SaveError::Io(e) => write!(f, "IO エラー: {e}"),
            SaveError::SafetensorsFormat(e) => write!(f, "safetensors 形式エラー: {e}"),
            SaveError::DataUnavailable { key } => {
                write!(f, "テンソルデータを取得できません（key={key}）")
            }
        }
    }
}

impl std::error::Error for SaveError {}

/// `name -> Tensor<f32>` のマップを safetensors バイト列へ変換する。
///
/// [`save_safetensors_f32`] から呼ばれる本体実装。バイト列を直接返す形に
/// してあるのは、テスト（`tests/st_save.rs`）がファイル I/O なしにヘッダ
/// 構造・決定性を検証できるようにするため（[`st_load::load_safetensors_f32_from_bytes`]
/// (crate::st_load::load_safetensors_f32_from_bytes) と対称の設計）。
///
/// `metadata` は safetensors の `__metadata__`（例: `{"format": "pt"}`）への
/// pass-through。PyTorch 側のロードに必須ではないため [`save_safetensors_f32`]
/// は `None` 固定の便宜関数として提供する。
///
/// 転置・キーリネームは一切行わない（モジュール冒頭ドキュメント参照）。
/// 非 contiguous（view）テンソルは [`Tensor::contiguous`] で論理 row-major に
/// 詰め直してから書き出す。
pub fn save_safetensors_f32_to_bytes(
    tensors: &HashMap<String, Tensor<f32>>,
    metadata: Option<HashMap<String, String>>,
) -> Result<Vec<u8>, SaveError> {
    // HashMap の反復順は非決定的なため、キーを昇順ソートしてから
    // TensorView を構築する。同一入力マップは常に同一バイト列を生成する
    // （決定的出力。モジュール冒頭ドキュメント参照）。
    let mut keys: Vec<&String> = tensors.keys().collect();
    keys.sort();

    // TensorView は借用データを保持するため、バイト列の所有権を先に
    // 全キー分確保してから views を構築する（借用チェッカ制約への対応）。
    let mut byte_buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(keys.len());
    for key in keys {
        let tensor = &tensors[key];
        let shape = tensor.shape().to_vec();

        // as_slice() は contiguous な場合のみ Some を返す（転置 view 等では
        // None）。非 contiguous の場合は contiguous() で論理 row-major に
        // 詰め直してからバイト列化する（st_load の逆変換に相当）。
        let owned;
        let data: &[f32] = match tensor.as_slice() {
            Some(s) => s,
            None => {
                owned = tensor.contiguous();
                owned.as_slice().ok_or_else(|| SaveError::DataUnavailable {
                    key: key.to_string(),
                })?
            }
        };

        // safetensors のテンソルデータは仕様上常にリトルエンディアン。
        // `unsafe` は使わず `to_le_bytes` で変換する（st_load の
        // `from_le_bytes` と対称）。
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        byte_buffers.push((key.to_string(), shape, bytes));
    }

    let views: Vec<(String, TensorView<'_>)> = byte_buffers
        .iter()
        .map(|(key, shape, bytes)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| SaveError::SafetensorsFormat(format!("{key}: {e:?}")))?;
            Ok((key.clone(), view))
        })
        .collect::<Result<Vec<_>, SaveError>>()?;

    serialize(views, metadata).map_err(|e| SaveError::SafetensorsFormat(format!("{e:?}")))
}

/// `path` へ safetensors ファイルとして書き出す
/// （[`save_safetensors_f32_to_bytes`] + 一時ファイル書き込み + `rename` の薄い合成）。
///
/// `metadata` は常に `None`（チェックポイントのメタ情報が必要になった場合は
/// [`save_safetensors_f32_to_bytes`] を直接使う）。
///
/// 一時ファイルは `path` と同一ディレクトリに書き込んでから `rename` する
/// （モジュール冒頭ドキュメント「ファイル書き込みの整合性」参照。同一
/// ファイルシステム内の `rename` は POSIX 上 atomic なため、途中クラッシュ
/// でも正規パスには完全なファイルか元のファイルのいずれかのみが存在する）。
pub fn save_safetensors_f32(
    path: &Path,
    tensors: &HashMap<String, Tensor<f32>>,
) -> Result<(), SaveError> {
    let bytes = save_safetensors_f32_to_bytes(tensors, None)?;

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    // プロセス ID を混ぜて同一ディレクトリへの並行書き出し時の一時ファイル
    // 名衝突を避ける。
    let tmp_path = dir.join(format!(".{file_name}.tmp.{}", std::process::id()));

    std::fs::write(&tmp_path, &bytes).map_err(SaveError::Io)?;
    std::fs::rename(&tmp_path, path).map_err(SaveError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_map_round_trips() {
        let tensors: HashMap<String, Tensor<f32>> = HashMap::new();
        let bytes = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
        let loaded = crate::st_load::load_safetensors_f32_from_bytes(&bytes).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn single_tensor_round_trips_with_bit_exact_values() {
        let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        );
        let bytes = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
        let loaded = crate::st_load::load_safetensors_f32_from_bytes(&bytes).unwrap();
        let out = &loaded["w"];
        assert_eq!(out.shape(), &[2, 2]);
        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (a, b) in out.as_slice().unwrap().iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn output_is_deterministic_across_calls() {
        let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
        tensors.insert("b".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
        tensors.insert("a".to_string(), Tensor::new(vec![2.0], &[1]).unwrap());
        tensors.insert("c".to_string(), Tensor::new(vec![3.0], &[1]).unwrap());

        let bytes1 = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
        let bytes2 = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
        assert_eq!(bytes1, bytes2);
    }
}
