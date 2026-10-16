//! `download` が受け付ける filename の検証（是正案 A。設計 §9.4）。
//!
//! facade の単一ファイル制約（`<cache_dir>/<name>/<version>/model.safetensors`）と合わせ、
//! リポジトリ直下の `model.safetensors` との完全一致のみを許可する。サブフォルダ・`..`・
//! 絶対パス・制御文字・shard 名はすべて接続前に拒否され、パストラバーサルとキャッシュキー衝突が
//! 構造的に起きない。引数自体は残す（将来の是正案 B への非破壊拡張のため）。

use crate::HfError;

/// 許可される唯一の filename。
pub(crate) const MODEL_FILENAME: &str = "model.safetensors";

pub(crate) fn validate(filename: &str) -> Result<(), HfError> {
    if filename == MODEL_FILENAME {
        Ok(())
    } else {
        Err(HfError::InvalidFilename {
            reason: "model.safetensors（リポジトリ直下）以外は許可されない",
        })
    }
}
