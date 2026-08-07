//! ロード済みキーマップに対する期待キー集合の充足検査（TASK-7.1b・REQ-7・イシュー #74）。
//!
//! v1 PoC-6 で「重みキー名の対応付けを誤ると無言で skip され、欠損に気づけない」詰まりが
//! 実測されている（`docs/spec/03-poc/poc-v2-6-interop/README.md` 詰まりポイント #2）。
//! `require_keys()` はこの詰まりへの対策として、safetensors ロード結果のキーマップに対して
//! 期待キー集合が全て存在するかを検査し、不足があれば型付きエラーで報告する
//! （無言 skip は禁止。`.claude/rules/coding-rust.md` 本番経路 `unwrap()`/`expect()` 禁止方針とも整合）。
//!
//! # 呼び出し契約
//!
//! safetensors ロード直後・モデル構築前に呼び出し、`Err` を伝播させる。呼び出し側が
//! 「このモデルが必要とするキー一覧」を明示的に渡す設計とし、余剰キー（期待集合に
//! 無いキー）の検出は本関数のスコープ外（TASK-7.1b はキー**不足**検査のみを要求する。
//! 余剰キー検出の要否は別途 Issue で追跡する）。
//!
//! # #73（TASK-7.1a）との関係
//!
//! 本イシュー着手時点で #73（safetensors パース→自作テンソルマッピング、`st_load` 相当
//! モジュール）は本クレートへ未マージだった。そのため `require_keys()` は特定のテンソル型
//! （`tensor-core::Tensor` 等）に結合させず、キーマップの値型 `V` を汎用化した独立実装と
//! している。#73 マージ後は `st_load::load_safetensors_f32()` が返す
//! `HashMap<String, Tensor<f32>>` に対してもそのまま呼び出せる（`V = Tensor<f32>` として
//! 単相化されるのみで API 変更は不要）。#73 の `LoadError` と本モジュールの [`LoadError`] が
//! 別型として並存する場合は、統合時にどちらか一方へ寄せる整理が必要になる（対象外事項として
//! PR 本文に記録する）。
//!
//! # PoC からの改善点
//!
//! 参照実装（`docs/spec/03-poc/poc-v2-6-interop/code/rust/src/st_load.rs:85-95`）は最初に
//! 見つかった不足キー 1 件のみを報告するが、本実装は**全不足キーを収集**し
//! [`LoadError::MissingKeys`] として一括報告する（診断性向上。検査の緩和ではない）。
//! 不足キー一覧は決定的な順序（ソート済み）で格納する。

use std::collections::HashMap;
use std::fmt;

/// `require_keys()` の失敗を表す型付きエラー。
///
/// `Display` + `std::error::Error` を実装し、呼び出し側が `?` でそのまま伝播できる
/// （本番経路での `unwrap()`/`expect()` を避けるため。`.claude/rules/coding-rust.md`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// 期待キー集合のうちマップに存在しなかったキー名の一覧（ソート済み・重複除去済み）。
    MissingKeys(Vec<String>),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::MissingKeys(keys) => {
                write!(f, "必須テンソルキーが見つかりません: {}", keys.join(", "))
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// 期待するキー集合がすべてマップに存在するか検査する。
///
/// safetensors ロード直後・モデル構築前に呼び出す契約（本モジュールの `//!` 参照）。
/// 不足キーがあれば全件を収集し `LoadError::MissingKeys` として返す（無言 skip 禁止。
/// v1 PoC-6 詰まりポイント #2 対策・TASK-7.1b・REQ-7）。
///
/// キーマップの値型 `V` は汎用化しており、`tensor-core::Tensor<f32>` 等どの値型の
/// `HashMap<String, V>` に対しても呼び出せる（#73 マージ前後で API 変更不要にするため。
/// 本モジュールの `//!` 「#73 との関係」節を参照）。
pub fn require_keys<V>(map: &HashMap<String, V>, keys: &[&str]) -> Result<(), LoadError> {
    let mut missing: Vec<String> = keys
        .iter()
        .filter(|k| !map.contains_key(**k))
        .map(|k| (*k).to_string())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    missing.sort();
    missing.dedup();
    Err(LoadError::MissingKeys(missing))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_key_is_reported_not_skipped() {
        let map: HashMap<String, ()> = HashMap::new();
        let err = require_keys(&map, &["fc1.weight"]).unwrap_err();
        assert_eq!(err, LoadError::MissingKeys(vec!["fc1.weight".to_string()]));
    }

    #[test]
    fn all_keys_present_is_ok() {
        let mut map: HashMap<String, ()> = HashMap::new();
        map.insert("fc1.weight".to_string(), ());
        map.insert("fc1.bias".to_string(), ());
        assert!(require_keys(&map, &["fc1.weight", "fc1.bias"]).is_ok());
    }

    #[test]
    fn multiple_missing_keys_are_reported_in_deterministic_order() {
        let map: HashMap<String, ()> = HashMap::new();
        let err = require_keys(&map, &["fc2.weight", "fc1.bias", "fc1.weight"]).unwrap_err();
        match err {
            LoadError::MissingKeys(keys) => {
                assert_eq!(
                    keys,
                    vec![
                        "fc1.bias".to_string(),
                        "fc1.weight".to_string(),
                        "fc2.weight".to_string(),
                    ]
                );
            }
        }
    }

    #[test]
    fn duplicate_expected_keys_do_not_duplicate_missing_report() {
        let map: HashMap<String, ()> = HashMap::new();
        let err = require_keys(&map, &["fc1.weight", "fc1.weight"]).unwrap_err();
        assert_eq!(err, LoadError::MissingKeys(vec!["fc1.weight".to_string()]));
    }

    #[test]
    fn empty_expected_keys_is_ok_even_for_empty_map() {
        let map: HashMap<String, ()> = HashMap::new();
        assert!(require_keys(&map, &[]).is_ok());
    }
}
