//! ローカルモデルレジストリ公開面（イシュー #2087・親 #2082）。
//!
//! 利用者がホームディレクトリ配下に手動配置したプリトレイン重みを、
//! 名前・バージョンで一元管理し同期ロードする入口 [`ModelRegistry`] を
//! 提供する。数値経路（`Op`／`BackendOps`／VJP／カーネル）には一切
//! 触れず、ホスト側のパス管理と既存の
//! [`crate::interop::safetensors::load_safetensors_f32`]（#2019）への
//! 委譲のみで完結する（REQ-12「任意 `BackendOps` 注入の公開 API を
//! 設けない」と矛盾しない）。
//!
//! # ディレクトリレイアウト（規定）
//!
//! ```text
//! <cache_dir>/                      既定 $HOME/.fandhe-ai/models
//!                                    （Windows は $USERPROFILE。
//!                                    [`ModelRegistry::new`] 参照）
//!   <name>/                         `[A-Za-z0-9._-]+`（先頭 '.' 不可）
//!     <version>/                    同上。semver 等の記法は解釈しない
//!                                    不透明な文字列として扱う
//!       model.safetensors           F32 テンソルのみ・キー = state_dict
//!                                    キー（転置・リネームなし。REQ-7）
//! ```
//!
//! 配置は利用者（または将来のダウンロード側。#2088 リモート取得。本
//! イシューのスコープ外）が行う。[`ModelRegistry`] はディレクトリの
//! 作成・削除を一切行わない**読み取り専用**のレジストリである。
//!
//! # `load` の戻り値型（受入条件からの逸脱・承認事項）
//!
//! イシュー受入条件の字面は `load` の戻り値要素型を単一 `Tensor` とする
//! が、本実装は [`std::collections::HashMap<String, crate::Tensor<f32>>`]
//! （state dict）を返す。safetensors ファイルは複数テンソルを保持し、
//! 消費先の [`crate::compat::Sequential::load_state_dict`] も map を
//! 取るため、単一 `Tensor` 化には恣意的な規則（どのキーを選ぶか）が
//! 必要になる。詳細・代替案（`load_tensor(name, version, key)` の追加）
//! は `docs/facade-model-registry-decision.md` を参照。
//!
//! # 非信頼入力の扱い（OWASP A01／A03。`.claude/rules/security.md`）
//!
//! `name`・`version` は利用者から渡される非信頼入力であり、パス
//! トラバーサル（`..`・絶対パス区切り混入等）を防ぐため
//! [`validate_component`] が許可文字集合（`[A-Za-z0-9._-]+`・先頭 `.`
//! 不可）でファイルシステムへ触れる前に fail-closed 検証する。
//! safetensors ファイル自体の検証（ヘッダ・dtype・shape）は
//! `crate::interop::safetensors` に一元化されており本モジュールは
//! 複製・迂回しない。
//!
//! # 対象外
//!
//! リモート取得・HF hub 連携（#2088）、`docs/model-distribution-
//! design.md` の作成（親 #2082 の成果物）、version 記法の解釈
//! （semver・hash 等）・最新版解決、`compat::Sequential` への直結
//! ラッパー（`Sequential::from_registry` 等）、非 F32 dtype、ファイル
//! サイズ上限・改竄検知（値の決定にはユーザー承認が要るため）。

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::Tensor;
use crate::interop::safetensors::{LoadError, load_safetensors_f32};

/// レイアウト規定上のファイル名（モジュール doc 参照）。
const MODEL_FILE_NAME: &str = "model.safetensors";

/// ローカルモデルレジストリの失敗を表す型付きエラー。`#[non_exhaustive]`:
/// 将来のフィールド追加（例: 改竄検知失敗）を非破壊にするため。
#[non_exhaustive]
#[derive(Debug)]
pub enum ModelError {
    /// 既定のキャッシュディレクトリを解決できない（[`ModelRegistry::new`]。
    /// Unix は `HOME`、Windows は `USERPROFILE` のいずれも未設定）。
    CacheDirUnavailable,
    /// `name`・`version` のいずれかが許可文字集合
    /// （`[A-Za-z0-9._-]+`・先頭 `.` 不可）に適合しない
    /// （パストラバーサル対策。ファイルシステムへ触れる前に拒否する）。
    InvalidComponent { kind: &'static str, value: String },
    /// `<cache_dir>/<name>/<version>/model.safetensors` が通常ファイル
    /// として存在しない。
    NotFound { name: String, version: String },
    /// safetensors デコード失敗（`crate::interop::safetensors::LoadError`
    /// を連鎖。ヘッダ不整合・未対応 dtype・shape 不整合等）。
    Load(LoadError),
    /// 上記以外の I/O 失敗（存在確認時の権限エラー等）。
    Io(std::io::Error),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::CacheDirUnavailable => {
                write!(
                    f,
                    "既定のモデルキャッシュディレクトリを解決できません（HOME／USERPROFILE 未設定）"
                )
            }
            ModelError::InvalidComponent { kind, value } => {
                write!(f, "不正な {kind}: {value:?}")
            }
            ModelError::NotFound { name, version } => {
                write!(
                    f,
                    "モデルが見つかりません（name={name}・version={version}）"
                )
            }
            ModelError::Load(e) => write!(f, "モデルのロードに失敗しました: {e}"),
            ModelError::Io(e) => write!(f, "モデルレジストリの I/O に失敗しました: {e}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Load(e) => Some(e),
            ModelError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// `name`・`version` の許可文字集合検証（OWASP A03 パストラバーサル
/// 対策）。`load`・[`ModelRegistry::available_models`] の両方が同一の
/// この関数を使う。`kind` はエラーメッセージ用の "name" / "version"。
fn validate_component(kind: &'static str, value: &str) -> Result<(), ModelError> {
    let ok = !value.is_empty()
        && !value.starts_with('.')
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(ModelError::InvalidComponent {
            kind,
            value: value.to_string(),
        })
    }
}

/// 既定キャッシュディレクトリを `home` から純粋に導出する（環境変数の
/// 読み取りは [`ModelRegistry::new`] 側で行い、本関数は環境非依存の
/// 単体テストを可能にするために切り出した private 純関数）。
fn default_cache_dir_from(home: Option<OsString>) -> Option<PathBuf> {
    home.map(|h| PathBuf::from(h).join(".fandhe-ai").join("models"))
}

/// ホームディレクトリ配下のキャッシュディレクトリを基盤とする
/// **ローカル限定**のモデルレジストリ。ディレクトリレイアウトは
/// モジュール doc の規定を参照。本レジストリはディレクトリの作成・
/// 削除を一切行わない読み取り専用の入口である。
pub struct ModelRegistry {
    root: PathBuf,
}

impl ModelRegistry {
    /// 既定のキャッシュルート（Unix: `$HOME/.fandhe-ai/models`、
    /// Windows: `$USERPROFILE/.fandhe-ai/models`）で
    /// [`ModelRegistry`] を構築する。ディレクトリは作成しない
    /// （読み取り専用のレジストリ）。`HOME`／`USERPROFILE` のいずれも
    /// 未設定な場合は [`ModelError::CacheDirUnavailable`] を返す。
    pub fn new() -> Result<Self, ModelError> {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
        let root = default_cache_dir_from(home).ok_or(ModelError::CacheDirUnavailable)?;
        Ok(Self { root })
    }

    /// キャッシュルートを明示指定して [`ModelRegistry`] を構築する
    /// （統合テスト・CI・非標準配置向け。実 `$HOME` に触れずに
    /// テストするための入口。環境変数の書き換えは `cargo test` の
    /// 並列実行でプロセスグローバル競合を起こすため採用しない）。
    pub fn with_cache_dir(dir: impl Into<PathBuf>) -> Self {
        Self { root: dir.into() }
    }

    /// キャッシュルートのパスを返す。
    pub fn cache_dir(&self) -> &Path {
        &self.root
    }

    /// `<cache_dir>/<name>/<version>/model.safetensors` を同期ロードし、
    /// state dict（キー = テンソル名）を返す。
    ///
    /// 1. `name`・`version` を [`validate_component`] でファイル
    ///    システムへ触れる前に検証する（失敗は
    ///    [`ModelError::InvalidComponent`]）。
    /// 2. パスの存在確認（`std::fs::metadata`）。存在しない、または
    ///    通常ファイルでない場合は [`ModelError::NotFound`]。権限
    ///    エラー等それ以外の I/O 失敗は [`ModelError::Io`]。
    /// 3. [`load_safetensors_f32`]（`crate::interop::safetensors`。
    ///    #2019）へ委譲する。転置・キーリネーム等の暗黙アダプタは
    ///    一切行わない（REQ-7 契約は `interop::safetensors` に一元化
    ///    済みでロジックを複製しない）。
    pub fn load(
        &self,
        name: &str,
        version: &str,
    ) -> Result<HashMap<String, Tensor<f32>>, ModelError> {
        validate_component("name", name)?;
        validate_component("version", version)?;
        let path = self.root.join(name).join(version).join(MODEL_FILE_NAME);
        match std::fs::metadata(&path) {
            // 通常ファイルとして存在する場合のみロードへ進む。
            Ok(meta) if meta.is_file() => {}
            // 存在しない、またはディレクトリ等の非ファイルは
            // 「モデルが見つからない」として丸める。
            Ok(_) => {
                return Err(ModelError::NotFound {
                    name: name.to_string(),
                    version: version.to_string(),
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ModelError::NotFound {
                    name: name.to_string(),
                    version: version.to_string(),
                });
            }
            // 権限エラー等それ以外の I/O 失敗はそのまま伝える。
            Err(e) => return Err(ModelError::Io(e)),
        }
        load_safetensors_f32(&path).map_err(ModelError::Load)
    }

    /// キャッシュルート配下に存在するモデルを列挙する。ルートが
    /// 存在しない、または読み取れない場合はレジストリが空とみなし
    /// 空 `Vec` を返す（エラー型を返さないのは戻り値型の契約による。
    /// [`load`](Self::load) 側の I/O エラー区別とは異なる扱い）。
    ///
    /// 列挙規則:
    /// - ルート直下の**ディレクトリ**で名前が
    ///   [`validate_component`] を通るものを `name` 候補とする
    ///   （ファイル・検証不合格名はスキップ）
    /// - `name` 直下のディレクトリで名前が検証を通り、かつ
    ///   `model.safetensors` が通常ファイルとして存在するものだけを
    ///   `version` として列挙する
    /// - version が 1 件もない `name` は結果に含めない
    /// - `name`・`versions` とも文字列昇順ソート（決定的出力）
    /// - ディレクトリ名は UTF-8 変換できるもののみ対象
    ///   （非 UTF-8 名はスキップ）
    pub fn available_models(&self) -> Vec<(String, Vec<String>)> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut result: Vec<(String, Vec<String>)> = Vec::new();
        for entry in entries.flatten() {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            if validate_component("name", &name).is_err() {
                continue;
            }
            let name_dir = entry.path();
            if !name_dir.is_dir() {
                continue;
            }
            let Ok(version_entries) = std::fs::read_dir(&name_dir) else {
                continue;
            };
            let mut versions: Vec<String> = Vec::new();
            for version_entry in version_entries.flatten() {
                let Ok(version) = version_entry.file_name().into_string() else {
                    continue;
                };
                if validate_component("version", &version).is_err() {
                    continue;
                }
                if version_entry.path().join(MODEL_FILE_NAME).is_file() {
                    versions.push(version);
                }
            }
            if versions.is_empty() {
                continue;
            }
            versions.sort();
            result.push((name, versions));
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_component_accepts_ascii_allowlist() {
        for value in ["a", "mlp", "v1", "1.2.3", "a_b-c.d", "A9"] {
            assert!(
                validate_component("name", value).is_ok(),
                "許可されるべき値が拒否された: {value:?}"
            );
        }
    }

    #[test]
    fn validate_component_rejects_traversal_and_unsafe_values() {
        for value in [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            ".hidden",
            "a b",
            "a\0b",
            "/etc/passwd",
        ] {
            assert!(
                validate_component("name", value).is_err(),
                "拒否されるべき値が許可された: {value:?}"
            );
        }
    }

    #[test]
    fn default_cache_dir_from_home_appends_fixed_suffix() {
        let home = Some(OsString::from("/home/x"));
        let dir = default_cache_dir_from(home).unwrap();
        assert_eq!(dir, PathBuf::from("/home/x/.fandhe-ai/models"));
    }

    #[test]
    fn default_cache_dir_from_none_is_none() {
        assert!(default_cache_dir_from(None).is_none());
    }
}
