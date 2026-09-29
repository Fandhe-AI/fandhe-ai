//! `compat::Sequential` の層構成ごとのディレクトリ保存・復元
//! （[`save_model`]・[`load_model`]・[`ModelIoError`]。イシュー #2369・親 #2362）。
//!
//! 役割: `dir/manifest.json`（層構成・キー・shape・参照する safetensors 名を持つ
//! 手書き JSON）と `dir/model.<gen>.safetensors`（F32 の重み。`interop::safetensors`
//! の既存バイト列 API を再利用）の 2 ファイルで [`Sequential`] を往復させる。
//! 形式・意味論・脅威モデルの正本は `docs/compat-model-io-decision.md`
//! （§4 形式・§5 意味論・§12 世代コミット方式・§13 ファイル I/O 脅威）。
//!
//! # 本バージョンで対応する範囲
//!
//! 未 `compile` の `Sequential` で、層が `add_linear`（bias あり）と
//! パラメータを持たない活性化 7 種（`add_relu`・`add_sigmoid`・`add_tanh`・
//! `add_silu`・`add_hardswish`・`add_gelu`・`add_gelu_tanh`）だけの場合に限る。
//! それ以外（他の層・`compile` 済み・BN の buffer・optimizer／AMP 状態）は
//! `save_model` が [`ModelIoError::UnsupportedModel`] で拒否し、`dir` には何も作らない
//! （全 30 層は #2370、BN buffer は #2371、compile 状態は #2372・#2373 で拡張する）。
//! manifest v1 のキー集合は先に確定済みで、拡張しても形式バージョンは上げない。
//!
//! # 書き込みの契約（世代コミット）
//!
//! - `model.<gen>.safetensors`（`<gen>` は 32 桁 16 進）は `create_new` で最終名へ直接書く。
//! - `manifest.json` は `create_new` の一時ファイルへ書いて `rename` する。
//!   この `rename` が唯一のコミット点で、途中で失敗しても既存の manifest は無傷。
//! - **既存ファイルは読まず、削除もしない**。再保存すると旧世代の
//!   `model.*.safetensors` が孤立ファイルとして残る。manifest が指していない
//!   `model.*.safetensors` は手動で削除してよい。
//! - 並行する save／load は非サポート。電源断耐性（fsync）は保証しない。
//! - Windows（非 unix）では `save_model`・`load_model` とも `ErrorKind::Unsupported` で
//!   fail-closed にする（`save_model` は `dir` への副作用より前に判定する）。
//!   理由と緩和条件は決定記録 §12.4。
//! - `load_model` の対応範囲（Linux x86_64／aarch64・macOS）は unix 全体より狭い。
//!   それ以外の unix では `save_model` が成功しても `load_model` は拒否される。
//!
//! # 非信頼入力の扱い
//!
//! `dir` の中身は非信頼として読む。`manifest.json`・`model.<gen>.safetensors` とも
//! `crate::fs_guard` の no-follow 手順（シンボリックリンク・非通常ファイル・
//! `(dev, ino)` 差し替えの拒否）で開き、`fstat` の実長を固定上限
//! （`MAX_MANIFEST_BYTES`・`MAX_MODEL_FILE_BYTES`）と比べてから読む。manifest の
//! 記載値（`safetensors_bytes`・`num_layers`・`in_features`／`out_features`）は
//! 一致確認にだけ使い、確保量の根拠にしない。JSON は serde を使わない手書きの
//! 厳格パーサ（未知キー・重複キー・過深ネスト・エスケープ・先頭ゼロ等を拒否）で読む。
//!
//! 呼び出し元: 利用者コード（`fandhe_ai::compat::{save_model, load_model}`）。
//! 呼び出し先: `Sequential`（層構成の記録・`state_dict`／`load_state_dict`）・
//! `crate::interop::safetensors`・`crate::fs_guard`。

use std::fmt;
use std::io;
use std::path::Path;

use fandhe_ai_autodiff::AutodiffError;

use super::sequential::{LayerSpec, Sequential};
use crate::fs_guard::{LeafError, MAX_MODEL_FILE_BYTES, OpenedLeaf, open_leaf_checked};
use crate::interop::safetensors::{load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes};

/// manifest のファイル名（`dir` 直下の固定名）。
const MANIFEST_FILE_NAME: &str = "manifest.json";
/// manifest の `format` 値（形式の識別子）。
const FORMAT_NAME: &str = "fandhe-ai.compat.sequential";
/// manifest の `format_version` 値。
const FORMAT_VERSION: u64 = 1;

/// manifest（JSON）のサイズ上限（バイト）。1 MiB。
///
/// 値の根拠: v1 の manifest は 1 層あたり 100〜150 B 程度で、`MAX_LAYERS`（4096 層）でも
/// 約 0.6 MiB に収まる。**候補値であり、上限値のユーザー承認（親 #2362 の承認項目 5）は
/// 実装時点で記録が確認できていない**（決定記録 §2 item 4）。
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
/// 層数の上限。4096。
///
/// 値の根拠: 既存テストに 1714 層の `Sequential` があり
/// （`sequential.rs` の `..._with_1714_layers`）、それを包含する 2 のべき乗。
/// 承認状況は `MAX_MANIFEST_BYTES` と同じ（候補値）。
const MAX_LAYERS: usize = 4096;
/// JSON のコンテナ（object／array）のネスト上限。ポリシー閾値ではなく v1 スキーマの
/// 最大ネスト（root → 配列 → 要素 object → params／shape）から導いた構造上の値。
const MAX_JSON_DEPTH: usize = 4;
/// 1 つの JSON 配列の要素数上限。`parameter_keys` は 1 層あたり最大 2 要素なので
/// `MAX_LAYERS` から導く（新しい閾値ではない）。
const MAX_ARRAY_LEN: usize = 2 * MAX_LAYERS;
/// 1 つの JSON object のキー数上限。v1 の最上位 object のキー数（10）に余裕を持たせた
/// 値で、重複キー検査の線形走査を有界にする。
const MAX_OBJECT_KEYS: usize = 16;
/// `create_new` が `AlreadyExists` を返したときの名前再生成の試行上限。
/// `crates/docs-site/src/build.rs` の同名定数と同値（決定記録 §12.3 手順 1）。
#[cfg(unix)]
const MAX_TMP_NAME_ATTEMPTS: usize = 8;

/// `save_model`／`load_model` のエラー。
///
/// `#[non_exhaustive]`（将来の variant 追加は非破壊）。
#[derive(Debug)]
#[non_exhaustive]
pub enum ModelIoError {
    /// ファイル I/O の失敗、または no-follow 手順の拒否（シンボリックリンク・
    /// 非通常ファイルは `ErrorKind::InvalidInput`）・非対応プラットフォーム
    /// （`ErrorKind::Unsupported`）。
    Io(std::io::Error),
    /// manifest の構文・スキーマ違反（未知キー・重複キー・不正な `safetensors_file` 等）。
    Manifest {
        /// 違反内容の説明。
        message: String,
    },
    /// safetensors のエンコード・デコード失敗。
    Safetensors(String),
    /// 本バージョンで保存・復元できないモデル（未対応の層・`compile` 済み等）。
    UnsupportedModel {
        /// 拒否した理由。
        reason: String,
    },
    /// manifest と safetensors の内容の不一致（バイト数・キー集合・shape）。
    Mismatch {
        /// 不一致の内容。
        message: String,
    },
    /// 層の構築・`load_state_dict` の失敗。
    Autodiff(AutodiffError),
    /// 固定上限の超過。
    TooLarge {
        /// 超過した対象。
        what: &'static str,
        /// 固定上限。
        limit: u64,
    },
}

impl fmt::Display for ModelIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelIoError::Io(e) => write!(f, "I/O エラー: {e}"),
            ModelIoError::Manifest { message } => write!(f, "manifest が不正です: {message}"),
            ModelIoError::Safetensors(m) => write!(f, "safetensors の処理に失敗しました: {m}"),
            ModelIoError::UnsupportedModel { reason } => {
                write!(f, "未対応のモデルです: {reason}")
            }
            ModelIoError::Mismatch { message } => {
                write!(f, "manifest と safetensors が一致しません: {message}")
            }
            ModelIoError::Autodiff(e) => write!(f, "モデルの構築に失敗しました: {e}"),
            ModelIoError::TooLarge { what, limit } => {
                write!(f, "{what} が固定上限 {limit} を超えています")
            }
        }
    }
}

impl std::error::Error for ModelIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelIoError::Io(e) => Some(e),
            ModelIoError::Autodiff(e) => Some(e),
            _ => None,
        }
    }
}

/// `Sequential` を `dir` へ保存する（`dir/manifest.json`＋`dir/model.<gen>.safetensors`）。
///
/// 対応範囲・世代コミット方式・既存ファイルを削除しない契約と手動掃除の手順・
/// Windows 非対応（fail-closed。理由と緩和条件は決定記録 §12.4）はモジュール doc を参照。
/// 検証（未対応の層・`compile` 済み等）は `dir` を作る前にすべて終えるため、
/// `UnsupportedModel` で失敗したときに `dir` には何も残らない。
pub fn save_model(model: &Sequential, dir: impl AsRef<Path>) -> Result<(), ModelIoError> {
    // 非 unix では `dir` への副作用（`create_dir_all` 等）より前にここで拒否する（§12.4 item 5）。
    save_platform_check()?;
    let prepared = prepare_save(model)?;
    write_prepared(dir.as_ref(), &prepared)
}

/// `save_model` が書いたディレクトリから `Sequential` を復元する。
///
/// 重みは safetensors の値を bit のまま設定し、`training` フラグも復元する。
/// 途中で失敗しても部分的に構築したモデルは返さない。非信頼入力の扱いはモジュール doc を参照。
pub fn load_model(dir: impl AsRef<Path>) -> Result<Sequential, ModelIoError> {
    load_from_dir_with_limits(dir.as_ref(), MAX_MANIFEST_BYTES, MAX_MODEL_FILE_BYTES)
}

// ---------------------------------------------------------------------
// プラットフォーム判定
// ---------------------------------------------------------------------

/// 保存が可能なプラットフォームかを判定する純関数（`unix` を引数化して
/// Linux のテストからも非 unix 側の拒否を検査できるようにする。
/// `onnx-interop` の `windows_component_reject_reason` と同型。決定記録 §12.4 item 5）。
fn save_platform_supported(is_unix: bool) -> io::Result<()> {
    if is_unix {
        Ok(())
    } else {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

/// [`save_platform_supported`] を実行環境の cfg で呼ぶ薄いラッパー。
fn save_platform_check() -> Result<(), ModelIoError> {
    save_platform_supported(cfg!(unix)).map_err(ModelIoError::Io)
}

// ---------------------------------------------------------------------
// スキーマ（層 ↔ kind・期待キー）
// ---------------------------------------------------------------------

/// 保存可能な層の manifest 上の `kind` 名。未対応の層は `None`。
fn spec_kind(spec: &LayerSpec) -> Option<&'static str> {
    match spec {
        LayerSpec::Linear { .. } => Some("linear"),
        LayerSpec::Relu => Some("relu"),
        LayerSpec::Sigmoid => Some("sigmoid"),
        LayerSpec::Tanh => Some("tanh"),
        LayerSpec::Silu => Some("silu"),
        LayerSpec::Hardswish => Some("hardswish"),
        LayerSpec::Gelu => Some("gelu"),
        LayerSpec::GeluTanh => Some("gelu_tanh"),
        LayerSpec::Unsupported { .. } => None,
    }
}

/// 層構成から導く期待キー列（`Sequential::state_dict` と同じ `"{index}.{name}"`。
/// 層順・各層内は weight → bias）。Linear の weight は `[in, out]`・bias は `[out]`。
fn expected_parameter_keys(specs: &[LayerSpec]) -> Vec<(String, Vec<usize>)> {
    let mut keys = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        if let LayerSpec::Linear {
            in_features,
            out_features,
        } = spec
        {
            keys.push((format!("{i}.weight"), vec![*in_features, *out_features]));
            keys.push((format!("{i}.bias"), vec![*out_features]));
        }
    }
    keys
}

/// `model.<32 桁の小文字 16 進>.safetensors` の完全一致パターンか
/// （パス区切り文字・`..` を含む値は必ず不一致になる。決定記録 §4）。
fn is_valid_safetensors_file_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.") else {
        return false;
    };
    let Some(generation) = rest.strip_suffix(".safetensors") else {
        return false;
    };
    generation.len() == 32
        && generation
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

// ---------------------------------------------------------------------
// 保存（検証 → 書き込み）
// ---------------------------------------------------------------------

/// 検証を終えた保存対象（`dir` への副作用なしで組み立てる）。
struct PreparedSave {
    training: bool,
    specs: Vec<LayerSpec>,
    parameter_keys: Vec<(String, Vec<usize>)>,
    safetensors: Vec<u8>,
}

/// 保存前の検証をすべて行い、書き込むバイト列と manifest の材料を返す。
/// ここで失敗すれば `dir` には一切触れていない（受入基準「`dir` に何も残らない」）。
fn prepare_save(model: &Sequential) -> Result<PreparedSave, ModelIoError> {
    if model.compiled.is_some() {
        return Err(ModelIoError::UnsupportedModel {
            reason: "compile 済みのモデルは保存できません（compile 状態の保存は未対応）".into(),
        });
    }
    let specs = model.specs();
    if specs.len() != model.layers().len() {
        return Err(ModelIoError::UnsupportedModel {
            reason: "層構成の記録と実際の層数が一致しません".into(),
        });
    }
    if specs.len() > MAX_LAYERS {
        return Err(ModelIoError::TooLarge {
            what: "層数",
            limit: MAX_LAYERS as u64,
        });
    }
    for (i, spec) in specs.iter().enumerate() {
        if let LayerSpec::Unsupported { kind } = spec {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("層 {i}（{kind}）は保存に未対応です"),
            });
        }
    }

    let expected = expected_parameter_keys(specs);
    let state = model.state_dict();
    let consistent = state.len() == expected.len()
        && expected
            .iter()
            .all(|(k, shape)| state.get(k).is_some_and(|t| t.shape() == shape.as_slice()));
    if !consistent {
        return Err(ModelIoError::UnsupportedModel {
            reason: "state_dict のキー・shape が層構成から導いた期待と一致しません".into(),
        });
    }
    let safetensors = save_safetensors_f32_to_bytes(&state, None)
        .map_err(|e| ModelIoError::Safetensors(e.to_string()))?;
    if safetensors.len() as u64 > MAX_MODEL_FILE_BYTES {
        return Err(ModelIoError::TooLarge {
            what: "model safetensors",
            limit: MAX_MODEL_FILE_BYTES,
        });
    }

    let prepared = PreparedSave {
        training: model.training(),
        specs: specs.to_vec(),
        parameter_keys: expected,
        safetensors,
    };
    // load が受理できる大きさかを、書き込み前に同じ上限で確認する（ファイル名は同じ長さの仮名）。
    let probe_name = format!("model.{}.safetensors", "0".repeat(32));
    let probe = render_manifest(&prepared, &probe_name, prepared.safetensors.len() as u64);
    if probe.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(ModelIoError::TooLarge {
            what: "manifest.json",
            limit: MAX_MANIFEST_BYTES,
        });
    }
    Ok(prepared)
}

/// manifest v1 を決定的な文字列にする（キー順固定・文字列はエスケープ不要な
/// プログラム生成の ASCII のみ）。`compiled` は `null`・`buffer_keys` は `[]`
/// （本バージョンの対応範囲。キー集合自体は将来拡張のため確定済み）。
fn render_manifest(p: &PreparedSave, safetensors_file: &str, safetensors_bytes: u64) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{{\"format\":\"{FORMAT_NAME}\",\"format_version\":{FORMAT_VERSION},\"training\":{},\"num_layers\":{},\"layers\":[",
        p.training,
        p.specs.len()
    ));
    for (i, spec) in p.specs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let kind = spec_kind(spec).unwrap_or("unsupported");
        let params = match spec {
            LayerSpec::Linear {
                in_features,
                out_features,
            } => format!("{{\"in_features\":{in_features},\"out_features\":{out_features}}}"),
            _ => "{}".to_string(),
        };
        s.push_str(&format!(
            "{{\"index\":{i},\"kind\":\"{kind}\",\"params\":{params}}}"
        ));
    }
    s.push_str("],\"parameter_keys\":[");
    for (i, (key, shape)) in p.parameter_keys.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
        s.push_str(&format!(
            "{{\"key\":\"{key}\",\"shape\":[{}]}}",
            dims.join(",")
        ));
    }
    s.push_str(&format!(
        "],\"buffer_keys\":[],\"safetensors_file\":\"{safetensors_file}\",\"safetensors_bytes\":{safetensors_bytes},\"compiled\":null}}"
    ));
    s
}

/// 非 unix の書き込み経路。`save_platform_check` が先に拒否するため通常は到達しないが、
/// 到達しても `dir` に触れず fail-closed にする。
#[cfg(not(unix))]
fn write_prepared(dir: &Path, prepared: &PreparedSave) -> Result<(), ModelIoError> {
    let _ = (dir, prepared);
    Err(ModelIoError::Io(io::Error::from(
        io::ErrorKind::Unsupported,
    )))
}

/// 世代コミット方式の書き込み（決定記録 §12.3 手順 1〜3）。unix 限定なのは、
/// 自己所有の一時ファイルの削除（§13.6）が `(dev, ino)` 照合（`MetadataExt`）に依るため。
#[cfg(unix)]
fn write_prepared(dir: &Path, p: &PreparedSave) -> Result<(), ModelIoError> {
    use std::io::Write;

    std::fs::create_dir_all(dir).map_err(ModelIoError::Io)?;

    // 手順 1: safetensors は最終名へ create_new で直接書く。衝突したら既存エントリに
    // 触れずに名前を作り直す。書き込みに失敗しても孤立ファイルとして残す（自動削除しない）。
    let (mut st_file, st_name) = create_new_with_retry(
        dir,
        || format!("model.{}.safetensors", generation_id()),
        MAX_TMP_NAME_ATTEMPTS,
    )?;
    st_file
        .write_all(&p.safetensors)
        .map_err(ModelIoError::Io)?;

    // 手順 2〜3: manifest は一時ファイルへ書いて rename する（唯一のコミット点）。
    let manifest = render_manifest(p, &st_name, p.safetensors.len() as u64);
    let (mut tmp_file, tmp_name) =
        create_new_with_retry(dir, tmp_manifest_name, MAX_TMP_NAME_ATTEMPTS)?;
    let tmp_path = dir.join(&tmp_name);
    let committed = tmp_file
        .write_all(manifest.as_bytes())
        .and_then(|()| std::fs::rename(&tmp_path, dir.join(MANIFEST_FILE_NAME)));
    if let Err(e) = committed {
        // 削除してよいのは、この呼び出しが作って fd を保持している一時ファイルだけ（§13.6）。
        remove_own_tmp(&tmp_file, &tmp_path);
        return Err(ModelIoError::Io(e));
    }
    Ok(())
}

/// `dir` 直下に `next_name()` の名前で `create_new` する。`AlreadyExists`（既存エントリの
/// 種類は問わない）は既存に一切触れずに名前を作り直し、`max_attempts` 回で諦める
/// （`Io(AlreadyExists)`。決定記録 §12.3 手順 1・2）。衝突を注入できるよう名前生成を引数にする。
#[cfg(unix)]
fn create_new_with_retry(
    dir: &Path,
    mut next_name: impl FnMut() -> String,
    max_attempts: usize,
) -> Result<(std::fs::File, String), ModelIoError> {
    for _ in 0..max_attempts {
        let name = next_name();
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join(&name))
        {
            Ok(file) => return Ok((file, name)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(ModelIoError::Io(e)),
        }
    }
    Err(ModelIoError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!("{max_attempts} 回試行しても一意なファイル名を確保できませんでした"),
    )))
}

/// 自分が作って fd を保持している一時ファイルの削除（決定記録 §13.6。best-effort）。
/// 保持ハンドルの `fstat` と削除直前の `symlink_metadata` の `(dev, ino)` が一致し、
/// かつ通常ファイルのときだけ `remove_file` する（差し替え検出時は削除しない）。
#[cfg(unix)]
fn remove_own_tmp(file: &std::fs::File, path: &Path) {
    use std::os::unix::fs::MetadataExt;
    let Ok(held) = file.metadata() else {
        return;
    };
    let Ok(on_disk) = std::fs::symlink_metadata(path) else {
        return;
    };
    if on_disk.file_type().is_file() && on_disk.dev() == held.dev() && on_disk.ino() == held.ino() {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(unix)]
static NAME_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// pid・ナノ秒・プロセス内カウンタの組（一時名・世代 ID の素材）。
#[cfg(unix)]
fn name_entropy() -> (u32, u128, u64) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let counter = NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (std::process::id(), nanos, counter)
}

/// 32 桁の 16 進世代 ID（pid・ナノ秒・カウンタを `RandomState` でハッシュ化した u64 を 2 つ
/// 連結する。衝突は `create_new` の再試行で検出できるため暗号学的乱数は不要。§12.3 手順 1）。
#[cfg(unix)]
fn generation_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let (pid, nanos, counter) = name_entropy();
    let mut id = String::with_capacity(32);
    for lane in 0..2u64 {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u32(pid);
        hasher.write_u128(nanos);
        hasher.write_u64(counter);
        hasher.write_u64(lane);
        id.push_str(&format!("{:016x}", hasher.finish()));
    }
    id
}

/// manifest 用一時ファイル名 `.manifest.json.tmp-{pid}-{counter}-{nanos}`
/// （`docs-site` の `write_file_creating_parent` と同型。§12.3 手順 2）。
#[cfg(unix)]
fn tmp_manifest_name() -> String {
    let (pid, nanos, counter) = name_entropy();
    format!(".manifest.json.tmp-{pid}-{counter}-{nanos}")
}

// ---------------------------------------------------------------------
// 復元
// ---------------------------------------------------------------------

/// `LeafError` を型付きエラーへ写す（`what` は上限超過時の対象名）。
fn map_leaf_error(what: &'static str) -> impl Fn(LeafError) -> ModelIoError {
    move |e| match e {
        LeafError::Io(e) => ModelIoError::Io(e),
        LeafError::TooLarge { limit } => ModelIoError::TooLarge { what, limit },
    }
}

/// [`load_model`] の本体。固定上限を引数化して、単体テストが実際の 1 GiB ファイルを
/// 作らずに上限超過の拒否を検証できるようにする（`model.rs::load_with_limit` と同型）。
fn load_from_dir_with_limits(
    dir: &Path,
    max_manifest: u64,
    max_model: u64,
) -> Result<Sequential, ModelIoError> {
    let manifest_bytes = open_leaf_checked(&dir.join(MANIFEST_FILE_NAME), max_manifest)
        .and_then(OpenedLeaf::read_exact_len)
        .map_err(map_leaf_error("manifest.json"))?;
    let manifest = parse_manifest(&manifest_bytes)?;

    // 上限判定は fstat 実長と固定上限だけで行い（open_leaf_checked）、その後で非信頼値
    // `safetensors_bytes` との一致だけを確認する（決定記録 §13.2 手順 4）。
    let opened = open_leaf_checked(&dir.join(&manifest.safetensors_file), max_model)
        .map_err(map_leaf_error("model safetensors"))?;
    if opened.len() != manifest.safetensors_bytes {
        return Err(ModelIoError::Mismatch {
            message: format!(
                "safetensors の実バイト数 {} が manifest の記載 {} と一致しません",
                opened.len(),
                manifest.safetensors_bytes
            ),
        });
    }
    let bytes = opened
        .read_exact_len()
        .map_err(map_leaf_error("model safetensors"))?;
    let tensors = load_safetensors_f32_from_bytes(&bytes)
        .map_err(|e| ModelIoError::Safetensors(e.to_string()))?;
    drop(bytes);

    // キー集合と shape の完全一致（無言 skip をしない。REQ-7）。
    let consistent = tensors.len() == manifest.parameter_keys.len()
        && manifest.parameter_keys.iter().all(|(key, shape)| {
            tensors
                .get(key)
                .is_some_and(|t| t.shape() == shape.as_slice())
        });
    if !consistent {
        return Err(ModelIoError::Mismatch {
            message: "safetensors のキー集合または shape が manifest と一致しません".into(),
        });
    }

    // ここまでで Linear の in／out は実テンソルの shape と一致済み（非信頼な整数だけで
    // 確保量を決めない）。層を積み、値を bit のまま設定する。
    let mut model = build_model(&manifest.specs)?;
    model
        .load_state_dict(tensors)
        .map_err(ModelIoError::Autodiff)?;
    model.set_training(manifest.training);
    Ok(model)
}

/// 層構成から未学習の `Sequential` を構築する（重みは直後に `load_state_dict` で上書きする
/// ため seed は意味を持たない。0 固定）。
fn build_model(specs: &[LayerSpec]) -> Result<Sequential, ModelIoError> {
    let mut model = Sequential::new();
    for spec in specs {
        model = match spec {
            LayerSpec::Linear {
                in_features,
                out_features,
            } => model
                .add_linear(*in_features, *out_features, 0)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::Relu => model.add_relu(),
            LayerSpec::Sigmoid => model.add_sigmoid(),
            LayerSpec::Tanh => model.add_tanh(),
            LayerSpec::Silu => model.add_silu(),
            LayerSpec::Hardswish => model.add_hardswish(),
            LayerSpec::Gelu => model.add_gelu(),
            LayerSpec::GeluTanh => model.add_gelu_tanh(),
            LayerSpec::Unsupported { kind } => {
                return Err(ModelIoError::UnsupportedModel {
                    reason: format!("層 {kind} は復元に未対応です"),
                });
            }
        };
    }
    Ok(model)
}

// ---------------------------------------------------------------------
// manifest の厳格パース（手書き。serde 非依存）
// ---------------------------------------------------------------------

/// 検証済みの manifest。
struct ParsedManifest {
    training: bool,
    specs: Vec<LayerSpec>,
    parameter_keys: Vec<(String, Vec<usize>)>,
    safetensors_file: String,
    safetensors_bytes: u64,
}

/// 本形式が受理する JSON 値（非負整数・エスケープなし文字列・bool・null・配列・object）。
#[derive(Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(u64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn manifest_error(message: impl Into<String>) -> ModelIoError {
    ModelIoError::Manifest {
        message: message.into(),
    }
}

/// エラー文言へ入れる非信頼文字列を短く切る。
fn clip(s: &str) -> String {
    s.chars().take(64).collect()
}

/// 厳格 JSON パーサ。入力は UTF-8 検証済みのバイト列。
struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn fail(&self, what: &str) -> ModelIoError {
        manifest_error(format!(
            "JSON 構文エラー: {what}（バイト位置 {}）",
            self.pos
        ))
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8, what: &str) -> Result<(), ModelIoError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.fail(what))
        }
    }

    fn document(mut self) -> Result<Json, ModelIoError> {
        let value = self.value(0)?;
        self.skip_ws();
        if self.pos != self.src.len() {
            return Err(self.fail("値の後ろに余分な内容があります"));
        }
        Ok(value)
    }

    /// `depth` は現在のコンテナのネスト数。コンテナへ入る前（再帰前）に上限を検査する。
    fn value(&mut self, depth: usize) -> Result<Json, ModelIoError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'0'..=b'9') => self.number().map(Json::Num),
            Some(b't') => self.literal(b"true", Json::Bool(true)),
            Some(b'f') => self.literal(b"false", Json::Bool(false)),
            Some(b'n') => self.literal(b"null", Json::Null),
            _ => Err(self.fail("値として解釈できません")),
        }
    }

    fn literal(&mut self, word: &[u8], value: Json) -> Result<Json, ModelIoError> {
        if self.src[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.fail("不正なリテラル"))
        }
    }

    /// 非負整数のみ。先頭ゼロ・符号・小数・指数は拒否し、u64 の桁あふれも拒否する。
    fn number(&mut self) -> Result<u64, ModelIoError> {
        let start = self.pos;
        let mut acc: u64 = 0;
        while let Some(b @ b'0'..=b'9') = self.peek() {
            if self.pos > start && self.src[start] == b'0' {
                return Err(self.fail("整数に先頭ゼロは使えません"));
            }
            acc = acc
                .checked_mul(10)
                .and_then(|v| v.checked_add(u64::from(b - b'0')))
                .ok_or_else(|| self.fail("整数が u64 に収まりません"))?;
            self.pos += 1;
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(self.fail("小数・指数は使えません"));
        }
        Ok(acc)
    }

    /// エスケープなし文字列。`\` と制御文字は拒否する（ライタがエスケープを出さないため）。
    fn string(&mut self) -> Result<String, ModelIoError> {
        self.expect(b'"', "文字列の開始")?;
        let start = self.pos;
        loop {
            match self.peek() {
                Some(b'"') => break,
                Some(b) if b < 0x20 || b == 0x7f || b == b'\\' => {
                    return Err(self.fail("文字列に制御文字またはエスケープは使えません"));
                }
                Some(_) => self.pos += 1,
                None => return Err(self.fail("文字列が閉じていません")),
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.fail("文字列が UTF-8 ではありません"))?
            .to_string();
        self.pos += 1;
        Ok(text)
    }

    fn array(&mut self, depth: usize) -> Result<Json, ModelIoError> {
        if depth >= MAX_JSON_DEPTH {
            return Err(self.fail("ネストが深すぎます"));
        }
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            if items.len() >= MAX_ARRAY_LEN {
                return Err(ModelIoError::TooLarge {
                    what: "JSON 配列の要素数",
                    limit: MAX_ARRAY_LEN as u64,
                });
            }
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.fail("配列の区切りが不正です")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, ModelIoError> {
        if depth >= MAX_JSON_DEPTH {
            return Err(self.fail("ネストが深すぎます"));
        }
        self.pos += 1;
        let mut entries: Vec<(String, Json)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(entries));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            if entries.iter().any(|(k, _)| *k == key) {
                return Err(manifest_error(format!("重複キー {}", clip(&key))));
            }
            if entries.len() >= MAX_OBJECT_KEYS {
                return Err(ModelIoError::TooLarge {
                    what: "JSON object のキー数",
                    limit: MAX_OBJECT_KEYS as u64,
                });
            }
            self.skip_ws();
            self.expect(b':', "キーの後ろの ':'")?;
            let value = self.value(depth + 1)?;
            entries.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(entries));
                }
                _ => return Err(self.fail("object の区切りが不正です")),
            }
        }
    }
}

/// object が `keys` と**完全に同じキー集合**を持つことを確認し、`keys` の順に値を返す
/// （未知キー・欠落は `Manifest`。重複キーはパーサが先に拒否済み）。
fn exact_fields<'a>(
    value: &'a Json,
    ctx: &str,
    keys: &[&str],
) -> Result<Vec<&'a Json>, ModelIoError> {
    let Json::Obj(entries) = value else {
        return Err(manifest_error(format!(
            "{ctx} は object である必要があります"
        )));
    };
    if let Some((unknown, _)) = entries.iter().find(|(k, _)| !keys.contains(&k.as_str())) {
        return Err(manifest_error(format!(
            "{ctx} に未知のキー {} があります",
            clip(unknown)
        )));
    }
    keys.iter()
        .map(|k| {
            entries
                .iter()
                .find(|(ek, _)| ek == k)
                .map(|(_, v)| v)
                .ok_or_else(|| manifest_error(format!("{ctx} にキー {k} がありません")))
        })
        .collect()
}

fn as_u64(value: &Json, ctx: &str) -> Result<u64, ModelIoError> {
    match value {
        Json::Num(n) => Ok(*n),
        _ => Err(manifest_error(format!(
            "{ctx} は非負整数である必要があります"
        ))),
    }
}

fn as_usize(value: &Json, ctx: &str) -> Result<usize, ModelIoError> {
    usize::try_from(as_u64(value, ctx)?)
        .map_err(|_| manifest_error(format!("{ctx} が usize に収まりません")))
}

fn as_str<'a>(value: &'a Json, ctx: &str) -> Result<&'a str, ModelIoError> {
    match value {
        Json::Str(s) => Ok(s),
        _ => Err(manifest_error(format!(
            "{ctx} は文字列である必要があります"
        ))),
    }
}

fn as_arr<'a>(value: &'a Json, ctx: &str) -> Result<&'a [Json], ModelIoError> {
    match value {
        Json::Arr(a) => Ok(a),
        _ => Err(manifest_error(format!("{ctx} は配列である必要があります"))),
    }
}

/// `kind` と `params` から層構成を復元する（未知の kind は `UnsupportedModel`）。
fn spec_from_kind(kind: &str, params: &Json) -> Result<LayerSpec, ModelIoError> {
    let simple = |spec: LayerSpec| -> Result<LayerSpec, ModelIoError> {
        exact_fields(params, "layers[].params", &[])?;
        Ok(spec)
    };
    match kind {
        "linear" => {
            let f = exact_fields(params, "layers[].params", &["in_features", "out_features"])?;
            Ok(LayerSpec::Linear {
                in_features: as_usize(f[0], "in_features")?,
                out_features: as_usize(f[1], "out_features")?,
            })
        }
        "relu" => simple(LayerSpec::Relu),
        "sigmoid" => simple(LayerSpec::Sigmoid),
        "tanh" => simple(LayerSpec::Tanh),
        "silu" => simple(LayerSpec::Silu),
        "hardswish" => simple(LayerSpec::Hardswish),
        "gelu" => simple(LayerSpec::Gelu),
        "gelu_tanh" => simple(LayerSpec::GeluTanh),
        other => Err(ModelIoError::UnsupportedModel {
            reason: format!("未対応の層 kind {}", clip(other)),
        }),
    }
}

/// manifest のバイト列を厳格に検証して [`ParsedManifest`] にする
/// （決定記録 §4・§13.5）。
fn parse_manifest(bytes: &[u8]) -> Result<ParsedManifest, ModelIoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| manifest_error("manifest が UTF-8 ではありません"))?;
    let root = Parser {
        src: text.as_bytes(),
        pos: 0,
    }
    .document()?;
    let f = exact_fields(
        &root,
        "manifest",
        &[
            "format",
            "format_version",
            "training",
            "num_layers",
            "layers",
            "parameter_keys",
            "buffer_keys",
            "safetensors_file",
            "safetensors_bytes",
            "compiled",
        ],
    )?;

    if as_str(f[0], "format")? != FORMAT_NAME {
        return Err(manifest_error("format が想定と異なります"));
    }
    if as_u64(f[1], "format_version")? != FORMAT_VERSION {
        return Err(manifest_error("format_version が未対応です"));
    }
    let Json::Bool(training) = f[2] else {
        return Err(manifest_error("training は bool である必要があります"));
    };
    let num_layers = as_usize(f[3], "num_layers")?;
    let layers = as_arr(f[4], "layers")?;
    let parameter_keys = as_arr(f[5], "parameter_keys")?;
    let buffer_keys = as_arr(f[6], "buffer_keys")?;
    let safetensors_file = as_str(f[7], "safetensors_file")?;
    let safetensors_bytes = as_u64(f[8], "safetensors_bytes")?;

    // 層数は配列を 1 要素ずつ数えた結果だけを信じる（`num_layers` は一致確認のみ）。
    if layers.len() > MAX_LAYERS {
        return Err(ModelIoError::TooLarge {
            what: "層数",
            limit: MAX_LAYERS as u64,
        });
    }
    if num_layers != layers.len() {
        return Err(manifest_error(
            "num_layers が layers の要素数と一致しません",
        ));
    }
    if !buffer_keys.is_empty() {
        return Err(ModelIoError::UnsupportedModel {
            reason: "buffer_keys を持つモデルは未対応です".into(),
        });
    }
    if !matches!(f[9], Json::Null) {
        return Err(ModelIoError::UnsupportedModel {
            reason: "compile 済みモデル（compiled が null でない）は未対応です".into(),
        });
    }
    if !is_valid_safetensors_file_name(safetensors_file) {
        return Err(manifest_error(
            "safetensors_file が model.<32 桁 16 進>.safetensors の形式ではありません",
        ));
    }

    let mut specs = Vec::with_capacity(layers.len());
    for (i, layer) in layers.iter().enumerate() {
        let lf = exact_fields(layer, "layers[]", &["index", "kind", "params"])?;
        if as_usize(lf[0], "layers[].index")? != i {
            return Err(manifest_error("layers[].index が連番ではありません"));
        }
        specs.push(spec_from_kind(as_str(lf[1], "layers[].kind")?, lf[2])?);
    }

    let mut keys = Vec::with_capacity(parameter_keys.len());
    for entry in parameter_keys {
        let kf = exact_fields(entry, "parameter_keys[]", &["key", "shape"])?;
        let key = as_str(kf[0], "parameter_keys[].key")?.to_string();
        let shape = as_arr(kf[1], "parameter_keys[].shape")?
            .iter()
            .map(|d| as_usize(d, "parameter_keys[].shape[]"))
            .collect::<Result<Vec<_>, _>>()?;
        keys.push((key, shape));
    }
    if keys != expected_parameter_keys(&specs) {
        return Err(ModelIoError::Mismatch {
            message: "parameter_keys が層構成から導いた期待キー・shape と一致しません".into(),
        });
    }

    Ok(ParsedManifest {
        training: *training,
        specs,
        parameter_keys: keys,
        safetensors_file: safetensors_file.to_string(),
        safetensors_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-ai-model-io-{}-{}-{label}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("一時ディレクトリを作成できるはず");
        dir
    }

    fn parse_json(s: &str) -> Result<Json, ModelIoError> {
        Parser {
            src: s.as_bytes(),
            pos: 0,
        }
        .document()
    }

    fn is_manifest_err<T>(r: Result<T, ModelIoError>) -> bool {
        matches!(r, Err(ModelIoError::Manifest { .. }))
    }

    #[test]
    fn parser_accepts_v1_shapes() {
        assert!(parse_json(r#"{"a":[1,2,{"b":null}],"c":true,"d":"x"}"#).is_ok());
        assert!(parse_json(" [ ] ").is_ok());
    }

    #[test]
    fn parser_rejects_duplicate_keys() {
        assert!(is_manifest_err(parse_json(r#"{"a":1,"a":2}"#)));
    }

    #[test]
    fn parser_rejects_leading_zero_fraction_exponent_and_sign() {
        for s in ["01", "1.5", "1e3", "-1", "18446744073709551616"] {
            assert!(is_manifest_err(parse_json(s)), "{s} は拒否されるはず");
        }
        assert!(matches!(parse_json("0"), Ok(Json::Num(0))));
    }

    #[test]
    fn parser_rejects_trailing_garbage_and_trailing_comma() {
        assert!(is_manifest_err(parse_json("{} x")));
        assert!(is_manifest_err(parse_json("[1,]")));
        assert!(is_manifest_err(parse_json(r#"{"a":1,}"#)));
    }

    #[test]
    fn parser_rejects_escapes_control_chars_and_bom() {
        assert!(is_manifest_err(parse_json(r#""a\nb""#)));
        assert!(is_manifest_err(parse_json("\"a\u{1}b\"")));
        assert!(is_manifest_err(parse_json("\u{feff}{}")));
    }

    #[test]
    fn parser_rejects_excess_depth_before_recursing() {
        // MAX_JSON_DEPTH 段までは受理し、超えたら拒否する。
        let ok = format!(
            "{}{}",
            "[".repeat(MAX_JSON_DEPTH),
            "]".repeat(MAX_JSON_DEPTH)
        );
        assert!(parse_json(&ok).is_ok());
        let deep = format!(
            "{}{}",
            "[".repeat(MAX_JSON_DEPTH + 1),
            "]".repeat(MAX_JSON_DEPTH + 1)
        );
        assert!(is_manifest_err(parse_json(&deep)));
        // 極端に深い入力でもスタックを使い切らずに拒否される。
        assert!(is_manifest_err(parse_json(&"[".repeat(100_000))));
    }

    #[test]
    fn parser_caps_array_length_and_object_keys() {
        let many = format!("[{}]", vec!["0"; MAX_ARRAY_LEN + 1].join(","));
        assert!(matches!(
            parse_json(&many),
            Err(ModelIoError::TooLarge { .. })
        ));
        let keys: Vec<String> = (0..=MAX_OBJECT_KEYS)
            .map(|i| format!("\"k{i}\":0"))
            .collect();
        assert!(matches!(
            parse_json(&format!("{{{}}}", keys.join(","))),
            Err(ModelIoError::TooLarge { .. })
        ));
    }

    #[test]
    fn safetensors_file_name_pattern_is_exact() {
        let good = format!("model.{}.safetensors", "0123456789abcdef".repeat(2));
        assert!(is_valid_safetensors_file_name(&good));
        for bad in [
            "../model.0123456789abcdef0123456789abcdef.safetensors",
            "model.0123456789ABCDEF0123456789ABCDEF.safetensors",
            "model.0123456789abcdef.safetensors",
            "model.0123456789abcdef0123456789abcde/.safetensors",
            "model.0123456789abcdef0123456789abcdef.safetensors/",
            "model.safetensors",
            "",
        ] {
            assert!(
                !is_valid_safetensors_file_name(bad),
                "{bad} は拒否されるはず"
            );
        }
    }

    #[test]
    fn non_unix_platform_is_rejected_as_unsupported() {
        let err = save_platform_supported(false).expect_err("非 unix は拒否されるはず");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        assert!(save_platform_supported(true).is_ok());
    }

    fn sample_model() -> Sequential {
        Sequential::new()
            .add_linear(3, 4, 1)
            .and_then(|m| m.add_relu().add_linear(4, 2, 2))
            .expect("構築できるはず")
    }

    #[test]
    fn prepare_save_rejects_unsupported_and_compiled_models() {
        let dropout = Sequential::new().add_dropout(0.5).expect("構築できるはず");
        assert!(matches!(
            prepare_save(&dropout),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        let custom = Sequential::new().add_relu().add_leaky_relu(0.1);
        assert!(matches!(
            prepare_save(&custom),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
    }

    #[test]
    fn manifest_round_trips_through_strict_parser() {
        let model = sample_model();
        let prepared = prepare_save(&model).expect("検証を通るはず");
        let name = format!("model.{}.safetensors", "a".repeat(32));
        let text = render_manifest(&prepared, &name, 123);
        let parsed = parse_manifest(text.as_bytes()).expect("自分が書いた manifest は読めるはず");
        assert_eq!(parsed.specs, prepared.specs);
        assert_eq!(parsed.parameter_keys, prepared.parameter_keys);
        assert_eq!(parsed.safetensors_file, name);
        assert_eq!(parsed.safetensors_bytes, 123);
        assert!(parsed.training);
    }

    #[test]
    fn parse_manifest_rejects_unknown_keys_and_unsupported_features() {
        let model = sample_model();
        let prepared = prepare_save(&model).expect("検証を通るはず");
        let name = format!("model.{}.safetensors", "b".repeat(32));
        let good = render_manifest(&prepared, &name, 1);
        let with_unknown = good.replacen("\"training\"", "\"extra\":1,\"training\"", 1);
        assert!(is_manifest_err(parse_manifest(with_unknown.as_bytes())));
        let compiled = good.replace("\"compiled\":null", "\"compiled\":{}");
        assert!(matches!(
            parse_manifest(compiled.as_bytes()),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        let buffers = good.replace("\"buffer_keys\":[]", "\"buffer_keys\":[{}]");
        assert!(matches!(
            parse_manifest(buffers.as_bytes()),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        let kind = good.replace("\"relu\"", "\"conv2d\"");
        assert!(matches!(
            parse_manifest(kind.as_bytes()),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        let bad_file = good.replace(&name, "../evil");
        assert!(is_manifest_err(parse_manifest(bad_file.as_bytes())));
    }

    #[test]
    fn parse_manifest_rejects_layer_count_over_bound() {
        let layers: Vec<String> = (0..=MAX_LAYERS)
            .map(|i| format!("{{\"index\":{i},\"kind\":\"relu\",\"params\":{{}}}}"))
            .collect();
        let text = format!(
            "{{\"format\":\"{FORMAT_NAME}\",\"format_version\":1,\"training\":true,\"num_layers\":{},\"layers\":[{}],\"parameter_keys\":[],\"buffer_keys\":[],\"safetensors_file\":\"model.{}.safetensors\",\"safetensors_bytes\":0,\"compiled\":null}}",
            MAX_LAYERS + 1,
            layers.join(","),
            "0".repeat(32)
        );
        assert!(matches!(
            parse_manifest(text.as_bytes()),
            Err(ModelIoError::TooLarge { what: "層数", .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn create_new_with_retry_gives_up_after_bound_without_touching_existing() {
        let dir = temp_dir("retry");
        let existing = dir.join("fixed");
        std::fs::write(&existing, b"keep").expect("書き込めるはず");
        let mut calls = 0usize;
        let result = create_new_with_retry(
            &dir,
            || {
                calls += 1;
                "fixed".to_string()
            },
            MAX_TMP_NAME_ATTEMPTS,
        );
        assert!(matches!(
            result,
            Err(ModelIoError::Io(e)) if e.kind() == io::ErrorKind::AlreadyExists
        ));
        assert_eq!(calls, MAX_TMP_NAME_ATTEMPTS);
        assert_eq!(std::fs::read(&existing).expect("読めるはず"), b"keep");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn create_new_with_retry_retries_past_a_collision() {
        let dir = temp_dir("retry-ok");
        std::fs::write(dir.join("first"), b"keep").expect("書き込めるはず");
        let mut names = ["first", "second"].into_iter();
        let (_file, name) = create_new_with_retry(
            &dir,
            || names.next().unwrap_or("third").to_string(),
            MAX_TMP_NAME_ATTEMPTS,
        )
        .expect("2 回目で成功するはず");
        assert_eq!(name, "second");
        assert_eq!(
            std::fs::read(dir.join("first")).expect("読めるはず"),
            b"keep"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn generation_id_is_32_lowercase_hex() {
        let id = generation_id();
        assert!(is_valid_safetensors_file_name(&format!(
            "model.{id}.safetensors"
        )));
        assert_ne!(id, generation_id());
    }

    #[cfg(unix)]
    #[test]
    fn remove_own_tmp_skips_replaced_path() {
        let dir = temp_dir("own-tmp");
        let path = dir.join("tmp");
        let file = std::fs::File::create(&path).expect("作れるはず");
        // 差し替え: 同名で別ファイル（別 inode）にする。
        std::fs::remove_file(&path).expect("消せるはず");
        std::fs::write(&path, b"other").expect("書き込めるはず");
        remove_own_tmp(&file, &path);
        assert!(path.exists(), "別ファイルへ差し替えられた場合は削除しない");
        let own = dir.join("own");
        let own_file = std::fs::File::create(&own).expect("作れるはず");
        remove_own_tmp(&own_file, &own);
        assert!(!own.exists(), "自分が作ったままのファイルは削除される");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn load_with_small_limits_reports_too_large() {
        let dir = temp_dir("limits");
        let model = sample_model();
        save_model(&model, &dir).expect("保存できるはず");
        assert!(matches!(
            load_from_dir_with_limits(&dir, 8, MAX_MODEL_FILE_BYTES),
            Err(ModelIoError::TooLarge {
                what: "manifest.json",
                limit: 8
            })
        ));
        assert!(matches!(
            load_from_dir_with_limits(&dir, MAX_MANIFEST_BYTES, 8),
            Err(ModelIoError::TooLarge {
                what: "model safetensors",
                limit: 8
            })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn manifest_of_max_layers_fits_the_manifest_bound() {
        // 上限の層数（Linear のみ・5 桁の次元）でも manifest が固定上限に収まることを示す
        // （`MAX_MANIFEST_BYTES` の根拠。次元がさらに大きければ save 側が `TooLarge` にする）。
        let specs: Vec<LayerSpec> = (0..MAX_LAYERS)
            .map(|_| LayerSpec::Linear {
                in_features: 65_536,
                out_features: 65_536,
            })
            .collect();
        let prepared = PreparedSave {
            training: true,
            parameter_keys: expected_parameter_keys(&specs),
            specs,
            safetensors: Vec::new(),
        };
        let text = render_manifest(
            &prepared,
            &format!("model.{}.safetensors", "0".repeat(32)),
            u64::MAX,
        );
        assert!(
            (text.len() as u64) <= MAX_MANIFEST_BYTES,
            "len={}",
            text.len()
        );
    }
}
