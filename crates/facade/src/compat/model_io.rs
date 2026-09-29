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
//! 未 `compile` の `Sequential` で、層が `add_*` 30 種（`add_linear`・活性化・`add_softmax`
//! 系・`add_flatten`・`add_dropout`・conv・正規化・embedding・MHA・TE・pooling。イシュー #2370）
//! だけの場合に限る。`kind` は文字列の allowlist、`params` は kind ごとの固定スキーマ
//! （決定記録 §4）で、`add_*` の引数（`seed` を除く）を記録し load が同じ `add_*` を呼んで再構築する。
//! 次は `save_model` が [`ModelIoError::UnsupportedModel`]／[`ModelIoError::TooLarge`] で拒否し、
//! `dir` には何も作らない（fail-closed。REQ-7 の無言 skip 禁止）:
//!
//! - `add_module` の利用者定義層・`compile` 済みモデル（compile 状態は #2372・#2373）
//! - running stats（buffer）が初期値でない BatchNorm（buffer の保存は #2371 まで暫定拒否）
//! - dropout・BatchNorm の層のモードがモデル全体と食い違うモデル（load が全層を
//!   `manifest.training` へ揃えるため復元後に forward がずれる）
//! - f32 引数が非有限の層・manifest が固定上限（配列長・サイズ等）を超えるモデル
//!   （save 側で load と同じ厳格パーサによる読み戻しを事前に行う）
//!
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
use crate::Tensor;
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
/// 約 0.6 MiB に収まる。**2026-09-29 ユーザー承認済み**（親 #2362 のコメント
/// <https://github.com/Fandhe-AI/fandhe-ai/issues/2362#issuecomment-5888987015>。
/// 決定記録 §2 item 4）。値の変更は再承認が必要。
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
/// 層数の上限。4096。
///
/// 値の根拠: 既存テストに 1714 層の `Sequential` があり
/// （`sequential.rs` の `..._with_1714_layers`）、それを包含する 2 のべき乗。
/// 承認状況は `MAX_MANIFEST_BYTES` と同じ（2026-09-29 ユーザー承認済み。#2362 コメント）。
const MAX_LAYERS: usize = 4096;
/// JSON のコンテナ（object／array）のネスト上限。ポリシー閾値ではなく v1 スキーマの
/// 最大ネスト（root → 配列 → 要素 object → params／shape）から導いた構造上の値。
const MAX_JSON_DEPTH: usize = 4;
/// 1 つの JSON 配列の要素数上限。#2369 で確定した値（`2 * MAX_LAYERS`）を維持し変更しない
/// （閾値の引き上げは再承認が必要）。層あたりのパラメータ数が多い層（TE は 16 個）を積んだ
/// モデルは `parameter_keys` がこの上限を先に超えうるが、その場合は `save_model` が
/// 書き込み前に `TooLarge` で拒否する（`verify_round_trip`）ため、読めない manifest は生まれない。
const MAX_ARRAY_LEN: usize = 2 * MAX_LAYERS;
/// 数値の字句長の上限（バイト）。f32 の最短往復表現（符号・仮数・指数で高々 15 バイト程度）と
/// u64（20 桁）に十分な余裕を持つ構造上の値で、巨大な字句によるコピー・パースを有界にする。
const MAX_NUMBER_LEXEME_LEN: usize = 64;
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

/// 保存可能な層の manifest 上の `kind` 名（文字列 allowlist の正）。`add_module` 由来の
/// 層は `None`（保存不可）。`spec_from_kind` の allowlist とは 30 種の往復テストで一致を担保する。
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
        LayerSpec::LeakyRelu { .. } => Some("leaky_relu"),
        LayerSpec::Elu { .. } => Some("elu"),
        LayerSpec::Softmax { .. } => Some("softmax"),
        LayerSpec::LogSoftmax { .. } => Some("log_softmax"),
        LayerSpec::Softplus { .. } => Some("softplus"),
        LayerSpec::Flatten { .. } => Some("flatten"),
        LayerSpec::Dropout { .. } => Some("dropout"),
        LayerSpec::Conv2d { .. } => Some("conv2d"),
        LayerSpec::Conv1d { .. } => Some("conv1d"),
        LayerSpec::LayerNorm { .. } => Some("layer_norm"),
        LayerSpec::RmsNorm { .. } => Some("rms_norm"),
        LayerSpec::BatchNorm1d { .. } => Some("batch_norm1d"),
        LayerSpec::BatchNorm2d { .. } => Some("batch_norm2d"),
        LayerSpec::Embedding { .. } => Some("embedding"),
        LayerSpec::MultiheadAttention { .. } => Some("multihead_attention"),
        LayerSpec::TransformerEncoder { .. } => Some("transformer_encoder"),
        LayerSpec::MaxPool2d { .. } => Some("max_pool2d"),
        LayerSpec::MaxPool1d { .. } => Some("max_pool1d"),
        LayerSpec::AvgPool2d { .. } => Some("avg_pool2d"),
        LayerSpec::AvgPool1d { .. } => Some("avg_pool1d"),
        LayerSpec::AdaptiveAvgPool2d { .. } => Some("adaptive_avg_pool2d"),
        LayerSpec::AdaptiveAvgPool1d { .. } => Some("adaptive_avg_pool1d"),
        LayerSpec::Unsupported { .. } => None,
    }
}

/// 層の f32 引数がすべて有限か（NaN／inf は JSON 数値として書けず、復元の
/// 正準形検査（[`as_f32`]）も通らないため、保存側で先に型付きで拒否する）。
fn spec_f32_fields_are_finite(spec: &LayerSpec) -> bool {
    match spec {
        LayerSpec::LeakyRelu { negative_slope } => negative_slope.is_finite(),
        LayerSpec::Elu { alpha } => alpha.is_finite(),
        LayerSpec::Softplus { beta, threshold } => beta.is_finite() && threshold.is_finite(),
        LayerSpec::Dropout { p } => p.is_finite(),
        LayerSpec::LayerNorm { eps, .. } | LayerSpec::RmsNorm { eps, .. } => eps.is_finite(),
        LayerSpec::BatchNorm1d { eps, momentum, .. }
        | LayerSpec::BatchNorm2d { eps, momentum, .. } => eps.is_finite() && momentum.is_finite(),
        _ => true,
    }
}

/// MultiheadAttention 相当（q／k／v／out の 4 本の Linear。`kdim = vdim = embed_dim`）の
/// パラメータ名と shape。`with_prefix` は TE の `self_attn.` 接頭辞付きの名前を返す。
fn mha_parameters(embed: usize, with_prefix: bool) -> Vec<(&'static str, Vec<usize>)> {
    let names: [&'static str; 8] = if with_prefix {
        [
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
        ]
    } else {
        [
            "q_proj.weight",
            "q_proj.bias",
            "k_proj.weight",
            "k_proj.bias",
            "v_proj.weight",
            "v_proj.bias",
            "out_proj.weight",
            "out_proj.bias",
        ]
    };
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let shape = if i % 2 == 0 {
                vec![embed, embed]
            } else {
                vec![embed]
            };
            (name, shape)
        })
        .collect()
}

/// 層 1 つが持つパラメータの（層内名, shape）。並びは `Module::named_parameters` の
/// 列挙順（weight → bias、子層は子の順）で、`Sequential::state_dict` のキー順と一致する。
/// 純粋な算術のみで導き、非信頼な整数で確保しない（`Vec` は高々 16 要素）。`groups` の
/// 割り切れは [`spec_from_kind`] が先に検査するため、ここでのゼロ除算はありえない
/// （防御的に `checked_div` で 0 へ落とし、期待キーの不一致として現れる）。
fn layer_parameters(spec: &LayerSpec) -> Vec<(&'static str, Vec<usize>)> {
    match spec {
        LayerSpec::Linear {
            in_features,
            out_features,
        } => vec![
            ("weight", vec![*in_features, *out_features]),
            ("bias", vec![*out_features]),
        ],
        LayerSpec::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            groups,
            ..
        } => vec![
            (
                "weight",
                vec![
                    *out_channels,
                    in_channels.checked_div(*groups).unwrap_or(0),
                    kernel_size[0],
                    kernel_size[1],
                ],
            ),
            ("bias", vec![*out_channels]),
        ],
        LayerSpec::Conv1d {
            in_channels,
            out_channels,
            kernel_size,
            groups,
            ..
        } => vec![
            (
                "weight",
                vec![
                    *out_channels,
                    in_channels.checked_div(*groups).unwrap_or(0),
                    *kernel_size,
                ],
            ),
            ("bias", vec![*out_channels]),
        ],
        LayerSpec::LayerNorm {
            normalized_size, ..
        } => vec![
            ("weight", vec![*normalized_size]),
            ("bias", vec![*normalized_size]),
        ],
        LayerSpec::RmsNorm {
            normalized_size, ..
        } => vec![("weight", vec![*normalized_size])],
        LayerSpec::BatchNorm1d { num_features, .. }
        | LayerSpec::BatchNorm2d { num_features, .. } => vec![
            ("weight", vec![*num_features]),
            ("bias", vec![*num_features]),
        ],
        LayerSpec::Embedding {
            num_embeddings,
            embedding_dim,
            ..
        } => vec![("weight", vec![*num_embeddings, *embedding_dim])],
        LayerSpec::MultiheadAttention { embed_dim, .. } => mha_parameters(*embed_dim, false),
        LayerSpec::TransformerEncoder {
            d_model,
            dim_feedforward,
            ..
        } => {
            let mut out = mha_parameters(*d_model, true);
            out.extend([
                ("linear1.weight", vec![*d_model, *dim_feedforward]),
                ("linear1.bias", vec![*dim_feedforward]),
                ("linear2.weight", vec![*dim_feedforward, *d_model]),
                ("linear2.bias", vec![*d_model]),
                ("norm1.weight", vec![*d_model]),
                ("norm1.bias", vec![*d_model]),
                ("norm2.weight", vec![*d_model]),
                ("norm2.bias", vec![*d_model]),
            ]);
            out
        }
        LayerSpec::Relu
        | LayerSpec::Sigmoid
        | LayerSpec::Tanh
        | LayerSpec::Silu
        | LayerSpec::Hardswish
        | LayerSpec::Gelu
        | LayerSpec::GeluTanh
        | LayerSpec::LeakyRelu { .. }
        | LayerSpec::Elu { .. }
        | LayerSpec::Softmax { .. }
        | LayerSpec::LogSoftmax { .. }
        | LayerSpec::Softplus { .. }
        | LayerSpec::Flatten { .. }
        | LayerSpec::Dropout { .. }
        | LayerSpec::MaxPool2d { .. }
        | LayerSpec::MaxPool1d { .. }
        | LayerSpec::AvgPool2d { .. }
        | LayerSpec::AvgPool1d { .. }
        | LayerSpec::AdaptiveAvgPool2d { .. }
        | LayerSpec::AdaptiveAvgPool1d { .. }
        | LayerSpec::Unsupported { .. } => Vec::new(),
    }
}

/// 層構成から導く期待キー列（`Sequential::state_dict` と同じ `"{index}.{name}"`。
/// 層順・各層内は [`layer_parameters`] の順）。
fn expected_parameter_keys(specs: &[LayerSpec]) -> Vec<(String, Vec<usize>)> {
    let mut keys = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        for (name, shape) in layer_parameters(spec) {
            keys.push((format!("{i}.{name}"), shape));
        }
    }
    keys
}

/// `render_params` の組み立て用フィールド列（キー, 描画済み JSON 値）。
type ParamFields = Vec<(String, String)>;

fn put_num(f: &mut ParamFields, key: &str, v: usize) {
    f.push((key.to_string(), v.to_string()));
}

/// f32 は `{:?}`（往復可能な最短表現。`-0.0` も保つ）。有限性は保存前に検査済み。
fn put_real(f: &mut ParamFields, key: &str, v: f32) {
    f.push((key.to_string(), format!("{v:?}")));
}

fn put_bool(f: &mut ParamFields, key: &str, v: bool) {
    f.push((key.to_string(), v.to_string()));
}

fn put_opt(f: &mut ParamFields, key: &str, v: Option<usize>) {
    f.push((
        key.to_string(),
        v.map_or_else(|| "null".to_string(), |x| x.to_string()),
    ));
}

/// `[usize; 2]` は `{base}_h`／`{base}_w` の平坦な 2 キーへ展開する
/// （`params` の中に配列を置くと `MAX_JSON_DEPTH` を超えるため。決定記録 §4）。
fn put_pair(f: &mut ParamFields, base: &str, v: [usize; 2]) {
    put_num(f, &format!("{base}_h"), v[0]);
    put_num(f, &format!("{base}_w"), v[1]);
}

fn put_opt_pair(f: &mut ParamFields, base: &str, v: Option<[usize; 2]>) {
    put_opt(f, &format!("{base}_h"), v.map(|x| x[0]));
    put_opt(f, &format!("{base}_w"), v.map(|x| x[1]));
}

/// 層の `params` を manifest 用の JSON object 文字列にする（キー順固定。整数は 10 進、
/// `None` は `null`）。`spec_from_kind` が読む固定スキーマの書き手側で、保存側の
/// 自己検証（[`verify_round_trip`]）はこの文字列の完全一致（f32 は bit 一致）で比較する。
fn render_params(spec: &LayerSpec) -> String {
    let mut f: ParamFields = Vec::new();
    match spec {
        LayerSpec::Linear {
            in_features,
            out_features,
        } => {
            put_num(&mut f, "in_features", *in_features);
            put_num(&mut f, "out_features", *out_features);
        }
        LayerSpec::LeakyRelu { negative_slope } => {
            put_real(&mut f, "negative_slope", *negative_slope)
        }
        LayerSpec::Elu { alpha } => put_real(&mut f, "alpha", *alpha),
        LayerSpec::Softmax { dim } | LayerSpec::LogSoftmax { dim } => put_num(&mut f, "dim", *dim),
        LayerSpec::Softplus { beta, threshold } => {
            put_real(&mut f, "beta", *beta);
            put_real(&mut f, "threshold", *threshold);
        }
        LayerSpec::Flatten { start_dim, end_dim } => {
            put_num(&mut f, "start_dim", *start_dim);
            put_num(&mut f, "end_dim", *end_dim);
        }
        LayerSpec::Dropout { p } => put_real(&mut f, "p", *p),
        LayerSpec::Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        } => {
            put_num(&mut f, "in_channels", *in_channels);
            put_num(&mut f, "out_channels", *out_channels);
            put_pair(&mut f, "kernel_size", *kernel_size);
            put_pair(&mut f, "stride", *stride);
            put_pair(&mut f, "padding", *padding);
            put_pair(&mut f, "dilation", *dilation);
            put_num(&mut f, "groups", *groups);
        }
        LayerSpec::Conv1d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        } => {
            put_num(&mut f, "in_channels", *in_channels);
            put_num(&mut f, "out_channels", *out_channels);
            put_num(&mut f, "kernel_size", *kernel_size);
            put_num(&mut f, "stride", *stride);
            put_num(&mut f, "padding", *padding);
            put_num(&mut f, "dilation", *dilation);
            put_num(&mut f, "groups", *groups);
        }
        LayerSpec::LayerNorm {
            normalized_size,
            eps,
        }
        | LayerSpec::RmsNorm {
            normalized_size,
            eps,
        } => {
            put_num(&mut f, "normalized_size", *normalized_size);
            put_real(&mut f, "eps", *eps);
        }
        LayerSpec::BatchNorm1d {
            num_features,
            eps,
            momentum,
        }
        | LayerSpec::BatchNorm2d {
            num_features,
            eps,
            momentum,
        } => {
            put_num(&mut f, "num_features", *num_features);
            put_real(&mut f, "eps", *eps);
            put_real(&mut f, "momentum", *momentum);
        }
        LayerSpec::Embedding {
            num_embeddings,
            embedding_dim,
            padding_idx,
        } => {
            put_num(&mut f, "num_embeddings", *num_embeddings);
            put_num(&mut f, "embedding_dim", *embedding_dim);
            put_opt(&mut f, "padding_idx", *padding_idx);
        }
        LayerSpec::MultiheadAttention {
            embed_dim,
            num_heads,
        } => {
            put_num(&mut f, "embed_dim", *embed_dim);
            put_num(&mut f, "num_heads", *num_heads);
        }
        LayerSpec::TransformerEncoder {
            d_model,
            num_heads,
            dim_feedforward,
        } => {
            put_num(&mut f, "d_model", *d_model);
            put_num(&mut f, "num_heads", *num_heads);
            put_num(&mut f, "dim_feedforward", *dim_feedforward);
        }
        LayerSpec::MaxPool2d {
            kernel_size,
            stride,
            padding,
            dilation,
        } => {
            put_pair(&mut f, "kernel_size", *kernel_size);
            put_opt_pair(&mut f, "stride", *stride);
            put_pair(&mut f, "padding", *padding);
            put_pair(&mut f, "dilation", *dilation);
        }
        LayerSpec::MaxPool1d {
            kernel_size,
            stride,
            padding,
            dilation,
        } => {
            put_num(&mut f, "kernel_size", *kernel_size);
            put_opt(&mut f, "stride", *stride);
            put_num(&mut f, "padding", *padding);
            put_num(&mut f, "dilation", *dilation);
        }
        LayerSpec::AvgPool2d {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        } => {
            put_pair(&mut f, "kernel_size", *kernel_size);
            put_opt_pair(&mut f, "stride", *stride);
            put_pair(&mut f, "padding", *padding);
            put_bool(&mut f, "count_include_pad", *count_include_pad);
        }
        LayerSpec::AvgPool1d {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        } => {
            put_num(&mut f, "kernel_size", *kernel_size);
            put_opt(&mut f, "stride", *stride);
            put_num(&mut f, "padding", *padding);
            put_bool(&mut f, "count_include_pad", *count_include_pad);
        }
        LayerSpec::AdaptiveAvgPool2d { output_size } => {
            put_pair(&mut f, "output_size", *output_size)
        }
        LayerSpec::AdaptiveAvgPool1d { output_size } => {
            put_num(&mut f, "output_size", *output_size)
        }
        LayerSpec::Relu
        | LayerSpec::Sigmoid
        | LayerSpec::Tanh
        | LayerSpec::Silu
        | LayerSpec::Hardswish
        | LayerSpec::Gelu
        | LayerSpec::GeluTanh
        | LayerSpec::Unsupported { .. } => {}
    }
    let body: Vec<String> = f.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    format!("{{{}}}", body.join(","))
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
        if !spec_f32_fields_are_finite(spec) {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("層 {i} の f32 引数が有限ではないため保存できません"),
            });
        }
        check_layer_state(model, i, spec)?;
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
    verify_round_trip(&prepared)?;
    Ok(prepared)
}

/// 層の状態が「構成だけで復元できる」ことを確認する（fail-closed。REQ-7 の無言 skip 禁止）。
///
/// - dropout・BatchNorm は train／eval で forward が変わる。load は `set_training` で
///   全層を `manifest.training` へ揃えるため、層のモードがコンテナと食い違う（`eval()` の後に
///   `add_*` した等）モデルは、復元後に forward がずれる。食い違いを拒否する。
///   他の層へは適用しない（`Module::training` の既定が `true` のため誤って eval モデルを拒否する）。
/// - BatchNorm の running stats（buffer）は本バージョンで保存しない（後続イシュー #2371）。
///   初期値（mean=0・var=1・追跡回数 0）以外を保存すると load 後に stats が黙って戻るため、
///   初期値のときだけ保存を許す。#2371 で buffer を保存できるようになった時点で撤廃する。
fn check_layer_state(model: &Sequential, i: usize, spec: &LayerSpec) -> Result<(), ModelIoError> {
    let layer = &model.layers()[i];
    let mode_dependent = matches!(
        spec,
        LayerSpec::Dropout { .. } | LayerSpec::BatchNorm1d { .. } | LayerSpec::BatchNorm2d { .. }
    );
    if mode_dependent && layer.training() != model.training() {
        return Err(ModelIoError::UnsupportedModel {
            reason: format!(
                "層 {i} のモード（training）がモデル全体と異なります（load は全層をモデルのモードへ揃えるため復元後に出力がずれます）。`train()`／`eval()` で揃えてから保存してください"
            ),
        });
    }
    match spec {
        LayerSpec::BatchNorm1d { .. } => {
            let bn = layer
                .as_batch_norm1d()
                .ok_or_else(|| bn_downcast_error(i))?;
            ensure_initial_bn_stats(
                i,
                &bn.running_mean(),
                &bn.running_var(),
                bn.num_batches_tracked(),
            )
        }
        LayerSpec::BatchNorm2d { .. } => {
            let bn = layer
                .as_batch_norm2d()
                .ok_or_else(|| bn_downcast_error(i))?;
            ensure_initial_bn_stats(
                i,
                &bn.running_mean(),
                &bn.running_var(),
                bn.num_batches_tracked(),
            )
        }
        _ => Ok(()),
    }
}

fn bn_downcast_error(i: usize) -> ModelIoError {
    ModelIoError::UnsupportedModel {
        reason: format!(
            "層 {i} を BatchNorm として取り出せないため running stats を確認できません"
        ),
    }
}

/// running stats が初期値（全要素 bit で mean=`0.0`・var=`1.0`・追跡回数 0）か。
fn ensure_initial_bn_stats(
    i: usize,
    mean: &Tensor<f32>,
    var: &Tensor<f32>,
    num_batches_tracked: u64,
) -> Result<(), ModelIoError> {
    let all_bits = |t: &Tensor<f32>, expect: f32| {
        t.as_slice()
            .is_some_and(|s| s.iter().all(|v| v.to_bits() == expect.to_bits()))
    };
    if num_batches_tracked == 0 && all_bits(mean, 0.0) && all_bits(var, 1.0) {
        Ok(())
    } else {
        Err(ModelIoError::UnsupportedModel {
            reason: format!(
                "層 {i} の BatchNorm は running stats が初期値ではありません（running stats の保存は後続イシュー #2371 の対応まで未対応）"
            ),
        })
    }
}

/// 保存する構成を manifest へ描画し、load と同じ厳格パーサで読み戻して元の構成と
/// 一致することを確認する（「保存できたのに読めない」ファイルを作らない。`dir` への
/// 副作用の前に行う）。manifest サイズ・配列長・キー数・深さ・f32 正準形の上限違反は
/// まとめてここで検出され、上限起因（`TooLarge`）はそのまま返し、それ以外の不一致は
/// 内部不整合として `UnsupportedModel` にする。ファイル名は同じ長さの仮名を使う。
fn verify_round_trip(prepared: &PreparedSave) -> Result<(), ModelIoError> {
    let probe_name = format!("model.{}.safetensors", "0".repeat(32));
    let text = render_manifest(prepared, &probe_name, prepared.safetensors.len() as u64);
    if text.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(ModelIoError::TooLarge {
            what: "manifest.json",
            limit: MAX_MANIFEST_BYTES,
        });
    }
    let parsed = match parse_manifest(text.as_bytes()) {
        Ok(p) => p,
        Err(e @ ModelIoError::TooLarge { .. }) => return Err(e),
        Err(e) => {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("保存する構成を load が読み戻せません（{e}）"),
            });
        }
    };
    let same_specs = parsed.specs.len() == prepared.specs.len()
        && parsed
            .specs
            .iter()
            .zip(&prepared.specs)
            .all(|(a, b)| spec_kind(a) == spec_kind(b) && render_params(a) == render_params(b));
    if parsed.training != prepared.training
        || parsed.parameter_keys != prepared.parameter_keys
        || !same_specs
    {
        return Err(ModelIoError::UnsupportedModel {
            reason: "保存する構成と読み戻した構成が一致しません".into(),
        });
    }
    Ok(())
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
        let params = render_params(spec);
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

/// 層構成から未学習の `Sequential` を構築する。保存側と同じ `add_*` を呼ぶため、
/// `add_*` が内部で固定する値（bias あり・`ceil_mode=false` 等）も自動的に再現される。
/// 重みは直後に `load_state_dict` で上書きするため `seed` は意味を持たない（0 固定。
/// 決定記録 §5）。コンストラクタの引数検査（`p` の範囲・kernel=0 等）の失敗は
/// `Autodiff` として返す（部分的に構築したモデルは返さない）。
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
            LayerSpec::LeakyRelu { negative_slope } => model.add_leaky_relu(*negative_slope),
            LayerSpec::Elu { alpha } => model.add_elu(*alpha),
            LayerSpec::Softmax { dim } => model.add_softmax(*dim),
            LayerSpec::LogSoftmax { dim } => model.add_log_softmax(*dim),
            LayerSpec::Softplus { beta, threshold } => model
                .add_softplus(*beta, *threshold)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::Flatten { start_dim, end_dim } => model.add_flatten(*start_dim, *end_dim),
            LayerSpec::Dropout { p } => model.add_dropout(*p).map_err(ModelIoError::Autodiff)?,
            LayerSpec::Conv2d {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => model
                .add_conv2d(
                    *in_channels,
                    *out_channels,
                    *kernel_size,
                    *stride,
                    *padding,
                    *dilation,
                    *groups,
                    0,
                )
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::Conv1d {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => model
                .add_conv1d(
                    *in_channels,
                    *out_channels,
                    *kernel_size,
                    *stride,
                    *padding,
                    *dilation,
                    *groups,
                    0,
                )
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::LayerNorm {
                normalized_size,
                eps,
            } => model
                .add_layer_norm(*normalized_size, *eps)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::RmsNorm {
                normalized_size,
                eps,
            } => model
                .add_rms_norm(*normalized_size, *eps)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::BatchNorm1d {
                num_features,
                eps,
                momentum,
            } => model
                .add_batch_norm1d(*num_features, *eps, *momentum)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::BatchNorm2d {
                num_features,
                eps,
                momentum,
            } => model
                .add_batch_norm2d(*num_features, *eps, *momentum)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::Embedding {
                num_embeddings,
                embedding_dim,
                padding_idx,
            } => model
                .add_embedding(*num_embeddings, *embedding_dim, *padding_idx, 0)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::MultiheadAttention {
                embed_dim,
                num_heads,
            } => model
                .add_multihead_attention(*embed_dim, *num_heads, 0)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::TransformerEncoder {
                d_model,
                num_heads,
                dim_feedforward,
            } => model
                .add_transformer_encoder(*d_model, *num_heads, *dim_feedforward, 0)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::MaxPool2d {
                kernel_size,
                stride,
                padding,
                dilation,
            } => model
                .add_max_pool2d(*kernel_size, *stride, *padding, *dilation)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::MaxPool1d {
                kernel_size,
                stride,
                padding,
                dilation,
            } => model
                .add_max_pool1d(*kernel_size, *stride, *padding, *dilation)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::AvgPool2d {
                kernel_size,
                stride,
                padding,
                count_include_pad,
            } => model
                .add_avg_pool2d(*kernel_size, *stride, *padding, *count_include_pad)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::AvgPool1d {
                kernel_size,
                stride,
                padding,
                count_include_pad,
            } => model
                .add_avg_pool1d(*kernel_size, *stride, *padding, *count_include_pad)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::AdaptiveAvgPool2d { output_size } => model
                .add_adaptive_avg_pool2d(*output_size)
                .map_err(ModelIoError::Autodiff)?,
            LayerSpec::AdaptiveAvgPool1d { output_size } => model
                .add_adaptive_avg_pool1d(*output_size)
                .map_err(ModelIoError::Autodiff)?,
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

/// 本形式が受理する JSON 値（非負整数・小数の生字句・エスケープなし文字列・bool・null・配列・object）。
#[derive(Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(u64),
    /// 小数点・指数・負号のいずれかを含む数値の生の字句（f32 欄用。型付き変換は [`as_f32`]）。
    Real(String),
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
            Some(b'0'..=b'9' | b'-') => self.number(),
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

    /// JSON 数値の完全な文法（`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`）で字句を切り出す。
    /// 先頭ゼロ・`1.`・`.5`・`1e` 等の不正な字句は構文エラー、字句長は
    /// `MAX_NUMBER_LEXEME_LEN` 以内に限る。符号・小数点・指数のいずれも無い字句は非負整数
    /// （`Json::Num`。u64 の桁あふれは拒否）、それ以外は生の字句のまま `Json::Real` にする
    /// （整数欄が `Real` を受け付けない厳格さは `as_u64` が保つ）。
    fn number(&mut self) -> Result<Json, ModelIoError> {
        let start = self.pos;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.fail("数値に先頭ゼロは使えません"));
                }
            }
            Some(b'1'..=b'9') => self.skip_digits(),
            _ => return Err(self.fail("数値の整数部が不正です")),
        }
        let mut is_real = negative;
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.require_digits("小数点の後ろに数字がありません")?;
            is_real = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            self.require_digits("指数部に数字がありません")?;
            is_real = true;
        }
        if self.pos - start > MAX_NUMBER_LEXEME_LEN {
            return Err(self.fail("数値の字句が長すぎます"));
        }
        // 切り出した範囲は ASCII のみ（上の文法で数字・符号・`.`・`e`／`E` に限る）。
        let lexeme = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.fail("数値が UTF-8 ではありません"))?;
        if is_real {
            return Ok(Json::Real(lexeme.to_string()));
        }
        lexeme
            .parse::<u64>()
            .map(Json::Num)
            .map_err(|_| self.fail("整数が u64 に収まりません"))
    }

    fn skip_digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
    }

    fn require_digits(&mut self, what: &str) -> Result<(), ModelIoError> {
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(self.fail(what));
        }
        self.skip_digits();
        Ok(())
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

/// f32 欄。`Json::Real`（小数点または指数を含む字句）だけを受理し、有限かつ
/// **正準形**（`{:?}` で書き戻した文字列と一致）のものに限る。`0.50`・`1`・`1E0` のような
/// 別表記は、ライタ（`{:?}`）が出さない改竄として拒否する（bit 一致の往復を保証する）。
fn as_f32(value: &Json, ctx: &str) -> Result<f32, ModelIoError> {
    let Json::Real(raw) = value else {
        return Err(manifest_error(format!(
            "{ctx} は小数表記の数値（f32）である必要があります"
        )));
    };
    let v: f32 = raw
        .parse()
        .map_err(|_| manifest_error(format!("{ctx} を f32 として読めません")))?;
    if !v.is_finite() {
        return Err(manifest_error(format!("{ctx} が有限の f32 ではありません")));
    }
    if format!("{v:?}") != *raw {
        return Err(manifest_error(format!(
            "{ctx} が正準な f32 表記ではありません"
        )));
    }
    Ok(v)
}

fn as_bool(value: &Json, ctx: &str) -> Result<bool, ModelIoError> {
    match value {
        Json::Bool(b) => Ok(*b),
        _ => Err(manifest_error(format!(
            "{ctx} は bool である必要があります"
        ))),
    }
}

/// `null` または非負整数。
fn as_opt_usize(value: &Json, ctx: &str) -> Result<Option<usize>, ModelIoError> {
    match value {
        Json::Null => Ok(None),
        other => as_usize(other, ctx).map(Some),
    }
}

/// `params` object の固定スキーマ読み取り。生成時に**キー集合の完全一致**（未知・欠落を
/// `Manifest` で拒否）を確認し、以降の取得は型付き（別の型は `Manifest`）で行う。
struct Params<'a> {
    entries: &'a [(String, Json)],
}

impl<'a> Params<'a> {
    fn new(params: &'a Json, keys: &[&str]) -> Result<Self, ModelIoError> {
        exact_fields(params, "layers[].params", keys)?;
        match params {
            Json::Obj(entries) => Ok(Params { entries }),
            _ => Err(manifest_error(
                "layers[].params は object である必要があります",
            )),
        }
    }

    fn get(&self, key: &str) -> Result<&'a Json, ModelIoError> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .ok_or_else(|| manifest_error(format!("params にキー {key} がありません")))
    }

    fn usize(&self, key: &str) -> Result<usize, ModelIoError> {
        as_usize(self.get(key)?, key)
    }

    fn f32(&self, key: &str) -> Result<f32, ModelIoError> {
        as_f32(self.get(key)?, key)
    }

    fn bool(&self, key: &str) -> Result<bool, ModelIoError> {
        as_bool(self.get(key)?, key)
    }

    fn opt_usize(&self, key: &str) -> Result<Option<usize>, ModelIoError> {
        as_opt_usize(self.get(key)?, key)
    }

    /// `{base}_h`／`{base}_w` の 2 キーを `[usize; 2]` として読む。
    fn pair(&self, base: &str) -> Result<[usize; 2], ModelIoError> {
        Ok([
            self.usize(&format!("{base}_h"))?,
            self.usize(&format!("{base}_w"))?,
        ])
    }

    /// `{base}_h`／`{base}_w` が「両方 `null`」または「両方整数」の `Option<[usize; 2]>`。
    /// 片方だけ `null` は書き手が出さない形のため `Manifest` で拒否する。
    fn opt_pair(&self, base: &str) -> Result<Option<[usize; 2]>, ModelIoError> {
        let h = self.opt_usize(&format!("{base}_h"))?;
        let w = self.opt_usize(&format!("{base}_w"))?;
        match (h, w) {
            (Some(h), Some(w)) => Ok(Some([h, w])),
            (None, None) => Ok(None),
            _ => Err(manifest_error(format!(
                "{base}_h と {base}_w は両方 null か両方整数である必要があります"
            ))),
        }
    }
}

/// conv の groups 整合（`groups >= 1`・`in`／`out` の割り切れ）。非信頼な整数による
/// 除算パニックを、期待 shape を導く前にここで `Manifest` として拒否する。
fn check_conv_groups(
    in_channels: usize,
    out_channels: usize,
    groups: usize,
) -> Result<(), ModelIoError> {
    if groups == 0 || !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
        return Err(manifest_error(
            "conv の groups は 1 以上で in_channels・out_channels を割り切る必要があります",
        ));
    }
    Ok(())
}

/// `kind` と `params` から層構成を復元する。kind は文字列の allowlist（未知の値は
/// `UnsupportedModel`）で、`params` は kind ごとの固定スキーマ（未知キー・欠落・型違いは
/// `Manifest`）。意味上の範囲（`p` の範囲・kernel=0 等）は load 時の `add_*` が既存の検査で
/// `Autodiff` として拒否するため、ここでは重複して検査しない（構造上の整合だけを見る）。
fn spec_from_kind(kind: &str, params: &Json) -> Result<LayerSpec, ModelIoError> {
    let none = |spec: LayerSpec| -> Result<LayerSpec, ModelIoError> {
        Params::new(params, &[])?;
        Ok(spec)
    };
    match kind {
        "linear" => {
            let p = Params::new(params, &["in_features", "out_features"])?;
            Ok(LayerSpec::Linear {
                in_features: p.usize("in_features")?,
                out_features: p.usize("out_features")?,
            })
        }
        "relu" => none(LayerSpec::Relu),
        "sigmoid" => none(LayerSpec::Sigmoid),
        "tanh" => none(LayerSpec::Tanh),
        "silu" => none(LayerSpec::Silu),
        "hardswish" => none(LayerSpec::Hardswish),
        "gelu" => none(LayerSpec::Gelu),
        "gelu_tanh" => none(LayerSpec::GeluTanh),
        "leaky_relu" => {
            let p = Params::new(params, &["negative_slope"])?;
            Ok(LayerSpec::LeakyRelu {
                negative_slope: p.f32("negative_slope")?,
            })
        }
        "elu" => {
            let p = Params::new(params, &["alpha"])?;
            Ok(LayerSpec::Elu {
                alpha: p.f32("alpha")?,
            })
        }
        "softmax" => {
            let p = Params::new(params, &["dim"])?;
            Ok(LayerSpec::Softmax {
                dim: p.usize("dim")?,
            })
        }
        "log_softmax" => {
            let p = Params::new(params, &["dim"])?;
            Ok(LayerSpec::LogSoftmax {
                dim: p.usize("dim")?,
            })
        }
        "softplus" => {
            let p = Params::new(params, &["beta", "threshold"])?;
            Ok(LayerSpec::Softplus {
                beta: p.f32("beta")?,
                threshold: p.f32("threshold")?,
            })
        }
        "flatten" => {
            let p = Params::new(params, &["start_dim", "end_dim"])?;
            Ok(LayerSpec::Flatten {
                start_dim: p.usize("start_dim")?,
                end_dim: p.usize("end_dim")?,
            })
        }
        "dropout" => {
            let p = Params::new(params, &["p"])?;
            Ok(LayerSpec::Dropout { p: p.f32("p")? })
        }
        "conv2d" => {
            let p = Params::new(
                params,
                &[
                    "in_channels",
                    "out_channels",
                    "kernel_size_h",
                    "kernel_size_w",
                    "stride_h",
                    "stride_w",
                    "padding_h",
                    "padding_w",
                    "dilation_h",
                    "dilation_w",
                    "groups",
                ],
            )?;
            let spec = LayerSpec::Conv2d {
                in_channels: p.usize("in_channels")?,
                out_channels: p.usize("out_channels")?,
                kernel_size: p.pair("kernel_size")?,
                stride: p.pair("stride")?,
                padding: p.pair("padding")?,
                dilation: p.pair("dilation")?,
                groups: p.usize("groups")?,
            };
            if let LayerSpec::Conv2d {
                in_channels,
                out_channels,
                groups,
                ..
            } = &spec
            {
                check_conv_groups(*in_channels, *out_channels, *groups)?;
            }
            Ok(spec)
        }
        "conv1d" => {
            let p = Params::new(
                params,
                &[
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "stride",
                    "padding",
                    "dilation",
                    "groups",
                ],
            )?;
            let in_channels = p.usize("in_channels")?;
            let out_channels = p.usize("out_channels")?;
            let groups = p.usize("groups")?;
            check_conv_groups(in_channels, out_channels, groups)?;
            Ok(LayerSpec::Conv1d {
                in_channels,
                out_channels,
                kernel_size: p.usize("kernel_size")?,
                stride: p.usize("stride")?,
                padding: p.usize("padding")?,
                dilation: p.usize("dilation")?,
                groups,
            })
        }
        "layer_norm" => {
            let p = Params::new(params, &["normalized_size", "eps"])?;
            Ok(LayerSpec::LayerNorm {
                normalized_size: p.usize("normalized_size")?,
                eps: p.f32("eps")?,
            })
        }
        "rms_norm" => {
            let p = Params::new(params, &["normalized_size", "eps"])?;
            Ok(LayerSpec::RmsNorm {
                normalized_size: p.usize("normalized_size")?,
                eps: p.f32("eps")?,
            })
        }
        "batch_norm1d" => {
            let p = Params::new(params, &["num_features", "eps", "momentum"])?;
            Ok(LayerSpec::BatchNorm1d {
                num_features: p.usize("num_features")?,
                eps: p.f32("eps")?,
                momentum: p.f32("momentum")?,
            })
        }
        "batch_norm2d" => {
            let p = Params::new(params, &["num_features", "eps", "momentum"])?;
            Ok(LayerSpec::BatchNorm2d {
                num_features: p.usize("num_features")?,
                eps: p.f32("eps")?,
                momentum: p.f32("momentum")?,
            })
        }
        "embedding" => {
            let p = Params::new(params, &["num_embeddings", "embedding_dim", "padding_idx"])?;
            Ok(LayerSpec::Embedding {
                num_embeddings: p.usize("num_embeddings")?,
                embedding_dim: p.usize("embedding_dim")?,
                padding_idx: p.opt_usize("padding_idx")?,
            })
        }
        "multihead_attention" => {
            let p = Params::new(params, &["embed_dim", "num_heads"])?;
            Ok(LayerSpec::MultiheadAttention {
                embed_dim: p.usize("embed_dim")?,
                num_heads: p.usize("num_heads")?,
            })
        }
        "transformer_encoder" => {
            let p = Params::new(params, &["d_model", "num_heads", "dim_feedforward"])?;
            Ok(LayerSpec::TransformerEncoder {
                d_model: p.usize("d_model")?,
                num_heads: p.usize("num_heads")?,
                dim_feedforward: p.usize("dim_feedforward")?,
            })
        }
        "max_pool2d" => {
            let p = Params::new(
                params,
                &[
                    "kernel_size_h",
                    "kernel_size_w",
                    "stride_h",
                    "stride_w",
                    "padding_h",
                    "padding_w",
                    "dilation_h",
                    "dilation_w",
                ],
            )?;
            Ok(LayerSpec::MaxPool2d {
                kernel_size: p.pair("kernel_size")?,
                stride: p.opt_pair("stride")?,
                padding: p.pair("padding")?,
                dilation: p.pair("dilation")?,
            })
        }
        "max_pool1d" => {
            let p = Params::new(params, &["kernel_size", "stride", "padding", "dilation"])?;
            Ok(LayerSpec::MaxPool1d {
                kernel_size: p.usize("kernel_size")?,
                stride: p.opt_usize("stride")?,
                padding: p.usize("padding")?,
                dilation: p.usize("dilation")?,
            })
        }
        "avg_pool2d" => {
            let p = Params::new(
                params,
                &[
                    "kernel_size_h",
                    "kernel_size_w",
                    "stride_h",
                    "stride_w",
                    "padding_h",
                    "padding_w",
                    "count_include_pad",
                ],
            )?;
            Ok(LayerSpec::AvgPool2d {
                kernel_size: p.pair("kernel_size")?,
                stride: p.opt_pair("stride")?,
                padding: p.pair("padding")?,
                count_include_pad: p.bool("count_include_pad")?,
            })
        }
        "avg_pool1d" => {
            let p = Params::new(
                params,
                &["kernel_size", "stride", "padding", "count_include_pad"],
            )?;
            Ok(LayerSpec::AvgPool1d {
                kernel_size: p.usize("kernel_size")?,
                stride: p.opt_usize("stride")?,
                padding: p.usize("padding")?,
                count_include_pad: p.bool("count_include_pad")?,
            })
        }
        "adaptive_avg_pool2d" => {
            let p = Params::new(params, &["output_size_h", "output_size_w"])?;
            Ok(LayerSpec::AdaptiveAvgPool2d {
                output_size: p.pair("output_size")?,
            })
        }
        "adaptive_avg_pool1d" => {
            let p = Params::new(params, &["output_size"])?;
            Ok(LayerSpec::AdaptiveAvgPool1d {
                output_size: p.usize("output_size")?,
            })
        }
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
        // 構文として不正な字句（先頭ゼロ・u64 桁あふれ・不完全な小数／指数）。
        for s in [
            "01",
            "-01",
            "18446744073709551616",
            "-",
            "1.",
            ".5",
            "1e",
            "1e+",
            "--1",
            "+1",
        ] {
            assert!(is_manifest_err(parse_json(s)), "{s} は拒否されるはず");
        }
        assert!(matches!(parse_json("0"), Ok(Json::Num(0))));
        assert!(matches!(
            parse_json("18446744073709551615"),
            Ok(Json::Num(u64::MAX))
        ));
    }

    #[test]
    fn parser_keeps_fraction_exponent_and_sign_as_real_lexemes() {
        // 小数・指数・負号は `Real`（生の字句）。整数欄は `as_u64` が `Real` を拒否する。
        for s in ["1.5", "1e3", "-1", "0.5", "-0.0", "1e-5", "2.5E+3"] {
            match parse_json(s) {
                Ok(Json::Real(raw)) => assert_eq!(raw, s),
                other => panic!("{s} は Real のはず: {other:?}"),
            }
            assert!(is_manifest_err(as_u64(
                &parse_json(s).expect("読めるはず"),
                "x"
            )));
        }
        // 字句長の上限（64 バイト）を超える数値は拒否する。
        let long = format!("0.{}", "1".repeat(MAX_NUMBER_LEXEME_LEN));
        assert!(is_manifest_err(parse_json(&long)));
    }

    #[test]
    fn as_f32_accepts_only_finite_canonical_real() {
        let real = |s: &str| parse_json(s).expect("字句として読めるはず");
        for v in [
            0.5f32,
            1.0,
            -0.0,
            1e-5,
            1e20,
            f32::MIN_POSITIVE,
            0.1,
            f32::MAX,
        ] {
            let text = format!("{v:?}");
            let got = as_f32(&real(&text), "x").expect("正準形は受理されるはず");
            assert_eq!(got.to_bits(), v.to_bits(), "{text}");
        }
        // 非正準（末尾ゼロ・大文字 E・整数字句）・非有限（inf）・型違いは Manifest。
        for s in [
            "0.50", "1E0", "1", "0", "-1", "1e39", "true", "null", "\"0.5\"",
        ] {
            assert!(
                is_manifest_err(as_f32(&real(s), "x")),
                "{s} は拒否されるはず"
            );
        }
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
        struct Custom;
        impl crate::nn::Module for Custom {
            fn forward<'t>(
                &self,
                _tape: crate::TapeRef<'t>,
                input: &crate::Var<'t>,
            ) -> Result<crate::Var<'t>, AutodiffError> {
                Ok(*input)
            }
        }
        // 利用者定義層（`add_module`）は構成を記録できないため保存対象外。
        let custom = Sequential::new().add_relu().add_module(Custom);
        assert!(matches!(
            prepare_save(&custom),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        // f32 引数が非有限のモデルは JSON へ書けないため型付きで拒否する。
        let nan = Sequential::new().add_relu().add_leaky_relu(f32::NAN);
        assert!(matches!(
            prepare_save(&nan),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        let inf = Sequential::new().add_elu(f32::INFINITY);
        assert!(matches!(
            prepare_save(&inf),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
    }

    /// 30 種の `add_*` をすべて 1 回以上含む構成記録（stride の `Some`／`None`・
    /// `padding_idx` の `Some`／`None`・`count_include_pad` の真偽を両方含める）。
    fn all_kind_specs() -> Vec<LayerSpec> {
        vec![
            LayerSpec::Linear {
                in_features: 3,
                out_features: 4,
            },
            LayerSpec::Relu,
            LayerSpec::Sigmoid,
            LayerSpec::Tanh,
            LayerSpec::Silu,
            LayerSpec::Hardswish,
            LayerSpec::Gelu,
            LayerSpec::GeluTanh,
            LayerSpec::LeakyRelu {
                negative_slope: 0.125,
            },
            LayerSpec::Elu { alpha: 1.5 },
            LayerSpec::Softmax { dim: 1 },
            LayerSpec::LogSoftmax { dim: 0 },
            LayerSpec::Softplus {
                beta: 2.0,
                threshold: 20.0,
            },
            LayerSpec::Flatten {
                start_dim: 1,
                end_dim: 3,
            },
            LayerSpec::Dropout { p: 0.25 },
            LayerSpec::Conv2d {
                in_channels: 4,
                out_channels: 6,
                kernel_size: [3, 2],
                stride: [1, 2],
                padding: [1, 0],
                dilation: [1, 1],
                groups: 2,
            },
            LayerSpec::Conv1d {
                in_channels: 4,
                out_channels: 2,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 2,
            },
            LayerSpec::LayerNorm {
                normalized_size: 5,
                eps: 1e-5,
            },
            LayerSpec::RmsNorm {
                normalized_size: 5,
                eps: 1e-6,
            },
            LayerSpec::BatchNorm1d {
                num_features: 3,
                eps: 1e-5,
                momentum: 0.1,
            },
            LayerSpec::BatchNorm2d {
                num_features: 3,
                eps: 1e-5,
                momentum: 0.1,
            },
            LayerSpec::Embedding {
                num_embeddings: 7,
                embedding_dim: 4,
                padding_idx: Some(0),
            },
            LayerSpec::Embedding {
                num_embeddings: 7,
                embedding_dim: 4,
                padding_idx: None,
            },
            LayerSpec::MultiheadAttention {
                embed_dim: 8,
                num_heads: 2,
            },
            LayerSpec::TransformerEncoder {
                d_model: 8,
                num_heads: 2,
                dim_feedforward: 16,
            },
            LayerSpec::MaxPool2d {
                kernel_size: [2, 2],
                stride: Some([2, 1]),
                padding: [0, 0],
                dilation: [1, 1],
            },
            LayerSpec::MaxPool2d {
                kernel_size: [2, 2],
                stride: None,
                padding: [1, 1],
                dilation: [1, 1],
            },
            LayerSpec::MaxPool1d {
                kernel_size: 2,
                stride: Some(2),
                padding: 0,
                dilation: 1,
            },
            LayerSpec::MaxPool1d {
                kernel_size: 2,
                stride: None,
                padding: 0,
                dilation: 1,
            },
            LayerSpec::AvgPool2d {
                kernel_size: [2, 2],
                stride: Some([1, 1]),
                padding: [1, 1],
                count_include_pad: true,
            },
            LayerSpec::AvgPool2d {
                kernel_size: [2, 2],
                stride: None,
                padding: [0, 0],
                count_include_pad: false,
            },
            LayerSpec::AvgPool1d {
                kernel_size: 2,
                stride: Some(1),
                padding: 1,
                count_include_pad: true,
            },
            LayerSpec::AvgPool1d {
                kernel_size: 2,
                stride: None,
                padding: 0,
                count_include_pad: false,
            },
            LayerSpec::AdaptiveAvgPool2d {
                output_size: [2, 3],
            },
            LayerSpec::AdaptiveAvgPool1d { output_size: 4 },
        ]
    }

    #[test]
    fn all_thirty_kinds_are_covered_and_round_trip_through_params_schema() {
        let specs = all_kind_specs();
        let kinds: std::collections::BTreeSet<&str> = specs.iter().filter_map(spec_kind).collect();
        assert_eq!(kinds.len(), 30, "kind allowlist は 30 種: {kinds:?}");
        for spec in &specs {
            let kind = spec_kind(spec).expect("保存可能な層");
            let text = render_params(spec);
            let json = parse_json(&text).expect("自分が書いた params は読めるはず");
            // 全 kind の params キー数は object のキー数上限に収まる。
            if let Json::Obj(entries) = &json {
                assert!(entries.len() <= MAX_OBJECT_KEYS, "{kind}");
            }
            let back = spec_from_kind(kind, &json).expect("往復できるはず");
            assert_eq!(&back, spec, "{kind}");
            assert_eq!(render_params(&back), text, "{kind}");
        }
    }

    #[test]
    fn expected_keys_match_real_state_dict_for_all_kinds() {
        // 期待キー・shape の手書き導出が実モデルの state_dict（層の実装）と一致することを、
        // 30 種すべてで機械的に確認する（推測で書いた shape の誤りをここで検出する）。
        let specs = all_kind_specs();
        let model = build_model(&specs).expect("30 種を構築できるはず");
        let state = model.state_dict();
        let expected = expected_parameter_keys(&specs);
        assert_eq!(state.len(), expected.len());
        for (k, shape) in &expected {
            let t = state.get(k).unwrap_or_else(|| panic!("キー {k} がない"));
            assert_eq!(t.shape(), shape.as_slice(), "{k}");
        }
        let prepared = prepare_save(&model).expect("30 種を含むモデルを検証できるはず");
        assert_eq!(prepared.parameter_keys, expected);
    }

    #[test]
    fn spec_from_kind_rejects_schema_violations() {
        let p = |s: &str| parse_json(s).expect("JSON として読めるはず");
        // 未知の kind は UnsupportedModel。
        assert!(matches!(
            spec_from_kind("no_such_kind", &p("{}")),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        // 未知キー・欠落キー・型違い（整数欄に小数・f32 欄に整数・文字列）は Manifest。
        for (kind, params) in [
            ("dropout", r#"{"p":0.5,"q":1}"#),
            ("dropout", "{}"),
            ("dropout", r#"{"p":1}"#),
            ("dropout", r#"{"p":"0.5"}"#),
            ("dropout", r#"{"p":0.50}"#),
            ("dropout", r#"{"p":1e39}"#),
            ("softmax", r#"{"dim":1.0}"#),
            ("softmax", r#"{"dim":-1}"#),
            ("relu", r#"{"x":1}"#),
            (
                "embedding",
                r#"{"num_embeddings":3,"embedding_dim":2,"padding_idx":1.0}"#,
            ),
            (
                "avg_pool1d",
                r#"{"kernel_size":2,"stride":null,"padding":0,"count_include_pad":1}"#,
            ),
            // stride の null 対が不整合（片方だけ null）。
            (
                "max_pool2d",
                r#"{"kernel_size_h":2,"kernel_size_w":2,"stride_h":null,"stride_w":1,"padding_h":0,"padding_w":0,"dilation_h":1,"dilation_w":1}"#,
            ),
            // groups=0・割り切れない groups（除算パニックせず Manifest）。
            (
                "conv1d",
                r#"{"in_channels":4,"out_channels":2,"kernel_size":3,"stride":1,"padding":0,"dilation":1,"groups":0}"#,
            ),
            (
                "conv1d",
                r#"{"in_channels":4,"out_channels":2,"kernel_size":3,"stride":1,"padding":0,"dilation":1,"groups":3}"#,
            ),
            (
                "conv2d",
                r#"{"in_channels":4,"out_channels":3,"kernel_size_h":3,"kernel_size_w":3,"stride_h":1,"stride_w":1,"padding_h":0,"padding_w":0,"dilation_h":1,"dilation_w":1,"groups":2}"#,
            ),
        ] {
            assert!(
                is_manifest_err(spec_from_kind(kind, &p(params))),
                "{kind} {params} は Manifest で拒否されるはず"
            );
        }
    }

    #[test]
    fn out_of_range_values_are_rejected_by_constructors_on_build() {
        // 意味上の範囲（p の範囲・kernel=0・beta<=0 等）は load 時の `add_*` が拒否する。
        for spec in [
            LayerSpec::Dropout { p: 2.0 },
            LayerSpec::Softplus {
                beta: -1.0,
                threshold: 20.0,
            },
            LayerSpec::MaxPool1d {
                kernel_size: 0,
                stride: None,
                padding: 0,
                dilation: 1,
            },
            LayerSpec::Embedding {
                num_embeddings: 3,
                embedding_dim: 2,
                padding_idx: Some(3),
            },
            LayerSpec::BatchNorm1d {
                num_features: 2,
                eps: 1e-5,
                momentum: 2.0,
            },
        ] {
            assert!(
                matches!(
                    build_model(std::slice::from_ref(&spec)),
                    Err(ModelIoError::Autodiff(_))
                ),
                "{spec:?} は構築時に拒否されるはず"
            );
        }
    }

    #[test]
    fn prepare_save_rejects_models_whose_manifest_cannot_be_read_back() {
        // TransformerEncoder は 16 パラメータ/層のため、4096 層より手前で
        // `MAX_ARRAY_LEN` を超える。書けるのに読めない manifest を作らず、書き込み前に拒否する。
        let mut model = Sequential::new();
        for _ in 0..(MAX_ARRAY_LEN / 16 + 1) {
            model = model
                .add_transformer_encoder(2, 1, 1, 0)
                .expect("構築できるはず");
        }
        assert!(matches!(
            prepare_save(&model),
            Err(ModelIoError::TooLarge {
                what: "JSON 配列の要素数",
                ..
            })
        ));
    }

    #[test]
    fn prepare_save_rejects_mode_mismatch_and_dirty_batch_norm_stats() {
        // dropout だけがコンテナと異なるモード（eval の後に積んだ）→ 拒否。
        let mut model = Sequential::new()
            .add_linear(2, 2, 1)
            .expect("構築できるはず");
        model.eval();
        let model = model.add_dropout(0.5).expect("構築できるはず");
        assert!(matches!(
            prepare_save(&model),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        // eval の後に積んだ層でも、モード依存でない層（relu 等）は拒否しない。
        let mut ok = Sequential::new()
            .add_linear(2, 2, 1)
            .expect("構築できるはず");
        ok.eval();
        let ok = ok.add_relu();
        assert!(prepare_save(&ok).is_ok());
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
        let kind = good.replace("\"relu\"", "\"no_such_kind\"");
        assert!(matches!(
            parse_manifest(kind.as_bytes()),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        // 既知の kind でも params が固定スキーマと合わなければ Manifest（改竄の拒否）。
        let wrong_params = good.replace("\"relu\"", "\"conv2d\"");
        assert!(is_manifest_err(parse_manifest(wrong_params.as_bytes())));
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
