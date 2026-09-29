//! `load_model` が改竄された manifest・safetensors を fail-closed で拒否することの統合テスト
//! （イシュー #2375・親 #2362。決定記録 `docs/compat-model-io-decision.md` §6・§13.5）。
//!
//! 役割: 決定記録 §6 が列挙する改竄パターンと §13.5 の「非信頼値ごとの信頼境界」の各行を、
//! 公開 API（`save_model`／`load_model`）と `std` だけで固定する。各ケースは次の 4 点を検証する。
//!
//! 1. `load_model` が `Err` を返す（部分的な `Sequential` を返さない）
//! 2. `ModelIoError` の variant が期待どおり（表の「期待 variant」）
//! 3. load は dir を変更しない（前後で全エントリの名前と内容が同一）
//! 4. 陽性対照: 改竄前の内容へ戻すと `Ok` になる（ベースラインが壊れていて「全部拒否されたので
//!    合格」になる偽陽性を防ぐ）
//!
//! `format_version: 1` は旧形式（BatchNorm を含まないモデルで受理される後方互換）のため負例に
//! 使わない。symlink・TOCTOU・非通常ファイルは #2376 の範囲で、既存テストを参照する。
//! 上限定数（`MAX_LAYERS` 等）は非公開のため、実値（4096／8192／16／1048576 等）で境界を突く。
//!
//! # 対応表 A（§6 の列挙 → テスト）
//!
//! | §6 の項目 | 本ファイルのテスト | 既存テスト | 期待 variant |
//! |---|---|---|---|
//! | 未知キー | `unknown_keys_are_rejected_at_every_nesting_level` | `compat_sequential_model_io.rs::load_rejects_unknown_key_and_unsupported_manifest_features` | `Manifest` |
//! | 重複キー | `duplicate_object_keys_are_rejected` | `_compiled.rs::tampered_state_keys_are_rejected`（配列要素） | `Manifest` |
//! | 版違い | `format_and_version_mismatch_is_rejected` | 単体 `parse_manifest_*` | `Manifest` |
//! | 層数の上限超過 | `layer_count_over_bound_is_rejected_before_opening_safetensors` | 単体 `parse_manifest_rejects_layer_count_over_bound` | `TooLarge` |
//! | JSON 深さの超過 | `json_depth_over_bound_is_rejected` | 単体 `parser_rejects_excess_depth_before_recursing` | `Manifest` |
//! | 配列長・キー数・数値字句長 | `json_container_and_lexeme_bounds_are_enforced` | 単体 `parser_caps_array_length_and_object_keys` | `TooLarge`／`Manifest` |
//! | manifest サイズ | `manifest_size_bound_is_exact` | `compat_sequential_model_io.rs::load_rejects_oversized_manifest` | `TooLarge` |
//! | safetensors サイズ | `oversized_safetensors_is_rejected_without_reading` | — | `TooLarge` |
//! | 非有限値 | `non_finite_and_non_numeric_f32_are_rejected` | `_layers.rs`・`_lbfgs.rs::tampered_config_fields_are_rejected` | `Manifest` |
//! | 壊れた manifest のバイト列 | `malformed_manifest_bytes_are_rejected` | — | `Manifest` |
//! | キー集合の不一致 | `safetensors_parameter_key_set_mismatch_is_rejected`・`manifest_parameter_key_set_mismatch_is_rejected` | `_batch_norm.rs`・`_compiled.rs::tampered_state_keys_are_rejected` | `Mismatch` |
//! | shape の不一致 | `safetensors_parameter_shape_mismatch_is_rejected` | `_layers.rs::tampered_parameter_shape_is_rejected` | `Mismatch` |
//! | safetensors の壊れた中身 | `corrupt_safetensors_payload_is_rejected` | — | `Safetensors` |
//! | `safetensors_file` パターン違反・区切り文字 | `safetensors_file_pattern_violations_are_rejected` | `compat_sequential_model_io.rs::load_rejects_bad_safetensors_file_pattern` | `Manifest` |
//! | 存在しない世代・旧世代参照 | `valid_pattern_pointing_to_missing_file_is_rejected`・`stale_generation_reference_is_rejected` | — | `Io(NotFound)`／`Mismatch` |
//! | `safetensors_bytes` の不一致 | `safetensors_bytes_tampering_is_rejected` | `compat_sequential_model_io.rs::load_rejects_safetensors_bytes_mismatch` | `Mismatch`／`Manifest` |
//! | optimizer 種別の取り違え | `optimizer_kind_swap_is_rejected` | `_compiled.rs::tampered_kind_is_rejected` | 組による |
//! | `history_len` | `history_len_type_and_bound_violations_are_rejected` | `_lbfgs.rs::history_len_mismatch_with_real_keys_is_rejected` | `Manifest`／`TooLarge`／`Mismatch` |
//! | `growth_tracker` の範囲外 | `growth_tracker_out_of_range_is_rejected` | `_compiled.rs::tampered_amp_state_is_rejected` | `Autodiff`／`Manifest` |
//!
//! # 対応表 B（§13.5 の非信頼値の各行 → テスト）
//!
//! | 非信頼値 | テスト |
//! |---|---|
//! | `dir`・コードが生成する世代 ID | N/A（信頼値。境界不要） |
//! | dir 配下のファイル名・種別（symlink・非通常ファイル） | `compat_sequential_model_io_fs_threats.rs`・`compat_sequential_model_io.rs::load_rejects_symlinked_manifest_and_safetensors`／`load_rejects_non_regular_manifest_and_safetensors`（#2376） |
//! | dir 配下の既存ファイルの削除 | 削除しない契約。本ファイルは全ケースで load 前後の dir 不変を検証する |
//! | manifest のバイト列 | `manifest_size_bound_is_exact`・`malformed_manifest_bytes_are_rejected`・`json_*` |
//! | `format`・`format_version` | `format_and_version_mismatch_is_rejected` |
//! | `num_layers`・`layers` | `layer_count_over_bound_*`・`num_layers_and_index_inconsistency_is_rejected` |
//! | `parameter_keys`・`buffer_keys` | `manifest_parameter_key_set_mismatch_is_rejected`（buffer は `_batch_norm.rs`） |
//! | `safetensors_file` | `safetensors_file_pattern_violations_are_rejected`・`valid_pattern_*`・`stale_generation_*` |
//! | `safetensors_bytes` | `safetensors_bytes_tampering_is_rejected`・`oversized_safetensors_*` |
//! | `compiled.optimizer.kind` | `optimizer_kind_swap_is_rejected` |
//! | `compiled.optimizer.config` | `non_finite_*`・`_compiled.rs::tampered_config_is_rejected` |
//! | `optimizer.history_len` | `history_len_type_and_bound_violations_are_rejected` |
//! | `config.history_size`・`line_search`・`max_eval` | `_lbfgs.rs::tampered_config_fields_are_rejected` |
//! | `amp.scale` | `non_finite_and_non_numeric_f32_are_rejected`・`_compiled.rs::tampered_amp_state_is_rejected` |
//! | `amp.growth_tracker` | `growth_tracker_out_of_range_is_rejected` |
//! | Lbfgs 状態テンソル | `_lbfgs.rs::non_finite_or_invalid_state_tensors_are_rejected` |
//! | rename・作成先パス | 保存側の契約（`compat_sequential_model_io_fs_threats.rs`） |

#![cfg(unix)]

use std::path::{Path, PathBuf};

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model,
    save_model,
};
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes,
};
use fandhe_ai::optim::{AdamConfig, AdamWConfig, LbfgsConfig, RmsPropConfig, SgdConfig};

mod common;
use common::temp_dir::TempDirGuard;

const MAX_MANIFEST: usize = 1 << 20;
const MAX_SAFETENSORS: u64 = 1 << 30;

// ---------------------------------------------------------------------
// 期待 variant と検証ヘルパー
// ---------------------------------------------------------------------

/// 期待する `ModelIoError` の variant（メッセージ本文には依存しない）。
#[derive(Clone, Copy, Debug)]
enum Want {
    Manifest,
    Mismatch,
    Autodiff,
    Safetensors,
    Unsupported,
    NotFound,
    TooLarge(&'static str, u64),
}

impl Want {
    fn matches(self, e: &ModelIoError) -> bool {
        match (self, e) {
            (Want::Manifest, ModelIoError::Manifest { .. })
            | (Want::Mismatch, ModelIoError::Mismatch { .. })
            | (Want::Autodiff, ModelIoError::Autodiff(_))
            | (Want::Safetensors, ModelIoError::Safetensors(_))
            | (Want::Unsupported, ModelIoError::UnsupportedModel { .. }) => true,
            (Want::NotFound, ModelIoError::Io(io)) => io.kind() == std::io::ErrorKind::NotFound,
            (Want::TooLarge(w, l), ModelIoError::TooLarge { what, limit }) => {
                w == *what && l == *limit
            }
            _ => false,
        }
    }
}

/// 保存済みの dir と、改竄前の manifest・safetensors（陽性対照の復元用）。
struct Fx {
    _guard: TempDirGuard,
    dir: PathBuf,
    manifest: Vec<u8>,
    st_name: String,
    st: Vec<u8>,
}

impl Fx {
    fn new(m: &Sequential, label: &str) -> Self {
        let guard = TempDirGuard::new(label);
        let dir = guard.path().join("m");
        save_model(m, &dir).expect("save");
        let manifest = std::fs::read(dir.join("manifest.json")).expect("manifest");
        let text = String::from_utf8(manifest.clone()).expect("utf8");
        let st_name = string_field(&text, "safetensors_file");
        let st = std::fs::read(dir.join(&st_name)).expect("safetensors");
        Self {
            _guard: guard,
            dir,
            manifest,
            st_name,
            st,
        }
    }

    fn text(&self) -> String {
        String::from_utf8(self.manifest.clone()).expect("utf8")
    }

    fn st_path(&self) -> PathBuf {
        self.dir.join(&self.st_name)
    }

    fn write_manifest(&self, bytes: &[u8]) {
        std::fs::write(self.dir.join("manifest.json"), bytes).expect("write manifest");
    }

    /// 改竄前の manifest・safetensors へ戻し、`load_model` が成功することを確かめる（陽性対照）。
    fn restore_and_assert_loads(&self, label: &str) {
        self.write_manifest(&self.manifest);
        std::fs::write(self.st_path(), &self.st).expect("restore safetensors");
        if let Err(e) = load_model(&self.dir) {
            panic!("{label}: 改竄前へ戻したのに load できない（ベースライン破損）: {e}");
        }
    }

    /// manifest を `bytes` へ差し替えて load が `want` で拒否され、dir が不変であることを検証する。
    fn reject_manifest(&self, label: &str, bytes: &[u8], want: Want) {
        self.write_manifest(bytes);
        self.assert_rejected(label, want);
        self.restore_and_assert_loads(label);
    }

    fn reject_text(&self, label: &str, text: &str, want: Want) {
        self.reject_manifest(label, text.as_bytes(), want);
    }

    /// 現在の dir の状態で load が `want` で拒否され、load が dir を変更しないことを検証する。
    fn assert_rejected(&self, label: &str, want: Want) {
        let before = snapshot(&self.dir);
        match load_model(&self.dir) {
            Ok(_) => panic!("{label}: 改竄されたのに load が成功した"),
            Err(e) => assert!(
                want.matches(&e),
                "{label}: variant が想定 {want:?} と異なる: {e:?}（{e}）"
            ),
        }
        assert_eq!(
            before,
            snapshot(&self.dir),
            "{label}: load が dir を変更した"
        );
    }

    /// safetensors を読み替えて書き戻し、manifest の `safetensors_bytes` も追従させる。
    fn rewrite_safetensors(
        &self,
        edit: impl FnOnce(&mut std::collections::HashMap<String, Tensor<f32>>),
    ) {
        let mut map = load_safetensors_f32_from_bytes(&self.st).expect("decode");
        edit(&mut map);
        let bytes = save_safetensors_f32_to_bytes(&map, None).expect("encode");
        std::fs::write(self.st_path(), &bytes).expect("write st");
        let text = set_field(&self.text(), "safetensors_bytes", &bytes.len().to_string());
        self.write_manifest(text.as_bytes());
    }
}

fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .expect("readdir")
        .flatten()
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("read"),
            )
        })
        .collect();
    v.sort();
    v
}

/// `"key":<value>` の value（次の `,` または `}` まで）を置き換える（最初の出現のみ）。
fn set_field(text: &str, key: &str, new_value: &str) -> String {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle).unwrap_or_else(|| panic!("{key} が無い")) + needle.len();
    let end = start + text[start..].find([',', '}']).expect("終端");
    format!("{}{}{}", &text[..start], new_value, &text[end..])
}

fn raw_field(text: &str, key: &str) -> String {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle).unwrap_or_else(|| panic!("{key} が無い")) + needle.len();
    let end = start + text[start..].find([',', '}']).expect("終端");
    text[start..end].to_string()
}

fn string_field(text: &str, key: &str) -> String {
    raw_field(text, key).trim_matches('"').to_string()
}

/// `from` が 1 回以上現れることを確かめてから最初の出現を `to` へ置き換える。
fn replace_once(text: &str, from: &str, to: &str) -> String {
    assert!(text.contains(from), "置換対象が manifest に無い: {from}");
    text.replacen(from, to, 1)
}

// ---------------------------------------------------------------------
// フィクスチャ
// ---------------------------------------------------------------------

fn data() -> (Tensor<f32>, Tensor<f32>) {
    let x: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.41 + 0.3).sin()).collect();
    let y: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.23 + 1.1).cos()).collect();
    (
        Tensor::new(x, &[8, 3]).expect("x"),
        Tensor::new(y, &[8, 2]).expect("y"),
    )
}

fn build() -> Sequential {
    Sequential::new()
        .add_linear(3, 4, 7)
        .and_then(|m| {
            m.add_relu()
                .add_dropout(0.25)
                .and_then(|m| m.add_linear(4, 2, 8))
        })
        .expect("構築できるはず")
}

/// 未 compile のモデル（Linear→ReLU→Dropout→Linear）。
fn plain() -> Fx {
    Fx::new(&build(), "plain")
}

/// AdamW＋AMP（F16）で 2 epoch 学習した compile 済みモデル。
fn amp() -> Fx {
    let (x, y) = data();
    let mut m = build();
    m.compile_with_amp(
        Optimizer::AdamW(AdamWConfig::default()),
        Loss::Mse,
        AmpConfig::new(AmpDType::F16),
    )
    .expect("compile_with_amp");
    m.fit(&x, &y, FitConfig::new(2, 4)).expect("fit");
    Fx::new(&m, "amp")
}

/// AMP なしで compile し、1 epoch 学習したモデル。
fn compiled(opt: Optimizer, label: &str) -> Fx {
    let (x, y) = data();
    let mut m = build();
    m.compile(opt, Loss::Mse).expect("compile");
    m.fit(&x, &y, FitConfig::new(1, 4)).expect("fit");
    Fx::new(&m, label)
}

/// 履歴が `history_size = 3` に到達した Lbfgs モデル。
fn lbfgs() -> Fx {
    let (x, y) = data();
    let mut m = build();
    m.compile(
        Optimizer::Lbfgs(LbfgsConfig {
            lr: 0.05,
            max_iter: 4,
            history_size: 3,
            ..LbfgsConfig::default()
        }),
        Loss::Mse,
    )
    .expect("compile");
    m.fit(&x, &y, FitConfig::new(3, 8)).expect("fit");
    Fx::new(&m, "lbfgs")
}

// ---------------------------------------------------------------------
// 構文・構造（§6: 未知キー・重複キー・版違い）
// ---------------------------------------------------------------------

#[test]
fn unknown_keys_are_rejected_at_every_nesting_level() {
    let fx = amp();
    let t = fx.text();
    for (label, from, to) in [
        ("最上位", "{\"format\":", "{\"zz\":0,\"format\":"),
        ("layers[]", "{\"index\":0,", "{\"zz\":0,\"index\":0,"),
        (
            "layers[].params",
            "\"params\":{\"in_features\":3",
            "\"params\":{\"zz\":0,\"in_features\":3",
        ),
        (
            "parameter_keys[]",
            "{\"key\":\"0.weight\",",
            "{\"zz\":0,\"key\":\"0.weight\",",
        ),
        ("compiled", "\"compiled\":{", "\"compiled\":{\"zz\":0,"),
        ("optimizer", "\"optimizer\":{", "\"optimizer\":{\"zz\":0,"),
        ("config", "\"config\":{", "\"config\":{\"zz\":0,"),
        ("amp", "\"amp\":{", "\"amp\":{\"zz\":0,"),
        (
            "grad_scaler_config",
            "\"grad_scaler_config\":{",
            "\"grad_scaler_config\":{\"zz\":0,",
        ),
    ] {
        fx.reject_text(label, &replace_once(&t, from, to), Want::Manifest);
    }
}

#[test]
fn duplicate_object_keys_are_rejected() {
    let fx = amp();
    let t = fx.text();
    for (label, from, to) in [
        (
            "最上位",
            "\"training\":true,",
            "\"training\":true,\"training\":true,",
        ),
        (
            "最上位（値違いの重複）",
            "\"training\":true,",
            "\"training\":true,\"training\":false,",
        ),
        ("params", "\"p\":0.25", "\"p\":0.25,\"p\":0.25"),
        ("config", "\"lr\":0.001,", "\"lr\":0.001,\"lr\":0.001,"),
        (
            "amp",
            "\"scale\":65536.0,",
            "\"scale\":65536.0,\"scale\":65536.0,",
        ),
    ] {
        fx.reject_text(label, &replace_once(&t, from, to), Want::Manifest);
    }
}

#[test]
fn format_and_version_mismatch_is_rejected() {
    let fx = plain();
    let t = fx.text();
    for v in [
        "0",
        "3",
        "2.0",
        "\"2\"",
        "-2",
        "null",
        "18446744073709551616",
    ] {
        let label = format!("format_version={v}");
        fx.reject_text(
            &label,
            &replace_once(
                &t,
                "\"format_version\":2",
                &format!("\"format_version\":{v}"),
            ),
            Want::Manifest,
        );
    }
    for v in [
        "\"fandhe-ai.compat.sequentiaL\"",
        "\"Fandhe-ai.compat.sequential\"",
        "\"fandhe-ai.compat.sequential \"",
        "\"\"",
        "1",
        "null",
    ] {
        let label = format!("format={v}");
        fx.reject_text(
            &label,
            &replace_once(
                &t,
                "\"format\":\"fandhe-ai.compat.sequential\"",
                &format!("\"format\":{v}"),
            ),
            Want::Manifest,
        );
    }
    // 旧形式（format_version 1）は BatchNorm を含まないモデルでは受理される（後方互換の意図の固定）。
    fx.write_manifest(replace_once(&t, "\"format_version\":2", "\"format_version\":1").as_bytes());
    assert!(load_model(&fx.dir).is_ok(), "旧形式は受理される契約");
    fx.restore_and_assert_loads("legacy");
}

// ---------------------------------------------------------------------
// 資源の上限（§6: 層数・深さ・サイズの超過）
// ---------------------------------------------------------------------

#[test]
fn layer_count_over_bound_is_rejected_before_opening_safetensors() {
    let fx = plain();
    let n = 4097;
    let layers: Vec<String> = (0..n)
        .map(|i| format!("{{\"index\":{i},\"kind\":\"relu\",\"params\":{{}}}}"))
        .collect();
    let t = format!(
        "{{\"format\":\"fandhe-ai.compat.sequential\",\"format_version\":2,\"training\":true,\
         \"num_layers\":{n},\"layers\":[{}],\"parameter_keys\":[],\"buffer_keys\":[],\
         \"safetensors_file\":\"{}\",\"safetensors_bytes\":8,\"compiled\":null}}",
        layers.join(","),
        fx.st_name
    );
    assert!(t.len() < MAX_MANIFEST, "サイズ上限より層数上限を先に突く");
    fx.reject_text("4097 層", &t, Want::TooLarge("層数", 4096));
}

#[test]
fn num_layers_and_index_inconsistency_is_rejected() {
    let fx = plain();
    let t = fx.text();
    for (label, from, to) in [
        ("num_layers+1", "\"num_layers\":4", "\"num_layers\":5"),
        ("num_layers-1", "\"num_layers\":4", "\"num_layers\":3"),
        ("index 0 重複", "{\"index\":1,", "{\"index\":0,"),
        ("index 飛び", "{\"index\":1,", "{\"index\":9,"),
        ("index 逆順", "{\"index\":3,", "{\"index\":1,"),
    ] {
        fx.reject_text(label, &replace_once(&t, from, to), Want::Manifest);
    }
}

#[test]
fn json_depth_over_bound_is_rejected() {
    let fx = plain();
    let t = fx.text();
    fx.reject_text(
        "深さ 5",
        &replace_once(&t, "\"compiled\":null", "\"compiled\":[[[[0]]]]"),
        Want::Manifest,
    );
    // 1 MiB 未満で再帰前の深さ検査に打ち切られ、スタックを使い切らずに Err が返る。
    let deep = "[".repeat(100_000);
    assert!(deep.len() < MAX_MANIFEST);
    fx.reject_text("[ x 100000", &deep, Want::Manifest);
}

#[test]
fn json_container_and_lexeme_bounds_are_enforced() {
    let fx = plain();
    let t = fx.text();
    let zeros = vec!["0"; 8193].join(",");
    fx.reject_text(
        "配列 8193 要素",
        &replace_once(&t, "\"compiled\":null", &format!("\"compiled\":[{zeros}]")),
        Want::TooLarge("JSON 配列の要素数", 8192),
    );
    let keys: Vec<String> = (0..17).map(|i| format!("\"k{i}\":0")).collect();
    fx.reject_text(
        "object 17 キー",
        &replace_once(
            &t,
            "\"kind\":\"relu\",\"params\":{}",
            &format!("\"kind\":\"relu\",\"params\":{{{}}}", keys.join(",")),
        ),
        Want::TooLarge("JSON object のキー数", 16),
    );
    fx.reject_text(
        "数値字句 65 桁",
        &set_field(&t, "safetensors_bytes", &"1".repeat(65)),
        Want::Manifest,
    );
}

#[test]
fn manifest_size_bound_is_exact() {
    let fx = plain();
    let mut exact = fx.manifest.clone();
    exact.resize(MAX_MANIFEST, b' ');
    fx.write_manifest(&exact);
    assert!(load_model(&fx.dir).is_ok(), "ちょうど上限は受理される");

    let mut over = fx.manifest.clone();
    over.resize(MAX_MANIFEST + 1, b' ');
    fx.reject_manifest(
        "上限 + 1",
        &over,
        Want::TooLarge("manifest.json", MAX_MANIFEST as u64),
    );
}

#[test]
fn oversized_safetensors_is_rejected_without_reading() {
    let fx = plain();
    // 疎ファイル（実際には 1 GiB を書かない）。長さだけが上限を超える。
    let len = MAX_SAFETENSORS + 1;
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(fx.st_path())
        .expect("open");
    f.set_len(len).expect("set_len");
    drop(f);
    let t = set_field(&fx.text(), "safetensors_bytes", &len.to_string());
    fx.write_manifest(t.as_bytes());
    fx.assert_rejected(
        "1 GiB + 1",
        Want::TooLarge("model safetensors", MAX_SAFETENSORS),
    );
    fx.restore_and_assert_loads("1 GiB + 1");
}

// ---------------------------------------------------------------------
// 値・整合性（§6: 非有限値・キー集合・shape）
// ---------------------------------------------------------------------

#[test]
fn non_finite_and_non_numeric_f32_are_rejected() {
    let fx = amp();
    let t = fx.text();
    for (label, key, v) in [
        ("amp.scale 1e39", "scale", "1e39"),
        ("amp.scale -1e39", "scale", "-1e39"),
        ("amp.scale NaN", "scale", "NaN"),
        ("amp.scale Infinity", "scale", "Infinity"),
        ("init_scale 1e39", "init_scale", "1e39"),
        ("growth_factor 1e39", "growth_factor", "1e39"),
        ("backoff_factor 1e39", "backoff_factor", "1e39"),
        ("config.lr 1e39", "lr", "1e39"),
    ] {
        fx.reject_text(label, &set_field(&t, key, v), Want::Manifest);
    }
}

#[test]
fn malformed_manifest_bytes_are_rejected() {
    let fx = plain();
    let good = fx.manifest.clone();
    let mut bom = vec![0xEF, 0xBB, 0xBF];
    bom.extend_from_slice(&good);
    let mut trailing = good.clone();
    trailing.push(b'x');
    let mut invalid_utf8 = good.clone();
    invalid_utf8.extend_from_slice(&[0xFF, 0xFE]);
    let truncated = good[..good.len() / 2].to_vec();
    for (label, bytes) in [
        ("非 UTF-8", invalid_utf8),
        ("途中で切れた JSON", truncated),
        ("空ファイル", Vec::new()),
        ("BOM 付き", bom),
        ("末尾の余分な内容", trailing),
    ] {
        fx.reject_manifest(label, &bytes, Want::Manifest);
    }
}

#[test]
fn safetensors_parameter_key_set_mismatch_is_rejected() {
    let fx = plain();
    type Edit = fn(&mut std::collections::HashMap<String, Tensor<f32>>);
    let cases: [(&str, Edit); 4] = [
        ("0.weight 削除", |m| {
            m.remove("0.weight");
        }),
        ("9.weight 追加", |m| {
            m.insert("9.weight".into(), Tensor::new(vec![0.0], &[1]).expect("t"));
        }),
        ("0.weight 改名", |m| {
            let t = m.remove("0.weight").expect("0.weight");
            m.insert("0.weigth".into(), t);
        }),
        ("無関係キー追加", |m| {
            m.insert("extra".into(), Tensor::new(vec![0.0], &[1]).expect("t"));
        }),
    ];
    for (label, edit) in cases {
        fx.rewrite_safetensors(edit);
        fx.assert_rejected(label, Want::Mismatch);
        fx.restore_and_assert_loads(label);
    }
}

#[test]
fn manifest_parameter_key_set_mismatch_is_rejected() {
    let fx = plain();
    let t = fx.text();
    let w = "{\"key\":\"0.weight\",\"shape\":[3,4]}";
    let b = "{\"key\":\"0.bias\",\"shape\":[4]}";
    for (label, from, to) in [
        ("1 件削除", format!("{w},"), String::new()),
        (
            "1 件追加",
            format!("{w},"),
            format!("{w},{{\"key\":\"9.weight\",\"shape\":[1]}},"),
        ),
        (
            "キー名の改名",
            "\"key\":\"0.weight\"".to_string(),
            "\"key\":\"0.weigth\"".to_string(),
        ),
        ("順序の入れ替え", format!("{w},{b}"), format!("{b},{w}")),
    ] {
        fx.reject_text(label, &replace_once(&t, &from, &to), Want::Mismatch);
    }
}

#[test]
fn safetensors_parameter_shape_mismatch_is_rejected() {
    let fx = plain();
    for (label, shape) in [("転置 [4,3]", vec![4usize, 3]), ("平坦化 [12]", vec![12])] {
        fx.rewrite_safetensors(|m| {
            let data = m["0.weight"]
                .contiguous()
                .as_slice()
                .expect("slice")
                .to_vec();
            m.insert("0.weight".into(), Tensor::new(data, &shape).expect("t"));
        });
        fx.assert_rejected(label, Want::Mismatch);
        fx.restore_and_assert_loads(label);
    }
}

#[test]
fn corrupt_safetensors_payload_is_rejected() {
    let fx = plain();
    let mut bad = fx.st.clone();
    bad[..8].copy_from_slice(&[0xFF; 8]);
    std::fs::write(fx.st_path(), &bad).expect("write");
    fx.assert_rejected("ヘッダ長を破壊", Want::Safetensors);
    let mut bad = fx.st.clone();
    bad[8] = b'!';
    std::fs::write(fx.st_path(), &bad).expect("write");
    fx.assert_rejected("ヘッダ JSON を破壊", Want::Safetensors);
    fx.restore_and_assert_loads("corrupt");
}

// ---------------------------------------------------------------------
// ファイル参照（§6: safetensors_file・パス区切り・safetensors_bytes）
// ---------------------------------------------------------------------

#[test]
fn safetensors_file_pattern_violations_are_rejected() {
    let fx = plain();
    let t = fx.text();
    let hex = fx.st_name["model.".len().."model.".len() + 32].to_string();
    let with_file = |raw: &str| set_field(&t, "safetensors_file", raw);
    let quoted = |s: String| format!("\"{s}\"");
    let cases: Vec<(&str, String)> = vec![
        ("空文字", quoted(String::new())),
        ("絶対パス /etc/passwd", quoted("/etc/passwd".into())),
        (
            "絶対パス（世代名付き）",
            quoted(format!("/abs/model.{hex}.safetensors")),
        ),
        ("./ 接頭辞", quoted(format!("./model.{hex}.safetensors"))),
        (
            "末尾スラッシュ",
            quoted(format!("model.{hex}.safetensors/")),
        ),
        (
            "サブディレクトリ",
            quoted(format!("sub/model.{hex}.safetensors")),
        ),
        ("..", quoted("..".into())),
        ("../ 接頭辞", quoted(format!("../model.{hex}.safetensors"))),
        ("31 桁", quoted(format!("model.{}.safetensors", &hex[..31]))),
        ("33 桁", quoted(format!("model.{hex}0.safetensors"))),
        (
            "大文字 16 進",
            quoted(format!(
                "model.{}.safetensors",
                hex.to_uppercase()
                    .replace('E', "F")
                    .replace('A', "B")
                    .replace('C', "D")
            )),
        ),
        (
            "非 16 進 g",
            quoted(format!("model.g{}.safetensors", &hex[1..])),
        ),
        ("前後の空白", quoted(format!(" model.{hex}.safetensors "))),
        ("大文字の名前", quoted(format!("MODEL.{hex}.SAFETENSORS"))),
        ("拡張子違い", quoted(format!("model.{hex}.safetensor"))),
        (
            "Windows 区切り",
            quoted(format!("sub\\\\model.{hex}.safetensors")),
        ),
        ("NUL", quoted(format!("model.{hex}.safetensors\\u0000"))),
        ("数値", "123".into()),
        ("null", "null".into()),
    ];
    for (label, raw) in cases {
        fx.reject_text(label, &with_file(&raw), Want::Manifest);
    }
}

#[test]
fn valid_pattern_pointing_to_missing_file_is_rejected() {
    let fx = plain();
    let missing = format!("model.{}.safetensors", "0".repeat(32));
    let t = set_field(&fx.text(), "safetensors_file", &format!("\"{missing}\""));
    fx.reject_text("存在しない世代", &t, Want::NotFound);
}

#[test]
fn stale_generation_reference_is_rejected() {
    // 構成の異なる 2 モデル（safetensors のバイト数が異なる）。B の manifest が A の世代を指すと、
    // 参照先は存在するが実バイト数が manifest の safetensors_bytes と一致しない（§12）。
    let a = plain();
    let b = amp();
    assert_ne!(a.st.len(), b.st.len(), "バイト数が異なる構成のはず");
    std::fs::write(b.dir.join(&a.st_name), &a.st).expect("旧世代を配置");
    let t = set_field(&b.text(), "safetensors_file", &format!("\"{}\"", a.st_name));
    b.reject_text("旧世代参照", &t, Want::Mismatch);
}

#[test]
fn safetensors_bytes_tampering_is_rejected() {
    let fx = plain();
    let t = fx.text();
    let n = fx.st.len() as u64;
    for (label, v) in [
        ("実値 + 1", (n + 1).to_string()),
        ("実値 - 1", (n - 1).to_string()),
        ("0", "0".into()),
        ("u64::MAX", u64::MAX.to_string()),
    ] {
        fx.reject_text(
            label,
            &set_field(&t, "safetensors_bytes", &v),
            Want::Mismatch,
        );
    }
    for (label, v) in [
        ("負値", "-1".to_string()),
        ("u64 桁あふれ", "18446744073709551616".to_string()),
        ("実数", format!("{n}.0")),
        ("文字列", format!("\"{n}\"")),
        ("null", "null".to_string()),
    ] {
        fx.reject_text(
            label,
            &set_field(&t, "safetensors_bytes", &v),
            Want::Manifest,
        );
    }
}

// ---------------------------------------------------------------------
// compile 状態（§6: optimizer 種別の取り違え・history_len・growth_tracker）
// ---------------------------------------------------------------------

#[test]
fn optimizer_kind_swap_is_rejected() {
    let sgd = || {
        compiled(
            Optimizer::Sgd(SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.0,
                weight_decay: 0.0,
                nesterov: false,
            }),
            "sgd",
        )
    };
    let adam = || compiled(Optimizer::Adam(AdamConfig::default()), "adam");
    let adamw = || compiled(Optimizer::AdamW(AdamWConfig::default()), "adamw");
    let rms = || compiled(Optimizer::RmsProp(RmsPropConfig::default()), "rmsprop");
    type Build<'a> = &'a dyn Fn() -> Fx;
    let cases: Vec<(Build, &str, &str, Want)> = vec![
        // config のキー集合が同じ組は safetensors の種別マーカーの不一致で拒否される。
        (&adamw, "adamw", "adam", Want::Autodiff),
        (&adam, "adam", "adamw", Want::Autodiff),
        // config のキー集合が異なる組は config の厳密なキー検査で拒否される。
        (&sgd, "sgd", "adam", Want::Manifest),
        (&adam, "adam", "sgd", Want::Manifest),
        (&rms, "rmsprop", "adagrad", Want::Manifest),
        (&sgd, "sgd", "lamb", Want::Manifest),
        // 非 lbfgs → lbfgs は history_len の欠落。
        (&sgd, "sgd", "lbfgs", Want::Manifest),
        (&adam, "adam", "lbfgs", Want::Manifest),
    ];
    for (mk, from, to, want) in cases {
        let fx = mk();
        let t = replace_once(
            &fx.text(),
            &format!("\"kind\":\"{from}\""),
            &format!("\"kind\":\"{to}\""),
        );
        fx.reject_text(&format!("{from} -> {to}"), &t, want);
    }
}

#[test]
fn history_len_type_and_bound_violations_are_rejected() {
    let fx = lbfgs();
    let t = fx.text();
    let len: i64 = raw_field(&t, "history_len").parse().expect("整数");
    assert_eq!(len, 3, "履歴は到達済みのはず");
    for v in ["null", "-1", "1.0", "\"3\""] {
        fx.reject_text(
            &format!("history_len={v}"),
            &set_field(&t, "history_len", v),
            Want::Manifest,
        );
    }
    fx.reject_text(
        "history_len=65537",
        &set_field(&t, "history_len", "65537"),
        Want::TooLarge("Lbfgs 履歴件数", 65536),
    );
    for v in [len - 1, len + 1] {
        fx.reject_text(
            &format!("history_len={v}"),
            &set_field(&t, "history_len", &v.to_string()),
            Want::Mismatch,
        );
    }
}

#[test]
fn growth_tracker_out_of_range_is_rejected() {
    let fx = amp();
    let t = fx.text();
    let interval: u64 = raw_field(&t, "growth_interval").parse().expect("整数");
    for v in [interval.to_string(), u64::MAX.to_string()] {
        fx.reject_text(
            &format!("growth_tracker={v}"),
            &set_field(&t, "growth_tracker", &v),
            Want::Autodiff,
        );
    }
    for v in ["-1", "1.0", "null", "18446744073709551616"] {
        fx.reject_text(
            &format!("growth_tracker={v}"),
            &set_field(&t, "growth_tracker", v),
            Want::Manifest,
        );
    }
    // 陽性境界: growth_interval - 1 は受理される。
    let ok = set_field(&t, "growth_tracker", &(interval - 1).to_string());
    fx.write_manifest(ok.as_bytes());
    assert!(
        load_model(&fx.dir).is_ok(),
        "growth_interval - 1 は受理される"
    );
    fx.restore_and_assert_loads("growth_tracker boundary");
}

// 未使用 variant の警告を避けつつ、未知の kind が UnsupportedModel であることを固定する。
#[test]
fn unknown_layer_kind_is_unsupported() {
    let fx = plain();
    let t = replace_once(&fx.text(), "\"kind\":\"relu\"", "\"kind\":\"gelu9\"");
    fx.reject_text("未知の層 kind", &t, Want::Unsupported);
}
