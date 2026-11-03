//! Phase 6 で足した 12 層の保存・読込・ONNX export を 1 か所で横断確認する統合テスト
//! （イシュー #3828・親 #3813。横断方針 #3750 の R1・R3・R4・R9・R10。設計記録
//! `docs/compat-new-layers-io-export-decision.md` §15）。層ごとの詳細（改竄・往復 bit 一致）は
//! 各層のテスト（`compat_sequential_model_io_layers.rs` ほか）が持つ。本ファイルは次を固定する。
//!
//! - 12 層それぞれの `save_model` が承認済みの kind を書き、`format_version` は 2 のまま。
//! - 保存した manifest を `load_model` で読み戻せ、保存し直した manifest が同一（kind・params 不変）。
//! - manifest の `buffer_keys` は BN3d だけが 2 件（`running_mean`／`running_var`）、ほかは 0 件。
//! - ONNX export は全層 `OnnxError::UnsupportedLayer`（0.14.0 では export しない。#3768 論点 4）。
//!
//! 実機（CUDA／Metal）は不要。

#![cfg(unix)]

use std::path::Path;

use fandhe_ai::InterpolateMode;
use fandhe_ai::compat::{Sequential, load_model, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};

mod common;
use common::temp_dir::TempDirGuard;

/// (承認済み kind, 期待 buffer キー数, その kind 1 層だけのモデル)。
fn cases() -> Vec<(&'static str, usize, Sequential)> {
    vec![
        (
            "upsample_scale_factor",
            0,
            Sequential::new()
                .add_upsample_scale_factor(vec![1.5, 2.0], InterpolateMode::Nearest)
                .unwrap(),
        ),
        (
            "hardshrink",
            0,
            Sequential::new().add_hardshrink(0.5).unwrap(),
        ),
        (
            "softshrink",
            0,
            Sequential::new().add_softshrink(0.5).unwrap(),
        ),
        (
            "batch_norm3d",
            2,
            Sequential::new().add_batch_norm3d(2, 1e-5, 0.1).unwrap(),
        ),
        (
            "channel_shuffle",
            0,
            Sequential::new().add_channel_shuffle(2).unwrap(),
        ),
        (
            "adaptive_avg_pool3d",
            0,
            Sequential::new()
                .add_adaptive_avg_pool3d([1, 2, 2])
                .unwrap(),
        ),
        (
            "adaptive_max_pool3d",
            0,
            Sequential::new()
                .add_adaptive_max_pool3d([1, 2, 2])
                .unwrap(),
        ),
        (
            "lp_pool1d",
            0,
            Sequential::new().add_lp_pool1d(2.0, 2, None).unwrap(),
        ),
        (
            "lp_pool2d",
            0,
            Sequential::new()
                .add_lp_pool2d(2.0, [2, 2], Some([1, 1]))
                .unwrap(),
        ),
        (
            "lp_pool3d",
            0,
            Sequential::new()
                .add_lp_pool3d(1.0, [2, 2, 2], None)
                .unwrap(),
        ),
        (
            "fractional_max_pool2d",
            0,
            Sequential::new()
                .add_fractional_max_pool2d([2, 2], [1, 1])
                .unwrap(),
        ),
        (
            "fractional_max_pool3d",
            0,
            Sequential::new()
                .add_fractional_max_pool3d([2, 2, 2], [1, 1, 1])
                .unwrap(),
        ),
    ]
}

fn manifest_text(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("manifest.json")).expect("manifest が読めるはず")
}

/// 世代名つきの `safetensors_file` を正規化して manifest を比べられる形にする。
fn normalized(manifest: &str) -> String {
    let key = "\"safetensors_file\":\"";
    let start = manifest.find(key).expect("キーがあるはず") + key.len();
    let end = start + manifest[start..].find('"').expect("閉じ引用符");
    format!("{}<generation>{}", &manifest[..start], &manifest[end..])
}

#[test]
fn twelve_new_layers_save_load_and_reject_onnx_export() {
    let cases = cases();
    assert_eq!(cases.len(), 12, "Phase 6 の新しい kind は 12 種（R9）");
    for (kind, buffers, model) in cases {
        let guard = TempDirGuard::new(kind);
        save_model(&model, guard.path()).unwrap_or_else(|e| panic!("{kind}: 保存できない: {e:?}"));
        let text = manifest_text(guard.path());
        assert!(
            text.contains("\"format_version\":2"),
            "{kind}: format_version"
        );
        assert!(
            text.contains(&format!("\"kind\":\"{kind}\"")),
            "{kind}: manifest に kind が無い"
        );
        let json: serde_json::Value = serde_json::from_str(&text).expect("JSON のはず");
        assert_eq!(
            json["buffer_keys"].as_array().expect("配列").len(),
            buffers,
            "{kind}: buffer_keys"
        );

        let loaded = load_model(guard.path()).unwrap_or_else(|e| panic!("{kind}: 読めない: {e:?}"));
        let again = TempDirGuard::new(&format!("{kind}-again"));
        save_model(&loaded, again.path()).expect("保存し直せるはず");
        assert_eq!(
            normalized(&text),
            normalized(&manifest_text(again.path())),
            "{kind}: 保存し直した manifest が異なる"
        );

        let err = OnnxModel::from_sequential(&model).unwrap_err();
        assert!(
            matches!(err, OnnxError::UnsupportedLayer { .. }),
            "{kind}: {err:?}"
        );
    }
}

/// v1 の manifest で拒否するのは buffer を持つ kind（`batch_norm3d`）だけ（R5）。
/// v1 に書き換えた BN3d は `Mismatch` 系で拒否され、buffer の無い `channel_shuffle` は受理される。
#[test]
fn only_buffered_new_kind_is_rejected_under_format_version_1() {
    for (kind, buffers, model) in cases() {
        if kind != "batch_norm3d" && kind != "channel_shuffle" {
            continue;
        }
        let guard = TempDirGuard::new(&format!("v1-{kind}"));
        save_model(&model, guard.path()).expect("保存できるはず");
        let text =
            manifest_text(guard.path()).replace("\"format_version\":2", "\"format_version\":1");
        std::fs::write(guard.path().join("manifest.json"), text).expect("書けるはず");
        let result = load_model(guard.path());
        assert_eq!(result.is_err(), buffers > 0, "{kind}: v1 の受理／拒否");
    }
}
