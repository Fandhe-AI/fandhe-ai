//! `fandhe_ai::interop::onnx::OnnxModel::from_path` の ONNX external data
//! （外部 `.onnx.data` ファイル）対応の facade 単独到達性テスト
//! （イシュー #2347・2026-09-28 ユーザー承認）。
//!
//! `OnnxModel::from_bytes` は external data を非対応のまま fail-closed に
//! 拒否する契約（`fandhe_ai_onnx_interop::onnx::external_data` モジュール
//! 冒頭コメント「不変条件」節）が facade 経由でも保たれることを固定する。
//! `OnnxModel::from_path` は PyTorch dynamo exporter の生出力
//! （`.onnx` + companion `.onnx.data`。再 inline 化していない）を読み込み・
//! 実行できることを、`crates/onnx-interop/tests/fixtures/
//! pytorch-onnx-external-data/`（イシュー #2347 で追加した fixture。
//! `crates/onnx-interop/tests/onnx_interp_pytorch_cnn_fixture.rs` の
//! `external_data_fixture_*` 系と同一 fixture・同一判定式）に対して確認
//! する。
//!
//! fixture は本クレート外を `CARGO_MANIFEST_DIR` 相対で直接参照する
//! （`tests/interop_onnx_import.rs` と同じ方針）。判定式は REQ-7 事前固定
//! 基準 `abs_err / (|ref| + 1e-6) <= 1e-3`（REQ-2 バックエンド間数値一致
//! OR 複合判定とは別指標）。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxModel, OnnxValue};

const EXTERNAL_DATA_CASE_NAMES: &[&str] =
    &["conv2d_basic", "conv2d_nobias", "conv2d_stride_dil_group"];

fn external_data_fixture_root() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures/pytorch-onnx-external-data"
    ))
}

/// `manifest.json` の `bits`（f32 の bit パターン列。u32 として記録）を
/// `Tensor<f32>` へ復元する（`onnx_interp_pytorch_cnn_fixture.rs::
/// tensor_from_record` と同型。要素数一致を先に検証してから
/// `f32::from_bits` で復元する。A03）。
fn tensor_from_manifest(record: &serde_json::Value) -> Tensor<f32> {
    let shape: Vec<usize> = record["shape"]
        .as_array()
        .expect("shape が配列でない")
        .iter()
        .map(|v| {
            let d = v.as_i64().expect("shape 要素が整数でない");
            assert!(d >= 0, "負の shape 次元: {d}");
            d as usize
        })
        .collect();
    let bits: Vec<u32> = record["bits"]
        .as_array()
        .expect("bits が配列でない")
        .iter()
        .map(|v| {
            let n = v.as_u64().expect("bits 要素が非負整数でない");
            u32::try_from(n).expect("bits 要素が u32 範囲外")
        })
        .collect();
    let expected_len: usize = shape.iter().product();
    assert_eq!(
        bits.len(),
        expected_len,
        "bits 長 {} と shape 由来の要素数 {} が不一致",
        bits.len(),
        expected_len
    );
    let values: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();
    Tensor::<f32>::new(values, &shape).expect("Tensor::new 失敗")
}

/// `OnnxModel::from_path` が external data fixture を読み込み・実行でき、
/// 出力が `manifest.json` の自己完結参照出力（このスクリプト自身が生成
/// した重みに対する PyTorch 参照値。`README.md`「実測結果」節参照）と
/// REQ-7 事前固定基準で一致することを確認する。
#[test]
fn from_path_resolves_external_data_and_matches_manifest_reference() {
    let root = external_data_fixture_root();
    let manifest_bytes = std::fs::read(root.join("manifest.json")).expect("manifest 読み込み失敗");
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("manifest.json の parse に失敗した");

    for &case_name in EXTERNAL_DATA_CASE_NAMES {
        let entry = &manifest[case_name];
        assert!(!entry.is_null(), "manifest.json に '{case_name}' が無い");

        let model_path = root.join(format!("{case_name}_dynamo.onnx"));
        let model = OnnxModel::from_path(&model_path)
            .unwrap_or_else(|e| panic!("{case_name}: from_path は成功するはず: {e}"));

        let input = tensor_from_manifest(&entry["input"]);
        let expected = tensor_from_manifest(&entry["output"]);

        let mut feeds = HashMap::new();
        feeds.insert("x".to_string(), OnnxValue::F32(input));
        let result = model
            .run(feeds)
            .unwrap_or_else(|e| panic!("{case_name}: run は成功するはず: {e}"));
        let actual = match &result["y"] {
            OnnxValue::F32(t) => t,
            other => panic!("{case_name}: OnnxValue::F32 を期待したが {other:?}"),
        };
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "{case_name}: 出力 shape 不一致"
        );

        let actual_slice = actual.as_slice().expect("as_slice 失敗");
        let expected_slice = expected.as_slice().expect("as_slice 失敗");
        let mut max_rel_err = 0.0f32;
        for (&a, &e) in actual_slice.iter().zip(expected_slice.iter()) {
            let abs_err = (a - e).abs();
            let rel_err = abs_err / (e.abs() + 1e-6);
            max_rel_err = max_rel_err.max(rel_err);
            assert!(
                rel_err <= 1e-3,
                "{case_name}: REQ-7 数値一致基準を超過: expected={e} actual={a} \
                 abs_err={abs_err} rel_err={rel_err}"
            );
        }
        eprintln!("{case_name}: max_rel_err={max_rel_err} (threshold=1e-3)");
    }
}

/// A6 回帰（facade 版）: `OnnxModel::from_bytes` は external data を
/// 非対応のまま fail-closed に拒否する（`from_path` が external data 対応
/// になっても、バイト列入口の挙動は不変であることの固定。
/// `crates/onnx-interop/tests/onnx_interp_pytorch_cnn_fixture.rs::
/// external_data_fixture_bytes_entry_point_still_rejects` の facade 版）。
#[test]
fn from_bytes_still_rejects_external_data_fixture() {
    let root = external_data_fixture_root();
    for &case_name in EXTERNAL_DATA_CASE_NAMES {
        let model_path = root.join(format!("{case_name}_dynamo.onnx"));
        let bytes = std::fs::read(&model_path).expect("fixture 読み込み失敗");
        let err = OnnxModel::from_bytes(&bytes)
            .expect_err(&format!("{case_name}: from_bytes は拒否されるはず"));
        assert!(
            format!("{err}").contains("raw_data バイト長不整合")
                || format!("{err:?}").contains("RawDataByteLenMismatch"),
            "{case_name}: RawDataByteLenMismatch 系のエラーを期待したが: {err:?}"
        );
    }
}
