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
//!
//! **プラットフォーム前提（Cursor Bugbot 指摘・PRRT_kwDOTuUCJc6mrW…
//! 対応。2026-09-28 イシュー #2349 で訂正）**: `onnx-interop` の
//! external data 実解決（封じ込めオープン）は unix・Windows の両方で
//! 成功する（`fandhe_ai_onnx_interop::onnx::external_data` モジュール
//! doc 参照）。しかし **facade（本クレート `fandhe-ai`）自体は Windows
//! ではビルドできない**: facade は `backend-cuda` へ無条件依存し、
//! `crates/backend-cuda/src/nvrtc.rs` は非 unix ターゲットで
//! `compile_error!` を発する（#509／PR #677。NVRTC キャッシュの fd pin
//! による TOCTOU 対策が `openat`・`/proc/self/fd` 等 unix 系 API に
//! 依存するため非 unix 向けフォールバックを提供しない設計）。したがって
//! `OnnxModel::from_path` の Windows 対応は本 crate（`onnx-interop`）側の
//! 対応だけでは完結せず、facade 経由の到達性は本ファイルでは検証できない
//! （backend-cuda の Windows 対応は別イシューでの起票候補。イシュー
//! #2349 PR 参照）。解決成功を前提とする
//! [`from_path_resolves_external_data_and_matches_manifest_reference`]
//! は `cfg(unix)` 限定のままとし、`cfg(not(any(unix, windows)))` 版
//! （`InvalidModel` で拒否されることの契約）は facade がビルドできる
//! unix・Windows 以外の環境（将来 backend-cuda が対応した場合）向けに
//! 残す（`not(unix)` のままだと、facade が将来 Windows でビルドできる
//! ようになった時点で「Windows では拒否される」という誤った契約を
//! 固定してしまう。onnx-interop 自体は既に Windows へ対応済みのため）。
//! `from_bytes` は external data 経由をそもそも通らないため
//! [`from_bytes_still_rejects_external_data_fixture`] は OS 非依存のまま
//! 全プラットフォームで実行する。

use std::path::PathBuf;

use fandhe_ai::interop::onnx::OnnxModel;
// external data の実解決を伴うテスト（cfg(unix) 限定。ファイル冒頭
// コメント参照）専用の import。非 unix ビルドでは未使用になるため
// 揃えて cfg する。
#[cfg(unix)]
use fandhe_ai::Tensor;
#[cfg(unix)]
use fandhe_ai::interop::onnx::OnnxValue;
#[cfg(unix)]
use std::collections::HashMap;

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
/// `from_path_resolves_external_data_and_matches_manifest_reference`
/// 専用のため `cfg(unix)` 限定（ファイル冒頭コメント参照）。
#[cfg(unix)]
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
///
/// **`cfg(unix)` 限定**（ファイル冒頭コメント参照）。
#[cfg(unix)]
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

/// [`from_path_resolves_external_data_and_matches_manifest_reference`]
/// の非 unix 契約版（Cursor Bugbot 指摘・PRRT_kwDOTuUCJc6mrW… 対応。
/// 2026-09-28 イシュー #2349 で訂正）。facade は Windows ではビルド
/// できないため（ファイル冒頭コメント参照）、本テストが実際に走るのは
/// facade がビルドできる非 unix 環境（現状存在しない。将来 backend-cuda
/// が対応した場合の回帰防止として残す）に限られる。その環境で
/// external data の実解決手段を持たない場合、companion `.onnx.data` が
/// 実在する fixture であっても `from_path` は常に
/// `OnnxError::InvalidModel`（`ExternalDataError::
/// UnsupportedPlatformForSecureResolve` 由来。`map_graph_error` の
/// catch-all 分岐）で拒否されることを固定する。`onnx-interop`
/// （external_data 自体）は Windows でも解決に対応済みのため、本テストは
/// `cfg(not(any(unix, windows)))` に限る（facade が将来 Windows で
/// ビルドできるようになった場合に、この「非対応」契約テストが誤って
/// 成功を期待しない側で固定されるのを防ぐ）。
#[cfg(not(any(unix, windows)))]
#[test]
fn from_path_rejects_external_data_fixture_as_unsupported_platform_when_not_unix() {
    let root = external_data_fixture_root();
    for &case_name in EXTERNAL_DATA_CASE_NAMES {
        let model_path = root.join(format!("{case_name}_dynamo.onnx"));
        let err = OnnxModel::from_path(&model_path)
            .expect_err(&format!("{case_name}: 非 unix では拒否されるはず"));
        assert!(
            matches!(
                &err,
                fandhe_ai::interop::onnx::OnnxError::InvalidModel { .. }
            ),
            "{case_name}: 非 unix では OnnxError::InvalidModel を期待したが: {err:?}"
        );
    }
}
