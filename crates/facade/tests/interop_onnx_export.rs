//! `fandhe_ai::interop::onnx::{OnnxModel::to_bytes, OnnxModel::to_path}`
//! の facade 単独（`fandhe_ai` と `std` のみ import）統合テスト（イシュー
//! #2018・AC1／AC3）。
//!
//! **本ファイルは意図的に `fandhe_ai_onnx_interop` を import しない**
//! （facade のみで export→import roundtrip が成立することを検証する
//! ため。内部クレート直接呼び出しとの一致確認は
//! `tests/interop_onnx_internal_parity.rs` の群 B が担う）。
//!
//! 比較はすべて bit 同一（f32 は `to_bits()`）。REQ-2／REQ-7 の tolerance
//! は使わない・導入もしない。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxError, OnnxExportOptions, OnnxModel, OnnxValue};

fn onnx_interop_fixture(rel: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures"
    ))
    .join(rel)
}

fn run_scalar_input(model: &OnnxModel, input: [f32; 2]) -> Vec<f32> {
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
    );
    let outputs = model.run(feeds).expect("run 成功");
    match &outputs["output"] {
        OnnxValue::F32(t) => t.as_slice().unwrap().to_vec(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    }
}

const SAMPLES: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 1.0], [0.3, 0.7], [2.0, -1.0]];

/// 1. `model.onnx`: `from_path` → `to_bytes(default)` → `from_bytes` →
/// 4 サンプル入力で `run`、元モデルの `run` 出力と bit 同一（AC1）。
#[test]
fn model_onnx_export_import_roundtrip_matches_original_bit_exact() {
    let original =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");
    let bytes = original
        .to_bytes(&OnnxExportOptions::default())
        .expect("to_bytes 成功");
    let reimported = OnnxModel::from_bytes(&bytes).expect("from_bytes(再 import) 成功");

    for input in SAMPLES {
        let original_out = run_scalar_input(&original, input);
        let reimported_out = run_scalar_input(&reimported, input);
        assert_eq!(
            original_out.len(),
            reimported_out.len(),
            "出力要素数が一致しない: input={input:?}"
        );
        assert!(!original_out.is_empty(), "空虚 pass 防止: 出力要素数が 0");
        for (a, b) in original_out.iter().zip(reimported_out.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "export→import roundtrip 後の出力が bit 不一致: input={input:?} \
                 original={a} reimported={b}"
            );
        }
    }
}

/// 2. `slice_repro.onnx`: 同上（narrow／slice 系 op を含む形状で確認）。
#[test]
fn slice_repro_onnx_export_import_roundtrip_matches_original_bit_exact() {
    let original =
        OnnxModel::from_path(onnx_interop_fixture("slice_repro.onnx")).expect("from_path 成功");
    let bytes = original
        .to_bytes(&OnnxExportOptions::default())
        .expect("to_bytes 成功");
    let reimported = OnnxModel::from_bytes(&bytes).expect("from_bytes(再 import) 成功");

    // slice_repro.onnx は入力名 "x"・[5, 6] 形状を要求する
    // （`interop_onnx_internal_parity.rs` の同名テストと同じ規約）。
    let input_data: Vec<f32> = (0..30).map(|i| i as f32 * 0.1 - 1.5).collect();
    let input = Tensor::<f32>::new(input_data, &[5, 6]).unwrap();

    let mut original_feeds = HashMap::new();
    original_feeds.insert("x".to_string(), OnnxValue::F32(input.clone()));
    let original_outputs = original.run(original_feeds).expect("original run 成功");

    let mut reimported_feeds = HashMap::new();
    reimported_feeds.insert("x".to_string(), OnnxValue::F32(input));
    let reimported_outputs = reimported
        .run(reimported_feeds)
        .expect("reimported run 成功");

    assert_eq!(
        original_outputs.len(),
        reimported_outputs.len(),
        "出力の個数が一致しない"
    );
    for (name, original_value) in &original_outputs {
        let reimported_value = reimported_outputs
            .get(name)
            .unwrap_or_else(|| panic!("出力 '{name}' が reimported 側に存在しない"));
        let (OnnxValue::F32(a), OnnxValue::F32(b)) = (original_value, reimported_value) else {
            panic!("出力 '{name}' は OnnxValue::F32 を期待");
        };
        assert_eq!(a.shape(), b.shape(), "出力 '{name}' の shape が不一致");
        assert!(
            !a.as_slice().unwrap_or(&[]).is_empty(),
            "空虚 pass 防止: 出力 '{name}' の要素数が 0"
        );
        for (x, y) in a
            .as_slice()
            .unwrap()
            .iter()
            .zip(b.as_slice().unwrap().iter())
        {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "出力 '{name}' が export→import roundtrip 後に bit 不一致: \
                 original={x} reimported={y}"
            );
        }
    }
}

/// 3. 不動点・決定性: `b1 = m.to_bytes()`、`b2 = from_bytes(b1).to_bytes()`
/// で `b1 == b2`。同一モデルへ `to_bytes` を 2 回呼んでも同一
/// （`HashMap` 走査順序に依存しない決定的出力）。
#[test]
fn to_bytes_is_deterministic_and_a_fixed_point() {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");
    let options = OnnxExportOptions::default();

    let b1 = model.to_bytes(&options).expect("1 回目の to_bytes 成功");
    let b1_again = model.to_bytes(&options).expect("2 回目の to_bytes 成功");
    assert_eq!(
        b1, b1_again,
        "同一モデルへの to_bytes 呼び出しが決定的でない（HashMap 走査順序への依存疑い）"
    );

    let reimported = OnnxModel::from_bytes(&b1).expect("from_bytes 成功");
    let b2 = reimported
        .to_bytes(&options)
        .expect("再 import 後の to_bytes 成功");
    assert_eq!(
        b1, b2,
        "to_bytes → from_bytes → to_bytes が不動点になっていない（b1 != b2）"
    );
}

/// 4. `to_path` → `from_path` roundtrip: 一意なファイル名で書き出し、
/// ファイル内容が `to_bytes` と一致・再 import の `run` が bit 同一。
/// 絶対パスは assert メッセージへ出さない（`.claude/rules/security.md`）。
#[test]
fn to_path_writes_bytes_matching_to_bytes_and_roundtrips() {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");
    let options = OnnxExportOptions::default();
    let expected_bytes = model.to_bytes(&options).expect("to_bytes 成功");

    let path = std::env::temp_dir().join(format!(
        "fandhe-ai-onnx-export-test-{}-{}.onnx",
        std::process::id(),
        "to_path_writes_bytes_matching_to_bytes_and_roundtrips"
    ));
    // panic 経路でも一時ファイルが残らないよう RAII で確実に削除する。
    struct CleanupGuard(PathBuf);
    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _guard = CleanupGuard(path.clone());

    model.to_path(&path, &options).expect("to_path 成功");
    let written = std::fs::read(&path).expect("一時ファイル読み込み成功");
    assert_eq!(
        written, expected_bytes,
        "to_path が書き出したバイト列が to_bytes と一致しない"
    );

    let reimported = OnnxModel::from_path(&path).expect("from_path(再 import) 成功");
    for input in SAMPLES {
        let original_out = run_scalar_input(&model, input);
        let reimported_out = run_scalar_input(&reimported, input);
        for (a, b) in original_out.iter().zip(reimported_out.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "to_path→from_path roundtrip 後の出力が bit 不一致: input={input:?}"
            );
        }
    }
}

/// 5. `OnnxExportOptions::default()` が 8／17。非既定値（例 9／18）でも
/// roundtrip `run` は bit 同一、かつ既定値のバイト列とは異なる。
#[test]
fn non_default_export_options_still_roundtrip_but_produce_different_bytes() {
    let default_options = OnnxExportOptions::default();
    assert_eq!(default_options.ir_version, 8);
    assert_eq!(default_options.opset_version, 17);

    let mut non_default_options = OnnxExportOptions::default();
    non_default_options.ir_version = 9;
    non_default_options.opset_version = 18;

    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");
    let default_bytes = model
        .to_bytes(&default_options)
        .expect("既定値での to_bytes 成功");
    let non_default_bytes = model
        .to_bytes(&non_default_options)
        .expect("非既定値での to_bytes 成功");
    assert_ne!(
        default_bytes, non_default_bytes,
        "ir_version／opset_version を変えてもバイト列が変化していない"
    );

    let reimported =
        OnnxModel::from_bytes(&non_default_bytes).expect("非既定値バイト列からの from_bytes 成功");
    for input in SAMPLES {
        let original_out = run_scalar_input(&model, input);
        let reimported_out = run_scalar_input(&reimported, input);
        for (a, b) in original_out.iter().zip(reimported_out.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "非既定 options でも export→import roundtrip 後の出力が bit 不一致: \
                 input={input:?}"
            );
        }
    }
}

/// 6. `to_path` の存在しない親ディレクトリ → `OnnxError::Io`。
#[test]
fn to_path_with_nonexistent_parent_directory_returns_io_error() {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path 成功");
    let path = std::env::temp_dir().join(format!(
        "fandhe-ai-onnx-export-test-nonexistent-parent-{}",
        std::process::id()
    ));
    let nested_path = path.join("nested").join("model.onnx");
    let err = model
        .to_path(&nested_path, &OnnxExportOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, OnnxError::Io(_)),
        "OnnxError::Io を期待したが {err:?}"
    );
    // 親ディレクトリ自体が存在しないため、失敗時にファイルが作成されて
    // いないことも確認する。
    assert!(
        !nested_path.exists(),
        "失敗した to_path がファイルを作成してしまっている"
    );
}

/// 7. allowlist 外 op の fail-closed（facade 単独版）: std のみで最小
/// protobuf（allowlist 外の合成 op_type を持つグラフ）を手組みし、
/// `from_bytes` は成功・`to_bytes`／`to_path` は
/// `OnnxError::UnsupportedOp` で拒否されることを確認する。
///
/// protobuf の length-delimited wire format（field number << 3 | wire
/// type）に従い、`ModelProto { graph: GraphProto { node, input, output } }`
/// を最小構成で手組みする（field 番号は `onnx-interop::onnx::proto` の
/// スキーマと一致させる必要があるため、対応表を明記する）:
/// `ModelProto.graph` = field 7 (LEN)、`GraphProto.node` = field 1
/// (LEN)、`GraphProto.input`/`output` = field 11/12 (LEN)、
/// `NodeProto.input` = field 1 (LEN)、`NodeProto.output` = field 2
/// (LEN)、`NodeProto.op_type` = field 4 (LEN)、`ValueInfoProto.name` =
/// field 1 (LEN)。
#[test]
fn synthetic_model_with_disallowed_op_type_is_rejected_at_export_fail_closed() {
    fn tag(field: u32, wire_type: u32) -> u8 {
        ((field << 3) | wire_type) as u8
    }

    fn len_delimited(field: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![tag(field, 2)];
        // 本テストのペイロードはすべて 127 バイト未満のため、varint の
        // 複数バイト表現（continuation bit）は不要（自己完結した最小
        // フィクスチャという設計判断）。
        assert!(
            payload.len() < 128,
            "test fixture: varint 単一バイト表現の前提が崩れている"
        );
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        out
    }

    fn value_info(name: &str) -> Vec<u8> {
        len_delimited(1, name.as_bytes())
    }

    let node = {
        let mut buf = Vec::new();
        buf.extend(len_delimited(1, b"x")); // NodeProto.input
        buf.extend(len_delimited(2, b"y")); // NodeProto.output
        buf.extend(len_delimited(4, b"NotARealOp")); // NodeProto.op_type
        buf
    };

    let graph = {
        let mut buf = Vec::new();
        buf.extend(len_delimited(1, &node)); // GraphProto.node
        buf.extend(len_delimited(11, &value_info("x"))); // GraphProto.input
        buf.extend(len_delimited(12, &value_info("y"))); // GraphProto.output
        buf
    };

    let model_bytes = len_delimited(7, &graph); // ModelProto.graph

    let model = OnnxModel::from_bytes(&model_bytes)
        .expect("from_bytes は成功するはず（allowlist 検査は export 時のみ）");

    let err = model.to_bytes(&OnnxExportOptions::default()).unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnsupportedOp { op_type } if op_type == "NotARealOp"),
        "OnnxError::UnsupportedOp を期待したが {err:?}"
    );

    let path = std::env::temp_dir().join(format!(
        "fandhe-ai-onnx-export-test-disallowed-op-{}.onnx",
        std::process::id()
    ));
    let err = model
        .to_path(&path, &OnnxExportOptions::default())
        .unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnsupportedOp { op_type } if op_type == "NotARealOp"),
        "to_path も同じ OnnxError::UnsupportedOp を返すはずだが {err:?}"
    );
    assert!(
        !path.exists(),
        "拒否された to_path がファイルを作成してしまっている"
    );
}

/// 8. `OnnxError` のワイルドカード付き `match` が facade 単独でも
/// コンパイルできることを固定する（`#[non_exhaustive]` の扱い）。
#[test]
fn onnx_error_wildcard_match_compiles_facade_only() {
    let err = OnnxModel::from_bytes(&[0x08u8, 0xffu8]).unwrap_err();
    let _label = match &err {
        OnnxError::Io(_) => "io",
        OnnxError::Decode { .. } => "decode",
        OnnxError::UnsupportedOp { .. } => "unsupported_op",
        _ => "other",
    };
}
