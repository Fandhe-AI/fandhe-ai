//! `onnx::proto` / `onnx::graph` のデコード・検証テスト（TASK-7.2a・イシュー #77）。
//!
//! コミット済みの極小 fixture（`model.onnx`・`slice_repro.onnx`）を使うテストは
//! CI（self-hosted）で常時実行する。`transformer.onnx`（12MB・非コミット）を使う
//! テストは実機依存テストと同じ運用で `#[ignore]` 分離し、環境変数でパスを指定
//! されたときのみ実行する（`tests/fixtures/README.md` の取得手順参照）。

use fandhe_ai_onnx_interop::onnx::graph::{GraphError, RawTensor, build_graph};
use fandhe_ai_onnx_interop::onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, SparseTensorProto, SparseTensorValueName,
    TensorProto, ValueInfoProto,
};
use prost::Message;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_model(name: &str) -> ModelProto {
    let bytes = std::fs::read(fixture_path(name))
        .unwrap_or_else(|e| panic!("fixture 読み込み失敗 {name}: {e}"));
    ModelProto::decode(bytes.as_slice()).unwrap_or_else(|e| panic!("decode 失敗 {name}: {e}"))
}

// --- model.onnx: MLP（Gemm/Relu x2 + Gemm/Sigmoid） ---

#[test]
fn model_onnx_decodes_expected_graph_structure() {
    let model = load_model("model.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    assert_eq!(graph.inputs, vec!["input".to_string()]);
    assert_eq!(graph.outputs, vec!["output".to_string()]);

    let op_types: Vec<&str> = graph.nodes.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(
        op_types,
        vec!["Gemm", "Relu", "Gemm", "Relu", "Gemm", "Sigmoid"]
    );

    let node_names: Vec<&str> = graph.nodes.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(
        node_names,
        vec![
            "/fc1/Gemm",
            "/relu/Relu",
            "/fc2/Gemm",
            "/relu_1/Relu",
            "/fc3/Gemm",
            "/sigmoid/Sigmoid"
        ]
    );

    assert_eq!(graph.initializers.len(), 6);
    let expect_f32_shape = |name: &str, shape: &[i64]| match graph.initializers.get(name) {
        Some(RawTensor::F32 { shape: s, .. }) => assert_eq!(s.as_slice(), shape, "{name} の shape"),
        other => panic!("{name} は F32 RawTensor のはずが {other:?}"),
    };
    expect_f32_shape("fc1.weight", &[8, 2]);
    expect_f32_shape("fc1.bias", &[8]);
    expect_f32_shape("fc2.weight", &[8, 8]);
    expect_f32_shape("fc2.bias", &[8]);
    expect_f32_shape("fc3.weight", &[1, 8]);
    expect_f32_shape("fc3.bias", &[1]);

    // AttributeProto（proto.rs のフィールド番号 name=1/f=2/i=3/type=20）を実
    // fixture の Gemm ノードで検証する。既存の
    // `attribute_proto_round_trips_with_tensor_field` は同一構造体での
    // 自己ラウンドトリップのみでタグ番号の誤りを検出できないため、実
    // fixture decode 経由での属性値アサーションを固定する（レビュー指摘 #77）。
    const ATTRIBUTE_TYPE_FLOAT: i32 = 1;
    const ATTRIBUTE_TYPE_INT: i32 = 2;
    let gemm = graph
        .nodes
        .iter()
        .find(|n| n.name == "/fc1/Gemm")
        .expect("/fc1/Gemm ノードが存在するはず");
    let attr = |name: &str| {
        gemm.attribute
            .iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("Gemm 属性 '{name}' が見つからないはず"))
    };
    let alpha = attr("alpha");
    assert_eq!(alpha.r#type, ATTRIBUTE_TYPE_FLOAT);
    assert_eq!(alpha.f, 1.0);
    let beta = attr("beta");
    assert_eq!(beta.r#type, ATTRIBUTE_TYPE_FLOAT);
    assert_eq!(beta.f, 1.0);
    let trans_b = attr("transB");
    assert_eq!(trans_b.r#type, ATTRIBUTE_TYPE_INT);
    assert_eq!(trans_b.i, 1);
}

// --- slice_repro.onnx: 動的境界 Slice パターン（Shape->Gather->Unsqueeze->Concat->Slice） ---

#[test]
fn slice_repro_onnx_decodes_expected_graph_structure() {
    let model = load_model("slice_repro.onnx");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    assert_eq!(graph.inputs, vec!["x".to_string()]);
    assert_eq!(graph.outputs, vec!["output".to_string()]);

    let op_types: Vec<&str> = graph.nodes.iter().map(|n| n.op_type.as_str()).collect();
    assert_eq!(
        op_types,
        vec!["Shape", "Gather", "Unsqueeze", "Concat", "Slice"]
    );

    assert_eq!(graph.initializers.len(), 4);
    let expect_i64 = |name: &str, shape: &[i64], data: &[i64]| match graph.initializers.get(name) {
        Some(RawTensor::I64 { shape: s, data: d }) => {
            assert_eq!(s.as_slice(), shape, "{name} の shape");
            assert_eq!(d.as_slice(), data, "{name} のデータ");
        }
        other => panic!("{name} は I64 RawTensor のはずが {other:?}"),
    };
    expect_i64("const_axes", &[2], &[0, 1]);
    expect_i64("const_4", &[1], &[4]);
    expect_i64("const_starts", &[2], &[0, 0]);
    expect_i64("const_gather_idx", &[1], &[0]);

    // AttributeProto（フィールド番号 i=3/type=20）を実 fixture の Gather
    // ノードで検証する（model.onnx の Gemm と合わせて実 fixture 経由の
    // 属性値アサーションを固定する。レビュー指摘 #77）。
    const ATTRIBUTE_TYPE_INT: i32 = 2;
    let gather = graph
        .nodes
        .iter()
        .find(|n| n.op_type == "Gather")
        .expect("Gather ノードが存在するはず");
    let axis = gather
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .expect("Gather 属性 'axis' が見つからないはず");
    assert_eq!(axis.r#type, ATTRIBUTE_TYPE_INT);
    assert_eq!(axis.i, 0);
}

// --- 不正入力の拒否テスト（長さ・形状検証の先行。OWASP A03・security.md） ---

#[test]
fn broken_protobuf_bytes_are_rejected_as_decode_error() {
    // protobuf のワイヤフォーマットとして不正なバイト列（varint の継続ビットが
    // 閉じない）。`prost::Message::decode` がエラーを返すことを確認し、パニック
    // しないことを保証する。
    let broken = [0xFFu8, 0xFF, 0xFF];
    let result = ModelProto::decode(broken.as_slice());
    assert!(
        result.is_err(),
        "壊れた protobuf バイト列は decode エラーになるべき"
    );
}

fn model_with_single_initializer(t: TensorProto) -> ModelProto {
    ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![],
            name: "g".to_string(),
            initializer: vec![t],
            input: vec![],
            output: vec![],
        }),
    }
}

#[test]
fn raw_data_len_mismatch_is_rejected() {
    // dims=[2] (F32) は 8 バイト期待だが raw_data は 4 バイトのみ与える。
    let t = TensorProto {
        dims: vec![2],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![],
        int64_data: vec![],
        name: "bad_tensor".to_string(),
        raw_data: vec![0, 0, 0, 0],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("raw_data 長不整合は拒否されるはず");
    match err {
        GraphError::RawDataByteLenMismatch {
            tensor_name,
            expected_bytes,
            actual_bytes,
        } => {
            assert_eq!(tensor_name, "bad_tensor");
            assert_eq!(expected_bytes, 8);
            assert_eq!(actual_bytes, 4);
        }
        other => panic!("RawDataByteLenMismatch を期待したが {other:?}"),
    }
}

// --- BOOL／FLOAT16 initializer の復号（イシュー #274） ---

#[test]
fn bool_tensor_raw_data_decodes_nonzero_as_true() {
    // ONNX/NumPy 慣習: raw_data は 1 バイト/要素、非ゼロ -> true（`decode_tensor`・
    // `ops::cast::cast_to_bool` と同じ解釈。`security.md` A03: 外部バイト列の
    // 変換規則をテストで固定化する）。
    let t = TensorProto {
        dims: vec![3],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::BOOL,
        float_data: vec![],
        int64_data: vec![],
        name: "bool_tensor".to_string(),
        raw_data: vec![0x00, 0x01, 0x02],
    };
    let model = model_with_single_initializer(t);
    let graph = build_graph(&model).expect("BOOL raw_data の復号は成功するはず");
    match graph.initializers.get("bool_tensor") {
        Some(RawTensor::Bool { data, shape }) => {
            assert_eq!(shape.as_slice(), &[3]);
            assert_eq!(data.as_slice(), &[false, true, true]);
        }
        other => panic!("RawTensor::Bool を期待したが {other:?}"),
    }
}

#[test]
fn bool_tensor_raw_data_byte_len_mismatch_is_rejected() {
    // dims=[4] は 4 バイト期待だが raw_data は 3 バイトのみ（要素数の積より前に
    // バイト長を検査する。`decode_tensor` の既存方針と同じ順序）。
    let t = TensorProto {
        dims: vec![4],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::BOOL,
        float_data: vec![],
        int64_data: vec![],
        name: "bad_bool_tensor".to_string(),
        raw_data: vec![0x00, 0x01, 0x00],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("BOOL raw_data 長不整合は拒否されるはず");
    match err {
        GraphError::RawDataByteLenMismatch {
            tensor_name,
            expected_bytes,
            actual_bytes,
        } => {
            assert_eq!(tensor_name, "bad_bool_tensor");
            assert_eq!(expected_bytes, 4);
            assert_eq!(actual_bytes, 3);
        }
        other => panic!("RawDataByteLenMismatch を期待したが {other:?}"),
    }
}

#[test]
fn float16_tensor_raw_data_decodes_little_endian_pairs() {
    // IEEE754 binary16 のリトルエンディアン 2 バイト/要素（onnx.proto3 `TensorProto`
    // 仕様）。既知の値（1.0・-2.5）の LE バイト列から正しく復号されることを固定化する。
    let one = half::f16::from_f32(1.0).to_le_bytes();
    let neg_two_point_five = half::f16::from_f32(-2.5).to_le_bytes();
    let mut raw_data = Vec::new();
    raw_data.extend_from_slice(&one);
    raw_data.extend_from_slice(&neg_two_point_five);

    let t = TensorProto {
        dims: vec![2],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT16,
        float_data: vec![],
        int64_data: vec![],
        name: "f16_tensor".to_string(),
        raw_data,
    };
    let model = model_with_single_initializer(t);
    let graph = build_graph(&model).expect("FLOAT16 raw_data の復号は成功するはず");
    match graph.initializers.get("f16_tensor") {
        Some(RawTensor::F16 { data, shape }) => {
            assert_eq!(shape.as_slice(), &[2]);
            assert_eq!(data[0].to_f32(), 1.0);
            assert_eq!(data[1].to_f32(), -2.5);
        }
        other => panic!("RawTensor::F16 を期待したが {other:?}"),
    }
}

#[test]
fn negative_dim_is_rejected() {
    let t = TensorProto {
        dims: vec![-1],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![],
        int64_data: vec![],
        name: "neg_dim_tensor".to_string(),
        raw_data: vec![],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("負の dim は拒否されるはず");
    match err {
        GraphError::NegativeDim { tensor_name, dim } => {
            assert_eq!(tensor_name, "neg_dim_tensor");
            assert_eq!(dim, -1);
        }
        other => panic!("NegativeDim を期待したが {other:?}"),
    }
}

#[test]
fn unknown_data_type_is_rejected() {
    let t = TensorProto {
        dims: vec![1],
        data_type: 99, // onnx.proto3 未定義域（本クレート未対応）
        float_data: vec![],
        int64_data: vec![],
        name: "unknown_dtype_tensor".to_string(),
        raw_data: vec![],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("未対応 data_type は拒否されるはず");
    match err {
        GraphError::UnknownDataType {
            tensor_name,
            data_type,
        } => {
            assert_eq!(tensor_name, "unknown_dtype_tensor");
            assert_eq!(data_type, 99);
        }
        other => panic!("UnknownDataType を期待したが {other:?}"),
    }
}

#[test]
fn element_count_overflow_is_rejected() {
    // usize（64bit）の範囲を明らかに超える dims の積（i64::MAX 同士の積は
    // 2^126 級で usize::MAX を大きく超える）。decode_tensor の要素データ復号
    // （バイト列走査）より前に checked_mul で拒否できることを確認する。
    let t = TensorProto {
        dims: vec![i64::MAX, i64::MAX],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![],
        int64_data: vec![],
        name: "overflow_tensor".to_string(),
        raw_data: vec![],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("要素数オーバーフローは拒否されるはず");
    match err {
        GraphError::ElementCountOverflow { tensor_name } => {
            assert_eq!(tensor_name, "overflow_tensor");
        }
        other => panic!("ElementCountOverflow を期待したが {other:?}"),
    }
}

#[test]
fn byte_length_multiply_overflow_is_rejected() {
    // dims の積（要素数）自体は usize に収まるが、要素サイズ（F32=4byte）を
    // 掛けた際にオーバーフローする境界値（2^62 は usize::MAX / 4 を超える）。
    // element_count の checked_mul だけでは弾けず、バイト長計算側の checked_mul
    // が拒否する必要があることの回帰確認（advisor 指摘: expected_bytes の乗算が
    // 素の `*` だとオーバーフロー時に debug ビルドは panic、release は 0 に
    // wrap して不正なテンソルを通してしまう）。
    let t = TensorProto {
        dims: vec![1i64 << 62],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![],
        int64_data: vec![],
        name: "byte_overflow_tensor".to_string(),
        raw_data: vec![],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("バイト長オーバーフローは拒否されるはず");
    match err {
        GraphError::ElementCountOverflow { tensor_name } => {
            assert_eq!(tensor_name, "byte_overflow_tensor");
        }
        other => panic!("ElementCountOverflow を期待したが {other:?}"),
    }
}

#[test]
fn empty_data_with_nonzero_dims_is_rejected_not_silently_accepted() {
    // float_data・raw_data のいずれも空だが dims=[2]（2 要素を期待）の不正入力。
    // TensorProto.data_location/external_data（本クレートが意図的に未定義）を
    // 使う参照専用テンソル等がこの形で decode されうるが、無言で「空データの
    // テンソル」として通してしまうと #78 のインタープリタに矛盾した
    // RawTensor（shape は 2 要素だが data は空）を渡すことになる（advisor 指摘）。
    // dims=[0] の真の空テンソルとは区別し、こちらは明示的に拒否する。
    let t = TensorProto {
        dims: vec![2],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![],
        int64_data: vec![],
        name: "empty_but_nonzero_dims_tensor".to_string(),
        raw_data: vec![],
    };
    let model = model_with_single_initializer(t);
    let err = build_graph(&model).expect_err("空データ・非ゼロ dims は拒否されるはず");
    match err {
        GraphError::RawDataByteLenMismatch {
            tensor_name,
            expected_bytes,
            actual_bytes,
        } => {
            assert_eq!(tensor_name, "empty_but_nonzero_dims_tensor");
            assert_eq!(expected_bytes, 8);
            assert_eq!(actual_bytes, 0);
        }
        other => panic!("RawDataByteLenMismatch を期待したが {other:?}"),
    }
}

#[test]
fn truly_empty_tensor_dims_zero_is_still_accepted() {
    // dims=[0] は「0 要素の空テンソル」を表す正当な値であり、上の
    // 「dims が非ゼロなのに data が空」の拒否対象と混同してはならない
    // （回帰確認: 一致検査を expected_bytes==0==raw_data.len() で通す経路）。
    let t = TensorProto {
        dims: vec![0],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![],
        int64_data: vec![],
        name: "truly_empty_tensor".to_string(),
        raw_data: vec![],
    };
    let model = model_with_single_initializer(t);
    let graph = build_graph(&model).expect("dims=[0] の空テンソルは受理されるはず");
    match graph.initializers.get("truly_empty_tensor") {
        Some(RawTensor::F32 { data, shape }) => {
            assert!(data.is_empty());
            assert_eq!(shape.as_slice(), &[0]);
        }
        other => panic!("F32 RawTensor を期待したが {other:?}"),
    }
}

#[test]
fn raw_data_takes_precedence_over_typed_float_data() {
    // `raw_data` と `float_data` が両方埋まっている細工モデル。ONNX リファレンス
    // 実装（onnxruntime 等）と同じ解決順序に合わせ、raw_data 側の値
    // （ここでは 9.0）を採用し、typed data 側（1.0）は無視されるべき
    // （Bugbot 指摘: typed data が raw_data を無言で shadow していた）。
    let t = TensorProto {
        dims: vec![1],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![1.0],
        int64_data: vec![],
        name: "both_fields".to_string(),
        raw_data: 9.0f32.to_le_bytes().to_vec(),
    };
    let model = model_with_single_initializer(t);
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    match graph.initializers.get("both_fields") {
        Some(RawTensor::F32 { data, .. }) => {
            assert_eq!(data.as_slice(), &[9.0]);
        }
        other => panic!("F32 RawTensor を期待したが {other:?}"),
    }
}

#[test]
fn raw_data_takes_precedence_over_typed_int64_data() {
    // FLOAT と同じ解決順序を INT64 でも確認する。
    let t = TensorProto {
        dims: vec![1],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::INT64,
        float_data: vec![],
        int64_data: vec![1],
        name: "both_fields_i64".to_string(),
        raw_data: 9i64.to_le_bytes().to_vec(),
    };
    let model = model_with_single_initializer(t);
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    match graph.initializers.get("both_fields_i64") {
        Some(RawTensor::I64 { data, .. }) => {
            assert_eq!(data.as_slice(), &[9]);
        }
        other => panic!("I64 RawTensor を期待したが {other:?}"),
    }
}

#[test]
fn duplicate_initializer_name_is_rejected_not_silently_overwritten() {
    // 同名の initializer が 2 つ含まれる不正な ONNX モデル。`HashMap::insert`
    // をそのまま使うと後勝ちで前者が無言上書きされてしまう（Bugbot 指摘）。
    // no-silent-skip 契約に従い明示的なエラーで拒否することを確認する。
    let t1 = TensorProto {
        dims: vec![1],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![1.0],
        int64_data: vec![],
        name: "dup".to_string(),
        raw_data: vec![],
    };
    let t2 = TensorProto {
        dims: vec![1],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![2.0],
        int64_data: vec![],
        name: "dup".to_string(),
        raw_data: vec![],
    };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![],
            name: "g".to_string(),
            initializer: vec![t1, t2],
            input: vec![],
            output: vec![],
        }),
    };
    let err = build_graph(&model).expect_err("initializer 名の重複は拒否されるはず");
    match err {
        GraphError::DuplicateInitializerName { tensor_name } => {
            assert_eq!(tensor_name, "dup");
        }
        other => panic!("DuplicateInitializerName を期待したが {other:?}"),
    }
}

#[test]
fn non_topological_node_order_is_rejected() {
    // ノード n1 が、まだどこからも生成されていない "phantom" を入力に取る
    // （トポロジカル順違反）。initializer・グラフ入力のいずれにも属さない。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["phantom".to_string()],
                output: vec!["y".to_string()],
                name: "n1".to_string(),
                op_type: "Identity".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    let err = build_graph(&model).expect_err("未生成入力の参照は拒否されるはず");
    match err {
        GraphError::NotTopologicallySorted {
            node_name,
            missing_input,
        } => {
            assert_eq!(node_name, "n1");
            assert_eq!(missing_input, "phantom");
        }
        other => panic!("NotTopologicallySorted を期待したが {other:?}"),
    }
}

#[test]
fn unknown_graph_output_is_rejected() {
    // GraphProto.output が、initializer にもグラフ入力にもどのノード出力にも
    // 一致しない名前（"nope"）を宣言している。唯一のノードは "y" を生成する
    // のみで "nope" は誰からも生成されない（レビュー指摘: 未検証コピーの再現）。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                name: "n1".to_string(),
                op_type: "Identity".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "nope".to_string(),
            }],
        }),
    };
    let err = build_graph(&model).expect_err("未生成の出力宣言は拒否されるはず");
    match err {
        GraphError::UnknownGraphOutput { tensor_name } => {
            assert_eq!(tensor_name, "nope");
        }
        other => panic!("UnknownGraphOutput を期待したが {other:?}"),
    }
}

#[test]
fn duplicate_node_output_name_is_rejected() {
    // 2 つのノードがともに出力 "y" を生成する（ONNX が要求するグラフ内 SSA
    // 違反）。`DuplicateInitializerName` と同一の欠陥クラス（レビュー指摘）。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![
                NodeProto {
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    name: "n1".to_string(),
                    op_type: "Identity".to_string(),
                    attribute: vec![],
                    domain: String::new(),
                },
                NodeProto {
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    name: "n2".to_string(),
                    op_type: "Identity".to_string(),
                    attribute: vec![],
                    domain: String::new(),
                },
            ],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    let err = build_graph(&model).expect_err("ノード出力名の重複は拒否されるはず");
    match err {
        GraphError::DuplicateOutputName {
            node_name,
            tensor_name,
        } => {
            assert_eq!(node_name, "n2");
            assert_eq!(tensor_name, "y");
        }
        other => panic!("DuplicateOutputName を期待したが {other:?}"),
    }
}

#[test]
fn node_output_shadowing_initializer_name_is_rejected() {
    // ノード出力名が既存 initializer 名と衝突する（無言シャドウの拒否）。
    let init = TensorProto {
        name: "w".to_string(),
        data_type: 1, // FLOAT
        dims: vec![1],
        float_data: vec![1.0],
        int64_data: vec![],
        raw_data: vec![],
    };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string()],
                output: vec!["w".to_string()],
                name: "n1".to_string(),
                op_type: "Identity".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![init],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "w".to_string(),
            }],
        }),
    };
    let err = build_graph(&model)
        .expect_err("initializer 名を無言シャドウするノード出力は拒否されるはず");
    match err {
        GraphError::DuplicateOutputName {
            node_name,
            tensor_name,
        } => {
            assert_eq!(node_name, "n1");
            assert_eq!(tensor_name, "w");
        }
        other => panic!("DuplicateOutputName を期待したが {other:?}"),
    }
}

#[test]
fn duplicate_graph_input_name_is_rejected_not_silently_absorbed() {
    // `GraphProto.input` に同名の ValueInfoProto が 2 件存在する（不正な
    // ONNX モデル）。`HashSet::insert` の戻り値を見ずに無言で吸収すると
    // `Graph.inputs` に重複名が残ってしまう（レビュー指摘 #77）。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                name: "n1".to_string(),
                op_type: "Identity".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![
                ValueInfoProto {
                    name: "x".to_string(),
                },
                ValueInfoProto {
                    name: "x".to_string(),
                },
            ],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    let err = build_graph(&model).expect_err("グラフ入力名の重複は拒否されるはず");
    match err {
        GraphError::DuplicateInputName { tensor_name } => {
            assert_eq!(tensor_name, "x");
        }
        other => panic!("DuplicateInputName を期待したが {other:?}"),
    }
}

#[test]
fn duplicate_graph_output_name_is_rejected_not_silently_absorbed() {
    // `GraphProto.output` に同名の ValueInfoProto が 2 件存在する（不正な
    // ONNX モデル）。入力側（duplicate_graph_input_name_is_rejected_not_
    // silently_absorbed）と対称の検査が無いと `Graph.outputs` に重複名が
    // 無言で複写され、no-silent-skip 契約と矛盾する（Bugbot 指摘）。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                name: "n1".to_string(),
                op_type: "Identity".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![
                ValueInfoProto {
                    name: "y".to_string(),
                },
                ValueInfoProto {
                    name: "y".to_string(),
                },
            ],
        }),
    };
    let err = build_graph(&model).expect_err("グラフ出力名の重複は拒否されるはず");
    match err {
        GraphError::DuplicateGraphOutputName { tensor_name } => {
            assert_eq!(tensor_name, "y");
        }
        other => panic!("DuplicateGraphOutputName を期待したが {other:?}"),
    }
}

#[test]
fn graph_input_name_matching_initializer_name_is_accepted() {
    // グラフ入力名が initializer 名と重複するのは pre-IR-4 の合法パターン
    // （拒否対象は `g.input` 内部の重複のみ。レビュー指摘 #77 で明示された
    // 区別を回帰させないための固定テスト）。
    let init = TensorProto {
        name: "x".to_string(),
        data_type: 1, // FLOAT
        dims: vec![1],
        float_data: vec![1.0],
        int64_data: vec![],
        raw_data: vec![],
    };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                name: "n1".to_string(),
                op_type: "Identity".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![init],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    assert!(build_graph(&model).is_ok());
}

#[test]
fn optional_empty_string_input_is_not_treated_as_missing() {
    // ONNX の省略可能入力は空文字列で表される規約（onnx.proto3）。
    // 空文字列入力はトポロジカル順検証の対象外とし、拒否されないことを確認する。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string(), String::new()],
                output: vec!["y".to_string()],
                name: "n1".to_string(),
                op_type: "Clip".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    assert!(build_graph(&model).is_ok());
}

#[test]
fn node_with_multiple_optional_empty_outputs_is_accepted() {
    // ONNX は入力側と同様に出力側も省略可能なものを空文字列で表す規約
    // （例: MaxPool の Indices・LSTM/GRU の Y_h/Y_c）。1 ノードが複数の
    // 空文字列出力を持っていても `DuplicateOutputName` を誤検出しないことを
    // 確認する（グラフ入力側の対称テスト。レビュー指摘 #77）。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string(), String::new(), String::new()],
                name: "n1".to_string(),
                op_type: "MaxPool".to_string(),
                attribute: vec![],
                domain: String::new(),
            }],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
        }),
    };
    assert!(build_graph(&model).is_ok());
}

#[test]
fn two_nodes_with_empty_trailing_output_are_both_accepted() {
    // 2 ノードがそれぞれ末尾出力を空文字列で省略するケース。空文字列出力を
    // 既知集合との衝突検査対象から除外していないと、2 ノード目で
    // `DuplicateOutputName { tensor_name: "" }` として誤拒否される
    // （レビュー指摘 #77）。
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            value_info: Vec::new(),
            sparse_initializer: Vec::new(),
            node: vec![
                NodeProto {
                    input: vec!["x".to_string()],
                    output: vec!["y1".to_string(), String::new()],
                    name: "n1".to_string(),
                    op_type: "MaxPool".to_string(),
                    attribute: vec![],
                    domain: String::new(),
                },
                NodeProto {
                    input: vec!["y1".to_string()],
                    output: vec!["y2".to_string(), String::new()],
                    name: "n2".to_string(),
                    op_type: "MaxPool".to_string(),
                    attribute: vec![],
                    domain: String::new(),
                },
            ],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y2".to_string(),
            }],
        }),
    };
    assert!(build_graph(&model).is_ok());
}

#[test]
fn model_without_graph_is_rejected() {
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: None,
    };
    let err = build_graph(&model).expect_err("graph 欠落は拒否されるはず");
    assert_eq!(err, GraphError::NoGraph);
}

// --- sparse_initializer の fail-closed 拒否（イシュー #2079） ---
//
// sparse テンソルは非対応（`docs/tensor-core-sparse-complex-decision.md`）。
// `prost::Message::decode` は未宣言フィールドを無言スキップするため、
// `sparse_initializer` だけが「最初から存在しない」ものとして扱われると
// no-silent-skip 契約（A03／A08）に反する。以下は存在検出のみで拒否される
// ことを確認する（中身の解釈はしない）。

fn dense_initializer_tensor(name: &str) -> TensorProto {
    TensorProto {
        dims: vec![1],
        data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![1.0],
        int64_data: vec![],
        name: name.to_string(),
        raw_data: vec![],
    }
}

fn sparse_tensor_with_values_name(name: &str) -> SparseTensorProto {
    // `SparseTensorProto.values` は `name`（tag=8）のみを宣言した軽量型
    // `SparseTensorValueName` として decode する（メモリ増幅対策。
    // イシュー #2079 codex-review 是正。`proto.rs` モジュール冒頭コメント
    // 「メモリ増幅対策」節参照）。`indices`／`dims` は構造体自体を宣言
    // しなくなったため、ここでの構築対象からも除いた。
    SparseTensorProto {
        values: Some(SparseTensorValueName {
            name: name.to_string(),
        }),
    }
}

#[test]
fn sparse_initializer_is_rejected_not_silently_skipped() {
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            sparse_initializer: vec![sparse_tensor_with_values_name("w_sparse")],
        }),
    };
    let err = build_graph(&model).expect_err("sparse_initializer の非空は拒否されるはず");
    match err {
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            assert_eq!(tensor_name, "w_sparse");
            assert_eq!(count, 1);
        }
        other => panic!("SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

#[test]
fn sparse_initializer_alongside_dense_initializer_is_rejected() {
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![],
            name: "g".to_string(),
            initializer: vec![dense_initializer_tensor("dense_w")],
            input: vec![],
            output: vec![],
            value_info: vec![],
            sparse_initializer: vec![
                sparse_tensor_with_values_name("w_sparse_1"),
                sparse_tensor_with_values_name("w_sparse_2"),
            ],
        }),
    };
    let err = build_graph(&model)
        .expect_err("dense initializer が併存していても sparse_initializer は拒否されるはず");
    match err {
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            assert_eq!(tensor_name, "w_sparse_1");
            assert_eq!(count, 2);
        }
        other => panic!("SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

#[test]
fn sparse_initializer_with_empty_values_name_is_still_rejected() {
    // `values` が `None`（非信頼入力）でも tensor_name は空文字列に fallback
    // し、拒否そのものは変わらないことを確認する。
    let sparse = SparseTensorProto { values: None };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            node: vec![],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![],
            value_info: vec![],
            sparse_initializer: vec![sparse],
        }),
    };
    let err =
        build_graph(&model).expect_err("values が None でも sparse_initializer は拒否されるはず");
    match err {
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            assert_eq!(tensor_name, "");
            assert_eq!(count, 1);
        }
        other => panic!("SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

#[test]
fn dense_twin_of_sparse_fixture_is_accepted() {
    // 対照テスト: fixture と同構造で sparse_initializer だけを空にしたモデルは
    // 成功する（sparse が拒否の唯一の原因であることの証明）。
    let relu = NodeProto {
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: "relu".to_string(),
        op_type: "Relu".to_string(),
        attribute: vec![],
        domain: String::new(),
    };
    let model = ModelProto {
        opset_import: Vec::new(),
        ir_version: 8,
        producer_name: "fandhe-ai-test".to_string(),
        graph: Some(GraphProto {
            node: vec![relu],
            name: "g".to_string(),
            initializer: vec![],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
            }],
            value_info: vec![],
            sparse_initializer: vec![],
        }),
    };
    let graph = build_graph(&model).expect("sparse_initializer が空なら成功するはず");
    assert_eq!(graph.inputs, vec!["x".to_string()]);
    assert_eq!(graph.outputs, vec!["y".to_string()]);
}

#[test]
fn raw_wire_bytes_with_graph_field_15_are_detected_as_sparse_initializer() {
    // 自前エンコーダ（`proto::encode_model`）を介さず、手書きの protobuf ワイヤ
    // バイト列を直接 decode する非循環検証。`GraphProto.sparse_initializer`
    // （field 15 → tag byte 0x7a）が外部バイト列から正しく読まれることの証明。
    //
    // ワイヤ構成（すべて length-delimited: tag = (field_no << 3) | 2）:
    //   ModelProto.graph            field  7 -> tag 0x3a
    //     GraphProto.sparse_initializer field 15 -> tag 0x7a
    //       SparseTensorProto.values field  1 -> tag 0x0a
    //         TensorProto.name      field  8 -> tag 0x42
    let tensor_name = b"probe_sparse";
    // TensorProto.name (field 8, tag 0x42) + varint length + bytes
    let mut tensor_proto_bytes = vec![0x42u8, tensor_name.len() as u8];
    tensor_proto_bytes.extend_from_slice(tensor_name);

    // SparseTensorProto.values (field 1, tag 0x0a) + varint length + TensorProto bytes
    let mut sparse_bytes = vec![0x0au8, tensor_proto_bytes.len() as u8];
    sparse_bytes.extend_from_slice(&tensor_proto_bytes);

    // GraphProto.sparse_initializer (field 15, tag 0x7a) + varint length + SparseTensorProto bytes
    let mut graph_bytes = vec![0x7au8, sparse_bytes.len() as u8];
    graph_bytes.extend_from_slice(&sparse_bytes);

    // ModelProto.graph (field 7, tag 0x3a) + varint length + GraphProto bytes
    let mut model_bytes = vec![0x3au8, graph_bytes.len() as u8];
    model_bytes.extend_from_slice(&graph_bytes);

    let model = ModelProto::decode(model_bytes.as_slice())
        .expect("手書きワイヤバイト列の decode に成功するはず");
    let graph = model
        .graph
        .as_ref()
        .expect("graph フィールドが decode されているはず");
    assert_eq!(graph.sparse_initializer.len(), 1);
    assert_eq!(
        graph.sparse_initializer[0]
            .values
            .as_ref()
            .map(|t| t.name.as_str()),
        Some("probe_sparse")
    );

    let err =
        build_graph(&model).expect_err("手書きワイヤ由来の sparse_initializer も拒否されるはず");
    match err {
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            assert_eq!(tensor_name, "probe_sparse");
            assert_eq!(count, 1);
        }
        other => panic!("SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

/// `decode_model` の bounded 事前走査（主対策。イシュー #2079 codex-review
/// 是正）が、`ModelProto::decode` を呼ぶ**前**に同じワイヤバイト列から
/// 同じ payload（tensor_name・count）を検出することを確認する
/// （`raw_wire_bytes_with_graph_field_15_are_detected_as_sparse_initializer`
/// の構造体側検証と対になる、`decode_model` 側の直接検証）。
#[test]
fn decode_model_rejects_raw_wire_bytes_with_graph_field_15_before_full_decode() {
    let tensor_name = b"probe_sparse";
    let mut tensor_proto_bytes = vec![0x42u8, tensor_name.len() as u8];
    tensor_proto_bytes.extend_from_slice(tensor_name);

    let mut sparse_bytes = vec![0x0au8, tensor_proto_bytes.len() as u8];
    sparse_bytes.extend_from_slice(&tensor_proto_bytes);

    let mut graph_bytes = vec![0x7au8, sparse_bytes.len() as u8];
    graph_bytes.extend_from_slice(&sparse_bytes);

    let mut model_bytes = vec![0x3au8, graph_bytes.len() as u8];
    model_bytes.extend_from_slice(&graph_bytes);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が sparse_initializer を拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(tensor_name, "probe_sparse");
            assert_eq!(count, 1);
        }
        other => panic!("DecodeModelError::SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

/// メモリ増幅対策の回帰固定（イシュー #2079 codex-review 指摘の本題）:
/// `sparse_initializer` の `values`（`TensorProto`）に巨大な `raw_data`
/// （tag=9・数 MiB）を持たせても、`decode_model` は bounded 事前走査が
/// `sparse_initializer`（tag=15）の存在を検出した時点で即座に拒否し、
/// `raw_data` の中身を一切構造体へ展開しない。修正前の実装（`raw_data`
/// を含む `TensorProto` 全体を `Vec<SparseTensorProto>` へ decode してから
/// `build_graph` が非空検査で拒否する経路）でも最終的には拒否されるが、
/// 拒否の**前**に `raw_data` 相当量のメモリ確保が発生していた（DoS）。
/// 本テストは拒否そのもの（payload の正しさ）を固定する回帰テストであり、
/// 「確保が起きないこと」自体は本テストでは直接測定できないが、
/// `SparseTensorProto`/`SparseTensorValueName` が `raw_data`（tag=9）を
/// 宣言していないことと合わせて、`prescan_sparse_initializer` が
/// `ModelProto::decode` より前に短絡することを保証する。
#[test]
fn decode_model_rejects_sparse_initializer_with_large_raw_data_payload_without_hanging() {
    const RAW_DATA_LEN: usize = 4 * 1024 * 1024; // 4 MiB

    let tensor_name = b"probe_large";
    let mut tensor_proto_bytes = vec![0x42u8, tensor_name.len() as u8];
    tensor_proto_bytes.extend_from_slice(tensor_name);
    // TensorProto.raw_data (field 9, tag 0x4a) + varint 長 + 4 MiB のダミーバイト列。
    // 4 MiB は 1 バイト varint 長では表現できないため 3 バイト varint で符号化する
    // （プロトコル的には妥当な符号化。decode_model がここへ踏み込まないことの
    // 検証が目的であり、値の中身は全て 0 で問題ない）。
    tensor_proto_bytes.push(0x4au8);
    encode_varint_into(RAW_DATA_LEN as u64, &mut tensor_proto_bytes);
    tensor_proto_bytes.extend(vec![0u8; RAW_DATA_LEN]);

    let mut sparse_bytes = vec![0x0au8];
    encode_varint_into(tensor_proto_bytes.len() as u64, &mut sparse_bytes);
    sparse_bytes.extend_from_slice(&tensor_proto_bytes);

    let mut graph_bytes = vec![0x7au8];
    encode_varint_into(sparse_bytes.len() as u64, &mut graph_bytes);
    graph_bytes.extend_from_slice(&sparse_bytes);

    let mut model_bytes = vec![0x3au8];
    encode_varint_into(graph_bytes.len() as u64, &mut model_bytes);
    model_bytes.extend_from_slice(&graph_bytes);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が巨大 raw_data 付き sparse_initializer を拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(tensor_name, "probe_large");
            assert_eq!(count, 1);
        }
        other => panic!("DecodeModelError::SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

/// protobuf の非 repeated メッセージフィールドは複数回出現するとマージされる
/// 仕様のため、`prescan_sparse_initializer` は `graph`（tag=7）の全出現を
/// 走査して `sparse_initializer` の出現数を合算する必要がある（1 回の
/// 出現だけを見ると分割された入力を見逃す）。本テストは `ModelProto` 直下に
/// `graph` フィールドを 2 回出現させ、1 個目に sparse を 1 件、2 個目に
/// sparse を 1 件持たせて count が 2 に合算されることを確認する。
#[test]
fn decode_model_sums_sparse_initializer_count_across_multiple_graph_occurrences() {
    fn sparse_initializer_field(name: &[u8]) -> Vec<u8> {
        let mut tensor_proto_bytes = vec![0x42u8, name.len() as u8];
        tensor_proto_bytes.extend_from_slice(name);
        let mut sparse_bytes = vec![0x0au8, tensor_proto_bytes.len() as u8];
        sparse_bytes.extend_from_slice(&tensor_proto_bytes);
        let mut field = vec![0x7au8, sparse_bytes.len() as u8];
        field.extend_from_slice(&sparse_bytes);
        field
    }

    let graph_chunk_1 = sparse_initializer_field(b"first");
    let graph_chunk_2 = sparse_initializer_field(b"second");

    let mut model_bytes = vec![0x3au8, graph_chunk_1.len() as u8];
    model_bytes.extend_from_slice(&graph_chunk_1);
    model_bytes.push(0x3au8);
    model_bytes.push(graph_chunk_2.len() as u8);
    model_bytes.extend_from_slice(&graph_chunk_2);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            // 最初の出現（"first"）の name を診断用に採用しつつ、2 出現分を合算する。
            assert_eq!(tensor_name, "first");
            assert_eq!(count, 2);
        }
        other => panic!("DecodeModelError::SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

/// `prescan_sparse_initializer` の「最初の要素」判定は出現順で最初の
/// `sparse_initializer` 要素を指し、`graph::build_graph` の
/// `g.sparse_initializer[0]` と同じ意味でなければならない（Cursor Bugbot
/// 指摘・#2079 是正）。最初の要素に `values.name` が無い場合、2 番目以降の
/// 要素まで名前を探しに行かず、空文字列のまま確定することを確認する
/// （探索を続けると `build_graph` 経由の `tensor_name`〈常に先頭要素・
/// 無名なら空文字列〉と食い違う）。
#[test]
fn decode_model_first_sparse_name_is_empty_when_first_element_has_no_name_even_if_later_has_one() {
    // 1 個目: values は存在するが name（tag=8）を持たない（TensorProto 側の
    // 他フィールドも省略した最小構成）。
    let sparse_1_values = vec![]; // 空の TensorProto（name フィールドなし）
    let mut sparse_1 = vec![0x0au8, sparse_1_values.len() as u8];
    sparse_1.extend_from_slice(&sparse_1_values);
    let mut graph_chunk_1 = vec![0x7au8, sparse_1.len() as u8];
    graph_chunk_1.extend_from_slice(&sparse_1);

    // 2 個目: values.name = "second"。
    let name = b"second";
    let mut sparse_2_values = vec![0x42u8, name.len() as u8];
    sparse_2_values.extend_from_slice(name);
    let mut sparse_2 = vec![0x0au8, sparse_2_values.len() as u8];
    sparse_2.extend_from_slice(&sparse_2_values);
    let mut graph_chunk_2 = vec![0x7au8, sparse_2.len() as u8];
    graph_chunk_2.extend_from_slice(&sparse_2);

    // 両方を同一 graph（tag=7）出現内に並べて置く（GraphProto 直下の
    // sparse_initializer は repeated フィールドのため出現順が保たれる）。
    let mut graph_bytes = graph_chunk_1;
    graph_bytes.extend_from_slice(&graph_chunk_2);

    let mut model_bytes = vec![0x3au8, graph_bytes.len() as u8];
    model_bytes.extend_from_slice(&graph_bytes);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            // 最初の要素に名前が無いため、2 個目の "second" を採用せず
            // 空文字列のまま確定する（build_graph の g.sparse_initializer[0]
            // 基準と同じ挙動）。
            assert_eq!(tensor_name, "");
            assert_eq!(count, 2);
        }
        other => panic!("DecodeModelError::SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

/// 同一 `SparseTensorProto` 内で `values`（singular message field。tag=1）
/// が複数回出現した場合、protobuf の後勝ちマージ規則により最後の出現の
/// `name` が採用されることを確認する（codex-review P2 是正）。
/// `prescan_sparse_initializer`（`decode_model` 経由）が報告する
/// `tensor_name` と、`ModelProto::decode`（層 2・構造体側の decode）が
/// 実際に decode した `SparseTensorProto.values.name` が一致することで、
/// 両経路の診断 payload の整合を確認する。
#[test]
fn decode_model_sparse_tensor_name_follows_protobuf_last_wins_merge_on_duplicate_values_field() {
    // values（tag=1）の 1 回目の出現: name = "first"。
    let mut values_1 = vec![0x42u8, b"first".len() as u8];
    values_1.extend_from_slice(b"first");
    let mut sparse_bytes = vec![0x0au8, values_1.len() as u8];
    sparse_bytes.extend_from_slice(&values_1);

    // values（tag=1）の 2 回目の出現（同一 SparseTensorProto 内）:
    // name = "second"。protobuf の後勝ちマージにより最終的な
    // `values.name` は "second" になるはず。
    let mut values_2 = vec![0x42u8, b"second".len() as u8];
    values_2.extend_from_slice(b"second");
    sparse_bytes.push(0x0au8);
    sparse_bytes.push(values_2.len() as u8);
    sparse_bytes.extend_from_slice(&values_2);

    let mut graph_bytes = vec![0x7au8, sparse_bytes.len() as u8];
    graph_bytes.extend_from_slice(&sparse_bytes);

    let mut model_bytes = vec![0x3au8, graph_bytes.len() as u8];
    model_bytes.extend_from_slice(&graph_bytes);

    // 層 1（`decode_model` の事前走査）が報告する tensor_name。
    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    let prescan_tensor_name = match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(count, 1, "SparseTensorProto の出現は 1 件のみ");
            tensor_name
        }
        other => panic!("DecodeModelError::SparseInitializerNotSupported を期待したが {other:?}"),
    };
    assert_eq!(
        prescan_tensor_name, "second",
        "同一 SparseTensorProto 内で values が複数回出現した場合は後勝ちで \
         最後の出現の name を採用するはず"
    );

    // 層 2（`ModelProto::decode` を直接呼ぶ経路。本クレート内テスト等）が
    // 実際に decode する `values.name` と一致することを確認する
    // （両経路の診断 payload の整合）。
    let decoded = ModelProto::decode(model_bytes.as_slice())
        .expect("prost の通常 decode は非空フィールドの重複出現を許容するはず");
    let graph = decoded.graph.expect("graph は必須");
    assert_eq!(graph.sparse_initializer.len(), 1);
    let decoded_name = graph.sparse_initializer[0]
        .values
        .as_ref()
        .map(|v| v.name.clone())
        .unwrap_or_default();
    assert_eq!(
        decoded_name, prescan_tensor_name,
        "prescan_sparse_initializer の tensor_name は ModelProto::decode 本体の \
         decode 結果と一致するはず"
    );
}

/// `values.name` が 256 バイトの診断名上限を超える場合、事前走査
/// （`decode_model` 経由。層 1）と `build_graph`（`ModelProto::decode` を
/// 直接呼ぶ経路。層 2）の両方が同じ `SPARSE_TENSOR_NAME_DIAG_CAP` で
/// 切り詰め、`tensor_name` payload が一致することを確認する
/// （codex-review P2 是正。2026-09-22。「診断契約の統一」の直接証明）。
#[test]
fn decode_model_and_build_graph_agree_on_256_byte_diag_name_cap_for_long_name() {
    // 300 バイトの ASCII 名（256 バイト上限を確実に超える）。
    let long_name: Vec<u8> = (0..300).map(|i| b'a' + (i % 26) as u8).collect();
    assert_eq!(long_name.len(), 300);

    let mut values_bytes = vec![0x42u8];
    encode_varint_len(&mut values_bytes, long_name.len());
    values_bytes.extend_from_slice(&long_name);

    let mut sparse_bytes = vec![0x0au8];
    encode_varint_len(&mut sparse_bytes, values_bytes.len());
    sparse_bytes.extend_from_slice(&values_bytes);

    let mut graph_bytes = vec![0x7au8];
    encode_varint_len(&mut graph_bytes, sparse_bytes.len());
    graph_bytes.extend_from_slice(&sparse_bytes);

    let mut model_bytes = vec![0x3au8];
    encode_varint_len(&mut model_bytes, graph_bytes.len());
    model_bytes.extend_from_slice(&graph_bytes);

    // 層 1: decode_model（事前走査）の tensor_name。
    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    let prescan_tensor_name = match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(count, 1);
            tensor_name
        }
        other => panic!("DecodeModelError::SparseInitializerNotSupported を期待したが {other:?}"),
    };
    assert_eq!(
        prescan_tensor_name.len(),
        256,
        "256 バイト上限で切り詰められるはず"
    );

    // 層 2: ModelProto::decode を直接呼び build_graph へ通す経路の
    // tensor_name（build_graph 自身が同じ上限で切り詰める）。
    let decoded = ModelProto::decode(model_bytes.as_slice())
        .expect("300 バイトの name フィールドは通常 decode では制限されない");
    let build_err =
        build_graph(&decoded).expect_err("sparse_initializer は build_graph でも拒否されるはず");
    let build_tensor_name = match build_err {
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            assert_eq!(count, 1);
            tensor_name
        }
        other => panic!("GraphError::SparseInitializerNotSupported を期待したが {other:?}"),
    };
    assert_eq!(
        build_tensor_name.len(),
        256,
        "build_graph 側も同じ 256 バイト上限で切り詰めるはず"
    );

    assert_eq!(
        prescan_tensor_name, build_tensor_name,
        "decode_model（層 1）と build_graph（層 2）の tensor_name payload は \
         一致するはず（診断契約の統一）"
    );
}

/// `bytes` へ protobuf varint 形式の長さを追記するテストヘルパ（本テスト
/// ファイル内の 300 バイト級の length-delimited フィールド構築に使う。
/// 既存の `sparse_1.len() as u8` 直書きパターンは 128 バイト以上の長さを
/// 正しく符号化できない〈varint の継続ビットが必要〉ため、このヘルパで
/// 汎用化する）。
fn encode_varint_len(bytes: &mut Vec<u8>, mut len: usize) {
    loop {
        let mut byte = (len & 0x7f) as u8;
        len >>= 7;
        if len != 0 {
            byte |= 0x80;
        }
        bytes.push(byte);
        if len == 0 {
            break;
        }
    }
}

/// 事前走査中に不正な形式（length-delimited フィールドの長さがバッファ
/// 終端を超える）に遭遇した場合、`prescan_sparse_initializer` は判定を
/// 確定させず `ModelProto::decode` 本体へ委ねる（本体が同じ不正入力を
/// 独立に検証し `DecodeModelError::Wire` で拒否する）ことを確認する。
#[test]
fn decode_model_falls_back_to_wire_decode_error_on_malformed_length_delimiter() {
    // tag=7（graph, LengthDelimited）を宣言しつつ、実際のバッファ長を超える
    // 長さ（0xff）を主張する不正なバイト列。
    let malformed = [0x3au8, 0xffu8, 0x01u8];
    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&malformed)
        .expect_err("壊れたバイト列を decode_model が受理してしまった");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::Wire(_) => {}
        other => panic!("DecodeModelError::Wire を期待したが {other:?}"),
    }
}

/// codex-review P0 指摘（イシュー #2079・2026-09-22 是正）の回帰固定その 1:
/// `sparse_initializer`（tag=15）を既に検出して `count` を加算した**後**、
/// 同じ `GraphProto` 内で事前走査が不正な形式（length-delimited フィールドの
/// 宣言長がバッファ終端を超える）に遭遇しても、`decode_model` は
/// 「sparse なし」と同一視して `ModelProto::decode` へフォールバックせず、
/// 検出済みの `SparseInitializerNotSupported` を確定して返すことを確認する
/// （修正前は事前走査の `?` 伝播により検出結果が握りつぶされ、`ModelProto::
/// decode` が同じ不正入力を独立に検証した結果〈本テストの構成では偶然
/// `Wire` エラーになる〉へ素通ししていた）。
#[test]
fn decode_model_still_rejects_sparse_initializer_when_malformed_field_follows_in_same_graph() {
    let tensor_name = b"probe_scan_error";
    let mut tensor_proto_bytes = vec![0x42u8, tensor_name.len() as u8];
    tensor_proto_bytes.extend_from_slice(tensor_name);

    let mut sparse_bytes = vec![0x0au8, tensor_proto_bytes.len() as u8];
    sparse_bytes.extend_from_slice(&tensor_proto_bytes);

    let mut graph_bytes = vec![0x7au8, sparse_bytes.len() as u8];
    graph_bytes.extend_from_slice(&sparse_bytes);
    // sparse_initializer（tag=15）検出後、同じ graph 内に不正な
    // length-delimited フィールド（宣言長のバッファ終端を超える不完全な
    // varint）を追記する。
    graph_bytes.push(0x12u8); // 任意のフィールド（tag=2, LengthDelimited）
    graph_bytes.push(0xffu8); // 継続ビットが立ったまま終端する不完全な varint 長

    let mut model_bytes = vec![0x3au8, graph_bytes.len() as u8];
    model_bytes.extend_from_slice(&graph_bytes);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(tensor_name, "probe_scan_error");
            assert_eq!(count, 1);
        }
        other => panic!(
            "検出済み sparse_initializer の後続走査失敗（同一 graph 内）でも \
             SparseInitializerNotSupported を期待したが {other:?}"
        ),
    }
}

/// codex-review P0 指摘の回帰固定その 2: 不正フィールドが `GraphProto` の
/// **外**（`ModelProto` 直下・`graph` フィールド出現の後）にある場合も、
/// 既に検出済みの `sparse_initializer` を見失わないことを確認する。
#[test]
fn decode_model_still_rejects_sparse_initializer_when_malformed_field_follows_graph_at_top_level() {
    let tensor_name = b"probe_top_level_scan_error";
    let mut tensor_proto_bytes = vec![0x42u8, tensor_name.len() as u8];
    tensor_proto_bytes.extend_from_slice(tensor_name);

    let mut sparse_bytes = vec![0x0au8, tensor_proto_bytes.len() as u8];
    sparse_bytes.extend_from_slice(&tensor_proto_bytes);

    let mut graph_bytes = vec![0x7au8, sparse_bytes.len() as u8];
    graph_bytes.extend_from_slice(&sparse_bytes);

    let mut model_bytes = vec![0x3au8, graph_bytes.len() as u8];
    model_bytes.extend_from_slice(&graph_bytes);
    // graph（tag=7）出現の後、ModelProto 直下に不正なフィールド（宣言長が
    // バッファ終端を超える）を追記する。
    model_bytes.push(0x12u8);
    model_bytes.push(0xffu8);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(tensor_name, "probe_top_level_scan_error");
            assert_eq!(count, 1);
        }
        other => panic!(
            "検出済み sparse_initializer の後続走査失敗（ModelProto 直下）でも \
             SparseInitializerNotSupported を期待したが {other:?}"
        ),
    }
}

/// codex-review P0 指摘の本題（メモリ増幅の実バイパス経路）の回帰固定:
/// `SPARSE_TENSOR_NAME_DIAG_CAP`（256 バイト）を大きく超える巨大な
/// `values.name`（1 MiB）を持つ sparse_initializer の直後に、事前走査が
/// 判定を確定できない不正フィールドを配置しても、`decode_model` が
/// `ModelProto::decode`（`SparseTensorValueName.name` を無制限に構造体へ
/// 展開する構造体側 decode 経路。本モジュール冒頭コメント「メモリ増幅
/// 対策」節の層 2）へ素通ししないことを固定する。修正前はこの組合せで
/// 事前走査が「判定不能」を返し、`ModelProto::decode` が巨大な `name` を
/// 構造体へ完全展開してから初めて拒否していた（拒否の**前**に巨大確保が
/// 発生する DoS。イシュー #2079 codex-review 指摘）。
#[test]
fn decode_model_does_not_fall_back_to_full_decode_when_large_name_sparse_initializer_is_followed_by_malformed_field()
 {
    const HUGE_NAME_LEN: usize = 1024 * 1024; // 1 MiB（256 バイト cap を大きく超える）
    let huge_name = vec![b'x'; HUGE_NAME_LEN];

    let mut tensor_proto_bytes = vec![0x42u8];
    encode_varint_into(huge_name.len() as u64, &mut tensor_proto_bytes);
    tensor_proto_bytes.extend_from_slice(&huge_name);

    let mut sparse_bytes = vec![0x0au8];
    encode_varint_into(tensor_proto_bytes.len() as u64, &mut sparse_bytes);
    sparse_bytes.extend_from_slice(&tensor_proto_bytes);

    let mut graph_bytes = vec![0x7au8];
    encode_varint_into(sparse_bytes.len() as u64, &mut graph_bytes);
    graph_bytes.extend_from_slice(&sparse_bytes);
    // sparse_initializer 検出後、同じ graph 内に不正な length-delimited
    // フィールド（宣言長がバッファ終端を超える）を追記する。
    graph_bytes.push(0x12u8);
    graph_bytes.push(0xffu8);

    let mut model_bytes = vec![0x3au8];
    encode_varint_into(graph_bytes.len() as u64, &mut model_bytes);
    model_bytes.extend_from_slice(&graph_bytes);

    let err = fandhe_ai_onnx_interop::onnx::proto::decode_model(&model_bytes)
        .expect_err("decode_model が拒否するはず");
    match err {
        fandhe_ai_onnx_interop::onnx::proto::DecodeModelError::SparseInitializerNotSupported {
            tensor_name,
            count,
        } => {
            assert_eq!(count, 1);
            assert_eq!(
                tensor_name.len(),
                256,
                "事前走査の診断名は SPARSE_TENSOR_NAME_DIAG_CAP で切り詰められるはず \
                 （巨大 name が構造体側 decode まで素通ししていないことの証明）"
            );
        }
        other => panic!(
            "巨大 name + 後続 malformed の組合せでも SparseInitializerNotSupported を \
             期待したが {other:?}（ModelProto::decode への素通しが疑われる）"
        ),
    }
}

/// varint を可変長（1〜10 バイト）で書き出す最小ヘルパー（LEB128）。
/// 上記の大サイズ payload テスト用に、既存の「1 バイト決め打ち」の
/// 手書きワイヤ構築ヘルパーでは表現できない長さを符号化するために使う。
fn encode_varint_into(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

#[test]
fn sparse_initializer_fixture_file_is_rejected() {
    let model = load_model("sparse_initializer.onnx");
    let err = build_graph(&model).expect_err("fixture の sparse_initializer は拒否されるはず");
    match err {
        GraphError::SparseInitializerNotSupported { tensor_name, count } => {
            assert_eq!(tensor_name, "w_sparse");
            assert_eq!(count, 1);
        }
        other => panic!("SparseInitializerNotSupported を期待したが {other:?}"),
    }
}

#[test]
fn attribute_proto_round_trips_with_tensor_field() {
    // AttributeProto.t（TensorProto を含む属性）が手書き derive で正しく
    // encode/decode できることを確認する（#78/#79 で Constant ノード等の属性
    // アクセスに使う前提の回帰確認）。
    let attr = AttributeProto {
        name: "value".to_string(),
        f: 0.0,
        i: 0,
        s: vec![],
        t: Some(TensorProto {
            dims: vec![1],
            data_type: fandhe_ai_onnx_interop::onnx::proto::data_type::INT64,
            float_data: vec![],
            int64_data: vec![7],
            name: "const_t".to_string(),
            raw_data: vec![],
        }),
        floats: vec![],
        ints: vec![],
        r#type: 0,
    };
    let bytes = attr.encode_to_vec();
    let decoded = AttributeProto::decode(bytes.as_slice()).expect("decode に成功するはず");
    assert_eq!(decoded, attr);
}

// --- transformer.onnx: 実機規模フィクスチャ（非コミット・#[ignore] 分離） ---

#[test]
#[ignore = "12MB の transformer.onnx を非コミット方針としているため。tests/fixtures/README.md の取得手順を参照"]
fn transformer_onnx_decodes_expected_graph_structure() {
    // `cargo test -- --ignored`（`make test-ignored` 含む）は #[ignore] を
    // 無視して本テストを実行するため、非コミットの transformer.onnx を
    // 取得していない環境（フィクスチャ未取得のマシン全般）では環境変数が
    // 未設定になる。その場合は fail ではなく早期 return でスキップし、
    // ワークスペース全体の --ignored 実行を hard-fail させない
    // （tests/fixtures/README.md の取得手順を参照。#77 Bugbot 指摘対応）。
    let Ok(path) = std::env::var("ONNX_INTEROP_TRANSFORMER_ONNX") else {
        eprintln!(
            "skip: ONNX_INTEROP_TRANSFORMER_ONNX 未設定のため transformer_onnx_decodes_expected_graph_structure をスキップします（tests/fixtures/README.md 参照）"
        );
        return;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("読み込み失敗 {path}: {e}"));
    let model = ModelProto::decode(bytes.as_slice()).expect("decode に成功するはず");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    // PoC-v2-6 実測値（docs/spec/03-poc/poc-v2-6-interop/evidence/transformer_probe.log）
    assert_eq!(graph.nodes.len(), 165);
    assert_eq!(graph.initializers.len(), 12);
    assert_eq!(graph.inputs, vec!["input".to_string()]);
    assert_eq!(graph.outputs, vec!["output".to_string()]);

    let mut op_types: Vec<&str> = graph.nodes.iter().map(|n| n.op_type.as_str()).collect();
    op_types.sort_unstable();
    op_types.dedup();
    assert_eq!(
        op_types.len(),
        20,
        "op_type 種別数は 20 のはず（transformer_probe.log）"
    );
}
