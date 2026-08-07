//! `onnx::proto` / `onnx::graph` のデコード・検証テスト（TASK-7.2a・イシュー #77）。
//!
//! コミット済みの極小 fixture（`model.onnx`・`slice_repro.onnx`）を使うテストは
//! CI（self-hosted）で常時実行する。`transformer.onnx`（12MB・非コミット）を使う
//! テストは実機依存テストと同じ運用で `#[ignore]` 分離し、環境変数でパスを指定
//! されたときのみ実行する（`tests/fixtures/README.md` の取得手順参照）。

use onnx_interop::onnx::graph::{GraphError, RawTensor, build_graph};
use onnx_interop::onnx::proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::BOOL,
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
        data_type: onnx_interop::onnx::proto::data_type::BOOL,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT16,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
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
        data_type: onnx_interop::onnx::proto::data_type::INT64,
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
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![1.0],
        int64_data: vec![],
        name: "dup".to_string(),
        raw_data: vec![],
    };
    let t2 = TensorProto {
        dims: vec![1],
        data_type: onnx_interop::onnx::proto::data_type::FLOAT,
        float_data: vec![2.0],
        int64_data: vec![],
        name: "dup".to_string(),
        raw_data: vec![],
    };
    let model = ModelProto {
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
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
        ir_version: 8,
        producer_name: "test".to_string(),
        graph: None,
    };
    let err = build_graph(&model).expect_err("graph 欠落は拒否されるはず");
    assert_eq!(err, GraphError::NoGraph);
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
            data_type: onnx_interop::onnx::proto::data_type::INT64,
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
