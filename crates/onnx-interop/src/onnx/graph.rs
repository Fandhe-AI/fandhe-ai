//! `ModelProto` を内部グラフ表現へ変換する（REQ-7・TASK-7.2a）。
//!
//! ONNX 仕様（`GraphProto` のドキュメント）は「`node` リストはトポロジカル順」を
//! 要求しているが、本実装は不正入力（外部フォーマットゆえに信頼できない前提。
//! OWASP A03・`security.md`）に備えて自前でトポロジカル順を検証する。検証に
//! 失敗した場合は `GraphError::NotTopologicallySorted` を返し、呼び出し元
//! （#78 のインタープリタ・将来の codegen）はこれを見て安全に停止できる。
//!
//! `decode_tensor` は要素データの復号より先に dims・要素数・データ長の整合を
//! 検査する（長さ・形状検証の先行。イシュー #77 の受け入れ要件・`security.md` A03）。

use super::proto::{GraphProto, ModelProto, NodeProto, TensorProto};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// グラフ構築・テンソル復号で発生しうるエラー。本番経路で `unwrap()` /
/// `expect()` を使わない方針（`coding-rust.md`）に従い、不正入力は必ずこの型で
/// 呼び出し元へ伝播する。
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphError {
    /// `ModelProto` に `graph` フィールドが存在しない。
    NoGraph,
    /// ノードの入力が、それ以前に生成された initializer / グラフ入力 / 先行ノード
    /// 出力のいずれにも属さない（トポロジカル順違反）。
    NotTopologicallySorted {
        node_name: String,
        missing_input: String,
    },
    /// `TensorProto.data_type` が本クレートの対応範囲外（無言 skip はしない）。
    UnknownDataType { tensor_name: String, data_type: i32 },
    /// `dims` に負の値が含まれる（不正な形状）。
    NegativeDim { tensor_name: String, dim: i64 },
    /// `dims` の積（要素数）が `i64`/`usize` の範囲を超える（不正な巨大形状による
    /// 資源枯渇攻撃を拒否する。OWASP A03）。
    ElementCountOverflow { tensor_name: String },
    /// 実データ（`float_data` / `int64_data`）の要素数が `dims` から導出した
    /// 期待要素数と一致しない（typed data フィールド用。要素数同士の比較）。
    DataLenMismatch {
        tensor_name: String,
        expected_elements: usize,
        actual_elements: usize,
    },
    /// `raw_data`（バイト列）の長さが `dims` から導出した期待バイト長と一致しない。
    /// `DataLenMismatch` と分離する理由（Bugbot 指摘）: 要素サイズの倍数でない
    /// バイト長を要素数へ切り捨て除算すると `expected_elements` と同じ値に丸まり、
    /// 「期待要素数と実要素数が同じなのにエラー」という自己矛盾した診断になる。
    /// バイト長のまま報告することで丸めによる矛盾を避ける。
    RawDataByteLenMismatch {
        tensor_name: String,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    /// `GraphProto.initializer` に同名の initializer が複数含まれる（不正な
    /// ONNX モデル）。`HashMap::insert` は同名キーを後勝ちで無言上書きするため、
    /// 本クレートが謳う no-silent-skip 契約に従い明示的に拒否する（Bugbot 指摘）。
    DuplicateInitializerName { tensor_name: String },
    /// ノードの出力名が、既に生成済み（先行ノード出力／initializer／グラフ入力）の
    /// 名前と衝突する（ONNX が要求するグラフ内 SSA 違反）。`DuplicateInitializerName`
    /// と同一の欠陥クラス（no-silent-skip 契約。レビュー指摘）。
    DuplicateOutputName {
        node_name: String,
        tensor_name: String,
    },
    /// `GraphProto.input` に同名の `ValueInfoProto` が複数含まれる。
    /// `DuplicateInitializerName`／`DuplicateOutputName` と同一の欠陥クラス
    /// （no-silent-skip 契約。レビュー指摘 #77）。initializer 名との重複は
    /// pre-IR-4 の合法パターンのため対象外で、`g.input` 内部の重複のみを拒否する。
    DuplicateInputName { tensor_name: String },
    /// `GraphProto.output` が、initializer／グラフ入力／いずれのノード出力にも
    /// 属さない名前を宣言している（後段が「グラフは既に妥当である」前提で
    /// 実装できるようにするための検証。レビュー指摘）。
    UnknownGraphOutput { tensor_name: String },
    /// `GraphProto.output` に同名の `ValueInfoProto` が複数含まれる。
    /// `DuplicateInputName` と対称の検査（Bugbot 指摘）: `g.input` 側のみ重複を
    /// 拒否し `g.output` 側を素通りさせると、不正な重複出力リストがそのまま
    /// `Graph.outputs` へ複写され、本モジュールが謳う no-silent-skip 契約
    /// （他の全名前衝突に適用している方針）と矛盾する。
    DuplicateGraphOutputName { tensor_name: String },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NoGraph => write!(f, "ModelProto に graph がありません"),
            GraphError::NotTopologicallySorted {
                node_name,
                missing_input,
            } => write!(
                f,
                "トポロジ矛盾: ノード '{node_name}' が未生成の入力 '{missing_input}' を参照"
            ),
            GraphError::UnknownDataType {
                tensor_name,
                data_type,
            } => write!(
                f,
                "未対応の TensorProto.data_type（tensor={tensor_name}）: {data_type}"
            ),
            GraphError::NegativeDim { tensor_name, dim } => {
                write!(f, "dims に負の値（tensor={tensor_name}）: {dim}")
            }
            GraphError::ElementCountOverflow { tensor_name } => {
                write!(
                    f,
                    "dims の積（要素数）がオーバーフロー（tensor={tensor_name}）"
                )
            }
            GraphError::DataLenMismatch {
                tensor_name,
                expected_elements,
                actual_elements,
            } => {
                write!(
                    f,
                    "データ長不整合（tensor={tensor_name}）: dims から期待される要素数={expected_elements} 実データ要素数={actual_elements}"
                )
            }
            GraphError::RawDataByteLenMismatch {
                tensor_name,
                expected_bytes,
                actual_bytes,
            } => {
                write!(
                    f,
                    "raw_data バイト長不整合（tensor={tensor_name}）: dims から期待されるバイト長={expected_bytes} 実バイト長={actual_bytes}"
                )
            }
            GraphError::DuplicateInitializerName { tensor_name } => {
                write!(f, "initializer 名の重複（tensor={tensor_name}）")
            }
            GraphError::DuplicateOutputName {
                node_name,
                tensor_name,
            } => {
                write!(
                    f,
                    "出力名の重複（node='{node_name}' tensor={tensor_name}）: 既存の initializer / グラフ入力 / 先行ノード出力と衝突"
                )
            }
            GraphError::DuplicateInputName { tensor_name } => {
                write!(f, "グラフ入力名の重複（tensor={tensor_name}）")
            }
            GraphError::UnknownGraphOutput { tensor_name } => {
                write!(
                    f,
                    "GraphProto.output が未生成のテンソルを参照（tensor={tensor_name}）"
                )
            }
            GraphError::DuplicateGraphOutputName { tensor_name } => {
                write!(f, "グラフ出力名の重複（tensor={tensor_name}）")
            }
        }
    }
}
impl std::error::Error for GraphError {}

/// 生の initializer / 定数テンソル（`tensor-core` の型へのマッピング前）。
/// `Bool`／`F16` はイシュー #274 で追加（`onnx::proto::data_type::BOOL`／`FLOAT16`）。
#[derive(Clone, Debug, PartialEq)]
pub enum RawTensor {
    F32 {
        data: Vec<f32>,
        shape: Vec<i64>,
    },
    I64 {
        data: Vec<i64>,
        shape: Vec<i64>,
    },
    Bool {
        data: Vec<bool>,
        shape: Vec<i64>,
    },
    F16 {
        data: Vec<half::f16>,
        shape: Vec<i64>,
    },
}

/// 実行順が確定したグラフ。`nodes` の順序がトポロジカル順であることは
/// `build_graph` が検証済み。
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<NodeProto>,
    pub initializers: HashMap<String, RawTensor>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// `dims` から要素数を安全に計算する。負の dim・オーバーフローはここで拒否し、
/// 要素データの復号（バイト列走査）より前に弾く（長さ・形状検証の先行）。
fn element_count(tensor_name: &str, dims: &[i64]) -> Result<usize, GraphError> {
    let mut count: usize = 1;
    for &d in dims {
        if d < 0 {
            return Err(GraphError::NegativeDim {
                tensor_name: tensor_name.to_string(),
                dim: d,
            });
        }
        count = count
            .checked_mul(d as usize)
            .ok_or_else(|| GraphError::ElementCountOverflow {
                tensor_name: tensor_name.to_string(),
            })?;
    }
    Ok(count)
}

/// `TensorProto` を `RawTensor` へ復号する。
///
/// 検証順序（イシュー #77 受け入れ要件: 要素データ復号より先に行う）:
/// 1. `dims` の非負性・要素数の `checked_mul`（`element_count`）
/// 2. 実データ（`raw_data` はバイト長 / `float_data`・`int64_data` は要素数）と
///    期待要素数の一致
/// 3. 上記を満たして初めてバイト列 -> 数値列の変換を行う
///
/// `raw_data` と typed data フィールド（`float_data`／`int64_data`）が両方
/// 埋まっている場合は `raw_data` を優先する（ONNX リファレンス実装
/// （onnxruntime 等）と同じ解決順序。Bugbot 指摘）。typed data を先に見ると、
/// 細工・不正な形式のモデルが `raw_data` 側と異なる値へデコードされ、かつ
/// `raw_data` のバイト長検証を一切通らずにすり抜けてしまう（no-silent-skip
/// 契約違反）ため、両方を無条件で検査対象にする。
///
/// 未対応 `data_type` は `UnknownDataType` で拒否する（無言 skip 禁止）。
pub(crate) fn decode_tensor(t: &TensorProto) -> Result<RawTensor, GraphError> {
    let shape = t.dims.clone();
    let expected_elements = element_count(&t.name, &shape)?;

    match t.data_type {
        super::proto::data_type::FLOAT => {
            if !t.raw_data.is_empty() {
                // raw_data が typed data より優先される（raw_data 優先の解決順序）。
                // expected_elements の乗算自体もオーバーフローしうるため checked_mul
                // で拒否する（巨大 dims による資源枯渇攻撃を弾く。A03）。
                let expected_bytes = expected_elements.checked_mul(4).ok_or_else(|| {
                    GraphError::ElementCountOverflow {
                        tensor_name: t.name.clone(),
                    }
                })?;
                if t.raw_data.len() != expected_bytes {
                    return Err(GraphError::RawDataByteLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_bytes,
                        actual_bytes: t.raw_data.len(),
                    });
                }
                let data: Vec<f32> = t
                    .raw_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Ok(RawTensor::F32 { data, shape })
            } else if !t.float_data.is_empty() {
                if t.float_data.len() != expected_elements {
                    return Err(GraphError::DataLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_elements,
                        actual_elements: t.float_data.len(),
                    });
                }
                Ok(RawTensor::F32 {
                    data: t.float_data.clone(),
                    shape,
                })
            } else {
                // raw_data・float_data のいずれも空。「dims=[0] 等の空テンソル」と
                // 「data フィールドが一つも埋まっていない不正入力（例: external_data
                // 参照だが本クレートは TensorProto.data_location 等を意図的に未定義。
                // プロトコル上は未宣言フィールドとして無言でスキップされてしまう）」を
                // 区別できないため、必ず expected_bytes との一致検査を通す（0 要素
                // なら 0 バイト一致で素通りする）。
                let expected_bytes = expected_elements.checked_mul(4).ok_or_else(|| {
                    GraphError::ElementCountOverflow {
                        tensor_name: t.name.clone(),
                    }
                })?;
                if t.raw_data.len() != expected_bytes {
                    return Err(GraphError::RawDataByteLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_bytes,
                        actual_bytes: t.raw_data.len(),
                    });
                }
                Ok(RawTensor::F32 {
                    data: Vec::new(),
                    shape,
                })
            }
        }
        super::proto::data_type::INT64 => {
            if !t.raw_data.is_empty() {
                // raw_data が typed data より優先される（raw_data 優先の解決順序）。
                let expected_bytes = expected_elements.checked_mul(8).ok_or_else(|| {
                    GraphError::ElementCountOverflow {
                        tensor_name: t.name.clone(),
                    }
                })?;
                if t.raw_data.len() != expected_bytes {
                    return Err(GraphError::RawDataByteLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_bytes,
                        actual_bytes: t.raw_data.len(),
                    });
                }
                // chunks_exact(8) により各チャンクは必ず 8 要素の &[u8] となるため、
                // 固定長配列への変換（try_into）は失敗しない。`unwrap()` の代わりに
                // 固定長配列を直接構築し本番経路の unwrap を避ける（coding-rust.md）。
                let data: Vec<i64> = t
                    .raw_data
                    .chunks_exact(8)
                    .map(|b| {
                        let mut buf = [0u8; 8];
                        buf.copy_from_slice(b);
                        i64::from_le_bytes(buf)
                    })
                    .collect();
                Ok(RawTensor::I64 { data, shape })
            } else if !t.int64_data.is_empty() {
                if t.int64_data.len() != expected_elements {
                    return Err(GraphError::DataLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_elements,
                        actual_elements: t.int64_data.len(),
                    });
                }
                Ok(RawTensor::I64 {
                    data: t.int64_data.clone(),
                    shape,
                })
            } else {
                // FLOAT 分岐と同じ理由（external_data 等で data が一つも埋まらない
                // 不正入力を「空テンソル」と誤認しないための一致検査。checked_mul で
                // 乗算オーバーフローも拒否する）。
                let expected_bytes = expected_elements.checked_mul(8).ok_or_else(|| {
                    GraphError::ElementCountOverflow {
                        tensor_name: t.name.clone(),
                    }
                })?;
                if t.raw_data.len() != expected_bytes {
                    return Err(GraphError::RawDataByteLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_bytes,
                        actual_bytes: t.raw_data.len(),
                    });
                }
                Ok(RawTensor::I64 {
                    data: Vec::new(),
                    shape,
                })
            }
        }
        super::proto::data_type::BOOL => {
            // BOOL は `raw_data` 経由のみ対応する（本クレートの `TensorProto` は
            // 部分実装〈proto.rs 冒頭コメント〉であり、ONNX 仕様上の typed data
            // フィールド `int32_data`（BOOL は 1 要素 1 int32 として符号化）は
            // 意図的に未定義。`transformer.onnx` 等の PyTorch エクスポートは
            // initializer を一貫して `raw_data` で埋めるため実用上問題ない。
            // 1 バイト/要素・非ゼロ→true は ONNX/NumPy の bool テンソル慣習
            // （`onnx-interop::ops::cast::cast_to_bool` と同じ解釈）。
            let expected_bytes = expected_elements;
            if t.raw_data.len() != expected_bytes {
                return Err(GraphError::RawDataByteLenMismatch {
                    tensor_name: t.name.clone(),
                    expected_bytes,
                    actual_bytes: t.raw_data.len(),
                });
            }
            let data: Vec<bool> = t.raw_data.iter().map(|&b| b != 0).collect();
            Ok(RawTensor::Bool { data, shape })
        }
        super::proto::data_type::FLOAT16 => {
            // FLOAT16 も BOOL と同じ理由で `raw_data` のみ対応する（typed data
            // フィールド `int32_data` は未定義）。IEEE754 binary16 のリトルエンディアン
            // 2 バイト/要素（onnx.proto3 `TensorProto` 仕様）。
            let expected_bytes = expected_elements.checked_mul(2).ok_or_else(|| {
                GraphError::ElementCountOverflow {
                    tensor_name: t.name.clone(),
                }
            })?;
            if t.raw_data.len() != expected_bytes {
                return Err(GraphError::RawDataByteLenMismatch {
                    tensor_name: t.name.clone(),
                    expected_bytes,
                    actual_bytes: t.raw_data.len(),
                });
            }
            let data: Vec<half::f16> = t
                .raw_data
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                .collect();
            Ok(RawTensor::F16 { data, shape })
        }
        other => Err(GraphError::UnknownDataType {
            tensor_name: t.name.clone(),
            data_type: other,
        }),
    }
}

/// `ModelProto` から実行可能な内部グラフを構築する。
///
/// 呼び出し元は #78（インタープリタ基盤）・将来の codegen。initializer の復号
/// （`decode_tensor`）とノード列のトポロジカル順検証をここで完結させ、後段は
/// 「グラフは既に妥当である」前提で実装できるようにする。
pub fn build_graph(model: &ModelProto) -> Result<Graph, GraphError> {
    let g: &GraphProto = model.graph.as_ref().ok_or(GraphError::NoGraph)?;

    // `HashMap::insert` は同名キーを後勝ちで無言上書きするため、事前に重複を
    // 検出して拒否する（不正な ONNX モデル。Bugbot 指摘・no-silent-skip 契約）。
    let mut initializers = HashMap::new();
    for init in &g.initializer {
        let decoded = decode_tensor(init)?;
        if initializers.insert(init.name.clone(), decoded).is_some() {
            return Err(GraphError::DuplicateInitializerName {
                tensor_name: init.name.clone(),
            });
        }
    }

    // トポロジカル順の検証: 既に生成済み（initializer またはグラフ入力・前段
    // ノード出力）の名前集合を逐次拡張しながら、各ノードの入力がすべて既知か
    // 確認する。ONNX は省略可能入力を空文字列で表す規約のためスキップする。
    let mut known: HashSet<String> = initializers.keys().cloned().collect();
    // `g.input` 内部の重複のみを検出するための専用集合。`known` には
    // initializer 名が既に入っているため `known.insert` の戻り値だけでは
    // 「initializer 名との正当な重複（pre-IR-4）」と「`g.input` 内部の
    // 不正な重複」を区別できない。分離した集合で `g.input` 内部の重複のみを
    // 明示的に拒否する（no-silent-skip 契約。レビュー指摘 #77）。
    let mut seen_inputs: HashSet<&str> = HashSet::new();
    for inp in &g.input {
        if !seen_inputs.insert(inp.name.as_str()) {
            return Err(GraphError::DuplicateInputName {
                tensor_name: inp.name.clone(),
            });
        }
        known.insert(inp.name.clone());
    }
    for node in &g.node {
        for input in &node.input {
            if input.is_empty() {
                continue;
            }
            if !known.contains(input) {
                return Err(GraphError::NotTopologicallySorted {
                    node_name: node.name.clone(),
                    missing_input: input.clone(),
                });
            }
        }
        for output in &node.output {
            // ONNX は入力側と同様に省略可能出力も空文字列で表す規約
            // （Dropout の mask・MaxPool の Indices・LSTM/GRU の Y_h/Y_c 等）。
            // 複数ノードがそれぞれ末尾出力を省略した正当なモデルを
            // `DuplicateOutputName { tensor_name: "" }` として誤拒否しないよう、
            // 直前のノード入力側の空文字列スキップと対称に扱う。
            if output.is_empty() {
                continue;
            }
            // `HashSet::insert` は既存要素があっても `false` を返すのみで無言で
            // 何もしない。戻り値を捨てると「2 ノードが同じ出力名を生成」「ノード
            // 出力が既存 initializer / グラフ入力名を無言でシャドウ」を見逃す
            // （`DuplicateInitializerName` と同一の欠陥クラス。レビュー指摘）ため、
            // 挿入前に既知集合との衝突を明示的に拒否する。
            if !known.insert(output.clone()) {
                return Err(GraphError::DuplicateOutputName {
                    node_name: node.name.clone(),
                    tensor_name: output.clone(),
                });
            }
        }
    }

    // グラフ出力（`GraphProto.output`）が initializer／グラフ入力／いずれかの
    // ノード出力によって実際に生成されているかを検証する。ここを素通りすると、
    // モジュール冒頭のドキュメントコメントが謳う「後段は『グラフは既に妥当で
    // ある』前提で実装できる」という契約が破れ、存在しないテンソル名の要求が
    // #78 のインタープリタまで漏れてしまう（レビュー指摘）。
    // `g.output` 内部の重複のみを検出するための専用集合。`seen_inputs` と対称の
    // 理由（Bugbot 指摘）: `known` には initializer・グラフ入力・ノード出力の
    // 名前が既に入っており、`known.contains` だけでは重複検出ができないため
    // 分離した集合で `g.output` 内部の重複のみを明示的に拒否する。
    let mut seen_outputs: HashSet<&str> = HashSet::new();
    for output in &g.output {
        if !known.contains(&output.name) {
            return Err(GraphError::UnknownGraphOutput {
                tensor_name: output.name.clone(),
            });
        }
        if !seen_outputs.insert(output.name.as_str()) {
            return Err(GraphError::DuplicateGraphOutputName {
                tensor_name: output.name.clone(),
            });
        }
    }

    Ok(Graph {
        nodes: g.node.clone(),
        initializers,
        inputs: g.input.iter().map(|v| v.name.clone()).collect(),
        outputs: g.output.iter().map(|v| v.name.clone()).collect(),
    })
}
