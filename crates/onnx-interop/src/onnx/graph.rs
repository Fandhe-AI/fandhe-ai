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
    /// 実データ（`raw_data` / `float_data` / `int64_data`）の長さが `dims` から
    /// 導出した期待要素数と一致しない。
    DataLenMismatch {
        tensor_name: String,
        expected_elements: usize,
        actual_elements: usize,
    },
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
        }
    }
}
impl std::error::Error for GraphError {}

/// 生の initializer / 定数テンソル（`tensor-core` の型へのマッピング前）。
#[derive(Clone, Debug, PartialEq)]
pub enum RawTensor {
    F32 { data: Vec<f32>, shape: Vec<i64> },
    I64 { data: Vec<i64>, shape: Vec<i64> },
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
/// 未対応 `data_type` は `UnknownDataType` で拒否する（無言 skip 禁止）。
fn decode_tensor(t: &TensorProto) -> Result<RawTensor, GraphError> {
    let shape = t.dims.clone();
    let expected_elements = element_count(&t.name, &shape)?;

    match t.data_type {
        super::proto::data_type::FLOAT => {
            if !t.float_data.is_empty() {
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
                // raw_data が空でも「dims=[0] 等の空テンソル」と「data フィールドが
                // 一つも埋まっていない不正入力（例: external_data 参照だが本クレートは
                // TensorProto.data_location 等を意図的に未定義。プロトコル上は未宣言
                // フィールドとして無言でスキップされてしまう）」を区別できないため、
                // 必ず expected_bytes との一致検査を通す（× 0 要素なら 0 バイト一致で
                // 素通りする）。expected_elements の乗算自体もオーバーフローしうるため
                // checked_mul で拒否する（巨大 dims による資源枯渇攻撃を弾く。A03）。
                let expected_bytes = expected_elements.checked_mul(4).ok_or_else(|| {
                    GraphError::ElementCountOverflow {
                        tensor_name: t.name.clone(),
                    }
                })?;
                if t.raw_data.len() != expected_bytes {
                    return Err(GraphError::DataLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_elements,
                        actual_elements: t.raw_data.len() / 4,
                    });
                }
                let data: Vec<f32> = t
                    .raw_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Ok(RawTensor::F32 { data, shape })
            }
        }
        super::proto::data_type::INT64 => {
            if !t.int64_data.is_empty() {
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
                    return Err(GraphError::DataLenMismatch {
                        tensor_name: t.name.clone(),
                        expected_elements,
                        actual_elements: t.raw_data.len() / 8,
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
            }
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

    let mut initializers = HashMap::new();
    for init in &g.initializer {
        initializers.insert(init.name.clone(), decode_tensor(init)?);
    }

    // トポロジカル順の検証: 既に生成済み（initializer またはグラフ入力・前段
    // ノード出力）の名前集合を逐次拡張しながら、各ノードの入力がすべて既知か
    // 確認する。ONNX は省略可能入力を空文字列で表す規約のためスキップする。
    let mut known: HashSet<String> = initializers.keys().cloned().collect();
    for inp in &g.input {
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
            known.insert(output.clone());
        }
    }

    Ok(Graph {
        nodes: g.node.clone(),
        initializers,
        inputs: g.input.iter().map(|v| v.name.clone()).collect(),
        outputs: g.output.iter().map(|v| v.name.clone()).collect(),
    })
}
