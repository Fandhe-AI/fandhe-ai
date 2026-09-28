//! テンソル本体（宣言長・dims の積で巨大化しうるバッファ）の失敗可能確保
//! ヘルパ群（PR #2348 codex P0 是正。security.md A04）。
//!
//! ## 役割・呼び出し文脈
//!
//! external data（`external_data`）由来のテンソルは `.onnx` 本体が小さくても
//! `max_total_bytes`（既定 64 GiB）まで宣言でき、利用可能メモリを超えうる。
//! `vec![..; n]`・`Vec::with_capacity(n)`・`collect`・`clone` のような無条件
//! 確保は失敗時に `handle_alloc_error` でプロセスを abort させるため、テンソル
//! 本体を扱う次の経路はすべて本モジュールの `Vec::try_reserve_exact` ベースの
//! ヘルパを経由し、失敗を [`AllocFailure`] として呼び出し元の型付きエラーへ
//! 写像する（`docs/onnx-external-data-decision.md` 4.3 節の洗い出し表）:
//!
//! - 読み込み経路: `external_data::load` の区間バッファ・external 由来
//!   initializer の復号（[`try_alloc_vec`]・[`decode_le_into`]。
//!   `ExternalDataError::AllocationFailed` へ写像）
//! - 実行経路: `interp::run` が実行ごとに initializer を実行時値へ複製する
//!   処理（[`try_clone_slice`]）・`Constant` 属性テンソルの復号
//!   （[`try_decode_tensor`]）。`onnx::autograd` の同じ 2 箇所も同様
//!   （`InterpError::AllocationFailed` へ写像）
//! - export 経路: `export::build_model_proto` のノード列複製
//!   （[`try_clone_nodes`]）・initializer のバイト列化
//!   （`ExportError::AllocationFailed` へ写像）
//!
//! 演算カーネル（`ops::*`・`interp_device`・`interp_ext`）の出力確保は一般の
//! 推論メモリであり本モジュールの対象外（同表）。
//!
//! ## 数値不変
//!
//! 本モジュールのヘルパは確保方式だけを変え、書き込む値は無条件確保版と
//! 同一（`clone` と同じ要素列・`graph::decode_tensor` と同じリトル
//! エンディアン変換）。[`try_decode_tensor`] と `graph::decode_tensor` の
//! 出力・エラー variant の一致は単体テストで固定する（`decode_tensor` 自体は
//! イシュー #2347 A6 により変更しない）。

use super::graph::{self, GraphError, RawTensor};
use super::proto::{AttributeProto, NodeProto, TensorProto, data_type};

/// 失敗可能確保の失敗（要求の診断情報）。呼び出し元のモジュールが自分の
/// エラー型（`ExternalDataError::AllocationFailed`・`InterpError::
/// AllocationFailed`・`ExportError::AllocationFailed`）へ写像する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AllocFailure {
    /// 確保しようとしたテンソルの名前（診断用。非信頼入力由来のため空文字列
    /// もありうる）。
    pub(crate) tensor_name: String,
    /// 要求バイト数（`u64` で飽和計算）。
    pub(crate) bytes: u64,
}

/// 要素数 `count` の空 `Vec<T>`（容量ちょうど・長さ 0）を
/// `try_reserve_exact` で用意する。`count * size_of::<T>()` が `isize::MAX`
/// を超える場合、`try_reserve_exact` はアロケータを呼ばずに
/// `CapacityOverflow` で失敗するため、巨大な宣言長も確保を試みる前に拒否
/// される。
pub(crate) fn try_alloc_vec<T>(tensor_name: &str, count: usize) -> Result<Vec<T>, AllocFailure> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(count).map_err(|_| AllocFailure {
        tensor_name: tensor_name.to_string(),
        bytes: (count as u64).saturating_mul(std::mem::size_of::<T>() as u64),
    })?;
    Ok(v)
}

/// `src` の失敗可能な複製（`src.to_vec()` と同じ要素列）。容量は
/// `src.len()` ちょうどで確保済みのため `extend_from_slice` は再確保しない。
pub(crate) fn try_clone_slice<T: Clone>(
    tensor_name: &str,
    src: &[T],
) -> Result<Vec<T>, AllocFailure> {
    let mut out = try_alloc_vec::<T>(tensor_name, src.len())?;
    out.extend_from_slice(src);
    Ok(out)
}

/// `raw` を `N` バイトずつのリトルエンディアン要素として `conv` で変換し、
/// [`try_alloc_vec`] で確保した Vec へ詰める。**前提**: 呼び出し元が
/// `raw.len()` が `N` の倍数であることを検査済み（余りは読まない）。
pub(crate) fn decode_le_into<const N: usize, T>(
    tensor_name: &str,
    raw: &[u8],
    conv: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, AllocFailure> {
    let (chunks, _rest) = raw.as_chunks::<N>();
    let mut out = try_alloc_vec::<T>(tensor_name, chunks.len())?;
    // 容量は `chunks.len()` ちょうど確保済みのため、`extend` は再確保しない。
    out.extend(chunks.iter().map(|b| conv(*b)));
    Ok(out)
}

/// [`try_decode_tensor`] のエラー: 形状・データ長の検証エラー
/// （`graph::decode_tensor` と同じ variant）か、確保失敗。
#[derive(Debug, PartialEq)]
pub(crate) enum TryDecodeError {
    Graph(GraphError),
    Alloc(AllocFailure),
}

/// `TensorProto` を `RawTensor` へ復号する `graph::decode_tensor` の
/// 失敗可能確保版（実行経路の `Constant` 属性テンソル用。PR #2348 codex P0
/// 是正）。`alloc_label` は確保失敗時の診断名（`Constant` 属性テンソルは
/// `name` が空のことが多いため呼び出し元がノード名等で補う）。
///
/// 対象は要素 Vec が dims の積（＝ external data の宣言長）に比例して
/// 巨大化しうる「`raw_data` 非空 かつ `data_type` が対応 4 型」の場合のみで、
/// `decode_tensor` と同じ順序（`element_count` → 期待バイト長の
/// `checked_mul` → `raw_data` 長の照合）で検証してから同じリトルエンディ
/// アン変換を失敗可能確保で行う。それ以外（typed data〈`float_data`／
/// `int64_data`〉・空の `raw_data`・未対応 `data_type`）は
/// `decode_tensor` へそのまま委譲する（typed data は external data に
/// ならず `.onnx` 本体の長さで有界。検証ロジックを二重実装する範囲を最小に
/// する）。出力・エラー variant が `decode_tensor` と一致することは単体
/// テスト `try_decode_tensor_matches_decode_tensor` で固定する。
pub(crate) fn try_decode_tensor(
    t: &TensorProto,
    alloc_label: &str,
) -> Result<RawTensor, TryDecodeError> {
    let elem_size: usize = match t.data_type {
        dt if dt == data_type::FLOAT => 4,
        dt if dt == data_type::INT64 => 8,
        dt if dt == data_type::BOOL => 1,
        dt if dt == data_type::FLOAT16 => 2,
        _ => return graph::decode_tensor(t).map_err(TryDecodeError::Graph),
    };
    if t.raw_data.is_empty() {
        return graph::decode_tensor(t).map_err(TryDecodeError::Graph);
    }
    let expected_elements =
        graph::element_count(&t.name, &t.dims).map_err(TryDecodeError::Graph)?;
    let expected_bytes = expected_elements.checked_mul(elem_size).ok_or_else(|| {
        TryDecodeError::Graph(GraphError::ElementCountOverflow {
            tensor_name: t.name.clone(),
        })
    })?;
    if t.raw_data.len() != expected_bytes {
        return Err(TryDecodeError::Graph(GraphError::RawDataByteLenMismatch {
            tensor_name: t.name.clone(),
            expected_bytes,
            actual_bytes: t.raw_data.len(),
        }));
    }
    let shape = t.dims.clone();
    let raw = t.raw_data.as_slice();
    let alloc = TryDecodeError::Alloc;
    Ok(match elem_size {
        4 => RawTensor::F32 {
            data: decode_le_into::<4, f32>(alloc_label, raw, f32::from_le_bytes).map_err(alloc)?,
            shape,
        },
        8 => RawTensor::I64 {
            data: decode_le_into::<8, i64>(alloc_label, raw, i64::from_le_bytes).map_err(alloc)?,
            shape,
        },
        1 => RawTensor::Bool {
            data: decode_le_into::<1, bool>(alloc_label, raw, |b| b[0] != 0).map_err(alloc)?,
            shape,
        },
        _ => RawTensor::F16 {
            data: decode_le_into::<2, half::f16>(alloc_label, raw, half::f16::from_le_bytes)
                .map_err(alloc)?,
            shape,
        },
    })
}

/// ノード属性テンソルの診断名（確保失敗時の `tensor_name`）。テンソル自身の
/// `name` があればそれ、無ければ `"{ノード名}:{属性名}"`（ノード名も空なら
/// `op_type` で代替）とする。読み込み段（`external_data::enumerate_tensors`）
/// が属性テンソルに付ける名前と同じ規則で、同じテンソルの確保失敗が
/// 読み込み・実行・export のどの段で起きても同じ名前で報告される。
pub(crate) fn attr_tensor_label(node: &NodeProto, attr_name: &str, t: &TensorProto) -> String {
    if !t.name.is_empty() {
        return t.name.clone();
    }
    let node_label = if node.name.is_empty() {
        node.op_type.as_str()
    } else {
        node.name.as_str()
    };
    format!("{node_label}:{attr_name}")
}

/// `TensorProto` の失敗可能な複製（`Clone::clone` と同値）。テンソル本体
/// （`raw_data`・`float_data`・`int64_data`）だけを失敗可能確保で複製し、
/// 小さなメタデータ（dims・名前・external data エントリ）は通常の clone。
/// 構造体リテラルを全フィールド列挙で書くため、`TensorProto` にフィールドが
/// 追加されるとコンパイルエラーで気付ける（複製漏れを防ぐ）。
fn try_clone_tensor(t: &TensorProto, label: &str) -> Result<TensorProto, AllocFailure> {
    Ok(TensorProto {
        dims: t.dims.clone(),
        data_type: t.data_type,
        float_data: try_clone_slice(label, &t.float_data)?,
        int64_data: try_clone_slice(label, &t.int64_data)?,
        name: t.name.clone(),
        raw_data: try_clone_slice(label, &t.raw_data)?,
        external_data: t.external_data.clone(),
        data_location: t.data_location,
    })
}

/// `AttributeProto` の失敗可能な複製（`Clone::clone` と同値）。`..a.clone()`
/// を使うと `t` の `raw_data` も無条件に複製されるため、全フィールドを
/// 列挙する（[`try_clone_tensor`] と同じ理由でフィールド追加も検出できる）。
fn try_clone_attribute(
    node: &NodeProto,
    a: &AttributeProto,
) -> Result<AttributeProto, AllocFailure> {
    let t = match a.t.as_ref() {
        Some(t) => Some(try_clone_tensor(t, &attr_tensor_label(node, &a.name, t))?),
        None => None,
    };
    Ok(AttributeProto {
        name: a.name.clone(),
        f: a.f,
        i: a.i,
        s: a.s.clone(),
        t,
        floats: a.floats.clone(),
        ints: a.ints.clone(),
        r#type: a.r#type,
    })
}

/// `NodeProto` 列の失敗可能な複製（`nodes.to_vec()` と同値。
/// `export::build_model_proto` 用）。external data から inline 化した
/// `Constant` 属性テンソルの `raw_data` は宣言長で巨大化しうるため、属性
/// テンソル本体だけを失敗可能確保で複製する。ノード数・名前・`ints`／
/// `floats` 等は `.onnx` 本体の長さで有界のため通常の clone。
pub(crate) fn try_clone_nodes(nodes: &[NodeProto]) -> Result<Vec<NodeProto>, AllocFailure> {
    let mut out = Vec::with_capacity(nodes.len());
    for n in nodes {
        let mut attribute = Vec::with_capacity(n.attribute.len());
        for a in &n.attribute {
            attribute.push(try_clone_attribute(n, a)?);
        }
        out.push(NodeProto {
            input: n.input.clone(),
            output: n.output.clone(),
            name: n.name.clone(),
            op_type: n.op_type.clone(),
            attribute,
            domain: n.domain.clone(),
        });
    }
    Ok(out)
}

/// 確保失敗の注入（`usize::MAX`・`isize::MAX` 超の要求）と、無条件確保版との
/// 出力一致を固定する単体テスト。実確保を伴わないため全プラットフォームで
/// 実行する。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::proto::StringStringEntryProto;

    fn tensor(name: &str, dt: i32, dims: Vec<i64>, raw: Vec<u8>) -> TensorProto {
        TensorProto {
            dims,
            data_type: dt,
            float_data: Vec::new(),
            int64_data: Vec::new(),
            name: name.to_string(),
            raw_data: raw,
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    #[test]
    fn try_alloc_vec_reports_unallocatable_request() {
        let err = try_alloc_vec::<u8>("t", usize::MAX).unwrap_err();
        assert_eq!(
            err,
            AllocFailure {
                tensor_name: "t".to_string(),
                bytes: usize::MAX as u64,
            }
        );
        // 要素サイズ 4 で isize::MAX バイトを超える要求（確保を試みる前に
        // CapacityOverflow で拒否される）。
        let err = try_alloc_vec::<f32>("w", (isize::MAX as usize) / 4 + 1).unwrap_err();
        assert_eq!(err.tensor_name, "w");
        assert_eq!(err.bytes, ((isize::MAX as u64) / 4 + 1) * 4);
    }

    #[test]
    fn try_clone_slice_matches_to_vec_with_exact_capacity() {
        let src = [1.5f32, -0.0, f32::NAN, 3.25];
        let out = try_clone_slice("x", &src).unwrap();
        assert_eq!(out.len(), src.len());
        assert_eq!(out.capacity(), src.len());
        for (a, b) in out.iter().zip(src.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(try_clone_slice::<u8>("e", &[]).unwrap().is_empty());
    }

    /// `try_decode_tensor` が `decode_tensor` と同じ `Ok`／`Err` を返すこと
    /// （4 dtype の raw_data・typed data・空テンソル・検証エラー各種）。
    #[test]
    fn try_decode_tensor_matches_decode_tensor() {
        let mut f32_raw = Vec::new();
        for v in [1.0f32, -2.5, f32::INFINITY] {
            f32_raw.extend_from_slice(&v.to_le_bytes());
        }
        let mut i64_raw = Vec::new();
        for v in [i64::MIN, 0, 7] {
            i64_raw.extend_from_slice(&v.to_le_bytes());
        }
        let mut f16_raw = Vec::new();
        for v in [half::f16::from_f32(0.5), half::f16::NEG_INFINITY] {
            f16_raw.extend_from_slice(&v.to_le_bytes());
        }
        let mut typed_f32 = tensor("tf", data_type::FLOAT, vec![2], Vec::new());
        typed_f32.float_data = vec![1.0, 2.0];
        let mut typed_i64 = tensor("ti", data_type::INT64, vec![2], Vec::new());
        typed_i64.int64_data = vec![3, 4];
        let mut both = tensor(
            "both",
            data_type::FLOAT,
            vec![1],
            9f32.to_le_bytes().to_vec(),
        );
        both.float_data = vec![1.0];

        let cases = vec![
            tensor("f", data_type::FLOAT, vec![3], f32_raw),
            tensor("i", data_type::INT64, vec![1, 3], i64_raw),
            tensor("b", data_type::BOOL, vec![4], vec![0, 1, 2, 255]),
            tensor("h", data_type::FLOAT16, vec![2], f16_raw),
            typed_f32,
            typed_i64,
            both,
            tensor("empty", data_type::FLOAT, vec![0], Vec::new()),
            tensor("empty_b", data_type::BOOL, vec![0], Vec::new()),
            // 検証エラー: raw_data 長の不一致・余りバイト
            tensor("short", data_type::FLOAT, vec![2], vec![0; 4]),
            tensor("odd", data_type::INT64, vec![1], vec![0; 9]),
            tensor("bool_short", data_type::BOOL, vec![3], vec![1]),
            // 検証エラー: 空 raw（data が一つも埋まっていない）
            tensor("missing", data_type::FLOAT, vec![2], Vec::new()),
            tensor("missing_b", data_type::BOOL, vec![2], Vec::new()),
            // 検証エラー: 負の dim・要素数の乗算オーバーフロー
            tensor("neg", data_type::FLOAT, vec![-1], vec![0; 4]),
            tensor(
                "ovf",
                data_type::FLOAT,
                vec![i64::MAX, i64::MAX],
                vec![0; 4],
            ),
            tensor(
                "ovf_bytes",
                data_type::INT64,
                vec![(usize::MAX / 8 + 1) as i64],
                vec![0; 8],
            ),
            // 未対応 dtype
            tensor("u8", 2, vec![1], vec![0]),
        ];
        for t in &cases {
            let expected = graph::decode_tensor(t);
            let actual = try_decode_tensor(t, "label");
            match (&expected, &actual) {
                (Ok(e), Ok(a)) => assert_raw_bits_eq(&t.name, e, a),
                (Err(e), Err(TryDecodeError::Graph(a))) => {
                    assert_eq!(
                        e, a,
                        "{}: エラー variant が decode_tensor と一致しない",
                        t.name
                    )
                }
                _ => panic!(
                    "{}: decode_tensor={expected:?} try_decode_tensor={actual:?}",
                    t.name
                ),
            }
        }
    }

    /// NaN を含む f32／f16 も bit 単位で比較する（`PartialEq` は NaN で偽）。
    fn assert_raw_bits_eq(name: &str, e: &RawTensor, a: &RawTensor) {
        match (e, a) {
            (
                RawTensor::F32 {
                    data: d1,
                    shape: s1,
                },
                RawTensor::F32 {
                    data: d2,
                    shape: s2,
                },
            ) => {
                assert_eq!(s1, s2, "{name}");
                let b1: Vec<u32> = d1.iter().map(|v| v.to_bits()).collect();
                let b2: Vec<u32> = d2.iter().map(|v| v.to_bits()).collect();
                assert_eq!(b1, b2, "{name}");
            }
            (
                RawTensor::F16 {
                    data: d1,
                    shape: s1,
                },
                RawTensor::F16 {
                    data: d2,
                    shape: s2,
                },
            ) => {
                assert_eq!(s1, s2, "{name}");
                let b1: Vec<u16> = d1.iter().map(|v| v.to_bits()).collect();
                let b2: Vec<u16> = d2.iter().map(|v| v.to_bits()).collect();
                assert_eq!(b1, b2, "{name}");
            }
            _ => assert_eq!(e, a, "{name}"),
        }
    }

    /// 属性テンソルの診断名は `external_data::enumerate_tensors` と同じ規則。
    #[test]
    fn attr_tensor_label_matches_loader_naming() {
        let t = tensor("", data_type::FLOAT, vec![0], Vec::new());
        let mut node = NodeProto {
            input: Vec::new(),
            output: vec!["y".to_string()],
            name: "n0".to_string(),
            op_type: "Constant".to_string(),
            attribute: Vec::new(),
            domain: String::new(),
        };
        assert_eq!(attr_tensor_label(&node, "value", &t), "n0:value");
        node.name.clear();
        assert_eq!(attr_tensor_label(&node, "value", &t), "Constant:value");
        let named = tensor("c", data_type::FLOAT, vec![0], Vec::new());
        assert_eq!(attr_tensor_label(&node, "value", &named), "c");
    }

    #[test]
    fn try_clone_nodes_equals_clone() {
        let mut t = tensor("", data_type::FLOAT, vec![2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        t.float_data = vec![0.5];
        t.int64_data = vec![9];
        t.external_data = vec![StringStringEntryProto {
            key: "k".to_string(),
            value: "v".to_string(),
        }];
        t.data_location = 1;
        let nodes = vec![
            NodeProto {
                input: vec!["a".to_string(), String::new()],
                output: vec!["y".to_string()],
                name: "n0".to_string(),
                op_type: "Constant".to_string(),
                attribute: vec![AttributeProto {
                    name: "value".to_string(),
                    f: 1.25,
                    i: -3,
                    s: b"s".to_vec(),
                    t: Some(t),
                    floats: vec![2.0],
                    ints: vec![4, 5],
                    r#type: 4,
                }],
                domain: "d".to_string(),
            },
            NodeProto {
                input: vec!["y".to_string()],
                output: vec!["z".to_string()],
                name: "n1".to_string(),
                op_type: "Relu".to_string(),
                attribute: Vec::new(),
                domain: String::new(),
            },
        ];
        assert_eq!(try_clone_nodes(&nodes).unwrap(), nodes.to_vec());
    }
}
