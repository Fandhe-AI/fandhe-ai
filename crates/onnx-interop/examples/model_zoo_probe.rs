//! ONNX Model Zoo（onnx/models）が公開する第三者モデルに対する import 実証
//! example（イシュー #2081・REQ-7）。
//!
//! `crates/onnx-interop/tests/model_zoo_parity.rs`（tier B ignored テスト）が
//! 検証する同じ経路（`decode → build_graph → run`）を、CLI として単体実行できる
//! 形で切り出したもの。CI では実行しない（第三者モデルは `#[ignore]`
//! テストと同じく `ONNX_INTEROP_MODEL_ZOO_DIR` 配下の非コミット fixture を
//! 前提とするため）。用途は 2 つ:
//!
//! 1. Model Zoo モデルの decode／`build_graph` 到達性・op ヒストグラムを
//!    人手で確認する調査ツール
//! 2. `docs/onnx-model-zoo-parity.md` の期待値表（`RunExpectation`）を
//!    実装 HEAD に合わせて再プローブするための再現手順
//!
//! 引数にはモデルディレクトリ（`<dir>/<basename>.onnx` と
//! `<dir>/test_data_set_0/` を含む Model Zoo 配布形式そのまま）を渡す。
//! 出力: opset・node／initializer 数・入出力名・op ヒストグラム・実行結果
//! （REQ-7 事前固定式 `abs_err / (|ref| + 1e-6) <= 1e-3` による max 相対誤差、
//! または `UnsupportedOp` エラー）。
//!
//! 本番経路（`fandhe_ai_onnx_interop` の公開 API）ではないため `unwrap` は
//! 使わず `Result` 経由でプロセス終了コードへ反映する。

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, InterpError, Value};
use fandhe_ai_onnx_interop::onnx::proto::{self, TensorProto, data_type};
use fandhe_ai_tensor_core::Tensor;
use prost::Message;

/// 引数で受け取ったモデルディレクトリから `<basename>.onnx` を探す。
/// Model Zoo tar.gz 展開形式（`<name>/<name>.onnx`）を前提に、ディレクトリ名
/// と同じ basename を優先し、無ければディレクトリ直下の唯一の `.onnx` を使う。
fn locate_onnx_file(dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(base) = dir.file_name() {
        let candidate = dir.join(format!("{}.onnx", base.to_string_lossy()));
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    let mut found: Option<PathBuf> = None;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "onnx") {
            if found.is_some() {
                return Err(format!(
                    "複数の .onnx ファイルが見つかり basename を特定できない: {}",
                    dir.display()
                )
                .into());
            }
            found = Some(path);
        }
    }
    found.ok_or_else(|| format!(".onnx ファイルが見つからない: {}", dir.display()).into())
}

/// `TensorProto`（FLOAT 型限定）を `Tensor<f32>` へ復号する。
///
/// `graph::decode_tensor` は `pub(crate)` のため crate 外（本 example は
/// バイナリターゲットであり crate 外扱い）から直接呼べない。ここでは
/// 同モジュールの検証順序（dims 非負・要素数 `checked_mul`・`raw_data` の
/// バイト長完全一致を先に検査してから数値へ変換）を鏡写しにする（A03）。
fn decode_f32_tensor_pb(bytes: &[u8]) -> Result<(String, Tensor<f32>), Box<dyn Error>> {
    let t = TensorProto::decode(bytes)?;
    if t.data_type != data_type::FLOAT {
        return Err(format!("FLOAT 以外の data_type: {}", t.data_type).into());
    }
    let mut expected_elements: usize = 1;
    for &d in &t.dims {
        if d < 0 {
            return Err(format!("負の dim: {d}").into());
        }
        expected_elements = expected_elements
            .checked_mul(d as usize)
            .ok_or("要素数オーバーフロー")?;
    }
    let expected_bytes = expected_elements
        .checked_mul(4)
        .ok_or("バイト数オーバーフロー")?;
    if t.raw_data.len() != expected_bytes {
        return Err(format!(
            "raw_data バイト長不一致: expected={expected_bytes} actual={}",
            t.raw_data.len()
        )
        .into());
    }
    let data: Vec<f32> = t
        .raw_data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let tensor = Tensor::<f32>::new(data, &shape)?;
    Ok((t.name, tensor))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args();
    let _argv0 = args.next();
    let dir = match args.next() {
        Some(a) => PathBuf::from(a),
        None => {
            eprintln!(
                "使い方: cargo run -p fandhe-ai-onnx-interop --example model_zoo_probe -- <モデルディレクトリ>"
            );
            std::process::exit(2);
        }
    };

    let onnx_path = locate_onnx_file(&dir)?;
    let bytes = std::fs::read(&onnx_path)?;
    let model = proto::decode_model(&bytes)?;
    let graph = build_graph(&model)?;

    println!("model: {}", onnx_path.display());
    println!(
        "opset_import: {:?}",
        model
            .opset_import
            .iter()
            .map(|o| (o.domain.clone(), o.version))
            .collect::<Vec<_>>()
    );
    println!("node_count: {}", graph.nodes.len());
    println!("initializer_count: {}", graph.initializers.len());
    println!("inputs: {:?}", graph.inputs);
    println!("outputs: {:?}", graph.outputs);

    let mut histogram: BTreeMap<String, usize> = BTreeMap::new();
    for node in &graph.nodes {
        *histogram.entry(node.op_type.clone()).or_insert(0) += 1;
    }
    println!("op_histogram: {histogram:?}");

    let test_dir = dir.join("test_data_set_0");
    let input_path = test_dir.join("input_0.pb");
    let output_path = test_dir.join("output_0.pb");
    if !input_path.is_file() || !output_path.is_file() {
        println!("test_data_set_0 が無いため run は実行しない");
        return Ok(());
    }

    let (_input_name, input_tensor) = decode_f32_tensor_pb(&std::fs::read(&input_path)?)?;
    let (_output_name, expected_tensor) = decode_f32_tensor_pb(&std::fs::read(&output_path)?)?;

    let input_graph_name = graph.inputs.first().cloned().ok_or("graph.inputs が空")?;
    let output_graph_name = graph.outputs.first().cloned().ok_or("graph.outputs が空")?;

    let mut feeds: HashMap<String, Value> = HashMap::new();
    feeds.insert(input_graph_name, Value::F32(input_tensor));

    match interp::run(&graph, feeds) {
        Ok(result) => match &result[&output_graph_name] {
            Value::F32(actual) => {
                let actual_slice = actual.as_slice().ok_or("actual as_slice 失敗")?;
                let expected_slice = expected_tensor.as_slice().ok_or("expected as_slice 失敗")?;
                if actual_slice.len() != expected_slice.len() {
                    println!(
                        "shape 不一致: actual_len={} expected_len={}",
                        actual_slice.len(),
                        expected_slice.len()
                    );
                    return Ok(());
                }
                let mut max_rel_err = 0.0f32;
                for (&a, &e) in actual_slice.iter().zip(expected_slice.iter()) {
                    let rel_err = (a - e).abs() / (e.abs() + 1e-6);
                    if rel_err > max_rel_err {
                        max_rel_err = rel_err;
                    }
                }
                println!("run: Ok, max_rel_err={max_rel_err}");
            }
            other => println!("run: Ok だが F32 以外の出力: {other:?}"),
        },
        Err(InterpError::UnsupportedOp(op)) => println!("run: UnsupportedOp({op:?})"),
        Err(e) => println!("run: Err({e})"),
    }

    Ok(())
}
