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

/// 読み込みを許容するファイルサイズ上限（1 GiB）。
///
/// `.onnx`／`input_0.pb`／`output_0.pb` は本 example の引数で指定される外部
/// ファイルであり、prost の decode（`TensorProto::decode`／`decode_model`）は
/// 長さ区切りフィールドの分だけ確保してからパースする。細工・破損した巨大
/// ファイルを検証前に丸ごと `std::fs::read` するとメモリ枯渇につながるため
/// （A03。`security.md`・`docs/facade-onnx-import-exposure-decision.md` と
/// 同じ脅威モデル）、読み込み前に `std::fs::metadata` でサイズを検査してから
/// 読み込む（fail-closed）。Model Zoo の実モデル（resnet50-v1-12 で
/// 約 100MB）は十分下回る値として 1 GiB を採用した。
const MAX_READ_BYTES: u64 = 1024 * 1024 * 1024;

/// サイズ上限を検査してからファイル全体を読み込む（A03 対策の共通経路）。
/// `.onnx`・`input_0.pb`・`output_0.pb` の 3 箇所すべてがこの関数を経由する。
///
/// `std::fs::metadata(path)` でサイズ検査した後に `std::fs::read(path)` で
/// パスを再度開くと、両呼び出しの間にファイル（または symlink 先）を
/// 差し替えられて上限検査を迂回される TOCTOU が生じる（codex-review 指摘・
/// PR #2225）。検査対象と読み込み対象を同一の `File` ハンドルに固定するため、
/// 一度だけ `open` し、そのハンドルに対して `metadata` 取得と
/// `MAX_READ_BYTES + 1` バイトまでの読み込みを行う。実読込量が上限を
/// 超えた場合は fail-closed で拒否する（`len` 事前検査と実読込量検査の
/// 二重防御。事前の `len` が小さくても読み込み中にファイルが伸長される
/// 可能性への保険を兼ねる）。
fn read_file_bounded(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| format!("ファイルオープン失敗: {} ({e})", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("メタデータ取得失敗: {} ({e})", path.display()))?;
    let len = metadata.len();
    if len > MAX_READ_BYTES {
        return Err(format!(
            "ファイルサイズ上限超過: {} ({len} bytes > {MAX_READ_BYTES} bytes)",
            path.display()
        )
        .into());
    }

    // 上限を 1 バイトでも超えたら検出できるよう MAX_READ_BYTES + 1 まで読む。
    let mut buf = Vec::new();
    let read_len = file
        .take(MAX_READ_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("読み込み失敗: {} ({e})", path.display()))?;
    if read_len as u64 > MAX_READ_BYTES {
        return Err(format!(
            "ファイルサイズ上限超過（読み込み時検査）: {} (> {MAX_READ_BYTES} bytes)",
            path.display()
        )
        .into());
    }
    Ok(buf)
}

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
    let bytes = read_file_bounded(&onnx_path)?;
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

    let (_input_name, input_tensor) = decode_f32_tensor_pb(&read_file_bounded(&input_path)?)?;
    let (_output_name, expected_tensor) = decode_f32_tensor_pb(&read_file_bounded(&output_path)?)?;

    let input_graph_name = graph.inputs.first().cloned().ok_or("graph.inputs が空")?;
    let output_graph_name = graph.outputs.first().cloned().ok_or("graph.outputs が空")?;

    let mut feeds: HashMap<String, Value> = HashMap::new();
    feeds.insert(input_graph_name, Value::F32(input_tensor));

    match interp::run(&graph, feeds) {
        Ok(result) => {
            let value = result
                .get(&output_graph_name)
                .ok_or_else(|| format!("run 結果に output '{output_graph_name}' が無い"))?;
            match value {
                Value::F32(actual) => {
                    // shape・非有限値・閾値超過はすべて Err で返す（`assert_req7`
                    // 〈tests/model_zoo_parity.rs〉と同じ fail-closed 判定を
                    // 切り出したもの。表示のみで `Ok(())` を返して REQ-7 判定を
                    // 素通りさせない）。shape は要素数（slice 長）一致だけでは
                    // 検出できない不一致（例: 期待 [1, 10] に対し実出力 [10]）を
                    // 見逃さないよう、slice 化する前に Tensor の shape 自体を
                    // 比較する（P2 指摘・Bugbot Low 指摘対応）。
                    let max_rel_err = check_req7(actual, &expected_tensor)?;
                    println!("run: Ok, max_rel_err={max_rel_err}");
                }
                // F32 以外の出力は REQ-7 判定不能のため Err（成功終了扱いにしない）。
                other => return Err(format!("run: Ok だが F32 以外の出力: {other:?}").into()),
            }
        }
        // UnsupportedOp は本 example の想定内の到達点（tests/model_zoo_parity.rs
        // の `RunExpectation::UnsupportedOp` と同じ調査結果）であり Err にしない。
        Err(InterpError::UnsupportedOp(op)) => println!("run: UnsupportedOp({op:?})"),
        // それ以外の実行エラーは表示のみで `Ok(())` を返さず Err として伝播する
        // （非 0 終了コードへ反映。P2 指摘: 実行エラーが成功終了扱いになっていた）。
        Err(e) => return Err(format!("run: Err({e})").into()),
    }

    Ok(())
}

/// REQ-7 事前固定式（`abs_err / (|ref| + 1e-6) <= 1e-3`）で全要素を fail-closed
/// 判定する（`tests/model_zoo_parity.rs::assert_req7` と同型の判定ロジック。
/// テスト側は `panic!` で失敗を示すが、本 example はプロセス終了コードへ
/// 反映するため `Result` で返す）。
///
/// shape 不一致（rank・各軸長を含む完全一致）・非有限な `rel_err`（NaN・inf
/// 入力を含む）・閾値超過はすべて `Err` として扱い、表示のみで `Ok` に丸めない
/// （REQ-7・P2 指摘対応）。
///
/// `actual.shape() == expected.shape()` を slice 化より先に検査する。要素数
/// （slice 長）のみの比較では、rank や各軸長が異なるが総要素数が一致する
/// 誤出力（例: 期待 `[1, 10]` に対し実出力 `[10]`、または ONNX Model Zoo の
/// 参照出力に典型的な `[1, 1000, 1, 1]` のような rank を持つ出力で軸が入れ替わる
/// 誤出力）を見逃す（codex-review P2・Cursor Bugbot Low 指摘対応）。
fn check_req7(actual: &Tensor<f32>, expected: &Tensor<f32>) -> Result<f32, String> {
    if actual.shape() != expected.shape() {
        return Err(format!(
            "shape 不一致: actual={:?} expected={:?}",
            actual.shape(),
            expected.shape()
        ));
    }
    let actual = actual.as_slice().ok_or("actual as_slice 失敗")?;
    let expected = expected.as_slice().ok_or("expected as_slice 失敗")?;
    let mut fail_count = 0usize;
    let mut max_rel_err = 0.0f32;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let rel_err = (a - e).abs() / (e.abs() + 1e-6);
        // rel_err が非有限（NaN・inf）になる場合、`rel_err > 1e-3` は false 判定
        // になり見逃されうる。fail-closed のため非有限は無条件で fail 扱いにし、
        // max_rel_err も INFINITY として表示に反映する（テスト側 assert_req7 と
        // 同じ扱い）。
        if rel_err.is_finite() {
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        } else {
            max_rel_err = f32::INFINITY;
        }
        if !rel_err.is_finite() || rel_err > 1e-3 {
            fail_count += 1;
        }
    }
    if fail_count > 0 {
        Err(format!(
            "REQ-7 判定式 fail: fail_count={fail_count} max_rel_err={max_rel_err}"
        ))
    } else {
        Ok(max_rel_err)
    }
}
