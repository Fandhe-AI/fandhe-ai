//! イシュー #2081（REQ-7）: ONNX Model Zoo（`onnx/models`。第三者公開モデル）
//! に対する import・実行の到達性検証。
//!
//! 自前生成 fixture（`model.onnx`・`slice_repro.onnx`・非コミット
//! `transformer.onnx`）はこれまで decode／`build_graph`／`run` の各段を
//! 検証してきたが、第三者が公開する実モデルに対する到達性は未立証だった
//! （REQ-7 の第三者 fixture による裏付け）。本ファイルは
//! `tests/fixtures/model-zoo/` 配下のモデルを `decode → build_graph → run`
//! の全経路で検証し、期待値表（[`RunExpectation`]）に対して fail-closed に
//! 固定する。
//!
//! ## HEAD 時点の位置づけ（green parity ではない）
//!
//! 選定モデル（mnist-12・squeezenet1.0-12・mobilenetv2-12・resnet50-v1-12）は
//! いずれも `Conv`（未対応 op。追跡先はイシュー #2199。`auto_pad`／group
//! conv は #2199 の受け入れ条件に含まれず別途追跡が必要 —
//! `docs/onnx-model-zoo-parity.md` §5）で `run` が止まるため、HEAD 時点では
//! 1 件も end-to-end 実行できない。本ファイルの成果は「ハーネス・被覆台帳・
//! fail-closed な期待値表」であり、未対応 op が実装され `run` が先へ進んだ
//! 場合は該当エントリの [`RunExpectation`] を更新する
//! （`docs/onnx-model-zoo-parity.md` §6 の期待値反転手順）。
//!
//! ## 判定式についての注記（REQ-2 との混同禁止）
//!
//! `Parity` 判定は `tests/onnx_interp.rs` 等と同じ REQ-7 事前固定基準
//! `abs_err / (|ref| + 1e-6) <= 1e-3` を用いる。`.claude/rules/coding-rust.md`
//! の REQ-2 バックエンド間数値一致 OR 複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）とは別指標であり、両者を混同してどちらかを緩和しない。
//! ONNX インタープリタはホスト CPU 実行のみ（GPU 経路・実測 baseline が無い）
//! のため REQ-2 の対象外（構造的 N/A。`docs/onnx-model-zoo-parity.md` §4）。

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use fandhe_ai_onnx_interop::onnx::graph::{Graph, build_graph};
use fandhe_ai_onnx_interop::onnx::interp::{InterpError, Value, run};
use fandhe_ai_onnx_interop::onnx::proto::{self, TensorProto, data_type};
use fandhe_ai_tensor_core::Tensor;
use prost::Message;

/// `run` の期待結果（fail-closed。`Ok` を緩く `is_ok()` 等で判定しない）。
enum RunExpectation {
    /// REQ-7 事前固定式で `output_0.pb` の参照値と全要素一致することを要求する。
    /// HEAD 時点ではどの `ZOO_MODELS` エントリも `Conv` 未対応で到達しないため
    /// 未構築（`dead_code` lint 対象）だが、`RunExpectation` は期待値反転手順
    /// （`docs/onnx-model-zoo-parity.md` §6）で使う契約上の variant であり、
    /// 削除すると sibling マージ後の反転先が無くなる。
    #[allow(dead_code)]
    Parity,
    /// `InterpError::UnsupportedOp(op)` の `op` が完全一致することを要求する
    /// （`is_err()` のような緩い判定は行わない。catch-all variant は設けない。
    /// `docs/onnx-model-zoo-parity.md` §6）。
    UnsupportedOp(&'static str),
}

/// Model Zoo モデル 1 件の期待値レコード。
struct ZooModel {
    /// `tests/fixtures/model-zoo/<dir>/` の `<dir>` 部分。
    dir: &'static str,
    opset_version: i64,
    node_count: usize,
    initializer_count: usize,
    inputs: &'static [&'static str],
    outputs: &'static [&'static str],
    /// `(op_type, count)` の組。`BTreeMap` へ変換して比較する（順不同一致）。
    op_histogram: &'static [(&'static str, usize)],
    expectation: RunExpectation,
}

/// tier A（コミット済み）+ tier B（`ONNX_INTEROP_MODEL_ZOO_DIR` 経由）の
/// 期待値表。HEAD 再プローブ結果は
/// `crates/onnx-interop/tests/fixtures/model-zoo/README.md` に記録済み。
const ZOO_MODELS: &[ZooModel] = &[
    ZooModel {
        dir: "mnist-12",
        opset_version: 12,
        node_count: 12,
        initializer_count: 8,
        inputs: &["Input3"],
        outputs: &["Plus214_Output_0"],
        op_histogram: &[
            ("Add", 3),
            ("Conv", 2),
            ("MatMul", 1),
            ("MaxPool", 2),
            ("Relu", 2),
            ("Reshape", 2),
        ],
        expectation: RunExpectation::UnsupportedOp("Conv"),
    },
    ZooModel {
        dir: "squeezenet1.0-12",
        opset_version: 12,
        node_count: 66,
        initializer_count: 53,
        inputs: &["data_0"],
        outputs: &["softmaxout_1"],
        op_histogram: &[
            ("Concat", 8),
            ("Conv", 26),
            ("Dropout", 1),
            ("GlobalAveragePool", 1),
            ("MaxPool", 3),
            ("Relu", 26),
            ("Softmax", 1),
        ],
        expectation: RunExpectation::UnsupportedOp("Conv"),
    },
    ZooModel {
        dir: "mobilenetv2-12",
        opset_version: 12,
        node_count: 105,
        initializer_count: 177,
        inputs: &["input"],
        outputs: &["output"],
        op_histogram: &[
            ("Add", 10),
            ("Clip", 35),
            ("Concat", 1),
            ("Constant", 1),
            ("Conv", 52),
            ("Gather", 1),
            ("Gemm", 1),
            ("GlobalAveragePool", 1),
            ("Reshape", 1),
            ("Shape", 1),
            ("Unsqueeze", 1),
        ],
        expectation: RunExpectation::UnsupportedOp("Conv"),
    },
    ZooModel {
        dir: "resnet50-v1-12",
        opset_version: 12,
        node_count: 175,
        initializer_count: 299,
        inputs: &["data"],
        outputs: &["resnetv17_dense0_fwd"],
        op_histogram: &[
            ("Add", 16),
            ("BatchNormalization", 53),
            ("Conv", 53),
            ("Flatten", 1),
            ("Gemm", 1),
            ("GlobalAveragePool", 1),
            ("MaxPool", 1),
            ("Relu", 49),
        ],
        expectation: RunExpectation::UnsupportedOp("Conv"),
    },
];

/// tier A（コミット済み）fixture ルート。
fn committed_fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model-zoo")
}

/// モデルディレクトリ（`<root>/<dir>/`）を解決する。`mnist-12` はコミット済み
/// ルート、それ以外（tier B）は `ONNX_INTEROP_MODEL_ZOO_DIR` 配下を見る。
fn model_dir(root: &Path, spec: &ZooModel) -> PathBuf {
    root.join(spec.dir)
}

fn find_spec(dir: &str) -> &'static ZooModel {
    ZOO_MODELS
        .iter()
        .find(|m| m.dir == dir)
        .unwrap_or_else(|| panic!("ZOO_MODELS に {dir} が無い（テスト定義側の不整合）"))
}

/// 読み込みを許容するファイルサイズ上限（1 GiB）。
///
/// `examples/model_zoo_probe.rs::MAX_READ_BYTES` と同値。理由も同じ
/// （A03。細工・破損した巨大 fixture を検証前に丸ごと読み込むとメモリ枯渇に
/// つながる）。tests と examples はコードを共有できないため定数・関数とも
/// 重複させている。
const MAX_READ_BYTES: u64 = 1024 * 1024 * 1024;

/// サイズ上限を検査してからファイル全体を読み込む
/// （`examples/model_zoo_probe.rs::read_file_bounded` と同型。TOCTOU 回避のため
/// `metadata` 取得と読み込みを同一 `File` ハンドルに対して行い、事前の `len`
/// 検査に加えて `take(max_bytes + 1)` による実読込量検査も行う fail-closed
/// 二重防御。上限は呼び出し元でテスト可能にするため引数化している）。
fn read_file_bounded_with_limit(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| format!("ファイルオープン失敗: {} ({e})", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("メタデータ取得失敗: {} ({e})", path.display()))?;
    let len = metadata.len();
    if len > max_bytes {
        return Err(format!(
            "ファイルサイズ上限超過: {} ({len} bytes > {max_bytes} bytes)",
            path.display()
        ));
    }

    let mut buf = Vec::new();
    let read_len = file
        .take(max_bytes + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("読み込み失敗: {} ({e})", path.display()))?;
    if read_len as u64 > max_bytes {
        return Err(format!(
            "ファイルサイズ上限超過（読み込み時検査）: {} (> {max_bytes} bytes)",
            path.display()
        ));
    }
    Ok(buf)
}

/// `MAX_READ_BYTES` 固定版（`load_model`・`load_tensor_pb` から呼ばれる本番経路）。
/// 失敗はこのファイルの既存スタイル（`panic!`）に合わせる。
fn read_file_bounded(path: &Path) -> Vec<u8> {
    read_file_bounded_with_limit(path, MAX_READ_BYTES)
        .unwrap_or_else(|e| panic!("fixture 読み込み失敗（上限付きローダー）: {e}"))
}

fn load_model(onnx_path: &Path) -> proto::ModelProto {
    let bytes = read_file_bounded(onnx_path);
    proto::decode_model(&bytes)
        .unwrap_or_else(|e| panic!("decode 失敗 {}: {e}", onnx_path.display()))
}

/// `TensorProto`（FLOAT 型限定）を `(name, Tensor<f32>)` へ復号する。
///
/// `graph::decode_tensor` は `pub(crate)` のため統合テスト（crate 外扱い）
/// から直接呼べない。ここでは同モジュールの検証順序（dims 非負・要素数
/// `checked_mul`・`raw_data` のバイト長完全一致を先に検査してから数値へ変換）
/// を鏡写しにする（A03・no-silent-skip 契約）。
fn load_tensor_pb(path: &Path) -> (String, Tensor<f32>) {
    let bytes = read_file_bounded(path);
    let t = TensorProto::decode(bytes.as_slice())
        .unwrap_or_else(|e| panic!("TensorProto decode 失敗 {}: {e}", path.display()));
    assert_eq!(
        t.data_type,
        data_type::FLOAT,
        "FLOAT 以外の data_type: {} ({})",
        t.data_type,
        path.display()
    );
    let mut expected_elements: usize = 1;
    for &d in &t.dims {
        assert!(d >= 0, "負の dim: {d} ({})", path.display());
        expected_elements = expected_elements
            .checked_mul(d as usize)
            .unwrap_or_else(|| panic!("要素数オーバーフロー: {}", path.display()));
    }
    let expected_bytes = expected_elements
        .checked_mul(4)
        .unwrap_or_else(|| panic!("バイト数オーバーフロー: {}", path.display()));
    assert_eq!(
        t.raw_data.len(),
        expected_bytes,
        "raw_data バイト長不一致 ({})",
        path.display()
    );
    let data: Vec<f32> = t
        .raw_data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let tensor = Tensor::<f32>::new(data, &shape)
        .unwrap_or_else(|e| panic!("Tensor::new 失敗 {}: {e:?}", path.display()));
    (t.name, tensor)
}

/// REQ-7 事前固定式で全要素検査する。fail 要素数・max 相対誤差をメッセージに
/// 含める（原因追跡のため）。
fn assert_req7(actual: &Tensor<f32>, expected: &Tensor<f32>) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape 不一致: actual={:?} expected={:?}",
        actual.shape(),
        expected.shape()
    );
    let actual_slice = actual.as_slice().expect("actual as_slice 失敗");
    let expected_slice = expected.as_slice().expect("expected as_slice 失敗");
    assert_eq!(actual_slice.len(), expected_slice.len());
    let mut fail_count = 0usize;
    let mut max_rel_err = 0.0f32;
    for (&a, &e) in actual_slice.iter().zip(expected_slice.iter()) {
        let rel_err = (a - e).abs() / (e.abs() + 1e-6);
        // rel_err が NaN になる（a・e のいずれかが NaN、または同符号の無限大同士等）
        // 場合、`rel_err > 1e-3` は false 判定になり fail_count に計上されない。
        // fail-closed（REQ-7）のため、非有限な rel_err は無条件で fail 扱いにする。
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
    assert_eq!(
        fail_count, 0,
        "REQ-7 判定式 fail: fail_count={fail_count} max_rel_err={max_rel_err}"
    );
}

/// グラフ構造（node／initializer 数・入出力名・op ヒストグラム・opset）を
/// 期待値と完全一致検査する。
fn assert_structure(graph: &Graph, model: &proto::ModelProto, spec: &ZooModel) {
    assert_eq!(
        graph.nodes.len(),
        spec.node_count,
        "{}: node_count 不一致",
        spec.dir
    );
    assert_eq!(
        graph.initializers.len(),
        spec.initializer_count,
        "{}: initializer_count 不一致",
        spec.dir
    );
    let graph_inputs: Vec<&str> = graph.inputs.iter().map(String::as_str).collect();
    let graph_outputs: Vec<&str> = graph.outputs.iter().map(String::as_str).collect();
    assert_eq!(graph_inputs, spec.inputs, "{}: inputs 不一致", spec.dir);
    assert_eq!(graph_outputs, spec.outputs, "{}: outputs 不一致", spec.dir);

    let mut histogram: BTreeMap<&str, usize> = BTreeMap::new();
    for node in &graph.nodes {
        *histogram.entry(node.op_type.as_str()).or_insert(0) += 1;
    }
    let expected_histogram: BTreeMap<&str, usize> = spec.op_histogram.iter().copied().collect();
    assert_eq!(
        histogram, expected_histogram,
        "{}: op ヒストグラム不一致",
        spec.dir
    );

    assert_eq!(
        model.opset_import.len(),
        1,
        "{}: opset_import は単一ドメインを想定",
        spec.dir
    );
    assert_eq!(
        model.opset_import[0].domain, "",
        "{}: opset domain は既定（空文字列）を想定",
        spec.dir
    );
    assert_eq!(
        model.opset_import[0].version, spec.opset_version,
        "{}: opset version 不一致",
        spec.dir
    );
}

/// `input_0.pb` を feed して `run` し、[`RunExpectation`] と fail-closed に
/// 照合する。
fn run_and_check(graph: &Graph, dir: &Path, spec: &ZooModel) {
    let test_dir = dir.join("test_data_set_0");
    let (_input_name, input_tensor) = load_tensor_pb(&test_dir.join("input_0.pb"));

    let mut feeds: HashMap<String, Value> = HashMap::new();
    feeds.insert(spec.inputs[0].to_string(), Value::F32(input_tensor));

    let result = run(graph, feeds);
    match spec.expectation {
        RunExpectation::Parity => {
            let result =
                result.unwrap_or_else(|e| panic!("{}: run 失敗（Parity 期待）: {e}", spec.dir));
            let (_output_name, expected_tensor) = load_tensor_pb(&test_dir.join("output_0.pb"));
            let actual = match &result[spec.outputs[0]] {
                Value::F32(t) => t,
                other => panic!("{}: Value::F32 を期待したが {other:?}", spec.dir),
            };
            assert_req7(actual, &expected_tensor);
        }
        RunExpectation::UnsupportedOp(expected_op) => {
            let err = result.expect_err(&format!(
                "{}: run が成功した（UnsupportedOp({expected_op}) を期待）。未対応 op が \
                 実装された場合は RunExpectation を Parity へ反転すること \
                 （docs/onnx-model-zoo-parity.md §6）",
                spec.dir
            ));
            match err {
                InterpError::UnsupportedOp(actual_op) => {
                    assert_eq!(
                        actual_op, expected_op,
                        "{}: UnsupportedOp の op_type が期待と不一致（別の op で \
                         止まった可能性。catch-all にせず RunExpectation を \
                         更新すること）",
                        spec.dir
                    );
                }
                other => panic!(
                    "{}: UnsupportedOp({expected_op}) を期待したが別エラー: {other}",
                    spec.dir
                ),
            }
        }
    }
}

// --- tier A: mnist-12（常時実行） ---

#[test]
fn mnist12_decodes_expected_graph_structure() {
    let spec = find_spec("mnist-12");
    let dir = model_dir(&committed_fixture_root(), spec);
    let model = load_model(&dir.join(format!("{}.onnx", spec.dir)));
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    assert_structure(&graph, &model, spec);
}

#[test]
fn mnist12_run_matches_expectation() {
    let spec = find_spec("mnist-12");
    let dir = model_dir(&committed_fixture_root(), spec);
    let model = load_model(&dir.join(format!("{}.onnx", spec.dir)));
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    run_and_check(&graph, &dir, spec);
}

#[test]
fn mnist12_fixture_byte_lengths_match_recorded_sizes() {
    // 依存クレート外の整合性ガード（`sha2` 等は許容依存外）。バイト長一致を
    // 無依存の最小限の改竄・破損検出として使う
    // （`docs/onnx-model-zoo-parity.md` §2。sha256 の正は
    // `tests/fixtures/model-zoo/README.md`）。
    let dir = committed_fixture_root().join("mnist-12");
    let onnx_len = std::fs::metadata(dir.join("mnist-12.onnx"))
        .expect("mnist-12.onnx が読めない")
        .len();
    let input_len = std::fs::metadata(dir.join("test_data_set_0/input_0.pb"))
        .expect("input_0.pb が読めない")
        .len();
    let output_len = std::fs::metadata(dir.join("test_data_set_0/output_0.pb"))
        .expect("output_0.pb が読めない")
        .len();
    assert_eq!(onnx_len, 26_143, "mnist-12.onnx のバイト長が記録値と不一致");
    assert_eq!(input_len, 3_157, "input_0.pb のバイト長が記録値と不一致");
    assert_eq!(output_len, 66, "output_0.pb のバイト長が記録値と不一致");
}

// --- tier B: squeezenet1.0-12 / mobilenetv2-12 / resnet50-v1-12（非コミット） ---

/// `ONNX_INTEROP_MODEL_ZOO_DIR` 経由で tier B モデルを検証する。未設定時は
/// `tests/onnx_transformer_e2e.rs` と同一運用（fail ではなく早期 return で
/// スキップし、`--ignored` を渡す `make test-ignored` を非破壊にする）。
fn run_tier_b(dir_name: &str) {
    let Ok(root) = std::env::var("ONNX_INTEROP_MODEL_ZOO_DIR") else {
        eprintln!(
            "skip: ONNX_INTEROP_MODEL_ZOO_DIR 未設定のため {dir_name} 系テストを \
             スキップします（tests/fixtures/model-zoo/README.md 参照）"
        );
        return;
    };
    let spec = find_spec(dir_name);
    let dir = model_dir(&PathBuf::from(root), spec);
    assert!(
        dir.is_dir(),
        "{}: ONNX_INTEROP_MODEL_ZOO_DIR は設定済みだがディレクトリが無い: {}（fail-closed。\
         README の取得手順で展開すること）",
        spec.dir,
        dir.display()
    );
    let model = load_model(&dir.join(format!("{}.onnx", spec.dir)));
    let graph = build_graph(&model).expect("build_graph は成功するはず");
    assert_structure(&graph, &model, spec);
    run_and_check(&graph, &dir, spec);
}

#[test]
#[ignore = "Model Zoo 非コミット fixture・tests/fixtures/model-zoo/README.md 参照"]
fn squeezenet1_0_12_matches_expectation() {
    run_tier_b("squeezenet1.0-12");
}

#[test]
#[ignore = "Model Zoo 非コミット fixture・tests/fixtures/model-zoo/README.md 参照"]
fn mobilenetv2_12_matches_expectation() {
    run_tier_b("mobilenetv2-12");
}

#[test]
#[ignore = "Model Zoo 非コミット fixture・tests/fixtures/model-zoo/README.md 参照"]
fn resnet50_v1_12_matches_expectation() {
    run_tier_b("resnet50-v1-12");
}

// --- 上限付きローダーの単体テスト（codex-review P0 指摘対応・PR #2225 sibling） ---

/// `read_file_bounded_with_limit` が上限超過を fail-closed で拒否し、上限
/// ちょうどは許容することを検証する（`tempfile` は許容依存外のため
/// `std::env::temp_dir()` 配下に自前で一時ファイルを作る。プロセス ID を
/// ファイル名へ含めて並行実行時の衝突を避ける）。
///
/// `take(max_bytes + 1)` による読み込み時検査（2 段目のガード）はファイルの
/// `len()` と実読込量が食い違うレース条件下でしか単独では踏めないため、本
/// テストでは `metadata().len()` 事前検査（1 段目）のみを対象にする。
#[test]
fn read_file_bounded_rejects_oversized_file() {
    let path = std::env::temp_dir().join(format!(
        "model_zoo_parity_bounded_{}_{}",
        std::process::id(),
        "rejects_oversized"
    ));
    let content = b"0123456789";
    std::fs::write(&path, content).expect("一時ファイル書き込み失敗");

    let too_small = read_file_bounded_with_limit(&path, (content.len() - 1) as u64);
    let exact = read_file_bounded_with_limit(&path, content.len() as u64);

    let _ = std::fs::remove_file(&path);

    assert!(
        too_small.is_err(),
        "上限を 1 バイト下回る指定で許容してしまった: {too_small:?}"
    );
    assert_eq!(
        exact.expect("上限ちょうどの指定は許容されるはず"),
        content.to_vec(),
        "上限ちょうどの指定で内容が変化した"
    );
}
