//! イシュー #80（TASK-7.2d・REQ-7）: ONNX 経路（MLP）の production オペ forward が
//! PoC-v2-6 参照値（`onnx_reference.json`）と数値一致することを固定化する。
//!
//! `tests/st_poc_v2_6_match.rs`（#75）は safetensors 経路の同種突合を担う。本ファイルは
//! **ONNX 経路**を担い、`docs/spec/03-poc/poc-v2-6-interop/code/fixtures/model.onnx` の
//! 重み（fc1/fc2/fc3 の weight/bias）を用いて同じ MLP（`2->8->8->1`、ReLU x2 + Sigmoid）を
//! `onnx_interop::ops::{gemm, relu, sigmoid}` で再現する。
//!
//! ONNX proto デコード層（#77 TASK-7.2a）は本イシュー着手時点で main 未マージのため、
//! `model.onnx` の initializer をテストコードで直接デコードせず、一回きりのオフライン
//! スクリプトで抽出済みの `onnx_weights.json` を読む（`tests/fixtures/onnx-reference/
//! README.md` に生成手順・sha256 を記録。テストコード内に独自 protobuf パーサを実装せず
//! #77 との重複を避ける）。`#77` マージ後に `model.onnx` 直接デコード経路へ切り替える
//! 判断は `#86`（TASK-7.4 end-to-end 実測）側で行う。
//!
//! ## 判定式についての注記（REQ-2 との混同禁止）
//!
//! `tests/st_poc_v2_6_match.rs` と同じ REQ-7 事前固定基準
//! `abs_err / (|ref| + 1e-6) <= 1e-3`（PoC-v2-6 `st_infer.rs`・v1 PoC-6 と同一）を用いる。
//! これは `.claude/rules/coding-rust.md` の REQ-2 バックエンド間数値一致 OR 複合判定
//! （相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）とは**別指標**であり、両者を混同して
//! どちらかを緩和しない。
//!
//! ONNX 経路と safetensors 経路の参照値は別実行のため 1e-6〜1e-9 オーダーで異なる
//! （`tests/fixtures/onnx-reference/README.md`）。本ファイルは `onnx_reference.json` の
//! みと突合し、safetensors 経路の `st_reference.json` とクロス比較しない。

use std::collections::HashMap;
use std::path::Path;

use onnx_interop::ops::{GemmAttrs, gemm, relu, sigmoid};
use serde::Deserialize;
use tensor_core::Tensor;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/onnx-reference");

const WEIGHT_KEYS: [&str; 6] = [
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
];

#[derive(Deserialize)]
struct OnnxReference {
    inputs: Vec<[f32; 2]>,
    outputs: Vec<f32>,
}

#[derive(Deserialize)]
struct OnnxWeightsFixture {
    weights: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

/// `onnx_weights.json`（`model.onnx` initializer 抽出済み fixture）を
/// `Tensor<f32>` へロードする。`shapes` の次元順は `[out_features, in_features]`
/// （safetensors 経路と同じネイティブレイアウト。README 参照）。
fn load_onnx_weights() -> HashMap<String, Tensor<f32>> {
    let raw = std::fs::read_to_string(fixture_path("onnx_weights.json")).unwrap();
    let fixture: OnnxWeightsFixture = serde_json::from_str(&raw).unwrap();

    let mut map = HashMap::with_capacity(WEIGHT_KEYS.len());
    for key in WEIGHT_KEYS {
        let values = fixture
            .weights
            .get(key)
            .unwrap_or_else(|| panic!("onnx_weights.json に {key} が無い"))
            .clone();
        let dims = fixture
            .shapes
            .get(key)
            .unwrap_or_else(|| panic!("onnx_weights.json の shapes に {key} が無い"))
            .clone();
        let t = Tensor::<f32>::new(values, &dims)
            .unwrap_or_else(|e| panic!("{key} の Tensor 構築に失敗: {e:?}"));
        map.insert(key.to_string(), t);
    }
    map
}

/// `onnx_weight_shapes.json`（PoC-v2-6 由来の shape 参照値）を読み、`onnx_weights.json`
/// の `shapes` と一致することを別経路で検算する（fixture 生成スクリプトの誤りを検出する
/// ための冗長チェック。README 記載の「抽出時に検証済み」を CI 上でも再確認する）。
fn load_reference_shapes() -> HashMap<String, Vec<usize>> {
    let raw = std::fs::read_to_string(fixture_path("onnx_weight_shapes.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

// --- テスト群 1: fixture の shape 整合性検証 ---

#[test]
fn onnx_weights_fixture_shapes_match_reference_shapes() {
    let map = load_onnx_weights();
    let ref_shapes = load_reference_shapes();
    assert_eq!(
        ref_shapes.len(),
        WEIGHT_KEYS.len(),
        "onnx_weight_shapes.json のキー数が想定と異なる"
    );
    for key in WEIGHT_KEYS {
        assert_eq!(
            map[key].shape(),
            ref_shapes[key].as_slice(),
            "shape mismatch for key {key}（fixture 生成スクリプトの抽出誤りの疑い）"
        );
    }
}

// --- テスト群 2（受け入れ条件本体）: production オペ経由 forward の PoC-v2-6 数値突合 ---
//
// PoC-v2-6 の ONNX グラフ（`model.onnx`）は `Gemm(transB=1)` ×3 + `Relu` ×2 + `Sigmoid`
// で構成される（safetensors 経路の `st_poc_v2_6_match.rs` と同一レイヤー順・活性化配置。
// 重みの shape が `[out_features, in_features]` ネイティブのまま格納されているのも同じ
// ため、`transpose_2d()` で明示転置してから `trans_b` なしで `gemm` に渡す）。

fn linear_via_gemm(x: &Tensor<f32>, w: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    let w_t = w.transpose_2d().unwrap();
    gemm(x, &w_t, Some(b), &GemmAttrs::default()).unwrap()
}

fn mlp_forward_via_ops(map: &HashMap<String, Tensor<f32>>, input: &[f32; 2]) -> f32 {
    let x = Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap();

    let h1 = relu(&linear_via_gemm(&x, &map["fc1.weight"], &map["fc1.bias"])).unwrap();
    let h2 = relu(&linear_via_gemm(&h1, &map["fc2.weight"], &map["fc2.bias"])).unwrap();
    let out = sigmoid(&linear_via_gemm(&h2, &map["fc3.weight"], &map["fc3.bias"])).unwrap();

    out.get(&[0, 0]).unwrap()
}

#[test]
fn ops_forward_matches_onnx_pytorch_reference_within_req7_tolerance() {
    let map = load_onnx_weights();

    let reference_json = std::fs::read_to_string(fixture_path("onnx_reference.json")).unwrap();
    let reference: OnnxReference = serde_json::from_str(&reference_json).unwrap();
    assert_eq!(reference.inputs.len(), reference.outputs.len());
    assert!(
        !reference.inputs.is_empty(),
        "fixture が空では突合にならない"
    );

    let mut max_rel_err = 0.0f32;
    for (input, &expected) in reference.inputs.iter().zip(reference.outputs.iter()) {
        let actual = mlp_forward_via_ops(&map, input);
        let abs_err = (actual - expected).abs();
        // REQ-7 事前固定判定式（本ファイル冒頭 `//!` 参照。REQ-2 の OR 複合判定とは別指標）。
        let rel_err = abs_err / (expected.abs() + 1e-6);
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            rel_err <= 1e-3,
            "REQ-7 数値一致基準を超過: input={input:?} expected={expected} actual={actual} \
             abs_err={abs_err} rel_err={rel_err}"
        );
    }
    // fixture 生成時の Python 検算（README 記載）では最大相対誤差 1.09e-06（f64/f32・
    // 丸め順序差に由来）。Rust 側 f32 演算でも閾値 1e-3 を十分下回ることをここで
    // 診断出力する（ループ内 assert が全件 <= 1e-3 を既に保証しているため追加 assert は
    // 行わない。`st_poc_v2_6_match.rs` と同じ理由で二重 assert を避ける）。
    eprintln!("max_rel_err={max_rel_err} (threshold=1e-3)");
}
