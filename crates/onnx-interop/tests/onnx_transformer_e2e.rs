//! イシュー #87（TASK-7.4a・REQ-7）: `transformer.onnx`（v1 資産、165 ノード・
//! オペ 20 種別）の end-to-end 推論実測テスト。
//!
//! `tests/onnx_decode.rs` の `transformer_onnx_decodes_expected_graph_structure`
//! は decode・`build_graph` までを固定化するが、本ファイルはさらに
//! `onnx::interp::run` によるグラフ実行までを通し、PyTorch 参照出力
//! （`tests/fixtures/pytorch-transformer/reference.json`）と REQ-7 事前固定
//! 判定式で数値一致を確認する（`docs/spec/04-requirements.md:161`）。
//!
//! ## 判定式についての注記（REQ-2 との混同禁止）
//!
//! `tests/onnx_interp.rs`・`tests/onnx_poc_v2_6_match.rs` と同じ REQ-7 事前固定
//! 基準 `abs_err / (|ref| + 1e-6) <= 1e-3` を用いる。`.claude/rules/coding-rust.md`
//! の REQ-2 バックエンド間数値一致 OR 複合判定（相対誤差 1e-3 未満 または絶対誤差
//! 1e-5 未満）とは別指標であり、両者を混同してどちらかを緩和しない。
//!
//! ## フィクスチャの出自
//!
//! - `transformer.onnx`（12MB 超・非コミット）: v1 実装リポ
//!   `Fandhe-AI/rust-ai-library-v1` commit `a14568897521f7bea6eac93218fe917cf2a25f04`
//!   の `crates/rust-ai-library/src/interop/onnx_model/transformer.onnx`。
//!   取得手順は `tests/fixtures/README.md` 参照
//! - `reference.json`（コミット済み）: 同 commit の
//!   `crates/rust-ai-library/tests/fixtures/pytorch-transformer/reference.json`。
//!   出自・sha256 は `tests/fixtures/pytorch-transformer/README.md` 参照

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use onnx_interop::onnx::graph::build_graph;
use onnx_interop::onnx::interp::{Value, run};
use onnx_interop::onnx::proto::ModelProto;
use prost::Message;
use serde::Deserialize;
use tensor_core::Tensor;

/// `reference.json` のうち本テストが使う部分のみを deserialize する。
/// `config` 等の付随メタデータ（onnxruntime セルフチェック値含む）は
/// 突合対象外のため構造体に含めない（未知フィールドは serde 既定で無視される）。
#[derive(Deserialize)]
struct TransformerReference {
    input_shape: Vec<usize>,
    input: Vec<Vec<Vec<f32>>>,
    output_shape: Vec<usize>,
    output: Vec<Vec<Vec<f32>>>,
}

fn flatten3(nested: &[Vec<Vec<f32>>]) -> Vec<f32> {
    nested
        .iter()
        .flat_map(|batch| batch.iter().flat_map(|row| row.iter().copied()))
        .collect()
}

#[test]
#[ignore = "12MB の transformer.onnx を非コミット方針としているため。tests/fixtures/README.md の取得手順を参照"]
fn transformer_onnx_end_to_end_matches_pytorch_reference_within_req7_tolerance() {
    // `cargo test -- --ignored`（`make test-ignored` 含む）でも非コミットの
    // transformer.onnx を取得していない環境では環境変数が未設定になるため、
    // fail ではなく早期 return でスキップする
    // （`tests/onnx_decode.rs` の transformer 系テストと同一運用。#77 Bugbot 指摘対応）。
    let Ok(path) = std::env::var("ONNX_INTEROP_TRANSFORMER_ONNX") else {
        eprintln!(
            "skip: ONNX_INTEROP_TRANSFORMER_ONNX 未設定のため \
             transformer_onnx_end_to_end_matches_pytorch_reference_within_req7_tolerance \
             をスキップします（tests/fixtures/README.md 参照）"
        );
        return;
    };

    let model_bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("読み込み失敗 {path}: {e}"));
    let model = ModelProto::decode(model_bytes.as_slice()).expect("decode に成功するはず");
    let graph = build_graph(&model).expect("build_graph は成功するはず");

    let reference_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pytorch-transformer/reference.json");
    let reference_json = std::fs::read_to_string(&reference_path)
        .unwrap_or_else(|e| panic!("reference.json 読み込み失敗 {reference_path:?}: {e}"));
    let reference: TransformerReference =
        serde_json::from_str(&reference_json).expect("reference.json パース失敗");

    let input_flat = flatten3(&reference.input);
    let input_tensor = Tensor::<f32>::new(input_flat, &reference.input_shape)
        .expect("input テンソル構築に成功するはず（shape は reference.json 由来）");

    let mut feeds = HashMap::new();
    feeds.insert("input".to_string(), Value::F32(input_tensor));

    // 受け入れ条件「推論が完走」の直接固定化。#87 の I64 算術対応
    // （`compute_add`/`compute_mul`/`compute_div`/`compute_mod`。
    // `crates/onnx-interop/src/onnx/interp.rs`）により
    // `/layers.0/self_attn/Div` の `InterpError::TypeMismatch` は解消済みで、
    // run 自体は成功する。本テスト実装時点の実測で唯一残る失敗点は本関数末尾の
    // REQ-7 数値一致基準（相対誤差 1e-3）の `assert!` で、16,384 要素中 7 要素が
    // 超過することを確認済み。#413 で erf 精度改善・主要オペ累積の f64 化を含む
    // 実装改善を試みたが、最も強い改善（`erf` 高精度化 + `MatMul`/`Gemm`/
    // `LayerNormalization`/`Softmax` 累積の f64 化）を適用しても 1 要素が残存し
    // 解消しないことを確認した（判定式が近ゼロ参照値で相対誤差を拡大する構造的性質が
    // 支配的。実測詳細・spec への改善提案は
    // docs/perf/onnx-transformer-e2e-error-analysis.md 参照）。
    // 本テストはこの失敗を隠蔽せず素直に panic させる
    // （no-silent-skip 契約。OWASP A08 の迂回経路を作らない方針に合わせる）。
    let start = Instant::now();
    let result = run(&graph, feeds).expect("run は成功するはず（受け入れ条件: 推論が完走）");
    let elapsed = start.elapsed();

    let output = match &result["output"] {
        Value::F32(t) => t,
        other => panic!("Value::F32 を期待したが {other:?}"),
    };
    assert_eq!(
        output.shape(),
        reference.output_shape.as_slice(),
        "出力 shape が reference.json の output_shape と一致しない"
    );

    let [batch, seq_len, d_model] = reference.output_shape[..] else {
        panic!(
            "output_shape は 3 次元 [batch, seq_len, d_model] のはず: {:?}",
            reference.output_shape
        );
    };

    let mut max_rel_err = 0.0f32;
    let mut exceed_count = 0usize;
    let total_elements = batch * seq_len * d_model;
    for (b, batch_ref) in reference.output.iter().enumerate() {
        for (s, row_ref) in batch_ref.iter().enumerate() {
            for (d, &expected) in row_ref.iter().enumerate() {
                let actual = output
                    .get(&[b, s, d])
                    .unwrap_or_else(|| panic!("out-of-range: ({b},{s},{d})"));
                let abs_err = (actual - expected).abs();
                // REQ-7 事前固定判定式（本ファイル冒頭 `//!` 参照）。
                let rel_err = abs_err / (expected.abs() + 1e-6);
                if rel_err > 1e-3 {
                    exceed_count += 1;
                }
                max_rel_err = max_rel_err.max(rel_err);
                assert!(
                    rel_err <= 1e-3,
                    "REQ-7 数値一致基準を超過: (b={b},s={s},d={d}) expected={expected} \
                     actual={actual} abs_err={abs_err} rel_err={rel_err}"
                );
            }
        }
    }

    // 実測記録（docs/perf/onnx-transformer-e2e-measurement.md への転記用）。
    // 実行時間は TASK-8.3（#154・bench-harness）による正式な性能下限確定までの参考値。
    eprintln!(
        "transformer.onnx e2e: max_rel_err={max_rel_err} (threshold=1e-3) \
         exceed_count={exceed_count}/{total_elements} elapsed={elapsed:?}"
    );
}
