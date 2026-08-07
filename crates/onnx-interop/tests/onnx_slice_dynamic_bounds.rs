//! イシュー #80（TASK-7.2d）: 動的境界 `Slice` パターンの ops 連結再現・PoC-v2-6 数値突合。
//!
//! v1 実装リポで踏んだ `burn-onnx` 失敗パターン（`tracel-ai/burn#5295`。バッチ次元を
//! `Shape → Gather → Unsqueeze → Concat` で実行時に取り出し `Slice` の `ends` を動的に
//! 構築するグラフが未対応だった）の最小再現を、本クレートの `ops` 関数を直接連結して
//! なぞる。ONNX proto デコード（#77 TASK-7.2a）・グラフ実行インタープリタ（#78。
//! 未実装・ブランチなし）のいずれにも依存しない（decode → 属性値解決 → 本モジュール
//! 呼び出しの結線は #78 以降の担当。`crates/onnx-interop/src/ops/mod.rs` 冒頭コメント
//! 参照）。本ファイルは「動的境界を要求するグラフ形状を production オペで正しく解ける
//! こと」のみを固定化する。
//!
//! ## パイプライン
//!
//! ONNX `Shape` は本クレートでは `Vec<i64>`（`tensor-core::Element` が `i64` 非対応の
//! ため。`ops::shape_ops` 冒頭コメント参照）を返す素の Rust 関数であり、`Gather`／
//! `Unsqueeze`／`Concat` は `Tensor<f32>` 上で動作する。実際の ONNX ランタイムは
//! `Shape` の出力（int64 テンソル）をそのまま後続オペへ渡すが、本クレートの型設計では
//! 素の `Vec<i64>` を一旦 `Tensor<f32>` へキャストしてから `Gather`／`Unsqueeze`／
//! `Concat` に通す（値は小さい非負整数のみのため f32 キャストで精度損失は起きない）。
//! これにより Gather（スカラーインデックスによる rank 縮退）・Unsqueeze（縮退した
//! スカラーを rank 1 へ戻す）・Concat（定数 4 と連結して `ends=[batch,4]` を構築する）の
//! 3 オペを実データで通し、最後に `Slice` へ渡す。
//!
//! ## 判定式についての注記（REQ-2 との混同禁止）
//!
//! `tests/onnx_poc_v2_6_match.rs` 冒頭コメントと同じ REQ-7 事前固定基準
//! `abs_err / (|ref| + 1e-6) <= 1e-3` を用いる。

use std::path::Path;

use onnx_interop::ops::{shape, unsqueeze};
use serde::Deserialize;
use tensor_core::Tensor;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/onnx-reference");

#[derive(Deserialize)]
struct SliceReproReference {
    inputs: Vec<Vec<f32>>,
    outputs: Vec<Vec<f32>>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

fn load_reference() -> SliceReproReference {
    let raw = std::fs::read_to_string(fixture_path("slice_repro_reference.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
}

/// burn#5295 再現パイプライン本体: `x` から `ends = [batch, 4]` を実行時に構築し、
/// `x[:, :4]`（先頭 4 列）を切り出す。`Shape → Gather(scalar index) → Unsqueeze →
/// Concat → Slice` を production オペで連結する（本ファイル冒頭コメント参照）。
fn dynamic_bounds_slice_via_ops(x: &Tensor<f32>) -> Tensor<f32> {
    // Shape: [batch, cols] を Vec<i64> で得る（ops::shape_ops::shape）。
    let dims = shape(x);
    assert_eq!(dims.len(), 2, "本パイプラインは 2 次元入力のみ想定する");

    // Shape の出力（int64 相当）を Tensor<f32> へキャストし、Gather/Unsqueeze/Concat の
    // 実入力として使う（本ファイル冒頭コメントの型設計上の理由）。
    let dims_f32: Vec<f32> = dims.iter().map(|&d| d as f32).collect();
    let dims_tensor = Tensor::<f32>::new(dims_f32, &[dims.len()]).unwrap();

    // Gather(dims_tensor, indices=[0], axis=0, indices_shape=[]): スカラーインデックス
    // により rank 0（スカラー形状）へ縮退する（ONNX Gather-13 の rank 縮退規則）。
    let batch_scalar =
        onnx_interop::ops::gather(&dims_tensor, &[0], &[], 0).expect("gather(batch) 失敗");
    assert!(
        batch_scalar.shape().is_empty(),
        "スカラーインデックスの Gather は rank 0 へ縮退するはず: {:?}",
        batch_scalar.shape()
    );

    // Unsqueeze(batch_scalar, axes=[0]): スカラーを shape [1] へ戻す（Concat の
    // rank 要件を満たすため。ONNX グラフでも同じ理由で Unsqueeze が挟まる）。
    let batch_1d = unsqueeze(&batch_scalar, &[0]).expect("unsqueeze(batch) 失敗");
    assert_eq!(batch_1d.shape(), &[1]);

    // Concat([batch_1d, four_1d], axis=0): ends = [batch, 4] を構築する。
    let four_1d = Tensor::<f32>::new(vec![4.0], &[1]).unwrap();
    let ends_tensor =
        onnx_interop::ops::concat(&[&batch_1d, &four_1d], 0).expect("concat(ends) 失敗");
    assert_eq!(ends_tensor.shape(), &[2]);

    // f32 -> i64 へ戻す（Slice の実行時パラメータは decode 層解決後の &[i64] を想定する
    // 型設計のため。`ops::slice` 冒頭コメント参照）。
    let batch_i64 = ends_tensor.get(&[0]).unwrap().round() as i64;
    let four_i64 = ends_tensor.get(&[1]).unwrap().round() as i64;
    assert_eq!(four_i64, 4);

    onnx_interop::ops::slice(
        x,
        &onnx_interop::ops::SliceParams {
            starts: &[0, 0],
            ends: &[batch_i64, four_i64],
            axes: Some(&[0, 1]),
            steps: None,
        },
    )
    .expect("slice(dynamic ends) 失敗")
}

#[test]
fn dynamic_bounds_pipeline_matches_slice_repro_reference() {
    let reference = load_reference();
    assert_eq!(reference.inputs.len(), reference.input_shape[0]);

    let flat: Vec<f32> = reference.inputs.iter().flatten().copied().collect();
    let x = Tensor::<f32>::new(flat, &reference.input_shape).unwrap();

    let y = dynamic_bounds_slice_via_ops(&x);
    assert_eq!(y.shape(), reference.output_shape.as_slice());

    let mut max_rel_err = 0.0f32;
    for (i, row) in reference.outputs.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            let actual = y.get(&[i, j]).unwrap();
            let abs_err = (actual - expected).abs();
            // REQ-7 事前固定判定式（本ファイル冒頭 `//!` 参照。REQ-2 の OR 複合判定とは
            // 別指標であり緩和しない）。
            let rel_err = abs_err / (expected.abs() + 1e-6);
            max_rel_err = max_rel_err.max(rel_err);
            assert!(
                rel_err <= 1e-3,
                "REQ-7 数値一致基準を超過: (i={i},j={j}) expected={expected} actual={actual} \
                 abs_err={abs_err} rel_err={rel_err}"
            );
        }
    }
    eprintln!("max_rel_err={max_rel_err} (threshold=1e-3)");
}

#[test]
fn dynamic_bounds_pipeline_rejects_mismatched_input_rank() {
    // 本パイプラインは 2 次元入力のみを想定するガード（`dims.len() == 2` assert）が
    // 想定外の rank に対して panic ではなく明確な assertion で失敗することを固定化する
    // （実装バグの早期検出目的。production オペ自体のエラーではなくテストヘルパーの
    // 契約であるため `#[should_panic]` で確認する）。
    let x = Tensor::<f32>::zeros(&[3]).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dynamic_bounds_slice_via_ops(&x)
    }));
    assert!(result.is_err(), "rank 1 入力は assert で拒否されるはず");
}
