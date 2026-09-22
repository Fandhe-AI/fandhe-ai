//! `fandhe_ai::interop::onnx::OnnxModel` と `fandhe_ai_onnx_interop` の直接
//! 呼び出し（`decode_model` → `build_graph` → `interp::run`）を、第三者公開
//! モデル（ONNX Model Zoo `mnist-12`。イシュー #2081）に対して突合する
//! テスト（`tests/interop_onnx_internal_parity.rs` の Model Zoo 版）。
//!
//! **本ファイルは意図的に `fandhe_ai` と `fandhe_ai_onnx_interop` の両方を
//! import する**（facade ラッパーが内部エラーを写像するだけの薄い層で
//! あり、迂回経路がないことを直接検証するため）。
//!
//! facade は `prost` を dev-dependency にも持たないため `.pb` を facade 側で
//! decode できない。そのため本ファイルは決定的な合成入力
//! （`[1,1,28,28]` の固定パターン）を使い、Model Zoo 同梱 `output_0.pb` との
//! REQ-7 parity 実測は内部クレート側 `crates/onnx-interop/tests/model_zoo_parity.rs`
//! が正であり本ファイルでは重複しない。
//!
//! ## HEAD 時点の期待値（green parity ではない）
//!
//! `mnist-12` は `Conv`（未対応 op。追跡先はイシュー #2199。`auto_pad` は
//! #2199 の受け入れ条件に含まれず別途追跡が必要）で `run` が止まるため、
//! HEAD では `OnnxModel::run` が `OnnxError::UnsupportedOp { op_type: "Conv" }`
//! を返すことを固定する。合わせて同一入力で内部クレート `interp::run` も
//! `InterpError::UnsupportedOp("Conv")` を返すことを検証し、facade が独自の
//! 迂回経路（例えば内部エラーを握りつぶして別の結果を返す等）を持たないこと
//! を確認する。未対応 op が実装され `run` が先へ進んだ場合は、facade 出力と
//! 内部クレート出力の bit 同一検査へ切り替える
//! （`docs/onnx-model-zoo-parity.md` §6・`tests/interop_onnx_internal_parity.rs`
//! と同型の判定）。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel, OnnxValue};

use fandhe_ai_onnx_interop::onnx::graph::build_graph;
use fandhe_ai_onnx_interop::onnx::interp::{self, InterpError, Value};
use fandhe_ai_onnx_interop::onnx::proto;

fn model_zoo_fixture(rel: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures/model-zoo"
    ))
    .join(rel)
}

/// `[1,1,28,28]`（mnist-12 の入力形）の決定的な合成入力。`.pb` を使わず
/// 固定パターンで多要素を網羅する（乱数生成器を新規導入しない。
/// `tests/interop_onnx_internal_parity.rs::slice_repro_onnx_...` と同じ方針）。
fn synthetic_mnist_input() -> Vec<f32> {
    (0..28 * 28).map(|i| ((i as f32) * 0.011) % 1.0).collect()
}

#[test]
fn mnist12_facade_and_internal_agree_on_unsupported_conv() {
    // facade 経由。
    let facade_model =
        OnnxModel::from_path(model_zoo_fixture("mnist-12/mnist-12.onnx")).expect("from_path 成功");

    let mut facade_feeds = HashMap::new();
    facade_feeds.insert(
        "Input3".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(synthetic_mnist_input(), &[1, 1, 28, 28]).unwrap()),
    );
    let facade_result = facade_model.run(facade_feeds);
    let facade_err = match facade_result {
        Ok(_) => panic!(
            "facade run が成功した（OnnxError::UnsupportedOp を期待）。Conv が実装された \
             場合は本テストを bit 同一検査へ更新すること（docs/onnx-model-zoo-parity.md §6）"
        ),
        Err(e) => e,
    };
    match &facade_err {
        OnnxError::UnsupportedOp { op_type } => {
            assert_eq!(op_type, "Conv", "facade: 未対応 op が Conv 以外");
        }
        other => panic!("facade: OnnxError::UnsupportedOp を期待したが {other:?}"),
    }

    // 内部クレート直接呼び出し（迂回経路の不在を確認する対照実験）。
    let bytes =
        std::fs::read(model_zoo_fixture("mnist-12/mnist-12.onnx")).expect("mnist-12.onnx 読込失敗");
    let internal_model = proto::decode_model(&bytes).expect("decode は成功するはず");
    let internal_graph = build_graph(&internal_model).expect("build_graph は成功するはず");

    let mut internal_feeds = HashMap::new();
    internal_feeds.insert(
        "Input3".to_string(),
        Value::F32(Tensor::<f32>::new(synthetic_mnist_input(), &[1, 1, 28, 28]).unwrap()),
    );
    let internal_result = interp::run(&internal_graph, internal_feeds);
    let internal_err = match internal_result {
        Ok(_) => panic!("internal run が成功した（InterpError::UnsupportedOp を期待）"),
        Err(e) => e,
    };
    match internal_err {
        InterpError::UnsupportedOp(op) => {
            assert_eq!(op, "Conv", "internal: 未対応 op が Conv 以外");
        }
        other => panic!("internal: UnsupportedOp を期待したが {other}"),
    }
}
