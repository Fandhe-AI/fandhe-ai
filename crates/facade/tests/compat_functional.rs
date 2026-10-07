//! `fandhe_ai::compat` の Functional API（`FunctionalBuilder`・`FunctionalModel`・`Node`。イシュー #2679・
//! 親 #2625。承認はルート #2499 のコメント issuecomment-6033824965。`docs/facade-functional-api-decision.md`
//! §10・§13・§16〜§18）の公開面を CPU で検証する統合テスト。`fandhe_ai` だけを import する。
//!
//! 検査項目:
//! - 単一ブロックの鎖が同じ層構成の `Sequential` と forward・パラメータ名・学習履歴まで bit 一致する
//! - 多入力・fan-out・結合 4 種（`concatenate`／`add`／`multiply`／`average`）の forward が手計算の参照と一致する
//! - 多入力・多出力グラフの `compile`→`fit`→`evaluate`（損失が下がる・`trainable_parameters`／`apply_parameters`）
//! - `state_dict`／`load_state_dict`／`named_parameters` の通し番号キー
//! - 拒否系（空の入力・出力、他ビルダーのハンドル、結合入力 2 件未満、未 compile の `fit`、`Optimizer::Lbfgs`）が
//!   panic せず型付きエラーになること
//!
//! 保存・復元は `compat_functional_model_io.rs`。ホスト計算のみのため `#[ignore]` 分離は行わない。

use std::sync::{Mutex, MutexGuard};

use fandhe_ai::compat::{
    FitConfig, FunctionalBuilder, FunctionalModel, Loss, Node, Optimizer, Sequential,
};
use fandhe_ai::optim::{LbfgsConfig, SgdConfig};
use fandhe_ai::{AutodiffError, Tensor};

/// グローバル RNG を暗黙に消費する経路（学習時 dropout・`shuffle(true)`）を避けるため通常は不要だが、
/// 将来の変更で RNG を使う検査を足したときに直列化できるよう置いておく。
fn rng_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

/// 決定的な擬似データ。
fn det(rows: usize, cols: usize, salt: f32) -> Tensor<f32> {
    tensor(
        (0..rows * cols)
            .map(|k| ((k as f32) * 0.37 + salt).sin())
            .collect(),
        &[rows, cols],
    )
}

/// REQ-2 統一複合判定。閾値の直書きを避け、既存の `assert_parity` を再利用する（tolerance は不変）。
fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    fandhe_ai_backend_cpu::assert_parity(label, actual, expected);
}

fn seq() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, 0x1111)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, 0x2222)
        .unwrap()
}

fn chain(block: Sequential) -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let y = b.apply(block, x).unwrap();
    b.build(&[x], &[y]).unwrap()
}

// ---------------------------------------------------------------------
// 単一ブロックの鎖は Sequential と一致する
// ---------------------------------------------------------------------

#[test]
fn single_block_chain_matches_sequential_forward_and_parameter_names() {
    let x = det(5, 4, 0.1);
    let mut model = chain(seq());
    let mut reference = seq();
    model.eval();
    reference.eval();

    let via_predict = model.predict(&[&x]).unwrap();
    assert_eq!(via_predict.len(), 1);
    assert_eq!(bits(&via_predict[0]), bits(&reference.predict(&x).unwrap()));

    // tape 上の forward も同じ値になる。
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let outs = model.forward(&tape, &[xv]).unwrap();
    assert_eq!(bits(&outs[0].to_tensor()), bits(&via_predict[0]));

    // 通し番号キー（ブロック 1 つなら Sequential と同じ `{i}.{name}`）。
    let named: Vec<String> = model
        .named_parameters()
        .unwrap()
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let want: Vec<String> = reference
        .named_parameters()
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(named, want);
    let sd = model.state_dict().unwrap();
    for (k, v) in reference.state_dict() {
        assert_eq!(bits(&sd[&k]), bits(&v), "{k}");
    }
}

#[test]
fn single_block_chain_fit_history_is_bit_identical_to_sequential_fit() {
    let _guard = rng_lock();
    let x = det(12, 4, 0.2);
    let y = det(12, 2, 0.8);
    let config = || FitConfig::new(4, 4);

    let mut model = chain(seq());
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let history = model.fit(&[&x], &[&y], config()).unwrap();

    let mut reference = seq();
    reference
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let ref_history = reference.fit(&x, &y, config()).unwrap();

    let a: Vec<u32> = history.loss.iter().map(|v| v.to_bits()).collect();
    let b: Vec<u32> = ref_history.loss.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a, b, "history.loss が Sequential::fit と bit 一致しない");
    let sd = model.state_dict().unwrap();
    for (k, v) in reference.state_dict() {
        assert_eq!(bits(&sd[&k]), bits(&v), "{k}");
    }
    // evaluate も同じ。
    let e1 = model.evaluate(&[&x], &[&y], 4).unwrap();
    let e2 = reference.evaluate(&x, &y, 4).unwrap();
    assert_eq!(e1.to_bits(), e2.to_bits());
}

// ---------------------------------------------------------------------
// 結合ノード 4 種の数値（手計算の参照）
// ---------------------------------------------------------------------

#[test]
fn merge_nodes_match_reference_values() {
    // パラメータを持たない恒等ブロック（relu）を挟んで、結合ノード 4 種を同じ入力 2 つへ適用する。
    let mut b = FunctionalBuilder::new();
    let a = b.input().unwrap();
    let c = b.input().unwrap();
    let ra = b.apply(Sequential::new().add_relu(), a).unwrap();
    let rc = b.apply(Sequential::new().add_relu(), c).unwrap();
    let cat = b.concatenate(&[ra, rc], 1).unwrap();
    let sum = b.add(&[ra, rc]).unwrap();
    let prod = b.multiply(&[ra, rc]).unwrap();
    let mean = b.average(&[ra, rc]).unwrap();
    let mut model = b.build(&[a, c], &[cat, sum, prod, mean]).unwrap();
    model.eval();

    let xa = det(3, 2, 0.3);
    let xc = det(3, 2, 1.1);
    let outs = model.predict(&[&xa, &xc]).unwrap();
    assert_eq!(outs.len(), 4);

    let ra_v: Vec<f32> = values(&xa).into_iter().map(|v| v.max(0.0)).collect();
    let rc_v: Vec<f32> = values(&xc).into_iter().map(|v| v.max(0.0)).collect();
    // concatenate（dim=1）: 行ごとに [ra 行, rc 行]。
    let mut cat_expected = Vec::new();
    for r in 0..3 {
        cat_expected.extend_from_slice(&ra_v[r * 2..r * 2 + 2]);
        cat_expected.extend_from_slice(&rc_v[r * 2..r * 2 + 2]);
    }
    assert_eq!(outs[0].shape(), &[3, 4]);
    assert_close("concatenate", &values(&outs[0]), &cat_expected);
    let add_expected: Vec<f32> = ra_v.iter().zip(&rc_v).map(|(p, q)| p + q).collect();
    let mul_expected: Vec<f32> = ra_v.iter().zip(&rc_v).map(|(p, q)| p * q).collect();
    let avg_expected: Vec<f32> = ra_v.iter().zip(&rc_v).map(|(p, q)| (p + q) / 2.0).collect();
    assert_close("add", &values(&outs[1]), &add_expected);
    assert_close("multiply", &values(&outs[2]), &mul_expected);
    assert_close("average", &values(&outs[3]), &avg_expected);
}

// ---------------------------------------------------------------------
// 多入力・多出力グラフの学習
// ---------------------------------------------------------------------

/// 2 入力・fan-out・加算結合・2 出力のグラフ（ブロック 4 つ）。
fn two_in_two_out() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let a = b.input().unwrap();
    let c = b.input().unwrap();
    let ha = b
        .apply(
            Sequential::new().add_linear(3, 4, 11).unwrap().add_relu(),
            a,
        )
        .unwrap();
    let hc = b
        .apply(
            Sequential::new().add_linear(2, 4, 12).unwrap().add_relu(),
            c,
        )
        .unwrap();
    let merged = b.add(&[ha, hc]).unwrap();
    let o1 = b
        .apply(Sequential::new().add_linear(4, 1, 13).unwrap(), merged)
        .unwrap();
    let o2 = b
        .apply(Sequential::new().add_linear(4, 2, 14).unwrap(), merged)
        .unwrap();
    b.build(&[a, c], &[o1, o2]).unwrap()
}

#[test]
fn multi_input_multi_output_graph_trains_and_evaluates() {
    let _guard = rng_lock();
    let (xa, xc) = (det(8, 3, 0.1), det(8, 2, 0.6));
    let (y1, y2) = (det(8, 1, 1.2), det(8, 2, 1.8));

    let mut model = two_in_two_out();
    assert!(model.training());
    // ブロック 4 つ・各 Linear が weight と bias を持つ。
    assert_eq!(model.trainable_parameters().len(), 8);
    // 未 compile の fit／evaluate は型付きエラー。
    assert!(matches!(
        model.fit(&[&xa, &xc], &[&y1, &y2], FitConfig::new(1, 4)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        model.evaluate(&[&xa, &xc], &[&y1, &y2], 4),
        Err(AutodiffError::InvalidArgument(_))
    ));

    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let before = model.evaluate(&[&xa, &xc], &[&y1, &y2], 4).unwrap();
    let history = model
        .fit(&[&xa, &xc], &[&y1, &y2], FitConfig::new(60, 4))
        .unwrap();
    assert_eq!(history.loss.len(), 60);
    assert!(history.loss.iter().all(|l| l.is_finite()));
    assert!(
        history.loss.last().unwrap() < history.loss.first().unwrap(),
        "loss が下がるはず: {:?}",
        history.loss
    );
    let after = model.evaluate(&[&xa, &xc], &[&y1, &y2], 4).unwrap();
    assert!(
        after < before,
        "evaluate が改善するはず: {before} -> {after}"
    );
}

#[test]
fn trainable_parameters_and_apply_parameters_round_trip_atomically() {
    let mut model = two_in_two_out();
    let originals: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();
    let shifted: Vec<Tensor<f32>> = originals
        .iter()
        .map(|p| tensor(values(p).iter().map(|v| v + 0.5).collect(), p.shape()))
        .collect();
    model.apply_parameters(shifted.clone()).unwrap();
    for (p, s) in model.trainable_parameters().into_iter().zip(&shifted) {
        assert_eq!(bits(p), bits(s));
    }
    // shape 違いは拒否され、状態は変わらない（原子性）。
    let mut bad = shifted.clone();
    bad[0] = tensor(vec![0.0; 5], &[5]);
    assert!(model.apply_parameters(bad).is_err());
    for (p, s) in model.trainable_parameters().into_iter().zip(&shifted) {
        assert_eq!(bits(p), bits(s));
    }
    // 個数違いも拒否される。
    assert!(model.apply_parameters(shifted[..2].to_vec()).is_err());
}

#[test]
fn state_dict_keys_are_global_layer_indices_and_load_state_dict_restores() {
    let mut model = two_in_two_out();
    let sd = model.state_dict().unwrap();
    // 通し番号は全ブロックを通した層の並び（Linear・ReLU・Linear・ReLU・Linear・Linear）で、
    // パラメータを持つ層 0・2・4・5 だけがキーを持つ。
    assert_eq!(sd.len(), 8, "{:?}", sd.keys());
    for i in [0, 2, 4, 5] {
        assert!(sd.contains_key(&format!("{i}.weight")), "{:?}", sd.keys());
        assert!(sd.contains_key(&format!("{i}.bias")), "{:?}", sd.keys());
    }
    let snapshot = sd.clone();
    let shifted: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .into_iter()
        .map(|p| tensor(values(p).iter().map(|v| v + 1.0).collect(), p.shape()))
        .collect();
    model.apply_parameters(shifted).unwrap();
    assert_ne!(
        bits(&model.state_dict().unwrap()["0.weight"]),
        bits(&snapshot["0.weight"])
    );
    model.load_state_dict(snapshot.clone()).unwrap();
    for (k, v) in &snapshot {
        assert_eq!(bits(&model.state_dict().unwrap()[k]), bits(v), "{k}");
    }
    // 欠落キーは拒否される。
    let mut partial = snapshot;
    partial.remove("0.weight");
    assert!(model.load_state_dict(partial).is_err());
}

#[test]
fn set_training_propagates_to_all_blocks() {
    let mut model = two_in_two_out();
    assert!(model.training());
    model.eval();
    assert!(!model.training());
    model.train();
    assert!(model.training());
    model.set_training(false);
    assert!(!model.training());
}

// ---------------------------------------------------------------------
// 拒否系（panic せず型付きエラー）
// ---------------------------------------------------------------------

#[test]
fn build_and_merge_reject_invalid_wiring_without_panicking() {
    // 空の入力・出力。
    let b = FunctionalBuilder::new();
    assert!(b.build(&[], &[]).is_err());
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    assert!(b.build(&[x], &[]).is_err());

    // 結合ノードは入力 2 件以上・重複なし。
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    assert!(b.add(&[x]).is_err());
    assert!(b.add(&[x, x]).is_err());
    assert!(b.concatenate(&[], 1).is_err());

    // 他ビルダーのハンドルは拒否される。
    let mut other = FunctionalBuilder::new();
    let foreign: Node = other.input().unwrap();
    let mut b = FunctionalBuilder::new();
    let own = b.input().unwrap();
    assert!(b.apply(Sequential::new().add_relu(), foreign).is_err());
    assert!(b.add(&[own, foreign]).is_err());
    assert!(b.build(&[foreign], &[own]).is_err());

    // どの出力にも寄与しないノード・未束縛の入力ノードは build が拒否する。
    let mut b = FunctionalBuilder::new();
    let a = b.input().unwrap();
    let c = b.input().unwrap();
    let y = b.apply(Sequential::new().add_relu(), a).unwrap();
    assert!(b.build(&[a], &[y]).is_err(), "未束縛の入力ノード");
    let mut b = FunctionalBuilder::new();
    let a = b.input().unwrap();
    let _dangling = b.apply(Sequential::new().add_relu(), a).unwrap();
    let y = b.apply(Sequential::new().add_relu(), a).unwrap();
    assert!(b.build(&[a], &[y]).is_err(), "出力に寄与しないノード");
    let _ = c;
}

#[test]
fn forward_rejects_wrong_input_count_and_compile_rejects_lbfgs() {
    let model = chain(seq());
    let x = det(2, 4, 0.0);
    assert!(model.predict(&[]).is_err());
    assert!(model.predict(&[&x, &x]).is_err());

    let mut model = chain(seq());
    assert!(matches!(
        model.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // 入力件数・目標件数の不一致は fit が拒否する（compile 済みでも）。
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let y = det(2, 2, 0.0);
    assert!(model.fit(&[&x, &x], &[&y], FitConfig::new(1, 2)).is_err());
    assert!(model.fit(&[&x], &[&y, &y], FitConfig::new(1, 2)).is_err());
}
