//! `LeNet`（参照モデル定義。イシュー #2201・親 #2190）の統合テスト。
//!
//! `crates/facade/examples/models/lenet.rs` を `#[path]` で直接取り込み、
//! `crates/facade/tests/example_mlp_mnist.rs` と同型の観点（構造・
//! forward・学習）を検証する。合成 MNIST 相当データ
//! （`example_models_common`）を使う（実 MNIST ではない）。

mod example_models_common;
#[path = "../examples/models/lenet.rs"]
mod lenet;

use example_models_common::{IMAGE_LEN, NUM_CLASSES, synthetic_mnist_flat};
use fandhe_ai::compat::{FitConfig, Loss, Optimizer};
use fandhe_ai::optim::AdamConfig;
use fandhe_ai::{AutodiffError, Tensor};
use lenet::LeNet;

fn synthetic_batch(n: usize, seed: u64) -> (Tensor<f32>, Tensor<i32>) {
    let (data, labels) = synthetic_mnist_flat(n, seed);
    let x = Tensor::new(data, &[n, 1, 28, 28]).expect("shape とデータ長は一致させている");
    let y = Tensor::new(labels, &[n]).expect("shape とデータ長は一致させている");
    (x, y)
}

// ---------------------------------------------------------------------
// AC2・AC5: 構造（Conv2d 2 層 + Dense 2 層）と PyTorch 対応表。
// ---------------------------------------------------------------------

#[test]
fn lenet_structure_matches_pytorch_reference() {
    let model = LeNet::new(NUM_CLASSES, 1).unwrap();
    assert_eq!(model.num_classes(), NUM_CLASSES);

    let named = model.sequential().named_parameters();
    let map = model.pytorch_param_map().unwrap();
    assert_eq!(named.len(), map.len());
    assert_eq!(
        named.len(),
        8,
        "Conv2d 2 層・Dense 2 層 = 8 パラメータテンソル"
    );

    for ((name, tensor), entry) in named.iter().zip(map.iter()) {
        assert_eq!(name, &entry.fandhe_key);
        assert_eq!(tensor.shape(), entry.fandhe_shape.as_slice());
    }

    let expected: &[(&str, &[usize], bool)] = &[
        ("conv1.weight", &[6, 1, 5, 5], false),
        ("conv1.bias", &[6], false),
        ("conv2.weight", &[16, 6, 5, 5], false),
        ("conv2.bias", &[16], false),
        ("fc1.weight", &[120, 256], true),
        ("fc1.bias", &[120], false),
        ("fc2.weight", &[10, 120], true),
        ("fc2.bias", &[10], false),
    ];
    for (entry, (pytorch_key, pytorch_shape, transpose)) in map.iter().zip(expected.iter()) {
        assert_eq!(&entry.pytorch_key, pytorch_key);
        assert_eq!(entry.pytorch_shape.as_slice(), *pytorch_shape);
        assert_eq!(entry.transpose, *transpose);
    }
}

// ---------------------------------------------------------------------
// AC3 代替: forward・predict の実行と shape・有限性。
// ---------------------------------------------------------------------

#[test]
fn lenet_forward_shape_and_eval_determinism() {
    let mut model = LeNet::new(NUM_CLASSES, 3).unwrap();
    model.sequential_mut().eval();
    let (x, _labels) = synthetic_batch(4, 321);

    let via_predict = model.predict(&x).unwrap();
    assert_eq!(via_predict.shape(), &[4, NUM_CLASSES]);
    let flat = via_predict
        .contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず Some")
        .to_vec();
    assert!(flat.iter().all(|v| v.is_finite()));

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = model.forward(&tape, &xv).unwrap();
    assert_eq!(
        via_forward.to_tensor().contiguous().as_slice().unwrap(),
        flat.as_slice(),
        "predict（tape 不要経路）と forward（tape 経由）は bit 完全一致する契約"
    );
}

#[test]
fn lenet_rejects_rank_mismatch_and_invalid_num_classes() {
    assert!(matches!(
        LeNet::new(0, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));

    let model = LeNet::new(NUM_CLASSES, 5).unwrap();
    // rank 3（[N, 28, 28]。channel 軸が欠けている）は forward 時に Err
    // になる（panic しない）。
    let bad = Tensor::new(vec![0.0f32; 4 * IMAGE_LEN], &[4, 28, 28]).unwrap();
    assert!(model.predict(&bad).is_err());
}

// ---------------------------------------------------------------------
// AC4: 事前登録した判定式（1 epoch 後 loss が学習前の 50% 以下）。
// ---------------------------------------------------------------------

#[test]
fn lenet_synthetic_mnist_one_epoch_loss_halves() {
    // LeNet は Dropout を持たず `FitConfig::shuffle` も既定 false のため
    // 学習経路はグローバル RNG を消費しない（決定的。他テストとの直列化は
    // 不要）。
    let mut model = LeNet::new(NUM_CLASSES, 11).unwrap();
    model
        .sequential_mut()
        .compile(
            Optimizer::Adam(AdamConfig {
                lr: 3e-3,
                ..AdamConfig::default()
            }),
            Loss::CrossEntropy,
        )
        .unwrap();

    let (x, y) = synthetic_batch(1024, 77);
    let batch_size = 16;

    let before = model.sequential_mut().evaluate(&x, &y, batch_size).unwrap();
    let history = model
        .sequential_mut()
        .fit(&x, &y, FitConfig::new(1, batch_size))
        .unwrap();
    let after = model.sequential_mut().evaluate(&x, &y, batch_size).unwrap();

    assert!(before.is_finite(), "before loss は有限であること: {before}");
    assert!(after.is_finite(), "after loss は有限であること: {after}");
    assert!(
        history.loss[0].is_finite(),
        "History.loss[0] は有限であること: {}",
        history.loss[0]
    );
    assert!(
        after <= 0.5 * before,
        "1 epoch 後の loss は学習前の 50% 以下であること（事前登録した判定式。\
         before={before}, after={after}）"
    );
}

// ---------------------------------------------------------------------
// codex-review 指摘（イシュー #2201 PR #2320）: pytorch_param_map の
// 整合検査（mlp.rs 側と同じ観点。LeNet には dropout 引数がないため
// 検証観点は sequential_mut() 経由の不整合検出のみ）。
// ---------------------------------------------------------------------

#[test]
fn lenet_pytorch_param_map_detects_sequential_mut_replacement() {
    let mut model = LeNet::new(NUM_CLASSES, 1).unwrap();
    // sequential_mut() 経由で内部 Sequential を全く別の構成へ差し替える
    // と、構成値から計算した対応表と実パラメータが不整合になる。
    // pytorch_param_map() はこれを検出して Err を返すこと。
    *model.sequential_mut() = fandhe_ai::compat::Sequential::new();
    assert!(matches!(
        model.pytorch_param_map(),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
