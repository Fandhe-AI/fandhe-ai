//! `Mlp`（参照モデル定義。イシュー #2201・親 #2190）の統合テスト。
//!
//! `crates/facade/examples/models/mlp.rs` を `#[path]` で直接取り込み、
//! `crates/facade/examples/reference_models.rs`（runnable example）とは
//! 別に、公開 API（`compat::Sequential` 経由）だけで組んだ MLP の構造・
//! forward・学習が成立することを検証する。
//!
//! 合成 MNIST 相当データ（`example_models_common`）を使う。実 MNIST では
//! ない（`docs/reference-models-decision.md` 参照）。

mod example_models_common;
#[path = "../examples/models/mlp.rs"]
mod mlp;

use std::sync::{Mutex, OnceLock};

use example_models_common::{IMAGE_LEN, NUM_CLASSES, synthetic_mnist_flat};
use fandhe_ai::compat::{FitConfig, Loss, Optimizer};
use fandhe_ai::optim::AdamConfig;
use fandhe_ai::{AutodiffError, Tensor};
use mlp::Mlp;

/// グローバル RNG（`fandhe_ai::manual_seed`／Dropout のマスク抽選）を
/// 消費するテストを直列化する（先例:
/// `crates/facade/tests/mnist_scale_train_reuse_bench.rs`）。
fn rng_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

fn synthetic_batch(n: usize, seed: u64) -> (Tensor<f32>, Tensor<i32>) {
    let (data, labels) = synthetic_mnist_flat(n, seed);
    let x = Tensor::new(data, &[n, IMAGE_LEN]).expect("shape とデータ長は一致させている");
    let y = Tensor::new(labels, &[n]).expect("shape とデータ長は一致させている");
    (x, y)
}

// ---------------------------------------------------------------------
// AC1・AC5: 構成と PyTorch 対応表。
// ---------------------------------------------------------------------

#[test]
fn mlp_structure_matches_pytorch_reference() {
    let model = Mlp::new(IMAGE_LEN, &[256, 128], NUM_CLASSES, 0.2).unwrap();
    assert_eq!(model.dropout(), 0.2);

    let named = model.sequential().named_parameters();
    let map = model.pytorch_param_map();

    // named_parameters() は weight → bias の層順（モジュール doc の順序
    // 契約）。対応表も同じ順で weight/bias を積んでいるため、そのまま
    // 突き合わせられる。
    assert_eq!(named.len(), map.len());
    let expected_pytorch_shapes: &[(&str, &[usize])] = &[
        ("0.weight", &[256, 784]),
        ("0.bias", &[256]),
        ("3.weight", &[128, 256]),
        ("3.bias", &[128]),
        ("6.weight", &[10, 128]),
        ("6.bias", &[10]),
    ];
    assert_eq!(map.len(), expected_pytorch_shapes.len());

    for ((name, tensor), entry) in named.iter().zip(map.iter()) {
        assert_eq!(name, &entry.fandhe_key);
        assert_eq!(tensor.shape(), entry.fandhe_shape.as_slice());
    }
    for (entry, (expected_key, expected_shape)) in map.iter().zip(expected_pytorch_shapes.iter()) {
        assert_eq!(&entry.pytorch_key, expected_key);
        assert_eq!(entry.pytorch_shape.as_slice(), *expected_shape);
    }
    // weight は転置あり・bias は転置なし（モジュール doc「重みレイアウトの
    // 契約」節）。
    for entry in &map {
        assert_eq!(entry.transpose, entry.fandhe_key.ends_with(".weight"));
    }
}

// ---------------------------------------------------------------------
// AC3 代替: forward・predict の実行と shape・有限性・決定性。
// ---------------------------------------------------------------------

#[test]
fn mlp_forward_shape_and_eval_determinism() {
    let _guard = rng_guard().lock().unwrap();
    fandhe_ai::manual_seed(7);

    let mut model = Mlp::new(IMAGE_LEN, &[256, 128], NUM_CLASSES, 0.2).unwrap();
    model.sequential_mut().eval();
    let (x, _labels) = synthetic_batch(4, 123);

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
        "eval モードでは predict（tape 不要経路）と forward（tape 経由）が \
         bit 完全一致する契約（compat::Sequential::forward/predict と同じ）"
    );
}

// ---------------------------------------------------------------------
// 無効引数（fail-closed。panic しない）。
// ---------------------------------------------------------------------

#[test]
fn mlp_rejects_invalid_args() {
    assert!(matches!(
        Mlp::new(0, &[256], 10, 0.2),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        Mlp::new(784, &[0, 128], 10, 0.2),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        Mlp::new(784, &[256], 0, 0.2),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // dropout の範囲・有限性検査は `Dropout::new`（`add_dropout`）へ
    // 委譲する契約（モジュール doc 参照）。
    assert!(Mlp::new(784, &[256], 10, 1.5).is_err());
    assert!(Mlp::new(784, &[256], 10, f32::NAN).is_err());
    // `with_seed`（明示シード版）も同じ検査を経由する。
    assert!(matches!(
        Mlp::with_seed(0, &[256], 10, 0.2, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------
// AC4: 事前登録した判定式（1 epoch 後 loss が学習前の 50% 以下）。
// ---------------------------------------------------------------------

#[test]
fn mlp_synthetic_mnist_one_epoch_loss_halves() {
    let _guard = rng_guard().lock().unwrap();
    fandhe_ai::manual_seed(42);

    let mut model = Mlp::new(IMAGE_LEN, &[256, 128], NUM_CLASSES, 0.2).unwrap();
    model
        .sequential_mut()
        .compile(
            Optimizer::Adam(AdamConfig {
                lr: 8e-3,
                ..AdamConfig::default()
            }),
            Loss::CrossEntropy,
        )
        .unwrap();

    let (x, y) = synthetic_batch(256, 99);
    let batch_size = 32;

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
