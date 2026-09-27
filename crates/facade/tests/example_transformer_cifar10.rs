//! `Transformer`（参照モデル定義。イシュー #2202・親 #2190）の統合テスト。
//!
//! `crates/facade/examples/models/{transformer,reference_module,
//! synthetic_cifar}.rs` を `#[path]` で直接取り込み、`example_resnet_
//! cifar10.rs` と同型の観点（構成・forward・学習）を検証する。
//!
//! `mod` 識別子は `main.rs`／`example_resnet_cifar10.rs` と同じ名前
//! （`reference_module`・`synthetic_cifar`）で宣言する（`transformer.rs`
//! 内部の `super::reference_module::…` 参照を解決するための契約。
//! `resnet.rs` モジュール doc「位置づけ」節参照）。

#[path = "../examples/models/reference_module.rs"]
mod reference_module;
#[path = "../examples/models/synthetic_cifar.rs"]
mod synthetic_cifar;
#[path = "../examples/models/transformer.rs"]
mod transformer;

use std::sync::{Mutex, OnceLock};

use fandhe_ai::optim::{Adam, AdamConfig};
use fandhe_ai::{AutodiffError, Tensor};
use reference_module::{ReferenceModule, accuracy, fit_epochs, sub_tensor_f32};
use synthetic_cifar::{IMG_C, IMG_H, IMG_W, NUM_CLASSES, synthetic_cifar10, to_row_tokens};
use transformer::{Transformer, TransformerConfig};

/// `example_resnet_cifar10.rs::rng_guard` と同じ直列化ガード
/// （実害はない現状でも将来のグローバル RNG 依存に備える）。
fn rng_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

/// `main.rs`・`example_resnet_cifar10.rs` と同じ局所 PRNG（SplitMix64）。
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn token_tensor(flat: Vec<f32>, n: usize) -> Tensor<f32> {
    Tensor::new(flat, &[n, IMG_H, IMG_C * IMG_W]).expect("shape とデータ長は一致させている")
}

fn labels_tensor(labels: Vec<i32>, n: usize) -> Tensor<i32> {
    Tensor::new(labels, &[n]).expect("shape とデータ長は一致させている")
}

fn synthetic_tokens(n: usize, seed: u64) -> (Tensor<f32>, Tensor<i32>) {
    let mut rng = SplitMix64(seed);
    let mut src = || rng.next_u64();
    let (flat, labels) = synthetic_cifar10(n, &mut src);
    (
        token_tensor(to_row_tokens(&flat, n), n),
        labels_tensor(labels, n),
    )
}

// ---------------------------------------------------------------------
// AC1: 構成（embed_dim・num_heads・num_layers・num_classes）。
// ---------------------------------------------------------------------

#[test]
fn transformer_config_cifar10_preset() {
    let config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    assert_eq!(config.seq_len, IMG_H);
    assert_eq!(config.in_features, IMG_C * IMG_W);
    assert_eq!(config.embed_dim, 16);
    assert_eq!(config.num_heads, 2);
    assert_eq!(config.num_layers, 1);
    assert_eq!(config.dim_feedforward, 32);
    assert_eq!(config.mlp_hidden, 32);
    assert_eq!(config.num_classes, NUM_CLASSES);

    let model = Transformer::new(config, 0x1234_5678).unwrap();
    assert_eq!(model.config().embed_dim, 16);
}

#[test]
fn transformer_rejects_invalid_args() {
    let mut config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    // embed_dim が num_heads で割り切れない。
    let mut bad = config;
    bad.num_heads = 3;
    assert!(matches!(
        Transformer::new(bad, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));

    // 各フィールド 0 は拒否する。
    config.seq_len = 0;
    assert!(matches!(
        Transformer::new(config, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));

    let mut config2 = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    config2.num_classes = 0;
    assert!(matches!(
        Transformer::new(config2, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn transformer_rejects_seq_len_times_embed_dim_overflow() {
    // `TransformerConfig` の各フィールドは pub のため、`cifar10` を
    // 経由しない直接構築でも `seq_len * embed_dim`
    // （位置符号テンソルの要素数）のオーバーフローを `Transformer::new`
    // が検出する必要がある（Codex レビュー指摘・イシュー #2202
    // PR #2325。`ResNet::new` の `width * 2`／`width * 4` と同型の
    // 横展開）。
    let mut config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    config.embed_dim = usize::MAX / 4;
    config.num_heads = 1;
    assert!(matches!(
        Transformer::new(config, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------
// AC3 代替: `Transformer` が `ReferenceModule` を実装すること
// （named_parameters の階層名・forward の shape・有限性）。
// ---------------------------------------------------------------------

#[test]
fn transformer_implements_reference_module() {
    let config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    let model = Transformer::new(config, 0x2468_ACE0).unwrap();
    let params = ReferenceModule::named_parameters(&model);
    assert!(params.iter().any(|(n, _)| n.starts_with("embed.")));
    assert!(params.iter().any(|(n, _)| n.starts_with("encoder.")));
    assert!(params.iter().any(|(n, _)| n.starts_with("head.")));
}

#[test]
fn transformer_predict_shape_and_eval_determinism() {
    let (x, _y) = synthetic_tokens(4, 0xAAAA_1111_BBBB_2222);
    let config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    let mut model = Transformer::new(config, 0x1357_9BDF).unwrap();
    ReferenceModule::set_training(&mut model, false);

    let via_predict = model.predict(&x).unwrap();
    assert_eq!(via_predict.shape(), &[4, NUM_CLASSES]);
    let flat_out = via_predict
        .contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず Some")
        .to_vec();
    assert!(flat_out.iter().all(|v| v.is_finite()));

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = ReferenceModule::forward(&model, &tape, &xv).unwrap();
    assert_eq!(
        via_forward.to_tensor().contiguous().as_slice().unwrap(),
        flat_out.as_slice(),
        "eval モードでは predict と forward が bit 完全一致する契約\
         （example_resnet_cifar10.rs と同じ検証）"
    );
}

#[test]
fn transformer_forward_rejects_wrong_input_shape() {
    let config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    let model = Transformer::new(config, 0x1111_2222).unwrap();
    // [N, seq_len, in_features] ではなく rank 2 を渡す。
    let bad_x = Tensor::new(vec![0.0f32; 4 * 16], &[4, 16]).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&bad_x);
    assert!(matches!(
        ReferenceModule::forward(&model, &tape, &xv),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------
// AC4: 10 epoch 学習後の held-out 精度 50% 以上（事前登録した判定式。
// main.rs::check_ac4 と同じ判定式・係数。テストは debug ビルドの実行
// 時間予算に収めるため main.rs より小さい構成
// （embed_dim=16・heads=2・layers=1・N を縮小）を使う。epochs（10）・
// 判定係数（0.50）は main.rs と揃えて変更しない（調整してよいのは
// データ設計・N・batch size・学習率のみという
// `docs/reference-models-decision.md` §7 の方針を踏襲）。
// ---------------------------------------------------------------------

#[test]
fn transformer_synthetic_cifar10_ten_epochs_reaches_50_percent_accuracy() {
    let _guard = rng_guard().lock().unwrap();

    const EPOCHS: usize = 10;
    const BATCH_SIZE: usize = 8;
    const N_TRAIN: usize = 32;
    const N_TEST: usize = 16;
    const LR: f32 = 5e-3;

    let (x_train, y_train) = synthetic_tokens(N_TRAIN, 0xABCD_EF01_2345_6789);
    let (x_test, y_test) = synthetic_tokens(N_TEST, 0x1357_9BDF_2468_ACE0);

    let config = TransformerConfig::cifar10(16, 2, 1, NUM_CLASSES);
    let mut model = Transformer::new(config, 0x7777_7777).unwrap();
    let sample_batch = sub_tensor_f32(&x_train, 0, BATCH_SIZE).unwrap();
    let sample_pred = model.predict(&sample_batch).unwrap();
    assert_eq!(sample_pred.shape(), &[BATCH_SIZE, NUM_CLASSES]);

    let mut opt = Adam::new(AdamConfig {
        lr: LR,
        ..AdamConfig::default()
    })
    .unwrap();

    let history = fit_epochs(&mut model, &x_train, &y_train, &mut opt, EPOCHS, BATCH_SIZE).unwrap();
    let test_acc = accuracy(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES).unwrap();

    assert!(
        history.iter().all(|v| v.is_finite()),
        "学習中に非有限の loss が発生した: {history:?}"
    );
    assert!(
        history.last().copied().unwrap_or(f32::INFINITY) < history[0],
        "最終 epoch の loss が初回 epoch の loss を下回らなかった: {history:?}"
    );
    assert!(
        test_acc >= 0.5,
        "held-out 精度が AC4 の下限 0.50 を下回った（実測: {test_acc:.4}）"
    );
}
