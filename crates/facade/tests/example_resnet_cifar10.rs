//! `ResNet`（参照モデル定義。イシュー #2202・親 #2190）の統合テスト。
//!
//! `crates/facade/examples/models/{resnet,reference_module,
//! synthetic_cifar}.rs` を `#[path]` で直接取り込み、
//! `crates/facade/examples/main.rs`（runnable example）とは別に、公開
//! API（`fandhe_ai::…`）だけで組んだ ResNet の構造・shortcut 分岐・
//! forward・学習が成立することを検証する（`example_mlp_mnist.rs`・
//! `example_lenet_mnist.rs` と同型の観点。#2201 先例）。
//!
//! 合成 CIFAR-10 相当データ（`synthetic_cifar`）を使う。実 CIFAR-10 では
//! ない（`docs/reference-models-decision.md` #2202 節参照）。
//!
//! `mod` 識別子は `main.rs` と同じ名前（`reference_module`・
//! `synthetic_cifar`）で宣言する。`resnet.rs` 内部の `super::
//! reference_module::…` 参照がどちらの取り込み元でも解決できるようにする
//! ための契約（`resnet.rs` モジュール doc「位置づけ」節参照）。

#[path = "../examples/models/reference_module.rs"]
mod reference_module;
#[path = "../examples/models/resnet.rs"]
mod resnet;
#[path = "../examples/models/synthetic_cifar.rs"]
mod synthetic_cifar;

use std::sync::{Mutex, OnceLock};

use fandhe_ai::optim::{Adam, AdamConfig};
use fandhe_ai::{AutodiffError, Tensor};
use reference_module::{ReferenceModule, Trainable, accuracy, fit_epochs, sub_tensor_f32};
use resnet::{ResNet, ResNetBlock};
use synthetic_cifar::{IMG_C, IMG_H, IMG_W, NUM_CLASSES, synthetic_cifar10, to_row_tokens};

/// グローバル状態は使わないが、学習系テストは決定的シード運用の
/// ルールを揃えるため他 example テストと同じ直列化ガードを踏襲する
/// （`example_mlp_mnist.rs::rng_guard` と同じ考え方。本テストの乱数は
/// すべて `SplitMix64` ローカル状態のみで完結するため実害はないが、
/// 将来 `fandhe_ai::manual_seed` を使う変更が入っても安全なように残す）。
fn rng_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

/// `main.rs` と同じ依存追加なしの局所 PRNG（SplitMix64）。
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

fn image_tensor(flat: Vec<f32>, n: usize) -> Tensor<f32> {
    Tensor::new(flat, &[n, IMG_C, IMG_H, IMG_W]).expect("shape とデータ長は一致させている")
}

fn labels_tensor(labels: Vec<i32>, n: usize) -> Tensor<i32> {
    Tensor::new(labels, &[n]).expect("shape とデータ長は一致させている")
}

// ---------------------------------------------------------------------
// AC1: 構成（depth・width・num_classes）と shortcut branch。
// ---------------------------------------------------------------------

#[test]
fn resnet_structure_and_shortcut_branches() {
    // depth=8 は 6n+2 の最小構成（n=1）: 3 ステージ x 1 block = 3 block。
    // stage2・stage3 の先頭 block だけ in/out チャネルが変わるため
    // projection shortcut を持ち、stage1 は identity shortcut。
    let model = ResNet::new(8, 4, NUM_CLASSES, 0x1111_2222).unwrap();
    assert_eq!(model.depth(), 8);
    assert_eq!(model.width(), 4);
    assert_eq!(model.num_classes(), NUM_CLASSES);
    assert_eq!(model.num_blocks(), 3);

    let projections: Vec<bool> = model
        .blocks()
        .iter()
        .map(ResNetBlock::has_projection_shortcut)
        .collect();
    assert_eq!(
        projections,
        vec![false, true, true],
        "stage1 は identity・stage2/3 先頭は projection shortcut（6n+2 構成。\
         resnet.rs モジュール doc 参照）"
    );

    // depth=14（n=2）: 3 ステージ x 2 block = 6 block。各ステージの
    // 2 block 目は stride=1・チャネル不変のため identity。
    let model14 = ResNet::new(14, 4, NUM_CLASSES, 0x3333_4444).unwrap();
    assert_eq!(model14.num_blocks(), 6);
    let projections14: Vec<bool> = model14
        .blocks()
        .iter()
        .map(ResNetBlock::has_projection_shortcut)
        .collect();
    assert_eq!(projections14, vec![false, false, true, false, true, false]);
}

#[test]
fn resnet_rejects_invalid_args() {
    assert!(matches!(
        ResNet::new(8, 0, NUM_CLASSES, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        ResNet::new(8, 4, 0, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // depth は 6n+2 でなければならない（7 は該当しない）。
    assert!(matches!(
        ResNet::new(7, 4, NUM_CLASSES, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // depth < 8（n < 1）も拒否する。
    assert!(matches!(
        ResNet::new(2, 4, NUM_CLASSES, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));

    assert!(matches!(
        ResNetBlock::new(0, 4, 1, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        ResNetBlock::new(4, 4, 0, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn resnet_rejects_width_multiplication_overflow() {
    // width * 4（stage3 の出力チャネル数）が usize をオーバーフローする
    // width を渡すと、`checked_mul` が `InvalidArgument` を返す
    // （Codex レビュー指摘・イシュー #2202 PR #2325）。
    let huge_width = usize::MAX / 3;
    assert!(matches!(
        ResNet::new(8, huge_width, NUM_CLASSES, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------
// AC3 代替: `ResNetBlock`・`ResNet` の両方が `ReferenceModule` を実装
// すること（block 単体の named_parameters・forward も検証）。
// ---------------------------------------------------------------------

#[test]
fn resnet_block_and_model_implement_reference_module() {
    let block = ResNetBlock::new(4, 8, 2, 0x5555_6666).unwrap();
    assert!(block.has_projection_shortcut());
    let block_params = ReferenceModule::named_parameters(&block);
    // main（conv(weight+bias)+BN(weight+bias) を 2 段）8 +
    // shortcut（conv(weight+bias)+BN(weight+bias)）4 = 12 パラメータ
    // テンソル（`nn::Conv2d`／`nn::BatchNorm2d` はいずれも
    // weight・bias の 2 テンソルを持つ契約。running_mean／running_var は
    // `named_parameters` に含めない。`crates/autodiff/src/nn/
    // batch_norm.rs` 参照）。
    assert_eq!(block_params.len(), 12);
    assert!(
        block_params
            .iter()
            .any(|(name, _)| name.starts_with("main.")),
        "main 経路のパラメータ名は main. プレフィックスを持つ"
    );
    assert!(
        block_params
            .iter()
            .any(|(name, _)| name.starts_with("shortcut.")),
        "projection shortcut を持つ block は shortcut. プレフィックスのパラメータも持つ"
    );

    let model = ResNet::new(8, 4, NUM_CLASSES, 0x7777_8888).unwrap();
    let model_params = ReferenceModule::named_parameters(&model);
    assert!(model_params.iter().any(|(n, _)| n.starts_with("stem.")));
    assert!(model_params.iter().any(|(n, _)| n.starts_with("layer.")));
    assert!(model_params.iter().any(|(n, _)| n.starts_with("head.")));
}

#[test]
fn resnet_predict_shape_and_eval_determinism() {
    let mut rng = SplitMix64(0xAAAA_BBBB_CCCC_DDDD);
    let mut src = || rng.next_u64();
    let (flat, _labels) = synthetic_cifar10(4, &mut src).unwrap();
    let x = image_tensor(flat, 4);

    let mut model = ResNet::new(8, 4, NUM_CLASSES, 0x9999_AAAA).unwrap();
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
         （example_mlp_mnist.rs と同じ検証）"
    );
}

// ---------------------------------------------------------------------
// `synthetic_cifar::to_row_tokens`（Transformer 側の行トークン化）の
// 形状・並べ替え検証。ResNet テストは行トークン化を使わないが、
// 共有ユーティリティファイル（synthetic_cifar.rs）を取り込む都合上、
// いずれかの取り込み元で to_row_tokens をテストしておく
// （dead_code を避けるための都合ではなく、両テストで共有する
// 変換ロジックを一箇所で検証する目的）。
// ---------------------------------------------------------------------

#[test]
fn to_row_tokens_reorders_chw_to_hcw() {
    let n = 2;
    // [N, C, H, W] = [2, 3, 32, 32] の連番データ。
    let flat: Vec<f32> = (0..n * IMG_C * IMG_H * IMG_W).map(|i| i as f32).collect();
    let rows = to_row_tokens(&flat, n).unwrap();
    assert_eq!(rows.len(), n * IMG_H * IMG_C * IMG_W);

    // 元 index (ni, c, h, w) の値は変わらず、並び順だけ (ni, h, c, w) に
    // なることを何点かサンプル検証する。
    for &(ni, c, h, w) in &[
        (0usize, 0usize, 0usize, 0usize),
        (1, 2, 31, 5),
        (0, 1, 10, 20),
    ] {
        let src = ((ni * IMG_C + c) * IMG_H + h) * IMG_W + w;
        let dst = ((ni * IMG_H + h) * IMG_C + c) * IMG_W + w;
        assert_eq!(rows[dst], flat[src]);
    }
}

#[test]
fn to_row_tokens_rejects_length_mismatch() {
    let n = 2;
    let expected = n * IMG_C * IMG_H * IMG_W;

    // 過多（release でも余剰を黙って捨てず検出する。Codex レビュー
    // 指摘・イシュー #2202 PR #2325）。
    let too_long: Vec<f32> = vec![0.0f32; expected + 1];
    assert!(to_row_tokens(&too_long, n).is_err());

    // 不足（従来は index out of bounds で panic していた）。
    let too_short: Vec<f32> = vec![0.0f32; expected - 1];
    assert!(to_row_tokens(&too_short, n).is_err());
}

#[test]
fn synthetic_cifar10_rejects_element_count_overflow() {
    // n * IMG_C * IMG_H * IMG_W が usize をオーバーフローする n を渡すと
    // `InvalidArgument` を返す（Codex レビュー指摘・イシュー #2202
    // PR #2325）。
    let huge_n = usize::MAX / (IMG_C * IMG_H * IMG_W) + 1;
    let mut rng = SplitMix64(1);
    let mut src = || rng.next_u64();
    assert!(synthetic_cifar10(huge_n, &mut src).is_err());
}

// ---------------------------------------------------------------------
// reference_module.rs の入力契約検証（イシュー #2202 PR #2325 レビュー
// 指摘の横展開。`resnet.rs`・`main.rs` の取り込み元に依存しない
// `reference_module` 単体の契約検証のため、共有ファイルの契約検証を
// 一箇所に集める `to_row_tokens` と同じ方針でここに置く）。
// ---------------------------------------------------------------------

#[test]
fn accuracy_rejects_label_shape_mismatch() {
    use reference_module::accuracy;

    let mut model = ResNet::new(8, 4, NUM_CLASSES, 0x1234_0001).unwrap();
    let mut rng = SplitMix64(0x1234_0002);
    let mut src = || rng.next_u64();
    let (flat, _labels) = synthetic_cifar10(4, &mut src).unwrap();
    let x = image_tensor(flat, 4);
    // y の要素数が x の先頭軸長（4）より多い（余剰ラベル）。
    let y = labels_tensor(vec![0, 1, 2, 3, 4], 5);

    assert!(matches!(
        accuracy(&mut model, &x, &y, 4, NUM_CLASSES),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn accuracy_rejects_out_of_range_label() {
    use reference_module::accuracy;

    let mut model = ResNet::new(8, 4, NUM_CLASSES, 0x1234_0003).unwrap();
    let mut rng = SplitMix64(0x1234_0004);
    let mut src = || rng.next_u64();
    let (flat, _labels) = synthetic_cifar10(4, &mut src).unwrap();
    let x = image_tensor(flat, 4);
    // NUM_CLASSES 未満でなければならないラベルに範囲外の値を混ぜる。
    let y = labels_tensor(vec![0, 1, 2, NUM_CLASSES as i32], 4);

    assert!(matches!(
        accuracy(&mut model, &x, &y, 4, NUM_CLASSES),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn sub_tensor_i32_rejects_rank_2_input() {
    use reference_module::sub_tensor_i32;

    let y = Tensor::new(vec![0i32, 1, 2, 3], &[2, 2]).unwrap();
    assert!(matches!(
        sub_tensor_i32(&y, 0, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn sub_tensor_f32_and_i32_reject_start_plus_len_overflow() {
    use reference_module::{sub_tensor_f32, sub_tensor_i32};

    // start・len がいずれも usize::MAX 近傍で、素の `+` では usize を
    // オーバーフローして本来の境界検査（> n）を素通りしうる組合せ
    // （Codex レビュー指摘・イシュー #2202 PR #2325）。
    let x = image_tensor(vec![0.0f32; 4 * IMG_C * IMG_H * IMG_W], 4);
    assert!(matches!(
        sub_tensor_f32(&x, usize::MAX - 1, 2),
        Err(AutodiffError::InvalidArgument(_))
    ));

    let y = labels_tensor(vec![0, 1, 2, 3], 4);
    assert!(matches!(
        sub_tensor_i32(&y, usize::MAX - 1, 2),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

/// `ReferenceModule::forward` が要素数は一致するが shape が異なる
/// logits（`[1, len*num_classes]`）を返す偽モデル（`accuracy` の shape
/// 完全一致検査を、要素数一致だけでは検出できないケースとして再現する
/// ためのテスト専用 fixture。Codex レビュー指摘・イシュー #2202
/// PR #2325）。
struct FlatLogitsModel {
    num_classes: usize,
}

impl ReferenceModule for FlatLogitsModel {
    fn forward<'t>(
        &self,
        tape: &'t fandhe_ai::Tape,
        x: &fandhe_ai::Var<'t>,
    ) -> Result<fandhe_ai::Var<'t>, AutodiffError> {
        let batch = x.to_tensor().shape()[0];
        let data = vec![0.0f32; batch * self.num_classes];
        // 正しい [batch, num_classes] ではなく [1, batch*num_classes]
        // （要素数は同じだが shape が異なる）を返す。
        let t = Tensor::new(data, &[1, batch * self.num_classes]).unwrap();
        Ok(tape.var(&t))
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        Vec::new()
    }

    fn set_training(&mut self, _training: bool) {}
}

#[test]
fn accuracy_rejects_logits_shape_mismatch_with_matching_element_count() {
    use reference_module::accuracy;

    let mut model = FlatLogitsModel {
        num_classes: NUM_CLASSES,
    };
    let x = image_tensor(vec![0.0f32; 4 * IMG_C * IMG_H * IMG_W], 4);
    let y = labels_tensor(vec![0, 1, 2, 3], 4);

    assert!(matches!(
        accuracy(&mut model, &x, &y, 4, NUM_CLASSES),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn cross_entropy_mean_rejects_logits_column_mismatch() {
    use reference_module::cross_entropy_mean;

    let tape = fandhe_ai::tape();
    // targets は [0, NUM_CLASSES) の範囲内だが、logits の列数が
    // num_classes（NUM_CLASSES）と一致しない（NUM_CLASSES - 1 列）。
    let logits_data = vec![0.0f32; 4 * (NUM_CLASSES - 1)];
    let logits_tensor = Tensor::new(logits_data, &[4, NUM_CLASSES - 1]).unwrap();
    let logits = tape.var(&logits_tensor);
    let targets = labels_tensor(vec![0, 1, 0, 1], 4);

    assert!(matches!(
        cross_entropy_mean(&tape, &logits, &targets, NUM_CLASSES),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn resnet_train_step_rejects_wrong_input_shape() {
    let mut model = ResNet::new(8, 4, NUM_CLASSES, 0x1234_0005).unwrap();
    // [N, 3, H, W] ではなく rank 2 を渡す。
    let bad_x = Tensor::new(vec![0.0f32; 4 * 16], &[4, 16]).unwrap();
    let y = labels_tensor(vec![0, 1, 2, 3], 4);
    let mut opt = Adam::new(AdamConfig::default()).unwrap();

    assert!(matches!(
        model.train_step(&bad_x, &y, &mut opt),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------
// AC4: 10 epoch 学習後の held-out 精度 50% 以上（事前登録した判定式。
// main.rs::check_ac4 と同じ判定式・係数。テストは debug ビルドの実行
// 時間予算に収めるため main.rs より小さい構成（width=4・N を縮小）を
// 使う。epochs（10）・判定係数（0.50）は main.rs と揃えて変更しない
// （調整してよいのはデータ設計・N・batch size・学習率のみという
// `docs/reference-models-decision.md` §7 の方針を踏襲）。
// ---------------------------------------------------------------------

#[test]
fn resnet_synthetic_cifar10_ten_epochs_reaches_50_percent_accuracy() {
    let _guard = rng_guard().lock().unwrap();

    const EPOCHS: usize = 10;
    const BATCH_SIZE: usize = 8;
    const N_TRAIN: usize = 32;
    const N_TEST: usize = 16;
    const LR: f32 = 8e-3;

    let mut rng = SplitMix64(0xC0FF_EE00_1234_5678);
    let mut train_src = || rng.next_u64();
    let (train_flat, train_labels) = synthetic_cifar10(N_TRAIN, &mut train_src).unwrap();
    let mut rng_test = SplitMix64(0xFEED_BEEF_8765_4321);
    let mut test_src = || rng_test.next_u64();
    let (test_flat, test_labels) = synthetic_cifar10(N_TEST, &mut test_src).unwrap();

    let x_train = image_tensor(train_flat, N_TRAIN);
    let y_train = labels_tensor(train_labels, N_TRAIN);
    let x_test = image_tensor(test_flat, N_TEST);
    let y_test = labels_tensor(test_labels, N_TEST);

    let mut model = ResNet::new(8, 4, NUM_CLASSES, 0xD00D_D00D).unwrap();
    // 学習前の predict 疎通確認（main.rs run_resnet と同じ手順）。
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
