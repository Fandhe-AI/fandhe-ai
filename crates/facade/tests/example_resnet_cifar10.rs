//! `ResNet`（参照モデル。イシュー #2202・親 #2190。公開は #2975）の統合テスト。
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

use fandhe_ai::models::ResNetBlock;
use fandhe_ai::optim::{Adam, AdamConfig};
use fandhe_ai::{AutodiffError, Tensor};
use reference_module::{
    ReferenceModule, accuracy, fit_epochs, heldout_loss, predict_in_eval, sub_tensor_f32,
};
use resnet::ResNet;
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
    // ブロック単体の引数検証は src の単体テスト（`models::resnet::tests`）へ移した（#2975）。
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
    // block 単体は非公開 API のため、`ResNet::new(8, 4, ..)` の block 1（4→8・stride 2・
    // projection shortcut あり）を `layer.1.` 接頭辞で絞って検証する。
    let model_for_block = ResNet::new(8, 4, NUM_CLASSES, 0x5555_6666).unwrap();
    assert!(model_for_block.blocks()[1].has_projection_shortcut());
    let all_params = ReferenceModule::named_parameters(&model_for_block);
    let block_params: Vec<_> = all_params
        .iter()
        .filter(|(name, _)| name.starts_with("layer.1."))
        .collect();
    // main（conv(weight+bias)+BN(weight+bias) を 2 段）8 + shortcut 4 = 12 パラメータテンソル。
    assert_eq!(block_params.len(), 12);
    assert!(
        block_params
            .iter()
            .any(|(name, _)| name.starts_with("layer.1.main.")),
        "main 経路のパラメータ名は main. プレフィックスを持つ"
    );
    assert!(
        block_params
            .iter()
            .any(|(name, _)| name.starts_with("layer.1.shortcut.")),
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

/// `predict_in_eval` が BatchNorm の running stats を汚染しないことの
/// 正のプローブ（イシュー #2202 PR #2325 レビュー指摘）。
///
/// 単に「`predict_in_eval` 前後で eval 出力が一致する」だけでは、
/// そもそも train モードの forward が running stats に影響しないなら
/// 自明に成立してしまう（否定ガードの空振り）。そこで 3 体の同一構成
/// （同一 seed）モデルを用意し、
/// 1. `baseline`: 構築直後に一度も推論を行わず eval へ切り替えてから
///    predict する
/// 2. `polluted`: 修正前の bug パターン（構築直後・train モードのまま
///    `model.predict(x)` を直接呼ぶ）を再現したのち eval へ切り替えて
///    predict する
/// 3. `fixed`: `predict_in_eval`（shape 確認用の想定呼び出し）を挟んだ
///    のち eval へ切り替えて predict する
///
/// `baseline != polluted` を先に確認することで「train モードの predict
/// が running stats を実際に書き換える」という前提（レビュー指摘の
/// 根拠）を検証し、そのうえで `baseline == fixed` を確認することで
/// `predict_in_eval` がその汚染を防ぐことを検証する。
#[test]
fn predict_in_eval_prevents_batchnorm_running_stats_pollution() {
    const SEED: u64 = 0x1234_5678_9ABC_DEF0;
    let mut rng = SplitMix64(0xBEEF_0000_CAFE_0001);
    let mut src = || rng.next_u64();
    let (flat, _labels) = synthetic_cifar10(4, &mut src).unwrap();
    let x = image_tensor(flat, 4);

    let mut baseline = ResNet::new(8, 4, NUM_CLASSES, SEED).unwrap();
    ReferenceModule::set_training(&mut baseline, false);
    let baseline_out = baseline
        .predict(&x)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec();

    let mut polluted = ResNet::new(8, 4, NUM_CLASSES, SEED).unwrap();
    // 修正前の bug パターン: 構築直後（training の既定値 true）のまま
    // 直接 predict を呼ぶ（BatchNorm が train モードの forward を実行
    // し running stats を更新する）。
    let _ = polluted.predict(&x).unwrap();
    ReferenceModule::set_training(&mut polluted, false);
    let polluted_out = polluted
        .predict(&x)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec();
    assert_ne!(
        baseline_out, polluted_out,
        "train モードの predict が BatchNorm running stats を実際に \
         書き換えることの前提確認（この前提が崩れているとレビュー \
         指摘自体が成立しない）"
    );

    let mut fixed = ResNet::new(8, 4, NUM_CLASSES, SEED).unwrap();
    let _ = predict_in_eval(&mut fixed, &x).unwrap();
    ReferenceModule::set_training(&mut fixed, false);
    let fixed_out = fixed
        .predict(&x)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec();
    assert_eq!(
        baseline_out, fixed_out,
        "predict_in_eval は running stats を汚染しない（呼び出し前後で \
         モードを保存・復元し、eval モードで forward するため）"
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
/// PR #2325）。`training` は実フィールドとして保持する（`accuracy` が
/// エラー経路でも呼び出し前のモードへ正しく復元することを検証する
/// ため。ダミーの no-op `set_training` では検証できない）。
struct FlatLogitsModel {
    num_classes: usize,
    training: bool,
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

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[test]
fn accuracy_rejects_logits_shape_mismatch_with_matching_element_count() {
    use reference_module::accuracy;

    let mut model = FlatLogitsModel {
        num_classes: NUM_CLASSES,
        training: true,
    };
    let x = image_tensor(vec![0.0f32; 4 * IMG_C * IMG_H * IMG_W], 4);
    let y = labels_tensor(vec![0, 1, 2, 3], 4);

    assert!(matches!(
        accuracy(&mut model, &x, &y, 4, NUM_CLASSES),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // エラー経路でも accuracy 呼び出し前のモード（train）へ復元される
    // ことを検証する（Codex レビュー指摘・イシュー #2202 PR #2325）。
    assert!(
        ReferenceModule::is_training(&model),
        "accuracy のエラー経路後も呼び出し前の training モードへ復元される契約"
    );
}

#[test]
fn accuracy_restores_original_training_mode_after_success() {
    use reference_module::accuracy;

    // train モードから呼んだ場合。
    let mut model_train = FlatLogitsModel {
        num_classes: NUM_CLASSES,
        training: true,
    };
    let x = image_tensor(vec![0.0f32; 4 * IMG_C * IMG_H * IMG_W], 4);
    let y = labels_tensor(vec![0, 1, 2, 3], 4);
    // このモデルは shape 不一致で必ず Err を返すが、モード復元は
    // 成功・失敗いずれの経路でも共通の実装（`model.set_training`
    // を呼んでから結果を返す）のため、eval モードから呼んだ場合も
    // 併せて検証する。
    let _ = accuracy(&mut model_train, &x, &y, 4, NUM_CLASSES);
    assert!(ReferenceModule::is_training(&model_train));

    let mut model_eval = FlatLogitsModel {
        num_classes: NUM_CLASSES,
        training: false,
    };
    let _ = accuracy(&mut model_eval, &x, &y, 4, NUM_CLASSES);
    assert!(!ReferenceModule::is_training(&model_eval));
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

#[test]
fn resnet_train_step_forces_train_mode() {
    // `train_step` は `Trainable`（pub trait）のメソッドとして
    // `fit_epochs` の内部ループを経由せず直接呼び出せる。eval モードの
    // モデルへ直接呼んでも、冒頭で train モードへ強制されることを
    // 検証する（Codex レビュー指摘・イシュー #2202 PR #2325）。
    let mut model = ResNet::new(8, 4, NUM_CLASSES, 0x1234_0006).unwrap();
    ReferenceModule::set_training(&mut model, false);
    assert!(!ReferenceModule::is_training(&model));

    let mut rng = SplitMix64(0x1234_0007);
    let mut src = || rng.next_u64();
    let (flat, labels) = synthetic_cifar10(4, &mut src).unwrap();
    let x = image_tensor(flat, 4);
    let y = labels_tensor(labels, 4);
    let mut opt = Adam::new(AdamConfig::default()).unwrap();

    model.train_step(&x, &y, &mut opt).unwrap();
    assert!(
        ReferenceModule::is_training(&model),
        "train_step は eval モードのモデルに対しても冒頭で train \
         モードを強制する契約"
    );
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
    // 学習前の predict 疎通確認（main.rs run_resnet と同じ手順。
    // `predict_in_eval` で BatchNorm running stats の汚染を防ぐ。
    // Codex レビュー指摘・イシュー #2202 PR #2325）。
    let sample_batch = sub_tensor_f32(&x_train, 0, BATCH_SIZE).unwrap();
    let sample_pred = predict_in_eval(&mut model, &sample_batch).unwrap();
    assert_eq!(sample_pred.shape(), &[BATCH_SIZE, NUM_CLASSES]);

    let mut opt = Adam::new(AdamConfig {
        lr: LR,
        ..AdamConfig::default()
    })
    .unwrap();

    let history = fit_epochs(&mut model, &x_train, &y_train, &mut opt, EPOCHS, BATCH_SIZE).unwrap();
    let test_acc = accuracy(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES).unwrap();
    let test_loss = heldout_loss(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES).unwrap();
    assert!(test_loss.is_finite(), "held-out loss が非有限: {test_loss}");

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
