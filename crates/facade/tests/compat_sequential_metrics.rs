//! `fit()` の分類 metrics 対応（accuracy・precision・recall・F1・
//! confusion matrix。イシュー #2072・親 #2059）の統合テスト。
//! `compat_sequential_fit.rs`／`compat_sequential_callbacks.rs` と同じ
//! 方針で `fandhe_ai` のみを import し、比較対象も facade 経由の既存
//! メソッド（`fit_with_callbacks`／`predict`／
//! `fandhe_ai::compat::MetricsResult::compute`）だけで組み立てる。
//!
//! **決定的シード**: 重み初期化・データ生成は固定シードで駆動する
//! （`.claude/rules/coding-rust.md`）。実機（CUDA/Metal）非依存のため
//! `#[ignore]` 分離は行わない（3 バックエンド bit 同一の検証は
//! `compat_sequential_metrics_backend_parity.rs` 側）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{
    Callback, EarlyStopping, FitConfig, Loss, LrSchedule, Metrics, MetricsResult, ModelCheckpoint,
    Monitor, MonitorMode, Optimizer, Sequential,
};
use fandhe_ai::optim::{SgdConfig, StepLr};
use fandhe_ai::{AutodiffError, Tensor};

const N: usize = 24;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 3; // クラス数（logits 次元）

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_VAL: u64 = 0xBADA55;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `compat_sequential_fit.rs::gen_classification_data` と同型の決定的
/// 生成（本ファイルはテストバイナリが分かれるため独立に定義する）。
fn gen_classification_data(seed: u64, classes: usize) -> (Tensor<f32>, Tensor<i32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y: Vec<i32> = (0..N).map(|i| (i % classes) as i32).collect();
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N]).unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let xd = x
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        let yd = y
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        if xd.len() != yd.len() {
            return false;
        }
        for (xv, yv) in xd.iter().zip(yd.iter()) {
            if xv.to_bits() != yv.to_bits() {
                return false;
            }
        }
    }
    true
}

const ALL_METRICS: [Metrics; 5] = [
    Metrics::Accuracy,
    Metrics::Precision,
    Metrics::Recall,
    Metrics::F1,
    Metrics::ConfusionMatrix,
];

// =====================================================================
// 1. fit_with_metrics(..., &[]) は fit_with_callbacks と bit 完全一致
// =====================================================================

#[test]
fn fit_with_metrics_empty_matches_fit_with_callbacks_bit_exact() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);

    let mut model_a = build_model();
    model_a
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let hist_a = model_a
        .fit_with_callbacks(
            &x,
            &y,
            FitConfig::new(3, N),
            Some((&x_val, &y_val)),
            &mut [],
        )
        .unwrap();

    let mut model_b = build_model();
    model_b
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let hist_b = model_b
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(3, N),
            Some((&x_val, &y_val)),
            &mut [],
            &[],
        )
        .unwrap();

    assert_eq!(hist_a, hist_b);
    assert!(hist_b.val_metrics.is_empty());
    assert!(params_bit_exact(
        &model_a.trainable_parameters(),
        &model_b.trainable_parameters()
    ));
}

// =====================================================================
// 2. metrics あり／なしで loss・val_loss・params が bit 完全一致
//    （loss 演算列の非変更）
// =====================================================================

#[test]
fn metrics_computation_does_not_perturb_loss_or_params() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);

    let mut without_metrics = build_model();
    without_metrics
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let hist_without = without_metrics
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(3, N),
            Some((&x_val, &y_val)),
            &mut [],
            &[],
        )
        .unwrap();

    let mut with_metrics = build_model();
    with_metrics
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let hist_with = with_metrics
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(3, N),
            Some((&x_val, &y_val)),
            &mut [],
            &ALL_METRICS,
        )
        .unwrap();

    assert_eq!(hist_without.loss, hist_with.loss);
    assert_eq!(hist_without.val_loss, hist_with.val_loss);
    assert_eq!(hist_without.lr, hist_with.lr);
    assert!(params_bit_exact(
        &without_metrics.trainable_parameters(),
        &with_metrics.trainable_parameters()
    ));
    assert_eq!(hist_with.val_metrics.len(), 3);
}

// =====================================================================
// 3. val_metrics が epochs 分埋まり、各要素が独立に求めた
//    MetricsResult::compute と一致する（fit(1)+fit(1) == fit(2) 契約を
//    利用した非トートロジー検証）
// =====================================================================

#[test]
fn val_metrics_matches_independent_predict_and_compute_per_epoch() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);
    const EPOCHS: usize = 3;

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let history = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(EPOCHS, N),
            Some((&x_val, &y_val)),
            &mut [],
            &[Metrics::Accuracy, Metrics::F1],
        )
        .unwrap();
    assert_eq!(history.val_metrics.len(), EPOCHS);

    // 独立の双子モデルを 1 epoch ずつ回し、各 epoch 末で eval モードの
    // `predict` + `MetricsResult::compute` を直接計算して突合する。
    let mut twin = build_model();
    twin.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    for epoch in 0..EPOCHS {
        twin.fit(&x, &y, FitConfig::new(1, N)).unwrap();
        twin.eval();
        let logits_val = twin.predict(&x_val).unwrap();
        let expected =
            MetricsResult::compute(&[Metrics::Accuracy, Metrics::F1], &logits_val, &y_val).unwrap();
        assert_eq!(
            history.val_metrics[epoch], expected,
            "epoch {epoch}: history.val_metrics が独立計算と一致しない"
        );
    }
}

// =====================================================================
// 4. ModelCheckpoint（Monitor::ValMetric(Accuracy)・MonitorMode::Max）
//    が accuracy 最大 epoch のスナップショットを保持する
// =====================================================================

#[test]
fn model_checkpoint_tracks_best_val_accuracy_with_max_mode() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);
    const EPOCHS: usize = 5;

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.2)), Loss::CrossEntropy)
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new()
            .monitor(Monitor::ValMetric(Metrics::Accuracy))
            .mode(MonitorMode::Max),
    )];
    let history = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(EPOCHS, N),
            Some((&x_val, &y_val)),
            &mut callbacks,
            &[Metrics::Accuracy],
        )
        .unwrap();

    let Callback::ModelCheckpoint(mc) = &callbacks[0] else {
        panic!("test fixture: callbacks[0] は ModelCheckpoint のはず");
    };
    let best_epoch = mc
        .best_epoch()
        .expect("test fixture: 5 epoch 学習したので best_epoch は Some のはず");
    let best_value = mc
        .best_value()
        .expect("test fixture: best_value は Some のはず");
    let expected_best_accuracy = history
        .val_metrics
        .iter()
        .map(|m| m.accuracy.expect("Accuracy を要求したので Some のはず"))
        .fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(
        history.val_metrics[best_epoch]
            .accuracy
            .expect("Accuracy を要求したので Some のはず"),
        best_value
    );
    assert_eq!(best_value, expected_best_accuracy);
}

// =====================================================================
// 5. EarlyStopping／LrSchedule も Monitor::ValMetric で駆動できる
// =====================================================================

#[test]
fn early_stopping_and_lr_schedule_are_driven_by_val_metric() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.2)), Loss::CrossEntropy)
        .unwrap();
    let mut callbacks = [
        Callback::EarlyStopping(
            EarlyStopping::new(2)
                .monitor(Monitor::ValMetric(Metrics::Accuracy))
                .mode(MonitorMode::Max),
        ),
        Callback::LrSchedule(LrSchedule::per_epoch(
            StepLr::new(0.2, 1, 0.5)
                .expect("test fixture: StepLr::new(0.2, 1, 0.5) は有効値のはず"),
        )),
    ];
    let history = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(8, N),
            Some((&x_val, &y_val)),
            &mut callbacks,
            &[Metrics::Accuracy],
        )
        .unwrap();

    // 少なくとも 1 epoch は進み、val_metrics・lr が矛盾なく記録される
    // ことのみを検査する（`EarlyStopping`／`LrSchedule` 自体のロジックは
    // `compat_sequential_callbacks.rs` で既に検証済みのため、本テストは
    // `Monitor::ValMetric` 経由でも同じ配線が機能することの確認に限る）。
    assert!(!history.loss.is_empty());
    assert_eq!(history.val_metrics.len(), history.loss.len());
    assert_eq!(history.lr.len(), history.loss.len());
}

// =====================================================================
// 6. fail-closed 検査（4 パターン）
// =====================================================================

#[test]
fn fit_with_metrics_rejects_non_empty_metrics_without_validation() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let prev_training = model.training();
    let err = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(1, N),
            None,
            &mut [],
            &[Metrics::Accuracy],
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
    assert_eq!(model.training(), prev_training);
}

#[test]
fn fit_with_metrics_rejects_mse_target() {
    let mut rng = Xorshift64Star::new(SEED_DATA);
    let x = Tensor::new(rng.fill_vec(N * D_IN), &[N, D_IN]).unwrap();
    let y = Tensor::new(rng.fill_vec(N * D_OUT), &[N, D_OUT]).unwrap();
    let x_val = x.clone();
    let y_val = y.clone();

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    let err = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(1, N),
            Some((&x_val, &y_val)),
            &mut [],
            &[Metrics::Accuracy],
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

#[test]
fn fit_with_metrics_rejects_val_metric_not_in_requested_metrics() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let mut callbacks = [Callback::EarlyStopping(
        EarlyStopping::new(1).monitor(Monitor::ValMetric(Metrics::Recall)),
    )];
    let err = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(1, N),
            Some((&x_val, &y_val)),
            &mut callbacks,
            &[Metrics::Accuracy], // Recall を含まない
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn fit_with_metrics_rejects_val_metric_confusion_matrix() {
    let (x, y) = gen_classification_data(SEED_DATA, D_OUT);
    let (x_val, y_val) = gen_classification_data(SEED_VAL, D_OUT);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let mut callbacks = [Callback::ModelCheckpoint(
        ModelCheckpoint::new().monitor(Monitor::ValMetric(Metrics::ConfusionMatrix)),
    )];
    let err = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(1, N),
            Some((&x_val, &y_val)),
            &mut callbacks,
            &[Metrics::ConfusionMatrix],
        )
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}
