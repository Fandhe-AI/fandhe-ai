//! `fandhe_ai::data`（Dataset／DataLoader。イシュー #1615・親 #1602）の
//! 統合テスト。
//!
//! `fandhe_ai::optim`（`optim_train_loop.rs`）と同じく回帰学習ループは
//! **`fandhe_ai` のみを import する**（分類スモークテストのみ、
//! `Reduction` が facade 未再エクスポート〈既存ギャップ。
//! `nll_kl_div_backend_parity.rs` 冒頭コメント参照〉のため
//! `fandhe_ai_autodiff::Reduction` を直接 import する）。
//!
//! グローバル RNG（`manual_seed`）を消費するシャッフルテストを含むため、
//! ファイル局所 `Mutex` で直列化する（`rng_tensor_generation.rs` と同型）。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{DataLoader, DataLoaderConfig, TensorDataset};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai_autodiff::Reduction;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

/// `optim_train_loop.rs::gen_regression_data` と同型の決定的データ
/// 生成（`N` サンプル・回帰ラベル）。
fn gen_dataset(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[])
        .unwrap_or_else(|| panic!("test fixture: スカラー shape [] のはず"))
}

/// `fandhe_ai::data::DataLoader` によるミニバッチ学習ループが
/// `fandhe_ai` のみへの依存で書け、loss が減少することを確認する
/// （受入基準: Dataset／DataLoader が既存の学習経路〈`Tape`／
/// `Sequential`／`optim`〉とそのまま組み合わさること）。
#[test]
fn minibatch_training_with_data_loader_converges_via_facade_only() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let (x_data, y_data) = gen_dataset(0xC0FFEE);
    let dataset = (
        TensorDataset::new(x_data).unwrap_or_else(|e| {
            panic!("test fixture: features TensorDataset::new が失敗した: {e}")
        }),
        TensorDataset::new(y_data)
            .unwrap_or_else(|e| panic!("test fixture: labels TensorDataset::new が失敗した: {e}")),
    );
    let loader = DataLoader::new(dataset, DataLoaderConfig::new(4).shuffle(true))
        .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗した: {e}"));
    assert_eq!(loader.len(), 4); // N=16, batch_size=4, drop_last=false

    let mut model = build_model();
    let mut sgd = Sgd::new(SgdConfig::new(0.05))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));

    const EPOCHS: usize = 30;
    let mut epoch_losses = Vec::with_capacity(EPOCHS);

    for _ in 0..EPOCHS {
        let mut epoch_loss_sum = 0.0f32;
        let mut batch_count = 0usize;
        for batch in loader.iter() {
            let (x_batch, y_batch) =
                batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));

            let updated = {
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                let x = tape.var(&x_batch);
                let y = tape.var(&y_batch);

                let pred = bound
                    .forward(&tape, &x)
                    .unwrap_or_else(|e| panic!("test fixture: forward が失敗した: {e}"));
                let loss = pred
                    .mse_loss(&y)
                    .unwrap_or_else(|e| panic!("test fixture: mse_loss が失敗した: {e}"));
                epoch_loss_sum += scalar(&loss.to_tensor());
                batch_count += 1;

                let grads = tape
                    .backward(&loss)
                    .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
                let grad_refs = bound
                    .trainable_grads(&grads)
                    .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
                let param_refs = model.trainable_parameters();
                sgd.step(&param_refs, &grad_refs)
                    .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"))
            };
            model
                .apply_parameters(updated)
                .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
        }
        epoch_losses.push(epoch_loss_sum / batch_count as f32);
    }

    assert_eq!(epoch_losses.len(), EPOCHS);
    let first = epoch_losses[0];
    let last = *epoch_losses.last().unwrap();
    assert!(
        last < first,
        "DataLoader によるミニバッチ学習で loss が減少するはず: first={first} last={last}"
    );
}

/// `(TensorDataset<f32>, TensorDataset<i32>)` のバッチが
/// `Var::cross_entropy_loss`（targets: `Tensor<i32>`）へそのまま渡せる
/// ことの型整合スモーク（分類タスクでの典型的な組み合わせ）。
#[test]
fn classification_batches_pair_f32_features_with_i32_targets() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let mut rng = Xorshift64Star::new(0xABCDEF);
    let features = Tensor::new(rng.fill_vec(N * D_IN), &[N, D_IN])
        .unwrap_or_else(|e| panic!("test fixture: features tensor 構築に失敗: {e}"));
    let num_classes = 3usize;
    let targets_data: Vec<i32> = (0..N).map(|i| (i % num_classes) as i32).collect();
    let targets = Tensor::new(targets_data, &[N])
        .unwrap_or_else(|e| panic!("test fixture: targets tensor 構築に失敗: {e}"));

    let dataset = (
        TensorDataset::new(features)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}")),
        TensorDataset::new(targets)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}")),
    );
    let loader = DataLoader::new(dataset, DataLoaderConfig::new(4))
        .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗した: {e}"));

    let model = Sequential::new()
        .add_linear(D_IN, num_classes, 0x3333_3333)
        .unwrap_or_else(|e| panic!("test fixture: 層の構築に失敗: {e}"));

    for batch in loader.iter() {
        let (x_batch, y_batch) =
            batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));

        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(&x_batch);
        let logits = bound
            .forward(&tape, &x)
            .unwrap_or_else(|e| panic!("test fixture: forward が失敗した: {e}"));
        // targets（`Tensor<i32>`）はそのまま渡す（`Var::cross_entropy_loss`
        // は非追跡の `&Tensor<i32>` を受け取る契約。`crates/autodiff/src/
        // var.rs::cross_entropy_loss` docstring 参照）。
        let loss = logits
            .cross_entropy_loss(&y_batch, 1, Reduction::Mean)
            .unwrap_or_else(|e| panic!("test fixture: cross_entropy_loss が失敗した: {e}"));
        let value = scalar(&loss.to_tensor());
        assert!(value.is_finite(), "loss は有限値のはず: {value}");
    }
}

/// バッチテンソルを `Tape::var` でアップロードした値が、DataLoader が
/// ホスト側で組み立てた値と bit 完全一致することの固定（モジュール冒頭
/// 「parity」注記の直接検証。CPU バックエンドは純粋コピーのため往復で
/// 数値変化が起きない）。
#[test]
fn batch_upload_to_cpu_tape_is_bit_identical() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let (x_data, _y_data) = gen_dataset(0x5EED);
    let dataset = TensorDataset::new(x_data.clone())
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}"));
    let loader = DataLoader::new(dataset, DataLoaderConfig::new(4))
        .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗した: {e}"));

    let mut start = 0usize;
    for batch in loader.iter() {
        let batch = batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));
        let len = batch.shape()[0];

        let tape = fandhe_ai::tape();
        let uploaded = tape.var(&batch).to_tensor();
        assert_eq!(uploaded.host_slice().as_ref(), batch.host_slice().as_ref());

        let expected = x_data
            .narrow(0, start, len)
            .unwrap_or_else(|e| panic!("test fixture: narrow が失敗した: {e}"))
            .contiguous();
        assert_eq!(
            batch.host_slice().as_ref(),
            expected.host_slice().as_ref(),
            "shuffle=false のバッチは narrow(0, start, len) と bit 完全一致するはず"
        );
        start += len;
    }
}

/// CUDA 実機での round-trip 確認（本エージェント実行環境には実機が
/// ないため未実測のまま GB10 セッションへ申し送り。
/// `rng_tensor_generation.rs::randn_upload_to_cuda_tape_round_trips`
/// と同型）。
#[test]
#[ignore = "実機（CUDA）依存。DGX Spark GB10 等で手動実行する"]
fn batch_upload_round_trips_on_cuda_tape() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let (x_data, _y_data) = gen_dataset(0x1234);
    let dataset = TensorDataset::new(x_data)
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}"));
    let loader = DataLoader::new(dataset, DataLoaderConfig::new(4))
        .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗した: {e}"));

    let tape = fandhe_ai::tape_for(fandhe_ai::Device::Cuda(0))
        .unwrap_or_else(|e| panic!("test fixture: CUDA tape_for が失敗した: {e}"));
    for batch in loader.iter() {
        let batch = batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));
        let uploaded = tape.var(&batch).to_tensor();
        assert_eq!(uploaded.host_slice().as_ref(), batch.host_slice().as_ref());
    }
}

/// Metal 実機での round-trip 確認（本エージェント実行環境には実機が
/// ないため未実測のまま Mac セッションへ申し送り。
/// `rng_tensor_generation.rs::randn_upload_to_metal_tape_round_trips`
/// と同型）。
#[test]
#[cfg(target_os = "macos")]
#[ignore = "実機（Metal）依存。Apple Silicon 実機で手動実行する"]
fn batch_upload_round_trips_on_metal_tape() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let (x_data, _y_data) = gen_dataset(0x1234);
    let dataset = TensorDataset::new(x_data)
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}"));
    let loader = DataLoader::new(dataset, DataLoaderConfig::new(4))
        .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗した: {e}"));

    let tape = fandhe_ai::tape_for(fandhe_ai::Device::Metal)
        .unwrap_or_else(|e| panic!("test fixture: Metal tape_for が失敗した: {e}"));
    for batch in loader.iter() {
        let batch = batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));
        let uploaded = tape.var(&batch).to_tensor();
        assert_eq!(uploaded.host_slice().as_ref(), batch.host_slice().as_ref());
    }
}
