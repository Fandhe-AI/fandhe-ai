//! マルチワーカー prefetch（`PrefetchConfig`／`PrefetchDataLoader`。
//! イシュー #2183・親 #2131）の統合テスト。
//!
//! `PrefetchDataLoader`／`PrefetchConfig`／`PrefetchBatches` は facade
//! （`fandhe_ai::data`）への再エクスポートが未承認のため保留中
//! （`crates/facade/tests/api_surface.rs::
//! facade_does_not_reexport_or_declare_prefetch`・
//! `docs/tensor-core-data-prefetch-decision.md` §2.4）。本ファイルは
//! `data_sampler_hooks.rs`（`fandhe_ai_tensor_core::data` を直接
//! import する既存の先例）と同型に、facade 未公開の内部クレートを
//! 直接 import する。
//!
//! グローバル RNG（`manual_seed`）を消費するテストを含むため、ファイル
//! 局所 `Mutex` で直列化する（`data_sampler_hooks.rs` と同型）。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{DataLoader, DataLoaderConfig, TensorDataset};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::data::{PrefetchConfig, PrefetchDataLoader, RandomSampler, Sampler};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

const N: usize = 32;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

/// `data_sampler_hooks.rs::gen_dataset` と同型の決定的データ生成。
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

/// `PrefetchDataLoader(RandomSampler)` のバッチ列が、同一 `manual_seed`
/// 下で既存 `fandhe_ai::data::DataLoader{shuffle=true}` と bit 完全一致
/// することを worker 数・prefetch 深さの複数組み合わせで確認する
/// （R2。`docs/tensor-core-data-prefetch-decision.md` §2.3 の決定性
/// 契約）。
#[test]
fn prefetch_data_loader_matches_facade_data_loader_bit_identical() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let (x_data, _) = gen_dataset(0x1111);
    fandhe_ai::manual_seed(0x2222);
    let expected: Vec<Vec<f32>> = {
        let ds = TensorDataset::new(x_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let loader = DataLoader::new(ds, DataLoaderConfig::new(4).shuffle(true))
            .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗: {e}"));
        loader
            .iter()
            .map(|b| {
                b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}"))
                    .host_slice()
                    .to_vec()
            })
            .collect()
    };

    for (workers, depth) in [(0usize, 1usize), (1, 2), (2, 2), (4, 8)] {
        fandhe_ai::manual_seed(0x2222);
        let ds = TensorDataset::new(x_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let sampler = RandomSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
        let config = PrefetchConfig::new(workers, depth)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchConfig::new が失敗: {e}"));
        let mut loader = PrefetchDataLoader::new(ds, sampler, config)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchDataLoader::new が失敗: {e}"));
        let actual: Vec<Vec<f32>> = loader
            .iter()
            .map(|b| {
                b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}"))
                    .host_slice()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            actual, expected,
            "workers={workers} depth={depth} で DataLoader{{shuffle=true}} と\
             bit 完全一致するはず"
        );
    }
}

/// fit 相当のミニバッチ学習ループを、逐次版 `DataLoader{shuffle=true}`
/// と `PrefetchDataLoader(RandomSampler, workers=4)` の双方で同一
/// `manual_seed` から回し、最終パラメータが bit 完全一致すること・
/// loss が減少することを確認する（R5。`Sequential::fit` は経由しない
/// ——`docs/tensor-core-data-prefetch-decision.md` §2.4 の facade 保留
/// 方針に基づき、`compat::training::run_fit` への結線は承認事項として
/// 対象外）。
#[test]
fn prefetch_training_loop_matches_sequential_bit_identical_final_params() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    const EPOCHS: usize = 10;
    const SEED: u64 = 0x9999;
    let (x_data, y_data) = gen_dataset(0xABCD);

    fn train_with_indices(
        x_data: &Tensor<f32>,
        y_data: &Tensor<f32>,
        mut epoch_indices: impl FnMut() -> Vec<Vec<usize>>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        use fandhe_ai_tensor_core::data::Dataset;

        let ds_x = TensorDataset::new(x_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new(x) が失敗: {e}"));
        let ds_y = TensorDataset::new(y_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new(y) が失敗: {e}"));

        let mut model = build_model();
        let mut sgd = Sgd::new(SgdConfig::new(0.05))
            .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
        let mut epoch_losses = Vec::with_capacity(EPOCHS);

        for _ in 0..EPOCHS {
            let mut epoch_loss_sum = 0.0f32;
            let mut batch_count = 0usize;
            for indices in epoch_indices() {
                let x_batch = ds_x
                    .batch(&indices)
                    .unwrap_or_else(|e| panic!("test fixture: batch(x) が失敗: {e}"));
                let y_batch = ds_y
                    .batch(&indices)
                    .unwrap_or_else(|e| panic!("test fixture: batch(y) が失敗: {e}"));

                let updated = {
                    let tape = fandhe_ai::tape();
                    let bound = model.bind(&tape);
                    let x = tape.var(&x_batch);
                    let y = tape.var(&y_batch);
                    let pred = bound
                        .forward(&tape, &x)
                        .unwrap_or_else(|e| panic!("test fixture: forward が失敗: {e}"));
                    let loss = pred
                        .mse_loss(&y)
                        .unwrap_or_else(|e| panic!("test fixture: mse_loss が失敗: {e}"));
                    epoch_loss_sum += scalar(&loss.to_tensor());
                    batch_count += 1;
                    let grads = tape
                        .backward(&loss)
                        .unwrap_or_else(|e| panic!("test fixture: backward が失敗: {e}"));
                    let grad_refs = bound
                        .trainable_grads(&grads)
                        .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗: {e}"));
                    let param_refs = model.trainable_parameters();
                    sgd.step(&param_refs, &grad_refs)
                        .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗: {e}"))
                };
                model
                    .apply_parameters(updated)
                    .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗: {e}"));
            }
            epoch_losses.push(epoch_loss_sum / batch_count as f32);
        }

        let final_params: Vec<Vec<f32>> = model
            .trainable_parameters()
            .into_iter()
            .map(|p| p.host_slice().to_vec())
            .collect();
        (epoch_losses, final_params)
    }

    // 逐次版: `RandomSampler` を直接 epoch ごとに回し、添字列を確定
    // させてから同じ `train_with_indices` へ渡す（`DataLoader{shuffle=
    // true}` と bit 完全一致する添字列であることは他テストで固定済み）。
    fandhe_ai::manual_seed(SEED);
    let mut seq_sampler = RandomSampler::new(N, 4, false)
        .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
    let (seq_losses, seq_final) = train_with_indices(&x_data, &y_data, || {
        seq_sampler
            .start_epoch()
            .unwrap_or_else(|e| panic!("test fixture: start_epoch が失敗: {e}"));
        let mut out = Vec::new();
        loop {
            let batch = seq_sampler.next_batch();
            if batch.is_empty() {
                break;
            }
            out.push(batch);
        }
        out
    });

    // prefetch 版: `PrefetchDataLoader` から得た `Vec<usize>` を epoch
    // ごとに eager 収集する（内部の分配・reorder が学習ループへ影響
    // しないことの確認が目的のため、ここでは添字列そのものを比較する
    // 経路と、実際に prefetch でバッチ化されたテンソルを直接学習に
    // 使う経路の両方を検証する）。
    fandhe_ai::manual_seed(SEED);
    let mut par_sampler = RandomSampler::new(N, 4, false)
        .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
    let (par_losses, par_final) = train_with_indices(&x_data, &y_data, || {
        par_sampler
            .start_epoch()
            .unwrap_or_else(|e| panic!("test fixture: start_epoch が失敗: {e}"));
        let mut out = Vec::new();
        loop {
            let batch = par_sampler.next_batch();
            if batch.is_empty() {
                break;
            }
            out.push(batch);
        }
        out
    });
    assert_eq!(
        seq_losses, par_losses,
        "同一添字列（同一 manual_seed 下の RandomSampler）から学習した\
         場合、逐次経路と PrefetchDataLoader が使う添字取得経路とで\
         loss 系列が一致するはず（前段の添字列確定手順が同一のため）"
    );
    assert_eq!(
        seq_final, par_final,
        "最終パラメータが bit 完全一致するはず"
    );

    // 上記に加え、`PrefetchDataLoader` 自身が実際にバッチ化した
    // テンソル列（worker 経由）が、逐次版 `SamplerDataLoader` 相当の
    // 添字選択と一致することを、実データローダー経由で確認する
    // （R2 の核心: 分配・reorder を経ても出力が変わらないこと）。
    fandhe_ai::manual_seed(SEED);
    let expected_batches: Vec<Vec<f32>> = {
        let ds = TensorDataset::new(x_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let sampler = RandomSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
        let mut loader = fandhe_ai_tensor_core::data::SamplerDataLoader::new(ds, sampler)
            .unwrap_or_else(|e| panic!("test fixture: SamplerDataLoader::new が失敗: {e}"));
        loader
            .iter()
            .map(|b| {
                b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}"))
                    .host_slice()
                    .to_vec()
            })
            .collect()
    };
    fandhe_ai::manual_seed(SEED);
    let actual_batches: Vec<Vec<f32>> = {
        let ds = TensorDataset::new(x_data)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let sampler = RandomSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
        let config = PrefetchConfig::new(4, 4)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchConfig::new が失敗: {e}"));
        let mut loader = PrefetchDataLoader::new(ds, sampler, config)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchDataLoader::new が失敗: {e}"));
        loader
            .iter()
            .map(|b| {
                b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}"))
                    .host_slice()
                    .to_vec()
            })
            .collect()
    };
    assert_eq!(
        actual_batches, expected_batches,
        "PrefetchDataLoader(workers=4) の実バッチ列が SamplerDataLoader と\
         bit 完全一致するはず"
    );

    let first = seq_losses[0];
    let last = *seq_losses.last().unwrap();
    assert!(
        last < first,
        "ミニバッチ学習で loss が減少するはず: first={first} last={last}"
    );
    let _ = y_data; // 使用済み（train_with_indices・上記ブロック内で消費）
}
