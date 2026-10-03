//! マルチワーカー prefetch（`PrefetchConfig`／`PrefetchDataLoader`。
//! イシュー #2183・親 #2131）の統合テスト。
//!
//! `PrefetchDataLoader`／`PrefetchConfig`／`PrefetchBatches` は #2506
//! （ルート #2499 の承認。`crates/facade/tests/api_surface.rs::
//! facade_reexports_prefetch_items_only_in_approved_shape`・
//! `docs/tensor-core-data-prefetch-decision.md` §4・§8）で facade
//! （`fandhe_ai::data`）へ公開済みであり、本ファイルは facade のみを
//! import する。
//!
//! グローバル RNG（`manual_seed`）を消費するテストを含むため、ファイル
//! 局所 `Mutex` で直列化する（`data_sampler_hooks.rs` と同型）。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{
    DataLoader, DataLoaderConfig, PrefetchConfig, PrefetchDataLoader, RandomSampler,
    SamplerDataLoader, TensorDataset,
};
use fandhe_ai::optim::{Sgd, SgdConfig};

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

/// fit 相当のミニバッチ学習ループを、逐次版 `SamplerDataLoader` と
/// `PrefetchDataLoader(RandomSampler, workers=4)` の双方で同一
/// `manual_seed` から回し、最終パラメータが bit 完全一致すること・
/// loss が減少することを確認する（R5。`Sequential::fit` は経由しない
/// ——`docs/tensor-core-data-prefetch-decision.md` §4 のとおり
/// `compat::training::run_fit` への結線は #2603 の担当で対象外）。
///
/// # 検証する経路（レビュー指摘・イシュー #2183 コメント対応）
///
/// 以前は par 側が `RandomSampler::next_batch` で得た添字列だけを
/// `train_with_indices`（逐次 `Dataset::batch` 呼び出し）へ渡しており、
/// `PrefetchDataLoader` が実際に配分・reorder したバッチを学習ステップ
/// へ使っていなかった（「添字列が一致する」ことしか検証できておらず、
/// worker 経由のバッチ組み立てを学習ループに結線した場合の bit 一致は
/// 未検証だった）。本テストは `PrefetchDataLoader::new((ds_x, ds_y),
/// sampler, config)`（[`Dataset`] のタプル実装。モジュール冒頭
/// 「決定性の契約」節）を直接学習ループへ結線し、`PrefetchBatches`
/// が yield する `(x_batch, y_batch)` をそのまま forward/backward に
/// 使うことで、実際に prefetch を使った学習の最終パラメータ一致を
/// 検証する。
#[test]
fn prefetch_training_loop_matches_sequential_bit_identical_final_params() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    const EPOCHS: usize = 10;
    const SEED: u64 = 0x9999;
    let (x_data, y_data) = gen_dataset(0xABCD);

    /// `epoch_batches` が 1 epoch 分の `(x_batch, y_batch)` 列を返す
    /// クロージャを受け取り、そのまま forward/backward/step に使う。
    /// 逐次版（`SamplerDataLoader`）・並列版（`PrefetchDataLoader`）の
    /// どちらも同じこの関数へ「実データローダーが yield したバッチ」を
    /// 渡すことで、ローダー実装の違いが学習結果へ影響しないことを
    /// 検証する。
    fn train_with_batches(
        mut epoch_batches: impl FnMut() -> Vec<(Tensor<f32>, Tensor<f32>)>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut model = build_model();
        let mut sgd = Sgd::new(SgdConfig::new(0.05))
            .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
        let mut epoch_losses = Vec::with_capacity(EPOCHS);

        for _ in 0..EPOCHS {
            let mut epoch_loss_sum = 0.0f32;
            let mut batch_count = 0usize;
            for (x_batch, y_batch) in epoch_batches() {
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

    // 逐次版: `SamplerDataLoader((ds_x, ds_y), RandomSampler)` が yield
    // する `(x_batch, y_batch)` をそのまま `train_with_batches` へ渡す。
    fandhe_ai::manual_seed(SEED);
    let (seq_losses, seq_final) = {
        let ds_x = TensorDataset::new(x_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new(x) が失敗: {e}"));
        let ds_y = TensorDataset::new(y_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new(y) が失敗: {e}"));
        let sampler = RandomSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
        let mut loader = SamplerDataLoader::new((ds_x, ds_y), sampler)
            .unwrap_or_else(|e| panic!("test fixture: SamplerDataLoader::new が失敗: {e}"));
        train_with_batches(|| {
            loader
                .iter()
                .map(|b| b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}")))
                .collect()
        })
    };

    // prefetch 版: `PrefetchDataLoader((ds_x, ds_y), RandomSampler,
    // workers=4)` が worker 経由で分配・組み立て・reorder したバッチを
    // そのまま学習ステップへ使う（レビュー指摘対応。上記ドキュメント
    // コメント「検証する経路」参照）。
    fandhe_ai::manual_seed(SEED);
    let (par_losses, par_final) = {
        let ds_x = TensorDataset::new(x_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new(x) が失敗: {e}"));
        let ds_y = TensorDataset::new(y_data.clone())
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new(y) が失敗: {e}"));
        let sampler = RandomSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
        let config = PrefetchConfig::new(4, 4)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchConfig::new が失敗: {e}"));
        let mut loader = PrefetchDataLoader::new((ds_x, ds_y), sampler, config)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchDataLoader::new が失敗: {e}"));
        train_with_batches(|| {
            loader
                .iter()
                .map(|b| b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}")))
                .collect()
        })
    };
    assert_eq!(
        seq_losses, par_losses,
        "SamplerDataLoader と PrefetchDataLoader(workers=4) の双方が\
         yield した実バッチで学習した場合、loss 系列が一致するはず"
    );
    assert_eq!(
        seq_final, par_final,
        "PrefetchDataLoader(workers=4) のバッチで学習した最終パラメータが\
         SamplerDataLoader 版と bit 完全一致するはず"
    );

    let first = seq_losses[0];
    let last = *seq_losses.last().unwrap();
    assert!(
        last < first,
        "ミニバッチ学習で loss が減少するはず: first={first} last={last}"
    );
}
