//! `Sampler`・collate・transform フック（イシュー #2182・親 #2131）の
//! 統合テスト。
//!
//! `SamplerDataLoader`／`HookedDataLoader`／`Sampler` 系（3 実装）は
//! #2505 で facade（`fandhe_ai::data`）へ公開済み
//! （`docs/tensor-core-data-sampler-hooks-decision.md` §5・§8）。本
//! ファイルは facade のみ経由で import する。
//!
//! グローバル RNG（`manual_seed`）を消費するテストを含むため、ファイル
//! 局所 `Mutex` で直列化する（`data_loader.rs` と同型）。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{
    DataError, DataLoader, DataLoaderConfig, HookedDataLoader, RandomSampler, SamplerDataLoader,
    SequentialSampler, TensorDataset, WeightedRandomSampler,
};
use fandhe_ai::optim::{Sgd, SgdConfig};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

/// `data_loader.rs::gen_dataset` と同型の決定的データ生成。
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

/// `HookedDataLoader`（`SequentialSampler`／`RandomSampler`、フック
/// なし）のバッチ列が、同一シードの下で既存 `fandhe_ai::data::
/// DataLoader` と bit 完全一致することを確認する（AC-5「単体でも組み
/// 合わせでも確認する」のうち、既存 API との等価性）。フックなしの
/// fast path（`Dataset::batch` への直行）が `DataLoader::iter`
/// （`gather_rows`）と同じ添字選択に対し同じ出力を返すことの固定。
#[test]
fn hooked_data_loader_matches_facade_data_loader_bit_identical() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    // shuffle なし（SequentialSampler）。
    {
        let (x_data, _) = gen_dataset(0x1111);
        let expected: Vec<Vec<f32>> = {
            let ds = TensorDataset::new(x_data.clone())
                .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
            let loader = DataLoader::new(ds, DataLoaderConfig::new(4))
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

        let ds2 = TensorDataset::new(x_data)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let sampler = SequentialSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: SequentialSampler::new が失敗: {e}"));
        let mut hooked = HookedDataLoader::new(ds2, sampler)
            .unwrap_or_else(|e| panic!("test fixture: HookedDataLoader::new が失敗: {e}"));
        let actual: Vec<Vec<f32>> = hooked
            .iter()
            .map(|b| {
                b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}"))
                    .host_slice()
                    .to_vec()
            })
            .collect();
        assert_eq!(actual, expected, "SequentialSampler は shuffle なしと一致");
    }

    // shuffle あり（RandomSampler。同一 manual_seed 下）。
    {
        fandhe_ai::manual_seed(0x2222);
        let (x_data, _) = gen_dataset(0x3333);
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

        fandhe_ai::manual_seed(0x2222);
        let ds2 = TensorDataset::new(x_data)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let sampler = RandomSampler::new(N, 4, false)
            .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
        let mut hooked = HookedDataLoader::new(ds2, sampler)
            .unwrap_or_else(|e| panic!("test fixture: HookedDataLoader::new が失敗: {e}"));
        let actual: Vec<Vec<f32>> = hooked
            .iter()
            .map(|b| {
                b.unwrap_or_else(|e| panic!("test fixture: batch が失敗: {e}"))
                    .host_slice()
                    .to_vec()
            })
            .collect();
        assert_eq!(
            actual, expected,
            "同一 manual_seed 下で RandomSampler は shuffle=true と bit 完全一致するはず"
        );
    }
}

/// `SamplerDataLoader`（タプルデータセット + `RandomSampler`）による
/// 分類バッチの型整合スモーク（AC-5）。特徴量とラベルが同一の抽選順
/// （同一添字）で取り出され、`Var::cross_entropy_loss` へそのまま渡せる
/// ことを確認する。
#[test]
fn sampler_data_loader_tuple_dataset_classification_smoke() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(0x4444);

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
    let sampler = RandomSampler::new(N, 4, false)
        .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
    let mut loader = SamplerDataLoader::new(dataset, sampler)
        .unwrap_or_else(|e| panic!("test fixture: SamplerDataLoader::new が失敗: {e}"));

    let model = Sequential::new()
        .add_linear(D_IN, num_classes, 0x3333_3333)
        .unwrap_or_else(|e| panic!("test fixture: 層の構築に失敗: {e}"));

    let mut total_samples = 0usize;
    for batch in loader.iter() {
        let (x_batch, y_batch) =
            batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));
        assert_eq!(
            x_batch.shape()[0],
            y_batch.shape()[0],
            "SamplerDataLoader はタプル成分間で同じ添字を使うため行数が一致するはず"
        );
        total_samples += x_batch.shape()[0];

        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(&x_batch);
        let logits = bound
            .forward(&tape, &x)
            .unwrap_or_else(|e| panic!("test fixture: forward が失敗した: {e}"));
        let loss = logits
            .cross_entropy_loss(&y_batch, 1, fandhe_ai_autodiff::Reduction::Mean)
            .unwrap_or_else(|e| panic!("test fixture: cross_entropy_loss が失敗した: {e}"));
        let value = scalar(&loss.to_tensor());
        assert!(value.is_finite(), "loss は有限値のはず: {value}");
    }
    assert_eq!(total_samples, N);
}

/// `WeightedRandomSampler` ＋ 正規化 transform ＋ カスタム collate を
/// 組み合わせた `HookedDataLoader` でミニバッチ学習ループを回し、loss
/// が減少することを確認する（AC-5「組み合わせても確認する」の中核
/// テスト）。
#[test]
fn weighted_sampler_transform_collate_training_converges() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(0x5555);

    // サンプル単位フックは `TensorDataset<T>`（単一 `Tensor` 型）限定
    // のため、特徴量とラベルを列方向に連結した単一テンソルを
    // `WeightedRandomSampler` で抽選し、バッチ取得後に列で分割する
    // （こうすることで特徴量とラベルが常に同じ添字で対応する）。
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x_raw = rng.fill_vec(N * D_IN);
    let y_raw = rng.fill_vec(N * D_OUT);
    let mut combined = Vec::with_capacity(N * (D_IN + D_OUT));
    for i in 0..N {
        combined.extend_from_slice(&x_raw[i * D_IN..(i + 1) * D_IN]);
        combined.extend_from_slice(&y_raw[i * D_OUT..(i + 1) * D_OUT]);
    }
    let combined_tensor = Tensor::new(combined, &[N, D_IN + D_OUT])
        .unwrap_or_else(|e| panic!("test fixture: combined tensor 構築に失敗: {e}"));

    // 重みは全サンプル同等（一様重み付き復元抽出。学習が収束すること
    // 自体の確認が目的であり、重みの偏りは対象外）。
    let weights = vec![1.0f32; N];

    let ds = TensorDataset::new(combined_tensor)
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}"));
    let sampler = WeightedRandomSampler::new(weights, N, true, 4, false)
        .unwrap_or_else(|e| panic!("test fixture: WeightedRandomSampler::new が失敗した: {e}"));

    // 正規化 transform: 各行を 2 倍する（恒等ではない変換であることの
    // 確認を兼ねる。特徴量・ラベルを同じ定数倍するため回帰問題としての
    // 最適化の難度は変わらない）。
    let mut loader = HookedDataLoader::new(ds, sampler)
        .unwrap_or_else(|e| panic!("test fixture: HookedDataLoader::new が失敗した: {e}"))
        .with_transform(|t| {
            let scaled: Vec<f32> = t.host_slice().iter().map(|&v| v * 2.0).collect();
            Tensor::new(scaled, t.shape())
                .unwrap_or_else(|e| panic!("test fixture: transform 内 Tensor::new が失敗: {e}"))
        })
        .with_try_collate(
            |samples: &[Tensor<f32>]| -> Result<Tensor<f32>, DataError> {
                fandhe_ai::data::default_collate(samples)
            },
        );

    let mut model = build_model();
    let mut sgd = Sgd::new(SgdConfig::new(0.02))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));

    const EPOCHS: usize = 30;
    let mut epoch_losses = Vec::with_capacity(EPOCHS);

    for _ in 0..EPOCHS {
        let mut epoch_loss_sum = 0.0f32;
        let mut batch_count = 0usize;

        for batch in loader.iter() {
            let combined_batch =
                batch.unwrap_or_else(|e| panic!("test fixture: batch の取得が失敗した: {e}"));
            let rows = combined_batch.shape()[0];
            let x_batch = combined_batch
                .narrow(1, 0, D_IN)
                .unwrap_or_else(|e| panic!("test fixture: x 列の narrow が失敗した: {e}"))
                .contiguous();
            let y_batch = combined_batch
                .narrow(1, D_IN, D_OUT)
                .unwrap_or_else(|e| panic!("test fixture: y 列の narrow が失敗した: {e}"))
                .contiguous();
            assert_eq!(x_batch.shape(), &[rows, D_IN]);
            assert_eq!(y_batch.shape(), &[rows, D_OUT]);

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
        "WeightedRandomSampler + transform + collate によるミニバッチ\
         学習で loss が減少するはず: first={first} last={last}"
    );
}
