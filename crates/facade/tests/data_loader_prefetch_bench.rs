//! DataLoader マルチワーカー prefetch（イシュー #2183・親 #2131）の
//! 性能 A/B（R3〜R5）。
//!
//! **事前登録した判定規則**（`docs/perf/logs/data-loader-prefetch-2183/
//! README.md` に計測前にコミット済み。`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値」）:
//!
//! 1. 5 run の中央値（[`bench_harness::median_q1_q3`]）を使う。
//! 2. checksum（yield された全バッチ・W2 の最終パラメータについて
//!    `to_bits` を fold した値）が A/B 間・5 run 間ですべて完全一致する
//!    こと（**hard assert**。マルチワーカー化で決定性契約〈R2〉が崩れて
//!    いないことの直接的な検証を兼ねる）。
//! 3. `ratio = B の中央値 / A の中央値 <= 1.00` を非後退条件とするが、
//!    本 Linux 開発機（共有機）は専有ゲートではないため **record_only**
//!    とし、性能比は hard assert しない（`docs/perf/logs/optimizer-
//!    device-step-2175/README.md` 等と同じ record_only 方針）。
//! 4. 改善量の上限（ステップ時間に占める取得時間の割合）は事前に約束
//!    せず、計測値をそのまま記録する。
//!
//! 実行: `cargo test -p fandhe-ai --release --test
//! data_loader_prefetch_bench -- --ignored --nocapture`

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::TensorDataset;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::data::{
    Dataset, PrefetchConfig, PrefetchDataLoader, RandomSampler, SequentialSampler,
};

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const TRIALS: usize = 5;

/// バッチの `to_bits`（`f32`）を fold した checksum（決定性契約 R2 の
/// 直接検証を兼ねる。A/B 間・5 run 間で完全一致する必要がある）。
fn fold_checksum(acc: u64, values: &[f32]) -> u64 {
    values.iter().fold(acc, |a, &v| {
        a.wrapping_mul(1_099_511_628_211)
            .wrapping_add(u64::from(v.to_bits()))
    })
}

// ---- W1: 取得（`Dataset::batch`）がボトルネックになる構成 ----

/// 添字だけで決まる純関数のまま、`batch()` に固定回数の整数演算ループ
/// （ホスト側の重い前処理を模す）を追加したデータセット。`TensorDataset`
/// への委譲前にコストを払うだけで、出力（決定性契約）は変えない。
struct HeavyDataset {
    inner: TensorDataset<f32>,
    cost_iters: u64,
}

impl Dataset for HeavyDataset {
    type Batch = Tensor<f32>;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validate(&self) -> Result<(), fandhe_ai_tensor_core::data::DataError> {
        self.inner.validate()
    }

    fn batch(
        &self,
        indices: &[usize],
    ) -> Result<Self::Batch, fandhe_ai_tensor_core::data::DataError> {
        let mut acc: u64 = indices.len() as u64;
        for i in 0..self.cost_iters {
            acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
        }
        std::hint::black_box(acc);
        self.inner.batch(indices)
    }
}

const W1_LEN: usize = 64;
const W1_BATCH_SIZE: usize = 4;
/// `batch()` 1 回あたりの重さ（整数演算ループの反復回数）。実測で
/// 数百マイクロ秒〜数ミリ秒程度になるよう調整。
const W1_COST_ITERS: u64 = 20_000_000;
/// 消費側の疑似学習ステップ固定コスト（busy loop）。
const W1_CONSUME_COST_ITERS: u64 = 500_000;

fn w1_consume_step(seed: &mut u64) {
    for _ in 0..W1_CONSUME_COST_ITERS {
        *seed = seed.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
    }
    std::hint::black_box(*seed);
}

/// W1 の 1 epoch を走らせ、`(所要秒, checksum)` を返す。`num_workers ==
/// 0` は逐次経路、`num_workers >= 1` は prefetch 経路。
fn run_w1_epoch(num_workers: usize) -> (f64, u64) {
    let ds = HeavyDataset {
        inner: TensorDataset::new(
            Tensor::new((0..W1_LEN as u32).map(|v| v as f32).collect(), &[W1_LEN])
                .expect("test fixture: W1 tensor 構築に失敗"),
        )
        .expect("test fixture: TensorDataset::new が失敗"),
        cost_iters: W1_COST_ITERS,
    };
    let sampler = SequentialSampler::new(W1_LEN, W1_BATCH_SIZE, false)
        .expect("test fixture: SequentialSampler::new が失敗");
    let depth = (2 * num_workers.max(1)).max(1);
    let config =
        PrefetchConfig::new(num_workers, depth).expect("test fixture: PrefetchConfig::new が失敗");
    let mut loader = PrefetchDataLoader::new(ds, sampler, config)
        .expect("test fixture: PrefetchDataLoader::new が失敗");

    let mut checksum = 0u64;
    let mut consume_seed = 0x1234_5678_u64;
    let start = Instant::now();
    for batch in loader.iter() {
        let batch = batch.expect("test fixture: W1 batch が失敗");
        checksum = fold_checksum(checksum, batch.host_slice().as_ref());
        w1_consume_step(&mut consume_seed);
    }
    (start.elapsed().as_secs_f64(), checksum)
}

fn run_w1_trials(num_workers: usize) -> (Vec<f64>, u64) {
    // warmup: 初回呼び出しのページング等の揺らぎを本計測から除く。
    let _ = run_w1_epoch(num_workers);

    let mut secs = Vec::with_capacity(TRIALS);
    let mut checksum: Option<u64> = None;
    for _ in 0..TRIALS {
        let (s, c) = run_w1_epoch(num_workers);
        secs.push(s);
        match checksum {
            None => checksum = Some(c),
            Some(expected) => assert_eq!(
                c, expected,
                "W1: num_workers={num_workers} で run 間の checksum が不一致\
                 （決定性契約 R2 違反）"
            ),
        }
    }
    (secs, checksum.expect("TRIALS >= 1"))
}

#[test]
#[ignore = "性能 A/B（R3〜R5）。--release --nocapture での明示実行を想定"]
fn w1_prefetch_reduces_epoch_wall_clock_when_batch_is_slow() {
    let (a_secs, a_checksum) = run_w1_trials(0);
    let a_median = median_q1_q3(&a_secs)
        .expect("TRIALS 個の non-NaN サンプル")
        .median;

    for workers in [2usize, 4] {
        let (b_secs, b_checksum) = run_w1_trials(workers);
        assert_eq!(
            b_checksum, a_checksum,
            "W1: num_workers={workers} が逐次版（num_workers=0）と checksum\
             不一致（決定性契約 R2 違反）"
        );
        let b_median = median_q1_q3(&b_secs)
            .expect("TRIALS 個の non-NaN サンプル")
            .median;
        let ratio = b_median / a_median.max(f64::EPSILON);
        println!(
            "[data_loader_prefetch_bench:W1] num_workers={workers} \
             a_median_s(seq)={a_median:.6} b_median_s(prefetch)={b_median:.6} \
             ratio={ratio:.3} checksum=0x{a_checksum:016x} \
             — record only, non-gating（本ファイル冒頭コメント参照。\
             共有 Linux 開発機は専有ゲートではない）"
        );
    }
}

// ---- W2: fit 相当の大型データセット学習ループ ----

const W2_N: usize = 512;
const W2_D_IN: usize = 16;
const W2_D_HIDDEN: usize = 32;
const W2_D_OUT: usize = 4;
const W2_BATCH_SIZE: usize = 16;
const W2_EPOCHS: usize = 3;
const W2_SEED: u64 = 0x2183_2183;

fn w2_gen_dataset() -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(0xDEC0_DE2183);
    let x = rng.fill_vec(W2_N * W2_D_IN);
    let y = rng.fill_vec(W2_N * W2_D_OUT);
    (
        Tensor::new(x, &[W2_N, W2_D_IN]).expect("test fixture: W2 x tensor 構築に失敗"),
        Tensor::new(y, &[W2_N, W2_D_OUT]).expect("test fixture: W2 y tensor 構築に失敗"),
    )
}

fn w2_build_model() -> Sequential {
    Sequential::new()
        .add_linear(W2_D_IN, W2_D_HIDDEN, 0x2183_0001)
        .expect("test fixture: 層 1 の構築に失敗")
        .add_relu()
        .add_linear(W2_D_HIDDEN, W2_D_OUT, 0x2183_0002)
        .expect("test fixture: 層 2 の構築に失敗")
}

/// `num_workers` に応じて逐次経路（`0`）または prefetch 経路（`>=1`）で
/// `W2_EPOCHS` epoch 分の学習を回し、`(所要秒, 最終パラメータの
/// checksum)` を返す。`Sequential::fit` は経由しない（facade 保留方針。
/// `docs/tensor-core-data-prefetch-decision.md` §2.4）。
///
/// 特徴量・ラベルの 2 つの `TensorDataset` はタプル `(ds_x, ds_y)` の
/// まま `PrefetchDataLoader` へ渡す（`Dataset for (A, B)` の
/// `Batch = (A::Batch, B::Batch)` 実装。`crates/tensor-core/src/
/// data.rs`）。これにより `Dataset::batch`（特徴量・ラベルの実バッチ
/// 構築）自体が worker スレッド側で実行され、次バッチの取得と学習
/// ステップ（消費側）が重ね合わさる（レビュー指摘・イシュー #2183
/// コメント。添字だけを worker 側で複製し実バッチ構築を consumer 側の
/// 逐次実行に残す旧構成では prefetch の効果を測定できていなかった）。
fn run_w2_training(num_workers: usize) -> (f64, u64) {
    fandhe_ai::manual_seed(W2_SEED);
    let (x_data, y_data) = w2_gen_dataset();
    let ds_x = TensorDataset::new(x_data).expect("test fixture: TensorDataset::new(x) が失敗");
    let ds_y = TensorDataset::new(y_data).expect("test fixture: TensorDataset::new(y) が失敗");
    let sampler = RandomSampler::new(W2_N, W2_BATCH_SIZE, false)
        .expect("test fixture: RandomSampler::new が失敗");
    let depth = (2 * num_workers.max(1)).max(1);
    let config =
        PrefetchConfig::new(num_workers, depth).expect("test fixture: PrefetchConfig::new が失敗");

    let mut model = w2_build_model();
    let mut sgd = Sgd::new(SgdConfig::new(0.02)).expect("test fixture: Sgd::new が失敗");

    let mut loader = PrefetchDataLoader::new((ds_x, ds_y), sampler, config)
        .expect("test fixture: PrefetchDataLoader::new が失敗");

    let start = Instant::now();
    for _ in 0..W2_EPOCHS {
        for batch in loader.iter() {
            let (x_batch, y_batch) = batch.expect("test fixture: W2 batch 取得が失敗");

            let updated = {
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                let x = tape.var(&x_batch);
                let y = tape.var(&y_batch);
                let pred = bound
                    .forward(&tape, &x)
                    .expect("test fixture: forward が失敗");
                let loss = pred.mse_loss(&y).expect("test fixture: mse_loss が失敗");
                let grads = tape.backward(&loss).expect("test fixture: backward が失敗");
                let grad_refs = bound
                    .trainable_grads(&grads)
                    .expect("test fixture: trainable_grads が失敗");
                let param_refs = model.trainable_parameters();
                sgd.step(&param_refs, &grad_refs)
                    .expect("test fixture: Sgd::step が失敗")
            };
            model
                .apply_parameters(updated)
                .expect("test fixture: apply_parameters が失敗");
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    let mut checksum = 0u64;
    for p in model.trainable_parameters() {
        checksum = fold_checksum(checksum, p.host_slice().as_ref());
    }
    (elapsed, checksum)
}

fn run_w2_trials(num_workers: usize) -> (Vec<f64>, u64) {
    let _ = run_w2_training(num_workers); // warmup

    let mut secs = Vec::with_capacity(TRIALS);
    let mut checksum: Option<u64> = None;
    for _ in 0..TRIALS {
        let (s, c) = run_w2_training(num_workers);
        secs.push(s);
        match checksum {
            None => checksum = Some(c),
            Some(expected) => assert_eq!(
                c, expected,
                "W2: num_workers={num_workers} で run 間の最終パラメータ\
                 checksum が不一致（同一 manual_seed 下では決定的なはず）"
            ),
        }
    }
    (secs, checksum.expect("TRIALS >= 1"))
}

#[test]
#[ignore = "性能 A/B（R3〜R5）。--release --nocapture での明示実行を想定"]
fn w2_prefetch_training_matches_sequential_checksum_and_records_ratio() {
    let (a_secs, a_checksum) = run_w2_trials(0);
    let a_median = median_q1_q3(&a_secs)
        .expect("TRIALS 個の non-NaN サンプル")
        .median;

    for workers in [2usize, 4] {
        let (b_secs, b_checksum) = run_w2_trials(workers);
        assert_eq!(
            b_checksum, a_checksum,
            "W2: num_workers={workers} が逐次版（num_workers=0）と最終\
             パラメータ checksum 不一致（決定性契約 R2 違反）"
        );
        let b_median = median_q1_q3(&b_secs)
            .expect("TRIALS 個の non-NaN サンプル")
            .median;
        let ratio = b_median / a_median.max(f64::EPSILON);
        println!(
            "[data_loader_prefetch_bench:W2] num_workers={workers} \
             a_median_s(seq)={a_median:.6} b_median_s(prefetch)={b_median:.6} \
             ratio={ratio:.3} final_param_checksum=0x{a_checksum:016x} \
             — record only, non-gating（本ファイル冒頭コメント参照。\
             共有 Linux 開発機は専有ゲートではない）"
        );
    }
}
