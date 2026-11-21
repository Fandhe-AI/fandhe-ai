//! 合成 MNIST 形式データで `compile` → `fit` → `evaluate` → `predict` を通す example（イシュー #4339・親 #4338）。
//!
//! 実行: `cargo run -p fandhe-ai --example mnist_mlp --release`（debug でも動くが遅い）。
//!
//! - 入力 784 次元・10 クラスの MNIST と同じ形の**合成データ**を使う。実 MNIST ではなく、
//!   ネットワーク取得もしない（オフラインで exit 0 になる）。
//! - 使うのは `fandhe_ai`（facade）の公開面と std だけ。`bench_harness` は非公開クレートで
//!   facade の dev-dependency のため、公開パッケージに同梱される example からは使えない。
//!   乱数は facade 公開の `Generator::new(固定シード)` の `multinomial`／`normal` で作る
//!   （グローバル RNG を消費せず、学習用・評価用・クラスごとにシードを固定して決定的）。
//! - `FitConfig::shuffle(true)` はグローバル RNG（`fandhe_ai::manual_seed`）を消費するため、
//!   `main` の先頭で固定シードを設定し、再現性を保つ。
//! - 兄弟イシューが土台にする分割: #4340 は `synthetic_mnist` と並ぶ IDX 読み込みを足し
//!   （引数なしは合成データのまま）、#4341 は `train_and_report` が返す学習済みモデルへ
//!   save_model／load_model の bit 一致確認を足す。

use fandhe_ai::compat::{FitConfig, Loss, Metrics, MetricsResult, Optimizer, Sequential};
use fandhe_ai::optim::AdamConfig;
use fandhe_ai::{AutodiffError, Generator, Tensor};

type DynError = Box<dyn std::error::Error>;

const IMAGE_LEN: usize = 28 * 28;
const NUM_CLASSES: usize = 10;
const HIDDEN: usize = 128;
const EPOCHS: usize = 3;
const BATCH_SIZE: usize = 64;
const N_TRAIN: usize = 4096;
const N_TEST: usize = 1024;
const LR: f32 = 1e-3;
const NOISE_STD: f32 = 0.15;
const MIN_ACCURACY: f32 = 0.5;
const SEED_GLOBAL: u64 = 42;
const SEED_TRAIN: u64 = 0x4339_0001;
const SEED_TEST: u64 = 0x4339_0002;

/// 画像 `[N, 784]` とクラスラベル `[N]` の組。
struct Dataset {
    x: Tensor<f32>,
    y: Tensor<i32>,
}

/// クラスごとに決定的な点灯画素パターン（プロトタイプ）を作る。
/// 点灯画素は `Generator::multinomial`（一様重み・復元抽出）で選ぶ。
fn class_prototype(class: usize) -> Result<Vec<f32>, DynError> {
    let mut proto = vec![0.2f32; IMAGE_LEN];
    let mut rng = Generator::new(0x9000 + class as u64);
    let weights = Tensor::new(vec![1.0f32; IMAGE_LEN], &[IMAGE_LEN])?;
    let lit = rng.multinomial(&weights, IMAGE_LEN / 6, true)?;
    for &idx in lit.host_slice().iter() {
        proto[idx as usize] = 0.6;
    }
    Ok(proto)
}

/// 合成 MNIST 形式データ `n` 件を生成する。ラベルは `i % 10` の巡回。
/// ノイズは `seed` 固定の `Generator::normal`（グローバル RNG は消費しない）。
fn synthetic_mnist(n: usize, seed: u64) -> Result<Dataset, DynError> {
    let protos = (0..NUM_CLASSES)
        .map(class_prototype)
        .collect::<Result<Vec<_>, _>>()?;
    let mut rng = Generator::new(seed);
    let noise = rng.normal(0.0, NOISE_STD, &[n * IMAGE_LEN])?;
    let noise = noise.host_slice();
    let mut data = Vec::with_capacity(n * IMAGE_LEN);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let class = i % NUM_CLASSES;
        let row = &noise[i * IMAGE_LEN..(i + 1) * IMAGE_LEN];
        data.extend(
            protos[class]
                .iter()
                .zip(row)
                .map(|(&p, &e)| (p + e).clamp(0.0, 1.0)),
        );
        labels.push(class as i32);
    }
    Ok(Dataset {
        x: Tensor::new(data, &[n, IMAGE_LEN])?,
        y: Tensor::new(labels, &[n])?,
    })
}

/// 784→128→10 の MLP を組んで Adam + CrossEntropy で compile する。
fn build_model() -> Result<Sequential, AutodiffError> {
    let mut model = Sequential::new()
        .add_linear(IMAGE_LEN, HIDDEN, 1)?
        .add_relu()
        .add_linear(HIDDEN, NUM_CLASSES, 2)?;
    model.compile(
        Optimizer::Adam(AdamConfig {
            lr: LR,
            ..AdamConfig::default()
        }),
        Loss::CrossEntropy,
    )?;
    Ok(model)
}

/// 学習結果（epoch ごとの学習 loss と test 指標）。
struct Report {
    epoch_loss: Vec<f32>,
    test_loss: f32,
    test_acc: f32,
}

/// `fit` → `evaluate` → `predict` を実行し、結果を標準出力へ表示する。
fn train_and_report(
    model: &mut Sequential,
    train: &Dataset,
    test: &Dataset,
) -> Result<Report, DynError> {
    let history = model.fit(
        &train.x,
        &train.y,
        FitConfig::new(EPOCHS, BATCH_SIZE).shuffle(true),
    )?;
    for (i, loss) in history.loss.iter().enumerate() {
        println!("epoch {}/{}: loss = {loss:.4}", i + 1, EPOCHS);
    }
    let test_loss = model.evaluate(&test.x, &test.y, BATCH_SIZE)?;
    let logits = model.predict(&test.x)?;
    let metrics = MetricsResult::compute(&[Metrics::Accuracy], &logits, &test.y)?;
    let test_acc = metrics.accuracy.ok_or("accuracy が計算されなかった")?;
    println!("test loss = {test_loss:.4}, test accuracy = {test_acc:.4}");
    Ok(Report {
        epoch_loss: history.loss,
        test_loss,
        test_acc,
    })
}

/// 健全性検査。失敗すると非 0 終了にする。
fn check_result(r: &Report) -> Result<(), DynError> {
    let (Some(&first), Some(&last)) = (r.epoch_loss.first(), r.epoch_loss.last()) else {
        return Err("学習履歴（history）が空である".into());
    };
    if !r.epoch_loss.iter().all(|l| l.is_finite()) || !r.test_loss.is_finite() {
        return Err("loss が有限でない".into());
    }
    if !r.test_acc.is_finite() {
        return Err("accuracy が有限でない".into());
    }
    if last.partial_cmp(&first) != Some(std::cmp::Ordering::Less) {
        return Err(format!("loss が減少していない（初回 {first}, 最終 {last}）").into());
    }
    if r.test_acc.partial_cmp(&MIN_ACCURACY) != Some(std::cmp::Ordering::Greater) {
        return Err(format!("test accuracy {} が下限 {MIN_ACCURACY} 以下", r.test_acc).into());
    }
    Ok(())
}

/// 本 example は引数を受け付けない（#4340 が path 引数を追加する）。
fn parse_args(args: impl Iterator<Item = std::ffi::OsString>) -> Result<(), DynError> {
    // UTF-8 でない引数でもパニックしないよう OsString のまま扱う。
    let extra: Vec<std::ffi::OsString> = args.collect();
    if extra.is_empty() {
        Ok(())
    } else {
        Err(format!("引数は受け付けない（指定: {extra:?}）").into())
    }
}

fn main() -> Result<(), DynError> {
    parse_args(std::env::args_os().skip(1))?;
    fandhe_ai::manual_seed(SEED_GLOBAL);
    let train = synthetic_mnist(N_TRAIN, SEED_TRAIN)?;
    let test = synthetic_mnist(N_TEST, SEED_TEST)?;
    println!("== MLP 784→{HIDDEN}→10 / 合成 MNIST 形式データ（train={N_TRAIN}, test={N_TEST}） ==");
    let mut model = build_model()?;
    let report = train_and_report(&mut model, &train, &test)?;
    check_result(&report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_is_deterministic_and_shaped() {
        let a = synthetic_mnist(20, 1).unwrap();
        let b = synthetic_mnist(20, 1).unwrap();
        assert_eq!(a.x.shape(), [20, IMAGE_LEN]);
        assert_eq!(a.y.shape(), [20]);
        let xa = a.x.as_slice().expect("画素スライスを取得できること");
        let xb = b.x.as_slice().expect("画素スライスを取得できること");
        assert_eq!(xa, xb);
        // 値域だけでなく 10 クラスの巡回（i % NUM_CLASSES）そのものを検証する。
        // 単一クラスへ退化したデータでも精度 1.0 になる回帰を防ぐ。
        let expected: Vec<i32> = (0..20).map(|i| (i % NUM_CLASSES) as i32).collect();
        assert_eq!(a.y.as_slice().unwrap(), expected.as_slice());
    }

    /// 固定シードの合成データで compile→fit→evaluate→predict を通し、`check_result` で検証する。
    #[test]
    fn train_and_report_passes_check_result() {
        fandhe_ai::manual_seed(SEED_GLOBAL);
        let train = synthetic_mnist(N_TRAIN, SEED_TRAIN).unwrap();
        let test = synthetic_mnist(N_TEST, SEED_TEST).unwrap();
        let mut model = build_model().unwrap();
        let report = train_and_report(&mut model, &train, &test).unwrap();
        assert_eq!(report.epoch_loss.len(), EPOCHS);
        check_result(&report).unwrap();
    }

    #[test]
    fn parse_args_rejects_extra() {
        assert!(parse_args(std::iter::empty()).is_ok());
        assert!(parse_args([std::ffi::OsString::from("x")].into_iter()).is_err());
    }

    #[test]
    fn check_result_rejects_bad() {
        let r = |l: Vec<f32>, tl, ta| Report {
            epoch_loss: l,
            test_loss: tl,
            test_acc: ta,
        };
        assert!(check_result(&r(vec![2.0, 1.0], 1.0, 0.9)).is_ok());
        assert!(check_result(&r(vec![], 1.0, 0.9)).is_err());
        assert!(check_result(&r(vec![f32::NAN, 1.0], 1.0, 0.9)).is_err());
        assert!(check_result(&r(vec![1.0, 2.0], 1.0, 0.9)).is_err());
        assert!(check_result(&r(vec![2.0, 1.0], 1.0, 0.1)).is_err());
    }
}
