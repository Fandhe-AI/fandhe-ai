//! MNIST 形式データで `compile` → `fit` → `evaluate` → `predict` を通す example（イシュー #4339・#4340・親 #4338）。
//!
//! 実行:
//! - 合成データ（既定・オフライン）: `cargo run -p fandhe-ai --example mnist_mlp --release`
//!   （debug でも動くが遅い）
//! - 実 MNIST: `cargo run -p fandhe-ai --example mnist_mlp --release -- <IDX ディレクトリ>`
//!
//! 実データの用意（利用者側の責務。example 自身はネットワークに出ず、sha256 値も同梱しない）:
//! 1. 信頼できる配布元から MNIST の 4 ファイル（`.gz`）を取得する。
//! 2. `sha256sum` 等で、配布元が示す値と照合してデータの真正性を確かめる。
//! 3. `gunzip` で展開し、次の固定ファイル名で 1 つのディレクトリに置く
//!    （std だけでは gzip を展開できないため、展開済みであることを前提にする）。
//!    - `train-images-idx3-ubyte`（`16 + N*784` バイト。標準の N は 60000）
//!    - `train-labels-idx1-ubyte`（`8 + N` バイト）
//!    - `t10k-images-idx3-ubyte`（`16 + N*784` バイト。標準の N は 10000）
//!    - `t10k-labels-idx1-ubyte`（`8 + N` バイト）
//!
//! - 引数なしのときは入力 784 次元・10 クラスの MNIST と同じ形の**合成データ**を使う
//!   （実 MNIST ではなく、オフラインで exit 0 になる）。
//! - IDX は magic・次元・件数・長さを先に検証し、不正な入力は型付きエラー（`IdxError`）で
//!   非 0 終了にする。ファイル長にも上限を置き、巨大なヘッダによるメモリ枯渇を防ぐ。
//! - 使うのは `fandhe_ai`（facade）の公開面と std だけ。`bench_harness` は非公開クレートで
//!   facade の dev-dependency のため、公開パッケージに同梱される example からは使えない。
//!   乱数は facade 公開の `Generator::new(固定シード)` の `multinomial`／`normal` で作る
//!   （グローバル RNG を消費せず、学習用・評価用・クラスごとにシードを固定して決定的）。
//! - `FitConfig::shuffle(true)` はグローバル RNG（`fandhe_ai::manual_seed`）を消費するため、
//!   `main` の先頭で固定シードを設定し、再現性を保つ。
//! - 兄弟イシューの分割: #4341 は `train_and_report` が返す学習済みモデルへ
//!   save_model／load_model の bit 一致確認を足す。

use fandhe_ai::compat::{FitConfig, Loss, Metrics, MetricsResult, Optimizer, Sequential};
use fandhe_ai::optim::AdamConfig;
use fandhe_ai::{AutodiffError, Generator, Tensor};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

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

// ---- IDX（MNIST 配布形式）読み込み（イシュー #4340） ----
// 外部ファイルを読むため、ヘッダ検証（magic → ndim → 次元 → 長さの厳密一致）を終えてから
// 確保・変換する。長さ計算は checked 演算で、不正入力でパニックしない。

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";
/// 画像件数の上限（標準 MNIST は最大 60000）。
const MAX_ITEMS: u64 = 100_000;
/// ファイル長の上限（train images は 47,040,016 バイト）。
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
const IDX_UBYTE: u8 = 0x08;
const IMAGE_SIDE: u32 = 28;

/// IDX 読み込みの型付きエラー。`main` で `DynError` へ載せて非 0 終了にする。
#[derive(Debug)]
enum IdxError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    NotRegularFile {
        path: PathBuf,
    },
    Gzipped,
    TooLarge {
        len: u64,
    },
    BadMagic {
        expected_ndim: u8,
        found: [u8; 4],
    },
    BadDims {
        rows: u32,
        cols: u32,
    },
    BadCount {
        count: u64,
    },
    LengthMismatch {
        expected: Option<u64>,
        actual: u64,
    },
    CountMismatch {
        images: usize,
        labels: usize,
    },
    LabelOutOfRange {
        index: usize,
        value: u8,
    },
}

impl fmt::Display for IdxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRegularFile { path } => write!(
                f,
                "IDX ファイル {} が通常ファイルでない（FIFO・デバイス等は受け付けない）",
                path.display()
            ),
            Self::Io { path, source } if source.kind() == std::io::ErrorKind::NotFound => write!(
                f,
                "IDX ファイル {} が見つからない（展開済みの 4 ファイルを置く）",
                path.display()
            ),
            Self::Io { path, source } => {
                write!(f, "IDX ファイル {} を読めない: {source}", path.display())
            }
            Self::Gzipped => write!(f, "gzip 圧縮のままである（gunzip で展開してから渡す）"),
            Self::TooLarge { len } => {
                write!(f, "ファイルが上限 {MAX_FILE_BYTES} バイトを超える（{len}）")
            }
            Self::BadMagic {
                expected_ndim,
                found,
            } => write!(
                f,
                "IDX の magic が不正（期待: 0 0 8 {expected_ndim}、実際: {found:?}）"
            ),
            Self::BadDims { rows, cols } => {
                write!(
                    f,
                    "画像サイズが {IMAGE_SIDE}x{IMAGE_SIDE} でない（{rows}x{cols}）"
                )
            }
            Self::BadCount { count } => {
                write!(f, "件数が不正（1..={MAX_ITEMS} の範囲外: {count}）")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "ファイル長がヘッダと一致しない（期待: {expected:?}、実際: {actual}）"
                )
            }
            Self::CountMismatch { images, labels } => {
                write!(f, "画像件数 {images} とラベル件数 {labels} が一致しない")
            }
            Self::LabelOutOfRange { index, value } => {
                write!(
                    f,
                    "ラベル {index} 件目の値 {value} が 0..={} の範囲外",
                    NUM_CLASSES - 1
                )
            }
        }
    }
}

impl std::error::Error for IdxError {}

fn be_u32(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// magic（先頭 4 バイト）を検証して件数を返す。gzip 先頭は専用エラーにする。
fn check_magic(bytes: &[u8], ndim: u8) -> Result<u64, IdxError> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        return Err(IdxError::Gzipped);
    }
    let mut found = [0u8; 4];
    for (d, s) in found.iter_mut().zip(bytes) {
        *d = *s;
    }
    if bytes.len() < 4
        || found[0] != 0
        || found[1] != 0
        || found[2] != IDX_UBYTE
        || found[3] != ndim
    {
        return Err(IdxError::BadMagic {
            expected_ndim: ndim,
            found,
        });
    }
    let count = u64::from(be_u32(bytes, 4).ok_or(IdxError::LengthMismatch {
        expected: None,
        actual: bytes.len() as u64,
    })?);
    if count == 0 || count > MAX_ITEMS {
        return Err(IdxError::BadCount { count });
    }
    Ok(count)
}

/// 画像ファイル（3 次元）を `(件数, [0,1] 正規化した画素)` へ解析する。
fn parse_idx_images(bytes: &[u8]) -> Result<(usize, Vec<f32>), IdxError> {
    let count = check_magic(bytes, 3)?;
    let short = || IdxError::LengthMismatch {
        expected: None,
        actual: bytes.len() as u64,
    };
    let rows = be_u32(bytes, 8).ok_or_else(short)?;
    let cols = be_u32(bytes, 12).ok_or_else(short)?;
    if rows != IMAGE_SIDE || cols != IMAGE_SIDE {
        return Err(IdxError::BadDims { rows, cols });
    }
    let expected = count
        .checked_mul(IMAGE_LEN as u64)
        .and_then(|n| n.checked_add(16));
    let actual = bytes.len() as u64;
    if expected != Some(actual) {
        return Err(IdxError::LengthMismatch { expected, actual });
    }
    let data = bytes[16..].iter().map(|&p| f32::from(p) / 255.0).collect();
    Ok((count as usize, data))
}

/// ラベルファイル（1 次元）を解析する。値は 0..=9 に限る。
fn parse_idx_labels(bytes: &[u8]) -> Result<Vec<i32>, IdxError> {
    let count = check_magic(bytes, 1)?;
    let expected = count.checked_add(8);
    let actual = bytes.len() as u64;
    if expected != Some(actual) {
        return Err(IdxError::LengthMismatch { expected, actual });
    }
    bytes[8..]
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if usize::from(value) < NUM_CLASSES {
                Ok(i32::from(value))
            } else {
                Err(IdxError::LabelOutOfRange { index, value })
            }
        })
        .collect()
}

/// ファイルを上限付きで読む。メタデータと実体の食い違いに備え `take` でも上限を掛ける。
fn read_idx_file(path: &Path) -> Result<Vec<u8>, IdxError> {
    let io = |source| IdxError::Io {
        path: path.to_path_buf(),
        source,
    };
    // 通常ファイルだけを受け付ける。FIFO 等の特殊ファイルは `open` が無期限に待ち得るため、
    // オープン前に種別を拒否し、パス差し替えに備え非ブロッキングで開いて種別を再検証する。
    if !std::fs::metadata(path).map_err(io)?.is_file() {
        return Err(IdxError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NONBLOCK);
    }
    let file = opts.open(path).map_err(io)?;
    let meta = file.metadata().map_err(io)?;
    if !meta.is_file() {
        return Err(IdxError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    let len = meta.len();
    if len > MAX_FILE_BYTES {
        return Err(IdxError::TooLarge { len });
    }
    let mut buf = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(io)?;
    if buf.len() as u64 > MAX_FILE_BYTES {
        return Err(IdxError::TooLarge {
            len: buf.len() as u64,
        });
    }
    Ok(buf)
}

/// 画像・ラベルの 1 組を読み、件数一致を検査して `Dataset` にする。
fn load_idx_split(dir: &Path, images: &str, labels: &str) -> Result<Dataset, DynError> {
    let (n, data) = parse_idx_images(&read_idx_file(&dir.join(images))?)?;
    let y = parse_idx_labels(&read_idx_file(&dir.join(labels))?)?;
    if y.len() != n {
        return Err(IdxError::CountMismatch {
            images: n,
            labels: y.len(),
        }
        .into());
    }
    Ok(Dataset {
        x: Tensor::new(data, &[n, IMAGE_LEN])?,
        y: Tensor::new(y, &[n])?,
    })
}

/// `dir` 直下の固定名 4 ファイルから `(train, test)` を読む。
fn load_idx_mnist(dir: &Path) -> Result<(Dataset, Dataset), DynError> {
    Ok((
        load_idx_split(dir, TRAIN_IMAGES, TRAIN_LABELS)?,
        load_idx_split(dir, TEST_IMAGES, TEST_LABELS)?,
    ))
}

/// データの取得元。
#[derive(Debug, PartialEq)]
enum DataSource {
    Synthetic,
    Idx(PathBuf),
}

/// 引数 0 個は合成データ、1 個は IDX ディレクトリ、2 個以上はエラー。
fn parse_args(args: impl Iterator<Item = std::ffi::OsString>) -> Result<DataSource, DynError> {
    // UTF-8 でない引数でもパニックしないよう OsString のまま扱う。
    let mut args: Vec<std::ffi::OsString> = args.collect();
    match args.len() {
        0 => Ok(DataSource::Synthetic),
        1 => Ok(DataSource::Idx(PathBuf::from(args.remove(0)))),
        _ => Err(format!("引数は IDX ディレクトリ 1 個まで（指定: {args:?}）").into()),
    }
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

fn main() -> Result<(), DynError> {
    let source = parse_args(std::env::args_os().skip(1))?;
    fandhe_ai::manual_seed(SEED_GLOBAL);
    let (train, test, kind) = match source {
        DataSource::Synthetic => (
            synthetic_mnist(N_TRAIN, SEED_TRAIN)?,
            synthetic_mnist(N_TEST, SEED_TEST)?,
            "合成 MNIST 形式データ",
        ),
        DataSource::Idx(dir) => {
            let (train, test) = load_idx_mnist(&dir)?;
            (train, test, "MNIST（IDX）")
        }
    };
    println!(
        "== MLP 784→{HIDDEN}→10 / {kind}（train={}, test={}） ==",
        train.x.shape()[0],
        test.x.shape()[0]
    );
    let mut model = build_model()?;
    let report = train_and_report(&mut model, &train, &test)?;
    check_result(&report)
}

#[cfg(test)]
#[path = "../tests/common/temp_dir.rs"]
mod temp_dir;

#[cfg(test)]
mod tests {
    use super::temp_dir::TempDirGuard;
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
        // 評価入力が学習入力の先頭 N_TEST 件と一致しない（別シードで生成された）ことを、
        // 実際に生成したデータで確認する。クラス原型とラベル順は共通のため入力でのみ区別できる。
        let train_x = train.x.as_slice().expect("画素スライスを取得できること");
        let test_x = test.x.as_slice().expect("画素スライスを取得できること");
        assert_ne!(&train_x[..test_x.len()], test_x);
        let mut model = build_model().unwrap();
        let report = train_and_report(&mut model, &train, &test).unwrap();
        assert_eq!(report.epoch_loss.len(), EPOCHS);
        check_result(&report).unwrap();
    }

    #[test]
    fn parse_args_variants() {
        use std::ffi::OsString;
        assert_eq!(
            parse_args(std::iter::empty()).unwrap(),
            DataSource::Synthetic
        );
        assert_eq!(
            parse_args([OsString::from("d")].into_iter()).unwrap(),
            DataSource::Idx(PathBuf::from("d"))
        );
        assert!(parse_args([OsString::from("a"), OsString::from("b")].into_iter()).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let bad = OsString::from_vec(vec![0xff, 0xfe]);
            assert!(matches!(
                parse_args([bad].into_iter()).unwrap(),
                DataSource::Idx(_)
            ));
        }
    }

    fn images_bytes(count: u32, rows: u32, cols: u32, body: usize) -> Vec<u8> {
        let mut v = vec![0, 0, 8, 3];
        for n in [count, rows, cols] {
            v.extend(n.to_be_bytes());
        }
        v.extend(std::iter::repeat_n(255u8, body));
        v
    }

    fn labels_bytes(vals: &[u8]) -> Vec<u8> {
        let mut v = vec![0, 0, 8, 1];
        v.extend((vals.len() as u32).to_be_bytes());
        v.extend(vals);
        v
    }

    #[test]
    fn idx_parses_valid() {
        let mut img = images_bytes(3, 28, 28, 3 * IMAGE_LEN);
        img[16] = 0;
        let (n, data) = parse_idx_images(&img).unwrap();
        assert_eq!(n, 3);
        assert_eq!(data.len(), 3 * IMAGE_LEN);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(
            parse_idx_labels(&labels_bytes(&[0, 5, 9])).unwrap(),
            vec![0, 5, 9]
        );
    }

    #[test]
    fn idx_rejects_bad_images() {
        let ok = images_bytes(2, 28, 28, 2 * IMAGE_LEN);
        let mut m = ok.clone();
        m[0] = 1;
        assert!(matches!(
            parse_idx_images(&m),
            Err(IdxError::BadMagic { .. })
        ));
        let mut m = ok.clone();
        m[2] = 9;
        assert!(matches!(
            parse_idx_images(&m),
            Err(IdxError::BadMagic { .. })
        ));
        // ラベルファイル（ndim=1）を画像として渡す。
        assert!(matches!(
            parse_idx_images(&labels_bytes(&[1, 2])),
            Err(IdxError::BadMagic { .. })
        ));
        assert!(matches!(
            parse_idx_images(&images_bytes(2, 27, 28, 2 * IMAGE_LEN)),
            Err(IdxError::BadDims { .. })
        ));
        assert!(matches!(
            parse_idx_images(&images_bytes(0, 28, 28, 0)),
            Err(IdxError::BadCount { .. })
        ));
        assert!(matches!(
            parse_idx_images(&images_bytes(100_001, 28, 28, 0)),
            Err(IdxError::BadCount { .. })
        ));
        assert!(matches!(
            parse_idx_images(&ok[..ok.len() - 1]),
            Err(IdxError::LengthMismatch { .. })
        ));
        let mut long = ok.clone();
        long.push(0);
        assert!(matches!(
            parse_idx_images(&long),
            Err(IdxError::LengthMismatch { .. })
        ));
        assert!(matches!(
            parse_idx_images(&ok[..10]),
            Err(IdxError::LengthMismatch { .. })
        ));
        assert!(matches!(
            parse_idx_images(&[0, 0, 8]),
            Err(IdxError::BadMagic { .. })
        ));
        assert!(matches!(
            parse_idx_images(&[0x1f, 0x8b, 8, 0, 0, 0]),
            Err(IdxError::Gzipped)
        ));
        // 件数が大きくても長さ不一致で拒否され、確保しない。
        assert!(matches!(
            parse_idx_images(&images_bytes(100_000, 28, 28, 10)),
            Err(IdxError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn idx_rejects_bad_labels() {
        assert!(matches!(
            parse_idx_labels(&labels_bytes(&[0, 10])),
            Err(IdxError::LabelOutOfRange {
                index: 1,
                value: 10
            })
        ));
        let ok = labels_bytes(&[1, 2, 3]);
        assert!(matches!(
            parse_idx_labels(&ok[..ok.len() - 1]),
            Err(IdxError::LengthMismatch { .. })
        ));
        let mut long = ok.clone();
        long.push(0);
        assert!(matches!(
            parse_idx_labels(&long),
            Err(IdxError::LengthMismatch { .. })
        ));
        assert!(matches!(
            parse_idx_labels(&images_bytes(1, 28, 28, IMAGE_LEN)),
            Err(IdxError::BadMagic { .. })
        ));
        assert!(matches!(
            parse_idx_labels(&[0x1f, 0x8b]),
            Err(IdxError::Gzipped)
        ));
    }

    #[test]
    fn load_idx_dir_roundtrip_and_missing() {
        let d = TempDirGuard::new("ok");
        let img = images_bytes(3, 28, 28, 3 * IMAGE_LEN);
        let lab = labels_bytes(&[0, 1, 2]);
        for (name, bytes) in [
            (TRAIN_IMAGES, &img),
            (TRAIN_LABELS, &lab),
            (TEST_IMAGES, &img),
            (TEST_LABELS, &lab),
        ] {
            std::fs::write(d.path().join(name), bytes).unwrap();
        }
        let (train, test) = load_idx_mnist(d.path()).unwrap();
        assert_eq!(train.x.shape(), [3, IMAGE_LEN]);
        assert_eq!(test.y.shape(), [3]);
        // 件数不一致
        std::fs::write(d.path().join(TEST_LABELS), labels_bytes(&[0, 1])).unwrap();
        let e = load_idx_mnist(d.path()).err().unwrap();
        assert!(matches!(
            e.downcast_ref::<IdxError>(),
            Some(IdxError::CountMismatch {
                images: 3,
                labels: 2
            })
        ));
        // 欠落
        std::fs::remove_file(d.path().join(TRAIN_IMAGES)).unwrap();
        let e = load_idx_mnist(d.path()).err().unwrap();
        assert!(matches!(
            e.downcast_ref::<IdxError>(),
            Some(IdxError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[cfg(unix)]
    #[test]
    fn read_idx_file_rejects_fifo_without_blocking() {
        let d = TempDirGuard::new("fifo");
        let p = d.path().join("fifo");
        let c = std::ffi::CString::new(p.to_str().unwrap()).unwrap();
        // SAFETY: c は NUL 終端の有効な C 文字列。
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        assert!(matches!(
            read_idx_file(&p),
            Err(IdxError::NotRegularFile { .. })
        ));
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
