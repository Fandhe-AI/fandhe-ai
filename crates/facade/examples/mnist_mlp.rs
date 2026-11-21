//! 合成 MNIST 形式データで `compile` → `fit` → `evaluate` → `predict` を通す example（イシュー #4339・親 #4338）。
//!
//! 実行: `cargo run -p fandhe-ai --example mnist_mlp --release`（debug でも動くが遅い）。
//! 保存・読み込みを含む終了コード 0 は Linux（x86_64／aarch64）と macOS が対象で、それ以外では非 0 終了する。
//!
//! - 入力 784 次元・10 クラスの MNIST と同じ形の**合成データ**を使う。実 MNIST ではなく、
//!   ネットワーク取得もしない（オフラインで exit 0 になる）。
//! - 使うのは `fandhe_ai`（facade）の公開面と std だけ。`bench_harness` は非公開クレートで
//!   facade の dev-dependency のため、公開パッケージに同梱される example からは使えない。
//!   乱数は facade 公開の `Generator::new(固定シード)` の `multinomial`／`normal` で作る
//!   （グローバル RNG を消費せず、学習用・評価用・クラスごとにシードを固定して決定的）。
//! - `FitConfig::shuffle(true)` はグローバル RNG（`fandhe_ai::manual_seed`）を消費するため、
//!   `main` の先頭で固定シードを設定し、再現性を保つ。
//! - 学習後に一時ディレクトリへ `save_model` し、`load_model` で復元したモデルの `predict` 出力が
//!   保存前と bit 一致することを確認する（#4341）。最大差を表示し、0 でなければ非 0 終了する。
//!   一時ディレクトリ（unix では 0o700 で排他作成）は成功・失敗を問わず終了時に片付けを試みる
//!   （後始末はベストエフォートで、削除に失敗すると残りうる）。保存・読み込みは load_model の対応環境
//!   （Linux x86_64／aarch64 と macOS）が対象で、それ以外では非 0 終了する。
//! - 兄弟イシュー #4340 は `synthetic_mnist` と並ぶ IDX 読み込みを足す（引数なしは合成データのまま）。

use fandhe_ai::compat::{
    FitConfig, Loss, Metrics, MetricsResult, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::optim::AdamConfig;
use fandhe_ai::{AutodiffError, Generator, Tensor};
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

/// 保存先の一時ディレクトリを排他的に作り、drop で再帰削除する RAII ガード。
///
/// `save_and_verify` から使われ、早期 return（`?`）でも後始末を試みる（ベストエフォート。`process::exit` は使わない）。
/// 新規依存を避けるため std のみで実装する。名前に pid と `RandomState` 由来の 128 bit を含め、
/// `create_dir` の排他作成で既存パス・事前配置 symlink を再利用しない（`AlreadyExists` のみ再試行）。
struct SaveDirGuard {
    path: PathBuf,
}

/// 排他的にディレクトリを作る。unix では作成モードを 0o700 にする（umask でさらに絞られうる）。
#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(path)
}

/// 排他的にディレクトリを作る（非 unix。モード指定なし）。
#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::DirBuilder::new().create(path)
}

/// 親ディレクトリ階層が信頼できることを確認する（unix）。`parent` を正規化し、その各祖先
/// （自身と `/` を含む）が、作成したディレクトリと同じ所有者（または root）の持ち物で、
/// かつ「グループ・その他に書き込み権限がない」か「sticky bit 付き」であることを要求する。
/// 別ユーザー所有の sticky ディレクトリや、他者が書ける祖先は、エントリの差し替えで
/// 保存先・削除対象をすり替えられるため拒否する。
#[cfg(unix)]
fn check_parent_trusted(parent: &Path, child: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let owner = std::fs::symlink_metadata(child)?.uid();
    let canon = std::fs::canonicalize(parent)?;
    for dir in canon.ancestors() {
        let m = std::fs::metadata(dir)?;
        let owner_ok = m.uid() == owner || m.uid() == 0;
        let sticky = m.mode() & 0o1000 != 0;
        let closed = m.mode() & 0o022 == 0;
        if !(owner_ok && (closed || sticky)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("一時ディレクトリの祖先 {} が信頼できない", dir.display()),
            ));
        }
    }
    Ok(())
}

impl SaveDirGuard {
    fn create() -> std::io::Result<Self> {
        use std::hash::{BuildHasher, Hasher};
        let base = std::env::temp_dir();
        let mut last = None;
        for _ in 0..8 {
            let rand = |_: ()| {
                std::collections::hash_map::RandomState::new()
                    .build_hasher()
                    .finish()
            };
            let name = format!(
                "fandhe-ai-mnist-mlp-{}-{:016x}{:016x}",
                std::process::id(),
                rand(()),
                rand(())
            );
            let path = base.join(name);
            match create_private_dir(&path) {
                Ok(()) => {
                    // 信頼検査を通るまで再帰削除ガードは作らない。拒否時は今作った空ディレクトリだけを
                    // 非再帰で消す（信頼できないパスに remove_dir_all を向けない）。
                    #[cfg(unix)]
                    if let Err(e) = check_parent_trusted(&base, &path) {
                        let _ = std::fs::remove_dir(&path);
                        return Err(e);
                    }
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| std::io::Error::other("一時ディレクトリを作成できなかった")))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SaveDirGuard {
    fn drop(&mut self) {
        // 後始末はベストエフォート（std の remove_dir_all は symlink を辿らない）。
        // 失敗しても本処理の結果は変えず、一時ディレクトリが残りうる。
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// 保存前後の `predict` 出力の比較結果。
struct RoundTripDiff {
    max_abs_diff: f32,
    bit_mismatches: usize,
    len: usize,
}

/// 2 つの出力を比較する。shape・要素数が違えば Err。最大差は NaN を飲み込まず NaN として残す。
fn compare_outputs(before: &Tensor<f32>, after: &Tensor<f32>) -> Result<RoundTripDiff, DynError> {
    if before.shape() != after.shape() {
        return Err(format!(
            "出力の shape が一致しない（保存前 {:?}, 復元後 {:?}）",
            before.shape(),
            after.shape()
        )
        .into());
    }
    let (b, a) = (before.host_slice(), after.host_slice());
    if b.len() != a.len() {
        return Err(format!("出力の要素数が一致しない（{} vs {}）", b.len(), a.len()).into());
    }
    let mut max_abs_diff = 0.0f32;
    let mut bit_mismatches = 0usize;
    for (&x, &y) in b.iter().zip(a.iter()) {
        if x.to_bits() != y.to_bits() {
            bit_mismatches += 1;
        }
        let d = (x - y).abs();
        if d.is_nan() || max_abs_diff.is_nan() {
            max_abs_diff = f32::NAN;
        } else if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    Ok(RoundTripDiff {
        max_abs_diff,
        bit_mismatches,
        len: b.len(),
    })
}

/// 往復の合否判定（fail-closed）。bit 不一致 0 かつ最大差が有限の 0 のときだけ合格。
/// `+0.0` と `-0.0` のように差が 0 でも bit が違えば不合格、NaN も不合格にする。
fn check_round_trip(d: &RoundTripDiff) -> Result<(), DynError> {
    if d.bit_mismatches == 0 && d.max_abs_diff.is_finite() && d.max_abs_diff == 0.0 {
        Ok(())
    } else {
        Err(format!(
            "save_model → load_model で predict 出力が一致しない（最大差 {:e}, bit 不一致 {}/{} 要素）",
            d.max_abs_diff, d.bit_mismatches, d.len
        )
        .into())
    }
}

/// 一時ディレクトリへ `save_model` → `load_model` し、同じ入力 `x` の `predict` 出力を比較する。
/// ガードは本関数内で drop され、成功・失敗どちらでも削除を試みる（削除失敗は無視するため残りうる）。
fn save_and_verify(model: &Sequential, x: &Tensor<f32>) -> Result<RoundTripDiff, DynError> {
    // 保存前の出力を比較基準にするため、save_model より前に計算する。
    let before = model.predict(x)?;
    let guard = SaveDirGuard::create()?;
    let dir = guard.path().join("model");
    save_model(model, &dir)?;
    let loaded = load_model(&dir)?;
    let after = loaded.predict(x)?;
    compare_outputs(&before, &after)
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
    check_result(&report)?;
    let diff = save_and_verify(&model, &test.x)?;
    println!(
        "save_model → load_model: predict 出力の最大差 = {:e}（bit 不一致 {}/{} 要素）",
        diff.max_abs_diff, diff.bit_mismatches, diff.len
    );
    check_round_trip(&diff)
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
        // 評価入力が学習入力の先頭 N_TEST 件と一致しない（別シードで生成された）ことを、
        // 実際に生成したデータで確認する。クラス原型とラベル順は共通のため入力でのみ区別できる。
        let train_x = train.x.as_slice().expect("画素スライスを取得できること");
        let test_x = test.x.as_slice().expect("画素スライスを取得できること");
        assert_ne!(&train_x[..test_x.len()], test_x);
        let mut model = build_model().unwrap();
        let report = train_and_report(&mut model, &train, &test).unwrap();
        assert_eq!(report.epoch_loss.len(), EPOCHS);
        check_result(&report).unwrap();
        // fit 後のモデルで保存・復元の往復を検証する（学習の重複を避ける）。
        #[cfg(any(
            all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ),
            target_os = "macos"
        ))]
        {
            let diff = save_and_verify(&model, &test.x).unwrap();
            assert_eq!(diff.bit_mismatches, 0);
            assert_eq!(diff.max_abs_diff, 0.0);
            check_round_trip(&diff).unwrap();
        }
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

    #[test]
    fn save_dir_guard_cleans_up() {
        let guard = SaveDirGuard::create().unwrap();
        let path = guard.path().to_path_buf();
        assert!(path.is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "グループ・その他の権限がないこと");
        }
        drop(guard);
        assert!(!path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn untrusted_parent_is_rejected_without_recursive_delete() {
        use std::os::unix::fs::PermissionsExt;
        let outer = SaveDirGuard::create().unwrap();
        let open = outer.path().join("open");
        std::fs::create_dir(&open).unwrap();
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o777)).unwrap();
        let child = open.join("child");
        create_private_dir(&child).unwrap();
        assert!(check_parent_trusted(&open, &child).is_err());
        let trusted = outer.path().join("closed");
        std::fs::create_dir(&trusted).unwrap();
        // umask に依らず他者書き込みなしにする。
        std::fs::set_permissions(&trusted, std::fs::Permissions::from_mode(0o755)).unwrap();
        let child2 = trusted.join("child");
        create_private_dir(&child2).unwrap();
        assert!(check_parent_trusted(&trusted, &child2).is_ok());
    }

    #[test]
    fn compare_outputs_detects_mismatch() {
        let t = |v: Vec<f32>, s: &[usize]| Tensor::new(v, s).unwrap();
        let base = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let same = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        check_round_trip(&compare_outputs(&base, &same).unwrap()).unwrap();
        let one = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.5], &[2, 3]);
        let d = compare_outputs(&base, &one).unwrap();
        assert!(d.max_abs_diff > 0.0);
        assert!(check_round_trip(&d).is_err());
        let nan = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN], &[2, 3]);
        assert!(check_round_trip(&compare_outputs(&base, &nan).unwrap()).is_err());
        let pz = t(vec![0.0], &[1]);
        let nz = t(vec![-0.0], &[1]);
        let d = compare_outputs(&pz, &nz).unwrap();
        assert_eq!(d.max_abs_diff, 0.0);
        assert_eq!(d.bit_mismatches, 1);
        assert!(check_round_trip(&d).is_err());
        let other = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        assert!(compare_outputs(&base, &other).is_err());
    }
}
