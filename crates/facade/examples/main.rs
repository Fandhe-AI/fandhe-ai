//! ResNet・Transformer（イシュー #2202・親 #2190）の学習 script（AC5）。
//!
//! `fandhe_ai`（facade）だけを使い、合成 CIFAR-10 相当データで
//! ResNet・Transformer を 10 epoch 学習し、epoch ごとの train loss・
//! held-out 精度をログ出力する。`cargo run -p fandhe-ai --example main`
//! で実行できる（`--release` を推奨。debug ビルドでは学習に数分かかる）。
//!
//! `crates/facade/examples/models/` 配下のファイルを個別に `#[path]`
//! 取り込みする（`reference_models.rs` の `mod models;` 経由の集約は
//! 使わない。`resnet.rs`／`transformer.rs`・依存する
//! `reference_module.rs`・`synthetic_cifar.rs` の 4 ファイルだけを
//! 直接取り込む契約。`docs/reference-models-decision.md` #2202 節
//! 「取り込み方」参照）。

#[path = "models/reference_module.rs"]
mod reference_module;
#[path = "models/resnet.rs"]
mod resnet;
#[path = "models/synthetic_cifar.rs"]
mod synthetic_cifar;
#[path = "models/transformer.rs"]
mod transformer;

use fandhe_ai::Tensor;
use fandhe_ai::optim::{Adam, AdamConfig};
use reference_module::{
    ReferenceModule, accuracy, fit_epochs, heldout_loss, predict_in_eval, sub_tensor_f32,
};
use resnet::ResNet;
use synthetic_cifar::{IMG_C, IMG_H, IMG_W, NUM_CLASSES, synthetic_cifar10, to_row_tokens};
use transformer::{TransformerClassifier, TransformerClassifierConfig};

const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 16;
const N_TRAIN: usize = 64;
const N_TEST: usize = 32;
const LR: f32 = 5e-3;

/// 依存追加なしの局所 PRNG（SplitMix64。テスト側の
/// `bench_harness::rng::Xorshift64Star` の代わりに使う。`main.rs` は
/// examples 同梱であり `bench_harness`〈非公開クレート〉を import
/// できないため）。
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

/// 平坦画素データ（`[N, 3, 32, 32]`）を素直な `Tensor<f32>` に包む。
fn image_tensor(flat: Vec<f32>, n: usize) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    Ok(Tensor::new(flat, &[n, IMG_C, IMG_H, IMG_W])?)
}

/// 行トークン化済みデータ（`[N, 32, 96]`）を `Tensor<f32>` に包む。
fn token_tensor(flat: Vec<f32>, n: usize) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    Ok(Tensor::new(flat, &[n, IMG_H, IMG_C * IMG_W])?)
}

fn labels_tensor(labels: Vec<i32>, n: usize) -> Result<Tensor<i32>, Box<dyn std::error::Error>> {
    Ok(Tensor::new(labels, &[n])?)
}

/// AC4 の判定式（`docs/reference-models-decision.md` #2202 節「判定式」）:
/// 全 epoch の loss が有限・最終 epoch の loss が初回より小さい・
/// held-out 精度が 0.50 以上。
fn check_ac4(name: &str, history: &[f32], test_acc: f32) -> Result<(), Box<dyn std::error::Error>> {
    // `history[0]` を後段で直接参照するため、空の history を先に拒否する
    // （`fit_epochs` は `epochs == 0` を拒否するため通常は空にならないが、
    // `check_ac4` 単体を呼ぶ側の契約としても空データを明示的にエラー
    // にする。`unwrap_or(INFINITY)` は `last` の半分のガードにしか
    // ならず `history[0]` の panic は防げない。イシュー #2202 PR #2325
    // レビュー方針の横展開）。
    if history.is_empty() {
        return Err(format!("{name}: 学習履歴（history）が空である").into());
    }
    if !history.iter().all(|v| v.is_finite()) {
        return Err(format!("{name}: 学習中に非有限の loss が発生した: {history:?}").into());
    }
    let last = history.last().copied().unwrap_or(f32::INFINITY);
    if !matches!(
        last.partial_cmp(&history[0]),
        Some(std::cmp::Ordering::Less)
    ) {
        return Err(format!(
            "{name}: 最終 epoch の loss が初回 epoch の loss を下回らなかった: {history:?}"
        )
        .into());
    }
    // `test_acc < 0.5` は NaN に対し false と評価され AC4 判定を誤って
    // 通過させるため、下限比較の前に有限性を検査する
    // （イシュー #2202 PR #2325 レビュー指摘）。
    if !test_acc.is_finite() {
        return Err(format!("{name}: held-out 精度が非有限（実測: {test_acc}）").into());
    }
    if test_acc < 0.5 {
        return Err(format!(
            "{name}: held-out 精度が判定式の下限 0.50 を下回った（実測: {test_acc:.4}）"
        )
        .into());
    }
    Ok(())
}

fn run_resnet() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n== ResNet(depth=8, width=8) を合成 CIFAR-10 相当データで学習 ==");
    let mut rng = SplitMix64(0xC0FF_EE00_ABCD_1234);
    let mut train_src = || rng.next_u64();
    let (train_flat, train_labels) = synthetic_cifar10(N_TRAIN, &mut train_src)?;
    let mut rng_test = SplitMix64(0xFEED_BEEF_0011_2233);
    let mut test_src = || rng_test.next_u64();
    let (test_flat, test_labels) = synthetic_cifar10(N_TEST, &mut test_src)?;

    let x_train = image_tensor(train_flat, N_TRAIN)?;
    let y_train = labels_tensor(train_labels, N_TRAIN)?;
    let x_test = image_tensor(test_flat, N_TEST)?;
    let y_test = labels_tensor(test_labels, N_TEST)?;

    let mut model = ResNet::new(8, 8, NUM_CLASSES, 0x5E5E_5E5E)?;
    println!(
        "構成: depth={} width={} num_classes={} num_blocks={} projection_shortcuts={}",
        model.depth(),
        model.width(),
        model.num_classes(),
        model.num_blocks(),
        model
            .blocks()
            .iter()
            .filter(|b| b.has_projection_shortcut())
            .count()
    );
    // 学習前の predict 出力 shape を確認しておく（推論経路の疎通確認）。
    // `ResNet::predict`（モード切り替えなし）を構築直後（training の
    // 既定値 true）に直接呼ぶと、BatchNorm が train モードの forward
    // を実行し running stats を汚染してしまう。`predict_in_eval` で
    // 一時的に eval へ切り替え、呼び出し前のモード（ここでは既定の
    // train）へ復元してから `fit_epochs` を開始する（Codex レビュー
    // 指摘・イシュー #2202 PR #2325）。
    let sample_batch = sub_tensor_f32(&x_train, 0, BATCH_SIZE)?;
    let sample_pred = predict_in_eval(&mut model, &sample_batch)?;
    println!(
        "初期状態の ResNet::predict 出力 shape: {:?}",
        sample_pred.shape()
    );
    for (name, tensor) in ReferenceModule::named_parameters(&model)
        .into_iter()
        .take(3)
    {
        println!("  named_parameters: {name} shape={:?}", tensor.shape());
    }

    let mut opt = Adam::new(AdamConfig {
        lr: LR,
        ..AdamConfig::default()
    })?;

    let history = fit_epochs(&mut model, &x_train, &y_train, &mut opt, EPOCHS, BATCH_SIZE)?;
    for (epoch, loss) in history.iter().enumerate() {
        println!(
            "ResNet epoch {}/{EPOCHS}: train loss = {loss:.4}",
            epoch + 1
        );
    }
    let test_acc = accuracy(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES)?;
    println!("ResNet held-out accuracy = {test_acc:.4}");
    let test_loss = heldout_loss(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES)?;
    println!("ResNet held-out loss = {test_loss:.4}");
    // 学習後、`ResNet::predict`（モード切り替えなし契約。モジュール doc
    // 参照）を eval モードへ明示的に切り替えてから呼ぶ（`*_predict_
    // shape_and_eval_determinism` テストと同じ呼び出し規律。冒頭の
    // shape 確認〈`predict_in_eval`〉とは異なり、ここでは呼び出し後の
    // モードを気にする後続処理が無いため、単純に eval へ切り替える
    // だけでよい）。
    ReferenceModule::set_training(&mut model, false);
    let final_pred = model.predict(&sample_batch)?;
    println!(
        "学習後（eval）の ResNet::predict 出力 shape: {:?}",
        final_pred.shape()
    );
    check_ac4("ResNet", &history, test_acc)
}

fn run_transformer() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n== Transformer(embed_dim=32, heads=4, layers=2) を合成 CIFAR-10 相当データで学習 =="
    );
    let mut rng = SplitMix64(0xABCD_EF01_2345_6789);
    let mut train_src = || rng.next_u64();
    let (train_flat, train_labels) = synthetic_cifar10(N_TRAIN, &mut train_src)?;
    let mut rng_test = SplitMix64(0x1357_9BDF_2468_ACE0);
    let mut test_src = || rng_test.next_u64();
    let (test_flat, test_labels) = synthetic_cifar10(N_TEST, &mut test_src)?;

    let x_train = token_tensor(to_row_tokens(&train_flat, N_TRAIN)?, N_TRAIN)?;
    let y_train = labels_tensor(train_labels, N_TRAIN)?;
    let x_test = token_tensor(to_row_tokens(&test_flat, N_TEST)?, N_TEST)?;
    let y_test = labels_tensor(test_labels, N_TEST)?;

    let config = TransformerClassifierConfig::cifar10(32, 4, 2, NUM_CLASSES)?;
    let mut model = TransformerClassifier::new(config, 0x7777_7777)?;
    let seen_config = model.config();
    println!(
        "構成: seq_len={} in_features={} embed_dim={} num_heads={} num_layers={} num_classes={}",
        seen_config.seq_len,
        seen_config.in_features,
        seen_config.embed_dim,
        seen_config.num_heads,
        seen_config.num_layers,
        seen_config.num_classes
    );
    // 学習前の predict 出力 shape を確認しておく（推論経路の疎通確認）。
    // `ResNet` と同じ理由で `predict_in_eval` を使う（`Transformer`
    // 自体は mode 依存層を持たないが、`ReferenceModule` の呼び出し
    // 規律をモデル間で統一する。Codex レビュー指摘・イシュー #2202
    // PR #2325）。
    let sample_batch = sub_tensor_f32(&x_train, 0, BATCH_SIZE)?;
    let sample_pred = predict_in_eval(&mut model, &sample_batch)?;
    println!(
        "初期状態の Transformer::predict 出力 shape: {:?}",
        sample_pred.shape()
    );

    let mut opt = Adam::new(AdamConfig {
        lr: LR,
        ..AdamConfig::default()
    })?;

    let history = fit_epochs(&mut model, &x_train, &y_train, &mut opt, EPOCHS, BATCH_SIZE)?;
    for (epoch, loss) in history.iter().enumerate() {
        println!(
            "Transformer epoch {}/{EPOCHS}: train loss = {loss:.4}",
            epoch + 1
        );
    }
    let test_acc = accuracy(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES)?;
    println!("Transformer held-out accuracy = {test_acc:.4}");
    let test_loss = heldout_loss(&mut model, &x_test, &y_test, BATCH_SIZE, NUM_CLASSES)?;
    println!("Transformer held-out loss = {test_loss:.4}");
    // `run_resnet` と同じ理由で、学習後の `Transformer::predict` は
    // eval モードへ明示的に切り替えてから呼ぶ。
    ReferenceModule::set_training(&mut model, false);
    let final_pred = model.predict(&sample_batch)?;
    println!(
        "学習後（eval）の Transformer::predict 出力 shape: {:?}",
        final_pred.shape()
    );
    check_ac4("Transformer", &history, test_acc)
}

/// CLI が受理する実行モード（`resnet|transformer|all` の完全一致
/// allowlist。[`parse_mode`] の戻り値）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Resnet,
    Transformer,
    All,
}

/// CLI 引数（`std::env::args().skip(1)` 相当の位置引数列）から
/// [`Mode`] を決定する（`.claude/rules/security.md` A03「シェル呼び出し
/// でユーザー入力を直接展開しない」——本 example はシェル呼び出しも
/// ファイル I/O も行わないが、allowlist 外の入力・余分な引数は usage
/// を返す fail-closed 方針を踏襲する）。第 2 引数以降が存在する場合
/// （`main resnet unexpected` 等）も拒否する（`resnet|transformer|all`
/// のみを受け付ける契約と、実装が第 1 引数しか見ていなかった不一致を
/// 解消する。Codex レビュー指摘・イシュー #2202 PR #2325）。
/// 引数無しは `all` 相当（既定モード）として受理する。
fn parse_mode(args: &[String]) -> Result<Mode, String> {
    if args.is_empty() {
        return Ok(Mode::All);
    }
    if args.len() > 1 {
        return Err(format!(
            "usage: main [resnet|transformer|all]（余分な引数: {:?}）",
            &args[1..]
        ));
    }
    match args[0].as_str() {
        "resnet" => Ok(Mode::Resnet),
        "transformer" => Ok(Mode::Transformer),
        "all" => Ok(Mode::All),
        other => Err(format!(
            "usage: main [resnet|transformer|all]（不明な引数: '{other}'）"
        )),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match parse_mode(&args) {
        Ok(mode) => mode,
        Err(usage) => {
            eprintln!("{usage}");
            std::process::exit(1);
        }
    };
    match mode {
        Mode::Resnet => run_resnet(),
        Mode::Transformer => run_transformer(),
        Mode::All => {
            run_resnet()?;
            run_transformer()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, parse_mode};

    #[test]
    fn parse_mode_accepts_allowlist() {
        assert_eq!(parse_mode(&[]).unwrap(), Mode::All);
        assert_eq!(parse_mode(&["resnet".to_string()]).unwrap(), Mode::Resnet);
        assert_eq!(
            parse_mode(&["transformer".to_string()]).unwrap(),
            Mode::Transformer
        );
        assert_eq!(parse_mode(&["all".to_string()]).unwrap(), Mode::All);
    }

    #[test]
    fn parse_mode_rejects_unknown_first_argument() {
        assert!(parse_mode(&["bogus".to_string()]).is_err());
    }

    #[test]
    fn parse_mode_rejects_extra_arguments() {
        // `main resnet unexpected` のように allowlist に一致する第 1
        // 引数があっても、第 2 引数以降が存在すれば拒否する
        // （Codex レビュー指摘・イシュー #2202 PR #2325）。
        assert!(parse_mode(&["resnet".to_string(), "unexpected".to_string()]).is_err());
    }

    #[test]
    fn check_ac4_rejects_empty_history() {
        assert!(super::check_ac4("test", &[], 0.9).is_err());
    }
}
