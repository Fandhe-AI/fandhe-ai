//! 参照モデル定義（`Mlp`・`LeNet`）の runnable example（イシュー #2201・
//! 親 #2190）。
//!
//! `fandhe_ai`（facade。唯一のサポートされる公開 API 面）だけを使い、
//! PyTorch の定番モデル（MLP・LeNet）を `compat::Sequential` の上で
//! 組めることを実行可能なコードで示す。`cargo run -p fandhe-ai --example
//! reference_models` で実行できる。
//!
//! rustdoc は `examples/` 配下を doctest しないため、AC3（「各型の生成と
//! forward を doctest で実行できる」）は本 example と統合テスト
//! （`crates/facade/tests/example_mlp_mnist.rs`・
//! `crates/facade/tests/example_lenet_mnist.rs`）で代替する
//! （`docs/reference-models-decision.md` 参照）。AC5（PyTorch との層ごとの
//! 重み対応を目視で確認できる）は本 example が標準出力へ表示する対応表で
//! 満たす。

mod models;

use fandhe_ai::compat::Sequential;
use models::lenet::LeNet;
use models::mlp::Mlp;

fn print_param_map(
    title: &str,
    rows: impl Iterator<Item = (String, String, Vec<usize>, Vec<usize>, bool)>,
) {
    println!("\n== {title} ==");
    println!(
        "{:<12} {:<16} {:<18} {:<18} transpose",
        "fandhe", "pytorch", "fandhe_shape", "pytorch_shape"
    );
    for (fandhe_key, pytorch_key, fandhe_shape, pytorch_shape, transpose) in rows {
        println!(
            "{fandhe_key:<12} {pytorch_key:<16} {fandhe_shape:<18?} {pytorch_shape:<18?} {transpose}"
        );
    }
}

fn run_mlp() -> Result<(), Box<dyn std::error::Error>> {
    // AC1 の 4 引数コンストラクタ（既定シード）。
    let mlp = Mlp::new(784, &[256, 128], 10, 0.2)?;
    println!(
        "Mlp: input_dim=784 hidden_dims=[256, 128] output_dim=10 dropout={}",
        mlp.dropout()
    );

    // `Mlp::with_seed` は明示シード版（再現したい場合に使う）。
    let mlp_seeded = Mlp::with_seed(784, &[256, 128], 10, 0.2, 12345)?;

    fandhe_ai::manual_seed(0);
    let x = fandhe_ai::rand(&[4, 784])?;
    let mut eval_mlp = mlp_seeded;
    eval_mlp.sequential_mut().eval();
    let y = eval_mlp.predict(&x)?;
    println!("Mlp::predict 出力 shape: {:?}", y.shape());

    // `Mlp::forward`（外部 `Tape` 上の経路）が `predict` と同じ出力を
    // 返すことも示す（eval モードでは bit 完全一致契約。統合テスト
    // `example_mlp_mnist.rs` で厳密に検証する）。
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = eval_mlp.forward(&tape, &xv)?;
    println!(
        "Mlp::forward 出力 shape: {:?}",
        via_forward.to_tensor().shape()
    );

    let seq: &Sequential = eval_mlp.sequential();
    println!(
        "Mlp 内部 Sequential::training() (eval 後): {}",
        seq.training()
    );

    print_param_map(
        "Mlp <-> PyTorch nn.Sequential(Linear, ReLU, Dropout, ...)",
        eval_mlp.pytorch_param_map()?.into_iter().map(|m| {
            (
                m.fandhe_key,
                m.pytorch_key,
                m.fandhe_shape,
                m.pytorch_shape,
                m.transpose,
            )
        }),
    );
    Ok(())
}

fn run_lenet() -> Result<(), Box<dyn std::error::Error>> {
    let lenet = LeNet::new(10, 67890)?;
    println!("\nLeNet: num_classes={}", lenet.num_classes());

    fandhe_ai::manual_seed(0);
    let x = fandhe_ai::rand(&[4, 1, 28, 28])?;
    let mut eval_lenet = lenet;
    eval_lenet.sequential_mut().eval();
    let y = eval_lenet.predict(&x)?;
    println!("LeNet::predict 出力 shape: {:?}", y.shape());

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = eval_lenet.forward(&tape, &xv)?;
    println!(
        "LeNet::forward 出力 shape: {:?}",
        via_forward.to_tensor().shape()
    );

    print_param_map(
        "LeNet <-> PyTorch Conv2d(1,6,5) -> Conv2d(6,16,5) -> Linear(256,120) -> Linear(120,10)",
        eval_lenet.pytorch_param_map()?.into_iter().map(|m| {
            (
                m.fandhe_key,
                m.pytorch_key,
                m.fandhe_shape,
                m.pytorch_shape,
                m.transpose,
            )
        }),
    );

    // `Sequential` を経由した学習系 API がそのまま呼べることも示す
    // （`sequential()`/`sequential_mut()` の意図。`compile` 前の
    // `Sequential::training()` は既定 `true`）。
    let seq: &Sequential = eval_lenet.sequential();
    println!(
        "LeNet 内部 Sequential::training() (eval 後): {}",
        seq.training()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_mlp()?;
    run_lenet()?;
    Ok(())
}
