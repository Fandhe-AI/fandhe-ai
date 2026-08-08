use autodiff::Tape;
use guardrail_labeled_changes_baseline::compat;
use guardrail_labeled_changes_baseline::model::Mlp;
use guardrail_labeled_changes_baseline::train::{train, xor_dataset};
use tensor_core::Tensor;

fn main() {
    const SEED: u64 = 0x5EED_0001;

    let model = Mlp::new(SEED).expect("main: shape は事前に妥当");
    let (x, y) = xor_dataset(8);

    println!("=== v2 自作コア上のミニ MLP 学習ログ（TASK-4.2a baseline） ===");
    let (model, final_loss) = train(model, x, y, 500, 1e-2).expect("main: 学習ループは失敗しない");
    println!("最終 loss = {final_loss:.6}");

    let infer_tape = Tape::new_with_ops(Box::new(backend_cpu::CpuBackendOps::new()));
    let infer_x = Tensor::new(vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])
        .expect("main: shape とデータ長は一致する");
    let infer_x_var = infer_tape.var(&infer_x);
    let (pred, _, _, _) = model
        .forward(&infer_tape, &infer_x_var)
        .expect("main: forward は失敗しない");
    let pred_tensor = pred.to_tensor();

    println!("=== 推論結果（期待値: 0, 1, 1, 0） ===");
    for i in 0..4 {
        let v = pred_tensor
            .get(&[i, 0])
            .expect("main: pred の shape は [4, 1]");
        println!("入力{i}: 予測={v:.4}");
    }

    println!("=== 互換 API 層デモ（LeakyReLU 追加後） ===");
    let compat_tape = Tape::new_with_ops(Box::new(backend_cpu::CpuBackendOps::new()));
    let compat_x_tensor = compat::array(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])
        .expect("main: shape とデータ長は一致する");
    let compat_x = compat_tape.var(&compat_x_tensor);
    let compat_model = compat::Sequential::new()
        .add_linear(2, 8, 0x1)
        .expect("main: shape は事前に妥当")
        .add_leaky_relu(0.1)
        .add_linear(8, 8, 0x2)
        .expect("main: shape は事前に妥当")
        .add_relu()
        .add_linear(8, 1, 0x3)
        .expect("main: shape は事前に妥当");
    let compat_out = compat_model
        .forward(&compat_tape, &compat_x)
        .expect("main: forward は失敗しない");
    println!(
        "互換 API 経由の未学習フォワード出力 shape: {:?}",
        compat_out.to_tensor().shape()
    );
}
