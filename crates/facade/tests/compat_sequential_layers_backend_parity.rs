//! `compat::Sequential::add_layer_norm`／`add_rms_norm`／
//! `add_batch_norm1d`／`add_batch_norm2d`／`add_embedding`／
//! `add_multihead_attention`（イシュー #1760・親 #1618）・
//! `add_transformer_encoder`（イシュー #2068・親 #2059）の CUDA／Metal
//! parity テスト（`nn_conv_backend_parity.rs` と同型）。
//!
//! CPU 側の正しさ検証（`predict`／`forward` bit 完全一致・手動合成との
//! 比較・学習ループ・`apply_parameters`）は `compat_sequential_layers.rs`
//! が Linux 実行可能な形で既に担う。本ファイルは新規カーネルを一切
//! 追加していない構成（LayerNorm／RmsNorm／BatchNorm／Embedding／
//! MultiheadAttention の各バックエンドカーネルは既存 issue で実装
//! 済み。`TransformerEncoderLayer` はそれらの合成のみで新規 `Op` を
//! 追加しない。`docs/compat-api-scope.md` §1.2 該当行参照）を前提に、
//! `compat::Sequential` 経由で組んだモデルの forward が CPU と
//! `assert_parity`（REQ-2 統一複合判定）で一致することのみを確認する。
//!
//! 実機実測は本エージェントの実行環境に CUDA／Metal 実機への到達
//! 手段がないため未実施のまま Mac／GB10 セッションへ申し送る
//! （`docs/compat-api-scope.md` §1.2「#1760」「#2068」追記・PR 本文に
//! 明記）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

const SEED1: u64 = 0x5555_6666;
const SEED2: u64 = 0x7777_8888;

/// LayerNorm→RmsNorm→BatchNorm1d→Embedding→MultiheadAttention の
/// 全 5 種を 1 モデルに混在させた構成の forward を CPU と対象デバイス
/// で比較する共通本体。埋め込み次元 4・head 数 2 で揃え、
/// `[B, L, E] = [2, 3, 4]` の MHA 入力に対して、embedding 側は独立に
/// `[N=6]` の id 列を LayerNorm／RmsNorm／BatchNorm1d へ通した
/// `[6, 4]` 出力を別モデルで検証する（rank 契約がモデルごとに異なる
/// ため 2 モデルに分ける。`compat_sequential_layers.rs::mixed_model`
/// と同型の組み合わせ）。
fn run_norm_embedding_parity(device: Device) {
    let model = Sequential::new()
        .add_embedding(6, 4, None, SEED1)
        .unwrap()
        .add_layer_norm(4, 1e-5)
        .unwrap()
        .add_rms_norm(4, 1e-6)
        .unwrap()
        .add_batch_norm1d(4, 1e-5, 0.1)
        .unwrap();
    let ids = tensor(vec![0.0, 2.0, 5.0, 1.0, 3.0, 4.0], &[6]);

    let cpu_out = model.predict(&ids).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let idv = tape.var(&ids);
    let device_out = model.forward(&tape, &idv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(Embedding→LayerNorm→RmsNorm→BatchNorm1d) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

fn run_batch_norm2d_parity(device: Device) {
    let model = Sequential::new().add_batch_norm2d(3, 1e-5, 0.1).unwrap();
    let x = tensor(
        (0..2 * 3 * 4 * 4)
            .map(|i| (i as f32) * 0.01 - 0.2)
            .collect(),
        &[2, 3, 4, 4],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(add_batch_norm2d) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

fn run_multihead_attention_parity(device: Device) {
    let model = Sequential::new()
        .add_multihead_attention(4, 2, SEED2)
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(add_multihead_attention) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

/// `add_transformer_encoder`（イシュー #2068・親 #2059）の parity。
/// `run_multihead_attention_parity` と同じ `[B, L, E] = [2, 3, 4]`
/// 入力形状を使う（`self_attn` の rank 契約を共有するため）。新規
/// カーネルは追加していない構成（`TransformerEncoderLayer::forward`
/// は既存の `MultiheadAttention`／`Linear`／`LayerNorm`／`relu` の
/// 合成のみ）を前提に、CPU と対象デバイスの forward 一致のみ確認する。
fn run_transformer_encoder_parity(device: Device) {
    let model = Sequential::new()
        .add_transformer_encoder(4, 2, 8, SEED2)
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    );

    let cpu_out = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_out = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_parity(
        "compat::Sequential(add_transformer_encoder) device vs CPU",
        &dense(&device_out),
        &dense(&cpu_out),
    );
}

// --- CUDA（本エージェント実行環境に実機なし。GB10 セッションへ申し送り） ---

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1760 の申し送り先（GB10 セッション）\
            へ引き継ぐ"]
fn cuda_norm_embedding_matches_cpu() {
    run_norm_embedding_parity(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_batch_norm2d_matches_cpu() {
    run_batch_norm2d_parity(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_multihead_attention_matches_cpu() {
    run_multihead_attention_parity(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2068 の申し送り先（GB10 セッション）\
            へ引き継ぐ"]
fn cuda_transformer_encoder_matches_cpu() {
    run_transformer_encoder_parity(Device::Cuda(0));
}

// --- Metal（本エージェント実行環境に実機なし。Mac セッションへ申し送り） ---

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #1760 の申し送り先（Mac セッション）へ \
            引き継ぐ"]
fn metal_norm_embedding_matches_cpu() {
    run_norm_embedding_parity(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須"]
fn metal_batch_norm2d_matches_cpu() {
    run_batch_norm2d_parity(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機必須"]
fn metal_multihead_attention_matches_cpu() {
    run_multihead_attention_parity(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2068 の申し送り先（Mac セッション）へ \
            引き継ぐ"]
fn metal_transformer_encoder_matches_cpu() {
    run_transformer_encoder_parity(Device::Metal);
}
