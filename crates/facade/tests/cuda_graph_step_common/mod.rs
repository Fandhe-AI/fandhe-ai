//! `cuda_graph_step_bit_identity.rs`・`cuda_graph_step_two_gpu_bit_identity.rs`
//! （イシュー #1349）が共有する学習ループ・出力ヘルパー。
//!
//! **分離した理由（codex-review P2 指摘対応・PR #1390）**: 2 GPU 構成の
//! 機械比較テスト（`graph_capture_matches_eager_baseline_bit_identical_
//! across_two_gpus`）は「opt-in OFF で開始する」前提を持つが、同じ
//! バイナリ内の他テスト（`graph_capture`・
//! `graph_capture_completes_training_loop_without_error`）は opt-in を
//! ON のまま／プロセスワイドに変更する前提を持つ。`cargo test
//! graph_capture` のような部分一致フィルタで 3 つとも同一プロセス・
//! 並行スレッドで選ばれてしまうと、いずれかのテストの前提
//! （「opt-in はプロセス内最初の CUDA デバイス初期化より前に固定」・
//! 「プロセスワイドな opt-in フラグが実行中に他スレッドから変わらない」）
//! が崩れる。2 GPU テストを別ファイル（＝別テストバイナリ・別プロセス）
//! へ分離することで、フィルタの部分一致に関わらず両者が同一プロセスで
//! 選ばれることを構造的になくす（`crates/backend-cuda/tests/
//! graph_capture_real_device.rs`／`graph_capture_real_device_optin_off.rs`
//! を opt-in ON／OFF で別ファイルに分けた既存パターンと同じ方針。
//! `docs/backend-cuda-graph-step-capture-design.md` 9 節参照）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

pub const BATCH: usize = 4;
pub const D_IN: usize = 8;
pub const D_HIDDEN: usize = 16;
pub const D_OUT: usize = 4;
pub const STEPS: usize = 10;
pub const LR: f32 = 0.05;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

pub fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

pub fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

pub fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// `device_param_store_train.rs::train_with_device_param_store` の CUDA
/// 版。各 step の loss（`f32` そのまま。ビット比較は呼び出し元が
/// `to_bits()` で行う）と、最終的にホストへ同期したパラメータ列を返す。
///
/// `ordinal` を引数化している理由（codex-review P2 指摘対応。イシュー
/// #1349）: opt-in（`FANDHE_AI_CUDA_GRAPH_STEP`／
/// `set_cuda_graph_step_enabled`）はプロセス内最初の CUDA デバイス
/// 初期化より前に固定される必要があるため（両呼び出し元ファイルの
/// 冒頭コメント参照）、同一プロセス内で「opt-in OFF の基準値」と
/// 「opt-in ON の capture 経路」を機械比較するには**異なる ordinal**を
/// 使う必要がある（`cuda_graph_step_two_gpu_bit_identity.rs` 参照）。
pub fn train_on_cuda(ordinal: usize, steps: usize, lr: f32) -> (Vec<f32>, Vec<Tensor<f32>>) {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(lr);
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape =
            fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    let final_tape =
        fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
    let synced = final_tape.sync_device_param_store_to_host(&store).unwrap();
    (log, synced)
}

/// loss 列・最終パラメータを `to_bits()` の 16 進表現で標準出力へ出す
/// （プロセス間比較のための決定的なテキスト表現。浮動小数点の表示
/// 誤差を避けるため `{:?}`／`{}` ではなくビット表現を使う）。
pub fn print_bit_identity_report(label: &str, log: &[f32], params: &[Tensor<f32>]) {
    println!("=== cuda_graph_step_bit_identity: {label} ===");
    for (i, loss) in log.iter().enumerate() {
        println!("step[{i}].loss.bits = {:#010x}", loss.to_bits());
    }
    for (p, tensor) in params.iter().enumerate() {
        let contiguous = tensor.contiguous();
        let slice = contiguous.as_slice().unwrap_or(&[]);
        for (i, v) in slice.iter().enumerate() {
            println!("param[{p}][{i}].bits = {:#010x}", v.to_bits());
        }
    }
}
