//! `fit_with_metrics`／`MetricsResult::compute`（イシュー #2072・親
//! #2059）の CUDA／Metal 3 バックエンド bit 同一テスト（受入基準 F）。
//! `compat_sequential_activation_shape_backend_parity.rs` と同型。
//!
//! metrics 算術自体はホスト側整数カウント＋`f64` 導出でバックエンド
//! 非依存（`crates/facade/src/compat/metrics.rs` モジュール doc「バック
//! エンド非依存性」節）のため、本テストが検証する対象は「forward
//! （logits）が CPU と対象デバイスで一致すれば、`MetricsResult` も
//! **bit 完全一致**する」という end-to-end 契約である
//! （`assert_parity` の許容誤差付き判定ではなく `assert_eq!` を使う）。
//!
//! **事前登録判定規則**（不一致が出た場合の切り分け手順）: logits の
//! 近接タイ（parity レベルの forward 差で `Var::argmax` の結果が反転
//! する）が原因である可能性が高い。metrics 算術自体はバックエンド
//! 非依存のため、不一致の原因が argmax 反転であることを確認できた
//! 場合でも、判定規則・tolerance・baseline は変更しない
//! （`docs/perf/logs/compat-sequential-metrics-2072/README.md` 参照）。
//!
//! 実機実測は本エージェントの実行環境に CUDA／Metal 実機への到達
//! 手段がないため未実施のまま Mac／GB10 セッションへ申し送る。

use fandhe_ai::compat::{Metrics, MetricsResult, Sequential};
use fandhe_ai::{Device, Tensor};

const SEED1: u64 = 0x5555_6666;
const SEED2: u64 = 0x7777_8888;

const ALL_SCALAR_METRICS: [Metrics; 4] = [
    Metrics::Accuracy,
    Metrics::Precision,
    Metrics::Recall,
    Metrics::F1,
];

fn tensor_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn tensor_i32(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `Linear → ReLU → Linear`（3 クラス分類）の forward を CPU と対象
/// デバイスで求め、両者の logits から独立に [`MetricsResult::compute`]
/// した結果が bit 完全一致することを確認する。
fn run_metrics_parity(device: Device) {
    const N: usize = 12;
    const D_IN: usize = 4;
    const D_OUT: usize = 3;

    let model = Sequential::new()
        .add_linear(D_IN, 6, SEED1)
        .unwrap()
        .add_relu()
        .add_linear(6, D_OUT, SEED2)
        .unwrap();
    let x = tensor_f32(
        (0..N * D_IN).map(|i| i as f32 * 0.05 - 0.3).collect(),
        &[N, D_IN],
    );
    let target = tensor_i32((0..N).map(|i| (i % D_OUT) as i32).collect(), &[N]);

    let cpu_logits = model.predict(&x).unwrap();

    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let device_logits = model.forward(&tape, &xv).unwrap().to_tensor();

    let cpu_result = MetricsResult::compute(&ALL_SCALAR_METRICS, &cpu_logits, &target).unwrap();
    let device_result =
        MetricsResult::compute(&ALL_SCALAR_METRICS, &device_logits, &target).unwrap();

    assert_eq!(
        cpu_result, device_result,
        "device forward の logits から求めた MetricsResult が CPU と bit 一致しない \
         （事前登録判定規則: logits の近接タイによる argmax 反転の可能性）"
    );
}

// --- CUDA（本エージェント実行環境に実機なし。GB10 セッションへ申し送り） ---

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #2072 の申し送り先（GB10 セッション）\
            へ引き継ぐ"]
fn cuda_metrics_result_matches_cpu() {
    run_metrics_parity(Device::Cuda(0));
}

// --- Metal（本エージェント実行環境に実機なし。Mac セッションへ申し送り） ---

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実行は #2072 の申し送り先（Mac セッション）へ \
            引き継ぐ"]
fn metal_metrics_result_matches_cpu() {
    run_metrics_parity(Device::Metal);
}
