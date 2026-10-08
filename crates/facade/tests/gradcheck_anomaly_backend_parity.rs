//! `fandhe_ai_autodiff::gradcheck::gradcheck`・`fandhe_ai_autodiff::anomaly::backward_detect_anomaly`
//! （イシュー #2671。内部実装との突合のため `fandhe_ai_autodiff` を直接 use する。facade の `Tape::gradcheck` は #2847 で公開）の
//! バックエンド間 parity テスト（`jacobian_hessian_backend_parity.rs` と同型）。
//!
//! 属性なし: 実 `CpuBackendOps` を結線した tape と `Tape::new()`（`NaiveOps`）を突き合わせる。
//! gradcheck・anomaly は新しい `BackendOps` メソッドを持たず既存 Op の合成のみで到達するため、
//! 確認するのは「合否・検出ノードが既存カーネルの新しい呼び出し形でバックエンド間一致すること」と
//! 「解析勾配が REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で一致すること」。
//! 手計算の期待値も 1 件固定する。`GradcheckOptions` は #223 承認済みの組のみを使う（tolerance 変更なし）。
//!
//! `#[ignore]`（CUDA／Metal〈`cfg(target_os = "macos")` 限定〉の `BackendOps` を結線した tape と CPU
//! tape の比較）: 実機への到達手段が本エージェント実行環境にないため未実施のまま GB10／Mac
//! セッションへ申し送る（`docs/perf/logs/gradcheck-anomaly-2671/README.md`）。

use fandhe_ai_autodiff::anomaly::backward_detect_anomaly;
use fandhe_ai_autodiff::gradcheck::{GradcheckOptions, gradcheck};
use fandhe_ai_autodiff::jacobian_ops::jacobian;
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn opts() -> GradcheckOptions {
    GradcheckOptions::new(1e-3, 1e-3, 1e-2, 1e-4).expect("承認済みの閾値の組は有効")
}

struct Outs {
    passed: bool,
    checked: usize,
    jac_shape: Vec<usize>,
    jac: Vec<f32>,
    anomaly_msg: String,
}

/// `make` が返すバックエンドのテープで gradcheck・解析ヤコビアン・anomaly 検出を実行する。
fn compute(make: &dyn Fn() -> Tape) -> Outs {
    let w1 = t((0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(), &[3, 4]);
    let w2 = t((0..8).map(|i| 0.2 - 0.05 * (i as f32)).collect(), &[4, 2]);
    let x = t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]);

    let report = gradcheck(
        || Ok(make()),
        |tape, xs| {
            let (w1, w2) = (tape.var_no_grad(&w1), tape.var_no_grad(&w2));
            xs[0].matmul(&w1)?.tanh().matmul(&w2)
        },
        std::slice::from_ref(&x),
        &opts(),
    )
    .expect("gradcheck は評価に成功する");

    let tape = make();
    let xv = tape.var(&x);
    let y = xv
        .matmul(&tape.var_no_grad(&w1))
        .unwrap()
        .tanh()
        .matmul(&tape.var_no_grad(&w2))
        .unwrap();
    let jac = jacobian(&tape, &y, &xv).unwrap();

    let bad_tape = make();
    let bx = bad_tape.var(&t(vec![-1.0, 2.0], &[2]));
    let loss = bx.log().unwrap().sum(None).unwrap();
    let anomaly_msg = match backward_detect_anomaly(&bad_tape, &loss) {
        Err(AutodiffError::Backward(m)) => m,
        other => panic!("NaN を検出できていない: {:?}", other.map(|_| ())),
    };

    Outs {
        passed: report.passed(),
        checked: report.checked_elements(),
        jac_shape: jac.shape().to_vec(),
        jac: jac.host_slice().into_owned(),
        anomaly_msg,
    }
}

fn cpu_tape() -> Tape {
    Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
}

fn assert_outs_match(label: &str, a: &Outs, b: &Outs) {
    assert!(a.passed && b.passed, "{label}: gradcheck 不合格");
    assert_eq!(a.checked, b.checked, "{label}: 検査要素数");
    assert_eq!(a.jac_shape, b.jac_shape, "{label}: jacobian shape");
    fandhe_ai_backend_cpu::parity::assert_parity(&format!("{label}: jacobian"), &a.jac, &b.jac);
    assert_eq!(a.anomaly_msg, b.anomaly_msg, "{label}: anomaly の検出結果");
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = compute(&cpu_tape);
    let naive = compute(&Tape::new);
    assert_eq!(cpu.jac_shape, vec![2, 2, 2, 3]);
    assert_eq!(cpu.checked, 2 * 2 * 6);
    assert_outs_match("cpu vs naive", &cpu, &naive);
}

/// 手計算の期待値を固定する（両経路が同じ誤りで一致していないことの確認）。
/// `x = [-1, 2]`・`loss = Σ log(x)` は forward の node 1（`log`）で NaN になる（葉 node 0 は有限）。
/// `y = x ⊙ x`・`x = [1, 2]` は `gradcheck` に合格し、検査要素数は `2 × 2 = 4`。
#[test]
fn cpu_matches_hand_computed_values() {
    let tape = cpu_tape();
    let x = tape.var(&t(vec![-1.0, 2.0], &[2]));
    let loss = x.log().unwrap().sum(None).unwrap();
    let Err(AutodiffError::Backward(msg)) = backward_detect_anomaly(&tape, &loss) else {
        panic!("NaN を検出できていない");
    };
    assert!(msg.contains("forward") && msg.contains("node 1"), "{msg}");

    let report = gradcheck(
        || Ok(cpu_tape()),
        |_t, xs| xs[0].mul(&xs[0]),
        &[t(vec![1.0, 2.0], &[2])],
        &opts(),
    )
    .unwrap();
    assert!(report.passed());
    assert_eq!(report.checked_elements(), 4);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/gradcheck-anomaly-2671/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(make_ops: impl Fn() -> Box<dyn BackendOps + Send>, label: &str) {
    let cpu = compute(&cpu_tape);
    let dev = compute(&|| Tape::new_with_ops(make_ops()));
    assert_outs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// gradcheck・anomaly の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/gradcheck-anomaly-2671/README.md 参照"]
fn metal_gradcheck_anomaly_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
        "metal",
    );
}

/// gradcheck・anomaly の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/gradcheck-anomaly-2671/README.md 参照"]
fn cuda_gradcheck_anomaly_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)),
        "cuda",
    );
}
