//! イシュー #3222: `CudaBackendOps::mse_loss_device`／`mse_loss_backward_device`
//! （および `_tracked`）の実機 parity（`#[ignore]` 分離。CUDA 実機 DGX Spark
//! GB10 等）。
//!
//! 判定は 2 本立て:
//! - ホスト入出力版（`CudaBackendOps::mse_loss`／`mse_loss_backward`）との
//!   **bit 一致**（NaN はクラス一致）。両者は同じ起動ヘルパ・同じカーネルを通る
//!   ため構造的に一致するはずで、tolerance は使わない。
//! - 素朴な CPU 参照との REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::
//!   assert_parity`。tolerance は変更しない）。有限入力のみ。
//!
//! forward の `partial` 永続ワークスペースの再利用（download を挟まない連続
//! 呼び出し・サイズ違い）とチェーン（forward → backward を同期なしで積み、
//! 最後に download）も確かめる。正式な実機記録（RULE.txt・5 run・`.rev-stamp`）
//! は #3224。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test mse_device_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, DeviceBuffer, DispatchFailureCell, MseReduction, Tensor};

/// 特殊値入りの入力パターン（`kind` 0 = 有限のみ、1 = inf 混入、2 = NaN 混入、
/// 3 = -0.0 混入）。
fn pattern(n: usize, salt: usize, kind: u8) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let base = (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.37;
            if i % 5 != 2 {
                return base;
            }
            match kind {
                1 => {
                    if (i / 5) % 2 == 0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                2 => f32::NAN,
                3 => -0.0,
                _ => base,
            }
        })
        .collect()
}

fn assert_bits_or_nan_class(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: len");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{what} idx {i}: expected NaN, got {a}");
        } else {
            assert_eq!(a.to_bits(), e.to_bits(), "{what} idx {i}: {a} vs {e}");
        }
    }
}

fn ops() -> CudaBackendOps {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    CudaBackendOps::new(device.ordinal())
}

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("tensor")
}

fn upload(ops: &CudaBackendOps, data: &[f32], shape: &[usize]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&tensor(data, shape))
        .expect("upload")
}

fn download(ops: &CudaBackendOps, buf: &DeviceBuffer<f32>) -> Vec<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .download(buf)
        .expect("download")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn mean_scale(n: usize, reduction: MseReduction) -> f32 {
    match reduction {
        MseReduction::Sum => 2.0,
        _ => 2.0 * (1.0 / n as f32),
    }
}

/// 形状: 1・ブロック境界・端数・MLP 形状・`num_blocks` が上限 1024 に張り付く大きさ。
const SHAPES: [&[usize]; 8] = [
    &[1],
    &[255],
    &[256],
    &[257],
    &[64, 10],
    &[4099],
    &[1024 * 256 + 1],
    &[2 * 1024 * 256 + 7],
];

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn forward_device_matches_host_bits_and_cpu_reference() {
    let ops = ops();
    for shape in SHAPES {
        let n: usize = shape.iter().product();
        for kind in 0..4u8 {
            let p = pattern(n, 1, kind);
            let t = pattern(n, 2, 0);
            let (pb, tb) = (upload(&ops, &p, shape), upload(&ops, &t, shape));
            for reduction in [MseReduction::Mean, MseReduction::Sum] {
                let what = format!("forward {shape:?} kind={kind} {reduction:?}");
                let want = ops
                    .mse_loss(&tensor(&p, shape), &tensor(&t, shape), reduction)
                    .expect("host mse_loss");
                let want = want.as_slice().expect("contiguous").to_vec();
                let out = ops.mse_loss_device(&pb, &tb, reduction).expect("device");
                assert_eq!(out.shape(), &[] as &[usize], "{what}: shape");
                assert_bits_or_nan_class(&download(&ops, &out), &want, &what);

                let token = DispatchFailureCell::default();
                let tracked = ops
                    .mse_loss_device_tracked(&pb, &tb, reduction, &token)
                    .expect("tracked");
                assert_bits_or_nan_class(&download(&ops, &tracked), &want, &what);

                if kind == 0 {
                    let sum_sq: f64 = p
                        .iter()
                        .zip(&t)
                        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                        .sum();
                    let r = match reduction {
                        MseReduction::Sum => sum_sq,
                        _ => sum_sq / n as f64,
                    } as f32;
                    fandhe_ai_backend_cpu::parity::assert_parity(
                        &format!("{what} cpu parity"),
                        &download(&ops, &out),
                        &[r],
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn backward_device_matches_host_bits_and_cpu_reference() {
    let ops = ops();
    for shape in SHAPES {
        let n: usize = shape.iter().product();
        for kind in 0..4u8 {
            let p = pattern(n, 3, kind);
            let t = pattern(n, 4, 0);
            let (pb, tb) = (upload(&ops, &p, shape), upload(&ops, &t, shape));
            let scales = [
                mean_scale(n, MseReduction::Mean),
                mean_scale(n, MseReduction::Sum),
                1.7,
            ];
            for scale in scales {
                let what = format!("backward {shape:?} kind={kind} scale={scale}");
                let want = ops
                    .mse_loss_backward(&tensor(&p, shape), &tensor(&t, shape), scale)
                    .expect("host backward");
                let want = want.as_slice().expect("contiguous").to_vec();
                let out = ops
                    .mse_loss_backward_device(&pb, &tb, scale)
                    .expect("device");
                assert_eq!(out.shape(), shape, "{what}: shape");
                assert_bits_or_nan_class(&download(&ops, &out), &want, &what);

                let token = DispatchFailureCell::default();
                let tracked = ops
                    .mse_loss_backward_device_tracked(&pb, &tb, scale, &token)
                    .expect("tracked");
                assert_bits_or_nan_class(&download(&ops, &tracked), &want, &what);

                if kind == 0 {
                    let reference: Vec<f32> =
                        p.iter().zip(&t).map(|(&a, &b)| scale * (a - b)).collect();
                    fandhe_ai_backend_cpu::parity::assert_parity(
                        &format!("{what} cpu parity"),
                        &download(&ops, &out),
                        &reference,
                    );
                }
            }
        }
    }
}

/// download を挟まず forward を連続で積んでも、永続 `partial` ワークスペースの
/// 再利用で結果が壊れない（サイズ違い = `num_blocks` 違いを含む）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn workspace_reuse_without_intermediate_sync_keeps_bits() {
    let ops = ops();
    let sizes = [4099usize, 257, 1024 * 256 + 1, 64, 4099];
    let mut inputs = Vec::new();
    for (k, &n) in sizes.iter().enumerate() {
        let p = pattern(n, 10 + k, 0);
        let t = pattern(n, 20 + k, 0);
        inputs.push((p, t, n));
    }
    let bufs: Vec<_> = inputs
        .iter()
        .map(|(p, t, n)| (upload(&ops, p, &[*n]), upload(&ops, t, &[*n])))
        .collect();
    let outs: Vec<_> = bufs
        .iter()
        .map(|(pb, tb)| {
            ops.mse_loss_device(pb, tb, MseReduction::Mean)
                .expect("device")
        })
        .collect();
    for (k, out) in outs.iter().enumerate() {
        let (p, t, n) = &inputs[k];
        let want = ops
            .mse_loss(&tensor(p, &[*n]), &tensor(t, &[*n]), MseReduction::Mean)
            .expect("host");
        assert_bits_or_nan_class(
            &download(&ops, out),
            want.as_slice().expect("contiguous"),
            &format!("workspace reuse #{k} n={n}"),
        );
    }
}

/// forward → backward を同期なしで積み、最後の download で両方がホスト版と bit 一致。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tracked_chain_without_sync_matches_host() {
    let ops = ops();
    let shape = [64usize, 10];
    let n = 640;
    let p = pattern(n, 5, 0);
    let t = pattern(n, 6, 0);
    let (pb, tb) = (upload(&ops, &p, &shape), upload(&ops, &t, &shape));
    let token = DispatchFailureCell::default();
    let scale = mean_scale(n, MseReduction::Mean);
    let loss = ops
        .mse_loss_device_tracked(&pb, &tb, MseReduction::Mean, &token)
        .expect("fwd");
    let grad = ops
        .mse_loss_backward_device_tracked(&pb, &tb, scale, &token)
        .expect("bwd");
    let want_loss = ops
        .mse_loss(&tensor(&p, &shape), &tensor(&t, &shape), MseReduction::Mean)
        .expect("host fwd");
    let want_grad = ops
        .mse_loss_backward(&tensor(&p, &shape), &tensor(&t, &shape), scale)
        .expect("host bwd");
    assert_bits_or_nan_class(
        &download(&ops, &loss),
        want_loss.as_slice().expect("contiguous"),
        "chain loss",
    );
    assert_bits_or_nan_class(
        &download(&ops, &grad),
        want_grad.as_slice().expect("contiguous"),
        "chain grad",
    );
    assert!(!token.is_set());
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn empty_and_error_paths() {
    let ops = ops();
    let pe = upload(&ops, &[], &[0, 5]);
    let te = upload(&ops, &[], &[0, 5]);
    let loss = ops
        .mse_loss_device(&pe, &te, MseReduction::Mean)
        .expect("empty fwd");
    assert_eq!(download(&ops, &loss), vec![0.0]);
    let grad = ops
        .mse_loss_backward_device(&pe, &te, 1.0)
        .expect("empty bwd");
    assert_eq!(grad.shape(), &[0, 5]);

    let a = upload(&ops, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = upload(&ops, &[1.0, 2.0, 3.0], &[3]);
    assert!(matches!(
        ops.mse_loss_device(&a, &b, MseReduction::Sum),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        ops.mse_loss_backward_device(&a, &b, 1.0),
        Err(BackendError::ShapeMismatch(_))
    ));
}
