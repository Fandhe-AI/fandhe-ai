//! イシュー #3219: `MetalBackendOps::mse_loss_device`／`mse_loss_backward_device`（および
//! `_tracked`）の実機 parity。macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`#![cfg(target_os = "macos")]` + `#[ignore]`）。
//!
//! 参照はホスト入出力版 `mse_loss`／`mse_loss_backward`。両者は同じ
//! `MetalMse::encode_mse_*_tracked` を共有するため、判定は tolerance を使わず bit 一致
//! （NaN はクラス一致）。wait 回数は `__diagnostic_batch_counters_snapshot` で確認する。
//! 正式な実機記録（RULE.txt・5 run）は #3224。
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal --test mse_device_parity -- --ignored`

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::{__diagnostic_batch_counters_snapshot, MetalBackendOps};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DispatchFailureCell, MseReduction, ShapeError, Tensor,
};

/// 診断カウンタはプロセス共有のため、ファイル内のテストを直列化する。
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Finite,
    WithInf,
    WithNan,
}

fn pattern(n: usize, salt: usize, kind: Kind) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let base = ((i * 31 + salt * 17) % 23) as f32 - 11.0;
            match kind {
                Kind::Finite => base * 0.37 + if i % 5 == 0 { -0.0 } else { 1e-3 },
                Kind::WithInf if (i + salt).is_multiple_of(11) => {
                    if i % 2 == 0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                Kind::WithNan if (i + salt).is_multiple_of(13) => f32::NAN,
                _ => base * 0.37,
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

fn upload(ops: &MetalBackendOps, data: &[f32], shape: &[usize]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), shape).unwrap())
        .expect("upload")
}

fn download(ops: &MetalBackendOps, buf: &DeviceBuffer<f32>) -> Vec<f32> {
    ops.memory_ops()
        .unwrap()
        .download(buf)
        .unwrap()
        .as_slice()
        .unwrap()
        .to_vec()
}

const NUMELS: [usize; 7] = [1, 7, 255, 256, 257, 4096, 256 * 1024 + 3];
const KINDS: [Kind; 3] = [Kind::Finite, Kind::WithInf, Kind::WithNan];

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn forward_bit_identical_to_host_mse_loss() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    for &n in &NUMELS {
        for kind in KINDS {
            let p = pattern(n, 1, kind);
            let t = pattern(n, 2, Kind::Finite);
            let p_t = Tensor::new(p.clone(), &[n]).unwrap();
            let t_t = Tensor::new(t.clone(), &[n]).unwrap();
            let p_d = upload(&ops, &p, &[n]);
            let t_d = upload(&ops, &t, &[n]);
            for r in [MseReduction::Mean, MseReduction::Sum] {
                let what = format!("n={n} {kind:?} {r:?}");
                let want = ops.mse_loss(&p_t, &t_t, r).unwrap();
                let want = want.as_slice().unwrap().to_vec();
                let out = ops.mse_loss_device(&p_d, &t_d, r).unwrap();
                assert_eq!(out.shape(), Vec::<usize>::new(), "{what}: shape");
                assert_eq!(out.numel(), 1);
                assert_bits_or_nan_class(&download(&ops, &out), &want, &what);

                let token = DispatchFailureCell::new();
                let out_t = ops.mse_loss_device_tracked(&p_d, &t_d, r, &token).unwrap();
                assert_bits_or_nan_class(
                    &download(&ops, &out_t),
                    &want,
                    &format!("tracked {what}"),
                );
                assert!(!token.is_set());
            }
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backward_bit_identical_to_host_mse_loss_backward() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let mut shapes: Vec<Vec<usize>> = NUMELS.iter().map(|&n| vec![n]).collect();
    shapes.push(vec![64, 10]);
    shapes.push(vec![7, 13]);
    for shape in &shapes {
        let n: usize = shape.iter().product();
        for kind in KINDS {
            let p = pattern(n, 3, kind);
            let t = pattern(n, 4, Kind::Finite);
            let p_t = Tensor::new(p.clone(), shape).unwrap();
            let t_t = Tensor::new(t.clone(), shape).unwrap();
            let p_d = upload(&ops, &p, shape);
            let t_d = upload(&ops, &t, shape);
            for scale in [2.0 / n as f32, 2.0, -0.75] {
                let what = format!("shape={shape:?} {kind:?} scale={scale}");
                let want = ops.mse_loss_backward(&p_t, &t_t, scale).unwrap();
                let want = want.as_slice().unwrap().to_vec();
                let out = ops.mse_loss_backward_device(&p_d, &t_d, scale).unwrap();
                assert_eq!(out.shape(), shape.as_slice(), "{what}: shape");
                assert_bits_or_nan_class(&download(&ops, &out), &want, &what);

                let token = DispatchFailureCell::new();
                let out_t = ops
                    .mse_loss_backward_device_tracked(&p_d, &t_d, scale, &token)
                    .unwrap();
                assert_bits_or_nan_class(
                    &download(&ops, &out_t),
                    &want,
                    &format!("tracked {what}"),
                );
                assert!(!token.is_set());
            }
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tracked_variants_issue_no_wait() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let n = 4099;
    let p = pattern(n, 5, Kind::Finite);
    let t = pattern(n, 6, Kind::Finite);
    let p_d = upload(&ops, &p, &[n]);
    let t_d = upload(&ops, &t, &[n]);
    // open バッチを残さないため、一度同期を済ませる。
    let _ = download(&ops, &p_d);

    let token = DispatchFailureCell::new();
    let before = __diagnostic_batch_counters_snapshot().unwrap();
    let loss = ops
        .mse_loss_device_tracked(&p_d, &t_d, MseReduction::Mean, &token)
        .unwrap();
    let grad = ops
        .mse_loss_backward_device_tracked(&p_d, &t_d, 2.0 / n as f32, &token)
        .unwrap();
    let after = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(
        after.encode_calls - before.encode_calls,
        3,
        "forward 2 + backward 1"
    );
    assert_eq!(
        after.wait_until_completed - before.wait_until_completed,
        0,
        "tracked 版は wait を出さない"
    );

    // 検証用 download（内部同期）の後に値を確認する。
    let host_p = Tensor::new(p.clone(), &[n]).unwrap();
    let host_t = Tensor::new(t.clone(), &[n]).unwrap();
    let want_l = ops.mse_loss(&host_p, &host_t, MseReduction::Mean).unwrap();
    assert_bits_or_nan_class(&download(&ops, &loss), want_l.as_slice().unwrap(), "loss");
    let want_g = ops
        .mse_loss_backward(&host_p, &host_t, 2.0 / n as f32)
        .unwrap();
    assert_bits_or_nan_class(&download(&ops, &grad), want_g.as_slice().unwrap(), "grad");
    assert!(!token.is_set());

    // 比較: 非 tracked 版はそれぞれ wait +1。
    let b = __diagnostic_batch_counters_snapshot().unwrap();
    let _ = ops.mse_loss_device(&p_d, &t_d, MseReduction::Mean).unwrap();
    let m = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(m.wait_until_completed - b.wait_until_completed, 1);
    let _ = ops.mse_loss_backward_device(&p_d, &t_d, 1.0).unwrap();
    let a = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(a.wait_until_completed - m.wait_until_completed, 1);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn error_paths_are_typed() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let p = upload(&ops, &[1.0; 6], &[6]);
    let t = upload(&ops, &[1.0; 6], &[2, 3]);
    let e = ops.mse_loss_device(&p, &t, MseReduction::Mean).unwrap_err();
    assert!(matches!(
        e,
        BackendError::ShapeMismatch(ShapeError::ShapeMismatch { .. })
    ));
    let e = ops.mse_loss_backward_device(&p, &t, 1.0).unwrap_err();
    assert!(matches!(e, BackendError::ShapeMismatch(_)));

    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_buf = cpu
        .memory_ops()
        .unwrap()
        .upload(&Tensor::new(vec![1.0f32; 6], &[6]).unwrap())
        .unwrap();
    let e = ops
        .mse_loss_device(&cpu_buf, &p, MseReduction::Sum)
        .unwrap_err();
    assert!(matches!(e, BackendError::DeviceMismatch));
    let e = ops.mse_loss_backward_device(&p, &cpu_buf, 1.0).unwrap_err();
    assert!(matches!(e, BackendError::DeviceMismatch));
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn zero_numel_dispatches_nothing() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let p = upload(&ops, &[], &[0]);
    let t = upload(&ops, &[], &[0]);
    let before = __diagnostic_batch_counters_snapshot().unwrap();
    let loss = ops.mse_loss_device(&p, &t, MseReduction::Mean).unwrap();
    let grad = ops.mse_loss_backward_device(&p, &t, 1.0).unwrap();
    let after = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(after.encode_calls, before.encode_calls);
    assert_eq!(loss.shape(), Vec::<usize>::new());
    assert_eq!(download(&ops, &loss), vec![0.0f32]);
    assert_eq!(grad.shape(), [0]);
    assert_eq!(grad.numel(), 0);
}
