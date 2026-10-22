//! イシュー #3221: `CudaBackendOps::relu_backward_mask_device`（および
//! `_tracked`）の実機 parity（`#[ignore]` 分離。CUDA 実機 DGX Spark GB10 等）。
//!
//! 参照は `autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)` と同一式
//! （`if m > 0.0 { g } else { 0.0 }`。Tape 経由で実関数と一致することは
//! `relu_mask_cpu_reference.rs` が Linux で担保）。判定に tolerance は使わず
//! bit 一致（NaN はクラス一致）。正式な実機記録（RULE.txt・5 run）は #3224。
//! 非正規化数（`1e-40`）を入れるのは NVRTC の ftz 有無で `m > 0.0` の結果が
//! CPU とずれないことを確かめるため。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test relu_mask_device_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    BackendOps, BinaryElementwiseOp, DeviceBuffer, DeviceBufferView, DeviceMatrixLayout,
    DispatchFailureCell, ShapeError, Tensor, UnaryElementwiseOp,
};

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    const SPECIAL: [f32; 11] = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.5,
        -2.25,
        1e-30,
        1e-40,
        -1e-40,
        3.0e38,
    ];
    (0..n)
        .map(|i| {
            if (i + salt).is_multiple_of(3) {
                SPECIAL[(i * 7 + salt) % SPECIAL.len()]
            } else {
                ((i * 31 + salt * 17) % 23) as f32 - 11.0
            }
        })
        .collect()
}

fn transpose(stored: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = stored[i * c + j];
        }
    }
    out
}

fn reference(g_logical: &[f32], mask: &[f32]) -> Vec<f32> {
    g_logical
        .iter()
        .zip(mask)
        .map(|(&g, &m)| if m > 0.0 { g } else { 0.0 })
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

fn upload(ops: &CudaBackendOps, data: &[f32]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), &[data.len()]).unwrap())
        .expect("upload")
}

/// Metal 側の `SHAPES` に揃えた形状 + m=64 の MLP 形状 + ブロック境界をまたぐ端数。
const SHAPES: [(usize, usize); 10] = [
    (1, 1),
    (1, 13),
    (13, 1),
    (7, 13),
    (63, 65),
    (64, 128),
    (64, 257),
    (2, 3),
    (257, 3),
    (3, 4099),
];

fn run_case(layout: DeviceMatrixLayout, r: usize, c: usize, g_off: usize, m_off: usize) {
    let ops = ops();
    let n = r * c;
    let g_store = pattern(g_off + n + 3, 1);
    let m_store = pattern(m_off + n + 3, 2);
    let g_buf = upload(&ops, &g_store);
    let m_buf = upload(&ops, &m_store);
    let (logical, g_logical) = match layout {
        DeviceMatrixLayout::Transposed => ([c, r], transpose(&g_store[g_off..g_off + n], r, c)),
        _ => ([r, c], g_store[g_off..g_off + n].to_vec()),
    };
    let want = reference(&g_logical, &m_store[m_off..m_off + n]);

    let g_shape = [r, c];
    let g_view = DeviceBufferView::new(&g_buf, g_off, &g_shape).unwrap();
    let m_view = DeviceBufferView::new(&m_buf, m_off, &logical).unwrap();

    let what = format!("{layout:?} {r}x{c} g_off={g_off} m_off={m_off}");
    let out = ops
        .relu_backward_mask_device(g_view, layout, m_view)
        .expect("non-tracked variant");
    assert_eq!(out.shape(), logical);
    let got = ops.memory_ops().unwrap().download(&out).unwrap();
    assert_bits_or_nan_class(got.as_slice().unwrap(), &want, &what);

    let token = DispatchFailureCell::new();
    let out_t = ops
        .relu_backward_mask_device_tracked(g_view, layout, m_view, &token)
        .expect("tracked variant");
    let got_t = ops.memory_ops().unwrap().download(&out_t).unwrap();
    assert_bits_or_nan_class(got_t.as_slice().unwrap(), &want, &format!("tracked {what}"));
    assert!(!token.is_set());
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn row_major_bit_identical_including_view_offsets() {
    for &(r, c) in &SHAPES {
        for (g_off, m_off) in [(0, 0), (5, 3), (0, 9)] {
            run_case(DeviceMatrixLayout::RowMajor, r, c, g_off, m_off);
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn transposed_bit_identical_including_view_offsets() {
    for &(r, c) in &SHAPES {
        for (g_off, m_off) in [(0, 0), (4, 6)] {
            run_case(DeviceMatrixLayout::Transposed, r, c, g_off, m_off);
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn error_paths_are_typed() {
    let ops = ops();
    let g_buf = upload(&ops, &[1.0; 12]);
    let m_buf = upload(&ops, &[1.0; 12]);

    let s1 = [12usize];
    let g1 = DeviceBufferView::new(&g_buf, 0, &s1).unwrap();
    let m1 = DeviceBufferView::new(&m_buf, 0, &s1).unwrap();
    let e = ops
        .relu_backward_mask_device(g1, DeviceMatrixLayout::RowMajor, m1)
        .unwrap_err();
    assert!(matches!(
        e,
        BackendError::ShapeMismatch(ShapeError::RankMismatch { .. })
    ));

    let sg = [3usize, 4];
    let sm = [4usize, 3];
    let g = DeviceBufferView::new(&g_buf, 0, &sg).unwrap();
    let m = DeviceBufferView::new(&m_buf, 0, &sm).unwrap();
    let e = ops
        .relu_backward_mask_device(g, DeviceMatrixLayout::RowMajor, m)
        .unwrap_err();
    assert!(matches!(
        e,
        BackendError::ShapeMismatch(ShapeError::ShapeMismatch { .. })
    ));

    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_buf = cpu
        .memory_ops()
        .unwrap()
        .upload(&Tensor::new(vec![1.0f32; 12], &[12]).unwrap())
        .unwrap();
    let gc = DeviceBufferView::new(&cpu_buf, 0, &sg).unwrap();
    let mm = DeviceBufferView::new(&m_buf, 0, &sg).unwrap();
    let e = ops
        .relu_backward_mask_device(gc, DeviceMatrixLayout::RowMajor, mm)
        .unwrap_err();
    assert!(matches!(e, BackendError::DeviceMismatch));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn zero_numel_returns_empty_output() {
    let ops = ops();
    let g_buf = upload(&ops, &[1.0; 4]);
    let m_buf = upload(&ops, &[1.0; 4]);
    let s = [0usize, 5];
    let g = DeviceBufferView::new(&g_buf, 0, &s).unwrap();
    let m = DeviceBufferView::new(&m_buf, 0, &s).unwrap();
    let out = ops
        .relu_backward_mask_device(g, DeviceMatrixLayout::RowMajor, m)
        .unwrap();
    assert_eq!(out.shape(), [0, 5]);
    assert_eq!(out.numel(), 0);
}

/// チェーン: `add → relu → mask 選択(tracked) → mul` を `download` 1 回で実行し、
/// 各演算をホストで合成した結果と bit 同一であること（中間で H2D／D2H・同期を
/// 挟まずに成立することの機能的検証。正式な同期回数の計測は #3224）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn chain_with_relu_mask_matches_per_op_host_composition() {
    let ops = ops();
    let mem = ops.memory_ops().unwrap();
    let (r, c) = (64usize, 129usize);
    let n = r * c;
    let a_h: Vec<f32> = (0..n).map(|i| ((i * 13) % 29) as f32 - 14.0).collect();
    let b_h: Vec<f32> = (0..n).map(|i| ((i * 7) % 17) as f32 - 8.0).collect();
    let a = Tensor::new(a_h.clone(), &[r, c]).unwrap();
    let b = Tensor::new(b_h.clone(), &[r, c]).unwrap();
    let a_buf = mem.upload(&a).unwrap();
    let b_buf = mem.upload(&b).unwrap();

    let sum = ops
        .binary_elementwise_device(BinaryElementwiseOp::Add, &a_buf, &b_buf)
        .unwrap();
    let act = ops
        .unary_elementwise_device(UnaryElementwiseOp::Relu, &sum)
        .unwrap();
    let shape = [r, c];
    let token = DispatchFailureCell::new();
    let sel = ops
        .relu_backward_mask_device_tracked(
            DeviceBufferView::new(&a_buf, 0, &shape).unwrap(),
            DeviceMatrixLayout::RowMajor,
            DeviceBufferView::new(&act, 0, &shape).unwrap(),
            &token,
        )
        .unwrap();
    let out = ops
        .binary_elementwise_device(BinaryElementwiseOp::Mul, &sel, &b_buf)
        .unwrap();
    let got = mem.download(&out).unwrap();

    let sum_h = ops.add(&a, &b).unwrap();
    let act_h = ops.relu(&sum_h).unwrap();
    let sel_h = reference(&a_h, act_h.contiguous().as_slice().unwrap());
    let want: Vec<f32> = sel_h.iter().zip(&b_h).map(|(x, y)| x * y).collect();
    assert_bits_or_nan_class(got.as_slice().unwrap(), &want, "chain");
    assert!(!token.is_set());
}
