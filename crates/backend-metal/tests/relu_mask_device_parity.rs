//! イシュー #3218: `MetalBackendOps::relu_backward_mask_device`（および
//! `_tracked`）の実機 parity。macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`#![cfg(target_os = "macos")]` + `#[ignore]`）。
//!
//! 参照は `autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)` と同一式
//! （`if m > 0.0 { g } else { 0.0 }`。Tape 経由で実関数と一致することは
//! `relu_mask_cpu_reference.rs` が Linux で担保）。判定に tolerance は使わず
//! bit 一致（NaN はクラス一致）。正式な実機記録（RULE.txt・5 run）は #3224。
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal --test relu_mask_device_parity -- --ignored`

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::{__diagnostic_batch_counters_snapshot, MetalBackendOps};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DeviceBufferView, DeviceMatrixLayout, DispatchFailureCell,
    ShapeError, Tensor,
};

/// 診断カウンタはプロセス共有のため、ファイル内のテストを直列化する。
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    const SPECIAL: [f32; 9] = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.5,
        -2.25,
        1e-30,
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

fn upload(ops: &MetalBackendOps, data: &[f32]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), &[data.len()]).unwrap())
        .expect("upload")
}

const SHAPES: [(usize, usize); 9] = [
    (64, 256),
    (1, 1),
    (1, 13),
    (13, 1),
    (7, 13),
    (63, 65),
    (64, 128),
    (64, 257),
    (2, 3),
];

fn run_case(layout: DeviceMatrixLayout, r: usize, c: usize, g_off: usize, m_off: usize) {
    let ops = MetalBackendOps::new();
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
        .expect("sync variant");
    assert_eq!(out.shape(), logical);
    let got = ops.memory_ops().unwrap().download(&out).unwrap();
    assert_bits_or_nan_class(got.as_slice().unwrap(), &want, &what);

    // tracked 版は待たない。synchronize 相当の download（内部同期）後に一致し、
    // token は未設定のまま。
    let token = DispatchFailureCell::new();
    let out_t = ops
        .relu_backward_mask_device_tracked(g_view, layout, m_view, &token)
        .expect("tracked variant");
    let got_t = ops.memory_ops().unwrap().download(&out_t).unwrap();
    assert_bits_or_nan_class(got_t.as_slice().unwrap(), &want, &format!("tracked {what}"));
    assert!(!token.is_set());
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn row_major_bit_identical_including_view_offsets() {
    let _g = serialize();
    for &(r, c) in &SHAPES {
        for (g_off, m_off) in [(0, 0), (5, 3), (0, 9)] {
            run_case(DeviceMatrixLayout::RowMajor, r, c, g_off, m_off);
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn transposed_bit_identical_including_view_offsets() {
    let _g = serialize();
    for &(r, c) in &SHAPES {
        for (g_off, m_off) in [(0, 0), (4, 6)] {
            run_case(DeviceMatrixLayout::Transposed, r, c, g_off, m_off);
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn error_paths_are_typed() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let g_buf = upload(&ops, &[1.0; 12]);
    let m_buf = upload(&ops, &[1.0; 12]);

    // rank != 2
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

    // mask 形状不一致
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

    // CPU デバイスの DeviceBuffer
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
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn zero_numel_returns_empty_output() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
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

/// #3224: MLP の L1 出力形 `[64, 256]` で、tracked 版が wait を出さず（encode は 1）、
/// 非 tracked 版の対照が wait +1 になること（カウンタが常に 0 を返す壊れ方の否定）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tracked_variant_issues_no_wait() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let (r, c) = (64usize, 256usize);
    let n = r * c;
    for layout in [DeviceMatrixLayout::RowMajor, DeviceMatrixLayout::Transposed] {
        let g_store = pattern(n, 1);
        let m_store = pattern(n, 2);
        let g_buf = upload(&ops, &g_store);
        let m_buf = upload(&ops, &m_store);
        let (logical, g_logical) = match layout {
            DeviceMatrixLayout::Transposed => ([c, r], transpose(&g_store, r, c)),
            _ => ([r, c], g_store.clone()),
        };
        let want = reference(&g_logical, &m_store);
        let g_shape = [r, c];
        let g = DeviceBufferView::new(&g_buf, 0, &g_shape).unwrap();
        let m = DeviceBufferView::new(&m_buf, 0, &logical).unwrap();
        // 先行する GPU 仕事を download で同期しておく。
        let _ = ops.memory_ops().unwrap().download(&g_buf).unwrap();

        let token = DispatchFailureCell::new();
        let b = __diagnostic_batch_counters_snapshot().unwrap();
        let out_t = ops
            .relu_backward_mask_device_tracked(g, layout, m, &token)
            .unwrap();
        let a = __diagnostic_batch_counters_snapshot().unwrap();
        assert_eq!(
            a.wait_until_completed - b.wait_until_completed,
            0,
            "{layout:?}"
        );
        assert_eq!(a.encode_calls - b.encode_calls, 1, "{layout:?}");
        assert!(!token.is_set());
        let got = ops.memory_ops().unwrap().download(&out_t).unwrap();
        assert_bits_or_nan_class(got.as_slice().unwrap(), &want, &format!("{layout:?}"));

        let b = __diagnostic_batch_counters_snapshot().unwrap();
        let _ = ops.relu_backward_mask_device(g, layout, m).unwrap();
        let a = __diagnostic_batch_counters_snapshot().unwrap();
        assert_eq!(
            a.wait_until_completed - b.wait_until_completed,
            1,
            "{layout:?}"
        );
    }
}

/// #3224: マスクが非正規化数（`f32::MIN_POSITIVE` 未満の正値）のときの選択。
/// CPU 参照は `m > 0.0` が真（g を通す）。マスクはカーネル側でビット判定に
/// 直した（#3218 reopen。float 比較は FTZ で偽になっていた）。回帰検出用で、
/// tolerance・判定式は変えない。共有 `pattern` へは混ぜない（他の形状の
/// 判定を巻き込まないため）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn denormal_mask_selects_like_cpu_reference() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let (r, c) = (64usize, 256usize);
    let n = r * c;
    let g_store: Vec<f32> = (0..n).map(|i| (i % 23) as f32 - 11.0).collect();
    let m_store: Vec<f32> = (0..n)
        .map(|i| match i % 12 {
            0 => 1e-40,
            1 => -1e-40,
            2 => 1.0e-45,
            3 => f32::from_bits(1),
            4 => f32::from_bits(0x007F_FFFF),
            5 => f32::MIN_POSITIVE,
            6 => -0.0,
            7 => 0.0,
            8 => f32::INFINITY,
            9 => f32::from_bits(0x7FC0_0000),
            10 => f32::from_bits(0xFFC0_0000),
            _ => 1.5,
        })
        .collect();
    assert!(m_store.iter().take(3).all(|v| v.abs() < f32::MIN_POSITIVE));
    let g_buf = upload(&ops, &g_store);
    let m_buf = upload(&ops, &m_store);
    let shape = [r, c];
    let g = DeviceBufferView::new(&g_buf, 0, &shape).unwrap();
    let m = DeviceBufferView::new(&m_buf, 0, &shape).unwrap();
    let want = reference(&g_store, &m_store);
    let out = ops
        .relu_backward_mask_device(g, DeviceMatrixLayout::RowMajor, m)
        .unwrap();
    let got = ops.memory_ops().unwrap().download(&out).unwrap();
    assert_bits_or_nan_class(got.as_slice().unwrap(), &want, "denormal mask");

    // Transposed layout でも同じ（格納 [r, c] → 論理 [c, r]。件数は増やさない）。
    let mshape_t = [c, r];
    let m_t = DeviceBufferView::new(&m_buf, 0, &mshape_t).unwrap();
    let g_t = DeviceBufferView::new(&g_buf, 0, &shape).unwrap();
    let mut g_logical = vec![0.0f32; n];
    for i in 0..r {
        for j in 0..c {
            g_logical[j * r + i] = g_store[i * c + j];
        }
    }
    let want_t = reference(&g_logical, &m_store);
    let out_t = ops
        .relu_backward_mask_device(g_t, DeviceMatrixLayout::Transposed, m_t)
        .unwrap();
    let got_t = ops.memory_ops().unwrap().download(&out_t).unwrap();
    assert_bits_or_nan_class(
        got_t.as_slice().unwrap(),
        &want_t,
        "denormal mask (transposed)",
    );
}
