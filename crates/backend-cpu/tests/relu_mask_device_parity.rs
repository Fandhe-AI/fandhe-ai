//! `CpuBackendOps::relu_backward_mask_device`（および `_into_tracked`）の受け入れ基準対応テスト
//! （イシュー #3253。決定記録 `docs/compat-resident-train-step-decision.md` §13）。
//!
//! 意味論は乗算ではなく**選択**（`mask > 0 ? g : +0.0`。参照は `autodiff::grad::elementwise_mul_mask`
//! と同じ式）。g の NaN payload・-0.0・inf を mask が真ならビットのまま保存し、mask が NaN なら +0.0。

use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, DeviceMatrixLayout, DispatchFailureCell, ShapeError, Tensor,
};

fn dev(mem: &CpuMemory, v: Vec<f32>, shape: &[usize]) -> DeviceBuffer<f32> {
    mem.upload(&Tensor::new(v, shape).unwrap()).unwrap()
}

fn read(mem: &CpuMemory, b: &DeviceBuffer<f32>) -> Vec<u32> {
    mem.download(b)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 特殊値を巡回させた g と mask（NaN payload・-0.0・inf・非正規化数・負値を含む）。
fn specials(len: usize, mask: bool) -> Vec<f32> {
    let g_pool = [
        f32::from_bits(0x7fc0_1234),
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.5,
        -2.5,
        0.0,
        1e-40,
    ];
    let m_pool = [f32::NAN, 0.0, -0.0, 1e-40, -1.0, 2.0, f32::INFINITY, -1e-40];
    let pool: &[f32] = if mask { &m_pool } else { &g_pool };
    (0..len)
        .map(|i| pool[(i * 3 + usize::from(mask)) % pool.len()])
        .collect()
}

fn reference(g_logical: &[f32], mask: &[f32]) -> Vec<u32> {
    g_logical
        .iter()
        .zip(mask)
        .map(|(&g, &m)| if m > 0.0 { g } else { 0.0 }.to_bits())
        .collect()
}

/// 格納 `[rows, cols]` を転置した論理 `[cols, rows]`（行優先）。
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

#[test]
fn matches_select_reference_row_major_and_transposed() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    for &(r, c) in &[(1usize, 1usize), (3, 5), (8, 8), (64, 33)] {
        let g = specials(r * c, false);
        let gd = dev(&mem, g.clone(), &[r, c]);
        let (gs, ls) = ([r, c], [c, r]);

        // RowMajor: 論理 [r, c]
        let mask = specials(r * c, true);
        let md = dev(&mem, mask.clone(), &[r, c]);
        let out = ops
            .relu_backward_mask_device(
                DeviceBufferView::new(&gd, 0, &gs).unwrap(),
                DeviceMatrixLayout::RowMajor,
                DeviceBufferView::new(&md, 0, &gs).unwrap(),
            )
            .unwrap();
        assert_eq!(out.shape(), &[r, c]);
        assert_eq!(read(&mem, &out), reference(&g, &mask), "row-major {r}x{c}");

        // Transposed: 格納 [r, c]・論理 [c, r]
        let mask_t = specials(r * c, true);
        let md_t = dev(&mem, mask_t.clone(), &[c, r]);
        let out = ops
            .relu_backward_mask_device(
                DeviceBufferView::new(&gd, 0, &gs).unwrap(),
                DeviceMatrixLayout::Transposed,
                DeviceBufferView::new(&md_t, 0, &ls).unwrap(),
            )
            .unwrap();
        assert_eq!(out.shape(), &[c, r]);
        assert_eq!(
            read(&mem, &out),
            reference(&transpose(&g, r, c), &mask_t),
            "transposed {r}x{c}"
        );
    }
}

#[test]
fn honors_view_offsets_and_into_overwrites_dirty_out() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    let (r, c) = (4usize, 6usize);
    let n = r * c;
    let g_all = specials(n + 5, false);
    let m_all = specials(n + 7, true);
    let gd = dev(&mem, g_all.clone(), &[n + 5]);
    let md = dev(&mem, m_all.clone(), &[n + 7]);
    let shape = [r, c];
    let expected = reference(&g_all[5..5 + n], &m_all[7..7 + n]);
    let alloc = ops
        .relu_backward_mask_device(
            DeviceBufferView::new(&gd, 5, &shape).unwrap(),
            DeviceMatrixLayout::RowMajor,
            DeviceBufferView::new(&md, 7, &shape).unwrap(),
        )
        .unwrap();
    assert_eq!(read(&mem, &alloc), expected);
    let mut out = dev(&mem, vec![f32::NAN; n], &[r, c]);
    ops.relu_backward_mask_device_into_tracked(
        DeviceBufferView::new(&gd, 5, &shape).unwrap(),
        DeviceMatrixLayout::RowMajor,
        DeviceBufferView::new(&md, 7, &shape).unwrap(),
        &mut out,
        &token,
    )
    .unwrap();
    assert_eq!(read(&mem, &out), expected);
}

#[test]
fn typed_errors_and_untouched_out() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    let g = dev(&mem, vec![1.0; 6], &[2, 3]);
    let m = dev(&mem, vec![1.0; 6], &[2, 3]);
    let g3 = dev(&mem, vec![1.0; 6], &[1, 2, 3]);
    let (s23, s123) = ([2, 3], [1, 2, 3]);
    // rank 違い
    assert!(matches!(
        ops.relu_backward_mask_device(
            DeviceBufferView::new(&g3, 0, &s123).unwrap(),
            DeviceMatrixLayout::RowMajor,
            DeviceBufferView::new(&m, 0, &s23).unwrap(),
        ),
        Err(BackendError::ShapeMismatch(ShapeError::RankMismatch { .. }))
    ));
    // mask 形が論理形と違う（Transposed の論理形は [3, 2]）
    assert!(matches!(
        ops.relu_backward_mask_device(
            DeviceBufferView::new(&g, 0, &s23).unwrap(),
            DeviceMatrixLayout::Transposed,
            DeviceBufferView::new(&m, 0, &s23).unwrap(),
        ),
        Err(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch { .. }
        ))
    ));
    // out 形違い
    let mut bad = dev(&mem, vec![9.0; 6], &[3, 2]);
    assert!(
        ops.relu_backward_mask_device_into_tracked(
            DeviceBufferView::new(&g, 0, &s23).unwrap(),
            DeviceMatrixLayout::RowMajor,
            DeviceBufferView::new(&m, 0, &s23).unwrap(),
            &mut bad,
            &token,
        )
        .is_err()
    );
    assert_eq!(read(&mem, &bad), vec![9.0f32.to_bits(); 6]);
    // 空形状の _into は確保経路へ任せる
    let e = dev(&mem, vec![], &[0, 3]);
    let s03 = [0, 3];
    let mut o = dev(&mem, vec![], &[0, 3]);
    assert!(matches!(
        ops.relu_backward_mask_device_into_tracked(
            DeviceBufferView::new(&e, 0, &s03).unwrap(),
            DeviceMatrixLayout::RowMajor,
            DeviceBufferView::new(&e, 0, &s03).unwrap(),
            &mut o,
            &token,
        ),
        Err(BackendError::Unsupported(_))
    ));
}
