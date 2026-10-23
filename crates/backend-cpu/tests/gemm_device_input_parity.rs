//! `CpuBackendOps::gemm_fp32_strict_into_device`（dW＋bias 縮約の融合）と `gemm_resident_lhs_device`
//! （d_input）の受け入れ基準対応テスト（イシュー #3253。決定記録 §13）。
//!
//! 比較先はホスト版（`gemm_fp32_strict_into`・`gemm_resident_lhs`）と、`autodiff::eval::
//! reduce_bias_grad_rows` と同式の `f64` 逐次和（`k == 1` は複製）。`to_bits()` 完全一致で比較する。

use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, DeviceMatrixLayout, DispatchFailureCell, Tensor,
};

fn data(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `reduce_bias_grad_rows` と同式（g: `[k, n]` 行優先）。
fn bias_reference(g: &[f32], k: usize, n: usize) -> Vec<f32> {
    if k == 1 {
        return g[..n].to_vec();
    }
    let mut acc = vec![0f64; n];
    for row in 0..k {
        for (col, a) in acc.iter_mut().enumerate() {
            *a += f64::from(g[row * n + col]);
        }
    }
    acc.into_iter().map(|v| v as f32).collect()
}

/// dW: a = x_t（格納 x:`[k, m]` の転置）, b = g（`[k, n]`）。ホスト版と bias 参照に bit 一致する。
fn check_dw(m: usize, n: usize, k: usize, out_offset: usize, with_bias: bool) {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    let x = data(21, k * m);
    let g = data(22, k * n);
    // ホスト版
    let x_t = Tensor::new(x.clone(), &[k, m])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let g_t = Tensor::new(g.clone(), &[k, n]).unwrap();
    let total = out_offset + m * n + 3 + n + 2;
    let mut host_out = dev(&mem, vec![7.0; total], &[total]);
    ops.gemm_fp32_strict_into(&x_t, &g_t, &mut host_out, out_offset)
        .unwrap();
    let host_w = read(&mem, &host_out)[out_offset..out_offset + m * n].to_vec();

    let (xd, gd) = (dev(&mem, x, &[k, m]), dev(&mem, g.clone(), &[k, n]));
    let (xs, gs) = ([k, m], [k, n]);
    let bias_off = out_offset + m * n + 3;
    let bias = with_bias.then_some((bias_off, n));
    // 事前に汚した out（0 埋めの確認）
    let mut out = dev(&mem, vec![f32::NAN; total], &[total]);
    let filled = ops
        .gemm_fp32_strict_into_device_tracked(
            DeviceBufferView::new(&xd, 0, &xs).unwrap(),
            DeviceMatrixLayout::Transposed,
            DeviceBufferView::new(&gd, 0, &gs).unwrap(),
            DeviceMatrixLayout::RowMajor,
            &mut out,
            out_offset,
            bias,
            &token,
        )
        .unwrap();
    assert_eq!(filled, with_bias);
    let got = read(&mem, &out);
    assert_eq!(
        got[out_offset..out_offset + m * n],
        host_w[..],
        "weight {m}x{n}x{k}"
    );
    // 書き込み範囲外は不変
    for (i, v) in got.iter().enumerate() {
        let in_w = (out_offset..out_offset + m * n).contains(&i);
        let in_b = with_bias && (bias_off..bias_off + n).contains(&i);
        if !in_w && !in_b {
            assert!(
                f32::from_bits(*v).is_nan(),
                "idx {i} が範囲外なのに変更された"
            );
        }
    }
    if with_bias {
        assert_eq!(
            got[bias_off..bias_off + n],
            bits(&bias_reference(&g, k, n))[..],
            "bias {m}x{n}x{k}"
        );
    }
}

#[test]
fn dw_matches_host_and_bias_reference() {
    for &(m, n, k) in &[(784, 64, 256), (5, 7, 3), (6, 3, 1), (1, 1, 1), (9, 5, 64)] {
        for off in [0usize, 4] {
            for with_bias in [false, true] {
                check_dw(m, n, k, off, with_bias);
            }
        }
    }
}

#[test]
fn dw_batch_one_bias_keeps_negative_zero() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let xd = dev(&mem, vec![1.0, 2.0], &[1, 2]);
    let gd = dev(&mem, vec![-0.0, 0.5, -0.0], &[1, 3]);
    let (xs, gs) = ([1, 2], [1, 3]);
    let mut out = dev(&mem, vec![1.0; 9], &[9]);
    ops.gemm_fp32_strict_into_device(
        DeviceBufferView::new(&xd, 0, &xs).unwrap(),
        DeviceMatrixLayout::Transposed,
        DeviceBufferView::new(&gd, 0, &gs).unwrap(),
        DeviceMatrixLayout::RowMajor,
        &mut out,
        0,
        Some((6, 3)),
    )
    .unwrap();
    let got = read(&mem, &out);
    assert_eq!(got[6], (-0.0f32).to_bits());
    assert_eq!(got[8], (-0.0f32).to_bits());
}

#[test]
fn dw_errors_leave_out_untouched() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    // x: [3, 2]（論理 a = x^T は [2, 3]）, g: [3, 4] → weight は 2*4 = 8 要素
    let xd = dev(&mem, data(1, 6), &[3, 2]);
    let gd = dev(&mem, data(2, 12), &[3, 4]);
    let (xs, gs) = ([3, 2], [3, 4]);
    let call = |out: &mut DeviceBuffer<f32>, off: usize, bias: Option<(usize, usize)>| {
        ops.gemm_fp32_strict_into_device(
            DeviceBufferView::new(&xd, 0, &xs).unwrap(),
            DeviceMatrixLayout::Transposed,
            DeviceBufferView::new(&gd, 0, &gs).unwrap(),
            DeviceMatrixLayout::RowMajor,
            out,
            off,
            bias,
        )
    };
    let mut out = dev(&mem, vec![5.0; 20], &[20]);
    // オーバーフロー・範囲超過
    assert!(matches!(
        call(&mut out, usize::MAX - 1, None),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        call(&mut out, 13, None),
        Err(BackendError::InvalidArgument(_))
    ));
    // bias 幅の不一致・範囲超過・weight との重なり
    assert!(matches!(
        call(&mut out, 0, Some((10, 3))),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        call(&mut out, 0, Some((18, 4))),
        Err(BackendError::InvalidArgument(_))
    ));
    assert!(matches!(
        call(&mut out, 0, Some((6, 4))),
        Err(BackendError::InvalidArgument(_))
    ));
    assert_eq!(read(&mem, &out), vec![5.0f32.to_bits(); 20]);
}

#[test]
fn dw_transposed_transposed_is_unsupported() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = dev(&mem, data(1, 6), &[3, 2]);
    let b = dev(&mem, data(2, 12), &[4, 3]);
    let (as_, bs) = ([3, 2], [4, 3]);
    let mut out = dev(&mem, vec![5.0; 8], &[8]);
    let r = ops.gemm_fp32_strict_into_device(
        DeviceBufferView::new(&a, 0, &as_).unwrap(),
        DeviceMatrixLayout::Transposed,
        DeviceBufferView::new(&b, 0, &bs).unwrap(),
        DeviceMatrixLayout::Transposed,
        &mut out,
        0,
        None,
    );
    assert!(matches!(r, Err(BackendError::Unsupported(_))));
    assert_eq!(read(&mem, &out), vec![5.0f32.to_bits(); 8]);
}

#[test]
fn d_input_matches_host_and_into_overwrites() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    for &(p, q, r) in &[
        (784usize, 64usize, 33usize),
        (5, 3, 7),
        (1, 1, 1),
        (9, 10, 64),
    ] {
        let w = data(31, p * q);
        let g = data(32, r * q); // g: [r, q]。論理 b = g^T = [q, r]
        let g_t = Tensor::new(g.clone(), &[r, q])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let wd = dev(&mem, w, &[p, q]);
        let gd = dev(&mem, g, &[r, q]);
        let (ws, gs) = ([p, q], [r, q]);
        let host = ops
            .gemm_resident_lhs(DeviceBufferView::new(&wd, 0, &ws).unwrap(), &g_t)
            .unwrap();
        let host = bits(host.contiguous().as_slice().unwrap());
        let got = ops
            .gemm_resident_lhs_device(
                DeviceBufferView::new(&wd, 0, &ws).unwrap(),
                DeviceBufferView::new(&gd, 0, &gs).unwrap(),
                DeviceMatrixLayout::Transposed,
            )
            .unwrap();
        assert_eq!(got.shape(), &[p, r]);
        assert_eq!(read(&mem, &got), host, "transposed {p}x{q}x{r}");
        let mut out = dev(&mem, vec![f32::NAN; p * r], &[p, r]);
        ops.gemm_resident_lhs_device_into_tracked(
            DeviceBufferView::new(&wd, 0, &ws).unwrap(),
            DeviceBufferView::new(&gd, 0, &gs).unwrap(),
            DeviceMatrixLayout::Transposed,
            &mut out,
            &token,
        )
        .unwrap();
        assert_eq!(read(&mem, &out), host, "into {p}x{q}x{r}");

        // RowMajor: b 格納 [q, r]
        let b = data(33, q * r);
        let bd = dev(&mem, b.clone(), &[q, r]);
        let bs = [q, r];
        let host = ops
            .gemm_resident_lhs(
                DeviceBufferView::new(&wd, 0, &ws).unwrap(),
                &Tensor::new(b, &[q, r]).unwrap(),
            )
            .unwrap();
        let got = ops
            .gemm_resident_lhs_device(
                DeviceBufferView::new(&wd, 0, &ws).unwrap(),
                DeviceBufferView::new(&bd, 0, &bs).unwrap(),
                DeviceMatrixLayout::RowMajor,
            )
            .unwrap();
        assert_eq!(
            read(&mem, &got),
            bits(host.contiguous().as_slice().unwrap())
        );
    }
}

#[test]
fn d_input_degenerate_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let token = DispatchFailureCell::new();
    // q == 0 → 全 0 の [p, r]
    let wd = dev(&mem, vec![], &[3, 0]);
    let bd = dev(&mem, vec![], &[0, 2]);
    let (ws, bs) = ([3, 0], [0, 2]);
    let got = ops
        .gemm_resident_lhs_device(
            DeviceBufferView::new(&wd, 0, &ws).unwrap(),
            DeviceBufferView::new(&bd, 0, &bs).unwrap(),
            DeviceMatrixLayout::RowMajor,
        )
        .unwrap();
    assert_eq!(got.shape(), &[3, 2]);
    assert_eq!(read(&mem, &got), vec![0u32; 6]);
    // p == 0 → 空
    let wd = dev(&mem, vec![], &[0, 2]);
    let bd = dev(&mem, vec![1.0; 6], &[2, 3]);
    let (ws, bs) = ([0, 2], [2, 3]);
    let got = ops
        .gemm_resident_lhs_device(
            DeviceBufferView::new(&wd, 0, &ws).unwrap(),
            DeviceBufferView::new(&bd, 0, &bs).unwrap(),
            DeviceMatrixLayout::RowMajor,
        )
        .unwrap();
    assert_eq!(got.shape(), &[0, 3]);
    // _into は退化形状で Unsupported（確保経路へ任せる）
    let mut out = dev(&mem, vec![], &[0, 3]);
    assert!(matches!(
        ops.gemm_resident_lhs_device_into_tracked(
            DeviceBufferView::new(&wd, 0, &ws).unwrap(),
            DeviceBufferView::new(&bd, 0, &bs).unwrap(),
            DeviceMatrixLayout::RowMajor,
            &mut out,
            &token,
        ),
        Err(BackendError::Unsupported(_))
    ));
}
