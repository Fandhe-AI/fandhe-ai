//! `ops/conv_pool.rs` の結線テスト（イシュー #3438・R7）。
//!
//! すべて `RocmBackendOps::with_host_model` を通して `BackendOps` の入口から呼び、`CpuBackendOps` と
//! `to_bits` で完全一致を確かめる。`assert_parity` の許容誤差は使わない（緩めもしない）。
//! HIPRTC のコンパイル結果そのものの検証ではない（実機確認は #3159）。

use super::*;
use crate::target::GfxTarget;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::BackendOps;

fn rocm() -> RocmBackendOps {
    RocmBackendOps::with_host_model(GfxTarget::Gfx942)
}

fn data(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 2001) as f32 / 1000.0
                - 1.0
        })
        .collect()
}

fn t(shape: &[usize], seed: u32) -> Tensor<f32> {
    Tensor::new(data(shape.iter().product(), seed), shape).unwrap()
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 許容誤差を使わない bit 完全一致（shape も一致）。
fn same(label: &str, got: &Tensor<f32>, want: &Tensor<f32>) {
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    assert_eq!(bits(got), bits(want), "{label}: bits");
}

fn cp(k: [usize; 2], s: [usize; 2], p: [usize; 2], d: [usize; 2], g: usize) -> Conv2dParams {
    Conv2dParams::new(k, s, p, d, g).unwrap()
}

fn pp(k: [usize; 2], s: [usize; 2], p: [usize; 2], d: [usize; 2]) -> Pool2dParams {
    Pool2dParams::new(k, Some(s), p, d).unwrap()
}

#[test]
fn im2col_and_col2im_match_cpu_bitwise() {
    let (o, c) = (rocm(), CpuBackendOps);
    // stride・padding・dilation・groups・重なり窓・1x1・端数ブロックを含む。
    let cases = [
        ([1, 1, 5, 5], cp([3, 3], [1, 1], [0, 0], [1, 1], 1)),
        ([2, 3, 7, 6], cp([3, 2], [2, 1], [1, 1], [1, 1], 1)),
        ([1, 4, 9, 8], cp([3, 3], [2, 2], [2, 1], [2, 1], 2)),
        ([2, 6, 6, 6], cp([2, 2], [1, 1], [0, 1], [1, 2], 3)),
        ([1, 2, 4, 4], cp([1, 1], [1, 1], [0, 0], [1, 1], 1)),
        ([3, 2, 11, 13], cp([5, 3], [1, 3], [2, 0], [1, 1], 1)),
        ([1, 3, 3, 3], cp([3, 3], [1, 1], [3, 3], [1, 1], 1)),
    ];
    for (i, (shape, params)) in cases.iter().enumerate() {
        let x = t(shape, 10 + i as u32);
        let want = c.im2col(&x, params).unwrap();
        let got = o.im2col(&x, params).unwrap();
        same(&format!("im2col[{i}]"), &got, &want);

        let col = t(want.shape(), 50 + i as u32);
        let want = c.col2im(&col, shape, params).unwrap();
        let got = o.col2im(&col, shape, params).unwrap();
        same(&format!("col2im[{i}]"), &got, &want);
    }
}

#[test]
fn col2im_cancellation_keeps_fixed_f64_order() {
    // 重なり窓で相殺が起きる列。f32 逐次和と f64 逐次和は一致しない（契約は f64）。
    let params = cp([3, 3], [1, 1], [1, 1], [1, 1], 1);
    let shape = [1usize, 1, 4, 4];
    let col_shape = im2col_out_shape(&shape, &params).unwrap();
    let n: usize = col_shape.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| [1e8f32, 1.0, -1e8, 1.0][i % 4]).collect();
    let col = Tensor::new(vals, &col_shape).unwrap();
    let want = CpuBackendOps.col2im(&col, &shape, &params).unwrap();
    let got = rocm().col2im(&col, &shape, &params).unwrap();
    same("col2im cancellation", &got, &want);
}

#[test]
fn pad_matches_cpu_bitwise() {
    let (o, c) = (rocm(), CpuBackendOps);
    type PadCase = (&'static [usize], Vec<(usize, usize)>);
    let cases: [PadCase; 6] = [
        (&[5], vec![(2, 3)]),
        (&[3, 4], vec![(0, 0), (1, 0)]),
        (&[3, 4], vec![(0, 2), (0, 0)]),
        (&[2, 3, 4], vec![(1, 1), (0, 2), (3, 0)]),
        (&[2, 2, 3, 5], vec![(1, 0), (0, 1), (2, 2), (0, 0)]),
        (&[7, 9], vec![(0, 0), (0, 0)]),
    ];
    for (i, (shape, pads)) in cases.iter().enumerate() {
        let x = t(shape, 70 + i as u32);
        for value in [0.0f32, -1.5, f32::NAN, f32::INFINITY] {
            let want = c.pad(&x, pads, value).unwrap();
            let got = o.pad(&x, pads, value).unwrap();
            same(&format!("pad[{i}] value={value}"), &got, &want);
        }
    }
}

#[test]
fn pad_empty_input_and_empty_output() {
    let (o, c) = (rocm(), CpuBackendOps);
    // 入力が空で出力が非空: value で埋まる。
    let x = Tensor::<f32>::zeros(&[0, 3]).unwrap();
    let pads = [(2, 1), (0, 0)];
    let want = c.pad(&x, &pads, 7.0).unwrap();
    let got = o.pad(&x, &pads, 7.0).unwrap();
    same("pad empty input", &got, &want);
    assert_eq!(got.shape(), &[3, 3]);
    // 出力が空。
    let got = o.pad(&x, &[(0, 0), (0, 0)], 7.0).unwrap();
    assert_eq!(got.shape(), &[0, 3]);
    // rank 0。
    let s = Tensor::new(vec![2.5f32], &[]).unwrap();
    same(
        "pad rank0",
        &o.pad(&s, &[], 1.0).unwrap(),
        &c.pad(&s, &[], 1.0).unwrap(),
    );
}

#[test]
fn max_pool_matches_cpu_values_and_indices() {
    let (o, c) = (rocm(), CpuBackendOps);
    let cases = [
        ([2, 3, 8, 8], pp([2, 2], [2, 2], [0, 0], [1, 1])),
        ([1, 2, 9, 7], pp([3, 3], [2, 1], [1, 1], [1, 1])),
        ([2, 2, 10, 10], pp([3, 3], [1, 1], [0, 0], [2, 2])),
        ([1, 1, 5, 6], pp([5, 3], [1, 1], [2, 1], [1, 1])),
        ([3, 2, 4, 4], pp([1, 1], [1, 1], [0, 0], [1, 1])),
    ];
    for (i, (shape, params)) in cases.iter().enumerate() {
        let x = t(shape, 90 + i as u32);
        let (wv, wi) = c.max_pool2d(&x, params).unwrap();
        let (gv, gi) = o.max_pool2d(&x, params).unwrap();
        same(&format!("max_pool[{i}] values"), &gv, &wv);
        assert_eq!(gi.shape(), wi.shape());
        assert_eq!(gi.as_slice(), wi.as_slice(), "max_pool[{i}] indices");
    }
}

#[test]
fn max_pool_ties_and_nan_follow_cpu() {
    let (o, c) = (rocm(), CpuBackendOps);
    // 値域を 3 値に絞って同値のタイを多発させ、NaN を点在させる。
    let shape = [2usize, 2, 6, 6];
    let n: usize = shape.iter().product();
    let mut vals: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 3) as f32).collect();
    for i in (0..n).step_by(11) {
        vals[i] = f32::NAN;
    }
    let x = Tensor::new(vals, &shape).unwrap();
    for params in [
        pp([3, 3], [1, 1], [1, 1], [1, 1]),
        pp([2, 2], [2, 2], [0, 0], [1, 1]),
    ] {
        let (wv, wi) = c.max_pool2d(&x, &params).unwrap();
        let (gv, gi) = o.max_pool2d(&x, &params).unwrap();
        same("max_pool nan/ties", &gv, &wv);
        assert_eq!(gi.as_slice(), wi.as_slice());
    }
    // padding 位置は勝者にならない: 全要素が負でも出力は負の入力値で、索引は入力平面内。
    let neg = Tensor::new(vec![-5.0f32; 16], &[1, 1, 4, 4]).unwrap();
    let (v, i) = o
        .max_pool2d(&neg, &pp([3, 3], [1, 1], [1, 1], [1, 1]))
        .unwrap();
    assert!(v.as_slice().unwrap().iter().all(|&x| x == -5.0));
    assert!(i.as_slice().unwrap().iter().all(|&x| (0..16).contains(&x)));
}

#[test]
fn avg_pool_matches_cpu_bitwise() {
    let (o, c) = (rocm(), CpuBackendOps);
    let cases = [
        ([2, 3, 8, 8], pp([2, 2], [2, 2], [0, 0], [1, 1])),
        ([1, 2, 9, 7], pp([3, 3], [2, 1], [1, 1], [1, 1])),
        ([1, 1, 5, 6], pp([5, 3], [1, 1], [2, 1], [1, 1])),
    ];
    for (i, (shape, params)) in cases.iter().enumerate() {
        let x = t(shape, 120 + i as u32);
        for cip in [true, false] {
            let want = c.avg_pool2d(&x, params, cip).unwrap();
            let got = o.avg_pool2d(&x, params, cip).unwrap();
            same(&format!("avg_pool[{i}] cip={cip}"), &got, &want);
        }
    }
    // 相殺列: f32 逐次和と f64 逐次和が食い違う入力でも CPU と一致する。
    let vals: Vec<f32> = (0..64).map(|i| [1e8f32, 1.0, -1e8, 1.0][i % 4]).collect();
    let x = Tensor::new(vals, &[1, 1, 8, 8]).unwrap();
    let params = pp([4, 4], [2, 2], [0, 0], [1, 1]);
    same(
        "avg_pool cancellation",
        &o.avg_pool2d(&x, &params, true).unwrap(),
        &c.avg_pool2d(&x, &params, true).unwrap(),
    );
}

#[test]
fn avg_pool_rejects_dilation_before_touching_device() {
    let x = t(&[1, 1, 8, 8], 3);
    let params = pp([2, 2], [1, 1], [0, 0], [2, 1]);
    // HostModel でも Hip でも同じく HIP に触れる前に Unsupported。
    for o in [rocm(), RocmBackendOps::new(GfxTarget::Gfx942)] {
        assert!(matches!(
            o.avg_pool2d(&x, &params, true),
            Err(BackendError::Unsupported(_))
        ));
    }
}

#[test]
fn adaptive_avg_pool_matches_cpu_bitwise() {
    let (o, c) = (rocm(), CpuBackendOps);
    let cases = [
        ([2, 3, 8, 8], [4, 4]),
        ([1, 2, 7, 5], [3, 2]),
        ([1, 1, 3, 3], [5, 7]), // 拡大
        ([2, 2, 10, 9], [1, 1]),
        ([1, 1, 6, 6], [6, 6]),
    ];
    for (i, (shape, out)) in cases.iter().enumerate() {
        let x = t(shape, 150 + i as u32);
        same(
            &format!("adaptive[{i}]"),
            &o.adaptive_avg_pool2d(&x, *out).unwrap(),
            &c.adaptive_avg_pool2d(&x, *out).unwrap(),
        );
    }
}

#[test]
fn non_contiguous_inputs_match_cpu() {
    let (o, c) = (rocm(), CpuBackendOps);
    let base = t(&[1, 6, 5, 4], 200);
    // 軸 2・3 を入れ替えた view（非連続）。
    let x = base.transpose(2, 3).unwrap();
    assert!(!x.is_contiguous());
    let params = cp([2, 2], [1, 1], [1, 0], [1, 1], 2);
    same(
        "im2col view",
        &o.im2col(&x, &params).unwrap(),
        &c.im2col(&x, &params).unwrap(),
    );
    let pool = pp([2, 2], [1, 1], [0, 0], [1, 1]);
    let (gv, gi) = o.max_pool2d(&x, &pool).unwrap();
    let (wv, wi) = c.max_pool2d(&x, &pool).unwrap();
    same("max_pool view", &gv, &wv);
    assert_eq!(gi.as_slice(), wi.as_slice());
    same(
        "avg_pool view",
        &o.avg_pool2d(&x, &pool, false).unwrap(),
        &c.avg_pool2d(&x, &pool, false).unwrap(),
    );
    same(
        "adaptive view",
        &o.adaptive_avg_pool2d(&x, [2, 3]).unwrap(),
        &c.adaptive_avg_pool2d(&x, [2, 3]).unwrap(),
    );
    let pads = [(0, 0), (1, 0), (0, 2), (1, 1)];
    same(
        "pad view",
        &o.pad(&x, &pads, 0.5).unwrap(),
        &c.pad(&x, &pads, 0.5).unwrap(),
    );
}

#[test]
fn shape_errors_are_typed() {
    let o = rocm();
    let x3 = t(&[2, 3, 4], 1);
    let cparams = cp([3, 3], [1, 1], [0, 0], [1, 1], 1);
    let pparams = pp([2, 2], [1, 1], [0, 0], [1, 1]);
    assert!(matches!(
        o.im2col(&x3, &cparams),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        o.max_pool2d(&x3, &pparams),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        o.avg_pool2d(&x3, &pparams, true),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        o.adaptive_avg_pool2d(&x3, [2, 2]),
        Err(BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        o.pad(&x3, &[(1, 1)], 0.0),
        Err(BackendError::ShapeMismatch(_))
    ));
    // col2im の d_col 形状不一致。
    let bad_col = t(&[1, 1, 9, 4], 2);
    assert!(matches!(
        o.col2im(&bad_col, &[1, 1, 5, 5], &cparams),
        Err(BackendError::ShapeMismatch(_))
    ));
    // 窓が入力より大きい。
    let x = t(&[1, 1, 2, 2], 3);
    assert!(matches!(
        o.im2col(&x, &cparams),
        Err(BackendError::ShapeMismatch(_))
    ));
}

#[test]
fn empty_outputs_do_not_touch_device() {
    let (o, c) = (rocm(), CpuBackendOps);
    let cparams = cp([3, 3], [1, 1], [1, 1], [1, 1], 1);
    let x = Tensor::<f32>::zeros(&[0, 2, 5, 5]).unwrap();
    same(
        "im2col N=0",
        &o.im2col(&x, &cparams).unwrap(),
        &c.im2col(&x, &cparams).unwrap(),
    );
    let col = c.im2col(&x, &cparams).unwrap();
    same(
        "col2im N=0",
        &o.col2im(&col, x.shape(), &cparams).unwrap(),
        &c.col2im(&col, x.shape(), &cparams).unwrap(),
    );
    let pool = pp([2, 2], [2, 2], [0, 0], [1, 1]);
    let (v, i) = o.max_pool2d(&x, &pool).unwrap();
    assert_eq!((v.numel(), i.numel()), (0, 0));
    assert_eq!(o.avg_pool2d(&x, &pool, true).unwrap().numel(), 0);
    assert_eq!(o.adaptive_avg_pool2d(&x, [2, 2]).unwrap().numel(), 0);
}

#[test]
fn max_pool_rejects_index_range_overflow_even_for_empty_batch() {
    let o = rocm();
    // N=0 でも H*W > i32::MAX は空出力の早期リターンより前に拒否する（CUDA と同順）。
    let x = Tensor::<f32>::zeros(&[0, 1, 65_536, 65_536]).unwrap();
    let pool = pp([1, 1], [1, 1], [0, 0], [1, 1]);
    assert!(matches!(
        o.max_pool2d(&x, &pool),
        Err(BackendError::ShapeMismatch(
            ShapeError::IndexRangeOverflow { .. }
        ))
    ));
}

#[test]
fn size_limits_map_to_unsupported() {
    let o = rocm();
    // pad: 入力が空で出力要素数が i32 を超える → Unsupported（巨大 Vec を確保しない）。
    let x = Tensor::<f32>::zeros(&[0]).unwrap();
    let big = i32::MAX as usize;
    assert!(matches!(
        o.pad(&x, &[(big, 1)], 0.0),
        Err(BackendError::Unsupported(_))
    ));
    // 出力 numel が i32 を超える avg_pool（入力は broadcast view で、確保前に Unsupported になる）。
    let v = Tensor::new(vec![1.0f32], &[1, 1, 1, 1])
        .unwrap()
        .broadcast_to(&[1, 1, 1, big + 1])
        .unwrap();
    let pool = pp([1, 1], [1, 1], [0, 0], [1, 1]);
    assert!(matches!(
        o.avg_pool2d(&v, &pool, true),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
fn hip_engine_without_rocm_orders_errors_before_device() {
    if crate::is_available() {
        return;
    }
    let o = RocmBackendOps::new(GfxTarget::Gfx942);
    let cparams = cp([3, 3], [1, 1], [1, 1], [1, 1], 1);
    let pool = pp([2, 2], [1, 1], [0, 0], [1, 1]);
    let x = t(&[1, 2, 5, 5], 5);
    let col = CpuBackendOps.im2col(&x, &cparams).unwrap();
    // 有効な入力は DeviceUnavailable（panic しない）。
    let unavailable = |r: Result<Tensor<f32>, BackendError>| {
        assert!(
            matches!(r, Err(BackendError::DeviceUnavailable(_))),
            "{r:?}"
        )
    };
    unavailable(o.im2col(&x, &cparams));
    unavailable(o.col2im(&col, x.shape(), &cparams));
    unavailable(o.pad(&x, &[(0, 0), (0, 0), (1, 1), (1, 1)], 0.0));
    unavailable(o.avg_pool2d(&x, &pool, true));
    unavailable(o.adaptive_avg_pool2d(&x, [2, 2]));
    assert!(matches!(
        o.max_pool2d(&x, &pool),
        Err(BackendError::DeviceUnavailable(_))
    ));
    // 形状エラー・Unsupported・空出力は HIP より先に確定する。
    assert!(matches!(
        o.im2col(&t(&[2, 3, 4], 1), &cparams),
        Err(BackendError::ShapeMismatch(_))
    ));
    let dil = pp([2, 2], [1, 1], [0, 0], [2, 1]);
    assert!(matches!(
        o.avg_pool2d(&x, &dil, true),
        Err(BackendError::Unsupported(_))
    ));
    let empty = Tensor::<f32>::zeros(&[0, 2, 5, 5]).unwrap();
    assert_eq!(o.max_pool2d(&empty, &pool).unwrap().0.numel(), 0);
    assert_eq!(o.pad(&empty, &[(0, 0); 4], 0.0).unwrap().numel(), 0);
}
