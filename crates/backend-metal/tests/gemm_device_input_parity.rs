//! イシュー #3220: `MetalBackendOps::gemm_fp32_strict_into_device`（dW。bias 縮約つき）・
//! `gemm_resident_lhs_device`（d_input）および `_tracked` 版の実機 parity。macOS 実機
//! （Apple Silicon）でのみコンパイル・実行する（`#![cfg(target_os = "macos")]` +
//! `#[ignore]`）。
//!
//! 参照はホスト入力版 `gemm_fp32_strict_into_with_bias_reduce_tracked`（dW）・
//! `gemm_resident_lhs`（d_input）。両者は同じ encode 本体・同じ `MatrixLayout` を共有する
//! ため、判定は tolerance を使わず bit 一致（NaN はクラス一致）。wait 回数は
//! `__diagnostic_batch_counters_snapshot` で確認する。トークン登録は in-crate テスト
//! （`ops::tests::gemm_device_tracked_registers_token_and_does_not_wait`）で確認する。
//! 正式な実機記録（RULE.txt・5 run・`.rev-stamp`）は #3224。
//!
//! 形状は (B, in, out)。dW は m = in・k = B・n = out、d_input は p = in・q = out・r = B。
//! ホスト版の `classify_2d` が TN/NT と判定しない縮退形状（in == 1 かつ B == 1 等）は
//! 比較の前提が崩れるため使わない。
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal --test gemm_device_input_parity -- --ignored --test-threads=1`

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

#[derive(Clone, Copy, Debug)]
enum Kind {
    Finite,
    /// NaN・inf・-0.0・巨大値を混ぜる。
    Special,
}

fn pattern(n: usize, salt: usize, kind: Kind) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let base = ((i * 31 + salt * 17) % 23) as f32 - 11.0;
            match kind {
                Kind::Finite => base * 0.37 + if i % 5 == 0 { -0.0 } else { 1e-3 },
                Kind::Special => match (i + salt) % 37 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => -0.0,
                    4 => 3.0e38,
                    _ => base * 0.37,
                },
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

fn upload_flat(ops: &MetalBackendOps, data: &[f32]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), &[data.len()]).unwrap())
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

/// 先頭に非ゼロ offset ぶんの番兵を置いた連結バッファへ `data` を upload し、
/// view の offset を返す（`view.offset() != 0` の経路を通す）。
const PAD: usize = 3;
fn upload_with_offset(ops: &MetalBackendOps, data: &[f32]) -> DeviceBuffer<f32> {
    let mut flat = vec![9.0f32; PAD];
    flat.extend_from_slice(data);
    flat.extend_from_slice(&[9.0, 9.0]);
    upload_flat(ops, &flat)
}

/// dW の out バッファ初期値（番兵）と、weight・bias の書き込み位置。
struct OutLayout {
    total: usize,
    out_offset: usize,
    bias_offset: usize,
}

fn out_layout(m: usize, n: usize) -> OutLayout {
    let out_offset = 3;
    let bias_offset = out_offset + m * n + 2;
    OutLayout {
        total: bias_offset + n + 4,
        out_offset,
        bias_offset,
    }
}

fn sentinel(total: usize) -> Vec<f32> {
    (0..total).map(|i| 7.5 + i as f32 * 0.125).collect()
}

/// (B, in, out) 形状: MLP 形状 + 端数形状。
const SHAPES: [(usize, usize, usize); 7] = [
    (64, 128, 256),
    (64, 256, 10),
    (64, 784, 128),
    (7, 13, 5),
    (63, 65, 67),
    (64, 257, 33),
    (3, 2, 1),
];

/// dW を device 版・ホスト版の両方で計算して bit 比較する。`nt == false` は TN
/// （a = x の格納 `[k, m]` を `Transposed`・b = g の格納 `[k, n]` を `RowMajor`）、
/// `nt == true` は NT（a の格納 `[m, k]` を `RowMajor`・b の格納 `[n, k]` を `Transposed`）。
fn check_dw(
    ops: &MetalBackendOps,
    (bsz, inf, outf): (usize, usize, usize),
    nt: bool,
    with_bias: bool,
    tracked: bool,
    kind: Kind,
) {
    let (m, k, n) = (inf, bsz, outf);
    let what = format!(
        "dW B={bsz} in={inf} out={outf} nt={nt} bias={with_bias} tracked={tracked} {kind:?}"
    );
    let a_data = pattern(m * k, 1, kind);
    let b_data = pattern(k * n, 2, Kind::Finite);
    let ol = out_layout(m, n);
    let bias = with_bias.then_some((ol.bias_offset, n));

    let (a_stored, a_layout, b_stored, b_layout) = if nt {
        (
            [m, k],
            DeviceMatrixLayout::RowMajor,
            [n, k],
            DeviceMatrixLayout::Transposed,
        )
    } else {
        (
            [k, m],
            DeviceMatrixLayout::Transposed,
            [k, n],
            DeviceMatrixLayout::RowMajor,
        )
    };

    // ホスト参照。
    let a_host = Tensor::new(a_data.clone(), &a_stored).unwrap();
    let b_host = Tensor::new(b_data.clone(), &b_stored).unwrap();
    let (a_host, b_host) = if nt {
        (a_host, b_host.transpose(0, 1).unwrap())
    } else {
        (a_host.transpose(0, 1).unwrap(), b_host)
    };
    let mut out_ref = upload_flat(ops, &sentinel(ol.total));
    let token_ref = DispatchFailureCell::new();
    let want_filled = ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            &a_host,
            &b_host,
            &mut out_ref,
            ol.out_offset,
            bias,
            &token_ref,
        )
        .expect("host dW");
    let want = download(ops, &out_ref);

    // device 版。
    let a_buf = upload_with_offset(ops, &a_data);
    let b_buf = upload_with_offset(ops, &b_data);
    let a_view = DeviceBufferView::new(&a_buf, PAD, &a_stored).unwrap();
    let b_view = DeviceBufferView::new(&b_buf, PAD, &b_stored).unwrap();
    let mut out = upload_flat(ops, &sentinel(ol.total));
    let token = DispatchFailureCell::new();
    let filled = if tracked {
        ops.gemm_fp32_strict_into_device_tracked(
            a_view,
            a_layout,
            b_view,
            b_layout,
            &mut out,
            ol.out_offset,
            bias,
            &token,
        )
    } else {
        ops.gemm_fp32_strict_into_device(
            a_view,
            a_layout,
            b_view,
            b_layout,
            &mut out,
            ol.out_offset,
            bias,
        )
    }
    .expect("device dW");
    assert_eq!(filled, want_filled, "{what}: bias_filled");
    assert_eq!(filled, with_bias, "{what}: bias_filled == with_bias");
    // download が内部同期する。範囲外（番兵）も含めて全体を bit 比較する。
    assert_bits_or_nan_class(&download(ops, &out), &want, &what);
    assert!(!token.is_set(), "{what}: token");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dw_tn_bit_identical_to_host() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    for shape in SHAPES {
        for kind in [Kind::Finite, Kind::Special] {
            for with_bias in [true, false] {
                for tracked in [false, true] {
                    check_dw(&ops, shape, false, with_bias, tracked, kind);
                }
            }
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dw_nt_bit_identical_to_host() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    for shape in [(64, 128, 256), (7, 13, 5), (63, 65, 67)] {
        for tracked in [false, true] {
            check_dw(&ops, shape, true, true, tracked, Kind::Finite);
        }
    }
}

/// d_input を device 版・ホスト版の両方で計算して bit 比較する。w は連結バッファ内の
/// offset 付き view（`[p, q] = [in, out]`）、b は g の格納 `[B, out]` を `Transposed`
/// （論理 `[out, B]`）または `RowMajor`（格納 `[out, B]`）で渡す。
fn check_dinput(
    ops: &MetalBackendOps,
    (bsz, inf, outf): (usize, usize, usize),
    b_transposed: bool,
    tracked: bool,
    kind: Kind,
) {
    let (p, q, r) = (inf, outf, bsz);
    let what = format!(
        "d_input B={bsz} in={inf} out={outf} b_transposed={b_transposed} tracked={tracked} {kind:?}"
    );
    let w_data = pattern(p * q, 3, Kind::Finite);
    let g_data = pattern(q * r, 4, kind);
    let (b_stored, b_layout) = if b_transposed {
        ([r, q], DeviceMatrixLayout::Transposed)
    } else {
        ([q, r], DeviceMatrixLayout::RowMajor)
    };

    let w_host_buf = upload_with_offset(ops, &w_data);
    let w_shape = [p, q];
    let w_view = DeviceBufferView::new(&w_host_buf, PAD, &w_shape).unwrap();
    let g_host = Tensor::new(g_data.clone(), &b_stored).unwrap();
    let g_host = if b_transposed {
        g_host.transpose(0, 1).unwrap()
    } else {
        g_host
    };
    let want = ops
        .gemm_resident_lhs(w_view, &g_host)
        .expect("host d_input");
    let want = want.as_slice().map_or_else(
        || want.contiguous().as_slice().unwrap().to_vec(),
        <[f32]>::to_vec,
    );

    let g_buf = upload_with_offset(ops, &g_data);
    let g_view = DeviceBufferView::new(&g_buf, PAD, &b_stored).unwrap();
    let token = DispatchFailureCell::new();
    let out = if tracked {
        ops.gemm_resident_lhs_device_tracked(w_view, g_view, b_layout, &token)
    } else {
        ops.gemm_resident_lhs_device(w_view, g_view, b_layout)
    }
    .expect("device d_input");
    assert_eq!(out.shape(), vec![p, r], "{what}: shape");
    assert_bits_or_nan_class(&download(ops, &out), &want, &what);
    assert!(!token.is_set(), "{what}: token");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dinput_bit_identical_to_host() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    for shape in SHAPES {
        for kind in [Kind::Finite, Kind::Special] {
            for tracked in [false, true] {
                check_dinput(&ops, shape, true, tracked, kind);
            }
        }
    }
    for shape in [(64, 128, 256), (7, 13, 5), (63, 65, 67)] {
        for tracked in [false, true] {
            check_dinput(&ops, shape, false, tracked, Kind::Finite);
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tracked_variants_issue_no_wait() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let (bsz, inf, outf) = (64usize, 128usize, 256usize);
    let (m, k, n) = (inf, bsz, outf);
    let x_data = pattern(m * k, 5, Kind::Finite);
    let g_data = pattern(k * n, 6, Kind::Finite);
    let w_data = pattern(m * n, 7, Kind::Finite);
    let x_buf = upload_with_offset(&ops, &x_data);
    let g_buf = upload_with_offset(&ops, &g_data);
    let w_buf = upload_with_offset(&ops, &w_data);
    let (xs, gs, ws) = ([k, m], [k, n], [m, n]);
    let x = DeviceBufferView::new(&x_buf, PAD, &xs).unwrap();
    let g = DeviceBufferView::new(&g_buf, PAD, &gs).unwrap();
    let w = DeviceBufferView::new(&w_buf, PAD, &ws).unwrap();
    let ol = out_layout(m, n);
    let mut out = upload_flat(&ops, &sentinel(ol.total));
    let bias = Some((ol.bias_offset, n));
    // open バッチを残さないため、一度同期を済ませる。
    let _ = download(&ops, &out);

    let token = DispatchFailureCell::new();
    let before = __diagnostic_batch_counters_snapshot().unwrap();
    let filled = ops
        .gemm_fp32_strict_into_device_tracked(
            x,
            DeviceMatrixLayout::Transposed,
            g,
            DeviceMatrixLayout::RowMajor,
            &mut out,
            ol.out_offset,
            bias,
            &token,
        )
        .unwrap();
    // d_input の出力確保（alloc_zeroed）を含めて wait が 0 であることも確認する。
    let din = ops
        .gemm_resident_lhs_device_tracked(w, g, DeviceMatrixLayout::Transposed, &token)
        .unwrap();
    let after = __diagnostic_batch_counters_snapshot().unwrap();
    assert!(filled);
    assert_eq!(
        after.encode_calls - before.encode_calls,
        2,
        "dW 1 + d_input 1"
    );
    assert_eq!(
        after.wait_until_completed - before.wait_until_completed,
        0,
        "tracked 版は wait を出さない"
    );

    // 検証用 download（内部同期）の後に値を確認する。
    let x_host = Tensor::new(x_data.clone(), &xs)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let g_host = Tensor::new(g_data.clone(), &gs).unwrap();
    let mut out_ref = upload_flat(&ops, &sentinel(ol.total));
    let t = DispatchFailureCell::new();
    ops.gemm_fp32_strict_into_with_bias_reduce_tracked(
        &x_host,
        &g_host,
        &mut out_ref,
        ol.out_offset,
        bias,
        &t,
    )
    .unwrap();
    assert_bits_or_nan_class(&download(&ops, &out), &download(&ops, &out_ref), "dW");
    let want = ops
        .gemm_resident_lhs(w, &g_host.transpose(0, 1).unwrap())
        .unwrap();
    assert_bits_or_nan_class(
        &download(&ops, &din),
        want.contiguous().as_slice().unwrap(),
        "d_input",
    );
    assert!(!token.is_set());

    // 比較: 非 tracked 版はそれぞれ wait +1。
    let b = __diagnostic_batch_counters_snapshot().unwrap();
    ops.gemm_fp32_strict_into_device(
        x,
        DeviceMatrixLayout::Transposed,
        g,
        DeviceMatrixLayout::RowMajor,
        &mut out,
        ol.out_offset,
        bias,
    )
    .unwrap();
    let mid = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(mid.wait_until_completed - b.wait_until_completed, 1);
    let _ = ops
        .gemm_resident_lhs_device(w, g, DeviceMatrixLayout::Transposed)
        .unwrap();
    let aft = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(aft.wait_until_completed - mid.wait_until_completed, 1);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dw_error_paths_are_typed_and_leave_out_untouched() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let (m, k, n) = (4usize, 6usize, 5usize);
    let a_buf = upload_flat(&ops, &pattern(m * k, 1, Kind::Finite));
    let b_buf = upload_flat(&ops, &pattern(k * n, 2, Kind::Finite));
    let (a_tn, b_tn) = ([k, m], [k, n]);
    let a = DeviceBufferView::new(&a_buf, 0, &a_tn).unwrap();
    let b = DeviceBufferView::new(&b_buf, 0, &b_tn).unwrap();
    let ol = out_layout(m, n);
    let init = sentinel(ol.total);
    let mut out = upload_flat(&ops, &init);
    let tn = (DeviceMatrixLayout::Transposed, DeviceMatrixLayout::RowMajor);

    let call =
        |a: DeviceBufferView<'_>,
         al,
         b: DeviceBufferView<'_>,
         bl,
         out: &mut DeviceBuffer<f32>,
         out_offset,
         bias| { ops.gemm_fp32_strict_into_device(a, al, b, bl, out, out_offset, bias) };

    // 範囲超過（weight）。
    let r = call(a, tn.0, b, tn.1, &mut out, ol.total - m * n + 1, None);
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    // 範囲超過（bias）。
    let r = call(
        a,
        tn.0,
        b,
        tn.1,
        &mut out,
        ol.out_offset,
        Some((ol.total - n + 1, n)),
    );
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    // weight と bias の重なり。
    let r = call(
        a,
        tn.0,
        b,
        tn.1,
        &mut out,
        ol.out_offset,
        Some((ol.out_offset, n)),
    );
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    // bn != n。
    let r = call(
        a,
        tn.0,
        b,
        tn.1,
        &mut out,
        ol.out_offset,
        Some((ol.bias_offset, n + 1)),
    );
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    // 内側次元の不一致（a の k を変える）。
    let a_bad_buf = upload_flat(&ops, &pattern((k + 1) * m, 1, Kind::Finite));
    let a_bad_shape = [k + 1, m];
    let a_bad = DeviceBufferView::new(&a_bad_buf, 0, &a_bad_shape).unwrap();
    let r = call(a_bad, tn.0, b, tn.1, &mut out, ol.out_offset, None);
    assert!(
        matches!(
            r,
            Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch { .. }
            ))
        ),
        "{r:?}"
    );
    // rank 1 の view。
    let flat = [m * k];
    let a_rank1 = DeviceBufferView::new(&a_buf, 0, &flat).unwrap();
    let r = call(a_rank1, tn.0, b, tn.1, &mut out, ol.out_offset, None);
    assert!(
        matches!(
            r,
            Err(BackendError::ShapeMismatch(ShapeError::RankMismatch { .. }))
        ),
        "{r:?}"
    );
    // NN（RowMajor / RowMajor）と TT は Unsupported。
    let a_nn_shape = [m, k];
    let a_nn = DeviceBufferView::new(&a_buf, 0, &a_nn_shape).unwrap();
    let r = call(
        a_nn,
        DeviceMatrixLayout::RowMajor,
        b,
        DeviceMatrixLayout::RowMajor,
        &mut out,
        ol.out_offset,
        None,
    );
    assert!(matches!(r, Err(BackendError::Unsupported(_))), "NN {r:?}");
    let b_tt_shape = [n, k];
    let b_tt = DeviceBufferView::new(&b_buf, 0, &b_tt_shape).unwrap();
    let r = call(
        a,
        DeviceMatrixLayout::Transposed,
        b_tt,
        DeviceMatrixLayout::Transposed,
        &mut out,
        ol.out_offset,
        None,
    );
    assert!(matches!(r, Err(BackendError::Unsupported(_))), "TT {r:?}");
    // 0 次元は Unsupported（wait なしで書けないため）。
    let z_buf = upload_flat(&ops, &[1.0]);
    let (z_a, z_b) = ([k, 0], [k, n]);
    let a_zero = DeviceBufferView::new(&z_buf, 0, &z_a).unwrap();
    let r = call(a_zero, tn.0, b, tn.1, &mut out, 0, None);
    assert!(
        matches!(r, Err(BackendError::Unsupported(_))),
        "0-dim {r:?}"
    );
    let _ = z_b;
    // CPU デバイスの buffer は DeviceMismatch。
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_buf = cpu
        .memory_ops()
        .expect("cpu memory_ops")
        .upload(&Tensor::new(pattern(m * k, 1, Kind::Finite), &[m * k]).unwrap())
        .expect("cpu upload");
    let a_cpu = DeviceBufferView::new(&cpu_buf, 0, &a_tn).unwrap();
    let r = call(a_cpu, tn.0, b, tn.1, &mut out, ol.out_offset, None);
    assert!(matches!(r, Err(BackendError::DeviceMismatch)), "{r:?}");

    // いずれの失敗でも out は書き換わっていない。
    assert_bits_or_nan_class(&download(&ops, &out), &init, "out untouched");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dinput_error_paths_and_zero_dims() {
    let _g = serialize();
    let ops = MetalBackendOps::new();
    let (p, q, r) = (4usize, 3usize, 5usize);
    let w_buf = upload_flat(&ops, &pattern(p * q, 1, Kind::Finite));
    let g_buf = upload_flat(&ops, &pattern(q * r, 2, Kind::Finite));
    let (ws, gs) = ([p, q], [r, q]);
    let w = DeviceBufferView::new(&w_buf, 0, &ws).unwrap();
    let g = DeviceBufferView::new(&g_buf, 0, &gs).unwrap();

    // 内側次元の不一致（b の論理行数 != w の列数）。
    let bad_shape = [q, r];
    let g_bad = DeviceBufferView::new(&g_buf, 0, &bad_shape).unwrap();
    let res = ops.gemm_resident_lhs_device(w, g_bad, DeviceMatrixLayout::Transposed);
    assert!(
        matches!(
            res,
            Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch { .. }
            ))
        ),
        "{res:?}"
    );
    // rank 1。
    let flat = [p * q];
    let w_flat = DeviceBufferView::new(&w_buf, 0, &flat).unwrap();
    let res = ops.gemm_resident_lhs_device(w_flat, g, DeviceMatrixLayout::Transposed);
    assert!(
        matches!(
            res,
            Err(BackendError::ShapeMismatch(ShapeError::RankMismatch { .. }))
        ),
        "{res:?}"
    );
    // CPU デバイス。
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_buf = cpu
        .memory_ops()
        .expect("cpu memory_ops")
        .upload(&Tensor::new(pattern(p * q, 1, Kind::Finite), &[p * q]).unwrap())
        .expect("cpu upload");
    let w_cpu = DeviceBufferView::new(&cpu_buf, 0, &ws).unwrap();
    let res = ops.gemm_resident_lhs_device(w_cpu, g, DeviceMatrixLayout::Transposed);
    assert!(matches!(res, Err(BackendError::DeviceMismatch)), "{res:?}");

    // 0 次元: dispatch も wait も出さない。
    let _ = download(&ops, &w_buf);
    let before = __diagnostic_batch_counters_snapshot().unwrap();
    // p == 0: 空。
    let w0_shape = [0, q];
    let w0 = DeviceBufferView::new(&w_buf, 0, &w0_shape).unwrap();
    let out = ops
        .gemm_resident_lhs_device(w0, g, DeviceMatrixLayout::Transposed)
        .unwrap();
    assert_eq!(out.shape(), vec![0, r]);
    // r == 0: 空。
    let g0_shape = [0, q];
    let g0 = DeviceBufferView::new(&g_buf, 0, &g0_shape).unwrap();
    let out = ops
        .gemm_resident_lhs_device(w, g0, DeviceMatrixLayout::Transposed)
        .unwrap();
    assert_eq!(out.shape(), vec![p, 0]);
    // q == 0: 全 0 の [p, r]。
    let wq_shape = [p, 0];
    let wq = DeviceBufferView::new(&w_buf, 0, &wq_shape).unwrap();
    let gq_shape = [r, 0];
    let gq = DeviceBufferView::new(&g_buf, 0, &gq_shape).unwrap();
    let out = ops
        .gemm_resident_lhs_device_tracked(
            wq,
            gq,
            DeviceMatrixLayout::Transposed,
            &DispatchFailureCell::new(),
        )
        .unwrap();
    let after = __diagnostic_batch_counters_snapshot().unwrap();
    assert_eq!(after.encode_calls, before.encode_calls, "dispatch なし");
    assert_eq!(
        after.wait_until_completed, before.wait_until_completed,
        "wait なし"
    );
    assert_eq!(out.shape(), vec![p, r]);
    assert!(download(&ops, &out).iter().all(|v| v.to_bits() == 0));
}
