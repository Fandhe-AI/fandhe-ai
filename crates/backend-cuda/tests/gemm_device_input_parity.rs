//! イシュー #3223: `CudaBackendOps::gemm_fp32_strict_into_device`（dW。bias 縮約つき）・
//! `gemm_resident_lhs_device`（d_input）および `_tracked` 版の実機 parity と race 検証。
//! CUDA 実機（DGX Spark GB10 等）でのみ実行する（`#[ignore]`）。
//!
//! 参照はホスト入力版 `gemm_fp32_strict_into_with_bias_reduce_tracked`（dW）・
//! `gemm_resident_lhs`（d_input）。両者は同じ GEMM カーネル選択
//! （`select_tiled_f32_kernel`）・同じ転置カーネルを使うため、判定は tolerance を
//! 使わず bit 一致（NaN はクラス一致）。tolerance・FMA 契約は変更しない。
//!
//! 形状は (B, in, out)。dW は m = in・k = B・n = out、d_input は p = in・q = out・r = B。
//!
//! # offset の 3 種（cp.async 整列。#3223 計画 §2.2）
//! 直接 GEMM へ渡すオペランドの view offset を 0・4 の倍数（そのまま渡す経路）・3
//! （16 バイト境界を外れ staging でコピーする経路）で試す。
//!
//! # race 検証
//! 同期を挟まずに dW／d_input を形状「小 → 大 → 小 → 大」で交互に積み（永続
//! ワークスペースの拡張と再利用の両方を通す）、入力バッファの一部を呼び出し直後に
//! drop し（`free_async` のストリーム順序を確認）、最後にまとめて download して各
//! 結果をホスト版と bit 比較する。2 スレッドが同じ ordinal から並行に積む版もある
//! （Mutex 下でのワークスペース共有）。
//!
//! 正式な実機記録（RULE.txt・5 run・`.rev-stamp`）は #3224。本ファイルの実機実行結果は
//! 本 PR 時点では未実測。
//!
//! 実行: `cargo test -p fandhe-ai-backend-cuda --release --test gemm_device_input_parity -- --ignored --nocapture --test-threads=1`

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DeviceBufferView, DeviceMatrixLayout, DispatchFailureCell, Tensor,
};

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

fn ops() -> CudaBackendOps {
    let device = CudaDevice::new(0).expect("CUDA device 0 must be available");
    CudaBackendOps::new(device.ordinal())
}

fn upload_flat(ops: &CudaBackendOps, data: &[f32]) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("memory_ops")
        .upload(&Tensor::new(data.to_vec(), &[data.len()]).unwrap())
        .expect("upload")
}

fn download(ops: &CudaBackendOps, buf: &DeviceBuffer<f32>) -> Vec<f32> {
    ops.memory_ops()
        .unwrap()
        .download(buf)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec()
}

/// 先頭に `pad` 要素の番兵を置いた連結バッファへ `data` を upload する
/// （view の offset を `pad` にする）。
fn upload_with_offset(ops: &CudaBackendOps, data: &[f32], pad: usize) -> DeviceBuffer<f32> {
    let mut flat = vec![9.0f32; pad];
    flat.extend_from_slice(data);
    flat.extend_from_slice(&[9.0, 9.0]);
    upload_flat(ops, &flat)
}

/// 直接渡すオペランドの offset 組（a の pad, b の pad）。0・4 の倍数・3（staging）と混在。
const PADS: [(usize, usize); 5] = [(0, 0), (4, 4), (3, 3), (0, 3), (3, 0)];

/// dW の out バッファ内の weight・bias の書き込み位置。
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

/// (B, in, out) 形状: MLP 形状 + 端数形状 + `k == 1`（bias の -0.0 直接コピー経路）。
const SHAPES: [(usize, usize, usize); 8] = [
    (64, 128, 256),
    (64, 256, 10),
    (64, 784, 128),
    (7, 13, 5),
    (63, 65, 67),
    (64, 257, 33),
    (3, 2, 1),
    (1, 33, 17),
];

/// dW のホスト参照（同じ入力・同じ out 初期値）を計算して out 全体を返す。
/// `nt == false` は TN（a = x の格納 `[k, m]`・b = g の格納 `[k, n]`）、`nt == true` は
/// NT（a の格納 `[m, k]`・b の格納 `[n, k]`）。
fn host_dw(
    ops: &CudaBackendOps,
    (m, k, n): (usize, usize, usize),
    nt: bool,
    a_data: &[f32],
    b_data: &[f32],
    bias: Option<(usize, usize)>,
) -> (Vec<f32>, bool) {
    let ol = out_layout(m, n);
    let (a_stored, b_stored) = if nt {
        ([m, k], [n, k])
    } else {
        ([k, m], [k, n])
    };
    let a_host = Tensor::new(a_data.to_vec(), &a_stored).unwrap();
    let b_host = Tensor::new(b_data.to_vec(), &b_stored).unwrap();
    let (a_host, b_host) = if nt {
        (a_host, b_host.transpose_2d().unwrap())
    } else {
        (a_host.transpose_2d().unwrap(), b_host)
    };
    let mut out_ref = upload_flat(ops, &sentinel(ol.total));
    let token = DispatchFailureCell::new();
    let filled = ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            &a_host,
            &b_host,
            &mut out_ref,
            ol.out_offset,
            bias,
            &token,
        )
        .expect("host dW");
    (download(ops, &out_ref), filled)
}

#[allow(clippy::too_many_arguments)]
fn check_dw(
    ops: &CudaBackendOps,
    (bsz, inf, outf): (usize, usize, usize),
    nt: bool,
    with_bias: bool,
    tracked: bool,
    kind: Kind,
    (a_pad, b_pad): (usize, usize),
) {
    let (m, k, n) = (inf, bsz, outf);
    let what = format!(
        "dW B={bsz} in={inf} out={outf} nt={nt} bias={with_bias} tracked={tracked} {kind:?} \
         pads=({a_pad},{b_pad})"
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
    let (want, want_filled) = host_dw(ops, (m, k, n), nt, &a_data, &b_data, bias);

    let a_buf = upload_with_offset(ops, &a_data, a_pad);
    let b_buf = upload_with_offset(ops, &b_data, b_pad);
    let a_view = DeviceBufferView::new(&a_buf, a_pad, &a_stored).unwrap();
    let b_view = DeviceBufferView::new(&b_buf, b_pad, &b_stored).unwrap();
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
    // download が同期する。範囲外（番兵）も含めて全体を bit 比較する。
    assert_bits_or_nan_class(&download(ops, &out), &want, &what);
    assert!(!token.is_set(), "{what}: token");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn dw_tn_bit_identical_to_host() {
    let ops = ops();
    for shape in SHAPES {
        for kind in [Kind::Finite, Kind::Special] {
            for with_bias in [true, false] {
                for tracked in [false, true] {
                    for pads in PADS {
                        check_dw(&ops, shape, false, with_bias, tracked, kind, pads);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn dw_nt_bit_identical_to_host() {
    let ops = ops();
    for shape in SHAPES {
        for with_bias in [true, false] {
            for tracked in [false, true] {
                for pads in PADS {
                    check_dw(&ops, shape, true, with_bias, tracked, Kind::Finite, pads);
                }
            }
        }
    }
    for pads in PADS {
        check_dw(&ops, (63, 65, 67), true, true, false, Kind::Special, pads);
    }
}

/// d_input のホスト参照。`b_transposed` は g の格納 `[r, q]` を転置（論理 `[q, r]`）、
/// それ以外は格納 `[q, r]` を行優先で渡す。
fn host_dinput(
    ops: &CudaBackendOps,
    w_view: DeviceBufferView<'_>,
    (p, q, r): (usize, usize, usize),
    b_transposed: bool,
    g_data: &[f32],
) -> Vec<f32> {
    let _ = p;
    let g_host = if b_transposed {
        Tensor::new(g_data.to_vec(), &[r, q])
            .unwrap()
            .transpose_2d()
            .unwrap()
    } else {
        Tensor::new(g_data.to_vec(), &[q, r]).unwrap()
    };
    let want = ops
        .gemm_resident_lhs(w_view, &g_host)
        .expect("host d_input");
    want.contiguous().as_slice().unwrap().to_vec()
}

fn check_dinput(
    ops: &CudaBackendOps,
    (bsz, inf, outf): (usize, usize, usize),
    b_transposed: bool,
    tracked: bool,
    kind: Kind,
    (w_pad, g_pad): (usize, usize),
) {
    let (p, q, r) = (inf, outf, bsz);
    let what = format!(
        "d_input B={bsz} in={inf} out={outf} b_transposed={b_transposed} tracked={tracked} \
         {kind:?} pads=({w_pad},{g_pad})"
    );
    let w_data = pattern(p * q, 3, Kind::Finite);
    let g_data = pattern(q * r, 4, kind);
    let (b_stored, b_layout) = if b_transposed {
        ([r, q], DeviceMatrixLayout::Transposed)
    } else {
        ([q, r], DeviceMatrixLayout::RowMajor)
    };
    let w_buf = upload_with_offset(ops, &w_data, w_pad);
    let w_shape = [p, q];
    let w_view = DeviceBufferView::new(&w_buf, w_pad, &w_shape).unwrap();
    let want = host_dinput(ops, w_view, (p, q, r), b_transposed, &g_data);

    let g_buf = upload_with_offset(ops, &g_data, g_pad);
    let g_view = DeviceBufferView::new(&g_buf, g_pad, &b_stored).unwrap();
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
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn dinput_bit_identical_to_host() {
    let ops = ops();
    for shape in SHAPES {
        for kind in [Kind::Finite, Kind::Special] {
            for tracked in [false, true] {
                for pads in PADS {
                    check_dinput(&ops, shape, true, tracked, kind, pads);
                }
            }
        }
    }
    for shape in SHAPES {
        for tracked in [false, true] {
            for pads in PADS {
                check_dinput(&ops, shape, false, tracked, Kind::Finite, pads);
            }
        }
    }
}

/// 同期を挟まずに dW・d_input を積み、最後にまとめて host と比較する 1 本分の仕事。
/// 形状「小 → 大 → 小 → 大」で交互に積む（永続ワークスペースの拡張と再利用の両方を通す）。
fn enqueue_chain_and_verify(ops: &CudaBackendOps, salt: usize) {
    // (B, in, out)。小・大・小・大。
    let chain = [(7, 13, 5), (64, 256, 128), (3, 2, 1), (64, 784, 128)];
    let mut expected_dw: Vec<Vec<f32>> = Vec::new();
    let mut expected_din: Vec<Vec<f32>> = Vec::new();
    // 参照はすべて先に計算する（device 側の積み込み中にホスト版の同期を挟まない）。
    let mut inputs = Vec::new();
    for (idx, &(bsz, inf, outf)) in chain.iter().enumerate() {
        let (m, k, n) = (inf, bsz, outf);
        let x = pattern(m * k, 10 + idx + salt, Kind::Finite);
        let g = pattern(k * n, 20 + idx + salt, Kind::Finite);
        let w = pattern(m * n, 30 + idx + salt, Kind::Finite);
        let ol = out_layout(m, n);
        let bias = Some((ol.bias_offset, n));
        let (dw, _) = host_dw(ops, (m, k, n), false, &x, &g, bias);
        expected_dw.push(dw);
        let w_buf = upload_with_offset(ops, &w, 3);
        let w_shape = [m, n];
        let w_view = DeviceBufferView::new(&w_buf, 3, &w_shape).unwrap();
        expected_din.push(host_dinput(ops, w_view, (m, n, k), true, &g));
        inputs.push((x, g, w));
    }

    let mut outs: Vec<DeviceBuffer<f32>> = Vec::new();
    let mut dins: Vec<DeviceBuffer<f32>> = Vec::new();
    for (idx, &(bsz, inf, outf)) in chain.iter().enumerate() {
        let (m, k, n) = (inf, bsz, outf);
        let (x, g, w) = &inputs[idx];
        let ol = out_layout(m, n);
        let bias = Some((ol.bias_offset, n));
        let (xs, gs, ws) = ([k, m], [k, n], [m, n]);
        let x_buf = upload_with_offset(ops, x, 3);
        let g_buf = upload_with_offset(ops, g, 4);
        let w_buf = upload_with_offset(ops, w, 3);
        let mut out = upload_flat(ops, &sentinel(ol.total));
        {
            let xv = DeviceBufferView::new(&x_buf, 3, &xs).unwrap();
            let gv = DeviceBufferView::new(&g_buf, 4, &gs).unwrap();
            ops.gemm_fp32_strict_into_device_tracked(
                xv,
                DeviceMatrixLayout::Transposed,
                gv,
                DeviceMatrixLayout::RowMajor,
                &mut out,
                ol.out_offset,
                bias,
                &DispatchFailureCell::new(),
            )
            .expect("chain dW");
        }
        // d_input は g の転置格納 `[k, n]` を Transposed（論理 `[n, k]`）で読む。
        {
            let wv = DeviceBufferView::new(&w_buf, 3, &ws).unwrap();
            let gv = DeviceBufferView::new(&g_buf, 4, &gs).unwrap();
            let din = ops
                .gemm_resident_lhs_device_tracked(
                    wv,
                    gv,
                    DeviceMatrixLayout::Transposed,
                    &DispatchFailureCell::new(),
                )
                .expect("chain d_input");
            dins.push(din);
        }
        outs.push(out);
        // 入力 DeviceBuffer の一部を呼び出し直後に drop する（free_async が
        // ストリーム順序で効いていること＝起動済みカーネルの読み出しを壊さないこと）。
        drop(x_buf);
        if idx % 2 == 0 {
            drop(g_buf);
        }
        drop(w_buf);
    }

    // 最後にまとめて download（ここで初めて同期する）。
    for (idx, out) in outs.iter().enumerate() {
        assert_bits_or_nan_class(
            &download(ops, out),
            &expected_dw[idx],
            &format!("chain dW #{idx}"),
        );
    }
    for (idx, din) in dins.iter().enumerate() {
        // d_input の参照は w の `[m, n]`・g を `[n, k]` の転置として読む形（上と同じ）。
        assert_bits_or_nan_class(
            &download(ops, din),
            &expected_din[idx],
            &format!("chain d_input #{idx}"),
        );
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn race_no_sync_chain_matches_host() {
    let ops = ops();
    for round in 0..3 {
        enqueue_chain_and_verify(&ops, round);
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn race_two_threads_share_workspace_under_mutex() {
    let ops = ops();
    // 同一 ordinal・同一 CudaGemm（ContextKey ごとの共有インスタンス）を 2 スレッドが
    // 並行に叩く。ワークスペースは Mutex 下で共有され、結果はホスト版と bit 一致する。
    std::thread::scope(|s| {
        let h1 = s.spawn(|| enqueue_chain_and_verify(&ops, 100));
        let h2 = s.spawn(|| enqueue_chain_and_verify(&ops, 200));
        h1.join().expect("thread 1");
        h2.join().expect("thread 2");
    });
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn dw_error_paths_are_typed_and_leave_out_untouched() {
    let ops = ops();
    let (m, k, n) = (4usize, 6usize, 5usize);
    let a_buf = upload_flat(&ops, &pattern(m * k, 1, Kind::Finite));
    let b_buf = upload_flat(&ops, &pattern(k * n, 2, Kind::Finite));
    let (a_tn, b_tn) = ([k, m], [k, n]);
    let a = DeviceBufferView::new(&a_buf, 0, &a_tn).unwrap();
    let b = DeviceBufferView::new(&b_buf, 0, &b_tn).unwrap();
    let ol = out_layout(m, n);
    let init = sentinel(ol.total);
    let mut out = upload_flat(&ops, &init);
    let (tr, rm) = (DeviceMatrixLayout::Transposed, DeviceMatrixLayout::RowMajor);

    // NN（RowMajor / RowMajor）と TT は Unsupported。
    let a_nn_shape = [m, k];
    let a_nn = DeviceBufferView::new(&a_buf, 0, &a_nn_shape).unwrap();
    let r = ops.gemm_fp32_strict_into_device(a_nn, rm, b, rm, &mut out, ol.out_offset, None);
    assert!(matches!(r, Err(BackendError::Unsupported(_))), "NN {r:?}");
    let b_tt_shape = [n, k];
    let b_tt = DeviceBufferView::new(&b_buf, 0, &b_tt_shape).unwrap();
    let r = ops.gemm_fp32_strict_into_device(a, tr, b_tt, tr, &mut out, ol.out_offset, None);
    assert!(matches!(r, Err(BackendError::Unsupported(_))), "TT {r:?}");
    // 0 次元は Unsupported。
    let z_buf = upload_flat(&ops, &[1.0]);
    let z_a = [k, 0];
    let a_zero = DeviceBufferView::new(&z_buf, 0, &z_a).unwrap();
    let r = ops.gemm_fp32_strict_into_device(a_zero, tr, b, rm, &mut out, 0, None);
    assert!(
        matches!(r, Err(BackendError::Unsupported(_))),
        "0-dim {r:?}"
    );
    // 範囲超過（weight）・bias の重なり。
    let r = ops.gemm_fp32_strict_into_device(a, tr, b, rm, &mut out, ol.total - m * n + 1, None);
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");
    let r = ops.gemm_fp32_strict_into_device(
        a,
        tr,
        b,
        rm,
        &mut out,
        ol.out_offset,
        Some((ol.out_offset, n)),
    );
    assert!(matches!(r, Err(BackendError::InvalidArgument(_))), "{r:?}");

    // いずれの失敗でも out は書き換わっていない。
    assert_bits_or_nan_class(&download(&ops, &out), &init, "out untouched");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn dinput_zero_dims_follow_host_contract() {
    let ops = ops();
    let (p, q, r) = (4usize, 3usize, 5usize);
    let w_buf = upload_flat(&ops, &pattern(p * q, 1, Kind::Finite));
    let g_buf = upload_flat(&ops, &pattern(q * r, 2, Kind::Finite));
    let (ws, gs) = ([p, q], [r, q]);
    let w = DeviceBufferView::new(&w_buf, 0, &ws).unwrap();
    let g = DeviceBufferView::new(&g_buf, 0, &gs).unwrap();
    let tr = DeviceMatrixLayout::Transposed;
    // 通常形状の健全性。
    let ok = ops.gemm_resident_lhs_device(w, g, tr).unwrap();
    assert_eq!(ok.shape(), vec![p, r]);
    // p == 0: 空。
    let w0_shape = [0, q];
    let w0 = DeviceBufferView::new(&w_buf, 0, &w0_shape).unwrap();
    let out = ops.gemm_resident_lhs_device(w0, g, tr).unwrap();
    assert_eq!(out.shape(), vec![0, r]);
    // r == 0: 空。
    let g0_shape = [0, q];
    let g0 = DeviceBufferView::new(&g_buf, 0, &g0_shape).unwrap();
    let out = ops.gemm_resident_lhs_device(w, g0, tr).unwrap();
    assert_eq!(out.shape(), vec![p, 0]);
    // q == 0: 全 0 の [p, r]。
    let wq_shape = [p, 0];
    let wq = DeviceBufferView::new(&w_buf, 0, &wq_shape).unwrap();
    let gq_shape = [r, 0];
    let gq = DeviceBufferView::new(&g_buf, 0, &gq_shape).unwrap();
    let out = ops.gemm_resident_lhs_device(wq, gq, tr).unwrap();
    assert_eq!(out.shape(), vec![p, r]);
    assert!(download(&ops, &out).iter().all(|v| v.to_bits() == 0));
}
