//! `CpuBackendOps::linear_forward_device` の受け入れ基準対応テスト
//! （イシュー #1028・`docs/inference-forward-fixed-cost-design.md`）。
//!
//! `fandhe_ai_tensor_core::BackendOps` に非破壊追加した
//! `linear_forward_device`（デフォルトは `BackendError::Unsupported`。
//! `crates/tensor-core/src/backend_ops.rs`）の CPU 実装が、旧経路
//! （`Sequential::predict` 等が `tape.ops()` 経由で呼ぶ非融合 `gemm` →
//! `add`（bias 行方向複製）→ `relu` の 3 段合成。`fandhe_ai_autodiff::
//! nn::linear::LinearVars::forward` と同型）と **bit 完全一致**すること
//! を検証する（§3.3 (b) の bit-exactness 契約）。`a`／`w`／`bias`／
//! 戻り値をいずれもデバイス常駐バッファへ upload/download してから
//! 呼ぶ点のみが異なり、数式は同一のため REQ-2 統一複合判定ではなく
//! bit 完全一致で比較する（`gemm_resident_parity.rs` と同じ方針）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, CpuMemory};
use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::{Activation, BackendError, BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 旧経路（`gemm` → `add` → `act`。`LinearVars::forward` + 活性化層と
/// 同型の非融合合成）をホスト常駐 `Tensor` で計算する参照実装。
fn old_path(
    ops: &CpuBackendOps,
    a: &Tensor<f32>,
    w: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Tensor<f32> {
    let mut y = ops.gemm(a, w).unwrap();
    if let Some(bias) = bias {
        y = ops.add(&y, bias).unwrap();
    }
    y = match act {
        Activation::None => y,
        Activation::Relu => ops.relu(&y).unwrap(),
        _ => unreachable!("test only exercises None/Relu"),
    };
    y
}

/// `linear_forward_device` が旧経路（`gemm` → `add` → `act`）と
/// bit 完全一致することを、MR/NR/MC/KC/NC 境界を跨ぐ複数形状・
/// bias 有無・activation（None/Relu）の組み合わせで確認する。
#[test]
fn linear_forward_device_matches_old_path_across_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();

    for &(m, k, n) in &[(1, 1, 1), (4, 8, 4), (37, 65, 33), (128, 129, 96)] {
        for has_bias in [false, true] {
            for act in [Activation::None, Activation::Relu] {
                // ReLU が層の出力符号に対して恒等に近くなる境界ケースを
                // 減らすため、bias 込みの分布を中心 0 付近からわずかに
                // ずらす（`Xorshift64Star::fill_vec` は [0,1) 一様分布の
                // ため、そのままでは gemm 出力が正に偏り ReLU が全恒等に
                // なりがちで parity 検証として弱くなる）。
                let a_raw = random_matrix(0x1000 + m as u64, m * k);
                let a_data: Vec<f32> = a_raw.iter().map(|v| v - 0.5).collect();
                let w_raw = random_matrix(0x2000 + n as u64, k * n);
                let w_data: Vec<f32> = w_raw.iter().map(|v| v - 0.5).collect();
                let a = tensor(a_data, &[m, k]);
                let w = tensor(w_data, &[k, n]);
                let bias = has_bias.then(|| {
                    let raw = random_matrix(0x3000 + n as u64, n);
                    tensor(raw.iter().map(|v| v - 0.5).collect(), &[n])
                });

                let expected = old_path(&ops, &a, &w, bias.as_ref(), act);

                let a_dev = mem.upload(&a).unwrap();
                let w_dev = mem.upload(&w).unwrap();
                let w_shape = [k, n];
                let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
                let bias_dev = bias.as_ref().map(|b| mem.upload(b).unwrap());
                let bias_shape = [n];
                let bias_view = bias_dev
                    .as_ref()
                    .map(|buf| DeviceBufferView::new(buf, 0, &bias_shape).unwrap());

                let actual_dev = ops
                    .linear_forward_device(&a_dev, w_view, bias_view, act)
                    .unwrap();
                let actual = mem.download(&actual_dev).unwrap();

                assert_eq!(
                    actual.shape(),
                    expected.shape(),
                    "m={m} k={k} n={n} has_bias={has_bias} act={act:?}"
                );
                let a_slice = actual.contiguous();
                let e_slice = expected.contiguous();
                assert_eq!(
                    a_slice.as_slice().unwrap(),
                    e_slice.as_slice().unwrap(),
                    "linear_forward_device は旧経路（gemm→add→act）と bit 完全一致するはず \
                     （m={m} k={k} n={n} has_bias={has_bias} act={act:?}）"
                );
            }
        }
    }
}

/// `w`（デバイス常駐）の shape が `a` の列数と噛み合わない場合は
/// カーネル本体へ触れる前に `ShapeMismatch` を返す（REQ-8・OWASP A03）。
#[test]
fn linear_forward_device_rejects_incompatible_shapes() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    let w = tensor(vec![1.0, 2.0], &[2, 1]);
    let a_dev = mem.upload(&a).unwrap();
    let w_dev = mem.upload(&w).unwrap();
    let w_shape = [2, 1];
    let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();

    let err = ops
        .linear_forward_device(&a_dev, w_view, None, Activation::None)
        .unwrap_err();

    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `bias` の shape が `[n]` 厳密一致でない場合も同様に拒否する。
#[test]
fn linear_forward_device_rejects_bias_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a = tensor(vec![1.0, 2.0], &[1, 2]);
    let w = tensor(vec![1.0, 2.0], &[2, 1]);
    let bias = tensor(vec![1.0, 2.0], &[2]); // n=1 のはずが [2]

    let a_dev = mem.upload(&a).unwrap();
    let w_dev = mem.upload(&w).unwrap();
    let w_shape = [2, 1];
    let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
    let bias_dev = mem.upload(&bias).unwrap();
    let bias_shape = [2];
    let bias_view = DeviceBufferView::new(&bias_dev, 0, &bias_shape).unwrap();

    let err = ops
        .linear_forward_device(&a_dev, w_view, Some(bias_view), Activation::None)
        .unwrap_err();

    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
// ---------------------------------------------------------------------
// イシュー #3196: 融合 epilogue 経路への切替に伴う bit 一致テスト。
// 比較はすべて `f32::to_bits()` の完全一致（NaN != NaN と -0.0 == +0.0 の
// 見落としを避けるため `assert_eq!` で `&[f32]` を直接比べない）。
// ---------------------------------------------------------------------

use fandhe_ai_backend_cpu::{gemm_blis_bias_act_parallel, gemm_blis_parallel};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 非融合 3 段合成（`gemm_blis_parallel` → 行方向 bias 加算 → `max(x, 0.0)`）。
/// 切替前の `linear_forward_device` と同一の参照実装。
fn unfused(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    (m, k, n): (usize, usize, usize),
    act: Activation,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    gemm_blis_parallel(a, w, &mut out, m, n, k).unwrap();
    if let Some(b) = bias {
        for row in out.chunks_mut(n.max(1)).take(m) {
            for (c, bv) in row.iter_mut().zip(b.iter()) {
                *c += bv;
            }
        }
    }
    if act == Activation::Relu {
        for v in out.iter_mut() {
            *v = v.max(0.0);
        }
    }
    out
}

fn fused(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    (m, k, n): (usize, usize, usize),
    act: Activation,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    gemm_blis_bias_act_parallel(a, w, &mut out, m, n, k, bias, act).unwrap();
    out
}

fn centered(seed: u64, len: usize) -> Vec<f32> {
    random_matrix(seed, len).iter().map(|v| v - 0.5).collect()
}

fn shapes() -> Vec<(usize, usize, usize)> {
    let mut s = vec![
        (64, 784, 256),
        (64, 256, 10),
        (1, 17, 19),
        (0, 4, 3),
        (3, 4, 0),
        (4, 1, 5),
    ];
    for m in [5, 15, 17] {
        for n in [1, 10, 17, 19] {
            s.push((m, 9, n));
        }
    }
    for k in [255, 256, 257] {
        s.push((7, k, 11));
    }
    s.push((9, 33, 600));
    s
}

fn assert_kernel_parity_all_shapes() {
    for shape in shapes() {
        let (m, k, n) = shape;
        let a = centered(0x11 + m as u64, m * k);
        let w = centered(0x22 + n as u64, k * n);
        let b = centered(0x33 + n as u64, n);
        for bias in [None, Some(&b[..])] {
            for act in [Activation::None, Activation::Relu] {
                assert_eq!(
                    bits(&fused(&a, &w, bias, shape, act)),
                    bits(&unfused(&a, &w, bias, shape, act)),
                    "融合と非融合が bit 一致するはず（shape={shape:?} bias={} act={act:?}）",
                    bias.is_some()
                );
            }
        }
    }
}

#[test]
fn fused_epilogue_matches_unfused_composition_bit_exact() {
    assert_kernel_parity_all_shapes();
}

#[test]
fn fused_epilogue_matches_unfused_across_thread_pools() {
    for threads in [1usize, 2, 0] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(assert_kernel_parity_all_shapes);
    }
}

#[test]
fn fused_epilogue_matches_unfused_with_nan_inf_and_negative_zero() {
    let shape = (6, 5, 7);
    let (m, k, n) = shape;
    let mut a = centered(0x41, m * k);
    let mut w = centered(0x42, k * n);
    let mut b = centered(0x43, n);
    a[0] = f32::NAN;
    a[7] = f32::INFINITY;
    w[3] = f32::NEG_INFINITY;
    w[11] = f32::INFINITY;
    b[1] = f32::INFINITY;
    b[2] = f32::NEG_INFINITY;
    b[4] = -0.0;
    for bias in [None, Some(&b[..])] {
        for act in [Activation::None, Activation::Relu] {
            assert_eq!(
                bits(&fused(&a, &w, bias, shape, act)),
                bits(&unfused(&a, &w, bias, shape, act)),
                "NaN/inf 入力でも bit 一致するはず（bias={} act={act:?}）",
                bias.is_some()
            );
        }
    }
    // GEMM 出力が 0 になる行と bias = -0.0 の組（ReLU を通る -0.0）。
    let (m, k, n) = (2, 3, 2);
    let a = vec![0.0f32; m * k];
    let w = vec![1.0f32; k * n];
    let b = vec![-0.0f32, 0.0];
    for act in [Activation::None, Activation::Relu] {
        assert_eq!(
            bits(&fused(&a, &w, Some(&b), (m, k, n), act)),
            bits(&unfused(&a, &w, Some(&b), (m, k, n), act)),
            "-0.0 の符号まで一致するはず（act={act:?}）"
        );
    }
}

fn run_linear_forward_device(
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    (m, k, n): (usize, usize, usize),
    act: Activation,
) -> Vec<f32> {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let a_dev = mem.upload(&tensor(a.to_vec(), &[m, k])).unwrap();
    let w_dev = mem.upload(&tensor(w.to_vec(), &[k, n])).unwrap();
    let w_shape = [k, n];
    let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
    let bias_dev = bias.map(|b| mem.upload(&tensor(b.to_vec(), &[n])).unwrap());
    let bias_shape = [n];
    let bias_view = bias_dev
        .as_ref()
        .map(|buf| DeviceBufferView::new(buf, 0, &bias_shape).unwrap());
    let out = ops
        .linear_forward_device(&a_dev, w_view, bias_view, act)
        .unwrap();
    mem.download(&out)
        .unwrap()
        .contiguous()
        .as_slice()
        .unwrap()
        .to_vec()
}

/// `linear_forward_device` が非融合合成と bit 一致する（MLP 形状・NaN/inf 含む）。
#[test]
fn linear_forward_device_matches_unfused_bit_exact_mlp_and_nonfinite() {
    for shape in [(64, 784, 256), (64, 256, 10)] {
        let (m, k, n) = shape;
        let a = centered(0x51, m * k);
        let w = centered(0x52, k * n);
        let b = centered(0x53, n);
        for bias in [None, Some(&b[..])] {
            for act in [Activation::None, Activation::Relu] {
                assert_eq!(
                    bits(&run_linear_forward_device(&a, &w, bias, shape, act)),
                    bits(&unfused(&a, &w, bias, shape, act)),
                    "shape={shape:?} bias={} act={act:?}",
                    bias.is_some()
                );
            }
        }
    }
    let shape = (4, 6, 5);
    let (m, k, n) = shape;
    let mut a = centered(0x61, m * k);
    let w = centered(0x62, k * n);
    let mut b = centered(0x63, n);
    a[2] = f32::NAN;
    a[9] = f32::INFINITY;
    b[0] = f32::NEG_INFINITY;
    b[3] = -0.0;
    for act in [Activation::None, Activation::Relu] {
        assert_eq!(
            bits(&run_linear_forward_device(&a, &w, Some(&b), shape, act)),
            bits(&unfused(&a, &w, Some(&b), shape, act)),
            "non-finite act={act:?}"
        );
    }
}

/// `linear_forward_device` と `gemm_resident_rhs_act`（学習・fresh 側の融合経路）
/// が同一入力で bit 一致する。
#[test]
fn linear_forward_device_matches_gemm_resident_rhs_act_bit_exact() {
    let ops = CpuBackendOps::new();
    let mem = CpuMemory::new();
    let shape = (64, 256, 10);
    let (m, k, n) = shape;
    let a = centered(0x71, m * k);
    let w = centered(0x72, k * n);
    let b = centered(0x73, n);
    let w_dev = mem.upload(&tensor(w.clone(), &[k, n])).unwrap();
    let b_dev = mem.upload(&tensor(b.clone(), &[n])).unwrap();
    for act in [Activation::None, Activation::Relu] {
        let w_shape = [k, n];
        let b_shape = [n];
        let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
        let b_view = DeviceBufferView::new(&b_dev, 0, &b_shape).unwrap();
        let expected = ops
            .gemm_resident_rhs_act(&tensor(a.clone(), &[m, k]), w_view, Some(b_view), act)
            .unwrap();
        let actual = run_linear_forward_device(&a, &w, Some(&b), shape, act);
        assert_eq!(
            bits(&actual),
            bits(expected.contiguous().as_slice().unwrap()),
            "act={act:?}"
        );
    }
}
