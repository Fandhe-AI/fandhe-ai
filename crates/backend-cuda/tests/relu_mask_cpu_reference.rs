//! イシュー #3221: `relu_backward_mask_device` のホストモデル
//! （`relu_mask_model::relu_mask_select_host_model` = CUDA カーネルの逐語再現）
//! が、CPU の実関数経路（`Tape` の `Op::Relu` VJP =
//! `autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)`）と bit 一致することを
//! Linux でも検証する（`elementwise_mul_mask` は private のため、公開 API の
//! `Tape::backward` 経由で実関数の出力を得る）。実機の GPU 出力との一致は
//! `relu_mask_device_parity.rs`（CUDA 実機・`#[ignore]`）が担う。
//!
//! 参照式: `if m > 0.0 { g } else { 0.0 }`（`grad.rs` の `Op::Relu` 分岐と同一）。
//! 判定に tolerance は使わず bit 一致（NaN はクラス一致）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::relu_mask_model::{plan_relu_mask, relu_mask_select_host_model};
use fandhe_ai_tensor_core::{DeviceMatrixLayout, Tensor};

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    const SPECIAL: [f32; 10] = [
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
    ];
    (0..n)
        .map(|i| {
            if (i + salt).is_multiple_of(3) {
                SPECIAL[(i * 5 + salt) % SPECIAL.len()]
            } else {
                ((i * 31 + salt * 17) % 23) as f32 - 11.0
            }
        })
        .collect()
}

/// CPU Tape 経由の `Op::Relu` VJP 出力（`x` に対する勾配 = `elementwise_mul_mask(g, x, >0)`）。
fn tape_reference(g_logical: &[f32], mask: &[f32], shape: &[usize]) -> Option<Vec<f32>> {
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let x = tape.var(&Tensor::new(mask.to_vec(), shape).unwrap());
    let gc = tape.var(&Tensor::new(g_logical.to_vec(), shape).unwrap());
    let y = x.relu();
    let loss = y.mul(&gc).ok()?.sum(None).ok()?;
    let grads = tape.backward(&loss).ok()?;
    let gx = grads.get(&x).ok()??;
    Some(gx.as_slice()?.to_vec())
}

fn assert_bits_or_nan_class(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "idx {i}: expected NaN, got {a}");
        } else {
            assert_eq!(a.to_bits(), e.to_bits(), "idx {i}: {a} vs {e}");
        }
    }
}

#[test]
fn host_model_matches_cpu_tape_relu_vjp() {
    let mut compared = 0usize;
    for (r, c) in [(1usize, 1usize), (1, 13), (7, 13), (63, 65), (64, 128)] {
        let n = r * c;
        let g = pattern(n, 1);
        let m = pattern(n, 2);
        let plan = plan_relu_mask(&[r, c], DeviceMatrixLayout::RowMajor, &[r, c]).unwrap();
        let model = relu_mask_select_host_model(&plan, &g, &m).unwrap();
        if let Some(want) = tape_reference(&g, &m, &[r, c]) {
            assert_bits_or_nan_class(&model, &want);
            compared += 1;
        }
        // 実関数経路が非有限入力を拒否する場合に備え、有限入力のみでも必ず検証する。
        let gf: Vec<f32> = g
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.5 })
            .collect();
        let mf: Vec<f32> = m
            .iter()
            .map(|v| if v.is_finite() { *v } else { -0.5 })
            .collect();
        let model_f = relu_mask_select_host_model(&plan, &gf, &mf).unwrap();
        let want_f = tape_reference(&gf, &mf, &[r, c]).expect("finite input must backprop");
        assert_bits_or_nan_class(&model_f, &want_f);
        compared += 1;
    }
    assert!(compared >= 5);
}

/// 転置 view: 格納 `[r, c]` の `g` を論理 `[c, r]` として読む形も、実関数経路
/// （論理形の連続 `g`）と bit 一致する。
#[test]
fn host_model_transposed_matches_cpu_tape_relu_vjp() {
    for (r, c) in [(1usize, 13usize), (13, 1), (7, 13), (257, 3), (3, 129)] {
        let n = r * c;
        let g_stored = pattern(n, 3);
        let m = pattern(n, 4);
        // 論理 [c, r] の行優先へ並べ替えた g（実関数経路に渡す形）。
        let mut g_logical = vec![0.0f32; n];
        for i in 0..r {
            for j in 0..c {
                g_logical[j * r + i] = g_stored[i * c + j];
            }
        }
        let plan = plan_relu_mask(&[r, c], DeviceMatrixLayout::Transposed, &[c, r]).unwrap();
        let model = relu_mask_select_host_model(&plan, &g_stored, &m).unwrap();
        let fin = |v: &[f32], d: f32| -> Vec<f32> {
            v.iter()
                .map(|x| if x.is_finite() { *x } else { d })
                .collect()
        };
        let want = tape_reference(&fin(&g_logical, 0.5), &fin(&m, -0.5), &[c, r])
            .expect("finite input must backprop");
        let model_f = relu_mask_select_host_model(
            &plan,
            &{
                let mut s = vec![0.0f32; n];
                let gl = fin(&g_logical, 0.5);
                for i in 0..r {
                    for j in 0..c {
                        s[i * c + j] = gl[j * r + i];
                    }
                }
                s
            },
            &fin(&m, -0.5),
        )
        .unwrap();
        assert_bits_or_nan_class(&model_f, &want);
        assert_eq!(model.len(), n);
    }
}
