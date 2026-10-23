//! イシュー #3250: 常駐入力の行列積（`BackendOps::gemm_device`）と推論チェーンの層列が、
//! 計測窓の中で入力の H2D（`MetalBuffer::new_with_data` = `newBufferWithBytes` の memcpy）を
//! 起こさないこと、`gemm_device` がホスト入力の `gemm` と bit 一致（NaN はクラス一致）する
//! ことを確かめる Metal 実機テスト（`docs/compat-resident-train-step-decision.md` §4・§10）。
//!
//! # 配置理由
//! H2D カウンタ（`buffer::diag_h2d_buffer_count`。`cfg(test)`・`pub(crate)`）へ到達するため
//! integration test ではなく `lib.rs` の兄弟モジュールとして置く。カウンタはスレッドローカル
//! なので、計測区間の呼び出しは同じスレッドで行う。対照としてホスト入力の `gemm` が H2D を
//! 数えること（カウンタが常に 0 を返す壊れ方をしていないこと）も確かめる。
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal resident_input_diag_tests_3250 -- --ignored --test-threads=1`

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView};
use fandhe_ai_tensor_core::{Activation, BackendOps, DispatchFailureCell, Tensor};

use crate::buffer::diag_h2d_buffer_count;
use crate::ops::MetalBackendOps;

fn tensor(salt: usize, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data = (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.37)
        .collect();
    Tensor::new(data, shape).expect("tensor")
}

fn upload(ops: &MetalBackendOps, t: &Tensor<f32>) -> DeviceBuffer<f32> {
    ops.memory_ops()
        .expect("Metal は MemoryOps を実装")
        .upload(t)
        .expect("upload")
}

fn download(ops: &MetalBackendOps, b: &DeviceBuffer<f32>) -> Tensor<f32> {
    ops.memory_ops()
        .expect("Metal は MemoryOps を実装")
        .download(b)
        .expect("download")
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

/// `gemm_device` が `gemm` と bit 一致する（8 の倍数の形状。split-K 経路に入る形状を含む）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_device_matches_host_gemm_bitwise() {
    let ops = MetalBackendOps::new();
    for &(m, k, n) in &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (64, 784, 256),
        (8, 8, 8),
        (16, 4096, 16),
    ] {
        let a = tensor(1, &[m, k]);
        let b = tensor(2, &[k, n]);
        let want = ops.gemm(&a, &b).expect("host gemm");
        let (ab, bb) = (upload(&ops, &a), upload(&ops, &b));
        let got = download(&ops, &ops.gemm_device(&ab, &bb).expect("gemm_device"));
        assert_eq!(got.shape(), &[m, n]);
        assert_bits_or_nan_class(
            got.as_slice().expect("contiguous"),
            want.as_slice().expect("contiguous"),
            &format!("{m}x{k}x{n}"),
        );
    }
}

/// 8 の倍数でない形状は、黙って別経路で計算せず `Unsupported`。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_device_rejects_non_multiple_of_8_with_unsupported() {
    let ops = MetalBackendOps::new();
    let (ab, bb) = (
        upload(&ops, &tensor(1, &[7, 13])),
        upload(&ops, &tensor(2, &[13, 5])),
    );
    assert!(matches!(
        ops.gemm_device(&ab, &bb),
        Err(fandhe_ai_tensor_core::BackendError::Unsupported(_))
    ));
}

/// 窓の中（`gemm_device` と結果の `download`）で入力の H2D が 0。対照: ホスト入力の `gemm` は
/// H2D を数える。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_device_window_has_no_input_h2d() {
    let ops = MetalBackendOps::new();
    let n = 1024usize;
    let (a, b) = (tensor(1, &[n, n]), tensor(2, &[n, n]));
    let (ab, bb) = (upload(&ops, &a), upload(&ops, &b));
    let _ = download(&ops, &ops.gemm_device(&ab, &bb).expect("warmup"));

    // 対照: ホスト入力版は窓の中で H2D が起きる。
    let c0 = diag_h2d_buffer_count();
    let _ = ops.gemm(&a, &b).expect("host gemm");
    assert!(
        diag_h2d_buffer_count() > c0,
        "host-input gemm must count H2D"
    );

    let before = diag_h2d_buffer_count();
    let out = ops.gemm_device(&ab, &bb).expect("gemm_device");
    let _ = download(&ops, &out);
    assert_eq!(diag_h2d_buffer_count() - before, 0, "input H2D in window");
}

/// 推論チェーンの層列（`predict_device_chain_resident` が呼ぶ列そのもの）が、upload 済みの
/// 入力から窓の中で H2D を起こさない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn chain_layers_window_has_no_input_h2d() {
    let ops = MetalBackendOps::new();
    let (batch, d_in, d_hid, d_out) = (64usize, 784usize, 256usize, 10usize);
    let x = upload(&ops, &tensor(1, &[batch, d_in]));
    let w1 = upload(&ops, &tensor(2, &[d_in, d_hid]));
    let b1 = upload(&ops, &tensor(3, &[d_hid]));
    let w2 = upload(&ops, &tensor(4, &[d_hid, d_out]));
    let token = DispatchFailureCell::new();
    let (w1s, b1s, w2s) = ([d_in, d_hid], [d_hid], [d_hid, d_out]);
    let run = || {
        let w1v = DeviceBufferView::new(&w1, 0, &w1s).unwrap();
        let b1v = DeviceBufferView::new(&b1, 0, &b1s).unwrap();
        let w2v = DeviceBufferView::new(&w2, 0, &w2s).unwrap();
        let h = ops
            .linear_forward_device_tracked(&x, w1v, Some(b1v), Activation::Relu, &token)
            .expect("layer 1");
        ops.linear_forward_device_tracked(&h, w2v, None, Activation::None, &token)
            .expect("layer 2")
    };
    let _ = download(&ops, &run());

    let before = diag_h2d_buffer_count();
    let out = run();
    let _ = download(&ops, &out);
    assert_eq!(diag_h2d_buffer_count() - before, 0, "input H2D in window");
}
