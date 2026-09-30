//! `gemm_resident_rhs_act` の encode-only 合流（イシュー #2113・opt-in・
//! 既定 OFF）の実機テスト。フラグ ON／OFF で出力が `to_bits` 完全一致する
//! こと（tolerance は使わない）と、op 単位の同期回数が意図どおり
//! （OFF: encode +2・wait +2／ON: encode +2・wait +1）になることを固定する。
//!
//! フラグ・診断カウンタはプロセスワイドのため、本ファイル内のテストは
//! static Mutex で直列化する。カウンタ検証は「open バッチなしから開始」
//! （直前に `synchronize` 済み）を前提とする。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test train_forward_encode_only_parity -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::{
    __diagnostic_batch_counters_snapshot, __set_train_forward_encode_only_enabled,
    __train_forward_encode_only_enabled, MetalBackendOps,
};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// フラグを一時設定し drop で元へ戻す（`serialize()` 保持下でのみ使う）。
struct FlagGuard(bool);

impl FlagGuard {
    fn set(enabled: bool) -> Self {
        let original = __train_forward_encode_only_enabled();
        __set_train_forward_encode_only_enabled(enabled);
        Self(original)
    }
}

impl Drop for FlagGuard {
    fn drop(&mut self) {
        __set_train_forward_encode_only_enabled(self.0);
    }
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 決定的疑似乱数（`-0.5` シフトで負値を含め ReLU が非自明になるようにする）。
fn xorshift_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は as_slice() が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 指定フラグ状態で `gemm_resident_rhs_act` を 1 回実行する。
fn run(
    encode_only: bool,
    a: &Tensor<f32>,
    w: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    act: Activation,
) -> Tensor<f32> {
    let _flag = FlagGuard::set(encode_only);
    let ops = MetalBackendOps::new();
    let mem = ops.memory_ops().expect("MemoryOps");
    let w_dev = mem.upload(w).unwrap();
    let w_shape = w.shape().to_vec();
    let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
    let bias_dev = bias.map(|b| mem.upload(b).unwrap());
    let bias_shape = bias.map(|b| b.shape().to_vec());
    let bias_view = match (&bias_dev, &bias_shape) {
        (Some(buf), Some(shape)) => Some(DeviceBufferView::new(buf, 0, shape).unwrap()),
        _ => None,
    };
    ops.gemm_resident_rhs_act(a, w_view, bias_view, act)
        .unwrap()
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn relu_merged_is_bit_identical_to_default_composition() {
    let _g = serialize();
    // タイル端数を含む非倍数の m/n/k を含める。
    for &(m, k, n) in &[
        (1, 1, 1),
        (4, 8, 4),
        (37, 65, 33),
        (64, 784, 256),
        (5, 3, 130),
    ] {
        for has_bias in [false, true] {
            let a = tensor(xorshift_fill(0x2468_ace0 ^ m as u64, m * k), &[m, k]);
            let w = tensor(xorshift_fill(0x1357_9bdf ^ k as u64, k * n), &[k, n]);
            let bias = has_bias.then(|| tensor(xorshift_fill(0xa5a5_a5a5 ^ n as u64, n), &[n]));
            let off = run(false, &a, &w, bias.as_ref(), Activation::Relu);
            let on = run(true, &a, &w, bias.as_ref(), Activation::Relu);
            assert_eq!(off.shape(), on.shape());
            assert_eq!(
                bits(&off),
                bits(&on),
                "ON/OFF は bit 完全一致のはず: m={m} k={k} n={n} has_bias={has_bias}"
            );
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn relu_merged_handles_transposed_lhs_and_empty_dims() {
    let _g = serialize();
    // 転置 view の `a`（`upload_operand_for_resident_gemm` の zero-repack 経路）。
    let (m, k, n) = (17, 23, 9);
    let a_t = tensor(xorshift_fill(0x77, k * m), &[k, m]);
    let a = a_t.transpose(0, 1).unwrap();
    let w = tensor(xorshift_fill(0x88, k * n), &[k, n]);
    let off = run(false, &a, &w, None, Activation::Relu);
    let on = run(true, &a, &w, None, Activation::Relu);
    assert_eq!(bits(&off), bits(&on));

    // m == 0・n == 0 は空 Tensor（形状も一致）。
    let a0 = tensor(Vec::new(), &[0, k]);
    let off0 = run(false, &a0, &w, None, Activation::Relu);
    let on0 = run(true, &a0, &w, None, Activation::Relu);
    assert_eq!(off0.shape(), on0.shape());
    let w0 = tensor(Vec::new(), &[k, 0]);
    let a1 = tensor(xorshift_fill(0x99, 3 * k), &[3, k]);
    let off1 = run(false, &a1, &w0, None, Activation::Relu);
    let on1 = run(true, &a1, &w0, None, Activation::Relu);
    assert_eq!(off1.shape(), on1.shape());
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn act_none_is_bit_identical_regardless_of_flag() {
    let _g = serialize();
    let (m, k, n) = (37, 65, 33);
    let a = tensor(xorshift_fill(1, m * k), &[m, k]);
    let w = tensor(xorshift_fill(2, k * n), &[k, n]);
    let bias = tensor(xorshift_fill(3, n), &[n]);
    let off = run(false, &a, &w, Some(&bias), Activation::None);
    let on = run(true, &a, &w, Some(&bias), Activation::None);
    assert_eq!(bits(&off), bits(&on));
}

/// op 単位のカウンタ。OFF は gemm 1 同期 + relu 1 同期、ON は 1 同期へ合流。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn relu_merged_reduces_waits_by_one_per_op() {
    let _g = serialize();
    let (m, k, n) = (37, 65, 33);
    let a = tensor(xorshift_fill(11, m * k), &[m, k]);
    let w = tensor(xorshift_fill(12, k * n), &[k, n]);
    let bias = tensor(xorshift_fill(13, n), &[n]);

    let mut deltas = Vec::new();
    for encode_only in [false, true] {
        // open バッチを残さない状態から開始する（ウォームアップ 1 回で
        // パイプライン・プールを温め、末尾は download 済みで synchronize 済み）。
        let _ = run(encode_only, &a, &w, Some(&bias), Activation::Relu);
        let before = __diagnostic_batch_counters_snapshot().unwrap();
        let _ = run(encode_only, &a, &w, Some(&bias), Activation::Relu);
        let after = __diagnostic_batch_counters_snapshot().unwrap();
        deltas.push((
            after.encode_calls - before.encode_calls,
            after.wait_until_completed - before.wait_until_completed,
        ));
    }
    println!(
        "op counters (encode, wait): OFF={:?} ON={:?}",
        deltas[0], deltas[1]
    );
    assert_eq!(
        deltas[0],
        (2, 2),
        "OFF: gemm と relu が各 1 encode・各 1 wait"
    );
    assert_eq!(deltas[1], (2, 1), "ON: encode は不変・wait は 1 へ合流");
}
