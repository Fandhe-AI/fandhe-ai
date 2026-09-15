//! `Var::conv1d`（`Var::conv2d` の reshape 併合。イシュー #1765・設計
//! `docs/conv-ops-design.md` §2／§8）の facade 到達経路（既存 `Var`
//! 再エクスポート経由。新規 `pub use`／`pub fn` は追加していない）の
//! 受け入れ条件対応テスト（`conv2d_backend_parity.rs` と同型）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward／
//!   backward を bit 同一で突き合わせる（`conv1d` 自体は
//!   `contiguous`／`reshape`／`conv2d` への委譲のみで新規カーネルを
//!   持たないため、`conv2d` 側の im2col／col2im・GEMM の bit 完全
//!   一致契約がそのまま伝播する）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` を CPU tape と
//!   `assert_parity`（REQ-2 複合判定。GEMM 由来の差を許容）で比較する。
//!   実機未実測のまま出荷し記入欄を残す。
//!
//! **CUDA（イシュー #1767「Conv1d を Conv2d の特化として実装する」）**:
//! CUDA は #1766 で `BackendOps::im2col`／`col2im` を override 済み
//! （bit 完全一致契約）で、`conv1d` 自身は #1765 の reshape 併合
//! （新規 `Op`／`BackendOps`／カーネルなし）のため、CUDA 経路も
//! `conv2d_backend_parity.rs` と同じく「CUDA im2col（bit 一致）→
//! CUDA GEMM（REQ-2 複合判定）→ CUDA add → CUDA col2im（bit 一致）」
//! の合成としてそのまま到達する（本ファイル追記時点でコード変更は
//! テストのみ・facade 新規公開面なし）。`cuda_conv1d_matches_manual_
//! reshape_conv2d_bit_exact` は「特化」契約——同一 CUDA tape 上で
//! `conv1d` と手動 reshape の `conv2d` が同一カーネル・同一形状を
//! 通るため forward／backward とも bit 完全一致する——を実機で固定
//! する（`crates/autodiff/tests/conv1d.rs::matches_manual_reshape_
//! conv2d_bit_exact` の CUDA 版）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする
/// （`conv2d_backend_parity.rs::VarSource` と同じ理由・同じ構成）。
trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

/// bit 完全一致（NaN 同士はクラス一致）の判定ヘルパー
/// （`conv2d_backend_parity.rs::assert_bits_eq` と同型）。
fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス一致しない（actual={a}, expected={e}）"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
    }
}

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- forward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv1d_forward_matches_naive_reference() {
    let x_shape = [1usize, 2, 9];
    let w_shape = [3usize, 2, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(1, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(2, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(3, &b_shape));
    let out_cpu = x_cpu
        .conv1d(&w_cpu, Some(&b_cpu), 1, 1, 1, 1)
        .expect("conv1d: 常に成功する形状")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(1, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(2, &w_shape));
    let b_naive = naive_tape.make_var(&leaf(3, &b_shape));
    let out_naive = x_naive
        .conv1d(&w_naive, Some(&b_naive), 1, 1, 1, 1)
        .expect("conv1d: 常に成功する形状")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_parity(
        "fandhe_ai::tape()（CpuBackendOps 経由 conv1d）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
    assert_bits_eq(
        "fandhe_ai::tape()（CpuBackendOps 経由 conv1d）vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

#[test]
fn cpu_conv1d_forward_matches_naive_reference_groups_no_bias() {
    let x_shape = [1usize, 4, 10];
    let w_shape = [4usize, 1, 3]; // depthwise（groups=4）

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(4, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(5, &w_shape));
    let out_cpu = x_cpu
        .conv1d(&w_cpu, None, 1, 1, 1, 4)
        .expect("conv1d: 常に成功する形状")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(4, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(5, &w_shape));
    let out_naive = x_naive
        .conv1d(&w_naive, None, 1, 1, 1, 4)
        .expect("conv1d: 常に成功する形状")
        .to_tensor();

    let cpu_slice = contiguous_slice(&out_cpu);
    let naive_slice = contiguous_slice(&out_naive);
    assert_bits_eq(
        "conv1d forward（groups=4・depthwise）: CpuBackendOps vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

// --- backward（属性なし: CPU vs NaiveOps） ---

#[test]
fn cpu_conv1d_backward_matches_naive_reference() {
    let x_shape = [1usize, 2, 9];
    let w_shape = [3usize, 2, 3];
    let b_shape = [3usize];

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&leaf(6, &x_shape));
    let w_cpu = cpu_tape.make_var(&leaf(7, &w_shape));
    let b_cpu = cpu_tape.make_var(&leaf(8, &b_shape));
    let out_cpu = x_cpu.conv1d(&w_cpu, Some(&b_cpu), 1, 1, 1, 1).unwrap();
    let loss_cpu = out_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");
    let db_cpu = grads_cpu.get(&b_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&leaf(6, &x_shape));
    let w_naive = naive_tape.make_var(&leaf(7, &w_shape));
    let b_naive = naive_tape.make_var(&leaf(8, &b_shape));
    let out_naive = x_naive
        .conv1d(&w_naive, Some(&b_naive), 1, 1, 1, 1)
        .unwrap();
    let loss_naive = out_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");
    let db_naive = grads_naive.get(&b_naive).unwrap().expect("到達する");

    assert_bits_eq(
        "conv1d backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_bits_eq(
        "conv1d backward（dw）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
    assert_bits_eq(
        "conv1d backward（db）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(db_cpu),
        &contiguous_slice(db_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 複合判定） ---

fn conv1d_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 9]));
    let w = tape.make_var(&leaf(2, &[3, 2, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    x.conv1d(&w, Some(&b), 1, 1, 1, 1)
        .expect("conv1d: 常に成功する形状")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`conv2d_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_conv1d_forward_matches_cpu() {
    let metal_out = conv1d_forward_on(Device::Metal);
    let cpu_out = conv1d_forward_on(Device::Cpu);

    // conv1d は conv2d への reshape 併合のため、conv2d 側と同じく
    // ホスト im2col（bit 一致）＋ GPU GEMM（REQ-2 複合判定）の合成
    // として forward 全体を REQ-2 複合判定で比較する。
    assert_parity(
        "conv1d forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv1d_forward_matches_cpu() {
    let cuda_out = conv1d_forward_on(Device::Cuda(0));
    let cpu_out = conv1d_forward_on(Device::Cpu);

    assert_parity(
        "conv1d forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

/// `device` 上で conv1d backward（forward → `sum(None)` →
/// `backward`）を実行し `(dx, dw, db)` を返す（イシュー #1767。
/// `conv2d_backend_parity.rs::conv2d_backward_on` と同型）。
fn conv1d_backward_on(device: Device) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(1, &[1, 2, 9]));
    let w = tape.make_var(&leaf(2, &[3, 2, 3]));
    let b = tape.make_var(&leaf(3, &[3]));
    let out = x
        .conv1d(&w, Some(&b), 1, 1, 1, 1)
        .expect("conv1d: 常に成功する形状");
    let loss = out.sum(None).expect("sum: 常に成功する");
    let grads = tape.backward(&loss).expect("backward: 常に成功する形状");
    let dx = grads.get(&x).unwrap().expect("到達する").clone();
    let dw = grads.get(&w).unwrap().expect("到達する").clone();
    let db = grads.get(&b).unwrap().expect("到達する").clone();
    (dx, dw, db)
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv1d_backward_matches_cpu() {
    let (dx_cuda, dw_cuda, db_cuda) = conv1d_backward_on(Device::Cuda(0));
    let (dx_cpu, dw_cpu, db_cpu) = conv1d_backward_on(Device::Cpu);

    assert_parity(
        "conv1d backward（dx）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&dx_cuda),
        &contiguous_slice(&dx_cpu),
    );
    assert_parity(
        "conv1d backward（dw）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&dw_cuda),
        &contiguous_slice(&dw_cpu),
    );
    assert_parity(
        "conv1d backward（db）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&db_cuda),
        &contiguous_slice(&db_cpu),
    );
}

/// イシュー #1767「Conv1d を Conv2d の特化として実装する」の中核
/// 契約: 同一 CUDA tape 上で `conv1d` と「手動 reshape → `conv2d` →
/// reshape」が forward／backward とも **bit 完全一致**する
/// （`crates/autodiff/tests/conv1d.rs::matches_manual_reshape_conv2d_
/// bit_exact` の CUDA 版。`conv1d` は新規カーネルを持たず reshape
/// 併合のみのため、CUDA の im2col／col2im／GEMM いずれも同一形状・
/// 同一カーネル呼び出しを経由し bit 同一が構造的に成立する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv1d_matches_manual_reshape_conv2d_bit_exact() {
    let tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");

    let n = 2usize;
    let cin = 3usize;
    let l = 9usize;
    let cout = 4usize;
    let cin_g = 3usize;
    let k = 3usize;
    let stride = 1usize;
    let padding = 1usize;
    let dilation = 1usize;
    let groups = 1usize;

    let x_data = leaf(11, &[n, cin, l]);
    let w_data = leaf(12, &[cout, cin_g, k]);
    let b_data = leaf(13, &[cout]);

    // conv1d 経路。
    let x1 = tape.make_var(&x_data);
    let w1 = tape.make_var(&w_data);
    let b1 = tape.make_var(&b_data);
    let y1 = x1
        .conv1d(&w1, Some(&b1), stride, padding, dilation, groups)
        .expect("conv1d: 常に成功する形状");
    let out1 = y1.to_tensor();
    let loss1 = y1.sum(None).expect("sum: 常に成功する");
    let grads1 = tape.backward(&loss1).expect("backward: 常に成功する形状");
    let dx1 = grads1.get(&x1).unwrap().expect("到達する").clone();
    let dw1 = grads1.get(&w1).unwrap().expect("到達する").clone();
    let db1 = grads1.get(&b1).unwrap().expect("到達する").clone();

    // 手動 reshape -> conv2d 経路（同一 tape 上・別 leaf ノード）。
    let x4_data = Tensor::new(
        x_data.contiguous().as_slice().expect("contiguous").to_vec(),
        &[n, cin, 1, l],
    )
    .expect("valid reshape");
    let w4_data = Tensor::new(
        w_data.contiguous().as_slice().expect("contiguous").to_vec(),
        &[cout, cin_g, 1, k],
    )
    .expect("valid reshape");
    let x2 = tape.make_var(&x4_data);
    let w2 = tape.make_var(&w4_data);
    let b2 = tape.make_var(&b_data);
    let y2 = x2
        .conv2d(
            &w2,
            Some(&b2),
            [1, stride],
            [0, padding],
            [1, dilation],
            groups,
        )
        .expect("conv2d: 常に成功する形状");
    let out2 = y2.to_tensor();
    let loss2 = y2.sum(None).expect("sum: 常に成功する");
    let grads2 = tape.backward(&loss2).expect("backward: 常に成功する形状");
    let dx2 = grads2.get(&x2).unwrap().expect("到達する").clone();
    let dw2 = grads2.get(&w2).unwrap().expect("到達する").clone();
    let db2 = grads2.get(&b2).unwrap().expect("到達する").clone();

    assert_bits_eq(
        "conv1d vs manual-reshape conv2d（CUDA forward）",
        &contiguous_slice(&out1),
        &contiguous_slice(&out2),
    );
    assert_bits_eq(
        "conv1d vs manual-reshape conv2d（CUDA d_input）",
        &contiguous_slice(&dx1),
        &contiguous_slice(&dx2),
    );
    assert_bits_eq(
        "conv1d vs manual-reshape conv2d（CUDA d_weight）",
        &contiguous_slice(&dw1),
        &contiguous_slice(&dw2),
    );
    assert_bits_eq(
        "conv1d vs manual-reshape conv2d（CUDA d_bias）",
        &contiguous_slice(&db1),
        &contiguous_slice(&db2),
    );
}

/// groups＋dilation を伴う 1d 形状の CUDA forward parity（イシュー
/// #1767。`conv2d_backend_parity.rs` の groups 系ケースと同型）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_conv1d_forward_matches_cpu_groups_dilation() {
    fn forward_groups_dilation(device: Device) -> Tensor<f32> {
        let tape =
            fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
        let x = tape.make_var(&leaf(21, &[2, 4, 11]));
        let w = tape.make_var(&leaf(22, &[4, 1, 3]));
        let b = tape.make_var(&leaf(23, &[4]));
        x.conv1d(&w, Some(&b), 1, 2, 2, 4)
            .expect("conv1d: 常に成功する形状")
            .to_tensor()
    }

    let cuda_out = forward_groups_dilation(Device::Cuda(0));
    let cpu_out = forward_groups_dilation(Device::Cpu);

    assert_parity(
        "conv1d forward（groups=4・dilation=2）: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}
