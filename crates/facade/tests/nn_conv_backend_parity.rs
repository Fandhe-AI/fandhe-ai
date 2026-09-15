//! `nn::Conv2d`／`nn::Conv1d`（イシュー #1770・親 #1645）を積んだ
//! `compat::Sequential` の 3 バックエンド parity テスト
//! （`conv2d_backend_parity.rs`／`conv1d_backend_parity.rs` と同型）。
//!
//! - 属性なし（CPU）: `fandhe_ai::tape()`（`CpuBackendOps`）上で
//!   `compat::Sequential(add_conv2d→add_relu)` の forward／backward が
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）上に同じ重みで組んだ
//!   `nn::Conv2d::from_parameters` 参照実装（`bind`／`trainable_grads`
//!   相当の経路）と bit 完全一致することを確認する。
//! - `#[ignore]`: CUDA（`tape_for(Device::Cuda(0))`）・Metal
//!   （`tape_for(Device::Metal)`。`cfg(target_os = "macos")` 限定）の
//!   forward／backward（入力勾配・weight／bias 勾配）を CPU と
//!   `assert_parity`（REQ-2 複合判定）で比較する。
//!
//! **イシュー #1771 で追加**（実機実測の正式な受け皿。
//! `docs/perf/logs/conv-realdevice-1771/`）:
//! - `{cuda,metal}_sequential_conv2d_backward_matches_cpu`: nn 層
//!   （`compat::Sequential(add_conv2d)`）の backward（weight／bias／
//!   入力勾配）を CPU と REQ-2 複合判定で比較する（forward のみだった
//!   既存 `{cuda,metal}_sequential_conv2d_matches_cpu` の backward 版）。
//! - `{cuda,metal}_sequential_conv1d_{forward,backward}_matches_cpu`:
//!   `add_conv1d` 版（`conv1d_backend_parity.rs` は `Var::conv1d` 直叩き
//!   のみだったため、nn 層〈`compat::Sequential`〉経由の 1d 版を補う）。
//! - `{cuda,metal}_sequential_conv1d_matches_manual_reshape_conv2d_
//!   bit_exact`: `conv1d_backend_parity.rs::conv1d_matches_manual_
//!   reshape_conv2d_on` の nn 層版。同一 GPU tape 上で `add_conv1d`
//!   モデルと `add_conv2d`（`[1,k]`・`[0,p]`・`[1,d]`）モデルへ同一
//!   重みを注入し、forward・勾配とも bit 完全一致することを確認する
//!   （「特化」契約。`conv1d` は reshape 併合のみで新規カーネルを
//!   持たないため機構的に成立する）。
//! - `{cuda,metal}_sequential_conv2d_sgd_steps_record_only`:
//!   `compat_sequential_conv.rs::train_loop_with_sgd_reduces_loss` と
//!   同じ形状・SGD 設定で 5 step を GPU／CPU 双方で回し、各 step の
//!   loss・最終パラメータを比較する（**record-only**。GEMM 由来の差が
//!   step をまたいで累積しうるため ADOPT／REJECT の判定対象にしない。
//!   `docs/conv-ops-design.md` §7・§15「#1771」参照）。
//!
//! **bit 行出力**（run-to-run 決定性の機械検査対象。
//! `docs/perf/logs/conv-realdevice-1771/README.md` 参照）: 上記の
//! 各 GPU 出力・勾配について `print_fold_bits` が
//! `<test>[<label>].fold_bits=<hex>` 形式で 1 行出力する
//! （`mse_backward_bench.rs::fold_bits` と同一の FNV-1a fold）。実機
//! ランブックはこの行を `grep` で抽出し 2 回起動の出力を `diff` して
//! run-to-run bit 同一を確認する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai_autodiff::Tape as RawTape;
use fandhe_ai_autodiff::nn::{Conv2d, Module};
use fandhe_ai_backend_cpu::parity::{assert_parity, compare};
use fandhe_ai_tensor_core::Tensor;

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

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

/// `t` の全要素の `to_bits()` を FNV-1a 相当で fold した診断用
/// チェックサムを `<label>.fold_bits=<hex>` 形式で 1 行出力する
/// （`mse_backward_bench.rs::fold_bits` と同一実装。実機ランブックが
/// この行を 2 回起動間で `diff` し run-to-run bit 同一を確認する）。
fn print_fold_bits(label: &str, t: &Tensor<f32>) {
    let mut acc: u64 = 0xcbf29ce484222325; // FNV-1a 相当の固定初期値（診断専用・暗号用途ではない）
    for &v in dense(t).iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

/// **record-only** 用の REQ-2 複合判定ラッパー（イシュー #1771・
/// PR #1882 レビュー指摘）。`{cuda,metal}_sequential_conv2d_sgd_steps_
/// record_only`（README「事前登録判定規則」5)。ADOPT／REJECT の
/// 判定対象にしない）は `assert_parity` で panic させると、5 step
/// 分の loss・パラメータのうち途中で fail した時点以降の
/// `print_fold_bits` 行が出力されず実測記録が欠落してしまう
/// （`--ignored` 全体も失敗扱いになり後続ケースの実行が止まる）。
/// 本関数は判定結果（PASS／FAIL と統計値）を `println!` するだけで
/// panic しない。合否は `docs/perf/logs/conv-realdevice-1771/` の
/// 事前登録判定規則どおり記録専用として扱う。
fn record_parity(context: &str, actual: &[f32], expected: &[f32]) {
    match compare(actual, expected) {
        Ok(report) => {
            let verdict = if report.passes() { "PASS" } else { "FAIL" };
            println!(
                "{context}: record-only 複合判定 {verdict}（fail_count={}/{}, \
                 max_abs_diff={:.3e}, max_rel_err={:.3e}, mean_abs_diff={:.3e}, \
                 mean_rel_err={:.3e}）",
                report.fail_count,
                report.total,
                report.max_abs_diff,
                report.max_rel_err,
                report.mean_abs_diff,
                report.mean_rel_err,
            );
        }
        Err(err) => {
            println!("{context}: record-only 複合判定 ERROR（{err}）");
        }
    }
}

// --- CPU: forward（compat::Sequential vs nn::Conv2d 直接） ---

#[test]
fn cpu_sequential_conv2d_predict_matches_naive_conv2d_layer() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];

    let x = leaf(11, &x_shape);
    let w = leaf(12, &w_shape);
    let b = leaf(13, &b_shape);

    // `compat::Sequential` 経由（`CpuBackendOps`。`predict` = tape 不要経路
    // または via-tape フォールバック）。`apply_parameters` で決定的な
    // 重みへ置き換える（`add_conv2d` 自体は seed 決定なので、直接
    // 比較用に固定値を注入する）。
    let mut model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    model.apply_parameters(vec![w.clone(), b.clone()]).unwrap();
    let seq_out = model.predict(&x).unwrap();

    // naive 参照実装（`NaiveOps` 上に `nn::Conv2d::from_parameters` を
    // 直接組む）。
    let naive_conv = Conv2d::from_parameters(w, Some(b), [1, 1], [1, 1], [1, 1], 1).unwrap();
    let naive_tape = RawTape::new();
    let naive_xv = naive_tape.var(&x);
    let naive_out = Module::forward(&naive_conv, &naive_tape, &naive_xv)
        .unwrap()
        .to_tensor();

    let seq_slice = dense(&seq_out);
    let naive_slice = dense(&naive_out);
    assert_parity(
        "compat::Sequential(add_conv2d) vs nn::Conv2d::from_parameters（NaiveOps）",
        &seq_slice,
        &naive_slice,
    );
    assert_bits_eq(
        "compat::Sequential(add_conv2d) vs nn::Conv2d::from_parameters（NaiveOps）",
        &seq_slice,
        &naive_slice,
    );
}

// --- CPU: backward（weight／bias／input 勾配） ---

#[test]
fn cpu_sequential_conv2d_backward_matches_naive_conv2d_layer() {
    let x_shape = [2usize, 2, 4, 4];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let target_shape = [2usize, 3, 4, 4];

    let x = leaf(21, &x_shape);
    let w = leaf(22, &w_shape);
    let b = leaf(23, &b_shape);
    let target = leaf(24, &target_shape);

    // compat::Sequential 経由（`bind` → `trainable_grads`）。
    let mut model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    model.apply_parameters(vec![w.clone(), b.clone()]).unwrap();

    let seq_tape = fandhe_ai::tape();
    let bound = model.bind(&seq_tape);
    let xv = seq_tape.var(&x);
    let tv = seq_tape.var(&target);
    let pred = bound.forward(&seq_tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    let seq_grads = seq_tape.backward(&loss).unwrap();
    let seq_grad_refs = bound.trainable_grads(&seq_grads).unwrap();
    let seq_input_grad = seq_grads.get(&xv).unwrap().unwrap();

    // naive 参照実装。
    let naive_conv = Conv2d::from_parameters(w, Some(b), [1, 1], [1, 1], [1, 1], 1).unwrap();
    let naive_tape = RawTape::new();
    let naive_xv = naive_tape.var(&x);
    let naive_tv = naive_tape.var(&target);
    let naive_vars = naive_conv.bind(&naive_tape);
    let naive_pred = naive_vars.forward(&naive_xv).unwrap();
    let naive_loss = naive_pred.mse_loss(&naive_tv).unwrap();
    let naive_grads = naive_tape.backward(&naive_loss).unwrap();
    let naive_weight_grad = naive_grads.get(&naive_vars.weight).unwrap().unwrap();
    let naive_bias_grad = naive_grads
        .get(naive_vars.bias.as_ref().unwrap())
        .unwrap()
        .unwrap();
    let naive_input_grad = naive_grads.get(&naive_xv).unwrap().unwrap();

    assert_eq!(seq_grad_refs.len(), 2, "weight・bias の 2 件");
    assert_bits_eq(
        "weight_grad: Sequential vs naive",
        &dense(seq_grad_refs[0]),
        &dense(naive_weight_grad),
    );
    assert_bits_eq(
        "bias_grad: Sequential vs naive",
        &dense(seq_grad_refs[1]),
        &dense(naive_bias_grad),
    );
    assert_bits_eq(
        "input_grad: Sequential vs naive",
        &dense(seq_input_grad),
        &dense(naive_input_grad),
    );
}

// --- CUDA（イシュー #1771 実行時記入欄。本 PR では #[ignore] のまま
// 未実行・未実測） ---

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ"]
fn cuda_sequential_conv2d_matches_cpu() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let x = leaf(31, &x_shape);
    let w = leaf(32, &w_shape);
    let b = leaf(33, &b_shape);

    let mut cpu_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    cpu_model
        .apply_parameters(vec![w.clone(), b.clone()])
        .unwrap();
    let cpu_out = cpu_model.predict(&x).unwrap();

    let cuda_tape = fandhe_ai::tape_for(fandhe_ai::Device::Cuda(0))
        .expect("CUDA 実機必須（本テストは #[ignore]）");
    let mut cuda_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    cuda_model.apply_parameters(vec![w, b]).unwrap();
    let xv = cuda_tape.var(&x);
    let cuda_out = cuda_model.forward(&cuda_tape, &xv).unwrap().to_tensor();

    assert_parity("CUDA vs CPU", &dense(&cuda_out), &dense(&cpu_out));
}

// --- Metal（イシュー #1771 実行時記入欄。本 PR では #[ignore] のまま
// 未実行・未実測） ---

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ"]
fn metal_sequential_conv2d_matches_cpu() {
    let x_shape = [1usize, 2, 5, 5];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let x = leaf(41, &x_shape);
    let w = leaf(42, &w_shape);
    let b = leaf(43, &b_shape);

    let mut cpu_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    cpu_model
        .apply_parameters(vec![w.clone(), b.clone()])
        .unwrap();
    let cpu_out = cpu_model.predict(&x).unwrap();

    let metal_tape = fandhe_ai::tape_for(fandhe_ai::Device::Metal)
        .expect("Metal 実機必須（本テストは #[ignore]）");
    let mut metal_model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    metal_model.apply_parameters(vec![w, b]).unwrap();
    let xv = metal_tape.var(&x);
    let metal_out = metal_model.forward(&metal_tape, &xv).unwrap().to_tensor();

    assert_parity("Metal vs CPU", &dense(&metal_out), &dense(&cpu_out));
}
// --- backward（イシュー #1771。既存 forward のみだった `{cuda,metal}_
// sequential_conv2d_matches_cpu` の backward 版） ---

/// `device` 上で conv2d backward（forward → `mse_loss` → `backward`）を
/// 実行し `(weight_grad, bias_grad, input_grad)` を返す（イシュー
/// #1771。`cpu_sequential_conv2d_backward_matches_naive_conv2d_layer`
/// と同型構成。CUDA／Metal から共用する）。
fn sequential_conv2d_backward_on(device: Device) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let x_shape = [2usize, 2, 4, 4];
    let w_shape = [3usize, 2, 3, 3];
    let b_shape = [3usize];
    let target_shape = [2usize, 3, 4, 4];

    let x = leaf(21, &x_shape);
    let w = leaf(22, &w_shape);
    let b = leaf(23, &b_shape);
    let target = leaf(24, &target_shape);

    let mut model = Sequential::new()
        .add_conv2d(2, 3, [3, 3], [1, 1], [1, 1], [1, 1], 1, 1)
        .unwrap();
    model.apply_parameters(vec![w, b]).unwrap();

    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let grad_refs = bound.trainable_grads(&grads).unwrap();
    let input_grad = grads.get(&xv).unwrap().expect("到達する");

    (
        grad_refs[0].clone(),
        grad_refs[1].clone(),
        input_grad.clone(),
    )
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ"]
fn cuda_sequential_conv2d_backward_matches_cpu() {
    let (w_cuda, b_cuda, dx_cuda) = sequential_conv2d_backward_on(Device::Cuda(0));
    let (w_cpu, b_cpu, dx_cpu) = sequential_conv2d_backward_on(Device::Cpu);

    assert_parity("weight_grad: CUDA vs CPU", &dense(&w_cuda), &dense(&w_cpu));
    assert_parity("bias_grad: CUDA vs CPU", &dense(&b_cuda), &dense(&b_cpu));
    assert_parity("input_grad: CUDA vs CPU", &dense(&dx_cuda), &dense(&dx_cpu));

    print_fold_bits(
        "cuda_sequential_conv2d_backward_matches_cpu[weight_grad]",
        &w_cuda,
    );
    print_fold_bits(
        "cuda_sequential_conv2d_backward_matches_cpu[bias_grad]",
        &b_cuda,
    );
    print_fold_bits(
        "cuda_sequential_conv2d_backward_matches_cpu[input_grad]",
        &dx_cuda,
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ"]
fn metal_sequential_conv2d_backward_matches_cpu() {
    let (w_metal, b_metal, dx_metal) = sequential_conv2d_backward_on(Device::Metal);
    let (w_cpu, b_cpu, dx_cpu) = sequential_conv2d_backward_on(Device::Cpu);

    assert_parity(
        "weight_grad: Metal vs CPU",
        &dense(&w_metal),
        &dense(&w_cpu),
    );
    assert_parity("bias_grad: Metal vs CPU", &dense(&b_metal), &dense(&b_cpu));
    assert_parity(
        "input_grad: Metal vs CPU",
        &dense(&dx_metal),
        &dense(&dx_cpu),
    );

    print_fold_bits(
        "metal_sequential_conv2d_backward_matches_cpu[weight_grad]",
        &w_metal,
    );
    print_fold_bits(
        "metal_sequential_conv2d_backward_matches_cpu[bias_grad]",
        &b_metal,
    );
    print_fold_bits(
        "metal_sequential_conv2d_backward_matches_cpu[input_grad]",
        &dx_metal,
    );
}

// --- Conv1d（nn 層。イシュー #1771。`conv1d_backend_parity.rs` は
// `Var::conv1d` 直叩きのみだったため nn 層〈`compat::Sequential`〉
// 経由の版を補う） ---

fn sequential_conv1d_forward_on(device: Device) -> Tensor<f32> {
    let x_shape = [1usize, 2, 9];
    let w_shape = [3usize, 2, 3];
    let b_shape = [3usize];

    let x = leaf(51, &x_shape);
    let w = leaf(52, &w_shape);
    let b = leaf(53, &b_shape);

    let mut model = Sequential::new()
        .add_conv1d(2, 3, 3, 1, 1, 1, 1, 1)
        .unwrap();
    model.apply_parameters(vec![w, b]).unwrap();

    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let xv = tape.var(&x);
    model.forward(&tape, &xv).unwrap().to_tensor()
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ"]
fn cuda_sequential_conv1d_forward_matches_cpu() {
    let cuda_out = sequential_conv1d_forward_on(Device::Cuda(0));
    let cpu_out = sequential_conv1d_forward_on(Device::Cpu);

    assert_parity(
        "conv1d forward（nn 層）: CUDA vs CPU",
        &dense(&cuda_out),
        &dense(&cpu_out),
    );
    print_fold_bits("cuda_sequential_conv1d_forward_matches_cpu[out]", &cuda_out);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ"]
fn metal_sequential_conv1d_forward_matches_cpu() {
    let metal_out = sequential_conv1d_forward_on(Device::Metal);
    let cpu_out = sequential_conv1d_forward_on(Device::Cpu);

    assert_parity(
        "conv1d forward（nn 層）: Metal vs CPU",
        &dense(&metal_out),
        &dense(&cpu_out),
    );
    print_fold_bits(
        "metal_sequential_conv1d_forward_matches_cpu[out]",
        &metal_out,
    );
}

/// `device` 上で conv1d backward（nn 層。forward → `mse_loss` →
/// `backward`）を実行し `(weight_grad, bias_grad, input_grad)` を返す
/// （イシュー #1771。CUDA／Metal から共用する）。
fn sequential_conv1d_backward_on(device: Device) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let x_shape = [1usize, 2, 9];
    let w_shape = [3usize, 2, 3];
    let b_shape = [3usize];
    let target_shape = [1usize, 3, 9];

    let x = leaf(61, &x_shape);
    let w = leaf(62, &w_shape);
    let b = leaf(63, &b_shape);
    let target = leaf(64, &target_shape);

    let mut model = Sequential::new()
        .add_conv1d(2, 3, 3, 1, 1, 1, 1, 1)
        .unwrap();
    model.apply_parameters(vec![w, b]).unwrap();

    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let grad_refs = bound.trainable_grads(&grads).unwrap();
    let input_grad = grads.get(&xv).unwrap().expect("到達する");

    (
        grad_refs[0].clone(),
        grad_refs[1].clone(),
        input_grad.clone(),
    )
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ"]
fn cuda_sequential_conv1d_backward_matches_cpu() {
    let (w_cuda, b_cuda, dx_cuda) = sequential_conv1d_backward_on(Device::Cuda(0));
    let (w_cpu, b_cpu, dx_cpu) = sequential_conv1d_backward_on(Device::Cpu);

    assert_parity(
        "conv1d weight_grad（nn 層）: CUDA vs CPU",
        &dense(&w_cuda),
        &dense(&w_cpu),
    );
    assert_parity(
        "conv1d bias_grad（nn 層）: CUDA vs CPU",
        &dense(&b_cuda),
        &dense(&b_cpu),
    );
    assert_parity(
        "conv1d input_grad（nn 層）: CUDA vs CPU",
        &dense(&dx_cuda),
        &dense(&dx_cpu),
    );

    print_fold_bits(
        "cuda_sequential_conv1d_backward_matches_cpu[weight_grad]",
        &w_cuda,
    );
    print_fold_bits(
        "cuda_sequential_conv1d_backward_matches_cpu[bias_grad]",
        &b_cuda,
    );
    print_fold_bits(
        "cuda_sequential_conv1d_backward_matches_cpu[input_grad]",
        &dx_cuda,
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ"]
fn metal_sequential_conv1d_backward_matches_cpu() {
    let (w_metal, b_metal, dx_metal) = sequential_conv1d_backward_on(Device::Metal);
    let (w_cpu, b_cpu, dx_cpu) = sequential_conv1d_backward_on(Device::Cpu);

    assert_parity(
        "conv1d weight_grad（nn 層）: Metal vs CPU",
        &dense(&w_metal),
        &dense(&w_cpu),
    );
    assert_parity(
        "conv1d bias_grad（nn 層）: Metal vs CPU",
        &dense(&b_metal),
        &dense(&b_cpu),
    );
    assert_parity(
        "conv1d input_grad（nn 層）: Metal vs CPU",
        &dense(&dx_metal),
        &dense(&dx_cpu),
    );

    print_fold_bits(
        "metal_sequential_conv1d_backward_matches_cpu[weight_grad]",
        &w_metal,
    );
    print_fold_bits(
        "metal_sequential_conv1d_backward_matches_cpu[bias_grad]",
        &b_metal,
    );
    print_fold_bits(
        "metal_sequential_conv1d_backward_matches_cpu[input_grad]",
        &dx_metal,
    );
}

// --- Conv1d の「特化」契約（nn 層版。イシュー #1771。
// `conv1d_backend_parity.rs::conv1d_matches_manual_reshape_conv2d_on`
// の nn 層〈`compat::Sequential`〉版）: 同一 GPU tape 上で
// `add_conv1d` モデルと `add_conv2d`（`[1,k]`・`[0,p]`・`[1,d]`）
// モデルへ同一重みを注入し、forward・勾配とも bit 完全一致することを
// 確認する。`conv1d` は reshape 併合のみで新規カーネルを持たないため
// 機構的に成立する契約。 ---

fn sequential_conv1d_matches_manual_reshape_conv2d_on(device: Device) {
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

    let x_data = leaf(71, &[n, cin, l]);
    let w_data = leaf(72, &[cout, cin_g, k]);
    let b_data = leaf(73, &[cout]);

    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");

    // conv1d 経路（nn 層）。
    let mut model1 = Sequential::new()
        .add_conv1d(cin, cout, k, stride, padding, dilation, groups, 1)
        .unwrap();
    model1
        .apply_parameters(vec![w_data.clone(), b_data.clone()])
        .unwrap();
    let bound1 = model1.bind(&tape);
    let x1 = tape.var(&x_data);
    let y1 = bound1.forward(&tape, &x1).unwrap();
    let out1 = y1.to_tensor();
    let loss1 = y1.sum(None).expect("sum: 常に成功する");
    let grads1 = tape.backward(&loss1).unwrap();
    let grad_refs1 = bound1.trainable_grads(&grads1).unwrap();
    let dx1 = grads1.get(&x1).unwrap().expect("到達する").clone();
    let dw1 = grad_refs1[0].clone();
    let db1 = grad_refs1[1].clone();

    // 手動 reshape -> conv2d 経路（同一 tape 上・別 leaf ノード・
    // 別モデル）。
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
    let mut model2 = Sequential::new()
        .add_conv2d(
            cin,
            cout,
            [1, k],
            [1, stride],
            [0, padding],
            [1, dilation],
            groups,
            2,
        )
        .unwrap();
    model2.apply_parameters(vec![w4_data, b_data]).unwrap();
    let bound2 = model2.bind(&tape);
    let x2 = tape.var(&x4_data);
    let y2 = bound2.forward(&tape, &x2).unwrap();
    let out2 = y2.to_tensor();
    let loss2 = y2.sum(None).expect("sum: 常に成功する");
    let grads2 = tape.backward(&loss2).unwrap();
    let grad_refs2 = bound2.trainable_grads(&grads2).unwrap();
    let dx2 = grads2.get(&x2).unwrap().expect("到達する").clone();
    let dw2 = grad_refs2[0].clone();
    let db2 = grad_refs2[1].clone();

    assert_bits_eq(
        "nn: conv1d vs manual-reshape conv2d（forward）",
        &dense(&out1),
        &dense(&out2),
    );
    assert_bits_eq(
        "nn: conv1d vs manual-reshape conv2d（d_input）",
        &dense(&dx1),
        &dense(&dx2),
    );
    assert_bits_eq(
        "nn: conv1d vs manual-reshape conv2d（d_weight）",
        &dense(&dw1),
        &dense(&dw2),
    );
    assert_bits_eq(
        "nn: conv1d vs manual-reshape conv2d（d_bias）",
        &dense(&db1),
        &dense(&db2),
    );

    print_fold_bits(
        "sequential_conv1d_matches_manual_reshape_conv2d[forward]",
        &out1,
    );
    print_fold_bits(
        "sequential_conv1d_matches_manual_reshape_conv2d[d_input]",
        &dx1,
    );
    print_fold_bits(
        "sequential_conv1d_matches_manual_reshape_conv2d[d_weight]",
        &dw1,
    );
    print_fold_bits(
        "sequential_conv1d_matches_manual_reshape_conv2d[d_bias]",
        &db1,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ"]
fn cuda_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact() {
    sequential_conv1d_matches_manual_reshape_conv2d_on(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ"]
fn metal_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact() {
    sequential_conv1d_matches_manual_reshape_conv2d_on(Device::Metal);
}

// --- 学習ループ（record-only。イシュー #1771）: ADOPT／REJECT の
// 判定対象にしない（GEMM 由来の差が step をまたいで累積しうるため）。
// 各 step の loss・最終パラメータの REQ-2 複合判定結果・bit fold を
// 記録するのみ。`compat_sequential_conv.rs::
// train_loop_with_sgd_reduces_loss` と同一形状・同一 SGD 設定・同一
// seed（CPU／GPU で同一初期重みにするため。`Conv2d::new` の重み初期化
// はホスト側 RNG のため device に依存しない）。 ---

/// `device` 上で conv 単体モデルを `STEPS` step SGD で学習し、各 step
/// の loss を `<label>[step<i>].loss.bits=<hex>` 形式で `println!` した
/// うえで最終パラメータを返す（**record-only**。イシュー #1771）。
fn sgd_steps_on(device: Device, label: &str) -> Vec<Tensor<f32>> {
    const STEPS: usize = 5;
    const SEED: u64 = 0x1234_5678;

    let mut model = Sequential::new()
        .add_conv2d(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, SEED)
        .unwrap();
    let x = leaf(81, &[2, 2, 4, 4]);
    let target = leaf(82, &[2, 4, 4, 4]);

    let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();

    for step in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape_for(device)
                .expect("実機が利用可能な前提のテストのため成功するはず");
            let bound = model.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let pred = bound.forward(&tape, &xv).unwrap();
            let loss = pred.mse_loss(&tv).unwrap();
            let loss_val = loss.to_tensor().get(&[]).unwrap();
            println!("{label}[step{step}].loss.bits={:#010x}", loss_val.to_bits());

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    model.trainable_parameters().into_iter().cloned().collect()
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実行は #1771 へ引き継ぎ。record-only（ADOPT／REJECT 非対象）"]
fn cuda_sequential_conv2d_sgd_steps_record_only() {
    let cuda_params = sgd_steps_on(
        Device::Cuda(0),
        "cuda_sequential_conv2d_sgd_steps_record_only.cuda",
    );
    let cpu_params = sgd_steps_on(
        Device::Cpu,
        "cuda_sequential_conv2d_sgd_steps_record_only.cpu",
    );

    assert_eq!(
        cuda_params.len(),
        cpu_params.len(),
        "パラメータ件数（weight/bias）が一致しない"
    );
    for (i, (c, p)) in cuda_params.iter().zip(cpu_params.iter()).enumerate() {
        // record-only（README「事前登録判定規則」5)）: ADOPT／REJECT の
        // 判定対象にしないため `assert_parity` ではなく `record_parity`
        // で判定結果のみ記録する（panic させると以降の `fold_bits` 行が
        // 欠落する。PR #1882 レビュー指摘）。
        record_parity(
            &format!("final_param[{i}]（record-only）: CUDA vs CPU"),
            &dense(c),
            &dense(p),
        );
        print_fold_bits(
            &format!("cuda_sequential_conv2d_sgd_steps_record_only[final_param_{i}]"),
            c,
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。実行は #1771 へ引き継ぎ。record-only（ADOPT／REJECT 非対象）"]
fn metal_sequential_conv2d_sgd_steps_record_only() {
    let metal_params = sgd_steps_on(
        Device::Metal,
        "metal_sequential_conv2d_sgd_steps_record_only.metal",
    );
    let cpu_params = sgd_steps_on(
        Device::Cpu,
        "metal_sequential_conv2d_sgd_steps_record_only.cpu",
    );

    assert_eq!(
        metal_params.len(),
        cpu_params.len(),
        "パラメータ件数（weight/bias）が一致しない"
    );
    for (i, (m, p)) in metal_params.iter().zip(cpu_params.iter()).enumerate() {
        // record-only（README「事前登録判定規則」5)）: ADOPT／REJECT の
        // 判定対象にしないため `assert_parity` ではなく `record_parity`
        // で判定結果のみ記録する（panic させると以降の `fold_bits` 行が
        // 欠落する。PR #1882 レビュー指摘）。
        record_parity(
            &format!("final_param[{i}]（record-only）: Metal vs CPU"),
            &dense(m),
            &dense(p),
        );
        print_fold_bits(
            &format!("metal_sequential_conv2d_sgd_steps_record_only[final_param_{i}]"),
            m,
        );
    }
}
