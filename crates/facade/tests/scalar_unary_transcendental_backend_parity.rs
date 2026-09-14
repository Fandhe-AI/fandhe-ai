//! `Var::log`／`log2`／`log10`／`sin`／`cos`／`tan`／`abs`／`neg`
//! （イシュー #1711）の facade 到達経路（既存 `Var` 再エクスポート経由。
//! `crates/facade/src/lib.rs` への新規 `pub use`／`pub fn` は追加してい
//! ない）の受け入れ条件対応テスト。`index_ops_backend_parity.rs`・
//! `shape_ops_backend_parity.rs` と同型の構成を踏襲する。
//!
//! バックエンド 3 クレート（`backend-cpu`／`backend-cuda`／
//! `backend-metal`）・`ScalarUnaryOp` enum・dispatch・CPU 参照実装・VJP
//! は #1592／#1634／#1635／#1636／#1700〜#1709 で実装済みのため、
//! 本ファイルは `Var` の新規 8 メソッドから既存カーネルへ到達できる
//! ことのみを検証する（新規カーネル実装は含まない）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::scalar_unary`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps` → `eval::scalar::
//!   unary` フォールバック）で forward／backward を REQ-2 統一複合
//!   判定（`assert_parity`）で突き合わせる。`neg`／`abs` は符号ビット
//!   演算（丸めを伴わない）のため bit 同一も併記する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。本エージェント実行環境に実機への到達手段がないため
//!   未実測のまま Mac／GB10 セッションへ申し送る。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`index_ops_backend_parity.rs`
/// の `VarSource` と同じ理由・同じ構成）。
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

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

/// `[-1, 1)` の一様乱数（`sin`／`cos`／`tan`／`abs`／`neg` 用。`tan` の
/// 極 `π/2 ≈ 1.57` を含まない範囲）。
fn general_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("general_leaf: shape 一致")
}

/// `[0.1, 1.1)` の正の一様乱数（`log`／`log2`／`log10` 用。定義域を
/// 厳密に正へ保つ）。
fn positive_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v * 0.5 + 0.6) // [-1,1) -> [0.1, 1.1)
        .collect();
    Tensor::new(data, shape).expect("positive_leaf: shape 一致")
}

// --- forward parity（属性なし: CPU vs NaiveOps） ---

macro_rules! unary_forward_parity_test {
    ($name:ident, $method:ident, $leaf_fn:ident, $bit_exact:expr) => {
        #[test]
        fn $name() {
            let shape = [2usize, 3];
            let x_val = $leaf_fn(11, &shape);

            let cpu_tape = fandhe_ai::tape();
            let x_cpu = cpu_tape.make_var(&x_val);
            let out_cpu = x_cpu.$method().unwrap().to_tensor();

            let naive_tape = fandhe_ai_autodiff::Tape::new();
            let x_naive = naive_tape.make_var(&x_val);
            let out_naive = x_naive.$method().unwrap().to_tensor();

            let cpu_slice = contiguous_slice(&out_cpu);
            let naive_slice = contiguous_slice(&out_naive);
            assert_parity(
                concat!(
                    "fandhe_ai::tape()（CpuBackendOps::scalar_unary::",
                    stringify!($method),
                    "）vs NaiveOps"
                ),
                &cpu_slice,
                &naive_slice,
            );
            if $bit_exact {
                assert_eq!(
                    cpu_slice, naive_slice,
                    concat!(stringify!($method), ": 符号ビット演算のため bit 同一のはず")
                );
            }
        }
    };
}

unary_forward_parity_test!(
    cpu_log_forward_matches_naive_reference,
    log,
    positive_leaf,
    false
);
unary_forward_parity_test!(
    cpu_log2_forward_matches_naive_reference,
    log2,
    positive_leaf,
    false
);
unary_forward_parity_test!(
    cpu_log10_forward_matches_naive_reference,
    log10,
    positive_leaf,
    false
);
unary_forward_parity_test!(
    cpu_sin_forward_matches_naive_reference,
    sin,
    general_leaf,
    false
);
unary_forward_parity_test!(
    cpu_cos_forward_matches_naive_reference,
    cos,
    general_leaf,
    false
);
unary_forward_parity_test!(
    cpu_tan_forward_matches_naive_reference,
    tan,
    general_leaf,
    false
);
unary_forward_parity_test!(
    cpu_abs_forward_matches_naive_reference,
    abs,
    general_leaf,
    true
);
unary_forward_parity_test!(
    cpu_neg_forward_matches_naive_reference,
    neg,
    general_leaf,
    true
);

// --- backward 合成（属性なし: CPU vs NaiveOps） ---

/// `matmul → log → sum` の CPU と NaiveOps の parity（`dW`／`dx`）。
/// `x`・`w` とも `positive_leaf`（[0.1,1.1)）を使い matmul 出力を厳密に
/// 正へ保つ（`general_leaf` だと負の要素が `log` で `NaN` になり
/// 勾配も `NaN` になって `assert_parity` が成立しないため）。
#[test]
fn cpu_matmul_log_backward_matches_naive_reference() {
    let x_shape = [3usize, 2];
    let w_shape = [2usize, 4];

    let x_val = positive_leaf(21, &x_shape);
    let w_val = positive_leaf(22, &w_shape);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_val);
    let w_cpu = cpu_tape.make_var(&w_val);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().log().unwrap();
    let loss_cpu = y_cpu.sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().expect("到達する");
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_val);
    let w_naive = naive_tape.make_var(&w_val);
    let y_naive = x_naive.matmul(&w_naive).unwrap().log().unwrap();
    let loss_naive = y_naive.sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().expect("到達する");
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "matmul→log backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dx_cpu),
        &contiguous_slice(dx_naive),
    );
    assert_parity(
        "matmul→log backward（dW）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(dw_cpu),
        &contiguous_slice(dw_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn unary_forward_on(
    device: Device,
    x: &Tensor<f32>,
    apply: impl for<'a> Fn(&'a Var<'a>) -> Var<'a>,
) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let v = tape.make_var(x);
    apply(&v).to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある
// （`index_ops_backend_parity.rs` と同じ理由）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_log_forward_matches_cpu() {
    let x = positive_leaf(11, &[2, 3]);
    let metal_out = unary_forward_on(Device::Metal, &x, |v| v.log().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.log().unwrap());
    assert_parity(
        "log forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sin_forward_matches_cpu() {
    let x = general_leaf(11, &[2, 3]);
    let metal_out = unary_forward_on(Device::Metal, &x, |v| v.sin().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.sin().unwrap());
    assert_parity(
        "sin forward: Metal tape_for vs CPU tape_for",
        &contiguous_slice(&metal_out),
        &contiguous_slice(&cpu_out),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_abs_and_neg_forward_matches_cpu_bit_exact() {
    let x = general_leaf(11, &[2, 3]);
    let metal_abs = unary_forward_on(Device::Metal, &x, |v| v.abs().unwrap());
    let cpu_abs = unary_forward_on(Device::Cpu, &x, |v| v.abs().unwrap());
    assert_eq!(
        contiguous_slice(&metal_abs),
        contiguous_slice(&cpu_abs),
        "abs: 符号ビット演算のため bit 同一のはず"
    );

    let metal_neg = unary_forward_on(Device::Metal, &x, |v| v.neg().unwrap());
    let cpu_neg = unary_forward_on(Device::Cpu, &x, |v| v.neg().unwrap());
    assert_eq!(
        contiguous_slice(&metal_neg),
        contiguous_slice(&cpu_neg),
        "neg: 符号ビット演算のため bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_log_forward_matches_cpu() {
    let x = positive_leaf(11, &[2, 3]);
    let cuda_out = unary_forward_on(Device::Cuda(0), &x, |v| v.log().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.log().unwrap());
    assert_parity(
        "log forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sin_forward_matches_cpu() {
    let x = general_leaf(11, &[2, 3]);
    let cuda_out = unary_forward_on(Device::Cuda(0), &x, |v| v.sin().unwrap());
    let cpu_out = unary_forward_on(Device::Cpu, &x, |v| v.sin().unwrap());
    assert_parity(
        "sin forward: CUDA tape_for vs CPU tape_for",
        &contiguous_slice(&cuda_out),
        &contiguous_slice(&cpu_out),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_abs_and_neg_forward_matches_cpu_bit_exact() {
    let x = general_leaf(11, &[2, 3]);
    let cuda_abs = unary_forward_on(Device::Cuda(0), &x, |v| v.abs().unwrap());
    let cpu_abs = unary_forward_on(Device::Cpu, &x, |v| v.abs().unwrap());
    assert_eq!(
        contiguous_slice(&cuda_abs),
        contiguous_slice(&cpu_abs),
        "abs: 符号ビット演算のため bit 同一のはず"
    );

    let cuda_neg = unary_forward_on(Device::Cuda(0), &x, |v| v.neg().unwrap());
    let cpu_neg = unary_forward_on(Device::Cpu, &x, |v| v.neg().unwrap());
    assert_eq!(
        contiguous_slice(&cuda_neg),
        contiguous_slice(&cpu_neg),
        "neg: 符号ビット演算のため bit 同一のはず"
    );
}
