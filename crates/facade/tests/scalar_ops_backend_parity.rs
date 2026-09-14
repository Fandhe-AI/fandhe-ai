//! `Var::sub`／`div`／`pow`／`sqrt`（イシュー #1710・親 #1593）・
//! `clamp`／比較演算 6 種（`gt`／`ge`／`lt`／`le`／`eq`／`ne`。イシュー
//! #1712・親 #1593）の facade 到達経路（既存 `Var` 再エクスポート経由。
//! 新規 `pub use`／`pub fn` は追加していない）の受け入れ条件対応テスト
//! （`softmax_backend_parity.rs`・`index_ops_backend_parity.rs` と
//! 同型）。
//!
//! いずれの演算も `Var::scalar_binary`／`scalar_unary`
//! （`ScalarBinaryOp`／`ScalarUnaryOp`。イシュー #1634）への薄い委譲
//! であり、forward・VJP 係数の解析値と数値微分（中央差分）の突合は
//! `crates/autodiff/src/grad.rs` の `#[cfg(test)]` 側（`unary_numeric_
//! grad_cases`／`binary_numeric_grad_cases` の `Sqrt`／`Sub`／`Div`／
//! `Pow`／`Clamp`／比較 6 種の行。#1710・#1712・#1634／#1686）で既に
//! 検証済みのため、本ファイルでは facade 到達経路（`fandhe_ai::
//! tape()`／`tape_for(Device)`）に限定した 3 バックエンド parity のみ
//! を担う。比較演算・`clamp` は出力が f32 の `0.0`／`1.0` または入力の
//! 選択（算術を含まない）のためバックエンド間で bit 同一となる
//! （`crates/backend-{cuda,metal}/tests/scalar_op_parity.rs` モジュール
//! doc と同じ主張）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps::scalar_unary`／
//!   `scalar_binary`）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`。
//!   既定 `Unsupported` のためホスト参照実装 `eval::scalar::unary`／
//!   `binary` へフォールバックする経路）の forward・backward を
//!   REQ-2 統一複合判定で突き合わせる。`Sub`／`Div`／`Sqrt`／`Clamp`／
//!   比較 6 種は IEEE 754 丸め契約・純粋な選択演算により bit 同一
//!   （`crates/backend-{cuda,metal}/tests/scalar_op_parity.rs` モジュール
//!   doc と同じ主張）のため forward の bit 同一も併記する（超越関数の
//!   `Pow` は複合判定のみ）。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` の同経路を CPU tape
//!   と比較する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（既存 parity テストと同じ
/// 構成）。
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

/// 一般値（`sub`。0 近傍・符号混在を含む）。
fn general_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel); // [-1, 1)
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

/// 正の値のみ（`div` の分母・`pow` の底・`sqrt` の入力。0 除算・
/// 定義域外を避ける。`scalar_op_parity.rs::positive_data` と同じ
/// オフセット方針）。
fn positive_leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v.abs() + 0.1) // [0.1, 1.1)
        .collect();
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

/// 比較演算（`gt`／`ge`／`lt`／`le`／`eq`／`ne`）用の `(a, b)` ペアを
/// 生成する。独立乱数 2 本では `eq` が全 `0.0`・`ne` が全 `1.0` になり
/// 検証にならないため、`a` から意図的に等値・やや大きい値・やや小さい
/// 値を混在させて派生させる（`crates/backend-cuda/tests/
/// scalar_op_parity.rs::comparison_pair_data` と同じ方針。イシュー
/// #1712）。
fn comparison_pair(seed: u64, shape: &[usize]) -> (Tensor<f32>, Tensor<f32>) {
    let a_data = general_leaf(seed, shape);
    let a_slice = contiguous_slice(&a_data);
    let b_slice: Vec<f32> = a_slice
        .iter()
        .enumerate()
        .map(|(i, &v)| match i % 3 {
            0 => v,        // 等値ケース
            1 => v + 0.25, // a < b
            _ => v - 0.25, // a > b
        })
        .collect();
    let b_data = Tensor::new(b_slice, shape).expect("valid tensor");
    (a_data, b_data)
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- 属性なし（CPU vs NaiveOps） ---

#[test]
fn cpu_sub_forward_matches_naive_reference_bit_exact() {
    let a_data = general_leaf(0xC0FFEE, &[2, 3]);
    let b_data = general_leaf(0xBEEF, &[2, 3]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&a_data)
        .sub(&cpu_tape.make_var(&b_data))
        .expect("同 shape の sub は成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&a_data)
        .sub(&naive_tape.make_var(&b_data))
        .expect("同 shape の sub は成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "sub: IEEE 754 丸め契約により CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

#[test]
fn cpu_div_forward_matches_naive_reference_bit_exact() {
    let a_data = general_leaf(0x1234, &[2, 3]);
    let b_data = positive_leaf(0x5678, &[2, 3]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&a_data)
        .div(&cpu_tape.make_var(&b_data))
        .expect("同 shape の div は成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&a_data)
        .div(&naive_tape.make_var(&b_data))
        .expect("同 shape の div は成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "div: IEEE 754 丸め契約により CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

#[test]
fn cpu_pow_forward_matches_naive_reference() {
    let a_data = positive_leaf(0xA1B2, &[2, 3]);
    let b_data = general_leaf(0xC3D4, &[2, 3]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&a_data)
        .pow(&cpu_tape.make_var(&b_data))
        .expect("同 shape の pow は成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&a_data)
        .pow(&naive_tape.make_var(&b_data))
        .expect("同 shape の pow は成功するはず")
        .to_tensor();

    assert_parity(
        "pow forward: CpuBackendOps vs NaiveOps フォールバック",
        &contiguous_slice(&cpu_out),
        &contiguous_slice(&naive_out),
    );
}

#[test]
fn cpu_sqrt_forward_matches_naive_reference_bit_exact() {
    let x_data = positive_leaf(0xF00D, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&x_data)
        .sqrt()
        .expect("sqrt は shape 不変で常に成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&x_data)
        .sqrt()
        .expect("sqrt は shape 不変で常に成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "sqrt: IEEE 754 丸め契約により CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

#[test]
fn cpu_clamp_forward_matches_naive_reference_bit_exact() {
    let x_data = general_leaf(0xCA11, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let cpu_out = cpu_tape
        .make_var(&x_data)
        .clamp(-0.5, 0.5)
        .expect("clamp は shape 不変で常に成功するはず")
        .to_tensor();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let naive_out = naive_tape
        .make_var(&x_data)
        .clamp(-0.5, 0.5)
        .expect("clamp は shape 不変で常に成功するはず")
        .to_tensor();

    assert_eq!(
        contiguous_slice(&cpu_out),
        contiguous_slice(&naive_out),
        "clamp: 算術を含まない純粋な選択演算のため CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
    );
}

// `ScalarBinaryOp` は autodiff クレート内部でのみ扱う（`Var::
// scalar_binary` は `pub(crate)`）ため、facade からは 6 種の個別公開
// メソッドを名前付き関数でラップして列挙・ループする。`Var::gt` 等の
// UFCS 参照をそのまま関数ポインタ配列へ入れると `for<'t> fn(&Var<'t>,
// &Var<'t>) -> ...`（`CmpFn`）へ強制変換できない——`'t` は
// `impl<'t> Var<'t>` の early-bound ライフトイムパラメータであり、
// メソッド自身の signature にのみ現れる late-bound パラメータではない
// ため、UFCS で取り出した関数項の型は特定の 't に固定されてしまい
// 高階トレイト境界（HRTB）へ汎化できない（実機確認済み: `("gt",
// Var::gt)` は E0308 "one type is more general than the other" で
// 拒否される）。そのため 't を明示的に汎化した通常の `fn` として
// ラップし、その関数項（HRTB を持つ）を配列へ格納する。
type CmpFn = for<'t> fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError>;
fn call_gt<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    a.gt(b)
}
fn call_ge<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    a.ge(b)
}
fn call_lt<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    a.lt(b)
}
fn call_le<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    a.le(b)
}
fn call_eq<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    a.eq(b)
}
fn call_ne<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, fandhe_ai_autodiff::AutodiffError> {
    a.ne(b)
}

#[test]
fn cpu_comparisons_forward_match_naive_reference_bit_exact() {
    let shape = [2, 3];
    let (a_data, b_data) = comparison_pair(0xC0DE, &shape);

    let ops: [(&str, CmpFn); 6] = [
        ("gt", call_gt),
        ("ge", call_ge),
        ("lt", call_lt),
        ("le", call_le),
        ("eq", call_eq),
        ("ne", call_ne),
    ];

    for (name, op) in ops {
        let cpu_tape = fandhe_ai::tape();
        let cpu_out = op(&cpu_tape.make_var(&a_data), &cpu_tape.make_var(&b_data))
            .unwrap_or_else(|e| panic!("{name}: 同 shape の比較は成功するはず: {e:?}"))
            .to_tensor();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let naive_out = op(&naive_tape.make_var(&a_data), &naive_tape.make_var(&b_data))
            .unwrap_or_else(|e| panic!("{name}: 同 shape の比較は成功するはず: {e:?}"))
            .to_tensor();

        assert_eq!(
            contiguous_slice(&cpu_out),
            contiguous_slice(&naive_out),
            "{name}: 比較演算は 0.0/1.0 の純粋な比較のため CpuBackendOps と NaiveOps フォールバックは bit 同一のはず"
        );
    }
}

/// `matmul → clamp → mse_loss`（clamp を epilogue に挟む合成経路）の
/// backward（`dW`）と、`matmul → gt` を含む合成経路の入力勾配が両
/// tape でゼロのまま一致することの parity（イシュー #1712）。
#[test]
fn cpu_clamp_and_comparison_backward_matches_naive_reference() {
    let w = Tensor::new(
        vec![0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3, 0.9],
        &[5, 2],
    )
    .expect("valid tensor");
    let target = Tensor::new(vec![0.2, 0.6, 0.3, 0.4, 0.1, 0.5], &[3, 2]).expect("valid tensor");
    let x_data = general_leaf(0x1357_9BDF, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let w_cpu = cpu_tape.make_var(&w);
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().clamp(-0.5, 0.5).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let w_naive = naive_tape.make_var(&w);
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive.matmul(&w_naive).unwrap().clamp(-0.5, 0.5).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");

    assert_parity(
        "clamp backward（dW）: CpuBackendOps vs NaiveOps",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );

    // gt を含む合成経路: 比較側の勾配は常にゼロのため mask 自体の
    // 勾配追跡は不要だが、mask を乗じた経路の入力勾配が両 tape で
    // 一致することを確認する（`x * (x > 0)` 型のマスク合成）。
    let cpu_tape2 = fandhe_ai::tape();
    let x2_cpu = cpu_tape2.make_var(&x_data);
    let zero_cpu = cpu_tape2.make_var(&Tensor::new(vec![0.0; 15], &[3, 5]).unwrap());
    let mask_cpu = x2_cpu.gt(&zero_cpu).unwrap();
    let masked_cpu = x2_cpu.mul(&mask_cpu).unwrap();
    let loss2_cpu = masked_cpu.sum(None).unwrap();
    let grads2_cpu = cpu_tape2.backward(&loss2_cpu).unwrap();
    let dx2_cpu = grads2_cpu.get(&x2_cpu).unwrap().expect("到達する");

    let naive_tape2 = fandhe_ai_autodiff::Tape::new();
    let x2_naive = naive_tape2.make_var(&x_data);
    let zero_naive = naive_tape2.make_var(&Tensor::new(vec![0.0; 15], &[3, 5]).unwrap());
    let mask_naive = x2_naive.gt(&zero_naive).unwrap();
    let masked_naive = x2_naive.mul(&mask_naive).unwrap();
    let loss2_naive = masked_naive.sum(None).unwrap();
    let grads2_naive = naive_tape2.backward(&loss2_naive).unwrap();
    let dx2_naive = grads2_naive.get(&x2_naive).unwrap().expect("到達する");

    assert_eq!(
        contiguous_slice(dx2_cpu),
        contiguous_slice(dx2_naive),
        "mask 合成（gt のゼロ勾配経由）: CpuBackendOps と NaiveOps は入力勾配が bit 同一のはず"
    );
}

/// `matmul → sub → mse_loss` backward（`d_input`／`dW`／`dbias`）の
/// CPU（`Op::ScalarBinary` の VJP。#1634）と NaiveOps の parity
/// （broadcast bias 縮約を含む合成経路）。
#[test]
fn cpu_scalar_ops_backward_matches_naive_reference_with_broadcast() {
    let w = Tensor::new(
        vec![0.5, -0.3, 0.2, 0.7, -0.6, 0.1, 0.4, -0.2, 0.3, 0.9],
        &[5, 2],
    )
    .expect("valid tensor");
    let bias = Tensor::new(vec![0.3, 0.7], &[2]).expect("valid tensor"); // sub の broadcast rhs
    let target = Tensor::new(vec![0.2, 0.6, 0.3, 0.4, 0.1, 0.5], &[3, 2]).expect("valid tensor");
    let x_data = general_leaf(0x9E37_79B9, &[3, 5]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&x_data);
    let w_cpu = cpu_tape.make_var(&w);
    let bias_cpu = cpu_tape.make_var(&bias);
    let t_cpu = cpu_tape.make_var(&target);
    let y_cpu = x_cpu.matmul(&w_cpu).unwrap().sub(&bias_cpu).unwrap();
    let loss_cpu = y_cpu.mse_loss(&t_cpu).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dw_cpu = grads_cpu.get(&w_cpu).unwrap().expect("到達する");
    let dbias_cpu = grads_cpu.get(&bias_cpu).unwrap().expect("到達する");

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&x_data);
    let w_naive = naive_tape.make_var(&w);
    let bias_naive = naive_tape.make_var(&bias);
    let t_naive = naive_tape.make_var(&target);
    let y_naive = x_naive.matmul(&w_naive).unwrap().sub(&bias_naive).unwrap();
    let loss_naive = y_naive.mse_loss(&t_naive).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dw_naive = grads_naive.get(&w_naive).unwrap().expect("到達する");
    let dbias_naive = grads_naive.get(&bias_naive).unwrap().expect("到達する");

    assert_parity(
        "sub backward（dW）: CpuBackendOps vs NaiveOps",
        dw_cpu.as_slice().expect("contiguous"),
        dw_naive.as_slice().expect("contiguous"),
    );
    assert_parity(
        "sub backward（dbias, broadcast 縮約）: CpuBackendOps vs NaiveOps",
        dbias_cpu.as_slice().expect("contiguous"),
        dbias_naive.as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---

fn sub_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = general_leaf(0x1111, &[3, 4]);
    let b = general_leaf(0x2222, &[3, 4]);
    tape.make_var(&a)
        .sub(&tape.make_var(&b))
        .expect("同 shape の sub は成功するはず")
        .to_tensor()
}

fn div_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = general_leaf(0x3333, &[3, 4]);
    let b = positive_leaf(0x4444, &[3, 4]);
    tape.make_var(&a)
        .div(&tape.make_var(&b))
        .expect("同 shape の div は成功するはず")
        .to_tensor()
}

fn pow_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let a = positive_leaf(0x5555, &[3, 4]);
    let b = general_leaf(0x6666, &[3, 4]);
    tape.make_var(&a)
        .pow(&tape.make_var(&b))
        .expect("同 shape の pow は成功するはず")
        .to_tensor()
}

fn sqrt_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = positive_leaf(0x7777, &[3, 4]);
    tape.make_var(&x)
        .sqrt()
        .expect("sqrt は shape 不変で常に成功するはず")
        .to_tensor()
}

fn clamp_forward_on(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = general_leaf(0x8888, &[3, 4]);
    tape.make_var(&x)
        .clamp(-0.5, 0.5)
        .expect("clamp は shape 不変で常に成功するはず")
        .to_tensor()
}

/// `op` は [`CmpFn`]（`call_gt` 等）。比較演算 6 種を実機横断で
/// まとめて検証するため device 生成を 1 関数に集約する（イシュー
/// #1712）。
fn comparison_forward_on(device: Device, op: CmpFn) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let (a_data, b_data) = comparison_pair(0x9999, &[3, 4]);
    op(&tape.make_var(&a_data), &tape.make_var(&b_data))
        .expect("同 shape の比較は成功するはず")
        .to_tensor()
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする必要がある。`#[ignore]`
// は実行のみをスキップしコンパイルはスキップしないため、Linux（CI の
// ubuntu-latest）では `cfg` ゲートがないと E0599 でビルド不能になる。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sub_forward_matches_cpu_bit_exact() {
    let metal_out = sub_forward_on(Device::Metal);
    let cpu_out = sub_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "sub: IEEE 754 丸め契約により Metal と CPU は bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_div_forward_matches_cpu_bit_exact() {
    let metal_out = div_forward_on(Device::Metal);
    let cpu_out = div_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "div: IEEE 754 丸め契約により Metal と CPU は bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_pow_forward_matches_cpu() {
    let metal_out = pow_forward_on(Device::Metal);
    let cpu_out = pow_forward_on(Device::Cpu);
    assert_parity(
        "pow forward: Metal tape_for vs CPU tape_for",
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_sqrt_forward_matches_cpu_bit_exact() {
    let metal_out = sqrt_forward_on(Device::Metal);
    let cpu_out = sqrt_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "sqrt: IEEE 754 丸め契約により Metal と CPU は bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_clamp_forward_matches_cpu_bit_exact() {
    let metal_out = clamp_forward_on(Device::Metal);
    let cpu_out = clamp_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&metal_out),
        contiguous_slice(&cpu_out),
        "clamp: 算術を含まない純粋な選択演算のため Metal と CPU は bit 同一のはず"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_comparisons_forward_match_cpu_bit_exact() {
    let ops: [(&str, CmpFn); 6] = [
        ("gt", call_gt),
        ("ge", call_ge),
        ("lt", call_lt),
        ("le", call_le),
        ("eq", call_eq),
        ("ne", call_ne),
    ];
    for (name, op) in ops {
        let metal_out = comparison_forward_on(Device::Metal, op);
        let cpu_out = comparison_forward_on(Device::Cpu, op);
        assert_eq!(
            contiguous_slice(&metal_out),
            contiguous_slice(&cpu_out),
            "{name}: 比較演算は 0.0/1.0 の純粋な比較のため Metal と CPU は bit 同一のはず"
        );
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sub_forward_matches_cpu_bit_exact() {
    let cuda_out = sub_forward_on(Device::Cuda(0));
    let cpu_out = sub_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "sub: IEEE 754 丸め契約により CUDA と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_div_forward_matches_cpu_bit_exact() {
    let cuda_out = div_forward_on(Device::Cuda(0));
    let cpu_out = div_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "div: IEEE 754 丸め契約により CUDA と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_pow_forward_matches_cpu() {
    let cuda_out = pow_forward_on(Device::Cuda(0));
    let cpu_out = pow_forward_on(Device::Cpu);
    assert_parity(
        "pow forward: CUDA tape_for vs CPU tape_for",
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_sqrt_forward_matches_cpu_bit_exact() {
    let cuda_out = sqrt_forward_on(Device::Cuda(0));
    let cpu_out = sqrt_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "sqrt: IEEE 754 丸め契約により CUDA と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_clamp_forward_matches_cpu_bit_exact() {
    let cuda_out = clamp_forward_on(Device::Cuda(0));
    let cpu_out = clamp_forward_on(Device::Cpu);
    assert_eq!(
        contiguous_slice(&cuda_out),
        contiguous_slice(&cpu_out),
        "clamp: 算術を含まない純粋な選択演算のため CUDA と CPU は bit 同一のはず"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_comparisons_forward_match_cpu_bit_exact() {
    let ops: [(&str, CmpFn); 6] = [
        ("gt", call_gt),
        ("ge", call_ge),
        ("lt", call_lt),
        ("le", call_le),
        ("eq", call_eq),
        ("ne", call_ne),
    ];
    for (name, op) in ops {
        let cuda_out = comparison_forward_on(Device::Cuda(0), op);
        let cpu_out = comparison_forward_on(Device::Cpu, op);
        assert_eq!(
            contiguous_slice(&cuda_out),
            contiguous_slice(&cpu_out),
            "{name}: 比較演算は 0.0/1.0 の純粋な比較のため CUDA と CPU は bit 同一のはず"
        );
    }
}
