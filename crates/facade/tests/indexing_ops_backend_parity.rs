//! `fandhe_ai_autodiff::indexing_ops`（イシュー #2148・facade 非公開の
//! 内部入口。`crates/autodiff/src/indexing_ops.rs` モジュール doc 参照）
//! のバックエンド間 parity テスト（`reduce_ops_backend_parity.rs` と
//! 同型）。
//!
//! `indexing_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::indexing_ops::*` を直接 use する（facade の dev
//! 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! 網羅表（`matrix_ops_backend_parity.rs` の教訓: 代表 1 演算で他を
//! 代替しない）:
//! - `advanced_indexing` forward（コピーのみのため bit 完全一致）:
//!   `cpu_advanced_indexing_forward_bit_matches_naive_reference`
//! - `index_put`（`accumulate: false`）forward（コピーのみのため bit
//!   完全一致。重複添字を含む）:
//!   `cpu_index_put_overwrite_forward_bit_matches_naive_reference`
//! - `index_put`（`accumulate: true`）forward（`ScatterReduce::Add` の
//!   `f64` 決定的集約契約）:
//!   `cpu_index_put_accumulate_forward_matches_naive_reference`
//! - 3 演算の backward（重複添字あり・なし × R 空・非空）:
//!   `cpu_indexing_backward_matches_naive_reference`
//! - `NaN`／`inf` の payload が [`advanced_indexing`] のコピー経路で
//!   保たれること: `cpu_advanced_indexing_preserves_nan_payload`
//! - `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os =
//!   "macos")` 限定〉／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較）: 上記と対称に `cuda_*`／`metal_*` という接頭辞で置く。
//!   forward（`advanced_indexing`／`index_put` 両 `accumulate`）に加え、
//!   backward（`Op::Gather`／`Op::Scatter` の VJP。重複添字による
//!   `scatter_add` を含む）も `*_indexing_backward_matches_cpu_reference`
//!   として同じ接頭辞で置く（codex-review 指摘・PR #2267。forward の
//!   みでは新しい合成経路が使う VJP が実機で比較されない）。
//!
//!   実機（DGX Spark GB10／Apple Silicon）への到達手段が本エージェント
//!   実行環境にないため未実施のまま Mac／GB10 セッションへ申し送る
//!   （`docs/perf/logs/indexing-inplace-2148/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::indexing_ops::{advanced_indexing, index_put};
use fandhe_ai_tensor_core::Tensor;

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

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn advanced_indexing_fixture() -> (Tensor<f32>, Tensor<i32>) {
    // [3, 2] から行 [2, 0, 2]（重複あり）を読む。
    (
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).expect("test fixture: shape 一致"),
        ti(vec![2, 0, 2], &[3]),
    )
}

fn index_put_fixture() -> (Tensor<f32>, Tensor<i32>, Tensor<f32>) {
    // [4] へ重複添字（0 を 2 回）で書き込む。
    (
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("test fixture: shape 一致"),
        ti(vec![0, 0, 2], &[3]),
        Tensor::new(vec![10.0, 20.0, 30.0], &[3]).expect("test fixture: shape 一致"),
    )
}

/// [`advanced_indexing`] forward が CPU（`fandhe_ai::tape()`）と
/// NaiveOps（`fandhe_ai_autodiff::Tape::new()`）で bit 完全一致する
/// ことを確認する（コピーのみで算術を含まないため）。
#[test]
fn cpu_advanced_indexing_forward_bit_matches_naive_reference() {
    let (data, idx) = advanced_indexing_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let out_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
        .unwrap()
        .to_tensor();
    let out_naive = advanced_indexing(&x_naive, &[idx]).unwrap().to_tensor();
    assert_eq!(
        f32_bits(&out_cpu),
        f32_bits(&out_naive),
        "advanced_indexing forward"
    );
}

/// [`index_put`]（`accumulate: false`）forward が CPU と NaiveOps で
/// bit 完全一致することを確認する（重複添字を含む。B の行優先走査で
/// 最後の書き手が勝つ決定的な結果になるため、算術を含まないコピーの
/// 組み合わせでも縮約順序に依存しない）。
#[test]
fn cpu_index_put_overwrite_forward_bit_matches_naive_reference() {
    let (data, idx, values) = index_put_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let v_cpu = cpu_tape.make_var(&values);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let v_naive = naive_tape.make_var(&values);

    let out_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, false)
        .unwrap()
        .to_tensor();
    let out_naive = index_put(&x_naive, &[idx], &v_naive, false)
        .unwrap()
        .to_tensor();
    assert_eq!(
        f32_bits(&out_cpu),
        f32_bits(&out_naive),
        "index_put(accumulate=false) forward"
    );
}

/// [`index_put`]（`accumulate: true`）forward が CPU と NaiveOps で
/// REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
/// 満たすことを確認する（`ScatterReduce::Add` の `f64` 決定的集約契約。
/// 縮約順序自体はバックエンドで異なりうるため厳密 bit 一致は要求しない）。
#[test]
fn cpu_index_put_accumulate_forward_matches_naive_reference() {
    let (data, idx, values) = index_put_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let v_cpu = cpu_tape.make_var(&values);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let v_naive = naive_tape.make_var(&values);

    let out_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, true)
        .unwrap()
        .to_tensor();
    let out_naive = index_put(&x_naive, &[idx], &v_naive, true)
        .unwrap()
        .to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "index_put(accumulate=true) forward: cpu vs naive",
        out_cpu.host_slice().as_ref(),
        out_naive.host_slice().as_ref(),
    );
}

/// `advanced_indexing`・`index_put`（`accumulate` 両方）の backward が
/// CPU と NaiveOps で REQ-2 統一複合判定を満たすことを確認する。重複
/// 添字あり・なし × R（残り軸）空・非空の組み合わせを網羅する。
#[test]
fn cpu_indexing_backward_matches_naive_reference() {
    // advanced_indexing: R 空・重複あり。
    {
        let data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("test fixture: shape 一致");
        let idx = ti(vec![0, 0, 1], &[3]);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let loss_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_cpu = cpu_tape
            .backward(&loss_cpu)
            .unwrap()
            .get(&x_cpu)
            .unwrap()
            .unwrap()
            .clone();
        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let loss_naive = advanced_indexing(&x_naive, &[idx])
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_naive = naive_tape
            .backward(&loss_naive)
            .unwrap()
            .get(&x_naive)
            .unwrap()
            .unwrap()
            .clone();
        fandhe_ai_backend_cpu::parity::assert_parity(
            "advanced_indexing backward (R empty, dup): cpu vs naive",
            dx_cpu.host_slice().as_ref(),
            dx_naive.host_slice().as_ref(),
        );
    }

    // advanced_indexing: R 非空・重複なし。
    {
        let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
            .expect("test fixture: shape 一致");
        let idx = ti(vec![2, 0], &[2]);
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let loss_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_cpu = cpu_tape
            .backward(&loss_cpu)
            .unwrap()
            .get(&x_cpu)
            .unwrap()
            .unwrap()
            .clone();
        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let loss_naive = advanced_indexing(&x_naive, &[idx])
            .unwrap()
            .sum(None)
            .unwrap();
        let dx_naive = naive_tape
            .backward(&loss_naive)
            .unwrap()
            .get(&x_naive)
            .unwrap()
            .unwrap()
            .clone();
        fandhe_ai_backend_cpu::parity::assert_parity(
            "advanced_indexing backward (R non-empty, no dup): cpu vs naive",
            dx_cpu.host_slice().as_ref(),
            dx_naive.host_slice().as_ref(),
        );
    }

    // index_put(accumulate=false): R 空・重複あり。
    {
        let (data, idx, values) = index_put_fixture();
        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let v_cpu = cpu_tape.make_var(&values);
        let loss_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, false)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();
        let dv_cpu = grads_cpu.get(&v_cpu).unwrap().unwrap().clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let v_naive = naive_tape.make_var(&values);
        let loss_naive = index_put(&x_naive, &[idx], &v_naive, false)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_naive = naive_tape.backward(&loss_naive).unwrap();
        let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();
        let dv_naive = grads_naive.get(&v_naive).unwrap().unwrap().clone();

        fandhe_ai_backend_cpu::parity::assert_parity(
            "index_put(accumulate=false) backward dx: cpu vs naive",
            dx_cpu.host_slice().as_ref(),
            dx_naive.host_slice().as_ref(),
        );
        fandhe_ai_backend_cpu::parity::assert_parity(
            "index_put(accumulate=false) backward dv: cpu vs naive",
            dv_cpu.host_slice().as_ref(),
            dv_naive.host_slice().as_ref(),
        );
    }

    // index_put(accumulate=true): R 非空・重複なし。
    {
        let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
            .expect("test fixture: shape 一致");
        let idx = ti(vec![0, 2], &[2]);
        let values =
            Tensor::new(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).expect("test fixture: shape 一致");

        let cpu_tape = fandhe_ai::tape();
        let x_cpu = cpu_tape.make_var(&data);
        let v_cpu = cpu_tape.make_var(&values);
        let loss_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, true)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();
        let dv_cpu = grads_cpu.get(&v_cpu).unwrap().unwrap().clone();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let x_naive = naive_tape.make_var(&data);
        let v_naive = naive_tape.make_var(&values);
        let loss_naive = index_put(&x_naive, &[idx], &v_naive, true)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_naive = naive_tape.backward(&loss_naive).unwrap();
        let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();
        let dv_naive = grads_naive.get(&v_naive).unwrap().unwrap().clone();

        fandhe_ai_backend_cpu::parity::assert_parity(
            "index_put(accumulate=true) backward dx: cpu vs naive",
            dx_cpu.host_slice().as_ref(),
            dx_naive.host_slice().as_ref(),
        );
        fandhe_ai_backend_cpu::parity::assert_parity(
            "index_put(accumulate=true) backward dv: cpu vs naive",
            dv_cpu.host_slice().as_ref(),
            dv_naive.host_slice().as_ref(),
        );
    }
}

/// [`advanced_indexing`] のコピー経路が `NaN`／`inf` の payload を
/// 保つことを確認する（CPU と NaiveOps で `to_bits()` レベルの一致。
/// モジュール doc「数値契約」参照）。
#[test]
fn cpu_advanced_indexing_preserves_nan_payload() {
    let nan_with_payload = f32::from_bits(0x7fc00001);
    let data = Tensor::new(vec![nan_with_payload, f32::INFINITY, 3.0], &[3])
        .expect("test fixture: shape 一致");
    let idx = ti(vec![0, 1], &[2]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    let out_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
        .unwrap()
        .to_tensor();
    let out_naive = advanced_indexing(&x_naive, &[idx]).unwrap().to_tensor();
    assert_eq!(f32_bits(&out_cpu), f32_bits(&out_naive));
    assert_eq!(
        out_cpu.host_slice()[0].to_bits(),
        nan_with_payload.to_bits()
    );
    assert!(out_cpu.host_slice()[1].is_infinite());
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/indexing-inplace-2148/README.md`）。
// ---------------------------------------------------------------------

/// [`advanced_indexing`] forward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn metal_advanced_indexing_forward_matches_cpu_reference() {
    let (data, idx) = advanced_indexing_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    let out_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
        .unwrap()
        .to_tensor();
    let out_metal = advanced_indexing(&x_metal, &[idx]).unwrap().to_tensor();
    assert_eq!(f32_bits(&out_cpu), f32_bits(&out_metal));
}

/// [`advanced_indexing`] forward の CPU／CUDA 実機（DGX Spark GB10）
/// 比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn cuda_advanced_indexing_forward_matches_cpu_reference() {
    let (data, idx) = advanced_indexing_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    let out_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
        .unwrap()
        .to_tensor();
    let out_cuda = advanced_indexing(&x_cuda, &[idx]).unwrap().to_tensor();
    assert_eq!(f32_bits(&out_cpu), f32_bits(&out_cuda));
}

/// [`index_put`]（両 `accumulate`）forward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn metal_index_put_forward_matches_cpu_reference() {
    let (data, idx, values) = index_put_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let v_cpu = cpu_tape.make_var(&values);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let v_metal = metal_tape.make_var(&values);

    let out_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, false)
        .unwrap()
        .to_tensor();
    let out_metal = index_put(&x_metal, std::slice::from_ref(&idx), &v_metal, false)
        .unwrap()
        .to_tensor();
    assert_eq!(f32_bits(&out_cpu), f32_bits(&out_metal));

    let out_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, true)
        .unwrap()
        .to_tensor();
    let out_metal = index_put(&x_metal, &[idx], &v_metal, true)
        .unwrap()
        .to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "index_put(accumulate=true) forward: cpu vs metal",
        out_cpu.host_slice().as_ref(),
        out_metal.host_slice().as_ref(),
    );
}

/// [`index_put`]（両 `accumulate`）forward の CPU／CUDA 実機（DGX
/// Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn cuda_index_put_forward_matches_cpu_reference() {
    let (data, idx, values) = index_put_fixture();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let v_cpu = cpu_tape.make_var(&values);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let v_cuda = cuda_tape.make_var(&values);

    let out_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, false)
        .unwrap()
        .to_tensor();
    let out_cuda = index_put(&x_cuda, std::slice::from_ref(&idx), &v_cuda, false)
        .unwrap()
        .to_tensor();
    assert_eq!(f32_bits(&out_cpu), f32_bits(&out_cuda));

    let out_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, true)
        .unwrap()
        .to_tensor();
    let out_cuda = index_put(&x_cuda, &[idx], &v_cuda, true)
        .unwrap()
        .to_tensor();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "index_put(accumulate=true) forward: cpu vs cuda",
        out_cpu.host_slice().as_ref(),
        out_cuda.host_slice().as_ref(),
    );
}

/// [`advanced_indexing`] backward（重複添字による `Op::Gather` の
/// `scatter_add` VJP）の CPU／Metal 実機比較（REQ-2 統一複合判定。
/// codex-review 指摘・PR #2267）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn metal_advanced_indexing_backward_matches_cpu_reference() {
    let data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("test fixture: shape 一致");
    let idx = ti(vec![0, 0, 1], &[3]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
        .unwrap()
        .sum(None)
        .unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let loss_metal = advanced_indexing(&x_metal, std::slice::from_ref(&idx))
        .unwrap()
        .sum(None)
        .unwrap();
    let dx_metal = metal_tape
        .backward(&loss_metal)
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "advanced_indexing backward (dup): cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );
}

/// [`advanced_indexing`] backward（重複添字による `Op::Gather` の
/// `scatter_add` VJP）の CPU／CUDA 実機（DGX Spark GB10）比較（REQ-2
/// 統一複合判定。codex-review 指摘・PR #2267）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn cuda_advanced_indexing_backward_matches_cpu_reference() {
    let data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("test fixture: shape 一致");
    let idx = ti(vec![0, 0, 1], &[3]);

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let loss_cpu = advanced_indexing(&x_cpu, std::slice::from_ref(&idx))
        .unwrap()
        .sum(None)
        .unwrap();
    let dx_cpu = cpu_tape
        .backward(&loss_cpu)
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();

    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let loss_cuda = advanced_indexing(&x_cuda, std::slice::from_ref(&idx))
        .unwrap()
        .sum(None)
        .unwrap();
    let dx_cuda = cuda_tape
        .backward(&loss_cuda)
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "advanced_indexing backward (dup): cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );
}

/// [`index_put`]（両 `accumulate`。重複添字を含む）backward の
/// CPU／Metal 実機比較（`Op::Scatter` の VJP。REQ-2 統一複合判定。
/// codex-review 指摘・PR #2267）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn metal_index_put_backward_matches_cpu_reference() {
    let (data, idx, values) = index_put_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let v_cpu = cpu_tape.make_var(&values);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);
    let v_metal = metal_tape.make_var(&values);

    for accumulate in [false, true] {
        let loss_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, accumulate)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();
        let dv_cpu = grads_cpu.get(&v_cpu).unwrap().unwrap().clone();

        let loss_metal = index_put(&x_metal, std::slice::from_ref(&idx), &v_metal, accumulate)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_metal = metal_tape.backward(&loss_metal).unwrap();
        let dx_metal = grads_metal.get(&x_metal).unwrap().unwrap().clone();
        let dv_metal = grads_metal.get(&v_metal).unwrap().unwrap().clone();

        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("index_put(accumulate={accumulate}) backward dx: cpu vs metal"),
            dx_cpu.host_slice().as_ref(),
            dx_metal.host_slice().as_ref(),
        );
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("index_put(accumulate={accumulate}) backward dv: cpu vs metal"),
            dv_cpu.host_slice().as_ref(),
            dv_metal.host_slice().as_ref(),
        );
    }
}

/// [`index_put`]（両 `accumulate`。重複添字を含む）backward の
/// CPU／CUDA 実機（DGX Spark GB10）比較（`Op::Scatter` の VJP。REQ-2
/// 統一複合判定。codex-review 指摘・PR #2267）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/indexing-inplace-2148/README.md 参照"]
fn cuda_index_put_backward_matches_cpu_reference() {
    let (data, idx, values) = index_put_fixture();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let v_cpu = cpu_tape.make_var(&values);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);
    let v_cuda = cuda_tape.make_var(&values);

    for accumulate in [false, true] {
        let loss_cpu = index_put(&x_cpu, std::slice::from_ref(&idx), &v_cpu, accumulate)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
        let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();
        let dv_cpu = grads_cpu.get(&v_cpu).unwrap().unwrap().clone();

        let loss_cuda = index_put(&x_cuda, std::slice::from_ref(&idx), &v_cuda, accumulate)
            .unwrap()
            .sum(None)
            .unwrap();
        let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
        let dx_cuda = grads_cuda.get(&x_cuda).unwrap().unwrap().clone();
        let dv_cuda = grads_cuda.get(&v_cuda).unwrap().unwrap().clone();

        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("index_put(accumulate={accumulate}) backward dx: cpu vs cuda"),
            dx_cpu.host_slice().as_ref(),
            dx_cuda.host_slice().as_ref(),
        );
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("index_put(accumulate={accumulate}) backward dv: cpu vs cuda"),
            dv_cpu.host_slice().as_ref(),
            dv_cuda.host_slice().as_ref(),
        );
    }
}
