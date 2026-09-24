//! `fandhe_ai_autodiff::rearrange_ops`（イシュー #2143・facade 非公開の
//! 内部入口。`crates/autodiff/src/rearrange_ops.rs` モジュール doc
//! 参照）のバックエンド間 parity テスト（`bool_ops_backend_parity.rs`・
//! `unique_backend_parity.rs` と同型）。
//!
//! `rearrange_ops` は facade から再エクスポートされないため、本テストは
//! `fandhe_ai_autodiff::rearrange_ops::*` を直接 use する（facade の dev
//! 依存に `fandhe-ai-autodiff` が既に含まれている）。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）と
//!   `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）で forward・backward
//!   の出力を突き合わせる。forward は `index_select`／`broadcast_to`
//!   のいずれも算術を含まない純粋なコピー演算のため bit 完全一致
//!   （`to_bits()` 比較）を主張する。backward（scatter-add）は
//!   `flip`／`roll` が bit 一致、`repeat`／`tile` は
//!   `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合
//!   判定）で比較する。
//! - `#[ignore]`: `tape_for(Device::Metal)`（`cfg(target_os =
//!   "macos")` 限定）／`tape_for(Device::Cuda(0))` で同じ経路を CPU
//!   tape と比較する。forward は `flip`／`roll`／`repeat`／`tile` の
//!   全 4 種を bit 完全一致で比較する（PR #2256 codex-review 指摘対応。
//!   当初は `flip` のみだった）。backward は `flip`／`roll`（各入力
//!   要素への寄与が常に 1 つのため bit 完全一致。`roll` は同指摘対応で
//!   追加）に加え、backward の scatter-add 合算順序が GPU 側で自明で
//!   ない `repeat`／`tile` backward（`fandhe_ai_backend_cpu::parity::
//!   assert_parity` 比較。README の「期待結果」節と対応）も対象に含む
//!   （イシュー #2143 レビュー指摘。
//!   `cpu_repeat_tile_backward_matches_naive_reference_within_tolerance`・
//!   `cpu_flip_roll_backward_bit_matches_naive_reference` が CPU 側の
//!   同型カバレッジ）。実機（DGX Spark GB10／Apple Silicon）への到達
//!   手段が本エージェント実行環境にないため未実施のまま Mac／GB10
//!   セッションへ申し送る
//!   （`docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md`）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::rearrange_ops::{flip, repeat, roll, tile};
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

fn f32_fixture_2x3() -> Tensor<f32> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("test fixture: shape 一致")
}

fn f32_fixture_1d() -> Tensor<f32> {
    Tensor::new(vec![1.5, -2.5, 0.0, -0.0, 3.0, f32::NAN], &[6]).expect("test fixture: shape 一致")
}

fn f32_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// forward（`flip`／`roll`／`repeat`／`tile`）が CPU（`fandhe_ai::tape()`）
/// と NaiveOps（`fandhe_ai_autodiff::Tape::new()`）で bit 完全一致する
/// ことを確認する（算術を含まない純粋なコピー演算のため）。
#[test]
fn cpu_forward_matches_naive_reference() {
    let data = f32_fixture_2x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    assert_eq!(
        f32_bits(&flip(&x_cpu, &[0, 1]).unwrap().to_tensor()),
        f32_bits(&flip(&x_naive, &[0, 1]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&roll(&x_cpu, &[1, -1], &[0, 1]).unwrap().to_tensor()),
        f32_bits(&roll(&x_naive, &[1, -1], &[0, 1]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&repeat(&x_cpu, &[2, 3]).unwrap().to_tensor()),
        f32_bits(&repeat(&x_naive, &[2, 3]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&tile(&x_cpu, &[3]).unwrap().to_tensor()),
        f32_bits(&tile(&x_naive, &[3]).unwrap().to_tensor())
    );
}

/// `NaN` payload を含む forward も CPU・NaiveOps 間で bit 一致すること
/// （コピー演算のため payload も保存される契約。`bool_ops_backend_
/// parity.rs`・`masked_select_preserves_nan_bits` と同種の確認）。
#[test]
fn cpu_forward_preserves_nan_bits() {
    let data = f32_fixture_1d();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);

    assert_eq!(
        f32_bits(&flip(&x_cpu, &[0]).unwrap().to_tensor()),
        f32_bits(&flip(&x_naive, &[0]).unwrap().to_tensor())
    );
}

/// `flip`／`roll` の backward は各入力要素への寄与が常に 1 つのため
/// bit 完全一致する（CPU と NaiveOps の突き合わせ）。
#[test]
fn cpu_flip_roll_backward_bit_matches_naive_reference() {
    let data = f32_fixture_2x3();
    let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let y_cpu = flip(&x_cpu, &[0, 1]).unwrap();
    let loss_cpu = y_cpu.mul(&w_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let w_naive = naive_tape.make_var(&weight);
    let y_naive = flip(&x_naive, &[0, 1]).unwrap();
    let loss_naive = y_naive.mul(&w_naive).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();

    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_naive));
}

/// `repeat`／`tile` の backward は `r` 個のコピーの勾配を合算する
/// （scatter-add）。CPU と NaiveOps はいずれもホスト参照実装
/// （`.claude/rules/coding-rust.md` REQ-2 統一複合判定）で比較する。
/// `tile` も `repeat` と同じ scatter-add 構造のため同一テストで
/// 突き合わせる（イシュー #2143 レビュー指摘: 実機側 `#[ignore]`
/// テスト（`cuda_repeat_tile_backward_matches_cpu_reference`／
/// `metal_repeat_tile_backward_matches_cpu_reference`）と対称の
/// カバレッジを CPU 側にも持たせる）。
#[test]
fn cpu_repeat_tile_backward_matches_naive_reference_within_tolerance() {
    let data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let weight = Tensor::new(vec![1.0, 0.5, -0.25, 2.0, 0.1, -1.0, 3.0, -2.0, 0.7], &[9]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let y_cpu = repeat(&x_cpu, &[3]).unwrap();
    let loss_cpu = y_cpu.mul(&w_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&data);
    let w_naive = naive_tape.make_var(&weight);
    let y_naive = repeat(&x_naive, &[3]).unwrap();
    let loss_naive = y_naive.mul(&w_naive).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "repeat backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );

    let tile_data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let tile_weight =
        Tensor::new(vec![1.0, 0.5, -0.25, 2.0, 0.1, -1.0, 3.0, -2.0, 0.7], &[9]).unwrap();

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&tile_data);
    let w_cpu = cpu_tape.make_var(&tile_weight);
    let y_cpu = tile(&x_cpu, &[3]).unwrap();
    let loss_cpu = y_cpu.mul(&w_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let naive_tape = fandhe_ai_autodiff::Tape::new();
    let x_naive = naive_tape.make_var(&tile_data);
    let w_naive = naive_tape.make_var(&tile_weight);
    let y_naive = tile(&x_naive, &[3]).unwrap();
    let loss_naive = y_naive.mul(&w_naive).unwrap().sum(None).unwrap();
    let grads_naive = naive_tape.backward(&loss_naive).unwrap();
    let dx_naive = grads_naive.get(&x_naive).unwrap().unwrap().clone();

    fandhe_ai_backend_cpu::parity::assert_parity(
        "tile backward: cpu vs naive",
        dx_cpu.host_slice().as_ref(),
        dx_naive.host_slice().as_ref(),
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md`）。
// ---------------------------------------------------------------------

/// forward（`flip`／`roll`／`repeat`／`tile` の全 4 種）が CPU と Metal
/// 実機で bit 完全一致することを確認する（イシュー #2143 レビュー指摘・
/// PR #2256 codex-review 指摘: 従来は `flip` のみで `roll`／`repeat`／
/// `tile` の実機 forward 経路が未検証だった）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md 参照"]
fn metal_forward_matches_cpu_reference() {
    let data = f32_fixture_2x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");
    let x_metal = metal_tape.make_var(&data);

    assert_eq!(
        f32_bits(&flip(&x_cpu, &[0, 1]).unwrap().to_tensor()),
        f32_bits(&flip(&x_metal, &[0, 1]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&roll(&x_cpu, &[1, -1], &[0, 1]).unwrap().to_tensor()),
        f32_bits(&roll(&x_metal, &[1, -1], &[0, 1]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&repeat(&x_cpu, &[2, 3]).unwrap().to_tensor()),
        f32_bits(&repeat(&x_metal, &[2, 3]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&tile(&x_cpu, &[3]).unwrap().to_tensor()),
        f32_bits(&tile(&x_metal, &[3]).unwrap().to_tensor())
    );
}

/// forward（`flip`／`roll`／`repeat`／`tile` の全 4 種）が CPU と CUDA
/// 実機（DGX Spark GB10）で bit 完全一致することを確認する（イシュー
/// #2143 レビュー指摘・PR #2256 codex-review 指摘。上記 Metal 版と対称）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md 参照"]
fn cuda_forward_matches_cpu_reference() {
    let data = f32_fixture_2x3();
    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");
    let x_cuda = cuda_tape.make_var(&data);

    assert_eq!(
        f32_bits(&flip(&x_cpu, &[0, 1]).unwrap().to_tensor()),
        f32_bits(&flip(&x_cuda, &[0, 1]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&roll(&x_cpu, &[1, -1], &[0, 1]).unwrap().to_tensor()),
        f32_bits(&roll(&x_cuda, &[1, -1], &[0, 1]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&repeat(&x_cpu, &[2, 3]).unwrap().to_tensor()),
        f32_bits(&repeat(&x_cuda, &[2, 3]).unwrap().to_tensor())
    );
    assert_eq!(
        f32_bits(&tile(&x_cpu, &[3]).unwrap().to_tensor()),
        f32_bits(&tile(&x_cuda, &[3]).unwrap().to_tensor())
    );
}

/// `roll` backward（scatter-add だが各入力要素への寄与が常に 1 つのため
/// bit 完全一致する契約。モジュール doc 参照）の CPU／Metal 実機比較
/// （イシュー #2143 レビュー指摘・PR #2256 codex-review 指摘: `roll`
/// backward の実機比較テストが欠けていた）。CPU 側の同型カバレッジは
/// `cpu_flip_roll_backward_bit_matches_naive_reference`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md 参照"]
fn metal_roll_backward_matches_cpu_reference() {
    let data = f32_fixture_2x3();
    let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let y_cpu = roll(&x_cpu, &[1, -1], &[0, 1]).unwrap();
    let loss_cpu = y_cpu.mul(&w_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let x_metal = metal_tape.make_var(&data);
    let w_metal = metal_tape.make_var(&weight);
    let y_metal = roll(&x_metal, &[1, -1], &[0, 1]).unwrap();
    let loss_metal = y_metal.mul(&w_metal).unwrap().sum(None).unwrap();
    let grads_metal = metal_tape.backward(&loss_metal).unwrap();
    let dx_metal = grads_metal.get(&x_metal).unwrap().unwrap().clone();

    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_metal));
}

/// `roll` backward の CPU／CUDA 実機（DGX Spark GB10）比較。上記 Metal
/// 版と対称（イシュー #2143 レビュー指摘・PR #2256 codex-review 指摘）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md 参照"]
fn cuda_roll_backward_matches_cpu_reference() {
    let data = f32_fixture_2x3();
    let weight = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let y_cpu = roll(&x_cpu, &[1, -1], &[0, 1]).unwrap();
    let loss_cpu = y_cpu.mul(&w_cpu).unwrap().sum(None).unwrap();
    let grads_cpu = cpu_tape.backward(&loss_cpu).unwrap();
    let dx_cpu = grads_cpu.get(&x_cpu).unwrap().unwrap().clone();

    let x_cuda = cuda_tape.make_var(&data);
    let w_cuda = cuda_tape.make_var(&weight);
    let y_cuda = roll(&x_cuda, &[1, -1], &[0, 1]).unwrap();
    let loss_cuda = y_cuda.mul(&w_cuda).unwrap().sum(None).unwrap();
    let grads_cuda = cuda_tape.backward(&loss_cuda).unwrap();
    let dx_cuda = grads_cuda.get(&x_cuda).unwrap().unwrap().clone();

    assert_eq!(f32_bits(&dx_cpu), f32_bits(&dx_cuda));
}

/// `repeat`／`tile` backward（scatter-add）の CPU／実機比較。README
/// 「期待結果」節が明記する「GPU の scatter 加算順序次第では厳密な bit
/// 一致にならない可能性があるため REQ-2 統一複合判定で比較する」を
/// 実測する手段（イシュー #2143 レビュー指摘: 従来の `#[ignore]` テスト
/// は `flip` の forward のみで repeat／tile backward の実機カバレッジが
/// 存在しなかった）。CPU 側の同型カバレッジは
/// `cpu_repeat_tile_backward_matches_naive_reference_within_tolerance`。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md 参照"]
fn metal_repeat_tile_backward_matches_cpu_reference() {
    let data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let weight = Tensor::new(vec![1.0, 0.5, -0.25, 2.0, 0.1, -1.0, 3.0, -2.0, 0.7], &[9]).unwrap();
    let metal_tape =
        fandhe_ai::tape_for(Device::Metal).expect("実機が利用可能な前提のテストのため成功するはず");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let dx_cpu = cpu_tape
        .backward(
            &repeat(&x_cpu, &[3])
                .unwrap()
                .mul(&w_cpu)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let x_metal = metal_tape.make_var(&data);
    let w_metal = metal_tape.make_var(&weight);
    let dx_metal = metal_tape
        .backward(
            &repeat(&x_metal, &[3])
                .unwrap()
                .mul(&w_metal)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "repeat backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );

    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let dx_cpu = cpu_tape
        .backward(
            &tile(&x_cpu, &[3])
                .unwrap()
                .mul(&w_cpu)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let x_metal = metal_tape.make_var(&data);
    let w_metal = metal_tape.make_var(&weight);
    let dx_metal = metal_tape
        .backward(
            &tile(&x_metal, &[3])
                .unwrap()
                .mul(&w_metal)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_metal)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "tile backward: cpu vs metal",
        dx_cpu.host_slice().as_ref(),
        dx_metal.host_slice().as_ref(),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md 参照"]
fn cuda_repeat_tile_backward_matches_cpu_reference() {
    let data = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let weight = Tensor::new(vec![1.0, 0.5, -0.25, 2.0, 0.1, -1.0, 3.0, -2.0, 0.7], &[9]).unwrap();
    let cuda_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("実機が利用可能な前提のテストのため成功するはず");

    let cpu_tape = fandhe_ai::tape();
    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let dx_cpu = cpu_tape
        .backward(
            &repeat(&x_cpu, &[3])
                .unwrap()
                .mul(&w_cpu)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let x_cuda = cuda_tape.make_var(&data);
    let w_cuda = cuda_tape.make_var(&weight);
    let dx_cuda = cuda_tape
        .backward(
            &repeat(&x_cuda, &[3])
                .unwrap()
                .mul(&w_cuda)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "repeat backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );

    let x_cpu = cpu_tape.make_var(&data);
    let w_cpu = cpu_tape.make_var(&weight);
    let dx_cpu = cpu_tape
        .backward(
            &tile(&x_cpu, &[3])
                .unwrap()
                .mul(&w_cpu)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_cpu)
        .unwrap()
        .unwrap()
        .clone();
    let x_cuda = cuda_tape.make_var(&data);
    let w_cuda = cuda_tape.make_var(&weight);
    let dx_cuda = cuda_tape
        .backward(
            &tile(&x_cuda, &[3])
                .unwrap()
                .mul(&w_cuda)
                .unwrap()
                .sum(None)
                .unwrap(),
        )
        .unwrap()
        .get(&x_cuda)
        .unwrap()
        .unwrap()
        .clone();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "tile backward: cpu vs cuda",
        dx_cpu.host_slice().as_ref(),
        dx_cuda.host_slice().as_ref(),
    );
}
