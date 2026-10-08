//! `fandhe_ai_autodiff::functional_ops::{vjp, hvp, vmap}`（イシュー #2877・親 #2841。内部実装の parity 層であるため
//! `fandhe_ai_autodiff` を直接 use する。facade 経由の結合テストは `functional_transforms_facade.rs`〈#2931〉。
//! `crates/autodiff/src/functional_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`jacobian_hessian_backend_parity.rs` と同型）。
//!
//! 属性なし: 実 `CpuBackendOps` を結線した tape と `Tape::new()`（`NaiveOps`）を突き合わせ、REQ-2
//! 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。3 変換は新しい
//! `BackendOps` メソッドを持たず既存 Op の合成のみで到達するため、ここで確認するのは「既存カーネルの
//! 新しい呼び出し形（余接重み付き backward・子テープ上の VJP・スライスごとの実行と stack）が
//! バックエンド間で一致すること」である。手計算の期待値も固定する。PyTorch 実行値との突合は
//! `crates/autodiff/tests/functional_ops_pytorch_parity.rs` が担当する（facade の dev-deps は
//! `serde` の derive を持たず、本ファイルは JSON を読まない）。
//!
//! `#[ignore]`（イシュー #2881）: CUDA／Metal（`cfg(target_os = "macos")` 限定）の `BackendOps` を結線した
//! tape と CPU tape の比較。実機に届かない環境では未実施のまま
//! `docs/perf/logs/functional-transforms-2881/README.md` へ申し送る。形状は小さく、Metal split-K が
//! 発動する形状は使わない（統一複合判定をそのまま適用できる）。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::functional_ops::{hvp, vjp, vmap};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

struct Outs {
    vjp_shape: Vec<usize>,
    vjp: Vec<f32>,
    hvp_shape: Vec<usize>,
    hvp: Vec<f32>,
    vmap_shape: Vec<usize>,
    vmap: Vec<f32>,
    vmap_t_shape: Vec<usize>,
    vmap_t: Vec<f32>,
}

fn compute(tape: &Tape, child: &Tape) -> Outs {
    let w1 = tape.var_no_grad(&t(
        (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
        &[3, 4],
    ));
    let w2 = tape.var_no_grad(&t(
        (0..8).map(|i| 0.2 - 0.05 * (i as f32)).collect(),
        &[4, 2],
    ));
    let x = tape.var(&t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]));
    let y = x.matmul(&w1).unwrap().tanh().matmul(&w2).unwrap();
    let u = t(vec![0.5, -1.0, 0.25, 2.0], &[2, 2]);
    let g = vjp(tape, &y, &x, &u).unwrap();

    let loss = y.sigmoid().sum(None).unwrap();
    let v = t(vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5], &[2, 3]);
    let hv = hvp(tape, &loss, &x, &v, child).unwrap();

    // rank 3 入力の各スライス `[2, 3]` に `tanh(s @ w)`。
    let w = tape.var_no_grad(&t(
        (0..6).map(|i| 0.3 - 0.1 * (i as f32)).collect(),
        &[3, 2],
    ));
    let batch = tape.var(&t(
        (0..18).map(|i| 0.05 * (i as f32) - 0.4).collect(),
        &[3, 2, 3],
    ));
    let m = vmap(tape, &batch, 0, |s| Ok(s.matmul(&w)?.tanh())).unwrap();
    // スライス出力が非 contiguous（`transpose`）になるケース。
    let mt = vmap(tape, &batch, 0, |s| s.transpose(0, 1)).unwrap();
    Outs {
        vjp_shape: g.shape().to_vec(),
        vjp: g.host_slice().into_owned(),
        hvp_shape: hv.shape().to_vec(),
        hvp: hv.host_slice().into_owned(),
        vmap_shape: m.to_tensor().shape().to_vec(),
        vmap: m.to_tensor().host_slice().into_owned(),
        vmap_t_shape: mt.to_tensor().shape().to_vec(),
        vmap_t: mt.to_tensor().host_slice().into_owned(),
    }
}

fn cpu_tape() -> Tape {
    Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
}

fn assert_outs_match(label: &str, a: &Outs, b: &Outs) {
    assert_eq!(a.vjp_shape, b.vjp_shape, "{label}: vjp shape");
    assert_eq!(a.hvp_shape, b.hvp_shape, "{label}: hvp shape");
    assert_eq!(a.vmap_shape, b.vmap_shape, "{label}: vmap shape");
    assert_eq!(
        a.vmap_t_shape, b.vmap_t_shape,
        "{label}: vmap(transpose) shape"
    );
    let p = fandhe_ai_backend_cpu::parity::assert_parity;
    p(&format!("{label}: vjp"), &a.vjp, &b.vjp);
    p(&format!("{label}: hvp"), &a.hvp, &b.hvp);
    p(&format!("{label}: vmap"), &a.vmap, &b.vmap);
    p(&format!("{label}: vmap(transpose)"), &a.vmap_t, &b.vmap_t);
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = compute(&cpu_tape(), &cpu_tape());
    let naive = compute(&Tape::new(), &Tape::new());
    assert_eq!(cpu.vjp_shape, vec![2, 3]);
    assert_eq!(cpu.hvp_shape, vec![2, 3]);
    assert_eq!(cpu.vmap_shape, vec![3, 2, 2]);
    assert_eq!(cpu.vmap_t_shape, vec![3, 3, 2]);
    assert_outs_match("cpu vs naive", &cpu, &naive);
}

/// 手計算の期待値を固定する（両経路が同じ誤りで一致していないことの確認）。
/// - vjp: `y = x ⊙ x`・`x = [1, 2]`・`u = [1, 1]` は `u ⊙ 2x = [2, 4]`。
/// - hvp: `loss = Σ x³`・`x = [1, 2]`・`v = e_0` は `6x ⊙ v = [6, 0]`。
/// - vmap: `x = [[1, 2], [3, 4]]` の各行に `s ⊙ s` を適用すると `[[1, 4], [9, 16]]`。
#[test]
fn cpu_matches_hand_computed_values() {
    let tape = cpu_tape();
    let child = cpu_tape();
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let sq = x.mul(&x).unwrap();
    let g = vjp(&tape, &sq, &x, &t(vec![1.0, 1.0], &[2])).unwrap();
    assert_eq!(g.host_slice().as_ref(), &[2.0, 4.0]);

    let loss = sq.mul(&x).unwrap().sum(None).unwrap();
    let hv = hvp(&tape, &loss, &x, &t(vec![1.0, 0.0], &[2]), &child).unwrap();
    assert_eq!(hv.host_slice().as_ref(), &[6.0, 0.0]);

    let b = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let m = vmap(&tape, &b, 0, |s| s.mul(s)).unwrap();
    assert_eq!(m.to_tensor().shape(), &[2, 2]);
    assert_eq!(m.to_tensor().host_slice().as_ref(), &[1.0, 4.0, 9.0, 16.0]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/functional-transforms-2881/README.md`）。
// ---------------------------------------------------------------------

/// 実機 `BackendOps` を結線した tape（子テープも同一バックエンド）と CPU tape の `compute` 結果を比較する。
fn assert_device_matches_cpu(make_ops: impl Fn() -> Box<dyn BackendOps + Send>, label: &str) {
    let cpu = compute(&cpu_tape(), &cpu_tape());
    let dev = compute(
        &Tape::new_with_ops(make_ops()),
        &Tape::new_with_ops(make_ops()),
    );
    assert_outs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// vjp・hvp・vmap の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-2881/README.md 参照"]
fn metal_functional_ops_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
        "metal",
    );
}

/// vjp・hvp・vmap の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-2881/README.md 参照"]
fn cuda_functional_ops_match_cpu_reference() {
    assert_device_matches_cpu(
        || Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)),
        "cuda",
    );
}
