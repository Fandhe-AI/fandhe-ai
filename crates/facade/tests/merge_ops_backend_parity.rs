//! `fandhe_ai_autodiff::merge_ops`（イシュー #2666・`merge_concatenate`／`merge_add`／
//! `merge_multiply`／`merge_average`。facade 非公開のため
//! `fandhe_ai_autodiff::merge_ops::*` を直接 use する。`crates/autodiff/src/merge_ops.rs`
//! モジュール doc 参照）のバックエンド間 parity テスト
//! （`tensor_product_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈CPU `BackendOps`〉と `fandhe_ai_autodiff::Tape::new()`
//! 〈`NaiveOps`＝任意メソッドは既定 `Unsupported` → ホスト参照実装へフォールバック〉の
//! 突き合わせ）: forward・全 backward を REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。両経路が同じ誤りで一致して
//! いないことを示すため、手計算の期待値も数件固定する。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os =
//! "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行
//! 環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/merge-ops-2666/README.md`）。形状は小さくし、Metal の split-K 経路が
//! 発動する形状は使わない（結合演算は gemm を呼ばない）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::merge_ops::{merge_add, merge_average, merge_concatenate, merge_multiply};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn mat(r: usize, c: usize, salt: usize) -> Tensor<f32> {
    t(
        (0..r * c)
            .map(|i| (((i * 13 + salt * 5 + 3) % 29) as f32) * 0.21 - 3.0)
            .collect(),
        &[r, c],
    )
}

struct Out {
    label: &'static str,
    value: Tensor<f32>,
    grads: Vec<Tensor<f32>>,
}

/// 損失 `Σ w ⊙ y` の重み（出力 shape ごとに決定的に作る）。
fn weights(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(
        (0..n).map(|i| 0.5 + 0.3 * ((i % 3) as f32)).collect(),
        shape,
    )
}

/// 1 ケース分（入力 `Var` → 出力 `Var` → 損失 `Σ w ⊙ y` → 全入力の勾配）を実行して
/// `outs` へ積む。`$tape` は具象 `Tape` 型ごとに展開するためマクロにしている。
macro_rules! run_case {
    ($tape:expr, $outs:expr, $label:expr, [$($inp:expr),+], |$xs:ident| $body:expr) => {{
        let tape = &$tape;
        let $xs: Vec<Var<'_>> = vec![$(tape.var(&$inp)),+];
        let y: Var<'_> = $body;
        let value = y.to_tensor();
        let w = tape.var(&weights(value.shape()));
        let loss = y.mul(&w).unwrap().sum(None).unwrap();
        let gs = tape.backward(&loss).unwrap();
        let grads = $xs
            .iter()
            .map(|x| gs.get(x).unwrap().unwrap().clone())
            .collect();
        $outs.push(Out {
            label: $label,
            value,
            grads,
        });
    }};
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let mut outs: Vec<Out> = Vec::new();
        macro_rules! case {
            ($label:expr, $inps:tt, |$xs:ident| $body:expr) => {
                run_case!(tape, outs, $label, $inps, |$xs| $body)
            };
        }
        case!(
            "concatenate(dim=0, 2 入力)",
            [mat(2, 3, 1), mat(3, 3, 2)],
            |xs| merge_concatenate(&xs, 0).unwrap()
        );
        case!(
            "concatenate(dim=1, 3 入力)",
            [mat(3, 2, 3), mat(3, 1, 4), mat(3, 4, 5)],
            |xs| merge_concatenate(&xs, 1).unwrap()
        );
        case!(
            "add(3 入力)",
            [mat(3, 4, 6), mat(3, 4, 7), mat(3, 4, 8)],
            |xs| { merge_add(&xs).unwrap() }
        );
        case!(
            "multiply(3 入力)",
            [mat(3, 4, 9), mat(3, 4, 10), mat(3, 4, 11)],
            |xs| { merge_multiply(&xs).unwrap() }
        );
        case!(
            "average(4 入力)",
            [mat(2, 5, 12), mat(2, 5, 13), mat(2, 5, 14), mat(2, 5, 15)],
            |xs| merge_average(&xs).unwrap()
        );
        outs
    }};
}

fn cpu_outputs() -> Vec<Out> {
    outputs_on!(fandhe_ai::tape())
}

fn naive_outputs() -> Vec<Out> {
    outputs_on!(fandhe_ai_autodiff::Tape::new())
}

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

fn assert_outputs_match(label: &str, a: &[Out], b: &[Out]) {
    assert_eq!(a.len(), b.len(), "{label}: 演算数");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.label, y.label);
        let l = format!("{label}: {}", x.label);
        assert_parity(&format!("{l} forward"), &x.value, &y.value);
        for (k, (gx, gy)) in x.grads.iter().zip(&y.grads).enumerate() {
            assert_parity(&format!("{l} backward[{k}]"), gx, gy);
        }
    }
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
}

/// 両経路が同じ誤りで一致していないことを示す手計算の期待値。
#[test]
fn hand_computed_expectations() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
    let c = tape.var(&t(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]));
    let host = |v: Var<'_>| v.to_tensor().host_slice().into_owned();
    assert_eq!(
        host(merge_add(&[a, b, c]).unwrap()),
        vec![7.0, 9.0, 12.0, 14.0]
    );
    assert_eq!(
        host(merge_multiply(&[a, b, c]).unwrap()),
        vec![5.0, 12.0, 42.0, 64.0]
    );
    let avg = host(merge_average(&[a, b, c]).unwrap());
    for (got, want) in avg.iter().zip([7.0 / 3.0, 3.0, 4.0, 14.0 / 3.0]) {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
    let cat = merge_concatenate(&[a, b], 1).unwrap().to_tensor();
    assert_eq!(cat.shape(), &[2, 4]);
    assert_eq!(
        cat.host_slice().into_owned(),
        vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/merge-ops-2666/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 結合演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/merge-ops-2666/README.md 参照"]
fn metal_merge_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 結合演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/merge-ops-2666/README.md 参照"]
fn cuda_merge_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
