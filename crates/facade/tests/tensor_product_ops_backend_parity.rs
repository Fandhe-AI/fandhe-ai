//! `fandhe_ai_autodiff::tensor_product_ops`（イシュー #2640・`kron`／`tensordot`／
//! `tensordot_axes`／`cdist`／`cross`。facade 非公開のため
//! `fandhe_ai_autodiff::tensor_product_ops::*` を直接 use する。
//! `crates/autodiff/src/tensor_product_ops.rs` モジュール doc 参照）のバックエンド間
//! parity テスト（`shape_view_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈CPU `BackendOps`〉と `fandhe_ai_autodiff::Tape::new()`
//! 〈`NaiveOps`＝任意メソッドは既定 `Unsupported` → ホスト参照実装へフォールバック〉の
//! 突き合わせ）: `kron` の forward は乗算 1 回のため bit 一致を要求し、それ以外の forward と
//! 全 backward は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。
//! 両経路が同じ誤りで一致していないことを示すため、手計算の期待値も数件固定する。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os =
//! "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行
//! 環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/tensor-product-ops-2640/README.md`）。`tensordot` の形状は小さくし、
//! Metal の split-K 経路が発動する形状は使わない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::tensor_product_ops::{cdist, cross, kron, tensordot, tensordot_axes};
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
    exact_forward: bool,
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
    ($tape:expr, $outs:expr, $label:expr, $exact:expr, [$($inp:expr),+], |$xs:ident| $body:expr) => {{
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
            exact_forward: $exact,
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
            ($label:expr, $exact:expr, $inps:tt, |$xs:ident| $body:expr) => {
                run_case!(tape, outs, $label, $exact, $inps, |$xs| $body)
            };
        }
        case!("kron(2x3, 3x2)", true, [mat(2, 3, 1), mat(3, 2, 2)], |xs| {
            kron(&xs[0], &xs[1]).unwrap()
        });
        case!(
            "tensordot(n=1)",
            false,
            [mat(4, 5, 3), mat(5, 3, 4)],
            |xs| { tensordot(&xs[0], &xs[1], 1).unwrap() }
        );
        case!(
            "tensordot(n=0)",
            false,
            [mat(3, 2, 5), mat(2, 3, 6)],
            |xs| { tensordot(&xs[0], &xs[1], 0).unwrap() }
        );
        case!(
            "tensordot_axes([0],[0])",
            false,
            [mat(5, 3, 7), mat(5, 4, 8)],
            |xs| tensordot_axes(&xs[0], &xs[1], &[0], &[0]).unwrap()
        );
        case!("cdist(p=1)", false, [mat(4, 3, 9), mat(5, 3, 10)], |xs| {
            cdist(&xs[0], &xs[1], 1.0).unwrap()
        });
        case!("cdist(p=2)", false, [mat(4, 3, 9), mat(5, 3, 10)], |xs| {
            cdist(&xs[0], &xs[1], 2.0).unwrap()
        });
        case!("cdist(p=3)", false, [mat(4, 3, 9), mat(5, 3, 10)], |xs| {
            cdist(&xs[0], &xs[1], 3.0).unwrap()
        });
        case!(
            "cross(dim=1)",
            false,
            [mat(4, 3, 11), mat(4, 3, 12)],
            |xs| { cross(&xs[0], &xs[1], 1).unwrap() }
        );
        case!(
            "cross(dim=0)",
            false,
            [mat(3, 4, 13), mat(3, 4, 14)],
            |xs| { cross(&xs[0], &xs[1], 0).unwrap() }
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

fn assert_bits_eq(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    for (i, (x, y)) in a.host_slice().iter().zip(b.host_slice().iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

fn assert_outputs_match(label: &str, a: &[Out], b: &[Out]) {
    assert_eq!(a.len(), b.len(), "{label}: 演算数");
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.label, y.label);
        let l = format!("{label}: {}", x.label);
        if x.exact_forward {
            assert_bits_eq(&format!("{l} forward"), &x.value, &y.value);
        } else {
            assert_parity(&format!("{l} forward"), &x.value, &y.value);
        }
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
    // kron([1,2],[[0,1]]) = [[0,1,0,2]]。
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape.var(&t(vec![0.0, 1.0], &[1, 2]));
    let k = kron(&a, &b).unwrap();
    assert_eq!(k.to_tensor().shape(), &[1, 4]);
    assert_eq!(
        k.to_tensor().host_slice().into_owned(),
        vec![0.0, 1.0, 0.0, 2.0]
    );
    // tensordot([[1,2],[3,4]], [[5,6],[7,8]], 1) = 行列積 [[19,22],[43,50]]。
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
    let d = tensordot(&x, &y, 1).unwrap();
    assert_eq!(
        d.to_tensor().host_slice().into_owned(),
        vec![19.0, 22.0, 43.0, 50.0]
    );
    // 全軸縮約は Frobenius 内積 5+12+21+32 = 70。
    let s = tensordot(&x, &y, 2).unwrap();
    assert_eq!(s.to_tensor().host_slice().into_owned(), vec![70.0]);
    // cdist: (0,0)-(3,4) の p=2 は 5、p=1 は 7。
    let p = tape.var(&t(vec![0.0, 0.0], &[1, 2]));
    let q = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
    let d2 = cdist(&p, &q, 2.0)
        .unwrap()
        .to_tensor()
        .host_slice()
        .into_owned();
    let d1 = cdist(&p, &q, 1.0)
        .unwrap()
        .to_tensor()
        .host_slice()
        .into_owned();
    assert!((d2[0] - 5.0).abs() < 1e-5);
    assert!((d1[0] - 7.0).abs() < 1e-5);
    // cross: x̂ × ŷ = ẑ。
    let ex = tape.var(&t(vec![1.0, 0.0, 0.0], &[3]));
    let ey = tape.var(&t(vec![0.0, 1.0, 0.0], &[3]));
    let c = cross(&ex, &ey, 0).unwrap();
    assert_eq!(c.to_tensor().host_slice().into_owned(), vec![0.0, 0.0, 1.0]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/tensor-product-ops-2640/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// テンソル積・距離・外積の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/tensor-product-ops-2640/README.md 参照"]
fn metal_tensor_product_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// テンソル積・距離・外積の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/tensor-product-ops-2640/README.md 参照"]
fn cuda_tensor_product_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
