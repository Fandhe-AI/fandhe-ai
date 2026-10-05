//! `fandhe_ai_autodiff::stat_reduce_ops`（イシュー #2637・`median`／`kthvalue`／
//! `quantile`／`nansum`／`nanmean`。facade 非公開のため
//! `fandhe_ai_autodiff::stat_reduce_ops::*` を直接 use する。
//! `crates/autodiff/src/stat_reduce_ops.rs` モジュール doc 参照）のバックエンド間 parity
//! テスト（`cumulative_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::stat_*`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` →
//! 共有ホストカーネルへフォールバック〉の突き合わせ）: 選択のみの演算の値は bit 一致
//! （NaN はクラス一致）・索引は完全一致、補間・`nan*` の forward と各 backward は
//! REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。両経路とも
//! 同じ共有カーネルを呼ぶため、ここで確認するのは「`BackendOps` 経由の配線（形状検証・
//! フォールバック）が結果を変えないこと」である。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os
//! = "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント
//! 実行環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/stat-reduce-ops-2637/README.md`）。CUDA／Metal はいずれも本 5 演算の
//! GPU カーネルを持たないため（既定 `Unsupported`）、この比較は「ホストへの
//! フォールバック経路が CPU tape と同じ結果になること」を確認するものであり、GPU
//! カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::stat_reduce_ops::{
    kthvalue, median, median_with_indices, nanmean, nansum, quantile,
};
use fandhe_ai_tensor_core::{QuantileInterpolation, Tensor};

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

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

/// 全要素が互いに異なる（タイなし）`[2, 3, 4]` 入力。`(13 i + 5) mod 29` は
/// `i < 29` で単射のため索引の完全一致を要求できる。
fn input() -> Tensor<f32> {
    let data: Vec<f32> = (0..24)
        .map(|i| (((i * 13 + 5) % 29) as f32) * 0.21 - 3.0)
        .collect();
    t(data, &[2, 3, 4])
}

/// [`input`] の 2 要素を NaN に置換した入力（`nansum`／`nanmean` 用）。
fn input_with_nan() -> Tensor<f32> {
    let mut data = input().host_slice().into_owned();
    data[5] = f32::NAN;
    data[17] = f32::NAN;
    t(data, &[2, 3, 4])
}

struct Out {
    label: &'static str,
    selection: bool,
    value: Tensor<f32>,
    index: Option<Vec<i32>>,
    grad: Tensor<f32>,
}

fn indices(i: &Tensor<i32>) -> Vec<i32> {
    i.contiguous().host_slice().into_owned()
}

/// 損失 `Σ w ⊙ y` の重み（出力 shape ごとに決定的に作る）。
fn weights(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(
        (0..n).map(|i| 0.5 + 0.3 * ((i % 3) as f32)).collect(),
        shape,
    )
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let mut outs: Vec<Out> = Vec::new();
        // `|x| body` は入力 `Var` を受けて `(値 Var, 索引)` を返す式。
        macro_rules! case {
            ($label:expr, $selection:expr, $inp:expr, |$x:ident| $body:expr) => {{
                let $x = tape.make_var(&$inp);
                let (y, idx): (Var<'_>, Option<Tensor<i32>>) = $body;
                let value = y.to_tensor();
                let w = tape.make_var(&weights(value.shape()));
                let loss = y.mul(&w).unwrap().sum(None).unwrap();
                let grad = tape
                    .backward(&loss)
                    .unwrap()
                    .get(&$x)
                    .unwrap()
                    .unwrap()
                    .clone();
                outs.push(Out {
                    label: $label,
                    selection: $selection,
                    value,
                    index: idx.as_ref().map(indices),
                    grad,
                });
            }};
        }
        case!("median(None)", true, input(), |x| (
            median(&x, None).unwrap(),
            None
        ));
        case!("median(Some(1))", true, input(), |x| {
            let (v, i) = median_with_indices(&x, 1).unwrap();
            (v, Some(i))
        });
        case!("kthvalue(k=2, dim=2)", true, input(), |x| {
            let (v, i) = kthvalue(&x, 2, 2).unwrap();
            (v, Some(i))
        });
        case!("quantile(linear, dim=1)", false, input(), |x| (
            quantile(&x, 0.3, Some(1), QuantileInterpolation::Linear).unwrap(),
            None
        ));
        case!("quantile(higher, None)", true, input(), |x| (
            quantile(&x, 0.7, None, QuantileInterpolation::Higher).unwrap(),
            None
        ));
        case!("nansum(dim=2)", false, input_with_nan(), |x| (
            nansum(&x, Some(2)).unwrap(),
            None
        ));
        case!("nanmean(None)", false, input_with_nan(), |x| (
            nanmean(&x, None).unwrap(),
            None
        ));
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
        if x.selection {
            assert_bits_eq(&format!("{l} 値"), &x.value, &y.value);
        } else {
            assert_parity(&format!("{l} forward"), &x.value, &y.value);
        }
        assert_eq!(x.index, y.index, "{l}: 索引");
        assert_parity(&format!("{l} backward"), &x.grad, &y.grad);
    }
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）: 全要素の下側中央値は
    // 24 個の昇順 12 番目、`nansum(dim=2)` の NaN 無視は NaN を含まない有限値。
    let mut sorted = input().host_slice().into_owned();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        cpu[0].value.host_slice()[0].to_bits(),
        sorted[(24 - 1) / 2].to_bits()
    );
    assert!(cpu[5].value.host_slice().iter().all(|v| v.is_finite()));
    // NaN 位置の `nansum` 勾配は 0（有限の重み × 0）。
    let g = cpu[5].grad.host_slice();
    assert_eq!(g[5], 0.0);
    assert_eq!(g[17], 0.0);
    // `nanmean(None)` の値は非 NaN 22 個の平均。
    let data = input_with_nan().host_slice().into_owned();
    let (sum, n) = data
        .iter()
        .filter(|v| !v.is_nan())
        .fold((0.0_f64, 0usize), |(s, n), &v| (s + f64::from(v), n + 1));
    assert_eq!(n, 22);
    assert!(
        (f64::from(cpu[6].value.host_slice()[0]) - sum / n as f64).abs() < 1e-6,
        "nanmean 値"
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/stat-reduce-ops-2637/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 5 演算の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/stat-reduce-ops-2637/README.md 参照"]
fn metal_stat_reduce_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 5 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/stat-reduce-ops-2637/README.md 参照"]
fn cuda_stat_reduce_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
