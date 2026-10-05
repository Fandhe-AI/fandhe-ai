//! `fandhe_ai_autodiff::shape_view_ops`（イシュー #2639・`unbind`／`movedim`／`swapaxes`／
//! `tensor_split`／`meshgrid`／`rot90`。facade 非公開のため
//! `fandhe_ai_autodiff::shape_view_ops::*` を直接 use する。
//! `crates/autodiff/src/shape_view_ops.rs` モジュール doc 参照）のバックエンド間 parity
//! テスト（`stat_reduce_ops_backend_parity.rs`・`rearrange_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈CPU `BackendOps`〉と `fandhe_ai_autodiff::Tape::new()`
//! 〈`NaiveOps`＝既定 `Unsupported` → ホスト参照実装へフォールバック〉の突き合わせ）:
//! forward はコピーのみのため NaN の payload まで bit 一致を要求し、backward は REQ-2
//! 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。両経路が同じ
//! 誤りで一致していないことを示すため、手計算の期待値も数件固定する。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os =
//! "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行
//! 環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/shape-view-ops-2639/README.md`）。`rot90` の `gather`／`scatter`
//! は CUDA／Metal が実カーネルを持つため、実機ではフォールバックではなく GPU カーネルが
//! 走る。それ以外の 5 関数は view のみ、または `Op::Narrow` の VJP（`concat` は
//! 3 バックエンドとも未 override でホスト参照実装）である。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::shape_view_ops::{
    MeshgridIndexing, meshgrid, movedim, rot90, swapaxes, tensor_split, tensor_split_indices,
    unbind,
};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

/// `[2, 3, 4]` の入力。一部に NaN（payload 付き）・±inf・-0.0 を混ぜ、コピー演算が値の
/// ビットパターンを保つことも確認する。
fn input() -> Tensor<f32> {
    let mut data: Vec<f32> = (0..24)
        .map(|i| (((i * 13 + 5) % 29) as f32) * 0.21 - 3.0)
        .collect();
    data[3] = f32::from_bits(0x7FC0_0001);
    data[10] = f32::INFINITY;
    data[14] = f32::NEG_INFINITY;
    data[20] = -0.0;
    t(data, &[2, 3, 4])
}

fn vec1(n: usize, salt: usize) -> Tensor<f32> {
    t(
        (0..n)
            .map(|i| ((i * 7 + salt) % 11) as f32 * 0.5 - 2.0)
            .collect(),
        &[n],
    )
}

struct Out {
    label: &'static str,
    values: Vec<Tensor<f32>>,
    grads: Vec<Tensor<f32>>,
}

/// 損失 `Σ w ⊙ y` の重み（出力 shape ごとに決定的に作る）。
fn weights(shape: &[usize], salt: usize) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(
        (0..n)
            .map(|i| 0.5 + 0.3 * (((i + salt) % 3) as f32))
            .collect(),
        shape,
    )
}

/// 1 ケース分（入力 `Var` の列 → 出力 `Var` の列 → 損失 `Σ w ⊙ y` → 全入力の勾配）を
/// 実行して `outs` へ積む。`$tape` は具象 `Tape` 型ごとに展開するためマクロにしている。
macro_rules! run_case {
    ($tape:expr, $outs:expr, $label:expr, [$($inp:expr),+], |$xs:ident| $body:expr) => {{
        let tape = &$tape;
        let $xs: Vec<Var<'_>> = vec![$(tape.var(&$inp)),+];
        let ys: Vec<Var<'_>> = $body;
        let mut loss: Option<Var<'_>> = None;
        let mut values = Vec::new();
        for (k, y) in ys.iter().enumerate() {
            let value = y.to_tensor();
            let w = tape.var(&weights(value.shape(), k));
            let term = y.mul(&w).unwrap().sum(None).unwrap();
            loss = Some(match loss {
                None => term,
                Some(a) => a.add(&term).unwrap(),
            });
            values.push(value);
        }
        let gs = tape.backward(&loss.unwrap()).unwrap();
        let grads = $xs
            .iter()
            .map(|x| gs.get(x).unwrap().unwrap().clone())
            .collect();
        $outs.push(Out {
            label: $label,
            values,
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
        case!("unbind(dim=1)", [input()], |xs| unbind(&xs[0], 1).unwrap());
        case!("unbind(dim=0)", [input()], |xs| unbind(&xs[0], 0).unwrap());
        case!("unbind(transposed, dim=2)", [input()], |xs| {
            let v = xs[0].transpose(0, 2).unwrap();
            unbind(&v, 2).unwrap()
        });
        case!("movedim([0,1],[2,0])", [input()], |xs| vec![
            movedim(&xs[0], &[0, 1], &[2, 0]).unwrap()
        ]);
        case!("swapaxes(0,2)", [input()], |xs| vec![
            swapaxes(&xs[0], 0, 2).unwrap()
        ]);
        case!("tensor_split(3, dim=2)", [input()], |xs| {
            tensor_split(&xs[0], 3, 2).unwrap()
        });
        case!("tensor_split_indices([1,3,2], dim=2)", [input()], |xs| {
            tensor_split_indices(&xs[0], &[1, 3, 2], 2).unwrap()
        });
        case!(
            "meshgrid(xy, 3 inputs)",
            [vec1(3, 1), vec1(2, 2), vec1(2, 3)],
            |xs| { meshgrid(&xs, MeshgridIndexing::Xy).unwrap() }
        );
        case!("meshgrid(ij, 2 inputs)", [vec1(3, 4), vec1(4, 5)], |xs| {
            meshgrid(&xs, MeshgridIndexing::Ij).unwrap()
        });
        case!("rot90(k=1, dims=[0,1])", [input()], |xs| vec![
            rot90(&xs[0], 1, [0, 1]).unwrap()
        ]);
        case!("rot90(k=-1, dims=[0,2])", [input()], |xs| vec![
            rot90(&xs[0], -1, [0, 2]).unwrap()
        ]);
        case!("rot90(k=2, dims=[1,2])", [input()], |xs| vec![
            rot90(&xs[0], 2, [1, 2]).unwrap()
        ]);
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
        assert_eq!(x.values.len(), y.values.len(), "{l}: 出力本数");
        for (k, (vx, vy)) in x.values.iter().zip(&y.values).enumerate() {
            assert_bits_eq(&format!("{l} forward[{k}]"), vx, vy);
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
    // rot90([[1,2,3],[4,5,6]], k=1) = [[3,6],[2,5],[1,4]]。
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let r = rot90(&x, 1, [0, 1]).unwrap();
    assert_eq!(r.to_tensor().shape(), &[3, 2]);
    assert_eq!(
        r.to_tensor().host_slice().into_owned(),
        vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]
    );
    // tensor_split(0..7, 3) の長さは [3, 2, 2]。
    let s = tape.var(&t((0..7).map(|i| i as f32).collect(), &[7]));
    let lens: Vec<usize> = tensor_split(&s, 3, 0)
        .unwrap()
        .iter()
        .map(|p| p.to_tensor().shape()[0])
        .collect();
    assert_eq!(lens, vec![3, 2, 2]);
    // tensor_split_indices の非単調列 [5, 3] は 10 要素を [5, 0, 7] に分ける。
    let s10 = tape.var(&t((0..10).map(|i| i as f32).collect(), &[10]));
    let lens: Vec<usize> = tensor_split_indices(&s10, &[5, 3], 0)
        .unwrap()
        .iter()
        .map(|p| p.to_tensor().shape()[0])
        .collect();
    assert_eq!(lens, vec![5, 0, 7]);
    // unbind(dim=1) の 2 本目は x[:, 1, :]。
    let cube = tape.var(&t((0..24).map(|i| i as f32).collect(), &[2, 3, 4]));
    let parts = unbind(&cube, 1).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(
        parts[1].to_tensor().host_slice().into_owned(),
        vec![4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]
    );
    // meshgrid(xy) は先頭 2 軸を入れ替えた shape [2, 3] を返す。
    let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![10.0, 20.0], &[2]));
    let g = meshgrid(&[a, b], MeshgridIndexing::Xy).unwrap();
    assert_eq!(g[0].to_tensor().shape(), &[2, 3]);
    assert_eq!(
        g[1].to_tensor().host_slice().into_owned(),
        vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    );
    // movedim・swapaxes の shape。
    assert_eq!(
        movedim(&cube, &[0], &[2]).unwrap().to_tensor().shape(),
        &[3, 4, 2]
    );
    assert_eq!(
        swapaxes(&cube, 0, 2).unwrap().to_tensor().shape(),
        &[4, 3, 2]
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/shape-view-ops-2639/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 形状演算 6 種（`unbind`／`movedim`／`swapaxes`／`tensor_split`／`meshgrid`／`rot90`）の
/// forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/shape-view-ops-2639/README.md 参照"]
fn metal_shape_view_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 形状演算 6 種の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/shape-view-ops-2639/README.md 参照"]
fn cuda_shape_view_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
