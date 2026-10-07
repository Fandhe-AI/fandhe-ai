//! `fandhe_ai_autodiff::nn::packed_sequence`（イシュー #2647・`pack_padded_sequence`／
//! `pad_packed_sequence`／`PackedSequence` と RNN 系の packed 実行。#2679 で
//! `fandhe_ai::nn::rnn` へ公開済みのため、型と自由関数は facade 経由で use する。生の
//! `fandhe_ai_autodiff::Tape`〈`NaiveOps` との突き合わせ用〉だけは内部クレートから import する。
//! `crates/autodiff/src/nn/packed_sequence.rs` モジュール doc 参照）のバックエンド間
//! parity テスト（`tensor_product_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈CPU `BackendOps`〉と `fandhe_ai_autodiff::Tape::new()`
//! 〈`NaiveOps`＝任意メソッドは既定 `Unsupported` → ホスト参照実装へフォールバック〉の
//! 突き合わせ）: pack／unpack の forward はコピーのみのため bit 一致を要求し、RNN の出力と
//! 全 backward は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で
//! 検証する。両経路が同じ誤りで一致していないことを示すため、手計算の期待値も固定する。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os =
//! "macos")` 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行
//! 環境にないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/packed-sequence-2647/README.md`）。形状は小さくし、Metal の split-K
//! 経路が発動する形状は使わない。

use fandhe_ai::Device;
use fandhe_ai::nn::rnn::{
    Gru, Lstm, PackedSequence, Rnn, RnnConfig, StackedGru, StackedLstm, gru_forward_packed,
    lstm_forward_packed, pack_padded_sequence, pad_packed_sequence, rnn_forward_packed,
    stacked_gru_forward_packed, stacked_lstm_forward_packed,
};
use fandhe_ai_autodiff::Var;
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn seq_tensor(shape: &[usize], salt: usize) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(
        (0..n)
            .map(|i| (((i * 13 + salt * 7 + 3) % 31) as f32) * 0.11 - 1.5)
            .collect(),
        shape,
    )
}

struct Out {
    label: &'static str,
    exact_forward: bool,
    values: Vec<Tensor<f32>>,
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

/// 1 ケース分（入力 `Var` → 出力 `Var` 群 → 損失 `Σ w ⊙ y` → 入力と追加 `Var`〈重みなど〉の
/// 勾配）を実行して `outs` へ積む。`$tape` は具象 `Tape` 型ごとに展開するためマクロにしている。
macro_rules! run_case {
    ($tape:expr, $outs:expr, $label:expr, $exact:expr, [$($inp:expr),+], |$xs:ident| $body:expr) => {{
        let tape = &$tape;
        let $xs: Vec<Var<'_>> = vec![$(tape.var(&$inp)),+];
        let (ys, extra): (Vec<Var<'_>>, Vec<Var<'_>>) = $body;
        let values: Vec<Tensor<f32>> = ys.iter().map(|y| y.to_tensor()).collect();
        let mut loss: Option<Var<'_>> = None;
        for y in &ys {
            let w = tape.var(&weights(y.to_tensor().shape()));
            let term = y.mul(&w).unwrap().sum(None).unwrap();
            loss = Some(match loss {
                None => term,
                Some(acc) => acc.add(&term).unwrap(),
            });
        }
        let gs = tape.backward(&loss.unwrap()).unwrap();
        let grads = $xs
            .iter()
            .chain(extra.iter())
            .map(|x| match gs.get(x).unwrap() {
                Some(g) => g.clone(),
                None => Tensor::zeros(x.to_tensor().shape()).unwrap(),
            })
            .collect();
        $outs.push(Out {
            label: $label,
            exact_forward: $exact,
            values,
            grads,
        });
    }};
}

fn cell_bundle<'t>(p: &fandhe_ai_autodiff::nn::RnnCellVars<'t>) -> Vec<Var<'t>> {
    let mut v = vec![p.weight_ih, p.weight_hh];
    v.extend(p.bias_ih);
    v.extend(p.bias_hh);
    v
}

fn gru_bundle<'t>(p: &fandhe_ai_autodiff::nn::GruCellVars<'t>) -> Vec<Var<'t>> {
    let mut v = vec![p.weight_ih, p.weight_hh];
    v.extend(p.bias_ih);
    v.extend(p.bias_hh);
    v
}

fn lstm_bundle<'t>(p: &fandhe_ai_autodiff::nn::LstmCellVars<'t>) -> Vec<Var<'t>> {
    let mut v = vec![p.weight_ih, p.weight_hh];
    v.extend(p.bias_ih);
    v.extend(p.bias_hh);
    v
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
        // pack（batch_first・未整列・タイあり）。
        case!(
            "pack(bf, unsorted)",
            true,
            [seq_tensor(&[4, 5, 3], 1)],
            |xs| {
                let p = pack_padded_sequence(&xs[0], &[2, 5, 3, 5], true, false).unwrap();
                (vec![*p.data()], vec![])
            }
        );
        // unpack（padding_value 非 0・total_length 指定）。
        case!(
            "unpack(pad=-2.5, total=6)",
            true,
            [seq_tensor(&[10, 2], 2)],
            |xs| {
                let p =
                    PackedSequence::new(xs[0], vec![3, 3, 2, 1, 1], Some(vec![1, 0, 2])).unwrap();
                let (padded, _) = pad_packed_sequence(&p, false, -2.5, Some(6)).unwrap();
                (vec![padded], vec![])
            }
        );
        // 単層 RNN／GRU／LSTM（未整列・h0 あり）。
        let rnn = Rnn::new(3, 4, true, 11).unwrap();
        case!(
            "rnn packed",
            false,
            [seq_tensor(&[5, 4, 3], 3), seq_tensor(&[4, 4], 4)],
            |xs| {
                let p = pack_padded_sequence(&xs[0], &[2, 5, 3, 5], false, false).unwrap();
                let o = rnn_forward_packed(&rnn, &p, Some(&xs[1])).unwrap();
                (vec![*o.output.data(), o.h_n], cell_bundle(&o.params))
            }
        );
        let gru = Gru::new(3, 4, true, 12).unwrap();
        case!(
            "gru packed",
            false,
            [seq_tensor(&[5, 4, 3], 5), seq_tensor(&[4, 4], 6)],
            |xs| {
                let p = pack_padded_sequence(&xs[0], &[2, 5, 3, 5], false, false).unwrap();
                let o = gru_forward_packed(&gru, &p, Some(&xs[1])).unwrap();
                (vec![*o.output.data(), o.h_n], gru_bundle(&o.params))
            }
        );
        let lstm = Lstm::new(3, 4, true, 13).unwrap();
        case!(
            "lstm packed",
            false,
            [
                seq_tensor(&[5, 4, 3], 7),
                seq_tensor(&[4, 4], 8),
                seq_tensor(&[4, 4], 9)
            ],
            |xs| {
                let p = pack_padded_sequence(&xs[0], &[2, 5, 3, 5], false, false).unwrap();
                let o = lstm_forward_packed(&lstm, &p, Some(&xs[1]), Some(&xs[2])).unwrap();
                (vec![*o.output.data(), o.h_n, o.c_n], lstm_bundle(&o.params))
            }
        );
        // 多層・双方向（GRU／LSTM）。
        let cfg = RnnConfig::new().with_num_layers(2).with_bidirectional(true);
        let sgru = StackedGru::new(3, 3, true, 21, cfg).unwrap();
        case!(
            "stacked gru 2l bi",
            false,
            [seq_tensor(&[4, 3, 3], 10)],
            |xs| {
                let p = pack_padded_sequence(&xs[0], &[1, 4, 3], false, false).unwrap();
                let o = stacked_gru_forward_packed(&sgru, &p, None).unwrap();
                let mut extra = Vec::new();
                for q in &o.params {
                    extra.extend(gru_bundle(q));
                }
                let mut ys = vec![*o.output.data()];
                ys.extend(o.h_n.iter().copied());
                (ys, extra)
            }
        );
        let slstm = StackedLstm::new(3, 3, true, 22, cfg).unwrap();
        case!(
            "stacked lstm 2l bi",
            false,
            [seq_tensor(&[4, 3, 3], 11)],
            |xs| {
                let p = pack_padded_sequence(&xs[0], &[1, 4, 3], false, false).unwrap();
                let o = stacked_lstm_forward_packed(&slstm, &p, None, None).unwrap();
                let mut extra = Vec::new();
                for q in &o.params {
                    extra.extend(lstm_bundle(q));
                }
                let mut ys = vec![*o.output.data()];
                ys.extend(o.h_n.iter().copied());
                ys.extend(o.c_n.iter().copied());
                (ys, extra)
            }
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
        assert_eq!(x.values.len(), y.values.len(), "{l}: 出力数");
        for (k, (vx, vy)) in x.values.iter().zip(&y.values).enumerate() {
            if x.exact_forward {
                assert_bits_eq(&format!("{l} forward[{k}]"), vx, vy);
            } else {
                assert_parity(&format!("{l} forward[{k}]"), vx, vy);
            }
        }
        assert_eq!(x.grads.len(), y.grads.len(), "{l}: 勾配数");
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
    // [T=3, B=2] time-major。lengths = [2, 3] → 長い系列 b=1 が先頭。
    let x = tape.var(&t(vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0], &[3, 2]));
    let p = pack_padded_sequence(&x, &[2, 3], false, false).unwrap();
    assert_eq!(p.batch_sizes(), &[2, 2, 1]);
    assert_eq!(p.sorted_indices(), Some(&[1usize, 0][..]));
    assert_eq!(
        p.data().to_tensor().host_slice().into_owned(),
        vec![1.0, 0.0, 11.0, 10.0, 21.0]
    );
    // 往復: 範囲外は padding_value、元のバッチ順へ戻る。
    let (padded, lens) = pad_packed_sequence(&p, false, -1.0, Some(4)).unwrap();
    assert_eq!(lens, vec![2, 3]);
    assert_eq!(
        padded.to_tensor().host_slice().into_owned(),
        vec![0.0, 1.0, 10.0, 11.0, -1.0, 21.0, -1.0, -1.0]
    );
    // 恒等に近い RNN: 重みゼロ・バイアスなしなら h は常に 0（tanh(0)）。
    let cell = fandhe_ai_autodiff::nn::RnnCell::from_parameters(
        t(vec![0.0; 2], &[1, 2]),
        t(vec![0.0; 4], &[2, 2]),
        None,
        None,
    )
    .unwrap();
    let rnn = Rnn::from_cell(cell);
    let xs = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2, 1]));
    let p2 = pack_padded_sequence(&xs, &[3, 2], false, true).unwrap();
    let o = rnn_forward_packed(&rnn, &p2, None).unwrap();
    assert_eq!(
        o.output.data().to_tensor().host_slice().into_owned(),
        vec![0.0; 10]
    );
    assert_eq!(o.h_n.to_tensor().host_slice().into_owned(), vec![0.0; 4]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/packed-sequence-2647/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// pack／unpack と RNN 系 packed 実行の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/packed-sequence-2647/README.md 参照"]
fn metal_packed_sequence_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// pack／unpack と RNN 系 packed 実行の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/packed-sequence-2647/README.md 参照"]
fn cuda_packed_sequence_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
