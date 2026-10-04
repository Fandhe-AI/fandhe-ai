//! `fandhe_ai::nn::rnn` の多層版（`StackedRnn`／`StackedLstm`／`StackedGru`・
//! `RnnConfig`。イシュー #2535）の facade 到達経路
//! （[`fandhe_ai::Tape::stacked_rnn_forward_seq`]／`stacked_lstm_forward_seq`／
//! `stacked_gru_forward_seq`）が、内部クレート `fandhe_ai_autodiff::nn` の
//! 直接呼び出しと CPU で forward・backward とも bit 一致することを固定する
//! 受け入れ基準対応テスト（`nn_rnn_facade_bit_identity.rs` と同型。
//! `docs/autodiff-rnn-stacked-config-decision.md` §9）。
//!
//! 参照腕の `Tape` は facade `tape()` と同じ
//! `Tape::new_with_ops(Box::new(CpuBackendOps::new()))`（`NaiveOps` は GEMM
//! 累積順が異なり bit 一致が保証されないため使わない）。facade 腕は
//! `fandhe_ai` パスのみ、参照腕は `fandhe_ai_autodiff`／
//! `fandhe_ai_backend_cpu` を import する。
//!
//! 層間 dropout はグローバル RNG を消費するため、RNG を使うテストは
//! ファイル局所 `Mutex` で直列化し、各腕の直前に `manual_seed` を呼ぶ。
//! CUDA／Metal 実機 parity は `rnn_stacked_backend_parity.rs` の
//! `#[ignore]` テストと `docs/perf/logs/rnn-stacked-2164/README.md` の
//! 申し送りが有効（本ファイルでは新設しない）。

use std::sync::{Mutex, MutexGuard};

const T: usize = 3;
const B: usize = 2;
const D: usize = 4;
const H: usize = 5;

/// 全テストが共有するグローバル RNG の直列化ロック。
fn rng_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn input_data() -> Vec<f32> {
    (0..T * B * D).map(|i| (i as f32) * 0.01 - 0.1).collect()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rnn,
    Lstm,
    Gru,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kind: Kind,
    layers: usize,
    bidirectional: bool,
    dropout: f32,
    with_state: bool,
}

/// forward 値（outputs 全 step・h_n・c_n）と全パラメータ勾配。
type Out = (Vec<f32>, Vec<Vec<f32>>);

mod facade_side {
    use super::{B, Case, D, H, Kind, Out, T, input_data};
    use fandhe_ai::nn::rnn::{RnnConfig, StackedGru, StackedLstm, StackedRnn};
    use fandhe_ai::{Gradients, Tensor, Var};

    fn host(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor()
            .contiguous()
            .as_slice()
            .expect("contiguous 後は Some")
            .to_vec()
    }

    fn grad(g: &Gradients, v: &Var<'_>) -> Vec<f32> {
        g.get(v)
            .expect("get")
            .map(|t| t.contiguous().as_slice().expect("slice").to_vec())
            .unwrap_or_default()
    }

    pub fn run(seed: u64, c: Case) -> Out {
        let tape = fandhe_ai::tape();
        let cfg = RnnConfig::new()
            .with_num_layers(c.layers)
            .with_bidirectional(c.bidirectional)
            .with_dropout(c.dropout);
        let x = Tensor::new(input_data(), &[T, B, D]).expect("x");
        let n = c.layers * if c.bidirectional { 2 } else { 1 };
        let zero = Tensor::<f32>::zeros(&[B, H]).expect("zeros");
        let h0: Vec<Var<'_>> = (0..n).map(|_| tape.var(&zero)).collect();
        let c0: Vec<Var<'_>> = (0..n).map(|_| tape.var(&zero)).collect();
        let h0s = c.with_state.then_some(&h0[..]);
        let c0s = c.with_state.then_some(&c0[..]);

        let mut fwd = Vec::new();
        let (outputs, h_n, c_n, params): (
            Vec<Var<'_>>,
            Vec<Var<'_>>,
            Vec<Var<'_>>,
            Vec<[Var<'_>; 4]>,
        );
        macro_rules! pack {
            ($p:expr) => {
                $p.iter()
                    .map(|p| {
                        [
                            p.weight_ih.clone(),
                            p.weight_hh.clone(),
                            p.bias_ih.clone().expect("bias"),
                            p.bias_hh.clone().expect("bias"),
                        ]
                    })
                    .collect::<Vec<_>>()
            };
        }
        match c.kind {
            Kind::Rnn => {
                let m = StackedRnn::new(D, H, true, seed, cfg).expect("new");
                let o = tape.stacked_rnn_forward_seq(&m, &x, h0s).expect("fwd");
                params = pack!(o.params);
                outputs = o.outputs;
                h_n = o.h_n;
                c_n = Vec::new();
            }
            Kind::Gru => {
                let m = StackedGru::new(D, H, true, seed, cfg).expect("new");
                let o = tape.stacked_gru_forward_seq(&m, &x, h0s).expect("fwd");
                params = pack!(o.params);
                outputs = o.outputs;
                h_n = o.h_n;
                c_n = Vec::new();
            }
            Kind::Lstm => {
                let m = StackedLstm::new(D, H, true, seed, cfg).expect("new");
                let o = tape
                    .stacked_lstm_forward_seq(&m, &x, h0s, c0s)
                    .expect("fwd");
                params = pack!(o.params);
                outputs = o.outputs;
                h_n = o.h_n;
                c_n = o.c_n;
            }
        }
        for o in outputs.iter().chain(&h_n).chain(&c_n) {
            fwd.extend(host(o));
        }
        let mut loss = outputs[0].sum(None).expect("sum");
        for v in outputs[1..].iter().chain(&h_n).chain(&c_n) {
            loss = loss.add(&v.sum(None).expect("sum")).expect("add");
        }
        let grads = tape.backward(&loss).expect("backward");
        let g = params
            .iter()
            .flat_map(|p| p.iter().map(|v| grad(&grads, v)))
            .collect();
        (fwd, g)
    }

    /// facade `Rnn::forward_seq` 側（既定 config 契約の再確認用）。
    pub fn single_layer_rnn(seed: u64) -> Out {
        use fandhe_ai::nn::rnn::Rnn;
        let tape = fandhe_ai::tape();
        let m = Rnn::new(D, H, true, seed).expect("new");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("x");
        let o = tape.rnn_forward_seq(&m, &x, None).expect("fwd");
        let mut fwd = Vec::new();
        for v in o.outputs.iter().chain(std::iter::once(&o.h_n)) {
            fwd.extend(host(v));
        }
        let mut loss = o.outputs[0].sum(None).expect("sum");
        for v in o.outputs[1..].iter().chain(std::iter::once(&o.h_n)) {
            loss = loss.add(&v.sum(None).expect("sum")).expect("add");
        }
        let grads = tape.backward(&loss).expect("backward");
        let p = &o.params;
        let g = vec![
            grad(&grads, &p.weight_ih),
            grad(&grads, &p.weight_hh),
            grad(&grads, p.bias_ih.as_ref().expect("bias")),
            grad(&grads, p.bias_hh.as_ref().expect("bias")),
        ];
        (fwd, g)
    }
}

mod reference_side {
    use super::{B, Case, D, H, Kind, Out, T, input_data};
    use fandhe_ai_autodiff::nn::{RnnConfig, StackedGru, StackedLstm, StackedRnn};
    use fandhe_ai_autodiff::{Gradients, Tape, Var};
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::Tensor;

    fn host(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor()
            .contiguous()
            .as_slice()
            .expect("contiguous 後は Some")
            .to_vec()
    }

    fn grad(g: &Gradients, v: &Var<'_>) -> Vec<f32> {
        g.get(v)
            .expect("get")
            .map(|t| t.contiguous().as_slice().expect("slice").to_vec())
            .unwrap_or_default()
    }

    pub fn run(seed: u64, c: Case) -> Out {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let cfg = RnnConfig::new()
            .with_num_layers(c.layers)
            .with_bidirectional(c.bidirectional)
            .with_dropout(c.dropout);
        let x = Tensor::new(input_data(), &[T, B, D]).expect("x");
        let n = c.layers * if c.bidirectional { 2 } else { 1 };
        let zero = Tensor::<f32>::zeros(&[B, H]).expect("zeros");
        let h0: Vec<Var<'_>> = (0..n).map(|_| tape.var(&zero)).collect();
        let c0: Vec<Var<'_>> = (0..n).map(|_| tape.var(&zero)).collect();
        let h0s = c.with_state.then_some(&h0[..]);
        let c0s = c.with_state.then_some(&c0[..]);

        let mut fwd = Vec::new();
        let (outputs, h_n, c_n, params): (
            Vec<Var<'_>>,
            Vec<Var<'_>>,
            Vec<Var<'_>>,
            Vec<[Var<'_>; 4]>,
        );
        macro_rules! pack {
            ($p:expr) => {
                $p.iter()
                    .map(|p| {
                        [
                            p.weight_ih.clone(),
                            p.weight_hh.clone(),
                            p.bias_ih.clone().expect("bias"),
                            p.bias_hh.clone().expect("bias"),
                        ]
                    })
                    .collect::<Vec<_>>()
            };
        }
        match c.kind {
            Kind::Rnn => {
                let m = StackedRnn::new(D, H, true, seed, cfg).expect("new");
                let o = m.forward_seq(&tape, &x, h0s).expect("fwd");
                params = pack!(o.params);
                outputs = o.outputs;
                h_n = o.h_n;
                c_n = Vec::new();
            }
            Kind::Gru => {
                let m = StackedGru::new(D, H, true, seed, cfg).expect("new");
                let o = m.forward_seq(&tape, &x, h0s).expect("fwd");
                params = pack!(o.params);
                outputs = o.outputs;
                h_n = o.h_n;
                c_n = Vec::new();
            }
            Kind::Lstm => {
                let m = StackedLstm::new(D, H, true, seed, cfg).expect("new");
                let o = m.forward_seq(&tape, &x, h0s, c0s).expect("fwd");
                params = pack!(o.params);
                outputs = o.outputs;
                h_n = o.h_n;
                c_n = o.c_n;
            }
        }
        for o in outputs.iter().chain(&h_n).chain(&c_n) {
            fwd.extend(host(o));
        }
        let mut loss = outputs[0].sum(None).expect("sum");
        for v in outputs[1..].iter().chain(&h_n).chain(&c_n) {
            loss = loss.add(&v.sum(None).expect("sum")).expect("add");
        }
        let grads = tape.backward(&loss).expect("backward");
        let g = params
            .iter()
            .flat_map(|p| p.iter().map(|v| grad(&grads, v)))
            .collect();
        (fwd, g)
    }
}

fn assert_bits_eq(label: &str, a: &Out, b: &Out) {
    assert_eq!(a.0.len(), b.0.len(), "{label}: forward 長");
    for (i, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}: forward[{i}] {x} vs {y}");
    }
    assert_eq!(a.1.len(), b.1.len(), "{label}: 勾配本数");
    for (k, (ga, gb)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(!ga.is_empty(), "{label}: 勾配[{k}] が空（未到達）");
        assert_eq!(ga.len(), gb.len(), "{label}: 勾配[{k}] 長");
        for (i, (x, y)) in ga.iter().zip(gb).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: grad[{k}][{i}] {x} vs {y}"
            );
        }
    }
}

fn check(seed: u64, c: Case) {
    let _g = rng_lock();
    fandhe_ai::manual_seed(1234);
    let f = facade_side::run(seed, c);
    fandhe_ai_autodiff::manual_seed(1234);
    let r = reference_side::run(seed, c);
    assert_bits_eq(&format!("{c:?}"), &f, &r);
}

fn case(kind: Kind, layers: usize, bidirectional: bool, dropout: f32, with_state: bool) -> Case {
    Case {
        kind,
        layers,
        bidirectional,
        dropout,
        with_state,
    }
}

#[test]
fn stacked_bidirectional_two_layers_match_reference() {
    for kind in [Kind::Rnn, Kind::Lstm, Kind::Gru] {
        check(7, case(kind, 2, true, 0.0, false));
    }
}

#[test]
fn stacked_lstm_explicit_state_matches_reference() {
    check(11, case(Kind::Lstm, 2, true, 0.0, true));
    check(12, case(Kind::Gru, 2, true, 0.0, true));
}

#[test]
fn stacked_unidirectional_three_layers_match_reference() {
    for kind in [Kind::Rnn, Kind::Lstm, Kind::Gru] {
        check(13, case(kind, 3, false, 0.0, false));
    }
}

#[test]
fn stacked_dropout_matches_reference_under_same_seed() {
    for kind in [Kind::Rnn, Kind::Lstm, Kind::Gru] {
        check(17, case(kind, 2, false, 0.5, false));
    }
    // dropout が実際に適用されている（0.0 版と出力が異なる）こと。
    let _g = rng_lock();
    fandhe_ai::manual_seed(1234);
    let with = facade_side::run(17, case(Kind::Rnn, 2, false, 0.5, false));
    fandhe_ai::manual_seed(1234);
    let without = facade_side::run(17, case(Kind::Rnn, 2, false, 0.0, false));
    assert_ne!(
        with.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        without.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "dropout=0.5 が forward に反映されていない"
    );
}

/// `RnnConfig::default()`（単層・単方向）の `StackedRnn` が facade `Rnn` と
/// bit 一致する（#2164 の既定 config 契約を facade 経由で再確認）。
#[test]
fn stacked_default_config_matches_facade_single_layer_rnn() {
    let _g = rng_lock();
    let single = facade_side::single_layer_rnn(21);
    let stacked = facade_side::run(21, case(Kind::Rnn, 1, false, 0.0, false));
    assert_bits_eq("default config vs Rnn", &stacked, &single);
}
