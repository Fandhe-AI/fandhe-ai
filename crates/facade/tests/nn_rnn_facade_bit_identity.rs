//! `fandhe_ai::nn::rnn`（`Rnn`／`Lstm`／`Gru`。イシュー #1955）の facade
//! 到達経路（[`fandhe_ai::Tape::rnn_forward_seq`]／`lstm_forward_seq`／
//! `gru_forward_seq`）が、内部クレート `fandhe_ai_autodiff::nn` の
//! 直接呼び出しと CPU で forward・backward とも bit 一致することを
//! 固定する受け入れ基準対応テスト（`docs/compat-api-scope.md` §5
//! 適用記録の受入基準 1）。
//!
//! facade 経路は `Tape::rnn_forward_seq` 等（`&self.0` を渡すだけの
//! 薄い委譲。`crates/facade/src/lib.rs` の `impl Tape` 参照）で数値
//! 経路を一切追加しないため、参照側と同じ `CpuBackendOps` へ結線すれば
//! bit 完全一致になる。参照側の `Tape` は `fandhe_ai_autodiff::Tape::
//! new()`（`NaiveOps`。ホスト参照実装）ではなく、facade `tape()`
//! （`crates/facade/src/lib.rs::tape`）と同じ
//! `fandhe_ai_autodiff::Tape::new_with_ops(Box::new(CpuBackendOps::
//! new()))` で構築する（`NaiveOps` は GEMM の累積順序が異なり bit
//! 一致が保証されないため使わない）。
//!
//! facade 腕（[`facade_side`]）は `fandhe_ai` パスのみを import し、
//! 参照腕（[`reference_side`]）は `fandhe_ai_autodiff`／
//! `fandhe_ai_backend_cpu` を import する。両腕を分離することで
//! 「facade のみの import で `forward_seq` に到達できる」ことを
//! コード上でも裏付ける。
//!
//! CUDA／Metal 実機 parity（受入基準 3）は末尾の `#[ignore]` テスト
//! （REQ-2 統一複合判定。tolerance 定数は新設・変更しない）で行う。
//! 本実装エージェント実行環境には GB10／Apple Silicon 実機への到達
//! 手段がなく未実測のまま申し送る
//! （`docs/perf/logs/facade-nn-rnn-1955/README.md`）。

use fandhe_ai::Device;

const T: usize = 3;
const B: usize = 2;
const D: usize = 4;
const H: usize = 5;

/// 決定的な入力データ（乱数不使用。`arange` 風の式）。`[T,B,D]` の
/// numel = 24。
fn input_data() -> Vec<f32> {
    (0..T * B * D).map(|i| (i as f32) * 0.01 - 0.1).collect()
}

/// facade 経路（`fandhe_ai` のみ import）。
mod facade_side {
    use super::{B, D, H, T, input_data};
    use fandhe_ai::nn::rnn::{Gru, Lstm, Rnn};
    use fandhe_ai::{Tape, Tensor};

    /// `Rnn::forward_seq` の forward・backward の両方を実行し、
    /// `(outputs 全 step + h_n を連結したホスト値, [weight_ih, weight_hh,
    /// bias_ih, bias_hh] の勾配ホスト値)` を返す。`bias=true` 前提
    /// （`bias_ih`／`bias_hh` は必ず `Some`）。`h0` を渡す場合は
    /// `with_h0=true`。
    pub fn rnn_forward_backward(seed: u64, with_h0: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let tape = fandhe_ai::tape();
        let rnn = Rnn::new(D, H, true, seed).expect("test fixture: Rnn::new は有効値のはず");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
        let h0_tensor = Tensor::zeros(&[B, H]).expect("test fixture: zeros は有効値のはず");
        let h0 = with_h0.then(|| tape.var(&h0_tensor));
        let out = tape
            .rnn_forward_seq(&rnn, &x, h0.as_ref())
            .expect("test fixture: rnn_forward_seq は有効値のはず");

        let mut fwd = flatten_seq(&tape, &out.outputs);
        fwd.extend(host_slice(&out.h_n));

        let loss = sum_all(&tape, &out.outputs, &out.h_n, None);
        let grads = tape
            .backward(&loss)
            .expect("test fixture: backward は有効値のはず");
        let params = vec![
            grad_or_empty(&grads, &out.params.weight_ih),
            grad_or_empty(&grads, &out.params.weight_hh),
            grad_or_empty_opt(&grads, out.params.bias_ih.as_ref()),
            grad_or_empty_opt(&grads, out.params.bias_hh.as_ref()),
        ];
        (fwd, params)
    }

    /// `Lstm::forward_seq`。`h0`／`c0` 両方を渡す場合は `with_state=true`。
    /// loss は `outputs` の総和 + `c_n` の総和（c 経路も backward
    /// させるため）。
    pub fn lstm_forward_backward(seed: u64, with_state: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let tape = fandhe_ai::tape();
        let lstm = Lstm::new(D, H, true, seed).expect("test fixture: Lstm::new は有効値のはず");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
        let zero = Tensor::zeros(&[B, H]).expect("test fixture: zeros は有効値のはず");
        let h0 = with_state.then(|| tape.var(&zero));
        let c0 = with_state.then(|| tape.var(&zero));
        let out = tape
            .lstm_forward_seq(&lstm, &x, h0.as_ref(), c0.as_ref())
            .expect("test fixture: lstm_forward_seq は有効値のはず");

        let mut fwd = flatten_seq(&tape, &out.outputs);
        fwd.extend(host_slice(&out.h_n));
        fwd.extend(host_slice(&out.c_n));

        let loss = sum_all(&tape, &out.outputs, &out.h_n, Some(&out.c_n));
        let grads = tape
            .backward(&loss)
            .expect("test fixture: backward は有効値のはず");
        let params = vec![
            grad_or_empty(&grads, &out.params.weight_ih),
            grad_or_empty(&grads, &out.params.weight_hh),
            grad_or_empty_opt(&grads, out.params.bias_ih.as_ref()),
            grad_or_empty_opt(&grads, out.params.bias_hh.as_ref()),
        ];
        (fwd, params)
    }

    /// `Gru::forward_seq`。`Rnn` と同型。
    pub fn gru_forward_backward(seed: u64, with_h0: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let tape = fandhe_ai::tape();
        let gru = Gru::new(D, H, true, seed).expect("test fixture: Gru::new は有効値のはず");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
        let h0_tensor = Tensor::zeros(&[B, H]).expect("test fixture: zeros は有効値のはず");
        let h0 = with_h0.then(|| tape.var(&h0_tensor));
        let out = tape
            .gru_forward_seq(&gru, &x, h0.as_ref())
            .expect("test fixture: gru_forward_seq は有効値のはず");

        let mut fwd = flatten_seq(&tape, &out.outputs);
        fwd.extend(host_slice(&out.h_n));

        let loss = sum_all(&tape, &out.outputs, &out.h_n, None);
        let grads = tape
            .backward(&loss)
            .expect("test fixture: backward は有効値のはず");
        let params = vec![
            grad_or_empty(&grads, &out.params.weight_ih),
            grad_or_empty(&grads, &out.params.weight_hh),
            grad_or_empty_opt(&grads, out.params.bias_ih.as_ref()),
            grad_or_empty_opt(&grads, out.params.bias_hh.as_ref()),
        ];
        (fwd, params)
    }

    fn host_slice(v: &fandhe_ai::Var<'_>) -> Vec<f32> {
        let t = v.to_tensor().contiguous();
        t.as_slice()
            .expect("to_tensor().contiguous() 後は as_slice が必ず Some を返す")
            .to_vec()
    }

    fn flatten_seq(_tape: &Tape, outputs: &[fandhe_ai::Var<'_>]) -> Vec<f32> {
        outputs.iter().flat_map(host_slice).collect()
    }

    /// `outputs` 全 step + `h_n`（+ `c_n`）の総和。LSTM は `c_n` も
    /// loss へ混ぜて c 経路の backward を通す。
    fn sum_all<'t>(
        _tape: &Tape,
        outputs: &[fandhe_ai::Var<'t>],
        h_n: &fandhe_ai::Var<'t>,
        c_n: Option<&fandhe_ai::Var<'t>>,
    ) -> fandhe_ai::Var<'t> {
        let mut acc = h_n.sum(None).expect("test fixture: sum は有効値のはず");
        for o in outputs {
            acc = acc
                .add(&o.sum(None).expect("test fixture: sum は有効値のはず"))
                .expect("test fixture: add は有効値のはず");
        }
        if let Some(c) = c_n {
            acc = acc
                .add(&c.sum(None).expect("test fixture: sum は有効値のはず"))
                .expect("test fixture: add は有効値のはず");
        }
        acc
    }

    fn grad_or_empty(grads: &fandhe_ai::Gradients, v: &fandhe_ai::Var<'_>) -> Vec<f32> {
        grads
            .get(v)
            .expect("test fixture: get は有効値のはず")
            .map(|t| {
                t.contiguous()
                    .as_slice()
                    .expect("contiguous() 後は as_slice が必ず Some を返す")
                    .to_vec()
            })
            .unwrap_or_default()
    }

    fn grad_or_empty_opt(grads: &fandhe_ai::Gradients, v: Option<&fandhe_ai::Var<'_>>) -> Vec<f32> {
        match v {
            Some(v) => grad_or_empty(grads, v),
            None => Vec::new(),
        }
    }
}

/// 参照腕（`fandhe_ai_autodiff`／`fandhe_ai_backend_cpu` を import。
/// facade を経由しない直接呼び出し）。
mod reference_side {
    use super::{B, D, H, T, input_data};
    use fandhe_ai_autodiff::nn::{Gru, Lstm, Rnn};
    use fandhe_ai_autodiff::{Gradients, Tape, Var};
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::Tensor;

    fn cpu_tape() -> Tape {
        Tape::new_with_ops(Box::new(CpuBackendOps::new()))
    }

    pub fn rnn_forward_backward(seed: u64, with_h0: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let tape = cpu_tape();
        let rnn = Rnn::new(D, H, true, seed).expect("test fixture: Rnn::new は有効値のはず");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
        let h0_tensor = Tensor::zeros(&[B, H]).expect("test fixture: zeros は有効値のはず");
        let h0 = with_h0.then(|| tape.var(&h0_tensor));
        let out = rnn
            .forward_seq(&tape, &x, h0.as_ref())
            .expect("test fixture: forward_seq は有効値のはず");

        let mut fwd = flatten_seq(&out.outputs);
        fwd.extend(host_slice(&out.h_n));

        let loss = sum_all(&out.outputs, &out.h_n, None);
        let grads = tape
            .backward(&loss)
            .expect("test fixture: backward は有効値のはず");
        let params = vec![
            grad_or_empty(&grads, &out.params.weight_ih),
            grad_or_empty(&grads, &out.params.weight_hh),
            grad_or_empty_opt(&grads, out.params.bias_ih.as_ref()),
            grad_or_empty_opt(&grads, out.params.bias_hh.as_ref()),
        ];
        (fwd, params)
    }

    pub fn lstm_forward_backward(seed: u64, with_state: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let tape = cpu_tape();
        let lstm = Lstm::new(D, H, true, seed).expect("test fixture: Lstm::new は有効値のはず");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
        let zero = Tensor::zeros(&[B, H]).expect("test fixture: zeros は有効値のはず");
        let h0 = with_state.then(|| tape.var(&zero));
        let c0 = with_state.then(|| tape.var(&zero));
        let out = lstm
            .forward_seq(&tape, &x, h0.as_ref(), c0.as_ref())
            .expect("test fixture: forward_seq は有効値のはず");

        let mut fwd = flatten_seq(&out.outputs);
        fwd.extend(host_slice(&out.h_n));
        fwd.extend(host_slice(&out.c_n));

        let loss = sum_all(&out.outputs, &out.h_n, Some(&out.c_n));
        let grads = tape
            .backward(&loss)
            .expect("test fixture: backward は有効値のはず");
        let params = vec![
            grad_or_empty(&grads, &out.params.weight_ih),
            grad_or_empty(&grads, &out.params.weight_hh),
            grad_or_empty_opt(&grads, out.params.bias_ih.as_ref()),
            grad_or_empty_opt(&grads, out.params.bias_hh.as_ref()),
        ];
        (fwd, params)
    }

    pub fn gru_forward_backward(seed: u64, with_h0: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let tape = cpu_tape();
        let gru = Gru::new(D, H, true, seed).expect("test fixture: Gru::new は有効値のはず");
        let x = Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
        let h0_tensor = Tensor::zeros(&[B, H]).expect("test fixture: zeros は有効値のはず");
        let h0 = with_h0.then(|| tape.var(&h0_tensor));
        let out = gru
            .forward_seq(&tape, &x, h0.as_ref())
            .expect("test fixture: forward_seq は有効値のはず");

        let mut fwd = flatten_seq(&out.outputs);
        fwd.extend(host_slice(&out.h_n));

        let loss = sum_all(&out.outputs, &out.h_n, None);
        let grads = tape
            .backward(&loss)
            .expect("test fixture: backward は有効値のはず");
        let params = vec![
            grad_or_empty(&grads, &out.params.weight_ih),
            grad_or_empty(&grads, &out.params.weight_hh),
            grad_or_empty_opt(&grads, out.params.bias_ih.as_ref()),
            grad_or_empty_opt(&grads, out.params.bias_hh.as_ref()),
        ];
        (fwd, params)
    }

    fn host_slice(v: &Var<'_>) -> Vec<f32> {
        let t = v.to_tensor().contiguous();
        t.as_slice()
            .expect("to_tensor().contiguous() 後は as_slice が必ず Some を返す")
            .to_vec()
    }

    fn flatten_seq(outputs: &[Var<'_>]) -> Vec<f32> {
        outputs.iter().flat_map(host_slice).collect()
    }

    fn sum_all<'t>(outputs: &[Var<'t>], h_n: &Var<'t>, c_n: Option<&Var<'t>>) -> Var<'t> {
        let mut acc = h_n.sum(None).expect("test fixture: sum は有効値のはず");
        for o in outputs {
            acc = acc
                .add(&o.sum(None).expect("test fixture: sum は有効値のはず"))
                .expect("test fixture: add は有効値のはず");
        }
        if let Some(c) = c_n {
            acc = acc
                .add(&c.sum(None).expect("test fixture: sum は有効値のはず"))
                .expect("test fixture: add は有効値のはず");
        }
        acc
    }

    fn grad_or_empty(grads: &Gradients, v: &Var<'_>) -> Vec<f32> {
        grads
            .get(v)
            .expect("test fixture: get は有効値のはず")
            .map(|t| {
                t.contiguous()
                    .as_slice()
                    .expect("contiguous() 後は as_slice が必ず Some を返す")
                    .to_vec()
            })
            .unwrap_or_default()
    }

    fn grad_or_empty_opt(grads: &Gradients, v: Option<&Var<'_>>) -> Vec<f32> {
        match v {
            Some(v) => grad_or_empty(grads, v),
            None => Vec::new(),
        }
    }
}

fn assert_bit_identical_vec(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: 要素数が一致しない（facade={} vs reference={}）",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: 要素 {i} が facade（{a}／bits {:x}）と reference（{e}／bits {:x}）で異なる",
            a.to_bits(),
            e.to_bits()
        );
    }
}

#[test]
fn rnn_forward_seq_facade_matches_reference_bit_exact() {
    for (seed, with_h0) in [(101u64, false), (102, true)] {
        let (facade_fwd, facade_grads) = facade_side::rnn_forward_backward(seed, with_h0);
        let (ref_fwd, ref_grads) = reference_side::rnn_forward_backward(seed, with_h0);
        assert_bit_identical_vec(
            &format!("Rnn forward（seed={seed}, with_h0={with_h0}）"),
            &facade_fwd,
            &ref_fwd,
        );
        let names = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"];
        for (name, (a, e)) in names.iter().zip(facade_grads.iter().zip(ref_grads.iter())) {
            assert_bit_identical_vec(
                &format!("Rnn grad {name}（seed={seed}, with_h0={with_h0}）"),
                a,
                e,
            );
        }
    }
}

#[test]
fn lstm_forward_seq_facade_matches_reference_bit_exact() {
    for (seed, with_state) in [(201u64, false), (202, true)] {
        let (facade_fwd, facade_grads) = facade_side::lstm_forward_backward(seed, with_state);
        let (ref_fwd, ref_grads) = reference_side::lstm_forward_backward(seed, with_state);
        assert_bit_identical_vec(
            &format!("Lstm forward（seed={seed}, with_state={with_state}）"),
            &facade_fwd,
            &ref_fwd,
        );
        let names = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"];
        for (name, (a, e)) in names.iter().zip(facade_grads.iter().zip(ref_grads.iter())) {
            assert_bit_identical_vec(
                &format!("Lstm grad {name}（seed={seed}, with_state={with_state}）"),
                a,
                e,
            );
        }
    }
}

#[test]
fn gru_forward_seq_facade_matches_reference_bit_exact() {
    for (seed, with_h0) in [(301u64, false), (302, true)] {
        let (facade_fwd, facade_grads) = facade_side::gru_forward_backward(seed, with_h0);
        let (ref_fwd, ref_grads) = reference_side::gru_forward_backward(seed, with_h0);
        assert_bit_identical_vec(
            &format!("Gru forward（seed={seed}, with_h0={with_h0}）"),
            &facade_fwd,
            &ref_fwd,
        );
        let names = ["weight_ih", "weight_hh", "bias_ih", "bias_hh"];
        for (name, (a, e)) in names.iter().zip(facade_grads.iter().zip(ref_grads.iter())) {
            assert_bit_identical_vec(
                &format!("Gru grad {name}（seed={seed}, with_h0={with_h0}）"),
                a,
                e,
            );
        }
    }
}

/// bias なし（`bias=false`）ケース: `bias_ih`／`bias_hh` が `None` の
/// まま forward・backward とも bit 一致することを確認する
/// （`weight_ih`／`weight_hh` の勾配比較を含む。codex-review 指摘対応:
/// 当初 forward の `h_n` 比較のみで backward を呼んでいなかった）。
#[test]
fn rnn_forward_seq_no_bias_facade_matches_reference_bit_exact() {
    // `bias=false` は各 `forward_backward` 内で固定していないため、
    // ここでは facade 側・参照側をそれぞれ直接構築して比較する。
    let tape = fandhe_ai::tape();
    let rnn = fandhe_ai::nn::rnn::Rnn::new(D, H, false, 401)
        .expect("test fixture: Rnn::new は有効値のはず");
    let x = fandhe_ai::Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
    let out = tape
        .rnn_forward_seq(&rnn, &x, None)
        .expect("test fixture: rnn_forward_seq は有効値のはず");
    assert!(out.params.bias_ih.is_none());
    assert!(out.params.bias_hh.is_none());

    let ref_tape = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
        fandhe_ai_backend_cpu::CpuBackendOps::new(),
    ));
    let ref_rnn = fandhe_ai_autodiff::nn::Rnn::new(D, H, false, 401)
        .expect("test fixture: Rnn::new は有効値のはず");
    let ref_x = fandhe_ai_tensor_core::Tensor::new(input_data(), &[T, B, D])
        .expect("test fixture: x shape 一致");
    let ref_out = ref_rnn
        .forward_seq(&ref_tape, &ref_x, None)
        .expect("test fixture: forward_seq は有効値のはず");
    assert!(ref_out.params.bias_ih.is_none());
    assert!(ref_out.params.bias_hh.is_none());

    let facade_h_n = out
        .h_n
        .to_tensor()
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec();
    let ref_h_n = ref_out
        .h_n
        .to_tensor()
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec();
    assert_bit_identical_vec("Rnn（bias=false）forward h_n", &facade_h_n, &ref_h_n);

    // 同じ loss（h_n の総和）を両腕で構築し backward を呼んで
    // weight_ih／weight_hh の勾配が bias あり経路と同様に bit 一致
    // することを確認する（bias なし経路特有の勾配欠落を検出する）。
    let facade_loss = out.h_n.sum(None).expect("test fixture: sum は有効値のはず");
    let facade_grads = tape
        .backward(&facade_loss)
        .expect("test fixture: backward は有効値のはず");
    let facade_dw_ih = facade_grads
        .get(&out.params.weight_ih)
        .expect("test fixture: get は有効値のはず")
        .expect("weight_ih は常に勾配を持つはず")
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec();
    let facade_dw_hh = facade_grads
        .get(&out.params.weight_hh)
        .expect("test fixture: get は有効値のはず")
        .expect("weight_hh は常に勾配を持つはず")
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec();

    let ref_loss = ref_out
        .h_n
        .sum(None)
        .expect("test fixture: sum は有効値のはず");
    let ref_grads = ref_tape
        .backward(&ref_loss)
        .expect("test fixture: backward は有効値のはず");
    let ref_dw_ih = ref_grads
        .get(&ref_out.params.weight_ih)
        .expect("test fixture: get は有効値のはず")
        .expect("weight_ih は常に勾配を持つはず")
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec();
    let ref_dw_hh = ref_grads
        .get(&ref_out.params.weight_hh)
        .expect("test fixture: get は有効値のはず")
        .expect("weight_hh は常に勾配を持つはず")
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec();

    assert_bit_identical_vec("Rnn（bias=false）grad weight_ih", &facade_dw_ih, &ref_dw_ih);
    assert_bit_identical_vec("Rnn（bias=false）grad weight_hh", &facade_dw_hh, &ref_dw_hh);
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。REQ-2 統一複合判定） ---

fn rnn_forward_on(device: Device) -> Vec<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let rnn = fandhe_ai::nn::rnn::Rnn::new(D, H, true, 501)
        .expect("test fixture: Rnn::new は有効値のはず");
    let x = fandhe_ai::Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
    let out = tape
        .rnn_forward_seq(&rnn, &x, None)
        .expect("test fixture: rnn_forward_seq は有効値のはず");
    out.h_n
        .to_tensor()
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec()
}

fn rnn_backward_dweight_ih_on(device: Device) -> Vec<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let rnn = fandhe_ai::nn::rnn::Rnn::new(D, H, true, 501)
        .expect("test fixture: Rnn::new は有効値のはず");
    let x = fandhe_ai::Tensor::new(input_data(), &[T, B, D]).expect("test fixture: x shape 一致");
    let out = tape
        .rnn_forward_seq(&rnn, &x, None)
        .expect("test fixture: rnn_forward_seq は有効値のはず");
    let loss = out.h_n.sum(None).expect("test fixture: sum は有効値のはず");
    let grads = tape
        .backward(&loss)
        .expect("test fixture: backward は有効値のはず");
    grads
        .get(&out.params.weight_ih)
        .expect("test fixture: get は有効値のはず")
        .expect("到達する")
        .contiguous()
        .as_slice()
        .expect("as_slice は Some のはず")
        .to_vec()
}

fn assert_req2_parity(label: &str, actual: &[f32], expected: &[f32]) {
    fandhe_ai_backend_cpu::parity::assert_parity(label, actual, expected);
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`crates/tensor-core/src/device.rs`）のため、この variant を参照する
// テスト関数はコンパイル自体を macOS 限定にする（他ファイルと同型）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_rnn_forward_matches_cpu() {
    let metal_out = rnn_forward_on(Device::Metal);
    let cpu_out = rnn_forward_on(Device::Cpu);
    assert_req2_parity(
        "Rnn forward h_n: Metal tape_for vs CPU tape_for",
        &metal_out,
        &cpu_out,
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_rnn_backward_dweight_ih_matches_cpu() {
    let metal_dw = rnn_backward_dweight_ih_on(Device::Metal);
    let cpu_dw = rnn_backward_dweight_ih_on(Device::Cpu);
    assert_req2_parity(
        "Rnn backward dweight_ih: Metal tape_for vs CPU tape_for",
        &metal_dw,
        &cpu_dw,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_rnn_forward_matches_cpu() {
    let cuda_out = rnn_forward_on(Device::Cuda(0));
    let cpu_out = rnn_forward_on(Device::Cpu);
    assert_req2_parity(
        "Rnn forward h_n: CUDA tape_for vs CPU tape_for",
        &cuda_out,
        &cpu_out,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_rnn_backward_dweight_ih_matches_cpu() {
    let cuda_dw = rnn_backward_dweight_ih_on(Device::Cuda(0));
    let cpu_dw = rnn_backward_dweight_ih_on(Device::Cpu);
    assert_req2_parity(
        "Rnn backward dweight_ih: CUDA tape_for vs CPU tape_for",
        &cuda_dw,
        &cpu_dw,
    );
}
