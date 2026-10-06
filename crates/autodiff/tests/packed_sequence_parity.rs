//! `nn::packed_sequence`（イシュー #2647・`pack_padded_sequence`／`pad_packed_sequence`／
//! `PackedSequence` と `Rnn`／`Lstm`／`Gru`・`Stacked*` の packed 実行）の `Tape`／`Var` を
//! 経由する end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/packed-sequence-pytorch-reference/packed_sequence_reference.json`・
//!   生成条件は同ディレクトリの `README.md`）と突合する。f32 は NaN／inf を運べるよう u32
//!   ビットパターンで保存されている。pack／unpack の forward はコピーのみのため NaN の
//!   payload まで bit 一致を要求し、勾配と RNN 出力は REQ-2 統一複合判定
//!   （`common::req2_close`。tolerance 定数は新設しない）で比較する。
//! - `error_cases` は torch の挙動と本実装のエラー有無を一対一で照合し、意図的な差分
//!   （`INTENDED_DIFFS`）は決定記録 §5 と一対応させる。
//! - 対照ケース: 全系列長 = T の packed 実行が既存 `forward_seq` と bit 一致し、PyTorch の
//!   非 packed 実行とも一致することで、pack 起因の誤りと既存セルの乖離を切り分ける。
//! - 意味論の独立検証: 可変長バッチの packed 実行が、各系列を `B = 1`・自分の長さで個別に
//!   `forward_seq` した結果と一致する。
//! - バックエンド到達性: `gather`／`scatter`／`concat` と LSTM／GRU の pointwise を
//!   `Unsupported` にするモックで、ホスト参照実装へフォールバックすること、他のエラーは
//!   握りつぶさず伝播することを固定する。CPU `BackendOps` 実装との一致は
//!   `crates/facade/tests/packed_sequence_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::nn::packed_sequence::{
    PackedSequence, gru_forward_packed, lstm_forward_packed, pack_padded_sequence,
    pad_packed_sequence, rnn_forward_packed, stacked_gru_forward_packed,
    stacked_lstm_forward_packed, stacked_rnn_forward_packed,
};
use fandhe_ai_autodiff::nn::{
    Gru, GruCell, Lstm, LstmCell, Module, Rnn, RnnCell, RnnConfig, StackedGru, StackedLstm,
    StackedRnn,
};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    BackendOps, GruPointwiseOutput, LstmPointwiseOutput, ScatterReduce, Tensor,
};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn values(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

// =====================================================================
// fixture
// =====================================================================

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    pack_cases: Vec<PackCase>,
    unpack_cases: Vec<UnpackCase>,
    rnn_cases: Vec<RnnCase>,
    error_cases: Vec<ErrCase>,
}

#[derive(Deserialize)]
struct PackCase {
    name: String,
    stored_shape: Vec<usize>,
    pre_transpose: bool,
    batch_first: bool,
    enforce_sorted: bool,
    lengths: Vec<usize>,
    x_bits: Vec<u32>,
    data_shape: Vec<usize>,
    data_bits: Vec<u32>,
    batch_sizes: Vec<usize>,
    sorted_indices: Option<Vec<usize>>,
    unsorted_indices: Option<Vec<usize>>,
    g_bits: Vec<u32>,
    x_grad_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct UnpackCase {
    name: String,
    batch_first: bool,
    padding_value_bits: u32,
    total_length: Option<usize>,
    data_shape: Vec<usize>,
    data_bits: Vec<u32>,
    batch_sizes: Vec<usize>,
    sorted_indices: Option<Vec<usize>>,
    out_shape: Vec<usize>,
    out_bits: Vec<u32>,
    lengths: Vec<usize>,
    g_bits: Vec<u32>,
    data_grad_bits: Vec<u32>,
}

#[derive(Deserialize, Clone)]
struct WeightSet {
    weight_ih_shape: Vec<usize>,
    weight_ih_bits: Vec<u32>,
    weight_hh_shape: Vec<usize>,
    weight_hh_bits: Vec<u32>,
    weight_ih_grad_bits: Vec<u32>,
    weight_hh_grad_bits: Vec<u32>,
    bias_ih_bits: Option<Vec<u32>>,
    bias_hh_bits: Option<Vec<u32>>,
    bias_ih_grad_bits: Option<Vec<u32>>,
    bias_hh_grad_bits: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct RnnCase {
    name: String,
    kind: String,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    bias: bool,
    x_shape: Vec<usize>,
    x_bits: Vec<u32>,
    lengths: Vec<usize>,
    enforce_sorted: bool,
    weights: Vec<WeightSet>,
    h0_bits: Option<Vec<u32>>,
    c0_bits: Option<Vec<u32>>,
    out_shape: Vec<usize>,
    out_bits: Vec<u32>,
    h_n_bits: Vec<u32>,
    c_n_bits: Option<Vec<u32>>,
    batch_sizes: Vec<usize>,
    g1_bits: Vec<u32>,
    g2_bits: Vec<u32>,
    g3_bits: Option<Vec<u32>>,
    x_grad_bits: Vec<u32>,
    h0_grad_bits: Option<Vec<u32>>,
    c0_grad_bits: Option<Vec<u32>>,
    padded_out_bits: Option<Vec<u32>>,
    padded_h_n_bits: Option<Vec<u32>>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    kind: String,
    shape: Option<Vec<usize>>,
    lengths: Option<Vec<usize>>,
    batch_first: Option<bool>,
    enforce_sorted: Option<bool>,
    total_length: Option<usize>,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/packed-sequence-pytorch-reference/packed_sequence_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

/// NaN の payload まで含めて bit 完全一致。
fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{context}[{i}]: actual={a} expected={e}"
        );
    }
}

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            common::req2_close(f64::from(a), f64::from(e)),
            "{context}[{i}]: actual={a} expected={e}"
        );
    }
}

fn grad_or_zeros(grads: &fandhe_ai_autodiff::Gradients, v: &Var<'_>) -> Vec<f32> {
    match grads.get(v).expect("get") {
        Some(g) => g.host_slice().into_owned(),
        None => vec![0.0; v.to_tensor().numel()],
    }
}

// =====================================================================
// pack / unpack
// =====================================================================

#[test]
fn fixture_is_pytorch_2_14_0_and_covers_expected_cases() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "fixture の torch バージョン: {}",
        fx.torch_version
    );
    assert!(fx.pack_cases.len() >= 16, "pack: {}", fx.pack_cases.len());
    assert!(
        fx.unpack_cases.len() >= 12,
        "unpack: {}",
        fx.unpack_cases.len()
    );
    assert!(fx.rnn_cases.len() >= 19, "rnn: {}", fx.rnn_cases.len());
    for kind in ["rnn", "lstm", "gru"] {
        assert!(fx.rnn_cases.iter().any(|c| c.kind == kind), "{kind}");
    }
    // 非有限値（NaN payload・±inf・-0.0）が実際に入力へ入っている。
    let mut payload_nan = false;
    let mut neg_zero = false;
    let mut inf = false;
    for c in fx
        .pack_cases
        .iter()
        .filter(|c| c.name.contains("nonfinite"))
    {
        for &b in &c.x_bits {
            payload_nan |= b == 0x7FC0_0001;
            neg_zero |= b == 0x8000_0000;
            inf |= b == 0x7F80_0000 || b == 0xFF80_0000;
        }
    }
    assert!(payload_nan && neg_zero && inf);
}

#[test]
fn pack_matches_pytorch() {
    let fx = load_fixture();
    for c in &fx.pack_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let leaf = tape.var(&t(from_bits(&c.x_bits), &c.stored_shape));
        let input = if c.pre_transpose {
            leaf.transpose(0, 1).expect("pre_transpose")
        } else {
            leaf
        };
        let p = pack_padded_sequence(&input, &c.lengths, c.batch_first, c.enforce_sorted)
            .unwrap_or_else(|e| panic!("{}: pack 失敗 {e:?}", c.name));
        assert_eq!(p.data().to_tensor().shape(), c.data_shape, "{}", c.name);
        assert_bits_eq(
            &values(p.data()),
            &from_bits(&c.data_bits),
            &format!("{} data", c.name),
        );
        assert_eq!(p.batch_sizes(), c.batch_sizes, "{}", c.name);
        assert_eq!(
            p.sorted_indices().map(<[usize]>::to_vec),
            c.sorted_indices,
            "{}: sorted_indices",
            c.name
        );
        assert_eq!(
            p.unsorted_indices().map(<[usize]>::to_vec),
            c.unsorted_indices,
            "{}: unsorted_indices",
            c.name
        );
        let g = tape.var_no_grad(&t(from_bits(&c.g_bits), &c.data_shape));
        let loss = p.data().mul(&g).expect("mul").sum(None).expect("sum");
        let grads = tape.backward(&loss).expect("backward");
        assert_close_all(
            &grad_or_zeros(&grads, &leaf),
            &from_bits(&c.x_grad_bits),
            &format!("{} x_grad", c.name),
        );
    }
}

#[test]
fn unpack_matches_pytorch() {
    let fx = load_fixture();
    for c in &fx.unpack_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let data = tape.var(&t(from_bits(&c.data_bits), &c.data_shape));
        let ps = PackedSequence::new(data, c.batch_sizes.clone(), c.sorted_indices.clone())
            .unwrap_or_else(|e| panic!("{}: new 失敗 {e:?}", c.name));
        let (out, lens) = pad_packed_sequence(
            &ps,
            c.batch_first,
            f32::from_bits(c.padding_value_bits),
            c.total_length,
        )
        .unwrap_or_else(|e| panic!("{}: unpack 失敗 {e:?}", c.name));
        assert_eq!(out.to_tensor().shape(), c.out_shape, "{}", c.name);
        assert_eq!(lens, c.lengths, "{}: lengths", c.name);
        assert_bits_eq(
            &values(&out),
            &from_bits(&c.out_bits),
            &format!("{} out", c.name),
        );
        let g = tape.var_no_grad(&t(from_bits(&c.g_bits), &c.out_shape));
        let loss = out.mul(&g).expect("mul").sum(None).expect("sum");
        let grads = tape.backward(&loss).expect("backward");
        assert_close_all(
            &grad_or_zeros(&grads, &data),
            &from_bits(&c.data_grad_bits),
            &format!("{} data_grad", c.name),
        );
    }
}

#[test]
fn round_trip_gradient_passes_only_valid_positions() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let lengths = [3usize, 1, 2];
    let x = tape.var(&t((0..12).map(|i| i as f32 + 1.0).collect(), &[4, 3]));
    let p = pack_padded_sequence(&x, &lengths, false, false).expect("pack");
    let (padded, lens) = pad_packed_sequence(&p, false, 0.0, None).expect("unpack");
    assert_eq!(lens, lengths.to_vec());
    let loss = padded.sum(None).expect("sum");
    let grads = tape.backward(&loss).expect("backward");
    let g = grad_or_zeros(&grads, &x);
    for tt in 0..4 {
        for (b, &len) in lengths.iter().enumerate() {
            let expected = if tt < len { 1.0 } else { 0.0 };
            assert_eq!(g[tt * 3 + b], expected, "t={tt} b={b}");
        }
    }
}

// --- error_cases ---

/// torch が例外を出さないのに本実装が型付きエラーにするケース（決定記録 §5）。
/// torch は `lengths` が `T` 超過・バッチ数不一致でも例外を出さず、範囲外参照や
/// 切り詰めた結果を返す。本実装は確保・添字生成の前にエラーにする。
const INTENDED_DIFFS: [&str; 3] = [
    "pack_batch_first_len_gt_T",
    "pack_len_count_mismatch",
    "pack_len_gt_T",
];

#[test]
fn error_cases_are_typed_errors_and_diffs_are_intended() {
    let fx = load_fixture();
    let mut diffs: Vec<String> = Vec::new();
    for e in &fx.error_cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let (before, failed) = match e.kind.as_str() {
            "pack" => {
                let shape = e.shape.clone().expect("shape");
                let x = tape.var(&t(vec![0.0; shape.iter().product()], &shape));
                let before = tape.len();
                let r = pack_padded_sequence(
                    &x,
                    e.lengths.as_deref().expect("lengths"),
                    e.batch_first.expect("bf"),
                    e.enforce_sorted.expect("es"),
                );
                if let Err(err) = &r {
                    assert!(
                        matches!(
                            err,
                            AutodiffError::InvalidArgument(_) | AutodiffError::Shape(_)
                        ),
                        "{}: 想定外のエラー種別 {err:?}",
                        e.name
                    );
                }
                (before, r.is_err())
            }
            "unpack" => {
                let x = tape.var(&t(vec![0.0; 16], &[4, 2, 2]));
                let ps = pack_padded_sequence(&x, &[4, 2], false, true).expect("pack");
                let before = tape.len();
                let r = pad_packed_sequence(&ps, false, 0.0, e.total_length);
                (before, r.is_err())
            }
            other => panic!("未知の kind: {other}"),
        };
        if failed {
            assert_eq!(tape.len(), before, "{}: エラー時にノードを残さない", e.name);
        }
        match (failed, e.torch_raises) {
            (true, true) | (false, false) => {}
            (true, false) => diffs.push(e.name.clone()),
            (false, true) => panic!("{}: torch は例外だが本実装は成功した", e.name),
        }
    }
    let mut expected: Vec<String> = INTENDED_DIFFS.iter().map(|s| s.to_string()).collect();
    diffs.sort();
    expected.sort();
    assert_eq!(diffs, expected, "意図的な差分の集合が決定記録 §5 と不一致");
}

// =====================================================================
// RNN 系
// =====================================================================

/// 全系列の差を吸収するための束ね。`Var` は `Copy`。
#[derive(Clone, Copy)]
struct PVars<'t> {
    w_ih: Var<'t>,
    w_hh: Var<'t>,
    b_ih: Option<Var<'t>>,
    b_hh: Option<Var<'t>>,
}

macro_rules! pvars {
    ($p:expr) => {
        PVars {
            w_ih: $p.weight_ih,
            w_hh: $p.weight_hh,
            b_ih: $p.bias_ih,
            b_hh: $p.bias_hh,
        }
    };
}

struct RnnRun<'t> {
    output: PackedSequence<'t>,
    h_n: Vec<Var<'t>>,
    c_n: Vec<Var<'t>>,
    params: Vec<PVars<'t>>,
}

enum Model {
    Rnn(Rnn),
    Lstm(Lstm),
    Gru(Gru),
    SRnn(StackedRnn),
    SLstm(StackedLstm),
    SGru(StackedGru),
}

/// `(weight_ih, weight_hh, bias_ih, bias_hh)`（本リポの `x·W` レイアウト）。
type CellTensors = (
    Tensor<f32>,
    Tensor<f32>,
    Option<Tensor<f32>>,
    Option<Tensor<f32>>,
);

fn weight_tensors(w: &WeightSet) -> CellTensors {
    let gh = w.weight_hh_shape[1];
    (
        t(from_bits(&w.weight_ih_bits), &w.weight_ih_shape),
        t(from_bits(&w.weight_hh_bits), &w.weight_hh_shape),
        w.bias_ih_bits.as_ref().map(|b| t(from_bits(b), &[gh])),
        w.bias_hh_bits.as_ref().map(|b| t(from_bits(b), &[gh])),
    )
}

/// fixture の重みを載せたモデルを組む。`stacked` が偽で単層・単方向のときは `Rnn`／
/// `Lstm`／`Gru`、それ以外は `Stacked*` を使う（接頭辞は `l{layer}[_reverse].*`）。
fn build_model(c: &RnnCase, stacked: bool) -> Model {
    let single = c.num_layers == 1 && !c.bidirectional;
    if single && !stacked {
        let (wi, wh, bi, bh) = weight_tensors(&c.weights[0]);
        return match c.kind.as_str() {
            "rnn" => Model::Rnn(Rnn::from_cell(
                RnnCell::from_parameters(wi, wh, bi, bh).expect("cell"),
            )),
            "lstm" => Model::Lstm(Lstm::from_cell(
                LstmCell::from_parameters(wi, wh, bi, bh).expect("cell"),
            )),
            "gru" => Model::Gru(Gru::from_cell(
                GruCell::from_parameters(wi, wh, bi, bh).expect("cell"),
            )),
            other => panic!("未知の kind: {other}"),
        };
    }
    let cfg = RnnConfig::new()
        .with_num_layers(c.num_layers)
        .with_bidirectional(c.bidirectional);
    let dirs = if c.bidirectional { 2 } else { 1 };
    macro_rules! fill {
        ($m:expr) => {{
            let mut m = $m;
            for layer in 0..c.num_layers {
                for d in 0..dirs {
                    let k = layer * dirs + d;
                    let prefix = if d == 0 {
                        format!("l{layer}")
                    } else {
                        format!("l{layer}_reverse")
                    };
                    let (wi, wh, bi, bh) = weight_tensors(&c.weights[k]);
                    m.set_parameter(&format!("{prefix}.weight_ih"), wi)
                        .expect("w_ih");
                    m.set_parameter(&format!("{prefix}.weight_hh"), wh)
                        .expect("w_hh");
                    if let Some(b) = bi {
                        m.set_parameter(&format!("{prefix}.bias_ih"), b)
                            .expect("b_ih");
                    }
                    if let Some(b) = bh {
                        m.set_parameter(&format!("{prefix}.bias_hh"), b)
                            .expect("b_hh");
                    }
                }
            }
            m
        }};
    }
    let (d, h, bias) = (c.input_size, c.hidden_size, c.bias);
    match c.kind.as_str() {
        "rnn" => Model::SRnn(fill!(StackedRnn::new(d, h, bias, 1, cfg).expect("new"))),
        "lstm" => Model::SLstm(fill!(StackedLstm::new(d, h, bias, 1, cfg).expect("new"))),
        "gru" => Model::SGru(fill!(StackedGru::new(d, h, bias, 1, cfg).expect("new"))),
        other => panic!("未知の kind: {other}"),
    }
}

fn run_model<'t>(
    model: &Model,
    input: &PackedSequence<'t>,
    h0: Option<&[Var<'t>]>,
    c0: Option<&[Var<'t>]>,
) -> Result<RnnRun<'t>, AutodiffError> {
    Ok(match model {
        Model::Rnn(m) => {
            let o = rnn_forward_packed(m, input, h0.map(|s| &s[0]))?;
            RnnRun {
                output: o.output,
                h_n: vec![o.h_n],
                c_n: vec![],
                params: vec![pvars!(o.params)],
            }
        }
        Model::Gru(m) => {
            let o = gru_forward_packed(m, input, h0.map(|s| &s[0]))?;
            RnnRun {
                output: o.output,
                h_n: vec![o.h_n],
                c_n: vec![],
                params: vec![pvars!(o.params)],
            }
        }
        Model::Lstm(m) => {
            let o = lstm_forward_packed(m, input, h0.map(|s| &s[0]), c0.map(|s| &s[0]))?;
            RnnRun {
                output: o.output,
                h_n: vec![o.h_n],
                c_n: vec![o.c_n],
                params: vec![pvars!(o.params)],
            }
        }
        Model::SRnn(m) => {
            let o = stacked_rnn_forward_packed(m, input, h0)?;
            RnnRun {
                output: o.output,
                h_n: o.h_n,
                c_n: vec![],
                params: o.params.iter().map(|p| pvars!(p)).collect(),
            }
        }
        Model::SGru(m) => {
            let o = stacked_gru_forward_packed(m, input, h0)?;
            RnnRun {
                output: o.output,
                h_n: o.h_n,
                c_n: vec![],
                params: o.params.iter().map(|p| pvars!(p)).collect(),
            }
        }
        Model::SLstm(m) => {
            let o = stacked_lstm_forward_packed(m, input, h0, c0)?;
            RnnRun {
                output: o.output,
                h_n: o.h_n,
                c_n: o.c_n,
                params: o.params.iter().map(|p| pvars!(p)).collect(),
            }
        }
    })
}

/// `[L*dirs, B, H]` の bits を `[B, H]` の葉 `L*dirs` 個へ分ける。
fn state_leaves<'t>(
    tape: &'t Tape,
    bits: &[u32],
    cells: usize,
    b: usize,
    h: usize,
) -> Vec<Var<'t>> {
    let v = from_bits(bits);
    (0..cells)
        .map(|k| tape.var(&t(v[k * b * h..(k + 1) * b * h].to_vec(), &[b, h])))
        .collect()
}

/// 1 件の RNN fixture を実行し、(forward 出力, 勾配) を fixture と突合する。
fn check_rnn_case(ops: Box<dyn BackendOps + Send>, c: &RnnCase, stacked: bool) {
    let label = format!("{}{}", c.name, if stacked { " [stacked]" } else { "" });
    let dirs = if c.bidirectional { 2 } else { 1 };
    let cells = c.num_layers * dirs;
    let b = c.x_shape[1];
    let h = c.hidden_size;
    let tape = Tape::new_with_ops(ops);
    let model = build_model(c, stacked);
    let x = tape.var(&t(from_bits(&c.x_bits), &c.x_shape));
    let packed = pack_padded_sequence(&x, &c.lengths, false, c.enforce_sorted).expect("pack");
    assert_eq!(packed.batch_sizes(), c.batch_sizes, "{label}: batch_sizes");
    let h0 = c
        .h0_bits
        .as_ref()
        .map(|bits| state_leaves(&tape, bits, cells, b, h));
    let c0 = c
        .c0_bits
        .as_ref()
        .map(|bits| state_leaves(&tape, bits, cells, b, h));
    let run = run_model(&model, &packed, h0.as_deref(), c0.as_deref())
        .unwrap_or_else(|e| panic!("{label}: forward 失敗 {e:?}"));

    assert_eq!(
        run.output.data().to_tensor().shape(),
        c.out_shape,
        "{label}: out shape"
    );
    assert_close_all(
        &values(run.output.data()),
        &from_bits(&c.out_bits),
        &format!("{label} out"),
    );
    let h_n_fx = from_bits(&c.h_n_bits);
    assert_eq!(run.h_n.len(), cells, "{label}");
    for (k, v) in run.h_n.iter().enumerate() {
        assert_close_all(
            &values(v),
            &h_n_fx[k * b * h..(k + 1) * b * h],
            &format!("{label} h_n[{k}]"),
        );
    }
    if let Some(bits) = &c.c_n_bits {
        let c_n_fx = from_bits(bits);
        for (k, v) in run.c_n.iter().enumerate() {
            assert_close_all(
                &values(v),
                &c_n_fx[k * b * h..(k + 1) * b * h],
                &format!("{label} c_n[{k}]"),
            );
        }
    }

    // 損失 = Σ(out.data ⊙ g1) + Σ(h_n ⊙ g2) (+ Σ(c_n ⊙ g3))
    let g1 = tape.var_no_grad(&t(from_bits(&c.g1_bits), &c.out_shape));
    let mut loss = run
        .output
        .data()
        .mul(&g1)
        .expect("mul")
        .sum(None)
        .expect("sum");
    let g2 = from_bits(&c.g2_bits);
    for (k, v) in run.h_n.iter().enumerate() {
        let g = tape.var_no_grad(&t(g2[k * b * h..(k + 1) * b * h].to_vec(), &[b, h]));
        loss = loss
            .add(&v.mul(&g).expect("mul").sum(None).expect("sum"))
            .expect("add");
    }
    if let Some(bits) = &c.g3_bits {
        let g3 = from_bits(bits);
        for (k, v) in run.c_n.iter().enumerate() {
            let g = tape.var_no_grad(&t(g3[k * b * h..(k + 1) * b * h].to_vec(), &[b, h]));
            loss = loss
                .add(&v.mul(&g).expect("mul").sum(None).expect("sum"))
                .expect("add");
        }
    }
    let grads = tape.backward(&loss).expect("backward");

    assert_close_all(
        &grad_or_zeros(&grads, &x),
        &from_bits(&c.x_grad_bits),
        &format!("{label} x_grad"),
    );
    for (k, (p, w)) in run.params.iter().zip(&c.weights).enumerate() {
        assert_close_all(
            &grad_or_zeros(&grads, &p.w_ih),
            &from_bits(&w.weight_ih_grad_bits),
            &format!("{label} w_ih_grad[{k}]"),
        );
        assert_close_all(
            &grad_or_zeros(&grads, &p.w_hh),
            &from_bits(&w.weight_hh_grad_bits),
            &format!("{label} w_hh_grad[{k}]"),
        );
        if let (Some(bi), Some(bh)) = (&p.b_ih, &p.b_hh) {
            assert_close_all(
                &grad_or_zeros(&grads, bi),
                &from_bits(w.bias_ih_grad_bits.as_ref().expect("bias grad")),
                &format!("{label} b_ih_grad[{k}]"),
            );
            assert_close_all(
                &grad_or_zeros(&grads, bh),
                &from_bits(w.bias_hh_grad_bits.as_ref().expect("bias grad")),
                &format!("{label} b_hh_grad[{k}]"),
            );
        }
    }
    if let (Some(h0), Some(bits)) = (&h0, &c.h0_grad_bits) {
        let fx = from_bits(bits);
        for (k, v) in h0.iter().enumerate() {
            assert_close_all(
                &grad_or_zeros(&grads, v),
                &fx[k * b * h..(k + 1) * b * h],
                &format!("{label} h0_grad[{k}]"),
            );
        }
    }
    if let (Some(c0), Some(bits)) = (&c0, &c.c0_grad_bits) {
        let fx = from_bits(bits);
        for (k, v) in c0.iter().enumerate() {
            assert_close_all(
                &grad_or_zeros(&grads, v),
                &fx[k * b * h..(k + 1) * b * h],
                &format!("{label} c0_grad[{k}]"),
            );
        }
    }
}

#[test]
fn rnn_packed_matches_pytorch() {
    let fx = load_fixture();
    for c in &fx.rnn_cases {
        check_rnn_case(common::naive_ops(), c, false);
        if c.num_layers == 1 && !c.bidirectional {
            // 単層・単方向は Stacked 経路でも同じ値になる。
            check_rnn_case(common::naive_ops(), c, true);
        }
    }
}

/// モデルの padded 版 `forward_seq`（step ごとの出力 `[B, dirs*H]`・`h_n`・`c_n`）。
type SeqOut<'t> = (Vec<Var<'t>>, Vec<Var<'t>>, Vec<Var<'t>>);

fn forward_seq_of<'t>(
    model: &Model,
    tape: &'t Tape,
    x: &Tensor<f32>,
    h0: Option<&[Var<'t>]>,
    c0: Option<&[Var<'t>]>,
) -> SeqOut<'t> {
    match model {
        Model::Rnn(m) => {
            let o = m.forward_seq(tape, x, h0.map(|s| &s[0])).expect("seq");
            (o.outputs, vec![o.h_n], vec![])
        }
        Model::Gru(m) => {
            let o = m.forward_seq(tape, x, h0.map(|s| &s[0])).expect("seq");
            (o.outputs, vec![o.h_n], vec![])
        }
        Model::Lstm(m) => {
            let o = m
                .forward_seq(tape, x, h0.map(|s| &s[0]), c0.map(|s| &s[0]))
                .expect("seq");
            (o.outputs, vec![o.h_n], vec![o.c_n])
        }
        Model::SRnn(m) => {
            let o = m.forward_seq(tape, x, h0).expect("seq");
            (o.outputs, o.h_n, vec![])
        }
        Model::SGru(m) => {
            let o = m.forward_seq(tape, x, h0).expect("seq");
            (o.outputs, o.h_n, vec![])
        }
        Model::SLstm(m) => {
            let o = m.forward_seq(tape, x, h0, c0).expect("seq");
            (o.outputs, o.h_n, o.c_n)
        }
    }
}

/// 対照ケース: 全系列長 = T の packed 実行は、既存 `forward_seq` と bit 一致し、PyTorch の
/// 非 packed 実行とも REQ-2 判定で一致する。
#[test]
fn full_length_packed_equals_forward_seq_and_pytorch_padded() {
    let fx = load_fixture();
    let controls: Vec<&RnnCase> = fx
        .rnn_cases
        .iter()
        .filter(|c| c.name.contains("control"))
        .collect();
    assert_eq!(controls.len(), 3);
    for c in controls {
        let stacked = !(c.num_layers == 1 && !c.bidirectional);
        let model = build_model(c, stacked);
        let dirs = if c.bidirectional { 2 } else { 1 };
        let (t_len, b, _d) = (c.x_shape[0], c.x_shape[1], c.x_shape[2]);
        let h = c.hidden_size;
        let tape = Tape::new_with_ops(common::naive_ops());
        let xt = t(from_bits(&c.x_bits), &c.x_shape);
        let x = tape.var(&xt);
        let packed = pack_padded_sequence(&x, &c.lengths, false, true).expect("pack");
        let run = run_model(&model, &packed, None, None).expect("packed");
        let (padded, _) = pad_packed_sequence(&run.output, false, 0.0, None).expect("unpack");
        let (outs, h_n, _c_n) = forward_seq_of(&model, &tape, &xt, None, None);
        let mut seq_flat: Vec<f32> = Vec::new();
        for o in &outs {
            seq_flat.extend(values(o));
        }
        assert_eq!(padded.to_tensor().shape(), &[t_len, b, dirs * h]);
        assert_bits_eq(
            &values(&padded),
            &seq_flat,
            &format!("{} vs forward_seq", c.name),
        );
        for (k, v) in run.h_n.iter().enumerate() {
            assert_bits_eq(
                &values(v),
                &values(&h_n[k]),
                &format!("{} h_n[{k}]", c.name),
            );
        }
        assert_close_all(
            &values(&padded),
            &from_bits(c.padded_out_bits.as_ref().expect("padded")),
            &format!("{} vs torch padded", c.name),
        );
        let ph = from_bits(c.padded_h_n_bits.as_ref().expect("padded h_n"));
        for (k, v) in run.h_n.iter().enumerate() {
            assert_close_all(
                &values(v),
                &ph[k * b * h..(k + 1) * b * h],
                &format!("{} vs torch padded h_n[{k}]", c.name),
            );
        }
    }
}

/// 意味論の独立検証: 可変長バッチの packed 実行は、各系列を `B = 1`・自分の長さで個別に
/// `forward_seq` した結果（各 step 出力と `h_n`）と一致する。
#[test]
fn packed_batch_equals_per_sequence_forward_seq() {
    let fx = load_fixture();
    for c in fx.rnn_cases.iter().filter(|c| !c.name.contains("control")) {
        for stacked in [false, true] {
            if stacked && !(c.num_layers == 1 && !c.bidirectional) {
                continue;
            }
            let label = format!("{}{}", c.name, if stacked { " [stacked]" } else { "" });
            let model = build_model(c, stacked);
            let dirs = if c.bidirectional { 2 } else { 1 };
            let cells = c.num_layers * dirs;
            let (b_len, d) = (c.x_shape[1], c.x_shape[2]);
            let h = c.hidden_size;
            let xv = from_bits(&c.x_bits);

            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(xv.clone(), &c.x_shape));
            let packed =
                pack_padded_sequence(&x, &c.lengths, false, c.enforce_sorted).expect("pack");
            let h0 = c
                .h0_bits
                .as_ref()
                .map(|bits| state_leaves(&tape, bits, cells, b_len, h));
            let c0 = c
                .c0_bits
                .as_ref()
                .map(|bits| state_leaves(&tape, bits, cells, b_len, h));
            let run = run_model(&model, &packed, h0.as_deref(), c0.as_deref()).expect("packed");
            let (padded, lens) =
                pad_packed_sequence(&run.output, false, 0.0, None).expect("unpack");
            let padded_v = values(&padded);
            assert_eq!(lens, c.lengths, "{label}");
            let t_out = padded.to_tensor().shape()[0];

            for (b, &len) in c.lengths.iter().enumerate() {
                let mut xb: Vec<f32> = Vec::with_capacity(len * d);
                for tt in 0..len {
                    xb.extend_from_slice(&xv[(tt * b_len + b) * d..(tt * b_len + b + 1) * d]);
                }
                let xt = t(xb, &[len, 1, d]);
                let slice_state = |bits: &Option<Vec<u32>>| -> Option<Vec<Var<'_>>> {
                    bits.as_ref().map(|bits| {
                        let v = from_bits(bits);
                        (0..cells)
                            .map(|k| {
                                let off = (k * b_len + b) * h;
                                tape.var(&t(v[off..off + h].to_vec(), &[1, h]))
                            })
                            .collect()
                    })
                };
                let h0b = slice_state(&c.h0_bits);
                let c0b = slice_state(&c.c0_bits);
                let (outs, h_n, c_n) =
                    forward_seq_of(&model, &tape, &xt, h0b.as_deref(), c0b.as_deref());
                for (tt, o) in outs.iter().enumerate() {
                    let ov = values(o);
                    let base = (tt * b_len + b) * dirs * h;
                    assert_close_all(
                        &padded_v[base..base + dirs * h],
                        &ov,
                        &format!("{label} seq{b} step{tt}"),
                    );
                }
                // 範囲外はパディング値 0。
                for tt in len..t_out {
                    let base = (tt * b_len + b) * dirs * h;
                    assert!(padded_v[base..base + dirs * h].iter().all(|&v| v == 0.0));
                }
                for (k, v) in h_n.iter().enumerate() {
                    let got = values(&run.h_n[k]);
                    assert_close_all(
                        &got[b * h..(b + 1) * h],
                        &values(v),
                        &format!("{label} seq{b} h_n[{k}]"),
                    );
                }
                for (k, v) in c_n.iter().enumerate() {
                    let got = values(&run.c_n[k]);
                    assert_close_all(
                        &got[b * h..(b + 1) * h],
                        &values(v),
                        &format!("{label} seq{b} c_n[{k}]"),
                    );
                }
            }
        }
    }
}

// =====================================================================
// 入力検証・決定性・dropout
// =====================================================================

fn small_input(tape: &Tape) -> PackedSequence<'_> {
    let x = tape.var(&t((0..12).map(|i| i as f32 * 0.1).collect(), &[3, 2, 2]));
    pack_padded_sequence(&x, &[3, 2], false, true).expect("pack")
}

#[test]
fn invalid_arguments_are_rejected_before_any_node_is_pushed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let packed = small_input(&tape);
    let rnn = Rnn::new(2, 3, true, 1).expect("rnn");
    let lstm = Lstm::new(2, 3, true, 1).expect("lstm");

    // h0 の shape 不一致（B=2 に対し 3 行）。
    let bad_h0 = tape.var(&Tensor::zeros(&[3, 3]).expect("zeros"));
    let before = tape.len();
    assert!(rnn_forward_packed(&rnn, &packed, Some(&bad_h0)).is_err());
    assert!(lstm_forward_packed(&lstm, &packed, Some(&bad_h0), None).is_err());
    assert_eq!(tape.len(), before, "bind 前に拒否する");

    // 入力幅の不一致。
    let wide = Rnn::new(5, 3, true, 1).expect("rnn");
    assert!(rnn_forward_packed(&wide, &packed, None).is_err());
    assert_eq!(tape.len(), before);

    // クロステープの h0。
    let other = Tape::new_with_ops(common::naive_ops());
    let foreign = other.var(&Tensor::zeros(&[2, 3]).expect("zeros"));
    assert!(matches!(
        rnn_forward_packed(&rnn, &packed, Some(&foreign)),
        Err(AutodiffError::TapeMismatch)
    ));
    assert_eq!(tape.len(), before);

    // stacked の状態スライス長不一致。
    let cfg = RnnConfig::new().with_num_layers(2);
    let st = StackedRnn::new(2, 3, true, 1, cfg).expect("stacked");
    let one = [tape.var(&Tensor::zeros(&[2, 3]).expect("zeros"))];
    let before = tape.len();
    assert!(stacked_rnn_forward_packed(&st, &packed, Some(&one)).is_err());
    assert_eq!(tape.len(), before);
}

#[test]
fn packed_execution_is_bit_deterministic() {
    let fx = load_fixture();
    let c = fx
        .rnn_cases
        .iter()
        .find(|c| c.name == "lstm_2l_bi_h0c0")
        .expect("case");
    let run_once = || -> Vec<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let model = build_model(c, true);
        let x = tape.var(&t(from_bits(&c.x_bits), &c.x_shape));
        let p = pack_padded_sequence(&x, &c.lengths, false, c.enforce_sorted).expect("pack");
        let run = run_model(&model, &p, None, None).expect("run");
        let mut out = values(run.output.data());
        for v in &run.h_n {
            out.extend(values(v));
        }
        out
    };
    assert_bits_eq(&run_once(), &run_once(), "決定性");
}

#[test]
fn interlayer_dropout_is_noop_in_eval_and_for_zero_probability() {
    let fx = load_fixture();
    let c = fx
        .rnn_cases
        .iter()
        .find(|c| c.name == "gru_2l")
        .expect("case");
    let base = |p: f32, training: bool| -> Vec<f32> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let cfg = RnnConfig::new().with_num_layers(2).with_dropout(p);
        let mut m = StackedGru::new(c.input_size, c.hidden_size, c.bias, 1, cfg).expect("new");
        let src = build_model(c, true);
        if let Model::SGru(s) = &src {
            for (name, v) in s.named_parameters() {
                m.set_parameter(&name, v.clone()).expect("set");
            }
        }
        m.set_training(training);
        let x = tape.var(&t(from_bits(&c.x_bits), &c.x_shape));
        let packed = pack_padded_sequence(&x, &c.lengths, false, c.enforce_sorted).expect("pack");
        values(
            stacked_gru_forward_packed(&m, &packed, None)
                .expect("run")
                .output
                .data(),
        )
    };
    let reference = base(0.0, true);
    assert_bits_eq(&base(0.5, false), &reference, "eval では no-op");
    assert_bits_eq(&base(0.0, true), &reference, "p=0 では no-op");
}

// =====================================================================
// バックエンド到達性（モック）
// =====================================================================

/// `gather`／`scatter`／`concat` と LSTM／GRU の pointwise だけを差し替える `BackendOps`。
/// 他は naive へ委譲する。
struct ReachMock {
    inner: Box<dyn BackendOps + Send>,
    calls: Arc<[AtomicUsize; 5]>,
    launch_failed: bool,
}

const GATHER: usize = 0;
const SCATTER: usize = 1;
const CONCAT: usize = 2;
const LSTM_PW: usize = 3;
const GRU_PW: usize = 4;

impl ReachMock {
    fn answer(&self, which: usize) -> BackendError {
        self.calls[which].fetch_add(1, Ordering::SeqCst);
        if self.launch_failed {
            BackendError::KernelLaunchFailed("simulated".into())
        } else {
            BackendError::Unsupported("mock".into())
        }
    }
}

impl BackendOps for ReachMock {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    fn gather(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(GATHER))
    }
    fn scatter(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(SCATTER))
    }
    fn concat(&self, _inputs: &[&Tensor<f32>], _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(self.answer(CONCAT))
    }
    fn lstm_pointwise(
        &self,
        _pre: &Tensor<f32>,
        _c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        Err(self.answer(LSTM_PW))
    }
    fn gru_pointwise(
        &self,
        _pre_i: &Tensor<f32>,
        _pre_h: &Tensor<f32>,
        _h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        Err(self.answer(GRU_PW))
    }
}

fn mock_tape(launch_failed: bool) -> (Tape, Arc<[AtomicUsize; 5]>) {
    let calls = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
    let tape = Tape::new_with_ops(Box::new(ReachMock {
        inner: common::naive_ops(),
        calls: Arc::clone(&calls),
        launch_failed,
    }));
    (tape, calls)
}

fn count(calls: &Arc<[AtomicUsize; 5]>, which: usize) -> usize {
    calls[which].load(Ordering::SeqCst)
}

/// 非 Unsupported のとき `out` と `h_n` の合計を損失に backward する。
fn packed_lstm_loss(
    tape: &Tape,
    c: &RnnCase,
) -> Result<(Vec<f32>, fandhe_ai_autodiff::Gradients), AutodiffError> {
    let model = build_model(c, true);
    let x = tape.var(&t(from_bits(&c.x_bits), &c.x_shape));
    let p = pack_padded_sequence(&x, &c.lengths, false, c.enforce_sorted)?;
    let run = run_model(&model, &p, None, None)?;
    let mut loss = run.output.data().sum(None)?;
    for v in run.h_n.iter().chain(&run.c_n) {
        loss = loss.add(&v.sum(None)?)?;
    }
    let (padded, _) = pad_packed_sequence(&run.output, true, 0.0, None)?;
    let out = values(&padded);
    let grads = tape.backward(&loss)?;
    Ok((out, grads))
}

#[test]
fn unsupported_backend_falls_back_to_host_reference() {
    let fx = load_fixture();
    for (name, pointwise) in [("lstm_2l_bi_h0c0", LSTM_PW), ("gru_2l_bi_h0", GRU_PW)] {
        let c = fx.rnn_cases.iter().find(|c| c.name == name).expect("case");
        let (tape, calls) = mock_tape(false);
        let (out_mock, grads_mock) = packed_lstm_loss(&tape, c).expect("fallback で成功する");
        assert!(
            count(&calls, GATHER) >= 1,
            "{name}: pack の gather を先に呼ぶ"
        );
        assert!(
            count(&calls, CONCAT) >= 1,
            "{name}: cat の concat を先に呼ぶ"
        );
        assert!(
            count(&calls, pointwise) >= 1,
            "{name}: セルの pointwise を先に呼ぶ"
        );
        assert!(
            count(&calls, SCATTER) >= 1,
            "{name}: backward で scatter を先に呼ぶ"
        );

        // 参照（naive）と一致する。
        let ref_tape = Tape::new_with_ops(common::naive_ops());
        let (out_ref, _) = packed_lstm_loss(&ref_tape, c).expect("naive");
        assert_close_all(&out_mock, &out_ref, &format!("{name} mock vs naive"));
        drop(grads_mock);
    }
}

#[test]
fn non_unsupported_backend_errors_are_propagated() {
    let fx = load_fixture();
    let c = fx
        .rnn_cases
        .iter()
        .find(|c| c.name == "lstm_2l_bi_h0c0")
        .expect("case");
    // pack の gather。
    let (tape, _) = mock_tape(true);
    let x = tape.var(&t(from_bits(&c.x_bits), &c.x_shape));
    assert!(matches!(
        pack_padded_sequence(&x, &c.lengths, false, false),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    // 全体（どこで失敗しても Backend エラーのまま伝播し、Unsupported へ化けない）。
    let (tape, _) = mock_tape(true);
    assert!(matches!(
        packed_lstm_loss(&tape, c),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}
