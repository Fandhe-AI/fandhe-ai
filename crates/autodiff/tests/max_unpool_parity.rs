//! `max_unpool_ops::max_unpool{1,2,3}d`（イシュー #2644）の `Tape`／`Var` を経由する
//! end-to-end 統合テスト。
//!
//! - 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/conv-transpose3d-max-unpool-pytorch-reference/
//!   conv_transpose3d_max_unpool_reference.json`・生成条件は同ディレクトリの `README.md`）と
//!   突合する。MaxUnpool は算術を含まないコピー（forward）と gather（勾配）なので **bit 一致**
//!   （NaN はクラス一致）で比較する。
//! - **重複索引の意味論（意図的な PyTorch 差分。`docs/autodiff-conv-transpose3d-max-unpool-decision.md`
//!   §5）**: forward は `ScatterReduce::Overwrite`（row-major 走査で最後の書き手が残る）。PyTorch 2.14.0
//!   の CPU カーネルは重複索引で書き込みを並列に競合させ、複数スレッド実行では勝者が実行ごとに変わる
//!   （実測。fixture は `torch.set_num_threads(1)` で生成しており、その場合は最後の書き手が残る）。
//!   重複位置の forward は PyTorch ではなくテスト内の独立オラクル（走査順シミュレーション）と手計算で
//!   固定する。勾配は本実装が forward の真の随伴（最後の書き手のみ上流を受ける）で、PyTorch は全書き手へ
//!   配る。勝者位置は PyTorch と一致し、敗者位置だけ 0（差分）。索引が重複しない通常ケースは全位置で
//!   PyTorch と一致する。
//! - 独立オラクル: last-writer 走査シミュレーション・手計算・中心差分（forward が入力の線形コピー
//!   なので厳密）・`max_pool` とのラウンドトリップ。
//! - `common::naive_ops()` は `scatter`／`gather` を override しないため、必ずホストフォールバック経路を
//!   通る。CPU `BackendOps` 実装との一致は `crates/facade/tests/
//!   conv_transpose3d_max_unpool_backend_parity.rs` が担当する。

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::max_unpool_ops::{max_unpool1d, max_unpool2d, max_unpool3d};
use fandhe_ai_autodiff::pool3d_ops::max_pool3d;
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, ShapeError, Tensor};
use serde::Deserialize;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor.host_slice().into_owned()
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn wave(n: usize, freq: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * freq).sin() * amp).collect()
}

/// NaN は NaN 同士（クラス一致）、それ以外は bit 完全一致。
fn assert_class_or_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{context}[{i}]: NaN のはず（actual={a}）");
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{context}[{i}]: actual={a} expected={e}"
            );
        }
    }
}

// --- fixture ---

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    max_unpool_cases: Vec<Case>,
    max_unpool_dup_cases: Vec<Case>,
    max_unpool_nonfinite_cases: Vec<Case>,
    max_unpool_error_cases: Vec<ErrCase>,
}

/// JSON の `kernel`／`stride`／`padding` はスカラーまたは配列。
#[derive(Deserialize)]
#[serde(untagged)]
enum Num {
    One(usize),
    Many(Vec<usize>),
}

impl Num {
    fn expand(&self, dim: usize) -> Vec<usize> {
        match self {
            Num::One(v) => vec![*v; dim],
            Num::Many(v) => v.clone(),
        }
    }
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dim: usize,
    kernel: Num,
    stride: Num,
    padding: Num,
    output_size: Option<Vec<usize>>,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    x_bits: Vec<u32>,
    index: Vec<i32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct ErrCase {
    name: String,
    torch_raises: bool,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/conv-transpose3d-max-unpool-pytorch-reference/conv_transpose3d_max_unpool_reference.json",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

/// 空間 rank（`dim`）ごとの入口を 1 つに束ねる。
fn unpool<'t>(
    dim: usize,
    x: &Var<'t>,
    idx: &Tensor<i32>,
    k: &[usize],
    s: Option<&[usize]>,
    p: &[usize],
    os: Option<&[usize]>,
) -> Result<Var<'t>, AutodiffError> {
    match dim {
        1 => max_unpool1d(x, idx, k[0], s.map(|s| s[0]), p[0], os.map(|o| o[0])),
        2 => max_unpool2d(
            x,
            idx,
            [k[0], k[1]],
            s.map(|s| [s[0], s[1]]),
            [p[0], p[1]],
            os.map(|o| [o[0], o[1]]),
        ),
        3 => max_unpool3d(
            x,
            idx,
            [k[0], k[1], k[2]],
            s.map(|s| [s[0], s[1], s[2]]),
            [p[0], p[1], p[2]],
            os.map(|o| [o[0], o[1], o[2]]),
        ),
        _ => unreachable!(),
    }
}

struct Run {
    out: Vec<f32>,
    out_shape: Vec<usize>,
    grad: Vec<f32>,
}

/// `(out * g).sum()` を損失とした forward 値と入力勾配を返す。
fn run_case(ops: Box<dyn BackendOps + Send>, c: &Case) -> Run {
    let tape = Tape::new_with_ops(ops);
    let xv = tape.var(&t(from_bits(&c.x_bits), &c.in_shape));
    let idx = ti(c.index.clone(), &c.in_shape);
    let y = unpool(
        c.dim,
        &xv,
        &idx,
        &c.kernel.expand(c.dim),
        Some(&c.stride.expand(c.dim)),
        &c.padding.expand(c.dim),
        c.output_size.as_deref(),
    )
    .unwrap_or_else(|e| panic!("{}: max_unpool 失敗: {e:?}", c.name));
    let out = y.to_tensor();
    let gv = tape.var_no_grad(&t(from_bits(&c.g_bits), out.shape()));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    Run {
        out: dense(&out),
        out_shape: out.shape().to_vec(),
        grad: dense(grads.get(&xv).unwrap().expect("入力へ勾配が届く")),
    }
}

/// 独立オラクル: `(n, c)` 平面ごとに row-major 走査し最後の書き手を求める。
/// 戻り値は（forward の期待値・各入力要素が勝者か）。
fn last_writer_oracle(c: &Case) -> (Vec<f32>, Vec<bool>) {
    let nc = c.in_shape[0] * c.in_shape[1];
    let in_plane = numel(&c.in_shape[2..]);
    let out_plane = numel(&c.out_shape[2..]);
    let x = from_bits(&c.x_bits);
    let mut out = vec![0.0f32; nc * out_plane];
    let mut owner = vec![usize::MAX; nc * out_plane];
    for (i, &xi) in x.iter().enumerate().take(nc * in_plane) {
        let pos = (i / in_plane) * out_plane + c.index[i] as usize;
        out[pos] = xi;
        owner[pos] = i;
    }
    let winner: Vec<bool> = (0..nc * in_plane)
        .map(|i| owner[(i / in_plane) * out_plane + c.index[i] as usize] == i)
        .collect();
    (out, winner)
}

/// 出力位置が複数回書かれる（重複索引の）入力要素か。
fn written_more_than_once(c: &Case) -> Vec<bool> {
    let in_plane = numel(&c.in_shape[2..]);
    let out_plane = numel(&c.out_shape[2..]);
    let nc = c.in_shape[0] * c.in_shape[1];
    let mut count = vec![0usize; nc * out_plane];
    for i in 0..nc * in_plane {
        count[(i / in_plane) * out_plane + c.index[i] as usize] += 1;
    }
    (0..nc * in_plane)
        .map(|i| count[(i / in_plane) * out_plane + c.index[i] as usize] > 1)
        .collect()
}

// --- 1. PyTorch fixture 突合 ---

#[test]
fn unique_index_cases_match_pytorch_bit_for_bit() {
    let fixture = load_fixture();
    assert!(
        fixture.torch_version.starts_with("2.14.0"),
        "fixture は PyTorch 2.14.0 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    // プール由来で索引が重複しないケース（stride >= kernel）は全位置で forward・勾配とも bit 一致。
    const UNIQUE: [&str; 8] = [
        "1d_k2_s2",
        "1d_odd_len_output_size",
        "1d_stride_gt_kernel",
        "2d_k2_s2",
        "2d_odd_output_size",
        "3d_k2_s2",
        "3d_aniso",
        "3d_odd_output_size",
    ];
    let mut seen = 0;
    for c in &fixture.max_unpool_cases {
        if !UNIQUE.contains(&c.name.as_str()) {
            continue;
        }
        seen += 1;
        assert!(
            written_more_than_once(c).iter().all(|d| !d),
            "{}: 重複索引を含まないはず",
            c.name
        );
        let r = run_case(common::naive_ops(), c);
        assert_eq!(r.out_shape, c.out_shape, "{}: 出力 shape", c.name);
        assert_class_or_bits_eq(&r.out, &from_bits(&c.out_bits), &format!("{} fwd", c.name));
        assert_class_or_bits_eq(
            &r.grad,
            &from_bits(&c.grad_bits),
            &format!("{} grad", c.name),
        );
    }
    assert_eq!(seen, UNIQUE.len());
}

#[test]
fn overlapping_window_cases_match_pytorch_forward_and_winner_gradients() {
    // 重なり窓（stride < kernel）のプール由来索引: 重複書き込みは同一の勝者要素由来で値が同じため
    // forward は全位置で PyTorch と bit 一致。勾配は勝者位置が PyTorch と一致し、敗者位置は 0
    // （PyTorch は全書き手へ上流を配る。意図的な差分）。
    let fixture = load_fixture();
    let cases: Vec<&Case> = fixture
        .max_unpool_cases
        .iter()
        .chain(&fixture.max_unpool_dup_cases)
        .filter(|c| !c.name.contains("manual"))
        .filter(|c| written_more_than_once(c).iter().any(|d| *d))
        .collect();
    assert!(
        cases.len() >= 7,
        "重複索引を含むプール由来ケース: {}",
        cases.len()
    );
    for c in cases {
        let r = run_case(common::naive_ops(), c);
        assert_class_or_bits_eq(&r.out, &from_bits(&c.out_bits), &format!("{} fwd", c.name));
        let (oracle, winner) = last_writer_oracle(c);
        assert_class_or_bits_eq(&r.out, &oracle, &format!("{} oracle", c.name));
        let torch_grad = from_bits(&c.grad_bits);
        let mut losers = 0;
        for (i, &is_winner) in winner.iter().enumerate() {
            if is_winner {
                assert_eq!(
                    r.grad[i].to_bits(),
                    torch_grad[i].to_bits(),
                    "{} win[{i}]",
                    c.name
                );
            } else {
                losers += 1;
                assert_eq!(r.grad[i], 0.0, "{} loser[{i}]", c.name);
                assert_ne!(
                    torch_grad[i], 0.0,
                    "{} loser[{i}]: PyTorch は上流を配る",
                    c.name
                );
            }
        }
        assert!(losers > 0, "{}: 敗者位置が存在するはず", c.name);
    }
}

#[test]
fn manual_duplicate_indices_follow_last_writer_not_pytorch_race() {
    // PyTorch 2.14.0 の重複索引 forward は複数スレッド実行で勝者が実行ごとに変わる（実測）ため、重複位置の
    // forward は独立オラクルと手計算で固定し、PyTorch とは非重複位置だけ比較する。fixture は単一スレッド
    // 生成なので全位置が最後の書き手と一致することも併せて確認する（再生成時の取り違え検出）。
    let fixture = load_fixture();
    let by_name = |n: &str| {
        fixture
            .max_unpool_dup_cases
            .iter()
            .find(|c| c.name == n)
            .unwrap_or_else(|| panic!("{n} がない"))
    };
    for c in fixture
        .max_unpool_dup_cases
        .iter()
        .filter(|c| c.name.contains("manual"))
    {
        let r = run_case(common::naive_ops(), c);
        let (oracle, winner) = last_writer_oracle(c);
        assert_class_or_bits_eq(&r.out, &oracle, &format!("{} oracle", c.name));
        let dup_target = {
            let in_plane = numel(&c.in_shape[2..]);
            let out_plane = numel(&c.out_shape[2..]);
            let mut count = vec![0usize; numel(&c.out_shape)];
            for i in 0..numel(&c.in_shape) {
                count[(i / in_plane) * out_plane + c.index[i] as usize] += 1;
            }
            count
        };
        let torch_out = from_bits(&c.out_bits);
        assert_class_or_bits_eq(
            &oracle,
            &torch_out,
            &format!("{} 単一スレッド fixture", c.name),
        );
        for (pos, &n) in dup_target.iter().enumerate() {
            if n <= 1 {
                assert_eq!(
                    r.out[pos].to_bits(),
                    torch_out[pos].to_bits(),
                    "{} pos {pos}",
                    c.name
                );
            }
        }
        // 勝者位置の勾配は PyTorch と一致、敗者位置は 0。
        let torch_grad = from_bits(&c.grad_bits);
        for (i, &w) in winner.iter().enumerate() {
            if w {
                assert_eq!(
                    r.grad[i].to_bits(),
                    torch_grad[i].to_bits(),
                    "{} win[{i}]",
                    c.name
                );
            } else {
                assert_eq!(r.grad[i], 0.0, "{} loser[{i}]", c.name);
            }
        }
    }
    // 手計算: 1d x=[1,2,3,4]・index=[1,1,3,3] -> out[1]=2・out[3]=4（最後の書き手）。
    let c = by_name("1d_manual_dup");
    let r = run_case(common::naive_ops(), c);
    assert_eq!(r.out, vec![0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    // 2d x=[1,2,3,4]・index=[0,0,5,0] -> out[0]=4・out[5]=3。
    let c = by_name("2d_manual_dup");
    let r = run_case(common::naive_ops(), c);
    assert_eq!((r.out[0], r.out[5]), (4.0, 3.0));
    assert_eq!(r.out.iter().filter(|v| **v != 0.0).count(), 2);
}

#[test]
fn nonfinite_inputs_are_copied_bit_for_bit() {
    let fixture = load_fixture();
    assert!(fixture.max_unpool_nonfinite_cases.len() >= 3);
    for c in &fixture.max_unpool_nonfinite_cases {
        assert!(
            from_bits(&c.x_bits).iter().any(|v| !v.is_finite()),
            "{}: 非有限値を含むはず",
            c.name
        );
        let r = run_case(common::naive_ops(), c);
        assert_class_or_bits_eq(&r.out, &from_bits(&c.out_bits), &format!("{} fwd", c.name));
        assert_class_or_bits_eq(
            &r.grad,
            &from_bits(&c.grad_bits),
            &format!("{} grad", c.name),
        );
    }
}

/// 引数エラー 1 件の仕様。
struct ErrSpec {
    dim: usize,
    x_shape: Vec<usize>,
    index: Vec<i32>,
    k: usize,
    stride: Option<usize>,
    output_size: Option<usize>,
}

/// 戻り値: (仕様, PyTorch 2.14.0 が拒否するか, 本実装が受理するか)。
fn err_spec(name: &str) -> (ErrSpec, bool, bool) {
    let base = || ErrSpec {
        dim: 1,
        x_shape: vec![1, 1, 3],
        index: vec![0, 1, 2],
        k: 2,
        stride: None,
        output_size: None,
    };
    match name {
        "index_out_of_range" => (
            ErrSpec {
                index: vec![0, 1, 6],
                ..base()
            },
            true,
            false,
        ),
        "index_negative" => (
            ErrSpec {
                index: vec![0, 1, -1],
                ..base()
            },
            true,
            false,
        ),
        "index_shape_mismatch" => (
            ErrSpec {
                index: vec![0, 1],
                ..base()
            },
            true,
            false,
        ),
        "output_size_eq_default_minus_stride" => (
            ErrSpec {
                output_size: Some(4),
                ..base()
            },
            true,
            false,
        ),
        "output_size_default_minus_stride_plus1" => (
            ErrSpec {
                output_size: Some(5),
                ..base()
            },
            false,
            true,
        ),
        "output_size_default_plus_stride_minus1" => (
            ErrSpec {
                output_size: Some(7),
                ..base()
            },
            false,
            true,
        ),
        "output_size_eq_default_plus_stride" => (
            ErrSpec {
                output_size: Some(8),
                ..base()
            },
            true,
            false,
        ),
        "output_size_far_below" => (
            ErrSpec {
                output_size: Some(3),
                ..base()
            },
            true,
            false,
        ),
        // PyTorch はバッチなし入力（rank 2）を受理する。本実装は rank 3 以上のみ（意図的な差分）。
        "batchless_input" => (
            ErrSpec {
                x_shape: vec![1, 3],
                ..base()
            },
            false,
            false,
        ),
        // PyTorch は kernel=0 を受理して空出力を返す。本実装は拒否する（意図的な差分）。
        "kernel_zero" => (ErrSpec { k: 0, ..base() }, false, false),
        "stride_zero" => (
            ErrSpec {
                stride: Some(0),
                ..base()
            },
            true,
            false,
        ),
        "batch_zero" => (
            ErrSpec {
                x_shape: vec![0, 1, 3],
                index: vec![],
                ..base()
            },
            false,
            true,
        ),
        // PyTorch は C=0 を拒否する。本実装は受理して空出力を返す（意図的な差分）。
        "channel_zero" => (
            ErrSpec {
                x_shape: vec![1, 0, 3],
                index: vec![],
                ..base()
            },
            true,
            true,
        ),
        // 2d 入口へ rank 5 入力。
        "rank_mismatch_2d_with_3d_input" => (
            ErrSpec {
                dim: 2,
                x_shape: vec![1, 1, 2, 2, 2],
                index: vec![0; 8],
                ..base()
            },
            true,
            false,
        ),
        other => panic!("未知の error case: {other}"),
    }
}

#[test]
fn fixture_error_cases_agree_with_torch_except_documented_differences() {
    let fixture = load_fixture();
    assert!(fixture.max_unpool_error_cases.len() >= 14);
    for c in &fixture.max_unpool_error_cases {
        let (spec, torch_raises, ours_ok) = err_spec(&c.name);
        assert_eq!(
            c.torch_raises, torch_raises,
            "{}: fixture の torch_raises が期待表と食い違う（再生成後は表を更新する）",
            c.name
        );
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![1.0; numel(&spec.x_shape)], &spec.x_shape));
        // index は shape 不一致ケースのため要素数ベースで作る（shape は x と同じにできない場合は別 shape）。
        let idx_shape: Vec<usize> = if spec.index.len() == numel(&spec.x_shape) {
            spec.x_shape.clone()
        } else {
            let mut s = spec.x_shape.clone();
            *s.last_mut().unwrap() = spec.index.len();
            s
        };
        let idx = ti(spec.index.clone(), &idx_shape);
        let k = vec![spec.k; spec.dim];
        let s = spec.stride.map(|s| vec![s; spec.dim]);
        let os = spec.output_size.map(|o| vec![o; spec.dim]);
        let r = unpool(
            spec.dim,
            &x,
            &idx,
            &k,
            s.as_deref(),
            &vec![0; spec.dim],
            os.as_deref(),
        );
        assert_eq!(r.is_ok(), ours_ok, "{}: 本実装の受理可否", c.name);
    }
}

// --- 2. 手計算・中心差分・ラウンドトリップ ---

#[test]
fn stride_none_defaults_to_kernel_and_padding_shrinks_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 1, 3]));
    let idx = ti(vec![0, 1, 2], &[1, 1, 3]);
    let y = max_unpool1d(&x, &idx, 2, None, 0, None).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 1, 6]);
    let y = max_unpool1d(&x, &idx, 2, None, 1, None).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 1, 4]);
    let y = max_unpool1d(&x, &idx, 2, Some(3), 0, Some(8)).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 1, 8]);
}

#[test]
fn gradient_matches_central_difference_with_duplicates() {
    // forward は入力の線形コピーなので中心差分は厳密に勾配と一致する（重複索引を含めて固定）。
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let index = vec![1, 1, 3, 3];
    let g: Vec<f32> = (0..8).map(|i| 0.5 + i as f32).collect();
    let forward = |xs: &[f32]| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(xs.to_vec(), &[1, 1, 4]));
        let y = max_unpool1d(&x, &ti(index.clone(), &[1, 1, 4]), 2, None, 0, None).unwrap();
        dense(&y.to_tensor())
            .iter()
            .zip(&g)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum()
    };
    let h = 1e-2f32;
    let numeric: Vec<f64> = (0..4)
        .map(|i| {
            let mut p = x_data.clone();
            p[i] += h;
            let mut m = x_data.clone();
            m[i] -= h;
            (forward(&p) - forward(&m)) / (2.0 * f64::from(h))
        })
        .collect();

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x_data.clone(), &[1, 1, 4]));
    let y = max_unpool1d(&x, &ti(index.clone(), &[1, 1, 4]), 2, None, 0, None).unwrap();
    let gv = tape.var_no_grad(&t(g.clone(), &[1, 1, 8]));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x).unwrap().unwrap());
    // 手計算: 勝者は入力 1（→ out[1]）と入力 3（→ out[3]）。敗者 0・2 は 0。
    assert_eq!(analytic, vec![0.0, g[1], 0.0, g[3]]);
    for (a, n) in analytic.iter().zip(&numeric) {
        assert!((f64::from(*a) - n).abs() < 1e-3, "analytic={a} numeric={n}");
    }
}

#[test]
fn roundtrip_with_max_pool_restores_maxima_and_zeros_elsewhere() {
    // 1d。
    let tape = Tape::new_with_ops(common::naive_ops());
    let data = wave(2 * 3 * 8, 0.37, 1.0);
    let x = tape.var(&t(data.clone(), &[2, 3, 8]));
    let (v, idx) = x.max_pool1d(2, Some(2), 0, 1, false).unwrap();
    let u = max_unpool1d(&v, &idx, 2, Some(2), 0, None).unwrap();
    assert_eq!(u.to_tensor().shape(), &[2, 3, 8]);
    assert_roundtrip(&data, &dense(&u.to_tensor()), &idx, 8);

    // 2d。
    let data = wave(2 * 4 * 6, 0.29, 1.0);
    let x = tape.var(&t(data.clone(), &[1, 2, 4, 6]));
    let (v, idx) = x
        .max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)
        .unwrap();
    let u = max_unpool2d(&v, &idx, [2, 2], Some([2, 2]), [0, 0], None).unwrap();
    assert_eq!(u.to_tensor().shape(), &[1, 2, 4, 6]);
    assert_roundtrip(&data, &dense(&u.to_tensor()), &idx, 24);

    // 3d。
    let data = wave(2 * 4 * 4 * 6, 0.19, 1.0);
    let x = tape.var(&t(data.clone(), &[1, 2, 4, 4, 6]));
    let (v, idx) = max_pool3d(&x, [2, 2, 2], None, [0; 3], [1; 3], false).unwrap();
    let u = max_unpool3d(&v, &idx, [2, 2, 2], None, [0; 3], None).unwrap();
    assert_eq!(u.to_tensor().shape(), &[1, 2, 4, 4, 6]);
    assert_roundtrip(&data, &dense(&u.to_tensor()), &idx, 96);
}

/// 最大値位置には元の値、それ以外は 0 であること。
fn assert_roundtrip(original: &[f32], unpooled: &[f32], idx: &Tensor<i32>, out_plane: usize) {
    assert_eq!(original.len(), unpooled.len());
    let index = idx.contiguous().host_slice().into_owned();
    let per_plane = index.len() / (original.len() / out_plane);
    let mut hit = vec![false; original.len()];
    for (i, &pos) in index.iter().enumerate() {
        hit[(i / per_plane) * out_plane + pos as usize] = true;
    }
    for (i, (&o, &u)) in original.iter().zip(unpooled).enumerate() {
        if hit[i] {
            assert_eq!(u.to_bits(), o.to_bits(), "最大値位置 {i}");
        } else {
            assert_eq!(u, 0.0, "非最大値位置 {i}");
        }
    }
}

#[test]
fn roundtrip_gradient_flows_to_pooled_values_only() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let data = wave(8, 0.9, 1.0);
    let x = tape.var(&t(data, &[1, 1, 8]));
    let (v, idx) = x.max_pool1d(2, Some(2), 0, 1, false).unwrap();
    let u = max_unpool1d(&v, &idx, 2, Some(2), 0, None).unwrap();
    let g: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let gv = tape.var_no_grad(&t(g.clone(), &[1, 1, 8]));
    let loss = u.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = dense(grads.get(&x).unwrap().unwrap());
    let index = idx.contiguous().host_slice().into_owned();
    let mut want = vec![0.0f32; 8];
    for &p in &index {
        want[p as usize] = g[p as usize];
    }
    assert_eq!(dx, want);
}

#[test]
fn non_contiguous_input_and_index_are_read_in_logical_order() {
    let base = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 3, 2]);
    let base_idx = ti(vec![0, 1, 2, 3, 4, 5], &[1, 1, 3, 2]);
    let x_view = base.transpose(2, 3).unwrap();
    let idx_view = base_idx.transpose(2, 3).unwrap();
    assert!(!x_view.is_contiguous());
    let x_copy = x_view.contiguous();
    let idx_copy = idx_view.contiguous();
    let run = |x: &Tensor<f32>, idx: &Tensor<i32>| {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        max_unpool2d(&xv, idx, [2, 2], None, [0, 0], None).map(|y| dense(&y.to_tensor()))
    };
    // 既定出力 4×6=24 要素の平面内に索引（0..5）が収まる。
    let a = run(&x_view, &idx_view).unwrap();
    let b = run(&x_copy, &idx_copy).unwrap();
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // 論理順（転置後の [1,1,2,3]）: 値 [1,3,5,2,4,6] が索引 [0,2,4,1,3,5] へ置かれる。
    assert_eq!(&a[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// --- 3. 引数検査・境界・テープ ---

#[test]
fn invalid_arguments_are_typed_errors_and_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 1, 3]));
    let ok = ti(vec![0, 1, 2], &[1, 1, 3]);
    let before = tape.len();
    // 索引値の範囲外・負値は InvalidArgument。
    for bad in [
        vec![0, 1, 6],
        vec![0, 1, -1],
        vec![i32::MAX, 0, 1],
        vec![i32::MIN, 0, 1],
    ] {
        assert!(matches!(
            max_unpool1d(&x, &ti(bad, &[1, 1, 3]), 2, None, 0, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
    // 形状・サイズの違反は Shape。
    assert!(matches!(
        max_unpool1d(&x, &ti(vec![0, 1], &[1, 1, 2]), 2, None, 0, None),
        Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
    ));
    assert!(matches!(
        max_unpool2d(&x, &ok, [2, 2], None, [0, 0], None),
        Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
    ));
    for r in [
        max_unpool1d(&x, &ok, 0, None, 0, None),
        max_unpool1d(&x, &ok, 2, Some(0), 0, None),
        max_unpool1d(&x, &ok, 2, None, 3, None),
        max_unpool1d(&x, &ok, 2, None, 0, Some(8)),
        max_unpool1d(&x, &ok, 2, None, 0, Some(4)),
    ] {
        assert!(matches!(r, Err(AutodiffError::Shape(_))));
    }
    let z = tape.var(&t(vec![], &[1, 1, 0]));
    assert!(matches!(
        max_unpool1d(&z, &ti(vec![], &[1, 1, 0]), 2, None, 0, None),
        Err(AutodiffError::Shape(_))
    ));
    assert_eq!(
        tape.len(),
        before + 1,
        "増えたのは z の入力ノード 1 件のみ（孤児ノード無し）"
    );
}

#[test]
fn empty_batch_and_channel_are_accepted() {
    let tape = Tape::new_with_ops(common::naive_ops());
    for shape in [[0usize, 2, 3], [2, 0, 3]] {
        let x = tape.var(&t(vec![], &shape));
        let y = max_unpool1d(&x, &ti(vec![], &shape), 2, None, 0, None).unwrap();
        assert_eq!(y.to_tensor().shape(), &[shape[0], shape[1], 6]);
        let grads = tape.backward(&y.sum(None).unwrap()).unwrap();
        assert_eq!(grads.get(&x).unwrap().unwrap().shape(), &shape);
    }
}

#[test]
fn huge_output_is_rejected_before_allocation() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[1, 1, 2]));
    let idx = ti(vec![0, 1], &[1, 1, 2]);
    // 出力長 (2-1)*stride + 2 は usize には収まるが確保バイト数が isize::MAX を超える。
    for stride in [usize::MAX / 2, usize::MAX / 4] {
        assert!(matches!(
            max_unpool1d(&x, &idx, 2, Some(stride), 0, None),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }
    assert!(matches!(
        max_unpool1d(&x, &idx, 2, Some(usize::MAX), 0, None),
        Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
    ));
}

#[test]
fn records_exactly_one_node_and_rejects_create_graph() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 1, 3]));
    let idx = ti(vec![0, 1, 2], &[1, 1, 3]);
    let before = tape.len();
    let y = max_unpool1d(&x, &idx, 2, None, 0, None).unwrap();
    assert_eq!(tape.len(), before + 1);
    let loss = y.sum(None).unwrap();
    assert!(tape.backward_create_graph(&loss, &child).is_err());
}

#[test]
fn results_are_bit_deterministic_run_to_run() {
    let fixture = load_fixture();
    for c in fixture
        .max_unpool_cases
        .iter()
        .chain(&fixture.max_unpool_dup_cases)
    {
        let a = run_case(common::naive_ops(), c);
        let b = run_case(common::naive_ops(), c);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.out), bits(&b.out), "{}", c.name);
        assert_eq!(bits(&a.grad), bits(&b.grad), "{}", c.name);
    }
}

// --- 4. フォールバックとエラー伝播（scatter＝forward・gather＝VJP） ---

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    LaunchFailed,
    WrongShape,
}

/// `scatter`／`gather` だけを差し替える `BackendOps`。それ以外は naive へ委譲する。
struct IndexMock {
    inner: Box<dyn BackendOps + Send>,
    scatter: (Mode, Arc<AtomicUsize>),
    gather: (Mode, Arc<AtomicUsize>),
}

impl BackendOps for IndexMock {
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
    fn scatter(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        self.scatter.1.fetch_add(1, Ordering::SeqCst);
        match self.scatter.0 {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
    fn gather(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.gather.1.fetch_add(1, Ordering::SeqCst);
        match self.gather.0 {
            Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
            Mode::LaunchFailed => Err(BackendError::KernelLaunchFailed("simulated".into())),
            Mode::WrongShape => Ok(t(vec![0.0; 3], &[3])),
        }
    }
}

fn mock_tape(scatter: Mode, gather: Mode) -> (Tape, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let s = Arc::new(AtomicUsize::new(0));
    let g = Arc::new(AtomicUsize::new(0));
    let tape = Tape::new_with_ops(Box::new(IndexMock {
        inner: common::naive_ops(),
        scatter: (scatter, Arc::clone(&s)),
        gather: (gather, Arc::clone(&g)),
    }));
    (tape, s, g)
}

fn sample() -> (Tensor<f32>, Tensor<i32>) {
    (
        t(vec![1.0, 2.0, 3.0], &[1, 1, 3]),
        ti(vec![1, 1, 5], &[1, 1, 3]),
    )
}

#[test]
fn unsupported_hooks_fall_back_to_host_and_match_naive() {
    let (x, idx) = sample();
    let (tape, s, g) = mock_tape(Mode::Unsupported, Mode::Unsupported);
    let xv = tape.var(&x);
    let y = max_unpool1d(&xv, &idx, 2, None, 0, None).unwrap();
    assert_eq!(s.load(Ordering::SeqCst), 1, "forward は scatter を先に呼ぶ");
    assert_eq!(g.load(Ordering::SeqCst), 0);
    assert_eq!(dense(&y.to_tensor()), vec![0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let gvec: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let gv = tape.var_no_grad(&t(gvec, &[1, 1, 6]));
    let loss = y.mul(&gv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    assert_eq!(g.load(Ordering::SeqCst), 1, "VJP は gather を先に呼ぶ");
    // 勝者は入力 1（→ out[1]、上流 2）と入力 2（→ out[5]、上流 6）。入力 0 は上書きされ 0。
    assert_eq!(dense(grads.get(&xv).unwrap().unwrap()), vec![0.0, 2.0, 6.0]);
}

#[test]
fn non_unsupported_errors_are_propagated_in_forward_and_backward() {
    let (x, idx) = sample();
    let (tape, _, _) = mock_tape(Mode::LaunchFailed, Mode::Unsupported);
    assert!(matches!(
        max_unpool1d(&tape.var(&x), &idx, 2, None, 0, None),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::LaunchFailed);
    let y = max_unpool1d(&tape.var(&x), &idx, 2, None, 0, None).unwrap();
    let loss = y.sum(None).unwrap();
    assert!(matches!(
        tape.backward(&loss),
        Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
    ));
}

#[test]
fn wrong_shape_from_backend_is_a_typed_error() {
    let (x, idx) = sample();
    let (tape, _, _) = mock_tape(Mode::WrongShape, Mode::Unsupported);
    assert!(matches!(
        max_unpool1d(&tape.var(&x), &idx, 2, None, 0, None),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
    let (tape, _, _) = mock_tape(Mode::Unsupported, Mode::WrongShape);
    let y = max_unpool1d(&tape.var(&x), &idx, 2, None, 0, None).unwrap();
    let loss = y.sum(None).unwrap();
    assert!(matches!(
        tape.backward(&loss),
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
    ));
}
