//! `nn::MultiheadAttention`（イシュー #1640・親 #1605 sub-issue (b)）の
//! 受け入れ条件対応テスト。
//!
//! `tests/nn_rnn.rs` と同じ構成: ①ブルートフォース参照実装（`f64`
//! アキュムレータ）との forward 一致 → ②多入力（重み・bias・q/k/v 入力）
//! の数値微分突合（mask なし／causal／`attn_mask` あり） → ③`Module`
//! trait（self-attention）との一致 → ④エラー経路（panic せず型付き
//! エラー） → ⑤非 contiguous 入力・決定性。`common::naive_ops()`
//! （`BackendOps` の softmax／gemm_batched 等はいずれも既定合成
//! フォールバック経由）のみを使い、`backend-cpu` 等の具体バックエンドへ
//! 依存しない（`common/mod.rs` 冒頭コメントの設計上の不変条件）。

mod common;

use fandhe_ai_autodiff::nn::{Linear, Module, MultiheadAttention};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

// `tests/nn_rnn.rs` と同じ許容誤差・数値微分ステップ幅（新しい許容
// 誤差は導入しない）。
const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < shape[axis] {
            return;
        }
        idx[axis] = 0;
    }
}

/// REQ-2 統一複合判定（`common::req2_close` に委譲。`einsum.rs::
/// assert_req2_close` と同型）。forward の突合に使う。
fn assert_req2_close(label: &str, actual: &Tensor<f32>, expected: &Tensor<f32>) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape が一致しない"
    );
    let shape = actual.shape().to_vec();
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return;
    }
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel {
        let av = actual.get(&idx).unwrap_or(0.0);
        let ev = expected.get(&idx).unwrap_or(0.0);
        assert!(
            common::req2_close(av as f64, ev as f64),
            "{label}[{idx:?}]: actual={av} expected={ev}"
        );
        increment_index(&mut idx, &shape);
    }
}

/// 数値微分突合用の許容誤差判定（`tests/nn_rnn.rs::assert_grad_close`
/// と同一実装）。
fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    if numel == 0 {
        return;
    }
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..numel {
        let av = analytic.get(&idx).unwrap_or(0.0);
        let nv = numeric.get(&idx).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{idx:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
        increment_index(&mut idx, &shape);
    }
}

/// 指定テンソルの各要素を中央差分で摂動し、`forward_loss` に対する
/// 数値勾配を計算する（`tests/nn_rnn.rs::numeric_grad` と同一実装）。
fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// =====================================================================
// フィクスチャ（B=2, L=3, S=4, E=4, H=2, Dh=2）。決定的な固定値。
// =====================================================================

const B: usize = 2;
const L: usize = 3;
const S: usize = 4;
const E: usize = 4;
const NUM_HEADS: usize = 2;

fn seq(seed: i64, len: usize) -> Vec<f32> {
    // 振幅を小さめ（±0.16 程度）に抑える。softmax を挟む attention は
    // 複数 key 間の相互作用でロジットの曲率が RNN（`tests/nn_rnn.rs`）
    // より大きくなりやすく、振幅が大きいと中央差分（`H=1e-3`）の高次
    // truncation 誤差が `REL_TOL`/`ABS_TOL` の限界に接近するため
    // （数値微分突合テストで実測確認済み）。forward 突合（ブルート
    // フォース参照実装との REQ-2 判定）はスケールに依存せず常に厳密な
    // ため、振幅を下げても forward 側のテスト網羅性は損なわれない。
    (0..len)
        .map(|i| ((seed + i as i64) % 17 - 8) as f32 * 0.02)
        .collect()
}

fn query_fixture() -> Tensor<f32> {
    t(seq(1, B * L * E), &[B, L, E])
}

fn key_fixture() -> Tensor<f32> {
    t(seq(11, B * S * E), &[B, S, E])
}

fn value_fixture() -> Tensor<f32> {
    t(seq(21, B * S * E), &[B, S, E])
}

// --- ブルートフォース参照実装（`f64` アキュムレータ）------------------

/// `y = x @ w (+ b)`（`x`: `[B, Len, E]`・`w`: `[E, E]`・`b`: `[E]`）を
/// `f64` で計算する（`nn::Linear`/`LinearVars::forward` と同じ
/// `matmul → add` 合成の素朴なループ版）。
fn project_host(
    x: &[f64],
    b: usize,
    len: usize,
    e: usize,
    w: &[f64],
    bias: Option<&[f64]>,
) -> Vec<f64> {
    let mut out = vec![0.0f64; b * len * e];
    for bi in 0..b {
        for li in 0..len {
            for oj in 0..e {
                let mut acc = 0.0f64;
                for k in 0..e {
                    acc += x[(bi * len + li) * e + k] * w[k * e + oj];
                }
                if let Some(bs) = bias {
                    acc += bs[oj];
                }
                out[(bi * len + li) * e + oj] = acc;
            }
        }
    }
    out
}

fn to_f64(t: &Tensor<f32>) -> Vec<f64> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// `MultiheadAttention` 全体（q/k/v projection → head 分割 → scaled dot
/// product attention（scale・mask／causal・softmax） → head 結合 → out
/// projection）を素朴な多重ループで `f64` 計算する参照実装。
/// `mha.q_proj().weight()` 等（本イシューで追加した public アクセサ）
/// から読み出した重みをそのまま使うため、`MultiheadAttentionVars::
/// forward` と入力を完全に揃えて突合できる。
#[allow(clippy::too_many_arguments)]
fn brute_force_mha(
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    mha: &MultiheadAttention,
    attn_mask: Option<&Tensor<bool>>,
    is_causal: bool,
) -> Tensor<f32> {
    let b = query.shape()[0];
    let l = query.shape()[1];
    let s = key.shape()[1];
    let e = mha.embed_dim();
    let h = mha.num_heads();
    let dh = e / h;

    let qw = to_f64(mha.q_proj().weight());
    let kw = to_f64(mha.k_proj().weight());
    let vw = to_f64(mha.v_proj().weight());
    let ow = to_f64(mha.out_proj().weight());
    let qb = mha.q_proj().bias().map(to_f64);
    let kb = mha.k_proj().bias().map(to_f64);
    let vb = mha.v_proj().bias().map(to_f64);
    let ob = mha.out_proj().bias().map(to_f64);

    let q_proj = project_host(&to_f64(query), b, l, e, &qw, qb.as_deref());
    let k_proj = project_host(&to_f64(key), b, s, e, &kw, kb.as_deref());
    let v_proj = project_host(&to_f64(value), b, s, e, &vw, vb.as_deref());

    // head 分割の添字ヘルパー（[B, Len, E] の f64 flat バッファから
    // [B, H, Len, Dh] の (bi, hi, li, di) を直接読む。`reshape` →
    // `permute` の合成を再計算するのではなく、添字計算のみで再現する）。
    let head_at = |buf: &[f64], bi: usize, hi: usize, li: usize, di: usize, len: usize| -> f64 {
        buf[(bi * len + li) * e + hi * dh + di]
    };

    let scale = 1.0f64 / (dh as f64).sqrt();
    let mut merged = vec![0.0f64; b * l * e];
    for bi in 0..b {
        for hi in 0..h {
            let mut scores = vec![0.0f64; l * s];
            for li in 0..l {
                for si in 0..s {
                    let mut acc = 0.0f64;
                    for di in 0..dh {
                        acc += head_at(&q_proj, bi, hi, li, di, l)
                            * scale
                            * head_at(&k_proj, bi, hi, si, di, s);
                    }
                    let blocked = if is_causal {
                        si > li
                    } else if let Some(mask) = attn_mask {
                        let mshape = mask.shape().to_vec();
                        let idx = broadcast_mask_index(&mshape, &[b, h, l, s], bi, hi, li, si);
                        !mask
                            .get(&idx)
                            .expect("attn_mask: broadcast 済み添字は範囲内のはず")
                    } else {
                        false
                    };
                    scores[li * s + si] = if blocked { f64::NEG_INFINITY } else { acc };
                }
            }
            for li in 0..l {
                let row = &scores[li * s..li * s + s];
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let weights: Vec<f64> = exps.iter().map(|&v| v / sum).collect();
                for di in 0..dh {
                    let mut acc = 0.0f64;
                    for (si, &w) in weights.iter().enumerate() {
                        acc += w * head_at(&v_proj, bi, hi, si, di, s);
                    }
                    merged[(bi * l + li) * e + hi * dh + di] = acc;
                }
            }
        }
    }

    let out = project_host(&merged, b, l, e, &ow, ob.as_deref());
    let data: Vec<f32> = out.iter().map(|&v| v as f32).collect();
    t(data, &[b, l, e])
}

/// `mask`（形状 `mshape`）を `[b, h, l, s]` へ NumPy 互換ブロードキャスト
/// したときの `(bi, hi, li, si)` に対応する `mask` 自身の添字を求める
/// （末尾軸から揃え、`mshape` 側の軸長が 1 なら 0 に固定する）。
fn broadcast_mask_index(
    mshape: &[usize],
    out_shape: &[usize; 4],
    bi: usize,
    hi: usize,
    li: usize,
    si: usize,
) -> Vec<usize> {
    let out_idx = [bi, hi, li, si];
    let rank = mshape.len();
    let offset = out_shape.len() - rank;
    (0..rank)
        .map(|axis| {
            let dim = mshape[axis];
            if dim == 1 { 0 } else { out_idx[offset + axis] }
        })
        .collect()
}

fn setup(seed: u64) -> MultiheadAttention {
    MultiheadAttention::new(E, NUM_HEADS, true, seed)
        .expect("test fixture: E % NUM_HEADS == 0 のため成功するはず")
}

// --- (a) forward: ブルートフォース参照実装との一致 ---------------------

#[test]
fn forward_matches_brute_force_reference_no_mask() {
    let mha = setup(7);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    let out = vars
        .forward(&q, &k, &v, None, false)
        .expect("mask なしの forward は成功するはず");
    let expected = brute_force_mha(
        &query_fixture(),
        &key_fixture(),
        &value_fixture(),
        &mha,
        None,
        false,
    );

    assert_req2_close("forward(no mask)", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_reference_causal() {
    // causal は self-attention（L == S）で使うのが自然なため、L=S=3 の
    // 別フィクスチャを使う。
    let mha = setup(9);
    let x = t(seq(31, B * L * E), &[B, L, E]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let xv = tape.var(&x);

    let out = vars
        .forward(&xv, &xv, &xv, None, true)
        .expect("causal self-attention の forward は成功するはず");
    let expected = brute_force_mha(&x, &x, &x, &mha, None, true);

    assert_req2_close("forward(causal)", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_reference_with_attn_mask() {
    let mha = setup(13);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    // `[B, 1, L, S]`: batch ごとに異なるパターンで 1 key 列を block する
    // （key padding mask 相当）。`true` = attend。
    let mut mask_data = vec![true; B * L * S];
    for li in 0..L {
        mask_data[li * S + (S - 1)] = false; // batch 0: 最終列を block
        mask_data[(L + li) * S] = false; // batch 1: 先頭列を block
    }
    let mask = Tensor::new(mask_data, &[B, 1, L, S]).unwrap();

    let out = vars
        .forward(&q, &k, &v, Some(&mask), false)
        .expect("attn_mask ありの forward は成功するはず");
    let expected = brute_force_mha(
        &query_fixture(),
        &key_fixture(),
        &value_fixture(),
        &mha,
        Some(&mask),
        false,
    );

    assert_req2_close("forward(attn_mask)", &out.to_tensor(), &expected);
}

#[test]
fn forward_matches_brute_force_reference_single_head() {
    let mha = MultiheadAttention::new(E, 1, true, 17).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    let out = vars.forward(&q, &k, &v, None, false).unwrap();
    let expected = brute_force_mha(
        &query_fixture(),
        &key_fixture(),
        &value_fixture(),
        &mha,
        None,
        false,
    );

    assert_req2_close("forward(num_heads=1)", &out.to_tensor(), &expected);
}

// --- (b) mask 極性 -----------------------------------------------------

#[test]
fn all_true_mask_matches_no_mask() {
    let mha = setup(23);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    let all_true = Tensor::new(vec![true; L * S], &[L, S]).unwrap();
    let with_mask = vars
        .forward(&q, &k, &v, Some(&all_true), false)
        .unwrap()
        .to_tensor();
    let without_mask = vars.forward(&q, &k, &v, None, false).unwrap().to_tensor();

    assert_req2_close("all-true mask == no mask", &with_mask, &without_mask);
}

// --- (c) 数値微分突合 ----------------------------------------------------

/// mask なしケース: 8 パラメータ（q/k/v/out の weight・bias）＋
/// query/key/value 入力のすべてで解析勾配と数値勾配が一致することを
/// 確認する（数値微分はコスト（forward 再評価回数 O(2*numel)）が高い
/// ため、フィクスチャを小さく保った状態でこの網羅チェックを 1 ケースに
/// 限定し、causal／mask ありのケースは query のみへ絞る）。
#[test]
fn backward_numeric_grad_all_params_no_mask() {
    let seed = 29u64;
    let q_data = query_fixture();
    let k_data = key_fixture();
    let v_data = value_fixture();

    let forward_loss = |q: &Tensor<f32>, k: &Tensor<f32>, v: &Tensor<f32>| -> (f32, Tape) {
        let tape = Tape::new_with_ops(common::naive_ops());
        let mha = setup(seed);
        let vars = mha.bind(&tape);
        let qv = tape.var(q);
        let kv = tape.var(k);
        let vv = tape.var(v);
        let out = vars.forward(&qv, &kv, &vv, None, false).unwrap();
        let loss = out.sum(None).unwrap();
        (scalar(&loss.to_tensor()), tape)
    };

    let mha = setup(seed);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let qv = tape.var(&q_data);
    let kv = tape.var(&k_data);
    let vv = tape.var(&v_data);
    let out = vars.forward(&qv, &kv, &vv, None, false).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    // --- weight/bias 勾配 ---
    let params: [(&str, &Var, &Tensor<f32>); 8] = [
        ("q.weight", &vars.q.weight, mha.q_proj().weight()),
        ("k.weight", &vars.k.weight, mha.k_proj().weight()),
        ("v.weight", &vars.v.weight, mha.v_proj().weight()),
        ("out.weight", &vars.out.weight, mha.out_proj().weight()),
        (
            "q.bias",
            vars.q.bias.as_ref().unwrap(),
            mha.q_proj().bias().unwrap(),
        ),
        (
            "k.bias",
            vars.k.bias.as_ref().unwrap(),
            mha.k_proj().bias().unwrap(),
        ),
        (
            "v.bias",
            vars.v.bias.as_ref().unwrap(),
            mha.v_proj().bias().unwrap(),
        ),
        (
            "out.bias",
            vars.out.bias.as_ref().unwrap(),
            mha.out_proj().bias().unwrap(),
        ),
    ];
    for (label, var, base) in params {
        let analytic = grads
            .get(var)
            .unwrap()
            .unwrap_or_else(|| panic!("{label}: backward で到達するはず"))
            .clone();
        // このパラメータだけを摂動した `MultiheadAttention` を都度
        // 再構築するのはシードから重みを引き直す手間が大きいため、
        // `from_parameters` で他 3 層を固定したまま対象の weight/bias
        // のみ差し替える。`numeric_grad` の第 1 引数は摂動の基準点（元の
        // パラメータ値）でなければならない（勾配テンソル `analytic` を
        // 渡すと shape は偶然一致するが摂動の出発点が勾配値になってしまい、
        // 全く異なる点での数値微分になる。当初の実装バグ）。
        let numeric = numeric_grad(base, |perturbed| {
            let mha2 = rebuild_with_override(&mha, label, &perturbed);
            let tape2 = Tape::new_with_ops(common::naive_ops());
            let vars2 = mha2.bind(&tape2);
            let qv2 = tape2.var(&q_data);
            let kv2 = tape2.var(&k_data);
            let vv2 = tape2.var(&v_data);
            let out2 = vars2.forward(&qv2, &kv2, &vv2, None, false).unwrap();
            scalar(&out2.sum(None).unwrap().to_tensor())
        });
        assert_grad_close(label, &analytic, &numeric);
    }

    // --- 入力（query/key/value）勾配 ---
    for (label, var, base) in [
        ("query", &qv, &q_data),
        ("key", &kv, &k_data),
        ("value", &vv, &v_data),
    ] {
        let analytic = grads
            .get(var)
            .unwrap()
            .unwrap_or_else(|| panic!("{label}: backward で到達するはず"))
            .clone();
        let numeric = numeric_grad(base, |perturbed| match label {
            "query" => forward_loss(&perturbed, &k_data, &v_data).0,
            "key" => forward_loss(&q_data, &perturbed, &v_data).0,
            _ => forward_loss(&q_data, &k_data, &perturbed).0,
        });
        assert_grad_close(label, &analytic, &numeric);
    }
}

/// `mha` の 8 パラメータのうち `label` が指すものだけを `override_tensor`
/// で差し替えた新しい `MultiheadAttention` を `from_parameters` で
/// 再構築する（`backward_numeric_grad_all_params_no_mask` 専用ヘルパー）。
fn rebuild_with_override(
    mha: &MultiheadAttention,
    label: &str,
    override_tensor: &Tensor<f32>,
) -> MultiheadAttention {
    let mut linears = [
        Linear::from_parameters(mha.q_proj().weight().clone(), mha.q_proj().bias().cloned())
            .unwrap(),
        Linear::from_parameters(mha.k_proj().weight().clone(), mha.k_proj().bias().cloned())
            .unwrap(),
        Linear::from_parameters(mha.v_proj().weight().clone(), mha.v_proj().bias().cloned())
            .unwrap(),
        Linear::from_parameters(
            mha.out_proj().weight().clone(),
            mha.out_proj().bias().cloned(),
        )
        .unwrap(),
    ];
    let (idx, is_weight) = match label {
        "q.weight" => (0, true),
        "q.bias" => (0, false),
        "k.weight" => (1, true),
        "k.bias" => (1, false),
        "v.weight" => (2, true),
        "v.bias" => (2, false),
        "out.weight" => (3, true),
        "out.bias" => (3, false),
        other => panic!("rebuild_with_override: 未知のラベル {other}"),
    };
    let (w, b) = if is_weight {
        (override_tensor.clone(), linears[idx].bias().cloned())
    } else {
        (linears[idx].weight().clone(), Some(override_tensor.clone()))
    };
    linears[idx] = Linear::from_parameters(w, b).unwrap();
    let [q, k, v, out] = linears;
    MultiheadAttention::from_parameters(mha.num_heads(), q, k, v, out).unwrap()
}

#[test]
fn backward_numeric_grad_query_causal() {
    let mha = setup(31);
    let x_data = t(seq(41, B * L * E), &[B, L, E]);

    let forward_loss = |x: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let vars = mha.bind(&tape);
        let xv = tape.var(x);
        let out = vars.forward(&xv, &xv, &xv, None, true).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let xv = tape.var(&x_data);
    let out = vars.forward(&xv, &xv, &xv, None, true).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = grads.get(&xv).unwrap().unwrap().clone();

    let numeric = numeric_grad(&x_data, |perturbed| forward_loss(&perturbed));
    assert_grad_close("query(causal, self-attention)", &analytic, &numeric);
}

#[test]
fn backward_numeric_grad_query_with_attn_mask() {
    let mha = setup(37);
    let q_data = query_fixture();
    let k_data = key_fixture();
    let v_data = value_fixture();
    let mut mask_data = vec![true; L * S];
    mask_data[S - 1] = false; // 1 要素だけ block（全 masked 行にならないよう注意）

    let mask = Tensor::new(mask_data, &[L, S]).unwrap();

    let forward_loss = |q: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let vars = mha.bind(&tape);
        let qv = tape.var(q);
        let kv = tape.var(&k_data);
        let vv = tape.var(&v_data);
        let out = vars.forward(&qv, &kv, &vv, Some(&mask), false).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let qv = tape.var(&q_data);
    let kv = tape.var(&k_data);
    let vv = tape.var(&v_data);
    let out = vars.forward(&qv, &kv, &vv, Some(&mask), false).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = grads.get(&qv).unwrap().unwrap().clone();

    let numeric = numeric_grad(&q_data, |perturbed| forward_loss(&perturbed));
    assert_grad_close("query(attn_mask)", &analytic, &numeric);
}

/// self-attention の入力 `x` が `query`／`key`／`value` の 3 役割を
/// 兼ねる場合、`x` の勾配は 3 経路からの寄与が合算される
/// （`backward.rs::accumulate` の fan-out 合算契約）ことを、self-
/// attention（causal 版。上の `backward_numeric_grad_query_causal`）と
/// 「3 引数を独立した別 `Var`（同一値）として渡した場合」の勾配和が
/// 一致することで確認する。
#[test]
fn self_attention_input_gradient_sums_three_roles() {
    let mha = setup(41);
    let x_data = t(seq(51, B * L * E), &[B, L, E]);

    // 経路 A: self-attention（q=k=v=同一 Var）。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let vars_a = mha.bind(&tape_a);
    let xa = tape_a.var(&x_data);
    let out_a = vars_a.forward(&xa, &xa, &xa, None, false).unwrap();
    let grads_a = tape_a.backward(&out_a.sum(None).unwrap()).unwrap();
    let grad_a = grads_a.get(&xa).unwrap().unwrap().clone();

    // 経路 B: q/k/v を独立した別 Var（同一値）として渡し、3 つの勾配を
    // 手動で合算する。
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let vars_b = mha.bind(&tape_b);
    let qb = tape_b.var(&x_data);
    let kb = tape_b.var(&x_data);
    let vb = tape_b.var(&x_data);
    let out_b = vars_b.forward(&qb, &kb, &vb, None, false).unwrap();
    let grads_b = tape_b.backward(&out_b.sum(None).unwrap()).unwrap();
    let gq = grads_b.get(&qb).unwrap().unwrap();
    let gk = grads_b.get(&kb).unwrap().unwrap();
    let gv = grads_b.get(&vb).unwrap().unwrap();
    let summed: Vec<f32> = gq
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .zip(gk.contiguous().as_slice().unwrap().iter())
        .zip(gv.contiguous().as_slice().unwrap().iter())
        .map(|((&a, &b), &c)| a + b + c)
        .collect();
    let summed = t(summed, &[B, L, E]);

    assert_req2_close("self-attention 入力勾配の fan-out 合算", &grad_a, &summed);
}

// --- (d) Module::forward（self-attention）------------------------------

#[test]
fn module_forward_matches_self_attention_composition() {
    let mha = setup(43);
    let x = t(seq(61, B * L * E), &[B, L, E]);

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xa = tape_a.var(&x);
    let out_a = Module::forward(&mha, &tape_a, &xa).unwrap();

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let vars_b = mha.bind(&tape_b);
    let xb = tape_b.var(&x);
    let out_b = vars_b.forward(&xb, &xb, &xb, None, false).unwrap();

    assert_req2_close(
        "Module::forward == bind(tape).forward(x,x,x,None,false)",
        &out_a.to_tensor(),
        &out_b.to_tensor(),
    );
}

// --- (e) エラー経路（panic せず型付きエラー）---------------------------

#[test]
fn forward_rejects_rank_mismatch() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let bad = tape.var(&t(vec![0.0; B * E], &[B, E])); // rank 2（rank 3 を要求）
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    let err = vars.forward(&bad, &k, &v, None, false).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn forward_rejects_batch_mismatch() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture()); // B=2
    let k = tape.var(&t(vec![0.0; 3 * S * E], &[3, S, E])); // B=3
    let v = tape.var(&t(vec![0.0; 3 * S * E], &[3, S, E]));

    let err = vars.forward(&q, &k, &v, None, false).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn forward_rejects_embed_dim_mismatch() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&t(vec![0.0; B * L * (E + 1)], &[B, L, E + 1]));
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    let err = vars.forward(&q, &k, &v, None, false).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn forward_rejects_key_value_shape_mismatch() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture()); // [B, S, E]
    let v = tape.var(&t(vec![0.0; B * (S + 1) * E], &[B, S + 1, E])); // S が食い違う

    let err = vars.forward(&q, &k, &v, None, false).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn forward_rejects_mask_and_causal_together() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let x = tape.var(&t(seq(1, B * L * E), &[B, L, E]));
    let mask = Tensor::new(vec![true; L * L], &[L, L]).unwrap();

    let err = vars.forward(&x, &x, &x, Some(&mask), true).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn forward_rejects_fully_masked_row() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    // 1 行（li=0）だけ全 key を block する。
    let mut mask_data = vec![true; L * S];
    mask_data[..S].fill(false);
    let mask = Tensor::new(mask_data, &[L, S]).unwrap();

    let err = vars.forward(&q, &k, &v, Some(&mask), false).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn forward_rejects_mask_broadcast_incompatible_shape() {
    let mha = setup(3);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    // S+1 は scores の S 軸へ broadcast 不能。
    let mask = Tensor::new(vec![true; L * (S + 1)], &[L, S + 1]).unwrap();

    let err = vars.forward(&q, &k, &v, Some(&mask), false).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn from_parameters_and_new_reject_panic_free() {
    // 主要な拒否経路（`nn::attention` の単体テストで網羅済み）に加え、
    // 統合テスト側からも `new`／`from_parameters` が panic しないことを
    // 確認する（`nn::attention` の unit テストはクレート内部限定のため、
    // 公開 API 経路としての再確認）。
    assert!(MultiheadAttention::new(0, 2, true, 1).is_err());
    assert!(MultiheadAttention::new(4, 0, true, 1).is_err());
    assert!(MultiheadAttention::new(5, 2, true, 1).is_err());
    assert!(MultiheadAttention::new(4, 2, true, 1).is_ok());
}

// --- (f) 非 contiguous 入力・決定性 -------------------------------------

#[test]
fn forward_accepts_non_contiguous_query_and_matches_contiguous() {
    let mha = setup(53);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);

    // [E, B, L] で持たせてから transpose(0, 2) で [L, B, E] を作り、
    // さらに transpose(0, 1) で [B, L, E]（非 contiguous）にする。
    let raw = t(seq(71, E * B * L), &[E, B, L]);
    let raw_v = tape.var(&raw);
    let q_noncontig = raw_v.transpose(0, 2).unwrap().transpose(0, 1).unwrap();
    assert_eq!(q_noncontig.to_tensor().shape(), &[B, L, E]);

    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());

    let out_noncontig = vars
        .forward(&q_noncontig, &k, &v, None, false)
        .expect("非 contiguous な query でも forward は成功するはず");

    // 同じ論理値を持つ contiguous な query（`to_tensor()` で materialize
    // した値）と比較する。
    let q_contig_data = q_noncontig.to_tensor();
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let vars2 = mha.bind(&tape2);
    let q_contig = tape2.var(&q_contig_data);
    let k2 = tape2.var(&key_fixture());
    let v2 = tape2.var(&value_fixture());
    let out_contig = vars2.forward(&q_contig, &k2, &v2, None, false).unwrap();

    assert_req2_close(
        "非 contiguous query == 同値の contiguous query",
        &out_noncontig.to_tensor(),
        &out_contig.to_tensor(),
    );
}

#[test]
fn new_is_deterministic_and_projections_are_independent() {
    let a = MultiheadAttention::new(E, NUM_HEADS, true, 99).unwrap();
    let b = MultiheadAttention::new(E, NUM_HEADS, true, 99).unwrap();
    assert_eq!(
        a.q_proj().weight().contiguous().as_slice().unwrap(),
        b.q_proj().weight().contiguous().as_slice().unwrap(),
        "同一シードなら q_proj.weight は決定的に一致するはず"
    );
    assert_ne!(
        a.q_proj().weight().contiguous().as_slice().unwrap(),
        a.k_proj().weight().contiguous().as_slice().unwrap(),
        "q_proj と k_proj の weight は独立に導出されるはず"
    );
}

// --- (g) bind の勾配取得・0 サイズスモーク ------------------------------

#[test]
fn bind_gradients_are_retrievable_for_all_eight_params() {
    let mha = setup(61);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);
    let q = tape.var(&query_fixture());
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());
    let out = vars.forward(&q, &k, &v, None, false).unwrap();
    let grads = tape.backward(&out.sum(None).unwrap()).unwrap();

    for (label, var, expected_shape) in [
        ("q.weight", &vars.q.weight, vec![E, E]),
        ("k.weight", &vars.k.weight, vec![E, E]),
        ("v.weight", &vars.v.weight, vec![E, E]),
        ("out.weight", &vars.out.weight, vec![E, E]),
        ("q.bias", vars.q.bias.as_ref().unwrap(), vec![E]),
        ("k.bias", vars.k.bias.as_ref().unwrap(), vec![E]),
        ("v.bias", vars.v.bias.as_ref().unwrap(), vec![E]),
        ("out.bias", vars.out.bias.as_ref().unwrap(), vec![E]),
    ] {
        let g = grads
            .get(var)
            .unwrap()
            .unwrap_or_else(|| panic!("{label}: backward で到達するはず"));
        assert_eq!(
            g.shape(),
            expected_shape.as_slice(),
            "{label}: shape 不一致"
        );
    }
}

#[test]
fn forward_zero_seq_len_does_not_panic() {
    let mha = setup(67);
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = mha.bind(&tape);

    // L=0: query が空。
    let q_empty = tape.var(&t(Vec::new(), &[B, 0, E]));
    let k = tape.var(&key_fixture());
    let v = tape.var(&value_fixture());
    let out = vars.forward(&q_empty, &k, &v, None, false);
    assert!(out.is_ok(), "L=0 は panic せず動作するはず: {out:?}");

    // S=0: key/value が空。
    let q = tape.var(&query_fixture());
    let k_empty = tape.var(&t(Vec::new(), &[B, 0, E]));
    let v_empty = tape.var(&t(Vec::new(), &[B, 0, E]));
    let out2 = vars.forward(&q, &k_empty, &v_empty, None, false);
    assert!(out2.is_ok(), "S=0 は panic せず動作するはず: {out2:?}");
}
