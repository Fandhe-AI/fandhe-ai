//! `Var::scaled_dot_product_attention`（イシュー #1639。親 #1605の
//! sub-issue (a)）の受け入れ条件対応テスト。
//!
//! - forward がブルートフォース `f64` ホスト参照実装（softmax(Q K^T *
//!   scale + mask) V を素朴な多重ループで計算）と REQ-2 統一複合判定で
//!   一致すること（rank 2・rank 4・バッチ broadcast・非正方形状・
//!   `is_causal`・明示 `attn_mask`・`scale` 上書き／既定値）。
//! - 手動合成（`mul`→`transpose`→`matmul`→`masked_fill`→`softmax`→
//!   `matmul`）と bit 完全一致すること（「新規カーネルを追加せず既存
//!   演算の合成として実装した」ことの機械的裏付け）。
//! - backward（q/k/v への勾配）が中央差分（数値微分）と一致すること
//!   （causal／mask あり・なしの両方）。
//! - 同一 `Var` の self-attention（`sdpa(x, x, x)`）で 3 経路の勾配が
//!   合算されること。
//! - 拒否系（rank<2・E/S 不一致・batch broadcast 不能・mask broadcast
//!   不能・mask+causal 同時指定・全 masked 行・scale 非有限／非正・
//!   `E==0 && scale==None`・テープ不一致）が panic せず型付きエラーに
//!   なること。
//! - 0 サイズ（L=0／S=0／Ev=0）で panic せず動作すること。
//! - 大きめの入力で NaN／Inf が出力に現れないこと。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn seq(seed: u64, len: usize) -> Vec<f32> {
    // matmul_batched.rs と同一の簡易 LCG（決定的疑似乱数。値域は概ね
    // [-1, 1)）。
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

// --- ブルートフォース f64 ホスト参照実装 ---

/// [`brute_sdpa`] の引数群（clippy `too_many_arguments` 回避のための
/// まとめ役。フィールドの意味は各呼び出し元のローカル変数名と対応）。
struct BruteSdpaArgs<'a> {
    batch: usize,
    l: usize,
    s: usize,
    e: usize,
    ev: usize,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    /// `[batch, l, s]`（`true` = masked）。`None` は無 mask。
    blocked: Option<&'a [bool]>,
    scale: f64,
}

/// `[batch..., L, E]`・`[batch..., S, E]`・`[batch..., S, Ev]`（バッチ
/// 次元は事前に broadcast 済みのフラット形状を渡す想定。本ファイルの
/// テストは全て等バッチまたは呼び出し側で broadcast 後の形状を渡す）を
/// 受け取り、`softmax(Q K^T * scale + mask) V` を素朴な多重ループで
/// `f64` 計算するブルートフォース参照実装。
fn brute_sdpa(args: BruteSdpaArgs<'_>) -> Vec<f32> {
    let BruteSdpaArgs {
        batch,
        l,
        s,
        e,
        ev,
        q,
        k,
        v,
        blocked,
        scale,
    } = args;
    let mut out = vec![0f32; batch * l * ev];
    for b in 0..batch {
        for i in 0..l {
            // scores[j] = scale * sum_e q[b,i,e] * k[b,j,e]
            let mut scores = vec![0f64; s];
            for (j, score) in scores.iter_mut().enumerate() {
                let mut acc = 0f64;
                for ee in 0..e {
                    let qv = q[(b * l + i) * e + ee] as f64;
                    let kv = k[(b * s + j) * e + ee] as f64;
                    acc += qv * kv;
                }
                *score = acc * scale;
                if let Some(bl) = blocked
                    && bl[(b * l + i) * s + j]
                {
                    *score = f64::NEG_INFINITY;
                }
            }
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exps = vec![0f64; s];
            let mut sum = 0f64;
            for (j, ex) in exps.iter_mut().enumerate() {
                let v_ = if max.is_finite() {
                    (scores[j] - max).exp()
                } else {
                    0.0
                };
                *ex = v_;
                sum += v_;
            }
            for ee in 0..ev {
                let mut acc = 0f64;
                for j in 0..s {
                    let w = if sum > 0.0 { exps[j] / sum } else { 0.0 };
                    acc += w * v[(b * s + j) * ev + ee] as f64;
                }
                out[(b * l + i) * ev + ee] = acc as f32;
            }
        }
    }
    out
}

fn req2_assert(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 長さ不一致");
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            common::req2_close(a as f64, e as f64),
            "{label}[{idx}]: actual={a} expected={e}"
        );
    }
}

// --- 1. forward: ブルートフォース f64 参照実装との REQ-2 突合 ---

#[test]
fn sdpa_forward_matches_brute_reference_rank2_default_scale() {
    let (l, s, e, ev) = (4usize, 5usize, 3usize, 2usize);
    let q = t(seq(1, l * e), &[l, e]);
    let k = t(seq(2, s * e), &[s, e]);
    let v = t(seq(3, s * ev), &[s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .to_tensor();

    let scale = 1.0 / (e as f64).sqrt();
    let expected = brute_sdpa(BruteSdpaArgs {
        batch: 1,
        l,
        s,
        e,
        ev,
        q: &dense(&q),
        k: &dense(&k),
        v: &dense(&v),
        blocked: None,
        scale,
    });
    req2_assert("rank2 default scale", &dense(&out), &expected);
}

#[test]
fn sdpa_forward_matches_brute_reference_rank4_batched() {
    let (b0, b1, l, s, e, ev) = (2usize, 3usize, 3usize, 4usize, 5usize, 2usize);
    let batch = b0 * b1;
    let q = t(seq(10, batch * l * e), &[b0, b1, l, e]);
    let k = t(seq(20, batch * s * e), &[b0, b1, s, e]);
    let v = t(seq(30, batch * s * ev), &[b0, b1, s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, Some(0.5))
        .unwrap()
        .to_tensor();

    let expected = brute_sdpa(BruteSdpaArgs {
        batch,
        l,
        s,
        e,
        ev,
        q: &dense(&q),
        k: &dense(&k),
        v: &dense(&v),
        blocked: None,
        scale: 0.5,
    });
    req2_assert("rank4 batched explicit scale", &dense(&out), &expected);
}

#[test]
fn sdpa_forward_matches_brute_reference_kv_batch_broadcast() {
    // k/v のバッチ次元が q より小さく broadcast されるケース
    // （`Var::matmul` の NumPy 互換バッチブロードキャスト）。
    let (b, l, s, e, ev) = (3usize, 2usize, 3usize, 4usize, 2usize);
    let q = t(seq(40, b * l * e), &[b, l, e]);
    let k = t(seq(50, s * e), &[1, s, e]);
    let v = t(seq(60, s * ev), &[1, s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .to_tensor();

    // ブルートフォース側は k/v を b 回複製して等バッチとして評価する。
    let k_bc: Vec<f32> = (0..b).flat_map(|_| dense(&k)).collect();
    let v_bc: Vec<f32> = (0..b).flat_map(|_| dense(&v)).collect();
    let scale = 1.0 / (e as f64).sqrt();
    let expected = brute_sdpa(BruteSdpaArgs {
        batch: b,
        l,
        s,
        e,
        ev,
        q: &dense(&q),
        k: &k_bc,
        v: &v_bc,
        blocked: None,
        scale,
    });
    req2_assert("kv batch broadcast", &dense(&out), &expected);
}

#[test]
fn sdpa_forward_is_causal_zeroes_upper_triangle_weights() {
    let (l, s, e, ev) = (4usize, 4usize, 2usize, 3usize);
    let q = t(seq(70, l * e), &[l, e]);
    let k = t(seq(80, s * e), &[s, e]);
    let v = t(seq(90, s * ev), &[s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, true, None)
        .unwrap()
        .to_tensor();

    let blocked: Vec<bool> = (0..l * s)
        .map(|idx| {
            let i = idx / s;
            let j = idx % s;
            j > i
        })
        .collect();
    let scale = 1.0 / (e as f64).sqrt();
    let expected = brute_sdpa(BruteSdpaArgs {
        batch: 1,
        l,
        s,
        e,
        ev,
        q: &dense(&q),
        k: &dense(&k),
        v: &dense(&v),
        blocked: Some(&blocked),
        scale,
    });
    req2_assert("is_causal", &dense(&out), &expected);
}

#[test]
fn sdpa_forward_explicit_attn_mask_matches_brute_reference() {
    // padding mask 相当: [1, s] を broadcast（bool `true` = attend）。
    let (b, l, s, e, ev) = (2usize, 3usize, 4usize, 2usize, 2usize);
    let q = t(seq(100, b * l * e), &[b, l, e]);
    let k = t(seq(110, b * s * e), &[b, s, e]);
    let v = t(seq(120, b * s * ev), &[b, s, ev]);
    // 最後の key 位置のみ padding（masked）とする。
    let mask_allowed = vec![true, true, true, false];
    let mask = Tensor::new(mask_allowed.clone(), &[1, 1, s]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, Some(&mask), false, None)
        .unwrap()
        .to_tensor();

    let blocked_row: Vec<bool> = mask_allowed.iter().map(|&a| !a).collect();
    let blocked: Vec<bool> = (0..b * l).flat_map(|_| blocked_row.clone()).collect();
    let scale = 1.0 / (e as f64).sqrt();
    let expected = brute_sdpa(BruteSdpaArgs {
        batch: b,
        l,
        s,
        e,
        ev,
        q: &dense(&q),
        k: &dense(&k),
        v: &dense(&v),
        blocked: Some(&blocked),
        scale,
    });
    req2_assert("explicit attn_mask", &dense(&out), &expected);
}

// --- 2. 合成一致: 手動合成との bit 完全一致 ---

#[test]
fn sdpa_forward_bit_matches_manual_composition_no_mask() {
    let (l, s, e, ev) = (3usize, 4usize, 2usize, 3usize);
    let q = t(seq(200, l * e), &[l, e]);
    let k = t(seq(210, s * e), &[s, e]);
    let v = t(seq(220, s * ev), &[s, ev]);
    let scale = 1.0 / (e as f32).sqrt();

    let tape1 = Tape::new_with_ops(common::naive_ops());
    let (q1, k1, v1) = (tape1.var(&q), tape1.var(&k), tape1.var(&v));
    let sdpa_out = Var::scaled_dot_product_attention(&q1, &k1, &v1, None, false, None)
        .unwrap()
        .to_tensor();

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let (q2, k2, v2) = (tape2.var(&q), tape2.var(&k), tape2.var(&v));
    let scale_var = tape2.var(&Tensor::scalar(scale));
    let manual_out = q2
        .mul(&scale_var)
        .unwrap()
        .matmul(&k2.transpose(0, 1).unwrap())
        .unwrap()
        .softmax(1)
        .unwrap()
        .matmul(&v2)
        .unwrap()
        .to_tensor();

    assert_eq!(dense(&sdpa_out), dense(&manual_out), "bit 完全一致するはず");
}

// --- 3. backward: 中央差分との突合 ---

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let a = dense(analytic);
    let n = dense(numeric);
    for (idx, (&av, &nv)) in a.iter().zip(n.iter()).enumerate() {
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{idx}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
        );
    }
}

/// `loss = sum(sdpa(q, k, v, mask, causal, scale))` を使い捨てテープで
/// 再評価する（中央差分の各サンプル点用）。
fn forward_loss_sum(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    mask: Option<&Tensor<bool>>,
    is_causal: bool,
    scale: Option<f32>,
) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(q);
    let kv = tape.var(k);
    let vv = tape.var(v);
    let loss = Var::scaled_dot_product_attention(&qv, &kv, &vv, mask, is_causal, scale)
        .unwrap()
        .sum(None)
        .unwrap();
    loss.to_tensor()
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn numeric_grad(target: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data = dense(target);
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

#[test]
fn sdpa_backward_matches_numeric_grad_no_mask() {
    let (l, s, e, ev) = (3usize, 3usize, 2usize, 2usize);
    let q = t(seq(300, l * e), &[l, e]);
    let k = t(seq(310, s * e), &[s, e]);
    let v = t(seq(320, s * ev), &[s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let loss = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dq = grads.get(&qv).unwrap().expect("q は loss に到達する");
    let dk = grads.get(&kv).unwrap().expect("k は loss に到達する");
    let dv = grads.get(&vv).unwrap().expect("v は loss に到達する");

    let num_dq = numeric_grad(&q, |q2| forward_loss_sum(&q2, &k, &v, None, false, None));
    let num_dk = numeric_grad(&k, |k2| forward_loss_sum(&q, &k2, &v, None, false, None));
    let num_dv = numeric_grad(&v, |v2| forward_loss_sum(&q, &k, &v2, None, false, None));
    assert_grad_close("dq (no mask)", dq, &num_dq);
    assert_grad_close("dk (no mask)", dk, &num_dk);
    assert_grad_close("dv (no mask)", dv, &num_dv);
}

#[test]
fn sdpa_backward_matches_numeric_grad_is_causal() {
    let (l, s, e, ev) = (3usize, 3usize, 2usize, 2usize);
    let q = t(seq(330, l * e), &[l, e]);
    let k = t(seq(340, s * e), &[s, e]);
    let v = t(seq(350, s * ev), &[s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let loss = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, true, None)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dq = grads.get(&qv).unwrap().expect("q は loss に到達する");
    let dk = grads.get(&kv).unwrap().expect("k は loss に到達する");
    let dv = grads.get(&vv).unwrap().expect("v は loss に到達する");

    let num_dq = numeric_grad(&q, |q2| forward_loss_sum(&q2, &k, &v, None, true, None));
    let num_dk = numeric_grad(&k, |k2| forward_loss_sum(&q, &k2, &v, None, true, None));
    let num_dv = numeric_grad(&v, |v2| forward_loss_sum(&q, &k, &v2, None, true, None));
    assert_grad_close("dq (causal)", dq, &num_dq);
    assert_grad_close("dk (causal)", dk, &num_dk);
    assert_grad_close("dv (causal)", dv, &num_dv);
}

#[test]
fn sdpa_backward_matches_numeric_grad_explicit_mask() {
    let (l, s, e, ev) = (2usize, 3usize, 2usize, 2usize);
    let q = t(seq(360, l * e), &[l, e]);
    let k = t(seq(370, s * e), &[s, e]);
    let v = t(seq(380, s * ev), &[s, ev]);
    let mask = Tensor::new(vec![true, true, false], &[1, s]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let qv = tape.var(&q);
    let kv = tape.var(&k);
    let vv = tape.var(&v);
    let loss = Var::scaled_dot_product_attention(&qv, &kv, &vv, Some(&mask), false, None)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dq = grads.get(&qv).unwrap().expect("q は loss に到達する");
    let dk = grads.get(&kv).unwrap().expect("k は loss に到達する");
    let dv = grads.get(&vv).unwrap().expect("v は loss に到達する");

    let num_dq = numeric_grad(&q, |q2| {
        forward_loss_sum(&q2, &k, &v, Some(&mask), false, None)
    });
    let num_dk = numeric_grad(&k, |k2| {
        forward_loss_sum(&q, &k2, &v, Some(&mask), false, None)
    });
    let num_dv = numeric_grad(&v, |v2| {
        forward_loss_sum(&q, &k, &v2, Some(&mask), false, None)
    });
    assert_grad_close("dq (explicit mask)", dq, &num_dq);
    assert_grad_close("dk (explicit mask)", dk, &num_dk);
    assert_grad_close("dv (explicit mask)", dv, &num_dv);
}

/// 同一 `Var` を q/k/v に渡す self-attention（勾配が 3 経路で合算される
/// ことの確認）。
#[test]
fn sdpa_self_attention_backward_matches_numeric_grad() {
    let (l, e) = (3usize, 2usize);
    let x = t(seq(390, l * e), &[l, e]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let loss = Var::scaled_dot_product_attention(&xv, &xv, &xv, None, false, None)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x, |x2| forward_loss_sum(&x2, &x2, &x2, None, false, None));
    assert_grad_close("self-attention dx", dx, &num_dx);
}

// --- 4. 拒否系（panic せず型付きエラー） ---

#[test]
fn sdpa_rejects_rank1_query() {
    let q = t(vec![1.0, 2.0], &[2]);
    let k = t(seq(400, 3 * 2), &[3, 2]);
    let v = t(seq(410, 3 * 2), &[3, 2]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn sdpa_rejects_e_mismatch() {
    let q = t(seq(420, 2 * 3), &[2, 3]);
    let k = t(seq(430, 3 * 4), &[3, 4]); // E=4 != q の E=3
    let v = t(seq(440, 3 * 2), &[3, 2]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn sdpa_rejects_s_mismatch_between_key_and_value() {
    let q = t(seq(450, 2 * 3), &[2, 3]);
    let k = t(seq(460, 4 * 3), &[4, 3]); // S=4
    let v = t(seq(470, 5 * 2), &[5, 2]); // S=5（不一致）
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn sdpa_rejects_batch_broadcast_incompatible() {
    let q = t(seq(480, 2 * 2 * 3), &[2, 2, 3]);
    let k = t(seq(490, 3 * 4 * 3), &[3, 4, 3]); // batch 3 と 2 は broadcast 不能
    let v = t(seq(500, 3 * 4 * 2), &[3, 4, 2]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn sdpa_rejects_mask_not_broadcastable() {
    let q = t(seq(510, 2 * 3), &[2, 3]);
    let k = t(seq(520, 4 * 3), &[4, 3]);
    let v = t(seq(530, 4 * 2), &[4, 2]);
    // mask shape [5] は scores shape [2, 4] へ broadcast 不能。
    let mask = Tensor::new(vec![true; 5], &[5]).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err =
        Var::scaled_dot_product_attention(&qv, &kv, &vv, Some(&mask), false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn sdpa_rejects_mask_and_causal_together() {
    let q = t(seq(540, 2 * 3), &[2, 3]);
    let k = t(seq(550, 2 * 3), &[2, 3]);
    let v = t(seq(560, 2 * 2), &[2, 2]);
    let mask = Tensor::new(vec![true, true], &[1, 2]).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err =
        Var::scaled_dot_product_attention(&qv, &kv, &vv, Some(&mask), true, None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn sdpa_rejects_fully_masked_row() {
    let q = t(seq(570, 2 * 3), &[2, 3]);
    let k = t(seq(580, 2 * 3), &[2, 3]);
    let v = t(seq(590, 2 * 2), &[2, 2]);
    // 行 0（i=0）は全 key が masked（j=0,1 とも false）。
    let mask = Tensor::new(vec![false, false, true, true], &[2, 2]).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err =
        Var::scaled_dot_product_attention(&qv, &kv, &vv, Some(&mask), false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn sdpa_rejects_non_finite_or_non_positive_scale() {
    let q = t(seq(600, 2 * 3), &[2, 3]);
    let k = t(seq(610, 2 * 3), &[2, 3]);
    let v = t(seq(620, 2 * 2), &[2, 2]);
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
        let err =
            Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, Some(bad)).unwrap_err();
        assert!(
            matches!(err, AutodiffError::InvalidArgument(_)),
            "scale={bad}"
        );
    }
}

#[test]
fn sdpa_rejects_zero_e_with_default_scale() {
    let q = t(Vec::new(), &[2, 0]);
    let k = t(Vec::new(), &[3, 0]);
    let v = t(seq(630, 3 * 2), &[3, 2]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let err = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn sdpa_allows_zero_e_with_explicit_scale() {
    // E == 0: scores は全要素ゼロ（空和）になる有限値のため、明示 scale
    // であれば拒否しない。
    let q = t(Vec::new(), &[2, 0]);
    let k = t(Vec::new(), &[3, 0]);
    let v = t(seq(640, 3 * 2), &[3, 2]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, Some(1.0))
        .unwrap()
        .to_tensor();
    assert_eq!(out.shape(), &[2, 2]);
    for &value in &dense(&out) {
        assert!(value.is_finite());
    }
}

#[test]
fn sdpa_rejects_cross_tape_query_key() {
    let q = t(seq(650, 2 * 3), &[2, 3]);
    let k = t(seq(660, 2 * 3), &[2, 3]);
    let v = t(seq(670, 2 * 2), &[2, 2]);
    let tape1 = Tape::new_with_ops(common::naive_ops());
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let qv = tape1.var(&q);
    let kv = tape2.var(&k);
    let vv = tape1.var(&v);
    let err = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

// --- 5. 0 サイズ ---

#[test]
fn sdpa_zero_l_returns_empty_tensor_without_panic() {
    let q = t(Vec::new(), &[0, 3]);
    let k = t(seq(680, 4 * 3), &[4, 3]);
    let v = t(seq(690, 4 * 2), &[4, 2]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .to_tensor();
    assert_eq!(out.shape(), &[0, 2]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn sdpa_zero_s_returns_empty_tensor_without_panic() {
    // S == 0（key/value が空）: 出力 shape は [L, Ev] = [3, 4]（numel は
    // 0 ではない）。各要素は「空集合上の和」として 0.0（有限）になる
    // （`weights.matmul(value)` の内部次元 0 の GEMM 契約）。
    let q = t(seq(700, 3 * 2), &[3, 2]);
    let k = t(Vec::new(), &[0, 2]);
    let v = t(Vec::new(), &[0, 4]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .to_tensor();
    assert_eq!(out.shape(), &[3, 4]);
    for &value in &dense(&out) {
        assert!(value.is_finite());
        assert_eq!(value, 0.0);
    }
}

#[test]
fn sdpa_zero_ev_returns_empty_tensor_without_panic() {
    let q = t(seq(710, 3 * 2), &[3, 2]);
    let k = t(seq(720, 4 * 2), &[4, 2]);
    let v = t(Vec::new(), &[4, 0]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .to_tensor();
    assert_eq!(out.shape(), &[3, 0]);
    assert_eq!(out.numel(), 0);
}

// --- 6. NaN/Inf 非発生 ---

#[test]
fn sdpa_large_inputs_produce_finite_output() {
    let (l, s, e, ev) = (3usize, 3usize, 2usize, 2usize);
    let q: Vec<f32> = seq(730, l * e).into_iter().map(|v| v * 50.0).collect();
    let k: Vec<f32> = seq(740, s * e).into_iter().map(|v| v * 50.0).collect();
    let v = seq(750, s * ev);
    let q = t(q, &[l, e]);
    let k = t(k, &[s, e]);
    let v = t(v, &[s, ev]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));
    let out = Var::scaled_dot_product_attention(&qv, &kv, &vv, None, false, None)
        .unwrap()
        .to_tensor();
    for &value in &dense(&out) {
        assert!(
            value.is_finite(),
            "output must stay finite for large inputs"
        );
    }
}
