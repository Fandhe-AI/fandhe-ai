//! イシュー #1744（親 #1610）: `fandhe_ai_autodiff::nn::optim::Lamb`
//! （LAMB。You et al., 2019, arXiv:1904.00962 Algorithm 2）の受け入れ
//! テスト。
//!
//! `torch.optim` 本体に LAMB 相当の実装はなく（`crates/autodiff/src/
//! nn/optim/lamb.rs` モジュール doc・`docs/compat-feature-gap.md`
//! §2.9）、実行環境に PyTorch／`torch_optimizer` も無いため、
//! `nn_optim_adamw.rs` のような PyTorch 参照値 fixture は用意しない。
//! 代わりに以下で受け入れを担保する:
//!
//! 1. テストファイル内に独立に書いた f64 参照実装（paper Algorithm 2
//!    そのままの形。`lamb.rs` 本体の「lr を先に折り込んだ」実装形とは
//!    異なる演算列）との突合（[`lamb_matches_independent_f64_reference`]）。
//! 2. 解析的恒等式 3 件（[`lamb_wd_gt_zero_zero_grad_is_wd_independent`]・
//!    [`lamb_wd_zero_matches_scaled_adamw_step`]・
//!    [`lamb_scale_invariant_update_wd_zero`]）。
//! 3. 再現性（[`lamb_step_is_deterministic`]）・収束
//!    （[`mlp_converges_with_lamb`]）。
//!
//! **数値判定の規律**: 1・2(a)(b) はバックエンド間統一複合判定
//! 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
//! （`.claude/rules/coding-rust.md`。判定式は `common::req2_close` へ
//! 委譲。新設 tolerance の緩和ではない）。2(c)（2 の冪スケール不変性）
//! のみ bit 完全一致（丸めなしに成立する構造的性質のため）。

// 本ファイルの範囲ループは `Tensor::get(&[i])`／`index_of(&shape, i)` へ
// フラット添字 `i` そのものを渡す必要があり、`enumerate()` で得られる
// 要素参照だけでは代替できない（`i` 自身が必須の引数）。
#![allow(clippy::needless_range_loop)]

mod common;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig, Lamb, LambConfig};
use fandhe_ai_tensor_core::Tensor;

/// `Tensor::get` は多次元添字を要求するため、行優先の平坦添字 `i` を
/// `shape` から多次元添字へ復元する（`nn_optim_adam.rs::index_of` と
/// 同一パターン）。
fn index_of(shape: &[usize], i: usize) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = vec![0usize; shape.len()];
    let mut rem = i;
    for d in (0..shape.len()).rev() {
        idx[d] = rem % shape[d];
        rem /= shape[d];
    }
    idx
}

// =====================================================================
// 独立 f64 参照実装（paper Algorithm 2 の素直な形）
// =====================================================================

/// 1 パラメータスロット分の `m`／`v`（f64）。`lamb.rs` 本体の
/// `SlotState`（f32）とは独立の型（意図的な複製。本体実装のバグと
/// 辻褄を合わせないため）。
#[derive(Default)]
struct RefSlot {
    m: Vec<f64>,
    v: Vec<f64>,
}

/// paper Algorithm 2 をそのまま（`trust = ‖x‖₂ / ‖u‖₂`。`lamb.rs` が
/// 採る「`lr` を先に折り込んだ `t = lr*u`・`f = lr*‖x‖/‖t‖`」という
/// 実装形とは異なる演算順で独立に計算する）。全演算を f64 で行い、
/// 最後に 1 回だけ `f32` へ downcast する。
#[allow(clippy::too_many_arguments)]
fn lamb_reference_step(
    slot: &mut RefSlot,
    param: &[f32],
    grad: &[f32],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    beta1_pow_t: f64,
    beta2_pow_t: f64,
) -> Vec<f32> {
    if slot.m.is_empty() {
        slot.m = vec![0.0; param.len()];
        slot.v = vec![0.0; param.len()];
    }
    let bias_correction1 = 1.0 - beta1_pow_t;
    let bias_correction2 = 1.0 - beta2_pow_t;

    let mut r = vec![0.0f64; param.len()];
    for i in 0..param.len() {
        let g = grad[i] as f64;
        let m = beta1 * slot.m[i] + (1.0 - beta1) * g;
        let v = beta2 * slot.v[i] + (1.0 - beta2) * g * g;
        slot.m[i] = m;
        slot.v[i] = v;
        let m_hat = m / bias_correction1;
        let v_hat = v / bias_correction2;
        r[i] = m_hat / (v_hat.sqrt() + eps);
    }

    let mut u = vec![0.0f64; param.len()];
    let mut norm_x_sq = 0.0f64;
    let mut norm_u_sq = 0.0f64;
    for i in 0..param.len() {
        let x = param[i] as f64;
        u[i] = r[i] + weight_decay * x;
        norm_x_sq += x * x;
        norm_u_sq += u[i] * u[i];
    }
    let norm_x = norm_x_sq.sqrt();
    let norm_u = norm_u_sq.sqrt();
    let trust = if norm_x == 0.0 || norm_u == 0.0 {
        1.0
    } else {
        norm_x / norm_u
    };

    (0..param.len())
        .map(|i| (param[i] as f64 - lr * trust * u[i]) as f32)
        .collect()
}

/// 受け入れ段 1: 決定的シードで 2 パラメータ（rank 2・rank 1）×10
/// step×3 ハイパーパラメータケースを、独立 f64 参照実装
/// （[`lamb_reference_step`]）へ統一複合判定で突合する。
#[test]
fn lamb_matches_independent_f64_reference() {
    struct Case {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    }
    let cases = [
        Case {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
        },
        Case {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.1,
        },
        Case {
            lr: 0.05,
            beta1: 0.8,
            beta2: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
        },
    ];

    let shape_a: [usize; 2] = [2, 3];
    let shape_b: [usize; 1] = [4];

    for (case_idx, case) in cases.iter().enumerate() {
        let cfg = LambConfig {
            lr: case.lr,
            beta1: case.beta1,
            beta2: case.beta2,
            eps: case.eps,
            weight_decay: case.weight_decay,
        };
        let mut opt =
            Lamb::new(cfg).unwrap_or_else(|e| panic!("case {case_idx}: Lamb::new 失敗: {e}"));
        let mut ref_a = RefSlot::default();
        let mut ref_b = RefSlot::default();

        let mut rng = Xorshift64Star::new(0xBEEF_1744 ^ (case_idx as u64));
        let mut param_a: Vec<f32> = rng.fill_vec(6);
        let mut param_b: Vec<f32> = rng.fill_vec(4);
        let mut beta1_pow_t = 1.0f64;
        let mut beta2_pow_t = 1.0f64;

        for step in 0..10 {
            let grad_a = rng.fill_vec(6);
            let grad_b = rng.fill_vec(4);

            let param_a_t = Tensor::new(param_a.clone(), &shape_a).unwrap();
            let param_b_t = Tensor::new(param_b.clone(), &shape_b).unwrap();
            let grad_a_t = Tensor::new(grad_a.clone(), &shape_a).unwrap();
            let grad_b_t = Tensor::new(grad_b.clone(), &shape_b).unwrap();

            let updated = opt
                .step(&[(&param_a_t, &grad_a_t), (&param_b_t, &grad_b_t)])
                .unwrap_or_else(|e| panic!("case {case_idx} step {step}: Lamb::step 失敗: {e}"));

            beta1_pow_t *= case.beta1 as f64;
            beta2_pow_t *= case.beta2 as f64;

            let expected_a = lamb_reference_step(
                &mut ref_a,
                &param_a,
                &grad_a,
                case.lr as f64,
                case.beta1 as f64,
                case.beta2 as f64,
                case.eps as f64,
                case.weight_decay as f64,
                beta1_pow_t,
                beta2_pow_t,
            );
            let expected_b = lamb_reference_step(
                &mut ref_b,
                &param_b,
                &grad_b,
                case.lr as f64,
                case.beta1 as f64,
                case.beta2 as f64,
                case.eps as f64,
                case.weight_decay as f64,
                beta1_pow_t,
                beta2_pow_t,
            );

            for i in 0..6 {
                let actual = updated[0].get(&index_of(&shape_a, i)).unwrap();
                assert!(
                    common::req2_close(actual as f64, expected_a[i] as f64),
                    "case={case_idx} step={step} param_a[{i}]: actual={actual} expected={}",
                    expected_a[i]
                );
            }
            for i in 0..4 {
                let actual = updated[1].get(&[i]).unwrap();
                assert!(
                    common::req2_close(actual as f64, expected_b[i] as f64),
                    "case={case_idx} step={step} param_b[{i}]: actual={actual} expected={}",
                    expected_b[i]
                );
            }

            param_a = (0..6)
                .map(|i| updated[0].get(&index_of(&shape_a, i)).unwrap())
                .collect();
            param_b = (0..4).map(|i| updated[1].get(&[i]).unwrap()).collect();
        }
    }
}

// =====================================================================
// 解析的恒等式
// =====================================================================

/// 恒等式 (a): 初回 step・勾配ゼロ・`weight_decay > 0` では
/// `r=0`・`u=weight_decay*x`・`trust=1/weight_decay` となり、
/// `x_new = x*(1-lr)` が `weight_decay` の値に依存しない
/// （`lamb.rs` モジュール doc「実装形」節の帰結。統一複合判定で
/// 2 つの `weight_decay` 値の結果が一致することを確認する）。
#[test]
fn lamb_wd_gt_zero_zero_grad_is_wd_independent() {
    let lr = 0.1f32;
    let param_data = vec![1.0f32, -2.0, 0.5];
    let grad_data = vec![0.0f32, 0.0, 0.0];

    let mut results = Vec::new();
    for weight_decay in [0.01f32, 0.5f32] {
        let cfg = LambConfig {
            lr,
            weight_decay,
            ..LambConfig::default()
        };
        let mut opt = Lamb::new(cfg).unwrap();
        let param = Tensor::new(param_data.clone(), &[3]).unwrap();
        let grad = Tensor::new(grad_data.clone(), &[3]).unwrap();
        let out = opt
            .step(&[(&param, &grad)])
            .unwrap_or_else(|e| panic!("weight_decay={weight_decay}: Lamb::step 失敗: {e}"));
        results.push((weight_decay, out));
    }

    let expected: Vec<f32> = param_data.iter().map(|x| x * (1.0 - lr)).collect();
    for (weight_decay, out) in &results {
        for i in 0..3 {
            let actual = out[0].get(&[i]).unwrap();
            assert!(
                common::req2_close(actual as f64, expected[i] as f64),
                "weight_decay={weight_decay} index={i}: actual={actual} expected={} \
                 （x*(1-lr) から乖離。trust ratio が weight_decay に依存してはならない）",
                expected[i]
            );
        }
    }
}

/// 恒等式 (b)（`lamb.rs` モジュール doc「実装形」節）: `weight_decay=0`
/// では `x - x_new = f*s`（`s` は `AdamW(weight_decay=0)` の更新量
/// `x - adamw_step(x, g)`。`f` はテスト側で独立に計算した trust
/// ratio）が成り立つ。`AdamW::step`／`Lamb::step` は毎 step 同一の
/// `param`（前 step の `Lamb::step` 出力）・`grad` を受け取る
/// （両者の内部モーメント状態は grad 系列にのみ依存するため、
/// 「同一状態から 1 step」という前提が毎 step 成立する）。
#[test]
fn lamb_wd_zero_matches_scaled_adamw_step() {
    let lr = 0.05f32;
    let cfg_lamb = LambConfig {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.0,
    };
    let cfg_adamw = AdamWConfig {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-6,
        weight_decay: 0.0,
    };
    let mut lamb = Lamb::new(cfg_lamb).unwrap();
    let mut adamw = AdamW::new(cfg_adamw).unwrap();

    let mut rng = Xorshift64Star::new(0xFEED_1744);
    let mut param = Tensor::new(vec![0.5, -1.2, 3.0, 0.7], &[4]).unwrap();

    for step in 0..5 {
        let grad_data = rng.fill_vec(4);
        let grad = Tensor::new(grad_data, &[4]).unwrap();

        let lamb_out = lamb
            .step(&[(&param, &grad)])
            .unwrap_or_else(|e| panic!("step {step}: Lamb::step 失敗: {e}"));
        let adamw_out = adamw
            .step(&[(&param, &grad)])
            .unwrap_or_else(|e| panic!("step {step}: AdamW::step 失敗: {e}"));

        let mut s = [0.0f64; 4];
        let mut norm_x_sq = 0.0f64;
        let mut norm_s_sq = 0.0f64;
        for i in 0..4 {
            let x = param.get(&[i]).unwrap() as f64;
            let adamw_new = adamw_out[0].get(&[i]).unwrap() as f64;
            s[i] = x - adamw_new;
            norm_x_sq += x * x;
            norm_s_sq += s[i] * s[i];
        }
        let norm_x = norm_x_sq.sqrt();
        let norm_s = norm_s_sq.sqrt();
        let f = if norm_x == 0.0 || norm_s == 0.0 {
            1.0
        } else {
            lr as f64 * norm_x / norm_s
        };

        for i in 0..4 {
            let x = param.get(&[i]).unwrap() as f64;
            let expected = x - f * s[i];
            let actual = lamb_out[0].get(&[i]).unwrap() as f64;
            assert!(
                common::req2_close(actual, expected),
                "step={step} index={i}: actual={actual} expected={expected}"
            );
        }

        param = lamb_out.into_iter().next().unwrap();
    }
}

/// 恒等式 (c)（`lamb.rs` モジュール doc「実装形」節）: `weight_decay=0`
/// のとき `m`／`v`／`t` は `x` に依存しない（`g_eff` が常に生の勾配
/// のため）。`x → c*x`（`c=4.0`。2 の冪なので f32／f64 いずれの演算
/// でも丸めなしにスケールする）とすると更新量 `x - x_new` が **bit
/// 完全に** `c` 倍になる（`trust` が `c` 倍・`r`＝`t` は不変。overflow・
/// 非正規化数に入らない値域を選ぶ）。
#[test]
fn lamb_scale_invariant_update_wd_zero() {
    let cfg = LambConfig {
        weight_decay: 0.0,
        ..LambConfig::default()
    };
    let mut opt_a = Lamb::new(cfg).unwrap();
    let mut opt_b = Lamb::new(cfg).unwrap();

    let base = [0.5f32, -1.25, 0.75, -0.5];
    let c = 4.0f32;
    let scaled: Vec<f32> = base.iter().map(|v| v * c).collect();

    let param_a = Tensor::new(base.to_vec(), &[4]).unwrap();
    let param_b = Tensor::new(scaled.clone(), &[4]).unwrap();
    let grad = Tensor::new(vec![0.3f32, -0.1, 0.05, 0.2], &[4]).unwrap();

    let out_a = opt_a.step(&[(&param_a, &grad)]).unwrap();
    let out_b = opt_b.step(&[(&param_b, &grad)]).unwrap();

    for i in 0..4 {
        let x_a = base[i];
        let x_b = scaled[i];
        let new_a = out_a[0].get(&[i]).unwrap();
        let new_b = out_b[0].get(&[i]).unwrap();
        let delta_a = x_a - new_a;
        let delta_b = x_b - new_b;
        let expected = delta_a * c;
        assert_eq!(
            delta_b.to_bits(),
            expected.to_bits(),
            "index={i}: delta_b={delta_b} ({:#010x}) expected={expected} ({:#010x}) \
             （2 の冪スケール不変性は bit 完全一致するはず）",
            delta_b.to_bits(),
            expected.to_bits(),
        );
    }
}

// =====================================================================
// 再現性・収束
// =====================================================================

/// 受け入れ条件「再現可能」の直接検証（`nn_optim_adamw.rs::
/// adamw_step_is_deterministic` と同型）: 同一入力で 2 回独立に
/// `Lamb::step` を 10 回呼び、結果がビット完全一致すること。
#[test]
fn lamb_step_is_deterministic() {
    fn run() -> Vec<f32> {
        let cfg = LambConfig {
            weight_decay: 0.05,
            ..LambConfig::default()
        };
        let mut opt = Lamb::new(cfg).unwrap();
        let mut param = Tensor::new(vec![1.0, -1.0, 0.5], &[3]).unwrap();
        for step in 0..10 {
            let grad = Tensor::new(vec![0.1 * step as f32, -0.2, 0.05], &[3]).unwrap();
            let out = opt.step(&[(&param, &grad)]).unwrap();
            param = out.into_iter().next().unwrap();
        }
        (0..3).map(|i| param.get(&[i]).unwrap()).collect()
    }

    let run1 = run();
    let run2 = run();
    assert_eq!(run1.len(), run2.len());
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "index={i}: 同一入力で Lamb::step の結果が一致しない"
        );
    }
}

/// LAMB が `Linear`+`Relu`+`MseLoss` の 2 層 MLP を収束させることを
/// 確認する（`nn_optim_adamw.rs::mlp_converges_with_adamw` と同型の
/// 収束テスト。trust ratio により実効ステップが縮むため、`AdamW` 用
/// lr（0.01）のままでは収束しないことを実測で確認したうえで lr を
/// 大きくしている）。
#[test]
fn mlp_converges_with_lamb() {
    const BATCH: usize = 4;
    const D_IN: usize = 8;
    const D_HIDDEN: usize = 16;
    const D_OUT: usize = 4;
    const STEPS: usize = 100;

    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x_data = Tensor::new(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]).unwrap();
    let y_data = Tensor::new(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]).unwrap();

    let relu = Relu;
    let mut l1 = Linear::new(D_IN, D_HIDDEN, true, 0x1111_1111).unwrap();
    let mut l2 = Linear::new(D_HIDDEN, D_OUT, true, 0x2222_2222).unwrap();

    let cfg = LambConfig {
        lr: 0.02,
        ..LambConfig::default()
    };
    let mut opt = Lamb::new(cfg).unwrap();

    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for _ in 0..STEPS {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let l1v = l1.bind(&tape);
        let l2v = l2.bind(&tape);

        let h1 = l1v.forward(&x).unwrap();
        let a1 = relu.forward(&h1);
        let h2 = l2v.forward(&a1).unwrap();
        let loss = h2.mse_loss(&y).unwrap();

        let loss_value = loss.to_tensor().get(&[]).unwrap();
        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }
        final_loss = loss_value;

        let grads = tape.backward(&loss).unwrap();
        let l1_weight_grad = grads.get(&l1v.weight).unwrap().unwrap();
        let l1_bias_grad = grads.get(l1v.bias.as_ref().unwrap()).unwrap().unwrap();
        let l2_weight_grad = grads.get(&l2v.weight).unwrap().unwrap();
        let l2_bias_grad = grads.get(l2v.bias.as_ref().unwrap()).unwrap().unwrap();

        let updated = opt
            .step(&[
                (l1.weight(), l1_weight_grad),
                (l1.bias().unwrap(), l1_bias_grad),
                (l2.weight(), l2_weight_grad),
                (l2.bias().unwrap(), l2_bias_grad),
            ])
            .unwrap();

        l1 = Linear::from_parameters(updated[0].clone(), Some(updated[1].clone())).unwrap();
        l2 = Linear::from_parameters(updated[2].clone(), Some(updated[3].clone())).unwrap();
    }

    let initial_loss = initial_loss.unwrap();
    assert!(
        final_loss < 0.5 * initial_loss,
        "LAMB での収束が不十分: initial={initial_loss} final={final_loss}"
    );
}
