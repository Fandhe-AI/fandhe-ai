//! イシュー #2197（親 #2172「LBFGS」）: `fandhe_ai_autodiff::nn::optim::
//! Lbfgs` の受け入れテスト。
//!
//! 受け入れ条件 5「PyTorch `torch.optim.LBFGS` 実行値 fixture と統一
//! 複合判定で一致」を、実 PyTorch 2.14.0+cpu 実行値
//! （`tests/fixtures/lbfgs-pytorch-reference/lbfgs_reference.json`。
//! README 参照）との突合で検証する。closure はホスト解析式
//! （最小二乗回帰の勾配）を f32 で直接計算し（fixture README §生成
//! 条件）、Tape 駆動 closure での検証は本ファイル末尾の別テストが担う。
//!
//! **契約: CI（self-hosted）は `docs/spec`（submodule）を checkout
//! しない**（`nn_optim_adamw.rs` 冒頭コメントと同じ制約）。本ファイルは
//! `tests/fixtures/lbfgs-pytorch-reference/`（本クレート配下に複製済み）
//! のみを参照する。
//!
//! 実機（CUDA/Metal）非依存・新規 `Op`/`BackendOps`/VJP 追加なしの
//! ホスト計算のため `#[ignore]` 分離は行わない。

mod common;

use std::fs;
use std::path::PathBuf;

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    #[allow(dead_code)]
    torch_version: String,
    cases: std::collections::BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
    init_w: Vec<f32>,
    init_b: f32,
    lr: f32,
    max_iter: usize,
    tolerance_grad: f32,
    tolerance_change: f32,
    history_size: usize,
    line_search_fn: Option<String>,
    steps: Vec<StepValues>,
}

#[derive(Deserialize)]
struct StepValues {
    w: Vec<f32>,
    b: Vec<f32>,
    func_evals: u64,
    n_iter: u64,
    orig_loss: f32,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lbfgs-pytorch-reference/lbfgs_reference.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    let fixture: Fixture = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("fixture のパースに失敗（JSON 構造が壊れている）: {e}"));
    // A03: 外部由来（テスト fixture とはいえ）データの要素数・shape を
    // 使う前に検証する（`nn_optim_rmsprop.rs::load_fixture` と同じ規律）。
    for (name, case) in &fixture.cases {
        let n = case.x.len();
        assert!(n > 0, "case {name}: x が空");
        let d = case.init_w.len();
        for (i, row) in case.x.iter().enumerate() {
            assert_eq!(row.len(), d, "case {name}: x[{i}] の列数が init_w と不一致");
        }
        assert_eq!(case.y.len(), n, "case {name}: y の長さが x の行数と不一致");
        for (i, step) in case.steps.iter().enumerate() {
            assert_eq!(step.w.len(), d, "case {name} step {i}: w の長さ不一致");
            assert_eq!(step.b.len(), 1, "case {name} step {i}: b の長さ不一致");
        }
    }
    fixture
}

/// 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
/// （`.claude/rules/coding-rust.md`）。`nn_optim_rmsprop.rs::assert_close`
/// と同じく `common::req2_close` へ委譲し閾値を再定義しない。
fn assert_close(actual: f32, expected: f32, context: &str) {
    assert!(
        common::req2_close(actual as f64, expected as f64),
        "{context}: actual={actual} expected={expected}"
    );
}

/// 目的関数 `loss = mean((X @ w + b - y)^2)` の closure（fixture
/// README「生成条件」と同じ解析式）。`Tensor<f32>` の入出力に対し
/// ホスト側 f32 演算で直接評価する（Tape 不使用）。
fn make_quadratic_closure(
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
) -> impl FnMut(&[Tensor<f32>]) -> (f32, Vec<Tensor<f32>>) {
    move |params: &[Tensor<f32>]| {
        let d = x[0].len();
        let n = x.len();
        let w: Vec<f32> = (0..d).map(|j| params[0].get(&[j]).unwrap()).collect();
        let b = params[1].get(&[0]).unwrap();

        let mut preds = Vec::with_capacity(n);
        for row in &x {
            let mut v = b;
            for j in 0..d {
                v += row[j] * w[j];
            }
            preds.push(v);
        }

        let mut loss = 0f32;
        for i in 0..n {
            let diff = preds[i] - y[i];
            loss += diff * diff;
        }
        loss /= n as f32;

        let mut grad_w = vec![0f32; d];
        let mut grad_b = 0f32;
        for i in 0..n {
            let diff = preds[i] - y[i];
            let coeff = 2.0 * diff / n as f32;
            for j in 0..d {
                grad_w[j] += coeff * x[i][j];
            }
            grad_b += coeff;
        }

        let grad_w_t = Tensor::new(grad_w, &[d]).unwrap();
        let grad_b_t = Tensor::new(vec![grad_b], &[1]).unwrap();
        (loss, vec![grad_w_t, grad_b_t])
    }
}

/// 受け入れ条件の本体: 全ケース・全 outer step の
/// `w`/`b`/`func_evals`/`n_iter`/`orig_loss` を PyTorch 実測値と突合
/// する。整数カウンタ（`func_evals`/`n_iter`）は分岐反転の検出器と
/// して完全一致で assert する（実装計画 §7）。
#[test]
fn lbfgs_matches_pytorch_reference() {
    let fixture = load_fixture();

    for (case_name, case) in &fixture.cases {
        let line_search = match case.line_search_fn.as_deref() {
            None => LbfgsLineSearch::None,
            Some("strong_wolfe") => LbfgsLineSearch::StrongWolfe,
            Some(other) => panic!("case {case_name}: 未知の line_search_fn: {other}"),
        };
        let cfg = LbfgsConfig {
            lr: case.lr,
            max_iter: case.max_iter,
            max_eval: None,
            tolerance_grad: case.tolerance_grad,
            tolerance_change: case.tolerance_change,
            history_size: case.history_size,
            line_search,
            line_search_steps: 1000, // fixture は line_search_steps >= max_eval を満たす設計（README 参照）
        };
        let mut opt =
            Lbfgs::new(cfg).unwrap_or_else(|e| panic!("case {case_name}: Lbfgs::new 失敗: {e}"));

        let d = case.init_w.len();
        let mut w = Tensor::new(case.init_w.clone(), &[d]).unwrap();
        let mut b = Tensor::new(vec![case.init_b], &[1]).unwrap();
        let mut closure = make_quadratic_closure(case.x.clone(), case.y.clone());

        for (step_idx, expected) in case.steps.iter().enumerate() {
            let updated = opt
                .step_closure(&[w.clone(), b.clone()], &mut closure)
                .unwrap_or_else(|e| panic!("case {case_name} step {step_idx}: step 失敗: {e}"));
            w = updated[0].clone();
            b = updated[1].clone();

            assert_eq!(
                opt.func_evals(),
                expected.func_evals,
                "case {case_name} step {step_idx}: func_evals 不一致（分岐反転の疑い）"
            );
            assert_eq!(
                opt.n_iter(),
                expected.n_iter,
                "case {case_name} step {step_idx}: n_iter 不一致（分岐反転の疑い）"
            );
            assert_close(
                opt.last_loss().unwrap(),
                expected.orig_loss,
                &format!("case={case_name} step={step_idx} orig_loss"),
            );
            for j in 0..d {
                assert_close(
                    w.get(&[j]).unwrap(),
                    expected.w[j],
                    &format!("case={case_name} step={step_idx} w[{j}]"),
                );
            }
            assert_close(
                b.get(&[0]).unwrap(),
                expected.b[0],
                &format!("case={case_name} step={step_idx} b"),
            );
        }
    }
}

/// 受け入れ条件「再現可能」: 同一入力で 2 回独立に `step_closure` を
/// 複数回呼び、結果が bit 完全一致すること（strong Wolfe・履歴 FIFO
/// 双方を経由させる）。
#[test]
fn lbfgs_step_is_deterministic() {
    fn run() -> Vec<f32> {
        let cfg = LbfgsConfig {
            line_search: LbfgsLineSearch::StrongWolfe,
            history_size: 3,
            max_iter: 5,
            ..LbfgsConfig::default()
        };
        let mut opt = Lbfgs::new(cfg).unwrap();
        let mut w = Tensor::new(vec![1.0, -1.0, 0.5], &[3]).unwrap();
        let mut closure = |p: &[Tensor<f32>]| {
            let v: Vec<f32> = (0..3).map(|i| p[0].get(&[i]).unwrap()).collect();
            let loss = v.iter().map(|x| x * x).sum::<f32>();
            let grad: Vec<f32> = v.iter().map(|x| 2.0 * x).collect();
            (loss, vec![Tensor::new(grad, &[3]).unwrap()])
        };
        for _ in 0..4 {
            let out = opt.step_closure(&[w.clone()], &mut closure).unwrap();
            w = out.into_iter().next().unwrap();
        }
        (0..3).map(|i| w.get(&[i]).unwrap()).collect()
    }

    let run1 = run();
    let run2 = run();
    assert_eq!(
        run1, run2,
        "同一入力で Lbfgs::step_closure の結果が一致しない"
    );
}

// =====================================================================
// Tape 駆動 closure（実運用形）での収束確認
// =====================================================================

/// `Linear` + `mse_loss` を closure 内部で毎回新しい `Tape` から構築し
/// （`nn_train_convergence.rs::run_regression_training` と同じ「毎
/// ステップ Tape を作り直す」運用パターンを、closure 内の「毎評価」に
/// 拡張したもの。L-BFGS の line search は 1 step の中で複数回評価する
/// ため、closure は呼ばれるたびに独立した `Tape` を構築し backward まで
/// 完結させる），strong Wolfe line search で線形回帰の loss が十分
/// 減少することを確認する（新規 tolerance は設けず「初期 loss から
/// 十分減少」判定を踏襲する。`nn_train_convergence.rs` と同型）。
#[test]
fn lbfgs_tape_driven_closure_converges() {
    use bench_harness::rng::Xorshift64Star;

    const BATCH: usize = 16;
    const D_IN: usize = 4;
    const D_OUT: usize = 1;
    const SEED_DATA: u64 = 0xFEED_5EED;
    const SEED_L: u64 = 0xABCD_1234;

    let mut rng = Xorshift64Star::new(SEED_DATA);
    let x_data = Tensor::new(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]).unwrap();
    let y_data = Tensor::new(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]).unwrap();

    let linear = Linear::new(D_IN, D_OUT, true, SEED_L).unwrap();
    let weight0 = linear.weight().clone();
    let bias0 = linear
        .bias()
        .expect("test fixture: bias=true で構築")
        .clone();

    let eval = |params: &[Tensor<f32>]| -> Result<(f32, Vec<Tensor<f32>>), AutodiffError> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let l = Linear::from_parameters(params[0].clone(), Some(params[1].clone()))?;
        let lv = l.bind(&tape);
        let pred = lv.forward(&x)?;
        let loss = pred.mse_loss(&y)?;
        let loss_value = loss.to_tensor().get(&[]).expect("mse_loss はスカラー");
        let grads = tape.backward(&loss)?;
        let w_grad = grads
            .get(&lv.weight)
            .unwrap()
            .expect("weight は requires_grad=true の葉")
            .clone();
        let b_grad = grads
            .get(lv.bias.as_ref().expect("test fixture: bias=true で構築"))
            .unwrap()
            .expect("bias は requires_grad=true の葉")
            .clone();
        Ok((loss_value, vec![w_grad, b_grad]))
    };

    let cfg = LbfgsConfig {
        line_search: LbfgsLineSearch::StrongWolfe,
        max_iter: 20,
        ..LbfgsConfig::default()
    };
    let mut opt = Lbfgs::new(cfg).unwrap();

    let initial_loss = eval(&[weight0.clone(), bias0.clone()]).unwrap().0;
    let updated = opt
        .try_step_closure(&[weight0, bias0], eval)
        .expect("Tape 駆動 closure での step が失敗した");

    let final_loss = eval(&updated).unwrap().0;

    assert!(
        final_loss < initial_loss * 0.5,
        "L-BFGS 1 step 後の loss が十分減少していない: initial={initial_loss} final={final_loss}"
    );
}
