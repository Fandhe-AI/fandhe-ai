//! イシュー #2197（親 #2172「LBFGS」）の受け入れ範囲確認: facade と
//! 同じ ops 構成（composition root・既定 CPU バックエンド
//! `fandhe_ai_backend_cpu::CpuBackendOps`）で組んだ closure を
//! `fandhe_ai_autodiff::nn::optim::Lbfgs` に渡した学習ループが線形
//! 回帰で収束することを確認する統合テスト（`facade` クレート配下の
//! 統合テスト。`nn::Linear::bind` は `&fandhe_ai_autodiff::Tape` を
//! 要求するため `fandhe_ai::tape()`〈`fandhe_ai::Tape` newtype〉は
//! 使わない。理由は下記 `eval` 直前のコメント参照）。
//!
//! **facade 公開面は本イシューでは拡張しない**（`fandhe_ai::optim` に
//! `Lbfgs` は存在しない）。`Lbfgs` を内部クレート
//! `fandhe_ai_autodiff` から直接 import する構成は
//! `compat_sequential_train.rs`（`fandhe_ai_autodiff::optim::Sgd`・
//! `fandhe_ai_autodiff::nn::optim::AdamW` を直接 import）と同型の先例
//! である。facade（`fandhe_ai::optim`）への `Lbfgs` 公開・`compile()`
//! の `Optimizer::Lbfgs` variant 追加は別イシュー #2198（facade 公開面
//! 拡張はユーザー承認事項）の担当範囲であり、本ファイルはそれを
//! 含まない。
//!
//! **数値判定の規律**: 収束判定は既存様式（最終 loss が初期 loss から
//! 十分減少すること）を踏襲し、新規の許容誤差は設けない
//! （`crates/autodiff/tests/nn_train_convergence.rs` と同型）。
//!
//! 実機（CUDA/Metal）非依存・ホスト計算のみのため `#[ignore]` 分離は
//! 行わない。

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 16;
const D_IN: usize = 4;
const D_OUT: usize = 1;
const SEED_L: u64 = 0x5EED_C0DE;

fn xorshift_fill(seed: u64, n: usize) -> Vec<f32> {
    // facade テストは `bench-harness` を dev-dependency に持たない
    // ため（`Cargo.toml` の追加はスコープ外）、決定的シード生成のみ
    // ローカルに再実装する（`nn_train_convergence.rs` の
    // `bench_harness::rng::Xorshift64Star` と同型の xorshift64*。
    // `[-1, 1)` の一様分布に写像する）。
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let unit = (bits >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        out.push((unit * 2.0 - 1.0) as f32);
    }
    out
}

#[test]
fn lbfgs_closure_over_facade_tape_converges() {
    // `y` を `x` と無関係な乱数にすると（線形回帰では説明できない
    // ノイズのみのため）loss がほぼ下がらない degenerate なケースに
    // なる。`y = X @ true_w + true_b + 小さいノイズ` という実際に
    // 線形回帰で説明可能なデータにする（fixture 生成条件
    // `lbfgs-pytorch-reference/README.md`「生成条件」と同じ方針）。
    let x_flat = xorshift_fill(0xC0FF_EE01, BATCH * D_IN);
    let true_w = xorshift_fill(0xC0FF_EE03, D_IN);
    let true_b = xorshift_fill(0xC0FF_EE04, 1)[0];
    let noise = xorshift_fill(0xC0FF_EE02, BATCH);
    let y_flat: Vec<f32> = (0..BATCH)
        .map(|i| {
            let row = &x_flat[i * D_IN..(i + 1) * D_IN];
            let dot: f32 = row.iter().zip(&true_w).map(|(a, b)| a * b).sum();
            dot + true_b + 0.05 * noise[i]
        })
        .collect();
    let x_data = Tensor::new(x_flat, &[BATCH, D_IN]).unwrap();
    let y_data = Tensor::new(y_flat, &[BATCH, D_OUT]).unwrap();

    let linear = Linear::new(D_IN, D_OUT, true, SEED_L).unwrap();
    let weight0 = linear.weight().clone();
    let bias0 = linear
        .bias()
        .expect("test fixture: bias=true で構築")
        .clone();

    // closure は呼ばれるたびに独立した `Tape` を構築し forward →
    // backward を完結させる（line search が 1 step 内で複数回評価する
    // ため）。`Linear::bind`／`Module::forward` は
    // `&fandhe_ai_autodiff::Tape` を要求するため（`compat::Sequential`
    // の公開シグネチャとは異なる。`nn::Linear` を直接使う手動ループの
    // ため）、facade の `fandhe_ai::tape()`（`fandhe_ai::Tape` newtype。
    // 内部 `fandhe_ai_autodiff::Tape` フィールドは `pub(crate)` で
    // crate 外に非公開）ではなく、`compat_sequential_train.rs::
    // manual_sgd_step` と同型の「facade と同じ ops 構成（既定
    // `CpuBackendOps`）の生 `fandhe_ai_autodiff::Tape`」を使う。
    let eval = |params: &[Tensor<f32>]| -> Result<(f32, Vec<Tensor<f32>>), AutodiffError> {
        let tape = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(
            fandhe_ai_backend_cpu::CpuBackendOps::new(),
        ));
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
    // strong Wolfe は 1 outer step 内で `max_iter` 回まで反復するが、
    // 停止判定（`opt_cond`/`|loss - prev_loss| < tolerance_change` 等）
    // により 1 step 目で収束末期に達しないことがあるため、複数 outer
    // step 呼び出しで確実な収束を確認する（`nn_train_convergence.rs`
    // の複数 step ループと同型。新規 tolerance は設けない）。
    let mut params = vec![weight0, bias0];
    for _ in 0..5 {
        params = opt
            .try_step_closure(&params, eval)
            .expect("facade Tape 駆動 closure での step が失敗した");
    }
    let final_loss = eval(&params).unwrap().0;

    assert!(
        final_loss < initial_loss * 0.5,
        "L-BFGS 1 step 後の loss が十分減少していない: initial={initial_loss} final={final_loss}"
    );
}
