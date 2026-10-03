//! 勾配累積 `FitConfig::accumulate_steps`（イシュー #2180 で実装・#2508
//! で公開）の受け入れテスト。#2180 では公開ビルダーが未承認だったため
//! `src/compat/training.rs` の `accumulate_tests`（`#[cfg(test)]` 限定の
//! テスト専用セッター経由）に置いていた単体テスト T1〜T6 を、公開ビルダー
//! `FitConfig::accumulate_steps` 経由の facade 公開 API のみで書き直して
//! 本ファイルへ移した（テストの意味とアサーションは不変）。
//!
//! 検証観点: 既定 `1` の bit 一致（T1）・端数 flush を含む手動窓ループとの
//! bit 一致（T2。手動参照は公開 `SequentialVars::forward`・`Var::mse_loss`
//! を使う。`forward_with_precision(None)`／`mse_loss_with(Mean)` と演算列が
//! bit 同一）・大バッチ等価（T3。REQ-2 統一複合判定の既存定数を再利用。
//! tolerance は不変）・`0`／AMP／L-BFGS 併用の fail-closed 拒否（T4〜T6）。
//! 決定記録は `docs/compat-grad-accumulation-decision.md`。
//! ホスト計算のみで実機非依存（`#[ignore]` 不要）。

use fandhe_ai::compat::{AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::{LbfgsConfig, Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, Tensor};

const D_IN: usize = 3;
const D_HIDDEN: usize = 4;
const D_OUT: usize = 2;
const SEED_L1: u64 = 0xACC0_1111;
const SEED_L2: u64 = 0xACC0_2222;

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

/// 決定的な擬似乱数生成（splitmix64）。`bench_harness::rng` の RNG
/// 内部実装型は `crates/facade/tests/api_surface.rs::
/// facade_does_not_expose_rng_internal_types` が facade `src/` から
/// の参照を禁止するため、本テスト専用の局所実装で代替する。値域は
/// `(-0.5, 0.5)` に正規化する。
fn deterministic_fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

/// `n` サンプル分の回帰データ（決定的シード生成。値そのものは
/// `data_loader.rs::gen_dataset` とは異なる局所実装〈上記
/// [`deterministic_fill`] 参照〉だが、シード駆動の決定性という
/// 性質は同じ）。
fn gen_regression_data(seed: u64, n: usize) -> (Tensor<f32>, Tensor<f32>) {
    let x = deterministic_fill(seed, n * D_IN);
    let y = deterministic_fill(seed ^ 0x5555_5555_5555_5555, n * D_OUT);
    (
        Tensor::new(x, &[n, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[n, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let xd = x.host_slice();
        let yd = y.host_slice();
        if xd.len() != yd.len() {
            return false;
        }
        if xd
            .iter()
            .zip(yd.iter())
            .any(|(xv, yv)| xv.to_bits() != yv.to_bits())
        {
            return false;
        }
    }
    true
}

// =================================================================
// T1（R3）: accumulate_steps == 1 は既定の `fit` と bit 完全一致する。
// =================================================================
#[test]
fn accumulate_steps_one_matches_default_fit_bit_exact() {
    const N: usize = 8;
    let (x, y) = gen_regression_data(0xAAAA, N);

    let mut default_model = build_model();
    default_model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
    let default_history = default_model
        .fit(&x, &y, FitConfig::new(3, 2))
        .unwrap_or_else(|e| panic!("default fit に失敗: {e}"));

    let mut explicit_model = build_model();
    explicit_model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
    let explicit_config = FitConfig::new(3, 2).accumulate_steps(1);
    let explicit_history = explicit_model
        .fit(&x, &y, explicit_config)
        .unwrap_or_else(|e| panic!("accumulate_steps=1 fit に失敗: {e}"));

    assert_eq!(
        default_history.loss, explicit_history.loss,
        "accumulate_steps=1 の History.loss が既定の fit と bit 一致しない"
    );
    assert!(
        params_bit_exact(
            &default_model.trainable_parameters(),
            &explicit_model.trainable_parameters()
        ),
        "accumulate_steps=1 の学習後パラメータが既定の fit と bit 一致しない"
    );
}

// =================================================================
// T2（R2・端数 flush）: N=3・バッチ数 7（割り切れない構成）を、
// 独立に組んだ手動累積ループ（bind → forward → loss_for → backward
// → trainable_grads → f32 逐次和 → 境界 step → epoch 末 flush）と
// bit 完全一致で突き合わせる。
// =================================================================
#[test]
fn accumulate_steps_three_matches_manual_window_loop_bit_exact() {
    const TOTAL: usize = 7;
    const EPOCHS: usize = 2;
    const LR: f32 = 0.05;
    let (x, y) = gen_regression_data(0xBEEF, TOTAL);

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));
    let config = FitConfig::new(EPOCHS, 1).accumulate_steps(3);
    fit_model
        .fit(&x, &y, config)
        .unwrap_or_else(|e| panic!("accumulate_steps=3 fit に失敗: {e}"));

    // 手動参照ループ: `fit` と同じ演算列を独立に組む（`batch_size=1`
    // のため 1 マイクロバッチ = 1 サンプル）。
    let mut manual_model = build_model();
    let mut manual_optimizer = Sgd::new(SgdConfig::new(LR))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new に失敗: {e}"));
    let x_slice = x.host_slice();
    let y_slice = y.host_slice();
    for _epoch in 0..EPOCHS {
        let mut acc: Option<Vec<Tensor<f32>>> = None;
        let mut micro = 0u32;
        for i in 0..TOTAL {
            let x_row = Tensor::new(x_slice[i * D_IN..(i + 1) * D_IN].to_vec(), &[1, D_IN])
                .unwrap_or_else(|e| panic!("test fixture: x_row 構築に失敗: {e}"));
            let y_row = Tensor::new(y_slice[i * D_OUT..(i + 1) * D_OUT].to_vec(), &[1, D_OUT])
                .unwrap_or_else(|e| panic!("test fixture: y_row 構築に失敗: {e}"));

            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x_row);
            let pred = bound
                .forward(&tape, &x_var)
                .unwrap_or_else(|e| panic!("manual forward に失敗: {e}"));
            let target_var = tape.var_no_grad(&y_row);
            let loss_var = pred
                .mse_loss(&target_var)
                .unwrap_or_else(|e| panic!("manual mse_loss に失敗: {e}"));
            let grads = tape
                .backward(&loss_var)
                .unwrap_or_else(|e| panic!("manual backward に失敗: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("manual trainable_grads に失敗: {e}"));

            micro += 1;
            if micro == 1 {
                acc = Some(grad_refs.iter().map(|g| (*g).clone()).collect());
            } else {
                let acc_buf = acc
                    .as_mut()
                    .unwrap_or_else(|| panic!("test fixture: 累積バッファが初期化されていない"));
                for (a, g) in acc_buf.iter_mut().zip(grad_refs.iter()) {
                    let a_slice = a.host_slice();
                    let g_slice = g.host_slice();
                    let summed: Vec<f32> = a_slice
                        .iter()
                        .zip(g_slice.iter())
                        .map(|(u, v)| u + v)
                        .collect();
                    *a = Tensor::new(summed, a.shape())
                        .unwrap_or_else(|e| panic!("test fixture: 加算結果構築に失敗: {e}"));
                }
            }

            if micro == 3 || i == TOTAL - 1 {
                let param_refs = manual_model.trainable_parameters();
                let acc_buf = acc
                    .as_ref()
                    .unwrap_or_else(|| panic!("test fixture: 累積バッファが初期化されていない"));
                let acc_refs: Vec<&Tensor<f32>> = acc_buf.iter().collect();
                let updated = manual_optimizer
                    .step(&param_refs, &acc_refs)
                    .unwrap_or_else(|e| panic!("manual optimizer.step に失敗: {e}"));
                drop(param_refs);
                manual_model
                    .apply_parameters(updated)
                    .unwrap_or_else(|e| panic!("manual apply_parameters に失敗: {e}"));
                acc = None;
                micro = 0;
            }
        }
    }

    assert!(
        params_bit_exact(
            &fit_model.trainable_parameters(),
            &manual_model.trainable_parameters()
        ),
        "accumulate_steps=3 の fit が独立な手動累積ループと bit 一致しない"
    );
}

// =================================================================
// T3（大バッチ等価）: SGD（momentum なし）で、`batch_size=B・
// accumulate_steps=N・lr=lr0/N` の構成（累積側）と
// `batch_size=N*B・accumulate_steps=1・lr=lr0` の構成（大バッチ側）
// が同じ最終パラメータへ収束することを REQ-2 の統一複合判定
// （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
// `fandhe_ai_backend_cpu::parity::assert_parity` が使う既存定数
// `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` をそのまま
// 再利用するだけで、新設・緩和ではない）で確認する。
//
// 数学的根拠: `Reduction::Mean` の MSE は 1 マイクロバッチ
// （サイズ B）の勾配平均 `(1/B)Σ_batch g` を計算する。累積側は
// これを N 回（重複なく `N*B` 件全体を分割）加算するため、境界での
// 累積勾配は `(1/B)Σ_all g`。大バッチ側は 1 バッチ（サイズ N*B）の
// 平均 `(1/(N*B))Σ_all g`。よって累積側の更新量 `(lr0/N)・(1/B)Σ_all
// g = (lr0/(N*B))Σ_all g` と、大バッチ側の更新量 `lr0・(1/(N*B))Σ_all
// g` は数学的に厳密に一致する（累積・大バッチいずれも和を取る順序が
// 異なるだけで、浮動小数の丸め順序差のみが REQ-2 判定対象）。
// =================================================================
#[test]
fn accumulate_matches_large_batch_equivalent_within_req2_tolerance() {
    const B: usize = 2;
    const N: u32 = 4;
    const TOTAL: usize = B * N as usize;
    const EPOCHS: usize = 3;
    const LR0: f32 = 0.2;

    let (x, y) = gen_regression_data(0xF00D, TOTAL);

    let mut accum_model = build_model();
    accum_model
        .compile(Optimizer::Sgd(SgdConfig::new(LR0 / N as f32)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile（累積側）に失敗: {e}"));
    let accum_config = FitConfig::new(EPOCHS, B)
        .drop_last(true)
        .accumulate_steps(N);
    accum_model
        .fit(&x, &y, accum_config)
        .unwrap_or_else(|e| panic!("累積側 fit に失敗: {e}"));

    let mut large_batch_model = build_model();
    large_batch_model
        .compile(Optimizer::Sgd(SgdConfig::new(LR0)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile（大バッチ側）に失敗: {e}"));
    let large_batch_config = FitConfig::new(EPOCHS, TOTAL).drop_last(true);
    large_batch_model
        .fit(&x, &y, large_batch_config)
        .unwrap_or_else(|e| panic!("大バッチ側 fit に失敗: {e}"));

    let accum_params = accum_model.trainable_parameters();
    let large_batch_params = large_batch_model.trainable_parameters();
    assert_eq!(accum_params.len(), large_batch_params.len());
    for (a, b) in accum_params.iter().zip(large_batch_params.iter()) {
        let a_slice = a.host_slice();
        let b_slice = b.host_slice();
        fandhe_ai_backend_cpu::parity::assert_parity(
            "accumulate_matches_large_batch_equivalent_within_req2_tolerance",
            &a_slice,
            &b_slice,
        );
    }
}

// =================================================================
// T4: accumulate_steps == 0 は InvalidArgument（fail-closed）。
// compiled 状態・train/eval モードは変更前のまま維持される。
// =================================================================
#[test]
fn accumulate_steps_zero_is_rejected() {
    let (x, y) = gen_regression_data(0xCCCC, 4);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));

    let prev_training = model.training();
    let config = FitConfig::new(1, 4).accumulate_steps(0);
    let err = model
        .fit(&x, &y, config)
        .expect_err("accumulate_steps == 0 は Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(
        model.is_compiled(),
        "エラー後も compiled 状態が維持されるはず"
    );
    assert_eq!(
        model.training(),
        prev_training,
        "エラー後は呼び出し前の train/eval モードへ復元されるはず"
    );
}

// =================================================================
// T5（AMP 併用拒否）: accumulate_steps > 1 は compile_with_amp と
// 併用できない（実装計画 §3.4「代替」方式。fail-closed）。
// =================================================================
#[test]
fn accumulate_steps_gt_one_rejected_with_amp() {
    let (x, y) = gen_regression_data(0xDDDD, 4);
    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.1)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp に失敗: {e}"));

    let config = FitConfig::new(1, 2).accumulate_steps(2);
    let err = model
        .fit(&x, &y, config)
        .expect_err("accumulate_steps > 1 と AMP の併用は Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(
        model.is_compiled(),
        "エラー後も compiled 状態が維持されるはず"
    );
}

// =================================================================
// T6（L-BFGS 併用拒否。イシュー #2172）: accumulate_steps > 1 は
// Optimizer::Lbfgs とも併用できない（`accumulate_steps_gt_one_rejected_with_amp`
// と同型。#2508 で compat_sequential_fit_lbfgs.rs の旧コメントから移設）。
// =================================================================
#[test]
fn accumulate_steps_gt_one_rejected_with_lbfgs() {
    let (x, y) = gen_regression_data(0x1BF65, 4);
    let mut model = build_model();
    model
        .compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile に失敗: {e}"));

    let config = FitConfig::new(1, 2).accumulate_steps(2);
    let err = model
        .fit(&x, &y, config)
        .expect_err("accumulate_steps > 1 と Optimizer::Lbfgs の併用は Err のはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(
        model.is_compiled(),
        "エラー後も compiled 状態が維持されるはず"
    );
}
