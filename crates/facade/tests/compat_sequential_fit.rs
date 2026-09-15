//! Keras 風 `compile()`／`fit()`／`evaluate()`（イシュー #1761・親
//! #1618）の統合テスト。`fandhe_ai` のみを import する（`compat_sequential_
//! train.rs` と異なり、本ファイルの手動ループ比較対象も `fandhe_ai::tape`／
//! `fandhe_ai::compat::Sequential` の既存メソッドだけで組み立てる——
//! `fit` が手動ループと**同一の演算列**であることを bit 完全一致で
//! 検証する狙いのため、比較対象自体も facade 経由で統一する）。
//!
//! **決定的シード**: 重み初期化（`add_linear` の `seed` 引数）・データ
//! 生成（`bench_harness::rng::Xorshift64Star`）は固定シードで駆動する
//! （`.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード
//! 設定ユーティリティを使う」）。`shuffle(true)` 系・Dropout train
//! モード系のテストはグローバル RNG（`fandhe_ai::manual_seed`）を消費
//! するため、ファイル局所 `Mutex` で直列化する（`data_loader.rs` と
//! 同型）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::{AdamWConfig, Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, Tensor};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

const N: usize = 16;
const D_IN: usize = 4;
const D_HIDDEN: usize = 8;
const D_OUT: usize = 2;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// 回帰用データ（`data_loader.rs::gen_dataset` と同型の決定的生成）。
fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

/// 分類用データ（logits `[N, C]`・ラベル `Tensor<i32>` `[N]`。
/// `t = idx % C` で決定的に割り当てる）。
fn gen_classification_data(seed: u64, classes: usize) -> (Tensor<f32>, Tensor<i32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(N * D_IN);
    let y: Vec<i32> = (0..N).map(|i| (i % classes) as i32).collect();
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N]).unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn params_bit_exact(a: &[&Tensor<f32>], b: &[&Tensor<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let xd = x
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        let yd = y
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous 化済み")
            .to_vec();
        if xd.len() != yd.len() {
            return false;
        }
        for (xv, yv) in xd.iter().zip(yd.iter()) {
            if xv.to_bits() != yv.to_bits() {
                return false;
            }
        }
    }
    true
}

// =====================================================================
// 1. Sgd/Mse で収束する（新設の収束判定。既存 tolerance の緩和ではない）
// =====================================================================

#[test]
fn fit_sgd_mse_converges() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.3)), Loss::Mse)
        .unwrap();

    let history = model.fit(&x, &y, FitConfig::new(50, N)).unwrap();
    assert_eq!(history.loss.len(), 50);
    let first = history.loss[0];
    let last = *history.loss.last().unwrap();
    assert!(
        last < 0.5 * first,
        "loss did not converge enough: first={first} last={last}"
    );
}

// =====================================================================
// 2. fit（shuffle=false・batch_size=N）は手動ループと bit 完全一致
// =====================================================================

#[test]
fn fit_matches_manual_loop_bit_exact() {
    const STEPS: usize = 5;
    const LR: f32 = 0.05;
    let (x, y) = gen_regression_data(SEED_DATA);

    // 手動ループ（`fit` と同一の演算列: bind → forward → mse_loss_with
    // 〈target は var_no_grad〉→ backward → trainable_grads → Sgd::step
    // → apply_parameters）。
    let mut manual_model = build_model();
    let mut manual_sgd = Sgd::new(SgdConfig::new(LR)).unwrap();
    let mut manual_losses = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = manual_model.bind(&tape);
            let x_var = tape.var(&x);
            let y_var = tape.var_no_grad(&y);

            let pred = bound.forward(&tape, &x_var).unwrap();
            let loss = pred.mse_loss(&y_var).unwrap();
            manual_losses.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = manual_model.trainable_parameters();
            manual_sgd.step(&param_refs, &grad_refs).unwrap()
        };
        manual_model.apply_parameters(updated).unwrap();
    }

    let mut fit_model = build_model();
    fit_model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap();
    let history = fit_model.fit(&x, &y, FitConfig::new(STEPS, N)).unwrap();

    assert_eq!(history.loss.len(), manual_losses.len());
    for (m, f) in manual_losses.iter().zip(history.loss.iter()) {
        assert_eq!(
            m.to_bits(),
            f.to_bits(),
            "loss diverged: manual={m} fit={f}"
        );
    }

    let manual_params = manual_model.trainable_parameters();
    let fit_params = fit_model.trainable_parameters();
    assert!(
        params_bit_exact(&manual_params, &fit_params),
        "final parameters diverged between manual loop and fit()"
    );
}

// =====================================================================
// 3. fit(1) を 2 回 == fit(2)（optimizer 状態が呼び出しをまたいで継続）
// =====================================================================

#[test]
fn fit_twice_equals_fit_once_with_double_epochs() {
    let (x, y) = gen_regression_data(SEED_DATA);

    let mut model_twice = build_model();
    model_twice
        .compile(Optimizer::AdamW(AdamWConfig::default()), Loss::Mse)
        .unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    model_twice.fit(&x, &y, FitConfig::new(1, N)).unwrap();

    let mut model_once = build_model();
    model_once
        .compile(Optimizer::AdamW(AdamWConfig::default()), Loss::Mse)
        .unwrap();
    model_once.fit(&x, &y, FitConfig::new(2, N)).unwrap();

    let params_twice = model_twice.trainable_parameters();
    let params_once = model_once.trainable_parameters();
    assert!(
        params_bit_exact(&params_twice, &params_once),
        "fit(1)+fit(1) diverged from fit(2)"
    );
}

// =====================================================================
// 4. shuffle(true) は manual_seed 固定で再現可能
// =====================================================================

#[test]
fn fit_minibatch_shuffle_is_reproducible_under_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (x, y) = gen_regression_data(SEED_DATA);

    let run = || {
        fandhe_ai::manual_seed(42);
        let mut model = build_model();
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
            .unwrap();
        let history = model
            .fit(&x, &y, FitConfig::new(3, N / 4).shuffle(true))
            .unwrap();
        (history, model)
    };

    let (history_a, model_a) = run();
    let (history_b, model_b) = run();

    assert_eq!(history_a.loss.len(), history_b.loss.len());
    for (a, b) in history_a.loss.iter().zip(history_b.loss.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let params_a = model_a.trainable_parameters();
    let params_b = model_b.trainable_parameters();
    assert!(params_bit_exact(&params_a, &params_b));
}

// =====================================================================
// 5. cross_entropy_loss（Tensor<i32> target）が収束方向
// =====================================================================

#[test]
fn fit_cross_entropy_with_i32_targets_converges() {
    const CLASSES: usize = 2;
    let (x, y) = gen_classification_data(SEED_DATA, CLASSES);
    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, CLASSES, SEED_L2)
        .unwrap();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();

    let history = model.fit(&x, &y, FitConfig::new(20, N)).unwrap();
    assert_eq!(history.loss.len(), 20);
    let first = history.loss[0];
    let last = *history.loss.last().unwrap();
    assert!(
        last <= first,
        "cross entropy loss did not move toward convergence: first={first} last={last}"
    );
}

// =====================================================================
// 6. loss × target dtype の不整合は拒否される
// =====================================================================

#[test]
fn fit_rejects_loss_target_dtype_mismatch() {
    let (x, y_f32) = gen_regression_data(SEED_DATA);
    let (_, y_i32) = gen_classification_data(SEED_DATA, 2);

    let mut model_ce = build_model();
    model_ce
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let err = model_ce.fit(&x, &y_f32, FitConfig::new(1, N)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));

    let mut model_mse = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, 2, SEED_L2)
        .unwrap();
    model_mse
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();
    let err = model_mse.fit(&x, &y_i32, FitConfig::new(1, N)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// =====================================================================
// 7. 未 compile／不正な引数の拒否（Err 後も compile 済み状態は維持）
// =====================================================================

#[test]
fn fit_and_evaluate_reject_uncompiled_model() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    assert!(!model.is_compiled());

    let err = model.fit(&x, &y, FitConfig::new(1, N)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));

    let err = model.evaluate(&x, &y, N).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn fit_rejects_zero_epochs() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let err = model.fit(&x, &y, FitConfig::new(0, N)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(
        model.is_compiled(),
        "Err 後も compile 済み状態が維持されること"
    );
}

#[test]
fn fit_rejects_huge_epochs_without_panicking() {
    // codex-review 指摘（PR #1877・イシュー #1761）: `FitConfig::new(usize::MAX, ..)`
    // のような巨大な `epochs` を渡すと、`History.loss` 用の
    // `Vec::with_capacity` が capacity overflow で panic していた
    // （本番経路の panic 禁止。`.claude/rules/security.md` A03 の精神）。
    // `try_reserve_exact` への切替後は panic せず型付きエラーを返し、
    // 呼び出し元の復元経路（compile 済み状態の維持）も機能することを
    // 確認する。
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let err = model
        .fit(&x, &y, FitConfig::new(usize::MAX, N))
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(
        model.is_compiled(),
        "Err 後も compile 済み状態が維持されること"
    );
}

#[test]
fn fit_rejects_sample_count_mismatch() {
    let (x, _) = gen_regression_data(SEED_DATA);
    let y_wrong = Tensor::<f32>::zeros(&[N - 1, D_OUT]).unwrap();
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let err = model.fit(&x, &y_wrong, FitConfig::new(1, N)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

#[test]
fn fit_rejects_zero_batch_size() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let err = model.fit(&x, &y, FitConfig::new(1, 0)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

#[test]
fn evaluate_rejects_zero_batch_size() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let err = model.evaluate(&x, &y, 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert!(model.is_compiled());
}

// =====================================================================
// 8. evaluate: 単一バッチは直接計算と bit 一致・分割時は重み付き平均
// =====================================================================

#[test]
fn evaluate_full_batch_matches_direct_loss_bit_exact() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let evaluated = model.evaluate(&x, &y, N).unwrap();

    let direct = {
        let tape = fandhe_ai::tape();
        let x_var = tape.var(&x);
        let y_var = tape.var_no_grad(&y);
        let pred = model.forward(&tape, &x_var).unwrap();
        let loss = pred.mse_loss(&y_var).unwrap();
        scalar(&loss.to_tensor())
    };

    assert_eq!(evaluated.to_bits(), direct.to_bits());
}

#[test]
fn evaluate_minibatch_matches_weighted_mean_of_manual_batches() {
    let (x, y) = gen_regression_data(SEED_DATA);
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .unwrap();

    let batch_size = N / 4;
    let evaluated = model.evaluate(&x, &y, batch_size).unwrap() as f64;

    // 手動でバッチ分割した重み付き平均（`evaluate` の集計方式と同型）。
    let mut weighted_sum = 0.0f64;
    let mut count = 0usize;
    let x_data = x.contiguous().as_slice().unwrap().to_vec();
    let y_data = y.contiguous().as_slice().unwrap().to_vec();
    for chunk_start in (0..N).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(N);
        let n_batch = chunk_end - chunk_start;
        let x_chunk = Tensor::new(
            x_data[chunk_start * D_IN..chunk_end * D_IN].to_vec(),
            &[n_batch, D_IN],
        )
        .unwrap();
        let y_chunk = Tensor::new(
            y_data[chunk_start * D_OUT..chunk_end * D_OUT].to_vec(),
            &[n_batch, D_OUT],
        )
        .unwrap();
        let tape = fandhe_ai::tape();
        let x_var = tape.var(&x_chunk);
        let y_var = tape.var_no_grad(&y_chunk);
        let pred = model.forward(&tape, &x_var).unwrap();
        let loss = pred.mse_loss(&y_var).unwrap();
        weighted_sum += scalar(&loss.to_tensor()) as f64 * n_batch as f64;
        count += n_batch;
    }
    let expected = (weighted_sum / count as f64) as f32;

    assert_eq!(evaluated as f32, expected);
    let _ = evaluated;
}

// =====================================================================
// 9. train/eval モードの復元・Dropout 決定性
// =====================================================================

#[test]
fn fit_restores_training_mode_and_evaluate_is_deterministic_with_dropout() {
    let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let (x, y) = gen_regression_data(SEED_DATA);

    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_dropout(0.5)
        .unwrap()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();

    // eval() で開始 -> fit 後も eval のまま復元されること。
    model.eval();
    assert!(!model.training());
    fandhe_ai::manual_seed(7);
    model.fit(&x, &y, FitConfig::new(2, N)).unwrap();
    assert!(
        !model.training(),
        "fit 後は呼び出し前のモード（eval）へ復元されること"
    );

    // eval モードでの evaluate は Dropout マスク非依存のため決定的。
    let e1 = model.evaluate(&x, &y, N).unwrap();
    let e2 = model.evaluate(&x, &y, N).unwrap();
    assert_eq!(e1.to_bits(), e2.to_bits());
    assert!(
        !model.training(),
        "evaluate 後も呼び出し前のモード（eval）へ復元されること"
    );

    // train() で開始 -> fit 後は train のまま復元されること。
    model.train();
    assert!(model.training());
    fandhe_ai::manual_seed(8);
    model.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    assert!(
        model.training(),
        "fit 後は呼び出し前のモード（train）へ復元されること"
    );
}

// =====================================================================
// 10. re-compile は optimizer 状態をリセットする
// =====================================================================

#[test]
fn recompile_replaces_optimizer_state() {
    // momentum 有り（`0.0` では optimizer 自体が状態を持たず reset の
    // 有無を区別できないため）。
    const LR: f32 = 0.05;
    const MOMENTUM: f32 = 0.9;
    let sgd_config = SgdConfig::new(LR).with_momentum(MOMENTUM);

    let (x, y) = gen_regression_data(SEED_DATA);

    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(sgd_config), Loss::Mse)
        .unwrap();
    // 1 回目の fit で momentum バッファへ velocity（1 回目は `velocity
    // = grad`）が積まれる。
    model.fit(&x, &y, FitConfig::new(1, N)).unwrap();

    // 再 compile（同一構成）: optimizer 状態（velocity）を破棄して
    // 新しい `Sgd` へ置き換える契約（`Sequential::compile` doc 参照）。
    model
        .compile(Optimizer::Sgd(sgd_config), Loss::Mse)
        .unwrap();

    // 再 compile 直後の現在パラメータを控えておき、`fit(1)` の効果を
    // 「velocity=None から始まる素の `Sgd::step` 1 回」と突き合わせる
    // （momentum バッファがリセットされていなければ、1 回目の fit で
    // 積まれた velocity が残ってしまい両者は一致しなくなる）。
    let params_before: Vec<Tensor<f32>> =
        model.trainable_parameters().into_iter().cloned().collect();

    model.fit(&x, &y, FitConfig::new(1, N)).unwrap();
    let fit_params = model.trainable_parameters();

    // 手動側: 同じ現在パラメータから、velocity=None の新規 `Sgd` で
    // 1 回だけ `bind → forward → mse_loss → backward → trainable_grads
    // → Sgd::step → apply_parameters` を実行する（`fit` の内部演算列と
    // 同一。§3.5 参照）。
    let mut manual_model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap();
    manual_model
        .apply_parameters(params_before)
        .expect("test fixture: 直前の shape と一致するため成功するはず");
    let mut fresh_sgd = Sgd::new(sgd_config).unwrap();
    let manual_updated = {
        let tape = fandhe_ai::tape();
        let bound = manual_model.bind(&tape);
        let x_var = tape.var(&x);
        let y_var = tape.var_no_grad(&y);
        let pred = bound.forward(&tape, &x_var).unwrap();
        let loss = pred.mse_loss(&y_var).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let grad_refs = bound.trainable_grads(&grads).unwrap();
        let param_refs = manual_model.trainable_parameters();
        fresh_sgd.step(&param_refs, &grad_refs).unwrap()
    };
    manual_model.apply_parameters(manual_updated).unwrap();
    let manual_params = manual_model.trainable_parameters();

    assert!(
        params_bit_exact(&fit_params, &manual_params),
        "再 compile 後の fit(1) が velocity=None の素の Sgd::step と一致しない\
         （optimizer 状態がリセットされていない可能性がある）"
    );
}
