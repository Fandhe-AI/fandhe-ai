//! `compile()`/`fit()` への L-BFGS 統合（イシュー #2172。2026-09-27
//! 所有者承認）の受け入れテスト。`fandhe_ai` のみを import する
//! **facade-only 契約**（`compat_sequential_lbfgs_manual.rs` とは対照的に
//! `fandhe_ai_autodiff` へ直接依存しない）。
//!
//! **line search**: #2502 で `LbfgsLineSearch` が facade 公開済みのため、
//! 学習曲線の比較は既定の固定ステップ（`LbfgsLineSearch::None`）で、
//! strong Wolfe 指定の `compile`/`fit` は縮小データの別テストで検証する。
//! 内部 import の手動ループ契約は `compat_sequential_lbfgs_manual.rs`。
//!
//! **受け入れ条件（親 #2172 コメント「残る受入条件」）**: fit（MNIST
//! 規模を模した合成回帰）での learning curve が SGD 相当同等以上の
//! 改善を示すことを検証する。`D_IN=784`（MNIST の実寸）・`D_OUT=10`
//! （MNIST のクラス数と同一）で次元を実寸に合わせ、CI 実行時間
//! （debug ビルドで数十秒以内目安。`.claude/rules/ci.md`
//! `test-timeout-minutes: 20`）を考慮してサンプル数・隠れ層次元・
//! epoch 数・`max_iter` を縮小する（`N=32`・`D_HIDDEN=16`・
//! `EPOCHS=3`・`max_iter=20`。実測所要時間は本ファイル冒頭コメント
//! 末尾および `docs/autodiff-lbfgs-decision.md` §9 参照）。
//!
//! **決定的シード**: 重み初期化（`add_linear` の `seed` 引数）・データ
//! 生成（`bench_harness::rng::Xorshift64Star`）は固定シードで駆動する
//! （`.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード
//! 設定ユーティリティを使う」）。両 optimizer は同一シードから独立に
//! 構築した同一初期重みのモデルで学習するため、初期条件は完全に揃う。
//!
//! **バッチサイズ（フルバッチ）**: `Lbfgs::try_step_closure` は 1 回の
//! `step`（= facade の 1 マイクロバッチ）内で `max_iter` 回までの内部
//! 反復を行う outer step 型の optimizer であり（`compat::training::
//! Optimizer::Lbfgs` doc 参照）、PyTorch の一般的な運用（フルバッチ・
//! `max_iter` を大きく取り `step` を epoch ごとに 1 回呼ぶ）に倣い
//! `batch_size = N`（1 epoch = 1 outer step）で検証する。
//!
//! 実機（CUDA/Metal）非依存・ホスト計算のみのため `#[ignore]` 分離は
//! 行わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::{LbfgsConfig, LbfgsLineSearch, SgdConfig};

const D_IN: usize = 784;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 10;
const N: usize = 32;

const SEED_DATA_X: u64 = 0xC0FF_EE60;
const SEED_DATA_Y: u64 = 0xC0FF_EE61;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// MNIST 規模を模した合成回帰データ（`compat_sequential_lbfgs_manual.rs::
/// gen_regression_data` と同型。分類ではなく回帰〈`Loss::Mse`〉にした
/// 理由は、line search なし固定ステップの L-BFGS でも安定して損失が
/// 減少する最小二乗様の滑らかな目的関数にするため——`compat::training::
/// Optimizer::Lbfgs` doc の「facade のみで使う場合の制約」節が示す
/// とおり、facade からは line search を選べない）。
fn gen_regression_data() -> (Tensor<f32>, Tensor<f32>) {
    let mut rng_x = Xorshift64Star::new(SEED_DATA_X);
    let mut rng_y = Xorshift64Star::new(SEED_DATA_Y);
    let x = rng_x.fill_vec(N * D_IN);
    let y = rng_y.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[N, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
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

const EPOCHS: usize = 3;

/// 受け入れ条件本体: 同一初期重み・同一データで L-BFGS（固定ステップ）
/// と SGD を同じ `epochs` 数だけ `fit` し、`evaluate` で測った最終損失を
/// 比較する。L-BFGS は 1 outer step（= 1 epoch。フルバッチ）あたり
/// `max_iter` 回までの内部反復を行うため、同じ `epochs` 数でも SGD より
/// 大きく損失が下がることが期待できる（L-BFGS の通常運用そのもの。
/// モジュール doc「バッチサイズ」節参照）。
#[test]
fn fit_lbfgs_learning_curve_matches_or_beats_sgd() {
    let (x_data, y_data) = gen_regression_data();

    let mut model_lbfgs = build_model();
    let mut model_sgd = build_model();

    // 初期重みが完全に一致することを前提とするため、ここで固定する
    // （`build_model` は決定的シードのみに依存するため本来自明だが、
    // 前提が崩れた場合に他の assert の意味を誤読しないよう明示検査する）。
    let init_lbfgs: Vec<Tensor<f32>> = model_lbfgs
        .trainable_parameters()
        .into_iter()
        .cloned()
        .collect();
    let init_sgd: Vec<Tensor<f32>> = model_sgd
        .trainable_parameters()
        .into_iter()
        .cloned()
        .collect();
    assert_eq!(
        init_lbfgs.len(),
        init_sgd.len(),
        "test fixture: 2 モデルの trainable_parameters 数が一致しない"
    );
    for (a, b) in init_lbfgs.iter().zip(init_sgd.iter()) {
        assert_eq!(
            a.contiguous().as_slice().unwrap(),
            b.contiguous().as_slice().unwrap(),
            "test fixture: 2 モデルの初期重みが bit 完全一致していない \
             （決定的シードの前提が崩れている）"
        );
    }

    model_lbfgs
        .compile(
            Optimizer::Lbfgs(LbfgsConfig {
                lr: 0.2,
                max_iter: 20,
                ..LbfgsConfig::default()
            }),
            Loss::Mse,
        )
        .unwrap_or_else(|e| panic!("Optimizer::Lbfgs の compile に失敗: {e}"));
    model_sgd
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap_or_else(|e| panic!("Optimizer::Sgd の compile に失敗: {e}"));

    let config = FitConfig::new(EPOCHS, N);

    let lbfgs_start = std::time::Instant::now();
    let history_lbfgs = model_lbfgs
        .fit(&x_data, &y_data, config)
        .unwrap_or_else(|e| panic!("Optimizer::Lbfgs での fit に失敗: {e}"));
    let lbfgs_elapsed = lbfgs_start.elapsed();

    let sgd_start = std::time::Instant::now();
    let history_sgd = model_sgd
        .fit(&x_data, &y_data, config)
        .unwrap_or_else(|e| panic!("Optimizer::Sgd での fit に失敗: {e}"));
    let sgd_elapsed = sgd_start.elapsed();

    let final_loss_lbfgs = model_lbfgs
        .evaluate(&x_data, &y_data, N)
        .unwrap_or_else(|e| panic!("Optimizer::Lbfgs での evaluate に失敗: {e}"));
    let final_loss_sgd = model_sgd
        .evaluate(&x_data, &y_data, N)
        .unwrap_or_else(|e| panic!("Optimizer::Sgd での evaluate に失敗: {e}"));

    // 実測値の可視化（`cargo test -- --nocapture` で確認可能。CI の
    // 合否判定には使わない）。
    println!(
        "lbfgs: history.loss={:?} final_eval_loss={final_loss_lbfgs} elapsed={lbfgs_elapsed:?}",
        history_lbfgs.loss
    );
    println!(
        "sgd:   history.loss={:?} final_eval_loss={final_loss_sgd} elapsed={sgd_elapsed:?}",
        history_sgd.loss
    );

    assert!(
        final_loss_lbfgs.is_finite(),
        "L-BFGS の最終損失が有限でない: {final_loss_lbfgs}"
    );
    assert!(
        final_loss_sgd.is_finite(),
        "SGD の最終損失が有限でない: {final_loss_sgd}"
    );
    assert_eq!(history_lbfgs.loss.len(), EPOCHS);
    assert_eq!(history_sgd.loss.len(), EPOCHS);

    // 受け入れ条件: L-BFGS の学習曲線が SGD 相当同等以上の改善を示す
    // （最終損失が SGD 以下であること）。
    assert!(
        final_loss_lbfgs <= final_loss_sgd,
        "L-BFGS（固定ステップ・{EPOCHS} epoch）の最終損失が SGD 相当を \
         上回っている（同等以上の改善という受け入れ条件を満たさない）: \
         lbfgs={final_loss_lbfgs} sgd={final_loss_sgd} \
         lbfgs_history={:?} sgd_history={:?}",
        history_lbfgs.loss,
        history_sgd.loss
    );
}

/// #2502: facade だけで `LbfgsLineSearch::StrongWolfe` を指定した
/// `compile`/`fit` が動き、損失が有限で初回より減少すること（縮小データ）。
#[test]
fn fit_lbfgs_strong_wolfe_via_facade_decreases_loss() {
    let (x_data, y_data) = gen_regression_data();
    let mut model = build_model();
    model
        .compile(
            Optimizer::Lbfgs(LbfgsConfig {
                lr: 1.0,
                max_iter: 10,
                line_search: LbfgsLineSearch::StrongWolfe,
                ..LbfgsConfig::default()
            }),
            Loss::Mse,
        )
        .unwrap_or_else(|e| panic!("StrongWolfe の compile に失敗: {e}"));
    let before = model
        .evaluate(&x_data, &y_data, N)
        .unwrap_or_else(|e| panic!("evaluate に失敗: {e}"));
    let history = model
        .fit(&x_data, &y_data, FitConfig::new(2, N))
        .unwrap_or_else(|e| panic!("StrongWolfe の fit に失敗: {e}"));
    let after = model
        .evaluate(&x_data, &y_data, N)
        .unwrap_or_else(|e| panic!("evaluate に失敗: {e}"));
    assert!(history.loss.iter().all(|l| l.is_finite()));
    assert!(
        after.is_finite() && after < before,
        "strong Wolfe の損失が減少していない: {before} -> {after}"
    );
}

// =====================================================================
// 非対応の組み合わせの fail-closed 拒否（イシュー #2172）
// =====================================================================

#[test]
fn compile_with_amp_rejects_lbfgs() {
    let mut model = build_model();
    let result = model.compile_with_amp(
        Optimizer::Lbfgs(LbfgsConfig::default()),
        Loss::Mse,
        fandhe_ai::compat::AmpConfig::new(fandhe_ai::compat::AmpDType::F16),
    );
    assert!(
        matches!(result, Err(fandhe_ai::AutodiffError::InvalidArgument(_))),
        "compile_with_amp は Optimizer::Lbfgs を fail-closed に拒否する \
         はず: {result:?}"
    );
    assert!(
        !model.is_compiled(),
        "compile_with_amp が拒否した後も is_compiled() が false のまま \
         （construct-before-assign）であるはず"
    );
}

// `accumulate_steps > 1` との併用拒否は、公開ビルダー化（#2508）に伴い
// `tests/compat_sequential_accumulate.rs::
// accumulate_steps_gt_one_rejected_with_lbfgs` に置く。カスタム学習 step
// フックとの併用拒否は、フック入口が引き続き `#[cfg(test)]` 限定・crate
// 内部専用のため外部統合テストクレートからは呼べず、
// `crates/facade/src/compat/training.rs::train_step_tests::
// custom_step_rejected_with_lbfgs` に置く。
