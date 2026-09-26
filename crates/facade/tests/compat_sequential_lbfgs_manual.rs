//! イシュー #2198（親 #2172「LBFGS」・ルート #2131）の facade 公開保留
//! （`crates/facade/src/lib.rs::LbfgsHoldDoctestGuard`）下での受け入れ
//! 条件の部分的な裏付け: 公開 API の `compat::Sequential`（`bind`／
//! `forward`／`trainable_parameters`／`trainable_grads`／
//! `apply_parameters`）と、`fandhe_ai_autodiff::nn::optim::{Lbfgs,
//! LbfgsConfig, LbfgsLineSearch}`（**内部 import**。facade
//! 再エクスポートは未承認のため保留中）で組んだ手動 closure ループを
//! 検証する。
//!
//! **本ファイルは `fandhe_ai_autodiff::nn::optim::Lbfgs` を直接 import
//! する契約ファイル**であり（`compat_sequential_optim_ext.rs` と同型の
//! 位置づけ）、facade 再エクスポートのみを使う契約のテストファイルへ
//! 混入させない。承認後（`docs/autodiff-lbfgs-decision.md` §8）に
//! `compile()`／`fit()` 統合を実装する際は、facade 再エクスポート版の
//! 別ファイル（`compat_sequential_fit_lbfgs.rs` 想定）を新設する。
//!
//! **数値判定の規律**: 収束判定は既存様式（最終 loss が初期 loss から
//! 十分減少すること）を踏襲し、新規の許容誤差は設けない
//! （`optim_lbfgs_closure.rs`・`crates/autodiff/tests/
//! nn_train_convergence.rs` と同型）。PyTorch との数値一致は #2197 の
//! fixture parity（`crates/autodiff/tests/nn_optim_lbfgs.rs::
//! lbfgs_matches_pytorch_reference`）が既に担保済みであり、本ファイルで
//! Python/PyTorch を実行することはない。
//!
//! 実機（CUDA/Metal）非依存・ホスト計算のみのため `#[ignore]` 分離は
//! 行わない。

use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::{AdamW, AdamWConfig, Lbfgs, LbfgsConfig, LbfgsLineSearch};
use fandhe_ai_tensor_core::Tensor;

const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

/// `optim_lbfgs_closure.rs::xorshift_fill` と同型の決定的シード生成
/// （facade テストは `bench-harness` を dev-dependency に持たないため
/// ローカルに再実装する）。`[-1, 1)` の一様分布に写像する。
fn xorshift_fill(seed: u64, n: usize) -> Vec<f32> {
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

fn build_model(d_in: usize, d_hidden: usize, d_out: usize) -> Sequential {
    Sequential::new()
        .add_linear(d_in, d_hidden, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(d_hidden, d_out, SEED_L2)
        .unwrap()
}

/// `model` の現在のパラメータで `x_data`／`y_data` に対する MSE loss を
/// 評価する（forward のみ・`Tape` は使い捨て）。`lbfgs_step_on_sequential`
/// が返す `opt.last_loss()` は L-BFGS の**その step の初回評価損失**
/// （`orig_loss`。line search 開始前・更新前のパラメータでの損失）で
/// あり、outer step 内の line search 更新後の損失ではない
/// （`crates/autodiff/src/nn/optim/lbfgs.rs` の `try_step_closure` は
/// `last_loss` を `orig_loss` にのみ設定する）。そのため収束判定で
/// 「最後の outer step 後」の loss を見たい場合は、最後の
/// `apply_parameters` 後に本関数で改めて評価しなければ、最後の step
/// 内での改善（または悪化）を見逃す（codex-review 指摘・PR #2300
/// discussion）。
fn eval_loss(model: &Sequential, x_data: &Tensor<f32>, y_data: &Tensor<f32>) -> f32 {
    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let x = tape.var(x_data);
    let y = tape.var(y_data);
    let pred = bound.forward(&tape, &x).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    scalar(&loss.to_tensor())
}

fn gen_regression_data(
    d_in: usize,
    d_out: usize,
    n: usize,
    seed: u64,
) -> (Tensor<f32>, Tensor<f32>) {
    let x = xorshift_fill(seed, n * d_in);
    let y = xorshift_fill(seed.wrapping_add(1), n * d_out);
    (tensor(x, &[n, d_in]), tensor(y, &[n, d_out]))
}

/// `compat::Sequential` 経由で `Lbfgs::try_step_closure` を 1 outer step
/// 実行する共通ヘルパー本体。`eval` は「試行 params 書き込み済みの
/// `model` から loss／勾配を評価する」処理のみを担い、snapshot 取得・
/// 試行 params の書き込み・`Err` 時の snapshot 復元は本関数が一元的に
/// 担う（fail-closed 復元契約 `docs/autodiff-lbfgs-decision.md` §8.3）。
///
/// `eval` を差し替えることで、通常経路（[`lbfgs_step_on_sequential`]）と
/// 障害注入経路（closure の 2 回目以降の評価でのみ失敗させるテスト）の
/// 両方が**同一の復元経路**を通る（codex-review 指摘・PR #2300
/// discussion: テスト内で復元ロジックを個別実装すると、本ヘルパーの
/// `Err` 経路が復元を忘れても検出できないため、ヘルパー経由に統一する）。
fn lbfgs_step_with_closure(
    model: &mut Sequential,
    opt: &mut Lbfgs,
    mut eval: impl FnMut(
        &mut Sequential,
        &[Tensor<f32>],
    ) -> Result<(f32, Vec<Tensor<f32>>), AutodiffError>,
) -> Result<f32, AutodiffError> {
    let snapshot: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();

    let result = opt.try_step_closure(&snapshot, |trial| {
        model.apply_parameters(trial.to_vec())?;
        eval(model, trial)
    });

    match result {
        Ok(updated) => {
            model.apply_parameters(updated)?;
            Ok(opt.last_loss().ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "lbfgs_step_with_closure: last_loss unavailable after successful step"
                        .to_string(),
                )
            })?)
        }
        Err(err) => {
            // closure がすでに試行 params を書き込んでいるため、snapshot
            // へ復元してから元のエラーを返す（fail-closed）。
            model.apply_parameters(snapshot)?;
            Err(err)
        }
    }
}

/// `compat::Sequential` 経由で `Lbfgs::try_step_closure` を 1 outer step
/// 実行する共通ヘルパー（通常経路）。[`lbfgs_step_with_closure`] に、
/// 現在の試行パラメータで forward → loss → backward →
/// `trainable_grads` を評価する `eval` を渡すだけの薄いラッパー
/// （line search が 1 step 内で複数回評価するため、`eval` は呼ばれる
/// たびに独立した `Tape` を構築する）。snapshot 取得・`Err` 時の復元は
/// [`lbfgs_step_with_closure`] が一元的に担う。
fn lbfgs_step_on_sequential(
    model: &mut Sequential,
    opt: &mut Lbfgs,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
) -> Result<f32, AutodiffError> {
    lbfgs_step_with_closure(model, opt, |model, _trial| {
        let tape = fandhe_ai::tape();
        let bound = model.bind(&tape);
        let x = tape.var(x_data);
        let y = tape.var(y_data);
        let pred = bound.forward(&tape, &x)?;
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y)?;
        let loss_value = scalar(&loss.to_tensor());
        let grads = tape.backward(&loss)?;
        let grad_refs = bound.trainable_grads(&grads)?;
        let grads_owned: Vec<Tensor<f32>> = grad_refs.into_iter().cloned().collect();
        Ok((loss_value, grads_owned))
    })
}

// =====================================================================
// 受け入れ条件 2: closure の複数回評価
// =====================================================================

#[test]
fn lbfgs_on_sequential_evaluates_closure_multiple_times() {
    const D_IN: usize = 8;
    const D_HIDDEN: usize = 16;
    const D_OUT: usize = 4;
    const N: usize = 4;
    const OUTER_STEPS: usize = 3;

    let (x_data, y_data) = gen_regression_data(D_IN, D_OUT, N, 0xC0FF_EE10);
    let mut model = build_model(D_IN, D_HIDDEN, D_OUT);
    let cfg = LbfgsConfig {
        line_search: LbfgsLineSearch::StrongWolfe,
        max_iter: 20,
        ..LbfgsConfig::default()
    };
    let mut opt = Lbfgs::new(cfg).unwrap();

    for _ in 0..OUTER_STEPS {
        lbfgs_step_on_sequential(&mut model, &mut opt, &x_data, &y_data).unwrap();
    }

    assert!(
        opt.func_evals() > OUTER_STEPS as u64,
        "strong Wolfe line search が closure を複数回評価していない: \
         func_evals={} outer_steps={OUTER_STEPS}",
        opt.func_evals()
    );
    assert!(
        opt.n_iter() >= OUTER_STEPS as u64,
        "n_iter は outer step 数以上のはず（`max_iter` まで内部反復しうる \
         ため厳密な 1 step = 1 iteration ではない）: n_iter={} \
         outer_steps={OUTER_STEPS}",
        opt.n_iter()
    );
}

// =====================================================================
// 受け入れ条件 4: 収束の定性確認（MNIST 規模を模した合成回帰）
// =====================================================================

#[test]
fn lbfgs_on_sequential_converges_mnist_shaped() {
    // MNIST 相当の次元感（784 入力・10 クラス出力）を模しつつ、debug
    // ビルドの CI 実行時間を考慮して縮小した合成回帰データ（PyTorch
    // 比較は #2197 の fixture parity が既に担保済みのため、本テストは
    // facade 統合の定性的な収束確認に限る）。
    const D_IN: usize = 64;
    const D_HIDDEN: usize = 32;
    const D_OUT: usize = 10;
    const N: usize = 32;
    const OUTER_STEPS: usize = 8;

    let (x_data, y_data) = gen_regression_data(D_IN, D_OUT, N, 0xC0FF_EE20);
    let mut model = build_model(D_IN, D_HIDDEN, D_OUT);
    let cfg = LbfgsConfig {
        line_search: LbfgsLineSearch::StrongWolfe,
        max_iter: 20,
        ..LbfgsConfig::default()
    };
    let mut opt = Lbfgs::new(cfg).unwrap();

    let initial_loss = eval_loss(&model, &x_data, &y_data);

    for _ in 0..OUTER_STEPS {
        lbfgs_step_on_sequential(&mut model, &mut opt, &x_data, &y_data).unwrap();
    }

    // 最後の outer step 完了後（`model` は最後の `apply_parameters` 済み
    // モデル）の loss を改めて評価する。`lbfgs_step_on_sequential` の
    // 戻り値（`opt.last_loss()` = その step の更新前損失）をそのまま
    // 使うと、最後の step 内の更新による改善（または悪化）が判定に
    // 反映されない（`eval_loss` の doc 参照）。
    let final_loss = eval_loss(&model, &x_data, &y_data);

    assert!(
        final_loss < initial_loss * 0.5,
        "L-BFGS {OUTER_STEPS} outer step 後の loss が十分減少していない: \
         initial={initial_loss} final={final_loss}"
    );
}

// =====================================================================
// 受け入れ条件 5: 非有限値の fail-closed とパラメータ復元
// =====================================================================

#[test]
fn lbfgs_on_sequential_nan_input_fails_closed_and_restores_params() {
    const D_IN: usize = 8;
    const D_HIDDEN: usize = 16;
    const D_OUT: usize = 4;
    const N: usize = 4;

    let (mut x_data_vec, _) = {
        let (x, y) = gen_regression_data(D_IN, D_OUT, N, 0xC0FF_EE30);
        (
            x.contiguous().as_slice().unwrap().to_vec(),
            y.contiguous().as_slice().unwrap().to_vec(),
        )
    };
    x_data_vec[0] = f32::NAN;
    let x_data = tensor(x_data_vec, &[N, D_IN]);
    let (_, y_data) = gen_regression_data(D_IN, D_OUT, N, 0xC0FF_EE30);

    let mut model = build_model(D_IN, D_HIDDEN, D_OUT);
    let snapshot_before: Vec<Tensor<f32>> =
        model.trainable_parameters().into_iter().cloned().collect();
    let mut opt = Lbfgs::new(LbfgsConfig::default()).unwrap();

    let result = lbfgs_step_on_sequential(&mut model, &mut opt, &x_data, &y_data);
    assert!(
        matches!(result, Err(AutodiffError::InvalidArgument(_))),
        "NaN を含む入力で fail-closed に InvalidArgument を返すはず: {result:?}"
    );

    // n_iter/func_evals は不変（Lbfgs::try_step_closure の doc 契約:
    // エラー時は self の内部状態が変化しない）。
    assert_eq!(opt.n_iter(), 0);
    assert_eq!(opt.func_evals(), 0);

    // モデルの trainable params は step 前の snapshot と bit 完全一致
    // （facade 側の fail-closed 復元。トライアル params の書き込みが
    // 残っていないこと）。
    let snapshot_after = model.trainable_parameters();
    assert_eq!(snapshot_before.len(), snapshot_after.len());
    for (before, after) in snapshot_before.iter().zip(snapshot_after.iter()) {
        let before_data = before.contiguous().as_slice().unwrap().to_vec();
        let after_data = after.contiguous().as_slice().unwrap().to_vec();
        assert_eq!(before_data.len(), after_data.len());
        for (a, b) in before_data.iter().zip(after_data.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "NaN 失敗後に params が snapshot から変化している"
            );
        }
    }
}

/// [`lbfgs_on_sequential_nan_input_fails_closed_and_restores_params`] は
/// NaN を含む入力を最初から使うため、closure の**1 回目の評価**（trial ==
/// 呼び出し時点の `params` そのもの。line search 開始前の初回評価）で
/// 失敗する（`Lbfgs::try_step_closure` は初回評価直後に
/// `validate_closure_output` で非有限値を検査するため。
/// `crates/autodiff/src/nn/optim/lbfgs.rs` の該当箇所 doc 参照）。
/// この経路では `model.apply_parameters(trial)` は呼ばれるが `trial` が
/// snapshot と bit 完全一致のため、復元検証が「実質何も変えていない
/// 状態への書き戻し」に留まり、line search が試行ステップ幅で
/// パラメータを実際に動かした**後**に失敗した場合の復元経路
/// （`docs/autodiff-lbfgs-decision.md` §8.3 の fail-closed 復元契約が
/// 本来カバーする対象）を検証できていない（codex-review 指摘・PR #2300
/// discussion）。
///
/// 本テストは strong Wolfe line search が**2 回目以降**に呼び出す
/// closure 内でのみ NaN を混入させる（呼び出し回数を数え、1 回目は
/// 有限の loss／勾配を返して line search を先へ進ませる）。これにより
/// 2 回目の呼び出し時点で `trial`（探索方向 `d` とステップ幅 `t` で
/// 更新済みの試行パラメータ）が 1 回目の snapshot から**実際に乖離して
/// いる**ことを確認できる。
///
/// snapshot 取得・試行 params の書き込み・`Err` 時の snapshot 復元は
/// [`lbfgs_step_with_closure`]（[`lbfgs_step_on_sequential`] と同じ共通
/// ヘルパー）に委ねる。本テストが検証したいのは「`lbfgs_step_on_sequential`
/// の `Err` 経路が実際に復元を行うこと」であり、テスト内で復元ロジックを
/// 個別実装すると、ヘルパー側の復元漏れをすり抜けてしまう（codex-review
/// 指摘・PR #2300 discussion）。
#[test]
fn lbfgs_on_sequential_restores_params_after_trial_write_then_failure() {
    const D_IN: usize = 8;
    const D_HIDDEN: usize = 16;
    const D_OUT: usize = 4;
    const N: usize = 4;

    let (x_data, y_data) = gen_regression_data(D_IN, D_OUT, N, 0xC0FF_EE50);
    let mut model = build_model(D_IN, D_HIDDEN, D_OUT);
    let snapshot_before: Vec<Tensor<f32>> =
        model.trainable_parameters().into_iter().cloned().collect();

    let cfg = LbfgsConfig {
        line_search: LbfgsLineSearch::StrongWolfe,
        max_iter: 20,
        ..LbfgsConfig::default()
    };
    let mut opt = Lbfgs::new(cfg).unwrap();

    let call_count = std::cell::Cell::new(0u32);
    let trial_diverged_before_failure = std::cell::Cell::new(false);
    let failure_call_index = std::cell::Cell::new(0u32);

    let result = lbfgs_step_with_closure(&mut model, &mut opt, |model, trial| {
        call_count.set(call_count.get() + 1);
        let n = call_count.get();

        if n == 1 {
            // 1 回目（line search 開始前の初回評価）は有限の loss／
            // 勾配を返し、line search を先へ進ませる（`d`／`t` が
            // 確定し 2 回目以降で trial が snapshot から乖離する）。
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x)?;
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y)?;
            let loss_value = scalar(&loss.to_tensor());
            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let grads_owned: Vec<Tensor<f32>> = grad_refs.into_iter().cloned().collect();
            Ok((loss_value, grads_owned))
        } else {
            // 2 回目以降: trial が 1 回目の snapshot から実際に乖離して
            // いることを確認してから、NaN を混入させた入力で forward し
            // fail-closed（`InvalidArgument`）を誘発する。
            let diverged = trial.iter().zip(snapshot_before.iter()).any(|(t, s)| {
                let td = t.contiguous().as_slice().unwrap().to_vec();
                let sd = s.contiguous().as_slice().unwrap().to_vec();
                td.iter()
                    .zip(sd.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            });
            trial_diverged_before_failure.set(diverged);
            failure_call_index.set(n);

            let mut x_nan_vec = x_data.contiguous().as_slice().unwrap().to_vec();
            x_nan_vec[0] = f32::NAN;
            let x_nan = tensor(x_nan_vec, &[N, D_IN]);

            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_nan);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x)?;
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y)?;
            let loss_value = scalar(&loss.to_tensor());
            let grads = tape.backward(&loss)?;
            let grad_refs = bound.trainable_grads(&grads)?;
            let grads_owned: Vec<Tensor<f32>> = grad_refs.into_iter().cloned().collect();
            Ok((loss_value, grads_owned))
        }
    });

    assert!(
        matches!(result, Err(AutodiffError::InvalidArgument(_))),
        "2 回目以降の closure 呼び出しで NaN を返した場合も fail-closed に \
         InvalidArgument を返すはず: {result:?}"
    );
    // `lbfgs_step_with_closure` の `Err` 経路が snapshot 復元を担うため、
    // ここでテスト側から追加の復元操作は行わない（復元自体を検証したい
    // ため、共通ヘルパーの復元処理を経由させたまま以降の assert に進む）。
    assert!(
        failure_call_index.get() >= 2,
        "本テストの前提（失敗が 2 回目以降の closure 呼び出しで起きる \
         こと）が崩れている: failure_call_index={}",
        failure_call_index.get()
    );
    assert!(
        trial_diverged_before_failure.get(),
        "失敗直前の trial が snapshot と bit 完全一致のままだった（line \
         search が実際にパラメータを動かした後の失敗経路を検証できて \
         いない）"
    );

    // `try_step_closure` が `Err` を返した後も `self`（n_iter/func_evals）
    // は不変（doc 契約）。
    assert_eq!(opt.n_iter(), 0);
    assert_eq!(opt.func_evals(), 0);

    // モデルの trainable params は snapshot_before と bit 完全一致
    // （2 回目以降の closure 呼び出しで実際に乖離した trial 値が書き
    // 込まれた後も、fail-closed 復元により元の値へ戻ること）。
    let snapshot_after = model.trainable_parameters();
    assert_eq!(snapshot_before.len(), snapshot_after.len());
    for (before, after) in snapshot_before.iter().zip(snapshot_after.iter()) {
        let before_data = before.contiguous().as_slice().unwrap().to_vec();
        let after_data = after.contiguous().as_slice().unwrap().to_vec();
        assert_eq!(before_data.len(), after_data.len());
        for (a, b) in before_data.iter().zip(after_data.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "trial 書き込み後の NaN 失敗後に params が snapshot から \
                 変化している"
            );
        }
    }
}

// =====================================================================
// 受け入れ条件 3: AdamW との交互実行
// =====================================================================

#[test]
fn lbfgs_and_adamw_can_alternate_on_same_sequential() {
    const D_IN: usize = 8;
    const D_HIDDEN: usize = 16;
    const D_OUT: usize = 4;
    const N: usize = 4;

    let (x_data, y_data) = gen_regression_data(D_IN, D_OUT, N, 0xC0FF_EE40);
    let mut model = build_model(D_IN, D_HIDDEN, D_OUT);
    let mut lbfgs = Lbfgs::new(LbfgsConfig::default()).unwrap();
    let mut adamw = AdamW::new(AdamWConfig::default()).unwrap();

    for i in 0..6 {
        let loss = if i % 2 == 0 {
            lbfgs_step_on_sequential(&mut model, &mut lbfgs, &x_data, &y_data).unwrap()
        } else {
            let updated = {
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                let x = tape.var(&x_data);
                let y = tape.var(&y_data);
                let pred = bound.forward(&tape, &x).unwrap();
                let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
                let loss_value = scalar(&loss.to_tensor());
                let grads = tape.backward(&loss).unwrap();
                let grad_refs = bound.trainable_grads(&grads).unwrap();
                let param_refs = model.trainable_parameters();
                let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> =
                    param_refs.into_iter().zip(grad_refs).collect();
                let updated = adamw.step(&params_and_grads).unwrap();
                (loss_value, updated)
            };
            model.apply_parameters(updated.1).unwrap();
            updated.0
        };
        assert!(loss.is_finite(), "step {i}: loss が有限でない: {loss}");
    }
}
