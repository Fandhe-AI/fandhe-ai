//! #1721（親 #1625）: 損失スケーリング・unscale・inf/nan 検出のコア
//! 関数（`fandhe_ai_autodiff::nn::optim::amp`）の受け入れテスト。
//!
//! `amp` は `Tensor<f32>` へ実体化済みの勾配に対する後処理のみを扱う
//! 純関数・純データ構造の集合（`amp.rs` doc 参照）。新規 `Op`／VJP を
//! 追加しないため数値微分突合は対象外で、代わりに
//! 以下を固定する:
//! - スケール往復（`scale_grads`/`unscale_grads`）の解析的検証
//! - `scale_loss` → `Tape::backward` → `unscale_grads` の bit 完全一致
//!   （非スケール backward との比較）
//! - inf/nan 検出・`should_skip_step` 契約
//! - 引数検証（fail-closed）
//! - `GradScaler` のスケール更新契約（backoff／growth）
//! - 適用順序契約（非有限検出は clip より先に判定する）

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::optim::{
    GradScaler, GradScalerConfig, clip_grad_norm, has_non_finite, scale_grads, scale_loss,
    unscale_grads,
};
use fandhe_ai_tensor_core::Tensor;

use bench_harness::rng::Xorshift64Star;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("test fixture: contiguous なテンソルのみ渡す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

// =====================================================================
// スケール往復（解析的検証）
// =====================================================================

#[test]
fn scale_then_unscale_is_bit_exact_for_power_of_two_scale() {
    // 2 のべき乗（65536.0）は `unscale_grads` doc が主張する
    // 「厳密な逆写像」を bit 単位で固定する（over/underflow から
    // 十分遠い乱数域）。
    let mut rng = Xorshift64Star::new(0x5EED_1234);
    let data: Vec<f32> = rng
        .fill_vec(64)
        .into_iter()
        .map(|v| (v - 0.5) * 10.0)
        .collect();
    let g = tensor(data, &[8, 8]);

    let scale = 65536.0f32;
    let scaled = scale_grads(&[&g], scale).unwrap();
    let result = unscale_grads(&[&scaled[0]], scale).unwrap();

    assert!(!result.found_non_finite);
    assert_eq!(
        bits(&result.grads[0]),
        bits(&g),
        "2 のべき乗 scale は bit 完全一致のはず"
    );
}

#[test]
fn scale_then_unscale_is_close_for_non_power_of_two_scale() {
    // 非 2 のべき乗（3.0）は丸め差が生じうるため相対誤差で検証する
    // （`unscale_grads` doc「2 のべき乗以外の scale では丸め差が生じ
    // うる」の固定）。
    let mut rng = Xorshift64Star::new(0x5EED_5678);
    let data: Vec<f32> = rng
        .fill_vec(32)
        .into_iter()
        .map(|v| (v - 0.5) * 10.0)
        .collect();
    let g = tensor(data.clone(), &[32]);

    let scale = 3.0f32;
    let scaled = scale_grads(&[&g], scale).unwrap();
    let result = unscale_grads(&[&scaled[0]], scale).unwrap();

    assert!(!result.found_non_finite);
    let out = result.grads[0].as_slice().unwrap();
    for (orig, roundtripped) in data.iter().zip(out.iter()) {
        let rel = (orig - roundtripped).abs() / orig.abs().max(1e-6);
        assert!(
            rel < 1e-6,
            "orig={orig} roundtripped={roundtripped} rel={rel}"
        );
    }
}

#[test]
fn scale_grads_and_unscale_grads_accept_empty_slice() {
    let scaled = scale_grads(&[], 2.0).unwrap();
    assert!(scaled.is_empty());

    let result = unscale_grads(&[], 2.0).unwrap();
    assert!(result.grads.is_empty());
    assert!(!result.found_non_finite);
}

#[test]
fn scale_grads_preserves_shape_for_non_contiguous_input() {
    // transpose 後の non-contiguous view でも shape を正しく引き継いで
    // 往復することを固定する（`Tensor::host_slice`/`as_slice` の
    // non-contiguous フォールバック経路の健全性）。
    let g = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let gt = g.transpose(0, 1).unwrap();
    assert_eq!(gt.shape(), &[3, 2]);

    let scaled = scale_grads(&[&gt], 4.0).unwrap();
    assert_eq!(scaled[0].shape(), &[3, 2]);
    let result = unscale_grads(&[&scaled[0]], 4.0).unwrap();
    assert_eq!(result.grads[0].shape(), &[3, 2]);

    let expected = gt.contiguous();
    assert_eq!(bits(&result.grads[0]), bits(&expected));
}

// =====================================================================
// backward 統合の bit 一致（本 sub の看板テスト）
// =====================================================================

#[test]
fn scale_loss_backward_unscale_matches_unscaled_backward_bit_exact() {
    use fandhe_ai_autodiff::nn::Linear;

    const BATCH: usize = 4;
    const D_IN: usize = 6;
    const D_OUT: usize = 3;

    let mut rng = Xorshift64Star::new(0xABCD_EF01);
    let x_data = tensor(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]);
    let y_data = tensor(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]);

    let l1 = Linear::new(D_IN, D_OUT, true, 0x2222_2222).expect("test fixture: shape は事前に妥当");

    // scale=2^8（2 のべき乗。scale_loss/unscale_grads とも bit 完全
    // 一致契約が成立する値。`Var::mul` の VJP チェーンを含め f32
    // 乗算・除算のみで構成されるため）。
    let scale = 256.0f32;

    // 経路 A: scale_loss → backward → unscale_grads。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xa = tape_a.var(&x_data);
    let ya = tape_a.var(&y_data);
    let l1a = l1.bind(&tape_a);
    let ha = l1a.forward(&xa).unwrap();
    let loss_a = ha.mse_loss(&ya).unwrap();
    let scaled_loss = scale_loss(&loss_a, scale).unwrap();
    let grads_a = tape_a.backward(&scaled_loss).unwrap();
    let w_grad_a = grads_a.get(&l1a.weight).unwrap().unwrap().clone();
    let b_grad_a = grads_a
        .get(l1a.bias.as_ref().expect("test fixture: bias=true"))
        .unwrap()
        .unwrap()
        .clone();
    let unscaled = unscale_grads(&[&w_grad_a, &b_grad_a], scale).unwrap();
    assert!(!unscaled.found_non_finite);

    // 経路 B: 非スケール backward（比較対象の正解値）。
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xb = tape_b.var(&x_data);
    let yb = tape_b.var(&y_data);
    let l1b = l1.bind(&tape_b);
    let hb = l1b.forward(&xb).unwrap();
    let loss_b = hb.mse_loss(&yb).unwrap();
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let w_grad_b = grads_b.get(&l1b.weight).unwrap().unwrap();
    let b_grad_b = grads_b
        .get(l1b.bias.as_ref().expect("test fixture: bias=true"))
        .unwrap()
        .unwrap();

    assert_eq!(
        bits(&unscaled.grads[0]),
        bits(w_grad_b),
        "weight 勾配が scale_loss→backward→unscale と非スケール backward で bit 完全一致しない"
    );
    assert_eq!(
        bits(&unscaled.grads[1]),
        bits(b_grad_b),
        "bias 勾配が scale_loss→backward→unscale と非スケール backward で bit 完全一致しない"
    );
}

#[test]
fn scale_loss_leaf_does_not_survive_tape_reset() {
    // `scale_loss` が登録するスカラー葉は forward の他演算より後に
    // 登録されるため、`Tape::reset` をまたいで蓄積しない
    // （`amp.rs::scale_loss` doc「スカラー葉は Tape::reset をまたいで
    // 蓄積しない」の固定）。
    use fandhe_ai_autodiff::nn::Linear;

    let l1 = Linear::new(4, 2, true, 0x3333_3333).expect("test fixture: shape は事前に妥当");
    let x_data = tensor(vec![1.0; 4 * 4], &[4, 4]);
    let y_data = tensor(vec![0.5; 4 * 2], &[4, 2]);

    let mut tape = Tape::new_with_ops(common::naive_ops());

    // `Tape::reset` doc: 最初の非葉ノード（本テストでは `forward` の
    // `matmul`）が記録される *前* に登録した葉（`x`/`y`/`weight`/
    // `bias`）のみが以後の reset で保持される。1 回目の step でこの
    // 葉プレフィックス長が固定されるため、`scale_loss` が step ごとに
    // 追加登録するスカラー葉（forward より後に登録される）は毎 reset
    // で確実に破棄され、2 回目以降の reset 後ノード数は 1 回目の
    // reset 後ノード数から増加しないはずである。
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let l1v = l1.bind(&tape);
    let h = l1v.forward(&x).unwrap();
    let loss = h.mse_loss(&y).unwrap();
    let _scaled = scale_loss(&loss, 128.0).unwrap();
    tape.reset();
    let len_after_first_reset = tape.len();

    for _ in 0..3 {
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let l1v = l1.bind(&tape);
        let h = l1v.forward(&x).unwrap();
        let loss = h.mse_loss(&y).unwrap();
        let _scaled = scale_loss(&loss, 128.0).unwrap();
        tape.reset();
        assert_eq!(
            tape.len(),
            len_after_first_reset,
            "reset 後のノード数が step を重ねるごとに増加しており、scale_loss の葉が蓄積している"
        );
    }
}

// =====================================================================
// inf/nan 検出
// =====================================================================

#[test]
fn has_non_finite_detects_inf_and_nan_across_multiple_tensors() {
    let finite = tensor(vec![1.0, 2.0], &[2]);
    let with_inf = tensor(vec![1.0, f32::INFINITY], &[2]);
    let with_neg_inf = tensor(vec![f32::NEG_INFINITY, 0.0], &[2]);
    let with_nan = tensor(vec![f32::NAN, 0.0], &[2]);

    assert!(!has_non_finite(&[&finite]));
    assert!(!has_non_finite(&[&finite, &finite]));
    assert!(has_non_finite(&[&finite, &with_inf]));
    assert!(has_non_finite(&[&with_neg_inf]));
    assert!(has_non_finite(&[&with_nan]));
    assert!(!has_non_finite(&[]));
}

#[test]
fn unscale_grads_found_non_finite_matches_has_non_finite_and_is_ok_not_err() {
    let with_nan = tensor(vec![1.0, f32::NAN], &[2]);
    let result = unscale_grads(&[&with_nan], 2.0).unwrap();
    assert!(result.found_non_finite);
    assert!(result.should_skip_step());
    assert_eq!(has_non_finite(&[&with_nan]), result.found_non_finite);

    let finite = tensor(vec![1.0, 2.0], &[2]);
    let clean = unscale_grads(&[&finite], 2.0).unwrap();
    assert!(!clean.found_non_finite);
    assert!(!clean.should_skip_step());
}

#[test]
fn unscale_grads_non_finite_step_should_be_skipped_before_clip() {
    // 適用順序契約（`nn/optim/mod.rs` doc）: 非有限検出は clip より
    // 先に判定する。同じ非有限勾配を `clip_grad_norm` に渡すと `Err`
    // になることを確認し、「先に should_skip_step で弾かなければ
    // ならない」理由を固定する。
    let with_inf = tensor(vec![1.0, f32::INFINITY], &[2]);
    let unscaled = unscale_grads(&[&with_inf], 1.0).unwrap();
    assert!(unscaled.should_skip_step());

    match clip_grad_norm(&[&unscaled.grads[0]], 1.0) {
        Err(clip_err) => assert!(
            format!("{clip_err}").contains("非有限"),
            "clip_err={clip_err}"
        ),
        Ok(_) => panic!("非有限勾配に対して clip_grad_norm が Ok を返した"),
    }
}

// =====================================================================
// 引数検証（fail-closed）
// =====================================================================

#[test]
fn scale_grads_rejects_invalid_scale() {
    let g = tensor(vec![1.0], &[1]);
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(scale_grads(&[&g], bad).is_err(), "bad={bad}");
    }
}

#[test]
fn unscale_grads_rejects_invalid_scale() {
    let g = tensor(vec![1.0], &[1]);
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(unscale_grads(&[&g], bad).is_err(), "bad={bad}");
    }
}

#[test]
fn scale_loss_rejects_invalid_scale() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let loss = tape.var(&Tensor::scalar(1.0f32));
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(scale_loss(&loss, bad).is_err(), "bad={bad}");
    }
}

#[test]
fn grad_scaler_new_rejects_invalid_config() {
    let base = GradScalerConfig::default();

    let mut bad_init = base;
    bad_init.init_scale = 0.0;
    assert!(GradScaler::new(bad_init).is_err());

    let mut bad_growth = base;
    bad_growth.growth_factor = 1.0;
    assert!(GradScaler::new(bad_growth).is_err());

    let mut bad_backoff_low = base;
    bad_backoff_low.backoff_factor = 0.0;
    assert!(GradScaler::new(bad_backoff_low).is_err());

    let mut bad_backoff_high = base;
    bad_backoff_high.backoff_factor = 1.0;
    assert!(GradScaler::new(bad_backoff_high).is_err());

    let mut bad_interval = base;
    bad_interval.growth_interval = 0;
    assert!(GradScaler::new(bad_interval).is_err());

    assert!(GradScaler::new(base).is_ok());
}

// =====================================================================
// スケール更新契約
// =====================================================================

#[test]
fn grad_scaler_default_matches_pytorch_defaults() {
    let config = GradScalerConfig::default();
    assert_eq!(config.init_scale, 65536.0);
    assert_eq!(config.growth_factor, 2.0);
    assert_eq!(config.backoff_factor, 0.5);
    assert_eq!(config.growth_interval, 2000);

    let scaler = GradScaler::new(config).unwrap();
    assert_eq!(scaler.scale(), 65536.0);
    assert_eq!(scaler.growth_tracker(), 0);
}

#[test]
fn grad_scaler_backoff_on_non_finite_resets_tracker() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 4,
    })
    .unwrap();

    scaler.update(false).unwrap();
    scaler.update(false).unwrap();
    assert_eq!(scaler.growth_tracker(), 2);

    scaler.update(true).unwrap();
    assert_eq!(scaler.scale(), 512.0);
    assert_eq!(scaler.growth_tracker(), 0);
}

#[test]
fn grad_scaler_grows_after_growth_interval_clean_steps() {
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 1024.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 3,
    })
    .unwrap();

    // interval-1 回では成長しない。
    scaler.update(false).unwrap();
    scaler.update(false).unwrap();
    assert_eq!(
        scaler.scale(),
        1024.0,
        "interval に達する前に成長してはならない"
    );

    // interval 回目で成長し tracker がリセットされる。
    scaler.update(false).unwrap();
    assert_eq!(scaler.scale(), 2048.0);
    assert_eq!(scaler.growth_tracker(), 0);
}

#[test]
fn grad_scaler_growth_that_would_overflow_is_skipped_and_scale_stays() {
    // 成長後の scale が非有限になる場合は成長をスキップし据え置く
    // （`GradScaler::update` doc 参照）。tracker は 0 へリセットされる。
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: f32::MAX / 1.5,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1,
    })
    .unwrap();

    let scale_before = scaler.scale();
    scaler.update(false).unwrap();
    assert_eq!(
        scaler.scale(),
        scale_before,
        "overflow する成長はスキップされ据え置かれるはず"
    );
    assert_eq!(scaler.growth_tracker(), 0);
}

#[test]
fn grad_scaler_backoff_to_invalid_scale_is_error() {
    // backoff 後の scale が非正規化数・0・非有限になる設定は `Err`
    // にする（fail-closed。以後の unscale が 0 除算・無意味な値に
    // なるため）。
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: f32::MIN_POSITIVE,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1,
    })
    .unwrap();

    let err = scaler.update(true).unwrap_err();
    assert!(format!("{err}").contains("scale"), "err={err}");
}

// =====================================================================
// 適用順序統合: GradScaler を用いたミニ学習ループ
// =====================================================================

#[test]
fn grad_scaler_skips_update_on_non_finite_step_and_updates_on_clean_step() {
    use fandhe_ai_autodiff::nn::Linear;

    const D_IN: usize = 4;
    const D_OUT: usize = 2;
    const BATCH: usize = 3;
    const LR: f32 = 0.01;

    let mut rng = Xorshift64Star::new(0xFEED_0001);
    let x_data = tensor(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]);
    let y_data = tensor(rng.fill_vec(BATCH * D_OUT), &[BATCH, D_OUT]);

    let mut l1 = Linear::new(D_IN, D_OUT, true, 0x4444_4444).expect("test fixture");
    let mut scaler = GradScaler::new(GradScalerConfig {
        init_scale: 8.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 100, // 本テストの step 数では成長条件に届かない
    })
    .unwrap();

    let weight_before = l1.weight().clone();

    // step 1: 通常 step（clean）。パラメータが更新されることを確認する。
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let l1v = l1.bind(&tape);
        let h = l1v.forward(&x).unwrap();
        let loss = h.mse_loss(&y).unwrap();
        let scaled_loss = scaler.scale_loss(&loss).unwrap();
        let grads = tape.backward(&scaled_loss).unwrap();
        let w_grad = grads.get(&l1v.weight).unwrap().unwrap().clone();
        let unscaled = scaler.unscale(&[&w_grad]).unwrap();
        assert!(!unscaled.should_skip_step());

        let new_weight = tensor(
            l1.weight()
                .as_slice()
                .unwrap()
                .iter()
                .zip(unscaled.grads[0].as_slice().unwrap())
                .map(|(&p, &g)| p - LR * g)
                .collect(),
            l1.weight().shape(),
        );
        l1 = Linear::from_parameters(new_weight, l1.bias().cloned())
            .expect("test fixture: shape はパラメータ更新前後で不変");
        scaler.update(unscaled.found_non_finite).unwrap();
    }

    assert_ne!(
        bits(l1.weight()),
        bits(&weight_before),
        "clean step ではパラメータが更新されるはず"
    );
    assert_eq!(
        scaler.scale(),
        8.0,
        "clean 1 step では backoff/growth いずれも起きないはず"
    );

    // step 2: 人為的に非有限勾配を注入した step（overflow を模擬）。
    // should_skip_step が true になり、パラメータ更新をスキップし
    // GradScaler が backoff することを確認する。
    let weight_before_step2 = l1.weight().clone();
    {
        let poisoned = tensor(vec![f32::INFINITY; D_IN * D_OUT], l1.weight().shape());
        let unscaled = scaler.unscale(&[&poisoned]).unwrap();
        assert!(unscaled.should_skip_step());

        // 契約どおり: skip する step ではパラメータ更新もクリップも
        // 行わない（l1 は変更しない）。
        scaler.update(unscaled.found_non_finite).unwrap();
    }

    assert_eq!(
        bits(l1.weight()),
        bits(&weight_before_step2),
        "非有限 step ではパラメータが更新されてはならない"
    );
    assert_eq!(
        scaler.scale(),
        4.0,
        "非有限検出後は backoff_factor 倍される"
    );
}
