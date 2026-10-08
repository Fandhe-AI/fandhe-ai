//! `Sequential::fit_with_prefetch`（`PrefetchDataLoader` の fit 結線。
//! イシュー #2605・親 #2603・ルート #2499。公開形は
//! `docs/tensor-core-data-prefetch-decision.md` §4 の案 B）の統合テスト。
//!
//! 意味論契約「任意の `(num_workers, prefetch_depth)` で `fit_with_metrics` と
//! bit 一致する（`History`・最終パラメータ・epoch 後のグローバル RNG 状態）」を
//! すべて `to_bits` 比較で固定する（tolerance は使わない）。facade
//! （`fandhe_ai`）のみを import する。グローバル RNG（`manual_seed`）を使うため
//! ファイル局所 `Mutex` で直列化する。
//!
//! CUDA／Metal の `#[ignore]` テストは追加しない: 新規 `Op`／`BackendOps`／VJP／
//! カーネルがなく、ローダー層はホスト側で完結するため（#2506 と同じ扱い）。

use std::sync::Mutex;

use fandhe_ai::compat::{
    Callback, EarlyStopping, FitConfig, History, Loss, Metrics, Optimizer, Sequential,
};
use fandhe_ai::data::PrefetchConfig;
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, Tensor};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// 決定的な擬似データ（バッチ数で割り切れない件数を渡して使う）。
fn synth(n: usize, d: usize, salt: u32) -> Vec<f32> {
    (0..n * d)
        .map(|i| {
            (((i as u32).wrapping_mul(2654435761).wrapping_add(salt) >> 8) % 1000) as f32 / 1000.0
        })
        .collect()
}

fn build_model(dropout: bool, classes: usize) -> Sequential {
    let mut m = Sequential::new()
        .add_linear(3, 6, 0x1111)
        .unwrap()
        .add_relu();
    if dropout {
        m = m.add_dropout(0.3).unwrap();
    }
    m.add_linear(6, classes, 0x2222).unwrap()
}

fn bits(m: &Sequential) -> Vec<(String, Vec<u32>)> {
    let mut v: Vec<(String, Vec<u32>)> = m
        .state_dict()
        .into_iter()
        .map(|(k, t)| {
            let data = t.host_slice().into_owned();
            (k, data.iter().map(|x| x.to_bits()).collect())
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn history_bits(h: &History) -> Vec<Vec<u32>> {
    vec![
        h.loss.iter().map(|x| x.to_bits()).collect(),
        h.val_loss.iter().map(|x| x.to_bits()).collect(),
        h.lr.iter().map(|x| x.to_bits()).collect(),
    ]
}

fn next_rng() -> Vec<u32> {
    fandhe_ai::rand(&[4])
        .unwrap()
        .host_slice()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn bit_identical_to_fit_with_metrics_f32() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let n = 11usize;
    let x = Tensor::new(synth(n, 3, 1), &[n, 3]).unwrap();
    let y = Tensor::new(synth(n, 2, 2), &[n, 2]).unwrap();
    for (workers, depth) in [(0usize, 1usize), (2, 2), (3, 1)] {
        for shuffle in [false, true] {
            for drop_last in [false, true] {
                let cfg = FitConfig::new(3, 4).shuffle(shuffle).drop_last(drop_last);

                fandhe_ai::manual_seed(42);
                let mut a = build_model(false, 2);
                a.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
                    .unwrap();
                let ha = a.fit_with_metrics(&x, &y, cfg, None, &mut [], &[]).unwrap();
                let ra = next_rng();

                fandhe_ai::manual_seed(42);
                let mut b = build_model(false, 2);
                b.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
                    .unwrap();
                let hb = b
                    .fit_with_prefetch(
                        &x,
                        &y,
                        cfg,
                        PrefetchConfig::new(workers, depth).unwrap(),
                        None,
                        &mut [],
                        &[],
                    )
                    .unwrap();
                let rb = next_rng();

                let tag = format!(
                    "workers={workers} depth={depth} shuffle={shuffle} drop_last={drop_last}"
                );
                assert_eq!(history_bits(&ha), history_bits(&hb), "History: {tag}");
                assert_eq!(bits(&a), bits(&b), "params: {tag}");
                assert_eq!(ra, rb, "RNG state: {tag}");
            }
        }
    }
}

#[test]
fn bit_identical_i32_with_validation_metrics_callbacks_and_dropout() {
    let _g = test_lock().lock().unwrap_or_else(|e| e.into_inner());
    let n = 13usize;
    let x = Tensor::new(synth(n, 3, 5), &[n, 3]).unwrap();
    let labels: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect();
    let y = Tensor::new(labels, &[n]).unwrap();
    let vx = Tensor::new(synth(5, 3, 9), &[5, 3]).unwrap();
    let vy = Tensor::new(vec![0i32, 1, 2, 0, 1], &[5]).unwrap();
    let cfg = FitConfig::new(4, 4).shuffle(true);
    let metrics = [Metrics::Accuracy];

    fandhe_ai::manual_seed(7);
    let mut a = build_model(true, 3);
    a.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let mut cba = [Callback::EarlyStopping(EarlyStopping::new(2))];
    let ha = a
        .fit_with_metrics(&x, &y, cfg, Some((&vx, &vy)), &mut cba, &metrics)
        .unwrap();
    let ra = next_rng();

    fandhe_ai::manual_seed(7);
    let mut b = build_model(true, 3);
    b.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .unwrap();
    let mut cbb = [Callback::EarlyStopping(EarlyStopping::new(2))];
    let hb = b
        .fit_with_prefetch(
            &x,
            &y,
            cfg,
            PrefetchConfig::new(2, 2).unwrap(),
            Some((&vx, &vy)),
            &mut cbb,
            &metrics,
        )
        .unwrap();
    let rb = next_rng();

    assert_eq!(history_bits(&ha), history_bits(&hb));
    assert_eq!(ha.val_metrics, hb.val_metrics);
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(ra, rb);
}

fn tiny() -> (Tensor<f32>, Tensor<f32>) {
    (
        Tensor::new(synth(4, 3, 1), &[4, 3]).unwrap(),
        Tensor::new(synth(4, 2, 2), &[4, 2]).unwrap(),
    )
}

fn prefetch() -> PrefetchConfig {
    PrefetchConfig::new(2, 2).unwrap()
}

#[test]
fn uncompiled_model_is_rejected_with_method_name() {
    let (x, y) = tiny();
    let mut m = build_model(false, 2);
    let err = m
        .fit_with_prefetch(&x, &y, FitConfig::new(1, 2), prefetch(), None, &mut [], &[])
        .unwrap_err();
    assert!(
        format!("{err}").contains("fit_with_prefetch"),
        "メッセージが公開メソッド名を名乗るはず: {err}"
    );
    assert!(!m.is_compiled());
}

#[test]
fn error_paths_match_fit_with_metrics_and_preserve_state() {
    let (x, y) = tiny();
    let bad_y = Tensor::new(synth(3, 2, 2), &[3, 2]).unwrap();
    let cases: Vec<(&str, FitConfig, &Tensor<f32>)> = vec![
        ("epochs=0", FitConfig::new(0, 2), &y),
        ("len mismatch", FitConfig::new(1, 2), &bad_y),
    ];
    for (label, cfg, yy) in cases {
        let mut m = build_model(false, 2);
        m.compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
            .unwrap();
        m.set_training(false);
        let reference = m
            .fit_with_metrics(&x, yy, cfg, None, &mut [], &[])
            .unwrap_err();
        let got = m
            .fit_with_prefetch(&x, yy, cfg, prefetch(), None, &mut [], &[])
            .unwrap_err();
        assert!(
            matches!(
                (&reference, &got),
                (
                    AutodiffError::InvalidArgument(_),
                    AutodiffError::InvalidArgument(_)
                )
            ) || std::mem::discriminant(&reference) == std::mem::discriminant(&got),
            "{label}: variant が一致しない: {reference:?} / {got:?}"
        );
        assert!(m.is_compiled(), "{label}: compile 状態が保たれるはず");
        assert!(!m.training(), "{label}: eval モードが復元されるはず");
    }
}
