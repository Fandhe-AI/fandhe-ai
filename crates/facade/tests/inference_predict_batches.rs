//! `Sequential::predict_batches` と `fandhe_ai::inference` の公開経路テスト
//! （イシュー #2582・親 #2581。決定記録
//! `docs/facade-predict-batches-phase-metrics-decision.md` §8.4 の確定形）。
//!
//! `fandhe_ai::` の公開面だけを import し、crate 内部の項目に触れない
//! （公開経路での到達可能性と振る舞いを固定する）。計測は thread-local で、
//! libtest は 1 テスト 1 スレッドで走るため各テスト冒頭の
//! `reset_phase_metrics()` だけで互いに干渉しない。`shuffle=true` 拒否時の
//! グローバル RNG 非消費の確認は、crate 内限定のロックを要するため
//! `compat/sequential.rs` の単体テスト側に置く。

use fandhe_ai::AutodiffError;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{DataLoader, DataLoaderConfig, TensorDataset};
use fandhe_ai::inference::{
    InferencePhase, PhaseMetrics, PhaseStat, PredictBatchInput, get_phase_metrics,
    reset_phase_metrics,
};

const SEED1: u64 = 1001;
const SEED2: u64 = 1002;

fn mlp_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 6, SEED1)
        .unwrap()
        .add_relu()
        .add_linear(6, 2, SEED2)
        .unwrap()
}

fn make_features(n: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..n * 4).map(|i| (i as f32) * 0.01 - 0.2).collect();
    Tensor::new(data, &[n, 4]).unwrap()
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は as_slice() が Some を返す")
        .to_vec()
}

fn plain_loader(n: usize, batch_size: usize) -> DataLoader<TensorDataset<f32>> {
    DataLoader::new(
        TensorDataset::new(make_features(n)).unwrap(),
        DataLoaderConfig::new(batch_size),
    )
    .unwrap()
}

/// バッチごとの出力が、同じ行を単体 `predict` した結果と bit 完全一致する
/// （端数バッチあり: N=7・batch_size=3）。
#[test]
fn predict_batches_matches_predict_per_batch_bit_exact() {
    reset_phase_metrics();
    let model = mlp_model();
    let features = make_features(7);
    let outputs = model.predict_batches(&plain_loader(7, 3)).unwrap();
    assert_eq!(outputs.len(), 3);

    let source = dense_vec(&features);
    let mut offset = 0usize;
    for (batch_out, rows) in outputs.iter().zip([3usize, 3, 1]) {
        let slice =
            Tensor::new(source[offset * 4..(offset + rows) * 4].to_vec(), &[rows, 4]).unwrap();
        let expected = model.predict(&slice).unwrap();
        assert_eq!(
            dense_vec(batch_out),
            dense_vec(&expected),
            "offset={offset}"
        );
        offset += rows;
    }
}

/// `(x, y)`・`(x, y, z)` のタプルデータセットでラベルが無視される。
#[test]
fn predict_batches_ignores_labels_in_tuple_datasets() {
    reset_phase_metrics();
    let model = mlp_model();
    let features = make_features(4);
    let labels = Tensor::<i32>::new(vec![0, 1, 0, 1], &[4]).unwrap();
    let extra = Tensor::<i32>::new(vec![5, 6, 7, 8], &[4]).unwrap();

    let pair = DataLoader::new(
        (
            TensorDataset::new(features.clone()).unwrap(),
            TensorDataset::new(labels.clone()).unwrap(),
        ),
        DataLoaderConfig::new(2),
    )
    .unwrap();
    let triple = DataLoader::new(
        (
            TensorDataset::new(features).unwrap(),
            TensorDataset::new(labels).unwrap(),
            TensorDataset::new(extra).unwrap(),
        ),
        DataLoaderConfig::new(2),
    )
    .unwrap();

    let plain = model.predict_batches(&plain_loader(4, 2)).unwrap();
    for loader_outputs in [
        model.predict_batches(&pair).unwrap(),
        model.predict_batches(&triple).unwrap(),
    ] {
        assert_eq!(loader_outputs.len(), plain.len());
        for (a, b) in loader_outputs.iter().zip(plain.iter()) {
            assert_eq!(dense_vec(a), dense_vec(b));
        }
    }
}

/// `drop_last=true` で端数バッチが捨てられる。
#[test]
fn predict_batches_respects_drop_last() {
    reset_phase_metrics();
    let model = mlp_model();
    let loader = DataLoader::new(
        TensorDataset::new(make_features(7)).unwrap(),
        DataLoaderConfig::new(3).drop_last(true),
    )
    .unwrap();
    assert_eq!(model.predict_batches(&loader).unwrap().len(), 2);
}

/// `shuffle=true` は `InvalidArgument` で拒否される（RNG 非消費の確認は
/// crate 内単体テスト側）。
#[test]
fn predict_batches_rejects_shuffle() {
    reset_phase_metrics();
    let model = mlp_model();
    let loader = DataLoader::new(
        TensorDataset::new(make_features(4)).unwrap(),
        DataLoaderConfig::new(2).shuffle(true),
    )
    .unwrap();
    let err = model.predict_batches(&loader).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
}

/// 空のローダーは `Ok(vec![])`・`batches() == 0`。
#[test]
fn predict_batches_on_empty_loader_returns_empty_vec() {
    reset_phase_metrics();
    let model = mlp_model();
    let empty = Tensor::<f32>::new(Vec::new(), &[0, 4]).unwrap();
    let loader =
        DataLoader::new(TensorDataset::new(empty).unwrap(), DataLoaderConfig::new(2)).unwrap();
    assert!(model.predict_batches(&loader).unwrap().is_empty());
    assert_eq!(get_phase_metrics().batches(), 0);
}

/// tape 不要経路の計測: `Forward.calls == batches`・`DataLoad.calls ==
/// batches + 2`（`iter()` の構築 1 回 + 枯渇を告げる最後の `None` 1 回）・
/// `TapeBuild`／`DeviceTransfer` は 0・`total.calls == 1`。
#[test]
fn predict_batches_records_phase_metrics_for_tape_free_path() {
    reset_phase_metrics();
    let model = mlp_model();
    let outputs = model.predict_batches(&plain_loader(5, 2)).unwrap();
    assert_eq!(outputs.len(), 3);

    let m = get_phase_metrics();
    assert_eq!(m.batches(), 3);
    assert_eq!(m.samples(), 5);
    assert_eq!(m.phase(InferencePhase::Forward).calls(), 3);
    assert_eq!(m.phase(InferencePhase::DataLoad).calls(), 5);
    assert_eq!(m.phase(InferencePhase::TapeBuild).calls(), 0);
    assert_eq!(m.phase(InferencePhase::DeviceTransfer).calls(), 0);
    assert_eq!(m.total().calls(), 1);
}

/// `samples` は出力の先頭軸ではなく入力バッチ軸から数える
/// （`add_flatten(0, 1)` で出力先頭軸が `N*F` になる構成）。
#[test]
fn predict_batches_counts_samples_from_input_batch_axis() {
    reset_phase_metrics();
    let model = Sequential::new()
        .add_linear(4, 6, SEED1)
        .unwrap()
        .add_flatten(0, 1);
    let outputs = model.predict_batches(&plain_loader(5, 2)).unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].shape().first().copied(), Some(12));

    let m = get_phase_metrics();
    assert_eq!(m.batches(), 3);
    assert_eq!(m.samples(), 5);
}

/// `Embedding` を含みフォールバックするモデルでは `TapeBuild.calls ==
/// batches`。
#[test]
fn predict_batches_records_tape_build_when_falling_back() {
    reset_phase_metrics();
    let model = Sequential::new().add_embedding(8, 4, None, SEED1).unwrap();
    let indices = Tensor::<f32>::new(vec![0.0, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
    let loader = DataLoader::new(
        TensorDataset::new(indices).unwrap(),
        DataLoaderConfig::new(2),
    )
    .unwrap();
    assert_eq!(model.predict_batches(&loader).unwrap().len(), 2);
    assert_eq!(
        get_phase_metrics().phase(InferencePhase::TapeBuild).calls(),
        2
    );
}

/// `since` は 2 回目の実行分だけを返す。
#[test]
fn predict_batches_since_returns_only_the_second_run_delta() {
    reset_phase_metrics();
    let model = mlp_model();
    let loader = plain_loader(4, 2);
    model.predict_batches(&loader).unwrap();
    let before = get_phase_metrics();
    model.predict_batches(&loader).unwrap();
    let delta = get_phase_metrics().since(&before);
    assert_eq!(delta.batches(), 2);
    assert_eq!(delta.samples(), 4);
    assert_eq!(delta.total().calls(), 1);
}

/// 別スレッドの計測が呼び出しスレッドへ漏れず、逆も漏れない。
#[test]
fn predict_batches_metrics_do_not_leak_across_threads() {
    reset_phase_metrics();
    let model = mlp_model();
    model.predict_batches(&plain_loader(2, 2)).unwrap();

    let other_thread_batches = std::thread::spawn(|| get_phase_metrics().batches())
        .join()
        .unwrap();
    assert_eq!(other_thread_batches, 0);
    assert_eq!(get_phase_metrics().batches(), 1);

    // 別スレッドで実行した計測は、そのスレッドにだけ残る。
    let spawned = std::thread::spawn(|| {
        let model = mlp_model();
        model.predict_batches(&plain_loader(6, 2)).unwrap();
        get_phase_metrics().batches()
    })
    .join()
    .unwrap();
    assert_eq!(spawned, 3);
    assert_eq!(get_phase_metrics().batches(), 1);
}

/// 途中のバッチで推論が失敗しても、それまでの計測が merge される。
#[test]
fn predict_batches_merges_metrics_even_when_a_batch_fails() {
    reset_phase_metrics();
    // 入力次元 3 の特徴量を in_features=4 の層へ流すと最初のバッチで失敗する。
    let model = mlp_model();
    let bad = Tensor::<f32>::new(vec![0.0; 6], &[2, 3]).unwrap();
    let loader =
        DataLoader::new(TensorDataset::new(bad).unwrap(), DataLoaderConfig::new(1)).unwrap();
    assert!(model.predict_batches(&loader).is_err());

    let m = get_phase_metrics();
    assert_eq!(
        m.total().calls(),
        1,
        "失敗した呼び出しも total に計上される"
    );
    assert!(m.phase(InferencePhase::DataLoad).calls() >= 1);
    assert_eq!(m.batches(), 0, "失敗したバッチは batches に数えない");
}

/// `PhaseStat::total()` と `total_micros()` の整合・`Default` がゼロ。
#[test]
fn phase_stat_total_is_consistent_with_total_micros() {
    reset_phase_metrics();
    let model = mlp_model();
    model.predict_batches(&plain_loader(6, 2)).unwrap();
    let m = get_phase_metrics();
    for phase in [
        InferencePhase::DataLoad,
        InferencePhase::TapeBuild,
        InferencePhase::Forward,
        InferencePhase::DeviceTransfer,
    ] {
        let stat: PhaseStat = m.phase(phase);
        assert_eq!(stat.total().as_micros(), stat.total_micros());
    }
    let zero = PhaseMetrics::default();
    assert_eq!(zero.total(), PhaseStat::default());
    assert_eq!(zero.total().total(), std::time::Duration::ZERO);
    assert_eq!(zero.batches(), 0);
    assert_eq!(zero.samples(), 0);
}

/// 公開面だけでシグネチャを固定する（`predict_batches` の引数・戻り値型と
/// `PredictBatchInput` の 3 形、`PhaseMetrics` の `Copy + Eq`）。
#[test]
fn public_signatures_are_pinned() {
    fn assert_input<T: PredictBatchInput>() {}
    assert_input::<Tensor<f32>>();
    assert_input::<(Tensor<f32>, Tensor<i32>)>();
    assert_input::<(Tensor<f32>, Tensor<i32>, Tensor<i32>)>();

    fn assert_copy_eq<T: Copy + Eq + Default + std::fmt::Debug>() {}
    assert_copy_eq::<PhaseMetrics>();
    assert_copy_eq::<PhaseStat>();

    type PredictBatchesFn =
        fn(&Sequential, &DataLoader<TensorDataset<f32>>) -> Result<Vec<Tensor<f32>>, AutodiffError>;
    let _: PredictBatchesFn = Sequential::predict_batches::<TensorDataset<f32>>;
    let _: fn() -> PhaseMetrics = get_phase_metrics;
    let _: fn() = reset_phase_metrics;
    let _: fn(&PhaseMetrics, &PhaseMetrics) -> PhaseMetrics = PhaseMetrics::since;
}

/// 利用例（#2583）: ラベル付きデータセットを `predict_batches` で推論し、
/// `reset_phase_metrics()` → 1 回目 → 2 回目の順で呼んで「2 回目だけ」の差分を
/// `since` で取る一連の手順（時間値は環境依存のため件数だけを固定する）。
#[test]
fn usage_example_labelled_dataset_with_before_after_metrics() {
    let model = mlp_model();
    let labels = Tensor::<i32>::new(vec![0, 1, 0, 1, 0], &[5]).unwrap();
    let loader = DataLoader::new(
        (
            TensorDataset::new(make_features(5)).unwrap(),
            TensorDataset::new(labels).unwrap(),
        ),
        DataLoaderConfig::new(2),
    )
    .unwrap();

    reset_phase_metrics();
    model.predict_batches(&loader).unwrap();
    let before = get_phase_metrics();
    let outputs = model.predict_batches(&loader).unwrap();
    let delta = get_phase_metrics().since(&before);

    // 5 サンプル・batch_size=2 → 3 バッチ（2・2・1）。累計は 2 回分、差分は 2 回目のみ。
    assert_eq!(outputs.len(), 3);
    assert_eq!(delta.batches(), 3);
    assert_eq!(delta.samples(), 5);
    assert_eq!(delta.total().calls(), 1);
    assert_eq!(get_phase_metrics().batches(), 6);
    assert_eq!(delta.phase(InferencePhase::Forward).calls(), 3);
    assert_eq!(delta.phase(InferencePhase::DeviceTransfer).calls(), 0);
}
