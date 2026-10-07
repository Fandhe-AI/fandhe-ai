//! `IterableDataset`・`IterableDataLoader`・`BatchSampler`（イシュー #2662・親 #2660）の
//! 統合テスト。
//!
//! これらは #2679 で `fandhe_ai::data` へ公開済み（承認はルート #2499 のコメント。
//! `docs/tensor-core-iterable-dataset-batch-sampler-decision.md` §5）のため、`fandhe_ai`
//! のみを import する（`data_dataset_compose.rs` と同型）。
//!
//! 実 PyTorch 2.14.0 の実行値 fixture
//! （`crates/tensor-core/tests/fixtures/iterable-batch-sampler-pytorch-reference/`）と突合する。
//! 整数表（添字バッチ列・バッチ数）は完全一致、f32 の行は bit 一致を確認したうえで
//! 統一複合判定（`assert_parity`）にも通す。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{BatchSampler, IterableDataLoader, IterableDataset};
use fandhe_ai::data::{DataError, Sampler, SamplerDataLoader, TensorDataset};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai_backend_cpu::assert_parity;
use serde_json::Value;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tensor-core/tests/fixtures/iterable-batch-sampler-pytorch-reference/iterable_batch_sampler_reference.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("test fixture: fixture 読み込みに失敗: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("test fixture: JSON 解析に失敗: {e}"))
}

fn arr<'a>(v: &'a Value, key: &str) -> &'a Vec<Value> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("test fixture: {key} は配列のはず"))
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap_or_else(|| panic!("test fixture: 配列のはず"))
        .iter()
        .map(|x| {
            x.as_u64()
                .unwrap_or_else(|| panic!("test fixture: u64 のはず"))
        })
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    u64s(v).into_iter().map(|x| x as usize).collect()
}

fn i32s(v: &Value) -> Vec<i32> {
    v.as_array()
        .unwrap_or_else(|| panic!("test fixture: 配列のはず"))
        .iter()
        .map(|x| {
            x.as_i64()
                .unwrap_or_else(|| panic!("test fixture: i64 のはず")) as i32
        })
        .collect()
}

/// `{shape, bits}` レコードから f32 テンソルを復元する（bit 保存のため再生成しない）。
fn tensor_of(rec: &Value) -> Tensor<f32> {
    let shape = usizes(&rec["shape"]);
    let data: Vec<f32> = u64s(&rec["bits"])
        .into_iter()
        .map(|b| f32::from_bits(b as u32))
        .collect();
    Tensor::new(data, &shape).unwrap_or_else(|e| panic!("test fixture: tensor 構築に失敗: {e}"))
}

/// bit 一致 + 統一複合判定。
fn assert_rows(context: &str, actual: &Tensor<f32>, expected: &Tensor<f32>) {
    assert_eq!(actual.shape(), expected.shape(), "{context}: shape");
    let a = actual.host_slice();
    let e = expected.host_slice();
    let a_bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let e_bits: Vec<u32> = e.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a_bits, e_bits, "{context}: bit 一致");
    assert_parity(context, &a, &e);
}

/// サンプル列をそのまま順に流すテスト用 `IterableDataset`（失敗要素も `Result` で保持する）。
struct VecStream<S: Clone> {
    items: Vec<Result<S, DataError>>,
}

impl<S: Clone> VecStream<S> {
    fn ok(samples: Vec<S>) -> Self {
        Self {
            items: samples.into_iter().map(Ok).collect(),
        }
    }
}

impl<S: Clone> IterableDataset for VecStream<S> {
    type Sample = S;
    fn iter_samples(&self) -> Box<dyn Iterator<Item = Result<S, DataError>> + '_> {
        Box::new(self.items.iter().cloned())
    }
}

/// `[n, d]` のテンソルを行ごとの `[d]` テンソルへ分解する。
fn split_rows(t: &Tensor<f32>) -> Vec<Tensor<f32>> {
    let n = t.shape()[0];
    let d: usize = t.shape()[1..].iter().product();
    let data = t.host_slice();
    (0..n)
        .map(|i| Tensor::new(data[i * d..(i + 1) * d].to_vec(), &t.shape()[1..]).unwrap())
        .collect()
}

#[test]
fn fixture_provenance_is_real_pytorch() {
    let f = fixture();
    let v = f["torch_version"].as_str().unwrap_or("");
    assert!(
        v.starts_with("2.14.0"),
        "実 PyTorch 2.14.0 系の実行値のはず: {v}"
    );
}

#[test]
fn batch_sampler_batches_match_pytorch() {
    let f = fixture();
    let cases = arr(&f, "batch_sampler_cases");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let mut sampler = BatchSampler::new(
            usizes(&case["order"]),
            case["batch_size"].as_u64().unwrap() as usize,
            case["drop_last"].as_bool().unwrap(),
        )
        .unwrap();
        sampler.start_epoch().unwrap();
        let mut got: Vec<Vec<usize>> = Vec::new();
        loop {
            let b = sampler.next_batch();
            if b.is_empty() {
                break;
            }
            got.push(b);
        }
        let expected: Vec<Vec<usize>> = arr(case, "batches").iter().map(usizes).collect();
        assert_eq!(got, expected, "{name}: 添字バッチ列");
        assert_eq!(
            sampler.num_batches(),
            Some(case["len"].as_u64().unwrap() as usize),
            "{name}: len(BatchSampler)"
        );
    }
}

#[test]
fn iterable_loader_batches_match_pytorch() {
    let f = fixture();
    let cases = arr(&f, "iterable_cases");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let rows = tensor_of(&case["rows"]);
        let loader = IterableDataLoader::new(
            VecStream::ok(split_rows(&rows)),
            case["batch_size"].as_u64().unwrap() as usize,
            case["drop_last"].as_bool().unwrap(),
        )
        .unwrap();
        let got: Vec<Tensor<f32>> = loader.iter().map(|b| b.unwrap()).collect();
        let expected = arr(case, "batches");
        assert_eq!(got.len(), expected.len(), "{name}: バッチ数");
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_rows(&format!("{name}[{i}]"), g, &tensor_of(e));
        }
    }
}

#[test]
fn iterable_tuple_batches_match_pytorch() {
    let f = fixture();
    for case in arr(&f, "tuple_cases") {
        let name = case["name"].as_str().unwrap_or("?");
        let xs = split_rows(&tensor_of(&case["xs"]));
        let ys = i32s(&case["ys"]);
        let samples: Vec<(Tensor<f32>, Tensor<i32>)> = xs
            .into_iter()
            .zip(ys)
            .map(|(x, y)| (x, Tensor::new(vec![y], &[]).unwrap()))
            .collect();
        let loader = IterableDataLoader::new(
            VecStream::ok(samples),
            case["batch_size"].as_u64().unwrap() as usize,
            case["drop_last"].as_bool().unwrap(),
        )
        .unwrap();
        let got: Vec<(Tensor<f32>, Tensor<i32>)> = loader.iter().map(|b| b.unwrap()).collect();
        let expected = arr(case, "batches");
        assert_eq!(got.len(), expected.len(), "{name}: バッチ数");
        for (i, ((gx, gy), e)) in got.iter().zip(expected).enumerate() {
            assert_rows(&format!("{name}[{i}].x"), gx, &tensor_of(&e["x"]));
            assert_eq!(gy.host_slice().to_vec(), i32s(&e["y"]), "{name}[{i}].y");
        }
    }
}

/// torch の成否との一対一照合。型の上で書けないケースは `STRUCTURALLY_IMPOSSIBLE`
/// （決定記録 §3 の意図的差異と対応）。
#[test]
fn error_cases_match_pytorch_modulo_intended_differences() {
    // iterable とシャッフル／sampler／batch_sampler の併用は torch が実行時 ValueError。
    // 本実装は型の上で存在しない。`len(DataLoader(iterable))` は長さ API を持たない。
    const STRUCTURALLY_IMPOSSIBLE: &[&str] = &[
        "iterable_shuffle_true",
        "iterable_sampler",
        "iterable_batch_sampler",
        "iterable_len",
    ];
    let f = fixture();
    for case in arr(&f, "error_cases") {
        let name = case["name"].as_str().unwrap_or("?");
        assert_eq!(case["torch_raises"].as_bool(), Some(true), "{name}");
        match name {
            n if STRUCTURALLY_IMPOSSIBLE.contains(&n) => {}
            "batch_sampler_zero_batch_size" => {
                assert_eq!(
                    BatchSampler::new((0..5).collect(), 0, false).unwrap_err(),
                    DataError::ZeroBatchSize
                );
            }
            "iterable_shape_mismatch" => {
                let samples = vec![
                    Tensor::<f32>::new(vec![0.0; 2], &[2]).unwrap(),
                    Tensor::<f32>::new(vec![0.0; 3], &[3]).unwrap(),
                    Tensor::<f32>::new(vec![0.0; 2], &[2]).unwrap(),
                ];
                let loader = IterableDataLoader::new(VecStream::ok(samples), 2, false).unwrap();
                assert!(matches!(
                    loader.iter().next(),
                    Some(Err(DataError::SampleShapeMismatch { position: 1, .. }))
                ));
            }
            "iterable_mid_stream_error" => {
                let mut items: Vec<Result<Tensor<f32>, DataError>> = (0..3)
                    .map(|i| Ok(Tensor::new(vec![i as f32, 0.0], &[2]).unwrap()))
                    .collect();
                items.push(Err(DataError::IterableStream {
                    reason: "stream failure".into(),
                }));
                let loader = IterableDataLoader::new(VecStream { items }, 2, false).unwrap();
                let mut ok_batches = 0usize;
                let mut it = loader.iter();
                let err = loop {
                    match it.next() {
                        Some(Ok(_)) => ok_batches += 1,
                        Some(Err(e)) => break e,
                        None => panic!("エラーが yield されるはず"),
                    }
                };
                assert!(matches!(err, DataError::IterableStream { .. }));
                assert_eq!(
                    Some(ok_batches as u64),
                    case["batches_before_error"].as_u64(),
                    "失敗前に完成したバッチ数が torch と一致"
                );
                assert!(it.next().is_none(), "エラー後は打ち切り");
            }
            other => panic!("未知の error ケース: {other}"),
        }
    }
}

const N: usize = 24;
const D_IN: usize = 4;
const D_OUT: usize = 2;

fn gen_xy(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    (
        Tensor::new(rng.fill_vec(N * D_IN), &[N, D_IN]).unwrap(),
        Tensor::new(rng.fill_vec(N * D_OUT), &[N, D_OUT]).unwrap(),
    )
}

/// `BatchSampler` の添字列で `TensorDataset` の行が指定順に供給される。
#[test]
fn batch_sampler_feeds_sampler_data_loader() {
    let (x, _) = gen_xy(0xABCD);
    let ds = TensorDataset::new(x.clone()).unwrap();
    let order = vec![5, 3, 3, 0, 23, 7, 1];
    let sampler = BatchSampler::new(order.clone(), 3, false).unwrap();
    let mut loader = SamplerDataLoader::new(ds, sampler).unwrap();
    let xd = x.host_slice();
    let mut got: Vec<f32> = Vec::new();
    for batch in loader.iter() {
        got.extend_from_slice(&batch.unwrap().host_slice());
    }
    let expected: Vec<f32> = order
        .iter()
        .flat_map(|&i| xd[i * D_IN..(i + 1) * D_IN].to_vec())
        .collect();
    assert_eq!(got, expected);
}

/// (x, y) ストリームを `IterableDataLoader` へ渡して学習し、loss が減少する。
#[test]
fn training_on_iterable_loader_converges() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let (x, y) = gen_xy(0xFEED);
    let samples: Vec<(Tensor<f32>, Tensor<f32>)> =
        split_rows(&x).into_iter().zip(split_rows(&y)).collect();
    let loader = IterableDataLoader::new(VecStream::ok(samples), 4, false).unwrap();

    let mut model = Sequential::new()
        .add_linear(D_IN, 8, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(8, D_OUT, 0x2222_2222)
        .unwrap();
    let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let mut losses = Vec::new();
    for _ in 0..30 {
        let (mut sum, mut cnt) = (0.0f32, 0usize);
        for batch in loader.iter() {
            let (xb, yb) = batch.unwrap();
            let updated = {
                let tape = fandhe_ai::tape();
                let bound = model.bind(&tape);
                let xv = tape.var(&xb);
                let yv = tape.var(&yb);
                let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&yv).unwrap();
                sum += loss.to_tensor().get(&[]).unwrap();
                cnt += 1;
                let grads = tape.backward(&loss).unwrap();
                let grad_refs = bound.trainable_grads(&grads).unwrap();
                let params = model.trainable_parameters();
                sgd.step(&params, &grad_refs).unwrap()
            };
            model.apply_parameters(updated).unwrap();
        }
        losses.push(sum / cnt as f32);
    }
    assert!(
        losses[losses.len() - 1] < losses[0],
        "loss が減少するはず: {losses:?}"
    );
}

/// 供給したバッチを `tape_for(device)` の `var` へ上げて戻した値がホスト値と bit 一致する。
fn assert_upload_round_trip(device: fandhe_ai::Device) {
    let (x, _) = gen_xy(0x1234);
    let loader = IterableDataLoader::new(VecStream::ok(split_rows(&x)), 5, false).unwrap();
    let tape = fandhe_ai::tape_for(device)
        .unwrap_or_else(|e| panic!("test fixture: tape_for が失敗した: {e}"));
    for batch in loader.iter() {
        let batch = batch.unwrap();
        let uploaded = tape.var(&batch).to_tensor();
        assert_eq!(uploaded.host_slice().as_ref(), batch.host_slice().as_ref());
    }
}

/// CUDA 実機での `Tape::var` アップロード round-trip（実機未実測。GB10 へ申し送り）。
#[test]
#[ignore = "実機（CUDA）依存。DGX Spark GB10 等で手動実行する"]
fn iterable_batch_upload_round_trips_on_cuda_tape() {
    assert_upload_round_trip(fandhe_ai::Device::Cuda(0));
}

/// Metal 実機での round-trip（実機未実測。Mac セッションへ申し送り）。
#[test]
#[cfg(target_os = "macos")]
#[ignore = "実機（Metal）依存。Apple Silicon 実機で手動実行する"]
fn iterable_batch_upload_round_trips_on_metal_tape() {
    assert_upload_round_trip(fandhe_ai::Device::Metal);
}
