//! `Subset`・`ConcatDataset`・`random_split`（イシュー #2661・親 #2660）の
//! 統合テスト。
//!
//! これらは **facade 非公開**（承認待ち #2677・公開は #2679。
//! `docs/tensor-core-dataset-compose-decision.md`）のため、内部クレート
//! `fandhe_ai_tensor_core::data` を直接 import する（#2182 時点の
//! `data_sampler_hooks.rs` と同型）。facade 公開後は import を
//! `fandhe_ai::data` へ切り替える。
//!
//! 実 PyTorch 2.14.0 の実行値 fixture
//! （`crates/tensor-core/tests/fixtures/dataset-compose-pytorch-reference/`）と突合する。
//! 整数表（長さ列・`cumulative_sizes`・添字）は完全一致、f32 の行は bit 一致を
//! 確認したうえで統一複合判定（`assert_parity`）にも通す。乱数列そのもの
//! （`randperm`）は本リポの RNG と一致しないため比較対象外。

use std::sync::Mutex;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::data::{DataError, DataLoader, DataLoaderConfig, Dataset, TensorDataset};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai_backend_cpu::assert_parity;
use fandhe_ai_tensor_core::data::{ConcatDataset, Subset, random_split, random_split_fractions};
use serde_json::Value;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tensor-core/tests/fixtures/dataset-compose-pytorch-reference/dataset_compose_reference.json"
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

/// `{shape, bits}` レコードから f32 テンソルを復元する（bit 保存のため再生成しない）。
fn tensor_of(rec: &Value) -> Tensor<f32> {
    let shape = usizes(&rec["shape"]);
    let data: Vec<f32> = u64s(&rec["bits"])
        .into_iter()
        .map(|b| f32::from_bits(b as u32))
        .collect();
    Tensor::new(data, &shape).unwrap_or_else(|e| panic!("test fixture: tensor 構築に失敗: {e}"))
}

fn dataset_of(rec: &Value) -> TensorDataset<f32> {
    TensorDataset::new(tensor_of(rec))
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset 構築に失敗: {e}"))
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

#[test]
fn fixture_provenance_is_real_pytorch() {
    let f = fixture();
    let v = f["torch_version"].as_str().unwrap_or("");
    assert!(
        v.starts_with("2.14.0"),
        "実 PyTorch 2.14.0 系の実行値のはず: {v}"
    );
    assert_eq!(f["split_structure_verified"], Value::Bool(true));
}

#[test]
fn fraction_lengths_match_pytorch() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let f = fixture();
    for case in arr(&f, "fraction_cases") {
        let n = case["n"].as_u64().unwrap_or(0) as usize;
        let fractions: Vec<f64> = u64s(&case["fractions_bits"])
            .into_iter()
            .map(f64::from_bits)
            .collect();
        let data = Tensor::new(vec![0.0f32; n], &[n, 1]).unwrap();
        let ds = TensorDataset::new(data).unwrap();
        let ctx = format!("n={n} fractions={fractions:?}");
        let result = random_split_fractions(ds, &fractions);
        if case["torch_raises"].as_bool() == Some(true) {
            assert!(result.is_err(), "{ctx}: torch が例外なら本実装も Err");
        } else {
            let lens: Vec<usize> = result
                .unwrap_or_else(|e| panic!("{ctx}: torch が成功なら Ok のはず: {e}"))
                .iter()
                .map(|s| s.len())
                .collect();
            assert_eq!(lens, usizes(&case["lengths"]), "{ctx}");
        }
    }
}

/// 意図的差異: torch は合計が 1 に近い整数列を割合として解釈する。本 API は型で
/// 経路を分けるため、整数長版は合計一致のみを要求する（決定記録 §3）。
#[test]
fn int_split_lengths_match_pytorch() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let f = fixture();
    for case in arr(&f, "int_split_cases") {
        let n = case["n"].as_u64().unwrap_or(0) as usize;
        let lengths = usizes(&case["lengths"]);
        let data = Tensor::new(vec![0.0f32; n], &[n, 1]).unwrap();
        let result = random_split(TensorDataset::new(data).unwrap(), &lengths);
        let ctx = format!("n={n} lengths={lengths:?}");
        if case["torch_raises"].as_bool() == Some(true) {
            assert!(
                matches!(result, Err(DataError::SplitLengthMismatch { .. })),
                "{ctx}"
            );
        } else {
            let parts = result.unwrap_or_else(|e| panic!("{ctx}: Ok のはず: {e}"));
            let got: Vec<usize> = parts.iter().map(|s| s.len()).collect();
            assert_eq!(got, lengths, "{ctx}");
            let mut all: Vec<usize> = parts.iter().flat_map(|s| s.indices().to_vec()).collect();
            all.sort_unstable();
            assert_eq!(all, (0..n).collect::<Vec<_>>(), "{ctx}: 順列の分割");
        }
    }
}

#[test]
fn subset_rows_match_pytorch() {
    let f = fixture();
    for case in arr(&f, "subset_cases") {
        let name = case["name"].as_str().unwrap_or("?");
        let base = dataset_of(&case["base"]);
        let outer = usizes(&case["outer_indices"]);
        let subset = if case["inner_indices"].is_null() {
            Subset::new(base, outer).unwrap()
        } else {
            let inner = Subset::new(base, usizes(&case["inner_indices"])).unwrap();
            // nested は Subset<Subset<_>>。型が異なるため別経路で検証する。
            let nested = Subset::new(inner, outer).unwrap();
            assert_eq!(
                nested.len() as u64,
                case["len"].as_u64().unwrap_or(u64::MAX)
            );
            let idx: Vec<usize> = (0..nested.len()).collect();
            assert_rows(
                name,
                &nested.batch(&idx).unwrap(),
                &tensor_of(&case["rows"]),
            );
            continue;
        };
        assert_eq!(
            subset.len() as u64,
            case["len"].as_u64().unwrap_or(u64::MAX),
            "{name}"
        );
        let idx: Vec<usize> = (0..subset.len()).collect();
        assert_rows(
            name,
            &subset.batch(&idx).unwrap(),
            &tensor_of(&case["rows"]),
        );
    }
}

#[test]
fn concat_rows_match_pytorch() {
    let f = fixture();
    for case in arr(&f, "concat_cases") {
        let name = case["name"].as_str().unwrap_or("?");
        match name {
            "plain" => {
                let parts: Vec<TensorDataset<f32>> =
                    arr(case, "parts").iter().map(dataset_of).collect();
                let cat = ConcatDataset::new(parts).unwrap();
                assert_eq!(cat.cumulative_sizes(), usizes(&case["cumulative_sizes"]));
                for q in arr(case, "queries") {
                    let qn = q["name"].as_str().unwrap_or("?");
                    let got = cat.batch(&usizes(&q["indices"])).unwrap();
                    assert_rows(&format!("{name}/{qn}"), &got, &tensor_of(&q["rows"]));
                }
            }
            "tuple_f32_i32" => {
                let parts: Vec<(TensorDataset<f32>, TensorDataset<i32>)> = arr(case, "xs")
                    .iter()
                    .zip(arr(case, "ys"))
                    .map(|(x, y)| {
                        let ys: Vec<i32> = y
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_i64().unwrap() as i32)
                            .collect();
                        let n = ys.len();
                        (
                            dataset_of(x),
                            TensorDataset::new(Tensor::new(ys, &[n]).unwrap()).unwrap(),
                        )
                    })
                    .collect();
                let cat = ConcatDataset::new(parts).unwrap();
                assert_eq!(cat.cumulative_sizes(), usizes(&case["cumulative_sizes"]));
                let (x, y) = cat.batch(&usizes(&case["indices"])).unwrap();
                assert_rows(name, &x, &tensor_of(&case["x_rows"]));
                let expected_y: Vec<i32> = case["y_rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect();
                assert_eq!(y.host_slice().as_ref(), expected_y.as_slice(), "{name}: y");
            }
            "subset_of_subsets" => {
                let base = std::sync::Arc::new(dataset_of(&case["base"]));
                let subs: Vec<Subset<TensorDataset<f32>>> = arr(case, "subset_indices")
                    .iter()
                    .map(|i| Subset::from_shared(base.clone(), usizes(i)).unwrap())
                    .collect();
                let cat = ConcatDataset::new(subs).unwrap();
                assert_eq!(cat.cumulative_sizes(), usizes(&case["cumulative_sizes"]));
                let got = cat.batch(&usizes(&case["indices"])).unwrap();
                assert_rows(name, &got, &tensor_of(&case["rows"]));
            }
            other => panic!("未知の concat ケース: {other}"),
        }
    }
}

/// torch の成否との一対一照合。意図的差異は `INTENDED_DIFFS`（決定記録 §3 と対応）。
#[test]
fn error_cases_match_pytorch_modulo_intended_differences() {
    // subset_oob_at_access: torch は参照時に遅延失敗、本実装は構築時に拒否する。
    const INTENDED_DIFFS: &[&str] = &["subset_oob_at_access"];
    let f = fixture();
    for case in arr(&f, "error_cases") {
        let name = case["name"].as_str().unwrap_or("?");
        assert_eq!(case["torch_raises"].as_bool(), Some(true), "{name}");
        match name {
            "empty_concat" => assert_eq!(
                ConcatDataset::<TensorDataset<f32>>::new(vec![]).unwrap_err(),
                DataError::EmptyConcat
            ),
            "subset_oob_at_access" => {
                assert!(INTENDED_DIFFS.contains(&name));
                let ds =
                    TensorDataset::new(Tensor::new(vec![0.0f32; 3], &[3, 1]).unwrap()).unwrap();
                assert!(matches!(
                    Subset::new(ds, vec![0, 5]),
                    Err(DataError::IndexOutOfRange { index: 5, len: 3 })
                ));
            }
            "concat_oob" => {
                let ds =
                    TensorDataset::new(Tensor::new(vec![0.0f32; 2], &[2, 1]).unwrap()).unwrap();
                let cat = ConcatDataset::new(vec![ds]).unwrap();
                assert!(matches!(
                    cat.batch(&[2]),
                    Err(DataError::IndexOutOfRange { index: 2, len: 2 })
                ));
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

/// `random_split` した train／val をローダーへ渡して学習し、loss が減少する。
#[test]
fn training_on_random_split_converges_and_val_is_disjoint() {
    let _g = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    fandhe_ai::manual_seed(2661);
    let (x, y) = gen_xy(0xFEED);
    let dataset = (
        TensorDataset::new(x).unwrap(),
        TensorDataset::new(y).unwrap(),
    );
    let mut parts = random_split(dataset, &[16, 8]).unwrap().into_iter();
    let (train, val) = (parts.next().unwrap(), parts.next().unwrap());
    let train_set: std::collections::HashSet<usize> = train.indices().iter().copied().collect();
    assert!(val.indices().iter().all(|i| !train_set.contains(i)));

    let val_loader = DataLoader::new(val, DataLoaderConfig::new(4)).unwrap();
    assert_eq!(val_loader.len(), 2);
    let loader = DataLoader::new(train, DataLoaderConfig::new(4).shuffle(true)).unwrap();

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

/// `ConcatDataset` を `DataLoader` へ渡し、全行が 1 回ずつ供給される。
#[test]
fn concat_dataset_feeds_data_loader() {
    let (x, _) = gen_xy(0xABCD);
    let a = x.narrow(0, 0, 10).unwrap().contiguous();
    let b = x.narrow(0, 10, 14).unwrap().contiguous();
    let cat = ConcatDataset::new(vec![
        TensorDataset::new(a).unwrap(),
        TensorDataset::new(b).unwrap(),
    ])
    .unwrap();
    let loader = DataLoader::new(cat, DataLoaderConfig::new(5)).unwrap();
    let mut got: Vec<f32> = Vec::new();
    for batch in loader.iter() {
        got.extend_from_slice(&batch.unwrap().host_slice());
    }
    assert_eq!(got, x.host_slice().to_vec(), "順序保存で元テンソルと一致");
}

/// CUDA 実機での `Tape::var` アップロード round-trip（実機未実測。GB10 へ申し送り）。
#[test]
#[ignore = "実機（CUDA）依存。DGX Spark GB10 等で手動実行する"]
fn composed_batch_upload_round_trips_on_cuda_tape() {
    let (x, _) = gen_xy(0x1234);
    let parts = random_split(TensorDataset::new(x).unwrap(), &[16, 8]).unwrap();
    let tape = fandhe_ai::tape_for(fandhe_ai::Device::Cuda(0))
        .unwrap_or_else(|e| panic!("test fixture: CUDA tape_for が失敗した: {e}"));
    for s in parts {
        let idx: Vec<usize> = (0..s.len()).collect();
        let batch = s.batch(&idx).unwrap();
        let uploaded = tape.var(&batch).to_tensor();
        assert_eq!(uploaded.host_slice().as_ref(), batch.host_slice().as_ref());
    }
}

/// Metal 実機での round-trip（実機未実測。Mac セッションへ申し送り）。
#[test]
#[cfg(target_os = "macos")]
#[ignore = "実機（Metal）依存。Apple Silicon 実機で手動実行する"]
fn composed_batch_upload_round_trips_on_metal_tape() {
    let (x, _) = gen_xy(0x1234);
    let parts = random_split(TensorDataset::new(x).unwrap(), &[16, 8]).unwrap();
    let tape = fandhe_ai::tape_for(fandhe_ai::Device::Metal)
        .unwrap_or_else(|e| panic!("test fixture: Metal tape_for が失敗した: {e}"));
    for s in parts {
        let idx: Vec<usize> = (0..s.len()).collect();
        let batch = s.batch(&idx).unwrap();
        let uploaded = tape.var(&batch).to_tensor();
        assert_eq!(uploaded.host_slice().as_ref(), batch.host_slice().as_ref());
    }
}
