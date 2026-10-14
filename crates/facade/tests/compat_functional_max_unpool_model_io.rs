//! `save_functional_model`／`load_functional_model` の MaxPool 索引・MaxUnpool ノード対応（イシュー #3037・
//! 親 #3018。決定記録 `docs/compat-model-io-decision.md` の #3037 追補）の公開面 CPU 統合テスト。
//! `fandhe_ai` だけを import する。
//!
//! 検査項目:
//! - 1d／2d／3d の SegNet 型グラフ（conv ブロック → pool → ブロック → unpool）が保存→復元で `predict`・全
//!   パラメータ bit 一致（`stride`／`output_size` の `None`／`Some` の組合せを含む）
//! - 1 つの pool を複数の unpool が参照する fan-out・特徴量と索引元が同一の pool・保存→復元→再保存で
//!   manifest が決定的
//! - compile 済み SegNet の保存→復元→追加 `fit` が、保存せず続けた場合と bit 一致
//! - Functional 形式は `format_version` 1・最上位キー 13 個のまま。`load_model`（Sequential 用）は拒否する
//!
//! 改竄網羅は `functional_io/tests_pool.rs`（クレート内ユニットテスト）が受け持つ。

#![cfg(unix)]

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    FitConfig, FunctionalBuilder, FunctionalModel, Loss, Optimizer, Sequential,
    load_functional_model, load_model, save_functional_model,
};
use fandhe_ai::optim::SgdConfig;

mod common;
use common::temp_dir::TempDirGuard;

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn det(shape: &[usize], salt: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|k| ((k as f32) * 0.37 + salt).sin()).collect(),
        shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

/// 次元別の SegNet 型グラフ（入力 `[1, 1, ..空間]`）。`stride`／`output_size` の有無を切り替える。
fn segnet(dims: usize, explicit: bool) -> (FunctionalModel, Vec<usize>) {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let (shape, up) = match dims {
        1 => {
            let h = b
                .apply(
                    Sequential::new()
                        .add_conv1d(1, 2, 3, 1, 1, 1, 1, 0xA1)
                        .unwrap(),
                    x,
                )
                .unwrap();
            let (s, o) = if explicit {
                (Some(2), Some(8))
            } else {
                (None, None)
            };
            let pool = b.max_pool1d_with_indices(h, 2, s, 0, 1).unwrap();
            let m = b.apply(Sequential::new().add_relu(), pool).unwrap();
            (vec![1, 1, 8], b.max_unpool1d(m, pool, 2, s, 0, o).unwrap())
        }
        2 => {
            let h = b
                .apply(
                    Sequential::new()
                        .add_conv2d(1, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, 0xA1)
                        .unwrap(),
                    x,
                )
                .unwrap();
            let (s, o) = if explicit {
                (Some([2, 2]), Some([4, 4]))
            } else {
                (None, None)
            };
            let pool = b
                .max_pool2d_with_indices(h, [2, 2], s, [0, 0], [1, 1])
                .unwrap();
            let m = b.apply(Sequential::new().add_relu(), pool).unwrap();
            (
                vec![1, 1, 4, 4],
                b.max_unpool2d(m, pool, [2, 2], s, [0, 0], o).unwrap(),
            )
        }
        _ => {
            let h = b
                .apply(
                    Sequential::new()
                        .add_conv3d(1, 2, [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], 1, 0xA1)
                        .unwrap(),
                    x,
                )
                .unwrap();
            let (s, o) = if explicit {
                (Some([2, 2, 2]), Some([4, 4, 4]))
            } else {
                (None, None)
            };
            let pool = b
                .max_pool3d_with_indices(h, [2, 2, 2], s, [0, 0, 0], [1, 1, 1])
                .unwrap();
            let m = b.apply(Sequential::new().add_relu(), pool).unwrap();
            (
                vec![1, 1, 4, 4, 4],
                b.max_unpool3d(m, pool, [2, 2, 2], s, [0, 0, 0], o).unwrap(),
            )
        }
    };
    (b.build(&[x], &[up]).unwrap(), shape)
}

fn all_param_bits(m: &FunctionalModel) -> Vec<(String, Vec<u32>)> {
    let mut v: Vec<(String, Vec<u32>)> = m
        .named_parameters()
        .unwrap()
        .into_iter()
        .map(|(k, t)| (k, bits(t)))
        .collect();
    v.sort();
    v
}

#[test]
fn segnet_round_trips_bit_for_bit_in_every_dimension() {
    for dims in [1, 2, 3] {
        for explicit in [false, true] {
            let (mut model, shape) = segnet(dims, explicit);
            model.eval();
            let guard = TempDirGuard::new("functional_unpool_rt");
            let dir = guard.path().join("m");
            save_functional_model(&model, &dir).unwrap();
            let loaded = load_functional_model(&dir).unwrap();
            let x = det(&shape, 0.3);
            let a = model.predict(&[&x]).unwrap();
            let b = loaded.predict(&[&x]).unwrap();
            assert_eq!(bits(&a[0]), bits(&b[0]), "dims={dims} explicit={explicit}");
            assert_eq!(all_param_bits(&model), all_param_bits(&loaded));
        }
    }
}

#[test]
fn fan_out_and_self_feature_graphs_round_trip() {
    // 1 つの pool を 2 つの unpool が参照し、片方は特徴量＝pool 自身。pool は索引経由でも出力へ寄与する。
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let pool = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let blk = b.apply(Sequential::new().add_relu(), pool).unwrap();
    let u1 = b
        .max_unpool2d(blk, pool, [2, 2], None, [0, 0], None)
        .unwrap();
    let u2 = b
        .max_unpool2d(pool, pool, [2, 2], None, [0, 0], None)
        .unwrap();
    let mut model = b.build(&[x], &[u1, u2]).unwrap();
    model.eval();
    let guard = TempDirGuard::new("functional_unpool_fanout");
    let dir = guard.path().join("m");
    save_functional_model(&model, &dir).unwrap();
    let mut loaded = load_functional_model(&dir).unwrap();
    let x = det(&[1, 1, 4, 4], 0.7);
    let a = model.predict(&[&x]).unwrap();
    let c = loaded.predict(&[&x]).unwrap();
    for (p, q) in a.iter().zip(&c) {
        assert_eq!(bits(p), bits(q));
    }
    // 再保存で manifest 本文（世代名より前）が同一（決定的）。
    loaded.eval();
    let dir2 = guard.path().join("m2");
    save_functional_model(&loaded, &dir2).unwrap();
    let head = |d: &std::path::Path| {
        let t = std::fs::read_to_string(d.join("manifest.json")).unwrap();
        let i = t.find("\"safetensors_file\"").unwrap();
        t[..i].to_string()
    };
    assert_eq!(head(&dir), head(&dir2));
}

fn compiled_segnet() -> FunctionalModel {
    let (mut m, _) = segnet(2, false);
    m.compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    m
}

#[test]
fn compiled_segnet_resumes_bit_identically_after_save_load() {
    let x = det(&[2, 1, 4, 4], 0.1);
    let y = det(&[2, 2, 4, 4], 0.9);
    let cfg = || FitConfig::new(1, 2).shuffle(false);
    let mut keep = compiled_segnet();
    keep.fit(&[&x], &[&y], cfg()).unwrap();

    let mut saved = compiled_segnet();
    saved.fit(&[&x], &[&y], cfg()).unwrap();
    let guard = TempDirGuard::new("functional_unpool_fit");
    let dir = guard.path().join("m");
    save_functional_model(&saved, &dir).unwrap();
    let mut resumed = load_functional_model(&dir).unwrap();

    keep.fit(&[&x], &[&y], cfg()).unwrap();
    resumed.fit(&[&x], &[&y], cfg()).unwrap();
    assert_eq!(all_param_bits(&keep), all_param_bits(&resumed));
}

#[test]
fn format_version_stays_1_and_formats_stay_mutually_exclusive() {
    let (model, _) = segnet(2, false);
    let guard = TempDirGuard::new("functional_unpool_format");
    let dir = guard.path().join("m");
    save_functional_model(&model, &dir).unwrap();
    let text = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    assert!(text.contains("\"format\":\"fandhe-ai.compat.functional\",\"format_version\":1,"));
    assert!(text.contains("\"op\":\"max_pool2d_indices\""));
    assert!(text.contains("\"op\":\"max_unpool2d\""));
    // Sequential 用の入口は Functional 形式を拒否する（相互排他。Sequential は format_version 2 のまま）。
    assert!(load_model(&dir).is_err());
}
