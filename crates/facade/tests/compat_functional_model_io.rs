//! `fandhe_ai::compat::{save_functional_model, load_functional_model}`（イシュー #2679・親 #2625。承認は
//! ルート #2499 のコメント issuecomment-6033824965。`docs/facade-functional-api-decision.md` §8・§16〜§18）の
//! 公開面を CPU で検証する統合テスト。`fandhe_ai` だけを import する。
//!
//! 検査項目:
//! - 多入力・結合・多出力グラフの保存→復元で `predict`・全パラメータ・モードが bit 一致する
//! - compile 済みモデルの保存→復元→追加 `fit` が、保存せず続けた場合と bit 一致する
//! - 形式の相互排他（`load_model` は Functional のディレクトリを、`load_functional_model` は `Sequential` の
//!   ディレクトリを拒否する）
//! - 保存不能なグラフ（manifest の kind を持たない層）は `ModelIoError::UnsupportedModel` で拒否し、保存先に
//!   何も作らない
//! - 改竄した manifest（未知の op・前方参照）・存在しないディレクトリは panic せず型付きエラー
//!
//! 内部実装の網羅的な改竄テスト（構造検証の各規則・safetensors 改竄・シンボリックリンク）はクレート内
//! ユニットテスト（`functional_io/tests.rs`）が受け持つ。

#![cfg(unix)]

use std::sync::{Mutex, MutexGuard};

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    FitConfig, FunctionalBuilder, FunctionalModel, Loss, ModelIoError, Optimizer, Sequential,
    load_functional_model, load_model, save_functional_model, save_model,
};
use fandhe_ai::optim::SgdConfig;

mod common;
use common::temp_dir::TempDirGuard;

/// グローバル RNG を暗黙に消費する経路を持つテストの直列化ロック。
fn rng_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn det(rows: usize, cols: usize, salt: f32) -> Tensor<f32> {
    tensor(
        (0..rows * cols)
            .map(|k| ((k as f32) * 0.37 + salt).sin())
            .collect(),
        &[rows, cols],
    )
}

fn lin(i: usize, o: usize, seed: u64) -> Sequential {
    Sequential::new().add_linear(i, o, seed).unwrap()
}

/// 2 入力・fan-out・結合 4 種・2 出力のグラフ（ブロック 4 つ）。
fn graph() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let a = b.input().unwrap();
    let c = b.input().unwrap();
    let ba = b.apply(lin(3, 4, 41).add_relu(), a).unwrap();
    let bc = b.apply(lin(2, 4, 42).add_relu(), c).unwrap();
    let sum = b.add(&[ba, bc]).unwrap();
    let prod = b.multiply(&[ba, bc]).unwrap();
    let mean = b.average(&[sum, prod]).unwrap();
    let cat = b.concatenate(&[ba, bc], 1).unwrap();
    let o1 = b.apply(lin(4, 3, 43).add_tanh(), mean).unwrap();
    let o2 = b.apply(lin(8, 2, 44), cat).unwrap();
    b.build(&[a, c], &[o1, o2]).unwrap()
}

fn inputs() -> (Tensor<f32>, Tensor<f32>) {
    (det(6, 3, 0.1), det(6, 2, 0.7))
}

fn targets() -> (Tensor<f32>, Tensor<f32>) {
    (det(6, 3, 1.3), det(6, 2, 1.9))
}

fn assert_same_params(a: &FunctionalModel, b: &FunctionalModel) {
    let pa = a.named_parameters().unwrap();
    let pb = b.named_parameters().unwrap();
    assert_eq!(pa.len(), pb.len());
    for ((ka, ta), (kb, tb)) in pa.iter().zip(&pb) {
        assert_eq!(ka, kb);
        assert_eq!(bits(ta), bits(tb), "{ka}");
    }
}

#[test]
fn graph_round_trips_bit_for_bit() {
    let mut model = graph();
    model.eval();
    let guard = TempDirGuard::new("functional_roundtrip");
    let dir = guard.path().join("m");
    save_functional_model(&model, &dir).expect("保存できるはず");

    let loaded = load_functional_model(&dir).expect("復元できるはず");
    assert_eq!(loaded.training(), model.training());
    assert_same_params(&model, &loaded);
    let (xa, xc) = inputs();
    let want = model.predict(&[&xa, &xc]).unwrap();
    let got = loaded.predict(&[&xa, &xc]).unwrap();
    assert_eq!(want.len(), got.len());
    for (w, g) in want.iter().zip(&got) {
        assert_eq!(bits(w), bits(g));
    }
}

#[test]
fn compiled_state_round_trips_and_continued_fit_is_bit_identical() {
    let _guard = rng_lock();
    let (xa, xc) = inputs();
    let (y1, y2) = targets();
    let config = || FitConfig::new(3, 3);

    let mut original = graph();
    original
        .compile(
            Optimizer::Sgd(SgdConfig::new(0.05).with_momentum(0.9)),
            Loss::Mse,
        )
        .unwrap();
    original
        .fit(&[&xa, &xc], &[&y1, &y2], config())
        .expect("事前の学習");

    let guard = TempDirGuard::new("functional_compiled");
    let dir = guard.path().join("m");
    save_functional_model(&original, &dir).expect("保存できるはず");
    let mut resumed = load_functional_model(&dir).expect("復元できるはず");
    assert_same_params(&original, &resumed);

    // 保存せず続けた場合と、復元して続けた場合の追加学習が bit 一致する（optimizer 状態も復元される）。
    let h1 = original
        .fit(&[&xa, &xc], &[&y1, &y2], config())
        .expect("続行");
    let h2 = resumed
        .fit(&[&xa, &xc], &[&y1, &y2], config())
        .expect("復元後の続行");
    let a: Vec<u32> = h1.loss.iter().map(|v| v.to_bits()).collect();
    let b: Vec<u32> = h2.loss.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a, b);
    assert_same_params(&original, &resumed);
}

#[test]
fn formats_are_mutually_exclusive() {
    let guard = TempDirGuard::new("functional_exclusive");

    // Functional のディレクトリを `load_model` は拒否する。
    let fdir = guard.path().join("functional");
    save_functional_model(&graph(), &fdir).expect("保存できるはず");
    assert!(load_model(&fdir).is_err());

    // `Sequential` のディレクトリを `load_functional_model` は拒否する。
    let sdir = guard.path().join("sequential");
    save_model(&lin(3, 2, 7).add_relu(), &sdir).expect("保存できるはず");
    assert!(load_functional_model(&sdir).is_err());
    // どちらも元の形式では復元できる。
    assert!(load_functional_model(&fdir).is_ok());
    assert!(load_model(&sdir).is_ok());
}

#[test]
fn unsupported_layer_is_rejected_without_touching_dir() {
    // `add_selu` は manifest の kind を持たない（保存不能。`Sequential` と同じ扱い）。
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let y = b.apply(lin(3, 2, 5).add_selu(), x).unwrap();
    let model = b.build(&[x], &[y]).unwrap();

    let guard = TempDirGuard::new("functional_unsupported");
    let dir = guard.path().join("m");
    let err = save_functional_model(&model, &dir).unwrap_err();
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err:?}"
    );
    assert!(!dir.exists(), "拒否時は保存先に何も作らない");
}

#[test]
fn load_rejects_missing_dir_and_tampered_manifests_without_panicking() {
    let guard = TempDirGuard::new("functional_tamper");
    assert!(load_functional_model(guard.path().join("missing")).is_err());

    let dir = guard.path().join("m");
    save_functional_model(&graph(), &dir).expect("保存できるはず");
    let path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&path).unwrap();

    // 未知の op 名・前方参照・入力添字の範囲外は、safetensors を開く前に型付きエラーで拒否される。
    let mut tampered = 0usize;
    for (from, to) in [
        ("\"op\":\"add\"", "\"op\":\"subtract\""),
        ("\"op\":\"concatenate\"", "\"op\":\"__rogue__\""),
        ("\"inputs\":[0]", "\"inputs\":[99]"),
        ("\"format_version\":1", "\"format_version\":2"),
    ] {
        if !original.contains(from) {
            continue;
        }
        tampered += 1;
        std::fs::write(&path, original.replacen(from, to, 1)).unwrap();
        assert!(
            load_functional_model(&dir).is_err(),
            "改竄 {from} -> {to} が受理された"
        );
    }
    assert!(
        tampered >= 2,
        "manifest の改竄対象が見つからない: {original}"
    );

    // 元に戻せば復元できる（改竄テストが保存形式を壊していないことの確認）。
    std::fs::write(&path, &original).unwrap();
    assert!(load_functional_model(&dir).is_ok());
}
