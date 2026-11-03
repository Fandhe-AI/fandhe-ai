//! 0.13.0 の `save_model` が書いた実ファイルを現行の `load_model` で読む後方互換テスト
//! （イシュー #3828・親 #3813。横断方針 #3750 の R3・R4・R9。設計記録
//! `docs/compat-new-layers-io-export-decision.md` §15）。
//!
//! fixture は `tests/fixtures/model-io-0.13.0/`（生成手順・sha256 は同 README）。0.13.0 に存在する
//! kind だけで組んだ 2 モデル（`mlp_bn`: linear / batch_norm1d / relu / dropout / layer_norm、
//! `cnn`: conv2d / batch_norm2d / relu / max_pool2d / flatten / linear）で、新しい 12 kind の追加が
//! 既存 kind のスキーマ・ファイル形式（`format_version` 2）を変えていないことを実ファイルで確かめる。
//!
//! 検証は次の 4 つ。`predict` の出力 bit は CPU カーネルのアーキテクチャ差で変わりうるため、
//! 0.13.0 側の値との比較には使わず（tolerance を持ち込まない）、同一ビルド内の比較だけに使う。
//! - (a) 現行 `load_model` が Ok で読め、`training` と `state_dict` が 0.13.0 の記録と bit 一致する。
//! - (b) 現行 `save_model` で保存し直したファイルが 0.13.0 のファイルと（世代名を除き）バイト一致する
//!   （manifest の kind・params・buffer_keys とテンソル値〈running stats を含む〉が不変）。
//! - (c) 保存し直したファイルを読み直した predict が、元の読み込みと bit 一致する。
//! - (d) 実ファイルの欠損・改竄は拒否される（非信頼入力として読まれる）。

#![cfg(unix)]

use std::path::{Path, PathBuf};

use fandhe_ai::Tensor;
use fandhe_ai::compat::{Sequential, load_model, save_model};

mod common;
use common::temp_dir::TempDirGuard;

fn fixture_dir(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/model-io-0.13.0")
        .join(case)
}

/// fixture を一時ディレクトリへコピーする（checkout 先の権限・所有者に依存させない）。
fn copy_case(case: &str, guard: &TempDirGuard) -> PathBuf {
    let dst = guard.path().join(case);
    std::fs::create_dir(&dst).expect("作れるはず");
    for entry in std::fs::read_dir(fixture_dir(case)).expect("読めるはず") {
        let entry = entry.expect("読めるはず");
        let name = entry.file_name();
        if name == "expected.json" {
            continue;
        }
        std::fs::copy(entry.path(), dst.join(&name)).expect("コピーできるはず");
    }
    dst
}

fn expected(case: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(fixture_dir(case).join("expected.json"))
        .expect("expected.json が読めるはず");
    serde_json::from_str(&text).expect("JSON のはず")
}

fn input(shape: &[usize], base: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let d: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 + base).sin()).collect();
    Tensor::new(d, shape).expect("作れるはず")
}

fn bits_of(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("連続のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn file_bytes(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .expect("読めるはず")
        .map(|e| {
            let e = e.expect("読めるはず");
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("読めるはず"),
            )
        })
        .collect();
    v.sort();
    v
}

fn check_case(case: &str, in_shape: &[usize], base: f32) {
    let guard = TempDirGuard::new(case);
    let dir = copy_case(case, &guard);
    let exp = expected(case);

    // (a) 現行リーダーで読め、パラメータが 0.13.0 の記録と bit 一致する。
    let model: Sequential = load_model(&dir).expect("0.13.0 のファイルを読めるはず");
    assert_eq!(model.training(), exp["training"].as_bool().expect("bool"));
    let sd = model.state_dict();
    let params = exp["params"].as_object().expect("object");
    assert_eq!(sd.len(), params.len(), "{case}: state_dict のキー数");
    for (key, want) in params {
        let got = sd
            .get(key)
            .unwrap_or_else(|| panic!("{case}: {key} が無い"));
        let shape: Vec<usize> = want["shape"]
            .as_array()
            .expect("array")
            .iter()
            .map(|d| d.as_u64().expect("u64") as usize)
            .collect();
        assert_eq!(got.shape(), shape.as_slice(), "{case}: {key} shape");
        let bits: Vec<u32> = want["bits"]
            .as_array()
            .expect("array")
            .iter()
            .map(|b| b.as_u64().expect("u64") as u32)
            .collect();
        assert_eq!(bits_of(got), bits, "{case}: {key} bit");
    }

    // (b) 現行リーダー＋ライターで保存し直すとバイト一致（kind・params・buffer が不変）。
    let resaved = guard.path().join(format!("{case}-resaved"));
    std::fs::create_dir(&resaved).expect("作れるはず");
    save_model(&model, &resaved).expect("保存できるはず");
    // safetensors のファイル名は保存ごとの世代名（内容ハッシュではない）なので、名前だけ正規化して比べる。
    let (old_files, new_files) = (file_bytes(&dir), file_bytes(&resaved));
    assert_eq!(
        old_files.len(),
        2,
        "{case}: manifest と safetensors の 2 ファイル"
    );
    assert_eq!(new_files.len(), 2, "{case}: 保存し直しも 2 ファイル");
    let normalize = |files: &[(String, Vec<u8>)]| -> (String, Vec<u8>) {
        let manifest = files
            .iter()
            .find(|(n, _)| n == "manifest.json")
            .expect("manifest があるはず");
        let st = files
            .iter()
            .find(|(n, _)| n.ends_with(".safetensors"))
            .expect("safetensors があるはず");
        let text = String::from_utf8(manifest.1.clone())
            .expect("UTF-8 のはず")
            .replace(&st.0, "<generation>.safetensors");
        (text, st.1.clone())
    };
    let (old_manifest, old_st) = normalize(&old_files);
    let (new_manifest, new_st) = normalize(&new_files);
    assert_eq!(old_manifest, new_manifest, "{case}: manifest");
    // バイト列は巨大になるため、一致の真偽だけを報告する。
    assert!(old_st == new_st, "{case}: safetensors がバイト不一致");
    let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
    assert!(manifest.contains("\"format_version\":2"), "{case}");

    // (c) 読み直した predict が元の読み込みと bit 一致する（eval の running stats 経路を含む）。
    let reloaded = load_model(&resaved).expect("読めるはず");
    let x = input(in_shape, base);
    let (a, b) = (
        model.predict(&x).expect("predict できるはず"),
        reloaded.predict(&x).expect("predict できるはず"),
    );
    assert_eq!(bits_of(&a), bits_of(&b), "{case}: predict bit");
    assert!(bits_of(&a).iter().all(|u| f32::from_bits(*u).is_finite()));
}

#[test]
fn mlp_bn_saved_by_0_13_0_loads_and_round_trips() {
    check_case("mlp_bn", &[5, 4], 0.1);
}

#[test]
fn cnn_saved_by_0_13_0_loads_and_round_trips() {
    check_case("cnn", &[3, 1, 4, 4], 0.2);
}

/// (d) fixture の safetensors を 1 バイト欠けさせると読み込みが拒否される。
#[test]
fn tampered_0_13_0_file_is_rejected() {
    let guard = TempDirGuard::new("tamper");
    let dir = copy_case("cnn", &guard);
    let st = std::fs::read_dir(&dir)
        .expect("読めるはず")
        .map(|e| e.expect("読めるはず").path())
        .find(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .expect("safetensors があるはず");
    let mut bytes = std::fs::read(&st).expect("読めるはず");
    // manifest の `safetensors_bytes` と食い違う長さにする。
    bytes.pop();
    std::fs::write(&st, bytes).expect("書けるはず");
    assert!(load_model(&dir).is_err());
}
