//! `functional_io` の pool／unpool ノード（`max_pool{1,2,3}d_indices`・`max_unpool{1,2,3}d`。イシュー #3037・
//! 決定記録 `docs/compat-model-io-decision.md` の #3037 追補）の改竄テスト。
//!
//! 役割: manifest 起因の拒否（件数・`layer_len`／`layer_start`・`indices_from` の種別と次元・params の
//! キー集合と型）が **safetensors を開く前**に起きることを、safetensors を削除した状態で読む
//! `load_after_edit`（`tests.rs::load_after_manifest_edit` と同型）で固定する。あわせて、未知 op が旧リーダー
//! 相当の `UnsupportedModel` で fail-closed になること・3 次元 params が `MAX_OBJECT_KEYS` 内に収まる
//! 往復・pool の kernel=0 改竄が panic せず型付きエラーになることを確認する。

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{load_functional_model, save_functional_model};
use crate::Tensor;
use crate::compat::functional::{FunctionalBuilder, FunctionalModel};
use crate::compat::{ModelIoError, Sequential};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(label: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "fandhe-functional-pool-{}-{label}-{n}",
        std::process::id()
    ))
}

struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 2D・3D の pool／unpool を両方含むグラフ。ノード: 0 入力 A・1 入力 B・2 pool2d(0)・3 pool3d(1)・
/// 4 block(2)・5 unpool2d(4, 2)・6 unpool3d(3, 3)。出力は `[5, 6]`。
fn pool_graph() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let a = b.input().expect("input a");
    let c = b.input().expect("input b");
    let p2 = b
        .max_pool2d_with_indices(a, [2, 2], None, [0, 0], [1, 1])
        .expect("pool2d");
    let p3 = b
        .max_pool3d_with_indices(c, [2, 2, 2], None, [0, 0, 0], [1, 1, 1])
        .expect("pool3d");
    let blk = b.apply(Sequential::new().add_relu(), p2).expect("block");
    let u2 = b
        .max_unpool2d(blk, p2, [2, 2], None, [0, 0], None)
        .expect("unpool2d");
    let u3 = b
        .max_unpool3d(p3, p3, [2, 2, 2], None, [0, 0, 0], None)
        .expect("unpool3d");
    b.build(&[a, c], &[u2, u3]).expect("build")
}

fn inputs() -> (Tensor<f32>, Tensor<f32>) {
    let a: Vec<f32> = (0..16).map(|k| ((k as f32) * 0.37).sin()).collect();
    let c: Vec<f32> = (0..32).map(|k| ((k as f32) * 0.21 + 0.5).sin()).collect();
    (
        Tensor::new(a, &[1, 1, 4, 4]).expect("a"),
        Tensor::new(c, &[1, 1, 2, 4, 4]).expect("b"),
    )
}

/// 保存 → manifest 編集 →（`keep_weights` が偽なら safetensors 削除）→ 読み込み。
fn load_after_edit(
    label: &str,
    edit: impl FnOnce(&str) -> String,
    keep_weights: bool,
) -> Result<FunctionalModel, ModelIoError> {
    let dir = temp_path(label);
    let _guard = Cleanup(dir.clone());
    save_functional_model(&pool_graph(), &dir).expect("save");
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).expect("read manifest");
    let edited = edit(&text);
    assert_ne!(edited, text, "{label}: 置換対象が manifest に無い\n{text}");
    std::fs::write(&path, edited).expect("write manifest");
    if !keep_weights {
        for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
            if entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
            {
                std::fs::remove_file(entry.path()).expect("remove safetensors");
            }
        }
    }
    load_functional_model(&dir)
}

fn replace_first(from: &'static str, to: &'static str) -> impl FnOnce(&str) -> String {
    move |s: &str| s.replacen(from, to, 1)
}

#[test]
fn pool_graph_round_trips_with_deterministic_manifest() {
    let mut model = pool_graph();
    model.eval();
    let dir = temp_path("roundtrip");
    let _guard = Cleanup(dir.clone());
    save_functional_model(&model, &dir).expect("save");
    let loaded = load_functional_model(&dir).expect("load");
    let (a, c) = inputs();
    let want = model.predict(&[&a, &c]).expect("predict");
    let got = loaded.predict(&[&a, &c]).expect("predict");
    for (w, g) in want.iter().zip(&got) {
        let wb: Vec<u32> = w.host_slice().iter().map(|v| v.to_bits()).collect();
        let gb: Vec<u32> = g.host_slice().iter().map(|v| v.to_bits()).collect();
        assert_eq!(wb, gb);
    }
    // 再保存で manifest 本文（世代名と safetensors_file を除く）が同一。
    let dir2 = temp_path("roundtrip2");
    let _guard2 = Cleanup(dir2.clone());
    let mut loaded = loaded;
    loaded.eval();
    save_functional_model(&loaded, &dir2).expect("resave");
    let strip = |d: &PathBuf| {
        let t = std::fs::read_to_string(d.join("manifest.json")).expect("read");
        let i = t.find("\"safetensors_file\"").expect("key");
        t[..i].to_string()
    };
    assert_eq!(strip(&dir), strip(&dir2));
}

#[test]
fn pool_node_structure_tampering_is_rejected_before_opening_safetensors() {
    let p2 = "\"op\":\"max_pool2d_indices\",\"inputs\":[0]";
    let u2 = "\"op\":\"max_unpool2d\",\"inputs\":[4,2]";
    let cases: Vec<(&str, &'static str, &'static str)> = vec![
        (
            "pool-no-input",
            p2,
            "\"op\":\"max_pool2d_indices\",\"inputs\":[]",
        ),
        (
            "pool-two-inputs",
            p2,
            "\"op\":\"max_pool2d_indices\",\"inputs\":[0,0]",
        ),
        (
            "unpool-one-input",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4]",
        ),
        (
            "unpool-three-inputs",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4,2,2]",
        ),
        (
            "pool-layer-len",
            "\"op\":\"max_pool2d_indices\",\"inputs\":[0],\"layer_start\":0,\"layer_len\":0",
            "\"op\":\"max_pool2d_indices\",\"inputs\":[0],\"layer_start\":0,\"layer_len\":1",
        ),
        (
            "pool-layer-start",
            "\"op\":\"max_pool2d_indices\",\"inputs\":[0],\"layer_start\":0",
            "\"op\":\"max_pool2d_indices\",\"inputs\":[0],\"layer_start\":1",
        ),
        // indices_from が block・input・別 unpool・次元違いの pool・前方参照・自己参照。
        (
            "indices-from-block",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4,4]",
        ),
        (
            "indices-from-input",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4,0]",
        ),
        (
            "indices-from-dim-mismatch",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4,3]",
        ),
        (
            "indices-from-unpool",
            "\"op\":\"max_unpool3d\",\"inputs\":[3,3]",
            "\"op\":\"max_unpool3d\",\"inputs\":[3,5]",
        ),
        (
            "indices-from-forward",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4,6]",
        ),
        (
            "indices-from-self",
            u2,
            "\"op\":\"max_unpool2d\",\"inputs\":[4,5]",
        ),
        // params: キー欠落（改名で欠落＋余分）・余分キー・負値・小数・文字列・bool・部分 null。
        ("param-missing", "\"kernel_size_h\"", "\"kernel_sizeX_h\""),
        (
            "param-extra",
            "\"dilation_w\":1}",
            "\"dilation_w\":1,\"extra\":1}",
        ),
        (
            "param-negative",
            "\"kernel_size_h\":2",
            "\"kernel_size_h\":-2",
        ),
        (
            "param-fraction",
            "\"kernel_size_h\":2",
            "\"kernel_size_h\":2.5",
        ),
        (
            "param-string",
            "\"kernel_size_h\":2",
            "\"kernel_size_h\":\"2\"",
        ),
        (
            "param-bool",
            "\"kernel_size_h\":2",
            "\"kernel_size_h\":true",
        ),
        (
            "param-partial-stride",
            "\"stride_h\":null",
            "\"stride_h\":1",
        ),
        (
            "param-partial-output-size",
            "\"output_size_h\":null",
            "\"output_size_h\":1",
        ),
    ];
    for (label, from, to) in cases {
        let r = load_after_edit(label, replace_first(from, to), false);
        assert!(
            matches!(r, Err(ModelIoError::Manifest { .. })),
            "{label}: Manifest 拒否のはずが {:?}",
            r.as_ref().err()
        );
    }
}

#[test]
fn unknown_unpool_op_is_unsupported_model() {
    // 旧リーダー相当（allowlist に無い op 名）は `UnsupportedModel` で fail-closed になる。
    let r = load_after_edit(
        "unknown-op",
        replace_first("\"op\":\"max_unpool2d\"", "\"op\":\"max_unpool4d\""),
        false,
    );
    assert!(
        matches!(r, Err(ModelIoError::UnsupportedModel { .. })),
        "{:?}",
        r.as_ref().err()
    );
}

#[test]
fn pool_with_zero_kernel_is_typed_error_not_panic() {
    // pool の 0 次元は構築時ではなく forward 時に `Var::max_pool*` が検出する。load で拒否されるか、
    // load が通っても predict が型付きエラーになること（panic しないこと）を固定する。
    let loaded = load_after_edit(
        "zero-kernel",
        replace_first("\"kernel_size_h\":2", "\"kernel_size_h\":0"),
        true,
    );
    if let Ok(model) = loaded {
        let (a, c) = inputs();
        assert!(model.predict(&[&a, &c]).is_err());
    }
}

#[test]
fn three_dimensional_params_stay_within_object_key_limit() {
    // 3D の params は 12 キー（`MAX_OBJECT_KEYS` 16 以内）。往復できること自体が上限内の証拠。
    let dir = temp_path("keys12");
    let _guard = Cleanup(dir.clone());
    save_functional_model(&pool_graph(), &dir).expect("save");
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("read");
    assert!(text.contains("\"kernel_size_d\""), "{text}");
    load_functional_model(&dir).expect("load");
}
