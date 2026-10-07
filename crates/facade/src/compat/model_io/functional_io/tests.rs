//! `functional_io`（Functional モデルの保存・復元。イシュー #2667）のクレート内ユニットテスト。
//!
//! 構成: (1) 多入力・fan-out・結合 4 種・多出力・BatchNorm・Dropout を含むグラフの往復（`predict` と
//! 全パラメータの bit 一致）・(2) compile 済み 6 optimizer の保存 → 復元 → 追加 `fit` が保存せず続けた場合と
//! bit 一致・(3) 拒否系で `dir` に何も残らない・(4) 形式の相互排他・(5) 構造検証の各規則の改竄テスト
//! （safetensors を置かない状態で manifest 起因のエラーになること＝開く前に拒否した証拠）・
//! (6) safetensors の改竄・シンボリックリンクの拒否・(7) 深さ・配列長の見積りの実証。
//!
//! テスト関数・ヘルパーの名前は `tests/api_surface.rs` の workspace 走査が数える名前
//! （`state_dict` 等）と衝突させない。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{load_functional_model, save_functional_model};
use crate::Tensor;
use crate::compat::functional::{FunctionalBuilder, FunctionalModel};
use crate::compat::{FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model};
use crate::optim::{
    AdagradConfig, AdamConfig, AdamWConfig, LambConfig, LbfgsConfig, RmsPropConfig, SgdConfig,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// テスト専用の一時ディレクトリの**パス**（作らない。`save_*` が作る）。
fn temp_path(label: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "fandhe-functional-io-{}-{label}-{n}",
        std::process::id()
    ))
}

/// 破棄時にディレクトリを削除するガード。
struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture tensor")
}

fn det_data(rows: usize, cols: usize, salt: f32) -> Tensor<f32> {
    let data = (0..rows * cols)
        .map(|k| ((k as f32) * 0.37 + salt).sin())
        .collect();
    t(data, &[rows, cols])
}

fn bits_of(x: &Tensor<f32>) -> Vec<u32> {
    x.host_slice().iter().map(|v| v.to_bits()).collect()
}

fn lin(i: usize, o: usize, seed: u64) -> Sequential {
    Sequential::new().add_linear(i, o, seed).expect("linear")
}

/// 2 入力・fan-out・結合 4 種・2 出力・BatchNorm（2 番目のブロック）・Dropout を含むグラフ。
/// ブロックは 4 つで、BatchNorm の通し番号キーが非ゼロの `layer_start` を持つ。
fn rich_graph(with_dropout: bool) -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let a = b.input().expect("input a");
    let c = b.input().expect("input b");
    let ba = b.apply(lin(3, 4, 41).add_relu(), a).expect("apply");
    let bb = b
        .apply(
            lin(2, 4, 42)
                .add_batch_norm1d(4, 1e-5, 0.1)
                .expect("bn")
                .add_relu(),
            c,
        )
        .expect("apply");
    let sum = b.add(&[ba, bb]).expect("add");
    let prod = b.multiply(&[ba, bb]).expect("multiply");
    let mean = b.average(&[sum, prod]).expect("average");
    let cat = b.concatenate(&[ba, bb], 1).expect("concatenate");
    let mut head = lin(4, 3, 43);
    if with_dropout {
        head = head.add_dropout(0.2).expect("dropout");
    }
    let o1 = b.apply(head.add_tanh(), mean).expect("apply");
    let o2 = b.apply(lin(8, 2, 44), cat).expect("apply");
    b.build(&[a, c], &[o1, o2]).expect("build")
}

fn rich_inputs() -> (Tensor<f32>, Tensor<f32>) {
    (det_data(6, 3, 0.1), det_data(6, 2, 0.7))
}

fn rich_targets() -> (Tensor<f32>, Tensor<f32>) {
    (det_data(6, 3, 1.3), det_data(6, 2, 1.9))
}

/// 2 モデルが同じ通し番号キー・パラメータ値・モード・eval 出力を持つこと。
fn assert_same_model(a: &mut FunctionalModel, b: &mut FunctionalModel) {
    assert_eq!(a.training(), b.training());
    let pa = a.named_parameters().expect("named");
    let pb = b.named_parameters().expect("named");
    assert_eq!(pa.len(), pb.len());
    for ((ka, ta), (kb, tb)) in pa.iter().zip(&pb) {
        assert_eq!(ka, kb);
        assert_eq!(bits_of(ta), bits_of(tb), "{ka}");
    }
    let (xa, xb) = rich_inputs();
    a.eval();
    b.eval();
    let oa = a.predict(&[&xa, &xb]).expect("predict");
    let ob = b.predict(&[&xa, &xb]).expect("predict");
    for (u, v) in oa.iter().zip(&ob) {
        assert_eq!(bits_of(u), bits_of(v));
    }
}

fn is_manifest_err<T>(r: &Result<T, ModelIoError>) -> bool {
    matches!(r, Err(ModelIoError::Manifest { .. }))
}

// ------------------------------------------------------------------ (1) 往復

#[test]
fn rich_graph_round_trips_bit_for_bit() {
    let _guard = crate::compat::global_rng_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let dir = temp_path("roundtrip");
    let _guard = Cleanup(dir.clone());
    let mut model = rich_graph(true);
    // BatchNorm の running stats を初期値から動かしてから保存する（buffer の往復を実証）。
    let (xa, xb) = rich_inputs();
    model.predict(&[&xa, &xb]).expect("train-mode predict");
    save_functional_model(&model, &dir).expect("save");
    let mut loaded = load_functional_model(&dir).expect("load");
    assert_eq!(loaded.training(), model.training());
    assert_same_model(&mut model, &mut loaded);

    // BatchNorm の running stats（buffer）も bit 一致で復元される。
    let stats = |m: &FunctionalModel| -> Vec<Vec<u32>> {
        m.blocks()
            .flat_map(|(_, b)| {
                b.layers().iter().filter_map(|l| {
                    l.as_batch_norm1d().map(|bn| {
                        let mut v = bits_of(&bn.running_mean());
                        v.extend(bits_of(&bn.running_var()));
                        v
                    })
                })
            })
            .collect()
    };
    assert_eq!(stats(&model), stats(&loaded));
    assert!(
        stats(&model)
            .iter()
            .flatten()
            .any(|b| *b != 0f32.to_bits() && *b != 1f32.to_bits()),
        "running stats が初期値から動いている前提"
    );
}

#[test]
fn eval_mode_flag_is_restored() {
    let dir = temp_path("evalflag");
    let _guard = Cleanup(dir.clone());
    let mut model = rich_graph(false);
    model.eval();
    save_functional_model(&model, &dir).expect("save");
    let loaded = load_functional_model(&dir).expect("load");
    assert!(!loaded.training());
}

#[test]
fn merge_only_graph_without_blocks_round_trips() {
    let dir = temp_path("noblocks");
    let _guard = Cleanup(dir.clone());
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.input().expect("input");
    let s = b.add(&[x, y]).expect("add");
    let c = b.concatenate(&[s, y], 1).expect("concat");
    let model = b.build(&[x, y], &[c]).expect("build");
    save_functional_model(&model, &dir).expect("save");
    let loaded = load_functional_model(&dir).expect("load");
    let (xa, xb) = (det_data(3, 2, 0.2), det_data(3, 2, 0.9));
    let want = model.predict(&[&xa, &xb]).expect("predict");
    let got = loaded.predict(&[&xa, &xb]).expect("predict");
    assert_eq!(bits_of(&want[0]), bits_of(&got[0]));
    assert!(loaded.training());
}

/// 入力順（`build` に渡した順）と出力順が往復で保たれる。
#[test]
fn input_and_output_order_survive_round_trip() {
    let dir = temp_path("order");
    let _guard = Cleanup(dir.clone());
    let mut b = FunctionalBuilder::new();
    let x0 = b.input().expect("x0");
    let x1 = b.input().expect("x1");
    let p = b.apply(lin(2, 2, 5), x0).expect("apply");
    let q = b.apply(lin(3, 2, 6), x1).expect("apply");
    // 入力は逆順、出力も逆順で `build` する。
    let model = b.build(&[x1, x0], &[q, p]).expect("build");
    save_functional_model(&model, &dir).expect("save");
    let loaded = load_functional_model(&dir).expect("load");
    let (i1, i0) = (det_data(4, 3, 0.1), det_data(4, 2, 0.4));
    let want = model.predict(&[&i1, &i0]).expect("predict");
    let got = loaded.predict(&[&i1, &i0]).expect("predict");
    for (u, v) in want.iter().zip(&got) {
        assert_eq!(bits_of(u), bits_of(v));
    }
}

// --------------------------------------- (2) compile 済みの保存 → 復元 → 追加 fit

fn six_optimizers() -> Vec<(&'static str, Optimizer)> {
    vec![
        (
            "sgd",
            Optimizer::Sgd(SgdConfig::new(0.05).with_momentum(0.9)),
        ),
        (
            "adamw",
            Optimizer::AdamW(AdamWConfig {
                lr: 0.01,
                ..AdamWConfig::default()
            }),
        ),
        (
            "adam",
            Optimizer::Adam(AdamConfig {
                lr: 0.01,
                ..AdamConfig::default()
            }),
        ),
        (
            "rmsprop",
            Optimizer::RmsProp(RmsPropConfig {
                lr: 0.02,
                alpha: 0.9,
                eps: 1e-8,
                weight_decay: 0.01,
                momentum: 0.1,
                centered: true,
            }),
        ),
        (
            "adagrad",
            Optimizer::Adagrad(AdagradConfig {
                lr: 0.1,
                lr_decay: 0.01,
                weight_decay: 0.01,
                initial_accumulator_value: 0.0,
                eps: 1e-10,
            }),
        ),
        (
            "lamb",
            Optimizer::Lamb(LambConfig {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-6,
                weight_decay: 0.01,
            }),
        ),
    ]
}

#[test]
fn compiled_state_resumes_bit_identically_for_every_optimizer() {
    let (xa, xb) = rich_inputs();
    let (y1, y2) = rich_targets();
    for (name, optimizer) in six_optimizers() {
        let dir = temp_path(name);
        let _guard = Cleanup(dir.clone());
        // Dropout なし（global RNG を使わず決定的）。BatchNorm は train モードで running stats を更新する。
        let mut continued = rich_graph(false);
        continued.compile(optimizer, Loss::Mse).expect("compile");
        continued
            .fit(&[&xa, &xb], &[&y1, &y2], FitConfig::new(2, 3))
            .expect("fit");
        save_functional_model(&continued, &dir).expect("save");
        let mut restored = load_functional_model(&dir).expect("load");
        assert!(restored.is_compiled(), "{name}");

        let h_cont = continued
            .fit(&[&xa, &xb], &[&y1, &y2], FitConfig::new(2, 3))
            .expect("fit");
        let h_rest = restored
            .fit(&[&xa, &xb], &[&y1, &y2], FitConfig::new(2, 3))
            .expect("fit");
        let hb = |h: &crate::compat::History| -> Vec<u32> {
            h.loss.iter().map(|v| v.to_bits()).collect()
        };
        assert_eq!(hb(&h_cont), hb(&h_rest), "{name}: 損失履歴");
        assert_same_model(&mut continued, &mut restored);
    }
}

// ----------------------------------------------------------------- (3) 拒否系

/// 失敗した保存が `dir` を作っていないこと。
fn assert_nothing_written(dir: &Path) {
    assert!(
        !dir.exists(),
        "dir に何も残してはならない: {}",
        dir.display()
    );
}

#[test]
fn save_rejects_custom_layers_and_mode_mismatch_without_touching_dir() {
    // add_module 由来の層。
    struct Passthrough;
    impl crate::nn::Module for Passthrough {
        fn forward<'t>(
            &self,
            _tape: crate::TapeRef<'t>,
            input: &crate::Var<'t>,
        ) -> Result<crate::Var<'t>, crate::AutodiffError> {
            Ok(*input)
        }
    }
    let dir = temp_path("reject-module");
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b
        .apply(Sequential::new().add_module(Passthrough), x)
        .expect("apply");
    let model = b.build(&[x], &[y]).expect("build");
    let r = save_functional_model(&model, &dir);
    assert!(
        matches!(r, Err(ModelIoError::UnsupportedModel { .. })),
        "{r:?}"
    );
    assert_nothing_written(&dir);

    // ブロック間のモード不一致。
    let dir = temp_path("reject-mode");
    let mut model = rich_graph(false);
    if let Some((_, block)) = model
        .nodes
        .iter_mut()
        .filter_map(|d| match d {
            crate::compat::functional::NodeDef::Block { block, .. } => Some(((), block)),
            _ => None,
        })
        .next()
    {
        block.set_training(false);
    }
    let r = save_functional_model(&model, &dir);
    assert!(
        matches!(r, Err(ModelIoError::UnsupportedModel { .. })),
        "{r:?}"
    );
    assert_nothing_written(&dir);
}

#[test]
fn save_rejects_too_many_layers_and_too_many_nodes_without_touching_dir() {
    // 総層数 > MAX_LAYERS（4096）。
    let dir = temp_path("reject-layers");
    let mut block = Sequential::new();
    for _ in 0..(super::MAX_LAYERS + 1) {
        block = block.add_relu();
    }
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("input");
    let y = b.apply(block, x).expect("apply");
    let model = b.build(&[x], &[y]).expect("build");
    let r = save_functional_model(&model, &dir);
    assert!(matches!(r, Err(ModelIoError::TooLarge { .. })), "{r:?}");
    assert_nothing_written(&dir);

    // ノード数 > MAX_ARRAY_LEN（配列長上限）。ブロックなしの結合チェーン。
    let dir = temp_path("reject-nodes");
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("x");
    let y = b.input().expect("y");
    let mut cur = b.add(&[x, y]).expect("add");
    for _ in 0..(super::MAX_ARRAY_LEN + 8) {
        cur = b.add(&[cur, y]).expect("add");
    }
    let model = b.build(&[x, y], &[cur]).expect("build");
    let r = save_functional_model(&model, &dir);
    assert!(matches!(r, Err(ModelIoError::TooLarge { .. })), "{r:?}");
    assert_nothing_written(&dir);
}

// ------------------------------------------------------ (4) 形式の相互排他

#[test]
fn formats_are_mutually_exclusive() {
    let fdir = temp_path("excl-fn");
    let _g1 = Cleanup(fdir.clone());
    save_functional_model(&rich_graph(false), &fdir).expect("save");
    assert!(
        is_manifest_err(&load_model(&fdir)),
        "既存 load_model は Functional 形式を Manifest で拒否する"
    );

    let sdir = temp_path("excl-seq");
    let _g2 = Cleanup(sdir.clone());
    save_model(&lin(2, 2, 3).add_relu(), &sdir).expect("save_model");
    assert!(
        is_manifest_err(&load_functional_model(&sdir)),
        "load_functional_model は Sequential 形式を Manifest で拒否する"
    );
}

// ----------------------------------------- (5) 構造検証の改竄（safetensors を置かない）

/// 2 入力・ブロック 2 つ・加算 1 つの小グラフ。manifest のノード列は
/// 0 input / 1 input / 2 block(inputs [0]) / 3 block(inputs [1]) / 4 add(inputs [2,3])。
fn tamper_graph() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let a = b.input().expect("a");
    let c = b.input().expect("b");
    let x = b.apply(lin(2, 2, 7), a).expect("apply");
    let y = b.apply(lin(2, 2, 8), c).expect("apply");
    let s = b.add(&[x, y]).expect("add");
    b.build(&[a, c], &[s]).expect("build")
}

/// `tamper_graph` を保存し、manifest の `from` を `to` へ置換して、**safetensors を削除した**状態で
/// 読み込んだ結果を返す（構造起因の拒否が safetensors を開く前に起きる証拠。開けば `Io` になる）。
fn load_after_manifest_edit(
    label: &str,
    from: &str,
    to: &str,
) -> Result<FunctionalModel, ModelIoError> {
    let dir = temp_path(label);
    let _guard = Cleanup(dir.clone());
    save_functional_model(&tamper_graph(), &dir).expect("save");
    let manifest_path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&manifest_path).expect("read manifest");
    assert!(
        text.contains(from),
        "置換対象が manifest に無い: {from}\n{text}"
    );
    std::fs::write(&manifest_path, text.replacen(from, to, 1)).expect("write manifest");
    for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
        if entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            std::fs::remove_file(entry.path()).expect("remove safetensors");
        }
    }
    load_functional_model(&dir)
}

#[test]
fn structural_tampering_is_rejected_before_opening_safetensors() {
    // (置換前, 置換後)。いずれも `Manifest` で拒否される。
    let manifest_cases: &[(&str, &str, &str)] = &[
        (
            "index-gap",
            "\"index\":3,\"op\":\"block\"",
            "\"index\":4,\"op\":\"block\"",
        ),
        (
            "self-reference",
            "\"index\":2,\"op\":\"block\",\"inputs\":[0]",
            "\"index\":2,\"op\":\"block\",\"inputs\":[2]",
        ),
        (
            "forward-reference",
            "\"index\":2,\"op\":\"block\",\"inputs\":[0]",
            "\"index\":2,\"op\":\"block\",\"inputs\":[4]",
        ),
        (
            "block-arity",
            "\"index\":2,\"op\":\"block\",\"inputs\":[0]",
            "\"index\":2,\"op\":\"block\",\"inputs\":[0,1]",
        ),
        (
            "merge-one-input",
            "\"op\":\"add\",\"inputs\":[2,3]",
            "\"op\":\"add\",\"inputs\":[2]",
        ),
        (
            "merge-duplicate-input",
            "\"op\":\"add\",\"inputs\":[2,3]",
            "\"op\":\"add\",\"inputs\":[2,2]",
        ),
        (
            "input-with-sources",
            "\"index\":1,\"op\":\"input\",\"inputs\":[]",
            "\"index\":1,\"op\":\"input\",\"inputs\":[0]",
        ),
        (
            "layer-start-gap",
            "\"index\":3,\"op\":\"block\",\"inputs\":[1],\"layer_start\":1",
            "\"index\":3,\"op\":\"block\",\"inputs\":[1],\"layer_start\":0",
        ),
        (
            "block-without-layers",
            "\"index\":3,\"op\":\"block\",\"inputs\":[1],\"layer_start\":1,\"layer_len\":1",
            "\"index\":3,\"op\":\"block\",\"inputs\":[1],\"layer_start\":1,\"layer_len\":0",
        ),
        (
            "layer-overrun",
            "\"index\":3,\"op\":\"block\",\"inputs\":[1],\"layer_start\":1,\"layer_len\":1",
            "\"index\":3,\"op\":\"block\",\"inputs\":[1],\"layer_start\":1,\"layer_len\":2",
        ),
        (
            "merge-with-layers",
            "\"index\":4,\"op\":\"add\",\"inputs\":[2,3],\"layer_start\":2,\"layer_len\":0",
            "\"index\":4,\"op\":\"add\",\"inputs\":[2,3],\"layer_start\":2,\"layer_len\":1",
        ),
        (
            "unlisted-input",
            "\"inputs\":[0,1],\"outputs\"",
            "\"inputs\":[0],\"outputs\"",
        ),
        (
            "duplicate-input",
            "\"inputs\":[0,1],\"outputs\"",
            "\"inputs\":[0,0],\"outputs\"",
        ),
        (
            "non-input-listed",
            "\"inputs\":[0,1],\"outputs\"",
            "\"inputs\":[0,2],\"outputs\"",
        ),
        (
            "empty-inputs",
            "\"inputs\":[0,1],\"outputs\"",
            "\"inputs\":[],\"outputs\"",
        ),
        ("outputs-out-of-range", "\"outputs\":[4]", "\"outputs\":[9]"),
        ("outputs-duplicate", "\"outputs\":[4]", "\"outputs\":[4,4]"),
        ("outputs-empty", "\"outputs\":[4]", "\"outputs\":[]"),
        ("dead-node", "\"outputs\":[4]", "\"outputs\":[2]"),
        (
            "add-with-params",
            "\"op\":\"add\",\"inputs\":[2,3],\"layer_start\":2,\"layer_len\":0,\"params\":{}",
            "\"op\":\"add\",\"inputs\":[2,3],\"layer_start\":2,\"layer_len\":0,\"params\":{\"dim\":1}",
        ),
        (
            "deep-nesting",
            "\"index\":0,\"op\":\"input\",\"inputs\":[],\"layer_start\":0,\"layer_len\":0,\"params\":{}",
            "\"index\":0,\"op\":\"input\",\"inputs\":[],\"layer_start\":0,\"layer_len\":0,\"params\":{\"a\":{\"b\":{}}}",
        ),
        (
            "unknown-top-level-key",
            "{\"format\"",
            "{\"extra\":1,\"format\"",
        ),
        (
            "wrong-format",
            "fandhe-ai.compat.functional",
            "fandhe-ai.compat.sequential",
        ),
        (
            "wrong-version",
            "\"format_version\":1",
            "\"format_version\":2",
        ),
        (
            "num-layers-mismatch",
            "\"num_layers\":2",
            "\"num_layers\":3",
        ),
    ];
    for (label, from, to) in manifest_cases {
        let r = load_after_manifest_edit(label, from, to);
        assert!(
            is_manifest_err(&r),
            "{label}: Manifest で拒否されるはず（safetensors が無いので Io なら開いてしまっている）: {:?}",
            r.err()
        );
    }
    // 未知の op は UnsupportedModel（allowlist）。
    let r = load_after_manifest_edit("unknown-op", "\"op\":\"add\"", "\"op\":\"subtract\"");
    assert!(
        matches!(r, Err(ModelIoError::UnsupportedModel { .. })),
        "{:?}",
        r.err()
    );
}

#[test]
fn zero_block_graph_with_training_false_is_rejected() {
    let dir = temp_path("zero-block-training");
    let _guard = Cleanup(dir.clone());
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("x");
    let y = b.input().expect("y");
    let s = b.add(&[x, y]).expect("add");
    let model = b.build(&[x, y], &[s]).expect("build");
    save_functional_model(&model, &dir).expect("save");
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).expect("read");
    std::fs::write(
        &path,
        text.replacen("\"training\":true", "\"training\":false", 1),
    )
    .expect("write");
    assert!(is_manifest_err(&load_functional_model(&dir)));
}

#[test]
fn concatenate_dim_is_required_and_round_trips() {
    let dir = temp_path("concat-dim");
    let _guard = Cleanup(dir.clone());
    let mut b = FunctionalBuilder::new();
    let x = b.input().expect("x");
    let y = b.input().expect("y");
    let c = b.concatenate(&[x, y], 1).expect("concat");
    let model = b.build(&[x, y], &[c]).expect("build");
    save_functional_model(&model, &dir).expect("save");
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.contains("\"params\":{\"dim\":1}"));
    // dim キーの欠落・未知キーは拒否。
    std::fs::write(&path, text.replacen("{\"dim\":1}", "{}", 1)).expect("write");
    assert!(is_manifest_err(&load_functional_model(&dir)));
    std::fs::write(&path, text.replacen("{\"dim\":1}", "{\"axis\":1}", 1)).expect("write");
    assert!(is_manifest_err(&load_functional_model(&dir)));
    std::fs::write(&path, &text).expect("restore");
    assert!(load_functional_model(&dir).is_ok());
}

// ------------------------------------------------ (6) safetensors・ファイル I/O の改竄

#[test]
fn safetensors_and_key_tampering_is_rejected() {
    let dir = temp_path("st-tamper");
    let _guard = Cleanup(dir.clone());
    save_functional_model(&tamper_graph(), &dir).expect("save");
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).expect("read");

    // safetensors_bytes の改竄 → Mismatch。
    let bytes_field = text
        .split("\"safetensors_bytes\":")
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .expect("bytes field")
        .to_string();
    let tampered = text.replacen(
        &format!("\"safetensors_bytes\":{bytes_field}"),
        "\"safetensors_bytes\":1",
        1,
    );
    std::fs::write(&path, tampered).expect("write");
    assert!(matches!(
        load_functional_model(&dir),
        Err(ModelIoError::Mismatch { .. })
    ));

    // parameter_keys の shape 改竄 → Mismatch（層構成から導いた期待と不一致）。
    let tampered = text.replacen(
        "\"key\":\"0.weight\",\"shape\":[2,2]",
        "\"key\":\"0.weight\",\"shape\":[2,3]",
        1,
    );
    assert_ne!(tampered, text);
    std::fs::write(&path, tampered).expect("write");
    assert!(matches!(
        load_functional_model(&dir),
        Err(ModelIoError::Mismatch { .. })
    ));
    std::fs::write(&path, &text).expect("restore");
    assert!(load_functional_model(&dir).is_ok());

    // safetensors の末尾を欠落させる → バイト数不一致。
    for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
        if entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            let data = std::fs::read(entry.path()).expect("read st");
            std::fs::write(entry.path(), &data[..data.len() - 1]).expect("truncate");
        }
    }
    assert!(load_functional_model(&dir).is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_manifest_or_safetensors_is_rejected() {
    use std::os::unix::fs::symlink;
    // manifest がシンボリックリンク。
    let dir = temp_path("symlink-manifest");
    let _guard = Cleanup(dir.clone());
    save_functional_model(&tamper_graph(), &dir).expect("save");
    let real = dir.join("manifest.real");
    std::fs::rename(dir.join("manifest.json"), &real).expect("rename");
    symlink(&real, dir.join("manifest.json")).expect("symlink");
    assert!(matches!(
        load_functional_model(&dir),
        Err(ModelIoError::Io(_))
    ));

    // safetensors がシンボリックリンク。
    let dir = temp_path("symlink-st");
    let _guard2 = Cleanup(dir.clone());
    save_functional_model(&tamper_graph(), &dir).expect("save");
    for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            let moved = dir.join("moved.bin");
            std::fs::rename(entry.path(), &moved).expect("rename");
            symlink(&moved, entry.path()).expect("symlink");
        }
    }
    assert!(matches!(
        load_functional_model(&dir),
        Err(ModelIoError::Io(_))
    ));
}

// ----------------------------------------------- (7) 深さ・配列長の見積りの実証

#[test]
fn wide_merge_node_round_trips_within_depth_and_array_limits() {
    // 入力 300 件の加算ノード。`nodes[].inputs`（深さ 3）と最上位 `inputs`（300 件）が上限内に収まる。
    let dir = temp_path("wide-merge");
    let _guard = Cleanup(dir.clone());
    let mut b = FunctionalBuilder::new();
    let inputs: Vec<_> = (0..300).map(|_| b.input().expect("input")).collect();
    let sum = b.add(&inputs).expect("add");
    let model = b.build(&inputs, &[sum]).expect("build");
    save_functional_model(&model, &dir).expect("save");
    let loaded = load_functional_model(&dir).expect("load");
    let tensors: Vec<Tensor<f32>> = (0..300).map(|i| det_data(2, 2, i as f32 * 0.01)).collect();
    let refs: Vec<&Tensor<f32>> = tensors.iter().collect();
    let want = model.predict(&refs).expect("predict");
    let got = loaded.predict(&refs).expect("predict");
    assert_eq!(bits_of(&want[0]), bits_of(&got[0]));
}

#[test]
fn compiled_lbfgs_is_never_saved_or_restored() {
    // `compile` が Lbfgs を拒否するため保存側に到達しないことの確認と、manifest 偽装の拒否。
    let mut model = tamper_graph();
    assert!(
        model
            .compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
            .is_err()
    );
    let dir = temp_path("lbfgs");
    let _guard = Cleanup(dir.clone());
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
        .expect("compile");
    save_functional_model(&model, &dir).expect("save");
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.contains("\"kind\":\"sgd\""), "{text}");
    std::fs::write(
        &path,
        text.replacen("\"kind\":\"sgd\"", "\"kind\":\"lbfgs\"", 1),
    )
    .expect("write");
    assert!(load_functional_model(&dir).is_err());
}
