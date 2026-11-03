//! param groups の `load_model` 側の互換と fail-closed 拒否の統合テスト
//! （イシュー #3836・親 #3755 の A3-c 読み込み側。決定記録
//! `docs/autodiff-param-groups-decision.md` §18.3・§18.4・§18.13）。
//!
//! 役割: 読み込み側の本体は #3835 で実装済み（`compat::model_io::compiled` の
//! `parse_param_groups`・`check_param_group_slots`）。本ファイルは公開 API（`save_model`／
//! `load_model`／`compile_with_param_groups`／`ParamGroupStep::step_with_groups`）と `std` だけで
//! 次の 4 点を固定する。判定は bit 一致と variant 照合のみで、tolerance は使わない。
//!
//! 1. `param_groups` キーのない旧形式の manifest が空 groups として読める（後方互換）
//! 2. 不正な groups は `Manifest`／`Mismatch`／`UnsupportedModel` で拒否され、`Ok` を返さず
//!    （部分状態を返さない）、load は dir を変更しない。元へ戻すと `Ok` になる（陽性対照）
//! 3. `format_version` 1 と `Lbfgs` では groups を受理しない
//! 4. 不正 groups の拒否集合が、facade 公開の `step_with_groups`・`save_model`・`load_model` の
//!    3 経路で一致する（検証器の食い違いによる迂回経路を残さない。§18.3 末尾）
//!
//! Functional manifest の groups 拒否は `functional_io/tests.rs` の
//! `param_groups_in_functional_manifest_are_rejected` が固定している。

#![cfg(unix)]

use std::path::{Path, PathBuf};

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::optim::{AdamWConfig, LbfgsConfig, ParamGroup, ParamGroupStep as _, Sgd, SgdConfig};

mod common;
use common::temp_dir::TempDirGuard;

// ---------------------------------------------------------------------
// フィクスチャとヘルパー
// ---------------------------------------------------------------------

/// Linear(3,4)→ReLU→Linear(4,2)。訓練対象スロットは 4 個（0.weight・0.bias・2.weight・2.bias）。
fn build() -> Sequential {
    Sequential::new()
        .add_linear(3, 4, 7)
        .and_then(|m| m.add_relu().add_linear(4, 2, 8))
        .expect("構築できるはず")
}

const N_SLOTS: usize = 4;

fn data() -> (Tensor<f32>, Tensor<f32>) {
    let x: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.41 + 0.3).sin()).collect();
    let y: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.23 + 1.1).cos()).collect();
    (
        Tensor::new(x, &[8, 3]).expect("x"),
        Tensor::new(y, &[8, 2]).expect("y"),
    )
}

fn sgd(lr: f32) -> Optimizer {
    Optimizer::Sgd(SgdConfig::new(lr))
}

fn with_groups(opt: Optimizer, groups: &[ParamGroup]) -> Sequential {
    let mut m = build();
    m.compile_with_param_groups(opt, Loss::Mse, groups)
        .expect("compile_with_param_groups");
    m
}

fn bits(m: &Sequential) -> Vec<Vec<u32>> {
    m.trainable_parameters()
        .iter()
        .map(|t| {
            t.as_slice()
                .expect("contiguous")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

fn fit_more(m: &mut Sequential, epochs: usize) {
    let (x, y) = data();
    m.fit(&x, &y, FitConfig::new(epochs, 4)).expect("fit");
}

#[derive(Clone, Copy, Debug)]
enum Want {
    Manifest,
    Mismatch,
    Unsupported,
}

impl Want {
    fn matches(self, e: &ModelIoError) -> bool {
        matches!(
            (self, e),
            (Want::Manifest, ModelIoError::Manifest { .. })
                | (Want::Mismatch, ModelIoError::Mismatch { .. })
                | (Want::Unsupported, ModelIoError::UnsupportedModel { .. })
        )
    }
}

/// 保存済み dir と、改竄前の manifest（陽性対照の復元用）。
struct Fx {
    _guard: TempDirGuard,
    dir: PathBuf,
    manifest: String,
}

impl Fx {
    fn new(m: &Sequential, label: &str) -> Self {
        let guard = TempDirGuard::new(label);
        let dir = guard.path().join("m");
        save_model(m, &dir).expect("save");
        let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest");
        Self {
            _guard: guard,
            dir,
            manifest,
        }
    }

    fn write(&self, text: &str) {
        std::fs::write(self.dir.join("manifest.json"), text).expect("write manifest");
    }

    /// manifest を `text` へ差し替え、load が `want` で拒否され dir が不変なことを確かめ、
    /// 元の manifest へ戻すと `Ok` になること（陽性対照）まで確かめる。
    fn reject(&self, label: &str, text: &str, want: Want) {
        self.write(text);
        let before = snapshot(&self.dir);
        match load_model(&self.dir) {
            Ok(_) => panic!("{label}: 改竄されたのに load が成功した"),
            Err(e) => assert!(
                want.matches(&e),
                "{label}: variant が {want:?} と異なる: {e:?}"
            ),
        }
        assert_eq!(
            before,
            snapshot(&self.dir),
            "{label}: load が dir を変更した"
        );
        self.write(&self.manifest);
        if let Err(e) = load_model(&self.dir) {
            panic!("{label}: 元へ戻したのに load できない（ベースライン破損）: {e}");
        }
    }
}

fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .expect("readdir")
        .flatten()
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).expect("read"),
            )
        })
        .collect();
    v.sort();
    v
}

/// manifest 内の `"param_groups":<object>` の object 部分の範囲を返す
/// （object は配列のみを含み、入れ子の `}` を持たない）。
fn groups_obj_range(text: &str) -> std::ops::Range<usize> {
    let needle = "\"param_groups\":";
    let start = text.find(needle).expect("param_groups が無い") + needle.len();
    let end = start + text[start..].find('}').expect("終端") + 1;
    start..end
}

/// `param_groups` の値を `obj` へ差し替える。
fn replace_groups(text: &str, obj: &str) -> String {
    let r = groups_obj_range(text);
    format!("{}{}{}", &text[..r.start], obj, &text[r.end..])
}

/// `,"param_groups":<obj>` を丸ごと取り除く（旧形式 = キーなし）。
fn strip_groups(text: &str) -> String {
    let r = groups_obj_range(text);
    let key_start = text[..r.start]
        .rfind(",\"param_groups\":")
        .expect("先頭のカンマ");
    format!("{}{}", &text[..key_start], &text[r.end..])
}

/// groups を持たない compile 済み manifest の `"amp":null}` を `param_groups` つきへ差し替える。
fn inject_groups(text: &str, obj: &str) -> String {
    assert!(
        text.contains("\"amp\":null}"),
        "amp:null で終わる compiled が必要"
    );
    text.replacen(
        "\"amp\":null}",
        &format!("\"amp\":null,\"param_groups\":{obj}}}"),
        1,
    )
}

fn base_groups() -> Vec<ParamGroup> {
    vec![
        ParamGroup::new(vec![0, 1], 0.1, 0.0),
        ParamGroup::new(vec![2], 0.2, 0.5),
    ]
}

/// groups 付きで 1 epoch 学習して保存した fixture。
fn grouped_fx(label: &str) -> Fx {
    let mut m = with_groups(sgd(0.05), &base_groups());
    fit_more(&mut m, 1);
    Fx::new(&m, label)
}

const BASE_OBJ: &str =
    "{\"lr\":[0.1,0.2],\"weight_decay\":[0.0,0.5],\"offsets\":[0,2,3],\"slots\":[0,1,2]}";

// ---------------------------------------------------------------------
// T1: 旧形式（キーなし）の後方互換
// ---------------------------------------------------------------------

#[test]
fn legacy_manifest_without_param_groups_loads_as_empty_groups() {
    let fx = grouped_fx("t1");
    assert_eq!(&fx.manifest[groups_obj_range(&fx.manifest)], BASE_OBJ);
    let legacy = strip_groups(&fx.manifest);
    assert!(!legacy.contains("param_groups"));

    // groups を保ったまま復元したモデル（判別用）。
    let mut kept = load_model(&fx.dir).expect("load");
    fx.write(&legacy);
    let mut loaded = load_model(&fx.dir).expect("旧形式は読める");

    // 参照: 保存時の重みを持つ groups なしモデル（momentum 0 なので optimizer 状態は無い）。
    let mut saved = with_groups(sgd(0.05), &base_groups());
    fit_more(&mut saved, 1);
    let mut reference = with_groups(sgd(0.05), &[]);
    reference
        .load_state_dict(saved.state_dict())
        .expect("重みを揃える");
    assert_eq!(bits(&loaded), bits(&reference));

    fit_more(&mut loaded, 3);
    fit_more(&mut reference, 3);
    fit_more(&mut kept, 3);
    assert_eq!(
        bits(&loaded),
        bits(&reference),
        "旧形式は空 groups として読まれる"
    );
    assert_ne!(
        bits(&loaded),
        bits(&kept),
        "判別: groups を保ったモデルとは結果が異なる入力であること"
    );

    // 再保存した manifest にもキーは出ない。
    let guard = TempDirGuard::new("t1_resave");
    let dir2 = guard.path().join("m");
    save_model(&loaded, &dir2).expect("再保存");
    let again = std::fs::read_to_string(dir2.join("manifest.json")).expect("manifest");
    assert!(!again.contains("param_groups"));
}

#[test]
fn compiled_without_groups_has_no_key_and_round_trips() {
    for opt in [sgd(0.05), Optimizer::AdamW(AdamWConfig::default())] {
        let mut m = with_groups(opt, &[]);
        fit_more(&mut m, 1);
        let fx = Fx::new(&m, "t1b");
        assert!(!fx.manifest.contains("param_groups"));
        let mut loaded = load_model(&fx.dir).expect("load");
        fit_more(&mut loaded, 2);
        fit_more(&mut m, 2);
        assert_eq!(bits(&loaded), bits(&m));
    }
}

// ---------------------------------------------------------------------
// T2: 改竄の網羅（Manifest）
// ---------------------------------------------------------------------

#[test]
fn tampered_param_groups_are_rejected_as_manifest() {
    let fx = grouped_fx("t2");
    let deep = format!("{}1{}", "[".repeat(40), "]".repeat(40));
    let cases: Vec<(&str, String)> = vec![
        ("lr 長不足", BASE_OBJ.replace("[0.1,0.2]", "[0.1]")),
        ("lr 長超過", BASE_OBJ.replace("[0.1,0.2]", "[0.1,0.2,0.3]")),
        ("wd 長不足", BASE_OBJ.replace("[0.0,0.5]", "[0.0]")),
        ("offsets 長不足", BASE_OBJ.replace("[0,2,3]", "[0,3]")),
        ("offsets 長超過", BASE_OBJ.replace("[0,2,3]", "[0,1,2,3]")),
        (
            "G=0",
            "{\"lr\":[],\"weight_decay\":[],\"offsets\":[0],\"slots\":[]}".to_string(),
        ),
        ("offsets[0]!=0", BASE_OBJ.replace("[0,2,3]", "[1,2,3]")),
        ("空 group", BASE_OBJ.replace("[0,2,3]", "[0,0,3]")),
        ("非狭義増加", BASE_OBJ.replace("[0,2,3]", "[0,2,2]")),
        ("offsets 逆順", BASE_OBJ.replace("[0,2,3]", "[0,3,2]")),
        ("offsets[G]>len", BASE_OBJ.replace("[0,2,3]", "[0,2,4]")),
        ("offsets[G]<len", BASE_OBJ.replace("[0,2,3]", "[0,1,2]")),
        ("group 内重複", BASE_OBJ.replace("[0,1,2]", "[0,0,2]")),
        ("group 間重複", BASE_OBJ.replace("[0,1,2]", "[0,1,1]")),
        ("負の lr", BASE_OBJ.replace("[0.1,0.2]", "[-0.1,0.2]")),
        ("負の wd", BASE_OBJ.replace("[0.0,0.5]", "[0.0,-0.5]")),
        ("非正準 0.10", BASE_OBJ.replace("[0.1,0.2]", "[0.10,0.2]")),
        ("非正準 1", BASE_OBJ.replace("[0.1,0.2]", "[1,0.2]")),
        ("非正準 1E0", BASE_OBJ.replace("[0.1,0.2]", "[1E0,0.2]")),
        (
            "f32 で inf になる字句",
            BASE_OBJ.replace("[0.1,0.2]", "[1e39,0.2]"),
        ),
        ("NaN 字句", BASE_OBJ.replace("[0.1,0.2]", "[NaN,0.2]")),
        ("inf 字句", BASE_OBJ.replace("[0.1,0.2]", "[inf,0.2]")),
        ("lr が配列でない", BASE_OBJ.replace("[0.1,0.2]", "0.1")),
        (
            "lr が文字列要素",
            BASE_OBJ.replace("[0.1,0.2]", "[\"0.1\",0.2]"),
        ),
        ("slots が非整数", BASE_OBJ.replace("[0,1,2]", "[0.5,1,2]")),
        ("slots が負数", BASE_OBJ.replace("[0,1,2]", "[-1,1,2]")),
        (
            "slots が u64 超過",
            BASE_OBJ.replace("[0,1,2]", "[0,1,18446744073709551616]"),
        ),
        ("未知キー", BASE_OBJ.replace("\"slots\"", "\"slotz\"")),
        ("余分なキー", BASE_OBJ.replace('}', ",\"extra\":[]}")),
        (
            "キー欠落",
            "{\"lr\":[0.1,0.2],\"weight_decay\":[0.0,0.5],\"offsets\":[0,2,3]}".to_string(),
        ),
        ("null", "null".to_string()),
        ("空 object", "{}".to_string()),
        ("配列", "[]".to_string()),
        ("深い入れ子", BASE_OBJ.replace("[0.1,0.2]", &deep)),
        (
            "lr キー重複",
            BASE_OBJ.replacen("\"lr\":[0.1,0.2],", "\"lr\":[0.1,0.2],\"lr\":[0.1,0.2],", 1),
        ),
    ];
    for (label, obj) in &cases {
        fx.reject(label, &replace_groups(&fx.manifest, obj), Want::Manifest);
    }

    // param_groups キー自体の重複（compiled の中で同名キーが 2 回）。
    let dup = fx.manifest.replacen(
        "\"param_groups\":",
        "\"param_groups\":{},\"param_groups\":",
        1,
    );
    fx.reject("param_groups キー重複", &dup, Want::Manifest);

    // 陽性対照: 同じ置換機構で正しい値を入れ直すと Ok（置換ヘルパーが壊れていない）。
    fx.write(&replace_groups(&fx.manifest, BASE_OBJ));
    load_model(&fx.dir).expect("同一値への置換は Ok");
}

// ---------------------------------------------------------------------
// T3: 範囲外スロット（Mismatch）
// ---------------------------------------------------------------------

#[test]
fn out_of_range_slots_are_rejected_as_mismatch_at_boundary() {
    let fx = grouped_fx("t3");
    let ok = BASE_OBJ.replace("[0,1,2]", &format!("[0,1,{}]", N_SLOTS - 1));
    fx.write(&replace_groups(&fx.manifest, &ok));
    load_model(&fx.dir).expect("n-1 は受理");
    for (label, slot) in [
        ("n", N_SLOTS.to_string()),
        ("n+1", (N_SLOTS + 1).to_string()),
        ("u64::MAX", u64::MAX.to_string()),
    ] {
        let obj = BASE_OBJ.replace("[0,1,2]", &format!("[0,1,{slot}]"));
        fx.reject(label, &replace_groups(&fx.manifest, &obj), Want::Mismatch);
    }
}

// ---------------------------------------------------------------------
// T4: 組み合わせと版
// ---------------------------------------------------------------------

#[test]
fn lbfgs_with_groups_is_unsupported() {
    let mut m = build();
    m.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
        .expect("compile");
    let (x, y) = data();
    m.fit(&x, &y, FitConfig::new(1, 8)).expect("fit");
    let fx = Fx::new(&m, "t4_lbfgs");
    fx.reject(
        "Lbfgs + groups",
        &inject_groups(&fx.manifest, BASE_OBJ),
        Want::Unsupported,
    );
}

#[test]
fn format_version_1_with_groups_is_rejected_but_legacy_loads() {
    let fx = grouped_fx("t4_v1");
    let v1 = fx
        .manifest
        .replacen("\"format_version\":2", "\"format_version\":1", 1);
    assert_ne!(v1, fx.manifest);
    fx.reject("版 1 + groups", &v1, Want::Manifest);

    // 陽性対照: groups を持たない版 1 は受理される（旧形式の後方互換）。
    let plain = Fx::new(&build(), "t4_v1_plain");
    let v1_plain = plain
        .manifest
        .replacen("\"format_version\":2", "\"format_version\":1", 1);
    assert_ne!(v1_plain, plain.manifest);
    plain.write(&v1_plain);
    load_model(&plain.dir).expect("groups なしの版 1 は受理される");
}

// ---------------------------------------------------------------------
// T5: 検証器の突合（step_with_groups・save_model・load_model）
// ---------------------------------------------------------------------

/// CSR 形の object を手で組む（拒否ケースの注入用。f32 は `{:?}` で書く）。
fn csr_obj(groups: &[ParamGroup]) -> String {
    let lr: Vec<String> = groups.iter().map(|g| format!("{:?}", g.lr)).collect();
    let wd: Vec<String> = groups
        .iter()
        .map(|g| format!("{:?}", g.weight_decay))
        .collect();
    let mut offsets = vec![0usize];
    let mut slots = Vec::new();
    for g in groups {
        slots.extend(g.params.iter().copied());
        offsets.push(slots.len());
    }
    let nums = |v: &[usize]| v.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
    format!(
        "{{\"lr\":[{}],\"weight_decay\":[{}],\"offsets\":[{}],\"slots\":[{}]}}",
        lr.join(","),
        wd.join(","),
        nums(&offsets),
        nums(&slots)
    )
}

#[test]
fn group_validators_agree_across_step_save_and_load() {
    let p: Vec<Tensor<f32>> = (0..N_SLOTS)
        .map(|_| Tensor::new(vec![1.0, 1.0], &[2]).expect("p"))
        .collect();
    let g = Tensor::new(vec![0.5, 0.5], &[2]).expect("g");
    let params: Vec<&Tensor<f32>> = p.iter().collect();
    let grads: Vec<&Tensor<f32>> = (0..N_SLOTS).map(|_| &g).collect();

    // 拒否集合: (label, groups, 範囲外か)。
    let rejects: Vec<(&str, Vec<ParamGroup>, bool)> = vec![
        ("空 params", vec![ParamGroup::new(vec![], 0.1, 0.0)], false),
        (
            "NaN lr",
            vec![ParamGroup::new(vec![0], f32::NAN, 0.0)],
            false,
        ),
        (
            "+inf lr",
            vec![ParamGroup::new(vec![0], f32::INFINITY, 0.0)],
            false,
        ),
        ("負の lr", vec![ParamGroup::new(vec![0], -0.1, 0.0)], false),
        (
            "NaN wd",
            vec![ParamGroup::new(vec![0], 0.1, f32::NAN)],
            false,
        ),
        ("負の wd", vec![ParamGroup::new(vec![0], 0.1, -0.5)], false),
        (
            "group 内重複",
            vec![ParamGroup::new(vec![0, 0], 0.1, 0.0)],
            false,
        ),
        (
            "group 間重複",
            vec![
                ParamGroup::new(vec![0, 1], 0.1, 0.0),
                ParamGroup::new(vec![1], 0.1, 0.0),
            ],
            false,
        ),
        (
            "範囲外",
            vec![ParamGroup::new(vec![N_SLOTS], 0.1, 0.0)],
            true,
        ),
    ];

    let base = grouped_fx("t5");
    for (label, groups, out_of_range) in &rejects {
        // (a) step_with_groups。
        let mut opt = Sgd::new(SgdConfig::new(0.1)).expect("sgd");
        assert!(
            opt.step_with_groups(&params, &grads, groups).is_err(),
            "{label}: step_with_groups が受理した"
        );

        // (b) save_model。保存先は作られない。
        let m = with_groups(sgd(0.1), groups);
        let guard = TempDirGuard::new("t5_save");
        let dir = guard.path().join("m");
        let err = save_model(&m, &dir).expect_err(label);
        let want = if *out_of_range {
            Want::Mismatch
        } else {
            Want::Unsupported
        };
        assert!(
            want.matches(&err),
            "{label}: save の variant が {want:?} と異なる: {err:?}"
        );
        assert!(!dir.exists(), "{label}: 拒否時に保存先を作った");

        // (c) load_model（manifest 注入）。
        let want = if *out_of_range {
            Want::Mismatch
        } else {
            Want::Manifest
        };
        base.reject(
            label,
            &replace_groups(&base.manifest, &csr_obj(groups)),
            want,
        );
    }

    // 受理集合: 3 経路すべてが受理し、往復後の続きの学習が bit 一致する。
    let accepts: Vec<(&str, Vec<ParamGroup>)> = vec![
        ("lr=0", vec![ParamGroup::new(vec![0], 0.0, 0.0)]),
        ("wd=-0.0", vec![ParamGroup::new(vec![1], 0.1, -0.0)]),
        (
            "非昇順 params",
            vec![ParamGroup::new(vec![3, 1, 2], 0.2, 0.01)],
        ),
        (
            "単一スロット",
            vec![ParamGroup::new(vec![N_SLOTS - 1], 0.3, 0.0)],
        ),
    ];
    for (label, groups) in &accepts {
        let mut opt = Sgd::new(SgdConfig::new(0.1)).expect("sgd");
        opt.step_with_groups(&params, &grads, groups)
            .unwrap_or_else(|e| panic!("{label}: step_with_groups が拒否: {e}"));

        let mut continuous = with_groups(sgd(0.1), groups);
        fit_more(&mut continuous, 1);
        let fx = Fx::new(&continuous, "t5_accept");
        let mut restored = load_model(&fx.dir).unwrap_or_else(|e| panic!("{label}: {e}"));
        fit_more(&mut continuous, 2);
        fit_more(&mut restored, 2);
        assert_eq!(
            bits(&restored),
            bits(&continuous),
            "{label}: 往復後に bit 不一致"
        );
    }
}
