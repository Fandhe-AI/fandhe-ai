//! `compat::Sequential` の facade `nn::Module` 実装に対する利用者シナリオの受入テスト
//! （イシュー #3816・親 #3741。承認根拠は #3768 の所有者回答コメント 6554564412、設計は
//! `docs/facade-nn-module-exposure-decision.md` §23.4・実装記録は §25）。
//!
//! 役割分担: 契約（委譲の等価性・fail-closed の維持）は #3815 の `nn_compat_sequential_module.rs` が
//! 固定済みで、本ファイルは利用者が facade だけで書くコード（独自の複合層・`ForwardHooked` 越し・
//! 多段の入れ子）で、学習モードの伝播・safetensors のファイル往復・パラメータ列挙が成り立つことを示す。
//! `fandhe_ai_autodiff` は import せず、tolerance は使わず bit 一致で比較する。実機依存の経路はない
//! （CPU 既定のみ。§23.4 の実機列はすべて「該当なし」）ため `#[ignore]` テストは置かない。
//! `api_surface.rs` の `compat_sequential_module_acceptance_coverage_map_matches` が、本ファイルの
//! テスト関数の存在・`#[ignore]` 不在・autodiff 非依存を機械的に固定する。

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32, load_safetensors_f32_from_bytes, save_safetensors_f32,
    save_safetensors_f32_to_bytes,
};
use fandhe_ai::nn::{ForwardHookCtx, ForwardHooked, Module};
use fandhe_ai::{AutodiffError, TapeRef, Tensor, Var, tape};

mod common;
use common::temp_dir::TempDirGuard;

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, shape).expect("テスト入力の構築")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("host tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn input() -> Tensor<f32> {
    t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3])
}

const SEED: u64 = 11;

/// 3 -> 4 の Linear→ReLU（融合の先読み走査が働く構成）。
fn stem(seed: u64) -> Sequential {
    Sequential::new().add_linear(3, 4, seed).unwrap().add_relu()
}

/// 4 -> 3 の Linear（残差で入力へ足せる形に戻す）。
fn head(seed: u64) -> Sequential {
    Sequential::new().add_linear(4, 3, seed).unwrap()
}

/// 利用者が書く複合層。`named_parameters` と `set_parameter` を対で override し、接頭辞を付け外しする。
struct Encoder {
    stem: Sequential,
    head: Sequential,
}

impl Encoder {
    fn new(seed: u64) -> Self {
        Self {
            stem: stem(seed),
            head: head(seed + 1),
        }
    }
}

impl Module for Encoder {
    fn forward<'t>(&self, tape: TapeRef<'t>, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let h = Module::forward(&self.stem, tape, x)?;
        let y = Module::forward(&self.head, tape, &h)?;
        y.add(x)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut v = Vec::new();
        for (n, t) in Module::named_parameters(&self.stem) {
            v.push((format!("stem.{n}"), t));
        }
        for (n, t) in Module::named_parameters(&self.head) {
            v.push((format!("head.{n}"), t));
        }
        v
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        match name.split_once('.') {
            Some(("stem", rest)) => Module::set_parameter(&mut self.stem, rest, value),
            Some(("head", rest)) => Module::set_parameter(&mut self.head, rest, value),
            _ => Err(AutodiffError::InvalidArgument(format!(
                "Encoder: 未知のパラメータ名 `{name}`"
            ))),
        }
    }
    fn set_training(&mut self, training: bool) {
        Module::set_training(&mut self.stem, training);
        Module::set_training(&mut self.head, training);
    }
    fn training(&self) -> bool {
        Module::training(&self.stem) && Module::training(&self.head)
    }
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("stem".to_string(), &self.stem as &dyn Module),
            ("head".to_string(), &self.head as &dyn Module),
        ]
    }
}

/// `named_parameters` だけ override し `set_parameter` を忘れた複合層（契約違反の例）。
struct HalfOverridden {
    block: Sequential,
}

impl Module for HalfOverridden {
    fn forward<'t>(&self, tape: TapeRef<'t>, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Module::forward(&self.block, tape, x)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        Module::named_parameters(&self.block)
            .into_iter()
            .map(|(n, t)| (format!("block.{n}"), t))
            .collect()
    }
}

fn run_trait<M: Module>(m: &M) -> (Vec<u32>, Vec<u32>) {
    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let out = Module::forward(m, tr, &x).unwrap();
    let loss = out.mul(&out).unwrap().sum(None).unwrap();
    let g = t.backward(&loss).unwrap();
    (
        bits(&out.to_tensor()),
        bits(g.get(&x).unwrap().expect("入力勾配")),
    )
}

fn assert_invalid_argument<T: std::fmt::Debug>(r: Result<T, AutodiffError>, what: &str) {
    match r {
        Err(AutodiffError::InvalidArgument(_)) => {}
        other => panic!("{what}: InvalidArgument を期待したが {other:?}"),
    }
}

fn snapshot(m: &impl Module) -> HashMap<String, Vec<u32>> {
    m.state_dict()
        .into_iter()
        .map(|(k, v)| (k, bits(&v)))
        .collect()
}

#[test]
fn user_composite_forward_matches_manual_composition_bit_exact() {
    let enc = Encoder::new(SEED);
    let (out_bits, grad_bits) = run_trait(&enc);

    let t = tape();
    let x = t.var(&input());
    let h = enc.stem.forward(&t, &x).unwrap();
    let y = enc.head.forward(&t, &h).unwrap().add(&x).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let g = t.backward(&loss).unwrap();
    assert_eq!(out_bits, bits(&y.to_tensor()));
    assert_eq!(grad_bits, bits(g.get(&x).unwrap().expect("入力勾配")));
}

#[test]
fn user_composite_named_parameters_enumerates_prefixed_keys() {
    let enc = Encoder::new(SEED);
    let keys: Vec<String> = enc.named_parameters().into_iter().map(|(k, _)| k).collect();
    assert_eq!(
        keys,
        [
            "stem.0.weight",
            "stem.0.bias",
            "head.0.weight",
            "head.0.bias"
        ]
    );
    // 3*4 + 4 + 4*3 + 3
    assert_eq!(enc.parameter_count(), 31);

    let names: Vec<String> = enc.children().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["stem", "head"]);
    let modules: Vec<String> = enc.named_modules().into_iter().map(|(n, _)| n).collect();
    assert_eq!(modules, ["stem", "head"]);
    let s = fandhe_ai::nn::summary(&enc);
    assert!(s.contains("Submodules: 2"), "{s}");
}

#[test]
fn user_composite_state_dict_safetensors_file_roundtrip() {
    let src = Encoder::new(SEED);
    let guard = TempDirGuard::new("acceptance-file-roundtrip-3816");
    let path = guard.path().join("encoder.safetensors");
    save_safetensors_f32(&path, &src.state_dict()).expect("save");
    let loaded = load_safetensors_f32(&path).expect("load");

    let mut dst = Encoder::new(SEED + 100);
    assert_ne!(snapshot(&src), snapshot(&dst), "別シードで構築されている");
    dst.load_state_dict(loaded).unwrap();
    assert_eq!(snapshot(&src), snapshot(&dst));
    assert_eq!(run_trait(&src), run_trait(&dst));
}

#[test]
fn user_composite_load_state_dict_rejects_unknown_and_missing_keys_without_mutation() {
    let mut enc = Encoder::new(SEED);
    let before = snapshot(&enc);
    // 検証前の部分適用を検出できるよう、現在値と異なる有効パラメータ（別シードの Encoder）を土台にする。
    let donor = Encoder::new(SEED + 100);
    assert_ne!(snapshot(&donor), before, "土台は現在値と異なる");
    let full = donor.state_dict();

    let mut missing = full.clone();
    missing.remove("head.0.bias");
    assert_invalid_argument(enc.load_state_dict(missing), "欠落キー");
    assert_eq!(snapshot(&enc), before);

    let mut extra = full.clone();
    extra.insert("stem.9.weight".into(), t32(&[0.0], &[1]));
    assert_invalid_argument(enc.load_state_dict(extra), "余剰キー");
    assert_eq!(snapshot(&enc), before);

    let mut wrong_prefix = full.clone();
    let v = wrong_prefix.remove("stem.0.weight").unwrap();
    wrong_prefix.insert("neck.0.weight".into(), v);
    assert_invalid_argument(enc.load_state_dict(wrong_prefix), "接頭辞の誤り");
    assert_eq!(snapshot(&enc), before);

    let mut bad_shape = full.clone();
    bad_shape.insert("stem.0.weight".into(), t32(&[0.0; 4], &[2, 2]));
    assert_invalid_argument(enc.load_state_dict(bad_shape), "shape 不一致");
    assert_eq!(snapshot(&enc), before);

    // set_parameter を override しない複合層は既定の load_state_dict が InvalidArgument で拒否する。
    let mut half = HalfOverridden { block: stem(SEED) };
    let sd = half.state_dict();
    assert!(!sd.is_empty());
    assert_invalid_argument(half.load_state_dict(sd), "set_parameter 未 override");
}

/// 最深に Dropout を含む compat::Sequential。
fn dropout_block(seed: u64) -> Sequential {
    Sequential::new()
        .add_linear(3, 4, seed)
        .unwrap()
        .add_relu()
        .add_dropout(0.5)
        .unwrap()
        .add_linear(4, 3, seed + 1)
        .unwrap()
}

/// compat::Sequential -> nn::Sequential -> compat::Sequential(Dropout) の 3 段。
fn three_level(seed: u64) -> Sequential {
    Sequential::new().add_module(fandhe_ai::nn::Sequential::new().add(dropout_block(seed)))
}

#[test]
fn training_mode_propagates_through_multi_level_nesting() {
    let mut outer = three_level(SEED);
    assert!(Module::training(&outer));
    Module::set_training(&mut outer, false);
    assert!(!Module::training(&outer));

    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let a = Module::forward(&outer, tr, &x).unwrap();
    let b = Module::forward(&outer, tr, &x).unwrap();
    assert_eq!(bits(&a.to_tensor()), bits(&b.to_tensor()), "eval は決定的");

    // inherent 経路で同じ構成を eval にした出力と一致する。
    let mut inherent = three_level(SEED);
    inherent.eval();
    let c = inherent.forward(&t, &x).unwrap();
    assert_eq!(bits(&a.to_tensor()), bits(&c.to_tensor()));

    Module::set_training(&mut outer, true);
    assert!(Module::training(&outer));
}

#[test]
fn batch_norm_running_stats_follow_training_mode_through_custom_module() {
    fn build() -> Sequential {
        Sequential::new().add_batch_norm1d(3, 1e-5, 0.1).unwrap()
    }
    let fwd = |m: &Sequential| -> Vec<u32> {
        let t = tape();
        let tr = TapeRef::from(&t);
        let x = tr.var(&input());
        bits(&Module::forward(m, tr, &x).unwrap().to_tensor())
    };
    let fwd_inherent = |m: &Sequential| -> Vec<u32> {
        let t = tape();
        let x = t.var(&input());
        bits(&m.forward(&t, &x).unwrap().to_tensor())
    };

    let mut user = build();
    Module::set_training(&mut user, false);
    let o1 = fwd(&user);
    assert_eq!(o1, fwd(&user), "eval の forward は統計を更新しない");

    Module::set_training(&mut user, true);
    let _ = fwd(&user); // running stats を更新する
    Module::set_training(&mut user, false);
    let o2 = fwd(&user);
    assert_ne!(o1, o2, "train forward 後は eval 出力が変わる");

    // 同じ操作列を inherent 経路で行った対照と bit 一致する。
    let mut reference = build();
    reference.eval();
    assert_eq!(o1, fwd_inherent(&reference));
    reference.train();
    let _ = fwd_inherent(&reference);
    reference.eval();
    assert_eq!(o2, fwd_inherent(&reference));
}

#[test]
fn multi_level_nesting_named_parameters_and_state_dict_roundtrip() {
    let src = three_level(SEED);
    let keys: Vec<String> = Module::named_parameters(&src)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(
        keys,
        ["0.0.0.weight", "0.0.0.bias", "0.0.3.weight", "0.0.3.bias"]
    );
    assert_eq!(Module::parameter_count(&src), 31);

    let bytes = save_safetensors_f32_to_bytes(&Module::state_dict(&src), None).expect("save");
    let loaded = load_safetensors_f32_from_bytes(&bytes).expect("load");
    let mut dst = three_level(SEED + 50);
    Module::load_state_dict(&mut dst, loaded).unwrap();
    for (k, v) in Module::state_dict(&src) {
        assert_eq!(bits(&Module::state_dict(&dst)[&k]), bits(&v), "{k}");
    }
}

/// 観察専用 hook（`output_value` は呼ばない）で発火回数を数える。
fn counting_hook(
    counter: &Arc<AtomicUsize>,
) -> impl Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync + 'static {
    let c = Arc::clone(counter);
    move |_ctx| {
        c.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn forward_hooked_compat_sequential_is_transparent() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut hooked = ForwardHooked::new(dropout_block(SEED), counting_hook(&calls));
    Module::set_training(&mut hooked, false);
    assert!(!Module::training(&hooked));
    assert!(
        !Module::training(hooked.inner()),
        "eval が Dropout まで透過"
    );

    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let out = Module::forward(&hooked, tr, &x).unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "forward 1 回につき 1 回発火"
    );
    let plain = hooked.inner().forward(&t, &x).unwrap();
    assert_eq!(bits(&out.to_tensor()), bits(&plain.to_tensor()));

    let inner_keys: Vec<String> = Module::named_parameters(hooked.inner())
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let keys: Vec<String> = Module::named_parameters(&hooked)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(keys, inner_keys, "接頭辞なしで inner と一致");

    let sd = Module::state_dict(&hooked);
    let mut other = ForwardHooked::new(dropout_block(SEED + 70), counting_hook(&calls));
    Module::load_state_dict(&mut other, sd.clone()).unwrap();
    for (k, v) in &sd {
        assert_eq!(bits(&Module::state_dict(&other)[k]), bits(v), "{k}");
    }

    // hook が Err を返すと forward も Err。
    let failing = ForwardHooked::new(stem(SEED), |_ctx| {
        Err(AutodiffError::InvalidArgument("hook 失敗".into()))
    });
    assert!(Module::forward(&failing, tr, &x).is_err());
}

#[test]
fn forward_hooked_inside_nested_containers_fires_once_per_forward() {
    let calls = Arc::new(AtomicUsize::new(0));
    let plain_block = || {
        Sequential::new()
            .add_linear(3, 4, SEED)
            .unwrap()
            .add_relu()
            .add_linear(4, 3, SEED + 1)
            .unwrap()
    };

    let nn_outer = fandhe_ai::nn::Sequential::new()
        .add(ForwardHooked::new(plain_block(), counting_hook(&calls)));
    let compat_outer =
        Sequential::new().add_module(ForwardHooked::new(plain_block(), counting_hook(&calls)));
    let reference = plain_block();

    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let expected = bits(&reference.forward(&t, &x).unwrap().to_tensor());

    let a = Module::forward(&nn_outer, tr, &x).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let b = Module::forward(&compat_outer, tr, &x).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(bits(&a.to_tensor()), expected);
    assert_eq!(bits(&b.to_tensor()), expected);
}

#[test]
fn outer_predict_with_nested_block_matches_module_forward_bit_exact() {
    let mut outer = Sequential::new()
        .add_module(fandhe_ai::nn::Sequential::new().add(dropout_block(SEED)))
        .add_relu();
    outer.eval();
    let predicted = outer.predict(&input()).unwrap();

    let t = tape();
    let tr = TapeRef::from(&t);
    let x = tr.var(&input());
    let via_module = Module::forward(&outer, tr, &x).unwrap();
    assert_eq!(bits(&predicted), bits(&via_module.to_tensor()));
}
