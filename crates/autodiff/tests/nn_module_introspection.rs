//! `Module::children`／`named_modules`／`parameter_count`・
//! `nn::ModuleDict`・`nn::summary`（イシュー #2134・親 #2131）の受け入れ
//! 条件対応テスト。
//!
//! `nn_attention.rs`・`nn_rnn.rs` と同じ構成で `mod common;` を読み込む
//! （本ファイルは `common::naive_ops()` を直接は使わないが、
//! `crates/autodiff/tests/*.rs` は `autodiff` の公開 API のみを経由する
//! 別クレート扱いという既存の設計方針を踏襲する）。数値経路
//! （`Op`／`BackendOps`／VJP）を経由しない CPU ホスト側の introspection
//! 機構のみを対象とするため、実機 `#[ignore]` テストは追加しない
//! （イシュー #2134 実装計画 §7）。

mod common;

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::{
    FeedForwardActivation, LAYER_NORM_DEFAULT_EPS, Linear, Module, ModuleDict, ModuleList,
    MultiheadAttention, Rnn, Sequential, TransformerEncoderLayer, summary,
};
use fandhe_ai_tensor_core::Tensor;

fn linear(in_features: usize, out_features: usize, seed: u64) -> Linear {
    Linear::new(in_features, out_features, true, seed)
        .expect("test fixture: Linear::new は正当な引数のみ渡す")
}

// --- 葉モジュールの既定契約 -----------------------------------------

#[test]
fn leaf_module_children_and_named_modules_are_empty() {
    let relu = Relu;
    assert!(relu.children().is_empty());
    assert!(relu.named_modules().is_empty());

    let lin = linear(4, 8, 1);
    assert!(lin.children().is_empty());
    assert!(lin.named_modules().is_empty());
}

#[test]
fn linear_parameter_count_matches_weight_and_bias_numel() {
    let with_bias = linear(4, 8, 1);
    assert_eq!(with_bias.parameter_count(), 4 * 8 + 8); // weight [8,4] + bias [8]

    let without_bias = Linear::new(4, 8, false, 2).unwrap();
    assert_eq!(without_bias.parameter_count(), 4 * 8);
}

#[test]
fn relu_parameter_count_is_zero() {
    assert_eq!(Relu.parameter_count(), 0);
}

// --- Sequential のネスト ---------------------------------------------

fn two_linear_sequential() -> Sequential {
    Sequential::new()
        .add(linear(4, 8, 1))
        .add(Relu)
        .add(linear(8, 2, 2))
}

#[test]
fn sequential_named_modules_paths_are_index_order() {
    let seq = two_linear_sequential();
    let paths: Vec<String> = seq
        .named_modules()
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert_eq!(paths, vec!["0", "1", "2"]);
}

#[test]
fn sequential_parameter_count_matches_named_parameters_numel_sum() {
    let seq = two_linear_sequential();
    let expected: usize = seq
        .named_parameters()
        .into_iter()
        .map(|(_, t)| t.numel())
        .sum();
    assert_eq!(seq.parameter_count(), expected);
    assert_eq!(seq.parameter_count(), (4 * 8 + 8) + (8 * 2 + 2));
}

/// ネスト（`Sequential` → `ModuleDict{"enc": Sequential(...), "head":
/// Linear}`）で `named_modules()` のパスが深さ優先・登録順（子自身 →
/// その子孫）になることを確認する。
#[test]
fn nested_module_dict_and_sequential_named_modules_are_depth_first_in_registration_order() {
    let inner = two_linear_sequential(); // "0","1","2"
    let mut dict = ModuleDict::new();
    dict.insert("enc", Box::new(inner)).unwrap();
    dict.insert("head", Box::new(linear(2, 1, 3))).unwrap();

    let mut outer = Sequential::new();
    outer.push(Box::new(dict));

    let paths: Vec<String> = outer
        .named_modules()
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    assert_eq!(
        paths,
        vec!["0", "0.enc", "0.enc.0", "0.enc.1", "0.enc.2", "0.head",]
    );
}

/// 汎用不変条件テスト: `named_modules()` の全 `(path, m)` について、
/// `m.named_parameters()` の名前を `"{path}.{name}"` にしたものが
/// すべてルートの `named_parameters()` に含まれる（`children` 名と
/// `named_parameters` 接頭辞のドリフトを一括検出。実装計画 §6.1）。
fn assert_children_names_match_named_parameters_prefixes(root: &dyn Module) {
    let root_names: std::collections::HashSet<String> = root
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    for (path, module) in root.named_modules() {
        for (name, _) in module.named_parameters() {
            let expected = format!("{path}.{name}");
            assert!(
                root_names.contains(&expected),
                "named_modules パス `{path}` の子パラメータ `{name}` に対応する \
                 `{expected}` が root.named_parameters() に含まれない \
                 （children 名と named_parameters 接頭辞のドリフト）"
            );
        }
    }
}

#[test]
fn multihead_attention_children_match_named_parameters_prefixes() {
    let mha = MultiheadAttention::new(4, 2, true, 42).unwrap();
    assert_children_names_match_named_parameters_prefixes(&mha);
    let child_names: Vec<String> = mha.children().into_iter().map(|(n, _)| n).collect();
    assert_eq!(child_names, vec!["q_proj", "k_proj", "v_proj", "out_proj"]);
}

#[test]
fn transformer_encoder_layer_children_match_named_parameters_prefixes() {
    let layer = TransformerEncoderLayer::new(
        4,
        2,
        8,
        FeedForwardActivation::Relu,
        LAYER_NORM_DEFAULT_EPS,
        7,
    )
    .unwrap();
    assert_children_names_match_named_parameters_prefixes(&layer);
    let child_names: Vec<String> = layer.children().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        child_names,
        vec!["self_attn", "linear1", "linear2", "norm1", "norm2"]
    );
}

#[test]
fn nested_sequential_and_module_dict_children_match_named_parameters_prefixes() {
    let mut dict = ModuleDict::new();
    dict.insert("enc", Box::new(two_linear_sequential()))
        .unwrap();
    dict.insert("head", Box::new(linear(2, 1, 9))).unwrap();
    assert_children_names_match_named_parameters_prefixes(&dict);
}

// --- Rnn: named_modules は空、parameter_count はテンソル数を反映 -----

#[test]
fn rnn_named_modules_is_empty_but_parameter_count_reflects_cell_tensors() {
    let rnn = Rnn::new(4, 8, true, 11).unwrap();
    assert!(
        rnn.named_modules().is_empty(),
        "RnnCell は Module を実装しないため named_modules は空のはず"
    );
    let expected: usize = rnn
        .named_parameters()
        .into_iter()
        .map(|(_, t)| t.numel())
        .sum();
    assert_eq!(rnn.parameter_count(), expected);
    assert_eq!(rnn.named_parameters().len(), 4); // weight_ih, weight_hh, bias_ih, bias_hh
}

// --- ModuleDict の Module trait 実装 ----------------------------------

#[test]
fn module_dict_insert_get_remove_and_ordering() {
    let mut dict = ModuleDict::new();
    assert!(dict.insert("a", Box::new(Relu)).unwrap().is_none());
    assert!(dict.insert("b", Box::new(Relu)).unwrap().is_none());
    assert_eq!(dict.keys().collect::<Vec<_>>(), vec!["a", "b"]);

    let replaced = dict.insert("a", Box::new(Relu)).unwrap();
    assert!(replaced.is_some());
    assert_eq!(
        dict.keys().collect::<Vec<_>>(),
        vec!["a", "b"],
        "同名キーの置換で挿入順位置がずれてはいけない"
    );

    assert!(dict.remove("a").is_some());
    assert!(!dict.contains_key("a"));
    assert_eq!(dict.len(), 1);
}

#[test]
fn module_dict_rejects_empty_and_dotted_keys() {
    let mut dict = ModuleDict::new();
    assert!(matches!(
        dict.insert("", Box::new(Relu)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        dict.insert("bad.key", Box::new(Relu)),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn module_dict_named_parameters_uses_key_prefix() {
    let mut dict = ModuleDict::new();
    dict.insert("l1", Box::new(linear(4, 8, 1))).unwrap();
    let names: Vec<String> = dict
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(names, vec!["l1.weight", "l1.bias"]);
}

#[test]
fn module_dict_set_parameter_success_and_error_paths() {
    let mut dict = ModuleDict::new();
    dict.insert("l1", Box::new(linear(4, 8, 1))).unwrap();

    let new_weight = Tensor::new(vec![9.0f32; 32], &[4, 8]).unwrap();
    dict.set_parameter("l1.weight", new_weight).unwrap();

    // 区切りなし。
    assert!(matches!(
        dict.set_parameter("weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // 未知キー。
    assert!(matches!(
        dict.set_parameter("missing.weight", Tensor::new(vec![0.0f32], &[1]).unwrap()),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn module_dict_state_dict_load_state_dict_round_trip_is_bit_identical() {
    let mut dict = ModuleDict::new();
    dict.insert("l1", Box::new(linear(4, 8, 1))).unwrap();
    dict.insert("l2", Box::new(linear(8, 2, 2))).unwrap();

    let before = dict.state_dict();
    dict.load_state_dict(dict.state_dict()).unwrap();
    let after = dict.state_dict();
    for (key, tensor) in &before {
        assert_eq!(
            tensor.contiguous().as_slice().unwrap(),
            after[key].contiguous().as_slice().unwrap(),
            "key `{key}` が往復後に変化した"
        );
    }
}

#[test]
fn module_dict_forward_and_forward_host_return_invalid_argument() {
    use fandhe_ai_autodiff::Tape;

    let dict = ModuleDict::new();
    let tape = Tape::new();
    let input_tensor = Tensor::new(vec![1.0f32], &[1]).unwrap();
    let input = tape.var(&input_tensor);
    assert!(matches!(
        dict.forward(&tape, &input),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn module_dict_set_training_propagates_to_children_and_own_flag() {
    use fandhe_ai_autodiff::nn::Dropout;

    let mut dict = ModuleDict::new();
    dict.insert("drop", Box::new(Dropout::new(0.5).unwrap()))
        .unwrap();
    assert!(dict.training());
    dict.set_training(false);
    assert!(!dict.training());
    assert!(!dict.get("drop").unwrap().training());
}

// --- nn::summary -------------------------------------------------------

#[test]
fn summary_matches_fixed_format_for_sequential() {
    let seq = two_linear_sequential();
    let expected = "Sequential(\n\
         \x20 (0): Linear [params: 40]\n\
         \x20 (1): Relu [params: 0]\n\
         \x20 (2): Linear [params: 18]\n\
         ) [params: 58]\n\
         Submodules: 3\n\
         Total parameters: 58\n";
    assert_eq!(summary(&seq), expected);
}

#[test]
fn summary_leaf_module_is_two_line_form() {
    let lin = linear(4, 8, 1);
    let expected = "Linear [params: 40]\nSubmodules: 0\nTotal parameters: 40\n";
    assert_eq!(summary(&lin), expected);
}

#[test]
fn summary_module_dict_nesting_has_correct_indent_and_submodule_count() {
    let mut dict = ModuleDict::new();
    dict.insert("l1", Box::new(linear(4, 8, 1))).unwrap();
    dict.insert("l2", Box::new(linear(8, 2, 2))).unwrap();

    let out = summary(&dict);
    assert!(out.starts_with("ModuleDict(\n"));
    assert!(out.contains("  (l1): Linear [params: 40]\n"));
    assert!(out.contains("  (l2): Linear [params: 18]\n"));
    assert!(out.contains(") [params: 58]\n"));
    assert!(out.contains("Submodules: 2\n"));
    assert!(out.contains("Total parameters: 58\n"));
}

// --- object safety: Box<dyn Module> 経由で新規メソッドを呼べること -----

#[test]
fn box_dyn_module_can_call_named_modules_and_parameter_count() {
    let boxed: Box<dyn Module> = Box::new(two_linear_sequential());
    assert_eq!(boxed.named_modules().len(), 3);
    assert_eq!(boxed.parameter_count(), 58);
}

#[test]
fn module_list_children_names_are_index_order() {
    let mut list = ModuleList::new();
    list.push(Box::new(Relu));
    list.push(Box::new(linear(2, 2, 5)));
    let names: Vec<String> = list.children().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, vec!["0", "1"]);
}
