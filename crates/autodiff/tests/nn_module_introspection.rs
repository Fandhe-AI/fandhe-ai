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

// --- 循環参照・共有子の回帰テスト（codex-review 指摘・PR #2231） -------
//
// `Module::children` は trait object を返す性質上、実装者が自身
// （`self`）や既出の `Module` を任意に返せる。`named_modules`／
// `nn::summary` はデータポインタベースの訪問済み集合で既出ノードへの
// 再帰を打ち切る（`crates/autodiff/src/nn/module.rs::
// collect_named_modules`・`crates/autodiff/src/nn/container.rs::
// write_module` 参照）。ここでは公開 API（本クレートの `Module` は
// crates.io 公開クレートの一部）経由で外部実装者が循環・共有構造を
// 作った場合でも panic（stack overflow）せず有限の結果を返すことを
// 確認する。

use fandhe_ai_autodiff::{Tape, Var};

/// `children()` から自身を返す循環参照 `Module`。
struct SelfReferencingModule;

impl Module for SelfReferencingModule {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        unreachable!("本テストでは forward は呼ばれない")
    }

    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("self".to_string(), self)]
    }
}

#[test]
fn named_modules_terminates_on_self_referencing_module() {
    let m = SelfReferencingModule;
    // 循環参照があっても panic（stack overflow）せず、ルート自身への
    // 再帰を打ち切って有限の結果を返す。
    assert!(m.named_modules().is_empty());
}

#[test]
fn summary_terminates_on_self_referencing_module() {
    let m = SelfReferencingModule;
    // `summary`（`write_module`）も同じ訪問済み集合を通しで使うため
    // 無限再帰しない。
    let out = summary(&m);
    assert!(out.contains("Submodules: 0\n"));
}

/// 2 ノードが互いを子として参照する循環（ルート自身の自己参照では
/// なく、子孫の間接的な循環）を構成するモック。`child` は構築後に
/// `Cell` 経由で設定する（`Module::children(&self)` は `&self` しか
/// 受け取れないため）。`edge_name` は「このノードから `child` への
/// 辺の名前」（`children()` が返すタプルの第 1 要素）であり、この
/// ノード自身の名前ではない点に注意（[`Module::children`] の命名
/// 契約どおり、辺は「子への参照」を指す）。
struct CyclicNode<'a> {
    edge_name: &'static str,
    child: std::cell::Cell<Option<&'a dyn Module>>,
}

impl<'a> Module for CyclicNode<'a> {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        unreachable!("本テストでは forward は呼ばれない")
    }

    fn children(&self) -> Vec<(String, &dyn Module)> {
        match self.child.get() {
            Some(child) => vec![(self.edge_name.to_string(), child)],
            None => Vec::new(),
        }
    }
}

#[test]
fn named_modules_terminates_on_indirect_cycle_between_two_modules() {
    // a --"b"--> b --"a"--> a という循環。
    let a = CyclicNode {
        edge_name: "b",
        child: std::cell::Cell::new(None),
    };
    let b = CyclicNode {
        edge_name: "a",
        child: std::cell::Cell::new(None),
    };
    a.child.set(Some(&b));
    b.child.set(Some(&a));

    // ルート a から辿ると: a(visited シード) -> 辺"b"で b(未訪問。
    // 列挙) -> 辺"b.a"で a(既出。打ち切り)。無限再帰せず b のみが
    // 列挙される。
    let modules = a.named_modules();
    let names: Vec<&str> = modules.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["b"]);
}

/// 2 つの親から同じ子 `Module` を参照する（循環ではなく共有）構成。
struct SharedChildParent<'a> {
    a: &'a dyn Module,
    b: &'a dyn Module,
}

impl<'a> Module for SharedChildParent<'a> {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        unreachable!("本テストでは forward は呼ばれない")
    }

    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![("a".to_string(), self.a), ("b".to_string(), self.b)]
    }
}

#[test]
fn named_modules_dedups_shared_child_module_like_pytorch_memo() {
    let leaf = linear(2, 2, 11);
    let shared: &dyn Module = &leaf;
    let parent = SharedChildParent {
        a: shared,
        b: shared,
    };

    // PyTorch の `named_modules()` は memo で重複を抑止する（同じ
    // 子が複数の親から共有される場合は最初の到達経路でのみ列挙）。
    let modules = parent.named_modules();
    let names: Vec<&str> = modules.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["a"]);
}

/// 回帰テスト（イシュー #2134 codex-review／Bugbot 指摘・PR #2231）:
/// `MultiheadAttention { q_proj: Linear, k_proj: Linear, ... }` は
/// 最初のフィールド `q_proj` がルート構造体の先頭（オフセット 0）に
/// 配置されうるため、`self as *const Self as *const ()` と
/// `&self.q_proj as *const dyn Module as *const ()` が異なる型
/// （`MultiheadAttention` と `Linear`）でありながら数値としては
/// 一致しうる。ノード同一性をデータポインタ単独で判定すると、これを
/// 「ルート自身の既出」と誤判定して `q_proj` 以降の子孫が丸ごと
/// `named_modules` から欠落する。型名込みの識別子（`(ポインタ, 型名)`
/// の組）で区別できることを固定する。
#[test]
fn named_modules_does_not_drop_first_child_that_aliases_root_address() {
    let mha = MultiheadAttention::new(4, 2, true, 1).unwrap();
    let names: Vec<String> = mha
        .named_modules()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(names, vec!["q_proj", "k_proj", "v_proj", "out_proj"]);
}

/// 上記回帰テストの入れ子版。`TransformerEncoderLayer` は
/// `MultiheadAttention` をさらに `self_attn` として先頭フィールドに
/// 持つため、2 段階のオフセット 0 誤判定（ルート→`self_attn`、
/// `self_attn`→`q_proj`）が連鎖しうる構成で、直接子（5 件）＋
/// `self_attn` の孫（4 件）の計 9 件が欠落なく列挙されることを固定
/// する。
#[test]
fn named_modules_does_not_drop_nested_offset_zero_descendants() {
    let layer = TransformerEncoderLayer::new(
        4,
        2,
        8,
        FeedForwardActivation::Relu,
        LAYER_NORM_DEFAULT_EPS,
        7,
    )
    .unwrap();
    let names: Vec<String> = layer
        .named_modules()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        names,
        vec![
            "self_attn",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "linear1",
            "linear2",
            "norm1",
            "norm2",
        ]
    );
}

/// 回帰テスト（イシュー #2134 codex-review／Bugbot 指摘・PR #2231）:
/// `Relu` は `struct Relu;`（フィールドなしのゼロサイズ型）のため、
/// `Box<Relu>` として複数インスタンスを保持すると、アロケータの
/// well-known dangling address を共有し同一データポインタを持ちうる。
/// `named_modules`／`summary` がこれをグローバルな訪問済み集合で
/// 「既出」と誤判定すると、`Sequential` に同種 ZST 活性化層を複数積んだ
/// 場合に 2 個目以降の `Relu`・後続レイヤーが出力から欠落する。本テスト
/// は同種 ZST を隣接させない配置（`Linear` を挟む）・隣接させる配置
/// （`Relu` を連続で積む）の両方で全レイヤーが欠落なく列挙されることを
/// 固定する。
#[test]
fn named_modules_does_not_drop_layers_after_repeated_zst_activation_siblings() {
    let seq = Sequential::new()
        .add(linear(4, 8, 21))
        .add(Relu)
        .add(linear(8, 8, 22))
        .add(Relu)
        .add(Relu) // 隣接する同種 ZST（アドレス衝突が最も起きやすい配置）
        .add(linear(8, 2, 23));

    let names: Vec<String> = seq
        .named_modules()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        names,
        vec!["0", "1", "2", "3", "4", "5"],
        "ZST 活性化層〈Relu〉を複数含む Sequential で後続レイヤーが欠落してはならない"
    );
}

/// 上記回帰テストの `summary` 版。同じ配置で `write_module` の再帰も
/// 全ノードを列挙し、`Submodules` 件数が欠落なく一致することを固定
/// する。
#[test]
fn summary_does_not_drop_layers_after_repeated_zst_activation_siblings() {
    let seq = Sequential::new()
        .add(linear(4, 8, 31))
        .add(Relu)
        .add(Relu)
        .add(linear(8, 2, 32));

    let out = summary(&seq);
    let relu_lines = out.matches("Relu [params: 0]").count();
    assert_eq!(relu_lines, 2, "summary 出力: {out}");
    assert!(out.contains("Submodules: 4\n"), "summary 出力: {out}");
}

/// `summary` 版のオフセット 0 誤判定回帰テスト（上記
/// `named_modules_does_not_drop_first_child_that_aliases_root_address`
/// と同じ根本原因を `write_module` 経由でも固定する）。
#[test]
fn summary_does_not_drop_first_child_that_aliases_root_address() {
    let mha = MultiheadAttention::new(4, 2, true, 1).unwrap();
    let out = summary(&mha);
    for expected in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        assert!(
            out.contains(&format!("({expected}): Linear")),
            "summary 出力に {expected} が見つからない: {out}"
        );
    }
    assert!(out.contains("Submodules: 4\n"), "summary 出力: {out}");
}
