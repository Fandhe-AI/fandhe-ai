//! `nn::EmbeddingBag`（イシュー #2161・親 #2131）の統合テスト。
//! 大半の契約（`sum`／`mean`／`max`・`padding_idx`・offsets 検査・
//! 孤児ノード非発生）は `crates/autodiff/src/nn/embedding_bag.rs` の
//! 単体テストで検証済みのため、本ファイルは公開 API（`autodiff` の
//! `pub` 面のみを経由する別クレート扱い）経由での到達性と、
//! `Module` trait 経由（`compat::Sequential` 相当の呼び出し規約）の
//! 検証に絞る。

mod common;

use fandhe_ai_autodiff::nn::{EmbeddingBag, EmbeddingBagMode, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

#[test]
fn public_api_forward_matches_manual_embedding_and_reduce() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = t((0..12).map(|v| v as f32).collect(), &[4, 3]);
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None)
        .expect("rank 2・num_embeddings > 0");
    let ids = Tensor::<i32>::new(vec![0, 1, 2, 3], &[2, 2]).expect("shape 一致");
    let out = bag.bind(&tape).forward(&ids).expect("正常な rank 2 ids");
    let got = out.to_tensor().host_slice().into_owned();
    // bag0 = rows[0,1] = [0,1,2]+[3,4,5] = [3,5,7]、
    // bag1 = rows[2,3] = [6,7,8]+[9,10,11] = [15,17,19]
    assert_eq!(got, vec![3.0, 5.0, 7.0, 15.0, 17.0, 19.0]);
}

/// `Module::forward`（`compat::Sequential` 相当の呼び出し経路）が
/// `EmbeddingBagVars::forward_from_var` と同一の値を返すこと
/// （`nn_embedding.rs` の `forward_from_var_matches_forward_with_raw_ids`
/// と同型の検証、公開 API 経由版）。
#[test]
fn module_forward_matches_direct_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Mean, None)
        .expect("rank 2・num_embeddings > 0");

    let ids_i32 = Tensor::<i32>::new(vec![0, 1, 2, 0], &[2, 2]).expect("shape 一致");
    let direct = bag
        .bind(&tape)
        .forward(&ids_i32)
        .expect("正常な rank 2 ids")
        .to_tensor();

    let ids_f32 = t(vec![0.0, 1.0, 2.0, 0.0], &[2, 2]);
    let input_var = tape.var(&ids_f32);
    let via_module = Module::forward(&bag, &tape, &input_var)
        .expect("f32 整数値 id は受理される")
        .to_tensor();

    assert_eq!(
        direct.host_slice().into_owned(),
        via_module.host_slice().into_owned()
    );
}

#[test]
fn module_forward_rejects_non_integer_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = t(vec![1.0, 2.0], &[1, 2]);
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None)
        .expect("rank 2・num_embeddings > 0");
    let bad_ids = t(vec![1.5], &[1, 1]);
    let input_var = tape.var(&bad_ids);
    let Err(err) = Module::forward(&bag, &tape, &input_var) else {
        panic!("非整数 id は Err を返すはず")
    };
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn module_named_parameters_and_downcast_hook() {
    let w = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None)
        .expect("rank 2・num_embeddings > 0");
    let params = Module::named_parameters(&bag);
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "weight");
    assert!(Module::as_embedding_bag(&bag).is_some());
    assert!(!Module::supports_forward_host(&bag));
}

#[test]
fn backward_accumulates_grad_for_repeated_ids_in_same_bag() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = t(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]);
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None)
        .expect("rank 2・num_embeddings > 0");
    let vars = bag.bind(&tape);
    // 同一 bag 内に id=0 が 2 回登場する。
    let ids = Tensor::<i32>::new(vec![0, 0], &[1, 2]).expect("shape 一致");
    let out = vars.forward(&ids).expect("正常な rank 2 ids");
    let loss = out.sum(None).expect("全軸縮約は失敗しない");
    let grads = tape.backward(&loss).expect("weight は requires_grad の葉");
    let dw = grads
        .get(&vars.weight)
        .expect("weight は requires_grad=true の葉")
        .expect("weight は loss に到達する");
    // id=0 の行が bag 内で 2 回参照されるため、勾配は 2 倍になる
    // （row0 = [2, 2]、row1（未参照）= [0, 0]）。
    assert_eq!(dw.host_slice().into_owned(), vec![2.0, 2.0, 0.0, 0.0]);
}
