//! `nn::Module` の `set_training`／`training`／`named_parameters`
//! （イシュー #1758）の契約テスト。
//!
//! 本 issue は数値演算を追加しない（新規 `Op`／`BackendOps`／VJP／GPU
//! カーネルはいずれも該当なし）ため、ここでの「parity」に相当する
//! 検証は CPU 上の bit 完全一致（モード切替の前後で `forward` の出力・
//! 勾配が変化しないこと）に限定する。CUDA／Metal への申し送りは不要
//! （数値経路に一切触れないため）。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::activation::{Relu, Softmax};
use fandhe_ai_autodiff::nn::{
    BatchNorm1d, BatchNorm2d, Gru, Linear, Lstm, Module, MultiheadAttention, RmsNorm, Rnn,
};
use fandhe_ai_tensor_core::Tensor;

/// 既定契約（`Module::training`／`set_training` の trait doc）: 無状態
/// モジュール（`Relu`・`Softmax` 等）は `training()` が常に `true`
/// （`set_training` を呼んでも変化しない）・`named_parameters()` は空。
#[test]
fn stateless_modules_default_contract() {
    let mut relu = Relu;
    assert!(relu.training());
    relu.set_training(false);
    assert!(
        relu.training(),
        "無状態モジュールは set_training を呼んでも training() が変化しない契約"
    );
    assert!(relu.named_parameters().is_empty());

    let mut softmax = Softmax::new(0);
    assert!(softmax.training());
    softmax.set_training(false);
    assert!(softmax.training());
    assert!(softmax.named_parameters().is_empty());
}

/// object safety: `Vec<Box<dyn Module>>` に `Linear` と活性化関数を
/// 混在させ、`set_training`／`named_parameters` を dyn 経由で呼べる
/// ことを確認する（既存 `as_linear`／`as_relu` と同じ dyn 呼び出し
/// パターン）。
#[test]
fn dyn_module_set_training_and_named_parameters() {
    let linear = Linear::new(3, 2, true, 42).expect("valid ctor args");
    let mut layers: Vec<Box<dyn Module>> = vec![Box::new(linear), Box::new(Relu)];

    for layer in &mut layers {
        layer.set_training(false);
    }

    // `Linear::named_parameters()` は weight/bias を返し、`Relu` は空。
    assert_eq!(layers[0].named_parameters().len(), 2);
    assert!(layers[1].named_parameters().is_empty());
}

/// `Linear`（bias あり）の命名契約: `weight` → `bias` の順で、各参照が
/// accessor（`weight()`／`bias()`）と同一ポインタであること。
#[test]
fn linear_named_parameters_with_bias() {
    let linear = Linear::new(3, 2, true, 42).expect("valid ctor args");
    let params = linear.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "weight");
    assert!(std::ptr::eq(params[0].1, linear.weight()));
    assert_eq!(params[1].0, "bias");
    assert!(std::ptr::eq(
        params[1].1,
        linear.bias().expect("bias=true で構築した")
    ));
}

/// `Linear`（bias なし）の命名契約: `weight` のみ。
#[test]
fn linear_named_parameters_without_bias() {
    let linear = Linear::new(3, 2, false, 42).expect("valid ctor args");
    let params = linear.named_parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "weight");
    assert!(linear.bias().is_none());
}

/// `RmsNorm`（affine あり／なし）の命名契約: `weight`（`Some` の場合の
/// み）。
#[test]
fn rms_norm_named_parameters() {
    let with_affine = RmsNorm::new(4, 1e-5).expect("valid ctor args");
    let params = with_affine.named_parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "weight");
    assert!(std::ptr::eq(
        params[0].1,
        with_affine.weight().expect("affine=true で構築した")
    ));

    let without_affine = RmsNorm::without_affine(1e-5).expect("valid ctor args");
    assert!(without_affine.named_parameters().is_empty());
}

/// `LayerNorm`（affine あり／なし）の命名契約: `weight` → `bias`
/// の順（各 `Some` の場合のみ）。
#[test]
fn layer_norm_named_parameters() {
    use fandhe_ai_autodiff::nn::LayerNorm;

    let with_affine = LayerNorm::new(4, 1e-5).expect("valid ctor args");
    let params = with_affine.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "weight");
    assert_eq!(params[1].0, "bias");

    let without_affine = LayerNorm::without_affine(1e-5).expect("valid ctor args");
    assert!(without_affine.named_parameters().is_empty());
}

/// `MultiheadAttention` の命名契約: `q_proj.*` → `k_proj.*` →
/// `v_proj.*` → `out_proj.*`（各 `weight`→`bias`）の 4 層 × 2 =
/// 8 エントリ（bias あり構成）。各参照が対応する accessor
/// （`q_proj().weight()` 等）と同一ポインタであることも確認する。
#[test]
fn multihead_attention_named_parameters() {
    let mha = MultiheadAttention::new(4, 2, true, 7).expect("valid ctor args");
    let params = mha.named_parameters();
    assert_eq!(params.len(), 8);
    let expected_names = [
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
    ];
    for (i, expected) in expected_names.iter().enumerate() {
        assert_eq!(&params[i].0, expected);
    }
    assert!(std::ptr::eq(params[0].1, mha.q_proj().weight()));
    assert!(std::ptr::eq(
        params[1].1,
        mha.q_proj().bias().expect("bias=true")
    ));
    assert!(std::ptr::eq(params[6].1, mha.out_proj().weight()));
}

/// `Rnn`／`Lstm`／`Gru`（bias あり）の命名契約: `cell.weight_ih` →
/// `cell.weight_hh` → `cell.bias_ih` → `cell.bias_hh`。参照が
/// `cell().weight_ih()` 等と同一ポインタであることも確認する。
#[test]
fn rnn_family_named_parameters_with_bias() {
    let rnn = Rnn::new(3, 5, true, 11).expect("valid ctor args");
    let params = rnn.named_parameters();
    assert_eq!(params.len(), 4);
    assert_eq!(params[0].0, "cell.weight_ih");
    assert_eq!(params[1].0, "cell.weight_hh");
    assert_eq!(params[2].0, "cell.bias_ih");
    assert_eq!(params[3].0, "cell.bias_hh");
    assert!(std::ptr::eq(params[0].1, rnn.cell().weight_ih()));
    assert!(std::ptr::eq(
        params[2].1,
        rnn.cell().bias_ih().expect("bias=true")
    ));

    let lstm = Lstm::new(3, 5, true, 12).expect("valid ctor args");
    let lstm_params = lstm.named_parameters();
    assert_eq!(lstm_params.len(), 4);
    assert_eq!(lstm_params[0].0, "cell.weight_ih");
    assert_eq!(lstm_params[3].0, "cell.bias_hh");

    let gru = Gru::new(3, 5, true, 13).expect("valid ctor args");
    let gru_params = gru.named_parameters();
    assert_eq!(gru_params.len(), 4);
    assert_eq!(gru_params[0].0, "cell.weight_ih");
    assert_eq!(gru_params[3].0, "cell.bias_hh");
}

/// `Rnn`（bias なし）の命名契約: `cell.weight_ih` → `cell.weight_hh`
/// のみ（bias 系は列挙されない）。
#[test]
fn rnn_named_parameters_without_bias() {
    let rnn = Rnn::new(3, 5, false, 11).expect("valid ctor args");
    let params = rnn.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "cell.weight_ih");
    assert_eq!(params[1].0, "cell.weight_hh");
}

/// 不変性: `set_training` を切り替えても `Linear::bind(&tape).forward`
/// の出力が bit 完全一致すること（本 issue は数値経路に触れないため
/// の構造的な保証を固定する）。
#[test]
fn set_training_does_not_change_forward_output() {
    let mut linear = Linear::new(3, 2, true, 42).expect("valid ctor args");
    let x = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

    let tape_before = Tape::new_with_ops(common::naive_ops());
    let xv_before = tape_before.var(&x);
    let out_before = linear
        .bind(&tape_before)
        .forward(&xv_before)
        .unwrap()
        .to_tensor();

    linear.set_training(false);
    assert!(
        linear.training(),
        "Linear は無状態のため set_training(false) 後も training()==true"
    );

    let tape_after = Tape::new_with_ops(common::naive_ops());
    let xv_after = tape_after.var(&x);
    let out_after = linear
        .bind(&tape_after)
        .forward(&xv_after)
        .unwrap()
        .to_tensor();

    assert_eq!(out_before.as_slice(), out_after.as_slice());
}

/// BatchNorm1d／2d（イシュー #1732・親 #1608）は本クレート内で初めて
/// train／eval でモードにより挙動が変わる層のため、`Module::
/// set_training`／`training` の trait doc「モード依存層は必ず
/// オーバーライドし自層のフィールドへ実際に保持すること」契約を
/// 満たすことを確認する（無状態モジュールの既定契約とは対照的な
/// テスト）。数値経路（forward の出力値そのもの）に触れるため
/// CUDA／Metal 実機 parity は未実測のまま Mac／GB10 セッションへ
/// 申し送り（`docs/batch-norm-ops-design.md`）。

#[test]
fn batch_norm_set_training_actually_changes_training() {
    let mut bn = BatchNorm1d::new(3, 1e-5, 0.1).expect("valid ctor args");
    assert!(
        bn.training(),
        "既定は train モード（PyTorch Module.training 初期値と同じ）"
    );
    bn.set_training(false);
    assert!(
        !bn.training(),
        "BatchNorm は状態を持つため set_training(false) 後は training()==false"
    );
    bn.set_training(true);
    assert!(bn.training());
}

/// `named_parameters` は `weight`／`bias` のみ列挙し、running stats
/// （buffer）は含めない契約（`nn::batch_norm` モジュール doc
/// comment 参照）。
#[test]
fn batch_norm_named_parameters_excludes_running_stats() {
    let bn = BatchNorm1d::new(3, 1e-5, 0.1).expect("valid ctor args");
    let params = bn.named_parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].0, "weight");
    assert_eq!(params[1].0, "bias");

    let without_affine = BatchNorm1d::without_affine(3, 1e-5, 0.1).expect("valid ctor args");
    assert!(without_affine.named_parameters().is_empty());
}

/// `Module::forward`（tape 経路）と `Module::forward_host`（tape 不要
/// 経路）が同一入力・同一モードで bit 完全一致することを確認する
/// （`RmsNorm`／`LayerNorm` と同じ構造的保証。`NaiveOps` は
/// `batch_norm_train`／`batch_norm_infer` をオーバーライドしないため
/// `eval::batch_norm_train_channels`／`batch_norm_infer_channels` への
/// フォールバック経路を両方（tape 経路・tape 不要経路）で叩く）。
#[test]
fn batch_norm_forward_and_forward_host_are_bit_identical_train_mode() {
    let bn = BatchNorm1d::new(2, 0.0, 0.1).expect("valid ctor args");
    let x = Tensor::new(vec![1.0, 2.0, -1.0, 0.5, 3.0, -0.5], &[3, 2]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_tape = bn.bind(&tape).forward(&xv).unwrap().to_tensor();

    let via_host =
        <BatchNorm1d as Module>::forward_host(&bn, common::naive_ops().as_ref(), &x).unwrap();

    assert_eq!(via_tape.as_slice(), via_host.as_slice());
}

/// eval モード版の bit 完全一致（`forward_host` は
/// `core.training()==false` のとき running stats を固定統計として
/// 使う）。
#[test]
fn batch_norm_forward_and_forward_host_are_bit_identical_eval_mode() {
    let mut bn = BatchNorm2d::new(2, 1e-5, 0.1).expect("valid ctor args");
    bn.set_training(false);
    let x = Tensor::new(vec![1.0f32; 16], &[2, 2, 2, 2]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_tape = bn.bind(&tape).forward(&xv).unwrap().to_tensor();

    let via_host =
        <BatchNorm2d as Module>::forward_host(&bn, common::naive_ops().as_ref(), &x).unwrap();

    assert_eq!(via_tape.as_slice(), via_host.as_slice());
}

/// train モードの `forward_host` は running stats を実際に更新する
/// （`nn::batch_norm` モジュール doc comment「running stats 更新契約」）。
#[test]
fn batch_norm_forward_host_updates_running_stats_in_train_mode() {
    let bn = BatchNorm1d::new(2, 0.0, 1.0).expect("valid ctor args");
    let x = Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap();
    assert_eq!(bn.num_batches_tracked(), 0);
    <BatchNorm1d as Module>::forward_host(&bn, common::naive_ops().as_ref(), &x).unwrap();
    assert_eq!(bn.num_batches_tracked(), 1);
}

/// rank 限定契約: `BatchNorm1d` は rank 4 を拒否し、`BatchNorm2d` は
/// rank 2 を拒否する（`Var::batch_norm` 自体は rank 2〜4 を一様に
/// 受理するため、本層が forward 時に追加検査する）。
#[test]
fn batch_norm_module_rejects_wrong_rank_via_forward_host() {
    let bn1d = BatchNorm1d::new(2, 1e-5, 0.1).expect("valid ctor args");
    let x_rank4 = Tensor::new(vec![1.0f32; 8], &[1, 2, 2, 2]).unwrap();
    let err = <BatchNorm1d as Module>::forward_host(&bn1d, common::naive_ops().as_ref(), &x_rank4)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(
            fandhe_ai_tensor_core::ShapeError::RankMismatch { .. }
        )
    ));

    let bn2d = BatchNorm2d::new(2, 1e-5, 0.1).expect("valid ctor args");
    let x_rank2 = Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap();
    let err = <BatchNorm2d as Module>::forward_host(&bn2d, common::naive_ops().as_ref(), &x_rank2)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_autodiff::AutodiffError::Shape(
            fandhe_ai_tensor_core::ShapeError::RankMismatch { .. }
        )
    ));
}
