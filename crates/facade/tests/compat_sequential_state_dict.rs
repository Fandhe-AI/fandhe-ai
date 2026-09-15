//! `compat::Sequential::state_dict`／`load_state_dict`（PyTorch
//! `Module.state_dict()`／`load_state_dict()` 相当。イシュー #1752）の
//! 公開 API 契約テストを、`fandhe_ai`（facade）経由でのみ検証する。
//!
//! 数値経路（`Op`／`BackendOps`／VJP）を一切追加しない機構のため、
//! parity の代替として「ロード後の `predict`／`bind().forward()` 出力が
//! 元モデルと bit 完全一致する」ことを CPU 上で確認する（CUDA／Metal
//! への申し送りは不要）。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, Tensor, tape};

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)
        .unwrap()
}

fn sample_input() -> Tensor<f32> {
    Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, 4]).unwrap()
}

/// `state_dict()` のキー集合が `named_parameters()` の名前集合と一致し、
/// 各値が `named_parameters()` の tensor と bit 同一。
#[test]
fn state_dict_keys_and_values_match_named_parameters() {
    let model = build_model();
    let named = model.named_parameters();
    let dict = model.state_dict();

    assert_eq!(named.len(), dict.len());
    for (name, tensor) in &named {
        let from_dict = dict
            .get(name.as_str())
            .unwrap_or_else(|| panic!("state_dict に `{name}` が存在しない"));
        assert_eq!(
            tensor.contiguous().as_slice().unwrap(),
            from_dict.contiguous().as_slice().unwrap(),
            "key `{name}` の値が named_parameters と state_dict とで不一致"
        );
    }
}

/// `b.load_state_dict(a.state_dict())` 後、同一入力の `predict` 出力が
/// `a` と bit 完全一致する（parity の代替。CPU）。
#[test]
fn load_state_dict_from_another_model_matches_predict_output() {
    let a = build_model();
    let mut b = Sequential::new()
        .add_linear(4, 8, /* seed = */ 999)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 1000)
        .unwrap();

    let x = sample_input();
    let out_a = a.predict(&x).unwrap();
    let out_b_before = b.predict(&x).unwrap();
    assert_ne!(
        out_a.contiguous().as_slice().unwrap(),
        out_b_before.contiguous().as_slice().unwrap(),
        "異なるシードで初期化した 2 モデルの predict が偶然一致した（テスト前提が崩れている）"
    );

    b.load_state_dict(a.state_dict()).unwrap();
    let out_b_after = b.predict(&x).unwrap();
    assert_eq!(
        out_a.contiguous().as_slice().unwrap(),
        out_b_after.contiguous().as_slice().unwrap(),
        "load_state_dict 後の predict が元モデルと bit 一致しない"
    );
}

/// `load_state_dict` は `bind().forward()`（tape 経路）にも同様に反映
/// される。
#[test]
fn load_state_dict_from_another_model_matches_bind_forward_output() {
    let a = build_model();
    let mut b = Sequential::new()
        .add_linear(4, 8, /* seed = */ 999)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 1000)
        .unwrap();

    b.load_state_dict(a.state_dict()).unwrap();

    let x = sample_input();
    let out_a_predict = a.predict(&x).unwrap();

    let t = tape();
    let bound = b.bind(&t);
    let xv = t.var(&x);
    let out_b_tape = bound.forward(&t, &xv).unwrap();

    assert_eq!(
        out_a_predict.contiguous().as_slice().unwrap(),
        out_b_tape.to_tensor().contiguous().as_slice().unwrap(),
        "tape 経路での forward が predict（tape 不要経路）と一致しない"
    );
}

/// 往復 `model.load_state_dict(model.state_dict())` で `predict` 出力が
/// 不変（no-op であること）。
#[test]
fn load_state_dict_round_trip_is_no_op_for_predict() {
    let mut model = build_model();
    let x = sample_input();
    let before = model.predict(&x).unwrap();

    model.load_state_dict(model.state_dict()).unwrap();

    let after = model.predict(&x).unwrap();
    assert_eq!(
        before.contiguous().as_slice().unwrap(),
        after.contiguous().as_slice().unwrap()
    );
}

fn assert_model_unchanged(model: &Sequential, x: &Tensor<f32>, expected: &Tensor<f32>) {
    let out = model.predict(x).unwrap();
    assert_eq!(
        expected.contiguous().as_slice().unwrap(),
        out.contiguous().as_slice().unwrap(),
        "load_state_dict が Err を返したのに predict 出力が変化した（アトミック性違反）"
    );
}

/// strict 拒否: キー欠落は `Err`・モデル無変更。
#[test]
fn load_state_dict_rejects_missing_key_and_leaves_model_unchanged() {
    let mut model = build_model();
    let x = sample_input();
    let before = model.predict(&x).unwrap();

    let mut state = model.state_dict();
    state.remove("0.bias");
    let err = model
        .load_state_dict(state)
        .expect_err("欠落キーは Err を返すはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_model_unchanged(&model, &x, &before);
}

/// strict 拒否: 余剰キーは `Err`・モデル無変更。
#[test]
fn load_state_dict_rejects_unexpected_key_and_leaves_model_unchanged() {
    let mut model = build_model();
    let x = sample_input();
    let before = model.predict(&x).unwrap();

    let mut state = model.state_dict();
    state.insert(
        "99.weight".to_string(),
        Tensor::new(vec![0.0f32], &[1]).unwrap(),
    );
    let err = model
        .load_state_dict(state)
        .expect_err("余剰キーは Err を返すはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_model_unchanged(&model, &x, &before);
}

/// strict 拒否: shape 不一致は `Err`・モデル無変更。
#[test]
fn load_state_dict_rejects_shape_mismatch_and_leaves_model_unchanged() {
    let mut model = build_model();
    let x = sample_input();
    let before = model.predict(&x).unwrap();

    let mut state = model.state_dict();
    state.insert(
        "0.weight".to_string(),
        Tensor::new(vec![1.0f32; 8], &[2, 4]).unwrap(),
    );
    let err = model
        .load_state_dict(state)
        .expect_err("shape 不一致は Err を返すはず");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_model_unchanged(&model, &x, &before);
}

/// `apply_parameters` 適用後の `state_dict()` が更新後の値を反映する
/// （位置対応契約と名前契約の整合）。
#[test]
fn state_dict_reflects_apply_parameters_updates() {
    let mut model = build_model();
    let updated: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .into_iter()
        .map(|t| {
            let data: Vec<f32> = t
                .contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v + 1.0)
                .collect();
            Tensor::new(data, t.shape()).unwrap()
        })
        .collect();
    model.apply_parameters(updated).unwrap();

    let dict = model.state_dict();
    let weight_0 = dict.get("0.weight").unwrap();
    let named_weight_0 = model
        .named_parameters()
        .into_iter()
        .find(|(name, _)| name == "0.weight")
        .unwrap()
        .1;
    assert_eq!(
        weight_0.contiguous().as_slice().unwrap(),
        named_weight_0.contiguous().as_slice().unwrap()
    );
}

/// `add_dropout` を含むモデルで Dropout がキーに現れない（index は
/// 活性化・Dropout を含む位置のまま）。
#[test]
fn state_dict_excludes_dropout_layer() {
    let model = Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_dropout(0.5)
        .unwrap()
        .add_linear(8, 2, /* seed = */ 43)
        .unwrap();

    let dict = model.state_dict();
    let keys: std::collections::BTreeSet<&str> = dict.keys().map(|k| k.as_str()).collect();
    // Dropout は index 2 に位置するが無状態のためキーに現れない。
    // 最後の Linear は index 3。
    assert_eq!(
        keys,
        ["0.weight", "0.bias", "3.weight", "3.bias"]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
    );
}
