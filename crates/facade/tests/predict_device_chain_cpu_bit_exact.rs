//! `DeviceParamStore::predict_device_chain`（イシュー #1688・親 #1581／
//! #1580・`docs/inference-chain-single-sync-design.md`）の CPU 上での
//! bit 完全一致回帰。
//!
//! `Sequential::predict_resident` は本イシューにより、`self.layers` が
//! `Linear`（＋`ReLU` 融合）のみで構成できる場合、内部を単一同期チェーン
//! （`build_device_chain_steps` → `predict_device_chain`）へ優先的に
//! 差し替える（バックエンドが対応しない場合は現行経路
//! `forward_from_flat_leaves` へ全体フォールバックする。決定 7）。
//!
//! CPU バックエンドは `BackendOps::linear_forward_device`（本番カーネル。
//! `crates/backend-cpu/src/ops.rs`）を実装済みのため、`predict_resident`
//! は常に chain 経路を通る。本テストはその chain 経路の出力が、
//! `Sequential::forward_resident`（`linear_forward_with_activation` を
//! 層ごとに呼ぶ旧経路）の出力と **bit 完全一致**することを確認する
//! （decision 6「新チェーン出力 ≡ 現行 `predict_resident` 出力の bit
//! 同一」）。あわせて `predict_resident` を連続 2 回呼び出し、run-to-run
//! bit 同一（decision 6 の 1 点目）も確認する。
//!
//! モデル A（`Linear→ReLU→Linear`）は chain の `ReLU` 融合経路を、
//! モデル B（`Linear→Linear`。層間に活性化なし）は `Activation::None`
//! のまま 2 層を辿る非融合の複数層経路をそれぞれ検証する。
//!
//! **`bias: None` 分岐について**: `Sequential::add_linear`（唯一の公開
//! 層追加 API）は常に `bias=true` の `Linear` を積む（PyTorch
//! `nn.Linear` の既定 `bias=True` に揃える設計。`Sequential::add_linear`
//! doc 参照）ため、bias なし層は公開 API からは構成できない。
//! `build_device_chain_steps` の `bias: None` 分岐は `crates/facade/src/
//! compat/sequential.rs` 内の同一クレート単体テスト（`Sequential { layers:
//! vec![Box::new(Linear::new(.., false, ..)), ..] }` で直接構築可能）で
//! 別途カバーする対象とする。

use fandhe_ai::compat::Sequential;
use fandhe_ai_tensor_core::Tensor;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

/// `model` を chain 経路（`predict_resident`）・旧経路
/// （`forward_resident` 経由。`&mut store` を要するため呼び出しごとに
/// 独立した `DeviceParamStore` を使う）の双方で forward し、出力が
/// bit 完全一致することを確認する共通ロジック。
fn assert_chain_matches_legacy_bit_exact(model: &Sequential, input: &Tensor<f32>) {
    // chain 経路（`predict_resident`。CPU は `linear_forward_device` 実装済み
    // のため常にこちらを通る）。
    let chain_init_tape = fandhe_ai::tape();
    let chain_store = model.init_device_param_store(&chain_init_tape).unwrap();
    drop(chain_init_tape);
    let chain_output = model.predict_resident(&chain_store, input).unwrap();

    // run-to-run bit 同一（decision 6）: 同一 store・同一入力で 2 回目も
    // 完全に同じ出力になることを確認する。
    let chain_output_again = model.predict_resident(&chain_store, input).unwrap();
    assert_eq!(
        dense_vec(&chain_output),
        dense_vec(&chain_output_again),
        "predict_resident の run-to-run 出力が bit 一致しない"
    );

    // 旧経路（`forward_resident`。`register_resident_params` を通すため
    // `&mut store` が必要。`abandon_pending_forward` で forward 後の
    // pending を明示的にクリアしてから比較する（本テストは backward・
    // step を続けないため）。
    let legacy_init_tape = fandhe_ai::tape();
    let mut legacy_store = model.init_device_param_store(&legacy_init_tape).unwrap();
    drop(legacy_init_tape);
    let legacy_tape = fandhe_ai::tape();
    let input_var = legacy_tape.var(input);
    let legacy_output = model
        .forward_resident(&legacy_tape, &input_var, &mut legacy_store)
        .unwrap()
        .to_tensor();
    legacy_store.abandon_pending_forward();

    assert_eq!(
        dense_vec(&chain_output),
        dense_vec(&legacy_output),
        "chain 経路（predict_device_chain）と旧経路（forward_resident）の出力が bit 一致しない"
    );
}

/// モデル A: `Linear(bias)→ReLU→Linear(bias)`。chain の `ReLU` 融合
/// （`Activation::Relu`）を経由する経路を検証する。
#[test]
fn predict_device_chain_matches_legacy_path_bit_exact_relu_fusion() {
    let model = Sequential::new()
        .add_linear(4, 8, 0x1001)
        .unwrap()
        .add_relu()
        .add_linear(8, 3, 0x2002)
        .unwrap();

    let input = tensor(
        (0..2 * 4).map(|i| (i as f32) * 0.1 - 0.5).collect(),
        &[2, 4],
    );

    assert_chain_matches_legacy_bit_exact(&model, &input);
}

/// モデル B: `Linear→Linear`（層間に活性化なし。`Activation::None` の
/// まま 2 層を辿る非融合の複数層経路）を検証する。
#[test]
fn predict_device_chain_matches_legacy_path_bit_exact_no_activation_fusion() {
    let model = Sequential::new()
        .add_linear(4, 8, 0x3003)
        .unwrap()
        .add_linear(8, 3, 0x4004)
        .unwrap();

    let input = tensor(
        (0..2 * 4).map(|i| (i as f32) * 0.07 + 0.2).collect(),
        &[2, 4],
    );

    assert_chain_matches_legacy_bit_exact(&model, &input);
}
