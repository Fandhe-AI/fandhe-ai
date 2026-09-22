//! `compat::Sequential::add_*`（LayerNorm／RmsNorm／BatchNorm1d／
//! BatchNorm2d／Embedding／MultiheadAttention。イシュー #1760・親
//! #1618）の facade 公開面を検証する統合テスト（`compat_sequential_
//! conv.rs` と同型。CPU のみで Linux 実行可能）。
//!
//! - `add_*` の無効引数拒否（各層 1 件ずつ）。
//! - `predict`（tape 不要経路）と `forward`（`fandhe_ai::tape()` 上）が
//!   bit 完全一致（層ごと単体・混在モデル）。
//! - `bind().forward` が手動合成（`Var::layer_norm`／`rms_norm`／
//!   `batch_norm_*`／`embedding`・`MultiheadAttentionVars::new` +
//!   `LinearVars`）と forward・勾配とも bit 完全一致。
//! - 全種混在モデルで `trainable_parameters()` の shape 列 ==
//!   `named_parameters()`・`bind().trainable_vars()`／
//!   `trainable_grads()` の件数一致。
//! - `TransformerEncoderLayer` 単体でも同様に `bind().trainable_vars()`
//!   ／`trainable_grads()` の件数が `trainable_parameters()`（16 件）
//!   と一致する（レビュー指摘の回帰ガード。`self.encoders` を収集し
//!   忘れると 0 件になり黙って学習されない罠を防ぐ）。
//! - SGD 学習ループで loss 減少。
//! - `apply_parameters`: shape 保存更新が `predict` に反映・
//!   BatchNorm の running stats／`training` が in-place 更新後も保持
//!   される（`Rebuilt` 方式〈層丸ごと再構築〉の罠を避けたことの回帰
//!   ガード）。
//! - BatchNorm モード: `eval()` 後の `predict` は running stats を
//!   更新しない・train モードの `M<=1` は `InvalidArgument`。
//! - Embedding の f32 id 入力の拒否（非整数・負・NaN）。
//! - 常駐経路ガード: 新規 6 種を含むモデルで `init_device_param_store`
//!   が `Unsupported`。

use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

const SEED1: u64 = 0x1111_2222;
const SEED2: u64 = 0x3333_4444;

// --- add_* の無効引数拒否 ---

#[test]
fn add_layer_norm_rejects_negative_eps() {
    let err = Sequential::new()
        .add_layer_norm(4, -1.0)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_rms_norm_rejects_nan_eps() {
    let err = Sequential::new()
        .add_rms_norm(4, f32::NAN)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_batch_norm1d_rejects_zero_num_features() {
    let err = Sequential::new()
        .add_batch_norm1d(0, 1e-5, 0.1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_batch_norm2d_rejects_momentum_out_of_range() {
    let err = Sequential::new()
        .add_batch_norm2d(4, 1e-5, 1.5)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_embedding_rejects_zero_num_embeddings() {
    let err = Sequential::new()
        .add_embedding(0, 4, None, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_embedding_rejects_padding_idx_out_of_range() {
    let err = Sequential::new()
        .add_embedding(4, 3, Some(4), SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_multihead_attention_rejects_indivisible_heads() {
    let err = Sequential::new()
        .add_multihead_attention(6, 4, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_transformer_encoder_rejects_indivisible_heads() {
    let err = Sequential::new()
        .add_transformer_encoder(6, 4, 8, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_transformer_encoder_rejects_zero_dim_feedforward() {
    let err = Sequential::new()
        .add_transformer_encoder(4, 2, 0, SEED1)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- predict と forward の bit 完全一致（層ごと単体） ---

#[test]
fn layer_norm_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_linear(4, 4, SEED1)
        .unwrap()
        .add_layer_norm(4, 1e-5)
        .unwrap()
        .add_relu()
        .add_linear(4, 2, SEED2)
        .unwrap();
    let x = tensor((0..3 * 4).map(|i| i as f32 * 0.1 - 0.5).collect(), &[3, 4]);

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(predicted.shape(), &[3, 2]);
    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn rms_norm_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_linear(4, 4, SEED1)
        .unwrap()
        .add_rms_norm(4, 1e-6)
        .unwrap();
    let x = tensor((0..2 * 4).map(|i| i as f32 * 0.2 - 0.3).collect(), &[2, 4]);

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn batch_norm1d_predict_matches_forward_bit_exact_rank2() {
    let model = Sequential::new().add_batch_norm1d(3, 1e-5, 0.1).unwrap();
    let x = tensor(
        (0..4 * 3).map(|i| (i as f32) * 0.15 - 0.4).collect(),
        &[4, 3],
    );

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn batch_norm1d_predict_matches_forward_bit_exact_rank3() {
    let model = Sequential::new().add_batch_norm1d(2, 1e-5, 0.1).unwrap();
    let x = tensor(
        (0..2 * 2 * 5).map(|i| (i as f32) * 0.05 - 0.2).collect(),
        &[2, 2, 5],
    );

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn batch_norm2d_predict_matches_forward_bit_exact() {
    let model = Sequential::new().add_batch_norm2d(3, 1e-5, 0.1).unwrap();
    let x = tensor(
        (0..2 * 3 * 4 * 4)
            .map(|i| (i as f32) * 0.01 - 0.2)
            .collect(),
        &[2, 3, 4, 4],
    );

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn embedding_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_embedding(6, 4, None, SEED1)
        .unwrap()
        .add_linear(4, 2, SEED2)
        .unwrap();
    // id は f32 として詰める（`add_embedding` doc「入力契約」参照）。
    let ids = tensor(vec![0.0, 3.0, 5.0, 1.0], &[4]);

    let predicted = model.predict(&ids).unwrap();
    let tape = fandhe_ai::tape();
    let idv = tape.var(&ids);
    let forwarded = model.forward(&tape, &idv).unwrap().to_tensor();

    assert_eq!(predicted.shape(), &[4, 2]);
    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn multihead_attention_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_multihead_attention(4, 2, SEED1)
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    );

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(predicted.shape(), &[2, 3, 4]);
    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn transformer_encoder_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_transformer_encoder(4, 2, 8, SEED1)
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    );

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(predicted.shape(), &[2, 3, 4]);
    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn transformer_encoder_bind_trainable_vars_and_grads_count_matches_trainable_parameters() {
    // レビュー指摘の回帰ガード: `SequentialVars::trainable_vars`／
    // `trainable_grads` が `self.encoders`（`TransformerEncoderLayer`）
    // を収集していないと `bound.trainable_vars().len()` が 0 になり、
    // `trainable_parameters()`（`named_parameters` への汎用委譲。16
    // パラメータ = self_attn.q/k/v/out〈weight+bias〉8 + linear1/linear2
    // 〈weight+bias〉4 + norm1/norm2〈weight+bias〉4）と件数が食い違う。
    let model = Sequential::new()
        .add_transformer_encoder(4, 2, 8, SEED1)
        .unwrap();
    let x = tensor(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    );
    let target = tensor(vec![0.0f32; 2 * 3 * 4], &[2, 3, 4]);

    let param_count = model.trainable_parameters().len();
    assert_eq!(param_count, 16);

    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let xv = tape.var(&x);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &xv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();

    assert_eq!(bound.trainable_vars().len(), param_count);

    let grads = tape.backward(&loss).unwrap();
    let grad_refs = bound.trainable_grads(&grads).unwrap();
    assert_eq!(grad_refs.len(), param_count);
}

// --- 全種混在モデル（Embedding→LayerNorm→RmsNorm→BatchNorm1d→
//     Linear。いずれも `[N, 4]` 形状で連鎖できる組み合わせ） ---

const MIXED_DIM: usize = 4;

fn mixed_model(embedding_seed: u64, linear_seed: u64) -> Sequential {
    Sequential::new()
        .add_embedding(8, MIXED_DIM, None, embedding_seed)
        .unwrap()
        .add_layer_norm(MIXED_DIM, 1e-5)
        .unwrap()
        .add_rms_norm(MIXED_DIM, 1e-6)
        .unwrap()
        .add_batch_norm1d(MIXED_DIM, 1e-5, 0.1)
        .unwrap()
        .add_linear(MIXED_DIM, 2, linear_seed)
        .unwrap()
}

fn mixed_ids() -> Tensor<f32> {
    tensor(vec![0.0, 2.0, 5.0, 7.0, 1.0], &[5])
}

#[test]
fn mixed_model_trainable_parameters_order_matches_named_parameters() {
    let model = mixed_model(SEED1, SEED2);

    let named = model.named_parameters();
    let named_shapes: Vec<Vec<usize>> = named
        .iter()
        .map(|(_, t)| t.contiguous().shape().to_vec())
        .collect();
    let trainable = model.trainable_parameters();
    let trainable_shapes: Vec<Vec<usize>> = trainable
        .iter()
        .map(|t| t.contiguous().shape().to_vec())
        .collect();

    assert_eq!(named_shapes, trainable_shapes);
    // Embedding(weight)・LayerNorm(weight,bias)・RmsNorm(weight)・
    // BatchNorm1d(weight,bias)・Linear(weight,bias) = 8 件。
    assert_eq!(trainable_shapes.len(), 8);
}

#[test]
fn mixed_model_bind_trainable_vars_and_grads_count_matches_trainable_parameters() {
    let model = mixed_model(SEED1, SEED2);
    let ids = mixed_ids();
    let target = tensor(vec![0.0f32; 5 * 2], &[5, 2]);

    let param_count = model.trainable_parameters().len();
    assert_eq!(param_count, 8);

    let tape = fandhe_ai::tape();
    let bound = model.bind(&tape);
    let idv = tape.var(&ids);
    let tv = tape.var(&target);
    let pred = bound.forward(&tape, &idv).unwrap();
    let loss = pred.mse_loss(&tv).unwrap();

    assert_eq!(bound.trainable_vars().len(), param_count);

    let grads = tape.backward(&loss).unwrap();
    let grad_refs = bound.trainable_grads(&grads).unwrap();
    assert_eq!(grad_refs.len(), param_count);
}

#[test]
fn mixed_model_sgd_training_loop_reduces_loss() {
    let mut model = mixed_model(SEED1, SEED2);
    let ids = mixed_ids();
    let target = tensor(
        (0..5 * 2).map(|i| ((i % 5) as f32) * 0.1 - 0.2).collect(),
        &[5, 2],
    );

    let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
    let mut losses = Vec::new();

    for _ in 0..30 {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let idv = tape.var(&ids);
            let tv = tape.var(&target);
            let pred = bound.forward(&tape, &idv).unwrap();
            let loss = pred.mse_loss(&tv).unwrap();
            losses.push(loss.to_tensor().get(&[]).unwrap());

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first * 0.9,
        "loss should decrease: first={first} last={last}"
    );
}

#[test]
fn mixed_model_init_device_param_store_rejects_resident_unsupported_layers() {
    let model = mixed_model(SEED1, SEED2);
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)));
}

// --- apply_parameters: shape 保存更新の反映・BatchNorm 状態保持 ---

#[test]
fn apply_parameters_updates_mixed_model_weight_used_by_subsequent_predict() {
    let mut model = mixed_model(SEED1, SEED2);
    let ids = mixed_ids();
    let before = model.predict(&ids).unwrap();

    let current = model.trainable_parameters();
    // Embedding.weight（先頭要素）を全ゼロへ置き換える。他は元の値の
    // まま渡す（shape 保存置換のみが要件であり値自体を変える必要は
    // ない・全パラメータを 1 パスで渡す契約を守るため）。
    let mut updated: Vec<Tensor<f32>> = current.iter().map(|t| (*t).clone()).collect();
    let emb_shape = updated[0].shape().to_vec();
    let emb_len: usize = emb_shape.iter().product();
    updated[0] = tensor(vec![0.0f32; emb_len], &emb_shape);
    model.apply_parameters(updated).unwrap();

    let after = model.predict(&ids).unwrap();
    assert_ne!(
        dense_vec(&before),
        dense_vec(&after),
        "embedding weight をゼロ化したのに predict 出力が変化していない"
    );
}

#[test]
fn apply_parameters_rejects_layer_norm_weight_shape_change() {
    let mut model = Sequential::new().add_layer_norm(4, 1e-5).unwrap();
    let wrong_weight = tensor(vec![1.0f32; 3], &[3]);
    let bias = tensor(vec![0.0f32; 4], &[4]);
    let err = model
        .apply_parameters(vec![wrong_weight, bias])
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn apply_parameters_preserves_batch_norm_running_stats_and_mode() {
    let mut model = Sequential::new().add_batch_norm1d(2, 1e-5, 0.5).unwrap();

    // train モードで複数回 forward し running stats を既定値（mean=0,
    // var=1）から動かす。
    for i in 0..5u32 {
        let x = tensor(
            vec![
                1.0 + i as f32,
                2.0 + i as f32,
                3.0 + i as f32,
                4.0 + i as f32,
            ],
            &[2, 2],
        );
        model.predict(&x).unwrap();
    }

    model.eval();
    let probe = tensor(vec![0.5, -0.5, 1.5, -1.5], &[2, 2]);
    let before = model.predict(&probe).unwrap();

    // 現在の weight／bias をそのまま渡す in-place 更新（値は変えない）。
    // `Rebuilt`（層丸ごと再構築）方式だと running stats／
    // num_batches_tracked がリセットされ eval 出力が変わってしまう
    // （`apply_parameters` doc「`Rebuilt` 方式…罠を避けたことの回帰
    // ガード」参照）。
    let current = model.trainable_parameters();
    let updated: Vec<Tensor<f32>> = current.iter().map(|t| (*t).clone()).collect();
    model.apply_parameters(updated).unwrap();

    let after = model.predict(&probe).unwrap();
    assert_eq!(
        dense_vec(&before),
        dense_vec(&after),
        "apply_parameters 後に running stats／training モードが変化した \
         （BatchNorm state 保持契約の違反）"
    );
}

// --- BatchNorm モード契約 ---

#[test]
fn batch_norm_eval_predict_does_not_update_running_stats() {
    let mut model = Sequential::new().add_batch_norm1d(2, 1e-5, 0.5).unwrap();
    model.eval();

    let x1 = tensor(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let out1 = model.predict(&x1).unwrap();
    let out2 = model.predict(&x1).unwrap();

    // eval モードは running stats を更新しないため、同一入力に対し
    // 同一出力を返す（train モードなら 2 回目の running stats が
    // 1 回目と異なり得るため出力も変わりうる）。
    assert_eq!(dense_vec(&out1), dense_vec(&out2));
}

#[test]
fn batch_norm1d_train_forward_rejects_batch_size_one() {
    let model = Sequential::new().add_batch_norm1d(3, 1e-5, 0.1).unwrap();
    // train モード（既定）・rank2 `[N=1, C=3]` は M=N*spatial=1<=1。
    let x = tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- Embedding f32 id の拒否 ---

#[test]
fn embedding_rejects_non_integer_id() {
    let model = Sequential::new().add_embedding(4, 2, None, SEED1).unwrap();
    let ids = tensor(vec![1.5], &[1]);
    let err = model.predict(&ids).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn embedding_rejects_negative_id() {
    let model = Sequential::new().add_embedding(4, 2, None, SEED1).unwrap();
    let ids = tensor(vec![-1.0], &[1]);
    let err = model.predict(&ids).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn embedding_rejects_nan_id() {
    let model = Sequential::new().add_embedding(4, 2, None, SEED1).unwrap();
    let ids = tensor(vec![f32::NAN], &[1]);
    let err = model.predict(&ids).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}
