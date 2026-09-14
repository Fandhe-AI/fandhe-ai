//! 受け入れ条件「Embedding の forward／backward が期待値と一致する」を
//! 直接検証する統合テスト（イシュー #1604）。
//!
//! - forward: 1-D／2-D ids で手計算期待値と突合（`Op::Embedding` の
//!   ノード shape が常に `[N, D]` であり、1-D 以外は `Var::reshape` で
//!   呼び出し側 shape へ戻ることの確認を兼ねる）。
//! - backward（解析）: 重複 id を含む例で `d_weight` が
//!   `scatter_add`（`ScatterReduce::Add` の決定的集約契約）の手計算と
//!   厳密一致することを確認する。
//! - backward（数値微分）: `weight.embedding(ids) → mul(self) → sum` の
//!   合成関数で解析勾配と中央差分を突合する（`padding_idx = None`
//!   限定。`tests/backward.rs` と同じ `H`／`assert_grad_close` 方針）。
//! - `padding_idx`: forward は当該行の現在値をそのまま返す（forward
//!   自体は特別扱いしない）が、backward の当該行は明示ゼロへ上書き
//!   されることを確認する。
//! - エラー経路: 範囲外 id・負 id・weight rank != 2・
//!   `padding_idx >= num_embeddings` で型付きエラーが返る（panic しない）。
//! - 空 ids: `[0]` → `[0, D]` を返し backward がゼロ勾配になる。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn i32t(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

// --- 1. forward 期待値一致 ---

#[test]
fn forward_1d_ids_matches_hand_computed_expectation() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let ids = i32t(vec![2, 0, 2], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let out = wv.embedding(&ids, None).unwrap();
    let out_t = out.to_tensor();

    assert_eq!(out_t.shape(), &[3, 2]);
    assert_eq!(dense_vec(&out_t), vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn forward_2d_ids_reshapes_to_ids_shape_plus_embedding_dim() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let ids = i32t(vec![0, 1, 2, 1], &[2, 2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let out = wv.embedding(&ids, None).unwrap();
    let out_t = out.to_tensor();

    assert_eq!(out_t.shape(), &[2, 2, 2]);
    assert_eq!(
        dense_vec(&out_t),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0]
    );
}

#[test]
fn forward_scalar_id_reshapes_to_embedding_dim_only() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let id = Tensor::<i32>::new(vec![1], &[]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let out = wv.embedding(&id, None).unwrap();
    let out_t = out.to_tensor();

    assert_eq!(out_t.shape(), &[2]);
    assert_eq!(dense_vec(&out_t), vec![3.0, 4.0]);
}

// --- 2. backward（解析）: 重複 id の scatter_add 決定的集約契約 ---

#[test]
fn backward_analytic_matches_scatter_add_with_duplicate_ids() {
    let weight = t(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]);
    // id=1 が 2 回出現（index 0・2）。upstream は sum の勾配で全て 1。
    let ids = i32t(vec![1, 0, 1], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let out = wv.embedding(&ids, None).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("weight は loss に到達する");

    // row0: id=0 が 1 回 -> [1,1]。row1: id=1 が 2 回 -> [2,2]。
    // row2: 未使用 -> [0,0]。
    assert_eq!(dense_vec(dw), vec![1.0, 1.0, 2.0, 2.0, 0.0, 0.0]);
}

// --- 3. backward（数値微分）: padding_idx = None 限定 ---

const H: f64 = 1e-3;
const TAU: f32 = 1e-4;
const REL_TOL: f32 = 1e-2;
const ABS_TOL: f32 = 1e-3;

fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
    assert_eq!(
        analytic.shape(),
        numeric.shape(),
        "{label}: shape が一致しない"
    );
    let shape = analytic.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for _flat in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel_ok = diff <= REL_TOL * nv.abs().max(av.abs());
        let abs_ok = diff <= ABS_TOL;
        assert!(
            rel_ok || abs_ok || diff <= TAU,
            "{label}[{index:?}]: analytic={av} numeric={nv} diff={diff}"
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn numeric_grad(target_tensor: &Tensor<f32>, perturb: impl Fn(Tensor<f32>) -> f32) -> Tensor<f32> {
    let shape = target_tensor.shape().to_vec();
    let numel: usize = shape.iter().product();
    let mut data: Vec<f32> = (0..numel)
        .map(|flat| {
            let mut idx = vec![0usize; shape.len()];
            let mut rem = flat;
            for axis in (0..shape.len()).rev() {
                idx[axis] = rem % shape[axis];
                rem /= shape[axis];
            }
            target_tensor.get(&idx).unwrap_or(0.0)
        })
        .collect();
    let mut grad = vec![0f32; numel];
    for i in 0..numel {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = perturb(t(data.clone(), &shape)) as f64;
        data[i] = (orig - H) as f32;
        let lm = perturb(t(data.clone(), &shape)) as f64;
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    t(grad, &shape)
}

#[test]
fn backward_matches_numeric_gradient_with_duplicate_ids() {
    let weight0 = t(vec![0.3, -0.2, 0.1, 0.5, -0.4, 0.2], &[3, 2]);
    let ids = i32t(vec![2, 0, 2, 1], &[4]);

    let forward = |weight: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let wv = tape.var(&weight);
        let out = wv.embedding(&ids, None).unwrap();
        let sq = out.mul(&out).unwrap();
        scalar(&sq.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight0);
    let out = wv.embedding(&ids, None).unwrap();
    let sq = out.mul(&out).unwrap();
    let loss = sq.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("weight は loss に到達する");

    let num_dw = numeric_grad(&weight0, forward);
    assert_grad_close("embedding dWeight", dw, &num_dw);
}

// --- 4. padding_idx: forward は素通し・backward は明示ゼロ ---

#[test]
fn padding_idx_forward_returns_current_row_value() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let ids = i32t(vec![0, 1, 2, 1], &[4]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    // padding_idx=1 でも forward は行 1 の現在値（[3,4]）をそのまま返す。
    let out = wv.embedding(&ids, Some(1)).unwrap();
    assert_eq!(
        dense_vec(&out.to_tensor()),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0]
    );
}

#[test]
fn padding_idx_backward_zeroes_that_row_only() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    // id=1（padding_idx）が index 1・3 の 2 回、id=0・id=2 が 1 回ずつ。
    let ids = i32t(vec![0, 1, 2, 1], &[4]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let out = wv.embedding(&ids, Some(1)).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("weight は loss に到達する");

    // row0: [1,1]（id=0 が 1 回）。row1（padding_idx）: 明示ゼロ。
    // row2: [1,1]（id=2 が 1 回）。
    assert_eq!(dense_vec(dw), vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
}

// --- 5. エラー経路 ---

#[test]
fn weight_rank_mismatch_is_rejected() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
    let ids = i32t(vec![0], &[1]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let err = wv.embedding(&ids, None).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn index_out_of_range_is_rejected() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let ids = i32t(vec![2], &[1]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let err = wv.embedding(&ids, None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn negative_index_is_rejected() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let ids = i32t(vec![-1], &[1]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let err = wv.embedding(&ids, None).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn padding_idx_out_of_range_is_rejected() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let ids = i32t(vec![0], &[1]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let err = wv.embedding(&ids, Some(2)).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- 6. 空 ids ---

#[test]
fn empty_ids_returns_empty_output_and_zero_gradient() {
    let weight = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let ids = i32t(vec![], &[0]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let wv = tape.var(&weight);
    let out = wv.embedding(&ids, None).unwrap();
    let out_t = out.to_tensor();
    assert_eq!(out_t.shape(), &[0, 2]);

    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("weight は loss に到達する");
    assert_eq!(dense_vec(dw), vec![0.0, 0.0, 0.0, 0.0]);
}

// --- 7. nn::Embedding の bind/forward 経由（薄い委譲の確認） ---

#[test]
fn nn_embedding_bind_forward_matches_var_embedding() {
    use fandhe_ai_autodiff::nn::Embedding;

    let weight = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let emb = Embedding::from_parameters(weight.clone(), None).unwrap();
    let ids = i32t(vec![2, 0], &[2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = emb.bind(&tape);
    let out = vars.forward(&ids).unwrap();

    let tape2 = Tape::new_with_ops(common::naive_ops());
    let wv2 = tape2.var(&weight);
    let expected = wv2.embedding(&ids, None).unwrap();

    assert_eq!(
        dense_vec(&out.to_tensor()),
        dense_vec(&expected.to_tensor())
    );
}
