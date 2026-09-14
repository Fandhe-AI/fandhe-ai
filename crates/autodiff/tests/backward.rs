//! 受け入れ条件「合成関数の end-to-end 勾配が期待値と一致する」を直接
//! 検証する統合テスト（TASK-1.5c・イシュー #18）。
//!
//! - 手計算できる小さな合成関数（`mul` の自己参照・`add`）で解析解と
//!   厳密一致を確認する（`mul` は勾配蓄積〈複数経路からの合算〉の検証を
//!   兼ねる）。
//! - MLP 1 層相当（`matmul → add(bias) → relu → mse_loss`）の合成関数で
//!   `Tape::backward` の解析勾配と中央差分（数値微分）を突合する。判定
//!   閾値は `grad.rs` の grad-check テスト（#17）が用いた値をそのまま
//!   再利用し、新しい許容誤差は導入しない（`H=1e-3`・相対 1e-2 または
//!   絶対 1e-3・`τ=1e-4`。承認追跡は Issue #223）。
//! - `Tape::backward`/`Gradients::get` の API 契約（クロステープ検査・
//!   未到達ノード・境界外アクセス・非スカラー loss の暗黙総和射影）を
//!   検証する。
//!
//! `Tape`/`Var` を経由する end-to-end 経路のみを対象とし、PoC-v2-2 の
//! 実測ケース網羅・回帰テスト化は #19（TASK-1.5d）のスコープのため
//! 含めない。

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(tensor: &Tensor<f32>) -> f32 {
    tensor
        .get(&[])
        .expect("test fixture: スカラー shape [] のはず")
}

// --- 1. mul の自己参照（勾配蓄積の検証）: loss = sum(x * x) → dx = 2x ---

#[test]
fn mul_self_reference_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]));

    // 同一 `Var` を `mul` の両引数に渡す。x のノードへは
    // `Mul(x, x)` の VJP から dA・dB 両方の寄与が流入するため、
    // `backward()` の蓄積（合算）ロジックを直接検証する。
    let y = x.mul(&x).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");

    // d/dx sum(x*x) = 2x
    assert_eq!(dx.get(&[0, 0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[0, 1]).unwrap(), -4.0);
    assert_eq!(dx.get(&[1, 0]).unwrap(), 6.0);
    assert_eq!(dx.get(&[1, 1]).unwrap(), 1.0);
}

// --- 2. add: loss = sum(a + b) → da = db = ones ---

#[test]
fn add_grad_is_ones_for_both_operands() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let b = tape.var(&t(vec![0.5, 1.5, -1.0], &[3]));

    let y = a.add(&b).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&a).unwrap().expect("a は loss に到達する");
    let db = grads.get(&b).unwrap().expect("b は loss に到達する");

    for i in 0..3 {
        assert_eq!(da.get(&[i]).unwrap(), 1.0);
        assert_eq!(db.get(&[i]).unwrap(), 1.0);
    }
}

// --- 3. MLP 1 層相当の合成関数: 数値微分との end-to-end 突合 ---
//
// `loss = mse_loss(relu(x.matmul(w) + b), target)`。ReLU 入力は
// `|value| >= 10h`（`h = 1e-3`）の固定値でキンクを回避する（#17 と同方針）。

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
    for flat in 0..numel {
        let av = analytic.get(&index).unwrap_or(0.0);
        let nv = numeric.get(&index).unwrap_or(0.0);
        let diff = (av - nv).abs();
        let rel = diff / av.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{flat:?} idx={index:?}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
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

/// `Tape`/`Var` を新規構築して forward を再評価し、スカラー loss 値
/// （f32）を返す。中央差分の各サンプル点でテープを 1 回使い捨てる
/// （`tape.rs` が前提とする学習ループ運用と同じパターン）。
fn forward_loss(x: &Tensor<f32>, w: &Tensor<f32>, b: &Tensor<f32>, target: &Tensor<f32>) -> f32 {
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(x);
    let wv = tape.var(w);
    let bv = tape.var(b);
    let tv = tape.var(target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    scalar(&loss.to_tensor())
}

/// 指定テンソルの各要素を中央差分で摂動し、`forward_loss` に対する
/// 数値勾配を計算する。f64 で集計し丸め誤差を抑える
/// （`grad.rs` の `numeric_grad_unary` と同方針）。
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

struct MlpFixture {
    x: Tensor<f32>,
    w: Tensor<f32>,
    b: Tensor<f32>,
    target: Tensor<f32>,
}

fn mlp_fixture() -> MlpFixture {
    // pre-relu = x @ w + b の各要素の絶対値は 0.1 以上
    // （[-0.15, -1.3, 3.25, -0.1]）。h=1e-3 の摂動で符号が変わらず
    // ReLU のキンクを踏まない。
    MlpFixture {
        x: t(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]),
        w: t(vec![0.5, -1.0, 1.5, 0.2], &[2, 2]),
        b: t(vec![0.1, -0.2], &[2]),
        target: t(vec![0.1, 0.0, 3.0, 0.0], &[2, 2]),
    }
}

#[test]
fn mlp_grad_w_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dw = grads.get(&wv).unwrap().expect("w は loss に到達する");

    let num_dw = numeric_grad(&f.w, |w| forward_loss(&f.x, &w, &f.b, &f.target));
    assert_grad_close("mlp dW", dw, &num_dw);
}

#[test]
fn mlp_grad_b_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_db = numeric_grad(&f.b, |b| forward_loss(&f.x, &f.w, &b, &f.target));
    assert_grad_close("mlp dB", db, &num_db);
}

#[test]
fn mlp_grad_x_matches_numeric() {
    let f = mlp_fixture();
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&f.x);
    let wv = tape.var(&f.w);
    let bv = tape.var(&f.b);
    let tv = tape.var(&f.target);
    let y = xv.matmul(&wv).unwrap().add(&bv).unwrap().relu();
    let loss = y.mse_loss(&tv).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&f.x, |x| forward_loss(&x, &f.w, &f.b, &f.target));
    assert_grad_close("mlp dX", dx, &num_dx);
}

// --- 4. API 契約テスト ---

#[test]
fn unreachable_leaf_returns_ok_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    // loss に一切関与しない葉ノード。
    let unused = tape.var(&t(vec![9.0], &[1]));
    let loss = x.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    assert!(grads.get(&unused).unwrap().is_none());
    assert!(grads.get(&x).unwrap().is_some());
}

#[test]
fn backward_with_foreign_tape_var_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let loss_a = tape_a.var(&t(vec![1.0], &[1])).sum(None).unwrap();
    // `tape_b` から `backward` を呼びつつ、`tape_a` の loss を渡す。
    let x_b = tape_b.var(&t(vec![2.0], &[1]));
    let _ = x_b.sum(None).unwrap(); // tape_b にも何かノードを積んでおく

    let result = tape_b.backward(&loss_a);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn gradients_get_with_foreign_tape_var_returns_tape_mismatch() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let loss_a = x_a.sum(None).unwrap();
    let grads_a = tape_a.backward(&loss_a).unwrap();

    let x_b = tape_b.var(&t(vec![3.0], &[1]));
    let result = grads_a.get(&x_b);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

#[test]
fn get_for_var_added_after_backward_returns_ok_none() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let loss = x.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();

    // backward 完了後に同一テープへ新規ノードを追加する
    // （`grads.grads.len()` を超える `NodeId` になる）。
    let after = tape.var(&t(vec![9.0], &[1]));
    assert!(grads.get(&after).unwrap().is_none());
}

#[test]
fn non_scalar_loss_seed_is_implicit_sum_projection() {
    // 非スカラー loss（shape [2]）は「暗黙の総和射影」
    // （`sum(loss).backward()` 相当）として扱われ、シードは全要素 1。
    // ここでは loss = x（恒等）とし、seed = ones と直接一致することを
    // 確認する（各要素が独立にそのまま出力へ伝わるため）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, -1.0], &[2]));
    let grads = tape.backward(&x).unwrap();
    let dx = grads.get(&x).unwrap().expect("x 自身が loss である");

    assert_eq!(dx.get(&[0]).unwrap(), 1.0);
    assert_eq!(dx.get(&[1]).unwrap(), 1.0);
}

// --- view 系ノード（reshape / transpose）の backward（イシュー #1047・
// 親 #1043「カーネル融合・autodiff 実行モデルの強化」） ---

/// 14. `reshape` 単体の backward: loss = sum(reshape(x, [4]))。
///     sum は形状に依存しないため dx は全要素 1（x の元 shape [2,2]）。
#[test]
fn reshape_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let r = x.reshape(&[4]).unwrap();
    let loss = r.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(dx.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

/// 15. `transpose` 単体の backward: loss = sum(transpose(x, 0, 1))。
///     sum は形状・順序に依存しないため dx は全要素 1（x の元 shape）。
#[test]
fn transpose_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let tr = x.transpose(0, 1).unwrap();
    let loss = tr.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(dx.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

/// 16. `reshape → matmul`（view を fallible 演算〈matmul〉へ渡す経路）。
///     `w` を単位行列にして matmul を恒等化し、reshape 単体の VJP
///     （zero-copy `reshape` の逆写像）が正しく合成されることを
///     解析解と厳密一致で検証する。
#[test]
fn reshape_then_matmul_backward_matches_analytical() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // x: shape [4] → reshape → [2,2]
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let w = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])); // 単位行列
    let r = x.reshape(&[2, 2]).unwrap();
    let y = r.matmul(&w).unwrap(); // y == r（w が単位行列のため）
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    // d(sum(r @ I))/dr = ones[2,2] → reshape の逆写像で dx = ones[4]
    assert_eq!(dx.shape(), &[4]);
    for i in 0..4 {
        assert_eq!(dx.get(&[i]).unwrap(), 1.0);
    }
}

/// 17. `relu → transpose → add`（view が融合境界になる経路）。融合
///     （`push_lazy`）と view（`push_view`）が混在する連鎖で、非融合
///     参照実装と同じ勾配になることを解析解と突合する。
///
///     `y = relu(x)` [2,3] → `t = transpose(y, 0, 1)` [3,2] →
///     `z = t + bias`（bias: [2]、列方向 broadcast）→ `loss = sum(z)`。
///     `dz = ones[3,2]` → `dbias = reduce_to_shape(dz, [2])`（各列 3 要素
///     の総和 = 3） → `dt = ones[3,2]` → `dy = transpose(dt, 0, 1) =
///     ones[2,3]` → `dx = dy * (x > 0)`（ReLU 劣勾配）。
#[test]
fn relu_transpose_add_fusion_boundary_matches_reference() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![-1.0, 2.0, -3.0, 4.0, 5.0, -6.0], &[2, 3]));
    let bias = tape.var(&t(vec![10.0, 20.0], &[2]));

    let y = x.relu();
    let tr = y.transpose(0, 1).unwrap();
    let z = tr.add(&bias).unwrap();
    let loss = z.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    let dbias = grads.get(&bias).unwrap().expect("bias は loss に到達する");

    // ReLU 劣勾配: x > 0 の位置のみ 1、それ以外 0。
    let expected_dx = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    for (idx, &expected) in expected_dx.iter().enumerate() {
        let (i, j) = (idx / 3, idx % 3);
        assert_eq!(
            dx.get(&[i, j]).unwrap(),
            expected,
            "dx[{i},{j}] 不一致（ReLU 劣勾配）"
        );
    }
    // bias は [2] へ 3 要素ずつ縮約されるため、各要素は 3.0。
    assert_eq!(dbias.get(&[0]).unwrap(), 3.0);
    assert_eq!(dbias.get(&[1]).unwrap(), 3.0);
}

/// 18. view ノード自身の fan-out（同一 `reshape` 結果を `mul` の両
///     オペランドとして 2 回消費し、勾配が合算されることを検証する）。
///     `loss = sum(r * r)`（`r = reshape(x, [2,2])`）→ `dr = 2r` →
///     `dx = reshape(2r, [4]) = 2x`。
#[test]
fn view_node_fan_out_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0, 0.5], &[4]));
    let r = x.reshape(&[2, 2]).unwrap();
    let y = r.mul(&r).unwrap();
    let loss = y.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[4]);
    assert_eq!(dx.get(&[0]).unwrap(), 2.0);
    assert_eq!(dx.get(&[1]).unwrap(), -4.0);
    assert_eq!(dx.get(&[2]).unwrap(), 6.0);
    assert_eq!(dx.get(&[3]).unwrap(), 1.0);
}

// --- permute / broadcast_to / expand / squeeze / unsqueeze / flatten
// （イシュー #1597） ---

/// `Tensor<f32>` を行優先で読み出す（`broadcast_to` の stride 0 view は
/// `as_slice()` が `None` を返すため、`Var::to_tensor()` の値比較に
/// `get`／`contiguous()` 経由の本ヘルパーを使う。`tape_recording.rs`
/// の同名ヘルパーと同型）。
fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

/// 19. `permute` 単体の backward: loss = sum(permute(x, perm))。sum は
///     形状・順序に依存しないため dx は全要素 1（x の元 shape）。
#[test]
fn permute_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let p = x.permute(&[1, 0]).unwrap();
    let loss = p.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(dx.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

/// 20. `broadcast_to` 単体の backward: loss = sum(broadcast_to(x, s))。
///     dx は各出力要素が入力のどの要素に対応するかの複製回数（解析値）
///     になる（`x: [1,3] → [2,3]` は 2 回複製されるため dx = [2,2,2]）。
#[test]
fn broadcast_to_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let b = x.broadcast_to(&[2, 3]).unwrap();
    let loss = b.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[1, 3]);
    for j in 0..3 {
        assert_eq!(dx.get(&[0, j]).unwrap(), 2.0);
    }
}

/// 21. `squeeze → unsqueeze → flatten` の連鎖（すべて `reshape` への
///     委譲）の backward。sum の入力なので勾配は全要素 1（元 shape）。
#[test]
fn squeeze_unsqueeze_flatten_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]));

    let sq = x.squeeze(Some(1)).unwrap(); // [2,3]
    let u = sq.unsqueeze(0).unwrap(); // [1,2,3]
    let f = u.flatten(1, 2).unwrap(); // [1,6]
    let loss = f.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 1, 3]);
    for i in 0..2 {
        for k in 0..3 {
            assert_eq!(dx.get(&[i, 0, k]).unwrap(), 1.0);
        }
    }
}

/// 22. bit 同一 parity: `broadcast_to` を明示してから `add` した結果
///     （forward・`dx`／`dy`）が、`add` の暗黙ブロードキャストのみで
///     計算した結果と bit 同一であることを検証する（`Op::BroadcastTo`
///     の VJP と `Op::Add` の暗黙ブロードキャスト VJP が同じ
///     `reduce_bias_grad` を使うため。イシュー #1597 の parity 要件・
///     codex-review P1 是正で `reduce_to_shape` から切替済み）。
#[test]
fn broadcast_to_then_add_matches_implicit_broadcast_add_bit_exact() {
    let x_data = vec![1.0f32, -2.0, 3.0];
    let y_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];

    // 経路 A: broadcast_to を明示してから add。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(x_data.clone(), &[3]));
    let y_a = tape_a.var(&t(y_data.clone(), &[2, 3]));
    let bx_a = x_a.broadcast_to(&[2, 3]).unwrap();
    let z_a = bx_a.add(&y_a).unwrap();
    let loss_a = z_a.sum(None).unwrap();
    let forward_a = dense_vec(&z_a.to_tensor());
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dx_a = dense_vec(grads_a.get(&x_a).unwrap().unwrap());
    let dy_a = dense_vec(grads_a.get(&y_a).unwrap().unwrap());

    // 経路 B: add の暗黙ブロードキャストのみ。
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_b.var(&t(x_data, &[3]));
    let y_b = tape_b.var(&t(y_data, &[2, 3]));
    let z_b = x_b.add(&y_b).unwrap();
    let loss_b = z_b.sum(None).unwrap();
    let forward_b = dense_vec(&z_b.to_tensor());
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dx_b = dense_vec(grads_b.get(&x_b).unwrap().unwrap());
    let dy_b = dense_vec(grads_b.get(&y_b).unwrap().unwrap());

    assert_eq!(forward_a, forward_b, "forward 値が bit 同一でない");
    assert_eq!(dx_a, dx_b, "dx が bit 同一でない");
    assert_eq!(dy_a, dy_b, "dy が bit 同一でない");
}

/// 22b. `Op::BroadcastTo` の VJP が `Op::Add` の暗黙ブロードキャスト
///      縮約（`reduce_bias_grad`。行方向縮約パターンは `f64`
///      アキュムレータ〈`eval::reduce_bias_grad_rows`〉経由）と同じ
///      数値契約であることを、相殺を含む上流勾配（行順
///      `[1e8, 1, -1e8]`）で検証する（codex-review P1 是正の回帰:
///      旧実装は `reduce_to_shape`〈`f32` 逐次和〉のみを使い、
///      `1e8 + 1` が `f32` 丸めで `1e8` へ吸収されたあと `-1e8` すると
///      `0.0` になってしまっていたが、`f64` 経由では `1.0` が正しい
///      解析値）。`loss = sum(broadcast_to(x, [3,1]) * c)` は
///      `dx = sum(c)`（`c` は broadcast_to の上流勾配そのものに一致
///      させるための定数）となり、`Op::Add` の行方向縮約と同一の
///      shape 構造（`g: [3,1]`・`target: [1]`）を `Op::BroadcastTo`
///      単体の VJP 経路で踏む。
#[test]
fn broadcast_to_backward_row_reduction_uses_f64_accumulator_on_cancelling_values() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0], &[1]));
    let c = tape.var(&t(vec![1.0e8, 1.0, -1.0e8], &[3, 1]));
    let bx = x.broadcast_to(&[3, 1]).unwrap();
    let z = bx.mul(&c).unwrap();
    let loss = z.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[1]);
    assert_eq!(
        dx.get(&[0]).unwrap(),
        1.0,
        "f64 アキュムレータ経由の解析値（1e8 + 1 - 1e8 = 1.0）と一致しない \
         （f32 逐次和のままだと 1e8 + 1 が丸めで 1e8 に吸収され 0.0 になる）"
    );
}

/// 23. bit 同一 parity: `x.permute(&[1,0])?.matmul(&w)` と
///     `x.transpose(0,1)?.matmul(&w)` の forward・`dx` が bit 同一で
///     あることを検証する（2 軸 swap の `perm` は `transpose` と同一
///     strides を生成するため。イシュー #1597 の parity 要件）。
#[test]
fn permute_then_matmul_matches_transpose_then_matmul_bit_exact() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let w_data = vec![1.0f32, -1.0, 0.5, 2.0]; // [2,2]

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let x_a = tape_a.var(&t(x_data.clone(), &[2, 3]));
    let w_a = tape_a.var(&t(w_data.clone(), &[2, 2]));
    let p_a = x_a.permute(&[1, 0]).unwrap(); // [3,2]
    let y_a = p_a.matmul(&w_a).unwrap();
    let loss_a = y_a.sum(None).unwrap();
    let forward_a = dense_vec(&y_a.to_tensor());
    let grads_a = tape_a.backward(&loss_a).unwrap();
    let dx_a = dense_vec(grads_a.get(&x_a).unwrap().unwrap());

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_b.var(&t(x_data, &[2, 3]));
    let w_b = tape_b.var(&t(w_data, &[2, 2]));
    let tr_b = x_b.transpose(0, 1).unwrap(); // [3,2]
    let y_b = tr_b.matmul(&w_b).unwrap();
    let loss_b = y_b.sum(None).unwrap();
    let forward_b = dense_vec(&y_b.to_tensor());
    let grads_b = tape_b.backward(&loss_b).unwrap();
    let dx_b = dense_vec(grads_b.get(&x_b).unwrap().unwrap());

    assert_eq!(forward_a, forward_b, "forward 値が bit 同一でない");
    assert_eq!(dx_a, dx_b, "dx が bit 同一でない");
}

/// 24. 異常系: `permute`（perm 長不一致・範囲外・重複軸）・
///     `broadcast_to`（縮小方向・非互換 shape）・`unsqueeze`（rank+1
///     超過）・`flatten`（`start_dim > end_dim`）が
///     `AutodiffError::Shape(..)` を返すことを検証する（`var.rs` 側の
///     検査順序の end-to-end 確認）。
#[test]
fn shape_op_error_paths_return_shape_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    assert!(matches!(
        x.permute(&[0]).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch { .. })
    ));
    assert!(matches!(
        x.permute(&[0, 0]).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::DuplicateAxis { .. })
    ));
    assert!(matches!(
        x.broadcast_to(&[3]).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::BroadcastIncompatible { .. })
    ));
    assert!(matches!(
        x.unsqueeze(3).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
    assert!(matches!(
        x.flatten(1, 0).unwrap_err(),
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

// --- cat / stack / narrow / split / chunk（イシュー #1598） ---

/// 24. `cat` の backward: `loss = sum(cat([x, y], dim=1) * c)` は
///     dim=1 で連結された各入力へ `c` の対応区間がそのまま流れる
///     （解析値）。
#[test]
fn cat_backward_distributes_upstream_to_each_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![5.0, 6.0], &[2, 1]));
    let c = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let cat = fandhe_ai_autodiff::Var::cat(&[x, y], 1).unwrap();
    assert_eq!(cat.to_tensor().shape(), &[2, 3]);
    let z = cat.mul(&c).unwrap();
    let loss = z.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    let dy = grads.get(&y).unwrap().expect("y は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 2.0, 4.0, 5.0]);
    assert_eq!(dense_vec(dy), vec![3.0, 6.0]);
}

/// 25. `cat(&[x, x])` の fan-out 合算: 同一 `Var` を 2 回連結した場合、
///     backward が両方の寄与を合算することを確認する（`Op::Concat`
///     doc「同一 NodeId の重複は `accumulate` が合算する」）。
#[test]
fn cat_self_reference_accumulates_gradient() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
    let cat = fandhe_ai_autodiff::Var::cat(&[x, x], 0).unwrap();
    let loss = cat.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![2.0, 2.0]);
}

/// 26. `stack` の backward: `loss = sum(stack([x, y], dim=0))` は
///     sum が形状・順序に依存しないため dx/dy は全要素 1。
#[test]
fn stack_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![3.0, 4.0], &[2]));
    let s = fandhe_ai_autodiff::Var::stack(&[x, y], 0).unwrap();
    assert_eq!(s.to_tensor().shape(), &[2, 2]);
    let loss = s.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    let dy = grads.get(&y).unwrap().expect("y は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 1.0]);
    assert_eq!(dense_vec(dy), vec![1.0, 1.0]);
}

/// 27. `split`／`chunk` の backward: 各出力を異なる係数で重み付けした
///     loss の解析勾配を検証する（各出力区間へ対応係数がそのまま
///     流れる）。
#[test]
fn split_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let parts = x.split(2, 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_tensor().shape(), &[2]);
    assert_eq!(parts[2].to_tensor().shape(), &[1]);

    // 各パートに係数 1, 2, 3 を掛けてから合算する。
    let coeffs = [1.0f32, 2.0, 3.0];
    let mut terms = Vec::with_capacity(parts.len());
    for (part, &coeff) in parts.iter().zip(coeffs.iter()) {
        let scaled = part.sum(None).unwrap();
        let c = tape.var(&t(vec![coeff], &[]));
        terms.push(scaled.mul(&c).unwrap());
    }
    let mut loss = terms[0];
    for term in &terms[1..] {
        loss = loss.add(term).unwrap();
    }

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    // part0=[1,2]*1, part1=[3,4]*2, part2=[5]*3
    assert_eq!(dense_vec(dx), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
}

/// 28. `chunk` の backward（`split` と同型の検証）。
#[test]
fn chunk_backward_matches_expected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let parts = x.chunk(3, 0).unwrap();
    // ceil(5/3) = 2 -> [2, 2, 1]
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].to_tensor().shape(), &[2]);
    assert_eq!(parts[1].to_tensor().shape(), &[2]);
    assert_eq!(parts[2].to_tensor().shape(), &[1]);

    let loss = parts[0]
        .sum(None)
        .unwrap()
        .add(&parts[1].sum(None).unwrap())
        .unwrap()
        .add(&parts[2].sum(None).unwrap())
        .unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 1.0, 1.0, 1.0, 1.0]);
}

/// 29. `narrow` 単体の backward: 未選択領域の勾配は 0。
#[test]
fn narrow_backward_zeros_unselected_region() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]));
    let n = x.narrow(0, 1, 2).unwrap();
    assert_eq!(n.to_tensor().shape(), &[2]);
    let loss = n.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![0.0, 1.0, 1.0, 0.0, 0.0]);
}

/// 30. `split` → `cat` の往復: forward が bit 同一・backward が全要素
///     1（sum の入力）であることを検証する。
#[test]
fn split_then_cat_roundtrip_matches_original_bit_exact_and_backward_is_ones() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]));
    let parts = x.split(2, 0).unwrap();
    let rejoined = fandhe_ai_autodiff::Var::cat(&parts, 0).unwrap();
    assert_eq!(dense_vec(&rejoined.to_tensor()), dense_vec(&x.to_tensor()));

    let loss = rejoined.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0; 6]);
}

/// 32. `cat` 経由で loss に到達したパラメータの勾配（`dim=1` の
///     連結なので非 contiguous な narrow view）を `optim::Sgd::step`
///     に渡して 1 step 更新できることを確認する（view 勾配の消費側
///     契約の回帰）。
#[test]
fn cat_gradient_view_can_be_consumed_by_sgd_step() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let w = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![5.0, 6.0], &[2, 1]));
    let cat = fandhe_ai_autodiff::Var::cat(&[w, y], 1).unwrap();
    let loss = cat.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let dw = grads
        .get(&w)
        .unwrap()
        .expect("w は loss に到達する")
        .clone();

    let mut sgd =
        fandhe_ai_autodiff::optim::Sgd::new(fandhe_ai_autodiff::optim::SgdConfig::new(0.1))
            .unwrap();
    let w_val = w.to_tensor();
    let updated = sgd
        .step(&[&w_val], &[&dw])
        .expect("view 勾配でも step が成功する");
    assert_eq!(updated[0].shape(), &[2, 2]);
}

/// 33. edge ケース: `chunk` の `shape[dim] == 0`（`chunks` 個の空
///     narrow）・`split` の `shape[dim] == 0`（1 個の空 narrow）・
///     全区間 0 長の `cat`。
#[test]
fn zero_length_edge_cases_for_split_chunk_and_cat() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let empty = tape.var(&t(Vec::new(), &[0]));

    let chunks = empty.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    for c in &chunks {
        assert_eq!(c.to_tensor().shape(), &[0]);
    }

    let splits = empty.split(4, 0).unwrap();
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].to_tensor().shape(), &[0]);

    let cat = fandhe_ai_autodiff::Var::cat(&[empty, empty], 0).unwrap();
    assert_eq!(cat.to_tensor().shape(), &[0]);
}

// --- cat / stack / narrow / split / chunk のエラー経路 ---

#[test]
fn cat_empty_list_is_invalid_argument() {
    let err = fandhe_ai_autodiff::Var::cat(&[], 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn stack_empty_list_is_invalid_argument() {
    let err = fandhe_ai_autodiff::Var::stack(&[], 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn cat_cross_tape_is_rejected() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x = tape_a.var(&t(vec![1.0], &[1]));
    let y = tape_b.var(&t(vec![2.0], &[1]));
    let err = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

#[test]
fn cat_rank_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let err = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch { .. })
    ));
}

#[test]
fn cat_axis_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0], &[3, 1]));
    let err = fandhe_ai_autodiff::Var::cat(&[x, y], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn stack_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err = fandhe_ai_autodiff::Var::stack(&[x], 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 2,
            rank: 2
        })
    ));
}

#[test]
fn stack_shape_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = fandhe_ai_autodiff::Var::stack(&[x, y], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}

#[test]
fn stack_non_contiguous_input_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let tr = x.transpose(0, 1).unwrap();
    let err = fandhe_ai_autodiff::Var::stack(&[tr], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NonContiguousReshape)
    ));
}

#[test]
fn narrow_out_of_bounds_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.narrow(0, 2, 2).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NarrowOutOfBounds { .. })
    ));
}

#[test]
fn split_zero_size_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.split(0, 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn chunk_zero_chunks_is_invalid_argument() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.chunk(0, 0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn split_with_sizes_sum_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.split_with_sizes(&[1, 1], 0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}
// --- Where／MaskedFill（イシュー #1637） ---

fn tb(data: Vec<bool>, shape: &[usize]) -> Tensor<bool> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// ①forward 値（解析）: `where_cond` が `cond` の真偽で `a`／`b` の
/// 要素を選択することを直接確認する。
#[test]
fn where_forward_selects_by_condition() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let b = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
    let cond = tb(vec![true, false, true, false], &[4]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 20.0, 3.0, 40.0]);
}

/// ②`where_cond` の backward を中央差分と突合する（`cond` は摂動対象
/// 外の定数マスク）。
#[test]
fn where_backward_matches_numeric() {
    let cond = tb(vec![true, false, true, false], &[4]);
    let a0 = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let b0 = t(vec![10.0, 20.0, 30.0, 40.0], &[4]);

    let forward = |a: &Tensor<f32>, b: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(a);
        let bv = tape.var(b);
        let out = fandhe_ai_autodiff::Var::where_cond(&cond, &av, &bv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a0);
    let bv = tape.var(&b0);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &av, &bv).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a は loss に到達する");
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");

    let num_da = numeric_grad(&a0, |a| forward(&a, &b0));
    let num_db = numeric_grad(&b0, |b| forward(&a0, &b));
    assert_grad_close("where dA", da, &num_da);
    assert_grad_close("where dB", db, &num_db);
}

/// ③broadcast（`x:[2,3]`, `y:[3]`, `cond:[2,3]`）で `dy` が行方向へ
/// 縮約されることを確認する。
#[test]
fn where_backward_broadcast_reduces_dy_to_input_shape() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let y = tape.var(&t(vec![10.0, 20.0, 30.0], &[3]));
    let cond = tb(vec![true, false, true, false, true, false], &[2, 3]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &x, &y).unwrap();
    assert_eq!(out.to_tensor().shape(), &[2, 3]);
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dy = grads.get(&y).unwrap().expect("y は loss に到達する");
    assert_eq!(dy.shape(), &[3]);
    // cond=[[T,F,T],[F,T,F]] → y が選ばれる位置は (0,1)・(1,0)・(1,2)。
    // 各列で合算: col0 = row1(1) = 1.0・col1 = row0(1) = 1.0・
    // col2 = row1(1) = 1.0（upstream は sum の勾配で全要素 1）。
    assert_eq!(dense_vec(dy), vec![1.0, 1.0, 1.0]);
}

/// ③b `cond` 単独が軸を拡張するケース（`a`／`b:[3]`・`cond:[2,1]` →
/// 出力 `[2, 3]`）。出力 shape を `a`／`b` だけから決めていた旧実装
/// では `cond` を `[3]` へ broadcast しようとして `Shape` エラーに
/// なっていた（codex-review 指摘・PR #1684）。forward・backward
/// （数値微分突合）の両方を検証する。
#[test]
fn where_cond_alone_expands_output_shape() {
    let cond = tb(vec![true, false], &[2, 1]);
    let a0 = t(vec![1.0, 2.0, 3.0], &[3]);
    let b0 = t(vec![10.0, 20.0, 30.0], &[3]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let av = tape.var(&a0);
    let bv = tape.var(&b0);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &av, &bv).unwrap();
    assert_eq!(out.to_tensor().shape(), &[2, 3]);
    // cond broadcast: row0=true（a 選択）・row1=false（b 選択）。
    assert_eq!(
        dense_vec(&out.to_tensor()),
        vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    );

    let forward = |a: &Tensor<f32>, b: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let av = tape.var(a);
        let bv = tape.var(b);
        let out = fandhe_ai_autodiff::Var::where_cond(&cond, &av, &bv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let da = grads.get(&av).unwrap().expect("a は loss に到達する");
    let db = grads.get(&bv).unwrap().expect("b は loss に到達する");
    assert_eq!(da.shape(), &[3]);
    assert_eq!(db.shape(), &[3]);

    let num_da = numeric_grad(&a0, |a| forward(&a, &b0));
    let num_db = numeric_grad(&b0, |b| forward(&a0, &b));
    assert_grad_close("where cond-only-expand dA", da, &num_da);
    assert_grad_close("where cond-only-expand dB", db, &num_db);
}

/// ④同一 `Var` を `a`／`b` 両方に指定した場合（`where(c, x, x)`）、
/// `accumulate` が合算し `dx = g`（全要素 upstream をそのまま通す）
/// ことを確認する。
#[test]
fn where_backward_same_var_both_sides_accumulates_to_upstream() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let cond = tb(vec![true, false, true, false], &[4]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &x, &x).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), dense_vec(&x.to_tensor()));
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0; 4]);
}

/// ⑤NaN が非選択側に留まることを確認する（forward 出力に NaN が
/// 現れない）。
#[test]
fn where_forward_isolates_nan_to_unselected_side() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, f32::NAN], &[2]));
    let b = tape.var(&t(vec![f32::NAN, 20.0], &[2]));
    let cond = tb(vec![true, false], &[2]);
    let out = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap();
    let v = dense_vec(&out.to_tensor());
    assert!(v[0].is_finite() && v[0] == 1.0);
    assert!(v[1].is_finite() && v[1] == 20.0);
}

/// ⑥エラー経路: `cond` が `a`／`b` の broadcast 後 shape へ
/// broadcast 不能なら `AutodiffError::Shape` を返す。
#[test]
fn where_cond_non_broadcastable_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let cond = tb(vec![true, false, true], &[3]);
    let err = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

/// ⑥エラー経路: 異なるテープの `Var` を `where_cond` に渡すと
/// `AutodiffError::TapeMismatch` を返す。
#[test]
fn where_cond_cross_tape_is_rejected() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let a = tape_a.var(&t(vec![1.0, 2.0], &[2]));
    let b = tape_b.var(&t(vec![3.0, 4.0], &[2]));
    let cond = tb(vec![true, false], &[2]);
    let err = fandhe_ai_autodiff::Var::where_cond(&cond, &a, &b).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

/// `masked_fill` の forward 値を確認する。
#[test]
fn masked_fill_forward_replaces_masked_positions() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
    let mask = tb(vec![true, false, true, false], &[4]);
    let out = x.masked_fill(&mask, -1.0).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![-1.0, 2.0, -1.0, 4.0]);
}

/// `masked_fill` の backward を中央差分と突合する（fill 位置の勾配は
/// 0）。
#[test]
fn masked_fill_backward_matches_numeric() {
    let mask = tb(vec![true, false, true, false], &[4]);
    let x0 = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let out = xv.masked_fill(&mask, -9.0).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let out = xv.masked_fill(&mask, -9.0).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("masked_fill dX", dx, &num_dx);
    // fill 位置の勾配は厳密に 0。
    assert_eq!(dense_vec(dx)[0], 0.0);
    assert_eq!(dense_vec(dx)[2], 0.0);
}

/// `masked_fill` のエラー経路: `mask` が `self` の shape へ
/// broadcast 不能なら `AutodiffError::Shape` を返す。
#[test]
fn masked_fill_non_broadcastable_mask_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let mask = tb(vec![true, false, true], &[3]);
    let err = x.masked_fill(&mask, 0.0).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

// --- Gather／Scatter（イシュー #1776） ---

fn i32t(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// ①forward 値（解析）: `gather` が `dim` 軸に沿って `index` の
/// 添字位置を独立に読み出すことを確認する。
#[test]
fn gather_forward_selects_along_dim() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let index = i32t(vec![0, 2, 1, 0], &[2, 2]);
    let out = x.gather(1, &index).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 3.0, 5.0, 4.0]);
}

/// ②`gather` の backward を中央差分と突合する（重複添字を含み、
/// `d_input` が読み出し回数分だけ加算蓄積されることを検証する）。
#[test]
fn gather_backward_matches_numeric_with_duplicate_indices() {
    // 全読み出しが x[0] を 2 回・x[1] を 1 回参照する。
    let index = i32t(vec![0, 0, 1], &[1, 3]);
    let x0 = t(vec![1.0, 2.0], &[1, 2]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let out = xv.gather(1, &index).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let out = xv.gather(1, &index).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("gather dX (duplicate index)", dx, &num_dx);
    // 解析的検証: x[0] は 2 回・x[1] は 1 回読まれるため dx=[2,1]。
    assert_eq!(dense_vec(dx), vec![2.0, 1.0]);
}

/// ③エラー経路: `dim` が rank 範囲外なら `AutodiffError::Shape`
/// （`AxisOutOfRange`）を返す。
#[test]
fn gather_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let index = i32t(vec![0], &[1]);
    let err = x.gather(1, &index).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 1,
            rank: 1
        })
    ));
}

/// ④エラー経路: `index` の値が `[0, shape[dim])` を外れると
/// `AutodiffError::InvalidArgument` を返す。
#[test]
fn gather_index_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let index = i32t(vec![5], &[1]);
    let err = x.gather(0, &index).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

/// ⑤エラー経路: `index` の負値も範囲外として拒否する
/// （PyTorch のような負値ラップアラウンドは対象外）。
#[test]
fn gather_negative_index_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let index = i32t(vec![-1], &[1]);
    let err = x.gather(0, &index).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

/// `index_select` が `gather` と同一の 1 ノードのみを記録し、1 次元
/// `index` を `dim` 軸へ拡張した結果が `gather` と一致することを
/// 確認する。
#[test]
fn index_select_matches_gather_single_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let index = i32t(vec![2, 0], &[2]);

    let before = tape.len();
    let out = x.index_select(1, &index).unwrap();
    assert_eq!(
        tape.len(),
        before + 1,
        "index_select は gather と同じ 1 ノードのみ追加するはず"
    );
    assert_eq!(out.to_tensor().shape(), &[2, 2]);
    assert_eq!(dense_vec(&out.to_tensor()), vec![3.0, 1.0, 6.0, 4.0]);
}

/// `index_select` のエラー経路: `index` が rank 1 でなければ
/// `AutodiffError::Shape(RankMismatch)` を返す。
#[test]
fn index_select_rank_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let index = i32t(vec![0, 1], &[1, 2]);
    let err = x.index_select(0, &index).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch {
            expected: 1,
            actual: 2
        })
    ));
}

/// `index_select` は strided（非 contiguous）な 1 次元 `index` も
/// 受け付ける必要がある。`Var::gather` は `index.contiguous()` を
/// 経由するが、`index_select` は内部で `index.reshape(..)` を直接
/// 呼んでいたため、非 contiguous な `index`（例: `broadcast_to` で
/// stride 0 に拡張した rank-1 テンソル）を渡すと
/// `ShapeError::NonContiguousReshape` になり `gather` との間で非対称
/// だった（Bugbot 指摘。イシュー #1776）。
#[test]
fn index_select_accepts_non_contiguous_index() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![10.0, 20.0, 30.0, 40.0], &[4]));
    // shape [1] を stride 0 で shape [3] へ拡張した非 contiguous な
    // 1 次元 index（全要素が同じ添字 2 を指す）。
    let index_base = i32t(vec![2], &[1]);
    let index = index_base.broadcast_to(&[3]).unwrap();
    assert!(!index.is_contiguous());

    let out = x.index_select(0, &index).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![30.0, 30.0, 30.0]);
}

/// `scatter`（`Overwrite`）の forward 値を確認する。
#[test]
fn scatter_overwrite_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 6], &[2, 3]));
    let index = i32t(vec![0, 2, 2, 1], &[2, 2]);
    let src = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let out = x.scatter(1, &index, &src).unwrap();
    assert_eq!(
        dense_vec(&out.to_tensor()),
        vec![1.0, 0.0, 2.0, 0.0, 4.0, 3.0]
    );
}

/// `scatter`（`Overwrite`）の backward を中央差分と突合する
/// （`d_input`／`d_src` の両方）。
#[test]
fn scatter_overwrite_backward_matches_numeric() {
    let index = i32t(vec![0, 2, 2, 1], &[2, 2]);
    let x0 = t(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let src0 = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let forward = |x: &Tensor<f32>, src: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let sv = tape.var(src);
        let out = xv.scatter(1, &index, &sv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let sv = tape.var(&src0);
    let out = xv.scatter(1, &index, &sv).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
    let dsrc = grads.get(&sv).unwrap().expect("src は loss に到達する");

    let num_dx = numeric_grad(&x0, |x| forward(&x, &src0));
    let num_dsrc = numeric_grad(&src0, |s| forward(&x0, &s));
    assert_grad_close("scatter(overwrite) dX", dx, &num_dx);
    assert_grad_close("scatter(overwrite) dSrc", dsrc, &num_dsrc);
}

/// `scatter`（`Overwrite`）で同一出力位置へ複数回書き込まれる場合、
/// forward の決定的集約契約（行優先走査順で「最後に処理された値」
/// のみが残る）どおり、上書きされて消えた重複書き込みへの `d_src`
/// は 0 になる必要がある（codex-review 指摘。イシュー #1776）。
/// `input=[0]`／`index=[0,0]`／`src=[2,3]`（dim=0）は
/// `out = scatter(...) = [3]`（`src[1]=3` が最後の書き手）となり、
/// `d_src` は `[0, 1]`（`src[0]` は 0・`src[1]` は 1）になるはず。
#[test]
fn scatter_overwrite_backward_last_writer_wins_duplicate_indices() {
    let index = i32t(vec![0, 0], &[2]);
    let x0 = t(vec![0.0], &[1]);
    let src0 = t(vec![2.0, 3.0], &[2]);

    let forward = |x: &Tensor<f32>, src: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let sv = tape.var(src);
        let out = xv.scatter(0, &index, &sv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let sv = tape.var(&src0);
    let out = xv.scatter(0, &index, &sv).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![3.0]);
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dsrc = grads.get(&sv).unwrap().expect("src は loss に到達する");
    assert_eq!(
        dense_vec(dsrc),
        vec![0.0, 1.0],
        "上書きされて消えた src[0] への勾配は 0・実際に出力へ残った src[1] は 1 のはず"
    );

    let num_dsrc = numeric_grad(&src0, |s| forward(&x0, &s));
    assert_grad_close("scatter(overwrite) duplicate dSrc", dsrc, &num_dsrc);
}

/// `scatter_out_shape` は `dim` 以外の軸で `index`／`src` が `input`
/// より小さいことを許容する（`index_shape[axis] <= input_shape[axis]`）
/// が、`d_src` を求める `gather` は `dim` 以外の軸の完全一致を要求
/// するため、`upstream` を `src_shape` へ narrow しないと有効な
/// forward 入力に対し backward が `ShapeMismatch` になっていた
/// （codex-review 指摘。イシュー #1776）。`input` shape `[3, 4]`・
/// `index`／`src` shape `[2, 2]`（dim=1。行 0・1 のみ、列 0 の
/// 位置のみ書き込む）で backward が成功し中央差分と一致することを
/// 確認する。
#[test]
fn scatter_overwrite_backward_narrows_upstream_for_shrunk_non_dim_axes() {
    let index = i32t(vec![1, 3, 0, 2], &[2, 2]);
    let x0 = t(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        &[3, 4],
    );
    let src0 = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let forward = |x: &Tensor<f32>, src: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let sv = tape.var(src);
        let out = xv.scatter(1, &index, &sv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let sv = tape.var(&src0);
    let out = xv.scatter(1, &index, &sv).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
    let dsrc = grads.get(&sv).unwrap().expect("src は loss に到達する");

    let num_dx = numeric_grad(&x0, |x| forward(&x, &src0));
    let num_dsrc = numeric_grad(&src0, |s| forward(&x0, &s));
    assert_grad_close("scatter(overwrite, shrunk non-dim axis) dX", dx, &num_dx);
    assert_grad_close(
        "scatter(overwrite, shrunk non-dim axis) dSrc",
        dsrc,
        &num_dsrc,
    );
}

/// `scatter_add` の forward 値（重複添字の加算）を確認する。
#[test]
fn scatter_add_forward_accumulates_duplicates() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![10.0, 0.0, 0.0], &[1, 3]));
    let index = i32t(vec![0, 0, 0], &[1, 3]);
    let src = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let out = x.scatter_add(1, &index, &src).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![16.0, 0.0, 0.0]);
}

/// `scatter_add` の backward を中央差分と突合する（`d_input` は
/// 恒等・`d_src` は各要素が自身の書き込み先の upstream をそのまま
/// 受け取ることを解析的にも確認する）。
#[test]
fn scatter_add_backward_matches_numeric_with_duplicate_indices() {
    let index = i32t(vec![0, 0, 1], &[1, 3]);
    let x0 = t(vec![0.1, 0.2], &[1, 2]);
    let src0 = t(vec![1.0, 2.0, 3.0], &[1, 3]);

    let forward = |x: &Tensor<f32>, src: &Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let sv = tape.var(src);
        let out = xv.scatter_add(1, &index, &sv).unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let sv = tape.var(&src0);
    let out = xv.scatter_add(1, &index, &sv).unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
    let dsrc = grads.get(&sv).unwrap().expect("src は loss に到達する");

    let num_dx = numeric_grad(&x0, |x| forward(&x, &src0));
    let num_dsrc = numeric_grad(&src0, |s| forward(&x0, &s));
    assert_grad_close("scatter_add dX", dx, &num_dx);
    assert_grad_close("scatter_add dSrc", dsrc, &num_dsrc);
    // Add: d_input は恒等（線形性）のため upstream（全要素 1）のまま。
    assert_eq!(dense_vec(dx), vec![1.0, 1.0]);
    // d_src はどの要素も自身の書き込み先の upstream(=1) をそのまま
    // 受け取るため全て 1（重複があっても影響しない）。
    assert_eq!(dense_vec(dsrc), vec![1.0, 1.0, 1.0]);
}

/// エラー経路: 異なる `Tape` の `Var` を `scatter` の `src` に渡すと
/// `AutodiffError::TapeMismatch` を返す。
#[test]
fn scatter_cross_tape_is_rejected() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let x = tape_a.var(&t(vec![1.0, 2.0], &[1, 2]));
    let src = tape_b.var(&t(vec![3.0, 4.0], &[1, 2]));
    let index = i32t(vec![0, 1], &[1, 2]);
    let err = x.scatter(1, &index, &src).unwrap_err();
    assert!(matches!(err, AutodiffError::TapeMismatch));
}

/// エラー経路: `index` と `src` の shape が一致しなければ
/// `AutodiffError::Shape(ShapeMismatch)` を返す。
#[test]
fn scatter_index_src_shape_mismatch_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let index = i32t(vec![0, 1], &[1, 2]);
    let src = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let err = x.scatter(1, &index, &src).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}

/// エラー経路: `scatter` の `index` 値が範囲外なら
/// `AutodiffError::InvalidArgument` を返す。
#[test]
fn scatter_index_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let index = i32t(vec![5], &[1, 1]);
    let src = tape.var(&t(vec![9.0], &[1, 1]));
    let err = x.scatter(1, &index, &src).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- Sort／Argsort／Topk（イシュー #1733） ---

/// `i32` 版 `dense_vec`（上記）。`sort`／`argsort`／`topk` の `index`
/// 出力を検証するテスト専用ヘルパー。
fn dense_vec_i32(tensor: &Tensor<i32>) -> Vec<i32> {
    tensor
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous() 後は必ず as_slice() が Some")
        .to_vec()
}

/// ①forward 値（解析）: 昇順 sort。同値なし・NaN なしの基本ケース。
#[test]
fn sort_ascending_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out, index) = x.sort(1, false).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 1.5, 3.0, 4.0]);
    assert_eq!(dense_vec_i32(&index), vec![1, 3, 0, 2]);
}

/// ①forward 値（解析）: 降順 sort。
#[test]
fn sort_descending_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out, index) = x.sort(1, true).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![4.0, 3.0, 1.5, 1.0]);
    assert_eq!(dense_vec_i32(&index), vec![2, 0, 3, 1]);
}

/// 同値（ties）の順序契約: `descending` の値に関わらず、同値要素は
/// 元インデックス昇順で並ぶ（`BackendOps::sort` doc 契約 1）。
#[test]
fn sort_ties_preserve_index_ascending_order_both_directions() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![2.0, 1.0, 1.0, 2.0], &[1, 4]));

    let (asc, asc_index) = x.sort(1, false).unwrap();
    assert_eq!(dense_vec(&asc.to_tensor()), vec![1.0, 1.0, 2.0, 2.0]);
    // 値 1.0 の元添字は {1, 2}・値 2.0 の元添字は {0, 3}。
    // いずれも昇順ソート結果内で元添字昇順のまま。
    assert_eq!(dense_vec_i32(&asc_index), vec![1, 2, 0, 3]);

    let (desc, desc_index) = x.sort(1, true).unwrap();
    assert_eq!(dense_vec(&desc.to_tensor()), vec![2.0, 2.0, 1.0, 1.0]);
    // 降順でも同値グループ内は元添字昇順のまま（単純な reverse() なら
    // {3, 0} になってしまうところを {0, 3} で確認する）。
    assert_eq!(dense_vec_i32(&desc_index), vec![0, 3, 1, 2]);
}

/// NaN の順序契約: NaN は任意の非 NaN より大きい・NaN 同士は同値
/// （元添字昇順）として扱う（`BackendOps::sort` doc 契約 2）。
#[test]
fn sort_nan_treated_as_largest_and_ties_among_nan() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![f32::NAN, 1.0, f32::NAN, 0.0], &[1, 4]));
    let (asc, asc_index) = x.sort(1, false).unwrap();
    let asc_vals = dense_vec(&asc.to_tensor());
    assert_eq!(&asc_vals[..2], &[0.0, 1.0]);
    assert!(asc_vals[2].is_nan() && asc_vals[3].is_nan());
    // NaN 2 個（元添字 0・2）は同値扱いで元添字昇順のまま末尾に来る。
    assert_eq!(dense_vec_i32(&asc_index), vec![3, 1, 0, 2]);
}

/// ±0 の順序契約: `-0.0` と `0.0` は同値（`partial_cmp` が `Equal`）
/// として扱われ、元添字昇順で並ぶ（`BackendOps::sort` doc 契約 3）。
#[test]
fn sort_negative_and_positive_zero_are_tied() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -0.0, 0.0], &[1, 3]));
    let (out, index) = x.sort(1, false).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![-0.0, 0.0, 1.0]);
    assert_eq!(dense_vec_i32(&index), vec![1, 2, 0]);
}

/// ②`sort` の backward を中央差分と突合する（tie-free 入力・dim=1・
/// 降順）。`Op::Sort` の VJP は `values = gather(input, dim, index)`
/// と数学的に同一のため gather と同じ scatter_add 式を使う。
#[test]
fn sort_backward_matches_numeric_tie_free_descending() {
    let x0 = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let (out, _index) = xv.sort(1, true).unwrap();
        scalar(&out.mul(&out).unwrap().sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let (out, _index) = xv.sort(1, true).unwrap();
    let loss = out.mul(&out).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("sort dX (tie-free, descending)", dx, &num_dx);
}

/// argsort は `sort` と同じ `index` を返し、テープへノードを追加
/// しない（非微分演算。実装計画「設計判断」§2.1 参照）。
#[test]
fn argsort_matches_sort_index_and_adds_no_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));

    let before = tape.len();
    let index = x.argsort(1, false).unwrap();
    assert_eq!(
        tape.len(),
        before,
        "argsort はテープへノードを追加しないはず"
    );
    assert_eq!(dense_vec_i32(&index), vec![1, 3, 0, 2]);
}

/// ③エラー経路: `dim` が rank 範囲外なら `AutodiffError::Shape`
/// （`AxisOutOfRange`）を返す（`sort`／`argsort` 共通）。
#[test]
fn sort_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err = x.sort(1, false).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 1,
            rank: 1
        })
    ));
    let err = x.argsort(1, false).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 1,
            rank: 1
        })
    ));
}

/// `dim` 軸長 0 の sort は空出力で成功する。
#[test]
fn sort_empty_dim_returns_empty_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[1, 0]));
    let (out, index) = x.sort(1, false).unwrap();
    assert_eq!(out.to_tensor().shape(), &[1, 0]);
    assert_eq!(dense_vec(&out.to_tensor()), Vec::<f32>::new());
    assert_eq!(dense_vec_i32(&index), Vec::<i32>::new());
}

/// ①forward 値（解析）: `topk`（`largest=true`）は降順 sort の
/// 先頭 `k` に等しい。
#[test]
fn topk_largest_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out, index) = x.topk(2, 1, true).unwrap();
    assert_eq!(out.to_tensor().shape(), &[1, 2]);
    assert_eq!(dense_vec(&out.to_tensor()), vec![4.0, 3.0]);
    assert_eq!(dense_vec_i32(&index), vec![2, 0]);
}

/// ①forward 値（解析）: `topk`（`largest=false`）は昇順 sort の
/// 先頭 `k` に等しい。
#[test]
fn topk_smallest_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out, index) = x.topk(2, 1, false).unwrap();
    assert_eq!(out.to_tensor().shape(), &[1, 2]);
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 1.5]);
    assert_eq!(dense_vec_i32(&index), vec![1, 3]);
}

/// `k == n`（対象軸のサイズと等しい）は sort と同一の結果になる。
#[test]
fn topk_k_equals_dim_size_matches_sort() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (topk_out, topk_index) = x.topk(4, 1, true).unwrap();
    let (sort_out, sort_index) = x.sort(1, true).unwrap();
    assert_eq!(
        dense_vec(&topk_out.to_tensor()),
        dense_vec(&sort_out.to_tensor())
    );
    assert_eq!(dense_vec_i32(&topk_index), dense_vec_i32(&sort_index));
}

/// `k == 0` は空出力で成功する。
#[test]
fn topk_k_zero_returns_empty_output() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]));
    let (out, index) = x.topk(0, 1, true).unwrap();
    assert_eq!(out.to_tensor().shape(), &[1, 0]);
    assert_eq!(dense_vec(&out.to_tensor()), Vec::<f32>::new());
    assert_eq!(dense_vec_i32(&index), Vec::<i32>::new());
}

/// ②`topk` の backward を中央差分と突合する（tie-free 入力・
/// `largest=true`・`k < n`）。非選択要素の勾配は 0 になる。
#[test]
fn topk_backward_matches_numeric_and_zeroes_unselected() {
    let x0 = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let (out, _index) = xv.topk(2, 1, true).unwrap();
        scalar(&out.mul(&out).unwrap().sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let (out, _index) = xv.topk(2, 1, true).unwrap();
    let loss = out.mul(&out).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("topk dX (tie-free, largest, k<n)", dx, &num_dx);
    // 選択されたのは値 3.0（index 0）・4.0（index 2）のみ。
    // 非選択（index 1・3）の勾配は 0。
    let dx_vec = dense_vec(dx);
    assert_eq!(dx_vec[1], 0.0);
    assert_eq!(dx_vec[3], 0.0);
}

/// ③エラー経路: `dim` が rank 範囲外なら `AutodiffError::Shape`
/// （`AxisOutOfRange`）を返す。
#[test]
fn topk_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err = x.topk(1, 1, true).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange {
            axis: 1,
            rank: 1
        })
    ));
}

/// エラー経路: `k > shape[dim]` は `AutodiffError::Shape`
/// （`NarrowOutOfBounds`）を返す。
#[test]
fn topk_k_exceeds_dim_size_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 3]));
    let err = x.topk(5, 1, true).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::NarrowOutOfBounds {
            dim: 1,
            start: 0,
            len: 5,
            dim_size: 3,
        })
    ));
}

// --- cumsum／cumprod（イシュー #1731） ---

/// ①forward: `cumsum` が各軸で累積和を返す（2 次元テンソルの両軸で
/// 確認）。
#[test]
fn cumsum_forward_along_each_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let along_rows = x.cumsum(0).unwrap();
    assert_eq!(
        dense_vec(&along_rows.to_tensor()),
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    );

    let along_cols = x.cumsum(1).unwrap();
    assert_eq!(
        dense_vec(&along_cols.to_tensor()),
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );
}

/// ②`cumsum` が `f64` アキュムレータを保持する契約を直接確認する
/// （`[1e8, 1.0, -1e8]` は f32 逐次アキュムレータなら末尾が `0.0` に
/// 丸まるが、f64 アキュムレータでは `1.0` が厳密に残る）。
#[test]
fn cumsum_uses_persistent_f64_accumulator() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e8, 1.0, -1e8], &[3]));
    let out = x.cumsum(0).unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), vec![1e8, 1e8, 1.0]);
}

/// ③`cumsum` の backward を中央差分・解析式（`dx[i] = Σ_{j>=i} w[j]`）
/// の双方と突合する（`loss = sum(cumsum(x) ⊙ w)`）。
#[test]
fn cumsum_backward_matches_numeric() {
    let x0 = t(vec![0.5, -1.5, 2.0, 0.25], &[4]);
    let w = t(vec![1.0, -2.0, 0.5, 3.0], &[4]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let wv = tape.var(&w);
        let out = xv.cumsum(0).unwrap();
        scalar(&out.mul(&wv).unwrap().sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let wv = tape.var(&w);
    let out = xv.cumsum(0).unwrap();
    let loss = out.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("cumsum dX", dx, &num_dx);

    // 解析式 dx[i] = Σ_{j>=i} w[j] との突合（厳密一致）。
    let w_data = dense_vec(&w);
    let expected: Vec<f32> = (0..w_data.len())
        .map(|i| w_data[i..].iter().sum())
        .collect();
    assert_eq!(dense_vec(dx), expected);
}

/// ④`cumprod` が各軸で累積積を返す。
#[test]
fn cumprod_forward_along_each_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));

    let along_rows = x.cumprod(0).unwrap();
    assert_eq!(
        dense_vec(&along_rows.to_tensor()),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]
    );

    let along_cols = x.cumprod(1).unwrap();
    assert_eq!(
        dense_vec(&along_cols.to_tensor()),
        vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]
    );
}

/// ⑤`cumprod` の `f64` アキュムレータ契約: `[1e-30, 1e-30, 1e30]` は
/// `out[1] = 1e-60` が `f32` では underflow して `0.0` になるが、
/// `out[2]`（`1e-60 * 1e30 = 1e-30` 相当）は f64 アキュムレータでは
/// 非零有限値のまま残る（f32 逐次アキュムレータなら `out[1]` が
/// `0.0` になった時点で以降も `0.0` のまま）。判別は `out[2]` の
/// 非零性・有限性のみで行う（`f32` 入力は `1e-30`／`1e30` に厳密で
/// はないため、リテラル比較はしない）。
#[test]
fn cumprod_underflow_recovers_via_f64_accumulator() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1e-30, 1e-30, 1e30], &[3]));
    let out = dense_vec(&x.cumprod(0).unwrap().to_tensor());
    assert_eq!(out[1], 0.0, "f32 では 1e-60 相当が underflow するはず");
    assert_ne!(
        out[2], 0.0,
        "f64 アキュムレータにより 1e-60*1e30 相当が非零で残るはず"
    );
    assert!(out[2].is_finite());
}

/// ⑥〜⑧`cumprod` の backward（厳密形 `d_x[i] = L[i]·S[i]`）を中央差分
/// と突合する。零要素の個数（0／1／2 個）を分けて検証し、最初の零
/// 要素より後ろの入力勾配が厳密に `0.0` になる解析的性質も確認する
/// （`L[i] = 0` となるため。最初の零要素自身の勾配は一般に非零）。
/// `dim` を指定できるようにして、`inner > 1`（縮約軸より後ろに別軸が
/// あり `idx_next = (o*axis_len+(a+1))*inner+i` のような lane 添字計算を
/// 経由する）多次元ケースも同一関数で検証できるようにしている
/// （レビュー指摘: rank-1 限定だったカバレッジの拡張）。
fn cumprod_backward_case_along(
    x0: Tensor<f32>,
    w: Tensor<f32>,
    dim: usize,
    label: &str,
    first_zero_along_axis: Option<usize>,
) {
    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let wv = tape.var(&w);
        let out = xv.cumprod(dim).unwrap();
        scalar(&out.mul(&wv).unwrap().sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let wv = tape.var(&w);
    let out = xv.cumprod(dim).unwrap();
    let loss = out.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close(label, dx, &num_dx);

    if let Some(fz) = first_zero_along_axis {
        // shape 全体を走査し、縮約軸（dim）上のインデックスが fz より
        // 後ろの要素はすべて勾配 0 になることを検証する（多次元でも
        // L[a] = 0 の性質は縮約軸方向にのみ依存するため）。
        let shape = dx.shape().to_vec();
        let numel: usize = shape.iter().product();
        let mut index = vec![0usize; shape.len()];
        for _ in 0..numel {
            if index[dim] > fz {
                let v = dx.get(&index).unwrap_or(0.0);
                assert_eq!(
                    v, 0.0,
                    "{label}: index {index:?} の勾配は 0 のはず（L[{fz}]=0 が dim={dim} 方向へ伝播）"
                );
            }
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
}

fn cumprod_backward_case(x0: Tensor<f32>, w: Tensor<f32>, label: &str, first_zero: Option<usize>) {
    cumprod_backward_case_along(x0, w, 0, label, first_zero);
}

#[test]
fn cumprod_backward_matches_numeric_without_zeros() {
    cumprod_backward_case(
        t(vec![0.5, -1.5, 2.0, 0.25], &[4]),
        t(vec![1.0, -2.0, 0.5, 3.0], &[4]),
        "cumprod dX (no zero)",
        None,
    );
}

#[test]
fn cumprod_backward_matches_numeric_with_single_zero() {
    cumprod_backward_case(
        t(vec![0.5, 0.0, 2.0, -0.25], &[4]),
        t(vec![1.0, -2.0, 0.5, 3.0], &[4]),
        "cumprod dX (single zero)",
        Some(1),
    );
}

#[test]
fn cumprod_backward_matches_numeric_with_two_zeros() {
    cumprod_backward_case(
        t(vec![0.5, 0.0, 2.0, 0.0, -0.25], &[5]),
        t(vec![1.0, -2.0, 0.5, 3.0, -1.0], &[5]),
        "cumprod dX (two zeros)",
        Some(1),
    );
}

/// ⑨`cumprod` backward の 3 次元・中間軸（`inner > 1`）ケース。shape
/// `[2, 4, 3]` で軸 1（`outer=2`・`axis_len=4`・`inner=3`）を縮約軸に取り、
/// `backend-cpu::scan` の lane 添字計算
/// `idx_next = (o*axis_len+(a+1))*inner+i` を実際に経由する経路を
/// 中央差分と突合する（レビュー指摘: rank-1 限定だったカバレッジの
/// 拡張）。全 lane（`outer × inner` の各組）の軸 1 = 0 位置を零要素に
/// 揃えることで、どの lane でも `L[a]=0 (a>=1)` が成り立ち、縮約軸方向
/// の勾配ゼロ伝播（`first_zero_along_axis = 0`）を全 lane 一律で検証
/// できるようにしている。
#[test]
fn cumprod_backward_matches_numeric_with_inner_axis() {
    #[rustfmt::skip]
    let x0 = t(
        vec![
            // outer=0
            0.0, 0.0, 0.0,
            1.5, 0.5, -0.5,
            -0.5, 2.0, 1.0,
            -2.0, 0.25, 3.0,
            // outer=1
            0.0, 0.0, 0.0,
            2.0, -1.5, 0.25,
            0.5, 0.5, -1.0,
            1.0, -0.25, 2.0,
        ],
        &[2, 4, 3],
    );
    #[rustfmt::skip]
    let w = t(
        vec![
            1.0, -2.0, 0.5,
            3.0, -1.0, 2.0,
            0.5, 1.0, -1.5,
            -2.0, 0.25, 1.0,
            2.0, 0.5, -1.0,
            -1.0, 1.5, 0.5,
            0.25, -2.0, 1.0,
            1.0, 0.5, -0.5,
        ],
        &[2, 4, 3],
    );
    cumprod_backward_case_along(x0, w, 1, "cumprod dX (3d inner axis)", Some(0));
}

/// ⑨.5 `cumprod` backward のオーバーフロー×零遮断の相互作用回帰テスト
/// （codex-review 指摘・PR #1819）。長い軸（`axis_len = 40`）の先頭を
/// 零要素、以降を `1e30`（`f32` 表現域には収まるが、`f64` でも十数個の連続積で表現域
/// `1.8e308` を突破する規模）で埋める。素朴な浮動小数点乗算のままだと
/// `S`（Horner 型再帰の後方累積）が零要素より後ろの区間で `inf` へ
/// 発散し、`L[a] = 0`（零要素を跨いだ排他的 prefix 積）との積
/// `0.0 * inf` が `NaN` を生む。零要素を跨いだ位置（`index >= 1`）の
/// 真の勾配は理論上つねに厳密な `0.0`（`d(y[b])/d(x[a])` の積が零要素
/// で遮断されるため）であるはずで、`NaN` に汚染されてはならない。
/// 零要素自身（`index == 0`）は `S[0]` 自体が発散しうるため `inf` は
/// 許容するが `NaN` は許容しない。中央差分（オーバーフロー環境では
/// 数値的に破綻するため）は使わず、解析勾配の有限性・零遮断の厳密性
/// のみを検証する。
#[test]
fn cumprod_backward_no_nan_when_zero_blocks_overflowing_suffix() {
    const AXIS_LEN: usize = 40;
    let mut data = vec![1e30f32; AXIS_LEN];
    data[0] = 0.0;
    let x0 = t(data, &[AXIS_LEN]);
    let w = t(vec![1.0; AXIS_LEN], &[AXIS_LEN]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let wv = tape.var(&w);
    let out = xv.cumprod(0).unwrap();
    let loss = out.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    for a in 0..AXIS_LEN {
        let v = dx.get(&[a]).unwrap_or(0.0);
        assert!(
            !v.is_nan(),
            "cumprod backward (overflow×zero): index {a} の勾配が NaN になってはならない（実際: {v}）"
        );
        if a >= 1 {
            assert_eq!(
                v, 0.0,
                "cumprod backward (overflow×zero): index {a} は L[{a}]=0 のため勾配は厳密に 0 のはず（実際: {v}）"
            );
        }
    }
}

/// ⑨.6 同上のオーバーフロー×零遮断の相互作用を、複数の零要素を含む
/// 長い軸で再確認する（codex-review 指摘の「複数零を含む回帰テスト」
/// 要求への対応）。`axis_len = 50` に零要素を 2 箇所（先頭付近と中間）
/// 配置し、両方の零要素より後ろで一貫して勾配が厳密 `0.0`・かつ
/// `NaN` が出現しないことを検証する。
#[test]
fn cumprod_backward_no_nan_with_multiple_zeros_and_overflow() {
    const AXIS_LEN: usize = 50;
    const FIRST_ZERO: usize = 2;
    const SECOND_ZERO: usize = 25;
    let mut data = vec![1e30f32; AXIS_LEN];
    data[FIRST_ZERO] = 0.0;
    data[SECOND_ZERO] = 0.0;
    let x0 = t(data, &[AXIS_LEN]);
    let w = t(vec![1.0; AXIS_LEN], &[AXIS_LEN]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let wv = tape.var(&w);
    let out = xv.cumprod(0).unwrap();
    let loss = out.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    for a in 0..AXIS_LEN {
        let v = dx.get(&[a]).unwrap_or(0.0);
        assert!(
            !v.is_nan(),
            "cumprod backward (multi-zero×overflow): index {a} の勾配が NaN になってはならない（実際: {v}）"
        );
        if a > FIRST_ZERO {
            assert_eq!(
                v, 0.0,
                "cumprod backward (multi-zero×overflow): index {a} は最初の零要素（{FIRST_ZERO}）より後ろなので勾配は厳密に 0 のはず（実際: {v}）"
            );
        }
    }
}

/// ⑨.7 `cumprod` backward の suffix 積自体がオーバーフローする反例
/// （codex-review P1 指摘・PR #1819）。`x = [0] + [2^-100]×11 +
/// [2^100]×11`（`axis_len = 23`）・`w = 1`（`loss = sum(cumprod(x))`）
/// では、forward は全要素 `0.0`（先頭が零のため）だが、`S`（Horner
/// 型後方再帰）を素朴な `f64` で計算すると軸後半の `2^100` の連続積で
/// `f64` 表現域（約 `1.8e308`）を超えて `inf` に発散し、その後軸前半の
/// `2^-100` を掛けても復元できず `dx[0]` が誤って `inf` になっていた
/// （零遮断ガードは「掛け算の一方が厳密に `0.0`」のケースのみを救う
/// ため、この「非零だが極端に大きい／小さい係数」の反例には対応でき
/// ない）。解析的には `dx[0] = S[0] = 1 + 2^100·(1 + 2^100·(…))`
/// のうち、`2^-100` の 11 個は `S` の対応する加算段では `2^100` 側の
/// 桁に対し無視できるほど小さく（53bit 仮数の丸めで消える）、隣接する
/// `2^-100·2^100 = 1` の相殺だけが効くため、厳密に `S[0] = 2.0`
/// （`f32` に厳密に丸まる）になる。`a >= 1` はいずれも `L[a] = 0`
/// （`x[0] = 0` を跨ぐため）で勾配は厳密に `0.0`。中央差分は
/// オーバーフロー環境では意味を持たないため使わず、解析的な厳密値
/// との `assert_eq!` で検証する。
#[test]
fn cumprod_backward_no_overflow_when_suffix_product_diverges() {
    const AXIS_LEN: usize = 23;
    let mut data = vec![0f32; AXIS_LEN];
    data[0] = 0.0;
    for v in data.iter_mut().skip(1).take(11) {
        *v = 2f32.powi(-100);
    }
    for v in data.iter_mut().skip(12).take(11) {
        *v = 2f32.powi(100);
    }
    let x0 = t(data, &[AXIS_LEN]);
    let w = t(vec![1.0; AXIS_LEN], &[AXIS_LEN]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let wv = tape.var(&w);
    let out = xv.cumprod(0).unwrap();
    let loss = out.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    for a in 0..AXIS_LEN {
        let v = dx.get(&[a]).unwrap_or(0.0);
        assert!(
            !v.is_nan(),
            "cumprod backward (suffix overflow): index {a} の勾配が NaN になってはならない（実際: {v}）"
        );
        assert!(
            v.is_finite(),
            "cumprod backward (suffix overflow): index {a} の勾配が有限であるはず（実際: {v}）"
        );
        if a == 0 {
            assert_eq!(
                v, 2.0,
                "cumprod backward (suffix overflow): dx[0] は厳密に 2.0 のはず（実際: {v}）"
            );
        } else {
            assert_eq!(
                v, 0.0,
                "cumprod backward (suffix overflow): index {a} は L[{a}]=0 のため勾配は厳密に 0 のはず（実際: {v}）"
            );
        }
    }
}

/// ⑨.8 上記と対称なケース: 軸前半が極端に大きい `2^100`、後半が
/// 極端に小さい `2^-100`（`axis_len = 22`・零要素なし）で、`w` を
/// 末尾のみ `1`（他は `0`）の one-hot にする。`旧実装`（素朴な `f64`
/// アキュムレータ）はこのケースを 2 通りの経路で壊していた:
/// `L`（先頭からの累積積。前半 11 個の `2^100` 連続積）が `f64`
/// 表現域を超えて `inf` へ発散し、`S`（後半の `2^-100` 連続積からの
/// 後方累積）は逆に `f64` 表現域の下限（アンダーフロー）を割って
/// `0.0` に潰れ、両者の積が `inf * 0 = NaN` あるいは意図せぬ `0.0` に
/// なりうる。解析的には `y[axis_len-1] = Π x[i]`（`2^100` を 11 回・
/// `2^-100` を 11 回掛けた積で厳密に `1.0`）に対し、`w` が末尾のみ
/// `1` なので `dx[a] = y[axis_len-1] / x[a] = 1 / x[a]`——前半
/// （`a <= 10`）は `2^-100`、後半（`a >= 11`）は `2^100` で、いずれも
/// `f32` に厳密に表現可能な値になる（`x[a]` はすべて厳密に 2 の
/// べき乗のため `1/x[a]` も厳密に 2 のべき乗）。
#[test]
fn cumprod_backward_no_overflow_symmetric_large_then_small() {
    const AXIS_LEN: usize = 22;
    let mut data = vec![0f32; AXIS_LEN];
    for v in data.iter_mut().take(11) {
        *v = 2f32.powi(100);
    }
    for v in data.iter_mut().skip(11).take(11) {
        *v = 2f32.powi(-100);
    }
    let x0 = t(data, &[AXIS_LEN]);
    let mut w_data = vec![0f32; AXIS_LEN];
    w_data[AXIS_LEN - 1] = 1.0;
    let w = t(w_data, &[AXIS_LEN]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let wv = tape.var(&w);
    let out = xv.cumprod(0).unwrap();
    let loss = out.mul(&wv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let x_data = dense_vec(&x0);
    for (a, &x_a) in x_data.iter().enumerate().take(AXIS_LEN) {
        let v = dx.get(&[a]).unwrap_or(0.0);
        assert!(
            !v.is_nan(),
            "cumprod backward (symmetric overflow/underflow): index {a} の勾配が NaN になってはならない（実際: {v}）"
        );
        let expected = 1.0f32 / x_a;
        assert_eq!(
            v, expected,
            "cumprod backward (symmetric overflow/underflow): index {a} は 1/x[{a}]={expected} のはず（実際: {v}）"
        );
    }
}

/// エラー経路: `cumsum`／`cumprod` の `dim` が範囲外なら
/// `AutodiffError::Shape(AxisOutOfRange)` を返す。
#[test]
fn cumsum_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.cumsum(1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

#[test]
fn cumprod_axis_out_of_range_is_rejected() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x.cumprod(1).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { .. })
    ));
}

/// 空 shape（`[0, 3]`）に対する `cumsum` は空出力を返す（部分積
/// オーバーフロー回避の早期 return を確認）。
#[test]
fn cumsum_on_empty_tensor_returns_empty() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(Vec::new(), &[0, 3]));
    let out = x.cumsum(0).unwrap();
    assert_eq!(out.to_tensor().shape(), &[0, 3]);
    assert_eq!(dense_vec(&out.to_tensor()), Vec::<f32>::new());
}

/// strided（transpose view）入力でも `cumsum` が contiguous 入力と
/// 同じ結果を返す（`x.contiguous()` 経由で正しく読める契約）。
#[test]
fn cumsum_on_transposed_view_matches_contiguous() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let tr = x.transpose(0, 1).unwrap(); // shape [3, 2]
    let out_view = tr.cumsum(0).unwrap();

    let x_t = t(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let x2 = tape2.var(&x_t);
    let out_contig = x2.cumsum(0).unwrap();

    assert_eq!(
        dense_vec(&out_view.to_tensor()),
        dense_vec(&out_contig.to_tensor())
    );
}

// --- interpolate（nearest。イシュー #1757） ---

/// ①forward 値（解析）: 1-D 整数倍アップサンプル（×2）。添字式
/// `src = (dst * in) / out`（整数除算＝床）どおり、各出力は
/// `[x0,x0,x1,x1,x2,x2]` となる。
#[test]
fn interpolate_nearest_forward_upsample_1d() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let out = x
        .interpolate(&[6], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap();
    assert_eq!(
        dense_vec(&out.to_tensor()),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
}

/// ②backward（解析＋数値微分突合）: 1-D 整数倍アップサンプル
/// （×2）では各入力要素がちょうど 2 個の出力から参照されるため、
/// 一様勾配（sum loss）に対する `d_input` は `[2,2,2]`。
#[test]
fn interpolate_nearest_backward_upsample_matches_numeric() {
    let x0 = t(vec![1.0, 2.0, 3.0], &[3]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let out = xv
            .interpolate(&[6], fandhe_ai_tensor_core::InterpolateMode::Nearest)
            .unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let out = xv
        .interpolate(&[6], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap();
    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");

    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("interpolate nearest upsample dX", dx, &num_dx);
    assert_eq!(dense_vec(dx), vec![2.0, 2.0, 2.0]);
}

/// ③backward（解析）: 非整数比ダウンサンプル（8→3）では参照されない
/// 入力要素の勾配が厳密に 0 になる（添字式どおり参照集合は
/// `{0, 2, 5}` のみ）。
#[test]
fn interpolate_nearest_backward_downsample_unreferenced_is_exact_zero() {
    let x0 = t((1..=8).map(|v| v as f32).collect(), &[8]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let out = xv
        .interpolate(&[3], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap();
    // src = (dst*8)/3: dst=0->0, dst=1->2, dst=2->5。
    assert_eq!(dense_vec(&out.to_tensor()), vec![1.0, 3.0, 6.0]);

    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
    assert_eq!(dense_vec(dx), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let forward = |x: Tensor<f32>| -> f32 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&x);
        let out = xv
            .interpolate(&[3], fandhe_ai_tensor_core::InterpolateMode::Nearest)
            .unwrap();
        scalar(&out.sum(None).unwrap().to_tensor())
    };
    let num_dx = numeric_grad(&x0, forward);
    assert_grad_close("interpolate nearest downsample dX", dx, &num_dx);
}

/// ④恒等サイズ（`size == 入力の空間軸`）では forward が入力の
/// コピー・backward の勾配が upstream と bit 完全一致することを
/// 確認する（算術を含まない純粋なコピー演算のため。src=dst の
/// 全単射で重複書き込みが発生しない）。
#[test]
fn interpolate_nearest_identity_size_is_bit_exact_passthrough() {
    let x0 = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x0);
    let out = xv
        .interpolate(&[4], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap();
    assert_eq!(dense_vec(&out.to_tensor()), dense_vec(&x0));

    let upstream = t(vec![10.0, 20.0, 30.0, 40.0], &[4]);
    let uv = tape.var(&upstream);
    let loss = out.mul(&uv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
    // `d(sum(out * upstream))/d(input) = upstream`（恒等写像のため
    // reduction を経ない bit 完全一致）。
    assert_eq!(dense_vec(dx), dense_vec(&upstream));
}

/// ⑤2-D（先頭 batch 軸付き）: 末尾 1 軸のみが空間軸で、batch 軸は
/// 素通しされ各行が独立にリサンプリングされることを確認する。
#[test]
fn interpolate_nearest_2d_leading_batch_axis_is_independent() {
    let tape = Tape::new_with_ops(common::naive_ops());
    // batch=2, spatial=2 -> spatial=4（各行 ×2 アップサンプル）。
    let x = tape.var(&t(vec![1.0, 2.0, 10.0, 20.0], &[2, 2]));
    let out = x
        .interpolate(&[4], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap();
    assert_eq!(out.to_tensor().shape(), &[2, 4]);
    assert_eq!(
        dense_vec(&out.to_tensor()),
        vec![1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 20.0, 20.0]
    );

    let loss = out.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2, 2]);
    assert_eq!(dense_vec(dx), vec![2.0, 2.0, 2.0, 2.0]);
}

/// ⑥エラー経路: `size` が空（0 軸指定）だと `AutodiffError::Shape`
/// （`RankMismatch { expected: 1, actual: 0 }`）を返す。
#[test]
fn interpolate_rejects_empty_size() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let err = x
        .interpolate(&[], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::RankMismatch {
            expected: 1,
            actual: 0
        })
    ));
}

/// ⑦エラー経路: 出力側の空間軸サイズが 0 だと
/// `AutodiffError::Shape`（`ShapeMismatch`）を返す（ゼロ除算回避の
/// 事前拒否）。
#[test]
fn interpolate_rejects_zero_output_spatial_axis() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let err = x
        .interpolate(&[0], fandhe_ai_tensor_core::InterpolateMode::Nearest)
        .unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(fandhe_ai_tensor_core::ShapeError::ShapeMismatch { .. })
    ));
}
