//! 演算ごとの勾配関数（VJP: vector-Jacobian product）と `Op` 単位の
//! ディスパッチ入口 `vjp()`。
//!
//! TASK-1.5a（#16）が記録したテープ構造（`tape::Op`/`TapeNode`）に対し、
//! 「出力側勾配（upstream）→ 各入力 `NodeId` への勾配」の変換層を提供
//! する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.5、
//! `docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/tape.rs` の
//! backward 実装）。`Tape::backward`（`backward.rs`・TASK-1.5c・#18）は
//! ノード列を発生順とは逆順に走査しながら本モジュールの `vjp()` を
//! 呼び、返り値（入力 `NodeId` ごとの勾配寄与）を蓄積する。**勾配の
//! 蓄積そのものは本モジュールの責務ではない**（`backward.rs` 側で
//! 複数の出力先から同一入力ノードへ流入する勾配を合算する）。
//!
//! 値計算は `eval.rs`（クレート非公開の暫定 CPU 参照実装）のヘルパー
//! を再利用し、forward と勾配計算で数式の実体を 2 か所に別実装しない
//! （PoC-v2-2 の方針を踏襲。`eval::matmul` は `f32::mul_add` による
//! FMA 契約統一済みのため、`MatMul` の VJP もここを経由するだけで
//! 契約を引き継ぐ）。

use tensor_core::Tensor;

use crate::eval::{self, build_tensor, dense_vec};
use crate::tape::{NodeId, Op, TapeNode};
use crate::var::Reduction;

/// ノード 1 個分の VJP。`upstream`（出力側勾配）と記録済みノード列
/// `nodes` から、各入力 `NodeId` への勾配寄与を返す。`out_value` は
/// 当該ノードの forward 記録値で、`Exp`/`Tanh`/`Sigmoid`/`Max` が
/// 再計算を避けて再利用する（`Sigmoid` は TASK-9.1b・#92 で追加）。
/// `Op::Leaf` は入力を持たないため空 `Vec` を返す。
pub(crate) fn vjp(
    op: &Op,
    out_value: &Tensor<f32>,
    upstream: &Tensor<f32>,
    nodes: &[TapeNode],
) -> Vec<(NodeId, Tensor<f32>)> {
    match *op {
        Op::Leaf => Vec::new(),
        Op::MatMul(a, b) => {
            let (da, db) = matmul_vjp(&nodes[a.0].value, &nodes[b.0].value, upstream);
            vec![(a, da), (b, db)]
        }
        Op::Add(a, b) => {
            let a_shape = nodes[a.0].value.shape();
            let b_shape = nodes[b.0].value.shape();
            let da = reduce_to_shape(upstream, a_shape);
            let db = reduce_to_shape(upstream, b_shape);
            vec![(a, da), (b, db)]
        }
        Op::Mul(a, b) => {
            let a_val = &nodes[a.0].value;
            let b_val = &nodes[b.0].value;
            let da = reduce_to_shape(&eval::mul(upstream, b_val), a_val.shape());
            let db = reduce_to_shape(&eval::mul(upstream, a_val), b_val.shape());
            vec![(a, da), (b, db)]
        }
        Op::Relu(a) => {
            // 劣勾配は x = 0 で 0 とする（PoC-v2-2 準拠）。NaN 入力は
            // マスク不成立（`v > 0.0` が false）となり勾配 0 を返す。
            let da = elementwise_mul_mask(upstream, &nodes[a.0].value, |v| v > 0.0);
            vec![(a, da)]
        }
        Op::Exp(a) => {
            // d/dx exp(x) = exp(x)。forward 記録値 `out_value` を
            // 再利用し `exp` を再計算しない。
            let da = eval::mul(upstream, out_value);
            vec![(a, da)]
        }
        Op::Tanh(a) => {
            // d/dx tanh(x) = 1 - tanh(x)^2。同じく `out_value` を再利用。
            let factor = tanh_grad_factor(out_value);
            let da = eval::mul(upstream, &factor);
            vec![(a, da)]
        }
        Op::Sigmoid(a) => {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))。
            // `Exp`/`Tanh` と同じく forward 記録値 `out_value`
            // （= sigmoid(x)）を再利用し再計算しない（TASK-9.1b・#92）。
            let factor = sigmoid_grad_factor(out_value);
            let da = eval::mul(upstream, &factor);
            vec![(a, da)]
        }
        Op::Sum { input, dim } => {
            let input_shape = nodes[input.0].value.shape();
            let da = unreduce_broadcast(upstream, input_shape, dim);
            vec![(input, da)]
        }
        Op::Max { input, dim } => {
            let da = max_vjp(&nodes[input.0].value, dim, out_value, upstream);
            vec![(input, da)]
        }
        Op::MseLoss {
            pred,
            target,
            reduction,
        } => {
            let (dpred, dtarget) = mse_loss_vjp(
                &nodes[pred.0].value,
                &nodes[target.0].value,
                upstream,
                reduction,
            );
            vec![(pred, dpred), (target, dtarget)]
        }
    }
}

/// 2 次元 `matmul` の転置。shape 検査は forward（`Var::matmul` →
/// `matmul_out_shape`）が済ませた 2 次元前提であり、`transpose(0, 1)`
/// は構造的に失敗しえない。それでも本番経路で `unwrap()`/`expect()`
/// を使わない方針（`.claude/rules/coding-rust.md`）のため、失敗時は
/// `debug_assert!` で契約違反を検知しつつ入力をそのまま返す
/// （到達すれば forward 側の shape 検査ロジックにバグがある）。
fn transpose2d(tensor: &Tensor<f32>) -> Tensor<f32> {
    match tensor.transpose(0, 1) {
        Ok(t) => t,
        Err(_) => {
            debug_assert!(
                false,
                "transpose2d: matmul VJP の rank-2 前提が崩れた（forward 側の契約違反）"
            );
            tensor.clone()
        }
    }
}

/// `MatMul(A, B)` の VJP: `dA = g @ Bᵀ`、`dB = Aᵀ @ g`
/// （`A: [m,k]`・`B: [k,n]`・`g: [m,n]`）。`eval::matmul` を再利用する
/// ため FMA 契約（`f32::mul_add`）は forward と自動的に統一される。
fn matmul_vjp(a: &Tensor<f32>, b: &Tensor<f32>, g: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
    let b_t = transpose2d(b);
    let a_t = transpose2d(a);
    let da = eval::matmul(g, &b_t);
    let db = eval::matmul(&a_t, g);
    (da, db)
}

/// ブロードキャストの逆演算。`add`/`mul` の VJP が返す勾配は forward
/// 出力の shape（ブロードキャスト後）を持つため、元の入力 shape
/// （`target_shape`）へ縮約する必要がある。NumPy 風ブロードキャスト
/// は「先頭に新設された軸」と「入力側が size 1 だった軸」を複製する
/// ため、その逆演算は同じ軸集合を合計で潰せばよい（PoC-v2-2 の
/// `sum_rows_data` を任意 shape・任意軸へ一般化した実装）。
fn reduce_to_shape(g: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let g_shape = g.shape().to_vec();
    if g_shape == target_shape {
        return g.clone();
    }
    debug_assert!(
        g_shape.len() >= target_shape.len(),
        "reduce_to_shape: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = g_shape.len() - target_shape.len();
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let mut data = dense_vec(g);
    let mut cur_shape = g_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            let mut reduced = vec![0f32; outer * inner];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + a) * inner + i;
                        reduced[o * inner + i] += data[src];
                    }
                }
            }
            data = reduced;
            cur_shape[axis] = 1;
        }
    }
    build_tensor(data, target_shape)
}

/// 同 shape の 2 テンソルに対する要素ごとの条件付き選択
/// （`g` をそのまま通すか 0 にするかを `mask_src` の値で決める）。
/// `Relu` の VJP（`g ⊙ 1[x > 0]`）専用の最小実装。
fn elementwise_mul_mask(
    g: &Tensor<f32>,
    mask_src: &Tensor<f32>,
    keep: impl Fn(f32) -> bool,
) -> Tensor<f32> {
    let shape = g.shape().to_vec();
    let g_data = dense_vec(g);
    let mask_data = dense_vec(mask_src);
    let out: Vec<f32> = g_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
        .collect();
    build_tensor(out, &shape)
}

/// `Tanh` の VJP 係数 `1 - tanh(x)^2` を forward 記録値 `out_value`
/// （= `tanh(x)`）から計算する（再計算を避ける）。
fn tanh_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| 1.0 - v * v).collect();
    build_tensor(out, &shape)
}

/// `Sigmoid` の VJP 係数 `sigmoid(x) * (1 - sigmoid(x))` を forward
/// 記録値 `out_value`（= `sigmoid(x)`）から計算する（TASK-9.1b・#92。
/// `tanh_grad_factor` と同型の out_value 再利用パターン）。
fn sigmoid_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| v * (1.0 - v)).collect();
    build_tensor(out, &shape)
}

/// `Sum` の VJP: 出力側勾配 `g` を入力 shape へブロードキャストして
/// 複製する（`sum` の逆演算は複製、`reduce_out_shape` が縮約軸を
/// 除去済み〈keepdim なし〉のため、`dim: Some(axis)` はいったん
/// size-1 軸を挿入してから `broadcast_to` する）。
fn unreduce_broadcast(g: &Tensor<f32>, input_shape: &[usize], dim: Option<usize>) -> Tensor<f32> {
    match dim {
        None => {
            let value = dense_vec(g).first().copied().unwrap_or(0.0);
            match Tensor::full(input_shape, value) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: dim=None の full() 構築が失敗した（契約違反）"
                    );
                    g.clone()
                }
            }
        }
        Some(axis) => {
            let mut inserted_shape = g.shape().to_vec();
            inserted_shape.insert(axis, 1);
            let reshaped = match g.contiguous().reshape(&inserted_shape) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: reduce_out_shape 逆算の reshape が失敗した（契約違反）"
                    );
                    return g.clone();
                }
            };
            match reshaped.broadcast_to(input_shape) {
                Ok(t) => t.contiguous(),
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: 挿入軸からの broadcast_to が失敗した（契約違反）"
                    );
                    reshaped
                }
            }
        }
    }
}

/// `Max` の VJP: 出力側勾配 `g` を、縮約軸に沿った最大値の位置のみへ
/// 伝播する。**同値タイは「最初に現れる最大要素 1 箇所のみ」へ伝播
/// する**（PyTorch `amax` の均等分配とは異なる、決定的な選択。
/// PoC-v2-2 のビット一致決定性方針・`train_repro` と整合させるための
/// 設計判断であり、`compat` 層〈REQ-9〉実装時に PyTorch 互換が必要に
/// なった場合は要再確認。追跡: Issue #224）。`out_value` は forward
/// 記録済みの縮約後最大値で、走査中に現れる要素と exact 一致するかで
/// argmax 位置を判定する（同一データ・同一 reduction 経路のため bit
/// 一致する）。
fn max_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_value: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let in_shape = input.shape().to_vec();
    let in_data = dense_vec(input);
    let g_data = dense_vec(g);
    let out_data = dense_vec(out_value);
    let mut grad = vec![0f32; in_data.len()];
    match dim {
        None => {
            if let (Some(target), Some(gv)) = (out_data.first(), g_data.first())
                && let Some(idx) = in_data.iter().position(|&v| v == *target)
            {
                grad[idx] = *gv;
            }
        }
        Some(axis) => {
            let outer: usize = in_shape[..axis].iter().product();
            let axis_len = in_shape[axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            for o in 0..outer {
                for i in 0..inner {
                    let out_idx = o * inner + i;
                    // `out_data`/`g_data` の要素数は `reduce_out_shape`
                    // の契約上 `outer * inner` と一致するはずだが、
                    // 本ファイルの他ヘルパー（`transpose2d`／
                    // `unreduce_broadcast` 等）と同様、契約違反時に
                    // release ビルドで境界外アクセス panic させず
                    // `debug_assert!` で検知しつつ安全側（当該要素の
                    // 勾配は 0 のまま）へフォールバックする
                    // （coding-rust.md「本番経路で unwrap/expect を
                    // 使わない」方針の趣旨に揃える）。
                    let (Some(&target), Some(&g_val)) =
                        (out_data.get(out_idx), g_data.get(out_idx))
                    else {
                        debug_assert!(
                            false,
                            "max_vjp: out_value/g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                        );
                        continue;
                    };
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        match in_data.get(src) {
                            Some(&v) if v == target => {
                                grad[src] = g_val;
                                break;
                            }
                            Some(_) => {}
                            None => {
                                debug_assert!(
                                    false,
                                    "max_vjp: input の要素数が in_shape と不一致（契約違反）"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    build_tensor(grad, &in_shape)
}

/// `MseLoss{pred, target, reduction}` の VJP: `dPred = g · 2(pred −
/// target) / n`（mean）／`g · 2(pred − target)`（sum）、
/// `dTarget = −dPred`（`g` はスカラー上流勾配）。#190 で sum 縮約を
/// 追加（`reduction` 分岐は forward の `eval::mse_loss` と対称）。
/// `n == 0` は mean・sum ともゼロ除算を避け zeros を返す。PoC-v2-2 は
/// `target` 側の勾配計算をスキップしていたが、本実装は数学的に完全な
/// VJP を返す（`target` 側を使うか捨てるかは #18 の勾配蓄積側の責務で
/// あり、ここでは両方提供する）。
fn mse_loss_vjp(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    g: &Tensor<f32>,
    reduction: Reduction,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = pred.shape().to_vec();
    let n = pred.numel();
    if n == 0 {
        let zeros = build_tensor(vec![0f32; 0], &shape);
        return (zeros.clone(), zeros);
    }
    let g_value = dense_vec(g).first().copied().unwrap_or(0.0);
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let scale = match reduction {
        Reduction::Mean => g_value * 2.0 / n as f32,
        Reduction::Sum => g_value * 2.0,
    };
    let dpred_data: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| scale * (p - t))
        .collect();
    let dtarget_data: Vec<f32> = dpred_data.iter().map(|&v| -v).collect();
    let dpred = build_tensor(dpred_data, &shape);
    let dtarget = build_tensor(dtarget_data, &shape);
    (dpred, dtarget)
}

#[cfg(test)]
mod tests {
    //! 受け入れ条件「各演算の解析勾配が数値微分と一致する」の直接検証。
    //!
    //! 各演算について、固定の重みテンソル `s`（forward 出力と同じ
    //! shape）によるスカラー射影 `L(x) = Σ (op(x) ⊙ s)` を定義し、
    //! 解析側（`vjp` 経由）と数値側（中央差分。f64 で集計）を突合する。
    //! `Tape`/`Var` を経由せず `eval.rs` の値計算を直接叩くため、
    //! `Tape::backward`（#18）が未実装でも検証できる。
    //!
    //! **判定基準**: 要素ごとに「相対誤差
    //! `|ad − num| / max(|ad|, |num|, τ)` が 1e-2 以下」または
    //! 「絶対誤差が 1e-3 以下」（`h = 1e-3`・`τ = 1e-4`）。PoC-v2-2 の
    //! 1e-4 は f64 前提（`torch.autograd.gradcheck` と同じ理由で f64
    //! 必須と PoC 自身が明記）だが、本実装は f32 のため中央差分の
    //! 丸め誤差床 `≈ ε_f32 · |L| / h ≈ 1e-4` を踏まえた本イシュー
    //! 新規の grad-check 専用閾値とする（バックエンド間数値一致判定
    //! 〈相対 1e-3 / 絶対 1e-5〉とは別系統）。
    //!
    //! **注意（ユーザー承認待ち）**: `CLAUDE.md` Conventions・
    //! `.claude/rules/delegation-impl.md` の「テスト許容誤差の変更は
    //! ユーザー承認必須」規定は、本件（新規 grad-check 閾値の設定）にも
    //! 適用されるべきとレビューで指摘された。数学的妥当性は確認済み
    //! だが正式なユーザー承認は未了のため、承認依頼を Issue #223 に
    //! 起票済み。値の変更が必要になった場合は同 Issue で追跡する。
    //!
    //! **キンク・タイ回避**: ReLU は `|x| >= 10h` の固定入力のみ、
    //! Max は同値タイのない固定入力のみを使う（固定値のため再生成
    //! ガードは不要。PoC-v2-2 `grad_check.rs` と同方針）。

    use super::*;

    const H: f64 = 1e-3;
    const TAU: f32 = 1e-4;
    const REL_TOL: f32 = 1e-2;
    const ABS_TOL: f32 = 1e-3;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape)
            .expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
        let a = dense_vec(analytic);
        let n = dense_vec(numeric);
        assert_eq!(
            a.len(),
            n.len(),
            "{label}: analytic/numeric の要素数が一致しない"
        );
        for (i, (&av, &nv)) in a.iter().zip(n.iter()).enumerate() {
            let diff = (av - nv).abs();
            let rel = diff / av.abs().max(nv.abs()).max(TAU);
            assert!(
                rel <= REL_TOL || diff <= ABS_TOL,
                "{label}[{i}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
            );
        }
    }

    /// `L(x) = Σ (forward(x) ⊙ s)` の f64 集計によるスカラー射影値。
    fn scalar_dot(a: &Tensor<f32>, s: &Tensor<f32>) -> f64 {
        dense_vec(a)
            .iter()
            .zip(dense_vec(s).iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    }

    /// 単一入力 `x` に対する `L` の中央差分勾配（要素ごと・f64 集計）。
    fn numeric_grad_unary(
        x: &Tensor<f32>,
        s: &Tensor<f32>,
        forward: impl Fn(&Tensor<f32>) -> Tensor<f32>,
    ) -> Tensor<f32> {
        let shape = x.shape().to_vec();
        let mut data = dense_vec(x);
        let mut grad = vec![0f32; data.len()];
        for i in 0..data.len() {
            let orig = data[i] as f64;
            data[i] = (orig + H) as f32;
            let lp = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = (orig - H) as f32;
            let lm = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = orig as f32;
            grad[i] = ((lp - lm) / (2.0 * H)) as f32;
        }
        build_tensor(grad, &shape)
    }

    // --- MatMul ---

    #[test]
    fn matmul_grad_matches_numeric() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let g = s.clone();
        let (da, db) = matmul_vjp(&a, &b, &g);

        let num_da = numeric_grad_unary(&a, &s, |x| eval::matmul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::matmul(&a, x));

        assert_grad_close("matmul dA", &da, &num_da);
        assert_grad_close("matmul dB", &db, &num_db);
    }

    // --- Add（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn add_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(same) dA", &da, &num_da);
        assert_grad_close("add(same) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(bias) dA", &da, &num_da);
        assert_grad_close("add(bias) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(scalar) dA", &da, &num_da);
        assert_grad_close("add(scalar) dB", &db, &num_db);
    }

    // --- Mul（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn mul_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(same) dA", &da, &num_da);
        assert_grad_close("mul(same) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(bias) dA", &da, &num_da);
        assert_grad_close("mul(bias) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(scalar) dA", &da, &num_da);
        assert_grad_close("mul(scalar) dB", &db, &num_db);
    }

    // --- Relu（正負混在。|x| >= 10h でキンク回避） ---

    #[test]
    fn relu_grad_matches_numeric() {
        let a = t(&[2.0, -3.0, 0.5, -0.02, 1.5, -1.5], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.0], &[2, 3]);

        let g = s.clone();
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        let num_da = numeric_grad_unary(&a, &s, eval::relu);

        assert_grad_close("relu dA", &da, &num_da);
    }

    #[test]
    fn relu_subgradient_at_zero_is_zero() {
        // x = 0 における劣勾配は 0 とする（PoC-v2-2 準拠。中央差分は
        // キンクで数値的に不安定なため、ここは解析式の直接検証のみ）。
        let a = t(&[0.0], &[1]);
        let g = t(&[3.0], &[1]);
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        assert_eq!(dense_vec(&da), vec![0.0]);
    }

    // --- Exp ---

    #[test]
    fn exp_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::exp(&a);
        let g = s.clone();
        let da = eval::mul(&g, &out_value);
        let num_da = numeric_grad_unary(&a, &s, eval::exp);

        assert_grad_close("exp dA", &da, &num_da);
    }

    // --- Tanh ---

    #[test]
    fn tanh_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::tanh(&a);
        let g = s.clone();
        let factor = tanh_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::tanh);

        assert_grad_close("tanh dA", &da, &num_da);
    }

    // --- Sigmoid（飽和域を含む） ---

    #[test]
    fn sigmoid_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid dA", &da, &num_da);
    }

    #[test]
    fn sigmoid_grad_saturated_region_matches_numeric() {
        // |x| が大きい飽和域（勾配 ≈ 0）でも中央差分と一致することを
        // 確認する（`eval::sigmoid` の数値安定形が飽和域で NaN/Inf を
        // 出さないことの間接検証も兼ねる）。
        let a = t(&[8.0, -8.0, 15.0, -15.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid(saturated) dA", &da, &num_da);
    }

    // --- Sum（dim: None / Some(0) / Some(1)） ---

    #[test]
    fn sum_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), None);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, None, &[]));

        assert_grad_close("sum(None) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(0));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(0), &[3]));

        assert_grad_close("sum(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(1));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(1), &[2]));

        assert_grad_close("sum(dim=1) dA", &da, &num_da);
    }

    // --- Max（dim: None / Some(0) / Some(1)。同値タイなし） ---

    #[test]
    fn max_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let g = s.clone();
        let da = max_vjp(&a, None, &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, None, &[]));

        assert_grad_close("max(None) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let out_value = eval::max(&a, Some(0), &[3]);
        let g = s.clone();
        let da = max_vjp(&a, Some(0), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(0), &[3]));

        assert_grad_close("max(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let g = s.clone();
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(1), &[2]));

        assert_grad_close("max(dim=1) dA", &da, &num_da);
    }

    // --- MseLoss（pred/target 両勾配） ---

    #[test]
    fn mse_loss_grad_mean_matches_numeric() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Mean));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Mean));

        assert_grad_close("mse(mean) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(mean) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_sum_matches_numeric() {
        // sum 縮約（#190）。scale が `2/n` ではなく `2` になる分岐を
        // mean と同じ数値微分ハーネスで検証する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Sum));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Sum));

        assert_grad_close("mse(sum) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(sum) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_n_zero_is_zero() {
        // numel() == 0 はゼロ除算を避け zeros を返す（ガード条件の
        // 直接検証。中央差分は空テンソルに対して定義できないため
        // 解析式のみで確認する）。mean/sum いずれも同じ早期 return
        // 経路（`n == 0` 分岐）を通るため mean のみ代表して検証する。
        let pred = build_tensor(Vec::new(), &[0]);
        let target = build_tensor(Vec::new(), &[0]);
        let g = t(&[1.0], &[]);
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert!(dense_vec(&dpred).is_empty());
        assert!(dense_vec(&dtarget).is_empty());
    }

    // --- reduce_to_shape（中間軸縮約。ランク同一で先頭・末尾以外の
    //     軸を broadcast 元へ潰す経路。add/mul の bias broadcast テスト
    //     は末尾軸・スカラーテストは rank 0 のみで、この経路は未カバー
    //     だった） ---

    #[test]
    fn reduce_to_shape_middle_axis_matches_numeric() {
        // g: [2,4,3] を target_shape [2,1,3]（中間軸 dim=1 が size 1）
        // へ縮約する。add(a, b) の b 側勾配として同じ経路を通す
        // （a: [2,4,3]、b: [2,1,3] からの broadcast）。
        let a = t(
            &[
                1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 1.5, -0.5, 2.5, -1.5, 0.5, -2.5, 3.0, -1.0, 0.5,
                -0.5, 1.0, -1.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0,
            ],
            &[2, 4, 3],
        );
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[2, 1, 3]);
        let s = t(
            &[
                1.0, -1.0, 2.0, 0.5, 1.0, -0.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0, 0.2, -0.2, 0.4,
                0.1, 0.2, -0.1, 0.4, -0.4, 0.1, -0.1, 0.3, -0.2,
            ],
            &[2, 4, 3],
        );

        let db = reduce_to_shape(&s, b.shape());
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("reduce_to_shape(middle axis) dB", &db, &num_db);
    }

    // --- vjp() ディスパッチの疎通確認（#18 との継ぎ目契約） ---
    //
    // Low 指摘: MatMul のみが vjp() 経由で疎通確認されており、他の
    // 8 演算（Add/Mul/Relu/Exp/Tanh/Sum/Max/MseLoss）は内部ヘルパーを
    // 直接呼ぶ形でしか検証されていなかった。各 match アームの配線
    // （`nodes[a.0]`/`nodes[b.0]` の対応順序）を通しで検証するため、
    // 全 9 演算（Leaf を除く）を vjp() 経由でテストする。Sigmoid
    // （TASK-9.1b・#92）追加により対象は 10 演算に拡大。

    fn leaf_node(value: Tensor<f32>) -> TapeNode {
        TapeNode {
            op: Op::Leaf,
            value,
        }
    }

    #[test]
    fn vjp_dispatch_matmul_returns_both_inputs() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let out_value = eval::matmul(&a, &b);
        let (expected_da, expected_db) = matmul_vjp(&a, &b, &t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]));
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::MatMul(NodeId(0), NodeId(1));
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_da));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_db));
    }

    #[test]
    fn vjp_dispatch_add_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::add(&a, &b);
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::Add(NodeId(0), NodeId(1));

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&g));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&g));
    }

    #[test]
    fn vjp_dispatch_mul_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::mul(&a, &b);
        let nodes = vec![leaf_node(a.clone()), leaf_node(b.clone())];
        let op = Op::Mul(NodeId(0), NodeId(1));

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&eval::mul(&g, &b)));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&eval::mul(&g, &a)));
    }

    #[test]
    fn vjp_dispatch_relu_returns_single_input() {
        let a = t(&[2.0, -3.0, 0.5, -0.02], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::relu(&a);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Relu(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&elementwise_mul_mask(&g, &a, |v| v > 0.0))
        );
    }

    #[test]
    fn vjp_dispatch_exp_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::exp(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Exp(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&eval::mul(&g, &out_value))
        );
    }

    #[test]
    fn vjp_dispatch_tanh_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::tanh(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Tanh(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &tanh_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sigmoid_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::sigmoid(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sigmoid(NodeId(0));

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &sigmoid_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sum_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::sum(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sum {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = unreduce_broadcast(&g, &[2, 3], Some(0));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_max_returns_single_input() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::max(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Max {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = max_vjp(&a, Some(0), &out_value, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_mse_loss_mean_returns_both_inputs_in_order() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Mean);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Mean,
        };

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_mse_loss_sum_returns_both_inputs_in_order() {
        // sum 縮約（#190）でも `Op::MseLoss` ディスパッチが reduction を
        // 正しく `mse_loss_vjp` へ引き渡すことを確認する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Sum);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Sum,
        };

        let grads = vjp(&op, &out_value, &g, &nodes);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }
}
