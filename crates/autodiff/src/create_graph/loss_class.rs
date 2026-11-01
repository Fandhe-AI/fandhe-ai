//! 損失のクラス軸系 6 variant（`CrossEntropyLossWithOptions`・`MultiMarginLoss`・
//! `MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・`CosineEmbeddingLoss`・`TripletMarginLoss`）の
//! `create_graph` 対応（イシュー #3800・親 #3719 系・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §40、実装記録は §55）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var` メソッド
//! （`cross_entropy_loss_with`／`multi_margin_loss`／`multilabel_margin_loss`／
//! `multilabel_soft_margin_loss`／`cosine_embedding_loss`／`triplet_margin_loss`）への委譲で足りるため
//! 親側に置く（forward の検証を迂回しない）。`CtcLoss` は承認により恒久的に対象外（fail-closed のまま）。
//!
//! **方式**（6 つとも V3: 値は 1 階のホスト関数〈f64 内部〉を `push_v3` で載せ、構造は二階を流すための
//! 合成。`s_b` は `loss_scale_bcast` 系で子テープ上の `g` を通すので、損失が根でなくても二階が `g` へ流れる）:
//! - `CrossEntropyLossWithOptions`（V3 + V1）: `dx = where(valid, s_b⊙(A⊙softmax(x) − B), 0)`。
//!   `A`・`B`・`valid` は class_weight・label_smoothing・ignore_index からホストが f64 で作る定数。
//!   Mean の分母は非 ignore サンプルの重み和 `W`（`g / const(W)`）。
//! - `MultiMarginLoss`: p=1 は区分定数（`K` をホスト定数化）、p=2 は `H = where(M, 2Z, 0)` を構造に積む。
//! - `MultiLabelMarginLoss`: x について区分定数（`K` をホスト定数化）。
//! - `MultiLabelSoftMarginLoss`: `dx = s_b⊙(const(w_c/C)⊙(σ(x) − t))`（σ は符号別の安定形）。
//! - `CosineEmbeddingLoss`（S1）: 行ごとに 2 のべき乗スケール `s` で `x/s` を作り、ε を `ε/s²` へ変換して
//!   `Mul`・`Sum`・`Sqrt`・`Div` でノルムを組む（`Op::VectorNorm` は使わない。承認は S1 の構造に対して出て
//!   いるため、`VectorNorm` が対象になった後も簡略化しない）。`coeff ∈ {−1, +1, 0}` は 1 階と同じ f64 の
//!   `cos` から決める V2 定数。
//! - `TripletMarginLoss`（S1 相当）: p ノルムをスケール形 `mx·(Σ(|v|/mx)^p)^(1/p)` で組む。`mx` は行ごとの
//!   差分の最大絶対値（ホスト定数）。p=2 は `u/norm'`（`u==0` のレーンの曲率を保つ）、p=1 の符号は V2 定数、
//!   一般の p は `sign(u)·(|u|/norm')^(p−1)`。`hinge_active`・`an_coeff`・`norm==0` は V2 の定数。
//!
//! 子テープへ積む Op は `Add`・`Sub`・`Mul`・`Div`・`Neg`・`Abs`・`Sqrt`・`Sigmoid`・`PowScalar`・`Sum`・
//! `Softmax`・`Gather`・`Where`・`BroadcastTo`・`Contiguous`・`Reshape`・定数 `Leaf` で、すべて
//! `supports_create_graph() == true`（閉包性。§40.11）。
//!
//! **総括判定（§40.12 の具体化）**: variant ごとに構造の各段と二階逆伝播の「cotangent × 微分係数」を
//! `f64` で上から押さえ、非有限または `STAGE_BOUND_GUARD`（2^125）以上の段があれば子テープへ最初に書く前に
//! 型付きエラーで拒否する（親の rollback で `child` は空に戻る）。入力の `±inf`・非有限の `g` も拒否する。
//! NaN のレーンは段検査から外して伝播させる。Triplet の `1<p<2` で差分が 0 のレーンは二階が発散する
//! （真値も特異）ため拒否する。
//!
//! **相殺の総括判定（第 2 の軸。§55.2）**: 段検査は大きな値しか見ないため、ε の吸収や差し引きの
//! 桁落ちで構造の二階が有限のまま f64 の参照から外れる入力（例: Cosine の平行な行で `ε'` が f32 の
//! `m'` から消える）を別に検査する。seed が 1 要素のときの二階の各要素を構造の演算順どおりに f64 で
//! 展開し、参照値と各項の絶対値の和から丸め誤差を見積もって REQ-2 の判定を外れうる入力を拒否する
//! （[`cancel_check`]。対象は Cosine・Triplet の x／g 経路、CEWO の Softmax 対角と g 経路、
//! MultiMargin p=2・MultiLabelSoft の g 経路）。保証はこの判定を通った入力に限る。
//!
//! GPU: 6 つとも 1 階はホスト関数（経路区分 H）。実機 parity は未実測（Phase 10 の #3857・#3859）。

use fandhe_ai_tensor_core::Tensor;

use super::{child_value, push_v3, reduce_dim_bcast};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::loss_ops::{CrossEntropyOptions, TripletMarginOptions};
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::{Reduction, Var};

fn invalid(msg: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {msg}（損失のクラス軸系。docs/autodiff-higher-order-grad-decision.md §40・§55 参照）"
    ))
}

/// 1 段の上界 `v`（f64）を検査する。NaN のレーンは伝播させるため素通し、非有限または
/// `STAGE_BOUND_GUARD` 以上は拒否する。
fn stage(v: f64) -> Result<(), AutodiffError> {
    if v.is_nan() {
        return Ok(());
    }
    if !v.is_finite() || v.abs() >= STAGE_BOUND_GUARD {
        return Err(invalid("構造の中間値が f32 範囲を超えうる入力のため拒否"));
    }
    Ok(())
}

/// 相殺（桁落ち）検査の丸め誤差係数 `(2·terms + 4)·2^-23`（`linalg_fft.rs::rounding_ok` と同じ形）。
/// `terms` は構造の最長の縮約長（Cosine・Triplet は行の幅 `d`、CEWO は Softmax の軸長 `C` 等）。
/// 縮約が f32 の逐次和でも成り立つ最悪値として長さに比例させる。
fn rounding_coeff(terms: usize) -> f64 {
    (2.0 * terms as f64 + 4.0) * (-23f64).exp2()
}

/// f32 構造の 1 要素の丸め誤差が REQ-2 の統一複合判定に収まるかの保守的検査（設計 §55.2 の
/// 「相殺の総括判定」）。`v` は同じ要素の f64 の参照値、`m` は同じ式を各項の絶対値の和で評価した
/// 大きさ（相殺が無ければ `|v|` と等しい）、`scale` は要素に掛かる上流係数（`|g|`・`1/n`・`1/W`・
/// 行スケール `1/s` 等の積）。誤差上界 `coeff·m·scale` が絶対 1e-5 未満か、相対の半分
/// （`linalg_fft.rs` と同じ保守的な閾値。許容自体は不変）以内でなければ拒否する。
/// ε の吸収（`Σa² + ε` で ε が f32 に残らない）も「`m` に比べて小さい `v`」として同じ式で見える。
/// NaN のレーンは伝播の契約に従い素通しにする。
fn cancel_check(v: f64, m: f64, scale: f64, coeff: f64, what: &str) -> Result<(), AutodiffError> {
    use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};
    if v.is_nan() || m.is_nan() || scale.is_nan() {
        return Ok(());
    }
    let bound = coeff * m * scale;
    if bound.is_finite()
        && (bound < ABSOLUTE_RESCUE_THRESHOLD || coeff * m <= (RELATIVE_TOLERANCE / 2.0) * v.abs())
    {
        return Ok(());
    }
    Err(invalid(&format!(
        "{what}: f32 構造の丸め誤差が相殺で増幅され二階が REQ-2 の判定を外れうるため拒否"
    )))
}

/// 速い経路: ブロック全体の誤差見積りの上界 `bound` が絶対判定を満たす（要素ごとの走査を省ける）か。
fn fast_ok(bound: f64) -> bool {
    bound.is_finite() && bound < fandhe_ai_tensor_core::tolerance::ABSOLUTE_RESCUE_THRESHOLD
}

/// `g` へ流れる二階（`∂dx/∂g`）の係数（Mean は `1/n`、Sum は `1`）。`g` の大きさは含まない。
fn g_scale(reduction: Reduction, n: usize) -> f64 {
    match reduction {
        Reduction::Mean => 1.0 / n as f64,
        Reduction::Sum => 1.0,
    }
}

/// 上流 `g`（スカラー）の絶対値を返す（有限・単一要素を要求し、段検査も通す）。
fn upstream_abs(g_value: &Tensor<f32>) -> Result<f64, AutodiffError> {
    let data = crate::eval::dense_vec(g_value);
    let Some(&gv) = data.first() else {
        return Err(invalid("上流勾配が空"));
    };
    if data.len() != 1 || !gv.is_finite() {
        return Err(invalid("上流勾配が非有限（または非スカラ）のため拒否"));
    }
    let gv = f64::from(gv).abs();
    stage(gv)?;
    Ok(gv)
}

/// reduction 後の scale 上界 `|g|/n`（Mean）／`|g|`（Sum）。
fn upstream_scale(
    g_value: &Tensor<f32>,
    reduction: Reduction,
    n: usize,
) -> Result<f64, AutodiffError> {
    let gv = upstream_abs(g_value)?;
    Ok(match reduction {
        Reduction::Mean => gv / n as f64,
        Reduction::Sum => gv,
    })
}

/// `g` への cotangent 上界（要素寄与の絶対値和 × 要素あたりの係数 `1/n`／`1`）を検査する。
fn check_g_cotangent(sum_abs: f64, reduction: Reduction, n: usize) -> Result<(), AutodiffError> {
    let v = match reduction {
        Reduction::Mean => sum_abs / n as f64,
        Reduction::Sum => sum_abs,
    };
    stage(v)
}

fn dense(parent: &Tape, parent_nodes: &[TapeNode], id: NodeId) -> Result<Vec<f32>, AutodiffError> {
    Ok(crate::eval::dense_vec(materialize_fallible(
        parent_nodes,
        parent.ops(),
        id,
    )?))
}

fn reject_inf(values: &[&[f32]]) -> Result<(), AutodiffError> {
    if values.iter().any(|v| v.iter().any(|x| x.is_infinite())) {
        return Err(invalid("入力に ±inf を含むため拒否"));
    }
    Ok(())
}

fn zeros_like<'c>(child: &'c Tape, shape: &[usize]) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::zeros(shape)?))
}

fn full_like<'c>(child: &'c Tape, shape: &[usize], v: f32) -> Result<Var<'c>, AutodiffError> {
    let n: usize = shape.iter().product();
    Ok(child.var_no_grad(&Tensor::new(vec![v; n], shape)?))
}

fn scalar(child: &Tape, v: f32) -> Var<'_> {
    child.var_no_grad(&Tensor::scalar(v))
}

fn const_vec<'c>(
    child: &'c Tape,
    data: Vec<f32>,
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    Ok(child.var_no_grad(&Tensor::new(data, shape)?))
}

fn mask(data: Vec<bool>, shape: &[usize]) -> Result<Tensor<bool>, AutodiffError> {
    Ok(Tensor::new(data, shape)?)
}

/// 親ノードの shape が揃い、上流が単一要素であることを確かめて要素 shape を返す。
fn checked_shape(
    parent_nodes: &[TapeNode],
    ids: &[NodeId],
    g: &Var<'_>,
) -> Result<Vec<usize>, AutodiffError> {
    let shape = parent_nodes[ids[0].0].shape.clone();
    if ids.iter().any(|id| parent_nodes[id.0].shape != shape)
        || g.shape().iter().product::<usize>() != 1
    {
        return Err(invalid("shape 不整合（内部契約違反）"));
    }
    Ok(shape)
}

fn sigma_pair_stable<'c>(
    child: &'c Tape,
    x_m: &Var<'c>,
    t: &Var<'c>,
    xs: &[f32],
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    // σ(x) − t を符号別に分解する（x ≥ 0: (1−t) − σ(−x)、x < 0: σ(x) − t）。大きい側の σ を
    // Sigmoid の出力から作ると VJP の y(1−y) が f32 の丸めで飽和域の曲率を 0 にするため。
    let nonneg = mask(xs.iter().map(|&x| x >= 0.0).collect(), shape)?;
    let one = scalar(child, 1.0);
    let form_pos = one.sub(t)?.sub(&x_m.neg()?.sigmoid())?;
    let form_neg = x_m.sigmoid().sub(t)?;
    Var::where_cond(&nonneg, &form_pos, &form_neg)
}

fn sigmoid64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// 正の有限値 `x` 以上の最小の 2 のべき乗（f64）。非正・非有限は 1。
fn pow2_ceil(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 || !x.is_finite() {
        return 1.0;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let frac = bits & ((1u64 << 52) - 1);
    let e = if frac == 0 { exp } else { exp + 1 };
    2f64.powi(e)
}

fn sign0(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------------
// CrossEntropyLossWithOptions
// ---------------------------------------------------------------------------------

/// `CrossEntropyLossWithOptions` の d_logits を子テープへ記録する。
pub(super) fn cewo_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, usize, Reduction),
    payload: (&Tensor<i32>, &CrossEntropyOptions),
    x_m: &Var<'c>,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (logits, class_dim, reduction) = spec;
    let (targets, options) = payload;
    let shape = checked_shape(parent_nodes, &[logits], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    if class_dim >= shape.len() {
        return Err(invalid("class_dim が範囲外（内部契約違反）"));
    }
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let target_data = crate::eval::dense_vec_i32(targets);
    if target_data.len() != outer * inner {
        return Err(invalid("targets の要素数が不整合（内部契約違反）"));
    }
    let eps = f64::from(options.label_smoothing_value());
    let ignore_index = options.ignore_index_value();
    let weights: Vec<f64> = match options.class_weight_value() {
        Some(cw) => crate::eval::dense_vec(cw)
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        None => vec![1.0; axis_len],
    };
    if weights.len() != axis_len {
        return Err(invalid("class_weight の長さが不整合（内部契約違反）"));
    }
    let total_all: f64 = weights.iter().sum();
    let in_range = |t: i32| t >= 0 && (t as usize) < axis_len;
    let mut denom_w = 0f64;
    for &t in &target_data {
        if ignore_index == Some(t) {
            continue;
        }
        if in_range(t) {
            denom_w += weights[t as usize];
        }
    }
    // 1 階 `cross_entropy_loss_with_options_vjp` の早期 return 2 種。logits が NaN でも 0 を返す
    // ため、`g` や logits に依存しない定数ゼロの cgrad にする。
    if total_all == 0.0 || (reduction == Reduction::Mean && denom_w == 0.0) {
        return zeros_like(child, &shape);
    }
    let xs = dense(parent, parent_nodes, logits)?;
    reject_inf(&[&xs])?;
    let g_value = child_value(child, g)?;
    let gv = upstream_abs(&g_value)?;
    let w32 = denom_w as f32;
    if reduction == Reduction::Mean && (!w32.is_finite() || w32 == 0.0) {
        return Err(invalid("CrossEntropy: 重み和 W が f32 で表せないため拒否"));
    }
    // 子テープの `g.div(W)` は逆伝播で Div の微分係数 `1/W`（f32）を先に作る。W が非ゼロ有限でも
    // 極小（例 1e-40 の非正規化数）だと `1/W` が f32 で overflow し、`g/W` が段検査を通っても
    // 二階勾配が非有限になる。係数そのものを f32 と段検査の両方で拒否する（設計 §40.12 の総括判定）。
    if reduction == Reduction::Mean {
        let inv_w32 = 1.0f32 / w32;
        if !inv_w32.is_finite() {
            return Err(invalid(
                "CrossEntropy: 重み和 W の逆数が f32 範囲を超えるため拒否",
            ));
        }
        stage(1.0 / denom_w)?;
        stage(f64::from(inv_w32))?;
    }
    let sb = match reduction {
        Reduction::Mean => gv / denom_w,
        Reduction::Sum => gv,
    };
    stage(sb)?;
    // `g` へ流れる二階の係数（Mean は子テープの `g / W`、Sum は `g`）。
    let gscale_ce = match reduction {
        Reduction::Mean => 1.0 / denom_w,
        Reduction::Sum => 1.0,
    };

    let numel: usize = shape.iter().product();
    let mut a_data = vec![0f32; numel];
    let mut b_data = vec![0f32; numel];
    let mut valid = vec![false; numel];
    let mut gsum = 0f64;
    for o in 0..outer {
        for i in 0..inner {
            let t = target_data[o * inner + i];
            if ignore_index == Some(t) || !in_range(t) {
                continue;
            }
            let w_t = weights[t as usize];
            let a_s = (1.0 - eps) * w_t + (eps / axis_len as f64) * total_all;
            let mut bmax = 0f64;
            for (a, &w_a) in weights.iter().enumerate() {
                let idx = (o * axis_len + a) * inner + i;
                let delta = if a == t as usize { 1.0 } else { 0.0 };
                let b = (1.0 - eps) * w_t * delta + (eps / axis_len as f64) * w_a;
                a_data[idx] = a_s as f32;
                b_data[idx] = b as f32;
                valid[idx] = true;
                bmax = bmax.max(b.abs());
                gsum += a_s.abs() + b.abs();
            }
            stage(a_s)?;
            stage(bmax)?;
            stage(a_s + bmax)?;
            stage(sb * a_s)?;
            stage(sb * (a_s + bmax))?;
            cewo_cancel_check(
                &xs,
                (o, i, inner, axis_len),
                (a_s, &weights, t as usize, eps),
                (sb, gscale_ce),
            )?;
        }
    }
    let denom_for_g = if reduction == Reduction::Mean {
        denom_w
    } else {
        1.0
    };
    stage(gsum / denom_for_g)?;

    let host = crate::grad::cross_entropy_loss_with_options_vjp(
        materialize_fallible(parent_nodes, parent.ops(), logits)?,
        targets,
        class_dim,
        reduction,
        options,
        &g_value,
    );

    let s_b = match reduction {
        Reduction::Mean => g.div(&scalar(child, w32))?.broadcast_to(&shape)?,
        Reduction::Sum => g.broadcast_to(&shape)?,
    };
    let a_c = const_vec(child, a_data, &shape)?;
    let b_c = const_vec(child, b_data, &shape)?;
    // ignore / 範囲外の行は Softmax の前で有限定数へ置換し、その行への微分も遮断する
    // （NaN logits が 0 の cotangent と乗算されて二階へ漏れるのを防ぐ）。
    let valid_m = mask(valid, &shape)?;
    let zeros = zeros_like(child, &shape)?;
    let x_safe = Var::where_cond(&valid_m, x_m, &zeros)?;
    let sm = x_safe.softmax(class_dim)?;
    let scaled = s_b.mul(&a_c.mul(&sm)?.sub(&b_c)?)?;
    let dx = Var::where_cond(&valid_m, &scaled, &zeros)?;
    push_v3(child, &dx, host)
}

/// CEWO の 1 サンプル（`(o, i)` の class 軸）の相殺検査（設計 §55.2 の相殺の総括判定）。
///
/// 構造 `s_b⊙(A⊙softmax(x) − B)` の二階は、x 経路が Softmax の VJP `s_b·A·y_i(δ_ij − y_j)`。対角の
/// `1 − y_j` は確信度の高い行（`y_j → 1`）で相殺し、f32 の `y_j` が 1 に丸まると曲率が 0 になる
/// （真値は他クラスの確率の和）。非対角は単項の積で相殺しない。g 経路（`∂dx/∂g`）は `A·y_c − B_c`
/// で、target の `A·y_t ≈ B_t` が同じく相殺する。参照値は f64 の softmax で、`1 − y_j` は他クラスの
/// 和から桁落ちなく作る。NaN の行は伝播させるため検査しない。
fn cewo_cancel_check(
    xs: &[f32],
    pos: (usize, usize, usize, usize),
    coef: (f64, &[f64], usize, f64),
    scales: (f64, f64),
) -> Result<(), AutodiffError> {
    let (o, i, inner, c) = pos;
    let (a_s, weights, t, eps) = coef;
    let (sb, gscale) = scales;
    let row: Vec<f64> = (0..c)
        .map(|a| f64::from(xs[(o * c + a) * inner + i]))
        .collect();
    if row.iter().any(|v| v.is_nan()) {
        return Ok(());
    }
    let mx = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let e: Vec<f64> = row.iter().map(|&v| (v - mx).exp()).collect();
    let total: f64 = e.iter().sum();
    let total_w: f64 = weights.iter().sum();
    let smooth = eps / c as f64;
    let coeff = rounding_coeff(c);
    let sba = sb * a_s;
    let (mut top1, mut top2) = (0f64, 0f64);
    for j in 0..c {
        let y = e[j] / total;
        // `1 − y_j` は他クラスの和で求める（`y_j` が 1 に近い行で f64 の差も桁落ちさせない）。
        let rest = if 2.0 * e[j] > total {
            e.iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|(_, v)| *v)
                .sum::<f64>()
        } else {
            total - e[j]
        } / total;
        cancel_check(
            sba * y * rest,
            sba * y * (1.0 + y),
            1.0,
            coeff,
            "CrossEntropy: Softmax の対角の曲率",
        )?;
        // `A·y_j − B_j = (1−ε)·w_t·(y_j − δ_jt) + (ε/C)·(Σw·y_j − w_j)`（`y_t − 1 = −rest_t`）。
        let hard = if j == t { -rest } else { y };
        let v = (1.0 - eps) * weights[t] * hard + smooth * (total_w * y - weights[j]);
        let b = (1.0 - eps) * weights[t] * if j == t { 1.0 } else { 0.0 } + smooth * weights[j];
        cancel_check(
            v,
            a_s * y + b.abs(),
            gscale,
            coeff,
            "CrossEntropy: g への二階（A·softmax − B）",
        )?;
        if y > top1 {
            top2 = top1;
            top1 = y;
        } else if y > top2 {
            top2 = y;
        }
    }
    // 非対角 `−s_b·A·y_i·y_j` は単項で `m == |v|`。`coeff` が相対の半分を超える長い軸のときだけ
    // 絶対判定に落ちるので、最大の積（上位 2 クラス）で確かめる。
    cancel_check(
        sba * top1 * top2,
        sba * top1 * top2,
        1.0,
        coeff,
        "CrossEntropy: Softmax の非対角の曲率",
    )
}

// ---------------------------------------------------------------------------------
// MultiMarginLoss
// ---------------------------------------------------------------------------------

/// `MultiMarginLoss` の d_input を子テープへ記録する。
pub(super) fn multi_margin_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, Reduction),
    payload: (&[i32], Option<&[f32]>, u8, f32),
    x_m: &Var<'c>,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (input, reduction) = spec;
    let (targets, weight, p, margin) = payload;
    let shape = checked_shape(parent_nodes, &[input], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    if p != 1 && p != 2 {
        return Err(invalid("MultiMargin: 未知の p は fail-closed"));
    }
    let (n, c) = crate::eval::margin_focal_loss::rows_cols(&shape);
    if shape.len() > 2 || targets.len() != n || weight.is_some_and(|w| w.len() != c) {
        return Err(invalid(
            "MultiMargin: payload の shape 不整合（内部契約違反）",
        ));
    }
    let xs = dense(parent, parent_nodes, input)?;
    reject_inf(&[&xs])?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    let margin64 = f64::from(margin);
    let numel = n * c;
    let dim = shape.len() - 1;
    let gs = g_scale(reduction, n);
    let coeff = rounding_coeff(c);

    let mut gsum = 0f64;
    // p=1 の K（符号付き個数 × w_y/C）／p=2 の行定数 c_w とマスク M。
    let mut k_data = vec![0f32; numel];
    let mut cw_data = vec![0f32; numel];
    let mut m_data = vec![false; numel];
    for i in 0..n {
        let y = targets[i];
        if y < 0 || (y as usize) >= c {
            return Err(invalid("MultiMargin: target 添字が範囲外（内部契約違反）"));
        }
        let y = y as usize;
        let w = weight.map_or(1.0, |w| f64::from(w[y]));
        let cw = w / c as f64;
        let row = &xs[i * c..(i + 1) * c];
        let xy = f64::from(row[y]);
        let mut count = 0f64;
        let mut sum_h = 0f64;
        let mut sum_zm = 0f64;
        for (j, &xj) in row.iter().enumerate() {
            cw_data[i * c + j] = cw as f32;
            if j == y {
                continue;
            }
            let z = margin64 - xy + f64::from(xj);
            if z > 0.0 {
                m_data[i * c + j] = true;
                count += 1.0;
                if p == 2 {
                    // 子テープは Z を f32 の `(margin - x_y) + x_j` で再構成する。桁落ちでホストの f64 値と
                    // ずれると H=where(M, 2Z, 0) の二階が誤る（push_v3 は dx の値しか補正しない）ため拒否する。
                    let z32 = (margin - row[y]) + xj;
                    if (z - f64::from(z32)).abs() > z.abs() * 2f64.powi(-10) {
                        return Err(invalid(
                            "MultiMargin: マージン項の f32 再構成が桁落ちでホストの f64 値とずれるため拒否",
                        ));
                    }
                    let h = 2.0 * z;
                    stage(z)?;
                    stage(h)?;
                    // g 経路（`∂dx_j/∂g = c_w·H_j`）の相殺: `Z = (margin − x_y) + x_j` の差し引き。
                    let zm = margin64.abs() + xy.abs() + f64::from(xj).abs();
                    cancel_check(
                        cw * h,
                        2.0 * cw * zm,
                        gs,
                        coeff,
                        "MultiMargin: マージン項 Z の差し引き",
                    )?;
                    sum_zm += zm;
                    sum_h += h;
                    stage(sum_h)?;
                    stage(cw * h)?;
                    stage(sb * cw * h)?;
                    k_data[i * c + j] = 0.0;
                } else {
                    k_data[i * c + j] = cw as f32;
                }
            }
        }
        if p == 2 {
            // target レーンの `−c_w·ΣH` も同じ相殺を持つ（各 `Z > 0` の和なので和自体は相殺しない）。
            cancel_check(
                cw * sum_h,
                2.0 * cw * sum_zm,
                gs,
                coeff,
                "MultiMargin: マージン項 Z の差し引き（target）",
            )?;
            stage(cw * sum_h)?;
            stage(sb * cw * sum_h)?;
            stage(sb * cw * 2.0)?;
            stage(sb * cw)?;
            gsum += 2.0 * cw * sum_h;
        } else {
            k_data[i * c + y] = (-cw * count) as f32;
            let kmax = cw * count;
            stage(kmax)?;
            stage(sb * kmax)?;
            gsum += 2.0 * kmax;
        }
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let host = crate::eval::margin_focal_loss::multi_margin_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        targets,
        weight,
        p,
        margin,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let zeros = zeros_like(child, &shape)?;
    let dx = if p == 1 {
        let nz = mask(k_data.iter().map(|&k| k != 0.0).collect(), &shape)?;
        let k_c = const_vec(child, k_data, &shape)?;
        Var::where_cond(&nz, &s_b.mul(&k_c)?, &zeros)?
    } else {
        let idx_shape: Vec<usize> = if shape.len() == 1 {
            vec![1]
        } else {
            vec![n, 1]
        };
        let idx = Tensor::<i32>::new(targets.to_vec(), &idx_shape)?;
        let x_y = x_m.gather(dim, &idx)?.broadcast_to(&shape)?;
        let z = scalar(child, margin).sub(&x_y)?.add(x_m)?;
        let h = Var::where_cond(&mask(m_data, &shape)?, &z.mul(&scalar(child, 2.0))?, &zeros)?;
        let sum_h = reduce_dim_bcast(&h, dim, &shape)?;
        let mut onehot = vec![false; numel];
        for (i, &y) in targets.iter().enumerate() {
            onehot[i * c + y as usize] = true;
        }
        let inner = Var::where_cond(&mask(onehot, &shape)?, &sum_h.neg()?, &h)?;
        let cw_c = const_vec(child, cw_data, &shape)?;
        s_b.mul(&cw_c)?.mul(&inner)?
    };
    push_v3(child, &dx, host)
}

// ---------------------------------------------------------------------------------
// MultiLabelMarginLoss
// ---------------------------------------------------------------------------------

/// `MultiLabelMarginLoss` の d_input を子テープへ記録する（x について区分定数）。
pub(super) fn multilabel_margin_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, Reduction),
    targets: &[i32],
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (input, reduction) = spec;
    let shape = checked_shape(parent_nodes, &[input], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    let (n, c) = crate::eval::margin_focal_loss::rows_cols(&shape);
    if shape.len() > 2 || targets.len() != n * c {
        return Err(invalid(
            "MultiLabelMargin: payload の shape 不整合（内部契約違反）",
        ));
    }
    let xs = dense(parent, parent_nodes, input)?;
    reject_inf(&[&xs])?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;

    let unit = 1.0 / c as f64;
    let mut k_data = vec![0f32; n * c];
    let mut gsum = 0f64;
    let mut in_set = vec![false; c];
    let mut row_k = vec![0f64; c];
    for i in 0..n {
        let row = &xs[i * c..(i + 1) * c];
        let row_t = &targets[i * c..(i + 1) * c];
        let len = row_t.iter().position(|&t| t < 0).unwrap_or(c);
        in_set.iter_mut().for_each(|m| *m = false);
        for &t in &row_t[..len] {
            if (t as usize) >= c {
                return Err(invalid(
                    "MultiLabelMargin: target 添字が範囲外（内部契約違反）",
                ));
            }
            in_set[t as usize] = true;
        }
        row_k.iter_mut().for_each(|v| *v = 0.0);
        for &t in &row_t[..len] {
            let xt = f64::from(row[t as usize]);
            for (d, &xd) in row.iter().enumerate() {
                if in_set[d] {
                    continue;
                }
                if 1.0 - xt + f64::from(xd) > 0.0 {
                    row_k[t as usize] -= unit;
                    row_k[d] += unit;
                }
            }
        }
        for (j, &v) in row_k.iter().enumerate() {
            stage(sb * v.abs())?;
            gsum += v.abs();
            k_data[i * c + j] = v as f32;
        }
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let host = crate::eval::margin_focal_loss::multilabel_margin_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        targets,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let nz = mask(k_data.iter().map(|&k| k != 0.0).collect(), &shape)?;
    let k_c = const_vec(child, k_data, &shape)?;
    let dx = Var::where_cond(&nz, &s_b.mul(&k_c)?, &zeros_like(child, &shape)?)?;
    push_v3(child, &dx, host)
}

// ---------------------------------------------------------------------------------
// MultiLabelSoftMarginLoss
// ---------------------------------------------------------------------------------

/// `MultiLabelSoftMarginLoss` の d_input を子テープへ記録する。
pub(super) fn multilabel_soft_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, Reduction),
    payload: (&Tensor<f32>, Option<&[f32]>),
    x_m: &Var<'c>,
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (input, reduction) = spec;
    let (target, weight) = payload;
    let shape = checked_shape(parent_nodes, &[input], g)?;
    if shape.contains(&0) {
        return zeros_like(child, &shape);
    }
    let (n, c) = crate::eval::margin_focal_loss::rows_cols(&shape);
    if shape.len() > 2 || target.shape() != shape.as_slice() || weight.is_some_and(|w| w.len() != c)
    {
        return Err(invalid(
            "MultiLabelSoftMargin: payload の shape 不整合（内部契約違反）",
        ));
    }
    let xs = dense(parent, parent_nodes, input)?;
    let ts = crate::eval::dense_vec(target);
    reject_inf(&[&xs, &ts])?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    let gs = g_scale(reduction, n);

    let mut coef = vec![0f32; n * c];
    let mut gsum = 0f64;
    for i in 0..n {
        for j in 0..c {
            let wc = weight.map_or(1.0, |w| f64::from(w[j])) / c as f64;
            coef[i * c + j] = wc as f32;
            let (x, t) = (f64::from(xs[i * c + j]), f64::from(ts[i * c + j]));
            if x.is_nan() || t.is_nan() {
                continue;
            }
            let diff = (sigmoid64(x) - t).abs();
            stage(wc)?;
            stage(wc * diff)?;
            stage(sb * wc)?;
            stage(sb * wc * diff)?;
            // g 経路（`∂dx/∂g = (w_c/C)·(σ(x) − t)`）の相殺。構造は符号別の安定形
            // （x ≥ 0: `(1−t) − σ(−x)`、x < 0: `σ(x) − t`）で、σ ≈ t のとき差し引きが桁落ちする。
            // x 経路の曲率 `σ(1−σ)` は安定形で単項になり相殺しない。
            let mag = if x >= 0.0 {
                (1.0 - t).abs() + sigmoid64(-x)
            } else {
                sigmoid64(x) + t.abs()
            };
            cancel_check(
                wc * diff,
                wc.abs() * mag,
                gs,
                rounding_coeff(1),
                "MultiLabelSoftMargin: σ(x) − t の差し引き",
            )?;
            gsum += wc * diff;
        }
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let host = crate::eval::margin_focal_loss::multilabel_soft_margin_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), input)?,
        target,
        weight,
        scale,
    );
    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let t_c = const_vec(child, ts, &shape)?;
    let stable = sigma_pair_stable(child, x_m, &t_c, &xs, &shape)?;
    let coef_c = const_vec(child, coef, &shape)?;
    let dx = s_b.mul(&coef_c.mul(&stable)?)?;
    push_v3(child, &dx, host)
}

// ---------------------------------------------------------------------------------
// CosineEmbeddingLoss
// ---------------------------------------------------------------------------------

/// `(n, d)`（rank 1 は `(1, d)`）。rank 3 以上は拒否する。
fn rows_d(shape: &[usize]) -> Result<(usize, usize), AutodiffError> {
    match shape {
        [d] => Ok((1, *d)),
        [n, d] => Ok((*n, *d)),
        _ => Err(invalid("入力の rank が 1／2 以外（内部契約違反）")),
    }
}

/// `CosineEmbeddingLoss` の (d_x1, d_x2) を子テープへ記録する。
pub(super) fn cosine_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, Reduction),
    payload: (&Tensor<f32>, f32),
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    const EPSILON: f64 = 1e-12;
    let (x1, x2, reduction) = spec;
    let (y, margin) = payload;
    let (a_m, b_m) = mirrors;
    let shape = checked_shape(parent_nodes, &[x1, x2], g)?;
    if shape.contains(&0) {
        return Ok((zeros_like(child, &shape)?, zeros_like(child, &shape)?));
    }
    let (n, d) = rows_d(&shape)?;
    let ys = crate::eval::dense_vec(y);
    if ys.len() < n {
        return Err(invalid("Cosine: y の長さが不整合（内部契約違反）"));
    }
    let a_s = dense(parent, parent_nodes, x1)?;
    let b_s = dense(parent, parent_nodes, x2)?;
    reject_inf(&[&a_s, &b_s])?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    let gs = g_scale(reduction, n);
    let dim = shape.len() - 1;
    let numel = n * d;

    let sqrt_eps = EPSILON.sqrt();
    let mut r1 = vec![1f32; numel];
    let mut r2 = vec![1f32; numel];
    let mut e1 = vec![0f32; numel];
    let mut e2 = vec![0f32; numel];
    let mut coeff = vec![0f32; numel];
    let mut act = vec![false; numel];
    let mut gsum = 0f64;
    for (i, &y_v) in ys.iter().enumerate().take(n) {
        let base = i * d;
        let ar: Vec<f64> = a_s[base..base + d].iter().map(|&v| f64::from(v)).collect();
        let br: Vec<f64> = b_s[base..base + d].iter().map(|&v| f64::from(v)).collect();
        let has_nan = ar.iter().chain(&br).any(|v| v.is_nan());
        let max_a = ar.iter().fold(0f64, |m, v| m.max(v.abs()));
        let max_b = br.iter().fold(0f64, |m, v| m.max(v.abs()));
        let (s1, s2) = if has_nan {
            (1.0, 1.0)
        } else {
            (
                pow2_ceil(max_a.max(sqrt_eps)),
                pow2_ceil(max_b.max(sqrt_eps)),
            )
        };
        let (inv1, inv2) = (1.0 / s1, 1.0 / s2);
        let (eps1, eps2) = (EPSILON / (s1 * s1), EPSILON / (s2 * s2));
        for k in 0..d {
            r1[base + k] = inv1 as f32;
            r2[base + k] = inv2 as f32;
            e1[base + k] = eps1 as f32;
            e2[base + k] = eps2 as f32;
        }
        // 1 階 `cosine_embedding_loss_vjp` と同じ f64 の cos と係数判定（V2）。
        let (mut m1, mut m2, mut dot) = (EPSILON, EPSILON, 0f64);
        for k in 0..d {
            m1 += ar[k] * ar[k];
            m2 += br[k] * br[k];
            dot += ar[k] * br[k];
        }
        let denom = m1.sqrt() * m2.sqrt();
        let cos = dot / denom;
        let y_i = f64::from(y_v);
        let cf = if y_i > 0.0 {
            -1.0
        } else if cos - f64::from(margin) >= 0.0 {
            1.0
        } else {
            0.0
        };
        if cf == 0.0 {
            continue;
        }
        for k in 0..d {
            coeff[base + k] = cf as f32;
            act[base + k] = true;
        }
        if has_nan {
            continue;
        }
        // スケール後の量で各段を上から押さえる（`a' = a/s1`、`m1' = Σa'² + ε/s1²`）。
        let m1p: f64 = ar.iter().map(|v| (v * inv1) * (v * inv1)).sum::<f64>() + eps1;
        let m2p: f64 = br.iter().map(|v| (v * inv2) * (v * inv2)).sum::<f64>() + eps2;
        let denp = m1p.sqrt() * m2p.sqrt();
        let dotp: f64 = ar
            .iter()
            .zip(&br)
            .map(|(a, b)| (a * inv1) * (b * inv2))
            .sum();
        let cosp = dotp / denp;
        let (ma, mb) = (max_a * inv1, max_b * inv2);
        let b1 = mb / denp + cosp.abs() * ma / m1p;
        let b2 = ma / denp + cosp.abs() * mb / m2p;
        let kmax = 1.0 / denp + 3.0 / m1p.min(m2p);
        stage(1.0 / denp)?;
        stage(1.0 / m1p)?;
        stage(1.0 / m2p)?;
        stage(inv1 * b1)?;
        stage(inv2 * b2)?;
        for (r, b) in [(inv1, b1), (inv2, b2)] {
            stage(sb * r * b)?;
            stage(sb * r * kmax)?;
            stage(sb * r * r * kmax)?;
        }
        stage(sb * inv1 * inv2 * kmax)?;
        cosine_cancel_check(&ar, &br, (inv1, inv2), (eps1, eps2), (sb, gs))?;
        gsum += inv1 * b1 + inv2 * b2;
    }
    check_g_cotangent(gsum, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let (h1, h2) = crate::grad::cosine_embedding_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), x1)?,
        materialize_fallible(parent_nodes, parent.ops(), x2)?,
        y,
        margin,
        scale,
    );

    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let (r1_c, r2_c) = (const_vec(child, r1, &shape)?, const_vec(child, r2, &shape)?);
    let (e1_c, e2_c) = (const_vec(child, e1, &shape)?, const_vec(child, e2, &shape)?);
    let ap = a_m.mul(&r1_c)?;
    let bp = b_m.mul(&r2_c)?;
    let m1p = reduce_dim_bcast(&ap.mul(&ap)?, dim, &shape)?.add(&e1_c)?;
    let m2p = reduce_dim_bcast(&bp.mul(&bp)?, dim, &shape)?.add(&e2_c)?;
    let den = m1p.sqrt()?.mul(&m2p.sqrt()?)?;
    let cos = reduce_dim_bcast(&ap.mul(&bp)?, dim, &shape)?.div(&den)?;
    let inner1 = bp.div(&den)?.sub(&cos.mul(&ap)?.div(&m1p)?)?;
    let inner2 = ap.div(&den)?.sub(&cos.mul(&bp)?.div(&m2p)?)?;
    let coeff_c = const_vec(child, coeff, &shape)?;
    let act_m = mask(act, &shape)?;
    let zeros = zeros_like(child, &shape)?;
    let sc = s_b.mul(&coeff_c)?;
    let dx1 = Var::where_cond(&act_m, &sc.mul(&r1_c)?.mul(&inner1)?, &zeros)?;
    let dx2 = Var::where_cond(&act_m, &sc.mul(&r2_c)?.mul(&inner2)?, &zeros)?;
    Ok((push_v3(child, &dx1, h1)?, push_v3(child, &dx2, h2)?))
}

/// Cosine の 1 行の相殺検査（設計 §55.2 の相殺の総括判定。PR #3947 の Codex P1 への対応）。
///
/// 構造 `inner1 = b'/den − cos·a'/m1'`（`dx2` は a と b を入れ替えた形）を、seed が 1 要素
/// （Hessian の 1 列）のときの二階の逆伝播の演算順どおりに f64 で展開し、各要素の参照値 `V` と
/// 各項の絶対値の和 `M` を求めて [`cancel_check`] に掛ける。展開（own は自分側のベクトル、other は
/// 相手側、`j` は seed の要素）:
/// - `cot_cos = −own_j/m_own`、`cot_D = cot_cos/den`、`cot_den = −other_j/den² + own_j·cos/(m_own·den)`
/// - `cot_m_own = cos·own_j/m_own² + cot_den·s_other/(2 s_own)`、`cot_m_other = cot_den·s_own/(2 s_other)`
/// - own への二階 `δ_ij·(−cos/m_own) + other_i·cot_D + 2 own_i·cot_m_own`
/// - other への二階 `δ_ij/den + own_i·cot_D + 2 other_i·cot_m_other`
/// - g への二階は `inner_j` そのもの（`other_j/den − cos·own_j/m_own`）
///
/// `M` では `cos` を `Σ|a'b'|/den`（内積の相殺も含める）に置き換える。平行に近い行（例 x1=[1]、x2=[1]）
/// では各項が O(1) で打ち消し合い、真の曲率は `ε'` の大きさ（`3ε/(1+ε)^3`）しか残らない。f32 では
/// `m' = Σa'² + ε'` の `ε'` が丸めで消えるため構造は 0 を返し、上流係数が大きいと REQ-2 を外れる。
/// この ε の吸収は `M` に比べて小さい `V` として同じ式で検出される（構造で ε を保つ方法は無い:
/// `ε'/m' < 2^-24` の行では f32 の `m'` に ε の情報が残らない）。
/// 係数は行の幅 `d` から [`rounding_coeff`]。上界 `coeff·M·scale` が O(d) の速い経路で絶対判定を
/// 満たす行は要素ごとの O(d²) の走査を省く。
fn cosine_cancel_check(
    ar: &[f64],
    br: &[f64],
    inv: (f64, f64),
    eps: (f64, f64),
    scales: (f64, f64),
) -> Result<(), AutodiffError> {
    let d = ar.len();
    let coeff = rounding_coeff(d);
    let (sb, gs) = scales;
    let ap: Vec<f64> = ar.iter().map(|v| v * inv.0).collect();
    let bp: Vec<f64> = br.iter().map(|v| v * inv.1).collect();
    let m1: f64 = ap.iter().map(|v| v * v).sum::<f64>() + eps.0;
    let m2: f64 = bp.iter().map(|v| v * v).sum::<f64>() + eps.1;
    let (s1, s2) = (m1.sqrt(), m2.sqrt());
    let den = s1 * s2;
    let dot: f64 = ap.iter().zip(&bp).map(|(a, b)| a * b).sum();
    let cos = dot / den;
    let cos_m: f64 = ap.iter().zip(&bp).map(|(a, b)| (a * b).abs()).sum::<f64>() / den;
    let amax = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
    // (own, other, m_own, s_own, s_other, r_own, r_other)
    let sides = [
        (&ap, &bp, m1, s1, s2, inv.0, inv.1),
        (&bp, &ap, m2, s2, s1, inv.1, inv.0),
    ];
    for (own, other, m_own, s_own, s_other, r_own, r_other) in sides {
        let (omax, tmax) = (amax(own), amax(other));
        // seed 要素 j ごとの cotangent（値と大きさ）。
        let per_j: Vec<[f64; 6]> = (0..d)
            .map(|j| {
                let (oj, tj) = (own[j], other[j]);
                let cot_d = -oj / (m_own * den);
                let cot_den = -tj / (den * den) + oj * cos / (m_own * den);
                let cot_mo = cos * oj / (m_own * m_own) + cot_den * s_other * 0.5 / s_own;
                let cot_mt = cot_den * s_own * 0.5 / s_other;
                let cot_d_m = oj.abs() / (m_own * den);
                let cot_den_m = tj.abs() / (den * den) + oj.abs() * cos_m / (m_own * den);
                let cot_mo_m =
                    cos_m * oj.abs() / (m_own * m_own) + cot_den_m * s_other * 0.5 / s_own;
                let cot_mt_m = cot_den_m * s_own * 0.5 / s_other;
                [cot_d, cot_mo, cot_mt, cot_d_m, cot_mo_m, cot_mt_m]
            })
            .collect();
        let max_of = |k: usize| per_j.iter().fold(0f64, |m, c| m.max(c[k]));
        let s_own_blk = sb * r_own * r_own;
        let s_cross_blk = sb * r_own * r_other;
        let fast_own = cos_m / m_own + tmax * max_of(3) + 2.0 * omax * max_of(4);
        let fast_cross = 1.0 / den + omax * max_of(3) + 2.0 * tmax * max_of(5);
        let need_own = !fast_ok(coeff * fast_own * s_own_blk);
        let need_cross = !fast_ok(coeff * fast_cross * s_cross_blk);
        for (j, c) in per_j.iter().enumerate() {
            let [cot_d, cot_mo, cot_mt, cot_d_m, cot_mo_m, cot_mt_m] = *c;
            // g への二階（`inner_j`）。
            cancel_check(
                other[j] / den - cos * own[j] / m_own,
                other[j].abs() / den + cos_m * own[j].abs() / m_own,
                gs * r_own,
                coeff,
                "Cosine: g への二階（b'/den − cos·a'/m' の差し引き）",
            )?;
            if !(need_own || need_cross) {
                continue;
            }
            for i in 0..d {
                let delta = if i == j { 1.0 } else { 0.0 };
                if need_own {
                    cancel_check(
                        delta * (-cos / m_own) + other[i] * cot_d + 2.0 * own[i] * cot_mo,
                        delta * cos_m / m_own
                            + other[i].abs() * cot_d_m
                            + 2.0 * own[i].abs() * cot_mo_m,
                        s_own_blk,
                        coeff,
                        "Cosine: 同じ側の二階（ε の吸収・平行に近い行の相殺）",
                    )?;
                }
                if need_cross {
                    cancel_check(
                        delta / den + own[i] * cot_d + 2.0 * other[i] * cot_mt,
                        delta / den + own[i].abs() * cot_d_m + 2.0 * other[i].abs() * cot_mt_m,
                        s_cross_blk,
                        coeff,
                        "Cosine: 相手側の二階（ε の吸収・平行に近い行の相殺）",
                    )?;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------
// TripletMarginLoss
// ---------------------------------------------------------------------------------

/// 差分ベクトル（行ごと）から作る p ノルム勾配の構造用定数（V2）。
struct PairInfo {
    inv_mx: Vec<f32>,
    sign: Vec<f32>,
    zero: Vec<bool>,
}

/// `rows`（ホスト f64 の差分）と `rec`（子テープが f32 の `Sub`→`Add` で再構成する差分）が構造として
/// 一致すること、および逆スケール `sb·inv_mx·(p+1)·d`（`pair_grad` の二階逆伝播で `v` へ流れる
/// cotangent の上界。設計 §40.12 の段別 cotangent×微分係数検査）が f32 範囲に収まることを検査する。
fn pair_info(
    rows: &[Vec<f64>],
    rec: &[Vec<f32>],
    d: usize,
    p: f64,
    sb: f64,
) -> Result<PairInfo, AutodiffError> {
    let mut inv_mx = vec![1f32; rows.len() * d];
    let mut sign = vec![0f32; rows.len() * d];
    let mut zero = vec![false; rows.len() * d];
    for (i, v) in rows.iter().enumerate() {
        let base = i * d;
        let has_nan = v.iter().any(|x| x.is_nan());
        for (k, &x) in v.iter().enumerate() {
            sign[base + k] = sign0(x) as f32;
        }
        if has_nan {
            continue;
        }
        let mx = v.iter().fold(0f64, |m, x| m.max(x.abs()));
        if p > 1.0 && p < 2.0 && v.contains(&0.0) {
            return Err(invalid(
                "Triplet: 1<p<2 で差分が 0 のレーンは二階が発散するため拒否",
            ));
        }
        // 子テープの f32 再構成が桁落ちでホストの f64 差分とずれる入力（0/0 や 1e-8 → 0）は拒否する。
        let r = &rec[i];
        if v.iter().zip(r.iter()).any(|(&h, &c)| {
            (h == 0.0) != (c == 0.0) || (h - f64::from(c)).abs() > mx * 2f64.powi(-10)
        }) {
            return Err(invalid(
                "Triplet: 差分の f32 再構成が桁落ちでホストの f64 差分とずれるため拒否",
            ));
        }
        if mx == 0.0 {
            for k in 0..d {
                zero[base + k] = true;
            }
            continue;
        }
        if mx < 2f64.powi(-100) {
            return Err(invalid("Triplet: 差分のスケールが小さすぎるため拒否"));
        }
        stage(mx)?;
        let inv = (1.0 / mx) as f32;
        if p != 1.0 {
            // 子テープの f32 構造（`u = v·inv`、`|u|^p` の和、`norm = ssum^(1/p)`）を、f32 に丸めた
            // `u` から f64 で再現して検査する（設計 §40.12）。`u` の最大成分は丸めで 1 を僅かに下回りうるため、
            // 巨大な p では `|u|^p` が全成分で 0 に潰れ、norm が 0 になって Div に非有限値が入る。
            let us: Vec<f64> = r.iter().map(|&c| f64::from((c * inv).abs())).collect();
            let ssum: f64 = us.iter().map(|u| u.powf(p)).sum();
            if !ssum.is_finite() || ssum < 2f64.powi(-100) {
                return Err(invalid(
                    "Triplet: f32 丸め後の |u|^p の和が下限を割る（norm が 0 へ潰れる）ため拒否",
                ));
            }
            // 上限側: f64 で有限でも、子テープの f32 `PowScalar`（`|u|^p` とその微分係数 `p·|u|^(p-1)`）と
            // 和 `ssum` が f32 範囲を超えると inf になり、逆伝播で 0×inf の NaN になる（Codex P1）。
            // 各項・和・微分係数を `stage` で検査する（設計 §40.12 の中間値・二階逆伝播の総括判定）。
            stage(ssum)?;
            for &u in &us {
                stage(u.powf(p))?;
                stage(p * u.powf(p - 1.0))?;
            }
            let norm = ssum.powf(1.0 / p);
            // 子テープの `norm` は f32 で丸められる。`(|u|/norm)^(p-1)` は norm の相対誤差を (p-1) 倍に
            // 増幅するため、大きな p で norm が 1 へ潰れる（例 p=1e8, d=2 で 2^(1/p)→1）入力は
            // 二階の曲率が解析値から外れる。f32 再現値との相対差 × (p-1) を検査して拒否する（Codex P1）。
            let norm32 = f64::from((ssum as f32).powf((1.0 / p) as f32));
            if p > 1.0 && (p - 1.0) * (norm32 - norm).abs() / norm > 2f64.powi(-10) {
                return Err(invalid(
                    "Triplet: p が大きく norm の f32 丸めが二階の曲率を損なうため拒否",
                ));
            }
            stage(1.0 / ssum)?;
            stage(1.0 / norm)?;
            if p > 1.0 && p < 2.0 {
                // `pair_grad` は `(|u|/norm)^(p-1)` を組むため、二階の係数は (p-1)·(|u|/norm)^(p-2)/norm
                // （norm ≥ 1 のとき `|u|^(p-2)` より大きい）。実際の評価式どおりに cotangent と積で検査する。
                for &u in &us {
                    let c = (p - 1.0) * (u / norm).powf(p - 2.0) / norm;
                    stage(c)?;
                    stage(sb * c * f64::from(inv))?;
                }
            }
            stage(sb * f64::from(inv) * (p + 1.0) * d as f64 / ssum.min(1.0))?;
        }
        for k in 0..d {
            inv_mx[base + k] = inv;
        }
    }
    Ok(PairInfo { inv_mx, sign, zero })
}

/// Triplet の 1 ペア・1 行について、二階の展開に使う f64 の量（`pair_grad` の構造に対応）。
/// `u = v·inv`、`N = ‖u‖_p`、`q = |u|/N`、`e = q^(p−1)`、対角係数 `dg = (p−1)·q^(p−2)`、
/// `fac = inv/N`、`sg = sign(u)`。全要素 0 の行（構造は定数 0）・NaN の行は `None`。
struct PairRow {
    fac: f64,
    pm1: f64,
    dg: Vec<f64>,
    e: Vec<f64>,
    sg: Vec<f64>,
}

fn triplet_pair_row(h: &[f64], info: &PairInfo, base: usize, p: f64) -> Option<PairRow> {
    if info.zero[base] || h.iter().any(|v| v.is_nan()) {
        return None;
    }
    let inv = f64::from(info.inv_mx[base]);
    let u: Vec<f64> = h.iter().map(|v| v * inv).collect();
    let norm = if p == 2.0 {
        u.iter().map(|x| x * x).sum::<f64>().sqrt()
    } else {
        u.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
    };
    if norm.is_nan() || norm <= 0.0 {
        return None;
    }
    let q: Vec<f64> = u.iter().map(|x| x.abs() / norm).collect();
    Some(PairRow {
        fac: inv / norm,
        pm1: p - 1.0,
        dg: q.iter().map(|&x| (p - 1.0) * x.powf(p - 2.0)).collect(),
        e: q.iter().map(|&x| x.powf(p - 1.0)).collect(),
        sg: u.iter().map(|&x| sign0(x)).collect(),
    })
}

/// Triplet の 1 行の相殺検査（設計 §55.2 の相殺の総括判定）。
///
/// 1 ペアの p ノルム勾配 `core_j = sign(u_j)·(|u_j|/N)^(p−1)` の二階は、v のレーン i へ
/// `fac·(δ_ij·dg_j − (p−1)·sg_i·sg_j·e_i·e_j)` で、対角は 1 成分が支配的な行（`q_j → 1`）で相殺する
/// （p=2 では `(1 − û_j²)/N`）。さらに `da` の a への二階は `H_ap − c_an·H_an`、`dp` の p への二階は
/// `H_ap − c_pn·H_pn` と 2 ペアの差になり、非対角でも相殺しうる。9 ブロック（出力 da／dp／dn ×
/// 入力 a／p／n）の各要素と、g への二階（`Σ w·core`）を [`cancel_check`] に掛ける。`coeff` は幅 `d` と
/// `p`、差分の再構成の条件数 `κ`（`(|x − y| + |eps|)/|v|`。`v ≈ 0` の桁落ちで `u` の相対誤差が増える）
/// から呼び出し元が作る。O(d) の速い経路で絶対判定を満たすブロックは O(d²) の走査を省く。
fn triplet_cancel_check(
    pr: &[Option<PairRow>; 3],
    coefs: (f64, f64),
    d: usize,
    scales: (f64, f64),
    coeff: f64,
) -> Result<(), AutodiffError> {
    let (c_an, c_pn) = coefs;
    let (sb, gs) = scales;
    // ペア添字 0 = ap、1 = an、2 = pn。重み w = (出力の係数)·(∂v/∂入力)。
    let outputs: [[(usize, f64); 2]; 3] = [
        [(0, 1.0), (1, -c_an)],
        [(0, -1.0), (2, -c_pn)],
        [(1, c_an), (2, c_pn)],
    ];
    let blocks: [[(usize, f64); 2]; 9] = [
        [(0, 1.0), (1, -c_an)],
        [(0, -1.0), (0, 0.0)],
        [(1, c_an), (1, 0.0)],
        [(0, -1.0), (0, 0.0)],
        [(0, 1.0), (2, -c_pn)],
        [(2, c_pn), (2, 0.0)],
        [(1, c_an), (1, 0.0)],
        [(2, c_pn), (2, 0.0)],
        [(1, -c_an), (2, -c_pn)],
    ];
    let live = |terms: &[(usize, f64); 2]| -> Vec<(&PairRow, f64)> {
        terms
            .iter()
            .filter(|&&(_, w)| w != 0.0)
            .filter_map(|&(k, w)| pr[k].as_ref().map(|r| (r, w)))
            .collect()
    };
    for terms in &outputs {
        let ts = live(terms);
        for j in 0..d {
            let v: f64 = ts.iter().map(|(r, w)| w * r.sg[j] * r.e[j]).sum();
            let m: f64 = ts.iter().map(|(r, w)| w.abs() * r.e[j]).sum();
            cancel_check(v, m, gs, coeff, "Triplet: g への二階（ペア勾配の差し引き）")?;
        }
    }
    for terms in &blocks {
        let ts = live(terms);
        if ts.is_empty() {
            continue;
        }
        let fast: f64 = ts
            .iter()
            .map(|(r, w)| {
                let dmax = r.dg.iter().fold(0f64, |m, x| m.max(x.abs()));
                let emax = r.e.iter().fold(0f64, |m, &x| m.max(x));
                w.abs() * r.fac * (dmax + r.pm1.abs() * emax * emax)
            })
            .sum();
        if fast_ok(coeff * fast * sb) {
            continue;
        }
        for j in 0..d {
            for i in 0..d {
                let delta = if i == j { 1.0 } else { 0.0 };
                let (mut v, mut m) = (0f64, 0f64);
                for (r, w) in &ts {
                    let off = r.pm1 * r.sg[i] * r.sg[j] * r.e[i] * r.e[j];
                    v += w * r.fac * (delta * r.dg[j] - off);
                    m += w.abs() * r.fac * (delta * r.dg[j].abs() + off.abs());
                }
                cancel_check(
                    v,
                    m,
                    sb,
                    coeff,
                    "Triplet: 二階（支配的な成分・2 ペアの差の相殺）",
                )?;
            }
        }
    }
    Ok(())
}

/// 差分 `v`（子テープ上の `Var`）から p ノルムの勾配 `∂‖v‖_p/∂v` の構造を組む。
fn pair_grad<'c>(
    child: &'c Tape,
    v: &Var<'c>,
    info: &PairInfo,
    shape: &[usize],
    p: f32,
) -> Result<Var<'c>, AutodiffError> {
    let sign_c = const_vec(child, info.sign.clone(), shape)?;
    if p == 1.0 {
        return Ok(sign_c);
    }
    let dim = shape.len() - 1;
    let zero_m = mask(info.zero.clone(), shape)?;
    let zeros = zeros_like(child, shape)?;
    let ones = full_like(child, shape, 1.0)?;
    let u = v.mul(&const_vec(child, info.inv_mx.clone(), shape)?)?;
    let core = if p == 2.0 {
        let ssum = Var::where_cond(&zero_m, &ones, &reduce_dim_bcast(&u.mul(&u)?, dim, shape)?)?;
        u.div(&ssum.sqrt()?)?
    } else {
        let au = u.abs()?;
        let ssum = Var::where_cond(
            &zero_m,
            &ones,
            &reduce_dim_bcast(&au.pow_scalar(p)?, dim, shape)?,
        )?;
        let norm = ssum.pow_scalar(1.0 / p)?;
        sign_c.mul(&au.div(&norm)?.pow_scalar(p - 1.0)?)?
    };
    Var::where_cond(&zero_m, &zeros, &core)
}

/// `TripletMarginLoss` の (d_anchor, d_positive, d_negative) を子テープへ記録する。
pub(super) fn triplet_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, NodeId, Reduction),
    options: &TripletMarginOptions,
    mirrors: (&Var<'c>, &Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<(Var<'c>, Var<'c>, Var<'c>), AutodiffError> {
    let (anchor, positive, negative, reduction) = spec;
    let (a_m, p_m, n_m) = mirrors;
    let shape = checked_shape(parent_nodes, &[anchor, positive, negative], g)?;
    if shape.contains(&0) {
        return Ok((
            zeros_like(child, &shape)?,
            zeros_like(child, &shape)?,
            zeros_like(child, &shape)?,
        ));
    }
    let (n, d) = rows_d(&shape)?;
    let numel = n * d;
    let a_s = dense(parent, parent_nodes, anchor)?;
    let p_s = dense(parent, parent_nodes, positive)?;
    let n_s = dense(parent, parent_nodes, negative)?;
    reject_inf(&[&a_s, &p_s, &n_s])?;
    let g_value = child_value(child, g)?;
    let sb = upstream_scale(&g_value, reduction, n)?;
    let p = options.p_value();
    let p64 = f64::from(p);
    let eps = f64::from(options.eps_value());
    let margin = f64::from(options.margin_value());
    let swap = options.swap_value();
    stage(p64)?;
    stage(sb * p64 * p64)?;

    let to64 = |s: &[f32]| -> Vec<f64> { s.iter().map(|&v| f64::from(v)).collect() };
    let (a64, p64v, n64) = (to64(&a_s), to64(&p_s), to64(&n_s));
    let params = crate::eval::TripletDistanceParams { p: p64, eps, swap };
    let mut rows_ap = Vec::with_capacity(n);
    let mut rows_an = Vec::with_capacity(n);
    let mut rows_pn = Vec::with_capacity(n);
    let (mut rec_ap, mut rec_an, mut rec_pn) = (Vec::new(), Vec::new(), Vec::new());
    let eps32 = options.eps_value();
    let mut act = vec![false; numel];
    let (mut m_one, mut m_zero, mut m_half) =
        (vec![false; numel], vec![false; numel], vec![false; numel]);
    let (mut act_rows, mut c_an_rows, mut kap_rows) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        let base = i * d;
        let (stats, diff_ap, diff_an, diff_pn) =
            crate::eval::triplet_distance_stats(&a64, &p64v, &n64, base, d, &params);
        let active = stats.d_ap - stats.neg_distance() + margin >= 0.0;
        act_rows.push(active);
        c_an_rows.push(if swap { stats.an_coeff } else { 1.0 });
        let kappa = |x: &[f32], y: &[f32], h: &[f64]| -> f64 {
            (0..d).fold(1f64, |m, k| {
                let num = (f64::from(x[base + k]) - f64::from(y[base + k])).abs() + eps.abs();
                if h[k] == 0.0 {
                    m
                } else {
                    m.max(num / h[k].abs())
                }
            })
        };
        kap_rows.push(
            kappa(&a_s, &p_s, &diff_ap)
                .max(kappa(&a_s, &n_s, &diff_an))
                .max(if swap {
                    kappa(&p_s, &n_s, &diff_pn)
                } else {
                    1.0
                }),
        );
        for k in 0..d {
            act[base + k] = active;
            if stats.an_coeff >= 1.0 {
                m_one[base + k] = true;
            } else if stats.an_coeff <= 0.0 {
                m_zero[base + k] = true;
            } else {
                m_half[base + k] = true;
            }
        }
        let row32 = |x: &[f32], y: &[f32]| -> Vec<f32> {
            (0..d)
                .map(|k| (x[base + k] - y[base + k]) + eps32)
                .collect()
        };
        rec_ap.push(row32(&a_s, &p_s));
        rec_an.push(row32(&a_s, &n_s));
        rec_pn.push(row32(&p_s, &n_s));
        rows_ap.push(diff_ap);
        rows_an.push(diff_an);
        rows_pn.push(diff_pn);
    }
    let info_ap = pair_info(&rows_ap, &rec_ap, d, p64, sb)?;
    let info_an = pair_info(&rows_an, &rec_an, d, p64, sb)?;
    let info_pn = if swap {
        Some(pair_info(&rows_pn, &rec_pn, d, p64, sb)?)
    } else {
        None
    };
    if p64 != 1.0 {
        let gs = g_scale(reduction, n);
        for i in 0..n {
            if !act_rows[i] {
                continue;
            }
            let row =
                |rows: &[Vec<f64>], info: &PairInfo| triplet_pair_row(&rows[i], info, i * d, p64);
            let pr = [
                row(&rows_ap, &info_ap),
                row(&rows_an, &info_an),
                info_pn.as_ref().and_then(|info| row(&rows_pn, info)),
            ];
            let nan = [&rows_ap[i], &rows_an[i]]
                .into_iter()
                .chain(swap.then_some(&rows_pn[i]))
                .any(|r| r.iter().any(|v| v.is_nan()));
            if nan {
                continue;
            }
            let c_an = c_an_rows[i];
            let c_pn = if swap { 1.0 - c_an } else { 0.0 };
            let coeff = rounding_coeff(d)
                + 2.0 * ((p64 - 1.0).abs() + 1.0) * (kap_rows[i] + 1.0) * (-23f64).exp2();
            triplet_cancel_check(&pr, (c_an, c_pn), d, (sb, gs), coeff)?;
        }
    }
    // 各勾配ベクトルの成分は |u|/norm' ≤ 1。3 本の寄与を足した上界で `g` への cotangent を押さえる。
    check_g_cotangent(3.0 * numel as f64, reduction, n)?;

    let scale = crate::grad::elementwise_loss_scale(&g_value, reduction, n);
    let (ha, hp, hn) = crate::grad::triplet_margin_loss_vjp(
        materialize_fallible(parent_nodes, parent.ops(), anchor)?,
        materialize_fallible(parent_nodes, parent.ops(), positive)?,
        materialize_fallible(parent_nodes, parent.ops(), negative)?,
        options,
        scale,
    );

    let s_b = super::loss_scale_bcast(child, g, reduction, false, n, &shape)?;
    let eps_c = scalar(child, options.eps_value());
    let zeros = zeros_like(child, &shape)?;
    let v_ap = a_m.sub(p_m)?.add(&eps_c)?;
    let v_an = a_m.sub(n_m)?.add(&eps_c)?;
    let g_ap = pair_grad(child, &v_ap, &info_ap, &shape, p)?;
    let g_an = pair_grad(child, &v_an, &info_an, &shape, p)?;
    let (ean, epn) = if let Some(info) = &info_pn {
        let v_pn = p_m.sub(n_m)?.add(&eps_c)?;
        let g_pn = pair_grad(child, &v_pn, info, &shape, p)?;
        let half = scalar(child, 0.5);
        let (m_one, m_zero, m_half) = (
            mask(m_one, &shape)?,
            mask(m_zero, &shape)?,
            mask(m_half, &shape)?,
        );
        let ean = Var::where_cond(
            &m_one,
            &g_an,
            &Var::where_cond(&m_half, &g_an.mul(&half)?, &zeros)?,
        )?;
        let epn = Var::where_cond(
            &m_zero,
            &g_pn,
            &Var::where_cond(&m_half, &g_pn.mul(&half)?, &zeros)?,
        )?;
        (ean, Some(epn))
    } else {
        (g_an, None)
    };
    let (da, dp, dn) = match &epn {
        Some(epn) => (g_ap.sub(&ean)?, g_ap.neg()?.sub(epn)?, ean.add(epn)?),
        None => (g_ap.sub(&ean)?, g_ap.neg()?, ean),
    };
    let act_m = mask(act, &shape)?;
    let wrap = |v: &Var<'c>| -> Result<Var<'c>, AutodiffError> {
        Var::where_cond(&act_m, &s_b.mul(v)?, &zeros)
    };
    Ok((
        push_v3(child, &wrap(&da)?, ha)?,
        push_v3(child, &wrap(&dp)?, hp)?,
        push_v3(child, &wrap(&dn)?, hn)?,
    ))
}
