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
//!   CE・CEWO の相殺判定の forward は総称関数 `cewo_fwd` で組み、従来の見積りが拒否した要素だけ `Lin` の
//!   上界で救う（#4093 で並行計算、#4094 で切り替え。設計 §82.10・§82.11）。
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
//! （[`struct_check`]）。構造が f32 で計算する中間値（スケーリング・正規化・累乗・softmax・σ・定数・
//! `s_b`）は f32 で再現し、そこから組んだ二階と参照の差も見積りに入れる。このため、f64 では非ゼロの
//! 値が f32 で 0 や非正規数へ underflow して曲率が消える入力も同じ判定で拒否される（例: Triplet の
//! 微小成分が `v·(1/mx)` で 0 になる）。対象は Cosine・Triplet の x／g 経路、CEWO の Softmax と
//! g 経路、MultiMargin・MultiLabelMargin・MultiLabelSoft の定数と g 経路、MultiLabelSoft の σ の
//! 曲率である。保証はこの判定を通った入力に限る。
//!
//! Cosine・CE・CEWO・Triplet は、この見積りが拒否した要素に限り `Lin`（丸めの一次伝播）の上界で再評価して救う
//! （Cosine は #4091・設計 §82.8、CE・CEWO は #4094・設計 §82.11、Triplet は #4096・設計 §82.13）。
//! 救済は片方の見積りが通れば受理する形なので拒否範囲は従来以下で、走査量が大きいテンソルは従来のまま。
//!
//! 既定オプションの `CrossEntropyLoss` も同じ CEWO の相殺判定を `A = 1`・`B = onehot`・`W = n` で通す
//! （[`ce_cancel_check`]。#3954・設計 §77）。
//!
//! GPU: 6 つとも 1 階はホスト関数（経路区分 H）。実機 parity は未実測（Phase 10 の #3857・#3859）。

use fandhe_ai_tensor_core::Tensor;

use super::lin::{
    LIN_BUDGET, LIN_U, Lin, LinMark, Num, entry_check_lin, lin_mark, lin_reset, lin_truncate,
};
use super::{child_value, judge, push_v3, reduce_dim_bcast};
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
    judge::stage(v, invalid)
}

/// 相殺（桁落ち）検査の丸め誤差係数 `(2·terms + 4)·2^-23`（`linalg_fft.rs::rounding_ok` と同じ形）。
/// `terms` は構造の最長の縮約長（Cosine・Triplet は行の幅 `d`、CEWO は Softmax の軸長 `C` 等）。
/// 縮約が f32 の逐次和でも成り立つ最悪値として長さに比例させる。
fn rounding_coeff(terms: usize) -> f64 {
    judge::rounding_coeff(terms)
}

/// f32 構造の 1 要素が REQ-2 の統一複合判定で f64 の参照に一致するかの保守的検査（設計 §55.2 の
/// 「相殺の総括判定」）。`v` は同じ要素の f64 の参照値、`v_str` は構造が f32 で計算する中間値
/// （スケーリング・正規化・累乗・softmax・σ・定数・`s_b` を f32 で再現した値）から同じ式で組んだ値、
/// `m` は同じ式を各項の絶対値の和で評価した大きさ（相殺が無ければ `|v|` と等しい）、`scale` は要素に
/// 掛かる上流係数（`|g|`・`1/n`・`1/W`・行スケール `1/s` 等の積）。誤差の見積り
/// `(|v_str − v| + coeff·m)·scale` が絶対 1e-5 未満か、相対の半分（`linalg_fft.rs` と同じ保守的な閾値。
/// 許容自体は不変）以内でなければ拒否する。
/// - ε の吸収・差し引きの桁落ちは「`m` に比べて小さい `v`」として `coeff·m` の項で見える。
/// - f32 の中間値の underflow（非ゼロが 0 や非正規数になる）は `|v_str − v|` の項で見える
///   （例: Triplet の微小成分が `v·(1/mx)` で 0 になり、その成分の曲率が 0 になる）。
///
/// 参照値 `v` が NaN のレーンは伝播の契約に従い素通しにする（`v_str` だけが非有限なら拒否）。
fn struct_check(
    v: f64,
    v_str: f64,
    m: f64,
    scale: f64,
    coeff: f64,
    what: &str,
) -> Result<(), AutodiffError> {
    let r = judge::struct_check(v, v_str, m, scale, coeff, what, invalid);
    // 計測用の記録（`#[cfg(test)]` 限定。判定の戻り値は変えず、そのまま返す）。
    #[cfg(test)]
    probe::check(what, [v, v_str, m, scale, coeff], r.is_ok());
    r
}

/// 拒否の内訳の計測（イシュー #4089・設計 §82）用の記録フック。`#[cfg(test)]` のみでコンパイルされ、
/// 本番ビルドには一切含まれない。[`struct_check`] の各呼び出しの入力と合否、および `deviates` が
/// 真になった行（`flag`。f32 再現値で組む経路）を、テストが開始した記録（`loss_class_tests.rs` の
/// `Probe`）にだけ積む。記録の有無で判定・戻り値・エラー文面は変わらない（迂回経路ではない）。
#[cfg(test)]
pub(super) mod probe {
    use std::cell::RefCell;

    /// 記録 1 件。`Check` は `struct_check` の 1 呼び出し、`Flag` は行の `flag`（族名つき）。
    #[derive(Clone, Debug)]
    pub(in crate::create_graph) enum Event {
        Check {
            what: String,
            /// `[v, v_str, m, scale, coeff]`
            vals: [f64; 5],
            ok: bool,
        },
        Flag(&'static str, bool),
    }

    thread_local! {
        pub(in crate::create_graph) static LOG: RefCell<Option<Vec<Event>>> =
            const { RefCell::new(None) };
        /// `true` の間は Cosine・CE・CEWO の判定が `Lin` の上界を使わない（旧判定との比較・変異確認用。#4091・#4094）。
        pub(in crate::create_graph) static FORCE_OLD: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    pub(super) fn check(what: &str, vals: [f64; 5], ok: bool) {
        // 合格の呼び出しは数が膨大（O(d^2)・O(C^2)）なので積まず、拒否した呼び出しだけを積む。
        if ok {
            return;
        }
        LOG.with(|l| {
            if let Some(log) = l.borrow_mut().as_mut() {
                log.push(Event::Check {
                    what: what.to_string(),
                    vals,
                    ok,
                });
            }
        });
    }

    pub(super) fn flag(family: &'static str, f: bool) {
        LOG.with(|l| {
            if let Some(log) = l.borrow_mut().as_mut() {
                log.push(Event::Flag(family, f));
            }
        });
    }
}

/// f32 で再現した値 `v_str` が構造と f64 で同じ（`v_str == v`）とみなせる要素の [`struct_check`]。
fn cancel_check(v: f64, m: f64, scale: f64, coeff: f64, what: &str) -> Result<(), AutodiffError> {
    struct_check(v, v, m, scale, coeff, what)
}

/// f32 で再現した中間値 `s` が f64 の値 `r` から、大きさ `mag` に対して丸め誤差の見積り `coeff` を
/// 超えてずれるか（underflow・非正規数・再構成の桁落ち等）。ずれる行は速い経路を使わず、f32 の再現値で
/// 組んだ構造の二階を要素ごとに参照と比べる。NaN は「ずれる」とする。
fn deviates(r: f64, s: f64, mag: f64, coeff: f64) -> bool {
    let d = (s - r).abs();
    d.is_nan() || d > coeff * mag
}

/// f32 の逐次和（構造の `Sum` の再現。和の順序の違いは `coeff` の長さ比例項が覆う）。
/// 本番の判定は総称関数（`Num::total`）へ移ったため、残る呼び出しは test の旧式との照合だけ。
#[cfg(test)]
fn sum32(it: impl Iterator<Item = f32>) -> f32 {
    it.fold(0f32, |a, x| a + x)
}

/// 構造の `s_b`（子テープの f32 の `g / n`・`g / W`・`g`）と f64 の `sb` の比。`sb == 0` は 1。
fn sb_ratio(sb32: f32, sb: f64) -> f64 {
    if sb == 0.0 {
        1.0
    } else {
        f64::from(sb32).abs() / sb
    }
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
    judge::upstream_abs(g_value, invalid)
}

/// reduction 後の scale 上界 `|g|/n`（Mean）／`|g|`（Sum）。
fn upstream_scale(
    g_value: &Tensor<f32>,
    reduction: Reduction,
    n: usize,
) -> Result<f64, AutodiffError> {
    judge::upstream_scale(g_value, reduction, n, invalid)
}

/// 子テープの `loss_scale_bcast` が f32 で作る `s_b`（Mean は `g / (n as f32)`、Sum は `g`）の再現。
fn upstream_scale32(g_value: &Tensor<f32>, reduction: Reduction, n: usize) -> f32 {
    judge::upstream_scale32(g_value, reduction, n)
}

/// `g` への cotangent 上界（要素寄与の絶対値和 × 要素あたりの係数 `1/n`／`1`）を検査する。
fn check_g_cotangent(sum_abs: f64, reduction: Reduction, n: usize) -> Result<(), AutodiffError> {
    judge::check_g_cotangent(sum_abs, reduction, n, invalid)
}

fn dense(parent: &Tape, parent_nodes: &[TapeNode], id: NodeId) -> Result<Vec<f32>, AutodiffError> {
    judge::dense(parent, parent_nodes, id)
}

fn reject_inf(values: &[&[f32]]) -> Result<(), AutodiffError> {
    judge::reject_inf(values, invalid)
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

/// 構造の `Sigmoid`（`eval::sigmoid` の f32 の式）の再現。
fn sigmoid32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
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
    // 判定する有効な行（ignore と範囲外を除く）の数。`Lin` の走査量の上限（`ce_lin_ctx`）に使う。
    let mut n_rows = 0usize;
    for &t in &target_data {
        if ignore_index == Some(t) {
            continue;
        }
        if in_range(t) {
            denom_w += weights[t as usize];
            n_rows += 1;
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
    // 構造の `s_b`（Mean は f32 の `g / W`）の f64 の `sb` に対する比（`g / W` の underflow 等）。
    let rho_ce = sb_ratio(
        upstream_scale32(&g_value, Reduction::Sum, 1)
            / if reduction == Reduction::Mean {
                w32
            } else {
                1.0
            },
        sb,
    );

    // `Lin` の上界で救済する判定の入力（符号つきの実際の `g` と Mean の分母 `W`。構造の `s_b = g / W` を f32 で再現する）。
    let g_signed = crate::eval::dense_vec(&g_value)
        .first()
        .map_or(0.0, |&v| f64::from(v));
    let lin_ctx = ce_lin_ctx(
        g_signed,
        (reduction == Reduction::Mean).then_some(denom_w),
        n_rows,
        axis_len,
    );

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
                (sb, gscale_ce, rho_ce),
                lin_ctx,
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

/// 既定オプションの `CrossEntropyLoss`（`Op::CrossEntropyLoss`）の相殺の総括判定（イシュー #3954、設計 §77）。
///
/// 親モジュールの `build_cgrads` の `Op::CrossEntropyLoss` 腕から、子テープへ何かを積む前に呼ばれる。
/// 既定 CE は CEWO の特殊ケース（重み全 1・ε=0・ignore なし: `A = 1`・`B = onehot`・`W = n`）なので、
/// 構造 `s_b⊙(softmax − onehot)` は変えず [`cewo_cancel_check`] をそのまま再利用する。確信度の高い行では
/// f32 の Softmax の VJP `y(1−y)` が 0 に潰れ、上流係数が大きいと REQ-2 を外れる値を沈黙で返すため、
/// 外れうる入力を型付き `Err` で拒否する（親の rollback で `child` は空に戻る）。
/// 非有限の `g` を新たに拒否することはしない（既存の受理挙動を保つ。NaN は素通し、inf は上界が非有限で拒否）。
pub(super) fn ce_cancel_check(
    parent: &Tape,
    parent_nodes: &[TapeNode],
    logits: NodeId,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
    g_value: &Tensor<f32>,
) -> Result<(), AutodiffError> {
    let shape = &parent_nodes[logits.0].shape;
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
    let n = outer * inner;
    let xs = dense(parent, parent_nodes, logits)?;
    let gv = crate::eval::dense_vec(g_value)
        .first()
        .map_or(0.0, |&v| f64::from(v).abs());
    let sb = match reduction {
        Reduction::Mean => gv / n as f64,
        Reduction::Sum => gv,
    };
    let rho = sb_ratio(upstream_scale32(g_value, reduction, n), sb);
    let gscale = g_scale(reduction, n);
    let weights = vec![1.0f64; axis_len];
    // 判定する有効な行（範囲外の target を除く）の数と、`Lin` の上界で救済する判定の入力。
    let n_rows = target_data
        .iter()
        .filter(|&&t| t >= 0 && (t as usize) < axis_len)
        .count();
    let g_signed = crate::eval::dense_vec(g_value)
        .first()
        .map_or(0.0, |&v| f64::from(v));
    let lin_ctx = ce_lin_ctx(
        g_signed,
        (reduction == Reduction::Mean).then_some(n as f64),
        n_rows,
        axis_len,
    );
    for o in 0..outer {
        for i in 0..inner {
            let t = target_data[o * inner + i];
            if t < 0 || (t as usize) >= axis_len {
                continue;
            }
            cewo_cancel_check(
                &xs,
                (o, i, inner, axis_len),
                (1.0, &weights, t as usize, 0.0),
                (sb, gscale, rho),
                lin_ctx,
            )?;
        }
    }
    Ok(())
}

/// CE・CEWO の 1 サンプル（class 軸 1 行）の Softmax の forward の中間値（`e = exp(x − max)`・`total = Σe`・`y = e/total`）。
/// `T` は `f64`（参照）・`f32`（子テープの構造の再現）・[`Lin`]（丸めの一次伝播の上界。#4093 で導入、#4094 で判定へ配線）。
///
/// 呼び出し元: [`cewo_cancel_check`]（`f64` と `f32`）、[`CeLinState`]（`Lin` と `f32`）、test 限定の [`cewo_lin_rows`]。
/// 最大値は構造と同じく f32 の fold max（f64 の fold max と同じ値）。和は [`Num::total`]（f32 は 0 からの逐次和で
/// 従来の `sum32` と bit 一致、f64 は `Iterator::sum` と同じ順序）。
/// `total` には [`Num::settle`] を掛けない: `y_j = e_j/total` で分子と分母が同じ `e_j` の事象を共有し、確信度の高い行の
/// 打ち消し `1 − y_j` を保つため。`y` は `settle` で事象 1 つへ束ね、二階の各要素の演算が O(C) に伸びるのを抑える。
struct CewoFwd<T> {
    e: Vec<T>,
    total: T,
    y: Vec<T>,
}

fn cewo_fwd<T: Num>(row32: &[f32]) -> CewoFwd<T> {
    let mx32 = row32.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mx = T::input(f64::from(mx32));
    let e: Vec<T> = row32
        .iter()
        .map(|&v| T::input(f64::from(v)).minus(mx).exp_bounded())
        .collect();
    let total = T::total(&e);
    let y = e.iter().map(|&v| v.over(total).settle(&[])).collect();
    CewoFwd { e, total, y }
}

/// seed 要素 `j` の二階のうち、要素 `i` に依らないスカラー部（`s_b`・`c = s_b·A`・`term = c·y_j`、`g` への二階）。
/// [`cewo_x`] が `x` への二階の 1 要素を作るときに使い回す。判定（[`CeLinState`]）は seed ごとに 1 回作り、
/// 要素ごとの値だけを作り直す（要素 1 つ分ずつ [`lin_truncate`] で戻し、判定時の事象数を抑える）。
#[derive(Clone, Copy)]
struct CewoSeed<T> {
    c: T,
    term: T,
    /// `g` への二階 `(A·y_j − B_j)·(1/W)`（Sum は `1/W` なし）。
    g: T,
}

/// CE・CEWO の構造 `s_b⊙(A⊙softmax(x) − B)` の seed 要素 `j` のスカラー部を `T` の演算で写す（#4093 で test 限定として
/// 導入、#4094 で本番の判定が使う）。
///
/// `coef = (A, B_j)`・`up = (g, Mean の分母 W)`。既定 CE は `A = 1`・`B = onehot`・`W = n` で同じ写しを共有する
/// （実際の構造には `·A` の乗算が無いが `A = 1` で f32 の値は同じ。`Lin` の余分な事象は過大評価で健全）。
/// 丸めの位置は `grad.rs` と `eval/scalar.rs::binary_partials` に合わせる。Softmax の VJP は実際のカーネルが
/// 差と積を f64 で計算して 1 回だけ f32 へ落とす混合精度だが、ここでは f32 の丸め事象として写す（健全な過大評価。
/// f64→f32 の二重丸めの余分は相対 2^-29 以下で `entry_check_lin` の余裕 `(1 + 16·N·u)` に収まる）。
fn cewo_seed<T: Num>(
    f: &CewoFwd<T>,
    j: usize,
    coef: (f64, f64),
    up: (f64, Option<f64>),
) -> CewoSeed<T> {
    let (a_s, b_j) = coef;
    let s_b = match up.1 {
        Some(w) => T::input(up.0).over(T::k(w)),
        None => T::input(up.0),
    };
    let c = s_b.times(T::k(a_s));
    let term = c.times(f.y[j]);
    let q = T::k(a_s).times(f.y[j]).minus(T::k(b_j));
    let g = match up.1 {
        Some(w) => q.times(T::k(1.0).over(T::k(w))),
        None => q,
    };
    CewoSeed { c, term, g }
}

/// seed `j` の `x` への二階の要素 `i`（対角は `y_j·(c − term)`、非対角は `y_i·(−term)`）。
fn cewo_x<T: Num>(f: &CewoFwd<T>, sd: &CewoSeed<T>, i: usize, j: usize) -> T {
    if i == j {
        f.y[j].times(sd.c.minus(sd.term))
    } else {
        f.y[i].times(sd.term.negate())
    }
}

/// [`cewo_rev`] の結果: seed 要素 `j` の `dx[j]` を 1 要素のスカラーに畳んで逆伝播したときの、logits `x` と `g` への二階。
#[cfg(test)]
pub(super) struct CewoRow<T> {
    pub(super) x: Vec<T>,
    pub(super) g: T,
}

/// [`cewo_seed`]・[`cewo_x`] の合成で seed `j` の二階の全要素を写す（test 限定。#4093 の soundness 掃引と bit 一致の錠が使う）。
#[cfg(test)]
fn cewo_rev<T: Num>(
    f: &CewoFwd<T>,
    j: usize,
    coef: (f64, f64),
    up: (f64, Option<f64>),
) -> CewoRow<T> {
    let sd = cewo_seed(f, j, coef, up);
    let x = (0..f.y.len()).map(|i| cewo_x(f, &sd, i, j)).collect();
    CewoRow { x, g: sd.g }
}

/// [`cewo_rev`] を `f32` で評価した二階。手組みの構造の実測と並べて soundness の「実測」側に使う（test 限定）。
#[cfg(test)]
pub(super) fn cewo_model_rows_f32(
    row32: &[f32],
    coef: (f64, &[f64]),
    up: (f64, Option<f64>),
    j: usize,
) -> CewoRow<f32> {
    let f = cewo_fwd::<f32>(row32);
    cewo_rev(&f, j, (coef.0, coef.1[j]), up)
}

/// [`cewo_rev`] の `Lin` 版の結果: 各要素の `(参照値 v, 一次の上界 bound)` と、そのときの事象の総数。
#[cfg(test)]
pub(super) struct LinRowCe {
    pub(super) x: Vec<(f64, f64)>,
    pub(super) g: (f64, f64),
    pub(super) events: usize,
}

/// `Lin` で forward と seed ごとの逆伝播を評価し、各要素の値と上界を返す（test 限定。#4093 の soundness 掃引・計測用）。
/// 厳密モード（[`lin_reset`]）で forward を 1 回作り、seed ごとに [`lin_mark`]／[`lin_truncate`] で領域を戻す。
#[cfg(test)]
pub(super) fn cewo_lin_rows(
    row32: &[f32],
    coef: (f64, &[f64]),
    up: (f64, Option<f64>),
    seeds: &[usize],
) -> Vec<LinRowCe> {
    use super::lin::lin_events;
    lin_reset();
    let f = cewo_fwd::<Lin>(row32);
    let mark = lin_mark();
    let out = seeds
        .iter()
        .map(|&j| {
            let r = cewo_rev(&f, j, (coef.0, coef.1[j]), up);
            let row = LinRowCe {
                x: r.x.iter().map(|l| (l.v, l.bound())).collect(),
                g: (r.g.v, r.g.bound()),
                events: lin_events(),
            };
            lin_truncate(mark);
            row
        })
        .collect();
    lin_reset();
    out
}

/// CE・CEWO の `Lin` 判定の入力（行に依らない量）。`g` は子テープ上の上流（符号つきの f32 の値）、`w` は Mean の分母
/// （f32 へ丸める前の f64。構造は `g / W` を f32 で作る）、`off` は非対角の全要素の救済走査（O(C²)）を許すか
/// （[`ce_lin_ctx`] が走査量の上限で決める）。
#[derive(Clone, Copy)]
struct CeLinCtx {
    g: f64,
    w: Option<f64>,
    off: bool,
}

/// [`CeLinState::ok`] が見る二階の種別（`G` は `g` への二階、`X` は x への二階の要素〈対角・非対角〉）。
#[derive(Clone, Copy)]
enum CeLinKind {
    G,
    X,
}

/// 事象の総数の上界（forward の `exp`・`total`・`y` と seed のスカラー、要素 1 つ分）。実測と固定は
/// `loss_class_tests.rs::ce_lin_events_stay_within_bound`（設計 §82.11）。
fn ce_lin_events_max(c: usize) -> usize {
    CE_LIN_EVENTS_PER_CLASS
        .saturating_mul(c)
        .saturating_add(CE_LIN_EVENTS_FIXED)
}

/// [`ce_lin_events_max`] のクラス 1 つあたりの事象数（`exp` の丸め・`y` の束ね）。
const CE_LIN_EVENTS_PER_CLASS: usize = 4;
/// [`ce_lin_events_max`] の定数分（`total` の丸め、seed のスカラー、`Mean` の `W`、要素 1 つ分）。
const CE_LIN_EVENTS_FIXED: usize = 40;

/// 1 行の `g` への二階と対角の曲率の判定（forward ＋ 全 seed の O(C) の要素）の走査量の見積り
/// （`Rf` の 1 反復相当。`LIN_BUDGET` と同じ単位）。forward の `total` が O(C²)、seed ごとの要素が O(C)。実測は設計 §82.11。
fn ce_lin_cost_diag(c: usize) -> usize {
    CE_LIN_COST_FWD
        .saturating_mul(c)
        .saturating_mul(c)
        .saturating_add(CE_LIN_COST_DIAG.saturating_mul(c))
}

/// 非対角の救済走査（全 `C(C−1)` 要素）の走査量の見積り。
fn ce_lin_cost_off(c: usize) -> usize {
    CE_LIN_COST_OFF.saturating_mul(c).saturating_mul(c)
}

const CE_LIN_COST_FWD: usize = 6;
const CE_LIN_COST_DIAG: usize = 200;
const CE_LIN_COST_OFF: usize = 24;

/// CE・CEWO の総括判定で [`Lin`] の上界を使うか、使うなら非対角の救済走査まで許すか（テンソル単位。
/// `cos_use_lin` と同じ方針）。事象の総数が余裕係数の上限（`16·事象数·u < 0.01`）に収まり、かつ対角・`g` の走査量が
/// [`LIN_BUDGET`] 以内のときだけ `Some`。非対角まで含めた走査量も収まるときだけ `off = true`。満たさないテンソルは
/// 従来の見積りだけで判定する（受理集合は従来と同じ。走査量の上限を超えたら `Lin` を使わない形で
/// `judge_budget` の代わりに守る。設計 §82.11）。`n_rows` は判定する有効な行の数。
fn ce_lin_ctx(g: f64, w: Option<f64>, n_rows: usize, c: usize) -> Option<CeLinCtx> {
    #[cfg(test)]
    if probe::FORCE_OLD.with(|f| f.get()) {
        return None;
    }
    if 16.0 * ce_lin_events_max(c) as f64 * LIN_U >= 0.01 {
        return None;
    }
    let diag = n_rows.saturating_mul(ce_lin_cost_diag(c));
    if diag > LIN_BUDGET {
        return None;
    }
    let off = diag.saturating_add(n_rows.saturating_mul(ce_lin_cost_off(c))) <= LIN_BUDGET;
    Some(CeLinCtx { g, w, off })
}

/// [`CeLinState`] が持つ seed `j` のスカラー（`Lin` 版と f32 版）と、その直後の領域の印。
type CeLinSeed = (usize, LinMark, CewoSeed<Lin>, CewoSeed<f32>);

/// CE・CEWO の 1 行の [`Lin`] 判定の状態。forward は行で最初に必要になったとき 1 回だけ作り、seed のスカラーは
/// `j` が変わるたびに作り直し、要素ごとの値は作ってすぐ [`lin_truncate`] で戻す（判定時の事象数を forward ＋ seed の
/// スカラー ＋ 要素 1 つ分に抑え、`entry_check_lin` の余裕係数の条件を満たしやすくする。`entry_check_lin` は判定時点の
/// 全体の事象数を余裕に使うので、要素ごとの truncate は必須）。`Drop` で thread-local の領域を解放する。
struct CeLinState {
    lin: CewoFwd<Lin>,
    f32m: CewoFwd<f32>,
    a_s: f64,
    up: (f64, Option<f64>),
    fwd_mark: LinMark,
    cur: Option<CeLinSeed>,
}

impl CeLinState {
    fn new(row32: &[f32], a_s: f64, ctx: CeLinCtx) -> Self {
        lin_reset();
        let lin = cewo_fwd::<Lin>(row32);
        let f32m = cewo_fwd::<f32>(row32);
        CeLinState {
            lin,
            f32m,
            a_s,
            up: (ctx.g, ctx.w),
            fwd_mark: lin_mark(),
            cur: None,
        }
    }

    fn seed(&mut self, j: usize, b_j: f64) -> (LinMark, CewoSeed<Lin>, CewoSeed<f32>) {
        if let Some((jj, m, a, b)) = self.cur
            && jj == j
        {
            return (m, a, b);
        }
        lin_truncate(self.fwd_mark);
        let a = cewo_seed(&self.lin, j, (self.a_s, b_j), self.up);
        let b = cewo_seed(&self.f32m, j, (self.a_s, b_j), self.up);
        let m = lin_mark();
        self.cur = Some((j, m, a, b));
        (m, a, b)
    }

    /// seed `j` の二階の要素（`kind`・`i`）が [`entry_check_lin`] を通るか。
    fn ok(&mut self, kind: CeLinKind, i: usize, j: usize, b_j: f64, what: &str) -> bool {
        let (mark, a, b) = self.seed(j, b_j);
        let ok = match kind {
            CeLinKind::G => entry_check_lin(a.g, b.g, what, invalid).is_ok(),
            CeLinKind::X => {
                let l = cewo_x(&self.lin, &a, i, j);
                let s = cewo_x(&self.f32m, &b, i, j);
                entry_check_lin(l, s, what, invalid).is_ok()
            }
        };
        lin_truncate(mark);
        ok
    }
}

impl Drop for CeLinState {
    fn drop(&mut self) {
        lin_reset();
    }
}
/// CEWO の 1 サンプル（`(o, i)` の class 軸）の相殺検査（設計 §55.2 の相殺の総括判定）。
///
/// 構造 `s_b⊙(A⊙softmax(x) − B)` の二階は、x 経路が Softmax の VJP `s_b·A·y_i(δ_ij − y_j)`。対角の
/// `1 − y_j` は確信度の高い行（`y_j → 1`）で相殺し、f32 の `y_j` が 1 に丸まると曲率が 0 になる
/// （真値は他クラスの確率の和）。非対角は単項の積で相殺しない。g 経路（`∂dx/∂g`）は `A·y_c − B_c`
/// で、target の `A·y_t ≈ B_t` が同じく相殺する。参照値は f64 の softmax で、`1 − y_j` は他クラスの
/// 和から桁落ちなく作る。NaN の行は伝播させるため検査しない。
///
/// **`Lin` の上界による救済（#4094・設計 §82.11）**: 対角・`g`・`flag` 行の非対角は、従来の見積り
/// （`coeff·m`）が拒否した要素だけ [`CeLinState`] で評価し直し、閾値内なら受理する（[`lin_elem`]）。非 `flag` 行の
/// 非対角は、従来の上位 2 クラスの積の近似が通れば受理（速い経路）、通らなければ `lin.off` のときだけ全非対角を
/// 要素ごとに評価し直す。`lin = None`（走査量・事象数の上限を超えるテンソル、`-inf` や `|x − max| > 2^23` のロジットで
/// `exp_bounded` が上界を持たない行を含む）は従来の見積りのまま。どちらも独立に健全な上界なので拒否範囲は従来以下。
fn cewo_cancel_check(
    xs: &[f32],
    pos: (usize, usize, usize, usize),
    coef: (f64, &[f64], usize, f64),
    scales: (f64, f64, f64),
    lin: Option<CeLinCtx>,
) -> Result<(), AutodiffError> {
    let (o, i, inner, c) = pos;
    let (a_s, weights, t, eps) = coef;
    let (sb, gscale, rho) = scales;
    let row32: Vec<f32> = (0..c).map(|a| xs[(o * c + a) * inner + i]).collect();
    if row32.iter().any(|v| v.is_nan()) {
        return Ok(());
    }
    // f64 の参照と、構造の f32 の Softmax（`eval::softmax_along` と同じ `exp(x − max)` の逐次和と除算）の再現は
    // 同じ総称関数 [`cewo_fwd`] で組む（#4093。従来は同じ式を 2 回書いていた）。小さな確率は f32 で 0 や
    // 非正規数へ underflow し、その曲率が消える。
    let ref64 = cewo_fwd::<f64>(&row32);
    let (e, total) = (ref64.e, ref64.total);
    let f32f = cewo_fwd::<f32>(&row32);
    let y32: Vec<f64> = f32f.y.iter().map(|&v| f64::from(v)).collect();
    let total_w: f64 = weights.iter().sum();
    let smooth = eps / c as f64;
    let coeff = rounding_coeff(c);
    let sba = sb * a_s;
    let a32 = f64::from(a_s as f32);
    let sba_str = rho * sb * a32;
    let mut flag = deviates(1.0, rho, 1.0, coeff) || deviates(a_s, a32, a_s.abs(), coeff);
    let ys: Vec<f64> = ref64.y;
    let (mut top1, mut top2) = (0f64, 0f64);
    // 構造の `B_j`（f64）。g への二階と `Lin` の seed が使う。
    let b_at = |j: usize| -> f64 {
        (1.0 - eps) * weights[t] * if j == t { 1.0 } else { 0.0 } + smooth * weights[j]
    };
    // `Lin` の上界の状態（従来の見積りが最初に拒否した要素で 1 回だけ作る。`Drop` で領域を解放する）。
    let mut lin_state: Option<CeLinState> = None;
    let mut rescue = |kind: CeLinKind, i: usize, j: usize, what: &str| -> bool {
        let Some(ctx) = lin else {
            return false;
        };
        lin_state
            .get_or_insert_with(|| CeLinState::new(&row32, a_s, ctx))
            .ok(kind, i, j, b_at(j), what)
    };
    for j in 0..c {
        let y = ys[j];
        let ys32 = y32[j];
        flag |= deviates(y, ys32, y, coeff);
        #[cfg(test)]
        probe::flag("CrossEntropy", flag);
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
        const DIAG_WHAT: &str = "CrossEntropy: Softmax の対角の曲率";
        lin_elem(
            [
                sba * y * rest,
                sba_str * ys32 * (1.0 - ys32),
                sba * y * (1.0 + y),
                1.0,
                coeff,
            ],
            DIAG_WHAT,
            || rescue(CeLinKind::X, j, j, DIAG_WHAT),
        )?;
        // `A·y_j − B_j = (1−ε)·w_t·(y_j − δ_jt) + (ε/C)·(Σw·y_j − w_j)`（`y_t − 1 = −rest_t`）。
        let hard = if j == t { -rest } else { y };
        let v = (1.0 - eps) * weights[t] * hard + smooth * (total_w * y - weights[j]);
        let b = b_at(j);
        let (a_f, b_f, y_f) = (a_s as f32, b as f32, ys32 as f32);
        const G_WHAT: &str = "CrossEntropy: g への二階（A·softmax − B）";
        lin_elem(
            [
                v,
                f64::from(a_f * y_f - b_f),
                a_s * y + b.abs(),
                gscale,
                coeff,
            ],
            G_WHAT,
            || rescue(CeLinKind::G, j, j, G_WHAT),
        )?;
        if y > top1 {
            top2 = top1;
            top1 = y;
        } else if y > top2 {
            top2 = y;
        }
    }
    #[cfg(test)]
    probe::flag("CrossEntropy", flag);
    if flag {
        // f32 の再現値が丸め誤差の見積りを超えてずれる行（確率の underflow 等）は、非対角
        // `−s_b·A·y_i·y_j` も要素ごとに f32 の再現値で組んだ値と参照を比べる。非対角の `Lin` 救済は
        // O(C²) なので `lin.off`（走査量が `LIN_BUDGET` 以内）のときだけ呼ぶ（設計 §82.11.1）。
        for jj in 0..c {
            for ii in 0..c {
                if ii == jj {
                    continue;
                }
                let v = sba * ys[ii] * ys[jj];
                const FLAG_WHAT: &str = "CrossEntropy: Softmax の非対角の曲率（f32 の underflow）";
                lin_elem(
                    [v, sba_str * y32[ii] * y32[jj], v.abs(), 1.0, coeff],
                    FLAG_WHAT,
                    || lin.is_some_and(|l| l.off) && rescue(CeLinKind::X, ii, jj, FLAG_WHAT),
                )?;
            }
        }
        return Ok(());
    }
    // 非対角 `−s_b·A·y_i·y_j` は単項で `m == |v|`。`coeff` が相対の半分を超える長い軸のときだけ
    // 絶対判定に落ちるので、最大の積（上位 2 クラス）で確かめる。単項の積は大きさに対して単調なので、最大の積が
    // 通れば全要素が通る（従来の速い経路。受理は変えない）。
    const OFF_WHAT: &str = "CrossEntropy: Softmax の非対角の曲率";
    let top_v = sba * top1 * top2;
    if judge::struct_check(top_v, top_v, top_v, 1.0, coeff, OFF_WHAT, invalid).is_ok() {
        return Ok(());
    }
    // 最大の積が拒否された。`Lin` を使えるテンソルでは、上位 2 クラスの近似ではなく全非対角を要素ごとに評価し直し
    // （従来の見積り → 通らなければ `Lin` の上界）、1 要素でも両方で拒否されれば従来のエラーを返す。受理は全要素を
    // 確かめた場合に限るので、従来の受理が拒否へ反転することはない（設計 §82.11）。
    if lin.is_some_and(|l| l.off) {
        let mut all_ok = true;
        'scan: for jj in 0..c {
            for ii in 0..c {
                if ii == jj {
                    continue;
                }
                let v = sba * ys[ii] * ys[jj];
                if judge::struct_check(v, v, v, 1.0, coeff, OFF_WHAT, invalid).is_ok() {
                    continue;
                }
                if !rescue(CeLinKind::X, ii, jj, OFF_WHAT) {
                    all_ok = false;
                    break 'scan;
                }
            }
        }
        if all_ok {
            return Ok(());
        }
    }
    cancel_check(top_v, top_v, 1.0, coeff, OFF_WHAT)
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
    let rho = sb_ratio(upstream_scale32(&g_value, reduction, n), sb);

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
        let mut sum_z32 = 0f64;
        let cw32 = f64::from(cw as f32);
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
                    // 構造の f32 の値（`z32`・定数 `c_w` の f32）で組んだ値と参照を比べる。
                    let zm = margin64.abs() + xy.abs() + f64::from(xj).abs();
                    struct_check(
                        cw * h,
                        2.0 * cw32 * f64::from(z32),
                        2.0 * cw * zm,
                        gs,
                        coeff,
                        "MultiMargin: マージン項 Z の差し引き",
                    )?;
                    sum_zm += zm;
                    sum_z32 += f64::from(z32);
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
            struct_check(
                cw * sum_h,
                2.0 * cw32 * sum_z32,
                2.0 * cw * sum_zm,
                gs,
                coeff,
                "MultiMargin: マージン項 Z の差し引き（target）",
            )?;
            // x 経路の二階は `±2·s_b·c_w`（target は個数倍）の定数。f32 の `s_b`・`c_w` の
            // underflow（非正規数・0）で曲率が消える入力を拒否する。
            if count > 0.0 {
                let v = 2.0 * sb * cw * count;
                struct_check(
                    v,
                    2.0 * rho * sb * cw32 * count,
                    v.abs(),
                    1.0,
                    coeff,
                    "MultiMargin: 定数 c_w・s_b の f32 underflow",
                )?;
            }
            stage(cw * sum_h)?;
            stage(sb * cw * sum_h)?;
            stage(sb * cw * 2.0)?;
            stage(sb * cw)?;
            gsum += 2.0 * cw * sum_h;
        } else {
            k_data[i * c + y] = (-cw * count) as f32;
            // g 経路の二階は定数 `K`（f32）。f32 で 0 や非正規数になる `K` を拒否する。
            if count > 0.0 {
                struct_check(
                    cw,
                    cw32,
                    cw.abs(),
                    gs,
                    coeff,
                    "MultiMargin: 定数 K の f32 underflow",
                )?;
                struct_check(
                    cw * count,
                    f64::from((cw * count) as f32),
                    (cw * count).abs(),
                    gs,
                    coeff,
                    "MultiMargin: 定数 K の f32 underflow（target）",
                )?;
            }
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
    let gs = g_scale(reduction, n);

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
            // g 経路の二階は定数 `K/C`（f32）。f32 で 0 や非正規数になる値を拒否する（区分定数なので
            // x 経路の二階は 0）。
            struct_check(
                v,
                f64::from(v as f32),
                v.abs(),
                gs,
                rounding_coeff(1),
                "MultiLabelMargin: 定数 K の f32 underflow",
            )?;
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
    let rho = sb_ratio(upstream_scale32(&g_value, reduction, n), sb);
    let coeff = rounding_coeff(1);

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
            // 構造の f32 の値（`Sigmoid(−|x|)`・定数 `w_c/C`・`s_b` の f32）で組んだ値と参照を比べる。
            // 飽和域の σ は f32 で非正規数や 0 へ underflow し、x 経路の曲率 `σ(1−σ)` が消える。
            let (x32, t32) = (xs[i * c + j], ts[i * c + j]);
            let s32 = sigmoid32(-x32.abs());
            let s64 = sigmoid64(-x.abs());
            let wc32 = wc as f32;
            let mag = if x >= 0.0 {
                (1.0 - t).abs() + s64
            } else {
                s64 + t.abs()
            };
            let form32 = if x >= 0.0 {
                (1.0 - t32) - s32
            } else {
                s32 - t32
            };
            struct_check(
                wc * (sigmoid64(x) - t),
                f64::from(wc32 * form32),
                wc.abs() * mag,
                gs,
                coeff,
                "MultiLabelSoftMargin: σ(x) − t の差し引き",
            )?;
            let curv = sb * wc * s64 * (1.0 - s64);
            struct_check(
                curv,
                rho * sb * f64::from(wc32) * f64::from(s32 * (1.0 - s32)),
                curv.abs(),
                1.0,
                coeff,
                "MultiLabelSoftMargin: 飽和域の σ(1−σ) の f32 underflow",
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
    let rho = sb_ratio(upstream_scale32(&g_value, reduction, n), sb);
    let dim = shape.len() - 1;
    let numel = n * d;
    // `Lin` の上界で相殺判定を救うか（テンソル単位。#4091・設計 §82.8）。`g` は子テープ上の f32 の値。
    let use_lin = cos_use_lin(n, d);
    let g_f32 = f64::from(
        crate::eval::dense_vec(&g_value)
            .first()
            .copied()
            .unwrap_or(0.0),
    );

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
        cosine_cancel_check(
            (&a_s[base..base + d], &b_s[base..base + d]),
            (inv1, inv2),
            (eps1, eps2),
            (sb, gs, rho),
            use_lin.then_some(CosLinCtx {
                g: g_f32,
                mean_n: matches!(reduction, Reduction::Mean).then_some(n),
                cf,
            }),
        )?;
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

/// Cosine の forward の中間値（`a' = a·(1/s₁)`・`m = Σa'² + ε'`・`s = √m`・`den = s₁s₂`・`dot = Σa'b'`・`cos = dot/den`）。
/// `T` は `f64`（参照）・`f32`（子テープの構造の再現）・[`Lin`]（丸めの一次伝播の上界。#4090・#4091）のいずれか。
///
/// 呼び出し元: [`cosine_cancel_check`]（`f64` と `f32` の再現、および相殺判定を救う `Lin` の上界 [`CosLinState`]）。
/// 演算の順序は子テープの forward（`loss_scale_bcast` 以外の `ap`・`bp`・`m1p`・`m2p`・`den`・`cos`）と同じで、
/// 和は [`Num::total`]。
struct CosFwd<T> {
    ap: Vec<T>,
    bp: Vec<T>,
    m1: T,
    m2: T,
    s1: T,
    s2: T,
    den: T,
    /// 内積（`cos` の分子）。逆伝播のモデル [`cos_rev_seed`] が読む。
    dot: T,
    cos: T,
}

/// [`CosFwd`] を `T` の演算で組む。`inv`＝行スケールの逆数 `(1/s₁, 1/s₂)`（2 のべき乗）、`eps`＝`(ε/s₁², ε/s₂²)`。
///
/// 行全体の縮約（`Σa'²`・`Σb'²`・`Σa'b'`）の直後に [`Num::settle`] を掛け、`Lin` の係数列を独立な事象 1 つへ束ねる
/// （#4091・設計 §82.8。`f64`・`f32` では恒等）。束ねないと `m1`・`den`・`cos` が全要素の事象を係数列に持ち、
/// 下流の二階の各要素の演算が O(d) になって判定の走査量が O(d³) に伸びる。束ねた事象は三角不等式で上界が
/// 広がる側にしか動かないので判定は緩まず、失うのは「要素ごとの事象と、それを直接使う値」の相関だけである。
fn cos_fwd<T: Num>(a32: &[f32], b32: &[f32], inv: (f64, f64), eps: (f64, f64)) -> CosFwd<T> {
    let scale = |v: &[f32], r: f64| -> Vec<T> {
        v.iter()
            .map(|&x| T::input(f64::from(x)).times(T::k(r)))
            .collect()
    };
    let (ap, bp) = (scale(a32, inv.0), scale(b32, inv.1));
    let sq = |v: &[T], e: f64| -> T {
        let terms: Vec<T> = v.iter().map(|&x| x.times(x)).collect();
        T::total(&terms).settle(&[]).plus(T::k(e))
    };
    let (m1, m2) = (sq(&ap, eps.0), sq(&bp, eps.1));
    let (s1, s2) = (m1.root(), m2.root());
    let den = s1.times(s2);
    let prods: Vec<T> = ap.iter().zip(&bp).map(|(&a, &b)| a.times(b)).collect();
    let dot = T::total(&prods).settle(&[]);
    let cos = dot.over(den);
    CosFwd {
        ap,
        bp,
        m1,
        m2,
        s1,
        s2,
        den,
        dot,
        cos,
    }
}

impl<T: Copy> CosFwd<T> {
    /// own と other を入れ替えた見え方（dx2 の逆伝播は dx1 の a と b を入れ替えた形）。
    fn swapped(&self) -> CosFwd<T> {
        CosFwd {
            ap: self.bp.clone(),
            bp: self.ap.clone(),
            m1: self.m2,
            m2: self.m1,
            s1: self.s2,
            s2: self.s1,
            den: self.den,
            dot: self.dot,
            cos: self.cos,
        }
    }
}

/// seed 要素 `j` の二階の逆伝播のうち、要素 `i` に依らないスカラーの cotangent（[`cos_rev_seed`] が作る）。
/// 要素ごとの値（[`cos_rev_own`]・[`cos_rev_other`]）はこれと forward から組む。
#[derive(Clone, Copy)]
struct CosSeed<T> {
    j: usize,
    d_m1: T,
    d_m2: T,
    d_dot: T,
    d_apj_c: T,
    d_bpj_x: T,
    d_g: T,
}

/// Cosine の二階の逆伝播（子テープの eager 逆伝播 `grad.rs::vjp` の演算）を `T` の演算で写す。seed 要素 `j` の
/// スカラー cotangent を作る段（#4090 で test 限定に導入、#4091 で判定へ配線）。
///
/// 構造は `dx_own = ((s_b·cf)·r_own)·inner`、`inner = bp/den − (cos·ap)/m_own`（`loss_class_tests.rs::CosCase::hand_rows`）。
/// 写し方（`grad.rs`・`eval/scalar.rs` の実装に合わせた丸めの位置）:
/// - `Mul` の VJP は `上流·相手の値`（丸め 1 回）。`Div` は係数 `1/b` と `−(a/b)/b` を f32 で作ってから上流を掛ける。
///   `Sub` の係数 ±1 は厳密。`Sqrt` の係数は `0.5/y`。
/// - 複数回使われる値（`ap`・`bp`・`m`・`den`）への寄与の合算と、`[1,1] → [1,d]` の broadcast の縮約は [`Num::total`]
///   （順序に依らない上界）。seed の行以外の上流は 0 なので縮約は 1 項と同じ。
/// - `where`・`reshape`・`narrow`・`contiguous`・`Sum` の VJP は値を変えない。
/// - `s_b` は子テープの `loss_scale_bcast` と同じく Sum は `g`、Mean（`mean_n = Some(n)`）は f32 の `g / n`。
///   `g` への二階は Mean で `Div` の係数 `1/n` を f32 で作ってから掛ける。
///
/// `f` は own 側から見た [`CosFwd`]（dx2 は [`CosFwd::swapped`]）、`r_own` は own 側の行スケールの逆数、`cf ∈ {−1, +1}`。
fn cos_rev_seed<T: Num>(
    f: &CosFwd<T>,
    j: usize,
    g: f64,
    mean_n: Option<usize>,
    cf: f64,
    r_own: f64,
) -> CosSeed<T> {
    let one = T::k(1.0);
    let (apj, bpj) = (f.ap[j], f.bp[j]);
    let sb = match mean_n {
        Some(n) => T::input(g).over(T::k(n as f64)),
        None => T::input(g),
    };
    let sc = if cf < 0.0 { sb.negate() } else { sb };
    let q = sc.times(T::k(r_own));
    let x = bpj.over(f.den);
    let cj = f.cos.times(apj);
    let y = cj.over(f.m1);
    let inner = x.minus(y);
    // dx_own[j] = q·inner を seed 1 で逆伝播する。
    let d_q = inner;
    let d_inner = q;
    let d_sc = d_q.times(T::k(r_own));
    let d_sb = if cf < 0.0 { d_sc.negate() } else { d_sc };
    let d_g = match mean_n {
        Some(n) => d_sb.times(one.over(T::k(n as f64))),
        None => d_sb,
    };
    let (d_x, d_y) = (d_inner, d_inner.negate());
    // X = bp_j / den
    let d_bpj_x = d_x.times(one.over(f.den));
    let d_den_x = d_x.times(bpj.over(f.den).negate().over(f.den));
    // Y = cj / m_own
    let d_cj = d_y.times(one.over(f.m1));
    let d_m1_y = d_y.times(cj.over(f.m1).negate().over(f.m1));
    // cj = cos · ap_j
    let d_cos = d_cj.times(apj);
    let d_apj_c = d_cj.times(f.cos);
    // cos = dot / den
    let d_dot = d_cos.times(one.over(f.den));
    let d_den_c = d_cos.times(f.dot.over(f.den).negate().over(f.den));
    let d_den = T::total(&[d_den_x, d_den_c]);
    // den = s1 · s2、s = sqrt(m)
    let d_s1 = d_den.times(f.s2);
    let d_s2 = d_den.times(f.s1);
    let d_m1 = T::total(&[d_m1_y, d_s1.times(f.s1.root_grad())]);
    let d_m2 = d_s2.times(f.s2.root_grad());
    CosSeed {
        j,
        d_m1,
        d_m2,
        d_dot,
        d_apj_c,
        d_bpj_x,
        d_g,
    }
}

/// seed `sd` の own 側（`a`／`b`）要素 `i` への二階。`m = Σ ap²` の `ap·ap` は同じ丸めの値を 2 回足す。
fn cos_rev_own<T: Num>(f: &CosFwd<T>, sd: &CosSeed<T>, i: usize, r_own: f64) -> T {
    let t1 = sd.d_m1.times(f.ap[i]);
    let o = [t1, t1, sd.d_dot.times(f.bp[i]), sd.d_apj_c];
    let n = if i == sd.j { 4 } else { 3 };
    T::total(&o[..n]).times(T::k(r_own))
}

/// seed `sd` の other 側要素 `i` への二階（[`cos_rev_own`] と対称）。
fn cos_rev_other<T: Num>(f: &CosFwd<T>, sd: &CosSeed<T>, i: usize, r_other: f64) -> T {
    let t2 = sd.d_m2.times(f.bp[i]);
    let h = [t2, t2, sd.d_dot.times(f.ap[i]), sd.d_bpj_x];
    let n = if i == sd.j { 4 } else { 3 };
    T::total(&h[..n]).times(T::k(r_other))
}

/// [`cos_rev_seed`]・[`cos_rev_own`]・[`cos_rev_other`] の結果: seed 要素 `j` の `dx_own[j]` を 1 要素のスカラーに畳んで
/// 逆伝播したときの、own 側（`a`／`b`）・other 側・`g` への二階。
#[cfg(test)]
pub(super) struct CosRow<T> {
    pub(super) own: Vec<T>,
    pub(super) other: Vec<T>,
    pub(super) g: T,
}

/// 全要素分の [`CosRow`] を作る合成（test 限定。判定は要素ごとに [`cos_rev_own`] 等を直接呼ぶ）。`r` は `(r_own, r_other)`。
#[cfg(test)]
fn cos_rev<T: Num>(
    f: &CosFwd<T>,
    j: usize,
    up: (f64, Option<usize>),
    cf: f64,
    r: (f64, f64),
) -> CosRow<T> {
    let sd = cos_rev_seed(f, j, up.0, up.1, cf, r.0);
    let d = f.ap.len();
    CosRow {
        own: (0..d).map(|i| cos_rev_own(f, &sd, i, r.0)).collect(),
        other: (0..d).map(|i| cos_rev_other(f, &sd, i, r.1)).collect(),
        g: sd.d_g,
    }
}

/// [`cos_rev`] を `f32` で評価した二階（`[side]`。side 0 = dx1〈own = a〉、1 = dx2〈own = b〉）。`f32` の演算は
/// 構造の演算の写しで、手組みの構造の実測と並べて soundness の「実測」側に使う（test 限定）。
#[cfg(test)]
pub(super) fn cosine_model_rows_f32(
    rows: (&[f32], &[f32]),
    inv: (f64, f64),
    eps: (f64, f64),
    up: (f64, Option<usize>),
    cf: f64,
    j: usize,
) -> [CosRow<f32>; 2] {
    let f = cos_fwd::<f32>(rows.0, rows.1, inv, eps);
    [
        cos_rev(&f, j, up, cf, inv),
        cos_rev(&f.swapped(), j, up, cf, (inv.1, inv.0)),
    ]
}

/// [`cos_rev`] の `Lin` 版の結果 1 側分: 各要素の `(参照値 v, 一次の上界 bound)` と、そのときの事象の総数。
#[cfg(test)]
pub(super) struct LinRow {
    pub(super) own: Vec<(f64, f64)>,
    pub(super) other: Vec<(f64, f64)>,
    pub(super) g: (f64, f64),
    pub(super) events: usize,
}

/// `Lin` で forward と seed ごとの逆伝播を評価し、各要素の値と上界を返す（`[seed][side]`。test 限定。#4090）。
/// 厳密モード（[`lin_reset`]）で forward を 1 回作り、seed ごとに [`lin_mark`]／[`lin_truncate`] で領域を戻す。
/// 判定本体（[`cosine_cancel_check`]）は要素ごとに領域を戻して N を下げる。ここは seed 全体の上界（soundness の
/// 掃引と事象数の計測）を取るための全要素版で、seed 内の事象数 `events` は判定より大きい。
#[cfg(test)]
pub(super) fn cosine_lin_rows(
    rows: (&[f32], &[f32]),
    inv: (f64, f64),
    eps: (f64, f64),
    up: (f64, Option<usize>),
    cf: f64,
    seeds: &[usize],
) -> Vec<[LinRow; 2]> {
    use super::lin::{Lin, lin_events, lin_mark, lin_reset, lin_truncate};
    lin_reset();
    let f = cos_fwd::<Lin>(rows.0, rows.1, inv, eps);
    let fs = f.swapped();
    let mark = lin_mark();
    let pack = |r: CosRow<Lin>| LinRow {
        own: r.own.iter().map(|l| (l.v, l.bound())).collect(),
        other: r.other.iter().map(|l| (l.v, l.bound())).collect(),
        g: (r.g.v, r.g.bound()),
        events: lin_events(),
    };
    let out = seeds
        .iter()
        .map(|&j| {
            let a = pack(cos_rev(&f, j, up, cf, inv));
            lin_truncate(mark);
            let b = pack(cos_rev(&fs, j, up, cf, (inv.1, inv.0)));
            lin_truncate(mark);
            [a, b]
        })
        .collect();
    lin_reset();
    out
}

/// Cosine の 1 側（dx1 なら own = a'・other = b'）の二階の展開に使う値。f64 の参照と、構造が f32 で
/// 計算する値（`a·(1/s)`・`Σa'² + ε'`・`sqrt`・内積・`cos`）の再現の 2 通りで作る。
struct CosSide {
    own: Vec<f64>,
    other: Vec<f64>,
    m_own: f64,
    s_own: f64,
    s_other: f64,
    den: f64,
    cos: f64,
}

impl CosSide {
    /// seed 要素 `j` の cotangent `(cot_D, cot_m_own, cot_m_other)`。
    fn cots(&self, j: usize) -> (f64, f64, f64) {
        let (oj, tj) = (self.own[j], self.other[j]);
        let cot_d = -oj / (self.m_own * self.den);
        let cot_den = -tj / (self.den * self.den) + oj * self.cos / (self.m_own * self.den);
        let cot_mo =
            self.cos * oj / (self.m_own * self.m_own) + cot_den * self.s_other * 0.5 / self.s_own;
        let cot_mt = cot_den * self.s_own * 0.5 / self.s_other;
        (cot_d, cot_mo, cot_mt)
    }

    fn own_entry(&self, c: (f64, f64, f64), i: usize, j: usize) -> f64 {
        let delta = if i == j { 1.0 } else { 0.0 };
        delta * (-self.cos / self.m_own) + self.other[i] * c.0 + 2.0 * self.own[i] * c.1
    }

    fn cross_entry(&self, c: (f64, f64, f64), i: usize, j: usize) -> f64 {
        let delta = if i == j { 1.0 } else { 0.0 };
        delta / self.den + self.own[i] * c.0 + 2.0 * self.other[i] * c.2
    }

    fn g_entry(&self, j: usize) -> f64 {
        self.other[j] / self.den - self.cos * self.own[j] / self.m_own
    }
}

/// Cosine の `Lin` 判定の入力（行に依らない量）。`g` は子テープ上の上流（f32 の値）、`mean_n` は Mean の分母、
/// `cf ∈ {−1, +1}` は損失の係数（`cf == 0` の行は判定に来ない）。
#[derive(Clone, Copy)]
struct CosLinCtx {
    g: f64,
    mean_n: Option<usize>,
    cf: f64,
}

/// [`CosLinState::ok`] が見る二階の種別（`G` は `g` への二階、`Own`・`Cross` は x への同じ側・相手側）。
#[derive(Clone, Copy)]
enum LinKind {
    G,
    Own,
    Cross,
}

/// 事象の総数の上界（forward の分と、seed 1 回分のスカラーの分、要素 1 つ分）。forward は `ap`・`bp` の積・
/// 二乗・内積の各 `d` 個の丸めが主で約 `5d`。実測と固定は `loss_class_tests.rs::cos_lin_events_stay_within_bound`
/// （設計 §82.8）。
fn cos_lin_events_max(d: usize) -> usize {
    5usize
        .saturating_mul(d)
        .saturating_add(COS_LIN_EVENTS_FIXED)
}

/// [`cos_lin_events_max`] の定数分（行のスカラー〈`m`・`s`・`den`・`cos`・`ε'`〉と seed のスカラー、`Mean` の `n`）。
const COS_LIN_EVENTS_FIXED: usize = 80;

/// 1 行を [`Lin`] で判定する走査量の見積り（`Rf` の 1 反復相当。`LIN_BUDGET` と同じ単位）。要素ごとの検査が
/// 2 側 × （同じ側・相手側）× `d²` 回で、実測は設計 §82.8 の表。
fn cos_lin_cost(d: usize) -> usize {
    COS_LIN_COST_PER_ELEM.saturating_mul(d).saturating_mul(d)
}

const COS_LIN_COST_PER_ELEM: usize = 1000;

/// Cosine の総括判定で [`Lin`] の上界を使うか（テンソル単位。`wn_use_lin`・`sn_use_lin` と同じ方針）。
/// 事象の総数が余裕係数の上限（`16·事象数·u < 0.01`）に収まり、かつ走査量が [`LIN_BUDGET`] 以内のときだけ。
/// 満たさないテンソルは従来の見積りだけで判定する（`Lin` を選んで全要素が余裕係数で拒否される、
/// あるいは判定時間が伸びる、という不利を避ける。受理集合は従来と同じ）。
fn cos_use_lin(n: usize, d: usize) -> bool {
    #[cfg(test)]
    if probe::FORCE_OLD.with(|c| c.get()) {
        return false;
    }
    16.0 * cos_lin_events_max(d) as f64 * LIN_U < 0.01
        && n.saturating_mul(cos_lin_cost(d)) <= LIN_BUDGET
}

/// 従来の見積り（[`judge::struct_check`]）を先に評価し、拒否した要素だけ `rescue`（`Lin` の上界）で救う。
/// どちらの見積りも f32 構造の誤差の上界として独立に健全なので、片方が閾値内なら REQ-2 の判定は成り立つ。
/// 従来の受理を拒否へ反転させることは構造的に起きない（拒否範囲は従来以下。tolerance 定数は不変）。
/// 最終的に拒否するときだけ計測の記録を積む（`vals = [v, v_str, m, scale, coeff]`）。
fn lin_elem(
    vals: [f64; 5],
    what: &str,
    rescue: impl FnOnce() -> bool,
) -> Result<(), AutodiffError> {
    let [v, v_str, m, scale, coeff] = vals;
    match judge::struct_check(v, v_str, m, scale, coeff, what, invalid) {
        Ok(()) => Ok(()),
        Err(e) => {
            if rescue() {
                return Ok(());
            }
            #[cfg(test)]
            probe::check(what, vals, false);
            Err(e)
        }
    }
}

/// [`CosLinState`] が持つ seed `(side, j)` のスカラー（`Lin` 版と f32 版）と、その直後の領域の印。
type CosLinSeed = (usize, usize, LinMark, CosSeed<Lin>, CosSeed<f32>);

/// Cosine の 1 行の [`Lin`] 判定の状態。forward は行で最初に必要になったとき 1 回だけ作り、seed のスカラーは
/// `(side, j)` が変わるたびに作り直し、要素ごとの値は作ってすぐ [`lin_truncate`] で戻す（判定時の事象数を
/// forward ＋ seed のスカラー ＋ 要素 1 つ分に抑え、`entry_check_lin` の余裕係数の条件を満たしやすくする）。
/// `Drop` で thread-local の領域を解放する。
struct CosLinState {
    lin: [CosFwd<Lin>; 2],
    f32m: [CosFwd<f32>; 2],
    /// side ごとの `(r_own, r_other)`。
    r: [(f64, f64); 2],
    ctx: CosLinCtx,
    fwd_mark: LinMark,
    cur: Option<CosLinSeed>,
}

impl CosLinState {
    fn new(rows: (&[f32], &[f32]), inv: (f64, f64), eps: (f64, f64), ctx: CosLinCtx) -> Self {
        lin_reset();
        let f = cos_fwd::<Lin>(rows.0, rows.1, inv, eps);
        let fs = f.swapped();
        let m = cos_fwd::<f32>(rows.0, rows.1, inv, eps);
        let ms = m.swapped();
        CosLinState {
            lin: [f, fs],
            f32m: [m, ms],
            r: [(inv.0, inv.1), (inv.1, inv.0)],
            ctx,
            fwd_mark: lin_mark(),
            cur: None,
        }
    }

    fn seed(&mut self, side: usize, j: usize) -> (LinMark, CosSeed<Lin>, CosSeed<f32>) {
        if let Some((s, jj, m, a, b)) = self.cur
            && s == side
            && jj == j
        {
            return (m, a, b);
        }
        lin_truncate(self.fwd_mark);
        let (c, r) = (self.ctx, self.r[side].0);
        let a = cos_rev_seed(&self.lin[side], j, c.g, c.mean_n, c.cf, r);
        let b = cos_rev_seed(&self.f32m[side], j, c.g, c.mean_n, c.cf, r);
        let m = lin_mark();
        self.cur = Some((side, j, m, a, b));
        (m, a, b)
    }

    /// seed `(side, j)` の二階の要素（`kind`・`i`）が [`entry_check_lin`] を通るか。
    fn ok(&mut self, kind: LinKind, side: usize, j: usize, i: usize, what: &str) -> bool {
        let (mark, a, b) = self.seed(side, j);
        let (r_own, r_other) = self.r[side];
        let ok = match kind {
            LinKind::G => entry_check_lin(a.d_g, b.d_g, what, invalid).is_ok(),
            LinKind::Own => {
                let l = cos_rev_own(&self.lin[side], &a, i, r_own);
                let s = cos_rev_own(&self.f32m[side], &b, i, r_own);
                entry_check_lin(l, s, what, invalid).is_ok()
            }
            LinKind::Cross => {
                let l = cos_rev_other(&self.lin[side], &a, i, r_other);
                let s = cos_rev_other(&self.f32m[side], &b, i, r_other);
                entry_check_lin(l, s, what, invalid).is_ok()
            }
        };
        lin_truncate(mark);
        ok
    }
}

impl Drop for CosLinState {
    fn drop(&mut self) {
        lin_reset();
    }
}
/// Cosine の 1 行の総括判定（設計 §55.2。相殺・ε の吸収・f32 の underflow）。
///
/// 構造 `inner1 = b'/den − cos·a'/m1'`（`dx2` は a と b を入れ替えた形）を、seed が 1 要素
/// （Hessian の 1 列）のときの二階の逆伝播の演算順どおりに展開し（[`CosSide`]）、各要素の f64 の参照値
/// `V`、各項の絶対値の和 `M`、構造の f32 の中間値から同じ式で組んだ `V_str` を [`struct_check`] に
/// 掛ける。展開（own は自分側のベクトル、other は相手側、`j` は seed の要素）:
/// - `cot_cos = −own_j/m_own`、`cot_D = cot_cos/den`、`cot_den = −other_j/den² + own_j·cos/(m_own·den)`
/// - `cot_m_own = cos·own_j/m_own² + cot_den·s_other/(2 s_own)`、`cot_m_other = cot_den·s_own/(2 s_other)`
/// - own への二階 `δ_ij·(−cos/m_own) + other_i·cot_D + 2 own_i·cot_m_own`
/// - other への二階 `δ_ij/den + own_i·cot_D + 2 other_i·cot_m_other`
/// - g への二階は `inner_j` そのもの（`other_j/den − cos·own_j/m_own`）
///
/// `M` では `cos` を `Σ|a'b'|/den`（内積の相殺も含める）に置き換える。平行に近い行（例 x1=[1]、x2=[1]）
/// では各項が O(1) で打ち消し合い、真の曲率は `ε'` の大きさ（`3ε/(1+ε)^3`）しか残らない。f32 では
/// `m' = Σa'² + ε'` の `ε'` が丸めで消えるため構造は 0 を返し、上流係数が大きいと REQ-2 を外れる
/// （`coeff·M` の項で検出。構造で ε を保つ方法は無い: `ε'/m' < 2^-24` の行では f32 の `m'` に ε の
/// 情報が残らない）。行スケール `a·(1/s)` で微小成分が 0 や非正規数になる行は `V_str` との差で検出する。
/// 係数は行の幅 `d` から [`rounding_coeff`]。f32 の再現値がどれも `coeff` 以内で一致し、かつ上界
/// `coeff·M·scale` が O(d) の速い経路で絶対判定を満たす行は、要素ごとの O(d²) の走査を省く。
///
/// forward の f64 参照と f32 再現は総称関数 [`cos_fwd`] で組む（#4090）。
///
/// **`Lin` の上界による救済（#4091・設計 §82.8）**: 上の見積り（`coeff·M`）が拒否した要素だけ、丸めの一次伝播
/// [`Lin`] の上界（[`CosLinState`]）で評価し直し、閾値内なら受理する（[`lin_elem`]）。どちらの見積りも独立に健全
/// なので片方が通れば REQ-2 の判定は成り立ち、従来の受理を拒否へ変えることは構造的に起きない（判定は緩めず、
/// 過剰な拒否を減らす向きだけ）。`lin = Some` はテンソル単位の条件 [`cos_use_lin`]（走査量と事象数の上限）を
/// 満たすときだけで、満たさないテンソルは従来の見積りのまま。`fast_ok` の速い経路・`flag` の行は変えていない。
fn cosine_cancel_check(
    rows: (&[f32], &[f32]),
    inv: (f64, f64),
    eps: (f64, f64),
    scales: (f64, f64, f64),
    lin: Option<CosLinCtx>,
) -> Result<(), AutodiffError> {
    let (a32, b32) = rows;
    let d = a32.len();
    let coeff = rounding_coeff(d);
    let (sb, gs, rho) = scales;
    // f64 の参照値と、構造の f32 の中間値の再現（子テープの `x·const(1/s)`・`Sum(x'²) + const(ε/s²)`・`Sqrt`・
    // `Mul`・`Div`）。どちらも同じ総称関数 [`cos_fwd`] で組む（#4090。従来は同じ式を 2 回書いていた）。
    let ref64 = cos_fwd::<f64>(a32, b32, inv, eps);
    let CosFwd {
        ap,
        bp,
        m1,
        m2,
        s1,
        s2,
        den,
        cos,
        ..
    } = ref64;
    let cos_m: f64 = ap.iter().zip(&bp).map(|(a, b)| (a * b).abs()).sum::<f64>() / den;
    let f32f = cos_fwd::<f32>(a32, b32, inv, eps);
    let (ap32, bp32) = (f32f.ap, f32f.bp);
    let (m1_32, m2_32, s1_32, s2_32) = (f32f.m1, f32f.m2, f32f.s1, f32f.s2);
    let (den32, cos32) = (f32f.den, f32f.cos);
    let to64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
    let (ap_s, bp_s) = (to64(&ap32), to64(&bp32));
    let flag = deviates(1.0, rho, 1.0, coeff)
        || ap
            .iter()
            .zip(&ap_s)
            .any(|(&r, &x)| deviates(r, x, r.abs(), coeff))
        || bp
            .iter()
            .zip(&bp_s)
            .any(|(&r, &x)| deviates(r, x, r.abs(), coeff))
        || deviates(m1, f64::from(m1_32), m1, coeff)
        || deviates(m2, f64::from(m2_32), m2, coeff)
        || deviates(den, f64::from(den32), den, coeff)
        || deviates(cos, f64::from(cos32), cos_m, coeff);
    #[cfg(test)]
    probe::flag("Cosine", flag);
    let side = |own: &[f64], other: &[f64], m_own, s_own, s_other, den, cos| CosSide {
        own: own.to_vec(),
        other: other.to_vec(),
        m_own,
        s_own,
        s_other,
        den,
        cos,
    };
    let (m1s, m2s, s1s, s2s) = (
        f64::from(m1_32),
        f64::from(m2_32),
        f64::from(s1_32),
        f64::from(s2_32),
    );
    let (dens, coss) = (f64::from(den32), f64::from(cos32));
    // (参照, f32 再現, r_own, r_other)
    let sides = [
        (
            side(&ap, &bp, m1, s1, s2, den, cos),
            side(&ap_s, &bp_s, m1s, s1s, s2s, dens, coss),
            inv.0,
            inv.1,
        ),
        (
            side(&bp, &ap, m2, s2, s1, den, cos),
            side(&bp_s, &ap_s, m2s, s2s, s1s, dens, coss),
            inv.1,
            inv.0,
        ),
    ];
    let amax = |v: &[f64]| v.iter().fold(0f64, |m, x| m.max(x.abs()));
    // `Lin` の上界の状態（従来の見積りが最初に拒否した要素で 1 回だけ作る。`Drop` で領域を解放する）。
    let mut lin_state: Option<CosLinState> = None;
    let mut rescue = |kind: LinKind, side: usize, j: usize, i: usize, what: &str| -> bool {
        let Some(ctx) = lin else {
            return false;
        };
        lin_state
            .get_or_insert_with(|| CosLinState::new(rows, inv, eps, ctx))
            .ok(kind, side, j, i, what)
    };
    for (side_idx, (rf, st, r_own, r_other)) in sides.iter().enumerate() {
        let (own, other, m_own, s_own, s_other) =
            (&rf.own, &rf.other, rf.m_own, rf.s_own, rf.s_other);
        let (omax, tmax) = (amax(own), amax(other));
        // seed 要素 j ごとの cotangent の大きさ（`M` 用。`cos` を `cos_m` に置き換える）。
        let mags: Vec<[f64; 3]> = (0..d)
            .map(|j| {
                let (oj, tj) = (own[j].abs(), other[j].abs());
                let cot_d_m = oj / (m_own * den);
                let cot_den_m = tj / (den * den) + oj * cos_m / (m_own * den);
                let cot_mo_m = cos_m * oj / (m_own * m_own) + cot_den_m * s_other * 0.5 / s_own;
                let cot_mt_m = cot_den_m * s_own * 0.5 / s_other;
                [cot_d_m, cot_mo_m, cot_mt_m]
            })
            .collect();
        let max_of = |k: usize| mags.iter().fold(0f64, |m, c| m.max(c[k]));
        let s_own_blk = sb * r_own * r_own;
        let s_cross_blk = sb * r_own * r_other;
        let fast_own = cos_m / m_own + tmax * max_of(0) + 2.0 * omax * max_of(1);
        let fast_cross = 1.0 / den + omax * max_of(0) + 2.0 * tmax * max_of(2);
        let need_own = flag || !fast_ok(coeff * fast_own * s_own_blk);
        let need_cross = flag || !fast_ok(coeff * fast_cross * s_cross_blk);
        for (j, mg) in mags.iter().enumerate() {
            let [cot_d_m, cot_mo_m, cot_mt_m] = *mg;
            // g への二階（`inner_j`）。g 経路は `s_b` を通らないので `rho` は掛けない。
            let gv = rf.g_entry(j);
            const G_WHAT: &str = "Cosine: g への二階（b'/den − cos·a'/m' の差し引き）";
            lin_elem(
                [
                    gv,
                    if flag { st.g_entry(j) } else { gv },
                    other[j].abs() / den + cos_m * own[j].abs() / m_own,
                    gs * r_own,
                    coeff,
                ],
                G_WHAT,
                || rescue(LinKind::G, side_idx, j, 0, G_WHAT),
            )?;
            if !(need_own || need_cross) {
                continue;
            }
            let (cr, cs) = (rf.cots(j), st.cots(j));
            for i in 0..d {
                let delta = if i == j { 1.0 } else { 0.0 };
                if need_own {
                    let v = rf.own_entry(cr, i, j);
                    const OWN_WHAT: &str =
                        "Cosine: 同じ側の二階（ε の吸収・平行に近い行の相殺・f32 の underflow）";
                    lin_elem(
                        [
                            v,
                            if flag {
                                rho * st.own_entry(cs, i, j)
                            } else {
                                v
                            },
                            delta * cos_m / m_own
                                + other[i].abs() * cot_d_m
                                + 2.0 * own[i].abs() * cot_mo_m,
                            s_own_blk,
                            coeff,
                        ],
                        OWN_WHAT,
                        || rescue(LinKind::Own, side_idx, j, i, OWN_WHAT),
                    )?;
                }
                if need_cross {
                    let v = rf.cross_entry(cr, i, j);
                    const CROSS_WHAT: &str =
                        "Cosine: 相手側の二階（ε の吸収・平行に近い行の相殺・f32 の underflow）";
                    lin_elem(
                        [
                            v,
                            if flag {
                                rho * st.cross_entry(cs, i, j)
                            } else {
                                v
                            },
                            delta / den + own[i].abs() * cot_d_m + 2.0 * other[i].abs() * cot_mt_m,
                            s_cross_blk,
                            coeff,
                        ],
                        CROSS_WHAT,
                        || rescue(LinKind::Cross, side_idx, j, i, CROSS_WHAT),
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

/// Triplet の 1 ペア・1 行について、二階の展開に使う量（`pair_grad` の構造に対応）。
/// `u = v·inv`、`N = ‖u‖_p`、`q = |u|/N`、`e = q^(p−1)`、対角係数 `dg = (p−1)·q^(p−2)`、
/// `fac = inv/N`。符号は 3 つに分ける（構造の子テープが f32 で決める符号と、ホストの V2 定数の符号が
/// 経路ごとに異なるため）:
/// - `sg`: 勾配 `core_j` の符号。p=2 の構造 `u/norm'` は f32 の `u` の符号、一般の p は V2 定数
///   `sign_c`（ホスト）。
/// - `sa`: 入力レーン i へ逆伝播するときの符号。p=2 は f32 の `u` の符号（`u/norm'` の `u`）、一般の p は
///   `Abs` の VJP が f32 の `u` から決める符号（`sign(0) = 0`）。
/// - `dsg`: 対角の δ 項に掛かる符号（p=2 は `∂(u_j/N)/∂u_j = 1/N` で符号なしの 1、一般の p は
///   `sg_j·sa_j`）。
///
/// 参照（f64）は 3 つともホストの差分の符号から作る（真の関数の微分）。
struct PairRow {
    fac: f64,
    pm1: f64,
    dg: Vec<f64>,
    e: Vec<f64>,
    sg: Vec<f64>,
    sa: Vec<f64>,
    dsg: Vec<f64>,
}

/// 符号 `s`（`sign0` の結果。`+1`／`-1`／`0`）を掛けた値。`s > 0` は恒等、`s < 0` は符号反転、`s == 0` は定数 0。
/// `Num` に `abs` が無いため、`|x|`（符号が分かっている `x` の絶対値）と、ホスト定数の符号 `sign_c` の乗算の
/// 両方をこれで書く（`f32` の積は符号に対称に丸めるので `x·s` と bit 一致する。underflow した `±0` も `|x|` と同じ `+0` になる）。
/// 呼び出し元: [`tri_fwd`]・[`tri_pair_seed`]・[`tri_pair_lane`]（#4095 で導入、#4096 で本番の判定が使う）。
fn by_sign<T: Num>(x: T, s: f64) -> T {
    if s > 0.0 {
        x
    } else if s < 0.0 {
        x.negate()
    } else {
        T::k(0.0)
    }
}

/// Triplet の 1 ペア・1 行の forward の中間値（`u = v·inv`・`ssum`・`N`・`q = |u|/N`・`e = q^(p−1)`・`dg = (p−1)·q^(p−2)`）。
/// `T` は `f64`（参照）・`f32`（子テープの構造の再現）・[`Lin`]（丸めの一次伝播の上界。#4095 で導入、#4096 で [`TriLinState`] が判定に使う）。
struct TriFwd<T> {
    u: Vec<T>,
    /// `|u|`（[`by_sign`] で作る）。二階の逆伝播のモデル（[`tri_pair_seed`]・[`tri_pair_lane`]）が読む。
    au: Vec<T>,
    /// p=2 は `Σu²`、一般の p は `Σ|u|^p`（構造の `ssum`）。二階の逆伝播のモデルが読む。
    ssum: T,
    norm: T,
    q: Vec<T>,
    e: Vec<T>,
    dg: Vec<T>,
}

/// [`TriFwd`] を `T` の演算で組む。`v`＝差分 `(x−y)+eps`（`T` ごとの値。本番は `f64` が `triplet_distance_stats` の差分、
/// `f32` が `row32` の再構成、`Lin` が [`tri_v`] で `minus`／`plus` の丸めの事象つきに作る）、`signs`＝各レーンの
/// 差分の符号（`|u|` の組み立てに使う。`f64` は `sign0(h)`、`f32`・`Lin` は `sign0(rec)`）、`inv32`＝行スケールの逆数
/// （f32 で厳密）、`p`＝`(p64, p32)`。
///
/// 呼び出し元: [`triplet_pair_row`]（`f64` の参照と `f32` の再現）と [`TriLinState`]（`Lin` と `f32` の並行計算。#4096）。
/// 演算順は従来の `triplet_pair_row` の逐語の式と同じ（bit 一致は test の `triplet_pair_row_generic_matches_previous_formulas` が錠）。
/// 指数は `e as f32` に頼らず `p32` から f32 で作った値（`1.0/p32`・`p32-1.0`・`p32-2.0`）を明示して渡す。
fn tri_fwd<T: Num>(v: &[T], signs: &[f64], inv32: f32, p: (f64, f32)) -> TriFwd<T> {
    let (p64, p32) = p;
    let inv = T::k(f64::from(inv32));
    let u: Vec<T> = v.iter().map(|&x| x.times(inv)).collect();
    let au: Vec<T> = u.iter().zip(signs).map(|(&x, &s)| by_sign(x, s)).collect();
    let (ssum, norm) = if p64 == 2.0 {
        let sq: Vec<T> = u.iter().map(|&x| x.times(x)).collect();
        let ssum = T::total(&sq);
        (ssum, ssum.root())
    } else {
        let pw: Vec<T> = au.iter().map(|&x| x.powf_bounded(p64, p32)).collect();
        let ssum = T::total(&pw);
        (ssum, ssum.powf_bounded(1.0 / p64, 1.0 / p32))
    };
    let q: Vec<T> = au.iter().map(|&x| x.over(norm)).collect();
    let (e, dg) = if p64 == 2.0 {
        (q.clone(), vec![T::k(1.0); q.len()])
    } else {
        (
            q.iter()
                .map(|&x| x.powf_bounded(p64 - 1.0, p32 - 1.0))
                .collect(),
            q.iter()
                .map(|&x| T::k(p64 - 1.0).times(x.powf_bounded(p64 - 2.0, p32 - 2.0)))
                .collect(),
        )
    };
    TriFwd {
        u,
        au,
        ssum,
        norm,
        q,
        e,
        dg,
    }
}

/// 1 ペア・1 行の f64 の参照値と、構造が f32 で計算する値（差分の再構成 `rec`・`v·(1/mx)`・`|u|^p` の
/// 逐次和・`ssum^(1/p)`・`|u|/N`・`PowScalar` とその微分係数）の再現を作る。3 つ目は f32 の再現値が
/// 丸め誤差の見積り `coeff` を超えてずれる（スケーリング後の微小成分の 0 化・非正規数化等）か。
/// 全要素 0 の行（構造は定数 0）・NaN の行は `None`。forward は総称関数 [`tri_fwd`] で組む（#4095。判定は不変）。
fn triplet_pair_row(
    h: &[f64],
    rec: &[f32],
    info: &PairInfo,
    base: usize,
    p: f64,
    coeff: f64,
) -> Option<(PairRow, PairRow, bool)> {
    if info.zero[base] || h.iter().any(|v| v.is_nan()) {
        return None;
    }
    let inv32 = info.inv_mx[base];
    let inv = f64::from(inv32);
    let p32 = p as f32;
    let signs64: Vec<f64> = h.iter().map(|&x| sign0(x)).collect();
    let f64f = tri_fwd::<f64>(h, &signs64, inv32, (p, p32));
    let norm = f64f.norm;
    if norm.is_nan() || norm <= 0.0 {
        return None;
    }
    let sg: Vec<f64> = (0..h.len())
        .map(|k| f64::from(info.sign[base + k]))
        .collect();
    let reference = PairRow {
        fac: inv / norm,
        pm1: p - 1.0,
        dg: f64f.dg.clone(),
        e: f64f.e.clone(),
        sg: sg.clone(),
        sa: sg.clone(),
        dsg: vec![1.0; h.len()],
    };
    // 構造の f32 の再現（`pair_grad`: `u = v·const(1/mx)`、p=2 は `u/sqrt(Σu²)`、一般の p は
    // `(|u|/ssum^(1/p))^(p−1)`）。
    let signs32: Vec<f64> = rec.iter().map(|&x| sign0(f64::from(x))).collect();
    let f32f = tri_fwd::<f32>(rec, &signs32, inv32, (p, p32));
    let (u, u32v, norm32) = (&f64f.u, &f32f.u, f32f.norm);
    let e32: Vec<f64> = f32f.e.iter().map(|&x| f64::from(x)).collect();
    let dg32: Vec<f64> = f32f.dg.iter().map(|&x| f64::from(x)).collect();
    // 構造が f32 の `u` から決める符号（離散量もホストの値を共有せず、構造の実際の経路から取る）。
    let s32: Vec<f64> = u32v.iter().map(|&x| sign0(f64::from(x))).collect();
    let (sg_s, dsg_s) = if p == 2.0 {
        (s32.clone(), vec![1.0; h.len()])
    } else {
        let d: Vec<f64> = sg.iter().zip(&s32).map(|(a, b)| a * b).collect();
        (sg.clone(), d)
    };
    let structural = PairRow {
        fac: inv / f64::from(norm32),
        pm1: f64::from(p32) - 1.0,
        dg: dg32,
        e: e32,
        sg: sg_s,
        sa: s32,
        dsg: dsg_s,
    };
    let flag = deviates(norm, f64::from(norm32), norm, coeff)
        || (0..h.len()).any(|k| {
            deviates(u[k], f64::from(u32v[k]), u[k].abs(), coeff)
                || deviates(reference.e[k], structural.e[k], reference.e[k], coeff)
                || deviates(
                    reference.dg[k],
                    structural.dg[k],
                    reference.dg[k].abs(),
                    coeff,
                )
                || structural.sa[k] != reference.sa[k]
        });
    Some((reference, structural, flag))
}

// ---------------------------------------------------------------------------------
// Triplet の二階の逆伝播のモデル（#4095 で test 限定として導入、#4096 で本番の判定が使う。設計 §82.12・§82.13）
// ---------------------------------------------------------------------------------

/// Triplet の 1 ケース（`[1, d]` の 1 行・全要素 active）の入力と、本番の前段（`pair_info`）が作る離散量。
/// 二階の逆伝播のモデル（[`tri_seed`]・[`tri_x`]）が読む。全要素 0 の行・符号が食い違う行は作れない
/// （[`tri_setup_row`] が `None`）。
pub(super) struct TriSetup {
    /// ペア `ap`・`an`・`pn` の `(x, y)`（差分 `v = (x − y) + eps`。`pn` は swap のときだけ使う）。
    pub(super) rows: [(Vec<f32>, Vec<f32>); 3],
    pub(super) eps: f32,
    pub(super) p: f32,
    pub(super) swap: bool,
    /// 上流 `(g, Mean の分母)`。
    pub(super) up: (f64, Option<usize>),
    inv32: [f32; 3],
    /// ホスト定数 `sign_c`（f64 の差分の符号）。
    sg: [Vec<f64>; 3],
    /// f32 の差分 `rec` の符号（`|u|` の組み立てに使う）。
    s_abs: [Vec<f64>; 3],
    /// f32 の `u = rec·inv` の符号（`Abs` の VJP が掛ける符号）。
    sa: [Vec<f64>; 3],
    /// `an_coeff` の分類: 0 = 1（`m_one`）、1 = 0（`m_zero`）、2 = 中間（`m_half`）。
    an_kind: u8,
}

impl TriSetup {
    /// `Lin` の救済に使ってよい行か。一般の p では二階の逆伝播が f32 の `u = rec·inv` の符号（`Abs` の VJP の `sa`）で
    /// 分岐する。微小成分が `u` で 0 へ underflow すると `sa = 0` になり、モデルは離散量ごと「その成分の曲率が 0」の
    /// 値を作る（`Lin.v` も 0 になって f32 の値と一致してしまう）。真の曲率は f64 の符号 `sg` の側で非ゼロなので、
    /// `Lin` の上界は実際のずれを覆えない。そのため `sa ≠ sg` のレーンがある行は救済しない（従来の判定が残る。
    /// p=2 の構造は `sa` を使わず `u/N` が連続なので対象外。設計 §82.13.1）。
    fn rescuable(&self) -> bool {
        self.p == 2.0 || (0..if self.swap { 3 } else { 2 }).all(|k| self.sa[k] == self.sg[k])
    }
}

/// `an_coeff`（swap でなければ 1）の分類（[`TriSetup`] の `an_kind`）。
fn an_kind_of(c_an: f64) -> u8 {
    if c_an >= 1.0 {
        0
    } else if c_an <= 0.0 {
        1
    } else {
        2
    }
}

/// 行の入力と `pair_info` の結果から [`TriSetup`] を作る（本番の判定〈[`TriLinState`]〉とテストの `triplet_setup` が共有）。
/// `rows = (a, p, n)` は幅 `d` の行、`infos[k]` はペア `k`（0 = ap、1 = an、2 = pn。pn は swap のときだけ）の `pair_info` の
/// 結果を `base` から読む。次の行は `None`（`Lin` の救済をしない。従来の判定がそのまま残る。設計 §82.13.1。
/// `u` の underflow による符号の食い違いは [`TriSetup::rescuable`] が別に見る）:
/// - 生きているペアの `info.zero`（全要素 0。従来はそのペアの寄与を除外する）
/// - 差分の f32 再構成の符号が f64 の差分の符号と食い違うレーンがある（`|u|` の一次の写しが健全でなくなるため。
///   本番ではその食い違いを `flag`〈`sa` の不一致〉が検出する）
fn tri_setup_row(
    rows: (&[f32], &[f32], &[f32]),
    opts: (f32, f32, bool),
    up: (f64, Option<usize>),
    infos: [Option<&PairInfo>; 3],
    base: usize,
    an_kind: u8,
) -> Option<TriSetup> {
    let (eps, p, swap) = opts;
    let d = rows.0.len();
    let pairs: [(&[f32], &[f32]); 3] = [(rows.0, rows.1), (rows.0, rows.2), (rows.1, rows.2)];
    let mut inv32 = [1f32; 3];
    let mut sg: [Vec<f64>; 3] = Default::default();
    let mut s_abs: [Vec<f64>; 3] = Default::default();
    let mut sa: [Vec<f64>; 3] = Default::default();
    for k in 0..if swap { 3 } else { 2 } {
        let info = infos[k]?;
        if info.zero[base] {
            return None;
        }
        let (x, y) = pairs[k];
        let rec: Vec<f32> = (0..d).map(|i| (x[i] - y[i]) + eps).collect();
        s_abs[k] = rec.iter().map(|&c| sign0(f64::from(c))).collect();
        sg[k] = info.sign[base..base + d]
            .iter()
            .map(|&s| f64::from(s))
            .collect();
        if s_abs[k] != sg[k] {
            return None;
        }
        inv32[k] = info.inv_mx[base];
        sa[k] = rec
            .iter()
            .map(|&c| sign0(f64::from(c * info.inv_mx[base])))
            .collect();
    }
    Some(TriSetup {
        rows: [
            (rows.0.to_vec(), rows.1.to_vec()),
            (rows.0.to_vec(), rows.2.to_vec()),
            (rows.1.to_vec(), rows.2.to_vec()),
        ],
        eps,
        p,
        swap,
        up,
        inv32,
        sg,
        s_abs,
        sa,
        an_kind,
    })
}

/// [`TriSetup`] を作る（`triplet_cgrad` の前段と同じ `triplet_distance_stats`・`pair_info`。test 限定）。
/// 前段で拒否される入力・全要素 0 の行・NaN の行・符号が食い違う行は `None`。
#[cfg(test)]
pub(super) fn triplet_setup(
    rows: (&[f32], &[f32], &[f32]),
    eps: f32,
    p: f32,
    swap: bool,
    up: (f64, Option<usize>),
) -> Option<TriSetup> {
    let d = rows.0.len();
    let to = |s: &[f32]| -> Vec<f64> { s.iter().map(|&v| f64::from(v)).collect() };
    let (a64, p64v, n64) = (to(rows.0), to(rows.1), to(rows.2));
    let p64 = f64::from(p);
    let sb = match up.1 {
        Some(n) => up.0.abs() / n as f64,
        None => up.0.abs(),
    };
    let params = crate::eval::TripletDistanceParams {
        p: p64,
        eps: f64::from(eps),
        swap,
    };
    let (stats, diff_ap, diff_an, diff_pn) =
        crate::eval::triplet_distance_stats(&a64, &p64v, &n64, 0, d, &params);
    let pairs: [(&[f32], &[f32]); 3] = [(rows.0, rows.1), (rows.0, rows.2), (rows.1, rows.2)];
    let diffs = [diff_ap, diff_an, diff_pn];
    let mut infos: [Option<PairInfo>; 3] = [None, None, None];
    for k in 0..if swap { 3 } else { 2 } {
        let (x, y) = pairs[k];
        let rec: Vec<f32> = (0..d).map(|i| (x[i] - y[i]) + eps).collect();
        if diffs[k].iter().any(|v| v.is_nan()) {
            return None;
        }
        infos[k] = Some(
            pair_info(
                std::slice::from_ref(&diffs[k]),
                std::slice::from_ref(&rec),
                d,
                p64,
                sb,
            )
            .ok()?,
        );
    }
    let refs = [infos[0].as_ref(), infos[1].as_ref(), infos[2].as_ref()];
    tri_setup_row(
        rows,
        (eps, p, swap),
        up,
        refs,
        0,
        an_kind_of(stats.an_coeff),
    )
}

/// 差分 `v = (x − y) + eps` を `T` の演算で作る（`Lin` は `Sub`・`Add` の丸めの事象として差分の桁落ち〈条件数 κ〉を運ぶ）。
fn tri_v<T: Num>(x: &[f32], y: &[f32], eps: f32) -> Vec<T> {
    x.iter()
        .zip(y)
        .map(|(&a, &b)| {
            T::input(f64::from(a))
                .minus(T::input(f64::from(b)))
                .plus(T::k(f64::from(eps)))
        })
        .collect()
}

/// 1 ペアの p ノルム勾配 `core = pair_grad(v)` の二階の逆伝播のうち、seed 要素 `j` で決まりレーン `k` に依らない部分
/// （p=2 は `d_u_div = up/N`・`d_ssum`、一般の p は `d_au_div`・`d_ssum`）。[`tri_pair_lane`] がレーンごとの値を作る。
#[derive(Clone, Copy)]
struct TriPairSeed<T> {
    d_div: T,
    d_ssum: T,
}

/// `PowScalar` の係数 `e·x^(e−1)`（実カーネルは指数も f32 で `e_str − 1.0`）。
fn tri_pgrad<T: Num>(x: T, e: f64, e_str: f32) -> T {
    T::k(e).times(x.powf_bounded(e - 1.0, e_str - 1.0))
}

/// 1 ペアの p ノルム勾配 `core = pair_grad(v)` の二階の逆伝播を `T` の演算で写す（`grad.rs::vjp` の eager VJP の丸めの位置に
/// 合わせる）。出力要素 `j` に上流 `up` を置いたときの seed 部（レーンに依らない量）を返す。
///
/// 写し方（`eval/scalar.rs::binary_partials`・`unary_grad_factor`）: `Mul` の VJP は `上流·相手の値`、`Div` の係数は
/// `1/b` と `−(a/b)/b` を先に作ってから上流を掛ける、`PowScalar` の係数は f32 の `e·x^(e−1)`（指数は `e_str` を明示。
/// `e − 1` も f32 で `e_str − 1.0`）、`Abs` の VJP は f32 の `u` の符号、`Sqrt` の係数は `0.5/y`。`norm`（`ssum`）は
/// broadcast された値で、VJP は lane の総和だが上流は lane `j` だけなので 1 項。ホスト定数 `sign_c` との積・`Abs` の符号は
/// [`by_sign`]。`u = v·inv` の VJP は `times(inv)`、`v = (x−y)+eps` の VJP は ±1 で厳密。
///
/// 注意: 実カーネルの `PowScalar` の係数は `e·x^((p−1)−1)`（`e_str − 1.0`）で、本番の再現 `dg`〈`p32 − 2.0`〉とは指数の
/// f32 の丸めが違いうる。本関数はカーネルどおりに写す（本番の式は判定不変のため変えない。設計 §82.12）。
fn tri_pair_seed<T: Num>(
    f: &TriFwd<T>,
    sg: &[f64],
    p: (f64, f32),
    j: usize,
    up: T,
) -> TriPairSeed<T> {
    let (p64, p32) = p;
    let one = T::k(1.0);
    if p64 == 2.0 {
        let d_u_div = up.times(one.over(f.norm));
        let d_s = up.times(f.u[j].over(f.norm).negate().over(f.norm));
        let d_ssum = d_s.times(f.norm.root_grad());
        return TriPairSeed {
            d_div: d_u_div,
            d_ssum,
        };
    }
    let d_e = by_sign(up, sg[j]);
    let d_q = d_e.times(tri_pgrad(f.q[j], p64 - 1.0, p32 - 1.0));
    let d_au_div = d_q.times(one.over(f.norm));
    let d_norm = d_q.times(f.q[j].negate().over(f.norm));
    let d_ssum = d_norm.times(tri_pgrad(f.ssum, 1.0 / p64, 1.0 / p32));
    TriPairSeed {
        d_div: d_au_div,
        d_ssum,
    }
}

/// [`tri_pair_seed`] の seed `j` に対する、差分 `v` のレーン `k` への cotangent（レーン部）。`sa`＝`Abs` の VJP の符号。
/// `jk = (j, k)`。
fn tri_pair_lane<T: Num>(
    f: &TriFwd<T>,
    sd: &TriPairSeed<T>,
    sa: &[f64],
    inv32: f32,
    p: (f64, f32),
    jk: (usize, usize),
) -> T {
    let (p64, p32) = p;
    let (j, k) = jk;
    let inv = T::k(f64::from(inv32));
    if p64 == 2.0 {
        let t = sd.d_ssum.times(f.u[k]);
        let dv = if k == j {
            T::total(&[t, t, sd.d_div])
        } else {
            T::total(&[t, t])
        };
        return dv.times(inv);
    }
    let pow_term = sd.d_ssum.times(tri_pgrad(f.au[k], p64, p32));
    let d_au = if k == j {
        T::total(&[sd.d_div, pow_term])
    } else {
        pow_term
    };
    by_sign(d_au, sa[k]).times(inv)
}

/// [`TriSetup`] の forward を `T` で組む（ペアごとの [`TriFwd`]。swap でなければ 2 本）。
fn tri_setup_fwds<T: Num>(st: &TriSetup) -> Vec<TriFwd<T>> {
    let p = (f64::from(st.p), st.p);
    (0..if st.swap { 3 } else { 2 })
        .map(|k| {
            let (x, y) = &st.rows[k];
            tri_fwd(&tri_v::<T>(x, y, st.eps), &st.s_abs[k], st.inv32[k], p)
        })
        .collect()
}

/// ペア `k` の勾配 `core_j`（`pair_grad` の出力。p=2 は `u_j/N`、一般の p は `sign_c_j·e_j`）。
fn tri_core<T: Num>(st: &TriSetup, f: &TriFwd<T>, k: usize, j: usize) -> T {
    if st.p == 2.0 {
        f.u[j].over(f.norm)
    } else {
        by_sign(f.e[j], st.sg[k][j])
    }
}

/// 出力 `o`（0 = da、1 = dp、2 = dn）の要素 `j` を seed にした二階の seed 部。
#[derive(Clone, Copy)]
struct TriSeed<T> {
    /// ペア別の seed 部（`None` は勾配が流れない、または swap でないときの pn）。
    pairs: [Option<TriPairSeed<T>>; 3],
    /// `g` への二階 `W_o[j]`（Mean は `Div` の係数 `1/n` を f32 で先に作って掛ける）。
    g: T,
}

/// 出力 `o` の要素 `j` を seed にした二階の seed 部（`triplet_cgrad` の構造の eager 逆伝播の写し）。上流 `s_b`（Sum は `g`、
/// Mean は f32 の `g/n`）、`swap` では `ean`・`epn` の mask（`m_one`／`m_half`〈`half` の `Mul`〉／`m_zero`）を通る。
fn tri_seed<T: Num>(st: &TriSetup, f: &[TriFwd<T>], o: usize, j: usize) -> TriSeed<T> {
    let p = (f64::from(st.p), st.p);
    let s = match st.up.1 {
        Some(n) => T::input(st.up.0).over(T::k(n as f64)),
        None => T::input(st.up.0),
    };
    let sn = s.negate();
    let half = T::k(0.5);
    let core = |k: usize| tri_core(st, &f[k], k, j);
    // ペア別の上流（`None` は勾配が流れない）と、forward の `W_o[j]`。
    let (ups, w): ([Option<T>; 3], T) = if !st.swap {
        match o {
            0 => ([Some(s), Some(sn), None], core(0).minus(core(1))),
            1 => ([Some(sn), None, None], core(0).negate()),
            _ => ([None, Some(s), None], core(1)),
        }
    } else {
        // 出力 → (g_ap, ean, epn) の上流。
        let (u_ap, u_ean, u_epn) = match o {
            0 => (Some(s), Some(sn), None),
            1 => (Some(sn), None, Some(sn)),
            _ => (None, Some(s), Some(s)),
        };
        let gate = |u: Option<T>, active: bool, mid: bool| -> Option<T> {
            if active {
                u
            } else if mid {
                u.map(|x| x.times(half))
            } else {
                None
            }
        };
        let (one_k, zero_k, half_k) = (st.an_kind == 0, st.an_kind == 1, st.an_kind == 2);
        let fwd_gate = |g: T, active: bool, mid: bool| -> T {
            if active {
                g
            } else if mid {
                g.times(half)
            } else {
                T::k(0.0)
            }
        };
        let ean = fwd_gate(core(1), one_k, half_k);
        let epn = fwd_gate(core(2), zero_k, half_k);
        let w = match o {
            0 => core(0).minus(ean),
            1 => core(0).negate().minus(epn),
            _ => ean.plus(epn),
        };
        (
            [
                u_ap,
                gate(u_ean, one_k, half_k),
                gate(u_epn, zero_k, half_k),
            ],
            w,
        )
    };
    let pairs = [0, 1, 2].map(|k| {
        if k >= f.len() {
            return None;
        }
        ups[k].map(|up| tri_pair_seed(&f[k], &st.sg[k], p, j, up))
    });
    let g = match st.up.1 {
        Some(n) => w.times(T::k(1.0).over(T::k(n as f64))),
        None => w,
    };
    TriSeed { pairs, g }
}

/// seed `(o, j)` の入力 `wrt`（0 = a、1 = p、2 = n）のレーン `i` への二階 1 要素（レーン部。`ij = (i, j)`）。a は ap・an の `+`、
/// p は ap の `−` と pn の `+`、n は an・pn の `−`。寄与が 2 つ以上なら [`Num::total`]、なければ 0。
fn tri_x<T: Num>(
    st: &TriSetup,
    f: &[TriFwd<T>],
    sd: &TriSeed<T>,
    wrt: usize,
    ij: (usize, usize),
) -> T {
    let p = (f64::from(st.p), st.p);
    let (i, j) = ij;
    let pick = |k: usize, neg: bool| -> Option<T> {
        sd.pairs[k].map(|s| {
            let v = tri_pair_lane(&f[k], &s, &st.sa[k], st.inv32[k], p, (j, i));
            if neg { v.negate() } else { v }
        })
    };
    let terms: [Option<T>; 2] = match wrt {
        0 => [pick(0, false), pick(1, false)],
        1 => [pick(0, true), pick(2, false)],
        _ => [pick(1, true), pick(2, true)],
    };
    let t: Vec<T> = terms.into_iter().flatten().collect();
    match t.len() {
        0 => T::k(0.0),
        1 => t[0],
        _ => T::total(&t),
    }
}

/// 出力 `o` の要素 `j` を seed にした二階の全要素（test 限定。[`tri_seed`]・[`tri_x`] の合成）。戻り値は
/// `[wrt(a, p, n, g)]` の行（g は 1 要素）。#4095 の soundness 掃引と bit 一致の錠が使う。
#[cfg(test)]
fn tri_rev<T: Num>(st: &TriSetup, f: &[TriFwd<T>], o: usize, j: usize) -> Vec<Vec<T>> {
    let d = f[0].u.len();
    let sd = tri_seed(st, f, o, j);
    let mut out: Vec<Vec<T>> = (0..3)
        .map(|wrt| (0..d).map(|i| tri_x(st, f, &sd, wrt, (i, j))).collect())
        .collect();
    out.push(vec![sd.g]);
    out
}

/// [`tri_rev`] を `f32` で評価した二階（`[o][wrt(a, p, n, g)]`。seed 要素 `j`）。構造の演算の写しで、手組みの構造の実測と並べて
/// soundness の「実測」側に使う（test 限定）。
#[cfg(test)]
pub(super) fn triplet_model_rows_f32(st: &TriSetup, j: usize) -> [Vec<Vec<f32>>; 3] {
    let f = tri_setup_fwds::<f32>(st);
    [0, 1, 2].map(|o| tri_rev(st, &f, o, j))
}

/// [`tri_rev`] の `Lin` 版の結果: `[o][wrt]` の各要素の `(参照値 v, 一次の上界 bound)` と、そのときの事象の総数。
#[cfg(test)]
pub(super) struct TriLinRows {
    pub(super) rows: [Vec<Vec<(f64, f64)>>; 3],
    pub(super) events: usize,
}

/// `Lin` で forward と seed ごとの逆伝播を評価し、各要素の値と上界を返す（`[seed]`。test 限定。#4095 の soundness 掃引・計測用）。
/// 厳密モード（[`lin_reset`]）で forward を 1 回作り、seed ごとに [`lin_mark`]／[`lin_truncate`] で領域を戻す。`Num::settle` と走査量の
/// 上限は入れない（本番の判定は [`TriLinState`]。#4096）。
#[cfg(test)]
pub(super) fn triplet_lin_rows(st: &TriSetup, seeds: &[usize]) -> Vec<TriLinRows> {
    use super::lin::lin_events;
    lin_reset();
    let f = tri_setup_fwds::<Lin>(st);
    let mark = lin_mark();
    let out = seeds
        .iter()
        .map(|&j| {
            let rows = [0, 1, 2].map(|o| {
                tri_rev(st, &f, o, j)
                    .iter()
                    .map(|r| r.iter().map(|l| (l.v, l.bound())).collect())
                    .collect()
            });
            let r = TriLinRows {
                rows,
                events: lin_events(),
            };
            lin_truncate(mark);
            r
        })
        .collect();
    lin_reset();
    out
}

/// Triplet の `Lin` 判定の入力（行に依らない量）。`g` は子テープ上の上流（符号つきの f32 の値）、`mean_n` は Mean の分母、
/// `x` は 9 ブロックの要素の救済走査（O(9·d³)）を許すか（[`tri_lin_ctx`] が走査量の上限で決める。`g` への二階の救済は
/// `Some` なら常に許す）。
#[derive(Clone, Copy)]
struct TriLinCtx {
    g: f64,
    mean_n: Option<usize>,
    x: bool,
}

/// 事象の総数の上界（forward のペアごとの丸めと seed のスカラー、要素 1 つ分）。実測と固定は
/// `loss_class_tests.rs::triplet_lin_events_stay_within_bound`（設計 §82.13）。
fn tri_lin_events_max(d: usize, swap: bool) -> usize {
    let pairs = if swap { 3 } else { 2 };
    TRI_LIN_EVENTS_PER_PAIR_LANE
        .saturating_mul(pairs)
        .saturating_mul(d)
        .saturating_add(TRI_LIN_EVENTS_FIXED)
}

/// [`tri_lin_events_max`] のペア 1 本・レーン 1 つあたりの事象数（差分・`u`・`|u|^p`・`q`・`e`・`dg` の丸め）。
const TRI_LIN_EVENTS_PER_PAIR_LANE: usize = 8;
/// [`tri_lin_events_max`] の定数分（`ssum`・`norm`、seed のスカラー、`Mean` の `n`、要素 1 つ分、swap の mask）。
const TRI_LIN_EVENTS_FIXED: usize = 120;

/// 1 行の `g` への二階（forward ＋ 全 `(o, j)` の `g` の要素）の走査量の見積り（`Rf` の 1 反復相当。`LIN_BUDGET` と同じ単位）。
/// forward のハブ値（`ssum`・`norm`）の係数列が O(d) なので forward は O(pairs·d²)、`g` の要素は 3d 個で、seed のスカラー
/// （ハブ値に依る演算）がそれぞれ O(pairs·d)。実測は設計 §82.13.2。
fn tri_lin_cost_g(d: usize, swap: bool) -> usize {
    let pairs = if swap { 3 } else { 2 };
    TRI_LIN_COST_FWD
        .saturating_add(TRI_LIN_COST_G.saturating_mul(3))
        .saturating_mul(pairs)
        .saturating_mul(d)
        .saturating_mul(d)
        .saturating_add(TRI_LIN_COST_ROW)
}

/// 9 ブロックの全 `9d²` 要素の救済走査の走査量の見積り。要素あたりの `Lin` 演算が O(d)（ハブ値の係数列を引く）。
fn tri_lin_cost_x(d: usize) -> usize {
    TRI_LIN_COST_X
        .saturating_mul(9)
        .saturating_mul(d)
        .saturating_mul(d)
        .saturating_mul(d)
        .saturating_add(TRI_LIN_COST_ROW)
}

/// forward のペア 1 本・`d²` あたり。
const TRI_LIN_COST_FWD: usize = 30;
/// `g` の要素 1 つ（seed のスカラー）のペア 1 本・`d` あたり。
const TRI_LIN_COST_G: usize = 18;
/// 9 ブロックの要素 1 つの `d` あたり。
const TRI_LIN_COST_X: usize = 10;
/// 行ごとの固定分（小さい `d` の下駄）。
const TRI_LIN_COST_ROW: usize = 3000;

/// Triplet の総括判定で [`Lin`] の上界を使うか、使うなら 9 ブロックの要素の救済走査まで許すか（テンソル単位。
/// `ce_lin_ctx` と同じ 2 段階）。事象の総数が余裕係数の上限（`16·事象数·u < 0.01`）に収まり、かつ `g` の走査量が
/// [`LIN_BUDGET`] 以内のときだけ `Some`。要素まで含めた走査量も収まるときだけ `x = true`。満たさないテンソルは従来の
/// 見積りだけで判定する（受理集合は従来と同じ。走査量の上限を超えたら `Lin` を使わない形で守る。設計 §82.13）。
/// `n_rows` は判定する活性な行の数。
fn tri_lin_ctx(
    g: f64,
    mean_n: Option<usize>,
    n_rows: usize,
    d: usize,
    swap: bool,
) -> Option<TriLinCtx> {
    #[cfg(test)]
    if probe::FORCE_OLD.with(|f| f.get()) {
        return None;
    }
    if 16.0 * tri_lin_events_max(d, swap) as f64 * LIN_U >= 0.01 {
        return None;
    }
    let cg = n_rows.saturating_mul(tri_lin_cost_g(d, swap));
    if cg > LIN_BUDGET {
        return None;
    }
    let x = cg.saturating_add(n_rows.saturating_mul(tri_lin_cost_x(d))) <= LIN_BUDGET;
    Some(TriLinCtx { g, mean_n, x })
}

/// [`TriLinState`] が持つ seed `(o, j)` のスカラー（`Lin` 版と f32 版）と、その直後の領域の印。
type TriLinSeed = (usize, usize, LinMark, TriSeed<Lin>, TriSeed<f32>);

/// Triplet の 1 行の [`Lin`] 判定の状態。forward は行で最初に必要になったとき 1 回だけ作り、seed のスカラーは
/// `(o, j)` が変わるたびに作り直し、要素ごとの値は作ってすぐ [`lin_truncate`] で戻す（判定時の事象数を forward ＋ seed の
/// スカラー ＋ 要素 1 つ分に抑え、`entry_check_lin` の余裕係数の条件を満たしやすくする。`entry_check_lin` は判定時点の
/// 全体の事象数を余裕に使うので、要素ごとの truncate は必須）。`Drop` で thread-local の領域を解放する。
/// `f32` 側はカーネルどおりに写したモデル（`dg32` の `p32 − 2.0` は使わない）なので `Lin` の健全性は指数の食い違いに依らない。
struct TriLinState {
    st: TriSetup,
    lin: Vec<TriFwd<Lin>>,
    f32m: Vec<TriFwd<f32>>,
    fwd_mark: LinMark,
    cur: Option<TriLinSeed>,
}

impl TriLinState {
    fn new(st: TriSetup) -> Self {
        lin_reset();
        let lin = tri_setup_fwds::<Lin>(&st);
        let f32m = tri_setup_fwds::<f32>(&st);
        TriLinState {
            st,
            lin,
            f32m,
            fwd_mark: lin_mark(),
            cur: None,
        }
    }

    fn seed(&mut self, o: usize, j: usize) -> (LinMark, TriSeed<Lin>, TriSeed<f32>) {
        if let Some((oo, jj, m, a, b)) = self.cur
            && oo == o
            && jj == j
        {
            return (m, a, b);
        }
        lin_truncate(self.fwd_mark);
        let a = tri_seed(&self.st, &self.lin, o, j);
        let b = tri_seed(&self.st, &self.f32m, o, j);
        let m = lin_mark();
        self.cur = Some((o, j, m, a, b));
        (m, a, b)
    }

    /// seed `(o, j)` の二階の要素（`wrt = None` は `g`、`Some(w)` は入力 `w` のレーン `i`。`ji = (j, i)`）が
    /// [`entry_check_lin`] を通るか。
    fn ok(&mut self, o: usize, wrt: Option<usize>, ji: (usize, usize), what: &str) -> bool {
        let (j, i) = ji;
        let (mark, a, b) = self.seed(o, j);
        let ok = match wrt {
            None => entry_check_lin(a.g, b.g, what, invalid).is_ok(),
            Some(w) => {
                let l = tri_x(&self.st, &self.lin, &a, w, (i, j));
                let s = tri_x(&self.st, &self.f32m, &b, w, (i, j));
                entry_check_lin(l, s, what, invalid).is_ok()
            }
        };
        lin_truncate(mark);
        ok
    }
}

impl Drop for TriLinState {
    fn drop(&mut self) {
        lin_reset();
    }
}

/// [`TriLinState`] を作るのに要る 1 行の入力（[`triplet_cancel_check`] が最初に拒否した要素で遅延して作る）。
struct TriRowInput<'a> {
    rows: (&'a [f32], &'a [f32], &'a [f32]),
    /// `(eps, p, swap)`。
    opts: (f32, f32, bool),
    infos: [Option<&'a PairInfo>; 3],
    base: usize,
    an_kind: u8,
}

/// Triplet の 1 行の総括判定（設計 §55.2。相殺・f32 の underflow）。
///
/// 1 ペアの p ノルム勾配 `core_j = sign(u_j)·(|u_j|/N)^(p−1)` の二階は、v のレーン i へ
/// `fac·(δ_ij·dg_j − (p−1)·sg_i·sg_j·e_i·e_j)` で、対角は 1 成分が支配的な行（`q_j → 1`）で相殺する
/// （p=2 では `(1 − û_j²)/N`）。さらに `da` の a への二階は `H_ap − c_an·H_an`、`dp` の p への二階は
/// `H_ap − c_pn·H_pn` と 2 ペアの差になり、非対角でも相殺しうる。9 ブロック（出力 da／dp／dn ×
/// 入力 a／p／n）の各要素と、g への二階（`Σ w·core`）を、f64 の参照と f32 の再現値（[`triplet_pair_row`]）
/// で組んで [`struct_check`] に掛ける。スケーリング後に微小成分が 0 になる行（例: 差分
/// `[1e10, 1e10, 1e-40]` の第 3 成分は `v·(1/mx)` で 0 になり、p > 2 でその成分の曲率 `dg` が 0 になる）は
/// `V_str` との差で拒否される。`coeff` は幅 `d` と `p`、差分の再構成の条件数 `κ`
/// （`(|x − y| + |eps|)/|v|`）から呼び出し元が作る。f32 の再現値がどれも `coeff` 以内で一致し、かつ
/// O(d) の速い経路で絶対判定を満たすブロックは O(d²) の走査を省く。
///
/// **`Lin` の上界による救済（#4096・設計 §82.13）**: 上の見積り（`coeff·m`）が拒否した要素だけ、丸めの一次伝播
/// [`Lin`] の上界（[`TriLinState`]）で評価し直し、閾値内なら受理する（[`lin_elem`]）。どちらの見積りも独立に健全
/// なので片方が通れば REQ-2 の判定は成り立ち、従来の受理を拒否へ変えることは構造的に起きない（判定は緩めず、
/// 過剰な拒否を減らす向きだけ）。`lin = Some` はテンソル単位の条件 [`tri_lin_ctx`]（走査量と事象数の上限。9 ブロックの
/// 要素の救済は `ctx.x` のときだけ）を満たすときだけで、満たさないテンソルは従来の見積りのまま。全要素 0 のペア・
/// 差分の符号が食い違う行・上界を持たない要素（巨大な p・非正規数域等）は救済せず従来の判定が残る。
/// `fast_ok` の速い経路・`flag` の式・`coeff` の式は変えていない。
fn triplet_cancel_check(
    pr: &[Option<(PairRow, PairRow, bool)>; 3],
    coefs: (f64, f64),
    d: usize,
    scales: (f64, f64, f64),
    coeff: f64,
    lin: Option<(TriLinCtx, TriRowInput<'_>)>,
) -> Result<(), AutodiffError> {
    let (c_an, c_pn) = coefs;
    let (sb, gs, rho) = scales;
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
    let live = |terms: &[(usize, f64); 2]| -> Vec<(&PairRow, &PairRow, f64)> {
        terms
            .iter()
            .filter(|&&(_, w)| w != 0.0)
            .filter_map(|&(k, w)| pr[k].as_ref().map(|(r, st, _)| (r, st, w)))
            .collect()
    };
    let flag = deviates(1.0, rho, 1.0, coeff) || pr.iter().flatten().any(|(_, _, f)| *f);
    #[cfg(test)]
    probe::flag("Triplet", flag);
    // `Lin` の上界の状態（従来の見積りが最初に拒否した要素で 1 回だけ作る。`Drop` で領域を解放する）。外側の `Option` は
    // 未作成、内側の `None` は救済しない行（[`tri_setup_row`] が `None`）。
    let mut lin_state: Option<Option<TriLinState>> = None;
    // `wrt = None` は `g` への二階、`Some(w)` は入力 `w`。`o` は出力、`(j, i)` は seed とレーン。
    let mut rescue = |o: usize, wrt: Option<usize>, ji: (usize, usize), what: &str| -> bool {
        let Some((ctx, inp)) = lin.as_ref() else {
            return false;
        };
        if wrt.is_some() && !ctx.x {
            return false;
        }
        let ctx = *ctx;
        let slot = lin_state.get_or_insert_with(|| {
            tri_setup_row(
                inp.rows,
                inp.opts,
                (ctx.g, ctx.mean_n),
                inp.infos,
                inp.base,
                inp.an_kind,
            )
            .filter(TriSetup::rescuable)
            .map(TriLinState::new)
        });
        slot.as_mut().is_some_and(|s| s.ok(o, wrt, ji, what))
    };
    for (o, terms) in outputs.iter().enumerate() {
        let ts = live(terms);
        for j in 0..d {
            let v: f64 = ts.iter().map(|(r, _, w)| w * r.sg[j] * r.e[j]).sum();
            let v_str: f64 = ts.iter().map(|(_, st, w)| w * st.sg[j] * st.e[j]).sum();
            let m: f64 = ts.iter().map(|(r, _, w)| w.abs() * r.e[j]).sum();
            const G_WHAT: &str = "Triplet: g への二階（ペア勾配の差し引き・f32 の underflow）";
            lin_elem([v, v_str, m, gs, coeff], G_WHAT, || {
                rescue(o, None, (j, 0), G_WHAT)
            })?;
        }
    }
    for (b, terms) in blocks.iter().enumerate() {
        let ts = live(terms);
        if ts.is_empty() {
            continue;
        }
        let fast: f64 = ts
            .iter()
            .map(|(r, _, w)| {
                let dmax = r.dg.iter().fold(0f64, |m, x| m.max(x.abs()));
                let emax = r.e.iter().fold(0f64, |m, &x| m.max(x));
                w.abs() * r.fac * (dmax + r.pm1.abs() * emax * emax)
            })
            .sum();
        if !flag && fast_ok(coeff * fast * sb) {
            continue;
        }
        for j in 0..d {
            for i in 0..d {
                let delta = if i == j { 1.0 } else { 0.0 };
                let (mut v, mut v_str, mut m) = (0f64, 0f64, 0f64);
                for (r, st, w) in &ts {
                    // seed は出力レーン j（`core_j`）、cotangent は入力レーン i へ流れる。
                    let off = r.pm1 * r.sa[i] * r.sg[j] * r.e[i] * r.e[j];
                    v += w * r.fac * (delta * r.dsg[j] * r.dg[j] - off);
                    m += w.abs() * r.fac * (delta * r.dg[j].abs() + off.abs());
                    let off_s = st.pm1 * st.sa[i] * st.sg[j] * st.e[i] * st.e[j];
                    v_str += w * st.fac * (delta * st.dsg[j] * st.dg[j] - off_s);
                }
                const X_WHAT: &str =
                    "Triplet: 二階（支配的な成分・2 ペアの差の相殺・f32 の underflow）";
                lin_elem([v, rho * v_str, m, sb, coeff], X_WHAT, || {
                    rescue(b / 3, Some(b % 3), (j, i), X_WHAT)
                })?;
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
        let rho = sb_ratio(upstream_scale32(&g_value, reduction, n), sb);
        // `Lin` の上界で相殺判定を救うか（テンソル単位。#4096・設計 §82.13）。`g` は子テープ上の f32 の値、判定する行は活性な行。
        let n_rows = act_rows.iter().filter(|&&a| a).count();
        let g_f32 = f64::from(
            crate::eval::dense_vec(&g_value)
                .first()
                .copied()
                .unwrap_or(0.0),
        );
        let lin_ctx = tri_lin_ctx(
            g_f32,
            matches!(reduction, Reduction::Mean).then_some(n),
            n_rows,
            d,
            swap,
        );
        for i in 0..n {
            if !act_rows[i] {
                continue;
            }
            let c_an = c_an_rows[i];
            let c_pn = if swap { 1.0 - c_an } else { 0.0 };
            let coeff = rounding_coeff(d)
                + 2.0 * ((p64 - 1.0).abs() + 1.0) * (kap_rows[i] + 1.0) * (-23f64).exp2();
            let row = |rows: &[Vec<f64>], rec: &[Vec<f32>], info: &PairInfo| {
                triplet_pair_row(&rows[i], &rec[i], info, i * d, p64, coeff)
            };
            let pr = [
                row(&rows_ap, &rec_ap, &info_ap),
                row(&rows_an, &rec_an, &info_an),
                info_pn
                    .as_ref()
                    .and_then(|info| row(&rows_pn, &rec_pn, info)),
            ];
            let nan = [&rows_ap[i], &rows_an[i]]
                .into_iter()
                .chain(swap.then_some(&rows_pn[i]))
                .any(|r| r.iter().any(|v| v.is_nan()));
            if nan {
                continue;
            }
            let base = i * d;
            let lin_row = lin_ctx.map(|ctx| {
                (
                    ctx,
                    TriRowInput {
                        rows: (
                            &a_s[base..base + d],
                            &p_s[base..base + d],
                            &n_s[base..base + d],
                        ),
                        opts: (eps32, p, swap),
                        infos: [Some(&info_ap), Some(&info_an), info_pn.as_ref()],
                        base,
                        an_kind: an_kind_of(c_an),
                    },
                )
            });
            triplet_cancel_check(&pr, (c_an, c_pn), d, (sb, gs, rho), coeff, lin_row)?;
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

#[cfg(test)]
#[path = "loss_class_tests.rs"]
mod tests;
