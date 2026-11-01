//! `Op::LayerNorm`・`Op::RmsNorm`・`Op::BatchNorm` の `create_graph`（子テープ方式の二階微分）対応
//! （イシュー #3078・#3079・親 #3075。方針の正は
//! `docs/autodiff-higher-order-grad-decision.md` §21.1・§21.3、実装記録は §27（LayerNorm・
//! RmsNorm）・§28（BatchNorm））。
//!
//! [`crate::create_graph`] の `validate_ancestors`／`replay_op`／`build_cgrads` から呼ばれる
//! 正規化専用の補助モジュール。`create_graph.rs` 本体を肥大させず、並行する
//! #3077（Softmax）との差分衝突を避けるために分離している。
//!
//! # 方式（§21.1 の V3＋S1）
//!
//! - **値（V3）**: 子テープ上の 1 階勾配ノードの値は 1 階と同じホスト関数
//!   （`grad.rs::rmsnorm_vjp_rows`／`layer_norm_vjp_rows`。f64 縮約契約を持つ）で計算する。
//!   CPU／`naive_ops` では 1 階 VJP も同じ関数へフォールバックするため、`g` が親の
//!   upstream と bit 一致する範囲では `cg.grad` も 1 階と bit 一致する。CUDA は 1 階が
//!   専用カーネルを通るので REQ-2 統一複合判定の範囲の一致のみを主張する（実機 parity は
//!   本イシューの対象外）。
//! - **構造（S1）**: 二階微分は、行ごとの 2 のべき乗スケール `s`（ホスト定数）で `x/s` を作って
//!   から二乗和・`rstd` を合成する f32 の `Var` 演算列 `S_k` を流れる。`x=1e20` のような
//!   有限入力でも f32 二乗が overflow しない。`S_k` の出力ごとに
//!   `Op::Contiguous`（VJP は恒等）のノードを `Tape::push_eager` で 1 個積み、値だけを
//!   ホスト関数の結果に差し替える（`create_graph.rs::reduce_bias_grad_var` と同型）。
//! - **閉包性（§21.1 (4)）**: 積む Op は `Reshape`・`Contiguous`・`Narrow`・`Mean`・`Sum`・
//!   `Mul`・`Add`・`ScalarBinary{Sub,Div}`・`ScalarUnary{Sqrt}` のみで、すべて
//!   `supports_create_graph == true`（孫テープ再生・`jvp_create_graph` が通る）。
//!   `Rsqrt`／`Reciprocal`／`Op::Var` は再生不可のため使わない。
//!
//! # `LayerNorm` の安定な中心化
//!
//! `Mean` の f32 出力を `x` から直接引くと `x=[16777216,16777218]` で偏差が `[0,2]` に
//! 狂う。そこで行の先頭要素 `x_ref` を先に引いた `d = x − x_ref` から `m = mean(d)`・
//! `dev = d − m` を作り、補正パス `dev2 = dev − mean(dev)` を 1 回加える。`s` は中心化前の
//! `x` ではなく `dev2` の実値から決める（定数行で `rstd_scaled=inf` になるのを避ける）。
//!
//! # 拒否条件（§21.3 の「S2 へ切り替える」より狭い fail-closed。所有者へ戻す論点）
//!
//! S2（ホストの統計定数＋二階項の手書き合成）は本イシューでは実装しない。代わりに
//! 子テープへ書く前に [`validate_norm_s1`] が親テープの `x` から次の入力を検出し
//! `Err(AutodiffError::Backward)` で拒否する（`child` は空のまま保たれる）。
//!
//! - `LayerNorm`: (a) `x` に非有限要素 (b) f32 の `x_i − x_0` が非有限
//!   (c) `hidden · max|x_i − x_0| ≥ 2^125`（`Mean` の f32 部分和が overflow しうる）
//!   (d) `eps == 0` で `max|x_i − x_0|` が `2^-100` 未満（定数行・極小偏差行。`rstd=inf`）
//! - `RmsNorm`: (a) 非有限要素 (d') `eps == 0` の全ゼロ行
//!
//! # 総括的な適用判定（構造段の上界。決定記録 §27.3）
//!
//! 上の入口検査は親の `x` だけで決まる。upstream `g`・`w`・`rstd` の組み合わせで構造の
//! 中間値が f32 を超える経路（`g=3e38` の縮約、`hidden·g` 規模の cotangent 和、低分散行の
//! `rstd` 増幅、`w=0` で `g·w` 検査を素通りする dw 経路の `rstd_s` cotangent 等。いずれも
//! PR #3346 のレビュー指摘）は個別条件では網羅できないため、子テープの構造を組む前に
//! [`check_structure_stages`] が **構造の各段を f64 で再計算し、その絶対値の上界**を一括で
//! 判定する。対象は dx／dw／db 構造の forward の各段（行方向和・行をまたぐ和を含む）と、
//! 二階の逆伝播（`cg.grad` の出力へ入る cotangent の各要素が絶対値 1 以下という前提）で
//! 各段に現れる cotangent・部分和・微分係数との積の上界で、境界ノード（`x`・`w`・`g` の
//! 写し）へ出る cotangent までを含む。どれか 1 段でも `2^125` 以上（または非有限）なら
//! `Err(Backward)` で拒否する（`child` は `backward_create_graph` のロールバックで空へ戻る）。
//! 上界は符号を見ない最悪値なので、拒否は「実際に overflow する」ことを意味しない
//! （保守的な拒否。構造の forward 出力も判定するので、1 階の値が有限でも `2^125` 以上なら
//! 拒否する。孫テープ再生では `Op::Contiguous` が構造の f32 値を再計算するため）。
//! 上に挙げた個別の経路は既知の失敗例であり、網羅を主張するのはこの判定の方である。
//! 保証は判定を通った入力に限り、判定を通った入力では 1 階・`cg.grad`・二階（seed の各要素の
//! 絶対値 ≤ 1）・孫テープ再生が有限で、`cg.grad` は 1 階と REQ-2 判定で一致する。孫テープ側で
//! 子テープ上の正規化の写しが再び微分される場合は、そこでも同じ判定が走り、通らなければ同じく
//! 型付きエラーになる。`g` を作る上流の Op の逆伝播、絶対値 1 を超える seed、三階以上の微分は
//! 対象外。
//!
//! # `BatchNorm`（イシュー #3079・決定記録 §28）
//!
//! チャネル軸を前に出した行 `[c, m = n·spatial]`（`Reshape`・`Permute`・`Contiguous`）を行として、
//! 上と同じ方式・同じ判定を使う（別方式を持たない）。
//!
//! - **train**: `LayerNorm` と同じ構造（安定な中心化・`dev2` から決める行スケール `s`・`rstd_s`・
//!   `gs` 付きの dx）。違いは `w`・`b` が行ごと（`[c]`）で、dw／db が行方向の `Sum(1)` である
//!   ことだけで、判定は [`ParamAxis::Row`] として同じ [`check_structure_stages`] を通す。入口検査は
//!   チャネル行ごとに `LayerNorm` と同じ (a)〜(d)（`hidden` の代わりに `m`）。
//! - **eval**: 統計量が定数（`fixed_stats`）なので、構造は `xhat = (x − mean)·rstd`（定数葉）と
//!   `dx = (g·w)·rstd` の要素積・dw／db の行方向和だけになる。判定は [`FixedStats`] 付きの
//!   [`check_structure_stages`] が `x − mean`・`xhat`・`g·w`・`dx`・`g·xhat`・縮約と、二階の逆伝播
//!   （`g`・`w`・`x` への cotangent）の上界を同じ閾値 `2^125` で判定する。
//! - **値**: 3 出力とも 1 階と同じ `grad.rs::batch_norm_vjp_channels`（V3）。`Op::Contiguous` では
//!   なく `Op::BatchNormGradHost` に載せ、孫テープ再生でもホスト関数で再計算する（構造側の f32
//!   縮約とホストの f64 縮約で丸めが異なる入力でも、孫テープの値が 1 階と bit 同一のまま。
//!   PR #3348 のレビュー指摘）。
//! - 以前の版の前置スケール（`x/p`・`s_g` の打ち切り・`±100` 段の 2 のべき乗乗算）は、逆伝播の
//!   倍率・中心化・平均の縮約で overflow が経路ごとに再発したため撤去し、上の判定に置き換えた
//!   （PR #3348 の Codex P1 × 8・Cursor Bugbot 1 件）。その結果 `[3e38,-3e38]` のような入力は
//!   受理から型付きエラーへ変わった（保証は判定を通った入力に限る）。

use fandhe_ai_tensor_core::{BackendOps, Tensor, batch_norm_layout, row_norm_layout};

use crate::error::AutodiffError;
use crate::eval::{build_tensor, dense_vec};
use crate::grad::{batch_norm_vjp_channels, layer_norm_vjp_rows, rmsnorm_vjp_rows};
use crate::tape::{BatchNormGradKind, NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

/// `Mean`（f32 部分和）が overflow しうるとみなす閾値 `2^125`（f32 最大値 `2^128` に対する余裕）。
const MEAN_OVERFLOW_GUARD: f64 = 4.253_529_586_511_73e37;
/// 総括的な適用判定（[`check_structure_stages`]）で、構造の 1 段の上界がこれ以上なら拒否する
/// 閾値 `2^125`（`MEAN_OVERFLOW_GUARD` と同値）。上界は f64 で符号を見ない最悪値として求めるので
/// 部分和の順序には依存しない。f32 最大値 `2^128` との 8 倍の差は、f32 丸めの累積と
/// 「seed の絶対値 1 以下」の前提の外側へのわずかな余裕として置く。
pub(crate) const STAGE_BOUND_GUARD: f64 = MEAN_OVERFLOW_GUARD;
/// `eps == 0` の `LayerNorm` で「偏差が極小で `rstd` を安全に作れない」とみなす閾値 `2^-100`。
const TINY_DEVIATION_GUARD: f64 = 7.888_609_052_210_118e-31;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（LayerNorm／RmsNorm／BatchNorm の S1 構造が安全に扱えない入力。\
         docs/autodiff-higher-order-grad-decision.md §27・§28 参照）"
    ))
}

fn shape_err(e: fandhe_ai_tensor_core::ShapeError) -> AutodiffError {
    AutodiffError::Shape(e)
}

/// `validate_ancestors` から `requires_grad == true` かつ対象 Op のノードごとに呼ばれる入口検査。
/// `RmsNorm`／`LayerNorm`／`BatchNorm` 以外の Op では何もしない（`BatchNorm` は
/// [`validate_batch_norm`] へ委ねる）。親テープの `x` 値だけで判定し、
/// バックエンドに依存しない保守的な拒否条件を適用する（モジュール doc「拒否条件」）。
pub(crate) fn validate_norm_s1(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    op: &Op,
) -> Result<(), AutodiffError> {
    let (input, eps, is_layer_norm) = match op {
        Op::RmsNorm { input, eps, .. } => (*input, *eps, false),
        Op::LayerNorm { input, eps, .. } => (*input, *eps, true),
        Op::BatchNorm {
            input,
            eps,
            fixed_stats,
            ..
        } => {
            let x = materialize_fallible(parent_nodes, parent_ops, *input)?;
            return validate_batch_norm(x, *eps, fixed_stats.as_ref());
        }
        _ => return Ok(()),
    };
    let x = materialize_fallible(parent_nodes, parent_ops, input)?;
    let (rows, hidden) = row_norm_layout(x.shape()).map_err(shape_err)?;
    if rows == 0 || hidden == 0 {
        return Ok(());
    }
    let data = dense_vec(x);
    if data.iter().any(|v| !v.is_finite()) {
        return Err(reject("正規化の入力に非有限要素がある"));
    }
    for r in 0..rows {
        let row = &data[r * hidden..(r + 1) * hidden];
        if is_layer_norm {
            let x0 = row[0];
            let mut max_dev = 0.0f64;
            for &v in row {
                if !(v - x0).is_finite() {
                    return Err(reject("LayerNorm の x_i − x_0 が f32 で非有限になる"));
                }
                max_dev = max_dev.max((v as f64 - x0 as f64).abs());
            }
            if hidden as f64 * max_dev >= MEAN_OVERFLOW_GUARD {
                return Err(reject(
                    "LayerNorm の偏差が大きく Mean の f32 部分和が overflow しうる",
                ));
            }
            if eps == 0.0 && max_dev < TINY_DEVIATION_GUARD {
                return Err(reject("eps=0 の LayerNorm で行が定数（または偏差が極小）"));
            }
        } else if eps == 0.0 && row.iter().all(|&v| v == 0.0) {
            return Err(reject("eps=0 の RmsNorm で行が全ゼロ"));
        }
    }
    Ok(())
}

/// `replay_op` から呼ばれる forward 再生（`Var::rms_norm`／`layer_norm` への薄い委譲）。
pub(crate) fn replay_norm<'c>(
    op: &Op,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match op {
        Op::RmsNorm { input, weight, eps } => {
            let x = get(mirror, *input)?;
            let w = weight.map(|id| get(mirror, id)).transpose()?;
            x.rms_norm(w.as_ref(), *eps)
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
        } => {
            let x = get(mirror, *input)?;
            let w = weight.map(|id| get(mirror, id)).transpose()?;
            let b = bias.map(|id| get(mirror, id)).transpose()?;
            x.layer_norm(w.as_ref(), b.as_ref(), *eps)
        }
        _ => Err(AutodiffError::Backward(
            "create_graph: replay_norm: RmsNorm／LayerNorm 以外の Op（内部契約違反）".into(),
        )),
    }
}

fn get<'c>(mirror: &[Option<Var<'c>>], id: NodeId) -> Result<Var<'c>, AutodiffError> {
    mirror[id.0].ok_or_else(|| {
        AutodiffError::Backward(format!(
            "create_graph: 祖先ノード（NodeId({}))の写しが未構築（内部不変条件違反）",
            id.0
        ))
    })
}

/// 2 のべき乗 `2^e`（`e` は `[-126, 127]` で f32 の正規数として厳密に表現できる）の指数を
/// `t` 以上になる最小の整数として求め、表現可能域へクランプする。`t` が 0・非有限なら 0（`s=1`）。
pub(crate) fn pow2_ceil_exponent(t: f64) -> i32 {
    if t.is_nan() || t.is_infinite() || t <= 0.0 {
        return 0;
    }
    let mut e = t.log2().ceil() as i32;
    while 2f64.powi(e) < t {
        e += 1;
    }
    while 2f64.powi(e - 1) >= t {
        e -= 1;
    }
    e.clamp(-126, 127)
}

/// 行ごとのスケール指数。`eps>0` は `max(maxabs, √eps)`、`eps=0` は `maxabs` を 2 のべき乗へ
/// 切り上げる（§21.1 (5)）。
fn scale_exponent(maxabs: f64, eps: f32) -> i32 {
    let t = if eps > 0.0 {
        maxabs.max((eps as f64).sqrt())
    } else {
        maxabs
    };
    pow2_ceil_exponent(t)
}

fn row_maxabs(data: &[f32], rows: usize, hidden: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| {
            data[r * hidden..(r + 1) * hidden]
                .iter()
                .fold(0.0f64, |m, &v| m.max((v as f64).abs()))
        })
        .collect()
}

/// 行ごとの指数から `s`（`[rows, 1]`）と `eps/s²`（`[rows, 1]`。f64 計算→ f32 丸め）の定数葉を作る。
fn scale_consts<'c>(
    child: &'c Tape,
    exps: &[i32],
    eps: f32,
) -> Result<(Var<'c>, Var<'c>), AutodiffError> {
    let s: Vec<f32> = exps.iter().map(|&e| 2f64.powi(e) as f32).collect();
    let eps_s: Vec<f32> = exps
        .iter()
        .map(|&e| (eps as f64 / 2f64.powi(2 * e)) as f32)
        .collect();
    let shape = [exps.len(), 1];
    let s_t = Tensor::new(s, &shape)?;
    let e_t = Tensor::new(eps_s, &shape)?;
    Ok((child.var_no_grad(&s_t), child.var_no_grad(&e_t)))
}

/// アフィン係数（`w`・`b`）が並ぶ軸。dw／db 構造の縮約方向と、`w` への cotangent の蓄積方向を決める。
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamAxis {
    /// `LayerNorm`／`RmsNorm`: `w`・`b` は `[hidden]`（列ごと）。dw／db は行をまたぐ `Sum(0)`。
    Column,
    /// `BatchNorm`（チャネル行 `[c, n·spatial]`）: `w`・`b` は `[c]`（行ごと）。dw／db は行方向の
    /// `Sum(1)`。
    Row,
}

/// `BatchNorm` eval の固定統計（チャネルごと。構造が定数葉に使う f32 値そのもの）。
struct FixedStats<'a> {
    mean: &'a [f32],
    /// `1/√(var+eps)` を f64 で求めて f32 へ丸めた値（構造の定数葉と同じ）。
    rstd: &'a [f32],
}

/// [`check_structure_stages`] の入力。`rms_norm_cgrads`／`layer_norm_cgrads`／`batch_norm_cgrads` が
/// 構造を組む直前に組み立てる。`src` は構造が `x/s` に使う行データ（`RmsNorm` は `x`、`LayerNorm`
/// と `BatchNorm` train は子テープ上の `dev2` の実値、`BatchNorm` eval はチャネル行の `x`）、`exps`
/// はその行スケール `s` の指数で、どちらも構造と同じ値を渡す（eval は `s` を使わないので空でよい）。
struct StageCheck<'a> {
    /// `LayerNorm`・`BatchNorm` train（中心化と `mean_dxhat` 項を持つ）なら `true`。
    centered: bool,
    /// `w`・`b` の軸（[`ParamAxis`]）。
    param_axis: ParamAxis,
    /// `Some` なら `BatchNorm` eval（統計量が定数の構造。`centered`・`exps` は見ない）。
    fixed: Option<FixedStats<'a>>,
    src: &'a [f32],
    exps: &'a [i32],
    eps: f32,
    /// 子テープ上の upstream `g` の実値（`[rows, hidden]` の密。`BatchNorm` はチャネル行の並び）。
    dy: &'a [f32],
    w: Option<&'a [f32]>,
    rows: usize,
    hidden: usize,
    /// dx／dw／db の各構造を実際に組むか（追跡されている寄与先だけ）。
    build_dx: bool,
    build_dw: bool,
    build_db: bool,
    /// 子テープ上の `x`・`g`・`w` の写しが勾配を受け取るか。受け取らないノードへの cotangent は
    /// `backward.rs` が捨てるので判定しない（`Tape::backward` の `requires_grad` 規約）。
    x_tracked: bool,
    g_tracked: bool,
    w_tracked: bool,
}

/// 構造の 1 段の上界 `v` を判定する。非有限または `2^125` 以上なら型付きエラー。
fn stage(v: f64, what: &str) -> Result<(), AutodiffError> {
    if v.is_finite() && v.abs() < STAGE_BOUND_GUARD {
        Ok(())
    } else {
        Err(reject(&format!(
            "構造段の上界判定: {what} の上界 {v:e} が 2^125 以上で f32 の安全域を外れる"
        )))
    }
}

/// 総括的な適用判定（モジュール doc「総括的な適用判定」・決定記録 §27.3・§28）。
/// `rms_norm_cgrads`／`layer_norm_cgrads`／`batch_norm_cgrads` が子テープへ構造を積む前に呼ぶ。
/// dx／dw／db 構造の forward の各段を f64 で再計算し、二階の逆伝播の各段の cotangent の上界を
/// 「`cg.grad` 出力への seed の各要素が絶対値 1 以下」として f64 で積み上げ、どれか 1 段でも f32 の
/// 安全域（`2^125` 未満）を外れれば拒否する。上界は構造の演算順（`binary_partials` の `Div` 係数
/// `-(a/b)/b`・`Sqrt` の `0.5/y`・`Mean` の `1/hidden` を含む）を逐語でなぞり、各段の
/// 「cotangent × 微分係数」と縮約の部分和を三角不等式で押さえる。定数葉（`s`・`gs`・`gs/hidden`・
/// `eps/s²`・eval の `mean`／`rstd`）への寄与は捨てられるので判定しない。`BatchNorm` はチャネル行
/// `[c, n·spatial]` を行として同じ判定を通し、違いは `w`・`b` の軸（[`ParamAxis::Row`]）と eval の
/// 定数統計（[`FixedStats`]）だけである。
/// 戻り値は dx 構造が使う upstream スケール `gs` の行ごとの指数（`build_dx == false` または eval
/// なら空）。
fn check_structure_stages(c: &StageCheck<'_>) -> Result<Vec<i32>, AutodiffError> {
    let h = c.hidden;
    let n = h as f64;
    if c.dy.iter().any(|v| !v.is_finite()) {
        return Err(reject("upstream に非有限要素がある"));
    }
    // `w`・`b` の添字（列ごと／行ごと）と、縮約・蓄積先ベクトルの長さ。
    let pidx = |r: usize, i: usize| match c.param_axis {
        ParamAxis::Column => i,
        ParamAxis::Row => r,
    };
    let plen = match c.param_axis {
        ParamAxis::Column => h,
        ParamAxis::Row => c.rows,
    };
    let wv = |r: usize, i: usize| c.w.map_or(1.0, |w| f64::from(w[pidx(r, i)]));
    let mut gs_exps = Vec::with_capacity(if c.build_dx { c.rows } else { 0 });
    // dw／db 構造の縮約（列ごとは行をまたぐ `Sum(0)`、行ごとは行方向の `Sum(1)`）と、
    // w への cotangent の蓄積（`g·w` の broadcast の逆）。
    let mut col_dw = vec![0.0f64; plen];
    let mut col_db = vec![0.0f64; plen];
    let mut a_w = vec![0.0f64; plen];
    for r in 0..c.rows {
        let src = &c.src[r * h..(r + 1) * h];
        let g: Vec<f64> = c.dy[r * h..(r + 1) * h]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let gw: Vec<f64> = (0..h).map(|i| g[i] * wv(r, i)).collect();
        for v in &gw {
            stage(*v, "dxhat = g·w")?;
        }
        let mut a_xhat = vec![0.0f64; h];
        let mut a_g = vec![0.0f64; h];

        if let Some(fs) = &c.fixed {
            // --- BatchNorm eval: xhat = (x − mean)·rstd（mean・rstd は定数葉）・dx = (g·w)·rstd ---
            let mu = f64::from(fs.mean[r]);
            let rho = f64::from(fs.rstd[r]);
            stage(rho, "固定統計の rstd")?;
            let mut xhat = vec![0.0f64; h];
            for i in 0..h {
                let d = f64::from(src[i]) - mu;
                stage(d, "x − mean（固定統計）")?;
                xhat[i] = d * rho;
                stage(xhat[i], "xhat（固定統計）")?;
            }
            if c.build_dx {
                for i in 0..h {
                    stage(gw[i] * rho, "dx 構造の出力（固定統計）")?;
                    // dx = gw·rstd → gw へ rstd、そこから g へ |w|·rstd、w へ |g|·rstd。
                    a_g[i] += rho * wv(r, i).abs();
                    a_w[pidx(r, i)] += rho * g[i].abs();
                }
            }
            if c.build_dw {
                for i in 0..h {
                    let prod = g[i].abs() * xhat[i].abs();
                    stage(prod, "g·xhat")?;
                    col_dw[pidx(r, i)] += prod;
                    a_xhat[i] += g[i].abs();
                    a_g[i] += xhat[i].abs();
                }
            }
            if c.build_db {
                for i in 0..h {
                    col_db[pidx(r, i)] += g[i].abs();
                    a_g[i] += 1.0;
                }
            }
            if c.g_tracked {
                for v in &a_g {
                    stage(*v, "g への cotangent")?;
                }
            }
            if c.x_tracked && c.build_dw {
                // xhat = (x − mean)·rstd → x へ a_xhat·rstd（mean は定数葉）。
                for v in &a_xhat {
                    stage(*v, "xhat への cotangent")?;
                    stage(v * rho, "x への cotangent（固定統計）")?;
                }
            }
            continue;
        }

        let e = c.exps[r];
        let s = 2f64.powi(e);
        // 構造と同じく f64 で計算して f32 へ丸めた `eps/s²`。
        let eps_s = f64::from((f64::from(c.eps) / 2f64.powi(2 * e)) as f32);

        // --- forward（統計量）: xs = src/s・ms・rstd_s・xhat・rstd = rstd_s/s ---
        let xs: Vec<f64> = src.iter().map(|&v| f64::from(v) / s).collect();
        let sq_sum: f64 = xs.iter().map(|v| v * v).sum();
        stage(sq_sum, "二乗和 Σ(x/s)²")?;
        let a = sq_sum / n + eps_s;
        if a.is_nan() || a <= 0.0 {
            return Err(reject(
                "スケール後の分散と eps/s² の和が 0 で rstd が非有限",
            ));
        }
        let rs = 1.0 / a.sqrt();
        stage(rs * rs, "rstd_s²（Div の分母側係数）")?;
        let k = rs / s;
        stage(k, "rstd = rstd_s/s")?;
        let xhat: Vec<f64> = xs.iter().map(|v| v * rs).collect();

        // 二階の逆伝播で行内に集まる cotangent の上界（seed の各要素の絶対値 ≤ 1）。
        let mut a_rs = 0.0f64;

        if c.build_dx {
            let m = gw.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let ge = pow2_ceil_exponent(m);
            gs_exps.push(ge);
            let gs = 2f64.powi(ge);
            let gsh = gs / n;
            let dxhat: Vec<f64> = gw.iter().map(|v| v / gs).collect();
            // forward: dot_gs = Σ(dxhat·xhat)·gs/hidden（中心化ありは sum_gs も）・inner・dx。
            let p_abs: f64 = dxhat.iter().zip(&xhat).map(|(d, x)| (d * x).abs()).sum();
            stage(p_abs, "Σ|dxhat·xhat|")?;
            stage(p_abs * gsh, "mean_dot·gs")?;
            let dot = dxhat.iter().zip(&xhat).map(|(d, x)| d * x).sum::<f64>() * gsh;
            let sumg = if c.centered {
                let sa: f64 = dxhat.iter().map(|d| d.abs()).sum();
                stage(sa * gsh, "mean_dxhat·gs")?;
                dxhat.iter().sum::<f64>() * gsh
            } else {
                0.0
            };
            let mut inner_abs = 0.0f64;
            for i in 0..h {
                let q = xhat[i] * dot;
                stage(gw[i].abs() + sumg.abs() + q.abs(), "dx の内側項")?;
                let inner = gw[i] - sumg - q;
                stage(k * inner, "dx 構造の出力")?;
                inner_abs += inner.abs();
            }
            // backward: dx = (rstd_s/s)·inner。inner へ k、k へ Σ|inner|、rstd_s へ Σ|inner|/s。
            stage(inner_abs, "rstd_s/s への cotangent")?;
            a_rs += inner_abs / s;
            let mut a_dxhat = vec![0.0f64; h];
            if c.centered {
                // sum_gs = Σdxhat·gsh → cotangent n·k が gsh 倍されて各 dxhat へ。
                stage(n * k, "sum_gs への cotangent")?;
                let a_sum0 = n * k * gsh;
                stage(a_sum0, "Σdxhat への cotangent")?;
                a_dxhat.fill(a_sum0);
            }
            // q = xhat·dot_gs → xhat へ k·|dot|、dot_gs へ k·Σ|xhat|、それが gsh 倍で Σ(dxhat·xhat) へ。
            let xhat_abs: f64 = xhat.iter().map(|v| v.abs()).sum();
            stage(k * xhat_abs, "mean_dot·gs への cotangent")?;
            let a_dot0 = k * xhat_abs * gsh;
            stage(a_dot0, "Σ(dxhat·xhat) への cotangent")?;
            for i in 0..h {
                a_xhat[i] += k * dot.abs() + a_dot0 * dxhat[i].abs();
                a_dxhat[i] += a_dot0 * xhat[i].abs();
                stage(a_dxhat[i], "dxhat への cotangent")?;
                // dxhat = gw/gs と inner の gw 項。
                let a_gw = k + a_dxhat[i] / gs;
                stage(a_gw, "g·w への cotangent")?;
                a_g[i] += a_gw * wv(r, i).abs();
                a_w[pidx(r, i)] += a_gw * g[i].abs();
            }
        }
        if c.build_dw {
            // dw = Σ g·xhat（列ごとは行をまたぐ和、行ごとは行方向の和）。seed が各要素へ broadcast。
            for i in 0..h {
                let prod = g[i].abs() * xhat[i].abs();
                stage(prod, "g·xhat")?;
                col_dw[pidx(r, i)] += prod;
                a_xhat[i] += g[i].abs();
                a_g[i] += xhat[i].abs();
            }
        }
        if c.build_db {
            for i in 0..h {
                col_db[pidx(r, i)] += g[i].abs();
                a_g[i] += 1.0;
            }
        }
        if c.g_tracked {
            for v in &a_g {
                stage(*v, "g への cotangent")?;
            }
        }
        if c.x_tracked && (c.build_dx || c.build_dw) {
            // xhat = xs·rstd_s → xs へ a_xhat·rstd_s、rstd_s へ Σ a_xhat·|xs|。
            for v in &a_xhat {
                stage(*v, "xhat への cotangent")?;
            }
            a_rs += a_xhat
                .iter()
                .zip(&xs)
                .map(|(a, x)| a * x.abs())
                .sum::<f64>();
            stage(a_rs, "rstd_s への cotangent")?;
            // rstd_s = 1/r（Div の分母側係数 -(1/r)/r = -rstd_s²）→ r = √a（係数 0.5/r）→ ms → 二乗。
            let a_r = a_rs * rs * rs;
            stage(a_r, "√(ms+eps/s²) への cotangent")?;
            let a_a = a_r * 0.5 * rs;
            stage(a_a, "ms への cotangent")?;
            let a_sq = a_a / n;
            let mut a_src = vec![0.0f64; h];
            for i in 0..h {
                let a_xs = a_xhat[i] * rs + 2.0 * a_sq * xs[i].abs();
                stage(a_xs, "x/s への cotangent")?;
                a_src[i] = a_xs / s;
                stage(a_src[i], "正規化入力への cotangent")?;
            }
            if c.centered {
                // dev2 = dev − Mean(dev)、dev = d − Mean(d)、d = x − x_ref（x_ref は行の先頭要素）。
                let s2: f64 = a_src.iter().sum();
                stage(s2, "Mean(dev) への cotangent")?;
                let a_dev: Vec<f64> = a_src.iter().map(|v| v + s2 / n).collect();
                let s1: f64 = a_dev.iter().sum();
                stage(s1, "Mean(d) への cotangent")?;
                let a_d: Vec<f64> = a_dev.iter().map(|v| v + s1 / n).collect();
                let s0: f64 = a_d.iter().sum();
                stage(s0, "x_ref への cotangent")?;
                for (i, v) in a_d.iter().enumerate() {
                    stage(if i == 0 { v + s0 } else { *v }, "x への cotangent")?;
                }
            }
        }
    }
    for v in &col_dw {
        stage(*v, "dw 構造の縮約")?;
    }
    for v in &col_db {
        stage(*v, "db 構造の縮約")?;
    }
    if c.w_tracked {
        for v in &a_w {
            stage(*v, "w への cotangent")?;
        }
    }
    Ok(gs_exps)
}

/// upstream 側のスケール `gs`（行ごとの 2 のべき乗。`[rows, 1]`）の定数葉を作る。指数は
/// [`check_structure_stages`] が `dxhat = g·w` の行最大絶対値（f64）から決めたもの。`dxhat/gs` を
/// 縮約へ渡すことで `Mean(dxhat⊙xhat)` の f32 部分和が `g=3e38` のような有限入力で overflow する
/// のを避ける（レビュー指摘 PR #3346。復元は縮約の直後に `gs/hidden` を掛けて行う）。
fn gs_const<'c>(child: &'c Tape, exps: &[i32]) -> Result<Var<'c>, AutodiffError> {
    let gs: Vec<f32> = exps.iter().map(|&e| 2f64.powi(e) as f32).collect();
    Ok(child.var_no_grad(&Tensor::new(gs, &[exps.len(), 1])?))
}

/// `gs/hidden`（行ごと。`[rows, 1]`）の定数葉。縮約（`Sum`）の直後にこれを掛けて `gs` を復元する。
/// `Mean` の逆伝播が cotangent を行方向に足し合わせる前に `gs` を掛けないため、`gs`・hidden 倍の
/// 中間 overflow を避ける（レビュー指摘 PR #3346 の二巡目。`gs/hidden` は f64 で計算し f32 へ丸める）。
fn gs_over_hidden<'c>(
    child: &'c Tape,
    gs: &Var<'c>,
    hidden: usize,
) -> Result<Var<'c>, AutodiffError> {
    let v = dense_vec(&child_value(child, gs)?);
    let c: Vec<f32> = v
        .iter()
        .map(|&g| (f64::from(g) / hidden as f64) as f32)
        .collect();
    Ok(child.var_no_grad(&Tensor::new(c, &[v.len(), 1])?))
}

/// 構造 `structure` の VJP 値を `value`（ホスト関数の結果）へ差し替えた `Op::Contiguous`
/// ノードを子テープへ積む。`Contiguous` の VJP は恒等なので二階の微分は `structure` を流れる。
fn tag_with_host_value<'c>(
    child: &'c Tape,
    structure: &Var<'c>,
    value: Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    if structure.shape() != value.shape() {
        return Err(AutodiffError::Backward(
            "create_graph: LayerNorm／RmsNorm の構造と値の shape が不一致（内部契約違反）".into(),
        ));
    }
    let id = child.push_eager(
        Op::Contiguous {
            input: structure.node_id(),
        },
        value,
    );
    Ok(Var::from_raw(child, id))
}

/// 子テープ上の `v` の実値（密）を読む。
fn child_value(child: &Tape, v: &Var<'_>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = child.nodes.borrow();
    Ok(materialize_fallible(&nodes, child.ops(), v.node_id())?.clone())
}

fn rows_hidden_2d<'c>(v: &Var<'c>, rows: usize, hidden: usize) -> Result<Var<'c>, AutodiffError> {
    v.contiguous()?.reshape(&[rows, hidden])
}

fn keep_dim<'c>(v: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    v.unsqueeze(1)
}

/// 安定な中心化（`LayerNorm`・`BatchNorm` train 共通。モジュール doc「`LayerNorm` の安定な中心化」）。
/// 行の先頭要素 `x_ref` を先に引いた `d = x − x_ref` から `dev = d − Mean(d)` を作り、補正パス
/// `dev2 = dev − Mean(dev)` を 1 回加える。`x2` は `[rows, hidden]`。`d` の f32 有限性と `Mean` の
/// 部分和が overflow しないことは入口の [`validate_norm_s1`] が保証する。
pub(crate) fn stable_centering<'c>(x2: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    let x_ref = x2.narrow(1, 0, 1)?;
    let d = x2.sub(&x_ref)?;
    let m = keep_dim(&d.mean(Some(1))?)?;
    let dev = d.sub(&m)?;
    dev.sub(&keep_dim(&dev.mean(Some(1))?)?)
}

/// [`scaled_stats`] が返す統計量の構造（行スケール `s`・`rstd_s`・`xhat`）。
struct ScaledStats<'c> {
    s_c: Var<'c>,
    rstd_s: Var<'c>,
    xhat: Var<'c>,
}

/// 行スケール `s`（指数 `exps`。[`check_structure_stages`] に渡したものと同じ）で `src2/s` を作り、
/// `rstd_s = 1/√(mean((src2/s)²) + eps/s²)`・`xhat = (src2/s)·rstd_s` を合成する（S1。3 正規化共通）。
fn scaled_stats<'c>(
    child: &'c Tape,
    src2: &Var<'c>,
    exps: &[i32],
    eps: f32,
) -> Result<ScaledStats<'c>, AutodiffError> {
    let (s_c, eps_s) = scale_consts(child, exps, eps)?;
    let one = child.var_no_grad(&Tensor::full(&[1], 1.0f32)?);
    let xs = src2.div(&s_c)?;
    let ms = keep_dim(&xs.mul(&xs)?.mean(Some(1))?)?;
    let rstd_s = one.div(&ms.add(&eps_s)?.sqrt()?)?;
    let xhat = xs.mul(&rstd_s)?;
    Ok(ScaledStats { s_c, rstd_s, xhat })
}

/// dx 構造（行 `[rows, hidden]`）: `(rstd_s/s)·(gw − [mean_dxhat·gs] − xhat·mean_dot·gs)`。
/// `gw = g·w` を行ごとの `gs`（指数 `gs_exps`）で割ってから縮約し、縮約の直後に `gs/hidden` を
/// 掛けて復元する（逆伝播で縮約前に大きな cotangent が復元されない。レビュー指摘 PR #3346）。
/// `centered`（`LayerNorm`・`BatchNorm` train）なら `mean_dxhat·gs` 項を持つ。
fn dx_rows<'c>(
    child: &'c Tape,
    gw: &Var<'c>,
    st: &ScaledStats<'c>,
    gs_exps: &[i32],
    hidden: usize,
    centered: bool,
) -> Result<Var<'c>, AutodiffError> {
    let gs_c = gs_const(child, gs_exps)?;
    let dxhat = gw.div(&gs_c)?;
    let gsh = gs_over_hidden(child, &gs_c, hidden)?;
    let inner = if centered {
        let sum_gs = keep_dim(&dxhat.sum(Some(1))?)?.mul(&gsh)?;
        let dot_gs = keep_dim(&dxhat.mul(&st.xhat)?.sum(Some(1))?)?.mul(&gsh)?;
        gw.sub(&sum_gs)?.sub(&st.xhat.mul(&dot_gs)?)?
    } else {
        let dot_gs = keep_dim(&dxhat.mul(&st.xhat)?.sum(Some(1))?)?.mul(&gsh)?;
        gw.sub(&st.xhat.mul(&dot_gs)?)?
    };
    st.rstd_s.div(&st.s_c)?.mul(&inner)
}

/// 空テンソル（`rows == 0 || hidden == 0`）の寄与。1 階と同じくゼロ定数を返す。
fn zero_contributions<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    x_shape: &[usize],
    hidden: usize,
    targets: &Targets,
) -> Result<Vec<(NodeId, Var<'c>)>, AutodiffError> {
    let mut out = Vec::new();
    let tracked = |id: NodeId| parent_nodes[id.0].requires_grad;
    if tracked(targets.input) {
        out.push((targets.input, child.var_no_grad(&Tensor::zeros(x_shape)?)));
    }
    for id in [targets.weight, targets.bias].into_iter().flatten() {
        if tracked(id) {
            out.push((id, child.var_no_grad(&Tensor::zeros(&[hidden])?)));
        }
    }
    Ok(out)
}

/// 寄与先の親ノード id（`weight`／`bias` は存在する場合のみ）。
pub(crate) struct Targets {
    pub(crate) input: NodeId,
    pub(crate) weight: Option<NodeId>,
    pub(crate) bias: Option<NodeId>,
}

/// `Op::RmsNorm` の子テープ上の VJP（V3 値＋S1 構造）。`g` は出力ノードの子テープ上の upstream。
/// 戻り値は `(親ノード id, 子テープ上の寄与)` の列で、呼び出し側（`build_cgrads`）が
/// `accumulate` する。追跡なしの親へは寄与を作らない。
pub(crate) fn rms_norm_cgrads<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    g: Var<'c>,
    targets: &Targets,
    eps: f32,
) -> Result<Vec<(NodeId, Var<'c>)>, AutodiffError> {
    let x_val = materialize_fallible(parent_nodes, parent_ops, targets.input)?.clone();
    let x_shape = x_val.shape().to_vec();
    let (rows, hidden) = row_norm_layout(&x_shape).map_err(shape_err)?;
    if rows == 0 || hidden == 0 {
        return zero_contributions(child, parent_nodes, &x_shape, hidden, targets);
    }
    let need_x = parent_nodes[targets.input.0].requires_grad;
    let need_w = targets
        .weight
        .is_some_and(|id| parent_nodes[id.0].requires_grad);

    // 値（V3）: 1 階と同じホスト関数。
    let x_slice = dense_vec(&x_val);
    let w_val = match targets.weight {
        Some(id) => Some(materialize_fallible(parent_nodes, parent_ops, id)?.clone()),
        None => None,
    };
    let w_slice = w_val.as_ref().map(dense_vec);
    let dy = dense_vec(&child_value(child, &g)?);
    let (dx_host, dw_host) = rmsnorm_vjp_rows(&x_slice, w_slice.as_deref(), eps, rows, hidden, &dy);

    // 総括的な適用判定（構造を積む前。モジュール doc「総括的な適用判定」）。
    let x_m = get(mirror, targets.input)?;
    let w_m = targets.weight.map(|id| get(mirror, id)).transpose()?;
    let exps: Vec<i32> = row_maxabs(&x_slice, rows, hidden)
        .into_iter()
        .map(|m| scale_exponent(m, eps))
        .collect();
    let gs_exps = check_structure_stages(&StageCheck {
        centered: false,
        param_axis: ParamAxis::Column,
        fixed: None,
        src: &x_slice,
        exps: &exps,
        eps,
        dy: &dy,
        w: w_slice.as_deref(),
        rows,
        hidden,
        build_dx: need_x,
        build_dw: need_w && dw_host.is_some(),
        build_db: false,
        x_tracked: x_m.requires_grad(),
        g_tracked: g.requires_grad(),
        w_tracked: w_m.as_ref().is_some_and(Var::requires_grad),
    })?;

    // 構造（S1）。
    let x2 = rows_hidden_2d(&x_m, rows, hidden)?;
    let g2 = rows_hidden_2d(&g, rows, hidden)?;
    let st = scaled_stats(child, &x2, &exps, eps)?;

    let mut out = Vec::new();
    if need_x {
        let gw = match &w_m {
            Some(w) => g2.mul(w)?,
            None => g2,
        };
        let dx_s = dx_rows(child, &gw, &st, &gs_exps, hidden, false)?.reshape(&x_shape)?;
        let v = tag_with_host_value(child, &dx_s, build_tensor(dx_host, &x_shape))?;
        out.push((targets.input, v));
    }
    if need_w && let (Some(wid), Some(dw)) = (targets.weight, dw_host) {
        let dw_s = g2.mul(&st.xhat)?.sum(Some(0))?;
        let v = tag_with_host_value(child, &dw_s, build_tensor(dw, &[hidden]))?;
        out.push((wid, v));
    }
    Ok(out)
}

/// `Op::LayerNorm` の子テープ上の VJP（V3 値＋S1 構造＋安定な中心化）。契約は
/// [`rms_norm_cgrads`] と同じ。
pub(crate) fn layer_norm_cgrads<'c>(
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    mirror: &[Option<Var<'c>>],
    g: Var<'c>,
    targets: &Targets,
    eps: f32,
) -> Result<Vec<(NodeId, Var<'c>)>, AutodiffError> {
    let x_val = materialize_fallible(parent_nodes, parent_ops, targets.input)?.clone();
    let x_shape = x_val.shape().to_vec();
    let (rows, hidden) = row_norm_layout(&x_shape).map_err(shape_err)?;
    if rows == 0 || hidden == 0 {
        return zero_contributions(child, parent_nodes, &x_shape, hidden, targets);
    }
    let need_x = parent_nodes[targets.input.0].requires_grad;
    let need_w = targets
        .weight
        .is_some_and(|id| parent_nodes[id.0].requires_grad);
    let need_b = targets
        .bias
        .is_some_and(|id| parent_nodes[id.0].requires_grad);

    // 値（V3）。
    let x_slice = dense_vec(&x_val);
    let w_val = match targets.weight {
        Some(id) => Some(materialize_fallible(parent_nodes, parent_ops, id)?.clone()),
        None => None,
    };
    let w_slice = w_val.as_ref().map(dense_vec);
    let dy = dense_vec(&child_value(child, &g)?);
    let (dx_host, dw_host, db_host) = layer_norm_vjp_rows(
        &x_slice,
        w_slice.as_deref(),
        targets.bias.is_some(),
        eps,
        rows,
        hidden,
        &dy,
    );

    // 構造: 安定な中心化（x_ref を先に引き、補正パスを 1 回加える）。
    let x_m = get(mirror, targets.input)?;
    let w_m = targets.weight.map(|id| get(mirror, id)).transpose()?;
    let x2 = rows_hidden_2d(&x_m, rows, hidden)?;
    let g2 = rows_hidden_2d(&g, rows, hidden)?;
    let dev2 = stable_centering(&x2)?;

    // `s` は中心化後偏差 `dev2` の子テープ上の実値から決める（中心化前の `x` からは取らない）。
    let dev2_host = dense_vec(&child_value(child, &dev2)?);
    let exps: Vec<i32> = row_maxabs(&dev2_host, rows, hidden)
        .into_iter()
        .map(|mx| scale_exponent(mx, eps))
        .collect();
    // 総括的な適用判定（中心化の 4 ノードの後・統計量と dx／dw／db の構造の前。ここで拒否しても
    // `backward_create_graph` のロールバックが `child` を空へ戻す）。
    let gs_exps = check_structure_stages(&StageCheck {
        centered: true,
        param_axis: ParamAxis::Column,
        fixed: None,
        src: &dev2_host,
        exps: &exps,
        eps,
        dy: &dy,
        w: w_slice.as_deref(),
        rows,
        hidden,
        build_dx: need_x,
        build_dw: need_w && dw_host.is_some(),
        build_db: need_b && db_host.is_some(),
        x_tracked: x_m.requires_grad(),
        g_tracked: g.requires_grad(),
        w_tracked: w_m.as_ref().is_some_and(Var::requires_grad),
    })?;
    let st = scaled_stats(child, &dev2, &exps, eps)?;

    let mut out = Vec::new();
    if need_x {
        let gw = match &w_m {
            Some(w) => g2.mul(w)?,
            None => g2,
        };
        let dx_s = dx_rows(child, &gw, &st, &gs_exps, hidden, true)?.reshape(&x_shape)?;
        let v = tag_with_host_value(child, &dx_s, build_tensor(dx_host, &x_shape))?;
        out.push((targets.input, v));
    }
    if need_w && let (Some(wid), Some(dw)) = (targets.weight, dw_host) {
        let dw_s = g2.mul(&st.xhat)?.sum(Some(0))?;
        let v = tag_with_host_value(child, &dw_s, build_tensor(dw, &[hidden]))?;
        out.push((wid, v));
    }
    if need_b && let (Some(bid), Some(db)) = (targets.bias, db_host) {
        // 構造は `g2.sum(0)`、値は `layer_norm_vjp_rows` の db（`rows == 1` の符号付きゼロの
        // 扱いまで 1 階と bit 一致させるため `reduce_bias_grad_var` ではなく V3 で書く）。
        let db_s = g2.sum(Some(0))?;
        let v = tag_with_host_value(child, &db_s, build_tensor(db, &[hidden]))?;
        out.push((bid, v));
    }
    Ok(out)
}

// ---- BatchNorm（イシュー #3079。決定記録 §28） ----

/// 長さ `c` の定数列を `[c, 1]` の定数葉（勾配なし）にする。
fn const_col<'c>(child: &'c Tape, vals: Vec<f32>) -> Result<Var<'c>, AutodiffError> {
    let len = vals.len();
    Ok(child.var_no_grad(&Tensor::new(vals, &[len, 1])?))
}

/// `[n, c, spatial]` 相当の `v`（任意 rank のチャネル第 1 軸テンソル）をチャネル行
/// `[c, n·spatial]` へ並べ替える（`m = n·spatial` は呼び出し側が `checked_mul` で確かめた値）。
fn to_channel_rows<'c>(
    v: &Var<'c>,
    n: usize,
    c: usize,
    spatial: usize,
    m: usize,
) -> Result<Var<'c>, AutodiffError> {
    v.contiguous()?
        .reshape(&[n, c, spatial])?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[c, m])
}

/// [`to_channel_rows`] の逆変換（`x_shape` へ戻す）。
fn from_channel_rows<'c>(
    v: &Var<'c>,
    n: usize,
    c: usize,
    spatial: usize,
    x_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    v.contiguous()?
        .reshape(&[c, n, spatial])?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(x_shape)
}

/// `n·spatial`（チャネル行の長さ）。空入力は `validate_ancestors` が先に拒否するが、積の overflow は
/// panic させず型付きエラーにする（本番経路 panic 禁止。PR #3348 レビュー指摘）。
fn channel_row_len(n: usize, spatial: usize) -> Result<usize, AutodiffError> {
    n.checked_mul(spatial).ok_or_else(|| {
        AutodiffError::Backward(
            "create_graph: BatchNorm の n·spatial が usize で overflow する（内部契約違反）".into(),
        )
    })
}

/// 親テープの `x`（`[n, c, spatial]` 相当の平坦化）のチャネル `ch` の行（`to_channel_rows` と同じ並び）。
fn channel_row(data: &[f32], c: usize, spatial: usize, m: usize, ch: usize) -> Vec<f32> {
    (0..m)
        .map(|i| data[(i / spatial) * (c * spatial) + ch * spatial + i % spatial])
        .collect()
}

/// [`validate_norm_s1`] の `BatchNorm` 腕。train はチャネル行ごとに `LayerNorm` と同じ (a)〜(d) を、
/// eval は `x`・固定統計の非有限要素を拒否する（eval の `x − mean`・`rstd` の大きさは
/// [`check_structure_stages`] の固定統計分岐が判定する）。
fn validate_batch_norm(
    x: &Tensor<f32>,
    eps: f32,
    fixed_stats: Option<&(Tensor<f32>, Tensor<f32>)>,
) -> Result<(), AutodiffError> {
    // 形状の積を取る前にゼロ軸を判定する（`[usize::MAX, 2, 0]` で積が overflow するため。空入力の
    // 拒否は `validate_ancestors` が担う）。
    if x.shape().contains(&0) {
        return Ok(());
    }
    let (n, c, spatial) = batch_norm_layout(x.shape()).map_err(shape_err)?;
    let m = channel_row_len(n, spatial)?;
    let data = dense_vec(x);
    if data.iter().any(|v| !v.is_finite()) {
        return Err(reject("正規化の入力に非有限要素がある"));
    }
    if let Some((mean, var)) = fixed_stats {
        if dense_vec(mean)
            .iter()
            .chain(dense_vec(var).iter())
            .any(|v| !v.is_finite())
        {
            return Err(reject("BatchNorm eval の固定統計に非有限要素がある"));
        }
        return Ok(());
    }
    for ch in 0..c {
        let row = channel_row(&data, c, spatial, m, ch);
        let x0 = row[0];
        let mut max_dev = 0.0f64;
        for &v in &row {
            if !(v - x0).is_finite() {
                return Err(reject("BatchNorm の x_i − x_0 が f32 で非有限になる"));
            }
            max_dev = max_dev.max((f64::from(v) - f64::from(x0)).abs());
        }
        if m as f64 * max_dev >= MEAN_OVERFLOW_GUARD {
            return Err(reject(
                "BatchNorm の偏差が大きく Mean の f32 部分和が overflow しうる",
            ));
        }
        if eps == 0.0 && max_dev < TINY_DEVIATION_GUARD {
            return Err(reject(
                "eps=0 の BatchNorm でチャネルが定数（または偏差が極小）",
            ));
        }
    }
    Ok(())
}

/// BatchNorm の `dx`／`dw`／`db` ホスト値ノード（`Op::BatchNormGradHost`）を子テープへ登録する。
/// `structure` は二階微分経路、前方値は 1 階と同じ `batch_norm_vjp_channels`（f64 縮約）を `g`・`x`・
/// `weight` の実体値へ適用して求める（V3）。孫テープでの再生（`replay_op`）からも同じ関数で呼ばれ、
/// 値は常にホスト関数で再計算される（`Op::Contiguous` と違い、f64 縮約の補正値が孫テープで
/// 構造側の f32 再計算へ置き換わらない。PR #3348 レビュー指摘・決定記録 §28）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn host_grad_node<'c>(
    child: &'c Tape,
    structure: &Var<'c>,
    g: &Var<'c>,
    x: &Var<'c>,
    w: Option<&Var<'c>>,
    has_bias: bool,
    eps: f32,
    fixed_stats: Option<(Tensor<f32>, Tensor<f32>)>,
    kind: BatchNormGradKind,
) -> Result<Var<'c>, AutodiffError> {
    let x_t = child_value(child, x)?;
    let (n, c, spatial) = batch_norm_layout(x_t.shape())?;
    let g_t = child_value(child, g)?;
    let w_dense = match w {
        Some(w) => Some(dense_vec(&child_value(child, w)?)),
        None => None,
    };
    let fixed_dense = fixed_stats
        .as_ref()
        .map(|(mu, var)| (dense_vec(mu), dense_vec(var)));
    let (dx, dw, db) = batch_norm_vjp_channels(
        &dense_vec(&x_t),
        w_dense.as_deref(),
        has_bias,
        eps,
        n,
        c,
        spatial,
        &dense_vec(&g_t),
        fixed_dense
            .as_ref()
            .map(|(mu, var)| (mu.as_slice(), var.as_slice())),
    );
    let missing = || {
        AutodiffError::Backward(
            "create_graph: BatchNorm ホスト値の出力が欠落（内部契約違反）".into(),
        )
    };
    let value = match kind {
        BatchNormGradKind::Dx => build_tensor(dx, x_t.shape()),
        BatchNormGradKind::Dw => build_tensor(dw.ok_or_else(missing)?, &[c]),
        BatchNormGradKind::Db => build_tensor(db.ok_or_else(missing)?, &[c]),
    };
    if structure.shape() != value.shape() {
        return Err(AutodiffError::Backward(
            "create_graph: BatchNorm の構造グラフとホスト値の shape が不一致（内部契約違反）"
                .into(),
        ));
    }
    let id = child.push_eager(
        Op::BatchNormGradHost {
            structure: structure.node_id(),
            g: g.node_id(),
            x: x.node_id(),
            weight: w.map(|w| w.node_id()),
            has_bias,
            eps,
            fixed_stats,
            kind,
        },
        value,
    );
    Ok(Var::from_raw(child, id))
}

/// [`batch_norm_cgrads`] の入力（親テープ側の実測値と子テープの写し）。
pub(crate) struct BatchNormCgradInputs<'a, 'c> {
    pub x_mirror: Var<'c>,
    pub w_mirror: Option<Var<'c>>,
    pub has_bias: bool,
    pub eps: f32,
    pub fixed_stats: Option<&'a (Tensor<f32>, Tensor<f32>)>,
    /// 親テープ上の `x` の実体値（`[n, c, ...]`）。
    pub x_host: &'a Tensor<f32>,
    /// 親テープの `x`／`weight`／`bias` が勾配を要求するか（要求する寄与先の構造だけを組む）。
    pub need_dx: bool,
    pub need_dw: bool,
    pub need_db: bool,
    pub n: usize,
    pub c: usize,
    pub spatial: usize,
}

/// 子テープ上の `dx`／`dw`／`db`（要求されなかった出力・存在しないパラメータは `None`）。
pub(crate) struct BatchNormCgrads<'c> {
    pub dx: Option<Var<'c>>,
    pub dw: Option<Var<'c>>,
    pub db: Option<Var<'c>>,
}

/// `Op::BatchNorm` の VJP を子テープ上の `Var` として構築する（V3 値＋S1 構造）。`g` は出力 `y` への
/// 上流勾配（子テープ上。`x` と同 shape）。
///
/// 構造はチャネル行 `[c, m = n·spatial]` 上で組む。train は `LayerNorm` と同じ構造（安定な中心化・
/// 行スケール `s`・`rstd_s`・`gs` 付きの dx）で、`w`・`b` が行ごと（`[c]`）であることと dw／db が
/// 行方向の `Sum(1)` であることだけが異なる。eval は統計量が定数なので `xhat = (x − mean)·rstd`
/// （定数葉）と `dx = (g·w)·rstd` の要素積だけになる。どちらも構造を積む前に
/// [`check_structure_stages`]（総括的な適用判定）を通し、判定を外れた入力は型付きエラーで拒否する
/// （`child` は `backward_create_graph` のロールバックで空へ戻る）。値は 3 出力とも
/// [`host_grad_node`]（1 階と同じホスト関数）で与える。
pub(crate) fn batch_norm_cgrads<'c>(
    child: &'c Tape,
    inp: &BatchNormCgradInputs<'_, 'c>,
    g: &Var<'c>,
) -> Result<BatchNormCgrads<'c>, AutodiffError> {
    let (n, c, spatial) = (inp.n, inp.c, inp.spatial);
    let m = channel_row_len(n, spatial)?;
    let x_shape = inp.x_host.shape().to_vec();
    let g_rows = to_channel_rows(g, n, c, spatial, m)?;
    let x_rows = to_channel_rows(&inp.x_mirror, n, c, spatial, m)?;
    // 判定に渡す upstream・w はチャネル行の並びの子テープ実値（構造と同じ値）。
    let dy = dense_vec(&child_value(child, &g_rows)?);
    let w_vals = match &inp.w_mirror {
        Some(w) => Some(dense_vec(&child_value(child, w)?)),
        None => None,
    };
    let build_dw = inp.need_dw && inp.w_mirror.is_some();
    let build_db = inp.need_db && inp.has_bias;
    let w_col = match &inp.w_mirror {
        Some(w) => Some(w.contiguous()?.reshape(&[c, 1])?),
        None => None,
    };
    let gw = || -> Result<Var<'c>, AutodiffError> {
        match &w_col {
            Some(w) => g_rows.mul(w),
            None => Ok(g_rows),
        }
    };
    let host = |s: &Var<'c>, kind: BatchNormGradKind| {
        host_grad_node(
            child,
            s,
            g,
            &inp.x_mirror,
            inp.w_mirror.as_ref(),
            inp.has_bias,
            inp.eps,
            inp.fixed_stats.cloned(),
            kind,
        )
    };

    let (xhat, dx_rows_s) = match inp.fixed_stats {
        None => {
            let dev2 = stable_centering(&x_rows)?;
            let dev2_host = dense_vec(&child_value(child, &dev2)?);
            let exps: Vec<i32> = row_maxabs(&dev2_host, c, m)
                .into_iter()
                .map(|mx| scale_exponent(mx, inp.eps))
                .collect();
            let gs_exps = check_structure_stages(&StageCheck {
                centered: true,
                param_axis: ParamAxis::Row,
                fixed: None,
                src: &dev2_host,
                exps: &exps,
                eps: inp.eps,
                dy: &dy,
                w: w_vals.as_deref(),
                rows: c,
                hidden: m,
                build_dx: inp.need_dx,
                build_dw,
                build_db,
                x_tracked: inp.x_mirror.requires_grad(),
                g_tracked: g.requires_grad(),
                w_tracked: inp.w_mirror.as_ref().is_some_and(Var::requires_grad),
            })?;
            let st = scaled_stats(child, &dev2, &exps, inp.eps)?;
            let dx = if inp.need_dx {
                Some(dx_rows(child, &gw()?, &st, &gs_exps, m, true)?)
            } else {
                None
            };
            (st.xhat, dx)
        }
        Some((mu, var)) => {
            let mean = dense_vec(mu);
            // 1 階（`batch_norm_vjp_channels`）と同じく f64 で `1/√(var+eps)` を求め、定数葉は f32。
            let rstd: Vec<f32> = dense_vec(var)
                .iter()
                .map(|&v| (1.0f64 / (f64::from(v) + f64::from(inp.eps)).sqrt()) as f32)
                .collect();
            let x_rows_host = dense_vec(&child_value(child, &x_rows)?);
            check_structure_stages(&StageCheck {
                centered: false,
                param_axis: ParamAxis::Row,
                fixed: Some(FixedStats {
                    mean: &mean,
                    rstd: &rstd,
                }),
                src: &x_rows_host,
                exps: &[],
                eps: inp.eps,
                dy: &dy,
                w: w_vals.as_deref(),
                rows: c,
                hidden: m,
                build_dx: inp.need_dx,
                build_dw,
                build_db,
                x_tracked: inp.x_mirror.requires_grad(),
                g_tracked: g.requires_grad(),
                w_tracked: inp.w_mirror.as_ref().is_some_and(Var::requires_grad),
            })?;
            let rstd_c = const_col(child, rstd)?;
            let xhat = x_rows.sub(&const_col(child, mean)?)?.mul(&rstd_c)?;
            let dx = if inp.need_dx {
                Some(gw()?.mul(&rstd_c)?)
            } else {
                None
            };
            (xhat, dx)
        }
    };

    let dx = match dx_rows_s {
        Some(rows) => Some(host(
            &from_channel_rows(&rows, n, c, spatial, &x_shape)?,
            BatchNormGradKind::Dx,
        )?),
        None => None,
    };
    let dw = if build_dw {
        let s = g_rows.mul(&xhat)?.sum(Some(1))?;
        Some(host(&s, BatchNormGradKind::Dw)?)
    } else {
        None
    };
    let db = if build_db {
        let s = g_rows.sum(Some(1))?;
        Some(host(&s, BatchNormGradKind::Db)?)
    } else {
        None
    };
    Ok(BatchNormCgrads { dx, dw, db })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_tape() -> Tape {
        Tape::new_with_ops(crate::default_ops::naive_ops())
    }

    #[test]
    fn channel_rows_round_trip_for_rank2_rank3_rank4() {
        for shape in [vec![3usize, 2], vec![2, 3, 4], vec![2, 2, 2, 3]] {
            let (n, c, spatial) = batch_norm_layout(&shape).unwrap();
            let m = channel_row_len(n, spatial).unwrap();
            let numel: usize = shape.iter().product();
            let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
            let tape = new_tape();
            let v = tape.var_no_grad(&Tensor::new(data.clone(), &shape).unwrap());
            let rows = to_channel_rows(&v, n, c, spatial, m).unwrap();
            assert_eq!(rows.shape(), vec![c, m]);
            // ホスト側の行抽出（入口検査が使う `channel_row`）と並びが一致する。
            let rv = rows.to_tensor();
            for ch in 0..c {
                assert_eq!(
                    &rv.as_slice().unwrap()[ch * m..(ch + 1) * m],
                    channel_row(&data, c, spatial, m, ch).as_slice()
                );
            }
            let back = from_channel_rows(&rows, n, c, spatial, &shape).unwrap();
            assert_eq!(back.to_tensor().as_slice().unwrap(), data.as_slice());
        }
    }

    #[test]
    fn channel_row_len_overflow_is_typed_error() {
        assert!(matches!(
            channel_row_len(usize::MAX, 2),
            Err(AutodiffError::Backward(_))
        ));
    }
}
