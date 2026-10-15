//! `Op::LayerNorm`・`Op::RmsNorm` の `create_graph`（子テープ方式の二階微分）対応
//! （イシュー #3078・親 #3075。方針の正は
//! `docs/autodiff-higher-order-grad-decision.md` §21.1・§21.3「#3078」、実装記録は §27）。
//!
//! [`crate::create_graph`] の `validate_ancestors`／`replay_op`／`build_cgrads` から呼ばれる
//! 行方向正規化専用の補助モジュール。`create_graph.rs` 本体を肥大させず、並行する
//! #3077（Softmax）・#3079（BatchNorm）との差分衝突を避けるために分離している。
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

use fandhe_ai_tensor_core::{BackendOps, Tensor, row_norm_layout};

use crate::error::AutodiffError;
use crate::eval::{build_tensor, dense_vec};
use crate::grad::{layer_norm_vjp_rows, rmsnorm_vjp_rows};
use crate::tape::{NodeId, Op, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

/// `Mean`（f32 部分和）が overflow しうるとみなす閾値 `2^125`（f32 最大値 `2^128` に対する余裕）。
const MEAN_OVERFLOW_GUARD: f64 = 4.253_529_586_511_73e37;
/// 総括的な適用判定（[`check_structure_stages`]）で、構造の 1 段の上界がこれ以上なら拒否する
/// 閾値 `2^125`（`MEAN_OVERFLOW_GUARD` と同値）。上界は f64 で符号を見ない最悪値として求めるので
/// 部分和の順序には依存しない。f32 最大値 `2^128` との 8 倍の差は、f32 丸めの累積と
/// 「seed の絶対値 1 以下」の前提の外側へのわずかな余裕として置く。
const STAGE_BOUND_GUARD: f64 = MEAN_OVERFLOW_GUARD;
/// `eps == 0` の `LayerNorm` で「偏差が極小で `rstd` を安全に作れない」とみなす閾値 `2^-100`。
const TINY_DEVIATION_GUARD: f64 = 7.888_609_052_210_118e-31;

fn reject(what: &str) -> AutodiffError {
    AutodiffError::Backward(format!(
        "create_graph: {what}（LayerNorm／RmsNorm の S1 構造が安全に扱えない入力。\
         docs/autodiff-higher-order-grad-decision.md §27 参照）"
    ))
}

fn shape_err(e: fandhe_ai_tensor_core::ShapeError) -> AutodiffError {
    AutodiffError::Shape(e)
}

/// `validate_ancestors` から `requires_grad == true` かつ対象 Op のノードごとに呼ばれる入口検査。
/// `RmsNorm`／`LayerNorm` 以外の Op では何もしない。親テープの `x` 値だけで判定し、
/// バックエンドに依存しない保守的な拒否条件を適用する（モジュール doc「拒否条件」）。
pub(crate) fn validate_norm_s1(
    parent_nodes: &[TapeNode],
    parent_ops: &dyn BackendOps,
    op: &Op,
) -> Result<(), AutodiffError> {
    let (input, eps, is_layer_norm) = match op {
        Op::RmsNorm { input, eps, .. } => (*input, *eps, false),
        Op::LayerNorm { input, eps, .. } => (*input, *eps, true),
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
fn pow2_ceil_exponent(t: f64) -> i32 {
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

/// [`check_structure_stages`] の入力。`rms_norm_cgrads`／`layer_norm_cgrads` が構造を組む直前に
/// 組み立てる。`src` は構造が `x/s` に使う行データ（`RmsNorm` は `x`、`LayerNorm` は子テープ上の
/// `dev2` の実値）、`exps` はその行スケール `s` の指数で、どちらも構造と同じ値を渡す。
struct StageCheck<'a> {
    /// `LayerNorm`（中心化と `mean_dxhat` 項を持つ）なら `true`。
    centered: bool,
    src: &'a [f32],
    exps: &'a [i32],
    eps: f32,
    /// 子テープ上の upstream `g` の実値（`[rows, hidden]` の密）。
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

/// 総括的な適用判定（モジュール doc「総括的な適用判定」・決定記録 §27.3）。`rms_norm_cgrads`／
/// `layer_norm_cgrads` が子テープへ構造を積む前に呼ぶ。dx／dw／db 構造の forward の各段を f64 で
/// 再計算し、二階の逆伝播の各段の cotangent の上界を「`cg.grad` 出力への seed の各要素が絶対値
/// 1 以下」として f64 で積み上げ、どれか 1 段でも f32 の安全域（`2^125` 未満）を外れれば拒否する。
/// 上界は構造の演算順（`binary_partials` の `Div` 係数 `-(a/b)/b`・`Sqrt` の `0.5/y`・`Mean` の
/// `1/hidden` を含む）を逐語でなぞり、各段の「cotangent × 微分係数」と行方向の部分和を三角不等式で
/// 押さえる。定数葉（`s`・`gs`・`gs/hidden`・`eps/s²`）への寄与は捨てられるので判定しない。
/// 戻り値は dx 構造が使う upstream スケール `gs` の行ごとの指数（`build_dx == false` なら空）。
fn check_structure_stages(c: &StageCheck<'_>) -> Result<Vec<i32>, AutodiffError> {
    let h = c.hidden;
    let n = h as f64;
    if c.dy.iter().any(|v| !v.is_finite()) {
        return Err(reject("upstream に非有限要素がある"));
    }
    let wv = |i: usize| c.w.map_or(1.0, |w| f64::from(w[i]));
    let mut gs_exps = Vec::with_capacity(if c.build_dx { c.rows } else { 0 });
    // 行をまたぐ和（dw／db 構造の `Sum(0)` と、w への cotangent の行方向蓄積）。
    let mut col_dw = vec![0.0f64; h];
    let mut col_db = vec![0.0f64; h];
    let mut a_w = vec![0.0f64; h];
    for r in 0..c.rows {
        let src = &c.src[r * h..(r + 1) * h];
        let g: Vec<f64> = c.dy[r * h..(r + 1) * h]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
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
        let gw: Vec<f64> = (0..h).map(|i| g[i] * wv(i)).collect();
        for v in &gw {
            stage(*v, "dxhat = g·w")?;
        }

        // 二階の逆伝播で行内に集まる cotangent の上界（seed の各要素の絶対値 ≤ 1）。
        let mut a_xhat = vec![0.0f64; h];
        let mut a_g = vec![0.0f64; h];
        let mut a_rs = 0.0f64;

        if c.build_dx {
            let m = gw.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let ge = pow2_ceil_exponent(m);
            gs_exps.push(ge);
            let gs = 2f64.powi(ge);
            let gsh = gs / n;
            let dxhat: Vec<f64> = gw.iter().map(|v| v / gs).collect();
            // forward: dot_gs = Σ(dxhat·xhat)·gs/hidden（LayerNorm は sum_gs も）・inner・dx。
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
                a_g[i] += a_gw * wv(i).abs();
                a_w[i] += a_gw * g[i].abs();
            }
        }
        if c.build_dw {
            // dw = Σ_rows g·xhat。seed u_j が各行へ broadcast される。
            for i in 0..h {
                let prod = g[i].abs() * xhat[i].abs();
                stage(prod, "g·xhat")?;
                col_dw[i] += prod;
                a_xhat[i] += g[i].abs();
                a_g[i] += xhat[i].abs();
            }
        }
        if c.build_db {
            for i in 0..h {
                col_db[i] += g[i].abs();
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
        stage(*v, "dw 構造の行をまたぐ和")?;
    }
    for v in &col_db {
        stage(*v, "db 構造の行をまたぐ和")?;
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
    let (s_c, eps_s) = scale_consts(child, &exps, eps)?;
    let one = child.var_no_grad(&Tensor::full(&[1], 1.0f32)?);

    let xs = x2.div(&s_c)?;
    let ms = keep_dim(&xs.mul(&xs)?.mean(Some(1))?)?;
    let rstd_s = one.div(&ms.add(&eps_s)?.sqrt()?)?;
    let xhat = xs.mul(&rstd_s)?;

    let mut out = Vec::new();
    if need_x {
        let gs_c = gs_const(child, &gs_exps)?;
        let gw = match &w_m {
            Some(w) => g2.mul(w)?,
            None => g2,
        };
        let dxhat = gw.div(&gs_c)?;
        // `mean_dot·gs` は縮約の直後に `gs/hidden` を掛けて作る（逆伝播で縮約前に大きな
        // cotangent が復元されない）。
        let gsh = gs_over_hidden(child, &gs_c, hidden)?;
        let dot_gs = keep_dim(&dxhat.mul(&xhat)?.sum(Some(1))?)?.mul(&gsh)?;
        let inner = gw.sub(&xhat.mul(&dot_gs)?)?;
        let dx_s = rstd_s.div(&s_c)?.mul(&inner)?.reshape(&x_shape)?;
        let v = tag_with_host_value(child, &dx_s, build_tensor(dx_host, &x_shape))?;
        out.push((targets.input, v));
    }
    if need_w && let (Some(wid), Some(dw)) = (targets.weight, dw_host) {
        let dw_s = g2.mul(&xhat)?.sum(Some(0))?;
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
    let x_ref = x2.narrow(1, 0, 1)?;
    let d = x2.sub(&x_ref)?;
    let m = keep_dim(&d.mean(Some(1))?)?;
    let dev = d.sub(&m)?;
    let dev2 = dev.sub(&keep_dim(&dev.mean(Some(1))?)?)?;

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
    let (s_c, eps_s) = scale_consts(child, &exps, eps)?;
    let one = child.var_no_grad(&Tensor::full(&[1], 1.0f32)?);

    let devs = dev2.div(&s_c)?;
    let var_s = keep_dim(&devs.mul(&devs)?.mean(Some(1))?)?;
    let rstd_s = one.div(&var_s.add(&eps_s)?.sqrt()?)?;
    let xhat = devs.mul(&rstd_s)?;

    let mut out = Vec::new();
    if need_x {
        let gs_c = gs_const(child, &gs_exps)?;
        let gw = match &w_m {
            Some(w) => g2.mul(w)?,
            None => g2,
        };
        let dxhat = gw.div(&gs_c)?;
        let gsh = gs_over_hidden(child, &gs_c, hidden)?;
        let sum_gs = keep_dim(&dxhat.sum(Some(1))?)?.mul(&gsh)?;
        let dot_gs = keep_dim(&dxhat.mul(&xhat)?.sum(Some(1))?)?.mul(&gsh)?;
        let inner = gw.sub(&sum_gs)?.sub(&xhat.mul(&dot_gs)?)?;
        let dx_s = rstd_s.div(&s_c)?.mul(&inner)?.reshape(&x_shape)?;
        let v = tag_with_host_value(child, &dx_s, build_tensor(dx_host, &x_shape))?;
        out.push((targets.input, v));
    }
    if need_w && let (Some(wid), Some(dw)) = (targets.weight, dw_host) {
        let dw_s = g2.mul(&xhat)?.sum(Some(0))?;
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
