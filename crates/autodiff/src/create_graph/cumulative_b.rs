//! 累積 2 Op（`Cumprod`・`Logcumsumexp`）の `create_graph` 対応
//! （イシュー #3803・親 #3732・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §41、実装記録は §64）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `build_cgrads`（1 階 VJP の子テープ記録）から
//! 呼ばれる。forward の再生（`replay_op`）は公開 `Var::cumprod`／`logcumsumexp` への
//! 1 行委譲で足りるため親側に置く。線形再帰の展開は兄弟モジュール [`super::scan`] を使う。
//!
//! **方式**（値は 1 階 `grad.rs`／`tensor_core::cumulative` の VJP と同じホスト関数の結果を
//! `push_v3` で載せる。契約は REQ-2 のまま。構造は二階微分を流すための replayable な `Var` 列）:
//! - `Cumprod`（V3 + 除算を使わない scan）: 1 階は `d_x = L ⊙ S`。`S[a] = g[a] + x[a+1]·S[a+1]`
//!   は係数 `A[a] = x[a+1]` の線形再帰（scan 補助で `⌈log2 n⌉` 段）、`L` は排他的 prefix 積で
//!   `Concat([1], Narrow(y, 0, n-1))`（`y` は forward の出力の写し。追加の scan を要さない）。
//!   除算を使わないので零が何個あっても一階・二階とも厳密（除算形＋最初の零マスクでは
//!   `x=[2,0,3]`・`g=[0,0,1]` の `H[j,z]` を取りこぼす）。
//! - `Logcumsumexp`（V3 + 係数が [0,1] の再帰 scan）: 1 階は `d_x = w ⊙ T`、`w = exp(x - out)`・
//!   `T[j] = g[j] + c[j]·T[j+1]`・`c[j] = exp(out[j] - out[j+1])`（`out` は forward の出力の写し）。
//!   `c`・`w` は [0,1] に収まり overflow しない。
//!
//! 子テープへ積む Op は `Narrow`・`Concat`・`Mul`・`Add`・`Sub`・`Exp`・`Cumprod`／`Logcumsumexp`
//! （写し）・定数葉で、すべて `supports_create_graph() == true`（閉包性。§41.10）。
//!
//! **総括判定（§41.12 の具体化。§64.2）**: 子テープへ構造を積む**前**に cgrad 関数の中で行う
//! （upstream `g` は `build_cgrads` でしか得られないため）。拒否は型付き `Err(Backward)` で、
//! 親の rollback により `child` は空に戻る。
//! - 入口規則（§33.14 論点 3）: `g` の非有限と `x` の `±inf` は拒否、`x` の NaN は伝播。
//!   `max|g|·axis_len` が `STAGE_BOUND_GUARD`（`2^125`）以上も拒否。計算量の上限（[`judge_cost`]
//!   が [`JUDGE_BUDGET`] を超える入力）も、lane の走査より前に拒否する。
//! - 第 1 軸（overflow）: 構造の各段と、二階の逆伝播の「cotangent × 微分係数」の上界（seed の
//!   各要素の絶対値 1 以下）が非有限または `2^125` 以上なら拒否（[`cumprod_stage_bounds`]・
//!   [`lcse_stage_bounds`]）。NaN を含む lane は NaN を 0 に置き換えた lane で上界を取る
//!   （入力の NaN が届かない有限の中間値は置き換え後の値と同じなので、その上界になる）。
//! - ホストの 1 階（V3 で `cg.grad` に載る値）: 非有限の要素は、NaN で、かつ入力の NaN が届く
//!   位置にあるものだけを伝播として通す（[`host_nonfinite_check`]）。
//! - 第 2 軸（相殺・丸め・underflow。二階の要素ごとの判定）: seed が 1 要素（値 1）のときの二階の
//!   各要素（`x` と `g` への要素）を、子テープが実際に行う f32 の演算列（構造の forward・
//!   scan の各段・逆伝播の段・`out` から作る係数・写しの VJP）を同じ順で再現した `V_str` と、
//!   同じ演算列を f64 の参照値で評価した `V`・各項の絶対値の和 `M` の 2 通りで求め、
//!   `|V_str − V| + coeff·M` が絶対 1e-5 未満か相対の半分以内でなければ拒否する（[`entry_check`]。
//!   `loss_class.rs`・`norm_reparam.rs` と同じ形）。scan の桁落ち・その prefix 乗数での増幅・
//!   逆伝播の fan-in での相殺・係数の丸め・合算後の桁落ちは、どれも「二階の要素が参照から外れる」
//!   形で現れ、この 1 つの判定で拒否される。`Logcumsumexp` の係数 `c`・`w` の照合（§41.12 が
//!   名指しする判定）は設計どおり残す。
//!
//! GPU: `cumprod` は CUDA／Metal にカーネルがあり（経路区分 `K`）、`logcumsumexp` はホストへ
//! フォールバックする（経路区分 `H`）。実機 parity は未実測（Phase 10 の #3857・#3859）。
//! 第 2 軸の再現は CPU の演算順で、GPU の演算順は保証の対象外。

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::cumulative::{
    CumulativeLayout, cumulative_layout, logcumsumexp_vjp_host,
};
use fandhe_ai_tensor_core::tolerance::{
    ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE, composite_close_f32,
};

use super::scan::{
    Num, Rf, linear_scan, scan_backward, scan_cotangent_bound, scan_forward, scan_lane_host,
    scan_steps,
};
use super::{child_value, push_v3};
use crate::create_graph_norm::STAGE_BOUND_GUARD;
use crate::error::AutodiffError;
use crate::grad::cumprod_vjp_along;
use crate::tape::{NodeId, Tape, TapeNode, materialize_fallible};
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 累積軸まわりの (outer, axis_len, inner)。lane は `(outer, inner)` の組ごと。
#[derive(Clone, Copy)]
struct Dims {
    outer: usize,
    n: usize,
    inner: usize,
}

impl Dims {
    fn new(shape: &[usize], dim: usize) -> Self {
        Dims {
            outer: shape[..dim].iter().product(),
            n: shape[dim],
            inner: shape[dim + 1..].iter().product(),
        }
    }

    fn lane<T: Copy>(&self, data: &[T], o: usize, i: usize) -> Vec<T> {
        (0..self.n)
            .map(|a| data[(o * self.n + a) * self.inner + i])
            .collect()
    }
}

/// 段の上界 `v` が `STAGE_BOUND_GUARD`（`2^125`）以上、または NaN なら `true`（拒否側に倒す）。
fn over_guard(v: f64) -> bool {
    v.is_nan() || v >= STAGE_BOUND_GUARD
}

/// 入口規則。`g` に非有限があれば拒否、`x` に `±inf` があれば拒否（NaN は通す）、
/// `max|g|·max(1, n)` が `STAGE_BOUND_GUARD` 以上でも拒否する。
fn check_entry(x: &[f32], g: &[f32], n: usize, what: &str) -> Result<(), AutodiffError> {
    let mut max_g = 0f64;
    for &v in g {
        if !v.is_finite() {
            return Err(invalid(format!(
                "{what} の上流勾配が非有限のため拒否（§41・§64 参照）"
            )));
        }
        max_g = max_g.max(f64::from(v).abs());
    }
    if x.iter().any(|v| v.is_infinite()) {
        return Err(invalid(format!(
            "{what} の入力に ±inf がある（NaN は伝播するが ±inf は拒否する。§41・§64 参照）"
        )));
    }
    if over_guard(max_g * (n.max(1) as f64)) {
        return Err(invalid(format!(
            "{what} の上流勾配が大きすぎ中間値が f32 範囲を超えうるため拒否（§41・§64 参照）"
        )));
    }
    Ok(())
}

/// ノード `id` の shape 整合を確かめ、入力 shape を返す。
fn checked_shape(
    parent_nodes: &[TapeNode],
    id: NodeId,
    input: NodeId,
    dim: usize,
    g: &Var<'_>,
) -> Result<Vec<usize>, AutodiffError> {
    let in_shape = parent_nodes[input.0].shape.clone();
    if dim >= in_shape.len() || parent_nodes[id.0].shape != in_shape || g.shape() != in_shape {
        return Err(invalid(format!(
            "累積 Op の shape 不整合（入力 {in_shape:?}・dim {dim}・記録値 {:?}・上流 {:?}。内部契約違反）",
            parent_nodes[id.0].shape,
            g.shape()
        )));
    }
    Ok(in_shape)
}

/// `dim` 軸の長さ 1 の定数（`fill` で満たす）を `shape` に合わせて作る。
fn unit_slab<'c>(
    child: &'c Tape,
    shape: &[usize],
    dim: usize,
    fill: f32,
) -> Result<Var<'c>, AutodiffError> {
    let mut s = shape.to_vec();
    s[dim] = 1;
    Ok(child.var_no_grad(&Tensor::full(&s, fill)?))
}

/// NaN を 0 に置き換えた写し（NaN を含む lane の第 1 軸の上界を取るため。モジュール doc）。
fn nan_to_zero(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|&x| if x.is_nan() { 0.0 } else { x })
        .collect()
}

// ---------------------------------------------------------------------
// 第 2 軸（二階の要素ごとの判定）と計算量の上限。`loss_class.rs::struct_check`・
// `norm_reparam.rs::entry_check` と同じ形（ヘルパーは共有しない。独立にマージするため）
// ---------------------------------------------------------------------

/// 第 2 軸の走査量の上限（`norm_reparam.rs` と同じ値）。走査量は [`judge_cost`] で数え、超える入力は
/// 計算量の fail-closed として拒否する（保守的。上限内の入力だけに保証を限る）。
const JUDGE_BUDGET: usize = 1 << 27;

/// 判定 1 lane の forward 側の走査量の係数（`n` 1 要素あたり。`steps = ⌈log2 n⌉`）。
const FWD_PER_STEP: usize = 22;
const FWD_BASE: usize = 40;
/// 判定 1 seed あたりの走査量の係数（`n` 1 要素あたり）。
const SEED_PER_STEP: usize = 3;
const SEED_BASE: usize = 16;

/// 総括判定の走査量（f32 の再現 1 系統が回す内側ループの反復回数の上界。f64 の参照 [`Rf`] も同じ
/// 回数を回す）。lane（`outer·inner` 本・長さ `n`）ごとに、forward 側が `n·(22·steps + 40)`
/// （lane の取り出し 4 本・NaN の検出・第 1 軸の段検査〈[`scan_lane_host`] `6·steps·n`・
/// [`scan_cotangent_bound`] `12·steps·n`・絶対値の VJP〉・ホストの 1 階の検査・係数の照合・
/// 第 2 軸の forward〈[`scan_forward`] `4·steps·n`〉）、seed（`n` 個）ごとに `n·(3·steps + 16)`
/// （[`scan_backward`] `3·steps·n`・seed と cotangent の組み立て・写しの VJP〈`cumprod_vjp_along`／
/// `logcumsumexp_vjp_host` の 2 パスと確保〉・`x` と `g` への判定 `2n`）。二階の要素は lane あたり
/// `2n²` 個あるので、seed 側が `n²` に比例する。`Cumprod`・`Logcumsumexp` で同じ式を使う
/// （両者の上界のうち大きい方）。
fn judge_cost(lanes: usize, n: usize) -> usize {
    let steps = scan_steps(n);
    let fwd = n.saturating_mul(steps.saturating_mul(FWD_PER_STEP).saturating_add(FWD_BASE));
    let per_seed = n.saturating_mul(
        steps
            .saturating_mul(SEED_PER_STEP)
            .saturating_add(SEED_BASE),
    );
    lanes.saturating_mul(fwd.saturating_add(n.saturating_mul(per_seed)))
}

fn judge_budget(cost: usize, what: &str) -> Result<(), AutodiffError> {
    if cost <= JUDGE_BUDGET {
        Ok(())
    } else {
        Err(invalid(format!(
            "{what}: 二階の総括判定の走査量 {cost} が上限 2^27 を超えるため拒否（保守的な fail-closed。§64 参照）"
        )))
    }
}

/// 第 2 軸の余裕の係数 `coeff = 2^-23`（`2u`・`u = 2^-24`。f32 の丸め 2 回分）。再現は子テープの
/// CPU の演算と値で一致する（`backward.rs` の逆順処理と `accumulate` の加算順・同じ `f32::exp`・同じ
/// 写しの VJP を使う。単体テスト `emulation_reproduces_child_tape_second_order` が代表入力で固定する）
/// ので、`|V_str − V|` が CPU での実際の誤差そのものになる。`coeff·M` は、f64 の参照 `V` 自身の丸め
/// （`2^-53` の桁で、二階の 1 項が通る演算の回数倍）と、再現が子テープと 1 回の丸めだけずれる場合への
/// 余裕で、相殺の大きい要素（`M ≫ |V|`）はこの項でも拒否される。CPU 以外（GPU の `exp` の実装・
/// cumprod のカーネル）の差は見込まない（保証の対象外。§64.4）。
const JUDGE_COEFF: f64 = 1.0 / 8_388_608.0;

/// 二階の 1 要素の判定。`r` は参照値 `V` と大きさ `M`、`s` は構造の f32 の再現値 `V_str`。
/// `|V_str − V| + coeff·M` が絶対 1e-5 未満か相対の半分以内（tolerance 定数は
/// `tensor-core::tolerance` から読み、値は変えない）でなければ拒否する。外すのは f32 の再現と参照の
/// **両方が NaN** で、参照の NaN が入力の NaN の届いたもの（taint 付き）の要素だけ（NaN は伝播させる
/// 契約）。片側だけ NaN・演算で生まれた NaN・どちらかが `±inf` は拒否する。
fn entry_check(r: Rf, s: f32, coeff: f64, what: &str) -> Result<(), AutodiffError> {
    if s.is_nan() && r.v.is_nan() && r.t {
        return Ok(());
    }
    let err = (f64::from(s) - r.v).abs() + coeff * r.m;
    if r.v.is_finite()
        && r.m.is_finite()
        && err.is_finite()
        && (err < ABSOLUTE_RESCUE_THRESHOLD || err <= (RELATIVE_TOLERANCE / 2.0) * r.v.abs())
    {
        return Ok(());
    }
    Err(invalid(format!(
        "{what}: 二階の要素が f32 構造の丸め・相殺・underflow・overflow で REQ-2 の判定を外れうるため拒否（§41・§64 参照）"
    )))
}

/// lane の二階の全要素（seed ごとの `x` と `g` への要素）を判定する。`seed` は seed `k` について
/// f32 の再現と f64 の参照の `(x への要素, g への要素)` を返す。
fn judge_lane(
    n: usize,
    coeff: f64,
    what: &str,
    mut seed: impl FnMut(usize) -> Result<SeedOut, AutodiffError>,
) -> Result<(), AutodiffError> {
    for k in 0..n {
        let o = seed(k)?;
        for (r, s) in o.x_ref.iter().zip(&o.x_str) {
            entry_check(*r, *s, coeff, what)?;
        }
        for (r, s) in o.g_ref.iter().zip(&o.g_str) {
            entry_check(*r, *s, coeff, what)?;
        }
    }
    Ok(())
}

/// seed 1 個分の二階（f32 の再現と f64 の参照）。
struct SeedOut {
    x_str: Vec<f32>,
    g_str: Vec<f32>,
    x_ref: Vec<Rf>,
    g_ref: Vec<Rf>,
}

/// seed `k` の単位ベクトルの `j` 成分。
fn unit<T: Num>(k: usize, j: usize, one: T) -> T {
    if j == k { one } else { T::zero() }
}

/// ホストの 1 階（V3 の値）の非有限の検査。非有限の要素は NaN で、かつ入力の NaN が届く位置に
/// あるものだけを伝播として通す。届く範囲は 1 階のホスト関数の依存範囲で、`excl_self` が `true`
/// （`Cumprod`: `d_x[k]` は `x_k` 以外の lane の全要素に依存）なら自分以外の NaN、`false`
/// （`Logcumsumexp`: lane の全要素に依存）なら lane の NaN。
fn host_nonfinite_check(
    host: &[f32],
    x: &[f32],
    excl_self: bool,
    what: &str,
) -> Result<(), AutodiffError> {
    let nan_count = x.iter().filter(|v| v.is_nan()).count();
    for (k, &h) in host.iter().enumerate() {
        if h.is_finite() {
            continue;
        }
        let reach = if excl_self {
            nan_count > usize::from(x[k].is_nan())
        } else {
            nan_count > 0
        };
        if !(h.is_nan() && reach) {
            return Err(invalid(format!(
                "{what} の 1 階のホスト値に、入力の NaN に依存しない非有限がある（§41・§64 参照）"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Cumprod
// ---------------------------------------------------------------------

/// `Cumprod` の第 1 軸（overflow）。構造の各段（`A`・`S`・`L`・`L⊙S`）をホストで同じ f32 演算で
/// 再計算し、非有限または `2^125` 以上なら拒否。さらに二階の逆伝播の各段（「cotangent × 微分係数」）を、
/// seed の各要素の絶対値を 1 として絶対値の f64 で歩いた上界（scan 側は [`scan_cotangent_bound`]、
/// `L` 側は絶対値の入力を 1 階と同じ cumprod の VJP に通す）も同じ閾値で判定する。0 への underflow は
/// この軸では拒否しない（二階の要素への影響は第 2 軸が見る）。NaN を含む lane は呼び出し側が NaN を
/// 0 に置き換えた `x`・`y` を渡す。
fn cumprod_stage_bounds(xl: &[f32], gl: &[f32], yl: &[f32]) -> Result<(), AutodiffError> {
    let n = xl.len();
    let mut a = vec![0f32; n];
    a[..n - 1].copy_from_slice(&xl[1..]);
    let (s, stage_max) = scan_lane_host(gl, &a, None);
    if over_guard(stage_max) {
        return Err(invalid(
            "Cumprod の scan 構造の段が 2^125 以上または非有限のため拒否（§41・§64 参照）".into(),
        ));
    }
    let mut l = vec![1f32; n];
    l[1..].copy_from_slice(&yl[..n - 1]);
    let mut cot_s = vec![0f64; n];
    let mut cot_y = vec![0f32; n];
    for j in 0..n {
        let prod = l[j] * s[j];
        if over_guard(f64::from(l[j]).abs()) || over_guard(f64::from(prod).abs()) {
            return Err(invalid(
                "Cumprod の排他的 prefix 積または出力段が 2^125 以上または非有限のため拒否（§41・§64 参照）"
                    .into(),
            ));
        }
        // seed の各要素の絶対値 1 以下の前提で、`d_x = L ⊙ S` への cotangent は
        // `S` には `|L|`、`L` には `|S|` が流れる（`L[j] = y[j-1]` なので `y[j-1]` へ）。
        cot_s[j] = f64::from(l[j]).abs();
        if j >= 1 {
            cot_y[j - 1] = s[j].abs();
        }
    }
    let (_, cot_a, walk_max) = scan_cotangent_bound(gl, &a, &cot_s);
    if over_guard(walk_max) {
        return Err(invalid(
            "Cumprod の二階の逆伝播の中間値が 2^125 以上または非有限のため拒否（§41・§64 参照）"
                .into(),
        ));
    }
    // `y` 経由の cotangent は 1 階と同じ cumprod の VJP（`WideFloat`）に、絶対値の入力で通す。
    let abs_x = Tensor::new(xl.iter().map(|v| v.abs()).collect::<Vec<f32>>(), &[n])?;
    let cot_y_t = Tensor::new(cot_y, &[n])?;
    let cot_x_l = crate::eval::dense_vec(&cumprod_vjp_along(&abs_x, &cot_y_t, 0));
    for (m, v) in cot_x_l.iter().enumerate() {
        // `A[a] = x[a+1]` への cotangent も同じ `x[m]` に加わる。
        let from_a = if m >= 1 { cot_a[m - 1] } else { 0.0 };
        if over_guard(f64::from(*v).abs() + from_a) {
            return Err(invalid(
                "Cumprod の二階の入力 x への cotangent が 2^125 以上または非有限のため拒否（§41・§64 参照）"
                    .into(),
            ));
        }
    }
    Ok(())
}

/// `Cumprod` の子テープの二階（seed `k`）を [`Num`] で再現する。`l`・`s`・`hist` は構造の forward
/// （`L`・最終 `S`・scan の各段）、`vjp` は写し `y_m = cumprod(x_m)` の VJP（f32 は 1 階と同じ
/// `cumprod_vjp_along`、参照は [`cumprod_vjp_ref`]）。返り値は `(x への要素, g への要素)`。
///
/// 子テープの Op 列（`cumprod_cgrad`）は `A = Concat(Narrow(x_m, 1, n-1), 0)`・
/// `L = Concat(1, Narrow(y_m, 0, n-1))`・`S = scan(g, A)`・`L ⊙ S`・`push_v3`（恒等）の順で、
/// 逆伝播はその逆順: `Mul` が `L` へ `u·S`、`S` へ `u·L`、scan の逆伝播が `g` と `A` へ、`L` の
/// `Narrow` が `y_m[j] ← cot_L[j+1]`、`A` の `Narrow` が `x_m[m] ← cot_A[m-1]`、最後に写しの VJP が
/// `x_m` へ足す（`n = 1` は `L` が定数 1・`S = g` で、`x` への経路は無い）。
fn cumprod_seed<T: Num>(
    k: usize,
    one: T,
    l: &[T],
    s: &[T],
    hist: &[(Vec<T>, Vec<T>)],
    vjp: impl FnOnce(Vec<T>) -> Result<Vec<T>, AutodiffError>,
) -> Result<(Vec<T>, Vec<T>), AutodiffError> {
    let n = l.len();
    if n == 1 {
        return Ok((vec![T::zero()], vec![unit(k, 0, one).mul(l[0])]));
    }
    let cot_l: Vec<T> = (0..n).map(|j| unit(k, j, one).mul(s[j])).collect();
    let cot_s: Vec<T> = (0..n).map(|j| unit(k, j, one).mul(l[j])).collect();
    let (cot_g, cot_a) = scan_backward(hist, cot_s);
    let cot_y: Vec<T> = (0..n)
        .map(|j| if j + 1 < n { cot_l[j + 1] } else { T::zero() })
        .collect();
    let from_y = vjp(cot_y)?;
    let cot_x = (0..n)
        .map(|m| {
            let from_a = if m >= 1 { cot_a[m - 1] } else { T::zero() };
            from_a.add(from_y[m])
        })
        .collect();
    Ok((cot_x, cot_g))
}

/// `cumprod_vjp_along`（`WideFloat`）と同じ演算順の f64 参照（`S` の後方再帰 → `L·S` と `L` の累積）。
/// `WideFloat` の非有限の扱い（`NaN·0 = NaN`）とも同じで、NaN の位置が一致する。
fn cumprod_vjp_ref(x: &[Rf], cot_y: &[Rf]) -> Vec<Rf> {
    let n = x.len();
    let mut s = vec![Rf::zero(); n];
    let mut run = Rf::zero();
    for a in (0..n).rev() {
        run = if a + 1 == n {
            cot_y[a]
        } else {
            cot_y[a].add(x[a + 1].mul(run))
        };
        s[a] = run;
    }
    let mut l = Rf::k(1.0);
    let mut out = vec![Rf::zero(); n];
    for a in 0..n {
        out[a] = l.mul(s[a]);
        l = l.mul(x[a]);
    }
    out
}

/// `Cumprod` の 1 lane の第 2 軸（モジュール doc）。`yl` は forward の記録値（f32）。
fn cumprod_lane_judge(xl: &[f32], gl: &[f32], yl: &[f32], coeff: f64) -> Result<(), AutodiffError> {
    let n = xl.len();
    // 構造の forward（f32）。`L` は記録値 `y` の写し、`A` は `x` を 1 つ詰めて末尾 0。
    let l32: Vec<f32> = (0..n)
        .map(|j| if j == 0 { 1.0 } else { yl[j - 1] })
        .collect();
    let a32: Vec<f32> = (0..n)
        .map(|j| if j + 1 < n { xl[j + 1] } else { 0.0 })
        .collect();
    let (s32, h32) = scan_forward::<f32>(gl, &a32);
    // f64 の参照。`L` は f64 の prefix 積（記録値の丸めを含まない）。
    let xr: Vec<Rf> = xl.iter().map(|&v| Rf::k(f64::from(v))).collect();
    let gr: Vec<Rf> = gl.iter().map(|&v| Rf::k(f64::from(v))).collect();
    let mut lr = vec![Rf::k(1.0); n];
    for j in 1..n {
        lr[j] = lr[j - 1].mul(xr[j - 1]);
    }
    let ar: Vec<Rf> = (0..n)
        .map(|j| if j + 1 < n { xr[j + 1] } else { Rf::zero() })
        .collect();
    let (sr, hr) = scan_forward::<Rf>(&gr, &ar);
    let x_t = Tensor::new(xl.to_vec(), &[n])?;
    judge_lane(n, coeff, "Cumprod", |k| {
        let (x_str, g_str) = cumprod_seed::<f32>(k, 1.0, &l32, &s32, &h32, |cot_y| {
            let t = Tensor::new(cot_y, &[n])?;
            Ok(crate::eval::dense_vec(&cumprod_vjp_along(&x_t, &t, 0)))
        })?;
        let (x_ref, g_ref) = cumprod_seed::<Rf>(k, Rf::k(1.0), &lr, &sr, &hr, |cot_y| {
            Ok(cumprod_vjp_ref(&xr, &cot_y))
        })?;
        Ok(SeedOut {
            x_str,
            g_str,
            x_ref,
            g_ref,
        })
    })
}

/// `Cumprod` の総括判定（モジュール doc）。`x`・`g`・`y`・`host` は同 shape の密な値
/// （`host` は 1 階のホスト値）。
fn check_cumprod_stages(
    x: &[f32],
    g: &[f32],
    y: &[f32],
    host: &[f32],
    d: Dims,
) -> Result<(), AutodiffError> {
    judge_budget(judge_cost(d.outer.saturating_mul(d.inner), d.n), "Cumprod")?;
    let coeff = JUDGE_COEFF;
    for o in 0..d.outer {
        for i in 0..d.inner {
            let xl = d.lane(x, o, i);
            let gl = d.lane(g, o, i);
            let yl = d.lane(y, o, i);
            let hl = d.lane(host, o, i);
            if xl.iter().any(|v| v.is_nan()) {
                cumprod_stage_bounds(&nan_to_zero(&xl), &gl, &nan_to_zero(&yl))?;
            } else {
                cumprod_stage_bounds(&xl, &gl, &yl)?;
            }
            host_nonfinite_check(&hl, &xl, true, "Cumprod")?;
            cumprod_lane_judge(&xl, &gl, &yl, coeff)?;
        }
    }
    Ok(())
}

/// `Cumprod` の d_input を子テープへ記録する（V3）。`x_m`・`y_m` は入力・出力の子テープ上の写し。
pub(super) fn cumprod_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, usize),
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (id, input, dim) = spec;
    let (x_m, y_m) = mirrors;
    let in_shape = checked_shape(parent_nodes, id, input, dim, g)?;
    if in_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&in_shape)?));
    }
    let d = Dims::new(&in_shape, dim);
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let out_val = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let g_value = child_value(child, g)?;
    let (x, y, gv) = (
        crate::eval::dense_vec(input_val),
        crate::eval::dense_vec(out_val),
        crate::eval::dense_vec(&g_value),
    );
    check_entry(&x, &gv, d.n, "Cumprod")?;
    let host = cumprod_vjp_along(input_val, &g_value, dim);
    check_cumprod_stages(&x, &gv, &y, &crate::eval::dense_vec(&host), d)?;

    let n = d.n;
    let (l, s) = if n == 1 {
        (unit_slab(child, &in_shape, dim, 1.0)?, *g)
    } else {
        let a = Var::cat(
            &[
                x_m.narrow(dim, 1, n - 1)?,
                unit_slab(child, &in_shape, dim, 0.0)?,
            ],
            dim,
        )?;
        let l = Var::cat(
            &[
                unit_slab(child, &in_shape, dim, 1.0)?,
                y_m.narrow(dim, 0, n - 1)?,
            ],
            dim,
        )?;
        (l, linear_scan(child, g, &a, dim, None)?)
    };
    let composed = l.mul(&s)?;
    push_v3(child, &composed, host)
}

// ---------------------------------------------------------------------
// Logcumsumexp
// ---------------------------------------------------------------------

/// `log(exp(a) + exp(b))` を f64 で安定に求める（`tensor_core::cumulative` の私的関数と
/// 同じ分岐順。参照値の再計算専用で、tensor-core の公開面は増やさない）。
fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b && !a.is_finite() {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

fn reject_coeff(what: &str) -> AutodiffError {
    invalid(format!(
        "Logcumsumexp の {what} が f64 参照と REQ-2 統一複合判定で一致せず（f32 の出力の丸めによる）拒否（§41・§64 参照）"
    ))
}

/// 1 lane の f64 の prefix（`out64`）と参照係数 `c64[j] = exp(out64[j] - out64[j+1])`（末尾 0）・
/// `w64[j] = exp(x[j] - out64[j])`。
fn lcse_ref_coeffs(xl: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = xl.len();
    let mut out64 = vec![0f64; n];
    for a in 0..n {
        let v = f64::from(xl[a]);
        out64[a] = if a == 0 {
            v
        } else {
            log_add_exp(out64[a - 1], v)
        };
    }
    let c64 = (0..n)
        .map(|j| {
            if j + 1 < n {
                (out64[j] - out64[j + 1]).exp()
            } else {
                0.0
            }
        })
        .collect();
    let w64 = (0..n)
        .map(|j| (f64::from(xl[j]) - out64[j]).exp())
        .collect();
    (out64, c64, w64)
}

/// `Logcumsumexp` の第 1 軸と係数の照合。f32 の出力に `±inf` があれば拒否し、入力に NaN が無いのに
/// NaN があっても拒否する。f32 構造の係数 `c`・`w` を f64 prefix の参照係数と REQ-2 の統一複合判定
/// （`tolerance::composite_close_f32`）で比べ（§41.12 が名指しする係数の食い違いの判定。両方 NaN の
/// 要素は伝播として外し、片側だけ NaN なら拒否）、scan の各段の f32 再計算と、二階の逆伝播の各段
/// （「cotangent × 微分係数」。seed の各要素の絶対値 1 以下）を絶対値の f64 で歩いた上界が、非有限または
/// `2^125` 以上なら拒否する（NaN の値は 0 に置き換えて上界を取る。モジュール doc）。
fn lcse_stage_bounds(xl: &[f32], gl: &[f32], ol: &[f32]) -> Result<(), AutodiffError> {
    let n = xl.len();
    let has_nan = xl.iter().any(|v| v.is_nan());
    if ol
        .iter()
        .any(|v| v.is_infinite() || (v.is_nan() && !has_nan))
    {
        return Err(invalid(
            "Logcumsumexp の出力が f32 で非有限のため拒否（§41・§64 参照）".into(),
        ));
    }
    let (_, c64, w64) = lcse_ref_coeffs(xl);
    let close = |s: f32, r: f64| (s.is_nan() && r.is_nan()) || composite_close_f32(s, r as f32);
    let mut c32 = vec![0f32; n];
    for j in 0..n.saturating_sub(1) {
        c32[j] = (ol[j] - ol[j + 1]).exp();
        if !close(c32[j], c64[j]) {
            return Err(reject_coeff("係数 exp(out[j]-out[j+1])"));
        }
    }
    let mut w32 = vec![0f32; n];
    for j in 0..n {
        w32[j] = (xl[j] - ol[j]).exp();
        if !close(w32[j], w64[j]) {
            return Err(reject_coeff("係数 exp(x-out)"));
        }
    }
    let c_abs = nan_to_zero(&c32);
    let (t32, stage_max) = scan_lane_host(gl, &c_abs, None);
    if over_guard(stage_max) {
        return Err(invalid(
            "Logcumsumexp の scan 構造の段が 2^125 以上または非有限のため拒否（§41・§64 参照）"
                .into(),
        ));
    }
    // 二階の逆伝播の各段の上界（seed の各要素の絶対値 1 以下。`Cumprod` と同じ流儀）。`w ⊙ T` への
    // cotangent は `T` へ `|w|`、`w` へ `|T|` が流れ、`Exp` で `|T|·w`、scan の逆伝播で `A`（`c`）へ、
    // `c` の `Exp` で `·c`、`y` への合算を経て写しの VJP（f64 の参照係数）へ入る。NaN は 0 に置き換える。
    let abs0 = |v: f64| if v.is_nan() { 0.0 } else { v.abs() };
    let cot_t: Vec<f64> = w32.iter().map(|&v| abs0(f64::from(v))).collect();
    let (_, cot_a, walk_max) = scan_cotangent_bound(gl, &c_abs, &cot_t);
    let cot_z: Vec<f64> = (0..n).map(|j| abs0(f64::from(t32[j])) * cot_t[j]).collect();
    let cot_dy: Vec<f64> = (0..n).map(|j| cot_a[j] * f64::from(c_abs[j])).collect();
    let mut vjp = vec![0f64; n];
    let mut t = 0f64;
    let mut bound = walk_max;
    for a in (0..n).rev() {
        let cot_y = cot_z[a] + if a >= 1 { cot_dy[a - 1] } else { 0.0 } + cot_dy[a];
        t = cot_y + if a + 1 < n { abs0(c64[a]) * t } else { 0.0 };
        vjp[a] = abs0(w64[a]) * t;
        bound = bound.max(cot_y).max(t).max(cot_z[a] + vjp[a]);
    }
    if over_guard(bound) {
        return Err(invalid(
            "Logcumsumexp の二階の逆伝播の中間値が 2^125 以上または非有限のため拒否（§41・§64 参照）"
                .into(),
        ));
    }
    Ok(())
}

/// `Logcumsumexp` の子テープの二階（seed `k`）を [`Num`] で再現する。`w`・`c`・`t`・`hist` は構造の
/// forward（`exp(x - out)`・`exp(out[j] - out[j+1])`〈末尾 0〉・最終 `T`・scan の各段）、`vjp` は写し
/// `y_m = logcumsumexp(x_m)` の VJP（f32 は 1 階と同じ `logcumsumexp_vjp_host`、参照は [`lcse_vjp_ref`]）。
///
/// 子テープの Op 列（`logcumsumexp_cgrad`）は `Q1 = Narrow(y_m, 0, n-1)`・`Q2 = Narrow(y_m, 1, n-1)`・
/// `c = Exp(Q1 - Q2)`・`A = Concat(c, 0)`・`T = scan(g, A)`・`w = Exp(x_m - y_m)`・`w ⊙ T`・`push_v3`
/// の順で、逆伝播はその逆順: `Mul` が `w` へ `u·T`、`T` へ `u·w`、`Exp` が `cot·w`、`Sub` が `x_m` へ
/// `+cot`・`y_m` へ `−cot`、scan の逆伝播が `g` と `A` へ、`c` の `Exp` が `cot·c`、`Q2` の `Narrow`
/// （`y_m[j] ← −cot_dy[j-1]`）、`Q1` の `Narrow`（`y_m[j] ← cot_dy[j]`）、最後に写しの VJP が `x_m`
/// へ足す。`y_m` への合算はこの順（`−cot_z` → `Q2` → `Q1`）で、合算後の桁落ちも再現に入る。
fn lcse_seed<T: Num>(
    k: usize,
    one: T,
    coef: (&[T], &[T]),
    t: &[T],
    hist: &[(Vec<T>, Vec<T>)],
    vjp: impl FnOnce(Vec<T>) -> Result<Vec<T>, AutodiffError>,
) -> Result<(Vec<T>, Vec<T>), AutodiffError> {
    let (w, c) = coef;
    let n = w.len();
    let cot_w: Vec<T> = (0..n).map(|j| unit(k, j, one).mul(t[j])).collect();
    let cot_t: Vec<T> = (0..n).map(|j| unit(k, j, one).mul(w[j])).collect();
    let cot_z: Vec<T> = (0..n).map(|j| cot_w[j].mul(w[j])).collect();
    let (cot_g, cot_a) = scan_backward(hist, cot_t);
    let cot_dy: Vec<T> = (0..n.saturating_sub(1))
        .map(|j| cot_a[j].mul(c[j]))
        .collect();
    let cot_y: Vec<T> = (0..n)
        .map(|j| {
            let mut acc = cot_z[j].neg();
            if j >= 1 {
                acc = acc.add(cot_dy[j - 1].neg());
            }
            if j + 1 < n {
                acc = acc.add(cot_dy[j]);
            }
            acc
        })
        .collect();
    let from_y = vjp(cot_y)?;
    let cot_x = (0..n).map(|m| cot_z[m].add(from_y[m])).collect();
    Ok((cot_x, cot_g))
}

/// `logcumsumexp_vjp_host` と同じ演算順の f64 参照（`T'[a] = cot_y[a] + c[a]·T'[a+1]`、
/// `d[a] = w[a]·T'[a]`）。`c`・`w` は f64 prefix の参照係数。
fn lcse_vjp_ref(c: &[Rf], w: &[Rf], cot_y: &[Rf]) -> Vec<Rf> {
    let n = cot_y.len();
    let mut out = vec![Rf::zero(); n];
    let mut t = Rf::zero();
    for a in (0..n).rev() {
        t = if a + 1 == n {
            cot_y[a]
        } else {
            cot_y[a].add(c[a].mul(t))
        };
        out[a] = w[a].mul(t);
    }
    out
}

/// `Logcumsumexp` の 1 lane の第 2 軸（モジュール doc）。`ol` は forward の記録値（f32）。
fn lcse_lane_judge(
    xl: &[f32],
    gl: &[f32],
    ol: &[f32],
    layout: &CumulativeLayout,
    coeff: f64,
) -> Result<(), AutodiffError> {
    let n = xl.len();
    // 構造の forward（f32）。係数は記録値 `out` の差と `x - out` の f32 の `Exp`。
    let c32: Vec<f32> = (0..n)
        .map(|j| {
            if j + 1 < n {
                (ol[j] - ol[j + 1]).exp()
            } else {
                0.0
            }
        })
        .collect();
    let w32: Vec<f32> = (0..n).map(|j| (xl[j] - ol[j]).exp()).collect();
    let (t32, h32) = scan_forward::<f32>(gl, &c32);
    // f64 の参照（f64 prefix の係数。係数の丸めを含まない）。
    let (_, c64, w64) = lcse_ref_coeffs(xl);
    let cr: Vec<Rf> = c64.iter().map(|&v| Rf::k(v)).collect();
    let wr: Vec<Rf> = w64.iter().map(|&v| Rf::k(v)).collect();
    let gr: Vec<Rf> = gl.iter().map(|&v| Rf::k(f64::from(v))).collect();
    let (tr, hr) = scan_forward::<Rf>(&gr, &cr);
    judge_lane(n, coeff, "Logcumsumexp", |k| {
        let (x_str, g_str) = lcse_seed::<f32>(k, 1.0, (&w32, &c32), &t32, &h32, |cot_y| {
            logcumsumexp_vjp_host(xl, &cot_y, layout).map_err(AutodiffError::Shape)
        })?;
        let (x_ref, g_ref) = lcse_seed::<Rf>(k, Rf::k(1.0), (&wr, &cr), &tr, &hr, |cot_y| {
            Ok(lcse_vjp_ref(&cr, &wr, &cot_y))
        })?;
        Ok(SeedOut {
            x_str,
            g_str,
            x_ref,
            g_ref,
        })
    })
}

/// `Logcumsumexp` の総括判定（モジュール doc）。`out` は forward の記録値（f32）、`host` は 1 階の
/// ホスト値。
fn check_logcumsumexp_stages(
    x: &[f32],
    g: &[f32],
    out: &[f32],
    host: &[f32],
    d: Dims,
) -> Result<(), AutodiffError> {
    judge_budget(
        judge_cost(d.outer.saturating_mul(d.inner), d.n),
        "Logcumsumexp",
    )?;
    let coeff = JUDGE_COEFF;
    let layout = cumulative_layout(&[d.n], 0).map_err(AutodiffError::Shape)?;
    for o in 0..d.outer {
        for i in 0..d.inner {
            let xl = d.lane(x, o, i);
            let gl = d.lane(g, o, i);
            let ol = d.lane(out, o, i);
            let hl = d.lane(host, o, i);
            lcse_stage_bounds(&xl, &gl, &ol)?;
            host_nonfinite_check(&hl, &xl, false, "Logcumsumexp")?;
            lcse_lane_judge(&xl, &gl, &ol, &layout, coeff)?;
        }
    }
    Ok(())
}

/// `Logcumsumexp` の d_input を子テープへ記録する（V3）。`x_m`・`y_m` は入力・出力の子テープ上の写し。
pub(super) fn logcumsumexp_cgrad<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    spec: (NodeId, NodeId, usize),
    mirrors: (&Var<'c>, &Var<'c>),
    g: &Var<'c>,
) -> Result<Var<'c>, AutodiffError> {
    let (id, input, dim) = spec;
    let (x_m, y_m) = mirrors;
    let in_shape = checked_shape(parent_nodes, id, input, dim, g)?;
    if in_shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&in_shape)?));
    }
    let d = Dims::new(&in_shape, dim);
    let input_val = materialize_fallible(parent_nodes, parent.ops(), input)?;
    let out_val = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let g_value = child_value(child, g)?;
    let (x, y, gv) = (
        crate::eval::dense_vec(input_val),
        crate::eval::dense_vec(out_val),
        crate::eval::dense_vec(&g_value),
    );
    check_entry(&x, &gv, d.n, "Logcumsumexp")?;
    let layout = cumulative_layout(&in_shape, dim).map_err(AutodiffError::Shape)?;
    let host = logcumsumexp_vjp_host(&x, &gv, &layout).map_err(AutodiffError::Shape)?;
    check_logcumsumexp_stages(&x, &gv, &y, &host, d)?;

    let n = d.n;
    let t = if n == 1 {
        *g
    } else {
        let c = y_m
            .narrow(dim, 0, n - 1)?
            .sub(&y_m.narrow(dim, 1, n - 1)?)?
            .exp();
        let a = Var::cat(&[c, unit_slab(child, &in_shape, dim, 0.0)?], dim)?;
        linear_scan(child, g, &a, dim, None)?
    };
    let w = x_m.sub(y_m)?.exp();
    let composed = w.mul(&t)?;
    push_v3(child, &composed, Tensor::new(host, &in_shape)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::naive_ops;

    fn d1(n: usize) -> Dims {
        Dims {
            outer: 1,
            n,
            inner: 1,
        }
    }

    fn cumprod_y(x: &[f32]) -> Vec<f32> {
        let mut acc = 1f32;
        x.iter()
            .map(|v| {
                acc *= *v;
                acc
            })
            .collect()
    }

    fn lse_out(x: &[f32]) -> Vec<f32> {
        let mut acc = 0f64;
        x.iter()
            .enumerate()
            .map(|(a, v)| {
                acc = if a == 0 {
                    f64::from(*v)
                } else {
                    log_add_exp(acc, f64::from(*v))
                };
                acc as f32
            })
            .collect()
    }

    fn cp(x: &[f32], g: &[f32]) -> Result<(), AutodiffError> {
        let n = x.len();
        check_entry(x, g, n, "Cumprod")?;
        let host = cumprod_vjp_along(
            &Tensor::new(x.to_vec(), &[n]).unwrap(),
            &Tensor::new(g.to_vec(), &[n]).unwrap(),
            0,
        );
        check_cumprod_stages(x, g, &cumprod_y(x), &crate::eval::dense_vec(&host), d1(n))
    }

    fn lse(x: &[f32], g: &[f32]) -> Result<(), AutodiffError> {
        let n = x.len();
        check_entry(x, g, n, "Logcumsumexp")?;
        let layout = cumulative_layout(&[n], 0).unwrap();
        let host = logcumsumexp_vjp_host(x, g, &layout).unwrap();
        check_logcumsumexp_stages(x, g, &lse_out(x), &host, d1(n))
    }

    #[test]
    fn cumprod_check_accepts_underflow_and_zeros_rejects_overflow() {
        // 0 への underflow（L[2] = 1e-60 → 0）は拒否しない。
        assert!(cp(&[1e-30, 1e-30, 1e30], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[2.0, 3.0, 4.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[2.0, 0.0, 3.0], &[0.0, 0.0, 1.0]).is_ok());
        assert!(cp(&[0.0, 0.0, 3.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[0.0, 2.0, 0.0, 1.0], &[1.0, 1.0, 1.0, 1.0]).is_ok());
        // 長さ 1（`L` は定数 1・`x` への二階は無い）。
        assert!(cp(&[3.0], &[2.0]).is_ok());
        // NaN は伝播（両方 NaN の要素だけを判定から外す）。
        assert!(cp(&[1.0, f32::NAN, 2.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(cp(&[f32::NAN], &[1.0]).is_ok());
        // 部分積が 2^125 を超えるものは拒否（零を含んでも W の上界で拒否）。
        let mut big = vec![1e30f32; 40];
        big[0] = 0.0;
        assert!(cp(&big, &[1.0; 40]).is_err());
        assert!(cp(&[1e30, 1e30, 1e30], &[1.0, 1.0, 1.0]).is_err());
        // NaN を含む lane でも、NaN の届かない有限の段の overflow は拒否する。
        assert!(cp(&[1e30, 1e30, 1e30, f32::NAN], &[1.0; 4]).is_err());
        // 二階の cotangent が実際に f32 に収まる大きな入力は受理される（部分積 1e60 を作らない）。
        assert!(cp(&[1e30, 1e30], &[1.0, 1.0]).is_ok());
        // 入口規則。
        assert!(cp(&[1.0, f32::INFINITY], &[1.0, 1.0]).is_err());
        assert!(cp(&[1.0, 2.0], &[f32::NAN, 1.0]).is_err());
        assert!(cp(&[1.0, 2.0], &[3e38, 1.0]).is_err());
    }

    /// PR #3959 のレビューで指摘された入力（すべて第 2 軸の 1 つの判定で拒否される）と、相殺・
    /// 増幅の無い同形の受理側。
    #[test]
    fn cumprod_check_rejects_reviewed_cancellations() {
        // scan の桁落ち: f32 scan は S[1] = (1e10+1)-1e10 = 0、真値は 1（H[1,0] が 0 になる）。
        assert!(cp(&[1.0; 4], &[0.0, 1e10, 1.0, -1e10]).is_err());
        assert!(cp(&[1.0; 4], &[0.0, 1e10, 1.0, 1e10]).is_ok());
        // prefix 乗数での増幅: S[2] が 0 に桁落ち（真値 1e-20）し、H[2,0] が 0（解析値 1）。
        let g = [0.0, 0.0, 1e-10, 1e-20, -1e-10];
        assert!(cp(&[1e20, 1.0, 1.0, 1.0, 1.0], &g).is_err());
        assert!(cp(&[1.0; 5], &g).is_ok());
        // 逆伝播の fan-in での相殺（P1 PRRT_kwDOTuUCJc6yo9xp）: H[0,1] は構造上
        // 1e10 + f32(-1e10 + 1) = 0、解析値 1。forward の S は別の結合順で 1 になる。
        assert!(cp(&[1.0; 4], &[1e10, 1e10, -1e10, 1.0]).is_err());
        assert!(cp(&[1.0; 4], &[1.0, 1.0, -1.0, 1.0]).is_ok());
    }

    #[test]
    fn logcumsumexp_check_rejects_reviewed_cancellations() {
        // 二階の桁落ち増幅: H[0,0] が f32 で 0、解析値 約 20.6。
        assert!(lse(&[0.0, -20.0], &[0.0, 1e10]).is_err());
        assert!(lse(&[0.0, -20.0], &[0.0, 1.0]).is_ok());
        // 各項は一致するが合算で相殺する（H[0,0] 解析値 0）。
        assert!(lse(&[0.0; 3], &[1001.0, -8000.0, 9000.0]).is_err());
        // 係数の丸め（P1 PRRT_kwDOTuUCJc6ypSUU）: f32 の out から作る c・w の誤差で
        // 構造の H[0,0] は約 3.83e-4、解析値 0。
        assert!(lse(&[10000.0; 3], &[0.0, -8.0, 9.0]).is_err());
        // オフセットの無い同形で上流を小さくすると受理される。
        assert!(lse(&[0.0; 3], &[0.0, -0.8, 0.9]).is_ok());
    }

    #[test]
    fn logcumsumexp_check_accepts_moderate_rejects_rounding_loss() {
        assert!(lse(&[0.0, 0.0], &[0.0, 1.0]).is_ok());
        assert!(lse(&[-1000.0, 1000.0], &[1.0, 1.0]).is_ok());
        assert!(lse(&[-100.0, -100.0, 80.0], &[1.0, 1.0, 1.0]).is_ok());
        assert!(lse(&[3.0, 3.0, 3.0, 3.0], &[1.0, -1.0, 1.0, 0.5]).is_ok());
        assert!(lse(&[2.0], &[1.5]).is_ok());
        assert!(lse(&[1.0, f32::NAN], &[1.0, 1.0]).is_ok());
        assert!(lse(&[f32::NAN, 1.0, 2.0], &[1.0, 1.0, 1.0]).is_ok());
        // f32 の出力が ln2 の差を失う（w=1 対 参照 0.5）。
        assert!(lse(&[1e8, 1e8], &[0.0, 1.0]).is_err());
        assert!(lse(&[1e8, 1e8 + 1.0, 1e8], &[1.0, 1.0, 1.0]).is_err());
        // 係数の相対誤差が 1e-3 を超える。
        assert!(lse(&[5e4, 5e4], &[1.0, 1.0]).is_err());
        // 入口規則。
        assert!(lse(&[f32::NEG_INFINITY, 0.0], &[1.0, 1.0]).is_err());
        assert!(lse(&[0.0, 0.0], &[3e38, 1.0]).is_err());
        assert!(lse(&[0.0, 0.0], &[3e38, -3e38]).is_err());
    }

    /// `Logcumsumexp` の第 1 軸は二階の逆伝播の上界も見る（seed の各要素の絶対値 1 以下では `y` への
    /// cotangent が `n²·max|g|` の桁になり、入口規則 `max|g|·n < 2^125` だけでは押さえられない）。
    #[test]
    fn logcumsumexp_stage_bounds_walk_second_order_backward() {
        let n = 64usize;
        let x = vec![0.0f32; n];
        let out = lse_out(&x);
        let big = vec![2f32.powi(118); n];
        assert!(check_entry(&x, &big, n, "t").is_ok());
        assert!(lcse_stage_bounds(&x, &big, &out).is_err());
        assert!(lcse_stage_bounds(&x, &vec![1.0f32; n], &out).is_ok());
        // NaN を含む lane も、NaN を 0 に置き換えた上界で、NaN の届かない有限の部分を同じく判定する。
        let mut xn = x.clone();
        xn[n - 1] = f32::NAN;
        assert!(lcse_stage_bounds(&xn, &big, &lse_out(&xn)).is_err());
        assert!(lcse_stage_bounds(&xn, &vec![1.0f32; n], &lse_out(&xn)).is_ok());
    }

    #[test]
    fn entry_check_rules() {
        let c = 1e-6;
        assert!(entry_check(Rf::k(1.0), 1.0, c, "t").is_ok());
        // 相殺: 値は一致しても M が大きく V が小さい。
        let cancel = Rf {
            v: 1.0,
            m: 2e10,
            t: false,
        };
        assert!(entry_check(cancel, 1.0, c, "t").is_err());
        // 値の食い違い（係数の丸め・underflow）。
        assert!(entry_check(Rf::k(1.0), 0.0, c, "t").is_err());
        // 小さな値は絶対救済で通る。
        assert!(entry_check(Rf::k(1e-7), 0.0, c, "t").is_ok());
        // NaN: 両方 NaN で taint 付きだけ外す。
        assert!(entry_check(Rf::k(f64::NAN), f32::NAN, c, "t").is_ok());
        let untainted = Rf {
            v: f64::NAN,
            m: f64::NAN,
            t: false,
        };
        assert!(entry_check(untainted, f32::NAN, c, "t").is_err());
        assert!(entry_check(Rf::k(1.0), f32::NAN, c, "t").is_err());
        assert!(entry_check(Rf::k(f64::NAN), 1.0, c, "t").is_err());
        assert!(entry_check(Rf::k(1.0), f32::INFINITY, c, "t").is_err());
    }

    #[test]
    fn host_nonfinite_check_rules() {
        // Cumprod: d_x[k] は自分以外の NaN に依存する。
        assert!(host_nonfinite_check(&[f32::NAN, 1.0], &[1.0, f32::NAN], true, "t").is_ok());
        assert!(host_nonfinite_check(&[1.0, f32::NAN], &[1.0, f32::NAN], true, "t").is_err());
        assert!(host_nonfinite_check(&[f32::INFINITY], &[1.0], true, "t").is_err());
        // Logcumsumexp: lane の NaN に依存する。
        assert!(host_nonfinite_check(&[1.0, f32::NAN], &[1.0, f32::NAN], false, "t").is_ok());
        assert!(host_nonfinite_check(&[f32::NAN, 1.0], &[1.0, 2.0], false, "t").is_err());
    }

    /// 走査量は二階の要素数（lane あたり `2n²`）に比例する seed 側を数え、上限を超える入力を拒否する
    /// （P2 PRRT_kwDOTuUCJc6ypSUd: 長さ 65536 の lane は上限を大きく超える）。
    #[test]
    fn judge_cost_counts_second_order_entries() {
        for n in [1usize, 2, 3, 8, 100, 1000] {
            let cost = judge_cost(1, n);
            assert!(cost >= 2 * n * n, "n={n}: {cost}");
            let steps = scan_steps(n);
            assert_eq!(
                cost,
                n * (22 * steps + 40) + n * n * (3 * steps + 16),
                "n={n}"
            );
        }
        assert!(judge_budget(JUDGE_BUDGET, "t").is_ok());
        assert!(judge_budget(JUDGE_BUDGET + 1, "t").is_err());
        assert!(judge_budget(judge_cost(1, 1024), "t").is_ok());
        assert!(judge_budget(judge_cost(1, 2048), "t").is_err());
        assert!(judge_budget(judge_cost(1, 65536), "t").is_err());
        assert!(judge_budget(judge_cost(usize::MAX, usize::MAX), "t").is_err());
        // lane が多いと短い軸でも上限を超える。
        assert!(judge_budget(judge_cost(1 << 20, 8), "t").is_err());
        assert!(judge_budget(judge_cost(64, 8), "t").is_ok());
        // 判定の入口（lane の走査の前）で拒否される。
        let n = 65536usize;
        let x = vec![1.0f32; n];
        let host = vec![1.0f32; n];
        assert!(check_cumprod_stages(&x, &x, &x, &host, d1(n)).is_err());
        assert!(check_logcumsumexp_stages(&x, &x, &x, &host, d1(n)).is_err());
    }

    /// `scan_forward::<f32>` は総括判定の第 1 軸の [`scan_lane_host`] と同じ演算順（bit 一致）。
    #[test]
    fn scan_forward_matches_scan_lane_host() {
        for n in [1usize, 2, 3, 5, 8, 9] {
            let b: Vec<f32> = (0..n)
                .map(|i| ((i as f32) * 0.7 + 0.3).sin() * 3.0)
                .collect();
            let a: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.3).cos() * 1.7).collect();
            let (s, _) = scan_forward::<f32>(&b, &a);
            let (h, _) = scan_lane_host(&b, &a, None);
            assert_eq!(
                s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "n={n}"
            );
        }
    }

    /// 子テープで実際に `backward` した二階（seed `k` の行。`x` と上流 `g` への要素）が、第 2 軸の
    /// f32 の再現 `V_str` と一致する（判定の前提。CPU の演算順を写すので値として一致する）。
    /// 上流 `g` は損失の重み `wc`（追跡する葉）で、`Sum`→`Mul` の VJP が `g = wc` を bit で通す。
    fn emulation_case(kind: &str, x0: &[f32], w: &[f32]) {
        let n = x0.len();
        let tape = Tape::new_with_ops(naive_ops());
        let child = Tape::new_with_ops(naive_ops());
        let x = tape.var(&Tensor::new(x0.to_vec(), &[n]).unwrap());
        let wc = tape.var(&Tensor::new(w.to_vec(), &[n]).unwrap());
        let y = match kind {
            "cumprod" => x.cumprod(0).unwrap(),
            _ => x.logcumsumexp(0).unwrap(),
        };
        let yv = y.value().host_slice().into_owned();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape
            .backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("{kind} {x0:?}: 受理されるはず: {e:?}"));
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        let cw = cg.child_var(&wc).unwrap().unwrap();
        let layout = cumulative_layout(&[n], 0).unwrap();
        let x_t = Tensor::new(x0.to_vec(), &[n]).unwrap();
        for k in 0..n {
            let row = child.backward(&gx.narrow(0, k, 1).unwrap().sum(None).unwrap());
            let row = row.unwrap();
            let read = |v: &Var<'_>| match row.get(v).unwrap() {
                Some(t) => t.host_slice().into_owned(),
                None => vec![0.0f32; n],
            };
            let (hx, hw) = (read(&cx), read(&cw));
            let (ex, eg) = if kind == "cumprod" {
                let l: Vec<f32> = (0..n)
                    .map(|j| if j == 0 { 1.0 } else { yv[j - 1] })
                    .collect();
                let a: Vec<f32> = (0..n)
                    .map(|j| if j + 1 < n { x0[j + 1] } else { 0.0 })
                    .collect();
                let (s, h) = scan_forward::<f32>(w, &a);
                cumprod_seed::<f32>(k, 1.0, &l, &s, &h, |cot_y| {
                    let t = Tensor::new(cot_y, &[n]).unwrap();
                    Ok(crate::eval::dense_vec(&cumprod_vjp_along(&x_t, &t, 0)))
                })
                .unwrap()
            } else {
                let c: Vec<f32> = (0..n)
                    .map(|j| {
                        if j + 1 < n {
                            (yv[j] - yv[j + 1]).exp()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let wv: Vec<f32> = (0..n).map(|j| (x0[j] - yv[j]).exp()).collect();
                let (t, h) = scan_forward::<f32>(w, &c);
                lcse_seed::<f32>(k, 1.0, (&wv, &c), &t, &h, |cot_y| {
                    Ok(logcumsumexp_vjp_host(x0, &cot_y, &layout).unwrap())
                })
                .unwrap()
            };
            // 値で一致（NaN は位置の一致）。
            let same = |a: f32, b: f32| a == b || (a.is_nan() && b.is_nan());
            for m in 0..n {
                assert!(
                    same(hx[m], ex[m]),
                    "{kind} {x0:?} k={k}: x への二階[{m}] 子テープ={} 再現={}",
                    hx[m],
                    ex[m]
                );
                assert!(
                    same(hw[m], eg[m]),
                    "{kind} {x0:?} k={k}: g への二階[{m}] 子テープ={} 再現={}",
                    hw[m],
                    eg[m]
                );
            }
        }
    }

    #[test]
    fn emulation_reproduces_child_tape_second_order() {
        for (x0, w) in [
            (
                vec![1.1f32, 0.9, 1.2, 0.8, 1.05],
                vec![0.7f32, -1.3, 0.4, 2.1, -0.6],
            ),
            (vec![2.0, 0.0, 3.0], vec![0.0, 0.0, 1.0]),
            (
                vec![0.0, 2.0, 0.0, 1.0, 0.5, 1.5, 0.3, 2.2, 0.9],
                vec![1.0; 9],
            ),
            (vec![1.3, -0.7], vec![0.9, -0.4]),
            (vec![1.7], vec![-2.5]),
            (vec![1.0, f32::NAN, 2.0, 0.5], vec![1.0, -1.0, 0.5, 2.0]),
            (vec![0.5, 2.0, f32::NAN], vec![1.0, 1.0, 1.0]),
            (vec![f32::NAN], vec![1.5]),
        ] {
            emulation_case("cumprod", &x0, &w);
        }
        for (x0, w) in [
            (
                vec![0.3f32, -0.5, 1.1, 0.2, -0.9],
                vec![0.7f32, -1.3, 0.4, 2.1, -0.6],
            ),
            (vec![0.0, 0.0, 0.0], vec![0.0, -0.8, 0.9]),
            (
                vec![-3.0, 2.0, 0.5, 1.0, -1.0, 4.0, 0.0, 1.5, 2.5],
                vec![1.0; 9],
            ),
            (vec![2.0, 2.0], vec![0.5, 1.0]),
            (vec![0.4], vec![1.5]),
            (vec![0.3, f32::NAN, 1.0], vec![1.0, -0.5, 2.0]),
            (vec![f32::NAN, 0.5], vec![1.0, 1.0]),
        ] {
            emulation_case("logcumsumexp", &x0, &w);
        }
    }
}
