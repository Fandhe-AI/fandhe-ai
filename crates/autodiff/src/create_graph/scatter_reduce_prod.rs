//! `Op::IndexedScatterReduce` の `Prod` mode の `create_graph` 対応
//! （イシュー #3977・親 #3719 系・追跡表 #3717 項目 4。設計
//! `docs/autodiff-higher-order-grad-decision.md` §41.10・§41.11・§41.13、実装記録は §74）。
//!
//! 兄弟モジュール [`super::scatter_reduce`] の `cgrad`（Prod のとき）から呼ばれる。区分つき scan
//! 補助 [`super::scan`]（§64）の上に載り、**除算を使わない**（除算形は 0 を含む寄与列の最初の零より
//! 後ろの混合二階を取りこぼす。§41.10）。総括判定の道具（予算・`entry_check`・`host_ref_check`）は
//! 兄弟 [`super::cumulative_b`] と共有し、値は変えない。
//!
//! **値**: 1 階は `scatter_reduce_vjp_host`（`tensor_core::indexed_update`。(仮数, 指数) の前置積×
//! 後置積）の結果を `push_v3` で載せる。構造は二階微分を流すためだけの `Var` 列。
//!
//! **構造**（出力位置 `q` ごとの寄与列を「区間」とし、全区間を連結した長さ `K` の 1 本の lane に並べる。
//! 区間内の並びは src の走査順→自己寄与〈`include_self`〉で、1 階の lane と同じ）:
//! 1. `v = Gather(Concat([src_m, self_m]), perm)`（寄与値の列。`src_m`・`input_m` は子テープ上の写し）。
//! 2. 排他的 suffix 積: `linear_scan_steps(b, A, cut)`。`b` は区間末尾で 1、`A[a] = v[a+1]`、`cut` は区間
//!    末尾で真（区間をまたぐ係数を断つ）。`S[last] = 1`・`S[a] = v[a+1]·S[a+1]`。
//! 3. 排他的 prefix 積: 区間内だけを逆順にした置換 `rev`（自己逆）で同じ scan を行い `Gather(·, rev)` で
//!    戻す。全体の逆順にしないのは、区間の位置を保って `b`・`cut` を共有するため。
//! 4. `d = Gather(g, q_of_slot) ⊙ (pre ⊙ suf)`、`d_src = Gather(d, src_slot)`、`d_input` は自己寄与の
//!    スロットを `Gather(Concat([d, 0]))` で戻し、未タッチ位置は `where_cond` で `g` を素通しする。
//!
//! 段数は最長区間長の `⌈log2⌉`（区間をまたぐ係数は `cut` で 0。余分な段は 0 を足すだけなので、減らすと
//! ノード数と `0·inf` の危険が減る）。子テープへ積む Op は `Narrow`・`Concat`・`Gather`・`Mul`・`Add`・
//! `Where`・`Reshape`・`Contiguous`・定数葉で、すべて `supports_create_graph() == true`（閉包性）。
//!
//! **総括判定**（§41.12・§64.2 と同じ形。すべて子テープへ構造を積む**前**に行い、拒否は型付き
//! `Err(Backward)` で親の rollback により `child` は空へ戻る）。区間は互いに独立なので判定は区間ごと
//! （窓）で行う。窓の再現が全体 lane の子テープと値で一致するのは、区間をまたぐ項が `cut` により
//! 厳密な `0 × 有限` になるため（有限でなければ第 1 軸が先に拒否する）:
//! - 入口: `±inf` は親の `validate`、上流 `g` の非有限・過大は親の `check_upstream_bound` が拒否済み。
//!   ここで追加に `max|g|·最長区間長 >= 2^125` と、**触れられた位置の寄与（src と自己寄与）の NaN** を
//!   拒否する。NaN の拒否は §41.16 点 8（NaN は伝播）からの逸脱で、`0 × NaN = NaN` が `cut` を越えて
//!   別区間へ漏れる（順伝播では後ろ→前、逆伝播では前→後ろ。Hillis–Steele の段の積が区間をまたぐ）ため
//!   NaN を区間に閉じ込められない。未タッチ位置の NaN は素通しなので影響しない。
//! - 計算量: 予算 [`JUDGE_BUDGET`]（`2^27`）に対し [`judge_cost`] を、区間走査の前に判定する。
//! - 第 1 軸（overflow）: 窓の 2 本の scan・`excl`・`d` を f32 で再計算し、非有限または `2^125` 以上なら
//!   拒否。二階の逆伝播の上界（seed の各要素の絶対値 1）も [`scan_cotangent_bound`] で同じ閾値。
//!   Hillis–Steele の段では隣接スロットの積が、排他的積が有限でも overflow しうる（例 `[1e30, 1e30,
//!   1e-30]`）。この入力は本軸で拒否する契約で、1 階の値は有限のまま変わらない。
//! - ホストの 1 階: 非有限は拒否し、値は f64 の独立参照（`g·Π_{l≠i} v_l`。構造の中間値を共有しない）と
//!   `HOST_COEFF` 規則で比べる。
//! - 第 2 軸（二階の要素ごと）: seed を区間内の 1 スロット（値 1）にしたときの `v` への要素と `g` への
//!   要素を、子テープの f32 演算列の再現 `V_str` と独立な f64 の解析形 `V`・`M` で比べ、
//!   [`entry_check`]（絶対 1e-5 未満または相対の半分以内）で拒否する。underflow で `excl` が 0 に潰れる
//!   入力（例 `[1e-30, 1e-30, 1e30, 1e30, 1e30]`）は本軸で落ちる。`[1e-30, 1e30]` は受理する。
//!
//! GPU: `indexed_scatter_reduce` は CPU のみ実装でホストへフォールバックする（経路区分 `H`）。
//! 実機 parity は未実測（Phase 10 の #3857・#3859）。第 2 軸の再現は CPU の演算順で、GPU の演算順は
//! 保証の対象外。

use fandhe_ai_tensor_core::Tensor;

use super::cumulative_b::{JUDGE_COEFF, entry_check, host_ref_check, judge_budget, over_guard};
use super::push_v3;
use super::scan::{
    Num, Rf, linear_scan_steps, scan_backward, scan_cotangent_bound, scan_cotangent_bound_steps,
    scan_forward, scan_lane_host, scan_lane_host_steps, scan_steps,
};
use super::scatter_reduce::Grads;
use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;

const WHAT: &str = "IndexedScatterReduce(Prod)";

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

/// 区間（出力位置 `q` の寄与列）。`start..start+len` がスロット範囲。
#[derive(Clone, Copy, Debug)]
struct Seg {
    q: usize,
    start: usize,
    len: usize,
}

/// ホストで作る置換・マスク群（全区間を連結した長さ `K` の lane）。
struct Plan {
    /// `src` の要素数 `S`。
    s_len: usize,
    /// スロット総数 `K`。
    k: usize,
    segs: Vec<Seg>,
    max_len: usize,
    /// スロット → `Concat([src_flat, self_vals])` 内の添字。
    perm: Vec<i32>,
    /// スロット → 区間内を逆順にしたスロット（自己逆）。
    rev: Vec<i32>,
    /// スロット → 出力位置 `q`。
    q_of_slot: Vec<i32>,
    /// src の位置 `p` → スロット。
    src_slot: Vec<i32>,
    /// 入力の位置 `q` → 自己寄与のスロット（無ければ `K`＝末尾の 0）。
    in_idx: Vec<i32>,
    /// 自己寄与の値を取る入力の位置（触れられた位置の昇順）。`include_self` のときだけ非空。
    self_idx: Vec<i32>,
    /// 区間末尾のスロット（`b` と `cut`）。
    last: Vec<bool>,
    /// 入力の位置が未タッチか。
    untouched: Vec<bool>,
}

fn to_i32(v: usize, what: &str) -> Result<i32, AutodiffError> {
    i32::try_from(v).map_err(|_| {
        invalid(format!(
            "{WHAT}: {what} {v} が i32 の範囲を超えるため拒否（§41.14 参照）"
        ))
    })
}

/// 位置規則 `pos`（src 要素 `p` の書き込み先）から [`Plan`] を作る。確保の前に要素数を検査する。
fn build_plan(pos: &[usize], n: usize, include_self: bool) -> Result<Plan, AutodiffError> {
    let s_len = pos.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (p, &q) in pos.iter().enumerate() {
        members[q].push(p);
    }
    let touched = members.iter().filter(|m| !m.is_empty()).count();
    let k = if include_self {
        s_len.checked_add(touched)
    } else {
        Some(s_len)
    }
    .ok_or_else(|| invalid(format!("{WHAT}: スロット数がオーバーフローするため拒否")))?;
    // 以降の i32 索引（K を末尾 0 の添字にも使う）がすべて収まることを先に確かめる。
    let k_i32 = to_i32(k, "スロット数")?;
    to_i32(n, "入力要素数")?;
    to_i32(s_len, "src 要素数")?;

    let mut plan = Plan {
        s_len,
        k,
        segs: Vec::new(),
        max_len: 0,
        perm: Vec::with_capacity(k),
        rev: Vec::with_capacity(k),
        q_of_slot: Vec::with_capacity(k),
        src_slot: vec![0; s_len],
        in_idx: vec![k_i32; n],
        self_idx: Vec::new(),
        last: vec![false; k],
        untouched: vec![false; n],
    };
    for (q, mem) in members.iter().enumerate() {
        if mem.is_empty() {
            plan.untouched[q] = true;
            continue;
        }
        let start = plan.perm.len();
        for &p in mem {
            plan.src_slot[p] = to_i32(plan.perm.len(), "スロット")?;
            plan.perm.push(to_i32(p, "src 位置")?);
            plan.q_of_slot.push(to_i32(q, "出力位置")?);
        }
        if include_self {
            plan.in_idx[q] = to_i32(plan.perm.len(), "スロット")?;
            plan.perm.push(to_i32(s_len + plan.self_idx.len(), "perm")?);
            plan.q_of_slot.push(to_i32(q, "出力位置")?);
            plan.self_idx.push(to_i32(q, "出力位置")?);
        }
        let len = plan.perm.len() - start;
        for i in 0..len {
            plan.rev.push(to_i32(start + len - 1 - i, "スロット")?);
        }
        plan.last[start + len - 1] = true;
        plan.max_len = plan.max_len.max(len);
        plan.segs.push(Seg { q, start, len });
    }
    Ok(plan)
}

// ---------------------------------------------------------------------
// 計算量の予算
// ---------------------------------------------------------------------

/// 区間 1 本・1 scan あたりの forward 側の走査量の係数（`m` 要素あたり。`steps = ⌈log2 m⌉`）。
const FWD_PER_STEP: usize = 22;
const FWD_BASE: usize = 44;
/// seed 1 個・1 scan あたりの走査量の係数（`m` 要素あたり）。
const SEED_PER_STEP: usize = 3;
const SEED_BASE: usize = 16;

/// 総括判定の走査量（f32 の再現 1 系統が回す内側ループの反復回数の上界。f64 の参照も同じ回数を回す）。
/// 区間（長さ `m`・`steps = ⌈log2 m⌉`）ごとに、forward 側が 2 scan 分の
/// `2·m·(22·steps + 44)` と窓の組み立て・参照・照合の `64·m`、seed（`m` 個）ごとに
/// 2 scan 分の `2·m·(3·steps + 16)` と参照・判定の `16·m`。区間は互いに独立なので `K²` ではなく
/// 区間ごとの和で数える。構築（置換・マスク）の `8·K` も加える。
fn judge_cost(lens: impl Iterator<Item = usize>, k: usize) -> usize {
    // 連結 lane の判定（`whole_lane_bounds`）は 4 scan 相当を K 要素で回す。
    let mut total = k.saturating_mul(8);
    let mut max_m = 0usize;
    for m in lens {
        max_m = max_m.max(m);
        let steps = scan_steps(m);
        let fwd = m.saturating_mul(
            2usize
                .saturating_mul(steps.saturating_mul(FWD_PER_STEP).saturating_add(FWD_BASE))
                .saturating_add(64),
        );
        let per_seed = m.saturating_mul(
            2usize
                .saturating_mul(
                    steps
                        .saturating_mul(SEED_PER_STEP)
                        .saturating_add(SEED_BASE),
                )
                .saturating_add(16),
        );
        total = total.saturating_add(fwd.saturating_add(m.saturating_mul(per_seed)));
    }
    total = total
        .saturating_add(k.saturating_mul(scan_steps(max_m).saturating_mul(32).saturating_add(64)));
    total
}

// ---------------------------------------------------------------------
// 窓（区間 1 本）の判定
// ---------------------------------------------------------------------

/// 区間の窓。`w` は寄与値、`g` は出力位置の上流勾配、`host` は 1 階のホスト値（スロット順）。
struct Window {
    w: Vec<f32>,
    g: f32,
    host: Vec<f32>,
}

/// 窓の scan の入力 `(b, A)`。`b` は末尾 1・`A[a] = w[a+1]`・`A[m-1] = 0`（`cut` 後の値）。
fn scan_inputs(w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let m = w.len();
    let mut b = vec![0f32; m];
    b[m - 1] = 1.0;
    let a: Vec<f32> = (0..m)
        .map(|i| if i + 1 < m { w[i + 1] } else { 0.0 })
        .collect();
    (b, a)
}

fn reversed(w: &[f32]) -> Vec<f32> {
    w.iter().rev().copied().collect()
}

/// 第 1 軸（モジュール doc）。`g` は出力位置の上流勾配。
fn stage_bounds(w: &[f32], g: f32) -> Result<(), AutodiffError> {
    let m = w.len();
    let (b, a_suf) = scan_inputs(w);
    let (_, a_pre) = scan_inputs(&reversed(w));
    let (suf, mx_s) = scan_lane_host(&b, &a_suf, None);
    let (sp, mx_p) = scan_lane_host(&b, &a_pre, None);
    if over_guard(mx_s) || over_guard(mx_p) {
        return Err(invalid(format!(
            "{WHAT}: scan 構造の段が 2^125 以上または非有限のため拒否（§74 参照）"
        )));
    }
    let pre: Vec<f32> = (0..m).map(|i| sp[m - 1 - i]).collect();
    let g_abs = f64::from(g).abs();
    let mut cot_suf = vec![0f64; m];
    let mut cot_sp = vec![0f64; m];
    for i in 0..m {
        let excl = pre[i] * suf[i];
        let d = g * excl;
        if over_guard(f64::from(excl).abs()) || over_guard(f64::from(d).abs()) {
            return Err(invalid(format!(
                "{WHAT}: 排他的積または出力段が 2^125 以上または非有限のため拒否（§74 参照）"
            )));
        }
        // seed の各要素の絶対値 1 以下の前提: `d = gg ⊙ (pre ⊙ suf)` への cotangent は
        // `pre` に `|g|·|suf|`、`suf` に `|g|·|pre|` が流れる。
        cot_suf[i] = g_abs * f64::from(pre[i]).abs();
        cot_sp[m - 1 - i] = g_abs * f64::from(suf[i]).abs();
    }
    let (_, ca_s, mx_cs) = scan_cotangent_bound(&b, &a_suf, &cot_suf);
    let (_, ca_p, mx_cp) = scan_cotangent_bound(&b, &a_pre, &cot_sp);
    if over_guard(mx_cs) || over_guard(mx_cp) {
        return Err(invalid(format!(
            "{WHAT}: 二階の逆伝播の中間値が 2^125 以上または非有限のため拒否（§74 参照）"
        )));
    }
    for j in 0..m {
        let from_s = if j >= 1 { ca_s[j - 1] } else { 0.0 };
        let jr = m - 1 - j;
        let from_p = if jr >= 1 { ca_p[jr - 1] } else { 0.0 };
        if over_guard(from_s + from_p) {
            return Err(invalid(format!(
                "{WHAT}: 二階の寄与列への cotangent が 2^125 以上または非有限のため拒否（§74 参照）"
            )));
        }
    }
    Ok(())
}

/// f64 の独立参照: 各要素を除いた積（除算なし。prefix 積×suffix 積）。構造の中間値を使わない。
fn excl_ref(w: &[Rf]) -> Vec<Rf> {
    let m = w.len();
    let mut pre = vec![Rf::k(1.0); m + 1];
    for i in 0..m {
        pre[i + 1] = pre[i].mul(w[i]);
    }
    let mut suf = vec![Rf::k(1.0); m + 1];
    for i in (0..m).rev() {
        suf[i] = suf[i + 1].mul(w[i]);
    }
    (0..m).map(|i| pre[i].mul(suf[i + 1])).collect()
}

/// 構造の forward（f32）。`suf`・`pre`・`excl` と各 scan の段履歴。
struct Fwd {
    hs: Vec<(Vec<f32>, Vec<f32>)>,
    hp: Vec<(Vec<f32>, Vec<f32>)>,
    suf: Vec<f32>,
    pre: Vec<f32>,
    excl: Vec<f32>,
}

fn forward32(w: &[f32]) -> Fwd {
    let m = w.len();
    let (b, a_suf) = scan_inputs(w);
    let (_, a_pre) = scan_inputs(&reversed(w));
    let (suf, hs) = scan_forward::<f32>(&b, &a_suf);
    let (sp, hp) = scan_forward::<f32>(&b, &a_pre);
    let pre: Vec<f32> = (0..m).map(|i| sp[m - 1 - i]).collect();
    let excl = (0..m).map(|i| pre[i] * suf[i]).collect();
    Fwd {
        hs,
        hp,
        suf,
        pre,
        excl,
    }
}

/// 子テープの二階（seed = スロット `k` の 1 要素）の f32 再現。`(寄与値 v への要素, g への要素)`。
///
/// 子テープの Op 列の逆順（`backward.rs` は生成の逆順に処理）: `d = gg ⊙ excl` が `excl` へ `seed·gg`、
/// `gg` へ `seed·excl`。`excl = pre ⊙ suf` が `pre` へ `·suf`、`suf` へ `·pre`。`pre = Gather(sp, rev)` は
/// `sp` への scatter（置換なので加算なし）。各 scan の逆伝播 [`scan_backward`] の `A` への cotangent を
/// `A[a] = v[a+1]` の `Narrow` で `v[a+1]` へ戻し（`A[last]` は `cut` で捨てる）、prefix 側は `rev` の
/// `Gather` で戻す。`v` への 2 経路の合算は f32 の 2 項の和（可換）。
fn seed32(f: &Fwd, g: f32, k: usize) -> (Vec<f32>, f32) {
    let m = f.excl.len();
    let cot_excl: Vec<f32> = (0..m)
        .map(|i| if i == k { 1.0 * g } else { 0.0 * g })
        .collect();
    let cot_suf: Vec<f32> = (0..m).map(|i| cot_excl[i] * f.pre[i]).collect();
    let mut cot_sp = vec![0f32; m];
    for i in 0..m {
        cot_sp[m - 1 - i] = cot_excl[i] * f.suf[i];
    }
    let (_, ca_s) = scan_backward(&f.hs, cot_suf);
    let (_, ca_p) = scan_backward(&f.hp, cot_sp);
    let v = (0..m)
        .map(|j| {
            let from_s = if j >= 1 { ca_s[j - 1] } else { 0.0 };
            let jr = m - 1 - j;
            let from_p = if jr >= 1 { ca_p[jr - 1] } else { 0.0 };
            from_p + from_s
        })
        .collect();
    (v, f.excl[k])
}

/// 独立な f64 の解析形の二階（seed = スロット `k`）。`∂d_k/∂g = Π_{l≠k} w_l`、
/// `∂d_k/∂w_j = g·Π_{l∉{k,j}} w_l`（`j≠k`）、`∂d_k/∂w_k = 0`。
fn seed_ref(w: &[Rf], g: Rf, k: usize) -> (Vec<Rf>, Rf) {
    let m = w.len();
    let g_entry = excl_ref(w)[k];
    let others: Vec<Rf> = (0..m).filter(|&l| l != k).map(|l| w[l]).collect();
    let ex = excl_ref(&others);
    let v = (0..m)
        .map(|j| match j.cmp(&k) {
            std::cmp::Ordering::Equal => Rf::k(0.0),
            std::cmp::Ordering::Less => g.mul(ex[j]),
            std::cmp::Ordering::Greater => g.mul(ex[j - 1]),
        })
        .collect();
    (v, g_entry)
}

/// 窓 1 本の判定（第 1 軸→ホストの 1 階→第 2 軸）。
fn judge_window(win: &Window) -> Result<(), AutodiffError> {
    let w = &win.w;
    stage_bounds(w, win.g)?;
    if win.host.iter().any(|h| !h.is_finite()) {
        return Err(invalid(format!(
            "{WHAT}: 1 階のホスト値に非有限がある（入力の NaN は事前に拒否済みのため範囲外の値）ため拒否"
        )));
    }
    let wr: Vec<Rf> = w.iter().map(|&v| Rf::k(f64::from(v))).collect();
    let gr = Rf::k(f64::from(win.g));
    let first: Vec<Rf> = excl_ref(&wr).into_iter().map(|e| gr.mul(e)).collect();
    host_ref_check(&first, &win.host, WHAT)?;
    let f = forward32(w);
    for k in 0..w.len() {
        let (v_str, g_str) = seed32(&f, win.g, k);
        let (v_ref, g_ref) = seed_ref(&wr, gr, k);
        for (r, s) in v_ref.iter().zip(&v_str) {
            entry_check(*r, *s, JUDGE_COEFF, WHAT)?;
        }
        entry_check(g_ref, g_str, JUDGE_COEFF, WHAT)?;
    }
    Ok(())
}

/// 全区間を連結した lane（子テープが実際に積む演算列）での総括判定（第 1 軸の連結版）。
///
/// 区間ごとの窓は区間をまたぐ項を含まないが、連結 scan の逆伝播では境界スロットの cotangent
/// （例 `1e30×1e30 = inf`）に `cut` の 0 を掛ける `inf × 0 = NaN` が次区間の非境界スロットへ流れうる
/// （Codex P1。入力例 `include_self=false`・`index=[0,0,1,1,1]`・`src=[1e30,1,1e-30,1,1e30]`）。
/// そのため forward（suffix・prefix の 2 scan）と二階の逆伝播の上界を連結 lane 上で再計算し、
/// 2^125 以上または非有限なら構築前に型付きエラーで拒否する（§74）。
fn whole_lane_bounds(
    plan: &Plan,
    x: &[f32],
    s: &[f32],
    g_host: &[f32],
) -> Result<(), AutodiffError> {
    let k = plan.k;
    // 子テープ（linear_scan_steps）が積む段数と一致させる（全体長 K の段数ではない）。
    let steps = scan_steps(plan.max_len);
    let v: Vec<f32> = plan
        .perm
        .iter()
        .map(|&a| {
            let a = a as usize;
            if a < plan.s_len {
                s[a]
            } else {
                x[plan.self_idx[a - plan.s_len] as usize]
            }
        })
        .collect();
    let v_rev: Vec<f32> = plan.rev.iter().map(|&r| v[r as usize]).collect();
    let b: Vec<f32> = plan
        .last
        .iter()
        .map(|&l| if l { 1.0 } else { 0.0 })
        .collect();
    let shifted = |w: &[f32]| -> Vec<f32> {
        (0..k)
            .map(|i| if i + 1 < k { w[i + 1] } else { 0.0 })
            .collect()
    };
    let a_suf = shifted(&v);
    let a_pre = shifted(&v_rev);
    let (suf, mx_s) = scan_lane_host_steps(&b, &a_suf, Some(&plan.last), steps);
    let (sp, mx_p) = scan_lane_host_steps(&b, &a_pre, Some(&plan.last), steps);
    if over_guard(mx_s) || over_guard(mx_p) {
        return Err(invalid(format!(
            "{WHAT}: 連結 scan の段が 2^125 以上または非有限のため拒否（区間をまたぐ積。§74 参照）"
        )));
    }
    let cut_a = |a: &[f32]| -> Vec<f32> {
        a.iter()
            .zip(&plan.last)
            .map(|(&x, &c)| if c { 0.0 } else { x })
            .collect()
    };
    let (a_suf_c, a_pre_c) = (cut_a(&a_suf), cut_a(&a_pre));
    let mut cot_suf = vec![0f64; k];
    let mut cot_sp = vec![0f64; k];
    let mut pre = vec![0f32; k];
    for i in 0..k {
        pre[i] = sp[plan.rev[i] as usize];
    }
    for i in 0..k {
        let gg = f64::from(g_host[plan.q_of_slot[i] as usize]).abs();
        cot_suf[i] = gg * f64::from(pre[i]).abs();
        cot_sp[plan.rev[i] as usize] = gg * f64::from(suf[i]).abs();
    }
    let (_, ca_s, mx_cs) = scan_cotangent_bound_steps(&b, &a_suf_c, &cot_suf, steps);
    let (_, ca_p, mx_cp) = scan_cotangent_bound_steps(&b, &a_pre_c, &cot_sp, steps);
    if over_guard(mx_cs) || over_guard(mx_cp) {
        return Err(invalid(format!(
            "{WHAT}: 連結 scan の二階の逆伝播の中間値が 2^125 以上または非有限のため拒否（区間をまたぐ積。§74 参照）"
        )));
    }
    for j in 0..k {
        let from_s = if j >= 1 { ca_s[j - 1] } else { 0.0 };
        let jr = plan.rev[j] as usize;
        let from_p = if jr >= 1 { ca_p[jr - 1] } else { 0.0 };
        if over_guard(from_s + from_p) {
            return Err(invalid(format!(
                "{WHAT}: 連結 scan の寄与列への cotangent が 2^125 以上または非有限のため拒否（§74 参照）"
            )));
        }
    }
    Ok(())
}

/// 全区間の総括判定。
fn judge(
    plan: &Plan,
    x: &[f32],
    s: &[f32],
    g_host: &[f32],
    hosts: (&[f32], &[f32]),
) -> Result<(), AutodiffError> {
    let (d_in, d_src) = hosts;
    judge_budget(judge_cost(plan.segs.iter().map(|g| g.len), plan.k), WHAT)?;
    whole_lane_bounds(plan, x, s, g_host)?;
    for seg in &plan.segs {
        let mut w = Vec::with_capacity(seg.len);
        let mut host = Vec::with_capacity(seg.len);
        for slot in seg.start..seg.start + seg.len {
            let a = plan.perm[slot] as usize;
            if a < plan.s_len {
                w.push(s[a]);
                host.push(d_src[a]);
            } else {
                w.push(x[seg.q]);
                host.push(d_in[seg.q]);
            }
        }
        judge_window(&Window {
            w,
            g: g_host[seg.q],
            host,
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// 構造の組み立て
// ---------------------------------------------------------------------

/// 呼び出し元（[`super::scatter_reduce::cgrad`]）が渡す入力。
pub(super) struct ProdArgs<'a, 'c> {
    pub(super) child: &'c Tape,
    pub(super) input_shape: &'a [usize],
    pub(super) src_shape: &'a [usize],
    pub(super) include_self: bool,
    /// 親テープの forward 値（入力・src）と上流の値。
    pub(super) x: &'a [f32],
    pub(super) s: &'a [f32],
    pub(super) g_host: &'a [f32],
    /// 各 src 要素の書き込み先（行優先線形添字）。
    pub(super) pos: &'a [usize],
    /// 1 階のホスト値 `(d_input, d_src)`。
    pub(super) d_in_host: Vec<f32>,
    pub(super) d_src_host: Vec<f32>,
    /// 子テープ上の写しと上流。
    pub(super) input_m: &'a Var<'c>,
    pub(super) src_m: &'a Var<'c>,
    pub(super) g: &'a Var<'c>,
    pub(super) need_input: bool,
    pub(super) need_src: bool,
}

fn idx_tensor(v: &[i32]) -> Result<Tensor<i32>, AutodiffError> {
    Ok(Tensor::new(v.to_vec(), &[v.len()])?)
}

/// 排他的 suffix 積の scan（`b`・`cut` は区間末尾）。`vv` は長さ `K` の寄与列。
fn suffix_scan<'c>(
    child: &'c Tape,
    vv: &Var<'c>,
    plan: &Plan,
    b: &Var<'c>,
    cut: &Tensor<bool>,
) -> Result<Var<'c>, AutodiffError> {
    let k = plan.k;
    let zero1 = child.var_no_grad(&Tensor::zeros(&[1])?);
    let a = if k == 1 {
        zero1
    } else {
        Var::cat(&[vv.narrow(0, 1, k - 1)?, zero1], 0)?
    };
    linear_scan_steps(child, b, &a, 0, Some(cut), scan_steps(plan.max_len))
}

/// `Prod` の VJP を子テープへ記録する（判定を通ったときだけ構造を積む）。
pub(super) fn cgrad<'c>(args: ProdArgs<'_, 'c>) -> Result<Grads<'c>, AutodiffError> {
    let ProdArgs {
        child,
        input_shape,
        src_shape,
        include_self,
        x,
        s,
        g_host,
        pos,
        d_in_host,
        d_src_host,
        input_m,
        src_m,
        g,
        need_input,
        need_src,
    } = args;
    let n = x.len();
    // 触れられた位置の寄与の NaN は拒否する（モジュール doc: `0 × NaN` が `cut` を越えて漏れる）。
    let mut touched = vec![false; n];
    for (&q, &v) in pos.iter().zip(s) {
        touched[q] = true;
        if v.is_nan() {
            return Err(invalid(format!(
                "{WHAT}: src に NaN がある（区間をまたぐ NaN の漏れを防げないため拒否。§74 参照）"
            )));
        }
    }
    if include_self && (0..n).any(|q| touched[q] && x[q].is_nan()) {
        return Err(invalid(format!(
            "{WHAT}: 触れられた位置の入力に NaN がある（include_self の自己寄与。§74 参照）"
        )));
    }
    let plan = build_plan(pos, n, include_self)?;
    let max_g = g_host.iter().fold(0f64, |a, v| a.max(f64::from(*v).abs()));
    if over_guard(max_g * plan.max_len.max(1) as f64) {
        return Err(invalid(format!(
            "{WHAT}: 上流勾配が大きすぎ中間値が f32 範囲を超えうるため拒否（§74 参照）"
        )));
    }
    judge(&plan, x, s, g_host, (&d_in_host, &d_src_host))?;

    // 構造。
    let k = plan.k;
    let s_len = plan.s_len;
    let src_flat = src_m.contiguous()?.reshape(&[s_len])?;
    let all = if include_self {
        let x_flat = input_m.contiguous()?.reshape(&[n])?;
        let self_vals = x_flat.gather(0, &idx_tensor(&plan.self_idx)?)?;
        Var::cat(&[src_flat, self_vals], 0)?
    } else {
        src_flat
    };
    let vv = all.gather(0, &idx_tensor(&plan.perm)?)?;
    let b_host: Vec<f32> = plan
        .last
        .iter()
        .map(|&l| if l { 1.0 } else { 0.0 })
        .collect();
    let b = child.var_no_grad(&Tensor::new(b_host, &[k])?);
    let cut = Tensor::new(plan.last.clone(), &[k])?;
    let rev = idx_tensor(&plan.rev)?;
    let suf = suffix_scan(child, &vv, &plan, &b, &cut)?;
    let v_rev = vv.gather(0, &rev)?;
    let sp = suffix_scan(child, &v_rev, &plan, &b, &cut)?;
    let pre = sp.gather(0, &rev)?;
    let excl = pre.mul(&suf)?;
    let g_flat = g.contiguous()?.reshape(&[n])?;
    let gg = g_flat.gather(0, &idx_tensor(&plan.q_of_slot)?)?;
    let d = gg.mul(&excl)?;

    let d_src = if need_src {
        let composed = d
            .gather(0, &idx_tensor(&plan.src_slot)?)?
            .reshape(src_shape)?;
        Some(push_v3(
            child,
            &composed,
            Tensor::new(d_src_host, src_shape)?,
        )?)
    } else {
        None
    };
    let d_input = if need_input {
        let zero1 = child.var_no_grad(&Tensor::zeros(&[1])?);
        let dz = Var::cat(&[d, zero1], 0)?;
        let gin = dz
            .gather(0, &idx_tensor(&plan.in_idx)?)?
            .reshape(input_shape)?;
        let mask = Tensor::new(plan.untouched.clone(), input_shape)?;
        let composed = Var::where_cond(&mask, g, &gin)?;
        Some(push_v3(
            child,
            &composed,
            Tensor::new(d_in_host, input_shape)?,
        )?)
    } else {
        None
    };
    Ok(Grads { d_input, d_src })
}

#[cfg(test)]
mod tests {
    use super::super::cumulative_b::JUDGE_BUDGET;
    use super::*;
    use crate::default_ops::naive_ops;
    use fandhe_ai_tensor_core::ScatterReduceMode;
    use fandhe_ai_tensor_core::indexed_update::{scatter_reduce_layout, scatter_reduce_vjp_host};

    fn same(a: f32, b: f32) -> bool {
        a == b
    }

    /// 解析形の参照が、定義どおりの総当たり（`Π_{l≠i}` を直接掛ける）と f64 の桁で一致する。
    #[test]
    fn prod_reference_matches_analytic_hessian() {
        let w = [1.5f32, -0.7, 2.25, 0.0, 3.0];
        let g = 1.75f32;
        let wr: Vec<Rf> = w.iter().map(|&v| Rf::k(f64::from(v))).collect();
        let gr = Rf::k(f64::from(g));
        let m = w.len();
        let brute = |skip: &[usize]| -> f64 {
            (0..m)
                .filter(|l| !skip.contains(l))
                .map(|l| f64::from(w[l]))
                .product()
        };
        for k in 0..m {
            let (v, ge) = seed_ref(&wr, gr, k);
            assert!((ge.v - brute(&[k])).abs() <= 1e-12 * brute(&[k]).abs().max(1.0));
            for (j, vj) in v.iter().enumerate() {
                let want = if j == k {
                    0.0
                } else {
                    f64::from(g) * brute(&[k, j])
                };
                assert!(
                    (vj.v - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "k={k} j={j} {} vs {want}",
                    vj.v
                );
            }
        }
        // 1 階の参照: d_i = g·Π_{l≠i} w_l。
        for (i, e) in excl_ref(&wr).iter().enumerate() {
            assert!((e.v - brute(&[i])).abs() <= 1e-12 * brute(&[i]).abs().max(1.0));
        }
    }

    /// 区間の窓の構造 forward（f32 再現）が、ホストの前置積×後置積と REQ-2 で一致する。
    #[test]
    fn forward32_matches_exclusive_products() {
        for w in [
            vec![2.0f32],
            vec![2.0, 3.0],
            vec![2.0, 0.0, 3.0],
            vec![1.5, -0.5, 4.0, 0.25, 2.0],
        ] {
            let f = forward32(&w);
            for i in 0..w.len() {
                let want: f64 = (0..w.len())
                    .filter(|&l| l != i)
                    .map(|l| f64::from(w[l]))
                    .product();
                assert!(
                    fandhe_ai_tensor_core::tolerance::composite_close_f32(f.excl[i], want as f32),
                    "{w:?} i={i} {} vs {want}",
                    f.excl[i]
                );
            }
        }
    }

    /// 置換・逆置換の整合: スロットの `perm` が `scatter_reduce_vjp_host` の lane 並び
    /// （src は走査順、自己寄与は末尾）と一致し、`rev` が区間内の自己逆置換になる。
    #[test]
    fn plan_matches_lane_order_and_rev_is_involution() {
        let layout = scatter_reduce_layout(&[4], &[5], &[5], 0).unwrap();
        let idx = [2i32, 0, 2, 2, 0];
        let pos = layout.positions(&idx).unwrap();
        let plan = build_plan(&pos, 4, true).unwrap();
        assert_eq!(plan.k, 5 + 2);
        // q=0: src 1,4 → 自己; q=2: src 0,2,3 → 自己。
        assert_eq!(plan.perm, vec![1, 4, 5, 0, 2, 3, 6]);
        assert_eq!(plan.q_of_slot, vec![0, 0, 0, 2, 2, 2, 2]);
        assert_eq!(plan.rev, vec![2, 1, 0, 6, 5, 4, 3]);
        for (i, &r) in plan.rev.iter().enumerate() {
            assert_eq!(plan.rev[r as usize] as usize, i);
        }
        assert_eq!(
            plan.last,
            vec![false, false, true, false, false, false, true]
        );
        assert_eq!(plan.in_idx, vec![2, 7, 6, 7]);
        assert_eq!(plan.untouched, vec![false, true, false, true]);
        // VJP の値を lane 並びのスロット対応で突き合わせる（slot → 値の取り方が 1 階と一致）。
        let x = [1.5f32, 2.0, 0.5, 4.0];
        let s = [2.0f32, 3.0, -1.0, 0.5, 4.0];
        let g = [1.0f32, 1.0, 2.0, 1.0];
        let (d_in, d_src) =
            scatter_reduce_vjp_host(&x, &idx, &s, &g, &layout, ScatterReduceMode::Prod, true)
                .unwrap();
        // q=0 の寄与は src[1]=3, src[4]=4, x[0]=1.5。src[1] を除く積 = 4·1.5。
        assert!((d_src[1] - 6.0).abs() < 1e-6);
        assert!((d_in[0] - 12.0).abs() < 1e-6);
    }

    /// 予算の境界: 1 区間なら長さ L まで受理し L+1 は拒否する。区間が多いと区間数に比例して効く。
    #[test]
    fn judge_cost_budget_boundary() {
        let mut l = 1usize;
        while judge_cost(std::iter::once(l + 1), l + 1) <= JUDGE_BUDGET {
            l += 1;
        }
        assert!(judge_cost(std::iter::once(l), l) <= JUDGE_BUDGET);
        assert!(judge_cost(std::iter::once(l + 1), l + 1) > JUDGE_BUDGET);
        assert!(judge_budget(judge_cost(std::iter::once(l), l), "t").is_ok());
        assert!(judge_budget(judge_cost(std::iter::once(l + 1), l + 1), "t").is_err());
        // 短い区間でも数が多ければ超える。
        assert!(judge_budget(judge_cost(std::iter::repeat_n(8, 1 << 20), 8 << 20), "t").is_err());
        assert!(judge_budget(judge_cost(std::iter::repeat_n(8, 64), 512), "t").is_ok());
        assert!(judge_budget(judge_cost(std::iter::once(usize::MAX), usize::MAX), "t").is_err());
        // 区間の独立性: 長さ 8 の区間 N 本の費用は 1 本の N 倍（K² にならない）。
        let one = judge_cost(std::iter::once(8), 0);
        assert_eq!(judge_cost(std::iter::repeat_n(8, 100), 0), one * 100);
    }

    fn window(w: &[f32], g: f32) -> Result<(), AutodiffError> {
        let wr: Vec<Rf> = w.iter().map(|&v| Rf::k(f64::from(v))).collect();
        let gr = Rf::k(f64::from(g));
        let host: Vec<f32> = excl_ref(&wr).iter().map(|e| (gr.v * e.v) as f32).collect();
        judge_window(&Window {
            w: w.to_vec(),
            g,
            host,
        })
    }

    #[test]
    fn judge_accepts_and_rejects_reviewed_vectors() {
        // 受理: 零を含む・符号混在・通常値・桁の離れた 2 要素。
        assert!(window(&[2.0, 0.0, 3.0], 1.0).is_ok());
        assert!(window(&[0.0, 0.0, 3.0], 2.0).is_ok());
        assert!(window(&[1.5, -0.5, 4.0, 0.25], 1.0).is_ok());
        assert!(window(&[1e-30, 1e30], 1.0).is_ok());
        assert!(window(&[3.0], 5.0).is_ok());
        // 第 1 軸: 隣接スロットの積が overflow する（排他的積は有限）。
        assert!(window(&[1e30, 1e30, 1e-30], 1.0).is_err());
        // `[1e30, 1e30]` は各排他的積が 1e30 で有限・2^125 未満のため受理する。
        assert!(window(&[1e30, 1e30], 1.0).is_ok());
        assert!(window(&[1e30, 1e30, 1e30], 1.0).is_err());
        // 第 2 軸: underflow で excl が 0 に潰れ、二階の要素が参照から外れる。
        assert!(window(&[1e-30, 1e-30, 1e30, 1e30, 1e30], 1.0).is_err());
    }

    #[test]
    fn host_ref_rule_rejects_wrong_first_order() {
        let w = [2.0f32, 3.0, 4.0];
        let wr: Vec<Rf> = w.iter().map(|&v| Rf::k(f64::from(v))).collect();
        let good: Vec<f32> = excl_ref(&wr).iter().map(|e| e.v as f32).collect();
        assert!(host_ref_check(&excl_ref(&wr), &good, WHAT).is_ok());
        let mut bad = good.clone();
        bad[1] += 1.0;
        assert!(host_ref_check(&excl_ref(&wr), &bad, WHAT).is_err());
    }

    /// 子テープで実際に `backward` した二階が、窓の f32 再現 `V_str` と値で一致する（判定の前提）。
    /// 入力の `x`・`src` と損失の重み `wc`（上流 `g`）を追跡する葉にする。
    #[allow(clippy::too_many_arguments)]
    fn emulation_case(
        x0: &[f32],
        x_shape: &[usize],
        idx0: &[i32],
        idx_shape: &[usize],
        s0: &[f32],
        dim: usize,
        include_self: bool,
        w: &[f32],
    ) {
        let tape = Tape::new_with_ops(naive_ops());
        let child = Tape::new_with_ops(naive_ops());
        let x = tape.var(&Tensor::new(x0.to_vec(), x_shape).unwrap());
        let src = tape.var(&Tensor::new(s0.to_vec(), idx_shape).unwrap());
        let wc = tape.var(&Tensor::new(w.to_vec(), x_shape).unwrap());
        let idx = Tensor::<i32>::new(idx0.to_vec(), idx_shape).unwrap();
        let y = x
            .scatter_reduce(dim, &idx, &src, ScatterReduceMode::Prod, include_self)
            .unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape
            .backward_create_graph(&loss, &child)
            .unwrap_or_else(|e| panic!("受理されるはず: {e:?}"));
        let n = x0.len();
        let s_len = s0.len();
        let gx = cg.grad(&x).unwrap().unwrap().reshape(&[n]).unwrap();
        let gs = cg.grad(&src).unwrap().unwrap().reshape(&[s_len]).unwrap();
        let (cx, cs, cw) = (
            cg.child_var(&x).unwrap().unwrap(),
            cg.child_var(&src).unwrap().unwrap(),
            cg.child_var(&wc).unwrap().unwrap(),
        );
        let layout = scatter_reduce_layout(x_shape, idx_shape, idx_shape, dim).unwrap();
        let pos = layout.positions(idx0).unwrap();
        let plan = build_plan(&pos, n, include_self).unwrap();
        for seg in &plan.segs {
            // スロット → (種別, 添字)。
            let slots: Vec<(bool, usize)> = (seg.start..seg.start + seg.len)
                .map(|sl| {
                    let a = plan.perm[sl] as usize;
                    if a < s_len { (true, a) } else { (false, seg.q) }
                })
                .collect();
            let wv: Vec<f32> = slots
                .iter()
                .map(|&(is_src, i)| if is_src { s0[i] } else { x0[i] })
                .collect();
            let gq = w[seg.q];
            let f = forward32(&wv);
            for k in 0..seg.len {
                let (is_src, i) = slots[k];
                let seed = if is_src {
                    gs.narrow(0, i, 1).unwrap().sum(None).unwrap()
                } else {
                    gx.narrow(0, i, 1).unwrap().sum(None).unwrap()
                };
                let row = child.backward(&seed).unwrap();
                let read = |v: &Var<'_>, len: usize| match row.get(v).unwrap() {
                    Some(t) => t.contiguous().host_slice().into_owned(),
                    None => vec![0.0f32; len],
                };
                let (hx, hs, hw) = (read(&cx, n), read(&cs, s_len), read(&cw, n));
                let (ev, eg) = seed32(&f, gq, k);
                for (j, &(js, ji)) in slots.iter().enumerate() {
                    let got = if js { hs[ji] } else { hx[ji] };
                    assert!(
                        same(got, ev[j]),
                        "seg q={} k={k} j={j}: v への二階 子テープ={got} 再現={}",
                        seg.q,
                        ev[j]
                    );
                }
                assert!(
                    same(hw[seg.q], eg),
                    "seg q={} k={k}: g への二階 子テープ={} 再現={eg}",
                    seg.q,
                    hw[seg.q]
                );
            }
        }
    }

    #[test]
    fn emulation_reproduces_child_tape_second_order() {
        // 複数区間・重複 index・単一スロット区間・零を含む区間（1 次元）。
        emulation_case(
            &[1.5, 2.0, 0.5, 3.0],
            &[4],
            &[2, 0, 2, 2, 3],
            &[5],
            &[2.0, 3.0, -1.5, 0.5, 4.0],
            0,
            false,
            &[0.7, 1.1, -1.3, 0.9],
        );
        emulation_case(
            &[1.5, 2.0, 0.5, 3.0],
            &[4],
            &[2, 0, 2, 2, 3],
            &[5],
            &[2.0, 0.0, -1.5, 0.5, 4.0],
            0,
            true,
            &[0.7, 1.1, -1.3, 0.9],
        );
        // 自己寄与に零。
        emulation_case(
            &[0.0, 2.0, 0.5],
            &[3],
            &[0, 0, 2],
            &[3],
            &[2.0, 3.0, 4.0],
            0,
            true,
            &[1.0, 2.0, 3.0],
        );
        // 非末尾 dim（dim=0, shape [3,2]）・index の非 dim 軸が input より小さい（[2,1]）。
        emulation_case(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &[1, 1],
            &[2, 1],
            &[2.0, 3.0],
            0,
            true,
            &[1.0, 0.5, -1.0, 2.0, 1.5, 0.25],
        );
        // 末尾 dim（rank 2）・複数行。
        emulation_case(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &[0, 0, 2, 1, 1, 1],
            &[2, 3],
            &[1.5, -2.0, 0.5, 3.0, 0.5, 2.0],
            1,
            true,
            &[1.0, 0.5, -1.0, 2.0, 1.5, 0.25],
        );
    }

    /// 複数区間で、区間の外の二階が 0 であること（NaN の無い入力で区間の漏れが無い）を子テープで固定する。
    #[test]
    fn segments_do_not_leak_into_each_other() {
        let tape = Tape::new_with_ops(naive_ops());
        let child = Tape::new_with_ops(naive_ops());
        let x = tape.var(&Tensor::new(vec![1.5f32, 2.0, 0.5], &[3]).unwrap());
        let src = tape.var(&Tensor::new(vec![2.0f32, 3.0, 4.0, 5.0], &[4]).unwrap());
        let wc = tape.var(&Tensor::new(vec![1.0f32, 1.0, 1.0], &[3]).unwrap());
        let idx = Tensor::<i32>::new(vec![0, 0, 2, 2], &[4]).unwrap();
        let y = x
            .scatter_reduce(0, &idx, &src, ScatterReduceMode::Prod, true)
            .unwrap();
        let loss = y.mul(&wc).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gs = cg.grad(&src).unwrap().unwrap();
        let cs = cg.child_var(&src).unwrap().unwrap();
        // src[0]（q=0 の区間）への seed は src[2..4]（q=2 の区間）に二階を作らない。
        let seed = gs.narrow(0, 0, 1).unwrap().sum(None).unwrap();
        let row = child.backward(&seed).unwrap();
        let hs = row.get(&cs).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(&hs[2..], &[0.0, 0.0]);
    }

    /// Codex P1 回帰: 区間ごとの窓は通るが連結 scan の逆伝播が区間をまたいで `inf × 0 = NaN` を作る入力は、
    /// 構築前に型付きエラーで拒否される（NaN の二階を返さない）。
    #[test]
    fn concatenated_scan_overflow_across_segments_is_rejected() {
        let tape = Tape::new_with_ops(naive_ops());
        let child = Tape::new_with_ops(naive_ops());
        let x = tape.var(&Tensor::new(vec![1.0f32, 1.0], &[2]).unwrap());
        let src = tape.var(&Tensor::new(vec![1e30f32, 1.0, 1e-30, 1.0, 1e30], &[5]).unwrap());
        let idx = Tensor::<i32>::new(vec![0, 0, 1, 1, 1], &[5]).unwrap();
        let y = x
            .scatter_reduce(0, &idx, &src, ScatterReduceMode::Prod, false)
            .unwrap();
        let loss = y.sum(None).unwrap();
        match tape.backward_create_graph(&loss, &child) {
            Err(AutodiffError::Backward(_)) => {}
            Ok(_) => panic!("連結 scan の区間またぎ overflow は拒否されるはず"),
            Err(e) => panic!("想定外のエラー: {e:?}"),
        }
    }

    /// 回帰: 複数区間で全体長 K の段数 > 最長区間長の段数のとき、ホスト側の総括判定は子テープと同じ
    /// 段数（`scan_steps(max_len)`）で走査する。余分な段が区間をまたいで作る overflow で、
    /// 安全な入力を誤って拒否しない。
    #[test]
    fn multi_segment_safe_input_with_k_above_max_len_is_accepted() {
        let tape = Tape::new_with_ops(naive_ops());
        let child = Tape::new_with_ops(naive_ops());
        let x = tape.var(&Tensor::new(vec![1.0f32; 4], &[4]).unwrap());
        // 4 区間 × 2 要素（K = 8 → 3 段、最長区間 2 → 1 段）。
        let src = tape.var(
            &Tensor::new(
                vec![1e19f32, 1e19, 1e19, 1e19, 1e19, 1e19, 1e19, 1e19],
                &[8],
            )
            .unwrap(),
        );
        let idx = Tensor::<i32>::new(vec![0, 0, 1, 1, 2, 2, 3, 3], &[8]).unwrap();
        let y = x
            .scatter_reduce(0, &idx, &src, ScatterReduceMode::Prod, false)
            .unwrap();
        let loss = y.sum(None).unwrap();
        tape.backward_create_graph(&loss, &child)
            .expect("安全な複数区間の Prod グラフは拒否されない");
    }
    // -----------------------------------------------------------------
    // 総括判定の走査時間の実測（イシュー #4085・決定記録 §83.1）。判定・定数は変えない
    // -----------------------------------------------------------------

    /// 決定的な LCG（[0, 1)）。計測の入力生成用。
    struct TimingLcg(u64);
    impl TimingLcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f64) / f64::from(1u32 << 31)
        }
    }

    /// 単調な条件 `ok(v)`（`v` が小さいほど真）が成り立つ最大の `v >= 1`（倍々＋二分探索）。
    fn timing_max_by(ok: impl Fn(usize) -> bool) -> usize {
        let (mut lo, mut hi) = (1usize, 2usize);
        assert!(ok(lo));
        while ok(hi) {
            lo = hi;
            hi *= 2;
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// ウォームアップ 1 回の後の 5 回計測（1 回が 6 s 超なら 3 回）の中央値と各回の値（ns）。
    fn timing_median(mut f: impl FnMut()) -> (f64, Vec<f64>) {
        let mut once = || {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        };
        let first = once();
        let reps = if first > 6e9 { 3 } else { 5 };
        let mut xs: Vec<f64> = (0..reps).map(|_| once()).collect();
        let all = xs.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (xs[xs.len() / 2], all)
    }

    /// scatter_reduce Prod の総括判定 `judge`（`whole_lane_bounds` と区間ごとの `judge_window` を含む）を、
    /// 走査量が `JUDGE_BUDGET` ぎりぎりの入力で計測する。形は (a) 1 区間の最大長、(b) 長さ 8 の区間を
    /// 上限まで並べた場合。`include_self = false`・寄与値は 1 近傍で受理される入力にし、`Ok`（全区間を
    /// 走査した）を assert する。実行: `cargo test -p fandhe-ai-autodiff --release --lib -- --ignored
    /// --nocapture --test-threads=1 judge_budget_timing`。出力は stderr の `JUDGETIME` 行のみ。
    #[test]
    #[ignore = "手動実行専用の計測（stderr 出力のみ。#4085・§83.1）"]
    fn judge_budget_timing_scatter_prod() {
        let one_seg = timing_max_by(|l| judge_cost(std::iter::once(l), l) <= JUDGE_BUDGET);
        let many8 = timing_max_by(|c| judge_cost(std::iter::repeat_n(8, c), 8 * c) <= JUDGE_BUDGET);
        let many64 =
            timing_max_by(|c| judge_cost(std::iter::repeat_n(64, c), 64 * c) <= JUDGE_BUDGET);
        for (segs, m) in [(1usize, one_seg), (many64, 64), (many8, 8)] {
            let k = segs * m;
            let cost = judge_cost(std::iter::repeat_n(m, segs), k);
            assert!(cost <= JUDGE_BUDGET);
            assert!(
                judge_cost(std::iter::repeat_n(m, segs + 1), k + m) > JUDGE_BUDGET || segs == 1
            );
            if segs == 1 {
                assert!(judge_cost(std::iter::once(m + 1), m + 1) > JUDGE_BUDGET);
            }
            let mut rng = TimingLcg(0x4085 + k as u64);
            let s: Vec<f32> = (0..k).map(|_| (0.99 + 0.02 * rng.next()) as f32).collect();
            let x = vec![1.0f32; segs];
            let g: Vec<f32> = (0..segs).map(|_| (0.5 + rng.next()) as f32).collect();
            let idx: Vec<i32> = (0..k).map(|p| (p / m) as i32).collect();
            let layout = scatter_reduce_layout(&[segs], &[k], &[k], 0).unwrap();
            let pos = layout.positions(&idx).unwrap();
            let plan = build_plan(&pos, segs, false).unwrap();
            let (d_in, d_src) =
                scatter_reduce_vjp_host(&x, &idx, &s, &g, &layout, ScatterReduceMode::Prod, false)
                    .unwrap();
            let run = || judge(&plan, &x, &s, &g, (&d_in, &d_src));
            // 全区間を走査した（途中で拒否して短絡していない）ことの担保。
            let result = run();
            assert!(result.is_ok(), "segs={segs} m={m}: {result:?}");
            let (med, all) = timing_median(|| {
                let _ = std::hint::black_box(run());
            });
            let entries = segs * 2 * m * m;
            eprintln!(
                "JUDGETIME kind=ScatterReduceProd segs={segs} m={m} k={k} cost={cost} cost/2^27={:.4} entries={entries} result=accepted(full-scan) median_ms={:.1} runs_ms={:?} ns_per_iter={:.3} ns_per_entry={:.3}",
                cost as f64 / JUDGE_BUDGET as f64,
                med / 1e6,
                all.iter()
                    .map(|v| (v / 1e5).round() / 10.0)
                    .collect::<Vec<_>>(),
                med / cost as f64,
                med / entries as f64,
            );
        }
    }
}
