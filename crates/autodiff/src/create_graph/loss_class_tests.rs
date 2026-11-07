//! `loss_class.rs`（損失のクラス軸系の `create_graph`）の相殺の総括判定が「過剰に拒否する入力」を
//! 族ごと（Cosine・CE〈既定 CE と CEWO〉・Triplet）に洗い出す計測と、後続（#4090〜#4096）が受理に
//! 変える入力の回帰テスト（イシュー #4089・親 #4071。設計記録は
//! `docs/autodiff-higher-order-grad-decision.md` §82）。
//!
//! 判定そのもの（`struct_check`・`*_cancel_check`・`pair_info`）・本番経路・tolerance 定数は変えない。
//! 本モジュールは `loss_class.rs` の `#[cfg(test)]` の記録フック（[`super::probe`]）で「どの検査で拒否したか」を
//! 読み、拒否された入力でも本番と同じ演算順の構造を子テープに手で組んで実際の f32 の二階を測り、独立な
//! f64 の閉じた式と REQ-2 で突き合わせる（判定は迂回しない。手組みが本番と一致することは
//! `*_hand_built_structure_matches_public_path` が bit 一致で確かめる）。
//!
//! 計測は `#[ignore]` の `loss_class_rejection_breakdown_measurement`（`--release`）。出力は `LCROW` 行。
//! GPU は未実測（CPU の値で代替しない）。

// 行列の添字（出力・入力・要素）で複数の配列を同時に引く数値テストのため、添字ループを許す
// （統合テスト `tests/create_graph_loss_class.rs` と同じ扱い）。
#![allow(clippy::needless_range_loop)]

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::tolerance::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

use super::probe::{Event, LOG};
use super::*;
use crate::loss_ops::{CrossEntropyOptions, TripletMarginOptions};
use crate::test_support::test_ops;
use crate::var::{Reduction, Var};

fn tt(v: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(v, shape).unwrap()
}

/// §81 と同じ決定的パターン `sin((i+phase)·0.731)·0.9 + 0.2`（乱数なし）。
fn pattern(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.731).sin() * 0.9 + 0.2)
        .collect()
}

/// REQ-2 の統一複合判定の閾値（絶対 1e-5 と相対 1e-3·|真値| の大きい方）。
fn req2_thr(truth: f64) -> f64 {
    ABSOLUTE_RESCUE_THRESHOLD.max(RELATIVE_TOLERANCE * truth.abs())
}

/// REQ-2 比 `|差| / 閾値`（1 未満で合格）。`norm_reparam.rs::tests::req2_ratio` と同じ式。
fn req2_ratio(actual: f64, truth: f64) -> f64 {
    (actual - truth).abs() / req2_thr(truth)
}

/// 2 のべき乗の上流係数 `2^exp`（Sum・n=1 では `s_b` の乗算が f32 で厳密）。
fn pow2(exp: i32) -> f32 {
    2f32.powi(exp)
}

/// 記録の RAII ガード。開始でスレッドローカルの記録を空で開き、Drop で閉じる（他テストへ漏らさない）。
struct Probe;

impl Probe {
    fn start() -> Self {
        LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
        Probe
    }
    fn take(&self) -> Vec<Event> {
        LOG.with(|l| {
            l.borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        LOG.with(|l| *l.borrow_mut() = None);
    }
}

/// 公開経路（`backward_create_graph`）の拒否の分類（計測用。本番が文面に依存するわけではない）。
#[derive(Clone, Debug, PartialEq)]
enum Verdict {
    Accept,
    /// `stage`（f32 範囲を超えうる上界）
    Stage,
    /// 入力の ±inf・上流の非有限
    NonFinite,
    /// Triplet の `pair_info` の各理由（再構成の桁落ち・スケール下限・大きな p の norm 丸め等）
    PairInfo,
    /// 相殺判定（`struct_check`）。`cls` は拒否した要素の分類、`flag` は f32 再現値の経路、
    /// `margin` は `err/閾値`（1 以上で拒否）。
    Cancel {
        cls: &'static str,
        flag: bool,
        margin: f64,
        what: String,
    },
    Other(String),
}

/// 拒否した `struct_check` の入力から、`coeff·m` の項と `|V_str − V|` の項のどちらが閾値を超えたかを分類する。
/// 閾値は判定と同じ `max(1e-5/scale, (1e-3/2)·|v|)`（`err·scale < 1e-5` または `err ≤ (1e-3/2)|v|` で通す）。
fn classify_check(vals: [f64; 5]) -> (&'static str, f64) {
    let [v, vs, m, scale, coeff] = vals;
    let ea = (vs - v).abs();
    let eb = coeff * m;
    let t = (ABSOLUTE_RESCUE_THRESHOLD / scale).max((RELATIVE_TOLERANCE / 2.0) * v.abs());
    let (a, b) = (ea / t, eb / t);
    let cls = match (a >= 1.0, b >= 1.0) {
        (true, true) => "両方",
        (false, true) => "coeff·m のみ",
        (true, false) => "|V_str−V| のみ",
        (false, false) => "合算でのみ",
    };
    (cls, (ea + eb) / t)
}

fn classify(err: Option<String>, events: &[Event]) -> Verdict {
    let Some(msg) = err else {
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::Check { ok: false, .. })),
            "受理なのに不合格の要素がある（記録のずれ）"
        );
        return Verdict::Accept;
    };
    if msg.contains("REQ-2 の判定を外れうるため拒否") {
        let pos = events
            .iter()
            .position(|e| matches!(e, Event::Check { ok: false, .. }))
            .expect("相殺判定で拒否したのに不合格の記録がない（記録のずれ）");
        let Event::Check { what, vals, .. } = &events[pos] else {
            unreachable!()
        };
        let flag = events[..pos]
            .iter()
            .rev()
            .find_map(|e| match e {
                Event::Flag(fam, f) if what.starts_with(fam) => Some(*f),
                _ => None,
            })
            .unwrap_or(false);
        let (cls, margin) = classify_check(*vals);
        Verdict::Cancel {
            cls,
            flag,
            margin,
            what: what.clone(),
        }
    } else if msg.contains("f32 範囲を超えうる") || msg.contains("逆数が f32 範囲") {
        Verdict::Stage
    } else if msg.contains("±inf") || msg.contains("非有限") {
        Verdict::NonFinite
    } else if msg.contains("Triplet:") {
        Verdict::PairInfo
    } else {
        Verdict::Other(msg)
    }
}

/// 公開経路を 1 回走らせる。`z0` は葉 `z`（1 次元）の値。拒否時は `child` が空であることを確かめる。
fn run_public<F>(z0: Vec<f32>, build: F) -> (Verdict, Vec<Event>)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let probe = Probe::start();
    let tape = Tape::new_with_ops(test_ops());
    let child = Tape::new_with_ops(test_ops());
    let n = z0.len();
    let z = tape.var(&tt(z0, &[n]));
    let loss = build(&tape, &z).expect("forward");
    let r = tape.backward_create_graph(&loss, &child);
    let events = probe.take();
    let err = match r {
        Ok(_) => None,
        Err(e) => {
            assert!(child.is_empty(), "拒否後に child が空でない");
            Some(match e {
                AutodiffError::Backward(m) => m,
                other => format!("{other:?}"),
            })
        }
    };
    (classify(err, &events), events)
}

/// 計測の補助: seed を決める（先頭 4・末尾 4・指定の特別な添字）。
fn seeds(n: usize, special: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n.min(4)).chain(n.saturating_sub(4)..n).collect();
    v.extend(special.iter().copied().filter(|&s| s < n));
    v.sort_unstable();
    v.dedup();
    v
}

/// 子テープの `dx` の要素 `j` を seed にした二階（`wrt` ごとの行）。逆伝播が失敗・到達しない場合は
/// 到達しない `wrt` を 0 とし、失敗は `None`。
fn child_rows(child: &Tape, dx: &Var<'_>, j: usize, wrt: &[&Var<'_>]) -> Option<Vec<Vec<f32>>> {
    let n: usize = dx.shape().iter().product();
    let gj = dx
        .contiguous()
        .ok()?
        .reshape(&[n])
        .ok()?
        .narrow(0, j, 1)
        .ok()?
        .sum(None)
        .ok()?;
    let g = child.backward(&gj).ok()?;
    Some(
        wrt.iter()
            .map(|w| match g.get(w).ok().flatten() {
                Some(t) => crate::eval::dense_vec(t),
                None => vec![0.0; w.shape().iter().product()],
            })
            .collect(),
    )
}

/// REQ-2 比の最大値と、参照自身の f64 の丸めの不確かさ（項の絶対値和 `m` から `4·2^-52·m`、同じ閾値で割った比）の最大値。
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    ratio: f64,
    unc: f64,
    n: usize,
}

impl Acc {
    fn add(&mut self, actual: f32, truth: f64, m: f64) {
        let a = f64::from(actual);
        let r = if a.is_nan() {
            f64::INFINITY
        } else {
            req2_ratio(a, truth)
        };
        self.ratio = self.ratio.max(r);
        self.unc = self.unc.max(4.0 * 2f64.powi(-52) * m / req2_thr(truth));
        self.n += 1;
    }
}

/// 1 行の計測結果。`x` は x（入力）への二階、`g` は `g`（上流）への二階の REQ-2 比。
#[derive(Clone, Debug)]
struct Meas {
    verdict: Verdict,
    x: Option<Acc>,
    g: Option<Acc>,
}

impl Meas {
    fn line(&self, family: &str, shape: &str, params: &str, exp: i32) -> String {
        let (v, cls, flag, margin, what) = match &self.verdict {
            Verdict::Accept => (
                "受理",
                "-".to_string(),
                "-",
                "-".to_string(),
                "-".to_string(),
            ),
            Verdict::Stage => ("段検査", "-".into(), "-", "-".into(), "-".into()),
            Verdict::NonFinite => ("入力の非有限", "-".into(), "-", "-".into(), "-".into()),
            Verdict::PairInfo => ("pair_info", "-".into(), "-", "-".into(), "-".into()),
            Verdict::Cancel {
                cls,
                flag,
                margin,
                what,
            } => (
                "相殺判定",
                (*cls).to_string(),
                if *flag { "flag" } else { "fast" },
                format!("{margin:.3e}"),
                what.split('（').next().unwrap_or("").replace('|', "/"),
            ),
            Verdict::Other(m) => (
                "その他",
                m.chars().take(40).collect(),
                "-",
                "-".into(),
                "-".into(),
            ),
        };
        let acc = |a: &Option<Acc>| {
            a.map_or("-".to_string(), |a| {
                format!("{:.2e}~{:.1e}", a.ratio, a.unc)
            })
        };
        let warn = match (&self.verdict, self.x, self.g) {
            (Verdict::Accept, Some(x), g)
                if x.ratio >= 1.0 || g.is_some_and(|g| g.ratio >= 1.0) =>
            {
                "要注意(受理なのに REQ-2 外)"
            }
            (Verdict::Cancel { .. }, Some(x), g)
                if x.ratio < 0.25 && g.is_none_or(|g| g.ratio < 0.25) =>
            {
                "過剰(REQ-2 内)"
            }
            (Verdict::Cancel { .. }, Some(x), g)
                if x.ratio >= 1.0 || g.is_some_and(|g| g.ratio >= 1.0) =>
            {
                "真に外れる"
            }
            _ => "",
        };
        format!(
            "LCROW {family}|{shape}|{params}|2^{exp}|{v}|{cls}|{flag}|{margin}|{what}|x={}|g={}|{warn}",
            acc(&self.x),
            acc(&self.g)
        )
    }
}

// =====================================================================================
// CE（既定 CE と CEWO）
// =====================================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
enum CeKind {
    /// 既定オプション（`Op::CrossEntropyLoss`。構造は `s_b⊙(softmax − onehot)`）
    Default,
    /// class_weight あり・Mean（`W = w_t`）
    Weighted,
    /// label_smoothing = 0.1・Sum
    Smooth,
}

#[derive(Clone, Copy, Debug)]
struct CeCase {
    kind: CeKind,
    c: usize,
    margin: f32,
    exp: i32,
}

const CE_TARGET: usize = 1;

impl CeCase {
    fn logits(&self) -> Vec<f32> {
        let mut x: Vec<f32> = pattern(self.c, 0.0).iter().map(|v| 0.1 * v).collect();
        x[CE_TARGET] += self.margin;
        x
    }

    fn weights(&self) -> Vec<f32> {
        match self.kind {
            CeKind::Weighted => (0..self.c)
                .map(|i| 0.5 + 0.4 * ((i as f32) * 0.731).sin())
                .collect(),
            _ => vec![1.0; self.c],
        }
    }

    fn eps(&self) -> f32 {
        if self.kind == CeKind::Smooth {
            0.1
        } else {
            0.0
        }
    }

    fn reduction(&self) -> Reduction {
        if self.kind == CeKind::Weighted {
            Reduction::Mean
        } else {
            Reduction::Sum
        }
    }

    fn options(&self) -> CrossEntropyOptions {
        match self.kind {
            CeKind::Default => CrossEntropyOptions::default(),
            CeKind::Weighted => {
                CrossEntropyOptions::default().class_weight(tt(self.weights(), &[self.c]))
            }
            CeKind::Smooth => CrossEntropyOptions::default().label_smoothing(self.eps()),
        }
    }

    fn name(&self) -> &'static str {
        match self.kind {
            CeKind::Default => "CE",
            CeKind::Weighted => "CEWO-weight",
            CeKind::Smooth => "CEWO-smooth",
        }
    }

    /// 公開経路の損失（上流の係数 `2^exp` を掛けた Sum）。葉 `z` は logits。
    fn build<'t>(&self, tape: &'t Tape, z: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let tgt = Tensor::<i32>::new(vec![CE_TARGET as i32], &[1]).unwrap();
        let l = z.reshape(&[1, self.c])?.cross_entropy_loss_with(
            &tgt,
            1,
            self.reduction(),
            &self.options(),
        )?;
        l.mul(&tape.var_no_grad(&Tensor::scalar(pow2(self.exp))))?
            .sum(None)
    }

    fn public(&self) -> (Verdict, Vec<Event>) {
        let case = *self;
        run_public(self.logits(), move |tape, z| case.build(tape, z))
    }

    /// 本番と同じ演算順の構造（既定 CE は `create_graph.rs` の腕、CEWO は `cewo_cgrad`）を子テープへ組み、
    /// seed ごとの `(x への二階の行, g への二階)` を返す。
    fn hand_rows(&self, seed_list: &[usize]) -> Vec<(usize, Vec<f32>, f32)> {
        let child = Tape::new_with_ops(test_ops());
        let shape = [1usize, self.c];
        let x_m = child.var(&tt(self.logits(), &shape));
        let g = child.var(&Tensor::scalar(pow2(self.exp)));
        let red = self.reduction();
        let dx = if self.kind == CeKind::Default {
            let s_b = super::super::loss_scale_bcast(&child, &g, red, false, 1, &shape).unwrap();
            let mut oh = vec![0f32; self.c];
            oh[CE_TARGET] = 1.0;
            let onehot = child.var_no_grad(&tt(oh, &shape));
            let y = x_m.softmax(1).unwrap();
            y.sub(&onehot).unwrap().mul(&s_b).unwrap()
        } else {
            let (a_s, b_v) = self.ab();
            let w_t = f64::from(self.weights()[CE_TARGET]);
            let denom_w = w_t;
            let w32 = denom_w as f32;
            let s_b = match red {
                Reduction::Mean => g
                    .div(&scalar(&child, w32))
                    .unwrap()
                    .broadcast_to(&shape)
                    .unwrap(),
                Reduction::Sum => g.broadcast_to(&shape).unwrap(),
            };
            let a_c = const_vec(&child, vec![a_s as f32; self.c], &shape).unwrap();
            let b_c = const_vec(&child, b_v.iter().map(|&b| b as f32).collect(), &shape).unwrap();
            let valid = mask(vec![true; self.c], &shape).unwrap();
            let zeros = zeros_like(&child, &shape).unwrap();
            let x_safe = Var::where_cond(&valid, &x_m, &zeros).unwrap();
            let sm = x_safe.softmax(1).unwrap();
            let scaled = s_b.mul(&a_c.mul(&sm).unwrap().sub(&b_c).unwrap()).unwrap();
            Var::where_cond(&valid, &scaled, &zeros).unwrap()
        };
        seed_list
            .iter()
            .map(|&j| {
                let r = child_rows(&child, &dx, j, &[&x_m, &g])
                    .unwrap_or_else(|| panic!("CE: seed {j} の子テープの逆伝播が失敗"));
                (j, r[0].clone(), r[1][0])
            })
            .collect()
    }

    /// `cewo_cgrad` の `A`・`B`（f64）。
    fn ab(&self) -> (f64, Vec<f64>) {
        let w: Vec<f64> = self.weights().iter().map(|&v| f64::from(v)).collect();
        let eps = f64::from(self.eps());
        let total: f64 = w.iter().sum();
        let c = self.c as f64;
        let w_t = w[CE_TARGET];
        let a = (1.0 - eps) * w_t + (eps / c) * total;
        let b = (0..self.c)
            .map(|k| {
                let delta = if k == CE_TARGET { 1.0 } else { 0.0 };
                (1.0 - eps) * w_t * delta + (eps / c) * w[k]
            })
            .collect();
        (a, b)
    }

    /// f64 の閉じた式: `(H[j][·], ∂/∂g, 各項の絶対値和)`。`1 − y_j` は他クラスの和で作る。
    fn reference(&self, j: usize) -> (Vec<f64>, f64, Vec<f64>, f64) {
        let x: Vec<f64> = self.logits().iter().map(|&v| f64::from(v)).collect();
        let mx = x.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let e: Vec<f64> = x.iter().map(|&v| (v - mx).exp()).collect();
        let total: f64 = e.iter().sum();
        let y: Vec<f64> = e.iter().map(|v| v / total).collect();
        let rest = |k: usize| -> f64 {
            if 2.0 * e[k] > total {
                e.iter()
                    .enumerate()
                    .filter(|&(i, _)| i != k)
                    .map(|(_, v)| *v)
                    .sum::<f64>()
                    / total
            } else {
                (total - e[k]) / total
            }
        };
        let (a, b) = self.ab();
        let w: Vec<f64> = self.weights().iter().map(|&v| f64::from(v)).collect();
        let w_t = w[CE_TARGET];
        let denom = if self.reduction() == Reduction::Mean {
            w_t
        } else {
            1.0
        };
        let s = f64::from(pow2(self.exp)) / denom;
        let row: Vec<f64> = (0..self.c)
            .map(|i| {
                if i == j {
                    s * a * y[j] * rest(j)
                } else {
                    -s * a * y[i] * y[j]
                }
            })
            .collect();
        let mabs: Vec<f64> = (0..self.c)
            .map(|i| s * a * y[j] * (if i == j { 1.0 } else { 0.0 } + y[i]))
            .collect();
        // g への二階 `(1/denom)(A y_j − B_j)`。`y_t − 1 = −rest_t` で桁落ちを避ける。
        let eps = f64::from(self.eps());
        let total_w: f64 = w.iter().sum();
        let smooth = eps / self.c as f64;
        let hard = if j == CE_TARGET { -rest(j) } else { y[j] };
        let gv = ((1.0 - eps) * w_t * hard + smooth * (total_w * y[j] - w[j])) / denom;
        let gm = (a * y[j] + b[j].abs()) / denom;
        (row, gv, mabs, gm)
    }

    fn measure(&self) -> Meas {
        let (verdict, _) = self.public();
        let sd = seeds(self.c, &[CE_TARGET, CE_TARGET + 1, self.c / 2]);
        let (mut x, mut g) = (Acc::default(), Acc::default());
        let rows = self.hand_rows(&sd);
        assert_eq!(rows.len(), sd.len(), "CE: 計測した seed 数が予定と違う");
        for (j, row, gval) in rows {
            let (rf, gv, mabs, gm) = self.reference(j);
            for i in 0..self.c {
                x.add(row[i], rf[i], mabs[i]);
            }
            g.add(gval, gv, gm);
        }
        assert_eq!(g.n, sd.len(), "CE: g の計測数が seed 数と違う");
        assert_eq!(x.n, sd.len() * self.c, "CE: x の計測数が違う");
        Meas {
            verdict,
            x: Some(x),
            g: Some(g),
        }
    }
}

fn ce_cases() -> Vec<CeCase> {
    let mut v = Vec::new();
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        for c in [2usize, 10, 100, 1000, 2000, 2100, 4000] {
            for margin in [5.0f32, 10.0, 20.0, 40.0] {
                for exp in [0i32, 13, 20] {
                    v.push(CeCase {
                        kind,
                        c,
                        margin,
                        exp,
                    });
                }
            }
        }
    }
    v
}

// =====================================================================================
// Cosine
// =====================================================================================

/// 1 行 `[1, d]` の Cosine。`b = a + delta·pattern`（`delta = 0` は完全に平行）。`y = +1` は `cf = −1`、
/// `y = −1`（margin 0）は平行に近い行で `cf = +1`。
#[derive(Clone, Copy, Debug)]
struct CosCase {
    d: usize,
    delta: f32,
    y: f32,
    exp: i32,
}

/// `cosine_cgrad` の ε（`loss_ops` の Cosine の定数）。手組みと `Lin` の並行計算が同じ値を使う。
const COS_EPSILON: f64 = 1e-12;

/// Cosine の行スケール `(inv, eps)`（`cosine_cgrad` と同じ式。`inv = 1/s`・`eps = ε/s²`、`s` は 2 のべき乗）。
fn cos_scales(a: &[f32], b: &[f32]) -> ((f64, f64), (f64, f64)) {
    let sqrt_eps = COS_EPSILON.sqrt();
    let amax = |v: &[f32]| v.iter().fold(0f64, |m, x| m.max(f64::from(*x).abs()));
    let (s1, s2) = (
        pow2_ceil(amax(a).max(sqrt_eps)),
        pow2_ceil(amax(b).max(sqrt_eps)),
    );
    (
        (1.0 / s1, 1.0 / s2),
        (COS_EPSILON / (s1 * s1), COS_EPSILON / (s2 * s2)),
    )
}

/// Cosine の手組みの seed ごとの二階: `[side][wrt]`（side 0 = dx1、1 = dx2。wrt は `[a, b, g]`）。
type CosRows = Vec<(usize, [Vec<Vec<f32>>; 2])>;

impl CosCase {
    fn ab(&self) -> (Vec<f32>, Vec<f32>) {
        let a = pattern(self.d, 0.0);
        let q = pattern(self.d, 3.0);
        let b = a.iter().zip(&q).map(|(a, q)| a + self.delta * q).collect();
        (a, b)
    }

    fn cf(&self) -> f64 {
        if self.y > 0.0 { -1.0 } else { 1.0 }
    }

    /// 葉 `z = [a | b]`。
    fn z0(&self) -> Vec<f32> {
        let (mut z, b) = self.ab();
        z.extend(b);
        z
    }

    fn build<'t>(&self, tape: &'t Tape, zv: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let d = self.d;
        let x1 = zv.narrow(0, 0, d)?.reshape(&[1, d])?;
        let x2 = zv.narrow(0, d, d)?.reshape(&[1, d])?;
        let l = x1.cosine_embedding_loss(&x2, &tt(vec![self.y], &[1]), 0.0, Reduction::Sum)?;
        l.mul(&tape.var_no_grad(&Tensor::scalar(pow2(self.exp))))?
            .sum(None)
    }

    fn public(&self) -> (Verdict, Vec<Event>) {
        let case = *self;
        run_public(self.z0(), move |tape, z| case.build(tape, z))
    }

    /// `cosine_cgrad` の構造（`[1, d]`・Sum・全要素 active）を同じ演算順で子テープへ組み、seed ごとの二階を返す。
    fn hand_rows(&self, seed_list: &[usize]) -> CosRows {
        let (a, b) = self.ab();
        self.hand_rows_with(a, b, seed_list)
    }

    /// [`CosCase::hand_rows`] を明示的な行 `(a, b)`（長さ `self.d`）で呼ぶ形（`y`・`exp` は `self` のまま。#4090）。
    fn hand_rows_with(&self, a: Vec<f32>, b: Vec<f32>, seed_list: &[usize]) -> CosRows {
        self.hand_rows_red(a, b, seed_list, None)
    }

    /// [`CosCase::hand_rows_with`] の Mean 版（`mean_n = Some(n)` は `s_b = g / n`。#4091）。行数は 1 のまま `n` を
    /// 分母に使うだけなので、Mean の `s_b` の丸めと `g` への `Div` の VJP を単体で検証できる。
    fn hand_rows_red(
        &self,
        a: Vec<f32>,
        b: Vec<f32>,
        seed_list: &[usize],
        mean_n: Option<usize>,
    ) -> CosRows {
        const EPSILON: f64 = COS_EPSILON;
        let d = self.d;
        let shape = [1usize, d];
        let child = Tape::new_with_ops(test_ops());
        let a_m = child.var(&tt(a.clone(), &shape));
        let b_m = child.var(&tt(b.clone(), &shape));
        let g = child.var(&Tensor::scalar(pow2(self.exp)));
        let sqrt_eps = EPSILON.sqrt();
        let max_a = a.iter().fold(0f64, |m, v| m.max(f64::from(*v).abs()));
        let max_b = b.iter().fold(0f64, |m, v| m.max(f64::from(*v).abs()));
        let (s1, s2) = (
            pow2_ceil(max_a.max(sqrt_eps)),
            pow2_ceil(max_b.max(sqrt_eps)),
        );
        let (inv1, inv2) = (1.0 / s1, 1.0 / s2);
        let (eps1, eps2) = (EPSILON / (s1 * s1), EPSILON / (s2 * s2));
        let (red, red_n) = match mean_n {
            Some(n) => (Reduction::Mean, n),
            None => (Reduction::Sum, 1),
        };
        let s_b = super::super::loss_scale_bcast(&child, &g, red, false, red_n, &shape).unwrap();
        let cv = |v: f32| const_vec(&child, vec![v; d], &shape).unwrap();
        let (r1_c, r2_c) = (cv(inv1 as f32), cv(inv2 as f32));
        let (e1_c, e2_c) = (cv(eps1 as f32), cv(eps2 as f32));
        let ap = a_m.mul(&r1_c).unwrap();
        let bp = b_m.mul(&r2_c).unwrap();
        let dim = 1;
        let m1p = reduce_dim_bcast(&ap.mul(&ap).unwrap(), dim, &shape)
            .unwrap()
            .add(&e1_c)
            .unwrap();
        let m2p = reduce_dim_bcast(&bp.mul(&bp).unwrap(), dim, &shape)
            .unwrap()
            .add(&e2_c)
            .unwrap();
        let den = m1p.sqrt().unwrap().mul(&m2p.sqrt().unwrap()).unwrap();
        let cos = reduce_dim_bcast(&ap.mul(&bp).unwrap(), dim, &shape)
            .unwrap()
            .div(&den)
            .unwrap();
        let inner1 = bp
            .div(&den)
            .unwrap()
            .sub(&cos.mul(&ap).unwrap().div(&m1p).unwrap())
            .unwrap();
        let inner2 = ap
            .div(&den)
            .unwrap()
            .sub(&cos.mul(&bp).unwrap().div(&m2p).unwrap())
            .unwrap();
        let coeff_c = cv(self.cf() as f32);
        let act_m = mask(vec![true; d], &shape).unwrap();
        let zeros = zeros_like(&child, &shape).unwrap();
        let sc = s_b.mul(&coeff_c).unwrap();
        let dx1 = Var::where_cond(
            &act_m,
            &sc.mul(&r1_c).unwrap().mul(&inner1).unwrap(),
            &zeros,
        )
        .unwrap();
        let dx2 = Var::where_cond(
            &act_m,
            &sc.mul(&r2_c).unwrap().mul(&inner2).unwrap(),
            &zeros,
        )
        .unwrap();
        seed_list
            .iter()
            .map(|&j| {
                let r1 = child_rows(&child, &dx1, j, &[&a_m, &b_m, &g])
                    .unwrap_or_else(|| panic!("Cosine: seed {j} の dx1 の逆伝播が失敗"));
                let r2 = child_rows(&child, &dx2, j, &[&a_m, &b_m, &g])
                    .unwrap_or_else(|| panic!("Cosine: seed {j} の dx2 の逆伝播が失敗"));
                (j, [r1, r2])
            })
            .collect()
    }

    /// f64 の閉じた式（元の変数）。`side = 0` は dx1（own = a）、`1` は dx2（own = b）。
    /// 戻り値は `([own への行, other への行], g への値, 各項の絶対値和, g の絶対値和)`。
    #[allow(clippy::type_complexity)]
    fn reference(&self, side: usize, j: usize) -> ([Vec<f64>; 2], f64, [Vec<f64>; 2], f64) {
        let (a32, b32) = self.ab();
        let (a, b): (Vec<f64>, Vec<f64>) = (
            a32.iter().map(|&v| f64::from(v)).collect(),
            b32.iter().map(|&v| f64::from(v)).collect(),
        );
        let (own, oth) = if side == 0 { (&a, &b) } else { (&b, &a) };
        let d = self.d;
        let m_own = own.iter().map(|v| v * v).sum::<f64>() + 1e-12;
        let m_oth = oth.iter().map(|v| v * v).sum::<f64>() + 1e-12;
        let den = (m_own * m_oth).sqrt();
        let dot: f64 = own.iter().zip(oth).map(|(x, y)| x * y).sum();
        let cos_m: f64 = own.iter().zip(oth).map(|(x, y)| (x * y).abs()).sum::<f64>() / den;
        let c = dot / den;
        let up = f64::from(pow2(self.exp));
        let s = up * self.cf();
        let g_oth = |i: usize| own[i] / den - c * oth[i] / m_oth;
        let (mut hs, mut ho) = (vec![0f64; d], vec![0f64; d]);
        let (mut ms, mut mo) = (vec![0f64; d], vec![0f64; d]);
        for i in 0..d {
            let delta = if i == j { 1.0 } else { 0.0 };
            // H_own,own[j][i] = −(oth_j own_i + oth_i own_j)/(den m_own) + 3c own_i own_j/m_own² − c δ/m_own
            hs[i] = s
                * (-(oth[j] * own[i] + oth[i] * own[j]) / (den * m_own)
                    + 3.0 * c * own[i] * own[j] / (m_own * m_own)
                    - c * delta / m_own);
            ms[i] = up
                * (((oth[j] * own[i]).abs() + (oth[i] * own[j]).abs()) / (den * m_own)
                    + 3.0 * cos_m * (own[i] * own[j]).abs() / (m_own * m_own)
                    + cos_m * delta / m_own);
            // H_own,oth[j][i] = δ/den − oth_i oth_j/(den m_oth) − g_oth_i own_j/m_own
            ho[i] = s * (delta / den - oth[i] * oth[j] / (den * m_oth) - g_oth(i) * own[j] / m_own);
            mo[i] = up
                * (delta / den
                    + (oth[i] * oth[j]).abs() / (den * m_oth)
                    + (own[i].abs() / den + cos_m * oth[i].abs() / m_oth) * own[j].abs() / m_own);
        }
        let gv = self.cf() * (oth[j] / den - c * own[j] / m_own);
        let gm = oth[j].abs() / den + cos_m * own[j].abs() / m_own;
        ([hs, ho], gv, [ms, mo], gm)
    }

    fn measure(&self) -> Meas {
        let (verdict, _) = self.public();
        let sd = seeds(self.d, &[]);
        let (mut x, mut g) = (Acc::default(), Acc::default());
        let all_rows = self.hand_rows(&sd);
        assert_eq!(
            all_rows.len(),
            sd.len(),
            "Cosine: 計測した seed 数が予定と違う"
        );
        for (j, rows) in all_rows {
            for side in 0..2 {
                let (h, gv, m, gm) = self.reference(side, j);
                // wrt は [a, b, g]。side 0 は own = a・other = b、side 1 は逆。
                let (own_idx, oth_idx) = if side == 0 { (0, 1) } else { (1, 0) };
                for i in 0..self.d {
                    x.add(rows[side][own_idx][i], h[0][i], m[0][i]);
                    x.add(rows[side][oth_idx][i], h[1][i], m[1][i]);
                }
                g.add(rows[side][2][0], gv, gm);
            }
        }
        assert_eq!(g.n, 2 * sd.len(), "Cosine: g の計測数が違う");
        assert_eq!(x.n, 4 * self.d * sd.len(), "Cosine: x の計測数が違う");
        Meas {
            verdict,
            x: Some(x),
            g: Some(g),
        }
    }
}

fn cos_cases() -> Vec<CosCase> {
    let mut v = Vec::new();
    for d in [1usize, 2, 8, 64, 512, 2048, 4096] {
        for delta in [0.0f32, 1e-2, 1e-4, 1e-6] {
            for y in [1.0f32, -1.0] {
                for exp in [0i32, 13, 20] {
                    v.push(CosCase { d, delta, y, exp });
                }
            }
        }
    }
    v
}
// =====================================================================================
// Triplet
// =====================================================================================

/// 1 行 `[1, d]` の Triplet。`a − p = [1, r·…]`（支配的な成分 1 と比 `r` の微小成分）、`a − n = [−2, r·…]`。
/// margin は十分大きく取り、常に active（`where(act, …)` の mask が全真）にする。
#[derive(Clone, Copy, Debug)]
struct TriCase {
    d: usize,
    ratio: f32,
    p: f32,
    swap: bool,
    exp: i32,
}

/// Triplet の手組みの seed ごとの二階: `[out(da, dp, dn)][wrt(a, p, n, g)]`。
type TriRows = Vec<(usize, Vec<Vec<Vec<f32>>>)>;

/// ノルム `‖v‖_p` のヘッセ行列の 1 行 `H[j][·]` と各項の絶対値和（`fac·(diag((p−1)q^(p−2)) − (p−1)(s⊙e)(s⊙e)ᵀ)`、
/// `q = |v|/‖v‖_p`、`e = q^(p−1)`）。対角は `(p−1)q_j^(p−2)(1 − q_j^p)` とし、`1 − q_j^p` は他の成分の和で作る。
fn norm_hess_row(v: &[f64], p: f64, j: usize) -> (Vec<f64>, Vec<f64>) {
    let mx = v.iter().fold(0f64, |m, x| m.max(x.abs()));
    let u: Vec<f64> = v.iter().map(|x| x.abs() / mx).collect();
    let up: Vec<f64> = u.iter().map(|x| x.powf(p)).collect();
    let s: f64 = up.iter().sum();
    let nn = s.powf(1.0 / p);
    let fac = 1.0 / (mx * nn);
    let q: Vec<f64> = u.iter().map(|x| x / nn).collect();
    let e: Vec<f64> = q.iter().map(|x| x.powf(p - 1.0)).collect();
    let sg: Vec<f64> = v.iter().map(|x| sign0(*x)).collect();
    let rest_j: f64 = up
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != j)
        .map(|(_, x)| *x)
        .sum::<f64>()
        / s;
    let pm1 = p - 1.0;
    let (mut h, mut m) = (vec![0f64; v.len()], vec![0f64; v.len()]);
    for i in 0..v.len() {
        if i == j {
            h[i] = fac * pm1 * q[j].powf(p - 2.0) * rest_j;
            m[i] = fac * pm1 * (q[j].powf(p - 2.0) + e[j] * e[j]);
        } else {
            h[i] = -fac * pm1 * sg[i] * sg[j] * e[i] * e[j];
            m[i] = fac * pm1 * e[i] * e[j];
        }
    }
    (h, m)
}

/// `where(act, s_b·v, 0)`（`triplet_cgrad` の `wrap`）。
fn wrap_act<'c>(
    act: &Tensor<bool>,
    s_b: &Var<'c>,
    zeros: &Var<'c>,
    v: &Var<'c>,
) -> Option<Var<'c>> {
    Var::where_cond(act, &s_b.mul(v).ok()?, zeros).ok()
}

impl TriCase {
    fn inputs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let a = pattern(self.d, 0.0);
        let vap: Vec<f32> = (0..self.d)
            .map(|k| {
                if k == 0 {
                    1.0
                } else {
                    self.ratio * (0.5 + 0.4 * ((k as f32) * 0.731).sin())
                }
            })
            .collect();
        // swap なしは `a − n ≈ [−2, …]`（`d_an > d_ap` 側で `p − n` の距離が大きい）。swap ありは `p − n` を
        // `a − n` より近くする配置（`p − n = [0.5, …]`、`a − n = (a − p) + (p − n) = [1.5, …]`）にして
        // `d_pn < d_an` とし、`an_coeff = 0`・`c_pn = 1` の `g_pn`／`epn` 経路（Bugbot 指摘）を実際に通す。
        let van: Vec<f32> = (0..self.d)
            .map(|k| {
                if self.swap {
                    let vpn = if k == 0 {
                        0.5
                    } else {
                        self.ratio * (0.7 + 0.2 * ((k as f32) * 0.5).cos())
                    };
                    vap[k] + vpn
                } else if k == 0 {
                    -2.0
                } else {
                    self.ratio * (0.7 + 0.2 * ((k as f32) * 0.5).cos())
                }
            })
            .collect();
        let p = a.iter().zip(&vap).map(|(a, v)| a - v).collect();
        let n = a.iter().zip(&van).map(|(a, v)| a - v).collect();
        (a, p, n)
    }

    fn options(&self) -> TripletMarginOptions {
        TripletMarginOptions::default()
            .margin(1000.0)
            .eps(0.0)
            .p(self.p)
            .swap(self.swap)
    }

    /// 葉 `z = [a | p | n]`。
    fn z0(&self) -> Vec<f32> {
        let (mut z, p, n) = self.inputs();
        z.extend(p);
        z.extend(n);
        z
    }

    fn build<'t>(&self, tape: &'t Tape, zv: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let d = self.d;
        let x = |k: usize| zv.narrow(0, k * d, d)?.reshape(&[1, d]);
        let l = x(0)?.triplet_margin_loss(&x(1)?, &x(2)?, &self.options(), Reduction::Sum)?;
        l.mul(&tape.var_no_grad(&Tensor::scalar(pow2(self.exp))))?
            .sum(None)
    }

    fn public(&self) -> (Verdict, Vec<Event>) {
        let case = *self;
        run_public(self.z0(), move |tape, z| case.build(tape, z))
    }

    fn f64_inputs(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        Self::f64_of(&self.inputs())
    }

    /// 明示的な行 `(a, p, n)` の f64 化（#4133: 既定の [`TriCase::inputs`] 以外の行を f64 参照へ渡す）。
    fn f64_of(rows: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let to = |s: &[f32]| -> Vec<f64> { s.iter().map(|&v| f64::from(v)).collect() };
        (to(&rows.0), to(&rows.1), to(&rows.2))
    }

    /// `(c_an, c_pn)`（`triplet_cgrad` と同じく swap でなければ `(1, 0)`）。
    fn coefs(&self) -> (f64, f64) {
        self.coefs_with(&self.inputs())
    }

    /// [`TriCase::coefs`] を明示的な行で呼ぶ形（#4133）。
    fn coefs_with(&self, rows: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> (f64, f64) {
        let (a, p, n) = Self::f64_of(rows);
        let params = crate::eval::TripletDistanceParams {
            p: f64::from(self.p),
            eps: 0.0,
            swap: self.swap,
        };
        let (stats, ..) = crate::eval::triplet_distance_stats(&a, &p, &n, 0, self.d, &params);
        let c_an = if self.swap { stats.an_coeff } else { 1.0 };
        (c_an, if self.swap { 1.0 - c_an } else { 0.0 })
    }

    /// `triplet_cgrad` の構造（`[1, d]`・Sum・eps 0・全要素 active）を同じ演算順で子テープへ組む。
    /// `pair_info` が拒否する入力は組めない（`None`）。
    fn hand_rows(&self, seed_list: &[usize]) -> Option<TriRows> {
        self.hand_rows_with(self.inputs(), 0.0, None, seed_list)
    }

    /// [`TriCase::hand_rows`] を明示的な行 `(a, p, n)`（長さ `self.d`）・`eps`・Mean の分母で呼ぶ形（#4095。`p`・`swap`・
    /// `exp` は `self` のまま）。`mean_n = Some(n)` は `s_b = g / n`（行数は 1 のまま `n` を分母に使うだけ）。
    fn hand_rows_with(
        &self,
        rows: (Vec<f32>, Vec<f32>, Vec<f32>),
        eps: f32,
        mean_n: Option<usize>,
        seed_list: &[usize],
    ) -> Option<TriRows> {
        let d = self.d;
        let shape = [1usize, d];
        let (a32, p32, n32) = rows;
        let to = |s: &[f32]| -> Vec<f64> { s.iter().map(|&v| f64::from(v)).collect() };
        let (a64, p64v, n64) = (to(&a32), to(&p32), to(&n32));
        let p64 = f64::from(self.p);
        let sb = match mean_n {
            Some(n) => f64::from(pow2(self.exp)) / n as f64,
            None => f64::from(pow2(self.exp)),
        };
        let params = crate::eval::TripletDistanceParams {
            p: p64,
            eps: f64::from(eps),
            swap: self.swap,
        };
        let (stats, diff_ap, diff_an, diff_pn) =
            crate::eval::triplet_distance_stats(&a64, &p64v, &n64, 0, d, &params);
        let row32 =
            |x: &[f32], y: &[f32]| -> Vec<f32> { (0..d).map(|k| (x[k] - y[k]) + eps).collect() };
        let (rec_ap, rec_an, rec_pn) = (row32(&a32, &p32), row32(&a32, &n32), row32(&p32, &n32));
        let info_ap = pair_info(&[diff_ap], &[rec_ap], d, p64, sb).ok()?;
        let info_an = pair_info(&[diff_an], &[rec_an], d, p64, sb).ok()?;
        let info_pn = if self.swap {
            Some(pair_info(&[diff_pn], &[rec_pn], d, p64, sb).ok()?)
        } else {
            None
        };
        let child = Tape::new_with_ops(test_ops());
        let a_m = child.var(&tt(a32, &shape));
        let p_m = child.var(&tt(p32, &shape));
        let n_m = child.var(&tt(n32, &shape));
        let g = child.var(&Tensor::scalar(pow2(self.exp)));
        let (red, red_n) = match mean_n {
            Some(n) => (Reduction::Mean, n),
            None => (Reduction::Sum, 1),
        };
        let s_b = super::super::loss_scale_bcast(&child, &g, red, false, red_n, &shape).ok()?;
        let eps_c = scalar(&child, eps);
        let zeros = zeros_like(&child, &shape).ok()?;
        let v_ap = a_m.sub(&p_m).ok()?.add(&eps_c).ok()?;
        let v_an = a_m.sub(&n_m).ok()?.add(&eps_c).ok()?;
        let g_ap = pair_grad(&child, &v_ap, &info_ap, &shape, self.p).ok()?;
        let g_an = pair_grad(&child, &v_an, &info_an, &shape, self.p).ok()?;
        let (da, dp, dn) = if let Some(info) = &info_pn {
            let v_pn = p_m.sub(&n_m).ok()?.add(&eps_c).ok()?;
            let g_pn = pair_grad(&child, &v_pn, info, &shape, self.p).ok()?;
            let half = scalar(&child, 0.5);
            let (c_one, c_zero, c_half) = (
                stats.an_coeff >= 1.0,
                stats.an_coeff <= 0.0,
                stats.an_coeff > 0.0 && stats.an_coeff < 1.0,
            );
            let m = |f: bool| mask(vec![f; d], &shape).ok();
            let (m_one, m_zero, m_half) = (m(c_one)?, m(c_zero)?, m(c_half)?);
            let ean = Var::where_cond(
                &m_one,
                &g_an,
                &Var::where_cond(&m_half, &g_an.mul(&half).ok()?, &zeros).ok()?,
            )
            .ok()?;
            let epn = Var::where_cond(
                &m_zero,
                &g_pn,
                &Var::where_cond(&m_half, &g_pn.mul(&half).ok()?, &zeros).ok()?,
            )
            .ok()?;
            (
                g_ap.sub(&ean).ok()?,
                g_ap.neg().ok()?.sub(&epn).ok()?,
                ean.add(&epn).ok()?,
            )
        } else {
            (g_ap.sub(&g_an).ok()?, g_ap.neg().ok()?, g_an)
        };
        let act_m = mask(vec![true; d], &shape).ok()?;
        let outs = [
            wrap_act(&act_m, &s_b, &zeros, &da)?,
            wrap_act(&act_m, &s_b, &zeros, &dp)?,
            wrap_act(&act_m, &s_b, &zeros, &dn)?,
        ];
        Some(
            seed_list
                .iter()
                .map(|&j| {
                    let r: Vec<_> = outs
                        .iter()
                        .map(|o| {
                            child_rows(&child, o, j, &[&a_m, &p_m, &n_m, &g]).unwrap_or_else(|| {
                                panic!("Triplet: seed {j} の子テープの逆伝播が失敗")
                            })
                        })
                        .collect();
                    (j, r)
                })
                .collect(),
        )
    }

    /// f64 の閉じた式。`x[out][wrt]` は `(truth の行, 各項の絶対値和の行)`、`g[out]` は `(truth, 絶対値和)`。
    fn reference(&self, j: usize) -> TriRef {
        self.reference_with(&self.inputs(), j)
    }

    /// [`TriCase::reference`] を明示的な行で呼ぶ形（#4133。`p`・`swap`・`exp` は `self`、`eps = 0`・Sum）。
    fn reference_with(&self, rows: &(Vec<f32>, Vec<f32>, Vec<f32>), j: usize) -> TriRef {
        let (a, p, n) = Self::f64_of(rows);
        let sub =
            |x: &[f64], y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(a, b)| a - b).collect() };
        let (vap, van, vpn) = (sub(&a, &p), sub(&a, &n), sub(&p, &n));
        let pe = f64::from(self.p);
        let (c_an, c_pn) = self.coefs_with(rows);
        let up = f64::from(pow2(self.exp));
        // (out, wrt) の [H_ap, H_an, H_pn] の係数。
        let table: [[[f64; 3]; 3]; 3] = [
            [[1.0, -c_an, 0.0], [-1.0, 0.0, 0.0], [0.0, c_an, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, -c_pn], [0.0, 0.0, c_pn]],
            [[0.0, c_an, 0.0], [0.0, 0.0, c_pn], [0.0, -c_an, -c_pn]],
        ];
        let gtab = [[1.0, -c_an, 0.0], [-1.0, 0.0, -c_pn], [0.0, c_an, c_pn]];
        let hs: Vec<(Vec<f64>, Vec<f64>)> = [&vap, &van, &vpn]
            .iter()
            .map(|v| norm_hess_row(v, pe, j))
            .collect();
        let core = |v: &[f64]| -> (f64, f64) {
            // g への二階: sign(v_j)·e_j と絶対値。
            let mx = v.iter().fold(0f64, |m, x| m.max(x.abs()));
            let nn = v
                .iter()
                .map(|x| (x.abs() / mx).powf(pe))
                .sum::<f64>()
                .powf(1.0 / pe);
            let e = ((v[j].abs() / mx) / nn).powf(pe - 1.0);
            (sign0(v[j]) * e, e)
        };
        let cores = [core(&vap), core(&van), core(&vpn)];
        let mut x = Vec::new();
        let mut g = Vec::new();
        for o in 0..3 {
            let mut per_w = Vec::new();
            for co in &table[o] {
                let truth: Vec<f64> = (0..self.d)
                    .map(|i| (0..3).map(|k| co[k] * hs[k].0[i]).sum::<f64>() * up)
                    .collect();
                let m: Vec<f64> = (0..self.d)
                    .map(|i| (0..3).map(|k| co[k].abs() * hs[k].1[i]).sum::<f64>() * up)
                    .collect();
                per_w.push((truth, m));
            }
            x.push(per_w);
            g.push((
                (0..3).map(|k| gtab[o][k] * cores[k].0).sum::<f64>(),
                (0..3).map(|k| gtab[o][k].abs() * cores[k].1).sum::<f64>(),
            ));
        }
        TriRef { x, g }
    }

    fn measure(&self) -> Meas {
        let (verdict, _) = self.public();
        let sd = seeds(self.d, &[0, 1]);
        let Some(rows) = self.hand_rows(&sd) else {
            return Meas {
                verdict,
                x: None,
                g: None,
            };
        };
        assert_eq!(
            rows.len(),
            sd.len(),
            "Triplet: 計測した seed 数が予定と違う"
        );
        let (mut x, mut g) = (Acc::default(), Acc::default());
        for (j, per_out) in rows {
            let r = self.reference(j);
            for o in 0..3 {
                for w in 0..3 {
                    for i in 0..self.d {
                        x.add(per_out[o][w][i], r.x[o][w].0[i], r.x[o][w].1[i]);
                    }
                }
                g.add(per_out[o][3][0], r.g[o].0, r.g[o].1);
            }
        }
        assert_eq!(g.n, 3 * sd.len(), "Triplet: g の計測数が違う");
        assert_eq!(x.n, 9 * self.d * sd.len(), "Triplet: x の計測数が違う");
        Meas {
            verdict,
            x: Some(x),
            g: Some(g),
        }
    }
}

/// [`TriCase::reference`] の結果。
struct TriRef {
    x: Vec<Vec<(Vec<f64>, Vec<f64>)>>,
    g: Vec<(f64, f64)>,
}

fn tri_cases() -> Vec<TriCase> {
    let mut v = Vec::new();
    for ratio in [1e-1f32, 1e-3, 1e-6] {
        for d in [2usize, 8, 64, 512] {
            for p in [2.0f32, 1.5, 3.0, 10.0, 50.0] {
                for swap in [false, true] {
                    for exp in [0i32, 20] {
                        v.push(TriCase {
                            d,
                            ratio,
                            p,
                            swap,
                            exp,
                        });
                    }
                }
            }
        }
    }
    v
}
/// 計測の全行（族・形状・パラメータ・上流の指数・結果）。
fn all_measurements() -> Vec<(String, Meas, String)> {
    let mut out = Vec::new();
    for c in cos_cases() {
        let m = c.measure();
        let l = m.line(
            "Cosine",
            &format!("[1,{}]", c.d),
            &format!("delta={:e},y={}", c.delta, c.y),
            c.exp,
        );
        out.push(("Cosine".to_string(), m, l));
    }
    for c in ce_cases() {
        let m = c.measure();
        let l = m.line(
            c.name(),
            &format!("[1,{}]", c.c),
            &format!("margin={}", c.margin),
            c.exp,
        );
        out.push((c.name().to_string(), m, l));
    }
    for c in tri_cases() {
        let m = c.measure();
        let l = m.line(
            "Triplet",
            &format!("[1,{}]", c.d),
            &format!("r={:e},p={},swap={}", c.ratio, c.p, c.swap),
            c.exp,
        );
        out.push(("Triplet".to_string(), m, l));
    }
    out
}

#[test]
#[ignore = "計測。--release で実行する"]
fn loss_class_rejection_breakdown_measurement() {
    let t0 = std::time::Instant::now();
    eprintln!(
        "LCROW family|shape|params|scale|verdict|class|path|margin(err/thr)|what|x=REQ2比~参照不確かさ|g=同|note"
    );
    for (_, _, l) in all_measurements() {
        eprintln!("{l}");
    }
    eprintln!("LCTIME {:.1}s", t0.elapsed().as_secs_f64());
}
// =====================================================================================
// 手組みの構造が本番と一致すること・閉じた式の検証（通常テスト）
// =====================================================================================

fn bits(v: &[f32]) -> Vec<u32> {
    // 符号付きゼロ（`-0.0` と `+0.0`）は同一視する（勾配の累積順で符号だけ変わる）。
    v.iter()
        .map(|x| if *x == 0.0 { 0 } else { x.to_bits() })
        .collect()
}

/// 公開経路（受理される入力）の `cg.grad(z)` の要素 `j` を seed にした二階（`z` への行）。
fn public_rows<F>(z0: Vec<f32>, idx: &[usize], build: F) -> Vec<Vec<f32>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> Result<Var<'t>, AutodiffError>,
{
    let tape = Tape::new_with_ops(test_ops());
    let child = Tape::new_with_ops(test_ops());
    let n = z0.len();
    let z = tape.var(&tt(z0, &[n]));
    let loss = build(&tape, &z).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("受理される入力");
    let gz = cg.grad(&z).unwrap().expect("1 階勾配");
    let cz = cg.child_var(&z).unwrap().expect("写し");
    idx.iter()
        .map(|&j| child_rows(&child, &gz, j, &[&cz]).expect("二階").remove(0))
        .collect()
}

#[test]
fn ce_hand_built_structure_matches_public_path() {
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        for exp in [0, 7] {
            let c = CeCase {
                kind,
                c: 10,
                margin: 5.0,
                exp,
            };
            assert_eq!(c.public().0, Verdict::Accept, "{c:?}");
            let sd: Vec<usize> = (0..c.c).collect();
            let hand = c.hand_rows(&sd);
            let case = c;
            let pubr = public_rows(c.logits(), &sd, move |t, z| case.build(t, z));
            for ((j, row, _), pr) in hand.iter().zip(&pubr) {
                assert_eq!(bits(row), bits(pr), "{c:?} seed {j}");
            }
        }
    }
}

#[test]
fn cosine_hand_built_structure_matches_public_path() {
    for (delta, y, exp) in [(1e-2f32, 1.0f32, 0), (0.0, -1.0, 7), (1e-4, 1.0, 3)] {
        let c = CosCase {
            d: 8,
            delta,
            y,
            exp,
        };
        assert_eq!(c.public().0, Verdict::Accept, "{c:?}");
        let sd: Vec<usize> = (0..c.d).collect();
        let hand = c.hand_rows(&sd);
        let idx: Vec<usize> = (0..2 * c.d).collect();
        let case = c;
        let pubr = public_rows(c.z0(), &idx, move |t, z| case.build(t, z));
        for (j, rows) in &hand {
            for side in 0..2 {
                let mut hr = rows[side][0].clone();
                hr.extend(&rows[side][1]);
                assert_eq!(
                    bits(&hr),
                    bits(&pubr[side * c.d + j]),
                    "{c:?} side {side} seed {j}"
                );
            }
        }
    }
}

#[test]
fn triplet_hand_built_structure_matches_public_path() {
    for (p, swap, exp) in [(2.0f32, false, 0), (2.0, true, 0), (1.5, false, 0)] {
        let c = TriCase {
            d: 6,
            ratio: 0.3,
            p,
            swap,
            exp,
        };
        assert_eq!(c.public().0, Verdict::Accept, "{c:?}");
        // swap あり: `g_pn`／`epn` の項を実際に通す配置（`c_pn = 1`）であること。
        assert_eq!(
            c.coefs(),
            if swap { (0.0, 1.0) } else { (1.0, 0.0) },
            "{c:?}"
        );
        let sd: Vec<usize> = (0..c.d).collect();
        let hand = c.hand_rows(&sd).expect("pair_info を通る");
        let idx: Vec<usize> = (0..3 * c.d).collect();
        let case = c;
        let pubr = public_rows(c.z0(), &idx, move |t, z| case.build(t, z));
        for (j, per_out) in &hand {
            for o in 0..3 {
                let hr: Vec<f32> = per_out[o][..3].iter().flatten().copied().collect();
                assert_eq!(
                    bits(&hr),
                    bits(&pubr[o * c.d + j]),
                    "{c:?} out {o} seed {j}"
                );
            }
        }
    }
}

/// 閉じた式の参照を f64 の勾配の中央差分で確かめる（O(1) の入力だけ。相殺する要素の検証には使わない）。
fn assert_close(actual: f64, truth: f64, ctx: &str) {
    assert!(
        (actual - truth).abs() <= 1e-6 * truth.abs().max(1e-2),
        "{ctx}: 中央差分 {actual:e} と閉じた式 {truth:e} が一致しない"
    );
}

#[test]
fn ce_closed_form_matches_f64_central_difference() {
    let h = 1e-5;
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        let c = CeCase {
            kind,
            c: 4,
            margin: 1.0,
            exp: 0,
        };
        let (a, b) = c.ab();
        let w_t = f64::from(c.weights()[CE_TARGET]);
        let denom = if kind == CeKind::Weighted { w_t } else { 1.0 };
        let grad = |x: &[f64]| -> Vec<f64> {
            let mx = x.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            let e: Vec<f64> = x.iter().map(|v| (v - mx).exp()).collect();
            let t: f64 = e.iter().sum();
            (0..x.len())
                .map(|k| (a * e[k] / t - b[k]) / denom)
                .collect()
        };
        let x0: Vec<f64> = c.logits().iter().map(|&v| f64::from(v)).collect();
        for j in 0..c.c {
            let (row, gv, ..) = c.reference(j);
            for i in 0..c.c {
                let (mut xp, mut xm) = (x0.clone(), x0.clone());
                xp[i] += h;
                xm[i] -= h;
                let fd = (grad(&xp)[j] - grad(&xm)[j]) / (2.0 * h);
                assert_close(fd, row[i], &format!("{kind:?} H[{j}][{i}]"));
            }
            assert_close(grad(&x0)[j], gv, &format!("{kind:?} g[{j}]"));
        }
    }
}

#[test]
fn cosine_closed_form_matches_f64_central_difference() {
    let h = 1e-6;
    for (delta, y) in [(0.3f32, 1.0f32), (0.3, -1.0)] {
        let c = CosCase {
            d: 3,
            delta,
            y,
            exp: 0,
        };
        let (a32, b32) = c.ab();
        let to = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
        let (a0, b0) = (to(&a32), to(&b32));
        // dx1 = cf·(b/den − cos·a/m1)
        let grad1 = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let m1 = a.iter().map(|v| v * v).sum::<f64>() + 1e-12;
            let m2 = b.iter().map(|v| v * v).sum::<f64>() + 1e-12;
            let den = (m1 * m2).sqrt();
            let cos = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / den;
            (0..a.len())
                .map(|k| c.cf() * (b[k] / den - cos * a[k] / m1))
                .collect()
        };
        for j in 0..c.d {
            let (hh, gv, ..) = c.reference(0, j);
            for i in 0..c.d {
                let (mut ap, mut am) = (a0.clone(), a0.clone());
                ap[i] += h;
                am[i] -= h;
                assert_close(
                    (grad1(&ap, &b0)[j] - grad1(&am, &b0)[j]) / (2.0 * h),
                    hh[0][i],
                    "cos H_aa",
                );
                let (mut bp, mut bm) = (b0.clone(), b0.clone());
                bp[i] += h;
                bm[i] -= h;
                assert_close(
                    (grad1(&a0, &bp)[j] - grad1(&a0, &bm)[j]) / (2.0 * h),
                    hh[1][i],
                    "cos H_ab",
                );
            }
            assert_close(grad1(&a0, &b0)[j], gv, "cos g");
        }
    }
}

#[test]
fn triplet_closed_form_matches_f64_central_difference() {
    let h = 1e-6;
    for (p, swap) in [(2.0f32, false), (3.0, false), (3.0, true), (1.5, false)] {
        let c = TriCase {
            d: 3,
            ratio: 0.3,
            p,
            swap,
            exp: 0,
        };
        let (a0, p0, n0) = c.f64_inputs();
        let (c_an, c_pn) = c.coefs();
        assert_eq!(c_pn == 1.0, swap, "{c:?}: swap ありは c_pn を通す");
        let pe = f64::from(p);
        let core = |v: &[f64]| -> Vec<f64> {
            let nn = v
                .iter()
                .map(|x| x.abs().powf(pe))
                .sum::<f64>()
                .powf(1.0 / pe);
            v.iter()
                .map(|x| sign0(*x) * (x.abs() / nn).powf(pe - 1.0))
                .collect()
        };
        let sub =
            |x: &[f64], y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(a, b)| a - b).collect() };
        let outs = |a: &[f64], p: &[f64], n: &[f64]| -> [Vec<f64>; 3] {
            let (gap, gan, gpn) = (core(&sub(a, p)), core(&sub(a, n)), core(&sub(p, n)));
            let k = c.d;
            [
                (0..k).map(|i| gap[i] - c_an * gan[i]).collect(),
                (0..k).map(|i| -gap[i] - c_pn * gpn[i]).collect(),
                (0..k).map(|i| c_an * gan[i] + c_pn * gpn[i]).collect(),
            ]
        };
        for j in 0..c.d {
            let r = c.reference(j);
            let base = outs(&a0, &p0, &n0);
            for o in 0..3 {
                assert_close(base[o][j], r.g[o].0, &format!("{c:?} g out {o}"));
                for w in 0..3 {
                    for i in 0..c.d {
                        let bump = |s: f64| -> [Vec<f64>; 3] {
                            let (mut a, mut p, mut n) = (a0.clone(), p0.clone(), n0.clone());
                            match w {
                                0 => a[i] += s,
                                1 => p[i] += s,
                                _ => n[i] += s,
                            }
                            outs(&a, &p, &n)
                        };
                        let fd = (bump(h)[o][j] - bump(-h)[o][j]) / (2.0 * h);
                        assert_close(
                            fd,
                            r.x[o][w].0[i],
                            &format!("{c:?} H out {o} wrt {w} [{j}][{i}]"),
                        );
                    }
                }
            }
        }
    }
}

// =====================================================================================
// 後続（#4090〜#4096）が受理に変える入力（設計 §82.4）
// =====================================================================================

/// 後続の既定 CE・CEWO の切り替え（#4093・#4094）で受理に変える入力（#4094 で C=100 の 2 件が受理に変わった。
/// C=1000・2100 は拒否のまま）。旧判定では「相殺判定の `coeff·m` の項だけ」で拒否されるが、実際の f32 の二階は f64 の閉じた式と REQ-2 の閾値の 1/4 以内で一致する。
fn ce_followup_candidates() -> Vec<CeCase> {
    let mk = |kind, c, margin, exp| CeCase {
        kind,
        c,
        margin,
        exp,
    };
    vec![
        mk(CeKind::Default, 100, 10.0, 0),
        mk(CeKind::Default, 2100, 10.0, 0),
        mk(CeKind::Weighted, 100, 10.0, 0),
        mk(CeKind::Smooth, 1000, 10.0, 0),
    ]
}

/// 後続の Cosine の切り替え（#4090・#4091）で受理に変える入力。
fn cosine_followup_candidates() -> Vec<CosCase> {
    let mk = |d, delta, y, exp| CosCase { d, delta, y, exp };
    vec![
        mk(512, 1e-2, 1.0, 20),
        mk(2048, 0.0, -1.0, 13),
        mk(4096, 1e-4, 1.0, 13),
    ]
}

/// 後続の Triplet の切り替え（#4095・#4096）で受理に変える入力（#4096 で k=1・k=3 の 2 件が受理に変わった。残る 3 件は
/// 拒否のまま。最後の 1 件は `flag` 経路〈f32 再現値で組む経路〉で拒否される入力）。
fn triplet_followup_candidates() -> Vec<TriCase> {
    let mk = |d, ratio, p, swap, exp| TriCase {
        d,
        ratio,
        p,
        swap,
        exp,
    };
    vec![
        mk(64, 1e-1, 3.0, false, 0),
        mk(64, 1e-1, 3.0, true, 0),
        mk(512, 1e-1, 2.0, false, 0),
        mk(64, 1e-3, 1.5, false, 0),
        mk(64, 1e-6, 10.0, false, 0),
    ]
}

/// 候補が「現在は相殺判定の `coeff·m` の項だけで拒否される」ことと「実際の f32 の二階が f64 の閉じた式と
/// REQ-2 の閾値の 1/4 以内で一致する」ことを固定する。後続が候補を受理に変えたら `verdict` の assert を受理へ更新し、
/// REQ-2 比の assert は残す（受理に変えても二階が REQ-2 内であることを守る）。
#[test]
fn loss_class_followup_candidates_are_req2_within_f64() {
    let req2 = |name: &str, m: Meas| {
        for (label, acc) in [("x", m.x), ("g", m.g)] {
            let a = acc.unwrap_or_else(|| panic!("{name}: {label} が未測定"));
            assert!(a.ratio < 0.25, "{name}: {label} の REQ-2 比 {:e}", a.ratio);
            assert!(a.unc < 0.05, "{name}: {label} の参照の不確かさ {:e}", a.unc);
        }
    };
    let check = |name: &str, m: Meas, want_flag: bool| {
        match &m.verdict {
            Verdict::Cancel { cls, flag, .. } => {
                assert_eq!(*cls, "coeff·m のみ", "{name}: 拒否の分類");
                assert_eq!(*flag, want_flag, "{name}: flag 経路");
            }
            v => panic!("{name}: 現在は相殺判定で拒否されるはず: {v:?}"),
        }
        req2(name, m);
    };
    // CE・CEWO（#4094）: C=100 の 2 件は `Lin` の上界で受理に変わった。C=1000・2100 は `Lin` の和の上界
    // （`γ_{C−1}·Σ|e|` が対角の `1 − y_j` の打ち消しへ乗る）でも閾値を超えるため拒否のまま（設計 §82.11）。
    for c in ce_followup_candidates() {
        let name = format!("{c:?}");
        let m = c.measure();
        if c.c == 100 {
            assert_eq!(m.verdict, Verdict::Accept, "{name}: Lin の上界で受理");
            req2(&name, m);
        } else {
            check(&name, m, false);
        }
    }
    for c in cosine_followup_candidates() {
        check(&format!("{c:?}"), c.measure(), false);
    }
    // Triplet（#4096）: d=64 の swap あり p=3 と p=1.5 の 2 件は `Lin` の上界で受理に変わった。残る 3 件は拒否のまま
    // （k=0 は 1 要素だけ上界が閾値を超える、k=2 は 9 ブロックの要素の救済走査が走査量の上限を超える、k=4 は `flag` 行で
    // 上界が閾値を超える要素が複数残る。設計 §82.13.3）。
    for (k, c) in triplet_followup_candidates().into_iter().enumerate() {
        assert_eq!(c.coefs().1 == 1.0, c.swap, "{c:?}: swap ありは c_pn を通す");
        let name = format!("{c:?}");
        let m = c.measure();
        if k == 1 || k == 3 {
            assert_eq!(m.verdict, Verdict::Accept, "{name}: Lin の上界で受理");
            req2(&name, m);
        } else {
            check(&name, m, k == 4);
        }
    }
}

/// 既存の拒否プローブの入力（Codex・Bugbot 指摘）に近い入力は拒否のままで、実際の f32 の二階が REQ-2 を
/// 外すこと（判定を緩めていないことの確認）。
#[test]
fn existing_rejection_probes_stay_rejected_and_truly_fail_req2() {
    let ce = CeCase {
        kind: CeKind::Default,
        c: 2,
        margin: 20.0,
        exp: 13,
    };
    let m = ce.measure();
    assert!(matches!(m.verdict, Verdict::Cancel { .. }), "{m:?}");
    assert!(
        m.x.unwrap().ratio >= 1.0,
        "CE の確信度の高い行は真に REQ-2 を外す"
    );
    let cos = CosCase {
        d: 1,
        delta: 0.0,
        y: 1.0,
        exp: 20,
    };
    let m = cos.measure();
    assert!(matches!(m.verdict, Verdict::Cancel { .. }), "{m:?}");
    assert!(
        m.x.unwrap().ratio >= 1.0,
        "Cosine の平行な行は真に REQ-2 を外す"
    );
}

#[test]
fn classify_check_splits_the_two_terms() {
    // scale 1: 閾値は max(1e-5, 5e-4·|v|)。v = 0 なら 1e-5。
    assert_eq!(classify_check([0.0, 0.0, 1.0, 1.0, 1e-4]).0, "coeff·m のみ");
    assert_eq!(
        classify_check([0.0, 1e-3, 0.0, 1.0, 1e-4]).0,
        "|V_str−V| のみ"
    );
    assert_eq!(classify_check([0.0, 1e-3, 1.0, 1.0, 1e-4]).0, "両方");
    assert_eq!(classify_check([0.0, 8e-6, 0.08, 1.0, 1e-4]).0, "合算でのみ");
}
// =====================================================================================
// #4090: Cosine の forward の総称化（判定不変）と `Lin` の上界の soundness
// =====================================================================================

/// 乱数の入力用の決定的な xorshift（`[0, 1)` の f64）。
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn signed(&mut self) -> f32 {
        (self.unit() * 2.0 - 1.0) as f32
    }
}

/// NaN を 1 つの bit 列へ正規化した f64 の bit 列（NaN の payload の違いを同一視）。
fn f64_bits(v: f64) -> u64 {
    if v.is_nan() { u64::MAX } else { v.to_bits() }
}

fn f32_bits(v: f32) -> u32 {
    if v.is_nan() { u32::MAX } else { v.to_bits() }
}

/// 総称化の前に `cosine_cancel_check` が書いていた f64 の参照と f32 の再現の式（逐語のコピー。前例: #4077 の
/// `LinRef`）。`cos_fwd` が全中間値で bit 一致することが、判定を変えていない主な回帰の錠になる。
/// 戻り値は `([m1, m2, s1, s2, den, dot, cos] の f64, 同 f32, ap ++ bp の f64, 同 f32)`。
#[allow(clippy::type_complexity)]
fn cos_fwd_previous(
    a32: &[f32],
    b32: &[f32],
    inv: (f64, f64),
    eps: (f64, f64),
) -> ([f64; 7], [f32; 7], Vec<f64>, Vec<f32>) {
    let ap: Vec<f64> = a32.iter().map(|&v| f64::from(v) * inv.0).collect();
    let bp: Vec<f64> = b32.iter().map(|&v| f64::from(v) * inv.1).collect();
    let m1: f64 = ap.iter().map(|v| v * v).sum::<f64>() + eps.0;
    let m2: f64 = bp.iter().map(|v| v * v).sum::<f64>() + eps.1;
    let (s1, s2) = (m1.sqrt(), m2.sqrt());
    let den = s1 * s2;
    let dot: f64 = ap.iter().zip(&bp).map(|(a, b)| a * b).sum();
    let cos = dot / den;
    let (r1f, r2f, e1f, e2f) = (inv.0 as f32, inv.1 as f32, eps.0 as f32, eps.1 as f32);
    let ap32: Vec<f32> = a32.iter().map(|&v| v * r1f).collect();
    let bp32: Vec<f32> = b32.iter().map(|&v| v * r2f).collect();
    let sum32 = |it: &mut dyn Iterator<Item = f32>| it.fold(0f32, |a, x| a + x);
    let m1_32 = sum32(&mut ap32.iter().map(|&v| v * v)) + e1f;
    let m2_32 = sum32(&mut bp32.iter().map(|&v| v * v)) + e2f;
    let (s1_32, s2_32) = (m1_32.sqrt(), m2_32.sqrt());
    let den32 = s1_32 * s2_32;
    let dot32 = sum32(&mut ap32.iter().zip(&bp32).map(|(&a, &b)| a * b));
    let cos32 = dot32 / den32;
    (
        [m1, m2, s1, s2, den, dot, cos],
        [m1_32, m2_32, s1_32, s2_32, den32, dot32, cos32],
        ap.into_iter().chain(bp).collect(),
        ap32.into_iter().chain(bp32).collect(),
    )
}

#[test]
fn cosine_fwd_generic_matches_previous_formulas() {
    let mut inputs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for c in cos_cases() {
        if c.d <= 512 && c.exp == 0 && c.y > 0.0 {
            inputs.push(c.ab());
        }
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for d in [1usize, 2, 3, 7, 16, 33] {
        for amp in [1e-30f32, 1e-8, 1e-3, 1.0, 1e3, 1e30] {
            let a: Vec<f32> = (0..d).map(|_| amp * rng.signed()).collect();
            let b: Vec<f32> = a.iter().map(|&x| x + 1e-3 * amp * rng.signed()).collect();
            // NaN・0 の行も通す。
            let mut an = a.clone();
            an[0] = f32::NAN;
            inputs.push((an, a.clone()));
            inputs.push((vec![0.0; d], a.clone()));
            inputs.push((a, b));
        }
    }
    for (a, b) in &inputs {
        let (inv, eps) = cos_scales(a, b);
        let (r64, r32, v64, v32) = cos_fwd_previous(a, b, inv, eps);
        let f64f = cos_fwd::<f64>(a, b, inv, eps);
        let f32f = cos_fwd::<f32>(a, b, inv, eps);
        let got64 = [
            f64f.m1, f64f.m2, f64f.s1, f64f.s2, f64f.den, f64f.dot, f64f.cos,
        ];
        let got32 = [
            f32f.m1, f32f.m2, f32f.s1, f32f.s2, f32f.den, f32f.dot, f32f.cos,
        ];
        for k in 0..7 {
            assert_eq!(f64_bits(got64[k]), f64_bits(r64[k]), "f64 中間値 {k}");
            assert_eq!(f32_bits(got32[k]), f32_bits(r32[k]), "f32 中間値 {k}");
        }
        let g64: Vec<u64> = f64f
            .ap
            .iter()
            .chain(&f64f.bp)
            .map(|&v| f64_bits(v))
            .collect();
        let g32: Vec<u32> = f32f
            .ap
            .iter()
            .chain(&f32f.bp)
            .map(|&v| f32_bits(v))
            .collect();
        assert_eq!(g64, v64.iter().map(|&v| f64_bits(v)).collect::<Vec<_>>());
        assert_eq!(g32, v32.iter().map(|&v| f32_bits(v)).collect::<Vec<_>>());
    }
    assert!(inputs.len() > 100, "入力が少なすぎる: {}", inputs.len());
}

#[test]
fn cosine_f32_model_stays_close_to_hand_built_structure() {
    // 受理される小さな入力では、`cos_rev::<f32>` の写しが手組みの構造（実際の eager 逆伝播）と REQ-2 の範囲で
    // 一致する。縮約の順序の違いで bit 一致しない要素は許す（bit 一致は求めない。soundness は別テストが正）。
    let mut worst = 0f64;
    for (delta, y, exp) in [(1e-2f32, 1.0f32, 0), (1.0, -1.0, 7), (1e-4, 1.0, 3)] {
        let c = CosCase {
            d: 8,
            delta,
            y,
            exp,
        };
        let (a, b) = c.ab();
        let (inv, eps) = cos_scales(&a, &b);
        let sd: Vec<usize> = (0..c.d).collect();
        let hand = c.hand_rows(&sd);
        let g = f64::from(pow2(exp));
        for (j, rows) in &hand {
            let model = cosine_model_rows_f32((&a, &b), inv, eps, (g, None), c.cf(), *j);
            for side in 0..2 {
                let (oi, ti) = if side == 0 { (0, 1) } else { (1, 0) };
                for i in 0..c.d {
                    for (h, m) in [
                        (rows[side][oi][i], model[side].own[i]),
                        (rows[side][ti][i], model[side].other[i]),
                    ] {
                        worst = worst.max(req2_ratio(f64::from(m), f64::from(h)));
                    }
                }
            }
        }
    }
    assert!(worst < 1.0, "f32 の写しが手組みから外れすぎ: {worst}");
}

/// 掃引の統計（空振りでないことの確認用）。
#[derive(Debug, Default)]
struct SweepStats {
    checked: usize,
    nonzero: usize,
    tight: usize,
    worst: f64,
}

/// 1 行 `(a, b)` について、手組みの構造の f32 の二階（実際の eager 逆伝播）と `cos_rev::<f32>` の双方が
/// `|実測 − Lin.v| ≤ Lin.bound()·(1 + 16·N·u)` を満たすことを確かめる（非有限の要素は除く）。`mean_n` は Mean の分母。
fn lin_sweep_row(c: &CosCase, a: &[f32], b: &[f32], mean_n: Option<usize>, st: &mut SweepStats) {
    use super::super::lin::LIN_U;
    let d = c.d;
    let (inv, eps) = cos_scales(a, b);
    let sd = seeds(d, &[]);
    let hand = c.hand_rows_red(a.to_vec(), b.to_vec(), &sd, mean_n);
    let up = (f64::from(pow2(c.exp)), mean_n);
    let lin = cosine_lin_rows((a, b), inv, eps, up, c.cf(), &sd);
    for (k, (j, rows)) in hand.iter().enumerate() {
        let model = cosine_model_rows_f32((a, b), inv, eps, up, c.cf(), *j);
        for side in 0..2 {
            let l = &lin[k][side];
            let slack = 1.0 + 16.0 * l.events as f64 * LIN_U;
            let (oi, ti) = if side == 0 { (0, 1) } else { (1, 0) };
            let mut check = |actual: f32, lv: (f64, f64)| {
                let actual = f64::from(actual);
                if !actual.is_finite() || !lv.0.is_finite() || !lv.1.is_finite() {
                    return;
                }
                let dev = (actual - lv.0).abs();
                assert!(
                    dev <= lv.1 * slack,
                    "{c:?} mean={mean_n:?} side={side} seed={j}: dev={dev:e} bound={:e}",
                    lv.1
                );
                st.checked += 1;
                if dev > 0.0 {
                    st.nonzero += 1;
                }
                if lv.1 > 0.0 {
                    st.worst = st.worst.max(dev / lv.1);
                    if dev > 0.1 * lv.1 {
                        st.tight += 1;
                    }
                }
            };
            for i in 0..d {
                check(rows[side][oi][i], l.own[i]);
                check(model[side].own[i], l.own[i]);
                check(rows[side][ti][i], l.other[i]);
                check(model[side].other[i], l.other[i]);
            }
            check(rows[side][2][0], l.g);
            check(model[side].g, l.g);
        }
    }
}

#[test]
fn lin_bound_covers_f32_deviation_for_cosine_sweep() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let mut st = SweepStats::default();
    for d in [1usize, 2, 3, 8, 16, 32] {
        for amp in [1e-3f32, 1.0, 1e3] {
            for delta in [0.0f32, 1e-6, 1e-4, 1e-2, 1.0] {
                for y in [1.0f32, -1.0] {
                    for exp in [0i32, 13, 20] {
                        let c = CosCase { d, delta, y, exp };
                        let a: Vec<f32> = (0..d).map(|_| amp * rng.signed()).collect();
                        let b: Vec<f32> =
                            a.iter().map(|&x| x + delta * amp * rng.signed()).collect();
                        // Sum と、Mean（分母 3・7。`s_b = g/n` の丸めと `g` への Div の VJP を含む）。
                        lin_sweep_row(&c, &a, &b, None, &mut st);
                        if exp == 13 {
                            lin_sweep_row(&c, &a, &b, Some(3), &mut st);
                            lin_sweep_row(&c, &a, &b, Some(7), &mut st);
                        }
                    }
                }
            }
        }
    }
    assert!(
        st.checked > 1000,
        "検査した要素が少なすぎる: {}",
        st.checked
    );
    assert!(
        st.nonzero > 0,
        "実測の誤差が全て 0 では soundness を確かめていない"
    );
    assert!(st.tight > 0, "上界が緩すぎて空振り: tight={}", st.tight);
    eprintln!(
        "LINCOS checked={} nonzero={} tight={} worst={:.4}",
        st.checked, st.nonzero, st.tight, st.worst
    );
}

/// 微小成分が行スケールで非正規数・0 へ落ちる行（`flag` が立つ行）の soundness。`a = [1, 微小…]`、`b` は平行に近い。
#[test]
fn lin_bound_covers_f32_deviation_for_cosine_underflow_rows() {
    let mut st = SweepStats::default();
    for d in [2usize, 4, 8] {
        for tiny in [1e-20f32, 1e-30, 1e-38, 1e-42] {
            for y in [1.0f32, -1.0] {
                let c = CosCase {
                    d,
                    delta: 0.0,
                    y,
                    exp: 13,
                };
                let a: Vec<f32> = (0..d).map(|i| if i == 0 { 1.0 } else { tiny }).collect();
                let b: Vec<f32> = a.iter().map(|&x| x * 1.0000001).collect();
                lin_sweep_row(&c, &a, &b, None, &mut st);
                lin_sweep_row(&c, &a, &b, Some(3), &mut st);
            }
        }
    }
    assert!(st.checked > 100, "検査した要素が少なすぎる: {}", st.checked);
}

// =====================================================================================
// #4091: Cosine の判定を `Lin` の上界へ切り替える（実測と固定）
// =====================================================================================

/// 行 `(a, b)` を `Lin` の判定（[`CosLinState`]）で全 seed 走査したときの要約。
#[derive(Debug, Default)]
struct LinScan {
    /// 走査した要素数（g・同じ側・相手側の合計）。
    checked: usize,
    /// `entry_check_lin` が拒否する要素数。
    rejected: usize,
    /// 判定時の事象数の最大。
    max_events: usize,
    /// forward だけの事象数。
    fwd_events: usize,
    /// 最大の `err / 閾値`（1 以上で拒否）。
    max_margin: f64,
    /// 走査にかかった秒数。
    secs: f64,
}

/// `seed_list` の seed について、`Lin` の判定を全要素（g・同じ側・相手側）で評価する（計測用。判定と同じ
/// [`CosLinState`] を使い、`entry_check_lin` と同じ式で `err/閾値` も出す）。
fn lin_scan(
    c: &CosCase,
    rows: (&[f32], &[f32]),
    mean_n: Option<usize>,
    seed_list: &[usize],
) -> LinScan {
    use super::super::lin::{LIN_U, lin_events, lin_truncate};
    let (inv, eps) = cos_scales(rows.0, rows.1);
    let ctx = CosLinCtx {
        g: f64::from(pow2(c.exp)),
        mean_n,
        cf: c.cf(),
    };
    let t0 = std::time::Instant::now();
    let mut st = CosLinState::new(rows, inv, eps, ctx);
    let mut out = LinScan {
        fwd_events: lin_events(),
        ..LinScan::default()
    };
    for &j in seed_list {
        for side in 0..2 {
            let (mark, sa, sb) = st.seed(side, j);
            let (r_own, r_other) = st.r[side];
            let mut visit = |l: Lin, s: f32, tag: &str, ii: usize| {
                let slack = 16.0 * lin_events() as f64 * LIN_U;
                let err = (f64::from(s) - l.v).abs() + l.bound() * (1.0 + slack);
                let thr = ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * l.v.abs());
                out.checked += 1;
                out.max_events = out.max_events.max(lin_events());
                let m = err / thr;
                out.max_margin = out.max_margin.max(m);
                if slack >= 0.01 || m >= 1.0 {
                    out.rejected += 1;
                    if std::env::var("LINCOS_VERBOSE").is_ok() && out.rejected <= 12 {
                        eprintln!(
                            "LINCOSREJ side={side} j={j} i={ii} {tag} v={:e} s={:e} bound={:e} thr={thr:e} margin={m:.3}",
                            l.v,
                            s,
                            l.bound()
                        );
                    }
                }
            };
            visit(sa.d_g, sb.d_g, "g", 0);
            for i in 0..c.d {
                let l = cos_rev_own(&st.lin[side], &sa, i, r_own);
                let s = cos_rev_own(&st.f32m[side], &sb, i, r_own);
                visit(l, s, "own", i);
                lin_truncate(mark);
                let l = cos_rev_other(&st.lin[side], &sa, i, r_other);
                let s = cos_rev_other(&st.f32m[side], &sb, i, r_other);
                visit(l, s, "oth", i);
                lin_truncate(mark);
            }
        }
    }
    out.secs = t0.elapsed().as_secs_f64();
    out
}

/// `LINCOSM`: 候補と d の掃引で、`Lin` の判定の事象数・全要素の余裕・1 行の判定時間（全 seed に換算）を出す。
/// `cargo test -p fandhe-ai-autodiff --release --lib cosine_lin_candidates_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn cosine_lin_candidates_measurement() {
    let mut cases = cosine_followup_candidates();
    for d in [64usize, 256, 512, 1024, 1536, 2048] {
        cases.push(CosCase {
            d,
            delta: 1e-2,
            y: 1.0,
            exp: 0,
        });
    }
    for c in cases {
        let (a, b) = c.ab();
        let sd = seeds(c.d, &[c.d / 2]);
        let r = lin_scan(&c, (&a, &b), None, &sd);
        let per_seed = r.secs / sd.len() as f64;
        eprintln!(
            "LINCOSM d={} delta={} y={} exp={} fwd_events={} max_events={} 16Nu={:.4} rejected={}/{} max_margin={:.3e} secs_per_seed={:.4e} row_secs_est={:.3} cost_units_est={:.3e}",
            c.d,
            c.delta,
            c.y,
            c.exp,
            r.fwd_events,
            r.max_events,
            16.0 * r.max_events as f64 / 16_777_216.0,
            r.rejected,
            r.checked,
            r.max_margin,
            per_seed,
            per_seed * c.d as f64,
            per_seed * c.d as f64 / 3.45e-9
        );
    }
}
/// 旧判定（`Lin` を使わない）の公開経路の verdict。
fn public_old(c: &CosCase) -> Verdict {
    probe::FORCE_OLD.with(|f| f.set(true));
    let v = c.public().0;
    probe::FORCE_OLD.with(|f| f.set(false));
    v
}

/// §82.2 の 168 行に `d ∈ {128, 256}` を足した行列。
fn cos_switch_cases() -> Vec<CosCase> {
    let mut v = cos_cases();
    for d in [128usize, 256] {
        for delta in [0.0f32, 1e-2, 1e-4, 1e-6] {
            for y in [1.0f32, -1.0] {
                for exp in [0i32, 13, 20] {
                    v.push(CosCase { d, delta, y, exp });
                }
            }
        }
    }
    v
}

/// `LINCOSSW`: 旧判定と新判定の受理・拒否を並べる。「旧受理 → 新拒否」は 0 件（either 受理の構造的な保証）、
/// 「旧拒否 → 新受理」の行は REQ-2 比 < 1 であることを assert する。
/// `cargo test -p fandhe-ai-autodiff --release --lib cosine_lin_switch_rejection_delta_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn cosine_lin_switch_rejection_delta_measurement() {
    let (mut total, mut old_rej, mut new_rej, mut flipped_to_accept) =
        (0usize, 0usize, 0usize, 0usize);
    for c in cos_switch_cases() {
        let old = public_old(&c);
        let new = c.public().0;
        total += 1;
        let (o, n) = (
            matches!(old, Verdict::Accept),
            matches!(new, Verdict::Accept),
        );
        old_rej += usize::from(!o);
        new_rej += usize::from(!n);
        assert!(!(o && !n), "旧受理 → 新拒否が出た: {c:?}");
        if !o && n {
            flipped_to_accept += 1;
            let m = c.measure();
            let (x, g) = (m.x.expect("x"), m.g.expect("g"));
            eprintln!(
                "LINCOSSW 旧拒否→新受理 d={} delta={} y={} exp={} REQ2比 x={:.3e} g={:.3e} 参照不確かさ x={:.1e}",
                c.d, c.delta, c.y, c.exp, x.ratio, g.ratio, x.unc
            );
            assert!(
                x.ratio < 1.0 && g.ratio < 1.0,
                "新たに受理した行が REQ-2 を外す: {c:?}"
            );
        }
    }
    eprintln!(
        "LINCOSSW total={total} old_rejected={old_rej} new_rejected={new_rej} flipped_to_accept={flipped_to_accept}"
    );
}
/// 判定時の事象数が `cos_lin_events_max(d)` に収まる（Sum・Mean、候補に近い入力と平行な行）。`cos_use_lin` の
/// 余裕係数の条件（`16·事象数·u < 0.01`）がこの上界に依るので、実測が上界を超えたらここで落ちる。
#[test]
fn cos_lin_events_stay_within_bound() {
    let mut worst_fixed = 0usize;
    for d in [1usize, 2, 8, 64, 256] {
        for (delta, amp) in [(1e-2f32, 1.0f32), (0.0, 1.0), (1.0, 1e3), (1e-4, 1e-3)] {
            for mean_n in [None, Some(5)] {
                let c = CosCase {
                    d,
                    delta,
                    y: 1.0,
                    exp: 7,
                };
                let a: Vec<f32> = pattern(d, 0.0).iter().map(|x| x * amp).collect();
                let q = pattern(d, 3.0);
                let b: Vec<f32> = a.iter().zip(&q).map(|(a, q)| a + delta * amp * q).collect();
                let sd = seeds(d, &[d / 2]);
                let r = lin_scan(&c, (&a, &b), mean_n, &sd);
                assert!(
                    r.max_events <= cos_lin_events_max(d),
                    "d={d} delta={delta} mean={mean_n:?}: 事象数 {} が上界 {} 超",
                    r.max_events,
                    cos_lin_events_max(d)
                );
                worst_fixed = worst_fixed.max(r.max_events.saturating_sub(5 * d));
            }
        }
    }
    assert!(
        worst_fixed <= COS_LIN_EVENTS_FIXED,
        "定数分 {worst_fixed} が COS_LIN_EVENTS_FIXED を超える"
    );
}

/// `cos_use_lin` の境界: 走査量の上限（`LIN_BUDGET`）と、余裕係数の上限（`16·事象数·u < 0.01`）の両方。
#[test]
fn cos_use_lin_boundaries() {
    use super::super::lin::{LIN_BUDGET, LIN_U};
    // 行数 1 で走査量が収まる最大の d。
    let dmax = (1..2000usize)
        .take_while(|&d| cos_lin_cost(d) <= LIN_BUDGET)
        .last()
        .unwrap();
    assert!(cos_use_lin(1, dmax) && !cos_use_lin(1, dmax + 1));
    // 行数が増えると走査量が増える（テンソル単位で選ぶ）。
    assert!(cos_use_lin(1, 128) && !cos_use_lin(10_000, 128));
    // 走査量の上限が事象数の上限より先に効く（d ≤ dmax なら余裕係数は常に満たす）。
    assert!(16.0 * cos_lin_events_max(dmax) as f64 * LIN_U < 0.01);
    // 余裕係数を満たさない d は走査量が足りていても選ばない（独立の条件であること）。
    let d_slack = (1..100_000usize)
        .find(|&d| 16.0 * cos_lin_events_max(d) as f64 * LIN_U >= 0.01)
        .unwrap();
    assert!(d_slack > dmax);
}

/// 従来の見積りが `coeff·m` の項だけで拒否していた行を、`Lin` の上界が受理に変える（d=128・上流 `2^20`）。
/// 受理に変えた行の二階が f64 の閉じた式と REQ-2 で一致すること（REQ-2 比 < 1。参照の不確かさ < 5%）。
#[test]
fn cosine_lin_accepts_rows_rejected_by_coeff_bound_and_matches_f64() {
    for y in [1.0f32, -1.0] {
        let c = CosCase {
            d: 128,
            delta: 1e-2,
            y,
            exp: 20,
        };
        match public_old(&c) {
            Verdict::Cancel { cls, .. } => assert_eq!(cls, "coeff·m のみ"),
            v => panic!("{c:?}: 旧判定は coeff·m で拒否するはず: {v:?}"),
        }
        let m = c.measure();
        assert_eq!(
            m.verdict,
            Verdict::Accept,
            "{c:?}: Lin の上界で受理に変わるはず"
        );
        for (label, acc) in [("x", m.x), ("g", m.g)] {
            let a = acc.expect("計測");
            assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
            assert!(a.unc < 0.05, "{c:?}: {label} の参照の不確かさ {:e}", a.unc);
        }
    }
}

/// 候補（`cosine_followup_candidates()`）を `Lin` で全要素走査すると、`Lin` の上界が閾値を超える要素が残る
/// （d=512 は走査量の上限でも本番では `Lin` を使わない）。受理に変えられない理由を固定する（設計 §82.8）。
/// `cargo test -p fandhe-ai-autodiff --release --lib cosine_followup_candidates_exceed_lin_bound -- --ignored`
#[test]
#[ignore = "d=2048・4096 の全要素走査は重い。--release で実行する"]
fn cosine_followup_candidates_exceed_lin_bound() {
    for c in cosine_followup_candidates() {
        let (a, b) = c.ab();
        let sd = seeds(c.d, &[c.d / 2]);
        let r = lin_scan(&c, (&a, &b), None, &sd);
        assert!(
            r.rejected > 0 && r.max_margin >= 1.0,
            "{c:?}: Lin の上界でも拒否されるはず: {r:?}"
        );
        assert!(!cos_use_lin(1, c.d), "{c:?}: 走査量の上限で Lin を使わない");
    }
}
// =====================================================================================
// #4093: CE・CEWO の相殺判定の上界を `Lin` で並行計算する（判定は変えない）
// =====================================================================================

/// 総称化の前に `cewo_cancel_check` が書いていた forward の式（逐語のコピー。前例: `cos_fwd_previous`）。
/// 戻り値は `(f64 の e, total, ys, f32 の e32, total32, y32)`。`cewo_fwd` が全中間値で bit 一致することが、
/// 判定を変えていない主な回帰の錠になる。
#[allow(clippy::type_complexity)]
fn cewo_fwd_previous(row32: &[f32]) -> (Vec<f64>, f64, Vec<f64>, Vec<f32>, f32, Vec<f64>) {
    let row: Vec<f64> = row32.iter().map(|&v| f64::from(v)).collect();
    let mx = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let e: Vec<f64> = row.iter().map(|&v| (v - mx).exp()).collect();
    let total: f64 = e.iter().sum();
    let mx32 = row32.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let e32: Vec<f32> = row32.iter().map(|&v| (v - mx32).exp()).collect();
    let total32 = sum32(e32.iter().copied());
    let y32: Vec<f64> = e32.iter().map(|&v| f64::from(v / total32)).collect();
    let ys: Vec<f64> = e.iter().map(|v| v / total).collect();
    (e, total, ys, e32, total32, y32)
}

#[test]
fn cewo_fwd_generic_matches_previous_formulas() {
    let mut inputs: Vec<Vec<f32>> = ce_cases().iter().map(CeCase::logits).collect();
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    for c in [1usize, 2, 3, 7, 16, 33] {
        for amp in [1e-30f32, 1e-3, 1.0, 30.0, 1e3] {
            let x: Vec<f32> = (0..c).map(|_| amp * rng.signed()).collect();
            inputs.push(x.clone());
            inputs.push(vec![amp; c]);
            // 余裕 0/20/40/90/120（f32 の exp が非正規数や 0 へ underflow する）。
            for margin in [0.0f32, 20.0, 40.0, 90.0, 120.0] {
                let mut y = x.clone();
                y[0] += margin;
                inputs.push(y);
            }
            // ±inf（既定 CE は `reject_inf` を通らない）を含む行。NaN の行は判定前に return するが関数は通す。
            for special in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
                let mut y = x.clone();
                y[c - 1] = special;
                inputs.push(y);
            }
        }
    }
    for row in &inputs {
        let (e, total, ys, e32, total32, y32) = cewo_fwd_previous(row);
        let f64f = cewo_fwd::<f64>(row);
        let f32f = cewo_fwd::<f32>(row);
        assert_eq!(f64_bits(f64f.total), f64_bits(total), "f64 total {row:?}");
        assert_eq!(f32_bits(f32f.total), f32_bits(total32), "f32 total {row:?}");
        for k in 0..row.len() {
            assert_eq!(f64_bits(f64f.e[k]), f64_bits(e[k]), "f64 e[{k}] {row:?}");
            assert_eq!(f64_bits(f64f.y[k]), f64_bits(ys[k]), "f64 y[{k}] {row:?}");
            assert_eq!(f32_bits(f32f.e[k]), f32_bits(e32[k]), "f32 e[{k}] {row:?}");
            assert_eq!(
                f64_bits(f64::from(f32f.y[k])),
                f64_bits(y32[k]),
                "f32 y[{k}] {row:?}"
            );
        }
    }
    assert!(inputs.len() > 200, "入力が少なすぎる: {}", inputs.len());
}

/// 任意の `A`・`B`・`W`・上流で、本番と同じ演算順の構造（既定 CE は `create_graph.rs` の腕、CEWO は `cewo_cgrad`）を
/// 子テープへ組み、seed ごとの `(x への二階の行, g への二階)` を返す（`CeCase::hand_rows` の一般化。既存は変えない）。
fn ce_hand_general(
    logits: &[f32],
    default: bool,
    coef: (f64, &[f64]),
    up: (Reduction, Option<f64>, i32),
    seed_list: &[usize],
) -> Vec<(usize, Vec<f32>, f32)> {
    let c = logits.len();
    let shape = [1usize, c];
    let (red, w, exp) = up;
    let child = Tape::new_with_ops(test_ops());
    let x_m = child.var(&tt(logits.to_vec(), &shape));
    let g = child.var(&Tensor::scalar(pow2(exp)));
    let dx = if default {
        let s_b = super::super::loss_scale_bcast(&child, &g, red, false, 1, &shape).unwrap();
        let mut oh = vec![0f32; c];
        for (o, b) in oh.iter_mut().zip(coef.1) {
            *o = *b as f32;
        }
        let onehot = child.var_no_grad(&tt(oh, &shape));
        x_m.softmax(1)
            .unwrap()
            .sub(&onehot)
            .unwrap()
            .mul(&s_b)
            .unwrap()
    } else {
        let s_b = match red {
            Reduction::Mean => g
                .div(&scalar(&child, w.unwrap() as f32))
                .unwrap()
                .broadcast_to(&shape)
                .unwrap(),
            Reduction::Sum => g.broadcast_to(&shape).unwrap(),
        };
        let a_c = const_vec(&child, vec![coef.0 as f32; c], &shape).unwrap();
        let b_c = const_vec(&child, coef.1.iter().map(|&b| b as f32).collect(), &shape).unwrap();
        let valid = mask(vec![true; c], &shape).unwrap();
        let zeros = zeros_like(&child, &shape).unwrap();
        let x_safe = Var::where_cond(&valid, &x_m, &zeros).unwrap();
        let sm = x_safe.softmax(1).unwrap();
        let scaled = s_b.mul(&a_c.mul(&sm).unwrap().sub(&b_c).unwrap()).unwrap();
        Var::where_cond(&valid, &scaled, &zeros).unwrap()
    };
    seed_list
        .iter()
        .map(|&j| {
            let r = child_rows(&child, &dx, j, &[&x_m, &g])
                .unwrap_or_else(|| panic!("CE: seed {j} の子テープの逆伝播が失敗"));
            (j, r[0].clone(), r[1][0])
        })
        .collect()
}

/// 1 行について、手組みの構造の f32 の二階（実際の eager 逆伝播）と `cewo_rev::<f32>` の双方が
/// `|実測 − Lin.v| ≤ Lin.bound()·(1 + 16·N·u)` を満たすことを確かめる（非有限の要素は除く）。
fn ce_sweep_row(
    logits: &[f32],
    default: bool,
    coef: (f64, &[f64]),
    up: (Reduction, Option<f64>, i32),
    st: &mut SweepStats,
) {
    use super::super::lin::LIN_U;
    let c = logits.len();
    let sd = seeds(c, &[CE_TARGET, CE_TARGET + 1, c / 2]);
    let hand = ce_hand_general(logits, default, coef, up, &sd);
    let upv = (
        f64::from(pow2(up.2)),
        if up.0 == Reduction::Mean { up.1 } else { None },
    );
    let lin = cewo_lin_rows(logits, coef, upv, &sd);
    for (k, (j, row, gj)) in hand.iter().enumerate() {
        let model = cewo_model_rows_f32(logits, coef, upv, *j);
        let l = &lin[k];
        let slack = 1.0 + 16.0 * l.events as f64 * LIN_U;
        let mut check = |actual: f32, lv: (f64, f64)| {
            let actual = f64::from(actual);
            if !actual.is_finite() || !lv.0.is_finite() || !lv.1.is_finite() {
                return;
            }
            let dev = (actual - lv.0).abs();
            assert!(
                dev <= lv.1 * slack,
                "c={c} default={default} red={:?} w={:?} seed={j}: dev={dev:e} bound={:e}",
                up.0,
                up.1,
                lv.1
            );
            st.checked += 1;
            if dev > 0.0 {
                st.nonzero += 1;
            }
            if lv.1 > 0.0 {
                st.worst = st.worst.max(dev / lv.1);
                if dev > 0.1 * lv.1 {
                    st.tight += 1;
                }
            }
        };
        for i in 0..c {
            check(row[i], l.x[i]);
            check(model.x[i], l.x[i]);
        }
        check(*gj, l.g);
        check(model.g, l.g);
    }
}

/// 掃引の 1 入力ぶん: 既定 CE（`A = 1`・`B = onehot`）と、乱数の `A`・`B` の CEWO を Sum／Mean で通す。
fn ce_sweep_input(logits: &[f32], exp: i32, rng: &mut Rng, st: &mut SweepStats) {
    let c = logits.len();
    let mut onehot = vec![0f64; c];
    onehot[CE_TARGET] = 1.0;
    let a_s = 0.3 + rng.unit();
    let b: Vec<f64> = (0..c)
        .map(|j| {
            let h = if j == CE_TARGET { 0.9 } else { 0.0 };
            h + 0.1 * rng.unit() / c as f64
        })
        .collect();
    let sum = (Reduction::Sum, None, exp);
    ce_sweep_row(logits, true, (1.0, &onehot), sum, st);
    ce_sweep_row(logits, false, (a_s, &b), sum, st);
    if exp == 13 {
        let mean1 = (Reduction::Mean, Some(1.0), exp);
        ce_sweep_row(logits, true, (1.0, &onehot), mean1, st);
        for w in [1.0f64, 3.0, 7.0, 0.37] {
            let mean = (Reduction::Mean, Some(w), exp);
            ce_sweep_row(logits, false, (a_s, &b), mean, st);
        }
    }
}

#[test]
fn lin_bound_covers_f32_deviation_for_ce_sweep() {
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let mut st = SweepStats::default();
    for c in [2usize, 3, 10, 33, 64] {
        for amp in [1e-3f32, 1.0, 30.0] {
            for margin in [0.0f32, 5.0, 10.0, 20.0, 40.0] {
                for exp in [0i32, 13, 20] {
                    let mut x: Vec<f32> = (0..c).map(|_| amp * rng.signed()).collect();
                    x[CE_TARGET] += margin;
                    ce_sweep_input(&x, exp, &mut rng, &mut st);
                }
            }
        }
    }
    assert!(
        st.checked > 1000,
        "検査した要素が少なすぎる: {}",
        st.checked
    );
    assert!(
        st.nonzero > 0,
        "実測の誤差が全て 0 では soundness を確かめていない"
    );
    assert!(st.tight > 0, "上界が緩すぎて空振り: tight={}", st.tight);
    eprintln!(
        "LINCE checked={} nonzero={} tight={} worst={:.4}",
        st.checked, st.nonzero, st.tight, st.worst
    );
}

/// 小さな確率が f32 の非正規数・0 へ落ちる行（`flag` が立つ行）の soundness。余裕 80〜120。
#[test]
fn lin_bound_covers_f32_deviation_for_ce_underflow_rows() {
    let mut rng = Rng(0x0f0f_1234_5678_9abc);
    let mut st = SweepStats::default();
    for c in [2usize, 5, 16] {
        for margin in [80.0f32, 90.0, 100.0, 110.0, 120.0] {
            let mut x: Vec<f32> = (0..c).map(|_| rng.signed()).collect();
            x[CE_TARGET] += margin;
            ce_sweep_input(&x, 13, &mut rng, &mut st);
        }
    }
    assert!(st.checked > 100, "検査した要素が少なすぎる: {}", st.checked);
}

#[test]
fn ce_f32_model_stays_close_to_hand_built_structure() {
    // 受理される小さな入力で、`cewo_rev::<f32>` の写し（Softmax の VJP の混合精度は f32 の丸めで近似）が
    // 手組みの構造と REQ-2 の範囲で一致する。bit 一致は求めない（soundness は別テストが正）。
    let mut worst = 0f64;
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        for exp in [0, 7] {
            let c = CeCase {
                kind,
                c: 10,
                margin: 5.0,
                exp,
            };
            let default = kind == CeKind::Default;
            let (a_s, b) = if default {
                let mut oh = vec![0f64; c.c];
                oh[CE_TARGET] = 1.0;
                (1.0, oh)
            } else {
                c.ab()
            };
            let w = f64::from(c.weights()[CE_TARGET]);
            let (red, wv) = if default {
                (Reduction::Sum, None)
            } else {
                (
                    c.reduction(),
                    (c.reduction() == Reduction::Mean).then_some(w),
                )
            };
            let sd: Vec<usize> = (0..c.c).collect();
            let hand = ce_hand_general(&c.logits(), default, (a_s, &b), (red, wv, exp), &sd);
            let up = (f64::from(pow2(exp)), wv);
            for (j, row, gj) in &hand {
                let m = cewo_model_rows_f32(&c.logits(), (a_s, &b), up, *j);
                for i in 0..c.c {
                    worst = worst.max(req2_ratio(f64::from(m.x[i]), f64::from(row[i])));
                }
                worst = worst.max(req2_ratio(f64::from(m.g), f64::from(*gj)));
            }
        }
    }
    assert!(worst < 1.0, "f32 の写しが手組みから外れすぎ: {worst}");
}

/// `LINCEM`: 候補の CE の入力で、`Lin` の上界から見た事象数・`16·N·u`・最大の上界・1 seed の時間を出す。
/// `cargo test -p fandhe-ai-autodiff --release --lib ce_lin_candidates_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn ce_lin_candidates_measurement() {
    use super::super::lin::LIN_U;
    let mut cases = ce_followup_candidates();
    cases.push(CeCase {
        kind: CeKind::Default,
        c: 33,
        margin: 10.0,
        exp: 0,
    });
    for c in cases {
        let logits = c.logits();
        let (a_s, b) = if c.kind == CeKind::Default {
            let mut oh = vec![0f64; c.c];
            oh[CE_TARGET] = 1.0;
            (1.0, oh)
        } else {
            c.ab()
        };
        let sd = seeds(c.c, &[CE_TARGET, c.c / 2]);
        let up = (f64::from(pow2(c.exp)), None);
        let t = std::time::Instant::now();
        let lin = cewo_lin_rows(&logits, (a_s, &b), up, &sd);
        let secs = t.elapsed().as_secs_f64() / sd.len() as f64;
        let n = lin.iter().map(|l| l.events).max().unwrap_or(0);
        let finite = lin
            .iter()
            .all(|l| l.g.1.is_finite() && l.x.iter().all(|v| v.1.is_finite()));
        let max_bound = lin
            .iter()
            .flat_map(|l| l.x.iter().map(|v| v.1).chain([l.g.1]))
            .fold(0f64, f64::max);
        eprintln!(
            "LINCEM {} c={} margin={} exp={} N={} 16Nu={:.4} bounds_finite={} max_bound={:.3e} secs_per_seed={:.4e}",
            c.name(),
            c.c,
            c.margin,
            c.exp,
            n,
            16.0 * n as f64 * LIN_U,
            finite,
            max_bound,
            secs
        );
    }
}
// =====================================================================================
// #4094: CE・CEWO の判定を `Lin` の上界へ切り替える（either 受理。設計 §82.11）
// =====================================================================================

/// `CeCase` の `(A, B)`（既定 CE は `A = 1`・`B = onehot`）と、`Lin` 判定の入力（上流は `2^exp`、Mean の分母は `w_t`）。
fn ce_lin_inputs(c: &CeCase) -> (Vec<f32>, f64, Vec<f64>, CeLinCtx) {
    let (a_s, b) = if c.kind == CeKind::Default {
        let mut oh = vec![0f64; c.c];
        oh[CE_TARGET] = 1.0;
        (1.0, oh)
    } else {
        c.ab()
    };
    let w = (c.reduction() == Reduction::Mean).then(|| f64::from(c.weights()[CE_TARGET]));
    let ctx = CeLinCtx {
        g: f64::from(pow2(c.exp)),
        w,
        off: true,
    };
    (c.logits(), a_s, b, ctx)
}

/// `seed_list` の seed について、`Lin` の判定を `g` と x への二階（`off_pairs` なら全 `i`、そうでなければ対角のみ）で
/// 評価する（計測用。判定と同じ [`CeLinState`] を使い、`entry_check_lin` と同じ式で `err/閾値` も出す）。
fn ce_lin_scan(c: &CeCase, seed_list: &[usize], off_pairs: bool) -> LinScan {
    ce_lin_scan_timed(c, seed_list, off_pairs, false).0
}

/// [`ce_lin_scan`] と、そのうち forward（`CeLinState::new`）にかかった秒数。
fn ce_lin_scan_timed(
    c: &CeCase,
    seed_list: &[usize],
    off_pairs: bool,
    off_only: bool,
) -> (LinScan, f64) {
    use super::super::lin::{LIN_U, lin_events, lin_truncate};
    let (logits, a_s, b, ctx) = ce_lin_inputs(c);
    let t0 = std::time::Instant::now();
    let mut st = CeLinState::new(&logits, a_s, ctx);
    let fwd_secs = t0.elapsed().as_secs_f64();
    let mut out = LinScan {
        fwd_events: lin_events(),
        ..LinScan::default()
    };
    for &j in seed_list {
        let (mark, sa, sb) = st.seed(j, b[j]);
        let mut visit = |l: Lin, s: f32| {
            let slack = 16.0 * lin_events() as f64 * LIN_U;
            let err = (f64::from(s) - l.v).abs() + l.bound() * (1.0 + slack);
            let thr = ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * l.v.abs());
            out.checked += 1;
            out.max_events = out.max_events.max(lin_events());
            let m = err / thr;
            out.max_margin = out.max_margin.max(m);
            if slack >= 0.01 || m >= 1.0 {
                out.rejected += 1;
            }
        };
        if !off_only {
            visit(sa.g, sb.g);
        }
        let idx: Vec<usize> = if off_only {
            (0..c.c).filter(|&i| i != j).collect()
        } else if off_pairs {
            (0..c.c).collect()
        } else {
            vec![j]
        };
        for i in idx {
            let l = cewo_x(&st.lin, &sa, i, j);
            let s = cewo_x(&st.f32m, &sb, i, j);
            visit(l, s);
            lin_truncate(mark);
        }
    }
    out.secs = t0.elapsed().as_secs_f64();
    (out, fwd_secs)
}

/// 旧判定（`Lin` を使わない）の公開経路の verdict。
fn public_old_ce(c: &CeCase) -> Verdict {
    probe::FORCE_OLD.with(|f| f.set(true));
    let v = c.public().0;
    probe::FORCE_OLD.with(|f| f.set(false));
    v
}

/// `LINCECOST`: 見積り定数を決める計測。種別 × C で forward の事象数・時間、対角と `g` の要素あたりの時間、
/// 非対角の要素あたりの時間を `Rf` の 1 反復（3.45 ns。§80.5）相当に直して出す。
/// `cargo test -p fandhe-ai-autodiff --release --lib ce_lin_cost_profile -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn ce_lin_cost_profile() {
    const RF: f64 = 3.45e-9;
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        for c in [2usize, 33, 100, 256, 512, 1000, 2100] {
            let case = CeCase {
                kind,
                c,
                margin: 10.0,
                exp: 0,
            };
            let sd: Vec<usize> = (0..16).map(|k| (k * c / 16 + CE_TARGET) % c).collect();
            let mut best = [f64::MAX; 3];
            let mut maxev = 0;
            for _ in 0..3 {
                let (d, f) = ce_lin_scan_timed(&case, &sd, false, false);
                let (o, f2) = ce_lin_scan_timed(&case, &sd, false, true);
                best[0] = best[0].min(f.min(f2));
                best[1] = best[1].min((d.secs - f) / sd.len() as f64);
                best[2] = best[2].min((o.secs - f2) / sd.len() as f64);
                maxev = maxev.max(d.max_events).max(o.max_events);
            }
            let (fwd_secs, per_diag, per_off) = (best[0], best[1], best[2]);
            let both = LinScan {
                max_events: maxev,
                ..LinScan::default()
            };
            let diag = LinScan {
                fwd_events: ce_lin_events_max(c),
                ..LinScan::default()
            };
            let fwd_it = fwd_secs / RF;
            let diag_it = per_diag * c as f64 / RF;
            let off_it = per_off * c as f64 / RF;
            eprintln!(
                "LINCECOST {} c={c} fwd_events={} max_events={} 16Nu={:.4} fwd_iter={fwd_it:.3e} diag_iter(all seeds)={diag_it:.3e} off_iter(all seeds)={off_it:.3e} est_diag={} est_off={} ratio_diag={:.2} ratio_off={:.2}",
                case.name(),
                diag.fwd_events,
                both.max_events,
                16.0 * both.max_events as f64 / 16_777_216.0,
                ce_lin_cost_diag(c),
                ce_lin_cost_off(c),
                ce_lin_cost_diag(c) as f64 / (fwd_it + diag_it),
                ce_lin_cost_off(c) as f64 / off_it.max(1.0),
            );
        }
    }
}

/// 判定時の事象数が `ce_lin_events_max(c)` に収まる（Sum・Mean、確信度の高い行・低い行）。`ce_lin_ctx` の余裕係数の
/// 条件（`16·事象数·u < 0.01`）がこの上界に依るので、実測が上界を超えたらここで落ちる。
#[test]
fn ce_lin_events_stay_within_bound() {
    let mut worst_fixed = 0usize;
    let mut worst_per = 0f64;
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        for c in [2usize, 3, 10, 33, 100, 256] {
            for margin in [0.0f32, 10.0, 40.0] {
                let case = CeCase {
                    kind,
                    c,
                    margin,
                    exp: 7,
                };
                let sd = seeds(c, &[CE_TARGET, c / 2]);
                let r = ce_lin_scan(&case, &sd, true);
                assert!(
                    r.max_events <= ce_lin_events_max(c),
                    "{case:?}: 事象数 {} が上界 {} 超",
                    r.max_events,
                    ce_lin_events_max(c)
                );
                worst_fixed = worst_fixed.max(
                    r.max_events
                        .saturating_sub(CE_LIN_EVENTS_PER_CLASS.saturating_mul(c)),
                );
                worst_per = worst_per.max(r.max_events as f64 / c as f64);
            }
        }
    }
    assert!(
        worst_fixed <= CE_LIN_EVENTS_FIXED,
        "定数分 {worst_fixed} が CE_LIN_EVENTS_FIXED を超える"
    );
    eprintln!("LINCEEV worst_fixed={worst_fixed} worst_per_class={worst_per:.3}");
}

/// `ce_lin_ctx` の境界: 余裕係数の上限（`16·事象数·u < 0.01`）、対角・`g` の走査量の上限、非対角まで許す走査量の上限、
/// テンソル単位の選択（行数が増えると偽）。
#[test]
fn ce_use_lin_boundaries() {
    use super::super::lin::{LIN_BUDGET, LIN_U};
    let some = |n: usize, c: usize| ce_lin_ctx(1.0, None, n, c);
    let slack_ok = |c: usize| 16.0 * ce_lin_events_max(c) as f64 * LIN_U < 0.01;
    // 行数 1 で対角・g の走査量が収まる最大の C。
    let cmax = (1..100_000usize)
        .take_while(|&c| ce_lin_cost_diag(c) <= LIN_BUDGET && slack_ok(c))
        .last()
        .unwrap();
    assert!(some(1, cmax).is_some() && some(1, cmax + 1).is_none());
    // 非対角まで許す最大の C は対角・g だけの最大以下（別々の条件）。
    let omax = (1..=cmax)
        .take_while(|&c| ce_lin_cost_diag(c) + ce_lin_cost_off(c) <= LIN_BUDGET)
        .last()
        .unwrap();
    assert!(some(1, omax).is_some_and(|l| l.off));
    if omax < cmax {
        assert!(some(1, omax + 1).is_some_and(|l| !l.off));
    }
    // 行数が増えると走査量が増える（テンソル単位で選ぶ）。
    assert!(some(1, 100).is_some() && some(1_000_000, 100).is_none());
    // 余裕係数を満たさない C は走査量が足りていても選ばない（独立の条件であること）。
    let c_slack = (1..1_000_000usize).find(|&c| !slack_ok(c)).unwrap();
    assert!(some(1, c_slack).is_none());
    // 旧判定の強制（変異確認・比較用）。
    probe::FORCE_OLD.with(|f| f.set(true));
    assert!(some(1, 10).is_none());
    probe::FORCE_OLD.with(|f| f.set(false));
}

/// 従来の見積りが `coeff·m` の項だけで拒否していた行を、`Lin` の上界が受理に変える（C=100 の既定 CE と CEWO-weight）。
/// 受理に変えた行の二階が f64 の閉じた式と REQ-2 で一致すること（REQ-2 比 < 1。参照の不確かさ < 5%）。
#[test]
fn ce_lin_accepts_rows_rejected_by_coeff_bound_and_matches_f64() {
    for kind in [CeKind::Default, CeKind::Weighted] {
        let c = CeCase {
            kind,
            c: 100,
            margin: 10.0,
            exp: 0,
        };
        match public_old_ce(&c) {
            Verdict::Cancel { cls, .. } => assert_eq!(cls, "coeff·m のみ"),
            v => panic!("{c:?}: 旧判定は coeff·m で拒否するはず: {v:?}"),
        }
        let m = c.measure();
        assert_eq!(
            m.verdict,
            Verdict::Accept,
            "{c:?}: Lin の上界で受理に変わるはず"
        );
        for (label, acc) in [("x", m.x), ("g", m.g)] {
            let a = acc.expect("計測");
            assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
            assert!(a.unc < 0.05, "{c:?}: {label} の参照の不確かさ {:e}", a.unc);
        }
    }
}
/// §82.2 の 252 行に `C ∈ {33, 64, 256, 512}` を足した行列。
fn ce_switch_cases() -> Vec<CeCase> {
    let mut v = ce_cases();
    for kind in [CeKind::Default, CeKind::Weighted, CeKind::Smooth] {
        for c in [33usize, 64, 256, 512] {
            for margin in [5.0f32, 10.0, 20.0, 40.0] {
                for exp in [0i32, 13, 20] {
                    v.push(CeCase {
                        kind,
                        c,
                        margin,
                        exp,
                    });
                }
            }
        }
    }
    v
}

/// `LINCESW`: 旧判定と新判定の受理・拒否を並べる。「旧受理 → 新拒否」は 0 件（either 受理の構造的な保証）、
/// 「旧拒否 → 新受理」の行は REQ-2 比 < 1 であることを assert する。
/// `cargo test -p fandhe-ai-autodiff --release --lib ce_lin_switch_rejection_delta_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn ce_lin_switch_rejection_delta_measurement() {
    let (mut total, mut old_rej, mut new_rej, mut flipped_to_accept) =
        (0usize, 0usize, 0usize, 0usize);
    for c in ce_switch_cases() {
        let old = public_old_ce(&c);
        let new = c.public().0;
        total += 1;
        let (o, n) = (
            matches!(old, Verdict::Accept),
            matches!(new, Verdict::Accept),
        );
        old_rej += usize::from(!o);
        new_rej += usize::from(!n);
        assert!(!(o && !n), "旧受理 → 新拒否が出た: {c:?}");
        if !o && n {
            flipped_to_accept += 1;
            let m = c.measure();
            let (x, g) = (m.x.expect("x"), m.g.expect("g"));
            eprintln!(
                "LINCESW 旧拒否→新受理 {} c={} margin={} exp={} REQ2比 x={:.3e} g={:.3e} 参照不確かさ x={:.1e}",
                c.name(),
                c.c,
                c.margin,
                c.exp,
                x.ratio,
                g.ratio,
                x.unc
            );
            assert!(
                x.ratio < 1.0 && g.ratio < 1.0,
                "新たに受理した行が REQ-2 を外す: {c:?}"
            );
        }
    }
    eprintln!(
        "LINCESW total={total} old_rejected={old_rej} new_rejected={new_rej} flipped_to_accept={flipped_to_accept}"
    );
}

/// 4 サンプル・Mean の CEWO の損失（上流の係数は 1）。
fn multi_build<'t>(
    tape: &'t Tape,
    z: &Var<'t>,
    c: usize,
    opts: &CrossEntropyOptions,
) -> Result<Var<'t>, AutodiffError> {
    let tgt = Tensor::<i32>::new(vec![0, 1, 2, 3], &[4]).unwrap();
    let l = z
        .reshape(&[4, c])?
        .cross_entropy_loss_with(&tgt, 1, Reduction::Mean, opts)?;
    l.mul(&tape.var_no_grad(&Tensor::scalar(1.0f32)))?.sum(None)
}

/// 複数サンプル（n > 1・Mean）と `ignore_index` を含む行で、旧受理 → 新拒否が出ない（either 受理）。
/// 行ごとに確信度が違う 4 サンプルの CEWO（class_weight あり・Mean）を、ignore の有無で通す。
#[test]
fn ce_lin_multi_sample_and_ignore_never_flip_accept_to_reject() {
    let c = 100usize;
    let weights: Vec<f32> = (0..c)
        .map(|i| 0.5 + 0.4 * ((i as f32) * 0.731).sin())
        .collect();
    for ignore in [None, Some(2i32)] {
        for margins in [[10.0f32, 10.0, 10.0, 10.0], [0.0, 5.0, 10.0, 20.0]] {
            let mut z0 = Vec::new();
            for (r, m) in margins.iter().enumerate() {
                let mut row: Vec<f32> = pattern(c, r as f32).iter().map(|v| 0.1 * v).collect();
                row[r] += m;
                z0.extend(row);
            }
            let mut opts = CrossEntropyOptions::default().class_weight(tt(weights.clone(), &[c]));
            if let Some(i) = ignore {
                opts = opts.ignore_index(i);
            }
            probe::FORCE_OLD.with(|f| f.set(true));
            let old = run_public(z0.clone(), |tape, z| multi_build(tape, z, c, &opts)).0;
            probe::FORCE_OLD.with(|f| f.set(false));
            let new = run_public(z0, |tape, z| multi_build(tape, z, c, &opts)).0;
            assert!(
                !(old == Verdict::Accept && new != Verdict::Accept),
                "ignore={ignore:?} margins={margins:?}: 旧受理 → 新拒否 old={old:?} new={new:?}"
            );
            eprintln!("LINCEMULTI ignore={ignore:?} margins={margins:?} old={old:?} new={new:?}");
        }
    }
}

/// 候補（`ce_followup_candidates()`）ごとの旧の余裕・`Lin` の最大 `err/閾値`・本番の判定を出す。
/// `cargo test -p fandhe-ai-autodiff --release --lib ce_followup_verdict_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn ce_followup_verdict_measurement() {
    for c in ce_followup_candidates() {
        let old = public_old_ce(&c);
        let new = c.public().0;
        let sd = seeds(c.c, &[CE_TARGET, c.c / 2]);
        let scan = ce_lin_scan(&c, &sd, c.c <= 1000);
        let old_margin = match &old {
            Verdict::Cancel { margin, .. } => *margin,
            _ => f64::NAN,
        };
        eprintln!(
            "LINCECAND {} c={} old_margin={old_margin:.3e} lin_max_err_over_thr={:.3e} lin_rejected={}/{} N={} new={new:?}",
            c.name(),
            c.c,
            scan.max_margin,
            scan.rejected,
            scan.checked,
            scan.max_events
        );
    }
}

/// 受理に変わらなかった候補（C=1000 の CEWO-smooth・C=2100 の既定 CE）は、`Lin` で対角と `g` を全 seed 走査しても
/// 上界が閾値を超える要素が残る（判定を緩めていないことと、受理に変えられない理由の固定。設計 §82.11）。
/// 実際の f32 の二階は REQ-2 内（`loss_class_followup_candidates_are_req2_within_f64`）なので、`Lin` の和の上界
/// `γ_{C−1}·Σ|e|` が実カーネルの和の順序に依らない最悪値であることによる過剰な拒否。
#[test]
#[ignore = "C=2100 の全 seed 走査は重い。--release で実行する"]
fn ce_followup_candidates_exceed_lin_bound() {
    for c in ce_followup_candidates().into_iter().filter(|c| c.c > 100) {
        let sd: Vec<usize> = (0..c.c).collect();
        let r = ce_lin_scan(&c, &sd, false);
        assert!(
            r.rejected > 0 && r.max_margin >= 1.0,
            "{c:?}: Lin の上界でも拒否されるはず: {r:?}"
        );
        assert!(
            matches!(c.public().0, Verdict::Cancel { .. }),
            "{c:?}: 本番の判定も拒否のまま"
        );
    }
}
/// `LINCETIME`: 本番経路（`Lin` を使う最大の C と、走査量の上限ぎりぎりの行数）の判定時間。
/// `cargo test -p fandhe-ai-autodiff --release --lib ce_lin_public_judge_time -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn ce_lin_public_judge_time() {
    for (n, c, margin) in [
        (1usize, 100usize, 10.0f32),
        (1, 2000, 5.0),
        (1, 2100, 10.0),
        (150, 256, 5.0),
        (1500, 33, 5.0),
    ] {
        let mut z0 = Vec::new();
        for r in 0..n {
            let mut row: Vec<f32> = pattern(c, r as f32).iter().map(|v| 0.1 * v).collect();
            row[r % c] += margin;
            z0.extend(row);
        }
        let tgt: Vec<i32> = (0..n).map(|r| (r % c) as i32).collect();
        let opts = CrossEntropyOptions::default().label_smoothing(0.1);
        let t = std::time::Instant::now();
        let v = run_public(z0, |_tape, z| {
            let t = Tensor::<i32>::new(tgt.clone(), &[n]).unwrap();
            z.reshape(&[n, c])?
                .cross_entropy_loss_with(&t, 1, Reduction::Sum, &opts)
        })
        .0;
        eprintln!(
            "LINCETIME n={n} c={c} margin={margin} lin={} secs={:.3} verdict={}",
            ce_lin_ctx(1.0, None, n, c).is_some(),
            t.elapsed().as_secs_f64(),
            if v == Verdict::Accept {
                "受理"
            } else {
                "拒否"
            }
        );
    }
}
/// `-inf` のロジット（既定 CE は `reject_inf` を通らない）と `|x − max| > 2^23` のロジットは `exp_bounded` が上界を持たず、
/// `Lin` の救済は常に偽になる。新たな拒否も受理も生まない（旧判定と同じ verdict。設計 §82.11）。
#[test]
fn ce_lin_non_finite_and_huge_gap_logits_keep_previous_verdict() {
    let rows: Vec<Vec<f32>> = vec![
        vec![0.0, f32::NEG_INFINITY, 0.5, -0.5],
        vec![10.0, f32::NEG_INFINITY, 0.0, 1.0],
        vec![0.0, -1.0e9, 0.5, -0.5],
        vec![30.0, 0.0, -2.0e7, 1.0],
        vec![f32::NAN, 0.0, 1.0, 2.0],
    ];
    for row in rows {
        let c = row.len();
        let run = || {
            run_public(row.clone(), |_tape, z| {
                let t = Tensor::<i32>::new(vec![1], &[1]).unwrap();
                z.reshape(&[1, c])?
                    .cross_entropy_loss(&t, 1, Reduction::Sum)
            })
            .0
        };
        probe::FORCE_OLD.with(|f| f.set(true));
        let old = run();
        probe::FORCE_OLD.with(|f| f.set(false));
        let new = run();
        assert_eq!(old, new, "{row:?}: 旧判定と新判定の verdict が違う");
    }
}
/// 非対角の全要素の救済走査（`CeLinCtx::off`）は、本番では走査量の上限（`ce_lin_ctx`）のため `coeff` が相対の半分を
/// 超える C ≥ 2096 では到達しない。判定の関数を直接呼んで、走査の分岐（上位 2 クラスの積が拒否される行を全非対角で
/// 評価し直す）が either 受理として働くこと（`off = false` や `lin = None` より受理が増える）を確かめる。
#[test]
fn ce_lin_off_diagonal_scan_is_either_accept_when_forced() {
    let c = 2200usize;
    // 上位 2 クラスが同じ大きさ（y ≈ 0.5 ずつ）の行。上流 1 でも上位 2 クラスの積が大きく、`coeff·|v|` が拒否する。
    let mut row = vec![0.0f32; c];
    row[0] = 30.0;
    row[1] = 30.0;
    let weights = vec![1.0f64; c];
    let g = 1.0f64;
    let call = |lin: Option<CeLinCtx>| {
        cewo_cancel_check(
            &row,
            (0, 0, 1, c),
            (1.0, &weights, 1, 0.0),
            (g, 1.0, 1.0),
            lin,
        )
    };
    let none = call(None);
    let no_off = call(Some(CeLinCtx {
        g,
        w: None,
        off: false,
    }));
    let off = call(Some(CeLinCtx {
        g,
        w: None,
        off: true,
    }));
    eprintln!(
        "LINCEOFF none={} no_off={} off={}",
        none.is_ok(),
        no_off.is_ok(),
        off.is_ok()
    );
    assert!(none.is_err(), "従来の見積りはこの行を拒否するはず");
    // 対角と `g` は `Lin` が救い、上位 2 クラスの積だけが `off = false` では拒否のまま。全非対角の走査が受理に変える。
    assert!(
        no_off.is_err() && off.is_ok(),
        "no_off={no_off:?} off={off:?}"
    );
}
/// `flag` の行（確率が f32 で underflow する行）の非対角の `Lin` 救済も `CeLinCtx::off` で止まること（レビュー指摘。
/// 設計 §82.11.1 の `LIN_BUDGET`）。`off = false` では救済を呼ばず従来の見積りだけで判定するので拒否のまま、
/// `off = true` のときだけ O(C²) の救済で受理に変わる。
#[test]
fn ce_lin_flag_row_off_diagonal_rescue_respects_off() {
    let c = 2200usize;
    // 上位 2 クラスが同じ大きさ（y ≈ 0.5 ずつ）、残りは f32 の exp が非正規数へ underflow する（`flag` が立つ）。
    let mut row = vec![-70.0f32; c];
    row[0] = 30.0;
    row[1] = 30.0;
    let weights = vec![1.0f64; c];
    let g = 1.0f64;
    let call = |lin: Option<CeLinCtx>| {
        cewo_cancel_check(
            &row,
            (0, 0, 1, c),
            (1.0, &weights, 1, 0.0),
            (g, 1.0, 1.0),
            lin,
        )
    };
    let none = call(None);
    let no_off = call(Some(CeLinCtx {
        g,
        w: None,
        off: false,
    }));
    probe::LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
    let off = call(Some(CeLinCtx {
        g,
        w: None,
        off: true,
    }));
    let log = probe::LOG.with(|l| l.borrow_mut().take()).unwrap();
    // 対角は `Lin` が救うので、この行は最後に `flag` が立つ（flag 分岐に入る）。
    assert!(
        log.iter()
            .any(|e| matches!(e, probe::Event::Flag("CrossEntropy", true))),
        "この行は flag が立つはず"
    );
    eprintln!(
        "LINCEFLAGOFF none={} no_off={} off={}",
        none.is_ok(),
        no_off.is_ok(),
        off.is_ok()
    );
    assert!(none.is_err(), "従来の見積りはこの行を拒否するはず");
    assert!(
        no_off.is_err(),
        "off = false では flag 行の非対角を救済しない: {no_off:?}"
    );
    assert!(off.is_ok(), "off = true では救済で受理: {off:?}");
}

// =====================================================================================
// #4095: Triplet の forward の総称化（判定不変）と `Lin` の上界の soundness
// =====================================================================================

/// 総称化の前に `triplet_pair_row` が書いていた f64 の参照と f32 の再現の式（逐語のコピー。前例: #4090 の
/// `cos_fwd_previous`）。`triplet_pair_row` が全フィールドで bit 一致することが、判定を変えていない主な回帰の錠になる。
fn triplet_pair_row_previous(
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
    let sg: Vec<f64> = (0..h.len())
        .map(|k| f64::from(info.sign[base + k]))
        .collect();
    let reference = PairRow {
        fac: inv / norm,
        pm1: p - 1.0,
        dg: q.iter().map(|&x| (p - 1.0) * x.powf(p - 2.0)).collect(),
        e: q.iter().map(|&x| x.powf(p - 1.0)).collect(),
        sg: sg.clone(),
        sa: sg.clone(),
        dsg: vec![1.0; h.len()],
    };
    // 構造の f32 の再現（`pair_grad`: `u = v·const(1/mx)`、p=2 は `u/sqrt(Σu²)`、一般の p は
    // `(|u|/ssum^(1/p))^(p−1)`）。
    let p32 = p as f32;
    let u32v: Vec<f32> = rec.iter().map(|&c| c * inv32).collect();
    let au: Vec<f32> = u32v.iter().map(|x| x.abs()).collect();
    let norm32 = if p == 2.0 {
        sum32(u32v.iter().map(|&x| x * x)).sqrt()
    } else {
        sum32(au.iter().map(|&x| x.powf(p32))).powf(1.0 / p32)
    };
    let q32: Vec<f32> = au.iter().map(|&x| x / norm32).collect();
    let (e32, dg32): (Vec<f64>, Vec<f64>) = if p == 2.0 {
        (
            q32.iter().map(|&x| f64::from(x)).collect(),
            vec![1.0; h.len()],
        )
    } else {
        (
            q32.iter().map(|&x| f64::from(x.powf(p32 - 1.0))).collect(),
            q32.iter()
                .map(|&x| f64::from((p32 - 1.0) * x.powf(p32 - 2.0)))
                .collect(),
        )
    };
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
/// `PairRow` の全フィールドを bit 列へ（NaN は 1 つの bit 列へ正規化）。
fn pair_row_bits(r: &PairRow) -> Vec<u64> {
    let mut v = vec![f64_bits(r.fac), f64_bits(r.pm1)];
    for s in [&r.dg, &r.e, &r.sg, &r.sa, &r.dsg] {
        v.push(s.len() as u64);
        v.extend(s.iter().map(|&x| f64_bits(x)));
    }
    v
}

/// 差分行 `(x, y, eps)` から `triplet_pair_row` の入力 `(h, rec, info)` を作る（`pair_info` の拒否は通さず、
/// 離散量だけを同じ規則で作る。underflow・符号付きゼロ・NaN の行も錠にかけるため）。
fn tri_pair_inputs(x: &[f32], y: &[f32], eps: f32) -> (Vec<f64>, Vec<f32>, PairInfo) {
    let d = x.len();
    let h: Vec<f64> = (0..d)
        .map(|k| (f64::from(x[k]) - f64::from(y[k])) + f64::from(eps))
        .collect();
    let rec: Vec<f32> = (0..d).map(|k| (x[k] - y[k]) + eps).collect();
    let mx = h.iter().fold(0f64, |m, v| m.max(v.abs()));
    let inv = if mx > 0.0 { (1.0 / mx) as f32 } else { 1.0 };
    let info = PairInfo {
        inv_mx: vec![inv; d],
        sign: h.iter().map(|&v| sign0(v) as f32).collect(),
        zero: vec![mx == 0.0; d],
    };
    (h, rec, info)
}

#[test]
fn triplet_pair_row_generic_matches_previous_formulas() {
    let mut rows: Vec<(Vec<f32>, Vec<f32>, f32)> = Vec::new();
    // TriCase のパターン。
    for c in [
        TriCase {
            d: 8,
            ratio: 0.3,
            p: 3.0,
            swap: false,
            exp: 0,
        },
        TriCase {
            d: 64,
            ratio: 1e-3,
            p: 1.5,
            swap: true,
            exp: 0,
        },
        TriCase {
            d: 64,
            ratio: 1e-6,
            p: 10.0,
            swap: false,
            exp: 0,
        },
    ] {
        let (a, p, n) = c.inputs();
        rows.push((a.clone(), p.clone(), 0.0));
        rows.push((a, n, 1e-3));
    }
    // underflow する微小成分・符号付きゼロ・全要素 0・NaN。
    rows.push((vec![1e10, 1e10, 1e-40], vec![0.0, 0.0, 0.0], 0.0));
    rows.push((vec![1.0, 1e-42, -1e-42], vec![0.0, 0.0, 0.0], 0.0));
    rows.push((vec![0.0, -0.0, 0.0], vec![0.0, 0.0, -0.0], 0.0));
    rows.push((vec![2.0, 2.0], vec![2.0, 2.0], 0.0));
    rows.push((vec![f32::NAN, 1.0], vec![0.0, 0.0], 0.0));
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    for d in [1usize, 2, 3, 7, 16, 33] {
        for amp in [1e-30f32, 1e-8, 1e-3, 1.0, 1e3, 1e30] {
            for eps in [0.0f32, 1e-6] {
                let x: Vec<f32> = (0..d).map(|_| amp * rng.signed()).collect();
                let y: Vec<f32> = (0..d).map(|_| amp * rng.signed()).collect();
                rows.push((x, y, eps));
            }
        }
    }
    let mut compared = 0usize;
    let mut some = 0usize;
    for (x, y, eps) in &rows {
        let zeros = vec![0f32; x.len()];
        let y = if y.len() == x.len() { y } else { &zeros };
        let (h, rec, info) = tri_pair_inputs(x, y, *eps);
        for p in [1.5f64, 2.0, 3.0, 10.0, 50.0, 1e3] {
            for coeff in [1e-6f64, 1e-2] {
                let new = triplet_pair_row(&h, &rec, &info, 0, p, coeff);
                let old = triplet_pair_row_previous(&h, &rec, &info, 0, p, coeff);
                compared += 1;
                match (new, old) {
                    (None, None) => {}
                    (Some((r, s, f)), Some((r0, s0, f0))) => {
                        some += 1;
                        assert_eq!(pair_row_bits(&r), pair_row_bits(&r0), "参照 p={p}");
                        assert_eq!(pair_row_bits(&s), pair_row_bits(&s0), "構造 p={p}");
                        assert_eq!(f, f0, "flag p={p}");
                    }
                    (n, o) => panic!(
                        "None/Some が食い違う p={p}: new={} old={}",
                        n.is_some(),
                        o.is_some()
                    ),
                }
            }
        }
    }
    assert!(compared > 500, "入力が少なすぎる: {compared}");
    assert!(some > 200, "Some の入力が少なすぎる: {some}");
}

/// 掃引の入力: `a − p = [1, r·…]`、`a − n = [−2, r·…]`（swap は `p − n = [0.5, r·…]`）の乱数版（[`TriCase::inputs`] と同型）。
fn tri_random_rows(
    d: usize,
    ratio: f32,
    swap: bool,
    amp: f32,
    rng: &mut Rng,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let small = |rng: &mut Rng| {
        ratio * (0.3 + 0.7 * rng.signed().abs()) * if rng.signed() < 0.0 { -1.0 } else { 1.0 }
    };
    let a: Vec<f32> = (0..d).map(|_| amp * rng.signed()).collect();
    let vap: Vec<f32> = (0..d)
        .map(|k| if k == 0 { amp } else { amp * small(rng) })
        .collect();
    let p: Vec<f32> = a.iter().zip(&vap).map(|(a, v)| a - v).collect();
    let n: Vec<f32> = if swap {
        let vpn: Vec<f32> = (0..d)
            .map(|k| if k == 0 { 0.5 * amp } else { amp * small(rng) })
            .collect();
        p.iter().zip(&vpn).map(|(p, v)| p - v).collect()
    } else {
        let van: Vec<f32> = (0..d)
            .map(|k| if k == 0 { -2.0 * amp } else { amp * small(rng) })
            .collect();
        a.iter().zip(&van).map(|(a, v)| a - v).collect()
    };
    (a, p, n)
}

/// 1 ケース（`rows`・`eps`・`c.p`・`c.swap`・`c.exp`・`mean_n`）について、手組みの構造の f32 の二階（実際の eager 逆伝播）と
/// `tri_rev::<f32>` の双方が `|実測 − Lin.v| ≤ Lin.bound()·(1 + 16·N·u)` を満たすことを確かめる（非有限の要素は除く）。
/// 戻り値は上界が有限だった要素数（`pair_info` などの前段で拒否された入力は `None`＝除外）。
fn lin_sweep_tri_row(
    c: &TriCase,
    rows: &(Vec<f32>, Vec<f32>, Vec<f32>),
    eps: f32,
    mean_n: Option<usize>,
    st: &mut SweepStats,
) -> Option<usize> {
    use super::super::lin::LIN_U;
    let up = (f64::from(pow2(c.exp)), mean_n);
    let setup = triplet_setup((&rows.0, &rows.1, &rows.2), eps, c.p, c.swap, up)?;
    let sd = seeds(c.d, &[]);
    let hand = c.hand_rows_with(rows.clone(), eps, mean_n, &sd)?;
    let lin = triplet_lin_rows(&setup, &sd);
    let mut finite = 0usize;
    for (k, (j, per_out)) in hand.iter().enumerate() {
        let model = triplet_model_rows_f32(&setup, *j);
        let slack = 1.0 + 16.0 * lin[k].events as f64 * LIN_U;
        for o in 0..3 {
            for w in 0..4 {
                for (i, lv) in lin[k].rows[o][w].iter().enumerate() {
                    for (label, actual) in [("hand", per_out[o][w][i]), ("model", model[o][w][i])] {
                        let actual = f64::from(actual);
                        if !actual.is_finite() || !lv.0.is_finite() || !lv.1.is_finite() {
                            continue;
                        }
                        let dev = (actual - lv.0).abs();
                        assert!(
                            dev <= lv.1 * slack,
                            "{c:?} eps={eps} mean={mean_n:?} {label} o={o} w={w} seed={j} i={i}: dev={dev:e} bound={:e}",
                            lv.1
                        );
                        st.checked += 1;
                        if label == "hand" {
                            finite += 1;
                        }
                        if dev > 0.0 {
                            st.nonzero += 1;
                        }
                        if lv.1 > 0.0 {
                            st.worst = st.worst.max(dev / lv.1);
                            if dev > 0.1 * lv.1 {
                                st.tight += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    Some(finite)
}

#[test]
fn lin_bound_covers_f32_deviation_for_triplet_sweep() {
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let mut st = SweepStats::default();
    let mut tried = 0usize;
    let mut used = 0usize;
    for d in [2usize, 3, 8, 16, 32, 64] {
        for p in [2.0f32, 1.5, 3.0, 10.0] {
            for ratio in [1e-1f32, 1e-3, 1e-6] {
                for eps in [0.0f32, 1e-4] {
                    for swap in [false, true] {
                        for exp in [0i32, 13, 20] {
                            let c = TriCase {
                                d,
                                ratio,
                                p,
                                swap,
                                exp,
                            };
                            let rows = tri_random_rows(d, ratio, swap, 1.0, &mut rng);
                            tried += 1;
                            let mut ok = lin_sweep_tri_row(&c, &rows, eps, None, &mut st).is_some();
                            if exp == 13 {
                                ok |= lin_sweep_tri_row(&c, &rows, eps, Some(3), &mut st).is_some();
                                ok |= lin_sweep_tri_row(&c, &rows, eps, Some(7), &mut st).is_some();
                            }
                            used += usize::from(ok);
                        }
                    }
                }
            }
        }
    }
    assert!(
        st.checked > 1000,
        "検査した要素が少なすぎる: {} (used {used}/{tried})",
        st.checked
    );
    assert!(
        st.nonzero > 0,
        "実測の誤差が全て 0 では soundness を確かめていない"
    );
    assert!(st.tight > 0, "上界が緩すぎて空振り: tight={}", st.tight);
    eprintln!(
        "LINTRI checked={} nonzero={} tight={} worst={:.4} used={used}/{tried}",
        st.checked, st.nonzero, st.tight, st.worst
    );
}

/// 微小成分が行スケールで非正規数・0 へ落ちる行の soundness。`a = 0`、`a − p = [1, 微小…]`、`a − n = [−2, 微小…]`。
/// 上界が有限だった要素数も出す（ρ > 1/2 で上界なしになる範囲を #4096 へ伝えるため）。
#[test]
fn lin_bound_covers_f32_deviation_for_triplet_underflow_rows() {
    let mut st = SweepStats::default();
    let mut finite_total = 0usize;
    for d in [2usize, 4, 8] {
        for tiny in [1e-20f32, 1e-30, 1e-38, 1e-42] {
            for p in [3.0f32, 10.0, 50.0] {
                let c = TriCase {
                    d,
                    ratio: tiny,
                    p,
                    swap: false,
                    exp: 13,
                };
                let a = vec![0.0f32; d];
                let vap: Vec<f32> = (0..d).map(|k| if k == 0 { 1.0 } else { tiny }).collect();
                let van: Vec<f32> = (0..d).map(|k| if k == 0 { -2.0 } else { tiny }).collect();
                let pp: Vec<f32> = vap.iter().map(|v| -v).collect();
                let nn: Vec<f32> = van.iter().map(|v| -v).collect();
                let rows = (a, pp, nn);
                let f = lin_sweep_tri_row(&c, &rows, 0.0, None, &mut st).unwrap_or(0);
                finite_total += f;
                eprintln!("LINTRIU d={d} tiny={tiny:e} p={p} finite_hand_elems={f}");
            }
        }
    }
    assert!(st.checked > 100, "検査した要素が少なすぎる: {}", st.checked);
    eprintln!("LINTRIU total finite={finite_total} checked={}", st.checked);
}

#[test]
fn triplet_f32_model_stays_close_to_hand_built_structure() {
    // 受理される小さな入力では、`tri_rev::<f32>` の写しが手組みの構造（実際の eager 逆伝播）と REQ-2 の範囲で一致する。
    // 縮約の順序の違いで bit 一致しない要素は許す（soundness は別テストが正）。
    let mut worst = 0f64;
    let mut n = 0usize;
    for (p, swap, exp) in [
        (2.0f32, false, 0),
        (3.0, false, 3),
        (2.0, true, 0),
        (3.0, true, 7),
        (1.5, false, 0),
    ] {
        let c = TriCase {
            d: 6,
            ratio: 0.3,
            p,
            swap,
            exp,
        };
        let rows = c.inputs();
        let up = (f64::from(pow2(exp)), None);
        let setup =
            triplet_setup((&rows.0, &rows.1, &rows.2), 0.0, p, swap, up).expect("pair_info を通る");
        let sd: Vec<usize> = (0..c.d).collect();
        let hand = c.hand_rows(&sd).expect("pair_info を通る");
        for (j, per_out) in &hand {
            let model = triplet_model_rows_f32(&setup, *j);
            for o in 0..3 {
                for w in 0..4 {
                    for (h, m) in per_out[o][w].iter().zip(&model[o][w]) {
                        worst = worst.max(req2_ratio(f64::from(*m), f64::from(*h)));
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(n > 500, "比較した要素が少なすぎる: {n}");
    assert!(worst < 1.0, "f32 の写しが手組みから外れすぎ: {worst}");
}

/// `LINTRIM`: §82.4 の Triplet の候補（`triplet_followup_candidates`）について、`Lin` の事象数 N・`16Nu`・上界の最大・
/// 閾値を超える要素の数・1 seed あたりの時間を出す。受理に変えてよいかは判断しない（#4096 の担当）。
/// `cargo test -p fandhe-ai-autodiff --release --lib triplet_lin_candidates_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn triplet_lin_candidates_measurement() {
    use super::super::lin::LIN_U;
    for c in triplet_followup_candidates() {
        let rows = c.inputs();
        let up = (f64::from(pow2(c.exp)), None);
        let Some(setup) = triplet_setup((&rows.0, &rows.1, &rows.2), 0.0, c.p, c.swap, up) else {
            eprintln!(
                "LINTRIM d={} ratio={} p={} swap={} setup 不可（前段で拒否）",
                c.d, c.ratio, c.p, c.swap
            );
            continue;
        };
        let sd = seeds(c.d, &[c.d / 2]);
        let t0 = std::time::Instant::now();
        let lin = triplet_lin_rows(&setup, &sd);
        let per_seed = t0.elapsed().as_secs_f64() / sd.len() as f64;
        let n = lin.iter().map(|r| r.events).max().unwrap_or(0);
        let (mut maxb, mut over, mut total, mut nonfinite) = (0f64, 0usize, 0usize, 0usize);
        for r in &lin {
            for o in 0..3 {
                for w in 0..4 {
                    for (v, b) in &r.rows[o][w] {
                        total += 1;
                        if !b.is_finite() || !v.is_finite() {
                            nonfinite += 1;
                            continue;
                        }
                        maxb = maxb.max(*b);
                        let thr =
                            ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * v.abs());
                        if *b * (1.0 + 16.0 * n as f64 * LIN_U) >= thr {
                            over += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "LINTRIM d={} ratio={} p={} swap={} N={} 16Nu={:.4} max_bound={:e} over_thr={}/{} nonfinite={} secs_per_seed={:.4e}",
            c.d,
            c.ratio,
            c.p,
            c.swap,
            n,
            16.0 * n as f64 * LIN_U,
            maxb,
            over,
            total,
            nonfinite,
            per_seed
        );
    }
}
// =====================================================================================
// #4096: Triplet の判定を `Lin` の上界へ切り替える（either 受理。設計 §82.13）
// =====================================================================================

/// `TriCase` の行を `Lin` の判定（[`TriLinState`]）で走査する（計測用。判定と同じ状態を使い、`entry_check_lin` と同じ式で
/// `err/閾値` も出す）。`with_x` なら 9 ブロックの全レーン（`seed_list` の各 `j` で `3×3×d` 要素）も評価する。`g` は
/// 出力 3 つ × `seed_list`。戻り値は走査結果と forward にかかった秒数。前段で拒否される入力・救済しない行は `None`。
fn tri_lin_scan_timed(
    c: &TriCase,
    mean_n: Option<usize>,
    seed_list: &[usize],
    with_x: bool,
) -> Option<(LinScan, f64)> {
    use super::super::lin::{LIN_U, lin_events, lin_truncate};
    let rows = c.inputs();
    let up = (f64::from(pow2(c.exp)), mean_n);
    let setup = triplet_setup((&rows.0, &rows.1, &rows.2), 0.0, c.p, c.swap, up)?;
    let t0 = std::time::Instant::now();
    let mut st = TriLinState::new(setup);
    let fwd_secs = t0.elapsed().as_secs_f64();
    let mut out = LinScan {
        fwd_events: lin_events(),
        ..LinScan::default()
    };
    for &j in seed_list {
        for o in 0..3 {
            let (mark, sa, sb) = st.seed(o, j);
            let mut visit = |l: Lin, s: f32| {
                let slack = 16.0 * lin_events() as f64 * LIN_U;
                let err = (f64::from(s) - l.v).abs() + l.bound() * (1.0 + slack);
                let thr = ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * l.v.abs());
                out.checked += 1;
                out.max_events = out.max_events.max(lin_events());
                let m = err / thr;
                // 非有限の上界（`Lin` の契約の外）は拒否として数える。
                if !m.is_finite() || m >= 1.0 || slack >= 0.01 {
                    out.rejected += 1;
                }
                if m.is_finite() {
                    out.max_margin = out.max_margin.max(m);
                }
            };
            visit(sa.g, sb.g);
            if with_x {
                for wrt in 0..3 {
                    for i in 0..c.d {
                        let l = tri_x(&st.st, &st.lin, &sa, wrt, (i, j));
                        let s = tri_x(&st.st, &st.f32m, &sb, wrt, (i, j));
                        visit(l, s);
                        lin_truncate(mark);
                    }
                }
            }
        }
    }
    out.secs = t0.elapsed().as_secs_f64();
    Some((out, fwd_secs))
}

/// 旧判定（`Lin` を使わない）の公開経路の verdict。
fn public_old_tri(c: &TriCase) -> Verdict {
    probe::FORCE_OLD.with(|f| f.set(true));
    let v = c.public().0;
    probe::FORCE_OLD.with(|f| f.set(false));
    v
}

/// `LINTRICAND`: §82.4 の候補について旧判定・新判定の verdict と、`Lin` の全 seed 走査（`g` と 9 ブロック）の最大
/// `err/閾値`・拒否要素数を出す（d=512 は seed を間引く）。
/// `cargo test -p fandhe-ai-autodiff --release --lib triplet_lin_followup_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn triplet_lin_followup_measurement() {
    for c in triplet_followup_candidates() {
        let old = public_old_tri(&c);
        let m = c.measure();
        let sd: Vec<usize> = if c.d > 100 {
            (0..c.d).step_by(c.d / 16).collect()
        } else {
            (0..c.d).collect()
        };
        let scan = tri_lin_scan_timed(&c, None, &sd, true).map(|(s, _)| s);
        eprintln!(
            "LINTRICAND {c:?} old={old:?} new={:?} x={:?} g={:?} scan={scan:?}",
            m.verdict, m.x, m.g
        );
    }
}

/// `LINTRICOST`: 見積り定数を決める計測。d × p × swap で forward の事象数・時間、`g` の要素あたりの時間、9 ブロックの
/// 要素あたりの時間を `Rf` の 1 反復（3.45 ns。§80.5）相当に直して出す。
/// `cargo test -p fandhe-ai-autodiff --release --lib triplet_lin_cost_profile -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn triplet_lin_cost_profile() {
    const RF: f64 = 3.45e-9;
    for swap in [false, true] {
        for p in [2.0f32, 1.5, 3.0, 10.0] {
            for d in [2usize, 8, 32, 64, 128, 256, 512] {
                let case = TriCase {
                    d,
                    ratio: 0.1,
                    p,
                    swap,
                    exp: 0,
                };
                let sd: Vec<usize> = (0..8).map(|k| k * d / 8).collect();
                let mut best = [f64::MAX; 3];
                let mut maxev = 0;
                let mut fwd_ev = 0;
                let mut ok = true;
                for _ in 0..3 {
                    let Some((g, f)) = tri_lin_scan_timed(&case, None, &sd, false) else {
                        ok = false;
                        break;
                    };
                    let Some((x, _)) = tri_lin_scan_timed(&case, None, &sd, true) else {
                        ok = false;
                        break;
                    };
                    best[0] = best[0].min(f);
                    best[1] = best[1].min((g.secs - f) / (sd.len() * 3) as f64);
                    best[2] = best[2].min((x.secs - g.secs) / (sd.len() * 3 * 3 * d) as f64);
                    maxev = maxev.max(x.max_events).max(g.max_events);
                    fwd_ev = g.fwd_events;
                }
                if !ok {
                    eprintln!("LINTRICOST d={d} p={p} swap={swap} setup 不可");
                    continue;
                }
                let (fwd_it, g_it, x_it) = (best[0] / RF, best[1] / RF, best[2] / RF);
                eprintln!(
                    "LINTRICOST d={d} p={p} swap={swap} fwd_events={fwd_ev} max_events={maxev} bound_events={} fwd_iter={fwd_it:.3e} g_iter/elem={g_it:.3e} x_iter/elem={x_it:.3e} est_g_total={:.3e} _={:.1} est_x_total={:.3e} meas_g_total={:.3e} meas_x_total={:.3e}",
                    tri_lin_events_max(d, swap),
                    tri_lin_cost_g(d, swap) as f64,
                    0.0,
                    tri_lin_cost_x(d) as f64,
                    fwd_it + g_it * (3 * d) as f64,
                    x_it * (9 * d * d) as f64,
                );
            }
        }
    }
}
/// 判定時の事象数が `tri_lin_events_max(d, swap)` に収まる（Sum・Mean、p ∈ {2, 1.5, 3, 10}、swap の有無）。`tri_lin_ctx` の
/// 余裕係数の条件（`16·事象数·u < 0.01`）がこの上界に依るので、実測が上界を超えたらここで落ちる。
#[test]
fn triplet_lin_events_stay_within_bound() {
    let mut worst_fixed = 0usize;
    let mut checked = 0usize;
    for swap in [false, true] {
        let pairs = if swap { 3 } else { 2 };
        for p in [2.0f32, 1.5, 3.0, 10.0] {
            for d in [2usize, 3, 8, 33, 64] {
                for ratio in [1e-1f32, 1e-3] {
                    for mean_n in [None, Some(3usize)] {
                        let case = TriCase {
                            d,
                            ratio,
                            p,
                            swap,
                            exp: 7,
                        };
                        let sd = seeds(d, &[d / 2]);
                        let Some((r, _)) = tri_lin_scan_timed(&case, mean_n, &sd, true) else {
                            continue;
                        };
                        checked += 1;
                        assert!(
                            r.max_events <= tri_lin_events_max(d, swap),
                            "{case:?} mean={mean_n:?}: 事象数 {} が上界 {} 超",
                            r.max_events,
                            tri_lin_events_max(d, swap)
                        );
                        worst_fixed = worst_fixed.max(
                            r.max_events
                                .saturating_sub(TRI_LIN_EVENTS_PER_PAIR_LANE * pairs * d),
                        );
                    }
                }
            }
        }
    }
    assert!(checked > 40, "検査した入力が少なすぎる: {checked}");
    assert!(
        worst_fixed <= TRI_LIN_EVENTS_FIXED,
        "定数分 {worst_fixed} が TRI_LIN_EVENTS_FIXED を超える"
    );
    eprintln!("LINTRIEV checked={checked} worst_fixed={worst_fixed}");
}

/// `tri_lin_ctx` の境界: 余裕係数の上限（`16·事象数·u < 0.01`）、`g` の走査量の上限、9 ブロックの要素まで許す走査量の上限
/// （2 段が独立であること）、テンソル単位の選択（行数が増えると偽）、swap で事象数と走査量が増えること、旧判定の強制。
#[test]
fn tri_use_lin_boundaries() {
    use super::super::lin::{LIN_BUDGET, LIN_U};
    let some = |n: usize, d: usize, swap: bool| tri_lin_ctx(1.0, None, n, d, swap);
    let slack_ok = |d: usize, swap: bool| 16.0 * tri_lin_events_max(d, swap) as f64 * LIN_U < 0.01;
    // 行数 1 で `g` の走査量と余裕係数が収まる最大の d。
    let dmax = (1..100_000usize)
        .take_while(|&d| tri_lin_cost_g(d, false) <= LIN_BUDGET && slack_ok(d, false))
        .last()
        .unwrap();
    assert!(some(1, dmax, false).is_some() && some(1, dmax + 1, false).is_none());
    // 9 ブロックの要素まで許す最大の d は `g` だけの最大以下（別々の条件）。
    let xmax = (1..=dmax)
        .take_while(|&d| tri_lin_cost_g(d, false) + tri_lin_cost_x(d) <= LIN_BUDGET)
        .last()
        .unwrap();
    assert!(some(1, xmax, false).is_some_and(|l| l.x));
    assert!(xmax < dmax, "x の上限は g の上限より小さいはず");
    assert!(some(1, xmax + 1, false).is_some_and(|l| !l.x));
    // 行数が増えると走査量が増える（テンソル単位で選ぶ）。
    assert!(some(1, 64, false).is_some() && some(1_000_000, 64, false).is_none());
    // 余裕係数を満たさない d は走査量が足りていても選ばない（独立の条件であること）。
    let d_slack = (1..1_000_000usize).find(|&d| !slack_ok(d, false)).unwrap();
    assert!(some(1, d_slack, false).is_none());
    // swap は事象数・走査量が増える。
    assert!(tri_lin_events_max(64, true) > tri_lin_events_max(64, false));
    assert!(tri_lin_cost_g(64, true) > tri_lin_cost_g(64, false));
    // 旧判定の強制（変異確認・比較用）。
    probe::FORCE_OLD.with(|f| f.set(true));
    assert!(some(1, 8, false).is_none());
    probe::FORCE_OLD.with(|f| f.set(false));
}

/// §82.4 の候補のうち、従来の見積りが `coeff·m` の項だけで拒否していた行を `Lin` の上界が受理に変える（d=64 の
/// swap あり p=3 と p=1.5）。受理に変えた行の二階が f64 の閉じた式と REQ-2 で一致すること（REQ-2 比 < 1。参照の不確かさ < 5%）。
#[test]
fn triplet_lin_accepts_rows_rejected_by_coeff_bound_and_matches_f64() {
    let mut accepted = 0usize;
    for (k, c) in triplet_followup_candidates().into_iter().enumerate() {
        if k != 1 && k != 3 {
            continue;
        }
        match public_old_tri(&c) {
            Verdict::Cancel { cls, .. } => assert_eq!(cls, "coeff·m のみ", "{c:?}"),
            v => panic!("{c:?}: 旧判定は coeff·m で拒否するはず: {v:?}"),
        }
        let m = c.measure();
        assert_eq!(
            m.verdict,
            Verdict::Accept,
            "{c:?}: Lin の上界で受理に変わるはず"
        );
        for (label, acc) in [("x", m.x), ("g", m.g)] {
            let a = acc.expect("計測");
            assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
            assert!(a.unc < 0.05, "{c:?}: {label} の参照の不確かさ {:e}", a.unc);
        }
        accepted += 1;
    }
    assert_eq!(accepted, 2);
}

/// 拒否のまま残る候補の理由を固定する（設計 §82.13.3）。上界を 0 にする変異では k=0・k=4 が受理に変わり落ちる。
/// - k=0（d=64・p=3・swap なし）: 全 seed を `Lin` で走査すると 1 要素だけ上界が閾値を超える（実際の二階は REQ-2 内）。
/// - k=4（d=64・r=1e-6・p=10・`flag` 行）: 閾値を超える要素が複数残る。
/// - k=2（d=512・p=2）: `g` への二階は救済できるが、9 ブロックの要素の救済走査は走査量の上限を超えるため行わない。
#[test]
fn triplet_followup_candidates_exceed_lin_bound() {
    let cs = triplet_followup_candidates();
    for k in [0usize, 4] {
        let c = cs[k];
        let m = c.measure();
        assert!(
            matches!(m.verdict, Verdict::Cancel { .. }),
            "{c:?}: 拒否のまま: {:?}",
            m.verdict
        );
        let sd: Vec<usize> = (0..c.d).collect();
        let (scan, _) = tri_lin_scan_timed(&c, None, &sd, true).expect("setup");
        assert!(
            scan.rejected >= 1,
            "{c:?}: Lin の上界が閾値を超える要素が残る: {scan:?}"
        );
        assert!(scan.max_margin >= 1.0 || scan.rejected > 0, "{c:?}");
        // 実際の二階は REQ-2 内（上界の項だけで拒否している）。
        let x = m.x.expect("計測");
        assert!(x.ratio < 0.25, "{c:?}: x の REQ-2 比 {:e}", x.ratio);
    }
    let c = cs[2];
    let ctx = tri_lin_ctx(1.0, None, 1, c.d, c.swap).expect("g の救済は走査量内");
    assert!(
        !ctx.x,
        "{c:?}: 9 ブロックの要素の救済は走査量の上限を超える"
    );
    assert!(matches!(c.measure().verdict, Verdict::Cancel { .. }));
}

/// 複数行・Mean・eps ≠ 0 の公開経路（`[n, d]` の 3 つ組）の verdict。`force_old` なら `Lin` を使わない旧判定。
#[allow(clippy::too_many_arguments)]
fn tri_multi_verdict(
    rows: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    d: usize,
    opts: (f32, f32, bool),
    reduction: Reduction,
    exp: i32,
    force_old: bool,
) -> Verdict {
    let (eps, p, swap) = opts;
    let n = rows.len();
    let mut z0: Vec<f32> = rows.iter().flat_map(|r| r.0.clone()).collect();
    z0.extend(rows.iter().flat_map(|r| r.1.clone()));
    z0.extend(rows.iter().flat_map(|r| r.2.clone()));
    let options = TripletMarginOptions::default()
        .margin(1000.0)
        .eps(eps)
        .p(p)
        .swap(swap);
    probe::FORCE_OLD.with(|f| f.set(force_old));
    let v = run_public(z0, move |tape, zv| {
        let x = |k: usize| zv.narrow(0, k * n * d, n * d)?.reshape(&[n, d]);
        let l = x(0)?.triplet_margin_loss(&x(1)?, &x(2)?, &options, reduction)?;
        l.mul(&tape.var_no_grad(&Tensor::scalar(pow2(exp))))?
            .sum(None)
    })
    .0;
    probe::FORCE_OLD.with(|f| f.set(false));
    v
}

/// Mean・eps ≠ 0・複数行・swap を通しても、旧受理 → 新拒否は 0 件（either 受理の構造的な保証の確認）。
/// 旧拒否 → 新受理に変わる件数も出す。
#[test]
fn triplet_lin_mean_eps_multi_row_never_flip_accept_to_reject() {
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let (mut total, mut flipped_to_accept, mut old_accept) = (0usize, 0usize, 0usize);
    for d in [4usize, 16, 64] {
        for p in [2.0f32, 1.5, 3.0, 10.0] {
            for swap in [false, true] {
                for eps in [0.0f32, 1e-4] {
                    for reduction in [Reduction::Sum, Reduction::Mean] {
                        for ratio in [1e-1f32, 1e-3, 1e-6] {
                            for exp in [0i32, 13] {
                                let rows: Vec<_> = (0..3)
                                    .map(|_| tri_random_rows(d, ratio, swap, 1.0, &mut rng))
                                    .collect();
                                let opts = (eps, p, swap);
                                let old = tri_multi_verdict(&rows, d, opts, reduction, exp, true);
                                let new = tri_multi_verdict(&rows, d, opts, reduction, exp, false);
                                total += 1;
                                match (
                                    matches!(old, Verdict::Accept),
                                    matches!(new, Verdict::Accept),
                                ) {
                                    (true, false) => panic!(
                                        "旧受理 → 新拒否 d={d} p={p} swap={swap} eps={eps} {reduction:?} ratio={ratio} exp={exp}: {new:?}"
                                    ),
                                    (true, true) => old_accept += 1,
                                    (false, true) => flipped_to_accept += 1,
                                    (false, false) => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(total >= 500, "{total}");
    assert!(old_accept > 0, "旧受理の入力が無い: {old_accept}");
    eprintln!(
        "LINTRIMULTI total={total} old_accept={old_accept} flipped_to_accept={flipped_to_accept}"
    );
}

/// 救済しない行（符号が食い違う・全要素 0 のペア）では [`tri_setup_row`] が `None` を返し、従来の判定が残る。
#[test]
fn triplet_lin_sign_mismatch_and_zero_rows_are_not_rescued() {
    let d = 3;
    let (a, p, n) = (
        vec![0.0f32; d],
        vec![-1.0f32, 0.5, 0.25],
        vec![2.0f32, -0.5, 0.75],
    );
    let info = |sign: Vec<f32>, zero: bool| PairInfo {
        inv_mx: vec![1.0; d],
        sign,
        zero: vec![zero; d],
    };
    // ap の差分 `a − p = [1, −0.5, −0.25]`、an の差分 `a − n = [−2, 0.5, −0.75]`。
    let good_ap = info(vec![1.0, -1.0, -1.0], false);
    let good_an = info(vec![-1.0, 1.0, -1.0], false);
    let mk = |ap: &PairInfo, an: &PairInfo| {
        tri_setup_row(
            (&a, &p, &n),
            (0.0, 3.0, false),
            (1.0, None),
            [Some(ap), Some(an), None],
            0,
            0,
        )
    };
    assert!(mk(&good_ap, &good_an).is_some(), "正のプローブ: 救済できる");
    let flipped = info(vec![1.0, 1.0, -1.0], false);
    assert!(mk(&flipped, &good_an).is_none(), "符号の食い違い");
    let zeroed = info(vec![1.0, -1.0, -1.0], true);
    assert!(mk(&good_ap, &zeroed).is_none(), "全要素 0 のペア");
    assert!(
        tri_setup_row(
            (&a, &p, &n),
            (0.0, 3.0, true),
            (1.0, None),
            [Some(&good_ap), Some(&good_an), None],
            0,
            0,
        )
        .is_none(),
        "swap で pn の pair_info が無い"
    );
}

/// 上界を持たない `Lin`（巨大な p・非正規数域）の行では、救済が働かず新判定が旧判定と一致する。
#[test]
fn triplet_lin_unbounded_rows_keep_previous_verdict() {
    let mut compared = 0usize;
    for d in [2usize, 8] {
        for p in [10.0f32, 50.0] {
            for ratio in [1e-6f32, 1e-20, 1e-30] {
                for swap in [false, true] {
                    let c = TriCase {
                        d,
                        ratio,
                        p,
                        swap,
                        exp: 0,
                    };
                    let old = public_old_tri(&c);
                    let new = c.public().0;
                    if matches!(old, Verdict::Accept) {
                        assert_eq!(new, Verdict::Accept, "{c:?}");
                    }
                    compared += 1;
                    if p == 50.0 && !matches!(old, Verdict::Accept) {
                        assert_eq!(new, old, "{c:?}: 上界を持たない行は旧判定のまま");
                    }
                }
            }
        }
    }
    assert!(compared >= 24);
}

/// `LINTRISW`: §82.2 の Triplet の行（`tri_cases`）について旧判定と新判定の受理・拒否を並べる。「旧受理 → 新拒否」は 0 件
/// （either 受理の構造的な保証）、「旧拒否 → 新受理」の行は REQ-2 比 < 1 であることを assert する。
/// `cargo test -p fandhe-ai-autodiff --release --lib triplet_lin_switch_rejection_delta_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn triplet_lin_switch_rejection_delta_measurement() {
    let (mut total, mut old_rej, mut new_rej, mut flipped) = (0usize, 0usize, 0usize, 0usize);
    let t0 = std::time::Instant::now();
    for c in tri_cases() {
        let old = public_old_tri(&c);
        let new = c.public().0;
        total += 1;
        let (o, n) = (
            matches!(old, Verdict::Accept),
            matches!(new, Verdict::Accept),
        );
        old_rej += usize::from(!o);
        new_rej += usize::from(!n);
        assert!(!(o && !n), "旧受理 → 新拒否 {c:?}: {new:?}");
        if !o && n {
            flipped += 1;
            let m = c.measure();
            for (label, acc) in [("x", m.x), ("g", m.g)] {
                let a = acc.expect("計測");
                assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
            }
            eprintln!("LINTRISW flip {c:?}");
        }
    }
    eprintln!(
        "LINTRISW total={total} old_reject={old_rej} new_reject={new_rej} flipped_to_accept={flipped} secs={:.1}",
        t0.elapsed().as_secs_f64()
    );
}

/// `LINTRITIME`: 公開経路（`backward_create_graph`）の判定時間。§83.1 の水準（0.5 秒未満）と比べる。
/// `cargo test -p fandhe-ai-autodiff --release --lib triplet_lin_public_judge_time -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn triplet_lin_public_judge_time() {
    for c in triplet_followup_candidates() {
        let mut best = f64::MAX;
        let mut best_old = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let _ = c.public();
            best = best.min(t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            let _ = public_old_tri(&c);
            best_old = best_old.min(t.elapsed().as_secs_f64());
        }
        eprintln!("LINTRITIME {c:?} new={best:.4}s old={best_old:.4}s");
    }
}
/// `u = rec·inv` が 0 へ underflow して `sa = 0`（f64 の符号 `sg` は非ゼロ）になる行は救済しない（[`TriSetup::rescuable`]）。
/// モデルは離散量ごと「曲率が 0」の値を作って f32 の値と一致してしまい、真の曲率とのずれを `Lin` の上界が覆えないため。
/// 統合テスト `triplet_scaled_component_underflow_is_rejected_or_exact` の入力（上流 1e12 倍は拒否）が正本。
#[test]
fn triplet_lin_underflowed_sign_rows_are_not_rescuable() {
    let a = vec![0.0f32; 3];
    let p = vec![-1e10f32, -1e10, -1e-40];
    let n = vec![-2e10f32, -1e10, -2e-40];
    let st = triplet_setup((&a, &p, &n), 0.0, 2.1, false, (1e12, None)).expect("setup は作れる");
    assert!(!st.rescuable(), "u が 0 へ underflow した行は救済しない");
    // 正のプローブ: 微小成分が無い同型の行は救済できる。
    let p2 = vec![-1e10f32, -1e10, -1e4];
    let n2 = vec![-2e10f32, -1e10, -2e4];
    let ok = triplet_setup((&a, &p2, &n2), 0.0, 2.1, false, (1.0, None)).expect("setup");
    assert!(ok.rescuable());
}
/// §82.2 の行のうち d ≤ 64 で、旧判定が拒否し新判定が受理に変えた行（旧受理 → 新拒否は 0 件）の二階が f64 の閉じた式と
/// REQ-2 で一致すること（REQ-2 比 < 1）。`triplet_lin_switch_rejection_delta_measurement`（`--release`・d=512 を含む）の
/// 通常テスト版。
#[test]
fn triplet_lin_flipped_rows_match_f64() {
    let (mut total, mut flipped) = (0usize, 0usize);
    for c in tri_cases().into_iter().filter(|c| c.d <= 64) {
        let old = public_old_tri(&c);
        let new = c.public().0;
        total += 1;
        let (o, n) = (
            matches!(old, Verdict::Accept),
            matches!(new, Verdict::Accept),
        );
        assert!(!(o && !n), "旧受理 → 新拒否 {c:?}: {new:?}");
        if !o && n {
            flipped += 1;
            let m = c.measure();
            for (label, acc) in [("x", m.x), ("g", m.g)] {
                let a = acc.expect("計測");
                assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
                assert!(a.unc < 0.05, "{c:?}: {label} の参照の不確かさ {:e}", a.unc);
            }
        }
    }
    assert!(total >= 150, "{total}");
    assert!(flipped >= 10, "受理に変わった行が少なすぎる: {flipped}");
}

// =====================================================================================
// #4133: Lin 救済の通常テストの穴（Triplet の `an_coeff` 中間・CE の複数サンプル／Mean／ignore_index／class_weight）。
// 判定のコードは変えず、`Lin` の上界と受理に変わった行の f64 照合を通常テストで固定する（設計 §82.14）
// =====================================================================================

/// 仮数の下位 13 bit を落とす（10 bit の仮数）。差分の f32 演算を厳密にして同値の同値性を保つ。
fn trim_mantissa(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & !0x1FFF)
}

/// swap で `d_an == d_pn` が厳密に成り立つ行（`an_coeff = 0.5`）。`a − n = u`、`a − p = 2u`（`p − n = −u`）。
/// `u = [0.5, ratio·…]`。`a` は 2^-6 の格子、`u` は 10 bit の仮数にして f32 の `n = a − u`・`p = a − 2u` を厳密にする。
/// 既存の掃引〔`tri_random_rows`〕の swap 行は `d_pn < d_an`（`an_coeff = 0`）で、この同値の行は含まれない。
fn tri_tie_rows(d: usize, ratio: f32, rng: &mut Rng) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..d)
        .map(|_| (rng.signed() * 64.0).round() / 64.0)
        .collect();
    let u: Vec<f32> = (0..d)
        .map(|k| {
            if k == 0 {
                0.5
            } else {
                let s = if rng.signed() < 0.0 { -1.0 } else { 1.0 };
                trim_mantissa(s * ratio * (0.3 + 0.7 * rng.signed().abs()))
            }
        })
        .collect();
    let n: Vec<f32> = a.iter().zip(&u).map(|(a, u)| a - u).collect();
    let p: Vec<f32> = a.iter().zip(&u).map(|(a, u)| a - 2.0 * u).collect();
    (a, p, n)
}

/// `an_coeff` が中間（0 でも 1 でもない）になる行か（f64 の統計で判定。本番と同じ `triplet_distance_stats`）。
fn tri_is_tie(rows: &(Vec<f32>, Vec<f32>, Vec<f32>), p: f32) -> bool {
    let (a, pp, n) = TriCase::f64_of(rows);
    let params = crate::eval::TripletDistanceParams {
        p: f64::from(p),
        eps: 0.0,
        swap: true,
    };
    let (stats, ..) = crate::eval::triplet_distance_stats(&a, &pp, &n, 0, rows.0.len(), &params);
    stats.an_coeff > 0.0 && stats.an_coeff < 1.0
}

impl TriCase {
    /// [`TriCase::measure`] を明示的な行（Sum・eps 0）で呼ぶ形。公開経路の verdict は `tri_multi_verdict` の 1 行。
    fn measure_rows(&self, rows: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> Meas {
        let verdict = tri_multi_verdict(
            std::slice::from_ref(rows),
            self.d,
            (0.0, self.p, self.swap),
            Reduction::Sum,
            self.exp,
            false,
        );
        let sd = seeds(self.d, &[0, 1]);
        let Some(hand) = self.hand_rows_with(rows.clone(), 0.0, None, &sd) else {
            return Meas {
                verdict,
                x: None,
                g: None,
            };
        };
        let (mut x, mut g) = (Acc::default(), Acc::default());
        for (j, per_out) in hand {
            let r = self.reference_with(rows, j);
            for o in 0..3 {
                for w in 0..3 {
                    for i in 0..self.d {
                        x.add(per_out[o][w][i], r.x[o][w].0[i], r.x[o][w].1[i]);
                    }
                }
                g.add(per_out[o][3][0], r.g[o].0, r.g[o].1);
            }
        }
        Meas {
            verdict,
            x: Some(x),
            g: Some(g),
        }
    }
}

/// swap で `an_coeff` が中間（`m_half` を通る）になる入力でも、`Lin` の上界が実際の f32 の二階を覆う（実際の子テープ〔hand〕と
/// f32 のモデルの双方。`lin_bound_covers_f32_deviation_for_triplet_sweep` と同じ式）。掃引の入力が本当に中間の入力であることを
/// `tri_is_tie` と `TriSetup::an_kind == 2` で固定する（既存の掃引は `an_coeff = 0` の swap 行しか通らない）。
#[test]
fn lin_bound_covers_f32_deviation_for_triplet_an_coeff_half_rows() {
    let mut rng = Rng(0x7e57_4133_a11c_0de5);
    let mut st = SweepStats::default();
    let (mut tried, mut used) = (0usize, 0usize);
    for d in [3usize, 8, 16, 32, 64] {
        for p in [2.0f32, 1.5, 3.0, 10.0] {
            for ratio in [1e-1f32, 1e-3, 1e-6] {
                for exp in [0i32, 13] {
                    let c = TriCase {
                        d,
                        ratio,
                        p,
                        swap: true,
                        exp,
                    };
                    let rows = tri_tie_rows(d, ratio, &mut rng);
                    tried += 1;
                    if !tri_is_tie(&rows, p) {
                        continue;
                    }
                    let up = (f64::from(pow2(exp)), None);
                    let setup = triplet_setup((&rows.0, &rows.1, &rows.2), 0.0, p, true, up);
                    if let Some(s) = &setup {
                        assert_eq!(s.an_kind, 2, "{c:?}: an_coeff は中間のはず");
                    }
                    let mut ok = lin_sweep_tri_row(&c, &rows, 0.0, None, &mut st).is_some();
                    if exp == 13 {
                        ok |= lin_sweep_tri_row(&c, &rows, 0.0, Some(3), &mut st).is_some();
                        ok |= lin_sweep_tri_row(&c, &rows, 0.0, Some(7), &mut st).is_some();
                    }
                    used += usize::from(ok);
                }
            }
        }
    }
    assert!(
        used >= 60,
        "an_coeff が中間で setup できた行が少なすぎる: {used}/{tried}"
    );
    assert!(
        st.checked > 1000,
        "検査した要素が少なすぎる: {}",
        st.checked
    );
    assert!(
        st.nonzero > 0,
        "実測の誤差が全て 0 では soundness を確かめていない"
    );
    eprintln!(
        "LINTRIHALF checked={} nonzero={} tight={} worst={:.4} used={used}/{tried}",
        st.checked, st.nonzero, st.tight, st.worst
    );
}

/// swap で `an_coeff` が中間になる行について、(1) 旧受理 → 新拒否は 0 件、(2) 旧拒否 → 新受理に変わった行の二階が f64 の
/// 閉じた式と REQ-2 で一致する（REQ-2 比 < 1・参照の不確かさ < 5%）。
#[test]
fn triplet_lin_an_coeff_half_rows_flip_only_to_accept_and_match_f64() {
    let mut rng = Rng(0x4133_c0ff_ee57_a11e);
    let (mut total, mut old_acc, mut flipped, mut max_ratio) = (0usize, 0usize, 0usize, 0f64);
    for d in [8usize, 16, 32, 64] {
        for p in [2.0f32, 1.5, 3.0] {
            for ratio in [1e-1f32, 1e-3, 1e-6] {
                for exp in [0i32, 13] {
                    let c = TriCase {
                        d,
                        ratio,
                        p,
                        swap: true,
                        exp,
                    };
                    let rows = tri_tie_rows(d, ratio, &mut rng);
                    if !tri_is_tie(&rows, p) {
                        continue;
                    }
                    let one = std::slice::from_ref(&rows);
                    let opts = (0.0, p, true);
                    let old = tri_multi_verdict(one, d, opts, Reduction::Sum, exp, true);
                    let new = tri_multi_verdict(one, d, opts, Reduction::Sum, exp, false);
                    total += 1;
                    let (o, n) = (old == Verdict::Accept, new == Verdict::Accept);
                    assert!(!(o && !n), "旧受理 → 新拒否 {c:?}: {new:?}");
                    old_acc += usize::from(o);
                    if !o && n {
                        flipped += 1;
                        let m = c.measure_rows(&rows);
                        for (label, acc) in [("x", m.x), ("g", m.g)] {
                            let a = acc.expect("計測");
                            assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
                            assert!(a.unc < 0.05, "{c:?}: {label} の参照の不確かさ {:e}", a.unc);
                            max_ratio = max_ratio.max(a.ratio);
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "LINTRIHALFFLIP total={total} old_accept={old_acc} flipped_to_accept={flipped} max_req2_ratio={max_ratio:.3e}"
    );
    assert!(total >= 50, "{total}");
    assert!(flipped >= 1, "受理に変わった行が無い: {flipped}");
}

/// 4 サンプル（ターゲット `0..4`）・Mean・`class_weight`・`ignore_index` の CEWO の x への二階の f64 参照。
/// 有効なサンプル `r` のブロックは `H[j][i] = (w_{t_r}/denom_w)·y_j(δ_ji − y_i)`、`denom_w` は有効サンプルの
/// `w_t` の和（ignore したサンプルは 0 行）。対角は `1 − y_j` を他クラスの和で作って桁落ちを避ける。上流は 1。
fn multi_ce_reference(
    z0: &[f32],
    c: usize,
    weights: &[f32],
    ignore: Option<i32>,
    j: usize,
) -> Vec<f64> {
    let valid = |r: usize| ignore != Some(r as i32);
    let denom_w: f64 = (0..4)
        .filter(|&r| valid(r))
        .map(|r| f64::from(weights[r]))
        .sum();
    let mut out = vec![0f64; z0.len()];
    let r = j / c;
    if !valid(r) {
        return out;
    }
    let x: Vec<f64> = z0[r * c..(r + 1) * c]
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let mx = x.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
    let e: Vec<f64> = x.iter().map(|&v| (v - mx).exp()).collect();
    let total: f64 = e.iter().sum();
    let y: Vec<f64> = e.iter().map(|v| v / total).collect();
    let jj = j % c;
    let rest_j = (total - e[jj]) / total;
    let rest_j = if 2.0 * e[jj] > total {
        e.iter()
            .enumerate()
            .filter(|&(i, _)| i != jj)
            .map(|(_, v)| *v)
            .sum::<f64>()
            / total
    } else {
        rest_j
    };
    let s = f64::from(weights[r]) / denom_w;
    for i in 0..c {
        out[r * c + i] = if i == jj {
            s * y[jj] * rest_j
        } else {
            -s * y[jj] * y[i]
        };
    }
    out
}

/// 複数サンプル（n = 4）・Mean（`denom_w`）・`ignore_index`・`class_weight` の行で、旧拒否 → 新受理に変わった行の二階が
/// f64 参照と REQ-2 で一致する（REQ-2 比 < 1）。旧受理 → 新拒否は 0 件。`denom_w`・ignore・重みの配線が受理を増やす方向に
/// ずれても（値が変わるため）通常 CI で検出できる。受理された行はすべて照合し、旧拒否 → 新受理の件数と最大比も出す。
#[test]
fn ce_lin_multi_sample_flipped_rows_match_f64() {
    let (mut total, mut accepted, mut flipped) = (0usize, 0usize, 0usize);
    let (mut max_ratio, mut max_ratio_flipped) = (0f64, 0f64);
    for c in [64usize, 100] {
        let weights: Vec<f32> = (0..c)
            .map(|i| 0.5 + 0.4 * ((i as f32) * 0.731).sin())
            .collect();
        for ignore in [None, Some(2i32), Some(0)] {
            for margins in [
                [10.0f32, 10.0, 10.0, 10.0],
                [0.0, 5.0, 10.0, 20.0],
                [8.0, 12.0, 10.0, 14.0],
            ] {
                let mut z0 = Vec::new();
                for (r, m) in margins.iter().enumerate() {
                    let mut row: Vec<f32> = pattern(c, r as f32).iter().map(|v| 0.1 * v).collect();
                    row[r] += m;
                    z0.extend(row);
                }
                let mut opts =
                    CrossEntropyOptions::default().class_weight(tt(weights.clone(), &[c]));
                if let Some(i) = ignore {
                    opts = opts.ignore_index(i);
                }
                probe::FORCE_OLD.with(|f| f.set(true));
                let old = run_public(z0.clone(), |tape, z| multi_build(tape, z, c, &opts)).0;
                probe::FORCE_OLD.with(|f| f.set(false));
                let new = run_public(z0.clone(), |tape, z| multi_build(tape, z, c, &opts)).0;
                total += 1;
                let (o, n) = (old == Verdict::Accept, new == Verdict::Accept);
                assert!(
                    !(o && !n),
                    "c={c} ignore={ignore:?} margins={margins:?}: 旧受理 → 新拒否 new={new:?}"
                );
                if !n {
                    continue;
                }
                accepted += 1;
                flipped += usize::from(!o);
                // 各サンプルのターゲット・隣・中央・末尾を seed にする。
                let idx: Vec<usize> = (0..4)
                    .flat_map(|r| [r * c + r, r * c + (r + 1) % c, r * c + c / 2, r * c + c - 1])
                    .collect();
                let rows = public_rows(z0.clone(), &idx, |tape, z| multi_build(tape, z, c, &opts));
                for (k, &j) in idx.iter().enumerate() {
                    let truth = multi_ce_reference(&z0, c, &weights, ignore, j);
                    for (i, &t) in truth.iter().enumerate() {
                        let ratio = req2_ratio(f64::from(rows[k][i]), t);
                        assert!(
                            ratio < 1.0,
                            "c={c} ignore={ignore:?} margins={margins:?} seed={j} i={i}: 実測 {} 真値 {t:e} 比 {ratio:e}",
                            rows[k][i]
                        );
                        max_ratio = max_ratio.max(ratio);
                        if !o {
                            max_ratio_flipped = max_ratio_flipped.max(ratio);
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "LINCEMULTIF total={total} accepted={accepted} flipped_to_accept={flipped} max_req2_ratio={max_ratio:.3e} max_req2_ratio_flipped={max_ratio_flipped:.3e}"
    );
    assert!(flipped >= 6, "受理に変わった行が少なすぎる: {flipped}");
}

/// 単一サンプルの CE・CEWO の行列（`ce_switch_cases` のうち `C ≤ 1000`）で、旧受理 → 新拒否は 0 件、旧拒否 → 新受理に変わった
/// 行は REQ-2 比 < 1 かつ参照の不確かさ < 5%。`ce_lin_switch_rejection_delta_measurement`（`--release`・全 396 行）の
/// 通常テスト版（`triplet_lin_flipped_rows_match_f64` と同型）。
#[test]
fn ce_lin_switch_rows_up_to_c1000_never_flip_to_reject_and_flipped_match_f64() {
    let (mut total, mut old_acc, mut flipped, mut max_ratio) = (0usize, 0usize, 0usize, 0f64);
    for c in ce_switch_cases().into_iter().filter(|c| c.c <= 1000) {
        let old = public_old_ce(&c);
        let new = c.public().0;
        total += 1;
        let (o, n) = (old == Verdict::Accept, new == Verdict::Accept);
        assert!(!(o && !n), "旧受理 → 新拒否 {c:?}: {new:?}");
        old_acc += usize::from(o);
        if !o && n {
            flipped += 1;
            let m = c.measure();
            let (x, g) = (m.x.expect("x"), m.g.expect("g"));
            for (label, a) in [("x", x), ("g", g)] {
                assert!(a.ratio < 1.0, "{c:?}: {label} の REQ-2 比 {:e}", a.ratio);
                assert!(a.unc < 0.05, "{c:?}: {label} の参照の不確かさ {:e}", a.unc);
                max_ratio = max_ratio.max(a.ratio);
            }
        }
    }
    eprintln!(
        "LINCESWN total={total} old_accept={old_acc} flipped_to_accept={flipped} max_req2_ratio={max_ratio:.3e}"
    );
    assert!(total >= 250, "{total}");
    assert!(flipped >= 20, "受理に変わった行が少なすぎる: {flipped}");
}

// =====================================================================================
// #4132: 拒否のまま残った 4 候補の `Lin` の上界を項ごとに分解する計測（設計 §82.15）
// =====================================================================================

/// 上界の内訳の計測の 1 要素分の記録（`entry_check_lin` と同じ式の入力）。
struct BdRow {
    /// 要素の識別（`g,j` / `x,j,i` / `g,o,j` / `x,wrt,i,j`）。run 間で添字を揃えて突き合わせる。
    key: String,
    v: f64,
    s: f32,
    bound: f64,
    /// 種別ごとの上界（添字は `lin::BD_*`）。
    bd: [f64; super::super::lin::BD_KINDS],
    /// 判定時点の余裕係数の `slack = 16·N·u`。
    slack: f64,
    /// 判定時点の事象数 `N`（アブレーションの run 間で一致することを検査する）。
    events: usize,
}

impl BdRow {
    fn new(key: String, l: Lin, s: f32) -> Self {
        use super::super::lin::{LIN_U, lin_breakdown, lin_events, lin_is_sparse};
        // 計測は厳密モード前提（疎モードの `pool_small_terms` は μ に依存して畳む項を選び、N を変えうるため）。
        assert!(!lin_is_sparse(), "計測は厳密モードで行う");
        BdRow {
            key,
            v: l.v,
            s,
            bound: l.bound(),
            bd: lin_breakdown(l),
            slack: 16.0 * lin_events() as f64 * LIN_U,
            events: lin_events(),
        }
    }
    /// 閾値（`entry_check_lin` と同じ。絶対救済と相対の半分の大きい方）。
    fn thr(&self) -> f64 {
        ABSOLUTE_RESCUE_THRESHOLD.max((RELATIVE_TOLERANCE / 2.0) * self.v.abs())
    }
    /// `err/閾値`。`sv`・`slack` が偽ならその項を 0 にする。
    fn margin(&self, sv: bool, slack: bool) -> f64 {
        let d = if sv {
            (f64::from(self.s) - self.v).abs()
        } else {
            0.0
        };
        let k = if slack { self.slack } else { 0.0 };
        (d + self.bound * (1.0 + k)) / self.thr()
    }
}

/// CE の `g` と対角（`seed_list` の各 seed）を `Lin` で走査して記録する（`ce_lin_scan_timed` と同じ演算）。
fn ce_bd_rows(c: &CeCase, seed_list: &[usize]) -> Vec<BdRow> {
    use super::super::lin::lin_truncate;
    let (logits, a_s, b, ctx) = ce_lin_inputs(c);
    let mut st = CeLinState::new(&logits, a_s, ctx);
    let mut out = Vec::new();
    for &j in seed_list {
        let (mark, sa, sb) = st.seed(j, b[j]);
        out.push(BdRow::new(format!("g,{j}"), sa.g, sb.g));
        let l = cewo_x(&st.lin, &sa, j, j);
        let s = cewo_x(&st.f32m, &sb, j, j);
        out.push(BdRow::new(format!("x,{j},{j}"), l, s));
        lin_truncate(mark);
    }
    out
}

/// Triplet の `g` と 9 ブロックの全レーンを `Lin` で走査して記録する（`tri_lin_scan_timed` と同じ演算）。
fn tri_bd_rows(c: &TriCase, seed_list: &[usize]) -> Vec<BdRow> {
    use super::super::lin::lin_truncate;
    let rows = c.inputs();
    let up = (f64::from(pow2(c.exp)), None);
    let setup = triplet_setup((&rows.0, &rows.1, &rows.2), 0.0, c.p, c.swap, up).expect("setup");
    let mut st = TriLinState::new(setup);
    let mut out = Vec::new();
    for &j in seed_list {
        for o in 0..3 {
            let (mark, sa, sb) = st.seed(o, j);
            out.push(BdRow::new(format!("g,{o},{j}"), sa.g, sb.g));
            for wrt in 0..3 {
                for i in 0..c.d {
                    let l = tri_x(&st.st, &st.lin, &sa, wrt, (i, j));
                    let s = tri_x(&st.st, &st.f32m, &sb, wrt, (i, j));
                    out.push(BdRow::new(format!("x,{o},{wrt},{i},{j}"), l, s));
                    lin_truncate(mark);
                }
            }
        }
    }
    out
}

/// 1 候補の内訳を出す。`collect` を「通常」と「ある種別の `μ` を 0 にして forward から計算し直した」run で呼び、
/// 通常の run で拒否された要素ごとに `err/閾値` の変化を 1 行（`LINBD`）、候補ごとの要約を `LINBDSUM` に出す。
fn lin_bound_breakdown_report(tag: &str, collect: impl Fn() -> Vec<BdRow>) {
    use super::super::lin::{BD_CONST, BD_LIBM_REM, BD_LIBM_ULP, BD_ROUND, BD_TOTAL, BdAblate};
    // (名前, 0 にする種別, `|s−v|` を残すか, 余裕係数を残すか)
    let scenarios: [(&str, &[u8], bool, bool); 10] = [
        ("base", &[], true, true),
        ("no_total", &[BD_TOTAL], true, true),
        ("no_round", &[BD_ROUND], true, true),
        ("no_const", &[BD_CONST], true, true),
        ("no_libm_ulp", &[BD_LIBM_ULP], true, true),
        ("no_libm_rem", &[BD_LIBM_REM], true, true),
        ("no_libm", &[BD_LIBM_ULP, BD_LIBM_REM], true, true),
        ("no_slack", &[], true, false),
        ("no_sv", &[], false, true),
        (
            "only_total",
            &[BD_ROUND, BD_CONST, BD_LIBM_ULP, BD_LIBM_REM],
            false,
            false,
        ),
    ];
    let runs: Vec<Vec<BdRow>> = scenarios
        .iter()
        .map(|&(_, kinds, _, _)| {
            let _g = BdAblate::new(kinds);
            collect()
        })
        .collect();
    let base = &runs[0];
    for r in &runs {
        assert_eq!(r.len(), base.len(), "{tag}: run 間で要素数が違う");
        // 計測契約（§82.15.1）: 事象は 0 の μ のまま数えるので、どのアブレーションでも N は base と一致する。
        for (i, (x, b)) in r.iter().zip(base).enumerate() {
            assert_eq!(x.key, b.key, "{tag}: 要素の対応がずれた i={i}");
            assert_eq!(
                x.events, b.events,
                "{tag}: アブレーションで事象数 N が変わった elem={}",
                b.key
            );
        }
    }
    let margin = |si: usize, i: usize| {
        let (_, _, sv, sl) = scenarios[si];
        runs[si][i].margin(sv, sl)
    };
    let is_rej = |si: usize, i: usize| {
        let m = margin(si, i);
        !m.is_finite() || m >= 1.0 || runs[si][i].slack >= 0.01
    };
    let names = ["round", "const", "total", "libm_ulp", "libm_rem", "other"];
    for i in 0..base.len() {
        if !is_rej(0, i) {
            continue;
        }
        let r = &base[i];
        let thr = r.thr();
        let mut line = format!(
            "LINBD cand={tag} elem={} v={:e} s={:e} thr={thr:e} margin={:.4} sv={:.4} slack_part={:.4} bound={:e}",
            r.key,
            r.v,
            r.s,
            margin(0, i),
            (f64::from(r.s) - r.v).abs() / thr,
            r.bound * r.slack / thr,
            r.bound,
        );
        for (k, n) in names.iter().enumerate() {
            line += &format!(
                " bd_{n}={:.4} frac_{n}={:.4}",
                r.bd[k] / thr,
                r.bd[k] / r.bound
            );
        }
        for si in 1..scenarios.len() {
            line += &format!(" {}={:.4}", scenarios[si].0, margin(si, i));
        }
        eprintln!("{line}");
    }
    for (si, (name, ..)) in scenarios.iter().enumerate() {
        let (mut mx, mut rej) = (0f64, 0usize);
        // 通常の run で拒否された要素の中での最大（その要素を救えたかが分かる）と、全要素での最大・拒否数。
        let mut mx_rej_set = 0f64;
        for i in 0..base.len() {
            let m = margin(si, i);
            if m.is_finite() {
                mx = mx.max(m);
                if is_rej(0, i) {
                    mx_rej_set = mx_rej_set.max(m);
                }
            }
            if is_rej(si, i) {
                rej += 1;
            }
        }
        eprintln!(
            "LINBDSUM cand={tag} scenario={name} checked={} rejected={rej} max_margin_all={mx:.4} max_margin_base_rejected_set={mx_rej_set:.4}",
            base.len()
        );
    }
}

/// `LINBD`: 拒否のまま残った 4 候補（CE の C=1000 CEWO-smooth・C=2100 既定 CE、Triplet の d=64・p=3・swap なしと
/// d=64・r=1e-6・p=10・swap なし）の `Lin` の上界を項ごとに分解する（設計 §82.15・イシュー #4132）。判定は変えず、
/// `#[cfg(test)]` の事象の種別と、種別ごとに `μ` を 0 にして計算し直す切り替え（`lin::BdAblate`）で測る。
/// `cargo test -p fandhe-ai-autodiff --release --lib lin_bound_breakdown_for_followup_candidates_measurement -- --ignored --nocapture`
#[test]
#[ignore = "計測。--release で実行する"]
fn lin_bound_breakdown_for_followup_candidates_measurement() {
    for c in ce_followup_candidates().into_iter().filter(|c| c.c > 100) {
        let sd: Vec<usize> = (0..c.c).collect();
        lin_bound_breakdown_report(&format!("{c:?}").replace(' ', ""), || ce_bd_rows(&c, &sd));
    }
    let cs = triplet_followup_candidates();
    for k in [0usize, 4] {
        let c = cs[k];
        let sd: Vec<usize> = (0..c.d).collect();
        lin_bound_breakdown_report(&format!("{c:?}").replace(' ', ""), || tri_bd_rows(&c, &sd));
    }
}
