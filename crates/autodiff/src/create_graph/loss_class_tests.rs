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
        let (a, p, n) = self.inputs();
        let to = |s: &[f32]| -> Vec<f64> { s.iter().map(|&v| f64::from(v)).collect() };
        (to(&a), to(&p), to(&n))
    }

    /// `(c_an, c_pn)`（`triplet_cgrad` と同じく swap でなければ `(1, 0)`）。
    fn coefs(&self) -> (f64, f64) {
        let (a, p, n) = self.f64_inputs();
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
        let d = self.d;
        let shape = [1usize, d];
        let (a32, p32, n32) = self.inputs();
        let (a64, p64v, n64) = self.f64_inputs();
        let p64 = f64::from(self.p);
        let sb = f64::from(pow2(self.exp));
        let params = crate::eval::TripletDistanceParams {
            p: p64,
            eps: 0.0,
            swap: self.swap,
        };
        let (stats, diff_ap, diff_an, diff_pn) =
            crate::eval::triplet_distance_stats(&a64, &p64v, &n64, 0, d, &params);
        let row32 =
            |x: &[f32], y: &[f32]| -> Vec<f32> { (0..d).map(|k| (x[k] - y[k]) + 0.0f32).collect() };
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
        let s_b =
            super::super::loss_scale_bcast(&child, &g, Reduction::Sum, false, 1, &shape).ok()?;
        let eps_c = scalar(&child, 0.0);
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
        let (a, p, n) = self.f64_inputs();
        let sub =
            |x: &[f64], y: &[f64]| -> Vec<f64> { x.iter().zip(y).map(|(a, b)| a - b).collect() };
        let (vap, van, vpn) = (sub(&a, &p), sub(&a, &n), sub(&p, &n));
        let pe = f64::from(self.p);
        let (c_an, c_pn) = self.coefs();
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

/// 後続の Triplet の切り替え（#4095・#4096）で受理に変える入力。最後の 1 件は `flag` 経路（f32 再現値で
/// 組む経路）で拒否される入力。
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
    for (k, c) in triplet_followup_candidates().into_iter().enumerate() {
        assert_eq!(c.coefs().1 == 1.0, c.swap, "{c:?}: swap ありは c_pn を通す");
        check(&format!("{c:?}"), c.measure(), k == 4);
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
