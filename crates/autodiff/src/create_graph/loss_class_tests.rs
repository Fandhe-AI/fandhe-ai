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
        const EPSILON: f64 = 1e-12;
        let d = self.d;
        let shape = [1usize, d];
        let (a, b) = self.ab();
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
        let s_b =
            super::super::loss_scale_bcast(&child, &g, Reduction::Sum, false, 1, &shape).unwrap();
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

/// 後続の既定 CE・CEWO の切り替え（#4093・#4094）で受理に変える入力。現在は「相殺判定の `coeff·m` の項だけ」
/// で拒否されるが、実際の f32 の二階は f64 の閉じた式と REQ-2 の閾値の 1/4 以内で一致する。
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
    let check = |name: &str, m: Meas, want_flag: bool| {
        match &m.verdict {
            Verdict::Cancel { cls, flag, .. } => {
                assert_eq!(*cls, "coeff·m のみ", "{name}: 拒否の分類");
                assert_eq!(*flag, want_flag, "{name}: flag 経路");
            }
            v => panic!("{name}: 現在は相殺判定で拒否されるはず: {v:?}"),
        }
        for (label, acc) in [("x", m.x), ("g", m.g)] {
            let a = acc.unwrap_or_else(|| panic!("{name}: {label} が未測定"));
            assert!(a.ratio < 0.25, "{name}: {label} の REQ-2 比 {:e}", a.ratio);
            assert!(a.unc < 0.05, "{name}: {label} の参照の不確かさ {:e}", a.unc);
        }
    };
    for c in ce_followup_candidates() {
        check(&format!("{c:?}"), c.measure(), false);
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
