//! `Op::BatchNorm` の `create_graph` 対応（イシュー #3079・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §28）の総括的な適用判定を正のプローブで確かめる
//! 統合テスト。
//!
//! BatchNorm の二階構造は `LayerNorm` と同じ S1 構造をチャネル行 `[c, n·spatial]` 上に組み、構造を
//! 積む前に `create_graph_norm::check_structure_stages`（#3078 で導入した構造段の上界判定を
//! `w`・`b` の軸と eval の固定統計へ拡張したもの）を通す。PR #3348 のレビュー指摘（Codex P1 × 8・
//! Cursor Bugbot 1 件）はいずれも「有限な入力・upstream で構造の中間値が f32 を超える」類型で、
//! ここではその全例について「受理なら 1 階・`cg.grad`・二階（seed の各要素の絶対値 ≤ 1）・孫テープ
//! 再生が有限かつ一致」「拒否なら `Err(Backward)` で `child` が空」を固定し、train／eval・
//! weight／bias の有無・微分対象 `x`／`w`／`b`・upstream の大きさを掛け合わせた掃引で同じ契約を
//! 確かめる（判定を外すと受理側の有限性検査が落ちる）。
//!
//! Hessian の有限差分突合・1 階一致・running stats 不変・閉包性などの基本的な受け入れ条件は
//! `tests/create_graph.rs` の BatchNorm 節が持つ。

#![allow(clippy::needless_range_loop)]

mod common;

use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(v: &Tensor<f32>) -> Vec<f32> {
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

fn assert_finite(v: &[f32], ctx: &str) {
    assert!(v.iter().all(|a| a.is_finite()), "{ctx}: 非有限 {v:?}");
}

fn assert_req2(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: 長さ");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(x), f64::from(y)),
            "{ctx}[{i}]: {x} vs {y}"
        );
    }
}

/// 非一様な固定定数。
fn pattern(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.9).sin() + 0.35 * (i as f32 + 1.0).cos() + 0.2)
        .collect()
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

fn alternating(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn scaled(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|a| a * s).collect()
}

const EPS: f32 = 1e-5;

/// 正のプローブの結果。
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Outcome {
    Accepted,
    Rejected,
}

/// BatchNorm のモード。`Eval` は固定統計（`mean`・`var`。長さ `c`）。
#[derive(Clone, Debug)]
enum Mode {
    Train,
    Eval { mean: Vec<f32>, var: Vec<f32> },
}

/// 二階の seed を掛ける 1 階勾配（`cg.grad` の対象）。
#[derive(Clone, Copy, Debug)]
enum Wrt {
    X,
    W,
    B,
}

/// `w`／`b` の扱い。`Tracked` は追跡ありの葉、`Const` は追跡なしの定数。
#[derive(Clone, Debug)]
enum Affine {
    Tracked(Vec<f32>),
    Const(Vec<f32>),
}

/// 正のプローブの 1 ケース。損失は `sum(bn(x; w, b) ⊙ c)`（`c` が upstream `g`）、`quadratic` なら
/// `sum(bn ⊙ bn ⊙ c)`（upstream `g = 2c·bn` が入力に依存する）。二階は `sum(cg.grad(wrt) ⊙ seed)`
/// を child 上で逆伝播する。
#[derive(Clone, Debug)]
struct Probe {
    mode: Mode,
    quadratic: bool,
    x0: Vec<f32>,
    shape: Vec<usize>,
    c: Vec<f32>,
    w: Option<Affine>,
    b: Option<Affine>,
    wrt: Wrt,
    seed: Vec<f32>,
}

fn affine_var<'t>(tape: &'t Tape, a: &Option<Affine>, ch: usize) -> Option<Var<'t>> {
    a.as_ref().map(|a| match a {
        Affine::Tracked(v) => tape.var(&t(v.clone(), &[ch])),
        Affine::Const(v) => tape.var_no_grad(&t(v.clone(), &[ch])),
    })
}

/// 判定の契約を 1 ケースについて確かめ、受理／拒否を返す。
fn probe(p: &Probe) -> Outcome {
    let ctx = format!("{p:?}");
    let ch = p.shape[1];
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(p.x0.clone(), &p.shape));
    let w = affine_var(&tape, &p.w, ch);
    let b = affine_var(&tape, &p.b, ch);
    let y = match &p.mode {
        Mode::Train => x.batch_norm(w.as_ref(), b.as_ref(), EPS),
        Mode::Eval { mean, var } => x.batch_norm_infer(
            w.as_ref(),
            b.as_ref(),
            &t(mean.clone(), &[ch]),
            &t(var.clone(), &[ch]),
            EPS,
        ),
    }
    .expect("forward");
    let cc = tape.var_no_grad(&t(p.c.clone(), &p.shape));
    let yc = if p.quadratic { y.mul(&y).unwrap() } else { y };
    let loss = yc.mul(&cc).unwrap().sum(None).unwrap();
    let (target, target_shape) = match p.wrt {
        Wrt::X => (x, p.shape.clone()),
        Wrt::W => (w.expect("wrt=W には w が要る"), vec![ch]),
        Wrt::B => (b.expect("wrt=B には b が要る"), vec![ch]),
    };

    let cg = match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => cg,
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: {e:?}");
            assert!(child.is_empty(), "{ctx}: 拒否時に child が空へ戻る");
            return Outcome::Rejected;
        }
    };

    // 1 階と cg.grad。
    let plain = dense(tape.backward(&loss).unwrap().get(&target).unwrap().unwrap());
    assert_finite(&plain, &format!("{ctx}: 1 階"));
    let gt = cg.grad(&target).unwrap().unwrap();
    let gt_v = dense(&gt.to_tensor());
    assert_finite(&gt_v, &format!("{ctx}: cg.grad"));
    assert_req2(&gt_v, &plain, &format!("{ctx}: cg.grad vs 1 階"));

    // 二階: 追跡ありの境界ノード（x・w・b の写し）すべてへの勾配が有限。`cg.grad` が追跡ありの
    // 写しへ依存しない場合（線形損失の db・eval の dx で w が定数など）は二階が恒等ゼロで、
    // seed 付きの和は追跡なしになり `child.backward` が「追跡対象の祖先を持たない」で拒否するため
    // 対象外（`Var::requires_grad` は crate 外から呼べないのでエラー文言で識別する）。
    let sv = child.var_no_grad(&t(p.seed.clone(), &target_shape));
    let prod = gt.mul(&sv).unwrap().sum(None).unwrap();
    let mut leaves = vec![cg.child_var(&x).unwrap().unwrap()];
    for v in [w, b].into_iter().flatten() {
        if let Ok(Some(cv)) = cg.child_var(&v) {
            leaves.push(cv);
        }
    }
    let grads = match child.backward(&prod) {
        Ok(g) => g,
        Err(AutodiffError::Backward(msg)) if msg.contains("勾配追跡対象の祖先を持たない") =>
        {
            return Outcome::Accepted;
        }
        Err(e) => panic!("{ctx}: child.backward の失敗 {e:?}"),
    };
    let hv: Vec<Option<Vec<f32>>> = leaves
        .iter()
        .map(|l| grads.get(l).unwrap().map(dense))
        .collect();
    for v in hv.iter().flatten() {
        assert_finite(v, &format!("{ctx}: 二階（child.backward）"));
    }

    // 孫テープ再生: 同じ二階勾配が有限かつ child と一致。child 上の二階のグラフが BatchNorm の写し
    // （quadratic の `g = 2c·bn`）を含むと孫テープ側でも同じ判定が走り、拒否された場合も契約どおり
    // `Err(Backward)` で grand が空になる。
    let grand = Tape::new_with_ops(common::naive_ops());
    let cg2 = match child.backward_create_graph(&prod, &grand) {
        Ok(cg2) => cg2,
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: 孫 {e:?}");
            assert!(grand.is_empty(), "{ctx}: 孫テープ再生の拒否時に grand が空");
            return Outcome::Accepted;
        }
    };
    // train の dw は正規化と同じ行（チャネル）で縮約するため、入力依存の損失（`g = 2c·bn`）では
    // `∂dw/∂x` が「`xhat` のスケール不変性（`Σxhat = 0`・`Σxhat² ≈ m`）で打ち消し合う `c` 規模の
    // 項の残差」になり、真値は `c·eps/var` 程度まで小さくなりうる。この桁落ちでは f32 の丸め誤差が
    // 真値を上回りうるので、子テープ（`grad.rs` の VJP）と孫テープ（子テープ上の `Var` 合成）の
    // 演算順の違いだけで REQ-2 判定を外れる。オーバーフローとは別の f32 精度の限界のため、
    // この組み合わせの x の写しは有限性のみを確かめる（決定記録 §28）。
    let ill_conditioned = p.quadratic && matches!(p.mode, Mode::Train) && matches!(p.wrt, Wrt::W);
    for (k, (l, h1)) in leaves.iter().zip(&hv).enumerate() {
        if let (Some(g2), Some(h1)) = (cg2.grad(l).unwrap(), h1) {
            let v = dense(&g2.to_tensor());
            assert_finite(&v, &format!("{ctx}: 孫テープ再生"));
            if !(ill_conditioned && k == 0) {
                assert_req2(&v, h1, &format!("{ctx}: 孫 vs child"));
            }
        }
    }
    Outcome::Accepted
}

/// `[n, 1]`（チャネル 1）の train プローブ。
fn train_col(x0: Vec<f32>, c: Vec<f32>) -> Probe {
    let n = x0.len();
    Probe {
        mode: Mode::Train,
        quadratic: false,
        x0,
        shape: vec![n, 1],
        c,
        w: None,
        b: None,
        wrt: Wrt::X,
        seed: ones(n),
    }
}

fn eval_col(x0: Vec<f32>, mean: f32, var: f32, c: Vec<f32>) -> Probe {
    Probe {
        mode: Mode::Eval {
            mean: vec![mean],
            var: vec![var],
        },
        ..train_col(x0, c)
    }
}

fn with_w(p: Probe, wrt: Wrt) -> Probe {
    Probe {
        w: Some(Affine::Tracked(vec![1.0])),
        seed: match wrt {
            Wrt::X => p.seed.clone(),
            _ => ones(1),
        },
        wrt,
        ..p
    }
}

/// PR #3348 のレビューで指摘された入力そのものと、判定後の期待結果。拒否は「構造の中間値が f32 を
/// 超えうる」ことを判定が検出した結果で、受理は構造の全段が安全域に収まる入力（いずれも
/// 「受理なら全段が有限で一致・拒否なら child 空」を [`probe`] が検査する）。
fn review_examples() -> Vec<(&'static str, Probe, Outcome)> {
    use Outcome::{Accepted, Rejected};
    vec![
        (
            "Codex 1: eval x=mean=3e38・var=0・追跡 w=1・loss=sum(y)・∂dw/∂x",
            with_w(eval_col(vec![3e38; 2], 3e38, 0.0, ones(2)), Wrt::W),
            Accepted,
        ),
        (
            "Codex 1（:221 同型）: train x=[3e38,3e38]・追跡 w=1・loss=sum(y)・∂dw/∂x",
            with_w(train_col(vec![3e38; 2], ones(2)), Wrt::W),
            Accepted,
        ),
        (
            "Codex 3: train x=[0,0]・w=1・g=[1e38,1e38]・cg.grad(x).sum の二階",
            with_w(train_col(vec![0.0; 2], vec![1e38; 2]), Wrt::X),
            Rejected,
        ),
        (
            "Codex 4: eval x=mean=3e38・var=0・追跡 w=1・g=1e20・∂dw/∂x",
            with_w(eval_col(vec![3e38; 2], 3e38, 0.0, vec![1e20; 2]), Wrt::W),
            Accepted,
        ),
        (
            "Codex 4（train 同型）: train x=[3e38,3e38]・追跡 w=1・g=1e20・∂dw/∂x",
            with_w(train_col(vec![3e38; 2], vec![1e20; 2]), Wrt::W),
            Accepted,
        ),
        (
            "Codex 5: train x=[3e38,-3e38]・w=1・g=1e38（中心化 x_i−x_0 が -inf）",
            with_w(train_col(vec![3e38, -3e38], vec![1e38; 2]), Wrt::X),
            Rejected,
        ),
        (
            "Codex 6: train x=[0,0]・w=1・g=1e38・cg.grad(w).sum の二階",
            with_w(train_col(vec![0.0; 2], vec![1e38; 2]), Wrt::W),
            Rejected,
        ),
        (
            "Codex 7: eval [3,1]・x=mean=0・追跡 b=1・loss=sum(y²·[1e38,1e38,-1e38])",
            Probe {
                quadratic: true,
                b: Some(Affine::Tracked(vec![1.0])),
                wrt: Wrt::B,
                seed: ones(1),
                ..eval_col(vec![0.0; 3], 0.0, 1.0, vec![1e38, 1e38, -1e38])
            },
            Rejected,
        ),
        (
            "Codex 8: train [3,1]・x=[3e38,-3e38,-3e38]・w=1・g=1e38（中心化の平均の縮約）",
            with_w(train_col(vec![3e38, -3e38, -3e38], vec![1e38; 3]), Wrt::X),
            Rejected,
        ),
        (
            "Cursor: train 同符号の巨大入力 x=[3e38,2.9e38]・追跡 w=1・∂dw/∂x（前置スケール不要の行）",
            with_w(train_col(vec![3e38, 2.9e38], ones(2)), Wrt::W),
            Accepted,
        ),
        (
            "先回り: eval x=2e38・mean=-2e38（x − mean が f32 を超える）・追跡 w",
            with_w(eval_col(vec![2e38, 1e38], -2e38, 1e30, ones(2)), Wrt::W),
            Rejected,
        ),
        (
            "先回り: train [4,1] g=±1e38（dx の gs 復元・行方向和）",
            with_w(
                train_col(vec![1.0, 2.0, 3.0, 4.5], scaled(&alternating(4), 1e38)),
                Wrt::X,
            ),
            Rejected,
        ),
    ]
}

#[test]
fn review_examples_match_expected_outcome() {
    for (label, p, expected) in review_examples() {
        assert_eq!(probe(&p), expected, "{label}");
    }
}

#[test]
fn rejected_review_examples_at_moderate_scale_are_accepted() {
    // upstream 起因で拒否された例は、同じ入力で upstream を 1e30 倍へ下げると判定を通り、全段が
    // 有限かつ一致する（拒否が過剰に広くないことの正のプローブ）。x 起因（中心化の overflow）の
    // 例は x を 1e30 倍へ下げて確かめる。
    for (label, mut p, expected) in review_examples() {
        if expected == Outcome::Accepted {
            continue;
        }
        let peak_c = p.c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if peak_c > 1e30 {
            p.c = scaled(&p.c, 1e30 / peak_c);
        }
        // x（と eval の mean）は同じ倍率で絶対値 1e30 以下へ縮める（x − mean の比を保つ）。
        let mut peak_x = p.x0.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if let Mode::Eval { mean, .. } = &p.mode {
            peak_x = mean.iter().fold(peak_x, |m, v| m.max(v.abs()));
        }
        if peak_x > 1e30 {
            let f = 1e30 / peak_x;
            p.x0 = scaled(&p.x0, f);
            if let Mode::Eval { mean, .. } = &mut p.mode {
                *mean = scaled(mean, f);
            }
        }
        assert_eq!(probe(&p), Outcome::Accepted, "{label}（縮小版）");
    }
}

/// eval の `x=mean=3e38`・`var=0` は `x − mean = 0` で `∂dw/∂x = g·rstd`（`rstd = 1/√eps ≈ 316.23`）が
/// 有限な正解を持つ（Codex 1・4 の値の突合）。
#[test]
fn eval_huge_constant_input_mixed_hessian_value() {
    for (g, expected) in [(1.0f32, 316.227_77f64), (1e20, 3.162_277_7e22)] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![3e38; 2], &[2, 1]));
        let w = tape.var(&t(vec![1.0], &[1]));
        let y = x
            .batch_norm_infer(
                Some(&w),
                None,
                &t(vec![3e38], &[1]),
                &t(vec![0.0], &[1]),
                EPS,
            )
            .unwrap();
        let k = tape.var_no_grad(&t(vec![g; 2], &[2, 1]));
        let loss = y.mul(&k).unwrap().sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let s2 = cg.grad(&w).unwrap().unwrap().sum(None).unwrap();
        let hv = child.backward(&s2).unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();
        let h = dense(hv.get(&cx).unwrap().expect("dw は x に依存する"));
        for v in h {
            assert!(
                common::req2_close(f64::from(v), expected),
                "g={g}: ∂dw/∂x = {v}（期待 {expected}）"
            );
        }
    }
}

/// Codex 7 の論点（孫テープ再生でホスト補正値が失われる）: 判定を通る入力でも、構造側の f32
/// 逐次和とホスト関数の f64 縮約は丸めが異なりうる（`g=[2^24,1,1]` は f32 逐次和が `2^24`、
/// f64 縮約は `2^24+2`）。`db` の値は子テープでも孫テープ再生でもホスト関数の値（1 階と bit 同一）
/// を保つ（`Op::BatchNormGradHost` が再生時にホスト関数で再計算する）。
#[test]
fn host_db_value_is_kept_in_grandchild_replay() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0; 3], &[3, 1]));
    let b = tape.var(&t(vec![0.5], &[1]));
    let y = x
        .batch_norm_infer(
            None,
            Some(&b),
            &t(vec![0.0], &[1]),
            &t(vec![1.0], &[1]),
            EPS,
        )
        .unwrap();
    // y = b = 0.5 なので upstream g = 2·y·k = k。
    let k = tape.var_no_grad(&t(vec![16_777_216.0, 1.0, 1.0], &[3, 1]));
    let loss = y.mul(&y).unwrap().mul(&k).unwrap().sum(None).unwrap();
    let plain = dense(tape.backward(&loss).unwrap().get(&b).unwrap().unwrap());
    assert_eq!(plain, vec![16_777_218.0], "1 階の db は f64 縮約");
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let db = cg.grad(&b).unwrap().expect("db");
    assert_eq!(
        dense(&db.to_tensor()),
        plain,
        "子テープの db は 1 階と bit 同一"
    );
    let s = db.mul(&db).unwrap().sum(None).unwrap();
    let cg2 = child.backward_create_graph(&s, &grand).unwrap();
    let db_grand = cg2.child_var(&db).unwrap().expect("孫テープ上の db の写し");
    assert_eq!(
        dense(&db_grand.to_tensor()),
        plain,
        "孫テープ再生の db もホスト関数の値（構造側の f32 逐次和 2^24 ではない）"
    );
    let cb = cg.child_var(&b).unwrap().unwrap();
    let g2 = dense(&cg2.grad(&cb).unwrap().expect("d s / d b").to_tensor());
    assert_finite(&g2, "孫テープの d s / d b");
}

/// 掃引用の 1 チャネルぶんの値（長さ `m`）。定数・交互符号・低分散・小スケール・大振幅を含む。
fn sweep_channel_values(m: usize) -> Vec<Vec<f32>> {
    vec![
        vec![0.75; m],
        alternating(m),
        (0..m)
            .map(|i| if i % 2 == 0 { 1.0 } else { 1.001 })
            .collect(),
        vec![1e-3; m],
        (0..m)
            .map(|i| [1e20, 2e20, -1e20].get(i).copied().unwrap_or(0.5))
            .collect(),
    ]
}

/// チャネル行の値 `vals`（チャネル `ch` は `vals * (1 + 0.25·ch)`）を `[n, c, spatial]` の並びへ置く。
fn place_channels(vals: &[f32], shape: &[usize]) -> Vec<f32> {
    let (n, c) = (shape[0], shape[1]);
    let spatial: usize = shape[2..].iter().product();
    let mut out = vec![0.0f32; n * c * spatial];
    for b in 0..n {
        for ch in 0..c {
            for s in 0..spatial {
                out[b * c * spatial + ch * spatial + s] =
                    vals[b * spatial + s] * (1.0 + 0.25 * ch as f32);
            }
        }
    }
    out
}

/// affine 構成と、二階の seed を掛けられる微分対象。
fn sweep_affines(w0: &[f32]) -> Vec<(Option<Affine>, Option<Affine>, Vec<Wrt>)> {
    vec![
        (None, None, vec![Wrt::X]),
        (
            Some(Affine::Tracked(w0.to_vec())),
            None,
            vec![Wrt::X, Wrt::W],
        ),
        (Some(Affine::Const(w0.to_vec())), None, vec![Wrt::X]),
        (
            Some(Affine::Tracked(w0.to_vec())),
            Some(Affine::Tracked(pattern(w0.len(), 5.0))),
            vec![Wrt::X, Wrt::W, Wrt::B],
        ),
    ]
}

#[test]
fn sweep_every_structure_path_is_finite_or_rejected() {
    // train／eval（通常分散・分散 0）× rank 2／3／4 × weight なし／あり（0 と非ゼロ）／定数・bias ×
    // チャネル値 × upstream の大きさ × 線形／入力依存の損失 × upstream と seed の符号パターンを
    // 掛け合わせ、どのケースも「受理なら全段が有限で一致」「拒否なら child 空」を満たすことを
    // 確かめる。判定（`check_structure_stages`）を外すと、巨大 upstream の受理ケースで構造の中間値
    // が overflow し有限性検査が落ちる。
    let scales = [1.0f32, 1e30, 1e36, 1e37, 3e37, 8e37, 3e38];
    // quadratic（g = 2c·bn）は上限を下げる。判定が守るのは BatchNorm の構造と、そこから g へ出る
    // cotangent までで、g を作る上流の Mul の逆伝播は対象外（決定記録 §27.3・§28）。
    let quadratic_scales = [1.0f32, 1e20, 1e30];
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for shape in [vec![3usize, 2], vec![2, 2, 2], vec![2, 1, 2, 2]] {
        let ch = shape[1];
        let m = shape[0] * shape[2..].iter().product::<usize>();
        let n_el = m * ch;
        let modes = [
            Mode::Train,
            Mode::Eval {
                mean: pattern(ch, 2.0),
                var: pattern(ch, 6.0).iter().map(|v| v.abs() + 0.5).collect(),
            },
            Mode::Eval {
                mean: pattern(ch, 4.0),
                var: vec![0.0; ch],
            },
        ];
        for mode in modes {
            for w0 in [vec![0.0f32; ch], pattern(ch, 3.0)] {
                for (w, b, wrts) in sweep_affines(&w0) {
                    for vals in sweep_channel_values(m) {
                        let x0 = place_channels(&vals, &shape);
                        let losses = scales
                            .iter()
                            .map(|&s| (s, false))
                            .chain(quadratic_scales.iter().map(|&s| (s, true)));
                        for (scale, quadratic) in losses {
                            // eval の xhat は固定統計からの距離で有界でない（x=1e20 で xhat≈1e20）。
                            // quadratic の `∂dw/∂w = Σ 2c·xhat²` は真値そのものが f32 を超え、`inf` が
                            // 正解になるので有限性を問わない（上流の `Mul` と 1 階 VJP の段。§27.3）。
                            let eval_mode = matches!(mode, Mode::Eval { .. });
                            if quadratic && eval_mode && vals.iter().any(|v| v.abs() > 1e10) {
                                continue;
                            }
                            for cpat in [ones(n_el), alternating(n_el)] {
                                for &wrt in &wrts {
                                    let sn = if let Wrt::X = wrt { n_el } else { ch };
                                    for seed in [ones(sn), alternating(sn)] {
                                        let p = Probe {
                                            mode: mode.clone(),
                                            quadratic,
                                            x0: x0.clone(),
                                            shape: shape.clone(),
                                            c: scaled(&cpat, scale),
                                            w: w.clone(),
                                            b: b.clone(),
                                            wrt,
                                            seed,
                                        };
                                        match probe(&p) {
                                            Outcome::Accepted => accepted += 1,
                                            Outcome::Rejected => rejected += 1,
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // 判定が全拒否・全受理に退化していないこと。
    assert!(
        accepted > 0 && rejected > 0,
        "accepted={accepted} rejected={rejected}"
    );
}

#[test]
fn rejected_input_leaves_child_empty_and_allows_retry() {
    for mode in [
        Mode::Train,
        Mode::Eval {
            mean: vec![0.0],
            var: vec![1.0],
        },
    ] {
        let child = Tape::new_with_ops(common::naive_ops());
        let try_build = |c: Vec<f32>| -> Result<(), AutodiffError> {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.5], &[4, 1]));
            let w = tape.var(&t(vec![1.0], &[1]));
            let y = match &mode {
                Mode::Train => x.batch_norm(Some(&w), None, EPS),
                Mode::Eval { mean, var } => x.batch_norm_infer(
                    Some(&w),
                    None,
                    &t(mean.clone(), &[1]),
                    &t(var.clone(), &[1]),
                    EPS,
                ),
            }
            .unwrap();
            let k = tape.var_no_grad(&t(c, &[4, 1]));
            let loss = y.mul(&k).unwrap().sum(None).unwrap();
            tape.backward_create_graph(&loss, &child).map(|_| ())
        };
        let err = try_build(vec![3e38, -3e38, 3e38, -3e38]).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty(), "{mode:?}: 拒否後に child が空へ戻る");
        let err = try_build(vec![f32::NAN, 1.0, 1.0, 1.0]).unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty());
        try_build(vec![1.0, -1.0, 1.0, -1.0]).expect("同じ child で再試行できる");
        assert!(!child.is_empty());
    }
}
