//! 構造変換 3 Op（`Dropout`・`Interpolate`・`MaxUnpool`）の `create_graph` 対応（イシュー #3793・
//! 決定記録 `docs/autodiff-higher-order-grad-decision.md` §35・§56）の受け入れ条件を確かめる統合テスト。
//!
//! 3 Op は入力について線形なので、素の `sum(op(x))` の Hessian は恒等的に 0 になる。二階は下流の
//! 非線形（本ファイルでは `sum(y ⊙ y ⊙ c)`）からだけ流れるため、突合は非線形を合成した損失で行う。
//!
//! - 1 階 `first_order()` が素の `Tape::backward` と bit 一致し、`cg.grad` が REQ-2 で一致する
//!   （CPU では `cg.grad` も bit 一致。`Dropout` の drop 位置だけ符号付きゼロの差を許す）。
//! - Hessian を独立な有限差分（1 階解析勾配の中央差分）と REQ-2 統一複合判定で突合する。
//! - 孫テープ（`jvp_create_graph`）の再生が独立な参照と一致する。
//! - `Dropout` の写しがグローバル RNG を消費しない（マスクを引き直さない）。
//! - 必須ベクトル（巨大値・非有限の上流・重複 index・空 batch・Area の打ち消し）。
//! - 隣接する恒久対象外 Op（`Op::Custom`）は fail-closed のままで `child` が空に保たれる。
//!
//! グローバル RNG を使うため、全テストを同一の `Mutex` で直列化する（`nn_dropout.rs` と同じ方式）。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::{Arc, Mutex, MutexGuard};

use fandhe_ai_autodiff::functional_ops::jvp_create_graph;
use fandhe_ai_autodiff::max_unpool_ops::max_unpool2d;
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape, Var, manual_seed, rand};
use fandhe_ai_tensor_core::{InterpolateMode, Tensor};

fn lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn flat_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        out[d] = idx % shape[d];
        idx /= shape[d];
    }
    out
}

fn shp(v: &Var<'_>) -> Vec<usize> {
    v.value().shape().to_vec()
}

fn dense(v: impl std::borrow::Borrow<Tensor<f32>>) -> Vec<f32> {
    let v = v.borrow();
    v.contiguous().as_slice().expect("contiguous").to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
}

/// 値の重複がなく最小間隔が 0.3 の入力（argmax がタイにならない）。
fn distinct(n: usize) -> Vec<f32> {
    let p = [29usize, 53, 101, 211, 401, 809, 1601]
        .into_iter()
        .find(|&p| p > n)
        .unwrap();
    (0..n)
        .map(|i| ((i * 37 + 5) % p) as f32 * 0.3 - 2.0)
        .collect()
}

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

/// `sum(y ⊙ y ⊙ c)`（`c` は y と同 shape の非一様な定数）。
fn quad<'t>(tape: &'t Tape, y: &Var<'t>) -> BuildResult<'t> {
    let shape = shp(y);
    let c = tape.var_no_grad(&t(common::det_seq(flat_len(&shape), 0.7), &shape));
    y.mul(y)?.mul(&c)?.sum(None)
}

const SEED: u64 = 20_261_101;

fn extract_scalar<'c>(
    v: &Var<'c>,
    shape: &[usize],
    flat_idx: usize,
) -> Result<Var<'c>, AutodiffError> {
    let idx = unravel(flat_idx, shape);
    let mut cur = *v;
    for (dim, &i) in idx.iter().enumerate() {
        cur = cur.narrow(dim, i, 1)?;
    }
    cur.sum(None)
}

fn finite_diff_hessian<F>(build: &F, x0: &[f32], shape: &[usize], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = flat_len(shape);
    let grad_at = |x: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), shape));
        let loss = build(&tape, &xv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let dx = grads.get(&xv).unwrap().expect("x は loss に到達する");
        (0..n)
            .map(|i| dx.get(&unravel(i, shape)).unwrap() as f64)
            .collect()
    };
    let mut hessian = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut xp = x0.to_vec();
        let mut xm = x0.to_vec();
        xp[i] += h as f32;
        xm[i] -= h as f32;
        let gp = grad_at(&xp);
        let gm = grad_at(&xm);
        for j in 0..n {
            hessian[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hessian
}

fn analytic_hessian<F>(build: &F, x0: &[f32], shape: &[usize]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = flat_len(shape);
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gx = cg.grad(&x).unwrap().expect("1 階勾配");
    let cx = cg.child_var(&x).unwrap().expect("写し");
    let mut hessian = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = extract_scalar(&gx, shape, j).unwrap();
        let row_grads = match child.backward(&gj) {
            Err(AutodiffError::GradientTrackingDisabled) => continue, // 定数（行は 0）
            other => other.expect("2 階"),
        };
        if let Some(row) = row_grads.get(&cx).unwrap() {
            for i in 0..n {
                hessian[j][i] = row.get(&unravel(i, shape)).unwrap() as f64;
            }
        }
    }
    hessian
}

fn hessian_case<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize], h: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let analytic = analytic_hessian(&build, x0, shape);
    let numeric = finite_diff_hessian(&build, x0, shape, h);
    for (j, (ra, rn)) in analytic.iter().zip(&numeric).enumerate() {
        for (i, (&a, &n)) in ra.iter().zip(rn).enumerate() {
            assert!(
                common::req2_close(a, n),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={n}"
            );
        }
    }
}

/// 1 階 `Tape::backward`・`first_order()`・`cg.grad` の突合。`first_order()` は常に bit 一致、
/// `cg.grad` は `exact_bits` なら bit 一致、そうでなければ `==`（±0 を同一視）と REQ-2。
fn first_order_case<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize], exact_bits: bool)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(x0.to_vec(), shape));
    let loss = build(&tape, &x).expect("forward");
    let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().expect("x"));
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let first = dense(cg.first_order().get(&x).unwrap().expect("x"));
    assert_eq!(bits(&first), bits(&want), "{ctx}: first_order()");
    let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
    assert_eq!(got.len(), want.len(), "{ctx}");
    for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
        if exact_bits {
            assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: cg.grad[{i}] {a} vs {b}");
        } else {
            assert!(
                a == b || common::req2_close(f64::from(a), f64::from(b)),
                "{ctx}: cg.grad[{i}] {a} vs {b}"
            );
        }
    }
}

/// 孫テープ: 出力が非スカラーの `op(x)` の `J·v` が、線形性から `op(v)` と一致する。
fn jvp_linear_case<F>(ctx: &str, build: F, x0: &[f32], shape: &[usize])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tangent: Vec<f32> = (0..flat_len(shape))
        .map(|i| ((i * 3 + 1) % 7) as f32 * 0.1 - 0.3)
        .collect();
    let reference = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let v = tape.var(&t(tangent.clone(), shape));
        dense(build(&tape, &v).unwrap().to_tensor())
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let grand = Tape::new_with_ops(common::naive_ops());
    let p = tape.var(&t(x0.to_vec(), shape));
    let out = build(&tape, &p).unwrap();
    let res = jvp_create_graph(&tape, &out, &p, &t(tangent, shape), &child, &grand)
        .unwrap_or_else(|e| panic!("{ctx}: jvp_create_graph: {e:?}"));
    let got = dense(res.output().to_tensor());
    assert_eq!(got.len(), reference.len(), "{ctx}");
    for (i, (&a, &b)) in got.iter().zip(&reference).enumerate() {
        assert!(a.is_finite(), "{ctx}: 孫テープの出力が非有限 [{i}]={a}");
        assert!(
            a == b || common::req2_close(f64::from(a), f64::from(b)),
            "{ctx}: jvp[{i}] {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------
// 対象の網羅（Interpolate 全 7 mode × align_corners・rank 3〜5）
// ---------------------------------------------------------------------

struct InterpCase {
    ctx: &'static str,
    shape: Vec<usize>,
    size: Vec<usize>,
    mode: InterpolateMode,
}

fn interp_cases() -> Vec<InterpCase> {
    let mut v = Vec::new();
    let mut push = |ctx, shape: &[usize], size: &[usize], mode| {
        v.push(InterpCase {
            ctx,
            shape: shape.to_vec(),
            size: size.to_vec(),
            mode,
        });
    };
    for (name, mode) in [
        ("nearest", InterpolateMode::Nearest),
        ("nearest_exact", InterpolateMode::NearestExact),
        ("area", InterpolateMode::Area),
    ] {
        push(name, &[1, 2, 4], &[6], mode); // アップ（rank 3）
        push(name, &[1, 2, 5], &[3], mode); // ダウン（Area は不均一な窓）
        push(name, &[1, 2, 3, 3], &[5, 4], mode); // rank 4
        push(name, &[1, 1, 2, 2, 3], &[3, 3, 2], mode); // rank 5
        push(name, &[1, 1, 4], &[4], mode); // 恒等
    }
    for ac in [false, true] {
        push(
            "linear",
            &[1, 2, 4],
            &[7],
            InterpolateMode::Linear { align_corners: ac },
        );
        push(
            "linear_down",
            &[1, 2, 6],
            &[4],
            InterpolateMode::Linear { align_corners: ac },
        );
        push(
            "bilinear",
            &[1, 2, 3, 3],
            &[5, 4],
            InterpolateMode::Bilinear { align_corners: ac },
        );
        push(
            "bicubic",
            &[1, 1, 3, 3],
            &[5, 4],
            InterpolateMode::Bicubic { align_corners: ac },
        );
        push(
            "trilinear",
            &[1, 1, 2, 2, 3],
            &[3, 3, 2],
            InterpolateMode::Trilinear { align_corners: ac },
        );
    }
    v
}

fn interp<'t>(x: &Var<'t>, c: &InterpCase) -> BuildResult<'t> {
    x.interpolate(&c.size, c.mode)
}

#[test]
fn structural_first_order_bit_identical_and_cgrad_req2() {
    let _g = lock();
    for c in interp_cases() {
        let x0 = distinct(flat_len(&c.shape));
        first_order_case(
            &format!("{} {:?}", c.ctx, c.mode),
            |tp, x| quad(tp, &interp(x, &c)?),
            &x0,
            &c.shape,
            true,
        );
    }
    // MaxUnpool（重なりなし）。
    let shape = [1usize, 2, 4, 4];
    first_order_case(
        "max_unpool",
        |tp, x| {
            let (y, idx) = x.max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)?;
            quad(
                tp,
                &max_unpool2d(&y, &idx, [2, 2], Some([2, 2]), [0, 0], None)?,
            )
        },
        &distinct(flat_len(&shape)),
        &shape,
        true,
    );
    // Dropout 3 種（drop 位置は ±0 を同一視）。
    let shape = [2usize, 3, 4, 4];
    let x0 = distinct(flat_len(&shape));
    for (name, kind) in [("dropout", 0), ("dropout2d", 1), ("alpha_dropout", 2)] {
        first_order_case(
            name,
            |tp, x| {
                manual_seed(SEED);
                let y = match kind {
                    0 => x.dropout(0.4, true)?,
                    1 => x.dropout2d(0.4, true)?,
                    _ => x.alpha_dropout(0.4, true)?,
                };
                quad(tp, &y)
            },
            &x0,
            &shape,
            false,
        );
    }
}

#[test]
fn structural_hessian_matches_finite_difference() {
    let _g = lock();
    for c in interp_cases() {
        let x0 = distinct(flat_len(&c.shape));
        hessian_case(
            &format!("{} {:?}", c.ctx, c.mode),
            |tp, x| quad(tp, &interp(x, &c)?),
            &x0,
            &c.shape,
            // 補間は線形・損失は二次なので 1 階勾配は x に対し厳密に線形。h を大きく取り丸め雑音を抑える。
            1.0,
        );
    }
    // MaxUnpool: 重なりなし・重複 index あり（stride 1 のプール由来で同じ位置を複数回指す）。
    let shape = [1usize, 1, 4, 4];
    hessian_case(
        "max_unpool 重なりなし",
        |tp, x| {
            let (y, idx) = x.max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)?;
            quad(
                tp,
                &max_unpool2d(&y, &idx, [2, 2], Some([2, 2]), [0, 0], None)?,
            )
        },
        &distinct(16),
        &shape,
        0.05,
    );
    let shape = [1usize, 1, 3, 3];
    hessian_case(
        "max_unpool 重複 index",
        |tp, x| {
            let (y, idx) = x.max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)?;
            quad(
                tp,
                &max_unpool2d(&y, &idx, [2, 2], Some([1, 1]), [0, 0], None)?,
            )
        },
        &distinct(9),
        &shape,
        0.05,
    );
    // Dropout: 固定マスクの下で Hessian = 2·c·mask²。
    let shape = [2usize, 2, 3, 3];
    let x0 = distinct(flat_len(&shape));
    for (name, kind) in [("dropout", 0), ("dropout2d", 1), ("alpha_dropout", 2)] {
        hessian_case(
            name,
            |tp, x| {
                manual_seed(SEED);
                let y = match kind {
                    0 => x.dropout(0.4, true)?,
                    1 => x.dropout2d(0.4, true)?,
                    _ => x.alpha_dropout(0.4, true)?,
                };
                quad(tp, &y)
            },
            &x0,
            &shape,
            1.0,
        );
    }
}

#[test]
fn structural_plain_sum_hessian_is_exact_zero() {
    let _g = lock();
    // 素の sum(op(x)) の 1 階勾配は x に依存しない定数なので Hessian は厳密に 0。
    let check = |ctx: &str,
                 shape: &[usize],
                 build: &dyn for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>| {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(flat_len(shape)), shape));
        let loss = build(&tape, &x).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let gx = cg.grad(&x).unwrap().expect("x");
        let cx = cg.child_var(&x).unwrap().expect("写し");
        for j in 0..flat_len(shape) {
            let gj = extract_scalar(&gx, shape, j).unwrap();
            let rg = match child.backward(&gj) {
                // 1 階勾配が x に依存しない定数なら追跡対象の祖先を持たず、微分自体が拒否される。
                Err(AutodiffError::GradientTrackingDisabled) => continue,
                Err(AutodiffError::Backward(m)) if m.contains("追跡対象の祖先を持たない") =>
                {
                    continue;
                }
                other => other.unwrap(),
            };
            if let Some(row) = rg.get(&cx).unwrap() {
                assert!(dense(row).iter().all(|&v| v == 0.0), "{ctx}: 非ゼロの二階");
            }
        }
    };
    for c in interp_cases() {
        check(c.ctx, &c.shape, &|_tp, x| interp(x, &c)?.sum(None));
    }
    check("max_unpool", &[1, 1, 4, 4], &|_tp, x| {
        let (y, idx) = x.max_pool2d([2, 2], Some([2, 2]), [0, 0], [1, 1], false)?;
        max_unpool2d(&y, &idx, [2, 2], Some([2, 2]), [0, 0], None)?.sum(None)
    });
    check("dropout", &[2, 3, 4], &|_tp, x| {
        manual_seed(SEED);
        x.dropout(0.5, true)?.sum(None)
    });
}

#[test]
fn structural_grandchild_replay_matches_reference() {
    let _g = lock();
    for c in interp_cases() {
        let x0 = distinct(flat_len(&c.shape));
        jvp_linear_case(
            &format!("{} {:?}", c.ctx, c.mode),
            |_tp, x| interp(x, &c),
            &x0,
            &c.shape,
        );
    }
    // index は x0 のプールで固定する（接ベクトルで引き直すと勝者が変わり参照が食い違う）。
    let idx = {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(9), &[1, 1, 3, 3]));
        x.max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
            .unwrap()
            .1
    };
    jvp_linear_case(
        "max_unpool 重複 index",
        |_tp, p| max_unpool2d(p, &idx, [2, 2], Some([1, 1]), [0, 0], None),
        &[0.5, -1.0, 2.0, 1.5],
        &[1, 1, 2, 2],
    );
    let shape = [2usize, 3, 4];
    jvp_linear_case(
        "dropout",
        |_tp, x| {
            manual_seed(SEED);
            x.dropout(0.5, true)
        },
        &distinct(flat_len(&shape)),
        &shape,
    );
}

#[test]
fn dropout_create_graph_does_not_consume_global_rng() {
    let _g = lock();
    let shape = [2usize, 3, 4, 4];
    let n = flat_len(&shape);
    let x0 = distinct(n);
    fn apply<'t>(x: &Var<'t>, kind: u8, p: f32) -> BuildResult<'t> {
        match kind {
            0 => x.dropout(p, true),
            1 => x.dropout2d(p, true),
            _ => x.alpha_dropout(p, true),
        }
    }
    for (name, kind, p) in [
        ("dropout", 0u8, 0.5f32),
        ("dropout2d", 1, 0.5),
        ("alpha_dropout", 2, 0.5),
        ("dropout p=1", 0, 1.0),
    ] {
        // 参照: forward だけ（マスク抽選）のあとの次の乱数。
        manual_seed(SEED);
        {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            apply(&x, kind, p).unwrap();
        }
        let want = bits(&dense(rand(&[8]).unwrap()));
        // backward_create_graph（と cg.grad の読み出し）は RNG を進めない。
        manual_seed(SEED);
        {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let loss = quad(&tape, &apply(&x, kind, p).unwrap()).unwrap();
            let cg = tape.backward_create_graph(&loss, &child).unwrap();
            let _ = cg.grad(&x).unwrap();
        }
        assert_eq!(
            bits(&dense(rand(&[8]).unwrap())),
            want,
            "{name}: create_graph"
        );
        // jvp_create_graph（親→子→孫の再生）も RNG を進めない。
        manual_seed(SEED);
        {
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let grand = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &shape));
            let y = apply(&x, kind, p).unwrap();
            let tangent = t(vec![0.5; n], &shape);
            jvp_create_graph(&tape, &y, &x, &tangent, &child, &grand).unwrap();
        }
        assert_eq!(
            bits(&dense(rand(&[8]).unwrap())),
            want,
            "{name}: jvp_create_graph"
        );
    }
}

#[test]
fn structural_mandatory_vectors() {
    let _g = lock();
    // (1) 巨大な上流 g=3e38（Bicubic は負の重みを含む）と符号交互の上流。1 階と bit 一致する。
    for c in interp_cases() {
        let x0 = distinct(flat_len(&c.shape));
        for pattern in [0, 1] {
            let ctx = format!("big {} {:?} p{pattern}", c.ctx, c.mode);
            let tape = Tape::new_with_ops(common::naive_ops());
            let child = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(x0.clone(), &c.shape));
            let y = interp(&x, &c).unwrap();
            let ys = shp(&y);
            let scale: Vec<f32> = (0..flat_len(&ys))
                .map(|i| {
                    if pattern == 1 && i % 2 == 1 {
                        -3e38
                    } else {
                        3e38
                    }
                })
                .collect();
            let loss = y
                .mul(&tape.var_no_grad(&t(scale, &ys)))
                .unwrap()
                .sum(None)
                .unwrap();
            let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().expect("x"));
            match tape.backward_create_graph(&loss, &child) {
                Ok(cg) => {
                    let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
                    assert_eq!(bits(&got), bits(&want), "{ctx}");
                }
                Err(AutodiffError::Backward(_)) => assert!(child.is_empty(), "{ctx}"),
                Err(e) => panic!("{ctx}: {e:?}"),
            }
        }
    }
    // (2) Dropout: 上流が inf のとき drop 位置は 1 階が NaN（inf·0）、cgrad は 0。keep 位置は bit 一致。
    {
        let shape = [4usize, 8];
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(32), &shape));
        manual_seed(SEED);
        let y = x.dropout(0.5, true).unwrap();
        let inf = tape.var_no_grad(&t(vec![f32::INFINITY; 32], &shape));
        let loss = y.mul(&inf).unwrap().sum(None).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().expect("x"));
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
        let (mut drops, mut keeps) = (0, 0);
        for (a, b) in got.iter().zip(&want) {
            if b.is_nan() {
                drops += 1;
                assert_eq!(*a, 0.0, "drop 位置の cgrad は 0");
            } else {
                keeps += 1;
                assert_eq!(a.to_bits(), b.to_bits(), "keep 位置は bit 一致");
            }
        }
        assert!(drops > 0 && keeps > 0, "drops={drops} keeps={keeps}");
    }
    // (3) Dropout: 負の有限上流の drop 位置は 1 階が -0.0、cgrad は +0.0（== では同一視される）。
    {
        let shape = [64usize];
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(64), &shape));
        manual_seed(SEED);
        let y = x.dropout(0.5, true).unwrap();
        let neg = tape.var_no_grad(&t(vec![-2.0; 64], &shape));
        let loss = y.mul(&neg).unwrap().sum(None).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().expect("x"));
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
        for (a, b) in got.iter().zip(&want) {
            assert!(a == b, "{a} vs {b}");
            if *b != 0.0 {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    // (4) Dropout p=1.0: 全 drop。cgrad は全ゼロ。
    {
        let shape = [3usize, 5];
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(15), &shape));
        let y = x.dropout(1.0, true).unwrap();
        let loss = quad(&tape, &y).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
        assert!(got.iter().all(|&v| v == 0.0));
    }
    // (5) MaxUnpool: 重複 index（last-writer）と NaN・inf の入力値。1 階と bit 一致する。
    {
        let shape = [1usize, 1, 3, 3];
        let mut x0 = distinct(9);
        x0[1] = f32::NAN;
        x0[4] = f32::INFINITY;
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(x0, &shape));
        let (p, idx) = x
            .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
            .unwrap();
        let up = max_unpool2d(&p, &idx, [2, 2], Some([1, 1]), [0, 0], None).unwrap();
        let us = shp(&up);
        let scale: Vec<f32> = (0..flat_len(&us)).map(|i| 1.0 + i as f32).collect();
        let loss = up
            .mul(&tape.var_no_grad(&t(scale, &us)))
            .unwrap()
            .sum(None)
            .unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().expect("x"));
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
        assert_eq!(bits(&got), bits(&want));
    }
    // (6) Interpolate の空 batch（outer == 0）。
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![], &[0, 2, 4]));
        let y = x.interpolate(&[6], InterpolateMode::Nearest).unwrap();
        let loss = y.sum(None).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let g = cg.grad(&x).unwrap().expect("x").to_tensor();
        assert_eq!(g.shape(), &[0, 2, 4]);
    }
    // (7) Area の不均一な窓（5→3）と打ち消し合う上流。構造判定を通るか型付きで拒否されるかを固定する
    // （通る場合は 1 階と REQ-2 で一致し、拒否される場合は child が空へ戻る）。
    {
        let shape = [1usize, 1, 5];
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(5), &shape));
        let y = x.interpolate(&[3], InterpolateMode::Area).unwrap();
        let ys = shp(&y);
        let up = tape.var_no_grad(&t(vec![1e8, -1e8, 1.0], &ys));
        let loss = y.mul(&up).unwrap().sum(None).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&x).unwrap().expect("x"));
        match tape.backward_create_graph(&loss, &child) {
            Ok(cg) => {
                let got = dense(cg.grad(&x).unwrap().expect("x").to_tensor());
                for (a, b) in got.iter().zip(&want) {
                    assert!(common::req2_close(f64::from(*a), f64::from(*b)), "{a} {b}");
                }
            }
            // 実測（2026-11-01, CPU naive）: 5→3 の打ち消し上流は構造判定を通る（§56.2）。
            Err(e) => panic!("5→3 は構造判定を通るはず: {e:?}"),
        }
    }
}

#[test]
fn structural_unsafe_inputs_rejected_child_empty_and_retry() {
    let _g = lock();
    let shape = [1usize, 1, 7];
    let child = Tape::new_with_ops(common::naive_ops());
    // Area の構造判定外れ（上流が inf だと集約前の中間和が f32 範囲を超える）は、構造を子テープへ積んだ
    // あとに型付きで拒否される。呼び出し側のロールバックで child は空へ戻り、同じ child で通常の
    // 入力を再試行できる（§27.3・§30 と同じ fail-closed）。
    {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(distinct(7), &shape));
        let y = x.interpolate(&[3], InterpolateMode::Area).unwrap();
        let ys = shp(&y);
        let up = tape.var_no_grad(&t(vec![f32::INFINITY, 1.0, 1.0], &ys));
        let loss = y.mul(&up).unwrap().sum(None).unwrap();
        let err = tape
            .backward_create_graph(&loss, &child)
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty());
    }
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(distinct(7), &shape));
    let loss = quad(&tape, &x.interpolate(&[3], InterpolateMode::Area).unwrap()).unwrap();
    tape.backward_create_graph(&loss, &child)
        .expect("通常の入力は再試行で成功する");
}

/// 恒等関数として振る舞う `CustomFunction`（`Op::Custom` を経由させるためだけの最小実装）。
struct IdentityCustomFn;

impl CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity_custom_fn"
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }

    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].contiguous())
    }

    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        Ok(vec![Some(upstream.contiguous())])
    }
}

#[test]
fn structural_neighbouring_unsupported_op_stays_fail_closed() {
    let _g = lock();
    // 構造変換の下流に恒久対象外の `Op::Custom` があれば、入口で拒否され child は空のまま。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(distinct(8), &[1, 2, 4]));
    let up = x.interpolate(&[6], InterpolateMode::Nearest).unwrap();
    let custom = tape
        .custom(Arc::new(IdentityCustomFn), &[up])
        .expect("Tape::custom 登録成功");
    let loss = custom.sum(None).unwrap();
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
    assert!(child.is_empty());
}
