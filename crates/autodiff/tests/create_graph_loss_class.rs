//! 損失のクラス軸系 6 variant（`CrossEntropyLossWithOptions`・`MultiMarginLoss`・
//! `MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・`CosineEmbeddingLoss`・`TripletMarginLoss`）の
//! `create_graph` 対応（イシュー #3800・決定記録 `docs/autodiff-higher-order-grad-decision.md` §40、
//! 実装記録 §55）の受け入れ条件を確かめる統合テスト。
//!
//! - `first_order()` が 1 階 `Tape::backward` と bit 一致し、`cg.grad`（6 つとも V3）も CPU で bit 一致する。
//! - Hessian を 1 階解析勾配の中央差分（h と h/2 の Richardson 外挿）と REQ-2 判定で突合する。
//! - 混合二階（Cosine の x1·x2、Triplet の a·p・a·n・p·n）と p=2 の `u==0` レーンの曲率保持。
//! - 総括判定: 必須ベクトルは有限か型付き拒否（拒否時は `child` が空で再試行可能）。
//! - 孫テープ（`jvp_create_graph`）・空テンソル・区分の境界・隣接 `Op::Custom` の fail-closed。
//!
//! GPU parity は `crates/facade/tests/create_graph_gpu_parity.rs`（実機待ち・`#[ignore]`）。

#![allow(clippy::needless_range_loop)]

mod common;

use std::sync::Arc;

use fandhe_ai_autodiff::loss_ops::{CrossEntropyOptions, TripletMarginOptions};
use fandhe_ai_autodiff::margin_focal_loss_ops::{MultiLabelSoftMarginOptions, MultiMarginOptions};
use fandhe_ai_autodiff::{AutodiffError, Reduction, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

type BuildResult<'t> = Result<Var<'t>, AutodiffError>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(v: impl std::borrow::Borrow<Tensor<f32>>) -> Vec<f32> {
    v.borrow()
        .contiguous()
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn norm_bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|a| {
            if a.is_nan() {
                f32::NAN.to_bits()
            } else {
                a.to_bits()
            }
        })
        .collect()
}

/// 葉 `z`（長さ `k·n`）を `k` 個の `[n]` へ切り出す。
fn parts<'t>(z: &Var<'t>, k: usize, n: usize) -> Vec<Var<'t>> {
    (0..k).map(|i| z.narrow(0, i * n, n).unwrap()).collect()
}

fn finite_diff_hessian<F>(build: F, z0: &[f32], h: f64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = z0.len();
    let grad_at = |z: &[f32]| -> Vec<f64> {
        let tape = Tape::new_with_ops(common::naive_ops());
        let zv = tape.var(&t(z.to_vec(), &[n]));
        let loss = build(&tape, &zv).expect("forward");
        let grads = tape.backward(&loss).expect("backward");
        let d = dense(grads.get(&zv).unwrap().expect("z は loss に到達する"));
        d.iter().map(|&v| f64::from(v)).collect()
    };
    let mut hess = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut zp = z0.to_vec();
        let mut zm = z0.to_vec();
        zp[i] += h as f32;
        zm[i] -= h as f32;
        let (gp, gm) = (grad_at(&zp), grad_at(&zm));
        for j in 0..n {
            hess[j][i] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    hess
}

fn analytic_hessian<F>(build: F, z0: &[f32]) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let n = z0.len();
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(z0.to_vec(), &[n]));
    let loss = build(&tape, &z).expect("forward");
    let cg = tape
        .backward_create_graph(&loss, &child)
        .expect("backward_create_graph");
    let gz = cg.grad(&z).unwrap().expect("1 階勾配");
    let cz = cg.child_var(&z).unwrap().expect("写し");
    let mut hess = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let gj = gz.narrow(0, j, 1).unwrap().sum(None).unwrap();
        // 区分定数な成分（Hinge・MarginRanking の root）は追跡対象の祖先を持たず逆伝播できない（= 全ゼロ）。
        let Ok(grads) = child.backward(&gj) else {
            continue;
        };
        // 区分定数な損失（Hinge・MarginRanking）の root では 2 階が z へ到達しない（全ゼロ）。
        if let Some(row) = grads.get(&cz).unwrap() {
            let row = dense(row);
            for i in 0..n {
                hess[j][i] = f64::from(row[i]);
            }
        }
    }
    hess
}

fn hessian_case<F>(ctx: &str, build: F, z0: &[f32], expect_nonzero: bool, h: f64)
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t> + Copy,
{
    let analytic = analytic_hessian(build, z0);
    let (coarse, fine) = (
        finite_diff_hessian(build, z0, h),
        finite_diff_hessian(build, z0, h / 2.0),
    );
    let numeric: Vec<Vec<f64>> = coarse
        .iter()
        .zip(&fine)
        .map(|(rc, rf)| {
            rc.iter()
                .zip(rf)
                .map(|(c, f)| (4.0 * f - c) / 3.0)
                .collect()
        })
        .collect();
    for (j, (ra, rn)) in analytic.iter().zip(&numeric).enumerate() {
        for (i, (&a, &nv)) in ra.iter().zip(rn).enumerate() {
            assert!(
                common::req2_close(a, nv),
                "{ctx}: hessian[{j}][{i}] analytic={a} numeric={nv}"
            );
        }
    }
    if expect_nonzero {
        assert!(
            analytic.iter().flatten().any(|v| v.abs() > 1e-3),
            "{ctx}: Hessian が全ゼロ（検査が空振り）"
        );
    }
}

// ---------------------------------------------------------------------------------
// ケース定義（葉 z を logits・x 等へ切り出して損失を作る。root と非 root の両方）
// ---------------------------------------------------------------------------------

type CaseFn = Box<dyn for<'t> Fn(&Var<'t>) -> BuildResult<'t>>;

struct Case {
    name: String,
    z0: Vec<f32>,
    build: CaseFn,
    /// root で Hessian が非零になりうるか（区分定数な損失は false）。
    root_nonzero: bool,
    /// 中央差分の刻み。
    step: f64,
}

fn vals(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.83).sin()).collect()
}

fn i32t(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture")
}

fn rn(r: Reduction) -> &'static str {
    match r {
        Reduction::Mean => "mean",
        Reduction::Sum => "sum",
        _ => "other",
    }
}

fn cewo_case(
    name: &str,
    shape: &[usize],
    class_dim: usize,
    targets: (Vec<i32>, Vec<usize>),
    opts: CrossEntropyOptions,
    r: Reduction,
) -> Case {
    let n: usize = shape.iter().product();
    let shape = shape.to_vec();
    Case {
        name: format!("cewo {name} {}", rn(r)),
        z0: vals(n, 0.3),
        build: Box::new(move |z| {
            z.reshape(&shape)?.cross_entropy_loss_with(
                &i32t(targets.0.clone(), &targets.1),
                class_dim,
                r,
                &opts,
            )
        }),
        root_nonzero: true,
        step: 0.04,
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    let w4 = || t(vec![1.0, 2.0, 0.5, 1.5], &[4]);
    for r in [Reduction::Mean, Reduction::Sum] {
        let tg = || (vec![0, 2, 3], vec![3]);
        v.push(cewo_case(
            "smooth",
            &[3, 4],
            1,
            tg(),
            CrossEntropyOptions::default().label_smoothing(0.1),
            r,
        ));
        v.push(cewo_case(
            "ignore",
            &[3, 4],
            1,
            tg(),
            CrossEntropyOptions::default().ignore_index(2),
            r,
        ));
        v.push(cewo_case(
            "weight",
            &[3, 4],
            1,
            tg(),
            CrossEntropyOptions::default().class_weight(w4()),
            r,
        ));
        v.push(cewo_case(
            "all",
            &[3, 4],
            1,
            tg(),
            CrossEntropyOptions::default()
                .label_smoothing(0.2)
                .ignore_index(0)
                .class_weight(w4()),
            r,
        ));
        v.push(cewo_case(
            "dim0",
            &[4, 3],
            0,
            (vec![1, 3, 0], vec![3]),
            CrossEntropyOptions::default()
                .label_smoothing(0.1)
                .class_weight(w4()),
            r,
        ));
        v.push(cewo_case(
            "rank3-mid",
            &[2, 3, 2],
            1,
            (vec![0, 2, 1, 1], vec![2, 2]),
            CrossEntropyOptions::default()
                .label_smoothing(0.15)
                .class_weight(t(vec![1.0, 0.5, 2.0], &[3])),
            r,
        ));
        // MultiMargin
        let mm_x = vec![
            0.3, -0.4, 0.8, 0.1, //
            0.9, -0.6, 0.2, -0.3, //
            0.1, 0.5, -0.2, 0.4,
        ];
        for p in [1u8, 2] {
            for weighted in [false, true] {
                v.push(Case {
                    name: format!("multi_margin p={p} w={weighted} {}", rn(r)),
                    z0: mm_x.clone(),
                    build: Box::new(move |z| {
                        let mut o = MultiMarginOptions::default().p(p).margin(1.0);
                        if weighted {
                            o = o.weight(t(vec![0.5, 1.5, 2.0, 1.0], &[4]));
                        }
                        z.reshape(&[3, 4])?
                            .multi_margin_loss(&i32t(vec![1, 0, 3], &[3]), &o, r)
                    }),
                    root_nonzero: p == 2,
                    step: 0.04,
                });
            }
        }
        v.push(Case {
            name: format!("multilabel_margin {}", rn(r)),
            z0: vec![
                0.2, -0.5, 0.1, 0.9, //
                0.4, 0.6, -0.7, 0.0, //
                0.5, -0.2, 0.3, 0.1,
            ],
            build: Box::new(move |z| {
                z.reshape(&[3, 4])?.multilabel_margin_loss(
                    &i32t(vec![3, 0, -1, -1, 1, -1, 0, 0, 2, 1, -1, -1], &[3, 4]),
                    r,
                )
            }),
            root_nonzero: false,
            step: 0.04,
        });
        for weighted in [false, true] {
            v.push(Case {
                name: format!("multilabel_soft w={weighted} {}", rn(r)),
                z0: vals(12, 1.1),
                build: Box::new(move |z| {
                    let mut o = MultiLabelSoftMarginOptions::default();
                    if weighted {
                        o = o.weight(t(vec![1.0, 2.0, 0.5, 1.5], &[4]));
                    }
                    let tg = t(
                        vec![1.0, 0.0, 0.3, 1.0, 0.0, 1.0, 0.5, 0.0, 0.2, 0.9, 1.0, 0.0],
                        &[3, 4],
                    );
                    z.reshape(&[3, 4])?.multilabel_soft_margin_loss(&tg, &o, r)
                }),
                root_nonzero: true,
                step: 0.04,
            });
        }
        // Cosine（x1・x2 を同じ葉 z に載せる）。
        let x1 = [
            0.5, -0.3, 0.8, 0.2, 0.6, 0.2, -0.5, 0.4, 0.3, -0.8, 0.2, 0.1f32,
        ];
        let x2 = [
            0.1, 0.7, -0.4, 0.6, 0.5, 0.3, -0.4, 0.5, -0.4, 0.7, 0.1, -0.5f32,
        ];
        v.push(Case {
            name: format!("cosine {}", rn(r)),
            z0: [x1.to_vec(), x2.to_vec()].concat(),
            build: Box::new(move |z| {
                let p = parts(z, 2, 12);
                p[0].reshape(&[3, 4])?.cosine_embedding_loss(
                    &p[1].reshape(&[3, 4])?,
                    &t(vec![1.0, -1.0, -1.0], &[3]),
                    0.2,
                    r,
                )
            }),
            root_nonzero: true,
            step: 0.02,
        });
        // Triplet
        let a = [
            0.5, -0.3, 0.8, 0.2, 0.6, 0.2, -0.5, 0.4, 0.3, -0.8, 0.2, 0.1f32,
        ];
        let pp = [
            0.3, -0.1, 0.5, 0.4, 0.1, 0.5, -0.2, 0.9, 0.0, -0.5, 0.6, 0.5f32,
        ];
        let nn = [
            0.9, 0.5, 0.1, -0.3, 0.7, -0.4, 0.3, 0.0, -0.6, 0.2, 0.7, -0.2f32,
        ];
        for (pv, margin) in [(1.0f32, 1.2f32), (2.0, 0.6), (3.0, 1.0)] {
            for swap in [false, true] {
                v.push(Case {
                    name: format!("triplet p={pv} swap={swap} {}", rn(r)),
                    z0: [a.to_vec(), pp.to_vec(), nn.to_vec()].concat(),
                    build: Box::new(move |z| {
                        let q = parts(z, 3, 12);
                        let o = TripletMarginOptions::default()
                            .p(pv)
                            .margin(margin)
                            .swap(swap);
                        q[0].reshape(&[3, 4])?.triplet_margin_loss(
                            &q[1].reshape(&[3, 4])?,
                            &q[2].reshape(&[3, 4])?,
                            &o,
                            r,
                        )
                    }),
                    root_nonzero: pv != 1.0,
                    step: 0.01,
                });
            }
        }
    }
    v
}

fn cos_loss<'t>(z: &Var<'t>, y: f32, margin: f32) -> BuildResult<'t> {
    let p = parts(z, 2, 2);
    p[0].reshape(&[1, 2])?.cosine_embedding_loss(
        &p[1].reshape(&[1, 2])?,
        &t(vec![y], &[1]),
        margin,
        Reduction::Sum,
    )
}

fn tri_loss<'t>(z: &Var<'t>, o: &TripletMarginOptions) -> BuildResult<'t> {
    let q = parts(z, 3, 2);
    q[0].reshape(&[1, 2])?.triplet_margin_loss(
        &q[1].reshape(&[1, 2])?,
        &q[2].reshape(&[1, 2])?,
        o,
        Reduction::Sum,
    )
}

fn tri_opts(pv: f32, swap: bool, eps: f32) -> TripletMarginOptions {
    TripletMarginOptions::default()
        .p(pv)
        .swap(swap)
        .eps(eps)
        .margin(1e30)
}

fn good_cewo<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
    z.reshape(&[1, 3])?.cross_entropy_loss_with(
        &i32t(vec![1], &[1]),
        1,
        Reduction::Mean,
        &CrossEntropyOptions::default().label_smoothing(0.1),
    )
}

fn tri_big_margin<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
    tri_loss(z, &TripletMarginOptions::default().margin(5.0))
}

fn tri_p15<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
    tri_loss(
        z,
        &TripletMarginOptions::default().p(1.5).eps(0.0).margin(5.0),
    )
}

fn tri_u0_lane<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
    let q = parts(z, 3, 4);
    q[0].triplet_margin_loss(
        &q[1],
        &q[2],
        &TripletMarginOptions::default().p(2.0).eps(0.0).margin(2.0),
        Reduction::Sum,
    )
}

#[test]
fn loss_class_first_order_bit_identical_and_cgrad_req2() {
    for c in cases() {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(c.z0.clone(), &[c.z0.len()]));
        let loss = (c.build)(&z).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let first = dense(cg.first_order().get(&z).unwrap().unwrap());
        let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
        assert_eq!(
            norm_bits(&first),
            norm_bits(&want),
            "{}: first_order()",
            c.name
        );
        // 6 つとも V3 なので CPU では cg.grad も bit 一致する。
        assert_eq!(norm_bits(&got), norm_bits(&want), "{}: cg.grad", c.name);
    }
}

/// 損失が根でない場合、子テープへ写された損失 Op は forward と bit 一致する値を持つ。
#[test]
fn replayed_loss_value_matches_forward() {
    for c in cases() {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(c.z0.clone(), &[c.z0.len()]));
        let l = (c.build)(&z).unwrap();
        let loss = l.mul(&l).unwrap();
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let lm = cg.child_var(&l).unwrap().expect("損失の写し");
        assert_eq!(
            norm_bits(&dense(lm.to_tensor())),
            norm_bits(&dense(l.to_tensor())),
            "{}: 写しの損失値",
            c.name
        );
    }
}

#[test]
fn loss_class_hessian_matches_finite_difference() {
    for c in cases() {
        hessian_case(
            &format!("{} root", c.name),
            |_, z| (c.build)(z),
            &c.z0,
            c.root_nonzero,
            c.step,
        );
        hessian_case(
            &format!("{} non-root", c.name),
            |tape, z| {
                let l = (c.build)(z)?.mul(&tape.var_no_grad(&Tensor::scalar(0.2f32)))?;
                l.mul(&l)
            },
            &c.z0,
            true,
            c.step,
        );
    }
}

/// 混合二階（Cosine の x1·x2、Triplet の a·p・a·n・p·n）が非零で有限差分と一致する。
/// p=2 で一部レーンだけ差分が 0 の行（`u_i == 0`・`norm' > 0`）の曲率保持も確かめる。
#[test]
fn loss_class_mixed_second_derivatives() {
    for c in cases().into_iter().filter(|c| {
        c.name.starts_with("cosine")
            || (c.name.starts_with("triplet p=2") && c.name.ends_with("sum"))
    }) {
        let h = analytic_hessian(|_, z| (c.build)(z), &c.z0);
        let k = if c.name.starts_with("cosine") { 2 } else { 3 };
        let blk = c.z0.len() / k;
        for b in 1..k {
            let mut mixed = 0f64;
            for j in 0..blk {
                for i in 0..blk {
                    mixed = mixed.max(h[j][b * blk + i].abs());
                }
            }
            assert!(mixed > 1e-3, "{}: 混合二階 (0, {b}) が全ゼロ", c.name);
        }
        hessian_case(
            &format!("{} mixed", c.name),
            |_, z| (c.build)(z),
            &c.z0,
            true,
            c.step,
        );
    }
    // p=2・eps=0 で 1 レーンの a−p が厳密に 0（norm' > 0）。曲率 1/norm' がそのレーンに残る。
    let a = [0.5f32, 0.2, -0.4, 0.8];
    let p = [0.5f32, 0.6, 0.1, 0.3]; // lane 0 が a == p
    let n = [1.2f32, -0.5, 0.7, 0.2];
    let z0 = [a, p, n].concat();
    hessian_case("triplet u==0 lane", tri_u0_lane, &z0, true, 0.01);
    let h = analytic_hessian(tri_u0_lane, &z0);
    assert!(
        h[0][0].abs() > 1e-3,
        "u==0 のレーンの曲率が消えた: {}",
        h[0][0]
    );
}

#[test]
fn loss_class_grandchild_replay_matches_directional_derivative() {
    for c in cases() {
        let z0 = &c.z0;
        let dir: Vec<f32> = (0..z0.len()).map(|i| 0.3 + 0.1 * i as f32).collect();
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let grand = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.clone(), &[z0.len()]));
        let l = (c.build)(&z).unwrap();
        let loss = l.mul(&tape.var_no_grad(&Tensor::scalar(0.5f32))).unwrap();
        let loss = loss.mul(&loss).unwrap();
        let want_grad = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let want: f64 = want_grad
            .iter()
            .zip(&dir)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let res = fandhe_ai_autodiff::functional_ops::jvp_create_graph(
            &tape,
            &loss,
            &z,
            &t(dir.clone(), &[dir.len()]),
            &child,
            &grand,
        )
        .unwrap();
        let got = f64::from(res.output().value().get(&[]).unwrap());
        assert!(
            common::req2_close(got, want),
            "{}: jvp={got} want={want}",
            c.name
        );
    }
}

// ---------------------------------------------------------------------------------
// 総括判定・必須ベクトル
// ---------------------------------------------------------------------------------

/// 損失を作って create_graph を試み、受理なら 3 段（1 階・`cg.grad`・2 階）が有限、拒否なら
/// 型付きエラーで `child` が空であることを確かめる。`true` = 受理。
fn try_case<F>(ctx: &str, build: F, z0: &[f32]) -> bool
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(z0.to_vec(), &[z0.len()]));
    let loss = build(&tape, &z).unwrap();
    let first = tape
        .backward(&loss)
        .unwrap()
        .get(&z)
        .unwrap()
        .map(dense)
        .unwrap_or_default();
    match tape.backward_create_graph(&loss, &child) {
        Ok(cg) => {
            assert!(
                first.iter().all(|v| v.is_finite()),
                "{ctx}: 1 階が非有限 {first:?}"
            );
            let gz = cg.grad(&z).unwrap().unwrap();
            assert!(
                dense(gz.to_tensor()).iter().all(|v| v.is_finite()),
                "{ctx}: cg.grad"
            );
            let s = gz.sum(None).unwrap();
            let cz = cg.child_var(&z).unwrap().unwrap();
            if let Ok(h) = child.backward(&s)
                && let Some(row) = h.get(&cz).unwrap()
            {
                assert!(dense(row).iter().all(|v| v.is_finite()), "{ctx}: 2 階");
            }
            true
        }
        Err(e) => {
            assert!(matches!(e, AutodiffError::Backward(_)), "{ctx}: {e:?}");
            assert!(child.is_empty(), "{ctx}: 拒否後に child が空でない");
            false
        }
    }
}

/// 損失へ `scale` を掛けた形で `try_case` する。
fn try_scaled<F>(ctx: &str, build: F, z0: &[f32], scale: f32) -> bool
where
    F: for<'t> Fn(&Var<'t>) -> BuildResult<'t>,
{
    try_case(
        ctx,
        |tape, z| {
            build(z)?
                .mul(&tape.var_no_grad(&Tensor::scalar(scale)))?
                .sum(None)
        },
        z0,
    )
}

/// 拒否されること、その後に安全な入力で同じ `child` に再試行して成功することを確かめる。
fn assert_rejected_then_retry_ok<F, G>(ctx: &str, bad: F, z_bad: &[f32], good: G, z_good: &[f32])
where
    F: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
    G: for<'t> Fn(&'t Tape, &Var<'t>) -> BuildResult<'t>,
{
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(z_bad.to_vec(), &[z_bad.len()]));
    let loss = bad(&tape, &z).unwrap();
    let err = tape.backward_create_graph(&loss, &child).unwrap_err();
    assert!(matches!(err, AutodiffError::Backward(_)), "{ctx}: {err:?}");
    assert!(child.is_empty(), "{ctx}: 拒否後に child が空でない");
    let z2 = tape.var(&t(z_good.to_vec(), &[z_good.len()]));
    let ok = good(&tape, &z2).unwrap();
    assert!(
        tape.backward_create_graph(&ok, &child).is_ok(),
        "{ctx}: 再試行"
    );
}

#[test]
fn loss_class_mandatory_vectors_three_stages() {
    // §40.12 の 3 つの反例: 拒否（Err(Backward)・child 空）→ 安全な入力で再試行成功。
    assert_rejected_then_retry_ok(
        "multi_margin p=2 H overflow",
        |_, z| {
            z.reshape(&[1, 2])?.multi_margin_loss(
                &i32t(vec![0], &[1]),
                &MultiMarginOptions::default()
                    .p(2)
                    .margin(0.0)
                    .weight(t(vec![1e-38, 1.0], &[2])),
                Reduction::Sum,
            )
        },
        &[-1e38, 1e38],
        |_, z| {
            z.reshape(&[1, 2])?.multi_margin_loss(
                &i32t(vec![0], &[1]),
                &MultiMarginOptions::default().p(2),
                Reduction::Sum,
            )
        },
        &[0.3, 0.5],
    );
    assert_rejected_then_retry_ok(
        "multilabel_soft weight=1e30 g=1e10",
        |tape, z| {
            z.reshape(&[1, 1])?
                .multilabel_soft_margin_loss(
                    &t(vec![0.5], &[1, 1]),
                    &MultiLabelSoftMarginOptions::default().weight(t(vec![1e30], &[1])),
                    Reduction::Mean,
                )?
                .mul(&tape.var_no_grad(&Tensor::scalar(1e10)))
        },
        &[0.0],
        |_, z| {
            z.reshape(&[1, 1])?.multilabel_soft_margin_loss(
                &t(vec![0.5], &[1, 1]),
                &MultiLabelSoftMarginOptions::default(),
                Reduction::Mean,
            )
        },
        &[0.2],
    );
    assert_rejected_then_retry_ok(
        "multi_margin p=1 weight=1e20 g=1e20",
        |tape, z| {
            z.reshape(&[1, 2])?
                .multi_margin_loss(
                    &i32t(vec![0], &[1]),
                    &MultiMarginOptions::default()
                        .p(1)
                        .margin(1.0)
                        .weight(t(vec![1e20, 1e20], &[2])),
                    Reduction::Sum,
                )?
                .mul(&tape.var_no_grad(&Tensor::scalar(1e20)))
        },
        &[0.0, 0.0],
        |_, z| {
            z.reshape(&[1, 2])?.multi_margin_loss(
                &i32t(vec![0], &[1]),
                &MultiMarginOptions::default(),
                Reduction::Sum,
            )
        },
        &[0.0, 0.0],
    );

    // CEWO: logits ±1e30（受理なら 3 段で有限）・g = ±3e38・巨大 class_weight。
    for x in [1e30f32, -1e30] {
        try_case(
            &format!("cewo logits={x}"),
            |_, z| {
                z.reshape(&[1, 3])?.cross_entropy_loss_with(
                    &i32t(vec![1], &[1]),
                    1,
                    Reduction::Mean,
                    &CrossEntropyOptions::default().label_smoothing(0.1),
                )
            },
            &[x, 0.0, -x],
        );
    }
    for scale in [3e38f32, -3e38] {
        assert!(!try_scaled(
            "cewo g=3e38",
            |z| z.reshape(&[1, 3])?.cross_entropy_loss_with(
                &i32t(vec![1], &[1]),
                1,
                Reduction::Sum,
                &CrossEntropyOptions::default().label_smoothing(0.1),
            ),
            &[0.1, 0.2, 0.3],
            scale
        ));
    }
    assert!(!try_case(
        "cewo class_weight 大",
        |_, z| z.reshape(&[1, 3])?.cross_entropy_loss_with(
            &i32t(vec![1], &[1]),
            1,
            Reduction::Sum,
            &CrossEntropyOptions::default().class_weight(t(vec![1e38, 1e38, 1e38], &[3])),
        ),
        &[0.1, 0.2, 0.3]
    ));
    // Multi*: x = ±1e30。
    try_case(
        "multi_margin x=1e30",
        |_, z| {
            z.reshape(&[1, 2])?.multi_margin_loss(
                &i32t(vec![0], &[1]),
                &MultiMarginOptions::default().p(2),
                Reduction::Sum,
            )
        },
        &[-1e30, 1e30],
    );
    // Cosine: 1e20 の行・零ベクトル行・1e-30 の行・g = ±3e38。
    for (name, x1, x2) in [
        ("1e20", [1e20f32, 1e20], [1e20f32, 1e20]),
        ("zero", [0.0, 0.0], [0.3, 0.1]),
        ("1e-30", [1e-30, 1e-30], [1e-30, -1e-30]),
    ] {
        assert!(
            try_case(
                &format!("cosine {name}"),
                |_, z| cos_loss(z, 1.0, 0.0),
                &[x1.to_vec(), x2.to_vec()].concat(),
            ),
            "{name}"
        );
    }
    for scale in [3e38f32, -3e38] {
        assert!(!try_scaled(
            "cosine g=3e38",
            |z| cos_loss(z, 1.0, 0.0),
            &[0.5, 0.1, 0.2, 0.7],
            scale
        ));
    }
    // Triplet: 差分 1e20・p=1024・差分がちょうど -eps・swap の同値・g = ±3e38。
    try_case(
        "triplet diff 1e20",
        |_, z| tri_loss(z, &tri_opts(2.0, false, 1e-6)),
        &[1e20, 0.0, 0.0, 0.0, 0.0, 1e20],
    );
    try_case(
        "triplet p=1024",
        |_, z| tri_loss(z, &tri_opts(1024.0, false, 1e-6)),
        &[0.5, 0.1, 0.2, 0.7, 0.1, 0.3],
    );
    try_case(
        "triplet diff == -eps",
        |_, z| tri_loss(z, &tri_opts(2.0, false, 1e-6)),
        &[0.0, 0.0, 1e-6, 1e-6, 0.4, 0.2],
    );
    try_case(
        "triplet swap tie",
        |_, z| tri_loss(z, &tri_opts(2.0, true, 0.0)),
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
    for scale in [3e38f32, -3e38] {
        assert!(!try_scaled(
            "triplet g=3e38",
            |z| tri_loss(z, &tri_opts(2.0, false, 1e-6)),
            &[0.5, 0.1, 0.2, 0.7, 0.1, 0.3],
            scale
        ));
    }
}

#[test]
fn loss_class_piecewise_boundaries_follow_first_order_rule() {
    let cmp = |ctx: &str, build: &dyn for<'t> Fn(&Var<'t>) -> BuildResult<'t>, z0: &[f32]| {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(z0.to_vec(), &[z0.len()]));
        let loss = build(&z).unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
        assert_eq!(norm_bits(&got), norm_bits(&want), "{ctx}");
        want
    };
    // MultiMargin: z == 0 ちょうど（x_y = 1・x_j = 0・margin = 1）は活性でない（厳密 z > 0）。
    let w = cmp(
        "multi_margin z==0",
        &|z| {
            z.reshape(&[1, 2])?.multi_margin_loss(
                &i32t(vec![0], &[1]),
                &MultiMarginOptions::default(),
                Reduction::Sum,
            )
        },
        &[1.0, 0.0],
    );
    assert!(w.iter().all(|v| *v == 0.0));
    // MultiLabelMargin: z == 0 ちょうど。
    let w = cmp(
        "multilabel_margin z==0",
        &|z| {
            z.reshape(&[1, 3])?
                .multilabel_margin_loss(&i32t(vec![0, -1, 0], &[1, 3]), Reduction::Sum)
        },
        &[1.0, 0.0, 0.0],
    );
    assert!(w.iter().all(|v| *v == 0.0));
    // Cosine: y = -1 で cos == margin（cos = 0・margin = 0）は活性（>=）。
    let w = cmp(
        "cosine cos==margin",
        &|z| {
            let p = parts(z, 2, 2);
            p[0].reshape(&[1, 2])?.cosine_embedding_loss(
                &p[1].reshape(&[1, 2])?,
                &t(vec![-1.0], &[1]),
                0.0,
                Reduction::Sum,
            )
        },
        &[1.0, 0.0, 0.0, 1.0],
    );
    assert!(w.iter().any(|v| *v != 0.0));
    // Triplet: hinge 境界（d_ap − d_an + margin == 0）は活性（>=）。d_ap = 1・d_an = 2・margin = 1。
    let w = cmp(
        "triplet hinge==0",
        &|z| {
            let q = parts(z, 3, 2);
            q[0].reshape(&[1, 2])?.triplet_margin_loss(
                &q[1].reshape(&[1, 2])?,
                &q[2].reshape(&[1, 2])?,
                &TripletMarginOptions::default().eps(0.0).margin(1.0),
                Reduction::Sum,
            )
        },
        &[0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
    );
    assert!(w.iter().any(|v| *v != 0.0));
    // Triplet swap の同値タイは 0.5 等分。
    cmp(
        "triplet swap tie",
        &|z| {
            let q = parts(z, 3, 2);
            q[0].reshape(&[1, 2])?.triplet_margin_loss(
                &q[1].reshape(&[1, 2])?,
                &q[2].reshape(&[1, 2])?,
                &TripletMarginOptions::default()
                    .eps(0.0)
                    .swap(true)
                    .margin(2.0),
                Reduction::Sum,
            )
        },
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
}

/// ignore 行・全クラス重み 0・`Mean && W==0` で logits に NaN があっても 0 を返す（NaN を出さない）。
#[test]
fn cewo_zero_weight_paths_return_zero_not_nan() {
    let cases: Vec<(&str, CrossEntropyOptions, Reduction)> = vec![
        (
            "all-zero weight",
            CrossEntropyOptions::default().class_weight(t(vec![0.0, 0.0, 0.0], &[3])),
            Reduction::Sum,
        ),
        (
            "mean && W==0 (target weight 0)",
            CrossEntropyOptions::default()
                .label_smoothing(0.1)
                .class_weight(t(vec![0.0, 1.0, 1.0], &[3])),
            Reduction::Mean,
        ),
        (
            "mean && all ignored",
            CrossEntropyOptions::default().ignore_index(0),
            Reduction::Mean,
        ),
    ];
    for (name, opts, r) in cases {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(vec![f32::NAN, 1.0, 2.0], &[3]));
        let loss = z
            .reshape(&[1, 3])
            .unwrap()
            .cross_entropy_loss_with(&i32t(vec![0], &[1]), 1, r, &opts)
            .unwrap();
        let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
        assert!(want.iter().all(|v| *v == 0.0), "{name}: 1 階");
        assert_eq!(norm_bits(&got), norm_bits(&want), "{name}");
    }
}

#[test]
fn loss_class_unsafe_inputs_rejected_child_empty_and_retry() {
    // ±inf の logits／入力は拒否。
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        assert_rejected_then_retry_ok(
            "cewo inf",
            good_cewo,
            &[bad, 0.0, 1.0],
            good_cewo,
            &[0.1, 0.2, 0.3],
        );
        assert_rejected_then_retry_ok(
            "triplet inf",
            tri_big_margin,
            &[bad, 0.0, 0.1, 0.2, 0.3, 0.4],
            tri_big_margin,
            &[0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
        );
    }
    // Triplet の 1<p<2 で差分が 0 のレーンは拒否。
    assert_rejected_then_retry_ok(
        "triplet 1<p<2 u==0",
        tri_p15,
        &[0.5, 0.2, 0.5, 0.6, 0.1, 0.3],
        tri_p15,
        &[0.5, 0.2, 0.4, 0.6, 0.1, 0.3],
    );
    // NaN は伝播し、値のクラスが 1 階と一致する（V3 は bit 一致）。
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![f32::NAN, 0.5, 0.4], &[3]));
    let loss = good_cewo(&tape, &z).unwrap();
    let want = dense(tape.backward(&loss).unwrap().get(&z).unwrap().unwrap());
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let got = dense(cg.grad(&z).unwrap().unwrap().to_tensor());
    assert_eq!(norm_bits(&got), norm_bits(&want));
}

#[test]
fn loss_class_empty_tensor_returns_zeros() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![], &[0, 3]));
    let x2 = tape.var(&t(vec![], &[0, 3]));
    let x3 = tape.var(&t(vec![], &[0, 3]));
    let tg_f = t(vec![], &[0, 3]);
    let y0 = t(vec![], &[0]);
    for loss in [
        x.cross_entropy_loss_with(
            &i32t(vec![], &[0]),
            1,
            Reduction::Sum,
            &CrossEntropyOptions::default().label_smoothing(0.1),
        )
        .unwrap(),
        x.multi_margin_loss(
            &i32t(vec![], &[0]),
            &MultiMarginOptions::default(),
            Reduction::Sum,
        )
        .unwrap(),
        x.multilabel_margin_loss(&i32t(vec![], &[0, 3]), Reduction::Sum)
            .unwrap(),
        x.multilabel_soft_margin_loss(
            &tg_f,
            &MultiLabelSoftMarginOptions::default(),
            Reduction::Sum,
        )
        .unwrap(),
        x.cosine_embedding_loss(&x2, &y0, 0.0, Reduction::Sum)
            .unwrap(),
        x.triplet_margin_loss(&x2, &x3, &TripletMarginOptions::default(), Reduction::Sum)
            .unwrap(),
    ] {
        let child = Tape::new_with_ops(common::naive_ops());
        let cg = tape.backward_create_graph(&loss, &child).unwrap();
        let g = cg.grad(&x).unwrap().unwrap();
        assert_eq!(g.value().shape(), &[0usize, 3]);
    }
}

// ---------------------------------------------------------------------------------
// 隣接の fail-closed
// ---------------------------------------------------------------------------------

struct IdentityCustomFn;

impl fandhe_ai_autodiff::CustomFunction for IdentityCustomFn {
    fn name(&self) -> &str {
        "identity"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok((*inputs[0]).clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![Some(upstream.clone())])
    }
}

#[test]
fn loss_class_neighbouring_unsupported_op_stays_fail_closed() {
    for c in cases().into_iter().filter(|c| c.name.ends_with("mean")) {
        let tape = Tape::new_with_ops(common::naive_ops());
        let child = Tape::new_with_ops(common::naive_ops());
        let z = tape.var(&t(c.z0.clone(), &[c.z0.len()]));
        let l = (c.build)(&z).unwrap();
        let out = tape.custom(Arc::new(IdentityCustomFn), &[l]).unwrap();
        let err = tape.backward_create_graph(&out, &child).unwrap_err();
        assert!(
            matches!(err, AutodiffError::Backward(_)),
            "{}: {err:?}",
            c.name
        );
        assert!(child.is_empty(), "{}", c.name);
    }
}

/// Triplet の差分が f32 の Sub→Add 再構成で桁落ちしてゼロになる入力は拒否する（0/0 の二階 NaN を防ぐ）。
#[test]
fn triplet_cancellation_in_f32_reconstruction_is_rejected() {
    fn build<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        tri_loss(
            z,
            &TripletMarginOptions::default()
                .p(2.0)
                .eps(1.0)
                .margin(2.0)
                .swap(false),
        )
    }
    assert_rejected_then_retry_ok(
        "triplet cancellation",
        build,
        &[1e-8, 0.0, 1.0, 1.0, 2.0, 0.0],
        build,
        &[0.5, 0.25, 1.0, 1.5, 2.0, 0.75],
    );
}

/// Triplet の逆スケール（sb·inv_mx）が f32 範囲を超える二階は拒否する。
#[test]
fn triplet_inverse_scale_second_order_overflow_is_rejected() {
    fn build<'t>(tape: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        tri_loss(
            z,
            &TripletMarginOptions::default().p(2.0).eps(0.0).margin(1.0),
        )?
        .mul(&tape.var_no_grad(&Tensor::scalar(1e10)))?
        .sum(None)
    }
    assert_rejected_then_retry_ok(
        "triplet inverse scale",
        build,
        &[0.0, 0.0, 1e-29, 0.0, 0.0, 2e-29],
        build,
        &[0.0, 0.0, 0.5, 0.0, 0.0, 0.25],
    );
}

/// MultiMargin p=2 の Z が f32 再構成で桁落ちする入力は拒否する（非 root の二階が H=0 で誤るのを防ぐ）。
#[test]
fn multi_margin_p2_cancellation_in_f32_reconstruction_is_rejected() {
    fn build<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        z.reshape(&[1, 2])?.multi_margin_loss(
            &i32t(vec![0], &[1]),
            &MultiMarginOptions::default().p(2).margin(1.0),
            Reduction::Sum,
        )
    }
    assert_rejected_then_retry_ok(
        "multi_margin p=2 cancellation",
        build,
        &[1e8, 1e8],
        build,
        &[0.3, 0.5],
    );
}

/// Triplet 1<p<2 で微小な非ゼロ成分が曲率の負べきを f32 範囲外にする入力は拒否する。
#[test]
fn triplet_p_between_1_and_2_tiny_component_is_rejected() {
    fn build<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        tri_loss(
            z,
            &TripletMarginOptions::default()
                .p(1.1)
                .eps(0.0)
                .margin(2.0)
                .swap(false),
        )
    }
    assert_rejected_then_retry_ok(
        "triplet p=1.1 tiny component",
        build,
        &[f32::from_bits(1), 1.0, 0.0, 0.0, 1.0, 2.0],
        build,
        &[0.5, 0.25, 1.0, 1.5, 2.0, 0.75],
    );
}

/// Triplet の f32 丸め後に最大成分が 1 を僅かに下回り、巨大な p で `|u|^p` が 0 に潰れる入力は拒否する
/// （norm=0 の Div で二階が NaN になるのを防ぐ。Codex P1）。
#[test]
fn triplet_large_p_f32_rounding_underflow_is_rejected() {
    fn build<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        tri_loss(
            z,
            &TripletMarginOptions::default()
                .p(1e10)
                .eps(0.0)
                .margin(3.0)
                .swap(false),
        )
    }
    assert_rejected_then_retry_ok(
        "triplet large p underflow",
        build,
        &[0.0, 0.0, -1.27, 0.0, -2.0, 0.0],
        build,
        &[0.5, 0.25, 1.0, 1.5, 2.0, 0.75],
    );
}

/// Triplet の累乗和 `ssum` が f64 では有限でも f32 の `PowScalar` で overflow する入力
/// （差分の f32 再構成と逆スケールの積が 1 を僅かに超え、p=1e9 で `|u|^p` が約 1e51）は、
/// 逆伝播の 0×inf による NaN を避けるため型付きエラーで拒否する（Codex P1）。
#[test]
fn triplet_large_p_f32_pow_overflow_is_rejected() {
    fn build<'t>(_: &'t Tape, z: &Var<'t>) -> BuildResult<'t> {
        let q = parts(z, 3, 1);
        q[0].reshape(&[1])?.triplet_margin_loss(
            &q[1].reshape(&[1])?,
            &q[0].reshape(&[1])?,
            &TripletMarginOptions::default()
                .p(1e9)
                .eps(1e-6)
                .margin(1.0)
                .swap(false),
            Reduction::Sum,
        )
    }
    assert_rejected_then_retry_ok(
        "triplet large p pow overflow",
        build,
        &[0.028_971_935, -0.456_534_95, 0.028_971_935],
        build,
        &[0.5, 0.25, 1.0],
    );
}

/// ignore 行の logits に NaN があっても二階が NaN にならない。
#[test]
fn cewo_ignore_row_nan_logits_keep_second_order_finite() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let child = Tape::new_with_ops(common::naive_ops());
    let z = tape.var(&t(vec![f32::NAN, 0.0, 0.0, 1.0], &[4]));
    let loss = z
        .reshape(&[2, 2])
        .unwrap()
        .cross_entropy_loss_with(
            &i32t(vec![-1, 0], &[2]),
            1,
            Reduction::Sum,
            &CrossEntropyOptions::default().ignore_index(-1),
        )
        .unwrap();
    let cg = tape.backward_create_graph(&loss, &child).unwrap();
    let gz = cg.grad(&z).unwrap().unwrap();
    let s = gz.sum(None).unwrap();
    let cz = cg.child_var(&z).unwrap().unwrap();
    let h = child.backward(&s).unwrap();
    let row = dense(h.get(&cz).unwrap().unwrap());
    assert!(row[2..].iter().all(|v| v.is_finite()), "{row:?}");
    assert_eq!(&row[..2], &[0.0, 0.0], "{row:?}");
}
