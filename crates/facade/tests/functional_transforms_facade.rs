//! facade `Tape::vjp`・`Tape::hvp`・`Tape::vmap`（イシュー #2931。決定記録
//! `docs/autodiff-functional-transforms-design.md` §23）の結合テスト。
//!
//! 属性なし（CI で実行）: facade 経由の結果が内部実装（`fandhe_ai_autodiff::functional_ops::*` に
//! CPU バックエンドのテープを渡した結果）と shape・値ともに一致すること、手計算値、入口検査のエラー伝播、
//! 追跡なしの `output`／`loss` に対する `vjp`／`hvp` の非対称（現状のまま・契約は変えない）、`vmap` の
//! 結果が同じテープ上で微分可能であることを固定する。数値の比較は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）を使い、tolerance は追加・変更しない。
//!
//! `#[ignore]`（CUDA／Metal 実機の facade 経由 3 変換）: 実機への到達手段が本エージェント実行環境に
//! ないため未実測のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/functional-transforms-facade-2931/README.md`）。内部 API 層の実機 parity は
//! `functional_ops_backend_parity.rs`（#2881）が担当し、本ファイルは facade 層を担当する。
//!
//! `Tape::vmap_grad`（#3072・§32）の結合テスト（#3073）も本ファイルが担当する。参照は `vmap` 系を
//! 経由しない明示ループ（スライスごとに新しいテープで `backward`）で、判定は `assert_parity` のみ
//! （tolerance は足さず変えない）。実機は `docs/perf/logs/functional-transforms-vmap-grad-3073/README.md`
//! へ申し送る。`DeviceMismatch`・結果要素数オーバーフロー・子テープの世代は facade から観測できず未検証
//! （§32.9・§32.10 と同じ）。

use fandhe_ai::{AutodiffError, Device, ShapeError, Tape, TapeRef, Tensor, Var, tape, tape_for};
use fandhe_ai_autodiff::Tape as InnerTape;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

/// 3 変換の出力（shape と値）。
struct Outs {
    vjp: (Vec<usize>, Vec<f32>),
    hvp: (Vec<usize>, Vec<f32>),
    vmap: (Vec<usize>, Vec<f32>),
}

fn pack(x: &Tensor<f32>) -> (Vec<usize>, Vec<f32>) {
    (x.shape().to_vec(), x.host_slice().into_owned())
}

/// `functional_ops_backend_parity.rs::compute` と同じ matmul・tanh・sigmoid の合成を facade 経由で評価する。
fn compute_facade(tape: &Tape, child: &Tape) -> Outs {
    let w1 = tape.var_no_grad(&t(
        (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
        &[3, 4],
    ));
    let w2 = tape.var_no_grad(&t(
        (0..8).map(|i| 0.2 - 0.05 * (i as f32)).collect(),
        &[4, 2],
    ));
    let x = tape.var(&t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]));
    let y = x.matmul(&w1).unwrap().tanh().matmul(&w2).unwrap();
    let u = t(vec![0.5, -1.0, 0.25, 2.0], &[2, 2]);
    let g = tape.vjp(&y, &x, &u).unwrap();

    let loss = y.sigmoid().sum(None).unwrap();
    let v = t(vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5], &[2, 3]);
    let hv = tape.hvp(&loss, &x, &v, child).unwrap();

    let w = tape.var_no_grad(&t(
        (0..6).map(|i| 0.3 - 0.1 * (i as f32)).collect(),
        &[3, 2],
    ));
    let batch = tape.var(&t(
        (0..18).map(|i| 0.05 * (i as f32) - 0.4).collect(),
        &[3, 2, 3],
    ));
    let m = tape.vmap(&batch, 0, |s| Ok(s.matmul(&w)?.tanh())).unwrap();
    Outs {
        vjp: pack(&g),
        hvp: pack(&hv),
        vmap: pack(&m.to_tensor()),
    }
}

/// 内部実装を直接呼んだ結果（facade と同じ CPU バックエンドのテープ）。
fn compute_internal() -> Outs {
    use fandhe_ai_autodiff::functional_ops::{hvp, vjp, vmap};
    let new = || InnerTape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()));
    let (tape, child) = (new(), new());
    let w1 = tape.var_no_grad(&t(
        (0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(),
        &[3, 4],
    ));
    let w2 = tape.var_no_grad(&t(
        (0..8).map(|i| 0.2 - 0.05 * (i as f32)).collect(),
        &[4, 2],
    ));
    let x = tape.var(&t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]));
    let y = x.matmul(&w1).unwrap().tanh().matmul(&w2).unwrap();
    let u = t(vec![0.5, -1.0, 0.25, 2.0], &[2, 2]);
    let g = vjp(&tape, &y, &x, &u).unwrap();
    let loss = y.sigmoid().sum(None).unwrap();
    let v = t(vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.5], &[2, 3]);
    let hv = hvp(&tape, &loss, &x, &v, &child).unwrap();
    let w = tape.var_no_grad(&t(
        (0..6).map(|i| 0.3 - 0.1 * (i as f32)).collect(),
        &[3, 2],
    ));
    let batch = tape.var(&t(
        (0..18).map(|i| 0.05 * (i as f32) - 0.4).collect(),
        &[3, 2, 3],
    ));
    let m = vmap(&tape, &batch, 0, |s| Ok(s.matmul(&w)?.tanh())).unwrap();
    Outs {
        vjp: pack(&g),
        hvp: pack(&hv),
        vmap: pack(&m.to_tensor()),
    }
}

fn assert_outs_match(label: &str, a: &Outs, b: &Outs) {
    let p = fandhe_ai_backend_cpu::parity::assert_parity;
    for (name, x, y) in [
        ("vjp", &a.vjp, &b.vjp),
        ("hvp", &a.hvp, &b.hvp),
        ("vmap", &a.vmap, &b.vmap),
    ] {
        assert_eq!(x.0, y.0, "{label}: {name} shape");
        p(&format!("{label}: {name}"), &x.1, &y.1);
    }
}

#[test]
fn facade_cpu_matches_internal_functional_ops() {
    let facade = compute_facade(&tape(), &tape());
    let internal = compute_internal();
    assert_eq!(facade.vjp.0, vec![2, 3]);
    assert_eq!(facade.hvp.0, vec![2, 3]);
    assert_eq!(facade.vmap.0, vec![3, 2, 2]);
    assert_outs_match("facade vs internal", &facade, &internal);
}

/// 手計算の期待値（両経路が同じ誤りで一致していないことの確認）。
#[test]
fn hand_computed_values() {
    let t0 = tape();
    let child = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0], &[3]));
    let y = x.mul(&x).unwrap();
    let g = t0.vjp(&y, &x, &t(vec![1.0, 0.5, 2.0], &[3])).unwrap();
    assert_eq!(g.host_slice().as_ref(), &[2.0, 2.0, 12.0]);

    let x3 = t0.var(&t(vec![1.0, 2.0], &[2]));
    let loss = x3.mul(&x3).unwrap().mul(&x3).unwrap().sum(None).unwrap();
    let hv = t0
        .hvp(&loss, &x3, &t(vec![1.0, 0.0], &[2]), &child)
        .unwrap();
    assert_eq!(hv.host_slice().as_ref(), &[6.0, 0.0]);

    let b = t0.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let m = t0.vmap(&b, 0, |s| s.mul(s)).unwrap();
    assert_eq!(m.to_tensor().shape(), &[2, 2]);
    assert_eq!(m.to_tensor().host_slice().as_ref(), &[1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn input_validation_errors_propagate() {
    let t0 = tape();
    let t1 = tape();
    let child = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2]));
    let y = x.mul(&x).unwrap();
    let u = t(vec![1.0, 1.0], &[2]);

    // 別テープの Var。
    let foreign = t1.var(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        t0.vjp(&y, &foreign, &u),
        Err(AutodiffError::TapeMismatch)
    ));
    let loss = y.sum(None).unwrap();
    assert!(matches!(
        t0.hvp(&loss, &foreign, &u, &child),
        Err(AutodiffError::TapeMismatch)
    ));

    // 追跡なしの input は両方 GradientTrackingDisabled。
    let nograd = t0.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        t0.vjp(&y, &nograd, &u),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(matches!(
        t0.hvp(&loss, &nograd, &u, &child),
        Err(AutodiffError::GradientTrackingDisabled)
    ));

    // cotangent／vector の shape 不一致（ブロードキャストなし）。
    let bad = t(vec![1.0, 1.0, 1.0], &[3]);
    assert!(matches!(t0.vjp(&y, &x, &bad), Err(AutodiffError::Shape(_))));
    let child2 = tape();
    let loss2 = x.mul(&x).unwrap().mul(&x).unwrap().sum(None).unwrap();
    assert!(matches!(
        t0.hvp(&loss2, &x, &bad, &child2),
        Err(AutodiffError::Shape(_))
    ));

    // hvp の非スカラー loss。
    let child3 = tape();
    assert!(matches!(
        t0.hvp(&y, &x, &u, &child3),
        Err(AutodiffError::InvalidArgument(_))
    ));
}

#[test]
fn vmap_errors_propagate() {
    let t0 = tape();
    let t1 = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    // in_dim >= rank。
    assert!(matches!(
        t0.vmap(&x, 2, |s| s.mul(s)),
        Err(AutodiffError::Shape(_))
    ));
    // 空バッチ。
    let empty = t0.var(&t(vec![], &[0, 2]));
    assert!(matches!(
        t0.vmap(&empty, 0, |s| s.mul(s)),
        Err(AutodiffError::InvalidArgument(_))
    ));
    // 別テープの入力。
    let foreign = t1.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    assert!(matches!(
        t0.vmap(&foreign, 0, |s| s.mul(s)),
        Err(AutodiffError::TapeMismatch)
    ));
    // クロージャの Err はそのまま伝播する。
    let r = t0.vmap(&x, 0, |_s| Err(AutodiffError::Backward("closure".into())));
    assert!(matches!(r, Err(AutodiffError::Backward(m)) if m == "closure"));
}

/// 追跡なしの `output`（`vjp`）と `loss`（`hvp`）の非対称（現状のまま・契約は変えない。§23.3）。
#[test]
fn untracked_output_and_loss_asymmetry() {
    let t0 = tape();
    let child = tape();
    let x = t0.var(&t(vec![1.0, 2.0], &[2]));
    let c = t0.var_no_grad(&t(vec![3.0, 4.0], &[2]));
    let u = t(vec![1.0, 1.0], &[2]);

    // 追跡なし output の vjp は全ゼロの Ok。
    let untracked_out = c.mul(&c).unwrap();
    let g = t0.vjp(&untracked_out, &x, &u).unwrap();
    assert_eq!(g.shape(), &[2]);
    assert_eq!(g.host_slice().as_ref(), &[0.0, 0.0]);

    // 追跡なし loss の hvp は backward_create_graph の Err を伝播する。
    let untracked_loss = c.mul(&c).unwrap().sum(None).unwrap();
    assert!(
        t0.hvp(&untracked_loss, &x, &u, &child).is_err(),
        "追跡なし loss の hvp は Err"
    );
}

/// `vmap` の結果は同じテープ上の微分可能な `Var`（`sum` → `backward` で入力勾配が得られる）。
#[test]
fn vmap_result_is_differentiable_on_the_same_tape() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let y = t0.vmap(&x, 0, |s| s.mul(s)).unwrap();
    let loss = y.sum(None).unwrap();
    let grads = t0.backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().expect("入力勾配あり");
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(gx.host_slice().as_ref(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/functional-transforms-facade-2931/README.md`）。
// ---------------------------------------------------------------------

/// 実機テープ（子テープも同一デバイス）での facade 経由 3 変換と CPU の結果を REQ-2 判定で比較する。
fn assert_device_matches_cpu(device: Device, label: &str) {
    let cpu = compute_facade(&tape(), &tape());
    let dev_tape: Tape = tape_for(device).expect("実機テープ生成");
    let dev_child: Tape = tape_for(device).expect("実機子テープ生成");
    let dev = compute_facade(&dev_tape, &dev_child);
    assert_outs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// facade 経由の vjp・hvp・vmap の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-facade-2931/README.md 参照"]
fn metal_functional_transforms_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// facade 経由の vjp・hvp・vmap の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-facade-2931/README.md 参照"]
fn cuda_functional_transforms_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}

/// `vmap` のクロージャは `FnMut`（呼び出し回数を数える可変キャプチャを許す）。バッチ軸の長さだけ呼ばれる。
#[test]
fn vmap_accepts_fnmut_closure_called_once_per_slice() {
    let t0 = tape();
    let x = t0.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]));
    let mut calls = 0usize;
    let y: Result<Var<'_>, AutodiffError> = t0.vmap(&x, 0, |s| {
        calls += 1;
        s.mul(s)
    });
    assert!(y.is_ok());
    assert_eq!(calls, 3);
}
// ---------------------------------------------------------------------
// `Tape::vmap_grad`（イシュー #3073・設計記録 §32）: ループ参照との一致・エラー経路・実機 parity
// ---------------------------------------------------------------------

type LossFn = for<'c> fn(TapeRef<'c>, &Var<'c>) -> Result<Var<'c>, AutodiffError>;

/// 決定的な入力系列（内部単体テスト `functional_ops_vmap_grad.rs` と同じ `sin` 系列）。
fn seq(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.37).sin() * 1.5)
        .collect()
}

fn vg_loss_sq<'c>(_t: TapeRef<'c>, s: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    s.mul(s)?.sum(None)
}

fn vg_loss_tanh<'c>(_t: TapeRef<'c>, s: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    s.tanh().sum(None)
}

/// 要素数 3 のスライスを `[1, 3] @ [3, 2]` で射影する matmul 合成（定数は `var_no_grad`）。
fn vg_loss_matmul<'c>(tr: TapeRef<'c>, s: &Var<'c>) -> Result<Var<'c>, AutodiffError> {
    let w = tr.var_no_grad(&t(seq(6, 3.0), &[3, 2]));
    s.reshape(&[1, 3])?.matmul(&w)?.tanh().sum(None)
}

fn vg_cases() -> [(&'static str, LossFn); 3] {
    [
        ("sq", vg_loss_sq),
        ("tanh", vg_loss_tanh),
        ("matmul", vg_loss_matmul),
    ]
}

const VG_SHAPES: [(&[usize], usize); 5] = [
    (&[1, 3], 0),
    (&[4, 3], 0),
    (&[3, 4], 1),
    (&[2, 3, 1], 0),
    (&[3, 2, 1], 1),
];

fn slice_shape_of(shape: &[usize], in_dim: usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter(|&(a, _)| a != in_dim)
        .map(|(_, &d)| d)
        .collect()
}

/// 参照: `vmap`／`vmap_grad`／`vjp` を経由せず、スライスごとに新しいテープで `backward` する
/// 明示ループ（§32.4・§23.3 の「`vmap` 実装を経由しない参照」）。結果は dim 0 方向に連結する。
fn vg_loop_reference(input: &Tensor<f32>, in_dim: usize, f: LossFn) -> Vec<f32> {
    let b = input.shape()[in_dim];
    let slice_shape = slice_shape_of(input.shape(), in_dim);
    let mut out = Vec::new();
    for i in 0..b {
        let sl = input
            .narrow(in_dim, i, 1)
            .expect("test fixture: narrow")
            .contiguous()
            .reshape(&slice_shape)
            .expect("test fixture: reshape");
        let tp = tape();
        let s = tp.var(&sl);
        let loss = f(TapeRef::from(&tp), &s).expect("test fixture: loss");
        let g = tp.backward(&loss).expect("test fixture: backward");
        out.extend(
            g.get(&s)
                .expect("test fixture: get")
                .expect("勾配あり")
                .host_slice()
                .iter()
                .copied(),
        );
    }
    out
}

/// 損失 × 形状の表で `Tape::vmap_grad`（子テープ `child` 使い回し）を CPU ループ参照と比較する。
/// 各ケースの結果を返し、呼び出し側が別デバイスの結果と比較できるようにする。
fn vg_run_table(child: &mut Tape, label: &str) -> Vec<(String, Vec<f32>)> {
    let mut all = Vec::new();
    for (name, f) in vg_cases() {
        for (shape, in_dim) in VG_SHAPES {
            let n: usize = shape.iter().product();
            let x = t(seq(n, 1.0), shape);
            let got = Tape::vmap_grad(&x, in_dim, child, f).expect("vmap_grad 成功");
            let mut exp_shape = vec![shape[in_dim]];
            exp_shape.extend(slice_shape_of(shape, in_dim));
            let ctx = format!("{label} {name} {shape:?} in_dim={in_dim}");
            assert_eq!(got.shape(), exp_shape, "{ctx}: shape");
            let got = got.host_slice().into_owned();
            fandhe_ai_backend_cpu::parity::assert_parity(
                &ctx,
                &got,
                &vg_loop_reference(&x, in_dim, f),
            );
            all.push((ctx, got));
        }
    }
    all
}

#[test]
fn vmap_grad_matches_loop_reference_across_losses_and_dims() {
    let mut child = tape();
    vg_run_table(&mut child, "cpu");
    assert_eq!(child.leaf_count(), 0, "戻り時の子テープは空");
}

#[test]
fn vmap_grad_sum_of_squares_matches_closed_form() {
    let x = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let mut child = tape();
    let g = Tape::vmap_grad(&x, 0, &mut child, vg_loss_sq).expect("成功");
    assert_eq!(g.shape(), &[2, 4]);
    let expected: Vec<f32> = x.host_slice().iter().map(|v| 2.0 * v).collect();
    fandhe_ai_backend_cpu::parity::assert_parity("2x", g.host_slice().as_ref(), &expected);
}

#[test]
fn vmap_grad_errors_propagate() {
    let mut child = tape();
    let x = t(seq(6, 1.0), &[2, 3]);

    // Phase A: 範囲外の in_dim（rank 2 の axis 2、rank 0 の axis 0）
    let r = Tape::vmap_grad(&x, 2, &mut child, vg_loss_sq);
    assert!(
        matches!(
            r,
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ),
        "{r:?}"
    );
    let scalar = t(vec![1.0], &[]);
    let r = Tape::vmap_grad(&scalar, 0, &mut child, vg_loss_sq);
    assert!(
        matches!(
            r,
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ),
        "{r:?}"
    );

    // Phase A: 空バッチ
    let empty = t(vec![], &[0, 3]);
    let r = Tape::vmap_grad(&empty, 0, &mut child, vg_loss_sq);
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");

    // Phase B: 1 要素でない損失
    let r = Tape::vmap_grad(&x, 0, &mut child, |_t, s| Ok(s.tanh()));
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");
    assert_eq!(child.leaf_count(), 0);

    // Phase B: 別テープの損失（要素数検査より先に TapeMismatch）
    let other: &'static Tape = Box::leak(Box::new(tape()));
    let foreign = other.var(&t(vec![1.0, 2.0], &[2]));
    let r = Tape::vmap_grad(&x, 0, &mut child, |_t, _s| Ok(foreign));
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)), "{r:?}");
    assert_eq!(child.leaf_count(), 0);

    // Phase B: クロージャの Err がそのまま伝播する
    let r = Tape::vmap_grad(&x, 0, &mut child, |_t, _s| {
        Err(AutodiffError::Backward("closure".to_string()))
    });
    match r {
        Err(AutodiffError::Backward(m)) => assert_eq!(m, "closure"),
        other => panic!("closure の Err が伝播すべき: {other:?}"),
    }

    // 失敗後も同じ子テープで成功呼び出しが通る（空に戻っている）
    let ok = Tape::vmap_grad(&x, 0, &mut child, vg_loss_sq).expect("失敗後の再利用");
    assert_eq!(ok.shape(), &[2, 3]);
    assert_eq!(child.leaf_count(), 0);

    // Phase A: 空でない子テープ
    let mut dirty = tape();
    {
        let _leaf = dirty.var(&t(vec![1.0], &[1]));
    }
    let r = Tape::vmap_grad(&x, 0, &mut dirty, vg_loss_sq);
    assert!(matches!(r, Err(AutodiffError::Backward(_))), "{r:?}");
}

#[test]
fn vmap_grad_closure_is_fnmut_called_once_per_slice() {
    let x = t(seq(9, 1.0), &[3, 3]);
    let mut child = tape();
    let mut calls = 0usize;
    let g = Tape::vmap_grad(&x, 0, &mut child, |_t, s| {
        calls += 1;
        s.mul(s)?.sum(None)
    });
    assert!(g.is_ok());
    assert_eq!(calls, 3);
}

/// 実機の子テープで `Tape::vmap_grad` した結果を CPU（子テープ・ループ参照）と REQ-2 判定で比較する。
fn assert_vmap_grad_device_matches_cpu(device: Device, label: &str) {
    let cpu = vg_run_table(&mut tape(), "cpu");
    let mut dev_child: Tape = tape_for(device).expect("実機テープ生成");
    let dev = vg_run_table(&mut dev_child, label);
    for ((ctx, c), (_, d)) in cpu.iter().zip(&dev) {
        fandhe_ai_backend_cpu::parity::assert_parity(&format!("cpu vs {ctx}"), d, c);
    }
}

/// facade 経由 `Tape::vmap_grad` の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/functional-transforms-vmap-grad-3073/README.md 参照"]
fn metal_vmap_grad_matches_cpu_reference() {
    assert_vmap_grad_device_matches_cpu(Device::Metal, "metal");
}

/// facade 経由 `Tape::vmap_grad` の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/functional-transforms-vmap-grad-3073/README.md 参照"]
fn cuda_vmap_grad_matches_cpu_reference() {
    assert_vmap_grad_device_matches_cpu(Device::Cuda(0), "cuda");
}
