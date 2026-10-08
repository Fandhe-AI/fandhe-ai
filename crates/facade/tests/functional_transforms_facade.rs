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

use fandhe_ai::{AutodiffError, Device, Tape, Tensor, Var, tape, tape_for};
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
