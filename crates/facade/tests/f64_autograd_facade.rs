//! facade 公開の f64 独立自動微分グラフ `TapeF64`／`VarF64`／`GradientsF64`
//! （イシュー #2599。公開形は `docs/autodiff-var-dtype-multiplexing-design.md`
//! §4.1・§10.1 の推奨案 D-2、承認根拠はルート #2499 の一括承認コメント
//! <https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965>）の
//! 公開経路テスト。
//!
//! 内部クレートを一切 import せず `fandhe_ai::*` の公開 API だけで組む（利用者視点の
//! 到達性確認）。演算・VJP の網羅検証は `crates/autodiff` 側（`f64_autograd` の単体テスト）と
//! `dtype_f64_integration.rs`／`var_f64_autograd_backend_parity.rs` が担い、本ファイルは
//! facade 経由でも同じ契約（解析勾配・エラー variant・f32 グラフとの独立性）が保たれる
//! ことを固定する。
//!
//! 判定方式: 数値比較は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
//! `req2_close`）。tolerance は新設・緩和しない。期待値は手計算の解析解。
//!
//! GPU 実機テストは `#[ignore]`（CI 非実行）。手順は
//! `docs/perf/logs/f64-autograd-facade-2599/README.md`。

use fandhe_ai::{AutodiffError, Device, GradientsF64, Tape, TapeF64, Tensor, VarF64};
use fandhe_ai_backend_cpu::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE};

fn t64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::<f64>::new(data, shape).expect("test fixture: tensor")
}

/// REQ-2 統一複合判定。閾値は backend-cpu の `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` を参照する（再定義しない）。
fn req2_close(a: f64, b: f64) -> bool {
    let abs = (a - b).abs();
    abs < ABSOLUTE_RESCUE_THRESHOLD
        || abs / a.abs().max(b.abs()).max(f64::MIN_POSITIVE) < RELATIVE_TOLERANCE
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: 要素数");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(req2_close(*a, *e), "{what}[{i}]: {a} vs {e}");
    }
}

fn vals(t: &Tensor<f64>) -> Vec<f64> {
    t.as_slice().expect("CPU contiguous").to_vec()
}

fn grad_of(grads: &GradientsF64, v: &VarF64<'_, '_>) -> Vec<f64> {
    vals(grads.get(v).expect("get").expect("勾配あり"))
}

#[test]
fn elementwise_chain_matches_analytic_gradient() {
    // loss = sum((x * y + x) / y ^ 2),  x=[1,2,3], y=[2,4,5]
    let tape = fandhe_ai::tape();
    let g = TapeF64::new(&tape);
    let xs = [1.0, 2.0, 3.0];
    let ys = [2.0, 4.0, 5.0];
    let x = g.var(&t64(xs.to_vec(), &[3]));
    let y = g.var(&t64(ys.to_vec(), &[3]));
    let two = g.var_no_grad(&t64(vec![2.0; 3], &[3]));
    let loss = x
        .mul(&y)
        .unwrap()
        .add(&x)
        .unwrap()
        .div(&y.pow(&two).unwrap())
        .unwrap()
        .sum(None)
        .unwrap();
    // 前向き値: sum(x(y+1)/y^2)
    let expected_loss: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| x * (y + 1.0) / (y * y))
        .sum();
    assert_close(&vals(&loss.value()), &[expected_loss], "loss");
    assert!(loss.shape().is_empty() || loss.shape() == vec![1]);

    let grads = g.backward(&loss).unwrap();
    // d/dx = (y+1)/y^2 ; d/dy = -x(y+2)/y^3
    let dx: Vec<f64> = ys.iter().map(|y| (y + 1.0) / (y * y)).collect();
    let dy: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| -x * (y + 2.0) / (y * y * y))
        .collect();
    assert_close(&grad_of(&grads, &x), &dx, "dx");
    assert_close(&grad_of(&grads, &y), &dy, "dy");
}

#[test]
fn matmul_sum_mean_max_match_analytic_gradient() {
    let tape = fandhe_ai::tape();
    let g = TapeF64::new(&tape);
    let a = g.var(&t64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
    let b = g.var(&t64(vec![1.0, 0.5, -1.0, 2.0, 0.0, 1.0], &[3, 2]));
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), vec![2, 2]);
    // C = A B（手計算）: row0=[1*1+2*(-1)+3*0, 1*0.5+2*2+3*1]=[-1, 7.5]
    //            row1=[4*1+5*(-1)+6*0, 4*0.5+5*2+6*1]=[-1, 18]
    assert_close(&vals(&c.value()), &[-1.0, 7.5, -1.0, 18.0], "C");
    let loss = c.sum(Some(1)).unwrap().mean(None).unwrap();
    let grads = g.backward(&loss).unwrap();
    // loss = mean_i(sum_j C_ij) = (1/2) sum_ij C_ij → dC = 0.5 全要素
    // dA = dC B^T : dA_ik = 0.5 * sum_j B_kj ; dB = A^T dC : dB_kj = 0.5 * sum_i A_ik
    assert_close(
        &grad_of(&grads, &a),
        &[0.75, 0.5, 0.5, 0.75, 0.5, 0.5],
        "dA",
    );
    assert_close(&grad_of(&grads, &b), &[2.5, 2.5, 3.5, 3.5, 4.5, 4.5], "dB");

    // max（全軸）: 勾配は最大要素にのみ流れる。
    let tape2 = fandhe_ai::tape();
    let g2 = TapeF64::new(&tape2);
    let x = g2.var(&t64(vec![1.0, 5.0, 3.0], &[3]));
    let m = x.max(None).unwrap();
    assert_close(&vals(&m.value()), &[5.0], "max");
    let grads2 = g2.backward(&m).unwrap();
    assert_close(&grad_of(&grads2, &x), &[0.0, 1.0, 0.0], "dmax");
}

#[test]
fn var_is_copy_and_value_shape_read_back() {
    let tape = fandhe_ai::tape();
    let g = TapeF64::new(&tape);
    let x = g.var(&t64(vec![1.5, -2.0], &[2]));
    let x2 = x; // Copy
    assert_eq!(x.shape(), x2.shape());
    assert_close(&vals(&x2.value()), &[1.5, -2.0], "value");
}

#[test]
fn backward_rejects_foreign_tape_variable_and_get_detects_mismatch() {
    let tape_a = fandhe_ai::tape();
    let tape_b = fandhe_ai::tape();
    let ga = TapeF64::new(&tape_a);
    let gb = TapeF64::new(&tape_b);
    let xa = ga.var(&t64(vec![1.0, 2.0], &[2]));
    let xb = gb.var(&t64(vec![1.0, 2.0], &[2]));
    // backward: 別テープの loss。
    let loss_b = xb.sum(None).unwrap();
    assert!(matches!(
        ga.backward(&loss_b).err(),
        Some(AutodiffError::TapeMismatch)
    ));
    // get: 別テープの変数。
    let loss_a = xa.sum(None).unwrap();
    let grads = ga.backward(&loss_a).unwrap();
    assert!(matches!(
        grads.get(&xb).err(),
        Some(AutodiffError::TapeMismatch)
    ));
}

#[test]
fn no_grad_leaf_and_unreached_leaf_contracts() {
    let tape = fandhe_ai::tape();
    let g = TapeF64::new(&tape);
    let x = g.var(&t64(vec![1.0, 2.0], &[2]));
    let frozen = g.var_no_grad(&t64(vec![3.0, 4.0], &[2]));
    let unreached = g.var(&t64(vec![9.0], &[1]));
    let loss = x.mul(&frozen).unwrap().sum(None).unwrap();
    let grads = g.backward(&loss).unwrap();
    assert_close(&grad_of(&grads, &x), &[3.0, 4.0], "dx");
    // var_no_grad の葉は勾配追跡不可。
    assert!(matches!(
        grads.get(&frozen).err(),
        Some(AutodiffError::GradientTrackingDisabled)
    ));
    // loss から未到達の葉は Ok(None)。
    assert!(grads.get(&unreached).unwrap().is_none());

    // 勾配追跡対象を持たない loss の backward はエラー。
    let only_frozen = g.var_no_grad(&t64(vec![1.0], &[1]));
    let loss2 = only_frozen.sum(None).unwrap();
    assert!(matches!(
        g.backward(&loss2).err(),
        Some(AutodiffError::Backward(_))
    ));
}

#[test]
fn non_scalar_loss_seeds_all_ones() {
    let tape = fandhe_ai::tape();
    let g = TapeF64::new(&tape);
    let x = g.var(&t64(vec![1.0, 2.0, 3.0], &[3]));
    let y = x.mul(&x).unwrap(); // 非スカラー
    let grads = g.backward(&y).unwrap();
    assert_close(&grad_of(&grads, &x), &[2.0, 4.0, 6.0], "全要素 1 シード");
}

#[test]
fn f32_graph_cast_enters_f64_graph_detached() {
    let tape = fandhe_ai::tape();
    let x32 = tape.var(&Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[3]).unwrap());
    let cast = x32.cast::<f64>().expect("f32 → f64 cast");
    let g = TapeF64::new(&tape);
    let x64 = g.var(&cast);
    let loss = x64.mul(&x64).unwrap().sum(None).unwrap();
    let grads = g.backward(&loss).unwrap();
    assert_close(&grad_of(&grads, &x64), &[2.0, 4.0, 6.0], "f64 側の勾配");
    // f32 側のグラフには勾配が流れない（独立グラフ）。f32 の loss を別途作ると勾配は
    // f64 グラフの backward に一切影響されない。
    let loss32 = x32.mul(&x32).unwrap().sum(None).unwrap();
    let grads32 = tape.backward(&loss32).unwrap();
    let gx = grads32.get(&x32).unwrap().expect("f32 勾配");
    assert_eq!(gx.as_slice().unwrap(), &[2.0f32, 4.0, 6.0]);
}

#[test]
fn rank3_matmul_is_rejected() {
    let tape = fandhe_ai::tape();
    let g = TapeF64::new(&tape);
    let a = g.var(&t64(vec![1.0; 8], &[2, 2, 2]));
    let b = g.var(&t64(vec![1.0; 8], &[2, 2, 2]));
    assert!(a.matmul(&b).is_err(), "rank≥3 のバッチ matmul は対象外");
}

/// GPU 実機: CPU 結果と REQ-2 統一複合判定で比較する（f64 ネイティブ／ホスト経路の差を許容内で確認）。
fn gpu_matches_cpu(device: Device) {
    fn run(tape: &Tape) -> (Vec<f64>, Vec<f64>) {
        let g = TapeF64::new(tape);
        let a = g.var(&t64(
            (0..12).map(|i| i as f64 * 0.25 - 1.0).collect(),
            &[3, 4],
        ));
        let b = g.var(&t64(
            (0..8).map(|i| 1.0 - i as f64 * 0.125).collect(),
            &[4, 2],
        ));
        let loss = a.matmul(&b).unwrap().mean(None).unwrap();
        let grads = g.backward(&loss).unwrap();
        (grad_of(&grads, &a), grad_of(&grads, &b))
    }
    let cpu = fandhe_ai::tape();
    let gpu = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテスト");
    let (ca, cb) = run(&cpu);
    let (ga, gb) = run(&gpu);
    assert_close(&ga, &ca, "dA(gpu vs cpu)");
    assert_close(&gb, &cb, "dB(gpu vs cpu)");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
fn f64_autograd_cuda_matches_cpu() {
    gpu_matches_cpu(Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要"]
fn f64_autograd_metal_matches_cpu() {
    gpu_matches_cpu(Device::Metal);
}
