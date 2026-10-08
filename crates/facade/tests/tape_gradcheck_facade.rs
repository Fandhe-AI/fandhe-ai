//! facade `Tape::gradcheck`・`GradcheckOptions`・`GradcheckReport`（イシュー #2847。決定記録
//! `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §11・§12）の結合テスト。
//!
//! 属性なし（CI で実行）: facade 経由の結果が内部実装
//! （`fandhe_ai_autodiff::gradcheck::gradcheck` に CPU バックエンドのテープを渡した結果）と
//! 全フィールド一致することを、合格する関数と、わざと誤った勾配を返す関数（`detach` で
//! 解析勾配だけが半分になる）の両方で確認する。入口検査・エラー伝播・テープ生成失敗の経路も固定する。
//! `GradcheckOptions` は #223 承認済みの組のみを使う（tolerance・baseline の追加なし）。
//!
//! `#[ignore]`（CUDA／Metal 実機の `Tape::gradcheck`）: 実機への到達手段が本エージェント実行環境に
//! ないため未実施のまま GB10／Mac セッションへ申し送る
//! （`docs/perf/logs/tape-gradcheck-facade-2847/README.md`）。

use fandhe_ai::{AutodiffError, Device, GradcheckOptions, GradcheckReport, Tape, Tensor};
use fandhe_ai_autodiff::Tape as InnerTape;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn opts() -> GradcheckOptions {
    GradcheckOptions::new(1e-3, 1e-3, 1e-2, 1e-4).expect("承認済みの閾値の組は有効")
}

/// 内部実装の結果（facade と同じ CPU バックエンドのテープを評価ごとに生成する）。
fn internal_cpu_report<F>(f: F, inputs: &[Tensor<f32>]) -> Result<GradcheckReport, AutodiffError>
where
    F: for<'a> Fn(
        &'a InnerTape,
        &[fandhe_ai::Var<'a>],
    ) -> Result<fandhe_ai::Var<'a>, AutodiffError>,
{
    fandhe_ai_autodiff::gradcheck::gradcheck(
        || {
            Ok(InnerTape::new_with_ops(Box::new(
                fandhe_ai_backend_cpu::CpuBackendOps::new(),
            )))
        },
        f,
        inputs,
        &opts(),
    )
}

#[test]
fn passing_function_matches_internal_report_and_hand_values() {
    let inputs = [t(vec![1.0, 2.0], &[2])];
    let report =
        Tape::gradcheck(Device::Cpu, |_t, xs| xs[0].mul(&xs[0]), &inputs, &opts()).unwrap();
    let internal = internal_cpu_report(|_t, xs| xs[0].mul(&xs[0]), &inputs).unwrap();
    assert_eq!(report, internal);
    assert!(report.passed());
    // 出力 2 要素 × 入力 2 要素。
    assert_eq!(report.checked_elements(), 4);
}

#[test]
fn passing_matmul_tanh_with_constants_matches_internal_report() {
    let w1 = t((0..12).map(|i| 0.1 * (i as f32) - 0.5).collect(), &[3, 4]);
    let w2 = t((0..8).map(|i| 0.2 - 0.05 * (i as f32)).collect(), &[4, 2]);
    let x = t(vec![0.3, -0.7, 1.1, 0.5, 0.2, -0.4], &[2, 3]);
    let report = Tape::gradcheck(
        Device::Cpu,
        |t, xs| {
            let (w1, w2) = (t.var_no_grad(&w1), t.var_no_grad(&w2));
            xs[0].matmul(&w1)?.tanh().matmul(&w2)
        },
        std::slice::from_ref(&x),
        &opts(),
    )
    .unwrap();
    let internal = internal_cpu_report(
        |t, xs| {
            let (w1, w2) = (t.var_no_grad(&w1), t.var_no_grad(&w2));
            xs[0].matmul(&w1)?.tanh().matmul(&w2)
        },
        std::slice::from_ref(&x),
    )
    .unwrap();
    assert_eq!(report, internal);
    assert!(report.passed());
    assert_eq!(report.checked_elements(), 2 * 2 * 6);
}

/// 解析勾配は `detach` で片側が落ちて `x`、数値勾配は `2x` になる誤った関数。
#[test]
fn wrong_gradient_function_fails_and_matches_internal_report() {
    let inputs = [t(vec![1.0, 2.0], &[2])];
    let report = Tape::gradcheck(
        Device::Cpu,
        |_t, xs| xs[0].mul(&xs[0].detach()?),
        &inputs,
        &opts(),
    )
    .unwrap();
    let internal = internal_cpu_report(|_t, xs| xs[0].mul(&xs[0].detach()?), &inputs).unwrap();
    assert_eq!(report, internal);
    assert!(!report.passed());
    // 絶対誤差は要素 1.0 で 1、要素 2.0 で 2（最悪要素は後者の対角）。
    assert!(report.max_abs_error() > 0.5, "{report:?}");
    let (_, row, col) = report.worst_location();
    assert_eq!(row, col, "最悪要素は対角のはず: {report:?}");
}

#[test]
fn entry_checks_are_reported_before_tape_creation() {
    // 空の inputs は、デバイスが不正でも `InvalidArgument` が先（入口検査がテープ生成より先）。
    let r = Tape::gradcheck(
        Device::Cuda(usize::MAX),
        |_t, xs| xs[0].mul(&xs[0]),
        &[],
        &opts(),
    );
    assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))), "{r:?}");
}

#[test]
fn closure_errors_and_invalid_outputs_propagate() {
    let inputs = [t(vec![1.0, 2.0], &[2])];
    // クロージャの Err はそのまま伝播する。
    let r = Tape::gradcheck(
        Device::Cpu,
        |_t, _xs| Err(AutodiffError::InvalidArgument("closure".into())),
        &inputs,
        &opts(),
    );
    assert!(
        matches!(&r, Err(AutodiffError::InvalidArgument(m)) if m == "closure"),
        "{r:?}"
    );
    // 別テープの Var を返すと TapeMismatch（`'static` へ leak した別テープの Var は任意の `'a` へ縮められる）。
    let other: &'static fandhe_ai::Tape = Box::leak(Box::new(fandhe_ai::tape()));
    let foreign = other.var(&t(vec![1.0, 2.0], &[2]));
    let r = Tape::gradcheck(Device::Cpu, |_t, _xs| Ok(foreign), &inputs, &opts());
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)), "{r:?}");
}

#[test]
fn tape_creation_failure_is_backend_error() {
    // 範囲外 ordinal は CUDA ドライバ不在の CI でも実機でも `tape_for` が失敗する指定。
    assert!(fandhe_ai::tape_for(Device::Cuda(usize::MAX)).is_err());
    let inputs = [t(vec![1.0, 2.0], &[2])];
    let r = Tape::gradcheck(
        Device::Cuda(usize::MAX),
        |_t, xs| xs[0].mul(&xs[0]),
        &inputs,
        &opts(),
    );
    assert!(matches!(r, Err(AutodiffError::Backend(_))), "{r:?}");
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/tape-gradcheck-facade-2847/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let inputs = [t(vec![0.3, 0.4, 0.5, 1.0, 2.0, -0.5], &[2, 3])];
    fn f<'a>(
        _t: fandhe_ai::TapeRef<'a>,
        xs: &[fandhe_ai::Var<'a>],
    ) -> Result<fandhe_ai::Var<'a>, AutodiffError> {
        xs[0].tanh().mul(&xs[0])
    }
    let cpu = Tape::gradcheck(Device::Cpu, f, &inputs, &opts()).unwrap();
    let dev = Tape::gradcheck(device, f, &inputs, &opts()).unwrap();
    assert_eq!(cpu.passed(), dev.passed(), "{label}: 合否");
    assert_eq!(
        cpu.checked_elements(),
        dev.checked_elements(),
        "{label}: 検査要素数"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/tape-gradcheck-facade-2847/README.md 参照"]
fn metal_gradcheck_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/tape-gradcheck-facade-2847/README.md 参照"]
fn cuda_gradcheck_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
