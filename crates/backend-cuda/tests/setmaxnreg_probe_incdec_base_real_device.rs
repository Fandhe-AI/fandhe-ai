//! producer/consumer warpgroup 非対称版・基準 arch（`compute_121` 相当）の
//! 実機プローブ（イシュー #484。親イシュー #480 の A-4）。
//!
//! B-3（タイル拡大時のレジスタ予算設計）が引き継ぐのは本ファイルの
//! producer/consumer 非対称パターンであり、`setmaxnreg.dec` 単体（片道）
//! 版より情報価値が高い。位置づけ・「使えないという結論も spike の正当な
//! 成果」方針は `setmaxnreg_probe_dec_base_real_device.rs` と共通のため
//! 二重管理しない。基準 arch 版・arch-accelerated 版（`..._incdec_accel_
//! ...`）を別ファイル（＝別プロセス）へ分離する理由は
//! `tests/setmaxnreg_common/mod.rs` 冒頭コメント参照。
//!
//! **実行時は必ず外部タイムアウトを付与すること**:
//! `timeout 120 cargo test -p backend-cuda --release --test
//! setmaxnreg_probe_incdec_base_real_device -- --ignored --nocapture`
//! （`num_regs` を実行ゲートに使わず外部タイムアウトへ委ねる方針の根拠は
//! `tests/setmaxnreg_common/mod.rs` 冒頭コメント「`num_regs` は診断専用」
//! 節）。

#[path = "setmaxnreg_common/mod.rs"]
mod setmaxnreg_common;

use backend_cuda::CudaDevice;
use setmaxnreg_common::{
    CONTROL_INCDEC, PROBE_SETMAXNREG_INCDEC, PRODUCER_CONSUMER_BLOCK_DIM,
    report_control_baseline_regs, report_environment, try_compile, try_load_and_run,
};

/// producer warpgroup が `setmaxnreg.dec`、consumer warpgroup が
/// `setmaxnreg.inc` を発行する非対称パターンの実機プローブ（基準 arch）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10、compute capability 12.1 相当）必須。\
            timeout 付き実行必須（本ファイル冒頭コメント）。実測記録は \
            docs/cuda-tensor-core-design.md「setmaxnreg プローブ結果（#484）」節"]
fn setmaxnreg_incdec_base_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    report_environment(&device);

    let arch = device.arch();

    let control_ok = report_control_baseline_regs(&device, "control_incdec", CONTROL_INCDEC, arch);

    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_incdec",
        PROBE_SETMAXNREG_INCDEC,
        control_ok,
        arch,
    ) {
        try_load_and_run(
            &device,
            ptx,
            "probe_setmaxnreg_incdec",
            arch,
            PRODUCER_CONSUMER_BLOCK_DIM,
            |x| x * 2.0,
        );
    }
}
