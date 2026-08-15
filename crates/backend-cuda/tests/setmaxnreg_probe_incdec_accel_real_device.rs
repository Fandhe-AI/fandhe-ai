//! producer/consumer warpgroup 非対称版・arch-accelerated 版
//! （`compute_121a`）の実機プローブ（イシュー #484。親イシュー #480 の
//! A-4）。
//!
//! setmaxnreg が最も受理されやすいのは arch-accelerated 版であり、B-3
//! （タイル拡大時のレジスタ予算設計）が引き継ぐ本命シナリオである。
//! 位置づけ・分割理由・実行契約は `setmaxnreg_probe_incdec_base_real_
//! device.rs`／`tests/setmaxnreg_common/mod.rs` と共通のため二重管理しない。
//!
//! **実行時は必ず外部タイムアウトを付与すること**:
//! `timeout 120 cargo test -p backend-cuda --release --test
//! setmaxnreg_probe_incdec_accel_real_device -- --ignored --nocapture`

#[path = "setmaxnreg_common/mod.rs"]
mod setmaxnreg_common;

use backend_cuda::CudaDevice;
use setmaxnreg_common::{
    CONTROL_INCDEC, PROBE_SETMAXNREG_INCDEC, PRODUCER_CONSUMER_BLOCK_DIM,
    report_control_baseline_regs, report_environment, try_compile, try_load_and_run,
};

/// producer warpgroup が `setmaxnreg.dec`、consumer warpgroup が
/// `setmaxnreg.inc` を発行する非対称パターンの実機プローブ
/// （arch-accelerated 版）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10、compute capability 12.1 相当）必須。\
            timeout 付き実行必須（本ファイル冒頭コメント）。実測記録は \
            docs/cuda-tensor-core-design.md「setmaxnreg プローブ結果（#484）」節"]
fn setmaxnreg_incdec_accel_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    report_environment(&device);

    let arch_accelerated = format!("{}a", device.arch());

    let control_ok =
        report_control_baseline_regs(&device, "control_incdec", CONTROL_INCDEC, &arch_accelerated);

    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_incdec_arch_accelerated",
        PROBE_SETMAXNREG_INCDEC,
        control_ok,
        &arch_accelerated,
    ) {
        try_load_and_run(
            &device,
            ptx,
            "probe_setmaxnreg_incdec",
            &arch_accelerated,
            PRODUCER_CONSUMER_BLOCK_DIM,
            |x| x * 2.0,
        );
    }
}
