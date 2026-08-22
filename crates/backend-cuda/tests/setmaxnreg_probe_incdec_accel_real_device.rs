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
//! `timeout 120 cargo test -p fandhe-ai-backend-cuda --release --test
//! setmaxnreg_probe_incdec_accel_real_device -- --ignored --nocapture`

#[path = "setmaxnreg_common/mod.rs"]
mod setmaxnreg_common;

use fandhe_ai_backend_cuda::CudaDevice;
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

    // `try_compile` の `label` は `try_load_and_run` と共有する診断ログ用
    // ラベルであり（`try_load_and_run` 側は同時に `module.load_function`
    // へ渡す実 CUDA シンボル名でもある）、base 版（`setmaxnreg_probe_
    // incdec_base_real_device.rs`）と同一の `"probe_setmaxnreg_incdec"` に
    // 揃える。arch-accelerated 版であることの区別は `arch` 引数
    // （`<arch>a`）側で既に付与されているため、`label` に
    // `_arch_accelerated` を別途付与すると `SETMAXNREG_PROBE_RESULT` の
    // compile 段階と load/execute 段階とで `kernel=` の値が食い違い、
    // 同一実行の結果を grep で突合する運用が壊れる（PR #636 レビュー
    // 指摘対応）。
    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_incdec",
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
