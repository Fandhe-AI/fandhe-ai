//! `setmaxnreg.dec` 単独発行版・arch-accelerated 版（`compute_121a`）の
//! 実機プローブ（イシュー #484。親イシュー #480 の A-4）。
//!
//! 位置づけ・「使えないという結論も spike の正当な成果」方針は
//! `setmaxnreg_probe_dec_base_real_device.rs` と共通のため二重管理しない
//! （本ファイルはそちらの arch-accelerated 対）。基準 arch 版と同一
//! プロセスで連続実行すると、基準 arch 側のデバイス例外が CUDA コンテキ
//! ストを汚染し本ファイルの結果に伝播しうるため、別ファイル（＝別
//! プロセス）へ分離している（`tests/setmaxnreg_common/mod.rs` 冒頭コメント
//! 「ファイル分割の理由」節参照）。
//!
//! **実行時は必ず外部タイムアウトを付与すること**:
//! `timeout 120 cargo test -p backend-cuda --release --test
//! setmaxnreg_probe_dec_accel_real_device -- --ignored --nocapture`

#[path = "setmaxnreg_common/mod.rs"]
mod setmaxnreg_common;

use backend_cuda::CudaDevice;
use setmaxnreg_common::{
    CONTROL_DEC, PROBE_SETMAXNREG_DEC, WARPGROUP_BLOCK_DIM, report_control_baseline_regs,
    report_environment, try_compile, try_load_and_run,
};

/// `setmaxnreg.dec` のみを発行するカーネルの実機プローブ（arch-accelerated
/// 版・`<arch>a`）。setmaxnreg は歴代 `sm_90a` 系の arch-accelerated feature
/// として案内されることが多く、基準 arch より受理されやすい本命シナリオ
/// である。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10、compute capability 12.1 相当）必須。\
            timeout 付き実行必須（本ファイル冒頭コメント）。実測記録は \
            docs/cuda-tensor-core-design.md「setmaxnreg プローブ結果（#484）」節"]
fn setmaxnreg_dec_accel_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    report_environment(&device);

    let arch_accelerated = format!("{}a", device.arch());

    let control_ok =
        report_control_baseline_regs(&device, "control_dec", CONTROL_DEC, &arch_accelerated);

    // `try_compile` の `label` は `try_load_and_run` と共有する診断ログ用
    // ラベルであり（`try_load_and_run` 側は同時に `module.load_function`
    // へ渡す実 CUDA シンボル名でもある）、base 版（`setmaxnreg_probe_dec_
    // base_real_device.rs`）と同一の `"probe_setmaxnreg_dec"` に揃える。
    // arch-accelerated 版であることの区別は `arch` 引数（`<arch>a`）側で
    // 既に付与されているため、`label` に `_arch_accelerated` を別途付与
    // すると `SETMAXNREG_PROBE_RESULT` の compile 段階と load/execute 段階
    // とで `kernel=` の値が食い違い、同一実行の結果を grep で突合する
    // 運用が壊れる（PR #636 レビュー指摘対応）。
    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_dec",
        PROBE_SETMAXNREG_DEC,
        control_ok,
        &arch_accelerated,
    ) {
        try_load_and_run(
            &device,
            ptx,
            "probe_setmaxnreg_dec",
            &arch_accelerated,
            WARPGROUP_BLOCK_DIM,
            |x| x + 1.0,
        );
    }
}
