//! `setmaxnreg.dec` 単独発行版・基準 arch（`compute_121` 相当）の実機
//! プローブ（イシュー #484。親イシュー #480 の A-4）。
//!
//! ## 位置づけ
//!
//! PTX ISA の `setmaxnreg.inc/dec.sync.aligned.u32` は、producer/consumer
//! warp 間でレジスタ予算を非対称配分する warp specialization パターン
//! （warpgroup＝連続 4 warp／128 スレッド単位の命令）向けに導入された。
//! 歴代 `sm_90a` 系の arch-accelerated feature として案内されることが多く、
//! コンシューマ系 Blackwell（SM12x）の `compute_121`／DGX Spark GB10 +
//! NVRTC（CUDA 13.0 系）で NVRTC が受理し実行できるかは本リポジトリで
//! 未検証だった（`docs/cuda-tensor-core-design.md` §2 は tcgen05/wgmma の
//! 非対応のみ確定済みで setmaxnreg は未記載）。この結果は後続 B-3
//! （タイル拡大時のレジスタ予算設計）の設計自由度の上限に直接影響する。
//!
//! 「使えない」という結論も spike の正当な成果である。そのため本ファイルの
//! テストは、NVRTC コンパイル・カーネル実行の失敗を `panic` させず
//! `SETMAXNREG_PROBE_RESULT: ...` 形式で標準出力へ構造的に記録したうえで
//! pass する設計とする（#64 の `tensor_core_real_device.rs` と同型の
//! 「実測記録テストは pass、CUDA デバイス自体の不在のみ fail-loud」方針。
//! 対して CUDA デバイス自体が使えない環境（`CudaDevice::new` 失敗）は
//! 既存規約どおり `.expect` で fail-loud にし、実機以外での silent green
//! を許さない）。
//!
//! ## ファイル分割・実行契約（PR #636 レビュー指摘再設計。共通ヘルパーは
//! [`setmaxnreg_common`] 参照）
//!
//! 基準 arch 版・arch-accelerated 版（`..._dec_accel_real_device.rs`）・
//! producer/consumer 非対称版（`..._incdec_base_...`／`..._incdec_accel_
//! ...`）を別ファイル（＝別 cargo test バイナリ＝別プロセス）へ分離した
//! 理由と、`num_regs` を実行ゲートに使わず外部タイムアウトへ委ねる理由は
//! `tests/setmaxnreg_common/mod.rs` 冒頭コメントを正とする（本ファイルでは
//! 二重管理しない）。
//!
//! **実行時は必ず外部タイムアウトを付与すること**:
//! `timeout 120 cargo test -p backend-cuda --release --test
//! setmaxnreg_probe_dec_base_real_device -- --ignored --nocapture`
//! （`docs/real-hardware-verification-env.md` の手順・
//! `docs/cuda-tensor-core-design.md` §13.1「実行契約」節）。
//!
//! ## A03（インジェクション）対応
//!
//! カーネルソースは [`setmaxnreg_common`] 内の `&'static str` コンパイル
//! 時定数のみを使い、外部入力・環境変数をソース文字列へ連結しない
//! （`nvrtc.rs` の既存契約と同じ方針。`.claude/rules/security.md`）。

#[path = "setmaxnreg_common/mod.rs"]
mod setmaxnreg_common;

use backend_cuda::CudaDevice;
use setmaxnreg_common::{
    CONTROL_DEC, PROBE_SETMAXNREG_DEC, WARPGROUP_BLOCK_DIM, report_control_baseline_regs,
    report_environment, try_compile, try_load_and_run,
};

/// `setmaxnreg.dec` のみを発行するカーネルの実機プローブ（基準 arch）。
///
/// `try_compile`/`try_load_and_run` の前に `report_control_baseline_regs`
/// で対照カーネル（[`CONTROL_DEC`]）のベースライン register/thread 数を
/// **診断ログとしてのみ**実測する（実行可否には使わない。
/// `setmaxnreg_common` 冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10、compute capability 12.1 相当）必須。\
            timeout 付き実行必須（本ファイル冒頭コメント）。実測記録は \
            docs/cuda-tensor-core-design.md「setmaxnreg プローブ結果（#484）」節"]
fn setmaxnreg_dec_base_probe() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    report_environment(&device);

    let arch = device.arch();

    let control_ok = report_control_baseline_regs(&device, "control_dec", CONTROL_DEC, arch);

    if let Some(ptx) = try_compile(
        "probe_setmaxnreg_dec",
        PROBE_SETMAXNREG_DEC,
        control_ok,
        arch,
    ) {
        try_load_and_run(
            &device,
            ptx,
            "probe_setmaxnreg_dec",
            arch,
            WARPGROUP_BLOCK_DIM,
            |x| x + 1.0,
        );
    }
}
