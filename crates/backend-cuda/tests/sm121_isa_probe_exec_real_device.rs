//! sm_121 ISA プローブ（イシュー #2122）の実機実行器。1 (probe, target) = 1 プロセス。
//!
//! 環境変数 `SM121_PROBE_ID`・`SM121_PROBE_TARGET` をレジストリ／target の
//! ホワイトリストで検索して 1 件だけ実行する。未指定・未知の値は panic
//! （fail-loud。「黙って成功」を許さない）。値はソースへ連結しない（A03）。
//!
//! 先に同じ target で対照 `ctl.copy` を通し、続けてプローブ自身の S1〜S6 を
//! 出力する（`SM121_PROBE_JSON`・`phase=exec`）。不一致（`mismatch`）を記録した
//! 後は非 0 で終了する。命令の拒否・実行時エラーは「測定結果」であり終了コードに
//! 反映しない。ハングは外部 `timeout`（`orchestrate.sh`）が打ち切る。
//! sticky なエラーが後続へ波及しないよう 1 プロセスに 1 件だけ実行する
//! （`setmaxnreg_common` と同じ分離方針）。
//!
//! 実行例: `SM121_PROBE_ID=ctl.copy SM121_PROBE_TARGET=compute_121 timeout 120
//! cargo test -p fandhe-ai-backend-cuda --all-features --test
//! sm121_isa_probe_exec_real_device -- --ignored --nocapture`。
//! `#[ignore]` なのは実機（DGX Spark GB10。開発機スモークは `compute_86`）と
//! NVRTC が必要なため。

#[path = "sm121_isa_probe_common/mod.rs"]
mod common;

use common::jsonl::Record;
use common::registry::probe_by_id;
use common::runner::{Cell, emit_cell, emit_env, run_control, run_pipeline, stage_archs};
use common::types::{Stage, Status, target_by_name};
use fandhe_ai_backend_cuda::CudaDevice;

fn required_env(name: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => v,
        _ => panic!("環境変数 {name} が未設定（1 プロセス 1 件を明示指定する契約）"),
    }
}

#[test]
#[ignore = "実機（GB10。開発機スモークは compute_86）と NVRTC が必要"]
fn sm121_isa_probe_exec_selected() {
    let id = required_env("SM121_PROBE_ID");
    let target_name = required_env("SM121_PROBE_TARGET");
    let probe = probe_by_id(&id).unwrap_or_else(|| panic!("未知の SM121_PROBE_ID: {id:?}"));
    let target = target_by_name(&target_name)
        .unwrap_or_else(|| panic!("未知の SM121_PROBE_TARGET: {target_name:?}"));

    let device = match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            // 記録して続行する（S3 は unavailable・G0 が cc 不明で不成立にする）。
            println!("SM121_PROBE_NOTE device_init_failed: {e}");
            None
        }
    };
    emit_env("exec", probe.id, target.name, device.as_ref());

    // 対照（ctl.copy 自身の場合は設計上不実施）。
    let ctl = if probe.id == "ctl.copy" {
        Cell::new(
            Stage::Ctl,
            Status::NotApplicableByDesign,
            "-",
            "this probe is the control",
        )
    } else {
        run_control(device.as_ref(), target)
    };
    emit_cell("exec", probe.id, target.name, target.virt, &ctl);

    let (virt, real) = stage_archs(&probe, target);
    let mut cells = 1u64;
    let mut mismatches = 0u64;
    run_pipeline(device.as_ref(), &probe, target, &mut |cell| {
        let arch = if cell.stage == Stage::S2NvrtcCubin {
            real
        } else {
            virt
        };
        emit_cell("exec", probe.id, target.name, arch, &cell);
        cells += 1;
        if cell.status == Status::Mismatch {
            mismatches += 1;
        }
    });
    Record::new()
        .int("v", 1)
        .str("kind", "done")
        .str("phase", "exec")
        .str("probe", probe.id)
        .str("target", target.name)
        .int("cells", cells)
        .int("mismatch", mismatches)
        .emit();
    assert_eq!(
        mismatches, 0,
        "S6 で不一致を記録した（{id} on {target_name}）。判定は aggregate.py が行う"
    );
}
