//! sm_121 ISA プローブ（イシュー #2122）の S1／S2 全行列（GPU 不要・NVRTC のみ）。
//!
//! 全プローブ × target について、S1（仮想アーキの NVRTC）と S2（実アーキで
//! ptxas をオフラインに通す）を 1 プロセスで回し、`SM121_PROBE_JSON` 行
//! （`phase=compile`）を出力する。加えて各プローブを本来の対応アーキ
//! （`target=home`。R-HOME の陽性対照）と Hopper（`target=hopper`・`sm_90a`。
//! R-HOPPER の列）でも S2 に通す。プローブの文法誤りを GB10 実測の前に
//! 見つける用途でもある（home の拒否は「判定不能（プローブ不良の疑い）」）。
//!
//! 実行: `cargo test -p fandhe-ai-backend-cuda --all-features --test
//! sm121_isa_probe_compile -- --ignored --nocapture`（`libnvrtc` が必要）。
//! 環境変数 `SM121_PROBE_TARGET_SET`（未指定 or `official` = GB10 の 3 target・
//! `dev` = 開発機スモーク用の `compute_86`）。結果は `aggregate.py` が読む。
//! `#[ignore]` なのは NVRTC（CUDA toolkit）が CI のホステッド runner に無いため。

#[path = "sm121_isa_probe_common/mod.rs"]
mod common;

use common::jsonl::Record;
use common::registry::probes;
use common::runner::{Cell, emit_cell, emit_env, nvrtc_compile};
use common::types::{HOPPER_ARCH, Kind, Stage, TARGETS, Target, real_arch_static};
use fandhe_ai_backend_cuda::nvrtc_version;

fn selected_targets() -> Vec<&'static Target> {
    match std::env::var("SM121_PROBE_TARGET_SET").as_deref() {
        Err(_) | Ok("official") => TARGETS.iter().filter(|t| t.official).collect(),
        Ok("dev") => TARGETS.iter().filter(|t| !t.official).collect(),
        Ok(other) => panic!("SM121_PROBE_TARGET_SET が未知の値: {other:?}（official|dev）"),
    }
}

#[test]
#[ignore = "NVRTC（CUDA toolkit の libnvrtc）が必要。GPU は不要"]
fn sm121_isa_probe_compile_matrix() {
    emit_env("compile", "-", "-", None);
    assert!(
        nvrtc_version().is_ok(),
        "libnvrtc を読み込めない（LD_LIBRARY_PATH を確認）。env レコードは出力済み"
    );
    let targets = selected_targets();
    let mut cells = 0u64;
    let mut all = probes();
    all.retain(|p| p.kind == Kind::Kernel);
    for probe in &all {
        for target in &targets {
            let (virt, real) = common::runner::stage_archs(probe, target);
            let s1 = match nvrtc_compile(Stage::S1NvrtcPtx, probe.src, virt) {
                Ok((_, c)) | Err(c) => c,
            };
            emit_cell("compile", probe.id, target.name, virt, &s1);
            let s2 = match nvrtc_compile(Stage::S2NvrtcCubin, probe.src, real) {
                Ok((_, c)) | Err(c) => c,
            };
            emit_cell("compile", probe.id, target.name, real, &s2);
            cells += 2;
        }
        // 陽性対照（R-HOME）と Hopper 列（R-HOPPER）は target 非依存の S2 のみ。
        for (label, arch) in [("home", probe.home_arch), ("hopper", HOPPER_ARCH)] {
            let arch = real_arch_static(arch)
                .unwrap_or_else(|| panic!("{}: アーキ {arch} がホワイトリストに無い", probe.id));
            let cell: Cell = match nvrtc_compile(Stage::S2NvrtcCubin, probe.src, arch) {
                Ok((_, c)) | Err(c) => c,
            };
            emit_cell("compile", probe.id, label, arch, &cell);
            cells += 1;
        }
    }
    Record::new()
        .int("v", 1)
        .str("kind", "done")
        .str("phase", "compile")
        .str("probe", "-")
        .str("target", "-")
        .int("cells", cells)
        .int("mismatch", 0)
        .emit();
}
