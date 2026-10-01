//! 段の実行器（S1 nvrtc_ptx → S2 nvrtc_cubin → S3 module_load → S4 launch →
//! S5 sync → S6 verify）と `SM121_PROBE_JSON` の出力。
//!
//! # 設計（RULE.txt R-STAGE・R-CTL）
//!
//! - **S1** は仮想アーキ（`compute_*`）で NVRTC を通す。PTX テキストを出すだけで
//!   inline PTX の中身は検証されない。**S2** は実アーキ（`sm_*`）を
//!   `--gpu-architecture` に渡し、NVRTC 内部で ptxas をオフラインで走らせる
//!   （Step 0 実測: NVRTC 13.0.88 は不正 opcode を ptxas のログ付きで拒否し、
//!   正しいカーネルでは PTX も返す）。したがって cubin の取得（`nvrtcGetCUBIN`
//!   の生 FFI）は不要で、本モジュールに `unsafe` の NVRTC 呼び出しはない。
//! - 呼び出す NVRTC は cudarc の `compile_ptx_with_opts` を直接使う（crate の
//!   `compile_ptx` は失敗時に include パス違いで再試行し、最後の試行のログしか
//!   返さず、`arch` を `Box::leak` する）。`arch` は `types.rs` の `&'static`
//!   ホワイトリストのみ（A03）。
//! - **S3** は S1 の PTX（仮想アーキ）をドライバ JIT でロードする。S2 の成果物は
//!   ロードしない。S2 は S1 と独立の枝（S2 の拒否は S3 を止めない。両者の
//!   食い違いは「判定不能」として aggregate.py が扱う）。
//! - 前段が失敗したら後段は `not_run`、設計上不実施の段（`Policy::stage_is_na`）は
//!   `not_applicable_by_design` を明示する。どのセルも省略しない。
//! - 1 (probe, target) = 1 プロセス。毎回最初に対照 `ctl.copy` を同じ target で
//!   通す（失敗なら `ctl` セルを `error` にし、aggregate.py が判定不能にする）。
//! - 出力は段ごとに即座に `println!`（行バッファ）し、後続段でハングしても
//!   既出の記録が残る。

use std::time::Duration;

use cudarc::driver::sys::CUfunction_attribute_enum as FuncAttr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileError, CompileOptions, Ptx, compile_ptx_with_opts};
use fandhe_ai_backend_cuda::{CudaDevice, nvrtc_version};

use super::jsonl::Record;
use super::registry::{Check, Outcome, ProbeSpec, attr_table, evaluate, probe_by_id};
use super::types::{Kind, Policy, Stage, Status, Target};

/// out バッファの初期値（書かれなかった語の検出用。期待値に現れないことは
/// registry テストが検査する）。
pub const SENTINEL: u32 = 0xFEED_FACE;

/// 1 段の結果。
#[derive(Debug, Clone)]
pub struct Cell {
    pub stage: Stage,
    pub status: Status,
    pub code: String,
    pub detail: String,
}

impl Cell {
    pub fn ok(stage: Stage, detail: impl Into<String>) -> Self {
        Self::new(stage, Status::Ok, "-", detail)
    }

    pub fn new(stage: Stage, status: Status, code: &str, detail: impl Into<String>) -> Self {
        Self {
            stage,
            status,
            code: code.to_string(),
            detail: detail.into(),
        }
    }

    pub fn is_ok(&self) -> bool {
        self.status == Status::Ok
    }
}

/// セルを `SM121_PROBE_JSON` 行として出力する。`target` は実行 target 名
/// （`compute_121` 等）または `home`・`hopper`、`arch` は実際に使ったアーキ。
pub fn emit_cell(phase: &str, probe: &str, target: &str, arch: &str, cell: &Cell) {
    Record::new()
        .int("v", 1)
        .str("kind", "cell")
        .str("phase", phase)
        .str("probe", probe)
        .str("target", target)
        .str("arch", arch)
        .str("stage", cell.stage.name())
        .str("status", cell.status.as_str())
        .str("code", &cell.code)
        .str("detail", &cell.detail)
        .emit();
}

/// 環境レコード（G0 の入力）。`device`／`cc` は GPU がなければ `none`。
pub fn emit_env(phase: &str, probe: &str, target: &str, device: Option<&CudaDevice>) {
    let nvrtc = match nvrtc_version() {
        Ok((major, minor)) => format!("{major}.{minor}"),
        Err(_) => "unavailable".to_string(),
    };
    let (name, cc) = match device {
        Some(d) => {
            let (major, minor) = d.compute_capability();
            (d.name().to_string(), format!("{major}.{minor}"))
        }
        None => ("none".to_string(), "none".to_string()),
    };
    Record::new()
        .int("v", 1)
        .str("kind", "env")
        .str("phase", phase)
        .str("probe", probe)
        .str("target", target)
        .str("device", &name)
        .str("cc", &cc)
        .str("nvrtc", &nvrtc)
        .emit();
}

/// NVRTC を `arch`（`&'static`）で実行し、成功なら PTX を返す。失敗は段のセルへ
/// 分類する（ptxas／フロントエンドの拒否 = `rejected`・ログは全文。それ以外の
/// 失敗 = `error`・NVRTC 不在 = `unavailable`）。
pub fn nvrtc_compile(stage: Stage, src: &str, arch: &'static str) -> Result<(Ptx, Cell), Cell> {
    if nvrtc_version().is_err() {
        return Err(Cell::new(
            stage,
            Status::Unavailable,
            "NVRTC_UNAVAILABLE",
            "libnvrtc could not be loaded",
        ));
    }
    let opts = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    match compile_ptx_with_opts(src, opts) {
        Ok(ptx) => {
            let text = ptx.to_src();
            let header: Vec<&str> = text
                .lines()
                .filter(|l| l.starts_with(".version") || l.starts_with(".target"))
                .collect();
            let detail = format!("arch={arch} ptx_bytes={} {}", text.len(), header.join(" "));
            let cell = Cell::ok(stage, detail);
            Ok((ptx, cell))
        }
        Err(CompileError::CompileError {
            nvrtc,
            options,
            log,
        }) => Err(Cell::new(
            stage,
            Status::Rejected,
            "NVRTC_COMPILE_ERROR",
            format!(
                "arch={arch} nvrtc={nvrtc:?} options={options:?} log={}",
                log.to_string_lossy()
            ),
        )),
        Err(other) => Err(Cell::new(
            stage,
            Status::Error,
            "NVRTC_OTHER_ERROR",
            format!("arch={arch} {other:?}"),
        )),
    }
}

/// 固定アーキ（`tc5.cross`）または target に応じた S1／S2 の使用アーキ。
pub fn stage_archs(probe: &ProbeSpec, target: &Target) -> (&'static str, &'static str) {
    probe.fixed_arch.unwrap_or((target.virt, target.real))
}

/// 設計上不実施の段なら `not_applicable_by_design`、そうでなければ上流失敗の
/// `not_run` を返す。
fn skipped(policy: Policy, stage: Stage, upstream: &str) -> Cell {
    if policy.stage_is_na(stage) {
        Cell::new(
            stage,
            Status::NotApplicableByDesign,
            "-",
            format!("policy={}", policy.as_str()),
        )
    } else {
        Cell::new(stage, Status::NotRun, "UPSTREAM_FAILED", upstream)
    }
}

fn na_or(policy: Policy, stage: Stage) -> Option<Cell> {
    policy
        .stage_is_na(stage)
        .then(|| skipped(policy, stage, ""))
}

fn driver_code(err: &cudarc::driver::DriverError) -> String {
    format!("{:?}", err.0)
}

fn value_or_err(r: Result<i32, cudarc::driver::DriverError>) -> String {
    match r {
        Ok(v) => v.to_string(),
        Err(e) => format!("err:{}", driver_code(&e)),
    }
}

/// 1 プローブ × 1 target の S1〜S6 を実行し、段ごとに `sink` へ渡す（常に S1〜S6 の
/// 6 セルをこの順で出す）。`device` が `None` なら S3 以降は `unavailable`／`not_run`。
pub fn run_pipeline(
    device: Option<&CudaDevice>,
    probe: &ProbeSpec,
    target: &Target,
    sink: &mut dyn FnMut(Cell),
) {
    let policy = probe.policy;
    if probe.kind == Kind::Attr {
        run_attr(device, probe, sink);
        return;
    }
    let (virt, real) = stage_archs(probe, target);

    // S1: 仮想アーキの NVRTC。
    let ptx = match nvrtc_compile(Stage::S1NvrtcPtx, probe.src, virt) {
        Ok((ptx, cell)) => {
            sink(cell);
            Some(ptx)
        }
        Err(cell) => {
            sink(cell);
            None
        }
    };

    // S2: 実アーキの NVRTC（ptxas をオフラインで通す）。S3 を止めない独立の枝。
    match nvrtc_compile(Stage::S2NvrtcCubin, probe.src, real) {
        Ok((_ptx, cell)) => sink(cell),
        Err(cell) => sink(cell),
    }

    // S3 以降（デバイス経路）。
    let Some(ptx) = ptx else {
        for stage in [
            Stage::S3ModuleLoad,
            Stage::S4Launch,
            Stage::S5Sync,
            Stage::S6Verify,
        ] {
            sink(skipped(policy, stage, "S1 did not produce PTX"));
        }
        return;
    };
    let Some(device) = device else {
        sink(Cell::new(
            Stage::S3ModuleLoad,
            Status::Unavailable,
            "NO_DEVICE",
            "CudaDevice::new(0) failed",
        ));
        for stage in [Stage::S4Launch, Stage::S5Sync, Stage::S6Verify] {
            sink(skipped(policy, stage, "no device"));
        }
        return;
    };
    device_stages(device, probe, ptx, sink);
}

fn device_stages(device: &CudaDevice, probe: &ProbeSpec, ptx: Ptx, sink: &mut dyn FnMut(Cell)) {
    let policy = probe.policy;

    // S3: ドライバ JIT。
    let module = match device.context().load_module(ptx) {
        Ok(m) => m,
        Err(e) => {
            sink(Cell::new(
                Stage::S3ModuleLoad,
                Status::Error,
                &driver_code(&e),
                format!("load_module: {e:?}"),
            ));
            for stage in [Stage::S4Launch, Stage::S5Sync, Stage::S6Verify] {
                sink(skipped(policy, stage, "S3 failed"));
            }
            return;
        }
    };
    let func = match module.load_function(probe.symbol) {
        Ok(f) => f,
        Err(e) => {
            sink(Cell::new(
                Stage::S3ModuleLoad,
                Status::Error,
                &driver_code(&e),
                format!("load_function({}): {e:?}", probe.symbol),
            ));
            for stage in [Stage::S4Launch, Stage::S5Sync, Stage::S6Verify] {
                sink(skipped(policy, stage, "S3 failed"));
            }
            return;
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (probe.grid, 1, 1),
        block_dim: (probe.block, 1, 1),
        shared_mem_bytes: 0,
    };
    // 参考値（実行可否のゲートには使わない。各値の取得失敗も記録する）。
    let mut detail = format!(
        "binary_version={} ptx_version={} num_regs={}",
        value_or_err(func.binary_version()),
        value_or_err(func.ptx_version()),
        value_or_err(func.num_regs()),
    );
    if probe.cluster > 0 {
        if probe.cluster > 8 {
            let r = func.set_attribute(
                FuncAttr::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            );
            detail.push_str(&format!(
                " non_portable_cluster_size_allowed={}",
                match r {
                    Ok(()) => "set".to_string(),
                    Err(e) => format!("err:{}", driver_code(&e)),
                }
            ));
        }
        let active = func.occupancy_max_active_clusters(cfg, device.stream());
        let potential = func.occupancy_max_potential_cluster_size(cfg, device.stream());
        detail.push_str(&format!(
            " occupancy_max_active_clusters={} occupancy_max_potential_cluster_size={}",
            match active {
                Ok(v) => v.to_string(),
                Err(e) => format!("err:{}", driver_code(&e)),
            },
            match potential {
                Ok(v) => v.to_string(),
                Err(e) => format!("err:{}", driver_code(&e)),
            }
        ));
    }
    sink(Cell::ok(Stage::S3ModuleLoad, detail));

    if policy == Policy::AcceptOnly {
        for stage in [Stage::S4Launch, Stage::S5Sync, Stage::S6Verify] {
            sink(skipped(policy, stage, ""));
        }
        return;
    }

    // S4: 起動。
    let input = (probe.make_input)();
    let stream = device.stream();
    let in_dev = match stream.clone_htod(&input) {
        Ok(b) => b,
        Err(e) => {
            sink(Cell::new(
                Stage::S4Launch,
                Status::Error,
                &driver_code(&e),
                format!("htod(in): {e:?}"),
            ));
            for stage in [Stage::S5Sync, Stage::S6Verify] {
                sink(skipped(policy, stage, "S4 failed"));
            }
            return;
        }
    };
    let mut out_dev = match stream.clone_htod(&vec![SENTINEL; probe.out_words]) {
        Ok(b) => b,
        Err(e) => {
            sink(Cell::new(
                Stage::S4Launch,
                Status::Error,
                &driver_code(&e),
                format!("htod(out): {e:?}"),
            ));
            for stage in [Stage::S5Sync, Stage::S6Verify] {
                sink(skipped(policy, stage, "S4 failed"));
            }
            return;
        }
    };
    let n_in = input.len() as i32;
    let n_out = probe.out_words as i32;
    // SAFETY: 引数は (in: n_in 語の device バッファ, n_in, out: n_out 語の device
    // バッファ, n_out)。カーネル側は全ロード・ストアを `LD`／`ST` マクロで
    // n_in／n_out に対して境界チェックしている（`kernels_mma.rs` 冒頭・REQ-8）。
    // 起動形状は registry の固定値（block・grid・cluster 次元）で、命令が
    // 拒否されて起こりうる実行時エラー（ILLEGAL_INSTRUCTION 等）は
    // `Result::Err` として捕捉し panic させない（ハングは外部 timeout に委ねる）。
    let launched = unsafe {
        stream
            .launch_builder(&func)
            .arg(&in_dev)
            .arg(&n_in)
            .arg(&mut out_dev)
            .arg(&n_out)
            .launch(cfg)
    };
    match launched {
        Ok(_) => sink(Cell::ok(
            Stage::S4Launch,
            format!("grid={} block={}", probe.grid, probe.block),
        )),
        Err(e) => {
            sink(Cell::new(
                Stage::S4Launch,
                Status::Error,
                &driver_code(&e),
                format!("launch: {e:?}"),
            ));
            for stage in [Stage::S5Sync, Stage::S6Verify] {
                sink(skipped(policy, stage, "S4 failed"));
            }
            return;
        }
    }

    // S5: 同期（実行時エラー・不正命令はここで現れる）。
    if let Err(e) = stream.synchronize() {
        sink(Cell::new(
            Stage::S5Sync,
            Status::Error,
            &driver_code(&e),
            format!("synchronize: {e:?}"),
        ));
        sink(skipped(policy, Stage::S6Verify, "S5 failed"));
        return;
    }
    let out = match stream.clone_dtoh(&out_dev) {
        Ok(v) => v,
        Err(e) => {
            sink(Cell::new(
                Stage::S5Sync,
                Status::Error,
                &driver_code(&e),
                format!("dtoh: {e:?}"),
            ));
            sink(skipped(policy, Stage::S6Verify, "S5 failed"));
            return;
        }
    };

    if policy == Policy::RecordOnly {
        // 値は S5 の detail へ記録するのみ（S6 は設計上不実施）。
        match evaluate(probe, &input, &out) {
            Outcome::Record(d) => sink(Cell::ok(Stage::S5Sync, format!("record: {d}"))),
            Outcome::Match(d) => sink(Cell::ok(Stage::S5Sync, format!("record: {d}"))),
            Outcome::Mismatch { detail, .. } => sink(Cell::new(
                Stage::S5Sync,
                Status::Error,
                "RECORD_INVALID",
                detail,
            )),
        }
        sink(skipped(policy, Stage::S6Verify, ""));
        return;
    }
    sink(Cell::ok(Stage::S5Sync, format!("out_words={}", out.len())));

    // S6: 検証（Verify のみ）。
    let cell = match evaluate(probe, &input, &out) {
        Outcome::Match(d) => Cell::ok(Stage::S6Verify, d),
        Outcome::Mismatch { detail, .. } => {
            Cell::new(Stage::S6Verify, Status::Mismatch, "MISMATCH", detail)
        }
        Outcome::Record(d) => Cell::new(Stage::S6Verify, Status::Error, "CHECK_CONTRACT", d),
    };
    sink(cell);
}

/// デバイス属性プローブ（カーネルなし。S6 のみ実施）。取得に失敗した属性は
/// `name=err:CODE` と記録し、1 つでもあれば S6 は `error`。
fn run_attr(device: Option<&CudaDevice>, probe: &ProbeSpec, sink: &mut dyn FnMut(Cell)) {
    let policy = probe.policy;
    for stage in [
        Stage::S1NvrtcPtx,
        Stage::S2NvrtcCubin,
        Stage::S3ModuleLoad,
        Stage::S4Launch,
        Stage::S5Sync,
    ] {
        if let Some(c) = na_or(policy, stage) {
            sink(c);
        }
    }
    let Some(device) = device else {
        sink(Cell::new(
            Stage::S6Verify,
            Status::Unavailable,
            "NO_DEVICE",
            "CudaDevice::new(0) failed",
        ));
        return;
    };
    let mut parts = Vec::new();
    let mut failed = false;
    for (name, attr) in attr_table(probe.id) {
        match device.context().attribute(*attr) {
            Ok(v) => parts.push(format!("{name}={v}")),
            Err(e) => {
                failed = true;
                parts.push(format!("{name}=err:{}", driver_code(&e)));
            }
        }
    }
    let detail = parts.join(" ");
    if failed {
        sink(Cell::new(
            Stage::S6Verify,
            Status::Error,
            "ATTR_QUERY_FAILED",
            detail,
        ));
    } else {
        sink(Cell::ok(Stage::S6Verify, detail));
    }
}

/// 対照カーネル `ctl.copy` を同じ target で S1〜S6 まで通し、`ctl` セルへ要約する
/// （R-CTL）。S1／S2 の拒否は toolchain が target を受け付けない
/// （`TARGET_UNSUPPORTED`）、それ以外の失敗は `CTL_FAILED`。
pub fn run_control(device: Option<&CudaDevice>, target: &Target) -> Cell {
    let Some(ctl) = probe_by_id("ctl.copy") else {
        return Cell::new(
            Stage::Ctl,
            Status::Error,
            "CTL_FAILED",
            "ctl.copy is not registered",
        );
    };
    let mut cells = Vec::new();
    run_pipeline(device, &ctl, target, &mut |c| cells.push(c));
    match cells.iter().find(|c| !c.is_ok()) {
        None => Cell::ok(Stage::Ctl, "ctl.copy S1..S6 all ok"),
        Some(bad) => {
            let code = if matches!(bad.stage, Stage::S1NvrtcPtx | Stage::S2NvrtcCubin)
                && bad.status == Status::Rejected
            {
                "TARGET_UNSUPPORTED"
            } else {
                "CTL_FAILED"
            };
            Cell::new(
                Stage::Ctl,
                Status::Error,
                code,
                format!(
                    "stage={} status={} code={} detail={}",
                    bad.stage.name(),
                    bad.status.as_str(),
                    bad.code,
                    bad.detail
                ),
            )
        }
    }
}

/// 外部 `timeout` が実行時間を管理するため、プロセス内に待ち時間の上限は持たない。
/// 診断用の経過時間表示にのみ使う。
pub fn elapsed_ms(start: std::time::Instant) -> u64 {
    let d: Duration = start.elapsed();
    d.as_millis().min(u128::from(u64::MAX)) as u64
}

/// `Check` が `Skip` でない（S6 で判定する）プローブか。
pub fn has_check(probe: &ProbeSpec) -> bool {
    !matches!(probe.check, Check::Skip)
}
