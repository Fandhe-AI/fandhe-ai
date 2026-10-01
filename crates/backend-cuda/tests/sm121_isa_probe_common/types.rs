//! sm_121 ISA プローブ（イシュー #2122）の共通型: 段（stage）・状態（status）・
//! 方針（policy）・target 表。
//!
//! 本モジュールは `sm121_isa_probe_registry`（CI で走る非 ignore テスト）・
//! `sm121_isa_probe_compile`・`sm121_isa_probe_exec_real_device` の 3 バイナリが
//! `#[path]` 経由で共有する。語彙は `docs/perf/logs/sm121-isa-probe-2122/RULE.txt`
//! の「R-STAGE」条項と 1 対 1 に対応し、`aggregate.py` の閉じた集合
//! （STAGES・STATUSES）と文字列が一致しなければならない（registry テストが
//! RULE.txt の `STAGE:`／`STATUS:` 行と突き合わせる）。

/// 段の実行順（RULE.txt R-STAGE）。`Ctl` は同一プロセス内の対照カーネル
/// （`ctl.copy`）の結果の要約であり、`S1`〜`S6` が当該プローブ自身の段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Ctl,
    S1NvrtcPtx,
    S2NvrtcCubin,
    S3ModuleLoad,
    S4Launch,
    S5Sync,
    S6Verify,
}

impl Stage {
    /// ログ・RULE.txt に現れる段名。
    pub const fn name(self) -> &'static str {
        match self {
            Stage::Ctl => "ctl",
            Stage::S1NvrtcPtx => "nvrtc_ptx",
            Stage::S2NvrtcCubin => "nvrtc_cubin",
            Stage::S3ModuleLoad => "module_load",
            Stage::S4Launch => "launch",
            Stage::S5Sync => "sync",
            Stage::S6Verify => "verify",
        }
    }
}

/// 段名の全集合（実行順）。RULE.txt の `STAGE:` 行と一致させる。
pub const ALL_STAGES: [Stage; 7] = [
    Stage::Ctl,
    Stage::S1NvrtcPtx,
    Stage::S2NvrtcCubin,
    Stage::S3ModuleLoad,
    Stage::S4Launch,
    Stage::S5Sync,
    Stage::S6Verify,
];

/// 状態の閉じた集合（RULE.txt R-STAGE）。`Timeout`／`ProcessFailed` は
/// プロセス内からは出せず（外部 `timeout`・異常終了の検出は orchestrate.sh と
/// aggregate.py が行う）、語彙の完全性のためだけにここへ持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Rejected,
    Error,
    Mismatch,
    Timeout,
    ProcessFailed,
    NotRun,
    Unavailable,
    NotApplicableByDesign,
}

impl Status {
    pub const fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Rejected => "rejected",
            Status::Error => "error",
            Status::Mismatch => "mismatch",
            Status::Timeout => "timeout",
            Status::ProcessFailed => "process_failed",
            Status::NotRun => "not_run",
            Status::Unavailable => "unavailable",
            Status::NotApplicableByDesign => "not_applicable_by_design",
        }
    }
}

/// 状態名の全集合。RULE.txt の `STATUS:` 行と一致させる。
pub const ALL_STATUSES: [Status; 9] = [
    Status::Ok,
    Status::Rejected,
    Status::Error,
    Status::Mismatch,
    Status::Timeout,
    Status::ProcessFailed,
    Status::NotRun,
    Status::Unavailable,
    Status::NotApplicableByDesign,
];

/// 実行段（S4〜S6）の扱い（RULE.txt R-STAGE・PROBE 行の `policy=`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// S1〜S6 をすべて実行し、S6 で期待値とビット一致を判定する。
    Verify,
    /// S1〜S3（受理段）のみ。S4〜S6 は設計上不実施（`not_applicable_by_design`）。
    AcceptOnly,
    /// S1〜S5 を実行し、出力値は S5 の detail へ記録するのみ（S6 は不実施）。
    RecordOnly,
    /// カーネルを持たないデバイス属性の記録（S6 のみ実施）。
    Attr,
}

impl Policy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Policy::Verify => "verify",
            Policy::AcceptOnly => "accept_only",
            Policy::RecordOnly => "record_only",
            Policy::Attr => "attr",
        }
    }

    /// 当該方針で、段が設計上不実施（`not_applicable_by_design` 固定）か。
    /// 実行結果に依らず静的に決まる（aggregate.py も同じ表を持つ）。
    pub const fn stage_is_na(self, stage: Stage) -> bool {
        match self {
            Policy::Verify => false,
            Policy::AcceptOnly => {
                matches!(stage, Stage::S4Launch | Stage::S5Sync | Stage::S6Verify)
            }
            Policy::RecordOnly => matches!(stage, Stage::S6Verify),
            Policy::Attr => matches!(
                stage,
                Stage::S1NvrtcPtx
                    | Stage::S2NvrtcCubin
                    | Stage::S3ModuleLoad
                    | Stage::S4Launch
                    | Stage::S5Sync
            ),
        }
    }
}

/// fragment レイアウトの検証状態（RULE.txt R-MMA。PROBE 行の `layout=`）。
/// `Unverified` のプローブの不一致は、ハードウェアの判定にせず
/// 「判定不能（LAYOUT_UNVERIFIED）」とする（事前登録）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// 受理後の値にレイアウト依存がない（要素ごとの演算など）。
    None,
    /// 開発機（sm_86）の実機でホスト参照モデルを検証済み。
    Verified,
    /// 開発機で実行できず、ホスト参照モデルのレイアウトが未検証。
    Unverified,
}

impl Layout {
    pub const fn as_str(self) -> &'static str {
        match self {
            Layout::None => "none",
            Layout::Verified => "verified",
            Layout::Unverified => "unverified",
        }
    }
}

/// 事前登録した想定（結果ではなく仮説）。想定外の受理は「判定不能
/// （UNEXPECTED_ACCEPT）」とし、黙って「成立」にしない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    None,
    /// sm_121 系 target では拒否される想定（tcgen05 など）。
    Reject121,
}

impl Expect {
    pub const fn as_str(self) -> &'static str {
        match self {
            Expect::None => "none",
            Expect::Reject121 => "reject121",
        }
    }
}

/// プローブの種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Kernel,
    Attr,
}

/// 実行 target（`SM121_PROBE_TARGET` のホワイトリスト）。`virt` は S1 と S3 で
/// 使う仮想アーキ、`real` は S2（ptxas をオフラインで通す）で使う実アーキ。
/// どちらも `&'static str`（cudarc の `CompileOptions::arch` が `'static`
/// を要求し、実行時文字列を `Box::leak` しないため。A03）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    pub name: &'static str,
    pub virt: &'static str,
    pub real: &'static str,
    /// 正式な GB10 実測の target か（`false` は開発機スモーク専用。
    /// 開発機スモークの結果は G0 を不成立にし、全セルを判定不能にする）。
    pub official: bool,
}

pub const TARGETS: [Target; 4] = [
    Target {
        name: "compute_121",
        virt: "compute_121",
        real: "sm_121",
        official: true,
    },
    Target {
        name: "compute_121a",
        virt: "compute_121a",
        real: "sm_121a",
        official: true,
    },
    Target {
        name: "compute_121f",
        virt: "compute_121f",
        real: "sm_121f",
        official: true,
    },
    Target {
        name: "compute_86",
        virt: "compute_86",
        real: "sm_86",
        official: false,
    },
];

/// ホワイトリスト検索（未知の名前は `None`。呼び出し側が fail-loud にする）。
pub fn target_by_name(name: &str) -> Option<&'static Target> {
    TARGETS.iter().find(|t| t.name == name)
}

/// S2 の `home`（本来の対応アーキ）・`hopper`（AC5 の Hopper 列）に使える
/// 実アーキの許可集合。レジストリの `home_arch` はここに含まれること。
pub const REAL_ARCH_WHITELIST: [&str; 8] = [
    "sm_80", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120a", "sm_121a",
];

/// AC5 の Hopper 列として全プローブに追加で通す実アーキ。
pub const HOPPER_ARCH: &str = "sm_90a";

/// ホワイトリスト内の `&'static str` を返す（`Box::leak` 回避。未知なら `None`）。
pub fn real_arch_static(arch: &str) -> Option<&'static str> {
    REAL_ARCH_WHITELIST.iter().copied().find(|a| *a == arch)
}

/// 仮想アーキの許可集合（`virt` に使える全値。tc5.cross の `compute_100a` を含む）。
pub const VIRT_ARCH_WHITELIST: [&str; 5] = [
    "compute_121",
    "compute_121a",
    "compute_121f",
    "compute_86",
    "compute_100a",
];

/// カーネルの起動方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launch {
    /// cudarc の safe な `launch_builder`（PR-A の経路）。
    Plain,
    /// `cuModuleLoadData`／`cuLaunchKernelEx`（raw。`CUtensorMap` の値渡しと、実行時の
    /// cluster 次元の指定が必要なプローブ用）。`cluster` は cluster 次元（0 = 属性なし）。
    Raw { cluster: u32 },
}
