//! TMA（AC3。イシュー #2122 PR-B）プローブの spec とレジストリ項目。`registry.rs` の
//! `probes()` が [`tma_probes`] を連結する（RULE.txt の `PROBE:` 行と 1 対 1）。
//!
//! すべてのプローブは global を 64x96 の f32 テンソル（要素 `r * 1000 + c`）とし、
//! `cuTensorMapEncodeTiled` の引数（box・swizzle・OOB fill・座標・expect_tx）を
//! [`TmaSpec`] で引数化する。期待値は持たず、観測値を候補モデルと突き合わせる
//! （`model_tma.rs`）。

use super::kernels_tma as kt;
use super::model_tma::{self as mt, Swz, TmaSpec};
use super::registry::{Check, Outcome, ProbeSpec};
use super::types::{Expect, Kind, Launch, Layout, Policy};

const fn load_spec(
    swizzle: Swz,
    box_rows: u32,
    box_cols: u32,
    oob_nan: bool,
    cx: i32,
    cy: i32,
    expect_tx: u32,
) -> TmaSpec {
    TmaSpec {
        global_rows: 64,
        global_cols: 96,
        box_rows,
        box_cols,
        swizzle,
        oob_nan,
        cx,
        cy,
        expect_tx,
        dump_words: box_rows * box_cols,
        readback_global: false,
        global_sentinel: false,
    }
}

/// 基本: 64x96 の f32、box 8 行 x 16 列、座標 (0,0)、box 全体の expect_tx（512 B）。
pub static SPEC_BASE: TmaSpec = load_spec(Swz::None, 8, 16, false, 0, 0, 512);
/// 座標: 要素座標（内側が先）か転置かを区別できる非対称な位置 (cx=24, cy=40)。
pub static SPEC_COORD: TmaSpec = load_spec(Swz::None, 8, 16, false, 24, 40, 512);
/// OOB（右下端をまたぐ box。列 96〜103・行 64〜67 が範囲外）。fill は NONE。
pub static SPEC_OOB_NONE: TmaSpec = load_spec(Swz::None, 8, 16, false, 88, 60, 512);
/// 同じ box で fill に `NAN_REQUEST_ZERO_FMA`。
pub static SPEC_OOB_NAN: TmaSpec = load_spec(Swz::None, 8, 16, true, 88, 60, 512);
/// 同じ box で expect_tx を範囲内のバイト数（8 列 x 4 行 x 4 B = 128）だけにする。
pub static SPEC_OOB_TX_PARTIAL: TmaSpec = load_spec(Swz::None, 8, 16, false, 88, 60, 128);
/// 負の座標（cx=-8, cy=-4。行 0〜3・列 0〜7 だけが範囲内）。
pub static SPEC_OOB_NEG: TmaSpec = load_spec(Swz::None, 8, 16, false, -8, -4, 512);
/// swizzle: box の内側が swizzle 幅ちょうど（32B=8 列・64B=16 列・128B=32 列）、8 行、座標 (0,0)。
pub static SPEC_SWZ32: TmaSpec = load_spec(Swz::B32, 8, 8, false, 0, 0, 256);
pub static SPEC_SWZ64: TmaSpec = load_spec(Swz::B64, 8, 16, false, 0, 0, 512);
pub static SPEC_SWZ128: TmaSpec = load_spec(Swz::B128, 8, 32, false, 0, 0, 1024);
/// store: global を番兵で初期化し、box（8x16）を (cx=16, cy=24) へ書いて global を読み戻す。
pub static SPEC_STORE: TmaSpec = TmaSpec {
    global_rows: 64,
    global_cols: 96,
    box_rows: 8,
    box_cols: 16,
    swizzle: Swz::None,
    oob_nan: false,
    cx: 16,
    cy: 24,
    expect_tx: 0,
    dump_words: 0,
    readback_global: true,
    global_sentinel: true,
};
/// prefetch: 基本と同じ box・座標。
pub static SPEC_PREFETCH: TmaSpec = load_spec(Swz::None, 8, 16, false, 0, 0, 512);
/// multicast: 基本と同じ box・座標（cluster 2 の両 CTA の smem へ転送）。
pub static SPEC_MULTICAST: TmaSpec = load_spec(Swz::None, 8, 16, false, 0, 0, 512);

/// raw 起動の tensor map 引数 ABI の対照（`ctl.rawmap`）: 後続引数が非自明な値になる spec。
pub static SPEC_RAWMAP: TmaSpec = load_spec(Swz::None, 8, 16, false, 24, -4, 512);

/// 全 TMA spec（registry テストが一括で検証する）。
pub fn all_tma_specs() -> Vec<(&'static str, &'static TmaSpec)> {
    vec![
        ("base", &SPEC_BASE),
        ("coord", &SPEC_COORD),
        ("oob_none", &SPEC_OOB_NONE),
        ("oob_nan", &SPEC_OOB_NAN),
        ("oob_tx_partial", &SPEC_OOB_TX_PARTIAL),
        ("oob_neg", &SPEC_OOB_NEG),
        ("swz32", &SPEC_SWZ32),
        ("swz64", &SPEC_SWZ64),
        ("swz128", &SPEC_SWZ128),
        ("store", &SPEC_STORE),
        ("prefetch", &SPEC_PREFETCH),
        ("multicast", &SPEC_MULTICAST),
        ("rawmap", &SPEC_RAWMAP),
    ]
}

fn in_base() -> Vec<u32> {
    mt::global_input(&SPEC_BASE)
}
fn in_coord() -> Vec<u32> {
    mt::global_input(&SPEC_COORD)
}
fn in_oob_none() -> Vec<u32> {
    mt::global_input(&SPEC_OOB_NONE)
}
fn in_oob_nan() -> Vec<u32> {
    mt::global_input(&SPEC_OOB_NAN)
}
fn in_oob_tx() -> Vec<u32> {
    mt::global_input(&SPEC_OOB_TX_PARTIAL)
}
fn in_oob_neg() -> Vec<u32> {
    mt::global_input(&SPEC_OOB_NEG)
}
fn in_swz32() -> Vec<u32> {
    mt::global_input(&SPEC_SWZ32)
}
fn in_swz64() -> Vec<u32> {
    mt::global_input(&SPEC_SWZ64)
}
fn in_swz128() -> Vec<u32> {
    mt::global_input(&SPEC_SWZ128)
}
fn in_store() -> Vec<u32> {
    mt::global_input(&SPEC_STORE)
}
fn in_prefetch() -> Vec<u32> {
    mt::global_input(&SPEC_PREFETCH)
}
fn in_multicast() -> Vec<u32> {
    mt::global_input(&SPEC_MULTICAST)
}
fn in_rawmap() -> Vec<u32> {
    mt::global_input(&SPEC_RAWMAP)
}
fn chk_rawmap(_: &[u32], o: &[u32]) -> Outcome {
    let want = [24u32, (-4i32) as u32, 512, 128, 1];
    if o == want {
        Outcome::Match("args_and_nonzero_map".to_string())
    } else {
        Outcome::Mismatch {
            first: 0,
            count: 1,
            detail: format!("expected={want:08x?} got={o:08x?}"),
        }
    }
}
fn in_ctl_raw() -> Vec<u32> {
    (0..64u32)
        .map(|i| i.wrapping_mul(0x9E37_79B1) ^ 0x7A5A_7473)
        .collect()
}
fn exp_ctl_raw(i: &[u32]) -> Vec<u32> {
    i.iter().copied().take(64).collect()
}
fn in_bulk() -> Vec<u32> {
    (0..64u32).map(|i| 0xB000_0000 | (i * 7 + 1)).collect()
}
fn in_small() -> Vec<u32> {
    vec![0u32; 4]
}

fn chk_base(i: &[u32], o: &[u32]) -> Outcome {
    mt::check_base(&SPEC_BASE, i, o)
}
fn chk_coord(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_coord(&SPEC_COORD, i, o)
}
fn chk_oob_none(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_oob(&SPEC_OOB_NONE, i, o)
}
fn chk_oob_nan(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_oob(&SPEC_OOB_NAN, i, o)
}
fn chk_oob_tx(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_oob(&SPEC_OOB_TX_PARTIAL, i, o)
}
fn chk_oob_neg(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_oob(&SPEC_OOB_NEG, i, o)
}
fn chk_swz32(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_swizzle(&SPEC_SWZ32, i, o)
}
fn chk_swz64(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_swizzle(&SPEC_SWZ64, i, o)
}
fn chk_swz128(i: &[u32], o: &[u32]) -> Outcome {
    mt::classify_swizzle(&SPEC_SWZ128, i, o)
}
fn chk_store(i: &[u32], o: &[u32]) -> Outcome {
    mt::check_store(&SPEC_STORE, i, o)
}
fn chk_prefetch(i: &[u32], o: &[u32]) -> Outcome {
    mt::check_prefetch(&SPEC_PREFETCH, i, o)
}
fn chk_multicast(i: &[u32], o: &[u32]) -> Outcome {
    mt::check_multicast(&SPEC_MULTICAST, i, o)
}
fn chk_bulk(i: &[u32], o: &[u32]) -> Outcome {
    mt::check_bulk(i, o)
}

/// `clu.rt*` の期待出力（各 block の cluster 内 rank と cluster の block 数）。
fn clu_expected(n: u32) -> Vec<u32> {
    (0..n).flat_map(|b| [b % n, n]).collect()
}
fn exp_clu_rt2(_: &[u32]) -> Vec<u32> {
    clu_expected(2)
}
fn exp_clu_rt4(_: &[u32]) -> Vec<u32> {
    clu_expected(4)
}

struct TmaArgs {
    id: &'static str,
    clause: &'static str,
    policy: Policy,
    src: &'static str,
    spec: &'static TmaSpec,
    launch: Launch,
    grid: u32,
    out_words: usize,
    make_input: fn() -> Vec<u32>,
    check: fn(&[u32], &[u32]) -> Outcome,
}

fn tma_probe(a: TmaArgs) -> ProbeSpec {
    ProbeSpec {
        id: a.id,
        ac: "AC3",
        clause: a.clause,
        home_arch: "sm_90",
        policy: a.policy,
        layout: Layout::None,
        expect: Expect::None,
        kind: Kind::Kernel,
        src: a.src,
        symbol: "",
        fixed_arch: None,
        block: 128,
        grid: a.grid,
        cluster: 0,
        out_words: a.out_words,
        make_input: a.make_input,
        check: Check::Custom(a.check),
        launch: a.launch,
        tma: Some(a.spec),
    }
}

/// AC3 の全プローブと、runtime cluster 次元の補助プローブ（`clu.rt*`）。
pub fn tma_probes() -> Vec<ProbeSpec> {
    let load_out = |s: &TmaSpec| 2 + s.dump_words as usize;
    let rec = Policy::RecordOnly;
    let ver = Policy::Verify;
    let raw0 = Launch::Raw { cluster: 0 };
    let load = |id, clause, policy, src, spec: &'static TmaSpec, launch, make_input, check| {
        tma_probe(TmaArgs {
            id,
            clause,
            policy,
            src,
            spec,
            launch,
            grid: 1,
            out_words: load_out(spec),
            make_input,
            check,
        })
    };
    let mut v = vec![
        load(
            "tma.base_cta",
            "R-TMA-BASE",
            ver,
            kt::TMA_LOAD_CTA,
            &SPEC_BASE,
            raw0,
            in_base,
            chk_base,
        ),
        load(
            "tma.base_cluster",
            "R-TMA-BASE",
            ver,
            kt::TMA_LOAD_CLUSTER,
            &SPEC_BASE,
            Launch::Raw { cluster: 1 },
            in_base,
            chk_base,
        ),
        load(
            "tma.coord",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_COORD,
            raw0,
            in_coord,
            chk_coord,
        ),
        load(
            "tma.oob_none",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_OOB_NONE,
            raw0,
            in_oob_none,
            chk_oob_none,
        ),
        load(
            "tma.oob_nan",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_OOB_NAN,
            raw0,
            in_oob_nan,
            chk_oob_nan,
        ),
        load(
            "tma.oob_tx_partial",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_OOB_TX_PARTIAL,
            raw0,
            in_oob_tx,
            chk_oob_tx,
        ),
        load(
            "tma.oob_neg",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_OOB_NEG,
            raw0,
            in_oob_neg,
            chk_oob_neg,
        ),
        load(
            "tma.swz32",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_SWZ32,
            raw0,
            in_swz32,
            chk_swz32,
        ),
        load(
            "tma.swz64",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_SWZ64,
            raw0,
            in_swz64,
            chk_swz64,
        ),
        load(
            "tma.swz128",
            "R-TMA-SEM",
            rec,
            kt::TMA_LOAD_CTA,
            &SPEC_SWZ128,
            raw0,
            in_swz128,
            chk_swz128,
        ),
        tma_probe(TmaArgs {
            id: "tma.store",
            clause: "R-TMA-XFER",
            policy: ver,
            src: kt::TMA_STORE_CTA,
            spec: &SPEC_STORE,
            launch: raw0,
            grid: 1,
            out_words: 1 + SPEC_STORE.global_words() as usize,
            make_input: in_store,
            check: chk_store,
        }),
        tma_probe(TmaArgs {
            id: "tma.prefetch",
            clause: "R-TMA-XFER",
            policy: ver,
            src: kt::TMA_PREFETCH,
            spec: &SPEC_PREFETCH,
            launch: raw0,
            grid: 1,
            out_words: 1,
            make_input: in_prefetch,
            check: chk_prefetch,
        }),
        tma_probe(TmaArgs {
            id: "tma.multicast",
            clause: "R-TMA-XFER",
            policy: ver,
            src: kt::TMA_MULTICAST,
            spec: &SPEC_MULTICAST,
            launch: Launch::Raw { cluster: 2 },
            grid: 2,
            out_words: 2 * load_out(&SPEC_MULTICAST),
            make_input: in_multicast,
            check: chk_multicast,
        }),
    ];
    // raw 起動経路の対照（R-CTL。開発機の sm_86 でも S6 まで通る）。
    v.push(ProbeSpec {
        id: "ctl.raw",
        ac: "BASE",
        clause: "R-CTL",
        home_arch: "sm_80",
        policy: Policy::Verify,
        layout: Layout::None,
        expect: Expect::None,
        kind: Kind::Kernel,
        src: kt::CTL_RAW,
        symbol: "",
        fixed_arch: None,
        block: 64,
        grid: 1,
        cluster: 0,
        out_words: 64,
        make_input: in_ctl_raw,
        check: Check::Exact(exp_ctl_raw),
        launch: Launch::Raw { cluster: 0 },
        tma: None,
    });
    let mut rawmap = tma_probe(TmaArgs {
        id: "ctl.rawmap",
        clause: "R-CTL",
        policy: ver,
        src: kt::CTL_RAWMAP,
        spec: &SPEC_RAWMAP,
        launch: raw0,
        grid: 1,
        out_words: 5,
        make_input: in_rawmap,
        check: chk_rawmap,
    });
    rawmap.ac = "BASE";
    // cuTensorMapEncodeTiled は sm_86 では CUDA_ERROR_NOT_SUPPORTED（開発機実測）のため sm_80 扱いにしない。
    rawmap.home_arch = "sm_90";
    rawmap.block = 32;
    v.push(rawmap);
    // tensor を使わない bulk コピー（PR-A と同じ (in, n_in, out, n) ABI。raw 起動で cluster 次元を与える）。
    for (id, src, cluster) in [
        ("tma.bulk_cta", kt::TMA_BULK_CTA, 0),
        ("tma.bulk_cluster", kt::TMA_BULK_CLUSTER, 1),
    ] {
        v.push(ProbeSpec {
            id,
            ac: "AC3",
            clause: "R-TMA-XFER",
            home_arch: "sm_90",
            policy: Policy::Verify,
            layout: Layout::None,
            expect: Expect::None,
            kind: Kind::Kernel,
            src,
            symbol: "",
            fixed_arch: None,
            block: 128,
            grid: 1,
            cluster: 0,
            out_words: 66,
            make_input: in_bulk,
            check: Check::Custom(chk_bulk),
            launch: Launch::Raw { cluster },
            tma: None,
        });
    }
    // runtime で cluster 次元を与える補助（clu.dims* と同じ出力を `CLUSTER_DIMENSION` 属性で観測）。
    for (id, src, n, exp) in [
        (
            "clu.rt2",
            kt::CLU_RT2,
            2u32,
            exp_clu_rt2 as fn(&[u32]) -> Vec<u32>,
        ),
        ("clu.rt4", kt::CLU_RT4, 4, exp_clu_rt4),
    ] {
        v.push(ProbeSpec {
            id,
            ac: "AC5",
            clause: "R-CLU",
            home_arch: "sm_90",
            policy: Policy::Verify,
            layout: Layout::None,
            expect: Expect::None,
            kind: Kind::Kernel,
            src,
            symbol: "",
            fixed_arch: None,
            block: 32,
            grid: n,
            cluster: 0,
            out_words: (2 * n) as usize,
            make_input: in_small,
            check: Check::Exact(exp),
            launch: Launch::Raw { cluster: n },
            tma: None,
        });
    }
    v
}
