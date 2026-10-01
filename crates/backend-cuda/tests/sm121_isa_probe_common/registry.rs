//! プローブの固定表（レジストリ）。ID・AC・条項・本来の対応アーキ・方針・
//! ソース・起動形状・入力と期待値の生成関数を 1 か所に持つ。
//!
//! 3 か所の突き合わせ（RULE.txt `PROBE:`／`CLAUSE:` 行 ↔ 本表 ↔
//! `aggregate.py` の `PROBES`／`CLAUSES`）が条項 ID の食い違いを機械的に
//! 検出する（`sm121_isa_probe_registry` テストと `aggregate.py --self-test`）。
//! PR-B（TMA の意味論）は本表へ `tma.*` を足し、RULE.txt へ対応行を足す形で
//! 追加できる構造にしてある。

use cudarc::driver::sys::CUdevice_attribute as Attr;

use super::kernels_arch::{self as ka, MACRO_CANDIDATES, TC5_MAGIC};
use super::kernels_cluster as kc;
use super::kernels_mma as km;
use super::model::{self, F32x2Op};
use super::types::{Expect, Kind, Layout, Policy};

/// 検証結果（S6／record_only の記録）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// 期待どおり（detail は記録）。
    Match(String),
    /// 不一致（最初の不一致の添字・不一致語数・内容）。
    Mismatch {
        first: usize,
        count: usize,
        detail: String,
    },
    /// 値の記録のみ（record_only）。
    Record(String),
}

/// 期待値の与え方。
#[derive(Clone, Copy)]
pub enum Check {
    /// 入力語から期待出力語を作り、全語のビット一致を要求する。
    Exact(fn(&[u32]) -> Vec<u32>),
    /// 入力・出力から判定する（ちょうど 1 lane が選ばれる等）。
    Custom(fn(&[u32], &[u32]) -> Outcome),
    /// 検証なし（accept_only・attr）。
    Skip,
}

pub struct ProbeSpec {
    pub id: &'static str,
    pub ac: &'static str,
    pub clause: &'static str,
    pub home_arch: &'static str,
    pub policy: Policy,
    pub layout: Layout,
    pub expect: Expect,
    pub kind: Kind,
    pub src: &'static str,
    pub symbol: &'static str,
    /// S1／S2 を実行 target ではなく固定の仮想アーキ・実アーキで行う
    /// （`tc5.cross` の `compute_100a`／`sm_100a`）。
    pub fixed_arch: Option<(&'static str, &'static str)>,
    pub block: u32,
    pub grid: u32,
    /// cluster 次元（0 = なし）。`__cluster_dims__` と一致させる。
    pub cluster: u32,
    pub out_words: usize,
    pub make_input: fn() -> Vec<u32>,
    pub check: Check,
}

impl ProbeSpec {
    /// RULE.txt の `PROBE:` 行（機械可読。レジストリとの完全一致を検査する）。
    pub fn rule_line(&self) -> String {
        format!(
            "PROBE: {} ac={} clause={} home={} policy={} layout={} expect={}",
            self.id,
            self.ac,
            self.clause,
            self.home_arch,
            self.policy.as_str(),
            self.layout.as_str(),
            self.expect.as_str()
        )
    }

    /// 開発機（sm_86）で S6 まで通せるプローブか（スモーク検証の対象）。
    pub fn runs_on_sm86(&self) -> bool {
        self.kind == Kind::Kernel && self.policy == Policy::Verify && self.home_arch == "sm_80"
    }
}

/// 比較して Outcome を返す（長さ違いは不一致。欠測は fail-closed）。
pub fn compare_words(expected: &[u32], got: &[u32]) -> Outcome {
    if expected.len() != got.len() {
        return Outcome::Mismatch {
            first: expected.len().min(got.len()),
            count: expected.len().abs_diff(got.len()).max(1),
            detail: format!("length expected={} got={}", expected.len(), got.len()),
        };
    }
    let mut first = None;
    let mut count = 0usize;
    for (i, (e, g)) in expected.iter().zip(got).enumerate() {
        if e != g {
            count += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    match first {
        None => Outcome::Match(format!("words={} bit_exact", expected.len())),
        Some(i) => Outcome::Mismatch {
            first: i,
            count,
            detail: format!(
                "first_index={i} expected=0x{:08x} got=0x{:08x} mismatched_words={count} of {}",
                expected[i],
                got[i],
                expected.len()
            ),
        },
    }
}

/// 方針に従い出力を評価する。
pub fn evaluate(spec: &ProbeSpec, input: &[u32], out: &[u32]) -> Outcome {
    match spec.check {
        Check::Exact(f) => compare_words(&f(input), out),
        Check::Custom(f) => f(input, out),
        Check::Skip => Outcome::Record("no_check".to_string()),
    }
}

// ------------------------------------------------------------ 入力・期待値の関数

fn zeros_1k() -> Vec<u32> {
    vec![0u32; 1024]
}

fn small_pattern() -> Vec<u32> {
    (0..256u32)
        .map(|i| i.wrapping_mul(7).wrapping_add(3))
        .collect()
}

fn in_f16_k16_f32() -> Vec<u32> {
    model::f16_family_input(16, false, false)
}
fn exp_f16_k16_f32(_: &[u32]) -> Vec<u32> {
    model::f16_family_expected(16, false)
}
fn in_f16_k8_f32() -> Vec<u32> {
    model::f16_family_input(8, false, false)
}
fn exp_f16_k8_f32(_: &[u32]) -> Vec<u32> {
    model::f16_family_expected(8, false)
}
fn in_f16_k16_f16() -> Vec<u32> {
    model::f16_family_input(16, false, true)
}
fn exp_f16_k16_f16(_: &[u32]) -> Vec<u32> {
    model::f16_family_expected(16, true)
}
fn in_bf16_k16() -> Vec<u32> {
    model::f16_family_input(16, true, false)
}
fn exp_bf16_k16(_: &[u32]) -> Vec<u32> {
    model::f16_family_expected(16, false)
}
fn in_bf16_k8() -> Vec<u32> {
    model::f16_family_input(8, true, false)
}
fn exp_bf16_k8(_: &[u32]) -> Vec<u32> {
    model::f16_family_expected(8, false)
}
fn in_tf32_k8() -> Vec<u32> {
    model::tf32_input(8)
}
fn exp_tf32_k8(_: &[u32]) -> Vec<u32> {
    model::tf32_expected(8)
}
fn in_tf32_k4() -> Vec<u32> {
    model::tf32_input(4)
}
fn exp_tf32_k4(_: &[u32]) -> Vec<u32> {
    model::tf32_expected(4)
}
fn exp_f64_m8n8k4(_: &[u32]) -> Vec<u32> {
    model::f64_m8n8k4_expected()
}
fn exp_ld_x1(_: &[u32]) -> Vec<u32> {
    model::ldmatrix_expected(1, false)
}
fn exp_ld_x2(_: &[u32]) -> Vec<u32> {
    model::ldmatrix_expected(2, false)
}
fn exp_ld_x4(_: &[u32]) -> Vec<u32> {
    model::ldmatrix_expected(4, false)
}
fn exp_ld_x4_trans(_: &[u32]) -> Vec<u32> {
    model::ldmatrix_expected(4, true)
}
fn exp_f32x2_add(i: &[u32]) -> Vec<u32> {
    model::f32x2_expected(i, F32x2Op::Add)
}
fn exp_f32x2_mul(i: &[u32]) -> Vec<u32> {
    model::f32x2_expected(i, F32x2Op::Mul)
}
fn exp_f32x2_fma(i: &[u32]) -> Vec<u32> {
    model::f32x2_expected(i, F32x2Op::Fma)
}
fn in_f32x2_2() -> Vec<u32> {
    model::f32x2_input(false)
}
fn in_f32x2_3() -> Vec<u32> {
    model::f32x2_input(true)
}
fn in_half2_f16() -> Vec<u32> {
    model::fma_half2_input(false)
}
fn exp_half2_f16(_: &[u32]) -> Vec<u32> {
    model::fma_half2_expected(false)
}
fn in_half2_bf16() -> Vec<u32> {
    model::fma_half2_input(true)
}
fn exp_half2_bf16(_: &[u32]) -> Vec<u32> {
    model::fma_half2_expected(true)
}
fn exp_ctl_copy(i: &[u32]) -> Vec<u32> {
    i.iter().copied().take(64).collect()
}
fn in_ctl() -> Vec<u32> {
    (0..64u32)
        .map(|i| i.wrapping_mul(0x9E37_79B1) ^ 0x5BD1_E995)
        .collect()
}
fn exp_snr_dec(i: &[u32]) -> Vec<u32> {
    i.iter().take(128).map(|v| v.wrapping_add(1)).collect()
}
fn exp_snr_incdec(i: &[u32]) -> Vec<u32> {
    i.iter().take(256).map(|v| v.wrapping_mul(2)).collect()
}
fn exp_hop(i: &[u32]) -> Vec<u32> {
    i.iter().take(64).map(|v| v.wrapping_add(1)).collect()
}
fn in_hop() -> Vec<u32> {
    small_pattern()
}

fn check_elect(_: &[u32], out: &[u32]) -> Outcome {
    let bad = out.iter().filter(|&&v| v > 1).count();
    let leaders: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v == 1)
        .map(|(i, _)| i)
        .collect();
    if out.len() == 32 && bad == 0 && leaders.len() == 1 {
        Outcome::Match(format!("leader_lane={}", leaders[0]))
    } else {
        Outcome::Mismatch {
            first: leaders.first().copied().unwrap_or(0),
            count: leaders.len().max(bad).max(1),
            detail: format!(
                "elected_lanes={leaders:?} non_boolean_words={bad} len={}",
                out.len()
            ),
        }
    }
}

fn check_tc5_alloc(_: &[u32], out: &[u32]) -> Outcome {
    if out.len() == 2 && out[1] == TC5_MAGIC {
        Outcome::Match(format!("tmem_addr=0x{:08x}", out[0]))
    } else {
        Outcome::Mismatch {
            first: 1,
            count: 1,
            detail: format!("completion_magic expected=0x{TC5_MAGIC:08x} got={out:08x?}"),
        }
    }
}

fn check_tc5_ld(_: &[u32], out: &[u32]) -> Outcome {
    if out.len() == 2 && out[1] == TC5_MAGIC {
        Outcome::Record(format!("tmem_ld_word=0x{:08x}", out[0]))
    } else {
        Outcome::Mismatch {
            first: 1,
            count: 1,
            detail: format!("completion_magic expected=0x{TC5_MAGIC:08x} got={out:08x?}"),
        }
    }
}

fn check_macro_arch(_: &[u32], out: &[u32]) -> Outcome {
    if out.len() != MACRO_CANDIDATES.len() {
        return Outcome::Mismatch {
            first: 0,
            count: 1,
            detail: format!(
                "length expected={} got={}",
                MACRO_CANDIDATES.len(),
                out.len()
            ),
        };
    }
    let pairs: Vec<String> = MACRO_CANDIDATES
        .iter()
        .zip(out)
        .map(|(name, v)| format!("{name}={v}"))
        .collect();
    Outcome::Record(pairs.join(" "))
}

fn exp_clu_dims_1(_: &[u32]) -> Vec<u32> {
    clu_expected(1)
}
fn exp_clu_dims_2(_: &[u32]) -> Vec<u32> {
    clu_expected(2)
}
fn exp_clu_dims_4(_: &[u32]) -> Vec<u32> {
    clu_expected(4)
}
fn exp_clu_dims_8(_: &[u32]) -> Vec<u32> {
    clu_expected(8)
}
fn exp_clu_dims_16(_: &[u32]) -> Vec<u32> {
    clu_expected(16)
}
fn clu_expected(n: u32) -> Vec<u32> {
    (0..n).flat_map(|b| [b % n, n]).collect()
}
fn exp_dsmem(_: &[u32]) -> Vec<u32> {
    vec![0xA001, 0xA000]
}
fn in_small() -> Vec<u32> {
    vec![0u32; 4]
}

fn empty() -> Vec<u32> {
    vec![0u32; 4]
}

macro_rules! probe {
    (
        $id:literal, $ac:literal, $clause:literal, $home:literal,
        $policy:ident, $layout:ident, $expect:ident, $src:expr, $block:literal, $grid:literal,
        $out:literal, $input:expr, $check:expr
    ) => {
        ProbeSpec {
            id: $id,
            ac: $ac,
            clause: $clause,
            home_arch: $home,
            policy: Policy::$policy,
            layout: Layout::$layout,
            expect: Expect::$expect,
            kind: Kind::Kernel,
            src: $src,
            symbol: "",
            fixed_arch: None,
            block: $block,
            grid: $grid,
            cluster: 0,
            out_words: $out,
            make_input: $input,
            check: $check,
        }
    };
}

/// 全プローブ（ID は RULE.txt の `PROBE:` 行と一致）。
pub fn probes() -> Vec<ProbeSpec> {
    let mut v = vec![
        // 対照・マクロ（AC1／BASE）
        probe!(
            "ctl.copy",
            "BASE",
            "R-CTL",
            "sm_80",
            Verify,
            None,
            None,
            ka::CTL_COPY,
            64,
            1,
            64,
            in_ctl,
            Check::Exact(exp_ctl_copy)
        ),
        probe!(
            "macro.arch",
            "AC1",
            "R-TC5",
            "sm_80",
            RecordOnly,
            None,
            None,
            ka::MACRO_ARCH,
            32,
            1,
            11,
            empty,
            Check::Custom(check_macro_arch)
        ),
        // AC1 tcgen05
        probe!(
            "tc5.alloc",
            "AC1",
            "R-TC5",
            "sm_100a",
            Verify,
            None,
            Reject121,
            ka::TC5_ALLOC,
            32,
            1,
            2,
            empty,
            Check::Custom(check_tc5_alloc)
        ),
        probe!(
            "tc5.ld",
            "AC1",
            "R-TC5",
            "sm_100a",
            RecordOnly,
            None,
            Reject121,
            ka::TC5_LD,
            32,
            1,
            2,
            empty,
            Check::Custom(check_tc5_ld)
        ),
        probe!(
            "tc5.cross",
            "AC1",
            "R-TC5",
            "sm_100a",
            AcceptOnly,
            None,
            Reject121,
            ka::TC5_ALLOC,
            32,
            1,
            2,
            empty,
            Check::Skip
        ),
        // AC4 mma
        probe!(
            "mma.tf32.m16n8k8",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_TF32_M16N8K8,
            32,
            1,
            128,
            in_tf32_k8,
            Check::Exact(exp_tf32_k8)
        ),
        probe!(
            "mma.tf32.m16n8k4",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_TF32_M16N8K4,
            32,
            1,
            128,
            in_tf32_k4,
            Check::Exact(exp_tf32_k4)
        ),
        probe!(
            "mma.f16.m16n8k16.f32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_F16_M16N8K16_F32,
            32,
            1,
            128,
            in_f16_k16_f32,
            Check::Exact(exp_f16_k16_f32)
        ),
        probe!(
            "mma.f16.m16n8k8.f32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_F16_M16N8K8_F32,
            32,
            1,
            128,
            in_f16_k8_f32,
            Check::Exact(exp_f16_k8_f32)
        ),
        probe!(
            "mma.f16.m16n8k16.f16",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_F16_M16N8K16_F16,
            32,
            1,
            64,
            in_f16_k16_f16,
            Check::Exact(exp_f16_k16_f16)
        ),
        probe!(
            "mma.bf16.m16n8k16.f32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_BF16_M16N8K16_F32,
            32,
            1,
            128,
            in_bf16_k16,
            Check::Exact(exp_bf16_k16)
        ),
        probe!(
            "mma.bf16.m16n8k8.f32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_BF16_M16N8K8_F32,
            32,
            1,
            128,
            in_bf16_k8,
            Check::Exact(exp_bf16_k8)
        ),
        probe!(
            "mma.f64.m8n8k4",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_F64_M8N8K4,
            32,
            1,
            128,
            model::f64_m8n8k4_input,
            Check::Exact(exp_f64_m8n8k4)
        ),
        probe!(
            "mma.f16.m8n8k4",
            "AC4",
            "R-MMA",
            "sm_80",
            AcceptOnly,
            None,
            None,
            km::MMA_F16_M8N8K4,
            32,
            1,
            256,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.f64.m16n8k4",
            "AC4",
            "R-MMA",
            "sm_90",
            AcceptOnly,
            None,
            None,
            km::MMA_F64_M16N8K4,
            32,
            1,
            256,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.f64.m16n8k8",
            "AC4",
            "R-MMA",
            "sm_90",
            AcceptOnly,
            None,
            None,
            km::MMA_F64_M16N8K8,
            32,
            1,
            256,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.f64.m16n8k16",
            "AC4",
            "R-MMA",
            "sm_90",
            AcceptOnly,
            None,
            None,
            km::MMA_F64_M16N8K16,
            32,
            1,
            256,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.s8.m16n8k32",
            "AC4",
            "R-MMA",
            "sm_80",
            AcceptOnly,
            None,
            None,
            km::MMA_S8_M16N8K32,
            32,
            1,
            128,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.e4m3.m16n8k32",
            "AC4",
            "R-MMA",
            "sm_89",
            AcceptOnly,
            None,
            None,
            km::MMA_E4M3_M16N8K32,
            32,
            1,
            128,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.e5m2.m16n8k32",
            "AC4",
            "R-MMA",
            "sm_89",
            AcceptOnly,
            None,
            None,
            km::MMA_E5M2_M16N8K32,
            32,
            1,
            128,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.f8f6f4.m16n8k32",
            "AC4",
            "R-MMA",
            "sm_120a",
            AcceptOnly,
            None,
            None,
            km::MMA_F8F6F4_M16N8K32,
            32,
            1,
            128,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.block_scale.m16n8k64",
            "AC4",
            "R-MMA",
            "sm_120a",
            AcceptOnly,
            None,
            None,
            km::MMA_BLOCK_SCALE_M16N8K64,
            32,
            1,
            128,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "mma.ldmatrix.x1",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_LDMATRIX_X1,
            32,
            1,
            128,
            model::ldmatrix_smem_input,
            Check::Exact(exp_ld_x1)
        ),
        probe!(
            "mma.ldmatrix.x2",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_LDMATRIX_X2,
            32,
            1,
            128,
            model::ldmatrix_smem_input,
            Check::Exact(exp_ld_x2)
        ),
        probe!(
            "mma.ldmatrix.x4",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_LDMATRIX_X4,
            32,
            1,
            128,
            model::ldmatrix_smem_input,
            Check::Exact(exp_ld_x4)
        ),
        probe!(
            "mma.ldmatrix.x4_trans",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            Verified,
            None,
            km::MMA_LDMATRIX_X4_TRANS,
            32,
            1,
            128,
            model::ldmatrix_smem_input,
            Check::Exact(exp_ld_x4_trans)
        ),
        probe!(
            "mma.stmatrix.x4",
            "AC4",
            "R-MMA",
            "sm_90",
            Verify,
            Unverified,
            None,
            km::MMA_STMATRIX_X4,
            32,
            1,
            128,
            model::stmatrix_input,
            Check::Exact(model::stmatrix_expected)
        ),
        // AC4 SIMT
        probe!(
            "simt.fma_f32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            None,
            None,
            km::SIMT_FMA_F32,
            32,
            1,
            32,
            model::fma_f32_input,
            Check::Exact(model::fma_f32_expected)
        ),
        probe!(
            "simt.fma_f64",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            None,
            None,
            km::SIMT_FMA_F64,
            32,
            1,
            64,
            model::fma_f64_input,
            Check::Exact(model::fma_f64_expected)
        ),
        probe!(
            "simt.fma_f16x2",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            None,
            None,
            km::SIMT_FMA_F16X2,
            32,
            1,
            32,
            in_half2_f16,
            Check::Exact(exp_half2_f16)
        ),
        probe!(
            "simt.fma_bf16x2",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            None,
            None,
            km::SIMT_FMA_BF16X2,
            32,
            1,
            32,
            in_half2_bf16,
            Check::Exact(exp_half2_bf16)
        ),
        probe!(
            "simt.f32x2_add",
            "AC4",
            "R-MMA",
            "sm_100",
            Verify,
            Unverified,
            None,
            km::SIMT_F32X2_ADD,
            32,
            1,
            64,
            in_f32x2_2,
            Check::Exact(exp_f32x2_add)
        ),
        probe!(
            "simt.f32x2_mul",
            "AC4",
            "R-MMA",
            "sm_100",
            Verify,
            Unverified,
            None,
            km::SIMT_F32X2_MUL,
            32,
            1,
            64,
            in_f32x2_2,
            Check::Exact(exp_f32x2_mul)
        ),
        probe!(
            "simt.f32x2_fma",
            "AC4",
            "R-MMA",
            "sm_100",
            Verify,
            Unverified,
            None,
            km::SIMT_F32X2_FMA,
            32,
            1,
            64,
            in_f32x2_3,
            Check::Exact(exp_f32x2_fma)
        ),
        probe!(
            "simt.cvt_tf32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            None,
            None,
            km::SIMT_CVT_TF32,
            32,
            1,
            32,
            model::cvt_tf32_input,
            Check::Exact(model::cvt_tf32_expected)
        ),
        probe!(
            "simt.elect_sync",
            "AC4",
            "R-MMA",
            "sm_90",
            Verify,
            None,
            None,
            km::SIMT_ELECT_SYNC,
            32,
            1,
            32,
            empty,
            Check::Custom(check_elect)
        ),
        probe!(
            "simt.redux_u32",
            "AC4",
            "R-MMA",
            "sm_80",
            Verify,
            None,
            None,
            km::SIMT_REDUX_U32,
            32,
            1,
            128,
            model::redux_u32_input,
            Check::Exact(model::redux_u32_expected)
        ),
        probe!(
            "simt.redux_f32",
            "AC4",
            "R-MMA",
            "sm_100a",
            AcceptOnly,
            None,
            None,
            km::SIMT_REDUX_F32,
            32,
            1,
            32,
            zeros_1k,
            Check::Skip
        ),
        // AC5 Hopper 差分・setmaxnreg
        probe!(
            "wgmma.m64n8k16",
            "AC5",
            "R-HOPPER",
            "sm_90a",
            AcceptOnly,
            None,
            None,
            ka::WGMMA_M64N8K16,
            128,
            1,
            512,
            zeros_1k,
            Check::Skip
        ),
        probe!(
            "hop.griddepcontrol",
            "AC5",
            "R-HOPPER",
            "sm_90",
            Verify,
            None,
            None,
            ka::HOP_GRIDDEPCONTROL,
            64,
            1,
            64,
            in_hop,
            Check::Exact(exp_hop)
        ),
        probe!(
            "hop.fence_proxy_async",
            "AC5",
            "R-HOPPER",
            "sm_90",
            Verify,
            None,
            None,
            ka::HOP_FENCE_PROXY_ASYNC,
            64,
            1,
            64,
            in_hop,
            Check::Exact(exp_hop)
        ),
        probe!(
            "snr.dec",
            "AC5",
            "R-SNR",
            "sm_90a",
            Verify,
            None,
            None,
            ka::SNR_DEC,
            128,
            1,
            128,
            in_hop,
            Check::Exact(exp_snr_dec)
        ),
        probe!(
            "snr.incdec",
            "AC5",
            "R-SNR",
            "sm_90a",
            Verify,
            None,
            None,
            ka::SNR_INCDEC,
            256,
            1,
            256,
            in_hop,
            Check::Exact(exp_snr_incdec)
        ),
    ];
    // cluster・DSMEM（R-CLU）。grid = cluster 次元（1 cluster）。
    let cluster_probes = [
        (
            "clu.dims1",
            kc::CLU_DIMS1,
            1u32,
            exp_clu_dims_1 as fn(&[u32]) -> Vec<u32>,
        ),
        ("clu.dims2", kc::CLU_DIMS2, 2, exp_clu_dims_2),
        ("clu.dims4", kc::CLU_DIMS4, 4, exp_clu_dims_4),
        ("clu.dims8", kc::CLU_DIMS8, 8, exp_clu_dims_8),
        ("clu.dims16", kc::CLU_DIMS16, 16, exp_clu_dims_16),
    ];
    for (id, src, n, exp) in cluster_probes {
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
            cluster: n,
            out_words: (2 * n) as usize,
            make_input: in_small,
            check: Check::Exact(exp),
        });
    }
    v.push(ProbeSpec {
        id: "clu.dsmem",
        ac: "AC5",
        clause: "R-CLU",
        home_arch: "sm_90",
        policy: Policy::Verify,
        layout: Layout::None,
        expect: Expect::None,
        kind: Kind::Kernel,
        src: kc::CLU_DSMEM,
        symbol: "",
        fixed_arch: None,
        block: 32,
        grid: 2,
        cluster: 2,
        out_words: 2,
        make_input: in_small,
        check: Check::Exact(exp_dsmem),
    });
    // デバイス属性（R-GUIDE の測定元）。カーネルを持たない。
    for id in ["attr.limits", "attr.cluster", "attr.misc"] {
        v.push(ProbeSpec {
            id,
            ac: "AC2",
            clause: "R-GUIDE",
            home_arch: "sm_80",
            policy: Policy::Attr,
            layout: Layout::None,
            expect: Expect::None,
            kind: Kind::Attr,
            src: "",
            symbol: "",
            fixed_arch: None,
            block: 0,
            grid: 0,
            cluster: 0,
            out_words: 0,
            make_input: empty,
            check: Check::Skip,
        });
    }
    // symbol は id の '.' を '_' に置換した C シンボル名（tc5.cross は tc5.alloc のソースを使うため
    // 例外として `tc5_alloc`）。
    for p in &mut v {
        p.symbol = symbol_of(p.id);
    }
    // tc5.cross は tc5.alloc と同じソース（固定アーキ compute_100a で S1／S2 を行う）。
    for p in &mut v {
        if p.id == "tc5.cross" {
            p.fixed_arch = Some(("compute_100a", "sm_100a"));
        }
    }
    v
}

/// `id` から C シンボル名を得る（`tc5.cross` のみ共有ソースのため `tc5_alloc`）。
/// 返す `&'static str` は固定表由来（実行時文字列の `Box::leak` を避ける）。
pub fn symbol_of(id: &str) -> &'static str {
    const SYMBOLS: &[(&str, &str)] = &[
        ("ctl.copy", "ctl_copy"),
        ("macro.arch", "macro_arch"),
        ("tc5.alloc", "tc5_alloc"),
        ("tc5.ld", "tc5_ld"),
        ("tc5.cross", "tc5_alloc"),
        ("mma.tf32.m16n8k8", "mma_tf32_m16n8k8"),
        ("mma.tf32.m16n8k4", "mma_tf32_m16n8k4"),
        ("mma.f16.m16n8k16.f32", "mma_f16_m16n8k16_f32"),
        ("mma.f16.m16n8k8.f32", "mma_f16_m16n8k8_f32"),
        ("mma.f16.m16n8k16.f16", "mma_f16_m16n8k16_f16"),
        ("mma.bf16.m16n8k16.f32", "mma_bf16_m16n8k16_f32"),
        ("mma.bf16.m16n8k8.f32", "mma_bf16_m16n8k8_f32"),
        ("mma.f64.m8n8k4", "mma_f64_m8n8k4"),
        ("mma.f16.m8n8k4", "mma_f16_m8n8k4"),
        ("mma.f64.m16n8k4", "mma_f64_m16n8k4"),
        ("mma.f64.m16n8k8", "mma_f64_m16n8k8"),
        ("mma.f64.m16n8k16", "mma_f64_m16n8k16"),
        ("mma.s8.m16n8k32", "mma_s8_m16n8k32"),
        ("mma.e4m3.m16n8k32", "mma_e4m3_m16n8k32"),
        ("mma.e5m2.m16n8k32", "mma_e5m2_m16n8k32"),
        ("mma.f8f6f4.m16n8k32", "mma_f8f6f4_m16n8k32"),
        ("mma.block_scale.m16n8k64", "mma_block_scale_m16n8k64"),
        ("mma.ldmatrix.x1", "mma_ldmatrix_x1"),
        ("mma.ldmatrix.x2", "mma_ldmatrix_x2"),
        ("mma.ldmatrix.x4", "mma_ldmatrix_x4"),
        ("mma.ldmatrix.x4_trans", "mma_ldmatrix_x4_trans"),
        ("mma.stmatrix.x4", "mma_stmatrix_x4"),
        ("simt.fma_f32", "simt_fma_f32"),
        ("simt.fma_f64", "simt_fma_f64"),
        ("simt.fma_f16x2", "simt_fma_f16x2"),
        ("simt.fma_bf16x2", "simt_fma_bf16x2"),
        ("simt.f32x2_add", "simt_f32x2_add"),
        ("simt.f32x2_mul", "simt_f32x2_mul"),
        ("simt.f32x2_fma", "simt_f32x2_fma"),
        ("simt.cvt_tf32", "simt_cvt_tf32"),
        ("simt.elect_sync", "simt_elect_sync"),
        ("simt.redux_u32", "simt_redux_u32"),
        ("simt.redux_f32", "simt_redux_f32"),
        ("wgmma.m64n8k16", "wgmma_m64n8k16"),
        ("hop.griddepcontrol", "hop_griddepcontrol"),
        ("hop.fence_proxy_async", "hop_fence_proxy_async"),
        ("snr.dec", "snr_dec"),
        ("snr.incdec", "snr_incdec"),
        ("clu.dims1", "clu_dims1"),
        ("clu.dims2", "clu_dims2"),
        ("clu.dims4", "clu_dims4"),
        ("clu.dims8", "clu_dims8"),
        ("clu.dims16", "clu_dims16"),
        ("clu.dsmem", "clu_dsmem"),
        ("attr.limits", ""),
        ("attr.cluster", ""),
        ("attr.misc", ""),
    ];
    SYMBOLS
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, v)| *v)
        .unwrap_or("")
}

/// ホワイトリスト検索（未知の ID は `None`。呼び出し側が fail-loud にする）。
pub fn probe_by_id(id: &str) -> Option<ProbeSpec> {
    probes().into_iter().find(|p| p.id == id)
}

/// 条項 ID の全集合（RULE.txt の `CLAUSE:` 行・`aggregate.py` の `CLAUSES` と一致させる）。
pub const CLAUSES: [&str; 12] = [
    "G0", "R-STAGE", "R-CTL", "R-HOME", "R-TC5", "R-GUIDE", "R-MMA", "R-CLU", "R-SNR", "R-HOPPER",
    "R-LEGACY", "R-COMMON",
];

/// デバイス属性プローブが記録する属性（名前 → CUdevice_attribute）。名前は
/// `guide_claims.tsv` が参照する。
pub fn attr_table(id: &str) -> &'static [(&'static str, Attr)] {
    match id {
        "attr.limits" => &[
            (
                "max_threads_per_multiprocessor",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            ),
            (
                "max_blocks_per_multiprocessor",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
            ),
            (
                "max_shared_memory_per_multiprocessor",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            ),
            (
                "max_shared_memory_per_block",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            ),
            (
                "max_shared_memory_per_block_optin",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            ),
            (
                "reserved_shared_memory_per_block",
                Attr::CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
            ),
            (
                "max_registers_per_multiprocessor",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
            ),
            (
                "max_registers_per_block",
                Attr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
            ),
            ("warp_size", Attr::CU_DEVICE_ATTRIBUTE_WARP_SIZE),
            (
                "multiprocessor_count",
                Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            ),
        ],
        "attr.cluster" => &[("cluster_launch", Attr::CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH)],
        "attr.misc" => &[
            ("clock_rate", Attr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE),
            (
                "memory_clock_rate",
                Attr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            ),
            (
                "global_memory_bus_width",
                Attr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
            ),
            ("l2_cache_size", Attr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
            (
                "unified_addressing",
                Attr::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
            ),
            ("managed_memory", Attr::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY),
            (
                "concurrent_managed_access",
                Attr::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
            ),
            (
                "pageable_memory_access",
                Attr::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            ),
            (
                "pageable_memory_access_uses_host_page_tables",
                Attr::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
            ),
            (
                "direct_managed_mem_access_from_host",
                Attr::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
            ),
            ("integrated", Attr::CU_DEVICE_ATTRIBUTE_INTEGRATED),
            (
                "host_native_atomic_supported",
                Attr::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
            ),
        ],
        _ => &[],
    }
}
