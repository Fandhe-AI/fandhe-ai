//! 融合 elementwise（`run_fused`）の HIP ソース（イシュー #3433・分解案 R2）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_fused_source`]）。呼び出し元は
//! `elementwise.rs` の `RocmFusedElementwise`。
//!
//! # 静的インタプリタ方式（CUDA 版との違い）
//! `backend-cuda::fused_elementwise` はプランごとに CUDA C ソースを `format!` で生成する。本移植は
//! A03（HIPRTC ソースは静的テンプレート＋検証済み数値のみ）に反するためそれを採らず、**静的カーネル
//! `fused_ew_f32` を 1 本だけ**用意し、プランは**デバイスバッファ（データ）**として渡す。
//!
//! - 引数 `(leaves, prog, out, numel, n_ops, leaf_count)`。`leaves` は全 leaf を連結した
//!   `[leaf_count * numel]`、`prog` は `[kind, a, b]` の `int` 3 つ組を `n_ops` 個並べたもの。
//! - kind: `0 = Input（a = leaf 添字）`・`1 = Add`・`2 = Mul`・`3 = Relu`・`4 = Exp`・`5 = Tanh`
//!   （Add/Mul は a・b、単項は a のみを使う）。
//! - レジスタ `float r[MAX_FUSED_OPS]` に発生順（トポロジカル順）で評価し、`out[gid] = r[n_ops-1]`。
//!   `MAX_FUSED_OPS` は `2 * MAX_FUSED_SEGMENT_NODES + 1`（演算ノード上限 + 葉の上限）で、検証済みの
//!   `u32` として `#define` で渡す。
//!
//! # 数値契約
//! - ソース先頭に `#pragma clang fp contract(off)`。融合の「mul の後に add」が暗黙に FMA 化すると
//!   CPU の `run_fused_elementwise`（縮約しない）と丸めがずれるため。FMA は使わない。
//! - relu は `x > 0 ? x : 0`（`kernels_elementwise.rs` と同じ）。`expf`／`tanhf` は GPU libm で、
//!   bit 一致は主張せず REQ-2 複合判定のみで検証する。縮約がなく `f64` アキュムレータ契約は該当しない。
//! - 性能: レジスタ配列を動的添字で引くため scratch に落ちうる。実機性能は保証しない（#3159）。
//!
//! # REQ-8
//! `gid < numel` に加え、`n_ops` の範囲・オペランド添字（`< i`）・leaf 添字（`< leaf_count`）・
//! 未知 kind をカーネル内で毎回検査する。違反時は quiet NaN を書く（0 を書く「もっともらしい誤値」を
//! 作らない）。leaf の添字計算は `long long`。ホスト側の検証（`elementwise::encode_program`）と二重化。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};
use fandhe_ai_tensor_core::MAX_FUSED_SEGMENT_NODES;

/// プログラムのレジスタ数上限（演算ノード上限 + 葉の上限）。
pub(crate) const MAX_FUSED_OPS: usize = MAX_FUSED_SEGMENT_NODES + (MAX_FUSED_SEGMENT_NODES + 1);

const _: () = assert!(MAX_FUSED_OPS == 2 * MAX_FUSED_SEGMENT_NODES + 1);
const _: () = assert!(MAX_FUSED_OPS <= u32::MAX as usize);

/// プログラムの opcode（ソースの `kind == N` と一致させる。テストで固定）。
pub(crate) const OP_INPUT: i32 = 0;
pub(crate) const OP_ADD: i32 = 1;
pub(crate) const OP_MUL: i32 = 2;
pub(crate) const OP_RELU: i32 = 3;
pub(crate) const OP_EXP: i32 = 4;
pub(crate) const OP_TANH: i32 = 5;

/// `(leaves, prog, out, numel, n_ops, leaf_count)`。
const FUSED_ARGS: [ArgKind; 6] = [
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::Ptr,
    ArgKind::I32,
    ArgKind::I32,
    ArgKind::I32,
];

pub(crate) const FUSED_SPEC: KernelSpec = KernelSpec {
    name: "fused_ew_f32",
    args: &FUSED_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM`・`MAX_FUSED_OPS` は前置する `#define` で与える。
pub(crate) const FUSED_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void fused_ew_f32(
    const float* __restrict__ leaves,
    const int* __restrict__ prog,
    float* __restrict__ out,
    int numel,
    int n_ops,
    int leaf_count)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    const float quiet_nan = __builtin_nanf("");
    if (n_ops <= 0 || n_ops > MAX_FUSED_OPS) {
        out[gid] = quiet_nan;
        return;
    }
    float r[MAX_FUSED_OPS];
    for (int i = 0; i < n_ops; i++) {
        int kind = prog[3 * i];
        int a = prog[3 * i + 1];
        int b = prog[3 * i + 2];
        float v = 0.0f;
        if (kind == 0) {
            if (a < 0 || a >= leaf_count) {
                out[gid] = quiet_nan;
                return;
            }
            v = leaves[(long long)a * (long long)numel + gid];
        } else if (kind == 1 || kind == 2) {
            if (a < 0 || a >= i || b < 0 || b >= i) {
                out[gid] = quiet_nan;
                return;
            }
            if (kind == 1) {
                v = r[a] + r[b];
            } else {
                v = r[a] * r[b];
            }
        } else if (kind == 3 || kind == 4 || kind == 5) {
            if (a < 0 || a >= i) {
                out[gid] = quiet_nan;
                return;
            }
            float x = r[a];
            if (kind == 3) {
                v = x > 0.0f ? x : 0.0f;
            } else if (kind == 4) {
                v = expf(x);
            } else {
                v = tanhf(x);
            }
        } else {
            out[gid] = quiet_nan;
            return;
        }
        r[i] = v;
    }
    out[gid] = r[n_ops - 1];
}
"#;

/// 融合カーネル用ソースを生成する（`BLOCK_DIM`・`MAX_FUSED_OPS` のみ前置）。
pub(crate) fn render_fused_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(
        &[
            ("BLOCK_DIM", INDEX_OPS_BLOCK_DIM),
            ("MAX_FUSED_OPS", MAX_FUSED_OPS as u32),
        ],
        FUSED_BODY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        assert_eq!(
            kernel_param_kinds(FUSED_BODY, FUSED_SPEC.name),
            FUSED_SPEC.args
        );
    }

    #[test]
    fn max_fused_ops_is_derived_from_segment_limit() {
        assert_eq!(MAX_FUSED_OPS, 2 * MAX_FUSED_SEGMENT_NODES + 1);
        assert_eq!(MAX_FUSED_OPS, 25);
    }

    #[test]
    fn source_defines_only_block_dim_and_max_ops_with_digit_values() {
        let s = render_fused_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(
            defines,
            ["#define BLOCK_DIM 256", "#define MAX_FUSED_OPS 25"]
        );
        for line in defines {
            let value = line.rsplit(' ').next().unwrap();
            assert!(value.chars().all(|c| c.is_ascii_digit()), "{line}");
        }
        assert!(FUSED_BODY.starts_with("#pragma clang fp contract(off)\n"));
    }

    #[test]
    fn opcodes_match_source() {
        for (op, name) in [
            (OP_INPUT, "kind == 0"),
            (OP_ADD, "kind == 1"),
            (OP_MUL, "kind == 2"),
            (OP_RELU, "kind == 3"),
            (OP_EXP, "kind == 4"),
            (OP_TANH, "kind == 5"),
        ] {
            assert!(FUSED_BODY.contains(name), "{name} (opcode {op})");
            assert_eq!(name, format!("kind == {op}"));
        }
    }

    #[test]
    fn req8_guards_present() {
        for guard in [
            "if (gid >= (long long)numel)",
            "n_ops <= 0 || n_ops > MAX_FUSED_OPS",
            "a < 0 || a >= leaf_count",
            "a < 0 || a >= i || b < 0 || b >= i",
            "(long long)a * (long long)numel + gid",
        ] {
            assert!(FUSED_BODY.contains(guard), "{guard}");
        }
        // 違反時は NaN。0 を書く誤値経路を作らない。
        assert!(!FUSED_BODY.contains("out[gid] = 0"));
        for banned in ["fma(", "__builtin_fma", "__shfl", "warpSize", "__expf"] {
            assert!(!FUSED_BODY.contains(banned), "{banned}");
        }
    }
}
