//! デバイス上パラメータ更新（SGD・Adam/AdamW の in-place 更新）の HIP カーネルソース
//! （`backend-cuda` の `kernels_sgd.rs`・`kernels_adam.rs` の移植。イシュー #3440・分解案 R9・親 #3150）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_optim_source`]）。呼び出し元は `optim.rs` の
//! `RocmOptim`。2 本のカーネルを 1 ソース（1 module）にまとめ、`#define BLOCK_DIM` だけを前置する
//! （A03: 静的テンプレート＋検証済み数値のみ。lr・beta などのハイパーパラメータはソースへ入れず
//! カーネル引数で渡す）。意味論の正は CPU 参照（`backend-cpu` の
//! `CpuBackendOps::{sgd,adam}_step_device`）で、演算列と項の並びを逐語に揃える。被覆計画は
//! `docs/backend-rocm-ops-coverage-plan.md` §4・§9。
//!
//! # 数値契約（CUDA 版からの書き換え点）
//! - CUDA 版は NVRTC の `--fmad` 既定 ON を避けるため `__fmul_rn`／`__fadd_rn`／`__fsub_rn`／
//!   `__fdiv_rn` を使っていた。ROCm の先例（`kernels_elementwise.rs` ほか）に従い、ソース先頭の
//!   `#pragma clang fp contract(off)` で暗黙の FMA 縮約を止め、素の `* + - /` で書く。
//!   `__f*_rn` は使わない（HIPRTC で通常演算子として定義されていると pragma が効かないため）。
//! - FMA を使うのは CPU 参照が `f32::mul_add` を使う箇所だけで、`__builtin_fmaf` で明示する。
//!   SGD は weight_decay・momentum の EMA・nesterov の 3 か所、Adam は coupled の `g_eff`・`m_new`・
//!   `v_new` の 3 か所。
//! - `p - lr * g` は CPU が縮約しないので FMA にしない（CUDA の NVRTC では縮約されうるが、ここでは
//!   `contract(off)` により CPU と同じ 2 回丸めになる）。
//! - `sqrtf` と `/` は HIP-Clang の既定（正しく丸められる前提。`kernels_scalar_op.rs` の Div・Sqrt と
//!   同じ立場）。近似 intrinsic（`rsqrtf`・`__fdividef`・`__fsqrt_rn` 等）は使わない。ホストモデルで
//!   CPU と bit 一致を検証するが、実機での bit 一致は主張せず #3159 で測る。
//! - 要素ごとの更新で縮約はなく、`f64` アキュムレータ契約（正規化統計・勾配の長軸縮約）は該当しない。
//!
//! # use_momentum の決め方
//! CUDA の `CudaSgd::run` は `velocity.is_some()` から決めるが、CPU 参照は `config.momentum != 0.0`
//! で決め、momentum 0 なら velocity が渡されても読まず書かない。trait doc と CPU に合わせ、
//! 呼び出し元（`optim.rs`）は `config.momentum != 0.0` を `use_momentum` として渡す。
//!
//! # REQ-8
//! 全カーネルが `gid >= numel` を検査してから読み書きする。添字は `long long`。ブロックは
//! `INDEX_OPS_BLOCK_DIM`（256 = 64 の倍数）で、wave 幅に依存する命令は使わない。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

const P: ArgKind = ArgKind::Ptr;
const I: ArgKind = ArgKind::I32;
const F: ArgKind = ArgKind::F32;

/// `(param, grad, velocity, numel, lr, momentum, dampening, weight_decay, nesterov, is_first_step, use_momentum)`。
const SGD_ARGS: [ArgKind; 11] = [P, P, P, I, F, F, F, F, I, I, I];
/// `(param, grad, m, v, numel, beta1, beta2, eps, weight_decay, decay_factor, step_size,
/// bias_correction2_sqrt, kind_decoupled, use_coupled_wd)`。
const ADAM_ARGS: [ArgKind; 14] = [P, P, P, P, I, F, F, F, F, F, F, F, I, I];

pub(crate) const SGD_STEP_SPEC: KernelSpec = KernelSpec {
    name: "sgd_step_f32",
    args: &SGD_ARGS,
};
pub(crate) const ADAM_STEP_SPEC: KernelSpec = KernelSpec {
    name: "adam_step_f32",
    args: &ADAM_ARGS,
};

/// 静的テンプレート本体。`BLOCK_DIM` は前置する `#define` で与える。
pub(crate) const OPTIM_BODY: &str = r#"#pragma clang fp contract(off)

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void sgd_step_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ velocity,
    int numel,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    int nesterov,
    int is_first_step,
    int use_momentum)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    float p = param[gid];
    float g = grad[gid];
    if (weight_decay != 0.0f) {
        g = __builtin_fmaf(weight_decay, p, g);
    }
    if (use_momentum) {
        float prev = velocity[gid];
        float b;
        if (is_first_step) {
            b = g;
        } else {
            b = __builtin_fmaf(momentum, prev, (1.0f - dampening) * g);
        }
        velocity[gid] = b;
        if (nesterov) {
            g = __builtin_fmaf(momentum, b, g);
        } else {
            g = b;
        }
    }
    param[gid] = p - lr * g;
}

extern "C" __global__ __launch_bounds__(BLOCK_DIM) void adam_step_f32(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int numel,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float decay_factor,
    float step_size,
    float bias_correction2_sqrt,
    int kind_decoupled,
    int use_coupled_wd)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)numel) {
        return;
    }
    float p = param[gid];
    float g = grad[gid];

    float g_eff = g;
    if (use_coupled_wd) {
        g_eff = __builtin_fmaf(weight_decay, p, g);
    }

    float p_eff = p;
    if (kind_decoupled) {
        p_eff = p * decay_factor;
    }

    float m_new = __builtin_fmaf(beta1, m[gid], (1.0f - beta1) * g_eff);
    float v_new = __builtin_fmaf(beta2, v[gid], ((1.0f - beta2) * g_eff) * g_eff);
    m[gid] = m_new;
    v[gid] = v_new;

    float denom = sqrtf(v_new) / bias_correction2_sqrt + eps;
    param[gid] = p_eff - (step_size * m_new) / denom;
}
"#;

/// SGD・Adam カーネル用ソースを生成する（`BLOCK_DIM` のみ前置）。
pub(crate) fn render_optim_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(&[("BLOCK_DIM", INDEX_OPS_BLOCK_DIM)], OPTIM_BODY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    const ALL: [KernelSpec; 2] = [SGD_STEP_SPEC, ADAM_STEP_SPEC];

    /// カーネル名から次のカーネル（または末尾）までの本体を切り出す。
    fn body_of(name: &str) -> &'static str {
        let start = OPTIM_BODY.find(&format!("void {name}(")).unwrap();
        let rest = &OPTIM_BODY[start + 1..];
        let end = rest
            .find("extern \"C\"")
            .map_or(OPTIM_BODY.len(), |e| start + 1 + e);
        &OPTIM_BODY[start..end]
    }

    #[test]
    fn spec_matches_source_signature() {
        for spec in ALL {
            assert_eq!(kernel_param_kinds(OPTIM_BODY, spec.name), spec.args);
        }
        assert_eq!(SGD_STEP_SPEC.args.len(), 11);
        assert_eq!(ADAM_STEP_SPEC.args.len(), 14);
    }

    #[test]
    fn source_defines_only_block_dim_and_starts_with_contract_off() {
        let s = render_optim_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define"))
            .collect();
        assert_eq!(defines, ["#define BLOCK_DIM 256"]);
        assert!(OPTIM_BODY.starts_with("#pragma clang fp contract(off)\n"));
        assert_eq!(s, render_optim_source().unwrap());
    }

    #[test]
    fn req8_guard_in_every_kernel() {
        assert_eq!(
            OPTIM_BODY.matches("if (gid >= (long long)numel)").count(),
            ALL.len()
        );
    }

    #[test]
    fn fma_only_where_cpu_uses_mul_add() {
        for name in ["sgd_step_f32", "adam_step_f32"] {
            assert_eq!(
                body_of(name).matches("__builtin_fmaf(").count(),
                3,
                "{name}"
            );
        }
        // `__builtin_fmaf(` を除いた残りに素の `fmaf(` が現れない（暗黙の別名経路を作らない）。
        assert!(!OPTIM_BODY.replace("__builtin_fmaf(", "").contains("fmaf("));
        // `p - lr * g` は縮約しない。
        assert!(body_of("sgd_step_f32").contains("param[gid] = p - lr * g;"));
    }

    #[test]
    fn banned_constructs_are_absent() {
        for banned in [
            "__fmul_rn",
            "__fadd_rn",
            "__fsub_rn",
            "__fdiv_rn",
            "rsqrtf",
            "__fdividef",
            "__fsqrt_rn",
            "#include",
            "INFINITY",
            "__shfl",
            "warpSize",
        ] {
            assert!(!OPTIM_BODY.contains(banned), "{banned}");
        }
        assert!(OPTIM_BODY.contains("sqrtf("));
    }
}
