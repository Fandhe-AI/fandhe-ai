//! デバイス上パラメータ更新（SGD in-place）の CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列。イシュー #935・
//! `docs/device-resident-update-design.md` §3.2・§5.2）。
//!
//! `sgd.rs`（呼び出し元）は本モジュールの定数を `nvrtc::compile_ptx` に
//! 渡し `CudaFunction` を得る。`kernels_elementwise.rs` と同じ理由で
//! ソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に
//! nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # 意味論の正
//!
//! `fandhe_ai_autodiff::optim::sgd::Sgd::step`（ホスト参照実装）が意味論の
//! 正。項順序（weight_decay → momentum〈`is_first_step` で `b ← g` 分岐〉
//! → nesterov → 減算）を厳密に踏襲し、`fmaf`（単精度 FMA）を用いる
//! （`backend-cpu::ops::CpuBackendOps::sgd_step_device` の `f32::mul_add`
//! と同じ FMA 契約統一方針。`.claude/rules/coding-rust.md`）。
//!
//! # in-place 更新（ホスト往復排除。本イシューの主目的）
//!
//! `elementwise.rs::CudaElementwise` の 5 カーネルと異なり、本カーネルは
//! 独立した `out` バッファを持たない: `param` を直接書き換える
//! （`velocity` も momentum 有効時は直接書き換える）。呼び出し元
//! （`sgd.rs::CudaSgd::run`）はホスト常駐 `&[f32]` ではなく、既に
//! デバイス上に存在する `CudaSlice<f32>`（`memory.rs::CudaBufferHandle`
//! 経由）へ直接ポインタを渡す。これにより `grad` のみを毎ステップ
//! アップロードし、`param`／`velocity` はステップをまたいでデバイス上に
//! 常駐させたまま更新できる（イシュー #935 の受け入れ条件）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `if (idx < numel)` の手動境界チェックを維持する（`kernels_elementwise.rs`
//! と同じ理由。`.claude/rules/coding-rust.md`）。
//!
//! # velocity 引数
//!
//! `momentum == 0` の場合、呼び出し元は `velocity` に `grad` 自身の
//! デバイスポインタを（未使用のダミーとして）渡してよい（`use_momentum`
//! フラグが 0 の分岐では `velocity` を一切読み書きしないため安全。
//! `sgd.rs::CudaSgd::run` 参照。`param` を使わない理由: `cudarc` の
//! `LaunchArgs::arg` は起動まで渡した借用を保持するため、`param` の
//! 可変借用を 2 引数分に再利用できない）。

/// 1 スレッドブロックあたりのスレッド数（1 次元）。`kernels_elementwise::
/// EW_BLOCK_DIM` と同じ値・同じ理由（PoC 実測なしの保守的な固定値）。
pub const SGD_BLOCK_DIM: u32 = 256;

/// SGD 1 ステップの in-place 更新カーネル（f32）。
///
/// パラメータ:
/// - `param`（読み書き）: 更新対象。
/// - `grad`（読み取り専用）: このステップの勾配。
/// - `velocity`（読み書き。`use_momentum == 0` の場合は未使用）:
///   momentum バッファ。
/// - `numel`: 要素数。
/// - `lr`／`momentum`／`dampening`／`weight_decay`: `SgdStepConfig` と
///   同一のハイパーパラメータ。
/// - `nesterov`／`is_first_step`／`use_momentum`: 0/1 の bool 相当フラグ
///   （NVRTC 側で `bool` 型を安定して扱うため `int` にしている。
///   `kernels_wmma.rs` 等の既存カーネルと同じ慣習）。
pub const SGD_STEP_F32: &str = r#"
extern "C" __global__ void sgd_step_f32(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float p = param[idx];
        float g = grad[idx];
        if (weight_decay != 0.0f) {
            g = fmaf(weight_decay, p, g);
        }
        if (use_momentum) {
            float prev = velocity[idx];
            float b;
            if (is_first_step) {
                b = g;
            } else {
                b = fmaf(momentum, prev, (1.0f - dampening) * g);
            }
            velocity[idx] = b;
            if (nesterov) {
                g = fmaf(momentum, b, g);
            } else {
                g = b;
            }
        }
        param[idx] = p - lr * g;
    }
}
"#;
