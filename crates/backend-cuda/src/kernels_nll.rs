//! 負対数尤度損失（`NLLLoss`）融合カーネルの CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列。イシュー #1738・親イシュー
//! #1609「損失関数の拡張」）。`kernels_mse.rs` を雛形にした同型構成。
//!
//! `nll.rs`（呼び出し元）は本モジュールの 3 定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る。`kernels_mse.rs` と同じ理由でソースを
//! `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA
//! ヘッダを一切要求しない。`.claude/rules/deps-policy.md`）。
//!
//! # forward の 2 段構成（`nll_partial_f32` → `nll_finalize_f32`）
//!
//! `kernels_mse.rs` と同じ 2 段 reduction 方式だが、縮約対象は要素
//! （`numel`）ではなく**サンプル**（`n_samples = outer*inner`。1 サンプル
//! = 1 スレッドの粒度）で、各サンプル `s` は正解クラス添字
//! `targets[s]`（`const int*`）を読み `−input[(o·C+t)·inner+i]` を
//! 加算する（`o = s/inner`, `i = s%inner`）。`class_dim` は `C`（クラス
//! 数）・`inner`・`outer` の 3 パラメータとしてホスト側から渡す。
//!
//! # 決定性（float atomicAdd を使わない理由）
//!
//! `kernels_mse.rs` と同じ理由・同じ設計（ブロック間は `blockIdx.x`
//! 昇順の逐次結合、ブロック内は `__shfl_xor_sync` の固定 offset 列）。
//!
//! # backward（`nll_backward_f32`）の書き込み一意性
//!
//! 1 サンプル 1 スレッドで `dinput[(o·C+t)·inner+i] = −scale` を書く。
//! `(o, i)` の組ごとに書き込み添字 `idx = o·C·inner + t·inner + i` は
//! 一意（`i < inner` かつ `t*inner` は `inner` の倍数のため `idx mod
//! inner = i` から `i` が、`(idx - i)/inner mod C = t` から `t` が、
//! `idx/(C·inner) = o` から `o` がそれぞれ一意に復元できる）ため、
//! 異なるサンプルが同じ `idx` へ書き込むことはなく `atomicAdd` は不要
//! （呼び出し元 `nll.rs::run_nll_backward_f32` が `dinput` を
//! `alloc_zeroed_f32` でゼロ初期化してから本カーネルを起動する契約。
//! ターゲット位置以外は `0.0` のまま残る）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `nll_partial_f32`・`nll_backward_f32` は `idx < n_samples`
//! （grid-stride ループの場合は `for (...; idx < n_samples; ...)`）の
//! 手動境界チェックを維持する。`nll_finalize_f32` も `idx <
//! num_partials` を維持する。`t`（`targets[idx]`）の範囲外検査
//! （`0 <= t < num_classes`）はホスト側起動 API（`nll.rs::
//! validate_nll_buffers`）が起動前に検証し拒否する契約だが、`t` は
//! カーネル自身も `0 <= t < num_classes` を手動検査してから
//! `input_idx`／`dinput` の添字算出へ用いる（PR #1850 codex-review P0
//! 是正 2: ホスト側検証を回避する経路〈将来の呼び出し口追加等〉が
//! 生じても範囲外読み書きへ波及しない縦深防御。REQ-8 の手動境界検査
//! 方針・`.claude/rules/coding-rust.md` unsafe 不変条件保証と同じ
//! 理由）。範囲外 `t` は寄与ゼロ（forward はスキップ・backward は
//! 書き込みなし）として扱う（ホスト側検証済みのため通常到達しない）。
//!
//! # 意味論の正
//!
//! `backend-cpu::nll`（`crates/backend-cpu/src/nll.rs`）・
//! `fandhe_ai_autodiff::eval::nll_loss` が意味論の正。累積順序（CPU:
//! 固定チャンク／CUDA: grid-stride → warp butterfly → ブロック間逐次
//! 結合）は異なるため、バックエンド間の数値突合は統一複合判定
//! 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で検証する
//! （`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_mse::MSE_BLOCK_DIM`
/// と同値・同じ理由）。
pub const NLL_BLOCK_DIM: u32 = 256;

/// forward 2 段目（`nll_finalize_f32`）が単一ブロックで処理しきれる
/// `partial` の最大長（`kernels_mse::MSE_MAX_BLOCKS` と同値・同じ理由）。
pub const NLL_MAX_BLOCKS: u32 = 1024;

/// forward 1 段目: 各ブロックが担当区間の `Σ −input[(o·C+t)·inner+i]`
/// を計算し `partial[blockIdx.x]` へ書く（`kernels_mse::MSE_PARTIAL_F32`
/// のサンプル版。縮約対象は `numel` ではなく `n_samples = outer*inner`）。
pub const NLL_PARTIAL_F32: &str = r#"
extern "C" __global__ void nll_partial_f32(
    const float* __restrict__ input,
    const int* __restrict__ targets,
    float* __restrict__ partial,
    int outer,
    int num_classes,
    int inner,
    int n_samples)
{
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n_samples; idx += stride) {
        int o = (int)(idx / inner);
        int i = (int)(idx % inner);
        int t = targets[idx];
        // REQ-8: `t` の手動境界検査（モジュール冒頭コメント参照）。
        // ホスト側 `validate_nll_buffers` が事前拒否する契約のため
        // 通常到達しないが、範囲外 `t` は寄与ゼロとして安全側へ倒す。
        if (t < 0 || t >= num_classes) {
            continue;
        }
        long long input_idx = ((long long)o * num_classes + t) * inner + i;
        acc -= input[input_idx];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_sum;
        }
    }
}
"#;

/// forward 2 段目: `partial`（`num_partials` 要素。1 ブロックのみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く
/// （`kernels_mse::MSE_FINALIZE_F32` と同一構造）。
pub const NLL_FINALIZE_F32: &str = r#"
extern "C" __global__ void nll_finalize_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int num_partials,
    float factor)
{
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = 0.0f;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc += partial[idx];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            out[0] = block_sum * factor;
        }
    }
}
"#;

/// backward: `dInput[(o·C+t)·inner+i] = −scale`（1 スレッド 1 サンプル。
/// `dinput` は呼び出し元が `alloc_zeroed_f32` で事前ゼロ初期化済み。
/// モジュール冒頭「backward の書き込み一意性」参照）。
pub const NLL_BACKWARD_F32: &str = r#"
extern "C" __global__ void nll_backward_f32(
    const int* __restrict__ targets,
    float* __restrict__ dinput,
    int outer,
    int num_classes,
    int inner,
    int n_samples,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        int o = idx / inner;
        int i = idx % inner;
        int t = targets[idx];
        // REQ-8: `t` の手動境界検査（モジュール冒頭コメント参照）。
        // ホスト側 `validate_nll_buffers` が事前拒否する契約のため
        // 通常到達しないが、範囲外 `t` は書き込みなしとして安全側へ
        // 倒す（`dinput` はゼロ初期化済みのため当該位置は `0.0` のまま
        // 残る）。
        if (t >= 0 && t < num_classes) {
            long long input_idx = ((long long)o * num_classes + t) * inner + i;
            dinput[input_idx] = -scale;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 境界検査・非 atomic 決定性・`long long` grid-stride 添字の
    /// 証跡（`kernels_mse.rs::tests` と同型の文字列検査）。
    #[test]
    fn nll_partial_f32_has_grid_stride_bound_check() {
        assert!(NLL_PARTIAL_F32.contains("idx < n_samples"));
        assert!(!NLL_PARTIAL_F32.contains("atomicAdd"));
        assert!(NLL_PARTIAL_F32.contains("long long stride = (long long)gridDim.x * blockDim.x;"));
        assert!(NLL_PARTIAL_F32.contains(
            "for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n_samples; idx += stride)"
        ));
    }

    #[test]
    fn nll_finalize_f32_has_bound_check() {
        assert!(NLL_FINALIZE_F32.contains("idx < num_partials"));
        assert!(!NLL_FINALIZE_F32.contains("atomicAdd"));
    }

    #[test]
    fn nll_backward_f32_has_bound_check() {
        assert!(NLL_BACKWARD_F32.contains("if (idx < n_samples)"));
        assert!(!NLL_BACKWARD_F32.contains("atomicAdd"));
    }

    /// PR #1850 codex-review P0 是正 2 の証跡: `t`（`targets[idx]`）の
    /// 手動境界検査（`0 <= t < num_classes`）がカーネルソース自身に
    /// 含まれることを確認する（`nll_partial_f32_has_grid_stride_bound_
    /// check`／`nll_backward_f32_has_bound_check` と同型の文字列検査）。
    #[test]
    fn nll_partial_f32_has_target_bound_check() {
        assert!(NLL_PARTIAL_F32.contains("if (t < 0 || t >= num_classes)"));
    }

    #[test]
    fn nll_backward_f32_has_target_bound_check() {
        assert!(NLL_BACKWARD_F32.contains("if (t >= 0 && t < num_classes)"));
    }
}
