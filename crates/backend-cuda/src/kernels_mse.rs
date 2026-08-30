//! 平均二乗誤差（MSE）融合カーネルの CUDA C カーネルソース（NVRTC 実行時
//! コンパイル用の静的文字列。イシュー #1045・親イシュー #1043）。
//!
//! `mse.rs`（呼び出し元）は本モジュールの 3 定数を `nvrtc::compile_ptx` に
//! 渡し `CudaFunction` を得る。`kernels_elementwise.rs`／`kernels_sgd.rs`
//! と同じ理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む
//! （ビルド時に nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit 非搭載
//! 環境でも `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # forward の 2 段構成（`mse_partial_f32` → `mse_finalize_f32`）
//!
//! `docs/kernel-fusion.md` 限界表が「reduction 融合はバックエンド実行
//! レベルで未実装」と記録していた対象のうち、MSE の `Σ(pred−target)²`
//! 全要素縮約を 1 ブロック 1 スレッドの単純カーネルではなく古典的な
//! 2 段 reduction で実装する:
//!
//! 1. **`mse_partial_f32`**: grid-stride ループで各ブロックが担当区間の
//!    `Σ(pred−target)²` を計算し、ブロックごとの部分和 1 個を
//!    `partial[blockIdx.x]` へ書く（起動ブロック数は `mse.rs` が
//!    `min(ceil_div(numel, MSE_BLOCK_DIM), MSE_MAX_BLOCKS)` で決定し、
//!    `partial` バッファの長さと必ず一致させる契約。ホスト側の起動
//!    グリッド数と `partial` 確保長がずれると `mse_finalize_f32` 側で
//!    無検査の境界外読み出しになるため、この一致はカーネル内では検査
//!    できない呼び出し元契約とする）。
//! 2. **`mse_finalize_f32`**: 1 ブロックのみを起動し、`partial`
//!    （高々 `MSE_MAX_BLOCKS` 要素）を再度同じ butterfly reduction で
//!    総和したのち `factor`（`Mean` は `1/n`、`Sum` は `1.0`。呼び出し元
//!    がホスト側で決定）を乗じて `out[0]` へ書く。
//!
//! # 決定性（float atomicAdd を使わない理由）
//!
//! いずれの段も **`atomicAdd` を使わない**。ブロック間の結合順序は
//! `blockIdx.x` 昇順（`partial` への書き込み添字）で固定され、ブロック内
//! butterfly も `__shfl_xor_sync` の固定 offset 列（16→1）で決定的なため、
//! 同一入力・同一 grid 構成であれば bit 決定的に同じ結果を返す
//! （`backend-cpu::mse::mse_sum_sq_f32` の固定チャンク決定性契約と同種の
//! 設計判断。`.claude/rules/coding-rust.md` の再現性要件）。`atomicAdd`
//! はスケジューリング順に依存する非決定的な浮動小数点加算となり、
//! バックエンド間 parity テスト（`fandhe_ai_backend_cpu::parity::
//! assert_parity`）の再現性を損なうため採用しない。
//!
//! # warp 内 butterfly（32 lane 幅で `MSE_BLOCK_DIM/32` warp 分の 0 埋め）
//!
//! `MSE_BLOCK_DIM = 256`（8 warp）に対し、各 warp の代表値を
//! `__shared__ float warp_sums[8]` へ格納したのち、warp 0 の 32 lane が
//! `lane < 8 ? warp_sums[lane] : 0.0f` を初期値として **フルマスク**
//! （`0xffffffff`）の 5 段 butterfly（offset 16→1）を行う。lane 8〜31 は
//! 初期値 0.0 のため総和に寄与せず、実質的に 8 要素分の総和が全 32 lane
//! に複製される（`kernels_softmax.rs`／`kernels_rmsnorm.rs` の 5 段
//! butterfly と同じ「CUDA の線形レーンレイアウトでは 2 段シャッフルでは
//! 全 lane を結合しきれない」という理由の再利用。冒頭コメント参照）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `mse_partial_f32`・`mse_backward_f32` は `if (idx < numel)`
//! （grid-stride ループの場合は `for (...; idx < numel; ...)`）の手動
//! 境界チェックを維持する。`mse_finalize_f32` も `idx < num_partials` を
//! 維持する。ベクトル化ロード等の最適化は本イシューでは適用しない
//! （必要になった場合も境界チェックは維持する契約。
//! `.claude/rules/coding-rust.md`）。
//!
//! # 意味論の正
//!
//! `backend-cpu::mse`（`crates/backend-cpu/src/mse.rs`）が意味論の正。
//! `diff*diff` の累積に `fmaf`（単精度 FMA）を用いる点は CPU 側
//! `f32::mul_add` と同じ FMA 契約統一方針（`.claude/rules/coding-rust.md`）
//! だが、累積順序（CPU: 固定チャンク逐次 → チャンク間逐次結合／CUDA:
//! grid-stride → warp butterfly → ブロック間逐次結合）は異なるため、
//! バックエンド間の数値突合は統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」で検証する（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（1 次元、8 warp 分）。
/// `kernels_sgd::SGD_BLOCK_DIM`・`kernels_elementwise::EW_BLOCK_DIM` と
/// 同じ値・同じ理由（PoC 実測なしの保守的な固定値）。8 warp（256/32）は
/// 下記 `warp_sums[8]` の固定長と対応する。
pub const MSE_BLOCK_DIM: u32 = 256;

/// forward 2 段目（`mse_finalize_f32`）が単一ブロックで処理しきれる
/// `partial` の最大長（＝ forward 1 段目の起動ブロック数の上限）。
/// `mse.rs` が `min(ceil_div(numel, MSE_BLOCK_DIM), MSE_MAX_BLOCKS)` で
/// 実際の起動ブロック数を決定し、`partial` バッファをその長さで確保する
/// （冒頭コメント「forward の 2 段構成」参照）。
pub const MSE_MAX_BLOCKS: u32 = 1024;

/// forward 1 段目: 各ブロックが担当区間の `Σ(pred[i]−target[i])²` を
/// 計算し `partial[blockIdx.x]` へ書く。
///
/// `partial` の長さは呼び出し元がグリッド次元（`gridDim.x`）と一致させて
/// 確保する契約（本ファイル冒頭コメント参照）。`numel` は `pred`／
/// `target` の要素数（`int`。`mse.rs::validate_mse_len` が `i32::MAX` 範囲
/// 検査済み）。
pub const MSE_PARTIAL_F32: &str = r#"
extern "C" __global__ void mse_partial_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ partial,
    int numel)
{
    __shared__ float warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = 0.0f;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        float diff = pred[idx] - target[idx];
        acc = fmaf(diff, diff, acc);
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

/// forward 2 段目: `partial`（`num_partials` 要素。1 ブロックのみで起動）
/// を総和し `factor` を乗じて `out[0]` へ書く。`factor` は `Mean` なら
/// `1.0 / n`、`Sum` なら `1.0`（ホスト側 `mse.rs` が決定する）。
pub const MSE_FINALIZE_F32: &str = r#"
extern "C" __global__ void mse_finalize_f32(
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

/// backward: `dPred[i] = scale·(pred[i]−target[i])`（1 スレッド 1 要素。
/// `elementwise.rs` の単項カーネルと同じ 1 次元 grid・境界検査）。
/// `dTarget = −dPred` はホスト側（`mse.rs`）が呼び出し元へ計算させない
/// 契約（`backend_ops.rs::BackendOps::mse_loss_backward` doc 参照。CUDA
/// カーネルは `dPred` のみを出力し、追加のカーネル起動・デバイス確保・
/// D2H を発生させない）。
pub const MSE_BACKWARD_F32: &str = r#"
extern "C" __global__ void mse_backward_f32(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ dpred,
    int numel,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dpred[idx] = scale * (pred[idx] - target[idx]);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 境界検査・非 atomic 決定性の証跡（`backend-metal` の
    /// `mse_source_evidence.rs` と対になる CUDA 側の文字列検査。ソース
    /// が実際に NVRTC でコンパイル可能かは実機依存だが、境界検査・
    /// atomicAdd 不使用は文字列検査で機械的に固定できる）。
    #[test]
    fn mse_partial_f32_has_grid_stride_bound_check() {
        assert!(MSE_PARTIAL_F32.contains("idx < numel"));
        assert!(!MSE_PARTIAL_F32.contains("atomicAdd"));
    }

    /// grid-stride ループの添字（`idx`／`stride`）が `long long` で
    /// 宣言されていることをソース文字列に対して検査する
    /// （`kernels_softmax.rs::tests::
    /// onepass_and_twopass_loop_indices_are_declared_long_long` と同じ
    /// 理由・同じ検査パターン）。`numel` は `i32::MAX` まで許容する
    /// （`mse.rs::validate_mse_len`）ため、`int` 添字のまま `idx += stride`
    /// を続けると `numel` 近傍で signed overflow（UB）により `idx <
    /// numel` の境界チェックを迂回しうる（Bugbot 指摘）。
    #[test]
    fn mse_partial_f32_grid_stride_loop_index_is_declared_long_long() {
        assert!(
            MSE_PARTIAL_F32.contains("long long stride = (long long)gridDim.x * blockDim.x;"),
            "stride が long long で宣言されていない"
        );
        assert!(
            MSE_PARTIAL_F32.contains(
                "for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride)"
            ),
            "idx ループ添字が long long で宣言されていない"
        );
        assert!(
            !MSE_PARTIAL_F32.contains("for (int idx = blockIdx.x * blockDim.x + threadIdx.x"),
            "grid-stride ループ添字が int へ縮退している"
        );
    }

    #[test]
    fn mse_finalize_f32_has_bound_check() {
        assert!(MSE_FINALIZE_F32.contains("idx < num_partials"));
        assert!(!MSE_FINALIZE_F32.contains("atomicAdd"));
    }

    #[test]
    fn mse_backward_f32_has_bound_check() {
        assert!(MSE_BACKWARD_F32.contains("if (idx < numel)"));
        assert!(!MSE_BACKWARD_F32.contains("atomicAdd"));
    }

    #[test]
    fn mse_kernels_use_fma_contract() {
        // `fmaf` は forward の 2 乗和累積のみ（backward は単純な乗減算
        // で FMA を要さない。`mse.rs::mse_loss_backward_f32` の
        // `scale * (p - t)` と同じ意味論）。
        assert!(MSE_PARTIAL_F32.contains("fmaf"));
    }
}
