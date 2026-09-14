// Huber／SmoothL1 損失融合カーネルの MSL ソース（イシュー #1739。CUDA 側
// `backend-cuda::kernels_huber`〈同イシュー〉の Metal 対応版。`shaders/
// mse.metal`〈イシュー #1045〉と同じ 2 段 reduction 構成）。
//
// `crate::huber` が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（Safe/Precise。`crate::pipeline::compile_options`。
// `shaders/mse.metal` と同一設定）で実行時コンパイルする。
//
// # forward の 2 段構成（`huber_partial_f32` → `huber_finalize_f32`）
//
// `mse.metal` 冒頭コメントと同じ 2 段 reduction 構成を、区分損失
// `l(pred−target)`（`kind`：0=Huber・1=SmoothL1、`delta` 引数で分岐）へ
// 拡張する。`huber_finalize_f32` は `mse_finalize_f32` と数式的に同一
// （総和 + スカラー乗算）だが、モジュール独立性を優先し本ファイル内に
// 複製する（`kernels_huber.rs` 冒頭コメントと同じ判断）。
//
// # simdgroup 内総和（`simd_sum`）+ threadgroup 間結合
//
// `mse.metal` 冒頭コメントと同一の設計判断（`HUBER_SIMDGROUPS_PER_TG=8`・
// simdgroup 代表値の固定順序逐次結合）をそのまま踏襲する。
//
// # REQ-8（カーネル境界検査規約）
//
// `huber_partial_f32`・`huber_backward_f32` は `idx < numel` の手動境界
// チェックを維持する。`huber_finalize_f32` も `idx < num_partials` を
// 維持する。
//
// # sign(d) の統一（`copysign`）
//
// `sign(d)`（`d = pred − target`）は 3 バックエンドとも `copysign` で
// 統一する（`.claude/rules/coding-rust.md` の丸め方針統一と同じ理由。
// 実装計画 §2.1）。Metal は組み込み `copysign` を使う。
//
// # 意味論の正
//
// `fandhe_ai_autodiff::eval::huber_elem_loss`／`huber_elem_grad` が
// 意味論の正。累積順序は CPU 側と異なるため、バックエンド間の数値突合
// は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で検証
// する（`.claude/rules/coding-rust.md`）。

#include <metal_stdlib>
using namespace metal;

// simdgroup 数（`HUBER_THREADGROUP_WIDTH / 32`。Rust 側 `huber.rs::
// HUBER_THREADGROUP_WIDTH` と同期させる固定値）。
constant uint HUBER_SIMDGROUPS_PER_TG = 8u;

/// 要素損失 `l(d)`（`d = pred − target`）。`kind == 1` を `SmoothL1`、
/// それ以外を `Huber` として扱う（ホスト側 `huber.rs` が `HuberKind` から
/// `uint` へ変換する契約。`#[non_exhaustive]` な `HuberKind` の未知
/// variant はホスト側で `Huber`〈`kind == 0`〉へ安全側フォールバックする
/// ——`eval::huber_elem_loss` と同型の判断）。
inline float huber_elem_loss(float d, uint kind, float delta) {
    float abs_d = fabs(d);
    if (kind == 1u) {
        // SmoothL1
        if (abs_d < delta) {
            // d*d を先に計算すると delta・d が巨大な有限値のとき
            // 中間積が overflow しうる。abs_d < delta 分岐内では
            // |d/delta| < 1 が保証されるため、先に delta で割って
            // から d を掛けることで中間値を |d| 以下に抑える
            // （backend-cpu::huber::elem_loss・autodiff::eval::
            // huber_elem_loss・CUDA kernels_huber.rs と同じ演算順序
            // で揃える）。
            return 0.5f * (d / delta) * d;
        }
        return abs_d - 0.5f * delta;
    }
    // Huber（既定・未知 kind の安全側フォールバック）
    if (abs_d < delta) {
        return 0.5f * d * d;
    }
    return delta * (abs_d - 0.5f * delta);
}

/// [`huber_elem_loss`] の `pred` に対する要素勾配。
inline float huber_elem_grad(float d, uint kind, float delta) {
    // d が NaN（pred／target のいずれかが NaN）のとき、abs_d < delta は
    // NaN 比較の規約により常に false となり else 側（copysign 系）へ
    // 落ちて有限な勾配（±1／±delta）を返してしまう。forward
    // （huber_elem_loss）は同じ分岐構造でも else 側の結果が
    // NaN - 0.5*delta = NaN となり自然に NaN を返すため、forward と
    // backward で NaN 伝播の有無が食い違う（イシュー #1739 レビュー
    // 指摘）。ここで明示的に NaN を伝播する（backend-cpu::huber::
    // elem_grad・autodiff::eval::huber_elem_grad・CUDA
    // kernels_huber.rs と同じ方針で揃える）。
    if (isnan(d)) {
        return d;
    }
    float abs_d = fabs(d);
    if (kind == 1u) {
        // SmoothL1
        if (abs_d < delta) {
            return d / delta;
        }
        return copysign(1.0f, d);
    }
    // Huber（既定・未知 kind の安全側フォールバック）
    if (abs_d < delta) {
        return d;
    }
    return copysign(delta, d);
}

/// forward 1 段目: 各 threadgroup が担当区間の
/// `Σ l(pred[i]−target[i])` を計算し `partial[tg_id]` へ書く。
///
/// `partial` の長さは呼び出し元が起動 threadgroup 数と一致させて確保
/// する契約（`mse.metal` 冒頭コメントと同型）。`numel`（`uint`）は
/// `pred`／`target` の要素数。
kernel void huber_partial_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& kind [[buffer(4)]],
    constant float& delta [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threadgroups_per_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[HUBER_SIMDGROUPS_PER_TG];

    float acc = 0.0f;
    // REQ-8: `mse.metal::mse_partial_f32` と同じ理由で `ulong` 添字を
    // 用いる（`numel` 近傍での unsigned wraparound 回避）。
    ulong stride = (ulong)grid_size * (ulong)tg_size;
    for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride) {
        acc += huber_elem_loss(pred[idx] - target[idx], kind, delta);
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < HUBER_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        partial[tg_id] = block_sum;
    }
}

/// forward 2 段目: `partial`（`num_partials` 要素。1 threadgroup のみで
/// 起動）を総和し `factor` を乗じて `out[0]` へ書く（`mse.metal::
/// mse_finalize_f32` と数式的に同一だが本ファイル専用に複製。冒頭
/// コメント参照）。
kernel void huber_finalize_f32(
    device const float* partial [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    constant float& factor [[buffer(3)]],
    uint tg_size [[threads_per_threadgroup]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[HUBER_SIMDGROUPS_PER_TG];

    float acc = 0.0f;
    for (uint idx = tid; idx < num_partials; idx += tg_size) {
        acc += partial[idx];
    }

    float simd_total = simd_sum(acc);
    if (lane == 0) {
        simd_sums[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0 && lane == 0) {
        float block_sum = 0.0f;
        for (uint i = 0; i < HUBER_SIMDGROUPS_PER_TG; ++i) {
            block_sum += simd_sums[i];
        }
        out[0] = block_sum * factor;
    }
}

/// backward: `dPred[i] = scale·grad_elem(pred[i]−target[i])`（1 スレッド
/// 1 要素。`mse.metal::mse_backward_f32` と同じ 1 次元グリッド・境界
/// 検査）。`dTarget = −dPred` はホスト側が計算する契約
/// （`backend_ops.rs::BackendOps::huber_loss_backward` doc 参照）。
kernel void huber_backward_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* dpred [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& kind [[buffer(4)]],
    constant float& delta [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < numel) {
        dpred[idx] = scale * huber_elem_grad(pred[idx] - target[idx], kind, delta);
    }
}
