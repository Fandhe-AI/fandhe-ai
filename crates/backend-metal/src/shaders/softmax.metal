// online softmax カーネル（イシュー #604）。CUDA 側の G-7（#594・OPEN）に
// 先行する Metal 実装であり、CUDA 直接の parity 相手はまだ存在しない
// （`softmax.rs` ドキュメンテーションコメント「#594 の gap」参照。両
// バックエンドとも CPU 参照実装〈REQ-2 統一複合判定〉を経由した推移的な
// 数値担保に留まる）。
//
// 意味論: softmax(x)_i = e^(x_i - max(x)) / sum_j e^(x_j - max(x))
// （行ごと）。実装は MFA の softmax 設計（`log2(e)` スケール + `exp2`・
// オンライン最大値更新・補正係数スキップ・範囲外レーンの明示除外）を
// 踏襲する。
//
// `log2(e)` のスケーリングは最大値減算の**後**に適用する（x ドメインで
// online max を追跡してから `(x - m) * log2(e)` を `exp2` に渡す）:
//
//   m_new = max(m, chunk_max(x))             （オンライン最大値更新。x ドメイン）
//   correction = (m_new > m) ? exp2((m - m_new) * log2(e)) : 1.0
//   l = l * correction + sum(valid ? exp2((x - m_new) * log2(e)) : 0.0)
//
// x を先に `log2(e)` 倍してから online max を取る構成（MFA 原設計の素朴な
// 適用）は、有限だが巨大な x（例: `f32::MAX` 付近）で `x * log2(e)` 自体が
// `+inf` へオーバーフローし、後続の `exp2(inf - inf) = NaN` を生む
// （イシュー #604 レビュー指摘・PR #714）。最大値減算を先に行うことで
// `x - m_new` は常に `<= 0` に収まり、スケール後の `exp2` 入力も
// オーバーフローしない。
//
// 範囲外レーンの扱い（PR #714 レビュー是正。旧版は「有限負値マージン
// sentinel」を x にも用いて sum 側の寄与も暗黙に潰す設計だったが、入力が
// `-f32::MAX` 付近の有限値の場合に sentinel と実データが数値的に拮抗し、
// 範囲外レーンの寄与が sum に混入して不正な結果になりうる欠陥があった）:
//
//   - online max の追跡には `SOFTMAX_NEG_FLT_MAX`（== `f32::MIN`。IEEE 754
//     単精度の最小有限値そのもので、マージン不要かつどんな有限入力より
//     真に小さいか等しい）を範囲外レーンの sentinel として使う。
//   - sum への寄与は `valid` フラグで明示的にゲートする
//     （`p = valid ? exp2(...) : 0.0`）。sentinel の大小関係に一切依存
//     しないため、実データが `SOFTMAX_NEG_FLT_MAX` 付近であっても sum が
//     汚染されない。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）固定。persistent threadgroup
// 方式・`grid_size` 引数・reduction 5 段 butterfly は `rmsnorm.metal` と
// 同一構造（同ファイル冒頭コメント参照）。
//
// 1 パス経路（`softmax_f32_onepass`）は最初の走査で threadgroup memory に
// `x`（範囲内レーンの生値のみ。`log2(e)` 未適用）を常駐しつつ online
// max/sum を蓄積し、確定した `m`・`l` で正規化して書き出す。2 パス経路
// （`softmax_f32_twopass`）は device メモリを再読する（`row_kernel::
// select_route` がホスト側で判定。`RMSNORM_ONEPASS_MAX_HIDDEN` と同じ
// 固定長 `SOFTMAX_ONEPASS_MAX_HIDDEN` を使う）。
//
// REQ-8: ループ添字は `row_base` を `ulong` で宣言し乗算オーバーフローを
// 避ける。境界外レーン（`idx >= hidden`）は `valid` フラグで分岐して除外
// する（範囲外メモリアクセス自体は行わない）。

#include <metal_stdlib>
using namespace metal;

constant uint SOFTMAX_ONEPASS_MAX_HIDDEN = 4096u;
constant uint SOFTMAX_SIMD_WIDTH = 32u;
constant float SOFTMAX_LOG2E = 1.4426950408889634f;
// `row_kernel::SOFTMAX_NEG_FLT_MAX`（Rust 側。`f32::MIN` の単一の真実源）
// と同じ値を MSL リテラルとして再現する（x ドメイン。`log2(e)` 適用前・
// online max 追跡の初期値／範囲外レーンの sentinel 専用。sum への寄与は
// `valid` フラグで別途ゲートするためマージンは不要。ファイル冒頭コメント
// 参照）。
constant float SOFTMAX_NEG_FLT_MAX = -3.402823466e+38f;

// 32 レーン全体の最大値を 5 段 butterfly で reduction する。
inline float softmax_reduce_max(float v) {
    v = max(v, simd_shuffle_xor(v, 16u));
    v = max(v, simd_shuffle_xor(v, 8u));
    v = max(v, simd_shuffle_xor(v, 4u));
    v = max(v, simd_shuffle_xor(v, 2u));
    v = max(v, simd_shuffle_xor(v, 1u));
    return v;
}

// 32 レーン全体の総和を 5 段 butterfly で reduction する。
inline float softmax_reduce_sum(float v) {
    v += simd_shuffle_xor(v, 16u);
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

kernel void softmax_f32_onepass(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& grid_size [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float smem[SOFTMAX_ONEPASS_MAX_HIDDEN];

    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;

        float m = SOFTMAX_NEG_FLT_MAX;
        float l = 0.0f;

        // --- パス 1: online max/sum を蓄積しつつ範囲内の `x` を smem へ常駐 ---
        for (uint chunk_start = 0u; chunk_start < hidden; chunk_start += SOFTMAX_SIMD_WIDTH) {
            uint idx = chunk_start + lane;
            bool valid = idx < hidden;
            // `chunk_start < hidden`（ループ条件）より lane 0 は必ず
            // valid のため、チャンク全体が invalid になることはない
            // （chunk_max が sentinel のみで決まる縮退ケースを排除）。
            float xv = valid ? x[row_base + idx] : SOFTMAX_NEG_FLT_MAX;
            if (valid) {
                smem[idx] = xv;
            }

            float chunk_max = softmax_reduce_max(xv);
            float m_new = max(m, chunk_max);
            // `m_new - m >= 0` かつ `xv - m_new <= 0` が常に成り立つため
            // （x ドメインで先に最大値減算してから `log2(e)` を適用する
            // 構成。ファイル冒頭コメント参照）、以下の `exp2` 入力は
            // 有限入力に対しオーバーフローしない。
            float correction = (m_new > m) ? exp2((m - m_new) * SOFTMAX_LOG2E) : 1.0f;
            // 範囲外レーンの寄与は sentinel の大小関係に依存せず 0.0 で
            // 明示的にゲートする（ファイル冒頭コメント参照）。
            float p = valid ? exp2((xv - m_new) * SOFTMAX_LOG2E) : 0.0f;
            float chunk_sum = softmax_reduce_sum(p);
            l = l * correction + chunk_sum;
            m = m_new;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // --- パス 2: 確定した m・l で正規化して書き出す（smem 再利用） ---
        // smem に書き込まれているのは valid だったレーンのみ（idx < hidden
        // で常に valid）。
        for (uint idx = lane; idx < hidden; idx += SOFTMAX_SIMD_WIDTH) {
            float xv = smem[idx];
            float p = exp2((xv - m) * SOFTMAX_LOG2E);
            out[row_base + idx] = p / l;
        }
    }
}

kernel void softmax_f32_twopass(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& grid_size [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    for (uint row = tg_id; row < rows; row += grid_size) {
        ulong row_base = (ulong)row * (ulong)hidden;

        float m = SOFTMAX_NEG_FLT_MAX;
        float l = 0.0f;

        // --- パス 1: online max/sum を蓄積（smem 不使用） ---
        for (uint chunk_start = 0u; chunk_start < hidden; chunk_start += SOFTMAX_SIMD_WIDTH) {
            uint idx = chunk_start + lane;
            bool valid = idx < hidden;
            float xv = valid ? x[row_base + idx] : SOFTMAX_NEG_FLT_MAX;

            float chunk_max = softmax_reduce_max(xv);
            float m_new = max(m, chunk_max);
            // onepass と同じオーバーフロー回避・範囲外レーン除外構成
            // （ファイル冒頭コメント参照）。
            float correction = (m_new > m) ? exp2((m - m_new) * SOFTMAX_LOG2E) : 1.0f;
            float p = valid ? exp2((xv - m_new) * SOFTMAX_LOG2E) : 0.0f;
            float chunk_sum = softmax_reduce_sum(p);
            l = l * correction + chunk_sum;
            m = m_new;
        }

        // --- パス 2: device メモリを再読して正規化して書き出す ---
        for (uint idx = lane; idx < hidden; idx += SOFTMAX_SIMD_WIDTH) {
            float xv = x[row_base + idx];
            float p = exp2((xv - m) * SOFTMAX_LOG2E);
            out[row_base + idx] = p / l;
        }
    }
}
