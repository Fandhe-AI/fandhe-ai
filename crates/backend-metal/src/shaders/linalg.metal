// 線形代数（LU 分解・LU 代入・cholesky・Householder QR）の f32 自作カーネル
// （イシュー #2676・親イシュー #2672・設計判断記録
// `docs/backend-gpu-linalg-decision.md` §3）。`crates/backend-cuda/src/
// kernels_linalg.rs`／`kernels_linalg_factor.rs`（double 版）の Metal 対応版。
//
// `crate::linalg_kernel::MetalLinalg`（`linalg_kernel.rs`）から実行時コンパイル
// され、`ops.rs::MetalBackendOps::{linalg_inv, linalg_solve, linalg_det,
// linalg_cholesky, linalg_qr}` の内部 opt-in ゲート（既定 OFF）の背後から
// 呼ばれる（Stage B・イシュー #3137 で起動を結線済み。`linalg_kernel.rs` 冒頭参照）。
// ホスト逐語モデルは `crate::linalg_model`（演算順序を 1 対 1 で対応させる）。
//
// ---- 数値方式 ----
//
// MSL は `double` 非対応のためネイティブ f32 で計算する。丸め契約:
// - `#pragma clang fp contract(off)`（ファイルスコープ）: `compile_options()`
//   の MathMode::Safe だけでは FMA 縮約が off にならないため明示する
//   （`adam.metal` と同じ判断）。`fma(` は使わない。
// - 除算は `precise::divide`、平方根は `precise::sqrt`。
// - 各出力要素の算術順序は CPU 参照実装の逐次ループ（k 昇順）と同一で、
//   並列化は要素間のみ。単一の threadgroup が 1 行列を処理する。
// - QR の Householder ノルムは scale/ssq 方式（LAPACK SLASSQ 系）。二乗を直接
//   作らないため有限入力で overflow しない。NaN／inf は明示的に伝播させる。
//
// ---- 同期規律 ----
//
// 1 threadgroup（grid の threadgroup_position_in_grid != 0 は何もしない）。
// 手順間は `threadgroup_barrier(mem_device | mem_threadgroup)` で区切る。
// 特異・非正定値などの分岐は `threadgroup` 変数へ書いてバリア後に全スレッドが
// 読み、一様に抜ける（早期 return でバリアを不一致にしない）。
//
// ---- REQ-8（カーネル境界検査規約）----
//
// 全添字アクセスは `< n`／`< k`／`< m` 等の手動境界チェックに従うループ範囲で
// 行う（最適化を理由に省略しない）。

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

// ---------------------------------------------------------------------------
// LAPACK SLASSQ 系 overflow-safe 2-ノルム。`crate::linalg::scale_ssq_norm_f32`
// と演算列が同一（scale=0・ssq=1 から添字昇順）。`p[i * stride]`（i < len）を読む。
// ---------------------------------------------------------------------------
static inline float scale_ssq_norm(device const float* p, uint stride, uint len, float div) {
    float scale = 0.0f;
    float ssq = 1.0f;
    bool has_inf = false;
    for (uint i = 0; i < len; i++) {
        float x = precise::divide(p[i * stride], div);
        if (isnan(x)) {
            return NAN;
        }
        if (isinf(x)) {
            has_inf = true;
            continue;
        }
        float a = fabs(x);
        if (a != 0.0f) {
            if (scale < a) {
                float r = precise::divide(scale, a);
                ssq = 1.0f + ssq * (r * r);
                scale = a;
            } else {
                float r = precise::divide(a, scale);
                ssq = ssq + (r * r);
            }
        }
    }
    if (has_inf) {
        return INFINITY;
    }
    return scale * precise::sqrt(ssq);
}

// ---------------------------------------------------------------------------
// 部分ピボット LU（in-place）。a: n*n（行優先・入出力）、perm: n（出力）、
// status: [0]=特異フラグ（ホストが 0 で初期化）、[1]=行入替回数（同）。
// ---------------------------------------------------------------------------
kernel void linalg_lu_f32(
    device float* a [[buffer(0)]],
    device int* perm [[buffer(1)]],
    device int* status [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_pos [[threadgroup_position_in_grid]])
{
    if (tg_pos != 0) {
        return;
    }
    threadgroup uint max_row_tg;

    for (uint i = tid; i < n; i += tg_size) {
        perm[i] = (int)i;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint k = 0; k < n; k++) {
        // ピボット探索は thread 0 が逐次に行う（厳密 `>` で同値は最小添字・
        // NaN は選ばれない。CPU 参照実装と同じ意味論）。
        if (tid == 0) {
            uint max_row = k;
            float max_val = fabs(a[k * n + k]);
            for (uint i = k + 1; i < n; i++) {
                float v = fabs(a[i * n + k]);
                if (v > max_val) {
                    max_val = v;
                    max_row = i;
                }
            }
            max_row_tg = max_row;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        uint max_row = max_row_tg;
        if (max_row != k) {
            for (uint c = tid; c < n; c += tg_size) {
                float tmp = a[k * n + c];
                a[k * n + c] = a[max_row * n + c];
                a[max_row * n + c] = tmp;
            }
            if (tid == 0) {
                int pk = perm[k];
                perm[k] = perm[max_row];
                perm[max_row] = pk;
                status[1] = status[1] + 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        float pivot = a[k * n + k];
        if (pivot == 0.0f) {
            // 全スレッドが同じ値を読んで一様に抜ける。
            if (tid == 0) {
                status[0] = 1;
            }
            return;
        }

        // フェーズ 1: 乗数（i > k の行ごと独立）。
        for (uint i = k + 1 + tid; i < n; i += tg_size) {
            a[i * n + k] = precise::divide(a[i * n + k], pivot);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // フェーズ 2: 更新（i > k, c > k の要素ごと独立。乗数 0 の行は飛ばす）。
        uint w = n - k - 1;
        uint total = w * w;
        for (uint idx = tid; idx < total; idx += tg_size) {
            uint i = k + 1 + idx / w;
            uint c = k + 1 + idx % w;
            float factor = a[i * n + k];
            if (factor != 0.0f) {
                a[i * n + c] = a[i * n + c] - factor * a[k * n + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// LU 代入。lu/perm/status は linalg_lu_f32 の出力、b: n*k、x: n*k（出力・
// 作業域を兼ねる）。1 スレッドが右辺 1 列を担当する（列間は独立）。特異なら
// 何も書かない。inv はホストが単位行列を b として渡す。
// ---------------------------------------------------------------------------
kernel void linalg_lu_solve_f32(
    device const float* lu [[buffer(0)]],
    device const int* perm [[buffer(1)]],
    device const int* status [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device float* x [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= k) {
        return;
    }
    if (status[0] != 0) {
        return;
    }
    uint c = gid;
    // 前進代入（L の対角 1 が暗黙）。
    for (uint i = 0; i < n; i++) {
        uint src = (uint)perm[i];
        float sum = (src < n) ? b[src * k + c] : 0.0f;
        for (uint j = 0; j < i; j++) {
            sum = sum - lu[i * n + j] * x[j * k + c];
        }
        x[i * k + c] = sum;
    }
    // 後退代入。
    for (uint ii = n; ii > 0; ii--) {
        uint i = ii - 1;
        float sum = x[i * k + c];
        for (uint j = i + 1; j < n; j++) {
            sum = sum - lu[i * n + j] * x[j * k + c];
        }
        x[i * k + c] = precise::divide(sum, lu[i * n + i]);
    }
}

// ---------------------------------------------------------------------------
// cholesky（下三角のみ読む）。a: n*n、l: n*n（出力・上三角は 0）、
// status[0]: 非正定値（対角が非有限または非正）で 1（ホストが 0 で初期化）。
// ---------------------------------------------------------------------------
kernel void linalg_cholesky_f32(
    device const float* a [[buffer(0)]],
    device float* l [[buffer(1)]],
    device int* status [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_pos [[threadgroup_position_in_grid]])
{
    if (tg_pos != 0) {
        return;
    }
    threadgroup uint fail_tg;
    if (tid == 0) {
        fail_tg = 0;
    }
    uint total = n * n;
    for (uint idx = tid; idx < total; idx += tg_size) {
        l[idx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint j = 0; j < n; j++) {
        if (tid == 0) {
            float sum = a[j * n + j];
            for (uint k = 0; k < j; k++) {
                sum = sum - l[j * n + k] * l[j * n + k];
            }
            if (!isfinite(sum) || sum <= 0.0f) {
                fail_tg = 1;
            } else {
                l[j * n + j] = precise::sqrt(sum);
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (fail_tg != 0) {
            if (tid == 0) {
                status[0] = 1;
            }
            return;
        }
        float ljj = l[j * n + j];
        for (uint i = j + 1 + tid; i < n; i += tg_size) {
            float sum = a[i * n + j];
            for (uint k = 0; k < j; k++) {
                sum = sum - l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = precise::divide(sum, ljj);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// reduced Householder QR。r: m*n（入力 A を上書きして上三角化・入出力）、
// q: m*k（出力 reduced Q）、v: m*k（反射ベクトル保存域・列 col の v は
// v[col*m ..] に長さ m-col）、applied: k（反射を適用した列は 1・ホストが 0 で
// 初期化）。対角符号の正規化（R の行と Q の列の反転）はホスト後処理。
// ---------------------------------------------------------------------------
kernel void linalg_qr_f32(
    device float* r [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* v [[buffer(2)]],
    device int* applied [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_pos [[threadgroup_position_in_grid]])
{
    if (tg_pos != 0) {
        return;
    }
    threadgroup float norm_x_tg;
    threadgroup float norm_v_tg;
    threadgroup float amax_tg;
    threadgroup float norm_xs_tg;
    uint kk = min(m, n);

    for (uint col = 0; col < kk; col++) {
        uint len = m - col;
        // 直前の反復が読む threadgroup 変数を thread 0 が上書きしないための区切り。
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) {
            norm_x_tg = scale_ssq_norm(r + col * n + col, n, len, 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float norm_x = norm_x_tg;
        if (norm_x == 0.0f) {
            continue;
        }
        // 列を最大絶対値 amax で正規化してから反射ベクトルを形成する（xi - alpha の
        // overflow 回避。v は後段で正規化されるためスケール不変。ホストモデル
        // `linalg_model::qr_f32` と同一演算列）。
        if (tid == 0) {
            float amax = 0.0f;
            for (uint i = 0; i < len; i++) {
                amax = max(amax, fabs(r[(col + i) * n + col]));
            }
            amax_tg = (isfinite(amax) && amax > 0.0f) ? amax : 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float amax_s = amax_tg;
        if (tid == 0) {
            norm_xs_tg = scale_ssq_norm(r + col * n + col, n, len, amax_s);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float norm_xs = norm_xs_tg;
        float x0s = precise::divide(r[col * n + col], amax_s);
        float alpha = (x0s >= 0.0f) ? -norm_xs : norm_xs;
        for (uint i = tid; i < len; i += tg_size) {
            float xi = precise::divide(r[(col + i) * n + col], amax_s);
            v[col * m + i] = (i == 0) ? (xi - alpha) : xi;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) {
            norm_v_tg = scale_ssq_norm(v + col * m, 1, len, 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float norm_v = norm_v_tg;
        if (norm_v == 0.0f) {
            continue;
        }
        for (uint i = tid; i < len; i += tg_size) {
            v[col * m + i] = precise::divide(v[col * m + i], norm_v);
        }
        if (tid == 0) {
            applied[col] = 1;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // r ← H r（列 c ≥ col ごとに独立。dot は i 昇順の逐次和）。
        for (uint c = col + tid; c < n; c += tg_size) {
            float dot = 0.0f;
            for (uint i = 0; i < len; i++) {
                dot = dot + v[col * m + i] * r[(col + i) * n + c];
            }
            if (dot != 0.0f) {
                for (uint i = 0; i < len; i++) {
                    uint idx = (col + i) * n + c;
                    // 2*vi*dot の 1 項化は |r|≈2e38 で overflow するため vi*dot を 2 回引く。
                    float t = v[col * m + i] * dot;
                    r[idx] = (r[idx] - t) - t;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // reduced Q を E_k から出発し反射を逆順に適用する。Q の列 c は互いに独立の
    // ため 1 スレッドが列ごとに完結させる（列間のバリア不要）。
    for (uint c = tid; c < kk; c += tg_size) {
        for (uint row = 0; row < m; row++) {
            q[row * kk + c] = (row == c) ? 1.0f : 0.0f;
        }
        for (uint ci = kk; ci > 0; ci--) {
            uint col = ci - 1;
            if (applied[col] == 0) {
                continue;
            }
            uint len = m - col;
            float dot = 0.0f;
            for (uint i = 0; i < len; i++) {
                dot = dot + v[col * m + i] * q[(col + i) * kk + c];
            }
            if (dot != 0.0f) {
                for (uint i = 0; i < len; i++) {
                    uint idx = (col + i) * kk + c;
                    q[idx] = q[idx] - (2.0f * v[col * m + i]) * dot;
                }
            }
        }
    }
}
