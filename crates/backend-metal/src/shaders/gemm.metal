// GEMM カーネル 3 段（naive・tiled・simdgroup_matrix。TASK-1.8b・#39〜
// TASK-1.8c・#40）。
//
// 移植元: docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/shaders/gemm.metal
// の `gemm_naive`/`gemm_tiled`/`gemm_simdgroup`。naive 段は #39 で
// productize 済み。本ファイルは #40 で tiled・simdgroup を追加し 3 段
// すべてを揃えた。
//
// `crate::pipeline` が `include_str!` で本ファイルを取り込み、実行時に
// `MTLCompileOptions`（Safe/Precise。PoC-v2-5 実測構成）でコンパイルする。
// `crate::gemm` がバッファ・`Dims` を結線してディスパッチする
// （`GemmVariant::Simdgroup` の場合、`crate::pad` で 8 の倍数へパディング
// 済みの実効次元を `Dims` として渡す契約）。

#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// `crate::gemm::Dims`（`#[repr(C)]`）とレイアウトを一致させる（12 バイト）。
// `GemmVariant::Simdgroup` の場合、m/n/k は `crate::pad::pad8` によって
// 8 の倍数に切り上げられた実効次元（呼び出し元の元形状ではない）。
struct Dims {
    uint m;
    uint n;
    uint k;
};

// 素朴な 3 重ループ（タイル化なし）。gid.y = 行、gid.x = 列。
//
// 手動境界チェック（`gid.y >= dims.m || gid.x >= dims.n` で早期 return）は
// 性能上の下限・最適化の達成を理由に省略しない（REQ-8・
// `.claude/rules/coding-rust.md`「カーネル実装の境界検査」）。dispatch 側
// （`crate::gemm::gemm_naive`）は grid を `div_ceil(16)` で切り上げるため、
// m・n が threadgroup サイズ（16）の倍数でない場合にこの境界チェックが
// 実際に効く（はみ出したスレッドの書き込みを防ぐ）。
//
// 内積の丸め方針（FMA 契約。REQ-2）: CPU 参照実装
// （`backend_cpu::parity::matmul_reference_fma`）は `f32::mul_add`・k 昇順
// 逐次加算を用いる。ここでも `fma()` を明示し、コンパイラの自動 FMA 融合
// （かかる場合とかからない場合がある）に丸め方針を委ねない
// （PoC の `acc += a*b` から変更。PoC-v2-5 の K=4096 ストレスケースで
// mul_add 化により CPU/GPU 間 fail_cells=0 を実測確認済み）。
kernel void gemm_naive(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.y >= dims.m || gid.x >= dims.n) {
        return;
    }
    float acc = 0.0;
    for (uint p = 0; p < dims.k; p++) {
        acc = fma(a[gid.y * dims.k + p], b[p * dims.n + gid.x], acc);
    }
    c[gid.y * dims.n + gid.x] = acc;
}

// threadgroup 共有メモリによるタイル化（TASK-1.8c・#40）。WGSL 版
// （`docs/spec/.../gemm.wgsl`）と同一のタイルサイズ・アルゴリズムで、
// 「同じアルゴリズムを Metal 直接で書いた場合の差」を切り分けられるよう
// にしている（PoC-v2-4 README「計測結果」。size=4096 で naive 比
// 約 1.67 倍）。
//
// 手動境界チェック（REQ-8）: タイルロード時は `(row < m && a_col < k)` /
// `(b_row < k && col < n)` の条件を満たさない要素を 0 埋めし、ストア時は
// `row < m && col < n` を満たす場合のみ書き込む。m・n・k が TILE（16）の
// 倍数でない場合にこの境界チェックが実際に効く（タイル端のはみ出し
// アクセス・書き込みを防ぐ。性能上の下限・最適化を理由に省略しない）。
//
// 内積の丸め方針（FMA 契約。REQ-2）: naive 段と同じ理由で `acc += a*b`
// （PoC 原文）ではなく `fma()` を明示する（CPU 参照実装 `f32::mul_add`
// との丸め方針統一。PoC-v2-5 実測確認済み）。
//
// バッファオフセットの 64-bit 化（PR #246 Bugbot 指摘対応）: `row * k`・
// `b_row * n`・`row * n` は `dims.m`/`n`/`k` が個々に `u32::MAX` 未満でも
// 積が `u32::MAX` を超えうる（例: m=n=k=100000 は各次元は収まるが
// `row * n` は最大 約 1.0e10 で 32-bit `uint` 溢れによりオフセットが
// ラップアラウンドし、書き込み先行が黙って不正になる）。`row`/`a_col`/
// `b_row`/`col` を `size_t`（64-bit）へ昇格してから乗算し、`u32::MAX` 超の
// 有効な行列サイズでも溢れないようにする。
constant uint TILE = 16;

kernel void gemm_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float tile_a[TILE][TILE];
    threadgroup float tile_b[TILE][TILE];

    uint m = dims.m, n = dims.n, k = dims.k;
    uint row = gid.y, col = gid.x;
    float acc = 0.0;

    uint num_tiles = (k + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE + lid.x;
        uint b_row = t * TILE + lid.y;

        tile_a[lid.y][lid.x] =
            (row < m && a_col < k) ? a[(size_t)row * (size_t)k + (size_t)a_col] : 0.0;
        tile_b[lid.y][lid.x] =
            (b_row < k && col < n) ? b[(size_t)b_row * (size_t)n + (size_t)col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc = fma(tile_a[lid.y][i], tile_b[i][lid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        c[(size_t)row * (size_t)n + (size_t)col] = acc;
    }
}

// `simdgroup_matrix`（8x8 ハードウェア行列演算命令。TASK-1.8c・#40）。
//
// 1 threadgroup = 1 simdgroup（32 スレッド）とし、各 simdgroup が C の
// 8x8 タイルを 1 つ担当する。`simdgroup_load`/`simdgroup_multiply_
// accumulate`/`simdgroup_store` は Apple GPU の行列専用命令にディスパッチ
// され、WGSL では表現できない（`crate::lib` クレートコメント・PoC-v2-4
// README「経路選定の比較判断」節が objc2-metal 直接経路を選定した中心的
// 根拠）。命令自体が Apple GPU の既定 FMA 契約で累積するため、CPU 参照
// 実装（`f32::mul_add`）との丸め方針統一は本命令のハードウェア契約に
// 委ねる（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。naive/tiled
// 段のような `fma()` の明示呼び出しは不要）。
//
// **制約と手動境界チェック（REQ-8）**: `dims.m`/`n`/`k` は
// `crate::gemm::MetalGemm::dispatch_variant` が `crate::pad::pad8` で
// 8 の倍数に切り上げた実効次元であり、呼び出し元はこの契約を守って
// バッファ（A・B・C）も実効次元ぶん確保・0 パディング済みでなければ
// ならない。この前提が成立する限り 8x8 タイル内部の `simdgroup_load`/
// `_store` は常に確保領域内に収まるが、性能上の下限・最適化の達成を
// 理由に境界チェック自体を省略しない方針（REQ-8）に従い、タイル原点
// （`row0`/`col0`）が実効次元を超える場合の早期 return を明示的に残す
// （通常の dispatch では `n_eff/8 × m_eff/8` の grid により到達しない
// 防御的チェックだが、将来 dispatch 側の grid 計算に誤りがあっても
// 未確保領域への書き込みを起こさないための境界検査として維持する）。
//
// バッファオフセットの 64-bit 化（PR #246 Bugbot 指摘対応）: `row0 * dims.k`・
// `p0 * dims.n`・`row0 * dims.n` は `dims.m`/`n`/`k` が個々に `u32::MAX`
// 未満でも積が `u32::MAX` を超えうる（tiled 段と同じ懸念。上記コメント
// 参照）。`row0`/`col0`/`p0` を `size_t`（64-bit）へ昇格してから乗算し、
// `u32::MAX` 超の有効な行列サイズでもポインタオフセットが溢れないように
// する。
kernel void gemm_simdgroup(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row0 = tgid.y * 8;
    uint col0 = tgid.x * 8;

    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }

    simdgroup_float8x8 acc(0.0f);
    uint k_tiles = dims.k / 8;
    for (uint t = 0; t < k_tiles; t++) {
        uint p0 = t * 8;
        simdgroup_float8x8 a_tile;
        simdgroup_float8x8 b_tile;
        simdgroup_load(a_tile, a + (size_t)row0 * (size_t)dims.k + (size_t)p0, dims.k);
        simdgroup_load(b_tile, b + (size_t)p0 * (size_t)dims.n + (size_t)col0, dims.n);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }
    simdgroup_store(acc, c + (size_t)row0 * (size_t)dims.n + (size_t)col0, dims.n);
}

// f16 GEMM（TASK-8.3b・#156）: REQ-8「自作カーネルでの f16 実測」の実測対象
// カーネル。`gemm_simdgroup`（f32 版）と同じく 1 threadgroup = 1 simdgroup が
// C の 8x8 タイルを 1 つ担当する構成をそのまま half 型へ写し替える。
//
// **累算精度契約（実装計画 3.1 節の判断）**: A・B・C（アキュムレータ含む）
// すべて `simdgroup_half8x8`（half 型統一）とする。MSL 仕様
// （`simdgroup_matrix<T,Cols,Rows>`・`simdgroup_multiply_accumulate(d,a,b,c)`
// はいずれも単一の型パラメータ `T` に対するテンプレートであり、
// `apple-silicon` スキル `references/msl/data-types.md`・
// `simdgroup-functions.md` のいずれにも「A/B が half・C が float」という
// 混在型オーバーロードは記載がない（Linux 実装環境では Metal コンパイラで
// 実地検証もできない）。未確認のオーバーロードを推定で使うより、仕様上
// 確実に成立する単一型テンプレートを選ぶ（advisor 助言）。
//
// この選択は CUDA 側 WMMA f16（`kernels_wmma.rs::gemm_wmma_f16`。
// `f32.f16.f16.f32`。f32 累算）と精度契約が異なる点に注意
// （`docs/perf/metal-f16-vs-mps-f16.md`「精度契約」節に明記する）。half
// 累算は f32 累算より桁落ちしやすく、K が大きいストレスケース（K=4096 等）
// では複合判定（REQ-2）を外れる可能性が高い。実機で外れた場合も緩和せず
// 事実を記録し #158（下限確定）へ引き継ぐ（`.claude/rules/coding-rust.md`
// 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。
//
// `ACC_T` を単一の typedef にしておくことで、実機で
// `simdgroup_float8x8` 混在アキュムレータが実際に使えると判明した場合に
// この 1 行（と `c` バッファの型・`crate::gemm::dispatch_f16` の出力型）
// だけを変更すれば切替できるようにする（reversible な設計。advisor 助言）。
// 現状は half 統一のため `ACC_T == MM_T`。
typedef simdgroup_half8x8 MM_T; // A・B の simdgroup タイル型
typedef simdgroup_half8x8 ACC_T; // アキュムレータ型（現状 half 統一）

// **手動境界チェック（REQ-8）**: `gemm_simdgroup`（f32 版）と同じ契約。
// `dims.m`/`n`/`k` は `crate::gemm::MetalGemm::dispatch_f16` が
// `crate::pad::pad8` で 8 の倍数に切り上げた実効次元であり、呼び出し元が
// A・B・C バッファもその実効次元ぶん確保・0 パディング済みであることを
// 前提とする。タイル原点（`row0`/`col0`）が実効次元を超える場合の早期
// return は、通常の dispatch（`n_eff/8 × m_eff/8` の grid）では到達しない
// 防御的チェックだが、性能上の下限・最適化の達成を理由に境界チェック自体を
// 省略しない方針（REQ-8）に従い明示的に残す（`gemm_simdgroup` 冒頭コメント
// と同じ判断）。
//
// バッファオフセットの 64-bit 化: `gemm_simdgroup`（PR #246 Bugbot 指摘
// 対応）と同じ理由で `row0`/`col0`/`p0` を `size_t` へ昇格してから乗算する
// （`u32::MAX` 超の有効な行列サイズでもポインタオフセットが溢れないように
// する）。
kernel void gemm_simdgroup_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint row0 = tgid.y * 8;
    uint col0 = tgid.x * 8;

    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }

    ACC_T acc = ACC_T(static_cast<half>(0.0h));
    uint k_tiles = dims.k / 8;
    for (uint t = 0; t < k_tiles; t++) {
        uint p0 = t * 8;
        MM_T a_tile;
        MM_T b_tile;
        simdgroup_load(a_tile, a + (size_t)row0 * (size_t)dims.k + (size_t)p0, dims.k);
        simdgroup_load(b_tile, b + (size_t)p0 * (size_t)dims.n + (size_t)col0, dims.n);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }
    simdgroup_store(acc, c + (size_t)row0 * (size_t)dims.n + (size_t)col0, dims.n);
}

// 動的タイル選択（TASK-1.8f・#188）: BM/BN/BK/WM/WN パラメータ化 GEMM。
//
// `gemm_simdgroup`（1 threadgroup = 1 simdgroup = C の 8x8 タイル 1 つ）は
// タイルサイズの自由度がなく、MLX steel カーネル方式（BM/BN/BK/WM/WN の
// パラメータ化＋行列サイズ別動的選択）の性能差の核心に対応できない
// （イシュー #188 本文・`docs/spec/v2-amendment-proposal-2026-08-06.md`
// 改定 1 根拠）。本カーネルは 1 threadgroup が C の BM×BN ブロックを担当し、
// 内部を WM×WN 個の simdgroup（threadgroup スレッド数 = WM*WN*32）で分担
// する。各 simdgroup は (BM/WM)/8 × (BN/WN)/8 個の `simdgroup_float8x8`
// アキュムレータを持ち、K 方向を BK 刻みでループする。
//
// BM/BN/BK/WM/WN・協調ロード有無（USE_TGP_STAGING）は MSL function
// constant として与える（`crate::pipeline::make_pipeline_with_constants`
// が `MTLFunctionConstantValues` 経由でパイプライン構築時に定数畳み込み・
// ループ展開させる。実行時コンパイル構成のため MLX steel のテンプレート
// 実体化と同等の効果が得られる。イシュー #188 計画「設計方針」節）。
// 整除制約（BM は WM*8 の倍数、BN は WN*8 の倍数、BK は 8 の倍数）は
// Rust 側 `crate::tile::TileConfig::validate` が事前検証する契約であり、
// カーネル側では前提として扱う。
constant uint BM [[function_constant(0)]];
constant uint BN [[function_constant(1)]];
constant uint BK [[function_constant(2)]];
constant uint WM [[function_constant(3)]];
constant uint WN [[function_constant(4)]];
constant bool USE_TGP_STAGING [[function_constant(5)]];

// threadgroup 共有メモリは function constant でサイズ指定できないため、
// `threadgroup float*` 引数＋エンコード時 `setThreadgroupMemoryLength_
// atIndex`（`crate::gemm::encode_dispatch_tiled`）で渡す。A タイル
// （BM×BK、先頭から）＋ B タイル（BK×BN、A タイルの直後）を 1 領域へ
// オフセット分割する（`USE_TGP_STAGING=false` の場合は未使用でも
// エンコード側は 0 バイトのバッファを渡さず最小長で確保する契約
// （`crate::tile::TileConfig::shared_mem_bytes` が 0 を返す設計。実引数
// 自体は残しシグネチャを固定しておくことでパイプライン切替を単純化する）。
kernel void gemm_simdgroup_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // 1 threadgroup が担当する C ブロックの原点（行優先: y=行, x=列）。
    uint row0 = tgid.y * BM;
    uint col0 = tgid.x * BN;

    // ブロック全体が実効次元を完全に超える場合は早期 return する
    // （REQ-8。dispatch 側 grid は div_ceil(BM)/div_ceil(BN) で切り上げる
    // ため、末尾ブロックは部分的にしか実効次元へ収まらないケースが実際に
    // 発生する）。
    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }

    // この simdgroup が担当するブロック内サブ領域（WM×WN 分担）。
    uint wm_idx = simd_id / WN;
    uint wn_idx = simd_id % WN;
    uint sub_bm = BM / WM; // この simdgroup が担当する行幅
    uint sub_bn = BN / WN; // この simdgroup が担当する列幅
    uint sub_row0 = row0 + wm_idx * sub_bm;
    uint sub_col0 = col0 + wn_idx * sub_bn;

    // アキュムレータ数は最大 8x8 個（BM/BN 512 以下・WM/WN >=1 の実運用
    // 候補では十分な固定上限。`crate::tile::CANDIDATES` の暫定値は
    // 最大でも 4x4 個に収まる）。本カーネル自体は acc_rows/acc_cols を
    // 検査しないため、`(BM/WM)/8`・`(BN/WN)/8` が MAX_ACC を超える構成が
    // 渡されるとこのローカル配列への範囲外書き込みになる。安全性は
    // 呼び出し元の `crate::tile::TileConfig::validate`（`TileConfig::MAX_ACC`
    // 定数と 1:1 対応）が [`crate::gemm::MetalGemm::pipeline_for_tile`] の
    // パイプライン構築前に必ず検査し、超過構成を拒否してフォールバック
    // することで担保する契約（レビュー指摘。#188 PR review）。
    constexpr uint MAX_ACC = 8;
    uint acc_rows = sub_bm / 8;
    uint acc_cols = sub_bn / 8;
    simdgroup_float8x8 acc[MAX_ACC][MAX_ACC];
    for (uint r = 0; r < acc_rows; r++) {
        for (uint c_ = 0; c_ < acc_cols; c_++) {
            acc[r][c_] = simdgroup_float8x8(0.0f);
        }
    }

    // threadgroup 共有メモリ上の A タイル（BM×BK）・B タイル（BK×BN）
    // オフセット（`USE_TGP_STAGING=true` の場合のみ使用）。
    threadgroup float* tile_a = shared_mem;
    threadgroup float* tile_b = shared_mem + (size_t)BM * (size_t)BK;

    uint k_full_tiles = dims.k / BK;
    uint k_tail = dims.k - k_full_tiles * BK; // BK の倍数でない末尾（0 埋め扱い）
    uint k_tile_count = k_full_tiles + (k_tail > 0 ? 1 : 0);

    for (uint t = 0; t < k_tile_count; t++) {
        uint p0 = t * BK;
        // 末尾タイルが BK に満たない場合の有効幅（境界チェック。REQ-8）。
        uint bk_eff = min(BK, dims.k - p0);

        if (USE_TGP_STAGING) {
            // 協調ロード: threadgroup 内の全スレッド（WM*WN*32 個）で
            // A タイル（BM*BK 要素）・B タイル（BK*BN 要素）を分担して
            // 共有メモリへロードする。実効次元・K タイル端をはみ出す
            // 要素は 0 埋めする（最適化を理由に境界チェックを省略しない。
            // REQ-8）。
            uint local_tid = simd_id * 32 + simd_lane;
            uint threads_total = WM * WN * 32;

            uint a_elems = BM * BK;
            for (uint idx = local_tid; idx < a_elems; idx += threads_total) {
                uint r = idx / BK;
                uint kk = idx % BK;
                uint global_row = row0 + r;
                uint global_k = p0 + kk;
                tile_a[idx] = (kk < bk_eff && global_row < dims.m && global_k < dims.k)
                    ? a[(size_t)global_row * (size_t)dims.k + (size_t)global_k]
                    : 0.0f;
            }

            uint b_elems = BK * BN;
            for (uint idx = local_tid; idx < b_elems; idx += threads_total) {
                uint kk = idx / BN;
                uint c_ = idx % BN;
                uint global_k = p0 + kk;
                uint global_col = col0 + c_;
                tile_b[idx] = (kk < bk_eff && global_k < dims.k && global_col < dims.n)
                    ? b[(size_t)global_k * (size_t)dims.n + (size_t)global_col]
                    : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk += 8) {
                for (uint r = 0; r < acc_rows; r++) {
                    simdgroup_float8x8 a_tile;
                    simdgroup_load(a_tile, tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)BK + (size_t)kk, BK);
                    for (uint c_ = 0; c_ < acc_cols; c_++) {
                        simdgroup_float8x8 b_tile;
                        simdgroup_load(b_tile, tile_b + (size_t)kk * (size_t)BN + (size_t)(wn_idx * sub_bn + c_ * 8), BN);
                        simdgroup_multiply_accumulate(acc[r][c_], a_tile, b_tile, acc[r][c_]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // 直接ロード: device メモリから simdgroup ごとに直接
            // `simdgroup_load` する（協調ロードの同期コストを避ける経路。
            // BK の倍数でない末尾タイルは 8 要素単位でのみ発行できるため
            // full な 8 の倍数分のみを直接ロードし、端数（bk_eff が 8 の
            // 倍数でない残り）は寄与させない。呼び出し元は
            // `crate::pad::pad8` で k を 8 の倍数へパディング済みの実効
            // 次元を渡す契約のため、通常この端数は発生しない
            // （`crate::gemm::MetalGemm::dispatch_auto` 参照）。
            uint bk_full8 = (bk_eff / 8) * 8;
            for (uint kk = 0; kk < bk_full8; kk += 8) {
                for (uint r = 0; r < acc_rows; r++) {
                    uint a_row = sub_row0 + r * 8;
                    // 境界チェック（REQ-8）: `dims.m` は `crate::pad::pad8`
                    // により常に 8 の倍数へ揃えられ、`a_row` も常に 8 の
                    // 倍数（`sub_row0`・`r*8` がいずれも 8 の倍数）のため、
                    // `a_row < dims.m` が成立すれば `a_row+8 <= dims.m` も
                    // 常に成立し、8 行分の読み出しが実効次元内に収まる
                    // ことが保証される。不成立時は 0 埋めタイルを使う
                    // （協調ロード経路の 0 埋めと同じ契約。レビュー指摘。
                    // #188 PR review）。
                    simdgroup_float8x8 a_tile = simdgroup_float8x8(0.0f);
                    if (a_row < dims.m) {
                        simdgroup_load(a_tile, a + (size_t)a_row * (size_t)dims.k + (size_t)(p0 + kk), dims.k);
                    }
                    for (uint c_ = 0; c_ < acc_cols; c_++) {
                        uint b_col = sub_col0 + c_ * 8;
                        // 上記と同じ理屈（`dims.n` も pad8 済みで 8 の倍数、
                        // `b_col` も常に 8 の倍数）。
                        simdgroup_float8x8 b_tile = simdgroup_float8x8(0.0f);
                        if (b_col < dims.n) {
                            simdgroup_load(b_tile, b + (size_t)(p0 + kk) * (size_t)dims.n + (size_t)b_col, dims.n);
                        }
                        simdgroup_multiply_accumulate(acc[r][c_], a_tile, b_tile, acc[r][c_]);
                    }
                }
            }
        }
    }

    // ストア: サブブロック原点（`sub_row0`/`sub_col0`）が実効次元を超える
    // 8x8 タイルは書き込みをスキップする（REQ-8。ブロック端の手動境界
    // チェックを維持する。最適化を理由に省略しない）。
    for (uint r = 0; r < acc_rows; r++) {
        uint out_row = sub_row0 + r * 8;
        if (out_row >= dims.m) {
            continue;
        }
        for (uint c_ = 0; c_ < acc_cols; c_++) {
            uint out_col = sub_col0 + c_ * 8;
            if (out_col >= dims.n) {
                continue;
            }
            simdgroup_store(acc[r][c_], c + (size_t)out_row * (size_t)dims.n + (size_t)out_col, dims.n);
        }
    }
}
