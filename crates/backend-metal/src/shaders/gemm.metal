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

    // オーバーフロー安全なタイル数計算: `(k + TILE - 1) / TILE` は
    // `k > u32::MAX - (TILE - 1)` で uint 加算がラップし、必要なタイルを
    // 処理せず誤った結果を正常応答してしまう（REQ-8・codex-review 指摘）。
    // 商と余りを別々に計算する `k / TILE + (k % TILE != 0)` はいずれの
    // 演算も k を超えない範囲に収まりオーバーフローしない。
    uint num_tiles = k / TILE + ((k % TILE != 0) ? 1u : 0u);
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

// GEMM epilogue（bias 加算・activation）を融合した tiled GEMM（f32）。
// イシュー #605（CUDA 側 `TILED_BIAS_ACT_F32`〈#599〉の Metal 対応版）。
//
// `gemm_tiled` のアキュムレーション部分（threadgroup 共有メモリタイリング・
// 内積ループ）はそのまま維持し、C への書き込み直前の epilogue で
// `has_bias` が真なら `acc + bias[col]`、`act == 1` なら続けて
// `max(v, 0)` を適用してから 1 回だけ書く（`gemm` → `add` → `relu` の
// 非融合合成のように中間結果をバッファへ書いて読み直すことをしない。
// CPU 側 `gemm_blis_bias_act_parallel`・CUDA 側 `TILED_BIAS_ACT_F32` と
// 同じ「epilogue をカーネル内で完結させる」設計思想。`docs/kernel-fusion.md`
// §2.2）。
//
// **数値契約**: アキュムレーション自体は `gemm_tiled` と完全に同一の
// 演算順序（同じ threadgroup タイリング・同じ `fma()` ループ）のため、
// `gemm`→`add`→`relu` の非融合合成（同じ `gemm_tiled` を経由した後に
// 別カーネルで bias 加算・relu を適用する経路）と bit 完全一致になる
// （epilogue の加算・比較は要素独立で演算順序に依存しないため。
// `.claude/rules/coding-rust.md` の FMA 契約統一節・
// `docs/kernel-fusion.md` §2.2「bit 完全一致」と同じ論拠）。
//
// **`bias` が `None`（`has_bias == 0`）の場合**: ホスト側
// （`crate::gemm::MetalGemm::run_tiled_bias_act_f32`）は 1 要素ダミー
// バッファではなく `n` 要素のゼロ初期化バッファを渡す契約とする
// （`crate::rmsnorm::MetalRmsNorm::run_rmsnorm_f32_raw` の `w_buf` 契約と
// 同じ理由: `has_bias` ガードは条件分岐として書かれているが、Metal
// コンパイラがこれを両辺無条件評価の select 命令へ最適化する可能性が
// あり、その場合 `has_bias == 0` でも `bias[col]`〈`col` は最大 `n - 1`〉
// が実際にロードされうる。1 要素バッファでは `n > 1` のとき範囲外読み出し
// になるため、コンパイラの最適化戦略に依存しない fail-closed な対策として
// `n` 要素確保する。CUDA 側は条件分岐がハードウェア的に安全なため 1 要素
// ダミーで足りるが、Metal 側はこの追加保証が必要）。
//
// # REQ-8（カーネル境界検査規約）
//
// タイルロード時の三項ガード・C への書き込み時の `row < m && col < n`
// ガードは `gemm_tiled` と同一（該当コメント参照）。epilogue の
// `bias[col]` 参照は書き込みガード（したがって `col < n`）の内側でのみ
// 行うため、`bias`（`n` 要素確保済み）への範囲外読み出しは発生しない。
kernel void gemm_tiled_bias_act(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* c [[buffer(3)]],
    constant Dims& dims [[buffer(4)]],
    constant int& has_bias [[buffer(5)]],
    constant int& act [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float tile_a[TILE][TILE];
    threadgroup float tile_b[TILE][TILE];

    uint m = dims.m, n = dims.n, k = dims.k;
    uint row = gid.y, col = gid.x;
    float acc = 0.0;

    // オーバーフロー安全なタイル数計算: `(k + TILE - 1) / TILE` は
    // `k > u32::MAX - (TILE - 1)` で uint 加算がラップし、必要なタイルを
    // 処理せず誤った結果を正常応答してしまう（REQ-8・codex-review 指摘）。
    // 商と余りを別々に計算する `k / TILE + (k % TILE != 0)` はいずれの
    // 演算も k を超えない範囲に収まりオーバーフローしない。
    uint num_tiles = k / TILE + ((k % TILE != 0) ? 1u : 0u);
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

    // REQ-8: C への書き込み時の手動境界チェック（`gemm_tiled` と同一）。
    // epilogue（bias 加算・activation）はこのガードの内側でのみ適用し、
    // 中間結果を別カーネルへ渡さず 1 回の書き込みで完結させる。
    if (row < m && col < n) {
        float v = acc;
        if (has_bias != 0) {
            v += bias[col];
        }
        if (act == 1) {
            v = v > 0.0f ? v : 0.0f;
        }
        c[(size_t)row * (size_t)n + (size_t)col] = v;
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
// **累算精度契約（イシュー #380 実機検証で確定。実装計画時点の half 統一
// 判断から変更）**: A・B は `simdgroup_half8x8`（`MM_T`）のまま、
// アキュムレータ（`ACC_T`）は `simdgroup_float8x8` の f32 累算とする。
// 実装計画時点（Linux 実装環境）では「A/B が half・C が float という
// 混在型オーバーロードは `apple-silicon` スキルの参照ドキュメントに記載が
// なく、実地検証もできない」ことを理由に half 統一を選んでいたが、Apple
// Silicon 実機（M4 Max・macOS 26.6）での `MTLDevice.makeLibrary(source:)`
// ランタイムコンパイルによる spike で以下が判明した:
//
//   1. `simdgroup_multiply_accumulate(simdgroup_float8x8&, simdgroup_half8x8,
//      simdgroup_half8x8, simdgroup_float8x8)` はコンパイル成功する
//      （A/B=half・アキュムレータ=float の混在オーバーロードは実在する）。
//   2. ただし `simdgroup_store(simdgroup_float8x8, device half*)` は
//      コンパイル不可（診断: "deduced conflicting types for parameter 'T'
//      ('float' vs. 'half')"）。float アキュムレータを half 出力バッファへ
//      直接 store する経路は存在しない。
//
// この 2 点から、`simdgroup_float8x8` → `threadgroup float` へ一旦
// `simdgroup_store` → `threadgroup_barrier` で同期 → スレッド単位で
// `(half)` へ変換して `device half*` へ書き戻す、という 2 段エピローグ
// （変種 B）を採用する。`device float*` へ直接 store する変種 A ではなく
// 変種 B を選ぶ理由: `dispatch_f16_prepared_unverified` のシグネチャ・
// `MetalHalfBuffer` の `c_buf`・C バッファの転送バイト数を変えずに済み、
// #383（f16 対 PyTorch MPS f16 実測）の比較手法・
// `f16_dispatch_prepared_rejects_undersized_and_misaligned_inputs` の
// 入力検証契約（`MetalError::ALenMismatch { expected: 64, actual: 32 }`）が
// 無傷で保たれる。
//
// 本変更は CPU 参照実装（`backend_cpu::parity::matmul_reference_fma`。
// f32 累算 → 最後に 1 回だけ f16 へ丸め）との累算精度契約を一致させる
// ものであり、`.claude/rules/coding-rust.md` の FMA 契約統一方針（CPU
// 参照は `f32::mul_add`、GPU 側の既定 FMA 契約と揃える）、および CUDA 側
// WMMA f16（`kernels_wmma.rs::gemm_wmma_f16`。`f32.f16.f16.f32`。f32 累算）
// との整合を高める（REQ-2 複合判定の閾値・`backend_cpu::parity` は無変更。
// 「許容誤差の緩和」ではなく「累算精度の向上」）。
typedef simdgroup_half8x8 MM_T; // A・B の simdgroup タイル型（half のまま）
typedef simdgroup_float8x8 ACC_T; // アキュムレータ型（f32 累算。#380 で変更）

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
//
// エピローグの `threadgroup float stage[64]` は 8x8 タイル 1 つ分の
// 固定サイズ静的配列であり、`gemm_simdgroup_tiled` の動的共有メモリ
// （`setThreadgroupMemoryLength_atIndex`）とは別経路のため、本カーネルの
// dispatch 側（`encode_dispatch_f16`）に動的長さ設定を追加する必要はない。
// `thread_index_in_threadgroup` は `encode_dispatch_f16`
// （`crate::gemm::gemm.rs`）が `threads_per_tg = SIMDGROUP_THREADGROUP_WIDTH
// = 32` で dispatch する前提（`gemm_simdgroup` と同一の 1 threadgroup = 1
// simdgroup 構成）にもとづき、32 スレッドで 64 要素をストライド 2 巡で
// 書き戻す。
kernel void gemm_simdgroup_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint row0 = tgid.y * 8;
    uint col0 = tgid.x * 8;

    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }

    ACC_T acc = ACC_T(0.0f);
    uint k_tiles = dims.k / 8;
    for (uint t = 0; t < k_tiles; t++) {
        uint p0 = t * 8;
        MM_T a_tile;
        MM_T b_tile;
        simdgroup_load(a_tile, a + (size_t)row0 * (size_t)dims.k + (size_t)p0, dims.k);
        simdgroup_load(b_tile, b + (size_t)p0 * (size_t)dims.n + (size_t)col0, dims.n);
        simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
    }

    // 変種 B（#380 spike 実測で確定）: f32 アキュムレータを一旦
    // threadgroup メモリへ f32 のまま store し、全スレッド完了を
    // `threadgroup_barrier` で同期してから、各スレッドが担当要素を
    // half へ変換して `device half*` の C バッファへ書き戻す。
    // `simdgroup_store(simdgroup_float8x8, device half*)` は型不一致で
    // コンパイル不可（spike で確認済み）なため、f32 アキュムレータから
    // 直接 half バッファへ store することはできない。
    threadgroup float stage[64];
    simdgroup_store(acc, stage, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage はタイル内ローカル座標（行優先・ストライド 8）で埋まっている
    // 一方、`c` は行列全体のストライド `dims.n` を持つため、書き戻し時に
    // タイルローカル添字 (r, col) からグローバル添字
    // `(row0 + r) * dims.n + col0 + col` への写像を明示的に行う
    // （ストライドを取り違えると `f16_parity_baseline_8x8x8`
    // （dims.n == 8 のため写像ミスが露見しない）は通っても、非正方・
    // 大型形状（`dims.n != 8`）でパリティ FAIL として現れる）。
    for (uint i = tid; i < 64; i += 32u) { // 32 = Rust 側 SIMDGROUP_THREADGROUP_WIDTH（gemm.rs）と一致
        uint r = i / 8;
        uint col = i % 8;
        size_t dst = (size_t)(row0 + r) * (size_t)dims.n + (size_t)(col0 + col);
        c[dst] = (half)stage[i];
    }
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
// threadgroup memory のパディング幅（イシュー #538。行末に加算する f32
// 要素数）。`crate::tile::TileConfig::validate` が 4 の倍数（0 を含む）へ
// 事前検証済みの契約で、staged 経路の A/B タイル行ストライドを
// `BK+TGP_PAD`/`BN+TGP_PAD` へずらしてバンクコンフリクトを回避する
// （MLX steel `gemm.h` の `tgp_padding_a`/`tgp_padding_b` と同族の技法。
// direct-load 経路〈USE_TGP_STAGING=false〉では未使用。`TileConfig::validate`
// が `staged=false` のとき `pad=0` を強制するため、値が実際に効くのは
// staged 経路のみ）。
constant uint TGP_PAD [[function_constant(6)]];

// threadgroup ID スウィズル（イシュー #540・実験的機構）を本番 dispatch で
// 実際に有効化するかどうかのゲート。`crate::tile::SWIZZLE_ENABLED`
// （既定 `false`）が `crate::pipeline::make_pipeline_with_constants` 経由で
// 畳み込む。実機での性能効果・数値一致が `docs/perf/
// metal-gemm-tgid-swizzle-ab.md` の判断基準を満たすまで `false` のまま
// 据え置き、恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）で動作する
// （PR #661 codex-review 指摘: 未検証のスウィズルを本番経路へ無条件適用
// しない）。index は #538 で index 6 を占有した TGP_PAD の直後、index 7
// （main への rebase 時点での未使用最小 index）を割り当てる。
constant bool SWIZZLE_ENABLED [[function_constant(7)]];

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
    // threadgroup ID スウィズル（`swizzle_log` 相当。イシュー #540・
    // 実験的機構。採否は `docs/perf/metal-gemm-tgid-swizzle-ab.md` の A/B
    // 計測で判断する）: dispatch grid 上で `tile`（`1 << SWIZZLE_LOG` = 4）
    // threadgroup を縦方向へ束ねて走査順を変え、近接時刻に実行される
    // threadgroup 群が B（列方向）の同一領域を再利用しやすくする（MLX
    // steel `swizzle_log`・DeepGEMM の L2 スウィズルと同種。MLX 自身は
    // classic 経路で `swizzle_log = 0`〈無効〉のまま据え置いており未実証の
    // 技法である点に留意）。`SWIZZLE_ENABLED`（既定 `false`。上記宣言参照）
    // が `true` の場合のみ `crate::tile::swizzled_grid` が張った grid
    // （`grid_w = tiles_n << SWIZZLE_LOG`・`grid_h = div_ceil(tiles_m,
    // tile)`）を tgid が走査する契約で、変換後の `(tid_y, tid_x)` が元の
    // `(tiles_m, tiles_n)` を過不足なく覆うことを `crate::tile` の
    // `swizzled_grid_covers_every_tile_exactly_once` テストが Linux 上で
    // 静的に検証する。`SWIZZLE_ENABLED=false`（本番既定。PR #661
    // codex-review 指摘）では恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）
    // となり、`crate::gemm::encode_dispatch_tiled` 側もこのとき
    // `swizzled_grid` を使わず素朴な `(tiles_n, tiles_m)` grid を張る
    // （両者は同じ `SWIZZLE_ENABLED` 値で同期させる契約）。
    constexpr uint SWIZZLE_LOG = 2;
    constexpr uint SWIZZLE_TILE = 1u << SWIZZLE_LOG;
    uint tid_y = SWIZZLE_ENABLED ? ((tgid.y << SWIZZLE_LOG) + (tgid.x & (SWIZZLE_TILE - 1))) : tgid.y;
    uint tid_x = SWIZZLE_ENABLED ? (tgid.x >> SWIZZLE_LOG) : tgid.x;

    // 1 threadgroup が担当する C ブロックの原点（行優先: y=行, x=列）。
    // `tid_y`/`tid_x` はスウィズル後のタイル座標であり、`tiles_m` が
    // `SWIZZLE_TILE` の倍数でない場合に `tid_y >= tiles_m` となる余剰
    // threadgroup が生じうるが、直後の早期 return（REQ-8 境界チェック）が
    // これを無害化する（省略しない）。
    uint row0 = tid_y * BM;
    uint col0 = tid_x * BN;

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

    // threadgroup 共有メモリ上の A タイル（BM×lda）・B タイル（BK×ldb）
    // オフセット（`USE_TGP_STAGING=true` の場合のみ使用）。
    //
    // 行ストライドのパディング（イシュー #538。TGP_PAD function constant。
    // 本ファイル冒頭 TGP_PAD 宣言のコメント参照）: `lda`/`ldb` を素の
    // `BK`/`BN` ではなく `+TGP_PAD` した値にすることで、`simdgroup_load` の
    // 列方向アクセス（`kk`/`wn_idx*sub_bn+c_*8` を跨ぐストライド走査）が
    // threadgroup メモリのバンク境界と整合してしまうことによるバンク
    // コンフリクトを回避する（MLX steel `gemm.h` の
    // `tgp_padding_a`/`tgp_padding_b`・metal-flash-attention の
    // leadingBlockDimensions 実値指定・TileKernels の `TILE_X + TILE_K`
    // 確保と同族の技法。CUDA 側 B-7 と同族。`TileConfig::shared_mem_bytes`
    // が本カーネルと同じ `bm*lda + bk*ldb` 総量を計算する契約）。
    uint lda = BK + TGP_PAD;
    uint ldb = BN + TGP_PAD;
    threadgroup float* tile_a = shared_mem;
    threadgroup float* tile_b = shared_mem + (size_t)BM * (size_t)lda;

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
            //
            // float4 ベクトルロード（イシュー #533。MLX `BlockLoader`
            // 〈loader.h〉が 128bit 幅相当の単位で読み出す方式を参考に、
            // 1 要素ずつのスカラーロードから変更）。アラインメント成立
            // 根拠:
            //   - `crate::tile::TileConfig::validate`（tile.rs）が
            //     `staged=true` の構成に対し要求する `bk % 8 == 0`・
            //     `bn % (wn*8) == 0`（`wn >= 1` のため `bn % 8 == 0` を
            //     含意）という既存の 8 整除検査が、`TileConfig::VEC_WIDTH`
            //     （4）整除を数学的に包含する（`8 | x ⟹ 4 | x`）。専用の
            //     VEC_WIDTH 整除検査 variant は追加していない（`pub enum
            //     TileConfigError` への variant 追加が下流の網羅的
            //     `match` を破壊する P1 指摘・PR #672 codex-review を受け
            //     既存 variant で拒否できる現行契約を維持する方針へ変更）。
            //     この間接包含はテスト（tile.rs
            //     `validate_ok_implies_vec_width_divisibility`）で固定
            //     済み（イシュー #535）。この前提により A タイルの行長
            //     BK・B タイルの行長 BN はともに 4 の倍数であり、4 要素
            //     グループが行境界をまたぐことはない
            //   - `MetalGemm::dispatch_variant`（gemm.rs）が `SimdgroupTiled`
            //     向けに m/n/k を `crate::pad::pad8` で 8 の倍数へ実効
            //     次元パディング済みで渡す契約のため、`p0 = t*BK`・
            //     `col0 = tgid.x*BN` は常に 8 の倍数（4 要素境界）に揃い、
            //     device メモリ側の読み出し先頭オフセットも 4 要素（16
            //     バイト）境界に揃う（MTLBuffer 先頭はページ境界確保）
            //   - 共有メモリ側オフセット（`tile_b` の `BM*lda`）も 4 の倍数
            //     のため `threadgroup float4*` 再解釈も 16 バイト境界に揃う
            //     （`lda = BK+TGP_PAD` は BK が上記 8 整除検査からの間接
            //     包含で常に 4 の倍数、TGP_PAD（`TileConfig::pad`）も
            //     `TGP_PAD_ELEMS=4` で常に 4 の倍数〈`TileConfig::validate`
            //     が dispatch 前に検証〉のため常に 4 の倍数。書き込み先
            //     添字 `r*lda+kk`／`kk*ldb+c_` も同じ理由で常に 4 要素境界
            //     に揃う。イシュー #538）
            // `TileConfig::validate` の既存 8 整除検査が上記を dispatch
            // 前に fail-closed で拒否するため、将来 `CANDIDATES` に
            // BK/BN が 8 で割り切れない構成を追加しようとしても検証段階
            // で弾かれ本カーネルへ到達しない（防衛層。イシュー #535）。
            // それでも「グループ全 4 要素が in-bounds か」を本シェーダ側
            // でも明示判定し、境界グループは範囲外アドレスへ一切触れず
            // 要素単位のスカラー読み出し + 0 埋めへフォールバックする
            // （最適化を理由に手動境界チェックを省略しない。REQ-8・
            // `.claude/rules/coding-rust.md`「カーネル実装の境界検査」）。
            // 共有メモリへ格納される値は変更前のスカラーループとビット単位
            // で一致するため、以降の `simdgroup_load`／MMA 発行順・数値
            // 結果は不変（論理添字 r/kk/c_ の導出・境界判定は非パディング
            // 平坦添字のまま維持し、書き込み先のみパディング込みストライド
            // へ変更しているため。イシュー #538 計画「設計方針」節）。
            // パディング列（`lda-BK`／`ldb-BN` 分の隙間）自体は
            // `simdgroup_load` が一切読まない（各行の読み出しは
            // `kk..kk+8 ⊆ 0..BK` に収まる）ため 0 埋め不要。
            uint local_tid = simd_id * 32 + simd_lane;
            uint threads_total = WM * WN * 32;

            uint a_vecs = (BM * BK) / 4;
            for (uint vi = local_tid; vi < a_vecs; vi += threads_total) {
                uint idx = vi * 4;
                uint r = idx / BK;
                uint kk = idx % BK;
                uint dst_idx = r * lda + kk; // パディング込みの書き込み先添字。
                uint global_row = row0 + r;
                uint global_k = p0 + kk;
                bool group_in_bounds =
                    (kk + 4 <= bk_eff) && (global_row < dims.m) && (global_k + 4 <= dims.k);
                if (group_in_bounds) {
                    device const float4* src = reinterpret_cast<device const float4*>(
                        a + (size_t)global_row * (size_t)dims.k + (size_t)global_k);
                    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(tile_a + dst_idx);
                    *dst = *src;
                } else {
                    for (uint e = 0; e < 4; e++) {
                        uint kk_e = kk + e;
                        uint global_k_e = global_k + e;
                        tile_a[dst_idx + e] = (kk_e < bk_eff && global_row < dims.m && global_k_e < dims.k)
                            ? a[(size_t)global_row * (size_t)dims.k + (size_t)global_k_e]
                            : 0.0f;
                    }
                }
            }

            uint b_vecs = (BK * BN) / 4;
            for (uint vi = local_tid; vi < b_vecs; vi += threads_total) {
                uint idx = vi * 4;
                uint kk = idx / BN;
                uint c_ = idx % BN;
                uint dst_idx = kk * ldb + c_; // パディング込みの書き込み先添字。
                uint global_k = p0 + kk;
                uint global_col = col0 + c_;
                bool group_in_bounds =
                    (kk < bk_eff) && (global_k < dims.k) && (global_col + 4 <= dims.n);
                if (group_in_bounds) {
                    device const float4* src = reinterpret_cast<device const float4*>(
                        b + (size_t)global_k * (size_t)dims.n + (size_t)global_col);
                    threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(tile_b + dst_idx);
                    *dst = *src;
                } else {
                    for (uint e = 0; e < 4; e++) {
                        uint c_e = c_ + e;
                        uint global_col_e = global_col + e;
                        tile_b[dst_idx + e] = (kk < bk_eff && global_k < dims.k && global_col_e < dims.n)
                            ? b[(size_t)global_k * (size_t)dims.n + (size_t)global_col_e]
                            : 0.0f;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk += 8) {
                for (uint r = 0; r < acc_rows; r++) {
                    simdgroup_float8x8 a_tile;
                    // ストライドを BK ではなく lda（パディング込み）にする
                    // ことで、パディングにより実際にずれた行の先頭アドレス
                    // を正しく指す（イシュー #538）。
                    simdgroup_load(a_tile, tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)lda + (size_t)kk, lda);
                    for (uint ci = 0; ci < acc_cols; ci++) {
                        // 蛇行（serpentine）走査: 奇数行 r では列を逆順（acc_cols-1-ci）に
                        // 辿る。a_tile は r ループ内で 1 回だけロードされ ci に
                        // 依らず不変なため、走査順が影響するのは tile_b からの
                        // `simdgroup_load` アドレス列（行切替時に直前で使った
                        // 列位置付近を再訪する）のみである（MLX `tile_matmad`
                        // 〈mma.h〉・CUTLASS `mma_tensor_op.h` 同型の技法。CUDA 側
                        // #497 B-6 と同一。#536）。この tile_b アクセス局所性向上は
                        // 期待効果であり、実測結果は `docs/perf/metal-gemm-serpentine-ab.md`
                        // に記録する（未計測時点では性能上の主張を断定しない）。
                        // acc[r][c_] ごとの累算オペランド列（K 方向の順序）は
                        // c_ の訪問順に依らず不変なので、結果はビット単位で
                        // 従来の行優先走査と一致する。
                        uint c_ = (r % 2 == 1) ? (acc_cols - 1 - ci) : ci;
                        simdgroup_float8x8 b_tile;
                        // ストライドを BN ではなく ldb（パディング込み）にする
                        // （上記 A タイル側と同じ理由。イシュー #538）。
                        simdgroup_load(b_tile, tile_b + (size_t)kk * (size_t)ldb + (size_t)(wn_idx * sub_bn + c_ * 8), ldb);
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
                    for (uint ci = 0; ci < acc_cols; ci++) {
                        // 蛇行（serpentine）走査: staged 経路と同じ理由・出典（#536）。
                        // c_ の訪問順を変えるだけで acc[r][c_] の累算オペランド列は
                        // 不変のため数値はビット単位で従来と一致する。
                        uint c_ = (r % 2 == 1) ? (acc_cols - 1 - ci) : ci;
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
