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

// イシュー #1040: `gemm_tiled_bias_act` の A/B オペランドを転置パターン
// （NN/NT/TN/TT）・stride 付きで読み出すためのパラメータ。`lda`/`ldb` は
// leading dimension（`trans_* == 0` なら行 stride、`trans_* == 1` なら
// 列 stride。`crate::layout::MatrixLayout::ld` と同じ意味）。
// `crate::gemm::GemmStrides`（repr(C)）とレイアウトを一致させる
// （4 × uint32 = 16 バイト。`crate::gemm` のレイアウト一致テスト参照）。
// NN（`trans_a == 0 && trans_b == 0`・`lda == k`・`ldb == n`）は既存の
// 密行優先アクセスと完全に同一の添字になり、`crate::gemm::
// dispatch_bias_act_prepared`（後方互換入口）はこの構成で
// `dispatch_strided_bias_act_prepared` へ委譲する。
struct GemmStrides {
    uint lda;
    uint ldb;
    uint trans_a;
    uint trans_b;
};

// === タイル化カーネル共通境界検査ヘルパ（イシュー #1038） ===
//
// `gemm_simdgroup_tiled`（f32・#188/#532/#538/#745）・
// `gemm_simdgroup_tiled_f16`（half・#796/#797）の 2 系統が手書きで
// 二重管理していた REQ-8 境界検査述語（ブロック原点の早期 return・
// 協調ロードのベクトルグループ in-bounds 判定・境界グループの要素単位
// スカラーフォールバック判定）をここへ集約する。いずれも副作用のない
// 純粋述語（bool を返すのみ）であり、呼び出し側の判定式は抽出前後で
// 完全に等価（フラグメントロード順・MMA 発行順・累算順は一切変更しない。
// #536/#538 と同じ「ビット単位で不変」の論法）。
//
// dtype 差分（協調ロードのベクトル幅 4/8・共有メモリレイアウト・
// エピローグ staging）はこの述語だけでは吸収できないため、
// `docs/backend-metal-aligned-load-decision.md`「検査省略型 variant
// 不採用」判断を踏まえ、本イシューでは境界判定式の共通化までに留め、
// ロード・ストア本体（型テンプレート化）は Mac 実機での MSL コンパイル
// 確認が可能なセッションへ持ち越す（イシュー #1038 計画 §3.1 フォールバック
// 基準。以下 5 関数のいずれも `BM`/`BN`/`BK`/`WM`/`WN` 等の function
// constant に依存しないため、両カーネルより前方のファイル冒頭
// （`Dims` 定義直後）へ配置してもカーネル側の宣言順序制約を受けない）。

// ブロック原点（`row0`/`col0`）が実効次元を完全に超えるかどうかの判定
// （REQ-8 第 1 ガード）。f32/f16 両カーネルとも早期 return の条件式として
// 同一に使う。
inline bool tiled_block_out_of_range(uint row0, uint col0, constant Dims& dims) {
    return row0 >= dims.m || col0 >= dims.n;
}

// 協調ロード（staged 経路）の A タイル側ベクトルグループ in-bounds 判定
// （REQ-8）。`vec_w` はロード幅（f32 版 float4=4、f16 版 half8 相当の
// 128bit 幅=8）。グループ全 `vec_w` 要素が実効次元・K タイル端の内側に
// 収まる場合のみベクトルロードを許可し、境界グループは呼び出し側で
// 要素単位のスカラーフォールバック（`tiled_a_elem_in_bounds`）へ回す。
inline bool tiled_a_group_in_bounds(
    uint kk, uint bk_eff, uint global_row, uint global_k, uint vec_w, constant Dims& dims
) {
    return (kk + vec_w <= bk_eff) && (global_row < dims.m) && (global_k + vec_w <= dims.k);
}

// 協調ロードの B タイル側ベクトルグループ in-bounds 判定（A タイル側と対。
// REQ-8）。
inline bool tiled_b_group_in_bounds(
    uint kk, uint bk_eff, uint global_k, uint global_col, uint vec_w, constant Dims& dims
) {
    return (kk < bk_eff) && (global_k < dims.k) && (global_col + vec_w <= dims.n);
}

// A タイル境界グループの要素単位スカラーフォールバック判定（REQ-8）。
// ベクトル幅に依存しないため f32/f16 で完全同一の述語。
inline bool tiled_a_elem_in_bounds(
    uint kk_e, uint bk_eff, uint global_row, uint global_k_e, constant Dims& dims
) {
    return kk_e < bk_eff && global_row < dims.m && global_k_e < dims.k;
}

// B タイル境界グループの要素単位スカラーフォールバック判定（A タイル側と
// 対。REQ-8）。
inline bool tiled_b_elem_in_bounds(
    uint kk, uint bk_eff, uint global_k, uint global_col_e, constant Dims& dims
) {
    return kk < bk_eff && global_k < dims.k && global_col_e < dims.n;
}

// === 転置ロード側境界検査ヘルパ（イシュー #1138） ===
//
// `gemm_simdgroup_tiled` の `TRANS_A`/`TRANS_B` 分岐（協調ロード）が使う
// 4 関数。上記 NN 用 4 関数（`tiled_a_group_in_bounds` 等）とは「ベクトル化
// 方向（fast dim）」が入れ替わる関係にある: A が転置（`TRANS_A`）される
// と device 側の連続方向が K→M へ変わるため、ベクトルロードは M 方向へ
// 発行し K 方向はスカラー判定になる（B・NN と同じ形の判定式に帰着する
// ため `tiled_at_*` は `tiled_b_*` と同型・`dims.n`→`dims.m` 置換）。逆に
// B が転置（`TRANS_B`）されると device 側の連続方向が N→K へ変わるため、
// ベクトルロードは K 方向・N 方向はスカラー判定になる（A・NN と同型・
// `dims.m`→`dims.n` 置換）。既存 5 関数の本体・シグネチャは一切変更せず
// （`tests/shader_source_evidence.rs`
// `gemm_metal_boundary_helpers_retain_req8_condition_expressions` が
// 厳密固定）、新規関数として追加する。
inline bool tiled_at_group_in_bounds(
    uint kk, uint bk_eff, uint global_row, uint global_k, uint vec_w, constant Dims& dims
) {
    return (kk < bk_eff) && (global_k < dims.k) && (global_row + vec_w <= dims.m);
}

inline bool tiled_at_elem_in_bounds(
    uint kk, uint bk_eff, uint global_row_e, uint global_k, constant Dims& dims
) {
    return kk < bk_eff && global_k < dims.k && global_row_e < dims.m;
}

inline bool tiled_bt_group_in_bounds(
    uint kk, uint bk_eff, uint global_col, uint global_k, uint vec_w, constant Dims& dims
) {
    return (kk + vec_w <= bk_eff) && (global_col < dims.n) && (global_k + vec_w <= dims.k);
}

inline bool tiled_bt_elem_in_bounds(
    uint kk_e, uint bk_eff, uint global_col, uint global_k_e, constant Dims& dims
) {
    return kk_e < bk_eff && global_col < dims.n && global_k_e < dims.k;
}

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
//
// イシュー #1040: A/B の添字を `GemmStrides`（`st`）に基づく転置対応式へ
// 一般化した。`st.trans_a == 0 && st.trans_b == 0 && st.lda == k &&
// st.ldb == n` の NN 構成では、以下の `A(row, kk)`/`B(kk, col)` は
// それぞれ従来の `a[row*k+kk]`/`b[kk*n+col]` と完全に同一の式へ簡約される
// ため、既存 `dispatch_bias_act_prepared`（NN 専用後方互換入口）の
// 数値結果は非後退（`crate::gemm` のビット同一テスト参照）。
// `lda`/`ldb` は転置有無に応じて「行 stride」または「列 stride」の
// いずれかを表す（`crate::layout::MatrixLayout` ドキュメンテーション
// コメント参照）。手動境界チェック（`row < m && a_col < k` 等）は
// 添字の変更後も変わらず維持する（REQ-8。境界検査省略の正当化に
// 最適化を用いない）。
kernel void gemm_tiled_bias_act(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* c [[buffer(3)]],
    constant Dims& dims [[buffer(4)]],
    constant int& has_bias [[buffer(5)]],
    constant int& act [[buffer(6)]],
    constant GemmStrides& st [[buffer(7)]],
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

        // A(row, a_col): `st.trans_a` が真なら列優先（転置 view）として
        // `a[a_col * lda + row]`、偽なら行優先として `a[row * lda +
        // a_col]`（`crate::layout` の `a_at` 参照実装・
        // `tests/shader_source_evidence.rs` の needle と一致させる）。
        tile_a[lid.y][lid.x] = (row < m && a_col < k)
            ? (st.trans_a != 0
                   ? a[(size_t)a_col * (size_t)st.lda + (size_t)row]
                   : a[(size_t)row * (size_t)st.lda + (size_t)a_col])
            : 0.0;
        // B(b_row, col): `st.trans_b` が真なら `b[col * ldb + b_row]`、
        // 偽なら `b[b_row * ldb + col]`。
        tile_b[lid.y][lid.x] = (b_row < k && col < n)
            ? (st.trans_b != 0
                   ? b[(size_t)col * (size_t)st.ldb + (size_t)b_row]
                   : b[(size_t)b_row * (size_t)st.ldb + (size_t)col])
            : 0.0;

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

// simdgroup 細粒度同期ゲート（イシュー #809・実験的機構）。
// `crate::tile::FINE_BARRIER_ENABLED`（既定 `false`）が
// `crate::pipeline::make_pipeline_with_constants` 経由で畳み込む。
// `gemm_simdgroup_tiled` の staged 経路（`USE_TGP_STAGING=true`）のみが
// 対象で、kk ループ内でフラグメント一括ロード（`a_frag`/`b_frag`。#745）
// と MMA 発行の間に `simdgroup_barrier(mem_flags::mem_none)`（MLX steel
// `mma.h` 型。`threadgroup_barrier` より軽量な simdgroup スコープの
// フェンス）を挿入するかどうかを切り替える。`tile_a`/`tile_b` の内容自体
// は前段の `threadgroup_barrier(mem_flags::mem_threadgroup)`（協調ロード
// 直後）で既に確定済みのため、この挿入は演算オペランド列を一切変えず
// （フラグメントロード順・MMA 発行順は不変）、有効化時も数値はビット単位
// で不変（#536・#538 と同じ論法）。採否は A/B 実測
// （`examples/gemm_fine_barrier_ab_bench.rs`・
// `docs/perf/metal-gemm-fine-barrier-ab.md`）で判断する。index は
// SWIZZLE_ENABLED（#540・index 7）の直後、index 8（main への rebase時点
// での未使用最小 index）を割り当てる。
constant bool FINE_BARRIER_ENABLED [[function_constant(8)]];

// 転置ロードゲート（イシュー #1138）: `gemm_simdgroup_tiled` を NT/TN/TT
// パターン（`GemmStrides.trans_a`/`trans_b`。`gemm_tiled_bias_act` の
// classic strided 経路が使う添字と同じ意味）へ拡張するための function
// constant。`crate::gemm::MetalGemm::pipeline_for_tile` が呼び出し元の
// `TransposePattern`（`crate::layout::TransposePattern`）から
// `make_pipeline_with_constants` 経由で畳み込む。index は
// FINE_BARRIER_ENABLED（#809・index 8）の直後、index 9/10（本ファイル内で
// 未使用の最小 index。#540/#538 の index 衝突再発防止として
// `tests/shader_source_evidence.rs` が index まで含めて固定する）を
// 割り当てる。`false`（NN）の場合は本カーネルの device 側アドレス式・
// threadgroup 側配置・フラグメントロード順・MMA 発行順が既存 NN 経路と
// 完全に同一になる（`crate::gemm` のビット同一テスト参照。
// `docs/backend-metal-transpose-collapse-design.md` §2）。
constant bool TRANS_A [[function_constant(9)]];
constant bool TRANS_B [[function_constant(10)]];

// 条件付き loop unroll ゲート（イシュー #1282）: `gemm_simdgroup_tiled` の
// アキュムレータ系ループ 10 箇所（acc 初期化・staged フラグメントロード・
// staged/direct-load MMA 発行・エピローグストア）を 6 ブロックの
// `if (UNROLL_ACC_ENABLED) { <unroll 版> } else { <非 unroll 版> }` で
// 挟み、`#pragma clang loop unroll(full)` を適用するかどうかを実行時
// コンパイル時定数で切り替える。E1 実験（`docs/perf/
// metal-gemm-n4096-kernel-gap.md` §7）で無条件付与を試み、
// `acc_rows*acc_cols>=16` の候補（`crate::tile::CANDIDATES[0]`/`[4]`/`[8]`）
// は改善する一方、本番 `dispatch_auto` が選ぶ `acc<=8` 系候補は後退した
// ため、`crate::tile::TileConfig::unroll_acc_loops`（acc 積 >=
// `crate::tile::UNROLL_ACC_MIN_PRODUCT`）で判定された候補のみへ適用を
// 限定する。値は `crate::gemm::MetalGemm::pipeline_for_tile` が
// フォールバック chain 巡回中の候補ごとに
// `crate::tile::unroll_acc_loops_for(candidate, instance_flag)` で導出し
// `make_pipeline_with_constants`（`crate::pipeline::GemmGateConstants`）
// 経由で畳み込む。**本番既定は `false`**（`crate::tile::
// UNROLL_ACC_ENABLED`）で、既定挙動は各ブロックの `else` 側（従来の
// 非 unroll ループとバイト同一）を通る。性能実測・本番既定の `true` への
// 切替判断は兄弟イシュー #1284 のスコープ（本ファイルの変更自体は
// `dispatch_auto` の既定挙動を変えない）。index は TRANS_B（#1138・
// index 10）の直後の 11（本ファイル内で未使用の最小 index。
// `tests/shader_source_evidence.rs` が index まで含めて固定する）。
// `gemm_simdgroup_tiled_f16`（下方）は本定数を参照しない
// （`crate::gemm::MetalGemm::pipeline_for_tile_f16` は常に `false` を渡す。
// `pipeline_for_tile_f16` 呼び出し側コメント参照）。
constant bool UNROLL_ACC_ENABLED [[function_constant(11)]];

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
    // イシュー #1138: NT/TN/TT 転置ロード（`TRANS_A`/`TRANS_B`）の lda/ldb
    // を受け取る。NN（`trans_a==0 && trans_b==0`）では `st.lda`/`st.ldb` を
    // 参照せず（`lda`/`ldb` は BK/BN + TGP_PAD の既存 NN 式のまま）、本
    // カーネルの NN 経路の数値・添字は非後退（`crate::gemm` のビット同一
    // テスト参照）。`crate::gemm::GemmStrides`（repr(C)。
    // `gemm_tiled_bias_act` と共用のレイアウト一致テスト対象）。
    constant GemmStrides& st [[buffer(4)]],
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
    if (tiled_block_out_of_range(row0, col0, dims)) {
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
    // 条件付き loop unroll（イシュー #1282。本ファイル冒頭
    // UNROLL_ACC_ENABLED 宣言のコメント参照）: unroll 版・非 unroll 版は
    // どちらも同じ acc[r][c_] = 0 の代入列を r/c_ 昇順で行うだけで、
    // 演算オペランド列は不変（数値は unroll 有無に関わらずビット単位で
    // 一致する）。
    if (UNROLL_ACC_ENABLED) {
#pragma clang loop unroll(full)
        for (uint r = 0; r < acc_rows; r++) {
#pragma clang loop unroll(full)
            for (uint c_ = 0; c_ < acc_cols; c_++) {
                acc[r][c_] = simdgroup_float8x8(0.0f);
            }
        }
    } else {
        for (uint r = 0; r < acc_rows; r++) {
            for (uint c_ = 0; c_ < acc_cols; c_++) {
                acc[r][c_] = simdgroup_float8x8(0.0f);
            }
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
    // イシュー #1138: `TRANS_A`/`TRANS_B` が真の場合、threadgroup タイルは
    // 転置後の物理レイアウト（A: BK×(BM+pad)・B: BN×(BK+pad)。本ファイル
    // 冒頭 TRANS_A/TRANS_B 宣言のコメント参照）で確保するため `lda`/`ldb`
    // を上書きする。上記 2 行（NN 既定値）は
    // `tests/shader_source_evidence.rs::gemm_simdgroup_tiled_source_uses_tgp_padding_stride`
    // が固定するテキストのため変更しない（NN 経路はこの分岐に入らず
    // 素通りする）。
    if (TRANS_A) {
        lda = BM + TGP_PAD;
    }
    if (TRANS_B) {
        ldb = BK + TGP_PAD;
    }
    // A タイルの行数（threadgroup 上の確保行数）。NN は BM（M 方向）、
    // TRANS_A は BK（K 方向）。B タイルのオフセットはこの行数 × lda で
    // 決まる（NN では従来の `BM * lda` と同じ値になる）。
    uint a_tile_rows = TRANS_A ? BK : BM;
    threadgroup float* tile_a = shared_mem;
    threadgroup float* tile_b = shared_mem + (size_t)a_tile_rows * (size_t)lda;

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
            if (TRANS_A) {
                // イシュー #1138: device 側 A は `[k, m]` 行優先（連続方向が
                // M）のため、ベクトル方向を K→M へ入れ替える（`tiled_at_*`
                // ヘルパは `tiled_b_*` と同型・M 方向を検査。本ファイル冒頭
                // ヘルパ群のコメント参照）。threadgroup タイルは
                // `BK×(BM+pad)`（行=K・列=M）で確保済み（上記 `a_tile_rows`）。
                for (uint vi = local_tid; vi < a_vecs; vi += threads_total) {
                    uint idx = vi * 4;
                    uint kk = idx / BM;
                    uint r = idx % BM;
                    uint dst_idx = kk * lda + r;
                    uint global_row = row0 + r;
                    uint global_k = p0 + kk;
                    bool group_in_bounds = tiled_at_group_in_bounds(kk, bk_eff, global_row, global_k, 4, dims);
                    if (group_in_bounds) {
                        device const float4* src = reinterpret_cast<device const float4*>(
                            a + (size_t)global_k * (size_t)st.lda + (size_t)global_row);
                        threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(tile_a + dst_idx);
                        *dst = *src;
                    } else {
                        for (uint e = 0; e < 4; e++) {
                            uint global_row_e = global_row + e;
                            tile_a[dst_idx + e] = tiled_at_elem_in_bounds(kk, bk_eff, global_row_e, global_k, dims)
                                ? a[(size_t)global_k * (size_t)st.lda + (size_t)global_row_e]
                                : 0.0f;
                        }
                    }
                }
            } else {
                for (uint vi = local_tid; vi < a_vecs; vi += threads_total) {
                    uint idx = vi * 4;
                    uint r = idx / BK;
                    uint kk = idx % BK;
                    uint dst_idx = r * lda + kk; // パディング込みの書き込み先添字。
                    uint global_row = row0 + r;
                    uint global_k = p0 + kk;
                    bool group_in_bounds = tiled_a_group_in_bounds(kk, bk_eff, global_row, global_k, 4, dims);
                    if (group_in_bounds) {
                        device const float4* src = reinterpret_cast<device const float4*>(
                            a + (size_t)global_row * (size_t)st.lda + (size_t)global_k);
                        threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(tile_a + dst_idx);
                        *dst = *src;
                    } else {
                        for (uint e = 0; e < 4; e++) {
                            uint kk_e = kk + e;
                            uint global_k_e = global_k + e;
                            tile_a[dst_idx + e] = tiled_a_elem_in_bounds(kk_e, bk_eff, global_row, global_k_e, dims)
                                ? a[(size_t)global_row * (size_t)st.lda + (size_t)global_k_e]
                                : 0.0f;
                        }
                    }
                }
            }

            uint b_vecs = (BK * BN) / 4;
            if (TRANS_B) {
                // イシュー #1138: device 側 B は `[n, k]` 行優先（連続方向が
                // K）のため、ベクトル方向を N→K へ入れ替える（`tiled_bt_*`
                // ヘルパは `tiled_a_*` と同型・K 方向をベクトル判定）。
                // threadgroup タイルは `BN×(BK+pad)`（行=N・列=K）で確保済み。
                for (uint vi = local_tid; vi < b_vecs; vi += threads_total) {
                    uint idx = vi * 4;
                    uint c_ = idx / BK;
                    uint kk = idx % BK;
                    uint dst_idx = c_ * ldb + kk;
                    uint global_k = p0 + kk;
                    uint global_col = col0 + c_;
                    bool group_in_bounds = tiled_bt_group_in_bounds(kk, bk_eff, global_col, global_k, 4, dims);
                    if (group_in_bounds) {
                        device const float4* src = reinterpret_cast<device const float4*>(
                            b + (size_t)global_col * (size_t)st.ldb + (size_t)global_k);
                        threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(tile_b + dst_idx);
                        *dst = *src;
                    } else {
                        for (uint e = 0; e < 4; e++) {
                            uint kk_e = kk + e;
                            uint global_k_e = global_k + e;
                            tile_b[dst_idx + e] = tiled_bt_elem_in_bounds(kk_e, bk_eff, global_col, global_k_e, dims)
                                ? b[(size_t)global_col * (size_t)st.ldb + (size_t)global_k_e]
                                : 0.0f;
                        }
                    }
                }
            } else {
                for (uint vi = local_tid; vi < b_vecs; vi += threads_total) {
                    uint idx = vi * 4;
                    uint kk = idx / BN;
                    uint c_ = idx % BN;
                    uint dst_idx = kk * ldb + c_; // パディング込みの書き込み先添字。
                    uint global_k = p0 + kk;
                    uint global_col = col0 + c_;
                    bool group_in_bounds = tiled_b_group_in_bounds(kk, bk_eff, global_k, global_col, 4, dims);
                    if (group_in_bounds) {
                        device const float4* src = reinterpret_cast<device const float4*>(
                            b + (size_t)global_k * (size_t)st.ldb + (size_t)global_col);
                        threadgroup float4* dst = reinterpret_cast<threadgroup float4*>(tile_b + dst_idx);
                        *dst = *src;
                    } else {
                        for (uint e = 0; e < 4; e++) {
                            uint c_e = c_ + e;
                            uint global_col_e = global_col + e;
                            tile_b[dst_idx + e] = tiled_b_elem_in_bounds(kk, bk_eff, global_k, global_col_e, dims)
                                ? b[(size_t)global_k * (size_t)st.ldb + (size_t)global_col_e]
                                : 0.0f;
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk += 8) {
                // フラグメントのレジスタ常駐化（イシュー #745）: kk ステップ
                // 先頭で A の acc_rows 個・B の acc_cols 個の simdgroup
                // フラグメントを一括ロードしてから TM×TN の外積 MMA を発行
                // する（MLX steel `mma.h` の tile_matmad 型構造。#487 診断
                // で確認した「アキュムレータは K 全域でレジスタ常駐済み・
                // 差分はフラグメントロード構造のみ」を踏まえた是正）。従来は
                // (r, ci) の内側ループで毎回 b_tile を再ロードしていたため
                // 1 kk ステップあたり threadgroup→register ロードが
                // TM + TM*TN 回発生していた（acc_rows=acc_cols=4 で
                // 4+16=20 回）。フラグメント配列へ先出しすることで
                // TM + TN 回（4+4=8 回）まで削減する。旧蛇行（serpentine）
                // 走査（#536）は b_tile 再ロードの局所性向上を狙った技法
                // だったが、本巻き上げにより再ロード自体が構造的に消滅する
                // ため効果の前提が失われ撤去し、MMA 発行順は行優先へ戻す
                // （direct-load 経路〈else 節〉はフラグメント再ロードが
                // 残るため #536 の蛇行走査を引き続き維持する）。
                // 各 acc[r][c_] の K 方向累算オペランド列（値・順序）は
                // ロードスケジューリングを変えても c_/r の訪問順によらず
                // 不変なため、結果はビット単位で従来と一致する（#536・#538
                // と同じ論法）。
                simdgroup_float8x8 a_frag[MAX_ACC];
                simdgroup_float8x8 b_frag[MAX_ACC];
                // 条件付き loop unroll（イシュー #1282。本ファイル冒頭
                // UNROLL_ACC_ENABLED 宣言のコメント参照）: unroll 版・非
                // unroll 版のどちらも a_frag[r] への `simdgroup_load` を r
                // 昇順で行うだけで、演算オペランド列は不変。
                if (UNROLL_ACC_ENABLED) {
#pragma clang loop unroll(full)
                    for (uint r = 0; r < acc_rows; r++) {
                        // ストライドを BK ではなく lda（パディング込み）にする
                        // ことで、パディングにより実際にずれた行の先頭アドレス
                        // を正しく指す（イシュー #538）。
                        //
                        // イシュー #1138: `TRANS_A` の場合 threadgroup タイルは
                        // `BK×(BM+pad)`（行=K・列=M）で確保されているため、
                        // `simdgroup_load` の `transpose_matrix=true` で
                        // K×M ブロックを M×K へ転置して読み出す（自然読み出しの
                        // 8x8 ブロックは K 方向 8・M 方向 8 で、transpose 後に
                        // A_frag の期待するレイアウト〈M 方向行・K 方向列〉と
                        // 一致する。`docs/backend-metal-transpose-collapse-design.md`
                        // §2・本ファイル冒頭 TRANS_A 宣言のコメント参照）。
                        if (TRANS_A) {
                            simdgroup_load(a_frag[r], tile_a + (size_t)kk * (size_t)lda + (size_t)(wm_idx * sub_bm + r * 8), lda, ulong2(0), true);
                        } else {
                            simdgroup_load(a_frag[r], tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)lda + (size_t)kk, lda);
                        }
                    }
                } else {
                    for (uint r = 0; r < acc_rows; r++) {
                        // ストライドを BK ではなく lda（パディング込み）にする
                        // ことで、パディングにより実際にずれた行の先頭アドレス
                        // を正しく指す（イシュー #538）。
                        //
                        // イシュー #1138: `TRANS_A` の場合 threadgroup タイルは
                        // `BK×(BM+pad)`（行=K・列=M）で確保されているため、
                        // `simdgroup_load` の `transpose_matrix=true` で
                        // K×M ブロックを M×K へ転置して読み出す（自然読み出しの
                        // 8x8 ブロックは K 方向 8・M 方向 8 で、transpose 後に
                        // A_frag の期待するレイアウト〈M 方向行・K 方向列〉と
                        // 一致する。`docs/backend-metal-transpose-collapse-design.md`
                        // §2・本ファイル冒頭 TRANS_A 宣言のコメント参照）。
                        if (TRANS_A) {
                            simdgroup_load(a_frag[r], tile_a + (size_t)kk * (size_t)lda + (size_t)(wm_idx * sub_bm + r * 8), lda, ulong2(0), true);
                        } else {
                            simdgroup_load(a_frag[r], tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)lda + (size_t)kk, lda);
                        }
                    }
                }
                // 条件付き loop unroll（B フラグメントロード版。上記 A タイル
                // 側と同じ理屈）。
                if (UNROLL_ACC_ENABLED) {
#pragma clang loop unroll(full)
                    for (uint c_ = 0; c_ < acc_cols; c_++) {
                        // ストライドを BN ではなく ldb（パディング込み）にする
                        // （上記 A タイル側と同じ理由。イシュー #538）。
                        //
                        // イシュー #1138: `TRANS_B` の場合 threadgroup タイルは
                        // `BN×(BK+pad)`（行=N・列=K）で確保されているため、
                        // 自然読み出しの 8x8 ブロック（N 方向 8・K 方向 8）を
                        // `transpose_matrix=true` で K×N へ転置し B_frag の
                        // 期待するレイアウト（K 方向行・N 方向列）に一致させる。
                        if (TRANS_B) {
                            simdgroup_load(b_frag[c_], tile_b + (size_t)(wn_idx * sub_bn + c_ * 8) * (size_t)ldb + (size_t)kk, ldb, ulong2(0), true);
                        } else {
                            simdgroup_load(b_frag[c_], tile_b + (size_t)kk * (size_t)ldb + (size_t)(wn_idx * sub_bn + c_ * 8), ldb);
                        }
                    }
                } else {
                    for (uint c_ = 0; c_ < acc_cols; c_++) {
                        // ストライドを BN ではなく ldb（パディング込み）にする
                        // （上記 A タイル側と同じ理由。イシュー #538）。
                        //
                        // イシュー #1138: `TRANS_B` の場合 threadgroup タイルは
                        // `BN×(BK+pad)`（行=N・列=K）で確保されているため、
                        // 自然読み出しの 8x8 ブロック（N 方向 8・K 方向 8）を
                        // `transpose_matrix=true` で K×N へ転置し B_frag の
                        // 期待するレイアウト（K 方向行・N 方向列）に一致させる。
                        if (TRANS_B) {
                            simdgroup_load(b_frag[c_], tile_b + (size_t)(wn_idx * sub_bn + c_ * 8) * (size_t)ldb + (size_t)kk, ldb, ulong2(0), true);
                        } else {
                            simdgroup_load(b_frag[c_], tile_b + (size_t)kk * (size_t)ldb + (size_t)(wn_idx * sub_bn + c_ * 8), ldb);
                        }
                    }
                }
                // simdgroup 細粒度同期（イシュー #809・実験的機構。本ファイル
                // 冒頭 FINE_BARRIER_ENABLED 宣言のコメント参照）。フラグメント
                // ロード完了後・MMA 発行前に挿入する MLX steel `mma.h` 型の
                // フェンス。function constant による条件のためコンパイル時
                // 畳み込みで `false`（本番既定）時の実行時コストはない。
                if (FINE_BARRIER_ENABLED) {
                    simdgroup_barrier(mem_flags::mem_none);
                }
                // 条件付き loop unroll（staged MMA 発行版）: unroll 版・非
                // unroll 版のどちらも acc[r][c_] への `simdgroup_multiply_
                // accumulate` 発行を r/c_ 昇順で行うだけで、K 方向累算
                // オペランド列は不変（数値はビット単位で従来と一致する）。
                if (UNROLL_ACC_ENABLED) {
#pragma clang loop unroll(full)
                    for (uint r = 0; r < acc_rows; r++) {
#pragma clang loop unroll(full)
                        for (uint c_ = 0; c_ < acc_cols; c_++) {
                            simdgroup_multiply_accumulate(acc[r][c_], a_frag[r], b_frag[c_], acc[r][c_]);
                        }
                    }
                } else {
                    for (uint r = 0; r < acc_rows; r++) {
                        for (uint c_ = 0; c_ < acc_cols; c_++) {
                            simdgroup_multiply_accumulate(acc[r][c_], a_frag[r], b_frag[c_], acc[r][c_]);
                        }
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
            // 条件付き loop unroll（direct-load MMA 発行版。本ファイル冒頭
            // UNROLL_ACC_ENABLED 宣言のコメント参照）: unroll 版・非 unroll 版の
            // どちらも同じ蛇行走査順（#536）で acc[r][c_] への
            // simdgroup_multiply_accumulate を発行するだけで、K 方向累算
            // オペランド列は不変（数値はビット単位で従来と一致する）。
            if (UNROLL_ACC_ENABLED) {
            #pragma clang loop unroll(full)
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
                        // イシュー #1138: `TRANS_A` の direct-load 経路は
                        // device 側 A（`[k, m]` 行優先、ld=`st.lda`）から
                        // `transpose_matrix=true` で 8x8 ブロックを転置
                        // 読み出しする（staged 経路のフラグメントロードと
                        // 同じ理屈）。
                        if (TRANS_A) {
                            simdgroup_load(a_tile, a + (size_t)(p0 + kk) * (size_t)st.lda + (size_t)a_row, st.lda, ulong2(0), true);
                        } else {
                            simdgroup_load(a_tile, a + (size_t)a_row * (size_t)st.lda + (size_t)(p0 + kk), st.lda);
                        }
                    }
#pragma clang loop unroll(full)
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
                            // イシュー #1138: `TRANS_B` の direct-load 経路は
                            // device 側 B（`[n, k]` 行優先、ld=`st.ldb`）から
                            // `transpose_matrix=true` で 8x8 ブロックを転置
                            // 読み出しする（A タイル側と同じ理屈）。
                            if (TRANS_B) {
                                simdgroup_load(b_tile, b + (size_t)b_col * (size_t)st.ldb + (size_t)(p0 + kk), st.ldb, ulong2(0), true);
                            } else {
                                simdgroup_load(b_tile, b + (size_t)(p0 + kk) * (size_t)st.ldb + (size_t)b_col, st.ldb);
                            }
                        }
                        simdgroup_multiply_accumulate(acc[r][c_], a_tile, b_tile, acc[r][c_]);
                    }
                }
            } else {
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
                        // イシュー #1138: `TRANS_A` の direct-load 経路は
                        // device 側 A（`[k, m]` 行優先、ld=`st.lda`）から
                        // `transpose_matrix=true` で 8x8 ブロックを転置
                        // 読み出しする（staged 経路のフラグメントロードと
                        // 同じ理屈）。
                        if (TRANS_A) {
                            simdgroup_load(a_tile, a + (size_t)(p0 + kk) * (size_t)st.lda + (size_t)a_row, st.lda, ulong2(0), true);
                        } else {
                            simdgroup_load(a_tile, a + (size_t)a_row * (size_t)st.lda + (size_t)(p0 + kk), st.lda);
                        }
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
                            // イシュー #1138: `TRANS_B` の direct-load 経路は
                            // device 側 B（`[n, k]` 行優先、ld=`st.ldb`）から
                            // `transpose_matrix=true` で 8x8 ブロックを転置
                            // 読み出しする（A タイル側と同じ理屈）。
                            if (TRANS_B) {
                                simdgroup_load(b_tile, b + (size_t)b_col * (size_t)st.ldb + (size_t)(p0 + kk), st.ldb, ulong2(0), true);
                            } else {
                                simdgroup_load(b_tile, b + (size_t)(p0 + kk) * (size_t)st.ldb + (size_t)b_col, st.ldb);
                            }
                        }
                        simdgroup_multiply_accumulate(acc[r][c_], a_tile, b_tile, acc[r][c_]);
                    }
                }
            }
            }
        }
    }

    // ストア: サブブロック原点（`sub_row0`/`sub_col0`）が実効次元を超える
    // 8x8 タイルは書き込みをスキップする（REQ-8。ブロック端の手動境界
    // チェックを維持する。最適化を理由に省略しない）。条件付き loop
    // unroll（イシュー #1282。本ファイル冒頭 UNROLL_ACC_ENABLED 宣言の
    // コメント参照）: unroll 版・非 unroll 版のどちらも同じ境界チェック
    // 付き `simdgroup_store` を r/c_ 昇順で行うだけで、書き込み先・値は
    // 不変（数値はビット単位で従来と一致する）。
    if (UNROLL_ACC_ENABLED) {
#pragma clang loop unroll(full)
        for (uint r = 0; r < acc_rows; r++) {
            uint out_row = sub_row0 + r * 8;
            if (out_row >= dims.m) {
                continue;
            }
#pragma clang loop unroll(full)
            for (uint c_ = 0; c_ < acc_cols; c_++) {
                uint out_col = sub_col0 + c_ * 8;
                if (out_col >= dims.n) {
                    continue;
                }
                simdgroup_store(acc[r][c_], c + (size_t)out_row * (size_t)dims.n + (size_t)out_col, dims.n);
            }
        }
    } else {
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
}

// half 版タイル化 GEMM 本体（イシュー #796。協調ロードの 8 要素ベクトル化・
// エピローグのタイル粒度統合はイシュー #797 で実施済み）: `gemm_simdgroup_tiled`
// （f32 版・BM/BN/BK/WM/WN・協調ロード＋レジスタ常駐フラグメント構成）を
// half 入力へ移植する。既存 `gemm_simdgroup_f16`（1 threadgroup =
// 1 simdgroup = C の 8x8 タイル 1 つの非タイル化構造）が対 PyTorch MPS
// f16 で大きく劣後する主因（親イシュー #787）に対応する。
//
// **スコープ境界**: 協調ロードの 8 要素（128bit）ベクトル化と、エピローグの
// barrier 粒度をタイル単位からサブタイル全体単位へ統合する対応は #797 で
// 完了済み（USE_TGP_STAGING 分岐・エピローグ実装の各コメント参照）。動的
// タイル選択（`tile::select`/`dispatch_auto`）への統合・バックエンド間
// 数値一致回帰テストの拡張は #798、実機再計測・ベースライン更新は #799。
//
// **function constant の再利用**: `BM`/`BN`/`BK`/`WM`/`WN`/
// `USE_TGP_STAGING`/`TGP_PAD`/`SWIZZLE_ENABLED`（index 0〜7。本ファイル
// 上方の `gemm_simdgroup_tiled` 直前の宣言）はファイルスコープの
// `constant` 宣言のため、本カーネルも再宣言せずそのまま参照できる。
// `crate::pipeline::make_pipeline_with_constants` は `function_name` 引数
// （`"gemm_simdgroup_tiled_f16"`）を切り替えるだけで同じ `TileConfig` から
// 特殊化パイプラインを構築できるため、Rust 側の変更が最小で済む（イシュー
// #796 計画「Rust 側」節）。
//
// **型**: A/B のタイル型は `MM_T`（`simdgroup_half8x8`）、アキュムレータは
// `ACC_T`（`simdgroup_float8x8`。f32 累算）を `gemm_simdgroup_f16`
// （本ファイル上方）の typedef からそのまま再利用する。混在オーバーロード
// `simdgroup_multiply_accumulate(simdgroup_float8x8&, simdgroup_half8x8,
// simdgroup_half8x8, simdgroup_float8x8)` の実在・`simdgroup_store` が
// float→half 直接変換をサポートしない制約は #380 実機 spike で確定済み
// （`gemm_simdgroup_f16` 冒頭コメント参照）。
//
// **threadgroup 共有メモリのレイアウト**: 先頭に staged 経路の A タイル
// （BM×(BK+TGP_PAD)。half）・B タイル（BK×(BN+TGP_PAD)。half）、直後に
// エピローグ staging 領域（f32。WM*WN simdgroup 分、各 simdgroup が担当
// サブタイル全体〈(BM/WM)*(BN/WN) 要素〉分。イシュー #797 でタイル粒度へ
// 拡大。合計 BM*BN 要素）を続ける。`USE_TGP_STAGING=false`（direct-load
// 経路。フォールバック終端 `TileConfig::SINGLE_SIMDGROUP_8X8` を含む）
// ではタイル領域を確保しない（`crate::tile::TileConfig::shared_mem_bytes_f16`
// 参照）ため、エピローグ領域の先頭オフセットは `USE_TGP_STAGING` の値で
// 分岐して決める（`shared_mem` の実確保長は
// `TileConfig::shared_mem_bytes_f16().max(16)` を呼び出し元がエンコード時に
// 指定する契約。`crate::gemm::encode_dispatch_tiled_f16` 参照）。A タイル＋
// B タイルの要素数（half 単位）は BM/BK/BN がいずれも 8 の倍数・TGP_PAD が
// 0 または 4（`TileConfig::validate` が保証する不変条件）のため常に偶数
// であり、half 2 バイト単位で確保しても続くエピローグ領域は f32 4 バイト
// 境界に整合する（`reinterpret_cast<threadgroup float*>` で安全に参照できる
// 根拠。`TileConfig::shared_mem_bytes_f16` ドキュメントコメント参照）。
//
// **手動境界チェック（REQ-8。省略禁止）**: ブロック原点の早期 return・
// staged ロード時の要素単位 in-bounds 判定＋0 埋め・direct-load 時の
// 行/列ガード＋0 埋めタイル・エピローグ書き戻し時の要素単位 in-bounds
// 判定を、f32 版 `gemm_simdgroup_tiled`・`gemm_simdgroup_f16` と同じ設計で
// 維持する。バッファオフセットは `size_t` へ昇格してから乗算する
// （PR #246 Bugbot 指摘の系譜。両カーネルの既存コメント参照）。
//
// **数値契約**: `dims` は呼び出し元が `pad8` で 8 の倍数へ揃えた実効次元
// （`gemm_simdgroup_tiled` と同じ契約）。累算は f32・K 昇順で、CPU 参照
// （f32 累算 → 最後に 1 回だけ f16 丸め）・CUDA WMMA f16
// （`f32.f16.f16.f32`）との精度契約に整合する（`gemm_simdgroup_f16` 冒頭
// コメント参照）。
kernel void gemm_simdgroup_tiled_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant Dims& dims [[buffer(3)]],
    threadgroup half* shared_mem [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // threadgroup ID スウィズル: `gemm_simdgroup_tiled`（f32 版）と同一の
    // 恒等/スウィズル変換（`SWIZZLE_ENABLED` 分岐・イシュー #540）。本番
    // 既定 `false` では `tid_y = tgid.y`・`tid_x = tgid.x`（恒等）になる
    // （同ファイル `gemm_simdgroup_tiled` コメント参照）。
    constexpr uint SWIZZLE_LOG = 2;
    constexpr uint SWIZZLE_TILE = 1u << SWIZZLE_LOG;
    uint tid_y = SWIZZLE_ENABLED ? ((tgid.y << SWIZZLE_LOG) + (tgid.x & (SWIZZLE_TILE - 1))) : tgid.y;
    uint tid_x = SWIZZLE_ENABLED ? (tgid.x >> SWIZZLE_LOG) : tgid.x;

    uint row0 = tid_y * BM;
    uint col0 = tid_x * BN;

    // REQ-8: ブロック全体が実効次元を完全に超える場合は早期 return する
    // （`gemm_simdgroup_tiled` と同一の判断根拠）。
    if (tiled_block_out_of_range(row0, col0, dims)) {
        return;
    }

    uint wm_idx = simd_id / WN;
    uint wn_idx = simd_id % WN;
    uint sub_bm = BM / WM;
    uint sub_bn = BN / WN;
    uint sub_row0 = row0 + wm_idx * sub_bm;
    uint sub_col0 = col0 + wn_idx * sub_bn;

    // `gemm_simdgroup_tiled`（f32 版）と同一の固定上限（`TileConfig::MAX_ACC`
    // と 1:1 対応。`TileConfig::validate` が dispatch 前に超過構成を拒否する
    // 契約も同一）。
    constexpr uint MAX_ACC = 8;
    uint acc_rows = sub_bm / 8;
    uint acc_cols = sub_bn / 8;
    ACC_T acc[MAX_ACC][MAX_ACC];
    for (uint r = 0; r < acc_rows; r++) {
        for (uint c_ = 0; c_ < acc_cols; c_++) {
            acc[r][c_] = ACC_T(0.0f);
        }
    }

    uint lda = BK + TGP_PAD;
    uint ldb = BN + TGP_PAD;
    threadgroup half* tile_a = shared_mem;
    threadgroup half* tile_b = shared_mem + (size_t)BM * (size_t)lda;
    // エピローグ staging 領域の先頭オフセットは USE_TGP_STAGING で分岐する
    // （direct-load 経路では shared_mem にタイル領域が確保されないため。
    // 上方レイアウトコメント参照。`tile_a`/`tile_b` 自体は direct-load
    // 経路でも計算されるが、その経路ではポインタ演算のみで実際に読み書き
    // されることはない。f32 版 `gemm_simdgroup_tiled` の同名変数と同じ
    // パターン）。
    threadgroup float* stage = USE_TGP_STAGING
        ? reinterpret_cast<threadgroup float*>(tile_b + (size_t)BK * (size_t)ldb)
        : reinterpret_cast<threadgroup float*>(shared_mem);

    uint k_full_tiles = dims.k / BK;
    uint k_tail = dims.k - k_full_tiles * BK; // BK の倍数でない末尾（0 埋め扱い）
    uint k_tile_count = k_full_tiles + (k_tail > 0 ? 1 : 0);

    for (uint t = 0; t < k_tile_count; t++) {
        uint p0 = t * BK;
        // 末尾タイルが BK に満たない場合の有効幅（境界チェック。REQ-8）。
        uint bk_eff = min(BK, dims.k - p0);

        if (USE_TGP_STAGING) {
            // 協調ロード（8 要素ベクトル化。イシュー #797。f32 版
            // `gemm_simdgroup_tiled` の float4〈128bit・4 要素〉ベクトル
            // ロード〈#533/#538〉と同じ構造を half 8 要素幅〈128bit〉へ
            // 移植する: threadgroup 内の全スレッド（WM*WN*32 個）が
            // 8 要素グループ単位で A タイル（BM*BK 要素）・B タイル
            // （BK*BN 要素）を分担してロードする。
            //
            // **アラインメント成立根拠**（f32 版 float4 ロードのコメントと
            // 同型の論拠。VEC_WIDTH_F16=8 版）:
            //   - `dims.k`/`dims.n` は呼び出し元が `crate::pad::pad8` で
            //     8 の倍数へ実効次元パディング済みで渡す契約
            //     （`crate::gemm::MetalGemm::dispatch_auto` 参照）
            //   - `p0`/`col0` は BK・BN のタイル境界（`TileConfig::validate`
            //     が `bk % 8 == 0`・`bn % (wn*8) == 0` ⟹ `bn % 8 == 0` を
            //     dispatch 前に fail-closed 保証）のため常に 8 の倍数
            //   - よって device 側の読み出し先頭オフセットは常に 8 half
            //     要素（16 バイト）境界に揃う（MTLBuffer 先頭はページ
            //     境界確保）
            // これにより `device const float4*` へ再解釈した 128bit（half
            // 8 要素分のビット幅）一括ロードが安全に成立する（数値変換は
            // 発生しないビットコピー）。
            //
            // **threadgroup 側は 4 要素（8 バイト）境界までしか保証されない**:
            // `lda = BK + TGP_PAD`・`ldb = BN + TGP_PAD` の `TGP_PAD` は
            // half 4 要素（8 バイト）単位（`TileConfig::TGP_PAD_ELEMS=4`）
            // のため、書き込み先添字は 16 バイト境界を跨ぐ書き込み先へ
            // 整合しない。よって 128bit 単発 store ではなく
            // `as_type<half4>` で分解した **2 回の half4（8 バイト境界）
            // store** とする（TGP_PAD の 16 バイト境界化・occupancy 影響は
            // #799 の計測課題）。
            //
            // **境界チェック（REQ-8。省略禁止）**: グループ全 8 要素の
            // in-bounds を明示判定し、境界グループは範囲外アドレスへ一切
            // 触れず要素単位スカラー読み出し + 0 埋めへフォールバックする
            // （f32 版 float4 ロードと同じ二層防御。上記
            // `TileConfig::validate` の dispatch 前検査が構成レベルで、
            // 本判定が実行時の端数タイルレベルで、それぞれ担保する）。
            // 共有メモリへ格納される値は変更前スカラーループとビット単位で
            // 一致する（格納順・添字導出は不変、ロード命令幅のみ変更のため。
            // 以降の `simdgroup_load`／MMA 発行順も不変）。
            uint local_tid = simd_id * 32 + simd_lane;
            uint threads_total = WM * WN * 32;

            uint a_vecs = (BM * BK) / 8;
            for (uint vi = local_tid; vi < a_vecs; vi += threads_total) {
                uint idx = vi * 8;
                uint r = idx / BK;
                uint kk = idx % BK;
                uint dst_idx = r * lda + kk; // パディング込みの書き込み先添字。
                uint global_row = row0 + r;
                uint global_k = p0 + kk;
                bool group_in_bounds = tiled_a_group_in_bounds(kk, bk_eff, global_row, global_k, 8, dims);
                if (group_in_bounds) {
                    device const float4* src = reinterpret_cast<device const float4*>(
                        a + (size_t)global_row * (size_t)dims.k + (size_t)global_k);
                    float4 v = *src; // half8 を float4（128bit）としてビットコピー。
                    threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(tile_a + dst_idx);
                    dst[0] = as_type<half4>(v.xy);
                    dst[1] = as_type<half4>(v.zw);
                } else {
                    for (uint e = 0; e < 8; e++) {
                        uint kk_e = kk + e;
                        uint global_k_e = global_k + e;
                        tile_a[dst_idx + e] = tiled_a_elem_in_bounds(kk_e, bk_eff, global_row, global_k_e, dims)
                            ? a[(size_t)global_row * (size_t)dims.k + (size_t)global_k_e]
                            : half(0.0h);
                    }
                }
            }

            uint b_vecs = (BK * BN) / 8;
            for (uint vi = local_tid; vi < b_vecs; vi += threads_total) {
                uint idx = vi * 8;
                uint kk = idx / BN;
                uint c_ = idx % BN;
                uint dst_idx = kk * ldb + c_; // パディング込みの書き込み先添字。
                uint global_k = p0 + kk;
                uint global_col = col0 + c_;
                bool group_in_bounds = tiled_b_group_in_bounds(kk, bk_eff, global_k, global_col, 8, dims);
                if (group_in_bounds) {
                    device const float4* src = reinterpret_cast<device const float4*>(
                        b + (size_t)global_k * (size_t)dims.n + (size_t)global_col);
                    float4 v = *src;
                    threadgroup half4* dst = reinterpret_cast<threadgroup half4*>(tile_b + dst_idx);
                    dst[0] = as_type<half4>(v.xy);
                    dst[1] = as_type<half4>(v.zw);
                } else {
                    for (uint e = 0; e < 8; e++) {
                        uint c_e = c_ + e;
                        uint global_col_e = global_col + e;
                        tile_b[dst_idx + e] = tiled_b_elem_in_bounds(kk, bk_eff, global_k, global_col_e, dims)
                            ? b[(size_t)global_k * (size_t)dims.n + (size_t)global_col_e]
                            : half(0.0h);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kk = 0; kk < BK; kk += 8) {
                // フラグメントのレジスタ常駐化（`gemm_simdgroup_tiled`
                // ・イシュー #745 と同じ構造。MLX steel `mma.h` の
                // tile_matmad 型）: kk ステップ先頭で A の acc_rows 個・
                // B の acc_cols 個の simdgroup フラグメントを一括ロードして
                // から TM×TN の外積 MMA を発行する。
                MM_T a_frag[MAX_ACC];
                MM_T b_frag[MAX_ACC];
                for (uint r = 0; r < acc_rows; r++) {
                    simdgroup_load(a_frag[r], tile_a + (size_t)(wm_idx * sub_bm + r * 8) * (size_t)lda + (size_t)kk, lda);
                }
                for (uint c_ = 0; c_ < acc_cols; c_++) {
                    simdgroup_load(b_frag[c_], tile_b + (size_t)kk * (size_t)ldb + (size_t)(wn_idx * sub_bn + c_ * 8), ldb);
                }
                for (uint r = 0; r < acc_rows; r++) {
                    for (uint c_ = 0; c_ < acc_cols; c_++) {
                        simdgroup_multiply_accumulate(acc[r][c_], a_frag[r], b_frag[c_], acc[r][c_]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // 直接ロード: `gemm_simdgroup_tiled` の direct-load 経路と同一
            // 構造（蛇行〈serpentine〉走査・8 の倍数保証による境界チェック
            // 省略可の論拠も同一。f32 版コメント参照）。
            uint bk_full8 = (bk_eff / 8) * 8;
            for (uint kk = 0; kk < bk_full8; kk += 8) {
                for (uint r = 0; r < acc_rows; r++) {
                    uint a_row = sub_row0 + r * 8;
                    MM_T a_tile = MM_T(0.0h);
                    if (a_row < dims.m) {
                        simdgroup_load(a_tile, a + (size_t)a_row * (size_t)dims.k + (size_t)(p0 + kk), dims.k);
                    }
                    for (uint ci = 0; ci < acc_cols; ci++) {
                        uint c_ = (r % 2 == 1) ? (acc_cols - 1 - ci) : ci;
                        uint b_col = sub_col0 + c_ * 8;
                        MM_T b_tile = MM_T(0.0h);
                        if (b_col < dims.n) {
                            simdgroup_load(b_tile, b + (size_t)(p0 + kk) * (size_t)dims.n + (size_t)b_col, dims.n);
                        }
                        simdgroup_multiply_accumulate(acc[r][c_], a_tile, b_tile, acc[r][c_]);
                    }
                }
            }
        }
    }

    // エピローグ（3 段・タイル粒度統合。イシュー #797。#380 で確定した
    // 変種 B〈f32 のまま staging → 最後に 1 回だけ half 変換〉を維持し
    // つつ、barrier 粒度のみをタイル単位からサブタイル全体単位へ統合する）:
    // `simdgroup_store(simdgroup_float8x8, device half*)` はコンパイル不可
    // （#380 spike で確認済み）なため、まず各 simdgroup 専用の staging
    // スラブ（`sub_bm*sub_bn` 要素。担当サブタイル全体分）へ全 acc タイルを
    // f32 のまま一括 store し、`simdgroup_barrier` で 1 回だけ同期してから
    // 32 レーンがサブタイル全体を half へ変換して `c` へ書き戻す。
    //
    // #796 時点は acc タイル（8x8）1 個ごとに store→barrier→書き戻し→
    // barrier を回しており、barrier が 1 simdgroup あたり
    // `2 * acc_rows * acc_cols` 回発生していた。各 simdgroup は自スラブの
    // みを読み書きするため他 simdgroup との競合はなく、サブタイル全体を
    // 一括 store してから読み出す構成に変えても正しさは変わらない
    // （store→barrier→読み出しの順序自体は不変。read-after-write の
    // 依存関係は 1 回の barrier で満たされる）。よって barrier は
    // 1 simdgroup あたり 1 回まで削減できる（書き戻し後に次の store が
    // 続かないため trailing barrier も不要。自 simdgroup 内の同期で足りる
    // ため `threadgroup_barrier` も不要）。
    threadgroup float* my_stage = stage + simd_id * (sub_bm * sub_bn);
    for (uint r = 0; r < acc_rows; r++) {
        for (uint c_ = 0; c_ < acc_cols; c_++) {
            // ストライドを `sub_bn`（担当サブタイルの列幅）にすることで、
            // acc タイル群をサブタイル内の正しい位置へ隙間なく配置する。
            simdgroup_store(acc[r][c_], my_stage + (size_t)(r * 8) * (size_t)sub_bn + (size_t)(c_ * 8), sub_bn);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // REQ-8: 要素単位の境界チェック（`out_row`/`out_col` が実効次元を
    // 超える要素への書き込みをスキップする）を維持する。タイル原点の
    // 早期 continue は不要（staging はローカル領域で安全に読み書きできる。
    // 実効次元外への書き込みは本判定で完全に代替される）。
    for (uint i = simd_lane; i < sub_bm * sub_bn; i += 32u) {
        uint rr = i / sub_bn;
        uint cc = i % sub_bn;
        uint out_row = sub_row0 + rr;
        uint out_col = sub_col0 + cc;
        if (out_row < dims.m && out_col < dims.n) {
            c[(size_t)out_row * (size_t)dims.n + (size_t)out_col] = (half)my_stage[i];
        }
    }
}
