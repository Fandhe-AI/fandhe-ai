//! `sort`／`topk`（`torch.sort`／`torch.topk` 相当。イシュー #1741）の
//! CUDA C カーネルソース（NVRTC 実行時コンパイル用の静的文字列）。
//!
//! `sort.rs`（呼び出し元）は本モジュールの 3 定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels_unique.rs` と
//! 同じ理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む
//! （ビルド時に nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit
//! 非搭載環境でも `cargo build --workspace` が成立する」契約を維持
//! する。`.claude/rules/deps-policy.md`）。
//!
//! アルゴリズム・鍵設計・ライン分解は [`crate::sort_model`] モジュール
//! doc が正であり、本モジュールの 3 カーネルは同モジュールの
//! `value_key`／`composite_key`／`line_layout` と数学的に同一の処理を
//! GPU 側で逐語的に再実装したもの（同モジュールの
//! `#[cfg(test)]` ホストモデルが両者の一致を Linux 実行可能テストで
//! 検証する）。
//!
//! # 3 カーネルの役割
//!
//! - [`SORT_BUILD_KEYS_U64`]（`sort_build_keys_u64`）: `input`（行優先
//!   contiguous）から各ライン `outer*inner` 本ぶんの 64bit 合成キー
//!   配列（`lines * padded` 長）を構築する。`i < dim_size` の位置は
//!   実データのキー、`i >= dim_size`（パディング域）は
//!   `0xFFFFFFFFFFFFFFFF`（全実キーより大きい）を書く。
//! - [`BITONIC_STEP_U64`]（`bitonic_step_u64`）: 標準的なビットニック
//!   ソートの 1 比較ステップを、ライン単位（`padded` 長ごと）に独立に
//!   適用する。整数 compare/swap のみで浮動小数点演算を含まないため
//!   決定的（`kernels_unique.rs::BITONIC_STEP_U32` と同じ設計）。
//! - [`SORT_FINALIZE_F32`]（`sort_finalize_f32`）: 各ラインの先頭
//!   `out_len` 要素（`sort` は `out_len == dim_size`、`topk` は
//!   `out_len == k`）を読み、キー下位 32bit（元添字）で `input` を
//!   gather して `values`／`index` へ書く。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルで自スレッドの担当範囲（`gid < total`）・`in_pos`／
//! `out_pos` が対応するバッファの確保長（`numel_in`／`numel_out`）に
//! 収まることを検査する。添字演算は `long long`（`kernels_reduce.rs`
//! の教訓と同じ理由。`i32::MAX` 近傍でのオーバーフロー回避）で行う。
//! 最適化を理由に境界検査を省略しない。
//!
//! # 決定性
//!
//! `sort_build_keys_u64`・`bitonic_step_u64` は整数演算のみ（浮動小数点
//! 比較は `isnan`／`==` の 2 種のみで、いずれも NVRTC の高速数学モード
//! に非依存）。`sort_finalize_f32` は `input` の要素を無演算で読み出す
//! だけ（`values` は算術演算を含まない gather）。全体として run-to-run
//! で bit 同一の出力を返す（`fandhe_ai_tensor_core::BackendOps::sort`
//! doc の順序契約 4）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_unique::
/// UNIQUE_BLOCK_DIM` と同じ値・同じ理由）。
pub const SORT_BLOCK_DIM: u32 = 256;

/// `sort_build_keys_u64`（本モジュール doc 参照）。
///
/// 引数: `input`（行優先 contiguous な `dim_size` 軸を含むテンソル）・
/// `keys`（出力。`lines * padded` 長）・`lines`・`dim_size`・`inner`・
/// `padded`・`descending`（0/1）・`numel_in`（`input` の確保長。境界
/// 検査用）。
pub const SORT_BUILD_KEYS_U64: &str = r#"
extern "C" __global__ void sort_build_keys_u64(
    const float* __restrict__ input,
    unsigned long long* __restrict__ keys,
    long long lines,
    long long dim_size,
    long long inner,
    long long padded,
    int descending,
    long long numel_in)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lines * padded;
    if (gid >= total) {
        return;
    }
    long long line = gid / padded;
    long long i = gid % padded;
    long long out_pos = line * padded + i;
    if (i < dim_size) {
        long long in_pos = (line / inner) * dim_size * inner + (line % inner) + i * inner;
        if (in_pos < 0 || in_pos >= numel_in) {
            // 防御的検査（ホスト側の line_layout 計算が正しければ到達
            // 不能。REQ-8 の縦深防御）。
            return;
        }
        float v = input[in_pos];
        unsigned int vk;
        if (isnan(v)) {
            vk = 0xFFFFFFFFu;
        } else {
            float vv = (v == 0.0f) ? 0.0f : v;
            unsigned int bits = __float_as_uint(vv);
            vk = (bits >> 31) ? (~bits) : (bits | 0x80000000u);
        }
        unsigned int hi = descending ? (~vk) : vk;
        unsigned int lo = (unsigned int)i;
        keys[out_pos] = (((unsigned long long)hi) << 32) | (unsigned long long)lo;
    } else {
        keys[out_pos] = 0xFFFFFFFFFFFFFFFFULL;
    }
}
"#;

/// `bitonic_step_u64`（本モジュール doc 参照）。`j`／`k` は呼び出し元
/// （`sort.rs`）がホスト側ループで管理するステップパラメータ。各ライン
/// （`padded` 長ごと）へ独立にビットニックソートの 1 比較ステップを
/// 適用する（`lines` 本すべてを 1 回のカーネル起動で処理する）。
pub const BITONIC_STEP_U64: &str = r#"
extern "C" __global__ void bitonic_step_u64(
    unsigned long long* keys,
    long long j,
    long long k,
    long long padded,
    long long lines)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lines * padded;
    if (gid >= total) {
        return;
    }
    long long line = gid / padded;
    long long i = gid % padded;
    long long ixj = i ^ j;
    if (ixj <= i || ixj >= padded) {
        return;
    }
    long long base = line * padded;
    unsigned long long a = keys[base + i];
    unsigned long long b = keys[base + ixj];
    bool ascending = (i & k) == 0;
    bool should_swap = ascending ? (a > b) : (a < b);
    if (should_swap) {
        keys[base + i] = b;
        keys[base + ixj] = a;
    }
}
"#;

/// `sort_finalize_f32`（本モジュール doc 参照）。
///
/// 引数: `input`（`sort_build_keys_u64` と同一）・`keys`（ソート済み。
/// `lines * padded` 長）・`values`／`index`（出力。`lines * out_len`
/// 長）・`lines`・`dim_size`・`inner`・`padded`・`out_len`・`numel_in`・
/// `numel_out`（境界検査用）。
pub const SORT_FINALIZE_F32: &str = r#"
extern "C" __global__ void sort_finalize_f32(
    const float* __restrict__ input,
    const unsigned long long* __restrict__ keys,
    float* __restrict__ values,
    int* __restrict__ index,
    long long lines,
    long long dim_size,
    long long inner,
    long long padded,
    long long out_len,
    long long numel_in,
    long long numel_out)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = lines * out_len;
    if (gid >= total) {
        return;
    }
    long long line = gid / out_len;
    long long o = gid % out_len;
    unsigned long long key = keys[line * padded + o];
    unsigned long long idxu = key & 0xFFFFFFFFULL;
    if (idxu >= (unsigned long long)dim_size) {
        // 防御的検査（ホスト側検証で到達不能な内部契約違反。REQ-8）。
        return;
    }
    long long idx = (long long)idxu;
    long long in_pos = (line / inner) * dim_size * inner + (line % inner) + idx * inner;
    long long out_pos = (line / inner) * out_len * inner + (line % inner) + o * inner;
    if (in_pos < 0 || in_pos >= numel_in || out_pos < 0 || out_pos >= numel_out) {
        return;
    }
    values[out_pos] = input[in_pos];
    index[out_pos] = (int)idx;
}
"#;
