//! `gemm_variant.rs`（イシュー #1035）が選択する DoubleBuffer / SplitK
//! カーネルソース。**opt-in・未計測の実験実装**であり、`gemm.rs` の
//! `internal-diagnostics` feature 限定コンストラクタ
//! （`new_with_f32_variant_selection`）からのみ NVRTC コンパイルされる。
//! 本番既定コンストラクタ（`CudaGemm::new`）・`kernels::TILED_F32` は
//! 一切変更しない（実装計画 §3・§8）。
//!
//! # 対応するカーネル
//!
//! - [`TILED_DB_F32`]: `kernels::TILED_F32` の smem 2 面（double-buffer）
//!   プリフェッチ版。cp.async は使わない（#1033 の cp.async 多段
//!   パイプラインとスコープを分離するための設計判断。実装計画 §3 の 3
//!   番）。
//! - [`SPLITK_PARTIAL_F32`]／[`SPLITK_REDUCE_F32`]: K 方向分割 GEMM の
//!   2 段構成（部分和 + 決定的縮約。`kernels_rmsnorm.rs` の dw split-K
//!   〈#597〉と同型で atomics を使わない）。
//!
//! いずれも REQ-8（カーネル実装の境界検査）に従い、手動境界チェックを
//! 省略しない。

/// [`TILED_DB_F32`] のタイル一辺（`kernels::TILE` と同じ値。
/// ホスト側〈`gemm.rs`〉の起動パラメータの単一の真実源はこの定数ではなく
/// `kernels::TILE` だが、本カーネルは `kernels::TILE` と同じ 32 を前提に
/// 手書きしているため、値変更時は両方を同時に見直す必要がある
/// （`kernels.rs::TILE` ドキュメンテーションコメント参照）。
pub const TILED_DB_TILE: u32 = 32;

/// tiled GEMM（f32）の smem 2 面（double-buffer）プリフェッチ版
/// （イシュー #1035）。
///
/// [`crate::kernels::TILED_F32`] は「タイルをロード → `__syncthreads()`
/// → 積和 → `__syncthreads()`」を K タイルごとに直列に繰り返すため、
/// ロードのレイテンシと積和の演算がオーバーラップしない。本カーネルは
/// smem を 2 面（`as_tile[2][TILE][TILE]`・`bs_tile[2][TILE][TILE]`）
/// 持ち、「次タイルのロードを発行してから今タイルの積和を行う」順に
/// 進めることでロードと演算をソフトウェアパイプライン化する
/// （`kernels_rmsnorm.rs::RMSNORM_BWD_DW_REDUCE_F32` の smem double
/// buffer と同じ考え方。cp.async は使わず通常のグローバルロード命令の
/// 発行順序のみで重ねる——#1033 の cp.async 多段パイプラインとスコープを
/// 分ける設計判断。実装計画 §3 の 3 番）。
///
/// # 手動境界チェック（REQ-8）
///
/// [`crate::kernels::TILED_F32`] と同じ三項ガード（末尾タイルの範囲外
/// 読み出しを 0.0f 充填で防ぐ）をプロローグ・本ループの両方のロードに
/// 適用する。`num_tiles` は `kernels::TILED_F32` と同じ桁溢れしない式
/// （`(k > 0) ? (k - 1) / TILE + 1 : 0`）を使う。
///
/// # 呼び出し元
///
/// `gemm.rs::CudaGemm::new_with_f32_variant_selection`
/// （`internal-diagnostics` feature 限定）が NVRTC コンパイルし、
/// `gemm_tiled_db_f32` エントリポイントをロードする。本番既定経路
/// （`run_tiled_f32`）からは呼ばれない。
pub const TILED_DB_F32: &str = r#"
#define TILE 32

extern "C" __global__ void gemm_tiled_db_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    __shared__ float as_tile[2][TILE][TILE];
    __shared__ float bs_tile[2][TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // 桁溢れしない num_tiles 計算（kernels::TILED_F32 と同一式）。
    int num_tiles = (k > 0) ? (k - 1) / TILE + 1 : 0;
    if (num_tiles == 0) {
        if (row < m && col < n) {
            c[row * n + col] = 0.0f;
        }
        return;
    }

    // プロローグ: タイル 0 を面 0 へロードする。REQ-8 三項ガード。
    {
        int a_col = threadIdx.x;
        int b_row = threadIdx.y;
        as_tile[0][threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        bs_tile[0][threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    }
    __syncthreads();

    int buf = 0;
    for (int t = 0; t < num_tiles; ++t) {
        int next_buf = buf ^ 1;
        // (1) 次タイルの global ロードを発行する（今タイルの積和より先に
        // コンパイラへ提示しレイテンシを重ねる）。末尾タイルでは発行しない。
        if (t + 1 < num_tiles) {
            int a_col = (t + 1) * TILE + threadIdx.x;
            int b_row = (t + 1) * TILE + threadIdx.y;
            as_tile[next_buf][threadIdx.y][threadIdx.x] =
                (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
            bs_tile[next_buf][threadIdx.y][threadIdx.x] =
                (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
        }

        // (2) 今タイルの積和。
#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += as_tile[buf][threadIdx.y][p] * bs_tile[buf][p][threadIdx.x];
        }
        __syncthreads();
        buf = next_buf;
    }

    // REQ-8: C への書き込み時の手動境界チェック（kernels::TILED_F32 と同じ）。
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}
"#;

/// split-K GEMM 第 1 カーネル（部分和生成。イシュー #1035）。
///
/// K 方向を `gridDim.z`（= `num_splits`）個の CTA へ分割し、各 `(bz)` が
/// 担当する K 範囲 `[bz*k_per_split, min((bz+1)*k_per_split, k))`
/// （`k_per_split = ceil(k / num_splits)`）についてのみ
/// [`crate::kernels::TILED_F32`] と同じ tiled 積和を行い、結果を
/// `c_partial[bz * m * n + row * n + col]` へ**一意に**書く（atomics
/// 不使用。各 `(bz, row, col)` の書き手は 1 CTA のみのため決定的。
/// `kernels_rmsnorm.rs::RMSNORM_BWD_DW_PARTIAL_F32` と同型の設計）。
///
/// # 末尾要素ブロックの扱い（REQ-8・決定的性）
///
/// `bz*k_per_split >= k` となる末尾分割（`num_splits` が `k` を割り切ら
/// ない場合に生じる）は K 範囲が空になるが、`acc = 0.0f` のまま
/// **無条件に** `c_partial` へ書く（`RMSNORM_BWD_DW_PARTIAL_F32` と同じ
/// 「早期 return せず全要素を書く」fail-closed 方針。`alloc_zeros` の
/// ゼロ初期化に依存しない）。
///
/// # 呼び出し元
///
/// `gemm.rs::CudaGemm::new_with_f32_variant_selection`
/// （`internal-diagnostics` feature 限定）。`c_partial` バッファサイズは
/// `gemm_variant::validate_split_k_launch` が起動前に cap 検査する。
pub const SPLITK_PARTIAL_F32: &str = r#"
#define TILE 32

extern "C" __global__ void gemm_splitk_partial_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c_partial,
    int m, int n, int k,
    int num_splits)
{
    __shared__ float as_tile[TILE][TILE];
    __shared__ float bs_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int bz = blockIdx.z;

    // k_per_split・k_start・k_end は long long（本ファイル冒頭コメントと
    // 同じくオーバーフロー安全のため。k/num_splits は int 範囲内だが、
    // 後続の bz * k_per_split の乗算を long long で行う）。
    long long k_per_split = ((long long)k + num_splits - 1) / num_splits;
    long long k_start = (long long)bz * k_per_split;
    long long k_end = k_start + k_per_split;
    if (k_end > k) {
        k_end = k;
    }

    float acc = 0.0f;
    if (k_start < k_end) {
        long long local_k = k_end - k_start;
        // 桁溢れしない num_tiles 計算（kernels::TILED_F32 と同じ式を
        // local_k に適用）。
        long long num_tiles = (local_k > 0) ? (local_k - 1) / TILE + 1 : 0;
        for (long long t = 0; t < num_tiles; ++t) {
            long long a_col = k_start + t * TILE + threadIdx.x;
            long long b_row = k_start + t * TILE + threadIdx.y;

            // REQ-8: タイルロード時の手動境界チェック（三項ガード）。
            // a_col/b_row は [k_start, k_end) の範囲内でなければ 0 埋め。
            as_tile[threadIdx.y][threadIdx.x] =
                (row < m && a_col < k_end) ? a[(long long)row * k + a_col] : 0.0f;
            bs_tile[threadIdx.y][threadIdx.x] =
                (b_row < k_end && col < n) ? b[b_row * n + col] : 0.0f;
            __syncthreads();

#pragma unroll
            for (int p = 0; p < TILE; ++p) {
                acc += as_tile[threadIdx.y][p] * bs_tile[p][threadIdx.x];
            }
            __syncthreads();
        }
    }

    // 無条件書き出し（末尾の空分割も含め全要素を必ず書く。本ファイル
    // 冒頭コメント「末尾要素ブロックの扱い」参照）。REQ-8: 範囲外
    // (row, col) は書かない。
    if (row < m && col < n) {
        c_partial[(long long)bz * m * n + (long long)row * n + col] = acc;
    }
}
"#;

/// [`SPLITK_REDUCE_F32`] のブロックあたりスレッド数（1 スレッド = 1
/// `(row, col)` 出力要素の縮約担当）。`kernels_rmsnorm.rs` の縮約カーネル
/// と異なり本カーネルは smem を使わず各スレッドがレジスタのみで
/// `num_splits` 回の逐次加算を行う（`num_splits <=
/// gemm_variant::SPLITK_MAX_SPLITS`〈32〉と小さく、smem 二段パイプライン
/// を要するほどの反復回数ではないための簡略化）。
pub const SPLITK_REDUCE_BLOCK_DIM: u32 = 256;

/// split-K GEMM 第 2 カーネル（縮約。イシュー #1035）。
///
/// [`SPLITK_PARTIAL_F32`] が書いた `[num_splits, m, n]` 形状の部分和
/// バッファを `num_splits` 次元方向に**固定順序**（`s = 0..num_splits`
/// の昇順）で縮約し、最終 `c` へ 1 回だけ書く（atomics 不使用・順序
/// 固定により決定的。複合判定・FMA 契約の枠内で許容誤差を変えない）。
///
/// # 手動境界チェック（REQ-8）
///
/// `idx`（= `row * n + col`）が `m * n` を超えるスレッドは早期 return
/// する（block-stride ループのため `__syncthreads()` 等のブロック同期
/// プリミティブは使わない——本カーネルは smem を使わないため
/// `kernels_rmsnorm.rs::RMSNORM_BWD_DW_REDUCE_F32` のような「全スレッドが
/// バリアへ到達しなければならない」制約が存在しない）。
pub const SPLITK_REDUCE_F32: &str = r#"
extern "C" __global__ void gemm_splitk_reduce_f32(
    const float* __restrict__ c_partial,
    float* __restrict__ c,
    int m, int n,
    int num_splits)
{
    long long total = (long long)m * (long long)n;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        // 固定順序（s 昇順）の逐次加算。decomposeせずレジスタのみで
        // 完結するため中間結果を HBM へ書き戻す第 3 パスを作らない。
        for (int s = 0; s < num_splits; ++s) {
            acc += c_partial[(long long)s * total + idx];
        }
        c[idx] = acc;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// `c_partial[bz * m * n + row * n + col] = acc;` の書き出しが
    /// 無条件（早期 return の外側）で 1 回だけ行われることを検査する
    /// （末尾要素ブロックの扱い・REQ-8）。
    #[test]
    fn splitk_partial_writes_c_partial_exactly_once() {
        let occurrences = SPLITK_PARTIAL_F32
            .matches("c_partial[(long long)bz * m * n + (long long)row * n + col] = acc;")
            .count();
        assert_eq!(occurrences, 1);
    }

    /// split-K 第 1 カーネルが atomics を使わない（決定的書き込み）ことを
    /// 検査する（`rmsnorm.rs` の dw split-K と同じ契約）。
    #[test]
    fn splitk_partial_does_not_use_atomics() {
        assert!(!SPLITK_PARTIAL_F32.contains("atomicAdd"));
        assert!(!SPLITK_REDUCE_F32.contains("atomicAdd"));
    }

    /// 縮約カーネルが `c` へ 1 回だけ書き、`c_partial` は読み出し専用
    /// （書き込みなし）であることを検査する（第 3 パスを作らない契約）。
    #[test]
    fn splitk_reduce_writes_c_exactly_once_and_never_writes_partial() {
        assert_eq!(SPLITK_REDUCE_F32.matches("c[idx] = acc;").count(), 1);
        assert!(SPLITK_REDUCE_F32.contains("const float* __restrict__ c_partial"));
        // `c_partial` は読み出し専用引数のため式中に出現しうるが、
        // `c_partial[...] = `（縮約結果を HBM へ書き戻す第 3 パス）という
        // 代入パターンは存在しない契約を検査する（読み出し自体は
        // 少なくとも 1 回存在することも合わせて確認し、テスト自体の
        // 前提崩れ〈シンボル名変更等〉を検出する）。
        assert!(SPLITK_REDUCE_F32.matches("c_partial[").count() >= 1);
        assert!(!SPLITK_REDUCE_F32.contains("c_partial[(long long)s * total + idx] ="));
    }

    /// 縮約の反復順序が `s = 0` から `num_splits` まで昇順の固定順序で
    /// あることを検査する（決定性の根拠）。
    #[test]
    fn splitk_reduce_iterates_splits_in_fixed_ascending_order() {
        assert!(SPLITK_REDUCE_F32.contains("for (int s = 0; s < num_splits; ++s)"));
    }

    /// tiled double-buffer カーネルの smem 面数が 2 であること（double
    /// buffer 契約）と、`TILED_DB_TILE` と一致することを検査する。
    #[test]
    fn tiled_db_smem_has_two_planes_matching_tile_const() {
        // カーネルソース側は `#define TILE <TILED_DB_TILE>` の CPP マクロ
        // 名（`TILE`）で宣言するため、期待文字列も定数値ではなくマクロ名
        // で組み立てる（`TILED_DB_TILE` は本ファイル冒頭コメントの通り
        // ホスト側判定ロジック専用の値の複製であり、ソース中のマクロ定義
        // 自体との一致は別途 `#define TILE {TILED_DB_TILE}` で検査する）。
        let expected_define = format!("#define TILE {TILED_DB_TILE}");
        assert!(TILED_DB_F32.contains(&expected_define));
        assert!(TILED_DB_F32.contains("__shared__ float as_tile[2][TILE][TILE];"));
        assert!(TILED_DB_F32.contains("__shared__ float bs_tile[2][TILE][TILE];"));
    }

    /// tiled double-buffer カーネルが C への書き込み時に手動境界チェック
    /// （REQ-8）を維持していることを検査する（`kernels::TILED_F32` と
    /// 同じガード式）。
    #[test]
    fn tiled_db_keeps_manual_bounds_check_on_store() {
        assert!(TILED_DB_F32.contains("if (row < m && col < n) {"));
    }

    /// tiled double-buffer カーネルのタイルロードが三項ガード（範囲外を
    /// 0.0f 充填）であり、`__restrict__` によるエイリアシング契約を
    /// 保っていることを検査する。
    #[test]
    fn tiled_db_load_uses_ternary_guard() {
        assert!(TILED_DB_F32.matches("? a[row * k + a_col] : 0.0f;").count() >= 1);
        assert!(TILED_DB_F32.matches("? b[b_row * n + col] : 0.0f;").count() >= 1);
        assert!(TILED_DB_F32.contains("const float* __restrict__ a"));
        assert!(TILED_DB_F32.contains("const float* __restrict__ b"));
    }

    /// split-K 部分和カーネルのタイルロードも三項ガードを維持している
    /// ことを検査する。
    #[test]
    fn splitk_partial_load_uses_ternary_guard() {
        assert!(
            SPLITK_PARTIAL_F32
                .contains("(row < m && a_col < k_end) ? a[(long long)row * k + a_col] : 0.0f;")
        );
        assert!(
            SPLITK_PARTIAL_F32.contains("(b_row < k_end && col < n) ? b[b_row * n + col] : 0.0f;")
        );
    }
}
