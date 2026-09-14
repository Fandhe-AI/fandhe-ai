//! `unique`（`torch.unique(input, sorted=True)` 相当）の CUDA C
//! カーネルソース（NVRTC 実行時コンパイル用の静的文字列。イシュー
//! #1734）。
//!
//! `unique.rs`（呼び出し元）は本定数を `nvrtc::compile_ptx` に渡し
//! `CudaFunction` を得る。`kernels_gather_scatter.rs` と同じ理由で
//! ソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に
//! nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # ビットニックソート 1 ステップ
//!
//! [`BITONIC_STEP_U32`]（`bitonic_step_u32`）はグローバルメモリ上の
//! `u32` キー配列に対しビットニックソートの 1 比較ステップを実行する
//! （`unique.rs::CudaUnique::run_unique_f32` がホスト側ループで `j`／`k`
//! を変えながら本カーネルを繰り返し起動する。ステップ分解の理由・
//! ホスト側ループ構造は `unique.rs` モジュール doc を参照）。整数
//! compare/swap のみを行い浮動小数点演算を一切含まないため、NVRTC の
//! math モード（`--use_fast_math` 等）に非依存で決定的である。
//!
//! キー自体は `u32` の昇順が IEEE 754 totalOrder と一致するよう
//! ホスト側で変換済み（`unique_model.rs::total_order_key` 参照。
//! カーネル自体は単純な整数比較のみを行う）。
//!
//! ビットニックソートは 2 のべき乗長の配列を要求するため、呼び出し元
//! はホスト側で実データを `u32::MAX`（totalOrder 上の最大キー。
//! `unique_model.rs` 参照）でパディングして `padded`（2 のべき乗）長に
//! してから本カーネルを繰り返し起動する。カーネル自身は `padded` 全域
//! （パディング要素も含む）に対して一様にビットニックソートの比較を
//! 行い、パディング値は最大キーであるため自然にソート後の末尾へ集まる
//! （呼び出し元が先頭 `n`〈実データ長〉要素だけを読み出す）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `n` 引数は `padded`（ソート対象配列全長。2 のべき乗）を指す
//! （実データ長ではない——グリッドの末尾ブロックが `padded` を超えて
//! スレッドを起動しうるための境界検査用）。`i`（自スレッドの添字）・
//! `ixj`（比較相手の添字）の両方をこの `n`（＝`padded`）未満かどうか
//! 検査する（`kernels_gather_scatter.rs` と同じ縦深防御方針。最適化を
//! 理由に省略しない）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_gather_scatter::
/// GATHER_SCATTER_BLOCK_DIM` と同じ値・同じ理由）。
pub const UNIQUE_BLOCK_DIM: u32 = 256;

/// ビットニックソート 1 ステップ（本モジュール doc 参照）。
///
/// `j`／`k` は呼び出し元（`unique.rs`）がホスト側ループで管理する
/// ステップパラメータ（標準的なビットニックソートの段・比較距離）。
/// `n` は `padded`（ソート対象配列全長。2 のべき乗）で、`i >= n` または
/// `ixj >= n`（あるいは `ixj <= i`。同一ペアの二重処理防止）なら
/// 何もしない。
pub const BITONIC_STEP_U32: &str = r#"
extern "C" __global__ void bitonic_step_u32(
    unsigned int* keys,
    int j,
    int k,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int ixj = i ^ j;
    if (ixj <= i || ixj >= n) {
        return;
    }
    unsigned int a = keys[i];
    unsigned int b = keys[ixj];
    bool ascending = (i & k) == 0;
    bool should_swap = ascending ? (a > b) : (a < b);
    if (should_swap) {
        keys[i] = b;
        keys[ixj] = a;
    }
}
"#;
