//! 最近傍リサンプリング（`torch.nn.functional.interpolate
//! (mode='nearest')` 相当。イシュー #1757）の CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列）。
//!
//! `interpolate.rs`（呼び出し元）は本モジュールの定数を
//! `nvrtc::compile_ptx` に渡し `CudaFunction` を得る。
//! `kernels_constant_pad.rs`（#1756 相当テンプレート）と同じ理由で
//! ソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に
//! nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! CPU 参照実装（`backend-cpu::interpolate`）と同じ意味論——出力の各
//! 要素は対応する入力の単一要素をそのままコピーする添字演算のみで
//! 決まる（算術を含まない）ため、数値契約は 3 バックエンド間
//! **bit 完全一致**（`.claude/rules/coding-rust.md` 数値契約節参照）。
//!
//! `kernels_constant_pad.rs` と同じく、rank 可変の shape を固定長
//! ローカル配列に頼らず扱うため、出力位置ごとの座標を「末尾軸から
//! `%`／`/=` で剥がして即座に使う」一過性のスカラー変数だけで処理し、
//! 座標配列を一切保持しない（rank に上限を設けない設計）。添字演算は
//! すべて `long long`（イシュー #1675 の教訓と同じ理由。`i32::MAX`
//! 近傍でのオーバーフロー回避）。
//!
//! [`INTERPOLATE_NEAREST_F32`]（`interpolate_nearest_f32`）: 出力位置
//! 1 個 = 1 スレッド。`idx`（出力 flat 添字）を `out_shape`（行優先の
//! 各軸サイズ配列。`rank` 要素）で末尾軸から剥がしながら各軸座標 `c`
//! を得て、`a >= rank - spatial_rank`（空間軸）なら
//! `src_c = (c * in_shape[a]) / out_shape[a]`（整数除算。ゼロ除算は
//! ホスト側 `interpolate_out_shape` が事前拒否済み）・そうでなければ
//! `src_c = c` として `in_strides` へ畳み込み `input` 側の読み出し
//! 位置を得る。`src_c` は数学的に `[0, in_shape[a])` の範囲内が保証
//! されるが、REQ-8 の縦深防御として `min(src_c, in_shape[a]-1)` を
//! 明示的に取る。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `if (idx < numel)` を維持する（グリッドがちょうど割り切れない場合の
//! 末尾ブロック対策）。ベクトル化ロード等の最適化は本イシューでは
//! 適用しない（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_constant_pad::
/// CONSTANT_PAD_BLOCK_DIM` と同じ値・同じ理由）。
pub const INTERPOLATE_BLOCK_DIM: u32 = 256;

/// `torch.nn.functional.interpolate(mode='nearest')` 相当（本ファイル
/// 冒頭コメント参照）。
pub const INTERPOLATE_NEAREST_F32: &str = r#"
extern "C" __global__ void interpolate_nearest_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    int rank,
    int spatial_start,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long in_flat = 0;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            long long src_c;
            if (a >= spatial_start) {
                long long in_size = (long long)in_shape[a];
                src_c = (c * in_size) / (long long)out_shape[a];
                long long max_c = in_size - 1;
                if (src_c > max_c) {
                    src_c = max_c;
                }
            } else {
                src_c = c;
            }
            in_flat += src_c * (long long)in_strides[a];
        }
        out[idx] = in[in_flat];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 の境界検査（`idx < numel`）を維持していることの機械検証
    /// （`kernels_constant_pad.rs` と同型の文字列テスト）。
    #[test]
    fn kernel_includes_bounds_check() {
        assert!(
            INTERPOLATE_NEAREST_F32.contains("if (idx < numel)"),
            "kernel source must retain the idx < numel bounds check"
        );
    }

    /// 添字演算は `long long`（`i32::MAX` 近傍でのオーバーフロー回避）。
    #[test]
    fn kernel_uses_long_long_for_flat_index_arithmetic() {
        assert!(INTERPOLATE_NEAREST_F32.contains("long long idx"));
        assert!(INTERPOLATE_NEAREST_F32.contains("long long src_c"));
    }

    /// 空間軸の src 添字は `min(src_c, in_shape[a]-1)` 相当の縦深防御
    /// クランプを持つ（REQ-8）。
    #[test]
    fn kernel_clamps_spatial_src_coord() {
        assert!(INTERPOLATE_NEAREST_F32.contains("if (src_c > max_c) {"));
    }

    /// 空間軸以外（`a < spatial_start`）は `src_c = c`（素通し）。
    #[test]
    fn kernel_passes_through_non_spatial_axes() {
        assert!(INTERPOLATE_NEAREST_F32.contains("src_c = c;"));
    }
}
