//! 定数パディング（`torch.nn.functional.pad(mode='constant')` 相当。
//! イシュー #1756）の CUDA C カーネルソース（NVRTC 実行時コンパイル用の
//! 静的文字列）。
//!
//! `constant_pad.rs`（呼び出し元）は本モジュールの定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels_gather_scatter.rs`
//! と同じ理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む
//! （ビルド時に nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit
//! 非搭載環境でも `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! CPU 参照実装（`backend-cpu::constant_pad`）と同じ意味論——出力の各
//! 要素は「`input` 内部位置ならそのままコピー・パディング領域なら
//! `value`」の 2 分岐のみで決まる算術を含まない純粋なコピー演算のため、
//! 数値契約は 3 バックエンド間 **bit 完全一致**（`.claude/rules/
//! coding-rust.md` 数値契約節参照）。
//!
//! `kernels_gather_scatter.rs` と同じく、rank 可変の shape を固定長
//! ローカル配列に頼らず扱うため、出力位置ごとの座標を「末尾軸から
//! `%`／`/=` で剥がして即座に使う」一過性のスカラー変数だけで処理し、
//! 座標配列を一切保持しない（rank に上限を設けない設計）。添字演算は
//! すべて `long long`（イシュー #1675 の教訓と同じ理由。`i32::MAX`
//! 近傍でのオーバーフロー回避）。
//!
//! [`CONSTANT_PAD_F32`]（`constant_pad_f32`）: 出力位置 1 個 = 1
//! スレッド。`idx`（出力 flat 添字）を `out_shape`（行優先の各軸サイズ
//! 配列。`rank` 要素）で末尾軸から剥がしながら各軸座標 `c` を得て、
//! `src_c = c - before[a]` が `[0, in_shape[a])` の範囲内であれば
//! `in_strides` へ畳み込んで `input` 側の読み出し位置を得る。1 軸でも
//! 範囲外なら `value` を書く（境界検査は 32bit 符号なし演算での
//! アンダーフローを避けるため `long long` の減算で判定する）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `if (idx < numel)` を維持する（グリッドがちょうど割り切れない場合の
//! 末尾ブロック対策）。ベクトル化ロード等の最適化は本イシューでは
//! 適用しない（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_gather_scatter::
/// GATHER_SCATTER_BLOCK_DIM` と同じ値・同じ理由）。
pub const CONSTANT_PAD_BLOCK_DIM: u32 = 256;

/// `torch.nn.functional.pad(mode='constant')` 相当（本ファイル冒頭
/// コメント参照）。
pub const CONSTANT_PAD_F32: &str = r#"
extern "C" __global__ void constant_pad_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ before,
    const int* __restrict__ in_strides,
    int rank,
    int numel,
    float value)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long in_flat = 0;
        bool inside = true;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            long long src_c = c - (long long)before[a];
            if (src_c < 0 || src_c >= (long long)in_shape[a]) {
                inside = false;
                break;
            }
            in_flat += src_c * (long long)in_strides[a];
        }
        out[idx] = inside ? in[in_flat] : value;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 の境界検査（`idx < numel`）を維持していることの機械検証
    /// （`kernels_gather_scatter.rs` と同型の文字列テスト）。
    #[test]
    fn kernel_includes_bounds_check() {
        assert!(
            CONSTANT_PAD_F32.contains("if (idx < numel)"),
            "kernel source must retain the idx < numel bounds check"
        );
    }

    /// 添字演算は `long long`（`i32::MAX` 近傍でのオーバーフロー回避）。
    #[test]
    fn kernel_uses_long_long_for_flat_index_arithmetic() {
        assert!(CONSTANT_PAD_F32.contains("long long idx"));
        assert!(CONSTANT_PAD_F32.contains("long long src_c"));
    }

    /// 範囲外（パディング領域）は `value` を書く分岐を持つ。
    #[test]
    fn kernel_has_padding_fallback_to_value() {
        assert!(CONSTANT_PAD_F32.contains("inside ? in[in_flat] : value"));
    }
}
