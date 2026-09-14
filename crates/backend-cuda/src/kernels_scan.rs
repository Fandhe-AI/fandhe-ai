//! 累積和／累積積（`torch.cumsum`／`torch.cumprod` 相当）の CUDA C
//! カーネルソース（NVRTC 実行時コンパイル用の静的文字列。イシュー
//! #1740・親イシュー #1731）。
//!
//! `scan.rs`（呼び出し元）は本モジュールの 2 定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels_gather_scatter.rs`
//! と同じ理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む
//! （ビルド時に nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit
//! 非搭載環境でも `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! # lane 逐次 scan（ブロック内並列化不可の理由）
//!
//! [`fandhe_ai_tensor_core::BackendOps::cumsum`]／`cumprod` doc が定める
//! 数値契約（`dim` 以外の軸の組＝lane ごとに `f64` アキュムレータを
//! 保持し、`dim` 添字昇順に逐次計算。各ステップの出力はその時点の
//! アキュムレータを `f32` へ downcast したスナップショットで、次
//! ステップは downcast 後の `f32` を読み戻さない）は
//! `backend-cpu::scan` の単一スレッド逐次実装と bit 完全一致すること
//! を要求する。ブロック内 scan アルゴリズム（Hillis–Steele や
//! work-efficient scan）は要素を対数段でペアワイズ結合するため、この
//! 「`dim` 添字昇順の逐次 `f64` 加算・乗算列」と異なる結合順序になり
//! 出力が bit 一致しなくなる（浮動小数点加算・乗算は結合則を満たさ
//! ない）。したがって本カーネルは「1 スレッド = 1 lane・`axis_len`
//! 方向は当該スレッド内で逐次」という構造のみを取り、lane 数が少ない
//! 形状（1-D 入力等）では並列度が出ない（正しさ優先の既知の制約。
//! `.claude/rules/out-of-scope-tracking.md` 対象。PR 本文に引き継ぎ
//! 事項として明記する）。
//!
//! # 数値方式
//!
//! `double acc` を `0.0`（cumsum）／`1.0`（cumprod）から開始し、
//! `axis_len` 回のループで `acc = acc + (double)x[idx]`（cumprod は
//! `acc = acc * (double)x[idx]`）を計算したうえで各ステップ即座に
//! `out[idx] = (float)acc` を書く（次ステップは `out[idx]` を読み
//! 戻さない。`kernels_gather_scatter.rs::SCATTER_ADD_F32` の `double`
//! アキュムレータと同じ精度規律。`.claude/rules/coding-rust.md`
//! 「勾配の長軸縮約」節と同じ「要素を `double` へ先に昇格してから
//! 逐次演算」方針を forward の scan へ適用したもの）。CUDA は native
//! `double` 演算のため `backend-cpu::scan`（`f64` アキュムレータ）と
//! 同一の IEEE 754 binary64 逐次演算列であり、両者は bit 完全一致する
//! （NVRTC 既定オプションのまま——加算のみ／乗算のみの連鎖で FMA 縮約
//! の余地がない）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `gid`（グローバルスレッド id＝lane 番号）が `lanes` 未満かどうかを
//! 検査してから処理する。添字計算は `long long`（64bit）で行い
//! `i32::MAX` 近傍でのオーバーフローを回避する
//! （`kernels_reduce.rs::REDUCE_SUM_LASTAXIS_F32` の教訓と同じ理由）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_gather_scatter::
/// GATHER_SCATTER_BLOCK_DIM` と同じ値・同じ理由）。
pub const SCAN_BLOCK_DIM: u32 = 256;

/// 累積和カーネル（本ファイル冒頭コメント参照）。
///
/// - `x`／`out`: `outer * axis_len * inner` 要素（呼び出し元が
///   `contiguous()` で稠密化した入力）
/// - `lanes = outer * inner`: `gid < lanes` の境界検査対象
/// - `axis_len`: `dim` 軸のサイズ（逐次ループの反復回数）
/// - `inner`: `dim` より後ろの軸の積（lane の flat 添字 `gid` を
///   `o = gid / inner`・`i = gid % inner` へ分解するための除数）
pub const CUMSUM_F32: &str = r#"
extern "C" __global__ void cumsum_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int lanes,
    int axis_len,
    int inner)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)lanes) {
        return;
    }
    long long o = gid / (long long)inner;
    long long i = gid % (long long)inner;
    double acc = 0.0;
    for (int a = 0; a < axis_len; a++) {
        long long idx = (o * (long long)axis_len + (long long)a) * (long long)inner + i;
        acc = acc + (double)x[idx];
        out[idx] = (float)acc;
    }
}
"#;

/// 累積積カーネル（本ファイル冒頭コメント参照）。[`CUMSUM_F32`] と
/// 同じ lane 分解・境界検査だが、アキュムレータは `1.0` から開始し
/// 積を蓄積する。
pub const CUMPROD_F32: &str = r#"
extern "C" __global__ void cumprod_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int lanes,
    int axis_len,
    int inner)
{
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (long long)lanes) {
        return;
    }
    long long o = gid / (long long)inner;
    long long i = gid % (long long)inner;
    double acc = 1.0;
    for (int a = 0; a < axis_len; a++) {
        long long idx = (o * (long long)axis_len + (long long)a) * (long long)inner + i;
        acc = acc * (double)x[idx];
        out[idx] = (float)acc;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8: 両カーネルとも `gid >= lanes` の境界検査を持つ。
    #[test]
    fn kernels_have_req8_boundary_check() {
        assert!(CUMSUM_F32.contains("if (gid >= (long long)lanes)"));
        assert!(CUMPROD_F32.contains("if (gid >= (long long)lanes)"));
    }

    /// 数値契約: `double` アキュムレータを使い、`float acc` は使わない
    /// （`f64` 相当の精度規律。`.claude/rules/coding-rust.md`）。
    #[test]
    fn kernels_use_double_accumulator_not_float() {
        assert!(CUMSUM_F32.contains("double acc = 0.0;"));
        assert!(CUMPROD_F32.contains("double acc = 1.0;"));
        assert!(!CUMSUM_F32.contains("float acc"));
        assert!(!CUMPROD_F32.contains("float acc"));
    }

    /// 各ステップで即座に `out[idx]` へ書く（downcast 後の値を読み
    /// 戻さない契約の機械的裏付け: `out[idx]` の代入が各ループ本体に
    /// 1 回ずつ現れる）。
    #[test]
    fn kernels_write_output_every_step() {
        assert_eq!(CUMSUM_F32.matches("out[idx] = (float)acc;").count(), 1);
        assert_eq!(CUMPROD_F32.matches("out[idx] = (float)acc;").count(), 1);
    }

    /// 関数名が `scan.rs::CudaScan::new` の `load_function` 呼び出しと
    /// 一致する。
    #[test]
    fn kernel_function_names_match_expected() {
        assert!(CUMSUM_F32.contains("__global__ void cumsum_f32("));
        assert!(CUMPROD_F32.contains("__global__ void cumprod_f32("));
    }
}
