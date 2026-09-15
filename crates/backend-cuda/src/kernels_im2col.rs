//! Conv2d の im2col／col2im（イシュー #1766・親 #1643・設計
//! `docs/conv-ops-design.md`）の CUDA C カーネルソース（NVRTC 実行時
//! コンパイル用の静的文字列）。
//!
//! `im2col.rs`（呼び出し元）は本モジュールの定数を `nvrtc::compile_ptx`
//! に渡し `CudaFunction` を得る。`kernels_constant_pad.rs`／
//! `kernels_gather_scatter.rs` と同じ理由でソースを `nvcc` 事前
//! コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA ヘッダを
//! 一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。
//! `.claude/rules/deps-policy.md`）。
//!
//! `Conv2dParams`（rank 固定 4D NCHW）を対象とするため
//! `kernels_gather_scatter.rs`／`kernels_constant_pad.rs` と異なり
//! rank 可変の shape 配列を H2D 転送しない。すべての形状パラメータを
//! カーネル起動時のスカラー `int` 引数として渡す（`im2col.rs::
//! CudaIm2col::run_im2col_f32`／`run_col2im_f32` 参照）。
//!
//! # 出力レイアウト（[`fandhe_ai_tensor_core::im2col_out_shape`] と同一）
//!
//! `im2col_f32`: `input: [N, Cin, H, W]` を `out: [N, G, K_g, P]`
//! （`K_g` 軸は `(c_in_g, kh, kw)` の row-major・`P` 軸は `(oh, ow)` の
//! row-major）へ展開する。`col2im_f32` はその随伴（転置畳み込み）で
//! `d_col: [N, G, K_g, P]` を `out: [N, Cin, H, W]` へ畳み戻す。
//!
//! # 数値契約
//!
//! [`IM2COL_F32`] は「`input` 内部位置ならそのままコピー・padding
//! 位置なら `0.0`」の 2 分岐のみで決まる純粋なコピー演算（算術を
//! 含まない）のため CPU 参照実装（`backend-cpu::im2col::im2col`）と
//! **bit 完全一致**（`.claude/rules/coding-rust.md` 数値契約節・
//! `kernels_constant_pad.rs::CONSTANT_PAD_F32` と同型）。
//!
//! [`COL2IM_F32`] は重なり窓（`stride < dilation·(kernel−1)+1`）の
//! 加算順を `(kh, kw)` row-major に固定し `double` アキュムレータへ
//! 逐次加算・最後に 1 回 `float` へ downcast する（`.claude/rules/
//! coding-rust.md` の勾配の長軸縮約規約。CPU 参照実装
//! `backend-cpu::im2col::col2im` の `f64` 逐次和と **bit 完全一致**。
//! CUDA は `double` をハードウェアでネイティブサポートするため
//! Metal（`soft_f64`）のようなソフトウェアエミュレーションは不要）。
//! ここでの加算は乗算を伴わない純粋な和のため FMA 融合の余地がなく、
//! Rust ホスト側の `acc += f64::from(v)` と CUDA の `acc += (double)v`
//! は同一の丸め結果になる。
//!
//! 座標計算（`h + p_h − kh·d_h` 等）は `backend-cpu::im2col` の
//! `checked_sub` 相当を `long long`（符号付き 64bit）演算で表現する
//! （減算結果が負になれば「この窓には寄与しない」。C の負数 `%` の
//! 符号曖昧性を避けるため、剰余を取る**前**に符号判定する。設計 doc
//! §6.2「訂正 2」と同じ理由）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! `if (idx < numel)` を維持する（グリッドがちょうど割り切れない場合の
//! 末尾ブロック対策）。添字演算はすべて `long long`（イシュー #1675 の
//! 教訓と同じ理由。`i32::MAX` 近傍でのオーバーフロー回避）。

/// 1 スレッドブロックあたりのスレッド数（`kernels_constant_pad::
/// CONSTANT_PAD_BLOCK_DIM` と同じ値・同じ理由）。
pub const IM2COL_BLOCK_DIM: u32 = 256;

/// Conv2d の im2col（`torch.nn.functional.unfold` の grouped 版相当。
/// モジュール doc 参照）。1 出力要素（`out` の 1 flat 添字）= 1
/// スレッド。
///
/// 引数はすべて呼び出し元（`im2col.rs::CudaIm2col::run_im2col_f32`）が
/// `i32` 範囲検査済みのスカラー: `n_batch, cin, groups, cin_g, h_in,
/// w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw, k_g, p, numel`
/// （`p` は出力 `P` 軸長 = `h_out * w_out`。`numel` は出力 `[N,G,K_g,P]`
/// の総要素数 = カーネル起動グリッドの範囲）。
pub const IM2COL_F32: &str = r#"
extern "C" __global__ void im2col_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n_batch, int cin, int groups, int cin_g,
    int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw,
    int ph, int pw, int dh, int dw,
    int k_g, int p, int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long p_idx = rem % p;
        rem /= p;
        long long k_idx = rem % k_g;
        rem /= k_g;
        long long g = rem % groups;
        rem /= groups;
        long long n = rem;

        long long kw_ = k_idx % kw;
        long long rest = k_idx / kw;
        long long kh_ = rest % kh;
        long long c_g = rest / kh;
        long long c = g * (long long)cin_g + c_g;

        long long ow = p_idx % w_out;
        long long oh = p_idx / w_out;

        long long h = oh * (long long)sh + kh_ * (long long)dh - (long long)ph;
        long long w = ow * (long long)sw + kw_ * (long long)dw - (long long)pw;

        float value = 0.0f;
        if (h >= 0 && h < h_in && w >= 0 && w < w_in) {
            long long in_idx = ((n * (long long)cin + c) * h_in + h) * w_in + w;
            value = in[in_idx];
        }
        out[idx] = value;
    }
}
"#;

/// Conv2d の col2im（[`IM2COL_F32`] の随伴＝転置畳み込みの fold。
/// モジュール doc 参照）。1 入力位置（`out` の 1 flat 添字）= 1
/// スレッド（`d_col` への atomic 書き込みは行わない。入力位置定常の
/// 走査で寄与するすべての `(kh, kw)` を単一スレッド内で `double`
/// アキュムレータへ逐次加算する）。
///
/// 引数は [`IM2COL_F32`] と同一の形状スカラー群＋`d_col`／`out`。
/// `numel` は `out`（`[N, Cin, H, W]`）の総要素数。
pub const COL2IM_F32: &str = r#"
extern "C" __global__ void col2im_f32(
    const float* __restrict__ d_col,
    float* __restrict__ out,
    int n_batch, int cin, int groups, int cin_g,
    int h_in, int w_in, int h_out, int w_out,
    int kh, int kw, int sh, int sw,
    int ph, int pw, int dh, int dw,
    int k_g, int p, int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long w = rem % w_in;
        rem /= w_in;
        long long h = rem % h_in;
        rem /= h_in;
        long long c = rem % cin;
        rem /= cin;
        long long n = rem;

        long long g = c / (long long)cin_g;
        long long c_g = c % (long long)cin_g;

        double acc = 0.0;
        for (int kh_ = 0; kh_ < kh; kh_++) {
            long long num_h = h + (long long)ph - (long long)kh_ * dh;
            if (num_h < 0) continue;
            if (num_h % sh != 0) continue;
            long long oh = num_h / sh;
            if (oh >= h_out) continue;
            for (int kw_ = 0; kw_ < kw; kw_++) {
                long long num_w = w + (long long)pw - (long long)kw_ * dw;
                if (num_w < 0) continue;
                if (num_w % sw != 0) continue;
                long long ow = num_w / sw;
                if (ow >= w_out) continue;

                long long k_idx = (c_g * kh + kh_) * kw + kw_;
                long long p_idx = oh * w_out + ow;
                long long col_idx = ((n * (long long)groups + g) * k_g + k_idx) * p + p_idx;
                acc += (double)d_col[col_idx];
            }
        }
        out[idx] = (float)acc;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 の境界検査（`idx < numel`）を維持していることの機械検証
    /// （`kernels_constant_pad.rs` と同型の文字列テスト）。
    #[test]
    fn kernels_include_bounds_check() {
        assert!(IM2COL_F32.contains("if (idx < numel)"));
        assert!(COL2IM_F32.contains("if (idx < numel)"));
    }

    /// 添字演算は `long long`（`i32::MAX` 近傍でのオーバーフロー回避。
    /// イシュー #1675 の教訓）。
    #[test]
    fn kernels_use_long_long_for_flat_index_arithmetic() {
        assert!(IM2COL_F32.contains("long long idx"));
        assert!(COL2IM_F32.contains("long long idx"));
    }

    /// [`COL2IM_F32`] は `double` アキュムレータへ逐次加算し最後に
    /// 1 回だけ `float` へ downcast する（モジュール doc の数値契約）。
    #[test]
    fn col2im_kernel_uses_double_accumulator_with_single_downcast() {
        assert!(COL2IM_F32.contains("double acc = 0.0;"));
        assert!(COL2IM_F32.contains("acc += (double)d_col[col_idx];"));
        // `(float)acc` の書き込みは 1 箇所のみ（ループの外・末尾）。
        assert_eq!(COL2IM_F32.matches("(float)acc").count(), 1);
    }

    /// [`IM2COL_F32`] は算術を含まない純粋なコピー演算のため `double`
    /// を一切使わない（モジュール doc の数値契約・bit 完全一致）。
    #[test]
    fn im2col_kernel_contains_no_double_arithmetic() {
        assert!(!IM2COL_F32.contains("double"));
    }

    /// col2im の座標計算は剰余を取る前に符号判定する（C の負数 `%` の
    /// 符号曖昧性を避けるため。設計 doc §6.2「訂正 2」）。
    #[test]
    fn col2im_checks_sign_before_modulo() {
        let idx_h = COL2IM_F32.find("if (num_h < 0) continue;").unwrap();
        let mod_h = COL2IM_F32.find("if (num_h % sh != 0)").unwrap();
        assert!(idx_h < mod_h);
        let idx_w = COL2IM_F32.find("if (num_w < 0) continue;").unwrap();
        let mod_w = COL2IM_F32.find("if (num_w % sw != 0)").unwrap();
        assert!(idx_w < mod_w);
    }
}
