//! f32 GEMM の HIP カーネルソース生成（`backend-cuda::kernels::TILED_F32` の移植）。
//!
//! 役割: HIPRTC へ渡すソース文字列の唯一の生成元。静的テンプレートに、検証済みの
//! [`GemmF32Tile`] の数値だけを `#define` として前置する（文字列入力は受けない。A03・
//! `backend-cuda::warp_geometry::render_defines` と同方針）。`gemm.rs` が呼び、`hiprtc.rs` が
//! [`GemmSource`] だけを受け取る（任意の `&str` はコンパイルできない）。
//!
//! CUDA 版との差分は積和 1 行のみ: `acc += a*b`（contraction 依存）を `__builtin_fmaf` の
//! 明示 FMA にした。warp 幅依存の命令（`__shfl*`・`warpSize`・lane mask）は使わないため
//! wave64（MI300X）でそのまま正しい。REQ-8 の手動境界チェック（guarded load・書き込み時の
//! row/col 個別判定）は CUDA 版と同形で維持する。判断の詳細は
//! docs/backend-rocm-hiprtc-gemm-decision.md §2・§4。

use crate::error::RocmError;

/// カーネルのエントリ名（`extern "C"` のため名前修飾なしで `hipModuleGetFunction` へ渡せる）。
pub(crate) const GEMM_F32_KERNEL_NAME: &str = "gemm_tiled_f32";

/// カーネル引数の個数（`a, b, c, m, n, k`）。`kernelParams` 配列長と一致させる。
pub(crate) const GEMM_F32_ARG_COUNT: usize = 6;

/// テンプレート内のカーネル引数宣言。起動側の `kernelParams`（ポインタ 3 + i32 3）と対応する。
pub(crate) const GEMM_F32_ARG_SIGNATURE: &str = "const float* __restrict__ a,\n    const float* __restrict__ b,\n    float* __restrict__ c,\n    int m, int n, int k";

/// 共有メモリ上限（64 KiB。gfx942 の LDS 容量以下の保守的な値）。
const LDS_LIMIT_BYTES: u64 = 64 * 1024;
/// ブロックあたりスレッド数の上限（HIP の既定上限）。
const MAX_THREADS: u32 = 1024;

/// 検証済みのブロックタイル設定。構築は [`GemmF32Tile::new`] のみ（fail-closed）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmF32Tile {
    bm: u32,
    bn: u32,
    bk: u32,
    tm: u32,
    tn: u32,
    pad: u32,
    threads_x: u32,
    threads_y: u32,
}

impl GemmF32Tile {
    /// CUDA 版と同一の既定（64×64×16、スレッドあたり 4×4、256 スレッド）。
    pub const DEFAULT: Self = Self {
        bm: 64,
        bn: 64,
        bk: 16,
        tm: 4,
        tn: 4,
        pad: 4,
        threads_x: 16,
        threads_y: 16,
    };

    /// 不変条件を検査して構築する。
    ///
    /// 引数は `(bm, bn, bk)`・`(tm, tn)`・`pad`・`(threads_x, threads_y)`。
    ///
    /// 検査: 全値が非 0／`bm == tm*threads_y`・`bn == tn*threads_x`／スレッド数が 1024 以下かつ
    /// 64 の倍数（wave64・wave32 とも端数なし）／タイルロードが THREADS で割り切れる／
    /// 共有メモリが 64 KiB 以下。中間計算はすべて checked。
    pub fn new(
        block: (u32, u32, u32),
        thread_tile: (u32, u32),
        pad: u32,
        threads: (u32, u32),
    ) -> Result<Self, RocmError> {
        let (bm, bn, bk) = block;
        let (tm, tn) = thread_tile;
        let (threads_x, threads_y) = threads;
        let bad = |detail: &str| RocmError::InvalidKernelConfig {
            detail: detail.to_string(),
        };
        if [bm, bn, bk, tm, tn, threads_x, threads_y].contains(&0) {
            return Err(bad("0 の値は指定できません"));
        }
        let ovf = || bad("算術オーバーフロー");
        if bm != tm.checked_mul(threads_y).ok_or_else(ovf)? {
            return Err(bad("bm == tm*threads_y を満たしません"));
        }
        if bn != tn.checked_mul(threads_x).ok_or_else(ovf)? {
            return Err(bad("bn == tn*threads_x を満たしません"));
        }
        let threads = threads_x.checked_mul(threads_y).ok_or_else(ovf)?;
        if threads > MAX_THREADS || threads % 64 != 0 {
            return Err(bad("スレッド数は 1024 以下かつ 64 の倍数が必要です"));
        }
        let a_elems = bm.checked_mul(bk).ok_or_else(ovf)?;
        let b_elems = bn.checked_mul(bk).ok_or_else(ovf)?;
        if a_elems % threads != 0 || b_elems % threads != 0 {
            return Err(bad("タイルロードが THREADS で割り切れません"));
        }
        let a_row = bm.checked_add(pad).ok_or_else(ovf)? as u64;
        let b_row = bn.checked_add(pad).ok_or_else(ovf)? as u64;
        let lds = (bk as u64) * (a_row + b_row) * 4;
        if lds > LDS_LIMIT_BYTES {
            return Err(bad("共有メモリが 64 KiB を超えます"));
        }
        Ok(Self {
            bm,
            bn,
            bk,
            tm,
            tn,
            pad,
            threads_x,
            threads_y,
        })
    }

    /// ブロックタイル M。
    pub fn bm(&self) -> u32 {
        self.bm
    }
    /// ブロックタイル N。
    pub fn bn(&self) -> u32 {
        self.bn
    }
    /// ブロックタイル K。
    pub fn bk(&self) -> u32 {
        self.bk
    }
    /// スレッドあたり出力 M。
    pub fn tm(&self) -> u32 {
        self.tm
    }
    /// スレッドあたり出力 N。
    pub fn tn(&self) -> u32 {
        self.tn
    }
    /// 共有メモリの bank conflict 回避パディング。
    pub fn pad(&self) -> u32 {
        self.pad
    }
    /// ブロックの x 次元スレッド数。
    pub fn threads_x(&self) -> u32 {
        self.threads_x
    }
    /// ブロックの y 次元スレッド数。
    pub fn threads_y(&self) -> u32 {
        self.threads_y
    }
    /// ブロックのスレッド総数（検証済みのためオーバーフローしない）。
    pub fn threads(&self) -> u32 {
        self.threads_x * self.threads_y
    }
}

/// [`render_gemm_f32_source`] だけが作れるコンパイル対象ソース（任意文字列は包めない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmSource(String);

impl GemmSource {
    /// ソース文字列。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// 静的テンプレート本体。`BM` 等のマクロは前置する `#define` で与える。
const GEMM_F32_BODY_HEAD: &str = r#"
extern "C" __global__ void gemm_tiled_f32(
    "#;

const GEMM_F32_BODY_TAIL: &str = r#")
{
    __shared__ float as_tile[BK][BM + PAD];
    __shared__ float bs_tile[BK][BN + PAD];

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // 桁溢れしない num_tiles 計算（k > 0 のとき (k-1)/BK + 1）。
    int num_tiles = (k > 0) ? (k - 1) / BK + 1 : 0;
    for (int t = 0; t < num_tiles; ++t) {
        int k_base = t * BK;

        // REQ-8: A タイルの guarded load（転置格納）。範囲外は 0 埋め。
#pragma unroll
        for (int i = 0; i < (BM * BK) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int mm = idx % BM;
            int kk = idx / BM;
            int row = block_row + mm;
            int col = k_base + kk;
            as_tile[kk][mm] = (row < m && col < k) ? a[row * k + col] : 0.0f;
        }

        // REQ-8: B タイルの guarded load。
#pragma unroll
        for (int i = 0; i < (BN * BK) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int nn = idx % BN;
            int kk = idx / BN;
            int b_row = k_base + kk;
            int b_col = block_col + nn;
            bs_tile[kk][nn] = (b_row < k && b_col < n) ? b[b_row * n + b_col] : 0.0f;
        }
        __syncthreads();

        // 単一アキュムレータ・k 昇順・明示 FMA（CPU 参照 f32::mul_add と契約を揃える）。
#pragma unroll
        for (int p = 0; p < BK; ++p) {
            float a_frag[TM];
            float b_frag[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = as_tile[p][threadIdx.y * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = bs_tile[p][threadIdx.x * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __builtin_fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // REQ-8: C への書き込み時の手動境界チェック（row/col を個別判定）。
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + threadIdx.y * TM + i;
        if (row < m) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                int col = block_col + threadIdx.x * TN + j;
                if (col < n) {
                    c[row * n + col] = acc[i][j];
                }
            }
        }
    }
}
"#;

/// 検証済みタイル設定から HIPRTC 用ソースを決定的に生成する。
pub fn render_gemm_f32_source(tile: &GemmF32Tile) -> GemmSource {
    let mut s = String::new();
    // 数値のみを `{}` で整形する（文字列入力なし）。
    for (name, v) in [
        ("BM", tile.bm),
        ("BN", tile.bn),
        ("BK", tile.bk),
        ("TM", tile.tm),
        ("TN", tile.tn),
        ("PAD", tile.pad),
        ("THREADS", tile.threads()),
    ] {
        s.push_str(&format!("#define {name} {v}\n"));
    }
    s.push_str(GEMM_F32_BODY_HEAD);
    s.push_str(GEMM_F32_ARG_SIGNATURE);
    s.push_str(GEMM_F32_BODY_TAIL);
    GemmSource(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_is_deterministic() {
        let t = GemmF32Tile::DEFAULT;
        assert_eq!(render_gemm_f32_source(&t), render_gemm_f32_source(&t));
    }

    #[test]
    fn defines_match_default() {
        let s = render_gemm_f32_source(&GemmF32Tile::DEFAULT);
        let s = s.as_str();
        for d in [
            "#define BM 64\n",
            "#define BN 64\n",
            "#define BK 16\n",
            "#define TM 4\n",
            "#define TN 4\n",
            "#define PAD 4\n",
            "#define THREADS 256\n",
        ] {
            assert!(s.contains(d), "{d}");
        }
        // `#define` 行はちょうど 7 本（検証済み数値以外の注入がない）。
        assert_eq!(s.matches("#define").count(), 7);
        for line in s.lines().filter(|l| l.starts_with("#define")) {
            let mut it = line.split_whitespace();
            it.next();
            it.next();
            let v = it.next().unwrap();
            assert!(v.chars().all(|c| c.is_ascii_digit()), "{line}");
            assert!(it.next().is_none());
        }
    }

    #[test]
    fn bounds_checks_and_fma_present() {
        let s = render_gemm_f32_source(&GemmF32Tile::DEFAULT);
        let s = s.as_str();
        assert!(s.contains("(row < m && col < k) ? a[row * k + col] : 0.0f"));
        assert!(s.contains("(b_row < k && b_col < n) ? b[b_row * n + b_col] : 0.0f"));
        assert!(s.contains("if (row < m)"));
        assert!(s.contains("if (col < n)"));
        assert!(s.contains("__builtin_fmaf(a_frag[i], b_frag[j], acc[i][j])"));
        assert!(!s.contains("+= a_frag"));
    }

    #[test]
    fn no_warp_width_dependence() {
        let s = render_gemm_f32_source(&GemmF32Tile::DEFAULT);
        for w in ["__shfl", "warpSize", "WARP_", "__ballot", "__activemask"] {
            assert!(!s.as_str().contains(w), "{w}");
        }
    }

    #[test]
    fn signature_matches_arg_count() {
        let s = render_gemm_f32_source(&GemmF32Tile::DEFAULT);
        assert!(s.as_str().contains(GEMM_F32_ARG_SIGNATURE));
        assert!(s.as_str().contains(GEMM_F32_KERNEL_NAME));
        // ポインタ 3 + int 3。
        assert_eq!(GEMM_F32_ARG_SIGNATURE.matches("float*").count(), 3);
        assert_eq!(GEMM_F32_ARG_SIGNATURE.matches("int ").count(), 3);
        assert_eq!(
            GEMM_F32_ARG_SIGNATURE.matches(',').count() + 1,
            GEMM_F32_ARG_COUNT
        );
    }

    #[test]
    fn default_is_valid() {
        let d = GemmF32Tile::DEFAULT;
        let v = GemmF32Tile::new((64, 64, 16), (4, 4), 4, (16, 16)).unwrap();
        assert_eq!(d, v);
        assert_eq!(d.threads(), 256);
    }

    #[test]
    fn rejects_invalid_tiles() {
        // 0 値・不整合・64 の倍数でない・割り切れない・LDS 超過・オーバーフロー。
        assert!(GemmF32Tile::new((0, 64, 16), (4, 4), 4, (16, 16)).is_err());
        assert!(GemmF32Tile::new((64, 64, 16), (4, 4), 4, (16, 8)).is_err());
        assert!(GemmF32Tile::new((32, 32, 16), (4, 4), 4, (4, 8)).is_err()); // 32 threads
        assert!(GemmF32Tile::new((64, 64, 15), (4, 4), 4, (16, 16)).is_err()); // 割り切れない
        assert!(GemmF32Tile::new((64, 64, 4096), (4, 4), 4, (16, 16)).is_err()); // LDS
        assert!(GemmF32Tile::new((u32::MAX, 64, 16), (1, 4), 4, (16, u32::MAX)).is_err());
        assert!(GemmF32Tile::new((64, 64, 16), (4, 4), u32::MAX, (16, 16)).is_err());
    }
}
