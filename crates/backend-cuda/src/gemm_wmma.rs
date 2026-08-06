//! f16 WMMA GEMM の起動 API（TASK-11.1b・#61）。
//!
//! `CudaWmmaGemm` は `kernels_wmma::WMMA_F16`（`m16n16k16` fragment・f32
//! アキュムレート）をコンパイル・保持し、以降はホスト側スライスを渡す
//! だけで GPU 実行できる境界を担う（`gemm.rs::CudaGemm` と同じ責務分割。
//! naive／tiled 経路とは意図的に別構造体・別ファイルへ分離した。PR #244
//! 「tiled」が `gemm.rs`／`kernels.rs`／`lib.rs` を並行編集中のため
//! 本体競合を避ける判断。実装計画 3.4 節参照）。
//!
//! ホスト側形状検証は `gemm.rs::validate_gemm_dims`（`pub(crate)`）を
//! そのまま再利用し、判定ロジックを複製しない（naive／tiled と同じ
//! `CudaError::InvalidShape` 契約を共有する）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::kernels_wmma;
use crate::nvrtc::compile_ptx;

/// WMMA f16 経路が要求する compute capability の下限（major）。
///
/// 設計メモ（`docs/cuda-tensor-core-design.md` 7 節）が記す「WMMA f16 経路は
/// compute capability 7.0 以降で有効化可能」を実装した定数。`CudaWmmaGemm::new`
/// はこの下限を NVRTC コンパイル前に検査し、満たさない場合は
/// `CudaError::TensorCoreUnsupported` を返す（フォールバック判断自体は
/// TASK-11.2／#66 のディスパッチ規則側の責務）。
const MIN_COMPUTE_CAPABILITY_MAJOR: i32 = 7;

/// `kernels_wmma::WMMA_TILE` に 1:1 対応するブロック次元。
///
/// 1 ブロック = 1 warp（32 スレッド）= C の `WMMA_TILE x WMMA_TILE`
/// タイル 1 個を計算する構成（`kernels_wmma.rs` 冒頭ドキュメントコメント
/// 「タイル構成」参照）。カーネル内の `lane`（`threadIdx.x`、0..31）を
/// 前提にしたガードロード／ストアと 1:1 対応するため、ここを変更する
/// 場合はカーネルソース側のスレッド分担ロジックも合わせて見直す必要がある。
const WMMA_BLOCK_DIM: (u32, u32, u32) = (32, 1, 1);

/// f16 WMMA GEMM カーネルのコンパイル済みハンドルを保持する。
///
/// `stream` は `CudaDevice` から `Arc` クローンで受け取る（`gemm.rs::CudaGemm`
/// と同じ共有契約。`device.rs` 参照）。
pub struct CudaWmmaGemm {
    stream: Arc<CudaStream>,
    wmma_f16: CudaFunction,
}

impl CudaWmmaGemm {
    /// `device` 上で f16 WMMA GEMM カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する。
    ///
    /// 手順: (1) `device.compute_capability()` が
    /// [`MIN_COMPUTE_CAPABILITY_MAJOR`] 未満なら NVRTC コンパイルを試みず
    /// `CudaError::TensorCoreUnsupported` を返す（設計メモ 7 節。cc 判定は
    /// コンパイル前に行うことで、非対応デバイス上での無駄な NVRTC 呼び出し
    /// ・コンパイル失敗の紛れ込みを避ける）。(2)
    /// `kernels_wmma::WMMA_F16` を `device.arch()` 向けに
    /// `nvrtc::compile_ptx` でコンパイル。(3) `device.context().load_module()`
    /// → `load_function("gemm_wmma_f16")`。カーネルコンパイル自体は
    /// `CudaGemm::new` と同じく `libnvrtc` 不在時に
    /// `CudaError::NvrtcUnavailable` を返す（`compile_ptx` のプローブゲート
    /// を経由。panic しない）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let (major, minor) = device.compute_capability();
        if major < MIN_COMPUTE_CAPABILITY_MAJOR {
            return Err(CudaError::TensorCoreUnsupported {
                detail: format!(
                    "WMMA requires compute capability >= {MIN_COMPUTE_CAPABILITY_MAJOR}.0, \
                     but device reports {major}.{minor}"
                ),
            });
        }

        let arch = device.arch();
        let ptx = compile_ptx(kernels_wmma::WMMA_F16, arch)?;
        let wmma_f16 = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_f16")?;

        Ok(Self {
            stream: device.stream().clone(),
            wmma_f16,
        })
    }

    /// f16 WMMA GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。入出力は
    /// `half::f16`、GPU 内部アキュムレートは f32
    /// （`kernels_wmma::WMMA_F16` 参照。数値契約は `CudaGemm::run_naive_f16`
    /// と同一。`kernels_wmma.rs` 冒頭ドキュメントコメント「数値契約」参照）。
    ///
    /// ホスト側形状検証（[`validate_gemm_dims`]）を naive／tiled 経路と
    /// 共有する（`gemm.rs` 参照）。グリッド次元は `kernels_wmma::WMMA_TILE`
    /// 単位の `div_ceil` で構築し、末尾タイルの余剰はカーネル内 REQ-8
    /// 境界チェック（`kernels_wmma.rs` 参照）に委ねる。
    ///
    /// `gemm.rs::launch_config` は 1 スレッド = C の 1 要素という naive／
    /// tiled の起動モデル前提であり、1 ブロック = 1 warp = `WMMA_TILE x
    /// WMMA_TILE` 要素という WMMA の起動モデルとは異なるため再利用しない
    /// （本関数専用のグリッド計算を用いる）。
    pub fn run_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;

        // gemm.rs::run_f32_kernel/run_f16_kernel と同一の根拠（該当コメント
        // 参照）: m==0/n==0 はカーネル起動自体を回避する no-op 形状。
        // 0 次元 grid の起動は CUDA ドライバが拒否するため（Cursor Bugbot
        // 指摘 #240 と同種の防御）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        // k==0 は A/B が空スライスになり 0 バイトデバイスバッファ確保を
        // 招くため、カーネル起動を回避し C = 全 0 を返す（GEMM の数学的
        // 定義どおりの契約。gemm.rs::run_f32_kernel の k==0 早期 return と
        // 同一の根拠）。
        if k == 0 {
            return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?;

        let cfg = wmma_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/
        // b.len()/(m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の
        // 5 個・型・個数が、ホスト側検証（validate_gemm_dims）済みの
        // m/n/k と 1:1 対応する。カーネル内の手動境界チェック（A/B タイル
        // guarded load・エピローグ guarded store。kernels_wmma.rs 参照、
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。グリッド
        // 次元は WMMA_TILE 単位の div_ceil で m/n を包含するよう構築して
        // おり（wmma_launch_config）、末尾タイルの余剰はカーネル内境界
        // チェックで弾かれる。共有メモリは静的 `__shared__` 配列のみを
        // 使用するため `shared_mem_bytes` は 0 のままでよい（動的共有
        // メモリを追加確保しない。gemm.rs::launch_config と同じ構成）。
        unsafe {
            self.stream
                .launch_builder(&self.wmma_f16)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        let c_host = self.stream.clone_dtoh(&c_dev)?;
        Ok(c_host)
    }
}

/// WMMA カーネル専用のグリッド次元計算。1 ブロック = 1 warp = C の
/// `WMMA_TILE x WMMA_TILE` タイル 1 個を担当するため、`div_ceil(n,
/// WMMA_TILE)` x `div_ceil(m, WMMA_TILE)` のグリッドを構築する
/// （`gemm.rs::launch_config` の「1 スレッド = C の 1 要素」前提とは
/// 異なるため独立関数とする。本ファイル冒頭ドキュメントコメント参照）。
fn wmma_launch_config(m: u32, n: u32) -> LaunchConfig {
    let tile = kernels_wmma::WMMA_TILE;
    let grid_dim = (n.div_ceil(tile), m.div_ceil(tile), 1);
    LaunchConfig {
        grid_dim,
        block_dim: WMMA_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wmma_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // 17x19 を WMMA_TILE=16 単位で覆うには grid (2, 2) が必要
        // （div_ceil(17,16)=2, div_ceil(19,16)=2）。
        let cfg = wmma_launch_config(17, 19);
        assert_eq!(cfg.grid_dim, (2, 2, 1));
        assert_eq!(cfg.block_dim, WMMA_BLOCK_DIM);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn wmma_launch_config_exact_multiple_shape_has_no_extra_tile() {
        let cfg = wmma_launch_config(32, 48);
        assert_eq!(cfg.grid_dim, (3, 2, 1));
    }
}
