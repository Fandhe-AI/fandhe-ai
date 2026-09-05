//! smem パディング + スウィズルによる転置カーネルの起動 API（NVRTC
//! コンパイル・保持・実行）。イシュー #601（親 #582 の G-11）。
//!
//! `CudaTranspose` は `kernels_transpose.rs` に定義したカーネルソースを
//! コンパイル・保持し、ホスト側スライスを渡すだけで GPU 実行できる境界を
//! 担う（`gemm.rs::CudaGemm` と同じ構造）。GEMM epilogue 融合変種
//! （`TILED_TRANSPOSED_F32`）もここに opt-in として保持し、`CudaGemm::new`
//! の eager コンパイル集合には追加しない（実装計画 3.3 節「公開面は
//! opt-in 専用」。`ops.rs`／`gemm_auto.rs` の本番ディスパッチには一切結線
//! しない）。
//!
//! `swizzle.rs` と同じく、本モジュールのホスト側関数（[`swizzled_smem_col`]）
//! はカーネル文字列側の同一整数式と単一真実源の関係にはない
//! （`kernels_transpose.rs` の needle テストが不一致を機械検出する）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_transpose;
use crate::nvrtc::compile_ptx;

/// smem 転置カーネル起動 1 回あたりのブロック次元
/// （`kernels_transpose::TRANSPOSE_TILE` x 同値。`gemm.rs::TILED_BLOCK_DIM`
/// と同じ「ブロック次元＝タイル一辺」の 1:1 対応契約）。
const TRANSPOSE_TILE_BLOCK_DIM: (u32, u32, u32) = (
    kernels_transpose::TRANSPOSE_TILE,
    kernels_transpose::TRANSPOSE_TILE,
    1,
);

/// naive 転置カーネル起動 1 回あたりのブロック次元（16x16。
/// `gemm.rs::NAIVE_BLOCK_DIM` と同じ値を踏襲。共有メモリを使わないため
/// タイル一辺との対応制約を受けない）。
const NAIVE_TRANSPOSE_BLOCK_DIM: (u32, u32, u32) = (16, 16, 1);

/// dtype 依存スウィズルのホスト側**参照実装**（純関数）。
///
/// smem タイル内の列インデックス `col` を、行 `row` に応じて XOR で
/// 並べ替える（TileLang 由来の一般形「周期 = 8 / dtype バイト数」を
/// `kernels_transpose::SWIZZLE_PERIOD_F32`/`SWIZZLE_PERIOD_F16` として
/// 具体化し、周期内の下位ビットのみを反転する CUTLASS/TileLang 系の
/// 定番スウィズル形）。
///
/// `period`（`8 / elem_bytes`）が 2 のべき乗であることを利用し、
/// `col` の `period` 未満の下位ビットのみを `row` の対応ビットと XOR する
/// （`mask = period - 1`）。XOR は対合（involution）のため、固定した
/// `row` に対する `col -> swizzled_col` の写像は `0..TILE` 上で全単射に
/// なる（`period` が `TILE` を割り切る限り、`period` 個のブロックそれぞれ
/// が独立に置換されるため。`TRANSPOSE_TILE`（32）は 2/4 いずれでも割り切れる。
/// 単体テスト `swizzle_is_bijective_per_row` が全域を機械検査する）。
///
/// # panic 契約
///
/// `elem_bytes` が 8 を割り切らない値（1/2/4/8 以外）で呼ばれた場合のみ
/// `debug_assert!` で検査する。呼び出し元（本ファイル内テスト・
/// `kernels_transpose.rs` の定数から導出される値のみ）はいずれも 2/4 を
/// 渡すため、通常経路では到達しない防御的検査（`swizzle.rs::
/// swizzled_block_idx` と同じ「外部入力を直接受けないため debug_assert
/// のみで足りる」判断）。
///
/// # 呼び出し文脈（生産経路からは到達しない参照実装）
///
/// 本関数は `kernels_transpose.rs::transpose_smem_source_f32`/
/// `transpose_smem_source_f16` が生成する CUDA C++ 側の `SMEM_COL` マクロ
/// と同一設計を Rust で独立に再実装したものであり、GPU 実行経路からは
/// 直接呼ばれない（呼ばれるのは生成された CUDA 文字列であり、この Rust
/// 関数そのものではない。`swizzle.rs` 冒頭コメントと同じ位置づけ）。
///
/// `#[allow(dead_code)]`: 呼び出し元は本ファイル末尾の `#[cfg(test)]`
/// テスト（`swizzle_is_bijective_per_row` 等）のみであり、`cargo build`
/// （テスト cfg なし）では未使用と判定される。`swizzle.rs::
/// swizzled_block_idx` の `#[allow(dead_code)]` と同じ判断パターン
/// （参照実装の正しさをテストで独立検証するための意図的な設計）。
#[allow(dead_code)]
pub fn swizzled_smem_col(row: u32, col: u32, elem_bytes: u32) -> u32 {
    debug_assert!(elem_bytes.is_power_of_two() && elem_bytes <= 8);
    let period = 8 / elem_bytes;
    debug_assert!(period.is_power_of_two());
    let mask = period - 1;
    (col & !mask) | ((col & mask) ^ (row & mask))
}

/// GEMM 呼び出しの `m`/`n` とホスト側スライス長の整合性を検証する。
///
/// `gemm.rs::validate_gemm_dims` と同じ思想（GPU 起動前に不整合な形状値を
/// 拒否する。OWASP A03・`.claude/rules/security.md`）だが、転置は `k` を
/// 持たず `src` の長さが `m*n` と一致するかのみを検証する。
pub(crate) fn validate_transpose_dims(src_len: usize, m: u32, n: u32) -> Result<(), CudaError> {
    let mn =
        (m as usize)
            .checked_mul(n as usize)
            .ok_or_else(|| CudaError::InvalidTransposeShape {
                detail: format!("m*n overflows usize: m={m}, n={n}"),
            })?;
    if src_len != mn {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!("src length mismatch: expected {mn} (m*n), actual {src_len}"),
        });
    }
    // カーネル引数（`int m, int n`）は C の 32bit 符号付き整数のため、
    // `gemm.rs::validate_gemm_dims` と同じ理由で i32::MAX 上限を検証する。
    if m > i32::MAX as u32 || n > i32::MAX as u32 {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!("m/n must fit in i32 (kernel argument type): m={m}, n={n}"),
        });
    }
    if mn > i32::MAX as usize {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!(
                "m*n must fit in i32 (kernel index arithmetic is 32bit int): m={m}, n={n}, \
                 m*n={mn}"
            ),
        });
    }
    Ok(())
}

/// 出力バッファ `dst_dev` の長さが `out_m*out_n` と一致することを検証する。
///
/// `gemm.rs::validate_output_len` の転置版。`launch_naive_f32`/
/// `launch_smem_f32`/`launch_tiled_transposed_f32` は出力バッファを起動側
/// で確保せず呼び出し元から受け取るため、[`validate_transpose_dims`] が
/// 検証する「入力長が `m*n` と一致する」だけでは C 側の OOB 書き込みを
/// 防げない（`gemm.rs::validate_output_len` ドキュメンテーションコメント
/// 「PR #349 codex-review 指摘 P0」と同じ理由）。
pub(crate) fn validate_transpose_output_len(
    dst_len: usize,
    out_m: u32,
    out_n: u32,
) -> Result<(), CudaError> {
    let expected = (out_m as usize)
        .checked_mul(out_n as usize)
        .ok_or_else(|| CudaError::InvalidTransposeShape {
            detail: format!("out_m*out_n overflows usize: out_m={out_m}, out_n={out_n}"),
        })?;
    if dst_len != expected {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!(
                "dst length mismatch: expected {expected} (out_m*out_n), actual {dst_len}"
            ),
        });
    }
    Ok(())
}

/// `m`/`n`（転置後の出力形状は `n`×`m`）から [`TRANSPOSE_TILE_BLOCK_DIM`]
/// 基準の `div_ceil` グリッドを構築する（`gemm.rs::launch_config` と同型）。
///
/// `pub(crate)`: イシュー #1214 の `gemm.rs::CudaGemm::transpose_to_pooled`
/// が VJP 専用 NT/TN 転置入口の smem 転置カーネル起動に同じグリッド算術を
/// 再利用する（`CudaTranspose`（本構造体）とは独立のコンパイル単位・
/// ハンドルだが、起動時のグリッド構成は同一カーネルソース
/// （`kernels_transpose::transpose_smem_source_f32`）を前提にしており
/// 重複させない）。
pub(crate) fn tiled_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_transpose::TRANSPOSE_TILE),
        m.div_ceil(kernels_transpose::TRANSPOSE_TILE),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: TRANSPOSE_TILE_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

fn naive_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(NAIVE_TRANSPOSE_BLOCK_DIM.0),
        m.div_ceil(NAIVE_TRANSPOSE_BLOCK_DIM.1),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: NAIVE_TRANSPOSE_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

/// naive／smem（パディング・パディング+スウィズル）転置カーネル、および
/// GEMM epilogue 融合転置（opt-in）のコンパイル済みハンドルを保持する。
pub struct CudaTranspose {
    stream: Arc<CudaStream>,
    naive_f32: CudaFunction,
    naive_f16: CudaFunction,
    smem_f32_pad: CudaFunction,
    smem_f32_swizzle: CudaFunction,
    smem_f16_pad: CudaFunction,
    smem_f16_swizzle: CudaFunction,
    /// GEMM epilogue 融合転置（`kernels_transpose::TILED_TRANSPOSED_F32`）。
    /// naive/smem 6 カーネルと同様 `#include` を使わず全 compute
    /// capability で成立するため `new` の早期 return（`?`）に合流させる
    /// （`gemm.rs::CudaGemm::tiled_bias_act_f32` と同じ扱い。WMMA 系のような
    /// `Option` 化は行わない）。`ops.rs`／`gemm_auto.rs` へは結線しない
    /// opt-in 経路であり、`CudaTranspose` を明示的に構築した呼び出し元
    /// （テスト・ベンチ）のみが到達する（実装計画 3.3 節）。
    tiled_transposed_f32: CudaFunction,
}

impl CudaTranspose {
    /// `device` 上で naive／smem／GEMM epilogue 融合の各転置カーネルを
    /// NVRTC コンパイルし保持するハンドルを構築する。
    ///
    /// `gemm.rs::CudaGemm::new` とは独立したコンパイル単位であり、
    /// `CudaGemm` の生存期間・コンパイル集合には一切影響しない
    /// （実装計画 3.3 節「`CudaGemm::new` の eager コンパイル集合には
    /// 追加しない」を、別構造体として分離することで満たす）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let naive_f32_ptx = compile_ptx(kernels_transpose::TRANSPOSE_NAIVE_F32, arch)?;
        let naive_f16_ptx = compile_ptx(kernels_transpose::TRANSPOSE_NAIVE_F16, arch)?;
        let smem_f32_pad_ptx =
            compile_ptx(&kernels_transpose::transpose_smem_source_f32(false), arch)?;
        let smem_f32_swizzle_ptx =
            compile_ptx(&kernels_transpose::transpose_smem_source_f32(true), arch)?;
        let smem_f16_pad_ptx =
            compile_ptx(&kernels_transpose::transpose_smem_source_f16(false), arch)?;
        let smem_f16_swizzle_ptx =
            compile_ptx(&kernels_transpose::transpose_smem_source_f16(true), arch)?;
        let tiled_transposed_f32_ptx = compile_ptx(kernels_transpose::TILED_TRANSPOSED_F32, arch)?;

        let naive_f32 = device
            .context()
            .load_module(naive_f32_ptx)?
            .load_function("transpose_naive_f32")?;
        let naive_f16 = device
            .context()
            .load_module(naive_f16_ptx)?
            .load_function("transpose_naive_f16")?;
        let smem_f32_pad = device
            .context()
            .load_module(smem_f32_pad_ptx)?
            .load_function("transpose_smem_f32")?;
        let smem_f32_swizzle = device
            .context()
            .load_module(smem_f32_swizzle_ptx)?
            .load_function("transpose_smem_f32")?;
        let smem_f16_pad = device
            .context()
            .load_module(smem_f16_pad_ptx)?
            .load_function("transpose_smem_f16")?;
        let smem_f16_swizzle = device
            .context()
            .load_module(smem_f16_swizzle_ptx)?
            .load_function("transpose_smem_f16")?;
        let tiled_transposed_f32 = device
            .context()
            .load_module(tiled_transposed_f32_ptx)?
            .load_function("gemm_tiled_transposed_f32")?;

        // 静的共有メモリ使用量が 48KiB 上限を下回ることをコンパイル時に
        // 機械検査する（`kernels_mma.rs::MMA_SHARED_MEM_BYTES` の const
        // assert と同型。`kernels_transpose.rs` 側の単体テスト
        // `tiled_transposed_shared_mem_within_48kib_limit` と同じ主張を
        // `new` の呼び出し経路上でも保証する）。
        const _: () = assert!(
            kernels_transpose::TILED_TRANSPOSED_SHARED_MEM_BYTES < 48 * 1024,
            "TILED_TRANSPOSED_SHARED_MEM_BYTES must stay below the 48KiB static shared \
             memory floor guaranteed across compute capabilities"
        );

        Ok(Self {
            stream: device.stream().clone(),
            naive_f32,
            naive_f16,
            smem_f32_pad,
            smem_f32_swizzle,
            smem_f16_pad,
            smem_f16_swizzle,
            tiled_transposed_f32,
        })
    }

    // --- f32: デバイス常駐 upload/launch/download 分離 API ---
    //
    // `gemm_mma.rs::CudaMmaGemm::upload_f16`/`alloc_output_f16`/`launch_f16`
    // と同じ理由（advisor 指摘）で、H2D/D2H を含まない「GPU 実行のみ」の
    // 区間をベンチマークが計測できるよう公開する。`examples/
    // gemm_transpose_bench.rs` は当初 `run_*`（転送込み）を計測していたが、
    // PCIe 転送時間（数 ms オーダー）がカーネル本体の実行時間（数百 us
    // オーダー）を支配し、naive/smem パディング/スウィズルの差がすべて
    // 転送時間の中に埋もれて「常に比 1.00 に近い値」しか観測できない
    // 欠陥があった。以下の `upload_f32`/`alloc_output_f32`/`launch_*_f32`/
    // `download_f32` を使い、転送をベンチのウォームアップ外（計測区間外）
    // へ切り出す。

    /// `src`（f32）をデバイスへ転送する。ベンチマークが計測区間外で使う
    /// ため公開する（`gemm_mma.rs::CudaMmaGemm::upload_f16` と同じ理由）。
    pub fn upload_f32(&self, src: &[f32]) -> Result<CudaSlice<f32>, CudaError> {
        Ok(self.stream.clone_htod(src)?)
    }

    /// `m x n` 転置出力用のゼロ初期化デバイスバッファを確保する。
    pub fn alloc_output_f32(&self, m: u32, n: u32) -> Result<CudaSlice<f32>, CudaError> {
        Ok(self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?)
    }

    /// デバイス常駐バッファ（f32）をホストへ回収する。
    ///
    /// 同期点（#1013）: 常駐 `launch_*_f32` は非同期投入のみで完了を
    /// 待たないため、本関数が readback ヘルパー経由で完了を確定する
    /// （`memory.rs::readback` ドキュメンテーションコメント参照）。
    pub fn download_f32(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaError> {
        crate::memory::readback(&self.stream, dev)
    }

    /// ストリームの完了を明示的に待つ（イシュー #1013。
    /// `gemm.rs::CudaGemm::synchronize` と同じ理由の公開 API）。
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.stream.synchronize()?)
    }

    /// 素朴転置（f32）を、デバイス常駐済みの `src_dev`/`dst_dev` に対して
    /// 起動し完了を待つ（H2D/D2H を含まない区間）。safe な公開 API のため
    /// `gemm_mma.rs::launch_f16` と同じく本関数自身が形状検証を行う
    /// （呼び出し元の事前検証に依存しない。PR #349 codex-review 指摘 P0
    /// と同型の判断）。
    pub fn launch_naive_f32(
        &self,
        src_dev: &CudaSlice<f32>,
        dst_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
    ) -> Result<(), CudaError> {
        validate_transpose_dims(src_dev.len(), m, n)?;
        validate_transpose_output_len(dst_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        self.launch_f32(
            &self.naive_f32,
            src_dev,
            dst_dev,
            m,
            n,
            naive_launch_config(m, n),
        )
    }

    /// smem タイル転置（f32）を、デバイス常駐済みの `src_dev`/`dst_dev`
    /// に対して起動する。[`Self::launch_naive_f32`] と同じ「upload/
    /// download を含まない」契約。`swizzle` の意味は [`Self::run_smem_f32`]
    /// と同じ。
    pub fn launch_smem_f32(
        &self,
        src_dev: &CudaSlice<f32>,
        dst_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        swizzle: bool,
    ) -> Result<(), CudaError> {
        validate_transpose_dims(src_dev.len(), m, n)?;
        validate_transpose_output_len(dst_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        let func = if swizzle {
            &self.smem_f32_swizzle
        } else {
            &self.smem_f32_pad
        };
        self.launch_f32(func, src_dev, dst_dev, m, n, tiled_launch_config(m, n))
    }

    /// GEMM epilogue 融合転置（opt-in）を、デバイス常駐済みの
    /// `a_dev`/`b_dev`/`c_t_dev` に対して起動する。[`Self::launch_naive_f32`]
    /// と同じ「upload/download を含まない」契約。
    pub fn launch_tiled_transposed_f32(
        &self,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_t_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_tiled_transposed_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_transpose_output_len(c_t_dev.len(), n, m)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // k==0 は `run_tiled_transposed_f32` の早期 return と同じ理由
            // （K 方向の累積対象が存在しない = C は全 0 という GEMM の数学的
            // 定義。`gemm.rs::run_tiled_bias_act_f32` の「k == 0」節と同じ
            // 契約）。ただし本関数は safe な公開 API であり、任意の
            // `CudaSlice<f32>`（確保・再利用した非ゼロバッファを含む）を
            // `c_t_dev` として受け取れる。カーネル起動を省略するだけでは
            // `c_t_dev` の既存内容が残ってしまい「正常終了時は必ず
            // `(A @ B)^T` の値になる」という数値契約を破る（codex-review
            // 指摘 P1・PR #690）。カーネル起動を省略する代わりに
            // `c_t_dev` を明示的にゼロクリアすることで、呼び出し元が
            // ゼロ初期化済みバッファ（`alloc_output_f32`）を渡したかに
            // 依らず数学的に正しい結果を保証する。
            // `memset_zeros` は同一ストリーム上への非同期投入に留まる。
            // 旧実装はここで `synchronize()` を挟んでいたが、イシュー
            // #1013 で本関数（常駐 API）の契約を「ストリームへの非同期
            // 投入のみ。完了保証は呼び出し元の次の同期点（`download_*`／
            // `MemoryOps::download`／明示 `synchronize`）に委ねる」へ
            // 統一した（`docs/backend-cuda-async-execution-design.md`
            // §3〜§4）。単一ストリームの FIFO 順序保証により、後続の
            // 同期点は本 `memset_zeros` を含む全ての先行投入を合わせて
            // 待つため、codex-review 指摘（PR #690）が懸念した「stale 値
            // の観測」「非同期実行エラーの取りこぼし」は生じない（同期点
            // 自体が readback ヘルパー等へ集約され、契約として保証される
            // ため）。
            self.stream.memset_zeros(c_t_dev)?;
            return Ok(());
        }

        let cfg = tiled_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数（a_dev/b_dev/c_t_dev・m_i/n_i/k_i）はホスト側
        // 検証（validate_tiled_transposed_gemm_dims・
        // validate_transpose_output_len）済みの m/n/k から導出しており、
        // カーネル内の手動境界チェック（アキュムレーション部の三項ガード
        // ＋epilogue の転置ストアガード。`kernels_transpose.rs::
        // TILED_TRANSPOSED_F32` ドキュメンテーションコメント参照、REQ-8）
        // と合わせて OOB 読み書きが起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(&self.tiled_transposed_f32)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_t_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点へ
        // 委ねる（本ファイル冒頭の該当コメント・設計文書 §3〜§4）。
        Ok(())
    }

    /// f32 カーネル共通の起動手続き（デバイス常駐版。`launch_naive_f32`/
    /// `launch_smem_f32` で共有。呼び出し元が既に形状検証済みのため本関数
    /// 自体は検証しない）。
    fn launch_f32(
        &self,
        func: &CudaFunction,
        src_dev: &CudaSlice<f32>,
        dst_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        cfg: LaunchConfig,
    ) -> Result<(), CudaError> {
        let (m_i, n_i) = (m as i32, n as i32);

        // SAFETY: src_dev（呼び出し元検証済みの m*n 要素）・dst_dev
        // （同じく m*n 要素）は呼び出し元（`launch_naive_f32`/
        // `launch_smem_f32`）が検証済みの m/n から導出しており、カーネル
        // 内の手動境界チェック（REQ-8。呼び出し元カーネルソース
        // ドキュメンテーションコメント参照）と合わせて OOB を防ぐ。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(src_dev)
                .arg(dst_dev)
                .arg(&m_i)
                .arg(&n_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点へ
        // 委ねる。
        Ok(())
    }

    /// 素朴転置（f32）を実行する。`dst[col*m+row] = src[row*n+col]`。
    /// smem 版との A/B 計測の基準点（実装計画 3.4 節）。
    ///
    /// `m == 0 || n == 0` は no-op（`gemm.rs::run_f32_kernel` と同じ理由。
    /// 0 次元グリッド起動を CUDA driver が拒否するため回避する）。
    ///
    /// **upload → launch → download の薄いラッパー**（[`Self::upload_f32`]・
    /// [`Self::launch_naive_f32`]・[`Self::download_f32`]。転送込みの
    /// 簡便 API として維持しつつ、H2D/D2H を除いた計測が必要な呼び出し元
    /// （`examples/gemm_transpose_bench.rs`）は分離 API を直接使う）。
    pub fn run_naive_f32(&self, src: &[f32], m: u32, n: u32) -> Result<Vec<f32>, CudaError> {
        validate_transpose_dims(src.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let src_dev = self.upload_f32(src)?;
        let mut dst_dev = self.alloc_output_f32(m, n)?;
        self.launch_naive_f32(&src_dev, &mut dst_dev, m, n)?;
        self.download_f32(&dst_dev)
    }

    /// 素朴転置（f16）を実行する。手順は [`Self::run_naive_f32`] と同一
    /// （f16 は launch 分離 API を持たない。実装計画のスコープは f32
    /// ベンチの精度確保が目的で、f16 は既存 `run_*` の転送込み計測で
    /// 十分と判断した。分離が必要になった場合は [`Self::run_naive_f32`]
    /// と同型で追加できる）。
    pub fn run_naive_f16(&self, src: &[f16], m: u32, n: u32) -> Result<Vec<f16>, CudaError> {
        validate_transpose_dims(src.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        self.run_f16(&self.naive_f16, src, m, n, naive_launch_config(m, n))
    }

    /// smem タイル転置（f32）を実行する。`swizzle == true` でパディング +
    /// dtype 依存スウィズル変種、`false` でパディングのみ変種を使う
    /// （実機 A/B でどちらが有効か計測するための opt-in 分岐。実装計画
    /// 3.4 節）。[`Self::run_naive_f32`] と同じ「薄いラッパー」構造。
    pub fn run_smem_f32(
        &self,
        src: &[f32],
        m: u32,
        n: u32,
        swizzle: bool,
    ) -> Result<Vec<f32>, CudaError> {
        validate_transpose_dims(src.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let src_dev = self.upload_f32(src)?;
        let mut dst_dev = self.alloc_output_f32(m, n)?;
        self.launch_smem_f32(&src_dev, &mut dst_dev, m, n, swizzle)?;
        self.download_f32(&dst_dev)
    }

    /// smem タイル転置（f16）を実行する。[`Self::run_naive_f16`] と同じ
    /// 理由で launch 分離 API は持たない。
    pub fn run_smem_f16(
        &self,
        src: &[f16],
        m: u32,
        n: u32,
        swizzle: bool,
    ) -> Result<Vec<f16>, CudaError> {
        validate_transpose_dims(src.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let func = if swizzle {
            &self.smem_f16_swizzle
        } else {
            &self.smem_f16_pad
        };
        self.run_f16(func, src, m, n, tiled_launch_config(m, n))
    }

    /// GEMM epilogue 融合転置（opt-in）を実行する。`C^T = (A @ B)^T`
    /// （`m x k` @ `k x n` → 転置して `n x m`）。中間バッファ `C` を HBM へ
    /// 書かず、epilogue で smem 経由の転置ストアまで完結させる
    /// （実装計画 3.3 節）。[`Self::run_naive_f32`] と同じ「薄いラッパー」
    /// 構造（[`Self::launch_tiled_transposed_f32`] を参照）。
    ///
    /// ホスト側形状検証は `gemm.rs::validate_gemm_dims`/
    /// `validate_tiled_k_bound` と同じ制約（`a.len()==m*k`・`b.len()==k*n`・
    /// `i32` 積オーバーフロー防止）をこのモジュール内で独立に再検証する
    /// （`gemm.rs` を変更しない実装方針・実装計画 4 節「変更しないもの」
    /// のため、検証ロジックを共有せず同型の検証を本モジュール内に閉じる）。
    pub fn run_tiled_transposed_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_tiled_transposed_gemm_dims(a.len(), b.len(), m, n, k)?;
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            // `gemm.rs::run_f32_kernel` の k==0 早期 return と同一の理由
            // （advisor 指摘: k==0 のとき a/b は 0 要素スライスになり、
            // `upload_f32` の `clone_htod` がそのまま 0 バイトのデバイス
            // バッファ確保を driver に要求する。一部環境の CUDA driver は
            // 0 バイト確保を `CUDA_ERROR_INVALID_VALUE` で拒否しうる。
            // `launch_tiled_transposed_f32` 側の k==0 スキップは
            // デバイス常駐入力を既に持つ呼び出し元向けの契約であり、
            // upload 自体を回避する本ガードとは独立に必要）。
            return Ok(vec![0.0f32; (n as usize) * (m as usize)]);
        }

        let a_dev = self.upload_f32(a)?;
        let b_dev = self.upload_f32(b)?;
        // 出力は n x m（転置後の形状）。
        let mut c_t_dev = self.alloc_output_f32(n, m)?;
        self.launch_tiled_transposed_f32(&a_dev, &b_dev, &mut c_t_dev, m, n, k)?;
        self.download_f32(&c_t_dev)
    }

    /// f16 カーネル共通の起動手続き（転送込み。f16 は launch 分離 API を
    /// 持たないため従来どおりの構造を保つ）。
    fn run_f16(
        &self,
        func: &CudaFunction,
        src: &[f16],
        m: u32,
        n: u32,
        cfg: LaunchConfig,
    ) -> Result<Vec<f16>, CudaError> {
        let src_dev = self.stream.clone_htod(src)?;
        let mut dst_dev = self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?;
        let (m_i, n_i) = (m as i32, n as i32);

        // SAFETY: run_f32/launch_f32 と同一の根拠。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&src_dev)
                .arg(&mut dst_dev)
                .arg(&m_i)
                .arg(&n_i)
                .launch(cfg)?;
        }
        // 同期点は readback ヘルパーへ集約（#1013）。
        let dst_host = crate::memory::readback(&self.stream, &dst_dev)?;
        Ok(dst_host)
    }
}

/// [`CudaTranspose::run_tiled_transposed_f32`] 専用の形状検証。
/// `gemm.rs::validate_gemm_dims` と同一の検証内容（`a.len()==m*k`・
/// `b.len()==k*n`・`i32` 積オーバーフロー防止）をこのモジュール内に
/// 独立実装する（上記メソッドのドキュメンテーションコメント参照）。
fn validate_tiled_transposed_gemm_dims(
    a_len: usize,
    b_len: usize,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), CudaError> {
    let (m_usize, n_usize, k_usize) = (m as usize, n as usize, k as usize);
    let mk = m_usize
        .checked_mul(k_usize)
        .ok_or_else(|| CudaError::InvalidTransposeShape {
            detail: format!("m*k overflows usize: m={m}, k={k}"),
        })?;
    let kn = k_usize
        .checked_mul(n_usize)
        .ok_or_else(|| CudaError::InvalidTransposeShape {
            detail: format!("k*n overflows usize: k={k}, n={n}"),
        })?;
    let mn = m_usize
        .checked_mul(n_usize)
        .ok_or_else(|| CudaError::InvalidTransposeShape {
            detail: format!("m*n overflows usize: m={m}, n={n}"),
        })?;
    if a_len != mk {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!("a length mismatch: expected {mk} (m*k), actual {a_len}"),
        });
    }
    if b_len != kn {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!("b length mismatch: expected {kn} (k*n), actual {b_len}"),
        });
    }
    if m > i32::MAX as u32 || n > i32::MAX as u32 || k > i32::MAX as u32 {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!("m/n/k must fit in i32 (kernel argument type): m={m}, n={n}, k={k}"),
        });
    }
    if mk > i32::MAX as usize || kn > i32::MAX as usize || mn > i32::MAX as usize {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!(
                "m*k, k*n, m*n must fit in i32 (kernel index arithmetic is 32bit int): \
                 m={m}, n={n}, k={k}, m*k={mk}, k*n={kn}, m*n={mn}"
            ),
        });
    }
    // tiled カーネルのタイルインデックス算術保護（`gemm.rs::
    // validate_tiled_k_bound` と同一根拠。`kernels_transpose::
    // TRANSPOSE_TILE` は `kernels::TILE` と同値の 32）。
    let limit = i32::MAX as u32 - (kernels_transpose::TRANSPOSE_TILE - 1);
    if k > limit {
        return Err(CudaError::InvalidTransposeShape {
            detail: format!(
                "k must not exceed i32::MAX - (TRANSPOSE_TILE - 1) for tiled kernel \
                 tile-index arithmetic: k={k}, limit={limit}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// [`swizzled_smem_col`] が固定行に対して全単射であることを、
    /// f32（`elem_bytes=4`・周期 2）・f16（`elem_bytes=2`・周期 4）双方の
    /// 全 `row`×`col` 組合せで機械検査する（実装計画 3.2 節「全単射性」）。
    #[test]
    fn swizzle_is_bijective_per_row() {
        for elem_bytes in [4u32, 2] {
            for row in 0..kernels_transpose::TRANSPOSE_TILE {
                let mut seen: HashSet<u32> =
                    HashSet::with_capacity(kernels_transpose::TRANSPOSE_TILE as usize);
                for col in 0..kernels_transpose::TRANSPOSE_TILE {
                    let sc = swizzled_smem_col(row, col, elem_bytes);
                    assert!(
                        sc < kernels_transpose::TRANSPOSE_TILE,
                        "elem_bytes={elem_bytes} row={row} col={col}: swizzled_col={sc} が \
                         TILE 範囲外です"
                    );
                    assert!(
                        seen.insert(sc),
                        "elem_bytes={elem_bytes} row={row} col={col}: swizzled_col={sc} が \
                         重複しています（全単射性違反）"
                    );
                }
            }
        }
    }

    /// バンク位相分散: f32（周期 2）・f16（周期 4）いずれも、同一列で
    /// 異なる行を辿ったときスウィズル後の列が周期分だけ変化することを
    /// 具体例で固定する（周期の設計意図「行ごとに位相をずらす」の直接的な
    /// 検証。実装計画 3.2 節「バンク位相分散」）。
    #[test]
    fn swizzle_shifts_phase_across_period_rows() {
        // f32（周期 2）: col=0 は row の偶奇で 0/1 を往復する。
        assert_eq!(swizzled_smem_col(0, 0, 4), 0);
        assert_eq!(swizzled_smem_col(1, 0, 4), 1);
        assert_eq!(swizzled_smem_col(2, 0, 4), 0);
        assert_eq!(swizzled_smem_col(3, 0, 4), 1);

        // f16（周期 4）: col=0 は row mod 4 に応じて 0..4 を巡回する。
        for row in 0..8u32 {
            let expected = row % 4;
            assert_eq!(swizzled_smem_col(row, 0, 2), expected, "row={row}");
        }
    }

    /// `row=0` では swizzle が恒等写像になる（`(col & mask) ^ (0 & mask)
    /// == col & mask`）ことを確認する（退化ケースの固定）。
    #[test]
    fn swizzle_row_zero_is_identity() {
        for elem_bytes in [4u32, 2] {
            for col in 0..kernels_transpose::TRANSPOSE_TILE {
                assert_eq!(swizzled_smem_col(0, col, elem_bytes), col);
            }
        }
    }

    /// [`validate_transpose_dims`] が `m*n` とスライス長の不一致・`i32`
    /// 上限超過を拒否することを検査する（`gemm.rs::validate_gemm_dims` の
    /// 単体テストと同型）。
    #[test]
    fn validate_transpose_dims_rejects_length_mismatch() {
        let err = validate_transpose_dims(5, 2, 3).expect_err("2*3=6 != 5 must be rejected");
        assert!(matches!(err, CudaError::InvalidTransposeShape { .. }));
    }

    #[test]
    fn validate_transpose_dims_accepts_matching_length() {
        validate_transpose_dims(6, 2, 3).expect("2*3=6 matches src length");
        validate_transpose_dims(0, 0, 0).expect("0x0 is a valid no-op shape");
    }

    /// [`validate_tiled_transposed_gemm_dims`] が `gemm.rs::
    /// validate_gemm_dims` と同型の検証（長さ不一致・i32 積オーバーフロー）
    /// を行うことを検査する。
    #[test]
    fn validate_tiled_transposed_gemm_dims_rejects_length_mismatch() {
        let err = validate_tiled_transposed_gemm_dims(5, 6, 2, 3, 2)
            .expect_err("a length 5 != m*k=4 must be rejected");
        assert!(matches!(err, CudaError::InvalidTransposeShape { .. }));
    }

    #[test]
    fn validate_tiled_transposed_gemm_dims_accepts_matching_shape() {
        // m=2, k=2, n=3: a is 2x2 (len 4), b is 2x3 (len 6).
        validate_tiled_transposed_gemm_dims(4, 6, 2, 3, 2).expect("shape matches");
    }
}
