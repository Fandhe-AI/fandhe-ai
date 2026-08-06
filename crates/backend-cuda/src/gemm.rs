//! naive／tiled GEMM の起動 API（NVRTC コンパイル・保持・実行）。
//!
//! `CudaGemm` は `crates/tensor-core` の演算グラフ実行（TASK-1.9・#43 で
//! `BackendOps` へ結線予定）から見て、「`CudaDevice` を渡すと naive／tiled
//! GEMM カーネル（f32/f16 各 2 種、計 4 カーネル）をコンパイル・保持し、
//! 以降はホスト側スライスを渡すだけで GPU 実行できる」境界を担う。カーネル
//! ソース自体は `kernels.rs`（NVRTC 文字列埋め込み）に閉じ込め、本モジュールは
//! コンパイル結果（`CudaFunction`）の保持とメモリ転送・起動手続きのみを扱う。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs`
//! の `CudaGemm::new`／`run_naive_f32`／`run_naive_f16`／`run_tiled_f32`／
//! `run_tiled_f16`。productize にあたり PoC から変更した点:
//!
//! 1. **型付きエラー化**（`.claude/rules/coding-rust.md`）: PoC の
//!    `CudaGemmError(String)` を廃し、`CudaError`（`error.rs`）に統一した。
//! 2. **ホスト側形状検証を追加**: PoC は形状検証を持たず、不整合な
//!    スライス長・オーバーフローする m/n/k をそのままカーネル引数へ渡す
//!    経路が存在した。本実装は GPU 起動前に [`validate_gemm_dims`] で
//!    拒否し `CudaError::InvalidShape` を返す（OWASP A03 対応。
//!    `.claude/rules/security.md`）。tiled 経路はさらに
//!    [`validate_tiled_k_bound`] で `k` の追加上限を検証する（後述）。
//! 3. **`Duration` 非返却**: PoC の `run_*` はカーネル実行時間を計測して
//!    返していたが、計測は `bench-harness` 側の責務であり本クレートの
//!    責務境界外と判断した（TASK-8.x・`bench-harness` の同期方式節参照）。
//! 4. **naive/tiled 共通ヘルパーへの整理**: PoC の `run_f32`/`run_f16`
//!    （`func`・`block_dim` を引数化した private ヘルパー）の構造を踏襲し、
//!    naive/tiled 双方の起動手続き（転送・起動・同期・回収）を
//!    [`CudaGemm::run_f32_kernel`]／[`CudaGemm::run_f16_kernel`] に集約する
//!    （#34 で naive 専用だった手続きを共通化）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels;
use crate::nvrtc::compile_ptx;

/// naive GEMM カーネル起動 1 回あたりのブロック次元（16x16 = 256 スレッド）。
///
/// PoC-v2-3（`cuda/mod.rs:174`）と同じ値を踏襲する。tiled 版の
/// `kernels::TILE`（32x32。[`TILED_BLOCK_DIM`]）とは独立したパラメータであり、
/// 共有メモリを使わない naive カーネルの `__shared__` 配列サイズ制約は受けない。
const NAIVE_BLOCK_DIM: (u32, u32, u32) = (16, 16, 1);

/// tiled GEMM カーネル起動 1 回あたりのブロック次元。
///
/// `kernels::TILE` x `kernels::TILE` に固定する必要がある（カーネル内
/// `__shared__ float as_tile[TILE][TILE]` 等はブロック内スレッド数と
/// 1:1 対応するコンパイル時定数のため、ここがずれるとタイル境界外の
/// スレッドが共有メモリを書かない一方でロード先が欠落し誤った積和になる）。
const TILED_BLOCK_DIM: (u32, u32, u32) = (kernels::TILE, kernels::TILE, 1);

/// naive／tiled GEMM カーネル（f32/f16 各 2 種）のコンパイル済みハンドルを保持する。
///
/// `stream` は [`CudaDevice`] から `Arc` クローンで受け取る（`device.rs` の
/// 共有契約どおり）。`new` 時に 4 カーネルを一括コンパイルするのは、
/// `nvrtc::compile_ptx` の呼び出し契約「`Box::leak` によるアーキテクチャ
/// 文字列リークはデバイスあたり定数回に限る」を守るためであり、
/// `run_naive_*`／`run_tiled_*` 呼び出しのたびに再コンパイルしない。
pub struct CudaGemm {
    stream: Arc<CudaStream>,
    naive_f32: CudaFunction,
    naive_f16: CudaFunction,
    tiled_f32: CudaFunction,
    tiled_f16: CudaFunction,
}

/// GEMM 呼び出しの `m`/`n`/`k` とホスト側スライス長の整合性を検証する。
///
/// GPU 起動前に呼ぶことで、不整合な形状値がカーネル引数（`int m, n, k`）や
/// デバイスバッファ確保サイズへそのまま渡る経路を断つ（OWASP A03。
/// `.claude/rules/security.md`「外部フォーマットパースは長さ・形状の検証を
/// 先に行う」と同じ思想を GEMM 起動入口に適用）。`backend-cpu::gemm::
/// validate_dims`（`crates/backend-cpu/src/gemm.rs:146`）と同種の検証だが、
/// 本関数はさらに「カーネル引数が C の `int`（`i32`）であること」を理由に
/// `i32::MAX` 上限チェックを追加で行う点が異なる（PoC には存在しなかった
/// 検証。上記モジュールコメント参照）。
///
/// `pub(crate)`: `tests/gemm_naive.rs` が実機非依存の単体テストとして
/// 直接呼べるよう `#[cfg(test)]` 外の通常関数として公開範囲をクレート内に
/// 限定する。
pub(crate) fn validate_gemm_dims(
    a_len: usize,
    b_len: usize,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), CudaError> {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;

    let mk = m_usize
        .checked_mul(k_usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("m*k overflows usize: m={m}, k={k}"),
        })?;
    let kn = k_usize
        .checked_mul(n_usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("k*n overflows usize: k={k}, n={n}"),
        })?;
    // m*n はカーネル引数には現れないが、`alloc_zeros::<f32>((m*n) as usize)`
    // の確保サイズ計算（`gemm.rs::run_f32`/`run_f16`）で使うため、こちらも
    // 起動前に検証する。
    let mn = m_usize
        .checked_mul(n_usize)
        .ok_or_else(|| CudaError::InvalidShape {
            detail: format!("m*n overflows usize: m={m}, n={n}"),
        })?;

    if a_len != mk {
        return Err(CudaError::InvalidShape {
            detail: format!("a length mismatch: expected {mk} (m*k), actual {a_len}"),
        });
    }
    if b_len != kn {
        return Err(CudaError::InvalidShape {
            detail: format!("b length mismatch: expected {kn} (k*n), actual {b_len}"),
        });
    }

    // カーネル引数（`int m, int n, int k`）は C の 32bit 符号付き整数のため、
    // i32::MAX を超える値を渡すと未定義の切り詰め・符号反転が起こりうる。
    // `u32` 引数の型レベル上限（4294967295）より厳しいこの制約をここで拒否する。
    if m > i32::MAX as u32 || n > i32::MAX as u32 || k > i32::MAX as u32 {
        return Err(CudaError::InvalidShape {
            detail: format!("m/n/k must fit in i32 (kernel argument type): m={m}, n={n}, k={k}"),
        });
    }

    // Cursor Bugbot 指摘（PR #240）: `kernels.rs` の naive カーネルは
    // `row * k + p`／`p * n + col`／`row * n + col` を C の `int`（i32）
    // 算術で計算する。m/n/k 各々が i32::MAX に収まっていても、その積
    // （mk・kn・mn）が i32::MAX を超えるとインデックス計算そのものが
    // 32bit 符号付き整数の範囲でラップし、範囲外読み書きを引き起こしうる。
    // ここでは実際にカーネルが触れる最大インデックス（各積 - 1）が
    // i32 に収まることを起動前に検証する。
    if mk > i32::MAX as usize || kn > i32::MAX as usize || mn > i32::MAX as usize {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "m*k, k*n, m*n must fit in i32 (kernel index arithmetic is 32bit int): \
                 m={m}, n={n}, k={k}, m*k={mk}, k*n={kn}, m*n={mn}"
            ),
        });
    }

    Ok(())
}

/// tiled カーネル専用の `k` 追加上限検証。
///
/// tiled カーネル（`kernels::TILED_F32`/`TILED_F16`）は各タイル反復で
/// `t * TILE + threadIdx.x`（`threadIdx.x` は最大 `TILE - 1`）を C の `int`
/// 算術で計算し `a_col`／`b_row` を得る。この値は `k` に近い最終タイルで
/// 最大 `k + TILE - 2` 程度に達しうるため、`k` が `i32::MAX - (TILE - 1)`
/// を超えると当該算術が i32 の範囲でオーバーフローしうる（実行前ガード。
/// `validate_gemm_dims` の i32 積ガードとは独立に、tiled 固有のタイル
/// インデックス算術を保護する）。`run_tiled_f32`／`run_tiled_f16` からのみ
/// 呼ばれ、naive 経路の契約（`validate_gemm_dims` のみ）は変更しない。
pub(crate) fn validate_tiled_k_bound(k: u32) -> Result<(), CudaError> {
    let limit = i32::MAX as u32 - (kernels::TILE - 1);
    if k > limit {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k must not exceed i32::MAX - (TILE - 1) for tiled kernel tile-index \
                 arithmetic: k={k}, limit={limit}, TILE={}",
                kernels::TILE
            ),
        });
    }
    Ok(())
}

impl CudaGemm {
    /// `device` 上で naive／tiled GEMM カーネル（f32/f16 各 2 種）を NVRTC
    /// コンパイルし保持するハンドルを構築する。
    ///
    /// 手順: `kernels::{NAIVE_F32,NAIVE_F16,TILED_F32,TILED_F16}` を
    /// `device.arch()` 向けに `nvrtc::compile_ptx` でコンパイル →
    /// `device.context().load_module()` →
    /// `load_function("gemm_naive_f32"/"gemm_naive_f16"/"gemm_tiled_f32"/
    /// "gemm_tiled_f16")`。カーネルコンパイル自体は `CudaDevice::new` と
    /// 同じく `libnvrtc` 不在時に `CudaError::NvrtcUnavailable` を返す
    /// （`compile_ptx` のプローブゲートを経由。panic しない）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let naive_f32_ptx = compile_ptx(kernels::NAIVE_F32, arch)?;
        let naive_f16_ptx = compile_ptx(kernels::NAIVE_F16, arch)?;
        let tiled_f32_ptx = compile_ptx(kernels::TILED_F32, arch)?;
        let tiled_f16_ptx = compile_ptx(kernels::TILED_F16, arch)?;

        let naive_f32 = device
            .context()
            .load_module(naive_f32_ptx)?
            .load_function("gemm_naive_f32")?;
        let naive_f16 = device
            .context()
            .load_module(naive_f16_ptx)?
            .load_function("gemm_naive_f16")?;
        let tiled_f32 = device
            .context()
            .load_module(tiled_f32_ptx)?
            .load_function("gemm_tiled_f32")?;
        let tiled_f16 = device
            .context()
            .load_module(tiled_f16_ptx)?
            .load_function("gemm_tiled_f16")?;

        Ok(Self {
            stream: device.stream().clone(),
            naive_f32,
            naive_f16,
            tiled_f32,
            tiled_f16,
        })
    }

    /// naive f32 GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// ホスト側形状検証（[`validate_gemm_dims`]）を先行させた後、
    /// 16x16 ブロック・`div_ceil` グリッドで [`Self::run_f32_kernel`] を呼ぶ
    /// （PoC-v2-3 `run_f32` を踏襲。計測用 `Duration` は返さない。
    /// モジュールコメント「PoC からの変更点」参照）。
    pub fn run_naive_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        self.run_f32_kernel(&self.naive_f32, a, b, m, n, k, NAIVE_BLOCK_DIM)
    }

    /// naive f16 GEMM を実行する。入出力は `half::f16`、GPU 内部アキュムレート
    /// は f32（`kernels::NAIVE_F16` 参照）。手順は [`Self::run_naive_f32`] と同一。
    pub fn run_naive_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        self.run_f16_kernel(&self.naive_f16, a, b, m, n, k, NAIVE_BLOCK_DIM)
    }

    /// tiled f32 GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// ホスト側形状検証は naive 版と同じ [`validate_gemm_dims`] に加え、
    /// tiled カーネル固有のタイルインデックス算術を保護する
    /// [`validate_tiled_k_bound`] を経由する（モジュールコメント
    /// 「PoC からの変更点」3 参照）。ブロック次元は
    /// [`TILED_BLOCK_DIM`]（`kernels::TILE` x `kernels::TILE`）で固定する。
    pub fn run_tiled_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        self.run_f32_kernel(&self.tiled_f32, a, b, m, n, k, TILED_BLOCK_DIM)
    }

    /// tiled f16 GEMM を実行する。手順は [`Self::run_tiled_f32`] と同一。
    pub fn run_tiled_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        self.run_f16_kernel(&self.tiled_f16, a, b, m, n, k, TILED_BLOCK_DIM)
    }

    /// f32 カーネル共通の起動手続き（naive/tiled 双方から呼ばれる）。
    ///
    /// 呼び出し元がホスト側形状検証（`validate_gemm_dims`／
    /// `validate_tiled_k_bound`）を終えている前提で、`clone_htod` で A・B
    /// を転送し `block_dim` に応じたグリッドでカーネルを起動、
    /// `synchronize` の後 `clone_dtoh` で C を回収する（PoC-v2-3 の
    /// `run_f32` と同じ構造）。
    #[allow(clippy::too_many_arguments)]
    fn run_f32_kernel(
        &self,
        func: &CudaFunction,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
        block_dim: (u32, u32, u32),
    ) -> Result<Vec<f32>, CudaError> {
        // Cursor Bugbot 指摘（PR #240）: `validate_gemm_dims` は
        // `backend-cpu::gemm_naive` と同様 m==0／n==0（a/c が空）を no-op
        // として許容するが、その形状のまま `launch_config` を呼ぶと
        // grid_dim の x（n 由来）または y（m 由来）が 0 になり、CUDA
        // ドライバは 0 次元の起動を拒否する。CPU 側の no-op 契約
        // （`backend-cpu/src/gemm.rs` の `n == 0` 早期 return コメント参照）
        // に揃え、カーネル起動自体を行わず空の結果を返す（naive/tiled 共通）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        // Cursor Bugbot 指摘（PR #244）: `k == 0` の場合 `a`（m*k 要素）・
        // `b`（k*n 要素）は共に空スライスになり、そのまま `clone_htod` を
        // 呼ぶと 0 バイトのデバイスバッファ確保を driver に要求する。一部
        // 環境の CUDA driver は 0 バイト確保を拒否するため（`cudarc` 経由の
        // `cuMemAlloc` は 0 バイトで `CUDA_ERROR_INVALID_VALUE` を返しうる）、
        // カーネル起動自体を回避し `m*n` 要素の全 0 ベクタを返す（K 方向の
        // 累積対象が存在しない = C は全 0 という GEMM の数学的定義どおりの
        // 契約。`tests/gemm_tiled.rs` の `tiled_f32_zero_k_returns_all_zero`
        // 参照）。
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?;

        let cfg = launch_config(m, n, block_dim);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/b.len()/
        // (m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の 5 個・型・
        // 個数がホスト側検証（validate_gemm_dims、tiled はさらに
        // validate_tiled_k_bound）済みの m/n/k と 1:1 対応し、カーネル内の
        // 手動境界チェック（naive: `if (row < m && col < n)`。tiled: タイル
        // ロード時の三項ガード＋書き込み時の同条件。kernels.rs 参照、
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。グリッド
        // 次元は `div_ceil` で m/n を包含するよう構築しており
        // （launch_config）、末尾ブロックの余剰スレッドはカーネル内境界
        // チェックで弾かれる。
        unsafe {
            self.stream
                .launch_builder(func)
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

    /// f16 カーネル共通の起動手続き。[`Self::run_f32_kernel`] と同一構造
    /// （naive/tiled 双方の `run_*_f16` から呼ばれる）。
    #[allow(clippy::too_many_arguments)]
    fn run_f16_kernel(
        &self,
        func: &CudaFunction,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
        block_dim: (u32, u32, u32),
    ) -> Result<Vec<f16>, CudaError> {
        // run_f32_kernel と同一の根拠（上記コメント参照）。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }

        // run_f32_kernel の k==0 早期 return と同一の根拠（上記コメント
        // 参照）。
        if k == 0 {
            return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
        }

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?;

        let cfg = launch_config(m, n, block_dim);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: run_f32_kernel と同一の根拠（上記コメント参照）。カーネル
        // 引数の型（__half*/int）・個数・デバイスバッファ長は
        // validate_gemm_dims（tiled はさらに validate_tiled_k_bound）で
        // 検証済みの m/n/k から導出しており、カーネル内手動境界チェック
        // （REQ-8）と合わせて OOB を防ぐ。
        unsafe {
            self.stream
                .launch_builder(func)
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

/// `block_dim` に対し `m`/`n` を切り上げ（`div_ceil`）で包含するグリッド
/// 次元を構築する。末尾ブロックが `m`/`n` を超える分はカーネル内の手動
/// 境界チェック（REQ-8）に委ねる契約（`kernels.rs` 参照）。
fn launch_config(m: u32, n: u32, block_dim: (u32, u32, u32)) -> LaunchConfig {
    let grid_dim = (n.div_ceil(block_dim.0), m.div_ceil(block_dim.1), 1);
    LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_gemm_dims_accepts_matching_lengths() {
        assert!(validate_gemm_dims(2 * 3, 3 * 4, 2, 4, 3).is_ok());
    }

    #[test]
    fn validate_gemm_dims_rejects_a_len_mismatch() {
        let err = validate_gemm_dims(5, 12, 2, 4, 3).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_b_len_mismatch() {
        let err = validate_gemm_dims(6, 11, 2, 4, 3).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_mk_overflow() {
        let err = validate_gemm_dims(0, 0, u32::MAX, 1, u32::MAX).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_i32_max_exceeding_dims() {
        // usize (64bit) では m*k はオーバーフローしないが、カーネル引数
        // が i32 のため i32::MAX 超過は別途拒否される必要がある。
        let m = (i32::MAX as u32) + 1;
        let a_len = m as usize;
        let err = validate_gemm_dims(a_len, 1, m, 1, 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_accepts_zero_m_or_n_as_noop_shape() {
        // m==0／n==0 は `backend-cpu::gemm_naive` と同じ no-op 形状として
        // 許容する（Cursor Bugbot 指摘 #240。カーネル起動自体は `run_naive_*`
        // 側の早期 return で回避するため、検証自体は拒否しない）。
        assert!(validate_gemm_dims(0, 3 * 4, 0, 4, 3).is_ok());
        assert!(validate_gemm_dims(4 * 3, 0, 4, 0, 3).is_ok());
    }

    #[test]
    fn validate_gemm_dims_rejects_mk_product_exceeding_i32_max() {
        // m*k は usize（64bit）に収まるが、カーネル側の `row * k + p` は
        // i32 算術のためインデックスがラップしうる（Cursor Bugbot 指摘
        // #240）。m/n/k 個々は i32::MAX 以下でも積が超過するケースを拒否する。
        //
        // Cursor Bugbot 指摘（PR #240 再指摘）: a_len/b_len を
        // validate_gemm_dims の長さチェック（mk/kn との一致検査）を
        // 通過する値にしないと、意図した i32 積ガードに到達する前に
        // 長さ不一致エラーで reject されてしまい、このガードを削除しても
        // テストが pass し続ける（意図した経路をロックできない）。
        // よって b_len は k*n（このケースは n=1 なので k）に正しく合わせる。
        let m: u32 = 1 << 16; // 65536
        let n: u32 = 1;
        let k: u32 = 1 << 16; // 65536 → m*k = 2^32 > i32::MAX
        let a_len = (m as usize) * (k as usize);
        let b_len = (k as usize) * (n as usize);
        let err = validate_gemm_dims(a_len, b_len, m, n, k).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_kn_product_exceeding_i32_max() {
        // 上記と同じ理由で a_len を m*k（このケースは m=1 なので k）に
        // 正しく合わせ、長さチェックを通過させたうえで k*n の積ガードへ
        // 到達させる。
        let m: u32 = 1;
        let k: u32 = 1 << 16;
        let n: u32 = 1 << 16;
        let a_len = (m as usize) * (k as usize);
        let b_len = (k as usize) * (n as usize);
        let err = validate_gemm_dims(a_len, b_len, m, n, k).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_gemm_dims_rejects_mn_product_exceeding_i32_max() {
        // 上記と同じ理由で b_len を k*n（このケースは k=1 なので n）に
        // 正しく合わせ、長さチェックを通過させたうえで m*n の積ガードへ
        // 到達させる。
        let m: u32 = 1 << 16;
        let n: u32 = 1 << 16;
        let k: u32 = 1;
        let a_len = (m as usize) * (k as usize);
        let b_len = (k as usize) * (n as usize);
        let err = validate_gemm_dims(a_len, b_len, m, n, k).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_tiled_k_bound_accepts_ordinary_k() {
        assert!(validate_tiled_k_bound(4096).is_ok());
        assert!(validate_tiled_k_bound(0).is_ok());
    }

    #[test]
    fn validate_tiled_k_bound_rejects_k_exceeding_limit() {
        let limit = i32::MAX as u32 - (kernels::TILE - 1);
        assert!(validate_tiled_k_bound(limit).is_ok());
        let err = validate_tiled_k_bound(limit + 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // 17x19 を 16x16 ブロックで覆うには grid (2, 2) が必要
        // （div_ceil(17,16)=2, div_ceil(19,16)=2）。
        let cfg = launch_config(17, 19, (16, 16, 1));
        assert_eq!(cfg.grid_dim, (2, 2, 1));
        assert_eq!(cfg.block_dim, (16, 16, 1));
    }
}
