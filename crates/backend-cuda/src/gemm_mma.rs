//! f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM の起動 API（TASK-11.1h・#187）。
//!
//! `CudaMmaGemm` は `kernels_mma::MMA_F16`（`m16n8k16` mma・3 ステージ
//! `cp.async` パイプライン）をコンパイル・保持し、以降はホスト側スライスを
//! 渡すだけで GPU 実行できる境界を担う（`gemm_wmma.rs::CudaWmmaGemm` と
//! 同じ責務分割。並行 issue #62/#63 が `gemm.rs`／`gemm_wmma.rs` を編集中の
//! ため、本イシューでは既存ファイルに触れず独立ファイルへ分離する）。
//!
//! ホスト側形状検証は `gemm.rs::validate_gemm_dims`（`pub(crate)`）を
//! そのまま再利用し、判定ロジックを複製しない。加えて本経路固有の
//! `cp.async` 16 バイト整列制約（[`validate_mma_alignment`]）を追加で
//! 検証する（`kernels_mma.rs` 冒頭ドキュメントコメント「整列制約」参照）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::kernels_mma;
use crate::nvrtc::compile_ptx;

/// `mma.sync`/`ldmatrix`/`cp.async` 経路が要求する compute capability の
/// 下限（major）。
///
/// `cp.async`・`ldmatrix` は LDGSTS（compute capability 8.0+）を要求する
/// （`nvidia-cuda` スキル `references/advanced/features/async-copies.md`
/// 「LDGSTS (CC 8.0+)」。`kernels_mma.rs` 冒頭コメント「命令選定・sm_80+
/// ゲート」）。WMMA 経路（cc>=7.0・`gemm_wmma.rs::MIN_COMPUTE_CAPABILITY_MAJOR`）
/// より厳しい下限であり、独立した定数として保持する。
const MIN_COMPUTE_CAPABILITY_MAJOR: i32 = 8;

/// `kernels_mma::MMA_BLOCK_THREADS` に 1:1 対応するブロック次元。
const MMA_BLOCK_DIM: (u32, u32, u32) = (kernels_mma::MMA_BLOCK_THREADS, 1, 1);

/// `cp.async.cg.shared.global` の 16 バイト転送粒度が要求するグローバル側
/// 整列制約を検証する（`kernels_mma.rs` 冒頭コメント「整列制約」参照）。
///
/// A の行ストライドは `k`、B の行ストライドは `n` であり、共有メモリ側の
/// タイル幅（`MMA_BK`/`MMA_BN`）が共に 8 の倍数であることと合わせて
/// `k % 8 == 0 && n % 8 == 0` を満たさない限り、行境界をまたぐ列オフセット
/// が 16 バイト境界からずれうる。`gemm.rs::validate_tiled_k_bound`
/// （tiled 経路の K 追加検証）と同種の「経路固有の追加検証」パターンで
/// あり、`validate_gemm_dims` の一般契約（`CudaError::InvalidShape`）を
/// 再利用する。
///
/// `pub(crate)`: `tests/gemm_mma.rs` から実機非依存の単体テストとして
/// 直接呼べるようにする（`validate_gemm_dims` と同じ公開範囲方針）。
pub(crate) fn validate_mma_alignment(n: u32, k: u32) -> Result<(), CudaError> {
    if !k.is_multiple_of(8) || !n.is_multiple_of(8) {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "mma.sync/cp.async path requires k % 8 == 0 && n % 8 == 0 \
                 (cp.async 16-byte transfer granularity; kernels_mma.rs \
                 doc comment \"整列制約\"), but got n={n}, k={k}"
            ),
        });
    }
    Ok(())
}

/// CUDA の grid 次元 y/z 成分の上限（65,535。全 compute capability 共通。
/// x 成分の上限は 2^31-1 と大きく実用的に問題にならないため x は検証
/// しない）。
const MAX_GRID_DIM_Y: u32 = 65_535;

/// `mma_launch_config` が構築するグリッドの y 成分（`m.div_ceil(MMA_BM)`）
/// が CUDA の上限（65,535）を超えないことを検証する（PR #255 レビュー
/// 指摘。超過するとホスト側の形状・整列検証はすべて通過した上で、
/// ドライバのカーネル起動が失敗する。`validate_mma_alignment` と同種の
/// 「経路固有の追加検証」パターン）。
///
/// `pub(crate)`: `tests/gemm_mma.rs` から実機非依存の単体テストとして
/// 直接呼べるようにする（`validate_mma_alignment` と同じ公開範囲方針）。
pub(crate) fn validate_mma_grid_bounds(m: u32) -> Result<(), CudaError> {
    let grid_y = m.div_ceil(kernels_mma::MMA_BM);
    if grid_y > MAX_GRID_DIM_Y {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "mma.sync path grid_dim.y (m.div_ceil(MMA_BM)={grid_y}) exceeds CUDA's \
                 {MAX_GRID_DIM_Y} limit for grid dimensions y/z (MMA_BM={}); m={m} is too large",
                kernels_mma::MMA_BM
            ),
        });
    }
    Ok(())
}

/// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM カーネルのコンパイル済み
/// ハンドルを保持する。
///
/// `stream` は `CudaDevice` から `Arc` クローンで受け取る（`gemm_wmma.rs`
/// と同じ共有契約）。
pub struct CudaMmaGemm {
    stream: Arc<CudaStream>,
    mma_f16: CudaFunction,
}

impl CudaMmaGemm {
    /// `device` 上で `mma.sync`/`ldmatrix`/`cp.async` GEMM カーネルを
    /// NVRTC コンパイルし保持するハンドルを構築する。
    ///
    /// 手順: (1) `device.compute_capability()` が
    /// [`MIN_COMPUTE_CAPABILITY_MAJOR`]（8.0）未満なら NVRTC コンパイルを
    /// 試みず `CudaError::TensorCoreUnsupported` を返す（`gemm_wmma.rs::new`
    /// と同じ判断。cc 判定をコンパイル前に行うことで、非対応デバイス上での
    /// 無駄な NVRTC 呼び出し・コンパイル失敗の紛れ込みを避ける）。(2)
    /// `kernels_mma::MMA_F16` を `device.arch()` 向けに `nvrtc::compile_ptx`
    /// でコンパイル。(3) `device.context().load_module()` →
    /// `load_function("gemm_mma_f16")`。`libnvrtc` 不在時は
    /// `CudaError::NvrtcUnavailable` を返す（`compile_ptx` のプローブゲート
    /// を経由。panic しない。本セッションの実行環境がまさにこの分岐
    /// ——CUDA driver はあるが NVRTC はない——であり、`tests/gemm_mma.rs`
    /// の環境適応テストで確認済み。`kernels_mma.rs` 冒頭「検証状態」参照）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        // カーネル定数の内部整合性（静的共有メモリの 48KiB 上限・`MMA_BK`
        // の `MMA_K` 整除性・`MMA_STAGES >= 2`）は `kernels_mma.rs` の
        // `const _: () = assert!(...)` でコンパイル時に検査済み（実機
        // コンパイルできない本セッションでも `cargo build` の時点で機械
        // 検出できる代替チェック。本ファイル冒頭コメント参照）。
        //
        // #492 でカーネルソース側の wait_group を「ループ内固定即値
        // （`"n"(STAGES - 2)`）＋ループ外 drain」構造へ整理したことにより、
        // ここでの追加確認はカーネルソース中のハードコード数字即値
        // （旧 `cp.async.wait_group 1;`）との対応検査ではなくなった
        // （ループ内 wait はもはや `MMA_STAGES` 由来の数字即値を持たず、
        // `"n"` 制約を通じてカーネル側が自身で `STAGES - 2` を計算する
        // ため）。下記は `MMA_WAIT_GROUP_IMMEDIATE` 定数自体の定義式
        // （`kernels_mma.rs` 側）との対応を明示するトートロジーだが、
        // `MMA_WAIT_GROUP_IMMEDIATE` を実利用する箇所として維持する
        // （dead-code 化を避ける。値が定数のため `debug_assert!` ではなく
        // `debug_assert_eq!` を使う。`assert!` 版は clippy
        // `assertions_on_constants` に抵触する）。段数非依存になった今でも
        // `MMA_K_STEPS_PER_STAGE` はカーネル内
        // `for (int kstep = 0; kstep < BK / MMA_K; ++kstep)` に対応する
        // Rust 側唯一の真実源であり続けるため、引き続き実利用しておく
        // （`kernels_mma.rs` 冒頭ドキュメントコメント参照）。
        debug_assert_eq!(
            kernels_mma::MMA_WAIT_GROUP_IMMEDIATE,
            kernels_mma::MMA_STAGES - 2
        );
        debug_assert_eq!(kernels_mma::MMA_K_STEPS_PER_STAGE, 2);

        let (major, minor) = device.compute_capability();
        if major < MIN_COMPUTE_CAPABILITY_MAJOR {
            return Err(CudaError::TensorCoreUnsupported {
                detail: format!(
                    "mma.sync/ldmatrix/cp.async path requires compute capability \
                     >= {MIN_COMPUTE_CAPABILITY_MAJOR}.0 (cp.async/ldmatrix are \
                     LDGSTS-only, sm_80+), but device reports {major}.{minor}"
                ),
            });
        }

        let arch = device.arch();
        let ptx = compile_ptx(kernels_mma::MMA_F16, arch)?;
        let mma_f16 = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_mma_f16")?;

        Ok(Self {
            stream: device.stream().clone(),
            mma_f16,
        })
    }

    /// f16 `mma.sync`/`ldmatrix`/`cp.async` GEMM を実行する。C = A @ B
    /// （`m x k` @ `k x n`）。入出力は `half::f16`、GPU 内部アキュムレートは
    /// f32（`kernels_mma::MMA_F16` 参照。数値契約は `CudaWmmaGemm::run_f16`
    /// と同一）。
    ///
    /// ホスト側形状検証を 3 段で行う: `validate_gemm_dims`（naive/tiled/WMMA
    /// と共通の一般契約。スライス長の整合性のみを見るため no-op 形状でも
    /// 常に先行させる）→ no-op 形状（`m==0 || n==0 || k==0`）の早期
    /// return → [`validate_mma_alignment`]／[`validate_mma_grid_bounds`]
    /// （本経路固有の `cp.async` 16 バイト整列制約・grid_dim.y 上限）。
    ///
    /// 整列検証・grid 上限検証を no-op 判定より後に置く（PR #255 レビュー
    /// 指摘）: 例えば `(m,n,k)=(8,7,0)` のような有効な no-op 形状は
    /// `n=7` が 8 の倍数でないため、整列検証を先に行うと実際には
    /// カーネルを起動しない形状まで誤って `CudaError::InvalidShape` で
    /// 拒否してしまう。整列・grid 上限はいずれもカーネル起動時にのみ
    /// 意味を持つ制約であるため、起動しないと決まった時点（no-op 判定）
    /// より後で検証すれば十分。
    ///
    /// グリッド次元は `kernels_mma::MMA_BM`/`MMA_BN` 単位の `div_ceil` で
    /// 構築し、末尾タイルの余剰はカーネル内 REQ-8 境界チェック
    /// （`kernels_mma.rs` 参照）に委ねる。
    pub fn run_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;

        // m==0/n==0（0 次元 grid はドライバが拒否する。`gemm.rs::run_f32_kernel`
        // ・`gemm_wmma.rs::run_f16` と同じ根拠）は起動自体を回避する。
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        // k==0 は A/B が空スライスになるため起動を回避し C = 全 0 を返す
        // （`gemm_wmma.rs::run_f16` の k==0 早期 return と同一契約）。
        if k == 0 {
            return Ok(vec![f16::ZERO; (m as usize) * (n as usize)]);
        }

        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;

        let (a_dev, b_dev) = self.upload_f16(a, b)?;
        let mut c_dev = self.alloc_output_f16(m, n)?;
        self.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)?;
        self.download_f16(&c_dev)
    }

    /// A・B をホスト→デバイスへ転送する（`run_f16` の H2D 部分の切り出し。
    /// ベンチマークが GPU 実行時間のみを計測できるよう、転送とカーネル
    /// 実行を分離する。PR #255 レビュー指摘 —— `examples/gemm_mma_bench.rs`
    /// が転送・バッファ確保込みで TFLOPS を算出していた問題への対処）。
    pub fn upload_f16(
        &self,
        a: &[f16],
        b: &[f16],
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>), CudaError> {
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        Ok((a_dev, b_dev))
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（`run_f16` のバッファ
    /// 確保部分の切り出し。[`upload_f16`] と同じ理由でベンチマークから
    /// 再利用できるよう公開する）。
    pub fn alloc_output_f16(&self, m: u32, n: u32) -> Result<CudaSlice<f16>, CudaError> {
        Ok(self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?)
    }

    /// デバイス常駐済みの A/B/C バッファに対してカーネルを起動し、完了を
    /// 待つ（H2D/D2H を含まない「GPU 実行のみ」の区間。[`upload_f16`]・
    /// [`alloc_output_f16`] と組み合わせてベンチマークの計測対象を絞る
    /// ために公開する）。
    ///
    /// safe な公開 API であるため、呼び出し元（`run_f16` あるいは
    /// ベンチマーク）の事前検証に依存せず、本関数自身が `run_f16` と同じ
    /// 形状検証（`validate_gemm_dims`・[`validate_mma_alignment`]・
    /// [`validate_mma_grid_bounds`]）およびデバイスバッファ長検証
    /// （`a_dev`/`b_dev`/`c_dev`）を行う（PR #349 codex-review 指摘 P0。
    /// `gemm.rs::launch_tiled_f32` のドキュメンテーションコメント参照。
    /// `gemm_wmma.rs::launch_f16` には同種の指摘があったが本関数は指摘に
    /// 明示されていなかった — 同一パターンの脆弱性のため一貫して修正する）。
    pub fn launch_f16(
        &self,
        a_dev: &CudaSlice<f16>,
        b_dev: &CudaSlice<f16>,
        c_dev: &mut CudaSlice<f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;

        let cfg = mma_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/
        // b.len()/(m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の
        // 5 個・型・個数が、上記で検証済みの m/n/k と 1:1 対応する。
        // カーネル内の手動境界チェック
        // （cp.async src-size ゼロ充填・エピローグ guarded store。
        // kernels_mma.rs 参照、REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。グリッド次元は MMA_BM/MMA_BN 単位の div_ceil で
        // m/n を包含するよう構築しており（mma_launch_config）、末尾タイル
        // の余剰はカーネル内境界チェックで弾かれる。共有メモリは静的
        // `__shared__` 配列のみを使用するため `shared_mem_bytes` は 0 の
        // ままでよい（`kernels_mma.rs` 冒頭コメント「タイル構成」の
        // 18432B は per-block 静的上限 48KiB 内であり動的共有メモリの
        // 追加確保・`cudaFuncSetAttribute` opt-in は不要）。
        unsafe {
            self.stream
                .launch_builder(&self.mma_f16)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// C をデバイス→ホストへ転送する（`run_f16` の D2H 部分の切り出し。
    /// [`upload_f16`] と同じ理由で公開する）。
    pub fn download_f16(&self, c_dev: &CudaSlice<f16>) -> Result<Vec<f16>, CudaError> {
        Ok(self.stream.clone_dtoh(c_dev)?)
    }
}

/// mma カーネル専用のグリッド次元計算。1 ブロック = C の `MMA_BM x MMA_BN`
/// タイル 1 個を担当するため、`div_ceil(n, MMA_BN)` x `div_ceil(m, MMA_BM)`
/// のグリッドを構築する（`gemm_wmma.rs::wmma_launch_config` と同じ設計。
/// `gemm.rs::launch_config` の「1 スレッド = C の 1 要素」前提とは異なる
/// ため独立関数とする）。
fn mma_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_mma::MMA_BN),
        m.div_ceil(kernels_mma::MMA_BM),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: MMA_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mma_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // MMA_BM=32, MMA_BN=64: 33x65 を覆うには grid (2, 2) が必要
        // （div_ceil(65,64)=2, div_ceil(33,32)=2）。
        let cfg = mma_launch_config(33, 65);
        assert_eq!(cfg.grid_dim, (2, 2, 1));
        assert_eq!(cfg.block_dim, MMA_BLOCK_DIM);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn mma_launch_config_exact_multiple_shape_has_no_extra_tile() {
        let cfg = mma_launch_config(64, 128);
        assert_eq!(cfg.grid_dim, (2, 2, 1));
    }

    #[test]
    fn validate_mma_alignment_accepts_multiples_of_eight() {
        assert!(validate_mma_alignment(64, 32).is_ok());
        assert!(validate_mma_alignment(8, 8).is_ok());
    }

    #[test]
    fn validate_mma_alignment_rejects_non_multiple_n() {
        let err = validate_mma_alignment(9, 32).expect_err("n=9 is not a multiple of 8");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_alignment_rejects_non_multiple_k() {
        let err = validate_mma_alignment(64, 17).expect_err("k=17 is not a multiple of 8");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_grid_bounds_accepts_shapes_within_limit() {
        // MMA_BM=32: div_ceil(65_535 * 32, 32) = 65_535（上限ちょうど）。
        assert!(validate_mma_grid_bounds(65_535 * kernels_mma::MMA_BM).is_ok());
    }

    #[test]
    fn validate_mma_grid_bounds_rejects_m_exceeding_grid_y_limit() {
        // MMA_BM=32: div_ceil(65_535*32 + 1, 32) = 65_536 > 65_535。
        let err = validate_mma_grid_bounds(65_535 * kernels_mma::MMA_BM + 1)
            .expect_err("grid_dim.y must exceed CUDA's 65,535 limit");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    /// `validate_mma_alignment` 単体では k=0/n=7 は整列制約違反として
    /// 拒否される（この関数自体は no-op 形状かどうかを考慮しない）。
    /// `run_f16` が no-op 早期 return をこの検証より前に行うことで
    /// `(m,n,k)=(8,7,0)` のような有効な no-op 形状を誤って拒否しない
    /// 契約は、実機依存の統合テスト
    /// `tests/gemm_mma.rs::mma_f16_accepts_noop_shape_with_misaligned_n_when_k_is_zero`
    /// （`#[ignore]`）で確認する（PR #255 レビュー指摘）。
    #[test]
    fn validate_mma_alignment_rejects_misaligned_n_independent_of_noop_shape() {
        assert!(validate_mma_alignment(7, 0).is_err());
    }

    /// `#define STAGES <kernels_mma::MMA_STAGES>` 行のみを `stages` へ
    /// 書き換えたソースを NVRTC コンパイル・実行する（#492 §5-5 の
    /// 実機必須テスト専用ヘルパー。`kernels_mma.rs::tests::
    /// mma_f16_source_with_stages` と同じ置換方針だが、こちらは
    /// NVRTC コンパイル・カーネル起動まで踏み込む点が異なる）。
    ///
    /// `CudaMmaGemm::new`/`run_f16` を再利用しない理由: それらは常に
    /// `kernels_mma::MMA_F16`（`STAGES=3` 固定の文字列）をコンパイルする
    /// ため、段数を差し替えた変種を実行するには本関数のように NVRTC
    /// コンパイル・モジュールロード・起動を直接組み立てる必要がある。
    /// 形状検証（`validate_gemm_dims`・[`validate_mma_alignment`]・
    /// [`validate_mma_grid_bounds`]）・グリッド構築（[`mma_launch_config`]）・
    /// SAFETY 根拠は `launch_f16` と同一（段数はブロックタイル形状・
    /// grid/block 次元に影響しないため）。
    fn run_f16_with_stages(
        device: &CudaDevice,
        stages: u32,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        let from = format!("#define STAGES {}\n", kernels_mma::MMA_STAGES);
        let to = format!("#define STAGES {stages}\n");
        assert_eq!(
            kernels_mma::MMA_F16.matches(&from).count(),
            1,
            "kernels_mma::MMA_F16 中の `{from:?}` の出現数が 1 ではありません \
             （run_f16_with_stages の前提が崩れています）"
        );
        let src = kernels_mma::MMA_F16.replacen(&from, &to, 1);

        let ptx = compile_ptx(&src, device.arch())
            .expect("stage-swapped MMA_F16 source must compile via NVRTC on real hardware");
        let func = device
            .context()
            .load_module(ptx)
            .expect("stage-swapped module load must succeed")
            .load_function("gemm_mma_f16")
            .expect("gemm_mma_f16 must be present in the stage-swapped module");

        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_mma_alignment(n, k)?;
        validate_mma_grid_bounds(m)?;

        let stream = device.stream();
        let a_dev = stream.clone_htod(a)?;
        let b_dev = stream.clone_htod(b)?;
        let mut c_dev = stream.alloc_zeros::<f16>((m as usize) * (n as usize))?;

        let cfg = mma_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: `launch_f16` と同一の引数構成（a_dev/b_dev/c_dev + m/n/k）
        // であり、段数の違いはカーネル内部の共有メモリ・パイプライン深さ
        // のみに影響し、引数の型・個数・対応関係は変わらない。REQ-8 の
        // 手動境界チェックも段数非依存（本ファイル冒頭コメント・
        // `kernels_mma.rs` 参照）。
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        stream.synchronize()?;
        Ok(stream.clone_dtoh(&c_dev)?)
    }

    /// #492 受け入れ基準（段数可変化）の実機検証: `stages ∈ {2, 3, 4}` の
    /// 各カーネル変種が、小形状・タイル端形状・大形状の全てで**ビット
    /// 一致**の出力を返すことを確認する。段数（パイプライン深さ）は
    /// cp.async の同期タイミングのみを変え、mma/ldmatrix の実行順序・
    /// アキュムレート順序は変えないため（本ファイル
    /// `kernels_mma::MMA_WAIT_GROUP_IMMEDIATE` ドキュメンテーションコメント
    /// 「正しさ」参照）、tolerance を使わない bit 等値で主張できる
    /// （`.claude/rules/coding-rust.md` の「バックエンド間数値一致テストの
    /// 許容誤差を単独で緩和しない」契約に抵触しない。段数間比較は
    /// バックエンド間比較ではなく同一バックエンド内の実装詳細比較の
    /// ため、tolerance の対象外）。
    ///
    /// `#[ignore]`: 本セッション（本ファイル冒頭コメント「検証状態」）は
    /// NVRTC 非搭載のため実行できない。DGX Spark GB10 等の実機で
    /// `cargo test -p backend-cuda --lib -- --ignored` から実行する
    /// （`gemm.rs::tests::wmma_tf32_basic_kernel_parity_does_not_regress`
    /// と同じ実行方法）。
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
    fn mma_f16_stage_count_does_not_change_bit_exact_output() {
        let device =
            CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

        // (m, n, k): 小形状（16x8x16。単一 mma タイル）・タイル端形状
        // （40x72x160。MMA_BM=32/MMA_BN=64/MMA_BK=32 の非整数倍）・
        // 大形状（256x256x4096。B-0/#491 parity 非後退契約の mma_f16 行と
        // 同一形状。docs/perf/cuda-parity-baseline.md 参照）を横断する
        // （#492 実装計画 §5-5）。
        let shapes: [(u32, u32, u32); 3] = [(16, 8, 16), (40, 72, 160), (256, 256, 4096)];
        let seed: u64 = 9999;

        for &(m, n, k) in &shapes {
            let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
            let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
            let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

            let mut outputs: Vec<(u32, Vec<f16>)> = Vec::new();
            for stages in [2u32, 3u32, 4u32] {
                let c =
                    run_f16_with_stages(&device, stages, &a, &b, m, n, k).unwrap_or_else(|err| {
                        panic!(
                            "stages={stages} run_f16_with_stages failed for shape \
                             (m={m}, n={n}, k={k}): {err}"
                        )
                    });
                outputs.push((stages, c));
            }

            let (base_stages, base_c) = &outputs[0];
            for (stages, c) in &outputs[1..] {
                assert_eq!(
                    c, base_c,
                    "shape (m={m}, n={n}, k={k}): stages={stages} の出力が \
                     stages={base_stages} と bit 一致しません（段数変更が \
                     mma/ldmatrix の実行順序に影響していないか確認すること）"
                );
            }
        }
    }
}
