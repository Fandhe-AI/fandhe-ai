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

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::kernels_wmma;
use crate::kernels_wmma_opt;
use crate::nvrtc::compile_ptx;

/// WMMA f16 経路が要求する compute capability の下限（major）。
///
/// 設計メモ（`docs/cuda-tensor-core-design.md` 7 節）が記す「WMMA f16 経路は
/// compute capability 7.0 以降で有効化可能」を実装した契約を、
/// `fandhe_ai_tensor_core::dispatch::CUDA_WMMA_MIN_CC`（ディスパッチ規則側の正本
/// 定数。`docs/dispatch-rules-design.md` §3.2「閾値定数は 1 箇所に集約
/// する」）の major 値からそのまま導出する。#68 レビュー指摘: 本クレート
/// 側で独立した定数を持つと #69（TASK-11.2c・閾値実測再確定）で片方のみ
/// 更新した際に判定不整合が生じうるため、値を複製せず参照する形へ変更
/// した。`CudaWmmaGemm::new` はこの下限を NVRTC コンパイル前に検査し、
/// 満たさない場合は `CudaError::TensorCoreUnsupported` を返す
/// （フォールバック判断自体は TASK-11.2／#66 のディスパッチ規則側の
/// 責務）。
const MIN_COMPUTE_CAPABILITY_MAJOR: i32 = fandhe_ai_tensor_core::dispatch::CUDA_WMMA_MIN_CC.0;

/// `kernels_wmma::WMMA_TILE` に 1:1 対応するブロック次元。
///
/// 1 ブロック = 1 warp（32 スレッド）= C の `WMMA_TILE x WMMA_TILE`
/// タイル 1 個を計算する構成（`kernels_wmma.rs` 冒頭ドキュメントコメント
/// 「タイル構成」参照）。カーネル内の `lane`（`threadIdx.x`、0..31）を
/// 前提にしたガードロード／ストアと 1:1 対応するため、ここを変更する
/// 場合はカーネルソース側のスレッド分担ロジックも合わせて見直す必要がある。
const WMMA_BLOCK_DIM: (u32, u32, u32) = (32, 1, 1);

/// f16 WMMA opt（共有メモリ・タイル最適化版。TASK-11.1d・#63）カーネル
/// 起動 1 回あたりのブロック次元（128 スレッド = 4 warp、
/// `kernels_wmma_opt::WMMA_F16_OPT_THREADS` を 1 次元ブロックとして
/// 起動する。`WMMA_BLOCK_DIM`（1 ブロック = 1 warp）とは独立した契約。
/// `gemm.rs::WMMA_TF32_OPT_BLOCK_DIM` と同じ理由で専用定数として分離する）。
const WMMA_OPT_BLOCK_DIM: (u32, u32, u32) = (kernels_wmma_opt::WMMA_F16_OPT_THREADS, 1, 1);

/// f16 WMMA GEMM カーネルのコンパイル済みハンドルを保持する。
///
/// `stream` は `CudaDevice` から `Arc` クローンで受け取る（`gemm.rs::CudaGemm`
/// と同じ共有契約。`device.rs` 参照）。
pub struct CudaWmmaGemm {
    stream: Arc<CudaStream>,
    wmma_f16: CudaFunction,
    /// TASK-11.1d（#63）で追加。共有メモリ・タイル最適化版 f16 WMMA
    /// カーネル（`kernels_wmma_opt::wmma_f16_opt_source()`）のコンパイル済みハンドル。
    /// `gemm.rs::CudaGemm::wmma_tf32_opt` と同じ理由（コンパイル失敗しうる
    /// 環境が `wmma_f16`〈基本版〉より広い）で `Option` にし、失敗を
    /// `new` の早期 return に合流させない。`run_f16` はこちらが `Some` なら
    /// 優先的に使い、`None` なら `wmma_f16`（基本版）へ自動フォールバック
    /// する。
    wmma_f16_opt: Option<CudaFunction>,
    /// TASK-11.1e（#64）で追加。`wmma_f16_opt` のコンパイル・ロードが
    /// 失敗した場合の理由を退避する（`gemm.rs::CudaGemm::wmma_tf32_opt_error`
    /// と同じ設計判断。PR #256 レビュー指摘「opt 経路の可用性を断定せず
    /// 計測すると基本版へのサイレントフォールバックで green になりうる」
    /// を f16 側にも適用し、実機実測テスト
    /// （`tests/tensor_core_real_device.rs`）が「どのカーネル変種を
    /// 測ったか」を記録できるようにする）。opt カーネルが利用可能な場合は
    /// `None`。
    wmma_f16_opt_error: Option<String>,
}

impl CudaWmmaGemm {
    /// `device` 上で f16 WMMA GEMM カーネルを NVRTC コンパイルし保持する
    /// ハンドルを構築する。
    ///
    /// 手順: (1) `device.compute_capability()` が
    /// `MIN_COMPUTE_CAPABILITY_MAJOR` 未満なら NVRTC コンパイルを試みず
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

        // TASK-11.1d（#63）: `kernels_wmma_opt::wmma_f16_opt_source()` の不変条件
        // （ブロックタイルが warp タイルの倍数・スレッド数が warp 数×32 に
        // 一致・warp タイルが fragment 辺の 2 倍・パディング行幅が half の
        // `ldm` 制約〈8 の倍数〉を満たす）をコンパイル時 const アサーション
        // で機械検査する（`gemm.rs::CudaGemm::new` の TF32 opt 側と同じ方針。
        // レビュー指摘 #62 の踏襲）。
        const _: () = assert!(
            kernels_wmma_opt::WMMA_F16_OPT_BLOCK_M
                .is_multiple_of(kernels_wmma_opt::WMMA_F16_OPT_WARP_TILE)
        );
        const _: () = assert!(
            kernels_wmma_opt::WMMA_F16_OPT_BLOCK_N
                .is_multiple_of(kernels_wmma_opt::WMMA_F16_OPT_WARP_TILE)
        );
        const _: () = assert!(
            (kernels_wmma_opt::WMMA_F16_OPT_BLOCK_M / kernels_wmma_opt::WMMA_F16_OPT_WARP_TILE)
                * (kernels_wmma_opt::WMMA_F16_OPT_BLOCK_N
                    / kernels_wmma_opt::WMMA_F16_OPT_WARP_TILE)
                * 32
                == kernels_wmma_opt::WMMA_F16_OPT_THREADS
        );
        const _: () = assert!(
            kernels_wmma_opt::WMMA_F16_OPT_WARP_TILE == kernels_wmma_opt::WMMA_F16_OPT_FRAG * 2
        );
        const _: () = assert!(kernels_wmma_opt::WMMA_F16_OPT_A_PAD.is_multiple_of(8));
        const _: () = assert!(kernels_wmma_opt::WMMA_F16_OPT_B_PAD.is_multiple_of(8));

        // opt カーネルは基本版と独立にコンパイルし、失敗を `new` の早期
        // return に合流させない（`Self::wmma_f16_opt` フィールドの
        // ドキュメンテーションコメント参照）。失敗理由は
        // `wmma_f16_opt_error` へ退避し、`wmma_f16_opt_unavailable_reason`
        // 経由でテストから参照できるようにする（`gemm.rs::CudaGemm::new`
        // の TF32 opt 側と同じ分岐構造）。
        let (wmma_f16_opt, wmma_f16_opt_error) = match compile_wmma_f16_opt(device, arch) {
            Ok(func) => (Some(func), None),
            Err(err) => (None, Some(err.to_string())),
        };

        Ok(Self {
            stream: device.stream().clone(),
            wmma_f16,
            wmma_f16_opt,
            wmma_f16_opt_error,
        })
    }

    /// 共有メモリ・タイル最適化版 f16 WMMA カーネル（`Self::wmma_f16_opt`）
    /// が `new` 時点でコンパイル・ロードに成功しているかを返す（TASK-11.1e・
    /// #64。`gemm.rs::CudaGemm::wmma_tf32_opt_available` の f16 側鏡写し）。
    ///
    /// `run_f16` は opt カーネルが `None` の場合に基本版（`Self::wmma_f16`）
    /// へ自動フォールバックするため、`run_f16` の戻り値の成否だけでは opt
    /// カーネルが実際に実行されたかを判定できない。実機実測テスト
    /// （`tests/tensor_core_real_device.rs`）はこの関数で事前に可用性を
    /// 確認し、フォールバックが起きていないことを保証したうえで計測する。
    pub fn wmma_f16_opt_available(&self) -> bool {
        self.wmma_f16_opt.is_some()
    }

    /// [`Self::wmma_f16_opt_available`] が `false` の場合の失敗理由
    /// （`Self::wmma_f16_opt_error` の公開読み取り口）。opt カーネルが
    /// 利用可能な場合は `None` を返す。テストが「opt カーネルが使用不能
    /// だった具体的な理由」をパニックメッセージへ含められるようにする。
    pub fn wmma_f16_opt_unavailable_reason(&self) -> Option<&str> {
        self.wmma_f16_opt_error.as_deref()
    }

    /// f16 WMMA GEMM を実行する。C = A @ B（`m x k` @ `k x n`）。入出力は
    /// `half::f16`、GPU 内部アキュムレートは f32
    /// （`kernels_wmma::WMMA_F16` 参照。数値契約は `CudaGemm::run_naive_f16`
    /// と同一。`kernels_wmma.rs` 冒頭ドキュメントコメント「数値契約」参照）。
    ///
    /// ホスト側形状検証（`validate_gemm_dims`）を naive／tiled 経路と
    /// 共有する（`gemm.rs` 参照）。グリッド次元は `kernels_wmma::WMMA_TILE`
    /// 単位の `div_ceil` で構築し、末尾タイルの余剰はカーネル内 REQ-8
    /// 境界チェック（`kernels_wmma.rs` 参照）に委ねる。
    ///
    /// `gemm.rs::launch_config` は 1 スレッド = C の 1 要素という naive／
    /// tiled の起動モデル前提であり、1 ブロック = 1 warp = `WMMA_TILE x
    /// WMMA_TILE` 要素という WMMA の起動モデルとは異なるため再利用しない
    /// （本関数専用のグリッド計算を用いる）。
    ///
    /// **TASK-11.1d（#63）フォールバック方針**: 共有メモリ・タイル最適化版
    /// （`Self::wmma_f16_opt`）が `new` 時点でコンパイル・ロードに成功
    /// していれば、そちらを優先的に使用する。`None` の場合は基本版
    /// （`Self::wmma_f16`）へ自動フォールバックし、公開シグネチャ・
    /// 呼び出し側の挙動は変えない（`gemm.rs::CudaGemm::run_wmma_tf32` と
    /// 同じ設計判断）。
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

        match self.wmma_f16_opt.as_ref() {
            Some(func) => {
                let cfg = wmma_opt_launch_config(m, n);
                self.launch_f16_kernel(func, a, b, m, n, k, cfg)
            }
            None => {
                let cfg = wmma_launch_config(m, n);
                self.launch_f16_kernel(&self.wmma_f16, a, b, m, n, k, cfg)
            }
        }
    }

    /// f16 WMMA カーネル共通の転送・起動・同期・回収手続き（基本版・opt
    /// 版どちらの `CudaFunction`／`LaunchConfig` からも呼ばれる。
    /// [`Self::run_f16`] が m==0/n==0/k==0 の早期 return を終えた後にのみ
    /// 呼ぶ契約）。
    #[allow(clippy::too_many_arguments)]
    fn launch_f16_kernel(
        &self,
        func: &CudaFunction,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
        cfg: LaunchConfig,
    ) -> Result<Vec<f16>, CudaError> {
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let mut c_dev = self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?;

        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/
        // b.len()/(m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の
        // 5 個・型・個数が、ホスト側検証（validate_gemm_dims）済みの
        // m/n/k と 1:1 対応する。カーネル内の手動境界チェック（A/B タイル
        // guarded load・エピローグ guarded store。基本版は
        // kernels_wmma.rs、opt 版は kernels_wmma_opt.rs 参照、REQ-8）と
        // 合わせて OOB 読み書きが起きない根拠とする。グリッド次元は
        // 呼び出し元（`run_f16`）が基本版／opt 版それぞれのタイル単位で
        // `div_ceil` 構築済み（`wmma_launch_config`／`wmma_opt_launch_config`）
        // であり、末尾タイルの余剰はカーネル内境界チェックで弾かれる。
        // 共有メモリは静的 `__shared__` 配列のみを使用するため
        // `shared_mem_bytes` は 0 のままでよい。
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
        // 同期点は readback ヘルパーへ集約（#1013）。
        let c_host = crate::memory::readback(&self.stream, &c_dev)?;
        Ok(c_host)
    }

    /// A・B（f16）をホスト→デバイスへ転送する（`run_f16` の H2D 部分の
    /// 切り出し。`gemm_mma.rs::CudaMmaGemm::upload_f16` ・
    /// `gemm.rs::CudaGemm::upload_f32` と同じ理由でベンチマークが転送と
    /// カーネル実行を分離できるよう公開する。PR #349 codex-review 指摘
    /// P1 対応。`gemm.rs::upload_f32` ドキュメンテーションコメント
    /// 「PyTorch 参照計測」参照）。
    pub fn upload_f16(
        &self,
        a: &[f16],
        b: &[f16],
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>), CudaError> {
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        Ok((a_dev, b_dev))
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（[`Self::upload_f16`]
    /// と同じ理由で公開する）。
    pub fn alloc_output_f16(&self, m: u32, n: u32) -> Result<CudaSlice<f16>, CudaError> {
        Ok(self
            .stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))?)
    }

    /// デバイス常駐済みの A/B/C バッファに対して f16 WMMA カーネルを起動
    /// し、完了を待つ（H2D/D2H を含まない「GPU 実行のみ」の区間。
    /// [`Self::upload_f16`]・[`Self::alloc_output_f16`] と組み合わせて
    /// ベンチマークの計測対象を絞るために公開する）。opt カーネルが
    /// 利用可能ならそちらを使用し、そうでなければ基本版へフォールバック
    /// する（`run_f16` と同一の選択ロジック `self.wmma_f16_opt.is_some()`
    /// を用いる）。
    ///
    /// safe な公開 API であるため、呼び出し元の事前検証（`run_f16` と同じ
    /// `validate_gemm_dims`）に依存せず、本関数自身がホスト側形状検証
    /// および `a_dev`/`b_dev`/`c_dev` のデバイスバッファ長検証を行う
    /// （PR #349 codex-review 指摘 P0。`gemm.rs::launch_tiled_f32` の
    /// ドキュメンテーションコメント参照。`run_f16` の m==0/n==0/k==0
    /// 早期 return はここでは適用しない — 呼び出し元がゼロ次元を渡した
    /// 場合は 0 要素グリッド起動を CUDA ドライバが拒否する形でエラーに
    /// なるが、`validate_gemm_dims` は m/n/k=0 を有効な形状として許容する
    /// ため、この経路の安全性はカーネル起動自体の失敗に委ねられる）。
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
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;

        let (func, cfg) = match self.wmma_f16_opt.as_ref() {
            Some(func) => (func, wmma_opt_launch_config(m, n)),
            None => (&self.wmma_f16, wmma_launch_config(m, n)),
        };
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: launch_f16_kernel と同一の根拠。カーネル引数は上記で
        // 検証済みの m/n/k と 1:1 対応し、カーネル内の手動境界チェック
        // （基本版は kernels_wmma.rs、opt 版は kernels_wmma_opt.rs 参照、
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点へ
        // 委ねる。
        Ok(())
    }

    /// [`Self::launch_f16`] の「基本版（`Self::wmma_f16`）を強制的に使う」
    /// 変種。opt カーネルの可用性に関わらず常に基本版を起動する点のみが
    /// [`Self::launch_f16`] と異なり、検証・引数構成・SAFETY 根拠は共有
    /// する（イシュー #1123: `wmma_f16_opt` の性能外れ値切り分けで、
    /// opt 版と基本版のカーネル単体 TFLOPS を同一計測プロトコルで比較
    /// するための診断専用入口。本番ディスパッチ〈`run_f16`／`launch_f16`
    /// の opt 優先フォールバック〉には影響しない）。
    pub fn launch_f16_basic(
        &self,
        a_dev: &CudaSlice<f16>,
        b_dev: &CudaSlice<f16>,
        c_dev: &mut CudaSlice<f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;

        let cfg = wmma_launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: launch_f16 と同一の根拠（基本版カーネル
        // `Self::wmma_f16` を固定で使う点のみが異なる）。
        unsafe {
            self.stream
                .launch_builder(&self.wmma_f16)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(cfg)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点へ
        // 委ねる。
        Ok(())
    }

    /// C（f16）をデバイス→ホストへ転送する（[`Self::upload_f16`] と同じ
    /// 理由で公開する）。
    ///
    /// 同期点（#1013）: 常駐 `launch_f16` は非同期投入のみで完了を待たない
    /// ため、本関数が readback ヘルパー経由で完了を確定する。
    pub fn download_f16(&self, c_dev: &CudaSlice<f16>) -> Result<Vec<f16>, CudaError> {
        crate::memory::readback(&self.stream, c_dev)
    }

    /// ストリームの完了を明示的に待つ（イシュー #1013。
    /// `gemm.rs::CudaGemm::synchronize` と同じ理由の公開 API）。
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.stream.synchronize()?)
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

/// f16 WMMA opt カーネル（`kernels_wmma_opt::wmma_f16_opt_source()`）を単独で
/// コンパイル・ロードする。`CudaWmmaGemm::new` から呼ばれ、戻り値の `Err`
/// は基本版（`wmma_f16`）の可用性から切り離すため `?` で早期 return せず
/// 呼び出し元で `.ok()` により握りつぶす（`Self::wmma_f16_opt` フィールドの
/// ドキュメンテーションコメント参照。`gemm.rs::compile_wmma_tf32_opt` と
/// 同じ設計判断）。
fn compile_wmma_f16_opt(device: &CudaDevice, arch: &str) -> Result<CudaFunction, CudaError> {
    let ptx = compile_ptx(kernels_wmma_opt::wmma_f16_opt_source(), arch)?;
    let func = device
        .context()
        .load_module(ptx)?
        .load_function("gemm_wmma_f16_opt")?;
    Ok(func)
}

/// [`wmma_launch_config`] の opt 版。ブロックタイル
/// `kernels_wmma_opt::WMMA_F16_OPT_BLOCK_M/N`（64×64）を単位に `div_ceil`
/// でグリッドを構築する。末尾ブロックの余剰は opt カーネル内の手動境界
/// チェック（REQ-8）に委ねる契約は基本版と共通。
fn wmma_opt_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_wmma_opt::WMMA_F16_OPT_BLOCK_N),
        m.div_ceil(kernels_wmma_opt::WMMA_F16_OPT_BLOCK_M),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: WMMA_OPT_BLOCK_DIM,
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

    #[test]
    fn wmma_opt_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // 65x63 を 64x64 ブロックタイルで覆うには grid (1, 2) が必要
        // （div_ceil(63,64)=1, div_ceil(65,64)=2）。
        let cfg = wmma_opt_launch_config(65, 63);
        assert_eq!(cfg.grid_dim, (1, 2, 1));
        assert_eq!(cfg.block_dim, WMMA_OPT_BLOCK_DIM);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn wmma_opt_launch_config_exact_multiple_shape_has_no_extra_tile() {
        let cfg = wmma_opt_launch_config(128, 192);
        assert_eq!(cfg.grid_dim, (3, 2, 1));
    }
}
