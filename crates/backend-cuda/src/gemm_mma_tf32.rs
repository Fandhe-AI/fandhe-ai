//! TF32 `mma.sync`(m16n8k8)/`ldmatrix`/`cp.async` GEMM の起動 API
//! （イシュー #801）。
//!
//! `CudaMmaTf32Gemm` は `kernels_mma_tf32::mma_tf32_source()` を NVRTC
//! コンパイル・保持し、以降はホスト側スライスを渡すだけで GPU 実行できる
//! 境界を担う（`gemm_mma.rs::CudaMmaGemm` と同じ責務分割・同じ API 形状。
//! `kernels_mma_tf32.rs` 冒頭コメント「位置づけ・非結線」参照）。
//!
//! ホスト側形状検証は `gemm.rs::validate_gemm_dims`（`pub(crate)`）を
//! そのまま再利用し判定ロジックを複製しない。加えて本経路固有の
//! `cp.async` 16 バイト（f32 4 要素）整列制約・グリッド上限・K タイル
//! 添字オーバーフロー上限を追加で検証する。
//!
//! compute capability ゲートは `gemm_mma.rs::check_min_compute_capability`
//! （`pub(crate)`）を再利用する（f16 mma.sync 経路〈cc>=8.0〉と同一の
//! `cp.async`/`ldmatrix` 命令セットが要求する下限のため、ゲート自体を
//! 独立定義せず単一の真実源を共有する）。
//!
//! **#839 で凍結判断済み**: #838 の DGX Spark GB10 実機実測で数値一致
//! 6 本中 4 本 FAIL（`m=16 n=8 k=8` で `fail_count=128/128`。TF32 精度差
//! では説明不能な機能欠陥の疑い）となったため、#839 は本経路を不採用
//! （凍結）と確定した。
//!
//! **#852 で原因調査・修正済み・ただし凍結は継続**: 原因は
//! `kernels_mma_tf32.rs::LDSM_A_FRAG` の A フラグメント ldmatrix 象限
//! マッピングの PTX ISA 誤読（機能欠陥）であり、修正後の実機再実測で
//! `m=16 n=8 k=8` の FAIL 件数は 128/128 → 12/128・`mean_rel_err` は
//! 0.97 → 8.9e-4 まで縮小した（主要な機能欠陥は解消。詳細は
//! `docs/perf/cuda-gemm-mma-tf32-ab.md` §8）。**しかし数値一致 6 本
//! 全 pass には至っておらず**、残存 FAIL の原因は TF32 丸め誤差・
//! 機能欠陥のいずれとも確定していない（GPU-GPU 相互一致テストの誤差が
//! CPU 参照比較より小さいことは、両経路が共有する TF32 丸め誤差成分の
//! 相殺でも説明でき、TF32 丸め誤差説を反証する論拠にはならない。詳細は
//! `docs/perf/cuda-gemm-mma-tf32-ab.md` §8.4）。再評価条件 (b)（数値一致
//! 6 本 pass）は未充足のため凍結は継続する。凍結解除の判断・性能
//! 再評価は #835 系の後続に委ねる。再評価条件全体は
//! `docs/perf/cuda-gemm-mma-tf32-ab.md` §5.1 参照。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::gemm_mma::check_min_compute_capability;
use crate::kernels_mma_tf32;
use crate::memory::GuardedSlice;
use crate::nvrtc::compile_ptx;

/// `kernels_mma_tf32::MMA_TF32_BLOCK_THREADS` に 1:1 対応するブロック
/// 次元。`gemm_mma_tf32x3.rs::CudaMmaTf32x3Gemm`（イシュー #1355）も
/// タイル構成を完全にエイリアスしているため同じブロック次元で起動する
/// （`kernels_mma_tf32x3.rs` 冒頭コメント「タイル構成のエイリアス」参照）。
pub(crate) const MMA_TF32_BLOCK_DIM: (u32, u32, u32) =
    (kernels_mma_tf32::MMA_TF32_BLOCK_THREADS, 1, 1);

/// TF32 `mma.sync`(m16n8k8) GEMM カーネルのコンパイル済みハンドルを保持
/// する。`stream` は `CudaDevice` から `Arc` クローンで受け取る
/// （`gemm_mma.rs::CudaMmaGemm` と同じ共有契約）。
///
/// **本番結線なし**（`kernels_mma_tf32.rs` 冒頭コメント「位置づけ・
/// 非結線」参照）: 本経路は `ops.rs`／`gemm_auto.rs`／`gemm.rs` の
/// ディスパッチから呼ばれない。テスト・A/B 比較（#802）専用の直接指定
/// API として提供する。**#839 で凍結判断済み**（本ファイル冒頭 `//!`
/// 参照）。
pub struct CudaMmaTf32Gemm {
    stream: Arc<CudaStream>,
    /// capture 排他（`with_driver_call`）に使うデバイス ordinal
    /// （codex-review P0 指摘対応・PR #1390 再々修正。`gemm.rs::
    /// CudaGemm::ordinal` と同じ役割）。
    ordinal: usize,
    mma_tf32: CudaFunction,
}

impl CudaMmaTf32Gemm {
    /// `device` 上で TF32 `mma.sync`(m16n8k8) GEMM カーネルを NVRTC
    /// コンパイルし保持するハンドルを構築する。
    ///
    /// 手順: (1) `check_min_compute_capability`（cc>=8.0。`gemm_mma.rs`
    /// と共有するゲート）→ (2) `kernels_mma_tf32::mma_tf32_source()` を
    /// `device.arch()` 向けに `nvrtc::compile_ptx` でコンパイル → (3)
    /// `device.context().load_module()` → `load_function("gemm_mma_tf32")`。
    /// `libnvrtc` 不在時は `CudaError::NvrtcUnavailable` を返す
    /// （`compile_ptx` のプローブゲート経由。panic しない。
    /// `gemm_mma.rs::CudaMmaGemm::new` と同一契約）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        check_min_compute_capability(device)?;

        let arch = device.arch();
        let ptx = compile_ptx(kernels_mma_tf32::mma_tf32_source(), arch)?;
        let mma_tf32 = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_mma_tf32")?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            mma_tf32,
        })
    }

    /// `Self` の公開 driver 呼び出し系メソッド（H2D 転送・確保・
    /// カーネル起動・D2H readback）を CUDA Graph capture 排他へ参加
    /// させる（codex-review P0 指摘対応・PR #1390 再々修正。実体は
    /// `context_cache::with_driver_call` へ委譲。`gemm.rs::CudaGemm::
    /// with_driver_call` doc コメント参照）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// A・B（f32）を渡すだけで GPU 実行し C（f32）を得る一括 API。
    ///
    /// 検証順序（`gemm_mma.rs::CudaMmaGemm::run_f16` と同一設計）:
    /// `validate_gemm_dims`（i32 積ガード含む）を常に先行させる → no-op
    /// 形状（`m==0 || n==0 || k==0`）の早期 return → 本経路固有の
    /// `validate_mma_tf32_alignment`／`validate_mma_tf32_grid_bounds`／
    /// `validate_mma_tf32_k_bound`。整列検証・上限検証を no-op 判定より
    /// 後に置く理由も同一（実際にはカーネルを起動しない形状まで誤って
    /// 拒否しないため）。
    ///
    /// **ゼロ次元形状の契約**（#853 で明文化。他バックエンドの同契約と
    /// 整合）: `m==0 || n==0` はカーネル起動なしで `Ok(空 Vec)` を返す。
    /// `k==0` はカーネル起動なしで `Ok(vec![0.0; m*n])`（GEMM 契約: K
    /// ループが空なら結果はゼロ行列）を返す。いずれの場合も
    /// `validate_gemm_dims` が先行するため、no-op 形状でも
    /// `a.len()==m*k`・`b.len()==k*n` の厳密一致（0 要素を含む）が必要
    /// — 呼び出し側が空スライスを渡してよいのは対応する次元が実際に 0
    /// のときだけであり、例えば `k==0` の no-op でも `n>0` なら `b` は
    /// `k*n==0` 要素、`m>0` なら `a` は `m*k==0` 要素という具合に、他の
    /// 次元の値に関わらず「その次元の積」で決まる長さを満たす必要が
    /// ある（`tests/gemm_mma_tf32.rs::
    /// mma_tf32_zero_dim_shape_returns_empty_without_launch` 参照）。
    pub fn run_tf32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;

        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        validate_mma_tf32_alignment(n, k)?;
        validate_mma_tf32_grid_bounds(m)?;
        validate_mma_tf32_k_bound(k)?;

        let (a_dev, b_dev) = self.upload_f32(a, b)?;
        let mut c_dev = self.alloc_output_f32(m, n)?;
        self.launch_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k)?;
        self.download_f32(&c_dev)
    }

    /// A・B をホスト→デバイスへ転送する（`run_tf32` の H2D 部分の
    /// 切り出し。ベンチマークが GPU 実行時間のみを計測できるよう、転送と
    /// カーネル実行を分離する。`gemm_mma.rs::CudaMmaGemm::upload_f16` と
    /// 同じ理由）。
    ///
    /// 戻り値は生の `CudaSlice<f32>` ではなく [`GuardedSlice<f32>`]
    /// （codex-review P0 指摘対応・PR #1390 再々修正。理由は
    /// `gemm.rs::CudaGemm::upload_f32` ドキュメンテーションコメント
    /// 参照）。
    pub fn upload_f32(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<(GuardedSlice<f32>, GuardedSlice<f32>), CudaError> {
        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let b_dev = self.stream.clone_htod(b)?;
            Ok((
                GuardedSlice::new(self.ordinal, a_dev),
                GuardedSlice::new(self.ordinal, b_dev),
            ))
        })
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（`run_tf32` のバッファ
    /// 確保部分の切り出し）。戻り値の型については [`Self::upload_f32`]
    /// ドキュメンテーションコメント参照。
    pub fn alloc_output_f32(&self, m: u32, n: u32) -> Result<GuardedSlice<f32>, CudaError> {
        self.with_driver_call(|| {
            let c_dev = self
                .stream
                .alloc_zeros::<f32>((m as usize) * (n as usize))?;
            Ok(GuardedSlice::new(self.ordinal, c_dev))
        })
    }

    /// デバイス常駐済みの A/B/C バッファに対してカーネルをストリームへ
    /// 非同期投入する（H2D/D2H を含まない「GPU 実行のみ」の区間）。
    ///
    /// 非同期投入契約（イシュー #1013）: 本関数は完了を待たない。完了
    /// 保証は呼び出し元の次の同期点（`download_f32` 等）に委ねる。
    ///
    /// safe な公開 API であるため、呼び出し元の事前検証に依存せず本関数
    /// 自身が `run_tf32` と同じ形状検証（`validate_gemm_dims`・
    /// `validate_mma_tf32_alignment`・`validate_mma_tf32_grid_bounds`・
    /// `validate_mma_tf32_k_bound`）およびデバイスバッファ長検証
    /// （`a_dev`/`b_dev`/`c_dev`）を行う（`gemm_mma.rs::CudaMmaGemm::
    /// launch_f16` ドキュメンテーションコメント「PR #349 codex-review
    /// 指摘 P0」と同一方針）。
    ///
    /// no-op 形状（`m == 0 || n == 0 || k == 0`）は `run_tf32` と同じ
    /// 契約で処理する（PR #823 codex-review P1 是正）: `run_tf32` は
    /// no-op 判定をカーネル起動前に済ませ `launch_tf32` を呼ばずに
    /// return するため、`run_tf32` 経由では本関数までゼロ形状が到達
    /// しない。しかし本関数は「本経路固有の 3 検証（整列・グリッド・K
    /// 上限）が no-op 判定より後に来る」ことを保証する唯一の場所ではなく、
    /// `run_tf32` を経由しない直接呼び出し（テスト・ベンチマーク。構造体
    /// ドキュメンテーションコメント「本番結線なし」参照）にも safe な
    /// 公開 API として同一契約を守る必要がある。`m == 0 || n == 0` は
    /// `mma_tf32_launch_config` がゼロの grid 次元（`div_ceil(0, BM/BN)
    /// == 0`）を生成し空グリッド起動（no-op）になるため検証・起動を
    /// 経ずに成功として早期 return する。`k == 0` は m,n > 0 のまま
    /// カーネルを起動すると K ループが一度も走らず `c_dev` の既存内容
    /// （呼び出し元がゼロ初期化していない場合は未定義）を GEMM の結果
    /// （ゼロ行列）へ更新しないため、`memset_zeros` で明示的にゼロ化
    /// してから return する（`run_tf32` の `vec![0.0f32; ...]` 一括
    /// API 版と同じ契約をデバイス常駐バッファ版で再現する）。
    pub fn launch_tf32(
        &self,
        a_dev: &GuardedSlice<f32>,
        b_dev: &GuardedSlice<f32>,
        c_dev: &mut GuardedSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        // codex-review P0 指摘対応（PR #1390 再々修正）: `Self::
        // with_driver_call` で `launch_mma_tf32_family`（共有起動本体）
        // 呼び出しを capture 排他へ参加させる。`GuardedSlice::as_raw`／
        // `as_raw_mut`（crate 内部限定）は `memory.rs::GuardedSlice`
        // ドキュメンテーションコメント「公開アクセス面」参照
        // （codex-review P0 再指摘対応・PR #1390 再々々修正）。
        self.with_driver_call(|| {
            launch_mma_tf32_family(
                &self.stream,
                &self.mma_tf32,
                a_dev.as_raw(),
                b_dev.as_raw(),
                c_dev.as_raw_mut(),
                m,
                n,
                k,
            )
        })
    }

    /// C をデバイス→ホストへ転送する（`run_tf32` の D2H 部分の切り出し）。
    ///
    /// 同期点（#1013）: 常駐 `launch_tf32` は非同期投入のみで完了を待たない
    /// ため、本関数が readback ヘルパー経由で完了を確定する。codex-review
    /// P0 指摘対応（PR #1390 再々修正）: `Self::with_driver_call` で
    /// capture 排他へ参加させる。
    pub fn download_f32(&self, c_dev: &GuardedSlice<f32>) -> Result<Vec<f32>, CudaError> {
        self.with_driver_call(|| crate::memory::readback(&self.stream, c_dev.as_raw()))
    }

    /// ストリームの完了を明示的に待つ（イシュー #1013。
    /// `gemm.rs::CudaGemm::synchronize` と同じ理由の公開 API）。
    /// codex-review P0 指摘対応（PR #1390 再々修正）: `Self::with_driver_call`
    /// で capture 排他へ参加させる。
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.with_driver_call(|| Ok(self.stream.synchronize()?))
    }
}

/// `CudaMmaTf32Gemm::launch_tf32`（単発 TF32）と
/// `gemm_mma_tf32x3::CudaMmaTf32x3Gemm::launch_tf32x3`（3×TF32。イシュー
/// #1355）が共有するカーネル起動本体。両カーネルはシグネチャ
/// （`(a, b, c, m, n, k)`）・タイル構成（`kernels_mma_tf32x3.rs` が
/// `kernels_mma_tf32` の定数をエイリアスして同一値に固定）・境界検査
/// 契約（REQ-8: cp.async ゼロ充填・整列クランプ・エピローグ guarded
/// store）が完全に同一であるため、形状検証・no-op/`k==0` 分岐・起動
/// 引数の組み立てをここへ集約し重複させない（`kernels_mma_tf32x3.rs`
/// 冒頭コメント「本番結線しない新規 unsafe」不使用の根拠でもある:
/// x3 経路はこのヘルパーの既存 `unsafe` 呼び出しを再利用するのみで、
/// 新規に `unsafe` ブロックを追加しない）。
///
/// 検証順序（`gemm_mma.rs::CudaMmaGemm::run_f16` と同一設計）:
/// `validate_gemm_dims`（i32 積ガード含む）を常に先行させる → no-op
/// 形状（`m==0 || n==0`）の早期 return → `k==0` の memset 早期 return →
/// 本経路固有の `validate_mma_tf32_alignment`／`validate_mma_tf32_grid_
/// bounds`／`validate_mma_tf32_k_bound`。
#[allow(clippy::too_many_arguments)] // 単発 TF32／3×TF32 の 2 呼び出し元が共有する起動本体のため引数集約は妥当（責務分割の代替案はホスト側の重複を招く）
pub(crate) fn launch_mma_tf32_family(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    a_dev: &CudaSlice<f32>,
    b_dev: &CudaSlice<f32>,
    c_dev: &mut CudaSlice<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(), CudaError> {
    validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
    crate::gemm::validate_output_len(c_dev.len(), m, n)?;

    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        // `memset_zeros` は非同期発行のみに留める（イシュー #1013
        // で本関数（常駐 API）の契約を「非同期投入のみ。完了保証は
        // 呼び出し元の次の同期点（`download_f32`／`MemoryOps::
        // download`／明示 `synchronize`）に委ねる」へ統一した。PR #823
        // codex-review 指摘（旧: この早期 return パスが `synchronize`
        // を呼ばずに戻っていたレース）は、単一ストリームの FIFO 順序
        // 保証により後続の同期点が本 `memset_zeros` を含む全ての先行
        // 投入を合わせて待つため、契約変更後も再発しない
        // （`transpose.rs` の同型分岐と同じ判断。設計文書 §3〜§4）。
        stream.memset_zeros(c_dev)?;
        return Ok(());
    }

    validate_mma_tf32_alignment(n, k)?;
    validate_mma_tf32_grid_bounds(m)?;
    validate_mma_tf32_k_bound(k)?;

    let cfg = mma_tf32_launch_config(m, n);
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

    // SAFETY: カーネル引数は a_dev/b_dev/c_dev（それぞれ a.len()/
    // b.len()/(m*n) 要素の確保済みデバイスバッファ）と m_i/n_i/k_i の
    // 5 個・型・個数が、上記で検証済みの m/n/k と 1:1 対応する。
    // カーネル内の手動境界チェック（cp.async src-size ゼロ充填・
    // エピローグ guarded store。`kernels_mma_tf32.rs`／
    // `kernels_mma_tf32x3.rs` 参照、REQ-8）と合わせて OOB 読み書きが
    // 起きない根拠とする。グリッド次元は `MMA_TF32_BM`/`MMA_TF32_BN`
    // 単位の div_ceil で m/n を包含するよう構築しており
    // （`mma_tf32_launch_config`）、末尾タイルの余剰はカーネル内境界
    // チェックで弾かれる。共有メモリは静的 `__shared__` 配列のみを
    // 使用するため `shared_mem_bytes` は 0 のままでよい
    // （`kernels_mma_tf32.rs::MMA_TF32_SHARED_MEM_BYTES` = 28,416B は
    // per-block 静的上限 48KiB 内。x3 経路は同一定数をエイリアスして
    // おり静的 SMEM サイズも完全に同一）。`func` は呼び出し元
    // （`CudaMmaTf32Gemm::new`／`CudaMmaTf32x3Gemm::new`）がそれぞれ
    // 対応するエントリポイント名（`gemm_mma_tf32`／`gemm_mma_tf32x3`）で
    // `load_function` 済みのハンドルであり、引数の型・個数はどちらの
    // カーネルも同一シグネチャを持つため共通で安全。
    unsafe {
        stream
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

/// TF32 `mma.sync`(m16n8k8) 経路が要求する cp.async 16 バイト（f32 4
/// 要素）整列制約を検証する（`gemm_mma.rs::validate_mma_alignment` の
/// f32 版。f16 は 8 要素/16B、f32 は 4 要素/16B である点のみ異なる。
/// `kernels_wmma_opt.rs::WMMA_TF32_STAGED_K_TILE` 系整列検証と同じ根拠）。
///
/// A の行ストライドは `k`、B の行ストライドは `n` であり、共有メモリ側の
/// タイル幅（`MMA_TF32_BK`/`MMA_TF32_BN`）が共に 4 の倍数であることと
/// 合わせて `k % 4 == 0 && n % 4 == 0` を満たさない限り、行境界をまたぐ
/// 列オフセットが 16 バイト境界からずれうる（`kernels_mma_tf32.rs::
/// MMA_TF32_BODY` 内 `LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP` コメント
/// 「REQ-8」参照）。
///
/// `validate_mma_alignment`（f16 mma.sync 経路）は独立経路の唯一のゲート
/// のため fail-closed に拒否する契約であり、本関数も同じ契約を採る
/// （TF32 staged 経路の `wmma_tf32_staged_alignment_ok` のような 3 段
/// フォールバック選択条件ではなく、`CudaMmaTf32Gemm` は単一カーネルの
/// 直接指定 API のため）。
pub(crate) fn validate_mma_tf32_alignment(n: u32, k: u32) -> Result<(), CudaError> {
    if !k.is_multiple_of(4) || !n.is_multiple_of(4) {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "TF32 mma.sync(m16n8k8) path requires k % 4 == 0 && n % 4 == 0 \
                 (cp.async 16-byte / f32 4-element transfer granularity; \
                 kernels_mma_tf32.rs doc comment \"整列制約\"), but got n={n}, k={k}"
            ),
        });
    }
    Ok(())
}

/// CUDA の grid 次元 y 成分の上限（65,535。全 compute capability 共通）に
/// 対する `mma_tf32_launch_config` の y 成分（`m.div_ceil(MMA_TF32_BM)`）
/// 超過を検証する（`gemm_mma.rs::validate_mma_grid_bounds` と同じ根拠・
/// 同じ「経路固有の追加検証」パターン）。
pub(crate) fn validate_mma_tf32_grid_bounds(m: u32) -> Result<(), CudaError> {
    const MAX_GRID_DIM_Y: u32 = 65_535;
    let grid_y = m.div_ceil(kernels_mma_tf32::MMA_TF32_BM);
    if grid_y > MAX_GRID_DIM_Y {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "TF32 mma.sync(m16n8k8) path grid_dim.y \
                 (m.div_ceil(MMA_TF32_BM)={grid_y}) exceeds CUDA's {MAX_GRID_DIM_Y} \
                 limit for grid dimensions y/z (MMA_TF32_BM={}); m={m} is too large",
                kernels_mma_tf32::MMA_TF32_BM
            ),
        });
    }
    Ok(())
}

/// K タイル添字（`t * MMA_TF32_BK`。カーネル内 `int` 算術）が `i32` を
/// オーバーフローしないことを検証する（`gemm.rs::
/// validate_wmma_tf32_staged_k_bound` と同型・同じ厳密算術: `ceil(k /
/// MMA_TF32_BK) * MMA_TF32_BK - 1` を `u64` で計算し `i32::MAX` と比較する）。
pub(crate) fn validate_mma_tf32_k_bound(k: u32) -> Result<(), CudaError> {
    let tile = u64::from(kernels_mma_tf32::MMA_TF32_BK);
    let max_computed_index = if k == 0 {
        0
    } else {
        (u64::from(k)).div_ceil(tile) * tile - 1
    };
    if max_computed_index > i32::MAX as u64 {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k tile-index arithmetic for TF32 mma.sync(m16n8k8) kernel would \
                 overflow i32: k={k}, max_computed_index={max_computed_index}, \
                 MMA_TF32_BK={}",
                kernels_mma_tf32::MMA_TF32_BK
            ),
        });
    }
    Ok(())
}

/// TF32 mma.sync カーネル専用のグリッド次元計算。1 ブロック = C の
/// `MMA_TF32_BM x MMA_TF32_BN` タイル 1 個を担当するため、
/// `div_ceil(n, MMA_TF32_BN)` x `div_ceil(m, MMA_TF32_BM)` のグリッドを
/// 構築する（`gemm_mma.rs::mma_launch_config` と同じ設計）。
/// `gemm_mma_tf32x3.rs`（イシュー #1355）もタイル定数を完全にエイリアス
/// しているため本関数をそのまま再利用する（`pub(crate)`）。
pub(crate) fn mma_tf32_launch_config(m: u32, n: u32) -> LaunchConfig {
    let grid_dim = (
        n.div_ceil(kernels_mma_tf32::MMA_TF32_BN),
        m.div_ceil(kernels_mma_tf32::MMA_TF32_BM),
        1,
    );
    LaunchConfig {
        grid_dim,
        block_dim: MMA_TF32_BLOCK_DIM,
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mma_tf32_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        // MMA_TF32_BM=64/MMA_TF32_BN=64。65x65 を覆うには grid (2, 2) が
        // 必要（div_ceil(65,64)=2）。
        let cfg = mma_tf32_launch_config(65, 65);
        assert_eq!(cfg.grid_dim, (2, 2, 1));
        assert_eq!(cfg.block_dim, MMA_TF32_BLOCK_DIM);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn mma_tf32_launch_config_exact_multiple_shape_has_no_extra_tile() {
        let cfg = mma_tf32_launch_config(128, 192);
        assert_eq!(cfg.grid_dim, (3, 2, 1));
    }

    #[test]
    fn validate_mma_tf32_alignment_accepts_multiples_of_four() {
        assert!(validate_mma_tf32_alignment(64, 32).is_ok());
        assert!(validate_mma_tf32_alignment(4, 4).is_ok());
    }

    #[test]
    fn validate_mma_tf32_alignment_rejects_non_multiple_n() {
        let err = validate_mma_tf32_alignment(9, 32).expect_err("n=9 is not a multiple of 4");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_tf32_alignment_rejects_non_multiple_k() {
        let err = validate_mma_tf32_alignment(64, 17).expect_err("k=17 is not a multiple of 4");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_tf32_grid_bounds_accepts_shapes_within_limit() {
        assert!(validate_mma_tf32_grid_bounds(65_535 * kernels_mma_tf32::MMA_TF32_BM).is_ok());
    }

    #[test]
    fn validate_mma_tf32_grid_bounds_rejects_m_exceeding_grid_y_limit() {
        let err = validate_mma_tf32_grid_bounds(65_535 * kernels_mma_tf32::MMA_TF32_BM + 1)
            .expect_err("grid_dim.y must exceed CUDA's 65,535 limit");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_tf32_k_bound_accepts_ordinary_k() {
        assert!(validate_mma_tf32_k_bound(0).is_ok());
        assert!(validate_mma_tf32_k_bound(4096).is_ok());
    }

    #[test]
    fn validate_mma_tf32_k_bound_accepts_full_range_up_to_i32_max() {
        // ceil(i32::MAX / MMA_TF32_BK) * MMA_TF32_BK - 1 は MMA_TF32_BK=16
        // では i32::MAX 未満に収まる（余裕を持って受理される）ことを確認。
        assert!(validate_mma_tf32_k_bound(i32::MAX as u32).is_ok());
    }
}
