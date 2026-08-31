//! `gemm_variant::select_f32_gemm_variant`（イシュー #1035）の opt-in
//! 実行経路。simple（現行 `TILED_F32`）／double-buffer の 2 カーネルを
//! 実際に NVRTC コンパイル・起動するハンドル
//! [`CudaGemmF32VariantSelection`] を提供する。SplitK の 2 カーネルも
//! 引き続きコンパイルするが、通常の [`CudaGemmF32VariantSelection::
//! run_f32`] 経由では選択されない（イシュー #1100。下記「SplitK の位置
//! づけ」参照）。
//!
//! # `internal-diagnostics` feature 限定である理由
//!
//! 本ランは NVRTC 実行不能環境（実機 CUDA toolkit 非搭載）のため、
//! double-buffer カーネルの数値検証・性能 A/B は実機待ちで `#[ignore]`
//! テストへ切り出すほかない（#1034・PR #740→#758 差し戻しの教訓: 未検証
//! カーネルを本番既定経路へ結線しない）。本モジュール全体を
//! `internal-diagnostics` feature（既定 off）でのみコンパイルすることで、
//! **本番既定コンストラクタ（`gemm.rs::CudaGemm::new`）・`run_tiled_f32`・
//! `kernels::TILED_F32` を一切変更せず**、既存の `CudaGemm` に一切のフィー
//! ルド追加も行わない（実装計画 §3・§9: `gemm.rs`・`kernels.rs` への編集を
//! 最小差分に抑え、並列実装中の #1032/#1033 とのコンフリクトを避ける）。
//!
//! # SplitK の位置づけ（イシュー #1100 で診断専用へ変更）
//!
//! GB10 実機実測（#1031）が SplitK 経路の REQ-2 複合判定 FAIL を検出し、
//! イシュー #1100 の計画セッションでその原因が split-K の演算順序
//! （K 方向分割 → 縮約）そのものと CPU 参照実装の丸め方式の非両立と判明
//! した（`gemm_variant.rs` 冒頭「SplitK 撤退の判断」参照。tolerance・
//! 参照実装の変更はユーザー承認必須のため是正不能）。[`gemm_variant::
//! select_f32_gemm_variant`] は SplitK を返さなくなったため
//! [`CudaGemmF32VariantSelection::run_f32`] からも自然に到達しなくなった
//! が、カーネル自体（インデックス計算・決定性）は誤りではないため、
//! spec 側での parity 契約再検討（`out-of-scope-tracking.md` 準拠）に
//! 備え [`CudaGemmF32VariantSelection::run_split_k_forced`]（診断専用・
//! ヒューリスティックを経由せず明示的に起動する）として引き続き到達可能に
//! する。
//!
//! # このモジュールの役割・呼び出し関係
//!
//! - [`CudaGemmF32VariantSelection::new`] は内部で `CudaGemm::new`
//!   （本番既定コンストラクタ）を 1 つ保持しつつ、`kernels_gemm_variants`
//!   の 2 種類のカーネル（DoubleBuffer 用 1 個・SplitK 用 2 個。診断専用）
//!   を fail-soft（コンパイル失敗はその変種のみ `None` 化し `new` 全体を
//!   失敗させない。`gemm.rs::CudaGemm::new` の `wmma_tf32` 系スロットと
//!   同じ設計）でコンパイルする。
//! - [`CudaGemmF32VariantSelection::run_f32`] が
//!   `gemm_variant::select_f32_gemm_variant` の判定結果に従い、
//!   Simple なら内部 `CudaGemm::run_tiled_f32` へ委譲し、DoubleBuffer
//!   なら本モジュールが保持するカーネルを起動する（SplitK は上記のとおり
//!   `run_split_k_forced` 経由でのみ到達する）。
//! - 呼び出し元: `examples/gemm_f32_variant_bench.rs`（5 回計測中央値
//!   A/B ベンチ）・`tests/gemm_f32_variants.rs`（`#[ignore]` 実機テスト）
//!   のみ。本番経路（`gemm_auto.rs::CudaGemmAuto::run_f32`）からは
//!   呼ばれない（スコープ外・実装計画 §8）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::{CudaGemm, validate_gemm_dims, validate_tiled_k_bound};
use crate::gemm_variant;
// `GemmVariantKind` を本モジュール経由で外部（`internal-diagnostics`
// feature 限定の `tests/gemm_f32_variants.rs`・`examples/gemm_f32_variant_
// bench.rs`）へ再公開する。`gemm_variant` モジュール自体は非公開
// （`lib.rs` の `mod gemm_variant;`）のため、これが無いと呼び出し側は
// `selected_variant` の戻り値を変数へ束縛して `Debug` 出力する以外の方法
// （`match`／`assert_eq!` で具体的な variant を検証する等）で扱えない
// （イシュー #1035 PR #1073 レビュー指摘: テストが実際に選ばれた variant
// を検証できず、fail-soft フォールバックが Simple へ静かに落ちても
// 気づけない問題への対処）。
pub use crate::gemm_variant::GemmVariantKind;
use crate::kernels_gemm_variants::{
    SPLITK_PARTIAL_F32, SPLITK_REDUCE_BLOCK_DIM, SPLITK_REDUCE_F32, TILED_DB_F32, TILED_DB_TILE,
};
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// [`TILED_DB_F32`]／split-K 部分和カーネルのブロック次元
/// （`gemm.rs::TILED_BLOCK_DIM` と同じ `kernels::TILE` x `kernels::TILE`
/// を `kernels_gemm_variants::TILED_DB_TILE` 経由で踏襲する。値そのものの
/// 単一の真実源は `kernels::TILE` であり、本定数はこのモジュール専用の
/// 複製である点は `kernels_gemm_variants.rs::TILED_DB_TILE` ドキュメント
/// コメントと同じ）。
const VARIANT_BLOCK_DIM: (u32, u32, u32) = (TILED_DB_TILE, TILED_DB_TILE, 1);

/// `m`／`n` を [`VARIANT_BLOCK_DIM`] で覆う 2D grid を計算する
/// （`gemm.rs::launch_config` と同一のロジック。`gemm.rs` 側の当該関数は
/// モジュール非公開のため複製する——本ファイル冒頭コメント「このモジュール
/// の役割」参照。値変更時は両方を同時に見直す必要がある）。
fn variant_grid_2d(m: u32, n: u32) -> (u32, u32) {
    (
        n.div_ceil(VARIANT_BLOCK_DIM.0),
        m.div_ceil(VARIANT_BLOCK_DIM.1),
    )
}

/// CUDA f32 GEMM の simple / double-buffer / split-K ヒューリスティック
/// 選択を実際に実行する opt-in ハンドル（イシュー #1035）。
pub struct CudaGemmF32VariantSelection {
    /// Simple（現行 `TILED_F32`）へのフォールバック先。本番既定
    /// コンストラクタ（`CudaGemm::new`）で構築するため、Simple 経路は
    /// 本番既定と完全に同一の数値契約・許容誤差を持つ。
    base: CudaGemm,
    stream: Arc<CudaStream>,
    /// DoubleBuffer／SplitK（診断専用）の出力バッファ確保に使うサイズ
    /// クラス別プール（イシュー #1020）。イシュー #1100 で `base`
    /// （`CudaGemm::run_tiled_f32`）と同一条件（都度 `alloc_zeros` では
    /// なくプール経由 `alloc_uninit_f32`）へ揃えた（本ファイル冒頭
    /// 「SplitK の位置づけ」・`run_double_buffer`／`run_split_k` 内の
    /// SAFETY コメント参照。GB10 実機実測〈#1031〉で DoubleBuffer が
    /// N=4096 において `TILED_F32` 単体比 0.08 倍まで悪化した主因の
    /// 有力仮説がバッファ管理差だったための是正）。
    allocator: Arc<CudaAllocator>,
    tiled_db_f32: Option<CudaFunction>,
    tiled_db_f32_error: Option<String>,
    splitk_partial_f32: Option<CudaFunction>,
    splitk_partial_f32_error: Option<String>,
    splitk_reduce_f32: Option<CudaFunction>,
    splitk_reduce_f32_error: Option<String>,
    num_sms: Option<u32>,
}

impl CudaGemmF32VariantSelection {
    /// `device` 上で 3 変種すべてを構築する。DoubleBuffer／SplitK 用
    /// カーネルのコンパイル失敗は該当変種のみ `None`（+ 理由文字列）へ
    /// 退避し、`new` 全体を失敗させない（fail-soft。`gemm.rs::CudaGemm::new`
    /// の `wmma_tf32` 系スロットと同じ設計判断）。`base`（Simple 経路）の
    /// 構築失敗のみ `new` を早期 return させる（naive/tiled 4 カーネルの
    /// コンパイル失敗は環境全体が CUDA GEMM を使用不能という重大な失敗の
    /// ため、`CudaGemm::new` 自身が `?` で表面化させる契約に揃える）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let base = CudaGemm::new(device)?;
        let stream = device.stream().clone();
        let arch = device.arch();

        let (tiled_db_f32, tiled_db_f32_error) =
            match compile_ptx(TILED_DB_F32, arch).and_then(|ptx| {
                Ok(device
                    .context()
                    .load_module(ptx)?
                    .load_function("gemm_tiled_db_f32")?)
            }) {
                Ok(func) => (Some(func), None),
                Err(e) => (None, Some(e.to_string())),
            };

        let (splitk_partial_f32, splitk_partial_f32_error) =
            match compile_ptx(SPLITK_PARTIAL_F32, arch).and_then(|ptx| {
                Ok(device
                    .context()
                    .load_module(ptx)?
                    .load_function("gemm_splitk_partial_f32")?)
            }) {
                Ok(func) => (Some(func), None),
                Err(e) => (None, Some(e.to_string())),
            };

        let (splitk_reduce_f32, splitk_reduce_f32_error) =
            match compile_ptx(SPLITK_REDUCE_F32, arch).and_then(|ptx| {
                Ok(device
                    .context()
                    .load_module(ptx)?
                    .load_function("gemm_splitk_reduce_f32")?)
            }) {
                Ok(func) => (Some(func), None),
                Err(e) => (None, Some(e.to_string())),
            };

        let num_sms = device.multiprocessor_count();
        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            base,
            stream,
            allocator,
            tiled_db_f32,
            tiled_db_f32_error,
            splitk_partial_f32,
            splitk_partial_f32_error,
            splitk_reduce_f32,
            splitk_reduce_f32_error,
            num_sms,
        })
    }

    /// `DoubleBuffer` カーネルが利用可能か（コンパイル・ロード成功）。
    pub fn double_buffer_available(&self) -> bool {
        self.tiled_db_f32.is_some()
    }

    /// `SplitK` の 2 カーネル（部分和・縮約）双方が利用可能か。
    pub fn split_k_available(&self) -> bool {
        self.splitk_partial_f32.is_some() && self.splitk_reduce_f32.is_some()
    }

    /// `DoubleBuffer` カーネルのコンパイル失敗理由（利用可能なら `None`）。
    pub fn double_buffer_error(&self) -> Option<&str> {
        self.tiled_db_f32_error.as_deref()
    }

    /// `SplitK` 部分和カーネルのコンパイル失敗理由。
    pub fn split_k_partial_error(&self) -> Option<&str> {
        self.splitk_partial_f32_error.as_deref()
    }

    /// `SplitK` 縮約カーネルのコンパイル失敗理由。
    pub fn split_k_reduce_error(&self) -> Option<&str> {
        self.splitk_reduce_f32_error.as_deref()
    }

    /// `device.multiprocessor_count()` の実測結果（`new` 時に 1 回だけ
    /// 取得しキャッシュする。取得失敗〈`None`〉は
    /// `gemm_variant::select_f32_gemm_variant` の fail-safe 分岐へ渡り、
    /// 常に `Simple` が選ばれる）。
    pub fn num_sms(&self) -> Option<u32> {
        self.num_sms
    }

    /// `m`/`n`/`k` に対して選ばれる変種を、実際の起動を伴わずに調べる
    /// （ベンチ・テストの可観測性用）。**SplitK は返らない**（イシュー
    /// #1100 で `gemm_variant::select_f32_gemm_variant` の選択候補から
    /// 撤退済み。本ファイル冒頭「SplitK の位置づけ」参照。`run_split_k_
    /// forced` で明示的に起動できる）。
    pub fn selected_variant(&self, m: u32, n: u32, k: u32) -> GemmVariantKind {
        gemm_variant::select_f32_gemm_variant(m, n, k, self.num_sms, self.double_buffer_available())
    }

    /// f32 GEMM を、`gemm_variant::select_f32_gemm_variant` が選ぶ変種で
    /// 実行する。C = A @ B（`m x k` @ `k x n`）。
    ///
    /// 形状検証は [`CudaGemm::run_tiled_f32`] と同じ
    /// `validate_gemm_dims`／`validate_tiled_k_bound` を、変種選択前に
    /// 必ず通す（分岐に関わらず検証を揃える。`rmsnorm.rs::
    /// validate_dw_split_launch` と同じ「分岐前に検証」の順序契約）。
    ///
    /// `GemmVariantKind::SplitK` 分岐は `selected_variant`（＝
    /// `gemm_variant::select_f32_gemm_variant`）がイシュー #1100 以降
    /// 返さなくなったため実行時に到達しないが、`GemmVariantKind` は
    /// 診断専用の `run_split_k_forced` からも使う共有型のため `match` は
    /// 引き続き網羅的に書く（`_` によるワイルドカードで握り潰さない）。
    pub fn run_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;

        match self.selected_variant(m, n, k) {
            GemmVariantKind::Simple => self.base.run_tiled_f32(a, b, m, n, k),
            GemmVariantKind::DoubleBuffer => self.run_double_buffer(a, b, m, n, k),
            GemmVariantKind::SplitK { num_splits } => self.run_split_k(a, b, m, n, k, num_splits),
        }
    }

    /// [`Self::run_split_k_forced`] 呼び出し前に、`m`/`n`/`k` から
    /// `gemm_variant::derive_split_count` が導く「妥当な分割数」を得る
    /// 診断専用ヘルパー（イシュー #1100）。`num_sms` 取得失敗時
    /// （`self.num_sms.is_none()` またはハードウェア異常で `0`）は
    /// 分割の意義を判定できないため `1`（=事実上分割なし）を返す。
    pub fn recommend_split_count(&self, m: u32, n: u32, k: u32) -> u32 {
        match self.num_sms {
            Some(num_sms) if num_sms > 0 => gemm_variant::recommend_split_count(m, n, k, num_sms),
            _ => 1,
        }
    }

    /// [`GemmVariantKind::SplitK`] 経路を、選択ヒューリスティックを経由
    /// せず明示的な `num_splits` で起動する診断専用エントリ（イシュー
    /// #1100。本ファイル冒頭「SplitK の位置づけ」参照）。`run_f32` からは
    /// 到達しない SplitK カーネル自体の数値・決定性検証（`tests/
    /// gemm_f32_variants.rs`）・spec 側の parity 契約再検討に備えた足場
    /// として公開する。`num_splits` の cap 検査（`gemm_variant::
    /// validate_split_k_launch`）は内部の [`Self::run_split_k`] が行う。
    /// 妥当な `num_splits` が分からない場合は [`Self::recommend_split_count`]
    /// を使う。
    pub fn run_split_k_forced(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
        num_splits: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;
        validate_tiled_k_bound(k)?;
        self.run_split_k(a, b, m, n, k, num_splits)
    }

    /// [`GemmVariantKind::DoubleBuffer`] 経路の起動（`m == 0 || n == 0`
    /// は `CudaGemm::run_f32_kernel` と同じ no-op 契約〈空ベクタ〉、
    /// `k == 0` は全 0 ベクタを返す。`gemm.rs` の当該コメント参照）。
    fn run_double_buffer(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }
        // `selected_variant` が DoubleBuffer を選ぶのは
        // `double_buffer_available()`（`tiled_db_f32.is_some()`）が
        // `true` の場合のみ（`gemm_variant::select_f32_gemm_variant` の
        // 契約）。ここに到達した時点で `Some` であることが保証されるが、
        // 呼び出し経路の変更に備え fail-closed に扱う（`unwrap` は
        // 使わない。`.claude/rules/coding-rust.md`）。
        let Some(func) = self.tiled_db_f32.as_ref() else {
            return Err(CudaError::InvalidShape {
                detail: "gemm double-buffer kernel unavailable despite being selected".to_string(),
            });
        };

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        // イシュー #1100（§2.2）: GB10 実機実測で DoubleBuffer が N=4096
        // において `TILED_F32` 単体比 0.08 倍まで悪化した有力仮説は、
        // 本経路が毎回 `stream.alloc_zeros`（都度 `cuMemAlloc`+memset+解放）
        // で C を確保していたバッファ管理差だった（`gemm.rs::run_f32_
        // kernel` は #1020 でプール経由 `alloc_uninit_f32` へ移行済み）。
        // `TILED_DB_F32` は `row < m && col < n` の全要素へ無条件書き込み
        // （`num_tiles == 0` の早期 return パスも `c[row*n+col] = 0.0f`
        // を無条件に書く。`kernels_gemm_variants::TILED_DB_F32` 参照）
        // ため、前利用データの残留は起動直後に必ず上書きされ露出しない
        // （`docs/backend-cuda-pool-allocator-decision.md` §「`alloc_
        // uninit` の適用」の確認済みケースと同型）。`base`（`gemm.rs`）と
        // 同一条件へ揃えることで A/B の対象を「カーネル自体の差」に限定
        // する狙いもある。
        let mut c_dev = self
            .allocator
            .alloc_uninit_f32((m as usize) * (n as usize))?;

        let (grid_x, grid_y) = variant_grid_2d(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: 引数は a_dev/b_dev（それぞれ a.len()/b.len() 要素）・
        // c_dev（m*n 要素の論理長ビュー。`PooledCudaHandle::as_view_mut`）・
        // m_i/n_i/k_i の 5 個で `gemm.rs::run_f32_kernel` と同一の対応
        // 関係。カーネル内の手動境界チェック（三項ガード＋書き込み時
        // ガード。`kernels_gemm_variants::TILED_DB_F32` 参照。REQ-8）と
        // 合わせて OOB 読み書きが起きない根拠とする。grid は `div_ceil` で
        // m/n を包含し末尾ブロックの余剰スレッドはカーネル内境界チェックで
        // 弾かれる。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(LaunchConfig {
                    grid_dim: (grid_x, grid_y, 1),
                    block_dim: VARIANT_BLOCK_DIM,
                    shared_mem_bytes: 0,
                })?;
        }
        // 同期点は readback ヘルパーへ集約する（`gemm.rs::run_f32_kernel`
        // と同じ契約。#1013）。プール割当ハンドルは `DevicePtr` を直接
        // 実装しないため論理長ビュー（`as_view()`）を渡す。
        crate::memory::readback(&self.stream, &c_dev.as_view())
    }

    /// [`GemmVariantKind::SplitK`] 経路の起動（部分和カーネル →
    /// 縮約カーネルの 2 段。`m == 0 || n == 0`／`k == 0` の no-op 契約は
    /// [`Self::run_double_buffer`] と同一）。
    fn run_split_k(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
        num_splits: u32,
    ) -> Result<Vec<f32>, CudaError> {
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }
        // `selected_variant` の契約により到達時点で両カーネルとも `Some`
        // のはずだが、`run_double_buffer` と同じ理由で fail-closed に扱う。
        let (Some(partial_func), Some(reduce_func)) = (
            self.splitk_partial_f32.as_ref(),
            self.splitk_reduce_f32.as_ref(),
        ) else {
            return Err(CudaError::InvalidShape {
                detail: "gemm split-k kernels unavailable despite being selected".to_string(),
            });
        };

        // 部分和バッファサイズを起動前に再検証する（`select_f32_gemm_
        // variant` が既に cap 検査済みだが、`gemm_variant::
        // validate_split_k_launch` ドキュメントコメントに従い分岐に
        // 関わらず独立して検証する契約）。
        gemm_variant::validate_split_k_launch(m, n, num_splits)?;

        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;

        let partial_len = (num_splits as usize)
            .checked_mul(m as usize)
            .and_then(|v| v.checked_mul(n as usize))
            .ok_or_else(|| CudaError::InvalidShape {
                detail: format!(
                    "gemm split-k partial buffer length overflowed usize: \
                     num_splits={num_splits}, m={m}, n={n}"
                ),
            })?;
        // イシュー #1100（§2.2）: `run_double_buffer` と同じ理由で
        // プール経由 `alloc_uninit_f32` へ揃える。`SPLITK_PARTIAL_F32` は
        // `row < m && col < n` を満たす全 `(bz, row, col)` へ無条件に
        // 書く（末尾の空分割も `acc=0.0f` のまま無条件出力。
        // `kernels_gemm_variants::SPLITK_PARTIAL_F32` ドキュメントコメント
        // 「末尾要素ブロックの扱い」参照）ため `c_partial_dev` は全要素が
        // 起動直後に上書きされる。`SPLITK_REDUCE_F32` も `idx < total` の
        // 全要素へ無条件に `c[idx] = acc;` を書く（`kernels_gemm_
        // variants::SPLITK_REDUCE_F32` 参照）ため `c_dev` も同様。
        let mut c_partial_dev = self.allocator.alloc_uninit_f32(partial_len)?;
        let mut c_dev = self
            .allocator
            .alloc_uninit_f32((m as usize) * (n as usize))?;

        let (grid_x, grid_y) = variant_grid_2d(m, n);
        let (m_i, n_i, k_i, num_splits_i) = (m as i32, n as i32, k as i32, num_splits as i32);

        // SAFETY（第 1 カーネル）: `c_partial_dev` は `num_splits*m*n`
        // 要素（論理長ビュー。`PooledCudaHandle::as_view_mut`）で
        // `blockIdx.z` が一意に担当範囲へのみ書くため CTA 間の書き込み
        // 競合は起きない（atomics 不使用でも決定的。`kernels_gemm_
        // variants::SPLITK_PARTIAL_F32` ドキュメントコメント「末尾要素
        // ブロックの扱い」参照）。
        unsafe {
            self.stream
                .launch_builder(partial_func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&mut c_partial_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&num_splits_i)
                .launch(LaunchConfig {
                    grid_dim: (grid_x, grid_y, num_splits),
                    block_dim: VARIANT_BLOCK_DIM,
                    shared_mem_bytes: 0,
                })?;
        }

        // SAFETY（第 2 カーネル）: 両カーネルは同一 `self.stream` 上へ
        // 順に enqueue されるため、stream 順序保証により縮約カーネルは
        // 部分和カーネルの完了後にのみ `c_partial_dev` を読む（明示的な
        // 追加同期は不要。`rmsnorm.rs` の dw split-K と同じ順序契約）。
        let total = (m as u64) * (n as u64);
        let reduce_grid = total
            .div_ceil(u64::from(SPLITK_REDUCE_BLOCK_DIM))
            .min(u64::from(u32::MAX)) as u32;
        let reduce_grid = reduce_grid.max(1);
        unsafe {
            self.stream
                .launch_builder(reduce_func)
                .arg(&c_partial_dev.as_view())
                .arg(&mut c_dev.as_view_mut())
                .arg(&m_i)
                .arg(&n_i)
                .arg(&num_splits_i)
                .launch(LaunchConfig {
                    grid_dim: (reduce_grid, 1, 1),
                    block_dim: (SPLITK_REDUCE_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // 同期点は readback ヘルパーへ集約する（`run_double_buffer` と
        // 同じ契約。#1013）。
        crate::memory::readback(&self.stream, &c_dev.as_view())
    }
}
