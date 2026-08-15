//! WMMA（Tensor Core）GEMM の共有メモリ・タイル最適化版カーネルソース
//! （TASK-11.1d・#63。NVRTC 実行時コンパイル用の静的文字列）。
//!
//! `kernels.rs::WMMA_TF32_F32`（#62）・`kernels_wmma.rs::WMMA_F16`（#61）は
//! いずれもブロックタイル = warp タイル = fragment タイルという「warp あたり
//! fragment 1 個のみ」の安全側最小構成であり、Tensor Core の計算密度を
//! 活かせていない（両ファイル冒頭ドキュメントコメント「タイル拡大・
//! warp あたり複数 fragment 化は #63 のスコープ」参照）。本モジュールは
//! `docs/cuda-tensor-core-design.md` 4.2 節が #63 に引き継いだ最適化
//! （タイル拡大・レジスタブロッキング・バンクコンフリクト回避パディング・
//! ダブルバッファリング）を実装する。
//!
//! `gemm.rs`／`gemm_wmma.rs`（呼び出し元）は本モジュールの定数・カーネルを
//! `nvrtc::compile_ptx` に渡して `CudaFunction` を得る。既存カーネルと同じく
//! ソースを `nvcc` で事前コンパイルせず文字列のまま埋め込み、ビルド時に
//! nvcc／CUDA ヘッダを一切要求しない契約（TASK-1.7・
//! `.claude/rules/deps-policy.md`）を維持する。
//!
//! # 公開 API への影響（フォールバック方針）
//!
//! REQ-11 は明示切替 API を提供しない方針（設計メモ 10 節）のため、公開 API
//! （`CudaGemm::run_wmma_tf32`／`CudaWmmaGemm::run_f16`）のシグネチャは変更
//! しない。本モジュールのカーネルは `gemm.rs`／`gemm_wmma.rs` 側で
//! 「opt カーネルが `CudaGemm::new`／`CudaWmmaGemm::new` 時点でコンパイル・
//! ロードに成功していればそちらを優先し、失敗していれば #61/#62 の基本
//! WMMA カーネルへ自動フォールバックする」という `Option` パターン
//! （`kernels.rs::WMMA_TF32_F32` の `wmma_tf32`／`wmma_tf32_error` と同方式）
//! で結線される。opt カーネルのコンパイル失敗は naive／tiled／基本 WMMA の
//! 可用性を道連れにしない。
//!
//! # サンドボックス制約と安全側設計
//!
//! このモジュールの CUDA C++ ソースは、CUDA toolkit／実機が存在しない
//! サンドボックス環境で作成されており、NVRTC による実コンパイル検証が
//! できない（#61 冒頭ドキュメントコメントと同じ制約）。緩和策:
//!
//! 1. opt カーネルは基本 WMMA カーネルと独立にコンパイル・ロードし、
//!    失敗しても `Option::None` として扱い基本カーネルへフォールバックする
//!    （上記「公開 API への影響」参照）。
//! 2. タイル・パディング定数は `#define` パラメータ化し、Rust 側の定数を
//!    「唯一の真実源」として文字列突合テストでロックする（`kernels.rs::
//!    tile_constant_matches_kernel_source_define` と同じ方式。実機での
//!    チューニング（#64）を小差分化する）。
//! 3. 不変条件（ブロックタイルが fragment 辺の倍数・スレッド数が warp 数
//!    ×32 に一致等）は `gemm.rs`／`gemm_wmma.rs` 側の `const` アサーション
//!    で機械検査する（`kernels.rs::WMMA_TF32_BLOCK_M` 系と同じ方式）。
//!
//! # タイル構成
//!
//! TF32 opt・f16 opt とも共通してブロックタイル 64×64・warp タイル 32×32
//! （2×2 warp グリッド、4 warp = 128 スレッド）・warp あたり `m16n16k*`
//! fragment 2×2 個（レジスタブロッキング）を採用する。K タイル幅・共有
//! メモリのパディング幅は各カーネルのドキュメンテーションコメント参照。
//!
//! # ダブルバッファリング
//!
//! A／B の共有メモリタイルを 2 面確保し、現在の K タイル（`cur`）の
//! `mma_sync` 計算と次の K タイル（`nxt`）のグローバル→共有メモリ
//! プリフェッチを同一ループ本体内で発行する。`cur`／`nxt` は互いに独立した
//! 配列要素（`tile[cur]`／`tile[nxt]`）を指すため、プリフェッチ書き込みと
//! 計算読み出しの間にレースは生じない。ループ末尾の 1 回の `__syncthreads()`
//! が「今回の `cur` 読み出し（計算）」と「今回の `nxt` 書き込み
//! （プリフェッチ）」の両方の完了を保証したうえで `cur`/`nxt` を入れ替える
//! （標準的な 2 段パイプラインの契約。`cp.async` 等の非同期コピー命令は
//! 使わず `__syncthreads()` ベースに限定する。#187 のスコープ外事項）。
//!
//! # 境界検査（REQ-8。省略禁止）
//!
//! 1. **guarded load**: グローバル→共有メモリのロードは全て
//!    `(gr < 境界) ? ... : 0` の三項ガード＋ゼロ充填を維持する
//!    （`kernels.rs::TILED_F32`・`kernels_wmma.rs::WMMA_F16` と同方式）。
//! 2. **エピローグ**: `store_matrix_sync` は fragment 全体を無条件に書く
//!    ため、いったん warp 専有の共有メモリ `c_tile`／`cs_tile` へ store し、
//!    `__syncthreads()` 後に要素単位のガード付きコピーでグローバル C へ
//!    書き戻す（`kernels.rs::WMMA_TF32_F32` エピローグと同方式）。
//! 3. K 端（`k` が K タイル幅の倍数でない）は `num_k_tiles` を桁溢れしない
//!    式 `(k > 0) ? (k - 1) / K_TILE + 1 : 0`（`kernels.rs::TILED_F32` と
//!    同一）で計算し、末尾タイルの余剰要素は guarded load のゼロ充填で
//!    処理する。
//!
//! # アライメント
//!
//! `load_matrix_sync`／`store_matrix_sync` が要求するのは要素サイズへの
//! 自然アライメント（f32: 4 byte、half: 2 byte。16 byte 境界に揃うと
//! 追加の高速パスが選択されるが必須要件ではない）。本モジュールの共有
//! メモリタイルはバンクコンフリクト回避のためパディング幅を `K_TILE`／
//! `BLOCK_N` の非 2 冪数（+4／+8）に取るため、warp オフセット先頭ポインタは
//! 32 byte 境界には揃わない。これは意図した設計判断であり（パディングと
//! 32 byte アライメントは両立不可能なため、バンクコンフリクト回避を優先
//! した）、要素サイズへの自然アライメントは配列宣言の `__align__(32)`
//! （配列全体の先頭アライメント）と要素サイズの倍数であるパディング幅
//! （f32: 4 の倍数、half: 8 の倍数）により常に満たされる。

/// TF32 opt GEMM のブロックタイル一辺（M・N とも 64。2×2 warp グリッド、
/// warp あたり 32×32 = `m16n16k8` fragment 2×2 個を担当する）。
pub const WMMA_TF32_OPT_BLOCK_M: u32 = 64;
pub const WMMA_TF32_OPT_BLOCK_N: u32 = 64;

/// TF32 opt GEMM の共有メモリ K タイル幅。fragment の K 次元
/// （`WMMA_TF32_OPT_FRAG_K` = 8）の 2 倍を 1 回のロードでまとめて取得し、
/// ロード回数を半減させる（設計メモ 4.2 節「k タイル TF32: 16」候補）。
/// `mma_sync` 自体は fragment の K=8 単位で 1 K タイルあたり 2 回発行する。
pub const WMMA_TF32_OPT_K_TILE: u32 = 16;

/// TF32 opt GEMM の fragment M・N 一辺（`m16n16k8` の 16）。
///
/// Rust 側での実利用は `gemm.rs::CudaGemm::new` 内の
/// `const _: () = assert!(...)`（ブロックタイル・warp タイルの倍数関係を
/// コンパイル時検査する）のみで、通常の実行時コードパスからは参照され
/// ない。rustc 1.88 系の dead-code 解析はネストした無名 `const _` 内から
/// のみ参照される `pub const` を誤って未使用と判定する（1.92 以降では
/// 解消済み。`cargo +1.88.0 clippy` と `cargo +1.92.0 clippy` の実測差分で
/// 確認済み。#149 PR CI 指摘対応）。実行時 `debug_assert` への置換は
/// 「CUDA 非搭載の通常 CI では `new` 自体が実行されず検査が効かない」
/// というレビュー指摘 #62 の踏襲事項に反するため行わない。
#[allow(dead_code)]
pub const WMMA_TF32_OPT_FRAG: u32 = 16;

/// TF32 opt GEMM の fragment K 一辺（`m16n16k8` の 8）。
/// `WMMA_TF32_OPT_K_TILE` は必ずこの倍数でなければならない
/// （`gemm.rs` の const アサーションで検査）。
///
/// [`WMMA_TF32_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_TF32_OPT_FRAG_K: u32 = 8;

/// TF32 opt GEMM の warp タイル一辺（32。fragment 辺 16 の 2 倍 =
/// レジスタブロッキング 2×2）。
///
/// [`WMMA_TF32_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_TF32_OPT_WARP_TILE: u32 = 32;

/// TF32 opt GEMM 1 ブロックあたりのスレッド数（4 warp = 128 スレッド。
/// `(WMMA_TF32_OPT_BLOCK_M / WMMA_TF32_OPT_WARP_TILE) *
/// (WMMA_TF32_OPT_BLOCK_N / WMMA_TF32_OPT_WARP_TILE) * 32` = 2×2 warp を
/// 1 次元ブロックとして起動する。`kernels.rs::WMMA_TF32_THREADS` と同じ
/// 「ホスト側ブロック次元とカーネル内 warp グリッドの 1:1 対応」契約）。
pub const WMMA_TF32_OPT_THREADS: u32 = 128;

use std::sync::LazyLock;

use crate::error::CudaError;
use crate::kernels_mma::DimSpec;

/// A タイル（`as_tile[2][BLOCK_M][A_PAD]`）の行幅（パディング後）。
/// `K_TILE`（16）に 4 要素加算し、f32 の `ldm` 制約（4 の倍数）を保ちながら
/// バンクコンフリクトを避ける（設計メモ 4.2 節・本ファイル冒頭
/// ドキュメンテーションコメント「アライメント」参照）。
///
/// [`WMMA_TF32_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_TF32_OPT_A_PAD: u32 = WMMA_TF32_OPT_K_TILE + 4;

/// B タイル（`bs_tile[2][K_TILE][B_PAD]`）の行幅（パディング後）。
/// `BLOCK_N`（64）に 4 要素加算する。A パディングと同じ根拠。
///
/// [`WMMA_TF32_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_TF32_OPT_B_PAD: u32 = WMMA_TF32_OPT_BLOCK_N + 4;

/// WMMA（Tensor Core）を用いた TF32 GEMM の共有メモリ・タイル最適化版
/// （TASK-11.1d・#63）。`kernels::WMMA_TF32_F32`（#62。warp あたり fragment
/// 1 個のみ）に対し、ブロックタイル 64×64・warp あたり fragment 2×2 個
/// （レジスタブロッキング）・バンクコンフリクト回避パディング・
/// ダブルバッファリングを適用する（本ファイル冒頭ドキュメントコメント
/// 参照）。数値契約（TF32 丸め・f32 累算）は `kernels::WMMA_TF32_F32` と
/// 同一（`wmma::__float_to_tf32` による明示変換。統一複合判定の閾値は
/// 変更しない）。
///
/// 受け入れ条件（#63）: tiled 実装（1.832 TFLOPS、PoC-v2-3、M=N=K=4096
/// の f32）を上回る実測（5 回中央値）。実測確定は #64（実機チューニング）
/// に引き継ぐ（本カーネルはサンドボックス環境でコンパイル未検証。上記
/// 「サンドボックス制約と安全側設計」参照）。
///
/// # テンプレート文字列展開（イシュー #516）
///
/// 本カーネルは [`render_wmma_tf32_opt`] のテンプレート展開結果であり、
/// [`WMMA_TF32_OPT_BLOCK_M`] 等 Rust 側タイル定数を初期値とする既定
/// [`WmmaOptKernelConfig`] で展開したソースを [`wmma_tf32_f32_opt_source`]
/// が 1 回だけキャッシュして返す。`kernels_mma.rs::MMA_F16` と同じ方針
/// （`DimSpec` による M/N/K の動的／静的焼き込み選択・fail-closed 構成
/// 検証）だが、本カーネルは `cp.async` パイプライン段数を持たない
/// （ダブルバッファ `cur`/`nxt` は 2 固定）ため config に `stages` 相当の
/// フィールドはない。
pub fn wmma_tf32_f32_opt_source() -> &'static str {
    &WMMA_TF32_F32_OPT_SOURCE
}

static WMMA_TF32_F32_OPT_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_wmma_tf32_opt_unchecked(&WmmaOptKernelConfig::default_tf32()));

/// [`render_wmma_tf32_opt`]／[`render_wmma_f16_opt`] に共通の構成値
/// （ブロックタイル・K タイル幅・shape 焼き込み方式）。
///
/// TF32 opt・f16 opt は fragment 辺（`WMMA_*_OPT_FRAG`）・warp タイル辺
/// （`WMMA_*_OPT_WARP_TILE`）・パディング増分（TF32: +4／f16: +8）が
/// 異なるため、これらは各 `render_*` 関数側の固定値として扱い、本構造体
/// では扱わない（実装計画 4.1 節「dtype の差し替え」: dtype 差異はテンプレート
/// 選択＝呼び出す `render_*` 関数で表現し、config フィールドとしては
/// 汎用化しない）。`Hash + Eq` を導出可能な単純型に留めているのは、
/// `kernels_mma::MmaKernelConfig` と同じ理由（後続 #504 のキャッシュキー
/// 構成要素として使えるようにするため）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WmmaOptKernelConfig {
    /// ブロックタイル M（warp タイル辺の倍数必須）。
    pub block_m: u32,
    /// ブロックタイル N（warp タイル辺の倍数必須）。
    pub block_n: u32,
    /// 共有メモリ K タイル幅。
    pub k_tile: u32,
    /// M 次元の焼き込み方式。
    pub dim_m: DimSpec,
    /// N 次元の焼き込み方式。
    pub dim_n: DimSpec,
    /// K 次元の焼き込み方式。
    pub dim_k: DimSpec,
}

impl WmmaOptKernelConfig {
    /// TF32 opt カーネルの既定構成（現行 Rust 側タイル定数と同一値。
    /// 全次元 `Dynamic`）。
    pub fn default_tf32() -> Self {
        Self {
            block_m: WMMA_TF32_OPT_BLOCK_M,
            block_n: WMMA_TF32_OPT_BLOCK_N,
            k_tile: WMMA_TF32_OPT_K_TILE,
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Dynamic,
        }
    }

    /// f16 opt カーネルの既定構成（現行 Rust 側タイル定数と同一値。
    /// 全次元 `Dynamic`）。
    pub fn default_f16() -> Self {
        Self {
            block_m: WMMA_F16_OPT_BLOCK_M,
            block_n: WMMA_F16_OPT_BLOCK_N,
            k_tile: WMMA_F16_OPT_K_TILE,
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Dynamic,
        }
    }

    /// [`RenderedWmmaOptKernel::validate_launch_shape`] の実体（起動前
    /// 検査）。`dim_m`/`dim_n`/`dim_k` それぞれについて
    /// [`DimSpec::matches_launch_dim`] を実引数 `m`/`n`/`k` に対して検査し、
    /// `Static` 次元の食い違いを fail-closed で拒否する（PR #643
    /// codex-review P1 指摘への対応。`kernels_mma::MmaKernelConfig::validate_launch_shape`
    /// と同じ設計）。`Dynamic` のみの既定 config（`default_tf32`／
    /// `default_f16`）では常に `Ok`。
    #[allow(dead_code)] // 理由は kernels_mma::DimSpec::matches_launch_dim と同じ
    pub fn validate_launch_shape(&self, m: u32, n: u32, k: u32) -> Result<(), CudaError> {
        self.dim_m.matches_launch_dim(m)?;
        self.dim_n.matches_launch_dim(n)?;
        self.dim_k.matches_launch_dim(k)?;
        Ok(())
    }
}

/// [`render_wmma_tf32_opt`]／[`render_wmma_f16_opt`] が返す、展開済み
/// カーネルソースと展開元 [`WmmaOptKernelConfig`] を 1 個にまとめた
/// descriptor（PR #643 codex-review P1 指摘への対応。
/// `kernels_mma::RenderedMmaKernel` と同じ設計）。
///
/// フィールドは非公開とし、ソース取得は
/// [`RenderedWmmaOptKernel::source_for_launch`] の 1 経路のみを公開する
/// （起動前検査を経ない裸の `source()` は公開しない。PR #643 codex-review
/// 再指摘への対応。`kernels_mma::RenderedMmaKernel` と同じ設計変更）。これに
/// より「`Static` 次元を含む config から得たソースは、必ずその config を
/// 経由した起動時形状検査を通らない限りカーネル起動へ渡せない」という
/// 構造的な契約になる（`RenderedMmaKernel` ドキュメンテーションコメント
/// 参照）。`mod kernels_wmma_opt` が非公開モジュールのため、既定構成のみ
/// 消費する現状の呼び出し元からは本構造体・以下の全メソッドが呼ばれず
/// dead-code 解析が誤検知する。`#[allow(dead_code)]` の理由は
/// [`render_wmma_tf32_opt`] と同じ。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedWmmaOptKernel {
    source: String,
    cfg: WmmaOptKernelConfig,
}

impl RenderedWmmaOptKernel {
    /// 起動を意図する形状 `m`/`n`/`k` を検証してから NVRTC
    /// （`nvrtc::compile_ptx`）へ渡すカーネルソース文字列を返す唯一の公開
    /// 経路（PR #643 codex-review P1 再指摘への対応。
    /// `kernels_mma::RenderedMmaKernel::source_for_launch` と同じ設計）。
    /// 裸の `source() -> &str` を独立に公開すると、呼び出し元が起動時形状
    /// 検査を一度も経由せずソースをコンパイル・起動できてしまう
    /// （`RenderedWmmaOptKernel` ドキュメンテーションコメント参照）。
    #[allow(dead_code)]
    pub fn source_for_launch(&self, m: u32, n: u32, k: u32) -> Result<&str, CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        Ok(&self.source)
    }

    /// コンパイル済みの `CudaFunction` を実際に起動する直前、呼び出しの
    /// 都度呼ぶ契約（[`Self::source_for_launch`] は最初の 1 回のみ・
    /// コンパイル前の検査であり、`Dynamic` 次元は起動ごとに異なる
    /// `m`/`n`/`k` を許容しうるため、実際の起動直前の再検査はこのメソッド
    /// が担う）。[`WmmaOptKernelConfig::validate_launch_shape`] へ委譲する。
    #[allow(dead_code)]
    pub fn validate_launch_shape(&self, m: u32, n: u32, k: u32) -> Result<(), CudaError> {
        self.cfg.validate_launch_shape(m, n, k)
    }
}

/// [`WmmaOptKernelConfig`] を TF32 opt カーネル向けに fail-closed 検証する
/// （実装計画 4.2 節。`WARP_TILE`=32・`FRAG`=16・`FRAG_K`=8 は本カーネルの
/// 固定ハードウェア形状であり config では扱わない）。
fn validate_wmma_tf32_opt_config(cfg: &WmmaOptKernelConfig) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };
    let warp_tile = WMMA_TF32_OPT_WARP_TILE;
    let frag_k = WMMA_TF32_OPT_FRAG_K;

    if cfg.block_m == 0 || cfg.block_n == 0 || cfg.k_tile == 0 {
        return Err(invalid(
            "block_m/block_n/k_tile must all be non-zero".to_string(),
        ));
    }
    if !cfg.block_m.is_multiple_of(warp_tile) || !cfg.block_n.is_multiple_of(warp_tile) {
        return Err(invalid(format!(
            "block_m ({}) and block_n ({}) must both be multiples of WARP_TILE ({warp_tile})",
            cfg.block_m, cfg.block_n
        )));
    }
    if !cfg.k_tile.is_multiple_of(frag_k) {
        return Err(invalid(format!(
            "k_tile ({}) must be a multiple of FRAG_K ({frag_k})",
            cfg.k_tile
        )));
    }

    let warp_grid_m = cfg.block_m / warp_tile;
    let warp_grid_n = cfg.block_n / warp_tile;
    let threads = warp_grid_m
        .checked_mul(warp_grid_n)
        .and_then(|w| w.checked_mul(32))
        .ok_or_else(|| invalid("block thread count overflow".to_string()))?;
    if threads > 1024 {
        return Err(invalid(format!(
            "block thread count {threads} exceeds CUDA's per-block limit (1024)"
        )));
    }

    let a_pad = cfg.k_tile + 4;
    let b_pad = cfg.block_n + 4;
    // ダブルバッファ（as_tile/bs_tile）＋エピローグ c_tile の静的共有メモリ
    // 合計（実装計画 4.2 節「SMEM 予算」）。全て同一カーネル関数内の
    // `__shared__` 宣言のためタイミングに関わらず合算される。
    let smem_bytes = 2u32
        .checked_mul(cfg.block_m)
        .and_then(|v| v.checked_mul(a_pad))
        .and_then(|v| v.checked_mul(4))
        .zip(
            2u32.checked_mul(cfg.k_tile)
                .and_then(|v| v.checked_mul(b_pad))
                .and_then(|v| v.checked_mul(4)),
        )
        .and_then(|(a, b)| a.checked_add(b))
        .zip(
            cfg.block_m
                .checked_mul(cfg.block_n)
                .and_then(|v| v.checked_mul(4)),
        )
        .and_then(|(ab, c)| ab.checked_add(c))
        .ok_or_else(|| invalid("shared memory byte count overflow".to_string()))?;
    if smem_bytes > 49_152 {
        return Err(invalid(format!(
            "static shared memory usage {smem_bytes} bytes exceeds the 48KiB per-block limit"
        )));
    }

    for (name, spec) in [
        ("dim_m", cfg.dim_m),
        ("dim_n", cfg.dim_n),
        ("dim_k", cfg.dim_k),
    ] {
        if let DimSpec::Static(0) = spec {
            return Err(invalid(format!(
                "{name} static value must not be zero (degenerate dimension)"
            )));
        }
    }

    Ok(())
}

/// `#define {macro_name} <param_name または static 値>` を 1 行生成する
/// （`kernels_mma::render_dim_define` と同方式。用途がモジュールをまたぐ
/// ためここでも定義する）。
fn render_dim_define(macro_name: &str, param_name: &str, spec: DimSpec) -> String {
    match spec {
        DimSpec::Dynamic => format!("#define {macro_name} {param_name}"),
        DimSpec::Static(value) => format!("#define {macro_name} {value}"),
    }
}

fn render_wmma_tf32_opt_unchecked(cfg: &WmmaOptKernelConfig) -> String {
    let warp_grid_n = cfg.block_n / WMMA_TF32_OPT_WARP_TILE;
    let threads = (cfg.block_m / WMMA_TF32_OPT_WARP_TILE) * warp_grid_n * 32;
    let a_pad = cfg.k_tile + 4;
    let b_pad = cfg.block_n + 4;
    let k_substeps = cfg.k_tile / WMMA_TF32_OPT_FRAG_K;
    let dim_m_define = render_dim_define("DIM_M", "m", cfg.dim_m);
    let dim_n_define = render_dim_define("DIM_N", "n", cfg.dim_n);
    let dim_k_define = render_dim_define("DIM_K", "k", cfg.dim_k);

    format!(
        "\n#include <mma.h>\n\n\
         using namespace nvcuda;\n\n\
         #define WMMA_TF32_OPT_BLOCK_M {block_m}\n\
         #define WMMA_TF32_OPT_BLOCK_N {block_n}\n\
         #define WMMA_TF32_OPT_K_TILE {k_tile}\n\
         #define WMMA_TF32_OPT_FRAG {frag}\n\
         #define WMMA_TF32_OPT_FRAG_K {frag_k}\n\
         #define WMMA_TF32_OPT_WARP_TILE {warp_tile}\n\
         #define WMMA_TF32_OPT_THREADS {threads}\n\
         #define WMMA_TF32_OPT_A_PAD {a_pad}\n\
         #define WMMA_TF32_OPT_B_PAD {b_pad}\n\
         #define WMMA_TF32_OPT_FRAG_ROWS 2\n\
         #define WMMA_TF32_OPT_FRAG_COLS 2\n\
         #define WMMA_TF32_OPT_K_SUBSTEPS {k_substeps}\n\
         #define WMMA_TF32_OPT_WARP_GRID_N {warp_grid_n}\n\
         {dim_m_define}\n\
         {dim_n_define}\n\
         {dim_k_define}\n\
         \n{WMMA_TF32_F32_OPT_BODY}",
        block_m = cfg.block_m,
        block_n = cfg.block_n,
        k_tile = cfg.k_tile,
        frag = WMMA_TF32_OPT_FRAG,
        frag_k = WMMA_TF32_OPT_FRAG_K,
        warp_tile = WMMA_TF32_OPT_WARP_TILE,
    )
}

/// [`WmmaOptKernelConfig`] を TF32 opt カーネルソースへ展開する
/// （イシュー #516）。展開前に [`validate_wmma_tf32_opt_config`] で
/// SMEM 予算・倍数関係・スレッド数上限を fail-closed 検査する。返す
/// [`RenderedWmmaOptKernel`] はソースと展開元 `cfg` を保持し、ホスト側の
/// 将来の非既定構成起動 API は `.source_for_launch(m, n, k)` で起動時形状
/// 検査を経たソース文字列を取得してから `nvrtc::compile_ptx` に渡すこと
/// （`RenderedWmmaOptKernel` ドキュメンテーションコメント参照。検査を経ない
/// 裸の `source()` は公開しないため呼び忘れが型で防がれる）。
///
/// `mod kernels_wmma_opt` が非公開モジュールのため、既定構成のみを使う
/// 現状の `gemm.rs`（[`wmma_tf32_f32_opt_source`] 経由）からは呼ばれず
/// rustc の dead-code 解析が誤検知する。非既定 config を渡す呼び出し元は
/// 後続 #504／#519 で追加される想定のため `#[allow(dead_code)]` を付す
/// （`kernels_mma::render_mma_f16` と同じ判断）。
#[allow(dead_code)]
pub fn render_wmma_tf32_opt(cfg: &WmmaOptKernelConfig) -> Result<RenderedWmmaOptKernel, CudaError> {
    validate_wmma_tf32_opt_config(cfg)?;
    Ok(RenderedWmmaOptKernel {
        source: render_wmma_tf32_opt_unchecked(cfg),
        cfg: *cfg,
    })
}

/// [`render_wmma_tf32_opt_unchecked`]／[`wmma_tf32_f32_opt_source`] が
/// 結合するカーネル本体テンプレート。`m`/`n`/`k`・warp グリッド分割数の
/// ハードコード `2` を `DIM_M`/`DIM_N`/`DIM_K`・`WMMA_TF32_OPT_WARP_GRID_N`
/// マクロへ置き換えてある（`kernels_mma.rs::MMA_F16_BODY` と同方針）。
const WMMA_TF32_F32_OPT_BODY: &str = r#"
extern "C" __global__ void gemm_wmma_tf32_opt(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // ダブルバッファ（cur/nxt）の A/B タイル。バンクコンフリクト回避の
    // ため行幅を K_TILE/BLOCK_N ちょうどではなく +4 要素パディングする
    // （本ファイル冒頭ドキュメントコメント「アライメント」参照）。
    __shared__ __align__(32) float as_tile[2][WMMA_TF32_OPT_BLOCK_M][WMMA_TF32_OPT_A_PAD];
    __shared__ __align__(32) float bs_tile[2][WMMA_TF32_OPT_K_TILE][WMMA_TF32_OPT_B_PAD];

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WMMA_TF32_OPT_WARP_GRID_N;
    const int warp_col = warp_id % WMMA_TF32_OPT_WARP_GRID_N;

    const int block_row_base = blockIdx.y * WMMA_TF32_OPT_BLOCK_M;
    const int block_col_base = blockIdx.x * WMMA_TF32_OPT_BLOCK_N;
    const int warp_row_base = warp_row * WMMA_TF32_OPT_WARP_TILE;
    const int warp_col_base = warp_col * WMMA_TF32_OPT_WARP_TILE;

    // レジスタブロッキング: warp あたり 2x2 = 4 個の accumulator fragment
    // を直接レジスタに保持し、K タイル反復間で使い回す。
    wmma::fragment<wmma::accumulator, WMMA_TF32_OPT_FRAG, WMMA_TF32_OPT_FRAG,
                   WMMA_TF32_OPT_FRAG_K, float> c_frag[WMMA_TF32_OPT_FRAG_ROWS][WMMA_TF32_OPT_FRAG_COLS];
#pragma unroll
    for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
            wmma::fill_fragment(c_frag[fi][fj], 0.0f);
        }
    }

    // 桁溢れしない num_k_tiles 計算（kernels.rs::TILED_F32 と同じ方式）。
    int num_k_tiles = (DIM_K > 0) ? (DIM_K - 1) / WMMA_TF32_OPT_K_TILE + 1 : 0;

    int cur = 0;
    // 初回タイル（t=0）のプリフェッチ。ループ内では「今回の cur を計算
    // しつつ次の nxt をプリフェッチする」構造にするため、t=0 分だけは
    // ループに入る前に用意する。
    if (num_k_tiles > 0) {
        for (int idx = tid; idx < WMMA_TF32_OPT_BLOCK_M * WMMA_TF32_OPT_K_TILE; idx += num_threads) {
            int lr = idx / WMMA_TF32_OPT_K_TILE;
            int lc = idx % WMMA_TF32_OPT_K_TILE;
            int gr = block_row_base + lr;
            int gc = lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            as_tile[cur][lr][lc] = (gr < DIM_M && gc < DIM_K) ? a[gr * DIM_K + gc] : 0.0f;
        }
        for (int idx = tid; idx < WMMA_TF32_OPT_K_TILE * WMMA_TF32_OPT_BLOCK_N; idx += num_threads) {
            int lr = idx / WMMA_TF32_OPT_BLOCK_N;
            int lc = idx % WMMA_TF32_OPT_BLOCK_N;
            int gr = lr;
            int gc = block_col_base + lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            bs_tile[cur][lr][lc] = (gr < DIM_K && gc < DIM_N) ? b[gr * DIM_N + gc] : 0.0f;
        }
    }
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        int nxt = cur ^ 1;

        // 次タイルのプリフェッチ（nxt バッファへ書く。cur バッファの
        // 計算読み出しとは独立したメモリ領域のためレースしない。本ファイル
        // 冒頭ドキュメントコメント「ダブルバッファリング」参照）。
        if (t + 1 < num_k_tiles) {
            int k_base_next = (t + 1) * WMMA_TF32_OPT_K_TILE;
            for (int idx = tid; idx < WMMA_TF32_OPT_BLOCK_M * WMMA_TF32_OPT_K_TILE; idx += num_threads) {
                int lr = idx / WMMA_TF32_OPT_K_TILE;
                int lc = idx % WMMA_TF32_OPT_K_TILE;
                int gr = block_row_base + lr;
                int gc = k_base_next + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                as_tile[nxt][lr][lc] = (gr < DIM_M && gc < DIM_K) ? a[gr * DIM_K + gc] : 0.0f;
            }
            for (int idx = tid; idx < WMMA_TF32_OPT_K_TILE * WMMA_TF32_OPT_BLOCK_N; idx += num_threads) {
                int lr = idx / WMMA_TF32_OPT_BLOCK_N;
                int lc = idx % WMMA_TF32_OPT_BLOCK_N;
                int gr = k_base_next + lr;
                int gc = block_col_base + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                bs_tile[nxt][lr][lc] = (gr < DIM_K && gc < DIM_N) ? b[gr * DIM_N + gc] : 0.0f;
            }
        }

        // cur バッファを用いた計算: K_TILE(16) を fragment K(8) 単位の
        // 2 サブステップに分け、各サブステップで 2x2 fragment（レジスタ
        // ブロッキング）を mma_sync する。
#pragma unroll
        for (int ks = 0; ks < WMMA_TF32_OPT_K_SUBSTEPS; ++ks) {
            int k_off = ks * WMMA_TF32_OPT_FRAG_K;

            wmma::fragment<wmma::matrix_a, WMMA_TF32_OPT_FRAG, WMMA_TF32_OPT_FRAG,
                           WMMA_TF32_OPT_FRAG_K, wmma::precision::tf32, wmma::row_major> a_frag[WMMA_TF32_OPT_FRAG_ROWS];
            wmma::fragment<wmma::matrix_b, WMMA_TF32_OPT_FRAG, WMMA_TF32_OPT_FRAG,
                           WMMA_TF32_OPT_FRAG_K, wmma::precision::tf32, wmma::row_major> b_frag[WMMA_TF32_OPT_FRAG_COLS];

#pragma unroll
            for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
                wmma::load_matrix_sync(
                    a_frag[fi],
                    &as_tile[cur][warp_row_base + fi * WMMA_TF32_OPT_FRAG][k_off],
                    WMMA_TF32_OPT_A_PAD);
#pragma unroll
                for (int e = 0; e < a_frag[fi].num_elements; ++e) {
                    a_frag[fi].x[e] = wmma::__float_to_tf32(a_frag[fi].x[e]);
                }
            }
#pragma unroll
            for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
                wmma::load_matrix_sync(
                    b_frag[fj],
                    &bs_tile[cur][k_off][warp_col_base + fj * WMMA_TF32_OPT_FRAG],
                    WMMA_TF32_OPT_B_PAD);
#pragma unroll
                for (int e = 0; e < b_frag[fj].num_elements; ++e) {
                    b_frag[fj].x[e] = wmma::__float_to_tf32(b_frag[fj].x[e]);
                }
            }

#pragma unroll
            for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
                for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
                    wmma::mma_sync(c_frag[fi][fj], a_frag[fi], b_frag[fj], c_frag[fi][fj]);
                }
            }
        }

        // 今回の cur 読み出し（計算）と今回の nxt 書き込み（プリフェッチ）
        // の両方の完了を待ってから cur/nxt を入れ替える（本ファイル冒頭
        // ドキュメントコメント「ダブルバッファリング」参照）。
        __syncthreads();
        cur = nxt;
    }

    // REQ-8: エピローグ store のガード条件。store_matrix_sync は fragment
    // 全体（16x16）を無条件で書くため、共有メモリへ一旦 store したうえで
    // 要素単位のガード付きコピーによりグローバル C への範囲外書き込みを防ぐ
    // （kernels.rs::WMMA_TF32_F32 エピローグと同方式）。
    __shared__ __align__(32) float c_tile[WMMA_TF32_OPT_BLOCK_M][WMMA_TF32_OPT_BLOCK_N];
#pragma unroll
    for (int fi = 0; fi < WMMA_TF32_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_OPT_FRAG_COLS; ++fj) {
            wmma::store_matrix_sync(
                &c_tile[warp_row_base + fi * WMMA_TF32_OPT_FRAG][warp_col_base + fj * WMMA_TF32_OPT_FRAG],
                c_frag[fi][fj], WMMA_TF32_OPT_BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < WMMA_TF32_OPT_BLOCK_M * WMMA_TF32_OPT_BLOCK_N; idx += num_threads) {
        int lr = idx / WMMA_TF32_OPT_BLOCK_N;
        int lc = idx % WMMA_TF32_OPT_BLOCK_N;
        int gr = block_row_base + lr;
        int gc = block_col_base + lc;
        if (gr < DIM_M && gc < DIM_N) {
            c[gr * DIM_N + gc] = c_tile[lr][lc];
        }
    }
}
"#;

/// f16 opt GEMM のブロックタイル一辺（M・N とも 64。TF32 opt と同じ 2×2
/// warp グリッド・warp あたり fragment 2×2 個構成）。
pub const WMMA_F16_OPT_BLOCK_M: u32 = 64;
pub const WMMA_F16_OPT_BLOCK_N: u32 = 64;

/// f16 opt GEMM の fragment M・N・K 一辺（`m16n16k16` の 16）。f16 fragment
/// は K=16 のため、TF32 opt と異なり共有メモリ K タイル幅とサブステップ
/// 分割が不要（1 ロード = 1 `mma_sync` 入力。`kernels_wmma.rs::WMMA_TILE`
/// と同じ値）。
///
/// Rust 側での実利用は `gemm_wmma.rs::CudaWmmaGemm::new` 内の
/// `const _: () = assert!(...)` のみで、通常の実行時コードパスからは
/// 参照されない。rustc 1.88 系の dead-code 解析はネストした無名 `const _`
/// 内からのみ参照される `pub const` を誤って未使用と判定する（1.92 以降
/// では解消済み。`cargo +1.88.0 clippy` と `cargo +1.92.0 clippy` の実測
/// 差分で確認済み。#149 PR CI 指摘対応）。実行時 `debug_assert` への置換
/// は「CUDA 非搭載の通常 CI では `new` 自体が実行されず検査が効かない」
/// というレビュー指摘 #62 の踏襲事項に反するため行わない。
#[allow(dead_code)]
pub const WMMA_F16_OPT_FRAG: u32 = 16;

/// f16 opt GEMM の共有メモリ K タイル幅（fragment K と同じ 16）。
///
/// [`WMMA_F16_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_F16_OPT_K_TILE: u32 = WMMA_F16_OPT_FRAG;

/// f16 opt GEMM の warp タイル一辺（32。TF32 opt と同じレジスタブロッキング
/// 2×2）。
///
/// [`WMMA_F16_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_F16_OPT_WARP_TILE: u32 = 32;

/// f16 opt GEMM 1 ブロックあたりのスレッド数（4 warp = 128 スレッド。
/// TF32 opt と同じ 2×2 warp グリッド構成）。
pub const WMMA_F16_OPT_THREADS: u32 = 128;

/// A タイル（`as_tile[2][BLOCK_M][A_PAD]`）の行幅（パディング後）。
/// `K_TILE`（16）に 8 要素加算し、half の `ldm` 制約（8 の倍数）を保ちながら
/// バンクコンフリクトを避ける（`kernels_wmma.rs` 冒頭ドキュメントコメント
/// 「ldm 制約」参照）。
///
/// [`WMMA_F16_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_F16_OPT_A_PAD: u32 = WMMA_F16_OPT_K_TILE + 8;

/// B タイル（`bs_tile[2][K_TILE][B_PAD]`）の行幅（パディング後）。
/// `BLOCK_N`（64）に 8 要素加算する。A パディングと同じ根拠。
///
/// [`WMMA_F16_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_F16_OPT_B_PAD: u32 = WMMA_F16_OPT_BLOCK_N + 8;

/// f16 WMMA GEMM の共有メモリ・タイル最適化版（TASK-11.1d・#63）。
/// `kernels_wmma::WMMA_F16`（#61。1 ブロック = 1 warp = fragment 1 個のみ）
/// に対し、ブロックタイル 64×64・warp あたり fragment 2×2 個（レジスタ
/// ブロッキング）・バンクコンフリクト回避パディング・ダブルバッファ
/// リングを適用する。数値契約（f16 入出力・f32 累算）は
/// `kernels_wmma::WMMA_F16` と同一。
///
/// # テンプレート文字列展開（イシュー #516）
///
/// [`WMMA_TF32_F32_OPT`] と同じ方針で [`render_wmma_f16_opt`] のテンプレート
/// 展開結果として得る。f16 opt のカーネル本体は K サブステップループを
/// 持たない（`WMMA_F16_OPT_K_TILE == WMMA_F16_OPT_FRAG` 固定。TF32 opt の
/// `K_SUBSTEPS` に相当する概念がない）ため、[`WmmaOptKernelConfig::k_tile`]
/// は本カーネルでは `WMMA_F16_OPT_FRAG`（16）固定のみを許容する
/// （[`validate_wmma_f16_opt_config`]）。K タイル可変化は本イシューの
/// スコープ外（PR 本文の out-of-scope 引き継ぎ参照）。
pub fn wmma_f16_opt_source() -> &'static str {
    &WMMA_F16_OPT_SOURCE
}

static WMMA_F16_OPT_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_wmma_f16_opt_unchecked(&WmmaOptKernelConfig::default_f16()));

/// [`WmmaOptKernelConfig`] を f16 opt カーネル向けに fail-closed 検証する。
/// `WARP_TILE`=32・`FRAG`=16 は固定ハードウェア形状。本カーネルは K
/// サブステップを持たないため `k_tile` は `FRAG`（16）固定のみを許容する
/// （本 const 上のドキュメンテーションコメント参照）。
fn validate_wmma_f16_opt_config(cfg: &WmmaOptKernelConfig) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };
    let warp_tile = WMMA_F16_OPT_WARP_TILE;
    let frag = WMMA_F16_OPT_FRAG;

    if cfg.block_m == 0 || cfg.block_n == 0 || cfg.k_tile == 0 {
        return Err(invalid(
            "block_m/block_n/k_tile must all be non-zero".to_string(),
        ));
    }
    if !cfg.block_m.is_multiple_of(warp_tile) || !cfg.block_n.is_multiple_of(warp_tile) {
        return Err(invalid(format!(
            "block_m ({}) and block_n ({}) must both be multiples of WARP_TILE ({warp_tile})",
            cfg.block_m, cfg.block_n
        )));
    }
    if cfg.k_tile != frag {
        return Err(invalid(format!(
            "k_tile ({}) must equal FRAG ({frag}); the f16 opt kernel body has no \
             K-substep loop (unlike the TF32 opt kernel), so K_TILE != FRAG is unsupported",
            cfg.k_tile
        )));
    }

    let warp_grid_m = cfg.block_m / warp_tile;
    let warp_grid_n = cfg.block_n / warp_tile;
    let threads = warp_grid_m
        .checked_mul(warp_grid_n)
        .and_then(|w| w.checked_mul(32))
        .ok_or_else(|| invalid("block thread count overflow".to_string()))?;
    if threads > 1024 {
        return Err(invalid(format!(
            "block thread count {threads} exceeds CUDA's per-block limit (1024)"
        )));
    }

    let a_pad = cfg.k_tile + 8;
    let b_pad = cfg.block_n + 8;
    // ダブルバッファ（as_tile/bs_tile。half=2byte）＋エピローグ cs_tile
    // （f32=4byte）の静的共有メモリ合計。
    let smem_bytes = 2u32
        .checked_mul(cfg.block_m)
        .and_then(|v| v.checked_mul(a_pad))
        .and_then(|v| v.checked_mul(2))
        .zip(
            2u32.checked_mul(cfg.k_tile)
                .and_then(|v| v.checked_mul(b_pad))
                .and_then(|v| v.checked_mul(2)),
        )
        .and_then(|(a, b)| a.checked_add(b))
        .zip(
            cfg.block_m
                .checked_mul(cfg.block_n)
                .and_then(|v| v.checked_mul(4)),
        )
        .and_then(|(ab, c)| ab.checked_add(c))
        .ok_or_else(|| invalid("shared memory byte count overflow".to_string()))?;
    if smem_bytes > 49_152 {
        return Err(invalid(format!(
            "static shared memory usage {smem_bytes} bytes exceeds the 48KiB per-block limit"
        )));
    }

    for (name, spec) in [
        ("dim_m", cfg.dim_m),
        ("dim_n", cfg.dim_n),
        ("dim_k", cfg.dim_k),
    ] {
        if let DimSpec::Static(0) = spec {
            return Err(invalid(format!(
                "{name} static value must not be zero (degenerate dimension)"
            )));
        }
    }

    Ok(())
}

fn render_wmma_f16_opt_unchecked(cfg: &WmmaOptKernelConfig) -> String {
    let warp_grid_n = cfg.block_n / WMMA_F16_OPT_WARP_TILE;
    let threads = (cfg.block_m / WMMA_F16_OPT_WARP_TILE) * warp_grid_n * 32;
    let a_pad = cfg.k_tile + 8;
    let b_pad = cfg.block_n + 8;
    let dim_m_define = render_dim_define("DIM_M", "m", cfg.dim_m);
    let dim_n_define = render_dim_define("DIM_N", "n", cfg.dim_n);
    let dim_k_define = render_dim_define("DIM_K", "k", cfg.dim_k);

    format!(
        "\n#include <mma.h>\n#include <cuda_fp16.h>\n\n\
         using namespace nvcuda;\n\n\
         #define WMMA_F16_OPT_BLOCK_M {block_m}\n\
         #define WMMA_F16_OPT_BLOCK_N {block_n}\n\
         #define WMMA_F16_OPT_K_TILE {k_tile}\n\
         #define WMMA_F16_OPT_FRAG {frag}\n\
         #define WMMA_F16_OPT_WARP_TILE {warp_tile}\n\
         #define WMMA_F16_OPT_THREADS {threads}\n\
         #define WMMA_F16_OPT_A_PAD {a_pad}\n\
         #define WMMA_F16_OPT_B_PAD {b_pad}\n\
         #define WMMA_F16_OPT_FRAG_ROWS 2\n\
         #define WMMA_F16_OPT_FRAG_COLS 2\n\
         #define WMMA_F16_OPT_WARP_GRID_N {warp_grid_n}\n\
         {dim_m_define}\n\
         {dim_n_define}\n\
         {dim_k_define}\n\
         \n{WMMA_F16_OPT_BODY}",
        block_m = cfg.block_m,
        block_n = cfg.block_n,
        k_tile = cfg.k_tile,
        frag = WMMA_F16_OPT_FRAG,
        warp_tile = WMMA_F16_OPT_WARP_TILE,
    )
}

/// [`WmmaOptKernelConfig`] を f16 opt カーネルソースへ展開する
/// （イシュー #516）。展開前に [`validate_wmma_f16_opt_config`] で不変
/// 条件を fail-closed 検査する。返す [`RenderedWmmaOptKernel`] の起動前
/// 検査契約は [`render_wmma_tf32_opt`] と同じ。`#[allow(dead_code)]` の
/// 理由も [`render_wmma_tf32_opt`] と同じ（非公開モジュール・現状は既定
/// 構成のみ消費・後続 #504／#519 が非既定 config の呼び出し元となる想定）。
#[allow(dead_code)]
pub fn render_wmma_f16_opt(cfg: &WmmaOptKernelConfig) -> Result<RenderedWmmaOptKernel, CudaError> {
    validate_wmma_f16_opt_config(cfg)?;
    Ok(RenderedWmmaOptKernel {
        source: render_wmma_f16_opt_unchecked(cfg),
        cfg: *cfg,
    })
}

/// [`render_wmma_f16_opt_unchecked`]／[`wmma_f16_opt_source`] が結合する
/// カーネル本体テンプレート（`WMMA_TF32_F32_OPT_BODY` と同方針の
/// `DIM_M`/`DIM_N`/`DIM_K`・`WMMA_F16_OPT_WARP_GRID_N` マクロ化）。
const WMMA_F16_OPT_BODY: &str = r#"
extern "C" __global__ void gemm_wmma_f16_opt(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m, int n, int k)
{
    __shared__ __align__(32) __half as_tile[2][WMMA_F16_OPT_BLOCK_M][WMMA_F16_OPT_A_PAD];
    __shared__ __align__(32) __half bs_tile[2][WMMA_F16_OPT_K_TILE][WMMA_F16_OPT_B_PAD];

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WMMA_F16_OPT_WARP_GRID_N;
    const int warp_col = warp_id % WMMA_F16_OPT_WARP_GRID_N;

    const int block_row_base = blockIdx.y * WMMA_F16_OPT_BLOCK_M;
    const int block_col_base = blockIdx.x * WMMA_F16_OPT_BLOCK_N;
    const int warp_row_base = warp_row * WMMA_F16_OPT_WARP_TILE;
    const int warp_col_base = warp_col * WMMA_F16_OPT_WARP_TILE;

    wmma::fragment<wmma::accumulator, WMMA_F16_OPT_FRAG, WMMA_F16_OPT_FRAG,
                   WMMA_F16_OPT_FRAG, float> c_frag[WMMA_F16_OPT_FRAG_ROWS][WMMA_F16_OPT_FRAG_COLS];
#pragma unroll
    for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
            wmma::fill_fragment(c_frag[fi][fj], 0.0f);
        }
    }

    // 桁溢れしない num_k_tiles 計算（kernels.rs::TILED_F32 と同じ方式）。
    int num_k_tiles = (DIM_K > 0) ? (DIM_K - 1) / WMMA_F16_OPT_K_TILE + 1 : 0;

    int cur = 0;
    if (num_k_tiles > 0) {
        for (int idx = tid; idx < WMMA_F16_OPT_BLOCK_M * WMMA_F16_OPT_K_TILE; idx += num_threads) {
            int lr = idx / WMMA_F16_OPT_K_TILE;
            int lc = idx % WMMA_F16_OPT_K_TILE;
            int gr = block_row_base + lr;
            int gc = lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            as_tile[cur][lr][lc] = (gr < DIM_M && gc < DIM_K) ? a[gr * DIM_K + gc] : __float2half(0.0f);
        }
        for (int idx = tid; idx < WMMA_F16_OPT_K_TILE * WMMA_F16_OPT_BLOCK_N; idx += num_threads) {
            int lr = idx / WMMA_F16_OPT_BLOCK_N;
            int lc = idx % WMMA_F16_OPT_BLOCK_N;
            int gr = lr;
            int gc = block_col_base + lc;
            // REQ-8: guarded load（範囲外はゼロ充填）。
            bs_tile[cur][lr][lc] = (gr < DIM_K && gc < DIM_N) ? b[gr * DIM_N + gc] : __float2half(0.0f);
        }
    }
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        int nxt = cur ^ 1;

        // 次タイルのプリフェッチ。kernels_wmma_opt.rs::WMMA_TF32_F32_OPT
        // と同じダブルバッファ契約（本ファイルの Rust 側ドキュメンテーション
        // コメント「ダブルバッファリング」参照）。
        if (t + 1 < num_k_tiles) {
            int k_base_next = (t + 1) * WMMA_F16_OPT_K_TILE;
            for (int idx = tid; idx < WMMA_F16_OPT_BLOCK_M * WMMA_F16_OPT_K_TILE; idx += num_threads) {
                int lr = idx / WMMA_F16_OPT_K_TILE;
                int lc = idx % WMMA_F16_OPT_K_TILE;
                int gr = block_row_base + lr;
                int gc = k_base_next + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                as_tile[nxt][lr][lc] = (gr < DIM_M && gc < DIM_K) ? a[gr * DIM_K + gc] : __float2half(0.0f);
            }
            for (int idx = tid; idx < WMMA_F16_OPT_K_TILE * WMMA_F16_OPT_BLOCK_N; idx += num_threads) {
                int lr = idx / WMMA_F16_OPT_BLOCK_N;
                int lc = idx % WMMA_F16_OPT_BLOCK_N;
                int gr = k_base_next + lr;
                int gc = block_col_base + lc;
                // REQ-8: guarded load（範囲外はゼロ充填）。
                bs_tile[nxt][lr][lc] = (gr < DIM_K && gc < DIM_N) ? b[gr * DIM_N + gc] : __float2half(0.0f);
            }
        }

        wmma::fragment<wmma::matrix_a, WMMA_F16_OPT_FRAG, WMMA_F16_OPT_FRAG,
                       WMMA_F16_OPT_FRAG, __half, wmma::row_major> a_frag[WMMA_F16_OPT_FRAG_ROWS];
        wmma::fragment<wmma::matrix_b, WMMA_F16_OPT_FRAG, WMMA_F16_OPT_FRAG,
                       WMMA_F16_OPT_FRAG, __half, wmma::row_major> b_frag[WMMA_F16_OPT_FRAG_COLS];

#pragma unroll
        for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
            wmma::load_matrix_sync(
                a_frag[fi],
                &as_tile[cur][warp_row_base + fi * WMMA_F16_OPT_FRAG][0],
                WMMA_F16_OPT_A_PAD);
        }
#pragma unroll
        for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
            wmma::load_matrix_sync(
                b_frag[fj],
                &bs_tile[cur][0][warp_col_base + fj * WMMA_F16_OPT_FRAG],
                WMMA_F16_OPT_B_PAD);
        }

#pragma unroll
        for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
            for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
                wmma::mma_sync(c_frag[fi][fj], a_frag[fi], b_frag[fj], c_frag[fi][fj]);
            }
        }

        // 今回の cur 読み出し（計算）と今回の nxt 書き込み（プリフェッチ）
        // の両方の完了を待ってから cur/nxt を入れ替える。
        __syncthreads();
        cur = nxt;
    }

    // REQ-8: エピローグ store のガード条件（kernels_wmma.rs::WMMA_F16
    // エピローグと同方式）。
    __shared__ __align__(32) float cs_tile[WMMA_F16_OPT_BLOCK_M][WMMA_F16_OPT_BLOCK_N];
#pragma unroll
    for (int fi = 0; fi < WMMA_F16_OPT_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_F16_OPT_FRAG_COLS; ++fj) {
            wmma::store_matrix_sync(
                &cs_tile[warp_row_base + fi * WMMA_F16_OPT_FRAG][warp_col_base + fj * WMMA_F16_OPT_FRAG],
                c_frag[fi][fj], WMMA_F16_OPT_BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < WMMA_F16_OPT_BLOCK_M * WMMA_F16_OPT_BLOCK_N; idx += num_threads) {
        int lr = idx / WMMA_F16_OPT_BLOCK_N;
        int lc = idx % WMMA_F16_OPT_BLOCK_N;
        int gr = block_row_base + lr;
        int gc = block_col_base + lc;
        if (gr < DIM_M && gc < DIM_N) {
            c[gr * DIM_N + gc] = __float2half(cs_tile[lr][lc]);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// `WMMA_TF32_OPT_*`（Rust 側の「唯一の真実源」）が既定構成の生成
    /// ソース内 `#define` と食い違わないことを検査する
    /// （`kernels.rs::wmma_tf32_constants_match_kernel_source_defines` と
    /// 同じ方式）。イシュー #516 でテンプレート展開へ移行したため、静的
    /// リテラルではなく `wmma_tf32_f32_opt_source()` を対象にする。
    #[test]
    fn wmma_tf32_opt_constants_match_kernel_source_defines() {
        let src = wmma_tf32_f32_opt_source();
        let checks = [
            ("WMMA_TF32_OPT_BLOCK_M", WMMA_TF32_OPT_BLOCK_M),
            ("WMMA_TF32_OPT_BLOCK_N", WMMA_TF32_OPT_BLOCK_N),
            ("WMMA_TF32_OPT_K_TILE", WMMA_TF32_OPT_K_TILE),
            ("WMMA_TF32_OPT_FRAG", WMMA_TF32_OPT_FRAG),
            ("WMMA_TF32_OPT_FRAG_K", WMMA_TF32_OPT_FRAG_K),
            ("WMMA_TF32_OPT_WARP_TILE", WMMA_TF32_OPT_WARP_TILE),
            ("WMMA_TF32_OPT_THREADS", WMMA_TF32_OPT_THREADS),
            ("WMMA_TF32_OPT_A_PAD", WMMA_TF32_OPT_A_PAD),
            ("WMMA_TF32_OPT_B_PAD", WMMA_TF32_OPT_B_PAD),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                src.contains(&expected),
                "wmma_tf32_f32_opt_source() の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
        }
    }

    /// `WMMA_F16_OPT_*`（Rust 側の「唯一の真実源」）が既定構成の生成
    /// ソース内 `#define` と食い違わないことを検査する。
    #[test]
    fn wmma_f16_opt_constants_match_kernel_source_defines() {
        let src = wmma_f16_opt_source();
        let checks = [
            ("WMMA_F16_OPT_BLOCK_M", WMMA_F16_OPT_BLOCK_M),
            ("WMMA_F16_OPT_BLOCK_N", WMMA_F16_OPT_BLOCK_N),
            ("WMMA_F16_OPT_K_TILE", WMMA_F16_OPT_K_TILE),
            ("WMMA_F16_OPT_FRAG", WMMA_F16_OPT_FRAG),
            ("WMMA_F16_OPT_WARP_TILE", WMMA_F16_OPT_WARP_TILE),
            ("WMMA_F16_OPT_THREADS", WMMA_F16_OPT_THREADS),
            ("WMMA_F16_OPT_A_PAD", WMMA_F16_OPT_A_PAD),
            ("WMMA_F16_OPT_B_PAD", WMMA_F16_OPT_B_PAD),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                src.contains(&expected),
                "wmma_f16_opt_source() の `#define {name}` が Rust 側の定数（{value}）と一致しません"
            );
        }
    }

    /// 既定構成（全次元 `Dynamic`）では `#define DIM_* <カーネル引数>`
    /// 形式でカーネル引数へ間接するのみで、既存カーネルとプリプロセス後
    /// 等価であることをロックする（`kernels_mma.rs` と同方針）。
    #[test]
    fn wmma_opt_default_config_dim_defines_alias_kernel_parameters() {
        for src in [wmma_tf32_f32_opt_source(), wmma_f16_opt_source()] {
            for expected in ["#define DIM_M m", "#define DIM_N n", "#define DIM_K k"] {
                assert!(
                    src.contains(expected),
                    "既定次元マクロ `{expected}` が見つかりません: {src}"
                );
            }
        }
    }

    /// TASK-11.3（tensor core 命令使用の証跡）を兼ねる。
    /// `kernels_wmma.rs::wmma_f16_source_uses_wmma_instructions` と同方式。
    #[test]
    fn wmma_tf32_opt_source_uses_wmma_instructions() {
        let src = wmma_tf32_f32_opt_source();
        for needle in [
            "#include <mma.h>",
            "wmma::fragment",
            "wmma::load_matrix_sync",
            "wmma::mma_sync",
            "wmma::store_matrix_sync",
            "wmma::fill_fragment",
            "wmma::__float_to_tf32",
        ] {
            assert!(
                src.contains(needle),
                "wmma_tf32_f32_opt_source() に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    #[test]
    fn wmma_f16_opt_source_uses_wmma_instructions() {
        let src = wmma_f16_opt_source();
        for needle in [
            "#include <mma.h>",
            "wmma::fragment",
            "wmma::load_matrix_sync",
            "wmma::mma_sync",
            "wmma::store_matrix_sync",
            "wmma::fill_fragment",
        ] {
            assert!(
                src.contains(needle),
                "wmma_f16_opt_source() に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// REQ-8: guarded load／guarded store がソースから除去されていない
    /// ことをロックする（`kernels_wmma.rs::wmma_f16_source_retains_req8_boundary_guards`
    /// と同方式）。needle は `DIM_*` マクロ化後の形式（イシュー #516）。
    #[test]
    fn wmma_tf32_opt_source_retains_req8_boundary_guards() {
        let src = wmma_tf32_f32_opt_source();
        for needle in [
            "gr < DIM_M && gc < DIM_K",
            "gr < DIM_K && gc < DIM_N",
            "gr < DIM_M && gc < DIM_N",
        ] {
            assert!(
                src.contains(needle),
                "wmma_tf32_f32_opt_source() に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    #[test]
    fn wmma_f16_opt_source_retains_req8_boundary_guards() {
        let src = wmma_f16_opt_source();
        for needle in [
            "gr < DIM_M && gc < DIM_K",
            "gr < DIM_K && gc < DIM_N",
            "gr < DIM_M && gc < DIM_N",
        ] {
            assert!(
                src.contains(needle),
                "wmma_f16_opt_source() に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    /// ダブルバッファリング（`__syncthreads()` ベースの 2 段パイプライン）
    /// がソースから除去されていないことをロックする。プリフェッチ分岐
    /// （`t + 1 < num_k_tiles`）と `cur`/`nxt` の入れ替えは #63 の受け入れ
    /// 条件（tiled 実装超過）を支える主要最適化のため、実装が回帰的に
    /// 単純化（例: ダブルバッファ除去）された場合に検出する。
    #[test]
    fn wmma_opt_sources_retain_double_buffering() {
        for src in [wmma_tf32_f32_opt_source(), wmma_f16_opt_source()] {
            assert!(
                src.contains("cur ^ 1"),
                "ダブルバッファの cur/nxt 切替が見つかりません"
            );
            assert!(
                src.contains("t + 1 < num_k_tiles"),
                "ダブルバッファのプリフェッチ分岐が見つかりません"
            );
        }
    }

    /// 非既定 config（block_m/block_n=128（2×4 warp グリッド）・K 次元を
    /// `Static(4096)` で焼き込み）での TF32 opt 特化 render を検査する
    /// （実装計画 7 節「特化 render」）。
    #[test]
    fn render_wmma_tf32_opt_specializes_tile_and_static_dim() {
        let cfg = WmmaOptKernelConfig {
            block_m: 64,
            block_n: 96,
            k_tile: 16,
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Static(4096),
        };
        let rendered = render_wmma_tf32_opt(&cfg).expect("有効な構成が拒否されました");
        // dim_m/dim_n=Dynamic・dim_k=Static(4096) のため、実起動形状は
        // k=4096 固定・m/n は任意（後段の validate_launch_shape 呼び出しと
        // 揃えて m=64・n=96 を使う）。
        let src = rendered
            .source_for_launch(64, 96, 4096)
            .expect("有効な起動形状が拒否されました");

        for expected in [
            "#define WMMA_TF32_OPT_BLOCK_M 64",
            "#define WMMA_TF32_OPT_BLOCK_N 96",
            "#define WMMA_TF32_OPT_WARP_GRID_N 3", // 96 / WARP_TILE(32)
            "#define DIM_K 4096",
            "#define DIM_M m",
        ] {
            assert!(
                src.contains(expected),
                "特化 render に `{expected}` が見つかりません: config={cfg:?}"
            );
        }
        assert!(src.contains("gr < DIM_M && gc < DIM_K"));

        // dim_k=Static(4096) のため、実際の起動形状 k=16（コンパイル時に
        // 焼き込んだ値と食い違う）は
        // RenderedWmmaOptKernel::validate_launch_shape で fail-closed に
        // 拒否されなければならない（PR #643 codex-review P1 指摘への対応）。
        assert!(rendered.validate_launch_shape(64, 96, 4096).is_ok());
        assert!(matches!(
            rendered.validate_launch_shape(64, 96, 16),
            Err(CudaError::InvalidKernelConfig { .. })
        ));
    }

    /// フェイルクローズド検証（TF32 opt）: SMEM 予算超過・倍数違反・
    /// ゼロ次元が全て `Err(CudaError::InvalidKernelConfig)` になることを
    /// 検査する。
    #[test]
    fn render_wmma_tf32_opt_rejects_invalid_configs() {
        let base = WmmaOptKernelConfig::default_tf32();
        let cases: [(&str, WmmaOptKernelConfig); 3] = [
            (
                "block_m not multiple of WARP_TILE",
                WmmaOptKernelConfig {
                    block_m: 50,
                    ..base
                },
            ),
            (
                "smem budget exceeded",
                WmmaOptKernelConfig {
                    block_m: 256,
                    block_n: 256,
                    ..base
                },
            ),
            (
                "static dim zero",
                WmmaOptKernelConfig {
                    dim_n: DimSpec::Static(0),
                    ..base
                },
            ),
        ];
        for (label, cfg) in cases {
            let result = render_wmma_tf32_opt(&cfg);
            assert!(
                matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
                "{label} は InvalidKernelConfig で拒否されるべきです: config={cfg:?}"
            );
        }
    }

    /// フェイルクローズド検証（f16 opt）: `k_tile != FRAG` が本カーネル
    /// 固有の制約（K サブステップ非対応）として拒否されることを検査する。
    #[test]
    fn render_wmma_f16_opt_rejects_k_tile_mismatch() {
        let cfg = WmmaOptKernelConfig {
            k_tile: 32,
            ..WmmaOptKernelConfig::default_f16()
        };
        assert!(matches!(
            render_wmma_f16_opt(&cfg),
            Err(CudaError::InvalidKernelConfig { .. })
        ));
    }

    /// [`WmmaOptKernelConfig::validate_launch_shape`] が dim_m/dim_n/dim_k
    /// のいずれか 1 つでも実 shape と食い違えば拒否することを検査する
    /// （PR #643 codex-review P1 指摘への対応。指摘箇所
    /// `kernels_wmma_opt.rs:322`（TF32 opt 検証器）・`:725`（f16 opt 検証器）
    /// 双方を `default_tf32`／`default_f16` それぞれで検査する。
    /// `kernels_mma::tests::mma_kernel_config_validate_launch_shape_rejects_k_mismatch`
    /// と同じ設計）。
    #[test]
    fn wmma_opt_config_validate_launch_shape_rejects_k_mismatch() {
        for base in [
            WmmaOptKernelConfig::default_tf32(),
            WmmaOptKernelConfig::default_f16(),
        ] {
            let cfg = WmmaOptKernelConfig {
                dim_k: DimSpec::Static(4096),
                ..base
            };

            assert!(cfg.validate_launch_shape(128, 128, 4096).is_ok());

            let result = cfg.validate_launch_shape(128, 128, 16);
            assert!(
                matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
                "dim_k=Static(4096) の関数を実際は K=16 で起動しようとする場合は拒否されるべきです: base={base:?}, result={result:?}"
            );
        }
    }

    /// 受け入れ基準 2（PTX/SASS ダンプによるコンパイル時展開の実確認）は
    /// CI・本環境に NVRTC／実機がないため通常 CI では実行しない
    /// （`kernels_mma.rs::tests::mma_f16_sources_compile_with_nvrtc_when_available`
    /// と同方針）。実測記録は #531/#534/#539 へ引き継ぐ。
    #[test]
    #[ignore = "requires NVRTC (libnvrtc); run manually on a CUDA-enabled host"]
    fn wmma_opt_sources_compile_with_nvrtc_when_available() {
        use crate::nvrtc::compile_ptx;

        let arch = "compute_80";
        for src in [wmma_tf32_f32_opt_source(), wmma_f16_opt_source()] {
            match compile_ptx(src, arch) {
                Ok(_) => {}
                Err(CudaError::NvrtcUnavailable { .. }) => return,
                Err(e) => panic!("既定構成カーネルソースの NVRTC コンパイルに失敗しました: {e}"),
            }
        }
    }
}
