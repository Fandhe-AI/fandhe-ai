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
//! （標準的な 2 段パイプラインの契約。`WMMA_TF32_F32_OPT_BODY`／
//! `WMMA_F16_OPT_BODY` は `cp.async` 等の非同期コピー命令は使わず
//! `__syncthreads()` ベースに限定する。#187 のスコープ外事項）。
//!
//! **イシュー #500 での追記**: 上記の「`cp.async` 不使用」は
//! `WMMA_TF32_F32_OPT_BODY`／`WMMA_F16_OPT_BODY` 2 経路に閉じた制約であり、
//! 本モジュール全体の制約ではない。TF32 opt-staged 経路
//! （`WMMA_TF32_F32_STAGED_BODY`。本ファイル該当節参照）は
//! `kernels_mma.rs::MMA_F16_BODY`（Phase B・#492〜#499）で確立した
//! `cp.async` 多段パイプライン・fragment 先読みダブルバッファを WMMA C++
//! API（`nvcuda::wmma`）の TF32 経路へ横展開したものであり、`cp.async` を
//! 使用する。既存 2 経路（`__syncthreads()` ベース）は変更せず併存させる
//! （`gemm.rs::run_wmma_tf32` の cp.async 16 バイト整列非対応形状の
//! フォールバック先として温存する）。
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

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::error::CudaError;
use crate::kernels_mma::{DimSpec, MMA_STATIC_SMEM_LIMIT_BYTES, render_dim_define};
use crate::nvrtc::MAX_PIPELINE_STAGES;

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
/// が 1 回だけキャッシュして返す。`kernels_mma.rs::mma_f16_source()` と同じ方針
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

    /// `m`/`n` から実際の起動グリッドを構築する（`gemm_wmma.rs::wmma_opt_launch_config`
    /// と同じ「ブロックタイル `block_m x block_n` を `div_ceil` で敷き詰める」
    /// 設計を、既定タイル定数ではなく本 `cfg` のタイル値〈`self.block_m`/
    /// `self.block_n`〉に一般化したもの。[`CompiledWmmaOptKernel::launch_f16`]／
    /// [`CompiledWmmaOptKernel::launch_tf32`] が本メソッドの戻り値を内部
    /// でのみ使い、呼び出し元へは一切公開しないことで、検証済み `m`/`n`/`k`
    /// とは無関係な grid/block 設定を持ち込んで起動する経路を型で塞ぐ
    /// （`kernels_mma::MmaKernelConfig::launch_config` と同じ設計）。
    ///
    /// warp タイル辺（32）は TF32 opt（`WMMA_TF32_OPT_WARP_TILE`）・f16 opt
    /// （`WMMA_F16_OPT_WARP_TILE`）のいずれも同一のハードウェア形状固定値
    /// （両定数の値は 32 で一致。`validate_wmma_tf32_opt_config`／
    /// `validate_wmma_f16_opt_config` が `block_m`/`block_n` をこの倍数と
    /// して検査済み）であり、`WmmaOptKernelConfig` は dtype を持たない
    /// （dtype 差異は呼び出す `render_*` 関数の選択で表現する設計。本構造体
    /// ドキュメンテーションコメント参照）ため、本メソッドはどちらの dtype
    /// でも共通のグリッド／ブロック計算として使える。
    ///
    /// 静的共有メモリ（`__shared__` 配列としてカーネルソースへ焼き込み
    /// 済み）のみを使う設計のため `shared_mem_bytes` は常に 0
    /// （`gemm_wmma.rs::wmma_opt_launch_config` と同じ契約）。
    #[allow(dead_code)] // 理由は validate_launch_shape と同じ（非公開モジュール）
    pub fn launch_config(&self, m: u32, n: u32) -> LaunchConfig {
        // TF32 opt・f16 opt 共通のハードウェア warp タイル辺（両定数とも
        // 32 で一致。値の根拠は本メソッドのドキュメンテーションコメント
        // 参照）。下記コンパイル時アサーションで
        // `WMMA_TF32_OPT_WARP_TILE`/`WMMA_F16_OPT_WARP_TILE` との一致を
        // 保証する（レビュー指摘: 将来どちらか一方の定数値のみが変更
        // された場合に `validate_wmma_*_opt_config` 側は正しい定数で
        // 倍数関係を検査する一方、本メソッドはハードコードされた 32 の
        // ままとなり両者が無言で食い違う経路を塞ぐ）。
        const WARP_TILE: u32 = 32;
        const _: () = assert!(
            WARP_TILE == WMMA_TF32_OPT_WARP_TILE && WARP_TILE == WMMA_F16_OPT_WARP_TILE,
            "WARP_TILE は WMMA_TF32_OPT_WARP_TILE/WMMA_F16_OPT_WARP_TILE と一致している必要があります"
        );
        let warp_grid_m = self.block_m / WARP_TILE;
        let warp_grid_n = self.block_n / WARP_TILE;
        LaunchConfig {
            grid_dim: (n.div_ceil(self.block_n), m.div_ceil(self.block_m), 1),
            block_dim: (warp_grid_m * warp_grid_n * 32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// [`render_wmma_tf32_opt`] が返す、展開済み TF32 opt カーネルソースと
/// 展開元 [`WmmaOptKernelConfig`] を 1 個にまとめた descriptor（PR #643
/// codex-review P1 指摘への対応。`kernels_mma::RenderedMmaKernel` と同じ
/// 設計）。
///
/// PR #643 codex-review 再々々々々指摘（P0。`WmmaOptKernelConfig` が
/// dtype を保持しないため、単一の `RenderedWmmaOptKernel`/
/// `CompiledWmmaOptKernel` 型に TF32・f16 双方の `compile`/`launch_f16`/
/// `launch_tf32` を持たせると、TF32 用にコンパイルした `CudaFunction` を
/// `launch_f16` で（またはその逆で）起動する誤りを型で防げなかった）
/// への対応として、TF32 経路専用の本型と f16 経路専用の
/// [`RenderedWmmaF16OptKernel`] へ分離した。dtype ごとに別の Rust 型が
/// 存在するため、[`Self::compile`] は TF32 用エントリポイント
/// `"gemm_wmma_tf32_opt"` のみを、対応する [`CompiledWmmaTf32OptKernel`]
/// は `launch_tf32` のみを公開し、f16 側のメソッドは型として存在しない
/// （呼び出し元が dtype を取り違えてもコンパイルが通らない）。
///
/// フィールドは非公開。生ソースを `&str`/`String` として外へ返す公開
/// メソッドは一切持たない（`kernels_mma::RenderedMmaKernel` と同じ理由。
/// ソースの受け渡し先を [`Self::compile`] 内部に限定する）。
///
/// `mod kernels_wmma_opt` が非公開モジュールのため、既定構成のみ消費する
/// 現状の呼び出し元からは本構造体・以下の全メソッドが呼ばれず dead-code
/// 解析が誤検知する。`#[allow(dead_code)]` の理由は [`render_wmma_tf32_opt`]
/// と同じ。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedWmmaTf32OptKernel {
    source: String,
    cfg: WmmaOptKernelConfig,
}

impl RenderedWmmaTf32OptKernel {
    /// カーネルソースを NVRTC コンパイル → 固定エントリポイント
    /// `"gemm_wmma_tf32_opt"` のロードまで descriptor 内部で完結させ、
    /// 結果（`CudaFunction`）と展開元 `cfg` を不可分に束ねた
    /// [`CompiledWmmaTf32OptKernel`] を返す唯一の公開経路
    /// （`kernels_mma::RenderedMmaKernel::compile` と同じ設計。PR #643
    /// codex-review 再々々々々指摘〈P0〉への対応。`RenderedWmmaTf32OptKernel`
    /// ドキュメンテーションコメント参照）。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledWmmaTf32OptKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_tf32_opt")?;
        Ok(CompiledWmmaTf32OptKernel {
            func,
            cfg: self.cfg,
        })
    }

    /// テスト専用のソース内容検査アクセサ（`#[cfg(test)]` のためリリース
    /// ビルドには存在しない。`kernels_mma::RenderedMmaKernel::source` と
    /// 同じ理由）。
    #[cfg(test)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// [`render_wmma_f16_opt`] が返す、展開済み f16 opt カーネルソースと
/// 展開元 [`WmmaOptKernelConfig`] を 1 個にまとめた descriptor
/// （[`RenderedWmmaTf32OptKernel`] と対になる f16 専用型。分離理由は同型の
/// ドキュメンテーションコメント参照）。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedWmmaF16OptKernel {
    source: String,
    cfg: WmmaOptKernelConfig,
}

impl RenderedWmmaF16OptKernel {
    /// カーネルソースを NVRTC コンパイル → 固定エントリポイント
    /// `"gemm_wmma_f16_opt"` のロードまで descriptor 内部で完結させ、
    /// 結果（`CudaFunction`）と展開元 `cfg` を不可分に束ねた
    /// [`CompiledWmmaF16OptKernel`] を返す唯一の公開経路
    /// （[`RenderedWmmaTf32OptKernel::compile`] と対称。エントリポイント名
    /// のみが異なる）。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledWmmaF16OptKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_f16_opt")?;
        Ok(CompiledWmmaF16OptKernel {
            func,
            cfg: self.cfg,
        })
    }

    /// テスト専用のソース内容検査アクセサ（[`RenderedWmmaTf32OptKernel::source`]
    /// と同じ理由）。
    #[cfg(test)]
    #[allow(dead_code)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// [`RenderedWmmaTf32OptKernel::compile`] が返す、TF32 opt カーネル専用の
/// コンパイル済み `CudaFunction` と展開元 [`WmmaOptKernelConfig`] を
/// 不可分に束ねた descriptor（PR #643 codex-review 再々々々々指摘〈P0〉
/// への対応。`kernels_mma::CompiledMmaKernel` と同じ設計）。
///
/// フィールドは非公開。`func` を取り出す・あるいは検証を経ずに起動できる
/// 公開経路は一切存在しない。唯一の起動経路 [`Self::launch_tf32`] は、
/// 検証済み shape 由来の grid/block・引数以外での起動が構造的に不可能な
/// 設計（`kernels_mma::CompiledMmaKernel::launch_f16` と同型）。本型は
/// TF32 専用（[`RenderedWmmaTf32OptKernel::compile`] のみが構築できる）
/// なので `launch_f16` は型として存在せず、f16 用にコンパイルした
/// `CudaFunction` を TF32 起動 API で（またはその逆で）呼ぶ経路は
/// コンパイルエラーになる。
#[allow(dead_code)]
pub struct CompiledWmmaTf32OptKernel {
    func: cudarc::driver::CudaFunction,
    cfg: WmmaOptKernelConfig,
}

impl CompiledWmmaTf32OptKernel {
    /// 検証済み shape でのみ起動できる、TF32 WMMA opt カーネルの
    /// `CudaFunction` へアクセスする唯一の公開経路（`CompiledWmmaTf32OptKernel`
    /// ドキュメンテーションコメント参照）。手順は
    /// `kernels_mma::CompiledMmaKernel::launch_f16` と同じ: (1)
    /// [`WmmaOptKernelConfig::validate_launch_shape`]、(2)
    /// `gemm.rs::validate_gemm_dims`／`validate_output_len`、(3)
    /// [`Self::validate_grid_bounds`]、を経てから初めて
    /// [`WmmaOptKernelConfig::launch_config`] が導出した `LaunchConfig` で
    /// `self.func` を起動する。バッファ型は `f32`（TF32 は f32 表現上の
    /// テンソルコア丸めであり、ホスト側バッファは f32 のまま。
    /// `gemm.rs::CudaGemm::launch_wmma_tf32` と同じ契約）。
    /// no-op 形状（`m==0 || n==0`）は 0 次元 grid のドライバ拒否を避けるため
    /// カーネル起動前に早期 return する（`kernels_mma::CompiledMmaKernel::launch_f16`
    /// と同じ根拠・同じ位置づけ。PR #643 codex-review P2 指摘への対応）。
    /// この経路では `c_dev` を一切書き換えない（`validate_output_len` が
    /// `c_dev.len() == m*n == 0` を既に保証しているため要素自体が存在せず、
    /// 「ゼロ初期化して返す」責務を持たない契約。`kernels_mma::CompiledMmaKernel::launch_f16`
    /// ドキュメンテーションコメントと同じ理由。`k==0`〈`m`/`n` は非ゼロ〉は
    /// 本 no-op 判定の対象外で、呼び出し元側 API の `k==0` 早期 return が
    /// 別途担う責務）。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_tf32(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        self.validate_grid_bounds(m)?;
        self.validate_k_tile_bound(k)?;

        let launch_config = self.cfg.launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: カーネル引数は a_dev/b_dev/c_dev（上記
        // validate_gemm_dims/validate_output_len で m*k/k*n/m*n 要素の
        // 確保済みデバイスバッファであることを検証済み）と m_i/n_i/k_i の
        // 5 個・型・個数が、検証済みの m/n/k と 1:1 対応する。grid/block は
        // 同じく検証済みの m/n から launch_config が導出したもののみを
        // 使い、呼び出し元が独自に構築した LaunchConfig を持ち込む経路は
        // 存在しない。カーネル内の手動境界チェック（REQ-8。基本版は
        // kernels_wmma.rs、opt 版は本ファイルの guarded load/store）と
        // 合わせて OOB 読み書きが起きない根拠とする。共有メモリは静的
        // `__shared__` 配列のみを使用するため shared_mem_bytes は 0 の
        // ままでよい。`self.func` は本型のコンストラクタである
        // `RenderedWmmaTf32OptKernel::compile` が `"gemm_wmma_tf32_opt"`
        // 固定名でロードしたものであり、f32 引数と TF32 カーネルシグネチャ
        // の対応は型で保証される。
        unsafe {
            stream
                .launch_builder(&self.func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(launch_config)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_*`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// grid_dim.y（`m.div_ceil(self.cfg.block_m)`）が CUDA の grid y/z
    /// 上限（65,535。全 compute capability 共通）を超えないことを検証する
    /// （`kernels_mma::CompiledMmaKernel::validate_grid_bounds` と同じ理由。
    /// 超過するとホスト側の他の検証はすべて通過した上で、ドライバの
    /// カーネル起動が失敗する）。
    fn validate_grid_bounds(&self, m: u32) -> Result<(), CudaError> {
        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.cfg.block_m);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "wmma opt path grid_dim.y (m.div_ceil(block_m)={grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (block_m={}); m={m} is \
                     too large",
                    self.cfg.block_m
                ),
            });
        }
        Ok(())
    }

    /// K タイル反復のインデックス算術（カーネル内 `t * WMMA_TF32_OPT_K_TILE
    /// + lc` の `int` 算術。`t` はタイル番号〈最大 `num_k_tiles - 1`〉、
    /// `lc` はタイル内オフセット〈最大 `k_tile - 1`〉）がオーバーフロー
    /// しないことを検証する（codex-review 指摘・PR #643 再レビュー。
    /// `gemm.rs::validate_wmma_tf32_opt_k_bound` と同型だが、こちらは
    /// `kernels_wmma_opt::WMMA_TF32_OPT_K_TILE` 固定ではなく本
    /// `cfg.k_tile`〈テンプレート展開元のタイル値。イシュー #516 で
    /// `k_tile` が可変になったため一般化が必須〉で計算する）。
    ///
    /// 実際にカーネルが計算しうる最大インデックスは `ceil(k / k_tile) *
    /// k_tile - 1`（`k == 0` のときは計算自体が発生しないため 0）であり、
    /// これが `i32::MAX` を超えると当該算術が i32 の範囲でオーバーフローしうる
    /// （符号付きオーバーフロー後に境界ガード式 `gk < DIM_K` が誤って
    /// 成立し REQ-8 の境界チェックを迂回しうるため P0）。
    /// `validate_wmma_tf32_opt_config` は `k_tile` が 8 の倍数であること
    /// 等は検査するが、`k` との組合せによる算術オーバーフローは起動時の
    /// `k` に依存するためここで検査する。
    fn validate_k_tile_bound(&self, k: u32) -> Result<(), CudaError> {
        validate_wmma_opt_k_tile_bound(k, self.cfg.k_tile)
    }
}

/// [`RenderedWmmaF16OptKernel::compile`] が返す、f16 opt カーネル専用の
/// コンパイル済み `CudaFunction` と展開元 [`WmmaOptKernelConfig`] を
/// 不可分に束ねた descriptor（[`CompiledWmmaTf32OptKernel`] と対称。
/// 分離理由は同型のドキュメンテーションコメント参照）。
#[allow(dead_code)]
pub struct CompiledWmmaF16OptKernel {
    func: cudarc::driver::CudaFunction,
    cfg: WmmaOptKernelConfig,
}

impl CompiledWmmaF16OptKernel {
    /// 検証済み shape でのみ起動できる、f16 WMMA opt カーネルの
    /// `CudaFunction` へアクセスする唯一の公開経路
    /// （[`CompiledWmmaTf32OptKernel::launch_tf32`] と同じ設計・検査手順。
    /// `self.func` は `RenderedWmmaF16OptKernel::compile` が
    /// `"gemm_wmma_f16_opt"` 固定名でロードしたものであり、f16 引数との
    /// 対応は型で保証される）。
    /// no-op 形状（`m==0 || n==0`）の早期 return・`c_dev` 非書き換え契約は
    /// [`CompiledWmmaTf32OptKernel::launch_tf32`] と同じ（同ドキュメンテーション
    /// コメント参照）。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_f16(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f16>,
        b_dev: &CudaSlice<f16>,
        c_dev: &mut CudaSlice<f16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        self.validate_grid_bounds(m)?;
        self.validate_k_tile_bound(k)?;

        let launch_config = self.cfg.launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: CompiledWmmaTf32OptKernel::launch_tf32 と同一の根拠
        // （型が f16 になる点のみ異なる）。
        unsafe {
            stream
                .launch_builder(&self.func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(launch_config)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_*`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// grid_dim.y の上限検査（[`CompiledWmmaTf32OptKernel::validate_grid_bounds`]
    /// と同じ理由・同じ実装）。
    fn validate_grid_bounds(&self, m: u32) -> Result<(), CudaError> {
        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.cfg.block_m);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "wmma opt path grid_dim.y (m.div_ceil(block_m)={grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (block_m={}); m={m} is \
                     too large",
                    self.cfg.block_m
                ),
            });
        }
        Ok(())
    }

    /// K タイル反復のインデックス添字が i32 でオーバーフローしないことを
    /// 検証する（[`CompiledWmmaTf32OptKernel::validate_k_tile_bound`] と
    /// 同型・同じ理由。codex-review 指摘・PR #643 再レビュー: TF32 経路のみ
    /// この検査が入っており f16 経路が欠けていたため、起動前に fail-closed
    /// で拒否する）。カーネル内の該当算術は本ファイルのテンプレート
    /// `k_base_next = (t + 1) * WMMA_F16_OPT_K_TILE`（`WMMA_F16_OPT_K_TILE`
    /// は本 `cfg.k_tile` の展開元）と同型。
    fn validate_k_tile_bound(&self, k: u32) -> Result<(), CudaError> {
        validate_wmma_opt_k_tile_bound(k, self.cfg.k_tile)
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

    // codex-review 指摘・PR #643 再レビュー: `k_tile`/`block_n` は直前まで
    // 「非ゼロ」「WARP_TILE/FRAG_K の倍数」のみを検査しており上限は無いため、
    // 極端な非既定構成（例: `u32::MAX` 近傍）では通常加算がオーバーフロー
    // しうる（debug では panic、release では wrap して以降の smem 判定が
    // 誤る）。fail-closed 検証契約（本ファイル冒頭）に従い `checked_add`
    // で明示的に拒否する。
    let a_pad = cfg
        .k_tile
        .checked_add(4)
        .ok_or_else(|| invalid("a_pad (k_tile + 4) overflow".to_string()))?;
    let b_pad = cfg
        .block_n
        .checked_add(4)
        .ok_or_else(|| invalid("b_pad (block_n + 4) overflow".to_string()))?;
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
    if smem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES {
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

/// [`CompiledWmmaTf32OptKernel::launch_tf32`]／[`CompiledWmmaF16OptKernel::launch_f16`]
/// の起動前検査の一部として呼ばれる、K タイル反復のインデックス算術
/// （カーネル内 `t * k_tile + lc` の `int` 算術。`t` はタイル番号〈最大
/// `num_k_tiles - 1`〉、`lc` はタイル内オフセット〈最大 `k_tile - 1`〉）が
/// i32 の範囲でオーバーフローしない純粋関数（`self` を要求しないため
/// device 実機なしで単体テスト可能。codex-review 指摘・PR #643 再レビュー）。
/// TF32 opt（`WMMA_TF32_OPT_K_TILE` 展開元）・f16 opt（`WMMA_F16_OPT_K_TILE`
/// 展開元）の双方が同一のタイル反復算術（本ファイル冒頭のカーネルソース
/// テンプレート `k_base_next = (t + 1) * {k_tile}` 参照）を持つため共有する。
///
/// `gemm.rs::validate_wmma_tf32_opt_k_bound` と同型だが、こちらは
/// `kernels_wmma_opt::WMMA_TF32_OPT_K_TILE` 固定ではなく引数 `k_tile`
/// 〈テンプレート展開元のタイル値。イシュー #516 で `k_tile` が可変に
/// なったため一般化が必須〉で計算する。実際にカーネルが計算しうる最大
/// インデックスは `ceil(k / k_tile) * k_tile - 1`（`k == 0` のときは
/// 計算自体が発生しないため 0）であり、これが `i32::MAX` を超えると
/// 当該算術が i32 の範囲でオーバーフローしうる（符号付きオーバーフロー
/// 後に境界ガード式 `gk < DIM_K` が誤って成立し REQ-8 の境界チェックを
/// 迂回しうるため P0）。`validate_wmma_tf32_opt_config`／
/// `validate_wmma_f16_opt_config` は `k_tile` が 8 の倍数であること等は
/// 検査するが、`k` との組合せによる算術オーバーフローは起動時の `k` に
/// 依存するためここで検査する。
fn validate_wmma_opt_k_tile_bound(k: u32, k_tile: u32) -> Result<(), CudaError> {
    let tile = k_tile as u64;
    let max_computed_index = if k == 0 {
        0
    } else {
        (k as u64).div_ceil(tile) * tile - 1
    };
    if max_computed_index > i32::MAX as u64 {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "k tile-index arithmetic for WMMA opt kernel would overflow i32: k={k}, \
                 max_computed_index={max_computed_index}, k_tile={k_tile}"
            ),
        });
    }
    Ok(())
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
/// [`RenderedWmmaTf32OptKernel`] はソースと展開元 `cfg` を保持し、ホスト側の
/// 将来の非既定構成起動 API は `.compile(device)` で `nvrtc::compile_ptx`
/// 等を実行して [`CompiledWmmaTf32OptKernel`] を得て、以降の起動は
/// `CompiledWmmaTf32OptKernel::launch_tf32(stream, a_dev, b_dev, c_dev, m, n, k)`
/// 経由でのみ行うこと（`RenderedWmmaTf32OptKernel`／`CompiledWmmaTf32OptKernel`
/// ドキュメンテーションコメント参照。生ソース・コンパイル済み
/// `CudaFunction` のいずれも検査を経ない形で外部へ返す公開メソッドは
/// 存在しない）。
///
/// `mod kernels_wmma_opt` が非公開モジュールのため、既定構成のみを使う
/// 現状の `gemm.rs`（[`wmma_tf32_f32_opt_source`] 経由）からは呼ばれず
/// rustc の dead-code 解析が誤検知する。非既定 config を渡す呼び出し元は
/// 後続 #504／#519 で追加される想定のため `#[allow(dead_code)]` を付す
/// （`kernels_mma::render_mma_f16` と同じ判断）。
#[allow(dead_code)]
pub fn render_wmma_tf32_opt(
    cfg: &WmmaOptKernelConfig,
) -> Result<RenderedWmmaTf32OptKernel, CudaError> {
    validate_wmma_tf32_opt_config(cfg)?;
    Ok(RenderedWmmaTf32OptKernel {
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

// ============================================================================
// TF32 opt-staged（イシュー #500・GEMM 性能改善ツリー #479 → Phase B の
// TF32 WMMA 経路への横展開）
//
// Phase B（#492〜#499・kernels_mma.rs）で f16 `mma.sync` 経路に確立した
// cp.async 多段パイプライン（B-1/B-5）・fragment 先読みダブルバッファ
// （B-4）を、上記 `WMMA_TF32_F32_OPT_BODY`（`__syncthreads()` ベースの
// 2 面ダブルバッファ・cp.async 不使用）と独立に、WMMA C++ API
// （`nvcuda::wmma`）の TF32 経路へ横展開したもの。
//
// `WMMA_TF32_F32_OPT_BODY` はフォールバック経路として削除しない（整列
// 非対応形状〈`gemm.rs::run_wmma_tf32` の 3 段選択で cp.async 16 バイト
// 整列条件〈n%4==0 && k%4==0〉を満たさない形状〉のフォールバック先として
// 温存する。`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` 2 節「技法の選別」
// 参照）。ただし本体の実装（TF32 丸め〈`__float_to_tf32`〉の適用位置等）
// は正当な性能是正・実機 A/B の結果として変更しうる（イシュー #800・
// #816・#851 の経緯を参照。本コメントが禁じるのは「削除」であり
// 「改変」ではない）。
// ============================================================================

/// TF32 opt-staged GEMM（イシュー #500）のブロックタイル一辺。既存 TF32
/// opt（[`WMMA_TF32_OPT_BLOCK_M`]/[`WMMA_TF32_OPT_BLOCK_N`]）と同一値
/// （64）を採用する。cp.async 多段化・fragment 先読みはブロックタイル
/// 自体を変えない技法のため（`docs/perf/cuda-gemm-wmma-tf32-phase-b.md`
/// 「B-3: タイル拡大」除外理由参照。SMEM 予算がステージ数増加分を
/// 吸収できるのは 64×64 のままだからであり、拡大は別途エピローグ SMEM
/// 再利用の設計が要る）。
pub const WMMA_TF32_STAGED_BLOCK_M: u32 = 64;
pub const WMMA_TF32_STAGED_BLOCK_N: u32 = 64;

/// TF32 opt-staged GEMM の共有メモリ K タイル幅。既存 TF32 opt
/// （[`WMMA_TF32_OPT_K_TILE`]）と同一値（16）。
pub const WMMA_TF32_STAGED_K_TILE: u32 = 16;

/// TF32 opt-staged GEMM の fragment M・N 一辺（`m16n16k8` の 16）。
///
/// [`WMMA_TF32_OPT_FRAG`] と同じ理由（コンパイル時 const アサーションの
/// みからの参照）で rustc 1.88 系 dead-code 誤検知の対象になるため
/// `#[allow(dead_code)]` を付す。
#[allow(dead_code)]
pub const WMMA_TF32_STAGED_FRAG: u32 = 16;

/// TF32 opt-staged GEMM の fragment K 一辺（`m16n16k8` の 8）。
#[allow(dead_code)]
pub const WMMA_TF32_STAGED_FRAG_K: u32 = 8;

/// TF32 opt-staged GEMM の warp タイル一辺（32。fragment 辺 16 の 2 倍 =
/// レジスタブロッキング 2×2。既存 TF32 opt と同一構成）。
#[allow(dead_code)]
pub const WMMA_TF32_STAGED_WARP_TILE: u32 = 32;

/// TF32 opt-staged GEMM 1 ブロックあたりのスレッド数（4 warp = 128
/// スレッド。既存 TF32 opt と同一）。
pub const WMMA_TF32_STAGED_THREADS: u32 = 128;

/// `cp.async` multi-stage pipelining のステージ数（イシュー #500）。
/// `kernels_mma.rs::MMA_STAGES` と同じ 3 段構成を採用する（B-1/B-5 の
/// 横展開）。共有メモリ予算はステージ数増加に伴い増えるが、ブロック
/// タイルを 64×64 に据え置くことで
/// `crates/backend-cuda/src/kernels_mma.rs::MMA_STATIC_SMEM_LIMIT_BYTES`
/// （48KiB）以内に収まる（`validate_wmma_tf32_staged_config` が実測検査
/// する。`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` 「SMEM 予算」節参照）。
pub const WMMA_TF32_STAGED_STAGES: u32 = 3;

// `kernels_mma.rs::MMA_STAGES` 定数直下コメント「正しさ」と同一の論証:
// ループ内 `cp.async.wait_group (STAGES-2)` の即値が非負であるための
// コンパイル時保証。
const _: () = assert!(
    WMMA_TF32_STAGED_STAGES >= 2,
    "kernels_wmma_opt::WMMA_TF32_F32_STAGED の cp.async パイプラインは \
     STAGES >= 2 を要求する（wait_group (STAGES-2) の即値が非負であることの \
     コンパイル時保証。kernels_mma.rs::MMA_STAGES 定数直下コメント「正しさ」\
     と同じ論証）"
);

/// A タイル（`as_tile[STAGES][BLOCK_M][A_PAD]`）の行幅（パディング後）。
/// `K_TILE`（16）に 4 要素加算する（既存 TF32 opt の [`WMMA_TF32_OPT_A_PAD`]
/// と同じ根拠）。cp.async の 16 バイト転送粒度は f32 4 要素のため、
/// パディング幅も 4 要素の倍数である必要がある（下記 const アサーション）。
/// [`WmmaTf32StagedKernelConfig::default_tf32_staged`] の `a_pad` 既定値の
/// 唯一の真実源（イシュー #743 でパディングを config フィールド化した後も、
/// 本番経路が展開する値そのものは本定数のまま変更していない）。
#[allow(dead_code)]
pub const WMMA_TF32_STAGED_A_PAD: u32 = WMMA_TF32_STAGED_K_TILE + 4;

/// B タイル（`bs_tile[STAGES][K_TILE][B_PAD]`）の行幅（パディング後）。
/// `BLOCK_N`（64）に 4 要素加算する。A パディングと同じ根拠。
/// [`WmmaTf32StagedKernelConfig::default_tf32_staged`] の `b_pad` 既定値の
/// 唯一の真実源。
///
/// # イシュー #743: SMEM バンクコンフリクト解析（既定値は未変更）
///
/// 実機 ncu 計測（2026-08-19・GB10）で TF32 staged 経路の
/// `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` が
/// M=N=K=2048 で 8.53M、4096 で 67.5M 検出された。理論解析
/// （PTX ISA `mma.m16n8k8.tf32` フラグメントの lane→(row,col) 対応を
/// `wmma::load_matrix_sync` の row_major 読み出しへ適用したモデル。
/// [`wmma_tf32_staged_b_fragment_ld_wavefronts`]／
/// [`wmma_tf32_staged_a_fragment_ld_wavefronts`] が同モデルを実装し値を
/// 固定する）によれば、本定数の 68（`BLOCK_N + 4`）は B フラグメント
/// ロード 1 命令あたり 2-way バンクコンフリクト（余剰 wavefront 1）を
/// 生み、この余剰が ncu 実測値（4096 で ≈67M 命令 × 余剰 1 ≒ 67.5M）と
/// ほぼ一致する。`B_PAD mod 32 ∈ {8, 24}`（例: 72 = `BLOCK_N + 8`）を
/// 満たすと 32 バンクを完全被覆しコンフリクトが理論上ゼロになる一方、
/// A 側（本ファイル [`WMMA_TF32_STAGED_A_PAD`] = 20）は既にコンフリクト
/// フリーである（詳細な位相計算・定量突合・SMEM/occupancy 影響・XOR
/// swizzle 不採用理由は
/// `docs/perf/cuda-gemm-wmma-tf32-staged-bank-conflict.md` を参照）。
///
/// 本定数自体は実機 ncu で「ld 有意減少・TFLOPS 非劣化・parity 全 pass・
/// bit 一致」を確認できるまで **68 のまま変更しない**（実装計画 §1・
/// #497/#499/#741/#742 と同じ「未計測の間は採用済みとして扱わない」判断。
/// 変更する場合は本定数 1 行の書き換えで済む設計とするため、パディング幅
/// 自体は [`WmmaTf32StagedKernelConfig`] の `a_pad`/`b_pad` フィールドへ
/// config 化してある）。
#[allow(dead_code)]
pub const WMMA_TF32_STAGED_B_PAD: u32 = WMMA_TF32_STAGED_BLOCK_N + 4;

const _: () = assert!(
    WMMA_TF32_STAGED_A_PAD.is_multiple_of(4),
    "cp.async 16 バイト転送は f32 4 要素粒度のため A_PAD は 4 要素の倍数が必要"
);
const _: () = assert!(
    WMMA_TF32_STAGED_B_PAD.is_multiple_of(4),
    "cp.async 16 バイト転送は f32 4 要素粒度のため B_PAD は 4 要素の倍数が必要"
);

/// [`WmmaTf32StagedKernelConfig`] で展開したソースを
/// [`wmma_tf32_f32_staged_source`] が 1 回だけキャッシュして返す
/// （[`wmma_tf32_f32_opt_source`] と同じ方針）。
pub fn wmma_tf32_f32_staged_source() -> &'static str {
    &WMMA_TF32_F32_STAGED_SOURCE
}

static WMMA_TF32_F32_STAGED_SOURCE: LazyLock<String> = LazyLock::new(|| {
    render_wmma_tf32_staged_unchecked(&WmmaTf32StagedKernelConfig::default_tf32_staged())
});

/// TF32 opt-staged カーネル（[`wmma_tf32_f32_staged_source`]）のブロック
/// 原点（`block_row_base`／`block_col_base`）を、イシュー #499（f16 経路。
/// `kernels_mma.rs::mma_f16_source_with_swizzle`）と同一設計の
/// threadblock swizzle remap へ置換した変種ソースを生成する（イシュー
/// #741）。
///
/// # 背景（イシュー #741）
///
/// ncu 実測（2026-08-19・GB10）で M=N=K=4096 時に TF32 opt-staged 経路の
/// L2 hit rate が 96.77%→76.51% へ崩壊しており（`docs/perf/
/// cuda-gemm-bottleneck-diagnosis.md`）、f16 経路で同型対策（#499）が
/// 4096 で約 1.60 倍（`docs/perf/cuda-gemm-swizzle-ab.md` 実測記録）を
/// 出したことから、TF32 staged 側にも同じ remap を横展開して A/B 計測
/// する。remap 自体は `mma_f16_source_with_swizzle` の式をブロックタイル
/// 幅（`WMMA_TF32_STAGED_BLOCK_M`/`_N`）向けに転用したもので、
/// `swizzle.rs::swizzled_block_idx` と単一の設計を共有する（本関数と
/// `swizzle.rs` のホスト側参照実装は独立実装のため、不一致は本ファイル
/// `tests` モジュールが機械検出する。`swizzle.rs` 冒頭コメント参照）。
///
/// # 呼び出し元
///
/// **イシュー #856（2026-08-22 GB10 実機 A/B・§7.4.1 サイズ条件付き新
/// 基準で採用）で本番結線済み**: `gemm.rs::CudaGemm::new`（本番既定
/// コンストラクタ・feature 非依存）が SM 数実測時に fail-soft でこの
/// 変種を追加コンパイルし、`run_wmma_tf32`／`launch_wmma_tf32` の staged
/// 分岐が形状ごとに `swizzle::should_apply_swizzle` で base／変種の
/// いずれを起動するか判定する（`gemm.rs::CudaGemm::
/// should_launch_wmma_tf32_staged_swizzle` 参照）。加えて
/// `gemm.rs::CudaGemm::new_with_tf32_staged_swizzle`（`internal-diagnostics`
/// feature ゲート・明示幅の強制適用診断入口）からも引き続き呼ばれる。
/// `ops.rs`／`gemm_auto.rs` の経路選択自体（staged→opt→basic の 3 段
/// フォールバック）は無変更。**`wmma_tf32_f32_staged_source()` 自体は変更しない**
/// （`wmma_tf32_f32_staged_source_with_swizzle_does_not_mutate_wmma_tf32_f32_staged_source`
/// が回帰検査する）。
///
/// # グリッド軸の対応
///
/// staged の launch config（[`WmmaTf32StagedKernelConfig::launch_config`]:
/// `grid_dim = (n.div_ceil(block_n), m.div_ceil(block_m), 1)`）に合わせ
/// `num_m_blocks = gridDim.y`・`num_n_blocks = gridDim.x` とする
/// （f16 側 `mma_f16_source_with_swizzle` と同一の対応）。
///
/// # エラー契約
///
/// `group_width < 2` は `CudaError::InvalidShape` で拒否する
/// （`group_width == 1` は恒等写像に等しく L2 再利用効果を持たない。
/// `mma_f16_source_with_swizzle` と同じ判断）。
///
/// # セキュリティ（OWASP A03 インジェクション対策）
///
/// 任意のソース文字列・任意の書式文字列を受ける公開 API は作らない。
/// 受理するのは `u32`（`group_width`）のみで、生成断片への埋め込みは
/// 固定テンプレート文字列内の数値 `format!` のみに限定する
/// （`mma_f16_source_with_swizzle` と同一契約）。
///
/// イシュー #856 追記: 旧 `#[allow(dead_code)]`（唯一の呼び出し元が
/// `internal-diagnostics` feature 限定の `new_with_tf32_staged_swizzle`
/// のみだった時期の対応）は、本番既定コンストラクタ `gemm.rs::
/// CudaGemm::new`（feature 非依存）が SM 数実測時にこの関数を直接呼ぶ
/// ようになったため撤去した（`cargo build` feature 指定なしでも呼び出し元
/// が存在し dead-code lint は発火しない）。
pub fn wmma_tf32_f32_staged_source_with_swizzle(
    group_width: u32,
) -> Result<String, crate::error::CudaError> {
    if group_width < 2 {
        return Err(crate::error::CudaError::InvalidShape {
            detail: format!(
                "wmma_tf32_f32_staged_source_with_swizzle requires group_width >= 2 \
                 (got {group_width}); group_width == 1 degenerates to the identity \
                 block mapping (swizzle.rs::swizzled_block_idx_group_width_one_is_identity_mapping) \
                 and offers no L2 reuse benefit"
            ),
        });
    }

    const ANCHOR: &str = "    const int block_row_base = blockIdx.y * WMMA_TF32_STAGED_BLOCK_M;\n    const int block_col_base = blockIdx.x * WMMA_TF32_STAGED_BLOCK_N;\n";
    let source = wmma_tf32_f32_staged_source();
    let occurrences = source.matches(ANCHOR).count();
    // `unwrap()`/`expect()`・panic 系マクロを本番経路で使わない方針
    // （coding-rust.md「エラーは型付きエラーとし、本番経路で unwrap()
    // / expect() を使わない」）に合わせ、`assert_eq!` ではなく型付き
    // エラーで返す。`wmma_tf32_f32_staged_source()` 側の不変条件は
    // `wmma_tf32_f32_staged_source_with_swizzle_does_not_mutate_wmma_tf32_f32_staged_source`
    // が別途 CI 上で回帰検査するため通常到達しない前提だが、
    // `new_with_tf32_staged_swizzle` から到達しうる公開関数として panic
    // を避ける。
    if occurrences != 1 {
        return Err(crate::error::CudaError::InvalidShape {
            detail: format!(
                "TF32 opt-staged カーネル中のブロック原点アンカー \
                 （block_row_base/block_col_base）の出現数が 1 ではありません \
                 （{occurrences} 件検出。wmma_tf32_f32_staged_source_with_swizzle \
                 の前提が崩れています）"
            ),
        });
    }

    let remap = format!(
        "    // イシュー #741: L2 再利用のためのタイル→SM 割り当てスウィズル\n\
         \x20   // （#499 の f16 経路と同一設計。remap 式は\n\
         \x20   // swizzle.rs::swizzled_block_idx・kernels_mma.rs::\n\
         \x20   // mma_f16_source_with_swizzle と単一の設計を共有する。本ファイル\n\
         \x20   // wmma_tf32_f32_staged_source_with_swizzle ドキュメンテーション\n\
         \x20   // コメント参照）。\n\
         \x20   // PR #667 codex-review P0 是正の踏襲: 線形 index・ブロック数・積は\n\
         \x20   // `long long`（64 bit）で計算する（`blockIdx.y * gridDim.x` 等の\n\
         \x20   // `int`（32 bit 符号付き）オーバーフロー防止。REQ-8「境界検査の\n\
         \x20   // 省略禁止」）。最終座標は `gridDim` 内であることを明示的に検査\n\
         \x20   // してから `int` へ縮小する（縮小前に範囲外を検査するため、この\n\
         \x20   // 縮小自体は新たな符号なし/符号付きオーバーフロー経路を導入\n\
         \x20   // しない）。\n\
         \x20   #define WMMA_TF32_STAGED_SWIZZLE_GROUP {group_width}\n\
         \x20   long long num_m_blocks = gridDim.y;\n\
         \x20   long long num_n_blocks = gridDim.x;\n\
         \x20   long long linear_idx = (long long)blockIdx.y * gridDim.x + blockIdx.x;\n\
         \x20   long long full_groups = num_m_blocks / WMMA_TF32_STAGED_SWIZZLE_GROUP;\n\
         \x20   long long remainder = num_m_blocks % WMMA_TF32_STAGED_SWIZZLE_GROUP;\n\
         \x20   long long full_group_blocks =\n\
         \x20       (long long)WMMA_TF32_STAGED_SWIZZLE_GROUP * num_n_blocks;\n\
         \x20   long long full_groups_total_blocks = full_groups * full_group_blocks;\n\
         \x20   long long m_block, n_block;\n\
         \x20   if (linear_idx < full_groups_total_blocks) {{\n\
         \x20       long long group_idx = linear_idx / full_group_blocks;\n\
         \x20       long long idx_in_group = linear_idx % full_group_blocks;\n\
         \x20       m_block = group_idx * WMMA_TF32_STAGED_SWIZZLE_GROUP +\n\
         \x20           (idx_in_group % WMMA_TF32_STAGED_SWIZZLE_GROUP);\n\
         \x20       n_block = idx_in_group / WMMA_TF32_STAGED_SWIZZLE_GROUP;\n\
         \x20   }} else {{\n\
         \x20       long long idx_in_group = linear_idx - full_groups_total_blocks;\n\
         \x20       m_block = full_groups * WMMA_TF32_STAGED_SWIZZLE_GROUP +\n\
         \x20           (idx_in_group % remainder);\n\
         \x20       n_block = idx_in_group / remainder;\n\
         \x20   }}\n\
         \x20   if (m_block < 0 || m_block >= num_m_blocks || n_block < 0 ||\n\
         \x20       n_block >= num_n_blocks) {{\n\
         \x20       return;\n\
         \x20   }}\n\
         \x20   const int block_row_base = (int)(m_block * WMMA_TF32_STAGED_BLOCK_M);\n\
         \x20   const int block_col_base = (int)(n_block * WMMA_TF32_STAGED_BLOCK_N);\n"
    );

    Ok(source.replacen(ANCHOR, &remap, 1))
}

/// [`WmmaTf32StagedKernelConfig::default_tf32_staged`] の `a_pad`/`b_pad`
/// のみを差し替えた変種ソースを生成する（イシュー #743。
/// [`wmma_tf32_f32_staged_source_with_swizzle`] と同じ「`CudaGemm::new` の
/// `wmma_tf32_staged` スロットを差し替える opt-in 変種」設計）。
///
/// # 背景（イシュー #743）
///
/// `WMMA_TF32_STAGED_B_PAD`（既定 68 = `BLOCK_N + 4`）直下コメントの理論
/// 解析どおり、B フラグメントロードは 2-way バンクコンフリクトを持ち、
/// `b_pad mod 32 ∈ {8, 24}`（例: 72）で理論上ゼロになる。本関数は
/// `a_pad`/`b_pad` のみを config 経由で差し替えたソースを生成し、
/// `gemm.rs::CudaGemm::new_with_tf32_staged_pads` から実機 ncu A/B 計測
/// （`docs/perf/cuda-gemm-wmma-tf32-staged-bank-conflict.md`）に使う。
///
/// パディングはタイルの行ストライドのみを変え、各要素の値・累積順序は
/// 変えない（`as_tile`/`bs_tile` は依然として `WMMA_TF32_STAGED_K_TILE`/
/// `WMMA_TF32_STAGED_BLOCK_N` 個の有効要素を保持し、パディング領域は
/// 読み書きされない）。よって swizzle 変種と同じ論法で base との
/// **bit 一致**を主張できる（`gemm.rs::
/// wmma_tf32_staged_pad_variant_matches_base_bit_exact_output` 参照）。
///
/// # エラー契約
///
/// `a_pad`/`b_pad` の妥当性（`k_tile`/`block_n` 以上・4 要素倍数・SMEM
/// 予算内）は [`validate_wmma_tf32_staged_config`] が fail-closed 検査する
/// （`render_wmma_tf32_staged` と同じ経路）。
///
/// # セキュリティ（OWASP A03）
///
/// [`wmma_tf32_f32_staged_source_with_swizzle`] と同じ契約: 受理するのは
/// `u32` 2 個のみで、生成断片への埋め込みは固定テンプレート文字列内の
/// 数値 `format!`（`render_wmma_tf32_staged_header`）のみに限定する。
///
/// `#[allow(dead_code)]` について: 本番ビルド（`internal-diagnostics`
/// feature 既定 off）からの唯一の呼び出し元
/// `gemm.rs::CudaGemm::new_with_tf32_staged_pads` が同 feature でゲート
/// されているため、`cargo build`（feature 指定なし）では呼び出し元が
/// 存在せず dead-code lint が誤検出する（swizzle 版と同じパターン）。
#[allow(dead_code)]
pub fn wmma_tf32_f32_staged_source_with_pads(
    a_pad: u32,
    b_pad: u32,
) -> Result<String, crate::error::CudaError> {
    let mut cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
    cfg.a_pad = a_pad;
    cfg.b_pad = b_pad;
    validate_wmma_tf32_staged_config(&cfg)?;
    Ok(render_wmma_tf32_staged_unchecked(&cfg))
}

/// [`render_wmma_tf32_staged`] に渡す構成値（イシュー #500）。
///
/// 既存 [`WmmaOptKernelConfig`]（TF32 opt・f16 opt 共通）へ `stages`
/// フィールドを追加する代替案は採らず、staged 専用の独立した struct と
/// した（実装計画 3.1 節。共有 struct へフィールドを追加すると、本ファイル
/// 内の f16 opt・TF32 opt 双方の既存 `WmmaOptKernelConfig { .. }` リテラル
/// 構築〈`default_tf32`/`default_f16`・各テストの部分構築〉が軒並み
/// 影響を受けるため、影響範囲を本カーネルに閉じる）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WmmaTf32StagedKernelConfig {
    /// ブロックタイル M（warp タイル辺の倍数必須）。
    pub block_m: u32,
    /// ブロックタイル N（warp タイル辺の倍数必須）。
    pub block_n: u32,
    /// 共有メモリ K タイル幅。
    pub k_tile: u32,
    /// `cp.async` multi-stage pipelining のステージ数（`>= 2` 必須。
    /// [`WMMA_TF32_STAGED_STAGES`] 直下コメント参照）。
    pub stages: u32,
    /// A タイル（`as_tile[STAGES][BLOCK_M][A_PAD]`）の行幅（パディング後。
    /// イシュー #743。既定値は [`WMMA_TF32_STAGED_A_PAD`]）。SMEM バンク
    /// コンフリクト計測のため config フィールド化してあるが、本番経路
    /// （[`default_tf32_staged`](Self::default_tf32_staged)）は従来の
    /// 定数値のまま byte 完全一致で展開する
    /// （`wmma_tf32_staged_default_config_render_is_byte_identical_to_production_source`
    /// が回帰検査する）。[`validate_wmma_tf32_staged_padding`] が
    /// `k_tile` との整合・4 要素倍数・余剰上限を fail-closed 検査する。
    pub a_pad: u32,
    /// B タイル（`bs_tile[STAGES][K_TILE][B_PAD]`）の行幅（パディング後。
    /// イシュー #743）。`a_pad` と同じ契約。既定値は
    /// [`WMMA_TF32_STAGED_B_PAD`]。
    pub b_pad: u32,
    /// M 次元の焼き込み方式。
    pub dim_m: DimSpec,
    /// N 次元の焼き込み方式。
    pub dim_n: DimSpec,
    /// K 次元の焼き込み方式。
    pub dim_k: DimSpec,
}

impl WmmaTf32StagedKernelConfig {
    /// TF32 opt-staged カーネルの既定構成（Rust 側タイル定数と同一値。
    /// 全次元 `Dynamic`）。`a_pad`/`b_pad` は
    /// [`WMMA_TF32_STAGED_A_PAD`]/[`WMMA_TF32_STAGED_B_PAD`]（本番経路の
    /// 唯一の真実源）をそのまま採用するため、本関数が返す構成の展開結果は
    /// イシュー #743 のパディング config 化の前後で byte 完全一致を保つ。
    pub fn default_tf32_staged() -> Self {
        Self {
            block_m: WMMA_TF32_STAGED_BLOCK_M,
            block_n: WMMA_TF32_STAGED_BLOCK_N,
            k_tile: WMMA_TF32_STAGED_K_TILE,
            stages: WMMA_TF32_STAGED_STAGES,
            a_pad: WMMA_TF32_STAGED_A_PAD,
            b_pad: WMMA_TF32_STAGED_B_PAD,
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Dynamic,
        }
    }

    /// [`WmmaOptKernelConfig::validate_launch_shape`] と同じ設計。
    #[allow(dead_code)] // 理由は WmmaOptKernelConfig::validate_launch_shape と同じ
    pub fn validate_launch_shape(&self, m: u32, n: u32, k: u32) -> Result<(), CudaError> {
        self.dim_m.matches_launch_dim(m)?;
        self.dim_n.matches_launch_dim(n)?;
        self.dim_k.matches_launch_dim(k)?;
        Ok(())
    }

    /// [`WmmaOptKernelConfig::launch_config`] と同じ設計（`stages` は
    /// grid/block 次元に影響しない。共有メモリは静的宣言のため
    /// `shared_mem_bytes` は常に 0）。
    #[allow(dead_code)] // 理由は WmmaOptKernelConfig::launch_config と同じ
    pub fn launch_config(&self, m: u32, n: u32) -> LaunchConfig {
        const WARP_TILE: u32 = 32;
        const _: () = assert!(
            WARP_TILE == WMMA_TF32_STAGED_WARP_TILE,
            "WARP_TILE は WMMA_TF32_STAGED_WARP_TILE と一致している必要があります"
        );
        let warp_grid_m = self.block_m / WARP_TILE;
        let warp_grid_n = self.block_n / WARP_TILE;
        LaunchConfig {
            grid_dim: (n.div_ceil(self.block_n), m.div_ceil(self.block_m), 1),
            block_dim: (warp_grid_m * warp_grid_n * 32, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// [`render_wmma_tf32_staged`] が返す、展開済み TF32 opt-staged カーネル
/// ソースと展開元 [`WmmaTf32StagedKernelConfig`] を 1 個にまとめた
/// descriptor（[`RenderedWmmaTf32OptKernel`] と同じ設計）。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedWmmaTf32StagedKernel {
    source: String,
    cfg: WmmaTf32StagedKernelConfig,
}

impl RenderedWmmaTf32StagedKernel {
    /// [`RenderedWmmaTf32OptKernel::compile`] と同じ設計。固定エント
    /// リポイント `"gemm_wmma_tf32_staged"`。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledWmmaTf32StagedKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_tf32_staged")?;
        Ok(CompiledWmmaTf32StagedKernel {
            func,
            cfg: self.cfg,
        })
    }

    #[cfg(test)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// [`RenderedWmmaTf32StagedKernel::compile`] が返す、コンパイル済み
/// `CudaFunction` と展開元 [`WmmaTf32StagedKernelConfig`] を不可分に
/// 束ねた descriptor（[`CompiledWmmaTf32OptKernel`] と同じ設計）。
#[allow(dead_code)]
pub struct CompiledWmmaTf32StagedKernel {
    func: cudarc::driver::CudaFunction,
    cfg: WmmaTf32StagedKernelConfig,
}

impl CompiledWmmaTf32StagedKernel {
    /// [`CompiledWmmaTf32OptKernel::launch_tf32`] と同じ検証手順・同じ
    /// no-op early return 契約に加え、cp.async 16 バイト整列制約
    /// （[`crate::gemm::wmma_tf32_staged_alignment_ok`]）を fail-closed で
    /// 検証する。
    ///
    /// `gemm.rs::run_wmma_tf32`（3 段フォールバックの経路選択）は同じ
    /// 判定関数を呼んで整列 NG の形状を staged 経路へそもそも渡さない
    /// ため通常はここで拒否されることはないが、本メソッドは
    /// `pub`（crate 内から直接到達可能）であり、将来の呼び出し元が
    /// 経路選択のフォールバックを経由せずに直接呼ぶ可能性がある。
    /// `gemm_mma.rs::CudaMmaGemm::launch`（`validate_mma_alignment` を
    /// 唯一のゲートとして起動前に必ず経由させる設計）と同じ
    /// 「起動 API 自体が fail-closed である」契約に揃えるため、経路選択
    /// 側の判定に依存せずここでも検証する（Bugbot 指摘 PR #678 review
    /// id 4945411529）。満たさない場合、cp.async の 16 バイト転送粒度が
    /// 要求するグローバル側整列を欠いたままカーネルを起動し、フォールト
    /// または silent corruption を招きうる。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_tf32_staged(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if !crate::gemm::wmma_tf32_staged_alignment_ok(n, k) {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "wmma tf32 staged path requires n % 4 == 0 && k % 4 == 0 (cp.async \
                     16-byte transfer granularity; gemm.rs::wmma_tf32_staged_alignment_ok \
                     doc comment), but got n={n}, k={k}"
                ),
            });
        }
        self.validate_grid_bounds(m)?;
        self.validate_k_tile_bound(k)?;

        let launch_config = self.cfg.launch_config(m, n);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: CompiledWmmaTf32OptKernel::launch_tf32 と同一の根拠。
        // カーネル引数は上記で検証済みの m/n/k から導出しており、opt-staged
        // カーネル内の手動境界チェック（guarded load・エピローグ store の
        // ガード付きコピー。REQ-8）と合わせて OOB 読み書きが起きない根拠
        // とする。`self.func` は本型のコンストラクタである
        // `RenderedWmmaTf32StagedKernel::compile` が
        // `"gemm_wmma_tf32_staged"` 固定名でロードしたものであり、f32
        // 引数と TF32 カーネルシグネチャの対応は型で保証される。
        unsafe {
            stream
                .launch_builder(&self.func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(launch_config)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_*`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }

    /// [`CompiledWmmaTf32OptKernel`] の同名メソッドと同じ理由。
    fn validate_grid_bounds(&self, m: u32) -> Result<(), CudaError> {
        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.cfg.block_m);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "wmma staged path grid_dim.y (m.div_ceil(block_m)={grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (block_m={}); m={m} is \
                     too large",
                    self.cfg.block_m
                ),
            });
        }
        Ok(())
    }

    /// [`CompiledWmmaTf32OptKernel::validate_k_tile_bound`] と同じ理由。
    /// [`validate_wmma_opt_k_tile_bound`] は `k_tile` を引数に取る汎用実装
    /// のため、TF32 opt・opt-staged の双方から共有できる。
    fn validate_k_tile_bound(&self, k: u32) -> Result<(), CudaError> {
        validate_wmma_opt_k_tile_bound(k, self.cfg.k_tile)
    }
}

/// TF32 opt-staged カーネルの A/B フラグメントロード（イシュー #743）が
/// 1 命令あたり要する SMEM バンク wavefront 数を理論モデルで算出する
/// 共通ヘルパ。GPU 不要（純粋な整数演算）で `internal-diagnostics`
/// feature 非依存に呼べるため、`#[cfg(test)]` 内のロック用テスト
/// （`wmma_tf32_staged_b_pad_72_is_bank_conflict_free_and_68_is_two_way`）
/// と `diagnostics` モジュール（計測 example・イシュー #743 §計測基盤）の
/// 双方から参照する単一ソース。
///
/// # モデルの前提（PTX ISA `mma.m16n8k8.tf32` フラグメントレイアウト）
///
/// `wmma::load_matrix_sync` の実際の lowering は不透明だが、TF32
/// `m16n16k8` は PTX `mma.m16n8k8.tf32` 相当の 32 レーン割り当て
/// （groupID = lane/4, thread-in-group = lane%4）に従うと仮定し、
/// row_major な `as_tile[..][A_PAD]`/`bs_tile[..][B_PAD]` からの 1 回の
/// フラグメントロード命令が発行する 32 レーン分のシェアードメモリアクセス
/// を次のようにモデル化する（32 バンク・4B/バンク・実装計画 §1.1(b)）:
///
/// - A（8 行 × 4 列、行ストライドが `a_pad`）: `bank(g, t) = (a_pad*g + t)
///   mod 32`（g=0..8 が行、t=0..4 が列）
/// - B（4 行 × 8 列、行ストライドが `b_pad`）: `bank(t, g) = (b_pad*t + g)
///   mod 32`（t=0..4 が行、g=0..8 が列）
///
/// wavefront 数は 32 バンクのうち最も競合したバンクへの多重度（同一
/// サイクルで処理しきれず追加 wavefront を要する回数）。この
/// lane→(row,col) 対応は WMMA lowering の実装詳細を保証するものではなく、
/// あくまで実測 ncu 値との定量突合（本ファイル
/// [`WMMA_TF32_STAGED_B_PAD`] ドキュメンテーションコメント参照）が
/// 支持する仮説であるため、実機 ncu 実測が最終的な正である。
fn wmma_tf32_staged_fragment_ld_wavefronts(pad: u32, outer_count: u32, inner_count: u32) -> u32 {
    const BANKS: usize = 32;
    let mut bank_hits = [0u32; BANKS];
    for outer in 0..outer_count {
        for inner in 0..inner_count {
            // `pad`/`outer`/`inner` は本ファイル内の小さい定数（タイル幅
            // 由来。u32::MAX に到達しえない）のみを受けるため折返し乗算で
            // 十分だが、境界検査省略禁止方針（REQ-8）に合わせ checked 系で
            // 計算し、万一のオーバーフローは wavefront 未定義として扱わず
            // 飽和させる（本関数は診断専用でありパニックさせない）。
            let bank = pad
                .checked_mul(outer)
                .and_then(|v| v.checked_add(inner))
                .map(|v| (v % BANKS as u32) as usize)
                .unwrap_or(0);
            bank_hits[bank] += 1;
        }
    }
    bank_hits.into_iter().max().unwrap_or(0)
}

/// A フラグメント（`as_tile`。行 8・列 4）のロード 1 命令あたりの SMEM
/// バンク wavefront 数（[`wmma_tf32_staged_fragment_ld_wavefronts`]
/// 参照）。既定 `a_pad`（[`WMMA_TF32_STAGED_A_PAD`] = 20）は 1
/// （コンフリクトフリー）。
#[allow(dead_code)]
pub fn wmma_tf32_staged_a_fragment_ld_wavefronts(a_pad: u32) -> u32 {
    wmma_tf32_staged_fragment_ld_wavefronts(a_pad, 8, 4)
}

/// B フラグメント（`bs_tile`。行 4・列 8）のロード 1 命令あたりの SMEM
/// バンク wavefront 数（[`wmma_tf32_staged_fragment_ld_wavefronts`]
/// 参照）。既定 `b_pad`（[`WMMA_TF32_STAGED_B_PAD`] = 68）は 2
/// （実機 ncu 実測の ld バンクコンフリクトの主因。本ファイル
/// [`WMMA_TF32_STAGED_B_PAD`] ドキュメンテーションコメント参照）。
#[allow(dead_code)]
pub fn wmma_tf32_staged_b_fragment_ld_wavefronts(b_pad: u32) -> u32 {
    wmma_tf32_staged_fragment_ld_wavefronts(b_pad, 4, 8)
}

/// [`WmmaTf32StagedKernelConfig`] を TF32 opt-staged カーネル向けに
/// fail-closed 検証する（[`validate_wmma_tf32_opt_config`] と同じ方針。
/// 追加で `stages`・cp.async 4 要素整列制約を検査する）。
/// `cfg.a_pad`/`cfg.b_pad`（イシュー #743）を fail-closed 検査する共通
/// ヘルパ。[`validate_wmma_tf32_staged_config`]（static 変種）・
/// [`validate_wmma_tf32_staged_dyn_config`]（`internal-diagnostics` 計測
/// 変種。イシュー #742）の双方から呼ばれる単一ソース。
///
/// 検査項目（実装計画 §3「pad 検証」）:
/// - `a_pad >= k_tile`・`b_pad >= block_n`（負のパディングは
///   `as_tile`/`bs_tile` の行内で隣接要素と重なり OOB 相当の破損を招く
///   ため拒否する）
/// - `a_pad`/`b_pad` は 4 要素（16 バイト）の倍数（cp.async 転送粒度。
///   [`WMMA_TF32_STAGED_A_PAD`]/[`WMMA_TF32_STAGED_B_PAD`] 直下の const
///   アサーションと同じ根拠を可変 config でも検査する）
/// - 余剰（`a_pad - k_tile`／`b_pad - block_n`）は 32 要素（バンク周期。
///   本ファイル [`WMMA_TF32_STAGED_B_PAD`] ドキュメンテーションコメント
///   「イシュー #743」節参照）以下。バンク周期を超える余剰はバンク
///   コンフリクト対策として無意味な SMEM 浪費にしかならないため、
///   誤設定を早期に拒否する
fn validate_wmma_tf32_staged_padding(cfg: &WmmaTf32StagedKernelConfig) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };
    const BANK_PERIOD_ELEMENTS: u32 = 32;

    if !cfg.a_pad.is_multiple_of(4) || !cfg.b_pad.is_multiple_of(4) {
        return Err(invalid(format!(
            "a_pad ({}) and b_pad ({}) must both be multiples of 4 (cp.async 16-byte transfer \
             granularity for f32 elements)",
            cfg.a_pad, cfg.b_pad
        )));
    }

    let a_extra = cfg.a_pad.checked_sub(cfg.k_tile).ok_or_else(|| {
        invalid(format!(
            "a_pad ({}) must be >= k_tile ({}) (negative padding would overlap adjacent \
             as_tile row elements)",
            cfg.a_pad, cfg.k_tile
        ))
    })?;
    let b_extra = cfg.b_pad.checked_sub(cfg.block_n).ok_or_else(|| {
        invalid(format!(
            "b_pad ({}) must be >= block_n ({}) (negative padding would overlap adjacent \
             bs_tile row elements)",
            cfg.b_pad, cfg.block_n
        ))
    })?;

    if a_extra > BANK_PERIOD_ELEMENTS || b_extra > BANK_PERIOD_ELEMENTS {
        return Err(invalid(format!(
            "padding extra beyond the tile width (a_extra={a_extra}, b_extra={b_extra}) must \
             not exceed the {BANK_PERIOD_ELEMENTS}-element SMEM bank period; extra padding \
             beyond one full bank period cannot reduce bank conflicts further and only wastes \
             shared memory (a_pad={}, k_tile={}, b_pad={}, block_n={})",
            cfg.a_pad, cfg.k_tile, cfg.b_pad, cfg.block_n
        )));
    }

    Ok(())
}

fn validate_wmma_tf32_staged_config(cfg: &WmmaTf32StagedKernelConfig) -> Result<(), CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };
    let warp_tile = WMMA_TF32_STAGED_WARP_TILE;
    let frag_k = WMMA_TF32_STAGED_FRAG_K;

    if cfg.block_m == 0 || cfg.block_n == 0 || cfg.k_tile == 0 {
        return Err(invalid(
            "block_m/block_n/k_tile must all be non-zero".to_string(),
        ));
    }
    // `cp.async.wait_group (STAGES-2)` の即値が非負であるための前提
    // （[`WMMA_TF32_STAGED_STAGES`] 直下コメント参照）。
    if cfg.stages < 2 {
        return Err(invalid(format!(
            "stages ({}) must be >= 2 (cp.async wait_group (STAGES-2) immediate must be \
             non-negative; kernels_mma.rs::MMA_STAGES 定数直下コメント「正しさ」と同じ論証)",
            cfg.stages
        )));
    }
    // `stages` の上限は SMEM 予算だけでは決まらない（codex-review 指摘:
    // block_m/block_n/k_tile を小さくした構成では SMEM 予算内のまま stages
    // を際限なく大きく取れてしまう）ため、別枠の上限検査が必要である。
    //
    // PR #678 Bugbot 指摘（Medium）: 当初の実装は「`cp.async.wait_group N`
    // の N は PTX 上 0〜7 の即値レンジに制限される」という誤った ISA 上限
    // を根拠に `stages <= 9` を要求していたが、この主張は誤りだった
    // （LLVM NVPTX は `wait_group 8`／`16` 相当の即値も発行・検査しており、
    // ISA 側に 0〜7 という固定上限は存在しない）。本リポ自身も同じ
    // `wait_group (STAGES-2)` 段数一般形を使う `derive_pipeline_stages`
    // （`nvrtc.rs`）で [`MAX_PIPELINE_STAGES`]（16）を段数の有効上限として
    // 扱っており、7 という値はこの既存の合意と矛盾していた。誤った制約で
    // SMEM に収まる正当な構成を fail-close 側で弾いてしまっていたため、
    // 上限を [`MAX_PIPELINE_STAGES`] に統一する（レジスタ圧・命令数増加に
    // 見合わない段数を弾くという本来の目的は維持しつつ、ISA 上の誤った
    // 根拠を取り除く）。
    if cfg.stages > MAX_PIPELINE_STAGES {
        return Err(invalid(format!(
            "stages ({}) exceeds MAX_PIPELINE_STAGES ({MAX_PIPELINE_STAGES}); this upper bound \
             matches derive_pipeline_stages (nvrtc.rs) so that this staged kernel's stage-count \
             ceiling stays consistent with the rest of the crate",
            cfg.stages
        )));
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
    // cp.async の 16 バイト転送粒度は f32 4 要素であるため、協調ロード
    // マクロ（`LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP`）のチャンク分割式
    // `(BLOCK_M * K_TILE) / 4`／`(K_TILE * BLOCK_N) / 4` が割り切れる必要が
    // ある（[`WMMA_TF32_STAGED_A_PAD`]/[`WMMA_TF32_STAGED_B_PAD`] の const
    // アサーションと同じ根拠を可変 config でも検査する）。
    if !cfg.block_m.is_multiple_of(4)
        || !cfg.block_n.is_multiple_of(4)
        || !cfg.k_tile.is_multiple_of(4)
    {
        return Err(invalid(format!(
            "block_m ({}), block_n ({}), k_tile ({}) must all be multiples of 4 (cp.async \
             16-byte transfer granularity for f32 elements)",
            cfg.block_m, cfg.block_n, cfg.k_tile
        )));
    }
    // イシュー #743: a_pad/b_pad は config フィールド化されているため、
    // ここで独立に fail-closed 検査する（共通ヘルパの契約は
    // validate_wmma_tf32_staged_padding ドキュメンテーションコメント
    // 参照）。
    validate_wmma_tf32_staged_padding(cfg)?;

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

    let a_pad = cfg.a_pad;
    let b_pad = cfg.b_pad;

    // ステージ数分の as_tile/bs_tile 多段バッファ + エピローグ c_tile の
    // 静的共有メモリ合計（`docs/perf/cuda-gemm-wmma-tf32-phase-b.md`
    // 「SMEM 予算」節の試算式と同一）。
    let stage_bytes_a = cfg
        .stages
        .checked_mul(cfg.block_m)
        .and_then(|v| v.checked_mul(a_pad))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("A stage shared memory byte count overflow".to_string()))?;
    let stage_bytes_b = cfg
        .stages
        .checked_mul(cfg.k_tile)
        .and_then(|v| v.checked_mul(b_pad))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("B stage shared memory byte count overflow".to_string()))?;
    let c_tile_bytes = cfg
        .block_m
        .checked_mul(cfg.block_n)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("c_tile shared memory byte count overflow".to_string()))?;
    let smem_bytes = stage_bytes_a
        .checked_add(stage_bytes_b)
        .and_then(|v| v.checked_add(c_tile_bytes))
        .ok_or_else(|| invalid("shared memory byte count overflow".to_string()))?;
    if smem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES {
        return Err(invalid(format!(
            "static shared memory usage {smem_bytes} bytes exceeds the 48KiB per-block limit \
             (stages={}, block_m={}, block_n={}, k_tile={})",
            cfg.stages, cfg.block_m, cfg.block_n, cfg.k_tile
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

/// [`render_wmma_tf32_staged_unchecked`]／[`render_wmma_tf32_staged_dyn_unchecked`]
/// が共有するヘッダ展開本体。`dynamic_smem`（`WMMA_TF32_STAGED_DYNAMIC_SMEM`
/// の値。0 = static `__shared__`〈既定・本番経路〉、1 = 動的共有メモリ
/// 〈`internal-diagnostics` 計測専用〉）のみが両呼び出し元の差分（イシュー
/// #742）。0 側は本 define の追加を除き従来の展開結果と byte 完全一致で
/// あり、[`WMMA_TF32_F32_STAGED_BODY`] 側の `#if` 分岐が static 宣言へ
/// フォールバックする（本番経路 parity への影響なし。
/// `docs/perf/cuda-parity-baseline.md`）。
fn render_wmma_tf32_staged_header(cfg: &WmmaTf32StagedKernelConfig, dynamic_smem: u32) -> String {
    let warp_grid_n = cfg.block_n / WMMA_TF32_STAGED_WARP_TILE;
    let threads = (cfg.block_m / WMMA_TF32_STAGED_WARP_TILE) * warp_grid_n * 32;
    // イシュー #743: パディング幅は cfg フィールド（呼び出し元が
    // validate_wmma_tf32_staged_padding で検査済み）から直接取る。
    // 本番経路（default_tf32_staged）では WMMA_TF32_STAGED_A_PAD/B_PAD と
    // 一致するため展開結果は従来と byte 完全一致のまま。
    let a_pad = cfg.a_pad;
    let b_pad = cfg.b_pad;
    let k_substeps = cfg.k_tile / WMMA_TF32_STAGED_FRAG_K;
    let dim_m_define = render_dim_define("DIM_M", "m", cfg.dim_m);
    let dim_n_define = render_dim_define("DIM_N", "n", cfg.dim_n);
    let dim_k_define = render_dim_define("DIM_K", "k", cfg.dim_k);

    format!(
        "\n#include <mma.h>\n\n\
         using namespace nvcuda;\n\n\
         #define WMMA_TF32_STAGED_BLOCK_M {block_m}\n\
         #define WMMA_TF32_STAGED_BLOCK_N {block_n}\n\
         #define WMMA_TF32_STAGED_K_TILE {k_tile}\n\
         #define WMMA_TF32_STAGED_FRAG {frag}\n\
         #define WMMA_TF32_STAGED_FRAG_K {frag_k}\n\
         #define WMMA_TF32_STAGED_WARP_TILE {warp_tile}\n\
         #define WMMA_TF32_STAGED_THREADS {threads}\n\
         #define WMMA_TF32_STAGED_A_PAD {a_pad}\n\
         #define WMMA_TF32_STAGED_B_PAD {b_pad}\n\
         #define WMMA_TF32_STAGED_FRAG_ROWS 2\n\
         #define WMMA_TF32_STAGED_FRAG_COLS 2\n\
         #define WMMA_TF32_STAGED_K_SUBSTEPS {k_substeps}\n\
         #define WMMA_TF32_STAGED_WARP_GRID_N {warp_grid_n}\n\
         #define WMMA_TF32_STAGED_STAGES {stages}\n\
         #define WMMA_TF32_STAGED_DYNAMIC_SMEM {dynamic_smem}\n\
         {dim_m_define}\n\
         {dim_n_define}\n\
         {dim_k_define}\n\
         \n{WMMA_TF32_F32_STAGED_BODY}",
        block_m = cfg.block_m,
        block_n = cfg.block_n,
        k_tile = cfg.k_tile,
        frag = WMMA_TF32_STAGED_FRAG,
        frag_k = WMMA_TF32_STAGED_FRAG_K,
        warp_tile = WMMA_TF32_STAGED_WARP_TILE,
        stages = cfg.stages,
    )
}

fn render_wmma_tf32_staged_unchecked(cfg: &WmmaTf32StagedKernelConfig) -> String {
    render_wmma_tf32_staged_header(cfg, 0)
}

/// [`WmmaTf32StagedKernelConfig`] を TF32 opt-staged カーネルソースへ展開
/// する（イシュー #500。[`render_wmma_tf32_opt`] と同じ設計）。展開前に
/// [`validate_wmma_tf32_staged_config`] で SMEM 予算・倍数関係・スレッド数
/// 上限を fail-closed 検査する。
#[allow(dead_code)]
pub fn render_wmma_tf32_staged(
    cfg: &WmmaTf32StagedKernelConfig,
) -> Result<RenderedWmmaTf32StagedKernel, CudaError> {
    validate_wmma_tf32_staged_config(cfg)?;
    Ok(RenderedWmmaTf32StagedKernel {
        source: render_wmma_tf32_staged_unchecked(cfg),
        cfg: *cfg,
    })
}

/// TF32 opt-staged 動的共有メモリ変種（`WMMA_TF32_STAGED_DYNAMIC_SMEM=1`）
/// が要求する共有メモリバイト数を計算する単一ソース（イシュー #742）。
///
/// [`WMMA_TF32_F32_STAGED_BODY`] の dyn 分岐は c_tile をステージバッファ
/// 先頭へエイリアスする（同分岐のコメント参照）ため、所要量は
/// `max(stages 段の as_tile+bs_tile 合計, c_tile)` であり、単純合算
/// （static 側 [`validate_wmma_tf32_staged_config`] の
/// `stage_bytes_a + stage_bytes_b + c_tile_bytes`）とは異なる。
/// [`validate_wmma_tf32_staged_dyn_config`]（起動前検証）と
/// `examples/gemm_wmma_tf32_staged_stages_bench.rs`（occupancy 概算表示。
/// `internal-diagnostics` feature 配下の `diagnostics` モジュール経由）の
/// 両方がこの関数を単一ソースとして呼ぶ。
#[allow(dead_code)]
pub fn wmma_tf32_staged_dyn_smem_bytes(cfg: &WmmaTf32StagedKernelConfig) -> Result<u64, CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };
    // イシュー #743: パディング幅は cfg フィールドから直接取る（本番既定
    // 構成では従来の k_tile+4/block_n+4 と同じ値になる）。
    let a_pad = u64::from(cfg.a_pad);
    let b_pad = u64::from(cfg.b_pad);

    let stage_bytes_a = u64::from(cfg.stages)
        .checked_mul(u64::from(cfg.block_m))
        .and_then(|v| v.checked_mul(a_pad))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("A stage shared memory byte count overflow (dyn)".to_string()))?;
    let stage_bytes_b = u64::from(cfg.stages)
        .checked_mul(u64::from(cfg.k_tile))
        .and_then(|v| v.checked_mul(b_pad))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("B stage shared memory byte count overflow (dyn)".to_string()))?;
    let stage_bytes_total = stage_bytes_a
        .checked_add(stage_bytes_b)
        .ok_or_else(|| invalid("stage shared memory byte count overflow (dyn)".to_string()))?;
    let c_tile_bytes = u64::from(cfg.block_m)
        .checked_mul(u64::from(cfg.block_n))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| invalid("c_tile shared memory byte count overflow (dyn)".to_string()))?;

    Ok(stage_bytes_total.max(c_tile_bytes))
}

/// [`validate_wmma_tf32_staged_config`] と同じ形状検査（stages 範囲・warp
/// タイル整合・cp.async 4 要素整列・スレッド数上限）を課すが、共有メモリ
/// 予算検査だけを static 48KiB 固定
/// （[`crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`]）ではなく
/// 呼び出し元が指定する `optin_budget_bytes`（デバイス実測の
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`。
/// `device.rs::shared_memory_per_block_optin`）に対して行う（イシュー
/// #742）。
///
/// static 側の [`validate_wmma_tf32_staged_config`] は変更しない
/// （本番経路の 48KiB fail-closed 拒否契約を保つため。実装計画 §1.1）。
/// 検査に成功した場合、算出した動的共有メモリバイト数を返す
/// （呼び出し元が [`RenderedWmmaTf32StagedDynKernel`] へ格納し、compile
/// 時の opt-in 属性設定・launch 時の `shared_mem_bytes` に再利用する）。
#[allow(dead_code)]
fn validate_wmma_tf32_staged_dyn_config(
    cfg: &WmmaTf32StagedKernelConfig,
    optin_budget_bytes: u32,
) -> Result<u64, CudaError> {
    let invalid = |detail: String| CudaError::InvalidKernelConfig { detail };
    let warp_tile = WMMA_TF32_STAGED_WARP_TILE;
    let frag_k = WMMA_TF32_STAGED_FRAG_K;

    if cfg.block_m == 0 || cfg.block_n == 0 || cfg.k_tile == 0 {
        return Err(invalid(
            "block_m/block_n/k_tile must all be non-zero".to_string(),
        ));
    }
    // [`WMMA_TF32_STAGED_STAGES`] 直下コメントと同じ根拠
    // （cp.async.wait_group (STAGES-2) の即値非負制約）。
    if cfg.stages < 2 {
        return Err(invalid(format!(
            "stages ({}) must be >= 2 (cp.async wait_group (STAGES-2) immediate must be \
             non-negative)",
            cfg.stages
        )));
    }
    // static 側と同じ理由（[`validate_wmma_tf32_staged_config`] 該当コメント
    // 参照）で `derive_pipeline_stages`（`nvrtc.rs`）と同じ上限を適用する。
    if cfg.stages > MAX_PIPELINE_STAGES {
        return Err(invalid(format!(
            "stages ({}) exceeds MAX_PIPELINE_STAGES ({MAX_PIPELINE_STAGES})",
            cfg.stages
        )));
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
    if !cfg.block_m.is_multiple_of(4)
        || !cfg.block_n.is_multiple_of(4)
        || !cfg.k_tile.is_multiple_of(4)
    {
        return Err(invalid(format!(
            "block_m ({}), block_n ({}), k_tile ({}) must all be multiples of 4 (cp.async \
             16-byte transfer granularity for f32 elements)",
            cfg.block_m, cfg.block_n, cfg.k_tile
        )));
    }
    // イシュー #743: static 側と同じ共通ヘルパで a_pad/b_pad を検査する
    // （validate_wmma_tf32_staged_padding ドキュメンテーションコメント
    // 参照）。
    validate_wmma_tf32_staged_padding(cfg)?;

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

    let smem_bytes = wmma_tf32_staged_dyn_smem_bytes(cfg)?;
    if smem_bytes > u64::from(optin_budget_bytes) {
        return Err(invalid(format!(
            "dynamic shared memory usage {smem_bytes} bytes exceeds the device opt-in budget \
             ({optin_budget_bytes} bytes; CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) \
             (stages={}, block_m={}, block_n={}, k_tile={})",
            cfg.stages, cfg.block_m, cfg.block_n, cfg.k_tile
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

    Ok(smem_bytes)
}

fn render_wmma_tf32_staged_dyn_unchecked(cfg: &WmmaTf32StagedKernelConfig) -> String {
    render_wmma_tf32_staged_header(cfg, 1)
}

/// [`WmmaTf32StagedKernelConfig`] を TF32 opt-staged カーネルの**動的共有
/// メモリ変種**ソースへ展開する（イシュー #742。`internal-diagnostics`
/// feature 配下の段数スイープ計測専用。[`render_wmma_tf32_staged`] の
/// 本番経路には一切関与しない）。展開前に
/// [`validate_wmma_tf32_staged_dyn_config`] で `optin_budget_bytes` に
/// 対する動的 SMEM 予算・倍数関係・スレッド数上限を fail-closed 検査する。
#[allow(dead_code)]
pub fn render_wmma_tf32_staged_dyn(
    cfg: &WmmaTf32StagedKernelConfig,
    optin_budget_bytes: u32,
) -> Result<RenderedWmmaTf32StagedDynKernel, CudaError> {
    let smem_bytes = validate_wmma_tf32_staged_dyn_config(cfg, optin_budget_bytes)?;
    Ok(RenderedWmmaTf32StagedDynKernel {
        source: render_wmma_tf32_staged_dyn_unchecked(cfg),
        cfg: *cfg,
        smem_bytes,
    })
}

/// [`render_wmma_tf32_staged_dyn`] が返す、展開済み動的 SMEM 変種ソース・
/// 展開元 config・算出済み共有メモリバイト数を 1 個にまとめた descriptor
/// （[`RenderedWmmaTf32StagedKernel`] と同じ設計。イシュー #742）。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RenderedWmmaTf32StagedDynKernel {
    source: String,
    cfg: WmmaTf32StagedKernelConfig,
    smem_bytes: u64,
}

impl RenderedWmmaTf32StagedDynKernel {
    /// [`RenderedWmmaTf32StagedKernel::compile`] と同じ設計だが、算出済み
    /// `smem_bytes` が static 48KiB 上限を超える場合のみ
    /// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` を opt-in 設定する
    /// （cudarc 0.19.8 の安全 API
    /// `CudaFunction::set_attribute`〈`unsafe` を要求しない〉。48KiB 以下
    /// では opt-in 不要かつ `cuFuncSetAttribute` が許容範囲内の値をそのまま
    /// 受理するため、常時呼んでも安全だが、48KiB 以下でも常に呼ぶと
    /// 「デバイス既定を明示的に狭めている」ように読めるため必要時のみに
    /// 限定する）。
    #[allow(dead_code)]
    pub fn compile(
        &self,
        device: &crate::device::CudaDevice,
    ) -> Result<CompiledWmmaTf32StagedDynKernel, CudaError> {
        let ptx = crate::nvrtc::compile_ptx(&self.source, device.arch())?;
        let func = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_wmma_tf32_staged")?;
        if self.smem_bytes > u64::from(MMA_STATIC_SMEM_LIMIT_BYTES) {
            let bytes_i32 =
                i32::try_from(self.smem_bytes).map_err(|_| CudaError::InvalidKernelConfig {
                    detail: format!(
                        "dynamic shared memory byte count {} exceeds i32 range required by \
                         cuFuncSetAttribute",
                        self.smem_bytes
                    ),
                })?;
            func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes_i32,
            )
            .map_err(CudaError::from)?;
        }
        Ok(CompiledWmmaTf32StagedDynKernel {
            func,
            cfg: self.cfg,
            smem_bytes: self.smem_bytes,
        })
    }

    #[cfg(test)]
    fn source(&self) -> &str {
        &self.source
    }
}

/// [`RenderedWmmaTf32StagedDynKernel::compile`] が返す、コンパイル済み
/// `CudaFunction`・展開元 config・動的共有メモリバイト数を不可分に束ねた
/// descriptor（イシュー #742）。
#[allow(dead_code)]
pub struct CompiledWmmaTf32StagedDynKernel {
    func: cudarc::driver::CudaFunction,
    cfg: WmmaTf32StagedKernelConfig,
    smem_bytes: u64,
}

impl CompiledWmmaTf32StagedDynKernel {
    /// [`CompiledWmmaTf32StagedKernel::launch_tf32_staged`] と同じ検証・
    /// 起動手順に加え、`LaunchConfig.shared_mem_bytes` へ算出済み
    /// `smem_bytes` を設定する（static 変種は常に 0）。
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn launch_tf32_staged_dyn(
        &self,
        stream: &CudaStream,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        self.cfg.validate_launch_shape(m, n, k)?;
        crate::gemm::validate_gemm_dims(a_dev.len(), b_dev.len(), m, n, k)?;
        crate::gemm::validate_output_len(c_dev.len(), m, n)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if !crate::gemm::wmma_tf32_staged_alignment_ok(n, k) {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "wmma tf32 staged (dyn) path requires n % 4 == 0 && k % 4 == 0 (cp.async \
                     16-byte transfer granularity), but got n={n}, k={k}"
                ),
            });
        }

        // static 変種 CompiledWmmaTf32StagedKernel と同一の grid/k タイル
        // 境界検査（validate_grid_bounds／validate_k_tile_bound と同じ
        // ロジック・同じ理由）。dyn 変種は独立 struct のため、汎用実装
        // `validate_wmma_opt_k_tile_bound` を直接呼ぶ。
        const MAX_GRID_DIM_Y: u32 = 65_535;
        let grid_y = m.div_ceil(self.cfg.block_m);
        if grid_y > MAX_GRID_DIM_Y {
            return Err(CudaError::InvalidShape {
                detail: format!(
                    "wmma staged (dyn) path grid_dim.y ({grid_y}) exceeds CUDA's \
                     {MAX_GRID_DIM_Y} limit for grid dimensions y/z (block_m={}); m={m} is \
                     too large",
                    self.cfg.block_m
                ),
            });
        }
        validate_wmma_opt_k_tile_bound(k, self.cfg.k_tile)?;

        let smem_bytes_u32 =
            u32::try_from(self.smem_bytes).map_err(|_| CudaError::InvalidKernelConfig {
                detail: format!(
                    "dynamic shared memory byte count {} exceeds u32 range required by \
                     LaunchConfig.shared_mem_bytes",
                    self.smem_bytes
                ),
            })?;
        let warp_grid_m = self.cfg.block_m / WMMA_TF32_STAGED_WARP_TILE;
        let warp_grid_n = self.cfg.block_n / WMMA_TF32_STAGED_WARP_TILE;
        let launch_config = LaunchConfig {
            grid_dim: (
                n.div_ceil(self.cfg.block_n),
                m.div_ceil(self.cfg.block_m),
                1,
            ),
            block_dim: (warp_grid_m * warp_grid_n * 32, 1, 1),
            shared_mem_bytes: smem_bytes_u32,
        };
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

        // SAFETY: CompiledWmmaTf32StagedKernel::launch_tf32_staged と同一の
        // 根拠。カーネル引数は上記で検証済みの m/n/k から導出しており、
        // opt-staged カーネル内の手動境界チェック（guarded load・エピローグ
        // store のガード付きコピー。REQ-8）と合わせて OOB 読み書きが起きない
        // 根拠とする。`shared_mem_bytes` は `RenderedWmmaTf32StagedDynKernel::compile`
        // が算出・opt-in 設定した値と同一（`self.smem_bytes`）であり、
        // カーネル側 `extern __shared__` の実際の使用量を過不足なく満たす。
        unsafe {
            stream
                .launch_builder(&self.func)
                .arg(a_dev)
                .arg(b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(launch_config)?;
        }
        // 非同期投入契約（#1013）。完了保証は呼び出し元の次の同期点
        // （`download_*`／`MemoryOps::download`／明示 `synchronize`）へ
        // 委ねる（設計文書 §3〜§4）。
        Ok(())
    }
}

/// [`render_wmma_tf32_staged_unchecked`]／[`wmma_tf32_f32_staged_source`]
/// が結合するカーネル本体テンプレート。
///
/// `kernels_mma.rs::MMA_F16_BODY`（cp.async 3 ステージ・issue
/// interleaving・ldmatrix 先読みダブルバッファ）の構造を、WMMA C++ API
/// （`nvcuda::wmma`）の TF32 経路へそのまま移植したもの。差分は
/// (1) グローバル→共有メモリのロード粒度が f16 8 要素/16B ではなく f32
/// 4 要素/16B である点、(2) `ldmatrix` の代わりに `wmma::load_matrix_sync`
/// を使い、ロード直後に `wmma::__float_to_tf32` による明示変換を適用する
/// 点（既存 `WMMA_TF32_F32_OPT_BODY` と同一の数値契約。2 面バッファの
/// どちらの呼び出しからもこの変換を経由するため、先読みバッファに変換
/// 漏れが生じない）のみで、cp.async 段数管理・commit/wait 配置・issue
/// interleaving の骨格は f16 版と同一の t/stage 添字算術を使う（正しさの
/// 論証も同一。本ファイル [`WMMA_TF32_STAGED_STAGES`] 定数直下コメント
/// 参照）。
///
/// **`_Pragma` 不使用の方針**: `kernels_mma.rs::MMA_F16_BODY` 冒頭
/// ドキュメンテーションコメント「マクロを『1 フラグメント単位』に留め」
/// と同じ理由（本ファイルは NVRTC 構文検証不能環境で書いており、
/// `_Pragma` 演算子は本ファイル・`kernels_mma.rs` のいずれにも前例がなく
/// NVRTC 上での挙動を実機なしで確認できない）で、fragment ロード＋TF32
/// 変換マクロ（`LDWM_A_FRAG`/`LDWM_B_FRAG`）内では `#pragma unroll` を
/// 使わない（`num_elements` 反復のみの小ループを展開しないだけで、正しさ
/// には影響しない）。`#pragma unroll` は既存 `WMMA_TF32_F32_OPT_BODY` と
/// 同じく、プリプロセッサ済みの実際の文出現位置（マクロ呼び出し側の
/// `fi`/`fj`/`ks` ループ）にのみ置く。
const WMMA_TF32_F32_STAGED_BODY: &str = r#"
// イシュー #500: グローバル→共有メモリの 16 バイト単位（f32 4 要素）
// 非同期コピー。src_size==16 で実データをコピーし、src_size==0 で共有
// メモリ側をゼロ充填する（REQ-8 境界検査。kernels_mma.rs::mma_cp_async16
// と同じ契約・同じ PTX 命令。関数名は同一 NVRTC コンパイル単位内での
// 衝突を避けるため本カーネル専用の接頭辞を付す）。
__device__ __forceinline__ void wmma_tf32_staged_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_size)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

extern "C" __global__ void gemm_wmma_tf32_staged(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // __align__(16): cp.async の 16 バイト転送先整列要件（本ファイル冒頭
    // ドキュメントコメント「アライメント」・kernels_mma.rs 冒頭コメント
    // 「整列制約」と同じ根拠）。A_PAD/B_PAD が 4 要素（16 バイト）の倍数
    // のため各行の先頭は常に整列する。
    //
    // イシュー #742: `WMMA_TF32_STAGED_DYNAMIC_SMEM` は既定 0（static
    // `__shared__`。本番経路〈gemm.rs 3 段フォールバック選択〉が使う唯一の
    // 分岐で、宣言は本行以下 static 側のみで従来と byte 完全一致）。
    // `render_wmma_tf32_staged_dyn`（計測専用。`internal-diagnostics`
    // feature 配下）のみが 1 を渡す。dyn 側は動的共有メモリ（opt-in 属性で
    // 48KiB を超える割り当てが可能。`device.rs::shared_memory_per_block_optin`）
    // を `extern __shared__` 経由で受け取り、`as_tile`/`bs_tile` へ
    // reinterpret_cast する。stages 段数を static 48KiB 制限（既定タイルで
    // stages<=3）を超えて計測するための変種であり、`validate_wmma_tf32_staged_config`
    // （static 48KiB 検査）は dyn 側には適用しない
    // （`wmma_tf32_staged_dyn_smem_bytes` が動的側の予算検査を別途行う）。
#if WMMA_TF32_STAGED_DYNAMIC_SMEM
    // __align__(32): as_tile/bs_tile としての利用は 16 バイト整列で足りるが、
    // 後段でこの領域先頭を c_tile としてエイリアスし
    // `wmma::store_matrix_sync` へ渡す（本関数末尾のエピローグ節参照）。
    // WMMA の load/store 対象ポインタは 256-bit（32 バイト）整列が
    // 契約であり（static 側 `c_tile` の `__align__(32)` と同じ根拠）、
    // 動的 extern __shared__ の整列指定はこの中で最も厳しい要求に
    // 揃える必要があるため 32 バイトを指定する。
    extern __shared__ __align__(32) unsigned char wmma_tf32_staged_smem[];
    float (*as_tile)[WMMA_TF32_STAGED_BLOCK_M][WMMA_TF32_STAGED_A_PAD] =
        reinterpret_cast<float(*)[WMMA_TF32_STAGED_BLOCK_M][WMMA_TF32_STAGED_A_PAD]>(wmma_tf32_staged_smem);
    float (*bs_tile)[WMMA_TF32_STAGED_K_TILE][WMMA_TF32_STAGED_B_PAD] =
        reinterpret_cast<float(*)[WMMA_TF32_STAGED_K_TILE][WMMA_TF32_STAGED_B_PAD]>(
            wmma_tf32_staged_smem
            + (size_t)WMMA_TF32_STAGED_STAGES * WMMA_TF32_STAGED_BLOCK_M * WMMA_TF32_STAGED_A_PAD * sizeof(float));
#else
    __shared__ __align__(16) float as_tile[WMMA_TF32_STAGED_STAGES][WMMA_TF32_STAGED_BLOCK_M][WMMA_TF32_STAGED_A_PAD];
    __shared__ __align__(16) float bs_tile[WMMA_TF32_STAGED_STAGES][WMMA_TF32_STAGED_K_TILE][WMMA_TF32_STAGED_B_PAD];
#endif

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WMMA_TF32_STAGED_WARP_GRID_N;
    const int warp_col = warp_id % WMMA_TF32_STAGED_WARP_GRID_N;

    const int block_row_base = blockIdx.y * WMMA_TF32_STAGED_BLOCK_M;
    const int block_col_base = blockIdx.x * WMMA_TF32_STAGED_BLOCK_N;
    const int warp_row_base = warp_row * WMMA_TF32_STAGED_WARP_TILE;
    const int warp_col_base = warp_col * WMMA_TF32_STAGED_WARP_TILE;

    // レジスタブロッキング（B-2、適用済み）: warp あたり 2x2 = 4 個の
    // accumulator fragment。既存 TF32 opt と同一構成。
    wmma::fragment<wmma::accumulator, WMMA_TF32_STAGED_FRAG, WMMA_TF32_STAGED_FRAG,
                   WMMA_TF32_STAGED_FRAG_K, float> c_frag[WMMA_TF32_STAGED_FRAG_ROWS][WMMA_TF32_STAGED_FRAG_COLS];
#pragma unroll
    for (int fi = 0; fi < WMMA_TF32_STAGED_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_STAGED_FRAG_COLS; ++fj) {
            wmma::fill_fragment(c_frag[fi][fj], 0.0f);
        }
    }

    // 桁溢れしない num_k_tiles 計算（既存 TF32 opt と同一式）。
    int num_k_tiles = (DIM_K > 0) ? (DIM_K - 1) / WMMA_TF32_STAGED_K_TILE + 1 : 0;

    // イシュー #500: cp.async 発行を K サブステップへ分散するための添字
    // 空間分割（kernels_mma.rs::MMA_F16_BODY「#496」節と同じ設計。
    // K_GROUPS は下記 ks ループの反復回数（WMMA_TF32_STAGED_K_SUBSTEPS）と
    // 必ず一致する）。
    #define K_GROUPS (WMMA_TF32_STAGED_K_TILE / WMMA_TF32_STAGED_FRAG_K)
    #define A_CHUNKS ((WMMA_TF32_STAGED_BLOCK_M * WMMA_TF32_STAGED_K_TILE) / 4)
    #define B_CHUNKS ((WMMA_TF32_STAGED_K_TILE * WMMA_TF32_STAGED_BLOCK_N) / 4)
    #define A_GROUP_CHUNKS ((A_CHUNKS + K_GROUPS - 1) / K_GROUPS)
    #define B_GROUP_CHUNKS ((B_CHUNKS + K_GROUPS - 1) / K_GROUPS)

    // REQ-8: A/B タイルを stage へ非同期ロードするマクロ。gr_c/gc_c は
    // 境界外チャンクでも 16 バイト整列を保ったままクランプする
    // （kernels_mma.rs::MMA_F16_BODY「#496」節直上コメント「REQ-8 追補」と
    // 同じ理由・同じ式。列方向は 4 要素境界〈f32 4 要素 = 16 バイト〉に
    // 切り下げる点のみ f16 版〈8 要素境界〉と異なる。行ストライド
    // （A は k・B は n）が 4 の倍数であることは `gemm.rs` 側の起動前
    // 整列検証〈cp.async 16 バイト整列条件〉が保証する）。
    #define LOAD_A_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * A_GROUP_CHUNKS + tid; \
             idx < A_CHUNKS && idx < ((g) + 1) * A_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (WMMA_TF32_STAGED_K_TILE / 4); \
            int col0 = (idx % (WMMA_TF32_STAGED_K_TILE / 4)) * 4; \
            int gr = block_row_base + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < DIM_M ? gr : (DIM_M > 0 ? DIM_M - 1 : 0); \
            int gc_c = gc < DIM_K ? gc : (DIM_K > 0 ? ((DIM_K - 1) / 4) * 4 : 0); \
            int valid = (gr < DIM_M && gc < DIM_K) ? 16 : 0; \
            wmma_tf32_staged_cp_async16(&as_tile[stage][row][col0], &a[(size_t)gr_c * DIM_K + gc_c], valid); \
        }

    #define LOAD_B_STAGE_GROUP(stage, k0, g) \
        for (int idx = (g) * B_GROUP_CHUNKS + tid; \
             idx < B_CHUNKS && idx < ((g) + 1) * B_GROUP_CHUNKS; \
             idx += num_threads) { \
            int row = idx / (WMMA_TF32_STAGED_BLOCK_N / 4); \
            int col0 = (idx % (WMMA_TF32_STAGED_BLOCK_N / 4)) * 4; \
            int gr = (k0) + row; \
            int gc = block_col_base + col0; \
            int gr_c = gr < DIM_K ? gr : (DIM_K > 0 ? DIM_K - 1 : 0); \
            int gc_c = gc < DIM_N ? gc : (DIM_N > 0 ? ((DIM_N - 1) / 4) * 4 : 0); \
            int valid = (gr < DIM_K && gc < DIM_N) ? 16 : 0; \
            wmma_tf32_staged_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * DIM_N + gc_c], valid); \
        }

    // プロローグ（下記）は K タイル 1 段分をまとめてロードする必要が
    // あるため、上記 2 マクロを全グループについて呼ぶ薄いラッパーとして
    // 再定義する（kernels_mma.rs::MMA_F16_BODY「#496」節と同じ設計）。
    #define LOAD_A_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_A_STAGE_GROUP(stage, k0, g_); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int g_ = 0; g_ < K_GROUPS; ++g_) { \
            LOAD_B_STAGE_GROUP(stage, k0, g_); \
        }

    // プロローグ: 最初の STAGES-1 タイルをロードし、それぞれ独立した
    // cp.async グループとして commit する（kernels_mma.rs::MMA_F16_BODY
    // プロローグと同一の「1 イテレーション = 必ず 1 commit」不変条件。
    // #492 の論証をそのまま踏襲する）。
    for (int s = 0; s < WMMA_TF32_STAGED_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            LOAD_A_STAGE(s, s * WMMA_TF32_STAGED_K_TILE);
            LOAD_B_STAGE(s, s * WMMA_TF32_STAGED_K_TILE);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % WMMA_TF32_STAGED_STAGES;
        int next_tile = t + WMMA_TF32_STAGED_STAGES - 1;
        int load_stage = next_tile % WMMA_TF32_STAGED_STAGES;

        // kernels_mma.rs::MMA_F16_BODY「#492」節と同一の段数一般形固定
        // 即値（`STAGES - 2`）・同一の正しさ論証（本ファイル
        // WMMA_TF32_STAGED_STAGES 定数直下コメント参照）。
        // 正しさ: プロローグの無条件 commit により、イテレーション t の
        // 時点での commit 総数は常に `(STAGES-1) + t`。`wait_group
        // (STAGES-2)` は未完了グループ数 <= STAGES-2 を保証するため、
        // 完了数 >= t+1、すなわちタイル t のグループの完了が全 t で
        // 保証される。
        asm volatile("cp.async.wait_group %0;\n" ::"n"(WMMA_TF32_STAGED_STAGES - 2));
        __syncthreads();

        // fragment 2 面バッファ（cur/nxt）。kernels_mma.rs::MMA_F16_BODY
        // 「#495」節と同じ「タイル内限定・クロスタイル先読みは不採用」
        // 設計（理由も同一: ループ内 wait_group はタイル t 自身のグループ
        // 完了しか保証しないため、タイル境界を跨ぐ先読みは wait/sync
        // 配置の大規模再構成を要し、NVRTC 構文検証不能な本環境ではリスク
        // が高い）。
        wmma::fragment<wmma::matrix_a, WMMA_TF32_STAGED_FRAG, WMMA_TF32_STAGED_FRAG,
                       WMMA_TF32_STAGED_FRAG_K, wmma::precision::tf32, wmma::row_major>
            a_frag[2][WMMA_TF32_STAGED_FRAG_ROWS];
        wmma::fragment<wmma::matrix_b, WMMA_TF32_STAGED_FRAG, WMMA_TF32_STAGED_FRAG,
                       WMMA_TF32_STAGED_FRAG_K, wmma::precision::tf32, wmma::row_major>
            b_frag[2][WMMA_TF32_STAGED_FRAG_COLS];

        // フラグメント 1 個を buf 面へロードし、TF32 明示変換を直後に
        // 適用するマクロ（既存 `WMMA_TF32_F32_OPT_BODY` と同一の数値
        // 契約。2 面バッファのどちらの呼び出しからもこのマクロを経由する
        // ため、先読みバッファに変換漏れが生じない）。`_Pragma` 不使用の
        // 方針は本ファイルこのテンプレート冒頭ドキュメンテーション
        // コメント参照（呼び出し側で `#pragma unroll` を付す）。
        #define LDWM_A_FRAG(buf, stage, kstep, fi) \
            do { \
                wmma::load_matrix_sync( \
                    a_frag[buf][fi], \
                    &as_tile[stage][warp_row_base + (fi) * WMMA_TF32_STAGED_FRAG][(kstep) * WMMA_TF32_STAGED_FRAG_K], \
                    WMMA_TF32_STAGED_A_PAD); \
                for (int e = 0; e < a_frag[buf][fi].num_elements; ++e) { \
                    a_frag[buf][fi].x[e] = wmma::__float_to_tf32(a_frag[buf][fi].x[e]); \
                } \
            } while (0)

        #define LDWM_B_FRAG(buf, stage, kstep, fj) \
            do { \
                wmma::load_matrix_sync( \
                    b_frag[buf][fj], \
                    &bs_tile[stage][(kstep) * WMMA_TF32_STAGED_FRAG_K][warp_col_base + (fj) * WMMA_TF32_STAGED_FRAG], \
                    WMMA_TF32_STAGED_B_PAD); \
                for (int e = 0; e < b_frag[buf][fj].num_elements; ++e) { \
                    b_frag[buf][fj].x[e] = wmma::__float_to_tf32(b_frag[buf][fj].x[e]); \
                } \
            } while (0)

        // warp プロローグ: kstep=0 のフラグメントをバッファ 0 へロードして
        // から ks ループへ入る（kernels_mma.rs #495 warp プロローグと同型）。
#pragma unroll
        for (int fi = 0; fi < WMMA_TF32_STAGED_FRAG_ROWS; ++fi) {
            LDWM_A_FRAG(0, compute_stage, 0, fi);
        }
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_STAGED_FRAG_COLS; ++fj) {
            LDWM_B_FRAG(0, compute_stage, 0, fj);
        }

        // K_SUBSTEPS は `#define` 由来のコンパイル時定数式のためトリップ
        // 回数は既知であり、`#pragma unroll` により cur/nxt がコンパイル
        // 時定数へ畳み込まれる（kernels_mma.rs #495 kstep ループと同じ
        // 必須の pragma。cosmetic ではない）。
#pragma unroll
        for (int ks = 0; ks < WMMA_TF32_STAGED_K_SUBSTEPS; ++ks) {
            int cur = ks % 2;
            int nxt = (ks + 1) % 2;

            // 次段（ks+1）のフラグメントを、現在段（ks）の mma_sync 発行前
            // に先読みしてバッファ nxt へロードする（kernels_mma.rs #495
            // と同じソフトウェアパイプライン化）。
            if (ks + 1 < WMMA_TF32_STAGED_K_SUBSTEPS) {
#pragma unroll
                for (int fi = 0; fi < WMMA_TF32_STAGED_FRAG_ROWS; ++fi) {
                    LDWM_A_FRAG(nxt, compute_stage, ks + 1, fi);
                }
#pragma unroll
                for (int fj = 0; fj < WMMA_TF32_STAGED_FRAG_COLS; ++fj) {
                    LDWM_B_FRAG(nxt, compute_stage, ks + 1, fj);
                }
            }

            // イシュー #500: cp.async issue interleaving
            // （kernels_mma.rs「#496」節と同旨）。K_GROUPS ==
            // K_SUBSTEPS のため全グループが ks ループで過不足なく発行
            // される。同期の正しさ論証も #496 節と同一（発行先
            // load_stage は本イテレーションの ldmatrix 相当
            // 〈load_matrix_sync〉から一切読まれない。load_stage !=
            // compute_stage は STAGES>=2 で常に成立）。
            if (next_tile < num_k_tiles) {
                LOAD_A_STAGE_GROUP(load_stage, next_tile * WMMA_TF32_STAGED_K_TILE, ks);
                LOAD_B_STAGE_GROUP(load_stage, next_tile * WMMA_TF32_STAGED_K_TILE, ks);
            }

#pragma unroll
            for (int fi = 0; fi < WMMA_TF32_STAGED_FRAG_ROWS; ++fi) {
#pragma unroll
                for (int fj = 0; fj < WMMA_TF32_STAGED_FRAG_COLS; ++fj) {
                    wmma::mma_sync(c_frag[fi][fj], a_frag[cur][fi], b_frag[cur][fj], c_frag[fi][fj]);
                }
            }
        }

        #undef LDWM_A_FRAG
        #undef LDWM_B_FRAG

        // #492/#496 と同じ「1 イテレーション = 必ず 1 commit」不変条件。
        asm volatile("cp.async.commit_group;\n");
        __syncthreads();
    }

    // #492 節と同じループ外 drain。ループ内固定即値 wait は最終タイル
    // 直前までしか保証しないため、抜けた直後に残存グループを
    // `wait_group 0` で掃き出してから読み出す。
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE
    #undef LOAD_A_STAGE_GROUP
    #undef LOAD_B_STAGE_GROUP
    #undef A_GROUP_CHUNKS
    #undef B_GROUP_CHUNKS
    #undef A_CHUNKS
    #undef B_CHUNKS
    #undef K_GROUPS

    // REQ-8: エピローグ store のガード条件。既存 TF32 opt
    // （WMMA_TF32_F32_OPT_BODY）エピローグと同一の guarded store 方式。
    //
    // イシュー #742 dyn 側: c_tile を `wmma_tf32_staged_smem` 先頭へ
    // エイリアスする（動的 SMEM 所要を
    // `max(stages 段バッファ, c_tile)` に抑え、stages=10 でも GB10 の
    // optin 予算 101,376B 以内に収める設計。
    // `docs/perf/cuda-gemm-wmma-tf32-staged-stages-sweep.md` 参照）。直上の
    // `cp.async.wait_group 0; __syncthreads();` により、このエイリアス化の
    // 時点で as_tile/bs_tile はどのスレッドからも二度と読まれないため、
    // 同一領域への上書きは安全（本コメント直上のループ外 drain 節参照）。
#if WMMA_TF32_STAGED_DYNAMIC_SMEM
    float (*c_tile)[WMMA_TF32_STAGED_BLOCK_N] =
        reinterpret_cast<float(*)[WMMA_TF32_STAGED_BLOCK_N]>(wmma_tf32_staged_smem);
#else
    __shared__ __align__(32) float c_tile[WMMA_TF32_STAGED_BLOCK_M][WMMA_TF32_STAGED_BLOCK_N];
#endif
#pragma unroll
    for (int fi = 0; fi < WMMA_TF32_STAGED_FRAG_ROWS; ++fi) {
#pragma unroll
        for (int fj = 0; fj < WMMA_TF32_STAGED_FRAG_COLS; ++fj) {
            wmma::store_matrix_sync(
                &c_tile[warp_row_base + fi * WMMA_TF32_STAGED_FRAG][warp_col_base + fj * WMMA_TF32_STAGED_FRAG],
                c_frag[fi][fj], WMMA_TF32_STAGED_BLOCK_N, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < WMMA_TF32_STAGED_BLOCK_M * WMMA_TF32_STAGED_BLOCK_N; idx += num_threads) {
        int lr = idx / WMMA_TF32_STAGED_BLOCK_N;
        int lc = idx % WMMA_TF32_STAGED_BLOCK_N;
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

    // codex-review 指摘（kernels_wmma_tf32_opt 側の同型指摘・PR #643 再
    // レビュー）と同じ理由: `k_tile` はここでは `FRAG` 固定だが `block_n`
    // には上限検査が無いため、fail-closed 契約に従い `checked_add` で
    // 明示的に拒否する（`validate_wmma_tf32_opt_config` と同方針）。
    let a_pad = cfg
        .k_tile
        .checked_add(8)
        .ok_or_else(|| invalid("a_pad (k_tile + 8) overflow".to_string()))?;
    let b_pad = cfg
        .block_n
        .checked_add(8)
        .ok_or_else(|| invalid("b_pad (block_n + 8) overflow".to_string()))?;
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
    if smem_bytes > MMA_STATIC_SMEM_LIMIT_BYTES {
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
/// 条件を fail-closed 検査する。返す [`RenderedWmmaF16OptKernel`] の起動前
/// 検査契約は [`render_wmma_tf32_opt`] と同じ。`#[allow(dead_code)]` の
/// 理由も [`render_wmma_tf32_opt`] と同じ（非公開モジュール・現状は既定
/// 構成のみ消費・後続 #504／#519 が非既定 config の呼び出し元となる想定）。
#[allow(dead_code)]
pub fn render_wmma_f16_opt(
    cfg: &WmmaOptKernelConfig,
) -> Result<RenderedWmmaF16OptKernel, CudaError> {
    validate_wmma_f16_opt_config(cfg)?;
    Ok(RenderedWmmaF16OptKernel {
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

    /// #851 レビュー指摘対応: 非 staged TF32 経路（`WMMA_TF32_F32_OPT_BODY`）
    /// の丸め位置を固定する。`wmma_tf32_opt_source_uses_wmma_instructions`
    /// は変換命令がどこかに 1 回存在することしか検査できず、A/B 片側や
    /// 一部の fragment load から `wmma::__float_to_tf32` 変換が欠落しても
    /// 通過してしまう（単純な部分文字列存在検査の限界）。staged 経路の
    /// `wmma_tf32_staged_source_applies_tf32_conversion_to_every_fragment_load`
    /// と同じ「出現回数の突合」方式を非 staged 経路にも適用し、
    /// `load_matrix_sync`（A・B 各 1 箇所 = 計 2）と `__float_to_tf32` 変換
    /// ループ（同じく計 2）の個数が一致することを固定する。
    #[test]
    fn wmma_tf32_opt_source_applies_tf32_conversion_to_every_fragment_load() {
        let src = wmma_tf32_f32_opt_source();
        let load_count = src.matches("wmma::load_matrix_sync(").count();
        let convert_count = src.matches("wmma::__float_to_tf32(").count();
        assert_eq!(
            load_count, convert_count,
            "wmma::load_matrix_sync の出現回数（{load_count}）と \
             wmma::__float_to_tf32 変換ループの出現回数（{convert_count}）が一致しません \
             （A/B いずれかの fragment load で TF32 変換が欠落している可能性）"
        );
        assert_eq!(
            load_count, 2,
            "A フラグメント・B フラグメントそれぞれ 1 箇所ずつ、計 2 箇所の \
             load_matrix_sync 呼び出しが期待されます（実際: {load_count}）"
        );
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
        // 揃えて m=64・n=96 を使う）。テスト専用アクセサ `source()`
        // （`#[cfg(test)]`。本番経路には存在しない）で生成内容のみを検査する。
        let src = rendered.source();

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
        // 焼き込んだ値と食い違う）は fail-closed に拒否されなければ
        // ならない。`CompiledWmmaOptKernel::launch_f16`／`launch_tf32` は
        // 実機依存の `CudaFunction`／`CudaStream` なしに単体テストできない
        // ため（`kernels_mma::render_mma_f16_specializes_tile_and_static_dim`
        // と同じ理由）、同じ検査を内部で実行する
        // `WmmaOptKernelConfig::validate_launch_shape` を直接検査する。
        assert!(cfg.validate_launch_shape(64, 96, 4096).is_ok());
        assert!(matches!(
            cfg.validate_launch_shape(64, 96, 16),
            Err(CudaError::InvalidKernelConfig { .. })
        ));
    }

    /// 決定性の機械検査（#516 実装計画 4 節・§8「スコープ外」の C-5/C-2
    /// キャッシュ系タスクが本 render の出力をハッシュ材料として使う前提の
    /// 検査。`kernels_mma::render_mma_f16_is_deterministic_for_same_config`
    /// と同じ方針）。同一 `WmmaOptKernelConfig` から `render_wmma_tf32_opt`
    /// を 2 回呼んで byte 単位一致することをロックする。
    #[test]
    fn render_wmma_tf32_opt_is_deterministic_for_same_config() {
        let cfg = WmmaOptKernelConfig {
            block_m: 64,
            block_n: 96,
            k_tile: 16,
            dim_m: DimSpec::Static(4096),
            dim_n: DimSpec::Static(4096),
            dim_k: DimSpec::Static(4096),
        };
        let first = render_wmma_tf32_opt(&cfg)
            .expect("有効な構成が拒否されました")
            .source()
            .to_owned();
        let second = render_wmma_tf32_opt(&cfg)
            .expect("有効な構成が拒否されました")
            .source()
            .to_owned();
        assert_eq!(
            first, second,
            "同一 WmmaOptKernelConfig からの render_wmma_tf32_opt が byte 一致しません \
             （キャッシュキー材料としての決定性契約が崩れています）"
        );
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
                // PR #643 codex-review P2 指摘への対応: 旧ケース
                // （block_m=256・block_n=256）は warp_grid_m(8)*warp_grid_n(8)
                // *32=2048 threads となり、smem 予算検査より前のスレッド数
                // 上限（1024）で拒否されてしまい SMEM の fail-closed 分岐を
                // 検査できていなかった。block_m=32（最小・warp_tile 1 個分）
                // × block_n=1024（warp_tile の 32 倍）なら
                // threads=(32/32)*(1024/32)*32=1024（ちょうど上限内で拒否
                // されない）で、k_tile は既定値（16）のまま
                // smem_bytes=8*32*(16+4)+8*16*(1024+4)+4*32*1024=267776
                // > 49152 のみが拒否理由になる。
                "smem budget exceeded",
                WmmaOptKernelConfig {
                    block_m: 32,
                    block_n: 1024,
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
            match &result {
                Err(CudaError::InvalidKernelConfig { detail }) => {
                    // PR #643 codex-review P2 指摘への対応: 拒否されたこと
                    // だけでなく、"smem budget exceeded" ケースが実際に
                    // SMEM 予算超過分岐（スレッド数上限分岐ではなく）で
                    // 拒否されたことを detail 文字列で確認する。
                    if label == "smem budget exceeded" {
                        assert!(
                            detail.contains("shared memory"),
                            "{label} は SMEM 予算超過として拒否されるべきです（実際の detail: {detail}）"
                        );
                    }
                }
                other => panic!(
                    "{label} は InvalidKernelConfig で拒否されるべきです: config={cfg:?}, result={other:?}"
                ),
            }
        }
    }

    /// フェイルクローズド検証（TF32 opt）: `validate_wmma_tf32_opt_config`
    /// の `a_pad = k_tile + 4`／`b_pad = block_n + 4` を通常加算のまま残すと
    /// fail-closed 検証器の中に唯一の非 `checked_*` 算術が残ることになる
    /// （codex-review 指摘・PR #643 再レビュー）。本検証では `k_tile` は
    /// `FRAG_K`（8）の倍数という制約上 `u32::MAX` 近傍でも `+4` 自体は
    /// オーバーフローしない（最大到達可能値は `u32::MAX - 7` で
    /// `+4 = u32::MAX - 3` は有効な `u32`）ため、本ケースの実際の拒否は
    /// 後続の `smem_bytes` 側 `checked_mul` チェーンが担う。それでも
    /// `a_pad`/`b_pad` 自体を `checked_add` 化したのは、他の演算がすべて
    /// `checked_*` で統一されている本関数の contract 上の一貫性のためで
    /// あり（`FRAG_K`/`WARP_TILE` を将来変更した場合に到達可能性の前提が
    /// 崩れても fail-closed のまま保たれる）、本テストは「拒否される」
    /// という外部から観測可能な契約が維持されていることを固定する。
    #[test]
    fn render_wmma_tf32_opt_rejects_k_tile_pad_overflow() {
        let cfg = WmmaOptKernelConfig {
            k_tile: u32::MAX - 7,
            ..WmmaOptKernelConfig::default_tf32()
        };
        assert!(
            matches!(
                render_wmma_tf32_opt(&cfg),
                Err(CudaError::InvalidKernelConfig { .. })
            ),
            "k_tile=u32::MAX-7 は InvalidKernelConfig で拒否されるべきです"
        );
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

    /// 非既定 config（block_m/block_n=128（4×4 warp グリッド）・K 次元を
    /// `Static(4096)` で焼き込み）での f16 opt 特化 render を検査する
    /// （PR レビュー指摘への対応:
    /// `render_wmma_tf32_opt_specializes_tile_and_static_dim` は TF32 opt
    /// 側の `WMMA_TF32_OPT_WARP_GRID_N` 置換・静的 dim `#define` を検査
    /// していたが、f16 opt 側は拒否系〈`render_wmma_f16_opt_rejects_k_tile_mismatch`〉
    /// のみで、`WMMA_F16_OPT_WARP_GRID_N` の非既定値焼き込みを検査する
    /// テストが欠けていた。f16 opt は K サブステップを持たないため
    /// `k_tile` は `WMMA_F16_OPT_FRAG`（16）固定のみが許容される
    /// （`validate_wmma_f16_opt_config` 参照）。TF32 opt テストと異なり
    /// block_m/block_n の非対称化はできない〈WARP_TILE=32 の倍数で
    /// warp_grid_n を既定の 2 から変える必要がある〉ため block_n のみ
    /// 128 に変えて `WMMA_F16_OPT_WARP_GRID_N` の置換を検査する）。
    #[test]
    fn render_wmma_f16_opt_specializes_tile_and_static_dim() {
        let cfg = WmmaOptKernelConfig {
            block_m: 64,
            block_n: 128,
            k_tile: WMMA_F16_OPT_FRAG,
            dim_m: DimSpec::Dynamic,
            dim_n: DimSpec::Dynamic,
            dim_k: DimSpec::Static(4096),
        };
        let rendered = render_wmma_f16_opt(&cfg).expect("有効な構成が拒否されました");
        // dim_m/dim_n=Dynamic・dim_k=Static(4096) のため、実起動形状は
        // k=4096 固定・m/n は任意（後段の validate_launch_shape 呼び出しと
        // 揃えて m=64・n=128 を使う）。テスト専用アクセサ `source()`
        // （`#[cfg(test)]`。本番経路には存在しない）で生成内容のみを検査する。
        let src = rendered.source();

        for expected in [
            "#define WMMA_F16_OPT_BLOCK_M 64",
            "#define WMMA_F16_OPT_BLOCK_N 128",
            "#define WMMA_F16_OPT_WARP_GRID_N 4", // 128 / WARP_TILE(32)
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
        // 焼き込んだ値と食い違う）は fail-closed に拒否されなければ
        // ならない（`render_wmma_tf32_opt_specializes_tile_and_static_dim`
        // と同じ理由。実機依存の `CudaFunction`／`CudaStream` なしに単体
        // テストできないため `validate_launch_shape` を直接検査する）。
        assert!(cfg.validate_launch_shape(64, 128, 4096).is_ok());
        assert!(matches!(
            cfg.validate_launch_shape(64, 128, 16),
            Err(CudaError::InvalidKernelConfig { .. })
        ));
    }

    /// 決定性の機械検査（`render_wmma_tf32_opt_is_deterministic_for_same_config`
    /// と同じ方針。f16 opt 側）。
    #[test]
    fn render_wmma_f16_opt_is_deterministic_for_same_config() {
        let cfg = WmmaOptKernelConfig {
            block_m: 64,
            block_n: 128,
            k_tile: WMMA_F16_OPT_FRAG,
            dim_m: DimSpec::Static(4096),
            dim_n: DimSpec::Static(4096),
            dim_k: DimSpec::Static(4096),
        };
        let first = render_wmma_f16_opt(&cfg)
            .expect("有効な構成が拒否されました")
            .source()
            .to_owned();
        let second = render_wmma_f16_opt(&cfg)
            .expect("有効な構成が拒否されました")
            .source()
            .to_owned();
        assert_eq!(
            first, second,
            "同一 WmmaOptKernelConfig からの render_wmma_f16_opt が byte 一致しません \
             （キャッシュキー材料としての決定性契約が崩れています）"
        );
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

    /// [`WmmaOptKernelConfig::launch_config`] が `block_m`/`block_n` を
    /// 単位に `div_ceil` でグリッドを構築し、`shared_mem_bytes` が常に 0
    /// （静的共有メモリのみの契約）であることを、TF32 opt・f16 opt
    /// 両既定構成で検査する（PR #643 codex-review P0 再指摘への対応:
    /// `CompiledWmmaOptKernel::launch_f16`／`launch_tf32` が本メソッドの
    /// 戻り値を内部起動にのみ使う設計の土台。両 dtype で warp タイル辺が
    /// 共通〈32〉であることの前提を併せて確認する）。
    #[test]
    fn wmma_opt_config_launch_config_grid_dim_covers_m_and_n_via_div_ceil() {
        for base in [
            WmmaOptKernelConfig::default_tf32(),
            WmmaOptKernelConfig::default_f16(),
        ] {
            // m=65（block_m=64 の 1 タイル分 +1 端数）・n=64（block_n=64 の
            // ちょうど 1 タイル分。TF32 opt・f16 opt とも既定 block_m/n は
            // 64x64）。
            let launch_config = base.launch_config(65, 64);
            assert_eq!(
                launch_config.grid_dim,
                (
                    64u32.div_ceil(base.block_n),
                    65u32.div_ceil(base.block_m),
                    1
                ),
                "base={base:?}"
            );
            let warp_grid_m = base.block_m / 32;
            let warp_grid_n = base.block_n / 32;
            assert_eq!(
                launch_config.block_dim,
                (warp_grid_m * warp_grid_n * 32, 1, 1),
                "base={base:?}"
            );
            assert_eq!(launch_config.shared_mem_bytes, 0, "base={base:?}");
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
        // TF32 opt は `k_tile / FRAG_K` 回（既定 16/8=2 回）の
        // `wmma::mma_sync` 呼び出しをコンパイル時に展開する
        // （`render_wmma_tf32_opt` の `WMMA_TF32_OPT_K_SUBSTEPS` 定義を
        // 参照）。この下限を PTX テキストの `wmma.mma.sync` 出現数で
        // 確認する（受け入れ基準 2）。f16 opt は 1 K タイル内が単一
        // フラグメント（`WMMA_F16_OPT_K_TILE == WMMA_F16_OPT_FRAG`）の
        // ため出現の有無のみを確認する。
        let tf32_min_mma_count = (WMMA_TF32_OPT_K_TILE / WMMA_TF32_OPT_FRAG_K) as usize;
        // イシュー #500: TF32 opt-staged は K_SUBSTEPS（既定 16/8=2）回の
        // `wmma::mma_sync` に加え、cp.async 命令（`cp.async.cg.shared.global`）
        // が PTX へ実際に出現することも確認する（NVRTC が cp.async を
        // 未対応アーキテクチャ向けに黙って別命令へ差し替えていないことの
        // 実機的傍証。`kernels_mma.rs` の同種テストと同方針）。
        let staged_min_mma_count = (WMMA_TF32_STAGED_K_TILE / WMMA_TF32_STAGED_FRAG_K) as usize;
        for (src, min_mma_count) in [
            (wmma_tf32_f32_opt_source(), tf32_min_mma_count),
            (wmma_f16_opt_source(), 1),
        ] {
            let ptx = match compile_ptx(src, arch) {
                Ok(ptx) => ptx,
                Err(CudaError::NvrtcUnavailable { .. }) => return,
                Err(e) => panic!("既定構成カーネルソースの NVRTC コンパイルに失敗しました: {e}"),
            };
            let mma_count = ptx.to_src().matches("wmma.mma.sync").count();
            assert!(
                mma_count >= min_mma_count,
                "PTX の wmma.mma.sync 出現数（{mma_count}）が下限（{min_mma_count}）未満です \
                 （コンパイル時展開の証跡が見つかりません）"
            );
        }

        let staged_ptx = match compile_ptx(wmma_tf32_f32_staged_source(), arch) {
            Ok(ptx) => ptx,
            Err(CudaError::NvrtcUnavailable { .. }) => return,
            Err(e) => {
                panic!("TF32 opt-staged カーネルソースの NVRTC コンパイルに失敗しました: {e}")
            }
        };
        let staged_src = staged_ptx.to_src();
        let staged_mma_count = staged_src.matches("wmma.mma.sync").count();
        assert!(
            staged_mma_count >= staged_min_mma_count,
            "TF32 opt-staged PTX の wmma.mma.sync 出現数（{staged_mma_count}）が下限 \
             （{staged_min_mma_count}）未満です"
        );
        assert!(
            staged_src.contains("cp.async"),
            "TF32 opt-staged PTX に cp.async 系命令が見つかりません"
        );
    }

    /// [`validate_wmma_opt_k_tile_bound`] が通常サイズの `k`/`k_tile`
    /// を受理することを確認する（回帰防止の基本ケース。TF32 opt・f16 opt
    /// 双方の展開元タイル値で確認する）。
    #[test]
    fn validate_wmma_opt_k_tile_bound_accepts_ordinary_k() {
        assert!(validate_wmma_opt_k_tile_bound(4096, WMMA_TF32_OPT_K_TILE).is_ok());
        assert!(validate_wmma_opt_k_tile_bound(0, WMMA_TF32_OPT_K_TILE).is_ok());
        assert!(validate_wmma_opt_k_tile_bound(4096, WMMA_F16_OPT_K_TILE).is_ok());
        assert!(validate_wmma_opt_k_tile_bound(0, WMMA_F16_OPT_K_TILE).is_ok());
    }

    /// codex-review 指摘（PR #643 再レビュー）の再現ケース: 既定
    /// （16）と異なる非既定 `k_tile`（例: 24。8 の倍数だが既定値ではない）
    /// と、最終タイルの `t * k_tile + lc` が `i32::MAX` を超える `k` の
    /// 組合せを `InvalidShape` として fail-closed に拒否することを検証
    /// する。
    #[test]
    fn validate_wmma_opt_k_tile_bound_rejects_i32_overflow_for_non_default_k_tile() {
        let k_tile: u32 = 24; // 8 の倍数だが既定 WMMA_TF32_OPT_K_TILE(16) とは異なる
        let k = i32::MAX as u32;
        let tile = k_tile as u64;
        let expected_max_index = (k as u64).div_ceil(tile) * tile - 1;
        assert!(
            expected_max_index > i32::MAX as u64,
            "テスト前提が崩れています: expected_max_index={expected_max_index} は i32::MAX 以下です"
        );

        let result = validate_wmma_opt_k_tile_bound(k, k_tile);
        assert!(
            matches!(result, Err(CudaError::InvalidShape { .. })),
            "i32 オーバーフローが起こりうる k/k_tile の組合せが拒否されませんでした: {result:?}"
        );
    }

    /// `k == 0` は算術自体が発生しない no-op 形状のため、`k_tile` の値に
    /// 関わらず常に受理されることを確認する（境界条件）。
    #[test]
    fn validate_wmma_opt_k_tile_bound_accepts_zero_k_regardless_of_tile() {
        assert!(validate_wmma_opt_k_tile_bound(0, u32::MAX).is_ok());
    }

    // ========================================================================
    // TF32 opt-staged（イシュー #500）
    // ========================================================================

    /// `WMMA_TF32_STAGED_*`（Rust 側の「唯一の真実源」）が既定構成の生成
    /// ソース内 `#define` と食い違わないことを検査する
    /// （`wmma_tf32_opt_constants_match_kernel_source_defines` と同方式）。
    #[test]
    fn wmma_tf32_staged_constants_match_kernel_source_defines() {
        let src = wmma_tf32_f32_staged_source();
        let checks = [
            ("WMMA_TF32_STAGED_BLOCK_M", WMMA_TF32_STAGED_BLOCK_M),
            ("WMMA_TF32_STAGED_BLOCK_N", WMMA_TF32_STAGED_BLOCK_N),
            ("WMMA_TF32_STAGED_K_TILE", WMMA_TF32_STAGED_K_TILE),
            ("WMMA_TF32_STAGED_FRAG", WMMA_TF32_STAGED_FRAG),
            ("WMMA_TF32_STAGED_FRAG_K", WMMA_TF32_STAGED_FRAG_K),
            ("WMMA_TF32_STAGED_WARP_TILE", WMMA_TF32_STAGED_WARP_TILE),
            ("WMMA_TF32_STAGED_THREADS", WMMA_TF32_STAGED_THREADS),
            ("WMMA_TF32_STAGED_A_PAD", WMMA_TF32_STAGED_A_PAD),
            ("WMMA_TF32_STAGED_B_PAD", WMMA_TF32_STAGED_B_PAD),
            ("WMMA_TF32_STAGED_STAGES", WMMA_TF32_STAGED_STAGES),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                src.contains(&expected),
                "wmma_tf32_f32_staged_source() の `#define {name}` が Rust 側の定数（{value}）と \
                 一致しません"
            );
        }
    }

    /// 既定構成（全次元 `Dynamic`）では `#define DIM_* <カーネル引数>`
    /// 形式でカーネル引数へ間接するのみであることをロックする
    /// （`wmma_opt_default_config_dim_defines_alias_kernel_parameters` と
    /// 同方針）。
    #[test]
    fn wmma_tf32_staged_default_config_dim_defines_alias_kernel_parameters() {
        let src = wmma_tf32_f32_staged_source();
        for expected in ["#define DIM_M m", "#define DIM_N n", "#define DIM_K k"] {
            assert!(
                src.contains(expected),
                "既定次元マクロ `{expected}` が見つかりません: {src}"
            );
        }
    }

    /// tensor core 命令・TF32 明示変換の証跡（`wmma_tf32_opt_source_uses_wmma_instructions`
    /// と同方式）。
    #[test]
    fn wmma_tf32_staged_source_uses_wmma_instructions() {
        let src = wmma_tf32_f32_staged_source();
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
                "wmma_tf32_f32_staged_source() に tensor core 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// イシュー #500 の主要技法（cp.async 多段パイプライン・issue
    /// interleaving）の証跡。`kernels_mma.rs::
    /// mma_f16_source_uses_mma_sync_ldmatrix_cp_async_instructions` と同方式。
    #[test]
    fn wmma_tf32_staged_source_uses_cp_async_instructions() {
        let src = wmma_tf32_f32_staged_source();
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
        ] {
            assert!(
                src.contains(needle),
                "wmma_tf32_f32_staged_source() に cp.async 命令 `{needle}` が見つかりません"
            );
        }
    }

    /// 段数一般形の固定即値 wait（`wait_group (STAGES-2)` 相当）と、
    /// ループ外 drain（無条件 `wait_group 0;`）の双方が存在することを
    /// 検査する（`kernels_mma.rs::
    /// mma_f16_source_has_unconditional_drain_wait_group_zero` と同方針。
    /// #492 の「段数非依存の drain をループ外へ移設する」設計が staged
    /// カーネルでも保たれていることをロックする）。
    #[test]
    fn wmma_tf32_staged_source_has_fixed_immediate_wait_and_unconditional_drain() {
        let src = wmma_tf32_f32_staged_source();
        assert!(
            src.contains(
                r#"asm volatile("cp.async.wait_group %0;\n" ::"n"(WMMA_TF32_STAGED_STAGES - 2));"#
            ),
            "wmma_tf32_f32_staged_source() に段数一般形の固定即値 wait_group が見つかりません"
        );
        let drain_pos = src
            .rfind("cp.async.wait_group 0;")
            .expect("wmma_tf32_f32_staged_source() に cp.async.wait_group 0; が見つかりません");
        let last_commit_pos = src.rfind("asm volatile(\"cp.async.commit_group;").expect(
            "wmma_tf32_f32_staged_source() にループ末尾の cp.async.commit_group が \
                 見つかりません",
        );
        assert!(
            drain_pos > last_commit_pos,
            "ループ外 drain（wait_group 0）がループ内最終 commit より前にあります \
             （drain_pos={drain_pos}, last_commit_pos={last_commit_pos}）"
        );
    }

    /// advisor 指摘（PR レビュー相当）: `load_matrix_sync` の呼び出し回数
    /// と TF32 明示変換ループ（`wmma::__float_to_tf32` を呼ぶ `for` ループ）
    /// の出現回数が一致することを検査する。fragment ロード直後の変換適用は
    /// 数値契約の根幹（既存 `WMMA_TF32_F32_OPT_BODY` と同一契約）であり、
    /// 2 面バッファのどちらか一方だけ変換が漏れる回帰を検出する（単純な
    /// 部分文字列存在検査では検出できないクラスの回帰のため、出現回数の
    /// 突合という強い形の検査にする）。
    ///
    /// PR #857 レビュー指摘（P1）対応: swizzle 変種
    /// （[`wmma_tf32_f32_staged_source_with_swizzle`]）は本番変種で通常版
    /// とは別のソース生成関数を通るため、通常版のみの検査では swizzle
    /// 側だけ変換が欠落する回帰を見逃す。本テストは通常版・swizzle 版
    /// （group_width=8）の双方を走査して同一契約を検査する
    /// （`wmma_tf32_f32_staged_source_with_swizzle_does_not_mutate_...`
    /// が示すとおり swizzle 変種は BODY テンプレートを共有するアンカー
    /// 置換のみのため、同一契約が両方で成立するはず）。
    #[test]
    fn wmma_tf32_staged_source_applies_tf32_conversion_to_every_fragment_load() {
        let sources = [
            wmma_tf32_f32_staged_source().to_string(),
            wmma_tf32_f32_staged_source_with_swizzle(8).expect("group_width=8 must be accepted"),
        ];
        for src in sources {
            let load_count = src.matches("wmma::load_matrix_sync(").count();
            let convert_count = src.matches("wmma::__float_to_tf32(").count();
            // マクロ定義側に 1 箇所ずつ（LDWM_A_FRAG・LDWM_B_FRAG）記述されている
            // のみで、呼び出し側は展開時に増えるが本テストはテンプレート文字列
            // （展開前のマクロ定義込みソース）を対象にしているため、定義箇所の
            // 個数（load: 2、convert: 2）が一致することを検査する。
            assert_eq!(
                load_count, convert_count,
                "wmma::load_matrix_sync の出現回数（{load_count}）と \
                 wmma::__float_to_tf32 変換ループの出現回数（{convert_count}）が一致しません \
                 （先読みバッファの一方で TF32 変換が欠落している可能性）"
            );
            assert_eq!(
                load_count, 2,
                "LDWM_A_FRAG/LDWM_B_FRAG マクロ定義それぞれ 1 箇所ずつ、計 2 箇所の \
                 load_matrix_sync 呼び出しが期待されます（実際: {load_count}）"
            );
        }
    }

    /// REQ-8: guarded load／guarded store の境界チェックがソースから
    /// 除去されていないことをロックする。
    #[test]
    fn wmma_tf32_staged_source_retains_req8_boundary_guards() {
        let src = wmma_tf32_f32_staged_source();
        for needle in [
            "gr < DIM_M && gc < DIM_K",
            "gr < DIM_K && gc < DIM_N",
            "gr < DIM_M && gc < DIM_N",
        ] {
            assert!(
                src.contains(needle),
                "wmma_tf32_f32_staged_source() に REQ-8 境界チェック `{needle}` が見つかりません"
            );
        }
    }

    /// `RenderedWmmaTf32StagedKernel::compile` が呼ぶ固定エントリポイント
    /// 名がソース内の関数定義名と一致することをロックする（`gemm.rs::
    /// compile_wmma_tf32_staged` の `load_function("gemm_wmma_tf32_staged")`
    /// と対応）。
    #[test]
    fn wmma_tf32_staged_source_defines_expected_entry_point() {
        let src = wmma_tf32_f32_staged_source();
        assert!(
            src.contains("extern \"C\" __global__ void gemm_wmma_tf32_staged("),
            "wmma_tf32_f32_staged_source() に固定エントリポイント \
             `gemm_wmma_tf32_staged` の定義が見つかりません"
        );
    }

    /// 同一 `WmmaTf32StagedKernelConfig` からの [`render_wmma_tf32_staged`]
    /// が byte 一致することをロックする（`RenderedWmmaTf32OptKernel` と
    /// 同方式の再現性検査。`RenderedWmmaTf32StagedKernel::source` テスト
    /// 専用アクセサをここで使用する）。
    #[test]
    fn render_wmma_tf32_staged_is_deterministic_for_same_config() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let a = render_wmma_tf32_staged(&cfg).expect("既定構成は検証を通過するはずです");
        let b = render_wmma_tf32_staged(&cfg).expect("既定構成は検証を通過するはずです");
        assert_eq!(
            a.source(),
            b.source(),
            "同一 WmmaTf32StagedKernelConfig からの render_wmma_tf32_staged が byte 一致しません \
             （非決定的な展開はキャッシュ・再現性の前提を壊す）"
        );
    }

    /// イシュー #742: 既定構成（`WMMA_TF32_STAGED_DYNAMIC_SMEM` を明示指定
    /// しない）の静的レンダー結果が、動的 SMEM 分岐追加後も本番経路が
    /// 従来どおり static `__shared__` 宣言のみを含むことをロックする
    /// （`#if` 分岐の 0 側が壊れていないことの直接証拠。実装計画 §1.1
    /// 「本番経路は一切変更しない」の受け皿）。
    #[test]
    fn render_wmma_tf32_staged_static_source_keeps_static_shared_declarations() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let rendered = render_wmma_tf32_staged(&cfg).expect("既定構成は検証を通過するはずです");
        let source = rendered.source();
        assert!(
            source.contains("#define WMMA_TF32_STAGED_DYNAMIC_SMEM 0"),
            "既定（static）レンダーは WMMA_TF32_STAGED_DYNAMIC_SMEM 0 を \
             定義するはずです: {source}"
        );
        assert!(
            source.contains(
                "__shared__ __align__(16) float as_tile[WMMA_TF32_STAGED_STAGES]\
                 [WMMA_TF32_STAGED_BLOCK_M][WMMA_TF32_STAGED_A_PAD];"
            ),
            "既定（static）レンダーは as_tile の static __shared__ 宣言を \
             含むはずです"
        );
        assert!(
            source.contains(
                "__shared__ __align__(32) float c_tile[WMMA_TF32_STAGED_BLOCK_M]\
                 [WMMA_TF32_STAGED_BLOCK_N];"
            ),
            "既定（static）レンダーは c_tile の static __shared__ 宣言を \
             含むはずです"
        );
        // 注意: `WMMA_TF32_F32_STAGED_BODY` は `#if WMMA_TF32_STAGED_DYNAMIC_SMEM`
        // による **C プリプロセッサ**分岐であり、Rust 側の文字列テンプレート
        // 自体は static・dyn 両分岐のソーステキストを常に含む（`#define
        // WMMA_TF32_STAGED_DYNAMIC_SMEM 0` により NVRTC コンパイル時に
        // dyn 側が prune される）。そのため `extern __shared__` の不在を
        // Rust レベルのテキスト検査で断定することはできない
        // （NVRTC 非搭載環境で本テストを実行するため、実際のプリプロセス
        // 結果は検査できない）。本テストは static 宣言側のテキストが
        // 変更されていないことのみを保証する。
    }

    /// [`render_wmma_tf32_staged_dyn`] が `WMMA_TF32_STAGED_DYNAMIC_SMEM 1`
    /// と要求 stages を正しく焼き込み、`extern __shared__` 分岐を含むことを
    /// 確認する（イシュー #742）。
    #[test]
    fn render_wmma_tf32_staged_dyn_sets_dynamic_smem_define_and_stages() {
        let cfg = WmmaTf32StagedKernelConfig {
            stages: 10,
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        // GB10 実測 optin 上限（101,376B。docs/perf/sm121-device-attributes.md）
        // 相当の予算を渡す。stages=10 の所要（94,720B）はこれ以内。
        let rendered = render_wmma_tf32_staged_dyn(&cfg, 101_376)
            .expect("stages=10 は optin 予算 101,376B 以内のはずです");
        let source = rendered.source();
        assert!(
            source.contains("#define WMMA_TF32_STAGED_DYNAMIC_SMEM 1"),
            "dyn レンダーは WMMA_TF32_STAGED_DYNAMIC_SMEM 1 を定義するはずです"
        );
        assert!(
            source.contains("#define WMMA_TF32_STAGED_STAGES 10"),
            "dyn レンダーは要求 stages=10 を焼き込むはずです"
        );
        assert!(
            source
                .contains("extern __shared__ __align__(32) unsigned char wmma_tf32_staged_smem[];"),
            "dyn レンダーは extern __shared__ 宣言を含むはずです"
        );
    }

    /// 同一 config からの [`render_wmma_tf32_staged_dyn`] が byte 一致する
    /// ことをロックする（static 側
    /// `render_wmma_tf32_staged_is_deterministic_for_same_config` と同方式）。
    #[test]
    fn render_wmma_tf32_staged_dyn_is_deterministic_for_same_config() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let a = render_wmma_tf32_staged_dyn(&cfg, 101_376)
            .expect("既定構成は optin 予算内で検証を通過するはずです");
        let b = render_wmma_tf32_staged_dyn(&cfg, 101_376)
            .expect("既定構成は optin 予算内で検証を通過するはずです");
        assert_eq!(
            a.source(),
            b.source(),
            "同一 WmmaTf32StagedKernelConfig からの render_wmma_tf32_staged_dyn が \
             byte 一致しません"
        );
    }

    /// [`wmma_tf32_staged_dyn_smem_bytes`] の期待値固定
    /// （実装計画 §1.1 の試算式と一致することの回帰検査。stages=3:
    /// 3*(64*20+16*68)*4 = 28,416B、stages=10: 10*9,472 = 94,720B。いずれも
    /// エピローグ c_tile（16,384B）を下回るため max 式の支配項は
    /// stages 段バッファ側）。
    #[test]
    fn wmma_tf32_staged_dyn_smem_bytes_matches_expected_values() {
        let cfg_stages_3 = WmmaTf32StagedKernelConfig::default_tf32_staged();
        assert_eq!(
            wmma_tf32_staged_dyn_smem_bytes(&cfg_stages_3).expect("計算に成功するはずです"),
            28_416,
            "stages=3 の動的 SMEM 所要バイト数が期待値と一致しません"
        );

        let cfg_stages_10 = WmmaTf32StagedKernelConfig {
            stages: 10,
            ..cfg_stages_3
        };
        assert_eq!(
            wmma_tf32_staged_dyn_smem_bytes(&cfg_stages_10).expect("計算に成功するはずです"),
            94_720,
            "stages=10 の動的 SMEM 所要バイト数が期待値と一致しません"
        );
    }

    /// [`validate_wmma_tf32_staged_dyn_config`] の fail-closed 検証:
    /// optin 予算をわずかに下回る budget（smem_bytes - 1）を拒否すること
    /// を確認する。
    #[test]
    fn validate_wmma_tf32_staged_dyn_config_rejects_budget_just_below_requirement() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let smem_bytes = wmma_tf32_staged_dyn_smem_bytes(&cfg).expect("計算に成功するはずです");
        let budget = u32::try_from(smem_bytes - 1).expect("テスト用の小さい値です");
        let result = validate_wmma_tf32_staged_dyn_config(&cfg, budget);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "所要バイト数を 1 バイト下回る optin 予算が拒否されませんでした: {result:?}"
        );
    }

    /// 上記の境界値検査: budget が所要バイト数ちょうどのときは受理される
    /// ことを確認する（過剰拒否を防ぐ境界値テスト）。
    #[test]
    fn validate_wmma_tf32_staged_dyn_config_accepts_budget_exactly_at_requirement() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let smem_bytes = wmma_tf32_staged_dyn_smem_bytes(&cfg).expect("計算に成功するはずです");
        let budget = u32::try_from(smem_bytes).expect("テスト用の小さい値です");
        let result = validate_wmma_tf32_staged_dyn_config(&cfg, budget);
        match result {
            Ok(actual) => assert_eq!(
                actual, smem_bytes,
                "受理はされたが返り値の smem_bytes が期待値と異なります"
            ),
            Err(e) => panic!("所要バイト数ちょうどの optin 予算が拒否されました: {e:?}"),
        }
    }

    /// イシュー #742 の中核受入条件: static 変種では 48KiB 超過で拒否される
    /// stages=8（既定タイル）が、動的変種では GB10 実測 optin 予算
    /// （101,376B）内であれば受理されることを確認する（static 側の
    /// `validate_wmma_tf32_staged_config_accepts_default_and_rejects_smem_overflow`
    /// と対になる検査）。
    #[test]
    fn validate_wmma_tf32_staged_dyn_config_accepts_stages_beyond_static_smem_limit() {
        let cfg = WmmaTf32StagedKernelConfig {
            stages: 8,
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        // static 側は 48KiB 超過（130,048B 相当。静的宣言の単純合算式との
        // 差は c_tile エイリアス化の有無によるが、動的側も 8 段バッファ
        // 単独で 8*9,472=75,776B のため static 48KiB は依然超過する）。
        assert!(
            validate_wmma_tf32_staged_config(&cfg).is_err(),
            "static 変種は stages=8 を 48KiB 超過として拒否するはずです（前提確認）"
        );
        // dyn 側は GB10 実測 optin 予算内であれば受理される。
        let result = validate_wmma_tf32_staged_dyn_config(&cfg, 101_376);
        assert!(
            result.is_ok(),
            "dyn 変種は stages=8 を optin 予算 101,376B 以内として受理するはずです: {result:?}"
        );
    }

    /// [`validate_wmma_tf32_staged_config`] のフェイルクローズド検証:
    /// `stages < 2` を拒否することを確認する（`WMMA_TF32_STAGED_STAGES`
    /// 定数直下コメント「正しさ」参照。`wait_group (STAGES-2)` の即値が
    /// 非負であるための前提）。
    #[test]
    fn validate_wmma_tf32_staged_config_rejects_stages_below_two() {
        let cfg = WmmaTf32StagedKernelConfig {
            stages: 1,
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_config(&cfg);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "stages=1 が拒否されませんでした: {result:?}"
        );
    }

    /// [`validate_wmma_tf32_staged_config`] のフェイルクローズド検証:
    /// `stages > MAX_PIPELINE_STAGES`（16）を拒否することを確認する
    /// （SMEM 予算チェックのみでは block_m/block_n/k_tile を小さくした
    /// 構成で stages を大きく取れてしまい、この段数上限の検査を素通り
    /// しうる。ここでは意図的に warp タイル辺の最小構成
    /// 〈block_m=block_n=WARP_TILE, k_tile=FRAG_K〉を使い、SMEM 予算内の
    /// まま stages=17 が拒否されることを示す。PR #678 Bugbot 指摘対応:
    /// 旧テストは誤った ISA 上限〈wait_group 即値 0〜7〉を根拠に
    /// stages=10 を拒否していたが、正しい上限は
    /// [`MAX_PIPELINE_STAGES`]（16）である）。
    #[test]
    fn validate_wmma_tf32_staged_config_rejects_stages_exceeding_max_pipeline_stages() {
        let cfg = WmmaTf32StagedKernelConfig {
            block_m: WMMA_TF32_STAGED_WARP_TILE,
            block_n: WMMA_TF32_STAGED_WARP_TILE,
            k_tile: WMMA_TF32_STAGED_FRAG_K,
            stages: MAX_PIPELINE_STAGES + 1, // 17 > MAX_PIPELINE_STAGES(16)
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_config(&cfg);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "SMEM 予算内だが stages=17（MAX_PIPELINE_STAGES 超過）が拒否されませんでした: \
             {result:?}"
        );
    }

    /// 上記の境界値検査: `stages=MAX_PIPELINE_STAGES`（16。ちょうど上限）は
    /// 同じ最小構成で受理されることを確認する（過剰拒否を防ぐ境界値
    /// テスト）。
    #[test]
    fn validate_wmma_tf32_staged_config_accepts_stages_at_max_pipeline_stages() {
        let cfg = WmmaTf32StagedKernelConfig {
            block_m: WMMA_TF32_STAGED_WARP_TILE,
            block_n: WMMA_TF32_STAGED_WARP_TILE,
            k_tile: WMMA_TF32_STAGED_FRAG_K,
            stages: MAX_PIPELINE_STAGES, // 16 == 上限ちょうど
            // イシュー #743: a_pad/b_pad は既定構成（block_m/block_n=64,
            // k_tile=16 前提の 20/68）を引き継ぐと、本テストの最小構成
            // （block_n=32, k_tile=8）に対しては余剰がバンク周期
            // （32 要素）を超えてしまい、意図しない
            // validate_wmma_tf32_staged_padding 拒否を招く。本テストの
            // 主眼（stages 上限の境界値）に無関係なため、最小構成に自然な
            // パディング（k_tile+4/block_n+4）へ合わせる。
            a_pad: WMMA_TF32_STAGED_FRAG_K + 4,
            b_pad: WMMA_TF32_STAGED_WARP_TILE + 4,
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_config(&cfg);
        assert!(
            result.is_ok(),
            "SMEM 予算内かつ MAX_PIPELINE_STAGES ちょうどの stages=16 が拒否されました: \
             {result:?}"
        );
    }

    /// `WmmaTf32StagedKernelConfig` の SMEM 予算超過を拒否することを検証
    /// する（`docs/perf/cuda-gemm-wmma-tf32-phase-b.md` 「SMEM 予算」節の
    /// 試算どおり、既定ブロックタイル 64×64・stages=3 は 44,800B で収まる
    /// 一方、stages を増やすと 48KiB 上限を超過することを確認する）。
    #[test]
    fn validate_wmma_tf32_staged_config_accepts_default_and_rejects_smem_overflow() {
        let default_cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        assert!(
            validate_wmma_tf32_staged_config(&default_cfg).is_ok(),
            "既定構成（stages=3, block_m=block_n=64, k_tile=16）は SMEM 予算内のはずです"
        );

        // stages を大きく増やすと、2*(block_m*a_pad + k_tile*b_pad)*4B が
        // ステージ数倍に増え、48KiB 上限を超える（試算:
        // stages=8 の場合 8*(64*20+16*68)*4 + 64*64*4 = 8*3552*4 + 16384
        // = 113664 + 16384 = 130048B > 49152B）。
        let overflow_cfg = WmmaTf32StagedKernelConfig {
            stages: 8,
            ..default_cfg
        };
        let result = validate_wmma_tf32_staged_config(&overflow_cfg);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "stages=8 の SMEM 超過構成が拒否されませんでした: {result:?}"
        );
    }

    /// `WmmaTf32StagedKernelConfig` の cp.async 4 要素整列制約（f32 16
    /// バイト転送粒度）を拒否することを検証する。
    #[test]
    fn validate_wmma_tf32_staged_config_rejects_non_multiple_of_four_k_tile() {
        let cfg = WmmaTf32StagedKernelConfig {
            k_tile: 10, // FRAG_K(8) の倍数でも 4 の倍数でもない
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_config(&cfg);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "k_tile=10（4 の倍数でない）が拒否されませんでした: {result:?}"
        );
    }

    // ========================================================================
    // TF32 opt-staged SMEM バンクコンフリクト対策（イシュー #743）
    // ========================================================================

    /// [`WmmaTf32StagedKernelConfig::default_tf32_staged`] からの
    /// [`render_wmma_tf32_staged`] 展開結果が、パディング config 化
    /// （イシュー #743）の前後で byte 完全一致であることをロックする
    /// （実装計画 §1「本番ディスパッチ経路は 1 バイトも変更しない」の
    /// 回帰防止。`wmma_tf32_staged_constants_match_kernel_source_defines`
    /// が個々の `#define` 値を検査するのに対し、本テストはソース全体の
    /// byte 一致を保証する）。
    #[test]
    fn wmma_tf32_staged_default_config_render_is_byte_identical_to_production_source() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let rendered = render_wmma_tf32_staged(&cfg).expect("既定構成は検証を通過するはずです");
        assert_eq!(
            rendered.source(),
            wmma_tf32_f32_staged_source(),
            "既定構成からの render_wmma_tf32_staged が本番経路の \
             wmma_tf32_f32_staged_source() と byte 一致しません"
        );
    }

    /// [`validate_wmma_tf32_staged_padding`] の fail-closed 検証: `a_pad`/
    /// `b_pad` が 4 の倍数でない場合を拒否する。
    #[test]
    fn validate_wmma_tf32_staged_padding_rejects_non_multiple_of_four() {
        let cfg = WmmaTf32StagedKernelConfig {
            b_pad: WMMA_TF32_STAGED_BLOCK_N + 5, // 4 の倍数でない
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_config(&cfg);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "b_pad が 4 の倍数でない構成が拒否されませんでした: {result:?}"
        );
    }

    /// [`validate_wmma_tf32_staged_padding`] の fail-closed 検証: `a_pad`/
    /// `b_pad` が対応するタイル幅（`k_tile`/`block_n`）を下回る場合を
    /// 拒否する（負のパディングは行内で隣接要素と重なる）。
    #[test]
    fn validate_wmma_tf32_staged_padding_rejects_pad_below_tile_width() {
        let cfg = WmmaTf32StagedKernelConfig {
            a_pad: WMMA_TF32_STAGED_K_TILE - 4, // k_tile を下回る
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_config(&cfg);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "a_pad が k_tile を下回る構成が拒否されませんでした: {result:?}"
        );
    }

    /// [`validate_wmma_tf32_staged_padding`] の境界値検査: 余剰
    /// （`pad - tile_width`）がバンク周期（32 要素）ちょうどなら受理され、
    /// それを 4 要素超えると拒否されることを確認する（過剰拒否を防ぐ
    /// 境界値テストを兼ねる）。パディング検査そのものの境界値のみを
    /// 検査対象とするため、[`validate_wmma_tf32_staged_config`]（SMEM
    /// 予算等の他の検査も課す上位関数）ではなく
    /// [`validate_wmma_tf32_staged_padding`] を直接呼ぶ。
    #[test]
    fn validate_wmma_tf32_staged_padding_boundary_at_bank_period() {
        let accepted = WmmaTf32StagedKernelConfig {
            b_pad: WMMA_TF32_STAGED_BLOCK_N + 32, // 余剰ちょうど 32
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_padding(&accepted);
        assert!(
            result.is_ok(),
            "余剰 32（バンク周期ちょうど）の b_pad が拒否されました: {result:?}"
        );

        let rejected = WmmaTf32StagedKernelConfig {
            b_pad: WMMA_TF32_STAGED_BLOCK_N + 36, // 余剰 32 超過
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        let result = validate_wmma_tf32_staged_padding(&rejected);
        assert!(
            matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
            "余剰 36（バンク周期 32 超過）の b_pad が拒否されませんでした: {result:?}"
        );
    }

    /// イシュー #743 の中核解析（本ファイル [`WMMA_TF32_STAGED_B_PAD`]
    /// ドキュメンテーションコメント「イシュー #743」節）を固定する
    /// ロックテスト: 既定 `b_pad`（68）は 2-way、候補 `b_pad=72`
    /// （`BLOCK_N + 8`）はコンフリクトフリー（1）、無パディング相当の
    /// `b_pad=64` は 4-way。`a_pad`（既定 20）は 1（コンフリクトフリー）。
    #[test]
    fn wmma_tf32_staged_b_pad_72_is_bank_conflict_free_and_68_is_two_way() {
        assert_eq!(
            wmma_tf32_staged_b_fragment_ld_wavefronts(WMMA_TF32_STAGED_BLOCK_N + 4), // 68
            2,
            "既定 b_pad=68 の B フラグメントロード wavefront 数が期待値（2-way）と \
             一致しません"
        );
        assert_eq!(
            wmma_tf32_staged_b_fragment_ld_wavefronts(WMMA_TF32_STAGED_BLOCK_N + 8), // 72
            1,
            "候補 b_pad=72 の B フラグメントロード wavefront 数が期待値（コンフリクト \
             フリー）と一致しません"
        );
        assert_eq!(
            wmma_tf32_staged_b_fragment_ld_wavefronts(WMMA_TF32_STAGED_BLOCK_N), // 64（無パディング）
            4,
            "無パディング相当 b_pad=64 の B フラグメントロード wavefront 数が期待値 \
             （4-way）と一致しません"
        );
        assert_eq!(
            wmma_tf32_staged_a_fragment_ld_wavefronts(WMMA_TF32_STAGED_K_TILE + 4), // 20
            1,
            "既定 a_pad=20 の A フラグメントロード wavefront 数が期待値（コンフリクト \
             フリー）と一致しません"
        );
    }

    /// [`wmma_tf32_staged_dyn_smem_bytes`] が config フィールド化された
    /// `a_pad`/`b_pad`（イシュー #743）を実際に反映することを検査する
    /// （既定 68→72 へ変更すると所要バイト数が増えることの回帰検査。
    /// `wmma_tf32_staged_dyn_smem_bytes_matches_expected_values` の既定値
    /// 固定と対になる）。試算: stages=3 の場合
    /// `3*(64*20 + 16*72)*4 = 3*(1280+1152)*4 = 3*2432*4 = 29,184B`。
    #[test]
    fn wmma_tf32_staged_dyn_smem_bytes_reflects_custom_b_pad() {
        let cfg = WmmaTf32StagedKernelConfig {
            b_pad: WMMA_TF32_STAGED_BLOCK_N + 8, // 72
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        assert_eq!(
            wmma_tf32_staged_dyn_smem_bytes(&cfg).expect("計算に成功するはずです"),
            29_184,
            "b_pad=72 の動的 SMEM 所要バイト数が期待値と一致しません"
        );
    }

    /// [`WmmaTf32StagedKernelConfig::validate_launch_shape`] が dim_m/dim_n/dim_k
    /// の `Static` 不一致を fail-closed で拒否することを検査する
    /// （`wmma_opt_config_validate_launch_shape_rejects_static_mismatch` と
    /// 同方針）。
    #[test]
    fn wmma_tf32_staged_config_validate_launch_shape_rejects_static_mismatch() {
        let cfg = WmmaTf32StagedKernelConfig {
            dim_m: DimSpec::Static(4096),
            ..WmmaTf32StagedKernelConfig::default_tf32_staged()
        };
        assert!(cfg.validate_launch_shape(4096, 64, 64).is_ok());
        let result = cfg.validate_launch_shape(16, 64, 64);
        assert!(
            result.is_err(),
            "Static(4096) 指定 dim_m に対し m=16 が拒否されませんでした: {result:?}"
        );
    }

    /// [`WmmaTf32StagedKernelConfig::launch_config`] が `block_m`/`block_n`
    /// を単位に `div_ceil` でグリッドを構築し、`block_dim` は warp タイル
    /// 由来のスレッド数であることを検査する。
    #[test]
    fn wmma_tf32_staged_config_launch_config_uses_block_tile_grid() {
        let cfg = WmmaTf32StagedKernelConfig::default_tf32_staged();
        let launch = cfg.launch_config(130, 65);
        assert_eq!(
            launch.grid_dim,
            (65u32.div_ceil(64), 130u32.div_ceil(64), 1)
        );
        assert_eq!(launch.block_dim, (128, 1, 1));
        assert_eq!(launch.shared_mem_bytes, 0);
    }

    /// イシュー #741 受け入れ基準: `group_width < 2` を拒否する
    /// （本ファイル `wmma_tf32_f32_staged_source_with_swizzle`
    /// ドキュメンテーションコメント「エラー契約」参照。
    /// `kernels_mma.rs::mma_f16_source_with_swizzle_rejects_group_width_below_two`
    /// と同型）。
    #[test]
    fn wmma_tf32_f32_staged_source_with_swizzle_rejects_group_width_below_two() {
        let err = wmma_tf32_f32_staged_source_with_swizzle(1)
            .expect_err("group_width=1 must be rejected");
        assert!(matches!(err, crate::error::CudaError::InvalidShape { .. }));
        let err = wmma_tf32_f32_staged_source_with_swizzle(0)
            .expect_err("group_width=0 must be rejected");
        assert!(matches!(err, crate::error::CudaError::InvalidShape { .. }));
    }

    /// イシュー #741 受け入れ基準: `group_width >= 2` では生成ソースに
    /// `#define WMMA_TF32_STAGED_SWIZZLE_GROUP <group_width>` と remap
    /// 断片が含まれ、かつ元のアンカー（`block_row_base`/`block_col_base`
    /// 直書き）は除去されていることを検査する（アンカー出現数 1 の pin を
    /// 兼ねる。`mma_f16_source_with_swizzle_contains_group_define_and_remap_fragment`
    /// と同型）。また、エントリポイント名 `gemm_wmma_tf32_staged` が
    /// 変種ソースにも保持されることも検査する（`gemm.rs::
    /// CudaGemm::new_with_tf32_staged_swizzle` が同名で `load_function`
    /// する契約）。
    #[test]
    fn wmma_tf32_f32_staged_source_with_swizzle_contains_group_define_and_remap_fragment() {
        for group_width in [2u32, 8, 16] {
            let src = wmma_tf32_f32_staged_source_with_swizzle(group_width)
                .unwrap_or_else(|err| panic!("group_width={group_width}: {err}"));

            let expected_define = format!("#define WMMA_TF32_STAGED_SWIZZLE_GROUP {group_width}");
            assert!(
                src.contains(&expected_define),
                "group_width={group_width}: 生成ソースに `{expected_define}` が \
                 見つかりません"
            );
            for needle in [
                "long long linear_idx = (long long)blockIdx.y * gridDim.x + blockIdx.x;",
                "long long full_groups = num_m_blocks / WMMA_TF32_STAGED_SWIZZLE_GROUP;",
                "long long remainder = num_m_blocks % WMMA_TF32_STAGED_SWIZZLE_GROUP;",
                "const int block_row_base = (int)(m_block * WMMA_TF32_STAGED_BLOCK_M);",
                "const int block_col_base = (int)(n_block * WMMA_TF32_STAGED_BLOCK_N);",
            ] {
                assert!(
                    src.contains(needle),
                    "group_width={group_width}: 生成ソースに remap 断片 `{needle}` \
                     が見つかりません"
                );
            }
            assert!(
                !src.contains(
                    "    const int block_row_base = blockIdx.y * WMMA_TF32_STAGED_BLOCK_M;\n"
                ),
                "group_width={group_width}: 元のアンカー（blockIdx.y 直書き）が \
                 remap 後も残っています"
            );
            assert!(
                src.contains("extern \"C\" __global__ void gemm_wmma_tf32_staged("),
                "group_width={group_width}: エントリポイント名 gemm_wmma_tf32_staged \
                 が変種ソースで保持されていません"
            );
        }
    }

    /// `wmma_tf32_f32_staged_source_with_swizzle` はアンカー置換のみを
    /// 行い、`wmma_tf32_f32_staged_source()`（base）自体は不変であることを
    /// ロックする（`mma_f16_source_with_swizzle_does_not_mutate_mma_f16_source`
    /// と同型。実装計画 3 節「本番カーネル本体・定数は無変更」の回帰防止）。
    #[test]
    fn wmma_tf32_f32_staged_source_with_swizzle_does_not_mutate_wmma_tf32_f32_staged_source() {
        let before = wmma_tf32_f32_staged_source();
        let _ =
            wmma_tf32_f32_staged_source_with_swizzle(8).expect("group_width=8 must be accepted");
        assert_eq!(
            wmma_tf32_f32_staged_source(),
            before,
            "wmma_tf32_f32_staged_source_with_swizzle 呼び出し後に \
             wmma_tf32_f32_staged_source() が変化しています"
        );
        assert!(
            wmma_tf32_f32_staged_source().contains(
                "    const int block_row_base = blockIdx.y * WMMA_TF32_STAGED_BLOCK_M;\n"
            ),
            "wmma_tf32_f32_staged_source() の元のアンカー行が失われています \
             （本番カーネルは無変更のはず）"
        );
    }
}
