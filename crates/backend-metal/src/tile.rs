//! 動的タイル選択（TASK-1.8f・#188）: `gemm_simdgroup_tiled` カーネル用の
//! BM/BN/BK/WM/WN パラメータと行列サイズ別の候補選択ロジック。
//!
//! [`crate::gemm::MetalGemm::dispatch_auto`] が [`select`] で `(m, n, k)`
//! から [`TileConfig`] を決定し、`crate::pipeline::make_pipeline_with_constants`
//! が本モジュールの [`TileConfig::function_constants`] 相当の値（MSL
//! function constant）でパイプラインをビルドする契約（`shaders/gemm.metal`
//! の `gemm_simdgroup_tiled` と 1:1 対応）。
//!
//! MLX steel カーネル（`mlx/backend/metal/kernels/steel/gemm/`）の
//! BM/BN/BK/WM/WN テンプレートパラメータ化と同種の設計を、MSL の
//! function constant（実行時コンパイル時定数畳み込み）で再現する
//! （イシュー #188 計画「設計方針」節参照）。
//!
//! 本モジュールは `objc2` 系 FFI に一切触れない純粋関数のみで構成する
//! ため `cfg(target_os = "macos")` を付けず、Linux（CI・本実装環境）の
//! `cargo test -p backend-metal` でも単体テストが回るようにしてある
//! （`crate::pad` と同じ設計判断。本ファイル冒頭のコメント参照）。
//!
//! **選択閾値は暫定値**: 下記 [`select`] のサイズ閾値・候補パラメータは
//! MLX steel の実装傾向を参考にした暫定初期値であり、Apple Silicon 実機
//! での `examples/gemm_bench.rs` 実測（`docs/perf/metal-gemm-dynamic-tile.md`
//! に記録）で確定させる前提（イシュー #188 計画のスコープ）。
//! `docs/dispatch-rules-design.md`（accelerated 経路選択は
//! `min(M,N,K) >= 512`）とはレイヤが異なる点に注意: 本モジュールは
//! 「accelerated（Metal）経路に入った後」のタイル構成選択であり、
//! バックエンド抽象層からの経路選択（#67/#68）はスコープ外。

/// `gemm_simdgroup_tiled`（`shaders/gemm.metal`）の 1 threadgroup が担当する
/// ブロック形状・K 分割幅・simdgroup 分担・共有メモリステージング有無。
///
/// - `bm`×`bn`: 1 threadgroup が担当する C のブロック（行×列）
/// - `bk`: K 方向のループ刻み幅
/// - `wm`×`wn`: ブロックを分担する simdgroup 数（threadgroup スレッド数は
///   `wm*wn*32`）。各 simdgroup は `(bm/wm)/8 × (bn/wn)/8` 個の
///   `simdgroup_float8x8` アキュムレータを保持する
/// - `staged`: `true` なら A・B タイルを threadgroup 共有メモリへ協調ロード
///   してから `simdgroup_load` する（`USE_TGP_STAGING` function constant）。
///   `false` なら device メモリから直接 `simdgroup_load` する（協調ロードの
///   同期コストを避けるが、行優先ロード時のキャッシュ局所性は劣る。
///   「必ず速いとは限らない」ため両経路を実装し実機実測で選択する。
///   計画「設計方針」節参照）
///
/// `pad`（イシュー #538・[`TileConfig::pad`] 参照）は staged 経路の共有
/// メモリタイル（A: BM×BK、B: BK×BN）の行末パディング要素数（`f32` 単位。
/// 両タイル共通）。`simdgroup_load` の列方向アクセスが行ストライドと
/// threadgroup メモリのバンク境界（16/32 バンク）と整合してしまうことに
/// よるバンクコンフリクトを、行ストライドを `BK+pad`（A）/`BN+pad`（B）へ
/// ずらして回避する（MLX steel `gemm.h` の
/// `tgp_padding_a`/`tgp_padding_b`〈`16/sizeof(T)` 要素〉・
/// metal-flash-attention の leadingBlockDimensions 実値指定・TileKernels の
/// `TILE_X + TILE_K` 確保と同族の技法。CUDA 側 B-7 と同族。#538 計画
/// 「設計方針」節）。`staged=false`（direct-load 経路）では共有メモリを
/// 使わないため `pad` は常に 0 になる（[`TileConfig::pad`] が `staged` から
/// 導出する。次段落参照）。
///
/// **破壊的変更を伴わない導入設計（イシュー #538 codex-review 指摘 P1
/// 再指摘対応・PR #673）**: 当初 `pad` を 7 番目の `pub` フィールドとして
/// 追加し `#[non_exhaustive]` を付与する案を試みたが、Rust の言語仕様上
/// 「既存の全フィールド `pub` な構造体へ新フィールドを追加する」こと自体が
/// 構造体リテラル構築を破壊し、`#[non_exhaustive]` はこれを緩和できない
/// （リテラル構築を将来的に禁止するだけで、既存の 6 フィールドリテラルを
/// 救済しない）ため、`without_padding`/`with_pad` コンストラクタを用意して
/// もなお「クレート外の既存リテラル構築コードが無改変でコンパイルできる」
/// という意味での破壊的変更にはならない、という再指摘を受けた
/// （codex-review 指摘 2026-08-15。対応案「既存型を変更せずフィールドを
/// 増やさない」を採用）。
///
/// 本設計では `pad` を構造体フィールドとして持たず、[`TileConfig::pad`]
/// メソッドで `staged` から一意に導出する（`CANDIDATES`（本ファイル）・
/// テスト・`examples/` の全 `staged: true` 構成が `pad=4` を、唯一の
/// `staged: false` 構成が `pad=0` を使っており、`pad` は `staged` の純関数
/// として矛盾なく表現できることを確認済み）。これにより:
/// - `TileConfig` は従来どおり 6 フィールド（`bm`/`bn`/`bk`/`wm`/`wn`/
///   `staged`）の全 `pub` 構造体のままであり、`#[non_exhaustive]` を
///   付与する必要がなく、既存の構造体リテラル構築コードは無改変で動作する
/// - `TileConfigError::PadNotMultipleOfFour`／`PadWithoutStaging`（`pad` が
///   構築時入力ではなく導出値になったことで到達不能になった検証）は削除
/// - `without_padding`／`with_pad`（本 PR で新設したコンストラクタ。`main`
///   に対する破壊的変更にはならない）も併せて削除する
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileConfig {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub wm: u32,
    pub wn: u32,
    pub staged: bool,
}

/// [`TileConfig::validate`] が返す検証エラー。
///
/// [`crate::gemm::MetalGemm`] のパイプライン遅延キャッシュ構築時に、
/// 不成立構成を候補から除外し次善構成へフォールバックする判断材料になる
/// （fail-closed。計画「パイプライン管理」節）。
///
/// `#[non_exhaustive]` は付与しない（イシュー #538 codex-review 指摘 P1
/// 再指摘対応・PR #673）: 当初 `PadNotMultipleOfFour`・`PadWithoutStaging`
/// variant を追加し `#[non_exhaustive]` で外部の exhaustive `match` 破壊を
/// 緩和する案を試みたが、[`TileConfig`] 側の設計変更（`pad` を `staged` から
/// 導出する方式へ変更。本ファイル冒頭 [`TileConfig`] ドキュメント参照）に
/// より両 variant 自体が到達不能になったため削除した。variant 追加を伴わない
/// ため `#[non_exhaustive]` の付与理由も解消している。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileConfigError {
    /// `bm` が `wm*8` の倍数でない（各 simdgroup の行分担が 8 の倍数に
    /// ならず `simdgroup_float8x8` タイルへ整除できない）。
    BmNotDivisibleByWm8 { bm: u32, wm: u32 },
    /// `bn` が `wn*8` の倍数でない（列分担側の同種制約）。
    BnNotDivisibleByWn8 { bn: u32, wn: u32 },
    /// `bk` が 8 の倍数でない（`simdgroup_load` の K 方向 8 要素タイル制約）。
    BkNotMultipleOfEight { bk: u32 },
    /// `wm*wn*32`（threadgroup スレッド数）が `max_threads_per_tg` を超える。
    TooManyThreads {
        threads: u32,
        max_threads_per_tg: u32,
    },
    /// [`TileConfig::shared_mem_bytes`] が `max_shared_mem_bytes` を超える
    /// （`staged=false` の場合は常に 0 バイトのため到達しない）。
    ExceedsSharedMemory {
        bytes: u32,
        max_shared_mem_bytes: u32,
    },
    /// `(bm/wm)/8`（1 simdgroup が担当する行方向の `simdgroup_float8x8`
    /// アキュムレータ数）が `shaders/gemm.metal` の `gemm_simdgroup_tiled`
    /// が確保するローカル配列 `acc[MAX_ACC][MAX_ACC]`（`MAX_ACC = 8`）の
    /// 行方向上限を超える。カーネル側は `acc_rows`/`acc_cols` の値を検査
    /// せずローカル配列へ書き込むため、ここで弾かないとレジスタ/スタック
    /// 破壊（範囲外書き込み）に直結する（レビュー指摘。#188 PR review）。
    AccRowsExceedsMax { acc_rows: u32, max_acc: u32 },
    /// 上記の列方向版（`(bn/wn)/8` が `MAX_ACC` を超える）。
    AccColsExceedsMax { acc_cols: u32, max_acc: u32 },
}

impl std::fmt::Display for TileConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileConfigError::BmNotDivisibleByWm8 { bm, wm } => {
                // イシュー #538 codex-review 指摘（P1・再指摘）: `validate`
                // は `wm.checked_mul(8)` の失敗も本 variant として返すため、
                // `wm=u32::MAX` 等の入力では `wm*8` の再計算がここで素の
                // `u32` 乗算のままオーバーフローし、overflow-checks 有効な
                // 本番ビルド（release でも `overflow-checks = true` の場合）
                // で `Display::fmt` 自体が panic していた
                // （`.claude/rules/coding-rust.md` 「本番経路で unwrap /
                // expect を使わない」と同じ精神の禁止事項＝本番経路の
                // panic を避ける）。`checked_mul` で再計算し、表現不能な
                // 場合は積を出さずそのまま `wm` を表示する。
                match wm.checked_mul(8) {
                    Some(wm8) => write!(f, "bm={bm} is not a multiple of wm*8={wm8}"),
                    None => write!(f, "bm={bm} is not a multiple of wm*8 (wm={wm} overflows)"),
                }
            }
            TileConfigError::BnNotDivisibleByWn8 { bn, wn } => {
                // 上記 `BmNotDivisibleByWm8` と同じ理由（イシュー #538
                // codex-review 指摘 P1・再指摘）。
                match wn.checked_mul(8) {
                    Some(wn8) => write!(f, "bn={bn} is not a multiple of wn*8={wn8}"),
                    None => write!(f, "bn={bn} is not a multiple of wn*8 (wn={wn} overflows)"),
                }
            }
            TileConfigError::BkNotMultipleOfEight { bk } => {
                write!(f, "bk={bk} is not a multiple of 8")
            }
            TileConfigError::TooManyThreads {
                threads,
                max_threads_per_tg,
            } => {
                write!(
                    f,
                    "threadgroup thread count {threads} exceeds device max {max_threads_per_tg}"
                )
            }
            TileConfigError::ExceedsSharedMemory {
                bytes,
                max_shared_mem_bytes,
            } => {
                write!(
                    f,
                    "threadgroup memory {bytes} bytes exceeds device max {max_shared_mem_bytes} bytes"
                )
            }
            TileConfigError::AccRowsExceedsMax { acc_rows, max_acc } => {
                write!(
                    f,
                    "(bm/wm)/8={acc_rows} exceeds gemm_simdgroup_tiled acc[][] row limit {max_acc}"
                )
            }
            TileConfigError::AccColsExceedsMax { acc_cols, max_acc } => {
                write!(
                    f,
                    "(bn/wn)/8={acc_cols} exceeds gemm_simdgroup_tiled acc[][] col limit {max_acc}"
                )
            }
        }
    }
}

impl std::error::Error for TileConfigError {}

impl TileConfig {
    /// 単一 simdgroup が C の 8×8 タイルを 1 つ担当する構成（既存
    /// `gemm_simdgroup`〈TASK-1.8c・#40〉と等価。[`select`] の微小形状
    /// フォールバック、および `staged` の有無を問わず常に妥当な下限構成
    /// として使う）。
    pub const SINGLE_SIMDGROUP_8X8: TileConfig = TileConfig {
        bm: 8,
        bn: 8,
        bk: 8,
        wm: 1,
        wn: 1,
        staged: false,
    };

    /// `shaders/gemm.metal` の `gemm_simdgroup_tiled` が確保するローカル
    /// アキュムレータ配列 `simdgroup_float8x8 acc[MAX_ACC][MAX_ACC]` の
    /// 固定上限（カーネル側 `constexpr uint MAX_ACC = 8;` と 1:1 対応。
    /// [`validate`](Self::validate) がこの値を超える `(bm/wm)/8`・
    /// `(bn/wn)/8` を拒否することで、カーネル側のローカル配列への範囲外
    /// 書き込みを未然に防ぐ）。
    pub const MAX_ACC: u32 = 8;

    /// staged 経路（[`TileConfig::pad`]）が使う共有メモリタイルの行末
    /// パディング要素数（`f32` 単位。イシュー #538）。`CANDIDATES`
    /// （本ファイル）の全 `staged: true` 構成が採用する値と一致させる
    /// 固定値であり、[`pad`](Self::pad) 経由でのみ参照する。
    const TGP_PAD_ELEMS: u32 = 4;

    /// staged 経路の共有メモリタイル（A: BM×BK、B: BK×BN）の行末パディング
    /// 要素数を `staged` から導出する（イシュー #538 codex-review 指摘 P1
    /// 再指摘対応・PR #673。本ファイル冒頭 [`TileConfig`] ドキュメント
    /// 「破壊的変更を伴わない導入設計」節参照）。
    ///
    /// `pad` を構造体フィールドとして持たずここで導出することで、
    /// `TileConfig` は従来どおり 6 フィールドの全 `pub` 構造体のまま
    /// 保たれ、既存の構造体リテラル構築コードを一切破壊しない。
    /// `direct-load`（`staged=false`）経路は共有メモリを使わずパディングも
    /// 無意味なため常に `0` を返す。
    pub const fn pad(&self) -> u32 {
        if self.staged { Self::TGP_PAD_ELEMS } else { 0 }
    }

    /// threadgroup 1 個あたりのスレッド数（`wm*wn*32`。1 simdgroup = 32
    /// スレッド）。`crate::gemm` のディスパッチが
    /// `threadsPerThreadgroup = (wm*wn*32, 1, 1)` を構成する際に使う。
    pub fn thread_count(&self) -> u32 {
        // イシュー #538 codex-review 指摘: `TileConfig` は全フィールド
        // `pub` かつ `u32` へ範囲制約がないため、`wm`/`wn` に大きな値を
        // 与えると `wm*wn*32` が `u32` で乗算オーバーフローし wrap する。
        // `validate` の `TooManyThreads` 検査（`threads > max_threads_per_tg`）
        // を wrap 後の小さい値が素通りしてしまう fail-closed 迂回を防ぐため、
        // `u64` へ拡張した checked 演算で計算し、表現不能な場合は
        // `u32::MAX`（`max_threads_per_tg` を必ず上回る）へ飽和させる
        // （`validate` 側の比較で確実に弾かれる。`checked_shared_mem_bytes`
        // と同じ設計）。
        (self.wm as u64)
            .checked_mul(self.wn as u64)
            .and_then(|v| v.checked_mul(32))
            .unwrap_or(u64::MAX)
            .min(u32::MAX as u64) as u32
    }

    /// A タイル（`bm`×`(bk+pad)`）＋ B タイル（`bk`×`(bn+pad)`）を保持する
    /// threadgroup 共有メモリのバイト数（`f32` 4 バイト換算）。`staged=false`
    /// の場合は直接 `simdgroup_load` するため共有メモリを使わず 0 を返す
    /// （`shaders/gemm.metal` の `USE_TGP_STAGING` 分岐と対応）。
    ///
    /// `pad`（イシュー #538。本ファイル冒頭 [`TileConfig`] ドキュメント参照）
    /// は A・B 両タイルの行末へ同じ要素数だけ加算する。`gemm.metal` 側の
    /// `lda = BK + TGP_PAD`・`ldb = BN + TGP_PAD` 行ストライドと 1:1 対応
    /// させることで、確保する共有メモリ量とカーネルが実際にアクセスする
    /// 範囲を常に一致させる（validate と合わせた fail-closed 契約）。
    ///
    /// `setThreadgroupMemoryLength` へ渡す実際のバイト長（16 バイト境界
    /// 整合が必要）は [`crate::gemm`] のディスパッチ側で `.max(16)` して
    /// 決定する（本メソッドが返す 0 バイトをそのまま渡さない。bugbot
    /// 指摘・#253 レビュー）。`staged=true` の場合は `bm`/`bk`/`bn` が
    /// [`TileConfig::validate`] により常に 8 の倍数へ・`pad` が 4 の倍数へ
    /// 制約されるため、この戻り値は常に 256 以上かつ 16 の倍数になる。
    pub fn shared_mem_bytes(&self) -> u32 {
        if !self.staged {
            return 0;
        }
        // イシュー #538 codex-review 指摘（P0）: 以前は `bm * (bk + pad) +
        // bk * (bn + pad)) * 4` を `u32` のまま計算していたため、`bm`/`bn`/
        // `bk`（いずれも `pub` フィールドで任意の `u32` を受け取れる）に
        // 大きな値を渡すと加算・乗算がオーバーフローし wrap する。release
        // ビルドでは panic せず小さな値へ wrap するため、`validate` の
        // `ExceedsSharedMemory` 検査（`bytes > max_shared_mem_bytes`）を
        // 迂回でき、Rust 側の確保長（`crate::gemm` の
        // `setThreadgroupMemoryLength`）と `shaders/gemm.metal` の
        // `lda = BK + TGP_PAD`／`ldb = BN + TGP_PAD` が実際にアクセスする
        // 範囲との契約が崩れ、threadgroup memory の範囲外アクセスに
        // つながる。`pad`（[`pad`](Self::pad)）自体は `staged` から導出する
        // 固定値（0 または `TGP_PAD_ELEMS`）のためオーバーフロー源には
        // ならないが、`bm`/`bn`/`bk` 側は依然として任意値のため以下の
        // checked 演算は必須のまま維持する。
        //
        // `u64` へ拡張し `checked_add`／`checked_mul` で計算する（`bm`/`bn`/
        // `bk`/`pad` は最大でも `u32::MAX` のため、各要素を `u64` へ広げた
        // 中間結果同士の乗算・加算は依然として `u64` を超えうるので
        // `checked_*` が必須。単純な `u64` キャストだけでは不十分）。
        // 表現不能な場合は `u32::MAX`（`validate` が要求する
        // `max_shared_mem_bytes` を必ず上回る値）へ飽和させ、
        // `ExceedsSharedMemory` で確実に fail-closed に拒否させる。
        let bm = self.bm as u64;
        let bn = self.bn as u64;
        let bk = self.bk as u64;
        let pad = self.pad() as u64;
        let compute = || -> Option<u64> {
            let a_row = bk.checked_add(pad)?;
            let b_row = bn.checked_add(pad)?;
            let a_tile = bm.checked_mul(a_row)?;
            let b_tile = bk.checked_mul(b_row)?;
            a_tile.checked_add(b_tile)?.checked_mul(4)
        };
        compute().unwrap_or(u64::MAX).min(u32::MAX as u64) as u32
    }

    /// `bm/bn/bk/wm/wn` の整除制約・デバイス上限（`max_threads_per_tg`:
    /// `MTLComputePipelineState::maxTotalThreadsPerThreadgroup`、
    /// `max_shared_mem_bytes`: `MTLDevice::maxThreadgroupMemoryLength`）
    /// との整合を検証する。[`crate::gemm::MetalGemm`] のパイプライン構築
    /// （macOS 実機のみ到達）から呼ばれるほか、本ファイル末尾の単体テストで
    /// GPU 非依存に検証できる（Linux CI でも実行可能）。
    pub fn validate(
        &self,
        max_threads_per_tg: u32,
        max_shared_mem_bytes: u32,
    ) -> Result<(), TileConfigError> {
        // イシュー #538 codex-review 指摘: `wm`/`wn` も `pub` フィールドで
        // 任意の `u32` を受け取れるため、`wm * 8`／`wn * 8` を素の `u32`
        // 乗算のまま行うと極端に大きい値でオーバーフローし wrap しうる
        // （wrap 後の小さい除数へ `bm`/`bn` がたまたま整除してしまうと、
        // 後続の `acc_rows`/`acc_cols`・`thread_count` 計算が想定外の
        // 構成を「妥当」と誤判定する）。`checked_mul` で拒否し、
        // オーバーフロー時は本来の不整合と同じ `BmNotDivisibleByWm8`／
        // `BnNotDivisibleByWn8` として fail-closed に扱う。
        let wm8 = self.wm.checked_mul(8);
        if self.wm == 0 || wm8.is_none_or(|wm8| !self.bm.is_multiple_of(wm8)) {
            return Err(TileConfigError::BmNotDivisibleByWm8 {
                bm: self.bm,
                wm: self.wm,
            });
        }
        let wn8 = self.wn.checked_mul(8);
        if self.wn == 0 || wn8.is_none_or(|wn8| !self.bn.is_multiple_of(wn8)) {
            return Err(TileConfigError::BnNotDivisibleByWn8 {
                bn: self.bn,
                wn: self.wn,
            });
        }
        if self.bk == 0 || !self.bk.is_multiple_of(8) {
            return Err(TileConfigError::BkNotMultipleOfEight { bk: self.bk });
        }

        // イシュー #538: `pad`（[`pad`](Self::pad)）は `staged` の純関数
        // として導出するため、`4` の倍数であること・`staged=false` では
        // `0` になることは型の設計自体で保証済み（`TGP_PAD_ELEMS = 4` が
        // 常に 4 の倍数）であり、ここでの実行時検証は不要（本ファイル
        // 冒頭 [`TileConfig`] ドキュメント「破壊的変更を伴わない導入設計」
        // 節参照）。

        // `shaders/gemm.metal` の `acc[MAX_ACC][MAX_ACC]` ローカル配列は
        // `acc_rows = (bm/wm)/8`・`acc_cols = (bn/wn)/8` を検査せず添字に
        // 使うため、ここで弾かないと範囲外書き込み（レジスタ/スタック
        // 破壊）に直結する（レビュー指摘。#188 PR review）。
        let acc_rows = (self.bm / self.wm) / 8;
        if acc_rows > Self::MAX_ACC {
            return Err(TileConfigError::AccRowsExceedsMax {
                acc_rows,
                max_acc: Self::MAX_ACC,
            });
        }
        let acc_cols = (self.bn / self.wn) / 8;
        if acc_cols > Self::MAX_ACC {
            return Err(TileConfigError::AccColsExceedsMax {
                acc_cols,
                max_acc: Self::MAX_ACC,
            });
        }

        let threads = self.thread_count();
        if threads > max_threads_per_tg {
            return Err(TileConfigError::TooManyThreads {
                threads,
                max_threads_per_tg,
            });
        }

        let bytes = self.shared_mem_bytes();
        if bytes > max_shared_mem_bytes {
            return Err(TileConfigError::ExceedsSharedMemory {
                bytes,
                max_shared_mem_bytes,
            });
        }

        Ok(())
    }
}

/// [`select`] が候補として巡回する構成（大 → 小の優先順）。MLX steel の
/// 実装傾向（大形状ほど大きい BM/BN・複数 simdgroup 分担）を参考にした
/// 暫定初期値（本ファイル冒頭コメント参照。実機実測で確定させる）。
///
/// `select` は先頭 4 要素（index 0〜3）に添字で直接依存する（`select` 本体
/// 参照）ため、既存 4 構成の並び順・個数は変更しない。イシュー #532 で
/// index 4〜6 に MLX steel classic 経路（`steel_gemm_fused.metal` の
/// `instantiate_gemm_shapes_helper` が実体化する 6 構成のうち本実装未収録
/// だった 3 つ）を追加した。`select` の選択組み込み・閾値調整は実機実測で
/// 確定させる後続スコープ（イシュー #532 計画「スコープ外」節）であり本
/// 追加では行わない。
///
/// `pub(crate)`（`pub` にしない）: 候補の並び順・個数は `select` が添字で
/// 依存する内部表現であり、クレート外へ安定 API として公開すると将来の
/// 候補調整が公開 API 契約に組み込まれてしまう（codex-review 指摘・PR
/// #651）。本セットを直接巡回するテストは統合テスト（`tests/` 配下・別
/// コンパイル単位）ではなく本ファイル末尾の `#[cfg(test)] mod tests`
/// （クレート内部・`pub(crate)` を参照可能）に置く。
pub(crate) const CANDIDATES: &[TileConfig] = &[
    // 大形状（正方）: 64x64 ブロックを 2x2=4 simdgroup で分担。
    TileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    },
    // 縦長（m がかなり大きく n が中程度）。
    TileConfig {
        bm: 64,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    },
    // 横長（n がかなり大きく m が中程度）。
    TileConfig {
        bm: 32,
        bn: 64,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    },
    // 中形状（正方）。
    TileConfig {
        bm: 32,
        bn: 32,
        bk: 16,
        wm: 2,
        wn: 2,
        staged: true,
    },
    // MLX steel classic 経路の未収録構成（イシュー #532）:
    // 大形状（正方）を少 simdgroup（wm=1,wn=2 の 64 スレッド）で分担する
    // 構成。acc_rows=(64/1)/8=8 が `TileConfig::MAX_ACC` ちょうどの境界。
    TileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 1,
        wn: 2,
        staged: true,
    },
    // MLX steel classic 経路の未収録構成（イシュー #532）: `bk=32` は本実装
    // 初採用。K 方向のループ刻みを既存候補の 2 倍にすることで、K=4096 等
    // 長い内積で `threadgroup_barrier` の往復回数を半減させる狙い（理論
    // 根拠。実機ベンチによる効果確認は後続スコープ）。SMEM は pad=4 込みで
    // `(64*36+32*36)*4=13824` バイト（イシュー #538。旧 pad=0 時点の
    // 12288 バイトから増加）で、32KiB 上限内（`TileConfig::validate` で
    // 機械検証）。
    TileConfig {
        bm: 64,
        bn: 32,
        bk: 32,
        wm: 2,
        wn: 2,
        staged: true,
    },
    // MLX steel classic 経路の未収録構成（イシュー #532）: `wm=4` の縦
    // 分担・`bk=8`（最小許容値）の小刻み K 分割構成。
    TileConfig {
        bm: 64,
        bn: 32,
        bk: 8,
        wm: 4,
        wn: 1,
        staged: true,
    },
    // 微小形状: 既存 gemm_simdgroup と等価な単一 simdgroup 8x8。
    TileConfig::SINGLE_SIMDGROUP_8X8,
];

/// `primary` を先頭に、常に妥当な [`TileConfig::SINGLE_SIMDGROUP_8X8`] を
/// 末尾に持つフォールバック候補列を返す（`primary` が既に単一 simdgroup
/// 構成なら 1 要素のみ）。
///
/// [`crate::gemm::MetalGemm::pipeline_for_tile`] が構成の検証・パイプライン
/// 構築失敗時（デバイス上限超過等）に順に試す（fail-closed。イシュー #188
/// 計画「パイプライン管理」節）。`SINGLE_SIMDGROUP_8X8` は
/// `wm=wn=1`・`bm=bn=bk=8` であり、[`TileConfig::validate`] が
/// Apple Silicon の一般的な上限（1024 スレッド・32KiB 共有メモリ）は
/// もちろん、`.claude/rules/coding-rust.md` が要求する最小構成でも常に
/// 通ることを本ファイル末尾のテストで保証する。
pub fn fallback_chain(primary: TileConfig) -> Vec<TileConfig> {
    if primary == TileConfig::SINGLE_SIMDGROUP_8X8 {
        vec![primary]
    } else {
        vec![primary, TileConfig::SINGLE_SIMDGROUP_8X8]
    }
}

/// `(m, n, k)` から [`TileConfig`] を選択する（暫定閾値。本ファイル冒頭
/// コメント参照）。[`crate::gemm::MetalGemm::dispatch_auto`] の入口。
///
/// 閾値はパディング前の元次元（`m`/`n`/`k`）に対して判定する
/// （`crate::pad::pad8` によるパディングは選択後・パイプライン確保直前に
/// `crate::gemm` 側で行う。本関数はパディングの有無を問わない純粋関数）。
pub fn select(m: usize, n: usize, k: usize) -> TileConfig {
    const SMALL: usize = 64;
    const LARGE: usize = 512;
    const ASPECT_RATIO: usize = 2;

    if m < SMALL || n < SMALL || k < SMALL {
        return TileConfig::SINGLE_SIMDGROUP_8X8;
    }

    let large = m >= LARGE && n >= LARGE;
    let tall = m >= n.saturating_mul(ASPECT_RATIO);
    let wide = n >= m.saturating_mul(ASPECT_RATIO);

    match (large, tall, wide) {
        (_, true, _) => CANDIDATES[1], // 64x32（縦長）
        (_, _, true) => CANDIDATES[2], // 32x64（横長）
        (true, _, _) => CANDIDATES[0], // 64x64（大形状・正方）
        _ => CANDIDATES[3],            // 32x32（中形状・正方）
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TileConfig::thread_count / shared_mem_bytes（pure） ---

    #[test]
    fn thread_count_is_wm_wn_times_32() {
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.thread_count(), 128);
        assert_eq!(TileConfig::SINGLE_SIMDGROUP_8X8.thread_count(), 32);
    }

    #[test]
    fn shared_mem_bytes_is_zero_when_not_staged() {
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: false,
        };
        assert_eq!(cfg.shared_mem_bytes(), 0);
    }

    #[test]
    fn shared_mem_bytes_includes_derived_pad_in_both_tile_strides_when_staged() {
        // イシュー #538 codex-review 指摘 P1 再指摘対応（PR #673）で `pad` を
        // `staged` からの導出値（[`TileConfig::pad`]）へ変更した。`staged:
        // true` は常に `pad()=TGP_PAD_ELEMS=4` を両タイルの行末へ加算する:
        // A: 64x(16+4)=64x20, B: 16x(64+4)=16x68 -> (1280+1088)*4 = 9472 バイト
        // （旧 pad=0 時点は 8192 バイトだった）。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.pad(), 4);
        assert_eq!(cfg.shared_mem_bytes(), 9472);
    }

    #[test]
    fn shared_mem_bytes_saturates_instead_of_wrapping_on_overflow() {
        // codex-review 指摘（P0・#538 PR レビュー）: `bm`/`bn`/`bk` は任意の
        // `u32` を受け取れる公開フィールドのため、以前の `u32` のみの演算
        // では `bm*(bk+pad) + bk*(bn+pad)) * 4` がオーバーフローして小さな
        // 値へ wrap し、`validate` の `ExceedsSharedMemory` 検査を迂回でき
        // ていた。`pad` は #538 codex-review 指摘 P1 再指摘対応（PR #673）で
        // `staged` からの導出値（[`TileConfig::pad`]。0 または
        // `TGP_PAD_ELEMS=4` の固定値）へ変わりオーバーフロー源ではなくなった
        // ため、本 regression test は `bk` に `u32::MAX` 近辺の値を与える
        // ケースへ retarget する。`u32::MAX` を返し、`validate` 側の
        // `bytes > max_shared_mem_bytes` 比較で必ず拒否されることを確認する
        // （wrap による小さい値への回帰を防ぐ regression test）。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: u32::MAX - 7, // 8 の倍数（u32::MAX 以下で最大）へ切り下げ（validate の BkNotMultipleOfEight を避ける）
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes(), u32::MAX);
        assert_eq!(
            cfg.validate(1024, 32 * 1024),
            Err(TileConfigError::ExceedsSharedMemory {
                bytes: u32::MAX,
                max_shared_mem_bytes: 32 * 1024,
            })
        );
    }

    #[test]
    fn thread_count_saturates_instead_of_wrapping_on_overflow() {
        // codex-review 指摘（P0・#538 PR レビュー）: `wm`/`wn` に極端に
        // 大きい値を与えると `wm*wn*32` が `u32` でオーバーフローし wrap
        // しうる。飽和させることで `validate` の `TooManyThreads` 検査が
        // 確実に働くことを確認する。
        let cfg = TileConfig {
            bm: 8,
            bn: 8,
            bk: 8,
            wm: 1 << 30,
            wn: 1 << 30,
            staged: false,
        };
        assert_eq!(cfg.thread_count(), u32::MAX);
    }

    #[test]
    fn pad_is_derived_purely_from_staged() {
        // イシュー #538 codex-review 指摘 P1 再指摘対応（PR #673）: `pad` は
        // 構造体フィールドではなく `staged` から導出する（本ファイル冒頭
        // [`TileConfig`] ドキュメント「破壊的変更を伴わない導入設計」節
        // 参照）。この設計により従来どおり 6 フィールドの構造体リテラル
        // 構築が無改変で動作し続けることを確認する。
        let staged_cfg = TileConfig {
            bm: 32,
            bn: 32,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(staged_cfg.pad(), 4);

        let direct_cfg = TileConfig {
            bm: 8,
            bn: 8,
            bk: 8,
            wm: 1,
            wn: 1,
            staged: false,
        };
        assert_eq!(direct_cfg.pad(), 0);
    }

    // --- TileConfig::validate（pure・GPU 非依存） ---

    #[test]
    fn validate_accepts_all_candidates_under_typical_device_limits() {
        // Apple Silicon の典型的な上限（1024 スレッド/threadgroup、
        // 32KiB threadgroup メモリ）で全候補・単一 simdgroup 構成を検証する。
        for cfg in CANDIDATES {
            cfg.validate(1024, 32 * 1024)
                .unwrap_or_else(|e| panic!("candidate {cfg:?} rejected: {e}"));
        }
    }

    #[test]
    fn validate_rejects_bm_not_divisible_by_wm8() {
        let cfg = TileConfig {
            bm: 60,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BmNotDivisibleByWm8 { bm: 60, wm: 2 }
        ));
    }

    #[test]
    fn validate_rejects_bn_not_divisible_by_wn8() {
        let cfg = TileConfig {
            bm: 64,
            bn: 60,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BnNotDivisibleByWn8 { bn: 60, wn: 2 }
        ));
    }

    #[test]
    fn bm_not_divisible_display_does_not_panic_on_overflowing_wm() {
        // イシュー #538 codex-review 指摘（P1・再指摘）: `validate` は
        // `wm.checked_mul(8)` の失敗（`wm=u32::MAX` 等）も
        // `BmNotDivisibleByWm8` として返すが、その `Display` 実装が `wm*8`
        // を未検査で再計算していたため、overflow-checks 有効な本番ビルド
        // では `to_string()`（エラー表示）自体が panic していた
        // （`.claude/rules/coding-rust.md` 「本番経路で unwrap/expect を
        // 使わない」と同じ精神＝本番経路の panic を避ける）。本ワークスペース
        // は `[profile.dev]` で `overflow-checks` を明示 `false` にしていない
        // ため既定の `true` が有効で、本テストは実際に overflow-checks 有効
        // なビルドで実行される（regression 検知が機能する前提）。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: u32::MAX,
            wn: 1,
            staged: false,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BmNotDivisibleByWm8 { bm: 64, wm } if wm == u32::MAX
        ));
        // `Display::fmt` 呼び出し自体が panic しないことを確認する
        // （wrap ではなく checked 演算で回避していることの regression test）。
        let _ = err.to_string();
    }

    #[test]
    fn bn_not_divisible_display_does_not_panic_on_overflowing_wn() {
        // 上記 `bm_not_divisible_display_does_not_panic_on_overflowing_wm`
        // と同じ理由（イシュー #538 codex-review 指摘 P1・再指摘）の `bn`/`wn` 版。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 1,
            wn: u32::MAX,
            staged: false,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BnNotDivisibleByWn8 { bn: 64, wn } if wn == u32::MAX
        ));
        let _ = err.to_string();
    }

    #[test]
    fn validate_rejects_bk_not_multiple_of_eight() {
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 12,
            wm: 2,
            wn: 2,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BkNotMultipleOfEight { bk: 12 }
        ));
    }

    #[test]
    fn validate_rejects_thread_count_exceeding_device_limit() {
        let cfg = TileConfig {
            bm: 128,
            bn: 128,
            bk: 16,
            wm: 4,
            wn: 4,
            staged: true,
        };
        assert_eq!(cfg.thread_count(), 512);
        let err = cfg.validate(256, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::TooManyThreads {
                threads: 512,
                max_threads_per_tg: 256
            }
        ));
    }

    #[test]
    fn validate_rejects_shared_memory_exceeding_device_limit() {
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        // pad() は staged=true から常に 4 を導出する（イシュー #538
        // codex-review 指摘 P1 再指摘対応・PR #673）ため 9472 バイト
        // （旧 pad=0 時点は 8192 バイトだった）。
        assert_eq!(cfg.shared_mem_bytes(), 9472);
        let err = cfg.validate(1024, 4096).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::ExceedsSharedMemory {
                bytes: 9472,
                max_shared_mem_bytes: 4096
            }
        ));
    }

    #[test]
    fn validate_rejects_acc_rows_exceeding_max() {
        // bm=128, wm=1 -> acc_rows=(128/1)/8=16 > MAX_ACC=8。
        // レビュー指摘の再現ケース（#188 PR review）:
        // カーネル側 acc[MAX_ACC][MAX_ACC] への範囲外書き込みに直結する。
        let cfg = TileConfig {
            bm: 128,
            bn: 8,
            bk: 8,
            wm: 1,
            wn: 1,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::AccRowsExceedsMax {
                acc_rows: 16,
                max_acc: 8
            }
        ));
    }

    #[test]
    fn validate_rejects_acc_cols_exceeding_max() {
        // bn=128, wn=1 -> acc_cols=(128/1)/8=16 > MAX_ACC=8。
        let cfg = TileConfig {
            bm: 8,
            bn: 128,
            bk: 8,
            wm: 1,
            wn: 1,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::AccColsExceedsMax {
                acc_cols: 16,
                max_acc: 8
            }
        ));
    }

    // --- fallback_chain（pure） ---

    #[test]
    fn fallback_chain_appends_single_simdgroup_when_primary_differs() {
        let chain = fallback_chain(CANDIDATES[0]);
        assert_eq!(chain, vec![CANDIDATES[0], TileConfig::SINGLE_SIMDGROUP_8X8]);
    }

    #[test]
    fn fallback_chain_is_single_element_when_primary_is_already_fallback() {
        let chain = fallback_chain(TileConfig::SINGLE_SIMDGROUP_8X8);
        assert_eq!(chain, vec![TileConfig::SINGLE_SIMDGROUP_8X8]);
    }

    // --- select（pure） ---

    #[test]
    fn select_falls_back_to_single_simdgroup_for_small_shapes() {
        assert_eq!(select(32, 32, 32), TileConfig::SINGLE_SIMDGROUP_8X8);
        assert_eq!(select(1000, 1000, 32), TileConfig::SINGLE_SIMDGROUP_8X8);
    }

    #[test]
    fn select_picks_large_square_config_above_threshold() {
        let cfg = select(1024, 1024, 1024);
        assert_eq!(cfg, CANDIDATES[0]);
        assert_eq!((cfg.bm, cfg.bn), (64, 64));
    }

    #[test]
    fn select_picks_mid_square_config_for_moderate_shapes() {
        let cfg = select(128, 128, 128);
        assert_eq!(cfg, CANDIDATES[3]);
        assert_eq!((cfg.bm, cfg.bn), (32, 32));
    }

    #[test]
    fn select_picks_tall_config_when_m_dominates() {
        let cfg = select(1024, 128, 256);
        assert_eq!(cfg, CANDIDATES[1]);
        assert_eq!((cfg.bm, cfg.bn), (64, 32));
    }

    #[test]
    fn select_picks_wide_config_when_n_dominates() {
        let cfg = select(128, 1024, 256);
        assert_eq!(cfg, CANDIDATES[2]);
        assert_eq!((cfg.bm, cfg.bn), (32, 64));
    }

    // --- イシュー #532: MLX classic 経路の未収録 3 構成 ---

    #[test]
    fn bk32_candidate_shared_mem_is_13824_bytes_within_32kib_limit() {
        // (64,32,32,2,2,pad=4): A=64x36, B=32x36 -> (2304+1152)*4 = 13824 バイト
        // （イシュー #538 で pad=4 導入。旧 pad=0 時点は 12288 バイトだった）。
        // 既存最大を上回るが 32KiB（32768 バイト）以内であることを固定する
        // （イシュー #532 計画「現状分析」節の事前検証値・#538 計画
        // 「事前検証値」節で更新）。
        let cfg = TileConfig {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes(), 13824);
        assert!(cfg.shared_mem_bytes() <= 32 * 1024);
        cfg.validate(1024, 32 * 1024)
            .unwrap_or_else(|e| panic!("bk=32 candidate rejected: {e}"));
    }

    #[test]
    fn wm4_bk8_candidate_has_128_threads_and_validates() {
        // (64,32,8,4,1): threads=4*1*32=128、acc_rows=(64/4)/8=2・
        // acc_cols=(32/1)/8=4 で MAX_ACC 拡張は不要（イシュー #532 計画
        // 「現状分析」節）。
        let cfg = TileConfig {
            bm: 64,
            bn: 32,
            bk: 8,
            wm: 4,
            wn: 1,
            staged: true,
        };
        assert_eq!(cfg.thread_count(), 128);
        cfg.validate(1024, 32 * 1024)
            .unwrap_or_else(|e| panic!("wm=4/bk=8 candidate rejected: {e}"));
    }

    #[test]
    fn wm1_wn2_candidate_acc_rows_hits_max_acc_boundary_and_validates() {
        // (64,64,16,1,2): acc_rows=(64/1)/8=8 が MAX_ACC=8 ちょうどの境界。
        // 超過ではなく境界一致で validate を通ることを固定する（イシュー
        // #532 計画「現状分析」節）。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 1,
            wn: 2,
            staged: true,
        };
        let acc_rows = (cfg.bm / cfg.wm) / 8;
        assert_eq!(acc_rows, TileConfig::MAX_ACC);
        cfg.validate(1024, 32 * 1024)
            .unwrap_or_else(|e| panic!("wm=1/wn=2 candidate rejected: {e}"));
    }

    #[test]
    fn candidates_include_the_three_mlx_classic_configs_added_in_issue_532() {
        // CANDIDATES への収録漏れ・削除を検知する回帰ガード。
        // `pad()` は #538 で `staged` から導出する設計にしたため（本ファイル
        // 冒頭 [`TileConfig`] ドキュメント参照）、全 `staged: true` 構成が
        // 自動的に `pad()=4` になる（比較用構造体リテラルへ pad は不要）。
        let expected = [
            TileConfig {
                bm: 64,
                bn: 64,
                bk: 16,
                wm: 1,
                wn: 2,
                staged: true,
            },
            TileConfig {
                bm: 64,
                bn: 32,
                bk: 32,
                wm: 2,
                wn: 2,
                staged: true,
            },
            TileConfig {
                bm: 64,
                bn: 32,
                bk: 8,
                wm: 4,
                wn: 1,
                staged: true,
            },
        ];
        for cfg in expected {
            assert!(
                CANDIDATES.contains(&cfg),
                "CANDIDATES に {cfg:?} が含まれていない"
            );
        }
    }

    #[test]
    fn select_result_always_validates_under_typical_device_limits() {
        for &(m, n, k) in &[
            (7usize, 13usize, 5usize),
            (64, 64, 64),
            (128, 128, 128),
            (1024, 128, 256),
            (128, 1024, 256),
            (2048, 2048, 2048),
            (4096, 512, 512),
        ] {
            let cfg = select(m, n, k);
            cfg.validate(1024, 32 * 1024)
                .unwrap_or_else(|e| panic!("select({m},{n},{k})={cfg:?} rejected: {e}"));
        }
    }

    // --- CANDIDATES を直接巡回する実機依存テスト（イシュー #532・PR #651 codex-review
    // 指摘対応） ---
    //
    // `CANDIDATES` は `pub(crate)`（クレート外非公開。本ファイル冒頭のコメント参照）の
    // ため、これを直接巡回するテストは統合テスト（`tests/` 配下・別コンパイル単位で
    // 公開 API しか見えない）ではなくここに置く。`crate::gemm::MetalGemm`・
    // `crate::context::MetalContext` は macOS 限定モジュール（`lib.rs` の
    // `cfg(target_os = "macos")`）のため、本 mod 自体は cfg なし（Linux でも
    // 単体テストが回る設計）だが以下 2 件のみ個別に `cfg(target_os = "macos")` を
    // 付ける。

    /// `CANDIDATES`（実セット。イシュー #532 で MLX classic 経路の未収録 3 構成を
    /// 追加済み）を全て、8 の倍数の中規模形状で検証する（構成別の一致確認）。
    /// ローカルに候補配列を複製せず実セットを直接巡回することで、本ファイル側の
    /// 追加・変更が本テストへ自動的に反映されドリフトしない。
    ///
    /// `dispatch_variant` だけを呼ぶと `MetalGemm::pipeline_for_tile` が構成
    /// 失敗時に `fallback_chain` で `TileConfig::SINGLE_SIMDGROUP_8X8` へ
    /// サイレントにフォールバックしても数値一致自体は通ってしまい、対象候補が
    /// 実際にコンパイル・実行されたことを保証しない（イシュー #532・PR #651
    /// codex-review 指摘 P2）。`MetalGemm::resolve_tile_config`（`pub(crate)`。
    /// PR #651 codex-review 再指摘 P1 で `#[doc(hidden)] pub` から変更。本
    /// `mod tests` はクレート境界の内側のため届く。`crate::gemm` 参照）で
    /// 実際に採用された構成を事前取得し `cfg` と一致することを assert して
    /// からディスパッチすることで、フォールバックが起きた場合は本テスト
    /// 自体を失敗させる。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn all_tile_candidates_match_cpu_reference_medium_shape() {
        use backend_cpu::parity::{assert_parity, matmul_reference_fma};
        use bench_harness::rng::Xorshift64Star;

        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = crate::gemm::MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for (i, &cfg) in CANDIDATES.iter().enumerate() {
            let resolved = gemm.resolve_tile_config(&ctx, cfg).unwrap_or_else(|err| {
                panic!("候補 {cfg:?} のパイプライン構築・検証（実デバイス上限）に失敗した: {err}")
            });
            assert_eq!(
                resolved, cfg,
                "候補 {cfg:?} が実デバイス上でサイレントに {resolved:?} へフォールバックした \
                 （構成失敗を検知できていない）"
            );

            let (m, n, k) = (256, 256, 256);
            let seed_a = 10 + i as u64;
            let seed_b = 20 + i as u64;

            let a = Xorshift64Star::new(seed_a).fill_vec(m * k);
            let b = Xorshift64Star::new(seed_b).fill_vec(k * n);

            let mut expected = vec![0.0f32; m * n];
            matmul_reference_fma(&a, &b, &mut expected, m, n, k)
                .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

            let actual = gemm
                .dispatch_variant(
                    &ctx,
                    crate::gemm::GemmVariant::SimdgroupTiled(cfg),
                    &a,
                    &b,
                    m,
                    n,
                    k,
                )
                .unwrap_or_else(|err| {
                    panic!("Metal SimdgroupTiled({cfg:?}) GEMM のディスパッチに失敗した: {err}")
                });

            assert_parity(
                &format!("metal SimdgroupTiled({cfg:?}) gemm m={m} n={n} k={k}"),
                &actual,
                &expected,
            );
        }
    }

    /// デバイス上限直接検証（イシュー #532 受け入れ基準「SMEM 上限内の実機確認」）:
    /// `MetalGemm::pipeline_for_tile` は候補が検証・パイプライン構築に失敗すると
    /// `fallback_chain` で単一 simdgroup へサイレントにフォールバックするため
    /// （`crate::gemm` 参照）、`dispatch_variant` の PASS だけでは各候補が実際に
    /// デバイス上限内で動いた証明にならない。
    ///
    /// 従前は `TileConfig::validate` を SMEM のみ実デバイス値（`maxThreadgroupMemoryLength`）
    /// で直接呼び、スレッド数上限（`maxTotalThreadsPerThreadgroup`）は
    /// `MTLComputePipelineState` 構築前には取得できないという理由で Apple Silicon の
    /// 一般値 1024 を仮定していた。これでは候補が一度もコンパイル・パイプライン
    /// 構築されないため、実測ではなく仮定に基づく検証に留まり、フォールバックの穴を
    /// 塞げていなかった（イシュー #532・PR #651 codex-review 指摘 P2/P3）。
    ///
    /// `MetalGemm::resolve_tile_config`（`pub(crate)`。PR #651 codex-review 再指摘 P1 で
    /// `#[doc(hidden)] pub` から変更。本 `mod tests` はクレート境界の内側のため届く。
    /// `crate::gemm` 参照）は実際に
    /// `MTLComputePipelineState` を構築し、SMEM（構築前の事前検証）・スレッド数
    /// （構築後の実測 `maxTotalThreadsPerThreadgroup`）の両方をデバイス実測値で検証
    /// したうえで採用構成を返す。返り値が `cfg` と一致することを assert することで、
    /// 「1024 固定仮定」に依らず両上限を実測で確認しつつフォールバックを検知する。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn all_tile_candidates_validate_under_actual_device_shared_memory_limit() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = crate::gemm::MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for &cfg in CANDIDATES {
            let resolved = gemm.resolve_tile_config(&ctx, cfg).unwrap_or_else(|err| {
                panic!(
                    "candidate {cfg:?} は実デバイス上限（SMEM・スレッド数）でのパイプライン \
                     構築・検証に失敗した: {err}"
                )
            });
            assert_eq!(
                resolved, cfg,
                "candidate {cfg:?} が実デバイス上限で {resolved:?} へサイレントに \
                 フォールバックした（SMEM・スレッド数いずれかの実測上限超過）"
            );
        }
    }

    /// 直接ロード経路（`staged=false`）構成のフォールバック検知
    /// （codex-review 再指摘対応。イシュー #532・PR #651）。
    ///
    /// `CANDIDATES` は `select` の自動選択対象のみで全て `staged=true`
    /// （MLX classic 経路〈#532〉の追加 3 構成も含め本ファイル中に
    /// `staged: false` の要素はない）ため、`all_tile_candidates_match_*`
    /// 系（`CANDIDATES` を巡回するテスト）は `staged=false` 構成を一切
    /// 検証しない。`tests/gemm_dynamic_tile_parity.rs` の
    /// `direct_load_path_matches_cpu_reference` 系は `run_case` から
    /// `resolve_tile_config` 呼び出しが外れているため（同ファイルの
    /// `run_case` コメント参照）、直接ロード経路固有の
    /// `TileConfig { staged: false, .. }` がサイレントに
    /// `TileConfig::SINGLE_SIMDGROUP_8X8` へフォールバックしても
    /// 統合テストの数値一致確認だけでは検知できない穴が残っていた
    /// （codex-review 指摘 `BUGBOT_BUG_ID: c65127ea-56c2-4c52-95c2-604b5739cf40`）。
    /// 本テストはその穴を埋めるクレート内検証で、`resolve_tile_config`
    /// （`pub(crate)`）で実際に採用された構成が指定 `cfg` と一致することを
    /// 確認する。ここで使う `cfg` の値は
    /// `tests/gemm_dynamic_tile_parity.rs` の
    /// `direct_load_path_matches_cpu_reference` /
    /// `direct_load_path_matches_cpu_reference_non_multiple_of_tile` が
    /// 使う `TileConfig`（`bm=32,bn=32,bk=16,wm=2,wn=2,staged=false`）と
    /// 同期させること（形状のみが異なり構成自体は共通のため 1 回の検証で足りる）。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn direct_load_path_config_resolves_without_fallback() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = crate::gemm::MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let cfg = TileConfig {
            bm: 32,
            bn: 32,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: false,
        };

        let resolved = gemm.resolve_tile_config(&ctx, cfg).unwrap_or_else(|err| {
            panic!("direct-load 構成 {cfg:?} のパイプライン構築・検証に失敗した: {err}")
        });
        assert_eq!(
            resolved, cfg,
            "direct-load 構成 {cfg:?} が実デバイス上でサイレントに {resolved:?} へ \
             フォールバックした（構成失敗を検知できていない）"
        );
    }
}
