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
                write!(f, "bm={bm} is not a multiple of wm*8={}", wm * 8)
            }
            TileConfigError::BnNotDivisibleByWn8 { bn, wn } => {
                write!(f, "bn={bn} is not a multiple of wn*8={}", wn * 8)
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

    /// threadgroup 1 個あたりのスレッド数（`wm*wn*32`。1 simdgroup = 32
    /// スレッド）。`crate::gemm` のディスパッチが
    /// `threadsPerThreadgroup = (wm*wn*32, 1, 1)` を構成する際に使う。
    pub fn thread_count(&self) -> u32 {
        self.wm * self.wn * 32
    }

    /// A タイル（`bm`×`bk`）＋ B タイル（`bk`×`bn`）を保持する threadgroup
    /// 共有メモリのバイト数（`f32` 4 バイト換算）。`staged=false` の場合は
    /// 直接 `simdgroup_load` するため共有メモリを使わず 0 を返す
    /// （`shaders/gemm.metal` の `USE_TGP_STAGING` 分岐と対応）。
    ///
    /// `setThreadgroupMemoryLength` へ渡す実際のバイト長（16 バイト境界
    /// 整合が必要）は [`crate::gemm`] のディスパッチ側で `.max(16)` して
    /// 決定する（本メソッドが返す 0 バイトをそのまま渡さない。bugbot
    /// 指摘・#253 レビュー）。`staged=true` の場合は `bm`/`bk`/`bn` が
    /// [`TileConfig::validate`] により常に 8 の倍数へ制約されるため、この
    /// 戻り値は常に 256 以上かつ 16 の倍数になる。
    pub fn shared_mem_bytes(&self) -> u32 {
        if !self.staged {
            return 0;
        }
        (self.bm * self.bk + self.bk * self.bn) * 4
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
        if self.wm == 0 || !self.bm.is_multiple_of(self.wm * 8) {
            return Err(TileConfigError::BmNotDivisibleByWm8 {
                bm: self.bm,
                wm: self.wm,
            });
        }
        if self.wn == 0 || !self.bn.is_multiple_of(self.wn * 8) {
            return Err(TileConfigError::BnNotDivisibleByWn8 {
                bn: self.bn,
                wn: self.wn,
            });
        }
        if self.bk == 0 || !self.bk.is_multiple_of(8) {
            return Err(TileConfigError::BkNotMultipleOfEight { bk: self.bk });
        }

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
    // 根拠。実機ベンチによる効果確認は後続スコープ）。SMEM は
    // `(64*32+32*32)*4=12288` バイトで、既存候補の最大 8192 バイトを
    // 上回るが 32KiB 上限内（`TileConfig::validate` で機械検証）。
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
    fn shared_mem_bytes_sums_a_and_b_tiles_when_staged() {
        // A: 64x16, B: 16x64 -> (1024+1024)*4 = 8192 バイト。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes(), 8192);
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
        assert_eq!(cfg.shared_mem_bytes(), 8192);
        let err = cfg.validate(1024, 4096).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::ExceedsSharedMemory {
                bytes: 8192,
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
    fn bk32_candidate_shared_mem_is_12288_bytes_within_32kib_limit() {
        // (64,32,32,2,2): A=64x32, B=32x32 -> (2048+1024)*4 = 12288 バイト。
        // 既存最大 8192 バイトを上回るが 32KiB（32768 バイト）以内である
        // ことを固定する（イシュー #532 計画「現状分析」節の事前検証値）。
        let cfg = TileConfig {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes(), 12288);
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
    /// `crate::gemm` 参照）で実際に採用された構成を事前取得し `cfg` と一致する
    /// ことを assert してからディスパッチすることで、フォールバックが起きた
    /// 場合は本テスト自体を失敗させる。
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
    /// `MetalGemm::resolve_tile_config`（`pub(crate)`。`crate::gemm` 参照）は実際に
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
}
