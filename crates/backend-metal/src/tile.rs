//! 動的タイル選択（TASK-1.8f・#188）: `gemm_simdgroup_tiled` カーネル用の
//! BM/BN/BK/WM/WN パラメータと行列サイズ別の候補選択ロジック。
//!
//! `crate::gemm::MetalGemm::dispatch_auto` が [`select`] で `(m, n, k)`
//! から [`TileConfig`] を決定し、`crate::pipeline::make_pipeline_with_constants`
//! が本モジュールの `TileConfig::function_constants` 相当の値（MSL
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
//! `cargo test -p fandhe-ai-backend-metal` でも単体テストが回るようにしてある
//! （`crate::pad` と同じ設計判断。本ファイル冒頭のコメント参照）。
//!
//! **選択閾値は暫定値**: 下記 [`select`] のサイズ閾値・候補パラメータは
//! MLX steel の実装傾向を参考にした暫定初期値であり、Apple Silicon 実機
//! での `examples/gemm_bench.rs` 実測（`docs/perf/metal-gemm-dynamic-tile.md`
//! に記録）で確定させる前提（イシュー #188 計画のスコープ）。**厳密一致
//! する `(m, n, k)` の実測点（[`select_with_occupancy`] 冒頭の厳密一致
//! テーブル参照）は測定済み最良候補で確定済み**: イシュー #744
//! （2026-08-19）で `m == n == k` の 512/1024/2048/4096 の 4 点を
//! `CANDIDATES[3]`（32x32/bk16/staged）一律選択で確定させたが、
//! #1038〜#1040 の staged 経路変更を経てイシュー #1039（2026-08-31/09-01）
//! で `CANDIDATES` 全 8 候補を再実測した結果 `CANDIDATES[3]` はもはや
//! 最良候補ではなく、この 4 点は測定済み最良候補（`CANDIDATES[5]`／
//! `[6]`／`[1]`／`[2]`）へ個別に更新した。同じ #1039 実測で、`m != n` の
//! 準正方長方形 `(1536,1024,1024)` についても 2 回計測（run1/run2）いずれも
//! 最良候補が一致したため測定済み最良候補を厳密一致テーブルへ追加した
//! （判断根拠・数値は `docs/perf/metal-gemm-tile-table.md`）。`m == n` だが
//! `k != m`（K 未実測の正方出力。例: (2048,2048,64)）・`(1024,1536,1536)`
//! は 2 回計測で最良候補の順位が入れ替わった（プロセス間変動）ため、
//! 「2 回一致した点のみ反映する」方針に従い厳密一致テーブルへは含めない
//! （`docs/perf/metal-gemm-tile-table.md` §5「順位不安定のため反映しない」
//! 節）。厳密一致テーブルの各エントリは `select_with_occupancy` 内部で
//! `shape_cfg` に合流させ、既存の occupancy 縮退判定（`params` が `Some`
//! の場合）を迂回しない（P1・codex-review 指摘・PR #1108 レビュー対応）。
//! **上記の厳密一致点以外**（`m == n == k` の未測定サイズ・`m > 4096` の
//! 正方立方形状・厳密一致点以外の K 未実測正方出力・準正方長方形）は
//! 引き続き暫定値のまま（実測は厳密に一致する `(m, n, k)` の点のみで
//! あり、この範囲を超えて性能選択閾値を無制限に拡張するのは根拠不一致に
//! なるため #744 是正前の挙動を維持する。PR #760 codex-review 指摘対応・
//! #1039 でも同一方針を踏襲）。
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
/// `crate::gemm::MetalGemm` のパイプライン遅延キャッシュ構築時に、
/// 不成立構成を候補から除外し次善構成へフォールバックする判断材料になる
/// （fail-closed。計画「パイプライン管理」節）。
///
/// 本 enum は `pub enum` かつクレート公開面（`crate::tile` は `pub mod`）
/// のため、variant の追加・削除はダウンストリームの網羅的 `match` を
/// 破壊しうる（`#[non_exhaustive]` を後付けする変更自体も同様に破壊的。
/// イシュー #535・codex-review 指摘・PR #672）。このため本 enum の
/// variant 集合は既存のまま変更しない: `staged=true` 構成のベクトル化
/// ロード（float4）整除制約は、新規 variant を増やす代わりに既存の
/// `BkNotMultipleOfEight`/`BnNotDivisibleByWn8` 検査（8 整除）が
/// `VEC_WIDTH`（4）整除を数学的に包含することを [`validate`](TileConfig::validate)
/// の実装コメントとテスト（本ファイル末尾 `validate_ok_implies_vec_width_divisibility`）
/// で担保する。
///
/// 同じ理由（`#[non_exhaustive]` 後付け自体が既存の網羅的 `match` を破壊する）
/// から、イシュー #538 codex-review 指摘 P1 再指摘対応・PR #673 でも
/// `PadNotMultipleOfFour`・`PadWithoutStaging` variant 追加案を取り止めた。
/// `pad`（構造体フィールドではなく [`TileConfig::pad`] が `staged` から導出。
/// 本ファイル冒頭 [`TileConfig`] ドキュメント参照）は型の設計自体で不変条件を
/// 保証するため、両 variant は追加せずとも到達不能に帰着する。
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

    /// `shaders/gemm.metal` の `USE_TGP_STAGING` 協調ロード（イシュー
    /// #533）が使う `float4`（128bit）ベクトルロード幅（要素数）。
    /// A/B タイルの行境界を 4 要素グループがまたがないという gemm.metal
    /// 側のアラインメント前提（同ファイル `USE_TGP_STAGING` 分岐のコメント
    /// 参照。MLX `loader.h` の `BCOLS % n_reads == 0` 相当）を、
    /// [`validate`](Self::validate) の既存 8 整除検査（`bk % 8 == 0`・
    /// `bn % (wn*8) == 0` ⟹ `wn >= 1` のため `bn % 8 == 0`）が数学的に
    /// 包含することをテスト（本ファイル末尾
    /// `validate_ok_implies_vec_width_divisibility`）で固定する（イシュー
    /// #535 計画「設計方針」節）。専用の検査 variant は追加しない
    /// （`TileConfigError` の公開面を変更しないための判断。上記 doc 参照）。
    pub const VEC_WIDTH: u32 = 4;

    /// `shaders/gemm.metal` の `gemm_simdgroup_tiled_f16`（イシュー #797）が
    /// 使う `half` 版協調ロードのベクトルロード幅（要素数）。half は
    /// 128bit = 8 要素のため [`VEC_WIDTH`](Self::VEC_WIDTH)（f32 の 4 要素）
    /// の 2 倍になる。[`validate`](Self::validate) の既存 8 整除検査
    /// （`bk % 8 == 0`・`bn % (wn*8) == 0` ⟹ `bn % 8 == 0`）がちょうど
    /// `VEC_WIDTH_F16`（8）整除を包含する（`VEC_WIDTH`〈4〉整除を包含する
    /// のと同じ論法。本ファイル末尾
    /// `validate_ok_implies_vec_width_f16_divisibility` で固定する）。
    pub const VEC_WIDTH_F16: u32 = 8;

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
    /// 無意味なため常に `0` を返す。`TGP_PAD_ELEMS`（4）は
    /// [`VEC_WIDTH`](Self::VEC_WIDTH) と同じ 4 であり、`pad` を加算した
    /// 行ストライド `lda`/`ldb` も常に 4 の倍数のまま維持される
    /// （`shaders/gemm.metal` の `USE_TGP_STAGING` float4 ロードの
    /// アラインメント前提。イシュー #535）。
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
    /// 整合が必要）は `crate::gemm` のディスパッチ側で `.max(16)` して
    /// 決定する（本メソッドが返す 0 バイトをそのまま渡さない。bugbot
    /// 指摘・#253 レビュー）。`staged=true` の場合は `bm`/`bk`/`bn` が
    /// [`TileConfig::validate`] により常に 8 の倍数へ制約され、`pad`
    /// （[`TileConfig::pad`]）は `staged` からの導出値として常に 4 の倍数
    /// （0 または `TGP_PAD_ELEMS=4`）になる（イシュー #538 codex-review 指摘
    /// P1 再指摘対応・PR #673 で実行時検証ではなく型の設計自体が保証する
    /// 方式へ変更した。本ファイル冒頭 [`TileConfig`] ドキュメント参照）ため、
    /// この戻り値は常に 256 以上かつ 16 の倍数になる。
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

    /// `shaders/gemm.metal` の `gemm_simdgroup_tiled_f16`（イシュー #796・
    /// エピローグのタイル粒度統合はイシュー #797）が確保する threadgroup
    /// 共有メモリのバイト数。`shared_mem_bytes`（f32 版）とレイアウトの
    /// 考え方は同じだが、2 点異なる:
    ///
    /// 1. staged 経路の A・B タイル要素は `half`（2 バイト）単位で確保する
    ///    （f32 版は `f32` 4 バイト単位）。
    /// 2. `gemm_simdgroup_f16` と同じ理由（`simdgroup_store` が
    ///    `simdgroup_float8x8` から `device half*` へ直接 store できない。
    ///    #380 実機 spike で確定済み）により、`staged` の有無を問わず常に
    ///    エピローグ staging 領域（f32）を追加で確保する。`gemm_simdgroup_tiled_f16`
    ///    の threadgroup メモリレイアウトはタイル領域（staged のみ）＋
    ///    エピローグ領域（常時）の順（同カーネル冒頭コメント参照）。
    ///
    /// **エピローグ領域のサイズ（イシュー #797 でタイル粒度へ拡大）**: 従来
    /// （#796）は 8x8 acc タイル 1 個分（64 要素）の staging スラブを
    /// simdgroup ごとに使い回し、acc タイル毎に `simdgroup_store`→
    /// `simdgroup_barrier`→書き戻し→`simdgroup_barrier` を回していた
    /// （barrier が 1 simdgroup あたり `2 * acc_rows * acc_cols` 回発生）。
    /// #797 では各 simdgroup 専用スラブを担当サブタイル全体
    /// （`sub_bm = bm/wm` × `sub_bn = bn/wn` 要素）へ拡大し、全 acc タイルを
    /// 一括 store → `simdgroup_barrier` 1 回 → 一括書き戻しへ再構成した
    /// （barrier 回数を `2 * acc_rows * acc_cols` から 1 回へ削減）。
    /// 1 simdgroup あたりの必要量は `sub_bm * sub_bn * 4` バイトで、
    /// `wm * wn` simdgroup 分を合計すると `wm*sub_bm * wn*sub_bn * 4 =
    /// bm * bn * 4` バイトに簡約される（`sub_bm = bm/wm`・`sub_bn = bn/wn`
    /// の定義より。`bm`/`bn`/`wm`/`wn` は [`TileConfig::validate`] が
    /// `bm % (wm*8) == 0`・`bn % (wn*8) == 0` を要求するため `wm | bm`・
    /// `wn | bn` が常に成立し割り切れる）。
    ///
    /// タイル領域の要素数合計（`bm*(bk+pad) + bk*(bn+pad)`）は `bm`/`bk`/`bn`
    /// が 8 の倍数・`pad` が 0 または 4（[`TileConfig::validate`] が保証する
    /// 不変条件）のため常に偶数になり、half 2 バイト単位で確保しても続く
    /// エピローグ領域の f32 4 バイト境界に整合する（`gemm_simdgroup_tiled_f16`
    /// が `reinterpret_cast<threadgroup float*>` で安全に参照できる根拠）。
    ///
    /// `crate::gemm::MetalGemm::pipeline_for_tile_f16`（macOS 限定・非公開の
    /// private メソッドのため intra-doc link にはしない）が
    /// [`TileConfig::validate`]（f32 版と共通）に加えてこのメソッドで
    /// デバイス上限超過を追加検査する（f32 版 `validate` は f32 単位の
    /// `shared_mem_bytes` しか見ないため、f16 版の実際の確保量を別途
    /// 検査する必要がある）。
    pub fn shared_mem_bytes_f16(&self) -> u32 {
        // エピローグ領域は `staged` を問わず常に必要（上記ドキュメント
        // コメント 2 点目）。`bm*bn*4`（#797 でタイル粒度へ拡大。上記
        // ドキュメントコメント参照）を `shared_mem_bytes`（f32 版）と同じ
        // checked u64 演算・飽和方針で計算する。
        let epilogue = (self.bm as u64)
            .checked_mul(self.bn as u64)
            .and_then(|v| v.checked_mul(4))
            .unwrap_or(u64::MAX);

        if !self.staged {
            return epilogue.min(u32::MAX as u64) as u32;
        }

        let bm = self.bm as u64;
        let bn = self.bn as u64;
        let bk = self.bk as u64;
        let pad = self.pad() as u64;
        let compute = || -> Option<u64> {
            let a_row = bk.checked_add(pad)?;
            let b_row = bn.checked_add(pad)?;
            let a_tile = bm.checked_mul(a_row)?;
            let b_tile = bk.checked_mul(b_row)?;
            let tile_bytes = a_tile.checked_add(b_tile)?.checked_mul(2)?; // half = 2 バイト
            tile_bytes.checked_add(epilogue)
        };
        compute().unwrap_or(u64::MAX).min(u32::MAX as u64) as u32
    }

    /// `bm/bn/bk/wm/wn` の整除制約・デバイス上限（`max_threads_per_tg`:
    /// `MTLComputePipelineState::maxTotalThreadsPerThreadgroup`、
    /// `max_shared_mem_bytes`: `MTLDevice::maxThreadgroupMemoryLength`）
    /// との整合を検証する。`crate::gemm::MetalGemm` のパイプライン構築
    /// （macOS 実機のみ到達）から呼ばれるほか、本ファイル末尾の単体テストで
    /// GPU 非依存に検証できる（Linux CI でも実行可能）。
    pub fn validate(
        &self,
        max_threads_per_tg: u32,
        max_shared_mem_bytes: u32,
    ) -> Result<(), TileConfigError> {
        // ベクトル化ロードの整除制約（イシュー #535）: `staged=true` の
        // 構成は gemm.metal の `USE_TGP_STAGING` float4 協調ロード経路
        // （イシュー #533）を通るため、A/B タイルの行長 `bk`/`bn` が
        // `VEC_WIDTH`（4）の倍数であることを要求する（MLX `loader.h` の
        // `BCOLS % n_reads == 0` 相当）。
        //
        // 専用の検査 variant（`TileConfigError` への新規追加）は設けない。
        // 下記の既存 8 整除検査（`bk % 8 == 0`・`wn >= 1` かつ
        // `bn % (wn*8) == 0` ⟹ `bn % 8 == 0`）が `VEC_WIDTH`（4）整除を
        // 数学的に包含するため、`staged=true` かつこれらの検査を通る
        // 構成は必ず `bk`/`bn` が 4 の倍数になる（`8 | x ⟹ 4 | x`）。
        // この不変条件は本ファイル末尾のテスト
        // `validate_ok_implies_vec_width_divisibility` で固定する。
        // enum への variant 追加・`#[non_exhaustive]` 化はいずれも
        // ダウンストリームの網羅的 `match` を破壊しうるため（codex-review
        // 指摘・PR #672）、`TileConfigError` の公開面は変更しない
        // （型定義側の doc コメント参照）。

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
///
/// **`CANDIDATES[0]`（64x64 大形状）は #744 是正後、`select` の形状判定では
/// `m == n == k` かつ `m <= 4096`（実測範囲内の真の正方立方形状）に対しては
/// 選ばれない**（2026-08-19 M4 Max 実機実測で `m == n == k` 全帯域
/// 〈512〜4096〉において `CANDIDATES[3]`〈32x32〉に一貫して劣後することを
/// 確認したため。`select_with_occupancy` 本体コメント・
/// `docs/perf/metal-tile-select-correction.md` 参照）。**`m == n == k` かつ
/// `m > 4096`（実測範囲外の正方立方形状）、`m == n` でも `k != m`（K 未
/// 実測の正方出力。例: (2048,2048,64)）、および `m != n` の準正方大形状
/// 長方形（`m,n >= 512` かつ縦長・横長いずれにも非該当。例: 1536x1024）は
/// #744 実測対象外のため、引き続き `CANDIDATES[0]` を返す**（PR #760
/// codex-review 指摘対応: 実測は `m == n == k` の 512/1024/2048/4096 の 4
/// 点のみであり、この範囲を超えて性能選択閾値を無制限に拡張するのは根拠
/// 不一致になるため #744 是正前の挙動を維持する）。それでも配列からは
/// 削除しない: `select` の添字依存（上記）を壊すうえ、`fallback_chain`・
/// occupancy 縮退判定（縦長/横長・準正方長方形・K 未実測正方出力・4096
/// 超の正方立方形状経路の比較対象）・#747（サイズ帯条件分岐）での再利用
/// 対象として残す。
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
/// `crate::gemm::MetalGemm::pipeline_for_tile` が構成の検証・パイプライン
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

/// イシュー #1039 の厳密一致テーブル（`exact_match_cfg`。下記
/// [`select_with_occupancy_for_device`] 内）が対象とする GPU コア数
/// （M4 Max 40 コア構成）。[`crate::device::probe_gpu_core_count`]（IOKit
/// 実測。macOS 限定）または [`OccupancyParams::gpu_core_count`] の実測値と
/// 比較し、一致した機種にのみ実測テーブルを適用する（P1・codex-review
/// 指摘・PR #1108 レビュー: `select()` がデバイス情報を受け取らず全 Apple
/// Silicon 機種へ無条件適用されていた問題への対応。`AGENTS.md`「実機固有値
/// をロジックへ直書きしない」規約に対し、値そのものはハードコードのままだが
/// 適用範囲を実測機種へ限定することで、M1〜M5 等の未実測機種は本テーブルを
/// 経由せず既存の形状クラス判定（縦長・横長・正方立方・大形状フォール
/// バック）へ従来どおり流れる）。
///
/// **単独では機種を一意に識別できない（P1・codex-review 指摘・PR #1108
/// レビュー）**: GPU コア数 40 は M4 Max だけでなく M3 Max の 40 コア構成
/// にも該当しうるため、本定数単独を機種ゲートに使わない。`verify_m4_max`
/// が [`crate::device::probe_soc_brand_string`]（SoC ブランド文字列の実測。
/// 例: `"Apple M4 Max"`）と組み合わせて検証する。
const M4_MAX_GPU_CORE_COUNT: u32 = 40;

/// `verify_m4_max` が [`crate::device::probe_soc_brand_string`] の実測値と
/// 完全一致比較する SoC ブランド名（P1・codex-review 指摘・PR #1108
/// レビュー）。
const M4_MAX_SOC_BRAND: &str = "Apple M4 Max";

/// `verify_m4_max` からのみ構築可能な、M4 Max 実機として検証済みである
/// ことを表す opaque 型（P1・codex-review 再指摘・PR #1108 レビュー）。
///
/// **背景（旧実装の問題）**: 是正前は `select_for_device`／
/// `select_with_occupancy_for_device` が生の `Option<u32>`（GPU コア数）を
/// 受け取り、内部で `gpu_core_count == Some(M4_MAX_GPU_CORE_COUNT)`
/// （コア数一致のみ）を検査していた。呼び出し元が `verify_m4_max` の戻り
/// 値をそのまま渡す契約は doc comment 上の記述に過ぎず型・実装では強制され
/// ないため、外部利用者や将来の呼び出し元が実測 GPU コア数（例: M3 Max の
/// 40 コア構成）を直接 `Some(40)` として渡すと、SoC ブランド照合
/// （`verify_m4_max`）を経ずに M4 Max 専用の厳密一致テーブルが誤って
/// 有効化されてしまう。
///
/// **是正方針**: フィールドを非公開にし、`verify_m4_max`（GPU コア数と
/// SoC ブランド文字列の両方が一致する場合にのみ構築）以外の経路では本型の
/// 値を作れない構造にする。`select_for_device`／
/// `select_with_occupancy_for_device` の `gpu_core_count` 引数を本型へ
/// 変更したことで、未検証の `u32` を渡してブランド照合を迂回することが
/// コンパイル時に不可能になる（`AGENTS.md`「実機固有値のハードコード回避」
/// 規約・公開 API 契約維持への対応）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedM4MaxGpuCoreCount(u32);

impl VerifiedM4MaxGpuCoreCount {
    /// 検証済みの GPU コア数を返す（常に `M4_MAX_GPU_CORE_COUNT` と
    /// 一致する。デバッグ表示・ログ等、値そのものを必要とする呼び出し元
    /// 向けのアクセサ。`M4_MAX_GPU_CORE_COUNT` は非公開定数のため、
    /// intra-doc link ではなくバッククォートのコードスパン表記とする
    /// （rustdoc `-D rustdoc::private-intra-doc-links` が非公開アイテムへの
    /// intra-doc link をビルド失敗にするため。PR #1108 レビュー）。
    pub fn gpu_core_count(self) -> u32 {
        self.0
    }
}

/// GPU コア数と SoC ブランド文字列を組み合わせ、イシュー #1039 の厳密一致
/// テーブル（`exact_match_cfg`）を適用してよい M4 Max 実機かどうかを検証
/// する（P1・codex-review 指摘・PR #1108 レビュー）。
///
/// **GPU コア数だけでは機種を一意に識別できない**: `gpu_core_count == 40`
/// は M4 Max だけでなく M3 Max の 40 コア構成にも該当しうる
/// （`crate::device` モジュールドキュメンテーションコメント「機種識別子
/// からの対応表推定」節が却下した「`hw.model` からの対応表推定」とは異なり、
/// 本関数は SoC ブランド文字列という OS が直接返す実測値を追加シグナルと
/// して使う点に注意）。`gpu_core_count` が `M4_MAX_GPU_CORE_COUNT`
/// （40）と一致し、かつ `soc_brand` が `M4_MAX_SOC_BRAND`（`"Apple M4
/// Max"`）と完全一致する場合にのみ [`VerifiedM4MaxGpuCoreCount`] を返す。
/// いずれかが不一致・取得不能（`None`）の場合は `None` を返し、
/// [`select_for_device`]／[`select_with_occupancy_for_device`] は厳密一致
/// テーブルを経由せず既存の形状クラス判定のみへ安全側でフォールバックする。
///
/// **本関数が [`VerifiedM4MaxGpuCoreCount`] の唯一の構築経路**（フィールド
/// 非公開のため。P1・codex-review 再指摘・PR #1108 レビュー）。
///
/// `crate::context::MetalContext::new` がデバイス初期化時に 1 回だけ呼び、
/// 結果を [`select_for_device`] へ渡す想定（ディスパッチ経路のホットパスに
/// FFI を持ち込まないための設計。`MetalContext::occupancy_params` と同じ
/// 判断）。
///
/// **`pub(crate)` に閉じる理由（P1・codex-review 再指摘・PR #1108
/// レビュー）**: 本関数は実機プローブを一切行わず、呼び出し元が渡した
/// `gpu_core_count`／`soc_brand` の文字列比較のみで判定する。以前は
/// `pub` だったため、外部呼び出し元（または未実測機種を利用する crate
/// 内呼び出し元）が `verify_m4_max(Some(40), Some("Apple M4 Max"))` の
/// ように実機プローブ（[`crate::device::probe_gpu_core_count`]・
/// [`crate::device::probe_soc_brand_string`]）を経由しない任意値を直接
/// 渡すことで、M4 Max 実機として未検証のまま [`VerifiedM4MaxGpuCoreCount`]
/// を偽造でき、M3 Max 等の未検証機種でも M4 Max 専用の厳密一致テーブルを
/// 有効化できてしまっていた（`AGENTS.md`「実機固有値をロジックへ直書き
/// しない」規約が求める安全側運用に反する）。本関数を `pub(crate)` に
/// 閉じ、公開経路を実機プローブを内部で行う
/// [`crate::context::MetalContext::verified_m4_max_gpu_core_count`] のみへ
/// 限定することで、crate 外部から任意値でトークンを生成する経路を
/// コンパイル時に排除する。
pub(crate) fn verify_m4_max(
    gpu_core_count: Option<u32>,
    soc_brand: Option<&str>,
) -> Option<VerifiedM4MaxGpuCoreCount> {
    if gpu_core_count == Some(M4_MAX_GPU_CORE_COUNT) && soc_brand == Some(M4_MAX_SOC_BRAND) {
        Some(VerifiedM4MaxGpuCoreCount(M4_MAX_GPU_CORE_COUNT))
    } else {
        None
    }
}

/// `(m, n, k)` から [`TileConfig`] を選択する（後方互換入口）。
///
/// **公開 API 互換性（P1・codex-review 指摘・PR #1108 レビュー）**: 本関数
/// の 3 引数シグネチャは PR #1108 以前から存在する公開 API
/// （`fandhe-ai-backend-metal` は crates.io 公開クレート）であり、破壊的
/// 変更を避けるため維持する。イシュー #1039 の M4 Max 実測厳密一致テーブル
/// はデバイス情報（`gpu_core_count`）を必要とするため、これを受け取る版は
/// 別名 [`select_for_device`] として追加し、本関数は `gpu_core_count: None`
/// で委譲する（＝厳密一致テーブルは経由せず既存の形状クラス判定のみ。
/// PR #1108 以前と完全一致する挙動）。
pub fn select(m: usize, n: usize, k: usize) -> TileConfig {
    select_for_device(m, n, k, None)
}

/// `(m, n, k)` と実測 GPU コア数から [`TileConfig`] を選択する
/// （[`select`] のデバイス情報対応版。P1・codex-review 指摘・PR #1108
/// レビューで新設）。[`select_with_occupancy_for_device(m, n, k,
/// gpu_core_count, None)`][select_with_occupancy_for_device] への委譲で
/// あり、occupancy 判定は行わない（形状のみによる選択）。
///
/// `gpu_core_count` はイシュー #1039 の厳密一致テーブル（M4 Max 実測）の
/// 適用可否判定にのみ使う（`verify_m4_max` 参照。呼び出し元が実機値を
/// 取得できない・意図的に無効化したい場合は `None` を渡せば従来の形状
/// クラス判定のみへフォールバックする）。**引数の型が
/// [`VerifiedM4MaxGpuCoreCount`]（`verify_m4_max` からのみ構築可能な
/// opaque 型）であるため、未検証の GPU コア数（生の `u32`）を渡して
/// ブランド照合を迂回することはコンパイル時に不可能**（P1・codex-review
/// 再指摘・PR #1108 レビュー）。
pub fn select_for_device(
    m: usize,
    n: usize,
    k: usize,
    gpu_core_count: Option<VerifiedM4MaxGpuCoreCount>,
) -> TileConfig {
    select_with_occupancy_for_device(m, n, k, gpu_core_count, None)
}

/// `(m, n, k)` と実測 GPU コア数から [`TileConfig`] を選択する（occupancy
/// 判定込み。イシュー #542。[`select_with_occupancy`] のデバイス情報対応版
/// で、P1・codex-review 指摘・PR #1108 レビューにより本名へ改名した）。
/// **`crate::gemm::MetalGemm::dispatch_auto` の入口としては不採用確定**
/// （本番ディスパッチは引き続き [`select_for_device`] を使う。イシュー
/// #747 で、#744 是正（段 1 が実測範囲内の正方立方形状に対し occupancy
/// 縮退を経ず直接 `CANDIDATES``[3]` を返すよう是正済み）により本関数の
/// 縮退経路が実測帯域〈512/1024/2048/4096〉で [`select_for_device`] と
/// 常に同一結果へ収束することを確認し、「小サイズ帯のみ occupancy 有効化」
/// という #747 の目的は #744 是正へ吸収されたと判断した。`crate::gemm`
/// モジュールドキュメンテーションコメント・`docs/perf/
/// metal-gemm-occupancy-select.md`「#747 判断」節参照）。
/// `examples/gemm_bench.rs` の旧/新比較セクションから明示的に呼ばれる。
///
/// 2 段階判定（MFA 型方法論。本ファイル「occupancy 目標算出」節参照）:
///
/// 1. **形状判定**: [`select_for_device`] と同一ロジックで形状優先の構成を
///    決める（閾値はパディング前の元次元 `m`/`n`/`k` に対して判定する。
///    `crate::pad::pad8` によるパディングは選択後・パイプライン確保直前に
///    `crate::gemm` 側で行う。本関数はパディングの有無を問わない純粋関数）。
/// 2. **occupancy 縮退**: 段 1 の結果が大タイル系（`CANDIDATES``[0..=2]`。
///    64×64・64×32・32×64）かつ `params` が `Some` のとき、
///    [`actual_groups`] と [`OccupancyParams::ideal_groups`]（係数
///    [`IDEAL_GROUPS_MULTIPLIER_F32`]）を比較し、[`is_underoccupied`] なら
///    `CANDIDATES``[3]`（32×32 中形状）へ縮退する。[`TileConfig::
///    SINGLE_SIMDGROUP_8X8`]（微小形状フォールバック）へはこの経路で縮退
///    させない（段 1 の `SMALL` 判定のみが担う責務であり、occupancy 縮退の
///    対象は「大タイル→中タイル」の 1 段のみ）。
///
/// **fail-safe フォールバック**: 以下のいずれかに該当する場合は occupancy
/// 判定を無効化し、段 1 の結果をそのまま返す（現行 [`select_for_device`]
/// と完全一致。安全側。#541 doc §5 の残課題に対する確定方針）:
/// - `params` が `None`（呼び出し元が実機値を取得できなかった、または
///   意図的に occupancy 判定を無効化したい場合）
/// - [`actual_groups`] が `None`（`cfg.bm`／`cfg.bn` が 0。`CANDIDATES`
///   内の構成では実質発生しないが fail-safe として扱う）
/// - [`OccupancyParams::ideal_groups`] が `None`（コア数 0・係数 0・SMEM
///   予算超過によりデバイス上でタイルが同時常駐不可能・オーバーフロー）
///
/// いずれの分岐も panic しない（`.claude/rules/coding-rust.md`「本番経路で
/// unwrap/expect を使わない」）。
///
/// `gpu_core_count` はイシュー #1039 の厳密一致テーブルの適用可否判定に
/// のみ使う。**引数の型が [`VerifiedM4MaxGpuCoreCount`]（`verify_m4_max`
/// からのみ構築可能な opaque 型）であるため、未検証の GPU コア数（生の
/// `u32`）を渡してブランド照合を迂回することはコンパイル時に不可能**
/// （P1・codex-review 再指摘・PR #1108 レビュー）。
pub fn select_with_occupancy_for_device(
    m: usize,
    n: usize,
    k: usize,
    gpu_core_count: Option<VerifiedM4MaxGpuCoreCount>,
    params: Option<OccupancyParams>,
) -> TileConfig {
    const SMALL: usize = 64;
    const ASPECT_RATIO: usize = 2;
    const LARGE: usize = 512;
    // `m == n`（真の正方形状）の実測上限。イシュー #744・2026-08-19 M4 Max
    // 実機実測（5 回中央値）は 512/1024/2048/4096 の 4 点のみで、この帯域を
    // 超える正方形状（例: 8192）は実測対象外（codex-review 指摘・PR #760
    // レビュー）。実測範囲外まで無条件に CANDIDATES[3] へ広げると、大規模
    // GEMM の性能退行を検知できないまま許容してしまうため、実測済み上限で
    // 打ち切り、それを超える正方形状は #744 是正前の挙動（`m,n >= LARGE`
    // なら CANDIDATES[0]）へ安全側でフォールバックする。
    const SQUARE_MEASURED_MAX: usize = 4096;

    if m < SMALL || n < SMALL || k < SMALL {
        return TileConfig::SINGLE_SIMDGROUP_8X8;
    }

    // イシュー #1039・2026-08-31/09-01 M4 Max 実機実測（`CANDIDATES` 全 8
    // 候補・`examples/gemm_transpose_tile_sweep.rs`）による厳密一致テーブル。
    // 以下の完全一致する `(m, n, k)` に限り、下の形状クラス判定（縦長・
    // 横長・正方立方・大形状フォールバック）より優先して測定済み最良候補を
    // `shape_cfg` として採用する。**occupancy 縮退は迂回しない**（P1・
    // codex-review 指摘・PR #1108 レビュー）: 早期 `return` にはせず、下の
    // `shape_cfg` 決定に合流させることで、#744 由来の `m == n && k == m
    // && m <= SQUARE_MEASURED_MAX => CANDIDATES[3]` 分岐と構造上同じ扱いに
    // なる（=既存の occupancy 判定を M4 Max 実測値にも一様に適用する）。
    // `select_for_device()` 経由（`params: None`）では従来どおり縮退せず
    // `shape_cfg` をそのまま返すため、本番ディスパッチの挙動自体は変わらない。
    //
    // **機種ゲート（P1・codex-review 再指摘・PR #1108 レビュー）**: 本テーブル
    // は M4 Max（40 コア構成）実機実測のみが根拠であり、`select_for_device()`
    // は元々デバイス情報を受け取らないため無対応のままでは M1〜M5 を含む全
    // Apple Silicon 機種へ無条件適用されてしまう（`AGENTS.md`「実機固有値を
    // ロジックへ直書きしない」規約への抵触）。**GPU コア数だけでは機種を
    // 一意に識別できない**（`gpu_core_count == 40` は M4 Max だけでなく
    // M3 Max の 40 コア構成にも該当しうる）ため、`gpu_core_count` 引数の
    // 型自体を [`VerifiedM4MaxGpuCoreCount`]（[`verify_m4_max`] からのみ
    // 構築可能な opaque 型。フィールド非公開）にすることで、SoC ブランド
    // 照合（`crate::device::probe_gpu_core_count`〈GPU コア数の IOKit
    // 実測〉と `crate::device::probe_soc_brand_string`〈SoC ブランド文字列
    // の実測〉の両方一致）を経ずに本テーブルを有効化することがコンパイル
    // 時に不可能になっている（`crate::context::MetalContext::new` が 1 回
    // だけキャッシュした値を `verified_m4_max_gpu_core_count` 経由で渡す）。
    // 未検証（`None`）な機種は下の形状クラス判定（縦長・横長・正方立方・
    // 大形状フォールバック）のみへ従来どおり流れる。他機種での候補の優劣は
    // 未実測のため、共通性を実測するまでテーブルを機種非依存へ拡張しない
    // （`select_exact_match_table_is_gated_by_m4_max_gpu_core_count` 参照）。
    //
    // #744（2026-08-19）は `m == n == k` の 512/1024/2048/4096 の 4 点で
    // `CANDIDATES[3]`（32x32）一律選択を確定させたが、#1038〜#1040 の
    // staged 経路変更（タイル variant 群の整理・転置ロード境界確立等）を
    // 経て、同じ 4 点で `CANDIDATES` 全候補を再実測した結果
    // `CANDIDATES[3]` はもはや最良候補ではなくなっていた（例:
    // size=4096 で `CANDIDATES[3]` 8.24〜8.36 TFLOPS に対し最良候補
    // `CANDIDATES[2]` 9.76〜9.91 TFLOPS。詳細値・判断根拠は
    // `docs/perf/metal-gemm-tile-table.md`）。この 4 点は 2 回計測
    // （run1/run2）いずれも順位が一致している。また `(1536, 1024, 1024)`
    // （準正方長方形）も 2 回計測いずれも `CANDIDATES[1]` が最良で一致した
    // ため採用する。**`m == n` だが `k != m`（K 未実測の正方出力）帯域の
    // `(2048, 2048, 64)`／`(2048, 2048, 512)`、および準正方長方形
    // `(1024, 1536, 1536)` は 2 回計測で最良候補の順位が入れ替わった
    // （プロセス間変動。`docs/perf/metal-gemm-tile-table.md` §3・§5）ため
    // 本テーブルへ含めない**（`docs/perf/metal-bench-noise-protocol.md`
    // が指摘する既知の系統誤差源。2 回一致した点のみ反映する方針は本ファイル
    // 冒頭の判断基準どおりであり、単一 run の結果で分岐を追加しない。
    // 再計測での順位確認は out-of-scope-tracking.md に沿って追跡する）。
    // **実測は下記の厳密な `(m, n, k)` タプルのみ**であり、近傍値・未測定点
    // へは拡張しない（実測範囲外への無根拠拡張をしない方針。#744/PR #760
    // と同一判断軸）。
    let exact_match_cfg = if gpu_core_count.is_some() {
        match (m, n, k) {
            // 正方立方（m == n == k）実測点。2 回計測いずれも順位一致。
            (512, 512, 512) => Some(CANDIDATES[5]), // 64x32/bk32/wm2wn2
            (1024, 1024, 1024) => Some(CANDIDATES[6]), // 64x32/bk8/wm4wn1
            (2048, 2048, 2048) => Some(CANDIDATES[1]), // 64x32/bk16/wm2wn2（縦長候補だが正方形状でも最良）
            (4096, 4096, 4096) => Some(CANDIDATES[2]), // 32x64/bk16/wm2wn2（横長候補だが正方形状でも最良）
            // 準正方長方形（m != n・縦横比 < ASPECT_RATIO）実測点。2 回計測
            // いずれも `CANDIDATES[1]` が最良で一致。
            (1536, 1024, 1024) => Some(CANDIDATES[1]),
            _ => None,
        }
    } else {
        // M4 Max（40 コア）以外の機種、またはコア数取得不能（`None`）の
        // 場合は本テーブルを評価しない（上記「機種ゲート」節）。
        None
    };

    let tall = m >= n.saturating_mul(ASPECT_RATIO);
    let wide = n >= m.saturating_mul(ASPECT_RATIO);

    // 真の正方**立方**形状（`m == n == k`。縦長・横長いずれにも該当しない
    // 場合の部分集合）かつ実測範囲内（`m <= SQUARE_MEASURED_MAX`）は
    // CANDIDATES[3] を返す。イシュー #744・2026-08-19 M4 Max 実機実測（5 回
    // 中央値）で 512〜4096 の `m == n == k`（立方 GEMM）全帯域において
    // CANDIDATES[0]（64x64 staged）が CANDIDATES[3]（32x32/bk16/staged）に
    // 一貫して劣後することを確認した（size=2048: 64x64 staged ≈1.18 TFLOPS
    // に対し 32x32 staged ≈3.31 TFLOPS、最良候補比で約 2.8 倍の逸失）。旧
    // 分岐（#188 導入時の `docs/perf/metal-gemm-dynamic-tile.md` #381 計測
    // ではほぼ同等だった）はその後の staged 経路変更（#533 float4 協調
    // ロード・#538 TGP パディング・#572 prepared 境界確立）を経て逆転して
    // おり、本分岐撤去は実測追従の是正（詳細・判断式は
    // `docs/perf/metal-tile-select-correction.md`）。
    // 4096 超の正方形状は実測範囲外のため、下の `m,n >= LARGE` 分岐へ落ちて
    // #744 是正前の挙動（CANDIDATES[0]）を維持する（codex-review 指摘・PR
    // #760 レビュー。実測なしに性能選択閾値を無制限拡張しない）。
    //
    // **`k == m` を要件に含める（P1・codex-review 指摘・PR #760 レビュー）**:
    // 2026-08-19 実測は `(512,512,512)`〜`(4096,4096,4096)` の立方 GEMM
    // （`m == n == k`）のみを対象としており、`m == n` だが `k` が大きく
    // 異なる形状（例: `(2048,2048,64)`）はこの実測に含まれない。`k` を
    // 条件から外すと立方 GEMM の実測結果を任意の `k` を持つ正方出力へ
    // 無根拠に拡張してしまうため、`k == m` を要件に加えて実測範囲へ厳密に
    // 限定する。`k` が測定範囲外（`m == n` だが `k != m`）の場合は下の
    // `m,n >= LARGE` 分岐（#744 是正前の挙動）へ安全側でフォールバックする。
    //
    // **`m != n`（縦長・横長いずれにも該当しないが正方でもない準正方長方形。
    // 例: 1536x1024〈比 1.5:1〉）は #744 実測対象外**（codex-review 指摘・PR
    // #760 レビュー。2026-08-19 実測は `m == n == k` の 4 点のみで、この帯域へ
    // 一律 CANDIDATES[3] を広げる根拠がなかった）。安全側として #744 以前の
    // 挙動（`m,n >= LARGE` なら CANDIDATES[0]、それ未満は CANDIDATES[3]）を
    // そのまま維持する。イシュー #747（occupancy 選定式のサイズ帯条件分岐）
    // は `m == n == k` の実測帯域については #744 是正で目的が吸収されたと
    // 判断済み（`select_with_occupancy` 本体コメント・`docs/perf/
    // metal-gemm-occupancy-select.md`「#747 判断」節）。この準正方長方形帯域
    // の候補比較実測は #747 のスコープには含まれず、実測は Mac 実機セッション
    // （実機ツリー #408 系）に委ねる（`docs/perf/metal-tile-select-correction.md`
    // 「実機確認結果（記入欄）」節）。縦長・横長の分岐自体も 2026-08-19 実測が
    // 正方形状のみを対象としているため変更しない（安全側）。
    let shape_cfg = exact_match_cfg.unwrap_or_else(|| match (tall, wide) {
        (true, _) => CANDIDATES[1], // 64x32（縦長）
        (_, true) => CANDIDATES[2], // 32x64（横長）
        _ if m == n && k == m && m <= SQUARE_MEASURED_MAX => CANDIDATES[3], // 32x32（真の正方立方・実測範囲内。#744）
        _ if m >= LARGE && n >= LARGE => CANDIDATES[0], // 大形状（4096 超の正方・K 未実測の正方形状含む・準正方大形状長方形）。#744 実測対象外・是正前の挙動を維持
        _ => CANDIDATES[3], // 32x32（準正方中形状長方形。#744 是正前と同一挙動）
    });

    // occupancy 縮退の対象は「大タイル系」（CANDIDATES[3]〈中形状・32x32〉
    // より threadgroup 分担面積 `bm*bn` が大きい構成）を選んだ場合のみ。
    // CANDIDATES[3] 自体（既に中形状）は縮退不要、SINGLE_SIMDGROUP_8X8 は
    // 段 1 の SMALL 判定のみが返しうる（上の match の到達条件上ここには
    // 来ない）。
    //
    // **`CANDIDATES[0..=2]` への固定列挙ではなく `bm*bn` 比較にする（P1・
    // codex-review 指摘・PR #1108 レビュー）**: 厳密一致テーブル
    // （`exact_match_cfg`）が返す `CANDIDATES[5]`／`[6]` は `bm=64,bn=32`
    // と `CANDIDATES[1]`（縦長・大タイル系）と同一の threadgroup 分担面積
    // を持つため、`actual_groups` の算出式（`m`/`n`/`bm`/`bn` のみに依存）
    // は `CANDIDATES[1]` と同一であり、`CANDIDATES[1]` が対象なら
    // `CANDIDATES[5]`／`[6]` も同じ理由で under-occupied になりうる。旧実装
    // の固定列挙（`CANDIDATES[0..=2]` のみ）はこの 2 候補を構造上常に対象外
    // にしており、`shape_cfg` 決定への合流（上のコメント参照）で「occupancy
    // 縮退を迂回しない」としていた意図（`docs/perf/metal-gemm-tile-table.md`
    // §5）と矛盾していた。`bm*bn` 比較にすることで、厳密一致テーブルの
    // 追加候補を含め、大タイル系と同じ threadgroup 分担面積を持つ構成には
    // 常に一様に occupancy 縮退が適用される（`select_with_occupancy_
    // shrinks_exact_match_candidates_when_underoccupied` 参照）。
    //
    // `m == n == k`（実測どおりの立方 GEMM）かつ `m <= SQUARE_MEASURED_MAX`
    // （実測範囲内）は #744 是正後 CANDIDATES[3] を直接返すため縮退対象から
    // 外れるが、`m != n` の準正方大形状長方形、`m == n` でも `k != m`（K
    // 未実測）の正方出力、および `SQUARE_MEASURED_MAX` 超の正方立方形状
    // （上記 `m,n >= LARGE` 分岐）は引き続き CANDIDATES[0] を返しうるため、
    // 縮退判定（actual/ideal 比較）は縦長・横長に加えてこの経路でも生きた
    // ままになる（#744 是正前と同一挙動。PR #760 レビュー対応でコメントを
    // 実装へ整合）。
    let mid_tile_area = (CANDIDATES[3].bm as u64) * (CANDIDATES[3].bn as u64);
    let shape_cfg_area = (shape_cfg.bm as u64) * (shape_cfg.bn as u64);
    let is_large_tile_candidate = shape_cfg_area > mid_tile_area;

    if !is_large_tile_candidate {
        return shape_cfg;
    }

    let Some(params) = params else {
        return shape_cfg;
    };

    let Some(actual) = actual_groups(m, n, shape_cfg) else {
        return shape_cfg;
    };

    let Some(ideal) = params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, shape_cfg) else {
        return shape_cfg;
    };

    if is_underoccupied(actual, ideal) {
        CANDIDATES[3] // 32x32（中形状）へ縮退
    } else {
        shape_cfg
    }
}

/// `(m, n, k)` から [`TileConfig`] を選択する（occupancy 判定込み。後方
/// 互換入口）。
///
/// **公開 API 互換性（P1・codex-review 指摘・PR #1108 レビュー）**: 本関数
/// の 4 引数シグネチャ（`gpu_core_count` を含まない）は PR #1108 以前から
/// 存在する公開 API（`fandhe-ai-backend-metal` は crates.io 公開クレート）
/// であり、破壊的変更を避けるため維持する。デバイス情報を受け取る版は
/// 別名 [`select_with_occupancy_for_device`] として追加し、本関数は
/// `gpu_core_count: None` で委譲する（＝厳密一致テーブルは経由せず既存の
/// 形状クラス判定＋occupancy 縮退のみ。PR #1108 以前と完全一致する挙動）。
pub fn select_with_occupancy(
    m: usize,
    n: usize,
    k: usize,
    params: Option<OccupancyParams>,
) -> TileConfig {
    select_with_occupancy_for_device(m, n, k, None, params)
}

/// threadgroup ID スウィズル（`swizzle_log` 相当。イシュー #540）の群幅を
/// 2 のべき乗指数で表す。`4 = 1 << SWIZZLE_LOG` threadgroup を 1 群として
/// dispatch grid 上で縦方向へ束ね、近接時刻に実行される threadgroup 群が
/// B（列方向）の同一領域を再利用しやすくする（L2 相当キャッシュのヒット
/// 率向上を狙う。MLX steel `swizzle_log` と同型・DeepGEMM の L2 スウィズル
/// と同種の技法。計画「設計方針」節）。
///
/// **実験的パラメータとして固定値のみ扱う**: `TileConfig` のフィールドには
/// しない。採否は `docs/perf/metal-gemm-tgid-swizzle-ab.md` の A/B 計測で
/// 判断し、不採用なら本定数を含む変更一式を revert する（既定 off の
/// 未使用パラメータとして残さない方針。out-of-scope-tracking.md の趣旨に
/// 反しないための意図的な二択設計）。
///
/// `pub(crate)` に留める（PR #661 codex-review 指摘）: `tile` モジュールは
/// `lib.rs` で `pub mod tile;` のため公開されており、`pub` のままだと
/// 実測結果次第で撤去されうる実験的な内部実装詳細が外部利用者の依存対象
/// （公開 API）になってしまう。シェーダ証跡検査（`SWIZZLE_LOG` の値と
/// `shaders/gemm.metal` 側リテラルの一致確認）はこのファイル末尾の
/// crate 内 unit test（`gemm_simdgroup_tiled_source_uses_tgid_swizzle`）が
/// 担う（旧 `tests/shader_source_evidence.rs` から移設。別クレート
/// コンパイル単位からは `pub(crate)` 定数を参照できないため）。
///
/// `#[cfg(any(test, target_os = "macos"))]`: 唯一の非テスト呼び出し元
/// `crate::pipeline::make_pipeline_with_constants`／`crate::gemm::
/// encode_dispatch_tiled` は `cfg(target_os = "macos")` 限定（`lib.rs`）の
/// ため、`target_os != "macos"` かつ非テストビルドではこの定数への到達
/// パスが存在せず `dead_code` lint（`clippy -D warnings`）が誤検知する
/// （`pub(crate)` 化前は `pub` だったため外部公開 API 扱いで lint 対象外
/// だった）。本ファイルは Linux 上でも純粋関数の単体テストを回す設計
/// （本ファイル冒頭コメント）のため `test` cfg でも到達可能にする。
#[cfg(any(test, target_os = "macos"))]
pub(crate) const SWIZZLE_LOG: u32 = 2;

/// スウィズルを本番 dispatch 経路（`crate::gemm::encode_dispatch_tiled`）と
/// シェーダ側 tgid 変換（`shaders/gemm.metal` の `SWIZZLE_ENABLED` function
/// constant）で実際に有効化するかどうかのゲート（PR #661 codex-review
/// 指摘）。
///
/// **既定は `false`（無効）**: 本イシュー（#540）は実機（Apple Silicon）
/// での性能効果・数値一致が `docs/perf/metal-gemm-tgid-swizzle-ab.md` の
/// 「判断基準」を満たすまで未検証のため、本番経路は従来の走査順
/// （`tid_y = tgid.y`・`tid_x = tgid.x`。恒等変換）のままにする。A/B 計測を
/// 実機セッションで行う際はこの値を一時的に `true` へ変更してベンチマークを
/// 実行し、採用判断後に応じて（採用: 既定を `true` へ確定 / 不採用:
/// スウィズル機構一式を revert）このコメントごと更新する。コミットした
/// 状態で `true` のまま残さない。
///
/// `#[cfg(any(test, target_os = "macos"))]` の理由は [`SWIZZLE_LOG`] の
/// doc comment を参照（同一の dead_code 誤検知回避）。
#[cfg(any(test, target_os = "macos"))]
pub(crate) const SWIZZLE_ENABLED: bool = false;

/// simdgroup 細粒度同期（イシュー #809）を本番 dispatch 経路
/// （`crate::gemm::MetalGemm::pipeline_for_tile`。**f32 経路のみ**）と
/// シェーダ側 `FINE_BARRIER_ENABLED` function constant で実際に有効化する
/// かどうかのゲート（[`SWIZZLE_ENABLED`] と同型の設計判断）。
/// `crate::gemm::MetalGemm::pipeline_for_tile_f16`（`gemm_simdgroup_tiled_f16`）
/// にも同じ値が伝播するが、当該カーネルは `FINE_BARRIER_ENABLED` を
/// 参照しないため無害な no-op であり f16 経路の挙動は変化しない
/// （`crate::gemm::MetalGemm` の `fine_barrier_enabled` フィールド
/// ドキュメンテーションコメント参照）。
///
/// **既定は `false`（無効）**: `gemm_simdgroup_tiled` の staged 経路 kk ループへ
/// `simdgroup_barrier(mem_flags::mem_none)` を挿入する構成の性能効果が
/// `docs/perf/metal-gemm-fine-barrier-ab.md` の判断基準を満たすまで、本番経路
/// はバリア非挿入のまま動作する。A/B 計測は実機セッションで
/// `examples/gemm_fine_barrier_ab_bench.rs` を使う（この定数自体は変更せず、
/// `MetalGemm::new_with_fine_barrier` へ明示的に `true` を渡した head
/// インスタンスで計測する。#540 の運用方式〈#746 で `SWIZZLE_ENABLED` を
/// instance フィールドへ格上げした判断〉を踏襲）。採用判断後に応じて
/// （採用: 既定を `true` へ確定 / 不採用: 本機構一式を revert）このコメント
/// ごと更新する。
#[cfg(any(test, target_os = "macos"))]
pub(crate) const FINE_BARRIER_ENABLED: bool = false;

/// `encode_dispatch_tiled`（`crate::gemm`）が呼ぶ、スウィズル後の dispatch
/// grid（`(grid_width, grid_height)` = `(threadgroups.width,
/// threadgroups.height)`）を計算する純粋関数。
///
/// `tiles_n`/`tiles_m` は素朴な `div_ceil(dims.n, cfg.bn)`/
/// `div_ceil(dims.m, cfg.bm)`（スウィズル前の threadgroup 数）。
/// `shaders/gemm.metal` の `gemm_simdgroup_tiled` 側 tgid 変換
/// （`tid_y = (tgid.y << SWIZZLE_LOG) + (tgid.x & (tile - 1))`・
/// `tid_x = tgid.x >> SWIZZLE_LOG`）と 1:1 対応する契約: この関数が返す
/// grid 全体を tgid が走査したとき、変換後の `(tid_y, tid_x)` が
/// `0..tiles_m × 0..tiles_n` の全域を過不足なく 1 回ずつ覆う必要がある
/// （`tiles_m` が `tile` の倍数でない場合に生じる余剰 threadgroup は
/// カーネル側の早期 return（`row0 >= dims.m || col0 >= dims.n`。REQ-8）
/// が無害化する。本ファイルの `swizzle_reference_remap` テストヘルパで
/// この対応を Linux 上でも静的に検証する）。
///
/// `tiles_n`/`tiles_m` は `crate::gemm::validate_effective_dims` 通過後の
/// 実効次元（`u32::MAX` 以下）由来のため、`tiles_n * (1 << SWIZZLE_LOG)`
/// （最大倍率 4）は 64bit `usize` 上でオーバーフローしない
/// （`u32::MAX * 4` は `usize` が 32bit 環境でも `u64` 相当の範囲に収まる
/// が、念のため `saturating_mul` で防御し、万一の桁あふれはパニックではなく
/// `usize::MAX` へ飽和させる。`checked_shl` はシフト量がビット幅以上の
/// ときのみ `None` を返す仕様で値側の桁あふれ自体は検知できない（`<<` と
/// 挙動が同一）ため、桁あふれを実際に検知できる乗算ベースの飽和演算を使う。
/// 実運用では到達しない経路であることをコメントで根拠づけるに留め、
/// 無限ループ等の未定義動作を避ける趣旨）。
///
/// `#[cfg(any(test, target_os = "macos"))]` の理由は [`SWIZZLE_LOG`] の
/// doc comment を参照（同一の dead_code 誤検知回避）。
#[cfg(any(test, target_os = "macos"))]
pub(crate) fn swizzled_grid(tiles_n: usize, tiles_m: usize) -> (usize, usize) {
    let tile = 1usize << SWIZZLE_LOG;
    let grid_w = tiles_n.saturating_mul(tile);
    let grid_h = tiles_m.div_ceil(tile);
    (grid_w, grid_h)
}

/// `crate::gemm::encode_dispatch_tiled` が使う dispatch grid を
/// `swizzle_enabled` に応じて決定する（PR #661 codex-review 指摘対応:
/// 未検証のスウィズルを本番経路へ無条件適用しない）。`swizzle_enabled` が
/// `false` の間は素朴な `(tiles_n, tiles_m)` grid（スウィズル前の
/// threadgroup 数）を返し、`shaders/gemm.metal` 側の恒等変換
/// （`SWIZZLE_ENABLED=false` 分岐）と同期する契約。`true` の場合は
/// [`swizzled_grid`] へ委譲する。
///
/// イシュー #746 で `bool` 引数へ格上げした: 従来はクレート定数
/// [`SWIZZLE_ENABLED`] を直接読んでいたが、`crate::gemm::MetalGemm` を
/// base（off）/head（on）の 2 インスタンスで同一プロセス内に構築し
/// interleaved に A/B 計測する運用（`docs/perf/metal-gemm-tgid-swizzle-ab.md`）
/// のため、呼び出し元（`encode_dispatch_tiled`）がインスタンス保持の値を
/// 渡せるようにする。呼び出し元は `crate::pipeline::
/// make_pipeline_with_constants` へ渡す function constant 値と**同じ**
/// `swizzle_enabled` を渡す責務を負う（シェーダ側 tgid 変換と grid 形状の
/// 同期契約が崩れるため）。
///
/// `#[cfg(any(test, target_os = "macos"))]` の理由は [`SWIZZLE_LOG`] の
/// doc comment を参照（同一の dead_code 誤検知回避）。
#[cfg(any(test, target_os = "macos"))]
pub(crate) fn tiled_dispatch_grid_with(
    tiles_n: usize,
    tiles_m: usize,
    swizzle_enabled: bool,
) -> (usize, usize) {
    if swizzle_enabled {
        swizzled_grid(tiles_n, tiles_m)
    } else {
        (tiles_n, tiles_m)
    }
}

/// [`tiled_dispatch_grid_with`] を本番既定値（[`SWIZZLE_ENABLED`]）で
/// 呼ぶ薄いラッパー。既存の crate 内 unit test
/// （`tiled_dispatch_grid_matches_swizzle_enabled_gate` 等。本ファイル末尾）
/// が「コミット状態の既定値が意図通りか」を検証する対象として維持する
/// （イシュー #746 の引数化で `MetalGemm` 本番経路〈`MetalGemm::new`〉が
/// 直接この関数を呼ぶことはなくなったが、既定値の非後退ロックとして残す）。
///
/// `#[cfg(test)]` 限定（`#[cfg(any(test, target_os = "macos"))]` ではない）:
/// 本番ディスパッチ経路（`crate::gemm::encode_dispatch_tiled`）は
/// `MetalGemm::swizzle_enabled` を明示的に渡す `tiled_dispatch_grid_with` を
/// 直接呼ぶため、macOS 非テストビルドではこの関数への到達パスが存在せず
/// `dead_code` lint（`clippy -D warnings`）が誤検知する。テスト専用のため
/// テストビルドのみで到達可能にする。
#[cfg(test)]
pub(crate) fn tiled_dispatch_grid(tiles_n: usize, tiles_m: usize) -> (usize, usize) {
    tiled_dispatch_grid_with(tiles_n, tiles_m, SWIZZLE_ENABLED)
}

// --- occupancy 目標算出（イシュー #541・D-7a）---
//
// MFA（metal-flash-attention）型の方法論「`actualGroups`（起動 threadgroup
// 数）と `idealGroups`（コア数×係数）を比較し、閾値でタイルを 2 段階切替
// する」の算出機構。[`select_with_occupancy`]（イシュー #542）が本節の
// 関数群を用いて閾値判定・タイル縮退を行う（#487 診断バイナリ
// `examples/gemm_diagnosis.rs::analytics` のローカル実装と等価な算出式を
// クレート内 API として一本化し、算式のドリフト〈重複実装間の食い違い〉
// を防ぐ狙い）。
//
// 本モジュール全体と同じ設計判断で、以下も `objc2` 系 FFI に一切触れない
// 純粋関数・純粋構造体のため `cfg(target_os = "macos")` を付けない
// （Linux・CI でも単体テストが回る）。GPU コア数の実機取得（IOKit FFI）は
// `crate::device::probe_gpu_core_count`（macOS 限定）が担い、その結果を
// [`OccupancyParams`] へ写像してから本モジュールの関数へ渡す構成にする
// （FFI 層と算出ロジック層を分離し、算出ロジックを非 macOS でもテスト
// 可能に保つため）。

/// `(m, n)` から実際に起動される threadgroup 数（`ceil(m/bm) * ceil(n/bn)`）。
///
/// `examples/gemm_diagnosis.rs::analytics::analyze` の `actual_groups` 算出
/// （`groups_m * groups_n`）と同一式。現行 GEMM にバッチ次元は存在しない
/// ため常に 1 件分の行列積を前提とする（MFA の `batchDimension` 相当は
/// 将来拡張。イシュー #541 計画「スコープ外」節）。
///
/// **fail-safe 契約**: `TileConfig` の `bm`／`bn` フィールドは公開のため
/// 呼び出し元が `0` を含む任意値を構築できる（`div_ceil(0)` は panic する）。
/// 本番経路で panic しない契約（`.claude/rules/coding-rust.md`）に従い、
/// `cfg.bm == 0` または `cfg.bn == 0`、あるいは `groups_m * groups_n` の
/// オーバーフロー（`usize` から `u64` へ拡張した値同士の積が `u64::MAX` を
/// 超える極端な `m`／`n`）はいずれも `None` を返す（codex-review 指摘。
/// PR #662）。通常経路（[`TileConfig::validate`] を通過した構成）では
/// `bm`／`bn` は常に非ゼロのため実質的には常に `Some` を返す。
pub fn actual_groups(m: usize, n: usize, cfg: TileConfig) -> Option<u64> {
    if cfg.bm == 0 || cfg.bn == 0 {
        return None;
    }
    let groups_m = (m as u64).div_ceil(cfg.bm as u64);
    let groups_n = (n as u64).div_ceil(cfg.bn as u64);
    groups_m.checked_mul(groups_n)
}

/// [`OccupancyParams::smem_groups_per_core`] が返す「1 コアあたり同時常駐
/// 可能な threadgroup 数」の上限キャップ。TileKernels（参照実装）由来の
/// 経験的上限であり、Apple Silicon の実際のレジスタ・実行ユニット制約に
/// よる同時実行数はこれより小さくなりうる（本キャップは threadgroup
/// memory 予算のみから導出される上限を頭打ちにするためのものであり、
/// レジスタ圧迫等の別要因は考慮しない）。M4 Max 向け確定値ではなく実機
/// 実測（#542）で見直しうる初期値（イシュー #541 計画「設計方針」節）。
pub const SMEM_GROUPS_PER_CORE_CAP: u32 = 16;

/// [`OccupancyParams::ideal_groups`] の既定係数（f32 アキュムレータ系）。
/// MFA の FP32 系経験式（`docs/perf/metal-gemm-bottleneck-diagnosis.md`
/// §3 の M4 Max 実測前提値・`examples/gemm_diagnosis.rs::analytics::
/// DeviceProfile::M4_MAX` と同じ出典）が採用する「コアあたり 6 threadgroup」
/// を初期値として踏襲する。**M4 Max 向けの確定値ではない**（本ファイル
/// 冒頭コメント・イシュー #541 計画「注意」節: MFA の具体数値は Apple7/8/9
/// の実測値であり実機実測で確定させる前提）。
pub const IDEAL_GROUPS_MULTIPLIER_F32: u32 = 6;

/// [`OccupancyParams::ideal_groups`] の全 16bit 系（A・B・アキュムレータ
/// すべて半精度）向け係数。MFA の経験式でレジスタ圧迫が緩む分だけ f32 系
/// より高い値（9）が使われる。[`IDEAL_GROUPS_MULTIPLIER_F32`] と同じく
/// 初期値であり実機実測で確定させる（`backend-metal` は現状 f32 GEMM
/// が主経路のため、本定数は将来 f16 系カーネルを追加した際の参照用に
/// 先行定義する）。
pub const IDEAL_GROUPS_MULTIPLIER_ALL_16BIT: u32 = 9;

/// occupancy 目標算出（[`OccupancyParams::ideal_groups`]）の入力パラメータ。
/// `crate::device::MetalOccupancyInfo`（macOS 実機からの実測値）から写像
/// して構築する。GPU コア数が取得不能（`None`）な場合の呼び出し側
/// フォールバック方針は [`select_with_occupancy`] が `params: None` の
/// fail-safe 分岐として確定済み（イシュー #542）。本構造体自体は
/// `gpu_core_count: u32` を必須値として持つ（呼び出し側が `Option` の
/// 解決を担う。`crate::context::MetalContext::new` が `MetalOccupancyInfo`
/// から `Option<OccupancyParams>` への写像を担う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyParams {
    /// GPU コア数（`crate::device::probe_gpu_core_count` の実測値、または
    /// CLI／設定ファイル由来の明示指定値）。
    pub gpu_core_count: u32,
    /// デバイスの threadgroup memory 上限バイト数
    /// （`MTLDevice::maxThreadgroupMemoryLength`）。
    pub max_threadgroup_memory_bytes: u32,
}

impl OccupancyParams {
    /// `cfg` の threadgroup memory 使用量から、1 コアに同時常駐できる
    /// threadgroup 数の上限を求める（TileKernels 型: `min(max_smem /
    /// smem_bytes, CAP)`）。
    ///
    /// - `cfg.staged == false`（`shared_mem_bytes() == 0`。device メモリから
    ///   直接 `simdgroup_load` する経路）は threadgroup memory を消費しない
    ///   ため、SMEM 制約による上限は存在せず [`SMEM_GROUPS_PER_CORE_CAP`]
    ///   をそのまま返す。
    /// - `cfg.shared_mem_bytes() == 0` かつ `staged == true` は
    ///   [`TileConfig::validate`] の整除制約上ありえない構成だが、ゼロ除算
    ///   を避けるため同じく `staged` 分岐で判定する（`shared_mem_bytes` の
    ///   値そのものでは分岐しない）。
    /// - `shared_mem_bytes() > max_threadgroup_memory_bytes`（デバイス上限
    ///   超過。[`TileConfig::validate`] が本来この構成自体を拒否する）は
    ///   0 を返す（同時常駐不可）。
    pub fn smem_groups_per_core(&self, cfg: TileConfig) -> u32 {
        if !cfg.staged {
            return SMEM_GROUPS_PER_CORE_CAP;
        }
        let bytes = cfg.shared_mem_bytes();
        if bytes == 0 || bytes > self.max_threadgroup_memory_bytes {
            return 0;
        }
        (self.max_threadgroup_memory_bytes / bytes).min(SMEM_GROUPS_PER_CORE_CAP)
    }

    /// コア飽和目標 threadgroup 数（`idealGroups`）を算出する。
    ///
    /// `gpu_core_count * min(multiplier, smem_groups_per_core(cfg))`
    /// （MFA 経験式にコア数×係数だけでなく threadgroup memory 予算による
    /// 同時常駐数上限を組み合わせた一般化。イシュー #541 計画「設計方針」
    /// §3.2）。`multiplier` には [`IDEAL_GROUPS_MULTIPLIER_F32`] 等の既定
    /// 係数、または `examples/gemm_diagnosis.rs` の `--ideal-groups-multiplier`
    /// 相当の明示指定値を渡す。
    ///
    /// `gpu_core_count == 0`・`multiplier == 0`・積のオーバーフローは
    /// `None`（fail-closed。`examples/gemm_diagnosis.rs::
    /// parse_device_profile_override` の CLI 検証と同方針。「occupancy
    /// 判定を無効化するフォールバック」として呼び出し側が扱う契約は
    /// [`select_with_occupancy`] が確定済み（イシュー #542）。
    ///
    /// `smem_groups_per_core(cfg) == 0`（`cfg` の threadgroup memory
    /// 使用量がデバイス上限を超える等、同時常駐不能な構成）の場合も
    /// `effective_multiplier` が 0 になり同様に `None` を返す（codex-review
    /// 指摘・PR #662）。ここを素通りして `Some(0)` を返すと、
    /// [`is_underoccupied`] へ渡した際に通常 `actual > 0` のため
    /// `actual <= 0` が false 判定となり、実行不能なタイルを
    /// 「under-occupied ではない」と誤判定してしまう（fail-closed 契約に
    /// 反する）。
    pub fn ideal_groups(&self, multiplier: u32, cfg: TileConfig) -> Option<u64> {
        if self.gpu_core_count == 0 || multiplier == 0 {
            return None;
        }
        let effective_multiplier = multiplier.min(self.smem_groups_per_core(cfg));
        if effective_multiplier == 0 {
            return None;
        }
        (self.gpu_core_count as u64).checked_mul(effective_multiplier as u64)
    }
}

/// `actual`（[`actual_groups`]）が `ideal`（[`OccupancyParams::ideal_groups`]）
/// 以下かどうかを判定する（MFA の小ブロック縮退条件相当: under-occupied
/// なら小さいタイルへ切り替える判断材料になる）。境界一致（`actual ==
/// ideal`）は under-occupied 側（`true`）に倒す（fail-safe: 「ちょうど
/// 目標」を「十分」と誤認せず縮退候補に含める）。閾値運用（[`select`] への
/// 組み込み）は [`select_with_occupancy`] が担う（イシュー #542）。
pub fn is_underoccupied(actual: u64, ideal: u64) -> bool {
    actual <= ideal
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用: [`verify_m4_max`] を経由して [`VerifiedM4MaxGpuCoreCount`]
    /// を構築するヘルパー（P1 是正・PR #1108 レビューで opaque 型化した
    /// ことに伴い、テストからも生の `u32` を直接渡せなくなったため。GPU
    /// コア数・SoC ブランド文字列の両方が実測一致した体で検証する）。
    fn verified_m4_max_for_test() -> Option<VerifiedM4MaxGpuCoreCount> {
        verify_m4_max(Some(M4_MAX_GPU_CORE_COUNT), Some(M4_MAX_SOC_BRAND))
    }

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

    // --- TileConfig::shared_mem_bytes_f16（イシュー #796・エピローグの
    // タイル粒度統合はイシュー #797。pure） ---

    #[test]
    fn shared_mem_bytes_f16_returns_epilogue_only_when_not_staged() {
        // `staged=false` はタイル領域を確保しないが、エピローグ staging
        // 領域（f32。#797 でタイル粒度〈bm*bn*4 バイト〉へ拡大。本メソッド
        // doc コメント「エピローグ領域のサイズ」節参照）は常に必要。
        // `SINGLE_SIMDGROUP_8X8`（bm=8,bn=8,staged=false）は
        // 8*8*4=256 バイト（16 の倍数。#796 時点の 1*1*64*4=256 と偶然一致
        // する。8x8 タイル 1 個のみを担当する最小構成のため）。
        assert_eq!(TileConfig::SINGLE_SIMDGROUP_8X8.shared_mem_bytes_f16(), 256);
    }

    #[test]
    fn shared_mem_bytes_f16_includes_half_tile_region_plus_epilogue_when_staged() {
        // `shared_mem_bytes_includes_derived_pad_in_both_tile_strides_when_staged`
        // （f32 版）と同一の cfg（bm=bn=64, bk=16, wm=wn=2, staged=true）。
        // タイル領域: A 64x(16+4)=64x20=1280 要素、B 16x(64+4)=16x68=1088
        // 要素、合計 2368 要素 * 2 バイト（half）= 4736 バイト。
        // エピローグ（#797 でタイル粒度へ拡大）: bm*bn*4 = 64*64*4 = 16384
        // バイト。合計 4736+16384 = 21120 バイト。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes_f16(), 21120);
    }

    #[test]
    fn shared_mem_bytes_f16_all_candidates_within_32kib_device_limit() {
        // #797 でエピローグ領域が `wm*wn*64*4` から `bm*bn*4` へ拡大した
        // ことで、`CANDIDATES` のいずれかが Apple Silicon の一般的な
        // threadgroup メモリ上限（32KiB=32768 バイト）を超過していないかを
        // 静的に固定する（超過構成は `pipeline_for_tile_f16` の
        // `shared_mem_bytes_f16() > max_shared_mem_bytes` 検査で拒否・
        // フォールバックされるが、その前提が崩れていないことをここで
        // 明示的に保証する。計画「設計方針」節の最大値実測 21120 バイト参照）。
        for cfg in CANDIDATES {
            let bytes = cfg.shared_mem_bytes_f16();
            assert!(
                bytes <= 32 * 1024,
                "cfg {cfg:?} の shared_mem_bytes_f16={bytes} が 32KiB を超過"
            );
        }
    }

    #[test]
    fn shared_mem_bytes_f16_is_multiple_of_16_for_all_candidates() {
        // `setThreadgroupMemoryLength_atIndex`（`crate::gemm::
        // encode_dispatch_tiled_f16`）は 16 バイト境界整合を要求する
        // （f32 版 `encode_dispatch_tiled` と同じ制約。本テストは
        // `CANDIDATES`・`SINGLE_SIMDGROUP_8X8` の全構成が f16 版でも
        // この制約を満たすことを確認する）。
        for cfg in CANDIDATES {
            let bytes = cfg.shared_mem_bytes_f16();
            assert!(
                bytes.is_multiple_of(16),
                "cfg {cfg:?} の shared_mem_bytes_f16={bytes} が 16 の倍数でない"
            );
        }
    }

    #[test]
    fn shared_mem_bytes_f16_fits_standard_shared_mem_limit_for_all_candidates() {
        // イシュー #798: `dispatch_f16_auto_unverified` は `tile::select` が
        // `CANDIDATES` から選んだ構成をそのまま `pipeline_for_tile_f16`
        // （f16 版デバイス上限検査。`shared_mem_bytes_f16` ドキュメント
        // コメント参照）へ渡す。標準 Apple Silicon 上限（32KiB。本ファイル
        // 内の `validate` 系テストが揃って使う `32 * 1024` と同じ値）を
        // どの候補も超えていなければ、`select` の出力がデバイス実測なしに
        // サイレント縮退（`SINGLE_SIMDGROUP_8X8` へのフォールバック）を
        // 起こさないことを CI（Linux・GPU 非依存）上で担保できる。
        const STANDARD_SHARED_MEM_LIMIT: u32 = 32 * 1024;
        for cfg in CANDIDATES {
            let bytes = cfg.shared_mem_bytes_f16();
            assert!(
                bytes <= STANDARD_SHARED_MEM_LIMIT,
                "cfg {cfg:?} の shared_mem_bytes_f16={bytes} が標準上限 \
                 {STANDARD_SHARED_MEM_LIMIT} を超過しており、標準 Apple \
                 Silicon 上ではサイレント縮退が起きる"
            );
        }
    }

    #[test]
    fn dispatch_f16_auto_shapes_resolve_to_expected_candidate() {
        // `tests/gemm_f16_auto_parity.rs`（`dispatch_f16_auto_unverified`。イシュー
        // #798）の各ケースが「どの `CANDIDATES` 分岐を検証しているか」の
        // 主張は、その統合テスト自体が Metal 実機依存・`#[ignore]` のため
        // 実機なしでは検証できない。`select`（`tile::select_with_occupancy`
        // への委譲）は `objc2` 系 FFI に触れない純粋関数のため、同じ形状を
        // ここで固定し Linux（CI）上でも分岐主張のドリフトを検知する。
        assert_eq!(
            select_for_device(2048, 256, 512, verified_m4_max_for_test()),
            CANDIDATES[1],
            "縦長 → CANDIDATES[1]"
        );
        assert_eq!(
            select_for_device(256, 2048, 512, verified_m4_max_for_test()),
            CANDIDATES[2],
            "横長 → CANDIDATES[2]"
        );
        assert_eq!(
            select_for_device(512, 512, 512, verified_m4_max_for_test()),
            CANDIDATES[5],
            "正方立方（イシュー #1039 実測点） → CANDIDATES[5]"
        );
        assert_eq!(
            select_for_device(1536, 1024, 512, verified_m4_max_for_test()),
            CANDIDATES[0],
            "準正方大形状長方形 → CANDIDATES[0]"
        );
        assert_eq!(
            select_for_device(32, 32, 32, verified_m4_max_for_test()),
            TileConfig::SINGLE_SIMDGROUP_8X8,
            "微小形状 → SINGLE_SIMDGROUP_8X8"
        );
        // 端数形状ケース（`gemm_f16_auto_parity.rs::
        // dispatch_f16_auto_matches_cpu_reference_non_multiple_of_8_boundary_shape`）
        // が staged 経路（CANDIDATES[3]）を踏むことを固定する。
        assert_eq!(
            select_for_device(521, 265, 131, verified_m4_max_for_test()),
            CANDIDATES[3],
            "8 非整列の端数形状（m,n,k は SMALL=64 以上） → CANDIDATES[3]"
        );
    }

    #[test]
    fn shared_mem_bytes_f16_saturates_instead_of_wrapping_on_overflow() {
        // `shared_mem_bytes_saturates_instead_of_wrapping_on_overflow`
        // （f32 版）と同じ regression 意図: `bk` に極端に大きい値を渡しても
        // `u32::MAX` へ飽和し、`validate` 相当の上限比較で確実に拒否される
        // ことを確認する（wrap による小さい値への回帰を防ぐ）。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: u32::MAX - 7, // 8 の倍数へ切り下げ
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes_f16(), u32::MAX);
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

    // --- TileConfig::validate: ベクトル化ロード整除制約（イシュー #535） ---

    #[test]
    fn validate_rejects_staged_bk_not_divisible_by_vec_width() {
        // staged=true・bk=6（4 の倍数でない）は既存の 8 整除検査
        // （BkNotMultipleOfEight）で弾かれる。専用 variant は設けない
        // 判断（型定義 doc・validate 実装コメント参照）の固定。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 6,
            wm: 2,
            wn: 2,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BkNotMultipleOfEight { bk: 6 }
        ));
    }

    #[test]
    fn validate_rejects_staged_bn_not_divisible_by_vec_width() {
        // staged=true・bn=6（4 の倍数でない）は既存の 8 整除検査
        // （BnNotDivisibleByWn8。wn=1 のため bn%8==0 相当）で弾かれる。
        // bm/bk/wm/wn は他の検査に触れない妥当値にしておく。
        let cfg = TileConfig {
            bm: 64,
            bn: 6,
            bk: 16,
            wm: 2,
            wn: 1,
            staged: true,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BnNotDivisibleByWn8 { bn: 6, wn: 1 }
        ));
    }

    #[test]
    fn non_staged_config_uses_same_divisibility_check_as_staged() {
        // staged=false（直接ロード経路）も staged=true と同じ 8 整除検査
        // （BkNotMultipleOfEight）のみで弾かれる。VEC_WIDTH 専用検査を
        // 追加しない設計のため staged の有無で拒否理由は変わらないことの
        // 回帰ガード。
        let cfg = TileConfig {
            bm: 64,
            bn: 64,
            bk: 6,
            wm: 2,
            wn: 2,
            staged: false,
        };
        let err = cfg.validate(1024, 32 * 1024).unwrap_err();
        assert!(matches!(
            err,
            TileConfigError::BkNotMultipleOfEight { bk: 6 }
        ));
    }

    #[test]
    fn validate_ok_implies_vec_width_divisibility() {
        // 不変条件テスト（イシュー #535・codex-review 指摘対応 PR #672）:
        // `TileConfigError` に専用の VEC_WIDTH 整除検査 variant を追加する
        // 代わりに、既存の 8 整除検査（`bk % 8 == 0`・
        // `bn % (wn*8) == 0` かつ `wn >= 1` ⟹ `bn % 8 == 0`）が
        // `VEC_WIDTH`（4）整除を数学的に包含することをここで固定する。
        // `validate` が Ok を返す `staged=true` 構成は必ず `bk`/`bn` が
        // 4 の倍数であることを bm/bn/bk/wm/wn の小さな全域で検査する。
        //
        // 注意（イシュー #535 review 再指摘）: この入力集合で `validate`
        // を通る `bk` は元々すべて 8 の倍数（＝ 4 の倍数でもある）ため、
        // 本テスト単体は「8 整除検査が 4 整除を包含する」という設計判断を
        // 独立に保証しない（将来 `bk % 8` の検査が誤って `bk % 4` へ
        // 弱められても、このテストは通り続けてしまう）。その退行を実際に
        // 検知するのは直後の
        // `validate_rejects_bk_that_is_vec_width_multiple_but_not_eight`
        // であり、本テストは「現状の実装で Ok になる構成が包含関係と
        // 矛盾しないこと」を固定する回帰ガードに留まる。
        for bm in [8u32, 16, 32, 64] {
            for bn in [8u32, 16, 32, 64] {
                for bk in [4u32, 6, 8, 12, 16, 24, 32] {
                    for wm in [1u32, 2, 4] {
                        for wn in [1u32, 2, 4] {
                            let cfg = TileConfig {
                                bm,
                                bn,
                                bk,
                                wm,
                                wn,
                                staged: true,
                            };
                            if cfg.validate(1024, 32 * 1024).is_ok() {
                                assert!(
                                    bk.is_multiple_of(TileConfig::VEC_WIDTH),
                                    "validate({cfg:?}) が Ok を返したが bk が VEC_WIDTH の倍数でない"
                                );
                                assert!(
                                    bn.is_multiple_of(TileConfig::VEC_WIDTH),
                                    "validate({cfg:?}) が Ok を返したが bn が VEC_WIDTH の倍数でない"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn validate_ok_implies_vec_width_f16_divisibility() {
        // `validate_ok_implies_vec_width_divisibility`（f32・VEC_WIDTH=4）
        // の half 版（イシュー #797）: `VEC_WIDTH_F16`（8）は既存の 8 整除
        // 検査（`bk % 8 == 0`・`bn % (wn*8) == 0` ⟹ `bn % 8 == 0`）と
        // ちょうど同じ整除数のため、`validate` が Ok を返す `staged=true`
        // 構成の `bk`/`bn` は必ず `VEC_WIDTH_F16` の倍数であることを固定する
        // （`gemm_simdgroup_tiled_f16` の 8 要素グループ協調ロードが
        // グループ境界で行/列をまたがない前提の裏付け）。
        for bm in [8u32, 16, 32, 64] {
            for bn in [8u32, 16, 32, 64] {
                for bk in [8u32, 16, 24, 32] {
                    for wm in [1u32, 2, 4] {
                        for wn in [1u32, 2, 4] {
                            let cfg = TileConfig {
                                bm,
                                bn,
                                bk,
                                wm,
                                wn,
                                staged: true,
                            };
                            if cfg.validate(1024, 32 * 1024).is_ok() {
                                assert!(
                                    bk.is_multiple_of(TileConfig::VEC_WIDTH_F16),
                                    "validate({cfg:?}) が Ok を返したが bk が VEC_WIDTH_F16 の倍数でない"
                                );
                                assert!(
                                    bn.is_multiple_of(TileConfig::VEC_WIDTH_F16),
                                    "validate({cfg:?}) が Ok を返したが bn が VEC_WIDTH_F16 の倍数でない"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn validate_rejects_bk_that_is_vec_width_multiple_but_not_eight() {
        // `validate_ok_implies_vec_width_divisibility` がトートロジーに
        // 陥る間隙を埋める直接的な回帰ガード（イシュー #535 review 再指摘）。
        // `bk` が `VEC_WIDTH`（4）の倍数だが 8 の倍数ではない値（4・12・20・
        // 28）を固定で用意し、`staged=true` かつ他フィールドは通常受理
        // される値（bm=64,bn=64,wm=2,wn=2）に揃えたうえで、`validate` が
        // 必ず `BkNotMultipleOfEight` で拒否することを検査する。将来
        // 「既存 8 整除検査が VEC_WIDTH 整除を包含するので専用 variant は
        // 不要」という設計判断（本ファイル [`TileConfig::validate`]
        // ドキュメント参照）を壊して `bk % 8` を `bk % VEC_WIDTH` へ
        // 弱める変更が入ると、このテストが直ちに失敗する。
        for bk in [4u32, 12, 20, 28] {
            let cfg = TileConfig {
                bm: 64,
                bn: 64,
                bk,
                wm: 2,
                wn: 2,
                staged: true,
            };
            assert!(
                bk.is_multiple_of(TileConfig::VEC_WIDTH),
                "テスト前提が崩れている: bk={bk} が VEC_WIDTH の倍数でない"
            );
            let err = cfg.validate(1024, 32 * 1024).unwrap_err();
            assert!(
                matches!(err, TileConfigError::BkNotMultipleOfEight { bk: rejected_bk } if rejected_bk == bk),
                "bk={bk}（VEC_WIDTH 倍数だが 8 の倍数でない）が validate({cfg:?}) で \
                 BkNotMultipleOfEight 以外の結果になった: {err:?}"
            );
        }
    }

    #[test]
    fn all_candidates_satisfy_vec_width_constraints() {
        // CANDIDATES 全 8 構成（イシュー #532 の D-1 で追加された MLX
        // classic 経路 3 構成・末尾の SINGLE_SIMDGROUP_8X8〈staged=false
        // のため VEC_WIDTH 検査の対象外〉を含む）が VEC_WIDTH 整除検査を
        // 含む validate() を通ることの明示確認（受け入れ基準 2）。
        // `validate_accepts_all_candidates_under_typical_device_limits`
        // と同じ入力集合だが、本テストは本イシューで追加した検査が
        // 新規に候補を拒否しないことを名前で明示するための固定。
        for cfg in CANDIDATES {
            cfg.validate(1024, 32 * 1024).unwrap_or_else(|e| {
                panic!("candidate {cfg:?} rejected by vec-width-aware validate: {e}")
            });
        }
    }

    #[test]
    fn direct_load_path_config_still_validates() {
        // `tests/gemm_dynamic_tile_parity.rs` の直接ロード経路構成
        // （bm=32,bn=32,bk=16,wm=2,wn=2,staged=false）が本イシューの
        // VEC_WIDTH 整除契約の明文化（既存 8 整除検査の間接包含・
        // コメント・不変条件テスト追加）後も引き続き validate() を
        // 通ることの確認（計画「実装ステップ」節）。
        let cfg = TileConfig {
            bm: 32,
            bn: 32,
            bk: 16,
            wm: 2,
            wn: 2,
            staged: false,
        };
        cfg.validate(1024, 32 * 1024)
            .unwrap_or_else(|e| panic!("direct-load 構成 {cfg:?} が validate() で拒否された: {e}"));
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
        assert_eq!(
            select_for_device(32, 32, 32, verified_m4_max_for_test()),
            TileConfig::SINGLE_SIMDGROUP_8X8
        );
        assert_eq!(
            select_for_device(1000, 1000, 32, verified_m4_max_for_test()),
            TileConfig::SINGLE_SIMDGROUP_8X8
        );
    }

    #[test]
    fn select_picks_measured_best_config_for_true_square_shapes() {
        // イシュー #1039・2026-08-31/09-01 M4 Max 実機実測（`CANDIDATES` 全
        // 8 候補・2 回計測いずれも同順位）: #744（2026-08-19）は `m == n
        // == k` の 512/1024/2048/4096 の 4 点で CANDIDATES[3]（32x32）一律
        // 選択を確定させたが、#1038〜#1040 の staged 経路変更を経て同じ
        // 4 点を再実測した結果 CANDIDATES[3] はもはや最良候補ではなかった
        // （size=4096: CANDIDATES[3] 8.24〜8.36 TFLOPS に対し最良候補
        // CANDIDATES[2] 9.76〜9.91 TFLOPS。詳細値は
        // `docs/perf/metal-gemm-tile-table.md`）。旧テスト名
        // `select_picks_mid_square_config_for_large_true_square_shapes` は
        // #744 是正時点の「一律 CANDIDATES[3]」前提を反映していたため
        // リネームし期待値を更新する。
        assert_eq!(
            select_for_device(512, 512, 512, verified_m4_max_for_test()),
            CANDIDATES[5]
        );
        assert_eq!(
            select_for_device(1024, 1024, 1024, verified_m4_max_for_test()),
            CANDIDATES[6]
        );
        assert_eq!(
            select_for_device(2048, 2048, 2048, verified_m4_max_for_test()),
            CANDIDATES[1]
        );
        assert_eq!(
            select_for_device(4096, 4096, 4096, verified_m4_max_for_test()),
            CANDIDATES[2]
        );
    }

    #[test]
    fn select_exact_match_table_is_gated_by_m4_max_gpu_core_count() {
        // P1・codex-review 指摘・PR #1108 レビュー対応の回帰テスト:
        // イシュー #1039 の厳密一致テーブル（`exact_match_cfg`）は M4 Max
        // （40 コア構成）実機実測のみが根拠であり、`gpu_core_count` が
        // `Some(M4_MAX_GPU_CORE_COUNT)` と一致する場合にのみ適用される
        // （`select_with_occupancy_for_device` 本体の「機種ゲート」節）。
        // 一致しない
        // コア数・取得不能（`None`）の場合は、実測点の `(m, n, k)` を
        // 与えても本テーブルを経由せず、既存の形状クラス判定（この 4 点は
        // いずれも `m == n == k` かつ `SQUARE_MEASURED_MAX` 以内の真の
        // 正方立方形状のため #744 是正後の CANDIDATES[3]）へ落ちることを
        // 固定する。
        for raw_gpu_core_count in [None, Some(24u32), Some(16), Some(60)] {
            assert_ne!(
                raw_gpu_core_count,
                Some(M4_MAX_GPU_CORE_COUNT),
                "本テストは M4 Max 以外のコア数のみを対象とする"
            );
            // opaque 型化（P1 是正・PR #1108 レビュー）により、未検証の
            // 生コア数を `select_for_device` へ直接渡すことはできない。
            // `verify_m4_max` を経由させると、コア数不一致（SoC ブランドの
            // 値に関わらず）により必ず `None` になることを利用し、元の
            // 「ゲートされない」意図を保ったまま新シグネチャへ対応する。
            let gpu_core_count = verify_m4_max(raw_gpu_core_count, Some(M4_MAX_SOC_BRAND));
            assert_eq!(
                gpu_core_count, None,
                "コア数が不一致のため verify_m4_max は必ず None を返す想定"
            );
            assert_eq!(
                select_for_device(512, 512, 512, gpu_core_count),
                CANDIDATES[3],
                "gpu_core_count={raw_gpu_core_count:?} では厳密一致テーブル \
                 （CANDIDATES[5]）を適用せず形状クラス判定（CANDIDATES[3]）\
                 へフォールバックする"
            );
            assert_eq!(
                select_for_device(1024, 1024, 1024, gpu_core_count),
                CANDIDATES[3],
                "gpu_core_count={raw_gpu_core_count:?} では厳密一致テーブル \
                 （CANDIDATES[6]）を適用しない"
            );
            assert_eq!(
                select_for_device(2048, 2048, 2048, gpu_core_count),
                CANDIDATES[3],
                "gpu_core_count={raw_gpu_core_count:?} では厳密一致テーブル \
                 （CANDIDATES[1]）を適用しない"
            );
            assert_eq!(
                select_for_device(4096, 4096, 4096, gpu_core_count),
                CANDIDATES[3],
                "gpu_core_count={raw_gpu_core_count:?} では厳密一致テーブル \
                 （CANDIDATES[2]）を適用しない"
            );
            // 準正方長方形の実測点（`(1536, 1024, 1024)`）も同様にゲートの
            // 対象であることを固定する。ゲートされない場合、`m != n` で
            // tall／wide いずれにも該当せず `m,n >= LARGE(512)` のため
            // `select_with_occupancy_keeps_near_square_large_rectangle_
            // without_shrink_when_occupied` と同じ経路（大形状フォール
            // バック分岐）を通り CANDIDATES[0] になる（#744 是正前の挙動）。
            assert_eq!(
                select_for_device(1536, 1024, 1024, gpu_core_count),
                CANDIDATES[0],
                "gpu_core_count={raw_gpu_core_count:?} では準正方長方形の \
                 厳密一致点（CANDIDATES[1]）も適用しない"
            );
        }
    }

    #[test]
    fn verify_m4_max_requires_both_gpu_core_count_and_soc_brand_to_match() {
        // P1・codex-review 指摘・PR #1108 レビュー対応の回帰テスト:
        // GPU コア数だけでは機種を一意に識別できない（`gpu_core_count ==
        // 40` は M4 Max だけでなく M3 Max の 40 コア構成にも該当しうる）。
        // `verify_m4_max` は GPU コア数と SoC ブランド文字列の両方が一致
        // した場合にのみ `Some(M4_MAX_GPU_CORE_COUNT)` を返すことを固定
        // する。

        // 両方一致: M4 Max 実機と判定する。
        assert_eq!(
            verify_m4_max(Some(M4_MAX_GPU_CORE_COUNT), Some(M4_MAX_SOC_BRAND)),
            Some(VerifiedM4MaxGpuCoreCount(M4_MAX_GPU_CORE_COUNT))
        );

        // GPU コア数は一致するが SoC ブランドが異なる（例: M3 Max の 40
        // コア構成）。機種を誤認しないよう `None` を返す。
        assert_eq!(
            verify_m4_max(Some(M4_MAX_GPU_CORE_COUNT), Some("Apple M3 Max")),
            None,
            "GPU コア数が一致しても SoC ブランドが異なる場合は M4 Max と \
             判定しない（M3 Max 等の同コア数構成との混同防止）"
        );

        // SoC ブランドは一致するが GPU コア数が異なる（binned 構成等）。
        assert_eq!(
            verify_m4_max(Some(32), Some(M4_MAX_SOC_BRAND)),
            None,
            "SoC ブランドが一致しても GPU コア数が異なる場合は M4 Max \
             実測テーブルの対象外とする（binned 構成の混同防止）"
        );

        // いずれかが取得不能（`None`）な場合も安全側で `None` を返す。
        assert_eq!(verify_m4_max(None, Some(M4_MAX_SOC_BRAND)), None);
        assert_eq!(verify_m4_max(Some(M4_MAX_GPU_CORE_COUNT), None), None);
        assert_eq!(verify_m4_max(None, None), None);
    }

    #[test]
    fn select_keeps_pre_744_behavior_for_true_square_beyond_measured_range() {
        // P1・codex-review 指摘対応（PR #760）: `m == n` の実測は
        // 512/1024/2048/4096 の 4 点のみ（#744）。4096 超の正方形状（実測
        // 対象外）へ CANDIDATES[3] 一律選択を無制限に広げないことを固定する
        // （#744 是正前の挙動 `m,n >= LARGE(512)` なら CANDIDATES[0] を維持）。
        let cfg = select_for_device(8192, 8192, 8192, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[0]);
        assert_eq!((cfg.bm, cfg.bn), (64, 64));
    }

    #[test]
    fn select_keeps_pre_744_behavior_for_square_output_with_unmeasured_k() {
        // P1・codex-review 指摘対応（PR #760）: #744 の実測は `m == n == k`
        // の立方 GEMM（512/1024/2048/4096）のみ。`m == n` だが `k` が
        // 異なる形状（例: (2048,2048,1536)）は依然実測範囲外のため、立方
        // GEMM の実測結果を任意の `k` を持つ正方出力へ拡張しない
        // （#744 是正前の挙動 `m,n >= LARGE(512)` なら CANDIDATES[0] を維持）。
        // `(2048,2048,64)`／`(2048,2048,512)` はイシュー #1039 で実測
        // したが 2 回計測で順位が入れ替わり厳密一致テーブルへは含めない
        // ため（`select_keeps_pre_744_behavior_for_rank_unstable_measured_points`
        // 参照）、本テストはそれ以外の `k`（実測対象外）で検証する。
        let cfg = select_for_device(2048, 2048, 1536, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[0]);
        assert_eq!((cfg.bm, cfg.bn), (64, 64));
    }

    #[test]
    fn select_keeps_pre_744_behavior_for_rank_unstable_measured_points() {
        // イシュー #1039・2026-08-31/09-01 M4 Max 実機実測（PR #1108
        // codex-review・cursor[bot] 指摘対応）: `(2048,2048,64)`・
        // `(2048,2048,512)`・`(1024,1536,1536)` は 2 回計測
        // （run1/run2。`docs/perf/logs/metal-gemm-tile-table-1039/`）で
        // 最良候補の順位が入れ替わった（プロセス間変動。
        // `docs/perf/metal-gemm-tile-table.md` §3・§5「順位不安定のため
        // 反映しない」節）ため、厳密一致テーブルへ含めない（本ファイル
        // 冒頭コメントの「2 回一致した点のみ反映する」方針どおり）。
        // よって #744 是正前の挙動（`m,n >= LARGE(512)` なら
        // CANDIDATES[0]）のまま変わらないことを固定する回帰ガード。
        assert_eq!(
            select_for_device(2048, 2048, 64, verified_m4_max_for_test()),
            CANDIDATES[0]
        );
        assert_eq!(
            select_for_device(2048, 2048, 512, verified_m4_max_for_test()),
            CANDIDATES[0]
        );
        assert_eq!(
            select_for_device(1024, 1536, 1536, verified_m4_max_for_test()),
            CANDIDATES[0]
        );
    }

    #[test]
    fn select_picks_mid_square_config_for_moderate_shapes() {
        let cfg = select_for_device(128, 128, 128, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[3]);
        assert_eq!((cfg.bm, cfg.bn), (32, 32));
    }

    #[test]
    fn select_keeps_pre_744_behavior_for_near_square_large_rectangle() {
        // PR #760 codex-review 指摘対応: 縦長・横長いずれにも該当しないが
        // `m != n` の準正方長方形一般に対して、実測されていない `(m, n,
        // k)` の組へ CANDIDATES[3] 一律選択を広げる根拠はない。この帯域は
        // 引き続き #744 是正前の挙動（`m,n >= 512` なら CANDIDATES[0]）を
        // 既定として維持する（実測点のみイシュー #1039 の厳密一致テーブル
        // で上書きする。下の `select_picks_measured_best_config_for_
        // near_square_large_rectangle` 参照）。本例は `(1536, 1024, 512)`
        // で `k` を実測点（1024）と変えて厳密一致から外している。
        let cfg = select_for_device(1536, 1024, 512, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[0]);
        assert_eq!((cfg.bm, cfg.bn), (64, 64));
    }

    #[test]
    fn select_picks_measured_best_config_for_near_square_large_rectangle() {
        // イシュー #1039・2026-08-31/09-01 M4 Max 実機実測: 準正方長方形
        // （`m != n`・縦横比 < 2）の `(1536,1024,1024)` は 2 回計測いずれも
        // `CANDIDATES[1]` が最良で一致しており、CANDIDATES[0]（是正前の
        // 安全側フォールバック値）比 約 5〜6 倍の逸失があることを確認した
        // （CANDIDATES[0] 1.16〜1.18 TFLOPS に対し CANDIDATES[1] 6.29〜
        // 6.35 TFLOPS。詳細値は `docs/perf/metal-gemm-tile-table.md`）。
        // `(1024,1536,1536)` は 2 回計測で順位が入れ替わったため厳密一致
        // テーブルへ含めない（
        // `select_keeps_pre_744_behavior_for_rank_unstable_measured_points`
        // 参照）。
        assert_eq!(
            select_for_device(1536, 1024, 1024, verified_m4_max_for_test()),
            CANDIDATES[1]
        );
    }

    #[test]
    fn select_keeps_pre_744_behavior_for_near_square_moderate_rectangle() {
        // 上と対の回帰ガード: `m,n < LARGE(512)` の準正方長方形（縦長・横長
        // いずれにも非該当）は #744 是正前後で挙動が変わらない
        // （どちらの経路でも CANDIDATES[3]）ことを固定する。`LARGE` 分岐の
        // 再導入（PR #760 レビュー対応）がこの帯域を誤って変えていないかの
        // 検証。
        let cfg = select_for_device(128, 192, 128, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[3]);
        assert_eq!((cfg.bm, cfg.bn), (32, 32));
    }

    #[test]
    fn select_picks_tall_config_when_m_dominates() {
        let cfg = select_for_device(1024, 128, 256, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[1]);
        assert_eq!((cfg.bm, cfg.bn), (64, 32));
    }

    #[test]
    fn select_picks_wide_config_when_n_dominates() {
        let cfg = select_for_device(128, 1024, 256, verified_m4_max_for_test());
        assert_eq!(cfg, CANDIDATES[2]);
        assert_eq!((cfg.bm, cfg.bn), (32, 64));
    }

    // --- select_with_occupancy（occupancy 縮退組み込み。イシュー #542）---

    #[test]
    fn select_with_occupancy_none_matches_select_for_representative_shapes() {
        // params=None は occupancy 判定を無効化し select() と完全一致する
        // （挙動不変の回帰ガード。#542 計画「fail-safe フォールバック方針」）。
        for &(m, n, k) in &[
            (32, 32, 32),
            (1000, 1000, 32),
            (1024, 1024, 1024),
            (128, 128, 128),
            (1024, 128, 256),
            (128, 1024, 256),
            (1536, 1024, 1024), // 準正方大形状長方形（PR #760 レビュー対応で追加）
            (128, 192, 128),    // 準正方中形状長方形（同上）
            (2048, 2048, 64),   // m==n だが K 未実測の正方出力（PR #760 レビュー対応で追加）
        ] {
            assert_eq!(
                select_with_occupancy_for_device(m, n, k, verified_m4_max_for_test(), None),
                select_for_device(m, n, k, verified_m4_max_for_test()),
                "m={m} n={n} k={k}"
            );
        }
    }

    /// M4 Max 想定値（#542 計画「現状分析」節の事前検証値: コア数 40・
    /// SMEM 32768 バイト）。`OccupancyParams` は実機実測値ではなく机上計算
    /// 用の暫定値であり、確定は Mac 実機セッション（実機ツリー #408 系）で
    /// 行う（`docs/perf/metal-gemm-occupancy-select.md` 参照）。
    fn m4_max_expected_params() -> OccupancyParams {
        OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        }
    }

    #[test]
    fn select_with_occupancy_true_square_shapes_bypass_occupancy_shrink_via_step1() {
        // #744 是正後、段 1（形状判定）は `m == n`（真の正方形状）に対して
        // 常に CANDIDATES[3]（大タイル系 CANDIDATES[0..=2] に非該当）を
        // 返すため、occupancy 縮退（段 2）の対象外となり params の値に
        // 関わらず CANDIDATES[3] のまま確定する。旧テスト
        // `select_with_occupancy_shrinks_512_square_under_m4_max_expected_params`・
        // `select_with_occupancy_keeps_large_squares_from_1024_under_m4_max_expected_params`
        // は「正方大形状 → CANDIDATES[0]」だった段 1 の挙動を前提に occupancy
        // 縮退の有無で 512 と 1024/2048/4096 を書き分けていたが、その分岐が
        // 撤去されたため意味を失い本テストへ統合する
        // （occupancy 縮退の検証自体は縦長/横長形状のテストが引き続き担う）。
        // イシュー #1039 の厳密一致テーブル（`select_with_occupancy` 冒頭）
        // がこれら 4 点の `shape_cfg` を測定済み最良候補へ差し替える（P1・
        // codex-review 指摘・PR #1108 レビュー対応で occupancy 判定を迂回
        // しない構造へ変更済み）。512/1024/2048/4096 いずれも
        // `is_large_tile_candidate`（`bm*bn` が CANDIDATES[3] 超）の対象
        // であり、この `m4_max_expected_params()` 下では 4 点とも
        // `is_underoccupied` が偽（over-occupied）のため実際には縮退しない
        // （期待値は `select` 側と同一。`select_picks_measured_best_config_
        // for_true_square_shapes` 参照。underoccupied 時に実際に縮退する
        // ことは `select_with_occupancy_shrinks_exact_match_candidates_
        // when_underoccupied` が別途固定する）。
        assert_eq!(
            select_with_occupancy_for_device(
                512,
                512,
                512,
                verified_m4_max_for_test(),
                Some(m4_max_expected_params())
            ),
            CANDIDATES[5]
        );
        assert_eq!(
            select_with_occupancy_for_device(
                1024,
                1024,
                1024,
                verified_m4_max_for_test(),
                Some(m4_max_expected_params())
            ),
            CANDIDATES[6]
        );
        assert_eq!(
            select_with_occupancy_for_device(
                2048,
                2048,
                2048,
                verified_m4_max_for_test(),
                Some(m4_max_expected_params())
            ),
            CANDIDATES[1]
        );
        assert_eq!(
            select_with_occupancy_for_device(
                4096,
                4096,
                4096,
                verified_m4_max_for_test(),
                Some(m4_max_expected_params())
            ),
            CANDIDATES[2]
        );
    }

    #[test]
    fn select_with_occupancy_shrinks_exact_match_candidates_when_underoccupied() {
        // P1・codex-review 指摘・PR #1108 レビュー対応の回帰テスト:
        // 厳密一致テーブル（イシュー #1039）が返す `CANDIDATES[5]`／`[6]`
        // （いずれも `bm=64,bn=32` で `CANDIDATES[1]`・縦長・大タイル系と
        // 同一の threadgroup 分担面積）が、`is_large_tile_candidate` の
        // 固定列挙（旧実装は `CANDIDATES[0..=2]` のみ）により構造上常に
        // occupancy 縮退の対象外（867 行目相当で早期 return）になっていた
        // 問題を固定する。`gpu_core_count` を小さくして意図的に
        // under-occupied な状況を作り、実際に `CANDIDATES[3]`（中形状）へ
        // 縮退することを検証する（`max_threadgroup_memory_bytes` は
        // `m4_max_expected_params()` と同じ 32768 バイトのままにし、
        // `smem_groups_per_core` によるキャップは変えず `gpu_core_count`
        // のみを縮小することで `ideal_groups` を下げる）。
        //
        // `is_underoccupied(actual, ideal)` は `actual <= ideal`（groups
        // 数が目標に対して少なすぎる＝under-occupied）で縮退する契約
        // （本ファイル [`is_underoccupied`] ドキュメント参照）。厳密一致点
        // は `m == n` が大きく `actual_groups` 自体が大きいため、通常の
        // `m4_max_expected_params()`（コア数 40）では over-occupied
        // （縮退しない。上の `..._bypass_occupancy_shrink_via_step1` が
        // 固定）。ここでは意図的にコア数を大きくし `ideal_groups` を
        // `actual_groups` 以上へ引き上げることで under-occupied を作る。
        //
        // (512,512,512) → 段 1 は `CANDIDATES[5]`（bm=64,bn=32,bk=32）。
        // actual_groups = ceil(512/64)*ceil(512/32) = 8*16 = 128。
        // shared_mem_bytes = (64*(32+4) + 32*(32+4))*4 = 13824 →
        // smem_groups_per_core = min(6, 32768/13824=2) = 2 →
        // ideal_groups = gpu_core_count(100) * 2 = 200。128 <= 200 のため
        // under-occupied（縮退）。
        let large_core_count_params = OccupancyParams {
            gpu_core_count: 100,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(
            select_with_occupancy_for_device(
                512,
                512,
                512,
                verified_m4_max_for_test(),
                Some(large_core_count_params)
            ),
            CANDIDATES[3],
            "CANDIDATES[5]（512 立方の厳密一致点）も under-occupied 時は縮退する"
        );

        // (1024,1024,1024) → 段 1 は `CANDIDATES[6]`（bm=64,bn=32,bk=8）。
        // actual_groups = ceil(1024/64)*ceil(1024/32) = 16*32 = 512。
        // shared_mem_bytes = (64*(8+4) + 8*(32+4))*4 = 4224 →
        // smem_groups_per_core = min(6, 32768/4224=7) = 6 →
        // ideal_groups = gpu_core_count(100) * 6 = 600。512 <= 600 のため
        // under-occupied（縮退）。
        assert_eq!(
            select_with_occupancy_for_device(
                1024,
                1024,
                1024,
                verified_m4_max_for_test(),
                Some(large_core_count_params)
            ),
            CANDIDATES[3],
            "CANDIDATES[6]（1024 立方の厳密一致点）も under-occupied 時は縮退する"
        );
    }

    #[test]
    fn select_with_occupancy_747_confirms_absorption_by_744_correction() {
        // イシュー #747「occupancy 選定式のサイズ帯条件分岐」の判断確定を
        // 直接固定する回帰テスト: 2026-08-19 M4 Max 実機比較（512 で
        // +45.6%・2048 で −5.6%・1024/4096 で ±1% 未満）が求めていた
        // 「小サイズ帯のみ occupancy 有効化」は、#744 是正（段 1 が実測
        // 帯域の正方立方形状に対し occupancy 縮退を経ず直接 CANDIDATES[3]
        // を返すよう是正）により吸収された。実測帯域〈512/1024/2048/4096〉
        // で occupancy 縮退込み（`Some(params)`）と形状のみ（`select`）が
        // 常に同一構成を返すことを固定し、`dispatch_auto` への
        // `select_with_occupancy` 組み込み不採用の判断根拠とする
        // （docs/perf/metal-gemm-occupancy-select.md「#747 判断」節）。
        for &size in &[512usize, 1024, 2048, 4096] {
            assert_eq!(
                select_with_occupancy_for_device(
                    size,
                    size,
                    size,
                    verified_m4_max_for_test(),
                    Some(m4_max_expected_params())
                ),
                select_for_device(size, size, size, verified_m4_max_for_test()),
                "size={size}"
            );
        }
    }

    #[test]
    fn select_with_occupancy_keeps_near_square_large_rectangle_without_shrink_when_occupied() {
        // PR #760 codex-review 指摘対応: `m != n` の準正方大形状長方形は
        // 段 1 で CANDIDATES[0]（大タイル系）を返すため（`LARGE` 分岐の
        // 再導入）、occupancy 縮退（段 2）が #744 是正前と同様に生きた
        // ままであることを固定する。`k=512`（`m=1536, n=1024` の実測点
        // `k=1024` はイシュー #1039 の厳密一致テーブルが `shape_cfg` を
        // 測定済み最良候補 `CANDIDATES[1]` へ差し替えるため、この境界計算
        // の検証に使えなくなった。`actual_groups`/`ideal_groups` は
        // `m`/`n`/`shape_cfg` のみに依存し `k` に依存しないため、`k` を
        // 変えても以下の計算は同一のまま成立する）。
        //
        // m=1536, n=1024, CANDIDATES[0]（bm=bn=64, pad=4）:
        // actual_groups = ceil(1536/64)*ceil(1024/64) = 24*16 = 384。
        // smem_groups_per_core = min(6, 32768/9472=3) = 3 →
        // ideal_groups = 40*3 = 120。384 > 120 のため over-occupied（縮退
        // しない）。
        let cfg = select_with_occupancy_for_device(
            1536,
            1024,
            512,
            verified_m4_max_for_test(),
            Some(m4_max_expected_params()),
        );
        assert_eq!(cfg, CANDIDATES[0]);
    }

    #[test]
    fn select_with_occupancy_shrinks_tall_shape_when_underoccupied() {
        // 縦長（512x128）: 段 1 は CANDIDATES[1]（64x32）を選ぶ。
        // actual_groups = ceil(512/64)*ceil(128/32) = 8*4 = 32。
        // CANDIDATES[1] の smem_groups_per_core=4（7424 バイト）→
        // ideal_groups=40*4=160。32 <= 160 で under-occupied のため
        // CANDIDATES[3] へ縮退する。
        let cfg = select_with_occupancy_for_device(
            512,
            128,
            256,
            verified_m4_max_for_test(),
            Some(m4_max_expected_params()),
        );
        assert_eq!(cfg, CANDIDATES[3]);
    }

    #[test]
    fn select_with_occupancy_shrinks_wide_shape_when_underoccupied() {
        // 横長（128x512）: 段 1 は CANDIDATES[2]（32x64）を選ぶ。縦長と対称の
        // 形状のため同じく under-occupied となり CANDIDATES[3] へ縮退する。
        let cfg = select_with_occupancy_for_device(
            128,
            512,
            256,
            verified_m4_max_for_test(),
            Some(m4_max_expected_params()),
        );
        assert_eq!(cfg, CANDIDATES[3]);
    }

    #[test]
    fn select_with_occupancy_shrinks_on_boundary_actual_equals_ideal() {
        // 境界一致（actual == ideal）は縮退側に倒れる fail-safe 契約
        // （`is_underoccupied` の既存契約。本テストは select_with_occupancy
        // 統合後もその契約が保たれることを固定する）。
        //
        // #744 是正で段 1 が正方形状に対し CANDIDATES[0]（大タイル系）を
        // 返さなくなったため、旧テストの m=n=768 正方形状は本境界を検証
        // できなくなった（段 1 で CANDIDATES[3] が確定し occupancy 縮退の
        // 対象外になる）。縦長形状（CANDIDATES[1]・64x32）へ retarget する:
        // gpu_core_count=24・十分大きい max_threadgroup_memory_bytes（SMEM
        // 制約が効かず effective_multiplier が IDEAL_GROUPS_MULTIPLIER_F32=6
        // のまま）なら ideal_groups(CANDIDATES[1]) = 24*6 = 144。
        // m=2304, n=128（tall: m >= 2*n）: 段 1 は CANDIDATES[1] を選ぶ。
        // actual_groups = ceil(2304/64) * ceil(128/32) = 36*4 = 144。
        let params = OccupancyParams {
            gpu_core_count: 24,
            max_threadgroup_memory_bytes: 1024 * 1024,
        };
        let cfg = select_with_occupancy_for_device(
            2304,
            128,
            768,
            verified_m4_max_for_test(),
            Some(params),
        );
        assert_eq!(cfg, CANDIDATES[3]);
    }

    #[test]
    fn select_with_occupancy_does_not_further_shrink_mid_or_tiny_candidates() {
        // 段 1 が CANDIDATES[3]（中形状）・SINGLE_SIMDGROUP_8X8（微小形状）を
        // 返す形状は、occupancy 縮退の対象（大タイル系のみ）に含まれない
        // ため occupancy パラメータの値に関わらず段 1 の結果を維持する。
        let extreme_params = OccupancyParams {
            gpu_core_count: 1,
            max_threadgroup_memory_bytes: 1,
        };
        assert_eq!(
            select_with_occupancy_for_device(
                128,
                128,
                128,
                verified_m4_max_for_test(),
                Some(extreme_params)
            ),
            CANDIDATES[3]
        );
        assert_eq!(
            select_with_occupancy_for_device(
                32,
                32,
                32,
                verified_m4_max_for_test(),
                Some(extreme_params)
            ),
            TileConfig::SINGLE_SIMDGROUP_8X8
        );
    }

    #[test]
    fn select_with_occupancy_falls_back_when_gpu_core_count_is_zero() {
        // ideal_groups() が None（gpu_core_count==0）の場合、occupancy
        // 判定を無効化し段 1 の結果（select() と同一）へフォールバックする。
        let params = OccupancyParams {
            gpu_core_count: 0,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(
            select_with_occupancy_for_device(
                1024,
                1024,
                1024,
                verified_m4_max_for_test(),
                Some(params)
            ),
            select_for_device(1024, 1024, 1024, verified_m4_max_for_test())
        );
    }

    #[test]
    fn select_with_occupancy_falls_back_when_smem_budget_is_exceeded() {
        // ideal_groups() が None（smem_groups_per_core(cfg)==0。デバイスの
        // threadgroup memory 上限が CANDIDATES[0] の使用量 9472 バイトに
        // 満たない）の場合も段 1 の結果へフォールバックする。
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 1024, // CANDIDATES[0] の 9472 バイト未満
        };
        assert_eq!(
            select_with_occupancy_for_device(
                1024,
                1024,
                1024,
                verified_m4_max_for_test(),
                Some(params)
            ),
            select_for_device(1024, 1024, 1024, verified_m4_max_for_test())
        );
    }

    #[test]
    fn select_with_occupancy_result_always_validates_under_typical_device_limits() {
        // select_result_always_validates_under_typical_device_limits（下記）の
        // occupancy 込み版。縮退後の構成も含め常に validate を通ることを
        // 固定する（REQ-8 境界検査・パイプライン確保時の panic 防止）。
        for &(m, n, k) in &[
            (7usize, 13usize, 5usize),
            (512, 512, 512),
            (512, 128, 256),
            (128, 512, 256),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 512, 512),
        ] {
            let cfg = select_with_occupancy_for_device(
                m,
                n,
                k,
                verified_m4_max_for_test(),
                Some(m4_max_expected_params()),
            );
            cfg.validate(1024, 32 * 1024).unwrap_or_else(|e| {
                panic!("select_with_occupancy_for_device({m}, {n}, {k}) rejected: {e}")
            });
        }
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
            let cfg = select_for_device(m, n, k, verified_m4_max_for_test());
            cfg.validate(1024, 32 * 1024)
                .unwrap_or_else(|e| panic!("select({m},{n},{k})={cfg:?} rejected: {e}"));
        }
    }

    // --- swizzled_grid（イシュー #540。pure） ---

    /// `shaders/gemm.metal` の tgid 変換式（`gemm_simdgroup_tiled_source_
    /// uses_tgid_swizzle`〈`shader_source_evidence.rs`〉が実在をロックする
    /// 式と 1:1 対応するホスト側リファレンス実装）。`swizzled_grid` が
    /// 張った grid 全体を tgid が走査したとき、この関数が返す
    /// `(tid_y, tid_x)` が `0..tiles_m × 0..tiles_n` を過不足なく 1 回ずつ
    /// 覆うことを下記テストで検証する（カーネル計算の欠落・二重書き込みを
    /// Linux 上で静的に防止する。計画「実装ステップ」5 節）。
    fn swizzle_reference_remap(tgid_y: usize, tgid_x: usize) -> (usize, usize) {
        let tile = 1usize << SWIZZLE_LOG;
        let tid_y = (tgid_y << SWIZZLE_LOG) + (tgid_x & (tile - 1));
        let tid_x = tgid_x >> SWIZZLE_LOG;
        (tid_y, tid_x)
    }

    #[test]
    fn swizzled_grid_covers_every_tile_exactly_once() {
        // `tile`（4）の倍数・非倍数・1 を含む組み合わせ（計画「実装ステップ」
        // 5 節）。カーネル側の早期 return（REQ-8）が吸収する余剰
        // threadgroup（`tid_y >= tiles_m`）はここでは無害化前提として除外
        // して数える。
        for &(tiles_m, tiles_n) in &[
            (1usize, 1usize),
            (4, 4),
            (8, 8),
            (5, 3),
            (7, 9),
            (16, 1),
            (1, 16),
            (13, 5),
        ] {
            let (grid_w, grid_h) = swizzled_grid(tiles_n, tiles_m);
            let mut covered = std::collections::HashSet::new();
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let (tid_y, tid_x) = swizzle_reference_remap(gy, gx);
                    // カーネル側早期 return と同じ境界（row0/col0 換算前の
                    // タイル座標）を吸収する: 実域外は書き込み対象にしない。
                    if tid_y >= tiles_m || tid_x >= tiles_n {
                        continue;
                    }
                    assert!(
                        covered.insert((tid_y, tid_x)),
                        "(tiles_m={tiles_m}, tiles_n={tiles_n}) で (tid_y={tid_y}, tid_x={tid_x}) \
                         が複数回書き込まれた（grid=({grid_w}, {grid_h})）"
                    );
                }
            }
            assert_eq!(
                covered.len(),
                tiles_m * tiles_n,
                "(tiles_m={tiles_m}, tiles_n={tiles_n}) で全タイルが被覆されなかった \
                 （被覆数: {}, 期待: {}, grid=({grid_w}, {grid_h})）",
                covered.len(),
                tiles_m * tiles_n
            );
        }
    }

    #[test]
    fn swizzled_grid_matches_div_ceil_scaling() {
        // grid_w = tiles_n << SWIZZLE_LOG（4 倍）・grid_h =
        // div_ceil(tiles_m, 1 << SWIZZLE_LOG)（計画「設計方針」節の式）。
        assert_eq!(swizzled_grid(1, 1), (4, 1));
        assert_eq!(swizzled_grid(2, 4), (8, 1));
        assert_eq!(swizzled_grid(2, 5), (8, 2));
        assert_eq!(swizzled_grid(3, 9), (12, 3));
    }

    #[test]
    fn tiled_dispatch_grid_matches_swizzle_enabled_gate() {
        // PR #661 codex-review 指摘対応: `SWIZZLE_ENABLED` は本番既定
        // `false` に固定されている（実機未検証のため。tile.rs 冒頭の
        // `SWIZZLE_ENABLED` doc comment参照）。ここでは
        // `tiled_dispatch_grid` が現在の `SWIZZLE_ENABLED` 値に応じて
        // 正しい枝へ分岐することのみを検査する（定数値そのものの妥当性は
        // 上記 doc comment の運用ルールが担保する）。
        let (tiles_n, tiles_m) = (3usize, 9usize);
        let expected = if SWIZZLE_ENABLED {
            swizzled_grid(tiles_n, tiles_m)
        } else {
            (tiles_n, tiles_m)
        };
        assert_eq!(tiled_dispatch_grid(tiles_n, tiles_m), expected);
    }

    /// `SWIZZLE_ENABLED` の**コミット状態既定値**が `false` に固定されて
    /// いることを恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）でロックする
    /// 独立テスト（PR #661 codex-review・Cursor Bugbot 指摘対応:
    /// review_r3791409328 と同一箇所を指す独立指摘 2 件。「SWIZZLE_ENABLED
    /// の誤有効化をテストが検出できない」）。
    ///
    /// [`tiled_dispatch_grid_matches_swizzle_enabled_gate`] の `expected` は
    /// 現在の `SWIZZLE_ENABLED` 値へ追従して分岐を選ぶため、`true` のまま
    /// 誤ってコミットされてもあの assert 自体は通ってしまう（分岐選択の
    /// 正しさしか検査していない）。本テストはそれとは別に、**恒等変換を
    /// 無条件に固定値で assert する**ことで「今のビルドで
    /// `SWIZZLE_ENABLED` が実際どちらの値でコンパイルされたか」を検出する。
    ///
    /// あえてコンパイル時 `const { assert!(...) }` にはしない
    /// （review_r3791409328 の教訓）: `docs/perf/metal-gemm-tgid-swizzle-
    /// ab.md` の A/B 計測手順は実機セッションで `SWIZZLE_ENABLED` を一時的に
    /// `true` へ書き換えたうえで `cargo test --release -- --ignored`
    /// （ignored テストのみ選択実行）を叩く運用のため、コンパイル時
    /// アサーションだとその一時変更のたびに crate 全体がビルド不能になり
    /// 計測できなくなる。本テストは通常の `#[test]`（非 `#[ignore]`）の
    /// ため `--ignored` フィルタでは実行対象外となり A/B 計測時の妨げには
    /// ならない一方、通常 CI（`cargo test`。SWIZZLE_ENABLED=true のまま
    /// 誤コミットされた状態を含む）では必ず実行され、`true` のままの
    /// コミットを実行時 panic で確実に検出する。
    #[test]
    fn tiled_dispatch_grid_is_identity_by_default() {
        let (tiles_n, tiles_m) = (3usize, 9usize);
        assert_eq!(
            tiled_dispatch_grid(tiles_n, tiles_m),
            (tiles_n, tiles_m),
            "SWIZZLE_ENABLED が true のままコミットされている疑いがあります。\
             実機未検証のスウィズルは本番既定 false に固定する契約です \
             （tile.rs 冒頭の SWIZZLE_ENABLED doc comment・PR #661 参照）。"
        );
    }

    /// `FINE_BARRIER_ENABLED` の**コミット状態既定値**が `false` に固定
    /// されていることをロックする独立テスト（`tiled_dispatch_grid_is_
    /// identity_by_default`〈`SWIZZLE_ENABLED` 用〉と同じ設計判断: この
    /// 定数は `crate::gemm::MetalGemm::new`（`target_os = "macos"` 限定）
    /// からのみ参照されるため、Linux 上の `tile` モジュール単体では
    /// 到達不能で dead_code 警告の対象になる。本テストは定数値を直接
    /// assert することで dead_code を解消しつつ、A/B 計測セッションで
    /// 一時的に `true` へ書き換えたまま誤コミットされた状態を通常 CI
    /// （`cargo test`）で確実に検出する（`tile.rs` 冒頭
    /// `FINE_BARRIER_ENABLED` doc comment参照）。
    #[test]
    fn fine_barrier_enabled_is_false_by_default() {
        // `assert!(!FINE_BARRIER_ENABLED, ..)` は clippy
        // `assertions_on_constants` に抵触する（値がコンパイル時定数のため。
        // `SWIZZLE_ENABLED` 側は `tiled_dispatch_grid(..)` という消費側関数
        // 呼び出しを経由することでこの lint を回避している。本定数には
        // tile.rs 内に消費側関数が無いため、`std::hint::black_box` で
        // 「コンパイル時に定数畳み込みされない値」へ変換して同じ回避を行う）。
        assert!(
            !std::hint::black_box(FINE_BARRIER_ENABLED),
            "FINE_BARRIER_ENABLED が true のままコミットされている疑いがあります。\
             実機未検証の simdgroup 細粒度同期は本番既定 false に固定する契約です \
             （tile.rs 冒頭 FINE_BARRIER_ENABLED doc comment・イシュー #809 参照）。"
        );
    }

    // --- shaders/gemm.metal のスウィズル証跡検査（イシュー #540・PR #661
    // codex-review 指摘対応） ---
    //
    // `SWIZZLE_LOG`/`SWIZZLE_ENABLED` は `pub(crate)`（クレート外非公開）の
    // ため、これらを参照するシェーダ証跡検査は統合テスト（`tests/` 配下・
    // 別コンパイル単位で公開 API しか見えない）ではなくここに置く
    // （旧 `tests/shader_source_evidence.rs::
    // gemm_simdgroup_tiled_source_uses_tgid_swizzle` から移設。直上の
    // `CANDIDATES` 巡回テストと同じ理由）。

    /// `crates/backend-metal/src/shaders/gemm.metal` のソース全文。
    const GEMM_METAL_SOURCE: &str = include_str!("shaders/gemm.metal");

    /// `gemm_simdgroup_tiled` カーネル本体（`kernel void
    /// gemm_simdgroup_tiled(` 開始位置から EOF まで）を切り出す。本ファイル
    /// 内で最後に定義されるカーネルのため EOF までのスライスで安全
    /// （`tests/shader_source_evidence.rs::gemm_simdgroup_tiled_kernel_body`
    /// と同一ロジック）。
    fn gemm_simdgroup_tiled_kernel_body() -> &'static str {
        let kernel_start = GEMM_METAL_SOURCE
            .find("kernel void gemm_simdgroup_tiled(")
            .expect("gemm_simdgroup_tiled カーネル本体が見つかりません");
        &GEMM_METAL_SOURCE[kernel_start..]
    }

    /// イシュー #540 の証跡: `gemm_simdgroup_tiled` 冒頭に threadgroup ID
    /// スウィズル（`swizzle_log` 相当）の tgid 変換が実装され、シェーダ側の
    /// `SWIZZLE_LOG` リテラルが `crate::tile::SWIZZLE_LOG` と一致している
    /// こと、かつ本番既定では `SWIZZLE_ENABLED` function constant により
    /// 恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）へフォールバックする
    /// 分岐がシェーダ側に実在することを Linux CI（ubuntu-latest）上で
    /// ロックする。Mac 実機依存の A/B 計測（`docs/perf/
    /// metal-gemm-tgid-swizzle-ab.md`）は別途実施し、改善が無ければこの
    /// テストごと変更を撤去（revert）する運用とする（`metal-gemm-
    /// serpentine-ab.md` と同じ運用。#536 前例踏襲）。
    ///
    /// PR #661 codex-review 指摘対応: 実機未検証のまま `SWIZZLE_ENABLED`
    /// を無条件 `true` で本番経路へ適用しないよう、シェーダ側が
    /// `SWIZZLE_ENABLED` 分岐を持つこと自体をここでロックする
    /// （`crate::gemm::encode_dispatch_tiled` 側の同期は本ファイルの
    /// `tiled_dispatch_grid_matches_swizzle_enabled_gate` が Rust 側の
    /// grid 計算の分岐を、[`swizzled_grid_covers_every_tile_exactly_once`]
    /// が `swizzled_grid` 自体の走査網羅性を、それぞれ別途確認する）。
    #[test]
    fn gemm_simdgroup_tiled_source_uses_tgid_swizzle() {
        let kernel_body = gemm_simdgroup_tiled_kernel_body();
        assert!(
            kernel_body.contains(&format!("constexpr uint SWIZZLE_LOG = {SWIZZLE_LOG};")),
            "gemm_simdgroup_tiled に SWIZZLE_LOG 定数（値 {SWIZZLE_LOG}。crate::tile::SWIZZLE_LOG と一致契約）が見つかりません"
        );
        // SWIZZLE_ENABLED の function constant 宣言はファイル冒頭
        // （カーネル本体の外・他の function constant と並べた位置）にある
        // ため、`kernel_body` ではなく `GEMM_METAL_SOURCE` 全文を検索する。
        // index まで含めて検査する（#538 の `TGP_PAD`〈index 6〉との
        // index 重複を機械的に検出するため。origin/main との merge
        // コンフリクト解決〈両者とも index 6 を主張していた〉で実際に
        // 衝突していたことがあり、名前だけの検査では黙って再発しうる）。
        assert!(
            GEMM_METAL_SOURCE.contains("constant bool SWIZZLE_ENABLED [[function_constant(7)]];"),
            "gemm.metal に SWIZZLE_ENABLED function constant（index 7。TGP_PAD〈#538・index 6〉の \
             直後）の宣言が見つかりません（実機未検証のまま本番 dispatch へ無条件適用しないための \
             ゲート。PR #661 codex-review 指摘）"
        );
        assert!(
            kernel_body.contains(
                "uint tid_y = SWIZZLE_ENABLED ? ((tgid.y << SWIZZLE_LOG) + (tgid.x & (SWIZZLE_TILE - 1))) : tgid.y;"
            ),
            "gemm_simdgroup_tiled に SWIZZLE_ENABLED ゲート付き tid_y スウィズル変換式が見つかりません"
        );
        assert!(
            kernel_body
                .contains("uint tid_x = SWIZZLE_ENABLED ? (tgid.x >> SWIZZLE_LOG) : tgid.x;"),
            "gemm_simdgroup_tiled に SWIZZLE_ENABLED ゲート付き tid_x スウィズル変換式が見つかりません"
        );
        assert!(
            kernel_body.contains("uint row0 = tid_y * BM;")
                && kernel_body.contains("uint col0 = tid_x * BN;"),
            "gemm_simdgroup_tiled の row0/col0 計算がスウィズル後の tid_y/tid_x を使っていません"
        );
    }

    // --- occupancy 目標算出（イシュー #541・D-7a）: actual_groups ---

    #[test]
    fn actual_groups_matches_m4_max_reference_table_for_64x64_tile() {
        // `docs/perf/metal-gemm-bottleneck-diagnosis.md` §3.2 の事前計算表
        // （64x64 タイル・正方形状）と一致することを固定する。
        let cfg = CANDIDATES[0]; // bm=bn=64
        assert_eq!(actual_groups(512, 512, cfg), Some(64));
        assert_eq!(actual_groups(1024, 1024, cfg), Some(256));
        assert_eq!(actual_groups(2048, 2048, cfg), Some(1024));
        assert_eq!(actual_groups(4096, 4096, cfg), Some(4096));
    }

    #[test]
    fn actual_groups_rounds_up_for_non_multiple_shapes() {
        // m=100 は bm=64 の倍数でないため ceil(100/64)=2 へ切り上がる。
        let cfg = CANDIDATES[0]; // bm=bn=64
        assert_eq!(actual_groups(100, 64, cfg), Some(2));
    }

    #[test]
    fn actual_groups_is_none_when_bm_is_zero() {
        let mut cfg = CANDIDATES[0];
        cfg.bm = 0;
        assert_eq!(actual_groups(512, 512, cfg), None);
    }

    #[test]
    fn actual_groups_is_none_when_bn_is_zero() {
        let mut cfg = CANDIDATES[0];
        cfg.bn = 0;
        assert_eq!(actual_groups(512, 512, cfg), None);
    }

    #[test]
    fn actual_groups_is_none_on_overflow() {
        // groups_m * groups_n が u64::MAX を超える極端な形状。
        let cfg = CANDIDATES[0]; // bm=bn=64
        assert_eq!(actual_groups(usize::MAX, usize::MAX, cfg), None);
    }

    // --- occupancy 目標算出: OccupancyParams::smem_groups_per_core ---

    #[test]
    fn smem_groups_per_core_divides_max_smem_by_tile_smem_bytes() {
        // CANDIDATES[0]（64x64x16, staged）: shared_mem_bytes = 9472
        // （イシュー #538・PR #673 で導入された行末パディング
        // `TileConfig::pad`＝`TGP_PAD_ELEMS=4` を含む値。pad 導入前は
        // 8192 だった）。
        let cfg = CANDIDATES[0];
        assert_eq!(cfg.shared_mem_bytes(), 9472);
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(params.smem_groups_per_core(cfg), 3); // 32768/9472=3（切り捨て）
    }

    #[test]
    fn smem_groups_per_core_bk32_candidate_yields_two() {
        // イシュー #532 の bk=32 候補: shared_mem_bytes = 13824
        // （#538・PR #673 のパディング込み。pad 導入前は 12288 だった）。
        let cfg = TileConfig {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes(), 13824);
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(params.smem_groups_per_core(cfg), 2); // 32768/13824=2（切り捨て）
    }

    #[test]
    fn smem_groups_per_core_is_zero_when_tile_exceeds_device_limit() {
        let cfg = TileConfig {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 2,
            wn: 2,
            staged: true,
        };
        // #538・PR #673 のパディング込みで 13824（pad 導入前は 12288）。
        assert_eq!(cfg.shared_mem_bytes(), 13824);
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 8192, // 13824 > 8192
        };
        assert_eq!(params.smem_groups_per_core(cfg), 0);
    }

    #[test]
    fn smem_groups_per_core_returns_cap_when_not_staged() {
        // staged=false（direct load）は threadgroup memory を使わないため
        // SMEM 制約による上限を持たず CAP をそのまま返す。
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(
            params.smem_groups_per_core(TileConfig::SINGLE_SIMDGROUP_8X8),
            SMEM_GROUPS_PER_CORE_CAP
        );
    }

    // --- occupancy 目標算出: OccupancyParams::ideal_groups ---

    #[test]
    fn ideal_groups_is_capped_by_smem_budget() {
        // CANDIDATES[0] は shared_mem_bytes=9472（#538・PR #673 のパディング
        // 込み）のため smem_groups_per_core=3 < multiplier=6 となり、
        // 実効係数は 3 に頭打ちされる（40*3=120。pad 導入前は
        // smem_groups_per_core=4・160 だった）。
        let cfg = CANDIDATES[0];
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(
            params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, cfg),
            Some(120)
        );
    }

    #[test]
    fn ideal_groups_matches_mfa_formula_when_smem_unconstrained() {
        // CANDIDATES[3]（32x32x16 中形状）: shared_mem_bytes=4864
        // （#538・PR #673 のパディング込み。pad 導入前は 4096 だった）のため
        // smem_groups_per_core=6 >= multiplier=6 で SMEM 制約が効かず、
        // 素の MFA 経験式（40*6=240。`docs/perf/
        // metal-gemm-bottleneck-diagnosis.md` §3 実測前提値）と一致する。
        let cfg = CANDIDATES[3];
        assert_eq!(cfg.shared_mem_bytes(), 4864);
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(
            params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, cfg),
            Some(240)
        );
    }

    #[test]
    fn ideal_groups_uses_cap_for_direct_load_config() {
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        // multiplier=20 > CAP=16 のため実効係数は 16 に頭打ちされる。
        assert_eq!(
            params.ideal_groups(20, TileConfig::SINGLE_SIMDGROUP_8X8),
            Some(640)
        );
    }

    #[test]
    fn ideal_groups_is_none_when_gpu_core_count_is_zero() {
        let params = OccupancyParams {
            gpu_core_count: 0,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(
            params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, CANDIDATES[0]),
            None
        );
    }

    #[test]
    fn ideal_groups_is_none_when_multiplier_is_zero() {
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 32 * 1024,
        };
        assert_eq!(params.ideal_groups(0, CANDIDATES[0]), None);
    }

    #[test]
    fn ideal_groups_is_none_when_smem_groups_per_core_is_zero() {
        // codex-review 指摘（PR #662）: cfg の threadgroup memory 使用量が
        // デバイス上限を超え smem_groups_per_core(cfg)==0 の場合、
        // effective_multiplier も 0 になる。これを検証せず素通りすると
        // `Some(0)` が返り、`is_underoccupied` へ渡した際に
        // `actual <= 0` が通常 false となって実行不能な構成を
        // 「under-occupied ではない」と誤判定してしまう（fail-closed
        // 契約違反）。`None` を返すことを固定する。
        let cfg = TileConfig {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 2,
            wn: 2,
            staged: true,
        };
        assert_eq!(cfg.shared_mem_bytes(), 13824);
        let params = OccupancyParams {
            gpu_core_count: 40,
            max_threadgroup_memory_bytes: 8192, // 13824 > 8192 のため常駐不可
        };
        assert_eq!(params.smem_groups_per_core(cfg), 0);
        assert_eq!(params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, cfg), None);
    }

    // --- occupancy 目標算出: is_underoccupied ---

    #[test]
    fn is_underoccupied_treats_exact_boundary_as_underoccupied() {
        assert!(is_underoccupied(160, 160));
    }

    #[test]
    fn is_underoccupied_is_false_when_actual_exceeds_ideal() {
        assert!(!is_underoccupied(161, 160));
    }

    #[test]
    fn is_underoccupied_is_true_when_actual_is_below_ideal() {
        assert!(is_underoccupied(64, 160));
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
        use bench_harness::rng::Xorshift64Star;
        use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

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

    /// `CANDIDATES` を全て、`gemm_simdgroup_tiled_f16`（イシュー #796）で
    /// 巡回し CPU 参照実装との一致を確認する
    /// （`all_tile_candidates_match_cpu_reference_medium_shape`（f32 版）と
    /// 同じ判断根拠・同じフォールバック検知手法。`MetalGemm::
    /// resolve_tile_config_f16` で実際に採用された構成が `cfg` と一致する
    /// ことを assert してからディスパッチする）。参照値は f16→f32→
    /// `matmul_reference_fma`→f16 丸め→f32 の 3 段階
    /// （`tests/cpu_metal_f16_parity.rs` 冒頭コメント「参照実装との比較
    /// 方法」と同一）。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn all_tile_candidates_match_cpu_reference_f16_tiled_medium_shape() {
        use bench_harness::rng::Xorshift64Star;
        use fandhe_ai_backend_cpu::parity::assert_parity;
        use half::f16;

        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = crate::gemm::MetalGemm::new(&ctx)
            .expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

        for (i, &cfg) in CANDIDATES.iter().enumerate() {
            let resolved = gemm.resolve_tile_config_f16(&ctx, cfg).unwrap_or_else(|err| {
                panic!("候補 {cfg:?}（f16）のパイプライン構築・検証（実デバイス上限）に失敗した: {err}")
            });
            assert_eq!(
                resolved, cfg,
                "候補 {cfg:?}（f16）が実デバイス上でサイレントに {resolved:?} へフォールバックした \
                 （構成失敗を検知できていない）"
            );

            let (m, n, k) = (256, 256, 256);
            let mut rng_a = Xorshift64Star::new(300 + i as u64);
            let mut rng_b = Xorshift64Star::new(400 + i as u64);
            let a_f16: Vec<f16> = rng_a.fill_vec_f16(m * k);
            let b_f16: Vec<f16> = rng_b.fill_vec_f16(k * n);

            let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
            let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
            let mut c_ref_f32 = vec![0.0f32; m * n];
            fandhe_ai_backend_cpu::parity::matmul_reference_fma(
                &a_f32,
                &b_f32,
                &mut c_ref_f32,
                m,
                n,
                k,
            )
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");
            let c_ref_rounded: Vec<f32> = c_ref_f32
                .iter()
                .map(|&x| f16::from_f32(x).to_f32())
                .collect();

            let c_gpu_f16 = gemm
                .dispatch_f16_tiled_unverified(&ctx, &a_f16, &b_f16, m, n, k, cfg)
                .unwrap_or_else(|err| {
                    panic!("Metal f16 SimdgroupTiled({cfg:?}) GEMM のディスパッチに失敗した: {err}")
                });
            let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

            assert_parity(
                &format!("metal f16 SimdgroupTiled({cfg:?}) gemm m={m} n={n} k={k}"),
                &c_gpu_f32,
                &c_ref_rounded,
            );
        }
    }

    /// イシュー #1038 の証跡: `CANDIDATES` を全て、8 の倍数ではあるが
    /// `bm`/`bn`（64・32）にも `bk`（32・16）にも揃わない非タイル倍数の
    /// 境界形状で検証する（f32 版）。`all_tile_candidates_match_cpu_
    /// reference_medium_shape`（m=n=k=256。全候補の `bm`/`bn`/`bk` を
    /// 割り切る「タイル倍数」形状）は境界検査（REQ-8）自体の実効性
    /// （ブロック原点の早期 return・協調ロードのベクトルグループ
    /// in-bounds 判定＋要素単位フォールバック）を一度も踏まない盲点が
    /// あり、これを埋める（#1038 計画「3.3 節」）。
    ///
    /// **形状の選定根拠**: `crate::gemm::MetalGemm::dispatch_variant` は
    /// `SimdgroupTiled` へ渡す前に `crate::pad::pad8` で m/n/k を 8 の倍数
    /// （実効次元）へ切り上げる契約のため、シェーダが見る境界は「8 の
    /// 倍数ではあるが `bm`/`bn`/`bk` の倍数ではない」ことで踏める。
    /// `CANDIDATES`（本ファイル上方）の `bm`/`bn` は 64・32 の 2 種、`bk`
    /// は 32・16・8 の 3 種（`TileConfig::SINGLE_SIMDGROUP_8X8` の
    /// bm=bn=bk=8 を除く）。m=100→pad8=104（104 mod 64=40・mod 32=8。
    /// いずれも非 0 のため `bm=64`・`bm=32` の両方でブロック端の部分タイル
    /// が生じる）、n=84→pad8=88（88 mod 64=24・mod 32=24。同様に両方の
    /// `bn` で部分タイルが生じる）、k=68→pad8=72（72 mod 32=8・mod 16=8。
    /// `bk=32`・`bk=16` の両方で K タイル末尾の端数〈`bk_eff<bk`〉が生じ、
    /// 協調ロードのベクトルグループ境界フォールバック〈`tiled_a_elem_
    /// in_bounds`/`tiled_b_elem_in_bounds`〉を実際に踏む）。`bk=8`
    /// （`wm4_bk8` 候補・`SINGLE_SIMDGROUP_8X8`）は pad8 後の k が既に
    /// 8 の倍数のため K タイル端数は生じないが、m/n 側のブロック端部分
    /// タイルは同様に踏む。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn all_tile_candidates_match_cpu_reference_non_multiple_boundary_shape() {
        use bench_harness::rng::Xorshift64Star;
        use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

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

            let (m, n, k) = (100, 84, 68);
            let seed_a = 1000 + i as u64;
            let seed_b = 2000 + i as u64;

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
                &format!(
                    "metal SimdgroupTiled({cfg:?}) gemm 非タイル倍数境界形状 m={m} n={n} k={k}"
                ),
                &actual,
                &expected,
            );
        }
    }

    /// 上記 f32 版の f16 版（イシュー #1038）:
    /// `all_tile_candidates_match_cpu_reference_f16_tiled_medium_shape`
    /// と同じ判断根拠・手法で、非タイル倍数の境界形状
    /// （`all_tile_candidates_match_cpu_reference_non_multiple_boundary_shape`
    /// と同一の m=100・n=84・k=68。選定根拠は同関数のコメント参照）を
    /// `gemm_simdgroup_tiled_f16` で検証する。参照値は f16→f32→
    /// `matmul_reference_fma`→f16 丸め→f32 の 3 段階
    /// （`all_tile_candidates_match_cpu_reference_f16_tiled_medium_shape`
    /// と同一手法）。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn all_tile_candidates_match_cpu_reference_f16_tiled_non_multiple_boundary_shape() {
        use bench_harness::rng::Xorshift64Star;
        use fandhe_ai_backend_cpu::parity::assert_parity;
        use half::f16;

        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = crate::gemm::MetalGemm::new(&ctx)
            .expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

        for (i, &cfg) in CANDIDATES.iter().enumerate() {
            let resolved = gemm.resolve_tile_config_f16(&ctx, cfg).unwrap_or_else(|err| {
                panic!("候補 {cfg:?}（f16）のパイプライン構築・検証（実デバイス上限）に失敗した: {err}")
            });
            assert_eq!(
                resolved, cfg,
                "候補 {cfg:?}（f16）が実デバイス上でサイレントに {resolved:?} へフォールバックした \
                 （構成失敗を検知できていない）"
            );

            let (m, n, k) = (100, 84, 68);
            let mut rng_a = Xorshift64Star::new(3000 + i as u64);
            let mut rng_b = Xorshift64Star::new(4000 + i as u64);
            let a_f16: Vec<f16> = rng_a.fill_vec_f16(m * k);
            let b_f16: Vec<f16> = rng_b.fill_vec_f16(k * n);

            let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
            let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
            let mut c_ref_f32 = vec![0.0f32; m * n];
            fandhe_ai_backend_cpu::parity::matmul_reference_fma(
                &a_f32,
                &b_f32,
                &mut c_ref_f32,
                m,
                n,
                k,
            )
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");
            let c_ref_rounded: Vec<f32> = c_ref_f32
                .iter()
                .map(|&x| f16::from_f32(x).to_f32())
                .collect();

            let c_gpu_f16 = gemm
                .dispatch_f16_tiled_unverified(&ctx, &a_f16, &b_f16, m, n, k, cfg)
                .unwrap_or_else(|err| {
                    panic!("Metal f16 SimdgroupTiled({cfg:?}) GEMM のディスパッチに失敗した: {err}")
                });
            let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

            assert_parity(
                &format!(
                    "metal f16 SimdgroupTiled({cfg:?}) gemm 非タイル倍数境界形状 m={m} n={n} k={k}"
                ),
                &c_gpu_f32,
                &c_ref_rounded,
            );
        }
    }

    /// `dispatch_f16_auto_unverified`（イシュー #798）が実際に呼ぶ経路——
    /// `tile::select(m, n, k)` の出力をそのまま `resolve_tile_config_f16`
    /// へ渡す——を、`select` の各分岐を代表する形状で検証する
    /// （`all_tile_candidates_match_cpu_reference_f16_tiled_medium_shape`
    /// が `CANDIDATES` を直接巡回するのに対し、本テストは `select` の
    /// 分岐判定ロジックが実際にどの候補へ写像するかを検証対象に含む）。
    /// 採用構成が `select` の返り値と一致することを assert することで、
    /// 自動経路がフォールバックなしに動作することを確認する。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn dispatch_f16_auto_tile_select_shapes_resolve_without_fallback() {
        let ctx = crate::context::MetalContext::new()
            .expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = crate::gemm::MetalGemm::new(&ctx)
            .expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");

        // `select_with_occupancy` 本体コメントの分岐判定に対応する代表形状:
        // 縦長 → CANDIDATES[1]、横長 → CANDIDATES[2]、正方立方（実測範囲内）
        // → CANDIDATES[3]、大形状 → CANDIDATES[0]、微小形状 →
        // SINGLE_SIMDGROUP_8X8。
        let shapes: &[(usize, usize, usize)] = &[
            (2048, 256, 512),  // 縦長 → CANDIDATES[1]
            (256, 2048, 512),  // 横長 → CANDIDATES[2]
            (512, 512, 512),   // 正方立方（イシュー #1039 実測点）→ CANDIDATES[5]
            (1536, 1024, 512), // 準正方大形状長方形 → CANDIDATES[0]
            (32, 32, 32),      // 微小形状 → SINGLE_SIMDGROUP_8X8
        ];

        for &(m, n, k) in shapes {
            let expected = select_for_device(m, n, k, verified_m4_max_for_test());
            let resolved = gemm
                .resolve_tile_config_f16(&ctx, expected)
                .unwrap_or_else(|err| {
                    panic!(
                        "shape (m={m}, n={n}, k={k}) が選んだ構成 {expected:?} の \
                     パイプライン構築・検証（実デバイス上限）に失敗した: {err}"
                    )
                });
            assert_eq!(
                resolved, expected,
                "shape (m={m}, n={n}, k={k}) の select 出力 {expected:?} が \
                 実デバイス上でサイレントに {resolved:?} へフォールバックした"
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
