//! 128×64×16 f32 pipeline カーネル（8×4 レジスタブロック・A フラグメント
//! XOR スウィズル・cp.async zfill 境界。イシュー #1343・親イシュー
//! #1342）。
//!
//! # 位置づけ（形状条件付きで本番結線済み。イシュー #1344）
//!
//! `kernels_tiled_pipeline.rs`（64×64×16・4×4 レジスタブロック。イシュー
//! #1033）の演算密度を 2 倍にした変種。GB10 実機実測（イシュー #1344。
//! `docs/perf/cuda-gemm-tiled-pipeline.md`「#1344」節）により、N・K が
//! いずれも 1024 以上の正方形状で 64×64 版より明確に優位（GPU-only 純
//! カーネル時間比の 5 回中央値: N=1024 で 1.05 倍・N=2048 で 1.12 倍・
//! N=4096 で 1.35 倍）と確認されたため、`gemm.rs::CudaGemm::
//! select_tiled_f32_kernel`（`run_tiled_f32` 系 3 入口が共有する選択
//! ロジック）が `tiled_pipeline_tile_kind`（N≥1024 かつ K≥1024 で
//! 128×64 を選ぶ純粋関数）経由で形状条件付きに本モジュールへ分岐する
//! （`gemm.rs::TILED_PIPELINE_128X64_PRODUCTION_ENABLED = true`）。
//! N/K が閾値未満の形状（N=512 以下で優位性なし・明確に劣後）は引き続き
//! 64×64 版のまま。`internal-diagnostics` feature 限定の診断入口
//! （`CudaGemm::new_with_tiled_pipeline_128x64`／
//! `CudaGemm::compile_tiled_pipeline_128x64_variant`）は変更前と同じ
//! 挙動を維持する。
//!
//! # `kernels_tiled_pipeline.rs` との差分
//!
//! | 項目 | 64×64（既存） | 128×64（本モジュール） |
//! |------|---------------|------------------------|
//! | ブロックタイル `BM×BN` | 64×64 | **128×64** |
//! | 1 スレッド担当（`THREAD_M×THREAD_N`） | 4×4 | **8×4** |
//! | ブロック内スレッド数 | 256（16×16） | 256（16×16。不変） |
//! | 共有メモリのバンク衝突対策 | 行幅パディング（`TP_A_PAD`/`TP_B_PAD`） | **A のみ XOR スウィズル（パディングなし）** |
//! | ステージ範囲 | 2〜4 | **2〜3**（後述「smem 予算」） |
//!
//! `THREAD_M` が 4→8 になったことで、A フラグメント読み（`ty*THREAD_M+i`）
//! が warp 内で 8 行離れた 2 グループへ分裂する（256 スレッド・16×16
//! 格子・warp=32 は `ty` 2 値 × `tx` 16 値から成るため）。A の行幅（`BK`=16
//! 要素=64 バイト）はパディングなしだと 8 行差のバイトオフセット差
//! （8×64B=512B）が常に 128B（32 バンク×4B）の倍数になり、**パディングでは
//! 解消できない 2-way バンク衝突**を生む（「なぜパディングでなく XOR か」
//! 節）。B は行幅 256 バイトで読み・書きとも連続 16 チャンクのため元々
//! 衝突しない（スウィズル不要）。
//!
//! # なぜパディングでなく XOR か
//!
//! A フラグメント読み `as_tile[stage][ty*8+i][kk]`（`i`=0..8）は、warp 内で
//! `ty` の 2 値（同一 warp 内の 32 レーンは `tx`=0..16 の 2 セット）に対応する
//! 2 つの行グループ（8 行差）へ分裂する。行幅を `w` 要素とすると、8 行差の
//! バイトオフセット差は `8*w*4` バイト。`w`=16（パディングなし）のとき
//! `8*16*4=512` バイトは 128 バイト（32 バンク×4B）の倍数であり、`w` を
//! どのようにパディングしても `8*w*4 mod 128 == 0` を崩すには `w` を 128B
//! グリッドからずらす必要があるが、`w` は cp.async 16 バイト（f32 4 要素）
//! 転送粒度の倍数でなければならず、この制約下でパディングのみでは 8 行差
//! バンク衝突を解消できない（`w` の候補はすべて 4 の倍数であり、4 の倍数
//! である限り `8*w*4` は常に 128 の倍数になる）。
//!
//! そこで **16 バイトチャンク（f32 4 要素）単位の XOR スウィズル**
//! `swz(row, chunk) = chunk ^ ((row >> 3) & 3)`（[`swizzled_chunk_a`]）を
//! 採用する。`row` と `row+8` は `(row>>3)&3` の値が必ず異なる（+8 は
//! `row>>3` を +1 するため）ため `swz(row,chunk) != swz(row+8,chunk)` が
//! 常に成立し、スウィズル後の物理アドレスは異なるバンクに写る（機械検査:
//! [`tests::a_fragment_swizzle_resolves_8_row_bank_conflict`]）。
//!
//! スウィズルは 16 バイトチャンク単位の**行内全単射**（[`swizzled_chunk_a`]
//! ドキュメンテーションコメント参照）であり、cp.async の「1 命令 = 16
//! バイト連続転送」という制約を壊さない（チャンク内部の 4 要素順序は
//! 変えず、チャンクの格納位置のみを並べ替える）。
//!
//! # bit 同一の論拠（`kernels_tiled_pipeline.rs` との出力比較の前提）
//!
//! 各出力要素は `acc=+0.0` から `kk`（K タイル内 0..`TP128_BK`）昇順・`t`
//! （K タイル番号）昇順の単一 `fmaf()` 連鎖で確定する。これは
//! `kernels_tiled_pipeline.rs` の 64×64 版・`kernels.rs::TILED_F32`
//! （classic 版）と同一の K 昇順単一アキュムレータ規約であり、ブロック
//! タイル `BM`/`THREAD_M` の値そのものは各出力要素の演算順序に影響しない
//! （縮約対象は常に全 K を `TP_BK`/`TP128_BK` 幅の K タイルで昇順に走査する
//! 単一の合計であり、どのスレッド・どのブロックが担当するかは K 方向の
//! 加算順序を変えない）。**A のスウィズルは共有メモリの物理格納位置のみを
//! 変える純粋なアドレス置換であり、読み出す値そのもの・読み出す順序（`kk`
//! 昇順）を変えない**ため、演算順序に対しても中立である。したがって
//! `TP128_BK`（=16。`TP_BK` と同値）を揃えている限り、本カーネルは
//! `kernels_tiled_pipeline.rs`（64×64 版）・`kernels.rs`（classic 版）の
//! いずれとも bit 完全一致する（cp.async zfill の末尾 `fmaf(0,0,acc)` は
//! RN 丸めで `acc` が `-0.0` に到達しない限り `acc` を変えないため bit
//! 中立。`kernels_tiled_pipeline.rs` 冒頭コメント「bit 同一の論拠」と同じ
//! 論拠）。実機での bit 一致自己検証は兄弟イシュー #1344 ではなく本イシュー
//! （#1343）のスコープであり、`tests/cpu_cuda_tiled_pipeline_parity.rs`
//! の `#[ignore]` テスト（T7/T8）が担う。
//!
//! **複数アキュムレータ・K 分割等の ILP 手法は本カーネルに適用しない**
//! （bit 同一を壊す。[`tests::tiled_pipeline_128x64_source_uses_single_accumulator`]
//! が単一 `acc` 宣言のみであることを機械検査する）。
//!
//! # 共有メモリ予算・ステージ範囲（GB10 occupancy 訂正を含む）
//!
//! パディングなしのため 1 ステージあたり `(BM*BK + BK*BN) * 4B` =
//! `(128*16 + 16*64) * 4` = 12,288 バイト（[`TP128_SMEM_BYTES_PER_STAGE`]）。
//! 4 段では 49,152 バイトとなり全 compute capability 共通の静的 48KiB
//! per-block 上限（[`crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`]）
//! ちょうどで余裕がなく、かつ #1137（`kernels_tiled_pipeline.rs`）の A/B
//! 実測で 4 段は GB10 で劣化することが確認済みのため、**本モジュールの
//! ステージ範囲は 2〜3（[`TP128_MIN_STAGES`]..=[`TP128_MAX_STAGES`]）に
//! 限定する**（3 段時点で 36,864 バイト。コンパイル時 assert で 48KiB 上限を
//! 検査する）。
//!
//! **GB10 occupancy 訂正（親イシュー #1342 想定「3 block/SM」の不成立）**:
//! `docs/perf/sm121-device-attributes.md` の GB10 実測
//! `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 102,400` バイトに対し、3 段の
//! 36,864 バイト × 3 block = 110,592 バイト（> 102,400）のため、GB10 では
//! **smem 制約により同時常駐は 2 block/SM（16 warp/SM）に留まる**（3
//! block/SM は smem 予算超過のため成立しない）。この訂正は本イシューの
//! 実装事実として `docs/perf/cuda-gemm-tiled-pipeline.md` に転記し、実測
//! （ptxas 資源値・実効 occupancy）による確認は #1344 の検証項目とする。
//!
//! # REQ-8（カーネル境界検査規約。省略しない）
//!
//! `kernels_tiled_pipeline.rs` と同一契約: A/B タイルの `cp.async` ロードは
//! 範囲外チャンクで `src_size = 0` を渡しゼロ充填し（列方向は f32 4 要素
//! 境界へ切り下げてクランプ。`n % 4 == 0 && k % 4 == 0` の整列制約により
//! チャンク単位の可否判定が要素単位の可否判定と一致する）、エピローグ
//! store は要素ごとに `if (r < m && cc < n)` の手動ガードを維持する
//! （`#pragma unroll` によるレジスタブロッキング展開は演算・分岐命令数を
//! 削減する最適化であり、境界チェックそのものは無効化しない）。

use std::sync::LazyLock;

use crate::error::CudaError;

/// ブロックタイル M（C の行方向。128。`kernels_tiled_pipeline::TP_BM`
/// の 2 倍）。
pub const TP128_BM: u32 = 128;
/// ブロックタイル N（C の列方向。64。`TP_BN` と同値）。
pub const TP128_BN: u32 = 64;
/// K タイル幅（16。`TP_BK` と同値。bit 同一の論拠の前提。冒頭コメント
/// 「bit 同一の論拠」参照）。
pub const TP128_BK: u32 = 16;

/// 1 スレッドが担当する C タイルの行数（8。`TP_THREAD_M`〈4〉の 2 倍）。
pub const TP128_THREAD_M: u32 = 8;
/// 1 スレッドが担当する C タイルの列数（4。`TP_THREAD_N` と同値）。
pub const TP128_THREAD_N: u32 = 4;

/// ブロック内スレッドグリッドの x 方向本数（`TP128_BN / TP128_THREAD_N`
/// = 16）。
pub const TP128_THREADS_X: u32 = TP128_BN / TP128_THREAD_N;
/// ブロック内スレッドグリッドの y 方向本数（`TP128_BM / TP128_THREAD_M`
/// = 16。`THREAD_M` が 2 倍でも `BM` も 2 倍のため 16 のまま不変）。
pub const TP128_THREADS_Y: u32 = TP128_BM / TP128_THREAD_M;
/// ブロックあたりスレッド総数（256。`TP_BLOCK_THREADS` と同値。
/// `kernels_tiled_pipeline::TP_BLOCK_THREADS` と同じ「1 次元ブロックとして
/// 起動し `tx = tid % THREADS_X`・`ty = tid / THREADS_X` で分解する」契約）。
pub const TP128_BLOCK_THREADS: u32 = TP128_THREADS_X * TP128_THREADS_Y;

/// `cp.async` 多段パイプラインの既定ステージ数（3。
/// `kernels_tiled_pipeline::TP_DEFAULT_STAGES` と同値）。
pub const TP128_DEFAULT_STAGES: u32 = 3;

/// パイプラインステージ数として受理する最小値（`cp.async.wait_group
/// STAGES-2` の u32 アンダーフロー防止。`kernels_tiled_pipeline::
/// TP_MIN_STAGES` と同一契約）。
pub const TP128_MIN_STAGES: u32 = 2;

/// PTX ISA の `cp.async.wait_group` 即値オペランドの上限（0〜7）。
const MAX_WAIT_GROUP_IMMEDIATE: u32 = 7;

/// パイプラインステージ数として受理する最大値（**3。
/// `kernels_tiled_pipeline::TP_MAX_STAGES`〈4〉と異なり本モジュールは
/// 4 を含まない**。冒頭コメント「共有メモリ予算・ステージ範囲」参照:
/// パディングなし・演算密度 2 倍の本タイル構成では 4 段が静的 48KiB
/// 上限ちょうどで余裕がなく、64×64 版の #1137 A/B 実測でも 4 段は GB10 で
/// 劣化することが確認済みのため 3 段までに絞る）。
pub const TP128_MAX_STAGES: u32 = 3;

/// 1 ステージあたりの A タイルロードチャンク数（16 バイト = f32 4 要素
/// 単位。`(BM * BK) / 4` = 512）。
pub const TP128_A_CHUNKS: u32 = (TP128_BM * TP128_BK) / 4;
/// 1 ステージあたりの B タイルロードチャンク数（`(BK * BN) / 4` = 256）。
pub const TP128_B_CHUNKS: u32 = (TP128_BK * TP128_BN) / 4;

/// A タイルの 1 行あたりチャンク数（`TP128_BK / 4` = 4）。
/// [`swizzled_chunk_a`] のマスク（`&3`）が全単射であるための前提
/// （`tests::a_swizzle_matches_cuda_macro_definition` が本値と CUDA 側
/// マクロの `& 3` を同期検査する）。
pub const TP128_A_CHUNKS_PER_ROW: u32 = TP128_BK / 4;

/// ステージあたりの静的共有メモリ使用量（バイト）。パディングなしのため
/// `(BM*BK + BK*BN) * 4B`（`kernels_tiled_pipeline::TP_SMEM_BYTES_PER_STAGE`
/// と異なりパディング項を含まない。冒頭コメント「共有メモリ予算」参照）。
pub const TP128_SMEM_BYTES_PER_STAGE: u32 = (TP128_BM * TP128_BK + TP128_BK * TP128_BN) * 4;

// コンパイル時契約検査（`kernels_tiled_pipeline.rs` 冒頭の const assert 群
// と同型。実機コンパイルできない環境でも `cargo build` の時点で機械検出
// できる代替チェック）。
const _: () = assert!(
    TP128_BM.is_multiple_of(TP128_THREAD_M),
    "TP128_BM must be a multiple of TP128_THREAD_M (per-thread register-blocked \
     output tile must evenly divide the block tile)"
);
const _: () = assert!(
    TP128_BN.is_multiple_of(TP128_THREAD_N),
    "TP128_BN must be a multiple of TP128_THREAD_N (per-thread register-blocked \
     output tile must evenly divide the block tile)"
);
const _: () = assert!(
    TP128_BLOCK_THREADS <= 1024,
    "TP128_BLOCK_THREADS must not exceed CUDA's per-block thread limit (1024)"
);
const _: () = assert!(
    TP128_BM.is_multiple_of(4) && TP128_BN.is_multiple_of(4) && TP128_BK.is_multiple_of(4),
    "TP128_BM/TP128_BN/TP128_BK must be multiples of 4 (cp.async 16-byte / f32 \
     4-element transfer granularity)"
);
const _: () = assert!(
    TP128_A_CHUNKS * 4 == TP128_BM * TP128_BK && TP128_B_CHUNKS * 4 == TP128_BK * TP128_BN,
    "TP128_BM*TP128_BK / TP128_BK*TP128_BN must be exact multiples of 4 (each \
     cp.async chunk transfers exactly 4 f32 elements; TP128_A_CHUNKS/TP128_B_CHUNKS \
     must not truncate)"
);
const _: () = assert!(
    TP128_A_CHUNKS_PER_ROW == 4,
    "swizzled_chunk_a()（および CUDA 側 TP128_SWZ_A マクロ）の `& 3` マスクは \
     A タイル 1 行あたりのチャンク数が厳密に 4（TP128_BK == 16）であることを \
     前提とする全単射性の根拠のため、TP128_BK は 16 固定とする"
);
const _: () = assert!(
    TP128_MIN_STAGES >= 2,
    "kernels_tiled_pipeline_128x64 の cp.async パイプラインは STAGES >= 2 を \
     前提とする（カーネルソース側の `STAGES - 2` 計算が u32 でアンダーフロー \
     しないため）"
);
const _: () = assert!(
    TP128_MAX_STAGES >= TP128_MIN_STAGES && TP128_MAX_STAGES <= MAX_WAIT_GROUP_IMMEDIATE + 2,
    "TP128_MAX_STAGES must fit the cp.async.wait_group immediate operand range \
     (STAGES - 2 must be in [0, 7])"
);
const _: () = assert!(
    TP128_DEFAULT_STAGES >= TP128_MIN_STAGES && TP128_DEFAULT_STAGES <= TP128_MAX_STAGES,
    "TP128_DEFAULT_STAGES must lie within [TP128_MIN_STAGES, TP128_MAX_STAGES]"
);
// 静的共有メモリ予算（全 compute capability 共通の per-block 48KiB）は
// 最悪ケース（TP128_MAX_STAGES=3）でも超過しないことをコンパイル時に検査
// する（冒頭コメント「共有メモリ予算・ステージ範囲」参照。段数を増やす
// ほど所要量は単調増加するため、この 1 点の検査で
// TP128_MIN_STAGES..=TP128_MAX_STAGES の全段数を保証できる）。
const _: () = assert!(
    TP128_SMEM_BYTES_PER_STAGE * TP128_MAX_STAGES
        <= crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES,
    "kernels_tiled_pipeline_128x64 static shared memory (at TP128_MAX_STAGES) \
     exceeds the 48KiB per-block limit shared by every compute capability"
);

/// A フラグメントの smem 格納位置スウィズルの**ホスト側参照実装**
/// （純関数。CUDA 側マクロ `TP128_SWZ_A` と同一設計を Rust で独立再実装
/// したもの。`transpose.rs::swizzled_smem_col` と同じ「単一真実源の関係に
/// はなく needle テストが不一致を機械検出する」位置づけ）。
///
/// `row`（A タイル内の行。0..[`TP128_BM`]）に応じて `chunk`（16 バイト
/// チャンク番号。0..[`TP128_A_CHUNKS_PER_ROW`]=4）を XOR で並べ替える。
/// `(row >> 3) & 3` は `row` を 8 行区切りでグループ化し、そのグループ
/// 番号（0..3）を鍵として `chunk` の下位 2 ビットへ XOR する。XOR は
/// 対合（involution）のため、固定した `row` に対する `chunk ->
/// swizzled_chunk` の写像は `0..TP128_A_CHUNKS_PER_ROW` 上で全単射になる
/// （鍵・値とも 2 ビット幅で閉じているため。機械検査:
/// [`tests::a_swizzle_is_bijective_per_row`]）。
///
/// `row` と `row+8` は `(row>>3)&3` の値が必ず異なるため（+8 は
/// `row>>3` を厳密に +1 する）、この 2 行に対する `swz` の戻り値は常に
/// 異なる。これが「なぜパディングでなく XOR か」（モジュール冒頭
/// コメント）で導出した 8 行差バンク衝突解消の根拠であり、
/// [`tests::a_fragment_swizzle_resolves_8_row_bank_conflict`] が全
/// `row`×`chunk` 組合せで機械検査する。
///
/// # 呼び出し文脈（生産経路からは到達しない参照実装）
///
/// 本関数は [`TILED_PIPELINE_128X64_F32_BODY`] が生成する CUDA C++ 側の
/// `TP128_SWZ_A` マクロと同一設計を Rust で独立に再実装したものであり、
/// GPU 実行経路からは直接呼ばれない（`transpose.rs::swizzled_smem_col`
/// 冒頭コメントと同じ位置づけ）。呼び出し元は本ファイル内の
/// `#[cfg(test)]` テスト（ホストモデル bit 一致・bank 衝突モデル）のみ。
///
/// `#[allow(dead_code)]`: 呼び出し元は本ファイル末尾の `#[cfg(test)]`
/// テストのみであり、`cargo build`（テスト cfg なし）では未使用と判定
/// される（`transpose.rs::swizzled_smem_col` の `#[allow(dead_code)]` と
/// 同じ判断パターン）。
#[allow(dead_code)]
pub(crate) fn swizzled_chunk_a(row: u32, chunk: u32) -> u32 {
    (chunk ^ ((row >> 3) & 3)) & (TP128_A_CHUNKS_PER_ROW - 1)
}

/// 本番結線（[`crate::gemm::CudaGemm::new_with_tiled_pipeline_128x64`]。
/// `internal-diagnostics` feature 限定の診断入口）が既定でコンパイルする
/// ステージ数（[`TP128_DEFAULT_STAGES`]）固定のカーネルソース。
///
/// カーネルソースはコンパイル時定数のみから `format!` で組み立て、外部
/// 入力文字列を連結しない（`nvrtc.rs` A03 節と同じ契約。
/// `.claude/rules/security.md` A03）。
pub fn tiled_pipeline_128x64_f32_source() -> &'static str {
    &TILED_PIPELINE_128X64_F32_SOURCE
}

static TILED_PIPELINE_128X64_F32_SOURCE: LazyLock<String> =
    LazyLock::new(|| render_source(TP128_DEFAULT_STAGES));

/// 任意のステージ数（[`TP128_MIN_STAGES`]..=[`TP128_MAX_STAGES`]）の
/// カーネルソースを生成する（`kernels_tiled_pipeline::
/// tiled_pipeline_f32_source_with_stages` と同じ位置づけ。
/// `examples/gemm_tiled_pipeline_bench.rs` が段数比較に使う）。
pub fn tiled_pipeline_128x64_f32_source_with_stages(stages: u32) -> Result<String, CudaError> {
    if !(TP128_MIN_STAGES..=TP128_MAX_STAGES).contains(&stages) {
        return Err(CudaError::InvalidKernelConfig {
            detail: format!(
                "tiled_pipeline_128x64_f32_source_with_stages stages ({stages}) must lie \
                 within [{TP128_MIN_STAGES}, {TP128_MAX_STAGES}]"
            ),
        });
    }
    Ok(render_source(stages))
}

fn render_source(stages: u32) -> String {
    format!(
        "\n#define TP128_BM {bm}\n\
         #define TP128_BN {bn}\n\
         #define TP128_BK {bk}\n\
         #define TP128_THREAD_M {thread_m}\n\
         #define TP128_THREAD_N {thread_n}\n\
         #define TP128_THREADS_X {threads_x}\n\
         #define TP128_STAGES {stages}\n\
         \n{body}",
        bm = TP128_BM,
        bn = TP128_BN,
        bk = TP128_BK,
        thread_m = TP128_THREAD_M,
        thread_n = TP128_THREAD_N,
        threads_x = TP128_THREADS_X,
        stages = stages,
        body = TILED_PIPELINE_128X64_F32_BODY,
    )
}

/// [`render_source`] が結合するカーネル本体テンプレート。
///
/// `TP128_STAGES` は `format!` で埋め込まれる `#define` のみに依存し、
/// 本体文字列自体はステージ数に非依存（配列サイズ・`STAGES - 2` 等の
/// 算術はすべて `TP128_STAGES` マクロ経由。`kernels_tiled_pipeline.rs::
/// TILED_PIPELINE_F32_BODY` と同型）。
const TILED_PIPELINE_128X64_F32_BODY: &str = r#"
// REQ-8: グローバル→共有メモリの 16 バイト単位（f32 4 要素）非同期
// コピー。src_size==16 で実データをコピーし、src_size==0 で共有メモリ側を
// ゼロ充填する（kernels_tiled_pipeline.rs::tp_cp_async16 と同じ契約・
// 同じ PTX 命令。関数名は同一 NVRTC コンパイル単位内での衝突を避けるため
// 本カーネル専用の接頭辞を付す）。
__device__ __forceinline__ void tp128_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_size)
{
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "r"(src_size)
    );
}

// A フラグメントの smem 格納位置スウィズル（16 バイトチャンク単位の
// 行内全単射。モジュール冒頭コメント「なぜパディングでなく XOR か」・
// Rust 側参照実装 `swizzled_chunk_a` 参照）。`& 3` は TP128_BK==16 の下で
// 1 行あたり厳密に 4 チャンクであることに依存する（Rust 側 const assert
// `TP128_A_CHUNKS_PER_ROW == 4` が前提を固定する）。
#define TP128_SWZ_A(row, chunk) (((chunk) ^ (((row) >> 3) & 3)) & 3)

extern "C" __global__ void gemm_tiled_pipeline_128x64_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k)
{
    // パディングなし（冒頭コメント「共有メモリ予算」参照）。A は
    // TP128_SWZ_A によるチャンク単位の並べ替えのみでバンク衝突を回避する
    // ため、行幅そのものは BK（16 要素=64 バイト）で確保する。B は
    // 元々連続 16 チャンクで衝突しないためスウィズル不要。
    __shared__ __align__(16) float as_tile[TP128_STAGES][TP128_BM][TP128_BK];
    __shared__ __align__(16) float bs_tile[TP128_STAGES][TP128_BK][TP128_BN];

    int block_row0 = blockIdx.y * TP128_BM;
    int block_col0 = blockIdx.x * TP128_BN;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int tx = tid % TP128_THREADS_X;
    int ty = tid / TP128_THREADS_X;

    int thread_row0 = block_row0 + ty * TP128_THREAD_M;
    int thread_col0 = block_col0 + tx * TP128_THREAD_N;

    // 単一アキュムレータ（複数アキュムレータ・K 分割等の ILP 手法は
    // bit 同一を壊すため使わない。モジュール冒頭コメント「bit 同一の
    // 論拠」参照）。
    float acc[TP128_THREAD_M][TP128_THREAD_N] = {};

    int num_k_tiles = (k > 0) ? (k - 1) / TP128_BK + 1 : 0;

    #define A_CHUNKS ((TP128_BM * TP128_BK) / 4)
    #define B_CHUNKS ((TP128_BK * TP128_BN) / 4)
    #define A_CHUNKS_PER_ROW (TP128_BK / 4)

    // REQ-8: 境界外チャンクでも 16 バイト整列を保ったままクランプする
    // （列方向は f32 4 要素境界へ切り下げ。`gemm.rs` 側の起動前整列検証
    // 〈n%4==0 && k%4==0〉と合わせて行ストライドの 4 要素倍数性を保証
    // する。`kernels_tiled_pipeline.rs::LOAD_A_STAGE` と同一式に加え、
    // 格納先アドレスのみ TP128_SWZ_A でチャンク単位に並べ替える）。
    #define LOAD_A_STAGE(stage, k0) \
        for (int idx = tid; idx < A_CHUNKS; idx += num_threads) { \
            int row = idx / A_CHUNKS_PER_ROW; \
            int chunk = idx % A_CHUNKS_PER_ROW; \
            int col0 = chunk * 4; \
            int gr = block_row0 + row; \
            int gc = (k0) + col0; \
            int gr_c = gr < m ? gr : (m > 0 ? m - 1 : 0); \
            int gc_c = gc < k ? gc : (k > 0 ? ((k - 1) / 4) * 4 : 0); \
            int valid = (gr < m && gc < k) ? 16 : 0; \
            int swz_chunk = TP128_SWZ_A(row, chunk); \
            tp128_cp_async16(&as_tile[stage][row][swz_chunk * 4], &a[(size_t)gr_c * k + gc_c], valid); \
        }

    #define LOAD_B_STAGE(stage, k0) \
        for (int idx = tid; idx < B_CHUNKS; idx += num_threads) { \
            int row = idx / (TP128_BN / 4); \
            int col0 = (idx % (TP128_BN / 4)) * 4; \
            int gr = (k0) + row; \
            int gc = block_col0 + col0; \
            int gr_c = gr < k ? gr : (k > 0 ? k - 1 : 0); \
            int gc_c = gc < n ? gc : (n > 0 ? ((n - 1) / 4) * 4 : 0); \
            int valid = (gr < k && gc < n) ? 16 : 0; \
            tp128_cp_async16(&bs_tile[stage][row][col0], &b[(size_t)gr_c * n + gc_c], valid); \
        }

    // プロローグ: kernels_tiled_pipeline.rs::TILED_PIPELINE_F32_BODY
    // プロローグと同一の「1 イテレーション = 必ず 1 commit」不変条件。
    for (int s = 0; s < TP128_STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            LOAD_A_STAGE(s, s * TP128_BK);
            LOAD_B_STAGE(s, s * TP128_BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < num_k_tiles; ++t) {
        int compute_stage = t % TP128_STAGES;
        int next_tile = t + TP128_STAGES - 1;
        int load_stage = next_tile % TP128_STAGES;

        // kernels_tiled_pipeline.rs と同一の段数一般形固定即値
        // （`STAGES - 2`）・同一の正しさ論証（非負性は上記
        // `TP128_MIN_STAGES >= 2` のコンパイル時 assert が担保する）。
        asm volatile("cp.async.wait_group %0;\n" ::"n"(TP128_STAGES - 2));
        __syncthreads();

        // compute_stage の共有メモリタイルを使い、TP128_THREAD_M x
        // TP128_THREAD_N の外積型レジスタブロッキングで積和する。A の
        // 読み出しは TP128_SWZ_A で書き込みと同一のチャンク並べ替えを
        // 逆算し、読み書きが一致した物理アドレスへアクセスする（値・
        // 演算順序自体はスウィズルの影響を受けない。モジュール冒頭
        // コメント「bit 同一の論拠」参照）。CPU 参照実装
        // （f32::mul_add）と同じ「明示的な融合積和」契約を保つため
        // fmaf() を使う。
#pragma unroll
        for (int kk = 0; kk < TP128_BK; ++kk) {
            int chunk = kk / 4;
            int elem = kk % 4;
            float a_reg[TP128_THREAD_M];
#pragma unroll
            for (int i = 0; i < TP128_THREAD_M; ++i) {
                int row = ty * TP128_THREAD_M + i;
                int swz_chunk = TP128_SWZ_A(row, chunk);
                a_reg[i] = as_tile[compute_stage][row][swz_chunk * 4 + elem];
            }
            float b_reg[TP128_THREAD_N];
#pragma unroll
            for (int j = 0; j < TP128_THREAD_N; ++j) {
                b_reg[j] = bs_tile[compute_stage][kk][tx * TP128_THREAD_N + j];
            }
#pragma unroll
            for (int i = 0; i < TP128_THREAD_M; ++i) {
#pragma unroll
                for (int j = 0; j < TP128_THREAD_N; ++j) {
                    acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        // 次タイル（load_stage）の cp.async 発行は本イテレーションの
        // compute_stage 読み取りの後に置く（`kernels_tiled_pipeline.rs`
        // と同一の WAR 安全性の論証。load_stage != compute_stage は
        // TP128_STAGES >= 2 のため常に成立する）。
        if (next_tile < num_k_tiles) {
            LOAD_A_STAGE(load_stage, next_tile * TP128_BK);
            LOAD_B_STAGE(load_stage, next_tile * TP128_BK);
        }

        // kernels_tiled_pipeline.rs と同一の「1 イテレーション = 必ず 1
        // commit」不変条件、および同一の syncthreads 配置。
        asm volatile("cp.async.commit_group;\n");
        __syncthreads();
    }

    // ループ外 drain（kernels_tiled_pipeline.rs と同一の正しさ論証）。
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    #undef LOAD_A_STAGE
    #undef LOAD_B_STAGE
    #undef A_CHUNKS
    #undef B_CHUNKS
    #undef A_CHUNKS_PER_ROW

    // REQ-8: エピローグの guarded store。`#pragma unroll` によるループ
    // 展開は演算・分岐命令数を削減する最適化であり、境界チェックそのもの
    // は無効化しない。
#pragma unroll
    for (int i = 0; i < TP128_THREAD_M; ++i) {
#pragma unroll
        for (int j = 0; j < TP128_THREAD_N; ++j) {
            int r = thread_row0 + i;
            int cc = thread_col0 + j;
            if (r < m && cc < n) {
                c[(size_t)r * n + cc] = acc[i][j];
            }
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// カーネルソースが `cp.async` の主要命令・スウィズルマクロを含み、
    /// A のロード側（`LOAD_A_STAGE` 内）・読み側（アキュムレータループ内）
    /// の両方から `TP128_SWZ_A(` を実際に使っていることを検査する
    /// （`kernels_tiled_pipeline.rs::tiled_pipeline_source_uses_cp_async_instructions`
    /// と同型の静的検査）。
    #[test]
    fn tiled_pipeline_128x64_source_uses_cp_async_and_swizzle() {
        let source = tiled_pipeline_128x64_f32_source();
        for needle in [
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "fmaf(",
            "#define TP128_SWZ_A(row, chunk)",
        ] {
            assert!(
                source.contains(needle),
                "tiled_pipeline_128x64_f32_source() が `{needle}` を含みません"
            );
        }
        let swz_use_count = source.matches("TP128_SWZ_A(").count();
        // マクロ定義自体（1 箇所）+ LOAD_A_STAGE 内の呼び出し（1 箇所）+
        // アキュムレータループ内の呼び出し（1 箇所）= 3 箇所。
        assert_eq!(
            swz_use_count, 3,
            "TP128_SWZ_A はマクロ定義・ロード側・読み側の 3 箇所で使われる必要があります \
             （実際: {swz_use_count} 箇所）"
        );
    }

    /// REQ-8 の手動境界検査（cp.async src_size ゼロ充填・エピローグ
    /// guarded store）がソースから省略されていないことを検査する
    /// （`kernels_tiled_pipeline.rs::tiled_pipeline_source_retains_manual_bounds_checks`
    /// と同型）。
    #[test]
    fn tiled_pipeline_128x64_source_retains_manual_bounds_checks() {
        let source = tiled_pipeline_128x64_f32_source();
        assert!(
            source.contains("int valid = (gr < m && gc < k) ? 16 : 0;"),
            "A タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("int valid = (gr < k && gc < n) ? 16 : 0;"),
            "B タイルロードの guarded cp.async（src_size ゼロ充填）が見当たりません"
        );
        assert!(
            source.contains("if (r < m && cc < n) {"),
            "エピローグの guarded store が見当たりません"
        );
    }

    /// 単一アキュムレータ（`acc[TP128_THREAD_M][TP128_THREAD_N]` が 1
    /// 宣言のみ）を機械検査する。複数アキュムレータ・K 分割等の ILP
    /// 手法は bit 同一を壊すため禁止する（モジュール冒頭コメント「bit
    /// 同一の論拠」参照）。
    #[test]
    fn tiled_pipeline_128x64_source_uses_single_accumulator() {
        let source = tiled_pipeline_128x64_f32_source();
        let acc_decl_count = source
            .matches("float acc[TP128_THREAD_M][TP128_THREAD_N]")
            .count();
        assert_eq!(
            acc_decl_count, 1,
            "アキュムレータ宣言は 1 箇所のみである必要があります（実際: {acc_decl_count} 箇所）"
        );
        assert!(
            !source.contains("acc2"),
            "分割アキュムレータ（acc2 等）の痕跡が見つかりました"
        );
    }

    /// Rust 側の唯一の真実源（`TP128_BM`/`TP128_BN`/`TP128_BK`/
    /// `TP128_THREAD_M`/`TP128_THREAD_N`/`TP128_THREADS_X`/既定
    /// `TP128_STAGES`）が生成済みカーネルソース内の `#define` と食い違わ
    /// ないことを検査する（`kernels_tiled_pipeline.rs::
    /// tiled_pipeline_constants_match_kernel_source_defines` と同型）。
    #[test]
    fn tiled_pipeline_128x64_constants_match_kernel_source_defines() {
        let source = tiled_pipeline_128x64_f32_source();
        let checks: [(&str, u32); 7] = [
            ("TP128_BM", TP128_BM),
            ("TP128_BN", TP128_BN),
            ("TP128_BK", TP128_BK),
            ("TP128_THREAD_M", TP128_THREAD_M),
            ("TP128_THREAD_N", TP128_THREAD_N),
            ("TP128_THREADS_X", TP128_THREADS_X),
            ("TP128_STAGES", TP128_DEFAULT_STAGES),
        ];
        for (name, value) in checks {
            let expected = format!("#define {name} {value}");
            assert!(
                source.contains(&expected),
                "tiled_pipeline_128x64_f32_source() の `#define {name}` が Rust 側の \
                 定数（{value}）と一致しません"
            );
        }
    }

    /// [`tiled_pipeline_128x64_f32_source_with_stages`] の範囲検証（2〜3）
    /// を検査する（`TP128_MAX_STAGES`=3 が `kernels_tiled_pipeline::
    /// TP_MAX_STAGES`=4 と異なることの回帰確認を兼ねる）。
    #[test]
    fn tiled_pipeline_128x64_source_with_stages_validates_range() {
        assert!(tiled_pipeline_128x64_f32_source_with_stages(1).is_err());
        assert!(tiled_pipeline_128x64_f32_source_with_stages(TP128_MAX_STAGES + 1).is_err());
        for stages in TP128_MIN_STAGES..=TP128_MAX_STAGES {
            let src = tiled_pipeline_128x64_f32_source_with_stages(stages)
                .unwrap_or_else(|e| panic!("stages={stages} must be accepted: {e}"));
            assert!(src.contains(&format!("#define TP128_STAGES {stages}")));
        }
    }

    /// `cp.async.commit_group` がプロローグ・本体ループ末尾の 2 箇所のみ
    /// から発行され、`wait_group` が本体ループ・drain の 2 箇所にのみ
    /// 現れることを検査する（`kernels_tiled_pipeline.rs::
    /// tiled_pipeline_commit_wait_group_counts` と同型）。
    #[test]
    fn tiled_pipeline_128x64_commit_wait_group_counts() {
        let source = tiled_pipeline_128x64_f32_source();
        let commit_count = source.matches("cp.async.commit_group;").count();
        let wait_count = source.matches("cp.async.wait_group").count();
        assert_eq!(
            commit_count, 2,
            "commit_group は 2 箇所（prologue・本体末尾）"
        );
        assert_eq!(wait_count, 2, "wait_group は 2 箇所（本体ループ・drain）");
    }

    /// [`swizzled_chunk_a`] が固定 `row` に対して全単射であることを、
    /// `TP128_BM` 全域（0..128）× `chunk`（0..4）の全組合せで機械検査する
    /// （`transpose.rs::swizzle_is_bijective_per_row` と同型）。
    #[test]
    fn a_swizzle_is_bijective_per_row() {
        for row in 0..TP128_BM {
            let mut seen: HashSet<u32> = HashSet::with_capacity(TP128_A_CHUNKS_PER_ROW as usize);
            for chunk in 0..TP128_A_CHUNKS_PER_ROW {
                let sc = swizzled_chunk_a(row, chunk);
                assert!(
                    sc < TP128_A_CHUNKS_PER_ROW,
                    "row={row} chunk={chunk}: swizzled_chunk={sc} が範囲外です"
                );
                assert!(
                    seen.insert(sc),
                    "row={row} chunk={chunk}: swizzled_chunk={sc} が重複しています \
                     （全単射性違反）"
                );
            }
        }
    }

    /// Rust 側 [`swizzled_chunk_a`] と CUDA 側 `TP128_SWZ_A` マクロが
    /// 同一設計（`chunk ^ ((row >> 3) & 3)`）であることを、生成済み
    /// ソース文字列の定義と付き合わせて検査する（`transpose.rs` 冒頭
    /// コメント「単一真実源の関係にはない」ため needle テストで同期する
    /// 契約）。
    #[test]
    fn a_swizzle_matches_cuda_macro_definition() {
        let source = tiled_pipeline_128x64_f32_source();
        assert!(
            source.contains("#define TP128_SWZ_A(row, chunk) (((chunk) ^ (((row) >> 3) & 3)) & 3)"),
            "CUDA 側 TP128_SWZ_A マクロの定義文字列が Rust 側 swizzled_chunk_a() \
             の設計と一致しません（needle 不一致）"
        );
    }

    /// モジュール冒頭コメント「なぜパディングでなく XOR か」の論拠を
    /// 機械検査する: A の行幅（パディングなし・[`TP128_BK`]=16 要素）では
    /// warp 内で 8 行離れる 2 グループ（`ty`/`ty+1` に対応する
    /// `row`/`row+8`）が、スウィズルなしでは常に同一バンクへ衝突し
    /// （行ストライド×8＝128 バイト＝32 バンク×4B の倍数）、
    /// [`swizzled_chunk_a`] を適用すると常に異なるバンクへ写ることを、
    /// 全 `row`（0..120）× `chunk`（0..4）組合せで確認する（設計時に導出
    /// した論拠の実装への固定化。§「なぜパディングでなく XOR か」参照）。
    #[test]
    fn a_fragment_swizzle_resolves_8_row_bank_conflict() {
        const ROW_STRIDE_ELEMENTS: u32 = TP128_BK; // パディングなし。
        const BANKS: u32 = 32; // 32 バンク × 4 バイト/バンク（f32 1 要素）。
        for row in 0..(TP128_BM - 8) {
            for chunk in 0..TP128_A_CHUNKS_PER_ROW {
                let raw_bank = |r: u32| (r * ROW_STRIDE_ELEMENTS + chunk * 4) % BANKS;
                let swz_bank =
                    |r: u32| (r * ROW_STRIDE_ELEMENTS + swizzled_chunk_a(r, chunk) * 4) % BANKS;
                assert_eq!(
                    raw_bank(row),
                    raw_bank(row + 8),
                    "raw layout must exhibit the 8-row bank collision this swizzle fixes \
                     (row={row}, chunk={chunk})"
                );
                assert_ne!(
                    swz_bank(row),
                    swz_bank(row + 8),
                    "swizzled layout must resolve the 8-row bank collision \
                     (row={row}, chunk={chunk})"
                );
            }
        }
    }

    /// カーネルの索引・zfill・8×4 累積アルゴリズムを Rust でホスト側
    /// モデル化し（A のスウィズルは物理格納位置のみを変える純粋な
    /// アドレス置換で演算順序に影響しないため、モデル化では省略できる。
    /// モジュール冒頭コメント「bit 同一の論拠」参照）、
    /// `fandhe_ai_backend_cpu::matmul_reference_fma` と bit 完全一致する
    /// ことを整列形状（`n % 4 == 0 && k % 4 == 0`）のランダムスイープで
    /// 確認する（GPU 不要。実機なしでもカーネル設計の索引バグを検出
    /// できる独立検証）。
    #[test]
    fn host_model_matches_reference_fma_bit_exact() {
        let shapes: &[(u32, u32, u32)] = &[
            (1, 4, 4),
            (3, 8, 12),
            (64, 64, 16),
            (65, 60, 20),
            (127, 64, 36),
            (128, 64, 64),
            (129, 68, 4),
            (192, 132, 64),
            (200, 64, 64),
            (256, 256, 256),
        ];
        for (idx, &(m, n, k)) in shapes.iter().enumerate() {
            let mut rng = bench_harness::rng::Xorshift64Star::new(0x1343_0000 + idx as u64);
            let a: Vec<f32> = (0..(m as usize) * (k as usize))
                .map(|_| rng.next_f32())
                .collect();
            let b: Vec<f32> = (0..(k as usize) * (n as usize))
                .map(|_| rng.next_f32())
                .collect();

            let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
            fandhe_ai_backend_cpu::matmul_reference_fma(
                &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
            )
            .expect("matmul_reference_fma shape validation must pass for well-formed input");

            let c_model = host_model_gemm(&a, &b, m, n, k);
            assert_eq!(
                c_model, c_ref,
                "host_model_gemm と matmul_reference_fma が bit 一致しません（m={m} n={n} k={k}）"
            );
        }
    }

    /// [`host_model_matches_reference_fma_bit_exact`] が呼ぶ、カーネルの
    /// タイル・K タイル・8×4 レジスタブロッキング・zfill 境界クランプを
    /// そのまま Rust で再現したホスト側モデル（`u32` 演算は CUDA 側の
    /// `int` 索引算術と同じ意味論。全形状は本テストの呼び出し元が
    /// 4 の倍数 n/k のみを渡す契約）。
    fn host_model_gemm(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
        let mut c = vec![0.0f32; (m as usize) * (n as usize)];
        let num_k_tiles = if k > 0 { (k - 1) / TP128_BK + 1 } else { 0 };
        // ブロック数はカーネルの grid 分解（`gemm.rs::
        // tiled_pipeline_launch_config`）と同じ `div_ceil`。本テストの
        // 呼び出し元は m > 0 のみを渡すため 0 除算は起こらない。
        let block_rows = m.div_ceil(TP128_BM);
        let block_cols = n.div_ceil(TP128_BN);

        for by in 0..block_rows {
            for bx in 0..block_cols {
                let block_row0 = by * TP128_BM;
                let block_col0 = bx * TP128_BN;
                for ty in 0..TP128_THREADS_Y {
                    for tx in 0..TP128_THREADS_X {
                        let mut acc = [[0.0f32; TP128_THREAD_N as usize]; TP128_THREAD_M as usize];
                        for t in 0..num_k_tiles {
                            let k0 = t * TP128_BK;
                            for kk in 0..TP128_BK {
                                let gk = k0 + kk;
                                let mut a_reg = [0.0f32; TP128_THREAD_M as usize];
                                for i in 0..TP128_THREAD_M {
                                    let row = block_row0 + ty * TP128_THREAD_M + i;
                                    a_reg[i as usize] = if row < m && gk < k {
                                        a[(row as usize) * (k as usize) + gk as usize]
                                    } else {
                                        0.0
                                    };
                                }
                                let mut b_reg = [0.0f32; TP128_THREAD_N as usize];
                                for j in 0..TP128_THREAD_N {
                                    let col = block_col0 + tx * TP128_THREAD_N + j;
                                    b_reg[j as usize] = if gk < k && col < n {
                                        b[(gk as usize) * (n as usize) + col as usize]
                                    } else {
                                        0.0
                                    };
                                }
                                for i in 0..TP128_THREAD_M as usize {
                                    for j in 0..TP128_THREAD_N as usize {
                                        acc[i][j] = a_reg[i].mul_add(b_reg[j], acc[i][j]);
                                    }
                                }
                            }
                        }
                        for i in 0..TP128_THREAD_M {
                            for j in 0..TP128_THREAD_N {
                                let r = block_row0 + ty * TP128_THREAD_M + i;
                                let cc = block_col0 + tx * TP128_THREAD_N + j;
                                if r < m && cc < n {
                                    c[(r as usize) * (n as usize) + cc as usize] =
                                        acc[i as usize][j as usize];
                                }
                            }
                        }
                    }
                }
            }
        }
        c
    }
}
