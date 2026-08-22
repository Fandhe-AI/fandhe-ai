//! タイル→SM 割り当てスウィズル（イシュー #499）のホスト側純関数群。
//!
//! GEMM ブロックグリッド（`kernels_mma::MMA_F16` の `blockIdx.y x blockIdx.x`
//! = `num_m_blocks x num_n_blocks`）を、CUDA が既定で割り当てる線形順序
//! （`blockIdx.y * gridDim.x + blockIdx.x`。行優先で N 方向を先に辿る）から
//! 「M 方向を `group_width` 個ずつグルーピングし、各グループ内で N を先に
//! 全走査してから次の M グループへ移る」順序へ並べ替える。同一グループ内の
//! `group_width` 個の SM ブロックが同じ B タイル（N 位置固定）を短時間の
//! うちに読むため、L2 キャッシュ上の B タイル再利用が高まる（DeepGEMM
//! `get_swizzled_block_idx` 同型。MLX にも shift/mask 式の同種機構が
//! あるが classic 経路では無効化されたままで未実証。実装計画 1 節「背景・
//! 目的」参照）。
//!
//! 本モジュールはホスト側の**参照実装**であり、`kernels_mma.rs::
//! mma_f16_source_with_swizzle` が生成する CUDA C++ 側の同一整数式と
//! 単一真実源の関係にはない（カーネル側は独立した文字列として保持する
//! ため、二重管理の不一致はテスト（`kernels_mma.rs` のソース検査・
//! `gemm_mma.rs` の実機 bit 一致テスト）で機械検出する）。実際の GPU
//! 実行（NVRTC コンパイル・カーネル起動）へは、本モジュールの関数自体は
//! 到達しない（[`swizzled_block_idx`] は本ファイルの単体テストのみが
//! 呼ぶ参照実装）。
//!
//! [`select_swizzle_group_width`] はイシュー #740 で一時
//! `gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ）へ本番結線
//! したが、PR #758 レビュー指摘（採用基準未達・事前確認未実施・本
//! モジュールが依拠する SM 数入力の誤り）により差し戻された。イシュー
//! #775 で 2026-08-20 GB10 実機再計測（4096: base 34.4089 → swizzle(動的
//! g8) 54.3055 TFLOPS・×1.578。512/1024/2048 は ×0.979〜0.992）を根拠に、
//! **サイズ条件付き適用**（[`should_apply_swizzle`]。総タイル数
//! `num_m_blocks * num_n_blocks >= SWIZZLE_APPLY_TILE_COUNT_THRESHOLD` の
//! 場合のみ）ロジックを実装したが、結線前必須確認〈レジスタスピル・bit
//! 一致・parity 非後退・実測〉が GB10 実機到達可能なセッションで未実施
//! だったため、当初は実機検証専用の opt-in 入口
//! `gemm_mma.rs::CudaMmaGemm::new_with_size_conditional_swizzle`
//! （`internal-diagnostics` feature 限定）からのみ `launch_f16` へ到達
//! するよう限定していた。イシュー #782 で 2026-08-21 の GB10 実機再計測
//! （A/B 実測・bit 一致の 2 項目を解消。parity 非後退・結線後
//! `cuda_floor_bench` 実測・レジスタスピル確認は当初「マージ後確認可」の
//! 未解消事項として残っていたが、PR #784 codex-review 指摘への対応として
//! 結線済みコード自身に対するマージ前検証（2026-08-21・DGX Spark GB10
//! 実機）で全項目解消済み。`docs/perf/cuda-gemm-swizzle-ab.md` §6.3 参照）を
//! 根拠にユーザー承認のもと `gemm_mma.rs::CudaMmaGemm::new`（本番既定
//! コンストラクタ）へ結線した。したがって [`select_swizzle_group_width`]
//! は通常ビルド（feature 指定なし）でも `new` から到達可能である。

/// グルーピング幅の選択候補（DeepGEMM 同型の 2 候補。実装計画 1 節）。
///
/// 呼び出し元（[`select_swizzle_group_width`]）はイシュー #782 で
/// `gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ・feature 非
/// 依存）から到達可能になった（本ファイル冒頭コメント参照）ため、通常
/// ビルド（feature 指定なし）でも到達する。
const GROUP_WIDTH_CANDIDATES: [u32; 2] = [8, 16];

/// グルーピング幅 `g` を仮定した場合の L2 footprint 近似コスト
/// （`usage = g * block_m + ceil_div(num_sms, g) * block_n`。実装計画 1
/// 節受け入れ基準 1 項）。
///
/// 本モジュールの remap（[`swizzled_block_idx`]）は **M 方向**を
/// `group_width` 個ずつグルーピングし、各グループ内で N を先に全走査する
/// 方式（本ファイル冒頭コメント参照）。1 グループに属する連続
/// `group_width` 個の SM ブロックは同じ N（B タイル。`block_n` 幅）を
/// 共有し M（A タイル。`block_m` 幅）だけが異なるため、footprint は
/// 「1 グループ内で異なる `group_width` 個の A タイル」= `g * block_m` と
/// 「同時に活性な `ceil_div(num_sms, g)` グループそれぞれが専有する 1 個の
/// B タイル」= `ceil_div(num_sms, g) * block_n` の和になる。DeepGEMM の
/// `get_swizzled_block_idx` 選択式はグルーピング軸が逆（N 方向グルーピング
/// のため `g * block_n + ceil_div(num_sms, g) * block_m`）であり、本方式
/// （M 方向グルーピング）へそのまま流用すると軸が入れ替わってしまう点に
/// 注意（Bugbot 指摘・PR #667 レビュー是正）。
///
/// 全項を `u64` で計算し、`u32` 同士の乗算オーバーフロー（`block_n`・
/// `num_sms` は実測値だが、桁溢れを事前に排除しておくことで呼び出し側の
/// 追加検証を不要にする。REQ-8 の「境界検査を省略しない」精神を数値計算
/// 側にも適用した安全側の実装）。
///
/// 呼び出し元（[`select_swizzle_group_width`]）はイシュー #782 で
/// `gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ・feature 非
/// 依存）から到達可能になったため、通常ビルド（feature 指定なし）でも
/// 到達する。
fn swizzle_group_usage(num_sms: u32, block_m: u32, block_n: u32, group_width: u32) -> u64 {
    let (num_sms, block_m, block_n, group_width) = (
        u64::from(num_sms),
        u64::from(block_m),
        u64::from(block_n),
        u64::from(group_width),
    );
    group_width * block_m + num_sms.div_ceil(group_width.max(1)) * block_n
}

/// `num_sms`（SM 数）・ブロックタイル寸法（`block_m`/`block_n`）から
/// グルーピング幅を動的に選ぶ（候補 `{8, 16}` に対し
/// [`swizzle_group_usage`] を最小化する候補を採用。実装計画 1 節受け入れ
/// 基準 1 項）。
///
/// 同値の場合は小さい方（`8`）を採用する（安全側: グルーピング幅が
/// 小さいほど 1 グループが専有する SM 数・L2 footprint が小さく、
/// 効果が過大に振れるリスクが低い）。
///
/// 呼び出し文脈: イシュー #775 でサイズ条件付き適用ロジックを実装した
/// 当初は `gemm_mma.rs::CudaMmaGemm::new_with_size_conditional_swizzle`
/// （`internal-diagnostics` feature 限定・実機検証専用 opt-in 入口）が、
/// swizzle 変種カーネルをコンパイルする際のグルーピング幅決定にこの関数を
/// 使っていた（`device.multiprocessor_count()` 実測値ベース。本ファイル
/// 冒頭コメント参照）。イシュー #782 で結線前必須確認が GB10 実機ゲート
/// （2026-08-21）で解消したことを根拠に、本番既定コンストラクタ
/// `gemm_mma.rs::CudaMmaGemm::new` へ同ロジックを昇格したため、現在は
/// `new`（feature 非依存・常時到達可能）が直接この関数を呼ぶ。診断用
/// ラッパー `lib.rs::diagnostics::mma_swizzle_group_width`
/// （`internal-diagnostics` feature 経由。`examples/gemm_mma_swizzle_
/// bench.rs` の A/B 計測用）からも引き続き呼ばれる。
pub fn select_swizzle_group_width(num_sms: u32, block_m: u32, block_n: u32) -> u32 {
    let mut best = GROUP_WIDTH_CANDIDATES[0];
    let mut best_usage = swizzle_group_usage(num_sms, block_m, block_n, best);
    for &candidate in &GROUP_WIDTH_CANDIDATES[1..] {
        let usage = swizzle_group_usage(num_sms, block_m, block_n, candidate);
        if usage < best_usage {
            best = candidate;
            best_usage = usage;
        }
    }
    best
}

/// 線形ブロック index（CUDA が割り当てる既定順序
/// `blockIdx.y * gridDim.x + blockIdx.x`。`num_m_blocks x num_n_blocks`
/// グリッドを行優先で N 方向を先に辿る順序）を、`group_width` 個の M
/// ブロックごとにグルーピングした順序へ remap し、`(m_block, n_block)`
/// を返す（本ファイル冒頭コメント参照）。
///
/// # 全単射性・端数処理
///
/// `num_m_blocks` が `group_width` で割り切れない場合、末尾グループは
/// `num_m_blocks % group_width`（`remainder`）個の M ブロックのみを持つ
/// グループへ縮小する（実装計画 1 節受け入れ基準 2 項「端数グループは
/// グループ幅を縮めて処理し、任意グリッド寸法で全単射になる方式。2 の
/// べき乗制約なし」）。この方式により `group_width` に対する整除性の
/// 事前条件を課さずに `0..num_m_blocks*num_n_blocks` から
/// `(0..num_m_blocks) x (0..num_n_blocks)` への全単射が任意の
/// `num_m_blocks`/`num_n_blocks`/`group_width`（いずれも `>= 1`）で成立
/// する（網羅テスト `swizzle_remap_is_bijective_over_grid` が機械検査）。
///
/// # panic 契約
///
/// `debug_assert!` のみで境界を検査する（外部入力を直接受けず、下記
/// 「呼び出し文脈」の呼び出し元がいずれも既に妥当性を保証した値のみを
/// 渡すため `Result` 化はオーバーヘッドと判断）。
///
/// # 呼び出し文脈（生産経路からは到達しない参照実装）
///
/// 本関数は `kernels_mma.rs::mma_f16_source_with_swizzle` が生成する
/// CUDA C++ 側の remap 式（本ファイル冒頭コメント「ホスト側の参照実装」
/// 節参照）と同一設計を Rust で独立に再実装したものであり、GPU 実行
/// 経路（`gemm_mma.rs::CudaMmaGemm::new_with_swizzle`）から直接呼ばれる
/// ことはない（呼ばれるのは生成された CUDA 文字列であり、この Rust
/// 関数そのものではない）。呼び出し元は本ファイル末尾の全単射性検査
/// テスト（`swizzle_remap_is_bijective_over_grid` 等）のみであり、
/// カーネル側の同一式との不一致は別経路（`kernels_mma.rs` のソース検査・
/// `gemm_mma.rs` の実機 bit 一致テスト）で検出する。
#[allow(dead_code)]
// 上記のとおり生産経路からの呼び出しを持たない意図的な設計（remap
// アルゴリズムの正しさをテストで独立検証するための参照実装）のため、
// `#[cfg(test)]` テストのみが呼び出し元となる。`cargo build`（テスト cfg
// なし）では未使用と判定されるため明示的に許可する
// （`kernels_mma.rs::MMA_SHARED_MEM_BYTES` の `#[allow(dead_code)]` と同じ
// 判断パターン）。
pub fn swizzled_block_idx(
    linear_idx: u32,
    num_m_blocks: u32,
    num_n_blocks: u32,
    group_width: u32,
) -> (u32, u32) {
    debug_assert!(num_m_blocks >= 1 && num_n_blocks >= 1);
    debug_assert!(group_width >= 1);
    debug_assert!(linear_idx < num_m_blocks * num_n_blocks);

    let full_groups = num_m_blocks / group_width;
    let remainder = num_m_blocks % group_width;
    let full_group_blocks = group_width * num_n_blocks;
    let full_groups_total_blocks = full_groups * full_group_blocks;

    if linear_idx < full_groups_total_blocks {
        let group_idx = linear_idx / full_group_blocks;
        let idx_in_group = linear_idx % full_group_blocks;
        let m_in_group = idx_in_group % group_width;
        let n_block = idx_in_group / group_width;
        let m_block = group_idx * group_width + m_in_group;
        (m_block, n_block)
    } else {
        // 末尾の縮小グループ（サイズ `remainder`。`remainder > 0` が
        // 上記 `if` の否定から保証される: `linear_idx <
        // num_m_blocks*num_n_blocks` かつ `linear_idx >=
        // full_groups_total_blocks` が両立するのは `remainder > 0` の
        // ときのみ）。
        let idx_in_group = linear_idx - full_groups_total_blocks;
        let m_in_group = idx_in_group % remainder;
        let n_block = idx_in_group / remainder;
        let m_block = full_groups * group_width + m_in_group;
        (m_block, n_block)
    }
}

/// swizzle 適用の閾値（総ブロックタイル数。`num_m_blocks * num_n_blocks`）。
///
/// イシュー #775 の 2026-08-20 GB10 実機 A/B 計測（`docs/perf/
/// cuda-gemm-swizzle-ab.md` §6）で、M=N=K=4096（`num_m_blocks=64・
/// num_n_blocks=32`。`kernels_mma::MMA_BM=64`/`MMA_BN=128` 単位。総タイル数
/// 2048）が ×1.578（base 34.4089 → swizzle 54.3055 TFLOPS）の改善を安定
/// 再現した一方、M=N=K=2048（`num_m_blocks=32・num_n_blocks=16`。総タイル数
/// 512）は ×0.992（中立〜微減）だった。実測で改善を確認した点（総タイル数
/// 2048）以上のみ適用する保守的な閾値とし、実測点未満（512〜2048 側）へは
/// 外挿しない（`should_apply_swizzle` 参照）。
///
/// **非正方形形状は本閾値だけでは判定しない（PR #784 codex-review P1 是正）**:
/// 上記実測点はいずれも正方形形状（M=N=K）であり、本閾値は総ブロックタイル数
/// `num_m_blocks * num_n_blocks` のみで判定するためアスペクト比を考慮しない。
/// したがって M=32768, N=512（総タイル数 32768/64 * 512/128 = 512*4=2048。
/// 閾値ちょうど）のような縦長・横長形状は、4096 正方形の実測点（総タイル数
/// も 2048）とは全く異なるメモリアクセスパターン・L2 再利用特性を持ちうる。
/// この非正方形形状への外挿は未検証（`docs/perf/cuda-gemm-swizzle-ab.md`
/// §6.2「未検証のまま残る事項」）であり、イシュー #775 の承認記録（`docs/
/// perf/cuda-gemm-swizzle-ab.md` §2「4096 級のみ適用・512〜2048 は劣化 5%
/// 以内」）が前提とする正方形実測を超えて適用してはならない。そのため
/// [`should_apply_swizzle`] は本閾値に加えて
/// [`SWIZZLE_APPLY_MIN_M_BLOCKS`]/[`SWIZZLE_APPLY_MIN_N_BLOCKS`]（両軸とも
/// 実測点 M=N=K=4096 級以上であることを要求する軸別ガード）を課し、
/// 未検証の非正方形形状（例: M=32768, N=512）を base 経路へフォールバック
/// させる。今後非正方形形状の A/B 計測を実施し改善を確認できた場合は、
/// 軸別ガードをアスペクト比考慮の判定式へ改訂することを検討すること。
pub const SWIZZLE_APPLY_TILE_COUNT_THRESHOLD: u64 = 2048;

/// M 方向ブロック数の適用下限（イシュー #775 実測点 M=N=K=4096 由来。
/// `4096 / kernels_mma::MMA_BM(64) = 64`）。[`SWIZZLE_APPLY_TILE_COUNT_THRESHOLD`]
/// ドキュメンテーションコメント「非正方形形状は本閾値だけでは判定しない」
/// 参照。`kernels_mma::MMA_BM` を直接参照せず定数値へ焼き込むのは、本
/// モジュールが `kernels_mma` に依存しない参照実装として独立性を保つ既存
/// 方針（本ファイル冒頭コメント「ホスト側の参照実装」節）に揃えるため。
/// 値の一致は `kernels_mma::MMA_BM`/`MMA_BN` からの再導出値との
/// `const` アサート（`gemm_mma.rs` 冒頭。`kernels_mma.rs:352`/`:387` と
/// 同型の設計時契約検査）で機械検証する（コンパイル時に評価されるため
/// 通常ビルドで常時検査される。GPU 実機テストの到達可否に依存しない）。
pub const SWIZZLE_APPLY_MIN_M_BLOCKS: u32 = 64;

/// N 方向ブロック数の適用下限（イシュー #775 実測点 M=N=K=4096 由来。
/// `4096 / kernels_mma::MMA_BN(128) = 32`）。導出根拠・独立性の理由は
/// [`SWIZZLE_APPLY_MIN_M_BLOCKS`] と同様。
pub const SWIZZLE_APPLY_MIN_N_BLOCKS: u32 = 32;

/// K 方向の適用下限（イシュー #775 実測点 M=N=K=4096 由来。K の生値を
/// そのまま下限とする。PR #784 codex-review 追加指摘の是正）。
///
/// [`SWIZZLE_APPLY_MIN_M_BLOCKS`]/[`SWIZZLE_APPLY_MIN_N_BLOCKS`] はグリッド
/// 分割軸（`MMA_BM`/`MMA_BN` 単位のブロック数）であるのに対し、K は
/// `mma_launch_config` のグリッド構築に現れず（`kernels_mma.rs` のカーネル
/// 内 `for` ループで `MMA_BK` 単位に消費されるのみ）、ブロック数への換算対象
/// ではない。そのため他 2 定数と異なり `kernels_mma::MMA_BK` からの再導出は
/// 行わず、実測承認点 M=N=K=4096 の生の K 値をそのまま定数化する
/// （`should_apply_swizzle` ドキュメンテーションコメント参照）。
///
/// `should_apply_swizzle` へ K ガードを追加しない場合、M=N=4096・K=8 の
/// ような形状（M/N 軸ガードは満たすが、メモリアクセス量・L2 再利用特性が
/// 実測承認点 M=N=K=4096 と大きく異なる）にも本番 swizzle が適用されて
/// しまう（codex-review P1 指摘・PR #784）。
pub const SWIZZLE_APPLY_MIN_K: u32 = 4096;

/// M/N の raw 次元（`m`/`n`）に対する正方形（`m == n`）レイ上の下限。
/// イシュー #775 実測点 M=N=K=4096 由来
/// （[`SWIZZLE_APPLY_MIN_M_BLOCKS`]/[`SWIZZLE_APPLY_MIN_N_BLOCKS`] は
/// `kernels_mma::MMA_BM`/`MMA_BN` 単位のブロック数下限であるのに対し、
/// 本定数は raw 次元の下限。PR #784 codex-review P1 是正で追加）。
///
/// **背景（PR #784 codex-review P1 指摘）**: 軸別ブロック数ガード
/// （[`SWIZZLE_APPLY_MIN_M_BLOCKS`]/[`SWIZZLE_APPLY_MIN_N_BLOCKS`]）は
/// 両軸それぞれの下限のみを課すため、M=8192, N=4096, K=4096 のような
/// 「両軸とも下限以上だが正方形ではない」形状を通してしまっていた。
/// 実測承認（`docs/perf/cuda-gemm-swizzle-ab.md` §2/§6.3）は M=N=K=4096 の
/// **正方形**のみを対象としており、アスペクト比の制約が欠けていた。
/// [`should_apply_swizzle`] へ `m == n`（要素次元での厳密一致）を新たな
/// AND 条件として追加し、実測承認済みの「正方形レイ上（M=N かつ
/// M=N>=4096 かつ K>=4096）」のみへ適用範囲を絞る。
pub const SWIZZLE_APPLY_MIN_SQUARE_DIM: u32 = 4096;

/// グリッド形状（`num_m_blocks x num_n_blocks`。`gemm_mma.rs::
/// mma_launch_config` が構築する grid_dim.y/x に対応）と raw 次元
/// （`m`/`n`）から、swizzle remap を適用すべきかを判定する（イシュー #775
/// のサイズ条件付き適用ロジック。[`SWIZZLE_APPLY_TILE_COUNT_THRESHOLD`]
/// ドキュメンテーションコメント参照）。
///
/// **正方形条件（`m == n`。PR #784 codex-review P1 是正で追加）**: 実測
/// 承認（`docs/perf/cuda-gemm-swizzle-ab.md` §2/§6.3）は M=N=K=4096 の
/// 正方形形状のみを対象としているため、raw 次元が厳密に一致する場合
/// （かつ [`SWIZZLE_APPLY_MIN_SQUARE_DIM`] 以上）のみを適用対象とする
/// （[`SWIZZLE_APPLY_MIN_SQUARE_DIM`] ドキュメンテーションコメント参照）。
///
/// 総タイル数閾値に加えて、M/N 各軸のブロック数がそれぞれ
/// [`SWIZZLE_APPLY_MIN_M_BLOCKS`]/[`SWIZZLE_APPLY_MIN_N_BLOCKS`] 以上
/// （実測点 M=N=K=4096 と同水準以上）であることを要求する（PR #784
/// codex-review P1 是正: 総タイル数のみの判定では M=32768, N=512 のような
/// 未検証の非正方形形状にも適用してしまうため）。
///
/// さらに `k`（グリッド分割軸ではなく `mma_launch_config` の grid 構築には
/// 現れない引数。呼び出し元の実 K 値をそのまま渡す）が
/// [`SWIZZLE_APPLY_MIN_K`] 以上であることも要求する（PR #784 codex-review
/// 追加指摘の是正: M/N 軸ガードのみでは M=N=4096, K=8 のような形状——
/// メモリアクセス量・L2 再利用特性が実測承認点 M=N=K=4096 と大きく異なる
/// ——にも適用してしまうため）。
///
/// 5 条件（正方形・raw 次元下限・総タイル数・軸別ブロック数・K）すべてを
/// 満たす場合のみ `true` を返し、いずれか一つでも未達なら base 経路へ
/// フォールバックする。
///
/// **呼び出し元が `mma_launch_config` と同一の `div_ceil`（`MMA_BM`/
/// `MMA_BN` 単位）で `num_m_blocks`/`num_n_blocks` を `m`/`n` から導出する
/// 限り（`gemm_mma.rs::should_launch_swizzle_kernel` がこれを保証する。
/// 本関数自体は純粋な述語であり `num_m_blocks`/`num_n_blocks` が `m`/`n`
/// と整合しない任意の組も受理できてしまうため、この保証は呼び出し元の
/// 責務である）、`tile_count_ok`・`axis_ok` は `square_ok && dim_ok` が
/// 成立する下では常に成立し冗長**（`m == n` かつ
/// `m >= SWIZZLE_APPLY_MIN_SQUARE_DIM`（= 4096）ならば、`div_ceil` の
/// 単調性から `num_m_blocks = m.div_ceil(MMA_BM) >=
/// 4096.div_ceil(MMA_BM) == SWIZZLE_APPLY_MIN_M_BLOCKS`・`num_n_blocks`
/// も同様に `SWIZZLE_APPLY_MIN_N_BLOCKS` 以上になり、したがって
/// `tile_count_ok`（両下限の積 `2048` 以上）も自動的に成立する）。この
/// 前提下では実効的な適用条件は「正方形レイ上（M=N かつ M=N>=4096）かつ
/// K>=4096」まで単純化できる（`should_apply_swizzle_min_square_dim_
/// implies_axis_and_tile_guards` が導出済みブロック数に対してこの含意を
/// 機械検証する。一方で `should_apply_swizzle_boundary_at_threshold` は
/// 本関数の述語としての一般性を検査するため、意図的に非対応の
/// `m`/`n`・ブロック数の組を渡す）。それでも `tile_count_ok`/`axis_ok` を
/// 条件として残すのは、イシュー #775
/// の承認記録が「総タイル数 2048」・「軸別ブロック数」を採用基準の根拠
/// としているため、正方形条件のみを唯一の真実源にせず両方を明示すること
/// で、将来正方形条件だけを緩和する変更が承認済みの下限（総タイル数
/// 2048・軸別ブロック数）を静かに割り込むのを防ぐ多層防御である（下記
/// `should_apply_swizzle_axis_and_tile_thresholds_stay_consistent` が
/// `SWIZZLE_APPLY_MIN_M_BLOCKS as u64 * SWIZZLE_APPLY_MIN_N_BLOCKS as u64
/// >= SWIZZLE_APPLY_TILE_COUNT_THRESHOLD` を機械検証する）。
///
/// 呼び出し元: `gemm_mma.rs::CudaMmaGemm::launch_f16`（`mma_f16_swizzle` が
/// `Some` の場合——本番既定コンストラクタ `new` が SM 数実測に成功した
/// 場合に発生。イシュー #782 で `new` へ結線済み——に、形状ごとに base／
/// swizzle 変種いずれのカーネルを起動するか、この関数の戻り値で分岐する）。
/// **イシュー #856 で `gemm.rs::CudaGemm::should_launch_wmma_tf32_staged_swizzle`
/// （TF32 opt-staged 経路。`wmma_tf32_staged_swizzle` が `Some` の場合）も
/// 同じ判定関数を呼ぶようになった**——ブロック数 `num_m_blocks`/
/// `num_n_blocks` は `WMMA_TF32_STAGED_BLOCK_M`/`_N`（64×64。f16 側の
/// `MMA_BM`/`_BN`=64×128 とは異なるブロックタイル）から独立に導出される
/// ため、両呼び出し元は同じ関数を異なるブロックタイル前提で共有する
/// （2026-08-22 GB10 実機 A/B: 4096 中央値 ×1.5434・512〜2048 劣化 5% 以内。
/// `docs/perf/cuda-gemm-swizzle-ab.md` §7.6/§7.7.6 参照）。`u64` 積で `u32`
/// 同士のオーバーフローを避ける（[`swizzle_group_usage`] と同じ安全側方針。
/// REQ-8 の「境界検査を省略しない」精神を数値計算側にも適用）。
pub fn should_apply_swizzle(m: u32, n: u32, num_m_blocks: u32, num_n_blocks: u32, k: u32) -> bool {
    let square_ok = m == n;
    let dim_ok = m >= SWIZZLE_APPLY_MIN_SQUARE_DIM;
    let tile_count_ok =
        u64::from(num_m_blocks) * u64::from(num_n_blocks) >= SWIZZLE_APPLY_TILE_COUNT_THRESHOLD;
    let axis_ok =
        num_m_blocks >= SWIZZLE_APPLY_MIN_M_BLOCKS && num_n_blocks >= SWIZZLE_APPLY_MIN_N_BLOCKS;
    let k_ok = k >= SWIZZLE_APPLY_MIN_K;
    square_ok && dim_ok && tile_count_ok && axis_ok && k_ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// `usage` 式の手計算ケース（`docs/perf/cuda-gemm-swizzle-ab.md` の
    /// 設計根拠と対応）。B-3（#494）時点の `MMA_BM=64`/`MMA_BN=128` を
    /// 使い、`num_sms` を数種類振って期待候補が固定されることを検査する。
    #[test]
    fn select_swizzle_group_width_matches_hand_computed_usage() {
        // usage(g) = g*block_m + ceil(num_sms/g)*block_n（本ファイル
        // `swizzle_group_usage` ドキュメンテーションコメント参照。M 方向
        // グルーピングのため block_m/block_n の役割は DeepGEMM の
        // N-grouping 式と入れ替わる。Bugbot 指摘・PR #667 レビュー是正）。
        // block_m=64, block_n=128 のとき:
        //
        // num_sms=1: usage(8) = 8*64 + ceil(1/8)*128 = 512 + 128 = 640
        //            usage(16) = 16*64 + ceil(1/16)*128 = 1024 + 128 = 1152
        // -> 8 が最小。
        assert_eq!(select_swizzle_group_width(1, 64, 128), 8);

        // num_sms=132（DGX Spark GB10 級の SM 数を想定した仮値）:
        // usage(8) = 8*64 + ceil(132/8)*128 = 512 + 17*128 = 512+2176=2688
        // usage(16) = 16*64 + ceil(132/16)*128 = 1024 + 9*128 = 1024+1152=2176
        // -> 16 が最小。
        assert_eq!(select_swizzle_group_width(132, 64, 128), 16);

        // num_sms を極端に大きくしても group_width=16 側が有利であり続ける
        // ことを確認する（block_n の係数が ceil_div 側に付くため num_sms
        // が増えるほど大きい g が漸近的に有利になる。num_sms=100000):
        // usage(8) = 512 + ceil(100000/8)*128 = 512 + 12500*128 = 1600512
        // usage(16) = 1024 + ceil(100000/16)*128 = 1024 + 6250*128 = 801024
        // -> 16 が最小。
        assert_eq!(select_swizzle_group_width(100_000, 64, 128), 16);
    }

    /// **注意（PR #758 レビュー是正・イシュー #777 で実測値反映。#781 codex-review
    /// 指摘で参照先節名を是正）**: この `28` は GB10（sm_121）の実測 SM 数ではない。
    /// `docs/perf/sm121-device-attributes.md` §「動作検証」の出力例にある
    /// `MULTIPROCESSOR_COUNT = 28` は、同節が明記するとおり RTX 3060（compute
    /// capability 8.6・sm_121 ではない）の例示ダンプの値である（「デバイス属性
    /// 実測表」節の値ではない。同表の `MULTIPROCESSOR_COUNT`〈SM 数〉は 48）。
    /// GB10（sm_121）自体の実測 SM 数は
    /// 48（同ドキュメント「デバイス属性実測表」節。2026-08-19 実測・イシュー #739、
    /// 2026-08-20 のベンチ起動診断〈`gemm_mma_swizzle_bench` の `num_sms=48` 出力〉で
    /// 再確認・イシュー #777。#781 codex-review 指摘是正: `cuda_floor_bench` は
    /// `swizzle_group_width()` が `None` であることを診断するのみで
    /// `multiprocessor_count()`／`num_sms` 出力は行わないため再確認元に含めない）
    /// であり、以前の版が主張していた
    /// 「本リポでは未実測」はもはや事実誤り（Cursor Bugbot 指摘・PR #758 当時は
    /// 未実測だった）。本テストは GB10 実機値のピン留めとしてではなく、
    /// `select_swizzle_group_width` の入力 `28`（他のテストケースと同様の代表値の
    /// 一つ）に対する回帰検知として位置づけ、GB10 実測値 48 判明後も入力・
    /// assert を変更せず維持する（本テストの非変更はイシュー #777 の受け入れ条件）。
    /// なお実測値 48 に対しても usage(8) = 8*64 + ceil(48/8)*128 = 512+768=1280 <
    /// usage(16) = 16*64 + ceil(48/16)*128 = 1024+384=1408 で g8 が選択されることは
    /// 机上計算および実機起動診断（`dynamic_group_width=8` 出力・イシュー #777）で
    /// 確認済みであり、本テストの結論（g8）と整合する。
    #[test]
    fn select_swizzle_group_width_pins_example_sm_count_28_to_g8() {
        // usage(8)  = 8*64  + ceil(28/8)*128  = 512  + 4*128 = 512+512=1024
        // usage(16) = 16*64 + ceil(28/16)*128 = 1024 + 2*128 = 1024+256=1280
        // -> 8 が最小。
        assert_eq!(select_swizzle_group_width(28, 64, 128), 8);
    }

    /// 同値のときは小さい方（8）を採用する（本ファイル
    /// `select_swizzle_group_width` ドキュメンテーションコメント
    /// 「安全側」参照）。
    #[test]
    fn select_swizzle_group_width_prefers_smaller_candidate_on_tie() {
        // usage(8) == usage(16) となる真の同値ケース: block_m=block_n=64,
        // num_sms=128 のとき
        //   usage(8)  = 8*64  + ceil(128/8)*64  = 512  + 16*64 = 512+1024=1536
        //   usage(16) = 16*64 + ceil(128/16)*64 = 1024 + 8*64  = 1024+512=1536
        // で両者が一致する。候補配列の順序（8 が先）と `<` 比較
        // （`<=` ではない）により同値時は先に見た 8 が残ることをこの
        // 組で直接検査する。
        assert_eq!(select_swizzle_group_width(128, 64, 64), 8);
    }

    /// **全単射性の網羅テスト**（実装計画 1 節受け入れ基準 2 項）:
    /// `num_m_blocks, num_n_blocks ∈ 1..=17` x `group_width ∈ {1, 3, 8,
    /// 16}` の全組合せで、`0..num_m_blocks*num_n_blocks` の remap 結果が
    /// 重複なく全タイルを被覆し、かつ常にグリッド内に収まることを検査
    /// する（REQ-8 の被覆・範囲根拠を数値計算側に適用したもの）。
    #[test]
    fn swizzle_remap_is_bijective_over_grid() {
        for num_m_blocks in 1..=17u32 {
            for num_n_blocks in 1..=17u32 {
                for group_width in [1u32, 3, 8, 16] {
                    let total = num_m_blocks * num_n_blocks;
                    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(total as usize);
                    for linear_idx in 0..total {
                        let (m, n) =
                            swizzled_block_idx(linear_idx, num_m_blocks, num_n_blocks, group_width);
                        assert!(
                            m < num_m_blocks && n < num_n_blocks,
                            "num_m_blocks={num_m_blocks} num_n_blocks={num_n_blocks} \
                             group_width={group_width} linear_idx={linear_idx}: \
                             remap 結果 (m={m}, n={n}) がグリッド範囲外です"
                        );
                        assert!(
                            seen.insert((m, n)),
                            "num_m_blocks={num_m_blocks} num_n_blocks={num_n_blocks} \
                             group_width={group_width} linear_idx={linear_idx}: \
                             remap 結果 (m={m}, n={n}) が重複しています（全単射性違反）"
                        );
                    }
                    assert_eq!(
                        seen.len(),
                        total as usize,
                        "num_m_blocks={num_m_blocks} num_n_blocks={num_n_blocks} \
                         group_width={group_width}: 被覆したタイル数が総タイル数と \
                         一致しません"
                    );
                }
            }
        }
    }

    /// `group_width == 1` は各グループがちょうど 1 個の M ブロックのみを
    /// 持つ退化ケースであり、恒等写像
    /// （`linear_idx = m_block * num_n_blocks + n_block`。既定の行優先
    /// 順序と一致）になることを検査する（`kernels_mma.rs::
    /// mma_f16_source_with_swizzle` はこの退化ケースをカーネルソース
    /// 生成側では拒否するが、本関数自体は入力として受理する。両者の
    /// 責務分離は本ファイル冒頭コメント参照）。
    #[test]
    fn swizzle_remap_group_width_one_is_identity_mapping() {
        let num_m_blocks = 5;
        let num_n_blocks = 4;
        for linear_idx in 0..(num_m_blocks * num_n_blocks) {
            let (m, n) = swizzled_block_idx(linear_idx, num_m_blocks, num_n_blocks, 1);
            assert_eq!(m, linear_idx / num_n_blocks);
            assert_eq!(n, linear_idx % num_n_blocks);
        }
    }

    /// グループ内の走査順（N を先に全走査してから次の M グループへ）を
    /// 小さな具体例で固定する（`num_m_blocks=5, num_n_blocks=2,
    /// group_width=2`。グループは `{0,1}`（フル）・`{2,3}`（フル）・
    /// `{4}`（remainder=1 の縮小グループ）の 3 個）。
    #[test]
    fn swizzle_remap_matches_hand_traced_group_order() {
        let num_m_blocks = 5;
        let num_n_blocks = 2;
        let group_width = 2;
        // グループ 0（m=0,1）: (m=0,n=0),(m=1,n=0),(m=0,n=1),(m=1,n=1)
        // グループ 1（m=2,3）: (m=2,n=0),(m=3,n=0),(m=2,n=1),(m=3,n=1)
        // グループ 2（m=4 のみ・remainder=1）: (m=4,n=0),(m=4,n=1)
        let expected: [(u32, u32); 10] = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (4, 0),
            (4, 1),
        ];
        for (linear_idx, &want) in expected.iter().enumerate() {
            let got =
                swizzled_block_idx(linear_idx as u32, num_m_blocks, num_n_blocks, group_width);
            assert_eq!(got, want, "linear_idx={linear_idx}");
        }
    }

    /// [`should_apply_swizzle`] の境界値検査（実装計画 2 節「swizzle.rs」）。
    /// M=N=K=4096（`num_m_blocks=64・num_n_blocks=32`。総タイル数 2048）は
    /// 閾値ちょうど・M=N=K=2048（`num_m_blocks=32・num_n_blocks=16`。総タイル数
    /// 512）は明確に未達（`docs/perf/cuda-gemm-swizzle-ab.md` §6 実測値と
    /// 対応する具体的な形状で検査する。閾値定数直下のドキュメンテーション
    /// コメント参照）。
    #[test]
    fn should_apply_swizzle_matches_4096_and_2048_measured_shapes() {
        // M=N=K=4096 相当: num_m_blocks=64, num_n_blocks=32, k=4096 → 総タイル数 2048。
        assert!(should_apply_swizzle(4096, 4096, 64, 32, 4096));
        // M=N=K=2048 相当: num_m_blocks=32, num_n_blocks=16, k=2048 → 総タイル数 512。
        assert!(!should_apply_swizzle(2048, 2048, 32, 16, 2048));
    }

    /// 閾値の丁度・1 個未満の境界を直接検査する（`SWIZZLE_APPLY_TILE_COUNT_
    /// THRESHOLD = 2048`。軸別ガード `SWIZZLE_APPLY_MIN_M_BLOCKS=64`/
    /// `SWIZZLE_APPLY_MIN_N_BLOCKS=32` を両方満たす形状のみで検査する。K は
    /// いずれのケースも `SWIZZLE_APPLY_MIN_K`（4096）以上を渡し、M/N 軸の
    /// 判定のみを分離して検査する）。
    ///
    /// **注意**: ここで渡す `m`/`n`（raw 次元）と `num_m_blocks`/
    /// `num_n_blocks`（ブロック数）は意図的に非対応の組（`m=n=
    /// SWIZZLE_APPLY_MIN_SQUARE_DIM` 固定・ブロック数のみをケースごとに
    /// 振る）にしている。本関数は純粋な述語であり、ブロック数が
    /// `mma_launch_config`／`should_launch_swizzle_kernel` の `div_ceil`
    /// 導出と整合しない引数を渡すこと自体は許容される（本番経路は
    /// `should_launch_swizzle_kernel` が常に `m`/`n` から一貫して
    /// ブロック数を導出するため、この非対応の組は生産経路には現れない）。
    /// `square_ok`/`dim_ok` を固定して他 3 条件（`tile_count_ok`/
    /// `axis_ok`）だけを単独で振ることで、この境界テストが検査したい
    /// 対象（軸別ガード・総タイル数ガード）を正しく分離する（`m=n=1` を
    /// 渡すと `dim_ok` で早期に短絡し、ブロック数側の条件が到達不能になる
    /// ため避ける）。
    #[test]
    fn should_apply_swizzle_boundary_at_threshold() {
        const M: u32 = SWIZZLE_APPLY_MIN_SQUARE_DIM;
        const N: u32 = SWIZZLE_APPLY_MIN_SQUARE_DIM;
        // 2047 タイル相当（89 x 23）は総タイル数・軸ガードいずれも未達で未適用。
        assert!(!should_apply_swizzle(M, N, 89, 23, 4096));
        // 2048 タイル（64 x 32）は総タイル数ちょうど・軸ガードも両方ちょうど満たし、
        // かつ正方形（m=n=SWIZZLE_APPLY_MIN_SQUARE_DIM）のため適用。
        assert!(should_apply_swizzle(M, N, 64, 32, 4096));
        // 総タイル数は 2048 以上（2048 x 1）だが N 軸ブロック数 1 が
        // SWIZZLE_APPLY_MIN_N_BLOCKS(32) 未達のため未適用（PR #784 是正）。
        assert!(!should_apply_swizzle(M, N, 2048, 1, 4096));
        // 同様に M 方向のみ大きい非正方形状（総タイル数 4096 は閾値以上）も
        // N 軸ブロック数 1 が未達のため未適用。
        assert!(!should_apply_swizzle(M, N, 4096, 1, 4096));
        assert!(!should_apply_swizzle(M, N, 1, 2047, 4096));
    }

    /// 非正方形形状（M 軸のみ実測水準・N 軸は小さい／逆に N 軸のみ実測水準・
    /// M 軸は小さい）は、総タイル数が閾値 2048 以上でも軸別ガードにより
    /// 未適用（base フォールバック）になることを検査する（codex-review P1
    /// 指摘・PR #784: 実測承認範囲〈M=N=K=4096 正方形〉を超える非正方形
    /// 形状への外挿を防ぐガード）。K はいずれも `SWIZZLE_APPLY_MIN_K`
    /// 以上を渡し、M/N 軸の判定のみを分離して検査する。
    #[test]
    fn should_apply_swizzle_rejects_unverified_skewed_shapes_even_when_tile_count_is_sufficient() {
        // 指摘に挙げられた具体例そのもの: M=32768, N=512
        // （num_m_blocks=32768/64=512, num_n_blocks=512/128=4）。
        // 総タイル数 512*4=2048（閾値ちょうど）だが N 軸ブロック数 4 が
        // SWIZZLE_APPLY_MIN_N_BLOCKS(32) 未達のため未適用（かつ非正方形）。
        assert!(!should_apply_swizzle(32768, 512, 512, 4, 4096));
        // 逆方向（N 軸のみ大きい・M 軸が小さい）: num_m_blocks=4,
        // num_n_blocks=512 → 総タイル数 2048 だが M 軸ブロック数 4 が
        // SWIZZLE_APPLY_MIN_M_BLOCKS(64) 未達のため未適用（かつ非正方形）。
        assert!(!should_apply_swizzle(512, 32768, 4, 512, 4096));
    }

    /// **本 PR（#784 codex-review P1 追加是正）の核心テスト**: M/N 各軸の
    /// ブロック数ガード・総タイル数ガードを両方満たす（＝旧判定なら適用
    /// されていた）が、raw 次元が正方形でない（`m != n`）形状は base 経路へ
    /// フォールバックすることを検査する。指摘に挙げられた具体例
    /// M=8192, N=4096, K=4096（両軸とも下限以上）とその転置形状を検査する。
    #[test]
    fn should_apply_swizzle_rejects_non_square_shapes_even_when_both_axis_guards_pass() {
        // M=8192, N=4096, K=4096: num_m_blocks=8192/64=128 (>=64),
        // num_n_blocks=4096/128=32 (>=32) と両軸ガードを満たすが m != n。
        assert!(!should_apply_swizzle(8192, 4096, 128, 32, 4096));
        // 転置形状 M=4096, N=8192: num_m_blocks=4096/64=64 (>=64),
        // num_n_blocks=8192/128=64 (>=32) と両軸ガードを満たすが m != n。
        assert!(!should_apply_swizzle(4096, 8192, 64, 64, 4096));
    }

    /// 正方形形状で軸別ガードの境界を直接検査する: 実測水準ちょうど
    /// （M=N=K=4096 相当）は適用、実測水準未満（M=N=K=2048 相当）は不適用
    /// （既存の総タイル数基準の判定と一致することを確認する回帰）。加えて
    /// 実測承認点を超える正方形レイ上の形状（M=N=K=8192）も適用対象になる
    /// ことを検査する（実測水準以上の正方形は引き続き適用対象。PR #784
    /// codex-review P1 是正の適用範囲「正方形レイ上（M=N かつ M=N>=4096）」）。
    #[test]
    fn should_apply_swizzle_axis_guard_matches_measured_square_shapes() {
        // M=N=K=4096 相当: num_m_blocks=64, num_n_blocks=32, k=4096（軸ガードちょうど）。
        assert!(should_apply_swizzle(4096, 4096, 64, 32, 4096));
        // M=N=K=2048 相当: num_m_blocks=32, num_n_blocks=16, k=2048（総タイル数・軸ガード共に未達）。
        assert!(!should_apply_swizzle(2048, 2048, 32, 16, 2048));
        // M=N=K=8192（実測承認点 4096 を超える正方形レイ上）: num_m_blocks=128,
        // num_n_blocks=64, k=4096。正方形かつ全ガード超過のため適用。
        assert!(should_apply_swizzle(8192, 8192, 128, 64, 4096));
    }

    /// K ガードの境界値検査（PR #784 codex-review 追加指摘の是正）:
    /// M/N 軸は実測承認点ちょうど（`num_m_blocks=64, num_n_blocks=32`。総
    /// タイル数 2048）を満たすが K が実測承認点未満の形状（例: 指摘に挙がった
    /// M=N=4096, K=8）は未適用へフォールバックし、K が実測承認点
    /// （[`SWIZZLE_APPLY_MIN_K`]=4096）ちょうど・以上の場合のみ適用される
    /// ことを検査する。
    #[test]
    fn should_apply_swizzle_rejects_shapes_with_k_below_measured_point() {
        // 指摘に挙げられた具体例そのもの: M=N=4096, K=8。
        assert!(!should_apply_swizzle(4096, 4096, 64, 32, 8));
        // 従来 (m,n,k)=(4096,4096,32) の bit 一致検証で使っていた
        // 省メモリ用 k=32 も、K ガード追加後は未適用になる
        // （`gemm_mma.rs` の実機 bit 一致テスト再構成と対応）。
        assert!(!should_apply_swizzle(4096, 4096, 64, 32, 32));
        // SWIZZLE_APPLY_MIN_K ちょうど 1 個未満。
        assert!(!should_apply_swizzle(
            4096,
            4096,
            64,
            32,
            SWIZZLE_APPLY_MIN_K - 1
        ));
    }

    /// K が実測承認点ちょうど・それ以上の場合は M/N 軸ガードと合わせて
    /// 適用されることを検査する（[`should_apply_swizzle_rejects_shapes_
    /// with_k_below_measured_point`] の裏側）。
    #[test]
    fn should_apply_swizzle_accepts_shapes_with_k_at_or_above_measured_point() {
        assert!(should_apply_swizzle(
            4096,
            4096,
            64,
            32,
            SWIZZLE_APPLY_MIN_K
        ));
        assert!(should_apply_swizzle(
            4096,
            4096,
            64,
            32,
            SWIZZLE_APPLY_MIN_K + 1
        ));
    }

    /// `u32::MAX` 同士の積で `u64` 計算がオーバーフローしないことを検査する
    /// （閾値定数直下ドキュメンテーションコメント「REQ-8 の境界検査を数値
    /// 計算側にも適用」参照）。`m == n == u32::MAX` を渡し正方形条件も
    /// 満たしたうえでオーバーフローしないことを確認する。
    #[test]
    fn should_apply_swizzle_does_not_overflow_on_large_inputs() {
        assert!(should_apply_swizzle(
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX
        ));
    }

    /// 軸別ガード（`SWIZZLE_APPLY_MIN_M_BLOCKS`/`SWIZZLE_APPLY_MIN_N_BLOCKS`）
    /// の積が承認済みの総タイル数閾値（`SWIZZLE_APPLY_TILE_COUNT_THRESHOLD`）
    /// 以上であることを検査する（`should_apply_swizzle` ドキュメンテーション
    /// コメント「`tile_count_ok`・`axis_ok` は `square_ok && dim_ok` の下では
    /// 常に冗長」節参照。将来どちらかの軸下限のみを緩和する変更が、承認済み
    /// の総タイル数下限を静かに割り込むのを防ぐ回帰検知）。
    #[test]
    fn should_apply_swizzle_axis_and_tile_thresholds_stay_consistent() {
        assert!(
            u64::from(SWIZZLE_APPLY_MIN_M_BLOCKS) * u64::from(SWIZZLE_APPLY_MIN_N_BLOCKS)
                >= SWIZZLE_APPLY_TILE_COUNT_THRESHOLD
        );
    }

    /// 実測承認済みの下限形状（M=N=K=4096）で `should_apply_swizzle` の
    /// ドキュメンテーションコメントが主張する「`square_ok && dim_ok` が
    /// 成立すれば `tile_count_ok`/`axis_ok` も自動的に成立する」を機械検証
    /// する（advisor 指摘: 縮小のみであることをコメントで主張するだけでなく
    /// テストで裏付ける）。`kernels_mma::MMA_BM`/`MMA_BN` 値をここへ焼き込む
    /// のは、本モジュールが `kernels_mma` に依存しない独立性を保つ既存方針
    /// （本ファイル冒頭コメント）に揃えるため（値の一致は `gemm_mma.rs` 冒頭
    /// の `const` アサートが別途検証する）。
    #[test]
    fn should_apply_swizzle_min_square_dim_implies_axis_and_tile_guards() {
        const MMA_BM: u32 = 64;
        const MMA_BN: u32 = 128;
        let m = SWIZZLE_APPLY_MIN_SQUARE_DIM;
        let n = SWIZZLE_APPLY_MIN_SQUARE_DIM;
        let num_m_blocks = m.div_ceil(MMA_BM);
        let num_n_blocks = n.div_ceil(MMA_BN);
        assert!(num_m_blocks >= SWIZZLE_APPLY_MIN_M_BLOCKS);
        assert!(num_n_blocks >= SWIZZLE_APPLY_MIN_N_BLOCKS);
        assert!(
            u64::from(num_m_blocks) * u64::from(num_n_blocks) >= SWIZZLE_APPLY_TILE_COUNT_THRESHOLD
        );
        // 上記が成立する具体形状に対し `should_apply_swizzle` 自体も適用と
        // 判定することを確認する（ドキュメンテーションコメントの主張と
        // 実装が整合していることの直接検証）。
        assert!(should_apply_swizzle(
            m,
            n,
            num_m_blocks,
            num_n_blocks,
            SWIZZLE_APPLY_MIN_K
        ));
    }
}
