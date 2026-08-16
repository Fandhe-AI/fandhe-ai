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
//! 呼ぶ参照実装。[`select_swizzle_group_width`] は
//! `lib.rs::diagnostics::mma_swizzle_group_width`〈`internal-diagnostics`
//! feature〉経由で `examples/gemm_mma_swizzle_bench.rs` から呼ばれる）。
//! GPU 実行自体は opt-in 経路（`gemm_mma.rs::CudaMmaGemm::
//! new_with_swizzle`。`group_width` は呼び出し元が明示的に渡す）のみが
//! 担い、本番ディスパッチ経路（`ops.rs`／`gemm_auto.rs`）には結線しない
//! （実装計画 2 節「実行環境の制約と安全側判断」）。

/// グルーピング幅の選択候補（DeepGEMM 同型の 2 候補。実装計画 1 節）。
///
/// `#[allow(dead_code)]` について: 本定数・[`swizzle_group_usage`]・
/// [`select_swizzle_group_width`] は、`internal-diagnostics` feature
/// （既定 off）を有効化した場合のみ到達可能な
/// `lib.rs::diagnostics::mma_swizzle_group_width` 経由でのみ本番ビルドから
/// 呼ばれる（`examples/gemm_mma_swizzle_bench.rs` 専用。本ファイル冒頭
/// コメント参照）。feature 既定 off のビルド（`cargo build -p
/// backend-cuda`／`cargo clippy` を feature 指定なしで実行した場合）では
/// 呼び出し元が存在せず dead-code lint が誤検出するため、`kernels_mma.rs::
/// MMA_SHARED_MEM_BYTES` と同じ判断パターンで明示的に許可する（本ファイル
/// 単体テストからは通常どおり呼ばれ、動作自体は機械検証済み）。
#[allow(dead_code)]
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
/// `#[allow(dead_code)]`: 上記 [`GROUP_WIDTH_CANDIDATES`] と同じ理由。
#[allow(dead_code)]
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
/// 呼び出し文脈: `lib.rs::diagnostics::mma_swizzle_group_width`
/// （`internal-diagnostics` feature 経由）が `device.multiprocessor_count()`
/// （`device.rs`）と `kernels_mma::MMA_BM`/`MMA_BN` を渡して呼ぶ。
/// `examples/gemm_mma_swizzle_bench.rs` はこの diagnostics ラッパー経由で
/// 選択結果を取得・表示し、[`CudaMmaGemm::new_with_swizzle`
/// ](crate::CudaMmaGemm::new_with_swizzle) へ明示的に渡す
/// `group_width` を決める（`new_with_swizzle` 自身は本関数を呼ばず、
/// 呼び出し元が渡した値をそのまま使う。本モジュール冒頭コメント参照）。
///
/// `#[allow(dead_code)]`: [`GROUP_WIDTH_CANDIDATES`] と同じ理由
/// （`internal-diagnostics` feature 既定 off のビルドでは呼び出し元
/// `diagnostics::mma_swizzle_group_width` 自体がコンパイルされないため）。
#[allow(dead_code)]
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
}
