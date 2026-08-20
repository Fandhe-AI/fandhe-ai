//! 2 次元タイルジョブ分配（イシュー #753・§3.2）の純関数群。
//!
//! 現行の並列経路（[`super::gemm_blis_parallel`]）は C を行方向へ
//! `m.div_ceil(num_threads)` 行ずつの静的パネルへ分割する
//! （`par_chunks_mut(panel_rows * n)`）。この方式は MC タイル境界と
//! パネル境界が一致しないため、MC タイル数が `num_threads` で割り切れ
//! ない形状では端数タイルが特定 worker へ偏る（#753 実装計画 §3.2）。
//!
//! 本モジュールはこの偏りを是正するための 2 段階の純関数を提供する:
//!
//! 1. [`tile_grid`]: M×N を MC×NC 単位のミニタイルへ区切った 2 次元
//!    ジョブ空間そのもの（イシューが指す「2 次元タイルジョブ分配」の
//!    対象空間）。重複なし・被覆完全であることを `mod tests` で検証する。
//! 2. [`row_ranges_for_workers`]: MC タイル**数**を [`split_evenly`]
//!    （gemm crate `gemm.rs` の n_jobs 分配方式を参照した均等割り）で
//!    worker 数へ分配してから、行範囲（連続区間）へ変換する。
//!
//! ## unsafe を使わない設計判断（PR #766 の教訓の反映）
//!
//! [`tile_grid`] が表す (行タイル, 列タイル) の完全な 2 次元ジョブ空間を
//! worker へ非連続に分配する実装（gemm crate 本来の方式）は、C への
//! 書き込みが行方向にも列方向にも入り組むため、`&mut [f32]` の借用検査を
//! 素朴には満たせず生ポインタ経由の `unsafe` ラッパーが必要になる。
//! PR #766 で「常に不活性な sysctl FFI」が P0/P1 指摘で撤去された経緯を
//! 踏まえ、`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の
//! 必要最小限に留める」方針との整合を優先し、#753 では **列方向は各
//! worker が担当行範囲の全列を内部で処理する（既存の
//! [`super::gemm_blis_ic_loop`]／[`super::gemm_blis_region`] がそのまま
//! 対応する）行方向のみの安全な分配**を採用する（[`row_ranges_for_workers`]）。
//! [`tile_grid`] は「偏りなく被覆する 2 次元ジョブ空間」という設計上の
//! 前提を独立に検証するための純関数として残し、実行時分配（安全な
//! `split_at_mut` 連鎖。[`super::gemm_blis_parallel_2d_with_blocks`]）とは
//! 別に単体テストする。設計判断の詳細は
//! `docs/perf/cpu-gemm-runtime-cache-detect.md` を参照。

use std::ops::Range;

/// M×N のミニタイル格子上の 1 タイル（行範囲・列範囲の組）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tile {
    pub row: Range<usize>,
    pub col: Range<usize>,
}

/// `[0, total)` を `size` 幅の連続区間へ分割する（最終区間のみ端数で
/// 短くなる）。`total == 0` または `size == 0` は空の結果を返す全域関数
/// （境界値であっても panic しない。REQ-8 境界検査の精神を純関数側にも
/// 適用する）。
pub(crate) fn bands(total: usize, size: usize) -> Vec<Range<usize>> {
    if total == 0 || size == 0 {
        return Vec::new();
    }
    (0..total)
        .step_by(size)
        .map(|start| start..(start + size).min(total))
        .collect()
}

/// M×N を `mc`×`nc` 単位のミニタイルへ row-major で区切った一覧
/// （#753 §3.2「ジョブ空間」）。`mod tests` で「重複なし・被覆完全」を
/// 検証する（実行時分配ロジックそのものではなく、分配の前提となる
/// ジョブ空間の定義を独立に固定するための純関数。モジュールドキュメント
/// 「unsafe を使わない設計判断」参照）。
pub(crate) fn tile_grid(m: usize, n: usize, mc: usize, nc: usize) -> Vec<Tile> {
    let mut tiles = Vec::new();
    for row in bands(m, mc) {
        for col in bands(n, nc) {
            tiles.push(Tile {
                row: row.clone(),
                col,
            });
        }
    }
    tiles
}

/// `[0, total)` を `workers` 個の区間へできるだけ均等に分割する（gemm
/// crate `gemm.rs` の n_jobs 分配方式を参照した端数タイル均等化。余り
/// `total % workers` 個の worker が 1 つ多く受け取るため、区間長の差は
/// 常に高々 1）。`workers == 0` または `total == 0` は空の結果を返す
/// 全域関数。
pub(crate) fn split_evenly(total: usize, workers: usize) -> Vec<Range<usize>> {
    if workers == 0 || total == 0 {
        return Vec::new();
    }
    let base = total / workers;
    let extra = total % workers;
    let mut ranges = Vec::with_capacity(workers);
    let mut start = 0;
    for w in 0..workers {
        let len = base + usize::from(w < extra);
        if len == 0 {
            continue;
        }
        ranges.push(start..start + len);
        start += len;
    }
    ranges
}

/// `blocks.mc` 単位の行タイル**数**を [`split_evenly`] で `workers` 個へ
/// 均等分配し、各 worker が担当する C の行範囲（連続・disjoint・`[0, m)`
/// を隙間なく被覆）を返す（#753）。
///
/// 従来の `m.div_ceil(num_threads)` による静的パネル分割はタイル境界を
/// 考慮せず行**数**のみを均等化するため、MC タイル数が `num_threads` で
/// 割り切れない形状ではタイル**数**が特定 worker（境界を跨ぐパネルを
/// 担当する worker）へ偏りうる。本関数はタイル数を先に均等化してから
/// 行範囲へ変換することで、この偏りを ±1 タイルへ抑える（モジュール
/// ドキュメント「unsafe を使わない設計判断」参照。実行側は
/// [`super::gemm_blis_parallel_2d_with_blocks`] が本関数の結果を
/// `split_at_mut` 連鎖で安全に disjoint 分割する）。
pub(crate) fn row_ranges_for_workers(m: usize, mc: usize, workers: usize) -> Vec<Range<usize>> {
    let row_bands = bands(m, mc);
    if row_bands.is_empty() {
        return Vec::new();
    }
    split_evenly(row_bands.len(), workers)
        .into_iter()
        .filter_map(|tile_idx_range| {
            let first = row_bands.get(tile_idx_range.start)?;
            let last = row_bands.get(tile_idx_range.end - 1)?;
            Some(first.start..last.end)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_covers_total_without_overlap_with_remainder_tail() {
        let b = bands(23, 8);
        assert_eq!(b, vec![0..8, 8..16, 16..23]);
    }

    #[test]
    fn bands_returns_empty_for_zero_total_or_zero_size() {
        assert!(bands(0, 8).is_empty());
        assert!(bands(23, 0).is_empty());
    }

    #[test]
    fn bands_exact_multiple_has_no_short_tail() {
        assert_eq!(bands(16, 8), vec![0..8, 8..16]);
    }

    /// [`tile_grid`] が表す 2 次元ジョブ空間の「重複なし・被覆完全」不変
    /// 条件を、境界を跨ぐ非整除な形状で全点走査により検証する（#753 §3.2
    /// のジョブ空間定義そのものの正しさを固定するテスト）。
    #[test]
    fn tile_grid_covers_every_point_exactly_once() {
        let (m, n, mc, nc) = (257usize, 193usize, 64usize, 48usize);
        let tiles = tile_grid(m, n, mc, nc);

        let mut coverage = vec![0u8; m * n];
        for tile in &tiles {
            for i in tile.row.clone() {
                for j in tile.col.clone() {
                    coverage[i * n + j] += 1;
                }
            }
        }
        assert!(
            coverage.iter().all(|&c| c == 1),
            "tile_grid はすべての (i,j) をちょうど 1 回ずつ被覆するはず"
        );
    }

    #[test]
    fn tile_grid_is_empty_for_zero_m_or_n() {
        assert!(tile_grid(0, 100, 64, 48).is_empty());
        assert!(tile_grid(100, 0, 64, 48).is_empty());
    }

    #[test]
    fn split_evenly_covers_total_contiguously_with_balanced_lengths() {
        let ranges = split_evenly(23, 5);
        // 連続・被覆完全（[0, 23) を隙間なく分割）。
        let mut expected_start = 0;
        for r in &ranges {
            assert_eq!(r.start, expected_start);
            expected_start = r.end;
        }
        assert_eq!(expected_start, 23);

        // 区間長の差は高々 1（n_jobs 均等分配の性質）。
        let lens: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        let min_len = *lens.iter().min().unwrap();
        let max_len = *lens.iter().max().unwrap();
        assert!(max_len - min_len <= 1);
    }

    #[test]
    fn split_evenly_returns_empty_for_zero_total_or_zero_workers() {
        assert!(split_evenly(0, 5).is_empty());
        assert!(split_evenly(23, 0).is_empty());
    }

    #[test]
    fn split_evenly_workers_exceeding_total_yields_at_most_total_ranges() {
        // workers > total の場合、余った worker には長さ 0 の区間が
        // 割り当たるはずのものを除外するため、区間数は total を超えない。
        let ranges = split_evenly(3, 10);
        assert_eq!(ranges.len(), 3);
        for r in &ranges {
            assert_eq!(r.end - r.start, 1);
        }
    }

    #[test]
    fn row_ranges_for_workers_covers_m_contiguously_and_disjointly() {
        let (m, mc, workers) = (523usize, 64usize, 5usize);
        let ranges = row_ranges_for_workers(m, mc, workers);

        let mut expected_start = 0;
        for r in &ranges {
            assert_eq!(r.start, expected_start, "行範囲は連続で隙間がないはず");
            assert!(r.end > r.start, "空の行範囲は含まれないはず");
            expected_start = r.end;
        }
        assert_eq!(expected_start, m, "行範囲は m を過不足なく被覆するはず");
    }

    #[test]
    fn row_ranges_for_workers_balances_tile_count_within_one() {
        // MC タイル数（523.div_ceil(64) = 9）が worker 数（4）で割り切れ
        // ない形状。タイル数の割り当て差が高々 1 であることを、各 worker
        // の担当行数から逆算したタイル数で検証する（境界を跨ぐ worker が
        // いないため `(end-start).div_ceil(mc)` で一致するタイル数が
        // 求まる）。
        let (m, mc, workers) = (523usize, 64usize, 4usize);
        let ranges = row_ranges_for_workers(m, mc, workers);
        let tile_counts: Vec<usize> = ranges
            .iter()
            .map(|r| (r.end - r.start).div_ceil(mc))
            .collect();
        let min_tiles = *tile_counts.iter().min().unwrap();
        let max_tiles = *tile_counts.iter().max().unwrap();
        assert!(
            max_tiles - min_tiles <= 1,
            "MC タイル数の worker 間の偏りは高々 1 のはず（実際: {tile_counts:?}）"
        );
    }

    #[test]
    fn row_ranges_for_workers_returns_empty_for_zero_m() {
        assert!(row_ranges_for_workers(0, 64, 4).is_empty());
    }
}
